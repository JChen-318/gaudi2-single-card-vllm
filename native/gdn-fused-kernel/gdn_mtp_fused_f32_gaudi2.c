// SPDX-License-Identifier: Apache-2.0
// Fused full-query MTP core (B<=16, T=8, H=16, HV=48, K=V=128).
//
// Inputs come straight from the model: the recurrent state pool, the
// per-token checkpoint slots and accepted counts, the raw BF16 conv output
// [q | k | v], the FP32 log decay and the BF16 beta. Outputs match the
// prepared core: FP32 attention output and all eight FP32 checkpoints.
//
// One index-space point is one (value head, request). Per token, Q/K
// conversion, L2 norms, k.q, exp(log decay) and beta are computed once and
// kept in local memory; the 128 state rows are then processed in tiles of
// ROWS rows, carried in registers across the eight tokens. Normalization is
// folded into scalars (rk = 1/|k|, rq = scale/|q|):
//   proj  = g rk (S.k)                     delta = beta (v - proj)
//   S'    = g S + (delta rk) k             out   = g rq S.q + rq k.q (delta rk)
// and the 2*ROWS dot products of a tile share one lane-interleaving
// reduction tree (lane i of the result holds dot product i).
#define ROWS 8
#define TOKENS 8

#define XOR_LUT(name, m)                                                                                       \
    uint64 name##_bytes = 0x80808080;                                                                          \
    name##_bytes += v_u32_xor_b(read_lane_id_4b_b() * 0x01010101, (m) * 0x01010101);                          \
    name##_bytes = v_u32_mov_group_b(name##_bytes, 0xFFFFFFFF, (SW_GROUP1_EN | SW_DUAL_GROUP0_EN), name##_bytes, \
                                     1, 0);                                                                    \
    name##_bytes = v_u32_add_vb(name##_bytes, 0x20202020, 0, name##_bytes, to_bool64(pred_group_1));           \
    name##_bytes = v_u32_mov_dual_group_all_b(name##_bytes, 0xFFFFFFFF, 0, 0, 0, 0,                            \
                                              MkWrA(0b11, 0b11, 0b11, 0b11), 0);                               \
    const uchar256 name = (uchar256)name##_bytes;

#define XOR8(v) v_f32_mov_group_b((v), 0xFFFFFFFF, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |           \
                                  SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, (v), 1, 0)
#define XOR16(v) v_f32_mov_dual_group_all_b((v), 0xFFFFFFFF, 1, 0, 3, 2, MkWrA(0b11, 0b11, 0b11, 0b11), (v))
#define XOR32(v) v_f32_mov_dual_group_all_b((v), 0xFFFFFFFF, 2, 3, 0, 1, MkWrA(0b11, 0b11, 0b11, 0b11), (v))
#define MERGE(a, b, keep_a, partner) \
    (v_f32_mov_vb((a), 0, (b), (keep_a)) + partner(v_f32_mov_vb((b), 0, (a), (keep_a))))
#define P1(v) v_f32_shuffle_b((v), lut_x1, 0, (v))
#define P2(v) v_f32_shuffle_b((v), lut_x2, 0, (v))
#define P4(v) v_f32_shuffle_b((v), lut_x4, 0, (v))

// BF16 [128] -> FP32 lanes 0..63 (v1) and 64..127 (v2), element order kept.
float64_pair_t bf16_to_f32_linear(bfloat128 input)
{
    bfloat128_pair_t unpacked;
    unpacked.v1 = v_bf16_unpack_b(
        input, ((e_group_0) << 8) | ((e_every_second_element) << 9) | ((e_lower_half_group) << 10), unpacked.v1);
    unpacked.v2 = v_bf16_unpack_b(
        input, ((e_group_1) << 8) | ((e_every_second_element) << 9) | ((e_lower_half_group) << 10), unpacked.v2);
    const bfloat128 first_groups = unpacked.v1;
    unpacked.v1 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 0, 1, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 1, 2, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 1, 3, MkWr(1, 1), unpacked.v1);
    unpacked.v2 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 2, 0, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 2, 1, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 3, 2, MkWr(1, 1), unpacked.v2);
    const float64_pair_t first = v_convert_bf16_to_f32_all_b(unpacked.v1);
    const float64_pair_t second = v_convert_bf16_to_f32_all_b(unpacked.v2);
    float64_pair_t result;
    result.v1 = first.v1;
    result.v2 = second.v1;
    return result;
}

// Per token: q_lo, q_hi, k_lo, k_hi; coefficient lanes 0..4: g, beta rk, ck = g rk, cq = g rq, cd = rq k.q.
__local__ float64 token_qk[TOKENS * 4];
__local__ float64 token_coef[TOKENS];

void main(tensor state_pool, tensor packed_qkv, tensor log_decay, tensor beta, tensor state_indices,
          tensor num_accepted, tensor output, tensor checkpoints)
{
    const uint64 lanes = read_lane_id_4b_b();
    const bool256 pred_group_1 = v_i1_mov_flavor_b(0xffffffff, 1, 0, (bool256)0);
    XOR_LUT(lut_x1, 1)
    XOR_LUT(lut_x2, 2)
    XOR_LUT(lut_x4, 4)
    const bool64 keep1 = v_u32_cmp_eq_b(lanes & 1, 0);
    const bool64 keep2 = v_u32_cmp_eq_b(lanes & 2, 0);
    const bool64 keep4 = v_u32_cmp_eq_b(lanes & 4, 0);
    const bool64 keep8 = v_u32_cmp_eq_b(lanes & 8, 0);
    const float q_scale = 0.08838834764831845f;  // 128 ** -0.5
    const int5 begin = get_index_space_offset();
    const int5 end = get_index_space_size() + begin;

    for (int batch = begin[3]; batch < end[3]; ++batch) {
        int5 batch_coords = {batch, 0, 0, 0, 0};
        int accepted = s_i32_ld_g((__global__ int*)gen_addr(batch_coords, num_accepted));
        accepted = accepted < 1 ? 1 : accepted;
        accepted = accepted > TOKENS ? TOKENS : accepted;
        int5 slot_coords = {accepted - 1, batch, 0, 0, 0};
        const int slot = s_i32_ld_g((__global__ int*)gen_addr(slot_coords, state_indices));

        for (int head = begin[2]; head < end[2]; ++head) {
            const int key_head = head / 3;
            // ---- per-token prologue, shared by all tiles of this head ----
            for (int token = 0; token < TOKENS; ++token) {
                int5 packed_coords = {key_head * 128, token, batch, 0, 0};
                const float64_pair_t q = bf16_to_f32_linear(v_bf16_ld_tnsr_b(packed_coords, packed_qkv));
                packed_coords[0] = 2048 + key_head * 128;
                const float64_pair_t k = bf16_to_f32_linear(v_bf16_ld_tnsr_b(packed_coords, packed_qkv));
                const float64 qq = v_f32_mac_b(q.v2, q.v2, q.v1 * q.v1);
                const float64 kk = v_f32_mac_b(k.v2, k.v2, k.v1 * k.v1);
                const float64 kq = v_f32_mac_b(k.v2, q.v2, k.v1 * q.v1);
                // lanes (mod 4): 0 q.q, 1 k.k, 2 and 3 k.q
                float64 norms = MERGE(MERGE(qq, kk, keep1, P1), kq + P1(kq), keep2, P2);
                norms += P4(norms);
                norms += XOR8(norms);
                norms += XOR16(norms);
                norms += XOR32(norms);
                const float64 rq = v_rsqrt_f32(v_broadcast_element_f32(norms, 0) + 1e-6f) * q_scale;
                const float64 rk = v_rsqrt_f32(v_broadcast_element_f32(norms, 1) + 1e-6f);
                const float64 k_dot_q = v_broadcast_element_f32(norms, 2);
                int5 scalar_coords = {head, token, batch, 0, 0};
                const float64 g = v_exp_f32(v_broadcast_element_f32(v_f32_ld_tnsr_b(scalar_coords, log_decay), 0));
                const float beta_value = (float)s_bf16_ld_g((__global__ bf16*)gen_addr(scalar_coords, beta));
                token_qk[token * 4 + 0] = q.v1;
                token_qk[token * 4 + 1] = q.v2;
                token_qk[token * 4 + 2] = k.v1;
                token_qk[token * 4 + 3] = k.v2;
                float64 coef = g;
                coef = v_f32_mov_vb(rk * beta_value, 0, coef, v_u32_cmp_eq_b(lanes, (uint64)1));
                coef = v_f32_mov_vb(g * rk, 0, coef, v_u32_cmp_eq_b(lanes, (uint64)2));
                coef = v_f32_mov_vb(g * rq, 0, coef, v_u32_cmp_eq_b(lanes, (uint64)3));
                coef = v_f32_mov_vb(rq * k_dot_q, 0, coef, v_u32_cmp_eq_b(lanes, (uint64)4));
                token_coef[token] = coef;
            }

            for (int tile = 0; tile < 128 / ROWS; ++tile) {
                float64 state_lo[ROWS];
                float64 state_hi[ROWS];
                #pragma unroll
                for (int row = 0; row < ROWS; ++row) {
                    int5 coords = {0, tile * ROWS + row, head, slot, 0};
                    state_lo[row] = v_f32_ld_tnsr_b(coords, state_pool);
                    coords[0] = 64;
                    state_hi[row] = v_f32_ld_tnsr_b(coords, state_pool);
                }
                for (int token = 0; token < TOKENS; ++token) {
                    const float64 q_lo = token_qk[token * 4 + 0];
                    const float64 q_hi = token_qk[token * 4 + 1];
                    const float64 k_lo = token_qk[token * 4 + 2];
                    const float64 k_hi = token_qk[token * 4 + 3];
                    const float64 coef = token_coef[token];
                    const float64 g = v_broadcast_element_f32(coef, 0);
                    const float64 beta_rk = v_broadcast_element_f32(coef, 1);
                    const float64 ck = v_broadcast_element_f32(coef, 2);
                    const float64 cq = v_broadcast_element_f32(coef, 3);
                    const float64 cd = v_broadcast_element_f32(coef, 4);
                    int5 value_coords = {4096 + head * 128 + tile * ROWS, token, batch, 0, 0};
                    const float64 values = bf16_to_f32_linear(v_bf16_ld_tnsr_b(value_coords, packed_qkv)).v1;

                    float64 part[2 * ROWS];
                    #pragma unroll
                    for (int row = 0; row < ROWS; ++row) {
                        part[row] = v_f32_mac_b(state_hi[row], k_hi, state_lo[row] * k_lo);
                        part[ROWS + row] = v_f32_mac_b(state_hi[row], q_hi, state_lo[row] * q_lo);
                    }
                    float64 level1[ROWS];
                    #pragma unroll
                    for (int i = 0; i < ROWS; ++i)
                        level1[i] = MERGE(part[2 * i], part[2 * i + 1], keep1, P1);
                    float64 level2[ROWS / 2];
                    #pragma unroll
                    for (int i = 0; i < ROWS / 2; ++i)
                        level2[i] = MERGE(level1[2 * i], level1[2 * i + 1], keep2, P2);
                    const float64 level3_0 = MERGE(level2[0], level2[1], keep4, P4);
                    const float64 level3_1 = MERGE(level2[2], level2[3], keep4, P4);
                    float64 sums = MERGE(level3_0, level3_1, keep8, XOR8);
                    sums += XOR16(sums);
                    sums += XOR32(sums);
                    // lanes 0..7: S.k per row; lanes 8..15: S.q per row (raw q, k)
                    const float64 q_sums = XOR8(sums);
                    // scaled_delta = delta rk; out = cq S.q + rq k.q (delta rk)
                    const float64 scaled_delta = (values - sums * ck) * beta_rk;
                    const float64 packed_output = v_f32_mac_b(scaled_delta, cd, q_sums * cq);

                    #pragma unroll
                    for (int row = 0; row < ROWS; ++row) {
                        const float64 delta_row = v_broadcast_element_f32(scaled_delta, row);
                        state_lo[row] = v_f32_mac_b(k_lo, delta_row, state_lo[row] * g);
                        state_hi[row] = v_f32_mac_b(k_hi, delta_row, state_hi[row] * g);
                        int5 coords = {0, tile * ROWS + row, head, token, batch};
                        v_f32_st_tnsr(coords, checkpoints, state_lo[row]);
                        coords[0] = 64;
                        v_f32_st_tnsr(coords, checkpoints, state_hi[row]);
                    }
                    int5 out_coords = {tile * ROWS, head, token, batch, 0};
                    v_f32_st_tnsr_partial(out_coords, output, packed_output, ROWS - 1, 0);
                }
            }
        }
    }
}
