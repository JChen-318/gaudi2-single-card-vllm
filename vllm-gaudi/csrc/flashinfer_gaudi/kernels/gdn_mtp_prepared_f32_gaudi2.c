// SPDX-License-Identifier: Apache-2.0
// Full-query MTP core. Q is normalized/scaled, K normalized, and V FP32.
// The caller resolves the accepted state before this pure, two-output op.
//
// Batched-reduction variant: per token, the 2*ROWS dot products of a tile
// (state rows with k, and with q, both on the undecayed state) share one
// lane-interleaving reduction tree instead of one full 64-lane reduction
// each. After the tree, lane i (mod 2*ROWS) holds dot product i, so the
// per-row delta and output are computed once for the whole tile:
//   proj = g * (S . k),  delta = beta * (v - proj),
//   S'   = g * S + delta * k,  out = S' . q = g * (S . q) + delta * (k . q).
#ifndef MTP_PREPARED_ROWS
#define MTP_PREPARED_ROWS 8
#endif

// Shuffle pattern: lane l reads lane l ^ m within its 8-lane dual group.
#define XOR_LUT(name, m)                                                                                       \
    uint64 name##_bytes = 0x80808080;                                                                          \
    name##_bytes += v_u32_xor_b(read_lane_id_4b_b() * 0x01010101, (m) * 0x01010101);                          \
    name##_bytes = v_u32_mov_group_b(name##_bytes, 0xFFFFFFFF, (SW_GROUP1_EN | SW_DUAL_GROUP0_EN), name##_bytes, \
                                     1, 0);                                                                    \
    name##_bytes = v_u32_add_vb(name##_bytes, 0x20202020, 0, name##_bytes, to_bool64(pred_group_1));           \
    name##_bytes = v_u32_mov_dual_group_all_b(name##_bytes, 0xFFFFFFFF, 0, 0, 0, 0,                            \
                                              MkWrA(0b11, 0b11, 0b11, 0b11), 0);                               \
    const uchar256 name = (uchar256)name##_bytes;

// Lane partners l ^ 8, l ^ 16, l ^ 32.
#define XOR8(v) v_f32_mov_group_b((v), 0xFFFFFFFF, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |           \
                                  SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, (v), 1, 0)
#define XOR16(v) v_f32_mov_dual_group_all_b((v), 0xFFFFFFFF, 1, 0, 3, 2, MkWrA(0b11, 0b11, 0b11, 0b11), (v))
#define XOR32(v) v_f32_mov_dual_group_all_b((v), 0xFFFFFFFF, 2, 3, 0, 1, MkWrA(0b11, 0b11, 0b11, 0b11), (v))

// Lanes whose bit is clear keep a's pair sum, the others b's.
#define MERGE(a, b, keep_a, partner) \
    (v_f32_mov_vb((a), 0, (b), (keep_a)) + partner(v_f32_mov_vb((b), 0, (a), (keep_a))))
#define P1(v) v_f32_shuffle_b((v), lut_x1, 0, (v))
#define P2(v) v_f32_shuffle_b((v), lut_x2, 0, (v))
#define P4(v) v_f32_shuffle_b((v), lut_x4, 0, (v))

void main(tensor initial_state, tensor prepared_qkv, tensor decay, tensor beta,
          tensor output, tensor checkpoints)
{
    const int rows = MTP_PREPARED_ROWS;
    const uint64 lanes = read_lane_id_4b_b();
    const bool256 pred_group_1 = v_i1_mov_flavor_b(0xffffffff, 1, 0, (bool256)0);
    XOR_LUT(lut_x1, 1)
    XOR_LUT(lut_x2, 2)
    XOR_LUT(lut_x4, 4)
    const bool64 keep1 = v_u32_cmp_eq_b(lanes & 1, 0);
    const bool64 keep2 = v_u32_cmp_eq_b(lanes & 2, 0);
    const bool64 keep4 = v_u32_cmp_eq_b(lanes & 4, 0);
#if MTP_PREPARED_ROWS == 8
    const bool64 keep8 = v_u32_cmp_eq_b(lanes & 8, 0);
#endif
    const int5 begin = get_index_space_offset();
    const int5 end = get_index_space_size() + begin;

    for (int batch = begin[3]; batch < end[3]; ++batch) {
        for (int head = begin[2]; head < end[2]; ++head) {
            const int key_head = head / 3;
            for (int tile = begin[1]; tile < end[1]; ++tile) {
                float64 state_lo[MTP_PREPARED_ROWS];
                float64 state_hi[MTP_PREPARED_ROWS];
                #pragma unroll
                for (int row = 0; row < rows; ++row) {
                    int5 coords = {0, tile * rows + row, head, batch, 0};
                    state_lo[row] = v_f32_ld_tnsr_b(coords, initial_state);
                    coords[0] = 64;
                    state_hi[row] = v_f32_ld_tnsr_b(coords, initial_state);
                }
                for (int token = 0; token < 8; ++token) {
                    int5 packed_coords = {key_head * 128, token, batch, 0, 0};
                    const float64 q_lo = v_f32_ld_tnsr_b(packed_coords, prepared_qkv);
                    packed_coords[0] += 64;
                    const float64 q_hi = v_f32_ld_tnsr_b(packed_coords, prepared_qkv);
                    packed_coords[0] = 2048 + key_head * 128;
                    const float64 k_lo = v_f32_ld_tnsr_b(packed_coords, prepared_qkv);
                    packed_coords[0] += 64;
                    const float64 k_hi = v_f32_ld_tnsr_b(packed_coords, prepared_qkv);
                    int5 scalar_coords = {head, token, batch, 0, 0};
                    const float64 decay_value = v_broadcast_element_f32(v_f32_ld_tnsr_b(scalar_coords, decay), 0);
                    const float64 beta_value = v_broadcast_element_f32(v_f32_ld_tnsr_b(scalar_coords, beta), 0);
                    packed_coords[0] = 4096 + head * 128 + tile * rows;
                    const float64 values = v_f32_ld_tnsr_b(packed_coords, prepared_qkv);

                    float64 kq = v_f32_mac_b(k_hi, q_hi, k_lo * q_lo);
                    kq = v_f32_reduce_add(kq);

                    // Dot-product partials on the undecayed state: k first, then q.
                    float64 part[2 * MTP_PREPARED_ROWS];
                    #pragma unroll
                    for (int row = 0; row < rows; ++row) {
                        part[row] = v_f32_mac_b(state_hi[row], k_hi, state_lo[row] * k_lo);
                        part[rows + row] = v_f32_mac_b(state_hi[row], q_hi, state_lo[row] * q_lo);
                    }
                    // Interleaving tree: after the level on lane bit j, lane bit j selects the operand.
                    float64 level1[MTP_PREPARED_ROWS];
                    #pragma unroll
                    for (int i = 0; i < rows; ++i)
                        level1[i] = MERGE(part[2 * i], part[2 * i + 1], keep1, P1);
                    float64 level2[MTP_PREPARED_ROWS / 2];
                    #pragma unroll
                    for (int i = 0; i < rows / 2; ++i)
                        level2[i] = MERGE(level1[2 * i], level1[2 * i + 1], keep2, P2);
#if MTP_PREPARED_ROWS == 8
                    float64 level3[2];
                    level3[0] = MERGE(level2[0], level2[1], keep4, P4);
                    level3[1] = MERGE(level2[2], level2[3], keep4, P4);
                    float64 sums = MERGE(level3[0], level3[1], keep8, XOR8);
#else
                    float64 sums = MERGE(level2[0], level2[1], keep4, P4);
                    sums += XOR8(sums);
#endif
                    sums += XOR16(sums);
                    sums += XOR32(sums);
                    // Lanes 0..rows-1: S.k per row; lanes rows..2*rows-1: S.q per row.
#if MTP_PREPARED_ROWS == 8
                    const float64 q_sums = XOR8(sums);
#else
                    const float64 q_sums = P4(sums);
#endif
                    const float64 delta = (values - sums * decay_value) * beta_value;
                    const float64 packed_output = v_f32_mac_b(delta, kq, q_sums * decay_value);

                    #pragma unroll
                    for (int row = 0; row < rows; ++row) {
                        const float64 delta_row = v_broadcast_element_f32(delta, row);
                        state_lo[row] = v_f32_mac_b(k_lo, delta_row, state_lo[row] * decay_value);
                        state_hi[row] = v_f32_mac_b(k_hi, delta_row, state_hi[row] * decay_value);
                        int5 coords = {0, tile * rows + row, head, token, batch};
                        v_f32_st_tnsr(coords, checkpoints, state_lo[row]);
                        coords[0] = 64;
                        v_f32_st_tnsr(coords, checkpoints, state_hi[row]);
                    }
                    int5 out_coords = {tile * rows, head, token, batch, 0};
                    v_f32_st_tnsr_partial(out_coords, output, packed_output, rows - 1, 0);
                }
            }
        }
    }
}
