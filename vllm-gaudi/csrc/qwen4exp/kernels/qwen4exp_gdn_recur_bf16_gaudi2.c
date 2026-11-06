// SPDX-License-Identifier: Apache-2.0
// H14: fused Gated-DeltaNet recurrence, one TPC launch for a whole batch of
// uniform-length sequences (decode: tokens_per_seq == 1; prefill: one span).
//
// Fused per (sequence, value head, token), in F32:
//   g    = -exp(A_log[h]) * softplus(a[t,h] + dt_bias[h])
//   beta = sigmoid(b[t,h])
//   q,k  = L2-normalized q/k of key head h/group (q also scaled by Dk^-1/2)
//   S    = exp(g) * S;  proj = S.k;  delta = beta * (v - proj)
//   S   += delta (x) k;             out   = S.q
// which is exactly the torch decode step of hpu_fused_recurrent_gated_delta_rule
// (and the per-token recurrence the chunk prefill evaluates).
//
// One index point owns ROWS consecutive Dv rows of one (head, sequence):
// it loads those state rows once, streams q/k/v/gating per token, and
// stores the final rows once. The 16 dot products of a tile (8 rows x {S.k, S.q}, from the
// undecayed state) share one lane-interleaving reduction tree:
//   proj_i = decay * (S_i.k)     delta_i = beta * (v_i - proj_i)
//   out_i  = decay * (S_i.q) + delta_i * (k.q)
//   S_i'   = decay * S_i + delta_i * k
//
// Layouts (TPC sees torch dims reversed, dim 0 = torch last dim):
//   mixed      BF16 [C, T]        C = (2*Hk + Hv) * 128, packed [q|k|v]
//   a, b       BF16 [Hv, T]
//   A_log      F32  [Hv];  dt_bias BF16 [Hv]
//   state_pool BF16 [128(Dk), 128(Dv), Hv, Slots]
//   indices    I32  [S]           (already mapped into [0, Slots) by the caller)
//   output     F32  [128(Dv), Hv, T] (caller rounds to BF16)
//   state_out  BF16 [128(Dk), 128(Dv), Hv, S]
// State is BF16 in the cache (the model's state dtype); all arithmetic is
// F32, with one BF16 rounding on the final state store, matching the torch
// path's index_copy_ cast into the BF16 pool. BF16->F32 conversion puts
// even source elements in .v1 and odd ones in .v2 (lane i holds source
// 2i / 2i+1); q/k/state/v stay in this domain throughout -- their
// elementwise products and dot products are invariant under the shuffle,
// and the BF16 state store applies the inverse shuffle. Only the per-row
// v values are extracted from the shuffled domain (memory row r is lane
// r/2 of the matching half).
#include "qwen4exp_gdn_common.h"

#define ROWS 8
// v_f32_reduce_add leaves its sum in lane 0; broadcast it before use as a
// per-vector scalar (same pattern as the in-tree flashqla Q/K prep kernel).
#define BCAST_SUM(x) v_broadcast_element_f32(v_f32_reduce_add(x), 0)

// Lane-interleaving reduction tree, copied from the native fused kernel:
// 16 dot products are reduced together so lane i holds dot product i.
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


void main(tensor mixed, tensor a, tensor b, tensor A_log, tensor dt_bias,
          tensor state_pool, tensor indices, tensor output, tensor state_out) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const uint64 lanes = (uint64)read_lane_id_4b_b();
    const bool256 pred_group_1 = v_i1_mov_flavor_b(0xffffffff, 1, 0, (bool256)0);
    XOR_LUT(lut_x1, 1)
    XOR_LUT(lut_x2, 2)
    XOR_LUT(lut_x4, 4)
    const bool64 keep1 = v_u32_cmp_eq_b(lanes & 1, 0);
    const bool64 keep2 = v_u32_cmp_eq_b(lanes & 2, 0);
    const bool64 keep4 = v_u32_cmp_eq_b(lanes & 4, 0);
    const bool64 keep8 = v_u32_cmp_eq_b(lanes & 8, 0);
    const int channels = get_dim_size(mixed, 0);
    const int tokens = get_dim_size(mixed, 1);
    const int value_heads = get_dim_size(a, 0);
    const int seqs = get_dim_size(indices, 0);
    const int dim = get_dim_size(state_pool, 0);  // 128
    const int key_heads = (channels / dim - value_heads) / 2;
    const int group = value_heads / key_heads;
    const int tokens_per_seq = tokens / seqs;
    float64 dim_vec = dim;
    const float64 q_scale = v_rsqrt_f32(dim_vec);
    const float64 eps = 1e-6f;
    const float64 zero64 = 0;

    for (int seq = start[2]; seq < end[2]; ++seq) {
        const int5 ic = {seq, 0, 0, 0, 0};
        const int slot = s_i32_ld_g((__global__ int*)gen_addr(ic, indices));
        for (int head = start[1]; head < end[1]; ++head) {
            const int key_head = head / group;
            // Load-at-offset + broadcast lane 0 (the native kernel's pattern
            // for per-head scalars): lane 0 of a load at `head` is its element.
            const int5 hc = {head, 0, 0, 0, 0};
            const float64 a_log_vec = v_broadcast_element_f32(v_f32_ld_tnsr_b(hc, A_log), 0);
            const float64 neg_exp_a = -v_exp_f32(a_log_vec);
            const float64 dt_vec = v_broadcast_element_f32(
                v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(hc, dt_bias)).v1, 0);

            for (int block = start[0]; block < end[0]; ++block) {
                const int row_base = block * ROWS;
                float64 state_lo[ROWS];
                float64 state_hi[ROWS];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    const int5 sc = {0, row_base + r, head, slot, 0};
                    const float128 s = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, state_pool));
                    state_lo[r] = s.v1;
                    state_hi[r] = s.v2;
                }

                for (int step = 0; step < tokens_per_seq; ++step) {
                    const int token = seq * tokens_per_seq + step;
                    const int5 ac = {head, token, 0, 0, 0};
                    const float64 a_vec = v_broadcast_element_f32(
                        v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ac, a)).v1, 0);
                    const float64 b_vec = v_broadcast_element_f32(
                        v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ac, b)).v1, 0);
                    // exp(g) as exp(g/2)^2: with the direct form, the exact
                    // combination softplus(a+dt)==ln2 and exp(A_log)==1
                    // produced decay == 1 on this TPC stack (probed in
                    // isolation); splitting the exponent is exact to F32.
                    const float64 g_half =
                        (neg_exp_a * qwen4exp_gdn_softplus(a_vec + dt_vec)) * 0.5f;
                    const float64 decay_half = v_exp_f32(g_half);
                    const float64 decay = decay_half * decay_half;
                    const float64 beta = qwen4exp_gdn_sigmoid(b_vec);

                    const int5 qc = {key_head * dim, token, 0, 0, 0};
                    const int5 kc = {(key_heads + key_head) * dim, token, 0, 0, 0};
                    const float128 q_raw = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(qc, mixed));
                    const float128 k_raw = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(kc, mixed));
                    const float64 q_sq = BCAST_SUM(
                        v_f32_mac_b(q_raw.v2, q_raw.v2, q_raw.v1 * q_raw.v1));
                    const float64 k_sq = BCAST_SUM(
                        v_f32_mac_b(k_raw.v2, k_raw.v2, k_raw.v1 * k_raw.v1));
                    const float64 q_inv = v_rsqrt_f32(q_sq + eps) * q_scale;
                    const float64 k_inv = v_rsqrt_f32(k_sq + eps);
                    const float64 q_lo = q_raw.v1 * q_inv, q_hi = q_raw.v2 * q_inv;
                    const float64 k_lo = k_raw.v1 * k_inv, k_hi = k_raw.v2 * k_inv;
                    const float64 kq = BCAST_SUM(v_f32_mac_b(k_hi, q_hi, k_lo * q_lo));

                    const int5 vc = {(2 * key_heads + head) * dim, token, 0, 0, 0};
                    const float128 v_raw = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(vc, mixed));

                    float64 part[2 * ROWS];
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        part[r] = v_f32_mac_b(state_hi[r], k_hi, state_lo[r] * k_lo);
                        part[ROWS + r] = v_f32_mac_b(state_hi[r], q_hi, state_lo[r] * q_lo);
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
                    // lanes 0..7: S.k per row; lanes 8..15: S.q per row
                    const float64 q_sums = XOR8(sums);
                    float64 v_packed = zero64;
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        float64 v_half;
                        if ((r & 1) == 0) { v_half = v_raw.v1; } else { v_half = v_raw.v2; }
                        const float64 vi = v_broadcast_element_f32(v_half, (row_base + r) / 2);
                        v_packed = v_f32_mov_vb(vi, 0, v_packed,
                                                v_u32_cmp_eq_b(lanes, (uint64)r));
                    }
                    const float64 delta_packed = (v_packed - sums * decay) * beta;
                    const float64 out_packed = v_f32_mac_b(delta_packed, kq, q_sums * decay);
                    const int5 oc = {row_base, head, token, 0, 0};
                    v_f32_st_tnsr_partial(oc, output, out_packed, ROWS - 1, 0);
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        const float64 delta_r = v_broadcast_element_f32(delta_packed, r);
                        state_lo[r] = v_f32_mac_b(k_lo, delta_r, state_lo[r] * decay);
                        state_hi[r] = v_f32_mac_b(k_hi, delta_r, state_hi[r] * decay);
                    }
                }

#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float128 s;
                    s.v1 = state_lo[r];
                    s.v2 = state_hi[r];
                    const int5 sc = {0, row_base + r, head, seq, 0};
                    v_bf16_st_tnsr(sc, state_out, v_convert_f32_to_bf16_all_b(s));
                }
            }
        }
    }
}
