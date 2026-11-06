// SPDX-License-Identifier: Apache-2.0
// H15 Optimization 3: fused router post-GEMM (top-k + renormalized softmax).
//
//   logits [T, E=512] BF16 --(this kernel)--> ids [T, K] I32, weights [T, K] BF16
//   weights = softmax(topk(logits).values.float(), dim=-1).to(bf16)
//
// Replaces torch.topk + .float() + softmax + .to(bf16) + .to(int32) (four-plus
// dispatched ops on a [T,512] / [T,10] plane, ~0.2 ms synchronized at T=1 in
// the H15 micro-benchmark, almost all launch/dispatch) with one TPC launch.
// The GEMM itself stays on MME (F.linear).
//
// Selection is an exact iterative argmax over the eight 64-lane F32 chunks
// (the deepseek_v41_router_logits_top6 selection pattern): each rank takes
// the remaining maximum, ties go to the lowest expert index (the winner is
// the minimum index among lanes equal to the max). The selected *value*
// sequence is therefore bit-identical to torch.topk's, and ids match
// torch.topk wherever the top-k boundary values are distinct. On exact
// BF16 ties torch.topk's HPU kernel returns a partition-dependent order
// (probed: (59,452)->[452,59] but (45,89)->[45,89], deterministic yet not
// an index order), which no independent selection can reproduce; this
// kernel's deterministic lowest-index rule is the documented difference,
// and bench_moe_h15.py verifies every differing id is an exact logit tie.
// Softmax is evaluated in F32 on the BF16 logits (identical values to
// values.float()): exp(v - max) / sum, then one BF16 rounding.
// exp matching Gaudi's stock F32 kernels (the exp_cephes form reproduced in
// csrc/deepseek_v4/kernels/deepseek_v41_router_logits_top6_gaudi2.c). The
// hardware v_exp_f32 approximation is off by enough ULPs to flip most BF16
// softmax weights; torch's HPU softmax is bit-identical to an exact CPU
// softmax at BF16, and this form reproduces that.
static inline float64 moe_exp_cephes(float64 input) {
    const float ln_2_1 = 0.693359375f;
    const float ln_2_2 = -2.12194440e-4f;
    const float log2_e = 1.44269504088896341f;
    const float64 c6 = .49998858346161135f;
    const float64 c5 = 0.1666634537038239f;
    const float64 c4 = 4.191878872870153e-2f;
    const float64 c3 = 8.380148765026943e-3f;
    float64 z = v_f32_mac_b(input, log2_e, 0.5f, 0);
    z = v_f32_nearbyint_b(z, SW_RD);
    float64 reduced = v_f32_mac_b(z, ln_2_1, input, SW_NEG);
    reduced = v_f32_mac_b(z, ln_2_2, reduced, SW_NEG);
    float64 result = v_f32_mac_b(reduced, c3, c4, 0);
    result = v_f32_mac_b(result, reduced, c5, 0);
    result = v_f32_mac_b(result, reduced, c6, 0);
    result = v_f32_mac_b(result, reduced, 1.0f, 0);
    result = v_f32_mac_b(result, reduced, 1.0f, 0);
    int64 bits = *((int64*)&result);
    int64 exponent = v_convert_f32_to_i32_b(z, SW_RD);
    exponent = v_i32_shl_b(exponent, 23);
    bits = v_i32_add_b(bits, exponent);
    result = *((float64*)&bits);
    const float64 lower = -87.336f;
    const float64 upper = 88.722f;
    const int64 plus_inf_bits = 0x7f800000;
    result = v_f32_sel_leq_f32_b(input, lower, 0.0f, result);
    result = v_f32_sel_geq_f32_b(input, upper, *((float64*)&plus_inf_bits), result);
    result = v_f32_sel_grt_u32_b(*((uint64*)&input) & 0x7fffffff, 0x7f800000, input, result);
    return result;
}

void main(tensor logits, tensor output_ids, tensor output_weights)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int experts = get_dim_size(logits, 0);
    const int topk = get_dim_size(output_ids, 0);
    const int64 lanes = (int64)V_LANE_ID_32;
    for (int token = start[0]; token < end[0]; ++token) {
        // 512 BF16 logits = four 128-lane loads = eight 64-lane F32 chunks.
        // BF16->F32 conversion de-interleaves: v1 lane j is element 2*j of
        // the 128-lane load and v2 lane j is element 2*j+1 (probed with
        // one-hot logits), so chunk 2*c covers experts c*128 + 2*lane and
        // chunk 2*c+1 covers c*128 + 2*lane + 1.
        float64 scores[8];
        int64 indices[8];
        bool64 eligible[8];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float128 v = v_convert_bf16_to_f32_all_b(
                v_bf16_ld_tnsr_b((int5){c * 128, token, 0, 0, 0}, logits));
            scores[2 * c] = v.v1;
            scores[2 * c + 1] = v.v2;
            indices[2 * c] = lanes + lanes + c * 128;
            indices[2 * c + 1] = lanes + lanes + 1 + c * 128;
            eligible[2 * c] = (bool64)1;
            eligible[2 * c + 1] = (bool64)1;
        }
        (void)experts;
        float64 packed_weights = 0;
        int64 packed_ids = 0;
        for (int rank = 0; rank < topk; ++rank) {
            int64_float64_pair_t local;
            local.v1 = 2147483647;
            local.v2 = as_float64((int64)0xff800000);
            float64 local_score = 0;
#pragma unroll
            for (int c = 7; c >= 0; --c) {
                local_score = v_f32_sel_geq_f32_vb(
                    scores[c], local.v2, scores[c], local_score,
                    0, local_score, eligible[c], 0);
                local = v_i32_sel2_geq_f32_vb(
                    scores[c], local.v2, indices[c], local.v1,
                    0, local, eligible[c], 0);
            }
            const float64 best = v_f32_reduce_max(local.v2);
            int64 winner = v_i32_sel_eq_f32_b(local.v2, best, local.v1, 2147483647);
            winner = v_i32_reduce_min(winner);
            const bool64 selected_lane = v_i32_cmp_eq_b(local.v1, winner);
            const float64 selected = v_f32_mov_vb(local_score, 0, 0, selected_lane, 0);
#pragma unroll
            for (int c = 0; c < 8; ++c)
                eligible[c] = eligible[c] & v_i32_cmp_neq_b(indices[c], winner);
            const float64 selected_weight = v_f32_reduce_add(selected);
            const bool64 output_lane = v_i32_cmp_eq_b(lanes, rank);
            packed_weights = v_f32_mov_vb(selected_weight, 0, packed_weights, output_lane, 0);
            packed_ids = v_i32_mov_vb(winner, 0, packed_ids, output_lane, 0);
        }
        const float64 maxv = v_broadcast_element_f32(packed_weights, 0);
        const float64 expv = moe_exp_cephes(packed_weights - maxv);
        const bool64 in_topk = v_i32_cmp_less_b(lanes, topk);
        const float64 exp_masked = v_f32_mov_vb(expv, 0, (float64)0, in_topk, 0);
        const float64 denom = v_broadcast_element_f32(v_f32_reduce_add(exp_masked), 0);
        float64 inv = v_reciprocal_f32(denom);
        inv = inv * (2.0f - denom * inv);
        inv = inv * (2.0f - denom * inv);
        const float64 weights = exp_masked * inv;
        const int5 at = {0, token, 0, 0, 0};
        v_i32_st_tnsr_partial(at, output_ids, packed_ids, topk - 1, 0);
        // SW_LINEAR: lanes 0..topk-1 of v1 become output elements 0..topk-1
        // (the plain conversion interleaves v1/v2 as even/odd elements).
        float128 wide;
        wide.v1 = weights;
        wide.v2 = 0;
        v_bf16_st_tnsr_partial(at, output_weights,
                               v_convert_f32_to_bf16_all_b(wide, SW_LINEAR), topk - 1, 0);
    }
}
