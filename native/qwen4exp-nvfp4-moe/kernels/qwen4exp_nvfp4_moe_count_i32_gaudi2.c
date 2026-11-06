// SPDX-License-Identifier: Apache-2.0
// MoE prefill bucketing, step 1 of 3: how many routed slots of each 64-slot
// block go to each expert.
//
// Port of Strata hpu/tpc/moe_mme_count.c with two deliberate differences:
//   * the shared slot of a row is NOT filtered here. The caller passes -1 for
//     every slot that must not be bucketed (fused shared expert, padding,
//     dropped token), so per_row/top_k disappear from the contract and only
//     one rule remains: ids < 0 are skipped.
//   * counts are I32, not F32, so the count -> prefix -> scatter chain is
//     integer arithmetic end to end and cannot round.
//
// TPC geometry: ids [S] I32 -> counts [blocks, E] I32 (torch order; dim 0 of the
// TPC view is the contiguous expert axis, so the TPC tensor is [E, blocks]).
// Index space: (blocks, E/64) - one member is one block and one group of 64
// experts. The block's ids are loaded once (one 64-lane load) and masked, then
// the 64 experts' compares and reductions run eight at a time so the eight
// independent dependent chains overlap (moe_mme_count.c's unroll).
//
// The 64-lane load is the granularity of the machine, so a tail block loads the
// last 64 slots of the list and masks the lanes that belong to an earlier block
// instead of reading past the tensor. A slot list shorter than 64 reads one
// window from slot 0 (same convention as the shipped indexed GEMV, whose 256-byte
// loads also cover whole vectors).
#include "qwen4exp_nvfp4_moe.h"

void main(tensor ids, tensor counts)
{
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int slots = get_dim_size(ids, 0);
    for (int block = begin[0]; block < end[0]; ++block) {
        const int first = block * QWEN4EXP_MOE_BLOCK_SLOTS;
        const int window = QWEN4EXP_MOE_BLOCK_SLOTS;
        const int base = first + window <= slots
                             ? first
                             : (slots > window ? slots - window : 0);
        const int lo = first - base;                       // first lane of this block
        const int hi = (first + window < slots ? window : slots - base) - 1;
        const int5 ip = {base, 0, 0, 0, 0};
        const int64 lane = (int64)V_LANE_ID_32;
        // Lanes outside [lo, hi] compare against -1, which no expert id equals;
        // ids < 0 (skip markers) stay negative and are skipped as well.
        const int64 raw = v_i32_ld_tnsr_b(ip, ids);
        int64 values = v_i32_sel_geq_i32_b(lane, lo, raw, -1);
        values = v_i32_sel_leq_i32_b(lane, hi, values, -1);
        for (int group = begin[1]; group < end[1]; ++group) {
            const int base = group * QWEN4EXP_MOE_BLOCK_SLOTS;
            for (int e0 = base; e0 < base + QWEN4EXP_MOE_BLOCK_SLOTS; e0 += 8) {
                // Eight experts at a time: the reductions are independent chains.
                const int64 c0 = v_i32_reduce_add(v_i32_sel_eq_i32_b(values, e0 + 0, 1, 0));
                const int64 c1 = v_i32_reduce_add(v_i32_sel_eq_i32_b(values, e0 + 1, 1, 0));
                const int64 c2 = v_i32_reduce_add(v_i32_sel_eq_i32_b(values, e0 + 2, 1, 0));
                const int64 c3 = v_i32_reduce_add(v_i32_sel_eq_i32_b(values, e0 + 3, 1, 0));
                const int64 c4 = v_i32_reduce_add(v_i32_sel_eq_i32_b(values, e0 + 4, 1, 0));
                const int64 c5 = v_i32_reduce_add(v_i32_sel_eq_i32_b(values, e0 + 5, 1, 0));
                const int64 c6 = v_i32_reduce_add(v_i32_sel_eq_i32_b(values, e0 + 6, 1, 0));
                const int64 c7 = v_i32_reduce_add(v_i32_sel_eq_i32_b(values, e0 + 7, 1, 0));
                v_i32_st_tnsr_partial((int5){e0 + 0, block, 0, 0, 0}, counts, c0, 0, 0);
                v_i32_st_tnsr_partial((int5){e0 + 1, block, 0, 0, 0}, counts, c1, 0, 0);
                v_i32_st_tnsr_partial((int5){e0 + 2, block, 0, 0, 0}, counts, c2, 0, 0);
                v_i32_st_tnsr_partial((int5){e0 + 3, block, 0, 0, 0}, counts, c3, 0, 0);
                v_i32_st_tnsr_partial((int5){e0 + 4, block, 0, 0, 0}, counts, c4, 0, 0);
                v_i32_st_tnsr_partial((int5){e0 + 5, block, 0, 0, 0}, counts, c5, 0, 0);
                v_i32_st_tnsr_partial((int5){e0 + 6, block, 0, 0, 0}, counts, c6, 0, 0);
                v_i32_st_tnsr_partial((int5){e0 + 7, block, 0, 0, 0}, counts, c7, 0, 0);
            }
        }
    }
}
