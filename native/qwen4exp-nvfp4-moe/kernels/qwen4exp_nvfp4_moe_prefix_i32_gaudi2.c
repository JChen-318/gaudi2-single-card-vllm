// SPDX-License-Identifier: Apache-2.0
// MoE prefill bucketing, step 2 of 3: per expert the exclusive running count
// over blocks (where that block's slots start in the expert's row list) and the
// expert's total.
//
// Port of Strata hpu/tpc/moe_mme_prefix.c, I32 instead of F32 so the prefix is
// exact integer arithmetic. Members are groups of QWEN4EXP_MOE_BLOCK_SLOTS
// experts; one 64-lane vector holds the same block of 64 experts while the
// loop walks the blocks, so a block costs one load, one store and one add.
//
// TPC geometry: counts [E, blocks] I32 -> offsets [E, blocks] I32,
// totals [E] I32. Index space: (E/64).
#include "qwen4exp_nvfp4_moe.h"

void main(tensor counts, tensor offsets, tensor totals)
{
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int blocks = get_dim_size(counts, 1);
    const int64 zero = {0};
    for (int group = begin[0]; group < end[0]; ++group) {
        int64 running = zero;
        for (int block = 0; block < blocks; ++block) {
            const int5 p = {group * QWEN4EXP_MOE_BLOCK_SLOTS, block, 0, 0, 0};
            const int64 count = v_i32_ld_tnsr_b(p, counts);
            v_i32_st_tnsr(p, offsets, running);
            running = v_i32_add_b(running, count);
        }
        v_i32_st_tnsr((int5){group * QWEN4EXP_MOE_BLOCK_SLOTS, 0, 0, 0, 0}, totals, running);
    }
}
