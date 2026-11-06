// SPDX-License-Identifier: Apache-2.0
// MoE prefill bucketing, step 3 of 3: every routed slot's place in its expert's
// bucket row, and the inverse map.
//
// Port of Strata hpu/tpc/moe_mme_scatter.c, with a fixed per-expert row stride
// instead of a cross-expert compact prefix: slot s routed to expert e becomes
// bucket row e*cap + offsets[e][block] + (earlier slots of this block routed to
// e). The arithmetic is exact integer arithmetic: offsets is the exclusive
// per-expert prefix over blocks (step 2), and "earlier" is a small open-addressed
// table over the block's at most 64 distinct experts (step 3's own technique).
// Because offsets[e][block] jumps by count[e][block] for the next block, rows of
// different blocks never collide; the bucket is [E, cap] and only the first
// totals[e] rows of expert e are ever read, so the stride may be any cap >= S.
//
// ids < 0 (skip markers, padding) produce place = -1 and no bucket write; ids
// outside [0, experts) are dropped the same way instead of corrupting the heap.
//
// TPC geometry:
//   ids     [S]       I32   (negative = skip)
//   offsets [E, blocks] I32 (step 2)
//   bucket  [E*cap]   I32   out
//   place   [S]       I32   out
// Index space: (blocks). One member is one 64-slot block. The scalar global
// loads (one id and one offset per slot) are the cost of this step - a scalar
// global load waits a full memory round trip - so they are issued in batches of
// eight like Strata's version, and a full block's places go out as one vector
// store.
#include "qwen4exp_nvfp4_moe.h"

__local__ int placed[QWEN4EXP_MOE_BLOCK_SLOTS];
__local__ int keys[QWEN4EXP_MOE_BLOCK_SLOTS];
__local__ int earlier[QWEN4EXP_MOE_BLOCK_SLOTS];

void main(tensor ids, tensor offsets, tensor bucket, tensor place,
          int experts, int slots, int cap)
{
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    for (int block = begin[0]; block < end[0]; ++block) {
        const int first = block * QWEN4EXP_MOE_BLOCK_SLOTS;
        const int last = first + QWEN4EXP_MOE_BLOCK_SLOTS < slots
                             ? first + QWEN4EXP_MOE_BLOCK_SLOTS : slots;
        for (int i = 0; i < QWEN4EXP_MOE_BLOCK_SLOTS; ++i) keys[i] = -1;
        for (int b0 = first; b0 < last; b0 += 8) {
            int id[8];
            int base[8];
            // Batch of eight: all eight id loads issue before any of them is used.
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int at = b0 + i < last ? b0 + i : last - 1;
                const int5 ip = {at, 0, 0, 0, 0};
                id[i] = b0 + i >= last ? -1 : s_i32_ld_g(gen_addr(ip, ids));
            }
            // Then the eight per-block bases: offsets[id][block], same batching.
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int e = id[i] < 0 ? 0 : id[i];
                const int5 op = {e, block, 0, 0, 0};
                base[i] = id[i] < 0 ? 0 : s_i32_ld_g(gen_addr(op, offsets));
            }
            for (int i = 0; i < 8 && b0 + i < last; ++i) {
                const int slot = b0 + i;
                int at = -1;
                if (id[i] >= 0 && id[i] < experts) {
                    // Open addressing over the block's experts; at most 64 distinct ids.
                    int h = id[i] & (QWEN4EXP_MOE_BLOCK_SLOTS - 1);
                    while (keys[h] != id[i] && keys[h] >= 0)
                        h = (h + 1) & (QWEN4EXP_MOE_BLOCK_SLOTS - 1);
                    if (keys[h] < 0) {
                        keys[h] = id[i];
                        earlier[h] = 0;
                    }
                    const int position = base[i] + earlier[h]++;
                    at = id[i] * cap + position;
                    s_i32_st_g((gen_addr((int5){at, 0, 0, 0, 0}, bucket)), slot);
                }
                placed[slot - first] = at;
            }
        }
        if (last - first == QWEN4EXP_MOE_BLOCK_SLOTS) {
            // Full block: the 64 places leave as one vector store, lane j taking placed[j].
            const int64 lane = (int64)V_LANE_ID_32;
            int64 row = -1;
            for (int j = 0; j < QWEN4EXP_MOE_BLOCK_SLOTS; ++j)
                row = v_i32_sel_eq_i32_b(lane, j, placed[j], row);
            v_i32_st_tnsr((int5){first, 0, 0, 0, 0}, place, row);
        } else {
            // Partial tail block: store the real slots only.
            for (int slot = first; slot < last; ++slot)
                s_i32_st_g((gen_addr((int5){slot, 0, 0, 0, 0}, place)), placed[slot - first]);
        }
    }
}
