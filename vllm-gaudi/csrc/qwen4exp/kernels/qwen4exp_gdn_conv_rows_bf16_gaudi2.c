// SPDX-License-Identifier: Apache-2.0
// H14: fused prefill causal conv1d + final conv-state + SiLU, one launch.
//
// Adapted from Strata's gdn_conv_final.c (hpu/tpc): each index point owns a
// block of 128 channels and ROW_BLOCK consecutive tokens of one sequence, and
// rebuilds its 3-tap window from the rows before its first token (the carried
// conv state for rows < 0, the input otherwise), so every row computes the
// same tap-ordered F32 sum as the torch token-major conv:
//   out[t] = silu(s0*w0 + s1*w1 + s2*w2 + x[t]*w3)
//
// H26: the member owning the **last real token of the sequence** publishes
// next_state = the three inputs before it. A bucketed prefill is padded to the
// bucket length, so the end of the slot is *not* the end of the sequence: the
// torch reference (hpu_causal_conv1d_fn) gathers the new state at
// actual_qlens + [0, 1, 2] inside cat(init_state, x). The kernel therefore
// derives each sequence's real length from `state_stop`, the runner's
// query_start_loc (cumsum of the actual query lengths, [S + 1] I32):
//   real = clamp(state_stop[seq + 1] - state_stop[seq], 0, tokens_per_seq)
// and only that member writes, so nothing is ever published from a padded row.
// real == 0 leaves the slot untouched in the reference (update_mask =
// actual_qlens > 0), which is what the pool row copy below reproduces.
//
// Layouts (TPC dim 0 = torch last dim):
//   mixed     BF16 [C, T];  conv_pool BF16 [C, 3, Slots]
//   weight_t  BF16 [C, 4];  indices I32 [S];  has_init I32 [S]
//   state_stop I32 [S + 1]; output BF16 [C, T]; next_state BF16 [C, 3, S]
#include "qwen4exp_gdn_common.h"

#define ROW_BLOCK 32

// next_state[col : col + 128, j, seq] for j = 0, 1, 2.
static inline void store_state(int col, int seq, int slot, int init, float128 s0, float128 s1,
                               float128 s2, tensor next_state) {
    const int5 n0 = {col, 0, seq, 0, 0};
    const int5 n1 = {col, 1, seq, 0, 0};
    const int5 n2 = {col, 2, seq, 0, 0};
    v_bf16_st_tnsr(n0, next_state, v_convert_f32_to_bf16_all_b(s0));
    v_bf16_st_tnsr(n1, next_state, v_convert_f32_to_bf16_all_b(s1));
    v_bf16_st_tnsr(n2, next_state, v_convert_f32_to_bf16_all_b(s2));
}

void main(tensor mixed, tensor conv_pool, tensor weight_t, tensor indices, tensor has_init,
          tensor state_stop, tensor output, tensor next_state) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int channels = get_dim_size(mixed, 0);
    const int tokens = get_dim_size(mixed, 1);
    const int seqs = get_dim_size(indices, 0);
    const int tokens_per_seq = tokens / seqs;
    for (int seq = start[2]; seq < end[2]; ++seq) {
        const int5 ic = {seq, 0, 0, 0, 0};
        const int slot = s_i32_ld_g((__global__ int*)gen_addr(ic, indices));
        const int init = s_i32_ld_g((__global__ int*)gen_addr(ic, has_init));
        // Real (unpadded) length of this sequence's span.
        const int5 ic_next = {seq + 1, 0, 0, 0, 0};
        const int span_start = s_i32_ld_g((__global__ int*)gen_addr(ic, state_stop));
        const int span_end = s_i32_ld_g((__global__ int*)gen_addr(ic_next, state_stop));
        int real = span_end - span_start;
        if (real < 0) real = 0;
        if (real > tokens_per_seq) real = tokens_per_seq;
        const int state_row = real - 1;
        for (int part = start[1]; part < end[1]; ++part) {
            const int first = part * ROW_BLOCK;
            const int stop = first + ROW_BLOCK < tokens_per_seq ? first + ROW_BLOCK : tokens_per_seq;
            for (int block = start[0]; block < end[0]; ++block) {
                const int col = block * 128;
                if (col >= channels) break;
                float128 weights[4];
#pragma unroll
                for (int tap = 0; tap < 4; ++tap) {
                    const int5 wc = {col, tap, 0, 0, 0};
                    weights[tap] = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, weight_t));
                }
                // Window w0..w2: the three inputs before `first`, oldest first.
                float128 w0 = {0}, w1 = {0}, w2 = {0};
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int src = first - 3 + j;
                    float128 old = {0};
                    if (src < 0) {
                        if (init) {
                            const int5 sc = {col, src + 3, slot, 0, 0};
                            old = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, conv_pool));
                        }
                    } else {
                        const int5 sc = {col, seq * tokens_per_seq + src, 0, 0, 0};
                        old = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, mixed));
                    }
                    if (j == 0) w0 = old; else if (j == 1) w1 = old; else w2 = old;
                }
                for (int row = first; row < stop; ++row) {
                    const int token = seq * tokens_per_seq + row;
                    const int5 xc = {col, token, 0, 0, 0};
                    const float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, mixed));
                    float128 sum = {0};
                    sum.v1 += w0.v1 * weights[0].v1; sum.v2 += w0.v2 * weights[0].v2;
                    sum.v1 += w1.v1 * weights[1].v1; sum.v2 += w1.v2 * weights[1].v2;
                    sum.v1 += w2.v1 * weights[2].v1; sum.v2 += w2.v2 * weights[2].v2;
                    sum.v1 += x.v1 * weights[3].v1; sum.v2 += x.v2 * weights[3].v2;
                    w0 = w1; w1 = w2; w2 = x;
                    const float128 y = qwen4exp_gdn_silu128(sum);
                    v_bf16_st_tnsr(xc, output, v_convert_f32_to_bf16_all_b(y));
                }
                // Publish the state after the *last real token* of the span:
                // (x[real - 3], x[real - 2], x[real - 1]) with negative
                // indices served from the carried conv state -- the same three
                // rows hpu_causal_conv1d_fn gathers at actual_qlens + [0, 1, 2].
                if (state_row >= first && state_row < stop) {
                    float128 s0 = {0}, s1 = {0}, s2 = {0};
#pragma unroll
                    for (int j = 0; j < 3; ++j) {
                        const int src = state_row - 2 + j;
                        float128 old = {0};
                        if (src < 0) {
                            if (init) {
                                const int5 sc = {col, src + 3, slot, 0, 0};
                                old = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, conv_pool));
                            }
                        } else {
                            const int5 sc = {col, seq * tokens_per_seq + src, 0, 0, 0};
                            old = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, mixed));
                        }
                        if (j == 0) s0 = old; else if (j == 1) s1 = old; else s2 = old;
                    }
                    store_state(col, seq, slot, init, s0, s1, s2, next_state);
                } else if (real == 0 && first == 0) {
                    // No real token: the reference leaves the cached rows as
                    // they are (update_mask = actual_qlens > 0), so copy them.
                    const int5 p0 = {col, 0, slot, 0, 0};
                    const int5 p1 = {col, 1, slot, 0, 0};
                    const int5 p2 = {col, 2, slot, 0, 0};
                    const float128 s0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p0, conv_pool));
                    const float128 s1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p1, conv_pool));
                    const float128 s2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p2, conv_pool));
                    store_state(col, seq, slot, init, s0, s1, s2, next_state);
                }
            }
        }
    }
}
