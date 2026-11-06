// SPDX-License-Identifier: Apache-2.0
// H14: fused decode causal conv1d + conv-state update + SiLU, one launch.
//
// Per (sequence, channel): out = silu(sum_tap state[tap] * w[tap] + x * w[3])
// with the state shifted by one token (next = [state1, state2, x]), exactly
// the torch hpu_causal_conv1d_fn_update decode step (F32 accumulation, tap
// order 0..3, SiLU in F32, BF16 outputs). Replaces the index_select /
// transpose / cat / 4-tap depthwise / activation / index_copy chain.
//
// Layouts (TPC dim 0 = torch last dim):
//   mixed     BF16 [C, B];  conv_pool BF16 [C, 3, Slots]
//   weight_t  BF16 [C, 4]   (cached tap-major transpose of conv1d.weight)
//   indices   I32  [B]      (caller-mapped into [0, Slots))
//   output    BF16 [C, B];  next_state BF16 [C, 3, B]
#include "qwen4exp_gdn_common.h"

void main(tensor mixed, tensor conv_pool, tensor weight_t, tensor indices,
          tensor output, tensor next_state) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int channels = get_dim_size(mixed, 0);
    for (int seq = start[1]; seq < end[1]; ++seq) {
        const int5 ic = {seq, 0, 0, 0, 0};
        const int slot = s_i32_ld_g((__global__ int*)gen_addr(ic, indices));
        for (int block = start[0]; block < end[0]; ++block) {
            const int col = block * 128;
            if (col >= channels) break;
            const int5 xc = {col, seq, 0, 0, 0};
            const bfloat128 x_raw = v_bf16_ld_tnsr_b(xc, mixed);
            float128 sum = {0};
            bfloat128 taps[3];
#pragma unroll
            for (int tap = 0; tap < 3; ++tap) {
                const int5 sc = {col, tap, slot, 0, 0};
                const int5 wc = {col, tap, 0, 0, 0};
                taps[tap] = v_bf16_ld_tnsr_b(sc, conv_pool);
                const float128 s = v_convert_bf16_to_f32_all_b(taps[tap]);
                const float128 w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, weight_t));
                sum.v1 += s.v1 * w.v1;
                sum.v2 += s.v2 * w.v2;
            }
            {
                const int5 wc = {col, 3, 0, 0, 0};
                const float128 x = v_convert_bf16_to_f32_all_b(x_raw);
                const float128 w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, weight_t));
                sum.v1 += x.v1 * w.v1;
                sum.v2 += x.v2 * w.v2;
                const float128 y = qwen4exp_gdn_silu128(sum);
                v_bf16_st_tnsr(xc, output, v_convert_f32_to_bf16_all_b(y));
            }
            // Shifted state, stored from the original BF16 taps (exact).
            const int5 n0 = {col, 0, seq, 0, 0};
            const int5 n1 = {col, 1, seq, 0, 0};
            const int5 n2 = {col, 2, seq, 0, 0};
            v_bf16_st_tnsr(n0, next_state, taps[1]);
            v_bf16_st_tnsr(n1, next_state, taps[2]);
            v_bf16_st_tnsr(n2, next_state, x_raw);
        }
    }
}
