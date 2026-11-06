// SPDX-License-Identifier: Apache-2.0
// H13: fused gated stream mix (Strata hc_mix_dual.c, BF16 form).
//   out[t, d] = (1/S) * sum_s sigmoid(gate[t,s,d]) * xn[t,s,d]
// Torch: xn/gate [T, S*D] BF16, out [T, D] BF16. Each 128-wide tile reads the
// S streams of xn/gate once and accumulates in F32 registers; the unfused
// version materialized sigmoid/gate products over the full [T, S*D] plane.
#include "qwen4exp_hc_common.h"

void main(tensor normed, tensor gate, tensor output) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(output, 0);
    const int total = get_dim_size(normed, 0);
    const int streams = total / width;
    const float inv_streams = 1.0f / (float)streams;
    for (int token = start[1]; token < end[1]; ++token) {
        for (int tile = start[0]; tile < end[0]; ++tile) {
            const int col = tile * 128;
            if (col >= width) break;
            float128 sum = {0};
            for (int stream = 0; stream < streams; ++stream) {
                const int5 p = {stream * width + col, token, 0, 0, 0};
                const float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, normed));
                const float128 g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, gate));
                sum.v1 += x.v1 * qwen4exp_hc_sigmoid(g.v1);
                sum.v2 += x.v2 * qwen4exp_hc_sigmoid(g.v2);
            }
            sum.v1 *= inv_streams;
            sum.v2 *= inv_streams;
            const int5 p = {col, token, 0, 0, 0};
            const bfloat128 out = v_convert_f32_to_bf16_all_b(sum);
            if (col + 128 <= width) {
                v_bf16_st_tnsr(p, output, out);
            } else {
                v_bf16_st_tnsr_partial(p, output, out, width - col - 1, 0);
            }
        }
    }
}
