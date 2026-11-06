// SPDX-License-Identifier: Apache-2.0
// H13: fused grouped GemmaRMSNorm for HyperConnection.
//
// Torch: x [T, S*D] BF16, gamma [Sg, D] BF16 (Sg == 1 shared or S streams),
// out [T, S*D] BF16. Per (token, stream) row of width D:
//   y = x * rsqrt(mean(x^2) + eps) * (1 + gamma[stream])   (F32 math, BF16 out)
// This replaces the ~12 eager ops of hc_grouped_gemma_rmsnorm (float, square,
// mean, rsqrt, unflatten/expand, multiply, cast) with one TPC launch. The sum
// of squares is a plain F32 accumulation (no max pre-scaling, cf. Strata's
// row_norm_dual fast path): BF16 inputs cannot overflow it in practice.
// TPC sees x [S*D, T], gamma [D, Sg]; one index point owns one (stream, token).
#include "qwen4exp_hc_common.h"

void main(tensor input, tensor gamma, tensor output, float epsilon) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(gamma, 0);
    const int gamma_rows = get_dim_size(gamma, 1);
    const float inv_width = 1.0f / (float)width;
    for (int token = start[1]; token < end[1]; ++token) {
        for (int stream = start[0]; stream < end[0]; ++stream) {
            const int base = stream * width;
            const int grow = gamma_rows == 1 ? 0 : stream;
            float128 squares = {0};
            for (int col = 0; col < width; col += 128) {
                const int5 p = {base + col, token, 0, 0, 0};
                const float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, input));
                squares.v1 += x.v1 * x.v1;
                squares.v2 += x.v2 * x.v2;
            }
            const float64 inverse = v_rsqrt_f32(
                v_f32_reduce_add(squares.v1 + squares.v2) * inv_width + epsilon);
            float128 inv128;
            inv128.v1 = inverse;
            inv128.v2 = inverse;
            for (int col = 0; col < width; col += 128) {
                const int5 p = {base + col, token, 0, 0, 0};
                const int5 g = {col, grow, 0, 0, 0};
                const float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, input));
                const float128 w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(g, gamma));
                float128 y;
                y.v1 = (x.v1 * inv128.v1) * (1.0f + w.v1);
                y.v2 = (x.v2 * inv128.v2) * (1.0f + w.v2);
                v_bf16_st_tnsr(p, output, v_convert_f32_to_bf16_all_b(y));
            }
        }
    }
}
