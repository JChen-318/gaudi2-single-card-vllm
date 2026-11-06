// SPDX-License-Identifier: Apache-2.0
// H13: fused combine + grouped GemmaRMSNorm (the combine_and_mix boundary).
//   combined[t,s,d] = residual[t,s,d] + block[t,d] * 2*sigmoid(inj[t,s]/S)
//   xn              = grouped_gemma_rmsnorm(combined)
// Pass 1 computes combined in F32, rounds it to BF16 (the unfused pipeline's
// combine output is a BF16 tensor, and the norm reads that tensor), stores
// it, and accumulates the row's sum of squares from the *rounded* values, so
// the variance sees exactly the reference's inputs. Pass 2 recomputes the
// rounded combined values from residual/block (a TPC kernel must not read
// its own output tensor: that memory is uninitialized on the first launch)
// and normalizes. One launch replaces combine (~8 ops) + norm (~12 ops).
#include "qwen4exp_hc_common.h"

void main(tensor residual, tensor block, tensor inject, tensor gamma,
          tensor combined_out, tensor normed_out, float epsilon) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(gamma, 0);
    const int total = get_dim_size(residual, 0);
    const int streams = total / width;
    const int gamma_rows = get_dim_size(gamma, 1);
    const float inv_width = 1.0f / (float)width;
    for (int token = start[1]; token < end[1]; ++token) {
        for (int stream = start[0]; stream < end[0]; ++stream) {
            const int5 ic = {stream, token, 0, 0, 0};
            const float inj = (float)s_bf16_ld_g(gen_addr(ic, inject));
            float64 scalar = inj * (1.0f / (float)streams);
            const float64 factor = qwen4exp_hc_sigmoid(scalar) * 2.0f;
            float128 factor128;
            factor128.v1 = factor;
            factor128.v2 = factor;
            const int base = stream * width;
            const int grow = gamma_rows == 1 ? 0 : stream;
            float128 squares = {0};
            for (int col = 0; col < width; col += 128) {
                const int5 p = {base + col, token, 0, 0, 0};
                const int5 b = {col, token, 0, 0, 0};
                const float128 r = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, residual));
                const float128 v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(b, block));
                float128 c;
                c.v1 = r.v1 + v.v1 * factor128.v1;
                c.v2 = r.v2 + v.v2 * factor128.v2;
                const bfloat128 rounded = v_convert_f32_to_bf16_all_b(c);
                v_bf16_st_tnsr(p, combined_out, rounded);
                const float128 cf = v_convert_bf16_to_f32_all_b(rounded);
                squares.v1 += cf.v1 * cf.v1;
                squares.v2 += cf.v2 * cf.v2;
            }
            const float64 inverse = v_rsqrt_f32(
                v_f32_reduce_add(squares.v1 + squares.v2) * inv_width + epsilon);
            float128 inv128;
            inv128.v1 = inverse;
            inv128.v2 = inverse;
            for (int col = 0; col < width; col += 128) {
                const int5 p = {base + col, token, 0, 0, 0};
                const int5 b = {col, token, 0, 0, 0};
                const int5 g = {col, grow, 0, 0, 0};
                const float128 r = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, residual));
                const float128 v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(b, block));
                float128 c;
                c.v1 = r.v1 + v.v1 * factor128.v1;
                c.v2 = r.v2 + v.v2 * factor128.v2;
                const float128 x = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(c));
                const float128 w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(g, gamma));
                float128 y;
                y.v1 = (x.v1 * inv128.v1) * (1.0f + w.v1);
                y.v2 = (x.v2 * inv128.v2) * (1.0f + w.v2);
                v_bf16_st_tnsr(p, normed_out, v_convert_f32_to_bf16_all_b(y));
            }
        }
    }
}
