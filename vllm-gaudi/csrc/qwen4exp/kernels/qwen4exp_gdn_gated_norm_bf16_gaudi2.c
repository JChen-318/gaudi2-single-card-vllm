// SPDX-License-Identifier: Apache-2.0
// H14: fused gated RMSNorm for the GDN epilogue, one launch.
//   y[r, d] = x[r, d] * rsqrt(mean_d(x^2) + eps) * w[d] * act(z[r, d])
// rows are (token, value-head) pairs of width 128; act is sigmoid (the
// Qwen4Exp output gate) or silu. F32 math, BF16 in/out. This is
// RMSNormGated.forward_static with norm_before_gate=True, replacing its
// ~10 eager dispatches (float casts, pow/mean/rsqrt, multiplies, silu).
//
// Layouts: x/z/output BF16 [128, R] (TPC), weight BF16 [128].
// Scalar params: { float eps; int act; } with act 0 = sigmoid, 1 = silu.
#include "qwen4exp_gdn_common.h"

void main(tensor input, tensor z, tensor weight, tensor output, float epsilon, int act) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(input, 0);
    const int rows = get_dim_size(input, 1);
    const float inv_width = 1.0f / (float)width;
    const int5 wc = {0, 0, 0, 0, 0};
    const float128 w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, weight));
    // 8 rows per index point in straight-line code, so the compiler can
    // overlap the rows' loads and reductions (Strata gdn_norm_rows8 form).
    for (int group = start[0]; group < end[0]; ++group) {
        float128 xs[8], gs[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = group * 8 + i < rows ? group * 8 + i : rows - 1;
            const int5 p = {0, row, 0, 0, 0};
            xs[i] = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, input));
            gs[i] = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, z));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = group * 8 + i;
            float64 squares = v_f32_mac_b(xs[i].v2, xs[i].v2, xs[i].v1 * xs[i].v1);
            const float64 inv = v_rsqrt_f32(
                v_broadcast_element_f32(v_f32_reduce_add(squares), 0) * inv_width + epsilon);
            float128 inv128;
            inv128.v1 = inv;
            inv128.v2 = inv;
            float128 gate;
            if (act == 1) {
                gate.v1 = gs[i].v1 * v_sigmoid_f32(gs[i].v1);
                gate.v2 = gs[i].v2 * v_sigmoid_f32(gs[i].v2);
            } else {
                gate.v1 = v_sigmoid_f32(gs[i].v1);
                gate.v2 = v_sigmoid_f32(gs[i].v2);
            }
            float128 y;
            y.v1 = (xs[i].v1 * inv128.v1) * w.v1 * gate.v1;
            y.v2 = (xs[i].v2 * inv128.v2) * w.v2 * gate.v2;
            const int5 p = {0, row, 0, 0, 0};
            if (row < rows) v_bf16_st_tnsr(p, output, v_convert_f32_to_bf16_all_b(y));
        }
    }
}
