// SPDX-License-Identifier: Apache-2.0
// H13: fused HyperConnection combine (Strata hc_write.c, BF16 form).
//   out[t, s, d] = residual[t, s, d] + block[t, d] * 2*sigmoid(inj[t, s]/S)
// Torch: residual [T, S*D], block [T, D], inj [T, S], out [T, S*D], all BF16.
// The per-stream scalar is read once per (stream, token) index point; the
// 128-wide tiles then stream residual/block from HBM exactly once.
#include "qwen4exp_hc_common.h"

void main(tensor residual, tensor block, tensor inject, tensor output) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(block, 0);
    const int streams = get_dim_size(inject, 0);
    for (int token = start[2]; token < end[2]; ++token) {
        for (int stream = start[1]; stream < end[1]; ++stream) {
            const int5 ic = {stream, token, 0, 0, 0};
            const float inj = (float)s_bf16_ld_g(gen_addr(ic, inject));
            float64 scalar = inj * (1.0f / (float)streams);
            const float64 factor = qwen4exp_hc_sigmoid(scalar) * 2.0f;
            float128 factor128;
            factor128.v1 = factor;
            factor128.v2 = factor;
            const int base = stream * width;
            for (int tile = start[0]; tile < end[0]; ++tile) {
                const int col = tile * 128;
                if (col >= width) break;
                const int5 p = {base + col, token, 0, 0, 0};
                const int5 b = {col, token, 0, 0, 0};
                const float128 r = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, residual));
                const float128 v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(b, block));
                float128 y;
                y.v1 = r.v1 + v.v1 * factor128.v1;
                y.v2 = r.v2 + v.v2 * factor128.v2;
                const bfloat128 out = v_convert_f32_to_bf16_all_b(y);
                if (col + 128 <= width) {
                    v_bf16_st_tnsr(p, output, out);
                } else {
                    v_bf16_st_tnsr_partial(p, output, out, width - col - 1, 0);
                }
            }
        }
    }
}
