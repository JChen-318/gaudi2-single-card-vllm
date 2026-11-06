// SPDX-License-Identifier: Apache-2.0
// H13: epilogue of the merged down+inject GEMM.
//   merged [T, M] BF16 (M = lowrank + streams + pad, or just lowrank)
//   -> low [T, lowrank] BF16: silu(merged[:, :lowrank] / streams)
//   -> inj [T, streams] BF16: merged[:, lowrank:lowrank+streams] (copy)
// Fusing this removes the split + .float()/divide/silu/cast chain (and the
// contiguous copy a sliced silu input would need). Tile 0 also writes the
// injection slice; when lowrank == M (mix-only module) nothing is written to
// inj and the caller discards it. Trailing tiles over-read (never over-store),
// the same contract as the indexed-dequant kernels.
#include "qwen4exp_hc_common.h"

void main(tensor merged, tensor low_out, tensor inject_out) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int lowrank = get_dim_size(low_out, 0);
    const int streams = get_dim_size(inject_out, 0);
    const int merged_width = get_dim_size(merged, 0);
    const float inv_streams = 1.0f / (float)streams;
    for (int token = start[1]; token < end[1]; ++token) {
        for (int tile = start[0]; tile < end[0]; ++tile) {
            const int col = tile * 128;
            if (col >= lowrank) break;
            const int5 p = {col, token, 0, 0, 0};
            const float128 v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, merged));
            float128 z;
            z.v1 = v.v1 * inv_streams;
            z.v2 = v.v2 * inv_streams;
            float128 y;
            y.v1 = z.v1 * qwen4exp_hc_sigmoid(z.v1);
            y.v2 = z.v2 * qwen4exp_hc_sigmoid(z.v2);
            const bfloat128 out = v_convert_f32_to_bf16_all_b(y);
            if (col + 128 <= lowrank) {
                v_bf16_st_tnsr(p, low_out, out);
            } else {
                v_bf16_st_tnsr_partial(p, low_out, out, lowrank - col - 1, 0);
            }
            if (tile == 0 && lowrank < merged_width) {
                const int5 src = {lowrank, token, 0, 0, 0};
                const int5 dst = {0, token, 0, 0, 0};
                const bfloat128 inj = v_bf16_ld_tnsr_b(src, merged);
                v_bf16_st_tnsr_partial(dst, inject_out, inj, streams - 1, 0);
            }
        }
    }
}
