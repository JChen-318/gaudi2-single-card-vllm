// SPDX-License-Identifier: MIT
// Probe kernel: one builtin exp per 128-element chunk; FN/TYPE chosen by -D.
void main(tensor in, tensor out) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    const int cols = get_dim_size(in, 0);
    for (int row = start[1]; row < end[1]; ++row) {
        for (int col = 0; col < cols; col += 128) {
            int5 p = {col, row, 0, 0, 0};
            bfloat128 x = v_bf16_ld_tnsr_b(p, in);
#ifdef BF16_FN
            bfloat128 y = BF16_FN(x);
#else
            float128 xf = v_convert_bf16_to_f32_all_b(x);
            float128 yf; yf.v1 = F32_FN(xf.v1); yf.v2 = F32_FN(xf.v2);
            bfloat128 y = v_convert_f32_to_bf16_all_b(yf);
#endif
            v_bf16_st_tnsr(p, out, y);
        }
    }
}
