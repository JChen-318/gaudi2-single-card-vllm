// SPDX-License-Identifier: MIT
// out = (a + b) / l for two BF16 partial PV results (key halves on two MMEs).
// Positive-denominator reciprocal as in normalize.c (no special-function LUT).
void main(tensor a, tensor b, tensor sums, tensor output) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    int cols = get_dim_size(a, 0);
    for (int row = start[1]; row < end[1]; ++row) {
        int5 rp = {0, row, 0, 0, 0};
        float64 l = v_broadcast_element_f32(v_f32_ld_tnsr_b(rp, sums), 0);
        l = v_f32_max_b(l, 1e-30f);
        int64 bits = *((int64 *)&l);
        bits = 0x7ef311c3 - bits;
        float64 inv = *((float64 *)&bits);
        inv *= 2.0f - l * inv;
        inv *= 2.0f - l * inv;
        inv *= 2.0f - l * inv;
        for (int col = 0; col < cols; col += 128) {
            int5 p = {col, row, 0, 0, 0};
            float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, a));
            float128 y = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, b));
            x.v1 = (x.v1 + y.v1) * inv; x.v2 = (x.v2 + y.v2) * inv;
            v_bf16_st_tnsr(p, output, v_convert_f32_to_bf16_all_b(x));
        }
    }
}
