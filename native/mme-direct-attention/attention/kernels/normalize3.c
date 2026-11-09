// SPDX-License-Identifier: MIT
// Merge two key halves with per-half maxima (FlashAttention-style):
//   m = max(mA, mB), wA = 2^((mA - m) log2 e), wB likewise,
//   out = (wA * a + wB * b) / (wA * lA + wB * lB).
// stats rows r and r + 128 hold {m, l} of halves A and B (FP32, 256 x 2).
// No special-function LUT: base-2 cubic (rel. err. 7.5e-5) and Newton reciprocal.
static float64 exp_nonpos(float64 x) {
    float64 t = v_f32_mac_b(x, 1.4426950408889634f, (float64)127.0f);
    t = v_f32_max_b(t, 1.0f);
    int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.05517604947090149f;
    p = v_f32_mac_b(p, f, 0.24261149764060974f);
    p = v_f32_mac_b(p, f, 0.6932601928710938f);
    p = v_f32_mac_b(p, f, 0.9999280571937561f);
    int64 bits = n << 23;
    return p * *((float64 *)&bits);
}

void main(tensor a, tensor b, tensor stats, tensor output) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    int cols = get_dim_size(a, 0);
    for (int row = start[1]; row < end[1]; ++row) {
        float64 sa = v_f32_ld_tnsr_b((int5){0, row, 0, 0, 0}, stats);
        float64 sb = v_f32_ld_tnsr_b((int5){0, row + 128, 0, 0, 0}, stats);
        float64 ma = v_broadcast_element_f32(sa, 0), la = v_broadcast_element_f32(sa, 1);
        float64 mb = v_broadcast_element_f32(sb, 0), lb = v_broadcast_element_f32(sb, 1);
        float64 m = v_f32_max_b(ma, mb);
        float64 wa = exp_nonpos(ma - m), wb = exp_nonpos(mb - m);
        float64 l = v_f32_max_b(wa * la + wb * lb, 1e-30f);
        int64 bits = *((int64 *)&l);
        bits = 0x7ef311c3 - bits;
        float64 inv = *((float64 *)&bits);
        inv *= 2.0f - l * inv;
        inv *= 2.0f - l * inv;
        inv *= 2.0f - l * inv;
        wa *= inv; wb *= inv;
        for (int col = 0; col < cols; col += 128) {
            int5 p = {col, row, 0, 0, 0};
            float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, a));
            float128 y = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, b));
            x.v1 = x.v1 * wa + y.v1 * wb; x.v2 = x.v2 * wa + y.v2 * wb;
            v_bf16_st_tnsr(p, output, v_convert_f32_to_bf16_all_b(x));
        }
    }
}
