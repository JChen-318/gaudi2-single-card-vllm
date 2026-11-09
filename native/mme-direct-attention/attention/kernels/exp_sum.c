// SPDX-License-Identifier: MIT
// One row per index point. Scores and probabilities may share SRAM storage.
// EXP_GROUP (1, 2 or 4) processes that many independent 128-column chunks per
// loop iteration: loads first, then interleavable math, then stores. Scores and
// probabilities alias, so the compiler cannot overlap iterations by itself.
#include "reduce_no_lut.h"
#ifndef EXP_GROUP
#define EXP_GROUP 1
#endif
#ifndef EXP_BASE2
static float64 approx_exp(float64 x) {
    x = v_f32_max_b(x, -87.0f);
    x = v_f32_min_b(x, 0.0f);
    int64 n = v_convert_f32_to_i32_b(x * 1.4426950408889634f, SW_RHNE);
    float64 r = x - v_convert_i32_to_f32_b(n, 0) * 0.6931471805599453f;
#ifdef EXP_FAST_CUBIC
    // Cubic interpolation at four Chebyshev nodes on [-ln(2)/2, ln(2)/2].
    // Residual relative error < 1.0e-4 in real arithmetic, below BF16 rounding.
    float64 p = 0.16767011875166407f;
    p = v_f32_mac_b(p, r, 0.5050222842135575f);
    p = v_f32_mac_b(p, r, 0.9999849286318752f);
    p = v_f32_mac_b(p, r, 0.9999245569508709f);
#else
    float64 p = 0.008333333333333333333f;
    p = v_f32_mac_b(p, r, 0.041666666666666664f);
    p = v_f32_mac_b(p, r, 0.16666666666666666f);
    p = v_f32_mac_b(p, r, 0.5f);
    p = v_f32_mac_b(p, r, 1.0f);
    p = v_f32_mac_b(p, r, 1.0f);
#endif
    int64 bits = (n + 127) << 23;
    float64 scale = *((float64 *)&bits);
    return p * scale;
}
#endif

#ifdef EXP_BASE2
// exp(x - b) = 2^t with t = x*log2(e) + (127 - b*log2(e)); the IEEE exponent
// bias is folded into t. t >= 1 keeps the exponent field valid (underflow gives
// ~1e-38, not 0). Relative-minimax polynomial for 2^f, f in [-0.5, 0.5]:
// degree 3 max rel. error 7.5e-5, degree 4 2.6e-6 (BF16 half-ulp is 2e-3).
static float64 approx_exp_biased(float64 x, float64 nb) {
    float64 t = v_f32_mac_b(x, 1.4426950408889634f, nb);
    t = v_f32_max_b(t, 1.0f);
    int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    float64 f = t - v_convert_i32_to_f32_b(n, 0);
#if EXP_BASE2 == 4
    float64 p = 0.009570609778165817f;
    p = v_f32_mac_b(p, f, 0.055917903780937195f);
    p = v_f32_mac_b(p, f, 0.24024732410907745f);
    p = v_f32_mac_b(p, f, 0.6931217908859253f);
    p = v_f32_mac_b(p, f, 0.9999992847442627f);
#else
    float64 p = 0.05517604947090149f;
    p = v_f32_mac_b(p, f, 0.24261149764060974f);
    p = v_f32_mac_b(p, f, 0.6932601928710938f);
    p = v_f32_mac_b(p, f, 0.9999280571937561f);
#endif
    int64 bits = n << 23;
    return p * *((float64 *)&bits);
}
#define EXP_EVAL(v) approx_exp_biased((v), nb)
#else
#define EXP_EVAL(v) approx_exp((v) - b)
#endif

// exp(x - b) for 128 BF16 scores; masks columns >= valid_cols; adds the rounded
// BF16 probabilities to acc so the row sum matches what PV consumes.
#define EXP_CHUNK(x, column, out, acc) do {                                          \
        float128 y_;                                                                \
        y_.v1 = EXP_EVAL((x).v1); y_.v2 = EXP_EVAL((x).v2);                         \
        bfloat128 pr_ = v_convert_f32_to_bf16_all_b(y_);                            \
        pr_ = v_bf16_sel_less_u16_b(read_lane_id_2b_b() + (column),                 \
                (unsigned short)valid_cols, pr_, (bfloat128)0);                     \
        (out) = pr_;                                                                \
        float128 rd_ = v_convert_bf16_to_f32_all_b(pr_);                            \
        (acc) += rd_.v1 + rd_.v2;                                                   \
    } while (0)

void main(tensor scores, tensor bounds, tensor probs, tensor sums, int valid_cols) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    const int cols = get_dim_size(scores, 0);
    for (int row = start[1]; row < end[1]; ++row) {
        int5 bp = {0, row, 0, 0, 0};
        float128 bv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(bp, bounds));
        float64 b = v_broadcast_element_f32(bv.v1, 0);
#ifdef EXP_BASE2
        float64 nb = 127.0f - b * 1.4426950408889634f;
#endif
        float64 sum = 0;
        int col = 0;
#if EXP_GROUP >= 2
        float64 s1 = 0;
#endif
#if EXP_GROUP == 4
        float64 s2 = 0, s3 = 0;
#endif
#if EXP_GROUP >= 2
        for (; col + 128 * EXP_GROUP <= cols; col += 128 * EXP_GROUP) {
            int5 p0 = {col, row, 0, 0, 0}, p1 = {col + 128, row, 0, 0, 0};
            float128 x0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p0, scores));
            float128 x1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p1, scores));
#if EXP_GROUP == 4
            int5 p2 = {col + 256, row, 0, 0, 0}, p3 = {col + 384, row, 0, 0, 0};
            float128 x2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p2, scores));
            float128 x3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p3, scores));
#endif
            bfloat128 o0, o1;
            EXP_CHUNK(x0, col, o0, sum);
            EXP_CHUNK(x1, col + 128, o1, s1);
#if EXP_GROUP == 4
            bfloat128 o2, o3;
            EXP_CHUNK(x2, col + 256, o2, s2);
            EXP_CHUNK(x3, col + 384, o3, s3);
#endif
            v_bf16_st_tnsr(p0, probs, o0);
            v_bf16_st_tnsr(p1, probs, o1);
#if EXP_GROUP == 4
            v_bf16_st_tnsr(p2, probs, o2);
            v_bf16_st_tnsr(p3, probs, o3);
#endif
        }
#endif
        for (; col < cols; col += 128) {
            int5 p = {col, row, 0, 0, 0};
            float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, scores));
            bfloat128 o;
            EXP_CHUNK(x, col, o, sum);
            v_bf16_st_tnsr(p, probs, o);
        }
#if EXP_GROUP >= 2
        sum += s1;
#endif
#if EXP_GROUP == 4
        sum += s2 + s3;
#endif
        v_f32_st_tnsr(bp, sums, sum64_no_lut(sum));
    }
}
