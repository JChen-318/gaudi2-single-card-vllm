// SPDX-License-Identifier: MIT
// Per-half softmax numerator for key-split scores (FlashAttention-style merge).
// Work unit u in [0, 256): half h = u / 128 (0: A = S[:, 0:first), 1: B), row u % 128.
// Each unit uses the exact max m_h of its own valid columns:
//   p = exp_bf16(s - m_h) written in place, l_h = sum of the stored BF16 p.
// stats[u] = {m_h, l_h} (FP32, 256 x 2). normalize3 merges the two halves.
// 256 units over 24 TPCs balance to <= 11 units each (vs 6 rows = 12 half-units).
// REQUIRES the Gaudi2 special-function tables and LUT_FUNC128_BASE_ADDR set.
#include "reduce_no_lut.h"
#include "reduce_max_no_lut.h"

#define LANES(column) (read_lane_id_2b_b() + (column))
#define LD(t, column) v_bf16_ld_tnsr_b((int5){(column), row, 0, 0, 0}, (t))
#define ST(t, column, y) v_bf16_st_tnsr((int5){(column), row, 0, 0, 0}, (t), (y))
#ifdef EXP_LOGQ
// Quantize log2(p) to 1/128, then correct the BF16 mantissa through private
// LUT128 slot 14. FP32 affine evaluation keeps s-m from rounding to BF16.
// The slot contains round(128*(2^(j/128)-1))-j, j in [0,127].
static bfloat128 exp_logq(bfloat128 score, float64 bias, bfloat128 ceiling) {
#ifdef EXP_LOGQ_BOUND80
    // Probe enforces ||q||*max||k|| <= 80 before launch. Thus BF16 scores
    // and sampled maxima are within +/-80.25 and the affine lower bound is
    // above -14000. Clipping s <= m_hat+64 bounds the upper value below 29000:
    // conversion fits int16 without relying on overflow/saturation semantics.
    score = v_bf16_min_b(score, ceiling);
#else
    (void)ceiling;
#endif
    float128 t = v_convert_bf16_to_f32_all_b(score);
    t.v1 = v_f32_mac_b(t.v1, 184.6649652337873f, bias);
    t.v2 = v_f32_mac_b(t.v2, 184.6649652337873f, bias);
    // Bound the affine value BEFORE integer conversion; do not rely on the
    // hardware's out-of-range conversion behavior. The packed result is finite.
#ifndef EXP_LOGQ_BOUND80
    t.v1 = v_f32_min_b(v_f32_max_b(t.v1, 128.0f), 32639.0f);
    t.v2 = v_f32_min_b(v_f32_max_b(t.v2, 128.0f), 32639.0f);
#endif
    short128 bits = v_convert_f32_to_i16_all_b(t, SW_RHNE);
#ifdef EXP_LOGQ_BOUND80
    bits = v_i16_max_b(bits, (short128)128);
#endif
    ushort128 idx = *(ushort128 *)&bits & (ushort128)127;
    short128 correction = v_i16_lookup_1c(idx, 142, 1, (short128)0, 1, 0);
    bits += correction;
    return *(bfloat128 *)&bits;
}
#define EXPV(x) exp_logq((x), logq_bias, logq_ceiling)
#elif defined(EXP_MANUAL)
// The finite-input core of the builtin exp_bf16 (same operations, constants and
// switch words as its LLVM IR), without the special-value handling (sel_leq,
// sel_grt, fclass, calc_fp_special). Inputs here are s - m <= 0 and finite; the
// clamp at -87 keeps the exponent sum from wrapping (result ~1e-38, not 0).
static bfloat128 exp_core(bfloat128 x) {
    x = v_bf16_max_b(x, (bfloat128)(-87.0f));
#ifdef EXP_MAX_STRIDE
    // m_hat may be below the true max: s - m_hat > 0. Clamp keeps exp and the FP32
    // row sum finite (only rows with max - m_hat > 64 are affected; measured <= 4).
    x = v_bf16_min_b(x, (bfloat128)64.0f);
#endif
    bfloat128 t = v_bf16_mac_b(x, (bfloat128)1.4453125f, (bfloat128)0.5f, 0, 1, 0);
    bfloat128 n = v_bf16_nearbyint_b(t, 198400, (bfloat128)0, 1, 0);
    short128 ni = v_convert_bf16_to_i16_b(n, 198464, (short128)0, 1, 0);
    bfloat128 r = v_bf16_mac_b(n, (bfloat128)(-0.6875f), x, 0, 1, 0);
    r = v_bf16_mac_b(n, (bfloat128)(-0.005645751953125f), r, 0, 1, 0);
    bfloat128 rb = r + (bfloat128)1.5f;
    ushort128 idx = v_u16_sub_b(*(ushort128 *)&rb, (ushort128)16199, 1, (ushort128)0, 1, 0);
    bfloat128 l = v_bf16_lookup_1c(idx, 138, 1, (bfloat128)0, 1, 0);
    short128 bits = (ni << 7) + *(short128 *)&l;
    return *(bfloat128 *)&bits;
}
#define EXPV(x) exp_core(v_bf16_sub_b((x), mb))
#else
#define EXPV(x) exp_bf16(v_bf16_sub_b((x), mb))
#endif
#define ACC(acc, y) ((acc) = v_bf16_mac_acc32_b((y), (bfloat128)1.0f, (acc), 0, 1, 0))

#ifdef EXP_G8
// Pass 2 with 8 independent chunks per iteration (loads first: S/P alias).
// FAILED on device 2026-11-09: wrong results (RMS 0.82, half sums < 1) with
// -DEXP_MANUAL -DEXP_G8 although 107 bundles/1024 columns; the generated code
// appears to reuse a lookup index register across chunks (not verified). Do not use.
#define PASS2_WIDE(t, full) \
        for (; c_ + 1024 <= (full); c_ += 1024) { \
            bfloat128 x0 = LD(t, c_), x1 = LD(t, c_ + 128), x2 = LD(t, c_ + 256), x3 = LD(t, c_ + 384); \
            bfloat128 x4 = LD(t, c_ + 512), x5 = LD(t, c_ + 640), x6 = LD(t, c_ + 768), x7 = LD(t, c_ + 896); \
            bfloat128 y0 = EXPV(x0), y1 = EXPV(x1), y2 = EXPV(x2), y3 = EXPV(x3); \
            bfloat128 y4 = EXPV(x4), y5 = EXPV(x5), y6 = EXPV(x6), y7 = EXPV(x7); \
            ST(t, c_, y0); ST(t, c_ + 128, y1); ST(t, c_ + 256, y2); ST(t, c_ + 384, y3); \
            ST(t, c_ + 512, y4); ST(t, c_ + 640, y5); ST(t, c_ + 768, y6); ST(t, c_ + 896, y7); \
            ACC(s0, y0); ACC(s1, y1); ACC(s2, y2); ACC(s3, y3); \
            ACC(s0, y4); ACC(s1, y5); ACC(s2, y6); ACC(s3, y7); \
        }
#else
#define PASS2_WIDE(t, full)
#endif

#ifdef EXP_MAX_STRIDE
// Sampled maximum: every EXP_MAX_STRIDE-th 128-column chunk (CPU study on three real
// blocks: end-to-end RMS unchanged to 1e-5; m - m_hat <= 3.94 at stride 8).
#define PASS1_SAMPLED(t, cols, full, lim) \
        for (; c_ < (full); c_ += 256 * EXP_MAX_STRIDE) { \
            m0 = v_bf16_max_b(m0, LD(t, c_)); \
            if (c_ + 128 * EXP_MAX_STRIDE < (full)) m1 = v_bf16_max_b(m1, LD(t, c_ + 128 * EXP_MAX_STRIDE)); \
        } \
        if ((full) == 0) \
            for (c_ = 0; c_ < (cols); c_ += 128) \
                m0 = v_bf16_max_b(m0, v_bf16_sel_less_u16_b(LANES(c_), (unsigned short)(lim), LD(t, c_), low));
#else
#define PASS1_SAMPLED(t, cols, full, lim) \
        for (; c_ + 512 <= (full); c_ += 512) {                                         \
            m0 = v_bf16_max_b(m0, LD(t, c_));       m1 = v_bf16_max_b(m1, LD(t, c_ + 128)); \
            m2 = v_bf16_max_b(m2, LD(t, c_ + 256)); m3 = v_bf16_max_b(m3, LD(t, c_ + 384)); \
        }                                                                               \
        for (; c_ < (full); c_ += 128) m0 = v_bf16_max_b(m0, LD(t, c_));               \
        for (; c_ < (cols); c_ += 128)                                                  \
            m0 = v_bf16_max_b(m0, v_bf16_sel_less_u16_b(LANES(c_), (unsigned short)(lim), \
                                                        LD(t, c_), low));
#endif

// One unit on half tensor t: cols columns, valid local columns lim, full = lim rounded down to 128.
#ifdef EXP_LOGQ
#define LOGQ_BIAS \
        float64 logq_bias = v_f32_mac_b(mx, -184.6649652337873f, (float64)16256.0f); \
        float128 logq_limit; logq_limit.v1 = mx + 64.0f; logq_limit.v2 = logq_limit.v1; \
        bfloat128 logq_ceiling = v_convert_f32_to_bf16_all_b(logq_limit)
#else
#define LOGQ_BIAS
#endif
#define UNIT(t, cols, full, lim) do {                                                   \
        bfloat128 m0 = low, m1 = low, m2 = low, m3 = low;                               \
        int c_ = 0;                                                                     \
        PASS1_SAMPLED(t, cols, full, lim);                                          \
        m0 = v_bf16_max_b(v_bf16_max_b(m0, m1), v_bf16_max_b(m2, m3));                  \
        float128 mf = v_convert_bf16_to_f32_all_b(m0);                                  \
        mx = max64_no_lut(v_f32_max_b(mf.v1, mf.v2));                                   \
        float128 mm; mm.v1 = mx; mm.v2 = mx;                                            \
        bfloat128 mb = v_convert_f32_to_bf16_all_b(mm); /* exact: mx is a BF16 value */ \
        LOGQ_BIAS; (void)mb;                                                           \
        float128 s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};                                \
        c_ = 0;                                                                         \
        PASS2_WIDE(t, full);                                                            \
        for (; c_ + 512 <= (full); c_ += 512) {                                         \
            bfloat128 x0 = LD(t, c_), x1 = LD(t, c_ + 128), x2 = LD(t, c_ + 256), x3 = LD(t, c_ + 384); \
            bfloat128 y0 = EXPV(x0), y1 = EXPV(x1), y2 = EXPV(x2), y3 = EXPV(x3);       \
            ST(t, c_, y0); ST(t, c_ + 128, y1); ST(t, c_ + 256, y2); ST(t, c_ + 384, y3); \
            ACC(s0, y0); ACC(s1, y1); ACC(s2, y2); ACC(s3, y3);                         \
        }                                                                               \
        for (; c_ < (full); c_ += 128) { bfloat128 y = EXPV(LD(t, c_)); ST(t, c_, y); ACC(s0, y); } \
        for (; c_ < (cols); c_ += 128) {                                                \
            bfloat128 y = v_bf16_sel_less_u16_b(LANES(c_), (unsigned short)(lim),        \
                                                EXPV(LD(t, c_)), (bfloat128)0);         \
            ST(t, c_, y); ACC(s0, y);                                                   \
        }                                                                               \
        lsum = ((s0.v1 + s0.v2) + (s1.v1 + s1.v2)) + ((s2.v1 + s2.v2) + (s3.v1 + s3.v2)); \
    } while (0)

void main(tensor stats, tensor sa, tensor sb, int valid_cols) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    const int ca = get_dim_size(sa, 0), cb = get_dim_size(sb, 0);
    const int la = valid_cols < ca ? valid_cols : ca;
    const int lb = valid_cols > ca ? valid_cols - ca : 0;
    const int fa = la / 128 * 128, fb = lb / 128 * 128;
    const bfloat128 low = (bfloat128)(-1.0e30f);
    for (int u = start[1]; u < end[1]; ++u) {
        const int row = u & 127;
        float64 mx, lsum;
        if (u < 128) UNIT(sa, ca, fa, la);
        else         UNIT(sb, cb, fb, lb);
        float64 l = sum64_no_lut(lsum);
        // stats row u: lane 0 = max, lane 1 = sum (the tensor is 2 columns wide).
        float64 out = v_f32_sel_eq_u32_b(read_lane_id_4b_b(), 0, mx, l);
        v_f32_st_tnsr((int5){0, u, 0, 0, 0}, stats, out);
    }
}
