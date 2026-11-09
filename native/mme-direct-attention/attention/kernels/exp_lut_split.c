// SPDX-License-Identifier: MIT
// exp_lut2 for key-split scores: S is stored as two contiguous halves
// A = S[:, 0:first) and B = S[:, first:keys) (one per MME). Pass 1 takes the
// exact row max over the valid columns of both halves; pass 2 writes
// p = exp_bf16(s - m) in place into both halves and the FP32 row sum.
// Column c of B is global column first + c; valid_cols is global.
// REQUIRES the Gaudi2 special-function tables and LUT_FUNC128_BASE_ADDR set.
#include "reduce_no_lut.h"
#include "reduce_max_no_lut.h"

#define LANES(column) (read_lane_id_2b_b() + (column))
#define LD(t, column) v_bf16_ld_tnsr_b((int5){(column), row, 0, 0, 0}, (t))
#define ST(t, column, y) v_bf16_st_tnsr((int5){(column), row, 0, 0, 0}, (t), (y))
#define EXPV(x) exp_bf16(v_bf16_sub_b((x), mb))
#define ACC(acc, y) ((acc) = v_bf16_mac_acc32_b((y), (bfloat128)1.0f, (acc), 0, 1, 0))

// Row max over one half; cols columns, the first `full` of which are valid,
// lim = number of valid columns in this half (local index).
#define HALF_MAX(t, cols, full, lim) do {                                              \
        int c_ = 0;                                                                    \
        for (; c_ + 512 <= (full); c_ += 512) {                                        \
            m0 = v_bf16_max_b(m0, LD(t, c_));       m1 = v_bf16_max_b(m1, LD(t, c_ + 128)); \
            m2 = v_bf16_max_b(m2, LD(t, c_ + 256)); m3 = v_bf16_max_b(m3, LD(t, c_ + 384)); \
        }                                                                              \
        for (; c_ < (full); c_ += 128) m0 = v_bf16_max_b(m0, LD(t, c_));              \
        for (; c_ < (cols); c_ += 128)                                                 \
            m0 = v_bf16_max_b(m0, v_bf16_sel_less_u16_b(LANES(c_), (unsigned short)(lim), \
                                                        LD(t, c_), low));              \
    } while (0)

#define HALF_EXP(t, cols, full, lim) do {                                              \
        int c_ = 0;                                                                    \
        for (; c_ + 512 <= (full); c_ += 512) {                                        \
            bfloat128 x0 = LD(t, c_), x1 = LD(t, c_ + 128), x2 = LD(t, c_ + 256), x3 = LD(t, c_ + 384); \
            bfloat128 y0 = EXPV(x0), y1 = EXPV(x1), y2 = EXPV(x2), y3 = EXPV(x3);      \
            ST(t, c_, y0); ST(t, c_ + 128, y1); ST(t, c_ + 256, y2); ST(t, c_ + 384, y3); \
            ACC(s0, y0); ACC(s1, y1); ACC(s2, y2); ACC(s3, y3);                        \
        }                                                                              \
        for (; c_ < (full); c_ += 128) { bfloat128 y = EXPV(LD(t, c_)); ST(t, c_, y); ACC(s0, y); } \
        for (; c_ < (cols); c_ += 128) {                                               \
            bfloat128 y = v_bf16_sel_less_u16_b(LANES(c_), (unsigned short)(lim),       \
                                                EXPV(LD(t, c_)), (bfloat128)0);        \
            ST(t, c_, y); ACC(s0, y);                                                  \
        }                                                                              \
    } while (0)

void main(tensor sa, tensor sb, tensor sums, int valid_cols) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    const int ca = get_dim_size(sa, 0), cb = get_dim_size(sb, 0);
    const int la = valid_cols < ca ? valid_cols : ca;            // valid columns in A
    const int lb = valid_cols > ca ? valid_cols - ca : 0;        // valid columns in B
    const int fa = la / 128 * 128, fb = lb / 128 * 128;
    const bfloat128 low = (bfloat128)(-1.0e30f);
    for (int row = start[1]; row < end[1]; ++row) {
        bfloat128 m0 = low, m1 = low, m2 = low, m3 = low;
        HALF_MAX(sa, ca, fa, la);
        HALF_MAX(sb, cb, fb, lb);
        m0 = v_bf16_max_b(v_bf16_max_b(m0, m1), v_bf16_max_b(m2, m3));
        float128 mf = v_convert_bf16_to_f32_all_b(m0);
        float64 mx = max64_no_lut(v_f32_max_b(mf.v1, mf.v2));
        float128 mm; mm.v1 = mx; mm.v2 = mx;
        bfloat128 mb = v_convert_f32_to_bf16_all_b(mm); // exact: mx is a BF16 value
        float128 s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};
        HALF_EXP(sa, ca, fa, la);
        HALF_EXP(sb, cb, fb, lb);
        float64 t = ((s0.v1 + s0.v2) + (s1.v1 + s1.v2)) + ((s2.v1 + s2.v2) + (s3.v1 + s3.v2));
        v_f32_st_tnsr((int5){0, row, 0, 0, 0}, sums, sum64_no_lut(t));
    }
}
