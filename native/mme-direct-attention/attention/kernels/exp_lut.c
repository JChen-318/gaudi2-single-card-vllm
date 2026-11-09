// SPDX-License-Identifier: MIT
// Two-pass, numerically stable softmax numerator with the TPC BF16 exp LUT.
// Pass 1: exact max m of the valid BF16 scores of the row.
// Pass 2: p = exp_bf16(s - m) (BF16 subtract, LUT exp), masked, stored in place;
//         the FP32 row sum adds the stored BF16 values (what PV consumes).
// Same tensor signature as exp_sum.c; the bound tensor is not needed.
// REQUIRES the Gaudi2 special-function tables and LUT_FUNC128_BASE_ADDR set.
// LUT_FAST: mask only chunks that contain invalid columns, and accumulate the
// row sum with one BF16 MAC into FP32 (mac_acc32) instead of convert + 2 adds.
#include "reduce_no_lut.h"
#include "reduce_max_no_lut.h"

#define VALID(column) (read_lane_id_2b_b() + (column))
#define MASK(x, column, fill) v_bf16_sel_less_u16_b(VALID(column), (unsigned short)valid_cols, (x), (fill))
#define LOAD(column) v_bf16_ld_tnsr_b((int5){(column), row, 0, 0, 0}, scores)
#define EXP_RAW(x) exp_bf16(v_bf16_sub_b((x), mb))
#define EXP_CHUNK(x, column) MASK(EXP_RAW(x), (column), (bfloat128)0)
#define STORE(column, y) v_bf16_st_tnsr((int5){(column), row, 0, 0, 0}, probs, (y))
#ifdef LUT_FAST
#define ACC(acc, y) ((acc) = v_bf16_mac_acc32_b((y), (bfloat128)1.0f, (acc), 0, 1, 0))
typedef float128 acc_t;
#else
#define ACC(acc, y) do { float128 f_ = v_convert_bf16_to_f32_all_b(y); (acc) += f_.v1 + f_.v2; } while (0)
typedef float64 acc_t;
#endif

void main(tensor scores, tensor bounds, tensor probs, tensor sums, int valid_cols) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    const int cols = get_dim_size(scores, 0);
    const bfloat128 low = (bfloat128)(-1.0e30f);
#ifdef LUT_FAST
    // Columns [0, full) are all valid; only chunks at or beyond full need masking.
    const int full = valid_cols / 128 * 128;
#endif
    for (int row = start[1]; row < end[1]; ++row) {
        bfloat128 m0 = low, m1 = low, m2 = low, m3 = low;
        int col = 0;
#ifdef LUT_FAST
        for (; col + 512 <= full; col += 512) {
            m0 = v_bf16_max_b(m0, LOAD(col));
            m1 = v_bf16_max_b(m1, LOAD(col + 128));
            m2 = v_bf16_max_b(m2, LOAD(col + 256));
            m3 = v_bf16_max_b(m3, LOAD(col + 384));
        }
        for (; col < full; col += 128)
            m0 = v_bf16_max_b(m0, LOAD(col));
#else
        for (; col + 512 <= cols; col += 512) {
            m0 = v_bf16_max_b(m0, MASK(LOAD(col), col, low));
            m1 = v_bf16_max_b(m1, MASK(LOAD(col + 128), col + 128, low));
            m2 = v_bf16_max_b(m2, MASK(LOAD(col + 256), col + 256, low));
            m3 = v_bf16_max_b(m3, MASK(LOAD(col + 384), col + 384, low));
        }
#endif
        for (; col < cols; col += 128)
            m0 = v_bf16_max_b(m0, MASK(LOAD(col), col, low));
        m0 = v_bf16_max_b(v_bf16_max_b(m0, m1), v_bf16_max_b(m2, m3));
        float128 mf = v_convert_bf16_to_f32_all_b(m0);
        float64 mx = max64_no_lut(v_f32_max_b(mf.v1, mf.v2));
        float128 mm; mm.v1 = mx; mm.v2 = mx;
        bfloat128 mb = v_convert_f32_to_bf16_all_b(mm); // exact: mx is a BF16 value

        acc_t s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};
        col = 0;
#ifdef LUT_FAST
        for (; col + 512 <= full; col += 512) {
            // Scores and probabilities alias: all loads before any store.
            bfloat128 x0 = LOAD(col), x1 = LOAD(col + 128), x2 = LOAD(col + 256), x3 = LOAD(col + 384);
            bfloat128 y0 = EXP_RAW(x0), y1 = EXP_RAW(x1), y2 = EXP_RAW(x2), y3 = EXP_RAW(x3);
            STORE(col, y0); STORE(col + 128, y1); STORE(col + 256, y2); STORE(col + 384, y3);
            ACC(s0, y0); ACC(s1, y1); ACC(s2, y2); ACC(s3, y3);
        }
        for (; col < full; col += 128) {
            bfloat128 y = EXP_RAW(LOAD(col));
            STORE(col, y); ACC(s0, y);
        }
#else
        for (; col + 512 <= cols; col += 512) {
            // Scores and probabilities alias: all loads before any store.
            bfloat128 x0 = LOAD(col), x1 = LOAD(col + 128), x2 = LOAD(col + 256), x3 = LOAD(col + 384);
            bfloat128 y0 = EXP_CHUNK(x0, col), y1 = EXP_CHUNK(x1, col + 128);
            bfloat128 y2 = EXP_CHUNK(x2, col + 256), y3 = EXP_CHUNK(x3, col + 384);
            STORE(col, y0); STORE(col + 128, y1); STORE(col + 256, y2); STORE(col + 384, y3);
            ACC(s0, y0); ACC(s1, y1); ACC(s2, y2); ACC(s3, y3);
        }
#endif
        for (; col < cols; col += 128) {
            bfloat128 y = EXP_CHUNK(LOAD(col), col);
            STORE(col, y); ACC(s0, y);
        }
        int5 bp = {0, row, 0, 0, 0};
#ifdef LUT_FAST
        float64 t = ((s0.v1 + s0.v2) + (s1.v1 + s1.v2)) + ((s2.v1 + s2.v2) + (s3.v1 + s3.v2));
        v_f32_st_tnsr(bp, sums, sum64_no_lut(t));
#else
        v_f32_st_tnsr(bp, sums, sum64_no_lut((s0 + s1) + (s2 + s3)));
#endif
    }
}
