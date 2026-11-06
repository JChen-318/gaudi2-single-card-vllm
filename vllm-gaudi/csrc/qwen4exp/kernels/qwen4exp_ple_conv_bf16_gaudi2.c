// SPDX-License-Identifier: Apache-2.0
// H29: fused PLE dilated short conv + state update + residual, one launch.
//
// Torch (pre-H29) per forward, per request span: history gather / cat, a tap
// loop (``arange + add + advanced index`` before H24, contiguous slices
// after), an F32 accumulator, a BF16 round trip, SiLU, the two residual adds
// and the state write — and at decode T=4 that whole sequence ran four times
// (4 spans x ~10 ops). This kernel does the conv for every span, the state
// roll and the residual in one launch:
//
//   conv[j] = sum_k w[k] * full[j + dilation*k]      full = [state || x]
//   out[j]  = bf16(hidden + bf16(gated + bf16(silu(bf16(conv[j])))))
//   next[r] = full[length + r]
//
// Rounding matches ``_conv_per_request`` / ``apply_ple`` exactly: the F32 tap
// accumulation is rounded to BF16 before SiLU, ``silu`` is the same HW
// sigmoid the GDN kernels use, and the two residual adds are separate BF16
// adds (each rounds, as the eager tensor adds do). Tap order is 0..K-1.
//
// Uniform span length is the fast-path contract (decode B x 1 token, single
// request prefill); the caller falls back to torch otherwise.
//
// Layouts (TPC dim 0 = torch's contiguous last dim):
//   x, gated, hidden, out  BF16 [T, C]     -> (C, T)       T = spans*length
//   state, next_state      BF16 [S, W, C]  -> (C, W, S)    W = (K-1)*dilation
//   weight                 BF16 [K, C]     -> (C, K)
// Index space (C/128, ceil(length/64), spans). The state uses vLLM's default
// SD conv-state orientation ((window, channels) per slot, the shape
// ``get_state_shape`` announces and the GDN conv pool uses), so the
// 128-wide vector runs along channels.
#include "qwen4exp_hc_common.h"

//: Rows per index-space tile (dim 1). Only the tile that owns them runs the
//: first ``window`` rows (whose taps reach into the state) and the last tile
//: publishes the next state, so the bulk loop has no per-row branch.
#define Q4_PLE_CONV_TILE 64

//: One output row: conv -> BF16 -> SiLU -> + gated (BF16 add) -> + hidden
//: (BF16 add) -> out. ``row`` is the token row inside the [T, C] planes,
//: ``col`` the 128-channel tile, ``acc`` the F32 tap accumulator.
#define Q4_PLE_CONV_EMIT(COL, ROW, ACC)                                                     \
    do {                                                                                    \
        const int5 _o = {(COL), (ROW), 0, 0, 0};                                            \
        const float128 _conv = v_convert_bf16_to_f32_all_b(                                 \
            v_convert_f32_to_bf16_all_b(ACC));                                              \
        float128 _act;                                                                       \
        _act.v1 = _conv.v1 * qwen4exp_hc_sigmoid(_conv.v1);                                  \
        _act.v2 = _conv.v2 * qwen4exp_hc_sigmoid(_conv.v2);                                  \
        const float128 _g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_o, gated));        \
        const float128 _h = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_o, hidden));       \
        float128 _sum;                                                                       \
        _sum.v1 = _act.v1 + _g.v1;                                                           \
        _sum.v2 = _act.v2 + _g.v2;                                                           \
        const float128 _mixed = v_convert_bf16_to_f32_all_b(                                 \
            v_convert_f32_to_bf16_all_b(_sum));                                              \
        float128 _out;                                                                       \
        _out.v1 = _h.v1 + _mixed.v1;                                                         \
        _out.v2 = _h.v2 + _mixed.v2;                                                         \
        v_bf16_st_tnsr(_o, out, v_convert_f32_to_bf16_all_b(_out));                           \
    } while (0)

void main(tensor x, tensor state, tensor weight, tensor gated, tensor hidden,
          tensor out, tensor next_state) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int channels = get_dim_size(x, 0);       // C
    const int tokens = get_dim_size(x, 1);         // spans * length
    const int window = get_dim_size(state, 1);     // (K-1) * dilation
    const int spans = get_dim_size(state, 2);      // S
    const int taps = get_dim_size(weight, 1);      // K
    const int length = tokens / spans;
    const int dilation = window / (taps - 1);
    const int tiles = (length + Q4_PLE_CONV_TILE - 1) / Q4_PLE_CONV_TILE;
    //: Rows whose taps can still reach into the state.
    const int window_rows = length < window ? length : window;
    for (int span = start[2]; span < end[2]; ++span) {
        const int row_base = span * length;
        for (int block = start[0]; block < end[0]; ++block) {
            const int col = block * 128;
            if (col >= channels) break;
            for (int tile = start[1]; tile < end[1]; ++tile) {
                const int tile_lo = tile * Q4_PLE_CONV_TILE;
                int tile_hi = tile_lo + Q4_PLE_CONV_TILE;
                if (tile_hi > length) tile_hi = length;
                int row = tile_lo;
                // Rows whose tap window straddles the state boundary.
                while (row < tile_hi && row < window_rows) {
                    float128 acc = {0};
                    for (int k = 0; k < taps; ++k) {
                        const int src = row + dilation * k;
                        const int5 wc = {col, k, 0, 0, 0};
                        const float128 w = v_convert_bf16_to_f32_all_b(
                            v_bf16_ld_tnsr_b(wc, weight));
                        float128 v;
                        if (src < window) {
                            // state is torch [S, W, C] (SD, vLLM's default
                            // conv-state orientation): TPC (C, W, S).
                            const int5 sc = {col, src, span, 0, 0};
                            v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(sc, state));
                        } else {
                            const int5 xc = {col, row_base + src - window, 0, 0, 0};
                            v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, x));
                        }
                        acc.v1 += v.v1 * w.v1;
                        acc.v2 += v.v2 * w.v2;
                    }
                    Q4_PLE_CONV_EMIT(col, row_base + row, acc);
                    ++row;
                }
                // Bulk rows: every tap comes from this span's own tokens.
                for (; row < tile_hi; ++row) {
                    float128 acc = {0};
                    for (int k = 0; k < taps; ++k) {
                        const int5 wc = {col, k, 0, 0, 0};
                        const int5 xc = {col, row_base + row + dilation * k - window, 0, 0, 0};
                        const float128 w = v_convert_bf16_to_f32_all_b(
                            v_bf16_ld_tnsr_b(wc, weight));
                        const float128 v = v_convert_bf16_to_f32_all_b(
                            v_bf16_ld_tnsr_b(xc, x));
                        acc.v1 += v.v1 * w.v1;
                        acc.v2 += v.v2 * w.v2;
                    }
                    Q4_PLE_CONV_EMIT(col, row_base + row, acc);
                }
                // Roll the window: next[r] = full[length + r]. The reads are
                // staged in registers before any store, so an aliased
                // state/next_state buffer would still be read-before-write.
                if (tile == tiles - 1) {
                    for (int r = 0; r < window; ++r) {
                        const int src = length + r;
                        const int5 nc = {col, r, span, 0, 0};
                        bfloat128 value;
                        if (src < window) {
                            const int5 sc = {col, src, span, 0, 0};
                            value = v_bf16_ld_tnsr_b(sc, state);
                        } else {
                            const int5 xc = {col, row_base + src - window, 0, 0, 0};
                            value = v_bf16_ld_tnsr_b(xc, x);
                        }
                        v_bf16_st_tnsr(nc, next_state, value);
                    }
                }
            }
        }
    }
}
