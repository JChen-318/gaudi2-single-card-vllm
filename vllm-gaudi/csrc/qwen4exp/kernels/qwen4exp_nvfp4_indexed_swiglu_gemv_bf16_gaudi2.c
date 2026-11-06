// SPDX-License-Identifier: Apache-2.0
// H15 Optimization 1: fused routed NVFP4 gate+up GEMV with in-kernel SwiGLU.
//
//   act[slot, n] = silu(gate_out[slot, n]) * up_out[slot, n]
//   gate_out = scale2_gate[e] * sum_k bf16(e2m1*e4m3)[gate, n, k] * x[x_row, k]
//   up_out   = scale2_up[e]   * sum_k bf16(e2m1*e4m3)[up,   n, k] * x[x_row, k]
//
// Same index space ([ceil(N/4), S]) and 4-row-per-member structure as the
// standalone indexed GEMV (qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c).
// Within a member the K loop runs as a gate pass followed by an up pass:
// each pass loads an activation tile once per 128-element part and shares
// it across its four rows, exactly like the standalone kernel. The gate
// pass reduces its four F32 accumulators to four BF16 scalars, packed into
// lanes 0..3 of one vector, and the up pass then accumulates and combines.
//
// Why not eight simultaneous accumulators (4 gate + 4 up in one K loop)?
// That form was built and measured first: it keeps 8 F32 accumulators and
// 16 weight/scale tile loads live at once, the TPC compiler spills vector
// registers (its own backend reports the spill budget exceeded for the
// local-cached-x variant; the interleaved form has ~50% more local stores
// in its assembly), and it ran slower than the two standalone GEMVs. The
// two-pass form has the standalone kernel's register footprint per pass,
// one launch instead of two, and neither gate_out, up_out, nor the
// silu/mul intermediates are materialized. x is re-read in the up pass
// (a [T, 2560] BF16 plane, cache-resident at decode sizes).
//
// Bit-exactness contract vs the exact (gate GEMV; up GEMV; silu; mul):
//   * each accumulator follows the standalone kernel's MAC order exactly
//     (K ascending, F32 lane tree via v_bf16_mac_acc32_b, then scale2);
//   * the F32 dot is rounded to BF16 and back (the standalone GEMV's store),
//     silu is evaluated in F32 (accurate exp + Newton reciprocal -- the
//     hardware v_sigmoid_f32 form and Habana's own BF16 silu kernel both
//     deviate from exact silu by ~1 BF16 ulp on ~1/3 of elements) on that
//     BF16 value and itself rounded to BF16 and back, and the product with
//     the BF16-rounded up value is rounded to BF16 for the store.
//     bench_moe_h15.py verifies 0 differing elements against the exact
//     chain (CPU silu on the standalone GEMV outputs) at T=1 and T=4.
//   * invalid expert (-1/out of range): both scale2 read as 0, dots are 0,
//     silu(0)*0 == 0, matching silu(0-row) * 0-row.
#include "qwen4exp_nvfp4_common.h"

#define ROWS_PER_MEMBER 4
#define K_PER_TILE 512
#define K_PER_PART 128

#define GEMV_ROW_PART(ACC, PACKED, RSCALE, PICK, XV, PART) { \
    const uchar256 packed_part = v_u8_mov_dual_group_all_b( \
        PACKED, 0xffffffff, PART, PART, PART, PART, MkWrA(3, 3, 3, 3), zero); \
    const uchar256 repeated = v_u8_shuffle_b( \
        packed_part, packed_indices, 0, zero); \
    const ushort128 nibble = v_u16_and_b( \
        v_u16_shr_b(*((ushort128*)&repeated), nibble_shifts), 15); \
    const bfloat128 e2 = as_bfloat128( \
        v_i16_lookup_1c(nibble, 457, 0, (short128)0)); \
    const uchar256 picked_bytes = v_u8_shuffle_b( \
        RSCALE, PICK, 0, zero); \
    const bfloat128 weight = v_bf16_mul_b(e2, *((bfloat128*)&picked_bytes)); \
    ACC = v_bf16_mac_acc32_b(weight, XV, ACC); \
}

// One 512-element tile of one pass: the pass's eight weight/scale loads
// first, one E4M3 decode per row, then per 128-part one shared activation
// load MACed into the pass's four accumulators.
#define GATE_TILE(GUARDED) { \
    const uchar256 gp0 = v_u8_ld_tnsr_b((int5){tile * 256, r0, expert, 0, 0}, gate_packed); \
    const uchar256 gp1 = v_u8_ld_tnsr_b((int5){tile * 256, r1, expert, 0, 0}, gate_packed); \
    const uchar256 gp2 = v_u8_ld_tnsr_b((int5){tile * 256, r2, expert, 0, 0}, gate_packed); \
    const uchar256 gp3 = v_u8_ld_tnsr_b((int5){tile * 256, r3, expert, 0, 0}, gate_packed); \
    const uchar256 gq0 = v_u8_ld_tnsr_b((int5){tile * 32, r0, expert, 0, 0}, gate_scales); \
    const uchar256 gq1 = v_u8_ld_tnsr_b((int5){tile * 32, r1, expert, 0, 0}, gate_scales); \
    const uchar256 gq2 = v_u8_ld_tnsr_b((int5){tile * 32, r2, expert, 0, 0}, gate_scales); \
    const uchar256 gq3 = v_u8_ld_tnsr_b((int5){tile * 32, r3, expert, 0, 0}, gate_scales); \
    const ushort128 gscale_bits = nvfp4_e4m3_vec_bits_packed( \
        nvfp4_pack_scale_rows(gq0, gq1, gq2, gq3), spread_idx); \
    const uchar256 gr0s = NVFP4_BROADCAST_SCALE(gscale_bits, 0); \
    const uchar256 gr1s = NVFP4_BROADCAST_SCALE(gscale_bits, 1); \
    const uchar256 gr2s = NVFP4_BROADCAST_SCALE(gscale_bits, 2); \
    const uchar256 gr3s = NVFP4_BROADCAST_SCALE(gscale_bits, 3); \
    if (!(GUARDED) || k_base + 0 * K_PER_PART < k_total) { \
        const bfloat128 x0 = v_bf16_ld_tnsr_b((int5){k_base, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(gacc0, gp0, gr0s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(gacc1, gp1, gr1s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(gacc2, gp2, gr2s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(gacc3, gp3, gr3s, pick_idx0, x0, 0) \
    } \
    if (!(GUARDED) || k_base + 1 * K_PER_PART < k_total) { \
        const bfloat128 x1 = v_bf16_ld_tnsr_b((int5){k1, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(gacc0, gp0, gr0s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(gacc1, gp1, gr1s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(gacc2, gp2, gr2s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(gacc3, gp3, gr3s, pick_idx1, x1, 1) \
    } \
    if (!(GUARDED) || k_base + 2 * K_PER_PART < k_total) { \
        const bfloat128 x2 = v_bf16_ld_tnsr_b((int5){k2, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(gacc0, gp0, gr0s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(gacc1, gp1, gr1s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(gacc2, gp2, gr2s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(gacc3, gp3, gr3s, pick_idx2, x2, 2) \
    } \
    if (!(GUARDED) || k_base + 3 * K_PER_PART < k_total) { \
        const bfloat128 x3 = v_bf16_ld_tnsr_b((int5){k3, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(gacc0, gp0, gr0s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(gacc1, gp1, gr1s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(gacc2, gp2, gr2s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(gacc3, gp3, gr3s, pick_idx3, x3, 3) \
    } \
}

#define UP_TILE(GUARDED) { \
    const uchar256 up0 = v_u8_ld_tnsr_b((int5){tile * 256, r0, expert, 0, 0}, up_packed); \
    const uchar256 up1 = v_u8_ld_tnsr_b((int5){tile * 256, r1, expert, 0, 0}, up_packed); \
    const uchar256 up2 = v_u8_ld_tnsr_b((int5){tile * 256, r2, expert, 0, 0}, up_packed); \
    const uchar256 up3 = v_u8_ld_tnsr_b((int5){tile * 256, r3, expert, 0, 0}, up_packed); \
    const uchar256 uq0 = v_u8_ld_tnsr_b((int5){tile * 32, r0, expert, 0, 0}, up_scales); \
    const uchar256 uq1 = v_u8_ld_tnsr_b((int5){tile * 32, r1, expert, 0, 0}, up_scales); \
    const uchar256 uq2 = v_u8_ld_tnsr_b((int5){tile * 32, r2, expert, 0, 0}, up_scales); \
    const uchar256 uq3 = v_u8_ld_tnsr_b((int5){tile * 32, r3, expert, 0, 0}, up_scales); \
    const ushort128 uscale_bits = nvfp4_e4m3_vec_bits_packed( \
        nvfp4_pack_scale_rows(uq0, uq1, uq2, uq3), spread_idx); \
    const uchar256 ur0s = NVFP4_BROADCAST_SCALE(uscale_bits, 0); \
    const uchar256 ur1s = NVFP4_BROADCAST_SCALE(uscale_bits, 1); \
    const uchar256 ur2s = NVFP4_BROADCAST_SCALE(uscale_bits, 2); \
    const uchar256 ur3s = NVFP4_BROADCAST_SCALE(uscale_bits, 3); \
    if (!(GUARDED) || k_base + 0 * K_PER_PART < k_total) { \
        const bfloat128 x0 = v_bf16_ld_tnsr_b((int5){k_base, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(uacc0, up0, ur0s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(uacc1, up1, ur1s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(uacc2, up2, ur2s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(uacc3, up3, ur3s, pick_idx0, x0, 0) \
    } \
    if (!(GUARDED) || k_base + 1 * K_PER_PART < k_total) { \
        const bfloat128 x1 = v_bf16_ld_tnsr_b((int5){k1, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(uacc0, up0, ur0s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(uacc1, up1, ur1s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(uacc2, up2, ur2s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(uacc3, up3, ur3s, pick_idx1, x1, 1) \
    } \
    if (!(GUARDED) || k_base + 2 * K_PER_PART < k_total) { \
        const bfloat128 x2 = v_bf16_ld_tnsr_b((int5){k2, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(uacc0, up0, ur0s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(uacc1, up1, ur1s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(uacc2, up2, ur2s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(uacc3, up3, ur3s, pick_idx2, x2, 2) \
    } \
    if (!(GUARDED) || k_base + 3 * K_PER_PART < k_total) { \
        const bfloat128 x3 = v_bf16_ld_tnsr_b((int5){k3, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(uacc0, up0, ur0s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(uacc1, up1, ur1s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(uacc2, up2, ur2s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(uacc3, up3, ur3s, pick_idx3, x3, 3) \
    } \
}

// Accurate sigmoid via exp + Newton-refined reciprocal. The hardware
// v_sigmoid_f32 approximation differs from torch's F.silu by enough F32
// ULPs to flip ~1/3 of the BF16 outputs at rounding boundaries (measured
// in bench_moe_h15.py); the exp form lands inside the BF16 rounding.
static inline float64 moe_sigmoid64(float64 x) {
    const float64 e = v_exp_f32(-x);
    const float64 denom = 1.0f + e;
    float64 inv = v_reciprocal_f32(denom);
    inv = inv * (2.0f - denom * inv);
    return inv;
}

// Round an F32 vector (all lanes the same scalar) to BF16 and back.
static inline float64 moe_bf16_round64(float64 v) {
    float128 wide;
    wide.v1 = v;
    wide.v2 = v;
    const bfloat128 bf = v_convert_f32_to_bf16_all_b(wide);
    return v_convert_bf16_to_f32_all_b(bf).v1;
}

#define SWIGLU_STORE_ROW(PACKED_GBF, LANE, UACC, ROW) { \
    const float64 gate_bf = v_broadcast_element_f32(PACKED_GBF, LANE); \
    const float64 up_bf = moe_bf16_round64( \
        v_broadcast_element_f32(v_f32_reduce_add(UACC.v1 + UACC.v2) * s2u, 0)); \
    const float64 silu_bf = moe_bf16_round64(gate_bf * moe_sigmoid64(gate_bf)); \
    const float64 prod = silu_bf * up_bf; \
    float64 lanes32 = 0; \
    lanes32 = v_f32_sel_eq_u32_b(V_LANE_ID_32, 0u, prod, lanes32); \
    float128 wide; \
    wide.v1 = lanes32; \
    wide.v2 = 0; \
    v_bf16_st_tnsr_partial((int5){ROW, slot, 0, 0, 0}, output, \
                           v_convert_f32_to_bf16_all_b(wide, SW_LINEAR), 0, 0); \
}

void main(tensor expert_ids, tensor gate_packed, tensor gate_scales, tensor gate_scale2,
          tensor up_packed, tensor up_scales, tensor up_scale2, tensor x, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int experts = get_dim_size(gate_packed, 2);
    const int k_total = get_dim_size(x, 0);
    const int x_rows = get_dim_size(x, 1);
    const int slots = get_dim_size(output, 1);
    const int packed_cols = get_dim_size(gate_packed, 0);
    const int rows_total = get_dim_size(output, 0);
    const int slots_per_token =
        (x_rows > 1 && x_rows < slots) ? slots / x_rows : 1;
    const uchar256 lanes = (uchar256)V_LANE_ID_8;
    const uchar256 zero = 0;
    const uchar256 packed_indices = v_u8_or_b(v_u8_shr_b(lanes, 2), 0x80);
    const short128 nibble_shifts = (short128)v_u16_shl_b(
        v_u16_and_b((ushort128)V_LANE_ID_16, 1), 2);
    const uchar256 pick_base = v_u8_add_b(
        v_u8_shl_b(v_u8_shr_b(lanes, 5), 1), v_u8_and_b(lanes, 1));
    const uchar256 pick_idx0 = v_u8_or_b(pick_base, 0x80);
    const uchar256 pick_idx1 = v_u8_or_b(v_u8_add_b(pick_base, 16), 0x80);
    const uchar256 pick_idx2 = v_u8_or_b(v_u8_add_b(pick_base, 32), 0x80);
    const uchar256 pick_idx3 = v_u8_or_b(v_u8_add_b(pick_base, 48), 0x80);
    // H21: byte spread pattern of the hardware E4M3 decode, built once.
    // H27: the packed four-row form, so one spread feeds a whole pass.
    const uchar256 spread_idx = nvfp4_e4m3_spread_index_packed();
    const int full_tiles = k_total / K_PER_TILE;
    const int tiles = (packed_cols + 255) / 256;
    for (int slot = start[1]; slot < end[1]; ++slot) {
        int x_row;
        if (x_rows == slots) {
            x_row = slot;
        } else if (x_rows == 1) {
            x_row = 0;
        } else {
            x_row = slot / slots_per_token;
        }
        const int expert = s_i32_ld_g(gen_addr((int5){slot, 0, 0, 0, 0}, expert_ids));
        const int ok = expert >= 0 && expert < experts;
        const float s2g = ok ? s_f32_ld_g(gen_addr((int5){expert, 0, 0, 0, 0}, gate_scale2)) : 0.0f;
        const float s2u = ok ? s_f32_ld_g(gen_addr((int5){expert, 0, 0, 0, 0}, up_scale2)) : 0.0f;
        for (int group = start[0]; group < end[0]; ++group) {
            const int n0 = group * ROWS_PER_MEMBER;
            const int n1 = n0 + 1;
            const int n2 = n0 + 2;
            const int n3 = n0 + 3;
            const int r0 = n0 < rows_total ? n0 : rows_total - 1;
            const int r1 = n1 < rows_total ? n1 : rows_total - 1;
            const int r2 = n2 < rows_total ? n2 : rows_total - 1;
            const int r3 = n3 < rows_total ? n3 : rows_total - 1;
            float128 gacc0 = {0};
            float128 gacc1 = {0};
            float128 gacc2 = {0};
            float128 gacc3 = {0};
            float128 uacc0 = {0};
            float128 uacc1 = {0};
            float128 uacc2 = {0};
            float128 uacc3 = {0};
            float64 packed_gbf = 0;
            if (ok) {
                for (int tile = 0; tile < full_tiles; ++tile) {
                    const int k_base = tile * K_PER_TILE;
                    const int k1 = k_base + 1 * K_PER_PART;
                    const int k2 = k_base + 2 * K_PER_PART;
                    const int k3 = k_base + 3 * K_PER_PART;
                    GATE_TILE(0)
                }
                for (int tile = full_tiles; tile < tiles; ++tile) {
                    const int k_base = tile * K_PER_TILE;
                    if (k_base >= k_total) break;
                    const int k1 = k_base + 1 * K_PER_PART < k_total
                        ? k_base + 1 * K_PER_PART : k_base;
                    const int k2 = k_base + 2 * K_PER_PART < k_total
                        ? k_base + 2 * K_PER_PART : k_base;
                    const int k3 = k_base + 3 * K_PER_PART < k_total
                        ? k_base + 3 * K_PER_PART : k_base;
                    GATE_TILE(1)
                }
                // Pack the four BF16 gate scalars into lanes 0..3 of one
                // vector: only one gate register stays live over the up pass
                // (four broadcast vectors measurably spill the accumulators).
                const int64 lane_id = (int64)V_LANE_ID_32;
                packed_gbf = v_f32_mov_vb(
                    moe_bf16_round64(v_broadcast_element_f32(
                        v_f32_reduce_add(gacc0.v1 + gacc0.v2) * s2g, 0)),
                    0, packed_gbf, v_i32_cmp_eq_b(lane_id, 0), 0);
                packed_gbf = v_f32_mov_vb(
                    moe_bf16_round64(v_broadcast_element_f32(
                        v_f32_reduce_add(gacc1.v1 + gacc1.v2) * s2g, 0)),
                    0, packed_gbf, v_i32_cmp_eq_b(lane_id, 1), 0);
                packed_gbf = v_f32_mov_vb(
                    moe_bf16_round64(v_broadcast_element_f32(
                        v_f32_reduce_add(gacc2.v1 + gacc2.v2) * s2g, 0)),
                    0, packed_gbf, v_i32_cmp_eq_b(lane_id, 2), 0);
                packed_gbf = v_f32_mov_vb(
                    moe_bf16_round64(v_broadcast_element_f32(
                        v_f32_reduce_add(gacc3.v1 + gacc3.v2) * s2g, 0)),
                    0, packed_gbf, v_i32_cmp_eq_b(lane_id, 3), 0);
                for (int tile = 0; tile < full_tiles; ++tile) {
                    const int k_base = tile * K_PER_TILE;
                    const int k1 = k_base + 1 * K_PER_PART;
                    const int k2 = k_base + 2 * K_PER_PART;
                    const int k3 = k_base + 3 * K_PER_PART;
                    UP_TILE(0)
                }
                for (int tile = full_tiles; tile < tiles; ++tile) {
                    const int k_base = tile * K_PER_TILE;
                    if (k_base >= k_total) break;
                    const int k1 = k_base + 1 * K_PER_PART < k_total
                        ? k_base + 1 * K_PER_PART : k_base;
                    const int k2 = k_base + 2 * K_PER_PART < k_total
                        ? k_base + 2 * K_PER_PART : k_base;
                    const int k3 = k_base + 3 * K_PER_PART < k_total
                        ? k_base + 3 * K_PER_PART : k_base;
                    UP_TILE(1)
                }
                // The SwiGLU epilogue belongs to this branch. It used to sit
                // after the if/else, merging packed_gbf/uacc through phi nodes
                // -- which let the TPC backend predicate it into the up K-tile
                // loop and spill 40 ld_l_v / 30 st_l_v per iteration there.
                // Keeping it inside the taken branch keeps it out of the loop
                // (probe: -81 instructions and -87 bundles in that block,
                // VLM traffic 101/72 -> 81/62 over the kernel) with no change
                // to the bits: an invalid expert produced silu(0)*0 == +0, and
                // the cold else branch stores +0 directly.
                if (n0 < rows_total) SWIGLU_STORE_ROW(packed_gbf, 0, uacc0, n0)
                if (n1 < rows_total) SWIGLU_STORE_ROW(packed_gbf, 1, uacc1, n1)
                if (n2 < rows_total) SWIGLU_STORE_ROW(packed_gbf, 2, uacc2, n2)
                if (n3 < rows_total) SWIGLU_STORE_ROW(packed_gbf, 3, uacc3, n3)
            } else {
                const bfloat128 zero_out = v_convert_f32_to_bf16_all_b(
                    (float128){0}, SW_LINEAR);
                if (n0 < rows_total)
                    v_bf16_st_tnsr_partial((int5){n0, slot, 0, 0, 0}, output,
                                           zero_out, 0, 0);
                if (n1 < rows_total)
                    v_bf16_st_tnsr_partial((int5){n1, slot, 0, 0, 0}, output,
                                           zero_out, 0, 0);
                if (n2 < rows_total)
                    v_bf16_st_tnsr_partial((int5){n2, slot, 0, 0, 0}, output,
                                           zero_out, 0, 0);
                if (n3 < rows_total)
                    v_bf16_st_tnsr_partial((int5){n3, slot, 0, 0, 0}, output,
                                           zero_out, 0, 0);
            }
        }
    }
}

#undef GEMV_ROW_PART
#undef GATE_TILE
#undef UP_TILE
#undef SWIGLU_STORE_ROW
