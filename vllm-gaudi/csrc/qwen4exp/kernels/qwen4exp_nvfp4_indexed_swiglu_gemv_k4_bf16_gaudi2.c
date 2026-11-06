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

// H31: one row of one 512-value tile in nibble-major order. u16 lane i of the 256 packed bytes
// holds the four codes of k = 4i..4i+3 (byte 2i: 4i low nibble, 4i+1 high; byte 2i+1: 4i+2,
// 4i+3), and all four share the block scale of k / 16 = i / 4. The caller passes x permuted so
// that x_j lane i is x[4i + j] (x_perm[tile * 512 + j * 128 + i]). 21 vector instructions per
// row and tile instead of 33: no per-part dual-group move, nibble shuffle or scale pick.
#define ROW_TILE(ACC, PACKED, BITS, DG, X0, X1, X2, X3) { \
    const ushort128 p16 = *((ushort128*)&PACKED); \
    const ushort128 c0 = v_u16_and_b(p16, 15); \
    const ushort128 c1 = v_u16_and_b(v_u16_shr_b(p16, 4), 15); \
    const ushort128 c2 = v_u16_and_b(v_u16_shr_b(p16, 8), 15); \
    const ushort128 c3 = v_u16_shr_b(p16, 12); \
    const uchar256 sbytes = v_u8_shuffle_b(NVFP4_BROADCAST_SCALE(BITS, DG), scale_idx4, 0, zero); \
    const bfloat128 sc = *((bfloat128*)&sbytes); \
    ACC = v_bf16_mac_acc32_b(v_bf16_mul_b(as_bfloat128(v_i16_lookup_1c(c0, 457, 0, (short128)0)), sc), X0, ACC); \
    ACC = v_bf16_mac_acc32_b(v_bf16_mul_b(as_bfloat128(v_i16_lookup_1c(c1, 457, 0, (short128)0)), sc), X1, ACC); \
    ACC = v_bf16_mac_acc32_b(v_bf16_mul_b(as_bfloat128(v_i16_lookup_1c(c2, 457, 0, (short128)0)), sc), X2, ACC); \
    ACC = v_bf16_mac_acc32_b(v_bf16_mul_b(as_bfloat128(v_i16_lookup_1c(c3, 457, 0, (short128)0)), sc), X3, ACC); \
}

#define PASS_TILE(PK, SC, A0, A1, A2, A3) { \
    const uchar256 w0 = v_u8_ld_tnsr_b((int5){tile * 256, r0, expert, 0, 0}, PK); \
    const uchar256 w1 = v_u8_ld_tnsr_b((int5){tile * 256, r1, expert, 0, 0}, PK); \
    const uchar256 w2 = v_u8_ld_tnsr_b((int5){tile * 256, r2, expert, 0, 0}, PK); \
    const uchar256 w3 = v_u8_ld_tnsr_b((int5){tile * 256, r3, expert, 0, 0}, PK); \
    const uchar256 q0 = v_u8_ld_tnsr_b((int5){tile * 32, r0, expert, 0, 0}, SC); \
    const uchar256 q1 = v_u8_ld_tnsr_b((int5){tile * 32, r1, expert, 0, 0}, SC); \
    const uchar256 q2 = v_u8_ld_tnsr_b((int5){tile * 32, r2, expert, 0, 0}, SC); \
    const uchar256 q3 = v_u8_ld_tnsr_b((int5){tile * 32, r3, expert, 0, 0}, SC); \
    const ushort128 bits = nvfp4_e4m3_vec_bits_packed( \
        nvfp4_pack_scale_rows(q0, q1, q2, q3), spread_idx); \
    const bfloat128 x0 = x_local[tile * 4 + 0]; \
    const bfloat128 x1 = x_local[tile * 4 + 1]; \
    const bfloat128 x2 = x_local[tile * 4 + 2]; \
    const bfloat128 x3 = x_local[tile * 4 + 3]; \
    ROW_TILE(A0, w0, bits, 0, x0, x1, x2, x3) \
    ROW_TILE(A1, w1, bits, 1, x0, x1, x2, x3) \
    ROW_TILE(A2, w2, bits, 2, x0, x1, x2, x3) \
    ROW_TILE(A3, w3, bits, 3, x0, x1, x2, x3) \
}

// H31: the activation row lives in local memory. At T = 1 every member of every TPC read the same
// 5 KB of x from global memory, four loads per tile: 26 of 84 us per call went to that one hot row.
// The row is copied once per slot (whenever x_row changes) and PASS_TILE reads it locally.
#define X_LOCAL_TILES 5
__local__ bfloat128 x_local[X_LOCAL_TILES * 4];

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
    // H31: byte b of dual group g reads the u16 scale lane 8g + ((b & 63) >> 3) of row DG's
    // broadcast: lane i = 32g + ((b & 63) >> 1) gets the scale of block i / 4.
    const uchar256 scale_idx4 = v_u8_or_b(v_u8_or_b(
        v_u8_shl_b(v_u8_add_b(v_u8_shl_b(v_u8_shr_b(lanes, 6), 3), v_u8_shr_b(v_u8_and_b(lanes, 63), 3)), 1),
        v_u8_and_b(lanes, 1)), 0x80);
    const uchar256 spread_idx = nvfp4_e4m3_spread_index_packed();
    // H31 layout: K must be whole 512-value tiles (2560 for this model); the host glue checks it.
    const int full_tiles = k_total / K_PER_TILE;
    (void)packed_cols;
    int local_row = -1;
    for (int slot = start[1]; slot < end[1]; ++slot) {
        int x_row;
        if (x_rows == slots) {
            x_row = slot;
        } else if (x_rows == 1) {
            x_row = 0;
        } else {
            x_row = slot / slots_per_token;
        }
        if (x_row != local_row) {
            for (int c = 0; c < full_tiles * 4; ++c)
                x_local[c] = v_bf16_ld_tnsr_b((int5){c * 128, x_row, 0, 0, 0}, x);
            local_row = x_row;
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
            // Rows n0..n3 are contiguous in the output: one partial store of up to four lanes.
            const int store_last = (rows_total - n0 < ROWS_PER_MEMBER ? rows_total - n0 : ROWS_PER_MEMBER) - 1;
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
                    PASS_TILE(gate_packed, gate_scales, gacc0, gacc1, gacc2, gacc3)
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
                    PASS_TILE(up_packed, up_scales, uacc0, uacc1, uacc2, uacc3)
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
                // H31: the four rows' epilogues run as one vector chain (lanes 0..3), one basic block and
                // one store. As four guarded per-row blocks each paid the whole reduce -> sigmoid -> round
                // latency chain alone: ~1,100 of the member's ~2,160 bundles were that, 2/3 of them NOPs.
                // Every lane does exactly the per-row arithmetic, so the bits are unchanged.
                float64 packed_ubf = 0;
                packed_ubf = v_f32_mov_vb(
                    moe_bf16_round64(v_broadcast_element_f32(
                        v_f32_reduce_add(uacc0.v1 + uacc0.v2) * s2u, 0)),
                    0, packed_ubf, v_i32_cmp_eq_b(lane_id, 0), 0);
                packed_ubf = v_f32_mov_vb(
                    moe_bf16_round64(v_broadcast_element_f32(
                        v_f32_reduce_add(uacc1.v1 + uacc1.v2) * s2u, 0)),
                    0, packed_ubf, v_i32_cmp_eq_b(lane_id, 1), 0);
                packed_ubf = v_f32_mov_vb(
                    moe_bf16_round64(v_broadcast_element_f32(
                        v_f32_reduce_add(uacc2.v1 + uacc2.v2) * s2u, 0)),
                    0, packed_ubf, v_i32_cmp_eq_b(lane_id, 2), 0);
                packed_ubf = v_f32_mov_vb(
                    moe_bf16_round64(v_broadcast_element_f32(
                        v_f32_reduce_add(uacc3.v1 + uacc3.v2) * s2u, 0)),
                    0, packed_ubf, v_i32_cmp_eq_b(lane_id, 3), 0);
                // Lanes 4.. are silu(0) * 0 = +0 and are not stored.
                const float64 silu_bf = moe_bf16_round64(packed_gbf * moe_sigmoid64(packed_gbf));
                float128 wide;
                wide.v1 = silu_bf * packed_ubf;
                wide.v2 = 0;
                v_bf16_st_tnsr_partial((int5){n0, slot, 0, 0, 0}, output,
                                       v_convert_f32_to_bf16_all_b(wide, SW_LINEAR), store_last, 0);
            } else {
                const bfloat128 zero_out = v_convert_f32_to_bf16_all_b(
                    (float128){0}, SW_LINEAR);
                v_bf16_st_tnsr_partial((int5){n0, slot, 0, 0, 0}, output, zero_out, store_last, 0);
            }
        }
    }
}

#undef ROW_TILE
#undef PASS_TILE
