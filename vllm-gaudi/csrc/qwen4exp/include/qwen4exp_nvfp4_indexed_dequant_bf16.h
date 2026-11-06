// SPDX-License-Identifier: Apache-2.0
// NVFP4 (packed E2M1 + per-16 E4M3 block scales) indexed dequant to BF16.
// This kernel replaces the nvfp4_mul_bits bit construction with the BF16
// factor product (H8 does the same in the fused GEMV). A v_u16_shuffle_b of a
// computed table did not follow the lane overlay (scale 81 came back as entry
// 6), so the factors are evaluated per lane. scale2 is folded after the BF16
// cast (H6).
//
// H7 (cookbook mxk_deq structure):
//   (a) E2M1 decode via the hardware ROM lookup function 457 (the stock
//       MXFP4 cast's table) instead of a 16-entry v_u8_shuffle_b table.
//   (b) The E4M3 scale decode runs ONCE per tile (32 distinct scale bytes
//       cover the 512 values) and each 128-row part picks its BF16 scale
//       lanes with the proven v_u8_shuffle_b byte convention; previously the
//       full per-lane E4M3 decode ran inside every part (4x per tile).
//
// H21 replaces that integer E4M3 decode with the hardware FP8 conversion
// (convert.f8_143) -- see qwen4exp_nvfp4_common.h. It drops the scale decode
// from ~21 vector instructions per row and tile to 3.
//
// H27: four rows per member, the H8 structure the GEMV kernel already uses,
// plus ONE convert.f8_143 for the four rows' scales (see the four-row block in
// qwen4exp_nvfp4_common.h):
//   * the member owns four consecutive output rows, so its eight packed/scale
//     loads are issued before the first decode and overlap in flight instead
//     of one row's two loads per member;
//   * the four rows' 32 scale bytes are packed into the four dual groups of
//     one register (3 mov_dg) and decoded by a single spread + convert, so the
//     scale path costs 3 + 1 + 6 per tile instead of 4 x (1 + 6);
//   * the E2M1 path, the MAC order, the scale2 fold and the store pattern are
//     unchanged, so the output stays bit-identical (the decode of a scale byte
//     is the same function of the same byte in both forms).
// A trailing group shorter than four rows clamps its loads to the last row and
// guards every store, exactly like the GEMV kernel.
//
// The decode helpers themselves live in qwen4exp_nvfp4_common.h, shared with
// the decode GEMV (H8).
#include "qwen4exp_nvfp4_common.h"

// H27: ROWS_PER_MEMBER must match kRowsPerMember in the host glue.
#define ROWS_PER_MEMBER 4

// H7(a)+(b)+(c): E2M1 from the ROM lookup (457), scale already decoded once
// per tile and picked per part, and the pre-H6 signed-zero restoration is
// dropped because the in-kernel BF16 scale2 multiply flushes -0 to +0 anyway.
// DST is the row's 128-element write cursor (advanced by this macro).
#define DECODE_AND_STORE_PART(DST, PACKED, RSCALE, PICK, PART) { \
    const uchar256 packed_part = v_u8_mov_dual_group_all_b( \
        PACKED, 0xffffffff, PART, PART, PART, PART, \
        MkWrA(3, 3, 3, 3), zero); \
    const uchar256 repeated = v_u8_shuffle_b( \
        packed_part, packed_indices, 0, zero); \
    const ushort128 nibble = v_u16_and_b( \
        v_u16_shr_b(*((ushort128*)&repeated), nibble_shifts), 15); \
    const uchar256 picked_bytes = v_u8_shuffle_b( \
        RSCALE, PICK, 0, zero); \
    const bfloat128 e2 = as_bfloat128( \
        v_i16_lookup_1c(nibble, 457, 0, (short128)0)); \
    const bfloat128 product = v_bf16_mul_b(e2, *((bfloat128*)&picked_bytes)); \
    /* scale2 after the BF16 cast, same flush-to-+0 as the Python fold. \
       Do not lift the K check over this body. */ \
    const bfloat128 scaled = v_bf16_mul_b(product, factor); \
    if (DST[0] < k_total) v_bf16_st_tnsr(DST, output, scaled); \
    DST[0] += 128; \
}

// The four 128-element parts of one row (one 256-byte packed load covers the
// whole 512-element tile). DG is the row's dual group in the packed decode.
#define DECODE_AND_STORE_ROW(PACKED, ROW, DG) { \
    const uchar256 rscale = NVFP4_BROADCAST_SCALE(scale_bits, DG); \
    int5 row_dst = {tile * 512, ROW, slot, 0, 0}; \
    DECODE_AND_STORE_PART(row_dst, PACKED, rscale, pick_idx0, 0) \
    DECODE_AND_STORE_PART(row_dst, PACKED, rscale, pick_idx1, 1) \
    DECODE_AND_STORE_PART(row_dst, PACKED, rscale, pick_idx2, 2) \
    DECODE_AND_STORE_PART(row_dst, PACKED, rscale, pick_idx3, 3) \
}

void main(tensor expert_ids, tensor packed, tensor scales, tensor scale2, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int experts = get_dim_size(packed, 2);
    const int k_total = get_dim_size(output, 0);
    const int rows_total = get_dim_size(output, 1);
    const uchar256 lanes = (uchar256)V_LANE_ID_8;
    const uchar256 packed_indices = v_u8_or_b(v_u8_shr_b(lanes, 2), 0x80);
    // byte-lane b picks BF16 scale byte 2*(b>>5) + (b&1) (+16*PART, |0x80).
    const uchar256 pick_base = v_u8_add_b(
        v_u8_shl_b(v_u8_shr_b(lanes, 5), 1), v_u8_and_b(lanes, 1));
    const uchar256 pick_idx0 = v_u8_or_b(pick_base, 0x80);
    const uchar256 pick_idx1 = v_u8_or_b(v_u8_add_b(pick_base, 16), 0x80);
    const uchar256 pick_idx2 = v_u8_or_b(v_u8_add_b(pick_base, 32), 0x80);
    const uchar256 pick_idx3 = v_u8_or_b(v_u8_add_b(pick_base, 48), 0x80);
    const short128 nibble_shifts = (short128)v_u16_shl_b(
        v_u16_and_b((ushort128)V_LANE_ID_16, 1), 2);
    // H27: byte spread pattern of the four-row hardware E4M3 decode.
    const uchar256 spread_idx = nvfp4_e4m3_spread_index_packed();
    const uchar256 zero = 0;
    for (int slot = start[2]; slot < end[2]; ++slot) {
        const int expert = s_i32_ld_g(gen_addr((int5){slot, 0, 0, 0, 0}, expert_ids));
        const int ok = expert >= 0 && expert < experts;
        const float s2 = ok
            ? s_f32_ld_g(gen_addr((int5){expert, 0, 0, 0, 0}, scale2)) : 0.0f;
        float64 s2_lanes = s2;
        float128 s2_wide;
        s2_wide.v1 = s2_lanes;
        s2_wide.v2 = s2_lanes;
        // Torch casts F32 scale2 to BF16, then multiplies. SW_LINEAR is that cast.
        const bfloat128 factor = v_convert_f32_to_bf16_all_b(s2_wide, SW_LINEAR);
        for (int group = start[1]; group < end[1]; ++group) {
            // H27: the member owns four consecutive rows. Reads AND writes past
            // the last row are clamped to it: the four loads then return the
            // same bytes, so the four rows decode to identical BF16 bits and
            // the duplicated stores write the value row ROWS_TOTAL-1 already
            // has. That keeps the whole group branch-free (an `if (row <
            // rows_total)` per row made the compiler split the tile body into
            // eight basic blocks with a jump chain between them).
            const int n0 = group * ROWS_PER_MEMBER;
            const int r0 = n0 < rows_total ? n0 : rows_total - 1;
            const int r1 = n0 + 1 < rows_total ? n0 + 1 : rows_total - 1;
            const int r2 = n0 + 2 < rows_total ? n0 + 2 : rows_total - 1;
            const int r3 = n0 + 3 < rows_total ? n0 + 3 : rows_total - 1;
            if (!ok) {
                // Cold path: -1 or out-of-range expert id zero-fills the rows.
                const bfloat128 zero_out = v_convert_f32_to_bf16_all_b(
                    (float128){0}, SW_LINEAR);
                const int rows_w[ROWS_PER_MEMBER] = {r0, r1, r2, r3};
                for (int tile = start[0]; tile < end[0]; ++tile) {
                    for (int row_i = 0; row_i < ROWS_PER_MEMBER; ++row_i) {
                        int5 zdst = {tile * 512, rows_w[row_i], slot, 0, 0};
                        for (int part = 0; part < 4; ++part) {
                            if (zdst[0] < k_total)
                                v_bf16_st_tnsr(zdst, output, zero_out);
                            zdst[0] += 128;
                        }
                    }
                }
                continue;
            }
            for (int tile = start[0]; tile < end[0]; ++tile) {
                // H27: all eight HBM loads of the member's four rows first, so
                // their round trips overlap; the packed planes are read once.
                const uchar256 p0 = v_u8_ld_tnsr_b(
                    (int5){tile * 256, r0, expert, 0, 0}, packed);
                const uchar256 p1 = v_u8_ld_tnsr_b(
                    (int5){tile * 256, r1, expert, 0, 0}, packed);
                const uchar256 p2 = v_u8_ld_tnsr_b(
                    (int5){tile * 256, r2, expert, 0, 0}, packed);
                const uchar256 p3 = v_u8_ld_tnsr_b(
                    (int5){tile * 256, r3, expert, 0, 0}, packed);
                const uchar256 q0 = v_u8_ld_tnsr_b(
                    (int5){tile * 32, r0, expert, 0, 0}, scales);
                const uchar256 q1 = v_u8_ld_tnsr_b(
                    (int5){tile * 32, r1, expert, 0, 0}, scales);
                const uchar256 q2 = v_u8_ld_tnsr_b(
                    (int5){tile * 32, r2, expert, 0, 0}, scales);
                const uchar256 q3 = v_u8_ld_tnsr_b(
                    (int5){tile * 32, r3, expert, 0, 0}, scales);
                // H27: one E4M3 decode for the four rows, then each row's BF16
                // scales are broadcast per 128-element part as before.
                const ushort128 scale_bits = nvfp4_e4m3_vec_bits_packed(
                    nvfp4_pack_scale_rows(q0, q1, q2, q3), spread_idx);
                DECODE_AND_STORE_ROW(p0, r0, 0)
                DECODE_AND_STORE_ROW(p1, r1, 1)
                DECODE_AND_STORE_ROW(p2, r2, 2)
                DECODE_AND_STORE_ROW(p3, r3, 3)
            }
        }
    }
}

#undef DECODE_AND_STORE_PART
#undef DECODE_AND_STORE_ROW
