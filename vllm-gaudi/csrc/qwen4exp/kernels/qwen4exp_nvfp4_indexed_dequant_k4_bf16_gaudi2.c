// SPDX-License-Identifier: Apache-2.0
// H31 k4 layout of the indexed NVFP4 dequant (prefill MoE). Every whole 512-value K tile is written
// in nibble-major order: u16 lane i of the row's 256 packed bytes holds the codes of k = 4i..4i+3,
// all under the block scale of i / 4, and output part j (positions tile*512 + j*128 .. +127) holds
// k = 4i + j. The values are the same BF16 bits as the natural-order kernel (same e2m1 * e4m3
// product, same scale2 multiply); only their positions along K move, so a GEMM whose activation is
// permuted the same way (decode_graph._k4_layout) gives the same products. A trailing partial tile
// (K % 512) keeps the natural order and the guarded per-part path. 21 vector instructions per row
// and whole tile instead of 33.
//
// Natural-order kernel notes: include/qwen4exp_nvfp4_indexed_dequant_bf16.h.
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


// The first 128-element part of one row (a tail tile with nothing past it).
#define DECODE_AND_STORE_ROW_PART0(PACKED, ROW, DG) { \
    const uchar256 rscale = NVFP4_BROADCAST_SCALE(scale_bits, DG); \
    int5 row_dst = {tile * 512, ROW, slot, 0, 0}; \
    DECODE_AND_STORE_PART(row_dst, PACKED, rscale, pick_idx0, 0) \
}

// H31: one row of one whole tile in nibble-major order (see the header).
#define K4_DECODE_AND_STORE_ROW(PACKED, ROW, DG) { \
    const ushort128 p16 = *((ushort128*)&PACKED); \
    const ushort128 c0 = v_u16_and_b(p16, 15); \
    const ushort128 c1 = v_u16_and_b(v_u16_shr_b(p16, 4), 15); \
    const ushort128 c2 = v_u16_and_b(v_u16_shr_b(p16, 8), 15); \
    const ushort128 c3 = v_u16_shr_b(p16, 12); \
    const uchar256 sbytes = v_u8_shuffle_b(NVFP4_BROADCAST_SCALE(scale_bits, DG), scale_idx4, 0, zero); \
    const bfloat128 sc = *((bfloat128*)&sbytes); \
    int5 dst = {tile * 512, ROW, slot, 0, 0}; \
    v_bf16_st_tnsr(dst, output, v_bf16_mul_b(v_bf16_mul_b(as_bfloat128(v_i16_lookup_1c(c0, 457, 0, (short128)0)), sc), factor)); \
    dst[0] += 128; \
    v_bf16_st_tnsr(dst, output, v_bf16_mul_b(v_bf16_mul_b(as_bfloat128(v_i16_lookup_1c(c1, 457, 0, (short128)0)), sc), factor)); \
    dst[0] += 128; \
    v_bf16_st_tnsr(dst, output, v_bf16_mul_b(v_bf16_mul_b(as_bfloat128(v_i16_lookup_1c(c2, 457, 0, (short128)0)), sc), factor)); \
    dst[0] += 128; \
    v_bf16_st_tnsr(dst, output, v_bf16_mul_b(v_bf16_mul_b(as_bfloat128(v_i16_lookup_1c(c3, 457, 0, (short128)0)), sc), factor)); \
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
    // H31: byte b of dual group g reads u16 scale lane 8g + ((b & 63) >> 3) of the row broadcast.
    const uchar256 scale_idx4 = v_u8_or_b(v_u8_or_b(
        v_u8_shl_b(v_u8_add_b(v_u8_shl_b(v_u8_shr_b(lanes, 6), 3), v_u8_shr_b(v_u8_and_b(lanes, 63), 3)), 1),
        v_u8_and_b(lanes, 1)), 0x80);
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
                if ((tile + 1) * 512 <= k_total) {
                    K4_DECODE_AND_STORE_ROW(p0, r0, 0)
                    K4_DECODE_AND_STORE_ROW(p1, r1, 1)
                    K4_DECODE_AND_STORE_ROW(p2, r2, 2)
                    K4_DECODE_AND_STORE_ROW(p3, r3, 3)
                } else if (k_total - tile * 512 <= 128) {
                    // H31: a one-part tail (K = 640: down_proj's last 128 inputs). The four-part
                    // form decoded all four parts and stored one: 211 of the member's bundles for
                    // a quarter tile, two thirds of the down dequant. Same stored values.
                    DECODE_AND_STORE_ROW_PART0(p0, r0, 0)
                    DECODE_AND_STORE_ROW_PART0(p1, r1, 1)
                    DECODE_AND_STORE_ROW_PART0(p2, r2, 2)
                    DECODE_AND_STORE_ROW_PART0(p3, r3, 3)
                } else {
                    DECODE_AND_STORE_ROW(p0, r0, 0)
                    DECODE_AND_STORE_ROW(p1, r1, 1)
                    DECODE_AND_STORE_ROW(p2, r2, 2)
                    DECODE_AND_STORE_ROW(p3, r3, 3)
                }
            }
        }
    }
}

#undef DECODE_AND_STORE_PART
#undef DECODE_AND_STORE_ROW
#undef DECODE_AND_STORE_ROW_PART0
#undef K4_DECODE_AND_STORE_ROW
