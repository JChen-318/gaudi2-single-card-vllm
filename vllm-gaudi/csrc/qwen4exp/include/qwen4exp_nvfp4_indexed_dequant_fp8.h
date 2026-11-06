// SPDX-License-Identifier: Apache-2.0
// NVFP4 (packed E2M1 + per-16 E4M3 block scales) indexed dequant to F8_E4M3.
//
// value = e2m1_true(nibble) * e4m3fn(scale_byte) * scale2[expert]
//
// The e2m1 x e4m3 product is the same exact BF16 bit construction as the BF16
// kernel (qwen4exp_nvfp4_decode.h); scale2 is applied here, BEFORE the F8
// encode, because the encode quantizes. The product w * scale2 is a single
// F32 rounding (BF16 has <= 7 significand bits, so bf16 -> f32 is exact).
//
// The F8 encode is round-to-nearest-even with saturation to +-448 and matches
// torch's float8_e4m3fn cast bit for bit wherever that cast is finite
// (|value| <= 464; torch turns larger magnitudes into the 0x7F/0xFF NaN
// codes, this kernel saturates to +-448 by contract). Verified against the
// TPC simulator on 26k+ boundary values (ties in every binade incl.
// subnormals, the 448..464 saturation region, all nibble x scale-byte
// products times a scale2 grid):
//
//   * v_convert_f32_to_f8_all_b(.., SW_RHNE | SW_FP8_BIAS7 | SW_LINEAR) is a
//     faithful RNE on |x| <= 272 but saturates its output at code 0x78 (+-256)
//     above 272 -- it can never emit 0x79..0x7E, so it is unusable alone.
//   * The "half trick" (encode x/2 and add 8 to the exponent field, the
//     vllm-gaudi csrc/deepseek_v4 KV-cache pattern) covers the top: for
//     |x| >= 248, x/2 lands in [124, 224], where scaling by 2 commutes with
//     RNE exactly and +8 on codes 0x70..0x76 reaches 0x78..0x7E with no
//     mantissa carry. It must NOT be used below that: x/2 crossing the E4M3
//     subnormal boundary (x in [0.029296875, 0.0302734375)) rounds at the
//     wrong boundary (45 observed mismatches vs torch).
//   * So the encode takes the half path exactly when the half code's
//     magnitude is >= 0x70 (|x/2| rounds to >= 128), else the direct convert.
//     The threshold also classifies x in [248, 256) correctly (x/2 rounds up
//     to 128 => the half path, which lands on 256 as RNE demands).
//
// Geometry, addressing and the scale shuffle are identical to the BF16 twin
// (qwen4exp_nvfp4_indexed_dequant_bf16.h), which stays bit-identical; only
// the scale2 input, the F32 scale2 multiply and the F8 store differ.
// Ragged-K safety: K % 128 == 0, and every 128-element store is
// bounds-checked against the output's K (Qwen4Exp expert K is 1280 or 2560;
// K/2 of 640 bytes = 2.5 tiles, so trailing tiles may over-read but never
// over-store).
#include "qwen4exp_nvfp4_decode.h"

// |x| is clamped to 448 first, so |x/2| <= 224 (code 0x76): +8 reaches at
// most 0x7E and never touches the 0x7F NaN code.
#define QWEN4EXP_NVFP4_F8_ENCODE(F32_128, CODES_U8) {                       \
    float256 direct_in = {0};                                               \
    direct_in.v1 = (F32_128).v1;                                            \
    direct_in.v2 = (F32_128).v2;                                            \
    float256 half_in = {0};                                                 \
    half_in.v1 = (F32_128).v1 * 0.5f;                                       \
    half_in.v2 = (F32_128).v2 * 0.5f;                                       \
    const uchar256 direct = as_uchar256(                                    \
        v_convert_f32_to_f8_all_b(                                          \
            direct_in, SW_RHNE | SW_FP8_BIAS7 | SW_LINEAR));                \
    const uchar256 halved = as_uchar256(                                    \
        v_convert_f32_to_f8_all_b(                                          \
            half_in, SW_RHNE | SW_FP8_BIAS7 | SW_LINEAR));                  \
    const uchar256 halved_mag = v_u8_and_b(halved, 0x7f);                   \
    const uchar256 shifted = v_u8_or_b(                                     \
        v_u8_and_b(halved, 0x80), v_u8_add_b(halved_mag, 8));               \
    CODES_U8 = v_u8_sel_geq_u8_b(halved_mag, 0x70, shifted, direct);        \
}

#define DECODE_AND_STORE_PART(PART) { \
    const uchar256 packed_part = v_u8_mov_dual_group_all_b( \
        packed_bytes, 0xffffffff, PART, PART, PART, PART, \
        MkWrA(3, 3, 3, 3), (uchar256){0}); \
    const uchar256 repeated = v_u8_shuffle_b( \
        packed_part, packed_indices, 0, (uchar256){0}); \
    const ushort128 nibble = v_u16_and_b( \
        v_u16_shr_b(*((ushort128*)&repeated), nibble_shifts), 15); \
    const uchar256 scale_bytes = v_u8_shuffle_b( \
        replicated_scale, v_u8_add_b(scale_indices, PART * 8), 0, \
        (uchar256){0}); \
    const ushort128 scale = v_u16_and_b(*((ushort128*)&scale_bytes), 255); \
    const ushort128 decoded = nvfp4_mul_bits(nibble, scale); \
    /* bf16 -> f32 is exact; the scale2 multiply is the one F32 rounding. \
     * A bare vector*scalar multiply flushes a zero product to +0.0 (TPC \
     * measurement); the MAC with a -0.0 addend keeps the IEEE product sign \
     * (+0 + -0 = +0, -0 + -0 = -0), so -0 weights encode to 0x80 as torch. */ \
    const float128 weight = \
        convert_bfloat128_to_float128(*((bfloat128*)&decoded), SW_LINEAR); \
    float128 value; \
    value.v1 = v_f32_mac_b(weight.v1, s2, -0.0f, 0); \
    value.v2 = v_f32_mac_b(weight.v2, s2, -0.0f, 0); \
    float128 clamped; \
    clamped.v1 = v_f32_min_b( \
        v_f32_max_b(value.v1, -448.0f), 448.0f); \
    clamped.v2 = v_f32_min_b( \
        v_f32_max_b(value.v2, -448.0f), 448.0f); \
    uchar256 codes; \
    QWEN4EXP_NVFP4_F8_ENCODE(clamped, codes); \
    /* Ragged-K safety: the last 256-byte tile may decode past K rows. */ \
    if (dst[0] < k_total) \
        v_f8_st_tnsr_partial( \
            dst, output, *((minifloat256*)&codes), 127, 0); \
    dst[0] += 128; \
}

void main(tensor expert_ids, tensor packed, tensor scales, tensor scale2, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int experts = get_dim_size(packed, 2);
    const int k_total = get_dim_size(output, 0);
    const uchar256 lanes = (uchar256)V_LANE_ID_8;
    const uchar256 packed_indices = v_u8_or_b(v_u8_shr_b(lanes, 2), 0x80);
    // u16 lane j fetches scale byte (2j)>>5 = j>>4: exactly the 16-element
    // group of output element j (+8 per 128-row part; max index 31, inside
    // the 64-byte dual-group window that mov_dual_group replicates).
    const uchar256 scale_indices = v_u8_or_b(v_u8_shr_b(lanes, 5), 0x80);
    const short128 nibble_shifts = (short128)v_u16_shl_b(
        v_u16_and_b((ushort128)V_LANE_ID_16, 1), 2);
    for (int slot = start[2]; slot < end[2]; ++slot) {
        const int expert = s_i32_ld_g(gen_addr((int5){slot, 0, 0, 0, 0}, expert_ids));
        const int ok = expert >= 0 && expert < experts;
        // Per-slot scalars are read once and reused across the member's rows
        // (a scalar global load waits a full memory round trip). The vector
        // MACs below take this scalar directly -- a 0x80-indexed v_f32_shuffle
        // broadcast only covers a 32-lane window, so it is not usable here.
        const float s2 = ok ? s_f32_ld_g(gen_addr((int5){expert, 0, 0, 0, 0}, scale2)) : 0.0f;
        for (int row = start[1]; row < end[1]; ++row) {
            for (int tile = start[0]; tile < end[0]; ++tile) {
                int5 dst = {tile * 512, row, slot, 0, 0};
                if (!ok) {
                    const uchar256 zeros = {0};
                    for (int part = 0; part < 4; ++part) {
                        if (dst[0] < k_total)
                            v_f8_st_tnsr_partial(
                                dst, output, *((minifloat256*)&zeros), 127, 0);
                        dst[0] += 128;
                    }
                    continue;
                }
                const uchar256 packed_bytes = v_u8_ld_tnsr_b(
                    (int5){tile * 256, row, expert, 0, 0}, packed);
                const uchar256 raw_scale = v_u8_ld_tnsr_b(
                    (int5){tile * 32, row, expert, 0, 0}, scales);
                const uchar256 replicated_scale = v_u8_mov_dual_group_all_b(
                    raw_scale, 0xffffffff, 0, 0, 0, 0,
                    MkWrA(3, 3, 3, 3), (uchar256){0});
                DECODE_AND_STORE_PART(0)
                DECODE_AND_STORE_PART(1)
                DECODE_AND_STORE_PART(2)
                DECODE_AND_STORE_PART(3)
            }
        }
    }
}

#undef DECODE_AND_STORE_PART
#undef QWEN4EXP_NVFP4_F8_ENCODE
