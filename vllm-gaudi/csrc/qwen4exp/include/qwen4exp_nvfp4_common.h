// SPDX-License-Identifier: Apache-2.0
// NVFP4 decode helpers shared by the Qwen4Exp TPC kernels (H7/H8).
//
// The weight of an NVFP4 element is e2m1(nibble) * e4m3(scale) and is
// representable in BF16 without rounding (at most 5 significant mantissa
// bits), so the kernels build it in BF16 instead of the integer bit
// construction in qwen4exp_nvfp4_decode.h.
//
// H7 findings that shaped this header (see qwen4exp_nvfp4_indexed_dequant_bf16.h):
//   (a) The E2M1 code decodes with the hardware ROM lookup function 457 (the
//       stock MXFP4 cast's table) in one instruction instead of a 16-entry
//       shuffle table.
//   (b) A 512-value tile has 32 distinct E4M3 scale bytes, so the E4M3 decode
//       runs ONCE per tile and each 128-value part picks its BF16 scale lanes
//       with v_u8_shuffle_b.
//   (c) H21: that per-tile E4M3 decode is the hardware FP8 conversion
//       (convert.f8_143), 3 vector instructions, and it needs no subnormal
//       table. See the H21 block below for the measured lane and exponent-bias
//       contract.
//
// v_u8_shuffle_b convention (verified on the TPC simulator): per output
// 64-byte dual group, bit 7 set selects the first source, bit 7 clear selects
// the fourth argument, and the low 6 bits are the byte offset inside that dual
// group. A 128-lane BF16 consumer therefore needs the same 64 bytes in every
// dual group, which is what v_u8_mov_dual_group_all_b provides.

static inline uchar256 nvfp4_replicate(ushort128 table) {
    return v_u8_mov_dual_group_all_b(
        *((uchar256*)&table), 0xffffffff, 0, 0, 0, 0,
        MkWrA(3, 3, 3, 3), (uchar256){0});
}

// ---------------------------------------------------------------------------
// H21: E4M3 block-scale decode through the hardware FP8 conversion
// (convert.f8_143). Replaces the ~21-instruction integer decode (unpack, 12
// shifts/ors, a 16-entry subnormal table lookup, sel_eq) that H7(b) hoisted to
// one call per row and tile.
//
// The two facts this rests on were measured on Gaudi2 with
// tools/qwen4exp_flashnext/probe_h21_convert_f8.py (all 128 codes 0x00..0x7F,
// one-hot lane mapping per source byte, four flag combinations):
//
//  (1) LANES. v_convert_f8_to_bf16_all_b(uchar256, flags) returns a
//      bfloat128_pair_t whose two halves are NOT the plain low/high split:
//        flags without SW_LINEAR: pair.v1 lane i = source byte 2i,
//                                 pair.v2 lane i = source byte 2i+1  (interleaved)
//        flags with    SW_LINEAR: pair.v1 lane i = source byte i,
//                                 pair.v2 lane i = source byte 128+i (linear,
//                                 compiled as 2x unpack.f8_143 + 2x convert)
//      The interleaved form is one instruction, so this decode uses it and
//      spreads the 32 scale bytes to the even byte positions first (one
//      v_u8_shuffle_b). v2 is the same data (index b>>1) and is dropped.
//
//  (2) EXPONENT BIAS. With SW_FP8_BIAS7, codes 0x78..0x7E (which E4M3 bias 7
//      defines as 256..448) come back as +Inf/NaN and 0x7F as NaN, because the
//      Gaudi2 FP8_143 conversion reserves exp == 15 for Inf/NaN. The bias
//      flags (SW_FP8_BIAS3/BIAS11/BIAS15) are accepted by the compiler but do
//      not change this instruction's result on Gaudi2 (measured: the BIAS15
//      row of the probe is bit-identical to BIAS7), so the fix is explicit --
//      see nvfp4_e4m3_vec_bits below.
//
// Measured: 7 vector instructions per row and tile (shuffle, and, sel_geq,
// sub, convert, shr, add) replacing 21, plus the caller's unchanged
// nvfp4_replicate. Bit exactness against the old integer decode is gated by
// test_nvfp4_tpc_hpu.py (synthetic + real gate_proj/down_proj) and by
// bench_h19_tpc.py --compare.
typedef _Float8_143 nvfp4_f8_256 __attribute__((ext_vector_type(256)));

// Byte b of the spread reads raw byte (b>>1): the 32 scale bytes of a tile end
// up at both byte 2v and 2v+1 of dual group 0. Bit 7 selects the first
// v_u8_shuffle_b source; bits 6..0 stay inside the 64-byte dual group.
static inline uchar256 nvfp4_e4m3_spread_index(void) {
    return v_u8_or_b(
        v_u8_and_b(v_u8_shr_b((uchar256)V_LANE_ID_8, 1), 0x3f), 0x80);
}

// H21: decode a tile's 32 E4M3 scale bytes to BF16 bits ONCE (raw_scale bytes
// 0..31, one per 16-value group of a 512-value tile). Lane v of the result
// holds the BF16 bits of scale byte v, exactly as the integer decode did.
//
// The 0x78..0x7F fix. The hardware conversion reserves exp == 15 for Inf/NaN,
// so codes 0x78..0x7E come back as +Inf/NaN instead of 256..448 and 0x7F as
// NaN. For those codes (s - 8) is exp == 14 and decodes to exactly half the
// E4M3 bias-7 value, so the decode uses s - 8 there and adds 0x0080 to the
// BF16 exponent afterwards (an exact doubling; +-0 stays +-0 because the add
// is masked). sel_geq produces 0x0808 in exactly those lanes, which serves
// both steps: 0x0808 subtracted from the u16 lane drops 8 from the byte, and
// 0x0808 >> 4 == 0x0080 is the exponent increment. The compare masks the sign
// bit, so it flags exactly the 16 exp == 15 magnitudes (0x78..0x7F and their
// negative twins 0xF8..0xFF) and leaves 0x80 (-0.0) alone.
static inline ushort128 nvfp4_e4m3_vec_bits(uchar256 raw_scale, uchar256 spread_idx) {
    // Spread byte v to both byte positions of u16 lane v first: pair.v1 of the
    // interleaved conversion then has scale byte v in lane v, and the u16 view
    // of the spread is s * 0x0101, which makes the exp == 15 test one compare.
    const uchar256 spread = v_u8_shuffle_b(raw_scale, spread_idx, 0, (uchar256){0});
    const ushort128 spread16 = *((ushort128*)&spread);
    // Test the exponent only: 0x80 (-0.0) and 0x88..0xF7 must NOT be treated
    // as exp == 15, so the sign bit is masked out of the compare.
    const ushort128 magnitude = v_u16_and_b(spread16, 0x7f7f);
    const ushort128 high = v_u16_sel_geq_u16_b(magnitude, 0x7878, 0x0808, (ushort128)0);
    const ushort128 adjusted = v_u16_sub_b(spread16, high);
    const bfloat128_pair_t converted = v_convert_f8_to_bf16_all_b(
        *((nvfp4_f8_256*)&adjusted), SW_FP8_BIAS7);
    return v_u16_add_b(as_ushort128(converted.v1), v_u16_shr_b(high, 4));
}

// ---------------------------------------------------------------------------
// H27: four rows of E4M3 scales decoded by ONE convert.f8_143
//
// H21 decodes one row (32 scale bytes) per tile and throws away six of the
// eight 32-lane blocks the conversion produces: without SW_LINEAR, pair.v1
// lane i holds source byte 2i and pair.v2 lane i holds source byte 2i+1
// (measured on Gaudi2 by probe_h21_convert_f8.py, all 256 source bytes), so
// the 128 lanes of pair.v1 cover the whole 256-byte input. The scale byte of
// a dual group's 32-byte low half lands in that dual group's 32 v1 lanes,
// because v_u8_shuffle_b reads the source's SAME dual group as the output byte
// (H19 probe, 100 % of lanes). Hence four rows in the four dual groups:
//
//   dual group 0 -> pair.v1 lanes   0..31   row 0 scales 0..31
//   dual group 1 -> pair.v1 lanes  32..63   row 1
//   dual group 2 -> pair.v1 lanes  64..95   row 2
//   dual group 3 -> pair.v1 lanes  96..127  row 3
//
// pair.v2 is the same 128 values from the odd byte of each u16 lane (the spread
// mirrors every scale byte into both halves of its lane) and stays unused.
//
// Cost for four rows and one tile: 3 mov_dg (pack) + spread + the six decode
// instructions instead of 4 x (spread + six decode) = 28. The 256-byte HBM
// scale load of each row already carries the 32 bytes it needs in dual group 0
// (the load is 256 contiguous bytes of that row's scale plane), so the pack is
// three register moves and no extra memory traffic.
//
// Byte lane b of every dual group reads raw byte (b>>1) & 0x1f: the low 32
// bytes of the dual group, each scale byte mirrored into both bytes of its u16
// lane. The 0x1f mask (vs 0x3f in the one-row form) is what makes all four dual
// groups read their own low half; with 0x3f the odd groups would read bytes
// 32..63, i.e. the next tile's scale bytes.
static inline uchar256 nvfp4_e4m3_spread_index_packed(void) {
    return v_u8_or_b(
        v_u8_and_b(v_u8_shr_b((uchar256)V_LANE_ID_8, 1), 0x1f), 0x80);
}

// Move row 1..3's 32 scale bytes (dual group 0 of their own load) into dual
// groups 1..3. The first row rides along as the merge operand of the first
// mov_dg, so four rows cost three instructions and no extra load.
//
// The merging form takes MkWr(lower, upper), i.e. one write-enable bit per
// 32-byte half of the destination dual group -- MkWr(1, 1) writes the whole
// 64-byte group. (Passing the four-field MkWrA(3, 3, 3, 3) of the "all" form
// here encodes wr_lg = wr_ug = 0, i.e. writes nothing: measured on Gaudi2 by
// probe_h27_scale_pack.py, dual groups 1..3 stayed zero.)
static inline uchar256 nvfp4_pack_scale_rows(
    uchar256 q0, uchar256 q1, uchar256 q2, uchar256 q3) {
    const uchar256 p01 = v_u8_mov_dual_group_b(
        q1, 0xffffffff, 0, 1, MkWr(1, 1), q0);
    const uchar256 p012 = v_u8_mov_dual_group_b(
        q2, 0xffffffff, 0, 2, MkWr(1, 1), p01);
    return v_u8_mov_dual_group_b(
        q3, 0xffffffff, 0, 3, MkWr(1, 1), p012);
}

// Same decode as nvfp4_e4m3_vec_bits, but the spread input is the four-row
// pack and the result carries row r in u16 lanes 32r..32r+31.
static inline ushort128 nvfp4_e4m3_vec_bits_packed(
    uchar256 packed_scale, uchar256 spread_idx) {
    const uchar256 spread = v_u8_shuffle_b(
        packed_scale, spread_idx, 0, (uchar256){0});
    const ushort128 spread16 = *((ushort128*)&spread);
    const ushort128 magnitude = v_u16_and_b(spread16, 0x7f7f);
    const ushort128 high = v_u16_sel_geq_u16_b(magnitude, 0x7878, 0x0808, (ushort128)0);
    const ushort128 adjusted = v_u16_sub_b(spread16, high);
    const bfloat128_pair_t converted = v_convert_f8_to_bf16_all_b(
        *((nvfp4_f8_256*)&adjusted), SW_FP8_BIAS7);
    return v_u16_add_b(as_ushort128(converted.v1), v_u16_shr_b(high, 4));
}

// Broadcast the 32 BF16 scales of row DG of a packed decode across the four
// dual groups, which is the form the per-128-element-part scale pick
// (v_u8_shuffle_b, same dual group) consumes. DG must be a literal.
#define NVFP4_BROADCAST_SCALE(BITS, DG) \
    v_u8_mov_dual_group_all_b(*((uchar256*)&(BITS)), 0xffffffff, DG, DG, DG, DG, \
                              MkWrA(3, 3, 3, 3), (uchar256){0})
