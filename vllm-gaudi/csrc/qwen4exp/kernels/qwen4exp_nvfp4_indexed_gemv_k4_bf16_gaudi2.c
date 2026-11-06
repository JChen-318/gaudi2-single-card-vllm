// SPDX-License-Identifier: Apache-2.0
// H31 k4 layout of the indexed NVFP4 GEMV (decode down_proj): every whole 512-value K tile runs in
// nibble-major order -- u16 lane i of the 256 packed bytes holds the codes of k = 4i..4i+3, all
// under the block scale of i / 4 -- so one row and tile costs 21 vector instructions instead of 33.
// The caller passes x with each whole tile permuted (x_k4[t * 512 + j * 128 + i] = x[t * 512 +
// 4i + j]) and a trailing partial tile (K % 512, 128 for K = 640) in natural order, which keeps
// the original guarded per-part loop. Same weights, same F32 accumulation; only the lane each
// product lands in changes, so the F32 sum order (and rarely the last BF16 bit) differs.
//
// Original kernel notes follow.
// SPDX-License-Identifier: Apache-2.0
// Indexed NVFP4 GEMV, decode step. One output element is
//   y[slot, row] = scale2[expert] * sum_k bf16(e2m1 * e4m3)[row, k] * x[x_row, k]
// accumulated in F32. Weights are never materialized: every weight is decoded
// from its nibble and per-16 E4M3 block scale in registers, so the NVFP4 group
// granularity is kept as stored (not requantized to IQ4 or FP8).
//
// H10: x may be per-slot [S, K] (down_proj, where every slot has its own
// activation) or per-token [T, K] with S = T * top_k (gate/up, where top_k
// consecutive slots share one token row and the caller must not replicate x).
// x_row is slot in the first case and slot / top_k in the second; this is an
// index change only, so the (slot, row) MAC order and output bits are
// unchanged. The x_rows == 1 decode case (T = 1) needs no per-slot division.
//
// TPC geometry (Synapse reverses torch): x [K, x_rows], packed [K/2, N, E],
// scales [K/16, N, E], output [N, S]. K is a multiple of 128. A 256-byte
// packed tile covers 512 K elements; a partial last tile decodes only the
// 128-wide parts still inside K.
//
// H8: two structures combined.
//  * Four rows per member, from native/qwen4exp-nvfp4-moe. One TPC member owns
//    one slot and four output rows, so the eight weight/scale loads of the four
//    rows are issued before the first MAC, one activation tile is shared by all
//    four rows, and an idle member has nothing to overlap. The MAC order per
//    (slot, row) is untouched (K ascending, F32 lane tree, then scale2), so the
//    output is bit-identical to the one-row kernel.
//  * H7's fast weight decode (see qwen4exp_nvfp4_common.h): the E4M3 scale
//    bytes decode ONCE per row and tile, the E2M1 code comes from the hardware
//    ROM lookup function 457, and each part picks its BF16 scale lanes with one
//    v_u8_shuffle_b. The BF16 factor product replaces nvfp4_mul_bits.
//    H21 turns that per-row-tile E4M3 decode into the hardware FP8 conversion
//    (convert.f8_143 + one exact x256), 3 vector instructions instead of 21.
//    Signed zero: v_bf16_mul_b flushes -0 to +0. With F32 round-to-nearest-even
//    a lane that starts at +0.0f keeps (+0.0f) + (-0.0f) == +0.0f, so the
//    flush cannot change the accumulated value or its sign: the F32 lane
//    accumulation stays 100% bit-identical to the integer decode.
//
// Why the product is not widened to F32 lanes by hand: a BF16 value is the top
// half of its F32 pattern, but on TPC v_u16_shl_b(v, 16) shifts 32-bit slots
// (it keeps every second element and drops the other), and the general
// widening (v_bf16_unpack_b + mov_dual_group + convert) costs 8 vector ops per
// 128 lanes. v_bf16_mac_acc32_b is the hardware BF16 multiply with F32
// accumulate: the e2m1 * e4m3 product has at most 5 significand bits, so it is
// exact in BF16, and the accumulation is the same F32 round-to-nearest add the
// scalar form would do. It is what the shipped one-row GEMV and the native
// four-row GEMV use, so the outputs stay bit-identical to both.
//
// The K loop is split: whole 512-element tiles run unguarded (no branches in
// the hot path), and the possibly-partial trailing tile is peeled into a cold
// guarded loop. Keeping the guards out of the hot path is also what keeps the
// four row accumulators in vector registers: the only local-memory traffic left
// is the eight early weight/scale loads parked while those accumulators are
// live (the native four-row GEMV has the same pattern, with more of it).
#include "qwen4exp_nvfp4_common.h"

#define ROWS_PER_MEMBER 4
#define K_PER_TILE 512   // one 256-byte packed load covers 512 E2M1 nibbles
#define K_PER_PART 128   // one 64-byte dual group of packed bytes

// Decode the PART-th 128-element K part of one row's weights and MAC it into
// ACC (F32 lanes, v1 = K elements 0..63 of the part, v2 = 64..127).
// PACKED is that row's 256 packed bytes, RSCALE its E4M3 scales already decoded
// to BF16 bits and replicated across the four dual groups, PICK the part's
// scale-byte pattern, XV the shared BF16 activation part.
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

// One 512-element tile for the member's four rows. GUARDED is 0 for a whole
// tile and 1 for the peeled trailing tile, where each part is only MACed while
// it lies inside K (the loads then repeat the last valid offset so they stay in
// bounds).
#define GEMV_TILE(GUARDED) { \
    /* All eight weight/scale loads first: independent, so their round trips \
       overlap. */ \
    const uchar256 p0 = v_u8_ld_tnsr_b((int5){tile * 256, r0, expert, 0, 0}, packed); \
    const uchar256 p1 = v_u8_ld_tnsr_b((int5){tile * 256, r1, expert, 0, 0}, packed); \
    const uchar256 p2 = v_u8_ld_tnsr_b((int5){tile * 256, r2, expert, 0, 0}, packed); \
    const uchar256 p3 = v_u8_ld_tnsr_b((int5){tile * 256, r3, expert, 0, 0}, packed); \
    const uchar256 q0 = v_u8_ld_tnsr_b((int5){tile * 32, r0, expert, 0, 0}, scales); \
    const uchar256 q1 = v_u8_ld_tnsr_b((int5){tile * 32, r1, expert, 0, 0}, scales); \
    const uchar256 q2 = v_u8_ld_tnsr_b((int5){tile * 32, r2, expert, 0, 0}, scales); \
    const uchar256 q3 = v_u8_ld_tnsr_b((int5){tile * 32, r3, expert, 0, 0}, scales); \
    /* H7(b): one E4M3 decode per row and tile, then each part picks its BF16 \
       scale lanes. H27: the member's four rows are decoded by ONE \
       convert.f8_143 (three mov_dg + one spread + six decode instructions \
       instead of four spreads and four decodes); the per-row broadcast below \
       is the same mov_dg the four-row dequant kernel uses. */ \
    const ushort128 scale_bits = nvfp4_e4m3_vec_bits_packed( \
        nvfp4_pack_scale_rows(q0, q1, q2, q3), spread_idx); \
    const uchar256 r0s = NVFP4_BROADCAST_SCALE(scale_bits, 0); \
    const uchar256 r1s = NVFP4_BROADCAST_SCALE(scale_bits, 1); \
    const uchar256 r2s = NVFP4_BROADCAST_SCALE(scale_bits, 2); \
    const uchar256 r3s = NVFP4_BROADCAST_SCALE(scale_bits, 3); \
    /* One BF16 activation vector per 128-element part, shared by the four \
       rows of that part. */ \
    if (!(GUARDED) || k_base + 0 * K_PER_PART < k_total) { \
        const bfloat128 x0 = v_bf16_ld_tnsr_b((int5){k_base, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(acc0, p0, r0s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(acc1, p1, r1s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(acc2, p2, r2s, pick_idx0, x0, 0) \
        GEMV_ROW_PART(acc3, p3, r3s, pick_idx0, x0, 0) \
    } \
    if (!(GUARDED) || k_base + 1 * K_PER_PART < k_total) { \
        const bfloat128 x1 = v_bf16_ld_tnsr_b((int5){k1, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(acc0, p0, r0s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(acc1, p1, r1s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(acc2, p2, r2s, pick_idx1, x1, 1) \
        GEMV_ROW_PART(acc3, p3, r3s, pick_idx1, x1, 1) \
    } \
    if (!(GUARDED) || k_base + 2 * K_PER_PART < k_total) { \
        const bfloat128 x2 = v_bf16_ld_tnsr_b((int5){k2, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(acc0, p0, r0s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(acc1, p1, r1s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(acc2, p2, r2s, pick_idx2, x2, 2) \
        GEMV_ROW_PART(acc3, p3, r3s, pick_idx2, x2, 2) \
    } \
    if (!(GUARDED) || k_base + 3 * K_PER_PART < k_total) { \
        const bfloat128 x3 = v_bf16_ld_tnsr_b((int5){k3, x_row, 0, 0, 0}, x); \
        GEMV_ROW_PART(acc0, p0, r0s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(acc1, p1, r1s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(acc2, p2, r2s, pick_idx3, x3, 3) \
        GEMV_ROW_PART(acc3, p3, r3s, pick_idx3, x3, 3) \
    } \
}

// H31 nibble-major row tile (see the header).
#define K4_ROW_TILE(ACC, PACKED, BITS, DG, X0, X1, X2, X3) { \
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

#define K4_TILE { \
    const uchar256 p0 = v_u8_ld_tnsr_b((int5){tile * 256, r0, expert, 0, 0}, packed); \
    const uchar256 p1 = v_u8_ld_tnsr_b((int5){tile * 256, r1, expert, 0, 0}, packed); \
    const uchar256 p2 = v_u8_ld_tnsr_b((int5){tile * 256, r2, expert, 0, 0}, packed); \
    const uchar256 p3 = v_u8_ld_tnsr_b((int5){tile * 256, r3, expert, 0, 0}, packed); \
    const uchar256 q0 = v_u8_ld_tnsr_b((int5){tile * 32, r0, expert, 0, 0}, scales); \
    const uchar256 q1 = v_u8_ld_tnsr_b((int5){tile * 32, r1, expert, 0, 0}, scales); \
    const uchar256 q2 = v_u8_ld_tnsr_b((int5){tile * 32, r2, expert, 0, 0}, scales); \
    const uchar256 q3 = v_u8_ld_tnsr_b((int5){tile * 32, r3, expert, 0, 0}, scales); \
    const ushort128 kbits = nvfp4_e4m3_vec_bits_packed( \
        nvfp4_pack_scale_rows(q0, q1, q2, q3), spread_idx); \
    const int k_base = tile * K_PER_TILE; \
    const bfloat128 x0 = v_bf16_ld_tnsr_b((int5){k_base, x_row, 0, 0, 0}, x); \
    const bfloat128 x1 = v_bf16_ld_tnsr_b((int5){k_base + 128, x_row, 0, 0, 0}, x); \
    const bfloat128 x2 = v_bf16_ld_tnsr_b((int5){k_base + 256, x_row, 0, 0, 0}, x); \
    const bfloat128 x3 = v_bf16_ld_tnsr_b((int5){k_base + 384, x_row, 0, 0, 0}, x); \
    K4_ROW_TILE(acc0, p0, kbits, 0, x0, x1, x2, x3) \
    K4_ROW_TILE(acc1, p1, kbits, 1, x0, x1, x2, x3) \
    K4_ROW_TILE(acc2, p2, kbits, 2, x0, x1, x2, x3) \
    K4_ROW_TILE(acc3, p3, kbits, 3, x0, x1, x2, x3) \
}

// F32 block sum, then scale2, then one BF16 element (lane 0).
// H31: the K = 640 path reads its slot's activation row from local memory (copied once per slot).
__local__ bfloat128 x_local[5];

void main(tensor expert_ids, tensor packed, tensor scales, tensor x, tensor scale2, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int experts = get_dim_size(packed, 2);
    const int k_total = get_dim_size(x, 0);
    // H10: torch x is [x_rows, K], so TPC x is [K, x_rows]. x_rows is S for a
    // per-slot activation and T (with S = T * top_k) for a per-token one.
    // S comes from the output: torch output [S, N] is TPC [N, S].
    const int x_rows = get_dim_size(x, 1);
    const int slots = get_dim_size(output, 1);
    const int packed_cols = get_dim_size(packed, 0);
    const int rows_total = get_dim_size(output, 0);
    // top_k, only meaningful in the per-token case. Both operands are scalar
    // invariants across the kernel, so this is computed once before the slot
    // loop, not per tile; the compiler speculates it in the prologue and
    // selects 1 outside the per-token case (checked in its assembly). The
    // per-slot division below only runs in the per-token branch.
    const int slots_per_token =
        (x_rows > 1 && x_rows < slots) ? slots / x_rows : 1;
    const uchar256 lanes = (uchar256)V_LANE_ID_8;
    // One shared zero for the shuffle/mov_dual_group filler operand.
    const uchar256 zero = 0;
    // H7 byte conventions, hoisted out of every loop.
    const uchar256 packed_indices = v_u8_or_b(v_u8_shr_b(lanes, 2), 0x80);
    const short128 nibble_shifts = (short128)v_u16_shl_b(
        v_u16_and_b((ushort128)V_LANE_ID_16, 1), 2);
    // byte-lane b picks BF16 scale byte 2*(b>>5) + (b&1) (+16*PART, |0x80).
    const uchar256 pick_base = v_u8_add_b(
        v_u8_shl_b(v_u8_shr_b(lanes, 5), 1), v_u8_and_b(lanes, 1));
    const uchar256 pick_idx0 = v_u8_or_b(pick_base, 0x80);
    const uchar256 pick_idx1 = v_u8_or_b(v_u8_add_b(pick_base, 16), 0x80);
    const uchar256 pick_idx2 = v_u8_or_b(v_u8_add_b(pick_base, 32), 0x80);
    const uchar256 pick_idx3 = v_u8_or_b(v_u8_add_b(pick_base, 48), 0x80);
    // H21: byte spread pattern of the hardware E4M3 decode, built once.
    // H27: the packed four-row form, so one spread feeds the whole member.
    const uchar256 spread_idx = nvfp4_e4m3_spread_index_packed();
    // H31: byte b of dual group g reads u16 scale lane 8g + ((b & 63) >> 3) of the row broadcast.
    const uchar256 scale_idx4 = v_u8_or_b(v_u8_or_b(
        v_u8_shl_b(v_u8_add_b(v_u8_shl_b(v_u8_shr_b(lanes, 6), 3), v_u8_shr_b(v_u8_and_b(lanes, 63), 3)), 1),
        v_u8_and_b(lanes, 1)), 0x80);
    // Whole tiles run branch-free; the possibly-partial tiles are peeled.
    const int full_tiles = k_total / K_PER_TILE;
    const int tiles = (packed_cols + 255) / 256;
    const int k640 = k_total == 640 && packed_cols == 320;
    int local_row = -1;
    // Upper half of each dual group: the tail's eight scales, bytes 32..39 of the row's scale load.
    const uchar256 spread_hi = v_u8_or_b(
        v_u8_add_b(v_u8_and_b(v_u8_shr_b(lanes, 1), 0x1f), 32), 0x80);
    for (int slot = start[1]; slot < end[1]; ++slot) {
        // H10: which x row this slot reads. Per-slot x is the identity;
        // x_rows == 1 (B = 1, T = 1 decode) is row 0 with zero divisions;
        // otherwise slots are token-major, so top_k consecutive slots share
        // one token row: x_row = slot / (slots / x_rows) = slot / top_k.
        int x_row;
        if (x_rows == slots) {
            x_row = slot;
        } else if (x_rows == 1) {
            x_row = 0;
        } else {
            x_row = slot / slots_per_token;
        }
        // Strata's moe_mme_scatter lesson: a scalar global load waits a full
        // memory round trip, so the per-slot scalars are read once per slot and
        // reused across the member's four rows.
        if (k640 && x_row != local_row) {
            for (int c = 0; c < 5; ++c)
                x_local[c] = v_bf16_ld_tnsr_b((int5){c * 128, x_row, 0, 0, 0}, x);
            local_row = x_row;
        }
        const int expert = s_i32_ld_g(gen_addr((int5){slot, 0, 0, 0, 0}, expert_ids));
        const int ok = expert >= 0 && expert < experts;
        const float s2 = ok ? s_f32_ld_g(gen_addr((int5){expert, 0, 0, 0, 0}, scale2)) : 0.0f;
        for (int group = start[0]; group < end[0]; ++group) {
            const int n0 = group * ROWS_PER_MEMBER;
            const int n1 = n0 + 1;
            const int n2 = n0 + 2;
            const int n3 = n0 + 3;
            // Reads past the last row are clamped to it; the store is guarded.
            const int r0 = n0 < rows_total ? n0 : rows_total - 1;
            const int r1 = n1 < rows_total ? n1 : rows_total - 1;
            const int r2 = n2 < rows_total ? n2 : rows_total - 1;
            const int r3 = n3 < rows_total ? n3 : rows_total - 1;
            float128 acc0 = {0};
            float128 acc1 = {0};
            float128 acc2 = {0};
            float128 acc3 = {0};
            if (ok && k640) {
                // H31: down_proj (K = 640 = one whole tile + a 128-value tail). Both tiles' sixteen
                // loads and the five activation loads sit in one basic block ahead of any decode, so
                // their round trips overlap; the generic form below issued the tail's loads only
                // after the whole tile had been decoded (the tail is a separate cold loop), which
                // left two exposed HBM round trips per four-row member.
                const uchar256 p0 = v_u8_ld_tnsr_b((int5){0, r0, expert, 0, 0}, packed);
                const uchar256 p1 = v_u8_ld_tnsr_b((int5){0, r1, expert, 0, 0}, packed);
                const uchar256 p2 = v_u8_ld_tnsr_b((int5){0, r2, expert, 0, 0}, packed);
                const uchar256 p3 = v_u8_ld_tnsr_b((int5){0, r3, expert, 0, 0}, packed);
                const uchar256 t0 = v_u8_ld_tnsr_b((int5){256, r0, expert, 0, 0}, packed);
                const uchar256 t1 = v_u8_ld_tnsr_b((int5){256, r1, expert, 0, 0}, packed);
                const uchar256 t2 = v_u8_ld_tnsr_b((int5){256, r2, expert, 0, 0}, packed);
                const uchar256 t3 = v_u8_ld_tnsr_b((int5){256, r3, expert, 0, 0}, packed);
                const uchar256 q0 = v_u8_ld_tnsr_b((int5){0, r0, expert, 0, 0}, scales);
                const uchar256 q1 = v_u8_ld_tnsr_b((int5){0, r1, expert, 0, 0}, scales);
                const uchar256 q2 = v_u8_ld_tnsr_b((int5){0, r2, expert, 0, 0}, scales);
                const uchar256 q3 = v_u8_ld_tnsr_b((int5){0, r3, expert, 0, 0}, scales);
                const bfloat128 x0 = x_local[0];
                const bfloat128 x1 = x_local[1];
                const bfloat128 x2 = x_local[2];
                const bfloat128 x3 = x_local[3];
                const bfloat128 xt = x_local[4];
                // One scale load per row: bytes 0..31 are the tile's blocks, 32..39 the tail's.
                const uchar256 kpack = nvfp4_pack_scale_rows(q0, q1, q2, q3);
                const ushort128 kbits = nvfp4_e4m3_vec_bits_packed(kpack, spread_idx);
                K4_ROW_TILE(acc0, p0, kbits, 0, x0, x1, x2, x3)
                K4_ROW_TILE(acc1, p1, kbits, 1, x0, x1, x2, x3)
                K4_ROW_TILE(acc2, p2, kbits, 2, x0, x1, x2, x3)
                K4_ROW_TILE(acc3, p3, kbits, 3, x0, x1, x2, x3)
                // Tail: natural order, the first 128-value part of tile 1 (the original per-part path).
                const ushort128 tbits = nvfp4_e4m3_vec_bits_packed(kpack, spread_hi);
                const uchar256 r0s = NVFP4_BROADCAST_SCALE(tbits, 0);
                const uchar256 r1s = NVFP4_BROADCAST_SCALE(tbits, 1);
                const uchar256 r2s = NVFP4_BROADCAST_SCALE(tbits, 2);
                const uchar256 r3s = NVFP4_BROADCAST_SCALE(tbits, 3);
                GEMV_ROW_PART(acc0, t0, r0s, pick_idx0, xt, 0)
                GEMV_ROW_PART(acc1, t1, r1s, pick_idx0, xt, 0)
                GEMV_ROW_PART(acc2, t2, r2s, pick_idx0, xt, 0)
                GEMV_ROW_PART(acc3, t3, r3s, pick_idx0, xt, 0)
            } else if (ok) {
                for (int tile = 0; tile < full_tiles; ++tile) {
                    K4_TILE
                }
                // Cold path: at most one partial tile for a consistent
                // geometry, and a harmless safety net if the packed plane is
                // longer than the activation plane.
                for (int tile = full_tiles; tile < tiles; ++tile) {
                    const int k_base = tile * K_PER_TILE;
                    if (k_base >= k_total) break;
                    const int k1 = k_base + 1 * K_PER_PART < k_total
                        ? k_base + 1 * K_PER_PART : k_base;
                    const int k2 = k_base + 2 * K_PER_PART < k_total
                        ? k_base + 2 * K_PER_PART : k_base;
                    const int k3 = k_base + 3 * K_PER_PART < k_total
                        ? k_base + 3 * K_PER_PART : k_base;
                    GEMV_TILE(1)
                }
            }
            // H31: the four rows reduce in one basic block into lanes 0..3 and leave in one store.
            // As four guarded per-row blocks each paid the reduce-tree latency alone: ~370 of the
            // member's ~530 bundles, 2/3 of them NOPs. Same per-row arithmetic, same bits.
            {
                const int64 lane_id = (int64)V_LANE_ID_32;
                float64 totals = 0;
                totals = v_f32_mov_vb(v_broadcast_element_f32(v_f32_reduce_add(acc0.v1 + acc0.v2) * s2, 0),
                                      0, totals, v_i32_cmp_eq_b(lane_id, 0), 0);
                totals = v_f32_mov_vb(v_broadcast_element_f32(v_f32_reduce_add(acc1.v1 + acc1.v2) * s2, 0),
                                      0, totals, v_i32_cmp_eq_b(lane_id, 1), 0);
                totals = v_f32_mov_vb(v_broadcast_element_f32(v_f32_reduce_add(acc2.v1 + acc2.v2) * s2, 0),
                                      0, totals, v_i32_cmp_eq_b(lane_id, 2), 0);
                totals = v_f32_mov_vb(v_broadcast_element_f32(v_f32_reduce_add(acc3.v1 + acc3.v2) * s2, 0),
                                      0, totals, v_i32_cmp_eq_b(lane_id, 3), 0);
                float128 wide;
                wide.v1 = totals;
                wide.v2 = 0;
                const int store_last = (rows_total - n0 < ROWS_PER_MEMBER ? rows_total - n0 : ROWS_PER_MEMBER) - 1;
                v_bf16_st_tnsr_partial((int5){n0, slot, 0, 0, 0}, output,
                                       v_convert_f32_to_bf16_all_b(wide, SW_LINEAR), store_last, 0);
            }
        }
    }
}

#undef GEMV_ROW_PART
#undef K4_ROW_TILE
#undef K4_TILE
#undef GEMV_TILE
