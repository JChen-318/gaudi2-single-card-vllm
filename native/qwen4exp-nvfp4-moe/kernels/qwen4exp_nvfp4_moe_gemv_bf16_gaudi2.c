// SPDX-License-Identifier: Apache-2.0
// Grouped NVFP4 MoE decode GEMV for the selected expert group.
//
// One launch covers the whole routed slot list of a decode step:
//   y[slot, row] = scale2[expert] * sum_k bf16(e2m1 * e4m3)[expert, row, k] * x[slot, k]
// accumulated in F32. The 4-bit weights are never materialized: every weight
// element is decoded from its nibble and per-16 E4M3 block scale in registers
// (nvfp4_mul_bits), so the NVFP4 group granularity is kept as stored.
//
// TPC geometry (Synapse reverses torch dims, FCD first):
//   expert_ids [S]        I32
//   packed     [K/2,N,E]  U8    (torch [E,N,K/2])
//   scales     [K/16,N,E] U8    (torch [E,N,K/16])
//   x          [K,S]      BF16  (torch [S,K])
//   scale2     [E]        F32
//   output     [N,S]      BF16  (torch [S,N])
// Index space: (N/ROWS_PER_MEMBER, S) - one member is one slot and a group of
// ROWS_PER_MEMBER output rows.
//
// Why rows per member: TPC executes every index-space member on one thread, so
// an idle member has nothing to overlap. A member that owns four rows issues
// the four rows' packed/scale loads (8 independent vector loads) before the
// first MAC, and reuses one x vector for all four rows. This is the same
// latency hiding Strata uses in hpu/tpc/iq4q8_gemv_packed.c (five row tiles and
// four K parts in flight per member) and it is bit-identical to the one-row
// kernel: the MAC order per (slot, row) is unchanged (K ascending, then F32
// tree reduction, then scale2).
#include "qwen4exp_nvfp4_moe_decode.h"

#define ROWS_PER_MEMBER 4
#define K_PER_TILE 512   // one 256-byte packed load covers 512 E2M1 nibbles
#define K_PER_PART 128   // one 64-byte dual group of packed bytes

// Decode the PART-th 128-element K part of one row and accumulate it into ACC.
// PACKED is that row's 256 packed bytes, RSCALE its E4M3 block scales already
// replicated across the four dual groups, XV the shared activation vector.
#define MOE_ROW_PART(ACC, PACKED, RSCALE, XV, PART) { \
    const uchar256 packed_part = v_u8_mov_dual_group_all_b( \
        PACKED, 0xffffffff, PART, PART, PART, PART, MkWrA(3, 3, 3, 3), (uchar256){0}); \
    const uchar256 repeated = v_u8_shuffle_b(packed_part, packed_indices, 0, (uchar256){0}); \
    const ushort128 nibble = v_u16_and_b( \
        v_u16_shr_b(*((ushort128*)&repeated), nibble_shifts), 15); \
    const uchar256 scale_bytes = v_u8_shuffle_b( \
        RSCALE, v_u8_add_b(scale_indices, (PART)*8), 0, (uchar256){0}); \
    const ushort128 scale = v_u16_and_b(*((ushort128*)&scale_bytes), 255); \
    const ushort128 decoded = nvfp4_mul_bits(nibble, scale); \
    const bfloat128 weight = *((bfloat128*)&decoded); \
    ACC = v_bf16_mac_acc32_b(weight, XV, ACC); \
}

// One K part for the four rows: the x vector is loaded once and reused.
#define MOE_PART(PART) \
    if (k_base + (PART)*K_PER_PART < k_total) { \
        const bfloat128 xv = v_bf16_ld_tnsr_b( \
            (int5){k_base + (PART)*K_PER_PART, slot, 0, 0, 0}, x); \
        MOE_ROW_PART(acc0, p0, r0, xv, PART) \
        MOE_ROW_PART(acc1, p1, r1, xv, PART) \
        MOE_ROW_PART(acc2, p2, r2, xv, PART) \
        MOE_ROW_PART(acc3, p3, r3, xv, PART) \
    }

// F32 block sum, then scale2, then one BF16 element (lane 0).
#define MOE_STORE_ROW(ACC, ROW) { \
    const float64 total = v_f32_reduce_add(ACC.v1 + ACC.v2) * s2; \
    float64 lanes32 = 0; \
    lanes32 = v_f32_sel_eq_u32_b(V_LANE_ID_32, 0u, total, lanes32); \
    float128 wide; \
    wide.v1 = lanes32; \
    wide.v2 = 0; \
    v_bf16_st_tnsr_partial((int5){ROW, slot, 0, 0, 0}, output, \
                           v_convert_f32_to_bf16_all_b(wide, SW_LINEAR), 0, 0); \
}

void main(tensor expert_ids, tensor packed, tensor scales, tensor x, tensor scale2, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int experts = get_dim_size(packed, 2);
    const int k_total = get_dim_size(x, 0);
    const int packed_cols = get_dim_size(packed, 0);
    const uchar256 lanes = (uchar256)V_LANE_ID_8;
    const uchar256 packed_indices = v_u8_or_b(v_u8_shr_b(lanes, 2), 0x80);
    const uchar256 scale_indices = v_u8_or_b(v_u8_shr_b(lanes, 5), 0x80);
    const short128 nibble_shifts = (short128)v_u16_shl_b(
        v_u16_and_b((ushort128)V_LANE_ID_16, 1), 2);
    const int tiles = (packed_cols + 255) / 256;
    for (int slot = start[1]; slot < end[1]; ++slot) {
        // Strata's moe_mme_scatter lesson: a scalar global load waits a full memory round trip, so
        // the per-slot scalars are read once per slot and reused across the member's four rows.
        const int expert = s_i32_ld_g(gen_addr((int5){slot, 0, 0, 0, 0}, expert_ids));
        const int ok = expert >= 0 && expert < experts;
        const float s2 = ok ? s_f32_ld_g(gen_addr((int5){expert, 0, 0, 0, 0}, scale2)) : 0.0f;
        for (int group = start[0]; group < end[0]; ++group) {
            const int row0 = group * ROWS_PER_MEMBER;
            float128 acc0 = {0};
            float128 acc1 = {0};
            float128 acc2 = {0};
            float128 acc3 = {0};
            if (ok) {
                for (int tile = 0; tile < tiles; ++tile) {
                    const int k_base = tile * K_PER_TILE;
                    if (k_base >= k_total) break;
                    // All eight weight loads first: independent, so the round trips overlap.
                    const uchar256 p0 = v_u8_ld_tnsr_b(
                        (int5){tile * 256, row0 + 0, expert, 0, 0}, packed);
                    const uchar256 p1 = v_u8_ld_tnsr_b(
                        (int5){tile * 256, row0 + 1, expert, 0, 0}, packed);
                    const uchar256 p2 = v_u8_ld_tnsr_b(
                        (int5){tile * 256, row0 + 2, expert, 0, 0}, packed);
                    const uchar256 p3 = v_u8_ld_tnsr_b(
                        (int5){tile * 256, row0 + 3, expert, 0, 0}, packed);
                    const uchar256 q0 = v_u8_ld_tnsr_b(
                        (int5){tile * 32, row0 + 0, expert, 0, 0}, scales);
                    const uchar256 q1 = v_u8_ld_tnsr_b(
                        (int5){tile * 32, row0 + 1, expert, 0, 0}, scales);
                    const uchar256 q2 = v_u8_ld_tnsr_b(
                        (int5){tile * 32, row0 + 2, expert, 0, 0}, scales);
                    const uchar256 q3 = v_u8_ld_tnsr_b(
                        (int5){tile * 32, row0 + 3, expert, 0, 0}, scales);
                    const uchar256 r0 = v_u8_mov_dual_group_all_b(
                        q0, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), (uchar256){0});
                    const uchar256 r1 = v_u8_mov_dual_group_all_b(
                        q1, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), (uchar256){0});
                    const uchar256 r2 = v_u8_mov_dual_group_all_b(
                        q2, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), (uchar256){0});
                    const uchar256 r3 = v_u8_mov_dual_group_all_b(
                        q3, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), (uchar256){0});
                    MOE_PART(0)
                    MOE_PART(1)
                    MOE_PART(2)
                    MOE_PART(3)
                }
            }
            MOE_STORE_ROW(acc0, row0 + 0)
            MOE_STORE_ROW(acc1, row0 + 1)
            MOE_STORE_ROW(acc2, row0 + 2)
            MOE_STORE_ROW(acc3, row0 + 3)
        }
    }
}

#undef MOE_ROW_PART
#undef MOE_PART
#undef MOE_STORE_ROW
