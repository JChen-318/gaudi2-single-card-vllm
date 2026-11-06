// SPDX-License-Identifier: Apache-2.0
// Geometry constants shared by the Qwen4Exp NVFP4 MoE TPC kernels and their
// host glue. Plain macros so the same header serves tpc-clang and g++.
#pragma once

// Slots are bucketed in blocks of this many routed positions (Strata's
// moe_mme_count/scatter block): one member loads one block's ids as one
// 64-lane vector and the per-block offsets are small integers.
#define QWEN4EXP_MOE_BLOCK_SLOTS 64

// Output rows one TPC member owns in the decode GEMV (latency hiding: the
// rows' weight loads are issued before any of their MACs).
#define QWEN4EXP_MOE_ROWS_PER_MEMBER 4

// K elements covered by one 256-byte packed load, and by one dual group of it.
#define QWEN4EXP_MOE_K_PER_TILE 512
#define QWEN4EXP_MOE_K_PER_PART 128

// Largest routed-slot count a single decode launch handles (same cap as the
// shipped one-row indexed GEMV).
#define QWEN4EXP_MOE_MAX_DECODE_SLOTS 64
