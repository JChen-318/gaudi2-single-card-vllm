// SPDX-License-Identifier: Apache-2.0
// Shared helpers for the H13 HyperConnection TPC kernels.
//
// The sigmoid is the TPC hardware sigmoid (the compiler-library form Strata
// uses under STRATA_FAST_MATH in hpu/tpc/gdn_math.h, and gdn_norm_fast.c uses
// directly). It agrees with the torch sigmoid to about an ulp in F32, well
// inside the BF16 output rounding, and avoids a per-element exp/divide.
#pragma once

static inline float64 qwen4exp_hc_sigmoid(float64 x) { return v_sigmoid_f32(x); }
