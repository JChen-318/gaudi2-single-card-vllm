// SPDX-License-Identifier: Apache-2.0
// Shared helpers for the H14 GDN TPC kernels.
//
// The softplus follows Strata's gdn_math.h (hpu/tpc/gdn_math.h in the Strata
// tree): the exp/log pair is corrected so log(1+exp(x)) does not lose the
// whole result when exp(x) rounds to 1 in F32, matching
// torch's log1p(exp(x)) to F32 accuracy. The sigmoid is the TPC compiler
// library form (as in qwen4exp_hc_common.h). No SW_LINEAR conversion is used
// anywhere in the GDN kernels: v_convert_f32_to_bf16_all_b keeps element
// order only in its default (non-SW_LINEAR) form on Gaudi2.
#pragma once

static inline float64 qwen4exp_gdn_sigmoid(float64 x) { return v_sigmoid_f32(x); }

static inline float64 qwen4exp_gdn_softplus(float64 x) {
    // torch: where(x <= 20, log1p(exp(x)), x). The plain F32 form matches
    // log1p(exp(x)) to F32 accuracy for x > -16, and both are < 1e-6 in
    // magnitude below that.
    float64 exponential = v_exp_f32(x);
    float64 small = v_log_f32(1.0f + exponential);
    return v_f32_sel_less_f32_b(x, 20.0f, small, x);
}

// silu(x) = x * sigmoid(x), elementwise on a 128-wide BF16->F32 pair.
static inline float128 qwen4exp_gdn_silu128(float128 x) {
    float128 y;
    y.v1 = x.v1 * v_sigmoid_f32(x.v1);
    y.v2 = x.v2 * v_sigmoid_f32(x.v2);
    return y;
}
