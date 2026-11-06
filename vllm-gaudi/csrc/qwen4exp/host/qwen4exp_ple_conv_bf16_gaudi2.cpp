// SPDX-License-Identifier: Apache-2.0
// Host glue for the H29 fused PLE short conv. Torch tensors:
//   x, gated, hidden, out  BF16 [T, C]     -> TPC (C, T)     T = S*length
//   state, next_state      BF16 [S, W, C]  -> TPC (C, W, S)  W = (K-1)*dilation
//   weight                 BF16 [K, C]     -> TPC (C, K)
// Index space (C/128, ceil(length/64), S): one member owns one 128-channel
// tile of one 64-row token tile of one span.
//
// The state is vLLM's default SD conv-state orientation ((window, channels)
// per slot) -- the shape ``MambaStateShapeCalculator`` announces for this
// layer and the layout the GDN conv pool uses -- so the 128-wide vector load
// runs along channels.
#include "qwen4exp_ple_conv_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_ple_conv_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_ple_conv_bf16_gaudi2_o_end;

namespace {
constexpr unsigned kRowTile = 64;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpPleConvBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_ple_conv_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpPleConvBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = in->inputTensors[0].geometry;
    const auto& pool = in->inputTensors[1].geometry;
    const auto& w = in->inputTensors[2].geometry;
    const auto& gated = in->inputTensors[3].geometry;
    const auto& hidden = in->inputTensors[4].geometry;
    const auto& y = in->outputTensors[0].geometry;
    const auto& nxt = in->outputTensors[1].geometry;
    for (int i = 0; i < 5; ++i) {
        if (in->inputTensors[i].geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (y.dataType != DATA_BF16 || nxt.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned C = x.maxSizes[0], T = x.maxSizes[1];
    const unsigned slots = pool.maxSizes[2], window = pool.maxSizes[1];
    const unsigned taps = w.maxSizes[1];
    if (x.dims != 2 || pool.dims != 3 || w.dims != 2 || gated.dims != 2 || hidden.dims != 2 ||
        y.dims != 2 || nxt.dims != 3 || C == 0 || C % 128 || T == 0 || slots == 0 ||
        window == 0 || taps < 2 || window % (taps - 1) || window / (taps - 1) == 0 ||
        T % slots || T / slots < 1 ||
        pool.maxSizes[0] != C || w.maxSizes[0] != C ||
        gated.maxSizes[0] != C || gated.maxSizes[1] != T ||
        hidden.maxSizes[0] != C || hidden.maxSizes[1] != T ||
        y.maxSizes[0] != C || y.maxSizes[1] != T ||
        nxt.maxSizes[0] != C || nxt.maxSizes[1] != window || nxt.maxSizes[2] != slots) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    const unsigned length = T / slots;
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = C / 128;
    out->indexSpaceGeometry[1] = (length + kRowTile - 1) / kRowTile;
    out->indexSpaceGeometry[2] = slots;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = b; p.mapping[dim].end_b = end;
    };
    // The [T, C] token-major planes: rows are addressed explicitly (a tap can
    // read a row outside this member's tile), so dim 1 carries no index-space
    // stride — the gdn_conv_rows pattern.
    map(out->inputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[0], 1, 1, 0, 0, (int)T - 1);
    // state [S, W, C] -> TPC (C, W, S)
    map(out->inputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[1], 1, 0, 0, 0, (int)window - 1);
    map(out->inputTensorAccessPattern[1], 2, 2, 0, 0, (int)slots - 1);
    map(out->inputTensorAccessPattern[2], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[2], 1, 0, 0, 0, (int)taps - 1);
    map(out->inputTensorAccessPattern[3], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[3], 1, 1, 0, 0, (int)T - 1);
    map(out->inputTensorAccessPattern[4], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[4], 1, 1, 0, 0, (int)T - 1);
    map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[0], 1, 1, 0, 0, (int)T - 1);
    map(out->outputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[1], 1, 0, 0, 0, (int)window - 1);
    map(out->outputTensorAccessPattern[1], 2, 2, 0, 0, (int)slots - 1);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_ple_conv_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_ple_conv_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
