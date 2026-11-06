// SPDX-License-Identifier: Apache-2.0
// Host glue for H13 hc_gate_mix. Torch: xn/gate [T, S*D] -> out [T, D].
// Index space (tile128, token).
#include "qwen4exp_hc_gate_mix_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_hc_gate_mix_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_hc_gate_mix_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpHcGateMixBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_hc_gate_mix_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpHcGateMixBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = in->inputTensors[0].geometry;
    const auto& g = in->inputTensors[1].geometry;
    const auto& y = in->outputTensors[0].geometry;
    if (x.dataType != DATA_BF16 || g.dataType != DATA_BF16 || y.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned total = x.maxSizes[0], tokens = x.maxSizes[1];
    const unsigned width = y.maxSizes[0];
    if (x.dims != 2 || g.dims != 2 || y.dims != 2 || width == 0 || tokens == 0 ||
        total == 0 || total % width || g.maxSizes[0] != total || g.maxSizes[1] != tokens ||
        y.maxSizes[1] != tokens) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = (width + 127) / 128;
    out->indexSpaceGeometry[1] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int bb, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = bb; p.mapping[dim].end_b = end;
    };
    for (unsigned i = 0; i < 2; ++i) {
        map(out->inputTensorAccessPattern[i], 0, 0, 0, 0, (int)total - 1);
        map(out->inputTensorAccessPattern[i], 1, 1, 1, 0, 0);
    }
    map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_hc_gate_mix_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_hc_gate_mix_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
