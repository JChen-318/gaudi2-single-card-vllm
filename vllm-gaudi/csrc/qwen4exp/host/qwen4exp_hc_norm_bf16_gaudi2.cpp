// SPDX-License-Identifier: Apache-2.0
// Host glue for H13 hc_norm. Torch: x [T, S*D], gamma [Sg, D] -> y [T, S*D]
// (TPC: x [S*D, T], gamma [D, Sg]). Index space (streams, tokens): one member
// owns one (stream, token) row of width D (D % 128 == 0, Qwen4Exp D=2560).
#include "qwen4exp_hc_norm_bf16_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char _binary___qwen4exp_hc_norm_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_hc_norm_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpHcNormBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_hc_norm_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpHcNormBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != sizeof(float))
        return GLUE_FAILED;
    const float eps = *static_cast<const float*>(in->nodeParams.nodeParams);
    if (!(eps > 0) || !std::isfinite(eps)) return GLUE_FAILED;
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = in->inputTensors[0].geometry;
    const auto& g = in->inputTensors[1].geometry;
    const auto& y = in->outputTensors[0].geometry;
    if (x.dataType != DATA_BF16 || g.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (y.dataType != DATA_BF16) {
        const_cast<TensorGeometry&>(y).dataType = DATA_BF16;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const unsigned total = x.maxSizes[0], tokens = x.maxSizes[1];
    const unsigned width = g.maxSizes[0], grows = g.maxSizes[1];
    if (x.dims != 2 || g.dims != 2 || y.dims != 2 || width == 0 || width % 128 ||
        total == 0 || total % width || tokens == 0 ||
        (grows != 1 && grows != total / width) ||
        y.maxSizes[0] != total || y.maxSizes[1] != tokens) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    const unsigned streams = total / width;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = streams;
    out->indexSpaceGeometry[1] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = b; p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, (int)total - 1);
    map(out->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[1], 0, 0, 0, 0, (int)width - 1);
    map(out->inputTensorAccessPattern[1], 1, 0, 0, 0, (int)grows - 1);
    map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, (int)total - 1);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &eps, sizeof(eps));
    const auto* start = &_binary___qwen4exp_hc_norm_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_hc_norm_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
