// SPDX-License-Identifier: Apache-2.0
// Host glue for H13 hc_combine_norm. Torch: residual [T, S*D], block [T, D],
// inj [T, S], gamma [Sg, D] -> (combined [T, S*D], xn [T, S*D]).
// Index space (streams, tokens): one member owns one (stream, token) row.
#include "qwen4exp_hc_combine_norm_bf16_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char _binary___qwen4exp_hc_combine_norm_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_hc_combine_norm_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpHcCombineNormBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_hc_combine_norm_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpHcCombineNormBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != sizeof(float))
        return GLUE_FAILED;
    const float eps = *static_cast<const float*>(in->nodeParams.nodeParams);
    if (!(eps > 0) || !std::isfinite(eps)) return GLUE_FAILED;
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& r = in->inputTensors[0].geometry;
    const auto& b = in->inputTensors[1].geometry;
    const auto& j = in->inputTensors[2].geometry;
    const auto& g = in->inputTensors[3].geometry;
    if (r.dataType != DATA_BF16 || b.dataType != DATA_BF16 || j.dataType != DATA_BF16 ||
        g.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    for (unsigned i = 0; i < 2; ++i)
        if (in->outputTensors[i].geometry.dataType != DATA_BF16)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned total = r.maxSizes[0], tokens = r.maxSizes[1];
    const unsigned width = g.maxSizes[0], grows = g.maxSizes[1];
    const unsigned streams = j.maxSizes[0];
    if (r.dims != 2 || b.dims != 2 || j.dims != 2 || g.dims != 2 || width == 0 ||
        width % 128 || streams == 0 || tokens == 0 || total != width * streams ||
        b.maxSizes[0] != width || b.maxSizes[1] != tokens || j.maxSizes[1] != tokens ||
        (grows != 1 && grows != streams)) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    for (unsigned i = 0; i < 2; ++i) {
        const auto& y = in->outputTensors[i].geometry;
        if (y.dims != 2 || y.maxSizes[0] != total || y.maxSizes[1] != tokens)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = streams;
    out->indexSpaceGeometry[1] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int bb, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = bb; p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, (int)total - 1);
    map(out->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[1], 0, 0, 0, 0, (int)width - 1);
    map(out->inputTensorAccessPattern[1], 1, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[2], 0, 0, 1, 0, 0);
    map(out->inputTensorAccessPattern[2], 1, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[3], 0, 0, 0, 0, (int)width - 1);
    map(out->inputTensorAccessPattern[3], 1, 0, 0, 0, (int)grows - 1);
    for (unsigned i = 0; i < 2; ++i) {
        map(out->outputTensorAccessPattern[i], 0, 0, 0, 0, (int)total - 1);
        map(out->outputTensorAccessPattern[i], 1, 1, 1, 0, 0);
    }
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &eps, sizeof(eps));
    const auto* start = &_binary___qwen4exp_hc_combine_norm_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_hc_combine_norm_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
