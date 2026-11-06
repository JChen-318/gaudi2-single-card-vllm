// SPDX-License-Identifier: Apache-2.0
// Host glue for H14 gdn_gated_norm. Torch: x/z [R,128] BF16, weight [128]
// BF16 -> y [R,128] BF16. Scalar params {float eps; int act;}. Index (R).
#include "qwen4exp_gdn_gated_norm_bf16_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char _binary___qwen4exp_gdn_gated_norm_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_gdn_gated_norm_bf16_gaudi2_o_end;

namespace {
struct NormParams {
    float eps;
    int act;
};
}  // namespace

tpc_lib_api::GlueCodeReturn Qwen4ExpGdnGatedNormBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_gdn_gated_norm_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpGdnGatedNormBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != sizeof(NormParams))
        return GLUE_FAILED;
    const auto* params = static_cast<const NormParams*>(in->nodeParams.nodeParams);
    if (!(params->eps > 0) || !std::isfinite(params->eps) || (params->act != 0 && params->act != 1))
        return GLUE_FAILED;
    if (in->inputTensorNr != 3) { in->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = in->inputTensors[0].geometry;
    const auto& z = in->inputTensors[1].geometry;
    const auto& w = in->inputTensors[2].geometry;
    const auto& y = in->outputTensors[0].geometry;
    if (x.dataType != DATA_BF16 || z.dataType != DATA_BF16 || w.dataType != DATA_BF16 ||
        y.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned width = x.maxSizes[0], rows = x.maxSizes[1];
    if (x.dims != 2 || z.dims != 2 || w.dims != 1 || y.dims != 2 || width != 128 ||
        rows == 0 || z.maxSizes[0] != width || z.maxSizes[1] != rows ||
        w.maxSizes[0] != width || y.maxSizes[0] != width || y.maxSizes[1] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = (rows + 7) / 8;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx_, int aa, int bb, int end) {
        p.mapping[dim].indexSpaceDim = idx_; p.mapping[dim].a = aa;
        p.mapping[dim].start_b = bb; p.mapping[dim].end_b = end;
    };
    for (unsigned i = 0; i < 2; ++i) {
        map(out->inputTensorAccessPattern[i], 0, 0, 0, 0, (int)width - 1);
        map(out->inputTensorAccessPattern[i], 1, 0, 8, 0, 7);
    }
    map(out->inputTensorAccessPattern[2], 0, 0, 0, 0, (int)width - 1);
    map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, (int)width - 1);
    map(out->outputTensorAccessPattern[0], 1, 0, 8, 0, 7);
    out->kernel.paramsNr = sizeof(NormParams) / sizeof(uint32_t);
    std::memcpy(out->kernel.scalarParams, params, sizeof(NormParams));
    const auto* start = &_binary___qwen4exp_gdn_gated_norm_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_gdn_gated_norm_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
