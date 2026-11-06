// SPDX-License-Identifier: Apache-2.0
// Host glue for H13 hc_combine. Torch: residual [T, S*D], block [T, D],
// inj [T, S] -> out [T, S*D] (TPC reversed). Index (tile128, stream, token).
#include "qwen4exp_hc_combine_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_hc_combine_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_hc_combine_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpHcCombineBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_hc_combine_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpHcCombineBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 3) { in->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& r = in->inputTensors[0].geometry;
    const auto& b = in->inputTensors[1].geometry;
    const auto& j = in->inputTensors[2].geometry;
    const auto& y = in->outputTensors[0].geometry;
    if (r.dataType != DATA_BF16 || b.dataType != DATA_BF16 || j.dataType != DATA_BF16 ||
        y.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned total = r.maxSizes[0], tokens = r.maxSizes[1];
    const unsigned width = b.maxSizes[0], streams = j.maxSizes[0];
    if (r.dims != 2 || b.dims != 2 || j.dims != 2 || y.dims != 2 || width == 0 ||
        streams == 0 || tokens == 0 || total != width * streams ||
        b.maxSizes[1] != tokens || j.maxSizes[1] != tokens ||
        y.maxSizes[0] != total || y.maxSizes[1] != tokens) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = (width + 127) / 128;
    out->indexSpaceGeometry[1] = streams;
    out->indexSpaceGeometry[2] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int bb, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = bb; p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, (int)total - 1);
    map(out->inputTensorAccessPattern[0], 1, 2, 1, 0, 0);
    map(out->inputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[1], 1, 2, 1, 0, 0);
    map(out->inputTensorAccessPattern[2], 0, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[2], 1, 2, 1, 0, 0);
    map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, (int)total - 1);
    map(out->outputTensorAccessPattern[0], 1, 2, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_hc_combine_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_hc_combine_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
