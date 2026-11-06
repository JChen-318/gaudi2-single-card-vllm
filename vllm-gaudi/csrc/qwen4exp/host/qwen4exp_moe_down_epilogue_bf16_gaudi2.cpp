// SPDX-License-Identifier: Apache-2.0
// Host glue for H15 down epilogue. Torch: slot_down [S,H] BF16,
// topk_weights [T,K] BF16 (S = T*K) -> out [T,H] BF16. Index (tile128, T).
#include "qwen4exp_moe_down_epilogue_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_moe_down_epilogue_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_moe_down_epilogue_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpMoeDownEpilogueBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_moe_down_epilogue_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpMoeDownEpilogueBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& down = in->inputTensors[0].geometry;
    const auto& weights = in->inputTensors[1].geometry;
    const auto& y = in->outputTensors[0].geometry;
    if (down.dataType != DATA_BF16 || weights.dataType != DATA_BF16 ||
        y.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned width = down.maxSizes[0], slots = down.maxSizes[1];
    const unsigned topk = weights.maxSizes[0], tokens = weights.maxSizes[1];
    if (down.dims != 2 || weights.dims != 2 || y.dims != 2 || width == 0 ||
        tokens == 0 || topk == 0 || slots != tokens * topk ||
        y.maxSizes[0] != width || y.maxSizes[1] != tokens)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = (width + 127) / 128;
    out->indexSpaceGeometry[1] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = b; p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, (int)width - 1);
    map(out->inputTensorAccessPattern[0], 1, 0, 0, 0, (int)slots - 1);
    map(out->inputTensorAccessPattern[1], 0, 0, 0, 0, (int)topk - 1);
    map(out->inputTensorAccessPattern[1], 1, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_moe_down_epilogue_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_moe_down_epilogue_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
