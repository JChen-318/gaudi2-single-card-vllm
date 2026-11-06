// SPDX-License-Identifier: Apache-2.0
// Host glue for H15 shared-expert epilogue. Torch: merged [T, 2*I+1] BF16
// ([gate | up | shared_logit]) -> act [T, I] BF16. Index space (tile128, T).
#include "qwen4exp_moe_shared_swiglu_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_moe_shared_swiglu_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_moe_shared_swiglu_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpMoeSharedSwigluBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpMoeSharedSwigluBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 1) { in->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& merged = in->inputTensors[0].geometry;
    const auto& y = in->outputTensors[0].geometry;
    if (merged.dataType != DATA_BF16 || y.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned total = merged.maxSizes[0], tokens = merged.maxSizes[1];
    const unsigned inter = y.maxSizes[0];
    if (merged.dims != 2 || y.dims != 2 || inter == 0 || tokens == 0 ||
        total != 2 * inter + 1 || y.maxSizes[1] != tokens)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = (inter + 127) / 128;
    out->indexSpaceGeometry[1] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = b; p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, (int)total - 1);
    map(out->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_moe_shared_swiglu_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_moe_shared_swiglu_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
