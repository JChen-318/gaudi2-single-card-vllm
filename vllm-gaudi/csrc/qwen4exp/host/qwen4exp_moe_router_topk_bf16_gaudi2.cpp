// SPDX-License-Identifier: Apache-2.0
// Host glue for H15 router top-k. Torch: logits [T,512] BF16 ->
// ids [T,K] I32, weights [T,K] BF16. Index space (T).
#include "qwen4exp_moe_router_topk_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_moe_router_topk_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_moe_router_topk_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpMoeRouterTopkBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_moe_router_topk_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpMoeRouterTopkBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 1) { in->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& logits = in->inputTensors[0].geometry;
    const auto& ids = in->outputTensors[0].geometry;
    const auto& weights = in->outputTensors[1].geometry;
    if (logits.dataType != DATA_BF16 || ids.dataType != DATA_I32 ||
        weights.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned experts = logits.maxSizes[0], tokens = logits.maxSizes[1];
    const unsigned topk = ids.maxSizes[0];
    if (logits.dims != 2 || ids.dims != 2 || weights.dims != 2 || experts != 512 ||
        tokens == 0 || topk == 0 || topk > 64 || weights.maxSizes[0] != topk ||
        ids.maxSizes[1] != tokens || weights.maxSizes[1] != tokens)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = b; p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, (int)experts - 1);
    map(out->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    for (unsigned i = 0; i < 2; ++i) {
        map(out->outputTensorAccessPattern[i], 0, 0, 0, 0, (int)topk - 1);
        map(out->outputTensorAccessPattern[i], 1, 0, 1, 0, 0);
    }
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_moe_router_topk_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_moe_router_topk_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
