// SPDX-License-Identifier: Apache-2.0
// Host GlueCode for the MoE prefill bucketing step 1 (counts per 64-slot block).
//
// Torch tensors:
//   ids    [S]          I32  routed expert per slot, < 0 = skip (shared slot,
//                            padding, dropped token)
//   counts [blocks, E]  I32  out, blocks = ceil(S/64); the TPC view is
//                            [E, blocks] because the expert axis is contiguous
// Index space: (blocks, E/64).
// Node params: { int32 experts } - the operator's int argument, which also
// fixes the output shape before the graph compiler knows the tensor.
//
// Caps: 64 <= E <= 4096 and E % 64 == 0 (one 64-lane vector = 64 experts),
// 1 <= S <= 65536.
#include "qwen4exp_nvfp4_moe_count_i32_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_common.hpp"

#include <cstring>

extern unsigned char _binary___qwen4exp_nvfp4_moe_count_i32_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_nvfp4_moe_count_i32_gaudi2_o_end;

namespace {
struct CountParams {
    int32_t experts;
};
}  // namespace

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4MoeCountI32Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_nvfp4_moe_count_i32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4MoeCountI32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    using namespace qwen4exp_moe;
    if (in->inputTensorNr != 1) {
        in->inputTensorNr = 1;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    CountParams params{};
    if (!read_params(in->nodeParams, &params)) {
        return GLUE_MISSING_PRIVATE_STRUCTURE;
    }
    auto& ids = in->inputTensors[0].geometry;
    auto& counts = in->outputTensors[0].geometry;
    if (ids.dataType != DATA_I32) {
        ids.dataType = DATA_I32;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (counts.dataType != DATA_I32) {
        counts.dataType = DATA_I32;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t slots = ids.maxSizes[0];
    const uint64_t experts = static_cast<uint64_t>(params.experts);
    if (ids.dims != 1 || slots < 1 || slots > 65536) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (params.experts < QWEN4EXP_MOE_BLOCK_SLOTS || params.experts > 4096 ||
        params.experts % QWEN4EXP_MOE_BLOCK_SLOTS != 0) {
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const uint64_t blocks =
        (slots + QWEN4EXP_MOE_BLOCK_SLOTS - 1) / QWEN4EXP_MOE_BLOCK_SLOTS;
    // TPC geometry: maxSizes[0] is torch's last dimension (the experts).
    if (counts.dims != 2 || counts.maxSizes[0] != experts || counts.maxSizes[1] != blocks) {
        set_shape(in->outputTensors[0], 2, DATA_I32, {experts, blocks});
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    index_space(out, 2, {blocks, experts / QWEN4EXP_MOE_BLOCK_SLOTS});
    // One 64-slot block per index-space step of dimension 0. The kernel reads the ids of
    // the block as one vector; a tail block reads the last 64 slots instead of reading past
    // the tensor (the extra lanes are masked), so the touched range is the whole slot list.
    map_dim(out->inputTensorAccessPattern[0], 0, 0, QWEN4EXP_MOE_BLOCK_SLOTS, 0,
            static_cast<float>(slots) - 1);
    // counts: dim 0 (expert) is the lane axis of the stores, dim 1 (block) the stepping axis.
    map_dim(out->outputTensorAccessPattern[0], 0, 1, QWEN4EXP_MOE_BLOCK_SLOTS, 0,
            static_cast<float>(QWEN4EXP_MOE_BLOCK_SLOTS) - 1);
    map_dim(out->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);
    write_params(out, params);
    return install_elf(out, &_binary___qwen4exp_nvfp4_moe_count_i32_gaudi2_o_start,
                       &_binary___qwen4exp_nvfp4_moe_count_i32_gaudi2_o_end);
}
