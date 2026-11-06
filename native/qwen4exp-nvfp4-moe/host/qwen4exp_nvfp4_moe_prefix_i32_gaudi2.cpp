// SPDX-License-Identifier: Apache-2.0
// Host GlueCode for the MoE prefill bucketing step 2 (per-expert prefix + totals).
//
// Torch tensors:
//   counts  [blocks,E] I32
//   offsets [blocks,E] I32  out, exclusive running count over blocks
//   totals  [E]        I32  out, routed slots per expert
// (torch order; the TPC view is [E, blocks] with the expert axis contiguous, so
// one 64-lane vector holds 64 experts of the same block.)
// Index space: (E/64) - 64 experts per member, one vector per block.
// No node params: every size is in the tensor geometry.
//
// Caps: 64 <= E <= 4096 and E % 64 == 0, blocks >= 1.
#include "qwen4exp_nvfp4_moe_prefix_i32_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_common.hpp"

#include <cstring>

extern unsigned char _binary___qwen4exp_nvfp4_moe_prefix_i32_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_nvfp4_moe_prefix_i32_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4MoePrefixI32Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4MoePrefixI32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    using namespace qwen4exp_moe;
    if (in->inputTensorNr != 1) {
        in->inputTensorNr = 1;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2) {
        in->outputTensorNr = 2;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    auto& counts = in->inputTensors[0].geometry;
    auto& offsets = in->outputTensors[0].geometry;
    auto& totals = in->outputTensors[1].geometry;
    if (counts.dataType != DATA_I32) {
        counts.dataType = DATA_I32;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (offsets.dataType != DATA_I32 || totals.dataType != DATA_I32) {
        offsets.dataType = DATA_I32;
        totals.dataType = DATA_I32;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t experts = counts.maxSizes[0];
    const uint64_t blocks = counts.maxSizes[1];
    if (counts.dims != 2 || blocks < 1 || blocks > 1024 ||
        experts < QWEN4EXP_MOE_BLOCK_SLOTS || experts > 4096 ||
        experts % QWEN4EXP_MOE_BLOCK_SLOTS != 0) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (offsets.dims != 2 || offsets.maxSizes[0] != experts ||
        offsets.maxSizes[1] != blocks) {
        set_shape(in->outputTensors[0], 2, DATA_I32, {experts, blocks});
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    if (totals.dims != 1 || totals.maxSizes[0] != experts) {
        set_shape(in->outputTensors[1], 1, DATA_I32, {experts});
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    index_space(out, 1, {experts / QWEN4EXP_MOE_BLOCK_SLOTS});
    auto& p_counts = out->inputTensorAccessPattern[0];
    map_dim(p_counts, 0, 0, QWEN4EXP_MOE_BLOCK_SLOTS, 0,
            static_cast<float>(QWEN4EXP_MOE_BLOCK_SLOTS) - 1);
    map_dim(p_counts, 1, 0, 1, 0, static_cast<float>(blocks) - 1);
    auto& p_offsets = out->outputTensorAccessPattern[0];
    map_dim(p_offsets, 0, 0, QWEN4EXP_MOE_BLOCK_SLOTS, 0,
            static_cast<float>(QWEN4EXP_MOE_BLOCK_SLOTS) - 1);
    map_dim(p_offsets, 1, 0, 1, 0, static_cast<float>(blocks) - 1);
    map_dim(out->outputTensorAccessPattern[1], 0, 0, QWEN4EXP_MOE_BLOCK_SLOTS, 0,
            static_cast<float>(QWEN4EXP_MOE_BLOCK_SLOTS) - 1);
    out->kernel.paramsNr = 0;
    return install_elf(out, &_binary___qwen4exp_nvfp4_moe_prefix_i32_gaudi2_o_start,
                       &_binary___qwen4exp_nvfp4_moe_prefix_i32_gaudi2_o_end);
}
