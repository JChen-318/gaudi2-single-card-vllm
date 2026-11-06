// SPDX-License-Identifier: Apache-2.0
// Host GlueCode for the MoE prefill bucketing step 3 (scatter into expert-major
// bucket rows + inverse place map).
//
// Torch tensors:
//   ids     [S]        I32  routed expert per slot, < 0 = skip
//   offsets [blocks,E] I32  step 2 (torch order; TPC view [E, blocks])
//   bucket  [E*cap]   I32  out, row e*cap+m holds a slot of expert e
//   place   [S]       I32  out, bucket row of the slot, -1 = skipped
// Index space: (blocks).
// Node params: { int32 experts, int32 slots, int32 cap } - the kernel's scalar
// arguments, in this order. cap >= slots is the per-expert row stride of the
// bucket; only the first totals[e] rows of expert e are meaningful, the rest is
// stale scratch and never read.
//
// Caps: 64 <= E <= 4096, E % 64 == 0, 1 <= S <= 65536, cap >= S,
// blocks == ceil(S/64).
#include "qwen4exp_nvfp4_moe_scatter_i32_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_common.hpp"

#include <cstring>

extern unsigned char _binary___qwen4exp_nvfp4_moe_scatter_i32_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_nvfp4_moe_scatter_i32_gaudi2_o_end;

namespace {
struct ScatterParams {
    int32_t experts;
    int32_t slots;
    int32_t cap;
};
}  // namespace

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4MoeScatterI32Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4MoeScatterI32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    using namespace qwen4exp_moe;
    if (in->inputTensorNr != 2) {
        in->inputTensorNr = 2;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2) {
        in->outputTensorNr = 2;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    ScatterParams params{};
    if (!read_params(in->nodeParams, &params)) {
        return GLUE_MISSING_PRIVATE_STRUCTURE;
    }
    auto& ids = in->inputTensors[0].geometry;
    auto& offsets = in->inputTensors[1].geometry;
    auto& bucket = in->outputTensors[0].geometry;
    auto& place = in->outputTensors[1].geometry;
    if (ids.dataType != DATA_I32 || offsets.dataType != DATA_I32) {
        ids.dataType = DATA_I32;
        offsets.dataType = DATA_I32;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (bucket.dataType != DATA_I32 || place.dataType != DATA_I32) {
        bucket.dataType = DATA_I32;
        place.dataType = DATA_I32;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t slots = ids.maxSizes[0];
    const uint64_t experts = static_cast<uint64_t>(params.experts);
    const uint64_t cap = static_cast<uint64_t>(params.cap);
    if (params.experts < QWEN4EXP_MOE_BLOCK_SLOTS || params.experts > 4096 ||
        params.experts % QWEN4EXP_MOE_BLOCK_SLOTS != 0) {
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    if (params.slots < 1 || params.slots > 65536 || params.cap < params.slots) {
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    const uint64_t blocks =
        (slots + QWEN4EXP_MOE_BLOCK_SLOTS - 1) / QWEN4EXP_MOE_BLOCK_SLOTS;
    if (ids.dims != 1 || slots != static_cast<uint64_t>(params.slots)) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (offsets.dims != 2 || offsets.maxSizes[0] != experts ||
        offsets.maxSizes[1] != blocks) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (bucket.dims != 1 || bucket.maxSizes[0] != experts * cap) {
        set_shape(in->outputTensors[0], 1, DATA_I32, {experts * cap});
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    if (place.dims != 1 || place.maxSizes[0] != slots) {
        set_shape(in->outputTensors[1], 1, DATA_I32, {slots});
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    index_space(out, 1, {blocks});
    // A tail block keeps its scalar reads inside [slots-64, slots) and its writes inside the
    // block, so the touched range is the whole slot list / one block column of offsets.
    map_dim(out->inputTensorAccessPattern[0], 0, 0, 1, 0,
            static_cast<float>(slots) - 1);
    auto& p_offsets = out->inputTensorAccessPattern[1];
    map_dim(p_offsets, 0, 0, 0, 0, static_cast<float>(experts) - 1);
    map_dim(p_offsets, 1, 0, 1, 0, 0);
    map_dim(out->outputTensorAccessPattern[0], 0, 0, 0, 0,
            static_cast<float>(experts * cap) - 1);
    map_dim(out->outputTensorAccessPattern[1], 0, 0, 1, 0,
            static_cast<float>(slots) - 1);
    write_params(out, params);
    return install_elf(out, &_binary___qwen4exp_nvfp4_moe_scatter_i32_gaudi2_o_start,
                       &_binary___qwen4exp_nvfp4_moe_scatter_i32_gaudi2_o_end);
}
