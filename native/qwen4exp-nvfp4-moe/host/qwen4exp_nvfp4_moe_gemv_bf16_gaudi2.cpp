// SPDX-License-Identifier: Apache-2.0
// Host GlueCode for the grouped NVFP4 MoE decode GEMV.
//
// Torch tensors (the order the operator takes them):
//   expert_ids [S]        I32  selected expert per routed slot
//   packed     [E,N,K/2]  U8   E2M1 nibbles, low nibble = even k
//   scales     [E,N,K/16] U8   per-16 E4M3 block scales
//   x          [S,K]      BF16 routed activations
//   scale2     [E]        F32  per-expert global scale
//   output     [S,N]      BF16
// Synapse reverses them (FCD first): x [K,S], packed [K/2,N,E],
// scales [K/16,N,E], output [N,S].
//
// Index space: (N/ROWS_PER_MEMBER, S). Each TPC member owns one slot and a
// group of QWEN4EXP_MOE_ROWS_PER_MEMBER output rows, so its weight loads for
// all of those rows can overlap (see the kernel header).
//
// Caps, all enforced here:
//   K % 128 == 0            (packed K/2 a multiple of 64; 128 keeps the
//                            nibble/scale packing of a dual group whole)
//   N % 4 == 0              (rows per member)
//   E in [64, 4096], E % 64 == 0 (a scale2 plane per expert)
//   1 <= S <= 64            (decode slot budget of one launch)
#include "qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_common.hpp"

#include <cstring>

extern unsigned char _binary___qwen4exp_nvfp4_moe_gemv_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_nvfp4_moe_gemv_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4MoeGemvBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4MoeGemvBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    using namespace qwen4exp_moe;
    if (in->inputTensorNr != 5) {
        in->inputTensorNr = 5;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const TensorDataType in_types[] = {DATA_I32, DATA_U8, DATA_U8, DATA_BF16, DATA_F32};
    for (unsigned i = 0; i < 5; ++i) {
        if (in->inputTensors[i].geometry.dataType != in_types[i]) {
            in->inputTensors[i].geometry.dataType = in_types[i];
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    auto& ids = in->inputTensors[0].geometry;
    auto& packed = in->inputTensors[1].geometry;
    auto& scales = in->inputTensors[2].geometry;
    auto& activation = in->inputTensors[3].geometry;
    auto& scale2 = in->inputTensors[4].geometry;
    auto& output = in->outputTensors[0].geometry;
    if (output.dataType != DATA_BF16) {
        output.dataType = DATA_BF16;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t k_half = packed.maxSizes[0];
    const uint64_t rows = packed.maxSizes[1];
    const uint64_t experts = packed.maxSizes[2];
    const uint64_t slots = ids.maxSizes[0];
    const uint64_t row_step = QWEN4EXP_MOE_ROWS_PER_MEMBER;
    const bool geometry_ok =
        ids.dims == 1 && slots >= 1 && slots <= QWEN4EXP_MOE_MAX_DECODE_SLOTS &&
        packed.dims == 3 && scales.dims == 3 &&
        k_half > 0 && k_half % (QWEN4EXP_MOE_K_PER_PART / 2) == 0 &&
        rows > 0 && rows % row_step == 0 &&
        experts >= QWEN4EXP_MOE_BLOCK_SLOTS && experts <= 4096 &&
        experts % QWEN4EXP_MOE_BLOCK_SLOTS == 0 &&
        scales.maxSizes[0] * 8 == k_half && scales.maxSizes[1] == rows &&
        scales.maxSizes[2] == experts &&
        activation.dims == 2 && activation.maxSizes[0] == k_half * 2 &&
        activation.maxSizes[1] == slots &&
        scale2.dims == 1 && scale2.maxSizes[0] == experts;
    if (!geometry_ok) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (output.dims != 2 || output.maxSizes[0] != rows || output.maxSizes[1] != slots) {
        set_shape(in->outputTensors[0], 2, DATA_BF16, {rows, slots});
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    index_space(out, 2, {rows / row_step, slots});
    // One packed/scale/activation row per index-space step of dimension 0 (row groups);
    // the K/2 and K dims are walked inside a member, so a = 0 with the full range.
    auto& p_packed = out->inputTensorAccessPattern[1];
    map_dim(p_packed, 0, 0, 0, 0, static_cast<float>(k_half) - 1);
    map_dim(p_packed, 1, 0, static_cast<float>(row_step), 0, 0);
    map_dim(p_packed, 2, 1, 0, 0, static_cast<float>(experts) - 1);
    auto& p_scales = out->inputTensorAccessPattern[2];
    map_dim(p_scales, 0, 0, 0, 0, static_cast<float>(k_half / 8) - 1);
    map_dim(p_scales, 1, 0, static_cast<float>(row_step), 0, 0);
    map_dim(p_scales, 2, 1, 0, 0, static_cast<float>(experts) - 1);
    // The expert is a runtime value: declare the conservative source range, only the
    // indexed expert is read.
    map_dim(out->inputTensorAccessPattern[0], 0, 1, 1, 0, 0);
    map_dim(out->inputTensorAccessPattern[3], 0, 0, 0, 0,
            static_cast<float>(k_half * 2) - 1);
    map_dim(out->inputTensorAccessPattern[3], 1, 1, 1, 0, 0);
    map_dim(out->inputTensorAccessPattern[4], 0, 1, 0, 0, static_cast<float>(experts) - 1);
    map_dim(out->outputTensorAccessPattern[0], 0, 0, static_cast<float>(row_step), 0, 0);
    map_dim(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    return install_elf(out, &_binary___qwen4exp_nvfp4_moe_gemv_bf16_gaudi2_o_start,
                       &_binary___qwen4exp_nvfp4_moe_gemv_bf16_gaudi2_o_end);
}
