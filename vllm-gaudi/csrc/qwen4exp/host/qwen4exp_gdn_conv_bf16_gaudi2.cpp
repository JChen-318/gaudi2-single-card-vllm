// SPDX-License-Identifier: Apache-2.0
// Host glue for H14 gdn_conv (decode). Torch: mixed [B,C] BF16,
// conv_pool [Slots,3,C] BF16, weight_t [4,C] BF16, indices [B] I32
// -> output [B,C] BF16, next_state [B,3,C] BF16. Index space (C/128, B).
#include "qwen4exp_gdn_conv_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_gdn_conv_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_gdn_conv_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpGdnConvBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_gdn_conv_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpGdnConvBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = in->inputTensors[0].geometry;
    const auto& pool = in->inputTensors[1].geometry;
    const auto& w = in->inputTensors[2].geometry;
    const auto& idx = in->inputTensors[3].geometry;
    const auto& y = in->outputTensors[0].geometry;
    const auto& nxt = in->outputTensors[1].geometry;
    if (x.dataType != DATA_BF16 || pool.dataType != DATA_BF16 || w.dataType != DATA_BF16 ||
        idx.dataType != DATA_I32 || y.dataType != DATA_BF16 || nxt.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned C = x.maxSizes[0], B = x.maxSizes[1], slots = pool.maxSizes[2];
    if (x.dims != 2 || pool.dims != 3 || w.dims != 2 || idx.dims != 1 || y.dims != 2 ||
        nxt.dims != 3 || C == 0 || C % 128 || B == 0 || idx.maxSizes[0] != B ||
        pool.maxSizes[0] != C || pool.maxSizes[1] != 3 || slots == 0 ||
        w.maxSizes[0] != C || w.maxSizes[1] != 4 ||
        y.maxSizes[0] != C || y.maxSizes[1] != B ||
        nxt.maxSizes[0] != C || nxt.maxSizes[1] != 3 || nxt.maxSizes[2] != B)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = C / 128;
    out->indexSpaceGeometry[1] = B;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx_, int aa, int bb, int end) {
        p.mapping[dim].indexSpaceDim = idx_; p.mapping[dim].a = aa;
        p.mapping[dim].start_b = bb; p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[1], 1, 0, 0, 0, 2);
    map(out->inputTensorAccessPattern[1], 2, 1, 0, 0, (int)slots - 1);
    map(out->inputTensorAccessPattern[2], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[2], 1, 0, 0, 0, 3);
    map(out->inputTensorAccessPattern[3], 0, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[1], 1, 0, 0, 0, 2);
    map(out->outputTensorAccessPattern[1], 2, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_gdn_conv_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_gdn_conv_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
