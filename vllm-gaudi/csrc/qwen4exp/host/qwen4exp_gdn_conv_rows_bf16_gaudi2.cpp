// SPDX-License-Identifier: Apache-2.0
// Host glue for H14 gdn_conv_rows (prefill). Torch: mixed [T,C] BF16,
// conv_pool [Slots,3,C] BF16, weight_t [4,C] BF16, indices [S] I32,
// has_init [S] I32, state_stop [S+1] I32 (query_start_loc: cumsum of the
// *actual* query lengths) -> output [T,C] BF16, next_state [S,3,C] BF16.
// Index space (C/128, ceil(tokens_per_seq/32), S).
#include "qwen4exp_gdn_conv_rows_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_gdn_conv_rows_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_gdn_conv_rows_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpGdnConvRowsBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_gdn_conv_rows_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpGdnConvRowsBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 6) { in->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = in->inputTensors[0].geometry;
    const auto& pool = in->inputTensors[1].geometry;
    const auto& w = in->inputTensors[2].geometry;
    const auto& idx = in->inputTensors[3].geometry;
    const auto& hi = in->inputTensors[4].geometry;
    const auto& stops = in->inputTensors[5].geometry;
    const auto& y = in->outputTensors[0].geometry;
    const auto& nxt = in->outputTensors[1].geometry;
    if (x.dataType != DATA_BF16 || pool.dataType != DATA_BF16 || w.dataType != DATA_BF16 ||
        idx.dataType != DATA_I32 || hi.dataType != DATA_I32 || stops.dataType != DATA_I32 ||
        y.dataType != DATA_BF16 || nxt.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned C = x.maxSizes[0], T = x.maxSizes[1], S = idx.maxSizes[0];
    const unsigned slots = pool.maxSizes[2];
    if (x.dims != 2 || pool.dims != 3 || w.dims != 2 || idx.dims != 1 || hi.dims != 1 ||
        stops.dims != 1 || y.dims != 2 || nxt.dims != 3 || C == 0 || C % 128 || T == 0 || S == 0 ||
        T % S || hi.maxSizes[0] != S || stops.maxSizes[0] != S + 1 || pool.maxSizes[0] != C ||
        pool.maxSizes[1] != 3 || slots == 0 || w.maxSizes[0] != C || w.maxSizes[1] != 4 ||
        y.maxSizes[0] != C || y.maxSizes[1] != T ||
        nxt.maxSizes[0] != C || nxt.maxSizes[1] != 3 || nxt.maxSizes[2] != S)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    constexpr unsigned kRowBlock = 32;
    const unsigned tps = T / S;
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = C / 128;
    out->indexSpaceGeometry[1] = (tps + kRowBlock - 1) / kRowBlock;
    out->indexSpaceGeometry[2] = S;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx_, int aa, int bb, int end) {
        p.mapping[dim].indexSpaceDim = idx_; p.mapping[dim].a = aa;
        p.mapping[dim].start_b = bb; p.mapping[dim].end_b = end;
    };
    for (unsigned i = 0; i <= 1; ++i) {
        auto& p = i == 0 ? out->inputTensorAccessPattern[0] : out->outputTensorAccessPattern[0];
        map(p, 0, 0, 128, 0, 127);
        map(p, 1, 1, 0, 0, (int)T - 1);
    }
    map(out->inputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[1], 1, 0, 0, 0, 2);
    map(out->inputTensorAccessPattern[1], 2, 2, 0, 0, (int)slots - 1);
    map(out->inputTensorAccessPattern[2], 0, 0, 128, 0, 127);
    map(out->inputTensorAccessPattern[2], 1, 0, 0, 0, 3);
    map(out->inputTensorAccessPattern[3], 0, 2, 1, 0, 0);
    map(out->inputTensorAccessPattern[4], 0, 2, 1, 0, 0);
    // query_start_loc: element `seq` and `seq + 1` are read per sequence.
    map(out->inputTensorAccessPattern[5], 0, 2, 1, 0, 0);
    map(out->outputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[1], 1, 0, 0, 0, 2);
    map(out->outputTensorAccessPattern[1], 2, 2, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_gdn_conv_rows_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_gdn_conv_rows_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
