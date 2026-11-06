// SPDX-License-Identifier: Apache-2.0
// Host glue for H13 hc_down_epilogue. Torch: merged [T, M] ->
// (low [T, lowrank], inj [T, streams]). Index space (tile128 over lowrank, token).
#include "qwen4exp_hc_down_epilogue_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_hc_down_epilogue_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_hc_down_epilogue_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpHcDownEpilogueBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_hc_down_epilogue_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpHcDownEpilogueBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 1) { in->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& m = in->inputTensors[0].geometry;
    const auto& lo = in->outputTensors[0].geometry;
    const auto& ij = in->outputTensors[1].geometry;
    if (m.dataType != DATA_BF16 || lo.dataType != DATA_BF16 || ij.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned merged = m.maxSizes[0], tokens = m.maxSizes[1];
    const unsigned lowrank = lo.maxSizes[0], streams = ij.maxSizes[0];
    if (m.dims != 2 || lo.dims != 2 || ij.dims != 2 || merged == 0 || tokens == 0 ||
        lowrank == 0 || lowrank > merged || streams == 0 ||
        lo.maxSizes[1] != tokens || ij.maxSizes[1] != tokens) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = (lowrank + 127) / 128;
    out->indexSpaceGeometry[1] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int bb, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = bb; p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, (int)merged - 1);
    map(out->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[1], 0, 0, 0, 0, (int)streams - 1);
    map(out->outputTensorAccessPattern[1], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_hc_down_epilogue_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_hc_down_epilogue_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
