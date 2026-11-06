// SPDX-License-Identifier: Apache-2.0
// Host glue for the H29 fused PLE gate. Torch tensors:
//   key, query  [T, S*H] BF16 -> TPC (S*H, T)
//   value       [T, H]   BF16 -> TPC (H, T)
//   gamma_*     [Sg, H]  BF16 -> TPC (H, Sg)      (Sg == 1 shared or S)
//   gated, conv_input [T, S*H] BF16
// Index space (streams, tokens): one member owns one (stream, token) row of
// width H (H % 128 == 0).
#include "qwen4exp_ple_gate_bf16_gaudi2.hpp"
#include <cmath>
#include <cstring>

extern unsigned char _binary___qwen4exp_ple_gate_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_ple_gate_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpPleGateBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_ple_gate_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpPleGateBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != sizeof(float))
        return GLUE_FAILED;
    const float eps = *static_cast<const float*>(in->nodeParams.nodeParams);
    if (!(eps > 0) || !std::isfinite(eps)) return GLUE_FAILED;
    if (in->inputTensorNr != 6) { in->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& k = in->inputTensors[0].geometry;
    const auto& q = in->inputTensors[1].geometry;
    const auto& v = in->inputTensors[2].geometry;
    const auto& gk = in->inputTensors[3].geometry;
    const auto& gq = in->inputTensors[4].geometry;
    const auto& gc = in->inputTensors[5].geometry;
    const auto& y0 = in->outputTensors[0].geometry;
    const auto& y1 = in->outputTensors[1].geometry;
    if (k.dataType != DATA_BF16 || q.dataType != DATA_BF16 || v.dataType != DATA_BF16 ||
        gk.dataType != DATA_BF16 || gq.dataType != DATA_BF16 || gc.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (y0.dataType != DATA_BF16 || y1.dataType != DATA_BF16) {
        const_cast<TensorGeometry&>(y0).dataType = DATA_BF16;
        const_cast<TensorGeometry&>(y1).dataType = DATA_BF16;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const unsigned total = k.maxSizes[0], tokens = k.maxSizes[1];
    const unsigned width = v.maxSizes[0], gamma_rows = gk.maxSizes[1];
    if (k.dims != 2 || q.dims != 2 || v.dims != 2 || gk.dims != 2 || gq.dims != 2 ||
        gc.dims != 2 || y0.dims != 2 || y1.dims != 2 || width == 0 || width % 128 ||
        total == 0 || total % width || tokens == 0 ||
        (gamma_rows != 1 && gamma_rows != total / width) ||
        v.maxSizes[1] != tokens || q.maxSizes[0] != total || q.maxSizes[1] != tokens ||
        gq.maxSizes[0] != width || gq.maxSizes[1] != gamma_rows ||
        gc.maxSizes[0] != width || gc.maxSizes[1] != gamma_rows ||
        y0.maxSizes[0] != total || y0.maxSizes[1] != tokens ||
        y1.maxSizes[0] != total || y1.maxSizes[1] != tokens) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = total / width;
    out->indexSpaceGeometry[1] = tokens;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = idx; p.mapping[dim].a = a;
        p.mapping[dim].start_b = b; p.mapping[dim].end_b = end;
    };
    // key/query/gated/conv_input: the member computes its own stream base,
    // exactly like hc_norm's (a=0, full range) stream axis.
    map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, (int)total - 1);
    map(out->inputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[1], 0, 0, 0, 0, (int)total - 1);
    map(out->inputTensorAccessPattern[1], 1, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[2], 0, 0, 0, 0, (int)width - 1);
    map(out->inputTensorAccessPattern[2], 1, 1, 1, 0, 0);
    for (int i = 3; i < 6; ++i) {
        map(out->inputTensorAccessPattern[i], 0, 0, 0, 0, (int)width - 1);
        map(out->inputTensorAccessPattern[i], 1, 0, 0, 0, (int)gamma_rows - 1);
    }
    map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, (int)total - 1);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[1], 0, 0, 0, 0, (int)total - 1);
    map(out->outputTensorAccessPattern[1], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &eps, sizeof(eps));
    const auto* start = &_binary___qwen4exp_ple_gate_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_ple_gate_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
