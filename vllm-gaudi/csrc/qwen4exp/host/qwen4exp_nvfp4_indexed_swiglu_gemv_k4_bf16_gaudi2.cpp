// SPDX-License-Identifier: Apache-2.0
// H31 k4 layout: activation whole 512-value tiles nibble-major (see the kernel header).
// Host glue for the H15 fused NVFP4 gate+up SwiGLU GEMV.
//
// Torch tensors (Synapse reverses them, FCD first):
//   expert_ids  [S]        I32
//   gate_packed [E,N,K/2]  U8    gate_scales [E,N,K/16] U8, gate_scale2 [E] F32
//   up_packed   [E,N,K/2]  U8    up_scales   [E,N,K/16] U8, up_scale2   [E] F32
//   x           [S,K] BF16 per-slot, or [T,K] BF16 per-token (S = T * top_k)
//   output      [S,N]      BF16  silu(gate_out) * up_out, neither materialized
// Index space (ceil(N/4), S): one member owns one slot and four output rows,
// accumulating gate and up together over a shared activation tile.
#include "qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2.hpp"
#include <cstring>
#include <initializer_list>

extern unsigned char _binary___qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4IndexedSwigluGemvK4BF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4IndexedSwigluGemvK4BF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 8) {
        in->inputTensorNr = 8;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const TensorDataType in_types[] = {DATA_I32, DATA_U8, DATA_U8, DATA_F32,
                                       DATA_U8, DATA_U8, DATA_F32, DATA_BF16};
    for (unsigned i = 0; i < 8; ++i) {
        if (in->inputTensors[i].geometry.dataType != in_types[i]) {
            in->inputTensors[i].geometry.dataType = in_types[i];
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    auto& ids = in->inputTensors[0].geometry;
    auto& gate_packed = in->inputTensors[1].geometry;
    auto& gate_scales = in->inputTensors[2].geometry;
    auto& gate_scale2 = in->inputTensors[3].geometry;
    auto& up_packed = in->inputTensors[4].geometry;
    auto& up_scales = in->inputTensors[5].geometry;
    auto& up_scale2 = in->inputTensors[6].geometry;
    auto& activation = in->inputTensors[7].geometry;
    auto& output = in->outputTensors[0].geometry;
    if (output.dataType != DATA_BF16) {
        output.dataType = DATA_BF16;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const unsigned k_half = gate_packed.maxSizes[0];
    const unsigned rows = gate_packed.maxSizes[1];
    const unsigned experts = gate_packed.maxSizes[2];
    const unsigned slots = ids.maxSizes[0];
    const unsigned x_rows = activation.maxSizes[1];
    auto plane_ok = [&](const TensorGeometry& packed, const TensorGeometry& scales,
                        const TensorGeometry& scale2) {
        return packed.dims == 3 && scales.dims == 3 && scale2.dims == 1 &&
               packed.maxSizes[0] == k_half && packed.maxSizes[1] == rows &&
               packed.maxSizes[2] == experts && scales.maxSizes[0] * 8 == k_half &&
               scales.maxSizes[1] == rows && scales.maxSizes[2] == experts &&
               scale2.maxSizes[0] == experts;
    };
    // H31: the kernel keeps the activation row in local memory, 20 vectors: K <= 2560.
    if (ids.dims != 1 || slots == 0 || slots > 512 || k_half == 0 || k_half % 256 || k_half > 1280 ||
        rows == 0 || experts == 0 ||
        !plane_ok(gate_packed, gate_scales, gate_scale2) ||
        !plane_ok(up_packed, up_scales, up_scale2) ||
        activation.dims != 2 || activation.maxSizes[0] != k_half * 2 ||
        x_rows == 0 || x_rows > slots || slots % x_rows != 0) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (output.dims != 2 || output.maxSizes[0] != rows || output.maxSizes[1] != slots) {
        output.dims = 2;
        output.maxSizes[0] = rows;
        output.maxSizes[1] = slots;
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    constexpr unsigned kRowsPerMember = 4;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = (rows + kRowsPerMember - 1) / kRowsPerMember;
    out->indexSpaceGeometry[1] = slots;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned index, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = index;
        p.mapping[dim].a = a;
        p.mapping[dim].start_b = b;
        p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 1, 1, 0, 0);
    for (unsigned i : {1u, 2u, 4u, 5u}) {
        auto& p = out->inputTensorAccessPattern[i];
        map(p, 0, 0, 0, 0, static_cast<int>(i == 1 || i == 4 ? k_half : k_half / 8) - 1);
        map(p, 1, 0, static_cast<int>(kRowsPerMember), 0, 0);
        map(p, 2, 1, 0, 0, static_cast<int>(experts) - 1);
    }
    for (unsigned i : {3u, 6u})
        map(out->inputTensorAccessPattern[i], 0, 1, 0, 0, static_cast<int>(experts) - 1);
    map(out->inputTensorAccessPattern[7], 0, 0, 0, 0, static_cast<int>(k_half * 2) - 1);
    if (x_rows == slots) {
        map(out->inputTensorAccessPattern[7], 1, 1, 1, 0, 0);
    } else {
        map(out->inputTensorAccessPattern[7], 1, 1, 0, 0, static_cast<int>(x_rows) - 1);
    }
    map(out->outputTensorAccessPattern[0], 0, 0, static_cast<int>(kRowsPerMember), 0, 0);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
