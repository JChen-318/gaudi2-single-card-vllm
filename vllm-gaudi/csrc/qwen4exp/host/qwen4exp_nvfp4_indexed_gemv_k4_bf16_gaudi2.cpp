// SPDX-License-Identifier: Apache-2.0
// H31 k4 layout: activation whole 512-value tiles nibble-major (see the kernel header).
// Host glue for the indexed NVFP4 GEMV.
//
// Torch tensors (Synapse reverses them, FCD first):
//   expert_ids [S]        I32
//   packed     [E,N,K/2]  U8
//   scales     [E,N,K/16] U8
//   x          [S,K]      BF16   per-slot activation, or
//              [T,K]      BF16   per-token activation (H10, S = T * top_k)
//   scale2     [E]        F32
//   output     [S,N]      BF16
// TPC therefore sees x [K,x_rows], packed [K/2,N,E], output [N,S].
//
// Index space (H8): (ceil(N/ROWS_PER_MEMBER), S). One TPC member owns one slot
// and four consecutive output rows, so its eight weight/scale loads for those
// rows are issued before the first MAC and its activation tile is loaded once
// and shared by the four rows (same structure as the native qwen4exp MoE
// decode GEMV). The row axis of every tensor therefore steps by
// ROWS_PER_MEMBER, and a trailing group shorter than four rows is masked by the
// kernel (it clamps the loads and guards the stores).
#include "qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4IndexedGemvK4BF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4IndexedGemvK4BF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
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
    const unsigned k_half = packed.maxSizes[0];
    const unsigned rows = packed.maxSizes[1];
    const unsigned experts = packed.maxSizes[2];
    const unsigned slots = ids.maxSizes[0];
    // H10: activation dim 1 (glue order, i.e. torch dim 0) is x_rows: either
    // one row per slot (x_rows == slots) or one row per token, in which case
    // slots must be an exact multiple of it (slots = x_rows * top_k).
    const unsigned x_rows = activation.maxSizes[1];
    if (ids.dims != 1 || slots == 0 || slots > 512 ||
        packed.dims != 3 || scales.dims != 3 || k_half == 0 || k_half % 64 ||
        rows == 0 || experts == 0 ||
        scales.maxSizes[0] * 8 != k_half || scales.maxSizes[1] != rows ||
        scales.maxSizes[2] != experts ||
        activation.dims != 2 || activation.maxSizes[0] != k_half * 2 ||
        x_rows == 0 || x_rows > slots || slots % x_rows != 0 ||
        scale2.dims != 1 || scale2.maxSizes[0] != experts) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (output.dims != 2 || output.maxSizes[0] != rows || output.maxSizes[1] != slots) {
        output.dims = 2;
        output.maxSizes[0] = rows;
        output.maxSizes[1] = slots;
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    constexpr unsigned kRowsPerMember = 4;  // ROWS_PER_MEMBER in the kernel
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
    for (unsigned i = 1; i <= 2; ++i) {
        auto& p = out->inputTensorAccessPattern[i];
        map(p, 0, 0, 0, 0, static_cast<int>(i == 1 ? k_half : k_half / 8) - 1);
        map(p, 1, 0, static_cast<int>(kRowsPerMember), 0, 0);
        map(p, 2, 1, 0, 0, static_cast<int>(experts) - 1);
    }
    map(out->inputTensorAccessPattern[3], 0, 0, 0, 0, static_cast<int>(k_half * 2) - 1);
    if (x_rows == slots) {
        map(out->inputTensorAccessPattern[3], 1, 1, 1, 0, 0);
    } else {
        // H10 broadcast: slot s reads x_row = s / top_k, which is not affine
        // (a == 1) in the slot index-space axis. Declare the full
        // [0, x_rows - 1] range (a == 0, as for the runtime expert axis) so
        // the TPC AGU/cache never clips x_row to the member's own slot.
        map(out->inputTensorAccessPattern[3], 1, 1, 0, 0, static_cast<int>(x_rows) - 1);
    }
    map(out->inputTensorAccessPattern[4], 0, 1, 0, 0, static_cast<int>(experts) - 1);
    map(out->outputTensorAccessPattern[0], 0, 0, static_cast<int>(kRowsPerMember), 0, 0);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
