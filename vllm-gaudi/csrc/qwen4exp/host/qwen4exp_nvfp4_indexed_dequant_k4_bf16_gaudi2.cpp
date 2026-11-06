// SPDX-License-Identifier: Apache-2.0
// Host GlueCode for the Qwen4Exp NVFP4 (E2M1 + per-16 E4M3 scales) indexed
// expert dequant kernel. Mirrors the DeepSeek V4 MXFP4 indexed-dequant glue.
//
// Torch-level tensors (E outermost, K/2 contiguous; matches the DeepSeek V4
// MXFP4 indexed kernels):
//   expert_ids [S]        I32  — runtime-selected expert ids (-1 -> zero fill)
//   packed     [E,N,K/2]  U8   — packed nibbles, low nibble first
//   scales     [E,N,K/16] U8   — E4M3 block scales, 16-element groups
//   output     [S,N,K]    BF16 — dequantized rows of the selected experts
//
// Synapse passes geometry outermost-dimension-first, i.e. maxSizes reversed
// versus torch: packed arrives as [K/2,N,E] — the same order the TPC kernel
// indexes (kernel dim 0 = torch's contiguous K/2 axis). All checks and
// access-pattern dims below use that glue order.
//
// Index space (tiles, N rows, S slots) with tiles = ceil(K/2 / 256): the
// expert id is a runtime value, so the packed/scale expert axes declare the
// full conservative source range; only the indexed expert is actually loaded
// per slot. K must be a multiple of 128 (Qwen4Exp uses 1280 and 2560).
#include "qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4IndexedDequantK4BF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpNvfp4IndexedDequantK4BF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 4) {
        in->inputTensorNr = 4;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const TensorDataType in_types[] = {DATA_I32, DATA_U8, DATA_U8, DATA_F32};
    for (unsigned i = 0; i < 4; ++i) {
        if (in->inputTensors[i].geometry.dataType != in_types[i]) {
            in->inputTensors[i].geometry.dataType = in_types[i];
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    // Synapse geometry is outermost-dimension-first: torch [E,N,K/2]
    // arrives as maxSizes = [K/2,N,E] (the TPC kernel's dim order, FCD first).
    auto& ids = in->inputTensors[0].geometry;
    auto& packed = in->inputTensors[1].geometry;
    auto& scales = in->inputTensors[2].geometry;
    auto& scale2 = in->inputTensors[3].geometry;
    auto& output = in->outputTensors[0].geometry;
    if (output.dataType != DATA_BF16) {
        output.dataType = DATA_BF16;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (ids.dims != 1 || ids.maxSizes[0] == 0 ||
        packed.dims != 3 || scales.dims != 3 || scale2.dims != 1 ||
        packed.maxSizes[0] == 0 || packed.maxSizes[0] % 64 ||
        packed.maxSizes[1] == 0 || packed.maxSizes[2] == 0 ||
        packed.maxSizes[0] != scales.maxSizes[0] * 8 ||
        packed.maxSizes[1] != scales.maxSizes[1] ||
        packed.maxSizes[2] != scales.maxSizes[2] ||
        scale2.maxSizes[0] != packed.maxSizes[2]) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (output.dims != 3 || output.maxSizes[0] != packed.maxSizes[0] * 2 ||
        output.maxSizes[1] != packed.maxSizes[1] ||
        output.maxSizes[2] != ids.maxSizes[0]) {
        output.dims = 3;
        output.maxSizes[0] = packed.maxSizes[0] * 2;
        output.maxSizes[1] = packed.maxSizes[1];
        output.maxSizes[2] = ids.maxSizes[0];
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    constexpr unsigned kRowsPerMember = 4;  // ROWS_PER_MEMBER in the kernel
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = (packed.maxSizes[0] + 255) / 256;  // ceil tiles
    out->indexSpaceGeometry[1] = (packed.maxSizes[1] + kRowsPerMember - 1) / kRowsPerMember;
    out->indexSpaceGeometry[2] = ids.maxSizes[0];
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned index, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = index;
        p.mapping[dim].a = a;
        p.mapping[dim].start_b = b;
        p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 2, 1, 0, 0);
    for (unsigned i = 1; i <= 2; ++i) {
        auto& p = out->inputTensorAccessPattern[i];
        map(p, 0, 0, i == 1 ? 256 : 32, 0, 255);
        // H27: one member owns kRowsPerMember consecutive output rows.
        map(p, 1, 1, kRowsPerMember, 0, 0);
        // Runtime expert IDs are not affine in the slot. Declare the
        // conservative source range; only the indexed expert is loaded.
        map(p, 2, 2, 0, 0, packed.maxSizes[2] - 1);
    }
    // scale2 is read once per slot at the runtime expert id.
    map(out->inputTensorAccessPattern[3], 0, 2, 0, 0, scale2.maxSizes[0] - 1);
    auto& p = out->outputTensorAccessPattern[0];
    map(p, 0, 0, 512, 0, 511);
    map(p, 1, 1, kRowsPerMember, 0, 0);
    map(p, 2, 2, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
