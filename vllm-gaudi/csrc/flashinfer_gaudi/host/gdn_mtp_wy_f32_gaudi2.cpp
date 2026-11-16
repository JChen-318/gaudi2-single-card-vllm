// SPDX-License-Identifier: Apache-2.0
// Glue for the WY-form MTP verification core (kernels/gdn_mtp_wy_f32_gaudi2.c). T = 8 or 16, one ELF each.
// Inputs: 0 source  F32 [128, 128, 48, B]      gathered initial states (row b = request b)
//         1 kvals   F32 [128, 16, T, B]         raw k
//         2 log_decay F32 [48, T, B]  3 beta BF16 [48, T, B]
//         4 rmat    F32 [128, 2T, 48, B]        rows t: S0 k_t, rows T+t: S0 q_t (raw k, q)
//         5 gram    F32 [2T, 2T, 16, B]         [k; q] [k; q]^T per key head (raw)
//         6 vals    F32 [128, 48, T, B]
// Outputs: 0 output F32 [128, 48, T, B]; 1 checkpoints F32 [128, 128, 48, T, B] (as the fused core).
#include <cstring>
#include <initializer_list>
#include "gdn_mtp_wy_f32_gaudi2.hpp"

extern unsigned char _binary_gdn_mtp_wy_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_wy_f32_gaudi2_o_end;
extern unsigned char _binary_gdn_mtp_wy8_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_wy8_f32_gaudi2_o_end;

namespace {
void Map(tpc_lib_api::TensorAccessPattern& pattern, unsigned dim, unsigned indexDim, int a, int first, int last)
{
    pattern.mapping[dim].indexSpaceDim = indexDim;
    pattern.mapping[dim].a = a;
    pattern.mapping[dim].start_b = first;
    pattern.mapping[dim].end_b = last;
}
bool Shape(const tpc_lib_api::Tensor& tensor, tpc_lib_api::TensorDataType type, unsigned rank)
{
    return tensor.geometry.dataType == type && tensor.geometry.dims == rank;
}
bool Sizes(const tpc_lib_api::Tensor& tensor, std::initializer_list<uint64_t> sizes)
{
    unsigned d = 0;
    for (uint64_t s : sizes) {
        if (tensor.geometry.maxSizes[d++] != s) return false;
    }
    return true;
}
}  // namespace

tpc_lib_api::GlueCodeReturn GdnMtpWyF32Gaudi2::GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(name, "flashinfer_gaudi_gdn_mtp_wy_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnMtpWyF32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 7) {
        in->inputTensorNr = 7;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2) {
        in->outputTensorNr = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto* t = in->inputTensors;
    if (!Shape(t[0], tpc_lib_api::DATA_F32, 4) || !Shape(t[1], tpc_lib_api::DATA_F32, 4) ||
        !Shape(t[2], tpc_lib_api::DATA_F32, 3) || !Shape(t[3], tpc_lib_api::DATA_BF16, 3) ||
        !Shape(t[4], tpc_lib_api::DATA_F32, 4) || !Shape(t[5], tpc_lib_api::DATA_F32, 4) ||
        !Shape(t[6], tpc_lib_api::DATA_F32, 4)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const uint64_t batch = t[1].geometry.maxSizes[3];
    const uint64_t tokens = t[1].geometry.maxSizes[2];
    if (batch == 0 || batch > 16 || (tokens != 8 && tokens != 16) ||
        !Sizes(t[0], {128, 128, 48, batch}) || !Sizes(t[1], {128, 16, tokens, batch}) ||
        !Sizes(t[2], {48, tokens, batch}) || !Sizes(t[3], {48, tokens, batch}) ||
        !Sizes(t[4], {128, 2 * tokens, 48, batch}) || !Sizes(t[5], {2 * tokens, 2 * tokens, 16, batch}) ||
        !Sizes(t[6], {128, 48, tokens, batch})) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    const uint64_t outputSizes[2][5] = {{128, 48, tokens, batch, 0}, {128, 128, 48, tokens, batch}};
    bool wrongOutput = false;
    for (unsigned i = 0; i < 2; ++i) {
        auto& geometry = in->outputTensors[i].geometry;
        wrongOutput |= !Shape(in->outputTensors[i], tpc_lib_api::DATA_F32, 4 + i);
        geometry.dataType = tpc_lib_api::DATA_F32;
        geometry.dims = 4 + i;
        for (unsigned d = 0; d < geometry.dims; ++d) {
            wrongOutput |= geometry.maxSizes[d] != outputSizes[i][d];
            geometry.maxSizes[d] = outputSizes[i][d];
        }
    }
    if (wrongOutput) return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;

    // One point per (value head, request); every tile and token runs inside.
    out->indexSpaceRank = 4;
    out->indexSpaceGeometry[0] = 1;
    out->indexSpaceGeometry[1] = 1;
    out->indexSpaceGeometry[2] = 48;
    out->indexSpaceGeometry[3] = batch;
    for (unsigned i = 0; i < 7; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    const int lastToken = static_cast<int>(tokens) - 1;
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[0], 1, 2, 1, 0, 0);
    Map(out->outputTensorAccessPattern[0], 2, 0, 0, 0, lastToken);
    Map(out->outputTensorAccessPattern[0], 3, 3, 1, 0, 0);
    Map(out->outputTensorAccessPattern[1], 0, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[1], 1, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[1], 2, 2, 1, 0, 0);
    Map(out->outputTensorAccessPattern[1], 3, 0, 0, 0, lastToken);
    Map(out->outputTensorAccessPattern[1], 4, 3, 1, 0, 0);
    out->kernel.paramsNr = 0;
    unsigned char* elf = tokens == 16 ? &_binary_gdn_mtp_wy_f32_gaudi2_o_start
                                      : &_binary_gdn_mtp_wy8_f32_gaudi2_o_start;
    unsigned char* elfEnd = tokens == 16 ? &_binary_gdn_mtp_wy_f32_gaudi2_o_end
                                         : &_binary_gdn_mtp_wy8_f32_gaudi2_o_end;
    const unsigned size = elfEnd - elf;
    const unsigned provided = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (provided < size) return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, elf, size);
    return tpc_lib_api::GLUE_SUCCESS;
}
