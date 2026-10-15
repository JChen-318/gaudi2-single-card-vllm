// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include "gdn_mtp_fused_f32_gaudi2.hpp"

extern unsigned char _binary_gdn_mtp_fused_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_fused_f32_gaudi2_o_end;

namespace {
void Map(tpc_lib_api::TensorAccessPattern& pattern, unsigned dim, unsigned indexDim,
         int a, int first, int last)
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
}

tpc_lib_api::GlueCodeReturn GdnMtpFusedF32Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME])
{
    std::strcpy(name, "flashinfer_gaudi_gdn_mtp_fused_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GdnMtpFusedF32Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 6) {
        in->inputTensorNr = 6;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 2) {
        in->outputTensorNr = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const auto* t = in->inputTensors;
    const uint64_t batch = t[1].geometry.maxSizes[2];
    if (!Shape(t[0], tpc_lib_api::DATA_F32, 4) || !Shape(t[1], tpc_lib_api::DATA_BF16, 3) ||
        !Shape(t[2], tpc_lib_api::DATA_F32, 3) || !Shape(t[3], tpc_lib_api::DATA_BF16, 3) ||
        !Shape(t[4], tpc_lib_api::DATA_I32, 2) || !Shape(t[5], tpc_lib_api::DATA_I32, 1)) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (batch == 0 || batch > 16 || t[0].geometry.maxSizes[0] != 128 || t[0].geometry.maxSizes[1] != 128 ||
        t[0].geometry.maxSizes[2] != 48 || t[0].geometry.maxSizes[3] == 0 ||
        t[1].geometry.maxSizes[0] != 10240 || t[1].geometry.maxSizes[1] != 8 ||
        t[4].geometry.maxSizes[0] != 8 || t[4].geometry.maxSizes[1] != batch ||
        t[5].geometry.maxSizes[0] != batch) {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    for (unsigned i = 2; i < 4; ++i) {
        const auto& gate = t[i].geometry;
        if (gate.maxSizes[0] != 48 || gate.maxSizes[1] != 8 || gate.maxSizes[2] != batch)
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    const uint64_t outputSizes[2][5] = {{128, 48, 8, batch, 0}, {128, 128, 48, 8, batch}};
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
    // The state slot is data dependent; the small inputs are read whole.
    for (unsigned i = 0; i < 6; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[0], 1, 2, 1, 0, 0);
    Map(out->outputTensorAccessPattern[0], 2, 0, 0, 0, 7);
    Map(out->outputTensorAccessPattern[0], 3, 3, 1, 0, 0);
    Map(out->outputTensorAccessPattern[1], 0, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[1], 1, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[1], 2, 2, 1, 0, 0);
    Map(out->outputTensorAccessPattern[1], 3, 0, 0, 0, 7);
    Map(out->outputTensorAccessPattern[1], 4, 3, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const unsigned size = &_binary_gdn_mtp_fused_f32_gaudi2_o_end - &_binary_gdn_mtp_fused_f32_gaudi2_o_start;
    const unsigned provided = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (provided < size) return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary_gdn_mtp_fused_f32_gaudi2_o_start, size);
    return tpc_lib_api::GLUE_SUCCESS;
}
