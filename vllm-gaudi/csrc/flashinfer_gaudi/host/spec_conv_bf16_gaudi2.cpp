// SPDX-License-Identifier: Apache-2.0
// Glue for kernels/spec_conv_bf16_gaudi2.c.
// Inputs: 0 x BF16 [C, B*T]; 1 pool BF16 [C, L, S]; 2 w BF16 [C, 4]; 3 idx I32 [B]; 4 acc I32 [B]
// Outputs: 0 y BF16 [C, B*T]; 1 state BF16 [C, L, B]  with L = T + 2, C % 128 == 0.
#include <cstring>
#include "spec_conv_bf16_gaudi2.hpp"

extern unsigned char _binary_spec_conv_bf16_gaudi2_o_start;
extern unsigned char _binary_spec_conv_bf16_gaudi2_o_end;

namespace {
void Map(tpc_lib_api::TensorAccessPattern& pattern, unsigned dim, unsigned indexDim, int a, int first, int last)
{
    pattern.mapping[dim].indexSpaceDim = indexDim;
    pattern.mapping[dim].a = a;
    pattern.mapping[dim].start_b = first;
    pattern.mapping[dim].end_b = last;
}
}  // namespace

tpc_lib_api::GlueCodeReturn SpecConvBf16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out)
{
    using namespace tpc_lib_api;
    if (!in || !out) return GLUE_FAILED;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = in->inputTensors[0].geometry;
    const auto& pool = in->inputTensors[1].geometry;
    const auto& w = in->inputTensors[2].geometry;
    const auto& idx = in->inputTensors[3].geometry;
    const auto& acc = in->inputTensors[4].geometry;
    if (x.dataType != DATA_BF16 || pool.dataType != DATA_BF16 || w.dataType != DATA_BF16 ||
        idx.dataType != DATA_I32 || acc.dataType != DATA_I32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (x.dims != 2 || pool.dims != 3 || w.dims != 2 || idx.dims != 1 || acc.dims != 1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t C = x.maxSizes[0], N = x.maxSizes[1], L = pool.maxSizes[1], B = idx.maxSizes[0];
    if (C == 0 || C % 128 || B == 0 || L < 3 || pool.maxSizes[0] != C || w.maxSizes[0] != C || w.maxSizes[1] != 4 ||
        acc.maxSizes[0] != B || N != B * (L - 2))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t outSizes[2][3] = {{C, N, 0}, {C, L, B}};
    bool wrong = false;
    for (unsigned i = 0; i < 2; ++i) {
        auto& g = in->outputTensors[i].geometry;
        wrong |= g.dataType != DATA_BF16 || g.dims != 2 + i;
        g.dataType = DATA_BF16;
        g.dims = 2 + i;
        for (unsigned d = 0; d < g.dims; ++d) {
            wrong |= g.maxSizes[d] != outSizes[i][d];
            g.maxSizes[d] = outSizes[i][d];
        }
    }
    if (wrong) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = C / 128;
    out->indexSpaceGeometry[1] = B;
    for (unsigned i = 0; i < 5; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    const int T = static_cast<int>(L - 2);
    Map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
    Map(out->outputTensorAccessPattern[0], 1, 1, T, 0, T - 1);
    Map(out->outputTensorAccessPattern[1], 0, 0, 128, 0, 127);
    Map(out->outputTensorAccessPattern[1], 1, 0, 0, 0, static_cast<int>(L) - 1);
    Map(out->outputTensorAccessPattern[1], 2, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const unsigned size = &_binary_spec_conv_bf16_gaudi2_o_end - &_binary_spec_conv_bf16_gaudi2_o_start;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary_spec_conv_bf16_gaudi2_o_start, size);
    return GLUE_SUCCESS;
}
