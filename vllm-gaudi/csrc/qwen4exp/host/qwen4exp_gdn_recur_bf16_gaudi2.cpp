// SPDX-License-Identifier: Apache-2.0
// Host glue for H14 gdn_recur. Torch tensors (TPC sees dims reversed):
//   mixed [T,C] BF16, a/b [T,Hv] BF16, A_log [Hv] F32, dt_bias [Hv] BF16,
//   state_pool [Slots,Hv,128,128] BF16, indices [S] I32
//   -> output [T,Hv,128] F32 (caller rounds to BF16), state_out BF16
// Index space (128/8 row blocks, Hv, S): one member owns 8 Dv rows.
#include "qwen4exp_gdn_recur_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___qwen4exp_gdn_recur_bf16_gaudi2_o_start;
extern unsigned char _binary___qwen4exp_gdn_recur_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Qwen4ExpGdnRecurBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, "custom_qwen4exp_gdn_recur_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Qwen4ExpGdnRecurBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 7) { in->inputTensorNr = 7; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& mixed = in->inputTensors[0].geometry;
    const auto& a = in->inputTensors[1].geometry;
    const auto& b = in->inputTensors[2].geometry;
    const auto& alog = in->inputTensors[3].geometry;
    const auto& dt = in->inputTensors[4].geometry;
    const auto& pool = in->inputTensors[5].geometry;
    const auto& idx = in->inputTensors[6].geometry;
    const auto& y = in->outputTensors[0].geometry;
    const auto& so = in->outputTensors[1].geometry;
    if (mixed.dataType != DATA_BF16 || a.dataType != DATA_BF16 || b.dataType != DATA_BF16 ||
        alog.dataType != DATA_F32 || dt.dataType != DATA_BF16 || pool.dataType != DATA_BF16 ||
        idx.dataType != DATA_I32 || y.dataType != DATA_F32 || so.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const unsigned C = mixed.maxSizes[0], T = mixed.maxSizes[1];
    const unsigned Hv = a.maxSizes[0], S = idx.maxSizes[0], slots = pool.maxSizes[3];
    if (mixed.dims != 2 || a.dims != 2 || b.dims != 2 || alog.dims != 1 || dt.dims != 1 ||
        pool.dims != 4 || idx.dims != 1 || y.dims != 3 || so.dims != 4 || C == 0 || C % 128 ||
        T == 0 || S == 0 || T % S || Hv == 0 || a.maxSizes[1] != T || b.maxSizes[0] != Hv ||
        b.maxSizes[1] != T || alog.maxSizes[0] != Hv || dt.maxSizes[0] != Hv ||
        pool.maxSizes[0] != 128 || pool.maxSizes[1] != 128 || pool.maxSizes[2] != Hv ||
        slots == 0 || (C / 128 <= Hv) || ((C / 128 - Hv) % 2) ||
        Hv % ((C / 128 - Hv) / 2) ||
        y.maxSizes[0] != 128 || y.maxSizes[1] != Hv || y.maxSizes[2] != T ||
        so.maxSizes[0] != 128 || so.maxSizes[1] != 128 || so.maxSizes[2] != Hv ||
        so.maxSizes[3] != S)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const unsigned tps = T / S;
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = 128 / 8;
    out->indexSpaceGeometry[1] = Hv;
    out->indexSpaceGeometry[2] = S;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned idx_, int aa, int bb, int end) {
        p.mapping[dim].indexSpaceDim = idx_; p.mapping[dim].a = aa;
        p.mapping[dim].start_b = bb; p.mapping[dim].end_b = end;
    };
    auto full = [&](TensorAccessPattern& p, unsigned dim, unsigned idx_, int size) {
        map(p, dim, idx_, 0, 0, (int)size - 1);
    };
    full(out->inputTensorAccessPattern[0], 0, 0, C);
    map(out->inputTensorAccessPattern[0], 1, 2, (int)tps, 0, (int)tps - 1);
    for (unsigned i = 1; i <= 2; ++i) {
        map(out->inputTensorAccessPattern[i], 0, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[i], 1, 2, (int)tps, 0, (int)tps - 1);
    }
    map(out->inputTensorAccessPattern[3], 0, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[4], 0, 1, 1, 0, 0);
    full(out->inputTensorAccessPattern[5], 0, 0, 128);
    map(out->inputTensorAccessPattern[5], 1, 0, 8, 0, 7);
    map(out->inputTensorAccessPattern[5], 2, 1, 1, 0, 0);
    full(out->inputTensorAccessPattern[5], 3, 2, slots);
    map(out->inputTensorAccessPattern[6], 0, 2, 1, 0, 0);
    map(out->outputTensorAccessPattern[0], 0, 0, 8, 0, 7);
    map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[0], 2, 2, (int)tps, 0, (int)tps - 1);
    full(out->outputTensorAccessPattern[1], 0, 0, 128);
    map(out->outputTensorAccessPattern[1], 1, 0, 8, 0, 7);
    map(out->outputTensorAccessPattern[1], 2, 1, 1, 0, 0);
    map(out->outputTensorAccessPattern[1], 3, 2, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = &_binary___qwen4exp_gdn_recur_bf16_gaudi2_o_start;
    const auto* end = &_binary___qwen4exp_gdn_recur_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
