// SPDX-License-Identifier: Apache-2.0
// Small validation helpers shared by the Qwen4Exp NVFP4 MoE glue classes.
// Same conventions as vllm-gaudi/csrc/qwen4exp/host: Synapse hands tensors over
// with the dimensions reversed (outermost torch dimension last), so every shape
// check below is written in TPC order, FCD (torch's contiguous axis) first.
#pragma once

#include "tpc_kernel_lib_interface.h"
#include "qwen4exp_nvfp4_moe.h"

#include <cstring>
#include <initializer_list>

namespace qwen4exp_moe {

// rank + dtype + max sizes must match exactly.
inline bool has_shape(const tpc_lib_api::Tensor& t, uint32_t dims,
                      tpc_lib_api::TensorDataType type,
                      std::initializer_list<uint64_t> sizes) {
    if (t.geometry.dims != dims || t.geometry.dataType != type) return false;
    uint32_t i = 0;
    for (uint64_t size : sizes) {
        if (i >= tpc_lib_api::MAX_TENSOR_DIM || t.geometry.maxSizes[i] != size) return false;
        ++i;
    }
    return true;
}

// Overwrite a tensor's geometry (used before returning GLUE_INCOMPATIBLE_*_SIZE).
inline void set_shape(tpc_lib_api::Tensor& t, uint32_t dims,
                      tpc_lib_api::TensorDataType type,
                      std::initializer_list<uint64_t> sizes) {
    t.geometry.dims = dims;
    t.geometry.dataType = type;
    uint32_t i = 0;
    for (uint64_t size : sizes) {
        t.geometry.maxSizes[i] = size;
        t.geometry.minSizes[i] = size;
        ++i;
    }
}

// The node params blob must be exactly the struct the kernel expects.
template<typename Params>
inline bool read_params(const tpc_lib_api::UserParams& user, Params* out) {
    if (!user.nodeParams || user.nodeParamsSize != sizeof(Params)) return false;
    std::memcpy(out, user.nodeParams, sizeof(Params));
    return true;
}

// Scalar kernel arguments travel as 32-bit words in declaration order.
template<typename Params>
inline void write_params(tpc_lib_api::HabanaKernelInstantiation* out, const Params& params) {
    static_assert(sizeof(Params) % sizeof(uint32_t) == 0,
                  "TPC scalar parameters are 32-bit words");
    static_assert(sizeof(Params) / sizeof(uint32_t) <= tpc_lib_api::MAX_SCALAR_PARAMS,
                  "too many TPC scalar parameters");
    out->kernel.paramsNr = sizeof(Params) / sizeof(uint32_t);
    std::memcpy(out->kernel.scalarParams, &params, sizeof(Params));
}

inline void index_space(tpc_lib_api::HabanaKernelInstantiation* out, uint32_t rank,
                        std::initializer_list<uint64_t> sizes) {
    out->indexSpaceRank = rank;
    uint32_t i = 0;
    for (uint64_t size : sizes) out->indexSpaceGeometry[i++] = size;
}

// One tensor dimension -> one index-space dimension.
//   a        elements the tensor dimension advances per index-space step
//   start_b  first element the kernels of this index-space dimension touch
//   end_b    last element they touch
inline void map_dim(tpc_lib_api::TensorAccessPattern& pattern, unsigned dim,
                    unsigned index_space_dim, float a, float start_b, float end_b) {
    pattern.mapping[dim].indexSpaceDim = index_space_dim;
    pattern.mapping[dim].a = a;
    pattern.mapping[dim].start_b = start_b;
    pattern.mapping[dim].end_b = end_b;
}

// Hand the embedded TPC ELF to the graph compiler.
inline tpc_lib_api::GlueCodeReturn install_elf(
    tpc_lib_api::HabanaKernelInstantiation* out,
    const unsigned char* start, const unsigned char* end) {
    const auto size = static_cast<uint32_t>(end - start);
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, size);
    return tpc_lib_api::GLUE_SUCCESS;
}

}  // namespace qwen4exp_moe
