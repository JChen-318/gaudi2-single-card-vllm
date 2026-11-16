// SPDX-License-Identifier: Apache-2.0
// custom_op::flashinfer_gaudi_spec_conv(x [B*T, C] bf16, pool [S, L, C] bf16, w_t [4, C] bf16, idx [B] i32,
//                                      acc [B] i32) -> (y [B*T, C] bf16, state [B, L, C] bf16), L = T + 2.
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"

namespace {
constexpr const char* schema = "custom_op::flashinfer_gaudi_spec_conv";
constexpr const char* guid = "flashinfer_gaudi_spec_conv_bf16_gaudi2";

void validate(const at::Tensor& x, const at::Tensor& pool, const at::Tensor& w, const at::Tensor& idx,
              const at::Tensor& acc) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && pool.scalar_type() == at::kBFloat16 &&
                w.scalar_type() == at::kBFloat16, "spec_conv: x, pool and w must be BF16");
    TORCH_CHECK(idx.scalar_type() == at::kInt && acc.scalar_type() == at::kInt, "spec_conv: idx/acc must be int32");
    TORCH_CHECK(x.dim() == 2 && pool.dim() == 3 && w.dim() == 2 && idx.dim() == 1 && acc.dim() == 1,
                "spec_conv: bad ranks");
    const auto C = x.size(1), L = pool.size(1), B = idx.size(0);
    TORCH_CHECK(C % 128 == 0 && pool.size(2) == C && w.size(0) == 4 && w.size(1) == C && acc.size(0) == B &&
                L >= 3 && x.size(0) == B * (L - 2), "spec_conv: bad shapes");
}

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(
        schema, guid,
        [](const at::Stack& stack) {
            const auto& x = stack.at(0).toTensor();
            const auto& pool = stack.at(1).toTensor();
            const auto& idx = stack.at(3).toTensor();
            habana::PartialOutputMetaData y, st;
            y.dtype = at::kBFloat16;
            y.shape = {x.size(0), x.size(1)};
            st.dtype = at::kBFloat16;
            st.shape = {idx.size(0), pool.size(1), x.size(1)};
            return habana::PartialOutputMetaDataVector{y, st};
        }, nullptr);
    return true;
}();

std::tuple<at::Tensor, at::Tensor> run(const at::Tensor& x, const at::Tensor& pool, const at::Tensor& w,
                                       const at::Tensor& idx, const at::Tensor& acc) {
    validate(x, pool, w, idx, acc);
    TORCH_CHECK(registered);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{x, pool, w, idx, acc};
    auto out = descriptor.execute(inputs);
    return {out.at(0), out.at(1)};
}

std::tuple<at::Tensor, at::Tensor> meta(const at::Tensor& x, const at::Tensor& pool, const at::Tensor& w,
                                        const at::Tensor& idx, const at::Tensor& acc) {
    validate(x, pool, w, idx, acc);
    return {at::empty({x.size(0), x.size(1)}, x.options()),
            at::empty({idx.size(0), pool.size(1), x.size(1)}, x.options())};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("flashinfer_gaudi_spec_conv(Tensor x, Tensor pool, Tensor w, Tensor idx, Tensor acc) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("flashinfer_gaudi_spec_conv", run);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("flashinfer_gaudi_spec_conv", meta);
}
