// SPDX-License-Identifier: Apache-2.0
// PyTorch registration for the Qwen4Exp NVFP4 MoE TPC library
// (native/qwen4exp-nvfp4-moe).
//
// Built against the public Gaudi custom-op API (hpu_custom_op_pt2.h, shipped
// inside the habana-torch-plugin wheel) like csrc/qwen4exp/pytorch, so no Gaudi
// Bridge source tree is required. The TPC kernel GUIDs resolve through
// GC_KERNEL_PATH, which must contain libqwen4exp_nvfp4_moe_kernels.so (directly,
// or as one entry of KIRO_TPC_LIBS behind native/tpc-kernel-shim).
//
// Operators (argument order = the host glue's tensor order, then the kernel's
// scalar arguments in declaration order):
//
//   custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2(ids, packed, scales, x, scale2)
//     ids    [S]        I32   selected expert per routed slot (-1 -> zero row)
//     packed [E,N,K/2]  U8    E2M1 nibbles, low nibble = even k
//     scales [E,N,K/16] U8    per-16 E4M3 block scales
//     x      [S,K]      BF16  routed activations        (K % 128 == 0, N % 4 == 0)
//     scale2 [E]        F32   per-expert global scale
//   -> [S,N] BF16
//
//   custom_qwen4exp_nvfp4_moe_count_i32_gaudi2(ids, experts)
//     ids [S] I32, experts an int (E % 64 == 0, 64 <= E <= 4096)
//   -> counts [blocks, E] I32, blocks = ceil(S/64)
//
//   custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2(counts)
//   -> offsets [blocks, E] I32, totals [E] I32
//
//   custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2(ids, offsets, experts, slots, cap)
//     ids [S] I32 (< 0 = skip), offsets [blocks, E] I32, cap >= slots
//   -> bucket [E*cap] I32 (view as [E, cap]), place [S] I32 (-1 = skipped)
//
// counts/offsets are block-major in torch order ([blocks, E]): the expert axis is
// the contiguous one, which is what the kernels vectorize over (64 experts per
// 64-lane vector). The bucket is expert-major, i.e. row e*cap+m is a routed slot
// of expert e, because that is the layout a grouped GEMM consumes.
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include <tuple>
#include <vector>

#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* kGemvSchema =
    "custom_op::custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2";
constexpr const char* kGemvGuid = "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2";
constexpr const char* kCountSchema =
    "custom_op::custom_qwen4exp_nvfp4_moe_count_i32_gaudi2";
constexpr const char* kCountGuid = "custom_qwen4exp_nvfp4_moe_count_i32_gaudi2";
constexpr const char* kPrefixSchema =
    "custom_op::custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2";
constexpr const char* kPrefixGuid = "custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2";
constexpr const char* kScatterSchema =
    "custom_op::custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2";
constexpr const char* kScatterGuid = "custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2";

constexpr int64_t kBlockSlots = 64;  // QWEN4EXP_MOE_BLOCK_SLOTS
constexpr int64_t kRowsPerMember = 4;  // QWEN4EXP_MOE_ROWS_PER_MEMBER

// Node params layouts must stay word-for-word identical to the host glue's
// structs (native/qwen4exp-nvfp4-moe/host/*_i32_gaudi2.cpp).
struct CountParams {
  int32_t experts;
};
struct ScatterParams {
  int32_t experts;
  int32_t slots;
  int32_t cap;
};

void tensor_contract(const at::Tensor& tensor, at::ScalarType dtype,
                     const at::Device& device) {
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.device() == device,
              "qwen4exp NVFP4 MoE dtype/device mismatch");
  TORCH_CHECK(tensor.is_contiguous() && !tensor.requires_grad(),
              "qwen4exp NVFP4 MoE requires contiguous inference inputs");
}

// ---------------------------------------------------------------- decode GEMV

std::vector<int64_t> gemv_shape(const at::Tensor& ids, const at::Tensor& packed,
                                const at::Tensor& scales, const at::Tensor& x,
                                const at::Tensor& scale2) {
  tensor_contract(ids, at::kInt, packed.device());
  tensor_contract(packed, at::kByte, packed.device());
  tensor_contract(scales, at::kByte, packed.device());
  tensor_contract(x, at::kBFloat16, packed.device());
  tensor_contract(scale2, at::kFloat, packed.device());
  const int64_t slots = ids.numel();
  TORCH_CHECK(ids.dim() == 1 && slots >= 1 && slots <= 64,
              "NVFP4 MoE GEMV expects [S] expert ids, 1 <= S <= 64");
  TORCH_CHECK(packed.dim() == 3 && packed.size(0) > 0 && packed.size(2) > 0 &&
                  packed.size(2) % 64 == 0 && packed.size(1) > 0 &&
                  packed.size(1) % kRowsPerMember == 0,
              "NVFP4 MoE GEMV weights must be [E,N,K/2] with K % 128 == 0 and N % 4 == 0");
  TORCH_CHECK(scales.dim() == 3 && scales.size(0) == packed.size(0) &&
                  scales.size(1) == packed.size(1) &&
                  scales.size(2) * 8 == packed.size(2),
              "NVFP4 MoE GEMV needs per-16 E4M3 scale planes [E,N,K/16]");
  TORCH_CHECK(scale2.dim() == 1 && scale2.size(0) == packed.size(0),
              "NVFP4 MoE GEMV needs a per-expert scale2 [E]");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == slots && x.size(1) == packed.size(2) * 2,
              "NVFP4 MoE GEMV activations must be [S,K]");
  return {slots, packed.size(1)};
}

bool register_qwen4exp_nvfp4_moe_gemv() {
  auto output_meta = [](const at::Stack& inputs) {
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = gemv_shape(inputs.at(0).toTensor(), inputs.at(1).toTensor(),
                              inputs.at(2).toTensor(), inputs.at(3).toTensor(),
                              inputs.at(4).toTensor());
    return habana::PartialOutputMetaDataVector{output};
  };
  habana::custom_op::registerUserCustomOp(kGemvSchema, kGemvGuid, output_meta, nullptr);
  return true;
}

const bool kGemvRegistered = register_qwen4exp_nvfp4_moe_gemv();

at::Tensor gemv_hpu(const at::Tensor& ids, const at::Tensor& packed,
                    const at::Tensor& scales, const at::Tensor& x,
                    const at::Tensor& scale2) {
  gemv_shape(ids, packed, scales, x, scale2);
  TORCH_CHECK(kGemvRegistered && ids.device().type() == at::kHPU,
              "NVFP4 MoE GEMV requires HPU");
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kGemvSchema);
  std::vector<c10::IValue> inputs{ids, packed, scales, x, scale2};
  return descriptor.execute(inputs).at(0);
}

at::Tensor gemv_meta(const at::Tensor& ids, const at::Tensor& packed,
                     const at::Tensor& scales, const at::Tensor& x,
                     const at::Tensor& scale2) {
  return at::empty(gemv_shape(ids, packed, scales, x, scale2),
                   ids.options().dtype(at::kBFloat16));
}

// ------------------------------------------------------------------- bucketing

std::vector<int64_t> count_shape(const at::Tensor& ids, int64_t experts) {
  tensor_contract(ids, at::kInt, ids.device());
  TORCH_CHECK(ids.dim() == 1 && ids.numel() >= 1 && ids.numel() <= 65536,
              "MoE count expects [S] routed ids, 1 <= S <= 65536");
  TORCH_CHECK(experts >= 64 && experts <= 4096 && experts % kBlockSlots == 0,
              "MoE count expects 64 <= experts <= 4096 with experts % 64 == 0");
  return {(ids.numel() + kBlockSlots - 1) / kBlockSlots, experts};
}

bool register_qwen4exp_nvfp4_moe_count() {
  auto output_meta = [](const at::Stack& inputs) {
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Int;
    output.shape = count_shape(inputs.at(0).toTensor(), inputs.at(1).toInt());
    return habana::PartialOutputMetaDataVector{output};
  };
  auto fill_params = [](const at::Stack& inputs, size_t& bytes) -> std::shared_ptr<void> {
    bytes = sizeof(CountParams);
    auto params = std::make_shared<CountParams>();
    params->experts = static_cast<int32_t>(inputs.at(1).toInt());
    return params;
  };
  habana::custom_op::registerUserCustomOp(kCountSchema, kCountGuid, output_meta,
                                          fill_params);
  return true;
}

const bool kCountRegistered = register_qwen4exp_nvfp4_moe_count();

std::vector<int64_t> prefix_offsets_shape(const at::Tensor& counts) {
  tensor_contract(counts, at::kInt, counts.device());
  TORCH_CHECK(counts.dim() == 2 && counts.size(1) >= 64 && counts.size(1) <= 4096 &&
                  counts.size(1) % kBlockSlots == 0 && counts.size(0) >= 1,
              "MoE prefix expects counts [blocks, E] with E % 64 == 0");
  return {counts.size(0), counts.size(1)};
}

bool register_qwen4exp_nvfp4_moe_prefix() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto offsets = prefix_offsets_shape(inputs.at(0).toTensor());
    habana::PartialOutputMetaData offsets_meta;
    offsets_meta.dtype = at::ScalarType::Int;
    offsets_meta.shape = offsets;
    habana::PartialOutputMetaData totals_meta;
    totals_meta.dtype = at::ScalarType::Int;
    totals_meta.shape = {offsets[1]};
    return habana::PartialOutputMetaDataVector{offsets_meta, totals_meta};
  };
  habana::custom_op::registerUserCustomOp(kPrefixSchema, kPrefixGuid, output_meta,
                                          nullptr);
  return true;
}

const bool kPrefixRegistered = register_qwen4exp_nvfp4_moe_prefix();

std::vector<int64_t> scatter_bucket_shape(const at::Tensor& ids,
                                          const at::Tensor& offsets,
                                          int64_t experts, int64_t slots,
                                          int64_t cap) {
  tensor_contract(ids, at::kInt, ids.device());
  tensor_contract(offsets, at::kInt, ids.device());
  TORCH_CHECK(ids.dim() == 1 && ids.numel() >= 1 && ids.numel() <= 65536,
              "MoE scatter expects [S] routed ids, 1 <= S <= 65536");
  TORCH_CHECK(slots == ids.numel(), "MoE scatter slots must match the ids length");
  TORCH_CHECK(cap >= slots, "MoE scatter cap must be >= slots");
  TORCH_CHECK(experts >= 64 && experts <= 4096 && experts % kBlockSlots == 0,
              "MoE scatter expects 64 <= experts <= 4096 with experts % 64 == 0");
  TORCH_CHECK(offsets.dim() == 2 &&
                  offsets.size(0) == (slots + kBlockSlots - 1) / kBlockSlots &&
                  offsets.size(1) == experts,
              "MoE scatter expects offsets [ceil(S/64), E]");
  return {experts * cap};
}

bool register_qwen4exp_nvfp4_moe_scatter() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto bucket = scatter_bucket_shape(
        inputs.at(0).toTensor(), inputs.at(1).toTensor(), inputs.at(2).toInt(),
        inputs.at(3).toInt(), inputs.at(4).toInt());
    habana::PartialOutputMetaData bucket_meta;
    bucket_meta.dtype = at::ScalarType::Int;
    bucket_meta.shape = bucket;
    habana::PartialOutputMetaData place_meta;
    place_meta.dtype = at::ScalarType::Int;
    place_meta.shape = {inputs.at(3).toInt()};
    return habana::PartialOutputMetaDataVector{bucket_meta, place_meta};
  };
  auto fill_params = [](const at::Stack& inputs, size_t& bytes) -> std::shared_ptr<void> {
    bytes = sizeof(ScatterParams);
    auto params = std::make_shared<ScatterParams>();
    params->experts = static_cast<int32_t>(inputs.at(2).toInt());
    params->slots = static_cast<int32_t>(inputs.at(3).toInt());
    params->cap = static_cast<int32_t>(inputs.at(4).toInt());
    return params;
  };
  habana::custom_op::registerUserCustomOp(kScatterSchema, kScatterGuid, output_meta,
                                          fill_params);
  return true;
}

const bool kScatterRegistered = register_qwen4exp_nvfp4_moe_scatter();

at::Tensor count_hpu(const at::Tensor& ids, int64_t experts) {
  count_shape(ids, experts);
  TORCH_CHECK(kCountRegistered && ids.device().type() == at::kHPU,
              "MoE count requires HPU");
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kCountSchema);
  std::vector<c10::IValue> inputs{ids, experts};
  // The schema is "-> Tensor": the custom-op API always hands back a vector.
  const auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1, "MoE count must return one tensor");
  return outputs[0];
}

at::Tensor count_meta(const at::Tensor& ids, int64_t experts) {
  return at::empty(count_shape(ids, experts), ids.options().dtype(at::kInt));
}

std::tuple<at::Tensor, at::Tensor> prefix_hpu(const at::Tensor& counts) {
  prefix_offsets_shape(counts);
  TORCH_CHECK(kPrefixRegistered && counts.device().type() == at::kHPU,
              "MoE prefix requires HPU");
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kPrefixSchema);
  std::vector<c10::IValue> inputs{counts};
  const auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2, "MoE prefix must return offsets and totals");
  return std::make_tuple(outputs[0], outputs[1]);
}

std::tuple<at::Tensor, at::Tensor> prefix_meta(const at::Tensor& counts) {
  const auto offsets = prefix_offsets_shape(counts);
  return std::make_tuple(at::empty(offsets, counts.options().dtype(at::kInt)),
                         at::empty({offsets[1]}, counts.options().dtype(at::kInt)));
}

std::tuple<at::Tensor, at::Tensor> scatter_hpu(const at::Tensor& ids,
                                               const at::Tensor& offsets, int64_t experts,
                                               int64_t slots, int64_t cap) {
  scatter_bucket_shape(ids, offsets, experts, slots, cap);
  TORCH_CHECK(kScatterRegistered && ids.device().type() == at::kHPU,
              "MoE scatter requires HPU");
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kScatterSchema);
  std::vector<c10::IValue> inputs{ids, offsets, experts, slots, cap};
  const auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2, "MoE scatter must return bucket and place");
  return std::make_tuple(outputs[0], outputs[1]);
}

std::tuple<at::Tensor, at::Tensor> scatter_meta(const at::Tensor& ids,
                                                const at::Tensor& offsets, int64_t experts,
                                                int64_t slots, int64_t cap) {
  const auto bucket = scatter_bucket_shape(ids, offsets, experts, slots, cap);
  return std::make_tuple(at::empty(bucket, ids.options().dtype(at::kInt)),
                         at::empty({slots}, ids.options().dtype(at::kInt)));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2(Tensor ids, Tensor packed,"
        " Tensor scales, Tensor x, Tensor scale2) -> Tensor");
  m.def("custom_qwen4exp_nvfp4_moe_count_i32_gaudi2(Tensor ids, int experts)"
        " -> Tensor");
  m.def("custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2(Tensor counts)"
        " -> (Tensor, Tensor)");
  m.def("custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2(Tensor ids, Tensor offsets,"
        " int experts, int slots, int cap) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2", gemv_hpu);
  m.impl("custom_qwen4exp_nvfp4_moe_count_i32_gaudi2", count_hpu);
  m.impl("custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2", prefix_hpu);
  m.impl("custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2", scatter_hpu);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2", gemv_meta);
  m.impl("custom_qwen4exp_nvfp4_moe_count_i32_gaudi2", count_meta);
  m.impl("custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2", prefix_meta);
  m.impl("custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2", scatter_meta);
}
