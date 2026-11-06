// SPDX-License-Identifier: Apache-2.0
// PyTorch registration for the Qwen4Exp NVFP4 indexed expert dequant.
//
// Built against the public Gaudi custom-op API (hpu_custom_op_pt2.h, shipped
// inside the habana-torch-plugin wheel) so no Gaudi Bridge source tree is
// required. The TPC kernel GUID resolves through GC_KERNEL_PATH, which must
// point at libqwen4exp_gaudi2_kernels.so before the HPU runtime initializes.
//
//   custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2(ids, packed, scales, scale2)
//     ids    [S]        I32  runtime expert selection (-1 -> zero rows)
//     packed [E,N,K/2]  U8   packed E2M1 nibbles, low nibble first (K/2 contiguous)
//     scales [E,N,K/16] U8   E4M3 block scales (16-element groups)
//     scale2 [E]        F32  per-expert scale, cast to BF16 then multiplied
//   -> [S,N,K] BF16 rows of the selected experts, scale2 already folded.
// Tensor order matches the DeepSeek V4 MXFP4 indexed kernels: the TPC engine
// sees dims reversed from torch, with the contiguous K/2 axis as its dim 0.
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include <memory>
#include <tuple>

#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* kSchema =
    "custom_op::custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2";
constexpr const char* kGuid =
    "custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2";

void tensor_contract(const at::Tensor& tensor, at::ScalarType dtype, const at::Device& device) {
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.device() == device,
              "indexed NVFP4 dtype/device mismatch");
  TORCH_CHECK(tensor.is_contiguous() && !tensor.requires_grad(),
              "indexed NVFP4 requires contiguous inference inputs");
}

std::vector<int64_t> dequant_shape(const at::Tensor& ids, const at::Tensor& packed,
                                   const at::Tensor& scales, const at::Tensor& scale2) {
  tensor_contract(ids, at::kInt, packed.device());
  tensor_contract(packed, at::kByte, packed.device());
  tensor_contract(scales, at::kByte, packed.device());
  tensor_contract(scale2, at::kFloat, packed.device());
  TORCH_CHECK(scale2.dim() == 1 && scale2.size(0) == packed.size(0),
              "indexed NVFP4 scale2 must be [E] F32");
  TORCH_CHECK(ids.dim() == 1 && ids.size(0) > 0 && ids.size(0) <= 512,
              "indexed NVFP4 expects [S] expert ids, 1 <= S <= 512");
  TORCH_CHECK(packed.dim() == 3 && scales.dim() == 3 && packed.size(0) > 0 &&
                  packed.size(2) > 0 && packed.size(2) % 64 == 0 && packed.size(1) > 0,
              "indexed NVFP4 requires [E,N,K/2] packed weights (K/2 contiguous), K % 128 == 0");
  TORCH_CHECK(scales.size(2) * 8 == packed.size(2) && scales.size(1) == packed.size(1) &&
                  scales.size(0) == packed.size(0),
              "indexed NVFP4 requires per-16 E4M3 scale planes [E,N,K/16]");
  return {ids.size(0), packed.size(1), packed.size(2) * 2};
}

bool register_qwen4exp_nvfp4_dequant() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& ids = inputs.at(0).toTensor();
    const auto& packed = inputs.at(1).toTensor();
    const auto& scales = inputs.at(2).toTensor();
    const auto& scale2 = inputs.at(3).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = dequant_shape(ids, packed, scales, scale2);
    return habana::PartialOutputMetaDataVector{output};
  };
  habana::custom_op::registerUserCustomOp(kSchema, kGuid, output_meta, nullptr);
  return true;
}

const bool kRegistered = register_qwen4exp_nvfp4_dequant();

at::Tensor dequant_hpu(const at::Tensor& ids, const at::Tensor& packed, const at::Tensor& scales,
                       const at::Tensor& scale2) {
  dequant_shape(ids, packed, scales, scale2);
  TORCH_CHECK(kRegistered && ids.device().type() == at::kHPU,
              "indexed NVFP4 dequant requires HPU");
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
  std::vector<c10::IValue> inputs{ids, packed, scales, scale2};
  return descriptor.execute(inputs).at(0);
}

at::Tensor dequant_meta(const at::Tensor& ids, const at::Tensor& packed, const at::Tensor& scales,
                        const at::Tensor& scale2) {
  return at::empty(dequant_shape(ids, packed, scales, scale2), ids.options().dtype(at::kBFloat16));
}

// H31: the k4-layout twin (whole 512-value K tiles nibble-major, see the kernel header).
constexpr const char* kDequantK4Schema =
    "custom_op::custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2";
constexpr const char* kDequantK4Guid =
    "custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2";

bool register_qwen4exp_nvfp4_dequant_k4() {
  auto output_meta = [](const at::Stack& inputs) {
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = dequant_shape(inputs.at(0).toTensor(), inputs.at(1).toTensor(),
                                 inputs.at(2).toTensor(), inputs.at(3).toTensor());
    return habana::PartialOutputMetaDataVector{output};
  };
  habana::custom_op::registerUserCustomOp(kDequantK4Schema, kDequantK4Guid, output_meta, nullptr);
  return true;
}

const bool kDequantK4Registered = register_qwen4exp_nvfp4_dequant_k4();

at::Tensor dequant_k4_hpu(const at::Tensor& ids, const at::Tensor& packed, const at::Tensor& scales,
                          const at::Tensor& scale2) {
  dequant_shape(ids, packed, scales, scale2);
  TORCH_CHECK(kDequantK4Registered && ids.device().type() == at::kHPU,
              "indexed NVFP4 dequant k4 requires HPU");
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kDequantK4Schema);
  std::vector<c10::IValue> inputs{ids, packed, scales, scale2};
  return descriptor.execute(inputs).at(0);
}

constexpr const char* kDequantFp8Schema =
    "custom_op::custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2";
constexpr const char* kDequantFp8Guid =
    "custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2";

// The F8 twin folds the per-expert scale2 before the E4M3 encode, so the
// kernel consumes it: value = e2m1 * e4m3fn(scale) * scale2[expert], then an
// RNE encode saturated to +-448 (torch.float8_e4m3fn semantics).
std::vector<int64_t> dequant_fp8_shape(const at::Tensor& ids, const at::Tensor& packed,
                                       const at::Tensor& scales, const at::Tensor& scale2) {
  tensor_contract(ids, at::kInt, packed.device());
  tensor_contract(packed, at::kByte, packed.device());
  tensor_contract(scales, at::kByte, packed.device());
  tensor_contract(scale2, at::kFloat, packed.device());
  TORCH_CHECK(ids.dim() == 1 && ids.size(0) > 0 && ids.size(0) <= 512,
              "indexed NVFP4 FP8 expects [S] expert ids, 1 <= S <= 512");
  TORCH_CHECK(packed.dim() == 3 && scales.dim() == 3 && scale2.dim() == 1 &&
                  packed.size(0) > 0 && packed.size(2) > 0 && packed.size(2) % 64 == 0 &&
                  packed.size(1) > 0,
              "indexed NVFP4 FP8 requires [E,N,K/2] packed weights (K/2 contiguous), K % 128 == 0");
  TORCH_CHECK(scales.size(2) * 8 == packed.size(2) && scales.size(1) == packed.size(1) &&
                  scales.size(0) == packed.size(0) && scale2.size(0) == packed.size(0),
              "indexed NVFP4 FP8 requires per-16 E4M3 scales [E,N,K/16] and scale2 [E]");
  return {ids.size(0), packed.size(1), packed.size(2) * 2};
}

bool register_qwen4exp_nvfp4_dequant_fp8() {
  auto output_meta = [](const at::Stack& inputs) {
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Float8_e4m3fn;
    output.shape = dequant_fp8_shape(inputs.at(0).toTensor(), inputs.at(1).toTensor(),
                                     inputs.at(2).toTensor(), inputs.at(3).toTensor());
    return habana::PartialOutputMetaDataVector{output};
  };
  habana::custom_op::registerUserCustomOp(kDequantFp8Schema, kDequantFp8Guid, output_meta, nullptr);
  return true;
}

const bool kDequantFp8Registered = register_qwen4exp_nvfp4_dequant_fp8();

at::Tensor dequant_fp8_hpu(const at::Tensor& ids, const at::Tensor& packed, const at::Tensor& scales,
                           const at::Tensor& scale2) {
  dequant_fp8_shape(ids, packed, scales, scale2);
  TORCH_CHECK(kDequantFp8Registered && ids.device().type() == at::kHPU,
              "indexed NVFP4 FP8 dequant requires HPU");
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kDequantFp8Schema);
  std::vector<c10::IValue> inputs{ids, packed, scales, scale2};
  return descriptor.execute(inputs).at(0);
}

at::Tensor dequant_fp8_meta(const at::Tensor& ids, const at::Tensor& packed,
                            const at::Tensor& scales, const at::Tensor& scale2) {
  return at::empty(dequant_fp8_shape(ids, packed, scales, scale2),
                   ids.options().dtype(at::kFloat8_e4m3fn));
}

constexpr const char* kGemvSchema =
    "custom_op::custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2";
constexpr const char* kGemvGuid =
    "custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2";

std::vector<int64_t> gemv_shape(const at::Tensor& ids, const at::Tensor& packed,
                                const at::Tensor& scales, const at::Tensor& activation,
                                const at::Tensor& scale2) {
  tensor_contract(ids, at::kInt, packed.device());
  tensor_contract(packed, at::kByte, packed.device());
  tensor_contract(scales, at::kByte, packed.device());
  tensor_contract(activation, at::kBFloat16, packed.device());
  tensor_contract(scale2, at::kFloat, packed.device());
  TORCH_CHECK(ids.dim() == 1 && ids.size(0) > 0 && ids.size(0) <= 512,
              "NVFP4 GEMV expects [S] expert ids, 1 <= S <= 512");
  TORCH_CHECK(packed.dim() == 3 && scales.dim() == 3 && scale2.dim() == 1,
              "NVFP4 GEMV weight planes must be packed [E,N,K/2], scales [E,N,K/16], scale2 [E]");
  // H10: activation may be per-slot [S,K] or per-token [T,K] with
  // S = T * top_k (gate/up read token rows directly, no replication).
  // The output below is still [S,N] either way.
  TORCH_CHECK(activation.dim() == 2 && activation.size(0) >= 1 &&
                  activation.size(0) <= ids.size(0) &&
                  ids.size(0) % activation.size(0) == 0 &&
                  activation.size(1) == packed.size(2) * 2,
              "NVFP4 GEMV activation must be [S,K] or [T,K] with S % T == 0");
  TORCH_CHECK(scale2.size(0) == packed.size(0) && scales.size(0) == packed.size(0) &&
                  scales.size(1) == packed.size(1) && scales.size(2) * 8 == packed.size(2),
              "NVFP4 GEMV scale planes do not match the packed weight");
  return {ids.size(0), packed.size(1)};
}

bool register_qwen4exp_nvfp4_gemv() {
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

const bool kGemvRegistered = register_qwen4exp_nvfp4_gemv();

at::Tensor gemv_hpu(const at::Tensor& ids, const at::Tensor& packed, const at::Tensor& scales,
                    const at::Tensor& activation, const at::Tensor& scale2) {
  gemv_shape(ids, packed, scales, activation, scale2);
  TORCH_CHECK(kGemvRegistered && ids.device().type() == at::kHPU, "NVFP4 GEMV requires HPU");
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kGemvSchema);
  std::vector<c10::IValue> inputs{ids, packed, scales, activation, scale2};
  return descriptor.execute(inputs).at(0);
}

at::Tensor gemv_meta(const at::Tensor& ids, const at::Tensor& packed, const at::Tensor& scales,
                     const at::Tensor& activation, const at::Tensor& scale2) {
  return at::empty(gemv_shape(ids, packed, scales, activation, scale2),
                   ids.options().dtype(at::kBFloat16));
}

// ---------------------------------------------------------------------------
// H13 HyperConnection fused ops (see kernels/qwen4exp_hc_*_bf16_gaudi2.c).
// All tensors BF16 contiguous; eps rides in nodeParams (engram pattern).
// ---------------------------------------------------------------------------

void hc_contract(const at::Tensor& t, const at::Device& dev, const char* what) {
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.device() == dev && t.is_contiguous() &&
                  !t.requires_grad(),
              what, " requires contiguous BF16 inference tensors");
}

std::shared_ptr<void> eps_params(const at::Stack& inputs, size_t index, size_t& size) {
  size = sizeof(float);
  return std::make_shared<float>(static_cast<float>(inputs.at(index).toDouble()));
}

#define HC_SCHEMA(name) "custom_op::custom_qwen4exp_hc_" name "_bf16_gaudi2"
#define HC_GUID(name) "custom_qwen4exp_hc_" name "_bf16_gaudi2"

bool register_hc_norm() {
  auto meta = [](const at::Stack& inputs) {
    const auto& x = inputs.at(0).toTensor();
    const auto& g = inputs.at(1).toTensor();
    hc_contract(x, x.device(), "hc_norm x");
    hc_contract(g, x.device(), "hc_norm gamma");
    TORCH_CHECK(x.dim() == 2 && g.dim() == 2 && g.size(1) > 0 && g.size(1) % 128 == 0 &&
                    x.size(1) > 0 && x.size(1) % g.size(1) == 0 &&
                    (g.size(0) == 1 || g.size(0) == x.size(1) / g.size(1)),
                "hc_norm expects x [T,S*D], gamma [1|S,D], D % 128 == 0");
    TORCH_CHECK(inputs.at(2).toDouble() > 0, "hc_norm eps must be positive");
    return habana::PartialOutputMetaDataVector{{at::kBFloat16, x.sizes().vec()}};
  };
  habana::custom_op::registerUserCustomOp(
      HC_SCHEMA("norm"), HC_GUID("norm"), meta,
      [](const at::Stack& s, size_t& size) { return eps_params(s, 2, size); });
  return true;
}
const bool kHcNormRegistered = register_hc_norm();

at::Tensor hc_norm_hpu(const at::Tensor& x, const at::Tensor& gamma, double eps) {
  TORCH_CHECK(kHcNormRegistered && x.device().type() == at::kHPU, "hc_norm requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(HC_SCHEMA("norm"));
  std::vector<c10::IValue> inputs{x, gamma, eps};
  return d.execute(inputs).at(0);
}
at::Tensor hc_norm_meta(const at::Tensor& x, const at::Tensor& gamma, double /*eps*/) {
  return at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
}

bool register_hc_combine() {
  auto meta = [](const at::Stack& inputs) {
    const auto& r = inputs.at(0).toTensor();
    hc_contract(r, r.device(), "hc_combine residual");
    hc_contract(inputs.at(1).toTensor(), r.device(), "hc_combine block");
    hc_contract(inputs.at(2).toTensor(), r.device(), "hc_combine inj");
    const auto& b = inputs.at(1).toTensor();
    const auto& j = inputs.at(2).toTensor();
    TORCH_CHECK(r.dim() == 2 && b.dim() == 2 && j.dim() == 2 && b.size(1) > 0 &&
                    j.size(1) > 0 && r.size(1) == b.size(1) * j.size(1) &&
                    b.size(0) == r.size(0) && j.size(0) == r.size(0),
                "hc_combine expects residual [T,S*D], block [T,D], inj [T,S]");
    return habana::PartialOutputMetaDataVector{{at::kBFloat16, r.sizes().vec()}};
  };
  habana::custom_op::registerUserCustomOp(HC_SCHEMA("combine"), HC_GUID("combine"), meta, nullptr);
  return true;
}
const bool kHcCombineRegistered = register_hc_combine();

at::Tensor hc_combine_hpu(const at::Tensor& r, const at::Tensor& b, const at::Tensor& j) {
  TORCH_CHECK(kHcCombineRegistered && r.device().type() == at::kHPU, "hc_combine requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(HC_SCHEMA("combine"));
  std::vector<c10::IValue> inputs{r, b, j};
  return d.execute(inputs).at(0);
}
at::Tensor hc_combine_meta(const at::Tensor& r, const at::Tensor& b, const at::Tensor& j) {
  return at::empty(r.sizes(), r.options().dtype(at::kBFloat16));
}

bool register_hc_combine_norm() {
  auto meta = [](const at::Stack& inputs) {
    const auto& r = inputs.at(0).toTensor();
    hc_contract(r, r.device(), "hc_combine_norm residual");
    for (int i = 1; i < 4; ++i) hc_contract(inputs.at(i).toTensor(), r.device(), "hc_combine_norm");
    const auto& b = inputs.at(1).toTensor();
    const auto& j = inputs.at(2).toTensor();
    const auto& g = inputs.at(3).toTensor();
    TORCH_CHECK(r.dim() == 2 && b.dim() == 2 && j.dim() == 2 && g.dim() == 2 &&
                    g.size(1) > 0 && g.size(1) % 128 == 0 && j.size(1) > 0 &&
                    r.size(1) == g.size(1) * j.size(1) && b.size(1) == g.size(1) &&
                    b.size(0) == r.size(0) && j.size(0) == r.size(0) &&
                    (g.size(0) == 1 || g.size(0) == j.size(1)),
                "hc_combine_norm expects residual [T,S*D], block [T,D], inj [T,S], gamma [1|S,D]");
    TORCH_CHECK(inputs.at(4).toDouble() > 0, "hc_combine_norm eps must be positive");
    return habana::PartialOutputMetaDataVector{{at::kBFloat16, r.sizes().vec()},
                                                {at::kBFloat16, r.sizes().vec()}};
  };
  habana::custom_op::registerUserCustomOp(
      HC_SCHEMA("combine_norm"), HC_GUID("combine_norm"), meta,
      [](const at::Stack& s, size_t& size) { return eps_params(s, 4, size); });
  return true;
}
const bool kHcCombineNormRegistered = register_hc_combine_norm();

std::tuple<at::Tensor, at::Tensor> hc_combine_norm_hpu(const at::Tensor& r, const at::Tensor& b,
                                                       const at::Tensor& j, const at::Tensor& g,
                                                       double eps) {
  TORCH_CHECK(kHcCombineNormRegistered && r.device().type() == at::kHPU,
              "hc_combine_norm requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      HC_SCHEMA("combine_norm"));
  std::vector<c10::IValue> inputs{r, b, j, g, eps};
  auto out = d.execute(inputs);
  return {out.at(0), out.at(1)};
}
std::tuple<at::Tensor, at::Tensor> hc_combine_norm_meta(const at::Tensor& r, const at::Tensor& b,
                                                        const at::Tensor& j, const at::Tensor& g,
                                                        double /*eps*/) {
  return {at::empty(r.sizes(), r.options().dtype(at::kBFloat16)),
          at::empty(r.sizes(), r.options().dtype(at::kBFloat16))};
}

bool register_hc_gate_mix() {
  auto meta = [](const at::Stack& inputs) {
    const auto& x = inputs.at(0).toTensor();
    hc_contract(x, x.device(), "hc_gate_mix xn");
    hc_contract(inputs.at(1).toTensor(), x.device(), "hc_gate_mix gate");
    const int64_t streams = inputs.at(2).toInt();
    TORCH_CHECK(x.dim() == 2 && inputs.at(1).toTensor().sizes() == x.sizes() && streams > 0 &&
                    x.size(1) % streams == 0,
                "hc_gate_mix expects xn/gate [T,S*D] and streams > 0");
    return habana::PartialOutputMetaDataVector{
        {at::kBFloat16, {x.size(0), x.size(1) / streams}}};
  };
  habana::custom_op::registerUserCustomOp(HC_SCHEMA("gate_mix"), HC_GUID("gate_mix"), meta, nullptr);
  return true;
}
const bool kHcGateMixRegistered = register_hc_gate_mix();

at::Tensor hc_gate_mix_hpu(const at::Tensor& x, const at::Tensor& g, int64_t streams) {
  TORCH_CHECK(kHcGateMixRegistered && x.device().type() == at::kHPU, "hc_gate_mix requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(HC_SCHEMA("gate_mix"));
  std::vector<c10::IValue> inputs{x, g, streams};
  return d.execute(inputs).at(0);
}
at::Tensor hc_gate_mix_meta(const at::Tensor& x, const at::Tensor& g, int64_t streams) {
  return at::empty({x.size(0), x.size(1) / streams}, x.options().dtype(at::kBFloat16));
}

bool register_hc_down_epilogue() {
  auto meta = [](const at::Stack& inputs) {
    const auto& m = inputs.at(0).toTensor();
    hc_contract(m, m.device(), "hc_down_epilogue merged");
    const int64_t lowrank = inputs.at(1).toInt();
    const int64_t streams = inputs.at(2).toInt();
    TORCH_CHECK(m.dim() == 2 && lowrank > 0 && lowrank <= m.size(1) && streams > 0,
                "hc_down_epilogue expects merged [T,M], 0 < lowrank <= M, streams > 0");
    return habana::PartialOutputMetaDataVector{{at::kBFloat16, {m.size(0), lowrank}},
                                                {at::kBFloat16, {m.size(0), streams}}};
  };
  habana::custom_op::registerUserCustomOp(HC_SCHEMA("down_epilogue"), HC_GUID("down_epilogue"),
                                          meta, nullptr);
  return true;
}
const bool kHcDownEpilogueRegistered = register_hc_down_epilogue();

std::tuple<at::Tensor, at::Tensor> hc_down_epilogue_hpu(const at::Tensor& m, int64_t lowrank,
                                                        int64_t streams) {
  TORCH_CHECK(kHcDownEpilogueRegistered && m.device().type() == at::kHPU,
              "hc_down_epilogue requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      HC_SCHEMA("down_epilogue"));
  std::vector<c10::IValue> inputs{m, lowrank, streams};
  auto out = d.execute(inputs);
  return {out.at(0), out.at(1)};
}
std::tuple<at::Tensor, at::Tensor> hc_down_epilogue_meta(const at::Tensor& m, int64_t lowrank,
                                                         int64_t streams) {
  return {at::empty({m.size(0), lowrank}, m.options().dtype(at::kBFloat16)),
          at::empty({m.size(0), streams}, m.options().dtype(at::kBFloat16))};
}

// ---------------------------------------------------------------------------
// H14 GDN fused ops (see kernels/qwen4exp_gdn_*_bf16_gaudi2.c).
// All activations BF16 contiguous, recurrent/conv state BF16, indices I32.
// ---------------------------------------------------------------------------

void gdn_contract(const at::Tensor& t, at::ScalarType dtype, const char* what) {
  TORCH_CHECK(t.scalar_type() == dtype && t.is_contiguous() && !t.requires_grad(), what,
              " requires contiguous inference tensors of the documented dtype");
}

#define GDN_SCHEMA(name) "custom_op::custom_qwen4exp_gdn_" name "_bf16_gaudi2"
#define GDN_GUID(name) "custom_qwen4exp_gdn_" name "_bf16_gaudi2"

bool register_gdn_recur() {
  auto meta = [](const at::Stack& inputs) {
    const auto& mixed = inputs.at(0).toTensor();
    const auto& a = inputs.at(1).toTensor();
    const auto& pool = inputs.at(5).toTensor();
    const auto& idx = inputs.at(6).toTensor();
    gdn_contract(mixed, at::kBFloat16, "gdn_recur mixed");
    gdn_contract(a, at::kBFloat16, "gdn_recur a");
    gdn_contract(inputs.at(2).toTensor(), at::kBFloat16, "gdn_recur b");
    gdn_contract(inputs.at(3).toTensor(), at::kFloat, "gdn_recur A_log");
    gdn_contract(inputs.at(4).toTensor(), at::kBFloat16, "gdn_recur dt_bias");
    gdn_contract(pool, at::kBFloat16, "gdn_recur state_pool");
    gdn_contract(idx, at::kInt, "gdn_recur indices");
    TORCH_CHECK(mixed.dim() == 2 && a.dim() == 2 && mixed.size(1) % 128 == 0 && idx.dim() == 1 &&
                    idx.size(0) > 0 && mixed.size(0) % idx.size(0) == 0 && pool.dim() == 4 &&
                    pool.size(1) == a.size(1) && pool.size(2) == 128 && pool.size(3) == 128,
                "gdn_recur expects mixed [T,C], a/b [T,Hv], state_pool [Slots,Hv,128,128], indices [S]");
    return habana::PartialOutputMetaDataVector{
        {at::kFloat, {mixed.size(0), a.size(1), 128}},
        {at::kBFloat16, {idx.size(0), a.size(1), 128, 128}}};
  };
  habana::custom_op::registerUserCustomOp(GDN_SCHEMA("recur"), GDN_GUID("recur"), meta, nullptr);
  return true;
}
const bool kGdnRecurRegistered = register_gdn_recur();

std::tuple<at::Tensor, at::Tensor> gdn_recur_hpu(const at::Tensor& mixed, const at::Tensor& a,
                                                 const at::Tensor& b, const at::Tensor& a_log,
                                                 const at::Tensor& dt_bias, const at::Tensor& pool,
                                                 const at::Tensor& idx) {
  TORCH_CHECK(kGdnRecurRegistered && mixed.device().type() == at::kHPU, "gdn_recur requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(GDN_SCHEMA("recur"));
  std::vector<c10::IValue> inputs{mixed, a, b, a_log, dt_bias, pool, idx};
  auto out = d.execute(inputs);
  return {out.at(0), out.at(1)};
}
std::tuple<at::Tensor, at::Tensor> gdn_recur_meta(const at::Tensor& mixed, const at::Tensor& a,
                                                  const at::Tensor& b, const at::Tensor& a_log,
                                                  const at::Tensor& dt_bias, const at::Tensor& pool,
                                                  const at::Tensor& idx) {
  return {at::empty({mixed.size(0), a.size(1), 128}, mixed.options().dtype(at::kFloat)),
          at::empty({idx.size(0), a.size(1), 128, 128}, mixed.options().dtype(at::kBFloat16))};
}

bool register_gdn_conv() {
  auto meta = [](const at::Stack& inputs) {
    const auto& x = inputs.at(0).toTensor();
    const auto& pool = inputs.at(1).toTensor();
    gdn_contract(x, at::kBFloat16, "gdn_conv mixed");
    gdn_contract(pool, at::kBFloat16, "gdn_conv conv_pool");
    gdn_contract(inputs.at(2).toTensor(), at::kBFloat16, "gdn_conv weight_t");
    gdn_contract(inputs.at(3).toTensor(), at::kInt, "gdn_conv indices");
    TORCH_CHECK(x.dim() == 2 && pool.dim() == 3 && pool.size(1) == 3 && pool.size(2) == x.size(1) &&
                    inputs.at(3).toTensor().size(0) == x.size(0),
                "gdn_conv expects mixed [B,C], conv_pool [Slots,3,C], indices [B]");
    return habana::PartialOutputMetaDataVector{
        {at::kBFloat16, x.sizes().vec()}, {at::kBFloat16, {x.size(0), 3, x.size(1)}}};
  };
  habana::custom_op::registerUserCustomOp(GDN_SCHEMA("conv"), GDN_GUID("conv"), meta, nullptr);
  return true;
}
const bool kGdnConvRegistered = register_gdn_conv();

std::tuple<at::Tensor, at::Tensor> gdn_conv_hpu(const at::Tensor& x, const at::Tensor& pool,
                                                const at::Tensor& w, const at::Tensor& idx) {
  TORCH_CHECK(kGdnConvRegistered && x.device().type() == at::kHPU, "gdn_conv requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(GDN_SCHEMA("conv"));
  std::vector<c10::IValue> inputs{x, pool, w, idx};
  auto out = d.execute(inputs);
  return {out.at(0), out.at(1)};
}
std::tuple<at::Tensor, at::Tensor> gdn_conv_meta(const at::Tensor& x, const at::Tensor& pool,
                                                 const at::Tensor& w, const at::Tensor& idx) {
  return {at::empty(x.sizes(), x.options().dtype(at::kBFloat16)),
          at::empty({x.size(0), 3, x.size(1)}, x.options().dtype(at::kBFloat16))};
}

bool register_gdn_conv_rows() {
  auto meta = [](const at::Stack& inputs) {
    const auto& x = inputs.at(0).toTensor();
    const auto& pool = inputs.at(1).toTensor();
    const auto& idx = inputs.at(3).toTensor();
    const auto& stops = inputs.at(5).toTensor();
    gdn_contract(x, at::kBFloat16, "gdn_conv_rows mixed");
    gdn_contract(pool, at::kBFloat16, "gdn_conv_rows conv_pool");
    gdn_contract(inputs.at(2).toTensor(), at::kBFloat16, "gdn_conv_rows weight_t");
    gdn_contract(idx, at::kInt, "gdn_conv_rows indices");
    gdn_contract(inputs.at(4).toTensor(), at::kInt, "gdn_conv_rows has_init");
    gdn_contract(stops, at::kInt, "gdn_conv_rows state_stop");
    TORCH_CHECK(x.dim() == 2 && pool.dim() == 3 && pool.size(1) == 3 && pool.size(2) == x.size(1) &&
                    idx.dim() == 1 && idx.size(0) > 0 && x.size(0) % idx.size(0) == 0 &&
                    stops.dim() == 1 && stops.size(0) == idx.size(0) + 1,
                "gdn_conv_rows expects mixed [T,C], conv_pool [Slots,3,C], indices [S], "
                "state_stop [S+1]");
    return habana::PartialOutputMetaDataVector{
        {at::kBFloat16, x.sizes().vec()}, {at::kBFloat16, {idx.size(0), 3, x.size(1)}}};
  };
  habana::custom_op::registerUserCustomOp(GDN_SCHEMA("conv_rows"), GDN_GUID("conv_rows"), meta,
                                          nullptr);
  return true;
}
const bool kGdnConvRowsRegistered = register_gdn_conv_rows();

std::tuple<at::Tensor, at::Tensor> gdn_conv_rows_hpu(const at::Tensor& x, const at::Tensor& pool,
                                                     const at::Tensor& w, const at::Tensor& idx,
                                                     const at::Tensor& hi, const at::Tensor& stops) {
  TORCH_CHECK(kGdnConvRowsRegistered && x.device().type() == at::kHPU,
              "gdn_conv_rows requires HPU");
  auto d =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(GDN_SCHEMA("conv_rows"));
  std::vector<c10::IValue> inputs{x, pool, w, idx, hi, stops};
  auto out = d.execute(inputs);
  return {out.at(0), out.at(1)};
}
std::tuple<at::Tensor, at::Tensor> gdn_conv_rows_meta(const at::Tensor& x, const at::Tensor& pool,
                                                      const at::Tensor& w, const at::Tensor& idx,
                                                      const at::Tensor& hi, const at::Tensor& stops) {
  return {at::empty(x.sizes(), x.options().dtype(at::kBFloat16)),
          at::empty({idx.size(0), 3, x.size(1)}, x.options().dtype(at::kBFloat16))};
}

struct GdnNormParams {
  float eps;
  int32_t act;
};

bool register_gdn_gated_norm() {
  auto meta = [](const at::Stack& inputs) {
    const auto& x = inputs.at(0).toTensor();
    gdn_contract(x, at::kBFloat16, "gdn_gated_norm x");
    gdn_contract(inputs.at(1).toTensor(), at::kBFloat16, "gdn_gated_norm z");
    gdn_contract(inputs.at(2).toTensor(), at::kBFloat16, "gdn_gated_norm weight");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 128, "gdn_gated_norm expects x [R,128]");
    TORCH_CHECK(inputs.at(3).toDouble() > 0, "gdn_gated_norm eps must be positive");
    return habana::PartialOutputMetaDataVector{{at::kBFloat16, x.sizes().vec()}};
  };
  habana::custom_op::registerUserCustomOp(
      GDN_SCHEMA("gated_norm"), GDN_GUID("gated_norm"), meta,
      [](const at::Stack& s, size_t& size) {
        size = sizeof(GdnNormParams);
        return std::shared_ptr<void>(
            new GdnNormParams{static_cast<float>(s.at(3).toDouble()),
                              static_cast<int32_t>(s.at(4).toInt())},
            [](void* p) { delete static_cast<GdnNormParams*>(p); });
      });
  return true;
}
const bool kGdnGatedNormRegistered = register_gdn_gated_norm();

at::Tensor gdn_gated_norm_hpu(const at::Tensor& x, const at::Tensor& z, const at::Tensor& w,
                              double eps, int64_t act) {
  TORCH_CHECK(kGdnGatedNormRegistered && x.device().type() == at::kHPU,
              "gdn_gated_norm requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      GDN_SCHEMA("gated_norm"));
  std::vector<c10::IValue> inputs{x, z, w, eps, act};
  return d.execute(inputs).at(0);
}
at::Tensor gdn_gated_norm_meta(const at::Tensor& x, const at::Tensor& z, const at::Tensor& w,
                               double /*eps*/, int64_t /*act*/) {
  return at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
}

// ---------------------------------------------------------------------------
// H15 MoE fused ops (kernels/qwen4exp_nvfp4_indexed_swiglu_gemv_*,
// qwen4exp_moe_router_topk_*, qwen4exp_moe_shared_swiglu_*,
// qwen4exp_moe_down_epilogue_*). All BF16 contiguous unless noted.
// ---------------------------------------------------------------------------

#define MOE_SCHEMA(name) "custom_op::custom_qwen4exp_" name "_bf16_gaudi2"
#define MOE_GUID(name) "custom_qwen4exp_" name "_bf16_gaudi2"

void moe_contract(const at::Tensor& t, at::ScalarType dtype, const char* what) {
  TORCH_CHECK(t.scalar_type() == dtype && t.is_contiguous() && !t.requires_grad(), what,
              " requires contiguous inference tensors of the documented dtype");
}

std::vector<int64_t> swiglu_gemv_shape(const at::Tensor& ids, const at::Tensor& gp,
                                       const at::Tensor& gs, const at::Tensor& gs2,
                                       const at::Tensor& up, const at::Tensor& us,
                                       const at::Tensor& us2, const at::Tensor& x) {
  moe_contract(ids, at::kInt, "swiglu_gemv ids");
  moe_contract(gp, at::kByte, "swiglu_gemv gate_packed");
  moe_contract(gs, at::kByte, "swiglu_gemv gate_scales");
  moe_contract(gs2, at::kFloat, "swiglu_gemv gate_scale2");
  moe_contract(up, at::kByte, "swiglu_gemv up_packed");
  moe_contract(us, at::kByte, "swiglu_gemv up_scales");
  moe_contract(us2, at::kFloat, "swiglu_gemv up_scale2");
  moe_contract(x, at::kBFloat16, "swiglu_gemv x");
  TORCH_CHECK(ids.dim() == 1 && ids.size(0) > 0 && ids.size(0) <= 512,
              "swiglu GEMV expects [S] expert ids, 1 <= S <= 512");
  TORCH_CHECK(gp.dim() == 3 && gp.sizes() == up.sizes() && gs.dim() == 3 && gs.sizes() == us.sizes() &&
                  gs2.dim() == 1 && gs2.sizes() == us2.sizes() && gs2.size(0) == gp.size(0) &&
                  gs.size(0) == gp.size(0) && gs.size(1) == gp.size(1) &&
                  gs.size(2) * 8 == gp.size(2),
              "swiglu GEMV gate/up planes must agree: packed [E,N,K/2], scales [E,N,K/16], scale2 [E]");
  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= ids.size(0) &&
                  ids.size(0) % x.size(0) == 0 && x.size(1) == gp.size(2) * 2,
              "swiglu GEMV activation must be [S,K] or [T,K] with S % T == 0");
  return {ids.size(0), gp.size(1)};
}

bool register_swiglu_gemv() {
  auto meta = [](const at::Stack& inputs) {
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = swiglu_gemv_shape(inputs.at(0).toTensor(), inputs.at(1).toTensor(),
                                     inputs.at(2).toTensor(), inputs.at(3).toTensor(),
                                     inputs.at(4).toTensor(), inputs.at(5).toTensor(),
                                     inputs.at(6).toTensor(), inputs.at(7).toTensor());
    return habana::PartialOutputMetaDataVector{output};
  };
  habana::custom_op::registerUserCustomOp(MOE_SCHEMA("nvfp4_indexed_swiglu_gemv"),
                                          MOE_GUID("nvfp4_indexed_swiglu_gemv"), meta, nullptr);
  return true;
}
const bool kSwigluGemvRegistered = register_swiglu_gemv();

at::Tensor swiglu_gemv_hpu(const at::Tensor& ids, const at::Tensor& gp, const at::Tensor& gs,
                           const at::Tensor& gs2, const at::Tensor& up, const at::Tensor& us,
                           const at::Tensor& us2, const at::Tensor& x) {
  swiglu_gemv_shape(ids, gp, gs, gs2, up, us, us2, x);
  TORCH_CHECK(kSwigluGemvRegistered && ids.device().type() == at::kHPU,
              "swiglu GEMV requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      MOE_SCHEMA("nvfp4_indexed_swiglu_gemv"));
  std::vector<c10::IValue> inputs{ids, gp, gs, gs2, up, us, us2, x};
  return d.execute(inputs).at(0);
}
at::Tensor swiglu_gemv_meta(const at::Tensor& ids, const at::Tensor& gp, const at::Tensor& gs,
                            const at::Tensor& gs2, const at::Tensor& up, const at::Tensor& us,
                            const at::Tensor& us2, const at::Tensor& x) {
  return at::empty(swiglu_gemv_shape(ids, gp, gs, gs2, up, us, us2, x),
                   ids.options().dtype(at::kBFloat16));
}

// H31 k4 layout ops: same contracts, activation in the nibble-major tile order.
bool register_swiglu_gemv_k4() {
  auto meta = [](const at::Stack& inputs) {
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = swiglu_gemv_shape(inputs.at(0).toTensor(), inputs.at(1).toTensor(),
                                     inputs.at(2).toTensor(), inputs.at(3).toTensor(),
                                     inputs.at(4).toTensor(), inputs.at(5).toTensor(),
                                     inputs.at(6).toTensor(), inputs.at(7).toTensor());
    return habana::PartialOutputMetaDataVector{output};
  };
  habana::custom_op::registerUserCustomOp(MOE_SCHEMA("nvfp4_indexed_swiglu_gemv_k4"),
                                          MOE_GUID("nvfp4_indexed_swiglu_gemv_k4"), meta, nullptr);
  return true;
}
const bool kSwigluGemvK4Registered = register_swiglu_gemv_k4();

at::Tensor swiglu_gemv_k4_hpu(const at::Tensor& ids, const at::Tensor& gp, const at::Tensor& gs,
                              const at::Tensor& gs2, const at::Tensor& up, const at::Tensor& us,
                              const at::Tensor& us2, const at::Tensor& x) {
  swiglu_gemv_shape(ids, gp, gs, gs2, up, us, us2, x);
  TORCH_CHECK(x.size(1) % 512 == 0, "swiglu GEMV k4 needs K in whole 512-value tiles");
  TORCH_CHECK(kSwigluGemvK4Registered && ids.device().type() == at::kHPU,
              "swiglu GEMV k4 requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      MOE_SCHEMA("nvfp4_indexed_swiglu_gemv_k4"));
  std::vector<c10::IValue> inputs{ids, gp, gs, gs2, up, us, us2, x};
  return d.execute(inputs).at(0);
}

bool register_gemv_k4() {
  auto meta = [](const at::Stack& inputs) {
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = gemv_shape(inputs.at(0).toTensor(), inputs.at(1).toTensor(),
                              inputs.at(2).toTensor(), inputs.at(3).toTensor(),
                              inputs.at(4).toTensor());
    return habana::PartialOutputMetaDataVector{output};
  };
  habana::custom_op::registerUserCustomOp(MOE_SCHEMA("nvfp4_indexed_gemv_k4"),
                                          MOE_GUID("nvfp4_indexed_gemv_k4"), meta, nullptr);
  return true;
}
const bool kGemvK4Registered = register_gemv_k4();

at::Tensor gemv_k4_hpu(const at::Tensor& ids, const at::Tensor& packed, const at::Tensor& scales,
                       const at::Tensor& activation, const at::Tensor& scale2) {
  gemv_shape(ids, packed, scales, activation, scale2);
  TORCH_CHECK(kGemvK4Registered && ids.device().type() == at::kHPU, "NVFP4 GEMV k4 requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      MOE_SCHEMA("nvfp4_indexed_gemv_k4"));
  std::vector<c10::IValue> inputs{ids, packed, scales, activation, scale2};
  return d.execute(inputs).at(0);
}

bool register_router_topk() {
  auto meta = [](const at::Stack& inputs) {
    const auto& logits = inputs.at(0).toTensor();
    const int64_t topk = inputs.at(1).toInt();
    moe_contract(logits, at::kBFloat16, "router_topk logits");
    TORCH_CHECK(logits.dim() == 2 && logits.size(1) == 512 && logits.size(0) > 0 &&
                    topk > 0 && topk <= 64,
                "router_topk expects logits [T,512] BF16 and 1 <= top_k <= 64");
    return habana::PartialOutputMetaDataVector{
        {at::kInt, {logits.size(0), topk}}, {at::kBFloat16, {logits.size(0), topk}}};
  };
  habana::custom_op::registerUserCustomOp(MOE_SCHEMA("moe_router_topk"),
                                          MOE_GUID("moe_router_topk"), meta, nullptr);
  return true;
}
const bool kRouterTopkRegistered = register_router_topk();

std::tuple<at::Tensor, at::Tensor> router_topk_hpu(const at::Tensor& logits, int64_t topk) {
  TORCH_CHECK(kRouterTopkRegistered && logits.device().type() == at::kHPU,
              "router_topk requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      MOE_SCHEMA("moe_router_topk"));
  std::vector<c10::IValue> inputs{logits, topk};
  auto out = d.execute(inputs);
  return {out.at(0), out.at(1)};
}
std::tuple<at::Tensor, at::Tensor> router_topk_meta(const at::Tensor& logits, int64_t topk) {
  return {at::empty({logits.size(0), topk}, logits.options().dtype(at::kInt)),
          at::empty({logits.size(0), topk}, logits.options().dtype(at::kBFloat16))};
}

bool register_shared_swiglu() {
  auto meta = [](const at::Stack& inputs) {
    const auto& merged = inputs.at(0).toTensor();
    moe_contract(merged, at::kBFloat16, "shared_swiglu merged");
    TORCH_CHECK(merged.dim() == 2 && merged.size(1) >= 3 && merged.size(1) % 2 == 1,
                "shared_swiglu expects merged [T, 2*I+1] BF16");
    return habana::PartialOutputMetaDataVector{
        {at::kBFloat16, {merged.size(0), (merged.size(1) - 1) / 2}}};
  };
  habana::custom_op::registerUserCustomOp(MOE_SCHEMA("moe_shared_swiglu"),
                                          MOE_GUID("moe_shared_swiglu"), meta, nullptr);
  return true;
}
const bool kSharedSwigluRegistered = register_shared_swiglu();

at::Tensor shared_swiglu_hpu(const at::Tensor& merged) {
  TORCH_CHECK(kSharedSwigluRegistered && merged.device().type() == at::kHPU,
              "shared_swiglu requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      MOE_SCHEMA("moe_shared_swiglu"));
  std::vector<c10::IValue> inputs{merged};
  return d.execute(inputs).at(0);
}
at::Tensor shared_swiglu_meta(const at::Tensor& merged) {
  return at::empty({merged.size(0), (merged.size(1) - 1) / 2},
                   merged.options().dtype(at::kBFloat16));
}

bool register_down_epilogue() {
  auto meta = [](const at::Stack& inputs) {
    const auto& down = inputs.at(0).toTensor();
    const auto& weights = inputs.at(1).toTensor();
    moe_contract(down, at::kBFloat16, "down_epilogue slot_down");
    moe_contract(weights, at::kBFloat16, "down_epilogue topk_weights");
    TORCH_CHECK(down.dim() == 2 && weights.dim() == 2 && weights.size(0) > 0 &&
                    down.size(0) == weights.size(0) * weights.size(1),
                "down_epilogue expects slot_down [T*K,H] and topk_weights [T,K] BF16");
    return habana::PartialOutputMetaDataVector{
        {at::kBFloat16, {weights.size(0), down.size(1)}}};
  };
  habana::custom_op::registerUserCustomOp(MOE_SCHEMA("moe_down_epilogue"),
                                          MOE_GUID("moe_down_epilogue"), meta, nullptr);
  return true;
}
const bool kDownEpilogueRegistered = register_down_epilogue();

at::Tensor down_epilogue_hpu(const at::Tensor& down, const at::Tensor& weights) {
  TORCH_CHECK(kDownEpilogueRegistered && down.device().type() == at::kHPU,
              "down_epilogue requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      MOE_SCHEMA("moe_down_epilogue"));
  std::vector<c10::IValue> inputs{down, weights};
  return d.execute(inputs).at(0);
}
at::Tensor down_epilogue_meta(const at::Tensor& down, const at::Tensor& weights) {
  return at::empty({weights.size(0), down.size(1)}, down.options().dtype(at::kBFloat16));
}

#undef MOE_SCHEMA
#undef MOE_GUID
#undef GDN_SCHEMA
#undef GDN_GUID
#undef HC_SCHEMA
#undef HC_GUID

// ---------------------------------------------------------------------------
// H29 PLE fused ops: the whole PLE gate (three grouped Gemma RMSNorms + the
// BF16 gate dot + sigmoid + gated value + the conv-input norm) and the whole
// dilated short conv (taps, SiLU, both residual adds, state roll) in one TPC
// launch each. See kernels/qwen4exp_ple_{gate,conv}_bf16_gaudi2.c.
// ---------------------------------------------------------------------------

#define PLE_SCHEMA(name) "custom_op::custom_qwen4exp_ple_" name "_bf16_gaudi2"
#define PLE_GUID(name) "custom_qwen4exp_ple_" name "_bf16_gaudi2"

bool register_ple_gate() {
  auto meta = [](const at::Stack& inputs) {
    const auto& k = inputs.at(0).toTensor();
    const auto& q = inputs.at(1).toTensor();
    const auto& v = inputs.at(2).toTensor();
    for (int i = 0; i < 6; ++i) hc_contract(inputs.at(i).toTensor(), k.device(), "ple_gate");
    TORCH_CHECK(k.dim() == 2 && v.dim() == 2 && q.sizes() == k.sizes() &&
                    v.size(0) == k.size(0) && v.size(1) > 0 && v.size(1) % 128 == 0 &&
                    k.size(1) % v.size(1) == 0,
                "ple_gate expects key/query [T,S*H], value [T,H], H % 128 == 0");
    const auto& gk = inputs.at(3).toTensor();
    TORCH_CHECK(gk.dim() == 2 && gk.size(0) > 0 && gk.size(1) % 128 == 0 &&
                    (gk.size(0) == 1 || gk.size(0) == k.size(1) / gk.size(1)),
                "ple_gate expects gamma [1|S, H]");
    TORCH_CHECK(inputs.at(6).toDouble() > 0, "ple_gate eps must be positive");
    return habana::PartialOutputMetaDataVector{{at::kBFloat16, k.sizes().vec()},
                                                {at::kBFloat16, k.sizes().vec()}};
  };
  habana::custom_op::registerUserCustomOp(
      PLE_SCHEMA("gate"), PLE_GUID("gate"), meta,
      [](const at::Stack& s, size_t& size) { return eps_params(s, 6, size); });
  return true;
}
const bool kPleGateRegistered = register_ple_gate();

std::tuple<at::Tensor, at::Tensor> ple_gate_hpu(const at::Tensor& key, const at::Tensor& query,
                                                const at::Tensor& value, const at::Tensor& gk,
                                                const at::Tensor& gq, const at::Tensor& gc,
                                                double eps) {
  TORCH_CHECK(kPleGateRegistered && key.device().type() == at::kHPU, "ple_gate requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      PLE_SCHEMA("gate"));
  std::vector<c10::IValue> inputs{key, query, value, gk, gq, gc, eps};
  auto out = d.execute(inputs);
  return {out.at(0), out.at(1)};
}
std::tuple<at::Tensor, at::Tensor> ple_gate_meta(const at::Tensor& key, const at::Tensor& query,
                                                 const at::Tensor& value, const at::Tensor& gk,
                                                 const at::Tensor& gq, const at::Tensor& gc,
                                                 double /*eps*/) {
  return {at::empty(key.sizes(), key.options().dtype(at::kBFloat16)),
          at::empty(key.sizes(), key.options().dtype(at::kBFloat16))};
}

bool register_ple_conv() {
  auto meta = [](const at::Stack& inputs) {
    const auto& x = inputs.at(0).toTensor();
    for (int i = 0; i < 5; ++i) hc_contract(inputs.at(i).toTensor(), x.device(), "ple_conv");
    const auto& pool = inputs.at(1).toTensor();
    const auto& w = inputs.at(2).toTensor();
    // x is torch [T, C]; the kernel's TPC dim 0 is C (the contiguous axis).
    // state is torch [S, W, C]: dim 0 is the span slot, dim 1 the conv window
    // (vLLM's default SD conv-state orientation).
    const int64_t C = x.size(1), T = x.size(0);
    const int64_t spans = pool.size(0), window = pool.size(1);
    TORCH_CHECK(x.dim() == 2 && pool.dim() == 3 && w.dim() == 2, "ple_conv dims");
    TORCH_CHECK(C > 0 && C % 128 == 0 && T > 0, "ple_conv C/T");
    TORCH_CHECK(pool.size(2) == C, "ple_conv pool C");
    TORCH_CHECK(w.size(1) == C && w.size(0) >= 2, "ple_conv weight");
    TORCH_CHECK(spans > 0 && window > 0 && window % (w.size(0) - 1) == 0, "ple_conv window");
    TORCH_CHECK(T % spans == 0, "ple_conv spans");
    return habana::PartialOutputMetaDataVector{{at::kBFloat16, x.sizes().vec()},
                                                {at::kBFloat16, pool.sizes().vec()}};
  };
  habana::custom_op::registerUserCustomOp(PLE_SCHEMA("conv"), PLE_GUID("conv"), meta,
                                          nullptr);
  return true;
}
const bool kPleConvRegistered = register_ple_conv();

std::tuple<at::Tensor, at::Tensor> ple_conv_hpu(const at::Tensor& x, const at::Tensor& state,
                                                const at::Tensor& weight,
                                                const at::Tensor& gated,
                                                const at::Tensor& hidden) {
  TORCH_CHECK(kPleConvRegistered && x.device().type() == at::kHPU, "ple_conv requires HPU");
  auto d = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      PLE_SCHEMA("conv"));
  std::vector<c10::IValue> inputs{x, state, weight, gated, hidden};
  auto out = d.execute(inputs);
  return {out.at(0), out.at(1)};
}
std::tuple<at::Tensor, at::Tensor> ple_conv_meta(const at::Tensor& x, const at::Tensor& state,
                                                 const at::Tensor& weight, const at::Tensor& gated,
                                                 const at::Tensor& hidden) {
  return {at::empty(x.sizes(), x.options().dtype(at::kBFloat16)),
          at::empty(state.sizes(), state.options().dtype(at::kBFloat16))};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2(Tensor ids, Tensor w, Tensor s,"
        " Tensor scale2) -> Tensor");
  m.def("custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2(Tensor ids, Tensor w, Tensor s,"
        " Tensor scale2) -> Tensor");
  m.def("custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2(Tensor ids, Tensor w, Tensor s,"
        " Tensor scale2) -> Tensor");
  m.def("custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2(Tensor ids, Tensor w, Tensor s,"
        " Tensor x, Tensor scale2) -> Tensor");
  m.def("custom_qwen4exp_hc_norm_bf16_gaudi2(Tensor x, Tensor gamma, float eps) -> Tensor");
  m.def("custom_qwen4exp_hc_combine_bf16_gaudi2(Tensor residual, Tensor block, Tensor inj) -> Tensor");
  m.def("custom_qwen4exp_hc_combine_norm_bf16_gaudi2(Tensor residual, Tensor block, Tensor inj,"
        " Tensor gamma, float eps) -> (Tensor, Tensor)");
  m.def("custom_qwen4exp_hc_gate_mix_bf16_gaudi2(Tensor xn, Tensor gate, int streams) -> Tensor");
  m.def("custom_qwen4exp_hc_down_epilogue_bf16_gaudi2(Tensor merged, int lowrank, int streams)"
        " -> (Tensor, Tensor)");
  m.def("custom_qwen4exp_gdn_recur_bf16_gaudi2(Tensor mixed, Tensor a, Tensor b, Tensor A_log,"
        " Tensor dt_bias, Tensor state_pool, Tensor indices) -> (Tensor, Tensor)");
  m.def("custom_qwen4exp_gdn_conv_bf16_gaudi2(Tensor mixed, Tensor conv_pool, Tensor weight_t,"
        " Tensor indices) -> (Tensor, Tensor)");
  m.def("custom_qwen4exp_gdn_conv_rows_bf16_gaudi2(Tensor mixed, Tensor conv_pool, Tensor weight_t,"
        " Tensor indices, Tensor has_init, Tensor state_stop) -> (Tensor, Tensor)");
  m.def("custom_qwen4exp_gdn_gated_norm_bf16_gaudi2(Tensor x, Tensor z, Tensor weight, float eps,"
        " int act) -> Tensor");
  m.def("custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2(Tensor ids, Tensor gate_packed,"
        " Tensor gate_scales, Tensor gate_scale2, Tensor up_packed, Tensor up_scales,"
        " Tensor up_scale2, Tensor x) -> Tensor");
  m.def("custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2(Tensor ids, Tensor gate_packed,"
        " Tensor gate_scales, Tensor gate_scale2, Tensor up_packed, Tensor up_scales,"
        " Tensor up_scale2, Tensor x) -> Tensor");
  m.def("custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2(Tensor ids, Tensor w, Tensor s,"
        " Tensor x, Tensor scale2) -> Tensor");
  m.def("custom_qwen4exp_moe_router_topk_bf16_gaudi2(Tensor logits, int top_k) -> (Tensor, Tensor)");
  m.def("custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2(Tensor merged) -> Tensor");
  m.def("custom_qwen4exp_moe_down_epilogue_bf16_gaudi2(Tensor slot_down, Tensor topk_weights)"
        " -> Tensor");
  m.def("custom_qwen4exp_ple_gate_bf16_gaudi2(Tensor key, Tensor query, Tensor value,"
        " Tensor gamma_key, Tensor gamma_query, Tensor gamma_conv, float eps)"
        " -> (Tensor, Tensor)");
  m.def("custom_qwen4exp_ple_conv_bf16_gaudi2(Tensor x, Tensor state, Tensor weight,"
        " Tensor gated, Tensor hidden) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2", dequant_hpu);
  m.impl("custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2", dequant_fp8_hpu);
  m.impl("custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2", dequant_k4_hpu);
  m.impl("custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2", gemv_hpu);
  m.impl("custom_qwen4exp_hc_norm_bf16_gaudi2", hc_norm_hpu);
  m.impl("custom_qwen4exp_hc_combine_bf16_gaudi2", hc_combine_hpu);
  m.impl("custom_qwen4exp_hc_combine_norm_bf16_gaudi2", hc_combine_norm_hpu);
  m.impl("custom_qwen4exp_hc_gate_mix_bf16_gaudi2", hc_gate_mix_hpu);
  m.impl("custom_qwen4exp_hc_down_epilogue_bf16_gaudi2", hc_down_epilogue_hpu);
  m.impl("custom_qwen4exp_gdn_recur_bf16_gaudi2", gdn_recur_hpu);
  m.impl("custom_qwen4exp_gdn_conv_bf16_gaudi2", gdn_conv_hpu);
  m.impl("custom_qwen4exp_gdn_conv_rows_bf16_gaudi2", gdn_conv_rows_hpu);
  m.impl("custom_qwen4exp_gdn_gated_norm_bf16_gaudi2", gdn_gated_norm_hpu);
  m.impl("custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2", swiglu_gemv_hpu);
  m.impl("custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2", swiglu_gemv_k4_hpu);
  m.impl("custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2", gemv_k4_hpu);
  m.impl("custom_qwen4exp_moe_router_topk_bf16_gaudi2", router_topk_hpu);
  m.impl("custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2", shared_swiglu_hpu);
  m.impl("custom_qwen4exp_moe_down_epilogue_bf16_gaudi2", down_epilogue_hpu);
  m.impl("custom_qwen4exp_ple_gate_bf16_gaudi2", ple_gate_hpu);
  m.impl("custom_qwen4exp_ple_conv_bf16_gaudi2", ple_conv_hpu);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2", dequant_meta);
  m.impl("custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2", dequant_fp8_meta);
  m.impl("custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2", dequant_meta);
  m.impl("custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2", gemv_meta);
  m.impl("custom_qwen4exp_hc_norm_bf16_gaudi2", hc_norm_meta);
  m.impl("custom_qwen4exp_hc_combine_bf16_gaudi2", hc_combine_meta);
  m.impl("custom_qwen4exp_hc_combine_norm_bf16_gaudi2", hc_combine_norm_meta);
  m.impl("custom_qwen4exp_hc_gate_mix_bf16_gaudi2", hc_gate_mix_meta);
  m.impl("custom_qwen4exp_hc_down_epilogue_bf16_gaudi2", hc_down_epilogue_meta);
  m.impl("custom_qwen4exp_gdn_recur_bf16_gaudi2", gdn_recur_meta);
  m.impl("custom_qwen4exp_gdn_conv_bf16_gaudi2", gdn_conv_meta);
  m.impl("custom_qwen4exp_gdn_conv_rows_bf16_gaudi2", gdn_conv_rows_meta);
  m.impl("custom_qwen4exp_gdn_gated_norm_bf16_gaudi2", gdn_gated_norm_meta);
  m.impl("custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2", swiglu_gemv_meta);
  m.impl("custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2", swiglu_gemv_meta);
  m.impl("custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2", gemv_meta);
  m.impl("custom_qwen4exp_moe_router_topk_bf16_gaudi2", router_topk_meta);
  m.impl("custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2", shared_swiglu_meta);
  m.impl("custom_qwen4exp_moe_down_epilogue_bf16_gaudi2", down_epilogue_meta);
  m.impl("custom_qwen4exp_ple_gate_bf16_gaudi2", ple_gate_meta);
  m.impl("custom_qwen4exp_ple_conv_bf16_gaudi2", ple_conv_meta);
}
