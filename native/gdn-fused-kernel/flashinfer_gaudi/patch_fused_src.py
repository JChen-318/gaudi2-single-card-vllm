"""Add the fused GDN MTP kernel and its custom op, and the 8-row prepared core, to a copy of csrc/flashinfer_gaudi.
python3 patch_fused_src.py <src_root>   (takes the kernel/glue files from native/gdn-fused-kernel)"""
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
here = Path(__file__).resolve().parent.parent


def edit(rel, pairs):
    path = root / rel
    s = path.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:80])
        s = s.replace(old, new)
    path.write_text(s, encoding="utf-8")


# Prepared core: replace the kernel with the 8-row batched-reduction version; its glue sizes the tiles.
shutil.copy(here / "gdn_mtp_prepared_f32_gaudi2.c", root / "kernels")
edit("host/gdn_mtp_prepared_f32_gaudi2.cpp", [("#define MTP_PREPARED_ROWS 4\n", "#define MTP_PREPARED_ROWS 8\n")])

shutil.copy(here / "gdn_mtp_fused_f32_gaudi2.c", root / "kernels")
shutil.copy(here / "gdn_mtp_fused_f32_gaudi2.cpp", root / "host")
shutil.copy(here / "gdn_mtp_fused_f32_gaudi2.hpp", root / "host")

edit("CMakeLists.txt", [
    ("        gdn_mtp_prepared_f32_gaudi2\n", "        gdn_mtp_prepared_f32_gaudi2\n        gdn_mtp_fused_f32_gaudi2\n"),
    ("    host/gdn_mtp_prepared_f32_gaudi2.cpp\n",
     "    host/gdn_mtp_prepared_f32_gaudi2.cpp\n    host/gdn_mtp_fused_f32_gaudi2.cpp\n"),
])

edit("host/entry_points.cpp", [
    ('#include "gdn_mtp_prepared_f32_gaudi2.hpp"\n',
     '#include "gdn_mtp_prepared_f32_gaudi2.hpp"\n#include "gdn_mtp_fused_f32_gaudi2.hpp"\n'),
    ("    *kernelCount = 9;\n    if (guids == nullptr || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;\n"
     "    if (capacity < 9) return tpc_lib_api::GLUE_FAILED;",
     "    *kernelCount = 10;\n    if (guids == nullptr || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;\n"
     "    if (capacity < 10) return tpc_lib_api::GLUE_FAILED;"),
    ("        std::memset(guids, 0, 9 * sizeof(tpc_lib_api::GuidInfo));",
     "        std::memset(guids, 0, 10 * sizeof(tpc_lib_api::GuidInfo));"),
    ("        std::strcpy(guids[8].name, AddRmsNormQuantBf16Gaudi2::name);\n",
     "        std::strcpy(guids[8].name, AddRmsNormQuantBf16Gaudi2::name);\n"
     "        GdnMtpFusedF32Gaudi2().GetKernelName(guids[9].name);\n"),
    ("    if (std::strcmp(params->guid.name, AddRmsNormQuantBf16Gaudi2::name) == 0) {\n"
     "        return AddRmsNormQuantBf16Gaudi2{}.GetGcDefinitions(params, instance);\n    }\n"
     "    return tpc_lib_api::GLUE_NODE_NOT_FOUND;",
     "    if (std::strcmp(params->guid.name, AddRmsNormQuantBf16Gaudi2::name) == 0) {\n"
     "        return AddRmsNormQuantBf16Gaudi2{}.GetGcDefinitions(params, instance);\n    }\n"
     "    GdnMtpFusedF32Gaudi2 fusedMtpKernel;\n    fusedMtpKernel.GetKernelName(kernelName);\n"
     "    if (std::strcmp(params->guid.name, kernelName) == 0) {\n"
     "        return fusedMtpKernel.GetGcDefinitions(params, instance);\n    }\n"
     "    return tpc_lib_api::GLUE_NODE_NOT_FOUND;"),
])

edit("pytorch/hpu_flashinfer_gaudi.cpp", [
    ('constexpr const char* kMtpPreparedGuid = "flashinfer_gaudi_gdn_mtp_prepared_f32_gaudi2";\n',
     'constexpr const char* kMtpPreparedGuid = "flashinfer_gaudi_gdn_mtp_prepared_f32_gaudi2";\n'
     'constexpr const char* kMtpFusedSchema = "custom_op::flashinfer_gaudi_gdn_mtp_fused";\n'
     'constexpr const char* kMtpFusedGuid = "flashinfer_gaudi_gdn_mtp_fused_f32_gaudi2";\n'),
    ("const bool kMtpPreparedRegistered = register_gdn_mtp_prepared();\n",
     "const bool kMtpPreparedRegistered = register_gdn_mtp_prepared();\n\n"
     "bool register_gdn_mtp_fused() {\n"
     "  auto output_meta = [](const at::Stack& inputs) {\n"
     "    const auto& pool = inputs.at(0).toTensor();\n"
     "    const auto& packed = inputs.at(1).toTensor();\n"
     "    habana::PartialOutputMetaData output;\n"
     "    output.dtype = at::ScalarType::Float;\n"
     "    output.shape = {packed.size(0), packed.size(1), pool.size(1), pool.size(2)};\n"
     "    habana::PartialOutputMetaData checkpoints;\n"
     "    checkpoints.dtype = at::ScalarType::Float;\n"
     "    checkpoints.shape = {packed.size(0), packed.size(1), pool.size(1), pool.size(2), pool.size(3)};\n"
     "    return habana::PartialOutputMetaDataVector{output, checkpoints};\n"
     "  };\n"
     "  habana::custom_op::registerUserCustomOp(kMtpFusedSchema, kMtpFusedGuid, output_meta, nullptr);\n"
     "  return true;\n"
     "}\n\n"
     "const bool kMtpFusedRegistered = register_gdn_mtp_fused();\n"),
    ("at::Tensor dflash2_select_path_hpu(\n",
     "void check_gdn_mtp_fused_inputs(\n"
     "    const at::Tensor& state_pool,\n"
     "    const at::Tensor& packed_qkv,\n"
     "    const at::Tensor& log_decay,\n"
     "    const at::Tensor& beta,\n"
     "    const at::Tensor& state_indices,\n"
     "    const at::Tensor& num_accepted) {\n"
     "  TORCH_CHECK(state_pool.scalar_type() == at::ScalarType::Float);\n"
     "  TORCH_CHECK(packed_qkv.scalar_type() == at::ScalarType::BFloat16);\n"
     "  TORCH_CHECK(log_decay.scalar_type() == at::ScalarType::Float);\n"
     "  TORCH_CHECK(beta.scalar_type() == at::ScalarType::BFloat16);\n"
     "  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int);\n"
     "  TORCH_CHECK(num_accepted.scalar_type() == at::ScalarType::Int);\n"
     "  TORCH_CHECK(state_pool.dim() == 4 && state_pool.size(0) > 0 && state_pool.size(1) == kMtpValueHeads &&\n"
     "              state_pool.size(2) == kMtpDim && state_pool.size(3) == kMtpDim);\n"
     "  TORCH_CHECK(packed_qkv.dim() == 3 && packed_qkv.size(1) == kMtpTokens &&\n"
     "              packed_qkv.size(2) == kMtpPackedWidth);\n"
     "  const auto batch = packed_qkv.size(0);\n"
     "  TORCH_CHECK(batch > 0 && batch <= kMtpMaxBatch);\n"
     "  TORCH_CHECK(log_decay.sizes() == at::IntArrayRef({batch, kMtpTokens, kMtpValueHeads}));\n"
     "  TORCH_CHECK(beta.sizes() == log_decay.sizes());\n"
     "  TORCH_CHECK(state_indices.sizes() == at::IntArrayRef({batch, kMtpTokens}));\n"
     "  TORCH_CHECK(num_accepted.dim() == 1 && num_accepted.size(0) == batch);\n"
     "  TORCH_CHECK(state_pool.is_contiguous() && packed_qkv.is_contiguous() && log_decay.is_contiguous() &&\n"
     "              beta.is_contiguous() && state_indices.is_contiguous() && num_accepted.is_contiguous());\n"
     "}\n\n"
     "std::tuple<at::Tensor, at::Tensor> gdn_mtp_fused_hpu(\n"
     "    const at::Tensor& state_pool,\n"
     "    const at::Tensor& packed_qkv,\n"
     "    const at::Tensor& log_decay,\n"
     "    const at::Tensor& beta,\n"
     "    const at::Tensor& state_indices,\n"
     "    const at::Tensor& num_accepted) {\n"
     "  check_gdn_mtp_fused_inputs(state_pool, packed_qkv, log_decay, beta, state_indices, num_accepted);\n"
     "  TORCH_CHECK(kMtpFusedRegistered);\n"
     "  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kMtpFusedSchema);\n"
     "  std::vector<c10::IValue> inputs{state_pool, packed_qkv, log_decay, beta, state_indices, num_accepted};\n"
     "  auto outputs = descriptor.execute(inputs);\n"
     "  TORCH_CHECK(outputs.size() == 2);\n"
     "  return {outputs.at(0), outputs.at(1)};\n"
     "}\n\n"
     "std::tuple<at::Tensor, at::Tensor> gdn_mtp_fused_meta(\n"
     "    const at::Tensor& state_pool,\n"
     "    const at::Tensor& packed_qkv,\n"
     "    const at::Tensor& log_decay,\n"
     "    const at::Tensor& beta,\n"
     "    const at::Tensor& state_indices,\n"
     "    const at::Tensor& num_accepted) {\n"
     "  check_gdn_mtp_fused_inputs(state_pool, packed_qkv, log_decay, beta, state_indices, num_accepted);\n"
     "  const auto batch = packed_qkv.size(0);\n"
     "  auto options = state_pool.options();\n"
     "  return {\n"
     "      at::empty({batch, kMtpTokens, kMtpValueHeads, kMtpDim}, options),\n"
     "      at::empty({batch, kMtpTokens, kMtpValueHeads, kMtpDim, kMtpDim}, options)};\n"
     "}\n\n"
     "at::Tensor dflash2_select_path_hpu(\n"),
    ('      "Tensor decay, Tensor beta) -> (Tensor, Tensor)");\n}\n',
     '      "Tensor decay, Tensor beta) -> (Tensor, Tensor)");\n'
     '  m.def(\n'
     '      "flashinfer_gaudi_gdn_mtp_fused(Tensor state_pool, Tensor packed_qkv, Tensor log_decay, "\n'
     '      "Tensor beta, Tensor state_indices, Tensor num_accepted) -> (Tensor, Tensor)");\n}\n'),
    ('  m.impl("flashinfer_gaudi_gdn_mtp_prepared", gdn_mtp_prepared_hpu);\n',
     '  m.impl("flashinfer_gaudi_gdn_mtp_prepared", gdn_mtp_prepared_hpu);\n'
     '  m.impl("flashinfer_gaudi_gdn_mtp_fused", gdn_mtp_fused_hpu);\n'),
    ('  m.impl("flashinfer_gaudi_gdn_mtp_prepared", gdn_mtp_prepared_meta);\n',
     '  m.impl("flashinfer_gaudi_gdn_mtp_prepared", gdn_mtp_prepared_meta);\n'
     '  m.impl("flashinfer_gaudi_gdn_mtp_fused", gdn_mtp_fused_meta);\n'),
])
print("patched", root)
