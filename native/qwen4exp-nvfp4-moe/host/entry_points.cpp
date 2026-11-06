// SPDX-License-Identifier: Apache-2.0
// Registration entry points for the Qwen4Exp NVFP4 MoE Gaudi2 TPC kernels.
//
// Same protocol as vllm-gaudi/csrc/qwen4exp/host/entry_points.cpp: the custom
// GUIDs are appended to the stock kernel list (libtpc_kernels.so is dlopen'ed
// and every stock query forwarded), and InstantiateTpcKernel dispatches the
// custom names to their GlueCode classes before falling back to stock. That
// keeps this library usable both directly (GC_KERNEL_PATH pointing at it) and
// behind native/tpc-kernel-shim.
//
// The four GUIDs:
//   custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2     decode: grouped 4-bit GEMV
//   custom_qwen4exp_nvfp4_moe_count_i32_gaudi2     prefill bucketing step 1
//   custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2    prefill bucketing step 2
//   custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2   prefill bucketing step 3
#include "qwen4exp_nvfp4_moe_count_i32_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_prefix_i32_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_scatter_i32_gaudi2.hpp"

#include <cstring>
#include <dlfcn.h>

namespace {
void* stock_library() {
    static void* handle = dlopen("/usr/lib/habanalabs/libtpc_kernels.so", RTLD_NOW | RTLD_LOCAL);
    return handle;
}
template<typename Function> Function stock_symbol(const char* name) {
    void* handle = stock_library();
    return handle ? reinterpret_cast<Function>(dlsym(handle, name)) : nullptr;
}
constexpr char kCustomPrefix[] = "custom_qwen4exp_nvfp4_moe_";
bool custom_guid(const char* name) {
    return std::strncmp(name, kCustomPrefix, sizeof(kCustomPrefix) - 1) == 0;
}
}  // namespace

namespace tpc_lib_api {
// The optional manipulation payload is opaque here: stock kernels forward it
// unchanged, while custom kernels keep their explicitly qualified layout.
struct _TensorManipulationSuggestion;
}

extern "C" {
uint64_t GetLibVersion() {
    auto stock = stock_symbol<tpc_lib_api::pfnGetLibVersion>("GetLibVersion");
    return stock ? stock() : 0;
}

tpc_lib_api::GlueCodeReturn GetSuggestedManipulation(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::_TensorManipulationSuggestion* suggestion) {
    if (!params || !suggestion) return tpc_lib_api::GLUE_FAILED;
    if (custom_guid(params->guid.name)) return tpc_lib_api::GLUE_FAILED;
    auto stock = stock_symbol<decltype(&GetSuggestedManipulation)>("GetSuggestedManipulation");
    return stock ? stock(params, suggestion) : tpc_lib_api::GLUE_FAILED;
}

tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::NodeDataLayouts* layouts, uint32_t* count) {
    if (!params || !count) return tpc_lib_api::GLUE_FAILED;
    if (custom_guid(params->guid.name)) {
        *count = 1;
        if (layouts) {
            // An explicit don't-care layout matches the optional-interface
            // default. Zero layouts is a compilation failure in Synapse 1.16.
            for (uint32_t i = 0; i < layouts->inputTensorNr; ++i)
                std::memset(layouts->inputs[i].layout, 'x', sizeof(layouts->inputs[i].layout));
            for (uint32_t i = 0; i < layouts->outputTensorNr; ++i)
                std::memset(layouts->outputs[i].layout, 'x', sizeof(layouts->outputs[i].layout));
            for (uint32_t i = 0; i < layouts->shapeTensorNr; ++i)
                std::memset(layouts->shapeTensors[i].layout, 'x', sizeof(layouts->shapeTensors[i].layout));
        }
        return tpc_lib_api::GLUE_SUCCESS;
    }
    auto stock = stock_symbol<tpc_lib_api::pfnGetSupportedDataLayout>("GetSupportedDataLayouts");
    return stock ? stock(params, layouts, count) : tpc_lib_api::GLUE_FAILED;
}

tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId deviceId,
    uint32_t* kernelCount, tpc_lib_api::GuidInfo* guids) {
    if (!kernelCount) return tpc_lib_api::GLUE_FAILED;
    auto stock = stock_symbol<decltype(&GetKernelGuids)>("GetKernelGuids");
    if (!stock) return tpc_lib_api::GLUE_FAILED;
    if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) return stock(deviceId, kernelCount, guids);
    constexpr uint32_t kCustomCount = 4;
    uint32_t stock_count = 0;
    auto status = stock(deviceId, &stock_count, nullptr);
    if (status != tpc_lib_api::GLUE_SUCCESS) return status;
    const uint32_t capacity = *kernelCount;
    const uint32_t total = kCustomCount + stock_count;
    *kernelCount = total;
    if (!guids || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;
    if (capacity < total) return tpc_lib_api::GLUE_FAILED;
    std::memset(guids, 0, kCustomCount * sizeof(*guids));
    Qwen4ExpNvfp4MoeGemvBF16Gaudi2().GetKernelName(guids[0].name);
    Qwen4ExpNvfp4MoeCountI32Gaudi2().GetKernelName(guids[1].name);
    Qwen4ExpNvfp4MoePrefixI32Gaudi2().GetKernelName(guids[2].name);
    Qwen4ExpNvfp4MoeScatterI32Gaudi2().GetKernelName(guids[3].name);
    // The stock query reports its own count: the reported total is ours, not stock's.
    auto status_with_guids = stock(deviceId, kernelCount, guids + kCustomCount);
    *kernelCount = total;
    return status_with_guids;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
    if (!params || !instance) return tpc_lib_api::GLUE_FAILED;
    if (custom_guid(params->guid.name)) {
        const char* name = params->guid.name;
        if (std::strcmp(name, "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2") == 0) {
            return Qwen4ExpNvfp4MoeGemvBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(name, "custom_qwen4exp_nvfp4_moe_count_i32_gaudi2") == 0) {
            return Qwen4ExpNvfp4MoeCountI32Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(name, "custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2") == 0) {
            return Qwen4ExpNvfp4MoePrefixI32Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(name, "custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2") == 0) {
            return Qwen4ExpNvfp4MoeScatterI32Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        return tpc_lib_api::GLUE_NODE_NOT_FOUND;
    }
    auto stock = stock_symbol<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
    return stock ? stock(params, instance) : tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId deviceId,
    const tpc_lib_api::ShapeInferenceParams* params, tpc_lib_api::ShapeInferenceOutput* output) {
    if (!params || !output) return tpc_lib_api::GLUE_FAILED;
    const auto* guid = params->pGuid ? params->pGuid : &params->guid;
    if (std::strncmp(guid->name, kCustomPrefix, sizeof(kCustomPrefix) - 1) == 0)
        return tpc_lib_api::GLUE_SUCCESS;
    auto stock = stock_symbol<decltype(&GetShapeInference)>("GetShapeInference");
    return stock ? stock(deviceId, params, output) : tpc_lib_api::GLUE_SUCCESS;
}
}
