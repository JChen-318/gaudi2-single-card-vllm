// SPDX-License-Identifier: Apache-2.0
// Registration entry points for the Qwen4Exp Gaudi2 TPC kernels.
//
// Protocol mirrors csrc/deepseek_v4/host/entry_points.cpp: custom GUIDs are
// appended to the stock kernel list (libtpc_kernels.so is dlopen'ed and every
// stock query forwarded), and InstantiateTpcKernel dispatches the custom
// names to their GlueCode classes before falling back to stock.
#include "qwen4exp_hc_combine_bf16_gaudi2.hpp"
#include "qwen4exp_hc_combine_norm_bf16_gaudi2.hpp"
#include "qwen4exp_hc_down_epilogue_bf16_gaudi2.hpp"
#include "qwen4exp_hc_gate_mix_bf16_gaudi2.hpp"
#include "qwen4exp_gdn_conv_bf16_gaudi2.hpp"
#include "qwen4exp_gdn_conv_rows_bf16_gaudi2.hpp"
#include "qwen4exp_gdn_gated_norm_bf16_gaudi2.hpp"
#include "qwen4exp_gdn_recur_bf16_gaudi2.hpp"
#include "qwen4exp_hc_norm_bf16_gaudi2.hpp"
#include "qwen4exp_moe_down_epilogue_bf16_gaudi2.hpp"
#include "qwen4exp_moe_router_topk_bf16_gaudi2.hpp"
#include "qwen4exp_moe_shared_swiglu_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2.hpp"
#include "qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2.hpp"
#include "qwen4exp_ple_conv_bf16_gaudi2.hpp"
#include "qwen4exp_ple_gate_bf16_gaudi2.hpp"
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
bool custom_guid(const char* name) {
    constexpr char prefix[] = "custom_qwen4exp_";
    return std::strncmp(name, prefix, sizeof(prefix) - 1) == 0;
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
    if (custom_guid(params->guid.name))
        return tpc_lib_api::GLUE_FAILED;
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
    constexpr uint32_t kCustomCount = 21;
    uint32_t stock_count = 0;
    auto status = stock(deviceId, &stock_count, nullptr);
    if (status != tpc_lib_api::GLUE_SUCCESS) return status;
    const uint32_t capacity = *kernelCount;
    *kernelCount = kCustomCount + stock_count;
    if (!guids || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;
    if (capacity < *kernelCount) return tpc_lib_api::GLUE_FAILED;
    std::memset(guids, 0, kCustomCount * sizeof(*guids));
    Qwen4ExpNvfp4IndexedDequantBF16Gaudi2().GetKernelName(guids[0].name);
    Qwen4ExpNvfp4IndexedDequantFP8Gaudi2().GetKernelName(guids[1].name);
    Qwen4ExpNvfp4IndexedGemvBF16Gaudi2().GetKernelName(guids[2].name);
    Qwen4ExpHcNormBF16Gaudi2().GetKernelName(guids[3].name);
    Qwen4ExpHcCombineBF16Gaudi2().GetKernelName(guids[4].name);
    Qwen4ExpHcCombineNormBF16Gaudi2().GetKernelName(guids[5].name);
    Qwen4ExpHcGateMixBF16Gaudi2().GetKernelName(guids[6].name);
    Qwen4ExpHcDownEpilogueBF16Gaudi2().GetKernelName(guids[7].name);
    Qwen4ExpGdnRecurBF16Gaudi2().GetKernelName(guids[8].name);
    Qwen4ExpGdnConvBF16Gaudi2().GetKernelName(guids[9].name);
    Qwen4ExpGdnConvRowsBF16Gaudi2().GetKernelName(guids[10].name);
    Qwen4ExpGdnGatedNormBF16Gaudi2().GetKernelName(guids[11].name);
    Qwen4ExpNvfp4IndexedSwigluGemvBF16Gaudi2().GetKernelName(guids[12].name);
    Qwen4ExpMoeRouterTopkBF16Gaudi2().GetKernelName(guids[13].name);
    Qwen4ExpMoeSharedSwigluBF16Gaudi2().GetKernelName(guids[14].name);
    Qwen4ExpMoeDownEpilogueBF16Gaudi2().GetKernelName(guids[15].name);
    Qwen4ExpPleGateBF16Gaudi2().GetKernelName(guids[16].name);
    Qwen4ExpPleConvBF16Gaudi2().GetKernelName(guids[17].name);
    Qwen4ExpNvfp4IndexedGemvK4BF16Gaudi2().GetKernelName(guids[18].name);
    Qwen4ExpNvfp4IndexedSwigluGemvK4BF16Gaudi2().GetKernelName(guids[19].name);
    Qwen4ExpNvfp4IndexedDequantK4BF16Gaudi2().GetKernelName(guids[20].name);
    return stock(deviceId, kernelCount, guids + kCustomCount);
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
    if (!params || !instance) return tpc_lib_api::GLUE_FAILED;
    if (custom_guid(params->guid.name)) {
        if (std::strcmp(params->guid.name,
                        "custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2") == 0) {
            return Qwen4ExpNvfp4IndexedDequantK4BF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name,
                        "custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2") == 0) {
            return Qwen4ExpNvfp4IndexedGemvK4BF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name,
                        "custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2") == 0) {
            return Qwen4ExpNvfp4IndexedSwigluGemvK4BF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name,
                        "custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2") == 0) {
            return Qwen4ExpNvfp4IndexedGemvBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name,
                        "custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2") == 0) {
            return Qwen4ExpNvfp4IndexedDequantFP8Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_hc_norm_bf16_gaudi2") == 0) {
            return Qwen4ExpHcNormBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_hc_combine_bf16_gaudi2") == 0) {
            return Qwen4ExpHcCombineBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_hc_combine_norm_bf16_gaudi2") == 0) {
            return Qwen4ExpHcCombineNormBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_hc_gate_mix_bf16_gaudi2") == 0) {
            return Qwen4ExpHcGateMixBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_hc_down_epilogue_bf16_gaudi2") == 0) {
            return Qwen4ExpHcDownEpilogueBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_gdn_recur_bf16_gaudi2") == 0) {
            return Qwen4ExpGdnRecurBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_gdn_conv_bf16_gaudi2") == 0) {
            return Qwen4ExpGdnConvBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_gdn_conv_rows_bf16_gaudi2") == 0) {
            return Qwen4ExpGdnConvRowsBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_gdn_gated_norm_bf16_gaudi2") == 0) {
            return Qwen4ExpGdnGatedNormBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name,
                        "custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2") == 0) {
            return Qwen4ExpNvfp4IndexedSwigluGemvBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_moe_router_topk_bf16_gaudi2") == 0) {
            return Qwen4ExpMoeRouterTopkBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2") == 0) {
            return Qwen4ExpMoeSharedSwigluBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_moe_down_epilogue_bf16_gaudi2") == 0) {
            return Qwen4ExpMoeDownEpilogueBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_ple_gate_bf16_gaudi2") == 0) {
            return Qwen4ExpPleGateBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        if (std::strcmp(params->guid.name, "custom_qwen4exp_ple_conv_bf16_gaudi2") == 0) {
            return Qwen4ExpPleConvBF16Gaudi2().GetGcDefinitions(
                const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
        }
        return Qwen4ExpNvfp4IndexedDequantBF16Gaudi2().GetGcDefinitions(
            const_cast<tpc_lib_api::HabanaKernelParams*>(params), instance);
    }
    auto stock = stock_symbol<decltype(&InstantiateTpcKernel)>("InstantiateTpcKernel");
    return stock ? stock(params, instance) : tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId deviceId,
    const tpc_lib_api::ShapeInferenceParams* params, tpc_lib_api::ShapeInferenceOutput* output) {
    if (!params || !output) return tpc_lib_api::GLUE_FAILED;
    const auto* guid = params->pGuid ? params->pGuid : &params->guid;
    if (std::strncmp(guid->name, "custom_qwen4exp_", 16) == 0) return tpc_lib_api::GLUE_SUCCESS;
    auto stock = stock_symbol<decltype(&GetShapeInference)>("GetShapeInference");
    return stock ? stock(deviceId, params, output) : tpc_lib_api::GLUE_SUCCESS;
}
}
