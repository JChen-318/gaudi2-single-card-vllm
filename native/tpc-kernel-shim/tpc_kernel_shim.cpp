// One TPC kernel library that serves the kernels of several libraries.
//
// The recipe disk cache (PT_HPU_RECIPE_CACHE_CONFIG) requires GC_KERNEL_PATH to name a single
// regular file. This shim loads the libraries listed in KIRO_TPC_LIBS (colon separated, same order
// and precedence as GC_KERNEL_PATH), reports the union of their GUIDs and forwards each call to the
// library that reported the GUID. The official libtpc_kernels.so is also a DT_NEEDED dependency,
// so dlsym() on this shim finds its other (legacy) entry points unchanged.
//
// KIRO_TPC_SHIM_LOG=1 prints one line per routed GUID and per fallback decision to stderr.
#include <dlfcn.h>
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tpc_kernel_lib_interface.h"

using namespace tpc_lib_api;

typedef GlueCodeReturn (*pfnGetSuggestedManipulation)(const HabanaKernelParams*, void*);

namespace {

// Used when KIRO_TPC_LIBS is unset: only the official library. Deployments list their own
// libraries first, e.g. KIRO_TPC_LIBS=/path/libcustom_a.so:/path/libcustom_b.so:/usr/lib/habanalabs/libtpc_kernels.so
const char* kDefaultLibs = "/usr/lib/habanalabs/libtpc_kernels.so";

struct Lib {
    std::string path;
    void* handle = nullptr;
    pfnGetKernelGuids guids = nullptr;
    pfnInstantiateTpcKernel instantiate = nullptr;
    pfnGetShapeInference shape = nullptr;
    pfnGetSupportedDataLayout layouts = nullptr;
    pfnGetSuggestedManipulation manipulation = nullptr;
    pfnGetLibVersion version = nullptr;
};

std::vector<Lib> g_libs;
std::once_flag g_load_once;
std::mutex g_mutex;
std::map<std::pair<int, std::string>, int> g_owner;  // (device, guid name) -> library index
bool g_log = false;

void load_libs() {
    g_log = getenv("KIRO_TPC_SHIM_LOG") && strcmp(getenv("KIRO_TPC_SHIM_LOG"), "0") != 0;
    const char* list = getenv("KIRO_TPC_LIBS");
    std::string paths = (list && *list) ? list : kDefaultLibs;
    size_t start = 0;
    while (start <= paths.size()) {
        size_t end = paths.find(':', start);
        std::string path = paths.substr(start, end == std::string::npos ? std::string::npos : end - start);
        start = end == std::string::npos ? paths.size() + 1 : end + 1;
        if (path.empty()) continue;
        Lib lib;
        lib.path = path;
        lib.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib.handle) {
            fprintf(stderr, "[tpc-shim] cannot load %s: %s\n", path.c_str(), dlerror());
            continue;
        }
        lib.guids = (pfnGetKernelGuids)dlsym(lib.handle, KERNEL_GUIDS_ENTRY_POINT_NAME);
        lib.instantiate = (pfnInstantiateTpcKernel)dlsym(lib.handle, KERNEL_INSTANTIATION_ENTRY_POINT_NAME);
        lib.shape = (pfnGetShapeInference)dlsym(lib.handle, GET_SHAPE_INFERENCE_ENTRY_POINT_NAME);
        lib.layouts = (pfnGetSupportedDataLayout)dlsym(lib.handle, GET_SUPPORTED_DATA_LAYOUTS_ENTRY_POINT_NAME);
        lib.manipulation = (pfnGetSuggestedManipulation)dlsym(lib.handle, "GetSuggestedManipulation");
        lib.version = (pfnGetLibVersion)dlsym(lib.handle, GET_LIB_VERSION_ENTRY_POINT_NAME);
        if (!lib.guids || !lib.instantiate) {
            fprintf(stderr, "[tpc-shim] %s is not a TPC kernel library\n", path.c_str());
            continue;
        }
        if (g_log) fprintf(stderr, "[tpc-shim] loaded %s\n", path.c_str());
        g_libs.push_back(lib);
    }
}

void ensure_loaded() { std::call_once(g_load_once, load_libs); }

// Collect every library's GUIDs for one device; earlier libraries win on duplicate names.
std::vector<GuidInfo> collect(DeviceId device) {
    std::vector<GuidInfo> all;
    std::set<std::string> seen;
    std::lock_guard<std::mutex> lock(g_mutex);
    for (size_t index = 0; index < g_libs.size(); ++index) {
        uint32_t count = 0;
        if (g_libs[index].guids(device, &count, nullptr) != GLUE_SUCCESS || count == 0) continue;
        std::vector<GuidInfo> part(count);
        memset(part.data(), 0, sizeof(GuidInfo) * count);
        uint32_t filled = count;
        if (g_libs[index].guids(device, &filled, part.data()) != GLUE_SUCCESS) continue;
        for (uint32_t i = 0; i < filled && i < count; ++i) {
            std::string name(part[i].name, strnlen(part[i].name, MAX_NODE_NAME));
            if (!seen.insert(name).second) continue;  // an earlier library already serves this GUID
            g_owner.emplace(std::make_pair((int)device, name), (int)index);
            all.push_back(part[i]);
        }
    }
    return all;
}

int owner_of(DeviceId device, const GuidInfo* guid) {
    if (!guid) return -1;
    auto key = std::make_pair((int)device, std::string(guid->name, strnlen(guid->name, MAX_NODE_NAME)));
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto found = g_owner.find(key);
        if (found != g_owner.end()) return found->second;
    }
    collect(device);  // GUIDs not listed yet for this device
    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_owner.find(key);
    if (found == g_owner.end()) {
        if (g_log) fprintf(stderr, "[tpc-shim] unknown guid %s\n", key.second.c_str());
        return -1;
    }
    return found->second;
}

}  // namespace

extern "C" {

GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* kernelCount, GuidInfo* guids) {
    ensure_loaded();
    if (!kernelCount) return GLUE_FAILED;
    std::vector<GuidInfo> all = collect(device);
    uint32_t capacity = *kernelCount;
    *kernelCount = (uint32_t)all.size();
    if (!guids || capacity == 0) return GLUE_SUCCESS;
    if (capacity < all.size()) return GLUE_FAILED;
    memcpy(guids, all.data(), sizeof(GuidInfo) * all.size());
    if (g_log) fprintf(stderr, "[tpc-shim] device %d: %zu guids from %zu libraries\n", (int)device, all.size(),
                       g_libs.size());
    return GLUE_SUCCESS;
}

GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    ensure_loaded();
    if (!params) return GLUE_FAILED;
    int index = owner_of(params->deviceId, &params->guid);
    if (index < 0) return GLUE_NODE_NOT_FOUND;
    return g_libs[index].instantiate(params, instance);
}

GlueCodeReturn GetShapeInference(DeviceId device, const ShapeInferenceParams* params, ShapeInferenceOutput* output) {
    ensure_loaded();
    if (!params) return GLUE_FAILED;
    int index = owner_of(device, params->pGuid ? params->pGuid : &params->guid);
    if (index < 0 || !g_libs[index].shape) return GLUE_NODE_NOT_FOUND;
    return g_libs[index].shape(device, params, output);
}

GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams* params, NodeDataLayouts* layouts,
                                       uint32_t* layoutCount) {
    ensure_loaded();
    if (!params || !layoutCount) return GLUE_FAILED;
    int index = owner_of(params->deviceId, &params->guid);
    if (index < 0) return GLUE_NODE_NOT_FOUND;
    if (g_libs[index].layouts) return g_libs[index].layouts(params, layouts, layoutCount);
    // The owning library has no layout entry point: report one layout-agnostic ("xxxxx") layout.
    if (g_log) fprintf(stderr, "[tpc-shim] agnostic layout for %s\n", params->guid.name);
    uint32_t capacity = *layoutCount;
    *layoutCount = 1;
    if (!layouts || capacity == 0) return GLUE_SUCCESS;
    auto fill = [](TensorDataLayout* tensors, uint32_t count) {
        for (uint32_t i = 0; tensors && i < count; ++i) {
            memset(tensors[i].layout, 0, sizeof(tensors[i].layout));
            memcpy(tensors[i].layout, "xxxxx", 5);
        }
    };
    fill(layouts[0].inputs, layouts[0].inputTensorNr);
    fill(layouts[0].outputs, layouts[0].outputTensorNr);
    fill(layouts[0].shapeTensors, layouts[0].shapeTensorNr);
    return GLUE_SUCCESS;
}

GlueCodeReturn GetSuggestedManipulation(const HabanaKernelParams* params, void* suggestion) {
    ensure_loaded();
    if (!params) return GLUE_FAILED;
    int index = owner_of(params->deviceId, &params->guid);
    if (index < 0 || !g_libs[index].manipulation) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    return g_libs[index].manipulation(params, suggestion);
}

uint64_t GetLibVersion() {
    ensure_loaded();
    for (auto& lib : g_libs)
        if (lib.version) return lib.version();  // the official library's version
    return 0;
}

}  // extern "C"
