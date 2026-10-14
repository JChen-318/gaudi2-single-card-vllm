// Offline check of the TPC shim (host only, no device): GUID union and legacy symbol reach.
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "tpc_kernel_lib_interface.h"
using namespace tpc_lib_api;

static std::vector<std::string> names(const char* path, std::set<std::string>* out) {
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("cannot open %s: %s\n", path, dlerror()); return {}; }
    auto fn = (pfnGetKernelGuids)dlsym(h, "GetKernelGuids");
    uint32_t n = 0;
    fn(DEVICE_ID_GAUDI2, &n, nullptr);
    std::vector<GuidInfo> g(n);
    memset(g.data(), 0, sizeof(GuidInfo) * n);
    uint32_t m = n;
    GlueCodeReturn rc = fn(DEVICE_ID_GAUDI2, &m, g.data());
    std::vector<std::string> result;
    for (uint32_t i = 0; i < m && i < n; ++i) {
        result.emplace_back(g[i].name, strnlen(g[i].name, MAX_NODE_NAME));
        if (out) out->insert(result.back());
    }
    printf("%-90s rc=%d guids=%u\n", path, (int)rc, m);
    return result;
}

int main(int argc, char** argv) {
    // usage: shim_check <shim.so> <lib1.so> [<lib2.so> ...]  (component libraries in KIRO_TPC_LIBS order;
    // the last one is the official libtpc_kernels.so)
    if (argc < 3) { printf("usage: %s <shim.so> <lib.so>...\n", argv[0]); return 2; }
    std::vector<const char*> libs(argv + 2, argv + argc);
    std::set<std::string> unique;
    size_t total = 0;
    std::vector<std::string> custom;
    for (size_t i = 0; i < libs.size(); ++i) {
        auto part = names(libs[i], &unique);
        total += part.size();
        if (i + 1 < libs.size()) custom.insert(custom.end(), part.begin(), part.end());
    }
    std::set<std::string> merged;
    auto merged_list = names(argv[1], &merged);
    printf("sum=%zu unique=%zu merged=%zu merged_unique=%zu\n", total, unique.size(), merged_list.size(), merged.size());
    int missing = 0;
    for (auto& s : unique) if (!merged.count(s)) ++missing;
    printf("missing_from_merged=%d\n", missing);
    printf("custom kernels:");
    for (auto& s : custom) printf(" %s%s", s.c_str(), merged.count(s) ? "" : "(MISSING)");
    printf("\n");
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    const char* legacy[] = {"HabanaKernel", "HabanaKernelV2", "GetKernelNames", "GetSupportedLayouts",
                            "GetShapeInferenceFunction", "GetLibVersion", "GetSupportedDataLayouts"};
    for (auto s : legacy) printf("dlsym(shim,%s)=%s\n", s, dlsym(h, s) ? "found" : "MISSING");
    auto ver = (uint64_t(*)())dlsym(h, "GetLibVersion");
    void* off = dlopen(libs.back(), RTLD_NOW | RTLD_LOCAL);
    auto over = (uint64_t(*)())dlsym(off, "GetLibVersion");
    printf("lib version shim=%llu official=%llu\n", (unsigned long long)ver(), (unsigned long long)over());
    return missing ? 1 : 0;
}
