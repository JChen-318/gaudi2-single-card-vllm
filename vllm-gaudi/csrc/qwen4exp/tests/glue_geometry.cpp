// SPDX-License-Identifier: Apache-2.0
// Host-side validation of the qwen4exp TPC library: GUID registration and
// GlueCode geometry, without any HPU device.
//
// Expects the shared library next to this executable
// (libqwen4exp_gaudi2_kernels.so in the CMake build dir; build with:
// cmake -B build -S . && cmake --build build). An explicit argv[1] overrides.
#include "tpc_kernel_lib_interface.h"
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <string>
#include <unistd.h>

static std::string default_library_path() {
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) return "./libqwen4exp_gaudi2_kernels.so";
    exe[n] = '\0';
    std::string path(exe);
    const size_t slash = path.rfind('/');
    const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
    return dir + "/libqwen4exp_gaudi2_kernels.so";
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    const std::string default_library = default_library_path();
    const char* library = argc > 1 ? argv[1] : default_library.c_str();
    void* handle = dlopen(library, RTLD_NOW);
    if (!handle) { printf("dlopen failed (%s): %s\n", library, dlerror()); return 1; }
    auto guids = (tpc_lib_api::GlueCodeReturn(*)(tpc_lib_api::DeviceId, uint32_t*, tpc_lib_api::GuidInfo*))dlsym(handle, "GetKernelGuids");
    auto instantiate = (tpc_lib_api::GlueCodeReturn(*)(const tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*))dlsym(handle, "InstantiateTpcKernel");
    if (!guids || !instantiate) { printf("symbols missing\n"); return 1; }
    uint32_t count = 0;
    if (guids(tpc_lib_api::DEVICE_ID_GAUDI2, &count, nullptr) != tpc_lib_api::GLUE_SUCCESS) { printf("count query failed\n"); return 1; }
    auto* info = new tpc_lib_api::GuidInfo[count];
    if (guids(tpc_lib_api::DEVICE_ID_GAUDI2, &count, info) != tpc_lib_api::GLUE_SUCCESS) { printf("guid query failed\n"); return 1; }
    const char* target = "custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2";
    bool found = false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::strcmp(info[i].name, target) == 0) found = true;
    printf("guid registered: %s (%u total guids incl. stock)\n", found ? "YES" : "NO", count);
    if (!found) return 1;

    // w13 shape: K=1280 rows, N=2560, E=512, S=10 -> torch packed [512,2560,640],
    // scales [512,2560,80], output [10,2560,1280]. Synapse geometry is
    // outermost-first (reversed): packed [640,2560,512], scales [80,2560,512],
    // output [1280,2560,10].
    tpc_lib_api::HabanaKernelParams params{};
    tpc_lib_api::Tensor in_tensors[4]{};
    tpc_lib_api::Tensor out_tensors[1]{};
    auto set_geom = [](tpc_lib_api::Tensor& tensor, tpc_lib_api::TensorDataType type,
                       uint32_t dims, std::initializer_list<uint64_t> sizes) {
        tensor.geometry.dataType = type;
        tensor.geometry.dims = dims;
        uint32_t i = 0;
        for (uint64_t size : sizes) tensor.geometry.maxSizes[i++] = size;
    };
    std::memset(&params, 0, sizeof(params));
    params.inputTensors = in_tensors;
    params.outputTensors = out_tensors;
    params.inputTensorNr = 4;
    params.outputTensorNr = 1;
    std::strcpy(params.guid.name, target);
    params.inputTensors[0].geometry.dataType = tpc_lib_api::DATA_I32;
    params.inputTensors[0].geometry.dims = 1;
    params.inputTensors[0].geometry.maxSizes[0] = 10;
    params.inputTensors[1].geometry.dataType = tpc_lib_api::DATA_U8;
    params.inputTensors[1].geometry.dims = 3;
    params.inputTensors[1].geometry.maxSizes[0] = 640;
    params.inputTensors[1].geometry.maxSizes[1] = 2560;
    params.inputTensors[1].geometry.maxSizes[2] = 512;
    params.inputTensors[2].geometry.dataType = tpc_lib_api::DATA_U8;
    params.inputTensors[2].geometry.dims = 3;
    params.inputTensors[2].geometry.maxSizes[0] = 80;
    params.inputTensors[2].geometry.maxSizes[1] = 2560;
    params.inputTensors[2].geometry.maxSizes[2] = 512;
    params.inputTensors[3].geometry.dataType = tpc_lib_api::DATA_F32;
    params.inputTensors[3].geometry.dims = 1;
    params.inputTensors[3].geometry.maxSizes[0] = 512;
    params.outputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
    params.outputTensors[0].geometry.dims = 3;
    params.outputTensors[0].geometry.maxSizes[0] = 1280;
    params.outputTensors[0].geometry.maxSizes[1] = 2560;
    params.outputTensors[0].geometry.maxSizes[2] = 10;
    tpc_lib_api::HabanaKernelInstantiation inst{};
    tpc_lib_api::TensorAccessPattern in_patterns[4]{};
    tpc_lib_api::TensorAccessPattern out_patterns[1]{};
    std::memset(&inst, 0, sizeof(inst));
    unsigned char elf_buffer[1 << 20];
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = in_patterns;
    inst.outputTensorAccessPattern = out_patterns;
    auto status = instantiate(&params, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS) { printf("instantiate failed: %d\n", status); return 1; }
    // H27: the dequant kernel owns ROWS_PER_MEMBER = 4 output rows per member,
    // so the row axis of the index space is ceil(N / 4) = 640 for N = 2560.
    printf("index space: %lu x %lu x %lu (expect 3 x 640 x 10)\n",
           inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1], inst.indexSpaceGeometry[2]);
    printf("elf size: %u bytes\n", inst.kernel.elfSize);
    if (inst.indexSpaceGeometry[0] != 3 || inst.indexSpaceGeometry[1] != 640 ||
        inst.indexSpaceGeometry[2] != 10) { printf("index space mismatch\n"); return 1; }
    // w2: K=2560 -> torch packed [..., ..., 1280] -> 5 exact tiles.
    params.inputTensors[1].geometry.maxSizes[0] = 1280;
    params.inputTensors[2].geometry.maxSizes[0] = 160;
    params.outputTensors[0].geometry.maxSizes[0] = 2560;
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = in_patterns;
    inst.outputTensorAccessPattern = out_patterns;
    status = instantiate(&params, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 5) {
        printf("w2 instantiate failed: %d tiles=%lu\n", status, inst.indexSpaceGeometry[0]);
        return 1;
    }
    printf("w2 tiles: 5 (exact)\n");

    // ---- FP8 twin: same geometry plus scale2 [E], F8_143 output -----------
    const char* fp8 = "custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2";
    bool fp8_found = false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::strcmp(info[i].name, fp8) == 0) fp8_found = true;
    printf("fp8 guid registered: %s\n", fp8_found ? "YES" : "NO");
    if (!fp8_found) return 1;
    tpc_lib_api::Tensor fin[4]{};
    tpc_lib_api::Tensor fout[1]{};
    std::memset(&params, 0, sizeof(params));
    params.inputTensors = fin;
    params.outputTensors = fout;
    params.inputTensorNr = 4;
    params.outputTensorNr = 1;
    std::strcpy(params.guid.name, fp8);
    set_geom(fin[0], tpc_lib_api::DATA_I32, 1, {10});
    set_geom(fin[1], tpc_lib_api::DATA_U8, 3, {640, 2560, 512});
    set_geom(fin[2], tpc_lib_api::DATA_U8, 3, {80, 2560, 512});
    set_geom(fin[3], tpc_lib_api::DATA_F32, 1, {512});
    set_geom(fout[0], tpc_lib_api::DATA_F8_143, 3, {1280, 2560, 10});
    tpc_lib_api::TensorAccessPattern fin_patterns[4]{};
    tpc_lib_api::TensorAccessPattern fout_patterns[1]{};
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = fin_patterns;
    inst.outputTensorAccessPattern = fout_patterns;
    status = instantiate(&params, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 3 ||
        inst.indexSpaceGeometry[1] != 2560 || inst.indexSpaceGeometry[2] != 10) {
        printf("fp8 instantiate failed: %d space=%lu x %lu x %lu\n", status,
               inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1], inst.indexSpaceGeometry[2]);
        return 1;
    }
    printf("fp8 index space: %lu x %lu x %lu (expect 3 x 2560 x 10)\n",
           inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1], inst.indexSpaceGeometry[2]);
    // w2: K=2560 -> 5 tiles, scale2 stays [512].
    fin[1].geometry.maxSizes[0] = 1280;
    fin[2].geometry.maxSizes[0] = 160;
    fout[0].geometry.maxSizes[0] = 2560;
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = fin_patterns;
    inst.outputTensorAccessPattern = fout_patterns;
    status = instantiate(&params, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 5) {
        printf("fp8 w2 instantiate failed: %d tiles=%lu\n", status, inst.indexSpaceGeometry[0]);
        return 1;
    }
    printf("fp8 w2 tiles: 5 (exact)\n");

    const char* gemv = "custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2";
    bool gemv_found = false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::strcmp(info[i].name, gemv) == 0) gemv_found = true;
    printf("gemv guid registered: %s\n", gemv_found ? "YES" : "NO");
    if (!gemv_found) return 1;
    tpc_lib_api::HabanaKernelParams gparams{};
    tpc_lib_api::Tensor gin[5]{};
    tpc_lib_api::Tensor gout[1]{};
    std::memset(&gparams, 0, sizeof(gparams));
    gparams.inputTensors = gin;
    gparams.outputTensors = gout;
    gparams.inputTensorNr = 5;
    gparams.outputTensorNr = 1;
    std::strcpy(gparams.guid.name, gemv);
    set_geom(gin[0], tpc_lib_api::DATA_I32, 1, {10});
    set_geom(gin[1], tpc_lib_api::DATA_U8, 3, {1280, 640, 512});
    set_geom(gin[2], tpc_lib_api::DATA_U8, 3, {160, 640, 512});
    set_geom(gin[3], tpc_lib_api::DATA_BF16, 2, {2560, 10});
    set_geom(gin[4], tpc_lib_api::DATA_F32, 1, {512});
    set_geom(gout[0], tpc_lib_api::DATA_BF16, 2, {640, 10});
    tpc_lib_api::TensorAccessPattern gin_patterns[5]{};
    tpc_lib_api::TensorAccessPattern gout_patterns[1]{};
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = gin_patterns;
    inst.outputTensorAccessPattern = gout_patterns;
    status = instantiate(&gparams, &inst);
    // H8: one member owns four output rows, so the row axis is N/4 (640/4 for
    // the gate shape) and every row-carrying tensor access pattern steps by 4.
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 160 ||
        inst.indexSpaceGeometry[1] != 10) {
        printf("gemv instantiate failed: %d space=%lu x %lu\n", status,
               inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1]);
        return 1;
    }
    if (inst.outputTensorAccessPattern[0].mapping[0].a != 4.0f ||
        inst.inputTensorAccessPattern[1].mapping[1].a != 4.0f ||
        inst.inputTensorAccessPattern[2].mapping[1].a != 4.0f) {
        printf("gemv row stride mismatch: out %g packed %g scales %g (expect 4)\n",
               inst.outputTensorAccessPattern[0].mapping[0].a,
               inst.inputTensorAccessPattern[1].mapping[1].a,
               inst.inputTensorAccessPattern[2].mapping[1].a);
        return 1;
    }
    // Per-slot x (x_rows == S == 10): dim 1 of x keeps the identity mapping.
    if (inst.inputTensorAccessPattern[3].mapping[1].a != 1.0f) {
        printf("gemv per-slot x mapping mismatch: a=%g (expect 1)\n",
               inst.inputTensorAccessPattern[3].mapping[1].a);
        return 1;
    }
    printf("gemv index space: %lu x %lu (expect 160 x 10, row stride 4)\n",
           inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1]);

    // ---- H10 broadcast x: one row per token, S = T * top_k -----------------
    // T=1, S=10: x is [1, K] in torch order, i.e. [2560, 1] here. The slot
    // axis stays 10 wide, and x dim 1 must declare the full [0, 0] range with
    // a == 0 so no member's x_row is clipped to its own slot.
    set_geom(gin[3], tpc_lib_api::DATA_BF16, 2, {2560, 1});
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = gin_patterns;
    inst.outputTensorAccessPattern = gout_patterns;
    status = instantiate(&gparams, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 160 ||
        inst.indexSpaceGeometry[1] != 10 ||
        inst.inputTensorAccessPattern[3].mapping[1].a != 0.0f ||
        inst.inputTensorAccessPattern[3].mapping[1].start_b != 0.0f ||
        inst.inputTensorAccessPattern[3].mapping[1].end_b != 0.0f) {
        printf("gemv H10 T=1 instantiate failed: %d space=%lu x %lu x mapping "
               "a=%g range=[%g,%g] (expect 160 x 10, a=0 range=[0,0])\n",
               status, inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1],
               inst.inputTensorAccessPattern[3].mapping[1].a,
               inst.inputTensorAccessPattern[3].mapping[1].start_b,
               inst.inputTensorAccessPattern[3].mapping[1].end_b);
        return 1;
    }
    printf("gemv H10 x_rows=1 (T=1, S=10): space 160 x 10, x dim1 a=0 range=[0,0]\n");
    // T=4, S=40 (top_k=10): x is [4, K] in torch order, i.e. [2560, 4] here.
    set_geom(gin[0], tpc_lib_api::DATA_I32, 1, {40});
    set_geom(gin[3], tpc_lib_api::DATA_BF16, 2, {2560, 4});
    set_geom(gout[0], tpc_lib_api::DATA_BF16, 2, {640, 40});
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = gin_patterns;
    inst.outputTensorAccessPattern = gout_patterns;
    status = instantiate(&gparams, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 160 ||
        inst.indexSpaceGeometry[1] != 40 ||
        inst.inputTensorAccessPattern[3].mapping[1].a != 0.0f ||
        inst.inputTensorAccessPattern[3].mapping[1].start_b != 0.0f ||
        inst.inputTensorAccessPattern[3].mapping[1].end_b != 3.0f) {
        printf("gemv H10 T=4 instantiate failed: %d space=%lu x %lu x mapping "
               "a=%g range=[%g,%g] (expect 160 x 40, a=0 range=[0,3])\n",
               status, inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1],
               inst.inputTensorAccessPattern[3].mapping[1].a,
               inst.inputTensorAccessPattern[3].mapping[1].start_b,
               inst.inputTensorAccessPattern[3].mapping[1].end_b);
        return 1;
    }
    printf("gemv H10 x_rows=4 (T=4, S=40): space 160 x 40, x dim1 a=0 range=[0,3]\n");
    // Negative: x_rows=3 does not divide S=40 and must still be rejected.
    set_geom(gin[3], tpc_lib_api::DATA_BF16, 2, {2560, 3});
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = gin_patterns;
    inst.outputTensorAccessPattern = gout_patterns;
    status = instantiate(&gparams, &inst);
    if (status == tpc_lib_api::GLUE_SUCCESS) {
        printf("gemv H10 invalid x_rows=3 for S=40 was accepted\n");
        return 1;
    }
    printf("gemv H10 invalid x_rows=3 for S=40 rejected: %d\n", status);

    // ---- H15 fused SwiGLU GEMV: gate shape N=640 K=2560, T=1 (x [2560,1]) --
    const char* swiglu = "custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2";
    bool swiglu_found = false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::strcmp(info[i].name, swiglu) == 0) swiglu_found = true;
    printf("h15 swiglu gemv guid registered: %s\n", swiglu_found ? "YES" : "NO");
    if (!swiglu_found) return 1;
    tpc_lib_api::HabanaKernelParams sparams{};
    tpc_lib_api::Tensor sin[8]{};
    tpc_lib_api::Tensor sout[1]{};
    std::memset(&sparams, 0, sizeof(sparams));
    sparams.inputTensors = sin;
    sparams.outputTensors = sout;
    sparams.inputTensorNr = 8;
    sparams.outputTensorNr = 1;
    std::strcpy(sparams.guid.name, swiglu);
    set_geom(sin[0], tpc_lib_api::DATA_I32, 1, {10});
    set_geom(sin[1], tpc_lib_api::DATA_U8, 3, {1280, 640, 512});
    set_geom(sin[2], tpc_lib_api::DATA_U8, 3, {160, 640, 512});
    set_geom(sin[3], tpc_lib_api::DATA_F32, 1, {512});
    set_geom(sin[4], tpc_lib_api::DATA_U8, 3, {1280, 640, 512});
    set_geom(sin[5], tpc_lib_api::DATA_U8, 3, {160, 640, 512});
    set_geom(sin[6], tpc_lib_api::DATA_F32, 1, {512});
    set_geom(sin[7], tpc_lib_api::DATA_BF16, 2, {2560, 1});
    set_geom(sout[0], tpc_lib_api::DATA_BF16, 2, {640, 10});
    tpc_lib_api::TensorAccessPattern sin_patterns[8]{};
    tpc_lib_api::TensorAccessPattern sout_patterns[1]{};
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = sin_patterns;
    inst.outputTensorAccessPattern = sout_patterns;
    status = instantiate(&sparams, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 160 ||
        inst.indexSpaceGeometry[1] != 10 ||
        inst.outputTensorAccessPattern[0].mapping[0].a != 4.0f) {
        printf("h15 swiglu instantiate failed: %d space=%lu x %lu\n", status,
               inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1]);
        return 1;
    }
    printf("h15 swiglu index space: 160 x 10 (row stride 4)\n");

    // ---- H15 router top-k: logits [512,4] -> ids/weights [10,4] ------------
    const char* router = "custom_qwen4exp_moe_router_topk_bf16_gaudi2";
    bool router_found = false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::strcmp(info[i].name, router) == 0) router_found = true;
    if (!router_found) { printf("h15 router guid missing\n"); return 1; }
    tpc_lib_api::HabanaKernelParams rparams{};
    tpc_lib_api::Tensor rin[1]{};
    tpc_lib_api::Tensor rout[2]{};
    std::memset(&rparams, 0, sizeof(rparams));
    rparams.inputTensors = rin;
    rparams.outputTensors = rout;
    rparams.inputTensorNr = 1;
    rparams.outputTensorNr = 2;
    std::strcpy(rparams.guid.name, router);
    set_geom(rin[0], tpc_lib_api::DATA_BF16, 2, {512, 4});
    set_geom(rout[0], tpc_lib_api::DATA_I32, 2, {10, 4});
    set_geom(rout[1], tpc_lib_api::DATA_BF16, 2, {10, 4});
    tpc_lib_api::TensorAccessPattern rin_patterns[1]{};
    tpc_lib_api::TensorAccessPattern rout_patterns[2]{};
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = rin_patterns;
    inst.outputTensorAccessPattern = rout_patterns;
    status = instantiate(&rparams, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 4) {
        printf("h15 router instantiate failed: %d space=%lu\n", status, inst.indexSpaceGeometry[0]);
        return 1;
    }
    printf("h15 router index space: 4 (tokens)\n");

    // ---- H15 shared SwiGLU epilogue: merged [1281,4] -> [640,4] ------------
    const char* shared = "custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2";
    bool shared_found = false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::strcmp(info[i].name, shared) == 0) shared_found = true;
    if (!shared_found) { printf("h15 shared guid missing\n"); return 1; }
    tpc_lib_api::HabanaKernelParams hparams{};
    tpc_lib_api::Tensor hin[1]{};
    tpc_lib_api::Tensor hout[1]{};
    std::memset(&hparams, 0, sizeof(hparams));
    hparams.inputTensors = hin;
    hparams.outputTensors = hout;
    hparams.inputTensorNr = 1;
    hparams.outputTensorNr = 1;
    std::strcpy(hparams.guid.name, shared);
    set_geom(hin[0], tpc_lib_api::DATA_BF16, 2, {1281, 4});
    set_geom(hout[0], tpc_lib_api::DATA_BF16, 2, {640, 4});
    tpc_lib_api::TensorAccessPattern hin_patterns[1]{};
    tpc_lib_api::TensorAccessPattern hout_patterns[1]{};
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = hin_patterns;
    inst.outputTensorAccessPattern = hout_patterns;
    status = instantiate(&hparams, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 5 ||
        inst.indexSpaceGeometry[1] != 4) {
        printf("h15 shared instantiate failed: %d space=%lu x %lu\n", status,
               inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1]);
        return 1;
    }
    printf("h15 shared index space: 5 x 4\n");

    // ---- H15 down epilogue: [40,2560 slots] x [4,10] -> [4,2560] -----------
    const char* depilogue = "custom_qwen4exp_moe_down_epilogue_bf16_gaudi2";
    bool depilogue_found = false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::strcmp(info[i].name, depilogue) == 0) depilogue_found = true;
    if (!depilogue_found) { printf("h15 down epilogue guid missing\n"); return 1; }
    tpc_lib_api::HabanaKernelParams dparams{};
    tpc_lib_api::Tensor din[2]{};
    tpc_lib_api::Tensor dout[1]{};
    std::memset(&dparams, 0, sizeof(dparams));
    dparams.inputTensors = din;
    dparams.outputTensors = dout;
    dparams.inputTensorNr = 2;
    dparams.outputTensorNr = 1;
    std::strcpy(dparams.guid.name, depilogue);
    set_geom(din[0], tpc_lib_api::DATA_BF16, 2, {2560, 40});
    set_geom(din[1], tpc_lib_api::DATA_BF16, 2, {10, 4});
    set_geom(dout[0], tpc_lib_api::DATA_BF16, 2, {2560, 4});
    tpc_lib_api::TensorAccessPattern din_patterns[2]{};
    tpc_lib_api::TensorAccessPattern dout_patterns[1]{};
    std::memset(&inst, 0, sizeof(inst));
    inst.kernel.kernelElf = elf_buffer;
    inst.kernel.elfSize = sizeof(elf_buffer);
    inst.inputTensorAccessPattern = din_patterns;
    inst.outputTensorAccessPattern = dout_patterns;
    status = instantiate(&dparams, &inst);
    if (status != tpc_lib_api::GLUE_SUCCESS || inst.indexSpaceGeometry[0] != 20 ||
        inst.indexSpaceGeometry[1] != 4) {
        printf("h15 down epilogue instantiate failed: %d space=%lu x %lu\n", status,
               inst.indexSpaceGeometry[0], inst.indexSpaceGeometry[1]);
        return 1;
    }
    printf("h15 down epilogue index space: 20 x 4\n");
    printf("glue validation: all tests passed\n");
    return 0;
}
