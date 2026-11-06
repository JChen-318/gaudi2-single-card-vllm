// SPDX-License-Identifier: Apache-2.0
// Host-side validation of the Qwen4Exp NVFP4 MoE TPC library: GUID
// registration, GlueCode geometry and the documented caps. No device is
// opened; the TPC ELFs are only copied into the caller's buffer.
//
//   cmake -S native/qwen4exp-nvfp4-moe -B build/native/qwen4exp-nvfp4-moe
//   cmake --build build/native/qwen4exp-nvfp4-moe --parallel
//   build/native/qwen4exp-nvfp4-moe/qwen4exp_nvfp4_moe_geometry_test \
//       build/native/qwen4exp-nvfp4-moe/libqwen4exp_nvfp4_moe_kernels.so
//
// Shapes are the ones this checkpoint serves: hidden 2560, moe intermediate
// 640, 512 experts, 10 routed slots per token.
#include "qwen4exp_nvfp4_moe.h"
#include "tpc_kernel_lib_interface.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

using tpc_lib_api::GlueCodeReturn;

int failures = 0;

void expect(bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

void set_geometry(tpc_lib_api::Tensor& tensor, tpc_lib_api::TensorDataType type,
                  std::initializer_list<uint64_t> sizes) {
    tensor.geometry.dataType = type;
    tensor.geometry.dims = static_cast<uint32_t>(sizes.size());
    uint32_t i = 0;
    for (uint64_t size : sizes) {
        tensor.geometry.maxSizes[i] = size;
        tensor.geometry.minSizes[i] = size;
        ++i;
    }
}

struct Instantiation {
    tpc_lib_api::HabanaKernelInstantiation inst{};
    std::vector<tpc_lib_api::TensorAccessPattern> in_patterns;
    std::vector<tpc_lib_api::TensorAccessPattern> out_patterns;
    std::vector<char> elf = std::vector<char>(1 << 20);

    Instantiation(unsigned inputs, unsigned outputs) {
        in_patterns.resize(inputs > 0 ? inputs : 1);
        out_patterns.resize(outputs > 0 ? outputs : 1);
        std::memset(in_patterns.data(), 0, sizeof(tpc_lib_api::TensorAccessPattern) * in_patterns.size());
        std::memset(out_patterns.data(), 0, sizeof(tpc_lib_api::TensorAccessPattern) * out_patterns.size());
        inst.kernel.kernelElf = elf.data();
        inst.kernel.elfSize = static_cast<uint32_t>(elf.size());
        inst.inputTensorAccessPattern = in_patterns.data();
        inst.outputTensorAccessPattern = out_patterns.data();
    }
};

struct Request {
    tpc_lib_api::HabanaKernelParams params{};
    std::vector<tpc_lib_api::Tensor> inputs;
    std::vector<tpc_lib_api::Tensor> outputs;

    Request(const char* guid, unsigned n_in, unsigned n_out) {
        inputs.resize(n_in);
        outputs.resize(n_out);
        std::memset(inputs.data(), 0, sizeof(tpc_lib_api::Tensor) * n_in);
        std::memset(outputs.data(), 0, sizeof(tpc_lib_api::Tensor) * n_out);
        params.inputTensors = inputs.data();
        params.outputTensors = outputs.data();
        params.inputTensorNr = n_in;
        params.outputTensorNr = n_out;
        params.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        std::strncpy(params.guid.name, guid, tpc_lib_api::MAX_NODE_NAME - 1);
    }
};

// Node params exactly as the operator passes them: the count kernel takes one
// int, the scatter kernel three (its declaration order).
struct CountParams {
    int32_t experts;
};

struct ScatterParams {
    int32_t experts;
    int32_t slots;
    int32_t cap;
};

}  // namespace

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    const char* library = argc > 1 ? argv[1]
                                   : "./libqwen4exp_nvfp4_moe_kernels.so";
    void* handle = dlopen(library, RTLD_NOW);
    if (!handle) {
        std::printf("dlopen failed (%s): %s\n", library, dlerror());
        return 1;
    }
    auto guids = reinterpret_cast<GlueCodeReturn (*)(tpc_lib_api::DeviceId, uint32_t*,
                                                     tpc_lib_api::GuidInfo*)>(
        dlsym(handle, "GetKernelGuids"));
    auto instantiate = reinterpret_cast<GlueCodeReturn (*)(
        const tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*)>(
        dlsym(handle, "InstantiateTpcKernel"));
    if (!guids || !instantiate) {
        std::printf("symbols missing\n");
        return 1;
    }

    std::printf("guid registration:\n");
    uint32_t count = 0;
    if (guids(tpc_lib_api::DEVICE_ID_GAUDI2, &count, nullptr) != tpc_lib_api::GLUE_SUCCESS) {
        std::printf("count query failed\n");
        return 1;
    }
    std::vector<tpc_lib_api::GuidInfo> info(count);
    if (guids(tpc_lib_api::DEVICE_ID_GAUDI2, &count, info.data()) != tpc_lib_api::GLUE_SUCCESS) {
        std::printf("guid query failed\n");
        return 1;
    }
    const char* custom[] = {
        "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2",
        "custom_qwen4exp_nvfp4_moe_count_i32_gaudi2",
        "custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2",
        "custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2",
    };
    for (const char* target : custom) {
        bool found = false;
        for (uint32_t i = 0; i < count; ++i)
            if (std::strcmp(info[i].name, target) == 0) found = true;
        char line[160];
        std::snprintf(line, sizeof(line), "%s reported", target);
        expect(found, line);
    }
    std::printf("  (%u guids total, stock included)\n", count);

    // ---- decode GEMV, w13: K = 2560, N = 1280 (2*640), E = 512, S = 10 ------
    std::printf("gemv w13 (K=2560 N=1280 E=512 S=10):\n");
    {
        Request r(custom[0], 5, 1);
        set_geometry(r.inputs[0], tpc_lib_api::DATA_I32, {10});
        set_geometry(r.inputs[1], tpc_lib_api::DATA_U8, {1280, 1280, 512});
        set_geometry(r.inputs[2], tpc_lib_api::DATA_U8, {160, 1280, 512});
        set_geometry(r.inputs[3], tpc_lib_api::DATA_BF16, {2560, 10});
        set_geometry(r.inputs[4], tpc_lib_api::DATA_F32, {512});
        set_geometry(r.outputs[0], tpc_lib_api::DATA_BF16, {1280, 10});
        Instantiation it(5, 1);
        const auto status = instantiate(&r.params, &it.inst);
        expect(status == tpc_lib_api::GLUE_SUCCESS, "instantiate");
        char line[160];
        std::snprintf(line, sizeof(line), "index space %lu x %lu (expect %u x 10)",
                      it.inst.indexSpaceGeometry[0], it.inst.indexSpaceGeometry[1],
                      1280 / QWEN4EXP_MOE_ROWS_PER_MEMBER);
        expect(it.inst.indexSpaceGeometry[0] == 1280 / QWEN4EXP_MOE_ROWS_PER_MEMBER &&
                   it.inst.indexSpaceGeometry[1] == 10, line);
        std::snprintf(line, sizeof(line), "elf %u bytes", it.inst.kernel.elfSize);
        expect(it.inst.kernel.elfSize > 0, line);
        expect(it.inst.inputTensorAccessPattern[1].mapping[1].a ==
                   static_cast<float>(QWEN4EXP_MOE_ROWS_PER_MEMBER),
               "packed row dim steps by ROWS_PER_MEMBER");
    }

    // ---- decode GEMV, w2: K = 640, N = 2560 --------------------------------
    std::printf("gemv w2 (K=640 N=2560 E=512 S=10):\n");
    {
        Request r(custom[0], 5, 1);
        set_geometry(r.inputs[0], tpc_lib_api::DATA_I32, {10});
        set_geometry(r.inputs[1], tpc_lib_api::DATA_U8, {320, 2560, 512});
        set_geometry(r.inputs[2], tpc_lib_api::DATA_U8, {40, 2560, 512});
        set_geometry(r.inputs[3], tpc_lib_api::DATA_BF16, {640, 10});
        set_geometry(r.inputs[4], tpc_lib_api::DATA_F32, {512});
        set_geometry(r.outputs[0], tpc_lib_api::DATA_BF16, {2560, 10});
        Instantiation it(5, 1);
        const auto status = instantiate(&r.params, &it.inst);
        expect(status == tpc_lib_api::GLUE_SUCCESS, "instantiate");
        expect(it.inst.indexSpaceGeometry[0] == 2560 / QWEN4EXP_MOE_ROWS_PER_MEMBER &&
                   it.inst.indexSpaceGeometry[1] == 10,
               "index space 640 x 10");
    }

    // ---- decode GEMV caps --------------------------------------------------
    std::printf("gemv caps:\n");
    {
        auto rejects = [&](const char* what, uint64_t ids, uint64_t k_half,
                           uint64_t rows, uint64_t k_elements) {
            Request r(custom[0], 5, 1);
            set_geometry(r.inputs[0], tpc_lib_api::DATA_I32, {ids});
            set_geometry(r.inputs[1], tpc_lib_api::DATA_U8, {k_half, rows, 512});
            set_geometry(r.inputs[2], tpc_lib_api::DATA_U8, {k_half / 8, rows, 512});
            set_geometry(r.inputs[3], tpc_lib_api::DATA_BF16, {k_elements, ids});
            set_geometry(r.inputs[4], tpc_lib_api::DATA_F32, {512});
            set_geometry(r.outputs[0], tpc_lib_api::DATA_BF16, {rows, ids});
            Instantiation it(5, 1);
            expect(instantiate(&r.params, &it.inst) == tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE,
                   what);
        };
        rejects("K % 128 != 0 rejected", 10, 640 + 32, 1280, 2 * (640 + 32));
        rejects("N % ROWS_PER_MEMBER != 0 rejected", 10, 1280, 1282, 2560);
        rejects("S > 64 rejected", 65, 1280, 1280, 2560);
        rejects("x K mismatch rejected", 10, 1280, 1280, 2560 - 128);
    }

    // ---- prefill bucketing: count -> prefix -> scatter ---------------------
    std::printf("bucket count (E=512 S=10):\n");
    {
        Request r(custom[1], 1, 1);
        CountParams params{512};
        r.params.nodeParams.nodeParams = &params;
        r.params.nodeParams.nodeParamsSize = sizeof(params);
        set_geometry(r.inputs[0], tpc_lib_api::DATA_I32, {10});
        set_geometry(r.outputs[0], tpc_lib_api::DATA_I32, {512, 1});
        Instantiation it(1, 1);
        expect(instantiate(&r.params, &it.inst) == tpc_lib_api::GLUE_SUCCESS, "instantiate");
        expect(it.inst.indexSpaceGeometry[0] == 1 && it.inst.indexSpaceGeometry[1] == 8,
               "index space 1 x 8 (blocks x E/64)");
        // A slot list longer than one block: S = 300 -> 5 blocks.
        CountParams wide{512};
        Request w(custom[1], 1, 1);
        w.params.nodeParams.nodeParams = &wide;
        w.params.nodeParams.nodeParamsSize = sizeof(wide);
        set_geometry(w.inputs[0], tpc_lib_api::DATA_I32, {300});
        set_geometry(w.outputs[0], tpc_lib_api::DATA_I32, {512, 5});
        Instantiation wit(1, 1);
        expect(instantiate(&w.params, &wit.inst) == tpc_lib_api::GLUE_SUCCESS,
               "instantiate S=300");
        expect(wit.inst.indexSpaceGeometry[0] == 5 && wit.inst.indexSpaceGeometry[1] == 8,
               "index space 5 x 8");
        // outputs re-derived when the graph compiler passes an empty geometry
        Request g(custom[1], 1, 1);
        g.params.nodeParams.nodeParams = &params;
        g.params.nodeParams.nodeParamsSize = sizeof(params);
        set_geometry(g.inputs[0], tpc_lib_api::DATA_I32, {10});
        set_geometry(g.outputs[0], tpc_lib_api::DATA_I32, {});
        Instantiation git(1, 1);
        expect(instantiate(&g.params, &git.inst) == tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE &&
                   g.outputs[0].geometry.maxSizes[0] == 512 &&
                   g.outputs[0].geometry.maxSizes[1] == 1,
               "counts shape re-derived [512, 1]");
        // E not a multiple of 64 is rejected
        CountParams bad{500};
        Request b(custom[1], 1, 1);
        b.params.nodeParams.nodeParams = &bad;
        b.params.nodeParams.nodeParamsSize = sizeof(bad);
        set_geometry(b.inputs[0], tpc_lib_api::DATA_I32, {10});
        set_geometry(b.outputs[0], tpc_lib_api::DATA_I32, {500, 1});
        Instantiation bit(1, 1);
        expect(instantiate(&b.params, &bit.inst) ==
                   tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION,
               "E % 64 != 0 rejected");
    }

    std::printf("bucket prefix (E=512 blocks=5):\n");
    {
        Request r(custom[2], 1, 2);
        set_geometry(r.inputs[0], tpc_lib_api::DATA_I32, {512, 5});
        set_geometry(r.outputs[0], tpc_lib_api::DATA_I32, {512, 5});
        set_geometry(r.outputs[1], tpc_lib_api::DATA_I32, {512});
        Instantiation it(1, 2);
        expect(instantiate(&r.params, &it.inst) == tpc_lib_api::GLUE_SUCCESS, "instantiate");
        expect(it.inst.indexSpaceGeometry[0] == 8, "index space 8 members");
        expect(it.inst.kernel.paramsNr == 0, "no scalar params");
    }

    std::printf("bucket scatter (E=512 S=300 cap=300):\n");
    {
        Request r(custom[3], 2, 2);
        ScatterParams params{512, 300, 300};
        r.params.nodeParams.nodeParams = &params;
        r.params.nodeParams.nodeParamsSize = sizeof(params);
        set_geometry(r.inputs[0], tpc_lib_api::DATA_I32, {300});
        set_geometry(r.inputs[1], tpc_lib_api::DATA_I32, {512, 5});
        set_geometry(r.outputs[0], tpc_lib_api::DATA_I32, {512u * 300u});
        set_geometry(r.outputs[1], tpc_lib_api::DATA_I32, {300});
        Instantiation it(2, 2);
        expect(instantiate(&r.params, &it.inst) == tpc_lib_api::GLUE_SUCCESS, "instantiate");
        expect(it.inst.indexSpaceGeometry[0] == 5, "index space 5 blocks");
        expect(it.inst.kernel.paramsNr == 3 &&
                   it.inst.kernel.scalarParams[0] == 512 &&
                   it.inst.kernel.scalarParams[1] == 300 &&
                   it.inst.kernel.scalarParams[2] == 300,
               "scalar params (experts, slots, cap)");
        // cap < slots would let a bucket row run past its expert's stride
        ScatterParams narrow{512, 300, 64};
        Request n(custom[3], 2, 2);
        n.params.nodeParams.nodeParams = &narrow;
        n.params.nodeParams.nodeParamsSize = sizeof(narrow);
        set_geometry(n.inputs[0], tpc_lib_api::DATA_I32, {300});
        set_geometry(n.inputs[1], tpc_lib_api::DATA_I32, {512, 5});
        set_geometry(n.outputs[0], tpc_lib_api::DATA_I32, {512u * 64u});
        set_geometry(n.outputs[1], tpc_lib_api::DATA_I32, {300});
        Instantiation nit(2, 2);
        expect(instantiate(&n.params, &nit.inst) ==
                   tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION,
               "cap < slots rejected");
        // slots scalar disagreeing with the ids length is rejected
        ScatterParams mismatch{512, 256, 300};
        Request m(custom[3], 2, 2);
        m.params.nodeParams.nodeParams = &mismatch;
        m.params.nodeParams.nodeParamsSize = sizeof(mismatch);
        set_geometry(m.inputs[0], tpc_lib_api::DATA_I32, {300});
        set_geometry(m.inputs[1], tpc_lib_api::DATA_I32, {512, 5});
        set_geometry(m.outputs[0], tpc_lib_api::DATA_I32, {512u * 300u});
        set_geometry(m.outputs[1], tpc_lib_api::DATA_I32, {300});
        Instantiation mit(2, 2);
        expect(instantiate(&m.params, &mit.inst) == tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE,
               "slots/ids length mismatch rejected");
    }

    if (failures == 0) std::printf("glue validation: all tests passed\n");
    else std::printf("glue validation: %d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
