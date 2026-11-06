// SPDX-License-Identifier: Apache-2.0
// Per-call latency estimate for the Qwen4Exp NVFP4 MoE decode GEMV, from the
// TPC simulator (host only, no device): the real layer shapes of this
// checkpoint, K = 2560 / N = 1280 (gate+up) and K = 640 / N = 2560 (down),
// S = 10 routed slots (one decode token, top-10 routing) and S = 1.
//
// The simulator returns the cycle count of the whole index space executed on
// one TPC. RunSimulation counts one cycle per issued instruction (cycles ==
// instructionsExecuted in every run), so the estimate below assumes 1 IPC -
// the same convention native/gdn-fused-kernel/README.md uses when its
// simulator cycles (2.5 M) are spread over Gaudi2's 24 TPCs and compared with a
// device trace (63 us at ~1.65 GHz). Treat the result as a compute-side
// estimate (+-10% for clock/occupancy); the simulator does not model HBM
// latency or bandwidth, and the kernel is compute-bound on the decode.
//
// E is the minimum the glue accepts (64) because a member's work does not
// depend on E: it reads the rows of one expert plane and only the expert index
// changes. run_expert_independence() measures that on a small shape.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "qwen4exp_nvfp4_moe.h"
#include "qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.hpp"
#include "tpc_test_core_api.h"

using namespace tpc_lib_api;
using namespace tpc_tests;

namespace {

int failures = 0;

constexpr int kTpcs = 24;             // Gaudi2 TPC engines
constexpr double kTpcClockHz = 1.7e9; // nominal TPC clock
constexpr int kLayers = 48;           // MoE layers of this checkpoint

TensorDesc2 describe(void* data, std::initializer_list<uint32_t> sizes, int element_bits) {
    TensorDesc2 desc = {};
    desc.baseAddrUnion.baseAddr = reinterpret_cast<uint64_t>(data);
    uint32_t stride = 1;
    uint32_t mask = 0;
    uint32_t dim = 0;
    for (uint32_t size : sizes) {
        desc.dimDescriptors[dim].size = size;
        desc.dimDescriptors[dim].stride = stride;
        stride *= size;
        mask |= 1u << dim;
        ++dim;
    }
    const int element_type = element_bits == 8 ? 0 : element_bits == 16 ? 1 : 2;
    desc.configuration = static_cast<uint32_t>(element_type) | (mask << 8) |
                         (static_cast<uint32_t>(dim - 1) << 16);
    return desc;
}

void geometry(Tensor& tensor, TensorDataType type, std::initializer_list<uint64_t> sizes) {
    tensor.geometry.dims = static_cast<uint32_t>(sizes.size());
    tensor.geometry.dataType = type;
    uint32_t i = 0;
    for (uint64_t size : sizes) {
        tensor.geometry.maxSizes[i] = size;
        tensor.geometry.minSizes[i] = size;
        ++i;
    }
}

struct Shape {
    int k;
    int rows;
    int experts;
    int slots;
};

struct Measurement {
    unsigned cycles = 0;
    long members = 0;
    double cycles_per_member = 0;
    double per_call_us = 0;
    bool ok = false;
};

// Uniform data: the simulator's cycle count does not depend on the values, only
// on the instruction stream (branches on `ok`/tile bounds are data-independent
// for valid ids and K a multiple of 512).
Measurement measure(const Shape& shape) {
    const int k = shape.k, rows = shape.rows, experts = shape.experts, slots = shape.slots;
    Measurement result;
    if (k % 128 != 0 || rows % 4 != 0 || experts < 64 || experts % 64 != 0 || slots < 1 ||
        slots > 64) {
        std::printf("  shape K=%d N=%d E=%d S=%d is outside the kernel caps\n", k, rows, experts,
                    slots);
        return result;
    }
    std::vector<uint8_t> packed(static_cast<size_t>(experts) * rows * (k / 2), 0x21);
    std::vector<uint8_t> scales(static_cast<size_t>(experts) * rows * (k / 16), 0x3C);
    std::vector<int32_t> ids(static_cast<size_t>(slots));
    for (int s = 0; s < slots; ++s) ids[s] = s % experts;
    std::vector<float> scale2(static_cast<size_t>(experts), 1.0f);
    std::vector<uint16_t> x(static_cast<size_t>(slots) * k, 0x3c00);
    std::vector<uint16_t> out(static_cast<size_t>(slots) * rows, 0);

    HabanaKernelParams params{};
    HabanaKernelInstantiation inst{};
    std::vector<Tensor> inputs(5);
    std::vector<Tensor> outputs(1);
    std::vector<TensorAccessPattern> in_ap(5);
    std::vector<TensorAccessPattern> out_ap(1);
    std::vector<char> elf(1 << 20);
    std::memset(inputs.data(), 0, sizeof(Tensor) * inputs.size());
    std::memset(outputs.data(), 0, sizeof(Tensor) * outputs.size());
    std::memset(in_ap.data(), 0, sizeof(TensorAccessPattern) * in_ap.size());
    std::memset(out_ap.data(), 0, sizeof(TensorAccessPattern) * out_ap.size());
    params.inputTensors = inputs.data();
    params.outputTensors = outputs.data();
    params.inputTensorNr = 5;
    params.outputTensorNr = 1;
    params.deviceId = DEVICE_ID_GAUDI2;
    std::strncpy(params.guid.name, "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2",
                 MAX_NODE_NAME - 1);
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = static_cast<uint32_t>(elf.size());
    inst.inputTensorAccessPattern = in_ap.data();
    inst.outputTensorAccessPattern = out_ap.data();

    geometry(inputs[0], DATA_I32, {static_cast<uint64_t>(slots)});
    geometry(inputs[1], DATA_U8,
             {static_cast<uint64_t>(k / 2), static_cast<uint64_t>(rows),
              static_cast<uint64_t>(experts)});
    geometry(inputs[2], DATA_U8,
             {static_cast<uint64_t>(k / 16), static_cast<uint64_t>(rows),
              static_cast<uint64_t>(experts)});
    geometry(inputs[3], DATA_BF16,
             {static_cast<uint64_t>(k), static_cast<uint64_t>(slots)});
    geometry(inputs[4], DATA_F32, {static_cast<uint64_t>(experts)});
    geometry(outputs[0], DATA_BF16,
             {static_cast<uint64_t>(rows), static_cast<uint64_t>(slots)});

    Qwen4ExpNvfp4MoeGemvBF16Gaudi2 glue;
    if (glue.GetGcDefinitions(&params, &inst) != GLUE_SUCCESS) {
        std::printf("  glue rejected K=%d N=%d E=%d S=%d\n", k, rows, experts, slots);
        return result;
    }
    std::vector<TensorDesc2> descs = {
        describe(ids.data(), {static_cast<uint32_t>(slots)}, 32),
        describe(packed.data(),
                 {static_cast<uint32_t>(k / 2), static_cast<uint32_t>(rows),
                  static_cast<uint32_t>(experts)},
                 8),
        describe(scales.data(),
                 {static_cast<uint32_t>(k / 16), static_cast<uint32_t>(rows),
                  static_cast<uint32_t>(experts)},
                 8),
        describe(x.data(), {static_cast<uint32_t>(k), static_cast<uint32_t>(slots)}, 16),
        describe(scale2.data(), {static_cast<uint32_t>(experts)}, 32),
        describe(out.data(), {static_cast<uint32_t>(rows), static_cast<uint32_t>(slots)}, 16),
    };
    VPEStats stats;
    result.cycles = RunSimulation(params, inst, descs, stats);
    result.members = static_cast<long>(inst.indexSpaceGeometry[0]) *
                     static_cast<long>(inst.indexSpaceGeometry[1]);
    if (result.members <= 0 || result.cycles == 0) return result;
    result.cycles_per_member = static_cast<double>(result.cycles) / result.members;
    // One member per thread: a TPC takes whole members, so the slowest TPC does
    // ceil(members / 24) of them.
    const long per_tpc = (result.members + kTpcs - 1) / kTpcs;
    const double per_call_cycles = static_cast<double>(per_tpc) * result.cycles_per_member;
    result.per_call_us = per_call_cycles / kTpcClockHz * 1e6;
    result.ok = true;
    return result;
}

Measurement report(const char* name, const Shape& shape) {
    const Measurement m = measure(shape);
    if (!m.ok) return m;
    std::printf("%-12s K=%4d N=%4d E=%3d S=%2d: members=%ld cycles=%u cycles/member=%.1f\n", name,
                shape.k, shape.rows, shape.experts, shape.slots, m.members, m.cycles,
                m.cycles_per_member);
    std::printf("             per call: %.3f ms  (%ld members / 24 TPCs x %.1f cycles / %.1f GHz)\n",
                m.per_call_us / 1e3, m.members, m.cycles_per_member, kTpcClockHz / 1e9);
    return m;
}

// E only selects the expert plane: a member's instruction stream does not
// depend on it. Measured, not assumed: the same shape at E = 64 and E = 128
// must produce the same cycle count.
void run_expert_independence() {
    const Shape small{512, 64, 64, 4};
    const Shape doubled{512, 64, 128, 4};
    const Measurement a = measure(small);
    const Measurement b = measure(doubled);
    if (!a.ok || !b.ok) return;
    std::printf("expert independence K=512 N=64 S=4: E=64 cycles=%u, E=128 cycles=%u  %s\n",
                a.cycles, b.cycles, a.cycles == b.cycles ? "identical" : "DIFFERENT");
    if (a.cycles != b.cycles) ++failures;
}

}  // namespace

int main() {
    std::printf("TPC simulator: NVFP4 MoE decode GEMV latency estimate (no device)\n");
    std::printf("assumptions: %d TPCs, %.1f GHz, simulator cycles at 1 IPC, E=64\n", kTpcs,
                kTpcClockHz / 1e9);
    run_expert_independence();
    const Measurement gate_up_10 = report("gate/up", Shape{2560, 1280, 64, 10});
    const Measurement down_10 = report("down", Shape{640, 2560, 64, 10});
    const Measurement gate_up_1 = report("gate/up", Shape{2560, 1280, 64, 1});
    if (gate_up_10.ok && down_10.ok) {
        // One decode token = gate + up (K=2560, N=1280) + down (K=640, N=2560)
        // per MoE layer, 48 layers.
        const double per_layer_ms = (2.0 * gate_up_10.per_call_us + down_10.per_call_us) / 1e3;
        std::printf("per token, S=10: %d layers x (2 x gate/up + down) = %.1f ms\n", kLayers,
                    per_layer_ms * kLayers);
    }
    if (gate_up_1.ok && down_10.ok) {
        const double down_1_us = down_10.per_call_us / 10.0;  // members scale with S
        const double per_layer_ms = (2.0 * gate_up_1.per_call_us + down_1_us) / 1e3;
        std::printf("per token, S=1:  %d layers x (2 x gate/up + down) = %.1f ms\n", kLayers,
                    per_layer_ms * kLayers);
    }
    if (failures == 0) std::printf("latency estimate: all checks passed\n");
    else std::printf("latency estimate: %d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
