// SPDX-License-Identifier: Apache-2.0
// TPC simulator harness for the Qwen4Exp NVFP4 MoE kernels (host only, no
// device). Runs the real TPC ELFs through libtpcsim_shared and compares the
// results against a scalar reference computed here:
//
//   * decode GEMV, one-hot activations: every output element is a single
//     decoded weight times the expert's scale2, so the comparison is exact
//     (bit for bit against a round-to-nearest BF16 of the same value). This is
//     what proves the kernel's nibble x E4M3-scale decode, its F32 accumulate
//     and its scale2 epilogue.
//   * decode GEMV, dense activations: multi-term F32 accumulation vs a
//     double-precision dot product (tolerance, the output is BF16).
//   * bucketing: count -> prefix -> scatter against the same integer
//     arithmetic computed in a loop, including ids < 0 (skip) and a tail block
//     that is not a full 64 slots.
//
// Shapes are small on purpose (K = 128, N = 4, E = 64, S = 70) so the whole
// run stays in the millisecond range.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "qwen4exp_nvfp4_moe.h"
#include "qwen4exp_nvfp4_moe_count_i32_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_prefix_i32_gaudi2.hpp"
#include "qwen4exp_nvfp4_moe_scatter_i32_gaudi2.hpp"
#include "tpc_test_core_api.h"

using namespace tpc_lib_api;
using namespace tpc_tests;

namespace {

constexpr int kBlock = QWEN4EXP_MOE_BLOCK_SLOTS;
constexpr int kRows = QWEN4EXP_MOE_ROWS_PER_MEMBER;

int failures = 0;
void check(bool ok, const char* what, const char* detail = "") {
    std::printf("  %-58s %s%s%s\n", what, ok ? "ok" : "FAILED",
                *detail ? " " : "", detail);
    if (!ok) ++failures;
}

// ------------------------------------------------------------------ helpers

uint16_t to_bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

double bf16_to_double(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

// E4M3 (float8_e4m3fn) -> double. 0x7F/0xFF are NaN and never appear in the data.
double e4m3_to_double(uint8_t bits) {
    const double sign = (bits & 0x80) ? -1.0 : 1.0;
    const int exponent = (bits >> 3) & 0xF;
    const int mantissa = bits & 0x7;
    if (exponent == 0xF && mantissa == 0x7) return std::nan("");
    const double significand = exponent == 0 ? mantissa / 8.0 : 1.0 + mantissa / 8.0;
    const double scale = exponent == 0 ? std::ldexp(1.0, -6) : std::ldexp(1.0, exponent - 7);
    return sign * significand * scale;
}

const double kE2M1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

// Decoded weight of (expert, row, k) from the same packed bytes and scales the
// kernel reads. packed is [E][N][K/2], scales [E][N][K/16], low nibble = even k.
double weight(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
              int experts, int rows, int k, int expert, int row, int kk) {
    const size_t packed_stride = static_cast<size_t>(rows) * (k / 2);
    const size_t scale_stride = static_cast<size_t>(rows) * (k / 16);
    const uint8_t byte = packed[static_cast<size_t>(expert) * packed_stride +
                                static_cast<size_t>(row) * (k / 2) + kk / 2];
    const uint8_t nibble = (kk % 2) ? (byte >> 4) & 0xF : byte & 0xF;
    const uint8_t scale = scales[static_cast<size_t>(expert) * scale_stride +
                                 static_cast<size_t>(row) * (k / 16) + kk / 16];
    const double sign = (nibble & 0x8) ? -1.0 : 1.0;
    return sign * kE2M1[nibble & 0x7] * e4m3_to_double(scale);
}

TensorDesc2 describe(const void* data, std::initializer_list<uint32_t> sizes, int element_bits) {
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
    // element size type: 0 = 8 bit, 1 = 16 bit, 2 = 32 bit
    const int element_type = element_bits == 8 ? 0 : element_bits == 16 ? 1 : 2;
    desc.configuration = static_cast<uint32_t>(element_type) |
                         (mask << 8) |
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

struct Fixture {
    HabanaKernelParams params{};
    HabanaKernelInstantiation inst{};
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    std::vector<TensorAccessPattern> in_ap;
    std::vector<TensorAccessPattern> out_ap;
    std::vector<char> elf = std::vector<char>(1 << 20);

    Fixture(const char* guid, unsigned n_in, unsigned n_out) {
        inputs.resize(n_in);
        outputs.resize(n_out);
        in_ap.resize(n_in);
        out_ap.resize(n_out);
        std::memset(inputs.data(), 0, sizeof(Tensor) * n_in);
        std::memset(outputs.data(), 0, sizeof(Tensor) * n_out);
        std::memset(in_ap.data(), 0, sizeof(TensorAccessPattern) * n_in);
        std::memset(out_ap.data(), 0, sizeof(TensorAccessPattern) * n_out);
        params.inputTensors = inputs.data();
        params.outputTensors = outputs.data();
        params.inputTensorNr = n_in;
        params.outputTensorNr = n_out;
        params.deviceId = DEVICE_ID_GAUDI2;
        std::strncpy(params.guid.name, guid, MAX_NODE_NAME - 1);
        inst.kernel.kernelElf = elf.data();
        inst.kernel.elfSize = static_cast<uint32_t>(elf.size());
        inst.inputTensorAccessPattern = in_ap.data();
        inst.outputTensorAccessPattern = out_ap.data();
    }
};

// ---------------------------------------------------------- decode GEMV runs

struct GemvCase {
    int k;
    int rows;
    int experts;
    int slots;
};

// Instantiate the GEMV glue for one shape, run the ELF in the simulator and
// leave the kernel's BF16 output in `out`. Returns the simulator's cycle count.
unsigned simulate_gemv(int k, int rows, int experts, int slots,
                       const std::vector<int32_t>& ids, const std::vector<uint8_t>& packed,
                       const std::vector<uint8_t>& scales, const std::vector<uint16_t>& x,
                       const std::vector<float>& scale2, std::vector<uint16_t>& out) {
    Fixture fixture("custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2", 5, 1);
    geometry(fixture.inputs[0], DATA_I32, {static_cast<uint64_t>(slots)});
    geometry(fixture.inputs[1], DATA_U8,
             {static_cast<uint64_t>(k / 2), static_cast<uint64_t>(rows),
              static_cast<uint64_t>(experts)});
    geometry(fixture.inputs[2], DATA_U8,
             {static_cast<uint64_t>(k / 16), static_cast<uint64_t>(rows),
              static_cast<uint64_t>(experts)});
    geometry(fixture.inputs[3], DATA_BF16,
             {static_cast<uint64_t>(k), static_cast<uint64_t>(slots)});
    geometry(fixture.inputs[4], DATA_F32, {static_cast<uint64_t>(experts)});
    geometry(fixture.outputs[0], DATA_BF16,
             {static_cast<uint64_t>(rows), static_cast<uint64_t>(slots)});
    Qwen4ExpNvfp4MoeGemvBF16Gaudi2 glue;
    const GlueCodeReturn status = glue.GetGcDefinitions(&fixture.params, &fixture.inst);
    if (status != GLUE_SUCCESS) {
        check(false, "instantiate", "glue failed");
        return 0;
    }
    std::vector<TensorDesc2> descs = {
        describe(ids.data(), {static_cast<uint32_t>(slots)}, 32),
        describe(packed.data(), {static_cast<uint32_t>(k / 2), static_cast<uint32_t>(rows),
                                 static_cast<uint32_t>(experts)}, 8),
        describe(scales.data(), {static_cast<uint32_t>(k / 16), static_cast<uint32_t>(rows),
                                 static_cast<uint32_t>(experts)}, 8),
        describe(x.data(), {static_cast<uint32_t>(k), static_cast<uint32_t>(slots)}, 16),
        describe(scale2.data(), {static_cast<uint32_t>(experts)}, 32),
        describe(out.data(), {static_cast<uint32_t>(rows), static_cast<uint32_t>(slots)}, 16),
    };
    VPEStats stats;
    return RunSimulation(fixture.params, fixture.inst, descs, stats);
}

void run_gemv(const GemvCase& shape, bool one_hot) {
    const int k = shape.k, rows = shape.rows, experts = shape.experts, slots = shape.slots;
    std::printf("gemv K=%d N=%d E=%d S=%d (%s activations):\n", k, rows, experts, slots,
                one_hot ? "one-hot" : "dense");
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> expert_dist(0, experts - 1);
    std::normal_distribution<float> normal(0.0f, 0.3f);

    std::vector<uint8_t> packed(static_cast<size_t>(experts) * rows * (k / 2));
    std::vector<uint8_t> scales(static_cast<size_t>(experts) * rows * (k / 16));
    for (auto& value : packed) value = static_cast<uint8_t>(byte_dist(rng));
    for (auto& value : scales) {
        value = static_cast<uint8_t>(byte_dist(rng));
        if (value == 0x7F) value = 0x3F;  // no NaN scales
        if (value == 0xFF) value = 0xBF;
    }
    std::vector<int32_t> ids(slots);
    std::vector<float> scale2(experts);
    std::vector<uint16_t> x(static_cast<size_t>(slots) * k);
    for (int s = 0; s < slots; ++s) ids[s] = expert_dist(rng);
    for (auto& value : scale2) value = 0.25f + 0.25f * (rng() % 8);
    if (one_hot) {
        std::fill(x.begin(), x.end(), 0);
        for (int s = 0; s < slots; ++s) x[static_cast<size_t>(s) * k + s] = to_bf16_bits(1.0f);
    } else {
        for (auto& value : x) value = to_bf16_bits(normal(rng));
    }
    std::vector<uint16_t> out(static_cast<size_t>(slots) * rows, 0xFFFF);

    const unsigned cycles =
        simulate_gemv(k, rows, experts, slots, ids, packed, scales, x, scale2, out);

    double worst = 0;
    double reference_scale = 0;
    int exact_mismatches = 0;
    for (int s = 0; s < slots; ++s) {
        const int expert = ids[s];
        for (int n = 0; n < rows; ++n) {
            double reference = 0;
            for (int kk = 0; kk < k; ++kk) {
                reference += weight(packed, scales, experts, rows, k, expert, n, kk) *
                             bf16_to_double(x[static_cast<size_t>(s) * k + kk]);
            }
            reference *= scale2[expert];
            const double got = bf16_to_double(out[static_cast<size_t>(s) * rows + n]);
            if (!(got == 0.0 && reference == 0.0))
                worst = std::fmax(worst, std::fabs(got - reference));
            reference_scale = std::fmax(reference_scale, std::fabs(reference));
            if (one_hot) {
                // single non-zero term: the sum is exact, so the only rounding is the
                // F32 scale2 multiply and the BF16 store.
                const double single = weight(packed, scales, experts, rows, k, expert, n, s) *
                                      scale2[expert];
                const uint16_t expected = to_bf16_bits(static_cast<float>(single));
                const double got_value = bf16_to_double(out[static_cast<size_t>(s) * rows + n]);
                // A single -0.0 weight plus 127 zero lanes sums to +0.0 in the kernel's F32
                // reduction (round-to-nearest), so signed zeros are the one allowed difference.
                const bool same = expected == out[static_cast<size_t>(s) * rows + n] ||
                                  (single == 0.0 && got_value == 0.0);
                if (!same) {
                    if (exact_mismatches < 8) {
                        std::printf("    s=%d n=%d e=%d expected=%.8g (0x%04x) got=%.8g (0x%04x)\n",
                                    s, n, expert, static_cast<float>(single), expected, got_value,
                                    out[static_cast<size_t>(s) * rows + n]);
                    }
                    ++exact_mismatches;
                }
            }
        }
    }
    char line[192];
    std::snprintf(line, sizeof(line), "cycles=%u max_abs=%.3e (ref max %.3e)", cycles, worst,
                  reference_scale);
    check(one_hot ? exact_mismatches == 0 : worst < 5e-3 * reference_scale + 1e-6,
          one_hot ? "outputs bit-exact vs decoded weights" : "outputs match double reference",
          line);
    if (one_hot) {
        std::snprintf(line, sizeof(line), "exact mismatches=%d of %d", exact_mismatches,
                      slots * rows);
        check(exact_mismatches == 0, "one-hot decode check", line);
    }
}

// Power-of-two data: every product is a power of two in [2**-3, 2**1], every
// partial sum of at most K = 1280 of them is an integer multiple of 2**-3 below
// 2**12 (15 significant bits), and scale2 is 2**-2 or 2**0 - so the F32 lane
// accumulation the kernel performs is exact, whatever the order. The stored
// BF16 must therefore equal the round-to-nearest BF16 of the exact double sum
// bit for bit, with only a signed zero free (as in the one-hot case).
void run_gemv_exact() {
    const int k = 1280, rows = 4, experts = 64, slots = 4;
    std::printf("gemv K=%d N=%d E=%d S=%d (power-of-two, exact):\n", k, rows, experts, slots);
    std::mt19937 rng(2024);
    const uint8_t magnitudes[4] = {0, 0, 1, 2};  // E2M1 0, 0, 0.5, 1.0
    auto nibble = [&]() -> uint8_t {
        return static_cast<uint8_t>(magnitudes[rng() % 4] | ((rng() & 1) ? 0x8 : 0x0));
    };
    std::vector<uint8_t> packed(static_cast<size_t>(experts) * rows * (k / 2));
    for (auto& byte : packed) byte = static_cast<uint8_t>(nibble() | (nibble() << 4));
    // E4M3 0x30 is 0.5, 0x38 is 1.0: both exact, mantissa zero.
    std::vector<uint8_t> scales(static_cast<size_t>(experts) * rows * (k / 16));
    for (auto& byte : scales) byte = (rng() & 1) ? 0x30 : 0x38;
    std::vector<int32_t> ids(slots);
    for (int s = 0; s < slots; ++s) ids[s] = static_cast<int32_t>(rng() % experts);
    const float activations[8] = {0.0f, 0.25f, -0.25f, 0.5f, -0.5f, 1.0f, -1.0f, 0.0f};
    std::vector<uint16_t> x(static_cast<size_t>(slots) * k);
    for (auto& value : x) value = to_bf16_bits(activations[rng() % 8]);
    std::vector<float> scale2(experts);
    for (auto& value : scale2) value = (rng() & 1) ? 0.25f : 1.0f;
    std::vector<uint16_t> out(static_cast<size_t>(slots) * rows, 0xFFFF);

    const unsigned cycles =
        simulate_gemv(k, rows, experts, slots, ids, packed, scales, x, scale2, out);

    int mismatches = 0;
    double worst = 0;
    for (int s = 0; s < slots; ++s) {
        for (int n = 0; n < rows; ++n) {
            double reference = 0;
            for (int kk = 0; kk < k; ++kk) {
                reference += weight(packed, scales, experts, rows, k, ids[s], n, kk) *
                             bf16_to_double(x[static_cast<size_t>(s) * k + kk]);
            }
            reference *= static_cast<double>(scale2[ids[s]]);
            // Exact in F32 -> the only rounding is the BF16 store.
            const uint16_t expected = to_bf16_bits(static_cast<float>(reference));
            const uint16_t got = out[static_cast<size_t>(s) * rows + n];
            const bool same = expected == got ||
                              (reference == 0.0 && bf16_to_double(got) == 0.0);
            if (!same) {
                if (mismatches < 8) {
                    std::printf("    s=%d n=%d exp=%.9g (0x%04x) got=%.9g (0x%04x)\n", s, n,
                                reference, expected, bf16_to_double(got), got);
                }
                ++mismatches;
            }
            worst = std::fmax(worst, std::fabs(bf16_to_double(got) - reference));
        }
    }
    char line[192];
    std::snprintf(line, sizeof(line), "cycles=%u max_abs=%.3e exact mismatches=%d of %d", cycles,
                  worst, mismatches, slots * rows);
    check(mismatches == 0, "outputs bit-exact on power-of-two data", line);
}

// --------------------------------------------------------------- bucketing

void run_bucketing(int slots) {
    const int experts = 64;
    const int cap = slots;
    const int blocks = (slots + kBlock - 1) / kBlock;
    std::printf("bucketing E=%d S=%d blocks=%d cap=%d:\n", experts, slots, blocks, cap);

    std::mt19937 rng(99);
    std::uniform_int_distribution<int> expert_dist(0, experts - 1);
    std::vector<int32_t> ids(slots);
    for (int s = 0; s < slots; ++s) {
        ids[s] = expert_dist(rng);
        if (s % 11 == 5) ids[s] = -1;      // skip marker (fused shared slot / padding)
        if (s == 17) ids[s] = experts + 5;  // out of range: dropped, must not corrupt the heap
    }

    // ---- count -----------------------------------------------------------
    std::vector<int32_t> counts(static_cast<size_t>(experts) * blocks, -1);
    {
        Fixture fixture("custom_qwen4exp_nvfp4_moe_count_i32_gaudi2", 1, 1);
        struct {
            int32_t experts;
        } params{experts};
        fixture.params.nodeParams.nodeParams = &params;
        fixture.params.nodeParams.nodeParamsSize = sizeof(params);
        geometry(fixture.inputs[0], DATA_I32, {static_cast<uint64_t>(slots)});
        geometry(fixture.outputs[0], DATA_I32,
                 {static_cast<uint64_t>(experts), static_cast<uint64_t>(blocks)});
        Qwen4ExpNvfp4MoeCountI32Gaudi2 glue;
        const GlueCodeReturn status = glue.GetGcDefinitions(&fixture.params, &fixture.inst);
        if (status != GLUE_SUCCESS) {
            check(false, "count instantiate", "glue failed");
            return;
        }
        std::vector<TensorDesc2> descs = {
            describe(ids.data(), {static_cast<uint32_t>(slots)}, 32),
            describe(counts.data(), {static_cast<uint32_t>(experts),
                                     static_cast<uint32_t>(blocks)}, 32),
        };
        VPEStats stats;
        const unsigned cycles = RunSimulation(fixture.params, fixture.inst, descs, stats);
        std::vector<int32_t> reference(counts.size(), 0);
        for (int s = 0; s < slots; ++s)
            if (ids[s] >= 0 && ids[s] < experts)
                ++reference[static_cast<size_t>(s / kBlock) * experts + ids[s]];
        int mismatches = 0;
        for (size_t i = 0; i < counts.size(); ++i)
            if (counts[i] != reference[i]) {
                if (mismatches < 8)
                    std::printf("    b=%zu e=%zu got=%d ref=%d\n", i / experts, i % experts,
                                counts[i], reference[i]);
                ++mismatches;
            }
        char line[128];
        std::snprintf(line, sizeof(line), "cycles=%u values=%zu", cycles, counts.size());
        check(mismatches == 0, "counts exact", line);
        if (mismatches) std::printf("    first mismatches: %d\n", mismatches);
    }

    // ---- prefix ----------------------------------------------------------
    std::vector<int32_t> offsets(static_cast<size_t>(experts) * blocks, -1);
    std::vector<int32_t> totals(experts, -1);
    {
        Fixture fixture("custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2", 1, 2);
        geometry(fixture.inputs[0], DATA_I32,
                 {static_cast<uint64_t>(experts), static_cast<uint64_t>(blocks)});
        geometry(fixture.outputs[0], DATA_I32,
                 {static_cast<uint64_t>(experts), static_cast<uint64_t>(blocks)});
        geometry(fixture.outputs[1], DATA_I32, {static_cast<uint64_t>(experts)});
        Qwen4ExpNvfp4MoePrefixI32Gaudi2 glue;
        const GlueCodeReturn status = glue.GetGcDefinitions(&fixture.params, &fixture.inst);
        if (status != GLUE_SUCCESS) {
            check(false, "prefix instantiate", "glue failed");
            return;
        }
        std::vector<TensorDesc2> descs = {
            describe(counts.data(), {static_cast<uint32_t>(experts),
                                     static_cast<uint32_t>(blocks)}, 32),
            describe(offsets.data(), {static_cast<uint32_t>(experts),
                                      static_cast<uint32_t>(blocks)}, 32),
            describe(totals.data(), {static_cast<uint32_t>(experts)}, 32),
        };
        VPEStats stats;
        const unsigned cycles = RunSimulation(fixture.params, fixture.inst, descs, stats);
        int mismatches = 0;
        for (int e = 0; e < experts; ++e) {
            int running = 0;
            for (int b = 0; b < blocks; ++b) {
                if (offsets[static_cast<size_t>(b) * experts + e] != running) ++mismatches;
                running += counts[static_cast<size_t>(b) * experts + e];
            }
            if (totals[e] != running) ++mismatches;
        }
        char line[128];
        std::snprintf(line, sizeof(line), "cycles=%u", cycles);
        check(mismatches == 0, "offsets/totals exact", line);
    }

    // ---- scatter ---------------------------------------------------------
    std::vector<int32_t> bucket(static_cast<size_t>(experts) * cap, -7);
    std::vector<int32_t> place(slots, -7);
    {
        Fixture fixture("custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2", 2, 2);
        struct {
            int32_t experts;
            int32_t slots;
            int32_t cap;
        } params{experts, slots, cap};
        fixture.params.nodeParams.nodeParams = &params;
        fixture.params.nodeParams.nodeParamsSize = sizeof(params);
        geometry(fixture.inputs[0], DATA_I32, {static_cast<uint64_t>(slots)});
        geometry(fixture.inputs[1], DATA_I32,
                 {static_cast<uint64_t>(experts), static_cast<uint64_t>(blocks)});
        geometry(fixture.outputs[0], DATA_I32,
                 {static_cast<uint64_t>(experts) * static_cast<uint64_t>(cap)});
        geometry(fixture.outputs[1], DATA_I32, {static_cast<uint64_t>(slots)});
        Qwen4ExpNvfp4MoeScatterI32Gaudi2 glue;
        const GlueCodeReturn status = glue.GetGcDefinitions(&fixture.params, &fixture.inst);
        if (status != GLUE_SUCCESS) {
            check(false, "scatter instantiate", "glue failed");
            return;
        }
        std::vector<TensorDesc2> descs = {
            describe(ids.data(), {static_cast<uint32_t>(slots)}, 32),
            describe(offsets.data(), {static_cast<uint32_t>(experts),
                                      static_cast<uint32_t>(blocks)}, 32),
            describe(bucket.data(), {static_cast<uint32_t>(experts * cap)}, 32),
            describe(place.data(), {static_cast<uint32_t>(slots)}, 32),
        };
        VPEStats stats;
        const unsigned cycles = RunSimulation(fixture.params, fixture.inst, descs, stats);

        std::vector<int32_t> bucket_ref(bucket.size(), -7);
        std::vector<int32_t> place_ref(slots, -1);
        std::vector<int> earlier(experts, 0);
        for (int b = 0; b < blocks; ++b) {
            std::fill(earlier.begin(), earlier.end(), 0);
            for (int s = b * kBlock; s < (b + 1) * kBlock && s < slots; ++s) {
                const int expert = ids[s];
                if (expert < 0 || expert >= experts) continue;
                const int position = offsets[static_cast<size_t>(b) * experts + expert] +
                                     earlier[expert]++;
                const int at = expert * cap + position;
                bucket_ref[at] = s;
                place_ref[s] = at;
            }
        }
        int bucket_mismatch = 0;
        for (int e = 0; e < experts; ++e)
            for (int m = 0; m < totals[e]; ++m)
                if (bucket[static_cast<size_t>(e) * cap + m] !=
                    bucket_ref[static_cast<size_t>(e) * cap + m])
                    ++bucket_mismatch;
        int place_mismatch = 0;
        for (int s = 0; s < slots; ++s)
            if (place[s] != place_ref[s]) ++place_mismatch;
        char line[160];
        std::snprintf(line, sizeof(line), "cycles=%u bucket=%d place=%d",
                      cycles, bucket_mismatch, place_mismatch);
        check(bucket_mismatch == 0 && place_mismatch == 0, "bucket rows and place exact", line);
        // every routed slot is reachable exactly once through place -> bucket
        int round_trip = 0;
        for (int s = 0; s < slots; ++s)
            if (place[s] >= 0 && bucket[place[s]] != s) ++round_trip;
        std::snprintf(line, sizeof(line), "broken round trips=%d", round_trip);
        check(round_trip == 0, "place is the inverse of bucket", line);
    }
}

}  // namespace

int main() {
    std::printf("TPC simulator: Qwen4Exp NVFP4 MoE kernels (no device)\n");
    run_gemv({128, 4, 64, 64}, /*one_hot=*/true);
    run_gemv({128, 4, 64, 2}, /*one_hot=*/false);
    run_gemv_exact();
    run_bucketing(70);
    run_bucketing(10);
    if (failures == 0) std::printf("simulator validation: all tests passed\n");
    else std::printf("simulator validation: %d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
