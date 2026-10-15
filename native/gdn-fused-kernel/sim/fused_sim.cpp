// TPC simulator harness for flashinfer_gaudi_gdn_mtp_fused_f32_gaudi2 against the prepared-route math.
// Usage: fused_sim [batch=1]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "gdn_mtp_fused_f32_gaudi2.hpp"
#include "tpc_test_core_api.h"

using namespace tpc_tests;
#define CFG(es, mask, last) ((((es)&0x3)) | ((((mask)) & ((1 << (DEF_NUM_DIMS_IN_IRF)) - 1)) << 8) | ((((last)&0x7)) << 16))

static uint16_t ToBf16(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, 4);
    bits += 0x7FFF + ((bits >> 16) & 1);
    return (uint16_t)(bits >> 16);
}
static float FromBf16(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float x;
    memcpy(&x, &bits, 4);
    return x;
}

static TensorDesc2 Desc(void* data, const std::vector<uint32_t>& sizes, int elementSizeType)
{
    TensorDesc2 d = {};
    d.baseAddrUnion.baseAddr = (uint64_t)data;
    uint32_t stride = 1, mask = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        d.dimDescriptors[i].size = sizes[i];
        d.dimDescriptors[i].stride = stride;
        stride *= sizes[i];
        mask |= 1u << i;
    }
    d.configuration = CFG(elementSizeType, mask, sizes.size() - 1);
    return d;
}

static void Geo(tpc_lib_api::Tensor& t, tpc_lib_api::TensorDataType type, const std::vector<uint64_t>& sizes)
{
    t.geometry.dims = sizes.size();
    t.geometry.dataType = type;
    for (size_t i = 0; i < sizes.size(); ++i) t.geometry.maxSizes[i] = t.geometry.minSizes[i] = sizes[i];
}

int main(int argc, char** argv)
{
    const int B = argc > 1 ? atoi(argv[1]) : 1;
    const int T = 8, H = 48, D = 128, SLOTS = B * T + 2;
    std::mt19937 rng(4321);
    std::normal_distribution<float> n01(0.f, 1.f);
    std::uniform_int_distribution<int> acc(1, 8);

    std::vector<float> pool((size_t)SLOTS * H * D * D), log_decay((size_t)B * T * H);
    std::vector<uint16_t> packed((size_t)B * T * 10240), beta((size_t)B * T * H);
    std::vector<int32_t> indices((size_t)B * T), accepted(B);
    for (auto& x : pool) x = 0.05f * n01(rng);
    for (auto& x : packed) x = ToBf16(0.3f * n01(rng));
    for (auto& x : log_decay) x = -0.1f * std::fabs(n01(rng));
    for (auto& x : beta) x = ToBf16(1.f / (1.f + std::exp(-n01(rng))));
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) indices[b * T + t] = 1 + b * T + t;
        accepted[b] = acc(rng);
    }
    if (B >= 2) accepted[1] = 0;  // clamps to 1

    std::vector<float> out((size_t)B * T * H * D, 0.f), ckpt((size_t)B * T * H * D * D, 0.f);
    tpc_lib_api::HabanaKernelParams in;
    tpc_lib_api::HabanaKernelInstantiation inst;
    memset(&in, 0, sizeof(in));
    memset(&inst, 0, sizeof(inst));
    std::vector<tpc_lib_api::Tensor> inT(6), outT(2);
    memset(inT.data(), 0, sizeof(tpc_lib_api::Tensor) * 6);
    memset(outT.data(), 0, sizeof(tpc_lib_api::Tensor) * 2);
    in.inputTensors = inT.data();
    in.outputTensors = outT.data();
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    in.inputTensorNr = 6;
    in.outputTensorNr = 2;
    Geo(inT[0], tpc_lib_api::DATA_F32, {128, 128, (uint64_t)H, (uint64_t)SLOTS});
    Geo(inT[1], tpc_lib_api::DATA_BF16, {10240, 8, (uint64_t)B});
    Geo(inT[2], tpc_lib_api::DATA_F32, {48, 8, (uint64_t)B});
    Geo(inT[3], tpc_lib_api::DATA_BF16, {48, 8, (uint64_t)B});
    Geo(inT[4], tpc_lib_api::DATA_I32, {8, (uint64_t)B});
    Geo(inT[5], tpc_lib_api::DATA_I32, {(uint64_t)B});
    Geo(outT[0], tpc_lib_api::DATA_F32, {128, 48, 8, (uint64_t)B});
    Geo(outT[1], tpc_lib_api::DATA_F32, {128, 128, 48, 8, (uint64_t)B});
    std::vector<tpc_lib_api::TensorAccessPattern> inAP(16), outAP(16);
    std::vector<tpc_lib_api::AuxTensor> aux(16);
    memset(inAP.data(), 0, sizeof(inAP[0]) * 16);
    memset(outAP.data(), 0, sizeof(outAP[0]) * 16);
    memset(aux.data(), 0, sizeof(aux[0]) * 16);
    inst.inputTensorAccessPattern = inAP.data();
    inst.outputTensorAccessPattern = outAP.data();
    inst.auxiliaryTensors = aux.data();
    std::vector<char> elf(1 << 20);
    inst.kernel.kernelElf = elf.data();
    inst.kernel.elfSize = elf.size();
    GdnMtpFusedF32Gaudi2 glue;
    const tpc_lib_api::GlueCodeReturn r = glue.GetGcDefinitions(&in, &inst);
    if (r != tpc_lib_api::GLUE_SUCCESS) {
        printf("glue failed %d\n", (int)r);
        return 1;
    }
    std::vector<TensorDesc2> descs = {
        Desc(pool.data(), {128, 128, (uint32_t)H, (uint32_t)SLOTS}, 2),
        Desc(packed.data(), {10240, 8, (uint32_t)B}, 1),
        Desc(log_decay.data(), {48, 8, (uint32_t)B}, 2),
        Desc(beta.data(), {48, 8, (uint32_t)B}, 1),
        Desc(indices.data(), {8, (uint32_t)B}, 2),
        Desc(accepted.data(), {(uint32_t)B}, 2),
        Desc(out.data(), {128, 48, 8, (uint32_t)B}, 2),
        Desc(ckpt.data(), {128, 128, 48, 8, (uint32_t)B}, 2),
    };
    VPEStats stats;
    const unsigned cycles = RunSimulation(in, inst, descs, stats);

    double maxOut = 0, maxCk = 0, refOut = 0, refCk = 0;
    std::vector<double> S(D * D), qn(D), kn(D);
    for (int b = 0; b < B; ++b) {
        int a = accepted[b] < 1 ? 1 : (accepted[b] > 8 ? 8 : accepted[b]);
        const int slot = indices[b * T + a - 1];
        for (int h = 0; h < H; ++h) {
            const int kh = h / 3;
            for (int i = 0; i < D * D; ++i) S[i] = pool[(((size_t)slot * H + h) * D) * D + i];
            for (int t = 0; t < T; ++t) {
                const uint16_t* p = &packed[((size_t)b * T + t) * 10240];
                double qq = 0, kk = 0;
                for (int i = 0; i < D; ++i) {
                    const double q = FromBf16(p[kh * 128 + i]), k = FromBf16(p[2048 + kh * 128 + i]);
                    qq += q * q;
                    kk += k * k;
                }
                for (int i = 0; i < D; ++i) {
                    qn[i] = FromBf16(p[kh * 128 + i]) / std::sqrt(qq + 1e-6) / std::sqrt(128.0);
                    kn[i] = FromBf16(p[2048 + kh * 128 + i]) / std::sqrt(kk + 1e-6);
                }
                const double g = std::exp((double)log_decay[((size_t)b * T + t) * H + h]);
                const double be = FromBf16(beta[((size_t)b * T + t) * H + h]);
                for (int row = 0; row < D; ++row) {
                    double* s = &S[row * D];
                    double proj = 0;
                    for (int i = 0; i < D; ++i) {
                        s[i] *= g;
                        proj += s[i] * kn[i];
                    }
                    const double delta = (FromBf16(p[4096 + h * 128 + row]) - proj) * be;
                    double o = 0;
                    for (int i = 0; i < D; ++i) {
                        s[i] += delta * kn[i];
                        o += s[i] * qn[i];
                    }
                    const double got = out[(((size_t)b * T + t) * H + h) * D + row];
                    maxOut = std::fmax(maxOut, std::fabs(got - o));
                    refOut = std::fmax(refOut, std::fabs(o));
                    const float* c = &ckpt[((((size_t)b * T + t) * H + h) * D + row) * D];
                    for (int i = 0; i < D; ++i) {
                        maxCk = std::fmax(maxCk, std::fabs(c[i] - s[i]));
                        refCk = std::fmax(refCk, std::fabs(s[i]));
                    }
                }
            }
        }
    }
    printf("batch=%d cycles=%u max_abs_out=%.3e (ref max %.3e) max_abs_ckpt=%.3e (ref max %.3e)\n", B, cycles, maxOut,
           refOut, maxCk, refCk);
    return (maxOut < 2e-4 * refOut + 1e-6 && maxCk < 2e-5 * refCk + 1e-7) ? 0 : 2;
}
