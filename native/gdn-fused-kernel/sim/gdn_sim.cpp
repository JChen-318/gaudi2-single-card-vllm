// TPC simulator harness for flashinfer_gaudi_gdn_mtp_prepared_f32_gaudi2 (CPU only).
// Usage: gdn_sim [batch=1] [heads_limit=48]  -> prints cycles and max errors vs a CPU reference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "gdn_mtp_prepared_f32_gaudi2.hpp"
#include "tpc_test_core_api.h"

using namespace tpc_tests;

#define CFG(es, mask, last) ((((es)&0x3)) | ((((mask)) & ((1 << (DEF_NUM_DIMS_IN_IRF)) - 1)) << 8) | ((((last)&0x7)) << 16))

static TensorDesc2 Desc(float* data, const std::vector<uint32_t>& sizes)
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
    d.configuration = CFG(2, mask, sizes.size() - 1);
    return d;
}

static void Geo(tpc_lib_api::Tensor& t, const std::vector<uint64_t>& sizes)
{
    t.geometry.dims = sizes.size();
    t.geometry.dataType = tpc_lib_api::DATA_F32;
    for (size_t i = 0; i < sizes.size(); ++i) t.geometry.maxSizes[i] = t.geometry.minSizes[i] = sizes[i];
}

int main(int argc, char** argv)
{
    const int B = argc > 1 ? atoi(argv[1]) : 1;
    const int T = 8, H = 48, D = 128;
    std::mt19937 rng(1234);
    std::normal_distribution<float> n01(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    std::vector<float> state((size_t)B * H * D * D), packed((size_t)B * T * 10240), decay((size_t)B * T * H),
        beta((size_t)B * T * H);
    for (auto& x : state) x = 0.05f * n01(rng);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            float* p = &packed[((size_t)b * T + t) * 10240];
            for (int i = 0; i < 10240; ++i) p[i] = 0.1f * n01(rng);
            // q and k rows are L2-normalized per key head (q also scaled), like the caller prepares them.
            for (int part = 0; part < 2; ++part)
                for (int kh = 0; kh < 16; ++kh) {
                    float* v = p + part * 2048 + kh * 128;
                    double s = 0;
                    for (int i = 0; i < 128; ++i) s += (double)v[i] * v[i];
                    const float scale = (float)(1.0 / std::sqrt(s)) * (part == 0 ? 1.0f / std::sqrt(128.f) : 1.0f);
                    for (int i = 0; i < 128; ++i) v[i] *= scale;
                }
        }
    for (auto& x : decay) x = std::exp(-0.1f * u01(rng));
    for (auto& x : beta) x = 1.f / (1.f + std::exp(-n01(rng)));

    std::vector<float> out((size_t)B * T * H * D, 0.f), ckpt((size_t)B * T * H * D * D, 0.f);

    tpc_lib_api::HabanaKernelParams in;
    tpc_lib_api::HabanaKernelInstantiation inst;
    memset(&in, 0, sizeof(in));
    memset(&inst, 0, sizeof(inst));
    std::vector<tpc_lib_api::Tensor> inT(4), outT(2);
    memset(inT.data(), 0, sizeof(tpc_lib_api::Tensor) * 4);
    memset(outT.data(), 0, sizeof(tpc_lib_api::Tensor) * 2);
    in.inputTensors = inT.data();
    in.outputTensors = outT.data();
    in.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    in.inputTensorNr = 4;
    in.outputTensorNr = 2;
    Geo(inT[0], {128, 128, (uint64_t)H, (uint64_t)B});
    Geo(inT[1], {10240, 8, (uint64_t)B});
    Geo(inT[2], {48, 8, (uint64_t)B});
    Geo(inT[3], {48, 8, (uint64_t)B});
    Geo(outT[0], {128, 48, 8, (uint64_t)B});
    Geo(outT[1], {128, 128, 48, 8, (uint64_t)B});
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

    GdnMtpPreparedF32Gaudi2 glue;
    tpc_lib_api::GlueCodeReturn r = glue.GetGcDefinitions(&in, &inst);
    if (r != tpc_lib_api::GLUE_SUCCESS) {
        printf("glue failed %d\n", (int)r);
        return 1;
    }
    std::vector<TensorDesc2> descs = {
        Desc(state.data(), {128, 128, (uint32_t)H, (uint32_t)B}),
        Desc(packed.data(), {10240, 8, (uint32_t)B}),
        Desc(decay.data(), {48, 8, (uint32_t)B}),
        Desc(beta.data(), {48, 8, (uint32_t)B}),
        Desc(out.data(), {128, 48, 8, (uint32_t)B}),
        Desc(ckpt.data(), {128, 128, 48, 8, (uint32_t)B}),
    };
    VPEStats stats;
    const unsigned cycles = RunSimulation(in, inst, descs, stats);

    // CPU reference in double.
    double maxOut = 0, maxCk = 0, refOutMax = 0;
    std::vector<double> S(D * D);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            const int kh = h / 3;
            for (int i = 0; i < D * D; ++i) S[i] = state[(((size_t)b * H + h) * D) * D + i];
            for (int t = 0; t < T; ++t) {
                const float* p = &packed[((size_t)b * T + t) * 10240];
                const float* q = p + kh * 128;
                const float* k = p + 2048 + kh * 128;
                const float* v = p + 4096 + h * 128;
                const double g = decay[((size_t)b * T + t) * H + h], be = beta[((size_t)b * T + t) * H + h];
                for (int row = 0; row < D; ++row) {
                    double* s = &S[row * D];
                    double proj = 0;
                    for (int i = 0; i < D; ++i) {
                        s[i] *= g;
                        proj += s[i] * k[i];
                    }
                    const double delta = (v[row] - proj) * be;
                    double o = 0;
                    for (int i = 0; i < D; ++i) {
                        s[i] += delta * k[i];
                        o += s[i] * q[i];
                    }
                    const double got = out[(((size_t)b * T + t) * H + h) * D + row];
                    maxOut = std::fmax(maxOut, std::fabs(got - o));
                    refOutMax = std::fmax(refOutMax, std::fabs(o));
                    const float* c = &ckpt[((((size_t)b * T + t) * H + h) * D + row) * D];
                    for (int i = 0; i < D; ++i) maxCk = std::fmax(maxCk, std::fabs(c[i] - s[i]));
                }
            }
        }
    printf("batch=%d cycles=%u max_abs_out=%.3e (ref max %.3e) max_abs_ckpt=%.3e\n", B, cycles, maxOut, refOutMax,
           maxCk);
    return (maxOut < 1e-4 && maxCk < 1e-5) ? 0 : 2;
}
