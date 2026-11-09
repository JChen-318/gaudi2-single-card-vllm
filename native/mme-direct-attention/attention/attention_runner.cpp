// SPDX-License-Identifier: MIT
// Independent, device-scheduled MME -> TPC -> MME -> TPC correctness prototype.
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#define HLTESTS_LIB_MODE
extern "C" {
#include <hlthunk_tests.h>
}
#undef fail
#include "gaudi2/gaudi2_tpc_descriptor.h"
#include "../desc128.h"

static uint16_t bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float fp(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static unsigned wreg(int fd, void *cb, unsigned size, unsigned addr, uint32_t value, bool eb = false) {
    hltests_pkt_info p{}; p.mb = MB_TRUE; p.eb = eb ? EB_TRUE : EB_FALSE;
    p.wreg32.reg_addr = (uint16_t)addr; p.wreg32.value = value;
    return hltests_add_wreg32_pkt(fd, cb, size, &p);
}
static unsigned wait(int fd, void *cb, unsigned size, unsigned queue, uint16_t sob, uint16_t mon, bool common, unsigned value = 1) {
    hltests_monitor_and_fence m{}; m.queue_id = (uint8_t)queue; m.cmdq_fence = common;
    m.sob_id = sob; m.mon_id = mon; m.sob_val = value; m.dec_fence = true;
    m.mon_payload = 1; m.mon_mode = SOB_EQUAL;
    return hltests_add_monitor_and_fence(fd, cb, size, &m);
}
// One MME master: 128 output rows, 128-column tiles. Input/output strides are
// element counts. Tensor and reduction fields match captured K=256/1024 jobs;
// column tiling retains the verified 128x128 geometry rather than slave lanes.
static unsigned gemm(int fd, void *cb, unsigned size, uint64_t a, uint64_t b, uint64_t c, uint16_t sob,
                     unsigned cols, unsigned reduction, unsigned leading = 0,
                     std::vector<uint32_t> *previous = nullptr) {
    uint32_t words[182]; memcpy(words, desc_words, sizeof(words));
    auto set = [&](unsigned off, uint32_t v) { words[(off - 8) / 4] = v; };
    if (!leading) leading = cols;
    auto tensor = [&](unsigned off, unsigned valid_cols, unsigned elements, unsigned loop0,
                      unsigned loop1, unsigned roi0, unsigned spatial) {
        set(off, valid_cols);
        for (unsigned i = 1; i < 5; ++i) set(off + i * 4, elements);
        set(off + 20, loop0); set(off + 24, loop1); set(off + 28, elements * 4);
        set(off + 32, elements); set(off + 36, elements);
        set(off + 40, roi0); set(off + 44, elements); set(off + 48, elements * 4); set(off + 52, elements);
        set(off + 56, spatial);
        for (unsigned i = 15; i < 18; ++i) set(off + i * 4, elements);
    };
    tensor(0x40, reduction, 128 * reduction, 0, 128 * reduction, reduction, 2 * reduction);
    tensor(0x98, cols, reduction * leading, 128, 0, 64, leading);
    tensor(0xf0, cols, 128 * leading, 128, 128 * leading, 128, 2 * leading);
    set(0x200, reduction); // aguIn[4][MASTER] row offset, used by A
    set(0x258, leading);   // aguOut[1][MASTER] row offset
    set(0x228, reduction - 1);
    set(0x280, cols / 128 - 1);
    set(0x10, (uint32_t)c); set(0x14, (uint32_t)(c >> 32));
    set(0x18, (uint32_t)a); set(0x1c, (uint32_t)(a >> 32));
    set(0x20, (uint32_t)b); set(0x24, (uint32_t)(b >> 32));
    set(0x14c, (uint32_t)(hltests_get_sob_base_addr(fd) + sob * 4));
    bool first_write = true;
    for (unsigned i = 0; i < 182; ++i) {
        if (previous && previous->size() == 182 && (*previous)[i] == words[i]) continue;
        size = wreg(fd, cb, size, 0xb008 + i * 4, words[i], first_write); first_write = false;
    }
    if (previous) previous->assign(words, words + 182);
    return wreg(fd, cb, size, 0xb004, 0x3ff, true);
}
struct Tensor { uint64_t address; unsigned cols, rows, type; };
static unsigned tpc(int fd, void *cb, unsigned size, uint64_t isa,
                    const std::vector<Tensor> &tensors, unsigned rows, uint16_t sob, unsigned first_row = 0,
                    unsigned valid_cols = 0, std::vector<uint32_t> *previous = nullptr) {
    gaudi2::Gaudi2TpcDesc desc{};
    for (unsigned i = 0; i < tensors.size(); ++i) {
        auto &t = desc.m_tensors[i]; auto b = tensors[i];
        t.base_addr_low._raw = (uint32_t)b.address; t.base_addr_high._raw = (uint32_t)(b.address >> 32);
        t.tensor_config.data_type = b.type; t.tensor_config.valid_dim_mask = 0x1f; t.tensor_config.last_dim = 1;
        t.dim_0_size.v = b.cols; t.dim_0_stride.v = 1;
        t.dim_1_size.v = b.rows; t.dim_1_stride.v = b.cols;
        t.dim_2_size.v = 1; t.dim_2_stride.v = b.cols * b.rows;
        t.dim_3_size.v = 1; t.dim_3_stride.v = b.cols * b.rows;
        t.dim_4_size.v = 1; t.dim_4_stride.v = b.cols * b.rows;
    }
    desc.m_so.message._raw = 0x80000001;
    desc.m_so.addr.v = (uint32_t)(hltests_get_sob_base_addr(fd) + sob * 4);
    auto &d = desc.m_desc;
    d.kernel_base_address_low._raw = (uint32_t)isa; d.kernel_base_address_high._raw = (uint32_t)(isa >> 32);
    d.tid_size_dim_0.v = 1; d.tid_size_dim_1.v = rows;
    d.tid_base_dim_1.v = first_row;
    d.srf[0].v = valid_cols;
    d.tid_size_dim_2.v = 1; d.tid_size_dim_3.v = 1; d.tid_size_dim_4.v = 1;
    d.kernel_config.aso_evict_l0 = 1; d.kernel_config.num_valid_srfs = 32;
    d.kernel_config.rd_rate_limit_rst_token = 8; d.kernel_config.wr_rate_limit_rst_token = 6;
    d.kernel_config.irf_32bit_compatibility = 1;
    auto *words = reinterpret_cast<const uint32_t *>(&desc);
    unsigned base = 0xb000 + offsetof(gaudi2::block_tpc, qm_tensor_0);
    bool first_write = true;
    for (unsigned i = 0; i < sizeof(desc) / 4; ++i) {
        if (previous && previous->size() == sizeof(desc) / 4 && (*previous)[i] == words[i]) continue;
        size = wreg(fd, cb, size, base + i * 4, words[i], first_write); first_write = false;
    }
    if (previous) previous->assign(words, words + sizeof(desc) / 4);
    size = wreg(fd, cb, size, 0xb000 + offsetof(gaudi2::block_tpc, tpc_cmd), 0xf, true);
    return wreg(fd, cb, size, 0xb000 + offsetof(gaudi2::block_tpc, tpc_execute), 1);
}
static uint64_t load_isa(int fd, const char *path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    if (code.empty() || code.size() > 0x10000) return 0;
    void *device = hltests_allocate_device_mem(fd, 0x10000, 0, NOT_CONTIGUOUS);
    void *host = hltests_allocate_host_mem(fd, 0x10000, NOT_HUGE_MAP);
    if (!device || !host) return 0;
    memset(host, 0, 0x10000); memcpy(host, code.data(), code.size());
    int rc = hltests_dma_transfer(fd, hltests_get_dma_down_qid(fd, STREAM0), EB_FALSE, MB_TRUE,
             hltests_get_device_va_for_host_ptr(fd, host), (uintptr_t)device, 0x10000, DMA_DIR_HOST_TO_DRAM);
    hltests_free_host_mem(fd, host);
    return rc ? 0 : (uintptr_t)device;
}

int main(int argc, char **argv) {
    setbuf(stdout, nullptr);
    if (argc < 3) { fprintf(stderr, "Usage: attention_runner exp_sum.bin normalize.bin [seed] [uniform|random] [keys] [tpcs] [repeats] [input.bin|-] [batch]\n"); return 2; }
    uint32_t rng = argc > 3 ? (uint32_t)strtoul(argv[3], nullptr, 0) : 17;
    bool uniform = argc > 4 && strcmp(argv[4], "uniform") == 0;
    unsigned keys = argc > 5 ? (unsigned)strtoul(argv[5], nullptr, 0) : 128;
    unsigned tpcs = argc > 6 ? (unsigned)strtoul(argv[6], nullptr, 0) : 1;
    unsigned repeats = argc > 7 ? (unsigned)strtoul(argv[7], nullptr, 0) : 1;
    bool batch = argc > 9 && strcmp(argv[9], "batch") == 0;
    unsigned iterations = batch ? repeats : 1;
    if (!keys || keys % 128 || keys > 37760 || !tpcs || tpcs > 24 || !repeats || repeats > 1000) return 2;
    if (hltests_init()) return 3;
    int fd = hltests_open("0000:08:00.0"); if (fd < 0) return 4;
    hlthunk_hw_ip_info hw{}; if (hlthunk_get_hw_ip_info(fd, &hw)) return 4;
    constexpr unsigned n = 128, matrix = n * n * 2;
    unsigned cbbytes = 16384 + iterations * 4096;
    unsigned wide = n * keys * 2;
    unsigned oq = 0, ok = matrix, ov = ok + wide, os = ov + wide;
    unsigned oo = os + wide, ob = oo + matrix, ol = ob + n * 2, og = ol + n * 4;
    unsigned total = og + 4096;
    if (total + 0x10000 > hw.sram_size) return 2;
    auto *host = (uint8_t *)hltests_allocate_host_mem(fd, total, NOT_HUGE_MAP);
    if (!host) return 4;
    auto *q = (uint16_t *)(host + oq), *kt = (uint16_t *)(host + ok), *v = (uint16_t *)(host + ov);
    auto *bounds = (uint16_t *)(host + ob);
    std::vector<uint16_t> key(keys * n), ref_p(n * keys), ref_o(n * n);
    std::vector<float> ref_l(n);
    std::vector<float> full_ref;
    unsigned valid_keys = keys;
    for (unsigned i = 0; i < n * n; ++i) {
        rng = rng * 1664525 + 1013904223; q[i] = bf(uniform ? 0 : ((int)(rng % 9) - 4) / 16.0f);
    }
    for (unsigned i = 0; i < keys * n; ++i) {
        rng = rng * 1664525 + 1013904223; key[i] = bf(((int)(rng % 9) - 4) / 16.0f);
        rng = rng * 1664525 + 1013904223; v[i] = bf(((int)(rng % 9) - 4) / 4.0f);
    }
    if (argc > 8 && strcmp(argv[8], "-") != 0) {
        std::ifstream input(argv[8], std::ios::binary);
        char magic[8]; uint32_t shape[4];
        input.read(magic, sizeof(magic)); input.read((char *)shape, sizeof(shape));
        if (!input || memcmp(magic, "G2ATTN1", 8) || shape[0] != keys || !shape[1] ||
            shape[1] > keys || shape[2] != n || shape[3] != n) return 2;
        valid_keys = shape[1]; full_ref.resize(n * n);
        input.read((char *)q, matrix); input.read((char *)key.data(), wide);
        input.read((char *)v, wide); input.read((char *)full_ref.data(), matrix * 2);
        if (!input) return 2;
        for (unsigned i = 0; i < n * n; ++i) if (!std::isfinite(fp(q[i])) || !std::isfinite(full_ref[i])) return 2;
        for (unsigned i = 0; i < keys * n; ++i) if (!std::isfinite(fp(key[i])) || !std::isfinite(fp(v[i]))) return 2;
    }
    float max_key_norm = 0;
    for (unsigned i = 0; i < keys; ++i) {
        float norm = 0;
        for (unsigned j = 0; j < n; ++j) { kt[j * keys + i] = key[i * n + j]; norm += fp(key[i * n + j]) * fp(key[i * n + j]); }
        max_key_norm = std::max(max_key_norm, sqrtf(norm));
    }
    for (unsigned i = 0; i < n; ++i) {
        float norm = 0;
        for (unsigned j = 0; j < n; ++j) norm += fp(q[i * n + j]) * fp(q[i * n + j]);
        bounds[i] = bf(sqrtf(norm) * max_key_norm);
        if (!std::isfinite(fp(bounds[i])) || fp(bounds[i]) > 80) {
            printf("UNSUPPORTED bound=%g row=%u: use stable-softmax fallback\n", fp(bounds[i]), i);
            return 11;
        }
        float sum = 0;
        for (unsigned j = 0; j < keys; ++j) {
            float score = 0;
            for (unsigned k = 0; k < n; ++k) score += fp(q[i * n + k]) * fp(kt[k * keys + j]);
            ref_p[i * keys + j] = j < valid_keys ? bf(expf(fp(bf(score)) - fp(bounds[i]))) : 0;
            sum += fp(ref_p[i * keys + j]);
        }
        ref_l[i] = sum;
        if (!std::isfinite(sum) || sum <= 1e-28f) {
            printf("UNSUPPORTED normalizer=%g row=%u: use stable-softmax fallback\n", sum, i);
            return 11;
        }
        for (unsigned j = 0; j < n; ++j) {
            float num = 0;
            for (unsigned k = 0; k < keys; ++k) num += fp(ref_p[i * keys + k]) * fp(v[k * n + j]);
            ref_o[i * n + j] = bf(fp(bf(num)) / sum);
        }
    }
    memset(host + os, 0xcd, wide + matrix);
    memset(host + ol, 0xcd, n * 4 + 4096);
    std::vector<uint8_t> immutable(host, host + os);
    std::vector<uint16_t> original_bounds(bounds, bounds + n);
    uint64_t data = hw.sram_base_address + 0x10000;
    unsigned down = hltests_get_dma_down_qid(fd, STREAM0), up = hltests_get_dma_up_qid(fd, STREAM0);
    unsigned mq = hltests_get_mme_qid(fd, 0, STREAM0);
    std::vector<unsigned> tqueues;
    for (unsigned i = 0; i < 64 && tqueues.size() < tpcs; ++i)
        if ((hw.tpc_enabled_mask_ext >> i) & 1) tqueues.push_back(hltests_get_tpc_qid(fd, i, STREAM0));
    if (tqueues.size() != tpcs) return 2;
    uint16_t so = hltests_get_first_avail_sob(fd), mon = hltests_get_first_avail_mon(fd);
    int rc = hltests_dma_transfer(fd, down, EB_FALSE, MB_TRUE, hltests_get_device_va_for_host_ptr(fd, host), data, total, DMA_DIR_HOST_TO_SRAM);
    if (rc) return 5;
    uint64_t exp_isa = load_isa(fd, argv[1]), norm_isa = load_isa(fd, argv[2]); if (!exp_isa || !norm_isa) return 6;
    hltests_clear_sobs(fd, 4);
    void *mc = hltests_allocate_host_mem(fd, cbbytes, NOT_HUGE_MAP);
    if (!mc) return 7;
    unsigned first = std::min(keys, 32768u);
    unsigned qk_signals = keys > first ? 2 : 1;
    std::vector<void *> tcommon, tupper;
    std::vector<unsigned> nt(tpcs, 0);
    std::vector<std::vector<uint32_t>> previous_tpc(tpcs);
    std::vector<uint32_t> previous_mme;
    for (unsigned i = 0; i < tpcs; ++i) {
        void *tc = hltests_allocate_host_mem(fd, cbbytes, NOT_HUGE_MAP);
        if (!tc) return 7;
        tcommon.push_back(tc);
    }
    unsigned nm = 0;
    for (unsigned step = 0; step < iterations; ++step) {
        auto *pm = batch ? &previous_mme : nullptr;
        // Reusing S/P and O requires every previous normalizer to have finished.
        if (step) nm = wait(fd, mc, nm, mq, so + 3, mon + 2 * tpcs + 3, true, step * tpcs);
        nm = gemm(fd, mc, nm, data + oq, data + ok, data + os, so, first, n, keys, pm);
        if (keys > first) {
            nm = wait(fd, mc, nm, mq, so, mon + 2 * tpcs + 2, true, step * qk_signals + 1);
            nm = gemm(fd, mc, nm, data + oq, data + ok + first * 2, data + os + first * 2,
                      so, keys - first, n, keys, pm);
        }
        nm = wait(fd, mc, nm, mq, so + 1, mon + tpcs, true, (step + 1) * tpcs);
        nm = gemm(fd, mc, nm, data + os, data + ov, data + oo, so + 2, n, keys, 0, pm);
        for (unsigned i = 0; i < tpcs; ++i) {
            auto *pt = batch ? &previous_tpc[i] : nullptr;
            unsigned first_row = n * i / tpcs, rows = n * (i + 1) / tpcs - first_row;
            void *tc = tcommon[i];
            nt[i] = wait(fd, tc, nt[i], tqueues[i], so, mon + i, true, (step + 1) * qk_signals);
            nt[i] = tpc(fd, tc, nt[i], exp_isa, {{data + os, keys, n, 6}, {data + ob, 1, n, 6},
                     {data + os, keys, n, 6}, {data + ol, 1, n, 7}}, rows, so + 1, first_row, valid_keys, pt);
            nt[i] = wait(fd, tc, nt[i], tqueues[i], so + 2, mon + tpcs + 1 + i, true, step + 1);
            nt[i] = tpc(fd, tc, nt[i], norm_isa, {{data + oo, n, n, 6}, {data + ol, 1, n, 7},
                     {data + oo, n, n, 6}}, rows, so + 3, first_row, 0, pt);
            if (nt[i] > cbbytes) return 7;
        }
        if (nm > cbbytes) return 7;
    }
    void *mu = hltests_create_cb(fd, 4096, INTERNAL, hw.sram_base_address + 0x2000);
    if (!mu) return 7;
    auto upper = [&](void *dst, void *src, unsigned count, uint64_t address) {
        hltests_pkt_info p{}; p.cp_dma.src_addr = hltests_get_device_va_for_host_ptr(fd, src); p.cp_dma.size = count;
        unsigned length = hltests_add_cp_dma_pkt(fd, dst, 0, &p);
        int status = hltests_dma_transfer(fd, down, EB_FALSE, MB_TRUE, hltests_get_device_va_for_host_ptr(fd, dst), address, length, DMA_DIR_HOST_TO_SRAM);
        return status ? 0 : length;
    };
    unsigned nmu = upper(mu, mc, nm, hw.sram_base_address + 0x2000);
    if (!nmu) return 7;
    std::vector<hltests_cs_chunk> chunks = {{mu, nmu, mq}};
    unsigned total_tpc_bytes = 0;
    for (unsigned i = 0; i < tpcs; ++i) {
        // Upper command buffers must stay below the data area at SRAM+0x10000.
        // Each is only one CP_DMA packet; reserve a separate 128-byte slot.
        uint64_t address = hw.sram_base_address + 0x3000 + 128 * i;
        void *tu = hltests_create_cb(fd, 128, INTERNAL, address);
        if (!tu) return 7;
        unsigned ntu = upper(tu, tcommon[i], nt[i], address); if (!ntu) return 7;
        tupper.push_back(tu);
        chunks.push_back({tu, ntu, tqueues[i]}); total_tpc_bytes += nt[i];
    }
    void *done = hltests_create_cb(fd, 4096, EXTERNAL, 0);
    if (!done) return 7;
    unsigned nd = wait(fd, done, 0, up, so + 3, mon + 2 * tpcs + 1, false, iterations * tpcs);
    hltests_pkt_info p{}; p.mb = MB_TRUE; nd = hltests_add_nop_pkt(fd, done, nd, &p);
    chunks.push_back({done, nd, up});
    uint64_t seq = 0;
    printf("ATTENTION queries=128 keys=%u dim=128 tpcs=%u uniform=%d batch_iterations=%u mme_bytes=%u tpc_bytes=%u SRAM=0x%lx bytes=%u\n", keys, tpcs, uniform, iterations, nm, total_tpc_bytes, data, total);
    std::vector<double> times;
    double cold_us = 0;
    // One warm-up plus repeats measured replays. Initial upload, SO clearing and
    // final readback are outside the timer. This is a resident microbenchmark.
    unsigned samples = batch ? 5 : repeats;
    for (unsigned run = 0; run <= samples; ++run) {
        hltests_clear_sobs(fd, 4);
        auto before = std::chrono::steady_clock::now();
        rc = hltests_submit_cs(fd, nullptr, 0, chunks.data(), chunks.size(), 0, &seq); if (rc) return 8;
        rc = hltests_wait_for_cs(fd, seq, 5000000); if (rc != HL_WAIT_CS_STATUS_COMPLETED) { printf("WAIT status=%d\n", rc); return 9; }
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - before).count();
        us /= iterations;
        if (run) times.push_back(us); else cold_us = us;
    }
    std::sort(times.begin(), times.end());
    printf("BENCH resident_submit_wait_us_per_attention cold=%g median=%g p10=%g p90=%g samples=%u batch_iterations=%u\n",
           cold_us, times[times.size() / 2], times[(times.size() - 1) / 10], times[(times.size() - 1) * 9 / 10], samples, iterations);
    rc = hltests_dma_transfer(fd, up, EB_FALSE, MB_TRUE, data, hltests_get_device_va_for_host_ptr(fd, host), total, DMA_DIR_SRAM_TO_HOST);
    if (rc) return 9;
    auto *actual_p = (uint16_t *)(host + os), *actual_o = (uint16_t *)(host + oo);
    auto *actual_l = (float *)(host + ol);
    double err2 = 0, ref2 = 0, full_err2 = 0, full_ref2 = 0;
    unsigned p_bad = 0, o_bits = 0, guard_bad = 0; float l_error = 0;
    bool finite = true;
    for (unsigned i = 0; i < n * n; ++i) {
        o_bits += actual_o[i] != ref_o[i];
        double error = fp(actual_o[i]) - fp(ref_o[i]); err2 += error * error; ref2 += (double)fp(ref_o[i]) * fp(ref_o[i]);
        finite = finite && std::isfinite(fp(actual_o[i]));
        if (!full_ref.empty()) {
            double e = fp(actual_o[i]) - full_ref[i]; full_err2 += e * e;
            full_ref2 += (double)full_ref[i] * full_ref[i];
        }
    }
    for (unsigned i = 0; i < n * keys; ++i) {
        p_bad += actual_p[i] != ref_p[i];
        finite = finite && std::isfinite(fp(actual_p[i])) && fp(actual_p[i]) >= 0;
    }
    for (unsigned i = 0; i < n; ++i) {
        l_error = std::max(l_error, fabsf(actual_l[i] - ref_l[i]) / ref_l[i]);
        finite = finite && std::isfinite(actual_l[i]) && actual_l[i] > 0;
    }
    for (unsigned i = og; i < total; ++i) guard_bad += host[i] != 0xcd;
    double rms = sqrt(err2 / std::max(ref2, 1e-30));
    double full_rms = sqrt(full_err2 / std::max(full_ref2, 1e-30));
    bool input_changed = memcmp(immutable.data(), host, os) || memcmp(original_bounds.data(), bounds, n * 2);
    printf("RESULT SRAM_ATTENTION p_bits_differ=%u output_bits_differ=%u relative_rms=%g rowsum_relative_max=%g guard_bad=%u input_changed=%d\n", p_bad, o_bits, rms, l_error, guard_bad, input_changed);
    if (!full_ref.empty()) printf("RESULT REAL_QKV fp32_softmax_relative_rms=%g valid_keys=%u\n", full_rms, valid_keys);
    hltests_destroy_cb(fd, done);
    for (void *tu : tupper) hltests_destroy_cb(fd, tu);
    for (void *tc : tcommon) hltests_free_host_mem(fd, tc);
    hltests_destroy_cb(fd, mu); hltests_free_host_mem(fd, mc);
    hltests_free_device_mem(fd, (void *)(uintptr_t)exp_isa); hltests_free_device_mem(fd, (void *)(uintptr_t)norm_isa);
    hltests_free_host_mem(fd, host); hltests_close(fd); hltests_fini();
    return !finite || !std::isfinite(rms) || rms > 0.01 || full_rms > 0.01 || l_error > 0.003 || guard_bad || input_changed ? 10 : 0;
}
