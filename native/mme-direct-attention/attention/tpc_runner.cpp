// SPDX-License-Identifier: MIT
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include <cmath>
#define HLTESTS_LIB_MODE
extern "C" {
#include <hlthunk_tests.h>
}
#undef fail
#include "gaudi2/gaudi2_tpc_descriptor.h"

static unsigned reg(int fd, void *cb, unsigned off, unsigned addr, uint32_t value, bool barrier = false)
{
    hltests_pkt_info p{};
    p.mb = MB_TRUE; p.eb = barrier ? EB_TRUE : EB_FALSE;
    p.wreg32.reg_addr = (uint16_t)addr; p.wreg32.value = value;
    return hltests_add_wreg32_pkt(fd, cb, off, &p);
}

static void tensor(gaudi2::block_tpc_tensor &t, uint64_t address, unsigned cols, unsigned rows)
{
    t.base_addr_low._raw = (uint32_t)address;
    t.base_addr_high._raw = (uint32_t)(address >> 32);
    t.tensor_config.data_type = 6; // BF16, from TensorDescriptor.h
    t.tensor_config.valid_dim_mask = 0x1f;
    t.tensor_config.last_dim = 1;
    t.dim_0_size.v = cols; t.dim_0_stride.v = 1;
    t.dim_1_size.v = rows; t.dim_1_stride.v = cols;
    t.dim_2_size.v = 1; t.dim_2_stride.v = cols * rows;
    t.dim_3_size.v = 1; t.dim_3_stride.v = cols * rows;
    t.dim_4_size.v = 1; t.dim_4_stride.v = cols * rows;
}

int main(int argc, char **argv)
{
    setbuf(stdout, nullptr);
    if (argc < 2 || argc > 3) { fprintf(stderr, "Usage: tpc_runner kernel.bin [exp]\n"); return 2; }
    bool exp_mode = argc == 3 && strcmp(argv[2], "exp") == 0;
    std::ifstream file(argv[1], std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(file)), {});
    if (code.empty() || code.size() > 0x10000) return 3;
    if (hltests_init()) return 4;
    int fd = hltests_open("0000:08:00.0"); if (fd < 0) return 5;
    hlthunk_hw_ip_info hw{}; hlthunk_get_hw_ip_info(fd, &hw);
    constexpr unsigned n = 128, bytes = n * n * 2, guard = 4096;
    constexpr unsigned total = 2 * bytes + guard, cbsize = 16384;
    auto *host = (uint16_t *)hltests_allocate_host_mem(fd, total, NOT_HUGE_MAP);
    void *isa = hltests_allocate_device_mem(fd, 0x10000, 0, NOT_CONTIGUOUS);
    auto *isa_host = (char *)hltests_allocate_host_mem(fd, 0x10000, NOT_HUGE_MAP);
    void *common = hltests_allocate_host_mem(fd, cbsize, NOT_HUGE_MAP);
    uint64_t upper_addr = hw.sram_base_address + 0x2000;
    void *upper = hltests_create_cb(fd, 4096, INTERNAL, upper_addr);
    void *done = hltests_create_cb(fd, 4096, EXTERNAL, 0);
    if (!host || !isa || !isa_host || !common || !upper || !done) return 6;
    auto bf = [](float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); };
    auto fp = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; };
    for (unsigned i = 0; i < n * n; ++i)
        host[i] = exp_mode ? bf(-(float)(i % 129) / 16) : ((i & 1) ? 0xbf80 : 0x4000);
    memset(host + n * n, 0xcd, bytes + guard);
    memset(isa_host, 0, 0x10000); memcpy(isa_host, code.data(), code.size());
    uint64_t data = hw.sram_base_address + 0x10000;
    unsigned down = hltests_get_dma_down_qid(fd, STREAM0), up = hltests_get_dma_up_qid(fd, STREAM0);
    unsigned tq = hltests_get_tpc_qid(fd, 0, STREAM0);
    uint16_t sob = hltests_get_first_avail_sob(fd), mon = hltests_get_first_avail_mon(fd);
    int rc = hltests_dma_transfer(fd, down, EB_FALSE, MB_TRUE,
            hltests_get_device_va_for_host_ptr(fd, host), data, total, DMA_DIR_HOST_TO_SRAM);
    if (rc) return 7;
    rc = hltests_dma_transfer(fd, down, EB_FALSE, MB_TRUE,
            hltests_get_device_va_for_host_ptr(fd, isa_host), (uintptr_t)isa, 0x10000, DMA_DIR_HOST_TO_DRAM);
    if (rc) return 8;
    hltests_clear_sobs(fd, 1);
    gaudi2::Gaudi2TpcDesc desc{};
    tensor(desc.m_tensors[0], data, n, n);
    tensor(desc.m_tensors[1], data + bytes, n, n);
    desc.m_so.message._raw = 0x80000001;
    desc.m_so.addr.v = (uint32_t)(hltests_get_sob_base_addr(fd) + sob * 4);
    auto &d = desc.m_desc;
    d.kernel_base_address_low._raw = (uint32_t)(uintptr_t)isa;
    d.kernel_base_address_high._raw = (uint32_t)((uintptr_t)isa >> 32);
    d.tid_size_dim_0.v = 1; d.tid_size_dim_1.v = n;
    d.tid_size_dim_2.v = 1; d.tid_size_dim_3.v = 1; d.tid_size_dim_4.v = 1;
    d.kernel_config.aso_evict_l0 = 1; d.kernel_config.num_valid_srfs = 32;
    d.kernel_config.rd_rate_limit_rst_token = 8;
    d.kernel_config.wr_rate_limit_rst_token = 6;
    d.kernel_config.irf_32bit_compatibility = 1;
    unsigned nc = 0;
    unsigned base = 0xb000 + offsetof(gaudi2::block_tpc, qm_tensor_0);
    const auto *words = reinterpret_cast<const uint32_t *>(&desc);
    for (unsigned i = 0; i < sizeof(desc) / 4; ++i) nc = reg(fd, common, nc, base + i * 4, words[i]);
    // SM_BASE_ADDRESS_HIGH is privileged on secured Gaudi2. Keep the value
    // installed by the driver; only write the job's low sync address above.
    nc = reg(fd, common, nc, 0xb000 + offsetof(gaudi2::block_tpc, tpc_cmd), 0xf, true);
    nc = reg(fd, common, nc, 0xb000 + offsetof(gaudi2::block_tpc, tpc_execute), 1);
    hltests_pkt_info p{};
    p.cp_dma.src_addr = hltests_get_device_va_for_host_ptr(fd, common); p.cp_dma.size = nc;
    unsigned nu = hltests_add_cp_dma_pkt(fd, upper, 0, &p);
    rc = hltests_dma_transfer(fd, down, EB_FALSE, MB_TRUE,
            hltests_get_device_va_for_host_ptr(fd, upper), upper_addr, nu, DMA_DIR_HOST_TO_SRAM);
    if (rc) return 9;
    hltests_monitor_and_fence m{};
    m.queue_id = (uint8_t)up; m.sob_id = sob; m.mon_id = mon;
    m.sob_val = 1; m.dec_fence = true; m.mon_payload = 1; m.mon_mode = SOB_EQUAL;
    unsigned nd = hltests_add_monitor_and_fence(fd, done, 0, &m);
    memset(&p, 0, sizeof(p)); p.mb = MB_TRUE;
    p.dma.src_addr = data + bytes;
    p.dma.dst_addr = hltests_get_device_va_for_host_ptr(fd, host) + bytes;
    p.dma.size = bytes + guard; p.dma.dma_dir = DMA_DIR_SRAM_TO_HOST;
    nd = hltests_add_dma_pkt(fd, done, nd, &p);
    hltests_cs_chunk chunks[2] = {{upper, nu, tq}, {done, nd, up}};
    uint64_t seq = 0;
    printf("TPC descriptor=%zu base=0x%x cmd=0x%lx q=%u bytes=%u isa=0x%lx\n",
           sizeof(desc), base, 0xb000 + offsetof(gaudi2::block_tpc, tpc_cmd), tq, nc, (uintptr_t)isa);
    rc = hltests_submit_cs(fd, nullptr, 0, chunks, 2, 0, &seq);
    if (rc) return 10;
    rc = hltests_wait_for_cs(fd, seq, 5000000);
    printf("WAIT status=%d\n", rc); if (rc != HL_WAIT_CS_STATUS_COMPLETED) return 11;
    unsigned bad = 0, guard_bad = 0;
    float max_relative = 0;
    for (unsigned i = 0; i < n * n; ++i) {
        uint16_t expected = exp_mode ? bf(expf(fp(host[i]))) : ((i & 1) ? 0 : 0x4000);
        float relative = fabsf(fp(host[n * n + i]) - fp(expected)) / fmaxf(fabsf(fp(expected)), 1e-30f);
        max_relative = fmaxf(max_relative, relative);
        if (exp_mode ? relative > 0.008f : host[n * n + i] != expected) {
            if (bad < 5) printf("DIFF %u %04x expected=%04x\n", i, host[n * n + i], expected);
            ++bad;
        }
    }
    for (unsigned i = 0; i < guard / 2; ++i) if (host[2 * n * n + i] != 0xcdcd) ++guard_bad;
    printf("RESULT TPC_SRAM_%s bad=%u max_relative=%g guard_bad=%u\n", exp_mode ? "EXP" : "RELU", bad, max_relative, guard_bad);
    hltests_destroy_cb(fd, done); hltests_destroy_cb(fd, upper);
    hltests_free_host_mem(fd, common); hltests_free_host_mem(fd, isa_host);
    hltests_free_device_mem(fd, isa); hltests_free_host_mem(fd, host);
    hltests_close(fd); hltests_fini();
    return bad || guard_bad ? 12 : 0;
}
