/* SPDX-License-Identifier: MIT
 * Fixed-shape Gaudi2 BF16 GEMM without Synapse, SCAL, or PyTorch at execution.
 * The descriptor originated from SynapseAI 1.24.1; see fixtures and README.
 */
#define HLTESTS_LIB_MODE
#include "hlthunk_tests.h"
#include <getopt.h>
#include <math.h>
#include <string.h>
#include "desc128.h"

enum { N = 128, MATRIX_BYTES = N * N * 2, GUARD_BYTES = 4096, CB_BYTES = 4096 };

static uint16_t to_bf16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

static float from_bf16(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void reference(const uint16_t *a, const uint16_t *b, uint16_t *c)
{
    for (unsigned i = 0; i < N; ++i)
        for (unsigned j = 0; j < N; ++j) {
            float sum = 0;
            for (unsigned k = 0; k < N; ++k)
                sum += from_bf16(a[i * N + k]) * from_bf16(b[k * N + j]);
            c[i * N + j] = to_bf16(sum);
        }
}

static unsigned append_gemm(int fd, void *cb, unsigned size,
                            uint64_t a, uint64_t b, uint64_t c, uint16_t sob)
{
    uint32_t words[182];
    memcpy(words, desc_words, sizeof(words));
#define SET(off, value) words[((off) - 8) / 4] = (uint32_t)(value)
    SET(0x10, c); SET(0x14, c >> 32);
    SET(0x18, a); SET(0x1c, a >> 32);
    SET(0x20, b); SET(0x24, b >> 32);
    /* The sync address register takes the low 32 bits of the LBW address. */
    SET(0x14c, hltests_get_sob_base_addr(fd) + sob * 4);
    SET(0x150, 0x80000001); /* SOB_ADD(1) */
#undef SET
    for (unsigned i = 0; i < 182; ++i) {
        struct hltests_pkt_info p = {0};
        p.mb = MB_TRUE;
        p.wreg32.reg_addr = (uint16_t)(0xb008 + i * 4);
        p.wreg32.value = words[i];
        size = hltests_add_wreg32_pkt(fd, cb, size, &p);
    }
    struct hltests_pkt_info p = {0};
    p.eb = EB_TRUE; p.mb = MB_TRUE;
    p.wreg32.reg_addr = 0xb004;
    /* Enable AGU inputs, EU, AP, outputs; copy shadow descriptor and advance. */
    p.wreg32.value = 0x3ff;
    return hltests_add_wreg32_pkt(fd, cb, size, &p);
}

static unsigned append_wait(int fd, void *cb, unsigned size, unsigned queue,
                            uint16_t sob, uint16_t mon, bool common_cp)
{
    struct hltests_monitor_and_fence m = {0};
    m.queue_id = (uint8_t)queue; m.cmdq_fence = common_cp;
    m.sob_id = sob; m.mon_id = mon; m.sob_val = 1;
    m.dec_fence = true; m.mon_payload = 1; m.mon_mode = SOB_EQUAL;
    return hltests_add_monitor_and_fence(fd, cb, size, &m);
}

static void usage(const char *exe)
{
    fprintf(stderr, "Usage: %s [--mode hbm|sram|chain] [--pattern random|identity] "
                    "[--seed UINT32] [--busid PCI_ID]\n", exe);
}

int main(int argc, char **argv)
{
    const char *mode = "hbm", *pattern = "random", *busid = "0000:08:00.0";
    uint32_t seed = 0x20261108;
    static const struct option options[] = {
        {"mode", required_argument, 0, 'm'}, {"pattern", required_argument, 0, 'p'},
        {"seed", required_argument, 0, 's'}, {"busid", required_argument, 0, 'b'},
        {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "", options, NULL)) != -1) {
        switch (opt) {
        case 'm': mode = optarg; break;
        case 'p': pattern = optarg; break;
        case 'b': busid = optarg; break;
        case 's': {
            char *end;
            errno = 0;
            unsigned long parsed = strtoul(optarg, &end, 0);
            if (errno || !*optarg || *end || parsed > UINT32_MAX) { usage(argv[0]); return 2; }
            seed = (uint32_t)parsed;
            break;
        }
        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 2;
        }
    }
    bool chain = strcmp(mode, "chain") == 0;
    bool sram = strcmp(mode, "sram") == 0 || chain;
    bool identity = strcmp(pattern, "identity") == 0;
    if (optind != argc || (!sram && strcmp(mode, "hbm")) ||
        (!identity && strcmp(pattern, "random"))) { usage(argv[0]); return 2; }

    setbuf(stdout, NULL);
    int fd = -1, result = 1;
    bool initialized = false, pending = false;
    uint16_t *host = NULL, *ref_c = NULL, *ref_d = NULL;
    void *device = NULL, *common = NULL, *upper = NULL, *done = NULL;
    const unsigned outputs = chain ? 2 : 1;
    const unsigned total = MATRIX_BYTES * (2 + outputs) + GUARD_BYTES;
    int rc = hltests_init();
    if (rc) goto cleanup;
    initialized = true;
    fd = hltests_open(busid);
    if (fd < 0) { fprintf(stderr, "Cannot open Gaudi2 at %s\n", busid); goto cleanup; }
    struct hlthunk_hw_ip_info hw = {0};
    if (hlthunk_get_hw_ip_info(fd, &hw) || !hltests_is_gaudi2(fd) ||
        !(hw.mme_enabled_mask & 1) || !hltests_is_legacy_mode_enabled(fd) ||
        hw.sram_size < 0x10000 + total) {
        fprintf(stderr, "Requires Gaudi2 MME0, legacy queues, and available SRAM\n");
        goto cleanup;
    }
    unsigned mq = hltests_get_mme_qid(fd, 0, STREAM0);
    unsigned up = hltests_get_dma_up_qid(fd, STREAM0);
    unsigned down = hltests_get_dma_down_qid(fd, STREAM0);
    uint16_t sob = hltests_get_first_avail_sob(fd), mon = hltests_get_first_avail_mon(fd);
    host = hltests_allocate_host_mem(fd, total, NOT_HUGE_MAP);
    if (!sram) device = hltests_allocate_device_mem(fd, total, 0, NOT_CONTIGUOUS);
    common = hltests_allocate_host_mem(fd, CB_BYTES, NOT_HUGE_MAP);
    uint64_t upper_address = hw.sram_base_address + 0x2000;
    upper = hltests_create_cb(fd, CB_BYTES, INTERNAL, upper_address);
    done = hltests_create_cb(fd, CB_BYTES, EXTERNAL, 0);
    ref_c = malloc(MATRIX_BYTES); ref_d = chain ? malloc(MATRIX_BYTES) : NULL;
    if (!host || (!sram && !device) || !common || !upper || !done || !ref_c || (chain && !ref_d)) {
        fprintf(stderr, "Allocation failed\n"); goto cleanup;
    }
    uint16_t *a = host, *b = a + N * N, *c = b + N * N, *d = c + N * N;
    uint32_t rng = seed;
    for (unsigned j = 0; j < N * N * 2; ++j) {
        rng = rng * 1664525 + 1013904223;
        host[j] = to_bf16((int)(rng % 9) - 4);
    }
    if (identity)
        for (unsigned i = 0; i < N; ++i)
            for (unsigned j = 0; j < N; ++j) b[i * N + j] = to_bf16(i == j ? 1.0f : 0.0f);
    memset(c, 0xcd, outputs * MATRIX_BYTES + GUARD_BYTES);
    reference(a, b, ref_c);
    if (chain) reference(ref_c, b, ref_d); /* BF16 materialization between the two GEMMs. */
    uint64_t da = sram ? hw.sram_base_address + 0x10000 : (uintptr_t)device;
    uint64_t db = da + MATRIX_BYTES, dc = db + MATRIX_BYTES, dd = dc + MATRIX_BYTES;
    uint64_t hva = hltests_get_device_va_for_host_ptr(fd, host);
    rc = hltests_dma_transfer(fd, down, EB_FALSE, MB_TRUE, hva, da, total,
                              sram ? DMA_DIR_HOST_TO_SRAM : DMA_DIR_HOST_TO_DRAM);
    if (rc) { fprintf(stderr, "Input DMA failed: %d\n", rc); goto cleanup; }
    hltests_clear_sobs(fd, (uint16_t)outputs);
    unsigned nc = append_gemm(fd, common, 0, da, db, dc, sob);
    if (chain) {
        nc = append_wait(fd, common, nc, mq, sob, mon, true);
        nc = append_gemm(fd, common, nc, dc, db, dd, (uint16_t)(sob + 1));
    }
    if (nc > CB_BYTES) { fprintf(stderr, "Command buffer too small\n"); goto cleanup; }
    struct hltests_pkt_info p = {0};
    p.cp_dma.src_addr = hltests_get_device_va_for_host_ptr(fd, common); p.cp_dma.size = nc;
    unsigned nu = hltests_add_cp_dma_pkt(fd, upper, 0, &p);
    rc = hltests_dma_transfer(fd, down, EB_FALSE, MB_TRUE,
                              hltests_get_device_va_for_host_ptr(fd, upper), upper_address,
                              nu, DMA_DIR_HOST_TO_SRAM);
    if (rc) { fprintf(stderr, "Command DMA failed: %d\n", rc); goto cleanup; }
    unsigned nd = append_wait(fd, done, 0, up, (uint16_t)(sob + outputs - 1),
                              (uint16_t)(mon + (chain ? 1 : 0)), false);
    memset(&p, 0, sizeof(p));
    p.mb = MB_TRUE; p.dma.src_addr = dc; p.dma.dst_addr = hva + MATRIX_BYTES * 2;
    p.dma.size = MATRIX_BYTES * outputs + GUARD_BYTES;
    p.dma.dma_dir = sram ? DMA_DIR_SRAM_TO_HOST : DMA_DIR_DRAM_TO_HOST;
    nd = hltests_add_dma_pkt(fd, done, nd, &p);
    struct hltests_cs_chunk chunks[2] = {{upper, nu, mq}, {done, nd, up}};
    uint64_t seq = 0;
    printf("CONFIG mode=%s pattern=%s seed=%u shape=128x128x128 mq=%u sob=%u "
           "A=0x%lx B=0x%lx C=0x%lx command_bytes=%u\n", mode, pattern, seed,
           mq, sob, da, db, dc, nc);
    rc = hltests_submit_cs(fd, NULL, 0, chunks, 2, 0, &seq);
    if (rc) { fprintf(stderr, "Submission failed: %d\n", rc); goto cleanup; }
    pending = true;
    rc = hltests_wait_for_cs(fd, seq, 5000000);
    if (rc != HL_WAIT_CS_STATUS_COMPLETED) {
        fprintf(stderr, "Completion failed: status=%d seq=%lu; closing may reset the device\n", rc, seq);
        goto cleanup;
    }
    pending = false;
    unsigned bad = 0, guard_bad = 0;
    float max_error = 0;
    for (unsigned which = 0; which < outputs; ++which) {
        const uint16_t *actual = which ? d : c, *expected = which ? ref_d : ref_c;
        for (unsigned i = 0; i < N * N; ++i) {
            if (actual[i] != expected[i]) {
                if (bad < 8) fprintf(stderr, "DIFF output=%u row=%u col=%u got=%g expected=%g\n",
                                      which, i / N, i % N, from_bf16(actual[i]), from_bf16(expected[i]));
                ++bad;
            }
            float error = fabsf(from_bf16(actual[i]) - from_bf16(expected[i]));
            if (error > max_error) max_error = error;
        }
    }
    uint16_t *guard = c + outputs * N * N;
    for (unsigned i = 0; i < GUARD_BYTES / 2; ++i) if (guard[i] != 0xcdcd) ++guard_bad;
    printf("RESULT mode=%s pattern=%s seed=%u checked=%u bad=%u max_abs_error=%g guard_bad=%u\n",
           mode, pattern, seed, outputs * N * N, bad, max_error, guard_bad);
    result = bad || guard_bad ? 1 : 0;

cleanup:
    free(ref_d); free(ref_c);
    /* A timed-out CS can still access these mappings. Leave them to driver teardown. */
    if (fd >= 0 && !pending) {
        if (done) hltests_destroy_cb(fd, done);
        if (upper) hltests_destroy_cb(fd, upper);
        if (common) hltests_free_host_mem(fd, common);
        if (device) hltests_free_device_mem(fd, device);
        if (host) hltests_free_host_mem(fd, host);
    }
    if (fd >= 0) hltests_close(fd);
    if (initialized) hltests_fini();
    return result;
}
