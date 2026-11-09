// SPDX-License-Identifier: MIT
// Isolated finite-batch experiment. Every query tile and output is distinct;
// only S/P is double buffered. All per-tile O_A/O_B/stats survive until NM.
struct BatchPlan {
    std::vector<PreparedEcb> qk, pv;
    PreparedEcb ex, nm;
    unsigned n = 0, tpcs = 0, sob = 0, monitor = 0;
    bool ready = false;
};
// Slot 1 (opt-in) is the same plan with its own Q/OA/OB/stats SRAM region, so
// the host can DMA group g+1's Q and group g-1's outputs while group g runs.
// Both slots share SOBs/monitors: only one slot is ever submitted at a time.
static BatchPlan batch_plans[2];
static BatchPlan &batch_plan = batch_plans[0];
static synStreamHandle batch_dma_stream = nullptr;
static bool batch_pending = false;
static int batch_split_fence = 0; // see context_batch_set_split_fence
static int batch_prefetch_desc = 0;
static int batch_mme_nobarrier = 0;
extern "C" int context_batch_set_mme_nobarrier(int on)
{
    if (batch_plans[0].ready || (on && !g_mme_nobarrier_allowed)) return -EINVAL;
    batch_mme_nobarrier = on != 0;
    return 0;
}
extern "C" int context_batch_set_prefetch_desc(int on)
{
    if (batch_plans[0].ready) return -EINVAL;
    batch_prefetch_desc = on != 0;
    return 0;
}

extern "C" int context_batch_layout(uint64_t base, unsigned keys, unsigned n, uint64_t *out)
{
    if (!out || base % 256 || keys < 256 || keys > 37760 || keys % 128 || n < 2 || n > 20) return -EINVAL;
    uint64_t a = base;
    auto take = [&](uint64_t bytes) { uint64_t r = a; a = (a + bytes + 255) & ~255ull; return r; };
    uint64_t matrix = 128 * 128 * 2, wide = 128 * keys * 2;
    // q, kt, v, S0, S1, OA, OB, stats, guard, end
    out[0] = take(n * matrix); out[1] = take(wide); out[2] = take(wide);
    out[3] = take(wide); out[4] = take(wide);
    out[5] = take(n * matrix); out[6] = take(n * matrix); out[7] = take(n * 2048);
    out[8] = take(4096); out[9] = a;
    return 0;
}

// Second Q/OA/OB/stats set placed after slot 0's end: {q, OA, OB, stats, end}.
extern "C" int context_batch_layout_alt(uint64_t base, unsigned keys, unsigned n, uint64_t *out)
{
    uint64_t L[10];
    int rc = context_batch_layout(base, keys, n, L); if (rc || !out) return rc ? rc : -EINVAL;
    uint64_t a = L[9];
    auto take = [&](uint64_t bytes) { uint64_t r = a; a = (a + bytes + 255) & ~255ull; return r; };
    const uint64_t matrix = 128 * 128 * 2;
    out[0] = take(n * matrix); out[1] = take(n * matrix); out[2] = take(n * matrix);
    out[3] = take(n * 2048); out[4] = a;
    return 0;
}

// Cache the decoded QM tensor/launch descriptor, emitting only changed words
// between consecutive exp kernels on the same TPC. No protected register writes.
// TPC_CMD for delta launches (2nd and later exp kernel of a TPC program):
// bit0 icache, bit1 dcache, bit2 lcache (LUT), bit3 tcache invalidate
// (synapseai-core gaudi2 tpc_regs.h reg_tpc_cmd). Default 0xf = all, as
// validated. The kernel code and LUT tables do not change between these
// launches; S does (MME), so dcache invalidation must stay. Opt-in experiment.
static uint32_t g_tpc_delta_cmd = 0xf;
extern "C" int context_set_tpc_delta_cmd(unsigned cmd)
{
    if (!(cmd & 0x2) || cmd > 0xf) return -EINVAL; // never drop dcache invalidation
    g_tpc_delta_cmd = cmd;
    return 0;
}

static int append_batch_exp(std::vector<char> &raw, std::vector<uint32_t> &previous,
                             uint64_t isa, const uint64_t *addresses, const unsigned *cols,
                             const unsigned *rows, const unsigned *types, const unsigned *strides,
                             unsigned valid, unsigned first, unsigned count, unsigned sob, const uint32_t *padding = nullptr)
{
    std::vector<char> temp;
    int rc = append_tpc(temp, isa, addresses, cols, rows, types, 3, valid, first, count, sob, strides, padding);
    if (rc) return rc;
    constexpr unsigned begin = 0xb5dc, words = sizeof(gaudi2::Gaudi2TpcDesc) / 4;
    std::vector<uint32_t> current(words, 0);
    for (unsigned off = 0; off < temp.size();) {
        auto *p = (const packet_wreg32 *)(temp.data() + off);
        if (p->opcode == PACKET_WREG_BULK) {
            auto *b = (const packet_wreg_bulk *)p;
            if (b->reg_offset < begin || (unsigned)b->reg_offset + (unsigned)b->size64 * 8 > begin + words * 4) return -EINVAL;
            memcpy(current.data() + (b->reg_offset - begin) / 4, temp.data() + off + 8, b->size64 * 8);
            off += 8 + b->size64 * 8;
        } else if (p->opcode == PACKET_WREG_32) {
            if (p->reg_offset >= begin && p->reg_offset < begin + words * 4)
                current[(p->reg_offset - begin) / 4] = p->value;
            off += 8;
        } else return -EINVAL;
    }
    append_private_fence(raw);
    if (previous.empty()) raw.insert(raw.end(), temp.begin(), temp.end());
    else {
        bool first_write = true;
        for (unsigned i = 0; i < words; ++i) {
            if (current[i] == previous[i]) continue;
            push_reg(raw, begin + i * 4, current[i], first_write); first_write = false;
        }
        push_reg(raw, 0xbd94, g_tpc_delta_cmd, true); // wait previous kernel, invalidate caches
        push_reg(raw, 0xbd98, 1);
    }
    previous.swap(current);
    return 0;
}

extern "C" int context_batch_prepare_slot(unsigned slot, uint64_t base, unsigned keys, unsigned valid, unsigned n,
                                           unsigned tpcs, const char *exp_path, const char *norm_path)
{
    if (slot > 1 || (slot == 1 && (!batch_plans[0].ready || batch_plans[0].n != n || batch_plans[0].tpcs != tpcs)))
        return -EINVAL;
    auto &p = batch_plans[slot];
    if (p.ready || pipe_plan.ready || attention_plan.ready || compiler_qk_words.empty() ||
        !g_wreg_bulk || !g_lut_enabled || !tpcs || tpcs > 24 || !valid || valid > keys) return -EINVAL;
    uint64_t L[10];
    int rc = context_batch_layout(base, keys, n, L); if (rc) return rc;
    uint64_t end = L[9];
    if (slot == 1) {
        uint64_t A[5];
        if ((rc = context_batch_layout_alt(base, keys, n, A))) return rc;
        L[0] = A[0]; L[5] = A[1]; L[6] = A[2]; L[7] = A[3]; end = A[4];
    }
    synDeviceInfoV2 info{};
    if (synDeviceGetInfoV2(0, &info) != synSuccess || base < info.sramBaseAddress + 0x200000 ||
        end > info.sramBaseAddress + info.sramSize) return -EINVAL;
    if ((rc = private_sync_resources(p.sob, p.monitor))) return rc;
    if (private_so_count < 3 * n || private_monitor_count < 28 * n + tpcs) return -ENOSPC;
    uint64_t exp = load_runtime_isa(exp_path), norm = load_runtime_isa(norm_path);
    if (!exp || !norm) return -EINVAL;
    p.n = n; p.tpcs = tpcs; p.qk.resize(n); p.pv.resize(n);
    const uint64_t matrix = 128 * 128 * 2;
    const unsigned first = (keys / 128 + 1) / 2 * 128, second = keys - first;
    std::vector<std::vector<char>> ex(24), nm(24);
    std::vector<std::vector<uint32_t>> cached(24);
    for (unsigned b = 0; b < n; ++b) {
        const uint64_t S = L[3 + (b & 1)], Q = L[0] + b * matrix;
        std::vector<std::vector<char>> qk(2), pv(2);
        const unsigned qk_fence_id = batch_split_fence ? 2 : 3;
        if (batch_mme_nobarrier) g_push_eng_barrier = 0;
        bool fenced[2] = {false, false};
        if (!batch_prefetch_desc) {
            append_private_fence(qk[0], qk_fence_id); append_private_fence(qk[1], qk_fence_id);
            fenced[0] = fenced[1] = true;
        }
        for (unsigned j = 0; j < compiler_qk_engines.size(); ++j) {
            unsigned e = compiler_qk_engines[j];
            if (!fenced[e]) { g_mme_prefence_id = qk_fence_id; fenced[e] = true; }
            append_compiler_mme(qk[e], compiler_qk_words.data() + j * 182, Q, L[1], S, p.sob + 3 * b);
        }
        for (unsigned e = 0; e < 2; ++e) if (!fenced[e]) append_private_fence(qk[e], qk_fence_id);
        for (unsigned h = 0; h < 2; ++h) {
            if (batch_prefetch_desc) g_mme_prefence_id = 3; else append_private_fence(pv[h]);
            rc = append_mme(pv[h], S + (h ? first * 2 : 0), L[2] + (h ? (uint64_t)first * 128 * 2 : 0),
                L[5 + h] + b * matrix, 128, h ? second : first, 128, p.sob + 3 * b + 2, keys);
            g_mme_prefence_id = -1;
            if (rc) { g_push_eng_barrier = 1; return rc; }
            if (batch_mme_nobarrier) g_finish_eng_barrier = 0;
            finish_program(qk[h]); finish_program(pv[h]);
            g_finish_eng_barrier = 1;
        }
        g_push_eng_barrier = 1;
        if ((rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, qk, p.qk[b])) ||
            (rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, pv, p.pv[b]))) return rc;
        uint64_t addresses[3] = {L[7] + b * 2048, S, S + first * 2};
        unsigned cols[3] = {2, first, second}, rows[3] = {256, 128, 128};
        unsigned types[3] = {7, 6, 6}, strides[3] = {2, keys, keys};
        for (unsigned i = 0; i < tpcs; ++i) {
            if (!b) push_lut_bases(ex[i]);
            unsigned unit = 256 * i / tpcs, units = 256 * (i + 1) / tpcs - unit;
            rc = append_batch_exp(ex[i], cached[i], exp, addresses, cols, rows, types, strides,
                                   valid, unit, units, p.sob + 3 * b + 1);
            if (rc) return rc;
        }
    }
    uint64_t addresses[4] = {L[5], L[6], L[7], L[5]};
    unsigned cols[4] = {128, 128, 2, 128}, rows[4] = {128 * n, 128 * n, 256 * n, 128 * n};
    unsigned types[4] = {6, 6, 7, 6};
    for (unsigned i = 0; i < 24; ++i) {
        if (i < tpcs) {
            unsigned row = 128 * n * i / tpcs, count = 128 * n * (i + 1) / tpcs - row;
            rc = append_tpc_fenced(nm[i], norm, addresses, cols, rows, types, 4, valid, row, count, own_sob);
            if (rc) return rc;
        }
        finish_program(ex[i]); finish_program(nm[i]);
    }
    if ((rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, ex, p.ex)) ||
        (rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, nm, p.nm))) return rc;
    p.ready = true;
    printf("BATCH_PLAN slot=%u blocks=%u keys=%u tpcs=%u all_outputs=1 q=0x%lx out=0x%lx end=0x%lx\n",
           slot, n, keys, tpcs, L[0], L[5], end); fflush(stdout);
    return 0;
}

extern "C" int context_batch_prepare(uint64_t base, unsigned keys, unsigned valid, unsigned n,
                                      unsigned tpcs, const char *exp_path, const char *norm_path)
{
    return context_batch_prepare_slot(0, base, keys, valid, n, tpcs, exp_path, norm_path);
}

// Opt-in scheduling/ablation controls (defaults keep the validated order).
// interleave: arm block b's monitors just before queueing block b's ECBs, so
// the scheduler's serial LBW arming overlaps engine work (monitors are GE, so
// a monitor armed after its SOB reached the target fires at arm time).
// ablation 1 = TPC only (profile-only): no MME ECB is queued and NO monitor
// targets an MME fence, so no stray MME fence credits can remain.
static int batch_interleave_arm = 0, batch_ablation = 0;
extern "C" int context_batch_set_options(int interleave, int ablation)
{
    if (batch_plan.ready || ablation < 0 || ablation > 2) return -EINVAL;
    batch_interleave_arm = interleave != 0; batch_ablation = ablation;
    return 0;
}
// split: QK programs fence on fence 2 (credits: both PVs of block b-2 done),
// PV programs keep fence 3 (credits: exp(b) done). Without the split both
// credit kinds share one counter and an early exp credit can release a QK
// that overwrites S still being read by the other MME's older PV.
extern "C" int context_batch_set_split_fence(int split)
{
    if (batch_plan.ready) return -EINVAL;
    batch_split_fence = split != 0;
    return 0;
}

static int batch_arm_block(const BatchPlan &p, unsigned b)
{
    // ablation 2 = MME only (profile-only): no exp ECB, no monitor targets a
    // TPC exp fence, PV credits fire at arm time.
    const bool tpc_only = batch_ablation == 1, mme_only = batch_ablation == 2;
    unsigned mon = p.monitor + 28 * b;
    int rc;
    if (!mme_only && (rc = arm_tpc_monitors(mon, p.tpcs, p.sob + 3 * b, tpc_only ? 0 : compiler_qk_engines.size()))) return rc;
    if (tpc_only) return 0;
    for (unsigned h = 0; h < 2; ++h) {
        uint64_t fence = h ? mme2_fence_address : mme0_fence_address;
        uint64_t qk_fence = batch_split_fence ? (h ? mme2_fence2_address : mme0_fence2_address) : fence;
        unsigned prior = b >= 2 ? p.sob + 3 * (b - 2) + 2 : p.sob + 3 * b;
        if ((rc = arm_private_monitor(mon + 24 + h, prior, b >= 2 ? 2 : 0, qk_fence)) ||
            (rc = arm_private_monitor(mon + 26 + h, p.sob + 3 * b + 1, mme_only ? 0 : p.tpcs, fence))) return rc;
    }
    return 0;
}

static int batch_arm_final(const BatchPlan &p)
{
    int rc;
    if ((rc = arm_tpc_monitors(p.monitor + 28 * p.n, p.tpcs, p.sob + 3 * (p.n - 1) + 2, batch_ablation == 1 ? 0 : 2))) return rc;
    return 0;
}

extern "C" int context_batch_submit(unsigned slot)
{
    if (slot > 1 || batch_pending) return -EINVAL;
    const auto &p = batch_plans[slot];
    if (!p.ready) return -EINVAL;
    own_begin();
    int rc;
    const bool mme = batch_ablation != 1;
    for (unsigned i = 0; i < 3 * p.n; ++i) {
        if ((rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + (p.sob + i) * 4, 0))) return rc;
    }
    if (!batch_interleave_arm) {
        for (unsigned b = 0; b < p.n; ++b)
            if ((rc = batch_arm_block(p, b))) return rc;
        if ((rc = batch_arm_final(p))) return rc;
        if ((batch_ablation != 2 && (rc = queue_ecb(p.ex))) || (mme && (rc = queue_ecb(p.qk[0])))) return rc;
        for (unsigned b = 1; b < p.n && mme; ++b)
            if ((rc = queue_ecb(p.qk[b])) || (rc = queue_ecb(p.pv[b - 1]))) return rc;
        if ((mme && (rc = queue_ecb(p.pv.back()))) || (rc = queue_ecb(p.nm))) return rc;
    } else {
        // Same ECB order; block b is armed before its QK(b) and after PV(b-2)
        // was queued, i.e. before any engine can reach a fence for block b.
        if ((rc = batch_arm_block(p, 0)) || (batch_ablation != 2 && (rc = queue_ecb(p.ex))) ||
            (mme && (rc = queue_ecb(p.qk[0])))) return rc;
        for (unsigned b = 1; b < p.n; ++b) {
            if ((rc = batch_arm_block(p, b))) return rc;
            if (mme && ((rc = queue_ecb(p.qk[b])) || (rc = queue_ecb(p.pv[b - 1])))) return rc;
        }
        if ((rc = batch_arm_final(p))) return rc;
        if ((mme && (rc = queue_ecb(p.pv.back()))) || (rc = queue_ecb(p.nm))) return rc;
    }
    __sync_synchronize();
    rc = own_submit();
    if (!rc) batch_pending = true;
    return rc;
}

extern "C" int context_batch_wait()
{
    if (!batch_pending) return -EINVAL;
    batch_pending = false;
    return scal_host_fence_counter_wait(own_fence, batch_plans[0].tpcs, 2000000);
}

extern "C" int context_batch_run()
{
    int rc = context_batch_submit(0);
    return rc ? rc : context_batch_wait();
}

// Asynchronous copies on the batch stream; context_batch_dma_sync waits for all.
extern "C" int context_batch_dma_enqueue(unsigned n, const uint64_t *src, const uint64_t *dst, const uint64_t *bytes)
{
    if (!n || n > 8 || !src || !dst || !bytes) return -EINVAL;
    if (!batch_dma_stream && synStreamCreateGeneric(&batch_dma_stream, 0, 0) != synSuccess) return -EIO;
    for (unsigned i = 0; i < n; ++i) {
        if (!bytes[i]) return -EINVAL;
        synStatus rc = synMemCopyAsync(batch_dma_stream, src[i], bytes[i], dst[i], DRAM_TO_DRAM);
        if (rc != synSuccess) return (int)rc;
    }
    return 0;
}

extern "C" int context_batch_dma_sync()
{
    return batch_dma_stream ? (int)synStreamSynchronize(batch_dma_stream) : 0;
}

extern "C" int context_batch_dma(unsigned n, const uint64_t *src, const uint64_t *dst, const uint64_t *bytes)
{
    if (!n || n > 8 || !src || !dst || !bytes) return -EINVAL;
    if (!batch_dma_stream && synStreamCreateGeneric(&batch_dma_stream, 0, 0) != synSuccess) return -EIO;
    for (unsigned i = 0; i < n; ++i) {
        if (!bytes[i]) return -EINVAL;
        synStatus rc = synMemCopyAsync(batch_dma_stream, src[i], bytes[i], dst[i], DRAM_TO_DRAM);
        if (rc != synSuccess) return (int)rc;
    }
    return (int)synStreamSynchronize(batch_dma_stream);
}
