// SPDX-License-Identifier: MIT
// Experimental M=256 query tiles, alternating key halves. Each native unit
// processes 256 Q x one key half; two units share Q and merge after all PVs.
// n retains the public 128-query-block count (even, <=20). S/P stays in SRAM.
// Q/K/V layout is fixed; KT is packed [128,halfA] followed by [128,halfB].
struct TilePlan {
    std::vector<PreparedEcb> qk, pv;
    PreparedEcb ex, nm;
    unsigned n = 0, tpcs = 0, sob = 0, monitor = 0;
    bool ready = false;
};
// Slot 1 (opt-in): same plan, separate Q/OA/OB/stats SRAM region, for host
// DMA overlap. Both slots share SOBs/monitors; one slot is submitted at a time.
static TilePlan tile_plans[2];
static TilePlan &tile_plan = tile_plans[0];
static bool tile_parallel_pv = false;
static int tile_interleave_arm = 0;
static bool tile_pending = false;
extern "C" int context_tile_set_options(int interleave)
{
    if (tile_plan.ready || (interleave && tile_parallel_pv)) return -EINVAL;
    tile_interleave_arm = interleave != 0;
    return 0;
}
extern "C" int context_tile_set_parallel_pv(int enabled)
{
    if (tile_plan.ready) return -EBUSY;
    tile_parallel_pv = enabled != 0;
    return 0;
}
struct TileMmeSet {
    std::vector<uint32_t> words;
    std::vector<unsigned> engines;
};
static TileMmeSet tile_mme[4]; // QK A/B, PV A/B; pinned M=256 compiler captures
extern "C" int context_tile_set_mme(unsigned slot, unsigned n, const uint32_t *words, const unsigned *engines)
{
    if (tile_plan.ready || slot > 3 || !words || !engines || !n || n > 8) return -EINVAL;
    unsigned count[2] = {};
    for (unsigned i = 0; i < n; ++i) {
        if (engines[i] > 1) return -EINVAL;
        ++count[engines[i]];
        unsigned header = words[i * 182 + (0x30 - 8) / 4];
        if (((header >> 24) & 15) != 9 || (header >> 28) != (slot < 2 ? 8u : 9u)) return -EINVAL;
    }
    if (!count[0] || count[0] != count[1] || (slot < 2 ? (n > 6 || n % 2) : n != 2)) return -EINVAL;
    tile_mme[slot].words.assign(words, words + n * 182);
    tile_mme[slot].engines.assign(engines, engines + n);
    return 0;
}

extern "C" int context_tile_layout(uint64_t base, unsigned keys, unsigned n, uint64_t *out)
{
    if (!out || base % 256 || keys < 256 || keys > 37760 || keys % 128 || n < 2 || n > 20 || n % 2) return -EINVAL;
    uint64_t a = base;
    auto take = [&](uint64_t bytes) { uint64_t r = a; a = (a + bytes + 255) & ~255ull; return r; };
    uint64_t matrix = 128 * 128 * 2, wide = 128 * keys * 2;
    // q, kt, v, S0, S1, OA, OB, stats, guard, end
    out[0] = take(n * matrix); out[1] = take(wide); out[2] = take(wide);
    unsigned first = (keys / 128 + 1) / 2 * 128;
    out[3] = take(256ull * first * 2); out[4] = take(256ull * (keys - first) * 2);
    out[5] = take(n * matrix); out[6] = take(n * matrix); out[7] = take(n * 2048);
    out[8] = take(4096); out[9] = a;
    return 0;
}

// Second Q/OA/OB/stats set after slot 0's end: {q, OA, OB, stats, end}.
extern "C" int context_tile_layout_alt(uint64_t base, unsigned keys, unsigned n, uint64_t *out)
{
    uint64_t L[10];
    int rc = context_tile_layout(base, keys, n, L); if (rc || !out) return rc ? rc : -EINVAL;
    uint64_t a = L[9];
    auto take = [&](uint64_t bytes) { uint64_t r = a; a = (a + bytes + 255) & ~255ull; return r; };
    const uint64_t matrix = 128 * 128 * 2;
    out[0] = take(n * matrix); out[1] = take(n * matrix); out[2] = take(n * matrix);
    out[3] = take(n * 2048); out[4] = a;
    return 0;
}

extern "C" int context_tile_prepare_slot(unsigned slot, uint64_t base, unsigned keys, unsigned valid, unsigned n,
                                          unsigned tpcs, const char *exp_path, const char *norm_path)
{
    if (slot > 1 || (slot == 1 && (!tile_plans[0].ready || tile_plans[0].n != n || tile_plans[0].tpcs != tpcs)))
        return -EINVAL;
    auto &p = tile_plans[slot];
    if (p.ready || batch_plan.ready || pipe_plan.ready || attention_plan.ready ||
        tile_mme[0].words.empty() || tile_mme[1].words.empty() ||
        tile_mme[2].words.empty() || tile_mme[3].words.empty() ||
        !g_wreg_bulk || !g_lut_enabled || !tpcs || tpcs > 24 || valid > keys ||
        valid <= (keys / 128 + 1) / 2 * 128) return -EINVAL;
    uint64_t L[10];
    int rc = context_tile_layout(base, keys, n, L); if (rc) return rc;
    uint64_t end = L[9];
    if (slot == 1) {
        uint64_t A[5];
        if ((rc = context_tile_layout_alt(base, keys, n, A))) return rc;
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
        unsigned h = b & 1;
        unsigned width = h ? second : first, local_valid = h ? valid - first : first;
        const uint64_t S = L[3 + h], Q = L[0] + (b / 2) * 2 * matrix;
        const uint64_t KT = L[1] + (h ? (uint64_t)first * 128 * 2 : 0);
        const uint64_t V = L[2] + (h ? (uint64_t)first * 128 * 2 : 0);
        const uint64_t O = L[5 + h] + (b / 2) * 2 * matrix;
        std::vector<std::vector<char>> qk(2), pv(2);
        for (unsigned e = 0; e < 2; ++e) {
            append_private_fence(qk[e]);
            if (!tile_parallel_pv || e == h) append_private_fence(pv[e]);
        }
        const auto &qs = tile_mme[h]; const auto &ps = tile_mme[h + 2];
        for (unsigned j = 0; j < qs.engines.size(); ++j)
            append_compiler_mme(qk[qs.engines[j]], qs.words.data() + j * 182, Q, KT, S, p.sob + 3 * b);
        for (unsigned j = 0; j < ps.engines.size(); ++j) {
            if (tile_parallel_pv && ps.engines[j] != 0) continue;
            // Capture engine 0 writes the complete 256x128 result; engine 1's
            // column AGU is >=256 while N=128. The isolated single-engine raw
            // tests establish that it contributes no output or cross-MME state.
            unsigned target = tile_parallel_pv ? h : ps.engines[j];
            // Both owners signal the pair's PV SOB. Future QK fences must wait
            // for BOTH PVs: early per-half credits could otherwise release the
            // other owner's older PV fence before that half's exp completed.
            unsigned pv_sob = p.sob + 3 * (tile_parallel_pv ? (b & ~1u) : b) + 2;
            append_compiler_mme(pv[target], ps.words.data() + j * 182, S, V, O, pv_sob);
        }
        for (unsigned e = 0; e < 2; ++e) { finish_program(qk[e]); finish_program(pv[e]); }
        if ((rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, qk, p.qk[b])) ||
            (rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, pv, p.pv[b]))) return rc;
        // Single 256-row score tile. Tensor 2 is an unused alias retained for the
        // three-tensor exp ABI; each half's stats occupy 256 x {max,sum}.
        uint64_t addresses[3] = {L[7] + b * 2048, S, S};
        unsigned cols[3] = {2, width, width}, rows[3] = {256, 256, 256};
        unsigned types[3] = {7, 6, 6}, strides[3] = {2, width, width};
        for (unsigned i = 0; i < tpcs; ++i) {
            if (!b) push_lut_bases(ex[i]);
            unsigned unit = 256 * i / tpcs, units = 256 * (i + 1) / tpcs - unit;
            rc = append_batch_exp(ex[i], cached[i], exp, addresses, cols, rows, types, strides,
                                   local_valid, unit, units, p.sob + 3 * b + 1);
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
    printf("TILE256_PLAN slot=%u half_tiles=%u keys=%u tpcs=%u all_outputs=1 parallel_pv=%d q=0x%lx end=0x%lx\n",
           slot, n, keys, tpcs, (int)tile_parallel_pv, L[0], end); fflush(stdout);
    return 0;
}

extern "C" int context_tile_prepare(uint64_t base, unsigned keys, unsigned valid, unsigned n,
                                     unsigned tpcs, const char *exp_path, const char *norm_path)
{
    return context_tile_prepare_slot(0, base, keys, valid, n, tpcs, exp_path, norm_path);
}

static int tile_arm_block(const TilePlan &p, unsigned b)
{
    int rc;
    {
        unsigned mon = p.monitor + 28 * b;
        for (unsigned i = 0; i < p.tpcs; ++i)
            if ((rc = arm_private_monitor(mon + i, p.sob + 3 * b, tile_mme[b & 1].engines.size(),
                                          tpc_static_fence_address(i)))) return rc;
        for (unsigned h = 0; h < 2; ++h) {
            uint64_t fence = h ? mme2_fence_address : mme0_fence_address;
            unsigned previous = tile_parallel_pv ? ((b - 2) & ~1u) : b - 2;
            unsigned prior = b >= 2 ? p.sob + 3 * previous + 2 : p.sob + 3 * b;
            if ((rc = arm_private_monitor(mon + 24 + h, prior,
                                          b >= 2 ? 2 : 0, fence))) return rc;
            if ((!tile_parallel_pv || h == (b & 1)) &&
                (rc = arm_private_monitor(mon + 26 + h, p.sob + 3 * b + 1, p.tpcs, fence))) return rc;
        }
    }
    return 0;
}

static int tile_arm_final(const TilePlan &p)
{
    int rc;
    for (unsigned i = 0; i < p.tpcs; ++i)
        if ((rc = arm_private_monitor(p.monitor + 28 * p.n + i,
                    p.sob + 3 * (p.n - (tile_parallel_pv ? 2 : 1)) + 2,
                    2, tpc_static_fence_address(i)))) return rc;
    return 0;
}

extern "C" int context_tile_submit(unsigned slot)
{
    if (slot > 1 || tile_pending) return -EINVAL;
    const auto &p = tile_plans[slot];
    if (!p.ready) return -EINVAL;
    own_begin();
    int rc;
    for (unsigned i = 0; i < 3 * p.n; ++i) {
        if ((rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + (p.sob + i) * 4, 0))) return rc;
    }
    if (tile_interleave_arm) {
        // Same ECB order as below; block b is armed just before QK(b).
        if ((rc = tile_arm_block(p, 0)) || (rc = queue_ecb(p.ex)) || (rc = queue_ecb(p.qk[0]))) return rc;
        for (unsigned b = 1; b < p.n; ++b)
            if ((rc = tile_arm_block(p, b)) || (rc = queue_ecb(p.qk[b])) || (rc = queue_ecb(p.pv[b - 1]))) return rc;
        if ((rc = tile_arm_final(p)) || (rc = queue_ecb(p.pv.back())) || (rc = queue_ecb(p.nm))) return rc;
        __sync_synchronize();
        rc = own_submit();
        if (!rc) tile_pending = true;
        return rc;
    }
    for (unsigned b = 0; b < p.n; ++b)
        if ((rc = tile_arm_block(p, b))) return rc;
    if ((rc = tile_arm_final(p))) return rc;
    if ((rc = queue_ecb(p.ex))) return rc;
    if (tile_parallel_pv) {
        // Per-pair QK A/B then two owner-only PV jobs. The other engine gets
        // only NOP, so its PV half can run independently while this half runs.
        for (unsigned b = 0; b < p.n; b += 2)
            if ((rc = queue_ecb(p.qk[b])) || (rc = queue_ecb(p.qk[b+1])) ||
                (rc = queue_ecb(p.pv[b])) || (rc = queue_ecb(p.pv[b+1]))) return rc;
    } else {
        if ((rc = queue_ecb(p.qk[0]))) return rc;
        for (unsigned b = 1; b < p.n; ++b)
            if ((rc = queue_ecb(p.qk[b])) || (rc = queue_ecb(p.pv[b - 1]))) return rc;
        if ((rc = queue_ecb(p.pv.back()))) return rc;
    }
    if ((rc = queue_ecb(p.nm))) return rc;
    __sync_synchronize();
    rc = own_submit();
    if (!rc) tile_pending = true;
    return rc;
}

extern "C" int context_tile_wait()
{
    if (!tile_pending) return -EINVAL;
    tile_pending = false;
    return scal_host_fence_counter_wait(own_fence, tile_plans[0].tpcs, 2000000);
}

extern "C" int context_tile_run()
{
    int rc = context_tile_submit(0);
    return rc ? rc : context_tile_wait();
}

