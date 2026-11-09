// SPDX-License-Identifier: MIT
// Experimental 512/1024 queries x four key chunks, SRAM S/P double buffer.
// Public n counts 128-query blocks (multiple of four, <=16). Each native unit
// processes 512 queries and one key quarter. All four partial outputs/stats
// survive; normalize_quad merges them. No single-owner PV shortcut here.
struct QuadPlan {
    std::vector<PreparedEcb> qk, pv, mme_batches;
    PreparedEcb ex, nm;
    unsigned n = 0, tpcs = 0, sob = 0, monitor = 0, keys = 0;
    bool ready = false;
};
// Slot 1 (opt-in): separate Q/O/stats SRAM region for host DMA overlap;
// --cross-group also assigns distinct synchronization pools to both slots.
static QuadPlan quad_plans[2];
static QuadPlan &quad_plan = quad_plans[0];
static int quad_interleave_arm = 0, quad_split_fence = 0, quad_ablation = 0, quad_prefetch_desc = 0, quad_mme_nobarrier = 0;
extern "C" int context_quad_set_mme_nobarrier(int on)
{
    if (quad_plan.ready || (on && !g_mme_nobarrier_allowed)) return -EINVAL;
    quad_mme_nobarrier = on != 0;
    return 0;
}
extern "C" int context_quad_set_prefetch_desc(int on)
{
    if (quad_plan.ready) return -EINVAL;
    quad_prefetch_desc = on != 0;
    return 0;
}
// 2 = MME only (profile-only): no exp ECB/monitors, PV credits fire at arm.
// 1 = TPC only (profile-only): no MME ECB and no monitor targets an MME fence;
//     exp and NM fences are released at arm time.
extern "C" int context_quad_set_ablation(int ablation)
{
    if (quad_plan.ready || ablation < 0 || ablation > 2) return -EINVAL;
    quad_ablation = ablation;
    return 0;
}
extern "C" int context_quad_set_split_fence(int split)
{
    if (quad_plan.ready) return -EINVAL;
    quad_split_fence = split != 0;
    return 0;
}
// Pack consecutive MME operations in one static CQ program (same operation
// order, same descriptor words and fences). Cuts repeated scheduler/ARC dispatch.
static unsigned quad_mme_pack = 0;
extern "C" int context_quad_set_mme_pack(unsigned units)
{
    if (quad_plan.ready || (units != 0 && units != 1 && units != 2 && units != 4)) return -EINVAL;
    quad_mme_pack = units;
    return 0;
}
static bool quad_pending = false;
// Cross-group pipelining (opt-in, needs --split-fence and interleaved arming):
// group g+1 (other slot) is submitted while group g still runs. Slot 1 uses its
// own SOB (+64) and monitor (+512) ranges. QK(0)/QK(1) of a group with a
// predecessor wait for the predecessor's PV(n-2)/PV(n-1) (same S buffers)
// instead of firing at arm time, so fence2 credits stay in fence order; fence3
// (exp->PV, TPC exp/NM) credits are ordered by in-order engine execution.
static int quad_cross_group = 0;
static unsigned quad_inflight = 0, quad_starts[2], quad_head = 0;
extern "C" int context_quad_set_cross_group(int on)
{
    if (quad_plan.ready || (on && (!quad_split_fence || !quad_interleave_arm))) return -EINVAL;
    quad_cross_group = on != 0;
    return 0;
}
extern "C" int context_quad_set_options(int interleave)
{
    if (quad_plan.ready) return -EINVAL;
    quad_interleave_arm = interleave != 0;
    return 0;
}
// Isolated experiment only: rely on FIFO MME execution for same-engine S
// reuse. The row ownership proof still applies; full-head replay must pass.
static bool quad_relaxed_qk_reuse = false;
extern "C" int context_quad_set_relaxed_qk_reuse(int on)
{
    if (quad_plan.ready) return -EINVAL;
    quad_relaxed_qk_reuse=on!=0;return 0;
}
static bool quad_local_qk_barrier = false;
extern "C" int context_quad_set_local_qk_barrier(int on)
{
    if (quad_plan.ready) return -EINVAL;
    quad_local_qk_barrier=on!=0;return 0;
}
// Exact zero padding: FP16(-88.0625)*FP16(184.625) rounds to -16256,
// index zero has LUT correction +16256; resulting BF16 bits are exactly zero.
static uint32_t quad_exp_padding=0xd581d581u;
extern "C" int context_quad_set_exp_shift(unsigned bits)
{
    if (quad_plan.ready || (bits!=0 && bits!=8)) return -EINVAL;
    quad_exp_padding=bits==8?0xd528d528u:0xd581d581u;return 0;
}
// Power-of-two Q scaling variant: Q is multiplied by exactly 1/16 and the
// exp multiplier is 261.25. Requires shift8 LUT. FP16(-58.3125)*261.25
// rounds to -15232, so LUT index zero (+15232) gives exact zero padding.
extern "C" int context_quad_set_exp_pow2_q(unsigned shift)
{
    if (quad_plan.ready || shift!=8) return -EINVAL;
    quad_exp_padding=0xd34ad34au;return 0;
}
static bool quad_exp_pad = false;
extern "C" int context_quad_set_exp_pad(int on)
{
    if (quad_plan.ready) return -EINVAL;
    quad_exp_pad=on!=0;return 0;
}
static unsigned quad_query_rows = 512;
static uint64_t quad_hbm_kt=0,quad_hbm_v=0;
extern "C" int context_quad_set_wide_kv(uint64_t kt,uint64_t v)
{
    if (quad_plan.ready || !kt || !v || kt%2 || v%2) return -EINVAL;
    quad_hbm_kt=kt;quad_hbm_v=v;quad_query_rows=1024;return 0;
}
static unsigned quad_row_pad = 0;
extern "C" int context_quad_set_row_pad(unsigned pad)
{
    if (quad_plan.ready || (pad!=0 && pad!=128 && pad!=384 && pad!=640)) return -EINVAL;
    quad_row_pad=pad;return 0;
}
static unsigned quad_key_quantum = 128;
extern "C" int context_quad_set_key_quantum(unsigned quantum)
{
    if (quad_plan.ready || (quantum != 128 && quantum != 512)) return -EINVAL;
    quad_key_quantum=quantum;return 0;
}
static unsigned quad_first_width(unsigned keys)
{
    return (keys+4*quad_key_quantum-1)/(4*quad_key_quantum)*quad_key_quantum;
}
// Reuse the validated MME-set container and loader from runtime_tiled.hpp.
static TileMmeSet quad_mme[4]; // QK first/last width, PV first/last width
extern "C" int context_quad_set_mme(unsigned slot, unsigned n, const uint32_t *words, const unsigned *engines)
{
    if (quad_plan.ready || slot > 3 || !words || !engines || !n || n > 8) return -EINVAL;
    unsigned counts[2] = {};
    for (unsigned i=0; i<n; ++i) {
        if (engines[i] > 1) return -EINVAL;
        ++counts[engines[i]];
        unsigned h=words[i*182+(0x30-8)/4];
        if (((h>>24)&15)!=9 || (h>>28)!=(slot<2?8u:9u)) return -EINVAL;
    }
    if (!counts[0] || counts[0]!=counts[1] || (slot>=2 && n!=2)) return -EINVAL;
    quad_mme[slot].words.assign(words,words+n*182);
    quad_mme[slot].engines.assign(engines,engines+n);
    return 0;
}

extern "C" int context_quad_layout(uint64_t base, unsigned keys, unsigned n, uint64_t *out)
{
    if (!out || base % 256 || keys < 512 || keys > 37760 || keys % 128 || n < 4 || n > (quad_query_rows==1024?24u:16u) || n % (quad_query_rows/128)) return -EINVAL;
    uint64_t a = base;
    auto take = [&](uint64_t bytes) { uint64_t r = a; a = (a + bytes + 255) & ~255ull; return r; };
    uint64_t matrix = 128 * 128 * 2, wide = 128 * keys * 2;
    // q, kt, v, S0, S1, OA, OB, stats, guard, end
    out[0] = take(n * matrix); out[1] = quad_hbm_kt?quad_hbm_kt:take(wide); out[2] = quad_hbm_v?quad_hbm_v:take(wide);
    unsigned width = quad_first_width(keys);
    out[3] = take((uint64_t)quad_query_rows * (width+quad_row_pad) * 2); out[4] = take((uint64_t)quad_query_rows * (width+quad_row_pad) * 2);
    out[5] = take(4 * n * matrix); out[6] = out[5] + n * matrix;
    out[7] = take(n * 4096);
    out[8] = take(4096); out[9] = a;
    return 0;
}

// Second set after slot 0's end: {q, O (4 partials), O+n*matrix, stats, end}.
extern "C" int context_quad_layout_alt(uint64_t base, unsigned keys, unsigned n, uint64_t *out)
{
    uint64_t L[10];
    int rc = context_quad_layout(base, keys, n, L); if (rc || !out) return rc ? rc : -EINVAL;
    uint64_t a = L[9];
    auto take = [&](uint64_t bytes) { uint64_t r = a; a = (a + bytes + 255) & ~255ull; return r; };
    const uint64_t matrix = 128 * 128 * 2;
    out[0] = take(n * matrix); out[1] = take(4 * n * matrix); out[2] = out[1] + n * matrix;
    out[3] = take(n * 4096); out[4] = a;
    return 0;
}

// Fixed M512 captures: QK output and PV input of engine e only touch rows
// whose parity is e. Include the tensor startOffset (engine 1 starts at row 1).
// Width is a multiple of 128, so the reduction never spills into the next row.
// This permits a local MME engine barrier in place of a round trip through an
// all-MME PV SOB when reusing S. Reject any unproved row geometry.
static bool quad_disjoint_rows()
{
    auto get=[](const uint32_t *w,unsigned off) { return w[(off-8)/4]; };
    for (unsigned slot=0; slot<4; ++slot) {
        const auto &set=quad_mme[slot];
        for (unsigned j=0; j<set.engines.size(); ++j) {
            const uint32_t *w=set.words.data()+182*j;
            unsigned e=set.engines[j],ten=slot<2?0xf0:0x40,width=get(w,ten);
            if (!width || width%128 || get(w,ten+4)!=quad_query_rows*(width+quad_row_pad) || get(w,0x290)!=0) return false;
            width+=quad_row_pad;
            unsigned start=get(w,ten+18*4),stride=get(w,ten+14*4);
            if (start%width || (start/width)%2!=e || stride%(2*width)) return false;
            if (slot<2) {
                if (get(w,0x40)!=128 || get(w,0x44)!=quad_query_rows*128 || (get(w,0x280)&~255u)!=(quad_query_rows/512-1)*256) return false;
                for (unsigned k=0; k<4; ++k)
                    if (get(w,0x22c+k*20+4)%(2*width)) return false;
            } else {
                if (get(w,0x280)!=(quad_query_rows/512-1)*256 || get(w,0xf0)!=128) return false;
                for (unsigned port : {0u,4u}) for (unsigned slave=0; slave<2; ++slave) {
                    unsigned off=0x15c+port*40+slave*20;
                    if (get(w,off)!=0 || get(w,off+4)%(2*width)) return false;
                }
            }
        }
    }
    return true;
}

extern "C" int context_quad_prepare_slot(unsigned slot, uint64_t base, unsigned keys, unsigned valid, unsigned n,
                                          unsigned tpcs, const char *exp_path, const char *norm_path)
{
    if (quad_relaxed_qk_reuse && !quad_local_qk_barrier) return -EINVAL;
    if (quad_local_qk_barrier && (!quad_split_fence || !quad_mme_nobarrier || !g_mme_async_commit ||
        quad_prefetch_desc || !quad_disjoint_rows())) return -EINVAL;
    if (quad_mme_pack && (!quad_interleave_arm || !quad_split_fence || n % quad_mme_pack)) return -EINVAL;
    if (slot > 1 || (slot == 1 && (!quad_plans[0].ready || quad_plans[0].n != n/(quad_query_rows/512) || quad_plans[0].tpcs != tpcs)))
        return -EINVAL;
    auto &p = quad_plans[slot];
    if (p.ready || batch_plan.ready || tile_plan.ready || pipe_plan.ready || attention_plan.ready ||
        quad_mme[0].words.empty() || quad_mme[1].words.empty() ||
        quad_mme[2].words.empty() || quad_mme[3].words.empty() ||
        !g_wreg_bulk || !g_lut_enabled || !tpcs || tpcs > 24 || valid > keys ||
        valid <= 3 * (quad_first_width(keys))) return -EINVAL;
    uint64_t L[10];
    int rc = context_quad_layout(base, keys, n, L); if (rc) return rc;
    uint64_t end = L[9];
    if (slot == 1) {
        uint64_t A[5];
        if ((rc = context_quad_layout_alt(base, keys, n, A))) return rc;
        L[0] = A[0]; L[5] = A[1]; L[6] = A[2]; L[7] = A[3]; end = A[4];
    }
    synDeviceInfoV2 info{};
    if (synDeviceGetInfoV2(0, &info) != synSuccess || base < info.sramBaseAddress + 0x200000 ||
        end > info.sramBaseAddress + info.sramSize) return -EINVAL;
    const unsigned factor=quad_query_rows/512;
    n/=factor; // native units; public layout/output count remains n*factor
    if ((rc = private_sync_resources(p.sob, p.monitor))) return rc;
    if (private_so_count < 3 * n || private_monitor_count < 28 * n + tpcs) return -ENOSPC;
    if (quad_cross_group) {
        if (3 * n > 64 || 28 * n + tpcs > 512 || private_so_count < 64 + 3 * n ||
            private_monitor_count < 512 + 28 * n + tpcs) return -ENOSPC;
        if (slot == 1) { p.sob += 64; p.monitor += 512; }
    }
    uint64_t exp = load_runtime_isa(exp_path), norm = load_runtime_isa(norm_path);
    if (!exp || !norm) return -EINVAL;
    p.n = n; p.tpcs = tpcs;p.keys=keys; p.qk.resize(n); p.pv.resize(n);
    const uint64_t matrix = 128 * 128 * 2;
    const unsigned first = quad_first_width(keys), last = keys - 3 * first;
    std::vector<std::vector<char>> ex(24), nm(24);
    std::vector<std::vector<std::vector<char>>> qraw(n), vraw(n);
    std::vector<std::vector<uint32_t>> cached(24);
    for (unsigned b = 0; b < n; ++b) {
        const unsigned part = b % 4, slot = part == 3 ? 1 : 0;
        unsigned width = slot ? last : first, local_valid = slot ? valid - 3 * first : first;
        const uint64_t S = L[3 + (b & 1)], Q = L[0] + (b / 4) * 4 * matrix * factor;
        const uint64_t KT = L[1] + (uint64_t)part * first * 128 * 2;
        const uint64_t V = L[2] + (uint64_t)part * first * 128 * 2;
        const uint64_t O = L[5] + part * n * matrix * factor + (b / 4) * 4 * matrix * factor;
        std::vector<std::vector<char>> qk(2), pv(2);
        const unsigned qk_fence_id = quad_split_fence ? 2 : 3;
        const bool local = quad_local_qk_barrier && b>=2;
        bool first_qk[2]={true,true};
        if (quad_mme_nobarrier) g_push_eng_barrier = 0;
        bool qf[2] = {false, false}, pf[2] = {false, false};
        if (!quad_prefetch_desc)
            for (unsigned e=0; e<2; ++e) {
                if (!local) append_private_fence(qk[e],qk_fence_id);
                append_private_fence(pv[e]);qf[e]=pf[e]=true;
            }
        const auto &qs=quad_mme[slot]; const auto &ps=quad_mme[slot+2];
        for (unsigned j=0; j<qs.engines.size(); ++j) {
            unsigned e = qs.engines[j];
            if (!qf[e]) { g_mme_prefence_id = qk_fence_id; qf[e] = true; }
            // This engine's immediately preceding operation is PV(b-2).
            // Its engine barrier guarantees that its own S rows were consumed;
            // the other MME only reads the opposite row parity (validated above).
            if (local && first_qk[e] && !quad_relaxed_qk_reuse) g_push_eng_barrier=1;
            append_compiler_mme(qk[e],qs.words.data()+j*182,Q,KT,S,p.sob+3*b);
            if (local) g_push_eng_barrier=0;
            first_qk[e]=false;
        }
        for (unsigned j=0; j<ps.engines.size(); ++j) {
            unsigned e = ps.engines[j];
            if (!pf[e]) { g_mme_prefence_id = 3; pf[e] = true; }
            append_compiler_mme(pv[e],ps.words.data()+j*182,S,V,O,p.sob+3*b+2);
        }
        g_mme_prefence_id = -1;
        for (unsigned e=0; e<2; ++e) {
            if (!qf[e]) append_private_fence(qk[e], qk_fence_id);
            if (!pf[e]) append_private_fence(pv[e]);
        }
        if (quad_mme_nobarrier) g_finish_eng_barrier = 0;
        for (unsigned e=0; e<2; ++e) { finish_program(qk[e]); finish_program(pv[e]); }
        g_push_eng_barrier = 1; g_finish_eng_barrier = 1;
        if (quad_mme_pack) { qraw[b].swap(qk); vraw[b].swap(pv); }
        else if ((rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, qk, p.qk[b])) ||
                 (rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, pv, p.pv[b]))) return rc;
        uint64_t addresses[3] = {L[7] + b * 4096 * factor, S, S};
        unsigned cols[3] = {2, quad_exp_pad?local_valid:width, quad_exp_pad?local_valid:width}, rows[3] = {quad_query_rows, quad_query_rows, quad_query_rows};
        const uint32_t padding[3]={0,quad_exp_padding,quad_exp_padding};
        unsigned types[3] = {7, 6, 6}, strides[3] = {2, width+quad_row_pad, width+quad_row_pad};
        for (unsigned i = 0; i < tpcs; ++i) {
            if (!b) push_lut_bases(ex[i]);
            unsigned unit = quad_query_rows * i / tpcs, units = quad_query_rows * (i + 1) / tpcs - unit;
            rc = append_batch_exp(ex[i], cached[i], exp, addresses, cols, rows, types, strides,
                                   local_valid, unit, units, p.sob + 3 * b + 1, quad_exp_pad?padding:nullptr);
            if (rc) return rc;
        }
    }
    if (quad_mme_pack) {
        // Global order is Q0,Q1,PV0,Q2,PV1,...,Qlast,PVlast-1,PVlast.
        // Preserve this across pack boundaries (not separate mini-pipelines).
        std::vector<const std::vector<std::vector<char>> *> operations;
        operations.push_back(&qraw[0]);
        for (unsigned b=1; b<n; ++b) { operations.push_back(&qraw[b]); operations.push_back(&vraw[b-1]); }
        operations.push_back(&vraw[n-1]);
        unsigned count = 2 * quad_mme_pack;
        p.mme_batches.resize(n/quad_mme_pack);
        for (unsigned start=0; start<operations.size(); start+=count) {
            std::vector<std::vector<char>> raw(2);
            for (unsigned op=start; op<start+count; ++op) for (unsigned e=0; e<2; ++e) {
                const auto &part = (*operations[op])[e];
                if (part.size()<8 || ((const packet_nop *)(part.data()+part.size()-8))->opcode!=PACKET_NOP) return -EINVAL;
                raw[e].insert(raw[e].end(),part.begin(),part.end()-8); // remove CQ switch only
            }
            for (auto &program : raw) finish_program(program);
            if ((rc=prepare_ecb(SCAL_MME_COMPUTE_GROUP,raw,p.mme_batches[start/count]))) return rc;
        }
        printf("MME_PACK slot=%u units=%u jobs=%zu original_jobs=%u\n",slot,quad_mme_pack,p.mme_batches.size(),2*n);fflush(stdout);
    }
    uint64_t addresses[3] = {L[5], L[7], L[5]};
    unsigned cols[3] = {128, 2, 128}, rows[3] = {quad_query_rows * n, quad_query_rows * n, 128 * n * factor};
    unsigned types[3] = {6, 7, 6};
    for (unsigned i = 0; i < 24; ++i) {
        if (i < tpcs) {
            unsigned row = 128 * n * factor * i / tpcs, count = 128 * n * factor * (i + 1) / tpcs - row;
            rc = append_tpc_fenced(nm[i], norm, addresses, cols, rows, types, 3, valid, row, count, own_sob);
            if (rc) return rc;
        }
        finish_program(ex[i]); finish_program(nm[i]);
    }
    if ((rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, ex, p.ex)) ||
        (rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, nm, p.nm))) return rc;
    p.ready = true;
    printf("QUAD512_PLAN slot=%u units=%u keys=%u tpcs=%u all_outputs=1 q=0x%lx end=0x%lx\n",
           slot, n, keys, tpcs, L[0], end); fflush(stdout);
    return 0;
}

extern "C" int context_quad_prepare(uint64_t base, unsigned keys, unsigned valid, unsigned n,
                                     unsigned tpcs, const char *exp_path, const char *norm_path)
{
    return context_quad_prepare_slot(0, base, keys, valid, n, tpcs, exp_path, norm_path);
}

static int quad_arm_block(const QuadPlan &p, unsigned b, const QuadPlan *prev = nullptr)
{
    int rc;
    {
        unsigned mon = p.monitor + 28 * b;
        if (quad_ablation != 2 &&
            (rc = arm_tpc_monitors(mon, p.tpcs, p.sob + 3 * b,
                                   quad_ablation == 1 ? 0 : quad_mme[b % 4 == 3 ? 1 : 0].engines.size()))) return rc;
        for (unsigned h = 0; h < 2 && quad_ablation != 1; ++h) {
            uint64_t fence = h ? mme2_fence_address : mme0_fence_address;
            uint64_t qk_fence = quad_split_fence ? (h ? mme2_fence2_address : mme0_fence2_address) : fence;
            unsigned prior = b >= 2 ? p.sob + 3 * (b - 2) + 2 : p.sob + 3 * b;
            unsigned target = b >= 2 ? 2 : 0;
            if (b < 2 && prev) { prior = prev->sob + 3 * (prev->n - 2 + b) + 2; target = 2; }
            if ((!quad_local_qk_barrier || b<2) &&
                (rc=arm_private_monitor(mon+24+h,prior,target,qk_fence))) return rc;
            if ((rc = arm_private_monitor(mon + 26 + h, p.sob + 3 * b + 1, quad_ablation == 2 ? 0 : p.tpcs, fence))) return rc;
        }
    }
    return 0;
}

static int quad_arm_final(const QuadPlan &p)
{
    int rc;
    if ((rc = arm_tpc_monitors(p.monitor + 28 * p.n, p.tpcs, p.sob + 3 * (p.n - 1) + 2, quad_ablation == 1 ? 0 : 2))) return rc;
    return 0;
}

extern "C" int context_quad_submit2(unsigned slot, int has_prev)
{
    if (slot > 1) return -EINVAL;
    if (quad_cross_group ? quad_inflight >= 2 : quad_pending) return -EINVAL;
    if (has_prev && (!quad_cross_group || !quad_plans[slot ^ 1].ready)) return -EINVAL;
    const auto &p = quad_plans[slot];
    const QuadPlan *prev = has_prev ? &quad_plans[slot ^ 1] : nullptr;
    if (!p.ready) return -EINVAL;
    const unsigned start = own_pi;
    own_begin();
    // Two runs in flight: the ring budget must cover both unconsumed runs.
    if (quad_cross_group && quad_inflight) own_run_start = quad_starts[quad_head];
    int rc;
    for (unsigned i = 0; i < 3 * p.n; ++i) {
        if ((rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + (p.sob + i) * 4, 0))) return rc;
    }
    if (quad_mme_pack) {
        const bool mme=quad_ablation!=1;
        if ((rc=quad_arm_block(p,0,prev)) || (quad_ablation!=2 && (rc=queue_ecb(p.ex))) ||
            (mme && (rc=queue_ecb(p.mme_batches[0])))) return rc;
        for (unsigned b=1; b<p.n; ++b) {
            if ((rc=quad_arm_block(p,b,prev))) return rc;
            // Batch k begins with an older PV then Q(k*pack+1): enqueue while
            // this block is armed; later fences wait for their own credits.
            if (mme && b%quad_mme_pack==0 && (rc=queue_ecb(p.mme_batches[b/quad_mme_pack]))) return rc;
        }
        if ((rc=quad_arm_final(p)) || (rc=queue_ecb(p.nm))) return rc;
    } else     if (quad_interleave_arm) {
        // Same ECB order; block b armed just before QK(b).
        const bool mme = quad_ablation != 1;
        if ((rc = quad_arm_block(p, 0, prev)) || (quad_ablation != 2 && (rc = queue_ecb(p.ex))) || (mme && (rc = queue_ecb(p.qk[0])))) return rc;
        for (unsigned b = 1; b < p.n; ++b)
            if ((rc = quad_arm_block(p, b, prev)) || (mme && ((rc = queue_ecb(p.qk[b])) || (rc = queue_ecb(p.pv[b - 1]))))) return rc;
        if ((rc = quad_arm_final(p)) || (mme && (rc = queue_ecb(p.pv.back()))) || (rc = queue_ecb(p.nm))) return rc;
    } else {
        for (unsigned b = 0; b < p.n; ++b)
            if ((rc = quad_arm_block(p, b))) return rc;
        if ((rc = quad_arm_final(p))) return rc;
        if ((quad_ablation != 2 && (rc = queue_ecb(p.ex))) || (rc = queue_ecb(p.qk[0]))) return rc;
        for (unsigned b = 1; b < p.n; ++b)
            if ((rc = queue_ecb(p.qk[b])) || (rc = queue_ecb(p.pv[b - 1]))) return rc;
        if ((rc = queue_ecb(p.pv.back())) || (rc = queue_ecb(p.nm))) return rc;
    }
    __sync_synchronize();
    rc = own_submit();
    if (!rc) {
        quad_pending = true;
        quad_starts[(quad_head + quad_inflight) & 1] = start;
        ++quad_inflight;
    }
    return rc;
}

extern "C" int context_quad_submit(unsigned slot)
{
    return context_quad_submit2(slot, 0);
}

extern "C" int context_quad_wait()
{
    if (!quad_inflight) return -EINVAL;
    --quad_inflight; quad_head ^= 1;
    quad_pending = quad_inflight != 0;
    return scal_host_fence_counter_wait(own_fence, quad_plans[0].tpcs, 2000000);
}

extern "C" int context_quad_run()
{
    int rc = context_quad_submit(0);
    return rc ? rc : context_quad_wait();
}


// Diagnostic for framework/raw interop; does not submit an ECB.
extern "C" int context_quad_audit_prepared()
{
    for (const auto &p:quad_plans) {
        if (!p.ready) continue;
        int rc;
        for (const auto &j:p.qk) if ((rc=audit_prepared_ecb(j))) return rc;
        for (const auto &j:p.pv) if ((rc=audit_prepared_ecb(j))) return rc;
        if ((rc=audit_prepared_ecb(p.ex)) || (rc=audit_prepared_ecb(p.nm))) return rc;
    }
    return 0;
}

// Heads run sequentially and share the SRAM/Q/O/TPC plans. Cache the MME B
// bindings once, so no framework copies or descriptor uploads occur per head.
struct QuadHeadBank { std::vector<PreparedEcb> qk[2],pv[2]; };
static std::vector<QuadHeadBank> quad_head_banks;
static unsigned quad_active_head=0;
extern "C" int context_quad_prepare_head_banks(unsigned heads)
{
    if (!heads || heads>64 || !quad_head_banks.empty() || quad_pending || quad_inflight ||
        !quad_plans[0].ready || !quad_plans[1].ready || quad_query_rows!=1024 || quad_mme_pack) return -EINVAL;
    const uint64_t stride=(uint64_t)quad_plan.keys*128*2;
    quad_head_banks.resize(heads);
    for (unsigned h=1;h<heads;++h) for (unsigned slot=0;slot<2;++slot) {
        const auto &p=quad_plans[slot];auto &bank=quad_head_banks[h];
        bank.qk[slot].resize(p.n);bank.pv[slot].resize(p.n);
        for (unsigned b=0;b<p.n;++b) {
            int rc=clone_mme_b(p.qk[b],quad_hbm_kt,quad_hbm_kt+h*stride,stride,bank.qk[slot][b]);
            if (!rc) rc=clone_mme_b(p.pv[b],quad_hbm_v,quad_hbm_v+h*stride,stride,bank.pv[slot][b]);
            if (rc) { fprintf(stderr,"HEAD_BANK_ERROR head=%u slot=%u unit=%u rc=%d\n",h,slot,b,rc);return rc; }
        }
    }
    printf("QUAD_HEAD_BANKS heads=%u native_units=%u stride=%lu verified=1\n",heads,quad_plan.n,stride);fflush(stdout);
    return 0;
}
// Opt-in continuous head streaming: same host-side swap while groups are in
// flight. Already-queued commands hold device addresses by value (queue_ecb),
// and every PreparedEcb stays owned by a bank or the plan, so nothing queued is
// freed or rewritten. Caller is the single submitting thread. S reuse across a
// head boundary uses the normal cross-group predecessor fences (has_prev=1).
extern "C" int context_quad_select_head_any(unsigned head)
{
    if (head>=quad_head_banks.size()) return -EINVAL;
    if (head==quad_active_head) return 0;
    for (unsigned slot=0;slot<2;++slot) {
        auto &p=quad_plans[slot];auto &old=quad_head_banks[quad_active_head];auto &next=quad_head_banks[head];
        p.qk.swap(old.qk[slot]);p.pv.swap(old.pv[slot]);
        p.qk.swap(next.qk[slot]);p.pv.swap(next.pv[slot]);
    }
    quad_active_head=head;return 0;
}
extern "C" int context_quad_select_head(unsigned head)
{
    if (head>=quad_head_banks.size() || quad_pending || quad_inflight) return -EINVAL;
    if (head==quad_active_head) return 0;
    for (unsigned slot=0;slot<2;++slot) {
        auto &p=quad_plans[slot];auto &old=quad_head_banks[quad_active_head];auto &next=quad_head_banks[head];
        p.qk.swap(old.qk[slot]);p.pv.swap(old.pv[slot]);
        p.qk.swap(next.qk[slot]);p.pv.swap(next.pv[slot]);
    }
    quad_active_head=head;return 0;
}
