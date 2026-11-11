// SPDX-License-Identifier: MIT
// Included by context_bridge.cpp after its packet builders and SCAL context.
// All resources below belong to the isolated profile's private SM2 pools.
struct PreparedEcb {
    scal_buffer_handle_t metadata = nullptr, code = nullptr;
    scal_buffer_info_t meta_info{}, code_info{};
    unsigned group = 0;
    std::vector<char> code_image;
    std::vector<unsigned> program_offsets,program_sizes;
    std::vector<char> debug_code, debug_metadata; // opt-in interop audit only
};

// Optional TPC special-function tables (Gaudi2 LUT); see context_attention_set_lut.
static bool g_lut_enabled = false;
static uint64_t g_lut_address[4]; // tables for LUT_FUNC256/128/64/32_BASE_ADDR
static unsigned g_lut_mask = 1u << 1; // bit t: write table t's base (bit 1 = 128-interval)
static constexpr unsigned lut_reg_first = 0xb000 + offsetof(gaudi2::block_tpc, lut_func32_base_addr_lo);
static constexpr unsigned lut_reg_last = 0xb000 + offsetof(gaudi2::block_tpc, lut_func256_base_addr_hi);

static bool validate_static_program(const std::vector<char> &raw, unsigned group)
{
    if (raw.empty() || raw.size() % 8 || raw.size() > 8191) return false;
    for (unsigned offset = 0; offset < raw.size(); offset += 8) {
        auto *p = (const packet_wreg32 *)(raw.data() + offset);
        if (p->opcode == PACKET_WREG_BULK) {
            auto *b = (const packet_wreg_bulk *)p;
            unsigned first = b->reg_offset, last = first + b->size64 * 8 - 4;
            bool allowed = group == SCAL_MME_COMPUTE_GROUP ?
                (first >= 0xb008 && last <= 0xb2dc) :
                (first >= 0xb5dc && last < 0xb5dc + sizeof(gaudi2::Gaudi2TpcDesc));
            if (!g_wreg_bulk || !allowed || !b->size64 || !b->msg_barrier || b->swtc ||
                (!b->eng_barrier && !(g_mme_nobarrier_allowed && group == SCAL_MME_COMPUTE_GROUP)) ||
                offset + 8 + b->size64 * 8 > raw.size()) return false;
            offset += b->size64 * 8;
            continue;
        }
        if (p->opcode == PACKET_WREG_32) {
            unsigned reg = p->reg_offset;
            bool allowed = group == SCAL_MME_COMPUTE_GROUP ?
                ((reg >= 0xb008 && reg <= 0xb2dc) || reg == 0xb004) :
                ((reg >= 0xb5dc && reg < 0xb5dc + sizeof(gaudi2::Gaudi2TpcDesc)) || reg == 0xbd94 || reg == 0xbd98 ||
                 (g_lut_enabled && reg >= lut_reg_first && reg <= lut_reg_last));
            if (!allowed || !p->msg_barrier || p->swtc) return false;
        } else if (p->opcode == PACKET_FENCE) {
            auto *f = (const packet_fence *)p;
            // Fence 3 everywhere; fence 2 only on MME programs (opt-in split QK fences).
            bool id_ok = f->id == 3 || (f->id == 2 && group == SCAL_MME_COMPUTE_GROUP);
            if (!id_ok || f->target_val != 1 || f->dec_val != 1) return false;
        } else if (p->opcode == PACKET_NOP) {
            auto *n = (const packet_nop *)p;
            bool barrier_ok = n->eng_barrier || (g_mme_nobarrier_allowed && group == SCAL_MME_COMPUTE_GROUP);
            if (offset + 8 != raw.size() || !n->swtc || !barrier_ok) return false;
        } else return false;
    }
    return ((const packet_nop *)(raw.data() + raw.size() - 8))->opcode == PACKET_NOP;
}

static int prepare_ecb(unsigned group, const std::vector<std::vector<char>> &programs, PreparedEcb &job)
{
    unsigned engines = programs.size();
    if (!engines || engines > 30) return -EINVAL;
    std::vector<unsigned> offsets;
    std::vector<char> combined;
    for (const auto &raw : programs) {
        if (!validate_static_program(raw, group)) return -EINVAL;
        combined.resize((combined.size() + 63) & ~63u, 0);
        offsets.push_back(combined.size());
        combined.insert(combined.end(), raw.begin(), raw.end());
    }
    combined.resize((combined.size() + 255) & ~255u, 0);
    scal_pool_handle_t host_pool = nullptr, hbm_pool = nullptr;
    if (scal_get_pool_handle_by_name(own_scal, "host_shared", &host_pool) ||
        scal_get_pool_handle_by_name(own_scal, "hbm_shared", &hbm_pool)) return -ENOENT;
    if (scal_allocate_aligned_buffer(host_pool, 512, 256, &job.metadata) ||
        scal_allocate_aligned_buffer(hbm_pool, combined.size(), 256, &job.code)) return -ENOMEM;
    if (scal_buffer_get_info(job.metadata, &job.meta_info) || scal_buffer_get_info(job.code, &job.code_info)) return -EINVAL;
    if (!job.meta_info.host_address || !job.meta_info.core_address || job.meta_info.core_address % 256) return -EINVAL;
    void *staging = nullptr; synStreamHandle upload = nullptr;
    if (synHostMalloc(0, combined.size(), 0, &staging) != synSuccess ||
        synStreamCreateGeneric(&upload, 0, 0) != synSuccess) return -ENOMEM;
    memcpy(staging, combined.data(), combined.size());
    auto status = synMemCopyAsync(upload, (uintptr_t)staging, combined.size(), job.code_info.device_address, HOST_TO_DRAM);
    if (status == synSuccess) status = synStreamSynchronize(upload);
    if (status == synSuccess) {
        memset(staging, 0, combined.size());
        status = synMemCopyAsync(upload, job.code_info.device_address, combined.size(), (uintptr_t)staging, DRAM_TO_HOST);
        if (status == synSuccess) status = synStreamSynchronize(upload);
    }
    bool equal = status == synSuccess && !memcmp(staging, combined.data(), combined.size());
    synStreamDestroy(upload); synHostFree(0, staging, 0);
    if (!equal) return -EIO;
    memset(job.meta_info.host_address, 0, 512);
    for (unsigned part = 0; part < 2; ++part) {
        auto *size = (eng_arc_cmd_list_size_t *)((char *)job.meta_info.host_address + 256 * part);
        size->cmd_type = ECB_CMD_LIST_SIZE; size->topology_start = 1; size->list_size = 256;
    }
    auto *dynamic = (eng_arc_cmd_nop_t *)((char *)job.meta_info.host_address + 4);
    dynamic->cmd_type = ECB_CMD_NOP; dynamic->switch_cq = 1; dynamic->yield = 1; dynamic->padding = 62;
    auto *entries = (eng_arc_cmd_static_desc_v2_t *)((char *)job.meta_info.host_address + 260);
    for (unsigned i = 0; i < engines; ++i) {
        entries[i].cmd_type = ECB_CMD_STATIC_DESC_V2; entries[i].cpu_index = i; entries[i].addr_index = 6;
        entries[i].size = programs[i].size(); entries[i].addr_offset = offsets[i]; entries[i].yield = i == engines - 1;
    }
    unsigned used = 4 + engines * sizeof(*entries);
    auto *pad = (eng_arc_cmd_nop_t *)((char *)job.meta_info.host_address + 256 + used);
    pad->cmd_type = ECB_CMD_NOP; pad->padding = (256 - used) / 4 - 1;
    job.group = group;
    job.code_image=combined;job.program_offsets=offsets;
    for (const auto &raw:programs) job.program_sizes.push_back(raw.size());
    if (const char *dir=std::getenv("MME_ECB_DEBUG_DIR")) {
        job.debug_code=combined;
        const char *meta=(const char *)job.meta_info.host_address;
        job.debug_metadata.assign(meta,meta+512);
        char path[1024];
        snprintf(path,sizeof(path),"%s/ecb-%u-%lx.bin",dir,group,job.code_info.device_address);
        std::ofstream f(path,std::ios::binary);f.write(combined.data(),combined.size());
        printf("ECB_ADDRESS group=%u code=0x%lx core=0x%x meta_core=0x%x bytes=%zu\n",
               group,job.code_info.device_address,job.code_info.core_address,job.meta_info.core_address,combined.size());fflush(stdout);
    }
    printf("PREPARED_ECB group=%u engines=%u hbm_bytes=%zu verified=1\n", group, engines, combined.size()); fflush(stdout);
    return 0;
}

// Return a prepared ECB's host metadata and device code buffers to their SCAL pools. Nothing may be queued.
static void release_ecb(PreparedEcb &job)
{
    if (job.code) scal_free_buffer(job.code);
    if (job.metadata) scal_free_buffer(job.metadata);
    job = PreparedEcb{};
}

// Rewrite a prepared ECB's device code in place. The new programs must reproduce the original layout exactly
// (engine count, every program's offset and size), so the static descriptors in the host metadata stay valid
// and no pool memory is allocated. The caller guarantees that nothing using the job is queued.
static int reupload_ecb(unsigned group, const std::vector<std::vector<char>> &programs, PreparedEcb &job)
{
    if (!job.code || job.group != group || programs.size() != job.program_sizes.size()) return -EINVAL;
    std::vector<char> combined;
    for (size_t i = 0; i < programs.size(); ++i) {
        if (!validate_static_program(programs[i], group)) return -EINVAL;
        combined.resize((combined.size() + 63) & ~63u, 0);
        if (combined.size() != job.program_offsets[i] || programs[i].size() != job.program_sizes[i]) return -E2BIG;
        combined.insert(combined.end(), programs[i].begin(), programs[i].end());
    }
    combined.resize((combined.size() + 255) & ~255u, 0);
    if (combined.size() != job.code_image.size()) return -E2BIG;
    if (combined == job.code_image) return 0;
    void *staging = nullptr; synStreamHandle upload = nullptr;
    if (synHostMalloc(0, combined.size(), 0, &staging) != synSuccess ||
        synStreamCreateGeneric(&upload, 0, 0) != synSuccess) return -ENOMEM;
    memcpy(staging, combined.data(), combined.size());
    auto status = synMemCopyAsync(upload, (uintptr_t)staging, combined.size(), job.code_info.device_address, HOST_TO_DRAM);
    if (status == synSuccess) status = synStreamSynchronize(upload);
    if (status == synSuccess) {
        memset(staging, 0, combined.size());
        status = synMemCopyAsync(upload, job.code_info.device_address, combined.size(), (uintptr_t)staging, DRAM_TO_HOST);
        if (status == synSuccess) status = synStreamSynchronize(upload);
    }
    bool equal = status == synSuccess && !memcmp(staging, combined.data(), combined.size());
    synStreamDestroy(upload); synHostFree(0, staging, 0);
    if (!equal) return -EIO;
    job.code_image.swap(combined);
    return 0;
}

// Clone validated MME programs, rebinding only operand B inside the full
// descriptor WREG_BULK packets. All geometry, fences and SRAM addresses stay.
static int clone_mme_b(const PreparedEcb &src,uint64_t old_base,uint64_t new_base,
                       uint64_t bytes,PreparedEcb &dst)
{
    if (src.group!=SCAL_MME_COMPUTE_GROUP || src.program_offsets.size()!=2 ||
        src.program_sizes.size()!=2 || src.code_image.empty() || !bytes) return -EINVAL;
    std::vector<std::vector<char>> programs(2);
    for (unsigned e=0;e<2;++e) {
        unsigned start=src.program_offsets[e],size=src.program_sizes[e],patches=0;
        if (start+size>src.code_image.size()) return -EINVAL;
        programs[e].assign(src.code_image.begin()+start,src.code_image.begin()+start+size);
        auto &raw=programs[e];
        for (unsigned off=0;off<size;) {
            if (off+8>size) return -EINVAL;
            auto *p=(packet_wreg32 *)(raw.data()+off);
            if (p->opcode==PACKET_WREG_BULK) {
                auto *b=(packet_wreg_bulk *)p;
                unsigned len=b->size64*8,reg=b->reg_offset;
                if (off+8+len>size) return -EINVAL;
                if (reg<=0xb020 && reg+len>=0xb028) {
                    char *ptr=raw.data()+off+8+(0xb020-reg);
                    uint64_t current=0;memcpy(&current,ptr,8);
                    if (current<old_base || current-old_base>=bytes) return -EINVAL;
                    uint64_t address=new_base+(current-old_base);memcpy(ptr,&address,8);++patches;
                }
                off+=8+len;
            } else {
                if (p->opcode==PACKET_WREG_32 && (p->reg_offset==0xb020 || p->reg_offset==0xb024)) return -EINVAL;
                off+=8;
            }
        }
        if (!patches) return -EINVAL;
    }
    return prepare_ecb(SCAL_MME_COMPUTE_GROUP,programs,dst);
}

// Read back only memory allocated by prepare_ecb; no engine submission or CSR scan.
static int audit_prepared_ecb(const PreparedEcb &p)
{
    if (p.debug_code.empty() || p.debug_metadata.size()!=512) return -EINVAL;
    if (memcmp(p.debug_metadata.data(),p.meta_info.host_address,512)) return -EILSEQ;
    void *host=nullptr;synStreamHandle stream=nullptr;
    if (synHostMalloc(0,p.debug_code.size(),0,&host)!=synSuccess ||
        synStreamCreateGeneric(&stream,0,0)!=synSuccess) return -EIO;
    auto rc=synMemCopyAsync(stream,p.code_info.device_address,p.debug_code.size(),(uintptr_t)host,DRAM_TO_HOST);
    if (rc==synSuccess) rc=synStreamSynchronize(stream);
    bool same=rc==synSuccess && !memcmp(host,p.debug_code.data(),p.debug_code.size());
    synStreamDestroy(stream);synHostFree(0,host,0);
    if (!same) fprintf(stderr,"ECB_AUDIT_MISMATCH group=%u code=0x%lx\n",p.group,p.code_info.device_address);
    return same?0:-EILSEQ;
}

static int queue_ecb(const PreparedEcb &job)
{
    struct { sched_arc_cmd_update_recipe_base_v2_t command; uint64_t address; } update{};
    update.command.opcode = SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2;
    update.command.num_recipe_addrs = 1; update.command.num_engine_group_type = 1;
    update.command.recipe_base_index0 = 6; update.command.engine_group_type[0] = job.group;
    update.address = job.code_info.device_address;
    int rc = own_append(&update, sizeof(update)); if (rc) return rc;
    sched_arc_cmd_dispatch_compute_ecb_list_t dispatch{};
    dispatch.opcode = SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3;
    dispatch.engine_group_type = job.group; dispatch.cmpt_stream_index = 3;
    dispatch.single_static_chunk = 1; dispatch.single_dynamic_chunk = 1;
    dispatch.static_ecb_list_offset = 1; dispatch.dynamic_ecb_list_addr_256 = job.meta_info.core_address >> 8;
    return own_append(&dispatch, sizeof(dispatch));
}

static void append_private_fence(std::vector<char> &raw, unsigned id = 3)
{
    unsigned offset = raw.size(); raw.resize(offset + sizeof(packet_fence), 0);
    auto *f = (packet_fence *)(raw.data() + offset);
    f->opcode = PACKET_FENCE; f->msg_barrier = 1;
    f->id = id; f->target_val = 1; f->dec_val = 1;
}

static unsigned private_monitor_count = 0;
static unsigned private_so_count = 0;

static int private_sync_resources(unsigned &sob, unsigned &monitor)
{
    scal_so_pool_handle_t sos = nullptr; scal_monitor_pool_handle_t mons = nullptr;
    if (scal_get_so_pool_handle_by_name(own_scal, "mme_attention_sos", &sos) ||
        scal_get_so_monitor_handle_by_name(own_scal, "mme_attention_monitors", &mons)) return -ENOENT;
    struct { scal_so_pool_info info; char reserve[512]; } s{};
    struct { scal_monitor_pool_info info; char reserve[512]; } m{};
    if (scal_so_pool_get_info(sos, &s.info) || scal_monitor_pool_get_info(mons, &m.info)) return -EINVAL;
    if (s.info.smIndex != 2 || m.info.smIndex != 2 || s.info.size < 3 || m.info.size < 50) return -EINVAL;
    sob = s.info.baseIdx; monitor = m.info.baseIdx; private_monitor_count = m.info.size; private_so_count = s.info.size;
    printf("PRIVATE_SYNC sm=2 sob=%u monitor=%u\n", sob, monitor); fflush(stdout);
    return 0;
}

// Monitor config, payload address and payload data persist after the monitor
// fires; only the arm register is consumed. rearm_only skips the four static
// writes once a monitor has been configured with the same values in this process.
static bool monitor_rearm_only = false;
static uint64_t monitor_configured[2048];
static uint8_t monitor_wr_num[2048]; // wr_num of the cached config (multi-write heads)

static int arm_private_monitor(unsigned monitor, unsigned sob, unsigned target, uint64_t destination)
{
    const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    packet_msg_short fields{};
    int rc;
    unsigned slot = monitor % 2048;
    uint64_t key = destination ^ ((uint64_t)((sob / 8) >> 8) << 1) ^ 1;
    if (!monitor_rearm_only || monitor_configured[slot] != key || monitor_wr_num[slot]) {
        fields.mon_config_register.msb_sid = (sob / 8) >> 8;
        if ((rc = own_lbw(base + 0x10000 + monitor * 4, fields.value))) return rc;
        if ((rc = own_lbw(base + 0x8000 + monitor * 4, (uint32_t)destination))) return rc;
        if ((rc = own_lbw(base + 0xa000 + monitor * 4, destination >> 32))) return rc;
        if ((rc = own_lbw(base + 0xc000 + monitor * 4, 1))) return rc;
        monitor_configured[slot] = key; monitor_wr_num[slot] = 0;
    }
    fields.value = 0;
    fields.mon_arm_register.sync_group_id = sob / 8;
    fields.mon_arm_register.mask = ~(1u << (sob & 7));
    // mode 0 = "greater than or equal" (as the habanalabs driver arms monitors).
    // The earlier mode 1 is "equal": with cumulative SOBs a monitor armed after its
    // SOB passed the target never fired (2 s timeout in a fast deferred-norm chain).
    fields.mon_arm_register.mode = 0; fields.mon_arm_register.sync_value = target;
    return own_lbw(base + 0xe000 + monitor * 4, fields.value);
}

extern "C" int context_scal_fence_test(uint64_t a, uint64_t b, uint64_t c)
{
    own_begin();
    unsigned sob, monitor;
    int rc = private_sync_resources(sob, monitor); if (rc) return rc;
    constexpr uint64_t fence_address = 0x1000007ffc0ca2dc; // MME0 static-CQ fence3 RDATA_4
    if (!own_host_counter) return -EINVAL;
    uint64_t initial = *own_host_counter;
    std::vector<std::vector<char>> programs(2);
    append_private_fence(programs[0]);
    rc = append_mme(programs[0], a, b, c, 128, 128, 128, own_sob); if (rc) return rc;
    finish_program(programs[0]); finish_program(programs[1]);
    PreparedEcb job;
    rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, programs, job); if (rc) return rc;
    rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob * 4, 0); if (rc) return rc;
    rc = arm_private_monitor(monitor, sob, 1, fence_address); if (rc) return rc;
    rc = queue_ecb(job); if (rc) return rc;
    __sync_synchronize();
    rc = own_submit(); if (rc) return rc;
    // The installed debug register-read API is unimplemented. Instead observe
    // the public host counter: the GEMM must remain blocked until our pulse.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    bool blocked = *own_host_counter == initial;
    // Pulse after dispatch: proves that the scheduler can release an engine
    // stalled in its static CQ, without a CPU wait between these operations.
    rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob * 4, 0x80000001); if (rc) return rc;
    rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + own_sob * 4, 0x80000001); if (rc) return rc;
    __sync_synchronize();
    rc = own_submit();
    if (!rc) rc = scal_host_fence_counter_wait(own_fence, 2, 2000000);
    printf("RESULT_DEVICE_FENCE status=%d blocked_before_pulse=%d counter=%lu\n", rc, blocked, *own_host_counter); fflush(stdout);
    return rc ? rc : (blocked ? 0 : -EIO);
}

static uint64_t tpc_static_fence_address(unsigned index)
{
    // Same physical ordering as the pinned profile's tpc engine cluster.
    return 0x1000007ffc00a2dc + (index / 6) * 0x200000 + (index % 6) * 0x10000;
}

// Opt-in multi-write monitors (Gaudi2 MON_CONFIG.wr_num = count-1): when the
// head monitor fires it writes the payloads of itself and the next count-1
// monitors (hl-thunk gaudi2 tests use the same consecutive-payload layout). The
// followers are configured with payloads only and never armed. Not long-SOB
// mode, so the H6-3342 long-monitor workaround does not apply.
static bool g_tpc_multi_monitor = false;
extern "C" int context_set_tpc_multi_monitor(int on)
{
    g_tpc_multi_monitor = on != 0;
    return 0;
}

static int arm_private_monitor_multi(unsigned monitor, unsigned count, unsigned sob, unsigned target,
                                     const uint64_t *destinations)
{
    if (!count || count > 4) return -EINVAL;
    const uint64_t base = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    packet_msg_short fields{};
    int rc;
    for (unsigned i = 0; i < count; ++i) {
        unsigned m = monitor + i, slot = m % 2048;
        uint8_t wr = i ? 0 : (uint8_t)(count - 1);
        uint64_t key = destinations[i] ^ ((uint64_t)((sob / 8) >> 8) << 1) ^ 1;
        if (monitor_rearm_only && monitor_configured[slot] == key && monitor_wr_num[slot] == wr) continue;
        fields.value = 0;
        fields.mon_config_register.msb_sid = (sob / 8) >> 8;
        fields.mon_config_register.wr_num = wr;
        if ((rc = own_lbw(base + 0x10000 + m * 4, fields.value))) return rc;
        if ((rc = own_lbw(base + 0x8000 + m * 4, (uint32_t)destinations[i]))) return rc;
        if ((rc = own_lbw(base + 0xa000 + m * 4, destinations[i] >> 32))) return rc;
        if ((rc = own_lbw(base + 0xc000 + m * 4, 1))) return rc;
        monitor_configured[slot] = key; monitor_wr_num[slot] = wr;
    }
    fields.value = 0;
    fields.mon_arm_register.sync_group_id = sob / 8;
    fields.mon_arm_register.mask = ~(1u << (sob & 7));
    fields.mon_arm_register.mode = 0; fields.mon_arm_register.sync_value = target;
    return own_lbw(base + 0xe000 + monitor * 4, fields.value);
}

// Release the fences of TPCs [0, tpcs) when sob >= target, using monitors
// first..first+tpcs-1 (one per TPC, or one armed head per four with multi-write).
static int arm_tpc_monitors(unsigned first, unsigned tpcs, unsigned sob, unsigned target)
{
    int rc;
    if (!g_tpc_multi_monitor) {
        for (unsigned i = 0; i < tpcs; ++i)
            if ((rc = arm_private_monitor(first + i, sob, target, tpc_static_fence_address(i)))) return rc;
        return 0;
    }
    for (unsigned i = 0; i < tpcs; i += 4) {
        unsigned count = tpcs - i < 4 ? tpcs - i : 4;
        uint64_t d[4];
        for (unsigned j = 0; j < count; ++j) d[j] = tpc_static_fence_address(i + j);
        if ((rc = arm_private_monitor_multi(first + i, count, sob, target, d))) return rc;
    }
    return 0;
}

extern "C" int context_scal_tpc_fence_test(uint64_t input, uint64_t output, const char *path)
{
    own_begin();
    unsigned sob, monitor;
    int rc = private_sync_resources(sob, monitor); if (rc) return rc;
    uint64_t isa = load_runtime_isa(path); if (!isa) return -EINVAL;
    std::vector<std::vector<char>> programs(24);
    uint64_t addresses[2] = {input, output};
    unsigned cols[2] = {128, 128}, rows[2] = {128, 128}, types[2] = {6, 6};
    for (unsigned i = 0; i < 24; ++i) {
        unsigned first = 128 * i / 24, count = 128 * (i + 1) / 24 - first;
        append_private_fence(programs[i]);
        rc = append_tpc(programs[i], isa, addresses, cols, rows, types, 2, 128, first, count, own_sob);
        if (rc) return rc;
        finish_program(programs[i]);
    }
    PreparedEcb job;
    rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, programs, job); if (rc) return rc;
    uint64_t initial = *own_host_counter;
    rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob * 4, 0); if (rc) return rc;
    for (unsigned i = 0; i < 24; ++i) {
        rc = arm_private_monitor(monitor + i, sob, 1, tpc_static_fence_address(i)); if (rc) return rc;
    }
    rc = queue_ecb(job); if (rc) return rc;
    __sync_synchronize();
    rc = own_submit(); if (rc) return rc;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    bool blocked = *own_host_counter == initial;
    rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob * 4, 0x80000001); if (rc) return rc;
    rc = own_lbw(mmDCORE2_SYNC_MNGR_OBJS_BASE + own_sob * 4, 0x80000001); if (rc) return rc;
    __sync_synchronize();
    rc = own_submit();
    if (!rc) rc = scal_host_fence_counter_wait(own_fence, 25, 2000000);
    printf("RESULT_TPC_DEVICE_FENCE status=%d blocked_before_pulse=%d counter=%lu engines=24\n", rc, blocked, *own_host_counter); fflush(stdout);
    return rc ? rc : (blocked ? 0 : -EIO);
}

// One isolated, quiesced-context plan. Buffers and ISA stay allocated until
// runtime teardown; this research API does not support concurrent callers.
//
// Timing ablation modes (stage breakdown). Only MODE_FULL computes the
// attention result; other modes run a subset of the same cached commands.
enum AttentionMode : unsigned {
    MODE_FULL = 0,      // QK -> exp/sum -> PV -> normalize (device fences)
    MODE_QK = 1,        // QK only
    MODE_QK_EXP = 2,    // QK -> exp/sum
    MODE_QK_EXP_PV = 3, // QK -> exp/sum -> PV
    MODE_EXP = 4,       // exp/sum alone (no fence)
    MODE_PV = 5,        // PV alone
    MODE_NORM = 6,      // normalize alone
    MODE_LBW = 7,       // same number of scheduler LBW writes as MODE_FULL, no ECB
    MODE_FLOOR = 8,     // one LBW completion pulse: submit/wait floor
    MODE_COUNT = 9
};
struct AttentionPlan {
    PreparedEcb mme, tpc;
    unsigned sob = 0, monitor = 0, tpcs = 0, qk_parts = 0, mode = 0, wait = 0;
    bool use_mme = false, use_tpc = false, ready = false;
    bool dual_qk = false; // QK columns split across DCORE0_MME_0 and DCORE2_MME_0
    bool chain = false;   // prepared for context_attention_run_n (serial blocks, one submit)
};
static bool attention_dual_qk = false;
static bool attention_chain = false;
// Static-CQ fence3 RDATA of the two MME masters (DCORE stride 0x200000, same as TPC).
static constexpr uint64_t mme0_fence_address = 0x1000007ffc0ca2dc;
static constexpr uint64_t mme2_fence_address = 0x1000007ffc4ca2dc;
// Static-CQ fence2 RDATA_4 (mmDCORE{0,2}_MME_QM_CP_FENCE2_RDATA_4 = 0x40CA2C8 /
// 0x44CA2C8 in hl-thunk gaudi2/asic_reg). Used only by the opt-in split-fence
// mode: S-reuse (QK) credits on fence 2, exp->PV credits on fence 3, so each
// counter receives credits in the same order as its fences consume them.
static constexpr uint64_t mme0_fence2_address = 0x1000007ffc0ca2c8;
static constexpr uint64_t mme2_fence2_address = 0x1000007ffc4ca2c8;
static constexpr unsigned chain_monitor_stride = 51;
static AttentionPlan attention_plan;
static double attention_last_us[3];
static unsigned attention_last_bytes = 0;

extern "C" int context_attention_prepare_mode(uint64_t base, unsigned keys, unsigned valid,
                                               unsigned tpcs, const char *exp_path, const char *norm_path,
                                               unsigned mode)
{
    auto &p = attention_plan;
    if (p.ready || !own_stream || !keys || keys % 128 || keys > 37760 ||
        !valid || valid > keys || !tpcs || tpcs > 24 || mode >= MODE_COUNT) return -EINVAL;
    int rc = private_sync_resources(p.sob, p.monitor); if (rc) return rc;
    p.mode = mode; p.tpcs = tpcs;
    p.dual_qk = attention_dual_qk;
    p.chain = attention_chain;
    if (p.chain && mode != MODE_FULL) return -EINVAL;
    if (p.dual_qk && (keys < 256 || (mode != MODE_FULL && mode != MODE_QK))) return -EINVAL;
    // Dual: MME0 takes the first ceil(chunks/2) 128-column chunks, MME2 the rest.
    unsigned first = p.dual_qk ? (keys / 128 + 1) / 2 * 128 : (keys < 32768 ? keys : 32768);
    p.qk_parts = keys > first ? 2 : 1;
    const bool qk = mode <= MODE_QK_EXP_PV;
    const bool exp_stage = mode == MODE_FULL || mode == MODE_QK_EXP || mode == MODE_QK_EXP_PV || mode == MODE_EXP;
    const bool pv = mode == MODE_FULL || mode == MODE_QK_EXP_PV || mode == MODE_PV;
    const bool norm_stage = mode == MODE_FULL || mode == MODE_NORM;
    p.use_mme = qk || pv; p.use_tpc = exp_stage || norm_stage;
    p.wait = (mode == MODE_FULL || mode == MODE_QK_EXP || mode == MODE_EXP || mode == MODE_NORM) ? tpcs : 1;
    if (p.dual_qk && mode == MODE_QK) p.wait = 2;
    if (p.use_mme || p.use_tpc) {
        uint64_t exp = load_runtime_isa(exp_path), norm = load_runtime_isa(norm_path);
        if (!exp || !norm) return -EINVAL;
        const uint64_t matrix = 128 * 128 * 2, wide = 128 * keys * 2;
        uint64_t q = base, k = q + matrix, v = k + wide, s = v + wide;
        uint64_t out = s + wide, bound = out + matrix, sum = bound + 256;
        std::vector<std::vector<char>> mme(2), tpc(24);
        if (qk && p.dual_qk) {
            unsigned qk_sob = mode == MODE_QK ? own_sob : p.sob;
            rc = append_mme(mme[0], q, k, s, first, 128, keys, qk_sob); if (rc) return rc;
            // Chain: MME2 must not overwrite S before the previous block's PV (on MME0) read it.
            if (p.chain) append_private_fence(mme[1]);
            rc = append_mme(mme[1], q, k + first * 2, s + first * 2, keys - first, 128, keys, qk_sob);
            if (rc) return rc;
        } else if (qk) {
            unsigned last_sob = mode == MODE_QK ? own_sob : p.sob;
            rc = append_mme(mme[0], q, k, s, first, 128, keys, p.qk_parts == 2 ? p.sob : last_sob);
            if (rc) return rc;
            if (p.qk_parts == 2) {
                append_private_fence(mme[0]);
                rc = append_mme(mme[0], q, k + first * 2, s + first * 2, keys - first, 128, keys, last_sob);
                if (rc) return rc;
            }
        }
        if (pv) {
            if (mode != MODE_PV) append_private_fence(mme[0]);
            rc = append_mme(mme[0], s, v, out, 128, keys, 128, mode == MODE_FULL ? p.sob + 2 : own_sob);
            if (rc) return rc;
        }
        finish_program(mme[0]); finish_program(mme[1]);
        uint64_t exp_addrs[4] = {s, bound, s, sum}, norm_addrs[3] = {out, sum, out};
        unsigned exp_cols[4] = {keys, 1, keys, 1}, norm_cols[3] = {128, 1, 128};
        unsigned rows[4] = {128, 128, 128, 128}, exp_types[4] = {6, 6, 6, 7}, norm_types[3] = {6, 7, 6};
        const unsigned exp_sob = (mode == MODE_FULL || mode == MODE_QK_EXP_PV) ? p.sob + 1 : own_sob;
        for (unsigned i = 0; i < 24; ++i) {
            if (i < tpcs) {
                unsigned row = 128 * i / tpcs, count = 128 * (i + 1) / tpcs - row;
                if (g_lut_enabled) {
                    const unsigned lo[4] = {
                        0xb000 + (unsigned)offsetof(gaudi2::block_tpc, lut_func256_base_addr_lo),
                        0xb000 + (unsigned)offsetof(gaudi2::block_tpc, lut_func128_base_addr_lo),
                        0xb000 + (unsigned)offsetof(gaudi2::block_tpc, lut_func64_base_addr_lo),
                        0xb000 + (unsigned)offsetof(gaudi2::block_tpc, lut_func32_base_addr_lo)};
                    // Synapse keeps its own LUT bases in these engine registers for the whole
                    // process; overwriting all four changed framework BF16 sigmoid. Only the
                    // tables in g_lut_mask (default: 128-interval, which holds BF16_EXP) are set.
                    bool first_write = true;
                    for (unsigned t = 0; t < 4; ++t) {
                        if (!(g_lut_mask & (1u << t))) continue;
                        push_reg(tpc[i], lo[t], (uint32_t)g_lut_address[t], first_write);
                        push_reg(tpc[i], lo[t] + 4, (uint32_t)(g_lut_address[t] >> 32));
                        first_write = false;
                    }
                }
                if (exp_stage) {
                    if (mode != MODE_EXP) append_private_fence(tpc[i]);
                    rc = append_tpc(tpc[i], exp, exp_addrs, exp_cols, rows, exp_types, 4, valid, row, count, exp_sob);
                    if (rc) return rc;
                }
                if (norm_stage) {
                    if (mode == MODE_FULL) append_private_fence(tpc[i]);
                    rc = append_tpc(tpc[i], norm, norm_addrs, norm_cols, rows, norm_types, 3, valid, row, count, own_sob);
                    if (rc) return rc;
                }
            }
            finish_program(tpc[i]);
        }
        if (p.use_mme && (rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, mme, p.mme))) return rc;
        if (p.use_tpc && (rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, tpc, p.tpc))) return rc;
    }
    p.ready = true;
    printf("ATTENTION_PLAN keys=%u valid=%u tpcs=%u qk_parts=%u mode=%u wait=%u dual_qk=%d first=%u cached=1\n",
           keys, valid, tpcs, p.qk_parts, mode, p.wait, (int)p.dual_qk, first);
    fflush(stdout); return 0;
}

extern "C" int context_attention_prepare(uint64_t base, unsigned keys, unsigned valid,
                                          unsigned tpcs, const char *exp_path, const char *norm_path)
{
    return context_attention_prepare_mode(base, keys, valid, tpcs, exp_path, norm_path, MODE_FULL);
}

static double attention_now_us()
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern "C" int context_attention_run()
{
    const auto &p = attention_plan;
    if (!p.ready || p.chain) return -EINVAL;
    own_begin(); // previous run completed: the whole ring is free
    const double t0 = attention_now_us();
    const unsigned pi0 = own_pi;
    const uint64_t sync = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    int rc;
    if (p.mode == MODE_FLOOR) {
        rc = own_lbw(sync + own_sob * 4, 0x80000001); if (rc) return rc;
    } else if (p.mode == MODE_LBW) {
        // MODE_FULL issues 3 SOB clears + 5 LBW per monitor arm.
        unsigned writes = 3 + 5 * (2 * p.tpcs + 1 + (p.qk_parts == 2 ? 1 : 0));
        for (unsigned i = 0; i < writes; ++i) { rc = own_lbw(sync + (p.sob + 7) * 4, 0); if (rc) return rc; }
        rc = own_lbw(sync + own_sob * 4, 0x80000001); if (rc) return rc;
    } else {
        for (unsigned i = 0; i < 3; ++i) {
            rc = own_lbw(sync + (p.sob + i) * 4, 0); if (rc) return rc;
        }
        const bool tpc_fenced = p.mode == MODE_FULL || p.mode == MODE_QK_EXP || p.mode == MODE_QK_EXP_PV;
        for (unsigned i = 0; tpc_fenced && i < p.tpcs; ++i) {
            rc = arm_private_monitor(p.monitor + i, p.sob, p.qk_parts, tpc_static_fence_address(i)); if (rc) return rc;
            if (p.mode == MODE_FULL) {
                rc = arm_private_monitor(p.monitor + 24 + i, p.sob + 2, 1, tpc_static_fence_address(i)); if (rc) return rc;
            }
        }
        constexpr uint64_t mme_fence = 0x1000007ffc0ca2dc;
        if (p.mode == MODE_FULL || p.mode == MODE_QK_EXP_PV) {
            rc = arm_private_monitor(p.monitor + 48, p.sob + 1, p.tpcs, mme_fence); if (rc) return rc;
        }
        if (p.qk_parts == 2 && !p.dual_qk && p.mode <= MODE_QK_EXP_PV) {
            rc = arm_private_monitor(p.monitor + 49, p.sob, 1, mme_fence); if (rc) return rc;
        }
        if (p.use_mme && (rc = queue_ecb(p.mme))) return rc;
        if (p.use_tpc && (rc = queue_ecb(p.tpc))) return rc;
    }
    __sync_synchronize();
    const double t1 = attention_now_us();
    rc = own_submit();
    const double t2 = attention_now_us();
    if (!rc) rc = scal_host_fence_counter_wait(own_fence, p.wait, 2000000);
    const double t3 = attention_now_us();
    attention_last_us[0] = t1 - t0; attention_last_us[1] = t2 - t1; attention_last_us[2] = t3 - t2;
    attention_last_bytes = own_pi - pi0;
    return rc;
}

// Upload gen_lut output (G2LUT6) to hbm_binary and enable LUT base writes in
// subsequent TPC programs. Buffers stay allocated until runtime teardown.
extern "C" int context_attention_set_lut(const char *path)
{
    if (attention_plan.ready || !own_scal || !path) return -EINVAL;
    std::ifstream file(path, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(file)), {});
    if (blob.size() < 28 || memcmp(blob.data(), "G2LUT6", 6)) return -EINVAL;
    uint32_t count, sizes[4];
    memcpy(&count, blob.data() + 8, 4); memcpy(sizes, blob.data() + 12, 16);
    if (count != 4) return -EINVAL;
    size_t offsets[4], total = 0;
    for (unsigned i = 0; i < 4; ++i) {
        if (!sizes[i] || sizes[i] % 128) return -EINVAL;
        offsets[i] = total; total += (sizes[i] + 8191) & ~8191u;
    }
    if (28 + total != blob.size()) return -EINVAL;
    scal_pool_handle_t pool = nullptr; scal_buffer_handle_t buffer = nullptr;
    if (scal_get_pool_handle_by_name(own_scal, "hbm_binary", &pool) ||
        scal_allocate_aligned_buffer(pool, total, 8192, &buffer)) return -ENOMEM;
    scal_buffer_info_t info{}; if (scal_buffer_get_info(buffer, &info) || !info.device_address) return -EINVAL;
    void *host = nullptr; synStreamHandle upload = nullptr;
    if (synHostMalloc(0, total, 0, &host) != synSuccess ||
        synStreamCreateGeneric(&upload, 0, 0) != synSuccess) return -ENOMEM;
    memcpy(host, blob.data() + 28, total);
    synStatus status = synMemCopyAsync(upload, (uintptr_t)host, total, info.device_address, HOST_TO_DRAM);
    if (status == synSuccess) status = synStreamSynchronize(upload);
    if (status == synSuccess) {
        memset(host, 0, total);
        status = synMemCopyAsync(upload, info.device_address, total, (uintptr_t)host, DRAM_TO_HOST);
        if (status == synSuccess) status = synStreamSynchronize(upload);
    }
    bool equal = status == synSuccess && !memcmp(host, blob.data() + 28, total);
    synStreamDestroy(upload); synHostFree(0, host, 0);
    if (!equal) return -EIO;
    for (unsigned i = 0; i < 4; ++i) g_lut_address[i] = info.device_address + offsets[i];
    g_lut_enabled = true;
    printf("LUT_UPLOADED bytes=%zu base=0x%lx verified=1\n", total, (unsigned long)info.device_address);
    fflush(stdout);
    return 0;
}

extern "C" int context_attention_set_lut_mask(unsigned mask)
{
    if (attention_plan.ready || !mask || mask > 15) return -EINVAL;
    g_lut_mask = mask;
    return 0;
}

extern "C" int context_attention_set_dual_qk(int enabled)
{
    if (attention_plan.ready) return -EBUSY;
    attention_dual_qk = enabled != 0;
    return 0;
}

extern "C" int context_attention_set_rearm_only(int enabled)
{
    monitor_rearm_only = enabled != 0;
    memset(monitor_configured, 0, sizeof(monitor_configured));
    return 0;
}

// Serial chain of n blocks (same cached programs, same SRAM addresses) in ONE
// submission. SOBs count cumulatively across blocks; block i uses its own
// monitors (stride 51) armed just before its own dispatch, so the scheduler
// arms later blocks while engines compute earlier ones.
extern "C" int context_attention_run_n(unsigned n)
{
    const auto &p = attention_plan;
    if (!p.ready || !p.chain || p.mode != MODE_FULL || !n || n * chain_monitor_stride > private_monitor_count)
        return -EINVAL;
    own_begin(); // run budget is checked per command (-ENOSPC if one run exceeds the ring)
    const double t0 = attention_now_us();
    const unsigned pi0 = own_pi;
    const uint64_t sync = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    int rc;
    for (unsigned i = 0; i < 3; ++i) {
        rc = own_lbw(sync + (p.sob + i) * 4, 0); if (rc) return rc;
    }
    for (unsigned b = 0; b < n; ++b) {
        const unsigned mon = p.monitor + b * chain_monitor_stride;
        for (unsigned i = 0; i < p.tpcs; ++i) {
            rc = arm_private_monitor(mon + i, p.sob, p.qk_parts * (b + 1), tpc_static_fence_address(i)); if (rc) return rc;
            rc = arm_private_monitor(mon + 24 + i, p.sob + 2, b + 1, tpc_static_fence_address(i)); if (rc) return rc;
        }
        rc = arm_private_monitor(mon + 48, p.sob + 1, p.tpcs * (b + 1), mme0_fence_address); if (rc) return rc;
        if (p.qk_parts == 2 && !p.dual_qk) {
            rc = arm_private_monitor(mon + 49, p.sob, 2 * b + 1, mme0_fence_address); if (rc) return rc;
        }
        if (p.dual_qk) {
            // Target b: block 0 fires at arm time (SOB already >= 0).
            rc = arm_private_monitor(mon + 50, p.sob + 2, b, mme2_fence_address); if (rc) return rc;
        }
        if ((rc = queue_ecb(p.mme)) || (rc = queue_ecb(p.tpc))) return rc;
    }
    __sync_synchronize();
    const double t1 = attention_now_us();
    rc = own_submit();
    const double t2 = attention_now_us();
    if (!rc) rc = scal_host_fence_counter_wait(own_fence, p.tpcs * n, 2000000);
    const double t3 = attention_now_us();
    attention_last_us[0] = t1 - t0; attention_last_us[1] = t2 - t1; attention_last_us[2] = t3 - t2;
    attention_last_bytes = own_pi - pi0;
    return rc;
}

extern "C" int context_attention_set_chain(int enabled)
{
    if (attention_plan.ready) return -EBUSY;
    attention_chain = enabled != 0;
    return 0;
}

// ---------------------------------------------------------------------------
// Software-pipelined blocks with double-buffered S, row sums and output.
// Buffers for parity x: S[x], L[x] (row sums), O[x]. Set 0 is the standard
// layout; set 1 follows the 4 KiB guard after L[0].
// Engine order:  MME: QK0 QK1 PV0 QK2 PV1 ... QK(n-1) PV(n-2) PV(n-1)
//                TPC: EX0 EX1 NM0 EX2 NM1 ... EX(n-1) NM(n-2) NM(n-1)
// so QK(b+1) on the MMEs overlaps EX(b) on the TPCs. Hazards:
//   S[x]  rewritten by QK(b+2): MME0 is in order after PV(b); MME2 fences on PV(b) (SOB_PV >= b+1).
//   L[x]  rewritten by EX(b+2): each TPC runs NM(b) before EX(b+2).
//   O[x]  rewritten by PV(b+2): PV waits all EX(b+2), which follow NM(b) on every TPC.
// SOBs are cumulative: sob+0 QK halves (2/block), sob+1 EX (tpcs/block), sob+2 PV (1/block).
struct PipelinePlan {
    PreparedEcb qk[2], pv[2], ex[2], nm[2];
    unsigned sob = 0, monitor = 0, tpcs = 0;
    bool ready = false;
    bool split = false; // PV split by key halves across MME0/MME2 (context_attention_prepare_split)
    unsigned qk_signals = 2;
    bool compiler_qk = false;
};
static PipelinePlan pipe_plan;
// Opt-in compiler master/slave QK. The captured full-width result uses one
// row-major S tensor; split-PV and exp consume strided views of its key halves.
static std::vector<uint32_t> compiler_qk_words;
static std::vector<unsigned> compiler_qk_engines;
static uint64_t attention_query_alt = 0;
extern "C" int context_attention_set_query_alt(uint64_t address)
{
    if (pipe_plan.ready || !address || address % 256) return -EINVAL;
    attention_query_alt = address;
    return 0;
}
extern "C" int context_attention_set_compiler_qk(unsigned n, const uint32_t *words, const unsigned *engines)
{
    if (pipe_plan.ready || !words || !engines || !n || n > 8) return -EINVAL;
    unsigned counts[2] = {};
    for (unsigned i = 0; i < n; ++i) {
        if (engines[i] > 1) return -EINVAL;
        ++counts[engines[i]];
    }
    if (!counts[0] || counts[0] != counts[1]) return -EINVAL;
    compiler_qk_words.assign(words, words + 182 * n);
    compiler_qk_engines.assign(engines, engines + n);
    return 0;
}

// Opt-in experiment: remove the commit engine barrier on COPY_AND_INC
// (0x3ff); retain explicit SRAM reuse and exp->PV dependency fences.
// Keep all default barriers unless this explicit isolated test is selected.
static bool g_mme_async_commit = false;
extern "C" int context_set_mme_async_commit(int enabled)
{
    if (enabled && !g_mme_nobarrier_allowed) return -EINVAL;
    g_mme_async_commit = enabled != 0;
    return 0;
}
static void append_compiler_mme(std::vector<char> &raw, const uint32_t *words,
                                uint64_t a, uint64_t b, uint64_t c, unsigned sob)
{
    uint32_t w[182]; memcpy(w, words, sizeof(w));
    auto set64 = [&](unsigned off, uint64_t value) { w[(off - 8) / 4] = value; w[(off - 8) / 4 + 1] = value >> 32; };
    set64(0x08, 0); set64(0x10, c); set64(0x18, a); set64(0x20, b);
    w[(0x14c - 8) / 4] = (uint32_t)(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob * 4);
    push_regs(raw, 0xb008, w, 182);
    if (g_mme_prefence_id >= 0) { append_fence_packet(raw, g_mme_prefence_id); g_mme_prefence_id = -1; }
    push_reg(raw, 0xb004, 0x3ff, !g_mme_async_commit);
}
// Deferred normalization: blocks only produce unnormalized O and row sums; NM runs
// once per parity after the last PV (a real integration normalizes per head).
// S[x] is still safe: EX(b+2) waits QK(b+2), which completes only after PV(b)
// (MME0 in order, MME2 fenced on PV(b)).
static bool attention_defer_norm = false;
// TPC pre-fence descriptors: write the next kernel's QM descriptor while the
// previous kernel may still run (no engine barrier), then fence, then
// tpc_cmd (cache invalidate, engine barrier) + execute. Relies on the QM copy
// being latched at execute; a wrong latch shows up as a wrong-buffer RMS failure.
static bool attention_tpc_prefence = false;

extern "C" int context_attention_set_tpc_prefence(int enabled)
{
    if (pipe_plan.ready) return -EBUSY;
    attention_tpc_prefence = enabled != 0;
    return 0;
}

// append_tpc() ends with two WREG_32 packets (tpc_cmd, tpc_execute). With
// prefence the private fence goes between the descriptor and those two.
static int append_tpc_fenced(std::vector<char> &raw, uint64_t isa, const uint64_t *addresses,
                             const unsigned *cols, const unsigned *rows, const unsigned *types,
                             unsigned count, unsigned valid_cols, unsigned first_row, unsigned work_rows, unsigned sob,
                             const unsigned *strides = nullptr)
{
    if (!attention_tpc_prefence) {
        append_private_fence(raw);
        return append_tpc(raw, isa, addresses, cols, rows, types, count, valid_cols, first_row, work_rows, sob, strides);
    }
    std::vector<char> t;
    int rc = append_tpc(t, isa, addresses, cols, rows, types, count, valid_cols, first_row, work_rows, sob, strides);
    if (rc) return rc;
    if (t.size() < 24) return -EINVAL;
    auto *first = (packet_wreg32 *)t.data(); // WREG_32 or WREG_BULK: same control-bit layout
    first->eng_barrier = 0;
    raw.insert(raw.end(), t.begin(), t.end() - 16);
    append_private_fence(raw);
    raw.insert(raw.end(), t.end() - 16, t.end());
    return 0;
}

extern "C" int context_attention_set_defer_norm(int enabled)
{
    if (pipe_plan.ready) return -EBUSY;
    attention_defer_norm = enabled != 0;
    return 0;
}

static void push_lut_bases(std::vector<char> &raw)
{
    const unsigned lo[4] = {
        0xb000 + (unsigned)offsetof(gaudi2::block_tpc, lut_func256_base_addr_lo),
        0xb000 + (unsigned)offsetof(gaudi2::block_tpc, lut_func128_base_addr_lo),
        0xb000 + (unsigned)offsetof(gaudi2::block_tpc, lut_func64_base_addr_lo),
        0xb000 + (unsigned)offsetof(gaudi2::block_tpc, lut_func32_base_addr_lo)};
    bool first_write = true;
    for (unsigned t = 0; t < 4; ++t) {
        if (!(g_lut_mask & (1u << t))) continue;
        push_reg(raw, lo[t], (uint32_t)g_lut_address[t], first_write);
        push_reg(raw, lo[t] + 4, (uint32_t)(g_lut_address[t] >> 32));
        first_write = false;
    }
}

extern "C" int context_attention_prepare_pipelined(uint64_t base, unsigned keys, unsigned valid,
                                                    unsigned tpcs, const char *exp_path, const char *norm_path)
{
    auto &p = pipe_plan;
    if (p.ready || attention_plan.ready || !own_stream || keys < 256 || keys % 128 || keys > 37760 ||
        !valid || valid > keys || !tpcs || tpcs > 24) return -EINVAL;
    int rc = private_sync_resources(p.sob, p.monitor); if (rc) return rc;
    uint64_t exp = load_runtime_isa(exp_path), norm = load_runtime_isa(norm_path);
    if (!exp || !norm) return -EINVAL;
    const uint64_t matrix = 128 * 128 * 2, wide = 128 * keys * 2;
    const uint64_t q = base, k = q + matrix, v = k + wide, s0 = v + wide, out0 = s0 + wide;
    const uint64_t bound = out0 + matrix, sum0 = bound + 256, guard = sum0 + 512;
    const uint64_t s1 = guard + 4096, out1 = s1 + wide, sum1 = out1 + matrix;
    const uint64_t S[2] = {s0, s1}, O[2] = {out0, out1}, L[2] = {sum0, sum1};
    const unsigned first = (keys / 128 + 1) / 2 * 128;
    p.tpcs = tpcs;
    for (unsigned x = 0; x < 2; ++x) {
        std::vector<std::vector<char>> qk(2), pv(2), ex(24), nm(24);
        rc = append_mme(qk[0], q, k, S[x], first, 128, keys, p.sob); if (rc) return rc;
        append_private_fence(qk[1]);
        rc = append_mme(qk[1], q, k + first * 2, S[x] + first * 2, keys - first, 128, keys, p.sob); if (rc) return rc;
        append_private_fence(pv[0]);
        rc = append_mme(pv[0], S[x], v, O[x], 128, keys, 128, p.sob + 2); if (rc) return rc;
        for (auto &r : qk) finish_program(r);
        for (auto &r : pv) finish_program(r);
        uint64_t ea[4] = {S[x], bound, S[x], L[x]}, na[3] = {O[x], L[x], O[x]};
        unsigned ec[4] = {keys, 1, keys, 1}, nc[3] = {128, 1, 128}, rows[4] = {128, 128, 128, 128};
        unsigned et[4] = {6, 6, 6, 7}, nt[3] = {6, 7, 6};
        for (unsigned i = 0; i < 24; ++i) {
            if (i < tpcs) {
                unsigned row = 128 * i / tpcs, count = 128 * (i + 1) / tpcs - row;
                if (g_lut_enabled) push_lut_bases(ex[i]);
                rc = append_tpc_fenced(ex[i], exp, ea, ec, rows, et, 4, valid, row, count, p.sob + 1); if (rc) return rc;
                rc = append_tpc_fenced(nm[i], norm, na, nc, rows, nt, 3, valid, row, count, own_sob); if (rc) return rc;
            }
            finish_program(ex[i]); finish_program(nm[i]);
        }
        if ((rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, qk, p.qk[x])) ||
            (rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, pv, p.pv[x])) ||
            (rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, ex, p.ex[x])) ||
            (rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, nm, p.nm[x]))) return rc;
    }
    p.ready = true;
    printf("PIPELINE_PLAN keys=%u valid=%u tpcs=%u first=%u set1=0x%lx\n", keys, valid, tpcs, first,
           (unsigned long)s1);
    fflush(stdout);
    return 0;
}

// Key-split layout (also used by the probe). Each MME owns one key half:
// K^T, S and the partial PV output are stored per half, contiguous, so every
// descriptor keeps the validated shape (leading = own width).
// out[]: q ktA ktB v s0A s0B o0A o0B bound sum0 guard s1A s1B o1A o1B sum1 end
extern "C" int context_attention_split_layout(uint64_t base, unsigned keys, uint64_t *out)
{
    if (!out || keys < 256 || keys % 128 || keys > 37760) return -EINVAL;
    const uint64_t first = (keys / 128 + 1) / 2 * 128, second = keys - first, matrix = 128 * 128 * 2;
    uint64_t a = base;
    auto take = [&](uint64_t bytes) { uint64_t r = a; a = (a + bytes + 255) & ~255ull; return r; };
    if (!compiler_qk_words.empty()) {
        out[0] = take(matrix); out[1] = take(128 * keys * 2); out[2] = out[1] + first * 2;
        out[3] = take(keys * 128 * 2);
        out[4] = take(128 * keys * 2); out[5] = out[4] + first * 2;
        out[6] = take(matrix); out[7] = take(matrix); out[8] = take(256); out[9] = take(2048); out[10] = take(4096);
        out[11] = take(128 * keys * 2); out[12] = out[11] + first * 2;
        out[13] = take(matrix); out[14] = take(matrix); out[15] = take(2048); out[16] = a;
        return 0;
    }
    out[0] = take(matrix); out[1] = take(128 * first * 2); out[2] = take(128 * second * 2);
    out[3] = take(keys * 128 * 2);
    out[4] = take(128 * first * 2); out[5] = take(128 * second * 2); out[6] = take(matrix); out[7] = take(matrix);
    out[8] = take(256); out[9] = take(2048); out[10] = take(4096); // 9: row sums, or 256x2 FP32 half stats
    out[11] = take(128 * first * 2); out[12] = take(128 * second * 2); out[13] = take(matrix); out[14] = take(matrix);
    out[15] = take(2048); out[16] = a;
    return 0;
}

// Split-PV pipeline. MME0: QK_A(b), PV_A(b); MME2: QK_B(b), PV_B(b); same engine
// order as the unsplit pipeline. S[x] half h is rewritten by QK_h(b+2) only after
// PV_h(b) on the same MME (in order), so no MME2 fence is needed. EX waits both
// QK halves (SOB_QK += 2/block); both PV halves wait EX (two monitors); NM waits
// both PV halves (SOB_PV += 2/block). NM computes (O_A + O_B) / l into O_A.
// Per-half maxima (exp_lut_half + normalize3): EX work units are (row, half),
// 256 per block; stats[u] = {m_h, l_h}. Same buffers and hazards as split.
static bool attention_half_max = false;

extern "C" int context_attention_set_half_max(int enabled)
{
    if (pipe_plan.ready) return -EBUSY;
    attention_half_max = enabled != 0;
    return 0;
}

extern "C" int context_attention_prepare_split(uint64_t base, unsigned keys, unsigned valid,
                                                unsigned tpcs, const char *exp_path, const char *norm_path)
{
    auto &p = pipe_plan;
    if (p.ready || attention_plan.ready || !own_stream || !valid || valid > keys || !tpcs || tpcs > 24) return -EINVAL;
    uint64_t L[17];
    int rc = context_attention_split_layout(base, keys, L); if (rc) return rc;
    if ((rc = private_sync_resources(p.sob, p.monitor))) return rc;
    uint64_t exp = load_runtime_isa(exp_path), norm = load_runtime_isa(norm_path);
    if (!exp || !norm) return -EINVAL;
    const unsigned first = (keys / 128 + 1) / 2 * 128, second = keys - first;
    const uint64_t q = L[0], ka = L[1], kb = L[2], v = L[3];
    const uint64_t SA[2] = {L[4], L[11]}, SB[2] = {L[5], L[12]}, OA[2] = {L[6], L[13]}, OB[2] = {L[7], L[14]};
    const uint64_t SUM[2] = {L[9], L[15]};
    p.tpcs = tpcs; p.split = true; p.compiler_qk = !compiler_qk_words.empty();
    p.qk_signals = p.compiler_qk ? compiler_qk_engines.size() : 2;
    for (unsigned x = 0; x < 2; ++x) {
        const uint64_t query = x && attention_query_alt ? attention_query_alt : q;
        std::vector<std::vector<char>> qk(2), pv(2), ex(24), nm(24);
        if (p.compiler_qk) {
            // Both engines write across both key halves. Before parity reuse,
            // wait for BOTH preceding PV halves, not just this engine's PV.
            append_private_fence(qk[0]); append_private_fence(qk[1]);
            for (unsigned i = 0; i < compiler_qk_engines.size(); ++i)
                append_compiler_mme(qk[compiler_qk_engines[i]], compiler_qk_words.data() + i * 182, query, ka, SA[x], p.sob);
        } else {
            if ((rc = append_mme(qk[0], query, ka, SA[x], first, 128, first, p.sob))) return rc;
            if ((rc = append_mme(qk[1], query, kb, SB[x], second, 128, second, p.sob))) return rc;
        }
        append_private_fence(pv[0]);
        if ((rc = append_mme(pv[0], SA[x], v, OA[x], 128, first, 128, p.sob + 2, p.compiler_qk ? keys : first))) return rc;
        append_private_fence(pv[1]);
        if ((rc = append_mme(pv[1], SB[x], v + (uint64_t)first * 128 * 2, OB[x], 128, second, 128, p.sob + 2, p.compiler_qk ? keys : second))) return rc;
        for (auto &r : qk) finish_program(r);
        for (auto &r : pv) finish_program(r);
        const bool half = attention_half_max;
        uint64_t ea[3] = {SA[x], SB[x], SUM[x]}, na[4] = {OA[x], OB[x], SUM[x], OA[x]};
        unsigned ec[3] = {first, second, 1}, nc[4] = {128, 128, 1, 128}, rows[4] = {128, 128, 128, 128};
        unsigned et[3] = {6, 6, 7}, nt[4] = {6, 6, 7, 6};
        // Half mode: EX tensors (stats 256x2, SA, SB), NM tensors (OA, OB, stats 256x2, OA).
        uint64_t ha[3] = {SUM[x], SA[x], SB[x]};
        unsigned hc[3] = {2, first, second}, hr[3] = {256, 128, 128}, ht[3] = {7, 6, 6};
        unsigned hs[3] = {2, p.compiler_qk ? keys : first, p.compiler_qk ? keys : second};
        unsigned es[3] = {p.compiler_qk ? keys : first, p.compiler_qk ? keys : second, 1};
        unsigned hnc[4] = {128, 128, 2, 128}, hnr[4] = {128, 128, 256, 128};
        for (unsigned i = 0; i < 24; ++i) {
            if (i < tpcs) {
                unsigned row = 128 * i / tpcs, count = 128 * (i + 1) / tpcs - row;
                unsigned unit = 256 * i / tpcs, units = 256 * (i + 1) / tpcs - unit;
                if (g_lut_enabled) push_lut_bases(ex[i]);
                if (half) {
                    if ((rc = append_tpc_fenced(ex[i], exp, ha, hc, hr, ht, 3, valid, unit, units, p.sob + 1, hs))) return rc;
                    if ((rc = append_tpc_fenced(nm[i], norm, na, hnc, hnr, nt, 4, valid, row, count, own_sob))) return rc;
                } else {
                    if ((rc = append_tpc_fenced(ex[i], exp, ea, ec, rows, et, 3, valid, row, count, p.sob + 1, es))) return rc;
                    if ((rc = append_tpc_fenced(nm[i], norm, na, nc, rows, nt, 4, valid, row, count, own_sob))) return rc;
                }
            }
            finish_program(ex[i]); finish_program(nm[i]);
        }
        if ((rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, qk, p.qk[x])) ||
            (rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, pv, p.pv[x])) ||
            (rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, ex, p.ex[x])) ||
            (rc = prepare_ecb(SCAL_TPC_COMPUTE_GROUP, nm, p.nm[x]))) return rc;
    }
    p.ready = true;
    printf("SPLIT_PLAN keys=%u valid=%u tpcs=%u first=%u second=%u half_max=%d end=0x%lx\n", keys, valid, tpcs,
           first, second, (int)attention_half_max, (unsigned long)L[16]);
    fflush(stdout);
    return 0;
}

extern "C" int context_attention_run_pipelined(unsigned n)
{
    const auto &p = pipe_plan;
    unsigned stride = p.compiler_qk ? 52 : chain_monitor_stride;
    if (!p.ready || n < 2 || n * stride > private_monitor_count) return -EINVAL;
    own_begin(); // run budget is checked per command (-ENOSPC if one run exceeds the ring)
    const double t0 = attention_now_us();
    const unsigned pi0 = own_pi;
    const uint64_t sync = mmDCORE2_SYNC_MNGR_OBJS_BASE;
    int rc;
    for (unsigned i = 0; i < 3; ++i) {
        rc = own_lbw(sync + (p.sob + i) * 4, 0); if (rc) return rc;
    }
    auto mon = [&](unsigned b) { return p.monitor + b * stride; };
    auto qk = [&](unsigned b) -> int {
        if (p.compiler_qk) {
            unsigned target = b >= 2 ? 2 * (b - 1) : 0;
            int r = arm_private_monitor(mon(b) + 49, p.sob + 2, target, mme0_fence_address);
            if (!r) r = arm_private_monitor(mon(b) + 51, p.sob + 2, target, mme2_fence_address);
            return r ? r : queue_ecb(p.qk[b & 1]);
        }
        if (p.split) return queue_ecb(p.qk[b & 1]);
        int r = arm_private_monitor(mon(b) + 50, p.sob + 2, b >= 2 ? b - 1 : 0, mme2_fence_address);
        return r ? r : queue_ecb(p.qk[b & 1]);
    };
    auto pv = [&](unsigned b) -> int {
        int r = arm_private_monitor(mon(b) + 48, p.sob + 1, p.tpcs * (b + 1), mme0_fence_address);
        if (!r && p.split) r = arm_private_monitor(mon(b) + 50, p.sob + 1, p.tpcs * (b + 1), mme2_fence_address);
        return r ? r : queue_ecb(p.pv[b & 1]);
    };
    auto ex = [&](unsigned b) -> int {
        for (unsigned i = 0; i < p.tpcs; ++i) {
            int r = arm_private_monitor(mon(b) + i, p.sob, p.qk_signals * (b + 1), tpc_static_fence_address(i)); if (r) return r;
        }
        return queue_ecb(p.ex[b & 1]);
    };
    auto nm = [&](unsigned b) -> int {
        for (unsigned i = 0; i < p.tpcs; ++i) {
            int r = arm_private_monitor(mon(b) + 24 + i, p.sob + 2, (p.split ? 2 : 1) * (b + 1),
                                        tpc_static_fence_address(i)); if (r) return r;
        }
        return queue_ecb(p.nm[b & 1]);
    };
    const bool defer = attention_defer_norm;
    if ((rc = qk(0)) || (rc = ex(0))) return rc;
    for (unsigned b = 1; b < n; ++b) {
        if ((rc = qk(b)) || (rc = ex(b)) || (rc = pv(b - 1))) return rc;
        if (!defer && (rc = nm(b - 1))) return rc;
    }
    if ((rc = pv(n - 1))) return rc;
    if (defer && (rc = nm(n - 2))) return rc;
    if ((rc = nm(n - 1))) return rc;
    __sync_synchronize();
    const double t1 = attention_now_us();
    rc = own_submit();
    const double t2 = attention_now_us();
    if (!rc) rc = scal_host_fence_counter_wait(own_fence, p.tpcs * (defer ? 2 : n), 2000000);
    const double t3 = attention_now_us();
    attention_last_us[0] = t1 - t0; attention_last_us[1] = t2 - t1; attention_last_us[2] = t3 - t2;
    attention_last_bytes = own_pi - pi0;
    return rc;
}

// ---------------------------------------------------------------------------
// Raw MME descriptors (compiler dumps converted by mme_dump.py). Descriptor i is
// 182 words for 0xb008..0xb2dc, run on MME engine[i] (0: DCORE0_MME_0, 1: DCORE2_MME_0),
// with base addresses A/B/Cout0/Cout1 patched in and so0Addr = the host-fence SOB.
// Each descriptor signals once; run waits for all n. Research API: one plan per process.
static PreparedEcb raw_mme_job;
static unsigned raw_mme_count = 0;
static bool raw_mme_ready = false;

extern "C" int context_mme_raw_prepare(unsigned n, const uint32_t *words, const unsigned *engine,
                                       const uint64_t *a, const uint64_t *b, const uint64_t *c0,
                                       const uint64_t *c1)
{
    if (raw_mme_ready || !own_stream || !n || n > 8 || !words || !engine) return -EINVAL;
    std::vector<std::vector<char>> programs(2);
    const bool nb = g_mme_nobarrier_allowed;
    if (nb) g_push_eng_barrier = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (engine[i] > 1) return -EINVAL;
        uint32_t w[182]; memcpy(w, words + 182 * i, sizeof(w));
        auto set = [&](unsigned off, uint32_t value) { w[(off - 8) / 4] = value; };
        set(0x08, (uint32_t)c1[i]); set(0x0c, (uint32_t)(c1[i] >> 32));
        set(0x10, (uint32_t)c0[i]); set(0x14, (uint32_t)(c0[i] >> 32));
        set(0x18, (uint32_t)a[i]);  set(0x1c, (uint32_t)(a[i] >> 32));
        set(0x20, (uint32_t)b[i]);  set(0x24, (uint32_t)(b[i] >> 32));
        set(0x14c, (uint32_t)(mmDCORE2_SYNC_MNGR_OBJS_BASE + own_sob * 4));
        push_regs(programs[engine[i]], 0xb008, w, 182);
        push_reg(programs[engine[i]], 0xb004, 0x3ff, true);
    }
    if (nb) g_finish_eng_barrier = 0;
    for (auto &r : programs) finish_program(r);
    g_push_eng_barrier = 1; g_finish_eng_barrier = 1;
    int rc = prepare_ecb(SCAL_MME_COMPUTE_GROUP, programs, raw_mme_job); if (rc) return rc;
    raw_mme_count = n; raw_mme_ready = true;
    return 0;
}

// Returns 0 and host elapsed time in *us (submit -> all n completions).
// Includes own_submit and host-fence wait; not MME write-active duration.
extern "C" int context_mme_raw_run(double *us)
{
    if (!raw_mme_ready || !us) return -EINVAL;
    own_begin();
    int rc = queue_ecb(raw_mme_job); if (rc) return rc;
    __sync_synchronize();
    const double t0 = attention_now_us();
    rc = own_submit();
    if (!rc) rc = scal_host_fence_counter_wait(own_fence, raw_mme_count, 2000000);
    *us = attention_now_us() - t0;
    return rc;
}

// Host-side split of the last run: enqueue, submit call, completion wait (us).
extern "C" int context_attention_last_timing(double *us, unsigned *bytes)
{
    if (!us || !bytes) return -EINVAL;
    for (unsigned i = 0; i < 3; ++i) us[i] = attention_last_us[i];
    *bytes = attention_last_bytes;
    return 0;
}

#include "runtime_batch.hpp"

#include "runtime_tiled.hpp"

#include "runtime_quad.hpp"
