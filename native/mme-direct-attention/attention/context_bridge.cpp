// SPDX-License-Identifier: MIT
// Borrow the existing Synapse context. Never open/close the device here.
#include <synapse_api.h>
#include <synapse_types.h>
#include <hlthunk.h>
#include "gaudi2/gaudi2_packets.h"
#include "gaudi2/gaudi2.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iterator>
#include <cstddef>
#include <chrono>
#include <thread>
#include "third_party/hcl/scal.h"
#include "third_party/hcl/gaudi2_arc_sched_packets.h"
#include "third_party/synapseai-core/gaudi2/asic_reg/gaudi2_blocks.h"
#include "third_party/synapseai-core/gaudi2/gaudi2_tpc_descriptor.h"
#include "../desc128.h"

static scal_stream_handle_t own_stream = nullptr;
static scal_host_fence_counter_handle_t own_fence = nullptr;
static void *own_commands = nullptr;
static unsigned own_pi = 0, own_alignment = 0, own_command_alignment = 0, own_sob = 0;
static scal_handle_t own_scal = nullptr;
static volatile const uint64_t *own_host_counter = nullptr;

// Circular command buffer (64 KiB). own_pi is a free-running byte counter; the
// buffer offset and the submitted 16-bit PI are own_pi & 0xffff (as HCL's
// Gaudi2CyclicBufferManager). Every caller waits for its submission to complete
// before the next run, so the whole ring is free at own_begin(). One run may use
// at most own_budget bytes: a full 64 KiB would make PI - CI == 0 (looks empty).
static constexpr unsigned own_ring = 65536, own_budget = 65536 - 512;
static unsigned own_run_start = 0;

static void own_begin() { own_run_start = own_pi; }

static char *own_slot() { return (char *)own_commands + (own_pi & (own_ring - 1)); }

static int own_reserve(unsigned bytes)
{
    // Commands may not cross a command_alignment boundary (a divisor of own_ring).
    unsigned remaining = own_command_alignment - own_pi % own_command_alignment;
    unsigned pad = remaining < bytes ? remaining : 0;
    if (own_pi - own_run_start + pad + bytes > own_budget) return -ENOSPC;
    if (pad) {
        auto *nop = (sched_arc_cmd_nop_t *)own_slot();
        nop->opcode = SCHED_COMPUTE_ARC_CMD_NOP; nop->padding_count = pad / 4 - 1;
        own_pi += pad;
    }
    return 0;
}

static int own_submit()
{
    return scal_stream_submit(own_stream, own_pi & (own_ring - 1), own_alignment);
}

static int own_append(const void *data, unsigned bytes)
{
    if (!own_commands || bytes > own_command_alignment || bytes % 4) return -EINVAL;
    int rc = own_reserve(bytes); if (rc) return rc;
    memcpy(own_slot(), data, bytes); own_pi += bytes;
    return 0;
}

static int run_ecb(unsigned group, unsigned engines, const char *program, unsigned raw_bytes)
{
    own_begin();
    if (!own_scal || own_alignment != 1 || !engines || engines > 30 ||
        raw_bytes < sizeof(packet_nop) || raw_bytes + sizeof(packet_nop) > 4096) return -EINVAL;
    scal_pool_handle_t pool = nullptr; scal_buffer_handle_t buffer = nullptr;
    if (scal_get_pool_handle_by_name(own_scal, "host_shared", &pool)) return -ENOENT;
    if (scal_allocate_aligned_buffer(pool, 8192, 256, &buffer)) return -ENOMEM;
    scal_buffer_info_t memory{}; if (scal_buffer_get_info(buffer, &memory)) return -EINVAL;
    if (!memory.host_address || !memory.core_address || memory.core_address % 256) return -EINVAL;
    memset(memory.host_address, 0, 8192);
    auto *raw = (char *)memory.host_address + 512;
    memcpy(raw, program, raw_bytes);
    memcpy(raw + raw_bytes, program + raw_bytes - sizeof(packet_nop), sizeof(packet_nop));
    // The engine FW supplies the HBM high address to QMAN static CP_DMAs.
    // host_shared is valid for ARC metadata, but not for the static program.
    // Keep raw commands in HBM and verify their upload before dispatch.
    scal_pool_handle_t hbm_pool = nullptr; scal_buffer_handle_t code_buffer = nullptr;
    if (scal_get_pool_handle_by_name(own_scal, "hbm_shared", &hbm_pool) ||
        scal_allocate_aligned_buffer(hbm_pool, 4096, 256, &code_buffer)) return -ENOMEM;
    scal_buffer_info_t code{};
    if (scal_buffer_get_info(code_buffer, &code) || !code.device_address) return -EINVAL;
    void *staging = nullptr; synStreamHandle upload = nullptr;
    if (synHostMalloc(0, 4096, 0, &staging) != synSuccess ||
        synStreamCreateGeneric(&upload, 0, 0) != synSuccess) return -ENOMEM;
    memset(staging, 0, 4096); memcpy(staging, raw, raw_bytes + sizeof(packet_nop));
    synStatus copied = synMemCopyAsync(upload, (uintptr_t)staging, 4096, code.device_address, HOST_TO_DRAM);
    if (copied == synSuccess) copied = synStreamSynchronize(upload);
    if (copied == synSuccess) {
        memset(staging, 0, 4096);
        copied = synMemCopyAsync(upload, code.device_address, 4096, (uintptr_t)staging, DRAM_TO_HOST);
        if (copied == synSuccess) copied = synStreamSynchronize(upload);
    }
    bool code_equal = copied == synSuccess && !memcmp(staging, raw, raw_bytes + sizeof(packet_nop));
    synStreamDestroy(upload); synHostFree(0, staging, 0);
    printf("SCAL_ECB_PROGRAM hbm=0x%lx uploaded_equal=%d bytes=%u\n", code.device_address, code_equal, raw_bytes + (unsigned)sizeof(packet_nop)); fflush(stdout);
    if (!code_equal) return -EIO;
    auto list = [](void *base) {
        auto *h = (eng_arc_cmd_list_size_t *)base;
        h->cmd_type = ECB_CMD_LIST_SIZE; h->topology_start = 1; h->list_size = 256;
    };
    list(memory.host_address); list((char *)memory.host_address + 256);
    auto *dynamic = (eng_arc_cmd_nop_t *)((char *)memory.host_address + 4);
    dynamic->cmd_type = ECB_CMD_NOP; dynamic->switch_cq = 1; dynamic->yield = 1; dynamic->padding = 62;
    auto *s = (eng_arc_cmd_static_desc_v2_t *)((char *)memory.host_address + 260);
    for (unsigned i = 0; i < engines; ++i) {
        s[i].cmd_type = ECB_CMD_STATIC_DESC_V2; s[i].cpu_index = i; s[i].addr_index = 6;
        s[i].size = i == 0 ? raw_bytes : sizeof(packet_nop);
        s[i].addr_offset = i == 0 ? 0 : raw_bytes; s[i].yield = i == engines - 1;
    }
    unsigned used = 4 + engines * sizeof(eng_arc_cmd_static_desc_v2_t);
    auto *pad = (eng_arc_cmd_nop_t *)((char *)memory.host_address + 256 + used);
    pad->cmd_type = ECB_CMD_NOP; pad->padding = (256 - used) / 4 - 1;
    struct { sched_arc_cmd_update_recipe_base_v2_t command; uint64_t address; } update{};
    update.command.opcode = SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2;
    update.command.num_recipe_addrs = 1; update.command.num_engine_group_type = 1;
    update.command.recipe_base_index0 = 6; update.command.engine_group_type[0] = group;
    update.address = code.device_address;
    int rc = own_append(&update, sizeof(update)); if (rc) return rc;
    sched_arc_cmd_dispatch_compute_ecb_list_t dispatch{};
    dispatch.opcode = SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3;
    dispatch.engine_group_type = group; dispatch.cmpt_stream_index = 3;
    dispatch.single_static_chunk = 1; dispatch.single_dynamic_chunk = 1;
    dispatch.static_ecb_list_offset = 1; dispatch.dynamic_ecb_list_addr_256 = memory.core_address >> 8;
    rc = own_append(&dispatch, sizeof(dispatch)); if (rc) return rc;
    __sync_synchronize();
    printf("SCAL_ECB core=0x%x source=0x%lx raw_bytes=%u pi=%u\n", memory.core_address, update.address, raw_bytes, own_pi); fflush(stdout);
    rc = own_submit();
    if (!rc) rc = scal_host_fence_counter_wait(own_fence, 1, 2000000);
    printf("SCAL_ECB group=%u completion_status=%d\n", group, rc); fflush(stdout);
    return rc;
}

static void push_reg(std::vector<char> &raw, uint16_t address, uint32_t value, bool barrier = false)
{
    unsigned offset = raw.size(); raw.resize(offset + sizeof(packet_wreg32), 0);
    auto *p = (packet_wreg32 *)(raw.data() + offset);
    p->opcode = PACKET_WREG_32; p->msg_barrier = 1; p->reg_offset = address;
    p->value = value; p->eng_barrier = barrier;
}

// Optional (default off): write contiguous descriptor words with one
// PACKET_WREG_BULK instead of one msg-barriered WREG_32 per word.
static bool g_wreg_bulk = false;
// Opt-in experiment (default 1 = validated behaviour): MME descriptor writes and
// the MME program-end NOP without engine barrier, so the next descriptor can be
// fetched/written while the previous op computes. The commit (0xb004) keeps its
// engine barrier. Only set to 0 around MME program construction.
static int g_push_eng_barrier = 1, g_finish_eng_barrier = 1;
static bool g_mme_nobarrier_allowed = false;

static void push_regs(std::vector<char> &raw, uint16_t address, const uint32_t *words, unsigned count)
{
    unsigned i = 0;
    if (g_wreg_bulk && address % 8 == 4 && count >= 3) {
        push_reg(raw, address, words[0], g_push_eng_barrier != 0); // align the bulk span to 8 bytes
        i = 1;
    }
    if (g_wreg_bulk && (address + i * 4) % 8 == 0 && count - i >= 2) {
        unsigned pairs = (count - i) / 2;
        unsigned offset = raw.size(); raw.resize(offset + sizeof(packet_wreg_bulk) + pairs * 8, 0);
        auto *p = (packet_wreg_bulk *)(raw.data() + offset);
        p->size64 = pairs; p->opcode = PACKET_WREG_BULK; p->msg_barrier = 1; p->eng_barrier = g_push_eng_barrier;
        p->reg_offset = address + i * 4;
        memcpy(raw.data() + offset + sizeof(packet_wreg_bulk), words + i, pairs * 8);
        i += pairs * 2;
    }
    for (; i < count; ++i) push_reg(raw, address + i * 4, words[i], i == 0 && g_push_eng_barrier);
}

extern "C" int context_set_wreg_bulk(int enabled)
{
    g_wreg_bulk = enabled != 0;
    return 0;
}

static void finish_program(std::vector<char> &raw)
{
    unsigned offset = raw.size(); raw.resize(offset + sizeof(packet_nop), 0);
    auto *p = (packet_nop *)(raw.data() + offset);
    p->opcode = PACKET_NOP; p->eng_barrier = g_finish_eng_barrier; p->msg_barrier = 1; p->swtc = 1;
}

extern "C" int context_set_mme_nobarrier_allowed(int on)
{
    g_mme_nobarrier_allowed = on != 0;
    return 0;
}

// Descriptor prefetch (opt-in): when >= 0, the NEXT append_mme/append_compiler_mme
// call writes its 182 descriptor registers, then a private fence with this id,
// then the commit; the value resets to -1. Writing the shadow descriptor while
// the previous committed op runs is what back-to-back descriptors already do.
static int g_mme_prefence_id = -1;
static void append_fence_packet(std::vector<char> &raw, unsigned id)
{
    unsigned offset = raw.size(); raw.resize(offset + sizeof(packet_fence), 0);
    auto *f = (packet_fence *)(raw.data() + offset);
    f->opcode = PACKET_FENCE; f->msg_barrier = 1;
    f->id = id; f->target_val = 1; f->dec_val = 1;
}

static int append_mme(std::vector<char> &raw, uint64_t a, uint64_t b, uint64_t c,
                      unsigned cols, unsigned reduction, unsigned leading, unsigned sob, unsigned a_stride = 0)
{
    if (!cols || cols % 128 || cols > 32768 || !reduction || reduction % 128 || reduction > 37760) return -EINVAL;
    if (!leading) leading = cols;
    if (leading < cols || leading > 37760) return -EINVAL;
    if (!a_stride) a_stride = reduction;
    if (a_stride < reduction || a_stride > 37760) return -EINVAL;
    uint32_t words[182]; memcpy(words, desc_words, sizeof(words));
    auto set = [&](unsigned off, uint32_t value) { words[(off - 8) / 4] = value; };
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
    tensor(0x40, reduction, 128 * a_stride, 0, 128 * a_stride, reduction, 2 * a_stride);
    tensor(0x98, cols, reduction * leading, 128, 0, 64, leading);
    tensor(0xf0, cols, 128 * leading, 128, 128 * leading, 128, 2 * leading);
    set(0x200, a_stride); set(0x258, leading); set(0x228, reduction - 1); set(0x280, cols / 128 - 1);
    set(0x10, (uint32_t)c); set(0x14, (uint32_t)(c >> 32));
    set(0x18, (uint32_t)a); set(0x1c, (uint32_t)(a >> 32));
    set(0x20, (uint32_t)b); set(0x24, (uint32_t)(b >> 32));
    set(0x14c, (uint32_t)(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob * 4));
    push_regs(raw, 0xb008, words, 182);
    if (g_mme_prefence_id >= 0) { append_fence_packet(raw, g_mme_prefence_id); g_mme_prefence_id = -1; }
    push_reg(raw, 0xb004, 0x3ff, true);
    return 0;
}

extern "C" int context_scal_mme_ex(uint64_t a, uint64_t b, uint64_t c,
                                    unsigned cols, unsigned reduction, unsigned leading)
{
    std::vector<char> raw;
    int rc = append_mme(raw, a, b, c, cols, reduction, leading, own_sob); if (rc) return rc;
    finish_program(raw);
    return run_ecb(SCAL_MME_COMPUTE_GROUP, 2, raw.data(), raw.size());
}

extern "C" int context_scal_mme_ecb(uint64_t a, uint64_t b, uint64_t c)
{
    return context_scal_mme_ex(a, b, c, 128, 128, 128);
}

// Loaded kernel binaries by path and content: a plan that is released and prepared again (another key-capacity
// bucket) reuses the same 64 KiB hbm_binary buffers instead of allocating new ones.
static std::map<std::string, std::pair<std::vector<char>, uint64_t>> g_isa_cache;

static uint64_t load_runtime_isa(const char *path)
{
    if (!own_scal) return 0;
    std::ifstream file(path, std::ios::binary);
    std::vector<char> isa((std::istreambuf_iterator<char>(file)), {});
    if (isa.empty() || isa.size() > 65536) return 0;
    auto cached = g_isa_cache.find(path);
    if (cached != g_isa_cache.end() && cached->second.first == isa) return cached->second.second;
    uint64_t address = 0;
    scal_pool_handle_t pool = nullptr; scal_buffer_handle_t buffer = nullptr;
    if (scal_get_pool_handle_by_name(own_scal, "hbm_binary", &pool) ||
        scal_allocate_aligned_buffer(pool, 65536, 256, &buffer)) return 0;
    scal_buffer_info_t binary{}; if (scal_buffer_get_info(buffer, &binary)) return 0;
    void *host = nullptr; synStreamHandle upload = nullptr;
    if (synHostMalloc(0, 65536, 0, &host) != synSuccess ||
        synStreamCreateGeneric(&upload, 0, 0) != synSuccess) return 0;
    memset(host, 0, 65536); memcpy(host, isa.data(), isa.size());
    synStatus copied = synMemCopyAsync(upload, (uintptr_t)host, 65536, binary.device_address, HOST_TO_DRAM);
    if (copied == synSuccess) copied = synStreamSynchronize(upload);
    synStreamDestroy(upload); synHostFree(0, host, 0);
    if (copied != synSuccess) return 0;
    address = binary.device_address;
    g_isa_cache[path] = {isa, address};
    return address;
}

// Per-kernel AXI rate-limiter reset tokens (kernel_config). 8/6 is what the
// SynapseAI_Core Gaudi2 runner uses; adjustable for experiments only.
static unsigned g_tpc_rd_token = 8, g_tpc_wr_token = 6;

extern "C" int context_set_tpc_rate_tokens(unsigned rd, unsigned wr)
{
    if (!rd || !wr || rd > 255 || wr > 255) return -EINVAL; // 0 stalls the TPC (2 s timeout)
    g_tpc_rd_token = rd; g_tpc_wr_token = wr;
    return 0;
}

static int append_tpc(std::vector<char> &raw, uint64_t isa, const uint64_t *addresses,
                       const unsigned *cols, const unsigned *rows, const unsigned *types,
                       unsigned count, unsigned valid_cols, unsigned first_row, unsigned work_rows, unsigned sob,
                       const unsigned *strides = nullptr, const uint32_t *padding = nullptr)
{
    if (!count || count > 4 || !work_rows || first_row + work_rows > rows[0] || rows[0] > 16384) return -EINVAL;
    gaudi2::Gaudi2TpcDesc desc{};
    for (unsigned i = 0; i < count; ++i) {
        if (!cols[i] || cols[i] > 37760 || !rows[i] || rows[i] > 16384 || (types[i] != 6 && types[i] != 7)) return -EINVAL;
        unsigned stride = strides ? strides[i] : cols[i];
        if (stride < cols[i] || stride > 37760) return -EINVAL;
        auto &t = desc.m_tensors[i]; uint64_t address = addresses[i];
        if (padding) t.padding_value._raw=padding[i];
        t.base_addr_low._raw = (uint32_t)address; t.base_addr_high._raw = address >> 32;
        t.tensor_config.data_type = types[i]; t.tensor_config.valid_dim_mask = 0x1f; t.tensor_config.last_dim = 1;
        t.dim_0_size.v = cols[i]; t.dim_0_stride.v = 1; t.dim_1_size.v = rows[i]; t.dim_1_stride.v = stride;
        t.dim_2_size.v = t.dim_3_size.v = t.dim_4_size.v = 1;
        t.dim_2_stride.v = t.dim_3_stride.v = t.dim_4_stride.v = stride * rows[i];
    }
    desc.m_so.message._raw = 0x80000001;
    desc.m_so.addr.v = (uint32_t)(mmDCORE2_SYNC_MNGR_OBJS_BASE + sob * 4);
    auto &d = desc.m_desc;
    d.kernel_base_address_low._raw = isa; d.kernel_base_address_high._raw = isa >> 32;
    d.tid_size_dim_0.v = 1; d.tid_size_dim_1.v = work_rows; d.tid_base_dim_1.v = first_row; d.srf[0].v = valid_cols;
    d.tid_size_dim_2.v = d.tid_size_dim_3.v = d.tid_size_dim_4.v = 1;
    d.kernel_config.aso_evict_l0 = 1; d.kernel_config.num_valid_srfs = 32;
    d.kernel_config.rd_rate_limit_rst_token = g_tpc_rd_token; d.kernel_config.wr_rate_limit_rst_token = g_tpc_wr_token;
    d.kernel_config.irf_32bit_compatibility = 1;
    auto *words = (const uint32_t *)&desc;
    push_regs(raw, 0xb000 + offsetof(gaudi2::block_tpc, qm_tensor_0), words, sizeof(desc) / 4);
    push_reg(raw, 0xb000 + offsetof(gaudi2::block_tpc, tpc_cmd), 0xf, true);
    push_reg(raw, 0xb000 + offsetof(gaudi2::block_tpc, tpc_execute), 1);
    return 0;
}

extern "C" int context_scal_tpc(const uint64_t *addresses, const unsigned *cols,
                                const unsigned *rows, const unsigned *types,
                                unsigned count, const char *path, unsigned valid_cols)
{
    uint64_t isa = load_runtime_isa(path); if (!isa) return -EINVAL;
    std::vector<char> raw;
    int rc = append_tpc(raw, isa, addresses, cols, rows, types, count, valid_cols, 0, rows[0], own_sob);
    if (rc) return rc;
    finish_program(raw);
    return run_ecb(SCAL_TPC_COMPUTE_GROUP, 24, raw.data(), raw.size());
}

extern "C" int context_scal_tpc_relu(uint64_t input, uint64_t output, const char *path)
{
    uint64_t addresses[2] = {input, output};
    unsigned cols[2] = {128, 128}, rows[2] = {128, 128}, types[2] = {6, 6};
    return context_scal_tpc(addresses, cols, rows, types, 2, path, 128);
}

static int own_lbw(uint64_t address, uint32_t value)
{
    if (!own_stream || !own_commands) return -EINVAL;
    int rc = own_reserve(sizeof(sched_arc_cmd_lbw_write_t)); if (rc) return rc;
    auto *write = (sched_arc_cmd_lbw_write_t *)own_slot();
    memset(write, 0, sizeof(*write)); write->opcode = SCHED_COMPUTE_ARC_CMD_LBW_WRITE;
    write->dst_addr = (uint32_t)address; write->src_data = value;
    own_pi += sizeof(*write);
    return 0;
}

extern "C" int context_scal_mme(uint64_t a, uint64_t b, uint64_t c)
{
    own_begin();
    if (!own_stream || own_alignment != 1) return -EINVAL;
    uint32_t words[182]; memcpy(words, desc_words, sizeof(words));
    auto set = [&](unsigned offset, uint32_t value) { words[(offset - 8) / 4] = value; };
    set(0x10, (uint32_t)c); set(0x14, (uint32_t)(c >> 32));
    set(0x18, (uint32_t)a); set(0x1c, (uint32_t)(a >> 32));
    set(0x20, (uint32_t)b); set(0x24, (uint32_t)(b >> 32));
    set(0x14c, (uint32_t)(mmDCORE2_SYNC_MNGR_OBJS_BASE + own_sob * 4));
    // Only the descriptor range and commit register verified in direct/evidence.
    constexpr uint64_t mme_descriptor = 0x1000007ffc0cb000;
    for (unsigned i = 0; i < 182; ++i) {
        int rc = own_lbw(mme_descriptor + 8 + i * 4, words[i]); if (rc) return rc;
    }
    int rc = own_lbw(mme_descriptor + 4, 0x3ff); if (rc) return rc;
    __sync_synchronize();
    rc = own_submit();
    if (!rc) rc = scal_host_fence_counter_wait(own_fence, 1, 2000000);
    printf("SCAL_MME pi=%u completion_status=%d\n", own_pi, rc); fflush(stdout);
    return rc;
}

// Exercise a stream added by the isolated test's SCAL config. It must not be a
// stream owned by Synapse/HCL: never replace their circular command buffers.
extern "C" int context_scal_nop(int fd)
{
    scal_handle_t scal = nullptr;
    scal_stream_handle_t stream = nullptr;
    scal_pool_handle_t pool = nullptr;
    scal_buffer_handle_t buffer = nullptr;
    if (scal_get_handle_from_fd(fd, &scal)) return -ENOENT;
    if (scal_get_stream_handle_by_name(scal, "mme_attention0", &stream)) return -ENOENT;
    struct { scal_stream_info_t info; unsigned char reserve[512]; } initial{};
    int initial_status = scal_stream_get_info(stream, &initial.info);
    if (!initial_status && initial.info.control_core_buffer) return -EBUSY;
    if (scal_get_pool_handle_by_name(scal, "host_shared", &pool)) return -ENOENT;
    if (scal_allocate_buffer(pool, 65536, &buffer)) return -ENOMEM;
    scal_buffer_info_t memory{};
    if (scal_buffer_get_info(buffer, &memory) || !memory.host_address) return -ENOMEM;
    memset(memory.host_address, 0, 65536);
    if (scal_stream_set_commands_buffer(stream, buffer)) return -EINVAL;
    struct { scal_stream_info_t info; unsigned char reserve[512]; } state{};
    if (scal_stream_get_info(stream, &state.info)) return -EINVAL;
    unsigned alignment = state.info.submission_alignment;
    printf("SCAL_OWN_STREAM index=%u alignment=%u command_alignment=%u\n",
           state.info.index, alignment, state.info.command_alignment);
    fflush(stdout);
    if (!alignment || alignment > 512 || (alignment & (alignment - 1))) return -EINVAL;
    scal_host_fence_counter_handle_t fence = nullptr;
    if (scal_get_host_fence_counter_handle_by_name(scal, "mme_attention_fence0", &fence)) return -ENOENT;
    struct { scal_host_fence_counter_info_t info; unsigned char reserve[512]; } f{};
    if (scal_host_fence_counter_get_info(fence, &f.info) || f.info.sm != 2) return -EINVAL;
    unsigned command_bytes = alignment < 16 ? 16 : alignment;
    printf("SCAL_OWN_FENCE sm=%u sob=%u\n", f.info.sm, f.info.so_index);
    for (unsigned run = 0; run < 4; ++run) {
        auto *nop = (sched_arc_cmd_nop_t *)((char *)memory.host_address + run * command_bytes);
        nop->opcode = SCHED_COMPUTE_ARC_CMD_NOP;
        nop->padding_count = 0;
        auto *marker = (sched_arc_cmd_lbw_write_t *)((char *)nop + 4);
        marker->opcode = SCHED_COMPUTE_ARC_CMD_LBW_WRITE;
        marker->dst_addr = (uint32_t)(mmDCORE2_SYNC_MNGR_OBJS_BASE + f.info.so_index * 4);
        marker->src_data = 0x80000001;
        if (command_bytes > 16) {
            auto *pad = (sched_arc_cmd_nop_t *)((char *)nop + 16);
            pad->opcode = SCHED_COMPUTE_ARC_CMD_NOP; pad->padding_count = (command_bytes - 16) / 4 - 1;
        }
        unsigned pi = (run + 1) * command_bytes;
        __sync_synchronize();
        if (scal_stream_submit(stream, pi, alignment)) return -EIO;
        int waited = scal_host_fence_counter_wait(fence, 1, 2000000);
        printf("SCAL_NOP iteration=%u pi=%u host_fence_wait=%d counter=%lu\n", run, pi, waited, *f.info.ctr); fflush(stdout);
        if (waited) return -ETIMEDOUT;
    }
    // The stream retains the bound buffer until Synapse/SCAL teardown. Do not
    // free a buffer still referenced by firmware, even after its last command.
    own_scal = scal; own_stream = stream; own_fence = fence; own_commands = memory.host_address;
    own_pi = 4 * command_bytes; own_alignment = alignment; own_run_start = own_pi;
    own_command_alignment = state.info.command_alignment; own_sob = f.info.so_index;
    own_host_counter = f.info.ctr;
    return 0;
}

// Read-only queries of the SCAL instance owned by Synapse. These do not allocate
// buffers, alter producer indices, or submit engine commands.
extern "C" int context_scal_info(int fd)
{
    scal_handle_t scal = nullptr;
    int rc = scal_get_handle_from_fd(fd, &scal);
    printf("SCAL_HANDLE status=%d handle=%p version=%s\n", rc, (void *)scal, scal_get_version_sha1());
    if (rc || !scal) return rc ? rc : -ENOENT;
    for (unsigned i = 0; i < 16; ++i) {
        scal_pool_handle_t pool = nullptr;
        if (scal_get_pool_handle_by_id(scal, i, &pool)) continue;
        // Reserve trailing space for fields added in newer library versions.
        struct { scal_memory_pool_info info; unsigned char reserve[512]; } b{};
        if (!scal_pool_get_info(pool, &b.info))
            printf("SCAL_POOL id=%u name=%s device=0x%lx size=%lu free=%lu\n",
                   i, b.info.name, b.info.device_base_address, b.info.totalSize, b.info.freeSize);
    }
    for (unsigned id = 0; id < 1; ++id) {
        scal_core_handle_t core = nullptr;
        if (scal_get_core_handle_by_id(scal, id, &core)) continue;
        struct { scal_control_core_info_t info; unsigned char reserve[512]; } c{};
        if (scal_control_core_get_info(core, &c.info)) continue;
        printf("SCAL_CORE id=%u name=%s\n", id, c.info.name);
        for (unsigned i = 0; i < 15; ++i) {
            scal_stream_handle_t stream = nullptr;
            if (scal_get_stream_handle_by_index(core, i, &stream)) continue;
            struct { scal_stream_info_t info; unsigned char reserve[512]; } s{};
            if (!scal_stream_get_info(stream, &s.info))
                printf("SCAL_STREAM name=%s index=%u pi=%u buffer=%p direct=%d\n",
                       s.info.name, s.info.index, s.info.current_pi,
                       (void *)s.info.control_core_buffer, s.info.isDirectMode);
        }
    }
    fflush(stdout);
    return 0;
}

extern "C" int context_info(unsigned device, uint64_t *values)
{
    synDeviceInfoV2 info{};
    auto status = synDeviceGetInfoV2(device, &info);
    if (status != synSuccess) return (int)status;
    values[0] = info.fd;
    values[1] = info.sramBaseAddress;
    values[2] = info.sramSize;
    values[3] = info.dramBaseAddress;
    values[4] = info.dramSize;
    return 0;
}

// A runtime-owned stream uses Synapse scheduling rather than a bare QMAN CS.
// For the SRAM experiment the caller must first quiesce all framework work and
// use an isolated test process; this function does not reserve SRAM ownership.
extern "C" int context_runtime_copy(unsigned device, uint64_t src, uint64_t dst, uint64_t bytes)
{
    synStreamHandle stream = nullptr;
    synStatus status = synStreamCreateGeneric(&stream, device, 0);
    if (status != synSuccess) return (int)status;
    status = synMemCopyAsync(stream, src, bytes, dst, DRAM_TO_DRAM);
    if (status == synSuccess) status = synStreamSynchronize(stream);
    synStatus destroyed = synStreamDestroy(stream);
    return status == synSuccess ? (int)destroyed : (int)status;
}

// Synchronous, experimental driver-owned PDMA copy. Call only after the
// framework stream is synchronized; this does not integrate with SCAL streams.
extern "C" int context_copy(int fd, uint64_t src, uint64_t dst, uint32_t bytes)
{
#ifndef GAUDI_ALLOW_UNSUPPORTED_QUEUE_SUBMIT
    // This path timed out with active Synapse and caused an automatic reset.
    // Keep the failed experiment opt-in at build time, outside the normal probe.
    (void)fd; (void)src; (void)dst; (void)bytes;
    return -EOPNOTSUPP;
#else
    void *memory = nullptr;
    if (posix_memalign(&memory, 4096, 4096)) return -ENOMEM;
    memset(memory, 0, 4096);
    auto *packet = static_cast<packet_lin_dma *>(memory);
    packet->opcode = PACKET_LIN_DMA;
    packet->msg_barrier = 1;
    packet->tsize = bytes;
    packet->src_addr = src;
    packet->dst_addr = dst;
    uint64_t address = hlthunk_host_memory_map(fd, memory, 0, 4096);
    if (!address) { fprintf(stderr, "command map failed errno=%d\n", errno); free(memory); return -ENOMEM; }
    hl_cs_chunk chunk{};
    chunk.cb_handle = address;
    chunk.cb_size = sizeof(*packet);
    // The installed driver enables stream 0; stream 3 is disabled.
    chunk.queue_index = GAUDI2_QUEUE_ID_PDMA_1_0;
    chunk.cs_chunk_flags = HL_CS_CHUNK_FLAGS_USER_ALLOC_CB;
    hlthunk_cs_in in{};
    in.chunks_execute = &chunk; in.num_chunks_execute = 1;
    hlthunk_cs_out out{};
    int rc = hlthunk_command_submission(fd, &in, &out);
    if (rc || out.status != HL_CS_STATUS_SUCCESS)
        fprintf(stderr, "context submission rc=%d status=%u errno=%d\n", rc, out.status, errno);
    if (!rc && out.status != HL_CS_STATUS_SUCCESS) rc = -EIO;
    if (!rc) {
        uint32_t status = 0;
        rc = hlthunk_wait_for_cs(fd, out.seq, 5000000, &status);
        if (!rc && status != HL_WAIT_CS_STATUS_COMPLETED) {
            // Keep mappings alive until process teardown on a pending CS.
            fprintf(stderr, "context_copy timeout seq=%lu status=%u\n", out.seq, status);
            return -ETIMEDOUT;
        }
    }
    hlthunk_memory_unmap(fd, address);
    free(memory);
    return rc;
#endif
}

extern "C" int context_read(int fd, uint64_t src, void *host, uint32_t bytes)
{
    uint64_t address = hlthunk_host_memory_map(fd, host, 0, bytes);
    if (!address) { fprintf(stderr, "host map failed errno=%d\n", errno); return -ENOMEM; }
    int rc = context_copy(fd, src, address, bytes);
    if (rc != -ETIMEDOUT) hlthunk_memory_unmap(fd, address);
    return rc;
}

#include "runtime_pipeline.hpp"
