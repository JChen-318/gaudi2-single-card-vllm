#pragma once
#include <stdint.h>

#define MAX_NUM_ARC_CPUS 69
#define SCAL_RESERVED_SRAM_SIZE_H6 0x800  // 2K
#define SCAL_MCID_START_ID_GAUDI3 (1 * 1024)
#define SCAL_MAX_DISCARD_MCID_COUNT_GAUDI3 (55*1024)
#define SCAL_MAX_DEGRADE_MCID_COUNT_GAUDI3 (8*1024)

#ifdef __cplusplus
extern "C" {
#endif

#define SCAL_FOREVER -1

#define SCAL_SUCCESS 0
#define SCAL_FAILURE -1
#define SCAL_TIMED_OUT -2
#define SCAL_FW_FILE_LOAD_ERR -3
#define SCAL_INVALID_PARAM -4
#define SCAL_OUT_OF_MEMORY -5
#define SCAL_NOT_READY -6
#define SCAL_RESET_REQUIRED -7
#define SCAL_INVALID_CONFIG -8
#define SCAL_UNSUPPORTED_CONFIG_VERSION -9
#define SCAL_FILE_NOT_FOUND -10
#define SCAL_NOT_FOUND -11
#define SCAL_NOT_IMPLEMENTED -12
#define SCAL_EMPTY_PATH -13
#define SCAL_UNSUPPORTED_TEST_CONFIG -14

#define SCAL_HIGH_PRIORITY_STREAM 1
#define SCAL_LOW_PRIORITY_STREAM  2

#define MAX_SLAVES_PER_CQ 5

#define SCAL_NUMBER_OF_GROUPS 16 // todo: remove once usage is removed in synapse

enum ScalComputeGroups
{
    SCAL_MME_COMPUTE_GROUP,
    SCAL_TPC_COMPUTE_GROUP,
    SCAL_EDMA_COMPUTE_GROUP,
    SCAL_RTR_COMPUTE_GROUP,
    SCAL_PDMA_TX_CMD_GROUP,
    SCAL_PDMA_TX_DATA_GROUP,
    SCAL_PDMA_RX_GROUP,
    SCAL_PDMA_RX_DEBUG_GROUP,
    SCAL_PDMA_DEV2DEV_DEBUG_GROUP,
    SCAL_CME_GROUP
};

enum ScalNetworkGarbageCollectorAndReductionGroups
{
    SCAL_EDMA_NETWORK_GC_REDUCTION_GROUP0 = 2, // for BW compatibility
    SCAL_EDMA_NETWORK_GC_REDUCTION_GROUP1
};

enum ScalNetworkScaleOutReceiveGroups
{
    SCAL_NIC_RECEIVE_SCALE_OUT_GROUP,
    SCAL_PDMA_NETWORK_SCALE_OUT_RECV_GROUP,
    SCAL_EDMA_NETWORK_SCALE_OUT_RECV_GROUP0
};

enum ScalNetworkScaleOutSendGroups
{
    SCAL_NIC_SEND_SCALE_OUT_GROUP,
    SCAL_EDMA_NETWORK_SCALE_OUT_SEND_GROUP0,
    SCAL_EDMA_NETWORK_SCALE_OUT_SEND_GROUP1,
    SCAL_PDMA_NETWORK_SCALE_OUT_SEND_GROUP
};

enum ScalNetworkScaleUpReceiveGroups
{
    SCAL_NIC_RECEIVE_SCALE_UP_GROUP,
    SCAL_EDMA_NETWORK_SCALE_UP_RECV_GROUP0,
    SCAL_EDMA_NETWORK_SCALE_UP_RECV_GROUP1
};

enum ScalNetworkScaleUpSendGroups
{
    SCAL_NIC_SEND_SCALE_UP_GROUP,
    SCAL_EDMA_NETWORK_SCALE_UP_SEND_GROUP0 = 2, // for BW compatibility
    SCAL_EDMA_NETWORK_SCALE_UP_SEND_GROUP1
};

#define DECLARE_HANDLE(name) struct name##__ { int unused; }; \
                             typedef struct name##__ *name

DECLARE_HANDLE(scal_handle_t);
DECLARE_HANDLE(scal_stream_handle_t);
DECLARE_HANDLE(scal_pool_handle_t);
DECLARE_HANDLE(scal_core_handle_t);
DECLARE_HANDLE(scal_buffer_handle_t);
DECLARE_HANDLE(scal_comp_group_handle_t);
DECLARE_HANDLE(scal_host_fence_counter_handle_t);
DECLARE_HANDLE(scal_so_pool_handle_t);
DECLARE_HANDLE(scal_monitor_pool_handle_t);
DECLARE_HANDLE(scal_cluster_handle_t);
DECLARE_HANDLE(scal_streamset_handle_t);
DECLARE_HANDLE(scal_arc_fw_config_handle_t);

typedef struct _scal_stream_info_t
{
    const char * name;
    scal_core_handle_t scheduler_handle;
    unsigned index;
    unsigned type;
    unsigned current_pi;
    scal_buffer_handle_t control_core_buffer;
    unsigned submission_alignment;/* submission counter ( == PI) is 16 bits wide. the alignment
                                    represents  what is the "quanta size" how much data represents
                                    pi increment - valid only after call to scal_stream_set_commands_buffer*/
    unsigned command_alignment; /* For FW performance considerations, command cannot cross
                                /  FW buffer boundaries */
    unsigned    priority;
    bool        isDirectMode;
    uint64_t    fenceCounterAddress;
} scal_stream_info_t;

typedef struct _scal_host_fence_counter_info_t
{
    const char              * name;
    unsigned                  sm;
    unsigned                  cq_index;
    unsigned                  so_index;
    unsigned                  master_monitor_index;
    volatile const uint64_t * ctr;              // current counter value
    const uint64_t          * request_counter;  // sum of all requested credit values
    bool                      isr_enabled;      // interrupts are enabled

} scal_host_fence_counter_info_t;

typedef struct _scal_completion_group_info_t
{
    scal_core_handle_t scheduler_handle;
    unsigned           index_in_scheduler;
    unsigned           dcore;
    unsigned           sos_base;
    unsigned           sos_num;
    unsigned           long_so_dcore;
    unsigned           long_so_index;
    uint64_t           current_value;
    bool               force_order;
    unsigned           num_slave_schedulers;
    scal_core_handle_t slave_schedulers[MAX_SLAVES_PER_CQ];
    unsigned           index_in_slave_schedulers[MAX_SLAVES_PER_CQ];
} scal_completion_group_info_t;

typedef struct _scal_completion_group_infoV2_t
{
    scal_core_handle_t scheduler_handle;
    unsigned           index_in_scheduler;
    unsigned           dcore; // TODO: tbd remove
    unsigned           sm;
    uint64_t           sm_base_addr;
    unsigned           sos_base;
    unsigned           sos_num;
    unsigned           long_so_dcore; // TODO: tbd remove
    unsigned           long_so_sm;
    uint64_t           long_so_sm_base_addr;
    unsigned           long_so_index;
    uint64_t           current_value;
    uint64_t           tdr_value;
    bool               tdr_enabled;
    unsigned           tdr_sos;
    unsigned           tdr_sos_dcore; // TODO: tbd remove
    uint64_t           timeoutUs;
    bool               timeoutDisabled;
    bool               force_order;
    unsigned           num_slave_schedulers;
    scal_core_handle_t slave_schedulers[MAX_SLAVES_PER_CQ];
    unsigned           index_in_slave_schedulers[MAX_SLAVES_PER_CQ];
    bool               isDirectMode;
} scal_completion_group_infoV2_t;

typedef struct _scal_completion_group_infoV3_t
{
    scal_core_handle_t scheduler_handle;
    unsigned           index_in_scheduler;
    unsigned           sm;
    uint64_t           sm_base_addr;
    unsigned           sos_base;
    unsigned           sos_num;
    unsigned           long_so_sm;
    uint64_t           long_so_sm_base_addr;
    unsigned           long_so_index;
    uint64_t           current_value;
    uint64_t           tdr_value;
    bool               tdr_enabled;
    unsigned           tdr_sos;
    uint64_t           timeoutUs;
    bool               timeoutDisabled;
    bool               force_order;
    unsigned           force_order_inc_value;
    unsigned           num_slave_schedulers;
    scal_core_handle_t slave_schedulers[MAX_SLAVES_PER_CQ];
    unsigned           index_in_slave_schedulers[MAX_SLAVES_PER_CQ];
    bool               isDirectMode;
} scal_completion_group_infoV3_t;

typedef struct _scal_buffer_info_t
{
    scal_pool_handle_t pool;
    uint32_t core_address;        // 0 when the pool is not mapped to the cores
    void *host_address;
    uint64_t device_address;
} scal_buffer_info_t;

typedef struct _scal_control_core_info_t
{
    scal_handle_t scal;
    const char*   name;
    unsigned      idx;
    uint64_t      dccm_message_queue_address;
} scal_control_core_info_t;

typedef struct _scal_control_core_infoV2_t
{
    scal_handle_t scal;
    const char*   name;
    unsigned      idx;
    uint64_t      dccm_message_queue_address;
    unsigned      hdCore;
} scal_control_core_infoV2_t;

typedef struct _scal_control_core_debug_info_t
{
    bool      isScheduler;
    uint32_t  heartBeat;
    uint32_t* arcRegs;
    uint32_t  arcRegsSize;
    uint32_t  returnedSize;
} scal_control_core_debug_info_t;

typedef struct _scal_memory_pool_info
{
    scal_handle_t scal;
    const char * name;
    unsigned idx;
    uint64_t device_base_address;
    void *host_base_address;
    uint32_t core_base_address;  // 0 when the pool is not mapped to the cores
    uint64_t totalSize;
    uint64_t freeSize;
} scal_memory_pool_info;

typedef struct _scal_memory_pool_infoV2
{
    scal_handle_t scal;
    const char * name;
    unsigned idx;
    uint64_t device_base_address;
    void *host_base_address;
    uint32_t core_base_address;  // 0 when the pool is not mapped to the cores
    uint64_t totalSize;
    uint64_t freeSize;
    uint64_t device_base_allocated_address;
} scal_memory_pool_infoV2;

typedef struct _scal_so_pool_info
{
    scal_handle_t scal;
    unsigned smIndex;
    uint64_t smBaseAddr;
    const char* name;
    unsigned size;
    unsigned baseIdx;
    unsigned dcoreIndex; // TODO: remove
}scal_so_pool_info;

typedef struct _scal_monitor_pool_info
{
    scal_handle_t scal;
    unsigned smIndex;
    uint64_t smBaseAddr;
    const char* name;
    unsigned size;
    unsigned baseIdx;
    unsigned dcoreIndex; // TODO: remove
}scal_monitor_pool_info;

typedef struct _scal_cluster_info_t
{
    const char * name;
    scal_core_handle_t engines[MAX_NUM_ARC_CPUS];
    unsigned numEngines;
    unsigned numCompletions;
} scal_cluster_info_t;

typedef struct _scal_streamset_info_t
{
    const char * name;
    bool         isDirectMode;
    unsigned     streamsAmount;
} scal_streamset_info_t;

typedef struct _scal_sm_info_t
{
    unsigned idx;
    volatile uint32_t *objs;  // null when the SM is not mapped to the user-space memory.
    volatile uint32_t *glbl;  // null when the SM is not mapped to the user-space memory.
} scal_sm_info_t;

typedef struct _scal_sm_base_addr_tuple_t
{
    unsigned smId;
    uint64_t smBaseAddr;
    unsigned spdmaMsgBaseIndex;
} scal_sm_base_addr_tuple_t;


static const char * const internalFileSignature = ":/";


// clang-format off
// must be called before scal_init. if it's not called a default path is used
int scal_set_logs_folder(const char * logs_folder);
int scal_init(int fd, const char * config_file_path, scal_handle_t * scal, scal_arc_fw_config_handle_t fwCfg);
void scal_destroy(const scal_handle_t scal);
int scal_get_fd(const scal_handle_t scal);
int scal_get_handle_from_fd(int fd, scal_handle_t* scal);
uint32_t scal_get_sram_size(const scal_handle_t scal);

int scal_get_pool_handle_by_name(const scal_handle_t scal, const char *pool_name, scal_pool_handle_t *pool);
int scal_get_pool_handle_by_id(const scal_handle_t scal, const unsigned pool_id, scal_pool_handle_t *pool);
int scal_pool_get_info(const scal_pool_handle_t pool, scal_memory_pool_info *info);
int scal_pool_get_infoV2(const scal_pool_handle_t pool, scal_memory_pool_infoV2 *info);

int scal_get_core_handle_by_name(const scal_handle_t scal, const char *core_name, scal_core_handle_t *core);
int scal_get_core_handle_by_id(const scal_handle_t scal, const unsigned core_id, scal_core_handle_t *core);
int scal_control_core_get_info(const scal_core_handle_t core, scal_control_core_info_t *info);
int scal_control_core_get_infoV2(const scal_core_handle_t core, scal_control_core_infoV2_t *info);
int scal_control_core_get_debug_info(const scal_core_handle_t core, uint32_t* arcRegs,
                                     uint32_t arcRegsSize, scal_control_core_debug_info_t *info);

int scal_get_stream_handle_by_name(const scal_handle_t scal, const char * stream_name, scal_stream_handle_t *stream);
int scal_get_used_sm_base_addrs(const scal_handle_t scal, unsigned * num_addrs, const scal_sm_base_addr_tuple_t ** sm_base_addr_db);
int scal_get_stream_handle_by_index(const scal_core_handle_t scheduler, const unsigned index, scal_stream_handle_t *stream);
int scal_stream_set_commands_buffer(const scal_stream_handle_t stream, const scal_buffer_handle_t buffer);
int scal_stream_set_priority(const scal_stream_handle_t stream, const unsigned priority);
int scal_stream_submit(const scal_stream_handle_t stream, const unsigned pi, const unsigned submission_alignment);
// note - scal_stream_get_info should be called AFTER scal_stream_set_commands_buffer
int scal_stream_get_info(const scal_stream_handle_t stream, scal_stream_info_t *info);
int scal_stream_get_commands_buffer_alignment(const scal_stream_handle_t stream, unsigned* ccb_buffer_alignment);

int scal_get_host_fence_counter_handle_by_name(const scal_handle_t scal, const char * host_fence_counter_name, scal_host_fence_counter_handle_t *host_fence_counter);
int scal_host_fence_counter_get_info(scal_host_fence_counter_handle_t host_fence_counter, scal_host_fence_counter_info_t *info);
int scal_host_fence_counter_wait(const scal_host_fence_counter_handle_t host_fence_counter, const uint64_t num_credits, const uint64_t timeoutUs);
int scal_host_fence_counter_enable_isr(scal_host_fence_counter_handle_t host_fence_counter, bool enable_isr);

int scal_get_completion_group_handle_by_name(const scal_handle_t scal, const char * cg_name, scal_comp_group_handle_t *comp_grp);
int scal_get_completion_group_handle_by_index(const scal_core_handle_t scheduler, const unsigned index, scal_comp_group_handle_t *comp_grp);
int scal_completion_group_wait(const scal_comp_group_handle_t comp_grp, const uint64_t target, const uint64_t timeout);
int scal_completion_group_wait_always_interupt(const scal_comp_group_handle_t comp_grp, const uint64_t target, const uint64_t timeout);
int scal_completion_group_register_timestamp(const scal_comp_group_handle_t comp_grp, const uint64_t target, uint64_t timestamps_handle, uint32_t timestamps_offset);
int scal_completion_group_get_info(const scal_comp_group_handle_t comp_grp, scal_completion_group_info_t *info);
int scal_completion_group_get_infoV2(const scal_comp_group_handle_t comp_grp, scal_completion_group_infoV2_t *info);
int scal_completion_group_get_infoV3(const scal_comp_group_handle_t comp_grp, scal_completion_group_infoV3_t *info);
int scal_completion_group_set_expected_ctr(scal_comp_group_handle_t comp_grp, uint64_t val);
int scal_completion_group_get_expected_ctr(scal_comp_group_handle_t comp_grp, uint64_t* val);


int scal_get_so_pool_handle_by_name(const scal_handle_t scal, const char *pool_name, scal_so_pool_handle_t *so_pool);
int scal_so_pool_get_info(const scal_so_pool_handle_t so_pool, scal_so_pool_info *info);

int scal_get_so_monitor_handle_by_name(const scal_handle_t scal, const char *pool_name, scal_monitor_pool_handle_t *monitor_pool);
int scal_monitor_pool_get_info(const scal_monitor_pool_handle_t mon_pool, scal_monitor_pool_info *info);

int scal_get_sm_info(const scal_handle_t scal, unsigned sm_idx, scal_sm_info_t *info);

int scal_allocate_buffer(const scal_pool_handle_t pool, const uint64_t size, scal_buffer_handle_t *buff);
int scal_allocate_aligned_buffer(const scal_pool_handle_t pool, const uint64_t size, const uint64_t alignment, scal_buffer_handle_t *buff);
int scal_free_buffer(const scal_buffer_handle_t buff);
int scal_buffer_get_info(const scal_buffer_handle_t buff, scal_buffer_info_t *info);

int scal_get_cluster_handle_by_name(const scal_handle_t scal, const char *cluster_name, scal_cluster_handle_t *cluster);
int scal_cluster_get_info(const scal_cluster_handle_t cluster, scal_cluster_info_t *info);

int scal_get_streamset_handle_by_name(const scal_handle_t scal, const char *streamset_name, scal_streamset_handle_t *streamset);
int scal_streamset_get_info(const scal_streamset_handle_t streamset_handle, scal_streamset_info_t* info);

int scal_bg_work(  const scal_handle_t scal, void (*logFunc)(int, const char*));
int scal_bg_workV2(const scal_handle_t scal, void (*logFunc)(int, const char*), char *errMsg, int errMsgSize);

void scal_write_mapped_reg(volatile uint32_t * pointer, uint32_t value);
uint32_t scal_read_mapped_reg(volatile uint32_t * pointer);

struct ibv_context;
int scal_nics_db_fifos_init_and_alloc(const scal_handle_t scal, struct ibv_context* ibv_ctxt);

typedef struct _scal_ibverbs_init_params
{
    struct ibv_context * ibv_ctxt;
    uint64_t             nicsMask;
    void *               ibverbsLibHandle;
} scal_ibverbs_init_params;

struct hbldv_usr_fifo;
int scal_nics_db_fifos_init_and_allocV2(const scal_handle_t scal, const scal_ibverbs_init_params* ibvInitParams, struct hbldv_usr_fifo ** createdFifoBuffers, uint32_t * createdFifoBuffersCount);

struct hbldv_usr_fifo_attr;
int scal_get_nics_db_fifos_params(const scal_handle_t scal, struct hbldv_usr_fifo_attr* nicUserDbFifoParams, unsigned* nicUserDbFifoParamsCount);

#define SCAL_TIMEOUT_NOT_SET UINT64_MAX
typedef struct _scal_timeouts_t
{
    uint64_t   timeoutUs;
    uint64_t   timeoutNoProgressUs;
} scal_timeouts_t;

int scal_set_timeouts(const scal_handle_t scal, const scal_timeouts_t * timeouts);
int scal_get_timeouts(const scal_handle_t scal, scal_timeouts_t * timeouts);
int scal_disable_timeouts(const scal_handle_t scal, bool disableTimeouts);

// debug
uint32_t scal_debug_read_reg(const scal_handle_t scal, uint64_t reg_address);
int scal_debug_write_reg(const scal_handle_t scal, uint64_t reg_address, uint32_t reg_value);
int scal_debug_memcpy(const scal_handle_t scal, uint64_t src, uint64_t dst, uint64_t size);
unsigned scal_debug_stream_get_curr_ci(const scal_stream_handle_t stream);
int scal_debug_background_work(const scal_handle_t scal);

const char* scal_get_version_sha1();
const char* scal_get_build_time();
// clang-format on

#ifdef __cplusplus
}
#endif
