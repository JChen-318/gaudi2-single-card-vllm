/* SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2020 HabanaLabs Ltd.
 * All Rights Reserved.
 */

#ifndef __GAUDI2_ARC_SCHED_PACKETS_H__
#define __GAUDI2_ARC_SCHED_PACKETS_H__

#include <stdint.h>
#include "gaudi2_arc_eng_packets.h"
/**
 * \file    gaudi2_arc_sched_packets.h
 * \brief   QMAN Command definitions for Host.
 *          This defines data structures to be used by Host to submit commands
 */

/**
 * \enum    sched_compute_arc_cmd_opcode_t
 * \brief   commands opcodes for scheduler ARC
 * \details Opcodes processed by the scheduler ARC
 */
enum sched_compute_arc_cmd_opcode_t {
	SCHED_COMPUTE_ARC_CMD_FENCE_WAIT = 0,
	SCHED_COMPUTE_ARC_CMD_LBW_WRITE = 1,
	SCHED_COMPUTE_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER = 3,
	SCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE = 4,
	SCHED_COMPUTE_ARC_CMD_NOP = 5,
	SCHED_COMPUTE_ARC_CMD_ALLOC_BARRIER_V2 = 6,
	SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3 = 7,
	SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER = 8,
	SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2 = 9,
	SCHED_COMPUTE_ARC_CMD_WAIT_FOR_EXT_SIGNAL = 10,
	SCHED_COMPUTE_ARC_CMD_LBW_READ = 11,			// TODO: Only in Debug
	SCHED_COMPUTE_ARC_CMD_MEM_FENCE = 12,			// TODO: Only in Debug
	SCHED_COMPUTE_ARC_CMD_COUNT = 13,
	SCHED_COMPUTE_ARC_CMD_SIZE = 0x1F
};
/**
 * \enum    sched_scaleup_send_arc_cmd_opcode_t
 * \brief   scaleup send sched commands opcodes
 * \details Opcodes processed by the scaleup send scheduler ARC
 */
enum sched_scaleup_send_arc_cmd_opcode_t {
	SCHED_SCALEUP_SEND_ARC_CMD_FENCE_WAIT = 0,
	SCHED_SCALEUP_SEND_ARC_CMD_LBW_WRITE = 1,
	SCHED_SCALEUP_SEND_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_SCALEUP_SEND_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_SCALEUP_SEND_ARC_CMD_NOP = 4,
	SCHED_SCALEUP_SEND_ARC_CMD_UPDATE_NIC_GLBL_CTXT = 5,
	SCHED_SCALEUP_SEND_ARC_CMD_UPDATE_NIC_COLL_CTXT = 6,
	SCHED_SCALEUP_SEND_ARC_CMD_NIC_COLL_OPS = 7,
	SCHED_SCALEUP_SEND_ARC_CMD_ALLOC_NIC_BARRIER = 8,
	SCHED_SCALEUP_SEND_ARC_CMD_NIC_PASSTHROUGH = 9,
	SCHED_SCALEUP_SEND_ARC_CMD_NIC_EDMA_OPS = 10,
	SCHED_SCALEUP_SEND_ARC_CMD_LBW_READ = 11,		// TODO: Only in Debug
	SCHED_SCALEUP_SEND_ARC_CMD_MEM_FENCE = 12,		// TODO: Only in Debug
	SCHED_SCALEUP_SEND_ARC_CMD_COUNT = 13,
	SCHED_SCALEUP_SEND_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_scaleup_recv_arc_cmd_opcode_t
 * \brief   scaleup recv sched commands opcodes
 * \details Opcodes processed by the scaleup recv scheduler ARC
 */
enum sched_scaleup_recv_arc_cmd_opcode_t {
	SCHED_SCALEUP_RECV_ARC_CMD_FENCE_WAIT = 0,
	SCHED_SCALEUP_RECV_ARC_CMD_LBW_WRITE = 1,
	SCHED_SCALEUP_RECV_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_SCALEUP_RECV_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_SCALEUP_RECV_ARC_CMD_NOP = 4,
	SCHED_SCALEUP_RECV_ARC_CMD_UPDATE_NIC_GLBL_CTXT = 5,
	SCHED_SCALEUP_RECV_ARC_CMD_UPDATE_NIC_COLL_CTXT = 6,
	SCHED_SCALEUP_RECV_ARC_CMD_NIC_COLL_OPS = 7,
	SCHED_SCALEUP_RECV_ARC_CMD_ALLOC_NIC_BARRIER = 8,
	SCHED_SCALEUP_RECV_ARC_CMD_NIC_PASSTHROUGH = 9,
	SCHED_SCALEUP_RECV_ARC_CMD_LBW_READ = 10,		// TODO: Only in Debug
	SCHED_SCALEUP_RECV_ARC_CMD_MEM_FENCE = 11,		// TODO: Only in Debug
	SCHED_SCALEUP_RECV_ARC_CMD_COUNT = 12,
	SCHED_SCALEUP_RECV_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_scaleout_send_arc_cmd_opcode_t
 * \brief   Scaleout send sched commands opcodes
 * \details Opcodes processed by the Scaleout send Scheduler ARC
 */
enum sched_scaleout_send_arc_cmd_opcode_t {
	SCHED_SCALEOUT_SEND_ARC_CMD_FENCE_WAIT = 0,
	SCHED_SCALEOUT_SEND_ARC_CMD_LBW_WRITE = 1,
	SCHED_SCALEOUT_SEND_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_SCALEOUT_SEND_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_SCALEOUT_SEND_ARC_CMD_NOP = 4,
	SCHED_SCALEOUT_SEND_ARC_CMD_ALLOC_NIC_BARRIER = 5,
	SCHED_SCALEOUT_SEND_ARC_CMD_NIC_COLL_OPS = 6,
	SCHED_SCALEOUT_SEND_ARC_CMD_UNUSED = 7, /* Unused */
	SCHED_SCALEOUT_SEND_ARC_CMD_UPDATE_NIC_GLBL_CTXT = 8,
	SCHED_SCALEOUT_SEND_ARC_CMD_PDMA_BATCH_TRANSFER = 9,
	SCHED_SCALEOUT_SEND_ARC_CMD_LBW_READ = 10,		// TODO: Only in Debug
	SCHED_SCALEOUT_SEND_ARC_CMD_MEM_FENCE = 11,		// TODO: Only in Debug
	SCHED_SCALEOUT_SEND_ARC_CMD_COUNT = 12,
	SCHED_SCALEOUT_SEND_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_scaleout_recv_arc_cmd_opcode_t
 * \brief   Scaleout Recv sched commands opcodes
 * \details Opcodes processed by the Scaleout Recv Scheduler ARC
 */
enum sched_scaleout_recv_arc_cmd_opcode_t {
	SCHED_SCALEOUT_RECV_ARC_CMD_FENCE_WAIT = 0,
	SCHED_SCALEOUT_RECV_ARC_CMD_LBW_WRITE = 1,
	SCHED_SCALEOUT_RECV_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_SCALEOUT_RECV_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_SCALEOUT_RECV_ARC_CMD_NOP = 4,
	SCHED_SCALEOUT_RECV_ARC_CMD_ALLOC_NIC_BARRIER = 5,
	SCHED_SCALEOUT_RECV_ARC_CMD_NIC_COLL_OPS = 6,
	SCHED_SCALEOUT_RECV_ARC_CMD_UNUSED = 7,
	SCHED_SCALEOUT_RECV_ARC_CMD_UPDATE_NIC_GLBL_CTXT = 8,
	SCHED_SCALEOUT_RECV_ARC_CMD_PDMA_BATCH_TRANSFER = 9,
	SCHED_SCALEOUT_RECV_ARC_CMD_NIC_EDMA_OPS = 10,
	SCHED_SCALEOUT_RECV_ARC_CMD_LBW_READ = 11,		// TODO: Only in Debug
	SCHED_SCALEOUT_RECV_ARC_CMD_MEM_FENCE = 12,		// TODO: Only in Debug
	SCHED_SCALEOUT_RECV_ARC_CMD_COUNT = 13,
	SCHED_SCALEOUT_RECV_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_gc_reduction_arc_cmd_opcode_t
 * \brief   garbage collection and reduction sched
 *          commands opcodes
 * \details Opcodes processed by the garbage collection and
 *          reduction Scheduler ARC
 */
enum sched_gc_reduction_arc_cmd_opcode_t {
	SCHED_GC_REDUCTION_ARC_CMD_FENCE_WAIT = 0,
	SCHED_GC_REDUCTION_ARC_CMD_LBW_WRITE = 1,
	SCHED_GC_REDUCTION_ARC_CMD_LBW_BURST_WRITE = 2,
	SCHED_GC_REDUCTION_ARC_CMD_FENCE_INC_IMMEDIATE = 3,
	SCHED_GC_REDUCTION_ARC_CMD_NOP = 4,
	SCHED_GC_REDUCTION_ARC_CMD_ALLOC_NIC_BARRIER = 5,
	SCHED_GC_REDUCTION_ARC_CMD_NIC_EDMA_OPS = 6,
	SCHED_GC_REDUCTION_ARC_CMD_LBW_READ = 7,		// TODO: Only in Debug
	SCHED_GC_REDUCTION_ARC_CMD_MEM_FENCE = 8,		// TODO: Only in Debug
	SCHED_GC_REDUCTION_ARC_CMD_COUNT = 9,
	SCHED_GC_REDUCTION_ARC_CMD_SIZE = 0x1F
};

/**
 * \enum    sched_compute_arc_stream_state_t
 * \brief   scheduler stream states
 * \details scheduler stream states, used by profiler to print events
 *	    These enums are for internal usage of scheduler firmware and
 *	    kept here to prepare profiler event table. This should not be
 *	    used by any other components other than ARC firmware.
 */
enum sched_compute_arc_stream_state_t {
	SCHED_COMPUTE_STREAM_IDLE = 0,
	SCHED_COMPUTE_STREAM_DMA_STARTED = 1,
	SCHED_COMPUTE_STREAM_PROCESSING_CMD = 2,
	SCHED_COMPUTE_STREAM_END_OF_CCB = 3,
	SCHED_COMPUTE_STREAM_SUSPENDED = 4,
	SCHED_COMPUTE_STREAM_STATE_COUNT = 5,
	SCHED_COMPUTE_STREAM_STATE_SIZE = 0xFF
};

/**
 * \struct  qman_cmd_nop_t
 * \brief   Stop Message
 * \details Command structure for Stop message
 */
struct sched_arc_cmd_nop_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_NOP */
	uint32_t padding_count:27;
	/**<
	 * number of padded DWORDs(32bits) at the end of the
	 * command for alignment purpose
	 */
	uint32_t padding[0];
	/**<
	 * Padding to align with 256 Bytes command buffer
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  qman_cmd_dispatch_compute_ecb_list_t
 * \brief   Dispatch Compute ECB list command
 * \details Command structure for Dispatch Compute ECB list
 */
struct sched_arc_cmd_dispatch_compute_ecb_list_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST */
	uint32_t reserved:2;
	/**<
	 * Reserved
	 */
	uint32_t cmpt_stream_index:2;
	/**<
	 * Compute stream index that should be used for SFG feature
	 * Firmware receives SOB IDs from SCAL using Engine config
	 * for each of the compute streams that should be used to signal from
	 * the graph
	 */
	uint32_t use_gc_nop:1;
	/**<
	 * use GC provided NOP kernel for TPC
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Various engine types supported by hardware
	 */
	uint32_t single_static_chunk:1;
	/**<
	 * This bit indicates static ecb list contains only one chunk
	 */
	uint32_t single_dynamic_chunk:1;
	/**<
	 * This bit indicates dynamic ecb list contains only one chunk
	 */
	uint32_t reserved1:16;
	/**<
	 * reserved
	 */
	struct {
		uint32_t static_ecb_list_offset:16;
		/**<
		 * Address offset of Static ECB List in memory with respect to
		 * dynamic ecb list
		 * This is in multiple of STATIC_COMPUTE_ECB_LIST_BUFF_SIZE
		 */
		uint32_t static_ecb_list_eng_offset:16;
		/**<
		 * Static ECB List Engine specific offset
		 * This is in multiple of STATIC_COMPUTE_ECB_LIST_BUFF_SIZE
		 */
	} __attribute__ ((aligned(4), __packed__));
	struct {
		uint32_t reserved2:8;
		/**<
		 * reserved
		 */
		uint32_t dynamic_ecb_list_addr_256:24;
		/**<
		 * Address of ECB List in memory
		 * This is STATIC_COMPUTE_ECB_LIST_BUFF_SIZE bytes aligned
		 */
	} __attribute__ ((aligned(4), __packed__));
} __attribute__ ((aligned(4), __packed__));

/**
 * \enum    sched_mon_exp_opcode_t
 * \brief   Opcodes for monitor expiration messages
 * \details Opcodes used by firmware for monitor expiration messages
 */
enum sched_mon_exp_opcode_t {
	MON_EXP_FENCE_UPDATE = 0,
	/**<
	 * To update firmware fence counter
	 */
	MON_EXP_COMP_FENCE_UPDATE = 1,
	/**<
	 * to indicate completion on the completion group
	 */
	MON_EXP_SO_SET_RESET = 2,
	/**<
	 * used internally by firmware
	 */
	MON_EXP_UPDATE_Q_CREDIT = 3,
	/**<
	 * used internally by firmware
	 */
	MON_EXP_MSG_COUNT = 4
	/**<
	 * Total number of expiration messages
	 */
};

/**
 * \struct  sched_mon_exp_generic_t
 * \brief   Generic message structure
 * \details Structure describes common fields for all the types of messages
 */
struct sched_mon_exp_generic_t {
	uint32_t opcode:2;
	/**<
	 * opcode
	 */
	uint32_t reserved:30;
	/**<
	 * reserved
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_mon_exp_fence_t
 * \brief   Monitor expiration structure to update fence
 * \details structure used for updating global fence counters
 */
struct sched_mon_exp_fence_t {
	union {
		struct {
			uint32_t opcode:2;
			/**<
			 * opcode : MON_EXP_FENCE_UPDATE
			 */
			uint32_t fence_id:6;
			/**<
			 * Index of the fence counter
			 */
			uint32_t reserved:24;
			/**<
			 * reserved
			 */
		} __attribute__ ((aligned(4), __packed__));
		uint32_t raw;
		/**<
		 * Raw field
		 */
	};
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_mon_exp_comp_fence_t
 * \brief   Monitor expiration structure to update fence counters in completion
 *	    groups
 * \details structure used for updating global fence counters which are part of
 *	    completion groups
 */
struct sched_mon_exp_comp_fence_t {
	union {
		struct {
			uint32_t opcode:2;
			/**<
			 * opcode : MON_EXP_COMP_FENCE_UPDATE
			 */
			uint32_t comp_group_index:4;
			/**<
			 * Index of the completion group
			 */
			uint32_t mon_id:11;
			/**<
			 * Monitor ID
			 */
			uint32_t mon_sm_id:2;
			/**<
			 * Monitor Sync Manager Instance ID
			 */
			uint32_t mon_index:2;
			/**<
			 * local index of the monitor in the completion group
			 * Starts from 0 to max (COMP_SYNC_GROUP_MAX_MON_GROUP_COUNT - 1)
			 */
			uint32_t update_slave_credit:1;
			/**<
			 * Indicates that this message is coming from Slaves
			 * to synchronize the SOB credit with Master
			 */
			uint32_t reserved:10;
			/**<
			 * reserved
			 */
		} __attribute__ ((aligned(4), __packed__));
		uint32_t raw;
		/**<
		 * Raw field
		 */
	};
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_mon_exp_update_q_credit_t
 * \brief   DCCM Queue Flow control message
 * \details Message that is posted when half of the DCCM Queue is consumed
 *	    by the Engine ARCs.
 */
struct sched_mon_exp_update_q_credit_t {
	union {
		struct {
			uint32_t opcode:2;
			/**<
			 * MON_EXP_UPDATE_Q_CREDIT
			 */
			uint32_t engine_group_type:4;
			/**<
			 * Engine Group type from scal.h file
			 */
			uint32_t :26;
			/**<
			 * Not used
			 */
		} __attribute__ ((aligned(4), __packed__));
		uint32_t raw;
		/**<
		 * Raw field
		 */
	};
} __attribute__ ((aligned(4), __packed__));

/**
 * \union  sched_mon_exp_msg_t
 * \brief   Various monitor expiration messages
 * \details Various monitor expiration messages shared used by Firmware and
 *	    software stack.
 */
union sched_mon_exp_msg_t {
	struct sched_mon_exp_generic_t generic;
	/**<
	 * generic message
	 */
	struct sched_mon_exp_fence_t fence;
	/**<
	 * monitor expiration message used for global fence counters
	 */
	struct sched_mon_exp_comp_fence_t comp_fence;
	/**<
	 * monitor expiration message used for fence counters of completion
	 * groups
	 */
	uint32_t raw;
	/**<
	 * raw bitfield
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_pdma_commands_params_t
 * \brief   PDMA batch transfer parameters
 * \details This structure provides required PDMA transfer parameters
 *	    for each transfer.
 */
struct sched_arc_pdma_commands_params_t {
	uint32_t transfer_size;
	/**<
	 * Transfer size in bytes
	 */
	uint64_t src_addr;
	/**<
	 * Source address
	 */
	uint64_t dst_addr;
	/**<
	 * Destination address
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * Maximum number of PDMA transfers that can be done
 * using single sched_arc_cmd_pdma_batch_transfer_t
 * command
 */
#define SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT	4

/**
 * \enum    sched_arc_pdma_dir_t
 * \brief   enums for PDMA direction
 * \details enums for PDMA direction
 */
enum sched_arc_pdma_dir_t {
	PDMA_DIR_DEV2HOST = 0,
	PDMA_DIR_HOST2DEV = 1,
	PDMA_DIR_DEV2DEV = 2,
	PDMA_DIR_COUNT = 3
};

/**
 * \struct  sched_arc_cmd_pdma_batch_transfer_t
 * \brief   PDMA commands transfer
 * \details This command allows multiple PDMA transfers to be submitted together.
 */
struct sched_arc_cmd_pdma_batch_transfer_t {
	struct {
		uint32_t opcode:5;
		/**<
		 * opcode of the command = SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER
		 */
		uint32_t engine_group_type:4;
		/**<
		 * PDMA Tx command group
		 */
		uint32_t watch_dog_sig_value:1;
		/**<
		 * Value to be used by firmware to increment the watch
		 * dog SOB
		 */
		uint32_t signal_to_cg:1;
		/**<
		 * 0 - depending on the ‘has_payload’ field, signal the payload
		 *     provided by the user to the address provided
		 * 1 - set the PDMA to write its completion message so it increments
		 *     the current SO in the CG completion group by 1
		 * When this bit is set, pay_addr field contains a 4 bit completion
		 * group index.
		 */
		uint32_t :21;
	};
	struct {
		uint32_t workload_type:4;
		/**<
		 * Refer to eng_pdma_arc_cmd_t
		 */
		uint32_t stream_ctxt_id:8;
		/**<
		 * ContextId for the stream
		 * [1:0] : sched_arc_pdma_dir_t
		 * [7:2] : Used only by profiler not used by Firmware
		 */
		uint32_t api_id:5;
		/**<
		 * API ID of the command which is used to tie Host API and H/W events
		 */
		uint32_t has_payload:1;
		/**<
		 * 0 - Does not have valid payload to write back on completion
		 * 1 - Has valid payload to write back on completion
		 */
		uint32_t batch_count:3;
		/**<
		 * Number of valid transfer parameters in the transfer_params
		 * Maximum number of transfers is limited to
		 * SCHED_PDMA_MAX_BATCH_TRANSFER_COUNT
		 */
		uint32_t memset:1;
		/**<
		 * Valid only for memcopy
		 */
		uint32_t reduction_ind:1;
		/**<
		 * Reduction indication
		 */
		uint32_t reduction_op:2;
		/**<
		 * Reduction operation to be performed
		 */
		uint32_t reduction_dtype:4;
		/**<
		 * Reduction data type
		 * 0xC- upscaling FP16
		 * 0xD- upscaling BF16
		 */
		uint32_t :3;
		/**<
		 * unused
		 */
	};
	uint32_t pay_data;
	/**<
	 * payload data, i.e. 32bit data that needs to be written when the
	 * transfer completes
	 */
	uint32_t pay_addr;
	/**<
	 * when signal_to_cg is 0, then this field specifies a
	 * payload address, i.e. address where the data needs to be
	 * written when the transfer completes, higher 32bits of the LBW
	 * address is harcoded in the firmware, so only lower32 bit LBW
	 * address is passed here
	 *
	 * When signal_to_cg is 1, this field contains a
	 * 4 bit completion group index, to which the PDMA completion
	 * is to be signalled. Index ranges between 0 to
	 * (COMP_GROUP_COUNT - 1)
	 */
	struct sched_arc_pdma_commands_params_t batch_params[0];
	/**<
	 * transfer parameter for each batch
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \enum    sched_pdma_operation_t
 * \brief   Which PDMA operation to perform
 * \details Opcodes for PDMA operation
 */
enum sched_pdma_operation_t {
	PDMA_OPERATION_MEMCOPY = 0,
	PDMA_OPERATION_MEMSET = 1,
	PDMA_OPERATION_CAST_DOWN = 2,
	PDMA_OPERATION_CAST_UP = 3,
	PDMA_OPERATION_SIZE = 4
};

/**
 * \struct  sched_arc_cmd_alloc_barrier_v2_t
 * \brief   allocate barrier
 * \details allocate required barrier resources so that barrier can be sent
 */
struct sched_arc_cmd_alloc_barrier_v2_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_ALLOC_BARRIER_V2 */
	uint32_t comp_group_index:4;
	/**<
	 * Completion Group index. Index bettween 0 to
	 * (COMP_SYNC_GROUP_COUNT - 1)
	 */
	uint32_t target_value:14;
	/**<
	 * Target value of the SOB
	 */
	uint32_t rel_so_set:1;
	/**<
	 * when set, releases and recycles the current SOs set.
	 */
	uint32_t allocate_so_set:1;
	/**<
	 * when set, an SO set is allocated.
	 * Setting this bit can avoid sending an explicit command for
	 * SO SET allocation.
	 * Here are the possible combos:
	 * allocate_so_set | rel_so_set |           comments
	 *          0            0          No SO SET will be alloted/freed
	 *          0            1          user has to send an explicit SO_SET_ALLOC, which will be FREED after Barrier
	 *          1            0          SO SET will be allocated but will never be FREED
	 *          1            1          Single command handles BARRIER + SO SET allocation + FREEing
	 */
	uint32_t b2b_exec_with_no_fence:1;
	/**<
	 * 0: Default: B2B List Execution is FENCED via SM for all-Engine idleness
	 * 1: B2B List Execution will happen with no FENCE
	 */
	uint32_t reserved:2;
	/**<
	 * reserved
	 */
	uint32_t num_engine_group_type:4;
	/**<
	 * Number of engine group types to which the alloc barrier
	 * needs to be sent to.
	 * when set, this command inserts QMAN FENCE_WAIT command
	 * in the QMAN interface to wait for the previous dispatch
	 * to be over.
	 */
	uint8_t engine_group_type[4];
	/**<
	 * Array of engine group types to which the alloc barrier
	 * needs to be sent
	 */
	uint32_t watch_dog_sig_value:15;
	/**<
	 * Value to be used by firmware to increment the watchdog SOB.
	 * Watchdog SOB ID is sent as part of Engine Config.
	 */
	uint32_t reserved2:17;
	/**<
	 * reserved
	 */
} __attribute__ ((aligned(4), __packed__));

/**<
 * Max number of fence IDs that can be accomodated in a single fence_id_arr element
 */
#define FENCE_ID_COUNT_IN_ARR_ELEMENT	4

/**<
 * Max fence count that can be accomodated in a single alloc nic barrier command
 */
#define MAX_FENCE_COUNT_IN_ALLOC_NIC_BARRIER	7

/**
 * \struct  sched_arc_fence_id_arr_t
 * \brief   array of fence IDs used in Alloc nic barrier command
 * \details fence Ids that are sent to the device as a part of alloc nic
 *          barrier command are sent via the above structure. These fence
 *          ids are incremented
 */
struct sched_arc_fence_id_arr_t {
	uint8_t fence_id[FENCE_ID_COUNT_IN_ARR_ELEMENT];
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_lbw_write_t
 * \brief   structure to store address and data for lbw write
 * \details address and data to be written through the lbw
 */
struct sched_arc_lbw_write_t {
	uint32_t addr;
	/**<
	 * destination LBU address
	 * Note: its LBU address and not LBW, it should start with
	 * 0xFxxxxxx
	 */
	uint32_t data;
	/**<
	 * source data to be written
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_alloc_nic_barrier_t
 * \brief   allocate NIC barrier
 * \details allocate required barrier resources so that barrier can be sent
 *          to NIC engines
 */
struct sched_arc_cmd_alloc_nic_barrier_t {
	union {
		struct {
			uint32_t opcode:5;
			/**< opcode of the command = SCHED_ARC_CMD_ALLOC_NIC_BARRIER */
			uint32_t comp_group_index:4;
			/**<
			 * Completion Group index. Index bettween 0 to
			 * (COMP_GROUP_COUNT - 1)
			 */
			uint32_t fence_count:4;
			/**<
			 * No of fences that needs to be incremented.
			 * Count can be between 0 to 7
			 */
			uint32_t required_sobs:7;
			/**<
			 * required number of sobs to be allocated
			 */
			uint32_t cmd_size_dwords:6;
			/**<
			 * Size of command in bytes
			 */
			uint32_t num_lbw_write:5;
			/**<
			 * Number of lbw writes
			 */
			uint32_t rsvd:1;
			/**<
			 * Reserved
			 */
		} __attribute__ ((aligned(4), __packed__));
		uint32_t raw_dword0;
	};
	struct sched_arc_fence_id_arr_t fence_arr[0];
	/**<
	 * array of fence Ids. Each element can contain upto 4 fence IDs
	 */
	struct sched_arc_lbw_write_t lbw_addr_data[0];
	/**
	 * array of data for LBW burst write operation
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_dispatch_barrier_t
 * \brief   Dispatch barrier
 * \details Dispatches barrier to an array of engine group types
 */
struct sched_arc_cmd_dispatch_barrier_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_DISPATCH_BARRIER */
	uint32_t watch_dog_sig_value:15;
	/**<
	 * Value to be used by firmware to increment the watch
	 * dog SOB
	 */
	uint32_t reserved:4;
	/**< reserved */
	uint32_t num_engine_group_type:8;
	/**<
	 * Number of engine group types to which the dispatch needs
	 * to be sent
	 */
	uint8_t engine_group_type[4];
	/**<
	 * Array of engine group types to which the dispatch needs
	 * to be sent
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_fence_wait_t
 * \brief   Wait for fence credit
 * \details Checks if the stream fence counter in the specified index has
 *          reached a certain amount of credits. If the stream already has
 *          enough credits it continues, otherwise it blocks until the
 *          required credits are received.
 *          Stream status after the execution:
 *          - The target value is subtracted from to the current value of the
 *          fence counter.
 *          - If the value (after the substruction) is negative the stream is
 *          suspended.
 *          - If the value is non-negative, the stream continues.
 */
struct sched_arc_cmd_fence_wait_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_FENCE_WAIT */
	uint32_t fence_id:6;
	/**< Index of the fence register of this stream */
	uint32_t :1;
	uint32_t target:6;
	/**< target value of the fence */
	uint32_t reserved:14;
	/**< reserved */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_fence_inc_immediate_t
 * \brief   Increment fence counters by 1
 * \details Increment fence counters by 1 specified in the array.
 */
struct sched_arc_cmd_fence_inc_immediate_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_FENCE_INC_IMMEDIATE */
	uint32_t reserved1:3;
	/**< reserved */
	uint32_t fence_count:8;
	/**< number of valid fence indexes in the array */
	uint32_t fence_index:6;
	/**< fence index value */
	uint32_t reserved2:10;
	/**< reserved */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_lbw_read_t
 * \brief   Reads a mem/register location using LBW interface
 * \details copy a range of registers from the LBW space to the destination
 *          buffer.
 */
struct sched_arc_cmd_lbw_read_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_LBW_READ */
	uint32_t reserved:27;
	/**<
	 * Reserved
	 */
	uint32_t src_addr;
	/**<
	 * source LBU address
	 * Note: its LBU address and not LBW, it should start with
	 * 0xFxxxxxx
	 */
	uint32_t dst_addr;
	/**<
	 * destination address
	 */
	uint32_t size;
	/**<
	 * size in bytes
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_sem_lbw_write_t
 * \brief   Write into mem/register location using LBW interface
 * \details write a range of values into destination LBW address.
 */
struct sched_arc_cmd_lbw_write_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_LBW_WRITE */
	uint32_t fence:1;
	/**<
	 * flag to indicate if firmware needs to do fence wait
	 */
	uint32_t fence_id:6;
	/**<
	 * fence id on which fence wait needs to be done. Valid only if
	 * fence is true
	 */
	uint32_t target:6;
	/**< target value of the fence */
	uint32_t reserved:14;
	/**<
	 * Reserved
	 */
	uint32_t dst_addr;
	/**<
	 * destination LBU address
	 * Note: its LBU address and not LBW, it should start with
	 * 0xFxxxxxx
	 */
	uint32_t src_data;
	/**<
	 * source data to be written
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_sem_lbw_burst_write_t
 * \brief   Write into mem/register location using LBW interface
 * \details write 4 DWORDs into destination LBW address.
 */
struct sched_arc_cmd_lbw_burst_write_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_LBW_BURST_WRITE */
	uint32_t num_lbw_write:5;
	/**<
	 * Number of lbw writes
	 */
	uint32_t reserved:20;
	/**<
	 * Reserved
	 */
	uint32_t block_next:1;
	/**<
	 * Block execution of next command by putting the stream into
	 * suspended state
	 */
	uint32_t wait_for_completion:1;
	/**<
	 * Wait until current write is completed
	 */
	struct {
		uint32_t addr;
		/**<
		 * destination LBU address
		 * Note: its LBU address and not LBW, it should start with
		 * 0xFxxxxxx
		 */
		uint32_t data;
		/**<
		 * source data to be written
		 */
	} addr_data[0];

} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_mem_fence_t
 * \brief   waits until mem transactions are completed
 * \details waits (in busy mode) until all the ARC memory transactions,
 *          DUP memory transactions and DMA memory transactions are done.
 */
struct sched_arc_cmd_mem_fence_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_MEM_FENCE */
	uint32_t arc:1;
	/**<
	 * wait until all the ARC transactions are completed
	 */
	uint32_t dup_eng:1;
	/**<
	 * wait until DUP engine is idle
	 */
	uint32_t arc_dma:1;
	/**<
	 * wait until ARC DMA engine is idle
	 */
	uint32_t reserved:24;
	/**<
	 * reserved
	 */
} __attribute__ ((aligned(4), __packed__));

/**<
 * Max number of engine groups in update recipe cmd
 */
#define SCHED_CMD_UPDATE_RECIPE_MAX_ENG_GROUPS 4

/**
 * \struct  sched_arc_cmd_update_recipe_base_v2_t
 * \brief   Update receipe base address
 * \details Update a particular recipe base address registers pointed by
 *          base index parameters
 */
struct sched_arc_cmd_update_recipe_base_v2_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_UPDATE_RECIPE_BASE_V2 */
	uint32_t num_recipe_addrs:3;
	/**<
	 * Number of recipe base addresses
	 */
	uint32_t num_engine_group_type:8;
	/**<
	 * Number of engine group types to which the dispatch needs
	 * to be sent
	 */
	union {
		struct {
			uint16_t recipe_base_index0:4;
	 		/**<
			 * Index of the base register-0 which needs to be updated
			 */
			uint16_t recipe_base_index1:4;
	 		/**<
			 * Index of the base register-1 which needs to be updated
			 */
			uint16_t recipe_base_index2:4;
	 		/**<
			 * Index of the base register-2 which needs to be updated
			 */
			uint16_t recipe_base_index3:4;
	 		/**<
			 * Index of the base register-3 which needs to be updated
			 */
		};
		uint16_t recipe_base_indices;
	};
	/**<
	 * Index 0 to 6 are used for Static and Dynamic descriptors
	 * Index 7 is used for storing the GC provided TPC NOP kernel
	 */
	uint8_t engine_group_type[SCHED_CMD_UPDATE_RECIPE_MAX_ENG_GROUPS];
	/**<
	 * Array of engine group types to which the dispatch needs
	 * to be sent
	 */
	uint64_t recipe_base_addr[0];
	/**<
	 * Base address values
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_update_nic_glbl_ctxt_t
 * \brief   Update NIC global context
 * \details This command is used to update a NIC global context
 */
struct sched_arc_cmd_update_nic_glbl_ctxt_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_ARC_CMD_UPDATE_NIC_GLBL_CTXT
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Engine group type to which this command needs to be
	 * send
	 */
	uint32_t reserved:7;
	/**<
	 * Reserved
	 */
	uint32_t cmd_size:16;
	/**<
	 * scheduler command size in bytes
	 * Note: This includes the scheduler header as well
	 */
	union {
		/*
		 * Engine ARC header for update global context command
		 * arc_cmd_update_glbl_ctxt_t is defined in
		 * gaudi2_arc_eng_packets.h as this is used by both
		 * engine arc and scheduler arc
		 */
		struct arc_cmd_update_glbl_ctxt_t cmd_update_glbl_ctxt;
		struct arc_scaleout_cmd_update_glbl_ctxt_t scaleout_cmd_update_glbl_ctxt;
		uint32_t dword0;
	};
	uint32_t so_lbw_address;
	/**<
	 * The address of an SO to increment by one once the global
	 * context update is complete.
	 */
	struct nic_glbl_ctxt_t glbl_ctxt[0];
	/**<
	 * This variable size array contains one or more nic_glbl_ctxt_t
	 * structure. size is given by num_dwords.
	 * All the NICs receive all the contexts but they pick only their own
	 * context from this array using CPU ID.
	 */
	struct nic_glbl_ctxt_v2_t glbl_ctxt_v2[0];
	/**<
	 * Contains infomration related to buffers which are used by NICs
	 * in reproduceable reduction
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_update_nic_coll_ctxt_t
 * \brief   Update NIC collective context
 * \details This command is used to update a particular NIC collective
 * 	    context
 */
struct sched_arc_cmd_update_nic_coll_ctxt_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_ARC_CMD_UPDATE_NIC_COLL_CTXT
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Engine group type to which this command needs to be
	 * send
	 */
	uint32_t reserved:23;
	/**<
	 * Reserved
	 */
	union {
		/*
		 * Engine ARC header for update collective context command
		 * arc_cmd_update_glbl_ctxt_t is defined in
		 * gaudi2_arc_eng_packets.h as this is used by both
		 * engine arc and scheduler arc
		 */
		struct arc_cmd_update_coll_ctxt_t cmd_update_coll_ctxt;
		uint32_t dowrd0;
	};
	struct nic_coll_ctxt_dword_t dwords[0];
	/**<
	 * Array of DWORDs to be updated in this command
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_nic_coll_ops_t
 * \brief   NIC collective operation to be performed
 * \details This command is used to inform NICs to perform
 * 	    a collective operation
 */
struct sched_arc_cmd_nic_coll_ops_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_ARC_CMD_NIC_COLL_OPS
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Context ID, that needs to be updated by this command
	 */
	uint32_t cmd_size:15;
	/**<
	 * Command Size in Bytes, including the size field
	 */
	uint32_t reserved:8;
	/**<
	 * Reserved
	 */
	struct arc_cmd_coll_ops_short_t cmd_coll_ops_short[0];
	struct arc_cmd_coll_ops_recv_short_inorder_v2_t cmd_coll_ops_short_inorder_v2[0];
	struct arc_cmd_coll_ops_long_t cmd_coll_ops_long[0];
} __attribute__ ((aligned(4), __packed__));

struct nic_passthrough_data_t {
	union {
		struct {
			uint32_t dup_mask:21;
			/**<
			 * value to write at DUP_NIC_ENG_MASK register to tell
			 * which eng should get broadcast
			 */
			uint32_t num_payload_dwords:1;
			/**<
			 * Number of dwords in payload:
			 * - 0 means 1 dword
			 * - 1 means 2 dwords
			 */
			uint32_t is_last_config:1;
			/**<
			 * Flag to signify last config
			 */
			uint32_t is_nop:1;
			/*
			 * Indicates that the payload is NOP command. Used for
			 * debug.
			 * 0 - send/recv DW
			 * 1 - nop command
			 */
			uint32_t reserved0:8;
		} __attribute__ ((aligned(4), __packed__));
		uint32_t dword0;
	};
	uint32_t payload0;
	uint32_t payload1[0];
	/**<
	 * Payload 1 is optional
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_nic_passthrough_t
 * \brief   Send raw engine arc commands to NIC
 * \details This command is used to send raw engine arc commands
 *          to NIC engine
 */
struct sched_arc_cmd_nic_passthrough_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_ARC_CMD_NIC_PASSTHROUGH
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Engine group type to which this command needs to be
	 * send.
	 */
	uint32_t reserved0:7;
	uint32_t cmd_dw_size:6;
	/**<
	 * Size of current command in dwords
	 */
	uint32_t required_q_credits_inbytes:6;
	/**<
	 * Engine Group queue credits required before pushing configs
	 * to respective engines
	 */
	uint32_t reserved1:4;
	struct nic_passthrough_data_t passthrough_data[0];
	/**<
	 * Variable size config_record_t array
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_nic_edma_ops_t
 * \brief   NIC EDMA related operations
 * \details This command is used to for doing various EDMA operations
 *	    on EDMAs associated with NICs.
 */
struct sched_arc_cmd_nic_edma_ops_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_ARC_CMD_NIC_EDMA_OPS
	 */
	uint32_t engine_group_type:4;
	/**<
	 * engine group type :
	 * SCAL_EDMA_NETWORK_REDUCTION_GROUP
	 * SCAL_EDMA_NETWORK_LOOPBACK_GROUP
	 * SCAL_EDMA_NETWORK_MEMSET_GROUP
	 */
	uint32_t reserved:7;
	/**<
	 * reserved
	 */
	uint32_t cmd_size:16;
	/**<
	 * Command Size in Bytes, it includes the dword that
	 * contains the size field
	 */
	struct arc_cmd_update_edma_nic_ctxt_v3_t edma_ctxt_v3[0];
	struct arc_cmd_nic_edma_lin_ops_v3_t lin_ops_v3[0];
	struct arc_cmd_nic_edma_lin_memset_v3_2_t edma_linear_memset[0];
	struct arc_cmd_nic_edma_sibo_memset_v3_t sibo_memset_v3[0];
	struct arc_cmd_nic_edma_sibo_ops_v3_t sibo_ops_v3[0];
	/**<
	 * NIC EDMA command payload that needs to be dispatched
	 * to NICs
	 */
} __attribute__ ((aligned(4), __packed__));

/**
 * \struct  sched_arc_cmd_nic_coll_ops_scaleout_t
 * \brief   NIC collective operation to be performed by scaleout sched
 * \details This command is used to inform NICs to perform
 * 	    a collective operation
 */
struct sched_arc_cmd_nic_coll_ops_scaleout_t {
	uint32_t opcode:5;
	/**<
	 * opcode of the command = SCHED_SCALEOUT_SEND/RECV_ARC_CMD_NIC_COLL_OPS
	 */
	uint32_t engine_group_type:4;
	/**<
	 * Engine group type. Either scaleout send or recv
	 */
	uint32_t cmd_size:15;
	/**<
	 * Command Size in Bytes, it includes the dword that
	 * contains the size field
	 */
	uint32_t reserved:8;
	/**<
	 * Reserved
	 */
	struct arc_cmd_coll_ops_scaleout_t cmd_coll_ops_scaleout;
} __attribute__ ((aligned(4), __packed__));


/**
 * \struct  sched_arc_cmd_wait_for_ext_signal_t
 * \brief   wait for external signal
 * \details adds wait to an array of engine group types for external signal
 */
struct sched_arc_cmd_wait_for_ext_signal_t {
	uint32_t opcode:5;
	/**< opcode of the command = SCHED_ARC_CMD_WAIT_FOR_EXT_SIGNAL */
	uint32_t sob_id:13;
	uint32_t sm_id:2;
	/**<
	 * SOB related info, for monitor arming
	 */
	uint32_t num_engine_group_type:8;
	/**<
	 * Number of engine group types to which the external signal needs
	 * to be sent
	 */
	uint32_t reserved:4;
	/**< reserved */
	union {
		struct {
			uint64_t target_value0:15;
			uint64_t target_value1:15;
			uint64_t target_value2:15;
			uint64_t target_value3:15;
			uint64_t reserved0:4;
		} __attribute__ ((aligned(4), __packed__));
		uint64_t target_value;
	};
	uint8_t engine_group_type[4];
	/**<
	 * Array of engine group types to which the external signal needs
	 * to be sent
	 */
} __attribute__ ((aligned(4), __packed__));


#endif /* __GAUDI2_ARC_SCHED_PACKETS_H__ */
