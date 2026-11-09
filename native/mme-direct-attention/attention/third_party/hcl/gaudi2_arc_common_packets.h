/* SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2020 HabanaLabs Ltd.
 * All Rights Reserved.
 */

#ifndef __GAUDI2_ARC_COMMON_PACKETS_H__
#define __GAUDI2_ARC_COMMON_PACKETS_H__

/**
 * \file    gaudi2_arc_common_packets.h
 * \brief   IDs for QMAN ARC CPUs
 *          This defines IDs that needs to be programmed into CPU ID
 *          register xx_ARC_AUX_ARC_NUM
 */
enum {
	/**
	 * CPU IDs for each ARC CPUs
	 */

	CPU_ID_SCHED_ARC0 = 0, /* FARM_ARC0 */
	CPU_ID_SCHED_ARC1 = 1, /* FARM_ARC1 */
	CPU_ID_SCHED_ARC2 = 2, /* FARM_ARC2 */
	CPU_ID_SCHED_ARC3 = 3, /* FARM_ARC3 */
	/* Dcore1 MME Engine ARC instance used as scheduler */
	CPU_ID_SCHED_ARC4 = 4, /* DCORE1_MME0 */
	/* Dcore3 MME Engine ARC instance used as scheduler */
	CPU_ID_SCHED_ARC5 = 5, /* DCORE3_MME0 */

	CPU_ID_TPC_QMAN_ARC0 = 6,   /* DCORE0_TPC0 */
	CPU_ID_TPC_QMAN_ARC1 = 7,   /* DCORE0_TPC1 */
	CPU_ID_TPC_QMAN_ARC2 = 8,   /* DCORE0_TPC2 */
	CPU_ID_TPC_QMAN_ARC3 = 9,   /* DCORE0_TPC3 */
	CPU_ID_TPC_QMAN_ARC4 = 10,  /* DCORE0_TPC4 */
	CPU_ID_TPC_QMAN_ARC5 = 11,  /* DCORE0_TPC5 */
	CPU_ID_TPC_QMAN_ARC6 = 12,  /* DCORE1_TPC0 */
	CPU_ID_TPC_QMAN_ARC7 = 13,  /* DCORE1_TPC1 */
	CPU_ID_TPC_QMAN_ARC8 = 14,  /* DCORE1_TPC2 */
	CPU_ID_TPC_QMAN_ARC9 = 15,  /* DCORE1_TPC3 */
	CPU_ID_TPC_QMAN_ARC10 = 16, /* DCORE1_TPC4 */
	CPU_ID_TPC_QMAN_ARC11 = 17, /* DCORE1_TPC5 */
	CPU_ID_TPC_QMAN_ARC12 = 18, /* DCORE2_TPC0 */
	CPU_ID_TPC_QMAN_ARC13 = 19, /* DCORE2_TPC1 */
	CPU_ID_TPC_QMAN_ARC14 = 20, /* DCORE2_TPC2 */
	CPU_ID_TPC_QMAN_ARC15 = 21, /* DCORE2_TPC3 */
	CPU_ID_TPC_QMAN_ARC16 = 22, /* DCORE2_TPC4 */
	CPU_ID_TPC_QMAN_ARC17 = 23, /* DCORE2_TPC5 */
	CPU_ID_TPC_QMAN_ARC18 = 24, /* DCORE3_TPC0 */
	CPU_ID_TPC_QMAN_ARC19 = 25, /* DCORE3_TPC1 */
	CPU_ID_TPC_QMAN_ARC20 = 26, /* DCORE3_TPC2 */
	CPU_ID_TPC_QMAN_ARC21 = 27, /* DCORE3_TPC3 */
	CPU_ID_TPC_QMAN_ARC22 = 28, /* DCORE3_TPC4 */
	CPU_ID_TPC_QMAN_ARC23 = 29, /* DCORE3_TPC5 */
	CPU_ID_TPC_QMAN_ARC24 = 30, /* DCORE0_TPC6 - Never present */

	CPU_ID_MME_QMAN_ARC0 = 31, /* DCORE0_MME0 */
	CPU_ID_MME_QMAN_ARC1 = 32, /* DCORE2_MME0 */

	CPU_ID_EDMA_QMAN_ARC0 = 33, /* DCORE0_EDMA0 */
	CPU_ID_EDMA_QMAN_ARC1 = 34, /* DCORE0_EDMA1 */
	CPU_ID_EDMA_QMAN_ARC2 = 35, /* DCORE1_EDMA0 */
	CPU_ID_EDMA_QMAN_ARC3 = 36, /* DCORE1_EDMA1 */
	CPU_ID_EDMA_QMAN_ARC4 = 37, /* DCORE2_EDMA0 */
	CPU_ID_EDMA_QMAN_ARC5 = 38, /* DCORE2_EDMA1 */
	CPU_ID_EDMA_QMAN_ARC6 = 39, /* DCORE3_EDMA0 */
	CPU_ID_EDMA_QMAN_ARC7 = 40, /* DCORE3_EDMA1 */

	CPU_ID_PDMA_QMAN_ARC0 = 41, /* DCORE0_PDMA0 */
	CPU_ID_PDMA_QMAN_ARC1 = 42, /* DCORE0_PDMA1 */

	CPU_ID_ROT_QMAN_ARC0 = 43, /* ROT0 */
	CPU_ID_ROT_QMAN_ARC1 = 44, /* ROT1 */

	CPU_ID_NIC_QMAN_ARC0 = 45,  /* NIC0_0 */
	CPU_ID_NIC_QMAN_ARC1 = 46,  /* NIC0_1 */
	CPU_ID_NIC_QMAN_ARC2 = 47,  /* NIC1_0 */
	CPU_ID_NIC_QMAN_ARC3 = 48,  /* NIC1_1 */
	CPU_ID_NIC_QMAN_ARC4 = 49,  /* NIC2_0 */
	CPU_ID_NIC_QMAN_ARC5 = 50,  /* NIC2_1 */
	CPU_ID_NIC_QMAN_ARC6 = 51,  /* NIC3_0 */
	CPU_ID_NIC_QMAN_ARC7 = 52,  /* NIC3_1 */
	CPU_ID_NIC_QMAN_ARC8 = 53,  /* NIC4_0 */
	CPU_ID_NIC_QMAN_ARC9 = 54,  /* NIC4_1 */
	CPU_ID_NIC_QMAN_ARC10 = 55, /* NIC5_0 */
	CPU_ID_NIC_QMAN_ARC11 = 56, /* NIC5_1 */
	CPU_ID_NIC_QMAN_ARC12 = 57, /* NIC6_0 */
	CPU_ID_NIC_QMAN_ARC13 = 58, /* NIC6_1 */
	CPU_ID_NIC_QMAN_ARC14 = 59, /* NIC7_0 */
	CPU_ID_NIC_QMAN_ARC15 = 60, /* NIC7_1 */
	CPU_ID_NIC_QMAN_ARC16 = 61, /* NIC8_0 */
	CPU_ID_NIC_QMAN_ARC17 = 62, /* NIC8_1 */
	CPU_ID_NIC_QMAN_ARC18 = 63, /* NIC9_0 */
	CPU_ID_NIC_QMAN_ARC19 = 64, /* NIC9_1 */
	CPU_ID_NIC_QMAN_ARC20 = 65, /* NIC10_0 */
	CPU_ID_NIC_QMAN_ARC21 = 66, /* NIC10_1 */
	CPU_ID_NIC_QMAN_ARC22 = 67, /* NIC11_0 */
	CPU_ID_NIC_QMAN_ARC23 = 68, /* NIC11_1 */

	CPU_ID_MAX = 69,
	CPU_ID_SCHED_MAX = 6,

	CPU_ID_ALL = 0xFE,
	CPU_ID_INVALID = 0xFF,
};

/**
 * Message codes of SCAL for firmware usage
 * Updated by SCAL into SCHED_SCAL_STATUS register
 */
enum {
	SCAL_INIT_COMPLETED = 0x00010000,
};
/**
 * \enum    qman_engine_type_t
 * \brief   Various engine types
 * \details Engine Types supported by Scheduler ARC
 */
enum qman_engine_type_t {
	QMAN_ENGINE_MME = 0,
	QMAN_ENGINE_TPC = 1,
	QMAN_ENGINE_NIC = 2,
	QMAN_ENGINE_RTR = 3,
	QMAN_ENGINE_EDMA = 4,
	QMAN_ENGINE_PDMA = 5,
	QMAN_ENGINE_TYPE_COUNT = 0x6,
	QMAN_ENGINE_LAST = 0x7,
};

/**
 * \enum    engine_type_t
 * \brief   Various engine types
 * \details Engine Types
 */
enum engine_type_t {
	ENG_TYPE_MME_COMPUTE = 0,
	ENG_TYPE_TPC_COMPUTE = 1,
	ENG_TYPE_EDMA_COMPUTE = 2,
	ENG_TYPE_TPC_MEDIA = 3,
	ENG_TYPE_RTR_MEDIA = 4,
	ENG_TYPE_PDMA_TX = 5,
	ENG_TYPE_PDMA_RX = 6,
	ENG_TYPE_NIC_SCALE_UP = 7,
	ENG_TYPE_NIC_SCALE_OUT = 8,
	ENG_TYPE_EDMA_NIC = 9,
	ENG_TYPE_COUNT = 0xA,
	ENG_TYPE_SIZE = 0xF
};

/**
 * \enum    scheduler_type_t
 * \brief   Various scheduler types
 * \details Scheduler Types
 */
enum scheduler_type_t {
	SCHED_TYPE_COMPUTE = 0,
	SCHED_TYPE_GARBAGE_REDUCTION = 1,
	SCHED_TYPE_SCALE_OUT_RECV = 2,
	SCHED_TYPE_SCALE_OUT_SEND = 3,
	SCHED_TYPE_SCALE_UP_RECV = 4,
	SCHED_TYPE_SCALE_UP_SEND = 5,
	SCHED_TYPE_COUNT = 6,
	SCHED_TYPE_SIZE = 0xF
};

/**
 * Max number of MMEs
 */
#define GAUDI2_MAX_MME_COUNT				2

/**
 * Max number of MMEs
 */
#define GAUDI2_MAX_EDMA_COUNT				5

/**
 * Total number of engine groups supported by firmware
 */
#define QMAN_ENGINE_GROUP_TYPE_COUNT		16

/**
 * Total number of Dcores
 */
#define SCHED_ARC_MAX_DCORE			4

/**
 * Total number of Streams supported by a scheduler instance
 */
#define SCHED_ARC_MAX_STREAMS			32

/**
 * Max Priorities
 */
#define SCHED_ARC_MAX_ACP_PRIORITY		4

/**
 * Max number of groups that can be created per engine type
 */
#define SCHED_ARC_MAX_ENGINE_TYPE_GROUPS	2

/**
 * Total number of Fence counters per scheduler instance
 */
#define SCHED_ARC_GLOBAL_FENCE_COUNTERS_COUNT	64

/**
 * Size of the DCCM CCB buffer in bytes. All the commands in the CCB
 * must not cross this boundary.
 */
#define SCHED_ARC_CCB_STREAM_BUFF_SIZE		256

/**<
 * Virtual SOB bitmap
 */
enum sched_cmpt_sync_scheme_bitmap {
	VIRTUAL_SOB_INDEX_TPC = 0,
	VIRTUAL_SOB_INDEX_MME = 1,
	VIRTUAL_SOB_INDEX_EDMA = 2,
	VIRTUAL_SOB_INDEX_ROT = 3,
	VIRTUAL_SOB_INDEX_DEBUG = 4,
	SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT = 5
};

/**
 * \enum    scheduler_type_t
 * \brief   Fence IDs used by Engines FW
 * \details Engine Firmware uses following two fences
 *	    for sync scheme and back to back execution
 */
enum {
	SYNC_SCHEME_FENCE_ID = 0,
	EXT_SIGNAL_FENCE_ID = SYNC_SCHEME_FENCE_ID,
	B2B_FENCE_ID = 1,
	GC_USED_FENCE_ID = 2
};

/**<
 * Sync scheme Monitor count per engine for compute
 */
#define SCHED_CMPT_ENG_SYNC_SCHEME_MON_COUNT		4

/**<
 * Long DBG SOB Monitor count per engine, needs a long monitor
 */
#define SCHED_CMPT_ENG_SYNC_SCHEME_DBG_MON_COUNT	4

/**<
 * Long EXT SIGNAL SOB Monitor count per engine, needs a long monitor
 */
#define SCHED_CMPT_ENG_EXT_SIG_MON_COUNT	4

/**<
 * B2B Monitor count per engine
 */
#define SCHED_CMPT_ENG_B2B_MON_COUNT			1

/**<
 * Max number of recipes that can be updated in batch
 * update recipe cmd
 */
#define ARC_CMD_UPDATE_RECIPE_MAX_RECIPES 4

/**
 * \enum    arc_regions_t
 * \brief   ARC address map
 * \details Address map of scheduler as well as engine ARC.
 *          Each region is of size 256MB. Driver programs the extension
 *          registers.
 */
enum arc_regions_t {
	ARC_REGION0_UNSED  = 0,
	/*
	 * Extension registers
	 * None
	 */
	ARC_REGION1_SRAM = 1,
	/*
	 * Extension registers
	 * AUX_SRAM_LSB_ADDR
	 * AUX_SRAM_MSB_ADDR
	 * ARC Address: 0x1000_0000
	 */
	ARC_REGION2_CFG = 2,
	/*
	 * Extension registers
	 * AUX_CFG_LSB_ADDR
	 * AUX_CFG_MSB_ADDR
	 * ARC Address: 0x2000_0000
	 */
	ARC_REGION3_GENERAL = 3,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_0
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_0
	 * ARC Address: 0x3000_0000
	 */
	ARC_REGION4_HBM0_FW = 4,
	/*
	 * Extension registers
	 * AUX_HBM0_LSB_ADDR
	 * AUX_HBM0_MSB_ADDR
	 * AUX_HBM0_OFFSET
	 * ARC Address: 0x4000_0000
	 */
	ARC_REGION5_HBM1_GC_DATA = 5,
	/*
	 * Extension registers
	 * AUX_HBM1_LSB_ADDR
	 * AUX_HBM1_MSB_ADDR
	 * AUX_HBM1_OFFSET
	 * ARC Address: 0x5000_0000
	 */
	ARC_REGION6_HBM2_GC_DATA = 6,
	/*
	 * Extension registers
	 * AUX_HBM2_LSB_ADDR
	 * AUX_HBM2_MSB_ADDR
	 * AUX_HBM2_OFFSET
	 * ARC Address: 0x6000_0000
	 */
	ARC_REGION7_HBM3_GC_DATA = 7,
	/*
	 * Extension registers
	 * AUX_HBM3_LSB_ADDR
	 * AUX_HBM3_MSB_ADDR
	 * AUX_HBM3_OFFSET
	 * ARC Address: 0x7000_0000
	 */
	ARC_REGION8_DCCM = 8,
	/*
	 * Extension registers
	 * None
	 * ARC Address: 0x8000_0000
	 */
	ARC_REGION9_PCIE = 9,
	/*
	 * Extension registers
	 * AUX_PCIE_LSB_ADDR
	 * AUX_PCIE_MSB_ADDR
	 * ARC Address: 0x9000_0000
	 */
	ARC_REGION10_GENERAL = 10,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_1
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_1
	 * ARC Address: 0xA000_0000
	 */
	ARC_REGION11_GENERAL = 11,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_2
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_2
	 * ARC Address: 0xB000_0000
	 */
	ARC_REGION12_GENERAL = 12,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_3
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_3
	 * ARC Address: 0xC000_0000
	 */
	ARC_REGION13_GENERAL = 13,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_4
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_4
	 * ARC Address: 0xD000_0000
	 */
	ARC_REGION14_GENERAL = 14,
	/*
	 * Extension registers
	 * AUX_GENERAL_PURPOSE_LSB_ADDR_5
	 * AUX_GENERAL_PURPOSE_MSB_ADDR_5
	 * ARC Address: 0xE000_0000
	 */
	ARC_REGION15_LBU = 15
	/*
	 * Extension registers
	 * None
	 * ARC Address: 0xF000_0000
	 */
};

#endif /* __GAUDI2_ARC_COMMON_PACKETS_H__ */
