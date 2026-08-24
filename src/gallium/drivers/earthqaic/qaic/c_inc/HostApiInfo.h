/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef HOST_API_INFO_H
#define HOST_API_INFO_H

#include <stdint.h>

#pragma pack(push, 1)

#define StatusQuerySerialSize 16
#define StatusQueryBoardSerialSize 32
#define fwQCImageVersionStringSize 38
#define fwOEMImageVersionStringSize 32
#define fwImageVariantStringSize 16
#define QDEVICE_FEATURE_MAX_BITS (64)

// Should be in sync with same symbol in SBL
#define SHARED_IMEM_SBL_IMAGE_NAME_LEN 0x18

// This value must match when receiving the results
#define deviceInfoFormatVersion 12

/*****************************************************************************
 *  All the NNC protocol structures should end on a 8byte boundary and all   *
 *  its structure members must be naturally aligned to itself. It is the     *
 *  responsibility of the maintainer/programmer to make sure that the above  *
 *  criteria are met while updating the NNC protocol structures.            *
 * ***************************************************************************/

typedef enum nnc_transaction_device_flag_bitpos_type {
  NNC_DEVICE_FEATURE_FLAG_AUTO_SKU_BITPOS = 0,  // Auto sku
  NNC_DEVICE_FEATURE_FLAG_DPS_FLAG_BIT = 1,     // Data plane switching enabled
  NNC_DEVICE_FEATURE_FLAG_HBE_FLAG_BIT = 2,     // Hybrid boot enabled
  NNC_DEVICE_FEATURE_FLAG_HBOOT_FLAG_BIT = 3,   // Current boot hybrid
  NNC_DEVICE_FEATURE_FLAG_SSR_FLAG_BIT = 4,     // Subsystem restart enabled
  NNC_DEVICE_FEATURE_FLAG_MQ_FLAG_BIT = 5,      // Multi-AIC100 enabled
  NNC_DEVICE_FEATURE_FLAG_ECC_DDR_FLAG_BIT = 6, // ECC enabled for DDR
  NNC_DEVICE_FEATURE_PAGE_RET_FLAG_BIT = 7,     // DDR page retirement enabled
  NNC_DEVICE_FEATURE_SECURE_BOOT_FLAG_BIT = 8,  // Secure boot enabled
  NNC_DEVICE_FEATURE_SOC_DEBUG_FLAG_BIT = 9,    // SoC Debug enabled
  NNC_DEVICE_FEATURE_FLAG_NUM_BITS
} nnc_transaction_device_flag_bitpos_type_t;

/// SKU Type
typedef enum nnc_transaction_sku_query_type {
  NNC_TRANSACTION_SKU_TYPE_DEV_INVALID_SKU = 0,
  NNC_TRANSACTION_SKU_TYPE_DEV_M_2_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_PCIE_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_PCIE_PRO_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_PCIE_LITE_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_PCIE_ULTRA_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_AUTO_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_PCIE_ULTRA_PLUS_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_PCIE_080_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_PCIE_ULTRA_080_SKU,
  NNC_TRANSACTION_SKU_TYPE_DEV_MAX_SKU,
  NNC_TRANSACTION_SKU_TYPE_ENUM_MAX = UINT8_MAX,
} nnc_transaction_sku_query_type_t;

/*****************************************************************************
 *  All the NNC protocol structures should end on a 8byte boundary and all   *
 *  its structure members must be naturally aligned to itself. It is the     *
 *  responsibility of the maintainer/programmer to make sure that the above  *
 *  criteria are met while updating the NNC protocol structures.            *
 * ***************************************************************************/

typedef struct host_api_info_data_header {
  /// Version format for this structure, this must be the first field
  uint8_t format_version;
  uint8_t padding[7];
} host_api_info_data_header_t;

typedef struct {
  // Total DDR in KB
  uint32_t dramTotal;
  // Free DDR in KB
  uint32_t dramFree;
  // DDR Fragmentation status = 100*(1 - largest_free_block/free_bytes)
  //  0:no frag, 100:high frag
  uint32_t fragmentationStatus;
  // Total number of virtual channels
  uint8_t vcTotal;
  // Number of free virtual channels
  uint8_t vcFree;
  // Total number of NSPs
  uint8_t nspTotal;
  // Number of free NSPs
  uint8_t nspFree;
  // Number of MCid
  uint32_t mcidTotal;
  // Free MCIDs
  uint32_t mcidFree;
  // Total Semaphore count
  uint32_t semTotal;
  // Free Semaphore count
  uint32_t semFree;
  // Total number of physical channels
  uint8_t pcTotal;
  // Number of dedicated physical channels
  uint8_t pcReserved;
  uint8_t padding[6];
} host_api_resource_info_t;

typedef struct {
  /// NSP Frequency;
  uint32_t nspFrequencyHz;
  /// DDR Frequency;
  uint32_t ddrFrequencyHz;
  /// COMPNOC Frequency;
  uint32_t compFrequencyHz;
  /// MEMNOC Frequency;
  uint32_t memFrequencyHz;
  /// SYSNOC Frequency;
  uint32_t sysFrequencyHz;
  uint32_t padding;
  /// Peak compute on device in ops/second. Assumes all ops are in int8.
  uint64_t peakCompute;
  /// Memory bandwidth from DRAM on device in Kbytes/second, averaged over last
  /// ~100 ms.
  uint64_t dramBw;
  uint64_t reserved1;
  uint64_t reserved2;
} host_api_performance_info_t;

/// Data structure returned to from device in status query
/// Message format NNC_STATUS_QUERY_DEV
typedef struct {
  /// Firmware version

  uint8_t fwVersion_major;
  uint8_t fwVersion_minor;
  uint8_t fwVersion_patch;
  uint8_t fwVersion_build;

  char fwQCImageVersionString[fwQCImageVersionStringSize];
  char fwOEMImageVersionString[fwQCImageVersionStringSize];
  char fwImageVariantString[fwImageVariantStringSize];

  /// Hardware version
  uint32_t hwVersion;

  /// Serial number
  char serial[StatusQuerySerialSize];

  /// Board serial number
  char board_serial[StatusQueryBoardSerialSize];

  /// The highest compiler version supported by the library
  uint32_t compilerVersion;

  host_api_resource_info_t resource_info;
  host_api_performance_info_t perf_info;

  /// Number of loaded constants
  uint32_t numLoadedConsts;
  /// Number of constants in-use
  uint32_t numConstsInUse;
  /// Number of loaded networks
  uint8_t numLoadedNWs;
  /// Number of active networks
  uint8_t numActiveNWs;

  /// Metadata version
  uint16_t metaVerMaj;
  uint16_t metaVerMin;

  /// API version
  uint16_t nncCommandProtocolMajorVersion;
  uint16_t nncCommandProtocolMinorVersion;

  /* SBL Image Version Name */
  char sbl_image_name[SHARED_IMEM_SBL_IMAGE_NAME_LEN];

  /// Reset required to retire additional DDR pages
  uint8_t need_reset_to_retire_pages;
  uint8_t padding1[1];

  /* PVS Image Version Number */
  uint32_t pvs_image_version;
  /* NSP Defective PG Mask */
  uint64_t nsp_pgmask;

  /*Timesync offset in microseconds*/
  uint64_t qts_time_offset_usecs;

  /// DDR pages that have been retired
  uint32_t num_retired_pages;

  // SKU type
  uint8_t sku_type;
  // Complex ID
  uint8_t complex_id;
  // Padding
  uint8_t padding3[2];
} host_api_info_dev_data_t;

/// Message Format Type NNC_STATUS_QUERY_NSP
/**
 * Data returned in NSP query version response
 */

#define nnncQCImageVersionStringSize 64
#define nnncOEMImageVersionStringSize 48
#define nnncImageVariantStringSize 48

typedef struct host_api_info_nsp_static_data {
  uint8_t major;
  uint8_t minor;
  uint8_t patch;
  uint8_t build;
  uint32_t padding;
  char qc[nnncQCImageVersionStringSize];
  char oem[nnncOEMImageVersionStringSize];
  char variant[nnncImageVariantStringSize];
} host_api_info_nsp_static_data_t;

/// Data structure returned to from device in status query
/// Message format NNC_STATUS_QUERY_NSP
typedef struct {
  host_api_info_nsp_static_data_t nsp_fw_version;
} host_api_info_nsp_data_t;

// This structure should be used by the host to copy all the responses
// into one combined status
typedef struct {
  host_api_info_data_header_t header;
  host_api_info_dev_data_t dev_data;
  host_api_info_nsp_data_t nsp_data;
} host_api_info_data_t;

#pragma pack(pop)

#endif
