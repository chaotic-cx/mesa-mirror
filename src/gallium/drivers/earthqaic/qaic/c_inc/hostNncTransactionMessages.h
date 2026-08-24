/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef AICLINUXRUNTIME_HOSTNNCTRANSACTIONMESSAGES_H
#define AICLINUXRUNTIME_HOSTNNCTRANSACTIONMESSAGES_H

#include "nncErrors.h"
#include "HostApiInfo.h"
#include <stdint.h>

#define MANAGE_MAX_MSG_LENGTH 4096

/*****************************************************************************
 *  All the NNC protocol structures should end on a 8byte boundary and all   *
 *  its structure members must be naturally aligned to itself. It is the     *
 *  responsibility of the maintainer/programmer to make sure that the above  *
 *  criteria are met while updating the NNC protocol structures.            *
 * ***************************************************************************/

/**
 * LRT lib and KMD exchange one NNC message at a time.
 * Each NNC message is composed of one or more transactions.
 * A transaction of type PASSTHROUGH encapsulates one and only one command.
 * The high level tiering is like this:
 *    nnc_msg
 *       |-- nnc_transaction
 *               |-- nnc_command (PASSTHROUGH transaction only)
 *
 * There is only one nnc_msg defined. One nnc_msg could contain more then one
 * transactions. One PASSTHROUGH transaction can have only one nnc_command.
 */
typedef enum nnc_transaction_type {

  NNC_TRANSACTION_TYPE_UNDEFINED = 0, //!< Undefined transaction
  NNC_TRANSACTION_TYPE_PASSTHROUGH_UK = 1,
  NNC_TRANSACTION_TYPE_PASSTHROUGH_KU = 2,
  NNC_TRANSACTION_TYPE_PASSTHROUGH_RESP =
      3, //!< Wrapper of command responses to host
  NNC_TRANSACTION_TYPE_PASSTHROUGH_REQ =
      4, //!< wrapper of command requestss from host
  NNC_TRANSACTION_TYPE_DMA_XFER_UK = 5,
  NNC_TRANSACTION_TYPE_DMA_XFER_REQ =
      6, //!< Contains DMA buffer information from host
  NNC_TRANSACTION_TYPE_SETUP_VC_UK = 7,
  NNC_TRANSACTION_TYPE_VC_SETUP_RESP = 8, //!< VC Setup response to driver
  NNC_TRANSACTION_TYPE_VC_SETUP_REQ = 9,  //!< VC Setup info from driver
  NNC_TRANSACTION_TYPE_TEARDOWN_VC_UK = 10,
  NNC_TRANSACTION_TYPE_VC_TEARDOWN_RESP =
      11, //!< VC Teardown response to driver
  NNC_TRANSACTION_TYPE_STATUS_UK = 12,
  NNC_TRANSACTION_TYPE_STATUS_KU = 13,
  NNC_TRANSACTION_TYPE_STATUS_RESP = 14, //!< QSM response of NNC version
  NNC_TRANSACTION_TYPE_STATUS_REQ = 15,  //!< Host request of NNC version
  NNC_TRANSACTION_TYPE_TERMINATE_RESP =
      16, //!< Response to the terminate request
  NNC_TRANSACTION_TYPE_TERMINATE_REQ =
      17, //!< Terminate the network and all resources related to a host process
  NNC_TRANSACTION_TYPE_DMA_XFER_CONT =
      18, //!< Resp to host to continue DMA transfer
  NNC_TRANSACTION_TYPE_VALIDATE_RESOURCE_GROUP_RESP =
      19, //!< QSM response of resource group validation request
  NNC_TRANSACTION_TYPE_VALIDATE_RESOURCE_GROUP_REQ =
      20, //!< Host request for validating resource group
  NNC_TRANSACTION_TYPE_INVALID = INT_MAX,
} nnc_transaction_type_t;

/**
 * nnc_commands contains information exchanged between QSM and LRT lib.
 * These commands are transparent to kernel mode driver, and are encapsulated in
 * PASSTHROUGH transactions.
 */
typedef enum nnc_command_type {
  NNC_COMMAND_TYPE_UNDEFINED = 0,
  NNC_COMMAND_TYPE_LOAD_CONSTANTS_REQ = 1,
  NNC_COMMAND_TYPE_LOAD_CONSTANTS_RESP = 2,
  NNC_COMMAND_TYPE_LOAD_ELF_REQ = 3,
  NNC_COMMAND_TYPE_LOAD_ELF_RESP = 4,
  NNC_COMMAND_TYPE_UNLOAD_CONSTANTS_REQ = 5,
  NNC_COMMAND_TYPE_UNLOAD_CONSTANTS_RESP = 6,
  NNC_COMMAND_TYPE_UNLOAD_ELF_REQ = 7,
  NNC_COMMAND_TYPE_UNLOAD_ELF_RESP = 8,
  NNC_COMMAND_TYPE_ACTIVATE_REQ = 9,
  NNC_COMMAND_TYPE_ACTIVATE_RESP = 10,
  NNC_COMMAND_TYPE_DEACTIVATE_REQ = 11,
  NNC_COMMAND_TYPE_DEACTIVATE_RESP = 12,
  NNC_COMMAND_TYPE_STATUS_REQ = 13,
  NNC_COMMAND_TYPE_STATUS_RESP = 14,
  NNC_COMMAND_TYPE_LOAD_CONSTANTS_EX_REQ = 15,
  NNC_COMMAND_TYPE_LOAD_CONSTANTS_PAYLOAD_REQ = 16,
  NNC_COMMAND_TYPE_TERMINATE_REQ = 17,
  NNC_COMMAND_TYPE_TERMINATE_RESP = 18,
  NNC_COMMAND_TYPE_RESERVE_VC_REQ = 19,
  NNC_COMMAND_TYPE_RESERVE_VC_RESP = 20,
  NNC_COMMAND_TYPE_RELEASE_VC_REQ = 22,
  NNC_COMMAND_TYPE_RELEASE_VC_RESP = 23,
  NNC_COMMAND_TYPE_AC_RESERVATION_REQ = 24,
  NNC_COMMAND_TYPE_AC_RESERVATION_RESP = 25,
  NNC_COMMAND_TYPE_AC_RELEASE_REQ = 26,
  NNC_COMMAND_TYPE_AC_RELEASE_RESP = 27,
  NNC_COMMAND_TYPE_CREATE_RESOURCE_GROUP_REQ = 28,
  NNC_COMMAND_TYPE_CREATE_RESOURCE_GROUP_RESP = 29,
  NNC_COMMAND_TYPE_RELEASE_RESOURCE_GROUP_REQ = 30,
  NNC_COMMAND_TYPE_RELEASE_RESOURCE_GROUP_RESP = 31,
  NNC_COMMAND_TYPE_RESOURCE_INFO_REQ =
      32, //!< Request Info on Resource Reservation
  NNC_COMMAND_TYPE_RESOURCE_INFO_RESP =
      33, //!< Response from Info Request on Resource Reservation
  NNC_COMMAND_TYPE_PERFORMANCE_INFO_REQ =
      34, //!< Request Info on performance stats
  NNC_COMMAND_TYPE_PERFORMANCE_INFO_RESP =
      35, //!< Response from Info Request on performance
  NNC_COMMAND_TYPE_ACTIVATE_STATE_CMD_REQ =
      36, //!< State Update Command Request for an Activation
  NNC_COMMAND_TYPE_ACTIVATE_STATE_CMD_RESP =
      37, //!< Returns the result of the acitvate state command request
  NNC_COMMAND_TYPE_DEVICE_CONFIG_REQ =
      38, //!< Request to change Device Configuration
  NNC_COMMAND_TYPE_DEVICE_CONFIG_RESP =
      39, //!< Response to Device Config request
  NNC_COMMAND_TYPE_DMA_BUFFER_RESERVATION_REQ =
      40, //!< Request to allocate buffer for DMA xfer to NSP
  NNC_COMMAND_TYPE_DMA_BUFFER_RESERVATION_RESP =
      41, //!< Response to DMA buffer reservation request
  NNC_COMMAND_TYPE_DMA_BUFFER_RELEASE_REQ =
      42, //!< Request to release buffer for DMA xfer to NSP
  NNC_COMMAND_TYPE_DMA_BUFFER_RELEASE_RESP =
      43, //!< Response to DMA buffer release request
  NNC_COMMAND_TYPE_UPDATE_AC_RESERVATION_REQ =
      44, //!< Request to update resources of Activation Reservation
  NNC_COMMAND_TYPE_UPDATE_AC_RESERVATION_RESP =
      45, //!< Response to update Activation Reservation Request
  NNC_COMMAND_TYPE_MQ_PREACTIVATE_REQ =
      46, //!< State Update Command Request for an Activation
  NNC_COMMAND_TYPE_MQ_PREACTIVATE_RESP =
      47, //!< Returns the result of the acitvate state command request
  NNC_COMMAND_TYPE_MQ_ACTIVATE_REQ =
      48, //!< Activate requests after pre-activation is complete
  NNC_COMMAND_TYPE_MQ_ACTIVATE_RESP = 49, //!< Response to MQ Activate request
  NNC_COMMAND_TYPE_MQTH_ACTIVATE_REQ =
      50, //!< Activate requests for MQ Through Host (before MQ_ACTIVATE)
  NNC_COMMAND_TYPE_MQTH_ACTIVATE_RESP =
      51, //!< Response to MQTH Activate request
  NNC_COMMAND_TYPE_MQTH_DEACTIVATE_REQ =
      52, //!< De-Activate requests for MQ Through Host (before MQ_ACTIVATE)
  NNC_COMMAND_TYPE_MQTH_DEACTIVATE_RESP =
      53, //!< De-Response to MQTH Activate request
  NNC_COMMAND_TYPE_MQ_POSTACTIVATE_REQ =
      54, //!< Request sent after MQ_ACTIVTE, only if P2P is present
  NNC_COMMAND_TYPE_MQ_POSTACTIVATE_RESP =
      55, //!< Response to MQ Post Activate request
  NNC_COMMAND_TYPE_ATTESTATION_REQ = 56,  //!< Attestation Request
  NNC_COMMAND_TYPE_ATTESTATION_RESP = 57, //!< Attestation Report
  NNC_COMMAND_TYPE_CONSTANTS_DECRYPTION_REQ = 58,
  NNC_COMMAND_TYPE_CONSTANTS_DECRYPTION_RESP = 59,
  NNC_COMMAND_TYPE_LOAD_ELF_DECRYPTION_REQ =
      60, //!< Load elf and decryption parameters in same message
  NNC_COMMAND_TYPE_LOAD_ELF_DECRYPTION_RESP = 61,
  NNC_COMMAND_TYPE_DELAYED_STATUS_REQ = 62, //!< For test purpose only
  NNC_COMMAND_TYPE_WORK_IN_PROGRESS_RESP = 63,
  NNC_COMMAND_TYPE_PROGRESS_CHECK_REQ = 64,

  NNC_COMMAND_TYPE_MAX = 65, //!< Maximum number of defined request codes
  NNC_COMMAND_TYPE_MAX_SIZE = INT_MAX,
} nnc_command_type_t;

typedef struct manage_msg_header {
  uint32_t length; /* size of data used, remainder is reserved */
  uint32_t count;  /* number of transactions in data */
} manage_msg_header_t;

typedef struct nnc_transaction_header {
  /** The type is defined as nnc_transaction_type */
  int32_t type;
  /** The length of the transaction, including header*/
  uint32_t len;
} nnc_transaction_header_t;

typedef struct manage_msg {
  manage_msg_header_t header;
  uint8_t data[MANAGE_MAX_MSG_LENGTH - sizeof(manage_msg_header_t)];
} manage_msg_t;

typedef struct nnc_transaction {
  nnc_transaction_header_t header;
  uint8_t data[];
} nnc_transaction_t;

typedef struct nnc_cmd {
  /** Command as defined in nnc_command_type */
  uint32_t type;
  uint32_t padding;
  uint8_t data[];
} nnc_cmd_t;

/** Passthrough request/response share the same format */
typedef struct nnc_transaction_passthrough {
  nnc_transaction_header_t header;
  nnc_cmd_t cmd;
} nnc_transaction_passthrough_t;

typedef struct addr_len_pair {
  uint64_t addr;
  uint64_t len;
} addr_len_pair_t;

/** Passthrough request/response share the same format */
typedef struct nnc_transaction_dma_xfer {
  nnc_transaction_header_t header;
  /** Tag used to associate this transaction with a elf/constnats load command
   */
  uint32_t tag;
  uint32_t count;
  addr_len_pair_t alp[];
} nnc_transaction_dma_xfer_t;

typedef struct nnc_transaction_vc_setup_uk {
  nnc_transaction_header_t header;
  uint32_t q_size;
  int eventfd;
  uint32_t options;
  uint32_t padding; /* Paddding must be 0 */
} nnc_transaction_vc_setup_uk_t;

typedef struct nnc_transaction_vc_teardown_uk {
  nnc_transaction_header_t header;
  uint32_t virtual_channel_id;
  uint32_t padding;
} nnc_transaction_vc_teardown_uk_t;

typedef struct nnc_cmd_status_request {
  uint32_t type;
  /** Specify type of info requested host_api_info_nsp_data_t */
  uint8_t info_type;
  uint8_t padding[3];
} nnc_cmd_status_query_request_t;

typedef struct nnc_cmd_delayed_status_request {
  uint32_t type;
  /**
   * This NNC command is for testing purpose only. FW starts work_in_progress
   * and progress_check call flow to avoid control path timeout. FW sends
   * delayed status response after delay_time_sec.
   */
  uint16_t delay_time_sec;
  uint8_t padding[2];
} nnc_cmd_delayed_status_query_request_t;

/**
 *  LRT                                                         FW
 *   | ---- NNC passthrough command (long processing time) ---> |
 *   |                                                          |
 * ===== Loop ===============================================================
 *   | <--- Work_in_progress_response (immediate response) ---- | Loop until
 *   | LRT runs Progress check timer                            | FW is done
 *   | ----- Progress_check_request (Upon timer expiry) ------> |
 * ==========================================================================
 *   |                                                          |
 *   |            FW Send response once processing is completed |
 *   | <---------- NNC passthrough command response ----------- |
 */
typedef struct nnc_cmd_work_in_progress_response {
  uint32_t type;
  uint32_t transaction_id; /* FW assigns a random number */
  uint8_t reserved[8];  /* Progress percentage or any future requirement */
} nnc_cmd_work_in_progress_response_t;

typedef struct nnc_cmd_progress_check_request {
  uint32_t type;
  /* Host should pass the transaction_id received in work_in_progress ind */
  uint32_t transaction_id;
} nnc_cmd_progress_check_request_t;

typedef struct nnc_cmd_status_query_response {
  uint32_t type;
  /**
   * Success indicates that all previously allocated resources have been
   * successfully returned to the pool.
   */
  nnc_status_code_t status_code;
  host_api_info_data_t info;
} nnc_cmd_status_query_response_t;

typedef struct nnc_transaction_status_request {
  nnc_transaction_header_t header;
} nnc_transaction_status_request_t;

// Response returned to the target
typedef struct nnc_transaction_status_response {
  nnc_transaction_header_t header;

  uint16_t major_version;
  uint16_t minor_version;
  /**
   * Success indicates the following:
   * - magic number provided is valid
   * - expected version matches that provided in the request
   * - sequence number has been successfully set to the value specified
   */
  nnc_status_code_t status_code;
  /**
   * Device flags that are required to be reported to kmd.
   * bit-0 : CRC check required for control messages. 1 for AUTO_SKU 0 otherwise
   */
  uint64_t device_flags;
} nnc_transaction_status_response_t;

typedef struct nnc_cmd_resource_info_request {
  uint32_t type;
  /**
   * ID of the process used by the host OS. This ID provided by host application
   * and may not be trusted.
   */
  uint32_t process_id;
  /**  resource_group_id is only valid if resource_type is
   *NNC_RESOURCE_TYPE_RESOURCE_GROUP
   *   otherwise system resource info is returned
   **/
  int32_t res_grp_id;
  uint32_t padding;
} nnc_cmd_resource_info_request_t;

/**
 * Response to create resource group
 */
typedef struct nnc_cmd_resource_info_response {
  uint32_t type;
  int32_t res_grp_id;
  host_api_resource_info_t resource_info;
  nnc_status_code_t status_code;
  uint32_t padding;
} nnc_cmd_resource_info_response_t;

/**
 * Query performance related info only
 */
typedef struct nnc_cmd_performance_info_request {
  uint32_t type;
  uint32_t padding;
} nnc_cmd_performance_info_request_t;

/**
 * Response to performance info query cmd
 */
typedef struct nnc_cmd_performance_info_response {
  uint32_t type;
  nnc_status_code_t status_code;
  host_api_performance_info_t performance_info;
} nnc_cmd_performance_info_response_t;

/**
 * Attestation Request is a pass through command for NNC.
 * Payload[] is directly transferred to XBL-Sec inside device.
 */
typedef struct nnc_cmd_attestation_request {
  uint32_t type;
  uint32_t size; // Size of data field
  uint8_t data[];
} nnc_cmd_attestation_request_t;

/**
 * Attestation Response is a pass through command for LRT.
 * Payload[] is directly transferred to QWES server for validation.
 */
typedef struct nnc_cmd_attestation_response {
  uint32_t type;
  uint32_t size; // Size of data field
  uint8_t data[];
} nnc_cmd_attestation_response_t;

#endif // AICLINUXRUNTIME_HOSTNNCTRANSACTIONMESSAGES_H
