/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef AICLINUXRUNTIME_HOSTNNCRESOURCERESERVATIONMESSAGES_H
#define AICLINUXRUNTIME_HOSTNNCRESOURCERESERVATIONMESSAGES_H

#include "nncErrors.h"
#include <stdint.h>

#define NNC_VC_RESERVATION_ID_DEFAULT (-1)
#define NNC_VC_RESERVATION_ID_SKIP (0xFE)

typedef enum nnc_cmd_device_config_options {
  NNC_DEV_CONFIG_DMA_PC = 1, //!< DMA Physical channel config
  NNC_DEV_CONFIG_MAX =
      NNC_DEV_CONFIG_DMA_PC, //!< Maximum number of defined config options
  NNC_INVAL_CONFIG = INT_MAX,
} nnc_cmd_device_config_options_t;

/**
 * Requests virtual channel reservation.
 *
 * @note
 * This request will configure the data path and associated VC ID to be used for
 * sending/receiving network I/O.
 */
typedef struct nnc_cmd_create_vc_reservation_request {
  uint32_t type;
  /**
   * ID of the process used by the host OS. This ID provided by host application
   * and may not be trusted.
   */
  uint32_t process_id;
  /** Identifies the previously loaded network weights and biases. 0 means no
   * constants. */
  /* Number of Semaphores to be used with this Virtual Channel*/
  /* Semaphores must be assigned in the VC programming */
  uint32_t num_sem;
  int32_t source_grp_id;
  /** Type nnc_dma_option_type_t of physical DMA channel mapping requested */
  uint32_t dma_option;
  /* Size of DDR block for datapath command and response*/
  uint32_t mem_cmd_block_size;
  /* Set to true (1) to include assignment of special use semaphore */
  uint8_t assign_special_use_sem;
  /* padding to align to 64 bit */
  uint8_t padding[7];
} nnc_cmd_create_vc_reservation_request_t;

/**
 * Response returned to the host after an nnc_cmd_create_vc_reservation_request.
 */
typedef struct nnc_cmd_create_vc_reservation_response {
  uint32_t type;
  /** Status of the nnc_reserve_vc_request. */
  nnc_status_code_t status_code;

  /**
   * Reservation ID to be used in activation reservation
   */
  uint32_t vcresid;
  /**
   * ID of the bi-directional virtual channel to send and receive bulk data from
   * the running network.
   */
  uint32_t virtual_channel_id;
  /* Type of VC mapping nnc_dma_option_type_t, may differ from requested type */
  uint32_t dma_type;
  /**
   * Dedicated physical channel mask, valid if
   * dma_type = NNC_DMA_OPTION_DEDICATED
   */
  uint32_t pc_mask;
  /* boolean value set to true if special_use_id included, othewise ignore
   * special_use_sem_id field */
  uint8_t special_use_sem_id_included;
  /* Align to 64 bits. */
  uint8_t padding;
  /* Datapath switching semaphore ID */
  uint16_t special_use_sem_id;
  /* Size of DDR block for datapath command and response */
  uint32_t mem_cmd_block_size;
  /* Address of DDR block for datapath command and response */
  uint64_t mem_cmd_block_base_addr;
} nnc_cmd_create_vc_reservation_response_t;

/**
 * Requests vc reservation release  of a previously reserved VC.
 */
typedef struct nnc_cmd_release_vc_reservation_request {
  uint32_t type;
  /**
   * Identifies the currently active reservation.
   */
  uint32_t vcresid;
} nnc_cmd_release_vc_reservation_request_t;

/**
 * Response returned to the host after an nnc_release_vc_request
 */
typedef struct nnc_cmd_release_vc_reservation_response {
  uint32_t type;
  /**
   * Success indicates that all previously allocated resources have been
   * successfully returned to the pool.
   */
  nnc_status_code_t status_code;
} nnc_cmd_release_vc_reservation_response_t;

/**
 * Requests create activation reservation.
 *
 * @note
 * This request will configure the data path and associated VC ID to be used for
 * sending/receiving network I/O.
 */
typedef struct nnc_cmd_create_activation_reservation_request {
  uint32_t type;
  /**
   * ID of the process used by the host OS. This ID provided by host application
   * and may not be trusted.
   */
  uint32_t process_id;
  /**  Identifies process */
  int32_t source_grp_id;
  uint32_t padding;

  uint64_t static_shddr_size;
  uint64_t dynamic_shddr_size;
  uint32_t num_nsp;
  uint32_t num_mcid;
  uint64_t l2region_size;
  /* Number of networks for which the activation reservation is done */
  uint16_t num_networks;
  uint8_t padding2[6];
} nnc_cmd_create_activation_reservation_request_t;

/**
 * Response returned to the host after a
 * nnc_cmd_create_activation_reservation_request
 */
typedef struct nnc_cmd_create_activation_reservation_response {
  uint32_t type;
  /** Status of the nnc_reserve_vc_request. */
  nnc_status_code_t status_code;

  /**
   * Reservation ID to be used in activation reservation
   */
  int32_t ac_res_id;
  /**
   * ID of the activation reservation
   */
  uint16_t pnsp_mask; // pnsp mask for all NSPs in the PG
  /**
   * ID of the activation reservation
   */
  uint8_t padding[2];

} nnc_cmd_create_activation_reservation_response_t;

/**
 * Requests release activation reservation.
 */
typedef struct nnc_cmd_release_activation_reservation_request {
  uint32_t type;
  /**
   * Identifies the currently active reservation.
   */
  int32_t ac_res_id;
} nnc_cmd_release_activation_reservation_request_t;

/**
 * Response returned to the host after a
 * nnc_cmd_release_activation_reservation_request
 */
typedef struct nnc_cmd_release_activation_reservation_response {
  uint32_t type;
  /**
   * Success indicates that all previously allocated resources have been
   * successfully returned to the pool.
   */
  nnc_status_code_t status_code;
} nnc_cmd_release_activation_reservation_response_t;

typedef struct nnc_cmd_update_ac_reservation_request {
  uint32_t type;
  /**
   * ID of the process used by the host OS. This ID provided by host application
   * and may not be trusted.
   */
  uint32_t process_id;
  /**
   * Reservation ID to be used in update of activation reservation
   */
  int32_t ac_res_id;
  /**
   * Specifies the number of nsps for the active reservation.
   */
  uint32_t num_nsp;
  /** reserved */
  uint64_t reserved;
} nnc_cmd_update_ac_reservation_request_t;

typedef struct nnc_cmd_update_ac_reservation_response {
  uint32_t type;
  /** Status of the nnc_update_ac_reservation_request. */
  nnc_status_code_t status_code;
} nnc_cmd_update_ac_reservation_response_t;

/**
 *  Configure physical DMA channels to be dedicated to single virtual channel
 */
typedef struct nnc_cmd_device_config_request {
  uint32_t type;
  /**
   * ID of the process used by the host OS. This ID provided by host application
   * and may not be trusted.
   */
  uint32_t process_id;
  /** DMA Physical channel config */
  nnc_cmd_device_config_options_t config_options;
  uint32_t num_pc_dedicated;
} nnc_cmd_device_config_request_t;

typedef struct nnc_cmd_device_config_response {
  uint32_t type;
  /** Status of the nnc_device_config_request. */
  nnc_status_code_t status_code;
} nnc_cmd_device_config_response_t;

#endif // AICLINUXRUNTIME_HOSTNNCRESOURCERESERVATIONMESSAGES_H
