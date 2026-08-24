/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef AICLINUXRUNTIME_HOSTNNCACTIVATEDEACTIVATEMESSAGES_H
#define AICLINUXRUNTIME_HOSTNNCACTIVATEDEACTIVATEMESSAGES_H

#include "nncErrors.h"
#include <stdint.h>

#define NNC_ACTIVATION_CMD_TYPE_MAX_COMMANDS (8)

typedef uint32_t activation_id_type_t;

typedef enum nnc_reservation_type {
  NNC_RESERVATION_NONE,
  NNC_RESERVATION_RESOURCE_GROUP,
  NNC_RESERVATION_ACTIVATION,
  NNC_RESERVATION_TYPE_ENUM_MAX = INT_MAX
} nnc_reservation_type_t;

typedef enum nnc_activation_state_type {
  NNC_ACTIVATION_STATE_CMD_STANDBY =
      0, //!< Transition Activation to Standby -> DCVS UP
  NNC_ACTIVATION_STATE_CMD_READY =
      1, //!< Transition Activation to Ready -> DCVS UP
  NNC_ACTIVATION_STATE_CMD_DEACTIVATE =
      2, //!< Transition to Deactivate and Low POWER -> DCVS DOWN
  NNC_ACTIVATION_STATE_INVAL = 3, //!< Transition Activation to Ready
  NNC_ACTIVATION_STATE_CMD_TYPE_MAX =
      4, //!< Maximum number of defined Activation Request
  NNC_ACTIVATION_STATE_ENUM_MAX = INT_MAX,
} nnc_activation_state_type_t;

/*****************************************************************************
 *  All the NNC protocol structures should end on a 8byte boundary and all   *
 *  its structure members must be naturally aligned to itself. It is the     *
 *  responsibility of the maintainer/programmer to make sure that the above  *
 *  criteria are met while updating the NNC protocol structures.            *
 * ***************************************************************************/

/**
 * Requests activation of a previously loaded elf, specified by elf_id.
 *
 * @note
 * This request will configure the data path and associated VC ID to be used for
 * sending/receiving network I/O.
 */
typedef struct nnc_cmd_activate_request {
  uint32_t type;
  /**
   * ID of the process used by the host OS. This ID provided by host application
   * and may not be trusted.
   */
  uint32_t process_id;
  /** Identifies the previously loaded network weights and biases. */
  uint32_t constants_id;
  /** Identifies the previously loaded network ELF. */
  uint32_t elf_id;
  /** // None, Resource Group or Activation Reservation */
  nnc_reservation_type_t reservation_type;
  /**  Resource Group or Activation Reservation */
  int32_t resource_res_id;
  /** VC reservation -1 for none */
  uint32_t opt_vc_setup; // Set to 1 if VC setup transaction is included
  int32_t vc_res_id;
  /** Define the initial activation state for this activation */
  nnc_activation_state_type_t initial_activation_state;
  /** Type nnc_dma_option_type_t of physical DMA channel mapping requested */
  uint32_t dma_type;

  /** Activation can be done for
   * - No oversubscription
   * - Oversubscription with Data Path Switching
   * - Oversubscription with Control Path Switching
   * Only in the case of Oversubscription with Data Path Switching is the
   * NNC Doorbell needed.
   * Activation process must know if this is required, this option is given
   * in the opt_support_datapath_switching
   * Set opt_support_datapath_switching to 1 if for datapath switching
   * the activation response will include valid info in the datapath doorbell
   */
  uint8_t opt_support_datapath_switching;
  /* Align to 64 bit boundary */
  uint8_t padding[7];
} nnc_cmd_activate_request_t;

/**
 * Response returned to the host after an nnc_activation_request.
 */
typedef struct nnc_cmd_activate_response {
  uint32_t type;
  /** Status of the nnc_activate_request. */
  nnc_status_code_t status_code;
  /**
   * Identifies the network just activated, if successful. A zero value
   * indicates an error occurred during activation. See the status_code
   * for more information on the error.
   */
  uint32_t activation_id;
  /**
   * ID of the bi-directional virtual channel to send and receive bulk data from
   * the running network.
   */
  uint32_t virtual_channel_id;
  uint64_t static_ddr_base;
  uint64_t dynamic_ddr_base;
  uint64_t mcid_base;
  /* mcid base address used for cmdObj dma buffers */
  uint64_t mcid_base_dma_cmd;
  /* Type of VC mapping nnc_dma_option_type_t, may differ from requested type */
  uint32_t dma_type;
  /**
   * Dedicated physical channel mask, valid if
   * dma_type = NNC_DMA_OPTION_DEDICATED
   */
  uint32_t pc_mask;
  /* Data path command NSP polling location address */
  uint64_t cmd_nsp_base_polling_addr;
  /* Size of each element in NSP Polling info array */
  uint32_t cmd_nsp_polling_info_size;
  /* Data path command nsp polling data */
  uint32_t cmd_nsp_polling_data;
  /* Data path command doorbell */
  uint64_t cmd_doorbell_addr;
  /* Data path command doorbell length, how much data is valid in doorbell_data
     This value is one of the nnc_doorbell_length_enum */
  uint8_t cmd_doorbell_length;
  /* Field set to true if the cmd_doorbell data is valid, otherwise data should
   * be ignored*/
  uint8_t cmd_doorbell_valid;
  /* Bit mask of Physical NSP IDs*/
  uint32_t nspMask;
  /* Align to 64 bits */
  uint8_t padding[2];

} nnc_cmd_activate_response_t;

/**
 * Requests deactivation of a previously activated network. No resources used by
 * this networ will be released automatically.  Use nnc_cmd_unload_elf_request
 * and nnc_cmd_unload_constants_request to request the unloading of the
 * resources used by this network.
 */
typedef struct nnc_cmd_deactivate_request {
  uint32_t type;
  /**
   * Identifies the currently active network that should be deactivated.
   */
  uint32_t activation_id;
  uint32_t opt_vc_teardown; // Set to 1 if VC teardown transaction is included
  uint32_t padding;
} nnc_cmd_deactivate_request_t;

/**
 * Response returned to the host after an nnc_deactivation_request
 */
typedef struct nnc_cmd_deactivate_response {
  uint32_t type;
  /**
   * Success indicates that all previously allocated resources have been
   * successfully returned to the pool.
   */
  nnc_status_code_t status_code;
} nnc_cmd_deactivate_response_t;

/**
 * Activation State Command Request
 * This command allows the caller to request a state transition on an existing
 * Activation
 */
typedef struct nnc_activation_state_cmd {
  activation_id_type_t acid;
  nnc_activation_state_type_t cmd;
} nnc_activaiton_state_cmd_t;

typedef struct nnc_activation_state_cmd_request {
  uint32_t type;
  uint32_t num_cmds;
  nnc_activaiton_state_cmd_t ac_cmd[NNC_ACTIVATION_CMD_TYPE_MAX_COMMANDS];
} nnc_activation_state_cmd_request_t;

/**
 *  Activation State Command Request
 */
typedef struct nnc_activation_state_cmd_response {
  uint32_t type;
  nnc_status_code_t status_code;
} nnc_activation_state_cmd_response_t;

#endif // AICLINUXRUNTIME_HOSTNNCACTIVATEDEACTIVATEMESSAGES_H
