/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef AICLINUXRUNTIME_HOSTNNCRESOURCEGROUPMESSAGES_H
#define AICLINUXRUNTIME_HOSTNNCRESOURCEGROUPMESSAGES_H

#include "nncErrors.h"
#include <stdint.h>

/*****************************************************************************
 *  All the NNC protocol structures should end on a 8byte boundary and all   *
 *  its structure members must be naturally aligned to itself. It is the     *
 *  responsibility of the maintainer/programmer to make sure that the above  *
 *  criteria are met while updating the NNC protocol structures.            *
 * ***************************************************************************/

/**
 * Requests create resource group
 */
typedef struct nnc_cmd_create_resource_group_request {
  uint32_t type;
  /**
   * ID of the process used by the host OS. This ID provided by host application
   * and may not be trusted.
   */
  uint32_t process_id;
  /** Identifies process */
  /**
   *  res_grp_id must be a valid Id generated from an active reservation, or
   * set to RESOURCE_GROUP_ID_DEFAULT for the default system resource
   * pool
   */
  int32_t source_grp_id;
  /** Resource Information */
  uint32_t num_nsp;
  uint32_t num_mcid;
  uint32_t num_sem;
  uint32_t num_vc;
  uint32_t padding;
  uint64_t static_shddr_size;
  uint64_t dynamic_shddr_size;
  uint64_t l2region_size;
  uint64_t mem_size;
  /* Number of networks for which the resource reservation is done */
  uint16_t num_networks;
  uint8_t padding2[6];
} nnc_cmd_create_resource_group_request_t;

/**
 * Response to create resource group
 */
typedef struct nnc_cmd_create_resource_group_response {
  uint32_t type;
  /** Status of the nnc_reserve_vc_request. */
  nnc_status_code_t status_code;
  /**
   * Resource group ID
   */
  int32_t res_grp_id;
  uint32_t padding;
} nnc_cmd_create_resource_group_response_t;

/**
 * Requests release resource group
 */
typedef struct nnc_cmd_release_resource_group_request {
  uint32_t type;
  /**
   * Resource Group ID
   */
  int32_t res_grp_id;
} nnc_cmd_release_resource_group_request_t;

/**
 * Response returned to the host from nnc_cmd_release_resource_group_request
 */
typedef struct nnc_cmd_release_resource_group_response {
  uint32_t type;
  /**
   * Success indicates that all previously allocated resources have been
   * successfully returned to the pool.
   */
  nnc_status_code_t status_code;
} nnc_cmd_release_resource_group_response_t;

#endif // AICLINUXRUNTIME_HOSTNNCRESOURCEGROUPMESSAGES_H
