/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef NSP_PROTOCOL_H
#define NSP_PROTOCOL_H

#include <stdint.h>

#pragma pack(push, 1)

#define NSP_CMD_NON_NPL_DMA_DESCRIPTORS (4)
// Indices to DMA Descriptor Buffer Arrays of cmdHandle_
// Positive index offsets from the beginning of the array
// Negative index offsets from end of array
// For more info, see QICmdObj
#define NSP_CMD_REQ_DMA_DESCR_INDEX (0)
#define NSP_CMD_EDB_DMA_DESCR_INDEX (1)
#define NSP_NPL_REQ_DMA_DESCR_INDEX (2)
#define NSP_CMD_RSP_DMA_DESCR_INDEX (-2)
#define NSP_CMD_DUMMY_DMA_DESCR_INDEX (-1)

typedef uint32_t NspCmd_t;

enum NspCmdType {
  NSP_CMD_DEACTIVATE = 1,
  NSP_CMD_SWITCH = 2,
  NSP_CMD_ACTIVATE_NW_THREAD = 3,
  NSP_CMD_LOOPBACK = 4
};

typedef struct NspCmdPacket {
  NspCmd_t nsp_cmd;  /* NspCmdType */
  uint32_t cmd_data; /* Nsp Cmd Data */
} NspCmdPacket_t;

typedef struct NspResponsePacket {
  int32_t rsp_data;
  uint32_t nsp_mask;
} NspResponsePacket_t;

typedef struct NspCmdResponsePacketVersion1 {
  NspCmdPacket_t cmd;
  NspResponsePacket_t rsp;
} NspCmdResponsePacketVersion1_t;

typedef struct NspCmdRspBlock {
  uint32_t version;  /* 1 */
  uint32_t reserved; /* 64 bit alignment */
  union UNNAMED {
    NspCmdResponsePacketVersion1_t cmdResponseV1;
  } cmdRsp;
} NspCmdRspBlock_t;

#pragma pack(pop)

#endif
