/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef JIT_COMMON_H
#define JIT_COMMON_H

#ifdef __cplusplus
#include <stdint.h>
#include <assert.h>
extern "C" {
#else
#include <stdint.h>
#include <assert.h>
#endif

#include "jit_device_stats.h"

#define AIC_RESPONSE_SLOT_SIZE 128
#define AIC_RESPONSE_STATUS_SUCCESS 0
#define AIC_CMD_RESPONSE_ERR_MSG_LEN 112
#define QSHIM_CONFIG_L2TCM_OFFSET (1024)
#define JIT_HEAP_MAX_ALPS 8

// CommandSet request

enum { CommandSetRequestCanary = 0x9C039A1C };

typedef struct AicCommandSetResponseInfo {
  uint64_t offset; // Offset in bytes from base of static DDR
  uint64_t size;   // size of response
} AicCommandSetResponseInfo_t;

typedef struct AicCommandSetRequest {
  uint32_t canary; // Must be CommandSetRequestCanary
  uint32_t length; // Include this header
  AicCommandSetResponseInfo_t cmdSetRespInfo;
  uint8_t body[]; // CommandSet body in FlabBuf
} AicCommandSetRequest_t;

// AIC CommandSet response is composed of a header and optionally a list of
// TLVs for command responses. The header and TLVs are of integer slot size.
// Each slot is of AIC_RESPONSE_SLOT_SIZE bytes.

enum ResponseType {
  CommandResponse,
  ContinuationResponse,
  ProfilingData,
  MaxResponse
};

typedef struct ResponseTLV {
  uint32_t type;
  uint32_t slotCount;
  uint8_t value[];
} ResponseTLV_t;

typedef struct ProfilingData {
  uint32_t commandSetID;
  uint32_t numTraceEvent;
  // Set to 1 if more data point requests encountered than allowed by allocated
  // profiling memory
  uint8_t dataOverFlow;
  uint8_t padding[3];
  // Profiling Data for this command
  uint32_t size; // Including this header
  uint8_t profilingData[];
} ProfilingData_t;

typedef struct CommandValue {
  uint32_t commandID;
  uint32_t status;
  uint8_t data[];
} CommandValue_t;

typedef struct DevMemChunk {
  uint64_t phy_addr;
  uint64_t size;
  uint32_t bufId;
  uint8_t padding[4];
} DevMemChunk_t;

typedef struct DevMemChunkList {
  uint32_t numChunks;
  uint8_t padding[4];
  DevMemChunk_t chunks[];
} DevMemChunkList_t;

typedef struct AllocBufferData {
  uint64_t devAddr;   // Device DMA address
  uint32_t devCookie; // Cookie to be sent back to free the buffer
} AllocBufferData_t;

typedef struct PreloadExeData {
  uint64_t execInfoOffset; // Offset from the static DDR for ExecInfo which
                           // contains information to track a device executable.
                           // LRT passes this field to each NSP for them to find
                           // this executable.
  uint64_t roSegmentAddr;  // DDR address allocated to for the Read-Only segment
  uint32_t numRwSegments;  // Number of Read-Write segments. Must match the
                           // rwSegmentNum in AicCmdPreloadExe.
  uint8_t padding[4];
  uint64_t rwSegmentPointers[]; // DDR addresses allocated for each Read-Write
                                // segments.

} PreloadExeData_t;

typedef struct ContinuationValue {
  uint64_t startAddr;
  uint64_t length;
} ContinuationValue_t;

// The header is one slot and contains the response for the CommandBuffer. The
// commandID is the same as the commandID in CommandBuffer. The status is 0 for
// success, non-zero for failure. The status is exclusively for CommandBuffer
// initialization, and for situations not covered by responses of commands. In
// other words you can have success status in header, but error status in
// command responses.

typedef struct AicCommandSetResponseHeader {
  uint32_t size;
  uint32_t commandID; // Help us identify NSP
  uint32_t status;    // JitStatusCode_t
  uint32_t tlvCount;  // 0 when no prof but >0 otherwise
  // AIC_RESPONSE_SLOT_SIZE - 4*sizeof(uint32_t)
  char errMsg[AIC_CMD_RESPONSE_ERR_MSG_LEN];
} AicCommandSetResponseHeader_t;

typedef struct AicCommandSetResponse {
  AicCommandSetResponseHeader_t header;
  ResponseTLV_t responseTLV[];
} AicCommandSetResponse_t;

typedef struct AicQShimInitConfig {
  uint32_t reqBufferSize;
  uint32_t ringBufferSize;
  uint32_t opTimeoutMs;
} AicQShimInitConfig_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JIT_COMMON_H
