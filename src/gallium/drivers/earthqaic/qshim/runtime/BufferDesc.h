// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef _QAIC_BUFFERDESC_H_
#define _QAIC_BUFFERDESC_H_

#include <stdint.h>

namespace qaic {

enum usageType_t {
  USAGE_INPUT = 0,
  USAGE_OUTPUT = 1,
  USAGE_INTERNAL = 2,
  USAGE_INVALID = 0xFFFFFFFF
};
static_assert(sizeof(usageType_t) == 4,
              "usageType_t is expected to be 4 bytes!");

enum memLoc_t { L2TCM = 0, VTCM = 1, DDR = 2, LOC_INVALID = 0xFFFFFFFF };
static_assert(sizeof(memLoc_t) == 4, "memLoc_t is expected to be 4 bytes!");

typedef struct {
  uint32_t offset;
  uint32_t size;
} BufferDescPartialHeader_t;

typedef struct {
  memLoc_t location;     // L2TCM, VTCM, DDR
  uint32_t offset;       // within location type/multicast space
  uint32_t size;         // bytes
  uint16_t waitDBNum;    // DB number to wait for for ready indication before
                         // accessing buffer
  uint16_t ioDBNum;      // DB number for ready indication to other QPC
  uint32_t waitDBVal;    // DB value to look for when waiting for ready
  uint32_t ioDBVal;      // DB value for ready indication to other QPC
  uint16_t ioMCID;       // MCID to use to send buffer data to other QPC
  uint16_t ioDBMCID;     // MCID to use to send buffer ready DB to other QPC
  uint16_t buffMCID;     // MCID to use to send buffer data within QPC
  uint16_t nspMask;      // Which NSPs have this buffer allocated (when not DDR)
  usageType_t usage;     // Input/Output/Internal usage
  uint32_t allowPartial; // If non-zero, buffer contains header
} BufferDesc_t;
static_assert(sizeof(BufferDesc_t) == 40,
              "BufferDesc_t is expected to be 40 bytes!");

const BufferDesc_t *getBufferInfo(int buffNum);
void *getBufferAddr(int buffNum);
uint32_t getBufferSize(int buffNum);
void broadcastToBuffer(int buffNum, int32_t dstOffset, int size,
                       const int8_t *src, const uint32_t *dbVal, int threadId,
                       bool waitForDone);
void broadcastToBuffer(int buffNum, int32_t dstOffset, int size,
                       const int8_t *src, int threadId, bool WaitForDone);
int inputBufferNum(int buffNum);
int outputBufferNum(int buffNum);
int internalBufferNum(int buffNum);

void udmaSubmitLinearDesc(int threadId, const int8_t *src, int size,
                          const int8_t *dst, const uint32_t *dbVal,
                          bool waitForDone, bool updateDbs, uint32_t dBNum,
                          bool order);
void udmaSubmit2DDesc(int threadId, const int8_t *src, const int8_t *dst,
                      int32_t height, int32_t width, int32_t destStride,
                      int32_t srcStride, const uint32_t *dbVal,
                      bool waitForDone, bool updateDbs, uint32_t DBNum,
                      bool order);
} // namespace qaic
#endif
