// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "BufferDesc.h"
#include "ComputeAPI.h"
#include "NSPContext.h"
#include "SerializedProgramDesc.h"
#include "libdev/libdev_defs.h"

namespace qaic {

inline const BufferDesc_t *_getBufferInfo(int buffNum) {
  return &_progBuffers[buffNum];
}

const BufferDesc_t *getBufferInfo(int buffNum) {
  return _getBufferInfo(buffNum);
}

BufferDescPartialHeader_t *getBufferBase(const BufferDesc_t *buff,
                                         uint32_t buffNum) {
  uint8_t *base;
  CoreInfo *ctx = getNSPContext();
  switch (buff->location) {
  case L2TCM:
    base = ctx->baseL2TCM;
    break;
  case VTCM:
    base = ctx->baseVTCM;
    break;
  case DDR:
    base = ctx->baseSharedDDR;
    break;
  default:
    ERR_FATAL(ctx->errFuncPtr, "Unknown buffer location type %d",
              buff->location, 0, 0);
    __builtin_unreachable();
  }
  if (buff->location != DDR && !(buff->nspMask & (1 << ctx->virtualNSPId))) {
    ERR_FATAL(ctx->errFuncPtr, "NSP%d doesn't have access to buffNum %d",
              ctx->virtualNSPId, buffNum, 0);
    __builtin_unreachable();
  }
  return (BufferDescPartialHeader_t *)(buff->offset + base);
}

void *getBufferAddr(int buffNum) {
  const BufferDesc_t *buff = _getBufferInfo(buffNum);
  BufferDescPartialHeader_t *pHeader = getBufferBase(buff, buffNum);
  if (buff->allowPartial && (buff->usage == USAGE_INPUT)) {
    if (!isBufferValid(buffNum)) {
      CoreInfo *ctx = getNSPContext();
      ERR_FATAL(ctx->errFuncPtr,
                "NSP%d trying to access buffNum %d which is partial, but isn't "
                "currently valid",
                ctx->virtualNSPId, buffNum, 0);
      __builtin_unreachable();
    }
    return (void *)((uintptr_t)pHeader + pHeader->offset);
  } else {
    return (void *)pHeader;
  }
}

uint32_t getBufferSize(int buffNum) {
  const BufferDesc_t *buff = _getBufferInfo(buffNum);
  if (buff->allowPartial && (buff->usage == USAGE_INPUT)) {
    if (!isBufferValid(buffNum)) {
      CoreInfo *ctx = getNSPContext();
      ERR_FATAL(ctx->errFuncPtr,
                "NSP%d trying to access buffNum %d which is partial, but isn't "
                "currently valid",
                ctx->virtualNSPId, buffNum, 0);
      __builtin_unreachable();
    }
    BufferDescPartialHeader_t *pHeader = getBufferBase(buff, buffNum);
    return pHeader->size;
  } else {
    return buff->size;
  }
}

void broadcastToBuffer(int buffNum, int32_t dstOffset, int size,
                       const int8_t *src, const uint32_t *dbVal, int threadId,
                       bool waitForDone) {
  const BufferDesc_t *buff = &_progBuffers[buffNum];
  CoreInfo *ctx = getNSPContext();
  assert(((uint32_t)(dstOffset + size) <= buff->size) &&
         "Broadcast would overrun target buffer!");

  // split large (UDMAMaxSize) transfers into multiple transfers
  while (size) {
    int transferSize = (size < UDMAMaxSize) ? size : UDMAMaxSize;
    bool last = (transferSize == size);

    // Detect broadcast to self, since we can't multicast to self
    if (buff->nspMask == (0x1 << ctx->virtualNSPId)) {
      int8_t *dst = (int8_t *)getBufferAddr(buffNum) + dstOffset;
      bool toUncached = buff->location == DDR;
      bool fromUncached =
          !((src >= (int8_t *)ctx->baseL2TCM &&
             src < ((int8_t *)ctx->baseL2TCM + libdev_l2tcm_size())) ||
            (src >= (int8_t *)ctx->baseVTCM &&
             src < ((int8_t *)ctx->baseVTCM +
                    libdev_vtcm_size()))); // src->location == DDR;

      libdev_copyVTCM(dst, src, transferSize, toUncached, fromUncached, dbVal,
                      /*dbOnlyCheckedLocally*/ true, /*updateDBs*/ last,
		      /*doRelease*/ true, threadId, buff->waitDBNum);
    }
    // Broadcast to the other NSPs
    if (buff->nspMask & ~(0x1 << ctx->virtualNSPId)) {
      int8_t *dst =
          (int8_t *)ctx->mcBaseAddresses[buff->buffMCID] + buff->offset;

      libdev_multicastVTCM(dst, dstOffset, src, transferSize, dbVal,
                           /*dbOnlyCheckedLocally*/ false, /*updateDBs*/ last,
			   threadId, buff->waitDBNum, /*mcId*/ 0);
    }

    size -= transferSize;
  }
  if (waitForDone)
    os_udma_wait();
}

void broadcastToBuffer(int buffNum, int32_t dstOffset, int size,
                       const int8_t *src, int threadId, bool waitForDone) {
  const BufferDesc_t *buff = &_progBuffers[buffNum];
  broadcastToBuffer(buffNum, dstOffset, size, src, &buff->waitDBVal, threadId,
                    waitForDone);
}

int inputBufferNum(int buffNum) {
  assert(buffNum < _progDesc->numInputBuffs);
  return buffNum;
};

int outputBufferNum(int buffNum) {
  assert(buffNum < _progDesc->numOutputBuffs);
  return buffNum + _progDesc->numInputBuffs;
}

int internalBufferNum(int buffNum) {
  assert(buffNum < _progDesc->numInternalBuffs);
  return buffNum + _progDesc->numInputBuffs + _progDesc->numOutputBuffs;
}

void udmaSubmitLinearDesc(int threadId, const int8_t *src, int size,
                          const int8_t *dst, const uint32_t *dbVal,
                          bool waitForDone, bool updateDBs, uint32_t dBNum,
                          bool order) {
    (void)threadId;
    (void)src;
    (void)size;
    (void)dst;
    (void)dbVal;
    (void)waitForDone;
    (void)updateDBs;
    (void)dBNum;
    (void)order;
}

void udmaSubmit2DDesc(int threadId, const int8_t *src, const int8_t *dst,
                      int32_t height, int32_t width, int32_t dstStride,
                      int32_t srcStride, const uint32_t *dbVal,
                      bool waitForDone, bool updateDBs, uint32_t dBNum,
                      bool order) {
    (void)threadId;
    (void)src;
    (void)dst;
    (void)height;
    (void)width;
    (void)dstStride;
    (void)srcStride;
    (void)dbVal;
    (void)waitForDone;
    (void)updateDBs;
    (void)dBNum;
    (void)order;
}

} // namespace qaic
