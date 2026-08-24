// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "NSPContext.h"
#include "BufferDesc.h"
#include "SerializedProgramDesc.h"
#include "libdev/libdev_assert.h"
#include "libdev/libdev_defs.h"
#include "libdev/os-inlines.h"

CoreInfo NSPContext;

void _nspContextInit(AICExecContext *ctx) {
  // AICExecContext
  NSPContext.virtualNSPId = ctx->virtualNSPId;
  NSPContext.baseL2TCM = ctx->baseL2TCM;
  NSPContext.baseVTCM = ctx->baseVTCM;
  NSPContext.baseConstantDataMem = ctx->baseConstantDataMem;
  NSPContext.baseSharedDDR = ctx->baseSharedDDR;
  NSPContext.baseL2CachedDDR = ctx->baseL2CachedDDR;
  NSPContext.semInfo = ctx->semaphoreListPtr;
  NSPContext.mcAddresses = ctx->mcAddresses;
  NSPContext.startTimeStamp = ctx->startTimeStamp;
  NSPContext.logFuncPtr = ctx->logFuncPtr;
  NSPContext.exitThread = ctx->exitThread;
  NSPContext.setPMUReg = ctx->setPMUReg;
  NSPContext.errFuncPtr = ctx->errFuncPtr;
  NSPContext.notifyHangPtr = ctx->notifyHangPtr;
  NSPContext.udmaReadFuncPtr = ctx->udmaReadFuncPtr;
  NSPContext.mmapFuncPtr = ctx->mmapFuncPtr;
  NSPContext.munmapFuncPtr = ctx->munmapFuncPtr;
  NSPContext.qdss_stm_port_vaddr = ctx->qdssSTMPortVaddr;
  // Other fields
}

void _udmaContextInit(int threadId) {
  // DMA descriptor setup
  CoreInfo *ctx = &NSPContext;
  const qaic::BufferDesc_t *udmaDescBuff =
      &qaic::_progBuffers[qaic::_progDesc->udmaDescBuffNum];
  uint32_t dmaDescStartOff = udmaDescBuff->offset;
  uint32_t dmaDescPerThreadSize =
      NUM_UDMA_CACHELINES_PER_THREAD * CACHE_LINE_SIZE;
  assert(udmaDescBuff->size ==
                qaic::_progDesc->numThreads * dmaDescPerThreadSize);

  ctx->dmaDescStart[threadId] = ctx->dmaDescNext[threadId] =
      (DMADescriptor *)&ctx
          ->baseL2TCM[dmaDescStartOff + dmaDescPerThreadSize * threadId];
  ctx->dmaDescEnd[threadId] =
      (DMADescriptor *)&ctx->baseL2TCM[dmaDescStartOff +
                                  dmaDescPerThreadSize * (threadId + 1)];

  // Initialize done bits to one so check in pushDMADesc will work the first
  // time a descriptor is used.
  for (DMADescriptor *desc = ctx->dmaDescStart[threadId],
            *descEnd = ctx->dmaDescEnd[threadId];
        desc != descEnd; ++desc) {
    desc->done = 1;
  }
#ifndef NDEBUG
  DMADescriptor *dummyStartDesc =
      (DMADescriptor
            *)(&ctx->baseL2TCM[qaic::_progDesc->udmaDummyStartDescOffset]);
#endif
  assert(dummyStartDesc->done == 1);

  ctx->resetDMATail(threadId);

  if (ctx->dmaDescStart[threadId] != nullptr) {
    assert(ctx->dmaDescStart[threadId] >=
                  (DMADescriptor *)&ctx->baseL2TCM[dmaDescStartOff]);
    // CoreInfo::getNextFreeDMADesc relies on dmaDescStart being aligned to
    // speed up code
    assert(
        (uintptr_t)(ctx->dmaDescStart[threadId]) % CACHE_LINE_SIZE == 0);
    assert(ctx->dmaDescEnd[threadId] <=
                  (DMADescriptor *)&ctx->baseL2TCM[libdev_l2tcm_size()]);
  }
}

void _udmaContextCleanup(int threadId) {
  CoreInfo *ctx = &NSPContext;
  os_udma_wait_done(ctx->dmaTail[threadId], threadId);
}

nsp_doorbell_t CoreInfo::exitDB() {
  uint32_t *dbs = (uint32_t *)this->baseL2TCM;
  return &dbs[qaic::_progDesc->exitDB];
}

CoreInfo *getNSPContext() { return &NSPContext; }
