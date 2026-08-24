// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "libdev_interface.h"

#include "libdev_defs.h"

CoreInfo *libdev_getcontext() { return getNSPContext(); }

DMADescriptor *CoreInfo::getNextFreeDMADesc(int threadId) {
  const int descStride = CACHE_LINE_SIZE / sizeof(DMADescriptor);

  auto *desc = dmaDescNext[threadId];
  // Stride through desc ring buffer by cache line to avoid store release
  // stalling setup of next desc
  auto *next = desc + descStride;
  if (next >= dmaDescEnd[threadId]) {
    // Advance to next column of descriptors
    uintptr_t dmaDescStartAddr = (uintptr_t)dmaDescStart[threadId];
    // Each thread's buffer is CACHE_LINE_SIZE aligned so we don't need to
    // handle here

    uintptr_t cacheLineOffset = (uintptr_t)(desc + 1) & (CACHE_LINE_SIZE - 1);
    next = (DMADescriptor *)(dmaDescStartAddr + cacheLineOffset);
  }

  dmaDescNext[threadId] = next;

  uintptr_t descAddr = (uintptr_t)desc;
  (void)descAddr;
  assert(descAddr >= (uintptr_t)dmaDescStart[threadId] &&
                descAddr + sizeof(DMADescriptor) <=
                    (uintptr_t)dmaDescEnd[threadId]);

  // Wait until DMA descriptor is done being used in case we use up all
  // available descriptors.
  if (!desc->done) {
    os_udma_wait_done(desc, threadId);
  }

  return desc;
}

void CoreInfo::fillOutDMADesc(DMADescriptor *desc, DMADescriptor *next, int8_t *dst,
                              int32_t dstOffset, const int8_t *src,
                              unsigned size, bool destBypass, bool srcBypass,
                              bool order,
                              __attribute__((unused)) bool isMulticast,
                              __attribute__((unused)) int threadId) {
  assert(size <= UDMAMaxSize);
  desc->dword0 =
      DMA_DESC_DWORD0((uintptr_t)next, DMA_LINEAR_DESC_WORD1(size, destBypass,
                                                             srcBypass, order));
  desc->dword1 = DMA_DESC_DWORD1((uintptr_t)src, (uintptr_t)dst + dstOffset);
}

// Need copies of DB update command that take in DBNum and global mcId
void CoreInfo::fillOutDMADescLocalDBUpdate(DMADescriptor *desc, DMADescriptor *next,
					   const uint32_t *dbVal, bool order,
					   int threadId, uint32_t DBNum) {

  nsp_doorbell_t localDBDst = (nsp_doorbell_t)getNSPContext()->baseL2TCM;

  int dstOffset = DBNum * DB_SIZE;
  int size = DB_SIZE;
  bool destBypass = false;
  bool srcBypass = false;
  bool isMulticast = false;
  fillOutDMADesc(desc, next, (int8_t *)localDBDst, dstOffset,
                 (const int8_t *)dbVal, size, destBypass, srcBypass, order,
                 isMulticast, threadId);
}

void CoreInfo::fillOutDMADescGlobalDBUpdate(DMADescriptor *desc, DMADescriptor *next,
					    const uint32_t *dbVal, bool order,
					    int threadId, uint32_t DBNum,
					    uint32_t mcId) {
  void *globalDBDst = mcBaseAddresses[mcId];
  int globalDBDstOffset = DBNum * DB_SIZE;

  int size = DB_SIZE;
  bool destBypass = true;
  bool srcBypass = false;
  bool isMulticast = true;
  fillOutDMADesc(desc, next, (int8_t *)globalDBDst, globalDBDstOffset,
                 (const int8_t *)dbVal, size, destBypass, srcBypass, order,
                 isMulticast, threadId);
}

// Submit the DMA descriptor list desc->...->tail to this thread's DMA engine
// for processing.
void CoreInfo::submitDMAs(DMADescriptor *desc, DMADescriptor *tail, bool doRelease,
                          int threadId) {
  assert(dmaTail[threadId] != nullptr);
  assert(desc != nullptr);

  os_udma_link(dmaTail[threadId], desc, doRelease);
  dmaTail[threadId] = tail;
}

void CoreInfo::resetDMATail(int threadId) {
  DMADescriptor *dummyStartDesc =
      (DMADescriptor *)(&getNSPContext()
                   ->baseL2TCM[qaic::_progDesc->udmaDummyStartDescOffset]);
  dmaTail[threadId] = dummyStartDesc;
}
