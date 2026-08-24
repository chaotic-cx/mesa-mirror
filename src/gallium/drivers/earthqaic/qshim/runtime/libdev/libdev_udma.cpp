// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "libdev_assert.h"
#include "libdev_defs.h"

void libdev_copy_dma_doorbells(
    int8_t *dst, int32_t dstOffset, const int8_t *src, unsigned size,
    bool isMulticast, bool destBypass, bool srcBypass, int threadId,
    const uint32_t *dbVal, bool dbOnlyCheckedLocally, bool updateDBs,
    bool noPayload, bool doRelease, uint32_t DBNum, uint32_t mcId) {
  // The noPayload == true path needs further testing before it can be used.
  assert(noPayload == false);

  CoreInfo *ctx = libdev_getcontext();

  // Large DMAs should have been split at the IR level.
  assert(size <= UDMAMaxSize);

  bool doLocalDBUpdate = updateDBs;
  bool doGlobalDBUpdate = doLocalDBUpdate && !dbOnlyCheckedLocally;

  DMADescriptor *payloadDesc =
      !noPayload ? ctx->getNextFreeDMADesc(threadId) : nullptr;
  DMADescriptor *localDBDesc =
      doLocalDBUpdate ? ctx->getNextFreeDMADesc(threadId)
                      : nullptr;
  DMADescriptor *globalDBDesc =
      doGlobalDBUpdate ? ctx->getNextFreeDMADesc(threadId)
                       : nullptr;

  assert(payloadDesc || localDBDesc);
  DMADescriptor *head = noPayload ? localDBDesc : payloadDesc;
  DMADescriptor *tail = doGlobalDBUpdate
                            ? globalDBDesc
                            : (doLocalDBUpdate ? localDBDesc : payloadDesc);

  // Payload data
  if (!noPayload) {
    assert(size != 0);
    bool order = false;
    ctx->fillOutDMADesc(payloadDesc, localDBDesc, dst, dstOffset, src, size,
                        destBypass, srcBypass, order, isMulticast, threadId);
  } else {
    assert(size == 0);
  }

  if (doLocalDBUpdate) {
    bool order = true;
    ctx->fillOutDMADescLocalDBUpdate(localDBDesc, globalDBDesc, dbVal, order,
                                     threadId, DBNum);

    if (doGlobalDBUpdate) {
      bool order = false;
      ctx->fillOutDMADescGlobalDBUpdate(globalDBDesc, nullptr, dbVal, order,
                                        threadId, DBNum, mcId);
    }
  }

  ctx->submitDMAs(head, tail, doRelease, threadId);

}

// Doesn't do multicast, so only between DDR/own VTCM
void libdev_copyVTCM(int8_t *dst, const int8_t *src, int size,
                     bool toUncached, bool fromUncached, const uint32_t *dbVal,
		     bool dbOnlyCheckedLocally, bool updateDBs, bool doRelease,
		     int threadId, uint32_t DBNum) {
  // Check that there is no overlap between src and dst buffers.
  assert((dst < src && dst + size <= src) ||
                (src < dst && src + size <= dst));

  bool isMulticast = false;
  int dstOffset = 0;
  bool destBypass = toUncached;
  bool srcBypass = fromUncached;
  libdev_copy_dma_doorbells(dst, dstOffset, src, size, isMulticast, destBypass,
			    srcBypass, threadId, dbVal,
			    dbOnlyCheckedLocally, updateDBs,
			    /*noPayload=*/false, doRelease, DBNum, 0);
}

void libdev_multicastVTCM(int8_t *dst, int32_t dstOffset, const int8_t *src,
                          int size, const uint32_t *dbVal,
                          bool dbOnlyCheckedLocally, bool updateDBs,
                          int threadId, uint32_t DBNum, uint32_t mcId) {
  void *dstBase = dst;
  bool isMulticast = true;
  bool destBypass = true;
  bool srcBypass = false;
  libdev_copy_dma_doorbells(
      (int8_t *)dstBase, dstOffset, src, size, isMulticast, destBypass,
      srcBypass, threadId, dbVal, dbOnlyCheckedLocally, updateDBs,
      /*noPayload=*/false, /*doRelease=*/true, DBNum, mcId);
}
