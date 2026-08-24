// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "os.h"

#include "../AICDefsInternal.h"
#include "libdev_assert.h"
#include "libdev_interface.h"
#include "os-dbg.h"
#include "os-inlines.h"

#include <inttypes.h>

typedef __fp16 float16;

void debugPrintUdmaDesc(const DMADescriptor *p) {
  DBG_PRINT_INFO("udmaDesc @0x%p: 0x%08" PRIx32 "->0x%08" PRIx32 " %uB", p,
                 p->src, p->dst, p->length);
  DBG_PRINT_INFO(
      "\tsrc: bypass=%d  dst: bypass=%u  order=%u done=%u next=0x%08" PRIx32,
      p->srcBypass, p->destBypass, p->order, p->done, p->next);
}

extern "C" {

static unsigned long long getElapsedCycles(uint64_t startCount,
                                           uint64_t endCount,
                                           bool satCheck = false) {
  unsigned long long result = 0;
  if (endCount > startCount)
    result = endCount - startCount;
  else
    result = ~(startCount - endCount) + 1;
  if (satCheck && result > 0xFFFFFFFF)
    return 0xFFFFFFFF;
  else
    return result;
}

// Returns true if timeout was hit.
void os_timeout_check(OSTimeoutCheckContext *timeoutCtx, uint32_t dbval,
                      bool doTimeoutCheck, bool debugLog) {
  int threadId = timeoutCtx->threadId;
  nsp_doorbell_t db = timeoutCtx->db;
  uint32_t expectedVal = timeoutCtx->expectedVal;
  CoreInfo *ctx = libdev_getcontext();
  nsp_doorbell_t exitDB = ctx->exitDB();

  // Check the exit DB
  if (os_doorbell_read1b(exitDB) == 1) {
    if (ctx->exitThread) {
      if (debugLog) {
        DBG_PRINT_INFO("t%d: thread received exit doorbell, exiting", threadId);
      }
      ctx->exitThread();
    }
    return;
  }

  // Check the DMA engine for errors.
  os_udma_poll();

  int waitTimeoutLogMS = ctx->waitTimeoutLogMS;
  if (waitTimeoutLogMS == 0) // Timeout is disabled
    return;

  // Write ulog message if this thread has been waiting for a long time
  if (timeoutCtx->waitStart == 0)
    timeoutCtx->waitStart = os_get_system_timestamp();
  auto waitTime =
      getElapsedCycles(timeoutCtx->waitStart, os_get_system_timestamp());
  if (waitTime < (unsigned) waitTimeoutLogMS * UTimerFreqMS)
    return;

  // E.g. waits for doorbells from the host may take an unbounded amount of
  // time, so don't time out waiting for them.
  if (!doTimeoutCheck)
    return;

  ++timeoutCtx->waitExceededCount;
  // FIXME: Split up print calls since too many arguments crashes
  DBG_PRINT_ERROR("t%d: network doorbell wait timeout exceeded (%d "
                  "time(s)): db 0x%" PRIxPTR " waiting for %" PRIu32
                  " last got %" PRIu32,
                  threadId, timeoutCtx->waitExceededCount, (uintptr_t)db,
                  expectedVal, dbval);
  uintptr_t caller0 = (uintptr_t)__builtin_return_address(0);
  uintptr_t caller1 = (uintptr_t)__builtin_return_address(1);
  DBG_PRINT_ERROR("t%d: call stack 0x%" PRIxPTR " 0x%" PRIxPTR, threadId,
                  (uintptr_t)caller0, (uintptr_t)caller1);
  timeoutCtx->waitStart = os_get_system_timestamp();

  const int waitExceededCountLimit = 10;
  if (timeoutCtx->waitExceededCount == waitExceededCountLimit &&
      ctx->notifyHangPtr) {
    // notifyHangPtr does nothing other than notify QSM. QSM then logs the
    // timeout.
    ctx->notifyHangPtr();
    // Terminate this thread
    DBG_PRINT_FATAL("t%d: Doorbell %d ms timeout limit exceeded. Terminating.",
                    threadId, waitExceededCountLimit * waitTimeoutLogMS);
    // For unknown reason, 1 or more prints before calling errFuncPtr is lost so
    // add dummy prints
    DBG_PRINT_FATAL("t%d:", threadId);
    DBG_PRINT_FATAL("t%d:", threadId);
    DBG_PRINT_FATAL("t%d:", threadId);
    ERR_FATAL(ctx->errFuncPtr, "t%d: Doorbell %d ms timeout limit exceeded.",
              threadId, waitExceededCountLimit * waitTimeoutLogMS, 0);
    assert(false && "ctx->errFuncPtr() function call should not have returned");
  }
}
} // extern "C"
