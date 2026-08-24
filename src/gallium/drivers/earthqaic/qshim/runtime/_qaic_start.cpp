// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "AICMetadataExecCtx.h"
#include "ComputeAPI.h"
#include "NSPContext.h"
#include "SerializedProgramDesc.h"

extern "C" {
void _qaic_start(void *ctx, uint8_t virtualThreadId, uint32_t stid);
void activate(void *ctx, uint8_t virtualThreadId, uint32_t stid);
}

static volatile bool _initsDone = false;

void _qaic_start(void *ctx, uint8_t virtualThreadId, uint32_t stid) {
  AICExecContext *qctx = (AICExecContext *)ctx;

  if (virtualThreadId == 0) {
    // library related start up code goes here

    _nspContextInit(qctx);
    qaic::_programDescInit(qctx);

    _initsDone = true;
  } else {
    while (!_initsDone)
      ;
  }
  qaic::logActivate(virtualThreadId);
  _udmaContextInit(virtualThreadId);

  // Jump to the user entry point.
  activate(qctx, virtualThreadId, stid);

  // Flush the log
  NN_LOG(qctx->logFuncPtr, NNC_LOG_MASK_INFO, "\n");
  NN_LOG(qctx->logFuncPtr, NNC_LOG_MASK_INFO, "\n");
  NN_LOG(qctx->logFuncPtr, NNC_LOG_MASK_INFO, "\n");
  NN_LOG(qctx->logFuncPtr, NNC_LOG_MASK_INFO, "\n");

  // Make sure DMAs finish
  _udmaContextCleanup(virtualThreadId);

  qaic::logDeactivate(virtualThreadId);

  // wait for exit DB non-zero before exiting
  volatile uint32_t *exitDB = (uint32_t *)(getNSPContext()->exitDB());
  while (*exitDB == 0)
    ;

  qctx->exitThread();
}
