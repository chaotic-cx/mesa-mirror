/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#include "ComputeAPI.h"
#include "JitHwDevLib.h"

extern JitQpcShim_QurtMutexDestroy_t g_qurtMutexDestroy;

extern "C" {
void activate(void *ctx, uint8_t virtualThreadId, uint32_t stid);
}

void activate(void *ctx, uint8_t virtualThreadId, uint32_t stid) {
  // Call the library function
  initializeQpcShim((AICExecContext *)ctx, virtualThreadId, stid);
  while (1) {
    run((AICExecContext *)ctx, virtualThreadId, stid);
    qaic::quitIfExitDBSet(virtualThreadId);
  }
}
