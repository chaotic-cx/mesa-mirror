// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "Exit.h"
#include "NSPContext.h"

namespace qaic {
_Noreturn void exit(int status) {
  CoreInfo *ctx = getNSPContext();
  NN_LOG(ctx->logFuncPtr, NNC_LOG_MASK_WARN, "Exiting thread with status %d",
         status);
  ctx->exitThread();
  __builtin_unreachable();
}
} // namespace qaic
