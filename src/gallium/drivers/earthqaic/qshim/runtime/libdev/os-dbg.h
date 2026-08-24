// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef OS_DBG_H
#define OS_DBG_H

#include "AICMetadataExecCtx.h"
#include "libdev_interface.h"

__attribute__((format(printf, 1, 2))) static inline void checkLog(const char *,
                                                                  ...) {
  // This function does nothing, it is just here to get the compiler to do
  // format string checking.
}

// Main ulog print macro. Should not be used directly.
#define DBG_PRINT(MASK, FORMAT, ...)                                           \
  do {                                                                         \
    nnc_log_fp fp = libdev_getcontext()->logFuncPtr;                           \
    NN_LOG(fp, MASK, FORMAT, ##__VA_ARGS__);                                   \
    checkLog(FORMAT, ##__VA_ARGS__);                                           \
  } while (0)

// Helper ulog print macros at different log levels.
#define DBG_PRINT_FATAL(FORMAT, ...)                                           \
  DBG_PRINT(NNC_LOG_MASK_FATAL, FORMAT, ##__VA_ARGS__)

#define DBG_PRINT_ERROR(FORMAT, ...)                                           \
  DBG_PRINT(NNC_LOG_MASK_ERROR, FORMAT, ##__VA_ARGS__)

#define DBG_PRINT_WARN(FORMAT, ...)                                            \
  DBG_PRINT(NNC_LOG_MASK_WARN, FORMAT, ##__VA_ARGS__)

#define DBG_PRINT_INFO(FORMAT, ...)                                            \
  DBG_PRINT(NNC_LOG_MASK_INFO, FORMAT, ##__VA_ARGS__)

#endif // OS_DBG_H
