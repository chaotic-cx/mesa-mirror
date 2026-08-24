// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef _QAIC_LOG_H_
#define _QAIC_LOG_H_

#define MASK_NONE 0x0
#define MASK_FATAL 0x1
#define MASK_ERROR 0x2
#define MASK_WARN 0x4
#define MASK_INFO 0x8
#define MASK_DEBUG 0x10
#define MASK_INVALID 0x7f

#define NNC_LOG_MASK_ALL (2 * MASK_DEBUG - 1)

/* Check if build is release version */
#ifndef NNC_LOG_COMPILE_MASK
#ifdef BLD_VAR_REL
#define NNC_LOG_COMPILE_MASK (MASK_DEBUG - 1)
#else
#define NNC_LOG_COMPILE_MASK NNC_LOG_MASK_ALL
#endif
#endif

#if ((NNC_LOG_COMPILE_MASK & MASK_FATAL) == MASK_FATAL)
#define LOG_FATAL(handle, msg, ...)                                            \
  NN_LOG(handle, NNC_LOG_MASK_FATAL, "Fatal: " msg, ##__VA_ARGS__)
#else
#define LOG_FATAL(handle, msg, ...)
#endif

#if ((NNC_LOG_COMPILE_MASK & MASK_ERROR) == MASK_ERROR)
#define LOG_ERROR(handle, msg, ...)                                            \
  NN_LOG(handle, NNC_LOG_MASK_ERROR, "Error: " msg, ##__VA_ARGS__)
#else
#define LOG_ERROR(handle, msg, ...)
#endif

#if ((NNC_LOG_COMPILE_MASK & MASK_WARN) == MASK_WARN)
#define LOG_WARN(handle, msg, ...)                                             \
  NN_LOG(handle, NNC_LOG_MASK_WARN, "Warning: " msg, ##__VA_ARGS__)
#else
#define LOG_WARN(handle, msg, ...)
#endif

#if ((NNC_LOG_COMPILE_MASK & MASK_INFO) == MASK_INFO)
#define LOG_INFO(handle, msg, ...)                                             \
  NN_LOG(handle, NNC_LOG_MASK_INFO, "Info: " msg, ##__VA_ARGS__)
#else
#define LOG_INFO(handle, msg, ...)
#endif

#if ((NNC_LOG_COMPILE_MASK & MASK_DEBUG) == MASK_DEBUG)
#define LOG_DEBUG(handle, msg, ...)                                            \
  NN_LOG(handle, NNC_LOG_MASK_DEBUG, "Debug: " msg, ##__VA_ARGS__)
#else
#define LOG_DEBUG(handle, msg, ...)
#endif

#endif /* _QAIC_LOG_H_ */
