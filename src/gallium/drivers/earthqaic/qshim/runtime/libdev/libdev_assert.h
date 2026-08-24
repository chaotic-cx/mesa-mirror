// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef LIBDEV_ASSERT_H
#define LIBDEV_ASSERT_H

#include "libdev_interface.h"
#include "os-dbg.h"

#ifndef NDEBUG
#define assert(X)                                                              \
  do {                                                                         \
    if (!(X)) {                                                                \
      DBG_PRINT_FATAL("assert:%s:%d caller pc=0x%p: %s", __FILE__, __LINE__,   \
                      __builtin_return_address(0), #X);                        \
      ERR_FATAL(libdev_getcontext()->errFuncPtr, #X, 0, 0, 0);                 \
    }                                                                          \
  } while (0)
#else
#define assert(X) ((void)0)
#endif

#endif
