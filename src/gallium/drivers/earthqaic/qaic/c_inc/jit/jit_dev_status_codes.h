/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef JIT_DEV_DEV_STATUS_CODES_H
#define JIT_DEV_DEV_STATUS_CODES_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  JIT_DEV_STATUS_SUCCESS = 0,
  // Runtime Linker failed to load symbol.
  JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS = 1,
  // Flat buffer verification failure.
  JIT_DEV_ERROR_INVALID_FLAT_BUFFER = 2,
  // Invalid parameter in cmd buffer
  JIT_DEV_ERROR_INVALID_PARAMETER = 3,
  // DDR Memory allocation failure.
  JIT_DEV_ERROR_MEM_ALLOC_FAILURE = 4,
  // CMDSET verfication failure.
  JIT_DEV_ERROR_CMDSET_VERIFICATION = 5,
  // CMD verification failure.
  JIT_DEV_ERROR_COMMANDS_VERIFICATION = 6,
  // VA allocation failure in MMAP .
  JIT_DEV_ERROR_MMAP = 7,
  // Local Memory(Heap) allocation failure.
  JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE = 8,
  // Doorbell resource allocation failure.
  JIT_DEV_ERROR_UDMA_DB_RES_FAILURE = 9,
  // DDR Memory free failure.
  JIT_DEV_ERROR_BUFFER_FREE_FAILURE = 10,
  // GOTPLT offsets exceededs allocated size.
  JIT_DEV_ERROR_GOTPLT_SIZE_OUT_OF_BOUND = 11,
  // Cache cleanup failed loading operator library.
  JIT_DEV_ERROR_CACHE_CLEANUP = 12,
  // Operator Kernel exceeded max execution time.
  JIT_DEV_ERROR_OPERATOR_KERNEL_TIMEOUT = 13,
  // Operator reported fatal error
  JIT_DEV_ERROR_FATAL = 14,

  // QShim API errors.

  // UDMA transfer size is invalid.
  JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_SIZE = 501,
  // 1DUDMA transfer desc with no attributes.
  JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_NO_DESC_ATTR = 502,
  // 2DUDMA transfer desc with invalid width attribute.
  JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_WIDTH = 503,
  // 2DUDMA transfer desc with invalid height attribute.
  JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_HEIGHT = 504,
  // 2DUDMA transfer desc with invalid src stride attribute.
  JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_SRC_STRIDE = 505,
  // 2DUDMA transfer desc with invalid dest stride attribute.
  JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_DST_STRIDE = 506,
  // Doorbell resource allocation failure for UDMA submit.
  JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_RES_ALLOC = 507,

  // UDMA wait error on DMPOLL.
  JIT_DEV_QSHIM_ERROR_UDMA_WAIT_USER_DMA_ERROR = 515,
  // UDMA wait initiated on invalid handle.
  JIT_DEV_QSHIM_ERROR_UDMA_WAIT_INVALID_HANDLE = 516,
  // UDMA wait initiated on invalid parameter.
  JIT_DEV_QSHIM_ERROR_UDMA_WAIT_INVALID_PARAMETER = 517,

  // Qshim Threads synchronisation failure.
  JIT_DEV_QSHIM_ERROR_SYNC_THREADS = 520,

  // Qshim operator mmap failure.
  JIT_DEV_QSHIM_ERROR_MMAP = 521,

  // Qshim operator mmap passed invalid parameter.
  JIT_DEV_QSHIM_ERROR_MMAP_INVALID_PARAMETER = 522,
  // Qshim api invalid parameters
  JIT_DEV_QSHIM_ERROR_PRNG_INVALID_PARAMETER = 523,
  // Qshim single thread call multiple PRNG submit
  JIT_DEV_QSHIM_ERROR_PRNG_MULTIPLE_SUBMISSION = 524,

  // Operator error codes.

  // Generic unspecified operator error code.
  JIT_DEV_OPERATOR_ERROR_CODE_GENERIC = 1000,
  // Start value to be used by operator implementer to define their own error
  // codes.
  JIT_DEV_OPERATOR_ERROR_CODES_START = 1001,
} JitDevStatusCode_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JIT_DEV_DEV_STATUS_CODES_H
