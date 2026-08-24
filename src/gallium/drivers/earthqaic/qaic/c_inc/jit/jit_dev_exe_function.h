/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef JIT_DEV_EXE_FUNCTION_H
#define JIT_DEV_EXE_FUNCTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AicJitPointerArray {
  uint32_t numPointers;
  void *pointers[];
} AicJitPointerArray;

// Core ID passed is same as position of bit set in QAicComputeUnitBitMap
// when creating command buffer for this NSP.
// Thread ID is the actual thread on which this operator
// is called. This value is from one of the bit positions set when
// adding dispatch command to command buffer.
// numCores is the number of NSPs on which the cmdBuf batch
// is executed.
// numThreads is the number of worker threads executing the
// dispatch command in a core. The value is most likely the same for all cores.

typedef struct AicJitEntryPointConfig {
  uint32_t coreID;
  uint32_t numCores;
  uint32_t threadID;
  uint32_t numThreads;
} AicJitEntryPointConfig;

// Func should return 0 for success and positive non-zero value for any error.
// Error code should start from JIT_DEV_OPERATOR_ERROR_CODES_START.
typedef uint32_t (*AicJitDevExeFunction)(
    const AicJitEntryPointConfig *entryConfig,
    const AicJitPointerArray *pointerArray);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JIT_DEV_EXE_FUNCTION_H
