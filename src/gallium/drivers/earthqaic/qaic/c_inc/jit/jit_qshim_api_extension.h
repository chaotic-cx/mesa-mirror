/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef JIT_QSHIM_API_EXTENSION_H
#define JIT_QSHIM_API_EXTENSION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ECC_ERROR_INVALID = 0,
  ECC_ERROR_CORRECTABLE = 1,
  ECC_ERROR_UNCORRECTABLE = 2,
  ECC_ERROR_MAX = ECC_ERROR_UNCORRECTABLE
} EccErrorType_t;

typedef enum {
  ECC_MEM_VTCM = 0,
} EccMemType_t;

// Qshim API to inject errors in NSP's on-chip memory such as L2TCM or VTCM.
// Initally, the API will only support error injection in VTCM but can be later
// extended to inject errors in L2TCM as well.
// errorType    -    Error type to inject. Correctable type will inject max of 1-bit
//                   error. Uncorrectable type will inject max of 2-bit error. NSP HW
//                   is capable of correcting 1-bit errors.
// memType      -    Memory type where to inject error. For now, Qshim API
//                   supports error injection in VTCM. Support for error injection
//                   in L1 instruction, L1 data cache, L2 cache, L2 tag, JTLB, UDMA
//                   can be added if required.
// errorAddrOffset - Offset into the memory type where to inject error.
// If invalid parameters are provided, JIT_DEV_QSHIM_ERROR_VTCM_ERROR_INJ_INVALID_PARAMETER
// error code is returned.
// If the VTCM error injection fails, JIT_DEV_QSHIM_ERROR_VTCM_ERROR_INJECTION error code
// is returned.
// Otherwise, JIT_DEV_STATUS_SUCCESS is returned.
uint32_t qshimErrorInjection(EccErrorType_t errorType, EccMemType_t memType, uint32_t errorAddrOffset);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JIT_QSHIM_API_EXTENSION_H
