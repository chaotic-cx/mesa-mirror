/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef JIT_DEV_ATTRIBUTES_H
#define JIT_DEV_ATTRIBUTES_H

#include <cstdint>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  DEVICE_FAMILY_AIC080 = 0,
  DEVICE_FAMILY_AIC100 = 1,
  DEVICE_FAMILY_AIC200 = 2,
  DEVICE_FAMILY_UNKNOWN = UINT32_MAX,
} DeviceFamilyType_t;

typedef enum {
  DEV_NSP_ARCH_V68 = 0,
  DEV_NSP_ARCH_V81 = 1,
  DEV_NSP_ARCH_UNKNOWN = UINT32_MAX,
} NspArchVersion_t;

// Function to get the NSP version based on chip family type
inline NspArchVersion_t get_nsp_version(DeviceFamilyType_t chipFamily) {
  switch (chipFamily) {
  case DEVICE_FAMILY_AIC080:
  case DEVICE_FAMILY_AIC100:
    return DEV_NSP_ARCH_V68;
  case DEVICE_FAMILY_AIC200:
    return DEV_NSP_ARCH_V81;
  default:
    return DEV_NSP_ARCH_UNKNOWN;
  }
}

// Function to get the maximum theoretical DDR bandwidth based on chip family
// type
inline uint64_t get_max_ddr_bandwidth(DeviceFamilyType_t chipFamily) {
  switch (chipFamily) {
  case DEVICE_FAMILY_AIC080:
  case DEVICE_FAMILY_AIC100:
    return 136ULL * 1024 * 1024 * 1024;
  default:
    assert(false);
    return 0;
  }
}

// Function to get the UDMA bandwidth based on the NSP version
inline uint64_t get_max_udma_bandwidth(DeviceFamilyType_t chipFamily) {
  switch (chipFamily) {
  case DEVICE_FAMILY_AIC080:
  case DEVICE_FAMILY_AIC100:
    return 32ULL * 1024 * 1024 * 1024;
  default:
    assert(false);
    return 0;
  }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JIT_DEV_ATTRIBUTES_H
