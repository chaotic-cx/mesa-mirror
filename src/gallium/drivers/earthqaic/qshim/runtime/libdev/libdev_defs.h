// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef LIBDEV_DEFS_H
#define LIBDEV_DEFS_H

#include "libdev_assert.h"

using namespace aic;

#include "os-inlines.h"

extern "C" {
void libdev_copyVTCM(int8_t *dst, const int8_t *src, int size,
                     bool toUncached, bool fromUncached, const uint32_t *dbVal,
		     bool dbOnlyCheckedLocally, bool updateDBs,
		     bool doRelease, int threadId, uint32_t DBNum);

void libdev_copy_dma_doorbells(
    int8_t *dst, int32_t dstOffset, const int8_t *src, unsigned size,
    bool isMulticast, bool destBypass, bool srcBypass, int threadId,
    const uint32_t *dbVal, bool dbOnlyCheckedLocally, bool updateDBs,
    bool noPayload, bool doRelease, uint32_t DBNum, uint32_t mcId);

void libdev_multicastVTCM(int8_t *dst, int32_t dstOffset, const int8_t *src,
                          int size, const uint32_t *dbVal,
                          bool dbOnlyCheckedLocally, bool updateDBs,
                          int threadId, uint32_t DBNum, uint32_t mcId);

inline int libdev_l2tcm_size() { return 1 * 1024 * 1024; }
inline int libdev_vtcm_size() { return 8 * 1024 * 1024; }
void *libdev_native_udma_handle(int threadId);
}

#endif // LIBDEV_DEFS_H
