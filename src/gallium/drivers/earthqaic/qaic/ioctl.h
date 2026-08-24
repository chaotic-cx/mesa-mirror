/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef EARTHQAIC_QAIC_IOCTL_H
#define EARTHQAIC_QAIC_IOCTL_H

#include "drm-uapi/qaic_accel.h"

const unsigned long _DRM_IOCTL_QAIC_MANAGE = DRM_IOCTL_QAIC_MANAGE;
const unsigned long _DRM_IOCTL_QAIC_CREATE_BO = DRM_IOCTL_QAIC_CREATE_BO;
const unsigned long _DRM_IOCTL_QAIC_MMAP_BO = DRM_IOCTL_QAIC_MMAP_BO;
const unsigned long _DRM_IOCTL_QAIC_ATTACH_SLICE_BO = DRM_IOCTL_QAIC_ATTACH_SLICE_BO;
const unsigned long _DRM_IOCTL_QAIC_EXECUTE_BO = DRM_IOCTL_QAIC_EXECUTE_BO;
const unsigned long _DRM_IOCTL_QAIC_PARTIAL_EXECUTE_BO = DRM_IOCTL_QAIC_PARTIAL_EXECUTE_BO;
const unsigned long _DRM_IOCTL_QAIC_WAIT_BO = DRM_IOCTL_QAIC_WAIT_BO;
const unsigned long _DRM_IOCTL_QAIC_PERF_STATS_BO = DRM_IOCTL_QAIC_PERF_STATS_BO;
const unsigned long _DRM_IOCTL_QAIC_DETACH_SLICE_BO = DRM_IOCTL_QAIC_DETACH_SLICE_BO;
const unsigned long _DRM_IOCTL_GEM_CLOSE = DRM_IOCTL_GEM_CLOSE;

#endif /* EARTHQAIC_QAIC_IOCTL_H */
