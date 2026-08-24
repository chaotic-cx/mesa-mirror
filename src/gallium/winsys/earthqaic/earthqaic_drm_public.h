/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef __EARTHQAIC_DRM_PUBLIC_H__
#define __EARTHQAIC_DRM_PUBLIC_H__

#include <stdbool.h>

struct pipe_screen;
struct pipe_screen_config;

struct pipe_screen *
earthqaic_drm_screen_create(int, const struct pipe_screen_config *);

#endif /* __EARTHQAIC_DRM_PUBLIC_H__ */
