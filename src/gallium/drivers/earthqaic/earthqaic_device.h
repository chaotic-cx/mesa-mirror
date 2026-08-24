/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef __EARTHQAIC_DEVICE_H__
#define __EARTHQAIC_DEVICE_H__

struct pipe_screen *
earthqaic_screen_create(int, const struct pipe_screen_config *,
                        struct renderonly *);

#endif /* __EARTHQAIC_DEVICE_H__ */
