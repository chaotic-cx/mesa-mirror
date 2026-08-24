/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "util/format/u_format.h"
#include "util/os_file.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_screen.h"

#include "drm-uapi/drm.h"
#include "earthqaic/earthqaic_device.h"
#include "earthqaic_drm_public.h"

struct pipe_screen *
earthqaic_drm_screen_create(int fd, const struct pipe_screen_config *config)
{
   return u_pipe_screen_lookup_or_create(os_dupfd_cloexec(fd), config, NULL,
                                         earthqaic_screen_create);
}
