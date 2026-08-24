/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#include <xf86drm.h>

/* Mesa headers */
#include "drm-uapi/drm.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "renderonly/renderonly.h"
#include "util/ralloc.h"
#include "util/u_atomic.h"
#include "util/u_dynarray.h"
#include "util/u_inlines.h"
#include "util/u_screen.h"
#include "util/u_surface.h"
#include "util/u_transfer.h"

/* Earthqaic headers */
#include "earthqaic_device.h"
