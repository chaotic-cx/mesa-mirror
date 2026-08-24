// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef LIBDEV_INTERFACE_H
#define LIBDEV_INTERFACE_H

#include <stdint.h>
#include "../NSPContext.h"
#include "../SerializedProgramDesc.h"

extern "C" {

// Expose globals for os* files since they are compiled ahead of time.
extern CoreInfo CoreCtx;

CoreInfo *libdev_getcontext();

} // extern "C"

#endif // LIBDEV_INTERFACE_H
