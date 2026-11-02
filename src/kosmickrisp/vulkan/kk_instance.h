/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_INSTANCE_H
#define KK_INSTANCE_H 1

#include "kk_private.h"

#include "util/xmlconfig.h"
#include "kk_drirc.h"
#include "vk_instance.h"

struct kk_instance {
   struct vk_instance vk;

   struct kk_drirc drirc;

   uint8_t driver_build_sha[VK_UUID_SIZE];
};

VK_DEFINE_HANDLE_CASTS(kk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)

#endif // KK_INSTANCE_H
