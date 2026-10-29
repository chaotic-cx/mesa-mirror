/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "vk_app_workarounds.h"

#include "vk_alloc.h"
#include "vk_log.h"

#include <string.h>

/* This layer implements application workarounds that are common to all drivers. */

VkResult
vk_device_init_app_workarounds(struct vk_device *device, const char *app_layer)
{
   /* Must be called only once, after the driver has set up device->dispatch_table. */
   assert(!device->app_workarounds_dispatch_table);

   if (!app_layer || !app_layer[0])
      return VK_SUCCESS;

   device->app_workarounds_dispatch_table =
      vk_alloc(&device->alloc, sizeof(*device->app_workarounds_dispatch_table), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device->app_workarounds_dispatch_table)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* The workarounds call down to the driver through this copy. */
   *device->app_workarounds_dispatch_table = device->dispatch_table;

   /* TODO: Add workarounds. */

   return VK_SUCCESS;
}
