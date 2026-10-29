/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "vk_app_workarounds.h"

#include "vk_alloc.h"
#include "vk_log.h"

#include "util/log.h"

#include <string.h>

/* This layer implements application workarounds that are common to all drivers. */

/* Metro Exodus (Linux native) */
static VKAPI_ATTR VkResult VKAPI_CALL
metro_exodus_GetSemaphoreCounterValue(VkDevice _device, VkSemaphore _semaphore, uint64_t *pValue)
{
   VK_FROM_HANDLE(vk_device, device, _device);

   /* See https://gitlab.freedesktop.org/mesa/mesa/-/issues/5119. */
   if (_semaphore == VK_NULL_HANDLE) {
      mesa_logw("Ignoring vkGetSemaphoreCounterValue() with NULL semaphore (game bug)!");
      return VK_SUCCESS;
   }

   return device->app_workarounds_dispatch_table->GetSemaphoreCounterValue(_device, _semaphore, pValue);
}

VkResult
vk_device_init_app_workarounds(struct vk_device *device, const char *app_layer)
{
   struct vk_device_dispatch_table *table = &device->dispatch_table;

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

#define SET_ENTRYPOINT(app_layer, entrypoint) table->entrypoint = app_layer##_##entrypoint;

   if (!strcmp(app_layer, "metroexodus")) {
      SET_ENTRYPOINT(metro_exodus, GetSemaphoreCounterValue);
   } else {
      mesa_logw("Unknown vk_app_layer '%s', ignoring", app_layer);
      vk_free(&device->alloc, device->app_workarounds_dispatch_table);
      device->app_workarounds_dispatch_table = NULL;
   }

#undef SET_ENTRYPOINT

   return VK_SUCCESS;
}
