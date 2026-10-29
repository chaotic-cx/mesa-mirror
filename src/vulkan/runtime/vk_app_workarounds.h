/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef VK_APP_WORKAROUNDS_H
#define VK_APP_WORKAROUNDS_H

#include "vk_device.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Installs the common application workarounds layer
 *
 * The workarounds are layered on top of device->dispatch_table and call down
 * through device->app_workarounds_dispatch_table. This must be called exactly
 * once by the driver, after device->dispatch_table is final.
 *
 * :param device:       |in|  The device
 * :param app_layer:    |in|  The application layer to enable, from the
 *                            vk_app_layer driconf option
 */
VkResult MUST_CHECK
vk_device_init_app_workarounds(struct vk_device *device, const char *app_layer);

#ifdef __cplusplus
}
#endif

#endif /* VK_APP_WORKAROUNDS_H */
