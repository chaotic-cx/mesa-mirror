
/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan/vulkan_core.h"

#include <utility>
#include <vector>

#include "gamma_util.h"

struct gamma_device {
   VkInstance instance;
   VkPhysicalDevice pdev;
   VkDevice device;
   VkQueue queue;
   uint32_t queue_family_index;
   VkCommandBuffer command_buffer;

   VkCommandPool staging_command_pool;
   VkCommandBuffer staging_command_buffer = VK_NULL_HANDLE;
   std::vector<std::pair<std::shared_ptr<gamma_backed_buffer>, std::shared_ptr<gamma_backed_buffer>>>
      pending_staging_buffers;

   void *upload_memory(std::shared_ptr<gamma_backed_buffer> dst);

   void flush_upload_memory();
   void finish_upload_memory();
};

using gamma_pdev_supported_cb = bool (*)(VkInstance instance, VkPhysicalDevice pdev);

int gamma_device_init(gamma_device *device, gamma_pdev_supported_cb pdev_supported, uint32_t instance_extension_count,
                      const char *const *instance_extensions, uint32_t device_extension_count,
                      const char *const *device_extensions, VkPhysicalDeviceFeatures *enabled_features,
                      const void *device_pnext);
