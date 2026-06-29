/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "gamma_device.h"
#include "vulkan/vulkan_core.h"
#include "gamma_util.h"

int
gamma_device_init(gamma_device *device, gamma_pdev_supported_cb pdev_supported, uint32_t instance_extension_count,
                  const char *const *instance_extensions, uint32_t device_extension_count,
                  const char *const *device_extensions, VkPhysicalDeviceFeatures *enabled_features,
                  const void *device_pnext)
{
   uint32_t api_version = VK_API_VERSION_1_0;
   vkEnumerateInstanceVersion(&api_version);

   if (api_version < VK_API_VERSION_1_4) {
      fprintf(stderr, "gamma: VK_API_VERSION_1_4 is required but not supported.\n");
      return 1;
   }

   uint32_t layer_property_count = 0;
   vkEnumerateInstanceLayerProperties(&layer_property_count, nullptr);
   std::vector<VkLayerProperties> layer_properties(layer_property_count);
   vkEnumerateInstanceLayerProperties(&layer_property_count, layer_properties.data());

   bool has_validation = false;
   for (const VkLayerProperties &properties : layer_properties) {
      if (!strcmp("VK_LAYER_KHRONOS_validation", properties.layerName)) {
         has_validation = true;
         break;
      }
   }

   if (has_validation)
      fprintf(stderr, "gamma: Enabling VK_LAYER_KHRONOS_validation.\n");

   VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "mesa.gamma",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "mesa.gamma",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = api_version,
   };

   const char *layers[] = {"VK_LAYER_KHRONOS_validation"};

   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = (uint32_t)(has_validation ? ARRAY_SIZE(layers) : 0),
      .ppEnabledLayerNames = layers,
      .enabledExtensionCount = instance_extension_count,
      .ppEnabledExtensionNames = instance_extensions,
   };

   gamma_check_vk_result(vkCreateInstance(&instance_info, nullptr, &device->instance));

   uint32_t physical_device_count = 0;
   gamma_check_vk_result(vkEnumeratePhysicalDevices(device->instance, &physical_device_count, nullptr));
   std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
   gamma_check_vk_result(vkEnumeratePhysicalDevices(device->instance, &physical_device_count, physical_devices.data()));

   for (VkPhysicalDevice pdev : physical_devices) {
      uint32_t extension_property_count = 0;
      vkEnumerateDeviceExtensionProperties(pdev, nullptr, &extension_property_count, nullptr);
      std::vector<VkExtensionProperties> extension_properties(extension_property_count);
      vkEnumerateDeviceExtensionProperties(pdev, nullptr, &extension_property_count, extension_properties.data());

      bool all_extensions_present = true;
      for (uint32_t i = 0; i < device_extension_count; ++i) {
         bool found = false;
         for (const VkExtensionProperties &properties : extension_properties) {
            if (!strcmp(device_extensions[i], properties.extensionName)) {
               found = true;
               break;
            }
         }

         if (!found) {
            all_extensions_present = false;
            break;
         }
      }

      if (!all_extensions_present)
         continue;
      if (pdev_supported && !pdev_supported(device->instance, pdev))
         continue;

      uint32_t queue_family_property_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(pdev, &queue_family_property_count, nullptr);
      std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
      vkGetPhysicalDeviceQueueFamilyProperties(pdev, &queue_family_property_count, queue_family_properties.data());

      for (uint32_t i = 0; i < queue_family_property_count; i++) {
         if (queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            device->queue_family_index = i;
            break;
         }
      }

      device->pdev = pdev;
      break;
   }

   if (!device->pdev) {
      fprintf(stderr, "gamma: Failed to find suitable VkPhysicalDevice\n");
      vkDestroyInstance(device->instance, nullptr);
      return 1;
   }

   float priority = 1.0;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = device->queue_family_index,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };

   VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = device_pnext,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = device_extension_count,
      .ppEnabledExtensionNames = device_extensions,
      .pEnabledFeatures = enabled_features,
   };
   gamma_check_vk_result(vkCreateDevice(device->pdev, &device_info, nullptr, &device->device));

   vkGetDeviceQueue(device->device, device->queue_family_index, 0, &device->queue);

   VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = device->queue_family_index,
   };
   gamma_check_vk_result(
      vkCreateCommandPool(device->device, &command_pool_info, nullptr, &device->staging_command_pool));

   return 0;
}

void *
gamma_device::upload_memory(std::shared_ptr<gamma_backed_buffer> dst)
{
   std::shared_ptr<gamma_backed_buffer> staging_buffer = gamma_create_backed_buffer(
      this, dst->size, gamma_memory_type_host_visible, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
   pending_staging_buffers.push_back({staging_buffer, dst});
   return staging_buffer->map;
}

void
gamma_device::flush_upload_memory()
{
   if (pending_staging_buffers.empty())
      return;

   assert(!staging_command_buffer);

   VkCommandBufferAllocateInfo command_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = staging_command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   gamma_check_vk_result(vkAllocateCommandBuffers(device, &command_buffer_info, &staging_command_buffer));

   VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
   };
   gamma_check_vk_result(vkBeginCommandBuffer(staging_command_buffer, &begin_info));

   for (const auto &staging_buffer : pending_staging_buffers) {
      VkBufferCopy copy = {
         .size = staging_buffer.first->size,
      };
      vkCmdCopyBuffer(staging_command_buffer, staging_buffer.first->buffer, staging_buffer.second->buffer, 1, &copy);
   }

   gamma_check_vk_result(vkEndCommandBuffer(staging_command_buffer));

   VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &staging_command_buffer,
   };
   gamma_check_vk_result(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
}

void
gamma_device::finish_upload_memory()
{
   if (!staging_command_buffer)
      return;

   gamma_check_vk_result(vkQueueWaitIdle(queue));

   vkFreeCommandBuffers(device, staging_command_pool, 1, &staging_command_buffer);
   staging_command_buffer = VK_NULL_HANDLE;

   pending_staging_buffers.clear();
}
