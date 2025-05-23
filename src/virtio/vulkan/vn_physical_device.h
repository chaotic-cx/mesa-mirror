/*
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: MIT
 *
 * based in part on anv and radv which are:
 * Copyright © 2015 Intel Corporation
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 */

#ifndef VN_PHYSICAL_DEVICE_H
#define VN_PHYSICAL_DEVICE_H

#include "vn_common.h"

#include "util/sparse_array.h"

#include "vn_wsi.h"

struct vn_format_properties_entry {
   atomic_bool valid;
   VkFormatProperties props;
   VkFormatProperties3 props3;
   VkBool32 srpq;
};

struct vn_image_format_properties {
   VkImageFormatProperties2 format;
   VkResult cached_result;

   VkExternalImageFormatProperties ext_image;
   VkHostImageCopyDevicePerformanceQuery host_copy;
   VkImageCompressionPropertiesEXT compression;
   VkSamplerYcbcrConversionImageFormatProperties ycbcr_conversion;
   VkFilterCubicImageViewImageFormatPropertiesEXT filter_cubic;
};

struct vn_image_format_cache_entry {
   struct vn_image_format_properties properties;
   uint8_t key[SHA1_DIGEST_LENGTH];
   struct list_head head;
};

struct vn_image_format_properties_cache {
   struct hash_table *ht;
   struct list_head lru;
   simple_mtx_t mutex;

   struct {
      uint32_t cache_hit_count;
      uint32_t cache_miss_count;
      uint32_t cache_skip_count;
   } debug;
};

struct vn_layered_api_properties {
   VkPhysicalDeviceLayeredApiPropertiesKHR api;
   VkPhysicalDeviceLayeredApiVulkanPropertiesKHR vk;
   VkPhysicalDeviceDriverProperties driver;
   VkPhysicalDeviceIDProperties id;
};

struct vn_physical_device {
   struct vn_physical_device_base base;

   struct vn_instance *instance;

   /* Between the driver and the app, properties.properties.apiVersion is what
    * we advertise and is capped by VN_MAX_API_VERSION and others.
    *
    * Between the driver and the renderer, renderer_version is the device
    * version we can use internally.
    */
   uint32_t renderer_version;

   /* For maintenance7 layered api properties. */
   struct vn_layered_api_properties layered_properties;

   /* Between the driver and the app, base.base.supported_extensions is what
    * we advertise.
    *
    * Between the driver and the renderer, renderer_extensions is what we can
    * use internally (after enabling).
    */
   struct vk_device_extension_table renderer_extensions;
   uint32_t *extension_spec_versions;

   /* passthrough ray tracing support */
   bool ray_tracing;

   /* Venus feedback encounters cacheline overflush issue on Intel JSL, and
    * has to workaround by further aligning up the feedback buffer alignment.
    */
   uint32_t wa_min_fb_align;

   VkDriverId renderer_driver_id;

   /* Static storage so that host copy properties query can be done once. */
   VkImageLayout copy_src_layouts[64];
   VkImageLayout copy_dst_layouts[64];

   VkQueueFamilyProperties2 *queue_family_properties;
   VkQueueFamilyGlobalPriorityProperties *global_priority_properties;
   uint32_t queue_family_count;
   bool sparse_binding_disabled;
   /* Track the queue family index to emulate a second queue. -1 means no
    * emulation is needed. To be noted that the emulation is a workaround for
    * Android 14+ UI framework and it does not handle wait-before-signal.
    */
   int emulate_second_queue;

   VkPhysicalDeviceMemoryProperties memory_properties;

   struct {
      VkExternalMemoryHandleTypeFlagBits renderer_handle_type;
      VkExternalMemoryHandleTypeFlags supported_handle_types;
   } external_memory;

   struct {
      bool fence_exportable;
      bool semaphore_exportable;
      bool semaphore_importable;
   } renderer_sync_fd;

   VkExternalFenceHandleTypeFlags external_fence_handles;
   VkExternalSemaphoreHandleTypeFlags external_binary_semaphore_handles;
   VkExternalSemaphoreHandleTypeFlags external_timeline_semaphore_handles;

   struct wsi_device wsi_device;

   simple_mtx_t format_update_mutex;
   struct util_sparse_array format_properties;

   struct vn_image_format_properties_cache image_format_cache;
};
VK_DEFINE_HANDLE_CASTS(vn_physical_device,
                       base.vk.base,
                       VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

void
vn_physical_device_fini(struct vn_physical_device *physical_dev);

#endif /* VN_PHYSICAL_DEVICE_H */
