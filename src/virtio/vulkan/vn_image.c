/*
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: MIT
 *
 * based in part on anv and radv which are:
 * Copyright © 2015 Intel Corporation
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 */

#include "vn_image.h"

#include "venus-protocol/vn_protocol_driver_image.h"
#include "venus-protocol/vn_protocol_driver_image_view.h"
#include "venus-protocol/vn_protocol_driver_sampler.h"
#include "venus-protocol/vn_protocol_driver_sampler_ycbcr_conversion.h"
#include "vk_format.h"

#include "vn_android.h"
#include "vn_device.h"
#include "vn_device_memory.h"
#include "vn_physical_device.h"
#include "vn_wsi.h"

#define IMAGE_REQS_CACHE_MAX_ENTRIES 500

/* image commands */

static inline uint32_t
vn_image_get_plane_count(const VkImageCreateInfo *create_info)
{
   if (!(create_info->flags & VK_IMAGE_CREATE_DISJOINT_BIT))
      return 1;

   /* TODO VkDrmFormatModifierPropertiesEXT::drmFormatModifierPlaneCount */
   assert(create_info->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT);
   return vk_format_get_plane_count(create_info->format);
}

static inline uint32_t
vn_image_get_plane(const VkImageAspectFlagBits plane_aspect)
{
   switch (plane_aspect) {
   case VK_IMAGE_ASPECT_PLANE_1_BIT:
      return 1;
   case VK_IMAGE_ASPECT_PLANE_2_BIT:
      return 2;
   default:
      return 0;
   }
}

static void
vn_image_fill_reqs(const struct vn_image_memory_requirements *req,
                   VkMemoryRequirements2 *out_reqs)
{
   union {
      VkBaseOutStructure *pnext;
      VkMemoryRequirements2 *two;
      VkMemoryDedicatedRequirements *dedicated;
   } u = { .two = out_reqs };

   while (u.pnext) {
      switch (u.pnext->sType) {
      case VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2:
         u.two->memoryRequirements = req->memory.memoryRequirements;
         break;
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS:
         u.dedicated->prefersDedicatedAllocation =
            req->dedicated.prefersDedicatedAllocation;
         u.dedicated->requiresDedicatedAllocation =
            req->dedicated.requiresDedicatedAllocation;
         break;
      default:
         break;
      }
      u.pnext = u.pnext->pNext;
   }
}

static void
vn_image_cache_debug_dump(struct vn_image_reqs_cache *cache)
{
   vn_log(NULL, "dumping image reqs cache statistics");
   vn_log(NULL, "  hit %u\n", cache->debug.cache_hit_count);
   vn_log(NULL, "  miss %u\n", cache->debug.cache_miss_count);
   vn_log(NULL, "  skip %u\n", cache->debug.cache_skip_count);
}

static bool
vn_image_get_image_reqs_key(struct vn_device *dev,
                            const VkImageCreateInfo *create_info,
                            uint8_t *key)
{
   struct mesa_sha1 sha1_ctx;

   if (!dev->image_reqs_cache.ht)
      return false;

   /* Strip the alias bit as the memory requirements are identical. */
   VkImageCreateInfo local_info;
   if (create_info->flags & VK_IMAGE_CREATE_ALIAS_BIT) {
      local_info = *create_info;
      local_info.flags &= ~VK_IMAGE_CREATE_ALIAS_BIT;
      create_info = &local_info;
   }

   _mesa_sha1_init(&sha1_ctx);

   /* Hash relevant fields in the pNext chain */
   vk_foreach_struct_const(src, create_info->pNext) {
      switch (src->sType) {
      case VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO: {
         struct VkExternalMemoryImageCreateInfo *ext_mem =
            (struct VkExternalMemoryImageCreateInfo *)src;
         _mesa_sha1_update(&sha1_ctx, &ext_mem->handleTypes,
                           sizeof(VkExternalMemoryHandleTypeFlags));
         break;
      }
      case VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO: {
         struct VkImageFormatListCreateInfo *format_list =
            (struct VkImageFormatListCreateInfo *)src;
         _mesa_sha1_update(&sha1_ctx, format_list->pViewFormats,
                           sizeof(VkFormat) * format_list->viewFormatCount);
         break;
      }
      case VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT: {
         struct VkImageDrmFormatModifierListCreateInfoEXT *format_mod_list =
            (struct VkImageDrmFormatModifierListCreateInfoEXT *)src;
         _mesa_sha1_update(
            &sha1_ctx, format_mod_list->pDrmFormatModifiers,
            sizeof(uint64_t) * format_mod_list->drmFormatModifierCount);
         break;
      }
      case VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT: {
         struct VkImageDrmFormatModifierExplicitCreateInfoEXT
            *format_mod_explicit =
               (struct VkImageDrmFormatModifierExplicitCreateInfoEXT *)src;
         _mesa_sha1_update(&sha1_ctx, &format_mod_explicit->drmFormatModifier,
                           sizeof(uint64_t));
         _mesa_sha1_update(
            &sha1_ctx, format_mod_explicit->pPlaneLayouts,
            sizeof(VkSubresourceLayout) *
               format_mod_explicit->drmFormatModifierPlaneCount);
         break;
      }
      case VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO: {
         struct VkImageStencilUsageCreateInfo *stencil_usage =
            (struct VkImageStencilUsageCreateInfo *)src;
         _mesa_sha1_update(&sha1_ctx, &stencil_usage->stencilUsage,
                           sizeof(VkImageUsageFlags));
         break;
      }
      default:
         /* Skip cache for unsupported pNext */
         dev->image_reqs_cache.debug.cache_skip_count++;
         return false;
      }
   }

   /* Hash contingous block of VkImageCreateInfo starting with
    * VkImageCreateInfo->flags and ending with VkImageCreateInfo->sharingMode
    *
    * There's no padding in involved in this hash block so no concern for C
    * enum sizes or alignment.
    */
   static const size_t create_image_hash_block_size =
      offsetof(VkImageCreateInfo, queueFamilyIndexCount) -
      offsetof(VkImageCreateInfo, flags);

   _mesa_sha1_update(&sha1_ctx, &create_info->flags,
                     create_image_hash_block_size);

   /* Follow pointer and hash pQueueFamilyIndices separately.
    * pQueueFamilyIndices is ignored if sharingMode is not
    * VK_SHARING_MODE_CONCURRENT
    */
   if (create_info->sharingMode == VK_SHARING_MODE_CONCURRENT) {
      _mesa_sha1_update(
         &sha1_ctx, create_info->pQueueFamilyIndices,
         sizeof(uint32_t) * create_info->queueFamilyIndexCount);
   }

   _mesa_sha1_update(&sha1_ctx, &create_info->initialLayout,
                     sizeof(create_info->initialLayout));
   _mesa_sha1_final(&sha1_ctx, key);

   return true;
}

void
vn_image_reqs_cache_init(struct vn_device *dev)
{
   struct vn_image_reqs_cache *cache = &dev->image_reqs_cache;

   if (VN_PERF(NO_ASYNC_IMAGE_CREATE))
      return;

   cache->ht = _mesa_hash_table_create(NULL, vn_cache_key_hash_function,
                                       vn_cache_key_equal_function);
   if (!cache->ht)
      return;

   simple_mtx_init(&cache->mutex, mtx_plain);
   list_inithead(&dev->image_reqs_cache.lru);
}

void
vn_image_reqs_cache_fini(struct vn_device *dev)
{
   const VkAllocationCallbacks *alloc = &dev->base.vk.alloc;
   struct vn_image_reqs_cache *cache = &dev->image_reqs_cache;

   if (!cache->ht)
      return;

   hash_table_foreach(cache->ht, hash_entry) {
      struct vn_image_reqs_cache_entry *cache_entry = hash_entry->data;
      list_del(&cache_entry->head);
      vk_free(alloc, cache_entry);
   }
   assert(list_is_empty(&dev->image_reqs_cache.lru));

   _mesa_hash_table_destroy(cache->ht, NULL);

   simple_mtx_destroy(&cache->mutex);

   if (VN_DEBUG(CACHE))
      vn_image_cache_debug_dump(cache);
}

static bool
vn_image_init_reqs_from_cache(struct vn_device *dev,
                              struct vn_image *img,
                              uint8_t *key)
{
   struct vn_image_reqs_cache *cache = &dev->image_reqs_cache;

   assert(cache->ht);

   simple_mtx_lock(&cache->mutex);
   struct hash_entry *hash_entry = _mesa_hash_table_search(cache->ht, key);
   if (hash_entry) {
      struct vn_image_reqs_cache_entry *cache_entry = hash_entry->data;
      for (uint32_t i = 0; i < cache_entry->plane_count; i++)
         img->requirements[i] = cache_entry->requirements[i];
      list_move_to(&cache_entry->head, &dev->image_reqs_cache.lru);
      p_atomic_inc(&cache->debug.cache_hit_count);
   } else {
      p_atomic_inc(&cache->debug.cache_miss_count);
   }
   simple_mtx_unlock(&cache->mutex);

   return !!hash_entry;
}

static struct vn_image_memory_requirements *
vn_image_get_reqs_from_cache(struct vn_device *dev,
                             uint8_t *key,
                             uint32_t plane)
{
   struct vn_image_memory_requirements *requirements = NULL;
   struct vn_image_reqs_cache *cache = &dev->image_reqs_cache;

   assert(cache->ht);

   simple_mtx_lock(&cache->mutex);
   struct hash_entry *hash_entry = _mesa_hash_table_search(cache->ht, key);
   if (hash_entry) {
      struct vn_image_reqs_cache_entry *cache_entry = hash_entry->data;
      requirements = &cache_entry->requirements[plane];
      list_move_to(&cache_entry->head, &dev->image_reqs_cache.lru);
      p_atomic_inc(&cache->debug.cache_hit_count);
   } else {
      p_atomic_inc(&cache->debug.cache_miss_count);
   }
   simple_mtx_unlock(&cache->mutex);

   return requirements;
}

static void
vn_image_store_reqs_in_cache(struct vn_device *dev,
                             uint8_t *key,
                             uint32_t plane_count,
                             struct vn_image_memory_requirements *requirements)
{
   const VkAllocationCallbacks *alloc = &dev->base.vk.alloc;
   struct vn_image_reqs_cache *cache = &dev->image_reqs_cache;
   struct vn_image_reqs_cache_entry *cache_entry;

   assert(cache->ht);

   simple_mtx_lock(&cache->mutex);

   /* Check if entry was added before lock */
   if (_mesa_hash_table_search(cache->ht, key)) {
      simple_mtx_unlock(&cache->mutex);
      return;
   }

   if (_mesa_hash_table_num_entries(cache->ht) ==
       IMAGE_REQS_CACHE_MAX_ENTRIES) {
      /* Evict/use the last entry in the lru list for this new entry */
      cache_entry =
         list_last_entry(&cache->lru, struct vn_image_reqs_cache_entry, head);

      _mesa_hash_table_remove_key(cache->ht, cache_entry->key);
      list_del(&cache_entry->head);
   } else {
      cache_entry = vk_zalloc(alloc, sizeof(*cache_entry), VN_DEFAULT_ALIGN,
                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!cache_entry) {
         simple_mtx_unlock(&cache->mutex);
         return;
      }
   }

   for (uint32_t i = 0; i < plane_count; i++)
      cache_entry->requirements[i] = requirements[i];

   memcpy(cache_entry->key, key, SHA1_DIGEST_LENGTH);
   cache_entry->plane_count = plane_count;

   _mesa_hash_table_insert(dev->image_reqs_cache.ht, cache_entry->key,
                           cache_entry);
   list_add(&cache_entry->head, &cache->lru);

   simple_mtx_unlock(&cache->mutex);
}

static void
vn_image_init_memory_requirements(struct vn_image *img,
                                  struct vn_device *dev,
                                  uint32_t plane_count)
{
   assert(plane_count <= ARRAY_SIZE(img->requirements));

   for (uint32_t i = 0; i < plane_count; i++) {
      img->requirements[i].memory.sType =
         VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
      img->requirements[i].memory.pNext = &img->requirements[i].dedicated;
      img->requirements[i].dedicated.sType =
         VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
      img->requirements[i].dedicated.pNext = NULL;
   }

   VkDevice dev_handle = vn_device_to_handle(dev);
   VkImage img_handle = vn_image_to_handle(img);
   if (plane_count == 1) {
      vn_call_vkGetImageMemoryRequirements2(
         dev->primary_ring, dev_handle,
         &(VkImageMemoryRequirementsInfo2){
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
            .image = img_handle,
         },
         &img->requirements[0].memory);

      /* AHB backed image requires dedicated allocation */
      if (img->deferred_info) {
         img->requirements[0].dedicated.prefersDedicatedAllocation = VK_TRUE;
         img->requirements[0].dedicated.requiresDedicatedAllocation = VK_TRUE;
      }
   } else {
      for (uint32_t i = 0; i < plane_count; i++) {
         vn_call_vkGetImageMemoryRequirements2(
            dev->primary_ring, dev_handle,
            &(VkImageMemoryRequirementsInfo2){
               .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
               .pNext =
                  &(VkImagePlaneMemoryRequirementsInfo){
                     .sType =
                        VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
                     .planeAspect = VK_IMAGE_ASPECT_PLANE_0_BIT << i,
                  },
               .image = img_handle,
            },
            &img->requirements[i].memory);
      }
   }
}

static VkResult
vn_image_deferred_info_init(struct vn_image *img,
                            const VkImageCreateInfo *create_info,
                            const VkAllocationCallbacks *alloc)
{
   struct vn_image_create_deferred_info *info = NULL;
   VkBaseOutStructure *dst = NULL;

   info = vk_zalloc(alloc, sizeof(*info), VN_DEFAULT_ALIGN,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!info)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   info->create = *create_info;
   dst = (void *)&info->create;

   vk_foreach_struct_const(src, create_info->pNext) {
      void *pnext = NULL;
      switch (src->sType) {
      case VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO: {
         /* 12.3. Images
          *
          * If viewFormatCount is zero, pViewFormats is ignored and the image
          * is created as if the VkImageFormatListCreateInfo structure were
          * not included in the pNext chain of VkImageCreateInfo.
          */
         if (!((const VkImageFormatListCreateInfo *)src)->viewFormatCount)
            break;

         memcpy(&info->list, src, sizeof(info->list));
         pnext = &info->list;

         /* need a deep copy for view formats array */
         const size_t size = sizeof(VkFormat) * info->list.viewFormatCount;
         VkFormat *view_formats = vk_zalloc(
            alloc, size, VN_DEFAULT_ALIGN, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
         if (!view_formats) {
            vk_free(alloc, info);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
         }

         memcpy(view_formats,
                ((const VkImageFormatListCreateInfo *)src)->pViewFormats,
                size);
         info->list.pViewFormats = view_formats;
         break;
      }
      case VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO:
         memcpy(&info->stencil, src, sizeof(info->stencil));
         pnext = &info->stencil;
         break;
      case VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID: {
         const uint32_t external_format =
            (uint32_t)((const VkExternalFormatANDROID *)src)->externalFormat;
         if (external_format != 0)
            info->create.format = external_format;
         break;
      }
      default:
         break;
      }

      if (pnext) {
         dst->pNext = pnext;
         dst = pnext;
      }
   }
   dst->pNext = NULL;

   img->deferred_info = info;

   return VK_SUCCESS;
}

static void
vn_image_deferred_info_fini(struct vn_image *img,
                            const VkAllocationCallbacks *alloc)
{
   if (!img->deferred_info)
      return;

   if (img->deferred_info->list.pViewFormats)
      vk_free(alloc, (void *)img->deferred_info->list.pViewFormats);

   vk_free(alloc, img->deferred_info);
}

static VkResult
vn_image_init(struct vn_device *dev,
              const VkImageCreateInfo *create_info,
              struct vn_image *img)
{
   VkDevice device = vn_device_to_handle(dev);
   VkImage image = vn_image_to_handle(img);
   VkResult result = VK_SUCCESS;

   /* Check if mem reqs in cache. If found, make async call */
   uint8_t key[SHA1_DIGEST_LENGTH] = { 0 };
   const bool cacheable = vn_image_get_image_reqs_key(dev, create_info, key);

   if (cacheable && vn_image_init_reqs_from_cache(dev, img, key)) {
      vn_async_vkCreateImage(dev->primary_ring, device, create_info, NULL,
                             &image);
      return VK_SUCCESS;
   }

   result = vn_call_vkCreateImage(dev->primary_ring, device, create_info,
                                  NULL, &image);
   if (result != VK_SUCCESS)
      return result;

   const uint32_t plane_count = vn_image_get_plane_count(create_info);
   vn_image_init_memory_requirements(img, dev, plane_count);

   if (cacheable)
      vn_image_store_reqs_in_cache(dev, key, plane_count, img->requirements);

   return VK_SUCCESS;
}

VkResult
vn_image_create(struct vn_device *dev,
                const VkImageCreateInfo *create_info,
                const VkAllocationCallbacks *alloc,
                struct vn_image **out_img)
{
   struct vn_image *img =
      vk_image_create(&dev->base.vk, create_info, alloc, sizeof(*img));
   if (!img)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   vn_object_set_id(img, vn_get_next_obj_id(), VK_OBJECT_TYPE_IMAGE);

   VkResult result = vn_image_init(dev, create_info, img);
   if (result != VK_SUCCESS) {
      vk_image_destroy(&dev->base.vk, alloc, &img->base.vk);
      return result;
   }

   *out_img = img;

   return VK_SUCCESS;
}

VkResult
vn_image_init_deferred(struct vn_device *dev,
                       const VkImageCreateInfo *create_info,
                       struct vn_image *img)
{
   VkResult result = vn_image_init(dev, create_info, img);
   img->deferred_info->initialized = result == VK_SUCCESS;
   return result;
}

static VkResult
vn_image_create_deferred(struct vn_device *dev,
                         const VkImageCreateInfo *create_info,
                         const VkAllocationCallbacks *alloc,
                         struct vn_image **out_img)
{
   struct vn_image *img =
      vk_image_create(&dev->base.vk, create_info, alloc, sizeof(*img));
   if (!img)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   vn_object_set_id(img, vn_get_next_obj_id(), VK_OBJECT_TYPE_IMAGE);

   VkResult result = vn_image_deferred_info_init(img, create_info, alloc);
   if (result != VK_SUCCESS) {
      vk_image_destroy(&dev->base.vk, alloc, &img->base.vk);
      return result;
   }

   *out_img = img;

   return VK_SUCCESS;
}

struct vn_image_create_info {
   VkImageCreateInfo create;
   VkExternalMemoryImageCreateInfo external;
   VkImageFormatListCreateInfo format_list;
   VkImageStencilUsageCreateInfo stencil;
   VkImageDrmFormatModifierListCreateInfoEXT modifier_list;
   VkImageDrmFormatModifierExplicitCreateInfoEXT modifier_explicit;
};

static const VkImageCreateInfo *
vn_image_fix_create_info(
   const VkImageCreateInfo *create_info,
   const VkExternalMemoryHandleTypeFlagBits renderer_handle_type,
   struct vn_image_create_info *local_info)
{
   local_info->create = *create_info;
   VkBaseOutStructure *cur = (void *)&local_info->create;

   vk_foreach_struct_const(src, create_info->pNext) {
      void *next = NULL;
      switch (src->sType) {
      case VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO:
         memcpy(&local_info->external, src, sizeof(local_info->external));
         local_info->external.handleTypes = renderer_handle_type;
         next = &local_info->external;
         break;
      case VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO:
         memcpy(&local_info->format_list, src,
                sizeof(local_info->format_list));
         next = &local_info->format_list;
         break;
      case VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO:
         memcpy(&local_info->stencil, src, sizeof(local_info->stencil));
         next = &local_info->stencil;
         break;
      case VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT:
         memcpy(&local_info->modifier_list, src,
                sizeof(local_info->modifier_list));
         next = &local_info->modifier_list;
         break;
      case VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT:
         memcpy(&local_info->modifier_explicit, src,
                sizeof(local_info->modifier_explicit));
         next = &local_info->modifier_explicit;
         break;
      default:
         break;
      }

      if (next) {
         cur->pNext = next;
         cur = next;
      }
   }

   cur->pNext = NULL;

   return &local_info->create;
}

VkResult
vn_CreateImage(VkDevice device,
               const VkImageCreateInfo *pCreateInfo,
               const VkAllocationCallbacks *pAllocator,
               VkImage *pImage)
{
   struct vn_device *dev = vn_device_from_handle(device);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;
   const VkExternalMemoryHandleTypeFlagBits renderer_handle_type =
      dev->physical_device->external_memory.renderer_handle_type;
   struct vn_image *img;
   VkResult result;

   const struct wsi_image_create_info *wsi_info = NULL;
   const VkNativeBufferANDROID *anb_info = NULL;
   const VkImageSwapchainCreateInfoKHR *swapchain_info = NULL;
   const VkExternalMemoryImageCreateInfo *external_info = NULL;
   bool ahb_info = false;

   vk_foreach_struct_const(pnext, pCreateInfo->pNext) {
      switch ((uint32_t)pnext->sType) {
      case VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA:
         wsi_info = (void *)pnext;
         break;
      case VK_STRUCTURE_TYPE_NATIVE_BUFFER_ANDROID:
         anb_info = (void *)pnext;
         break;
      case VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR:
         swapchain_info = (void *)pnext;
         if (!swapchain_info->swapchain)
            swapchain_info = NULL;
         break;
      case VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO:
         external_info = (void *)pnext;
         if (!external_info->handleTypes)
            external_info = NULL;
         else if (
            external_info->handleTypes ==
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID)
            ahb_info = true;
         break;
      default:
         break;
      }
   }

   /* No need to fix external handle type for:
    * - common wsi image: dma_buf is hard-coded in wsi_configure_native_image
    * - common wsi image alias: it aligns with wsi_info on external handle
    * - Android wsi image: VK_ANDROID_native_buffer involves no external info
    * - AHB external image: deferred creation reconstructs external info
    *
    * Must fix the external handle type for:
    * - non-AHB external image requesting handle types different from renderer
    *
    * Will have to fix more when renderer handle type is no longer dma_buf.
    */
   if (wsi_info) {
      result = vn_wsi_create_image(dev, pCreateInfo, wsi_info, alloc, &img);
   } else if (anb_info) {
      result =
         vn_android_image_from_anb(dev, pCreateInfo, anb_info, alloc, &img);
   } else if (ahb_info) {
      result = vn_image_create_deferred(dev, pCreateInfo, alloc, &img);
   } else if (swapchain_info) {
#if DETECT_OS_ANDROID
      result = vn_image_create_deferred(dev, pCreateInfo, alloc, &img);
#else
      result = wsi_common_create_swapchain_image(
         &dev->physical_device->wsi_device, pCreateInfo,
         swapchain_info->swapchain, (VkImage *)&img);
#endif
   } else {
      struct vn_image_create_info local_info;
      if (external_info &&
          external_info->handleTypes != renderer_handle_type) {
         pCreateInfo = vn_image_fix_create_info(
            pCreateInfo, renderer_handle_type, &local_info);
      }

      result = vn_image_create(dev, pCreateInfo, alloc, &img);
   }

   if (result != VK_SUCCESS)
      return vn_error(dev->instance, result);

   *pImage = vn_image_to_handle(img);
   return VK_SUCCESS;
}

void
vn_DestroyImage(VkDevice device,
                VkImage image,
                const VkAllocationCallbacks *pAllocator)
{
   struct vn_device *dev = vn_device_from_handle(device);
   struct vn_image *img = vn_image_from_handle(image);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;

   if (!img)
      return;

   if (img->wsi.memory && img->wsi.memory_owned) {
      VkDeviceMemory mem_handle = vn_device_memory_to_handle(img->wsi.memory);
      vn_FreeMemory(device, mem_handle, pAllocator);
   }

   /* must not ask renderer to destroy uninitialized deferred image */
   if (!img->deferred_info || img->deferred_info->initialized)
      vn_async_vkDestroyImage(dev->primary_ring, device, image, NULL);

   vn_image_deferred_info_fini(img, alloc);

   vk_image_destroy(&dev->base.vk, alloc, &img->base.vk);
}

void
vn_GetImageMemoryRequirements2(VkDevice device,
                               const VkImageMemoryRequirementsInfo2 *pInfo,
                               VkMemoryRequirements2 *pMemoryRequirements)
{
   const struct vn_image *img = vn_image_from_handle(pInfo->image);

   uint32_t plane = 0;
   const VkImagePlaneMemoryRequirementsInfo *plane_info =
      vk_find_struct_const(pInfo->pNext,
                           IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO);
   if (plane_info)
      plane = vn_image_get_plane(plane_info->planeAspect);

   vn_image_fill_reqs(&img->requirements[plane], pMemoryRequirements);
}

void
vn_GetImageSparseMemoryRequirements2(
   VkDevice device,
   const VkImageSparseMemoryRequirementsInfo2 *pInfo,
   uint32_t *pSparseMemoryRequirementCount,
   VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   struct vn_device *dev = vn_device_from_handle(device);

   /* see vn_GetPhysicalDeviceSparseImageFormatProperties2 */
   if (dev->physical_device->sparse_binding_disabled) {
      *pSparseMemoryRequirementCount = 0;
      return;
   }

   /* TODO local or per-device cache */
   vn_call_vkGetImageSparseMemoryRequirements2(
      dev->primary_ring, device, pInfo, pSparseMemoryRequirementCount,
      pSparseMemoryRequirements);
}

static VkResult
vn_image_bind_wsi_memory(struct vn_device *dev,
                         uint32_t count,
                         const VkBindImageMemoryInfo *infos)
{
   STACK_ARRAY(VkBindImageMemoryInfo, local_infos, count);
   typed_memcpy(local_infos, infos, count);

   for (uint32_t i = 0; i < count; i++) {
      VkBindImageMemoryInfo *info = &local_infos[i];
      struct vn_device_memory *mem =
         vn_device_memory_from_handle(info->memory);

      if (!mem) {
#if DETECT_OS_ANDROID
         mem = vn_android_get_wsi_memory_from_bind_info(dev, info);
         if (!mem) {
            STACK_ARRAY_FINISH(local_infos);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
         }
#else
         const VkBindImageMemorySwapchainInfoKHR *swapchain_info =
            vk_find_struct_const(info->pNext,
                                 BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR);
         assert(swapchain_info);

         mem = vn_device_memory_from_handle(wsi_common_get_memory(
            swapchain_info->swapchain, swapchain_info->imageIndex));
#endif
         info->memory = vn_device_memory_to_handle(mem);
         info->memoryOffset = 0;
      }
      assert(mem && info->memory != VK_NULL_HANDLE);
   }

   vn_async_vkBindImageMemory2(dev->primary_ring, vn_device_to_handle(dev),
                               count, local_infos);

   STACK_ARRAY_FINISH(local_infos);

   return VK_SUCCESS;
}

VkResult
vn_BindImageMemory2(VkDevice device,
                    uint32_t bindInfoCount,
                    const VkBindImageMemoryInfo *pBindInfos)
{
   struct vn_device *dev = vn_device_from_handle(device);

   for (uint32_t i = 0; i < bindInfoCount; i++) {
      if (pBindInfos[i].memory == VK_NULL_HANDLE)
         return vn_image_bind_wsi_memory(dev, bindInfoCount, pBindInfos);
   }

   vn_async_vkBindImageMemory2(dev->primary_ring, device, bindInfoCount,
                               pBindInfos);

   for (uint32_t i = 0; i < bindInfoCount; i++) {
      const VkBindMemoryStatus *bind_status =
         vk_find_struct((void *)pBindInfos[i].pNext, BIND_MEMORY_STATUS);
      if (bind_status)
         *bind_status->pResult = VK_SUCCESS;
   }

   return VK_SUCCESS;
}

VkResult
vn_GetImageDrmFormatModifierPropertiesEXT(
   VkDevice device,
   VkImage image,
   VkImageDrmFormatModifierPropertiesEXT *pProperties)
{
   struct vn_device *dev = vn_device_from_handle(device);

   /* TODO local cache */
   return vn_call_vkGetImageDrmFormatModifierPropertiesEXT(
      dev->primary_ring, device, image, pProperties);
}

static VkImageAspectFlags
vn_image_get_aspect(struct vn_image *img, VkImageAspectFlags aspect)
{
   if (!img->deferred_info)
      return aspect;

   switch (aspect) {
   case VK_IMAGE_ASPECT_COLOR_BIT:
   case VK_IMAGE_ASPECT_DEPTH_BIT:
   case VK_IMAGE_ASPECT_STENCIL_BIT:
   case VK_IMAGE_ASPECT_PLANE_0_BIT:
      return VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT;
   case VK_IMAGE_ASPECT_PLANE_1_BIT:
      return VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT;
   case VK_IMAGE_ASPECT_PLANE_2_BIT:
      return VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT;
   default:
      break;
   }
   UNREACHABLE("unexpected aspect");
}

void
vn_GetImageSubresourceLayout(VkDevice device,
                             VkImage image,
                             const VkImageSubresource *pSubresource,
                             VkSubresourceLayout *pLayout)
{
   struct vn_device *dev = vn_device_from_handle(device);
   struct vn_image *img = vn_image_from_handle(image);

   /* override aspect mask for ahb images with tiling modifier */
   VkImageSubresource local_subresource;
   const VkImageAspectFlags aspect =
      vn_image_get_aspect(img, pSubresource->aspectMask);
   if (aspect != pSubresource->aspectMask) {
      local_subresource = *pSubresource;
      local_subresource.aspectMask = aspect;
      pSubresource = &local_subresource;
   }

   /* TODO local cache */
   vn_call_vkGetImageSubresourceLayout(dev->primary_ring, device, image,
                                       pSubresource, pLayout);
}

/* image view commands */

VkResult
vn_CreateImageView(VkDevice device,
                   const VkImageViewCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkImageView *pView)
{
   struct vn_device *dev = vn_device_from_handle(device);
   struct vn_image *img = vn_image_from_handle(pCreateInfo->image);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;

   VkImageViewCreateInfo local_info;
   if (pCreateInfo->format == VK_FORMAT_UNDEFINED) {
      local_info = *pCreateInfo;
      local_info.format = img->base.vk.format;
      pCreateInfo = &local_info;

      assert(pCreateInfo->format != VK_FORMAT_UNDEFINED);
   }

   struct vn_image_view *view =
      vk_zalloc(alloc, sizeof(*view), VN_DEFAULT_ALIGN,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!view)
      return vn_error(dev->instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   vn_object_base_init(&view->base, VK_OBJECT_TYPE_IMAGE_VIEW, &dev->base);
   view->image = img;

   VkImageView view_handle = vn_image_view_to_handle(view);
   vn_async_vkCreateImageView(dev->primary_ring, device, pCreateInfo, NULL,
                              &view_handle);

   *pView = view_handle;

   return VK_SUCCESS;
}

void
vn_DestroyImageView(VkDevice device,
                    VkImageView imageView,
                    const VkAllocationCallbacks *pAllocator)
{
   struct vn_device *dev = vn_device_from_handle(device);
   struct vn_image_view *view = vn_image_view_from_handle(imageView);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;

   if (!view)
      return;

   vn_async_vkDestroyImageView(dev->primary_ring, device, imageView, NULL);

   vn_object_base_fini(&view->base);
   vk_free(alloc, view);
}

/* sampler commands */

VkResult
vn_CreateSampler(VkDevice device,
                 const VkSamplerCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkSampler *pSampler)
{
   struct vn_device *dev = vn_device_from_handle(device);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;

   struct vn_sampler *sampler =
      vk_zalloc(alloc, sizeof(*sampler), VN_DEFAULT_ALIGN,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!sampler)
      return vn_error(dev->instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   vn_object_base_init(&sampler->base, VK_OBJECT_TYPE_SAMPLER, &dev->base);

   VkSampler sampler_handle = vn_sampler_to_handle(sampler);
   vn_async_vkCreateSampler(dev->primary_ring, device, pCreateInfo, NULL,
                            &sampler_handle);

   *pSampler = sampler_handle;

   return VK_SUCCESS;
}

void
vn_DestroySampler(VkDevice device,
                  VkSampler _sampler,
                  const VkAllocationCallbacks *pAllocator)
{
   struct vn_device *dev = vn_device_from_handle(device);
   struct vn_sampler *sampler = vn_sampler_from_handle(_sampler);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;

   if (!sampler)
      return;

   vn_async_vkDestroySampler(dev->primary_ring, device, _sampler, NULL);

   vn_object_base_fini(&sampler->base);
   vk_free(alloc, sampler);
}

/* sampler YCbCr conversion commands */

VkResult
vn_CreateSamplerYcbcrConversion(
   VkDevice device,
   const VkSamplerYcbcrConversionCreateInfo *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkSamplerYcbcrConversion *pYcbcrConversion)
{
   struct vn_device *dev = vn_device_from_handle(device);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;
   const VkExternalFormatANDROID *ext_info =
      vk_find_struct_const(pCreateInfo->pNext, EXTERNAL_FORMAT_ANDROID);

   VkSamplerYcbcrConversionCreateInfo local_info;
   if (ext_info && ext_info->externalFormat) {
      assert(pCreateInfo->format == VK_FORMAT_UNDEFINED);

      local_info = *pCreateInfo;
      local_info.format = ext_info->externalFormat;
      local_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      local_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      local_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      local_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
      pCreateInfo = &local_info;

      assert(pCreateInfo->format != VK_FORMAT_UNDEFINED);
   }

   struct vn_sampler_ycbcr_conversion *conv =
      vk_zalloc(alloc, sizeof(*conv), VN_DEFAULT_ALIGN,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!conv)
      return vn_error(dev->instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   vn_object_base_init(&conv->base, VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION,
                       &dev->base);

   VkSamplerYcbcrConversion conv_handle =
      vn_sampler_ycbcr_conversion_to_handle(conv);
   vn_async_vkCreateSamplerYcbcrConversion(dev->primary_ring, device,
                                           pCreateInfo, NULL, &conv_handle);

   *pYcbcrConversion = conv_handle;

   return VK_SUCCESS;
}

void
vn_DestroySamplerYcbcrConversion(VkDevice device,
                                 VkSamplerYcbcrConversion ycbcrConversion,
                                 const VkAllocationCallbacks *pAllocator)
{
   struct vn_device *dev = vn_device_from_handle(device);
   struct vn_sampler_ycbcr_conversion *conv =
      vn_sampler_ycbcr_conversion_from_handle(ycbcrConversion);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &dev->base.vk.alloc;

   if (!conv)
      return;

   vn_async_vkDestroySamplerYcbcrConversion(dev->primary_ring, device,
                                            ycbcrConversion, NULL);

   vn_object_base_fini(&conv->base);
   vk_free(alloc, conv);
}

void
vn_GetDeviceImageMemoryRequirements(
   VkDevice device,
   const VkDeviceImageMemoryRequirements *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   struct vn_device *dev = vn_device_from_handle(device);

   uint8_t key[SHA1_DIGEST_LENGTH] = { 0 };
   const bool cacheable =
      vn_image_get_image_reqs_key(dev, pInfo->pCreateInfo, key);

   if (cacheable) {
      uint32_t plane = 0;
      if (pInfo->pCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT)
         plane = vn_image_get_plane(pInfo->planeAspect);

      const struct vn_image_memory_requirements *cached_reqs =
         vn_image_get_reqs_from_cache(dev, key, plane);
      if (cached_reqs) {
         vn_image_fill_reqs(cached_reqs, pMemoryRequirements);
         return;
      }

      const uint32_t plane_count =
         vn_image_get_plane_count(pInfo->pCreateInfo);
      STACK_ARRAY(VkDeviceImageMemoryRequirements, req_info, plane_count);
      STACK_ARRAY(struct vn_image_memory_requirements, reqs, plane_count);

      /* Retrieve reqs for all planes so the cache entry is complete */
      for (uint32_t i = 0; i < plane_count; i++) {
         req_info[i].sType =
            VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS;
         req_info[i].pNext = NULL;
         req_info[i].pCreateInfo = pInfo->pCreateInfo;
         req_info[i].planeAspect = VK_IMAGE_ASPECT_PLANE_0_BIT << i;

         reqs[i].memory.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
         reqs[i].memory.pNext = &reqs[i].dedicated;
         reqs[i].dedicated.sType =
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
         reqs[i].dedicated.pNext = NULL;

         vn_call_vkGetDeviceImageMemoryRequirements(
            dev->primary_ring, device, &req_info[i], &reqs[i].memory);
      }
      vn_image_fill_reqs(&reqs[plane], pMemoryRequirements);
      vn_image_store_reqs_in_cache(dev, key, plane_count, reqs);

      STACK_ARRAY_FINISH(req_info);
      STACK_ARRAY_FINISH(reqs);
   } else {
      vn_call_vkGetDeviceImageMemoryRequirements(dev->primary_ring, device,
                                                 pInfo, pMemoryRequirements);
   }
}

void
vn_GetDeviceImageSparseMemoryRequirements(
   VkDevice device,
   const VkDeviceImageMemoryRequirements *pInfo,
   uint32_t *pSparseMemoryRequirementCount,
   VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   struct vn_device *dev = vn_device_from_handle(device);

   /* see vn_GetPhysicalDeviceSparseImageFormatProperties2 */
   if (dev->physical_device->sparse_binding_disabled) {
      *pSparseMemoryRequirementCount = 0;
      return;
   }

   /* TODO per-device cache */
   vn_call_vkGetDeviceImageSparseMemoryRequirements(
      dev->primary_ring, device, pInfo, pSparseMemoryRequirementCount,
      pSparseMemoryRequirements);
}

void
vn_GetDeviceImageSubresourceLayout(VkDevice device,
                                   const VkDeviceImageSubresourceInfo *pInfo,
                                   VkSubresourceLayout2 *pLayout)
{
   struct vn_device *dev = vn_device_from_handle(device);

   /* TODO per-device cache */
   vn_call_vkGetDeviceImageSubresourceLayout(dev->primary_ring, device, pInfo,
                                             pLayout);
}

void
vn_GetImageSubresourceLayout2(VkDevice device,
                              VkImage image,
                              const VkImageSubresource2 *pSubresource,
                              VkSubresourceLayout2 *pLayout)
{
   struct vn_device *dev = vn_device_from_handle(device);
   struct vn_image *img = vn_image_from_handle(image);

   /* override aspect mask for ahb images with tiling modifier */
   VkImageSubresource2 local_subresource;
   const VkImageAspectFlags aspect =
      vn_image_get_aspect(img, pSubresource->imageSubresource.aspectMask);
   if (aspect != pSubresource->imageSubresource.aspectMask) {
      local_subresource = *pSubresource;
      local_subresource.imageSubresource.aspectMask = aspect;
      pSubresource = &local_subresource;
   }

   /* TODO local cache */
   vn_call_vkGetImageSubresourceLayout2(dev->primary_ring, device, image,
                                        pSubresource, pLayout);
}
