/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>
#include "vk_descriptors.h"
#include "vk_common_entrypoints.h"
#include "util/macros.h"
#include "util/u_qsort.h"
#include "vk_alloc.h"
#include "vk_device.h"
#include "vk_util.h"

static int
binding_index_compare(const void *av, const void *bv, void *data)
{
   const VkDescriptorSetLayoutBinding *bindings = data;
   uint32_t a = bindings[*(const uint32_t *)av].binding;
   uint32_t b = bindings[*(const uint32_t *)bv].binding;

   return (a > b) - (a < b);
}

VkResult
vk_create_sorted_bindings(struct vk_device *device,
                          const VkDescriptorSetLayoutCreateInfo *create_info,
                          const VkAllocationCallbacks *alloc,
                          struct vk_sorted_bindings **sorted_out)
{
   const uint32_t count = create_info->bindingCount;
   const VkDescriptorSetLayoutBindingFlagsCreateInfo *flags_info =
      vk_find_struct_const(create_info->pNext,
                           DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(create_info->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);

   assert(!flags_info || !flags_info->bindingCount ||
          flags_info->bindingCount == count);

   VK_MULTIALLOC(ma);
   VK_MULTIALLOC_DECL(&ma, struct vk_sorted_bindings, sorted, 1);
   VK_MULTIALLOC_DECL(&ma, VkDescriptorSetLayoutBinding, bindings, count);
   VK_MULTIALLOC_DECL(&ma, VkDescriptorBindingFlags, flags, count);
   VK_MULTIALLOC_DECL(&ma, VkMutableDescriptorTypeListEXT, mutable_type_lists,
                      count);
   VK_MULTIALLOC_DECL(&ma, uint32_t, indices, count);

   *sorted_out = NULL;
   if (!vk_multialloc_zalloc2(&ma, &device->alloc, alloc,
                              VK_SYSTEM_ALLOCATION_SCOPE_COMMAND))
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   for (uint32_t i = 0; i < count; i++)
      indices[i] = i;

   if (count > 1)
      util_qsort_r(indices, count, sizeof(*indices), binding_index_compare,
                   (void *)create_info->pBindings);

   for (uint32_t i = 0; i < count; i++) {
      const uint32_t pbindings_idx = indices[i];

      bindings[i] = create_info->pBindings[pbindings_idx];

      if (flags_info && pbindings_idx < flags_info->bindingCount)
         flags[i] = flags_info->pBindingFlags[pbindings_idx];

      if (bindings[i].descriptorType == VK_DESCRIPTOR_TYPE_MUTABLE_EXT) {
         assert(mutable_info &&
                pbindings_idx < mutable_info->mutableDescriptorTypeListCount);
         mutable_type_lists[i] =
            mutable_info->pMutableDescriptorTypeLists[pbindings_idx];
      }
   }

   sorted->count = count;
   sorted->bindings = bindings;
   sorted->flags = flags;
   sorted->mutable_type_lists = mutable_type_lists;
   *sorted_out = sorted;

   return VK_SUCCESS;
}

void
vk_destroy_sorted_bindings(struct vk_device *device,
                           const VkAllocationCallbacks *alloc,
                           struct vk_sorted_bindings *sorted)
{
   vk_free2(&device->alloc, alloc, sorted);
}

/*
 * For drivers that don't have mutable state in buffers, images, image views, or
 * samplers, there's no need to save/restore anything to get the same
 * descriptor back as long as the user uses the same GPU virtual address. In
 * this case, the following EXT_descriptor_buffer functions are trivial.
 */
VKAPI_ATTR VkResult VKAPI_CALL
vk_common_GetBufferOpaqueCaptureDescriptorDataEXT(VkDevice device,
                                                  const VkBufferCaptureDescriptorDataInfoEXT *pInfo,
                                                  void *pData)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vk_common_GetImageOpaqueCaptureDescriptorDataEXT(VkDevice device,
                                                 const VkImageCaptureDescriptorDataInfoEXT *pInfo,
                                                 void *pData)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vk_common_GetImageViewOpaqueCaptureDescriptorDataEXT(VkDevice device,
                                                     const VkImageViewCaptureDescriptorDataInfoEXT *pInfo,
                                                     void *pData)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vk_common_GetSamplerOpaqueCaptureDescriptorDataEXT(VkDevice _device,
                                                   const VkSamplerCaptureDescriptorDataInfoEXT *pInfo,
                                                   void *pData)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vk_common_GetAccelerationStructureOpaqueCaptureDescriptorDataEXT(VkDevice device,
                                                                 const VkAccelerationStructureCaptureDescriptorDataInfoEXT *pInfo,
                                                                 void *pData)
{
   return VK_SUCCESS;
}
