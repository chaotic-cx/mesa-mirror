/*
 * Copyright © 2022 Collabora Ltd
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

#include "vk_pipeline_layout.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_descriptor_set_layout.h"
#include "vk_device.h"
#include "vk_log.h"

#include "util/mesa-sha1.h"

static void
vk_pipeline_layout_init(struct vk_device *device,
                        struct vk_pipeline_layout *layout,
                        const VkPipelineLayoutCreateInfo *pCreateInfo)
{
   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
   assert(pCreateInfo->setLayoutCount <= MESA_VK_MAX_DESCRIPTOR_SETS);

   vk_object_base_init(device, &layout->base, VK_OBJECT_TYPE_PIPELINE_LAYOUT);

   layout->ref_cnt = 1;
   layout->create_flags = pCreateInfo->flags;
   layout->set_count = pCreateInfo->setLayoutCount;
   layout->push_descriptor_idx = UINT32_MAX;
   layout->destroy = vk_pipeline_layout_destroy;

   uint32_t dynamic_descriptor_count = 0;
   for (uint32_t s = 0; s < pCreateInfo->setLayoutCount; s++) {
      VK_FROM_HANDLE(vk_descriptor_set_layout, set_layout,
                     pCreateInfo->pSetLayouts[s]);

      layout->dynamic_descriptor_offset[s] = dynamic_descriptor_count;
      if (set_layout != NULL) {
         layout->set_layouts[s] = vk_descriptor_set_layout_ref(set_layout);
         if (set_layout->flags &
             VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) {
            assert(layout->push_descriptor_idx == UINT32_MAX);
            layout->push_descriptor_idx = s;
         }
         dynamic_descriptor_count += set_layout->dynamic_descriptor_count;
      } else {
         layout->set_layouts[s] = NULL;
      }
   }

   assert(pCreateInfo->pushConstantRangeCount <
          MESA_VK_MAX_PUSH_CONSTANT_RANGES);
   layout->push_range_count = pCreateInfo->pushConstantRangeCount;
   for (uint32_t r = 0; r < pCreateInfo->pushConstantRangeCount; r++)
      layout->push_ranges[r] = pCreateInfo->pPushConstantRanges[r];
}

void *
vk_pipeline_layout_zalloc(struct vk_device *device, size_t size,
                          const VkPipelineLayoutCreateInfo *pCreateInfo)
{
   /* Because we're reference counting and lifetimes may not be what the
    * client expects, these have to be allocated off the device and not as
    * their own object.
    */
   struct vk_pipeline_layout *layout =
      vk_zalloc(&device->alloc, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (layout == NULL)
      return NULL;

   vk_pipeline_layout_init(device, layout, pCreateInfo);
   return layout;
}

void *
vk_pipeline_layout_multizalloc(struct vk_device *device,
                               struct vk_multialloc *ma,
                               const VkPipelineLayoutCreateInfo *pCreateInfo)
{
   struct vk_pipeline_layout *layout =
      vk_multialloc_zalloc(ma, &device->alloc,
                           VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (layout == NULL)
      return NULL;

   vk_pipeline_layout_init(device, layout, pCreateInfo);
   return layout;
}


VKAPI_ATTR VkResult VKAPI_CALL
vk_common_CreatePipelineLayout(VkDevice _device,
                               const VkPipelineLayoutCreateInfo *pCreateInfo,
                               UNUSED const VkAllocationCallbacks *pAllocator,
                               VkPipelineLayout *pPipelineLayout)
{
   VK_FROM_HANDLE(vk_device, device, _device);

   struct vk_pipeline_layout *layout =
      vk_pipeline_layout_zalloc(device, sizeof(struct vk_pipeline_layout),
                                pCreateInfo);
   if (layout == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pPipelineLayout = vk_pipeline_layout_to_handle(layout);

   return VK_SUCCESS;
}

void
vk_pipeline_layout_destroy(struct vk_device *device,
                           struct vk_pipeline_layout *layout)
{
   assert(layout && layout->ref_cnt == 0);

   for (uint32_t s = 0; s < layout->set_count; s++) {
      if (layout->set_layouts[s] != NULL)
         vk_descriptor_set_layout_unref(device, layout->set_layouts[s]);
   }

   vk_object_free(device, NULL, layout);
}

VKAPI_ATTR void VKAPI_CALL
vk_common_DestroyPipelineLayout(VkDevice _device,
                                VkPipelineLayout pipelineLayout,
                                UNUSED const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vk_device, device, _device);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, pipelineLayout);

   if (layout == NULL)
      return;

   vk_pipeline_layout_unref(device, layout);
}
