/*
 * Copyright © 2026 Arm Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#include "panvk_buffer.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_desc_state.h"
#include "panvk_entrypoints.h"
#include "panvk_sampler.h"

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetDescriptorSetLayoutSizeEXT)(VkDevice _device,
                                              VkDescriptorSetLayout _layout,
                                              VkDeviceSize *pSize)
{
   VK_FROM_HANDLE(panvk_descriptor_set_layout, layout, _layout);
   *pSize = layout->desc_count * PANVK_DESCRIPTOR_SIZE;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetDescriptorSetLayoutBindingOffsetEXT)(
   VkDevice _device, VkDescriptorSetLayout _layout, uint32_t binding,
   VkDeviceSize *pOffset)
{
   VK_FROM_HANDLE(panvk_descriptor_set_layout, layout, _layout);
   assert(binding < layout->binding_count);

   const uint32_t index = layout->bindings[binding].desc_idx;
   *pOffset = index * PANVK_DESCRIPTOR_SIZE;
}

#if PAN_ARCH >= 9
VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetDescriptorEXT)(VkDevice _device,
                                 const VkDescriptorGetInfoEXT *pCreateInfo,
                                 size_t size, void *pDescriptor)
{
   /* note: pDescriptor might not have the required alignment for a direct cast
    * to packed descriptor types, so we pack on stack and copy instead. */

   const VkDescriptorAddressInfoEXT *buffer_info = NULL;
   const VkDescriptorImageInfo *image_info = NULL;
   const VkSampler *sampler_info = NULL;

   bool is_texel_buffer = false;
   bool is_storage = false;

   switch (pCreateInfo->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      buffer_info = pCreateInfo->data.pUniformTexelBuffer;
      is_texel_buffer = true;
      break;
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      buffer_info = pCreateInfo->data.pStorageTexelBuffer;
      is_texel_buffer = true;
      is_storage = true;
      break;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      buffer_info = pCreateInfo->data.pUniformBuffer;
      break;
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      buffer_info = pCreateInfo->data.pStorageBuffer;
      is_storage = true;
      break;
   case VK_DESCRIPTOR_TYPE_SAMPLER:
      sampler_info = pCreateInfo->data.pSampler;
      break;
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      image_info = pCreateInfo->data.pCombinedImageSampler;
      break;
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      image_info = pCreateInfo->data.pSampledImage;
      break;
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      image_info = pCreateInfo->data.pStorageImage;
      is_storage = true;
      break;
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      image_info = pCreateInfo->data.pInputAttachmentImage;
      break;
   default:
      break;
   }

   if (sampler_info) {
      VK_FROM_HANDLE(panvk_sampler, sampler, *sampler_info);
      /* multi-descriptor samplers are only used with separate_reconstruction
       * ycbcr, which requires CombinedImageSampler. */
      assert(size == PANVK_DESCRIPTOR_SIZE && sampler->desc_count == 1);
      memcpy(pDescriptor, sampler->descs, PANVK_DESCRIPTOR_SIZE);
   } else if (image_info) {
      VK_FROM_HANDLE(panvk_image_view, view, image_info->imageView);
      uint32_t tex_size = PANVK_DESCRIPTOR_SIZE;

      if (image_info->imageView) {
         tex_size *= panvk_image_get_tex_count(PAN_ARCH, view->vk.view_format);
         assert(size >= tex_size);
         if (is_storage)
            memcpy(pDescriptor, &view->descs.storage_tex, PANVK_DESCRIPTOR_SIZE);
         else
            memcpy(pDescriptor, view->descs.tex, tex_size);
      } else {
         struct mali_null_descriptor_packed nulldesc;
         pan_pack(&nulldesc, NULL_DESCRIPTOR, cfg)
            ;
         memcpy(pDescriptor, &nulldesc, PANVK_DESCRIPTOR_SIZE);
      }

      if (image_info->sampler &&
          pCreateInfo->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
         VK_FROM_HANDLE(panvk_sampler, sampler, image_info->sampler);
         assert(size == tex_size * 2);
         memcpy((pDescriptor + tex_size), sampler->descs,
                sampler->desc_count * PANVK_DESCRIPTOR_SIZE);
      }
   } else if (buffer_info && buffer_info->address != 0) {
      assert(size == PANVK_DESCRIPTOR_SIZE);
      uint64_t buffer_size = align(buffer_info->range, is_storage ? 4 : 16);
      uint32_t stride = 0;
      enum mali_buffer_type buffer_type = MALI_BUFFER_TYPE_SIMPLE;
      struct MALI_INTERNAL_CONVERSION conv = {0};

      if (is_texel_buffer) {
         conv = (struct MALI_INTERNAL_CONVERSION){
            .memory_format = GENX(pan_format_from_pipe_format)(
                                vk_format_to_pipe_format(buffer_info->format))
                                ->hw,
            .raw = false,
         };
         stride = util_format_get_blocksize(
            vk_format_to_pipe_format(buffer_info->format));
         buffer_type = MALI_BUFFER_TYPE_STRUCTURE;
         buffer_size = buffer_info->range;
      }
      struct mali_buffer_packed buffer;
      pan_pack(&buffer, BUFFER, cfg) {
         cfg.buffer_type = buffer_type;
         cfg.address = buffer_info->address;

#if PAN_ARCH >= 11
         cfg.size = buffer_size & BITFIELD_MASK(32);
         cfg.size_hi = buffer_size >> 32;
#else
         cfg.size = buffer_size;
#endif

         cfg.conversion = conv;
         cfg.stride = stride;
      }
      memcpy(pDescriptor, &buffer, PANVK_DESCRIPTOR_SIZE);
   } else {
      struct mali_null_descriptor_packed nulldesc;
      pan_pack(&nulldesc, NULL_DESCRIPTOR, cfg)
         ;
      memcpy(pDescriptor, &nulldesc, PANVK_DESCRIPTOR_SIZE);
   }
}
#endif /* PAN_ARCH >= 9 */
