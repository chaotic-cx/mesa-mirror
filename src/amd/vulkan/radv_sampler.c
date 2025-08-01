/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "vk_log.h"

#include "ac_descriptors.h"

#include "radv_device.h"
#include "radv_entrypoints.h"
#include "radv_physical_device.h"
#include "radv_sampler.h"

static unsigned
radv_tex_wrap(VkSamplerAddressMode address_mode)
{
   switch (address_mode) {
   case VK_SAMPLER_ADDRESS_MODE_REPEAT:
      return V_008F30_SQ_TEX_WRAP;
   case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
      return V_008F30_SQ_TEX_MIRROR;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
      return V_008F30_SQ_TEX_CLAMP_LAST_TEXEL;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
      return V_008F30_SQ_TEX_CLAMP_BORDER;
   case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE:
      return V_008F30_SQ_TEX_MIRROR_ONCE_LAST_TEXEL;
   default:
      UNREACHABLE("illegal tex wrap mode");
      break;
   }
   return 0;
}

static unsigned
radv_tex_compare(VkCompareOp op)
{
   switch (op) {
   case VK_COMPARE_OP_NEVER:
      return V_008F30_SQ_TEX_DEPTH_COMPARE_NEVER;
   case VK_COMPARE_OP_LESS:
      return V_008F30_SQ_TEX_DEPTH_COMPARE_LESS;
   case VK_COMPARE_OP_EQUAL:
      return V_008F30_SQ_TEX_DEPTH_COMPARE_EQUAL;
   case VK_COMPARE_OP_LESS_OR_EQUAL:
      return V_008F30_SQ_TEX_DEPTH_COMPARE_LESSEQUAL;
   case VK_COMPARE_OP_GREATER:
      return V_008F30_SQ_TEX_DEPTH_COMPARE_GREATER;
   case VK_COMPARE_OP_NOT_EQUAL:
      return V_008F30_SQ_TEX_DEPTH_COMPARE_NOTEQUAL;
   case VK_COMPARE_OP_GREATER_OR_EQUAL:
      return V_008F30_SQ_TEX_DEPTH_COMPARE_GREATEREQUAL;
   case VK_COMPARE_OP_ALWAYS:
      return V_008F30_SQ_TEX_DEPTH_COMPARE_ALWAYS;
   default:
      UNREACHABLE("illegal compare mode");
      break;
   }
   return 0;
}

static unsigned
radv_tex_filter(VkFilter filter, unsigned max_ansio)
{
   switch (filter) {
   case VK_FILTER_NEAREST:
      return (max_ansio > 1 ? V_008F38_SQ_TEX_XY_FILTER_ANISO_POINT : V_008F38_SQ_TEX_XY_FILTER_POINT);
   case VK_FILTER_LINEAR:
      return (max_ansio > 1 ? V_008F38_SQ_TEX_XY_FILTER_ANISO_BILINEAR : V_008F38_SQ_TEX_XY_FILTER_BILINEAR);
   case VK_FILTER_CUBIC_EXT:
   default:
      fprintf(stderr, "illegal texture filter");
      return 0;
   }
}

static unsigned
radv_tex_mipfilter(VkSamplerMipmapMode mode)
{
   switch (mode) {
   case VK_SAMPLER_MIPMAP_MODE_NEAREST:
      return V_008F38_SQ_TEX_Z_FILTER_POINT;
   case VK_SAMPLER_MIPMAP_MODE_LINEAR:
      return V_008F38_SQ_TEX_Z_FILTER_LINEAR;
   default:
      return V_008F38_SQ_TEX_Z_FILTER_NONE;
   }
}

static unsigned
radv_tex_bordercolor(VkBorderColor bcolor)
{
   switch (bcolor) {
   case VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
   case VK_BORDER_COLOR_INT_TRANSPARENT_BLACK:
      return V_008F3C_SQ_TEX_BORDER_COLOR_TRANS_BLACK;
   case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
   case VK_BORDER_COLOR_INT_OPAQUE_BLACK:
      return V_008F3C_SQ_TEX_BORDER_COLOR_OPAQUE_BLACK;
   case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
   case VK_BORDER_COLOR_INT_OPAQUE_WHITE:
      return V_008F3C_SQ_TEX_BORDER_COLOR_OPAQUE_WHITE;
   case VK_BORDER_COLOR_FLOAT_CUSTOM_EXT:
   case VK_BORDER_COLOR_INT_CUSTOM_EXT:
      return V_008F3C_SQ_TEX_BORDER_COLOR_REGISTER;
   default:
      break;
   }
   return 0;
}

static unsigned
radv_tex_aniso_filter(unsigned filter)
{
   return MIN2(util_logbase2(filter), 4);
}

static unsigned
radv_tex_filter_mode(VkSamplerReductionMode mode)
{
   switch (mode) {
   case VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE:
      return V_008F30_SQ_IMG_FILTER_MODE_BLEND;
   case VK_SAMPLER_REDUCTION_MODE_MIN:
      return V_008F30_SQ_IMG_FILTER_MODE_MIN;
   case VK_SAMPLER_REDUCTION_MODE_MAX:
      return V_008F30_SQ_IMG_FILTER_MODE_MAX;
   default:
      break;
   }
   return 0;
}

static uint32_t
radv_get_max_anisotropy(struct radv_device *device, const VkSamplerCreateInfo *pCreateInfo)
{
   if (device->force_aniso >= 0)
      return device->force_aniso;

   if (pCreateInfo->anisotropyEnable && pCreateInfo->maxAnisotropy > 1.0f)
      return (uint32_t)pCreateInfo->maxAnisotropy;

   return 0;
}

static uint32_t
radv_register_border_color(struct radv_device *device, VkClearColorValue value)
{
   uint32_t index;

   mtx_lock(&device->border_color_data.mutex);

   for (index = 0; index < RADV_BORDER_COLOR_COUNT; index++) {
      if (!device->border_color_data.used[index]) {
         /* Copy to the GPU wrt endian-ness. */
         util_memcpy_cpu_to_le32(&device->border_color_data.colors_gpu_ptr[index], &value, sizeof(VkClearColorValue));

         device->border_color_data.used[index] = true;
         break;
      }
   }

   mtx_unlock(&device->border_color_data.mutex);

   return index;
}

static void
radv_unregister_border_color(struct radv_device *device, uint32_t index)
{
   mtx_lock(&device->border_color_data.mutex);

   device->border_color_data.used[index] = false;

   mtx_unlock(&device->border_color_data.mutex);
}

void
radv_sampler_init(struct radv_device *device, struct radv_sampler *sampler, const VkSamplerCreateInfo *pCreateInfo)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   vk_sampler_init(&device->vk, &sampler->vk, pCreateInfo);

   uint32_t max_aniso = radv_get_max_anisotropy(device, pCreateInfo);
   uint32_t max_aniso_ratio = radv_tex_aniso_filter(max_aniso);
   unsigned filter_mode = radv_tex_filter_mode(sampler->vk.reduction_mode);
   unsigned depth_compare_func = V_008F30_SQ_TEX_DEPTH_COMPARE_NEVER;
   bool trunc_coord = ((pCreateInfo->minFilter == VK_FILTER_NEAREST && pCreateInfo->magFilter == VK_FILTER_NEAREST) ||
                       pdev->info.conformant_trunc_coord) &&
                      !instance->drirc.disable_trunc_coord;
   bool uses_border_color = pCreateInfo->addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                            pCreateInfo->addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                            pCreateInfo->addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
   VkBorderColor border_color = uses_border_color ? pCreateInfo->borderColor : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
   uint32_t border_color_ptr;
   bool disable_cube_wrap = pCreateInfo->flags & VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT;

   if (pCreateInfo->compareEnable)
      depth_compare_func = radv_tex_compare(pCreateInfo->compareOp);

   sampler->border_color_index = RADV_BORDER_COLOR_COUNT;

   if (vk_border_color_is_custom(border_color)) {
      sampler->border_color_index = radv_register_border_color(device, sampler->vk.border_color_value);
   }

   /* If we don't have a custom color, set the ptr to 0 */
   border_color_ptr = sampler->border_color_index != RADV_BORDER_COLOR_COUNT ? sampler->border_color_index : 0;

   struct ac_sampler_state ac_state = {
      .address_mode_u = radv_tex_wrap(pCreateInfo->addressModeU),
      .address_mode_v = radv_tex_wrap(pCreateInfo->addressModeV),
      .address_mode_w = radv_tex_wrap(pCreateInfo->addressModeW),
      .max_aniso_ratio = max_aniso_ratio,
      .depth_compare_func = depth_compare_func,
      .unnormalized_coords = pCreateInfo->unnormalizedCoordinates ? 1 : 0,
      .cube_wrap = !disable_cube_wrap,
      .trunc_coord = trunc_coord,
      .filter_mode = filter_mode,
      .mag_filter = radv_tex_filter(pCreateInfo->magFilter, max_aniso),
      .min_filter = radv_tex_filter(pCreateInfo->minFilter, max_aniso),
      .mip_filter = radv_tex_mipfilter(pCreateInfo->mipmapMode),
      .min_lod = pCreateInfo->minLod,
      .max_lod = pCreateInfo->maxLod,
      .lod_bias = pCreateInfo->mipLodBias,
      .aniso_single_level = !instance->drirc.disable_aniso_single_level,
      .border_color_type = radv_tex_bordercolor(border_color),
      .border_color_ptr = border_color_ptr,
   };

   ac_build_sampler_descriptor(pdev->info.gfx_level, &ac_state, sampler->state);
}

void
radv_sampler_finish(struct radv_device *device, struct radv_sampler *sampler)
{
   if (sampler->border_color_index != RADV_BORDER_COLOR_COUNT)
      radv_unregister_border_color(device, sampler->border_color_index);

   vk_sampler_finish(&sampler->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_CreateSampler(VkDevice _device, const VkSamplerCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                   VkSampler *pSampler)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   struct radv_sampler *sampler;

   sampler = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*sampler), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!sampler)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   radv_sampler_init(device, sampler, pCreateInfo);

   *pSampler = radv_sampler_to_handle(sampler);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
radv_DestroySampler(VkDevice _device, VkSampler _sampler, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_sampler, sampler, _sampler);

   if (!sampler)
      return;

   radv_sampler_finish(device, sampler);
   vk_free2(&device->vk.alloc, pAllocator, sampler);
}
