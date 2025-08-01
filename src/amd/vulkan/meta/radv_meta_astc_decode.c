/* Copyright (c) 2017-2023 Hans-Kristian Arntzen
 *
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <stdbool.h>

#include "radv_meta.h"
#include "sid.h"
#include "vk_format.h"

static void
decode_astc(struct radv_cmd_buffer *cmd_buffer, struct radv_image_view *src_iview, struct radv_image_view *dst_iview,
            VkImageLayout layout, const VkOffset3D *offset, const VkExtent3D *extent)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_meta_state *state = &device->meta_state;
   struct vk_texcompress_astc_write_descriptor_buffer desc_buffer;
   VkFormat format = src_iview->image->vk.format;
   int blk_w = vk_format_get_blockwidth(format);
   int blk_h = vk_format_get_blockheight(format);

   vk_texcompress_astc_fill_write_descriptor_buffer(&device->vk, state->astc_decode, &desc_buffer,
                                                    radv_image_view_to_handle(src_iview), layout,
                                                    radv_image_view_to_handle(dst_iview), format);

   VK_FROM_HANDLE(radv_buffer, luts_buf, state->astc_decode->luts_buf);
   radv_cs_add_buffer(device->ws, cmd_buffer->cs, luts_buf->bo);

   radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, state->astc_decode->p_layout,
                              VK_TEXCOMPRESS_ASTC_WRITE_DESC_SET_COUNT, desc_buffer.descriptors);

   VkPipeline pipeline =
      vk_texcompress_astc_get_decode_pipeline(&device->vk, &state->alloc, state->astc_decode, state->cache, format);
   if (pipeline == VK_NULL_HANDLE)
      return;

   radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd_buffer), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   bool is_3Dimage = (src_iview->image->vk.image_type == VK_IMAGE_TYPE_3D) ? true : false;
   int push_constants[5] = {offset->x / blk_w, offset->y / blk_h, extent->width + offset->x, extent->height + offset->y,
                            is_3Dimage};

   const VkPushConstantsInfoKHR pc_info = {
      .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO_KHR,
      .layout = device->meta_state.astc_decode->p_layout,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(push_constants),
      .pValues = push_constants,
   };

   radv_CmdPushConstants2(radv_cmd_buffer_to_handle(cmd_buffer), &pc_info);

   struct radv_dispatch_info info = {
      .blocks[0] = DIV_ROUND_UP(extent->width, blk_w * 2),
      .blocks[1] = DIV_ROUND_UP(extent->height, blk_h * 2),
      .blocks[2] = extent->depth,
      .offsets[0] = 0,
      .offsets[1] = 0,
      .offsets[2] = offset->z,
      .unaligned = 0,
   };
   radv_compute_dispatch(cmd_buffer, &info);
}

static VkImageViewType
get_view_type(const struct radv_image *image)
{
   switch (image->vk.image_type) {
   case VK_IMAGE_TYPE_2D:
      return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
   case VK_IMAGE_TYPE_3D:
      return VK_IMAGE_VIEW_TYPE_3D;
   default:
      UNREACHABLE("bad VkImageViewType");
   }
}

static void
image_view_init(struct radv_device *device, struct radv_image *image, VkFormat format, VkImageAspectFlags aspectMask,
                uint32_t baseMipLevel, uint32_t baseArrayLayer, uint32_t layerCount, struct radv_image_view *iview)
{
   VkImageViewCreateInfo iview_create_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = radv_image_to_handle(image),
      .viewType = get_view_type(image),
      .format = format,
      .subresourceRange =
         {
            .aspectMask = aspectMask,
            .baseMipLevel = baseMipLevel,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = baseArrayLayer + layerCount,
         },
   };

   radv_image_view_init(iview, device, &iview_create_info, NULL);
}

void
radv_meta_decode_astc(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, VkImageLayout layout,
                      const VkImageSubresourceLayers *subresource, VkOffset3D offset, VkExtent3D extent)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_meta_saved_state saved_state;
   radv_meta_save(&saved_state, cmd_buffer,
                  RADV_META_SAVE_COMPUTE_PIPELINE | RADV_META_SAVE_CONSTANTS | RADV_META_SAVE_DESCRIPTORS);

   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   const uint32_t base_slice = is_3d ? offset.z : subresource->baseArrayLayer;
   const uint32_t slice_count = is_3d ? extent.depth : vk_image_subresource_layer_count(&image->vk, subresource);

   extent = vk_image_sanitize_extent(&image->vk, extent);
   offset = vk_image_sanitize_offset(&image->vk, offset);

   struct radv_image_view src_iview, dst_iview;
   image_view_init(device, image, VK_FORMAT_R32G32B32A32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, subresource->mipLevel,
                   subresource->baseArrayLayer, vk_image_subresource_layer_count(&image->vk, subresource), &src_iview);
   image_view_init(device, image, VK_FORMAT_R8G8B8A8_UINT, VK_IMAGE_ASPECT_PLANE_1_BIT, subresource->mipLevel,
                   subresource->baseArrayLayer, vk_image_subresource_layer_count(&image->vk, subresource), &dst_iview);

   VkExtent3D extent_copy = {
      .width = extent.width,
      .height = extent.height,
      .depth = slice_count,
   };
   decode_astc(cmd_buffer, &src_iview, &dst_iview, layout, &(VkOffset3D){offset.x, offset.y, base_slice}, &extent_copy);

   radv_image_view_finish(&src_iview);
   radv_image_view_finish(&dst_iview);

   radv_meta_restore(&saved_state, cmd_buffer);
}
