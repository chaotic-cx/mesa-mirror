/*
 * Copyright 2010 Jerome Glisse <glisse@freedesktop.org>
 * Copyright 2015-2021 Advanced Micro Devices, Inc.
 * Copyright 2023 Valve Corporation
 * All Rights Reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_sdma.h"
#include "meta/radv_meta.h"
#include "util/macros.h"
#include "radv_cs.h"
#include "radv_formats.h"

#include "ac_cmdbuf_sdma.h"

struct radv_sdma_chunked_copy_info {
   unsigned extent_horizontal_blocks;
   unsigned extent_vertical_blocks;
   unsigned aligned_row_pitch;
   unsigned num_rows_per_copy;
};

ALWAYS_INLINE static uint32_t
radv_sdma_surf_type_from_aspect_mask(const VkImageAspectFlags aspectMask)
{
   if (aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)
      return 1;
   else if (aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)
      return 2;

   return 0;
}

static struct radv_sdma_chunked_copy_info
radv_sdma_get_chunked_copy_info(const struct radv_device *const device, const struct ac_sdma_surf *const img,
                                const VkExtent3D extent)
{
   const uint32_t blk_h = util_format_get_blockheight(img->img_format);
   const unsigned extent_horizontal_blocks = extent.width;
   const unsigned extent_vertical_blocks = extent.height;
   const unsigned aligned_row_pitch = align(extent_horizontal_blocks, 4);
   const unsigned aligned_row_bytes = aligned_row_pitch * img->bpp;

   /* Assume that we can always copy at least one full row at a time. */
   const unsigned max_num_rows_per_copy =
      MIN2(RADV_SDMA_TRANSFER_TEMP_BYTES / aligned_row_bytes, extent.height * blk_h);
   assert(max_num_rows_per_copy);

   /* Ensure that the number of rows copied at a time is a power of two. */
   const unsigned num_rows_per_copy = MAX2(1, util_next_power_of_two(max_num_rows_per_copy + 1) / 2);

   const struct radv_sdma_chunked_copy_info r = {
      .extent_horizontal_blocks = extent_horizontal_blocks,
      .extent_vertical_blocks = extent_vertical_blocks,
      .aligned_row_pitch = aligned_row_pitch,
      .num_rows_per_copy = num_rows_per_copy,
   };

   return r;
}

static uint32_t
radv_sdma_get_bpe(const struct radv_image *const image, const VkImageSubresourceLayers *subresource)
{
   VkFormat format = vk_format_get_aspect_format(image->vk.format, subresource->aspectMask);

   if (vk_format_is_96bit(format))
      format = radv_meta_get_96bit_channel_format(format);

   return vk_format_get_blocksize(format);
}

struct ac_sdma_surf
radv_sdma_get_buf_surf(const struct radv_image *const image, const VkDeviceMemoryImageCopyKHR *const region)
{
   const struct vk_image_buffer_layout layout = vk_image_memory_copy_layout(&image->vk, region);

   assert(util_bitcount(region->imageSubresource.aspectMask) == 1);

   const uint32_t texel_scale = radv_sdma_get_texel_scale(image);
   const unsigned pitch = (layout.row_stride_B / layout.element_size_B) * texel_scale;
   const unsigned slice_pitch = layout.image_stride_B / layout.element_size_B;

   const uint32_t bpe = radv_sdma_get_bpe(image, &region->imageSubresource);

   const struct ac_sdma_surf info = {
      .va = region->addressRange.address,
      .pitch = pitch,
      .slice_pitch = slice_pitch,
      .bpp = bpe,
      .is_secure = image->vk.create_flags & VK_IMAGE_CREATE_2_PROTECTED_BIT_KHR,
   };

   return info;
}

struct ac_sdma_surf
radv_sdma_get_surf(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *const image, VkImageLayout image_layout,
                   const VkImageSubresourceLayers subresource, const VkOffset3D offset)
{
   assert(util_bitcount(subresource.aspectMask) == 1);

   const struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const unsigned plane_idx = radv_plane_from_aspect(subresource.aspectMask);
   const unsigned binding_idx = image->disjoint ? plane_idx : 0;
   const struct radeon_surf *const surf = &image->planes[plane_idx].surface;
   const struct radv_image_binding *binding = &image->bindings[binding_idx];
   const uint64_t va = binding->addr;
   const uint32_t bpe = radv_sdma_get_bpe(image, &subresource);
   const uint32_t blk_w = vk_format_get_blockwidth(image->vk.format);
   const VkFormat aspect_format = vk_format_get_aspect_format(image->vk.format, subresource.aspectMask);
   const uint32_t first_level = subresource.mipLevel;
   const uint32_t texel_scale = radv_sdma_get_texel_scale(image);
   const bool is_stencil = subresource.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT;
   const struct legacy_surf_level *legacy_level = NULL;

   if (pdev->info.gfx_level < GFX9)
      legacy_level = ac_surface_get_legacy_level(surf, first_level, is_stencil);

   struct ac_sdma_surf info = {
      .surf = surf,
      .img_format = radv_format_to_pipe_format(image->vk.format),
      .format = radv_format_to_pipe_format(aspect_format),
      .is_color = vk_format_is_color(aspect_format),
      .extent =
         {
            .width = vk_format_get_plane_width(image->vk.format, plane_idx, image->vk.extent.width),
            .height = vk_format_get_plane_height(image->vk.format, plane_idx, image->vk.extent.height),
         },
      .bpp = bpe,
      .first_level = first_level,
      .num_levels = image->vk.mip_levels,
      .is_stencil = is_stencil,
      .is_secure = image->vk.create_flags & VK_IMAGE_CREATE_2_PROTECTED_BIT_KHR,
   };

   info.offset.x = offset.x * texel_scale;
   info.offset.y = offset.y;
   info.offset.z = image->vk.image_type == VK_IMAGE_TYPE_3D ? offset.z : subresource.baseArrayLayer;

   const VkExtent3D img_extent = {
      .width = vk_format_get_plane_width(image->vk.format, plane_idx, image->vk.extent.width),
      .height = vk_format_get_plane_height(image->vk.format, plane_idx, image->vk.extent.height),
   };

   const VkExtent3D img_extent_el = vk_image_extent_to_elements(&image->vk, img_extent);

   info.extent.width = img_extent_el.width;
   info.extent.height = img_extent_el.height;
   info.extent.depth = image->vk.image_type == VK_IMAGE_TYPE_3D ? image->vk.extent.depth : image->vk.array_layers;

   uint64_t surf_offset = 0;

   if (pdev->info.gfx_level >= GFX9) {
      if (is_stencil)
         surf_offset = surf->u.gfx9.zs.stencil_offset;
      else
         surf_offset = surf->u.gfx9.surf_offset;
   } else {
      surf_offset = (uint64_t)legacy_level->offset_256B * 256;
   }

   if (surf->is_linear) {
      info.va = va + surf_offset;

      if (pdev->info.gfx_level >= GFX9) {
         info.va += surf->u.gfx9.offset[first_level];
         info.pitch = surf->u.gfx9.pitch[first_level] / blk_w;
         info.slice_pitch = surf->u.gfx9.surf_slice_size / bpe;
      } else {
         info.pitch = legacy_level->nblk_x * texel_scale;
         info.slice_pitch = legacy_level->slice_size_dw * 4 / bpe;
      }
   } else {
      const uint32_t queue_mask = radv_image_queue_family_mask(image, cmd_buffer->qf);
      const bool htile_compressed =
         radv_layout_is_htile_compressed(device, image, subresource.mipLevel, image_layout, queue_mask);
      const bool dcc_compressed =
         radv_layout_dcc_compressed(device, image, subresource.mipLevel, image_layout, queue_mask);

      /* 1D resources should be linear. */
      if (pdev->info.gfx_level >= GFX9)
         assert(surf->u.gfx9.resource_type != RADEON_RESOURCE_1D);

      info.va = va + surf_offset;

      if (pdev->info.gfx_level >= GFX9 || surf->u.legacy.level[first_level].mode == RADEON_SURF_MODE_2D)
         info.va |= surf->tile_swizzle << 8;

      if (dcc_compressed || htile_compressed) {
         assert(pdev->info.sdma_supports_compression && pdev->info.gfx_level < GFX12);

         info.is_compressed = true;
         info.meta_va = va + surf->meta_offset;
         info.surf_type = radv_sdma_surf_type_from_aspect_mask(subresource.aspectMask);
         info.htile_enabled = htile_compressed;
      }
   }

   if (pdev->info.gfx_level >= GFX12)
      info.is_compressed = binding->bo && binding->bo->gfx12_allow_dcc;

   return info;
}

void
radv_sdma_emit_nop(const struct radv_device *device, struct radv_cmd_stream *cs)
{
   radeon_check_space(device->ws, cs->b, 1);
   ac_emit_sdma_nop(cs->b);
}

void
radv_sdma_copy_memory(const struct radv_device *device, struct radv_cmd_stream *cs, uint64_t src_va, uint64_t dst_va,
                      uint64_t size, bool tmz)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   while (size > 0) {
      radeon_check_space(device->ws, cs->b, 8);
      uint64_t bytes_written = ac_emit_sdma_copy_linear(cs->b, pdev->info.sdma_ip_version, src_va, dst_va, size, tmz);

      size -= bytes_written;
      src_va += bytes_written;
      dst_va += bytes_written;
   }
}

void
radv_sdma_fill_memory(const struct radv_device *device, struct radv_cmd_stream *cs, uint64_t va, uint64_t size,
                      const uint32_t value)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   while (size > 0) {
      radeon_check_space(device->ws, cs->b, 5);
      uint64_t bytes_written = ac_emit_sdma_constant_fill(cs->b, pdev->info.sdma_ip_version, va, size, value);

      size -= bytes_written;
      va += bytes_written;
   }
}

static void
radv_sdma_emit_copy_linear_sub_window(const struct radv_device *device, struct radv_cmd_stream *cs,
                                      const struct ac_sdma_surf *const src, const struct ac_sdma_surf *const dst,
                                      const VkExtent3D extent)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   radeon_check_space(device->ws, cs->b, 13);
   ac_emit_sdma_copy_linear_sub_window(cs->b, pdev->info.sdma_ip_version, src, dst, extent.width, extent.height,
                                       extent.depth);
}

static void
radv_sdma_emit_copy_tiled_sub_window(const struct radv_device *device, struct radv_cmd_stream *cs,
                                     const struct ac_sdma_surf *const tiled, const struct ac_sdma_surf *const linear,
                                     const VkExtent3D extent, const bool detile)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   bool tmz = tiled->is_secure;
   assert(!tmz || linear->is_secure);

   radeon_check_space(device->ws, cs->b, 17);
   ac_emit_sdma_copy_tiled_sub_window(cs->b, &pdev->info, linear, tiled, detile, extent.width, extent.height,
                                      extent.depth, tmz);
}

static void
radv_sdma_emit_copy_t2t_sub_window(const struct radv_device *device, struct radv_cmd_stream *cs,
                                   const struct ac_sdma_surf *const src, const struct ac_sdma_surf *const dst,
                                   const VkExtent3D extent)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   radeon_check_space(device->ws, cs->b, 18);
   ac_emit_sdma_copy_t2t_sub_window(cs->b, &pdev->info, src, dst, extent.width, extent.height, extent.depth);
}

static void
radv_sdma_copy_buffer_image_split(const struct radv_device *device, struct radv_cmd_stream *cs,
                                  const struct ac_sdma_surf *buf, const struct ac_sdma_surf *img,
                                  const VkExtent3D extent, bool to_image)
{
   const struct radv_physical_device *const pdev = radv_device_physical(device);
   const uint32_t max_extent = ac_sdma_max_img_extent(pdev->info.sdma_ip_version);
   const uint32_t first_part = util_next_power_of_two(max_extent) / 2;
   VkExtent3D e = extent;
   struct ac_sdma_surf b = *buf;
   struct ac_sdma_surf i = *img;

   if (extent.width > max_extent) {
      e.width = first_part;
      radv_sdma_copy_buffer_image(device, cs, &b, &i, e, to_image);

      const uint32_t second_part = extent.width - first_part;
      e.width = second_part;
      b.offset.x += first_part;
      i.offset.x += first_part;
      radv_sdma_copy_buffer_image(device, cs, &b, &i, e, to_image);

      return;
   }

   if (extent.height > max_extent) {
      e.height = first_part;
      radv_sdma_copy_buffer_image(device, cs, &b, &i, e, to_image);

      const uint32_t second_part = extent.height - first_part;
      e.height = second_part;
      b.offset.y += first_part;
      i.offset.y += first_part;
      radv_sdma_copy_buffer_image(device, cs, &b, &i, e, to_image);
   }
}

void
radv_sdma_copy_buffer_image(const struct radv_device *device, struct radv_cmd_stream *cs,
                            const struct ac_sdma_surf *buf, const struct ac_sdma_surf *img, const VkExtent3D extent,
                            bool to_image)
{
   const struct radv_physical_device *const pdev = radv_device_physical(device);
   const uint32_t max_extent = ac_sdma_max_img_extent(pdev->info.sdma_ip_version);

   if (extent.width > max_extent || extent.height > max_extent) {
      radv_sdma_copy_buffer_image_split(device, cs, buf, img, extent, to_image);
      return;
   }

   if (img->surf->is_linear) {
      if (to_image)
         radv_sdma_emit_copy_linear_sub_window(device, cs, buf, img, extent);
      else
         radv_sdma_emit_copy_linear_sub_window(device, cs, img, buf, extent);
   } else {
      radv_sdma_emit_copy_tiled_sub_window(device, cs, img, buf, extent, !to_image);
   }
}

static void
radv_sdma_single_copy_buffer_image_unaligned(const struct radv_device *device, struct radv_cmd_stream *cs,
                                             const struct ac_sdma_surf *buf, const struct ac_sdma_surf *img_in,
                                             const VkExtent3D base_extent, const uint64_t temp_bo_va, bool to_image)
{
   const struct radv_physical_device *const pdev = radv_device_physical(device);

   bool copy_from_dst = false;
   uint32_t x_diff = 0;
   struct ac_sdma_surf img = *img_in;
   VkExtent3D extent = base_extent;
   extent.depth = 1;

   /* SDMA v2-v3:
    * Align image offset and copy extent to dwords,
    * which is required by the HW, otherwise SDMA can hang.
    *
    * For image to buffer copies, this just means that a wider chunk
    * of the image is copied to the temporary buffer and then we
    * only copy the necessary parts using linear copy.
    *
    * For buffer to image copies, this means that we need to copy
    * a wider chunk from the temporary buffer to the destination image.
    * To make that work, we need to first copy part of the
    * destination image to the temporary buffer.
    */
   if (pdev->info.sdma_ip_version < SDMA_4_0) {
      /* Align the offset down to dwords */
      uint32_t adjusted_x = ((img.offset.x * img.bpp) & 0xfffffffc) / img.bpp;
      assert(adjusted_x <= img.offset.x);
      x_diff = img.offset.x - adjusted_x;

      /* Align the width up to dwords, taking into account the adjusted offset. */
      uint32_t adjusted_width = align((extent.width + x_diff) * img.bpp, 4) / img.bpp;
      assert(adjusted_width >= extent.width);

      if (to_image)
         copy_from_dst = adjusted_x != img.offset.x || adjusted_width != extent.width;

      img.offset.x = adjusted_x;
      extent.width = adjusted_width;
   }

   const struct radv_sdma_chunked_copy_info info = radv_sdma_get_chunked_copy_info(device, &img, extent);
   struct ac_sdma_surf tmp = {
      .surf = img.surf,
      .va = temp_bo_va,
      .format = img.format,
      .bpp = img.bpp,
      .pitch = info.aligned_row_pitch,
      .slice_pitch = info.aligned_row_pitch * info.extent_vertical_blocks,
   };

   bool tmz = buf->is_secure;
   assert(!tmz || img_in->is_secure);

   for (unsigned slice = 0; slice < base_extent.depth; ++slice) {
      for (unsigned row = 0; row < info.extent_vertical_blocks; row += info.num_rows_per_copy) {
         const unsigned rows = MIN2(info.extent_vertical_blocks - row, info.num_rows_per_copy);

         img.offset.y = img_in->offset.y + row;
         img.offset.z = img_in->offset.z + slice;
         extent.height = rows;
         tmp.slice_pitch = tmp.pitch * rows;

         if (copy_from_dst) {
            /* This workaround is necessary to dword-align some copy operations.
             * 1 bytes per pixel - need 4 pixels to dword-align
             * 2 bytes per pixel - need 2 pixels to dword-align
             * 4 bytes per pixel (or more) - not needed (every pixel is dword aligned)
             */
            const uint32_t pixels_from_dst = img.bpp == 1 ? 4 : 2;

            if (extent.width <= pixels_from_dst * 2) {
               /* Image is small, just copy the whole thing. */
               radv_sdma_copy_buffer_image(device, cs, &tmp, &img, extent, false);
            } else {
               /* We aligned the left edge (X offset) to make the copy possible.
                * Need to copy a few pixels from the left edge of the destination image.
                */
               if (img.offset.x != img_in->offset.x) {
                  VkExtent3D left_extent = extent;
                  left_extent.width = pixels_from_dst;
                  radv_sdma_copy_buffer_image(device, cs, &tmp, &img, left_extent, false);
               }

               /* We aligned the right edge (X offset + width) to make the copy possible.
                * Need to copy a few pixels from the right edge of the destination image.
                */
               if ((img.offset.x + extent.width) != (img_in->offset.x + base_extent.width)) {
                  struct ac_sdma_surf right_img = img;
                  struct ac_sdma_surf right_tmp = tmp;
                  VkExtent3D right_extent = extent;
                  right_extent.width = pixels_from_dst;
                  right_img.offset.x += extent.width - right_extent.width;
                  right_tmp.va += (extent.width - right_extent.width) * img.bpp;
                  radv_sdma_copy_buffer_image(device, cs, &right_tmp, &right_img, right_extent, false);
               }
            }

            /* Wait for the copies to finish. */
            radv_sdma_emit_nop(device, cs);
         }

         if (!to_image) {
            /* Copy the rows from the image to the temporary buffer. */
            radv_sdma_copy_buffer_image(device, cs, &tmp, &img, extent, false);

            /* Wait for the copy to finish. */
            radv_sdma_emit_nop(device, cs);
         }

         /* buffer to image: copy each row from source buffer to temporary buffer.
          * image to buffer: copy each row from temporary buffer to destination buffer.
          */
         for (unsigned r = 0; r < rows; ++r) {
            const uint64_t buf_va = buf->va + slice * buf->slice_pitch * img.bpp + (row + r) * buf->pitch * img.bpp;
            const uint64_t tmp_va = tmp.va + (r * info.aligned_row_pitch + x_diff) * img.bpp;
            radv_sdma_copy_memory(device, cs, to_image ? buf_va : tmp_va, to_image ? tmp_va : buf_va,
                                  base_extent.width * img.bpp, tmz);
         }

         /* Wait for the copy to finish. */
         radv_sdma_emit_nop(device, cs);

         if (to_image) {
            /* Copy the rows from the temporary buffer to the destination image. */
            radv_sdma_copy_buffer_image(device, cs, &tmp, &img, extent, true);

            /* Wait for the copy to finish. */
            radv_sdma_emit_nop(device, cs);
         }
      }
   }
}

void
radv_sdma_copy_buffer_image_unaligned(const struct radv_device *device, struct radv_cmd_stream *cs,
                                      const struct ac_sdma_surf *buf, const struct ac_sdma_surf *img,
                                      const VkExtent3D extent, const uint64_t temp_bo_va, bool to_image)
{
   const struct radv_physical_device *const pdev = radv_device_physical(device);

   if (pdev->info.sdma_ip_version >= SDMA_4_0) {
      radv_sdma_single_copy_buffer_image_unaligned(device, cs, buf, img, extent, temp_bo_va, to_image);
      return;
   }

   /* SDMA v3.x and older have restrictions on image offset and extent alignment.
    * Instead of applying the full unaligned buffer copy workaround which
    * causes us to copy the image twice, we can try to align the image
    * so that we can perform the bulk of the copy on a properly aligned image
    * and only apply the workaround near the left and right edges.
    * This way, we minimize the perf loss caused by the misalignment.
    */
   const uint32_t pixel_alignment = MAX2(8 * img->bpp, 16) / img->bpp;
   struct ac_sdma_surf aligned_img = *img;
   struct ac_sdma_surf aligned_buf = *buf;
   VkExtent3D aligned_extent = extent;

   aligned_img.offset.x = align(img->offset.x, pixel_alignment);
   aligned_extent.width -= aligned_img.offset.x - img->offset.x;
   aligned_extent.width = align(aligned_extent.width, pixel_alignment) - pixel_alignment;
   aligned_buf.va = buf->va + (aligned_img.offset.x - img->offset.x) * img->bpp;

   if (extent.width <= pixel_alignment * 2 || !aligned_extent.width ||
       ac_sdma_use_unaligned_buffer_image_copy(&pdev->info, &aligned_buf, &aligned_img, aligned_extent.width,
                                               aligned_extent.height, aligned_extent.depth)) {
      radv_sdma_single_copy_buffer_image_unaligned(device, cs, buf, img, extent, temp_bo_va, to_image);
      return;
   }

   /* Use aligned copy for the bulk of the image. */
   radv_sdma_copy_buffer_image(device, cs, &aligned_buf, &aligned_img, aligned_extent, to_image);

   /* Use unaligned copy workaround for the left edge. */
   VkExtent3D left_extent = extent;
   left_extent.width = aligned_img.offset.x - img->offset.x;

   if (left_extent.width)
      radv_sdma_single_copy_buffer_image_unaligned(device, cs, buf, img, left_extent, temp_bo_va, to_image);

   /* Use unaligned copy workaround for the right edge. */
   VkExtent3D right_extent = extent;
   right_extent.width = img->offset.x + extent.width - aligned_img.offset.x - aligned_extent.width;
   struct ac_sdma_surf right_buf = *buf;
   right_buf.va = buf->va + (extent.width - right_extent.width) * img->bpp;
   struct ac_sdma_surf right_img = *img;
   right_img.offset.x = img->offset.x + extent.width - right_extent.width;

   if (right_extent.width)
      radv_sdma_single_copy_buffer_image_unaligned(device, cs, &right_buf, &right_img, right_extent, temp_bo_va,
                                                   to_image);
}

static void
radv_sdma_copy_image_split(const struct radv_device *device, struct radv_cmd_stream *cs, const struct ac_sdma_surf *src,
                           const struct ac_sdma_surf *dst, const VkExtent3D extent)
{
   const struct radv_physical_device *const pdev = radv_device_physical(device);
   const uint32_t max_extent = ac_sdma_max_img_extent(pdev->info.sdma_ip_version);
   const uint32_t first_part = util_next_power_of_two(max_extent) / 2;
   VkExtent3D e = extent;
   struct ac_sdma_surf s = *src;
   struct ac_sdma_surf d = *dst;

   if (extent.width > max_extent) {
      e.width = first_part;
      radv_sdma_copy_image(device, cs, &s, &d, e);

      const uint32_t second_part = extent.width - first_part;
      e.width = second_part;
      s.offset.x += first_part;
      d.offset.x += first_part;
      radv_sdma_copy_image(device, cs, &s, &d, e);

      return;
   }

   if (extent.height > max_extent) {
      e.height = first_part;
      radv_sdma_copy_image(device, cs, &s, &d, e);

      const uint32_t second_part = extent.height - first_part;
      e.height = second_part;
      s.offset.y += first_part;
      d.offset.y += first_part;
      radv_sdma_copy_image(device, cs, &s, &d, e);
   }
}

void
radv_sdma_copy_image(const struct radv_device *device, struct radv_cmd_stream *cs, const struct ac_sdma_surf *src,
                     const struct ac_sdma_surf *dst, const VkExtent3D extent)
{
   const struct radv_physical_device *const pdev = radv_device_physical(device);
   const uint32_t max_extent = ac_sdma_max_img_extent(pdev->info.sdma_ip_version);

   if (extent.width > max_extent || extent.height > max_extent) {
      radv_sdma_copy_image_split(device, cs, src, dst, extent);
      return;
   }

   if (src->surf->is_linear) {
      if (dst->surf->is_linear) {
         radv_sdma_emit_copy_linear_sub_window(device, cs, src, dst, extent);
      } else {
         radv_sdma_emit_copy_tiled_sub_window(device, cs, dst, src, extent, false);
      }
   } else {
      if (dst->surf->is_linear) {
         radv_sdma_emit_copy_tiled_sub_window(device, cs, src, dst, extent, true);
      } else {
         radv_sdma_emit_copy_t2t_sub_window(device, cs, src, dst, extent);
      }
   }
}

void
radv_sdma_copy_image_t2t_scanline(const struct radv_device *device, struct radv_cmd_stream *cs,
                                  const struct ac_sdma_surf *src, const struct ac_sdma_surf *dst,
                                  const VkExtent3D extent, const uint64_t temp_bo_va)
{
   const struct radv_physical_device *const pdev = radv_device_physical(device);
   const struct radv_sdma_chunked_copy_info info = radv_sdma_get_chunked_copy_info(device, src, extent);
   const uint64_t temp_bufimg_va = temp_bo_va + RADV_SDMA_TRANSFER_TEMP_BYTES;
   struct ac_sdma_surf t2l_src = *src;
   struct ac_sdma_surf t2l_dst = {
      .surf = src->surf,
      .va = temp_bo_va,
      .format = src->format,
      .bpp = src->bpp,
      .is_compressed = src->is_compressed,
      .pitch = info.aligned_row_pitch,
   };
   struct ac_sdma_surf l2t_dst = *dst;
   struct ac_sdma_surf l2t_src = {
      .surf = dst->surf,
      .va = temp_bo_va,
      .format = dst->format,
      .bpp = dst->bpp,
      .is_compressed = dst->is_compressed,
      .pitch = info.aligned_row_pitch,
   };

   for (unsigned slice = 0; slice < extent.depth; ++slice) {
      for (unsigned row = 0; row < info.extent_vertical_blocks; row += info.num_rows_per_copy) {
         const unsigned rows = MIN2(info.extent_vertical_blocks - row, info.num_rows_per_copy);

         const VkExtent3D new_extent = {
            .width = info.extent_horizontal_blocks,
            .height = rows,
            .depth = 1,
         };

         t2l_src.offset.y = src->offset.y + row;
         t2l_src.offset.z = src->offset.z + slice;
         t2l_dst.slice_pitch = t2l_dst.pitch * rows;

         if (!ac_sdma_use_unaligned_buffer_image_copy(&pdev->info, &t2l_dst, &t2l_src, new_extent.width,
                                                      new_extent.height, new_extent.depth))
            radv_sdma_copy_buffer_image(device, cs, &t2l_dst, &t2l_src, new_extent, false);
         else
            radv_sdma_copy_buffer_image_unaligned(device, cs, &t2l_dst, &t2l_src, new_extent, temp_bufimg_va, false);
         radv_sdma_emit_nop(device, cs);

         l2t_dst.offset.y = dst->offset.y + row;
         l2t_dst.offset.z = dst->offset.z + slice;
         l2t_src.slice_pitch = l2t_src.pitch * rows;

         if (!ac_sdma_use_unaligned_buffer_image_copy(&pdev->info, &l2t_src, &l2t_dst, new_extent.width,
                                                      new_extent.height, new_extent.depth))
            radv_sdma_copy_buffer_image(device, cs, &l2t_src, &l2t_dst, new_extent, true);
         else
            radv_sdma_copy_buffer_image_unaligned(device, cs, &l2t_src, &l2t_dst, new_extent, temp_bufimg_va, true);
         radv_sdma_emit_nop(device, cs);
      }
   }
}

bool
radv_sdma_supports_image(const struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image,
                         VkImageLayout image_layout, const VkImageSubresourceLayers *subresource, bool to_image)
{
   const struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const uint32_t queue_mask = radv_image_queue_family_mask(image, cmd_buffer->qf);

   /* Fallback to compute when SDMA doesn't support compression to avoid disabling DCC/HTILE for
    * concurrent images. Only GFX9 and older, Navi10 and GFX1013 are affected.
    */
   if (!pdev->info.sdma_supports_compression &&
       (radv_layout_dcc_compressed(device, image, subresource->mipLevel, image_layout, queue_mask) ||
        radv_layout_is_htile_compressed(device, image, subresource->mipLevel, image_layout, queue_mask)))
      return false;

   /* Only GFX11 can write the uncompressed DWORD to HTILE for depth-only or stencil-only and
    * without MSAA. Otherwise, fallback to compute.
    */
   if (pdev->info.gfx_level < GFX12 && to_image &&
       radv_layout_is_htile_compressed(device, image, subresource->mipLevel, image_layout, queue_mask) &&
       !radv_image_decompress_htile_on_image_stores(device, image))
      return false;

   if (radv_is_format_emulated(pdev, image->vk.format))
      return to_image ? false : true;

   if (!pdev->info.sdma_supports_sparse && (image->vk.create_flags & VK_IMAGE_CREATE_2_SPARSE_RESIDENCY_BIT_KHR))
      return false;

   if (image->vk.samples != VK_SAMPLE_COUNT_1_BIT)
      return false;

   return true;
}
