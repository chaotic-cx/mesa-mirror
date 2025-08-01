/*
 * Copyright © 2019 Red Hat.
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

#include "lvp_private.h"
#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_surface.h"
#include "pipe/p_state.h"
#include "frontend/winsys_handle.h"
#include "vk_android.h"

static VkResult
lvp_image_create(VkDevice _device,
                 const VkImageCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks* alloc,
                 VkImage *pImage)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   struct lvp_image *image;
   VkResult result = VK_SUCCESS;
   bool android_surface = false;
   const VkSubresourceLayout *layouts = NULL;
   uint64_t modifier;
   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);

#ifdef HAVE_LIBDRM
   unsigned num_layouts = 1;
   enum pipe_format pipe_format = lvp_vk_format_to_pipe_format(pCreateInfo->format);
   const VkImageDrmFormatModifierExplicitCreateInfoEXT *modinfo = (void*)vk_find_struct_const(pCreateInfo->pNext,
                                                                  IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT);

   if (modinfo && pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) {
      assert(modinfo->drmFormatModifier == DRM_FORMAT_MOD_LINEAR);
      assert(modinfo->drmFormatModifierPlaneCount == util_format_get_num_planes(pipe_format));
      num_layouts = modinfo->drmFormatModifierPlaneCount;
      layouts = modinfo->pPlaneLayouts;
   }

   /* planar not supported yet */
   assert(num_layouts == 1);
   if (num_layouts > 1) {
      mesa_loge("lavapipe: planar drm formats are not supported");
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   modifier = DRM_FORMAT_MOD_LINEAR;
#endif

   image = vk_image_create(&device->vk, pCreateInfo, alloc, sizeof(*image));
   if (image == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   image->alignment = 64;
   if (image->vk.create_flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT)
      image->alignment = 64 * 1024;

   image->plane_count = vk_format_get_plane_count(pCreateInfo->format);
   image->disjoint = image->plane_count > 1 &&
                     (pCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT);

   /* This section is removed by the optimizer for non-ANDROID builds */
   VkImageDrmFormatModifierExplicitCreateInfoEXT eci;
   VkSubresourceLayout a_plane_layouts[LVP_MAX_PLANE_COUNT];
   if (vk_image_is_android_native_buffer(&image->vk)) {
      result = vk_android_get_anb_layout(
         pCreateInfo, &eci, a_plane_layouts, LVP_MAX_PLANE_COUNT);
      if (result != VK_SUCCESS)
         goto fail;

      modifier = eci.drmFormatModifier;
      layouts = a_plane_layouts;
      android_surface = true;
   }

   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(pCreateInfo->format);
   for (unsigned p = 0; p < image->plane_count; p++) {
      struct pipe_resource template;
      VkFormat format = ycbcr_info ?
         ycbcr_info->planes[p].format : pCreateInfo->format;
      const uint8_t width_scale = ycbcr_info ?
         ycbcr_info->planes[p].denominator_scales[0] : 1;
      const uint8_t height_scale = ycbcr_info ?
         ycbcr_info->planes[p].denominator_scales[1] : 1;
      memset(&template, 0, sizeof(template));

      template.screen = device->pscreen;
      switch (pCreateInfo->imageType) {
      case VK_IMAGE_TYPE_1D:
         template.target = pCreateInfo->arrayLayers > 1 ? PIPE_TEXTURE_1D_ARRAY : PIPE_TEXTURE_1D;
         break;
      default:
      case VK_IMAGE_TYPE_2D:
         template.target = pCreateInfo->arrayLayers > 1 ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D;
         break;
      case VK_IMAGE_TYPE_3D:
         template.target = PIPE_TEXTURE_3D;
         break;
      }

      template.format = lvp_vk_format_to_pipe_format(format);

      bool is_ds = util_format_is_depth_or_stencil(template.format);

      if (pCreateInfo->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
         template.bind |= PIPE_BIND_RENDER_TARGET;
         /* sampler view is needed for resolve blits */
         if (pCreateInfo->samples > 1)
            template.bind |= PIPE_BIND_SAMPLER_VIEW;
      }

      if (pCreateInfo->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
         if (!is_ds)
            template.bind |= PIPE_BIND_RENDER_TARGET;
         else
            template.bind |= PIPE_BIND_DEPTH_STENCIL;
      }

      if (pCreateInfo->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
         template.bind |= PIPE_BIND_DEPTH_STENCIL;

      if (pCreateInfo->usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))
         template.bind |= PIPE_BIND_SAMPLER_VIEW;

      if (pCreateInfo->usage & (VK_IMAGE_USAGE_STORAGE_BIT |
                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))
         template.bind |= PIPE_BIND_SHADER_IMAGE;

      if (pCreateInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT)
         template.flags |= PIPE_RESOURCE_FLAG_SPARSE;

      template.width0 = pCreateInfo->extent.width / width_scale;
      template.height0 = pCreateInfo->extent.height / height_scale;
      template.depth0 = pCreateInfo->extent.depth;
      template.array_size = pCreateInfo->arrayLayers;
      template.last_level = pCreateInfo->mipLevels - 1;
      template.nr_samples = pCreateInfo->samples;
      template.nr_storage_samples = pCreateInfo->samples;

#ifdef HAVE_LIBDRM
      if (android_surface || (modinfo && pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT)) {
         struct winsys_handle whandle;
         whandle.type = WINSYS_HANDLE_TYPE_UNBACKED;
         whandle.layer = 0;
         whandle.plane = p;
         whandle.handle = 0;
         whandle.stride = layouts[p].rowPitch;
         whandle.array_stride = layouts[p].arrayPitch;
         whandle.image_stride = layouts[p].depthPitch;
         image->offset = layouts[p].offset;
         whandle.format = pCreateInfo->format;
         whandle.modifier = modifier;
         image->planes[p].bo = device->pscreen->resource_from_handle(device->pscreen,
                                                           &template,
                                                           &whandle,
                                                           PIPE_HANDLE_USAGE_EXPLICIT_FLUSH);
         image->planes[p].size = whandle.size;
      } else
#endif
      {
         image->planes[p].bo = device->pscreen->resource_create_unbacked(device->pscreen,
                                                               &template,
                                                               &image->planes[p].size);
      }
      if (!image->planes[p].bo) {
         result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }

      image->planes[p].size = align64(image->planes[p].size, image->alignment);

      image->size += image->planes[p].size;
   }

   /* This section is removed by the optimizer for non-ANDROID builds */
   if (vk_image_is_android_native_buffer(&image->vk)) {
      result = vk_android_import_anb(&device->vk, pCreateInfo, alloc,
                                     &image->vk);
      if (result != VK_SUCCESS) {
         mesa_logw("Failed to import memory");
         goto fail;
      }
   }

   *pImage = lvp_image_to_handle(image);

   return VK_SUCCESS;
fail:
   vk_image_destroy(&device->vk, alloc, &image->vk);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_CreateImage(VkDevice _device,
                const VkImageCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator,
                VkImage *pImage)
{
#if !DETECT_OS_ANDROID
   LVP_FROM_HANDLE(lvp_device, device, _device);
   const VkImageSwapchainCreateInfoKHR *swapchain_info =
      vk_find_struct_const(pCreateInfo->pNext, IMAGE_SWAPCHAIN_CREATE_INFO_KHR);
   if (swapchain_info && swapchain_info->swapchain != VK_NULL_HANDLE) {
      return wsi_common_create_swapchain_image(&device->physical_device->wsi_device,
                                               pCreateInfo,
                                               swapchain_info->swapchain,
                                               pImage);
   }
#endif
   return lvp_image_create(_device, pCreateInfo, pAllocator,
                           pImage);
}

VKAPI_ATTR void VKAPI_CALL
lvp_DestroyImage(VkDevice _device, VkImage _image,
                 const VkAllocationCallbacks *pAllocator)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_image, image, _image);

   if (!_image)
     return;
   for (unsigned p = 0; p < image->plane_count; p++)
      pipe_resource_reference(&image->planes[p].bo, NULL);
   vk_image_destroy(&device->vk, pAllocator, &image->vk);
}

#include "lvp_conv.h"
#include "util/u_sampler.h"
#include "util/u_inlines.h"

static inline char conv_depth_swiz(char swiz) {
   switch (swiz) {
   case PIPE_SWIZZLE_Y:
   case PIPE_SWIZZLE_Z:
      return PIPE_SWIZZLE_0;
   case PIPE_SWIZZLE_W:
      return PIPE_SWIZZLE_1;
   default:
      return swiz;
   }
}

static struct pipe_sampler_view *
lvp_create_samplerview(struct pipe_context *pctx, struct lvp_image_view *iv, VkFormat plane_format, unsigned image_plane)
{
   if (!iv)
      return NULL;

   struct pipe_sampler_view templ;
   enum pipe_format pformat;
   if (iv->vk.aspects == VK_IMAGE_ASPECT_DEPTH_BIT)
      pformat = lvp_vk_format_to_pipe_format(plane_format);
   else if (iv->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      pformat = util_format_stencil_only(lvp_vk_format_to_pipe_format(plane_format));
   else
      pformat = lvp_vk_format_to_pipe_format(plane_format);
   u_sampler_view_default_template(&templ,
                                   iv->image->planes[image_plane].bo,
                                   pformat);
   if (iv->vk.view_type == VK_IMAGE_VIEW_TYPE_1D)
      templ.target = PIPE_TEXTURE_1D;
   if (iv->vk.view_type == VK_IMAGE_VIEW_TYPE_2D)
      templ.target = PIPE_TEXTURE_2D;
   if (iv->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE)
      templ.target = PIPE_TEXTURE_CUBE;
   if (iv->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
      templ.target = PIPE_TEXTURE_CUBE_ARRAY;
   templ.u.tex.first_layer = iv->vk.base_array_layer;
   templ.u.tex.last_layer = iv->vk.base_array_layer + iv->vk.layer_count - 1;
   templ.u.tex.first_level = iv->vk.base_mip_level;
   templ.u.tex.last_level = iv->vk.base_mip_level + iv->vk.level_count - 1;
   templ.swizzle_r = vk_conv_swizzle(iv->vk.swizzle.r, PIPE_SWIZZLE_X);
   templ.swizzle_g = vk_conv_swizzle(iv->vk.swizzle.g, PIPE_SWIZZLE_Y);
   templ.swizzle_b = vk_conv_swizzle(iv->vk.swizzle.b, PIPE_SWIZZLE_Z);
   templ.swizzle_a = vk_conv_swizzle(iv->vk.swizzle.a, PIPE_SWIZZLE_W);

   /* depth stencil swizzles need special handling to pass VK CTS
    * but also for zink GL tests.
    * piping A swizzle into R fixes GL_ALPHA depth texture mode
    * only swizzling from R/0/1 (for alpha) fixes VK CTS tests
    * and a bunch of zink tests.
   */
   if (iv->vk.aspects == VK_IMAGE_ASPECT_DEPTH_BIT ||
       iv->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT) {
      templ.swizzle_r = conv_depth_swiz(templ.swizzle_r);
      templ.swizzle_g = conv_depth_swiz(templ.swizzle_g);
      templ.swizzle_b = conv_depth_swiz(templ.swizzle_b);
      templ.swizzle_a = conv_depth_swiz(templ.swizzle_a);
   }

   return pctx->create_sampler_view(pctx, iv->image->planes[image_plane].bo, &templ);
}

static struct pipe_image_view
lvp_create_imageview(const struct lvp_image_view *iv, VkFormat plane_format, unsigned image_plane)
{
   struct pipe_image_view view = {0};
   if (!iv)
      return view;

   view.resource = iv->image->planes[image_plane].bo;
   if (iv->vk.aspects == VK_IMAGE_ASPECT_DEPTH_BIT)
      view.format = lvp_vk_format_to_pipe_format(plane_format);
   else if (iv->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      view.format = util_format_stencil_only(lvp_vk_format_to_pipe_format(plane_format));
   else
      view.format = lvp_vk_format_to_pipe_format(plane_format);

   if (iv->vk.view_type == VK_IMAGE_VIEW_TYPE_3D) {
      view.u.tex.first_layer = iv->vk.storage.z_slice_offset;
      view.u.tex.last_layer = view.u.tex.first_layer + iv->vk.storage.z_slice_count - 1;
   } else {
      view.u.tex.first_layer = iv->vk.base_array_layer,
      view.u.tex.last_layer = iv->vk.base_array_layer + iv->vk.layer_count - 1;

      if (view.resource->target == PIPE_TEXTURE_3D)
         view.u.tex.is_2d_view_of_3d = true;
   }

   if (iv->vk.view_type == VK_IMAGE_VIEW_TYPE_1D ||
       iv->vk.view_type == VK_IMAGE_VIEW_TYPE_2D) {
      /*
       * There's no target field in pipe_image_view, but
       * there's a single_layer_view which the mesa state tracker
       * uses for a similar purpose, although not exactly the same,
       * here we just use it to indicate the view is of a non-array
       * type.
       * Note that the layered-ness must match between shader dcl
       * and view (but not between view and resource).
       * We ignore VK_IMAGE_VIEW_TYPE_CUBE here, should be fine
       * since there's no difference in accessing cube and cube arrays
       * (as layer and face combine into one var), and for size queries
       * we fix up targets separately (always using array types).
       */
      assert(view.u.tex.first_layer == view.u.tex.last_layer);
      view.u.tex.single_layer_view = 1;
   }

   view.u.tex.level = iv->vk.base_mip_level;
   return view;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_CreateImageView(VkDevice _device,
                    const VkImageViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkImageView *pView)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_image, image, pCreateInfo->image);
   struct lvp_image_view *view;

   view = vk_image_view_create(&device->vk, false, pCreateInfo,
                               pAllocator, sizeof(*view));
   if (view == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   view->pformat = lvp_vk_format_to_pipe_format(view->vk.format);
   view->image = image;
   view->surface.texture = NULL;

   if (image->vk.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT |
                            VK_IMAGE_ASPECT_STENCIL_BIT)) {
      assert(image->plane_count == 1);
      assert(lvp_image_aspects_to_plane(image, view->vk.aspects) == 0);
      view->plane_count = 1;
      view->planes[0].image_plane = 0;
   } else {
      /* For other formats, retrieve the plane count from the aspect mask
       * and then walk through the aspect mask to map each image plane
       * to its corresponding view plane
       */
      assert(util_bitcount(view->vk.aspects) ==
             vk_format_get_plane_count(view->vk.format));
      view->plane_count = 0;
      u_foreach_bit(aspect_bit, view->vk.aspects) {
         uint8_t image_plane = lvp_image_aspects_to_plane(image, 1u << aspect_bit);
         view->planes[view->plane_count++].image_plane = image_plane;
      }
   }

   simple_mtx_lock(&device->queue.lock);

   for (unsigned view_plane = 0; view_plane < view->plane_count; view_plane++) {
      const uint8_t image_plane = view->planes[view_plane].image_plane;
      const struct vk_format_ycbcr_info *ycbcr_info =
         vk_format_get_ycbcr_info(view->vk.format);
      assert(ycbcr_info || view_plane == 0);
      VkFormat plane_format = ycbcr_info ?
         ycbcr_info->planes[view_plane].format : view->vk.format;

      if (image->planes[image_plane].bo->bind & PIPE_BIND_SHADER_IMAGE) {
         view->planes[view_plane].iv = lvp_create_imageview(view, plane_format, image_plane);
         view->planes[view_plane].image_handle = (void *)(uintptr_t)device->queue.ctx->create_image_handle(device->queue.ctx, &view->planes[view_plane].iv);
      }

      if (image->planes[image_plane].bo->bind & PIPE_BIND_SAMPLER_VIEW) {
         view->planes[view_plane].sv = lvp_create_samplerview(device->queue.ctx, view, plane_format, image_plane);
         view->planes[view_plane].texture_handle = (void *)(uintptr_t)device->queue.ctx->create_texture_handle(device->queue.ctx, view->planes[view_plane].sv, NULL);
      }
   }

   simple_mtx_unlock(&device->queue.lock);

   *pView = lvp_image_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
lvp_DestroyImageView(VkDevice _device, VkImageView _iview,
                     const VkAllocationCallbacks *pAllocator)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_image_view, iview, _iview);

   if (!_iview)
     return;

   simple_mtx_lock(&device->queue.lock);

   for (uint8_t plane = 0; plane < iview->plane_count; plane++) {
      device->queue.ctx->delete_image_handle(device->queue.ctx, (uint64_t)(uintptr_t)iview->planes[plane].image_handle);

      pipe_sampler_view_reference(&iview->planes[plane].sv, NULL);
      device->queue.ctx->delete_texture_handle(device->queue.ctx, (uint64_t)(uintptr_t)iview->planes[plane].texture_handle);
   }
   simple_mtx_unlock(&device->queue.lock);

   vk_image_view_destroy(&device->vk, pAllocator, &iview->vk);
}

VKAPI_ATTR void VKAPI_CALL lvp_GetImageSubresourceLayout(
    VkDevice                                    _device,
    VkImage                                     _image,
    const VkImageSubresource*                   pSubresource,
    VkSubresourceLayout*                        pLayout)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_image, image, _image);
   uint64_t value;

   const uint8_t p = lvp_image_aspects_to_plane(image, pSubresource->aspectMask);
   const struct lvp_image_plane *plane = &image->planes[p];

   device->pscreen->resource_get_param(device->pscreen,
                                       NULL,
                                       plane->bo,
                                       0,
                                       pSubresource->arrayLayer,
                                       pSubresource->mipLevel,
                                       PIPE_RESOURCE_PARAM_STRIDE,
                                       0, &value);

   pLayout->rowPitch = value;

   device->pscreen->resource_get_param(device->pscreen,
                                       NULL,
                                       plane->bo,
                                       0,
                                       pSubresource->arrayLayer,
                                       pSubresource->mipLevel,
                                       PIPE_RESOURCE_PARAM_OFFSET,
                                       0, &value);

   pLayout->offset = value;

   device->pscreen->resource_get_param(device->pscreen,
                                       NULL,
                                       plane->bo,
                                       0,
                                       pSubresource->arrayLayer,
                                       pSubresource->mipLevel,
                                       PIPE_RESOURCE_PARAM_LAYER_STRIDE,
                                       0, &value);

   if (plane->bo->target == PIPE_TEXTURE_3D) {
      pLayout->depthPitch = value;
      pLayout->arrayPitch = 0;
   } else {
      pLayout->depthPitch = 0;
      pLayout->arrayPitch = value;
   }
   pLayout->offset += plane->plane_offset;
   pLayout->size = plane->size;
}

VKAPI_ATTR void VKAPI_CALL lvp_GetImageSubresourceLayout2KHR(
    VkDevice                       _device,
    VkImage                        _image,
    const VkImageSubresource2KHR*  pSubresource,
    VkSubresourceLayout2KHR*       pLayout)
{
   lvp_GetImageSubresourceLayout(_device, _image, &pSubresource->imageSubresource, &pLayout->subresourceLayout);
   VkSubresourceHostMemcpySizeEXT *size = vk_find_struct(pLayout, SUBRESOURCE_HOST_MEMCPY_SIZE_EXT);
   if (size)
      size->size = pLayout->subresourceLayout.size;
}

VKAPI_ATTR void VKAPI_CALL lvp_GetDeviceImageSubresourceLayoutKHR(
    VkDevice                                    _device,
    const VkDeviceImageSubresourceInfoKHR*      pInfo,
    VkSubresourceLayout2KHR*                    pLayout)
{
   VkImage image;
   /* technically supposed to be able to do this without creating an image, but that's harder */
   if (lvp_image_create(_device, pInfo->pCreateInfo, NULL, &image) != VK_SUCCESS)
      return;
   lvp_GetImageSubresourceLayout2KHR(_device, image, pInfo->pSubresource, pLayout);
   lvp_DestroyImage(_device, image, NULL);
}

VKAPI_ATTR VkResult VKAPI_CALL lvp_CreateBuffer(
    VkDevice                                    _device,
    const VkBufferCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBuffer*                                   pBuffer)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   struct lvp_buffer *buffer;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);

   /* gallium has max 32-bit buffer sizes */
   if (pCreateInfo->size > UINT32_MAX)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   buffer = vk_buffer_create(&device->vk, pCreateInfo,
                             pAllocator, sizeof(*buffer));
   if (buffer == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   {
      struct pipe_resource template;
      memset(&template, 0, sizeof(struct pipe_resource));

      if (pCreateInfo->usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
         template.bind |= PIPE_BIND_CONSTANT_BUFFER;

      template.screen = device->pscreen;
      template.target = PIPE_BUFFER;
      template.format = PIPE_FORMAT_R8_UNORM;
      template.width0 = buffer->vk.size;
      template.height0 = 1;
      template.depth0 = 1;
      template.array_size = 1;
      if (buffer->vk.usage & VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT_KHR)
         template.bind |= PIPE_BIND_SAMPLER_VIEW;
      if (buffer->vk.usage & VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR)
         template.bind |= PIPE_BIND_SHADER_BUFFER;
      if (buffer->vk.usage & VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT_KHR)
         template.bind |= PIPE_BIND_SHADER_IMAGE;
      template.flags = PIPE_RESOURCE_FLAG_DONT_OVER_ALLOCATE;
      if (pCreateInfo->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT)
         template.flags |= PIPE_RESOURCE_FLAG_SPARSE;
      buffer->bo = device->pscreen->resource_create_unbacked(device->pscreen,
                                                             &template,
                                                             &buffer->total_size);
      if (!buffer->bo) {
         vk_free2(&device->vk.alloc, pAllocator, buffer);
         return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      if (pCreateInfo->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) {
         buffer->map = device->queue.ctx->buffer_map(device->queue.ctx, buffer->bo, 0,
                                                     PIPE_MAP_READ | PIPE_MAP_WRITE | PIPE_MAP_PERSISTENT,
                                                     &(struct pipe_box){ 0 }, &buffer->transfer);

         buffer->vk.device_address = (VkDeviceAddress)(uintptr_t)buffer->map;
      }
   }
   *pBuffer = lvp_buffer_to_handle(buffer);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL lvp_DestroyBuffer(
    VkDevice                                    _device,
    VkBuffer                                    _buffer,
    const VkAllocationCallbacks*                pAllocator)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_buffer, buffer, _buffer);

   if (!_buffer)
     return;

   if (buffer->map) {
      simple_mtx_lock(&device->bda_lock);
      struct hash_entry *he = _mesa_hash_table_search(&device->bda, buffer->map);
      if (he)
         _mesa_hash_table_remove(&device->bda, he);
      simple_mtx_unlock(&device->bda_lock);

      if (buffer->bo->flags & PIPE_RESOURCE_FLAG_SPARSE)
         device->queue.ctx->buffer_unmap(device->queue.ctx, buffer->transfer);
   }
   pipe_resource_reference(&buffer->bo, NULL);
   vk_buffer_destroy(&device->vk, pAllocator, &buffer->vk);
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL lvp_GetBufferDeviceAddress(
   VkDevice                                    _device,
   const VkBufferDeviceAddressInfo*            pInfo)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_buffer, buffer, pInfo->buffer);
   simple_mtx_lock(&device->bda_lock);
   _mesa_hash_table_insert(&device->bda, buffer->map, buffer);
   simple_mtx_unlock(&device->bda_lock);

   return (VkDeviceAddress)(uintptr_t)buffer->map;
}

VKAPI_ATTR uint64_t VKAPI_CALL lvp_GetBufferOpaqueCaptureAddress(
    VkDevice                                    device,
    const VkBufferDeviceAddressInfo*            pInfo)
{
   return 0;
}

VKAPI_ATTR uint64_t VKAPI_CALL lvp_GetDeviceMemoryOpaqueCaptureAddress(
    VkDevice                                    device,
    const VkDeviceMemoryOpaqueCaptureAddressInfo* pInfo)
{
   return 0;
}

static struct pipe_sampler_view *
lvp_create_samplerview_buffer(struct pipe_context *pctx, struct lvp_buffer_view *bv)
{
   if (!bv)
      return NULL;

   struct pipe_resource *bo = ((struct lvp_buffer *)bv->vk.buffer)->bo;
   struct pipe_sampler_view templ;
   memset(&templ, 0, sizeof(templ));
   templ.target = PIPE_BUFFER;
   templ.swizzle_r = PIPE_SWIZZLE_X;
   templ.swizzle_g = PIPE_SWIZZLE_Y;
   templ.swizzle_b = PIPE_SWIZZLE_Z;
   templ.swizzle_a = PIPE_SWIZZLE_W;
   templ.format = bv->pformat;
   templ.u.buf.offset = bv->vk.offset;
   templ.u.buf.size = bv->vk.range;
   templ.texture = bo;
   templ.context = pctx;
   return pctx->create_sampler_view(pctx, bo, &templ);
}

static struct pipe_image_view
lvp_create_imageview_buffer(const struct lvp_buffer_view *bv)
{
   struct pipe_image_view view = {0};
   if (!bv)
      return view;
   view.resource = ((struct lvp_buffer *)bv->vk.buffer)->bo;
   view.format = bv->pformat;
   view.u.buf.offset = bv->vk.offset;
   view.u.buf.size = bv->vk.range;
   return view;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_CreateBufferView(VkDevice _device,
                     const VkBufferViewCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkBufferView *pView)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_buffer, buffer, pCreateInfo->buffer);
   struct lvp_buffer_view *view;

   view = vk_buffer_view_create(&device->vk,
                                pCreateInfo,
                                pAllocator,
                                sizeof(*view));
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   view->pformat = lvp_vk_format_to_pipe_format(pCreateInfo->format);

   simple_mtx_lock(&device->queue.lock);

   if (buffer->bo->bind & PIPE_BIND_SAMPLER_VIEW) {
      view->sv = lvp_create_samplerview_buffer(device->queue.ctx, view);
      view->texture_handle = (void *)(uintptr_t)device->queue.ctx->create_texture_handle(device->queue.ctx, view->sv, NULL);
   }

   if (buffer->bo->bind & PIPE_BIND_SHADER_IMAGE) {
      view->iv = lvp_create_imageview_buffer(view);
      view->image_handle = (void *)(uintptr_t)device->queue.ctx->create_image_handle(device->queue.ctx, &view->iv);
   }

   simple_mtx_unlock(&device->queue.lock);

   *pView = lvp_buffer_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
lvp_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
                      const VkAllocationCallbacks *pAllocator)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_buffer_view, view, bufferView);

   if (!bufferView)
     return;

   simple_mtx_lock(&device->queue.lock);

   pipe_sampler_view_reference(&view->sv, NULL);
   device->queue.ctx->delete_texture_handle(device->queue.ctx, (uint64_t)(uintptr_t)view->texture_handle);

   device->queue.ctx->delete_image_handle(device->queue.ctx, (uint64_t)(uintptr_t)view->image_handle);

   simple_mtx_unlock(&device->queue.lock);

   vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_CopyMemoryToImageEXT(VkDevice _device, const VkCopyMemoryToImageInfoEXT *pCopyMemoryToImageInfo)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_image, image, pCopyMemoryToImageInfo->dstImage);
   for (unsigned i = 0; i < pCopyMemoryToImageInfo->regionCount; i++) {
      const VkMemoryToImageCopyEXT *copy = &pCopyMemoryToImageInfo->pRegions[i];
      const VkImageAspectFlagBits aspects = copy->imageSubresource.aspectMask;
      uint8_t plane = lvp_image_aspects_to_plane(image, aspects);
      struct pipe_box box = {
         .x = copy->imageOffset.x,
         .y = copy->imageOffset.y,
         .width = copy->imageExtent.width,
         .height = copy->imageExtent.height,
         .depth = 1,
      };
      switch (image->planes[plane].bo->target) {
      case PIPE_TEXTURE_CUBE:
      case PIPE_TEXTURE_CUBE_ARRAY:
      case PIPE_TEXTURE_2D_ARRAY:
      case PIPE_TEXTURE_1D_ARRAY:
         /* these use layer */
         box.z = copy->imageSubresource.baseArrayLayer;
         box.depth = vk_image_subresource_layer_count(&image->vk, &copy->imageSubresource);
         break;
      case PIPE_TEXTURE_3D:
         /* this uses depth */
         box.z = copy->imageOffset.z;
         box.depth = copy->imageExtent.depth;
         break;
      default:
         break;
      }

      unsigned stride = util_format_get_stride(image->planes[plane].bo->format, copy->memoryRowLength ? copy->memoryRowLength : box.width);
      unsigned layer_stride = util_format_get_2d_size(image->planes[plane].bo->format, stride, copy->memoryImageHeight ? copy->memoryImageHeight : box.height);
      device->queue.ctx->texture_subdata(device->queue.ctx, image->planes[plane].bo, copy->imageSubresource.mipLevel, 0,
                                         &box, copy->pHostPointer, stride, layer_stride);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_CopyImageToMemoryEXT(VkDevice _device, const VkCopyImageToMemoryInfoEXT *pCopyImageToMemoryInfo)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_image, image, pCopyImageToMemoryInfo->srcImage);

   for (unsigned i = 0; i < pCopyImageToMemoryInfo->regionCount; i++) {
      const VkImageToMemoryCopyEXT *copy = &pCopyImageToMemoryInfo->pRegions[i];

      const VkImageAspectFlagBits aspects = copy->imageSubresource.aspectMask;
      uint8_t plane = lvp_image_aspects_to_plane(image, aspects);

      struct pipe_box box = {
         .x = copy->imageOffset.x,
         .y = copy->imageOffset.y,
         .width = copy->imageExtent.width,
         .height = copy->imageExtent.height,
         .depth = 1,
      };
      switch (image->planes[plane].bo->target) {
      case PIPE_TEXTURE_CUBE:
      case PIPE_TEXTURE_CUBE_ARRAY:
      case PIPE_TEXTURE_2D_ARRAY:
      case PIPE_TEXTURE_1D_ARRAY:
         /* these use layer */
         box.z = copy->imageSubresource.baseArrayLayer;
         box.depth = vk_image_subresource_layer_count(&image->vk, &copy->imageSubresource);
         break;
      case PIPE_TEXTURE_3D:
         /* this uses depth */
         box.z = copy->imageOffset.z;
         box.depth = copy->imageExtent.depth;
         break;
      default:
         break;
      }
      struct pipe_transfer *xfer;
      uint8_t *data = device->queue.ctx->texture_map(device->queue.ctx, image->planes[plane].bo, copy->imageSubresource.mipLevel,
                                                     PIPE_MAP_READ | PIPE_MAP_UNSYNCHRONIZED | PIPE_MAP_THREAD_SAFE, &box, &xfer);
      if (!data)
         return VK_ERROR_MEMORY_MAP_FAILED;

      unsigned stride = util_format_get_stride(image->planes[plane].bo->format, copy->memoryRowLength ? copy->memoryRowLength : box.width);
      unsigned layer_stride = util_format_get_2d_size(image->planes[plane].bo->format, stride, copy->memoryImageHeight ? copy->memoryImageHeight : box.height);
      util_copy_box(copy->pHostPointer, image->planes[plane].bo->format, stride, layer_stride,
                    /* offsets are all zero because texture_map handles the offset */
                    0, 0, 0, box.width, box.height, box.depth, data, xfer->stride, xfer->layer_stride, 0, 0, 0);
      pipe_texture_unmap(device->queue.ctx, xfer);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_CopyImageToImageEXT(VkDevice _device, const VkCopyImageToImageInfoEXT *pCopyImageToImageInfo)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_image, src_image, pCopyImageToImageInfo->srcImage);
   LVP_FROM_HANDLE(lvp_image, dst_image, pCopyImageToImageInfo->dstImage);

   /* basically the same as handle_copy_image() */
   for (unsigned i = 0; i < pCopyImageToImageInfo->regionCount; i++) {

      const VkImageAspectFlagBits src_aspects = pCopyImageToImageInfo->pRegions[i].srcSubresource.aspectMask;
      uint8_t src_plane = lvp_image_aspects_to_plane(src_image, src_aspects);
      const VkImageAspectFlagBits dst_aspects = pCopyImageToImageInfo->pRegions[i].dstSubresource.aspectMask;
      uint8_t dst_plane = lvp_image_aspects_to_plane(dst_image, dst_aspects);

      struct pipe_box src_box;
      src_box.x = pCopyImageToImageInfo->pRegions[i].srcOffset.x;
      src_box.y = pCopyImageToImageInfo->pRegions[i].srcOffset.y;
      src_box.width = pCopyImageToImageInfo->pRegions[i].extent.width;
      src_box.height = pCopyImageToImageInfo->pRegions[i].extent.height;
      if (src_image->planes[src_plane].bo->target == PIPE_TEXTURE_3D) {
         src_box.depth = pCopyImageToImageInfo->pRegions[i].extent.depth;
         src_box.z = pCopyImageToImageInfo->pRegions[i].srcOffset.z;
      } else {
         src_box.depth = vk_image_subresource_layer_count(&src_image->vk, &pCopyImageToImageInfo->pRegions[i].srcSubresource);
         src_box.z = pCopyImageToImageInfo->pRegions[i].srcSubresource.baseArrayLayer;
      }

      unsigned dstz = dst_image->planes[dst_plane].bo->target == PIPE_TEXTURE_3D ?
                      pCopyImageToImageInfo->pRegions[i].dstOffset.z :
                      pCopyImageToImageInfo->pRegions[i].dstSubresource.baseArrayLayer;
      device->queue.ctx->resource_copy_region(device->queue.ctx, dst_image->planes[dst_plane].bo,
                                              pCopyImageToImageInfo->pRegions[i].dstSubresource.mipLevel,
                                              pCopyImageToImageInfo->pRegions[i].dstOffset.x,
                                              pCopyImageToImageInfo->pRegions[i].dstOffset.y,
                                              dstz,
                                              src_image->planes[src_plane].bo,
                                              pCopyImageToImageInfo->pRegions[i].srcSubresource.mipLevel,
                                              &src_box);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_TransitionImageLayoutEXT(VkDevice device, uint32_t transitionCount, const VkHostImageLayoutTransitionInfoEXT *pTransitions)
{
   /* no-op */
   return VK_SUCCESS;
}

VkResult
lvp_buffer_bind_sparse(struct lvp_device *device,
                       struct lvp_queue *queue,
                       VkSparseBufferMemoryBindInfo *bind)
{
   LVP_FROM_HANDLE(lvp_buffer, buffer, bind->buffer);

   for (uint32_t i = 0; i < bind->bindCount; i++) {
      LVP_FROM_HANDLE(lvp_device_memory, mem, bind->pBinds[i].memory);
      device->pscreen->resource_bind_backing(device->pscreen,
                                             buffer->bo,
                                             mem ? mem->pmem : NULL,
                                             bind->pBinds[i].memoryOffset,
                                             bind->pBinds[i].size,
                                             bind->pBinds[i].resourceOffset);
   }

   return VK_SUCCESS;
}

VkResult
lvp_image_bind_opaque_sparse(struct lvp_device *device,
                             struct lvp_queue *queue,
                             VkSparseImageOpaqueMemoryBindInfo *bind_info)
{
   LVP_FROM_HANDLE(lvp_image, image, bind_info->image);

   for (uint32_t i = 0; i < bind_info->bindCount; i++) {
      const VkSparseMemoryBind *bind = &bind_info->pBinds[i];
      LVP_FROM_HANDLE(lvp_device_memory, mem, bind->memory);

      uint32_t plane_index;
      uint32_t offset;
      if (bind->resourceOffset < image->planes[0].size) {
         plane_index = 0;
         offset = bind->resourceOffset;
      } else if (bind->resourceOffset < image->planes[0].size + image->planes[1].size) {
         plane_index = 1;
         offset = bind->resourceOffset - image->planes[0].size;
      } else {
         plane_index = 2;
         offset = bind->resourceOffset - image->planes[0].size - image->planes[1].size;
      }

      device->pscreen->resource_bind_backing(device->pscreen,
                                             image->planes[plane_index].bo,
                                             mem ? mem->pmem : NULL,
                                             bind->memoryOffset,
                                             bind->size,
                                             offset);
   }

   return VK_SUCCESS;
}

VkResult
lvp_image_bind_sparse(struct lvp_device *device,
                      struct lvp_queue *queue,
                      VkSparseImageMemoryBindInfo *bind_info)
{
   LVP_FROM_HANDLE(lvp_image, image, bind_info->image);

   enum pipe_format format = vk_format_to_pipe_format(image->vk.format);

   for (uint32_t i = 0; i < bind_info->bindCount; i++) {
      const VkSparseImageMemoryBind *bind = &bind_info->pBinds[i];
      LVP_FROM_HANDLE(lvp_device_memory, mem, bind->memory);

      uint8_t plane = lvp_image_aspects_to_plane(image, bind->subresource.aspectMask);

      uint32_t depth = 1;
      uint32_t z = 0;
      uint32_t dimensions = 2;
      switch (image->planes[plane].bo->target) {
      case PIPE_TEXTURE_CUBE:
      case PIPE_TEXTURE_CUBE_ARRAY:
      case PIPE_TEXTURE_2D_ARRAY:
      case PIPE_TEXTURE_1D_ARRAY:
         /* these use layer */
         z = bind->subresource.arrayLayer;
         break;
      case PIPE_TEXTURE_3D:
         /* this uses depth */
         z = bind->offset.z;
         depth = bind->extent.depth;
         dimensions = 3;
         break;
      default:
         break;
      }

      uint32_t sparse_tile_size[3] = {
         util_format_get_tilesize(format, dimensions, image->vk.samples, 0),
         util_format_get_tilesize(format, dimensions, image->vk.samples, 1),
         util_format_get_tilesize(format, dimensions, image->vk.samples, 2),
      };

      uint32_t sparse_block_base[3] = {
         bind->offset.x / (sparse_tile_size[0] * util_format_get_blockwidth(format)),
         bind->offset.y / (sparse_tile_size[1] * util_format_get_blockheight(format)),
         z / (sparse_tile_size[2] * util_format_get_blockdepth(format)),
      };

      uint32_t sparse_block_counts[3] = {
         DIV_ROUND_UP(bind->extent.width, sparse_tile_size[0] * util_format_get_blockwidth(format)),
         DIV_ROUND_UP(bind->extent.height, sparse_tile_size[1] * util_format_get_blockheight(format)),
         DIV_ROUND_UP(depth, sparse_tile_size[2] * util_format_get_blockdepth(format)),
      };

      uint32_t sparse_block_count = sparse_block_counts[0] * sparse_block_counts[1] * sparse_block_counts[2];

      for (uint32_t block = 0; block < sparse_block_count; block++) {
         uint32_t start_x = (sparse_block_base[0] + block % sparse_block_counts[0]) * sparse_tile_size[0];
         uint32_t start_y = (sparse_block_base[1] + (block / sparse_block_counts[0]) % sparse_block_counts[1]) *
                            sparse_tile_size[1];
         uint32_t start_z = (sparse_block_base[2] + (block / sparse_block_counts[0] / sparse_block_counts[1]) % sparse_block_counts[2]) *
                            sparse_tile_size[2];

         uint64_t offset = llvmpipe_get_texel_offset(image->planes[plane].bo, bind->subresource.mipLevel, start_x, start_y, start_z);
         device->pscreen->resource_bind_backing(device->pscreen,
                                                image->planes[plane].bo,
                                                mem ? mem->pmem : NULL,
                                                bind->memoryOffset + block * 64 * 1024,
                                                64 * 1024,
                                                offset);
      }
   }

   return VK_SUCCESS;
}
