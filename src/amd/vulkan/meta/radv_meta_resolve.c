/*
 * Copyright © 2016 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <stdbool.h>

#include "nir/radv_meta_nir.h"
#include "radv_entrypoints.h"
#include "radv_meta.h"
#include "vk_format.h"

struct radv_resolve_key {
   enum radv_meta_object_key_type type;
   uint32_t fs_key;
};

static VkResult
get_pipeline(struct radv_device *device, unsigned fs_key, VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   const VkFormat format = radv_fs_key_format_exemplars[fs_key];
   struct radv_resolve_key key;
   VkResult result;

   result = radv_meta_get_noop_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_RESOLVE_HW;
   key.fs_key = fs_key;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *vs_module = radv_meta_nir_build_vs_generate_vertices(device);
   nir_shader *fs_module = radv_meta_nir_build_resolve_fs(device);

   const VkGraphicsPipelineCreateInfoRADV radv_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO_RADV,
      .custom_blend_mode = V_028808_CB_RESOLVE,
   };

   const VkGraphicsPipelineCreateInfo pipeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &radv_info,
      .stageCount = 2,
      .pStages =
         (VkPipelineShaderStageCreateInfo[]){
            {
               .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .stage = VK_SHADER_STAGE_VERTEX_BIT,
               .module = vk_shader_module_handle_from_nir(vs_module),
               .pName = "main",
            },
            {
               .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
               .module = vk_shader_module_handle_from_nir(fs_module),
               .pName = "main",
            },
         },
      .pVertexInputState =
         &(VkPipelineVertexInputStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 0,
            .vertexAttributeDescriptionCount = 0,
         },
      .pInputAssemblyState =
         &(VkPipelineInputAssemblyStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA,
            .primitiveRestartEnable = false,
         },
      .pViewportState =
         &(VkPipelineViewportStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
         },
      .pRasterizationState =
         &(VkPipelineRasterizationStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = false,
            .rasterizerDiscardEnable = false,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
         },
      .pMultisampleState =
         &(VkPipelineMultisampleStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = 1,
            .sampleShadingEnable = false,
            .pSampleMask = NULL,
            .alphaToCoverageEnable = false,
            .alphaToOneEnable = false,
         },
      .pColorBlendState =
         &(VkPipelineColorBlendStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = false,
            .attachmentCount = 2,
            .pAttachments =
               (VkPipelineColorBlendAttachmentState[]){
                  {
                     .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                       VK_COLOR_COMPONENT_A_BIT,
                  },
                  {
                     .colorWriteMask = 0,

                  }},
         },
      .pDynamicState =
         &(VkPipelineDynamicStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 2,
            .pDynamicStates =
               (VkDynamicState[]){
                  VK_DYNAMIC_STATE_VIEWPORT,
                  VK_DYNAMIC_STATE_SCISSOR,
               },
         },
      .layout = *layout_out,
   };

   struct vk_meta_rendering_info render = {
      .color_attachment_count = 2,
      .color_attachment_formats = {format, format},
   };

   result = vk_meta_create_graphics_pipeline(&device->vk, &device->meta_state.device, &pipeline_create_info, &render,
                                             &key, sizeof(key), pipeline_out);

   ralloc_free(vs_module);
   ralloc_free(fs_module);
   return result;
}

static void
emit_resolve(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *src_image, const struct radv_image *dst_image,
             VkFormat vk_format)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   VkCommandBuffer cmd_buffer_h = radv_cmd_buffer_to_handle(cmd_buffer);
   unsigned fs_key = radv_format_meta_fs_key(device, vk_format);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result = get_pipeline(device, fs_key, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   cmd_buffer->state.flush_bits |= radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, 0, src_image, NULL) |
                                   radv_dst_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT, 0, src_image, NULL);

   radv_CmdBindPipeline(cmd_buffer_h, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

   radv_CmdDraw(cmd_buffer_h, 3, 1, 0, 0);
   cmd_buffer->state.flush_bits |= radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, 0, dst_image, NULL);
}

enum radv_resolve_method {
   RESOLVE_HW,
   RESOLVE_COMPUTE,
   RESOLVE_FRAGMENT,
};

static bool
image_hw_resolve_compat(const struct radv_device *device, struct radv_image *src_image, struct radv_image *dst_image)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   if (pdev->info.gfx_level >= GFX9) {
      return dst_image->planes[0].surface.u.gfx9.swizzle_mode == src_image->planes[0].surface.u.gfx9.swizzle_mode;
   } else {
      return dst_image->planes[0].surface.micro_tile_mode == src_image->planes[0].surface.micro_tile_mode;
   }
}

static void
radv_pick_resolve_method_images(struct radv_device *device, struct radv_image *src_image, VkFormat src_format,
                                struct radv_image *dst_image, unsigned dst_level, VkImageLayout dst_image_layout,
                                struct radv_cmd_buffer *cmd_buffer, enum radv_resolve_method *method)

{
   uint32_t queue_mask = radv_image_queue_family_mask(dst_image, cmd_buffer->qf, cmd_buffer->qf);

   if (vk_format_is_color(src_format)) {
      /* Using the fragment resolve path is currently a hint to
       * avoid decompressing DCC for partial resolves and
       * re-initialize it after resolving using compute.
       * TODO: Add support for layered and int to the fragment path.
       */
      if (radv_layout_dcc_compressed(device, dst_image, dst_level, dst_image_layout, queue_mask)) {
         *method = RESOLVE_FRAGMENT;
      } else if (!image_hw_resolve_compat(device, src_image, dst_image)) {
         /* The micro tile mode only needs to match for the HW
          * resolve path which is the default path for non-DCC
          * resolves.
          */
         *method = RESOLVE_COMPUTE;
      }

      if (src_format == VK_FORMAT_R16G16_UNORM || src_format == VK_FORMAT_R16G16_SNORM)
         *method = RESOLVE_COMPUTE;
      else if (vk_format_is_int(src_format))
         *method = RESOLVE_COMPUTE;
      else if (src_image->vk.array_layers > 1 || dst_image->vk.array_layers > 1)
         *method = RESOLVE_COMPUTE;
   } else {
      assert(dst_image_layout == VK_IMAGE_LAYOUT_UNDEFINED);
      if (src_image->vk.array_layers > 1 || dst_image->vk.array_layers > 1 ||
          (dst_image->planes[0].surface.flags & RADEON_SURF_NO_RENDER_TARGET))
         *method = RESOLVE_COMPUTE;
      else
         *method = RESOLVE_FRAGMENT;
   }
}

static void
radv_meta_resolve_hardware_image(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src_image,
                                 VkImageLayout src_image_layout, struct radv_image *dst_image,
                                 VkImageLayout dst_image_layout, const VkImageResolve2 *region)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_meta_saved_state saved_state;

   radv_meta_save(&saved_state, cmd_buffer, RADV_META_SAVE_GRAPHICS_PIPELINE);

   assert(src_image->vk.samples > 1);
   assert(dst_image->vk.samples == 1);

   /* From the Vulkan 1.0 spec:
    *
    *    - The aspectMask member of srcSubresource and dstSubresource must
    *      only contain VK_IMAGE_ASPECT_COLOR_BIT
    */
   assert(region->srcSubresource.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
   assert(region->dstSubresource.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
   /* Multi-layer resolves are handled by compute */
   assert(vk_image_subresource_layer_count(&src_image->vk, &region->srcSubresource) == 1 &&
          vk_image_subresource_layer_count(&dst_image->vk, &region->dstSubresource) == 1);
   /**
    * From Vulkan 1.0.6 spec: 18.6 Resolving Multisample Images
    *
    *    extent is the size in texels of the source image to resolve in width,
    *    height and depth. 1D images use only x and width. 2D images use x, y,
    *    width and height. 3D images use x, y, z, width, height and depth.
    *
    *    srcOffset and dstOffset select the initial x, y, and z offsets in
    *    texels of the sub-regions of the source and destination image data.
    *    extent is the size in texels of the source image to resolve in width,
    *    height and depth. 1D images use only x and width. 2D images use x, y,
    *    width and height. 3D images use x, y, z, width, height and depth.
    */
   const struct VkExtent3D extent = vk_image_sanitize_extent(&src_image->vk, region->extent);
   const struct VkOffset3D dstOffset = vk_image_sanitize_offset(&dst_image->vk, region->dstOffset);

   uint32_t queue_mask = radv_image_queue_family_mask(dst_image, cmd_buffer->qf, cmd_buffer->qf);

   if (radv_layout_dcc_compressed(device, dst_image, region->dstSubresource.mipLevel, dst_image_layout, queue_mask)) {
      VkImageSubresourceRange range = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = region->dstSubresource.mipLevel,
         .levelCount = 1,
         .baseArrayLayer = 0,
         .layerCount = 1,
      };

      cmd_buffer->state.flush_bits |= radv_init_dcc(cmd_buffer, dst_image, &range, 0xffffffff);
   }

   VkRect2D resolve_area = {
      .offset = {dstOffset.x, dstOffset.y},
      .extent = {extent.width, extent.height},
   };

   radv_CmdSetViewport(radv_cmd_buffer_to_handle(cmd_buffer), 0, 1,
                       &(VkViewport){.x = resolve_area.offset.x,
                                     .y = resolve_area.offset.y,
                                     .width = resolve_area.extent.width,
                                     .height = resolve_area.extent.height,
                                     .minDepth = 0.0f,
                                     .maxDepth = 1.0f});

   radv_CmdSetScissor(radv_cmd_buffer_to_handle(cmd_buffer), 0, 1, &resolve_area);

   struct radv_image_view src_iview;
   radv_image_view_init(&src_iview, device,
                        &(VkImageViewCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                           .image = radv_image_to_handle(src_image),
                           .viewType = VK_IMAGE_VIEW_TYPE_2D,
                           .format = src_image->vk.format,
                           .subresourceRange =
                              {
                                 .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1,
                              },
                        },
                        NULL);

   struct radv_image_view dst_iview;
   radv_image_view_init(&dst_iview, device,
                        &(VkImageViewCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                           .image = radv_image_to_handle(dst_image),
                           .viewType = radv_meta_get_view_type(dst_image),
                           .format = dst_image->vk.format,
                           .subresourceRange =
                              {
                                 .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = region->dstSubresource.mipLevel,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1,
                              },
                        },
                        NULL);

   const VkRenderingAttachmentInfo color_atts[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = radv_image_view_to_handle(&src_iview),
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      },
      {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = radv_image_view_to_handle(&dst_iview),
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      },
   };

   const VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_INPUT_ATTACHMENT_NO_CONCURRENT_WRITES_BIT_MESA,
      .renderArea = resolve_area,
      .layerCount = 1,
      .colorAttachmentCount = 2,
      .pColorAttachments = color_atts,
   };

   radv_CmdBeginRendering(radv_cmd_buffer_to_handle(cmd_buffer), &rendering_info);

   emit_resolve(cmd_buffer, src_image, dst_image, dst_iview.vk.format);

   radv_CmdEndRendering(radv_cmd_buffer_to_handle(cmd_buffer));

   radv_image_view_finish(&src_iview);
   radv_image_view_finish(&dst_iview);

   radv_meta_restore(&saved_state, cmd_buffer);
}

static void
resolve_image(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src_image, VkImageLayout src_image_layout,
              struct radv_image *dst_image, VkImageLayout dst_image_layout, const VkImageResolve2 *region,
              enum radv_resolve_method resolve_method)
{
   switch (resolve_method) {
   case RESOLVE_HW:
      radv_meta_resolve_hardware_image(cmd_buffer, src_image, src_image_layout, dst_image, dst_image_layout, region);
      break;
   case RESOLVE_FRAGMENT:
      radv_decompress_resolve_src(cmd_buffer, src_image, src_image_layout, region);

      radv_meta_resolve_fragment_image(cmd_buffer, src_image, src_image_layout, dst_image, dst_image_layout, region);
      break;
   case RESOLVE_COMPUTE:
      radv_decompress_resolve_src(cmd_buffer, src_image, src_image_layout, region);

      radv_meta_resolve_compute_image(cmd_buffer, src_image, src_image->vk.format, src_image_layout, dst_image,
                                      dst_image->vk.format, dst_image_layout, region);
      break;
   default:
      assert(!"Invalid resolve method selected");
   }
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdResolveImage2(VkCommandBuffer commandBuffer, const VkResolveImageInfo2 *pResolveImageInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_image, src_image, pResolveImageInfo->srcImage);
   VK_FROM_HANDLE(radv_image, dst_image, pResolveImageInfo->dstImage);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   VkImageLayout src_image_layout = pResolveImageInfo->srcImageLayout;
   VkImageLayout dst_image_layout = pResolveImageInfo->dstImageLayout;
   enum radv_resolve_method resolve_method = pdev->info.gfx_level >= GFX11 ? RESOLVE_FRAGMENT : RESOLVE_HW;

   radv_suspend_conditional_rendering(cmd_buffer);

   /* we can use the hw resolve only for single full resolves */
   if (pResolveImageInfo->regionCount == 1) {
      if (pResolveImageInfo->pRegions[0].srcOffset.x || pResolveImageInfo->pRegions[0].srcOffset.y ||
          pResolveImageInfo->pRegions[0].srcOffset.z)
         resolve_method = RESOLVE_COMPUTE;
      if (pResolveImageInfo->pRegions[0].dstOffset.x || pResolveImageInfo->pRegions[0].dstOffset.y ||
          pResolveImageInfo->pRegions[0].dstOffset.z)
         resolve_method = RESOLVE_COMPUTE;

      if (pResolveImageInfo->pRegions[0].extent.width != src_image->vk.extent.width ||
          pResolveImageInfo->pRegions[0].extent.height != src_image->vk.extent.height ||
          pResolveImageInfo->pRegions[0].extent.depth != src_image->vk.extent.depth)
         resolve_method = RESOLVE_COMPUTE;
   } else
      resolve_method = RESOLVE_COMPUTE;

   for (uint32_t r = 0; r < pResolveImageInfo->regionCount; r++) {
      const VkImageResolve2 *region = &pResolveImageInfo->pRegions[r];

      radv_pick_resolve_method_images(device, src_image, src_image->vk.format, dst_image,
                                      region->dstSubresource.mipLevel, dst_image_layout, cmd_buffer, &resolve_method);

      resolve_image(cmd_buffer, src_image, src_image_layout, dst_image, dst_image_layout, region, resolve_method);
   }

   radv_resume_conditional_rendering(cmd_buffer);
}

static void
radv_cmd_buffer_resolve_rendering_hw(struct radv_cmd_buffer *cmd_buffer, struct radv_image_view *src_iview,
                                     VkImageLayout src_layout, struct radv_image_view *dst_iview,
                                     VkImageLayout dst_layout)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_meta_saved_state saved_state;

   radv_meta_save(&saved_state, cmd_buffer, RADV_META_SAVE_GRAPHICS_PIPELINE | RADV_META_SAVE_RENDER);

   VkRect2D *resolve_area = &saved_state.render.area;

   radv_CmdSetViewport(radv_cmd_buffer_to_handle(cmd_buffer), 0, 1,
                       &(VkViewport){.x = resolve_area->offset.x,
                                     .y = resolve_area->offset.y,
                                     .width = resolve_area->extent.width,
                                     .height = resolve_area->extent.height,
                                     .minDepth = 0.0f,
                                     .maxDepth = 1.0f});

   radv_CmdSetScissor(radv_cmd_buffer_to_handle(cmd_buffer), 0, 1, resolve_area);

   struct radv_image *src_img = src_iview->image;
   struct radv_image *dst_img = dst_iview->image;
   uint32_t queue_mask = radv_image_queue_family_mask(dst_img, cmd_buffer->qf, cmd_buffer->qf);

   if (radv_layout_dcc_compressed(device, dst_img, dst_iview->vk.base_mip_level, dst_layout, queue_mask)) {
      VkImageSubresourceRange range = {
         .aspectMask = dst_iview->vk.aspects,
         .baseMipLevel = dst_iview->vk.base_mip_level,
         .levelCount = 1,
         .baseArrayLayer = 0,
         .layerCount = 1,
      };

      cmd_buffer->state.flush_bits |= radv_init_dcc(cmd_buffer, dst_img, &range, 0xffffffff);
   }

   const VkRenderingAttachmentInfo color_atts[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = radv_image_view_to_handle(src_iview),
         .imageLayout = src_layout,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      },
      {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = radv_image_view_to_handle(dst_iview),
         .imageLayout = dst_layout,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      },
   };

   const VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_INPUT_ATTACHMENT_NO_CONCURRENT_WRITES_BIT_MESA,
      .renderArea = saved_state.render.area,
      .layerCount = 1,
      .viewMask = saved_state.render.view_mask,
      .colorAttachmentCount = 2,
      .pColorAttachments = color_atts,
   };

   radv_CmdBeginRendering(radv_cmd_buffer_to_handle(cmd_buffer), &rendering_info);

   emit_resolve(cmd_buffer, src_img, dst_img, dst_iview->vk.format);

   radv_CmdEndRendering(radv_cmd_buffer_to_handle(cmd_buffer));

   radv_meta_restore(&saved_state, cmd_buffer);
}

/**
 * Emit any needed resolves for the current subpass.
 */
void
radv_cmd_buffer_resolve_rendering(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   enum radv_resolve_method resolve_method = pdev->info.gfx_level >= GFX11 ? RESOLVE_FRAGMENT : RESOLVE_HW;

   bool has_color_resolve = false;
   for (uint32_t i = 0; i < render->color_att_count; ++i) {
      if (render->color_att[i].resolve_iview != NULL)
         has_color_resolve = true;
   }
   bool has_ds_resolve = render->ds_att.resolve_iview != NULL;

   if (!has_color_resolve && !has_ds_resolve)
      return;

   radv_describe_begin_render_pass_resolve(cmd_buffer);

   if (render->ds_att.resolve_iview != NULL) {
      struct radv_image_view *src_iview = render->ds_att.iview;
      struct radv_image_view *dst_iview = render->ds_att.resolve_iview;

      radv_pick_resolve_method_images(device, src_iview->image, src_iview->vk.format, dst_iview->image,
                                      dst_iview->vk.base_mip_level, VK_IMAGE_LAYOUT_UNDEFINED, cmd_buffer,
                                      &resolve_method);

      if ((src_iview->vk.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) && render->ds_att.resolve_mode != VK_RESOLVE_MODE_NONE) {
         if (resolve_method == RESOLVE_FRAGMENT) {
            radv_depth_stencil_resolve_rendering_fs(cmd_buffer, VK_IMAGE_ASPECT_DEPTH_BIT, render->ds_att.resolve_mode);
         } else {
            assert(resolve_method == RESOLVE_COMPUTE);
            radv_depth_stencil_resolve_rendering_cs(cmd_buffer, VK_IMAGE_ASPECT_DEPTH_BIT, render->ds_att.resolve_mode);
         }
      }

      if ((src_iview->vk.aspects & VK_IMAGE_ASPECT_STENCIL_BIT) &&
          render->ds_att.stencil_resolve_mode != VK_RESOLVE_MODE_NONE) {
         if (resolve_method == RESOLVE_FRAGMENT) {
            radv_depth_stencil_resolve_rendering_fs(cmd_buffer, VK_IMAGE_ASPECT_STENCIL_BIT,
                                                    render->ds_att.stencil_resolve_mode);
         } else {
            assert(resolve_method == RESOLVE_COMPUTE);
            radv_depth_stencil_resolve_rendering_cs(cmd_buffer, VK_IMAGE_ASPECT_STENCIL_BIT,
                                                    render->ds_att.stencil_resolve_mode);
         }
      }

      /* From the Vulkan spec 1.2.165:
       *
       * "VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT specifies
       *  write access to a color, resolve, or depth/stencil
       *  resolve attachment during a render pass or via
       *  certain subpass load and store operations."
       *
       * Yes, it's counterintuitive but it makes sense because ds
       * resolve operations happen late at the end of the subpass.
       *
       * That said, RADV is wrong because it executes the subpass
       * end barrier *before* any subpass resolves instead of after.
       *
       * TODO: Fix this properly by executing subpass end barriers
       * after subpass resolves.
       */
      cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_FLUSH_AND_INV_DB;
      if (radv_htile_enabled(dst_iview->image, dst_iview->vk.base_mip_level))
         cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_FLUSH_AND_INV_DB_META;
   }

   if (has_color_resolve) {
      uint32_t layer_count = render->layer_count;
      VkRect2D resolve_area = render->area;
      struct radv_resolve_barrier barrier;

      if (render->view_mask)
         layer_count = util_last_bit(render->view_mask);

      /* Resolves happen before the end-of-subpass barriers get executed, so we have to make the
       * attachment shader-readable.
       */
      barrier.src_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      barrier.dst_stage_mask = VK_PIPELINE_STAGE_2_RESOLVE_BIT;
      barrier.src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      barrier.dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      radv_emit_resolve_barrier(cmd_buffer, &barrier);

      for (uint32_t i = 0; i < render->color_att_count; ++i) {
         if (render->color_att[i].resolve_iview == NULL)
            continue;

         struct radv_image_view *src_iview = render->color_att[i].iview;
         VkImageLayout src_layout = render->color_att[i].layout;
         struct radv_image *src_img = src_iview->image;
         struct radv_image_view *dst_iview = render->color_att[i].resolve_iview;
         VkImageLayout dst_layout = render->color_att[i].resolve_layout;
         struct radv_image *dst_img = dst_iview->image;

         radv_pick_resolve_method_images(device, src_img, src_iview->vk.format, dst_img, dst_iview->vk.base_mip_level,
                                         dst_layout, cmd_buffer, &resolve_method);
         VkImageResolve2 region = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2,
            .extent =
               {
                  .width = resolve_area.extent.width,
                  .height = resolve_area.extent.height,
                  .depth = 1,
               },
            .srcSubresource =
               (VkImageSubresourceLayers){
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = src_iview->vk.base_mip_level,
                  .baseArrayLayer = src_iview->vk.base_array_layer,
                  .layerCount = layer_count,
               },
            .dstSubresource =
               (VkImageSubresourceLayers){
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = dst_iview->vk.base_mip_level,
                  .baseArrayLayer = dst_iview->vk.base_array_layer,
                  .layerCount = layer_count,
               },
            .srcOffset = {resolve_area.offset.x, resolve_area.offset.y, 0},
            .dstOffset = {resolve_area.offset.x, resolve_area.offset.y, 0},
         };

         switch (resolve_method) {
         case RESOLVE_HW:
            radv_cmd_buffer_resolve_rendering_hw(cmd_buffer, src_iview, src_layout, dst_iview, dst_layout);
            break;
         case RESOLVE_COMPUTE:
            radv_decompress_resolve_src(cmd_buffer, src_iview->image, src_layout, &region);

            radv_cmd_buffer_resolve_rendering_cs(cmd_buffer, src_iview, src_layout, dst_iview, dst_layout, &region);
            break;
         case RESOLVE_FRAGMENT:
            radv_decompress_resolve_src(cmd_buffer, src_iview->image, src_layout, &region);

            radv_cmd_buffer_resolve_rendering_fs(cmd_buffer, src_iview, src_layout, dst_iview, dst_layout);
            break;
         default:
            UNREACHABLE("Invalid resolve method");
         }
      }
   }

   radv_describe_end_render_pass_resolve(cmd_buffer);
}

/**
 * Decompress CMask/FMask before resolving a multisampled source image inside a
 * subpass.
 */
void
radv_decompress_resolve_rendering_src(struct radv_cmd_buffer *cmd_buffer)
{
   const struct radv_rendering_state *render = &cmd_buffer->state.render;

   uint32_t layer_count = render->layer_count;
   if (render->view_mask)
      layer_count = util_last_bit(render->view_mask);

   for (uint32_t i = 0; i < render->color_att_count; ++i) {
      if (render->color_att[i].resolve_iview == NULL)
         continue;

      struct radv_image_view *src_iview = render->color_att[i].iview;
      VkImageLayout src_layout = render->color_att[i].layout;
      struct radv_image *src_image = src_iview->image;

      VkImageResolve2 region = {0};
      region.sType = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2;
      region.srcSubresource.aspectMask = src_iview->vk.aspects;
      region.srcSubresource.mipLevel = 0;
      region.srcSubresource.baseArrayLayer = src_iview->vk.base_array_layer;
      region.srcSubresource.layerCount = layer_count;

      radv_decompress_resolve_src(cmd_buffer, src_image, src_layout, &region);
   }
}

/**
 * Decompress CMask/FMask before resolving a multisampled source image.
 */
void
radv_decompress_resolve_src(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src_image,
                            VkImageLayout src_image_layout, const VkImageResolve2 *region)
{
   VkImageMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
      .oldLayout = src_image_layout,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .image = radv_image_to_handle(src_image),
      .subresourceRange = (VkImageSubresourceRange){
         .aspectMask = region->srcSubresource.aspectMask,
         .baseMipLevel = 0,
         .levelCount = 1,
         .baseArrayLayer = region->srcSubresource.baseArrayLayer,
         .layerCount = vk_image_subresource_layer_count(&src_image->vk, &region->srcSubresource),
      }};

   VkSampleLocationsInfoEXT sample_loc_info;
   if (src_image->vk.create_flags & VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT) {
      /* If the depth/stencil image uses different sample
       * locations, we need them during HTILE decompressions.
       */
      struct radv_sample_locations_state *sample_locs = &cmd_buffer->state.render.sample_locations;

      sample_loc_info = (VkSampleLocationsInfoEXT){
         .sType = VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT,
         .sampleLocationsPerPixel = sample_locs->per_pixel,
         .sampleLocationGridSize = sample_locs->grid_size,
         .sampleLocationsCount = sample_locs->count,
         .pSampleLocations = sample_locs->locations,
      };
      barrier.pNext = &sample_loc_info;
   }

   VkDependencyInfo dep_info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
   };

   radv_CmdPipelineBarrier2(radv_cmd_buffer_to_handle(cmd_buffer), &dep_info);
}
