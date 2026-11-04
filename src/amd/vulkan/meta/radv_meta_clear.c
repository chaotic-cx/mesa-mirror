/*
 * Copyright © 2015 Intel Corporation
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir/radv_meta_nir.h"
#include "radv_entrypoints.h"
#include "radv_formats.h"
#include "radv_meta.h"
#include "radv_tracepoints.h"

#include "util/format_rgb9e5.h"
#include "util/format_srgb.h"
#include "vk_format.h"
#include "vk_shader_module.h"

#include "ac_formats.h"

#define RANGES_INTERSECT(first1, last1, first2, last2) (MAX2(first1, first2) <= MIN2(last1, last2))

/* This tracks all pending flushes and their layer range per attachment and codepath (draw and compute/CP DMA)
 * independently. If a clear's region (using compute/CP DMA/draw) intersects another region, the pending
 * flushes are applied between them.
 *
 * If clears intersect the layer range, compute/CP DMA clears always apply all pending flushes, while draws
 * ignore pending draw flushes since draws always maintain ordering.
 *
 * When pending flush flags accumulate, the layer range can only expand. Only clearing the flush flags resets
 * the layer range.
 *
 * Note that DCC/CMASK/HTILE can intersect even if the pixel regions or aspects don't.
 */
struct radv_sync_range_info {
   enum ac_barrier_flags pending_flush;
   unsigned first_layer, last_layer;
};

struct radv_sync_att_info {
   struct radv_sync_range_info compute_dma; /* Compute + CP DMA */
   struct radv_sync_range_info draw;
};

enum att_id {
   ATT_ZS = 0,
   ATT_COLOR0 = 1,
   NUM_ATTACHMENTS = ATT_COLOR0 + 8,
};

struct radv_clear_sync_info {
   /* 0: Z/S, 1-8: Color attachments. */
   struct radv_sync_att_info att[NUM_ATTACHMENTS];
};

static void
set_pending_flush_flags(struct radv_sync_range_info *info, enum ac_barrier_flags flush_bits, unsigned first_layer,
                        unsigned layer_count)
{
   const unsigned last_layer = first_layer + layer_count - 1;

   if (info->pending_flush) {
      info->pending_flush |= flush_bits;
      info->first_layer = MIN2(info->first_layer, first_layer);
      info->last_layer = MAX2(info->last_layer, last_layer);
   } else {
      info->pending_flush = flush_bits;
      info->first_layer = first_layer;
      info->last_layer = last_layer;
   }
}

static bool
htile_may_be_bound(struct radv_cmd_buffer *cmd_buffer)
{
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   const struct radv_image_view *ds_iview = render->ds_att.iview;

   return render->ds_att.format != VK_FORMAT_UNDEFINED &&
          (!ds_iview || radv_htile_enabled(ds_iview->image, ds_iview->vk.base_mip_level));
}

static void
set_pending_draw_flush_flags(struct radv_cmd_buffer *cmd_buffer, struct radv_clear_sync_info *sync_info,
                             const struct radv_attachment *att, enum att_id att_id, unsigned first_layer,
                             unsigned layer_count)
{
   const struct radv_rendering_state *render = &cmd_buffer->state.render;

   const VkImageSubresourceRange range = {
      .aspectMask = att->iview->vk.aspects,
      .baseMipLevel = att->iview->vk.base_mip_level,
      .levelCount = att->iview->vk.level_count,
      .baseArrayLayer = att->iview->vk.base_array_layer + first_layer,
      .layerCount = layer_count,
   };

   if (att->iview->vk.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
      set_pending_flush_flags(
         &sync_info->att[att_id].draw,
         radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, 0, att->iview->image, &range),
         first_layer, layer_count);
   } else {
      set_pending_flush_flags(
         &sync_info->att[att_id].draw,
         radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, 0, att->iview->image, &range),
         first_layer, layer_count);

      struct radv_image_view *ds_iview = render->ds_att.iview;

      /* HTILE is loaded and cached by draws even during color-only clears that don't touch Z/S, which means
       * compute can't touch HTILE in parallel with slow color clears. This will make all non-draw Z/S clears
       * wait for pending draws.
       */
      if (htile_may_be_bound(cmd_buffer)) {
         const VkImageSubresourceRange ds_range = {
            .aspectMask = ds_iview ? ds_iview->vk.aspects : 0,
            .baseMipLevel = ds_iview ? ds_iview->vk.base_mip_level : 0,
            .levelCount = ds_iview ? ds_iview->vk.level_count : 0,
            .baseArrayLayer = ds_iview ? ds_iview->vk.base_array_layer + first_layer : 0,
            .layerCount = ds_iview ? layer_count : 0,
         };

         set_pending_flush_flags(&sync_info->att[ATT_ZS].draw,
                                 radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, 0,
                                                       ds_iview ? ds_iview->image : NULL, ds_iview ? &ds_range : NULL),
                                 first_layer, layer_count);
      }
   }
}

static void
apply_pending_compute_dma_sync_flags(struct radv_cmd_buffer *cmd_buffer, const struct radv_clear_sync_info *sync_info)
{
   for (unsigned a = 0; a < ARRAY_SIZE(sync_info->att); a++)
      cmd_buffer->state.flush_bits |= sync_info->att[a].compute_dma.pending_flush;
}

static void
sync_for_clears(struct radv_cmd_buffer *cmd_buffer, struct radv_clear_sync_info *sync_info, enum att_id att_id,
                bool draw, unsigned first_layer, unsigned layer_count)
{
   /* This table shows when a barrier is necessary between 2 clears to the same region:
    *
    *       previous clear | next clear | need to sync?
    *       -----------------------------------------------------------------------
    *       CP DMA         | CP DMA     | no - CP DMA fills always maintain ordering
    *       compute        | CP DMA     | yes
    *       draw           | CP DMA     | yes
    *       CP DMA         | compute    | yes
    *       compute        | compute    | yes
    *       draw           | compute    | yes
    *       CP DMA         | draw       | yes
    *       compute        | draw       | yes
    *       draw           | draw       | no - draws always maintain ordering
    *
    * TODO: We don't skip barriers between two CP DMA fills because we don't know before a fast clear
    * whether CP DMA or compute is going to be used.
    */
   const struct radv_sync_att_info *att = &sync_info->att[att_id];
   const unsigned last_layer = first_layer + layer_count - 1;
   enum ac_barrier_flags wait_flags = 0;

   /* Only synchronize if the ranges intersect. */
   if (RANGES_INTERSECT(att->compute_dma.first_layer, att->compute_dma.last_layer, first_layer, last_layer))
      wait_flags |= att->compute_dma.pending_flush;

   if (!draw && RANGES_INTERSECT(att->draw.first_layer, att->draw.last_layer, first_layer, last_layer))
      wait_flags |= att->draw.pending_flush;

   /* HTILE is loaded and cached by draws even during color-only clears that don't touch Z/S, which means
    * we can't fast-clear HTILE in parallel with slow color clears. This makes all draws wait for pending
    * non-draw Z/S clears.
    */
   if (draw && htile_may_be_bound(cmd_buffer))
      wait_flags |= sync_info->att[ATT_ZS].compute_dma.pending_flush;

   if (wait_flags) {
      cmd_buffer->state.flush_bits |= wait_flags;

      /* Clear the applied flags from other attachments. */
      for (unsigned a = 0; a < ARRAY_SIZE(sync_info->att); a++) {
         sync_info->att[a].compute_dma.pending_flush &= ~wait_flags;
         sync_info->att[a].draw.pending_flush &= ~wait_flags;
      }
   }
}

static bool
radv_is_clear_rect_full(const struct radv_image_view *iview, const VkClearRect *clear_rect, uint32_t view_mask,
                        bool need_all_slices)
{
   const struct radv_image *image = iview->image;

   if (clear_rect->rect.offset.x || clear_rect->rect.offset.y ||
       clear_rect->rect.extent.width != image->vk.extent.width ||
       clear_rect->rect.extent.height != image->vk.extent.height)
      return false;

   const unsigned view_mask_layers = util_bitcount(view_mask);

   /* All view mask layers must be consecutive. */
   if (view_mask && view_mask != BITFIELD_MASK(view_mask_layers))
      return false;

   const unsigned num_layers = view_mask ? view_mask_layers : clear_rect->layerCount;
   const bool all_slices = num_layers == image->vk.array_layers;

   /* All slices must be fast-cleared when comp-to-single isn't supported. */
   if (!all_slices && need_all_slices)
      return false;

   return true;
}

static VkResult
get_color_pipeline_layout(struct radv_device *device, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_CLEAR_COLOR;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      .size = 16,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                      layout_out);
}

struct radv_clear_color_key {
   enum radv_meta_object_key_type type;
   VkFormat format;
   uint8_t samples;
   uint8_t frag_output;
   bool multiview;
};

static VkResult
get_color_pipeline(struct radv_device *device, uint32_t samples, uint32_t frag_output, VkFormat format,
                   uint32_t view_mask, VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   struct radv_clear_color_key key;
   VkResult result;

   result = get_color_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_CLEAR_COLOR;
   key.format = format;
   key.samples = samples;
   key.frag_output = frag_output;
   key.multiview = view_mask != 0;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *vs_module, *fs_module;
   radv_meta_nir_build_clear_color_shaders(&vs_module, &fs_module, frag_output, view_mask != 0);

   VkPipelineColorBlendAttachmentState blend_attachment_state[MAX_RTS] = {0};
   blend_attachment_state[frag_output] = (VkPipelineColorBlendAttachmentState){
      .blendEnable = false,
      .colorWriteMask =
         VK_COLOR_COMPONENT_A_BIT | VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT,
   };

   const VkGraphicsPipelineCreateInfo pipeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = fs_module ? 2 : 1,
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
            .rasterizerDiscardEnable = false,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = false,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f,
         },
      .pMultisampleState =
         &(VkPipelineMultisampleStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = samples,
            .sampleShadingEnable = false,
            .pSampleMask = NULL,
            .alphaToCoverageEnable = false,
            .alphaToOneEnable = false,
         },
      .pColorBlendState =
         &(VkPipelineColorBlendStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = false,
            .attachmentCount = MAX_RTS,
            .pAttachments = blend_attachment_state,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
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
      .view_mask = view_mask,
      .color_attachment_count = MAX_RTS,
   };

   for (uint32_t i = 0; i < MAX_RTS; i++)
      render.color_attachment_formats[i] = format;

   result = vk_meta_create_graphics_pipeline(&device->vk, &device->meta_state.device, &pipeline_create_info, &render,
                                             &key, sizeof(key), pipeline_out);

   ralloc_free(vs_module);
   ralloc_free(fs_module);
   return result;
}

static void
emit_render_color_clear(struct radv_cmd_buffer *cmd_buffer, const VkClearAttachment *clear_att,
                        const VkClearRect *clear_rect, uint32_t view_mask)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   uint32_t samples;
   VkFormat format;
   VkClearColorValue clear_value = clear_att->clearValue.color;
   VkCommandBuffer cmd_buffer_h = radv_cmd_buffer_to_handle(cmd_buffer);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   assert(clear_att->aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
   assert(clear_att->colorAttachment < render->color_att_count);
   const struct radv_attachment *color_att = &render->color_att[clear_att->colorAttachment];

   /* When a framebuffer is bound to the current command buffer, get the
    * number of samples from it. Otherwise, get the number of samples from
    * the render pass because it's likely a secondary command buffer.
    */
   if (color_att->iview) {
      samples = color_att->iview->image->vk.samples;
      format = color_att->iview->vk.format;
   } else {
      samples = render->max_samples;
      format = color_att->format;
   }
   assert(format != VK_FORMAT_UNDEFINED);

   switch (format) {
   case VK_FORMAT_R64_UINT:
      format = VK_FORMAT_R32G32_UINT;
      break;
   case VK_FORMAT_R64_SINT:
      format = VK_FORMAT_R32G32_SINT;
      break;
   case VK_FORMAT_R64_SFLOAT:
      format = VK_FORMAT_R32G32_SFLOAT;
      break;
   default:
      break;
   }

   assert(util_is_power_of_two_nonzero(samples));

   result = get_color_pipeline(device, samples, clear_att->colorAttachment, format, view_mask, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, &clear_value);

   radv_meta_bind_graphics_pipeline(cmd_buffer, pipeline);

   radv_meta_set_viewport_and_scissor(cmd_buffer, clear_rect->rect.offset.x, clear_rect->rect.offset.y,
                                      clear_rect->rect.extent.width, clear_rect->rect.extent.height);

   if (view_mask) {
      radv_CmdDraw(cmd_buffer_h, 3, 1, 0, 0);
   } else {
      radv_CmdDraw(cmd_buffer_h, 3, clear_rect->layerCount, 0, clear_rect->baseArrayLayer);
   }
}

static enum radv_fast_clear_support
radv_can_fast_clear_depth(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview,
                          VkImageLayout image_layout, VkImageAspectFlags aspects, const VkClearRect *clear_rect,
                          const VkClearDepthStencilValue clear_value, uint32_t view_mask,
                          VkImageAspectFlags *valid_aspect, unsigned *max_levels);

struct radv_clear_ds_layout_key {
   enum radv_meta_object_key_type type;
};

static VkResult
get_depth_stencil_pipeline_layout(struct radv_device *device, VkPipelineLayout *layout_out)
{
   struct radv_clear_ds_layout_key key;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_CLEAR_DS;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .size = 4,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                      layout_out);
}

struct radv_clear_ds_key {
   enum radv_meta_object_key_type type;
   VkImageAspectFlags aspects;
   uint8_t samples;
   bool multiview;
};

static VkResult
get_depth_stencil_pipeline(struct radv_device *device, int samples, VkImageAspectFlags aspects, uint32_t view_mask,
                           VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   struct radv_clear_ds_key key;
   VkResult result;

   result = get_depth_stencil_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_CLEAR_DS;
   key.aspects = aspects;
   key.samples = samples;
   key.multiview = view_mask != 0;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *vs_module = radv_meta_nir_build_clear_depthstencil_vertex_shader(view_mask != 0);
   nir_shader *fs_module = radv_meta_nir_build_fs_noop();

   VkGraphicsPipelineCreateInfoRADV radv_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO_RADV,
   };

   const VkGraphicsPipelineCreateInfo pipeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &radv_info,
      .stageCount = fs_module ? 2 : 1,
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
            .pNext =
               &(VkPipelineRasterizationDepthClipStateCreateInfoEXT){
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT,
                  .depthClipEnable = false,
               },
            .rasterizerDiscardEnable = false,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = false,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f,
         },
      .pMultisampleState =
         &(VkPipelineMultisampleStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = samples,
            .sampleShadingEnable = false,
            .pSampleMask = NULL,
            .alphaToCoverageEnable = false,
            .alphaToOneEnable = false,
         },
      .pDepthStencilState =
         &(VkPipelineDepthStencilStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = !!(aspects & VK_IMAGE_ASPECT_DEPTH_BIT),
            .depthCompareOp = VK_COMPARE_OP_ALWAYS,
            .depthWriteEnable = !!(aspects & VK_IMAGE_ASPECT_DEPTH_BIT),
            .depthBoundsTestEnable = false,
            .stencilTestEnable = !!(aspects & VK_IMAGE_ASPECT_STENCIL_BIT),
            .front =
               {
                  .passOp = VK_STENCIL_OP_REPLACE,
                  .compareOp = VK_COMPARE_OP_ALWAYS,
                  .writeMask = UINT32_MAX,
                  .reference = 0, /* dynamic */
               },
            .back = {0 /* dont care */},
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f,
         },
      .pColorBlendState =
         &(VkPipelineColorBlendStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = false,
            .attachmentCount = 0,
            .pAttachments = NULL,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
         },
      .pDynamicState =
         &(VkPipelineDynamicStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 3,
            .pDynamicStates =
               (VkDynamicState[]){
                  VK_DYNAMIC_STATE_VIEWPORT,
                  VK_DYNAMIC_STATE_SCISSOR,
                  VK_DYNAMIC_STATE_STENCIL_REFERENCE,
               },
         },
      .layout = *layout_out,
   };

   struct vk_meta_rendering_info render = {
      .view_mask = view_mask,
      .depth_attachment_format = (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED,
      .stencil_attachment_format = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_FORMAT_S8_UINT : VK_FORMAT_UNDEFINED,
   };

   result = vk_meta_create_graphics_pipeline(&device->vk, &device->meta_state.device, &pipeline_create_info, &render,
                                             &key, sizeof(key), pipeline_out);

   ralloc_free(vs_module);
   ralloc_free(fs_module);
   return result;
}

static void
emit_depthstencil_clear(struct radv_cmd_buffer *cmd_buffer, VkClearDepthStencilValue clear_value,
                        VkImageAspectFlags aspects, const VkClearRect *clear_rect, uint32_t view_mask)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   struct radv_image_view *iview = render->ds_att.iview;
   uint32_t samples;
   VkCommandBuffer cmd_buffer_h = radv_cmd_buffer_to_handle(cmd_buffer);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   /* When a framebuffer is bound to the current command buffer, get the
    * number of samples from it. Otherwise, get the number of samples from
    * the render pass because it's likely a secondary command buffer.
    */
   if (iview) {
      samples = iview->image->vk.samples;
   } else {
      assert(render->ds_att.format != VK_FORMAT_UNDEFINED);
      samples = render->max_samples;
   }

   assert(util_is_power_of_two_nonzero(samples));

   result = get_depth_stencil_pipeline(device, samples, aspects, view_mask, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   if (!(aspects & VK_IMAGE_ASPECT_DEPTH_BIT))
      clear_value.depth = 1.0f;

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &clear_value.depth);

   if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) {
      radv_meta_set_stencil_reference(cmd_buffer, VK_STENCIL_FACE_FRONT_BIT, clear_value.stencil);
   }

   radv_meta_bind_graphics_pipeline(cmd_buffer, pipeline);

   radv_meta_set_viewport_and_scissor(cmd_buffer, clear_rect->rect.offset.x, clear_rect->rect.offset.y,
                                      clear_rect->rect.extent.width, clear_rect->rect.extent.height);

   if (view_mask) {
      radv_CmdDraw(cmd_buffer_h, 3, 1, 0, 0);
   } else {
      radv_CmdDraw(cmd_buffer_h, 3, clear_rect->layerCount, 0, clear_rect->baseArrayLayer);
   }

   if (iview && radv_image_has_hiz_metadata(iview->image) && (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) &&
       radv_is_clear_rect_full(iview, clear_rect, view_mask, true)) {
      const VkImageSubresourceRange range = {
         .aspectMask = aspects,
         .baseMipLevel = iview->vk.base_mip_level,
         .levelCount = iview->vk.level_count,
         .baseArrayLayer = iview->vk.base_array_layer,
         .layerCount = iview->vk.layer_count,
      };

      radv_update_hiz_metadata(cmd_buffer, iview->image, &range, true);
   }
}

struct radv_clear_htile_mask_key {
   enum radv_meta_object_key_type type;
   bool mask_zbase_conditionally;
   bool full_mask;
};

static VkResult
get_clear_htile_mask_pipeline(struct radv_device *device, VkPipeline *pipeline_out, VkPipelineLayout *layout_out,
                              bool mask_zbase_conditionally, bool full_mask)
{
   struct radv_clear_htile_mask_key key;
   memset(&key, 0, sizeof(key));

   key.type = RADV_META_OBJECT_KEY_CLEAR_HTILE;
   key.mask_zbase_conditionally = mask_zbase_conditionally;
   key.full_mask = full_mask;

   VkResult result;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = (mask_zbase_conditionally ? 6 : 4) * 4,
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                        layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_clear_htile_mask_shader(mask_zbase_conditionally, full_mask);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static uint32_t
clear_htile_mask(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image, struct radeon_winsys_bo *bo,
                 uint64_t va, uint64_t size, uint32_t htile_value, uint32_t htile_mask, uint64_t value_select_cond_va)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;
   uint64_t block_count = DIV_ROUND_UP(size, 1024);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result =
      get_clear_htile_mask_pipeline(device, &pipeline, &layout, value_select_cond_va != 0, htile_mask == 0xffffffff);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return 0;
   }

   radv_cs_add_buffer(device->ws, cs->b, bo);

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   const unsigned constants[] = {
      va, va >> 32, htile_value & htile_mask, ~htile_mask, value_select_cond_va, value_select_cond_va >> 32,
   };
   const unsigned constants_size = (value_select_cond_va ? 6 : 4) * 4;

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constants_size, constants);

   /* The condition value is loaded by SMEM, but is stored by the CP. */
   if (value_select_cond_va)
      cmd_buffer->state.flush_bits |= AC_BARRIER_INV_SMEM;

   radv_CmdDispatchBase(radv_cmd_buffer_to_handle(cmd_buffer), 0, 0, 0, block_count, 1, 1);

   /* PFP can overwrite the predicate while the shader is in flight, so PFP_SYNC_ME is needed.
    * However, a single barrier with PFP_SYNC_ME after this dispatch is not enough if the predicate
    * is overwritten before the barrier. That never happens though. All functions that overwrite
    * the predicate only do it after a barrier + dispatch or CP DMA clear, so we are safe.
    */
   return AC_BARRIER_SYNC_CS | (value_select_cond_va ? AC_BARRIER_PFP_SYNC_ME : 0) |
          radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, 0,
                                image, NULL);
}

static uint32_t
radv_get_htile_fast_clear_value(const struct radv_device *device, const struct radv_image *image,
                                VkClearDepthStencilValue value)
{
   if (radv_image_tile_stencil_disabled(device, image)) {
      return HTILE_Z_CLEAR_REG(value.depth);
   } else {
      if (radv_image_has_vrs_htile(device, image))
         return HTILE_ZS_VRS_CLEAR_REG(value.depth);
      else
         return HTILE_ZS_CLEAR_REG(value.depth);
   }
}

static uint32_t
radv_get_htile_mask(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image, VkImageAspectFlags aspects,
                    enum radv_meta_htile_preserve_flags preserve)
{
   const struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   assert(!(preserve & (RADV_HTILE_DYNAMIC_ZRANGE_PRECISION)));
   assert(preserve != RADV_HTILE_PRESERVE_ALL);

   if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
      /* Clear the whole dword on SDMA because SDMA can't do read-modify-write if not executing
       * as a gang with compute.
       * TODO: We can't do this if we want to pass CTS.
       */
      return UINT32_MAX;
   }

   /* Only preserve depth/stencil fields when they exist, and don't preserve them when they don't,
    * so that we get read-modify-write less often.
    */
   if (vk_format_has_depth(image->vk.format)) {
      if (!(aspects & VK_IMAGE_ASPECT_DEPTH_BIT))
         preserve |= RADV_HTILE_PRESERVE_Z_COMPRESSION | RADV_HTILE_PRESERVE_HIZ_RANGE;
   } else {
      assert(!(aspects & VK_IMAGE_ASPECT_DEPTH_BIT));
      preserve &= ~(RADV_HTILE_PRESERVE_Z_COMPRESSION | RADV_HTILE_PRESERVE_HIZ_RANGE);
   }

   if (vk_format_has_stencil(image->vk.format)) {
      if (!(aspects & VK_IMAGE_ASPECT_STENCIL_BIT))
         preserve |= RADV_HTILE_PRESERVE_STENCIL_COMPRESSION;
   } else {
      assert(!(aspects & VK_IMAGE_ASPECT_STENCIL_BIT));
      preserve &= ~RADV_HTILE_PRESERVE_STENCIL_COMPRESSION;
   }

   if (preserve == RADV_HTILE_PRESERVE_ALL)
      return 0;

   const uint32_t zmask_mask = preserve & RADV_HTILE_PRESERVE_Z_COMPRESSION ? 0 : ~0;
   const uint32_t zrange_mask = preserve & RADV_HTILE_PRESERVE_HIZ_RANGE ? 0 : ~0;
   const uint32_t smem_mask = preserve & RADV_HTILE_PRESERVE_STENCIL_COMPRESSION ? 0 : ~0;
   const uint32_t vrs_mask = preserve & RADV_HTILE_PRESERVE_VRS ? 0 : ~0;
   uint32_t mask = 0;

   // clang-format off
   if (radv_image_has_vrs_htile(device, image)) {
      assert(zmask_mask || zrange_mask || smem_mask || vrs_mask);

      mask |= HTILE_ZS_VRS_CODE(.zmask = zmask_mask,
                                .zdelta = zrange_mask,
                                .zbase = zrange_mask,
                                .sr0 = 0x3,
                                .smem = smem_mask,
                                .vrs_x = vrs_mask,
                                .vrs_y = vrs_mask);
   } else if (!radv_image_tile_stencil_disabled(device, image)) {
      if (!zmask_mask && !zrange_mask && !smem_mask)
         return 0;

      mask |= HTILE_ZS_CODE(.zmask = zmask_mask,
                            .zdelta = zrange_mask,
                            .zbase = zrange_mask,
                            .sr0 = 0x3,
                            .sr1 = 0x3,
                            .smem = smem_mask,
                            .unused = 0x3);
   } else {
      assert(aspects == VK_IMAGE_ASPECT_DEPTH_BIT);
      mask |= HTILE_Z_CODE(.zmask = zmask_mask,
                           .minz = zrange_mask,
                           .maxz = zrange_mask);
   }
   // clang-format on

   return mask;
}

static bool
radv_is_fast_clear_depth_allowed(VkClearDepthStencilValue value)
{
   return value.depth == 1.0f || value.depth == 0.0f;
}

static bool
radv_is_fast_clear_stencil_allowed(VkClearDepthStencilValue value)
{
   return value.stencil == 0;
}

enum radv_fast_clear_support {
   FAST_CLEAR_VALID,
   FAST_CLEAR_INVALID_OTHER,
   FAST_CLEAR_INVALID_SPLIT_LEVEL_RANGE,
   FAST_CLEAR_INVALID_SPLIT_ZS_ASPECTS,
};

/* If FAST_CLEAR_INVALID_SPLIT_LEVEL_RANGE is returned, only "max_levels" number of levels can be fast-cleared.
 * If FAST_CLEAR_INVALID_SPLIT_ZS_ASPECTS is returned, only "valid_aspect" can be fast-cleared.
 */
static enum radv_fast_clear_support
radv_can_fast_clear_depth(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview,
                          VkImageLayout image_layout, VkImageAspectFlags aspects, const VkClearRect *clear_rect,
                          const VkClearDepthStencilValue clear_value, uint32_t view_mask,
                          VkImageAspectFlags *valid_aspect, unsigned *max_levels)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);

   if (pdev->drirc.performance.image_meta_path == RADV_IMAGE_META_PATH_FRAGMENT ||
       pdev->drirc.performance.image_meta_path == RADV_IMAGE_META_PATH_COMPUTE)
      return FAST_CLEAR_INVALID_OTHER;

   if (!iview || !iview->support_fast_clear)
      return FAST_CLEAR_INVALID_OTHER;

   if (!radv_layout_is_htile_compressed(device, iview->image, iview->vk.base_mip_level, image_layout,
                                        radv_image_queue_family_mask(iview->image, cmd_buffer->qf)))
      return FAST_CLEAR_INVALID_OTHER;

   if (!radv_is_clear_rect_full(iview, clear_rect, view_mask, true))
      return FAST_CLEAR_INVALID_OTHER;

   if (radv_tc_compat_htile_enabled(iview->image, iview->vk.base_mip_level)) {
      const bool depth_valid = radv_is_fast_clear_depth_allowed(clear_value);
      const bool stencil_valid = radv_is_fast_clear_stencil_allowed(clear_value);

      if (aspects == VK_IMAGE_ASPECT_DEPTH_BIT) {
         if (!depth_valid)
            return FAST_CLEAR_INVALID_OTHER;
      } else if (aspects == VK_IMAGE_ASPECT_STENCIL_BIT) {
         if (!stencil_valid)
            return FAST_CLEAR_INVALID_OTHER;
      } else {
         if (!depth_valid && !stencil_valid)
            return FAST_CLEAR_INVALID_OTHER;
         else if (depth_valid != stencil_valid) {
            if (valid_aspect)
               *valid_aspect = depth_valid ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
            return FAST_CLEAR_INVALID_SPLIT_ZS_ASPECTS;
         }
      }
   } else {
      if (device->vk.enabled_extensions.EXT_depth_range_unrestricted && (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) &&
          (clear_value.depth < 0.0 || clear_value.depth > 1.0)) {
         if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) {
            if (valid_aspect)
               *valid_aspect = VK_IMAGE_ASPECT_STENCIL_BIT;
            return FAST_CLEAR_INVALID_SPLIT_ZS_ASPECTS;
         } else {
            return FAST_CLEAR_INVALID_OTHER;
         }
      }
   }

   if (iview->image->vk.mip_levels > 1) {
      const unsigned last_level = iview->vk.base_mip_level + iview->vk.level_count - 1;
      const unsigned num_meta_levels = iview->image->planes[0].surface.num_meta_levels;

      if (last_level >= num_meta_levels) {
         if (iview->vk.base_mip_level < num_meta_levels) {
            if (max_levels)
               *max_levels = num_meta_levels - iview->vk.base_mip_level;
            return FAST_CLEAR_INVALID_SPLIT_LEVEL_RANGE;
         } else {
            return FAST_CLEAR_INVALID_OTHER;
         }
      }
   }

   return FAST_CLEAR_VALID;
}

static void
radv_fast_clear_depth(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview,
                      VkClearDepthStencilValue clear_value, VkImageAspectFlags aspects,
                      struct radv_clear_sync_info *sync_info, enum att_id att_id, bool handle_zs_aspect_hazard)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   uint32_t clear_word, flush_bits;

   clear_word = radv_get_htile_fast_clear_value(device, iview->image, clear_value);

   VkImageSubresourceRange range = {
      .aspectMask = aspects,
      .baseMipLevel = iview->vk.base_mip_level,
      .levelCount = iview->vk.level_count,
      .baseArrayLayer = iview->vk.base_array_layer,
      .layerCount = iview->vk.layer_count,
   };

   /* Wait for any previous clears or draws accessing HTILE. */
   if (sync_info)
      sync_for_clears(cmd_buffer, sync_info, att_id, false, 0, range.layerCount);

   flush_bits = radv_clear_htile_rmw(cmd_buffer, iview->image, &range, clear_word, RADV_HTILE_PRESERVE_VRS);

   if (handle_zs_aspect_hazard && iview->image->planes[0].surface.has_stencil &&
       !(aspects == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) {
      /* Synchronize after performing a depth-only or a stencil-only
       * fast clear because the driver uses an optimized path which
       * performs a read-modify-write operation, and the two separate
       * aspects might use the same HTILE memory.
       */
      cmd_buffer->state.flush_bits |= flush_bits;
   }

   radv_update_ds_clear_metadata(cmd_buffer, iview, clear_value, aspects);

   if (sync_info)
      set_pending_flush_flags(&sync_info->att[att_id].compute_dma, flush_bits, 0, range.layerCount);
}

static uint32_t
radv_get_cmask_fast_clear_value(const struct radv_image *image)
{
   /* Default value when no DCC. */
   uint32_t value = image->vk.samples > 1 ? CMASK_MSAA_FMASK_CLEAR_0_COLOR_CLEAR_REG : CMASK_NOAA_COLOR_CLEAR_REG;

   /* The fast-clear value is different for images that have both DCC and
    * CMASK metadata.
    */
   if (radv_image_has_dcc(image)) {
      return image->vk.samples > 1 ? CMASK_MSAA_FMASK_CLEAR_0_COLOR_EXPANDED : CMASK_NOAA_COLOR_EXPANDED;
   }

   return value;
}

uint32_t
radv_clear_cmask(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, const VkImageSubresourceRange *range,
                 uint32_t value)
{
   uint64_t offset, size;

   radv_image_get_cmask_clear_copy_buffer_range(image, range->baseArrayLayer,
                                                vk_image_subresource_layer_count(&image->vk, range),
                                                &offset, &size);
   return radv_fill_image(cmd_buffer, image, offset, size, value);
}

uint32_t
radv_clear_fmask(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, const VkImageSubresourceRange *range,
                 uint32_t value)
{
   uint64_t fmask_offset = image->planes[0].surface.fmask_offset;
   unsigned slice_size = image->planes[0].surface.fmask_slice_size;
   uint64_t size;

   /* MSAA images do not support mipmap levels. */
   assert(range->baseMipLevel == 0 && vk_image_subresource_level_count(&image->vk, range) == 1);

   fmask_offset += slice_size * range->baseArrayLayer;
   size = slice_size * vk_image_subresource_layer_count(&image->vk, range);

   return radv_fill_image(cmd_buffer, image, fmask_offset, size, value);
}

uint32_t
radv_clear_dcc(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, const VkImageSubresourceRange *range,
               uint32_t value)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   uint32_t level_count = vk_image_subresource_level_count(&image->vk, range);
   uint32_t layer_count = vk_image_subresource_layer_count(&image->vk, range);
   uint32_t flush_bits = 0;

   for (uint32_t l = 0; l < level_count; l++) {
      uint64_t dcc_offset = image->planes[0].surface.meta_offset;
      uint32_t level = range->baseMipLevel + l;
      uint64_t size;

      if (pdev->info.gfx_level >= GFX10) {
         /* DCC for mipmaps+layers is currently disabled. */
         dcc_offset += image->planes[0].surface.meta_slice_size * range->baseArrayLayer +
                       image->planes[0].surface.u.gfx9.meta_levels[level].offset;
         size = image->planes[0].surface.u.gfx9.meta_levels[level].size * layer_count;
      } else if (pdev->info.gfx_level == GFX9) {
         /* Mipmap levels and layers aren't implemented. */
         assert(level == 0);
         size = image->planes[0].surface.meta_size;
      } else {
         const struct legacy_surf_dcc_level *dcc_level = &image->planes[0].surface.u.legacy.color.dcc_level[level];

         /* If dcc_fast_clear_size is 0 (which might happens for
          * mipmaps) the fill buffer operation below is a no-op.
          * This can only happen during initialization as the
          * fast clear path fallbacks to slow clears if one
          * level can't be fast cleared.
          */
         dcc_offset += dcc_level->dcc_offset + dcc_level->dcc_slice_fast_clear_size * range->baseArrayLayer;
         size = dcc_level->dcc_slice_fast_clear_size * vk_image_subresource_layer_count(&image->vk, range);
      }

      /* Do not clear this level if it can't be compressed. */
      if (!size)
         continue;

      flush_bits |= radv_fill_image(cmd_buffer, image, dcc_offset, size, value);
   }

   return flush_bits;
}

static VkResult
get_clear_dcc_comp_to_single_pipeline_layout(struct radv_device *device, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_CLEAR_DCC_COMP_TO_SINGLE;

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 1,
      .pBindings = &binding,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 24,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key, sizeof(key),
                                      layout_out);
}

struct radv_clear_dcc_comp_to_single_key {
   enum radv_meta_object_key_type type;
   bool is_msaa;
};

static VkResult
get_clear_dcc_comp_to_single_pipeline(struct radv_device *device, bool is_msaa, VkPipeline *pipeline_out,
                                      VkPipelineLayout *layout_out)
{
   struct radv_clear_dcc_comp_to_single_key key;
   VkResult result;

   result = get_clear_dcc_comp_to_single_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_CLEAR_DCC_COMP_TO_SINGLE;
   key.is_msaa = is_msaa;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_clear_dcc_comp_to_single_shader(is_msaa);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static uint32_t
radv_clear_dcc_comp_to_single(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image,
                              const VkImageSubresourceRange *range, uint32_t color_values[4])
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   unsigned bytes_per_pixel = vk_format_get_blocksize(image->vk.format);
   unsigned layer_count = vk_image_subresource_layer_count(&image->vk, range);
   bool is_msaa = image->vk.samples > 1;
   struct radv_image_view iview;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;
   VkFormat format;

   switch (bytes_per_pixel) {
   case 1:
      format = VK_FORMAT_R8_UINT;
      break;
   case 2:
      format = VK_FORMAT_R16_UINT;
      break;
   case 4:
      format = VK_FORMAT_R32_UINT;
      break;
   case 8:
      format = VK_FORMAT_R32G32_UINT;
      break;
   case 16:
      format = VK_FORMAT_R32G32B32A32_UINT;
      break;
   default:
      UNREACHABLE("Unsupported number of bytes per pixel");
   }

   result = get_clear_dcc_comp_to_single_pipeline(device, is_msaa, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return 0;
   }

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   for (uint32_t l = 0; l < vk_image_subresource_level_count(&image->vk, range); l++) {
      uint32_t width, height;

      /* Do not write the clear color value for levels without DCC. */
      if (!radv_dcc_enabled(image, range->baseMipLevel + l))
         continue;

      width = u_minify(image->vk.extent.width, range->baseMipLevel + l);
      height = u_minify(image->vk.extent.height, range->baseMipLevel + l);

      const VkImageViewUsage2CreateInfoKHR iview_usage_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_2_CREATE_INFO_KHR,
         .usage = VK_IMAGE_USAGE_2_STORAGE_BIT_KHR,
      };

      radv_image_view_init(&iview, device,
                           &(VkImageViewCreateInfo){
                              .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                              .pNext = &iview_usage_info,
                              .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                              .image = radv_image_to_handle(image),
                              .viewType = VK_IMAGE_VIEW_TYPE_2D,
                              .format = format,
                              .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                   .baseMipLevel = range->baseMipLevel + l,
                                                   .levelCount = 1,
                                                   .baseArrayLayer = range->baseArrayLayer,
                                                   .layerCount = layer_count},
                           },
                           &(struct radv_image_view_extra_create_info){.disable_compression = true});

      radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1,
                                 (VkDescriptorGetInfoEXT[]){{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                                             .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                             .data.pStorageImage = (VkDescriptorImageInfo[]){
                                                                {
                                                                   .sampler = VK_NULL_HANDLE,
                                                                   .imageView = radv_image_view_to_handle(&iview),
                                                                   .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                                },
                                                             }}});

      unsigned dcc_width = DIV_ROUND_UP(width, image->planes[0].surface.u.gfx9.color.dcc_block_width);
      unsigned dcc_height = DIV_ROUND_UP(height, image->planes[0].surface.u.gfx9.color.dcc_block_height);

      const unsigned constants[6] = {
         image->planes[0].surface.u.gfx9.color.dcc_block_width,
         image->planes[0].surface.u.gfx9.color.dcc_block_height,
         color_values[0],
         color_values[1],
         color_values[2],
         color_values[3],
      };

      radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants);

      radv_unaligned_dispatch(cmd_buffer, dcc_width, dcc_height, layer_count);

      radv_image_view_finish(&iview);
   }

   return AC_BARRIER_SYNC_CS | radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                     VK_ACCESS_2_SHADER_WRITE_BIT, 0, image, NULL);
}

uint32_t
radv_clear_htile_rmw(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image,
                     const VkImageSubresourceRange *range, uint32_t value, enum radv_meta_htile_preserve_flags flags)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const bool zrange_precision_wa =
      range->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT && !(flags & RADV_HTILE_PRESERVE_HIZ_RANGE) &&
      flags & RADV_HTILE_DYNAMIC_ZRANGE_PRECISION && cmd_buffer->qf != RADV_QUEUE_TRANSFER &&
      radv_image_has_tc_compat_zrange_metadata(device, image) &&
      radv_tc_compat_htile_enabled(image, range->baseMipLevel);
   const uint32_t layer_count = vk_image_subresource_layer_count(&image->vk, range);
   uint32_t level_count = vk_image_subresource_level_count(&image->vk, range);
   uint32_t flush_bits = 0;
   uint32_t htile_mask;

   /* The ZRANGE_PRECISION workaround that masks the zbase HTILE field in the RMW shader only works
    * if zbase is the maximum value because the shader only zeroes it.
    */
   assert(!zrange_precision_wa || (ac_htile_dword){.dword = value}.zs.zbase == 0x3FFF);

   flags &= ~RADV_HTILE_DYNAMIC_ZRANGE_PRECISION;

   htile_mask = radv_get_htile_mask(cmd_buffer, image, range->aspectMask, flags);

   if (!htile_mask)
      return 0;

   /* Invalidate the VMEM cache for read-modify-write HTILE clears because HTILE could be cached in VMEM
    * from previous such clears, causing VMEM loads to return stale values. This happens when HTILE bits
    * of depth and stencil are cleared separately - when clearing stencil bits, stale depth bits predating
    * the depth bits clear could be returned from the VMEM cache.
    */
   cmd_buffer->state.flush_bits |=
      radv_dst_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            htile_mask != UINT32_MAX ? VK_ACCESS_2_SHADER_STORAGE_READ_BIT : 0, 0, image, range);

   /* If we have 2 clear values and multiple levels, then we need to iterate levels because the clear
    * value selection is per level.
    */
   if (level_count != image->vk.mip_levels || (image->vk.mip_levels > 1 && zrange_precision_wa)) {
      assert(pdev->info.gfx_level >= GFX10);

      /* Clear individuals levels separately. Each layer is a separate mip tree, so it must be cleared
       * separately too.
       */
      for (uint32_t layer = 0; layer < layer_count; layer++) {
         for (uint32_t l = 0; l < level_count; l++) {
            uint32_t level = range->baseMipLevel + l;
            uint64_t htile_offset = image->planes[0].surface.meta_offset +
                                    image->planes[0].surface.meta_slice_size * (range->baseArrayLayer + layer) +
                                    image->planes[0].surface.u.gfx9.meta_levels[level].offset;
            uint32_t size = image->planes[0].surface.u.gfx9.meta_levels[level].size;

            /* Do not clear this level if HTILE is disabled for it. */
            if (!size)
               continue;

            if (htile_mask == UINT32_MAX && !zrange_precision_wa) {
               /* Clear the whole HTILE buffer. */
               flush_bits |= radv_fill_image(cmd_buffer, image, htile_offset, size, value);
            } else {
               /* Only clear depth or stencil bytes in the HTILE buffer. */
               flush_bits |= clear_htile_mask(cmd_buffer, image, image->bindings[0].bo,
                                              image->bindings[0].addr + htile_offset, size, value, htile_mask,
                                              zrange_precision_wa ? radv_get_tc_compat_zrange_va(image, level) : 0);
            }
         }
      }
   } else {
      uint64_t size = image->planes[0].surface.meta_slice_size * layer_count;
      uint64_t htile_offset =
         image->planes[0].surface.meta_offset + image->planes[0].surface.meta_slice_size * range->baseArrayLayer;

      if (htile_mask == UINT32_MAX && !zrange_precision_wa) {
         /* Clear the whole HTILE buffer. */
         flush_bits = radv_fill_image(cmd_buffer, image, htile_offset, size, value);
      } else {
         /* Only clear depth or stencil bytes in the HTILE buffer. */
         flush_bits =
            clear_htile_mask(cmd_buffer, image, image->bindings[0].bo, image->bindings[0].addr + htile_offset, size,
                             value, htile_mask, zrange_precision_wa ? radv_get_tc_compat_zrange_va(image, 0) : 0);
      }
   }

   return flush_bits;
}

uint32_t
radv_clear_htile_rmw_to_uncompressed(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image,
                                     const VkImageSubresourceRange *range, enum radv_meta_htile_preserve_flags preserve)
{
   const struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);

   return radv_clear_htile_rmw(cmd_buffer, image, range, radv_get_htile_initial_value(device, image), preserve);
}

static uint32_t
radv_dcc_single_clear_value(const struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   return pdev->info.gfx_level >= GFX11 ? GFX11_DCC_CLEAR_SINGLE : GFX9_DCC_CLEAR_SINGLE;
}

static void
gfx8_get_fast_clear_parameters(struct radv_device *device, const struct radv_image_view *iview,
                               const VkClearColorValue *clear_value, uint32_t *reset_value,
                               bool *can_avoid_fast_clear_elim)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   bool values[4] = {0};
   int extra_channel;
   bool main_value = false;
   bool extra_value = false;
   bool has_color = false;
   bool has_alpha = false;

   /* comp-to-single allows to perform DCC fast clears without requiring a FCE. */
   if (iview->image->support_comp_to_single) {
      *reset_value = GFX9_DCC_CLEAR_SINGLE;
      *can_avoid_fast_clear_elim = true;
   } else {
      *reset_value = GFX8_DCC_CLEAR_REG;
      *can_avoid_fast_clear_elim = false;
   }

   const struct util_format_description *desc = radv_format_description(iview->vk.format);
   if (iview->vk.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || iview->vk.format == VK_FORMAT_R5G6B5_UNORM_PACK16 ||
       iview->vk.format == VK_FORMAT_B5G6R5_UNORM_PACK16)
      extra_channel = -1;
   else if (desc->layout == UTIL_FORMAT_LAYOUT_PLAIN) {
      if (ac_alpha_is_on_msb(&pdev->info, radv_format_to_pipe_format(iview->vk.format)))
         extra_channel = desc->nr_channels - 1;
      else
         extra_channel = 0;
   } else
      return;

   for (int i = 0; i < 4; i++) {
      int index = desc->swizzle[i] - PIPE_SWIZZLE_X;
      if (desc->swizzle[i] < PIPE_SWIZZLE_X || desc->swizzle[i] > PIPE_SWIZZLE_W)
         continue;

      if (desc->channel[i].pure_integer && desc->channel[i].type == UTIL_FORMAT_TYPE_SIGNED) {
         /* Use the maximum value for clamping the clear color. */
         int max = BITFIELD_MASK(desc->channel[i].size - 1);

         values[i] = clear_value->int32[i] != 0;
         if (clear_value->int32[i] != 0 && MIN2(clear_value->int32[i], max) != max)
            return;
      } else if (desc->channel[i].pure_integer && desc->channel[i].type == UTIL_FORMAT_TYPE_UNSIGNED) {
         /* Use the maximum value for clamping the clear color. */
         unsigned max = BITFIELD_MASK(desc->channel[i].size);

         values[i] = clear_value->uint32[i] != 0U;
         if (clear_value->uint32[i] != 0U && MIN2(clear_value->uint32[i], max) != max)
            return;
      } else {
         values[i] = clear_value->float32[i] != 0.0F;
         if (clear_value->float32[i] != 0.0F && clear_value->float32[i] != 1.0F)
            return;
      }

      if (index == extra_channel) {
         extra_value = values[i];
         has_alpha = true;
      } else {
         main_value = values[i];
         has_color = true;
      }
   }

   /* If alpha isn't present, make it the same as color, and vice versa. */
   if (!has_alpha)
      extra_value = main_value;
   else if (!has_color)
      main_value = extra_value;

   for (int i = 0; i < 4; ++i)
      if (values[i] != main_value && desc->swizzle[i] - PIPE_SWIZZLE_X != extra_channel &&
          desc->swizzle[i] >= PIPE_SWIZZLE_X && desc->swizzle[i] <= PIPE_SWIZZLE_W)
         return;

   /* Only DCC clear code 0000 is allowed for signed<->unsigned formats. */
   if ((main_value || extra_value) && iview->image->dcc_sign_reinterpret)
      return;

   *can_avoid_fast_clear_elim = true;

   if (main_value) {
      if (extra_value)
         *reset_value = GFX8_DCC_CLEAR_1111;
      else
         *reset_value = GFX8_DCC_CLEAR_1110;
   } else {
      if (extra_value)
         *reset_value = GFX8_DCC_CLEAR_0001;
      else
         *reset_value = DCC_CLEAR_0000;
   }
}

static bool
gfx11_get_fast_clear_parameters(struct radv_device *device, const struct radv_image_view *iview,
                                const VkClearColorValue *clear_value, uint32_t *reset_value)
{
   const struct util_format_description *desc = radv_format_description(iview->vk.format);
   unsigned start_bit = UINT_MAX;
   unsigned end_bit = 0;

   /* Find the used bit range. */
   for (unsigned i = 0; i < 4; i++) {
      unsigned swizzle = desc->swizzle[i];

      if (swizzle >= PIPE_SWIZZLE_0)
         continue;

      start_bit = MIN2(start_bit, desc->channel[swizzle].shift);
      end_bit = MAX2(end_bit, desc->channel[swizzle].shift + desc->channel[swizzle].size);
   }

   union {
      uint8_t ub[16];
      uint16_t us[8];
      uint32_t ui[4];
   } value;
   memset(&value, 0, sizeof(value));
   util_format_pack_rgba(radv_format_to_pipe_format(iview->vk.format), &value, clear_value, 1);

   /* Check the cases where all components or bits are either all 0 or all 1. */
   bool all_bits_are_0 = true;
   bool all_bits_are_1 = true;
   bool all_words_are_fp16_1 = false;
   bool all_words_are_fp32_1 = false;

   for (unsigned i = start_bit; i < end_bit; i++) {
      bool bit = value.ub[i / 8] & BITFIELD_BIT(i % 8);

      all_bits_are_0 &= !bit;
      all_bits_are_1 &= bit;
   }

   if (start_bit % 16 == 0 && end_bit % 16 == 0) {
      all_words_are_fp16_1 = true;
      for (unsigned i = start_bit / 16; i < end_bit / 16; i++)
         all_words_are_fp16_1 &= value.us[i] == 0x3c00;
   }

   if (start_bit % 32 == 0 && end_bit % 32 == 0) {
      all_words_are_fp32_1 = true;
      for (unsigned i = start_bit / 32; i < end_bit / 32; i++)
         all_words_are_fp32_1 &= value.ui[i] == 0x3f800000;
   }

   if (all_bits_are_0 || all_bits_are_1 || all_words_are_fp16_1 || all_words_are_fp32_1) {
      if (all_bits_are_0)
         *reset_value = DCC_CLEAR_0000;
      else if (all_bits_are_1)
         *reset_value = GFX11_DCC_CLEAR_1111_UNORM;
      else if (all_words_are_fp16_1)
         *reset_value = GFX11_DCC_CLEAR_1111_FP16;
      else if (all_words_are_fp32_1)
         *reset_value = GFX11_DCC_CLEAR_1111_FP32;
      return true;
   }

   if (desc->nr_channels == 2 && desc->channel[0].size == 8) {
      if (value.ub[0] == 0x00 && value.ub[1] == 0xff) {
         *reset_value = GFX11_DCC_CLEAR_0001_UNORM;
         return true;
      } else if (value.ub[0] == 0xff && value.ub[1] == 0x00) {
         *reset_value = GFX11_DCC_CLEAR_1110_UNORM;
         return true;
      }
   } else if (desc->nr_channels == 4 && desc->channel[0].size == 8) {
      if (value.ub[0] == 0x00 && value.ub[1] == 0x00 && value.ub[2] == 0x00 && value.ub[3] == 0xff) {
         *reset_value = GFX11_DCC_CLEAR_0001_UNORM;
         return true;
      } else if (value.ub[0] == 0xff && value.ub[1] == 0xff && value.ub[2] == 0xff && value.ub[3] == 0x00) {
         *reset_value = GFX11_DCC_CLEAR_1110_UNORM;
         return true;
      }
   } else if (desc->nr_channels == 4 && desc->channel[0].size == 16) {
      if (value.us[0] == 0x0000 && value.us[1] == 0x0000 && value.us[2] == 0x0000 && value.us[3] == 0xffff) {
         *reset_value = GFX11_DCC_CLEAR_0001_UNORM;
         return true;
      } else if (value.us[0] == 0xffff && value.us[1] == 0xffff && value.us[2] == 0xffff && value.us[3] == 0x0000) {
         *reset_value = GFX11_DCC_CLEAR_1110_UNORM;
         return true;
      }
   }

   if (iview->image->support_comp_to_single) {
      *reset_value = GFX11_DCC_CLEAR_SINGLE;
      return true;
   }

   return false;
}

/* If FAST_CLEAR_INVALID_SPLIT_LEVEL_RANGE is returned, only "max_levels" number of levels can be fast-cleared. */
static enum radv_fast_clear_support
radv_can_fast_clear_color(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview,
                          VkImageLayout image_layout, const VkClearRect *clear_rect, VkClearColorValue clear_value,
                          uint32_t view_mask, unsigned *max_levels)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   uint32_t clear_color[2];

   if (pdev->drirc.performance.image_meta_path == RADV_IMAGE_META_PATH_FRAGMENT ||
       pdev->drirc.performance.image_meta_path == RADV_IMAGE_META_PATH_COMPUTE)
      return FAST_CLEAR_INVALID_OTHER;

   if (!iview || !iview->support_fast_clear)
      return FAST_CLEAR_INVALID_OTHER;

   if (!radv_layout_can_fast_clear(device, iview->image, iview->vk.base_mip_level, image_layout,
                                   radv_image_queue_family_mask(iview->image, cmd_buffer->qf)))
      return FAST_CLEAR_INVALID_OTHER;

   if (!radv_is_clear_rect_full(iview, clear_rect, view_mask, !iview->image->support_comp_to_single))
      return FAST_CLEAR_INVALID_OTHER;

   /* DCC */

   /* Images that support comp-to-single clears don't have clear values. */
   if (!iview->image->support_comp_to_single) {
      if (!radv_format_pack_clear_color(iview->vk.format, clear_color, &clear_value))
         return FAST_CLEAR_INVALID_OTHER;

      if (!radv_image_has_clear_value(iview->image) && (clear_color[0] != 0 || clear_color[1] != 0))
         return FAST_CLEAR_INVALID_OTHER;
   }

   if (radv_dcc_enabled(iview->image, iview->vk.base_mip_level)) {
      bool can_avoid_fast_clear_elim;
      uint32_t reset_value;

      if (pdev->info.gfx_level >= GFX11) {
         if (!gfx11_get_fast_clear_parameters(device, iview, &clear_value, &reset_value))
            return FAST_CLEAR_INVALID_OTHER;
      } else {
         gfx8_get_fast_clear_parameters(device, iview, &clear_value, &reset_value, &can_avoid_fast_clear_elim);
      }

      if (iview->image->vk.mip_levels > 1) {
         if (pdev->info.gfx_level >= GFX9) {
            const unsigned last_level = iview->vk.base_mip_level + iview->vk.level_count - 1;
            const unsigned num_meta_levels = iview->image->planes[0].surface.num_meta_levels;

            if (last_level >= num_meta_levels) {
               if (iview->vk.base_mip_level < num_meta_levels) {
                  if (max_levels)
                     *max_levels = num_meta_levels - iview->vk.base_mip_level;
                  return FAST_CLEAR_INVALID_SPLIT_LEVEL_RANGE;
               } else {
                  return FAST_CLEAR_INVALID_OTHER;
               }
            }
         } else {
            for (uint32_t l = 0; l < iview->vk.level_count; l++) {
               uint32_t level = iview->vk.base_mip_level + l;
               struct legacy_surf_dcc_level *dcc_level =
                  &iview->image->planes[0].surface.u.legacy.color.dcc_level[level];

               /* Do not fast clears if one level can't be
                * fast cleared.
                */
               if (!dcc_level->dcc_fast_clear_size) {
                  if (l > 0) {
                     if (max_levels)
                        *max_levels = l;
                     return FAST_CLEAR_INVALID_SPLIT_LEVEL_RANGE;
                  } else {
                     return FAST_CLEAR_INVALID_OTHER;
                  }
               }
            }
         }
      }
   }

   return FAST_CLEAR_VALID;
}

static void
radv_fast_clear_color(struct radv_cmd_buffer *cmd_buffer, const struct radv_image_view *iview,
                      const VkClearAttachment *clear_att, const VkClearRect *clear_rect, uint32_t view_mask,
                      struct radv_clear_sync_info *sync_info, enum att_id att_id)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   VkClearColorValue clear_value = clear_att->clearValue.color;
   uint32_t clear_color[4], flush_bits = 0;
   uint32_t cmask_clear_value;

   VkImageSubresourceRange range = {
      .aspectMask = iview->vk.aspects,
      .baseMipLevel = iview->vk.base_mip_level,
      .levelCount = iview->vk.level_count,
      .baseArrayLayer = iview->vk.base_array_layer + clear_rect->baseArrayLayer,
      .layerCount = view_mask ? util_last_bit(view_mask) : clear_rect->layerCount,
   };

   /* Wait for previous clears accessing DCC/CMASK. */
   if (sync_info)
      sync_for_clears(cmd_buffer, sync_info, att_id, false, clear_rect->baseArrayLayer, range.layerCount);

   /* DCC */
   radv_format_pack_clear_color(iview->vk.format, clear_color, &clear_value);

   cmask_clear_value = radv_get_cmask_fast_clear_value(iview->image);

   /* clear cmask buffer */
   bool need_decompress_pass = false;
   if (radv_dcc_enabled(iview->image, iview->vk.base_mip_level)) {
      uint32_t reset_value;
      bool can_avoid_fast_clear_elim = true;

      if (pdev->info.gfx_level >= GFX11) {
         ASSERTED bool result = gfx11_get_fast_clear_parameters(device, iview, &clear_value, &reset_value);
         assert(result);
      } else {
         gfx8_get_fast_clear_parameters(device, iview, &clear_value, &reset_value, &can_avoid_fast_clear_elim);
      }

      if (radv_image_has_cmask(iview->image)) {
         flush_bits = radv_clear_cmask(cmd_buffer, iview->image, &range, cmask_clear_value);
      }

      if (!can_avoid_fast_clear_elim)
         need_decompress_pass = true;

      flush_bits |= radv_clear_dcc(cmd_buffer, iview->image, &range, reset_value);

      radv_update_dcc_metadata(cmd_buffer, iview->image, &range, true);

      if (reset_value == radv_dcc_single_clear_value(device)) {
         /* Write the clear color to the first byte of each 256B block when the image supports DCC
          * fast clears with comp-to-single.
          */
         if (vk_format_get_blocksize(iview->image->vk.format) == 16) {
            flush_bits |= radv_clear_dcc_comp_to_single(cmd_buffer, iview->image, &range, clear_value.uint32);
         } else {
            clear_color[2] = clear_color[3] = 0;
            flush_bits |= radv_clear_dcc_comp_to_single(cmd_buffer, iview->image, &range, clear_color);
         }
      }
   } else {
      flush_bits = radv_clear_cmask(cmd_buffer, iview->image, &range, cmask_clear_value);

      /* Fast clearing with CMASK should always be eliminated. */
      need_decompress_pass = true;
   }

   if (sync_info) {
      set_pending_flush_flags(&sync_info->att[att_id].compute_dma, flush_bits, clear_rect->baseArrayLayer,
                              range.layerCount);
   }

   /* Update the FCE predicate to perform a fast-clear eliminate. */
   radv_update_fce_metadata(cmd_buffer, iview->image, &range, need_decompress_pass);

   radv_update_color_clear_metadata(cmd_buffer, iview, clear_att->colorAttachment, clear_color);
}

/**
 * The parameters mean that same as those in vkCmdClearAttachments.
 */
static void
emit_clear(struct radv_cmd_buffer *cmd_buffer, const VkClearAttachment *clear_att, const VkClearRect *clear_rect,
           uint32_t view_mask, struct radv_clear_sync_info *sync_info, enum att_id att_id)
{
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   const unsigned conservative_layer_count = view_mask ? util_last_bit(view_mask) : clear_rect->layerCount;
   VkImageAspectFlags aspects = clear_att->aspectMask;

   if (aspects & VK_IMAGE_ASPECT_COLOR_BIT) {
      assert(clear_att->colorAttachment < render->color_att_count);
      const struct radv_attachment *color_att = &render->color_att[clear_att->colorAttachment];

      if (color_att->format == VK_FORMAT_UNDEFINED)
         return;

      VkClearColorValue clear_value = clear_att->clearValue.color;

      if (radv_can_fast_clear_color(cmd_buffer, color_att->iview, color_att->layout, clear_rect, clear_value, view_mask,
                                    NULL) == FAST_CLEAR_VALID) {
         radv_fast_clear_color(cmd_buffer, color_att->iview, clear_att, clear_rect, view_mask, sync_info, att_id);
      } else {
         /* Wait for any previous compute and CP DMA clears accessing DCC/CMASK/FMASK or the color image. */
         if (sync_info)
            sync_for_clears(cmd_buffer, sync_info, att_id, true, clear_rect->baseArrayLayer, conservative_layer_count);

         emit_render_color_clear(cmd_buffer, clear_att, clear_rect, view_mask);

         /* Track pending draws for any future compute/CP DMA clear. */
         if (sync_info && color_att->iview) {
            set_pending_draw_flush_flags(cmd_buffer, sync_info, color_att, att_id, clear_rect->baseArrayLayer,
                                         conservative_layer_count);
         }
      }
   } else {
      const struct radv_attachment *ds_att = &render->ds_att;

      if (ds_att->format == VK_FORMAT_UNDEFINED)
         return;

      /* Ignore aspects that have no effects. */
      aspects &= render->ds_att_aspects;
      if (!aspects)
         return;

      VkClearDepthStencilValue clear_value = clear_att->clearValue.depthStencil;
      assert(aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));

      /* Determine fast Z/S clear eligibility. */
      VkImageAspectFlags fast_clear_aspects = 0;

      if (aspects == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) &&
          ds_att->layout != ds_att->stencil_layout) {
         if (radv_can_fast_clear_depth(cmd_buffer, ds_att->iview, ds_att->layout, aspects, clear_rect, clear_value,
                                       view_mask, NULL, NULL) == FAST_CLEAR_VALID)
            fast_clear_aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;

         if (radv_can_fast_clear_depth(cmd_buffer, ds_att->iview, ds_att->stencil_layout, aspects, clear_rect,
                                       clear_value, view_mask, NULL, NULL) == FAST_CLEAR_VALID)
            fast_clear_aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;
      } else {
         VkImageLayout layout = aspects & VK_IMAGE_ASPECT_DEPTH_BIT ? ds_att->layout : ds_att->stencil_layout;

         if (radv_can_fast_clear_depth(cmd_buffer, ds_att->iview, layout, aspects, clear_rect, clear_value, view_mask,
                                       NULL, NULL) == FAST_CLEAR_VALID)
            fast_clear_aspects = aspects;
      }

      /* First do the fast clear, then do the slow clear. */
      const VkImageAspectFlags slow_clear_aspects = aspects & ~fast_clear_aspects;

      if (fast_clear_aspects) {
         radv_fast_clear_depth(cmd_buffer, ds_att->iview, clear_att->clearValue.depthStencil, fast_clear_aspects,
                               sync_info, att_id, false);
      }

      if (slow_clear_aspects) {
         if (sync_info)
            sync_for_clears(cmd_buffer, sync_info, att_id, true, clear_rect->baseArrayLayer, conservative_layer_count);

         emit_depthstencil_clear(cmd_buffer, clear_att->clearValue.depthStencil, slow_clear_aspects, clear_rect,
                                 view_mask);

         /* Track pending draws for any future compute/CP DMA clear. */
         if (sync_info && ds_att->iview) {
            set_pending_draw_flush_flags(cmd_buffer, sync_info, ds_att, att_id, clear_rect->baseArrayLayer,
                                         conservative_layer_count);
         }
      }
   }
}

static bool
radv_rendering_needs_clear(const VkRenderingInfo *pRenderingInfo)
{
   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      if (pRenderingInfo->pColorAttachments[i].imageView != VK_NULL_HANDLE &&
          pRenderingInfo->pColorAttachments[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         return true;
   }

   if (pRenderingInfo->pDepthAttachment != NULL && pRenderingInfo->pDepthAttachment->imageView != VK_NULL_HANDLE &&
       pRenderingInfo->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
      return true;

   if (pRenderingInfo->pStencilAttachment != NULL && pRenderingInfo->pStencilAttachment->imageView != VK_NULL_HANDLE &&
       pRenderingInfo->pStencilAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
      return true;

   return false;
}

static void
radv_subpass_clear_attachment(struct radv_cmd_buffer *cmd_buffer, const VkClearAttachment *clear_att,
                              struct radv_clear_sync_info *sync_info, enum att_id att_id)
{
   const struct radv_rendering_state *render = &cmd_buffer->state.render;

   VkClearRect clear_rect = {
      .rect = render->area,
      .baseArrayLayer = 0,
      .layerCount = render->layer_count,
   };

   radv_describe_begin_render_pass_clear(cmd_buffer, clear_att->aspectMask);

   emit_clear(cmd_buffer, clear_att, &clear_rect, render->view_mask, sync_info, att_id);

   radv_describe_end_render_pass_clear(cmd_buffer);
}

/**
 * Emit any pending attachment clears for the current subpass.
 *
 * @see radv_attachment_state::pending_clear_aspects
 */
void
radv_cmd_buffer_clear_rendering(struct radv_cmd_buffer *cmd_buffer, const VkRenderingInfo *pRenderingInfo)
{
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   struct radv_clear_sync_info sync_info = {0};

   if (!radv_rendering_needs_clear(pRenderingInfo))
      return;

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_utrace_begin_clear_rendering(cmd_buffer);

   radv_meta_begin_rendering(cmd_buffer);

   if (render->ds_att.iview != NULL) {
      VkClearAttachment clear_att = {.aspectMask = 0};

      if (pRenderingInfo->pDepthAttachment != NULL && pRenderingInfo->pDepthAttachment->imageView != VK_NULL_HANDLE &&
          pRenderingInfo->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
         clear_att.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
         clear_att.clearValue.depthStencil.depth = pRenderingInfo->pDepthAttachment->clearValue.depthStencil.depth;
      }

      if (pRenderingInfo->pStencilAttachment != NULL &&
          pRenderingInfo->pStencilAttachment->imageView != VK_NULL_HANDLE &&
          pRenderingInfo->pStencilAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
         clear_att.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
         clear_att.clearValue.depthStencil.stencil =
            pRenderingInfo->pStencilAttachment->clearValue.depthStencil.stencil;
      }

      if (clear_att.aspectMask != 0) {
         radv_subpass_clear_attachment(cmd_buffer, &clear_att, &sync_info, ATT_ZS);
      }
   }

   assert(render->color_att_count == pRenderingInfo->colorAttachmentCount);
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].iview == NULL ||
          pRenderingInfo->pColorAttachments[i].loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;

      VkClearAttachment clear_att = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = i,
         .clearValue = pRenderingInfo->pColorAttachments[i].clearValue,
      };

      radv_subpass_clear_attachment(cmd_buffer, &clear_att, &sync_info, ATT_COLOR0 + i);
   }

   radv_meta_end_rendering(cmd_buffer);
   apply_pending_compute_dma_sync_flags(cmd_buffer, &sync_info);

   radv_utrace_end_clear_rendering(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

static void
radv_clear_image_layer(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, VkImageLayout image_layout,
                       const VkImageSubresourceRange *range, VkFormat format, int level, unsigned layer_count,
                       const VkClearValue *clear_val)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_image_view iview;
   uint32_t width = u_minify(image->vk.extent.width, range->baseMipLevel + level);
   uint32_t height = u_minify(image->vk.extent.height, range->baseMipLevel + level);

   const VkImageViewUsage2CreateInfoKHR iview_usage_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_2_CREATE_INFO_KHR,
      .usage = (vk_format_is_color(format) ? VK_IMAGE_USAGE_2_COLOR_ATTACHMENT_BIT_KHR
                                           : VK_IMAGE_USAGE_2_DEPTH_STENCIL_ATTACHMENT_BIT_KHR),
   };

   radv_image_view_init(&iview, device,
                        &(VkImageViewCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                           .pNext = &iview_usage_info,
                           .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                           .image = radv_image_to_handle(image),
                           .viewType = radv_meta_get_view_type(image, true),
                           .format = format,
                           .subresourceRange = {.aspectMask = range->aspectMask,
                                                .baseMipLevel = range->baseMipLevel + level,
                                                .levelCount = 1,
                                                .baseArrayLayer = range->baseArrayLayer,
                                                .layerCount = layer_count},
                        },
                        NULL);

   VkClearAttachment clear_att = {
      .aspectMask = range->aspectMask,
      .colorAttachment = 0,
      .clearValue = *clear_val,
   };

   VkClearRect clear_rect = {
      .rect =
         {
            .offset = {0, 0},
            .extent = {width, height},
         },
      .baseArrayLayer = 0,
      .layerCount = layer_count,
   };

   VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = radv_image_view_to_handle(&iview),
      .imageLayout = image_layout,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };

   VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_LOCAL_READ_CONCURRENT_ACCESS_CONTROL_BIT_KHR,
      .renderArea =
         {
            .offset = {0, 0},
            .extent = {width, height},
         },
      .layerCount = layer_count,
   };

   if (image->vk.aspects & VK_IMAGE_ASPECT_COLOR_BIT) {
      rendering_info.colorAttachmentCount = 1;
      rendering_info.pColorAttachments = &att;
   }
   if (image->vk.aspects & VK_IMAGE_ASPECT_DEPTH_BIT)
      rendering_info.pDepthAttachment = &att;
   if (image->vk.aspects & VK_IMAGE_ASPECT_STENCIL_BIT)
      rendering_info.pStencilAttachment = &att;

   radv_CmdBeginRendering(radv_cmd_buffer_to_handle(cmd_buffer), &rendering_info);

   emit_clear(cmd_buffer, &clear_att, &clear_rect, 0, NULL, ATT_ZS);

   const VkRenderingEndInfoKHR end_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_END_INFO_KHR,
   };

   radv_CmdEndRendering2KHR(radv_cmd_buffer_to_handle(cmd_buffer), &end_info);

   radv_image_view_finish(&iview);
}

/**
 * This fast-clears either the whole range or its subset. If part of the range is not fast-cleared,
 * the function returns the remaining ranges left to clear in the last parameter. It returns the number
 * of returned ranges.
 *
 * The caller should check remaining_rangeN->aspectMask to see whether there is anything left to clear.
 */
static unsigned
radv_fast_clear_range(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, VkFormat format,
                      VkImageLayout image_layout, const VkImageSubresourceRange *range, const VkClearValue *clear_val,
                      VkImageSubresourceRange remaining_ranges[2])
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_image_view iview;

   /* Set the remaining ranges to none. */
   remaining_ranges[0].aspectMask = 0;
   remaining_ranges[1].aspectMask = 0;

   const VkImageViewUsage2CreateInfoKHR iview_usage_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_2_CREATE_INFO_KHR,
      .usage = VK_IMAGE_USAGE_2_STORAGE_BIT_KHR,
   };

   radv_image_view_init(&iview, device,
                        &(VkImageViewCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                           .pNext = &iview_usage_info,
                           .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                           .image = radv_image_to_handle(image),
                           .viewType = radv_meta_get_view_type(image, false),
                           .format = format,
                           .subresourceRange =
                              {
                                 .aspectMask = range->aspectMask,
                                 .baseMipLevel = range->baseMipLevel,
                                 .levelCount = vk_image_subresource_level_count(&image->vk, range),
                                 .baseArrayLayer = range->baseArrayLayer,
                                 .layerCount = vk_image_subresource_layer_count(&image->vk, range),
                              },
                        },
                        NULL);

   VkClearRect clear_rect = {
      .rect =
         {
            .offset = {0, 0},
            .extent =
               {
                  u_minify(image->vk.extent.width, range->baseMipLevel),
                  u_minify(image->vk.extent.height, range->baseMipLevel),
               },
         },
      .baseArrayLayer = 0,
      .layerCount = vk_image_subresource_layer_count(&image->vk, range),
   };

   VkClearAttachment clear_att = {
      .aspectMask = range->aspectMask,
      .colorAttachment = 0,
      .clearValue = *clear_val,
   };

   const bool is_color = !(range->aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
   const unsigned orig_level_count = iview.vk.level_count;
   unsigned max_fast_clear_levels = orig_level_count;
   bool reduce_iview = false;

   /* Check the whole range. */
   enum radv_fast_clear_support support =
      is_color ? radv_can_fast_clear_color(cmd_buffer, &iview, image_layout, &clear_rect, clear_att.clearValue.color, 0,
                                           &max_fast_clear_levels)
               : radv_can_fast_clear_depth(cmd_buffer, &iview, image_layout, clear_att.aspectMask, &clear_rect,
                                           clear_att.clearValue.depthStencil, 0, &clear_att.aspectMask,
                                           &max_fast_clear_levels);

   if (support == FAST_CLEAR_INVALID_SPLIT_ZS_ASPECTS) {
      /* Check only the aspect that can be fast-cleared. This can only return FAST_CLEAR_VALID,
       * FAST_CLEAR_INVALID_SPLIT_LEVEL_RANGE, or FAST_CLEAR_INVALID_OTHER.
       */
      support = radv_can_fast_clear_depth(cmd_buffer, &iview, image_layout, clear_att.aspectMask, &clear_rect,
                                          clear_att.clearValue.depthStencil, 0, NULL, &max_fast_clear_levels);
      assert(support != FAST_CLEAR_INVALID_SPLIT_ZS_ASPECTS);
      /* The image view will lose one aspect. */
      reduce_iview = true;
   }

   if (support == FAST_CLEAR_INVALID_SPLIT_LEVEL_RANGE) {
      support = FAST_CLEAR_VALID;
      /* The image view will lose the trailing levels that can't be fast-cleared. */
      reduce_iview = true;
   }

   if (support != FAST_CLEAR_VALID) {
      radv_image_view_finish(&iview);

      /* The caller should use a slow clear for the whole range. */
      remaining_ranges[0] = *range;
      return 1;
   }

   if (reduce_iview) {
      /* Reduce the aspect mask and/or the level count of the view. */
      radv_image_view_finish(&iview);
      radv_image_view_init(&iview, device,
                           &(VkImageViewCreateInfo){
                              .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                              .pNext = &iview_usage_info,
                              .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                              .image = radv_image_to_handle(image),
                              .viewType = radv_meta_get_view_type(image, false),
                              .format = format,
                              .subresourceRange =
                                 {
                                    .aspectMask = clear_att.aspectMask,
                                    .baseMipLevel = range->baseMipLevel,
                                    .levelCount = max_fast_clear_levels,
                                    .baseArrayLayer = range->baseArrayLayer,
                                    .layerCount = vk_image_subresource_layer_count(&image->vk, range),
                                 },
                           },
                           NULL);

      /* Return the remaining ranges that can't be fast-cleared, so that the caller can slow-clear them. */
      if (range->aspectMask != clear_att.aspectMask) {
         if (orig_level_count != max_fast_clear_levels) {
            /* The whole range of one aspect and the trailing level range of the other aspect remain to be
             * cleared by the caller.
             *
             * The first returned range is just 1 aspect of the leading levels that were not fast-cleared.
             */
            remaining_ranges[0] = *range;
            remaining_ranges[0].aspectMask &= ~clear_att.aspectMask;
            remaining_ranges[0].levelCount = max_fast_clear_levels;

            /* The second returned range is both aspects for the trailing levels that were not fast-cleared. */
            remaining_ranges[1] = *range;
            remaining_ranges[1].baseMipLevel += max_fast_clear_levels;
            remaining_ranges[1].levelCount = orig_level_count - max_fast_clear_levels;
         } else {
            /* Only one aspect remains to be cleared by the caller. */
            remaining_ranges[0] = *range;
            remaining_ranges[0].aspectMask &= ~clear_att.aspectMask;
         }
      } else {
         if (orig_level_count != max_fast_clear_levels) {
            /* Only a trailing level range remains to be cleared by the caller. */
            remaining_ranges[0] = *range;
            remaining_ranges[0].baseMipLevel += max_fast_clear_levels;
            remaining_ranges[0].levelCount = orig_level_count - max_fast_clear_levels;
         } else {
            UNREACHABLE("the view shouldn't have been reduced");
         }
      }
   }

   /* Execute the fast clear for the whole range or whatever is left. */
   if (is_color) {
      radv_fast_clear_color(cmd_buffer, &iview, &clear_att, &clear_rect, 0, NULL, ATT_COLOR0);
   } else {
      radv_fast_clear_depth(cmd_buffer, &iview, clear_att.clearValue.depthStencil, clear_att.aspectMask, NULL, ATT_ZS,
                            true);
   }

   radv_image_view_finish(&iview);
   return !!remaining_ranges[0].aspectMask + !!remaining_ranges[1].aspectMask;
}

static void
radv_cmd_clear_image(VkCommandBuffer commandBuffer, VkImage image_h, VkImageLayout image_layout,
                     const VkClearValue *clear_value, uint32_t range_count, const VkImageSubresourceRange *ranges)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_image, image, image_h);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   VkFormat format = image->vk.format;
   VkClearValue internal_clear_value;
   bool cs = false;

   if (vk_format_is_color(image->vk.format)) {
      cs = cmd_buffer->qf == RADV_QUEUE_COMPUTE || !radv_image_is_renderable(pdev, image) ||
           (pdev->drirc.performance.image_meta_path == RADV_IMAGE_META_PATH_COMPUTE ||
            pdev->drirc.performance.image_meta_path == RADV_IMAGE_META_PATH_FAST_CLEAR_ELSE_COMPUTE);
   }

   if (ranges->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
      internal_clear_value.color = clear_value->color;

      if (vk_format_is_srgb(image->vk.format)) {
         format = vk_format_no_srgb(image->vk.format);

         for (unsigned i = 0; i < 3; i++) {
            internal_clear_value.color.float32[i] = util_format_linear_to_srgb_float(clear_value->color.float32[i]);
         }
      }

      if (format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 && pdev->info.gfx_level < GFX10_3) {
         format = VK_FORMAT_R32_UINT;
         internal_clear_value.color.uint32[0] = float3_to_rgb9e5(clear_value->color.float32);
      }
   } else {
      internal_clear_value.depthStencil = clear_value->depthStencil;
   }

   /* radv_fast_clear_range can return up to 2 ranges per input range. */
   STACK_ARRAY(VkImageSubresourceRange, remaining_ranges, range_count * 2);
   if (!remaining_ranges) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   radv_suspend_conditional_rendering(cmd_buffer);
   radv_meta_begin(cmd_buffer);

   unsigned num_remaining_ranges = 0;

   /* Execute all fast clears together. */
   for (unsigned r = 0; r < range_count; r++) {
      /* If nothing or only a subset of the range is cleared, the function returns 1-2 ranges that remain
       * to be cleared.
       */
      num_remaining_ranges += radv_fast_clear_range(cmd_buffer, image, format, image_layout, &ranges[r],
                                                    &internal_clear_value, &remaining_ranges[num_remaining_ranges]);
   }

   /* Execute all slow clears together. */
   for (unsigned r = 0; r < num_remaining_ranges; r++) {
      const VkImageSubresourceRange *range = &remaining_ranges[r];
      const unsigned level_count = vk_image_subresource_level_count(&image->vk, range);

      for (uint32_t l = 0; l < level_count; ++l) {
         const uint32_t layer_count = image->vk.image_type == VK_IMAGE_TYPE_3D
                                         ? u_minify(image->vk.extent.depth, range->baseMipLevel + l)
                                         : vk_image_subresource_layer_count(&image->vk, range);
         if (cs) {
            for (uint32_t s = 0; s < layer_count; ++s) {
               struct radv_meta_blit2d_surf surf;
               surf.format = format;
               surf.image = image;
               surf.level = range->baseMipLevel + l;
               surf.layer = range->baseArrayLayer + s;
               surf.aspect_mask = range->aspectMask;
               surf.disable_compression = false;
               radv_meta_clear_image_cs(cmd_buffer, &surf, &internal_clear_value.color);
            }
         } else {
            radv_clear_image_layer(cmd_buffer, image, image_layout, range, format, l, layer_count,
                                   &internal_clear_value);
         }
      }
   }

   radv_meta_end(cmd_buffer);
   radv_resume_conditional_rendering(cmd_buffer);

   STACK_ARRAY_FINISH(remaining_ranges);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image_h, VkImageLayout imageLayout,
                        const VkClearColorValue *pColor, uint32_t rangeCount, const VkImageSubresourceRange *pRanges)
{
   radv_cmd_clear_image(commandBuffer, image_h, imageLayout, (const VkClearValue *)pColor, rangeCount, pRanges);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage image_h, VkImageLayout imageLayout,
                               const VkClearDepthStencilValue *pDepthStencil, uint32_t rangeCount,
                               const VkImageSubresourceRange *pRanges)
{
   radv_cmd_clear_image(commandBuffer, image_h, imageLayout, (const VkClearValue *)pDepthStencil, rangeCount, pRanges);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount, const VkClearAttachment *pAttachments,
                         uint32_t rectCount, const VkClearRect *pRects)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   const struct radv_rendering_state *render = &cmd_buffer->state.render;
   VkClearAttachment z_clear = {0}, s_clear = {0}, combined_zs_clear = {0};
   struct radv_clear_sync_info sync_info = {0};

   /* This merges separate Z/S clears into a combined Z/S clear if both Z and S clears occur only once.
    * Combined Z/S clears are more efficient since slow clears get only one draw instead of two and fast
    * clears get one HTILE clear instead of "HTILE RMW -> barrier -> HTILE RMW".
    */
   for (unsigned a = 0; a < attachmentCount; a++) {
      switch (pAttachments[a].aspectMask) {
      case VK_IMAGE_ASPECT_COLOR_BIT:
         continue;
      case VK_IMAGE_ASPECT_DEPTH_BIT:
         if (!z_clear.aspectMask) {
            z_clear = pAttachments[a];
            continue;
         }
         break;
      case VK_IMAGE_ASPECT_STENCIL_BIT:
         if (!s_clear.aspectMask) {
            s_clear = pAttachments[a];
            continue;
         }
         break;
      }

      assert(pAttachments[a].aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
      z_clear.aspectMask = s_clear.aspectMask = 0; /* do nothing */
      break;
   }

   if (z_clear.aspectMask && s_clear.aspectMask) {
      combined_zs_clear.aspectMask = z_clear.aspectMask | s_clear.aspectMask;
      combined_zs_clear.clearValue.depthStencil.depth = z_clear.clearValue.depthStencil.depth;
      combined_zs_clear.clearValue.depthStencil.stencil = s_clear.clearValue.depthStencil.stencil;
   }

   /* Attachments can have draws in flight. Set that there are pending draw flushes, so that compute/CP DMA
    * clears wait.
    */
   if (render->ds_att.iview) {
      set_pending_draw_flush_flags(cmd_buffer, &sync_info, &render->ds_att, ATT_ZS, 0,
                                   render->ds_att.iview->vk.layer_count);
   }

   for (unsigned a = 0; a < render->color_att_count; a++) {
      if (render->color_att[a].iview) {
         set_pending_draw_flush_flags(cmd_buffer, &sync_info, &render->color_att[a], ATT_COLOR0 + a, 0,
                                      render->color_att[a].iview->vk.layer_count);
      }
   }

   radv_meta_begin_rendering(cmd_buffer);

   /* Execute the combined Z/S clear first. */
   if (combined_zs_clear.aspectMask) {
      for (uint32_t r = 0; r < rectCount; ++r)
         emit_clear(cmd_buffer, &combined_zs_clear, &pRects[r], cmd_buffer->state.render.view_mask, &sync_info, ATT_ZS);
   }

   /* Execute the remaining clears. */
   for (uint32_t a = 0; a < attachmentCount; ++a) {
      /* Combined Z/S clears have already been done above. */
      if (combined_zs_clear.aspectMask & pAttachments[a].aspectMask)
         continue;

      const enum att_id att_id = pAttachments[a].aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                                    ? ATT_ZS
                                    : ATT_COLOR0 + pAttachments[a].colorAttachment;

      for (uint32_t r = 0; r < rectCount; ++r)
         emit_clear(cmd_buffer, &pAttachments[a], &pRects[r], cmd_buffer->state.render.view_mask, &sync_info, att_id);
   }

   radv_meta_end_rendering(cmd_buffer);
   apply_pending_compute_dma_sync_flags(cmd_buffer, &sync_info);
}
