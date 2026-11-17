/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "meta/radv_meta.h"
#include "nir/nir.h"
#include "nir/nir_serialize.h"
#include "tools/radv_rmv.h"
#include "util/mesa-blake3.h"
#include "util/os_time.h"
#include "radv_device.h"
#include "radv_entrypoints.h"
#include "radv_formats.h"
#include "radv_physical_device.h"
#include "radv_pipeline_binary.h"
#include "radv_pipeline_cache.h"
#include "radv_shader.h"
#include "shader_enums.h"
#include "vk_pipeline.h"
#include "vk_util.h"

#include "ac_shader_util.h"

static bool
radv_is_static_vrs_enabled(const struct vk_graphics_pipeline_state *state)
{
   if (!state->fsr)
      return false;

   return state->fsr->fragment_size.width != 1 || state->fsr->fragment_size.height != 1 ||
          state->fsr->combiner_ops[0] != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR ||
          state->fsr->combiner_ops[1] != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
}

static bool
radv_pipeline_has_ds_attachments(const struct vk_render_pass_state *rp)
{
   return rp->depth_attachment_format != VK_FORMAT_UNDEFINED || rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;
}

static bool
radv_pipeline_has_color_attachments(const struct vk_render_pass_state *rp)
{
   for (uint32_t i = 0; i < rp->color_attachment_count; ++i) {
      if (rp->color_attachment_formats[i] != VK_FORMAT_UNDEFINED)
         return true;
   }

   return false;
}

static bool
color_outputs_need_ps_epilog(const struct vk_graphics_pipeline_state *state,
                             VkGraphicsPipelineLibraryFlagBitsEXT lib_flags)
{
   /* Use a PS epilog when the fragment shader is compiled without the fragment output interface. */
   if ((state->shader_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
       (lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) &&
       !(lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT))
      return true;

   /* These dynamic states need to compile PS epilogs on-demand. */
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ENABLES) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_WRITE_MASKS) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE))
      return true;

   return false;
}

static bool
radv_pipeline_uses_vrs_attachment(const struct radv_graphics_pipeline *pipeline,
                                  const struct vk_graphics_pipeline_state *state)
{
   VkPipelineCreateFlags2 create_flags = pipeline->base.create_flags;
   if (state->rp)
      create_flags |= state->pipeline_flags;

   return (create_flags & VK_PIPELINE_CREATE_2_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR) != 0;
}

static uint64_t
radv_dynamic_state_mask(VkDynamicState state)
{
   switch (state) {
   case VK_DYNAMIC_STATE_VIEWPORT:
      return RADV_DYNAMIC_VIEWPORT;
   case VK_DYNAMIC_STATE_SCISSOR:
      return RADV_DYNAMIC_SCISSOR;
   case VK_DYNAMIC_STATE_LINE_WIDTH:
      return RADV_DYNAMIC_LINE_WIDTH;
   case VK_DYNAMIC_STATE_DEPTH_BIAS:
      return RADV_DYNAMIC_DEPTH_BIAS;
   case VK_DYNAMIC_STATE_BLEND_CONSTANTS:
      return RADV_DYNAMIC_BLEND_CONSTANTS;
   case VK_DYNAMIC_STATE_DEPTH_BOUNDS:
      return RADV_DYNAMIC_DEPTH_BOUNDS;
   case VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK:
      return RADV_DYNAMIC_STENCIL_COMPARE_MASK;
   case VK_DYNAMIC_STATE_STENCIL_WRITE_MASK:
      return RADV_DYNAMIC_STENCIL_WRITE_MASK;
   case VK_DYNAMIC_STATE_STENCIL_REFERENCE:
      return RADV_DYNAMIC_STENCIL_REFERENCE;
   case VK_DYNAMIC_STATE_DISCARD_RECTANGLE_EXT:
      return RADV_DYNAMIC_DISCARD_RECTANGLE;
   case VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_EXT:
      return RADV_DYNAMIC_SAMPLE_LOCATIONS;
   case VK_DYNAMIC_STATE_LINE_STIPPLE:
      return RADV_DYNAMIC_LINE_STIPPLE;
   case VK_DYNAMIC_STATE_CULL_MODE:
      return RADV_DYNAMIC_CULL_MODE;
   case VK_DYNAMIC_STATE_FRONT_FACE:
      return RADV_DYNAMIC_FRONT_FACE;
   case VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY:
      return RADV_DYNAMIC_PRIMITIVE_TOPOLOGY;
   case VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE:
      return RADV_DYNAMIC_DEPTH_TEST_ENABLE;
   case VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE:
      return RADV_DYNAMIC_DEPTH_WRITE_ENABLE;
   case VK_DYNAMIC_STATE_DEPTH_COMPARE_OP:
      return RADV_DYNAMIC_DEPTH_COMPARE_OP;
   case VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE:
      return RADV_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE;
   case VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE:
      return RADV_DYNAMIC_STENCIL_TEST_ENABLE;
   case VK_DYNAMIC_STATE_STENCIL_OP:
      return RADV_DYNAMIC_STENCIL_OP;
   case VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE:
      return RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE;
   case VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR:
      return RADV_DYNAMIC_FRAGMENT_SHADING_RATE;
   case VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT:
      return RADV_DYNAMIC_PATCH_CONTROL_POINTS;
   case VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE:
      return RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE;
   case VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE:
      return RADV_DYNAMIC_DEPTH_BIAS_ENABLE;
   case VK_DYNAMIC_STATE_LOGIC_OP_EXT:
      return RADV_DYNAMIC_LOGIC_OP;
   case VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE:
      return RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE;
   case VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT:
      return RADV_DYNAMIC_COLOR_WRITE_ENABLE;
   case VK_DYNAMIC_STATE_VERTEX_INPUT_EXT:
      return RADV_DYNAMIC_VERTEX_INPUT;
   case VK_DYNAMIC_STATE_POLYGON_MODE_EXT:
      return RADV_DYNAMIC_POLYGON_MODE;
   case VK_DYNAMIC_STATE_TESSELLATION_DOMAIN_ORIGIN_EXT:
      return RADV_DYNAMIC_TESS_DOMAIN_ORIGIN;
   case VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT:
      return RADV_DYNAMIC_LOGIC_OP_ENABLE;
   case VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT:
      return RADV_DYNAMIC_LINE_STIPPLE_ENABLE;
   case VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT:
      return RADV_DYNAMIC_ALPHA_TO_COVERAGE_ENABLE;
   case VK_DYNAMIC_STATE_SAMPLE_MASK_EXT:
      return RADV_DYNAMIC_SAMPLE_MASK;
   case VK_DYNAMIC_STATE_DEPTH_CLIP_ENABLE_EXT:
      return RADV_DYNAMIC_DEPTH_CLIP_ENABLE;
   case VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT:
      return RADV_DYNAMIC_CONSERVATIVE_RAST_MODE;
   case VK_DYNAMIC_STATE_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT:
      return RADV_DYNAMIC_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE;
   case VK_DYNAMIC_STATE_PROVOKING_VERTEX_MODE_EXT:
      return RADV_DYNAMIC_PROVOKING_VERTEX_MODE;
   case VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT:
      return RADV_DYNAMIC_DEPTH_CLAMP_ENABLE;
   case VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT:
      return RADV_DYNAMIC_COLOR_WRITE_MASK;
   case VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT:
      return RADV_DYNAMIC_COLOR_BLEND_ENABLE;
   case VK_DYNAMIC_STATE_RASTERIZATION_SAMPLES_EXT:
      return RADV_DYNAMIC_RASTERIZATION_SAMPLES;
   case VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT:
      return RADV_DYNAMIC_LINE_RASTERIZATION_MODE;
   case VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT:
      return RADV_DYNAMIC_COLOR_BLEND_EQUATION;
   case VK_DYNAMIC_STATE_DISCARD_RECTANGLE_ENABLE_EXT:
      return RADV_DYNAMIC_DISCARD_RECTANGLE_ENABLE;
   case VK_DYNAMIC_STATE_DISCARD_RECTANGLE_MODE_EXT:
      return RADV_DYNAMIC_DISCARD_RECTANGLE_MODE;
   case VK_DYNAMIC_STATE_ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT:
      return RADV_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE;
   case VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_ENABLE_EXT:
      return RADV_DYNAMIC_SAMPLE_LOCATIONS_ENABLE;
   case VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT:
      return RADV_DYNAMIC_ALPHA_TO_ONE_ENABLE;
   case VK_DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT:
      return RADV_DYNAMIC_DEPTH_CLAMP_RANGE;
   case VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT:
      return RADV_DYNAMIC_VIEWPORT | RADV_DYNAMIC_VIEWPORT_WITH_COUNT;
   case VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT:
      return RADV_DYNAMIC_SCISSOR | RADV_DYNAMIC_SCISSOR_WITH_COUNT;
   default:
      UNREACHABLE("Unhandled dynamic state");
   }
}

#define RADV_DYNAMIC_CB_STATES                                                                                         \
   (RADV_DYNAMIC_LOGIC_OP_ENABLE | RADV_DYNAMIC_LOGIC_OP | RADV_DYNAMIC_COLOR_WRITE_ENABLE |                           \
    RADV_DYNAMIC_COLOR_WRITE_MASK | RADV_DYNAMIC_COLOR_BLEND_ENABLE | RADV_DYNAMIC_COLOR_BLEND_EQUATION |              \
    RADV_DYNAMIC_BLEND_CONSTANTS)

#define RADV_DYNAMIC_DS_STATES                                                                                         \
   (RADV_DYNAMIC_DEPTH_BOUNDS | RADV_DYNAMIC_STENCIL_COMPARE_MASK | RADV_DYNAMIC_STENCIL_WRITE_MASK |                  \
    RADV_DYNAMIC_STENCIL_REFERENCE | RADV_DYNAMIC_DEPTH_TEST_ENABLE | RADV_DYNAMIC_DEPTH_WRITE_ENABLE |                \
    RADV_DYNAMIC_DEPTH_COMPARE_OP | RADV_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE | RADV_DYNAMIC_STENCIL_TEST_ENABLE |         \
    RADV_DYNAMIC_STENCIL_OP)

static bool
radv_pipeline_is_blend_enabled(const struct radv_graphics_pipeline *pipeline, const struct vk_color_blend_state *cb)
{
   /* If we don't know then we have to assume that blend may be enabled. cb may also be NULL in this
    * case.
    */
   if (pipeline->dynamic_states & (RADV_DYNAMIC_COLOR_BLEND_ENABLE | RADV_DYNAMIC_COLOR_WRITE_MASK))
      return true;

   /* If we have the blend enable state, then cb being NULL indicates no attachments are written. */
   if (cb) {
      for (uint32_t i = 0; i < cb->attachment_count; i++) {
         if (cb->attachments[i].write_mask && cb->attachments[i].blend_enable)
            return true;
      }
   }

   return false;
}

static uint64_t
radv_pipeline_needed_dynamic_state(const struct radv_device *device, const struct radv_graphics_pipeline *pipeline,
                                   const struct vk_graphics_pipeline_state *state)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   bool raster_enabled =
      !state->rs->rasterizer_discard_enable || (pipeline->dynamic_states & RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE);
   uint64_t states = RADV_DYNAMIC_ALL;

   if (pdev->info.gfx_level < GFX10_3)
      states &= ~RADV_DYNAMIC_FRAGMENT_SHADING_RATE;

   /* Disable dynamic states that are useless to mesh shading. */
   if (radv_pipeline_has_stage(pipeline, MESA_SHADER_MESH)) {
      if (!raster_enabled)
         return RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE;

      states &= ~(RADV_DYNAMIC_VERTEX_INPUT | RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE |
                  RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE | RADV_DYNAMIC_PRIMITIVE_TOPOLOGY);
   }

   /* Disable dynamic states that are useless when rasterization is disabled. */
   if (!raster_enabled) {
      states = RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE;

      if (state->ia)
         states |= RADV_DYNAMIC_PRIMITIVE_TOPOLOGY | RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE;

      if (state->vi)
         states |= RADV_DYNAMIC_VERTEX_INPUT | RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE;

      if (pipeline->active_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
         states |= RADV_DYNAMIC_PATCH_CONTROL_POINTS | RADV_DYNAMIC_TESS_DOMAIN_ORIGIN;

      return states;
   }

   if (!state->rs->depth_bias.enable && !(pipeline->dynamic_states & RADV_DYNAMIC_DEPTH_BIAS_ENABLE))
      states &= ~RADV_DYNAMIC_DEPTH_BIAS;

   if (!(pipeline->dynamic_states & RADV_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE) &&
       (!state->ds || !state->ds->depth.bounds_test.enable))
      states &= ~RADV_DYNAMIC_DEPTH_BOUNDS;

   if (!(pipeline->dynamic_states & RADV_DYNAMIC_STENCIL_TEST_ENABLE) &&
       (!state->ds || !state->ds->stencil.test_enable))
      states &= ~(RADV_DYNAMIC_STENCIL_COMPARE_MASK | RADV_DYNAMIC_STENCIL_WRITE_MASK | RADV_DYNAMIC_STENCIL_REFERENCE |
                  RADV_DYNAMIC_STENCIL_OP);

   if (!(pipeline->dynamic_states & RADV_DYNAMIC_DISCARD_RECTANGLE_ENABLE) && !state->dr->rectangle_count)
      states &= ~RADV_DYNAMIC_DISCARD_RECTANGLE;

   if (!(pipeline->dynamic_states & RADV_DYNAMIC_SAMPLE_LOCATIONS_ENABLE) &&
       (!state->ms || !state->ms->sample_locations_enable))
      states &= ~RADV_DYNAMIC_SAMPLE_LOCATIONS;

   if (!radv_pipeline_is_blend_enabled(pipeline, state->cb))
      states &= ~RADV_DYNAMIC_BLEND_CONSTANTS;

   if (!(pipeline->active_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT))
      states &= ~(RADV_DYNAMIC_PATCH_CONTROL_POINTS | RADV_DYNAMIC_TESS_DOMAIN_ORIGIN);

   if (pipeline->dynamic_states & RADV_DYNAMIC_VERTEX_INPUT)
      states &= ~RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE;

   if (!radv_pipeline_has_color_attachments(state->rp))
      states &= ~RADV_DYNAMIC_CB_STATES;

   if (!radv_pipeline_has_ds_attachments(state->rp))
      states &= ~RADV_DYNAMIC_DS_STATES;

   return states;
}

static uint32_t
radv_get_attrib_stride(const VkPipelineVertexInputStateCreateInfo *vi, uint32_t attrib_binding)
{
   for (uint32_t i = 0; i < vi->vertexBindingDescriptionCount; i++) {
      const VkVertexInputBindingDescription *input_binding = &vi->pVertexBindingDescriptions[i];

      if (input_binding->binding == attrib_binding)
         return input_binding->stride;
   }

   return 0;
}

#define ALL_GRAPHICS_LIB_FLAGS                                                                                         \
   (VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT |                                                      \
    VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT |                                                   \
    VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT |                                                             \
    VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT)

static VkGraphicsPipelineLibraryFlagBitsEXT
shader_stage_to_pipeline_library_flags(VkShaderStageFlagBits stage)
{
   assert(util_bitcount(stage) == 1);
   switch (stage) {
   case VK_SHADER_STAGE_VERTEX_BIT:
   case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
   case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
   case VK_SHADER_STAGE_GEOMETRY_BIT:
   case VK_SHADER_STAGE_TASK_BIT_EXT:
   case VK_SHADER_STAGE_MESH_BIT_EXT:
      return VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;
   case VK_SHADER_STAGE_FRAGMENT_BIT:
      return VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;
   default:
      UNREACHABLE("Invalid shader stage");
   }
}

static void
radv_pipeline_import_graphics_info(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                   const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   /* Mark all states declared dynamic at pipeline creation. */
   if (pCreateInfo->pDynamicState) {
      uint32_t count = pCreateInfo->pDynamicState->dynamicStateCount;
      for (uint32_t s = 0; s < count; s++) {
         pipeline->dynamic_states |= radv_dynamic_state_mask(pCreateInfo->pDynamicState->pDynamicStates[s]);
      }
   }

   /* Mark all active stages at pipeline creation. */
   for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *sinfo = &pCreateInfo->pStages[i];

      pipeline->active_stages |= sinfo->stage;
   }

   if (pipeline->active_stages & VK_SHADER_STAGE_MESH_BIT_EXT) {
      pipeline->last_vgt_api_stage = MESA_SHADER_MESH;
   } else {
      pipeline->last_vgt_api_stage = util_last_bit(pipeline->active_stages & BITFIELD_MASK(MESA_SHADER_FRAGMENT)) - 1;
   }
}

static bool
radv_should_import_lib_binaries(const VkPipelineCreateFlags2 create_flags)
{
   return !(create_flags & (VK_PIPELINE_CREATE_2_LINK_TIME_OPTIMIZATION_BIT_EXT |
                            VK_PIPELINE_CREATE_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT));
}

static void
radv_graphics_pipeline_import_lib(const struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                  struct radv_graphics_lib_pipeline *lib, bool import_pipeline_binaries)
{
   bool import_binaries = false;

   /* There should be no common blocks between a lib we import and the current
    * pipeline we're building.
    */
   assert((pipeline->active_stages & lib->base.active_stages) == 0);

   pipeline->dynamic_states |= lib->base.dynamic_states;
   pipeline->active_stages |= lib->base.active_stages;

   /* Import binaries when LTO is disabled and when the library doesn't retain any shaders. */
   if (!import_pipeline_binaries &&
       (lib->base.has_pipeline_binaries || radv_should_import_lib_binaries(pipeline->base.create_flags))) {
      import_binaries = true;
   }

   if (import_binaries) {
      /* Import the compiled shaders. */
      for (uint32_t s = 0; s < ARRAY_SIZE(lib->base.base.shaders); s++) {
         if (!lib->base.base.shaders[s])
            continue;

         pipeline->base.shaders[s] = radv_shader_ref(lib->base.base.shaders[s]);
      }

      /* Import the GS copy shader if present. */
      if (lib->base.base.gs_copy_shader) {
         assert(!pipeline->base.gs_copy_shader);
         pipeline->base.gs_copy_shader = radv_shader_ref(lib->base.base.gs_copy_shader);
      }
   }
}

static void
radv_pipeline_init_input_assembly_state(const struct radv_device *device, struct radv_graphics_pipeline *pipeline)
{
   pipeline->ia_multi_vgt_param = radv_compute_ia_multi_vgt_param(device, pipeline->base.shaders);
}

static bool
radv_pipeline_uses_ds_feedback_loop(const struct radv_graphics_pipeline *pipeline,
                                    const struct vk_graphics_pipeline_state *state)
{
   VkPipelineCreateFlags2 create_flags = pipeline->base.create_flags;
   if (state->rp)
      create_flags |= state->pipeline_flags;

   return (create_flags & VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) != 0;
}

static void
radv_pipeline_init_vertex_input_state(const struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                      const struct vk_graphics_pipeline_state *state)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader *vs = radv_get_shader(pipeline->base.shaders, MESA_SHADER_VERTEX);
   struct radv_dynamic_state *dynamic = &pipeline->dynamic_state;

   if (vs->info.vs.use_per_attribute_vb_descs) {
      const enum amd_gfx_level gfx_level = pdev->info.gfx_level;
      const bool alpha_adjust = pdev->info.compiler_info.has_vtx_format_alpha_adjust_bug;
      const struct ac_vtx_format_info *vtx_info_table = ac_get_vtx_format_info_table(gfx_level, alpha_adjust);

      dynamic->vertex_input.bindings_match_attrib = true;

      u_foreach_bit (i, state->vi->attributes_valid) {
         uint32_t binding = state->vi->attributes[i].binding;
         uint32_t offset = state->vi->attributes[i].offset;

         dynamic->vertex_input.attribute_mask |= BITFIELD_BIT(i);
         dynamic->vertex_input.bindings[i] = binding;
         dynamic->vertex_input.bindings_match_attrib &= binding == i;

         if (state->vi->bindings[binding].input_rate) {
            dynamic->vertex_input.instance_rate_inputs |= BITFIELD_BIT(i);
            dynamic->vertex_input.divisors[i] = state->vi->bindings[binding].divisor;

            if (state->vi->bindings[binding].divisor == 0) {
               dynamic->vertex_input.zero_divisors |= BITFIELD_BIT(i);
            } else if (state->vi->bindings[binding].divisor > 1) {
               dynamic->vertex_input.nontrivial_divisors |= BITFIELD_BIT(i);
            }
         }

         dynamic->vertex_input.offsets[i] = offset;

         enum pipe_format format = radv_format_to_pipe_format(state->vi->attributes[i].format);
         const struct ac_vtx_format_info *vtx_info = &vtx_info_table[format];
         const uint32_t hw_format = vtx_info->hw_format[vtx_info->num_channels - 1];

         dynamic->vertex_input.formats[i] = format;
         uint8_t format_align_req_minus_1 = vtx_info->chan_byte_size >= 4 ? 3 : (vtx_info->element_size - 1);
         dynamic->vertex_input.format_align_req_minus_1[i] = format_align_req_minus_1;
         uint8_t component_align_req_minus_1 =
            MIN2(vtx_info->chan_byte_size ? vtx_info->chan_byte_size : vtx_info->element_size, 4) - 1;
         dynamic->vertex_input.component_align_req_minus_1[i] = component_align_req_minus_1;
         dynamic->vertex_input.format_sizes[i] = vtx_info->element_size;
         dynamic->vertex_input.alpha_adjust_lo |= (vtx_info->alpha_adjust & 0x1) << i;
         dynamic->vertex_input.alpha_adjust_hi |= (vtx_info->alpha_adjust >> 1) << i;
         if (G_008F0C_DST_SEL_X(vtx_info->dst_sel) == V_008F0C_SQ_SEL_Z) {
            dynamic->vertex_input.post_shuffle |= BITFIELD_BIT(i);
         }

         if (vtx_info->has_hw_format & BITFIELD_BIT(vtx_info->num_channels - 1)) {
            if (pdev->info.gfx_level >= GFX10) {
               dynamic->vertex_input.non_trivial_format[i] =
                  vtx_info->dst_sel | S_008F0C_FORMAT_GFX10(hw_format) |
                  S_008F0C_RESOURCE_LEVEL(pdev->info.compiler_info.has_desc_resource_level);
            } else {
               dynamic->vertex_input.non_trivial_format[i] = vtx_info->dst_sel |
                                                             S_008F0C_NUM_FORMAT((hw_format >> 4) & 0x7) |
                                                             S_008F0C_DATA_FORMAT(hw_format & 0xf);
            }
         } else {
            dynamic->vertex_input.nontrivial_formats |= BITFIELD_BIT(i);
         }
      }

      dynamic->vertex_input.vbo_misaligned_mask_invalid = dynamic->vertex_input.attribute_mask;
   } else {
      u_foreach_bit (i, vs->info.vs.vb_desc_usage_mask) {
         dynamic->vertex_input.bindings[i] = i;
      }
   }
}

static void
radv_pipeline_init_dynamic_state(const struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                 const struct vk_graphics_pipeline_state *state)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   uint64_t needed_states = radv_pipeline_needed_dynamic_state(device, pipeline, state);
   struct radv_dynamic_state *dynamic = &pipeline->dynamic_state;
   uint64_t states = needed_states;

   /* Initialize non-zero values for default dynamic state. */
   dynamic->vk.rs.line.width = 1.0f;
   dynamic->vk.fsr.fragment_size.width = 1u;
   dynamic->vk.fsr.fragment_size.height = 1u;
   dynamic->vk.ds.depth.bounds_test.max = 1.0f;
   dynamic->vk.ds.stencil.front.compare_mask = ~0;
   dynamic->vk.ds.stencil.front.write_mask = ~0;
   dynamic->vk.ds.stencil.back.compare_mask = ~0;
   dynamic->vk.ds.stencil.back.write_mask = ~0;
   dynamic->vk.ms.rasterization_samples = VK_SAMPLE_COUNT_1_BIT;

   pipeline->needed_dynamic_state = needed_states;

   states &= ~pipeline->dynamic_states;

   /* Input assembly. */
   if (states & RADV_DYNAMIC_PRIMITIVE_TOPOLOGY) {
      dynamic->vk.ia.primitive_topology = radv_translate_prim(state->ia->primitive_topology);
   }

   if (states & RADV_DYNAMIC_PRIMITIVE_RESTART_ENABLE) {
      dynamic->vk.ia.primitive_restart_enable = state->ia->primitive_restart_enable;
   }

   /* Tessellation. */
   if (states & RADV_DYNAMIC_PATCH_CONTROL_POINTS) {
      dynamic->vk.ts.patch_control_points = state->ts->patch_control_points;
   }

   if (states & RADV_DYNAMIC_TESS_DOMAIN_ORIGIN) {
      dynamic->vk.ts.domain_origin = state->ts->domain_origin;
   }

   /* Viewport. */
   if (states & RADV_DYNAMIC_VIEWPORT) {
      typed_memcpy(dynamic->vk.vp.viewports, state->vp->viewports, state->vp->viewport_count);
      for (unsigned i = 0; i < state->vp->viewport_count; i++)
         radv_get_viewport_xform(&dynamic->vk.vp.viewports[i], dynamic->vp_xform[i].scale,
                                 dynamic->vp_xform[i].translate);
   }

   if (states & RADV_DYNAMIC_VIEWPORT_WITH_COUNT) {
      dynamic->vk.vp.viewport_count = state->vp->viewport_count;
   }

   /* Scissor. */
   if (states & RADV_DYNAMIC_SCISSOR) {
      typed_memcpy(dynamic->vk.vp.scissors, state->vp->scissors, state->vp->scissor_count);
   }

   if (states & RADV_DYNAMIC_SCISSOR_WITH_COUNT) {
      dynamic->vk.vp.scissor_count = state->vp->scissor_count;
   }

   if (states & RADV_DYNAMIC_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) {
      dynamic->vk.vp.depth_clip_negative_one_to_one = state->vp->depth_clip_negative_one_to_one;
   }

   if (states & RADV_DYNAMIC_DEPTH_CLAMP_RANGE) {
      dynamic->vk.vp.depth_clamp_mode = state->vp->depth_clamp_mode;
      dynamic->vk.vp.depth_clamp_range = state->vp->depth_clamp_range;
   }

   /* Discard rectangles. */
   if (needed_states & RADV_DYNAMIC_DISCARD_RECTANGLE) {
      dynamic->vk.dr.rectangle_count = state->dr->rectangle_count;
      if (states & RADV_DYNAMIC_DISCARD_RECTANGLE) {
         typed_memcpy(dynamic->vk.dr.rectangles, state->dr->rectangles, state->dr->rectangle_count);
      }
   }

   /* Rasterization. */
   if (states & RADV_DYNAMIC_LINE_WIDTH) {
      dynamic->vk.rs.line.width = state->rs->line.width;
   }

   if (states & RADV_DYNAMIC_DEPTH_BIAS) {
      dynamic->vk.rs.depth_bias.constant_factor = state->rs->depth_bias.constant_factor;
      dynamic->vk.rs.depth_bias.clamp = state->rs->depth_bias.clamp;
      dynamic->vk.rs.depth_bias.slope_factor = state->rs->depth_bias.slope_factor;
      dynamic->vk.rs.depth_bias.representation = state->rs->depth_bias.representation;
   }

   if (states & RADV_DYNAMIC_CULL_MODE) {
      dynamic->vk.rs.cull_mode = state->rs->cull_mode;
   }

   if (states & RADV_DYNAMIC_FRONT_FACE) {
      dynamic->vk.rs.front_face = state->rs->front_face;
   }

   if (states & RADV_DYNAMIC_LINE_STIPPLE) {
      dynamic->vk.rs.line.stipple.factor = state->rs->line.stipple.factor;
      dynamic->vk.rs.line.stipple.pattern = state->rs->line.stipple.pattern;
   }

   if (states & RADV_DYNAMIC_DEPTH_BIAS_ENABLE) {
      dynamic->vk.rs.depth_bias.enable = state->rs->depth_bias.enable;
   }

   if (states & RADV_DYNAMIC_RASTERIZER_DISCARD_ENABLE) {
      dynamic->vk.rs.rasterizer_discard_enable = state->rs->rasterizer_discard_enable;
   }

   if (states & RADV_DYNAMIC_POLYGON_MODE) {
      dynamic->vk.rs.polygon_mode = radv_translate_fill(state->rs->polygon_mode);
   }

   if (states & RADV_DYNAMIC_LINE_STIPPLE_ENABLE) {
      dynamic->vk.rs.line.stipple.enable = state->rs->line.stipple.enable;
   }

   if (states & RADV_DYNAMIC_DEPTH_CLIP_ENABLE) {
      dynamic->vk.rs.depth_clip_enable = state->rs->depth_clip_enable;
   }

   if (states & RADV_DYNAMIC_CONSERVATIVE_RAST_MODE) {
      dynamic->vk.rs.conservative_mode = state->rs->conservative_mode;
   }

   if (states & RADV_DYNAMIC_PROVOKING_VERTEX_MODE) {
      dynamic->vk.rs.provoking_vertex = state->rs->provoking_vertex;
   }

   if (states & RADV_DYNAMIC_DEPTH_CLAMP_ENABLE) {
      dynamic->vk.rs.depth_clamp_enable = state->rs->depth_clamp_enable;
   }

   if (states & RADV_DYNAMIC_LINE_RASTERIZATION_MODE) {
      dynamic->vk.rs.line.mode = state->rs->line.mode;
   }

   /* Fragment shading rate. */
   if (states & RADV_DYNAMIC_FRAGMENT_SHADING_RATE) {
      dynamic->vk.fsr = *state->fsr;
   }

   /* Multisample. */
   if (states & RADV_DYNAMIC_ALPHA_TO_COVERAGE_ENABLE) {
      dynamic->vk.ms.alpha_to_coverage_enable = state->ms->alpha_to_coverage_enable;
   }

   if (states & RADV_DYNAMIC_ALPHA_TO_ONE_ENABLE) {
      dynamic->vk.ms.alpha_to_one_enable = state->ms->alpha_to_one_enable;
   }

   if (states & RADV_DYNAMIC_SAMPLE_MASK) {
      dynamic->vk.ms.sample_mask = state->ms->sample_mask & 0xffff;
   }

   if (states & RADV_DYNAMIC_RASTERIZATION_SAMPLES) {
      dynamic->vk.ms.rasterization_samples = state->ms->rasterization_samples;
   }

   if (states & RADV_DYNAMIC_SAMPLE_LOCATIONS_ENABLE) {
      dynamic->vk.ms.sample_locations_enable = state->ms->sample_locations_enable;
   }

   if (states & RADV_DYNAMIC_SAMPLE_LOCATIONS) {
      unsigned count = state->ms->sample_locations->per_pixel * state->ms->sample_locations->grid_size.width *
                       state->ms->sample_locations->grid_size.height;

      dynamic->sample_location.per_pixel = state->ms->sample_locations->per_pixel;
      dynamic->sample_location.grid_size = state->ms->sample_locations->grid_size;
      dynamic->sample_location.count = count;
      typed_memcpy(&dynamic->sample_location.locations[0], state->ms->sample_locations->locations, count);
   }

   /* Depth stencil. */
   if (states & RADV_DYNAMIC_DEPTH_BOUNDS) {
      dynamic->vk.ds.depth.bounds_test.min = state->ds->depth.bounds_test.min;
      dynamic->vk.ds.depth.bounds_test.max = state->ds->depth.bounds_test.max;
   }

   if (states & RADV_DYNAMIC_STENCIL_COMPARE_MASK) {
      dynamic->vk.ds.stencil.front.compare_mask = state->ds->stencil.front.compare_mask;
      dynamic->vk.ds.stencil.back.compare_mask = state->ds->stencil.back.compare_mask;
   }

   if (states & RADV_DYNAMIC_STENCIL_WRITE_MASK) {
      dynamic->vk.ds.stencil.front.write_mask = state->ds->stencil.front.write_mask;
      dynamic->vk.ds.stencil.back.write_mask = state->ds->stencil.back.write_mask;
   }

   if (states & RADV_DYNAMIC_STENCIL_REFERENCE) {
      dynamic->vk.ds.stencil.front.reference = state->ds->stencil.front.reference;
      dynamic->vk.ds.stencil.back.reference = state->ds->stencil.back.reference;
   }

   if (states & RADV_DYNAMIC_DEPTH_TEST_ENABLE) {
      dynamic->vk.ds.depth.test_enable = state->ds->depth.test_enable;
   }

   if (states & RADV_DYNAMIC_DEPTH_WRITE_ENABLE) {
      dynamic->vk.ds.depth.write_enable = state->ds->depth.write_enable;
   }

   if (states & RADV_DYNAMIC_DEPTH_COMPARE_OP) {
      dynamic->vk.ds.depth.compare_op = state->ds->depth.compare_op;
   }

   if (states & RADV_DYNAMIC_DEPTH_BOUNDS_TEST_ENABLE) {
      dynamic->vk.ds.depth.bounds_test.enable = state->ds->depth.bounds_test.enable;
   }

   if (states & RADV_DYNAMIC_STENCIL_TEST_ENABLE) {
      dynamic->vk.ds.stencil.test_enable = state->ds->stencil.test_enable;
   }

   if (states & RADV_DYNAMIC_STENCIL_OP) {
      dynamic->vk.ds.stencil.front.op.compare = state->ds->stencil.front.op.compare;
      dynamic->vk.ds.stencil.front.op.fail = radv_translate_stencil_op(state->ds->stencil.front.op.fail);
      dynamic->vk.ds.stencil.front.op.pass = radv_translate_stencil_op(state->ds->stencil.front.op.pass);
      dynamic->vk.ds.stencil.front.op.depth_fail = radv_translate_stencil_op(state->ds->stencil.front.op.depth_fail);

      dynamic->vk.ds.stencil.back.op.compare = state->ds->stencil.back.op.compare;
      dynamic->vk.ds.stencil.back.op.fail = radv_translate_stencil_op(state->ds->stencil.back.op.fail);
      dynamic->vk.ds.stencil.back.op.pass = radv_translate_stencil_op(state->ds->stencil.back.op.pass);
      dynamic->vk.ds.stencil.back.op.depth_fail = radv_translate_stencil_op(state->ds->stencil.back.op.depth_fail);
   }

   /* Color blend. */
   /* Section 9.2 of the Vulkan 1.0.15 spec says:
    *
    *    pColorBlendState is [...] NULL if the pipeline has rasterization
    *    disabled or if the subpass of the render pass the pipeline is
    *    created against does not use any color attachments.
    */
   if (states & RADV_DYNAMIC_BLEND_CONSTANTS) {
      typed_memcpy(dynamic->vk.cb.blend_constants, state->cb->blend_constants, 4);
   }

   if (states & RADV_DYNAMIC_LOGIC_OP) {
      if ((pipeline->dynamic_states & RADV_DYNAMIC_LOGIC_OP_ENABLE) || state->cb->logic_op_enable) {
         dynamic->vk.cb.logic_op = radv_translate_blend_logic_op(state->cb->logic_op);
      }
   }

   if (states & RADV_DYNAMIC_COLOR_WRITE_ENABLE) {
      u_foreach_bit (i, state->cb->color_write_enables) {
         dynamic->color_write_enable |= BITFIELD_RANGE(i * 4, 4);
      }
   }

   if (states & RADV_DYNAMIC_LOGIC_OP_ENABLE) {
      dynamic->vk.cb.logic_op_enable = state->cb->logic_op_enable;
   }

   if (states & RADV_DYNAMIC_COLOR_WRITE_MASK) {
      for (unsigned i = 0; i < state->cb->attachment_count; i++) {
         dynamic->color_write_mask |= (uint32_t)state->cb->attachments[i].write_mask << (4 * i);
      }
   }

   if (states & RADV_DYNAMIC_COLOR_BLEND_ENABLE) {
      for (unsigned i = 0; i < state->cb->attachment_count; i++) {
         dynamic->color_blend_enable |= state->cb->attachments[i].blend_enable << i;
      }
   }

   if (states & RADV_DYNAMIC_COLOR_BLEND_EQUATION) {
      for (unsigned i = 0; i < state->cb->attachment_count; i++) {
         const struct vk_color_blend_attachment_state *att = &state->cb->attachments[i];

         radv_translate_blend_equation(pdev, att->color_blend_op, att->src_color_blend_factor,
                                       att->dst_color_blend_factor, att->alpha_blend_op, att->src_alpha_blend_factor,
                                       att->dst_alpha_blend_factor, &dynamic->blend_eq.att[i].cb_blend_control,
                                       &dynamic->blend_eq.att[i].sx_mrt_blend_opt);
      }

      dynamic->blend_eq.mrt0_is_dual_src = radv_can_enable_dual_src(&state->cb->attachments[0]);
   }

   if (states & RADV_DYNAMIC_DISCARD_RECTANGLE_ENABLE) {
      dynamic->vk.dr.enable = state->dr->rectangle_count > 0;
   }

   if (states & RADV_DYNAMIC_DISCARD_RECTANGLE_MODE) {
      dynamic->vk.dr.mode = state->dr->mode;
   }

   if (states & RADV_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE) {
      bool uses_ds_feedback_loop = radv_pipeline_uses_ds_feedback_loop(pipeline, state);

      dynamic->feedback_loop_aspects =
         uses_ds_feedback_loop ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_NONE;
   }

   if (states & RADV_DYNAMIC_VERTEX_INPUT_BINDING_STRIDE) {
      u_foreach_bit (i, state->vi->bindings_valid) {
         dynamic->vk.vi_binding_strides[i] = state->vi->bindings[i].stride;
      }
   }

   if (states & RADV_DYNAMIC_VERTEX_INPUT) {
      radv_pipeline_init_vertex_input_state(device, pipeline, state);
   }

   for (uint32_t i = 0; i < MAX_RTS; i++) {
      dynamic->vk.cal.color_map[i] = state->cal ? state->cal->color_map[i] : i;
      dynamic->vk.ial.color_map[i] = state->ial ? state->ial->color_map[i] : i;
   }

   dynamic->vk.ial.depth_att = state->ial ? state->ial->depth_att : MESA_VK_ATTACHMENT_UNUSED;
   dynamic->vk.ial.stencil_att = state->ial ? state->ial->stencil_att : MESA_VK_ATTACHMENT_UNUSED;

   pipeline->dynamic_state.mask = states;
}

static bool
radv_pipeline_needs_noop_fs(struct radv_graphics_pipeline *pipeline, const struct radv_graphics_state_key *gfx_state)
{
   if (pipeline->base.type == RADV_PIPELINE_GRAPHICS &&
       !(radv_pipeline_to_graphics(&pipeline->base)->active_stages & VK_SHADER_STAGE_FRAGMENT_BIT))
      return true;

   if (pipeline->base.type == RADV_PIPELINE_GRAPHICS_LIB &&
       (gfx_state->lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) &&
       !(radv_pipeline_to_graphics_lib(&pipeline->base)->base.active_stages & VK_SHADER_STAGE_FRAGMENT_BIT))
      return true;

   return false;
}

static struct radv_ps_epilog_key
radv_pipeline_generate_ps_epilog_key(const struct radv_compiler_info *compiler_info,
                                     const struct vk_graphics_pipeline_state *state)
{
   struct radv_ps_epilog_state ps_epilog = {0};

   if (state->ms && state->ms->alpha_to_coverage_enable)
      ps_epilog.need_src_alpha |= 0x1;

   if (state->cb) {
      for (uint32_t i = 0; i < state->cb->attachment_count; i++) {
         VkBlendOp eqRGB = state->cb->attachments[i].color_blend_op;
         VkBlendFactor srcRGB = state->cb->attachments[i].src_color_blend_factor;
         VkBlendFactor dstRGB = state->cb->attachments[i].dst_color_blend_factor;

         /* Ignore other blend targets if dual-source blending is enabled to prevent wrong
          * behaviour.
          */
         if (i > 0 && ps_epilog.mrt0_is_dual_src)
            continue;

         ps_epilog.color_write_mask |= (unsigned)state->cb->attachments[i].write_mask << (4 * i);
         if (!((ps_epilog.color_write_mask >> (i * 4)) & 0xf))
            continue;

         ps_epilog.color_blend_enable |= state->cb->attachments[i].blend_enable << i;

         if (!((ps_epilog.color_blend_enable >> i) & 0x1u))
            continue;

         if (i == 0 && radv_can_enable_dual_src(&state->cb->attachments[i])) {
            ps_epilog.mrt0_is_dual_src = true;
         }

         radv_normalize_blend_factor(eqRGB, &srcRGB, &dstRGB);

         if (srcRGB == VK_BLEND_FACTOR_SRC_ALPHA || dstRGB == VK_BLEND_FACTOR_SRC_ALPHA ||
             srcRGB == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE || dstRGB == VK_BLEND_FACTOR_SRC_ALPHA_SATURATE ||
             srcRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA || dstRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA ||
             srcRGB == VK_BLEND_FACTOR_SRC1_ALPHA || dstRGB == VK_BLEND_FACTOR_SRC1_ALPHA ||
             srcRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA || dstRGB == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA)
            ps_epilog.need_src_alpha |= 1 << i;
      }
   }

   if (state->rp) {
      ps_epilog.color_attachment_count = state->rp->color_attachment_count;

      for (uint32_t i = 0; i < ps_epilog.color_attachment_count; i++) {
         ps_epilog.color_attachment_formats[i] = state->rp->color_attachment_formats[i];
      }
   }

   if (state->ms)
      ps_epilog.alpha_to_one = state->ms->alpha_to_one_enable;

   const bool may_have_depth_attachment = !state->rp || state->rp->depth_attachment_format != VK_FORMAT_UNDEFINED;
   const bool may_have_stencil_attachment = !state->rp || state->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;

   ps_epilog.ignore_depth_output =
      !may_have_depth_attachment || (state->ds && !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE) &&
                                     !state->ds->depth.test_enable);
   ps_epilog.ignore_stencil_output =
      !may_have_stencil_attachment ||
      (state->ds && !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE) &&
       !state->ds->stencil.test_enable);

   ps_epilog.lower_1bit_sample_mask_to_discard =
      state->ms && !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
      state->ms->rasterization_samples == 1;

   for (uint32_t i = 0; i < MAX_RTS; i++) {
      ps_epilog.color_attachment_mappings[i] = state->cal ? state->cal->color_map[i] : i;
   }

   return radv_generate_ps_epilog_key(compiler_info, &ps_epilog);
}

static struct radv_graphics_state_key
radv_generate_graphics_state_key(const struct radv_compiler_info *compiler_info,
                                 const struct vk_graphics_pipeline_state *state,
                                 VkGraphicsPipelineLibraryFlagBitsEXT lib_flags, uint32_t custom_blend_mode)
{
   struct radv_graphics_state_key key;

   memset(&key, 0, sizeof(key));

   key.lib_flags = lib_flags;

   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI)) {
      key.vs.has_prolog = true;
   }

   /* Make sure to require a VS prolog when the VS is compiled without the vertex input state (this
    * can happen with GPL).
    */
   if ((state->shader_stages & VK_SHADER_STAGE_VERTEX_BIT) && !state->vi) {
      key.vs.has_prolog = true;
   }

   /* Vertex input state */
   if (state->vi) {
      key.vi.attributes_valid = state->vi->attributes_valid;

      u_foreach_bit (i, state->vi->attributes_valid) {
         uint32_t binding = state->vi->attributes[i].binding;
         uint32_t offset = state->vi->attributes[i].offset;
         enum pipe_format format = radv_format_to_pipe_format(state->vi->attributes[i].format);

         key.vi.vertex_attribute_formats[i] = format;
         key.vi.vertex_attribute_bindings[i] = binding;
         key.vi.vertex_attribute_offsets[i] = offset;
         key.vi.instance_rate_divisors[i] = state->vi->bindings[binding].divisor;

         if (state->vi->bindings[binding].input_rate) {
            key.vi.instance_rate_inputs |= 1u << i;
         }

         const struct ac_vtx_format_info *vtx_info = ac_get_vtx_format_info(
            compiler_info->ac->gfx_level, compiler_info->ac->has_vtx_format_alpha_adjust_bug, format);
         unsigned attrib_align = vtx_info->chan_byte_size ? vtx_info->chan_byte_size : vtx_info->element_size;

         /* If offset is misaligned, then the buffer offset must be too. Just skip updating
          * vertex_binding_align in this case.
          */
         if (offset % attrib_align == 0) {
            key.vi.vertex_binding_align[binding] = MAX2(key.vi.vertex_binding_align[binding], attrib_align);
         }
      }
   }

   if (state->ts)
      key.ts.patch_control_points = state->ts->patch_control_points;

   /* On GFX11+, DB always reads alpha-to-coverage from the first export but also mrtz must be
    * the first export if it's present. That means alpha-to-coverage comes from mrtz.w if it's
    * present, else it comes from mrt0.w. That's because the first export (mrtz or mrt0) is always
    * forwarded to DB, and DB uses it to determine the final coverage.
    *
    * Additionally if alpha-to-one is enabled, alpha-to-coverage must be exported via mrtz.w on all
    * generations because alpha-to-one forces mrt0.w to 1 and there is no other way to export alpha
    * for coverage. Pre-GFX11 HW can optionally export 2 alpha values for mrt0:
    * - via mrtz.w, which is only used for alpha-to-coverage (it's the color 0 output alpha value
    *   if alpha-to-one is enabled)
    * - via mrt0.w, which is only used by blending or written to color attachment 0 (1 if alpha-to-one
    *   is enabled)
    */
   bool alpha_to_one_unknown = !state->ms || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE);

   key.ms.alpha_to_coverage_unknown =
      !state->ms || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE);
   key.ms.alpha_to_coverage_enable = !key.ms.alpha_to_coverage_unknown && state->ms->alpha_to_coverage_enable;
   key.ms.alpha_to_one_enable = !alpha_to_one_unknown && state->ms->alpha_to_one_enable;

   if (state->ms) {
      key.ms.sample_shading_enable = state->ms->sample_shading_enable;
      if (state->ms->sample_shading_enable)
         key.ms.min_sample_shading = state->ms->min_sample_shading;
      key.ms.max_sample_shading_enable = state->ms->sample_shading_enable && state->ms->min_sample_shading == 1;

      if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
          state->ms->rasterization_samples > 1) {
         key.ms.rasterization_samples = state->ms->rasterization_samples;

         if (state->ms->sample_shading_enable) {
            key.ms.ps_iter_samples =
               util_next_power_of_two(ceilf(state->ms->rasterization_samples * state->ms->min_sample_shading));
         }
      }
   }

   if (state->ia) {
      key.ia.topology = radv_translate_prim(state->ia->primitive_topology);
   }

   key.rs.polygon_mode_unknown = !state->rs || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE);

   if (!key.rs.polygon_mode_unknown) {
      switch (state->rs->polygon_mode) {
      case VK_POLYGON_MODE_FILL:
      case VK_POLYGON_MODE_LINE:
      case VK_POLYGON_MODE_POINT:
         key.rs.polygon_mode = state->rs->polygon_mode;
         break;
      default:
         UNREACHABLE("unexpected polygon mode");
      }
   }

   if (state->rs) {
      if (compiler_info->ac->gfx_level >= GFX10)
         key.rs.provoking_vtx_last = state->rs->provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;

      if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_CULL_MODE))
         key.rs.cull_mode = state->rs->cull_mode;

      /* Don't generate face culling code if face culling is disabled and rasterization is enabled.
       * The only thing the face culling would do is cull zero-area triangles.
       *
       * When rasterizer discard is dynamic, the driver dynamically enables both front and back face culling
       * through a user SGPR.
       */
      key.rs.skip_ngg_cull_face = compiler_info->key.use_ngg_culling &&
                                  !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_CULL_MODE) &&
                                  !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE) &&
                                  !state->rs->cull_mode && !state->rs->rasterizer_discard_enable;

      key.rs.skip_all_ngg_culling = compiler_info->key.use_ngg_culling &&
                                    !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE) &&
                                    !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_CONSERVATIVE_MODE) &&
                                    !state->rs->rasterizer_discard_enable &&
                                    state->rs->conservative_mode == VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;

      key.rs.rasterizer_discard = !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE) &&
                                  state->rs->rasterizer_discard_enable;
   }

   key.dynamic_rasterization_samples = BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) ||
                                       (!!(state->shader_stages & VK_SHADER_STAGE_FRAGMENT_BIT) && !state->ms);

   const bool vrs_disabled_by_msaa_8x = !key.dynamic_rasterization_samples && key.ms.rasterization_samples == 8;
   const bool vrs_disabled_by_sample_shading = key.ms.sample_shading_enable;
   const bool vrs_is_possible = !vrs_disabled_by_msaa_8x && !vrs_disabled_by_sample_shading;

   key.ps.force_vrs_enabled = vrs_is_possible && compiler_info->force_vrs_enabled && !radv_is_static_vrs_enabled(state);
   key.vrs_may_be_enabled =
      vrs_is_possible && (radv_is_static_vrs_enabled(state) || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_FSR) ||
                          key.ps.force_vrs_enabled);

   if (key.vrs_may_be_enabled && compiler_info->ac->has_vrs_frag_pos_z_bug)
      key.adjust_frag_coord_z = true;

   key.custom_sample_locs_may_be_enabled = BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_SAMPLE_LOCATIONS_ENABLE) ||
                                           !state->ms || state->ms->sample_locations_enable;

   key.ps.color_outputs_need_epilog = color_outputs_need_ps_epilog(state, lib_flags);
   key.ps.epilog = radv_pipeline_generate_ps_epilog_key(compiler_info, state);

   /* Use the PS epilog if a FS depth/stencil/samplemask output is present and could be eliminated based on dynamic
    * state.
    */
   key.ps.depth_output_needs_epilog = !key.ps.epilog.ignore_depth_output &&
                                      (!state->rp || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE));

   key.ps.stencil_output_needs_epilog =
      !key.ps.epilog.ignore_stencil_output &&
      (!state->rp || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE));

   key.ps.sample_mask_output_needs_epilog =
      !key.ps.epilog.lower_1bit_sample_mask_to_discard &&
      (!state->ms || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES));

   /* Set whether alpha_to_one makes MRT0 alpha dead. */
   key.ps.mrt0_alpha_is_dead = !alpha_to_one_unknown && !key.ms.alpha_to_coverage_unknown &&
                               key.ms.alpha_to_one_enable && !key.ms.alpha_to_coverage_enable;

   if (compiler_info->key.use_ngg) {
      VkShaderStageFlags ngg_stage;

      if (state->shader_stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
         ngg_stage = VK_SHADER_STAGE_GEOMETRY_BIT;
      } else if (state->shader_stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
         ngg_stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
      } else {
         ngg_stage = VK_SHADER_STAGE_VERTEX_BIT;
      }

      key.dynamic_provoking_vtx_mode =
         BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_PROVOKING_VERTEX) &&
         (ngg_stage == VK_SHADER_STAGE_VERTEX_BIT || ngg_stage == VK_SHADER_STAGE_GEOMETRY_BIT);
   }

   key.smooth_lines_may_be_enabled =
      compiler_info->smooth_lines && (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_LINE_MODE) || !state->rs ||
                                      state->rs->line.mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH);

   key.dcc_decompress_gfx11 =
      compiler_info->ac->gfx_level >= GFX11 && custom_blend_mode == V_028808_CB_DCC_DECOMPRESS_GFX11;

   return key;
}

static struct radv_graphics_pipeline_key
radv_generate_graphics_pipeline_key(const struct radv_device *device, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                    const struct vk_graphics_pipeline_state *state,
                                    VkGraphicsPipelineLibraryFlagBitsEXT lib_flags)
{
   const struct radv_compiler_info *compiler_info = &device->compiler_info;
   VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   struct radv_graphics_pipeline_key key = {0};
   uint32_t custom_blend_mode = 0;

   const VkGraphicsPipelineCreateInfoRADV *radv_info =
      vk_find_struct_const(pCreateInfo->pNext, GRAPHICS_PIPELINE_CREATE_INFO_RADV);
   if (radv_info) {
      custom_blend_mode = radv_info->custom_blend_mode;
   }

   key.gfx_state = radv_generate_graphics_state_key(compiler_info, state, lib_flags, custom_blend_mode);

   for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *stage = &pCreateInfo->pStages[i];
      mesa_shader_stage s = vk_to_mesa_shader_stage(stage->stage);

      key.stage_info[s] = radv_pipeline_get_shader_key(compiler_info, stage, create_flags, pCreateInfo->pNext);

      if (state->mv && state->mv->view_mask)
         key.stage_info[s].multiview = true;

      if (s == MESA_SHADER_MESH && (state->shader_stages & VK_SHADER_STAGE_TASK_BIT_EXT))
         key.stage_info[s].has_task_shader = true;
   }

   return key;
}

static void
radv_pipeline_retain_shaders(struct radv_retained_shaders *retained_shaders, struct radv_shader_stage *stages)
{
   for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
      if (stages[s].stage == MESA_SHADER_NONE)
         continue;

      int64_t stage_start = os_time_get_nano();

      /* Serialize the NIR shader to reduce memory pressure. */
      struct blob blob;

      blob_init(&blob);
      nir_serialize(&blob, stages[s].nir, true);
      blob_finish_get_buffer(&blob, &retained_shaders->stages[s].serialized_nir,
                             &retained_shaders->stages[s].serialized_nir_size);

      memcpy(retained_shaders->stages[s].shader_blake3, stages[s].shader_blake3, sizeof(stages[s].shader_blake3));
      memcpy(&retained_shaders->stages[s].key, &stages[s].key, sizeof(stages[s].key));

      stages[s].feedback.duration += os_time_get_nano() - stage_start;
   }
}

static void
radv_pipeline_import_retained_shaders(const struct radv_device *device, struct radv_graphics_lib_pipeline *lib,
                                      struct radv_shader_stage *stages)
{
   struct radv_retained_shaders *retained_shaders = &lib->retained_shaders;

   /* Import the stages (SPIR-V only in case of cache hits). */
   for (uint32_t i = 0; i < lib->stage_count; i++) {
      const VkPipelineShaderStageCreateInfo *sinfo = &lib->stages[i];
      mesa_shader_stage s = vk_to_mesa_shader_stage(sinfo->stage);

      radv_pipeline_stage_init(lib->base.base.create_flags, sinfo, &lib->layout, &lib->stage_keys[s], &stages[s]);
   }

   /* Import the NIR shaders (after SPIRV->NIR). */
   for (uint32_t s = 0; s < ARRAY_SIZE(lib->base.base.shaders); s++) {
      if (!retained_shaders->stages[s].serialized_nir_size)
         continue;

      int64_t stage_start = os_time_get_nano();

      /* Deserialize the NIR shader. */
      const struct nir_shader_compiler_options *options = &device->compiler_info.nir_options[s];
      struct blob_reader blob_reader;
      blob_reader_init(&blob_reader, retained_shaders->stages[s].serialized_nir,
                       retained_shaders->stages[s].serialized_nir_size);

      stages[s].stage = s;
      stages[s].nir = nir_deserialize(NULL, options, &blob_reader);
      stages[s].entrypoint = nir_shader_get_entrypoint(stages[s].nir)->function->name;
      memcpy(stages[s].shader_blake3, retained_shaders->stages[s].shader_blake3, sizeof(stages[s].shader_blake3));
      memcpy(&stages[s].key, &retained_shaders->stages[s].key, sizeof(stages[s].key));

      radv_shader_layout_init(&lib->layout, s, &stages[s].layout);

      stages[s].feedback.flags |= VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT;

      stages[s].feedback.duration += os_time_get_nano() - stage_start;
   }
}

static void
radv_pipeline_load_retained_shaders(const struct radv_device *device, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                    struct radv_shader_stage *stages)
{
   const VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);

   /* Nothing to load if no libs are imported. */
   if (!libs_info)
      return;

   /* Nothing to load if fast-linking is enabled and if there is no retained shaders. */
   if (radv_should_import_lib_binaries(create_flags))
      return;

   for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
      VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
      struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

      radv_pipeline_import_retained_shaders(device, gfx_pipeline_lib, stages);
   }
}

static bool
radv_is_fast_linking_enabled(const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   const VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);

   if (!libs_info)
      return false;

   return !(create_flags & VK_PIPELINE_CREATE_2_LINK_TIME_OPTIMIZATION_BIT_EXT);
}

static bool
radv_skip_graphics_pipeline_compile(const struct radv_device *device, const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   const VkPipelineBinaryInfoKHR *binary_info = vk_find_struct_const(pCreateInfo->pNext, PIPELINE_BINARY_INFO_KHR);
   const VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   VkShaderStageFlagBits binary_stages = 0;
   VkShaderStageFlags active_stages = 0;

   /* No compilation when pipeline binaries are imported. */
   if (binary_info && binary_info->binaryCount > 0)
      return true;

   /* Do not skip for libraries. */
   if (create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)
      return false;

   /* Do not skip when fast-linking isn't enabled. */
   if (!radv_is_fast_linking_enabled(pCreateInfo))
      return false;

   for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *sinfo = &pCreateInfo->pStages[i];
      active_stages |= sinfo->stage;
   }

   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
   if (libs_info) {
      for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

         assert(pipeline_lib->type == RADV_PIPELINE_GRAPHICS_LIB);

         active_stages |= gfx_pipeline_lib->base.active_stages;

         for (uint32_t s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
            if (!gfx_pipeline_lib->base.base.shaders[s])
               continue;

            binary_stages |= mesa_to_vk_shader_stage(s);
         }
      }
   }

   if (pdev->info.gfx_level >= GFX9) {
      /* On GFX9+, TES is merged with GS and VS is merged with TCS or GS. */
      if (binary_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
         binary_stages |= VK_SHADER_STAGE_VERTEX_BIT;
      }

      if (binary_stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
         if (binary_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
            binary_stages |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
         } else {
            binary_stages |= VK_SHADER_STAGE_VERTEX_BIT;
         }
      }
   }

   /* Only skip compilation when all binaries have been imported. */
   return binary_stages == active_stages;
}

void
radv_graphics_pipeline_state_finish(struct radv_device *device, struct radv_graphics_pipeline_state *gfx_state)
{
   radv_pipeline_layout_finish(device, &gfx_state->layout);
   vk_free(&device->vk.alloc, gfx_state->vk_data);

   if (gfx_state->stages) {
      for (uint32_t i = 0; i < MESA_VULKAN_SHADER_STAGES; i++)
         radv_pipeline_stage_finish(&gfx_state->stages[i]);

      ralloc_free(gfx_state->stages[MESA_SHADER_GEOMETRY].gs_copy_shader);

      free(gfx_state->stages);
   }
}

VkResult
radv_generate_graphics_pipeline_state(struct radv_device *device, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                      struct radv_graphics_pipeline_state *gfx_state)
{
   VK_FROM_HANDLE(radv_pipeline_layout, pipeline_layout, pCreateInfo->layout);
   const VkPipelineCreateFlags2 create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   VkResult result;

   memset(gfx_state, 0, sizeof(*gfx_state));

   VkGraphicsPipelineLibraryFlagBitsEXT needed_lib_flags = ALL_GRAPHICS_LIB_FLAGS;
   if (create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
      const VkGraphicsPipelineLibraryCreateInfoEXT *lib_info =
         vk_find_struct_const(pCreateInfo->pNext, GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
      needed_lib_flags = lib_info ? lib_info->flags : 0;
   }

   radv_pipeline_layout_init(device, &gfx_state->layout, false);

   /* If we have libraries, import them first. */
   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
   if (libs_info) {
      blake3_hash lib_hashes[4] = {0};
      struct mesa_blake3 ctx;

      for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
         const struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

         /* Hash libraries by flags to keep the key stable if the import order changes. */
         if (gfx_pipeline_lib->lib_flags) {
            const unsigned slot = ffs(gfx_pipeline_lib->lib_flags) - 1;
            assert(slot < ARRAY_SIZE(lib_hashes));
            memcpy(lib_hashes[slot], gfx_pipeline_lib->base.base.blake3, sizeof(blake3_hash));
         }

         vk_graphics_pipeline_state_merge(&gfx_state->vk, &gfx_pipeline_lib->graphics_state);

         radv_pipeline_layout_merge(&gfx_state->layout, &gfx_pipeline_lib->layout);

         needed_lib_flags &= ~gfx_pipeline_lib->lib_flags;
      }

      _mesa_blake3_init(&ctx);
      _mesa_blake3_update(&ctx, lib_hashes, sizeof(lib_hashes));
      _mesa_blake3_final(&ctx, gfx_state->libs_blake3);
   }

   result = vk_graphics_pipeline_state_fill(&device->vk, &gfx_state->vk, pCreateInfo, NULL /* driver_mv */,
                                            NULL /* driver_rp */, 0, NULL, NULL, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT,
                                            &gfx_state->vk_data);
   if (result != VK_SUCCESS)
      goto fail;

   if (pipeline_layout)
      radv_pipeline_layout_merge(&gfx_state->layout, pipeline_layout);

   radv_pipeline_layout_hash(&gfx_state->layout);

   gfx_state->compilation_required = !radv_skip_graphics_pipeline_compile(device, pCreateInfo);
   if (gfx_state->compilation_required) {
      gfx_state->key = radv_generate_graphics_pipeline_key(device, pCreateInfo, &gfx_state->vk, needed_lib_flags);

      gfx_state->stages = calloc(MESA_VULKAN_SHADER_STAGES, sizeof(struct radv_shader_stage));
      if (!gfx_state->stages) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }

      for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         gfx_state->stages[i].stage = MESA_SHADER_NONE;
         gfx_state->stages[i].next_stage = MESA_SHADER_NONE;
      }

      for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
         const VkPipelineShaderStageCreateInfo *sinfo = &pCreateInfo->pStages[i];
         mesa_shader_stage stage = vk_to_mesa_shader_stage(sinfo->stage);

         radv_pipeline_stage_init(create_flags, sinfo, &gfx_state->layout, &gfx_state->key.stage_info[stage],
                                  &gfx_state->stages[stage]);
      }

      radv_pipeline_load_retained_shaders(device, pCreateInfo, gfx_state->stages);
   }

   return VK_SUCCESS;

fail:
   radv_graphics_pipeline_state_finish(device, gfx_state);
   return result;
}

void
radv_graphics_pipeline_hash(const struct radv_device *device, const struct radv_graphics_pipeline_state *gfx_state,
                            unsigned char *hash)
{
   blake3_hasher ctx;

   _mesa_blake3_init(&ctx);
   radv_pipeline_hash(device, &gfx_state->layout, &ctx);

   _mesa_blake3_update(&ctx, &gfx_state->key.gfx_state, sizeof(gfx_state->key.gfx_state));

   _mesa_blake3_update(&ctx, gfx_state->libs_blake3, sizeof(gfx_state->libs_blake3));

   if (gfx_state->stages) {
      for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
         const struct radv_shader_stage *stage = &gfx_state->stages[s];

         if (stage->stage == MESA_SHADER_NONE)
            continue;

         _mesa_blake3_update(&ctx, stage->shader_blake3, sizeof(stage->shader_blake3));
         _mesa_blake3_update(&ctx, &stage->key, sizeof(stage->key));
      }
   }

   _mesa_blake3_final(&ctx, hash);
}

static VkResult
radv_graphics_pipeline_compile(struct radv_graphics_pipeline *pipeline, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                               const struct radv_graphics_pipeline_state *gfx_state, struct radv_device *device,
                               struct vk_pipeline_cache *cache, bool fast_linking_enabled)
{
   const struct radv_compiler_info *compiler_info = &device->compiler_info;
   struct radv_shader_binary *binaries[MESA_VULKAN_SHADER_STAGES] = {NULL};
   struct radv_shader_binary *gs_copy_binary = NULL;
   bool skip_shaders_cache = radv_pipeline_skip_shaders_cache(device, &pipeline->base);
   struct radv_shader_stage *stages = gfx_state->stages;
   const VkPipelineCreationFeedbackCreateInfo *creation_feedback =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_CREATION_FEEDBACK_CREATE_INFO);
   VkPipelineCreationFeedback pipeline_feedback = {
      .flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT,
   };
   VkResult result = VK_SUCCESS;
   const bool retain_shaders =
      !!(pipeline->base.create_flags & VK_PIPELINE_CREATE_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT);
   struct radv_retained_shaders *retained_shaders = NULL;

   int64_t pipeline_start = os_time_get_nano();

   /* Skip the shaders cache when any of the below are true:
    * - fast-linking is enabled because it's useless to cache unoptimized pipelines
    * - graphics pipeline libraries are created with the RETAIN_LINK_TIME_OPTIMIZATION flag and
    *   module identifiers are used (ie. no SPIR-V provided).
    */
   if (fast_linking_enabled) {
      skip_shaders_cache = true;
   } else if (retain_shaders) {
      assert(pipeline->base.create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR);
      for (uint32_t i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         if (stages[i].stage != MESA_SHADER_NONE && !stages[i].spirv.size) {
            skip_shaders_cache = true;
            break;
         }
      }
   }

   bool found_in_application_cache = true;
   if (!skip_shaders_cache &&
       radv_graphics_pipeline_cache_search(device, cache, pipeline, &found_in_application_cache)) {
      if (found_in_application_cache)
         pipeline_feedback.flags |= VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;

      if (retain_shaders) {
         /* For graphics pipeline libraries created with the RETAIN_LINK_TIME_OPTIMIZATION flag, we
          * need to retain the stage info because we can't know if the LTO pipelines will
          * be find in the shaders cache.
          */
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(&pipeline->base);

         gfx_pipeline_lib->stages = radv_copy_shader_stage_create_info(device, pCreateInfo->stageCount,
                                                                       pCreateInfo->pStages, gfx_pipeline_lib->mem_ctx);
         if (!gfx_pipeline_lib->stages)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

         gfx_pipeline_lib->stage_count = pCreateInfo->stageCount;

         for (unsigned i = 0; i < pCreateInfo->stageCount; i++) {
            mesa_shader_stage s = vk_to_mesa_shader_stage(pCreateInfo->pStages[i].stage);
            gfx_pipeline_lib->stage_keys[s] = gfx_state->key.stage_info[s];
         }
      }

      result = VK_SUCCESS;
      goto done;
   }

   if (pipeline->base.create_flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)
      return VK_PIPELINE_COMPILE_REQUIRED;

   if (retain_shaders) {
      struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(&pipeline->base);
      retained_shaders = &gfx_pipeline_lib->retained_shaders;
   }

   const bool noop_fs = radv_pipeline_needs_noop_fs(pipeline, &gfx_state->key.gfx_state);

   struct radv_shader_debug_info debug[MESA_VULKAN_SHADER_STAGES] = {0};
   struct radv_shader_debug_info gs_copy_debug = {0};
   const VkShaderStageFlags nir_cache_misses = radv_graphics_shaders_lookup_nir(compiler_info, cache, stages);
   radv_graphics_shaders_spirv_to_nir(compiler_info, stages, pipeline->base.is_internal);
   radv_graphics_shaders_insert_nir(compiler_info, cache, stages, nir_cache_misses);

   if (retained_shaders)
      radv_pipeline_retain_shaders(retained_shaders, stages);

   radv_graphics_shaders_compile(compiler_info, stages, &gfx_state->key.gfx_state, noop_fs, debug, binaries,
                                 &gs_copy_debug, &gs_copy_binary);
   radv_graphics_shaders_create(device, cache, skip_shaders_cache, pipeline->base.shaders, binaries, debug,
                                &pipeline->base.gs_copy_shader, gs_copy_binary, &gs_copy_debug);

   if (!skip_shaders_cache) {
      radv_pipeline_cache_insert(device, cache, &pipeline->base);
   }

   free(gs_copy_binary);
   for (int i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      free(binaries[i]);
      if (stages[i].nir) {
         if (radv_can_dump_shader_stats(&device->compiler_info, stages[i].nir) && pipeline->base.shaders[i]) {
            radv_dump_shader_stats(device, &pipeline->base, pipeline->base.shaders[i], stderr);
         }
      }
   }

done:
   pipeline_feedback.duration = os_time_get_nano() - pipeline_start;

   if (creation_feedback) {
      *creation_feedback->pPipelineCreationFeedback = pipeline_feedback;

      if (creation_feedback->pipelineStageCreationFeedbackCount > 0) {
         uint32_t num_feedbacks = 0;

         for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
            mesa_shader_stage s = vk_to_mesa_shader_stage(pCreateInfo->pStages[i].stage);
            creation_feedback->pPipelineStageCreationFeedbacks[num_feedbacks++] = stages[s].feedback;
         }

         /* Stages imported from graphics pipeline libraries are defined as additional entries in the
          * order they were imported.
          */
         const VkPipelineLibraryCreateInfoKHR *libs_info =
            vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
         if (libs_info) {
            for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
               VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
               struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

               if (!gfx_pipeline_lib->base.active_stages)
                  continue;

               radv_foreach_stage (s, gfx_pipeline_lib->base.active_stages) {
                  creation_feedback->pPipelineStageCreationFeedbacks[num_feedbacks++] = stages[s].feedback;
               }
            }
         }

         assert(num_feedbacks == creation_feedback->pipelineStageCreationFeedbackCount);
      }
   }

   return result;
}

static void
radv_pipeline_init_extra(struct radv_graphics_pipeline *pipeline, const VkGraphicsPipelineCreateInfoRADV *radv_info,
                         const struct vk_graphics_pipeline_state *state)
{
   pipeline->custom_blend_mode = radv_info->custom_blend_mode;
}

static VkResult
radv_graphics_pipeline_import_binaries(struct radv_device *device, struct radv_graphics_pipeline *pipeline,
                                       const VkPipelineBinaryInfoKHR *binary_info)
{
   struct mesa_blake3 ctx;

   _mesa_blake3_init(&ctx);

   for (uint32_t i = 0; i < binary_info->binaryCount; i++) {
      VK_FROM_HANDLE(radv_pipeline_binary, pipeline_binary, binary_info->pPipelineBinaries[i]);
      struct radv_shader *shader;
      struct blob_reader blob;

      blob_reader_init(&blob, pipeline_binary->data, pipeline_binary->size);

      shader = radv_shader_deserialize(device, pipeline_binary->key, sizeof(pipeline_binary->key), &blob);
      if (!shader)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      if (shader->info.stage == MESA_SHADER_VERTEX && i > 0) {
         /* The GS copy-shader is a VS placed after all other stages. */
         pipeline->base.gs_copy_shader = shader;
      } else {
         pipeline->base.shaders[shader->info.stage] = shader;
      }

      _mesa_blake3_update(&ctx, pipeline_binary->key, sizeof(pipeline_binary->key));
   }

   _mesa_blake3_final(&ctx, pipeline->base.blake3);

   pipeline->has_pipeline_binaries = true;

   return VK_SUCCESS;
}

static VkResult
radv_graphics_pipeline_init(struct radv_graphics_pipeline *pipeline, struct radv_device *device,
                            struct vk_pipeline_cache *cache, const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   bool fast_linking_enabled = radv_is_fast_linking_enabled(pCreateInfo);
   struct radv_graphics_pipeline_state gfx_state;
   VkResult result = VK_SUCCESS;

   pipeline->last_vgt_api_stage = MESA_SHADER_NONE;

   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);

   const VkPipelineBinaryInfoKHR *binary_info = vk_find_struct_const(pCreateInfo->pNext, PIPELINE_BINARY_INFO_KHR);
   const bool import_pipeline_binaries = binary_info && binary_info->binaryCount > 0;

   /* If we have libraries, import them first. */
   if (libs_info) {
      for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

         assert(pipeline_lib->type == RADV_PIPELINE_GRAPHICS_LIB);

         radv_graphics_pipeline_import_lib(device, pipeline, gfx_pipeline_lib, import_pipeline_binaries);
      }
   }

   radv_pipeline_import_graphics_info(device, pipeline, pCreateInfo);

   result = radv_generate_graphics_pipeline_state(device, pCreateInfo, &gfx_state);
   if (result != VK_SUCCESS)
      return result;

   if (import_pipeline_binaries) {
      result = radv_graphics_pipeline_import_binaries(device, pipeline, binary_info);
   } else {
      radv_graphics_pipeline_hash(device, &gfx_state, pipeline->base.blake3);

      if (gfx_state.compilation_required) {
         result =
            radv_graphics_pipeline_compile(pipeline, pCreateInfo, &gfx_state, device, cache, fast_linking_enabled);
      }
   }

   if (result != VK_SUCCESS) {
      radv_graphics_pipeline_state_finish(device, &gfx_state);
      return result;
   }

   pipeline->base.pipeline_hash = *(uint64_t *)pipeline->base.blake3;

   if (!radv_pipeline_has_stage(pipeline, MESA_SHADER_MESH))
      radv_pipeline_init_input_assembly_state(device, pipeline);
   radv_pipeline_init_dynamic_state(device, pipeline, &gfx_state.vk);

   pipeline->uses_out_of_order_rast = gfx_state.vk.rs->rasterization_order_amd == VK_RASTERIZATION_ORDER_RELAXED_AMD;
   pipeline->uses_vrs_attachment = radv_pipeline_uses_vrs_attachment(pipeline, &gfx_state.vk);
   pipeline->ignore_ds_state = gfx_state.vk.rp && !radv_pipeline_has_ds_attachments(gfx_state.vk.rp);

   const VkGraphicsPipelineCreateInfoRADV *radv_info =
      vk_find_struct_const(pCreateInfo->pNext, GRAPHICS_PIPELINE_CREATE_INFO_RADV);
   if (radv_info) {
      radv_pipeline_init_extra(pipeline, radv_info, &gfx_state.vk);
   }

   radv_graphics_pipeline_state_finish(device, &gfx_state);
   return result;
}

static VkResult
radv_graphics_pipeline_create(VkDevice _device, VkPipelineCache _cache, const VkGraphicsPipelineCreateInfo *pCreateInfo,
                              const VkAllocationCallbacks *pAllocator, VkPipeline *pPipeline)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(vk_pipeline_cache, cache, _cache);
   struct radv_graphics_pipeline *pipeline;
   VkResult result;

   pipeline = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pipeline), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pipeline == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   radv_pipeline_init(device, &pipeline->base, RADV_PIPELINE_GRAPHICS);
   pipeline->base.create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);
   pipeline->base.is_internal = _cache == device->meta_state.cache;

   result = radv_graphics_pipeline_init(pipeline, device, cache, pCreateInfo);
   if (result != VK_SUCCESS) {
      radv_pipeline_destroy(device, &pipeline->base, pAllocator);
      return result;
   }

   radv_pipeline_report_pso_history(device, &pipeline->base);

   *pPipeline = radv_pipeline_to_handle(&pipeline->base);
   radv_rmv_log_graphics_pipeline_create(device, &pipeline->base, pipeline->base.is_internal);
   return VK_SUCCESS;
}

void
radv_destroy_graphics_pipeline(struct radv_device *device, struct radv_graphics_pipeline *pipeline)
{
   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      if (pipeline->base.shaders[i])
         radv_shader_unref(device, pipeline->base.shaders[i]);
   }

   if (pipeline->base.gs_copy_shader)
      radv_shader_unref(device, pipeline->base.gs_copy_shader);
}

static VkResult
radv_graphics_lib_pipeline_init(struct radv_graphics_lib_pipeline *pipeline, struct radv_device *device,
                                struct vk_pipeline_cache *cache, const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   VK_FROM_HANDLE(radv_pipeline_layout, pipeline_layout, pCreateInfo->layout);
   VkResult result;

   const VkGraphicsPipelineLibraryCreateInfoEXT *lib_info =
      vk_find_struct_const(pCreateInfo->pNext, GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
   const VkPipelineLibraryCreateInfoKHR *libs_info =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);
   bool fast_linking_enabled = radv_is_fast_linking_enabled(pCreateInfo);

   struct vk_graphics_pipeline_state *state = &pipeline->graphics_state;

   pipeline->base.last_vgt_api_stage = MESA_SHADER_NONE;
   pipeline->lib_flags = lib_info ? lib_info->flags : 0;

   radv_pipeline_layout_init(device, &pipeline->layout, false);

   const VkPipelineBinaryInfoKHR *binary_info = vk_find_struct_const(pCreateInfo->pNext, PIPELINE_BINARY_INFO_KHR);
   const bool import_pipeline_binaries = binary_info && binary_info->binaryCount > 0;

   /* If we have libraries, import them first. */
   if (libs_info) {
      for (uint32_t i = 0; i < libs_info->libraryCount; i++) {
         VK_FROM_HANDLE(radv_pipeline, pipeline_lib, libs_info->pLibraries[i]);
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(pipeline_lib);

         vk_graphics_pipeline_state_merge(state, &gfx_pipeline_lib->graphics_state);

         radv_pipeline_layout_merge(&pipeline->layout, &gfx_pipeline_lib->layout);

         radv_graphics_pipeline_import_lib(device, &pipeline->base, gfx_pipeline_lib, import_pipeline_binaries);

         pipeline->lib_flags |= gfx_pipeline_lib->lib_flags;
      }
   }

   result = vk_graphics_pipeline_state_fill(&device->vk, state, pCreateInfo, NULL /* driver_mv */, NULL /* driver_rp */,
                                            0, NULL, NULL, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, &pipeline->state_data);
   if (result != VK_SUCCESS)
      return result;

   radv_pipeline_import_graphics_info(device, &pipeline->base, pCreateInfo);

   if (pipeline_layout)
      radv_pipeline_layout_merge(&pipeline->layout, pipeline_layout);

   if (import_pipeline_binaries) {
      result = radv_graphics_pipeline_import_binaries(device, &pipeline->base, binary_info);
   } else {
      struct radv_graphics_pipeline_state gfx_state;

      result = radv_generate_graphics_pipeline_state(device, pCreateInfo, &gfx_state);
      if (result != VK_SUCCESS)
         return result;

      radv_graphics_pipeline_hash(device, &gfx_state, pipeline->base.base.blake3);

      result =
         radv_graphics_pipeline_compile(&pipeline->base, pCreateInfo, &gfx_state, device, cache, fast_linking_enabled);

      radv_graphics_pipeline_state_finish(device, &gfx_state);
   }

   if (result != VK_SUCCESS)
      return result;

   pipeline->base.base.pipeline_hash = *(uint64_t *)pipeline->base.base.blake3;

   return result;
}

static VkResult
radv_graphics_lib_pipeline_create(VkDevice _device, VkPipelineCache _cache,
                                  const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                  const VkAllocationCallbacks *pAllocator, VkPipeline *pPipeline)
{
   VK_FROM_HANDLE(vk_pipeline_cache, cache, _cache);
   VK_FROM_HANDLE(radv_device, device, _device);
   struct radv_graphics_lib_pipeline *pipeline;
   VkResult result;

   pipeline = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pipeline), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pipeline == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   radv_pipeline_init(device, &pipeline->base.base, RADV_PIPELINE_GRAPHICS_LIB);
   pipeline->base.base.create_flags = vk_graphics_pipeline_create_flags(pCreateInfo);

   pipeline->mem_ctx = ralloc_context(NULL);

   result = radv_graphics_lib_pipeline_init(pipeline, device, cache, pCreateInfo);
   if (result != VK_SUCCESS) {
      radv_pipeline_destroy(device, &pipeline->base.base, pAllocator);
      return result;
   }

   radv_pipeline_report_pso_history(device, &pipeline->base.base);

   *pPipeline = radv_pipeline_to_handle(&pipeline->base.base);

   return VK_SUCCESS;
}

void
radv_destroy_graphics_lib_pipeline(struct radv_device *device, struct radv_graphics_lib_pipeline *pipeline)
{
   struct radv_retained_shaders *retained_shaders = &pipeline->retained_shaders;

   radv_pipeline_layout_finish(device, &pipeline->layout);

   vk_free(&device->vk.alloc, pipeline->state_data);

   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      free(retained_shaders->stages[i].serialized_nir);
   }

   ralloc_free(pipeline->mem_ctx);

   radv_destroy_graphics_pipeline(device, &pipeline->base);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_CreateGraphicsPipelines(VkDevice _device, VkPipelineCache pipelineCache, uint32_t count,
                             const VkGraphicsPipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator,
                             VkPipeline *pPipelines)
{
   VkResult result = VK_SUCCESS;
   unsigned i = 0;

   for (; i < count; i++) {
      const VkPipelineCreateFlagBits2 create_flags = vk_graphics_pipeline_create_flags(&pCreateInfos[i]);
      VkResult r;
      if (create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
         r = radv_graphics_lib_pipeline_create(_device, pipelineCache, &pCreateInfos[i], pAllocator, &pPipelines[i]);
      } else {
         r = radv_graphics_pipeline_create(_device, pipelineCache, &pCreateInfos[i], pAllocator, &pPipelines[i]);
      }
      if (r != VK_SUCCESS) {
         result = r;
         pPipelines[i] = VK_NULL_HANDLE;

         if (create_flags & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT)
            break;
      }
   }

   for (; i < count; ++i)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}
