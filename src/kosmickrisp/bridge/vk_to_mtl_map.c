/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vk_to_mtl_map.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/format/u_format.h"

#include "vulkan/vulkan.h"
#include "vk_meta.h"

struct mtl_origin
vk_offset_3d_to_mtl_origin(const struct VkOffset3D *offset)
{
   struct mtl_origin ret = {
      .x = offset->x,
      .y = offset->y,
      .z = offset->z,
   };
   return ret;
}

struct mtl_size
vk_extent_3d_to_mtl_size(const struct VkExtent3D *extent)
{
   struct mtl_size ret = {
      .x = extent->width,
      .y = extent->height,
      .z = extent->depth,
   };
   return ret;
}

enum mtl_primitive_type
vk_primitive_topology_to_mtl_primitive_type(enum VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return MTL_PRIMITIVE_TYPE_POINT;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return MTL_PRIMITIVE_TYPE_LINE;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return MTL_PRIMITIVE_TYPE_LINE_STRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
   case VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA:
#pragma GCC diagnostic pop
      /* Triangle fans are emulated meaning we'll translate the index buffer to
       * triangle list or generate a index buffer if there's none */
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return MTL_PRIMITIVE_TYPE_TRIANGLE;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP;
   default:
      UNREACHABLE("Primitive topology not supported!");
   }
}

enum mtl_primitive_topology_class
vk_primitive_topology_to_mtl_primitive_topology_class(
   enum VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return MTL_PRIMITIVE_TOPOLOGY_CLASS_POINT;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
   /* Adjacency draws are unrolled to lines or triangles */
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
      return MTL_PRIMITIVE_TOPOLOGY_CLASS_LINE;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
   case VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA:
#pragma GCC diagnostic pop
      return MTL_PRIMITIVE_TOPOLOGY_CLASS_TRIANGLE;
   default:
      UNREACHABLE("Primitive topology not supported!");
   }
}

enum mtl_load_action
vk_attachment_load_op_to_mtl_load_action(enum VkAttachmentLoadOp op)
{
   switch (op) {
   case VK_ATTACHMENT_LOAD_OP_LOAD:
      return MTL_LOAD_ACTION_LOAD;
   case VK_ATTACHMENT_LOAD_OP_CLEAR:
      return MTL_LOAD_ACTION_CLEAR;
   case VK_ATTACHMENT_LOAD_OP_DONT_CARE:
   case VK_ATTACHMENT_LOAD_OP_NONE:
      return MTL_LOAD_ACTION_DONT_CARE;
   default:
      assert(false && "Unsupported VkAttachmentLoadOp");
      return MTL_LOAD_ACTION_DONT_CARE;
   };
}

enum mtl_store_action
vk_attachment_store_op_to_mtl_store_action(enum VkAttachmentStoreOp op)
{
   switch (op) {
   case VK_ATTACHMENT_STORE_OP_STORE:
      return MTL_STORE_ACTION_STORE;
   case VK_ATTACHMENT_STORE_OP_DONT_CARE:
   case VK_ATTACHMENT_STORE_OP_NONE:
      return MTL_STORE_ACTION_DONT_CARE;
   default:
      assert(false && "Unsupported VkAttachmentStoreOp");
      return MTL_STORE_ACTION_UNKNOWN;
   };
}

enum mtl_sampler_address_mode
vk_sampler_address_mode_to_mtl_sampler_address_mode(
   enum VkSamplerAddressMode mode)
{
   switch (mode) {
   case VK_SAMPLER_ADDRESS_MODE_REPEAT:
      return MTL_SAMPLER_ADDRESS_MODE_REPEAT;
   case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
      return MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
      return MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
      return MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR;
   case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE:
      return MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
   default:
      UNREACHABLE("Unsupported address mode");
   }
}

enum mtl_sampler_border_color
vk_border_color_to_mtl_sampler_border_color(enum VkBorderColor color)
{
   switch (color) {
   case VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
   case VK_BORDER_COLOR_INT_TRANSPARENT_BLACK:
      return MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK;
   case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
   case VK_BORDER_COLOR_INT_OPAQUE_BLACK:
      return MTL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK;
   case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
   case VK_BORDER_COLOR_INT_OPAQUE_WHITE:
      return MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE;
   default:
      UNREACHABLE("Unsupported address mode");
   }
}

enum mtl_sampler_min_mag_filter
vk_filter_to_mtl_sampler_min_mag_filter(enum VkFilter filter)
{
   switch (filter) {
   case VK_FILTER_NEAREST:
      return MTL_SAMPLER_MIN_MAG_FILTER_NEAREST;
   case VK_FILTER_LINEAR:
      return MTL_SAMPLER_MIN_MAG_FILTER_LINEAR;
   default:
      UNREACHABLE("Unsupported address mode");
   }
}

enum mtl_sampler_mip_filter
vk_sampler_mipmap_mode_to_mtl_sampler_mip_filter(enum VkSamplerMipmapMode mode)
{
   switch (mode) {
   case VK_SAMPLER_MIPMAP_MODE_NEAREST:
      return MTL_SAMPLER_MIP_FILTER_NEAREST;
   case VK_SAMPLER_MIPMAP_MODE_LINEAR:
      return MTL_SAMPLER_MIP_FILTER_LINEAR;
   default:
      UNREACHABLE("Unsupported address mode");
   }
}

enum mtl_sampler_reduction_mode
vk_sampler_reduction_mode_to_mtl_sampler_reduction_mode(
   enum VkSamplerReductionMode mode)
{
   switch (mode) {
   case VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE:
      return MTL_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE;
   case VK_SAMPLER_REDUCTION_MODE_MIN:
      return MTL_SAMPLER_REDUCTION_MODE_MINIMUM;
   case VK_SAMPLER_REDUCTION_MODE_MAX:
      return MTL_SAMPLER_REDUCTION_MODE_MAXIMUM;
   default:
      UNREACHABLE("Unsupported sampler reduction mode");
   }
}

enum mtl_compare_function
vk_compare_op_to_mtl_compare_function(enum VkCompareOp op)
{
   switch (op) {
   case VK_COMPARE_OP_NEVER:
      return MTL_COMPARE_FUNCTION_NEVER;
   case VK_COMPARE_OP_LESS:
      return MTL_COMPARE_FUNCTION_LESS;
   case VK_COMPARE_OP_EQUAL:
      return MTL_COMPARE_FUNCTION_EQUAL;
   case VK_COMPARE_OP_LESS_OR_EQUAL:
      return MTL_COMPARE_FUNCTION_LESS_EQUAL;
   case VK_COMPARE_OP_GREATER:
      return MTL_COMPARE_FUNCTION_GREATER;
   case VK_COMPARE_OP_NOT_EQUAL:
      return MTL_COMPARE_FUNCTION_NOT_EQUAL;
   case VK_COMPARE_OP_GREATER_OR_EQUAL:
      return MTL_COMPARE_FUNCTION_GREATER_EQUAL;
   case VK_COMPARE_OP_ALWAYS:
      return MTL_COMPARE_FUNCTION_ALWAYS;
   default:
      UNREACHABLE("Unsupported address mode");
   }
}

enum mtl_winding
vk_front_face_to_mtl_winding(enum VkFrontFace face)
{
   switch (face) {
   case VK_FRONT_FACE_CLOCKWISE:
      return MTL_WINDING_CLOCKWISE;
   case VK_FRONT_FACE_COUNTER_CLOCKWISE:
      return MTL_WINDING_COUNTER_CLOCKWISE;
   default:
      assert(false && "Unsupported VkFrontFace");
      return MTL_WINDING_CLOCKWISE;
   }
}

enum mtl_cull_mode
vk_front_face_to_mtl_cull_mode(enum VkCullModeFlagBits mode)
{
   switch (mode) {
   case VK_CULL_MODE_NONE:
   case VK_CULL_MODE_FRONT_AND_BACK: // Emulated with scissor
      return MTL_CULL_MODE_NONE;
   case VK_CULL_MODE_FRONT_BIT:
      return MTL_CULL_MODE_FRONT;
   case VK_CULL_MODE_BACK_BIT:
      return MTL_CULL_MODE_BACK;
   default:
      UNREACHABLE("Unsupported VkCullModeFlags");
   }
}

enum mtl_depth_resolve_filter
vk_resolve_mode_to_mtl_depth_resolve_filter(
   enum VkResolveModeFlagBits resolve_mode)
{
   switch (resolve_mode) {
   case VK_RESOLVE_MODE_SAMPLE_ZERO_BIT:
      return MTL_DEPTH_RESOLVE_FILTER_SAMPLE_0;
   case VK_RESOLVE_MODE_MIN_BIT:
      return MTL_DEPTH_RESOLVE_FILTER_MIN;
   case VK_RESOLVE_MODE_MAX_BIT:
      return MTL_DEPTH_RESOLVE_FILTER_MAX;
   default:
      UNREACHABLE("Unsupported VkResolveMode for depth");
   }
}

enum mtl_stencil_resolve_filter
vk_resolve_mode_to_mtl_stencil_resolve_filter(
   enum VkResolveModeFlagBits resolve_mode)
{
   switch (resolve_mode) {
   case VK_RESOLVE_MODE_SAMPLE_ZERO_BIT:
      return MTL_STENCIL_RESOLVE_FILTER_SAMPLE_0;
   default:
      UNREACHABLE("Unsupported VkResolveMode for stencil");
   }
}

enum mtl_index_type
index_size_in_bytes_to_mtl_index_type(unsigned bytes)
{
   switch (bytes) {
   case 2u:
      return MTL_INDEX_TYPE_UINT16;
   case 4u:
      return MTL_INDEX_TYPE_UINT32;
   default:
      UNREACHABLE("Unsupported byte size for index");
   }
}

unsigned
mtl_index_type_to_size_B(enum mtl_index_type type)
{
   switch (type) {
   case MTL_INDEX_TYPE_UINT16:
      return 2u;
   case MTL_INDEX_TYPE_UINT32:
      return 4u;
   default:
      UNREACHABLE("Unhandled index type");
   }
}

const char *
mtl_command_queue_error_to_string(enum mtl_command_queue_error error)
{
   switch (error) {
   case MTL_COMMAND_QUEUE_ERROR_NONE:
      return "MTL_COMMAND_QUEUE_ERROR_NONE";
   case MTL_COMMAND_QUEUE_ERROR_TIMEOUT:
      return "MTL_COMMAND_QUEUE_ERROR_TIMEOUT";
   case MTL_COMMAND_QUEUE_ERROR_NOT_PERMITTED:
      return "MTL_COMMAND_QUEUE_ERROR_NOT_PERMITTED";
   case MTL_COMMAND_QUEUE_ERROR_OUT_OF_MEMORY:
      return "MTL_COMMAND_QUEUE_ERROR_OUT_OF_MEMORY";
   case MTL_COMMAND_QUEUE_ERROR_ACCESS_REVOKED:
      return "MTL_COMMAND_QUEUE_ERROR_ACCESS_REVOKED";
   case MTL_COMMAND_QUEUE_ERROR_INTERNAL:
      return "MTL_COMMAND_QUEUE_ERROR_INTERNAL";
   default:
      UNREACHABLE("Unsupported error value");
   }
}

enum mtl_stages
vk_pipeline_stages_to_mtl_stages(VkPipelineStageFlagBits2 stages, bool is_src,
                                 bool cond_rendering_ext_enabled)
{
   enum mtl_stages result = 0;
   u_foreach_bit64(i, (uint64_t)stages) {
      VkPipelineStageFlagBits2 stage = (VkPipelineStageFlagBits2)(1ul << i);
      if (stage == VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT ||
          stage == VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT ||
          stage == VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT ||
          stage == VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT) {
         /* We can ignore unrolling here because that does the required barrier
          * itself. */
         result |= MTL_STAGE_VERTEX;
      } else if (stage == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT ||
                 stage == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT ||
                 stage == VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT ||
                 stage == VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT) {
         result |= MTL_STAGE_FRAGMENT;
      } else if (stage == VK_PIPELINE_STAGE_2_CLEAR_BIT) {
         /* TODO: Dispatch is required because of vkCmdUpdateBuffer and
          * vkCmdFillBuffer, implement those as copies to get rid of Dispatch
          * here. */
         result |= MTL_STAGE_BLIT | MTL_STAGE_FRAGMENT | MTL_STAGE_DISPATCH;
      } else if (stage == VK_PIPELINE_STAGE_2_COPY_BIT) {
         /* vkCmdCopyQueryResults does the required barrier itself,
          * so we only need BLIT here. */
         result |= MTL_STAGE_BLIT;
      } else if (stage == VK_PIPELINE_STAGE_2_RESOLVE_BIT) {
         result |= MTL_STAGE_FRAGMENT;
      } else if (stage == VK_PIPELINE_STAGE_2_BLIT_BIT) {
         result |= MTL_STAGE_BLIT | MTL_STAGE_FRAGMENT;
      } else if (stage == VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT) {
         /* vkCmdWriteBufferMarker2AMD does the required barrier itself,
          * so we don't need DISPATCH for that here. */
         /* TODO: Implement vkCmdUpdateBuffer & vkCmdFillBuffer as copies
          *       to get rid of DISPATCH. See above. */
         result |= MTL_STAGE_BLIT | MTL_STAGE_FRAGMENT | MTL_STAGE_DISPATCH;
      } else if (stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) {
         result |= MTL_STAGE_DISPATCH;
      } else if (stage == VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) {
         result |= MTL_STAGE_DISPATCH | MTL_STAGE_VERTEX;
      } else if (stage == VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT) {
         result |= MTL_STAGE_DISPATCH;
      } else if (stage == VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT) {
         result |= MTL_STAGE_MESH;
      } else if (stage == VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT) {
         result |= MTL_STAGE_OBJECTS;
      } else if (stage ==
                    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR ||
                 stage ==
                    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR) {
         result |= MTL_STAGE_ACCELERATION_STRUCTURE;
      } else if (stage ==
                    VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT ||
                 stage == VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT ||
                 stage == VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT) {
         /* Emulated stages do barrier beforehand anyway which covers DISPATCH.
          * Tess Eval or Geometry can be compiled to a Metal vertex shader, and
          * we need at least one stage for execution dependency chain tracking. */
         result |= MTL_STAGE_VERTEX;
      } else if (stage == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) {
         result |= is_src ? 0u : MTL_STAGE_ALL;
      } else if (stage == VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT) {
         result |= is_src ? MTL_STAGE_ALL : 0u;
      } else if (stage == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) {
         result |= MTL_STAGE_ALL;
      } else if (stage == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT) {
         result |= MTL_STAGES_ALL_RENDER_ENCODER;
         if (cond_rendering_ext_enabled)
            result |= MTL_STAGE_DISPATCH;
      } else if (stage == VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT) {
         result |= MTL_STAGES_ALL_RENDER_ENCODER &
                   ~(MTL_STAGE_FRAGMENT | MTL_STAGE_TILE);
      } else if (stage == VK_PIPELINE_STAGE_2_HOST_BIT) {
         /* HOST is a pseudo-stage that cannot build dependency chains. */
      } else {
         assert(false);
      }
   }
   return result;
}

void
vk_dependency_info_to_mtl(const struct VkDependencyInfo *pDependencyInfo,
                          enum mtl_stages *after_stages,
                          enum mtl_stages *before_stages,
                          bool cond_rendering_ext_enabled)
{
   VkPipelineStageFlagBits2 src_stages_temp = 0u;
   VkPipelineStageFlagBits2 dst_stages_temp = 0u;
   for (uint i = 0u; i < pDependencyInfo->memoryBarrierCount; i++) {
      src_stages_temp |= pDependencyInfo->pMemoryBarriers[i].srcStageMask;
      dst_stages_temp |= pDependencyInfo->pMemoryBarriers[i].dstStageMask;
   }
   for (uint i = 0u; i < pDependencyInfo->imageMemoryBarrierCount; i++) {
      src_stages_temp |= pDependencyInfo->pImageMemoryBarriers[i].srcStageMask;
      dst_stages_temp |= pDependencyInfo->pImageMemoryBarriers[i].dstStageMask;
   }
   for (uint i = 0u; i < pDependencyInfo->bufferMemoryBarrierCount; i++) {
      src_stages_temp |= pDependencyInfo->pBufferMemoryBarriers[i].srcStageMask;
      dst_stages_temp |= pDependencyInfo->pBufferMemoryBarriers[i].dstStageMask;
   }

   if (after_stages)
      *after_stages = vk_pipeline_stages_to_mtl_stages(
         src_stages_temp, true, cond_rendering_ext_enabled);
   if (before_stages)
      *before_stages = vk_pipeline_stages_to_mtl_stages(
         dst_stages_temp, false, cond_rendering_ext_enabled);
}
