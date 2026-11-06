/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

/* Vulkan graphics state to register translation, shared by pipelines and command buffers. */

#ifndef RADV_GRAPHICS_STATE_H
#define RADV_GRAPHICS_STATE_H

#include "amdgfxregs.h"

#include "radv_constants.h"

#include "vk_graphics_state.h"
#include "vk_internal_exts.h"

struct radv_physical_device;

struct radv_sample_locations_state {
   VkSampleCountFlagBits per_pixel;
   VkExtent2D grid_size;
   uint32_t count;
   VkSampleLocationEXT locations[MAX_SAMPLE_LOCATIONS];

   /* Derived state. */
   int8_t hw_locations[4][8][2]; /* [pixel in quad][sample][dim] */
   bool xmax_right_exclusion;
   bool ymax_bottom_exclusion;
   bool allow_small_prim_ngg_culling;
   uint8_t log2_small_prim_ngg_culling_scaling_factor;
};

struct radv_viewport_xform_state {
   float scale[3];
   float translate[3];
};

struct radv_blend_equation_state {
   struct {
      uint32_t cb_blend_control;
      uint32_t sx_mrt_blend_opt;
   } att[MAX_RTS];

   bool mrt0_is_dual_src;
};

struct radv_vertex_input_state {
   uint32_t attribute_mask;

   uint32_t instance_rate_inputs;
   uint32_t nontrivial_divisors;
   uint32_t zero_divisors;
   uint32_t post_shuffle;
   /* Having two separate fields instead of a single uint64_t makes it easier to remove attributes
    * using bitwise arithmetic.
    */
   uint32_t alpha_adjust_lo;
   uint32_t alpha_adjust_hi;
   uint32_t nontrivial_formats;

   uint8_t bindings[MAX_VERTEX_ATTRIBS];
   uint32_t divisors[MAX_VERTEX_ATTRIBS];
   uint32_t offsets[MAX_VERTEX_ATTRIBS];
   uint8_t formats[MAX_VERTEX_ATTRIBS];
   uint8_t format_align_req_minus_1[MAX_VERTEX_ATTRIBS];
   uint8_t component_align_req_minus_1[MAX_VERTEX_ATTRIBS];
   uint8_t format_sizes[MAX_VERTEX_ATTRIBS];
   uint32_t non_trivial_format[MAX_VERTEX_ATTRIBS];

   uint32_t vbo_misaligned_mask;
   uint32_t vbo_unaligned_mask;
   uint32_t vbo_misaligned_mask_invalid;

   bool bindings_match_attrib;
};

struct radv_dynamic_state {
   struct vk_dynamic_graphics_state vk;

   /**
    * Bitmask of (1ull << VK_DYNAMIC_STATE_*).
    * Defines the set of saved dynamic state.
    */
   uint64_t mask;

   struct radv_viewport_xform_state vp_xform[MAX_VIEWPORTS];

   struct radv_vertex_input_state vertex_input;

   struct radv_sample_locations_state sample_location;

   VkImageAspectFlags feedback_loop_aspects;

   uint32_t color_write_enable;
   uint32_t color_write_mask;
   uint8_t color_blend_enable;

   struct radv_blend_equation_state blend_eq;
};

static inline uint32_t
radv_translate_prim(unsigned topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return V_008958_DI_PT_POINTLIST;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return V_008958_DI_PT_LINELIST;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return V_008958_DI_PT_LINESTRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return V_008958_DI_PT_TRILIST;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return V_008958_DI_PT_TRISTRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return V_008958_DI_PT_TRIFAN;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      return V_008958_DI_PT_LINELIST_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
      return V_008958_DI_PT_LINESTRIP_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      return V_008958_DI_PT_TRILIST_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
      return V_008958_DI_PT_TRISTRIP_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
      return V_008958_DI_PT_PATCH;
   case VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA:
      return V_008958_DI_PT_RECTLIST;
   default:
      UNREACHABLE("unhandled primitive type");
   }
}

static inline uint32_t
radv_translate_fill(VkPolygonMode func)
{
   switch (func) {
   case VK_POLYGON_MODE_FILL:
      return V_028814_X_DRAW_TRIANGLES;
   case VK_POLYGON_MODE_LINE:
      return V_028814_X_DRAW_LINES;
   case VK_POLYGON_MODE_POINT:
      return V_028814_X_DRAW_POINTS;
   default:
      assert(0);
      return V_028814_X_DRAW_POINTS;
   }
}

static inline uint32_t
radv_translate_stencil_op(VkStencilOp op)
{
   switch (op) {
   case VK_STENCIL_OP_KEEP:
      return V_02842C_STENCIL_KEEP;
   case VK_STENCIL_OP_ZERO:
      return V_02842C_STENCIL_ZERO;
   case VK_STENCIL_OP_REPLACE:
      return V_02842C_STENCIL_REPLACE_TEST;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
      return V_02842C_STENCIL_ADD_CLAMP;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
      return V_02842C_STENCIL_SUB_CLAMP;
   case VK_STENCIL_OP_INVERT:
      return V_02842C_STENCIL_INVERT;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP:
      return V_02842C_STENCIL_ADD_WRAP;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP:
      return V_02842C_STENCIL_SUB_WRAP;
   default:
      return 0;
   }
}

static inline uint32_t
radv_translate_blend_logic_op(VkLogicOp op)
{
   switch (op) {
   case VK_LOGIC_OP_CLEAR:
      return V_028808_ROP3_CLEAR;
   case VK_LOGIC_OP_AND:
      return V_028808_ROP3_AND;
   case VK_LOGIC_OP_AND_REVERSE:
      return V_028808_ROP3_AND_REVERSE;
   case VK_LOGIC_OP_COPY:
      return V_028808_ROP3_COPY;
   case VK_LOGIC_OP_AND_INVERTED:
      return V_028808_ROP3_AND_INVERTED;
   case VK_LOGIC_OP_NO_OP:
      return V_028808_ROP3_NO_OP;
   case VK_LOGIC_OP_XOR:
      return V_028808_ROP3_XOR;
   case VK_LOGIC_OP_OR:
      return V_028808_ROP3_OR;
   case VK_LOGIC_OP_NOR:
      return V_028808_ROP3_NOR;
   case VK_LOGIC_OP_EQUIVALENT:
      return V_028808_ROP3_EQUIVALENT;
   case VK_LOGIC_OP_INVERT:
      return V_028808_ROP3_INVERT;
   case VK_LOGIC_OP_OR_REVERSE:
      return V_028808_ROP3_OR_REVERSE;
   case VK_LOGIC_OP_COPY_INVERTED:
      return V_028808_ROP3_COPY_INVERTED;
   case VK_LOGIC_OP_OR_INVERTED:
      return V_028808_ROP3_OR_INVERTED;
   case VK_LOGIC_OP_NAND:
      return V_028808_ROP3_NAND;
   case VK_LOGIC_OP_SET:
      return V_028808_ROP3_SET;
   default:
      UNREACHABLE("Unhandled logic op");
   }
}

static inline bool
radv_is_dual_src(VkBlendFactor factor)
{
   switch (factor) {
   case VK_BLEND_FACTOR_SRC1_COLOR:
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR:
   case VK_BLEND_FACTOR_SRC1_ALPHA:
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA:
      return true;
   default:
      return false;
   }
}

static ALWAYS_INLINE bool
radv_can_enable_dual_src(const struct vk_color_blend_attachment_state *att)
{
   VkBlendOp eqRGB = att->color_blend_op;
   VkBlendFactor srcRGB = att->src_color_blend_factor;
   VkBlendFactor dstRGB = att->dst_color_blend_factor;
   VkBlendOp eqA = att->alpha_blend_op;
   VkBlendFactor srcA = att->src_alpha_blend_factor;
   VkBlendFactor dstA = att->dst_alpha_blend_factor;
   bool eqRGB_minmax = eqRGB == VK_BLEND_OP_MIN || eqRGB == VK_BLEND_OP_MAX;
   bool eqA_minmax = eqA == VK_BLEND_OP_MIN || eqA == VK_BLEND_OP_MAX;

   if (!eqRGB_minmax && (radv_is_dual_src(srcRGB) || radv_is_dual_src(dstRGB)))
      return true;
   if (!eqA_minmax && (radv_is_dual_src(srcA) || radv_is_dual_src(dstA)))
      return true;
   return false;
}

static inline void
radv_normalize_blend_factor(VkBlendOp op, VkBlendFactor *src_factor, VkBlendFactor *dst_factor)
{
   if (op == VK_BLEND_OP_MIN || op == VK_BLEND_OP_MAX) {
      *src_factor = VK_BLEND_FACTOR_ONE;
      *dst_factor = VK_BLEND_FACTOR_ONE;
   }
}

void radv_get_viewport_xform(const VkViewport *viewport, float scale[3], float translate[3]);

void radv_translate_blend_equation(const struct radv_physical_device *pdev, VkBlendOp eqRGB, VkBlendFactor srcRGB,
                                   VkBlendFactor dstRGB, VkBlendOp eqA, VkBlendFactor srcA, VkBlendFactor dstA,
                                   uint32_t *cb_blend_control_out, uint32_t *sx_mrt_blend_opt_out);

#endif /* RADV_GRAPHICS_STATE_H */
