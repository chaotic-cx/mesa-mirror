/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_PIPELINE_GRAPHICS_H
#define RADV_PIPELINE_GRAPHICS_H

#include "amdgfxregs.h"

#include "radv_descriptor_set.h"
#include "radv_graphics_state.h"
#include "radv_pipeline.h"
#include "radv_pipeline_layout.h"
#include "radv_shader.h"
#include "radv_shader_graphics.h"

#include "vk_graphics_state.h"

#define VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO_RADV (VkStructureType)2000290001

#define VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO_RADV_cast VkGraphicsPipelineCreateInfoRADV

typedef struct VkGraphicsPipelineCreateInfoRADV {
   VkStructureType sType;
   const void *pNext;
   uint32_t custom_blend_mode;
} VkGraphicsPipelineCreateInfoRADV;

struct radv_sqtt_shaders_reloc {
   struct radeon_winsys_bo *bo;
   union radv_shader_arena_block *alloc;
   uint64_t va[MESA_VULKAN_SHADER_STAGES];
};

struct radv_graphics_pipeline {
   struct radv_pipeline base;

   uint64_t dynamic_states;
   uint64_t needed_dynamic_state;

   VkShaderStageFlags active_stages;

   struct radv_dynamic_state dynamic_state;

   struct radv_ia_multi_vgt_param_helpers ia_multi_vgt_param;

   /* Last pre-PS API stage */
   mesa_shader_stage last_vgt_api_stage;

   /* Custom blend mode for internal operations. */
   unsigned custom_blend_mode;

   /* Whether the pipeline uses out-of-order rasterization. */
   bool uses_out_of_order_rast;

   /* Whether the pipeline uses a VRS attachment. */
   bool uses_vrs_attachment;

   /* Whether depth/stencil states must be ignored because the pipeline has no depth/stencil attachments. */
   bool ignore_ds_state;

   /* For relocation of shaders with RGP. */
   struct radv_sqtt_shaders_reloc *sqtt_shaders_reloc;

   /* Whether the pipeline imported binaries. */
   bool has_pipeline_binaries;
};

RADV_DECL_PIPELINE_DOWNCAST(graphics, RADV_PIPELINE_GRAPHICS)

struct radv_retained_shaders {
   struct {
      void *serialized_nir;
      size_t serialized_nir_size;
      unsigned char shader_blake3[BLAKE3_KEY_LEN];
      struct radv_shader_stage_key key;
   } stages[MESA_VULKAN_SHADER_STAGES];
};

struct radv_graphics_lib_pipeline {
   struct radv_graphics_pipeline base;

   struct vk_graphics_pipeline_state graphics_state;

   /* For vk_graphics_pipeline_state */
   void *state_data;

   struct radv_pipeline_layout layout;

   VkGraphicsPipelineLibraryFlagsEXT lib_flags;

   struct radv_retained_shaders retained_shaders;

   void *mem_ctx;

   unsigned stage_count;
   VkPipelineShaderStageCreateInfo *stages;
   struct radv_shader_stage_key stage_keys[MESA_VULKAN_SHADER_STAGES];
};

RADV_DECL_PIPELINE_DOWNCAST(graphics_lib, RADV_PIPELINE_GRAPHICS_LIB)

static inline bool
radv_pipeline_has_stage(const struct radv_graphics_pipeline *pipeline, mesa_shader_stage stage)
{
   return pipeline->base.shaders[stage];
}

void radv_destroy_graphics_pipeline(struct radv_device *device, struct radv_graphics_pipeline *pipeline);

void radv_destroy_graphics_lib_pipeline(struct radv_device *device, struct radv_graphics_lib_pipeline *pipeline);

struct radv_graphics_pipeline_key {
   struct radv_graphics_state_key gfx_state;

   struct radv_shader_stage_key stage_info[MESA_VULKAN_SHADER_STAGES];
};

struct radv_graphics_pipeline_state {
   struct vk_graphics_pipeline_state vk;
   void *vk_data;

   bool compilation_required;

   struct radv_shader_stage *stages;

   struct radv_graphics_pipeline_key key;

   struct radv_pipeline_layout layout;

   blake3_hash libs_blake3;
};

void radv_graphics_pipeline_hash(const struct radv_device *device, const struct radv_graphics_pipeline_state *gfx_state,
                                 unsigned char *hash);

VkResult radv_generate_graphics_pipeline_state(struct radv_device *device,
                                               const VkGraphicsPipelineCreateInfo *pCreateInfo,
                                               struct radv_graphics_pipeline_state *gfx_state);

void radv_graphics_pipeline_state_finish(struct radv_device *device, struct radv_graphics_pipeline_state *gfx_state);

#endif /* RADV_PIPELINE_GRAPHICS_H */
