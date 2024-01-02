/*
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VK_UTIL_COMPILER_H
#define VK_UTIL_COMPILER_H

#include <vulkan/vulkan_core.h>

#include "compiler/shader_enums.h"
#include "util/bitscan.h"

static inline gl_shader_stage
vk_to_mesa_shader_stage(VkShaderStageFlagBits vk_stage)
{
   assert(util_bitcount((uint32_t) vk_stage) == 1);
   return (gl_shader_stage) (ffs((uint32_t) vk_stage) - 1);
}

static inline VkShaderStageFlagBits
mesa_to_vk_shader_stage(gl_shader_stage mesa_stage)
{
   return (VkShaderStageFlagBits) (1 << ((uint32_t) mesa_stage));
}

struct nir_spirv_specialization;

struct nir_spirv_specialization*
vk_spec_info_to_nir_spirv(const VkSpecializationInfo *spec_info,
                          uint32_t *out_num_spec_entries);
#endif
