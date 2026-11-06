/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_SHADER_COMPUTE_H
#define RADV_SHADER_COMPUTE_H

#include <stdbool.h>
#include <stdint.h>

struct radv_compiler_info;
struct radv_device;
struct radv_physical_device;
struct radv_shader;
struct radv_shader_binary;
struct radv_shader_debug_info;
struct radv_shader_info;
struct radv_shader_stage;

struct radv_compute_shader_metadata {
   uint32_t wave32;
   uint32_t grid_base_sgpr;
   uint32_t push_const_sgpr;
   uint64_t inline_push_const_mask;
   uint32_t indirect_descriptors_sgpr;
   uint32_t heap_resource_sgpr;
   uint32_t heap_sampler_sgpr;
   uint32_t cs_state_sgpr;
};

void radv_get_compute_shader_metadata(const struct radv_device *device, const struct radv_shader *cs,
                                      struct radv_compute_shader_metadata *metadata);

struct radv_shader_binary *radv_compile_cs(const struct radv_compiler_info *compiler_info,
                                           struct radv_shader_stage *cs_stage, bool is_internal,
                                           struct radv_shader_debug_info *dbg);

void radv_precompute_registers_hw_cs(struct radv_device *device, struct radv_shader *shader);

#endif /* RADV_SHADER_COMPUTE_H */
