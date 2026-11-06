/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_shader_compute.h"
#include "nir/nir.h"
#include "radv_device.h"
#include "radv_physical_device.h"
#include "radv_shader.h"
#include "radv_shader_args.h"

#include "ac_nir.h"
#include "sid.h"

static uint32_t
radv_get_compute_resource_limits(const struct radv_physical_device *pdev, const struct radv_shader_info *info)
{
   unsigned threads_per_threadgroup;
   unsigned threadgroups_per_cu = 1;
   unsigned waves_per_threadgroup;
   unsigned max_waves_per_sh = 0;

   /* Calculate best compute resource limits. */
   threads_per_threadgroup = info->cs.block_size[0] * info->cs.block_size[1] * info->cs.block_size[2];
   waves_per_threadgroup = DIV_ROUND_UP(threads_per_threadgroup, info->wave_size);

   if (pdev->info.gfx_level >= GFX10 && waves_per_threadgroup == 1)
      threadgroups_per_cu = 2;

   return ac_get_compute_resource_limits(&pdev->info, waves_per_threadgroup, max_waves_per_sh, threadgroups_per_cu);
}

void
radv_get_compute_shader_metadata(const struct radv_device *device, const struct radv_shader *cs,
                                 struct radv_compute_shader_metadata *metadata)
{
   const struct radv_userdata_locations *locs = &cs->info.user_sgprs_locs;
   uint32_t upload_sgpr = 0, inline_sgpr = 0;

   memset(metadata, 0, sizeof(*metadata));

   metadata->wave32 = cs->info.wave_size == 32;

   metadata->grid_base_sgpr = radv_get_user_sgpr(cs, AC_UD_CS_GRID_SIZE);

   upload_sgpr = radv_get_user_sgpr(cs, AC_UD_PUSH_CONSTANTS);
   inline_sgpr = radv_get_user_sgpr(cs, AC_UD_INLINE_PUSH_CONSTANTS);

   metadata->push_const_sgpr = upload_sgpr | (inline_sgpr << 16);
   metadata->inline_push_const_mask = cs->info.inline_push_constant_mask;

   if (cs->info.descriptor_heap) {
      metadata->heap_resource_sgpr =
         ((cs->info.user_data_0 + locs->descriptor_heaps[RADV_HEAP_RESOURCE].sgpr_idx * 4) - SI_SH_REG_OFFSET) >> 2;
      metadata->heap_sampler_sgpr =
         ((cs->info.user_data_0 + locs->descriptor_heaps[RADV_HEAP_SAMPLER].sgpr_idx * 4) - SI_SH_REG_OFFSET) >> 2;
   } else {
      metadata->indirect_descriptors_sgpr = radv_get_user_sgpr(cs, AC_UD_INDIRECT_DESCRIPTORS);
   }

   metadata->cs_state_sgpr = radv_get_user_sgpr(cs, AC_UD_CS_STATE);
}

struct radv_shader_binary *
radv_compile_cs(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *cs_stage, bool is_internal,
                struct radv_shader_debug_info *dbg)
{

   /* Compile SPIR-V shader to NIR. */
   cs_stage->nir = radv_shader_spirv_to_nir(compiler_info, cs_stage, NULL, is_internal);

   NIR_PASS(_, cs_stage->nir, ac_nir_lower_indirect_derefs);
   NIR_PASS(_, cs_stage->nir, nir_lower_vars_to_ssa);

   radv_optimize_nir(cs_stage->nir, cs_stage->key.optimisations_disabled);

   /* Run the shader info pass. */
   radv_nir_shader_info_init(cs_stage->stage, MESA_SHADER_NONE, &cs_stage->info);
   radv_nir_shader_info_pass(compiler_info, cs_stage->nir, &cs_stage->layout, &cs_stage->key, NULL, false,
                             &cs_stage->info);

   radv_declare_shader_args(compiler_info, NULL, cs_stage, MESA_SHADER_NONE, dbg);

   cs_stage->info.user_sgprs_locs = cs_stage->args.user_sgprs_locs;
   cs_stage->info.inline_push_constant_mask = cs_stage->args.ac.inline_push_const_mask;

   /* Postprocess NIR. */
   radv_postprocess_nir(compiler_info, NULL, cs_stage);

   dbg->dump_shader = radv_can_dump_shader(compiler_info, cs_stage->nir);
   bool dump_nir = dbg->dump_shader && compiler_info->debug.dump_nir;

   if (dbg->dump_shader) {
      simple_mtx_lock(compiler_info->debug.shader_dump_mtx);

      if (dump_nir) {
         nir_print_shader(cs_stage->nir, stderr);
      }
   }

   /* Compile NIR shader to AMD assembly. */
   struct radv_shader_binary *cs_binary = radv_shader_nir_to_asm(compiler_info, cs_stage, &cs_stage->nir, 1, NULL);

   /* Dump NIR after nir_to_asm, because ACO modifies it. */
   char *nir_string = NULL;
   if (cs_stage->key.keep_executable_info)
      nir_string = radv_dump_nir_shaders(compiler_info, &cs_stage->nir, 1);

   dbg->nir_string = nir_string;
   dbg->stages = 1 << MESA_SHADER_COMPUTE;

   radv_shader_dump_asm(compiler_info, dbg, cs_binary, &cs_stage->info);

   if (dbg->dump_shader)
      simple_mtx_unlock(compiler_info->debug.shader_dump_mtx);

   if (cs_stage->key.keep_executable_info && cs_stage->spirv.size) {
      dbg->spirv = malloc(cs_stage->spirv.size);
      memcpy(dbg->spirv, cs_stage->spirv.data, cs_stage->spirv.size);
      dbg->spirv_size = cs_stage->spirv.size;
   }

   return cs_binary;
}

void
radv_precompute_registers_hw_cs(struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader_info *info = &shader->info;
   struct radv_shader_regs *regs = &shader->regs;

   regs->cs.compute_resource_limits = radv_get_compute_resource_limits(pdev, info);
   if (pdev->info.gfx_level >= GFX12) {
      regs->cs.compute_num_thread_x = S_00B81C_NUM_THREAD_FULL_GFX12(info->cs.block_size[0]);
      regs->cs.compute_num_thread_y = S_00B820_NUM_THREAD_FULL_GFX12(info->cs.block_size[1]);

      if (info->cs.derivative_group == DERIVATIVE_GROUP_QUADS) {
         regs->cs.compute_num_thread_x |= S_00B81C_INTERLEAVE_BITS_X(1);
         regs->cs.compute_num_thread_y |= S_00B820_INTERLEAVE_BITS_Y(1);
      }
   } else {
      regs->cs.compute_num_thread_x = S_00B81C_NUM_THREAD_FULL_GFX6(info->cs.block_size[0]);
      regs->cs.compute_num_thread_y = S_00B820_NUM_THREAD_FULL_GFX6(info->cs.block_size[1]);
   }
   regs->cs.compute_num_thread_z = S_00B824_NUM_THREAD_FULL(info->cs.block_size[2]);
}
