/*
 * Copyright © 2022 Intel Corporation
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

#include "anv_private.h"

#include "compiler/intel_nir.h"
#include "compiler/brw/brw_compiler.h"
#include "compiler/brw/brw_nir.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "dev/intel_debug.h"
#include "intel/compiler/intel_nir.h"
#include "util/macros.h"

#include "vk_nir.h"

#include "anv_internal_kernels.h"
#include "anv_nir.h"

#include "intel_gfx90_shaders_binding.h"
#include "intel_gfx110_shaders_binding.h"
#include "intel_gfx120_shaders_binding.h"
#include "intel_gfx125_shaders_binding.h"
#include "intel_gfx200_shaders_binding.h"
#include "intel_gfx300_shaders_binding.h"
#include "intel_gfx350_shaders_binding.h"

nir_def *
anv_resinfo(nir_builder *b,
            const struct intel_device_info *devinfo,
            nir_def *surface_state,
            nir_def *lod, nir_def *nr_comps,
            nir_def *dim, nir_def *is_array,
            bool is_indirect)
{
   if (is_indirect) {
      return anv_genX(devinfo, resinfo_indirect)(b, surface_state, lod,
                                                 nr_comps, dim, is_array);
   } else {
      return anv_genX(devinfo, resinfo)(b, surface_state, lod,
                                        nr_comps, dim, is_array);
   }
}

static nir_def *
_load_param(nir_builder *b, const struct intel_device_info *devinfo,
            unsigned bit_size, unsigned base, unsigned range)
{
   return
      (b->shader->info.stage == MESA_SHADER_COMPUTE && devinfo->verx10 >= 125) ?
      nir_load_global_constant_uniform_block_intel(
         b, 1, bit_size,
         nir_iadd_imm(b,
                      nir_load_inline_data_intel(b, 1, 64, nir_imm_int(b, 0),
                                                 .base = 0, .range = 8),
                      base),
         .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE,
         .align_mul = 64) :
      nir_load_push_data_intel(b, 1, bit_size, nir_imm_int(b, 0),
                               .base = base, .range = range);
}

#define load_param(b, devinfo, bit_size, struct_name, field_name)       \
   _load_param(b, devinfo, bit_size, offsetof(struct_name, field_name), bit_size / 8)

static nir_def *
load_fragment_index(nir_builder *b)
{
   nir_def *pos_in = nir_f2i32(b, nir_build_frag_coord(b, 2));
   return nir_iadd(b,
                   nir_imul_imm(b, nir_channel(b, pos_in, 1), 8192),
                   nir_channel(b, pos_in, 0));
}

static nir_def *
load_compute_index(nir_builder *b)
{
   return nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
}

static uint32_t
anv_call_internal_shader(nir_builder *b,
                         const struct intel_device_info *devinfo,
                         enum anv_internal_kernel_name shader_name)
{
   switch (shader_name) {
   case ANV_INTERNAL_KERNEL_GENERATED_DRAWS:
      anv_genX(devinfo, libanv_write_draw)(
         b,
         load_param(b, devinfo, 64, struct anv_gen_indirect_params, generated_cmds_addr),
         load_param(b, devinfo, 64, struct anv_gen_indirect_params, wa_insts_addr),
         load_param(b, devinfo, 64, struct anv_gen_indirect_params, indirect_data_addr),
         load_param(b, devinfo, 64, struct anv_gen_indirect_params, draw_id_addr),
         load_param(b, devinfo, 32, struct anv_gen_indirect_params, indirect_data_stride),
         load_param(b, devinfo, 64, struct anv_gen_indirect_params, draw_count_addr),
         load_param(b, devinfo, 32, struct anv_gen_indirect_params, draw_base),
         load_param(b, devinfo, 32, struct anv_gen_indirect_params, instance_multiplier),
         load_param(b, devinfo, 32, struct anv_gen_indirect_params, max_draw_count),
         load_param(b, devinfo, 32, struct anv_gen_indirect_params, flags),
         load_param(b, devinfo, 32, struct anv_gen_indirect_params, mocs),
         load_param(b, devinfo, 32, struct anv_gen_indirect_params, cmd_primitive_size),
         load_param(b, devinfo, 32, struct anv_gen_indirect_params, ring_count),
         load_param(b, devinfo, 64, struct anv_gen_indirect_params, gen_addr),
         load_param(b, devinfo, 64, struct anv_gen_indirect_params, end_addr),
         load_fragment_index(b));
      return sizeof(struct anv_gen_indirect_params);

   case ANV_INTERNAL_KERNEL_COPY_QUERY_RESULTS_COMPUTE:
   case ANV_INTERNAL_KERNEL_COPY_QUERY_RESULTS_FRAGMENT:
      anv_genX(devinfo, libanv_query_copy)(
         b,
         load_param(b, devinfo, 64, struct anv_query_copy_params, destination_addr),
         load_param(b, devinfo, 32, struct anv_query_copy_params, destination_stride),
         load_param(b, devinfo, 64, struct anv_query_copy_params, query_data_addr),
         load_param(b, devinfo, 32, struct anv_query_copy_params, query_base),
         load_param(b, devinfo, 32, struct anv_query_copy_params, num_queries),
         load_param(b, devinfo, 32, struct anv_query_copy_params, query_data_offset),
         load_param(b, devinfo, 32, struct anv_query_copy_params, query_stride),
         load_param(b, devinfo, 32, struct anv_query_copy_params, num_items),
         load_param(b, devinfo, 32, struct anv_query_copy_params, flags),
         shader_name == ANV_INTERNAL_KERNEL_COPY_QUERY_RESULTS_COMPUTE ?
         load_compute_index(b) : load_fragment_index(b));
      return sizeof(struct anv_query_copy_params);

   case ANV_INTERNAL_KERNEL_MEMCPY_COMPUTE:
      anv_genX(devinfo, libanv_memcpy)(
         b,
         load_param(b, devinfo, 64, struct anv_memcpy_params, dst_addr),
         load_param(b, devinfo, 64, struct anv_memcpy_params, src_addr),
         load_param(b, devinfo, 32, struct anv_memcpy_params, num_dwords),
         nir_imul_imm(b, load_compute_index(b), 4));
      return sizeof(struct anv_memcpy_params);

   case ANV_INTERNAL_KERNEL_DGC_GFX_COMPUTE:
   case ANV_INTERNAL_KERNEL_DGC_GFX_FRAGMENT:
      anv_genX(devinfo, libanv_preprocess_gfx_generate)(
         b,
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, cmd_addr),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, cmd_stride),
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, data_addr),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, data_stride),
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, seq_addr),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, seq_stride),
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, seq_count_addr),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, max_seq_count),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, cmd_prolog_size),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, data_prolog_size),
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, state_addr),
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, const_addr),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, const_size),
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, driver_const_addr),
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, return_addr),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, flags),
         shader_name == ANV_INTERNAL_KERNEL_DGC_GFX_COMPUTE ?
         load_compute_index(b) : load_fragment_index(b));
      return sizeof(struct anv_dgc_gfx_params);

   case ANV_INTERNAL_KERNEL_DGC_CS_COMPUTE:
   case ANV_INTERNAL_KERNEL_DGC_CS_FRAGMENT:
      anv_genX(devinfo, libanv_preprocess_cs_generate)(
         b,
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, cmd_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, cmd_stride),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, data_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, data_stride),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, seq_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, seq_stride),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, seq_count_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, max_seq_count),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, cmd_prolog_size),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, data_prolog_size),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, layout_addr),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, indirect_set_addr),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, interface_descriptor_data_addr),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, const_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, const_size),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, driver_const_addr),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, return_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, flags),
         shader_name == ANV_INTERNAL_KERNEL_DGC_CS_COMPUTE ?
         load_compute_index(b) : load_fragment_index(b));
      return sizeof(struct anv_dgc_cs_params);

   case ANV_INTERNAL_KERNEL_DGC_CS_POSTPROCESS_COMPUTE:
      anv_genX(devinfo, libanv_postprocess_cs_generate)(
         b,
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, cmd_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, cmd_stride),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, data_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, data_stride),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, seq_count_addr),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, max_seq_count),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, cmd_prolog_size),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, data_prolog_size),
         load_param(b, devinfo, 32, struct anv_dgc_cs_params, data_stride),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, indirect_set_addr),
         load_param(b, devinfo, 64, struct anv_dgc_cs_params, return_addr),
         load_compute_index(b));
      return sizeof(struct anv_dgc_cs_params);

   case ANV_INTERNAL_KERNEL_DGC_GFX_POSTPROCESS_FRAGMENT:
      anv_genX(devinfo, libanv_postprocess_gfx_generate)(
         b,
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, data_addr),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, data_stride),
         load_param(b, devinfo, 64, struct anv_dgc_gfx_params, seq_count_addr),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, max_seq_count),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, data_prolog_size),
         load_param(b, devinfo, 32, struct anv_dgc_gfx_params, flags),
         load_fragment_index(b));
      return sizeof(struct anv_dgc_gfx_params);

   case ANV_INTERNAL_KERNEL_DGC_DUMP_COMPUTE:
   case ANV_INTERNAL_KERNEL_DGC_DUMP_FRAGMENT:
      anv_genX(devinfo, libanv_dgc_dump)(
         b,
         load_param(b, devinfo, 64, struct anv_dgc_dump_params, cmd_addr),
         load_param(b, devinfo, 32, struct anv_dgc_dump_params, n_dwords),
         load_param(b, devinfo, 64, struct anv_dgc_dump_params, call_addr));
      return sizeof(struct anv_dgc_dump_params);

   case ANV_INTERNAL_KERNEL_DGC_RT_COMPUTE:
   case ANV_INTERNAL_KERNEL_DGC_RT_FRAGMENT:
      anv_genX(devinfo, libanv_preprocess_rt_generate)(
         b,
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, cmd_addr),
         load_param(b, devinfo, 32, struct anv_dgc_rt_params, cmd_stride),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, data_addr),
         load_param(b, devinfo, 32, struct anv_dgc_rt_params, data_stride),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, seq_addr),
         load_param(b, devinfo, 32, struct anv_dgc_rt_params, seq_stride),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, seq_count_addr),
         load_param(b, devinfo, 32, struct anv_dgc_rt_params, max_seq_count),
         load_param(b, devinfo, 32, struct anv_dgc_rt_params, cmd_prolog_size),
         load_param(b, devinfo, 32, struct anv_dgc_rt_params, data_prolog_size),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, layout_addr),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, compute_walker_addr),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, rtdg_global_addr),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, const_addr),
         load_param(b, devinfo, 32, struct anv_dgc_rt_params, const_size),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, driver_const_addr),
         load_param(b, devinfo, 64, struct anv_dgc_rt_params, return_addr),
         load_param(b, devinfo, 32, struct anv_dgc_rt_params, flags),
         shader_name == ANV_INTERNAL_KERNEL_DGC_RT_COMPUTE ?
         load_compute_index(b) : load_fragment_index(b));
      return sizeof(struct anv_dgc_rt_params);

   default:
      UNREACHABLE("Invalid shader name");
      return -1;
   }
}

static bool
lower_base_workgroup_id(nir_builder *b, nir_intrinsic_instr *intrin,
                        UNUSED void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_base_workgroup_id)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);
   nir_def_rewrite_uses(&intrin->def, nir_imm_zero(b, 3, 32));
   return true;
}

static void
check_sends(struct genisa_stats *stats, unsigned send_count)
{
   assert(send_count == 0 || stats->spills == 0);
   assert(send_count == 0 || stats->fills == 0);
   assert(send_count == 0 || stats->sends == send_count);
}

static struct anv_shader_internal *
compile_shader(struct anv_device *device,
               enum anv_internal_kernel_name shader_name,
               mesa_shader_stage stage,
               const char *name,
               const void *hash_key,
               uint32_t hash_key_size,
               uint32_t sends_count_expectation,
               const uint16_t local_size[3])
{
   const nir_shader_compiler_options *nir_options =
      &device->physical->compiler->nir_options[stage];

   nir_builder b = nir_builder_init_simple_shader(stage, nir_options,
                                                  "%s", name);

   uint32_t uniform_size = anv_call_internal_shader(&b, device->info, shader_name);

   nir_shader *nir = b.shader;

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_opt_gcm, true);

   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 1,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);

   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_per_member_structs);

   if (stage == MESA_SHADER_COMPUTE) {
      if (local_size[0] != 0) {
         memcpy(nir->info.workgroup_size, local_size, sizeof(nir->info.workgroup_size));
      } else {
         nir->info.workgroup_size[0] = 16;
         nir->info.workgroup_size[1] = 1;
         nir->info.workgroup_size[2] = 1;
      }
   }

   struct brw_compiler *compiler = device->physical->compiler;
   struct brw_nir_compiler_opts opts = {};
   brw_preprocess_nir(compiler, nir, &opts);

   NIR_PASS(_, nir, nir_propagate_invariant, false);

   if (stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_input_attachments,
                 &(nir_input_attachment_options) { });
   } else {
      nir_lower_compute_system_values_options options = {
         .has_base_workgroup_id = true,
         .lower_cs_local_id_to_index = true,
         .lower_workgroup_id_to_index = mesa_shader_stage_is_mesh(stage),
      };
      NIR_PASS(_, nir, nir_lower_compute_system_values, &options);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_base_workgroup_id,
                 nir_metadata_control_flow, NULL);
   }

   /* Reset sizes before gathering information */
   nir->global_mem_size = 0;
   nir->scratch_size = 0;
   nir->info.shared_size = 0;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dce);

   union brw_any_prog_key key;
   memset(&key, 0, sizeof(key));

   key.base.use_efficient_64bit = device->physical->uses_efficient_64bit;

   union brw_any_prog_data prog_data;
   memset(&prog_data, 0, sizeof(prog_data));

   if (stage == MESA_SHADER_COMPUTE) {
      /* Pick SIMD16, it shouldn't spill prior Xe2 and it's the native size
       * after.
       */
      nir->info.min_subgroup_size = nir->info.max_subgroup_size = 16;

      NIR_PASS(_, nir, brw_nir_lower_cs_intrinsics,
                 device->info, &prog_data.cs);
   }

   /* Do vectorizing here. For some reason when trying to do it in the back
    * this just isn't working.
    */
   struct brw_nir_vectorize_mem_cb_data vectorize_cb_data = {
      .devinfo = device->info,
   };
   nir_load_store_vectorize_options options = {
      .modes = nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_mem_global,
      .callback = brw_nir_should_vectorize_mem,
      .cb_data = &vectorize_cb_data,
      .robust_modes = (nir_variable_mode)0,
   };
   NIR_PASS(_, nir, nir_opt_load_store_vectorize, &options);

   prog_data.base.push_sizes[0] = uniform_size;

   void *temp_ctx = ralloc_context(NULL);

   const unsigned *program;
   if (stage == MESA_SHADER_FRAGMENT) {
      struct genisa_stats stats[3] = {};
      struct brw_compile_params params = {
         .mem_ctx = temp_ctx,
         .nir = nir,
         .key = &key.fs.base,
         .prog_data = (struct brw_stage_prog_data *)&prog_data.fs,
         .stats = stats,
         .log_data = device,
         .debug_flag = DEBUG_WM,
      };
      prog_data.base.push_sizes[0] = align(prog_data.base.push_sizes[0], REG_SIZE);
      program = brw_compile(compiler, &params);

      if (!ANV_DEBUG(SHADER_PRINT)) {
         unsigned stat_idx = 0;
         if (prog_data.fs.dispatch_8) {
            check_sends(&stats[stat_idx++], sends_count_expectation);
         }
         if (prog_data.fs.dispatch_16) {
            check_sends(&stats[stat_idx++], sends_count_expectation);
         }
         if (prog_data.fs.dispatch_32) {
            check_sends(&stats[stat_idx++], sends_count_expectation *
                                            (device->info->ver < 20 ? 2 : 1));
         }
      } else {
         for (uint32_t i = 0; i < ARRAY_SIZE(stats); i++) {
            assert(stats[i].spills == 0);
            assert(stats[i].fills == 0);
         }
      }
   } else {
      brw_cs_fill_push_const_info(device->info, &prog_data.cs, -1);
      prog_data.base.push_sizes[0] = align(prog_data.base.push_sizes[0], REG_SIZE);

      struct genisa_stats stats;
      struct brw_compile_params params = {
         .mem_ctx = temp_ctx,
         .nir = nir,
         .key = &key.cs.base,
         .prog_data = (struct brw_stage_prog_data *)&prog_data.cs,
         .stats = &stats,
         .log_data = device,
         .debug_flag = DEBUG_CS,
      };
      program = brw_compile(compiler, &params);

      if (!ANV_DEBUG(SHADER_PRINT)) {
         check_sends(&stats, sends_count_expectation);
      }
   }

   /* Complex shaders are allowed to spill */
   assert(sends_count_expectation == 0 || prog_data.base.total_scratch == 0);
   assert(program != NULL);
   struct anv_shader_internal *kernel = NULL;
   if (program == NULL)
      goto exit;

   struct anv_pipeline_bind_map empty_bind_map = {};
   struct anv_push_descriptor_info empty_push_desc_info = {};
   struct anv_shader_upload_params upload_params = {
      .stage               = nir->info.stage,
      .key_data            = hash_key,
      .key_size            = hash_key_size,
      .kernel_data         = program,
      .kernel_size         = prog_data.base.program_size,
      .prog_data           = &prog_data.base,
      .prog_data_size      = sizeof(prog_data),
      .bind_map            = &empty_bind_map,
      .push_desc_info      = &empty_push_desc_info,
   };

   kernel = anv_device_upload_kernel(device, device->internal_cache, &upload_params);

exit:
   ralloc_free(temp_ctx);
   ralloc_free(nir);

   return kernel;
}

struct internal_shader_key {
   char name[40];
   bool efficient_64bit;
};

VkResult
anv_device_get_internal_shader(struct anv_device *device,
                               enum anv_internal_kernel_name name,
                               struct anv_shader_internal **out_bin)
{
   const struct {
      struct internal_shader_key key;

      mesa_shader_stage stage;
      uint16_t local_size[3];
      uint32_t send_count;
   } internal_kernels[] = {
      [ANV_INTERNAL_KERNEL_GENERATED_DRAWS] = {
         .key        = {
            .name    = "anv-generated-indirect-draws",
         },
         .stage      = MESA_SHADER_FRAGMENT,
         .send_count =  (device->info->ver == 9 ?
                         /* 1 load +
                          * 4 stores +
                          * 2 * (2 loads + 2 stores) +
                          * 3 stores
                          */
                         16 :
                         /* 1 load +
                          * 2 * (2 loads + 3 stores) +
                          * 3 stores
                          */
                         14) +
         /* 3 loads + 3 stores */
         (intel_needs_workaround(device->info, 16011107343) ? 6 : 0) +
         /* 3 loads + 3 stores */
         (intel_needs_workaround(device->info, 22018402687) ? 6 : 0),
      },
      [ANV_INTERNAL_KERNEL_COPY_QUERY_RESULTS_COMPUTE] = {
         .key        = {
            .name    = "anv-copy-query-compute",
         },
         .stage      = MESA_SHADER_COMPUTE,
         .send_count = device->info->verx10 >= 125 ?
                       9 /* 4 loads + 4 stores + 1 EOT */ :
                       8 /* 3 loads + 4 stores + 1 EOT */,
      },
      [ANV_INTERNAL_KERNEL_COPY_QUERY_RESULTS_FRAGMENT] = {
         .key        = {
            .name    = "anv-copy-query-fragment",
         },
         .stage      = MESA_SHADER_FRAGMENT,
         .send_count = 8 /* 3 loads + 4 stores + 1 EOT */,
      },
      [ANV_INTERNAL_KERNEL_MEMCPY_COMPUTE] = {
         .key        = {
            .name    = "anv-memcpy-compute",
         },
         .stage      = MESA_SHADER_COMPUTE,
         .send_count = device->info->verx10 >= 125 ?
                       10 /* 5 loads (1 pull constants) + 4 stores + 1 EOT */ :
                       9 /* 4 loads + 4 stores + 1 EOT */,
      },
      [ANV_INTERNAL_KERNEL_DGC_GFX_COMPUTE] = {
         .key        = {
            .name    = "anv-dgc-gfx-compute",
         },
         .stage      = MESA_SHADER_COMPUTE,
         .send_count = 0 /* too complex */,
      },
      [ANV_INTERNAL_KERNEL_DGC_GFX_FRAGMENT] = {
         .key        = {
            .name    = "anv-dgc-gfx-fragment",
         },
         .stage      = MESA_SHADER_FRAGMENT,
         .send_count = 0 /* too complex */,
      },
      [ANV_INTERNAL_KERNEL_DGC_GFX_POSTPROCESS_FRAGMENT] = {
         .key        = {
            .name    = "anv-dgc-postprocess-fragment",
         },
         .stage      = MESA_SHADER_FRAGMENT,
         .send_count = 0,
      },
      [ANV_INTERNAL_KERNEL_DGC_CS_COMPUTE] = {
         .key        = {
            .name    = "anv-dgc-cs-compute",
         },
         .stage      = MESA_SHADER_COMPUTE,
         .send_count = 0 /* too complex */,
      },
      [ANV_INTERNAL_KERNEL_DGC_CS_FRAGMENT] = {
         .key        = {
            .name    = "anv-dgc-cs-fragment",
         },
         .stage      = MESA_SHADER_FRAGMENT,
         .send_count = 0 /* too complex */,
      },
      [ANV_INTERNAL_KERNEL_DGC_CS_POSTPROCESS_COMPUTE] = {
         .key        = {
            .name    = "anv-dgc-postprocess-compute",
         },
         .stage      = MESA_SHADER_COMPUTE,
         .send_count = device->info->verx10 >= 125 ? 11 : 8,
      },
      [ANV_INTERNAL_KERNEL_DGC_RT_COMPUTE] = {
         .key        = {
            .name    = "anv-dgc-rt-compute",
         },
         .stage      = MESA_SHADER_COMPUTE,
         .send_count = 0 /* too complex */,
      },
      [ANV_INTERNAL_KERNEL_DGC_RT_FRAGMENT] = {
         .key        = {
            .name    = "anv-dgc-rt-fragment",
         },
         .stage      = MESA_SHADER_FRAGMENT,
         .send_count = 0 /* too complex */,
      },
      [ANV_INTERNAL_KERNEL_DGC_DUMP_COMPUTE] = {
         .key        = {
            .name    = "anv-dgc-dump-compute",
         },
         .stage      = MESA_SHADER_COMPUTE,
         .local_size = { 1, 1, 1 },
         .send_count = 0 /* too complex */,
      },
      [ANV_INTERNAL_KERNEL_DGC_DUMP_FRAGMENT] = {
         .key        = {
            .name    = "anv-dgc-dump-fragment",
         },
         .stage      = MESA_SHADER_FRAGMENT,
         .send_count = 0 /* too complex */,
      },
   };

   struct anv_shader_internal *bin =
      p_atomic_read(&device->internal_kernels[name]);
   if (bin != NULL) {
      *out_bin = bin;
      return VK_SUCCESS;
   }

   struct internal_shader_key key = internal_kernels[name].key;
   key.efficient_64bit = device->physical->uses_efficient_64bit;

   bin =
      anv_device_search_for_kernel(device,
                                   device->internal_cache,
                                   &key, sizeof(key),
                                   NULL);
   if (bin != NULL)
      goto success;

   bin = compile_shader(device,
                        name,
                        internal_kernels[name].stage,
                        internal_kernels[name].key.name,
                        &key, sizeof(key),
                        internal_kernels[name].send_count,
                        internal_kernels[name].local_size);
   if (bin == NULL)
      return vk_errorf(device, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "Unable to compiler internal kernel");

 success:
   /* The cache already has a reference and it's not going anywhere so
    * there is no need to hold a second reference.
    */
   anv_shader_internal_unref(device, bin);

   p_atomic_set(&device->internal_kernels[name], bin);

   *out_bin = bin;
   return VK_SUCCESS;
}

VkResult
anv_device_init_internal_kernels(struct anv_device *device)
{
   const struct intel_l3_weights w =
      intel_get_default_l3_weights(device->info,
                                   true /* wants_dc_cache */,
                                   false /* needs_slm */);
   device->internal_kernels_l3_config = intel_get_l3_config(device->info, w);

   return VK_SUCCESS;
}

void
anv_device_finish_internal_kernels(struct anv_device *device)
{
}
