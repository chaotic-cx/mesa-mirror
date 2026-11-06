/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_shader.h"
#include "meta/radv_meta.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "nir/nir_xfb_info.h"
#include "nir/radv_meta_nir.h"
#include "nir/radv_nir.h"
#include "spirv/nir_spirv.h"
#include "tools/radv_debug_hang.h"
#include "tools/radv_debug_nir.h"
#include "util/memstream.h"
#include "util/mesa-blake3.h"
#include "util/streaming-load-memcpy.h"
#include "util/u_atomic.h"
#include "ac_shader_util.h"
#include "radv_cs.h"
#include "radv_entrypoints.h"
#if RADV_LLVM_BACKEND
#include "radv_nir_to_llvm.h"
#endif
#include "radv_sdma.h"
#include "radv_shader_args.h"
#include "radv_shader_compute.h"
#include "radv_shader_graphics.h"

#include "ac_binary.h"
#include "ac_nir.h"
#if defined(USE_LIBELF)
#include "ac_rtld.h"
#endif
#include "aco_interface.h"
#include "sid.h"
#include "vk_debug_report.h"
#include "vk_nir.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_nir_lower_descriptor_heaps.h"
#include "vk_sampler.h"
#include "vk_semaphore.h"
#include "vk_sync.h"
#include "vk_ycbcr_conversion.h"

#include "nir/radv_nir_rt_stage_functions.h"
#include "aco_shader_info.h"
#include "radv_aco_shader_info.h"
#if AMD_LLVM_AVAILABLE
#include "ac_llvm_util.h"
#endif

static void
get_nir_options_for_stage(struct radv_compiler_info *compiler_info, mesa_shader_stage stage)
{
   nir_shader_compiler_options *options = &compiler_info->nir_options[stage];
   const bool split_fma = (stage <= MESA_SHADER_GEOMETRY || stage == MESA_SHADER_MESH) && compiler_info->key.split_fma;

   ac_nir_set_options(compiler_info->ac, compiler_info->key.use_llvm, options);

   if (split_fma) {
      if (options->float_mul_add16 & nir_float_muladd_support_has_ffma)
         options->float_mul_add16 |= nir_float_muladd_support_prefers_split;
      options->float_mul_add32 |= nir_float_muladd_support_prefers_split;
      options->float_mul_add64 |= nir_float_muladd_support_prefers_split;
   }
   options->max_unroll_iterations = 32;
   options->max_unroll_iterations_aggressive = 128;
   options->lower_doubles_options = nir_lower_drcp | nir_lower_dsqrt | nir_lower_drsq | nir_lower_ddiv;
   options->frag_coord_form = nir_frag_coord_xy_z_w_separate | nir_frag_coord_use_w_rcp;
   options->io_options |= nir_io_mediump_is_32bit | nir_io_radv_intrinsic_component_workaround;
   options->varying_expression_max_cost = ac_nir_varying_expression_max_cost;
}

void
radv_get_nir_options(struct radv_compiler_info *compiler_info)
{
   for (mesa_shader_stage stage = MESA_SHADER_VERTEX; stage < MESA_VULKAN_SHADER_STAGES; stage++)
      get_nir_options_for_stage(compiler_info, stage);
}

static uint8_t
vectorize_vec2_16bit(const nir_instr *instr, const void *_)
{
   if (instr->type != nir_instr_type_alu)
      return 0;

   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   switch (alu->op) {
   case nir_op_f2e4m3fn:
   case nir_op_f2e4m3fn_sat:
   case nir_op_f2e4m3fn_satfn:
   case nir_op_f2e5m2:
   case nir_op_f2e5m2_sat:
   case nir_op_e4m3fn2f:
   case nir_op_e5m22f:
      return 2;
   default:
      break;
   }

   const unsigned bit_size = alu->def.bit_size;
   if (bit_size == 16)
      return 2;
   else if (bit_size == 8)
      return 4;
   else
      return 1;
}

bool
radv_is_traversal_shader(nir_shader *nir)
{
   return nir && nir->info.stage == MESA_SHADER_INTERSECTION && nir->info.internal;
}

bool
radv_is_meta_shader(nir_shader *nir)
{
   /* The built-in traversal shader is marked as "internal", to distinguish
    * it from intersection shaders even though both share the INTERSECTION
    * shader stage. It is not a meta shader, though, so special-case it here.
    */
   if (radv_is_traversal_shader(nir))
      return false;
   return nir && nir->info.internal;
}

static uint64_t
radv_dump_flag_for_stage(const mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return RADV_DEBUG_DUMP_VS;
   case MESA_SHADER_TESS_CTRL:
      return RADV_DEBUG_DUMP_TCS;
   case MESA_SHADER_TESS_EVAL:
      return RADV_DEBUG_DUMP_TES;
   case MESA_SHADER_GEOMETRY:
      return RADV_DEBUG_DUMP_GS;
   case MESA_SHADER_FRAGMENT:
      return RADV_DEBUG_DUMP_PS;
   case MESA_SHADER_TASK:
      return RADV_DEBUG_DUMP_TASK;
   case MESA_SHADER_MESH:
      return RADV_DEBUG_DUMP_MESH;
   default:
      return RADV_DEBUG_DUMP_CS;
   }
}

bool
radv_can_dump_shader(const struct radv_compiler_info *compiler_info, nir_shader *nir)
{
   if (radv_is_meta_shader(nir) && nir->info.stage != MESA_SHADER_INTERSECTION)
      return compiler_info->debug.dump_meta_shaders;

   if (!nir)
      return false;

   return compiler_info->debug.dump_shaders & mesa_to_vk_shader_stage(nir->info.stage);
}

bool
radv_can_dump_shader_stats(const struct radv_compiler_info *compiler_info, nir_shader *nir)
{
   /* Only dump non-meta shader stats. */
   return compiler_info->debug.dump_shader_stats && !radv_is_meta_shader(nir);
}

void
radv_optimize_nir(struct nir_shader *shader, bool optimize_conservatively)
{
   bool progress;

   NIR_PASS(_, shader, nir_lower_alu_width, vectorize_vec2_16bit, NULL);
   NIR_PASS(_, shader, nir_opt_fp_math_ctrl);

   struct set *skip = _mesa_pointer_set_create(NULL);
   do {
      progress = false;

      NIR_LOOP_PASS(progress, skip, shader, nir_split_array_vars, nir_var_function_temp);
      NIR_LOOP_PASS(progress, skip, shader, nir_shrink_vec_array_vars, nir_var_function_temp | nir_var_mem_shared);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_copy_prop_vars);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_dead_write_vars);

      NIR_LOOP_PASS(_, skip, shader, nir_lower_phis_to_scalar, ac_nir_lower_phis_to_scalar_cb, NULL);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_copy_prop);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_remove_phis);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_dce);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_dead_cf);
      bool opt_loop_progress = false;
      NIR_LOOP_PASS_NOT_IDEMPOTENT(opt_loop_progress, skip, shader, nir_opt_loop);
      if (opt_loop_progress) {
         progress = true;
         NIR_LOOP_PASS(progress, skip, shader, nir_opt_copy_prop);
         NIR_LOOP_PASS(progress, skip, shader, nir_opt_remove_phis);
         NIR_LOOP_PASS(progress, skip, shader, nir_opt_dce);
      }
      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, shader, nir_opt_if, nir_opt_if_optimize_phi_true_false);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_cse);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_uub, &(nir_opt_uub_options){0});

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 8,
         .indirect_load_ok = true,
         .expensive_alu_ok = true,
         .discard_ok = true,
      };
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_peephole_select, &peephole_select_options);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_constant_folding);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_intrinsics);
      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, shader, nir_opt_algebraic);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_phi_to_bool);
      NIR_LOOP_PASS(progress, skip, shader, nir_opt_phi_precision);

      NIR_LOOP_PASS(progress, skip, shader, nir_opt_undef);

      if (shader->options->max_unroll_iterations) {
         bool unroll_progess = false;
         NIR_LOOP_PASS_NOT_IDEMPOTENT(unroll_progess, skip, shader, nir_opt_loop_unroll);
         /* Loop unrolling can add trivial phis for constants defined in the loop,
          * which can break all kinds of validation if they aren't removed immediately.
          */
         if (unroll_progess) {
            progress = true;
            NIR_LOOP_PASS(progress, skip, shader, nir_opt_remove_phis);
         }
      }
   } while (progress && !optimize_conservatively);
   _mesa_set_destroy(skip, NULL);

   NIR_PASS(progress, shader, nir_opt_shrink_vectors, true);
   NIR_PASS(progress, shader, nir_remove_dead_variables,
            nir_var_function_temp | nir_var_shader_in | nir_var_shader_out | nir_var_mem_shared, NULL);

   if (shader->info.stage == MESA_SHADER_FRAGMENT && shader->info.fs.uses_discard)
      NIR_PASS(progress, shader, nir_opt_move_discards_to_top);

   NIR_PASS(progress, shader, nir_opt_move, nir_move_load_ubo);

   /* radv_get_rt_shader_entrypoint returns the entrypoint for non-RT shaders too. */
   nir_function_impl *entrypoint = radv_get_rt_shader_entrypoint(shader);
   nir_shader_gather_info(shader, entrypoint);
}

void
radv_optimize_nir_algebraic_early(nir_shader *nir)
{
   bool progress = true;
   bool had_opt_fp_math_ctrl = false;
   while (progress) {
      progress = false;
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 8,
         .indirect_load_ok = true,
         .expensive_alu_ok = true,
         .discard_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole_select_options);
      NIR_PASS(_, nir, nir_opt_undef);
      NIR_PASS(_, nir, nir_opt_phi_to_bool);
      NIR_PASS(_, nir, nir_opt_generate_bfi);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_algebraic);

      if (!had_opt_fp_math_ctrl) {
         NIR_PASS(progress, nir, nir_opt_fp_math_ctrl);
         had_opt_fp_math_ctrl = true;
      }
   }
}

void
radv_optimize_nir_algebraic_late(nir_shader *nir)
{
   /* Invariant position doesn't cover generic VS outputs used
    * to compute position in later stages.
    */
   if (nir->info.stage != MESA_SHADER_VERTEX || nir->info.next_stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_opt_reassociate_for_fma);

   /* Do late algebraic optimization to turn add(a,
    * neg(b)) back into subs, then the mandatory cleanup
    * after algebraic.  Note that it may produce fnegs,
    * and if so then we need to keep running to squash
    * fneg(fneg(a)).
    */
   bool more_late_algebraic = true;
   struct set *skip = _mesa_pointer_set_create(NULL);
   while (more_late_algebraic) {
      more_late_algebraic = false;
      NIR_LOOP_PASS_NOT_IDEMPOTENT(more_late_algebraic, skip, nir, nir_opt_algebraic_late);
      NIR_LOOP_PASS(_, skip, nir, nir_opt_constant_folding);
      NIR_LOOP_PASS(_, skip, nir, nir_opt_copy_prop);
      NIR_LOOP_PASS(_, skip, nir, nir_opt_dce);
      NIR_LOOP_PASS(_, skip, nir, nir_opt_cse);
   }
   _mesa_set_destroy(skip, NULL);

   NIR_PASS(_, nir, nir_opt_uadd_carry, 32 | 64);
}

void
radv_optimize_nir_algebraic(nir_shader *nir, bool opt_offsets, bool opt_mqsad, enum amd_gfx_level gfx_level)
{
   radv_optimize_nir_algebraic_early(nir);

   if (opt_offsets) {
      const nir_opt_offsets_options offset_options = {
         .uniform_max = 0,
         .buffer_max = ~0,
         .shared_max = UINT16_MAX,
         .shared_atomic_max = UINT16_MAX,
         .allow_offset_wrap_cb = ac_nir_allow_offset_wrap_cb,
         .cb_data = &gfx_level,
      };
      NIR_PASS(_, nir, nir_opt_offsets, &offset_options);
   }
   if (opt_mqsad)
      NIR_PASS(_, nir, nir_opt_mqsad);

   radv_optimize_nir_algebraic_late(nir);
}

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

struct radv_shader_debug_data {
   struct vk_debug_report *debug_report;
   const struct vk_object_base *object;
};

static void
radv_spirv_nir_debug(void *private_data, enum nir_spirv_debug_level level, size_t spirv_offset, const char *message)
{
   struct radv_shader_debug_data *debug_data = private_data;

   static const VkDebugReportFlagsEXT vk_flags[] = {
      [NIR_SPIRV_DEBUG_LEVEL_INFO] = VK_DEBUG_REPORT_INFORMATION_BIT_EXT,
      [NIR_SPIRV_DEBUG_LEVEL_WARNING] = VK_DEBUG_REPORT_WARNING_BIT_EXT,
      [NIR_SPIRV_DEBUG_LEVEL_ERROR] = VK_DEBUG_REPORT_ERROR_BIT_EXT,
   };
   char buffer[256];

   snprintf(buffer, sizeof(buffer), "SPIR-V offset %lu: %s", (unsigned long)spirv_offset, message);

   vk_debug_report(debug_data->debug_report, vk_flags[level], debug_data->object, 0, 0, "radv", buffer);
}

static void
radv_shader_choose_subgroup_size(const struct radv_compiler_info *compiler_info, nir_shader *nir,
                                 const struct radv_shader_stage_key *stage_key, unsigned spirv_version)
{
   VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
      .requiredSubgroupSize = stage_key->subgroup_required_size * 32,
   };

   /* Do not allow for the SPIR-V 1.6 varying subgroup size rules. */
   if (compiler_info->key.no_implicit_varying_subgroup_size)
      spirv_version = 0x10000;

   vk_set_subgroup_size(nir, compiler_info->subgroup_size, compiler_info->min_subgroup_size,
                        compiler_info->max_subgroup_size, spirv_version,
                        rss_info.requiredSubgroupSize ? &rss_info : NULL, stage_key->subgroup_allow_varying,
                        stage_key->subgroup_require_full);

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   unsigned wave_size;

   if (nir->info.min_subgroup_size == nir->info.max_subgroup_size) {
      wave_size = nir->info.min_subgroup_size;
   } else if (mesa_shader_stage_uses_workgroup(nir->info.stage)) {
      const unsigned local_size =
         nir->info.workgroup_size[0] * nir->info.workgroup_size[1] * nir->info.workgroup_size[2];

      unsigned default_wave_size;
      if (nir->info.ray_queries)
         default_wave_size = compiler_info->key.rt_wave_size;
      else if (nir->info.stage == MESA_SHADER_MESH)
         default_wave_size = compiler_info->key.ge_wave_size;
      else
         default_wave_size = compiler_info->key.cs_wave_size;

      /* Games don't always request full subgroups when they should, which can cause bugs if cswave32
       * is enabled. Furthermore, if cooperative matrices or subgroup info are used, we can't transparently change
       * the subgroup size.
       */
      const bool require_full_subgroups =
         (default_wave_size == 32 &&
          (nir->info.uses_wide_subgroup_intrinsics ||
           (nir->info.stage == MESA_SHADER_COMPUTE && nir->info.cs.has_cooperative_matrix)) &&
          local_size % RADV_SUBGROUP_SIZE == 0);

      /* Use wave32 for small workgroups. */
      if (local_size <= 32)
         wave_size = 32;
      else if (require_full_subgroups)
         wave_size = 64;
      else
         wave_size = default_wave_size;
   } else if (nir->info.stage == MESA_SHADER_GEOMETRY &&
              (compiler_info->ac->gfx_level >= GFX10 && compiler_info->ac->gfx_level <= GFX10_3)) {
      /* Legacy GS doesn't support wave32. */
      wave_size = 64;
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      if (nir->info.ray_queries)
         wave_size = compiler_info->key.rt_wave_size;
      else
         wave_size = compiler_info->key.ps_wave_size;
   } else if (mesa_shader_stage_is_rt(nir->info.stage)) {
      wave_size = compiler_info->key.rt_wave_size;
   } else {
      wave_size = compiler_info->key.ge_wave_size;
   }

   if (nir->info.api_subgroup_size == 0) {
      /* Report real wave_size for allow_varying. */
      nir->info.api_subgroup_size = wave_size;
   }

   nir->info.max_subgroup_size = wave_size;

   /* We might still decide to use ngg later. */
   if (nir->info.stage == MESA_SHADER_GEOMETRY)
      nir->info.min_subgroup_size = compiler_info->key.ge_wave_size;
   else
      nir->info.min_subgroup_size = wave_size;
}

struct radv_lower_ycbcr_state {
   const struct radv_shader_layout *layout;
   const struct vk_sampler_state_array *embedded_samplers;
};

static const struct vk_ycbcr_conversion_state *
ycbcr_conversion_lookup(const void *data, uint32_t set, uint32_t binding, uint32_t array_index)
{
   const struct radv_lower_ycbcr_state *state = data;

   if (set == VK_NIR_YCBCR_SET_IMMUTABLE_SAMPLERS) {
      const struct vk_sampler_state_array *embedded_samplers = state->embedded_samplers;
      assert(binding < embedded_samplers->sampler_count);
      return &embedded_samplers->samplers[binding].ycbcr_conversion;
   } else {

      const struct radv_descriptor_set_layout *set_layout = state->layout->set[set].layout;
      const struct vk_ycbcr_conversion_state *ycbcr_samplers = radv_immutable_ycbcr_samplers(set_layout, binding);

      if (!ycbcr_samplers)
         return NULL;

      return ycbcr_samplers + array_index;
   }
}

nir_shader *
radv_shader_spirv_to_nir(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stage,
                         const struct radv_spirv_to_nir_options *options, bool is_internal)
{
   struct vk_sampler_state_array embedded_samplers;
   nir_shader *nir;
   bool progress;

   if (stage->internal_nir) {
      stage->internal_nir->options = &compiler_info->nir_options[stage->internal_nir->info.stage];

      /* Some things such as our meta clear/blit code will give us a NIR
       * shader directly.  In that case, we just ignore the SPIR-V entirely
       * and just use the NIR shader.  We don't want to alter meta and RT
       * shaders IR directly, so clone it first. */
      nir = nir_shader_clone(NULL, stage->internal_nir);
      nir_validate_shader(nir, "in internal shader");

      assert(exec_list_length(&nir->functions) == 1);
      radv_shader_choose_subgroup_size(compiler_info, nir, &stage->key, 0);
   } else {
      uint32_t *spirv = (uint32_t *)stage->spirv.data;
      assert(stage->spirv.size % 4 == 0);

      const bool dump_spirv = compiler_info->debug.dump_spirv &&
                              (is_internal ? compiler_info->debug.dump_meta_shaders
                                           : compiler_info->debug.dump_shaders & mesa_to_vk_shader_stage(stage->stage));
      if (dump_spirv)
         spirv_print_asm(stderr, (const uint32_t *)stage->spirv.data, stage->spirv.size / 4);

      struct nir_spirv_specialization *spec = vk_spec_info_to_nir_spirv(stage->spec_info);
      struct radv_shader_debug_data spirv_debug_data = {
         .debug_report = compiler_info->debug.debug_report,
         .object = stage->spirv.object,
      };
      const struct spirv_to_nir_options spirv_options = {
         .amd_gcn_shader = true,
         .amd_shader_ballot = true,
         .amd_shader_explicit_vertex_parameter = true,
         .amd_trinary_minmax = true,
         .capabilities = &compiler_info->spirv_caps,
         .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,
         .ssbo_addr_format = nir_address_format_vec2_index_32bit_offset,
         .phys_ssbo_addr_format = nir_address_format_64bit_global,
         .push_const_addr_format = nir_address_format_logical,
         .shared_addr_format = nir_address_format_32bit_offset,
         .constant_addr_format = nir_address_format_64bit_global,
         .debug =
            {
               .func = radv_spirv_nir_debug,
               .private_data = &spirv_debug_data,
            },
         .workarounds =
            {
               .force_tex_non_uniform = compiler_info->key.tex_non_uniform,
               .force_ssbo_non_uniform = compiler_info->key.ssbo_non_uniform,
               .lower_terminate_to_discard = compiler_info->key.lower_terminate_to_discard,
               .force_nan_preserve_min_max = compiler_info->key.force_nan_preserve_min_max,
               .force_exact_glsl_fma = compiler_info->key.force_exact_glsl_fma,
            },
         .emit_debug_break = compiler_info->debug.trap_enabled,
         .debug_info = compiler_info->key.nir_debug_info,
         .printf = compiler_info->debug.printf_enabled,
         .sampler_descriptor_size = compiler_info->sampler_descriptor_size,
         .sampler_descriptor_alignment = compiler_info->sampler_descriptor_alignment,
         .image_descriptor_size = compiler_info->image_descriptor_size,
         .image_descriptor_alignment = compiler_info->image_descriptor_alignment,
         .buffer_descriptor_size = compiler_info->buffer_descriptor_size,
         .buffer_descriptor_alignment = compiler_info->buffer_descriptor_alignment,
      };
      nir = spirv_to_nir(spirv, stage->spirv.size / 4, spec, stage->stage, stage->entrypoint, &spirv_options,
                         &compiler_info->nir_options[stage->stage]);
      nir->info.internal |= is_internal;
      assert(nir->info.stage == stage->stage);

      /* Shorten the shader name for internal shaders. */
      if (is_internal) {
         while (strstr(nir->info.name, "src/")) {
            nir->info.name = strstr(nir->info.name, "src/") + 4;
         }
         nir->info.name = ralloc_strdup(nir, nir->info.name);
      }

      nir_validate_shader(nir, "after spirv_to_nir");

      vtn_free_specialization(spec);

      const struct nir_lower_sysvals_to_varyings_options sysvals_to_varyings = {
         .point_coord = true,
         .primitive_id = nir->info.stage == MESA_SHADER_FRAGMENT,
      };
      NIR_PASS(_, nir, nir_lower_sysvals_to_varyings, &sysvals_to_varyings);

      /* We have to lower away local constant initializers right before we
       * inline functions.  That way they get properly initialized at the top
       * of the function and not at the top of its caller.
       */
      NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
      NIR_PASS(_, nir, nir_lower_returns);

      progress = false;
      NIR_PASS(progress, nir, nir_inline_functions);

      /* Inlining leaves the now-unused function implementations in the
       * shader.  Drop them before running whole-shader cleanup passes.
       * Cooperative matrix call functions are not inlined and must remain.
       */
      nir_remove_non_cmat_call_entrypoints(nir);

      if (progress) {
         NIR_PASS(_, nir, nir_opt_copy_prop_vars);
         NIR_PASS(_, nir, nir_opt_copy_prop);
      }
      NIR_PASS(_, nir, nir_opt_deref);

      if (nir->info.stage == MESA_SHADER_FRAGMENT)
         NIR_PASS(_, nir, vk_nir_lower_input_attachment_loads);

      /* Make sure we lower constant initializers on output variables so that
       * nir_remove_dead_variables below sees the corresponding stores
       */
      NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_shader_out);

      /* Now that we've deleted all but the main function, we can go ahead and
       * lower the rest of the constant initializers.
       */
      NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);

      radv_shader_choose_subgroup_size(compiler_info, nir, &stage->key, vk_spirv_version(spirv, stage->spirv.size));

      progress = false;
      struct nir_lower_coopmat_args coopmat_args = {
         .m_gran = 16,
         .n_gran = 16,
         .k_gran = 16,
      };
      NIR_PASS(progress, nir, nir_lower_cooperative_matrix_flexible_dimensions, &coopmat_args);
      if (progress) {
         NIR_PASS(_, nir, nir_opt_deref);
         NIR_PASS(_, nir, nir_opt_dce);
         NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp | nir_var_shader_temp, NULL);
      }

      NIR_PASS(progress, nir, radv_nir_lower_cooperative_matrix, compiler_info->ac->gfx_level,
               nir->info.max_subgroup_size);
      if (progress) {
         NIR_PASS(_, nir, nir_opt_dce);
         NIR_PASS(progress, nir, nir_inline_functions);
         nir_remove_non_entrypoints(nir); /* remove the late inlined functions */
         if (progress) {
            NIR_PASS(_, nir, nir_opt_copy_prop_vars);
            NIR_PASS(_, nir, nir_opt_copy_prop);
         }
         NIR_PASS(_, nir, nir_opt_deref);
         NIR_PASS(_, nir, nir_opt_dce);
      }

      NIR_PASS(_, nir, nir_lower_disordered_control_barriers);

      /* Split member structs.  We do this before lower_io_to_temporaries so that
       * it doesn't lower system values to temporaries by accident.
       */
      NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_split_per_member_structs);

      if (nir->info.stage == MESA_SHADER_FRAGMENT)
         NIR_PASS(_, nir, nir_lower_input_attachments,
                  &(nir_input_attachment_options){
                     .use_ia_coord_intrin = true,
                  });

      nir_remove_dead_variables_options dead_vars_opts = {
         .can_remove_var = nir_vk_is_not_xfb_output,
      };
      NIR_PASS(_, nir, nir_remove_dead_variables,
               nir_var_shader_in | nir_var_shader_out | nir_var_system_value | nir_var_mem_shared, &dead_vars_opts);

      /* Variables can make nir_propagate_invariant more conservative
       * than it needs to be.
       */
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);

      NIR_PASS(_, nir, nir_lower_vars_to_ssa);

      NIR_PASS(_, nir, nir_propagate_invariant, true);

      nir_gather_clip_cull_distance_sizes_from_vars(nir);
      NIR_PASS(_, nir, nir_merge_clip_cull_distance_vars);

      if (nir->info.stage == MESA_SHADER_VERTEX || nir->info.stage == MESA_SHADER_TESS_EVAL ||
          nir->info.stage == MESA_SHADER_GEOMETRY)
         nir_shader_gather_xfb_info(nir);

      NIR_PASS(_, nir, nir_lower_doubles, NULL, nir->options->lower_doubles_options);

      NIR_PASS(_, nir, nir_normalize_sin_cos);
   }

   if (nir->info.uses_abort) {
      nir_lower_abort_options abort_options = {
         .ptr_bit_size = 64,
      };

      NIR_PASS(_, nir, nir_lower_abort, &abort_options);
   }

   if (options && options->lower_view_index_to_device_index)
      NIR_PASS(_, nir, nir_lower_view_index_to_device_index);

   NIR_PASS(_, nir, nir_lower_system_values);

   if (compiler_info->ac->gfx_level < GFX12 && nir->info.derivative_group == DERIVATIVE_GROUP_QUADS) {
      nir_lower_compute_system_values_options csv_options = {
         .shuffle_local_ids_for_quad_derivatives = true,
         .lower_local_invocation_index = true,
      };

      NIR_PASS(_, nir, nir_opt_cse); /* CSE load_local_invocation_id */
      NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
   }

   bool lower_local_invocation_index = false;

   if (nir->info.stage == MESA_SHADER_COMPUTE || nir->info.stage == MESA_SHADER_TASK ||
       (nir->info.stage == MESA_SHADER_MESH && compiler_info->ac->gfx_level >= GFX11)) {
      lower_local_invocation_index = nir->info.derivative_group == DERIVATIVE_GROUP_QUADS ||
                                     (((nir->info.workgroup_size[0] == 1) + (nir->info.workgroup_size[1] == 1) +
                                       (nir->info.workgroup_size[2] == 1)) == 2);
   }

   nir_lower_compute_system_values_options csv_options = {
      /* Mesh shaders run as NGG which can implement local_invocation_index from
       * the wave ID in merged_wave_info, but they don't have local_invocation_ids on GFX10.3.
       */
      .lower_cs_local_id_to_index = nir->info.stage == MESA_SHADER_MESH && compiler_info->ac->gfx_level < GFX11,
      .lower_local_invocation_index = lower_local_invocation_index,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);

   /* Vulkan uses the separate-shader linking model */
   nir->info.separate_shader = true;

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   if (nir->info.ray_queries > 0) {
      /* Lower shared variables early to prevent the over allocation of shared memory in
       * radv_nir_lower_ray_queries.  */
      if (nir->info.stage == MESA_SHADER_COMPUTE) {
         NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, shared_var_info);
         NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
      }

      NIR_PASS(_, nir, nir_opt_ray_queries);
      NIR_PASS(_, nir, nir_opt_ray_query_ranges);
      NIR_PASS(_, nir, radv_nir_lower_ray_queries, compiler_info);
   }

   nir_lower_tex_options tex_options = {
      .lower_txp = ~0,
      .lower_txf_offset = true,
      .lower_tg4_offsets = true,
      .lower_txs_cube_array = true,
      .lower_to_fragment_fetch_amd = compiler_info->key.use_fmask,
      .lower_lod_zero_width = true,
      .lower_invalid_implicit_lod = true,
      .lower_1d = compiler_info->ac->gfx_level == GFX9,
      .optimize_txd = true,
      .lower_tg4_shadow_to_16bit = compiler_info->ac->has_fma_mix,
   };

   NIR_PASS(_, nir, nir_lower_tex, &tex_options);

   static const nir_lower_image_options image_options = {
      .lower_cube_size = true,
   };

   NIR_PASS(_, nir, nir_lower_image, &image_options);

   /* Depending on the variable mode mask, this lowers indirect IO, moves all input loads
    * to the beginning, and moves all output stores to the end. This is an aggressive
    * lowering and code motion pass.
    */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, nir_shader_get_entrypoint(nir),
               nir_var_shader_in | nir_var_shader_out);
   } else if (nir->info.stage == MESA_SHADER_TESS_EVAL || nir->info.stage == MESA_SHADER_GEOMETRY ||
              nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, nir_shader_get_entrypoint(nir), nir_var_shader_out);
   }

   NIR_PASS(_, nir, nir_split_var_copies);

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);

   nir_lower_subgroups_options lower_subgroup_options = {
      .subgroup_size = nir->info.api_subgroup_size,
      .ballot_bit_size = nir->info.api_subgroup_size,
      .ballot_components = 1,
      .lower_to_scalar = 1,
      .lower_subgroup_masks = 1,
      .lower_relative_shuffle = 1,
      .lower_rotate_to_shuffle = compiler_info->key.use_llvm,
      .lower_shuffle_to_32bit = 1,
      .lower_vote_feq = 1,
      .lower_vote_ieq = 1,
      .lower_vote_bool_eq = 1,
      .lower_quad_broadcast_dynamic = 1,
      .lower_quad_broadcast_dynamic_to_const = compiler_info->ac->gfx_level <= GFX7,
      .lower_shuffle_to_swizzle_amd = 1,
      .lower_ballot_bit_count_to_mbcnt_amd = 1,
      .lower_boolean_reduce = !compiler_info->key.use_llvm,
      .lower_boolean_shuffle = true,
   };
   NIR_PASS(_, nir, nir_lower_subgroups, &lower_subgroup_options);

   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_opt_shrink_stores, !compiler_info->key.disable_shrink_image_store);
   if (nir->info.stage == MESA_SHADER_FRAGMENT && nir->info.fs.uses_discard)
      NIR_PASS(_, nir, nir_lower_discard_if, nir_move_terminate_out_of_loops);

   const nir_opt_access_options opt_access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &opt_access_options);

   if (!stage->key.optimisations_disabled) {
      /* Only run this pass once before nir_lower_var_copies is called,
       * so that we don't introduce any new copy_deref instructions later.
       */
      NIR_PASS(_, nir, nir_opt_find_array_copies);
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);

      radv_optimize_nir(nir, false);

      NIR_PASS(_, nir, nir_opt_scalar_array_vars_to_vec, nir_var_function_temp | nir_var_mem_shared);

      bool memcpy_progress = false;
      NIR_PASS(memcpy_progress, nir, nir_opt_memcpy);
      if (memcpy_progress)
         NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_opt_deref);
   }

   /* We call nir_lower_var_copies() after the first radv_optimize_nir()
    * to remove any copies introduced by nir_opt_find_array_copies().
    */
   NIR_PASS(_, nir, nir_lower_var_copies);

   NIR_PASS(_, nir, nir_lower_memcpy);

   unsigned lower_flrp = (nir->options->lower_flrp16 ? 16 : 0) | (nir->options->lower_flrp32 ? 32 : 0) |
                         (nir->options->lower_flrp64 ? 64 : 0);
   if (lower_flrp != 0)
      NIR_PASS(progress, nir, nir_lower_flrp, lower_flrp, false /* always precise */);

   if (stage->key.descriptor_heap) {
      const vk_nir_lower_descriptor_heaps_options heap_options = {
         .lower_shader_record_index_to_non_uniform = true,
      };

      progress = false;
      NIR_PASS(progress, nir, vk_nir_lower_descriptor_heaps, stage->layout.mapping, &heap_options, &embedded_samplers);
      if (progress) {
         NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_uniform | nir_var_image, NULL);
         NIR_PASS(_, nir, nir_opt_dce);
      }
   }

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const, nir_address_format_32bit_offset);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo | nir_var_mem_ssbo,
            nir_address_format_vec2_index_32bit_offset);

   NIR_PASS(_, nir, radv_nir_lower_intrinsics_early, options && options->lower_view_index_to_zero);

   /* Lower deref operations for compute shared memory. */
   if (nir->info.stage == MESA_SHADER_COMPUTE || nir->info.stage == MESA_SHADER_TASK ||
       nir->info.stage == MESA_SHADER_MESH) {
      nir_variable_mode var_modes = nir_var_mem_shared;

      if (nir->info.stage == MESA_SHADER_TASK || nir->info.stage == MESA_SHADER_MESH)
         var_modes |= nir_var_mem_task_payload;

      nir_opt_shared_vars_to_subgroup_options shared_to_subgroup_options = {
         .optimize_constant_access_to_uniform = true,
         .optimize_divergent_access_to_shuffle = compiler_info->ac->gfx_level >= GFX8,
         .linear_workgroup_ids = nir->info.derivative_group != DERIVATIVE_GROUP_QUADS,
         .ballot_num_components = 1,
         .ballot_size = nir->info.max_subgroup_size,
      };

      bool shared_to_subgroup = false;
      NIR_PASS(shared_to_subgroup, nir, nir_opt_shared_vars_to_subgroup, &shared_to_subgroup_options);
      if (shared_to_subgroup) {
         NIR_PASS(_, nir, nir_opt_dead_write_vars);
         NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_mem_shared, NULL);
         NIR_PASS(_, nir, nir_lower_subgroups, &lower_subgroup_options);
      }

      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, var_modes, shared_var_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, var_modes, nir_address_format_32bit_offset);

      if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
         const unsigned chunk_size = 16; /* max single store size */
         const unsigned shared_size = align(nir->info.shared_size, chunk_size);
         NIR_PASS(_, nir, nir_zero_initialize_shared_memory, shared_size, chunk_size);
      }
   }

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global | nir_var_mem_constant, nir_address_format_64bit_global);

   /* Lower large variables that are always constant with load_constant
    * intrinsics, which get turned into PC-relative loads from a data
    * section next to the shader.
    */
   NIR_PASS(_, nir, nir_opt_large_constants, glsl_get_natural_size_align_bytes, 16);

   /* Lower primitive shading rate to match HW requirements. */
   if ((nir->info.stage == MESA_SHADER_VERTEX || nir->info.stage == MESA_SHADER_GEOMETRY ||
        nir->info.stage == MESA_SHADER_MESH) &&
       nir->info.outputs_written & VARYING_BIT_PRIMITIVE_SHADING_RATE) {
      /* Lower primitive shading rate to match HW requirements. */
      NIR_PASS(_, nir, radv_nir_lower_primitive_shading_rate, compiler_info->ac->gfx_level);
   }

   /* Lower immutable/embedded sampler derefs to vec4. */
   NIR_PASS(_, nir, radv_nir_lower_immediate_samplers, compiler_info, stage, &embedded_samplers);

   progress = false;

   struct radv_lower_ycbcr_state lower_ycbcr_state = {
      .layout = &stage->layout,
      .embedded_samplers = &embedded_samplers,
   };

   /* Remove deref_cast(undefined) texture derefs in dead control flow before nir_vk_lower_ycbcr_tex.
    * An earlier radv_optimize_nir() would have done this if optimisations_disabled=false.
    */
   if (stage->key.optimisations_disabled)
      NIR_PASS(_, nir, nir_opt_dead_cf);

   NIR_PASS(progress, nir, nir_vk_lower_ycbcr_tex, ycbcr_conversion_lookup, &lower_ycbcr_state);
   /* Gather info in the case that nir_vk_lower_ycbcr_tex might have emitted resinfo instructions. */
   if (progress)
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   if (compiler_info->ac->has_cs_regalloc_hang_bug && mesa_shader_stage_is_compute(nir->info.stage)) {
      const uint32_t wg_size = nir->info.workgroup_size[0] * nir->info.workgroup_size[1] * nir->info.workgroup_size[2];

      if (wg_size > 256) {
         NIR_PASS(progress, nir, nir_lower_workgroup_size, 256);

         if (!stage->key.optimisations_disabled)
            radv_optimize_nir(nir, false);

         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      }
   }

   if (stage->key.descriptor_heap)
      vk_sampler_state_array_finish(&embedded_samplers);

   return nir;
}

static unsigned
get_size_class(unsigned size, bool round_up)
{
   size = round_up ? util_logbase2_ceil(size) : util_logbase2(size);
   unsigned size_class = MAX2(size, RADV_SHADER_ALLOC_MIN_SIZE_CLASS) - RADV_SHADER_ALLOC_MIN_SIZE_CLASS;
   return MIN2(size_class, RADV_SHADER_ALLOC_NUM_FREE_LISTS - 1);
}

static void
remove_hole(struct radv_shader_free_list *free_list, union radv_shader_arena_block *hole)
{
   unsigned size_class = get_size_class(hole->size, false);
   list_del(&hole->freelist);
   if (list_is_empty(&free_list->free_lists[size_class]))
      free_list->size_mask &= ~(1u << size_class);
}

static void
add_hole(struct radv_shader_free_list *free_list, union radv_shader_arena_block *hole)
{
   unsigned size_class = get_size_class(hole->size, false);
   list_addtail(&hole->freelist, &free_list->free_lists[size_class]);
   free_list->size_mask |= 1u << size_class;
}

static union radv_shader_arena_block *
alloc_block_obj(struct radv_device *device)
{
   if (!list_is_empty(&device->shader_block_obj_pool)) {
      union radv_shader_arena_block *block =
         list_first_entry(&device->shader_block_obj_pool, union radv_shader_arena_block, pool);
      list_del(&block->pool);
      return block;
   }

   return malloc(sizeof(union radv_shader_arena_block));
}

static void
free_block_obj(struct radv_device *device, union radv_shader_arena_block *block)
{
   list_del(&block->pool);
   list_add(&block->pool, &device->shader_block_obj_pool);
}

VkResult
radv_shader_wait_for_upload(struct radv_device *device, uint64_t seq)
{
   if (!seq)
      return VK_SUCCESS;

   const VkSemaphoreWaitInfo wait_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .pSemaphores = &device->shader_upload_sem,
      .semaphoreCount = 1,
      .pValues = &seq,
   };
   return device->vk.dispatch_table.WaitSemaphores(radv_device_to_handle(device), &wait_info, UINT64_MAX);
}

static struct radv_shader_arena *
radv_create_shader_arena(struct radv_device *device, struct radv_shader_free_list *free_list, unsigned min_size,
                         unsigned arena_size, bool replayable, uint64_t replay_va)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   union radv_shader_arena_block *alloc = NULL;
   struct radv_shader_arena *arena = calloc(1, sizeof(struct radv_shader_arena));
   if (!arena)
      goto fail;

   if (!arena_size)
      arena_size = MAX2(
         RADV_SHADER_ALLOC_MIN_ARENA_SIZE << MIN2(RADV_SHADER_ALLOC_MAX_ARENA_SIZE_SHIFT, device->shader_arena_shift),
         min_size);
   arena->size = arena_size;

   enum radeon_bo_flag flags = RADEON_FLAG_NO_INTERPROCESS_SHARING | RADEON_FLAG_32BIT;
   if (device->shader_use_invisible_vram)
      flags |= RADEON_FLAG_NO_CPU_ACCESS;
   else
      flags |= (pdev->info.cpdma_prefetch_writes_memory ? 0 : RADEON_FLAG_READ_ONLY);

   if (replayable)
      flags |= RADEON_FLAG_REPLAYABLE;

   VkResult result;
   result = radv_bo_create(device, NULL, arena_size, RADV_SHADER_ALLOC_ALIGNMENT, RADEON_DOMAIN_VRAM, flags,
                           RADV_BO_PRIORITY_SHADER, replay_va, true, &arena->bo);
   if (result != VK_SUCCESS)
      goto fail;

   list_inithead(&arena->entries);
   alloc = alloc_block_obj(device);
   if (!alloc)
      goto fail;

   list_inithead(&alloc->freelist);
   alloc->arena = arena;
   alloc->offset = 0;
   alloc->size = arena_size;
   list_addtail(&alloc->list, &arena->entries);
   if (free_list)
      add_hole(free_list, alloc);

   if (!(flags & RADEON_FLAG_NO_CPU_ACCESS)) {
      arena->ptr = (char *)radv_buffer_map(device->ws, arena->bo);
      if (!arena->ptr)
         goto fail;
   }

   if (replay_va)
      arena->type = RADV_SHADER_ARENA_REPLAYED;
   else if (replayable)
      arena->type = RADV_SHADER_ARENA_REPLAYABLE;
   else
      arena->type = RADV_SHADER_ARENA_DEFAULT;

   return arena;

fail:
   if (alloc)
      free_block_obj(device, alloc);
   if (arena && arena->bo)
      radv_bo_destroy(device, NULL, arena->bo);
   free(arena);
   return NULL;
}

/* Inserts a block at an arbitrary place into a hole, splitting the hole as needed */
static union radv_shader_arena_block *
insert_block(struct radv_device *device, union radv_shader_arena_block *hole, uint32_t offset_in_hole, uint32_t size,
             struct radv_shader_free_list *free_list)
{
   uint32_t hole_begin = hole->offset;
   uint32_t hole_end = hole->offset + hole->size;

   /* The block might not lie exactly at the beginning or end
    * of the hole. Resize the hole to fit the block exactly,
    * and insert new holes before (left_hole) or after (right_hole) as needed.
    * left_hole or right_hole are skipped if the allocation lies exactly at the
    * beginning or end of the hole to avoid 0-sized holes. */
   union radv_shader_arena_block *left_hole = NULL;
   union radv_shader_arena_block *right_hole = NULL;

   if (offset_in_hole) {
      left_hole = alloc_block_obj(device);
      if (!left_hole)
         return NULL;
      list_inithead(&left_hole->freelist);
      left_hole->arena = hole->arena;
      left_hole->offset = hole->offset;
      left_hole->size = offset_in_hole;

      if (free_list)
         add_hole(free_list, left_hole);
   }

   if (hole->size > offset_in_hole + size) {
      right_hole = alloc_block_obj(device);
      if (!right_hole) {
         free(left_hole);
         return NULL;
      }
      list_inithead(&right_hole->freelist);
      right_hole->arena = hole->arena;
      right_hole->offset = hole_begin + offset_in_hole + size;
      right_hole->size = hole_end - right_hole->offset;

      if (free_list)
         add_hole(free_list, right_hole);
   }

   if (left_hole) {
      hole->offset += left_hole->size;
      hole->size -= left_hole->size;

      list_addtail(&left_hole->list, &hole->list);
   }
   if (right_hole) {
      hole->size -= right_hole->size;

      list_add(&right_hole->list, &hole->list);
   }

   if (free_list)
      remove_hole(free_list, hole);
   return hole;
}

/* Segregated fit allocator, implementing a good-fit allocation policy.
 *
 * This is an variation of sequential fit allocation with several lists of free blocks ("holes")
 * instead of one. Each list of holes only contains holes of a certain range of sizes, so holes that
 * are too small can easily be ignored while allocating. Because this also ignores holes that are
 * larger than necessary (approximating best-fit allocation), this could be described as a
 * "good-fit" allocator.
 *
 * Typically, shaders are allocated and only free'd when the device is destroyed. For this pattern,
 * this should allocate blocks for shaders fast and with no fragmentation, while still allowing
 * free'd memory to be re-used.
 */
union radv_shader_arena_block *
radv_alloc_shader_memory(struct radv_device *device, uint32_t size, bool replayable, void *ptr)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   size = ac_align_shader_binary_for_prefetch(pdev->info.gfx_level, pdev->info.instr_prefetch_distance, size);
   size = align(size, RADV_SHADER_ALLOC_ALIGNMENT);

   mtx_lock(&device->shader_arena_mutex);

   struct radv_shader_free_list *free_list = replayable ? &device->capture_replay_free_list : &device->shader_free_list;

   /* Try to use an existing hole. Unless the shader is very large, this should only have to look
    * at the first one available.
    */
   unsigned free_list_mask = BITFIELD_MASK(RADV_SHADER_ALLOC_NUM_FREE_LISTS);
   unsigned size_class = ffs(free_list->size_mask & (free_list_mask << get_size_class(size, true)));
   if (size_class) {
      size_class--;

      list_for_each_entry (union radv_shader_arena_block, hole, &free_list->free_lists[size_class], freelist) {
         if (hole->size < size)
            continue;

         assert(hole->offset % RADV_SHADER_ALLOC_ALIGNMENT == 0);

         if (size == hole->size) {
            remove_hole(free_list, hole);
            hole->freelist.next = ptr;
            mtx_unlock(&device->shader_arena_mutex);
            return hole;
         } else {
            union radv_shader_arena_block *alloc = alloc_block_obj(device);
            if (!alloc) {
               mtx_unlock(&device->shader_arena_mutex);
               return NULL;
            }
            list_addtail(&alloc->list, &hole->list);
            alloc->freelist.prev = NULL;
            alloc->freelist.next = ptr;
            alloc->arena = hole->arena;
            alloc->offset = hole->offset;
            alloc->size = size;

            remove_hole(free_list, hole);
            hole->offset += size;
            hole->size -= size;
            add_hole(free_list, hole);

            mtx_unlock(&device->shader_arena_mutex);
            return alloc;
         }
      }
   }

   struct radv_shader_arena *arena = radv_create_shader_arena(device, free_list, size, 0, replayable, 0);
   union radv_shader_arena_block *alloc = NULL;
   if (!arena)
      goto fail;

   alloc =
      insert_block(device, list_entry(arena->entries.next, union radv_shader_arena_block, list), 0, size, free_list);
   alloc->freelist.prev = NULL;
   alloc->freelist.next = ptr;

   ++device->shader_arena_shift;
   list_addtail(&arena->list, &device->shader_arenas);

   mtx_unlock(&device->shader_arena_mutex);
   return alloc;

fail:
   mtx_unlock(&device->shader_arena_mutex);
   free(alloc);
   if (arena) {
      free(arena->list.next);
      radv_bo_destroy(device, NULL, arena->bo);
   }
   free(arena);
   return NULL;
}

static union radv_shader_arena_block *
get_hole(struct radv_shader_arena *arena, struct list_head *head)
{
   if (head == &arena->entries)
      return NULL;

   union radv_shader_arena_block *hole = list_entry(head, union radv_shader_arena_block, list);
   return hole->freelist.prev ? hole : NULL;
}

void
radv_free_shader_memory(struct radv_device *device, union radv_shader_arena_block *alloc)
{
   if (!alloc)
      return;

   mtx_lock(&device->shader_arena_mutex);

   union radv_shader_arena_block *hole_prev = get_hole(alloc->arena, alloc->list.prev);
   union radv_shader_arena_block *hole_next = get_hole(alloc->arena, alloc->list.next);

   union radv_shader_arena_block *hole = alloc;

   struct radv_shader_free_list *free_list;

   switch (alloc->arena->type) {
   case RADV_SHADER_ARENA_DEFAULT:
      free_list = &device->shader_free_list;
      break;
   case RADV_SHADER_ARENA_REPLAYABLE:
      free_list = &device->capture_replay_free_list;
      break;
   case RADV_SHADER_ARENA_REPLAYED:
      free_list = NULL;
      break;
   default:
      UNREACHABLE("invalid shader arena type");
   }

   /* merge with previous hole */
   if (hole_prev) {
      if (free_list)
         remove_hole(free_list, hole_prev);

      hole_prev->size += hole->size;
      free_block_obj(device, hole);

      hole = hole_prev;
   }

   /* merge with next hole */
   if (hole_next) {
      if (free_list)
         remove_hole(free_list, hole_next);

      hole_next->offset -= hole->size;
      hole_next->size += hole->size;
      free_block_obj(device, hole);

      hole = hole_next;
   }

   if (list_is_singular(&hole->list)) {
      struct radv_shader_arena *arena = hole->arena;
      free_block_obj(device, hole);

      radv_bo_destroy(device, NULL, arena->bo);
      list_del(&arena->list);

      if (device->capture_replay_arena_vas) {
         struct hash_entry *arena_entry = NULL;
         hash_table_foreach (&device->capture_replay_arena_vas->table, entry) {
            if (entry->data == arena) {
               arena_entry = entry;
               break;
            }
         }
         _mesa_hash_table_remove(&device->capture_replay_arena_vas->table, arena_entry);
      }

      free(arena);
   } else if (free_list) {
      add_hole(free_list, hole);
   } else {
      /* Mark it as a hole, allowing merges when adjacent blocks are freed later. */
      list_inithead(&hole->freelist);
   }

   mtx_unlock(&device->shader_arena_mutex);
}

union radv_shader_arena_block *
radv_replay_shader_arena_block(struct radv_device *device, const struct radv_serialized_shader_arena_block *src,
                               void *ptr)
{
   mtx_lock(&device->shader_arena_mutex);

   union radv_shader_arena_block *ret_block = NULL;

   uint64_t va = src->arena_va;
   void *data = _mesa_hash_table_u64_search(device->capture_replay_arena_vas, va);

   if (!data) {
      struct radv_shader_arena *arena = radv_create_shader_arena(device, NULL, 0, src->arena_size, true, src->arena_va);
      if (!arena)
         goto out;

      _mesa_hash_table_u64_insert(device->capture_replay_arena_vas, src->arena_va, arena);
      list_addtail(&arena->list, &device->shader_arenas);
      data = arena;
   }

   uint32_t block_begin = src->offset;
   uint32_t block_end = src->offset + src->size;

   struct radv_shader_arena *arena = data;
   list_for_each_entry (union radv_shader_arena_block, hole, &arena->entries, list) {
      /* Only consider holes, not allocated shaders */
      if (!hole->freelist.prev)
         continue;

      uint32_t hole_begin = hole->offset;
      uint32_t hole_end = hole->offset + hole->size;

      if (hole_end < block_end)
         continue;

      /* If another allocated block overlaps the current replay block, allocation is impossible */
      if (hole_begin > block_begin)
         goto out;

      union radv_shader_arena_block *block = insert_block(device, hole, block_begin - hole_begin, src->size, NULL);
      if (!block)
         goto out;

      block->freelist.prev = NULL;
      block->freelist.next = ptr;

      ret_block = hole;
      break;
   }

out:
   mtx_unlock(&device->shader_arena_mutex);
   return ret_block;
}

void
radv_init_shader_arenas(struct radv_device *device)
{
   mtx_init(&device->shader_arena_mutex, mtx_plain);

   device->shader_free_list.size_mask = 0;
   device->capture_replay_free_list.size_mask = 0;

   list_inithead(&device->shader_arenas);
   list_inithead(&device->shader_block_obj_pool);
   for (unsigned i = 0; i < RADV_SHADER_ALLOC_NUM_FREE_LISTS; i++) {
      list_inithead(&device->shader_free_list.free_lists[i]);
      list_inithead(&device->capture_replay_free_list.free_lists[i]);
   }
}

void
radv_destroy_shader_arenas(struct radv_device *device)
{
   list_for_each_entry_safe (union radv_shader_arena_block, block, &device->shader_block_obj_pool, pool)
      free(block);

   list_for_each_entry_safe (struct radv_shader_arena, arena, &device->shader_arenas, list) {
      radv_bo_destroy(device, NULL, arena->bo);
      free(arena);
   }
   mtx_destroy(&device->shader_arena_mutex);
}

VkResult
radv_init_shader_upload_queue(struct radv_device *device)
{
   if (!device->shader_use_invisible_vram)
      return VK_SUCCESS;

   VkDevice vk_device = radv_device_to_handle(device);
   struct radeon_winsys *ws = device->ws;

   const struct vk_device_dispatch_table *disp = &device->vk.dispatch_table;
   VkResult result = VK_SUCCESS;

   result = ws->ctx_create(ws, RADEON_CTX_PRIORITY_MEDIUM, &device->shader_upload_hw_ctx);
   if (result != VK_SUCCESS)
      return result;
   mtx_init(&device->shader_upload_hw_ctx_mutex, mtx_plain);

   mtx_init(&device->shader_dma_submission_list_mutex, mtx_plain);
   cnd_init(&device->shader_dma_submission_list_cond);
   list_inithead(&device->shader_dma_submissions);

   for (unsigned i = 0; i < RADV_SHADER_UPLOAD_CS_COUNT; i++) {
      struct radv_shader_dma_submission *submission = calloc(1, sizeof(struct radv_shader_dma_submission));

      result = radv_create_cmd_stream(device, AMD_IP_SDMA, false, &submission->cs);
      if (result != VK_SUCCESS) {
         free(submission);
         return result;
      }

      list_addtail(&submission->list, &device->shader_dma_submissions);
   }

   const VkSemaphoreTypeCreateInfo sem_type = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
   };
   const VkSemaphoreCreateInfo sem_create = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &sem_type,
   };
   result = disp->CreateSemaphore(vk_device, &sem_create, NULL, &device->shader_upload_sem);
   if (result != VK_SUCCESS)
      return result;

   return VK_SUCCESS;
}

void
radv_destroy_shader_upload_queue(struct radv_device *device)
{
   if (!device->shader_use_invisible_vram)
      return;

   struct vk_device_dispatch_table *disp = &device->vk.dispatch_table;
   struct radeon_winsys *ws = device->ws;

   /* Upload queue should be idle assuming that shaders are not leaked */
   if (device->shader_upload_sem)
      disp->DestroySemaphore(radv_device_to_handle(device), device->shader_upload_sem, NULL);

   list_for_each_entry_safe (struct radv_shader_dma_submission, submission, &device->shader_dma_submissions, list) {
      if (submission->cs)
         radv_destroy_cmd_stream(device, submission->cs);
      if (submission->bo)
         radv_bo_destroy(device, NULL, submission->bo);
      list_del(&submission->list);
      free(submission);
   }

   cnd_destroy(&device->shader_dma_submission_list_cond);
   mtx_destroy(&device->shader_dma_submission_list_mutex);

   if (device->shader_upload_hw_ctx) {
      mtx_destroy(&device->shader_upload_hw_ctx_mutex);
      ws->ctx_destroy(device->shader_upload_hw_ctx);
   }
}

static bool
radv_should_use_wgp_mode(enum amd_gfx_level gfx_level, mesa_shader_stage stage, const struct radv_shader_info *info)
{
   if (gfx_level < GFX10)
      return false;

   /* Disable the WGP mode on gfx10.3 because it can hang. (it
    * happened on VanGogh) Let's disable it on all chips that
    * disable exactly 1 CU per SA for GS.
    */
   if (gfx_level > GFX10 && info->is_ngg)
      return false;

   if (stage == MESA_SHADER_FRAGMENT)
      return false;

   /* VS+TCS programs might have an unknown LDS size if the input patch size is dynamic. */
   bool uses_lds =
      radv_calculate_lds_size(info, gfx_level) > 0 || (stage == MESA_SHADER_TESS_CTRL && !info->num_tess_patches);

   /* LDS is faster with CU mode. */
   return !uses_lds;
}

#if defined(USE_LIBELF)
static bool
radv_open_rtld_binary(enum amd_gfx_level gfx_level, const struct radv_shader_binary *binary,
                      struct ac_rtld_binary *rtld_binary)
{
   const char *elf_data = (const char *)((struct radv_shader_binary_rtld *)binary)->data;
   size_t elf_size = ((struct radv_shader_binary_rtld *)binary)->elf_size;

   struct ac_rtld_open_info open_info = {
      .gfx_level = gfx_level,
      .shader_type = binary->info.stage,
      .wave_size = binary->info.wave_size,
      .num_parts = 1,
      .elf_ptrs = &elf_data,
      .elf_sizes = &elf_size,
   };

   return ac_rtld_open(rtld_binary, open_info);
}
#endif

unsigned
radv_get_inst_pref_size(const struct radv_compiler_info *compiler_info, unsigned exec_size)
{
   enum amd_gfx_level gfx_level = compiler_info->ac->gfx_level;
   unsigned prefetch_distance = compiler_info->hw.instr_prefetch_distance;

   return ac_get_instr_prefetch_size(gfx_level, prefetch_distance, exec_size);
}

static void
radv_precompute_registers_pgm(const struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const enum amd_gfx_level gfx_level = pdev->info.gfx_level;
   const struct radv_shader_info *info = &shader->info;
   enum ac_hw_stage hw_stage = radv_select_hw_stage(info, gfx_level);

   /* Special case for merged shaders compiled separately with ESO on GFX9+. */
   if (info->merged_shader_compiled_separately) {
      if (info->stage == MESA_SHADER_VERTEX && info->next_stage == MESA_SHADER_TESS_CTRL) {
         hw_stage = AC_HW_HULL_SHADER;
      } else if ((info->stage == MESA_SHADER_VERTEX || info->stage == MESA_SHADER_TESS_EVAL) &&
                 info->next_stage == MESA_SHADER_GEOMETRY) {
         hw_stage = info->is_ngg ? AC_HW_NEXT_GEN_GEOMETRY_SHADER : AC_HW_LEGACY_GEOMETRY_SHADER;
      }
   }

   struct radv_shader_regs *regs = &shader->regs;
   switch (hw_stage) {
   case AC_HW_NEXT_GEN_GEOMETRY_SHADER:
      assert(gfx_level >= GFX10);
      if (gfx_level >= GFX12) {
         regs->pgm_lo = R_00B224_SPI_SHADER_PGM_LO_ES;
      } else {
         regs->pgm_lo = R_00B320_SPI_SHADER_PGM_LO_ES;
      }

      if (gfx_level >= GFX12) {
         regs->pgm_rsrc4 = R_00B220_SPI_SHADER_PGM_RSRC4_GS;
      } else if (gfx_level >= GFX11) {
         regs->pgm_rsrc4 = R_00B204_SPI_SHADER_PGM_RSRC4_GS;
      }

      regs->pgm_rsrc1 = R_00B228_SPI_SHADER_PGM_RSRC1_GS;
      regs->pgm_rsrc2 = R_00B22C_SPI_SHADER_PGM_RSRC2_GS;
      break;
   case AC_HW_LEGACY_GEOMETRY_SHADER:
      assert(gfx_level < GFX11);
      if (gfx_level >= GFX10) {
         regs->pgm_lo = R_00B320_SPI_SHADER_PGM_LO_ES;
      } else if (gfx_level >= GFX9) {
         regs->pgm_lo = R_00B210_SPI_SHADER_PGM_LO_ES;
      } else {
         regs->pgm_lo = R_00B220_SPI_SHADER_PGM_LO_GS;
      }

      regs->pgm_rsrc1 = R_00B228_SPI_SHADER_PGM_RSRC1_GS;
      regs->pgm_rsrc2 = R_00B22C_SPI_SHADER_PGM_RSRC2_GS;
      break;
   case AC_HW_EXPORT_SHADER:
      assert(gfx_level < GFX9);
      regs->pgm_lo = R_00B320_SPI_SHADER_PGM_LO_ES;
      regs->pgm_rsrc1 = R_00B328_SPI_SHADER_PGM_RSRC1_ES;
      regs->pgm_rsrc2 = R_00B32C_SPI_SHADER_PGM_RSRC2_ES;
      break;
   case AC_HW_LOCAL_SHADER:
      assert(gfx_level < GFX9);
      regs->pgm_lo = R_00B520_SPI_SHADER_PGM_LO_LS;
      regs->pgm_rsrc1 = R_00B528_SPI_SHADER_PGM_RSRC1_LS;
      regs->pgm_rsrc2 = R_00B52C_SPI_SHADER_PGM_RSRC2_LS;
      break;
   case AC_HW_HULL_SHADER:
      if (gfx_level >= GFX12) {
         regs->pgm_lo = R_00B424_SPI_SHADER_PGM_LO_LS;
      } else if (gfx_level >= GFX10) {
         regs->pgm_lo = R_00B520_SPI_SHADER_PGM_LO_LS;
      } else if (gfx_level >= GFX9) {
         regs->pgm_lo = R_00B410_SPI_SHADER_PGM_LO_LS;
      } else {
         regs->pgm_lo = R_00B420_SPI_SHADER_PGM_LO_HS;
      }

      if (gfx_level >= GFX12) {
         regs->pgm_rsrc4 = R_00B420_SPI_SHADER_PGM_RSRC4_HS;
      } else if (gfx_level >= GFX11) {
         regs->pgm_rsrc4 = R_00B404_SPI_SHADER_PGM_RSRC4_HS;
      }

      regs->pgm_rsrc1 = R_00B428_SPI_SHADER_PGM_RSRC1_HS;
      regs->pgm_rsrc2 = R_00B42C_SPI_SHADER_PGM_RSRC2_HS;
      break;
   case AC_HW_VERTEX_SHADER:
      assert(gfx_level < GFX11);
      regs->pgm_lo = R_00B120_SPI_SHADER_PGM_LO_VS;
      regs->pgm_rsrc1 = R_00B128_SPI_SHADER_PGM_RSRC1_VS;
      regs->pgm_rsrc2 = R_00B12C_SPI_SHADER_PGM_RSRC2_VS;
      break;
   case AC_HW_PIXEL_SHADER:
      regs->pgm_lo = R_00B020_SPI_SHADER_PGM_LO_PS;
      regs->pgm_rsrc1 = R_00B028_SPI_SHADER_PGM_RSRC1_PS;
      regs->pgm_rsrc2 = R_00B02C_SPI_SHADER_PGM_RSRC2_PS;
      break;
   case AC_HW_COMPUTE_SHADER:
      regs->pgm_lo = R_00B830_COMPUTE_PGM_LO;
      regs->pgm_rsrc1 = R_00B848_COMPUTE_PGM_RSRC1;
      regs->pgm_rsrc2 = R_00B84C_COMPUTE_PGM_RSRC2;
      regs->pgm_rsrc3 = R_00B8A0_COMPUTE_PGM_RSRC3;
      break;
   default:
      UNREACHABLE("invalid hw stage");
      break;
   }
}

static void
radv_precompute_registers(struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader_info *info = &shader->info;

   radv_precompute_registers_pgm(device, shader);

   switch (info->stage) {
   case MESA_SHADER_VERTEX:
      if (!info->vs.as_es) {
         if (info->vs.as_ls && pdev->info.gfx_level >= GFX9) {
            radv_precompute_registers_hw_hs(device, shader);
         } else if (info->is_ngg) {
            radv_precompute_registers_hw_ngg(device, shader);
         } else {
            radv_precompute_registers_hw_vs(device, shader);
         }
      }
      break;
   case MESA_SHADER_TESS_CTRL: {
      radv_precompute_registers_hw_hs(device, shader);
      break;
   }
   case MESA_SHADER_TESS_EVAL:
      if (!info->tes.as_es) {
         if (info->is_ngg) {
            radv_precompute_registers_hw_ngg(device, shader);
         } else {
            radv_precompute_registers_hw_vs(device, shader);
         }
      }
      break;
   case MESA_SHADER_GEOMETRY:
      if (info->is_ngg) {
         radv_precompute_registers_hw_ngg(device, shader);
      } else {
         radv_precompute_registers_hw_gs(device, NULL, shader);
      }
      break;
   case MESA_SHADER_MESH:
      radv_precompute_registers_hw_ms(device, shader);
      break;
   case MESA_SHADER_FRAGMENT:
      radv_precompute_registers_hw_fs(device, shader);
      break;
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_TASK:
      radv_precompute_registers_hw_cs(device, shader);
      break;
   default:
      break;
   }

   if (info->stage == MESA_SHADER_VERTEX || info->stage == MESA_SHADER_TESS_EVAL ||
       info->stage == MESA_SHADER_GEOMETRY || info->stage == MESA_SHADER_MESH)
      radv_precompute_vgt_ps_inputs(pdev, shader);
}

static bool
radv_postprocess_binary_config(const struct radv_compiler_info *compiler_info, struct radv_shader_binary *binary,
                               const struct radv_shader_args *args)
{
   enum amd_gfx_level gfx_level = compiler_info->ac->gfx_level;
   struct ac_shader_config *config = &binary->config;
   unsigned exec_size;

   if (binary->type == RADV_BINARY_TYPE_RTLD) {
#if !defined(USE_LIBELF)
      return false;
#else
      struct ac_rtld_binary rtld_binary = {0};
      rtld_binary.options.exact_float_mode = !compiler_info->key.use_llvm;

      if (!radv_open_rtld_binary(compiler_info->ac->gfx_level, binary, &rtld_binary)) {
         return false;
      }

      exec_size = rtld_binary.exec_size;

      if (!ac_rtld_read_config(compiler_info->ac, &rtld_binary, config)) {
         ac_rtld_close(&rtld_binary);
         return false;
      }

      /* Calculate LDS allocation requirements. The ELF's LDS size can be too small for LLVM, but ACO might use LDS for
       * VGPR spilling. */
      unsigned lds_size = radv_calculate_lds_size(&binary->info, compiler_info->ac->gfx_level);
      config->lds_size = MAX2(config->lds_size, lds_size);

      ac_rtld_close(&rtld_binary);
#endif
   } else {
      struct radv_shader_binary_legacy *bin = (struct radv_shader_binary_legacy *)binary;

      exec_size = bin->exec_size;
   }

   const struct radv_shader_info *info = &binary->info;
   mesa_shader_stage stage = binary->info.stage;
   bool scratch_enabled = config->scratch_bytes_per_wave > 0;
   const bool trap_enabled = compiler_info->debug.trap_enabled;
   /* On GFX12, TRAP_PRESENT doesn't exist for compute shaders and it's enabled by default. */
   const enum ac_hw_stage hw_stage = radv_select_hw_stage(info, gfx_level);
   const bool trap_present = trap_enabled && (gfx_level < GFX12 || hw_stage != AC_HW_COMPUTE_SHADER);
   unsigned vgpr_comp_cnt = 0;
   unsigned num_input_vgprs = args->ac.num_vgprs_used;

   if (stage == MESA_SHADER_FRAGMENT) {
      num_input_vgprs = ac_get_fs_input_vgpr_cnt(config);
   }

   unsigned num_vgprs = MAX2(config->num_vgprs, num_input_vgprs);
   /* +2 for the ring offsets, +3 for scratch wave offset and VCC */
   unsigned num_sgprs = MAX2(config->num_sgprs, args->ac.num_sgprs_used + 2 + 3);
   unsigned num_shared_vgprs = config->num_shared_vgprs;
   /* shared VGPRs are introduced in Navi and are allocated in blocks of 8 (RDNA ref 3.6.5) */
   assert((gfx_level >= GFX10 && num_shared_vgprs % 8 == 0) || (gfx_level < GFX10 && num_shared_vgprs == 0));
   unsigned num_shared_vgpr_blocks = num_shared_vgprs / 8;
   unsigned excp_en = 0, excp_en_msb = 0;
   bool dx10_clamp = gfx_level < GFX11_7;

   config->num_vgprs = num_vgprs;
   config->num_sgprs = num_sgprs;
   config->num_shared_vgprs = num_shared_vgprs;

   config->rsrc2 = S_00B12C_USER_SGPR(args->num_user_sgprs) | S_00B12C_SCRATCH_EN(scratch_enabled) |
                   S_00B12C_TRAP_PRESENT(trap_present);

   if (trap_enabled) {
      /* Configure the shader exceptions like memory violation, etc. */
      if (compiler_info->debug.trap_excp_flags & RADV_TRAP_EXCP_MEM_VIOL) {
         excp_en |= 1 << 8;     /* for the graphics stages */
         excp_en_msb |= 1 << 1; /* for the compute stage */
      }

      if (compiler_info->debug.trap_excp_flags & RADV_TRAP_EXCP_FLOAT_DIV_BY_ZERO)
         excp_en |= 1 << 2;
      if (compiler_info->debug.trap_excp_flags & RADV_TRAP_EXCP_FLOAT_OVERFLOW)
         excp_en |= 1 << 3;
      if (compiler_info->debug.trap_excp_flags & RADV_TRAP_EXCP_FLOAT_UNDERFLOW)
         excp_en |= 1 << 4;

      if (compiler_info->debug.trap_excp_flags &
          (RADV_TRAP_EXCP_FLOAT_DIV_BY_ZERO | RADV_TRAP_EXCP_FLOAT_OVERFLOW | RADV_TRAP_EXCP_FLOAT_UNDERFLOW)) {
         /* It seems needed to disable DX10_CLAMP, otherwise the float exceptions aren't thrown. */
         dx10_clamp = false;
      }
   }

   if (gfx_level < GFX11) {
      config->rsrc2 |= S_00B12C_SO_BASE0_EN(!!info->so.strides[0]) | S_00B12C_SO_BASE1_EN(!!info->so.strides[1]) |
                       S_00B12C_SO_BASE2_EN(!!info->so.strides[2]) | S_00B12C_SO_BASE3_EN(!!info->so.strides[3]) |
                       S_00B12C_SO_EN(!!info->so.enabled_stream_buffers_mask);
   }

   config->rsrc1 = S_00B848_VGPRS((num_vgprs - 1) / (info->wave_size == 32 ? 8 : 4)) | S_00B848_DX10_CLAMP(dx10_clamp) |
                   S_00B848_FLOAT_MODE(config->float_mode);

   if (gfx_level >= GFX10) {
      config->rsrc2 |= S_00B22C_USER_SGPR_MSB_GFX10(args->num_user_sgprs >> 5);
   } else {
      config->rsrc1 |= S_00B228_SGPRS((num_sgprs - 1) / 8);
      config->rsrc2 |= S_00B22C_USER_SGPR_MSB_GFX9(args->num_user_sgprs >> 5);
   }

   mesa_shader_stage es_stage = MESA_SHADER_NONE;
   if (gfx_level >= GFX9) {
      es_stage = stage == MESA_SHADER_GEOMETRY ? info->gs.es_type : stage;
   }

   if (info->merged_shader_compiled_separately) {
      /* Update the stage for merged shaders compiled separately with ESO on GFX9+. */
      if (stage == MESA_SHADER_VERTEX && info->vs.as_ls) {
         stage = MESA_SHADER_TESS_CTRL;
      } else if (stage == MESA_SHADER_VERTEX && info->vs.as_es) {
         es_stage = MESA_SHADER_VERTEX;
         stage = MESA_SHADER_GEOMETRY;
      } else if (stage == MESA_SHADER_TESS_EVAL && info->tes.as_es) {
         es_stage = MESA_SHADER_TESS_EVAL;
         stage = MESA_SHADER_GEOMETRY;
      }
   }

   assert(config->lds_size <= compiler_info->ac->lds_size_per_workgroup);
   unsigned lds_alloc = ac_shader_encode_lds_size(config->lds_size, gfx_level, stage);

   switch (stage) {
   case MESA_SHADER_TESS_EVAL:
      if (info->is_ngg) {
         if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
            config->rsrc1 |= S_00B228_MEM_ORDERED(config->mem_ordered);
         config->rsrc2 |= S_00B22C_OC_LDS_EN(1) | S_00B22C_EXCP_EN(excp_en);
      } else if (info->tes.as_es) {
         assert(gfx_level <= GFX8);
         vgpr_comp_cnt = info->uses_prim_id ? 3 : 2;

         config->rsrc2 |= S_00B12C_OC_LDS_EN(1) | S_00B12C_EXCP_EN(excp_en);
      } else {
         bool enable_prim_id = info->outinfo.export_prim_id || info->uses_prim_id;
         vgpr_comp_cnt = enable_prim_id ? 3 : 2;

         if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
            config->rsrc1 |= S_00B128_MEM_ORDERED(config->mem_ordered);
         config->rsrc2 |= S_00B12C_OC_LDS_EN(1) | S_00B12C_EXCP_EN(excp_en);
      }
      config->rsrc2 |= S_00B22C_SHARED_VGPR_CNT(num_shared_vgpr_blocks);
      break;
   case MESA_SHADER_TESS_CTRL:
      if (gfx_level >= GFX9) {
         /* We need at least 2 components for LS.
          * VGPR0-3: (VertexID, RelAutoindex, InstanceID / StepRate0, InstanceID).
          * StepRate0 is set to 1. so that VGPR3 doesn't have to be loaded.
          */
         if (gfx_level >= GFX10) {
            if (info->vs.needs_instance_id) {
               vgpr_comp_cnt = gfx_level >= GFX12 ? 1 : 3;
            } else if (gfx_level <= GFX10_3) {
               vgpr_comp_cnt = 1;
            }
            config->rsrc2 |= S_00B42C_EXCP_EN_GFX6(excp_en);
         } else {
            vgpr_comp_cnt = info->vs.needs_instance_id ? 2 : 1;
            config->rsrc2 |= S_00B42C_EXCP_EN_GFX9(excp_en);
         }
      } else {
         config->rsrc2 |= S_00B12C_OC_LDS_EN(1) | S_00B12C_EXCP_EN(excp_en);
      }
      if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
         config->rsrc1 |= S_00B428_MEM_ORDERED(config->mem_ordered);
      config->rsrc1 |= S_00B428_WGP_MODE(config->wgp_mode);
      config->rsrc2 |= S_00B42C_SHARED_VGPR_CNT(num_shared_vgpr_blocks);
      break;
   case MESA_SHADER_VERTEX:
      if (info->is_ngg) {
         if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
            config->rsrc1 |= S_00B228_MEM_ORDERED(config->mem_ordered);
      } else if (info->vs.as_ls) {
         assert(gfx_level <= GFX8);
         /* We need at least 2 components for LS.
          * VGPR0-3: (VertexID, RelAutoindex, InstanceID / StepRate0, InstanceID).
          * StepRate0 is set to 1. so that VGPR3 doesn't have to be loaded.
          *
          * On GFX12, InstanceID is in VGPR1.
          */
         vgpr_comp_cnt = info->vs.needs_instance_id ? 2 : 1;
      } else if (info->vs.as_es) {
         assert(gfx_level <= GFX8);
         /* VGPR0-3: (VertexID, InstanceID / StepRate0, ...) */
         vgpr_comp_cnt = info->vs.needs_instance_id ? 1 : 0;
      } else {
         /* VGPR0-3: (VertexID, InstanceID / StepRate0, PrimID, InstanceID)
          * If PrimID is disabled. InstanceID / StepRate1 is loaded instead.
          * StepRate0 is set to 1. so that VGPR3 doesn't have to be loaded.
          */
         if (info->vs.needs_instance_id && gfx_level >= GFX10) {
            vgpr_comp_cnt = 3;
         } else if (info->outinfo.export_prim_id) {
            vgpr_comp_cnt = 2;
         } else if (info->vs.needs_instance_id) {
            vgpr_comp_cnt = 1;
         } else {
            vgpr_comp_cnt = 0;
         }

         if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
            config->rsrc1 |= S_00B128_MEM_ORDERED(config->mem_ordered);
      }
      config->rsrc2 |= S_00B12C_SHARED_VGPR_CNT(num_shared_vgpr_blocks) | S_00B12C_EXCP_EN(excp_en);
      break;
   case MESA_SHADER_MESH:
      if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
         config->rsrc1 |= S_00B228_MEM_ORDERED(config->mem_ordered);
      config->rsrc2 |= S_00B12C_SHARED_VGPR_CNT(num_shared_vgpr_blocks) | S_00B12C_EXCP_EN(excp_en);
      break;
   case MESA_SHADER_FRAGMENT:
      if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
         config->rsrc1 |= S_00B028_MEM_ORDERED(config->mem_ordered);
      config->rsrc1 |= S_00B028_LOAD_PROVOKING_VTX(info->ps.load_provoking_vtx);
      config->rsrc2 |= S_00B02C_SHARED_VGPR_CNT(num_shared_vgpr_blocks) | S_00B02C_EXCP_EN(excp_en) |
                       S_00B02C_LOAD_COLLISION_WAVEID(info->ps.pops && gfx_level < GFX11);
      break;
   case MESA_SHADER_GEOMETRY:
      if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
         config->rsrc1 |= S_00B228_MEM_ORDERED(config->mem_ordered);
      config->rsrc2 |= S_00B22C_SHARED_VGPR_CNT(num_shared_vgpr_blocks) | S_00B22C_EXCP_EN(excp_en);
      break;
   case MESA_SHADER_RAYGEN:
   case MESA_SHADER_CLOSEST_HIT:
   case MESA_SHADER_MISS:
   case MESA_SHADER_CALLABLE:
   case MESA_SHADER_INTERSECTION:
   case MESA_SHADER_ANY_HIT:
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_TASK:
      if (gfx_level >= GFX10 && gfx_level <= GFX11_7)
         config->rsrc1 |= S_00B848_MEM_ORDERED(config->mem_ordered);
      config->rsrc1 |= S_00B848_WGP_MODE(config->wgp_mode);
      config->rsrc2 |= S_00B84C_TGID_X_EN(info->cs.uses_block_id[0]) | S_00B84C_TGID_Y_EN(info->cs.uses_block_id[1]) |
                       S_00B84C_TGID_Z_EN(info->cs.uses_block_id[2]) |
                       S_00B84C_TIDIG_COMP_CNT(info->cs.uses_thread_id[2]   ? 2
                                               : info->cs.uses_thread_id[1] ? 1
                                                                            : 0) |
                       S_00B84C_TG_SIZE_EN(info->cs.uses_local_invocation_idx) | S_00B84C_LDS_SIZE(lds_alloc) |
                       S_00B84C_EXCP_EN(excp_en) | S_00B84C_EXCP_EN_MSB(excp_en_msb);
      config->rsrc3 |= S_00B8A0_SHARED_VGPR_CNT(num_shared_vgpr_blocks);

      if (gfx_level >= GFX12)
         config->rsrc3 |= S_00B8A0_INST_PREF_SIZE_GFX12(radv_get_inst_pref_size(compiler_info, exec_size));
      else if (gfx_level >= GFX11)
         config->rsrc3 |= S_00B8A0_INST_PREF_SIZE_GFX11(radv_get_inst_pref_size(compiler_info, exec_size));

      break;
   default:
      UNREACHABLE("unsupported shader type");
      break;
   }

   if (gfx_level >= GFX10 && info->is_ngg &&
       (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_EVAL || stage == MESA_SHADER_GEOMETRY ||
        stage == MESA_SHADER_MESH)) {
      unsigned gs_vgpr_comp_cnt, es_vgpr_comp_cnt;

      /* VGPR5-8: (VertexID, UserVGPR0, UserVGPR1, UserVGPR2 / InstanceID)
       *
       * On GFX12, InstanceID is in VGPR1.
       */
      if (es_stage == MESA_SHADER_VERTEX) {
         if (info->vs.needs_instance_id) {
            es_vgpr_comp_cnt = gfx_level >= GFX12 ? 1 : 3;
         } else {
            es_vgpr_comp_cnt = 0;
         }
      } else if (es_stage == MESA_SHADER_TESS_EVAL) {
         bool enable_prim_id = info->outinfo.export_prim_id || info->uses_prim_id;
         es_vgpr_comp_cnt = enable_prim_id ? 3 : 2;
      } else if (es_stage == MESA_SHADER_MESH) {
         es_vgpr_comp_cnt = 0;
      } else {
         UNREACHABLE("Unexpected ES shader stage");
      }

      if (stage == MESA_SHADER_MESH && compiler_info->ac->gfx_level >= GFX11) {
         /* Only VGPR0 is used for X/Y/Z local invocation ID */
         gs_vgpr_comp_cnt = 0;
      } else if (gfx_level >= GFX12) {
         if (info->gs.vertices_in >= 4) {
            gs_vgpr_comp_cnt = 2; /* VGPR2 contains offsets 3-5 */
         } else if (info->uses_prim_id ||
                    (es_stage == MESA_SHADER_VERTEX &&
                     (info->outinfo.export_prim_id || info->outinfo.export_prim_id_per_primitive))) {
            gs_vgpr_comp_cnt = 1; /* VGPR1 contains PrimitiveID. */
         } else {
            gs_vgpr_comp_cnt = 0; /* VGPR0 contains offsets 0-2, GS invocation ID. */
         }
      } else {
         /* GS vertex offsets in NGG:
          * - in passthrough mode, they are all packed into VGPR0
          * - in the default mode: VGPR0: offsets 0, 1; VGPR1: offsets 2, 3
          *
          * The vertex offset 2 is always needed when NGG isn't in passthrough mode
          * and uses triangle input primitives, including with NGG culling.
          */
         bool need_gs_vtx_offset2 = !info->is_ngg_passthrough || info->gs.vertices_in >= 3;

         /* TES only needs vertex offset 2 for triangles or quads. */
         if (stage == MESA_SHADER_TESS_EVAL)
            need_gs_vtx_offset2 &= info->tes._primitive_mode == TESS_PRIMITIVE_TRIANGLES ||
                                   info->tes._primitive_mode == TESS_PRIMITIVE_QUADS;

         if (info->uses_invocation_id) {
            gs_vgpr_comp_cnt = 3; /* VGPR3 contains InvocationID. */
         } else if (info->uses_prim_id ||
                    (es_stage == MESA_SHADER_VERTEX &&
                     (info->outinfo.export_prim_id || info->outinfo.export_prim_id_per_primitive))) {
            gs_vgpr_comp_cnt = 2; /* VGPR2 contains PrimitiveID. */
         } else if (need_gs_vtx_offset2) {
            gs_vgpr_comp_cnt = 1; /* VGPR1 contains offsets 2, 3 */
         } else {
            gs_vgpr_comp_cnt = 0; /* VGPR0 contains offsets 0, 1 (or passthrough prim) */
         }
      }

      config->rsrc1 |= S_00B228_GS_VGPR_COMP_CNT(gs_vgpr_comp_cnt) | S_00B228_WGP_MODE(config->wgp_mode);
      config->rsrc2 |= S_00B22C_ES_VGPR_COMP_CNT(es_vgpr_comp_cnt) | S_00B22C_LDS_SIZE(lds_alloc) |
                       S_00B22C_OC_LDS_EN(es_stage == MESA_SHADER_TESS_EVAL);
   } else if (gfx_level >= GFX9 && stage == MESA_SHADER_GEOMETRY) {
      unsigned gs_vgpr_comp_cnt, es_vgpr_comp_cnt;

      if (es_stage == MESA_SHADER_VERTEX) {
         /* VGPR0-3: (VertexID, InstanceID / StepRate0, ...) */
         if (info->vs.needs_instance_id) {
            es_vgpr_comp_cnt = gfx_level >= GFX10 ? 3 : 1;
         } else {
            es_vgpr_comp_cnt = 0;
         }
      } else if (es_stage == MESA_SHADER_TESS_EVAL) {
         es_vgpr_comp_cnt = info->uses_prim_id ? 3 : 2;
      } else {
         UNREACHABLE("invalid shader ES type");
      }

      /* If offsets 4, 5 are used, GS_VGPR_COMP_CNT is ignored and
       * VGPR[0:4] are always loaded.
       */
      if (info->uses_invocation_id) {
         gs_vgpr_comp_cnt = 3; /* VGPR3 contains InvocationID. */
      } else if (info->uses_prim_id) {
         gs_vgpr_comp_cnt = 2; /* VGPR2 contains PrimitiveID. */
      } else if (info->gs.vertices_in >= 3) {
         gs_vgpr_comp_cnt = 1; /* VGPR1 contains offsets 2, 3 */
      } else {
         gs_vgpr_comp_cnt = 0; /* VGPR0 contains offsets 0, 1 */
      }

      config->rsrc1 |= S_00B228_GS_VGPR_COMP_CNT(gs_vgpr_comp_cnt) | S_00B228_WGP_MODE(config->wgp_mode);
      config->rsrc2 |=
         S_00B22C_ES_VGPR_COMP_CNT(es_vgpr_comp_cnt) | S_00B22C_OC_LDS_EN(es_stage == MESA_SHADER_TESS_EVAL);
   } else if (gfx_level >= GFX9 && stage == MESA_SHADER_TESS_CTRL) {
      config->rsrc1 |= S_00B428_LS_VGPR_COMP_CNT(vgpr_comp_cnt);
   } else {
      config->rsrc1 |= S_00B128_VGPR_COMP_CNT(vgpr_comp_cnt);
   }

   return true;
}

/* This struct has pointers into radv_shader_binary_legacy::data for easy access to the different sections. */
struct radv_shader_binary_layout {
   void *stats;
   void *code;
   void *symbols;
   void *ir;
   void *disasm;
   void *debug_info;
};

static struct radv_shader_binary_layout
radv_shader_binary_get_layout(struct radv_shader_binary_legacy *binary)
{
   struct radv_shader_binary_layout layout = {0};

   uint32_t offset = 0;

   layout.stats = binary->data + offset;
   offset += binary->stats_size;

   layout.code = binary->data + offset;
   offset += binary->code_size;

   layout.symbols = binary->data + offset;
   offset += binary->symbols_size;

   layout.ir = binary->data + offset;
   offset += binary->ir_size;

   layout.disasm = binary->data + offset;
   offset += binary->disasm_size;

   layout.debug_info = binary->data + offset;
   offset += binary->debug_info_size;

   return layout;
}

bool
radv_shader_get_code(struct radv_device *device, const struct radv_shader *shader, void *dest_ptr)
{
   if (shader->binary->type == RADV_BINARY_TYPE_RTLD) {
#if !defined(USE_LIBELF)
      return false;
#else
      const struct radv_physical_device *pdev = radv_device_physical(device);
      struct ac_rtld_binary rtld_binary = {0};

      if (!radv_open_rtld_binary(pdev->info.gfx_level, shader->binary, &rtld_binary))
         return false;

      struct ac_rtld_upload_info info = {
         .binary = &rtld_binary,
         .rx_va = radv_shader_get_va(shader),
         .rx_ptr = dest_ptr,
      };

      if (!ac_rtld_upload(&info)) {
         ac_rtld_close(&rtld_binary);
         return false;
      }

      ac_rtld_close(&rtld_binary);
#endif
   } else {
      struct radv_shader_binary_legacy *bin = (struct radv_shader_binary_legacy *)shader->binary;
      struct radv_shader_binary_layout layout = radv_shader_binary_get_layout(bin);
      memcpy(dest_ptr, layout.code, bin->code_size);

      if (bin->symbols_size) {
         struct aco_symbol *symbols = layout.symbols;
         unsigned num_symbols = bin->symbols_size / sizeof(struct aco_symbol);

         for (unsigned i = 0; i < num_symbols; i++) {
            uint32_t *symbol_data = (uint32_t *)dest_ptr + symbols[i].offset;

            switch (symbols[i].id) {
            case aco_symbol_const_data_addr:
               /* RADV never inserts code between the shader end and constant data,
                * so we don't have to do anything here.
                */
               break;
            case aco_symbol_abort_buffer_addr_lo:
               *symbol_data = device->shader_abort.buffer_addr;
               break;
            case aco_symbol_abort_buffer_addr_hi:
               *symbol_data = device->shader_abort.buffer_addr >> 32;
               break;
            case aco_symbol_abort_buffer_size:
               *symbol_data = device->shader_abort.buffer_size;
               break;
            default:
               UNREACHABLE("unimplemented shader symbol");
            }
         }
      }
   }

   return true;
}

static VkResult
radv_shader_dma_resize_upload_buf(struct radv_device *device, struct radv_shader_dma_submission *submission,
                                  uint64_t size)
{
   if (submission->bo)
      radv_bo_destroy(device, NULL, submission->bo);

   VkResult result = radv_bo_create(
      device, NULL, size, RADV_SHADER_ALLOC_ALIGNMENT, RADEON_DOMAIN_GTT,
      RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING | RADEON_FLAG_32BIT | RADEON_FLAG_GTT_WC,
      RADV_BO_PRIORITY_UPLOAD_BUFFER, 0, true, &submission->bo);
   if (result != VK_SUCCESS)
      return result;

   submission->ptr = radv_buffer_map(device->ws, submission->bo);
   submission->bo_size = size;

   return VK_SUCCESS;
}

struct radv_shader_dma_submission *
radv_shader_dma_pop_submission(struct radv_device *device)
{
   struct radv_shader_dma_submission *submission;

   mtx_lock(&device->shader_dma_submission_list_mutex);

   while (list_is_empty(&device->shader_dma_submissions))
      cnd_wait(&device->shader_dma_submission_list_cond, &device->shader_dma_submission_list_mutex);

   submission = list_first_entry(&device->shader_dma_submissions, struct radv_shader_dma_submission, list);
   list_del(&submission->list);

   mtx_unlock(&device->shader_dma_submission_list_mutex);

   return submission;
}

void
radv_shader_dma_push_submission(struct radv_device *device, struct radv_shader_dma_submission *submission, uint64_t seq)
{
   submission->seq = seq;

   mtx_lock(&device->shader_dma_submission_list_mutex);

   list_addtail(&submission->list, &device->shader_dma_submissions);
   cnd_signal(&device->shader_dma_submission_list_cond);

   mtx_unlock(&device->shader_dma_submission_list_mutex);
}

struct radv_shader_dma_submission *
radv_shader_dma_get_submission(struct radv_device *device, struct radeon_winsys_bo *bo, uint64_t va, uint64_t size)
{
   struct radv_shader_dma_submission *submission = radv_shader_dma_pop_submission(device);
   struct radv_cmd_stream *cs = submission->cs;
   struct radeon_winsys *ws = device->ws;
   VkResult result;

   /* Wait for potentially in-flight submission to settle */
   result = radv_shader_wait_for_upload(device, submission->seq);
   if (result != VK_SUCCESS)
      goto fail;

   radv_reset_cmd_stream(device, cs);

   if (submission->bo_size < size) {
      result = radv_shader_dma_resize_upload_buf(device, submission, size);
      if (result != VK_SUCCESS)
         goto fail;
   }

   radv_sdma_copy_memory(device, cs, radv_buffer_get_va(submission->bo), va, size, false);
   radv_cs_add_buffer(ws, cs->b, submission->bo);
   radv_cs_add_buffer(ws, cs->b, bo);

   result = radv_finalize_cmd_stream(device, cs);
   if (result != VK_SUCCESS)
      goto fail;

   return submission;

fail:
   radv_shader_dma_push_submission(device, submission, 0);

   return NULL;
}

/*
 * If upload_seq_out is NULL, this function blocks until the DMA is complete. Otherwise, the
 * semaphore value to wait on device->shader_upload_sem is stored in *upload_seq_out.
 */
bool
radv_shader_dma_submit(struct radv_device *device, struct radv_shader_dma_submission *submission,
                       uint64_t *upload_seq_out)
{
   struct radv_cmd_stream *cs = submission->cs;
   struct radeon_winsys *ws = device->ws;
   VkResult result;

   mtx_lock(&device->shader_upload_hw_ctx_mutex);

   uint64_t upload_seq = device->shader_upload_seq + 1;

   struct vk_semaphore *semaphore = vk_semaphore_from_handle(device->shader_upload_sem);
   struct vk_sync *sync = vk_semaphore_get_active_sync(semaphore);
   const struct vk_sync_signal signal_info = {
      .sync = sync,
      .signal_value = upload_seq,
      .stage_mask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };

   struct radv_winsys_submit_info submit = {
      .ip_type = AMD_IP_SDMA,
      .queue_index = 0,
      .cs_array = &cs->b,
      .cs_count = 1,
   };

   result = ws->cs_submit(device->shader_upload_hw_ctx, &submit, 0, NULL, 1, &signal_info);
   if (result != VK_SUCCESS) {
      mtx_unlock(&device->shader_upload_hw_ctx_mutex);
      radv_shader_dma_push_submission(device, submission, 0);
      return false;
   }
   device->shader_upload_seq = upload_seq;
   mtx_unlock(&device->shader_upload_hw_ctx_mutex);

   radv_shader_dma_push_submission(device, submission, upload_seq);

   if (upload_seq_out) {
      *upload_seq_out = upload_seq;
   } else {
      result = radv_shader_wait_for_upload(device, upload_seq);
      if (result != VK_SUCCESS)
         return false;
   }

   return true;
}

bool
radv_shader_upload(struct radv_device *device, struct radv_shader *shader)
{
   if (device->shader_use_invisible_vram) {
      struct radv_shader_dma_submission *submission =
         radv_shader_dma_get_submission(device, shader->bo, shader->va, shader->code_size);
      if (!submission)
         return false;

      if (!radv_shader_get_code(device, shader, submission->ptr)) {
         radv_shader_dma_push_submission(device, submission, 0);
         return false;
      }

      if (!radv_shader_dma_submit(device, submission, &shader->upload_seq))
         return false;
   } else {
      void *dest_ptr = shader->alloc->arena->ptr + shader->alloc->offset;

      if (!radv_shader_get_code(device, shader, dest_ptr))
         return false;
   }
   return true;
}

unsigned
radv_get_max_waves(const struct radv_device *device, const struct ac_shader_config *conf,
                   const struct radv_shader_info *info)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radeon_info *gpu_info = &pdev->info;
   const enum amd_gfx_level gfx_level = gpu_info->gfx_level;
   const uint8_t wave_size = info->wave_size;
   mesa_shader_stage stage = info->stage;
   unsigned max_simd_waves = gpu_info->compiler_info.max_waves_per_simd;
   unsigned lds_increment = ac_shader_get_lds_alloc_granularity(gfx_level);
   unsigned lds_per_workgroup = align(conf->lds_size, lds_increment);
   unsigned waves_per_workgroup = DIV_ROUND_UP(info->workgroup_size, wave_size);

   if (stage == MESA_SHADER_FRAGMENT) {
      lds_per_workgroup += align(info->ps.num_inputs * 48, lds_increment);
   }

   if (conf->num_sgprs && gfx_level < GFX10) {
      unsigned sgprs = align(conf->num_sgprs, gfx_level >= GFX8 ? 16 : 8);
      max_simd_waves = MIN2(max_simd_waves, gpu_info->compiler_info.num_physical_sgprs_per_simd / sgprs);
   }

   if (conf->num_vgprs) {
      unsigned physical_vgprs = gpu_info->compiler_info.num_physical_wave64_vgprs_per_simd * (64 / wave_size);
      unsigned vgprs = align(conf->num_vgprs, wave_size == 32 ? 8 : 4);
      if (gfx_level >= GFX10_3) {
         unsigned real_vgpr_gran = gpu_info->compiler_info.num_physical_wave64_vgprs_per_simd / 64;
         vgprs = util_align_npot(vgprs, real_vgpr_gran * (wave_size == 32 ? 2 : 1));
      }
      max_simd_waves = MIN2(max_simd_waves, physical_vgprs / vgprs);
   }

   unsigned simd_per_cu_wgp = gpu_info->compiler_info.num_simd_per_compute_unit;
   if (conf->wgp_mode)
      simd_per_cu_wgp *= 2;

   if (lds_per_workgroup) {
      unsigned lds_per_cu_wgp =
         gpu_info->compiler_info.lds_size_per_workgroup * (gfx_level >= GFX10 && conf->wgp_mode ? 2 : 1);
      unsigned max_cu_wgp_waves = lds_per_cu_wgp / lds_per_workgroup * waves_per_workgroup;
      max_simd_waves = MIN2(max_simd_waves, DIV_ROUND_UP(max_cu_wgp_waves, simd_per_cu_wgp));
   }

   return gfx_level >= GFX10 ? max_simd_waves * (wave_size / 32) : max_simd_waves;
}

unsigned
radv_get_max_scratch_waves(const struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const unsigned num_cu = pdev->info.num_cu;

   return MIN2(device->scratch_waves, 4 * num_cu * shader->max_waves);
}

struct radv_shader_binary_debug_info
radv_parse_binary_debug_info(const struct radv_compiler_info *compiler_info, const struct radv_shader_binary *binary)
{
   struct radv_shader_binary_debug_info dbg = {0};
   dbg.type = binary->type;
   if (binary->type == RADV_BINARY_TYPE_RTLD) {
#if !defined(USE_LIBELF)
      return dbg;
#else
      struct radv_shader_binary_rtld *bin = (struct radv_shader_binary_rtld *)binary;
      struct ac_rtld_binary rtld_binary = {0};

      if (!radv_open_rtld_binary(compiler_info->ac->gfx_level, binary, &rtld_binary))
         return dbg;

      const char *disasm_data;
      size_t disasm_size;
      if (ac_rtld_get_section_by_name(&rtld_binary, ".AMDGPU.disasm", &disasm_data, &disasm_size)) {
         dbg.disasm_string = malloc(disasm_size + 1);
         memcpy(dbg.disasm_string, disasm_data, disasm_size);
         dbg.disasm_string[disasm_size] = 0;
      }

      const char *stats_data;
      size_t stats_size;
      if (ac_rtld_get_section_by_name(&rtld_binary, ".ACO.stats", &stats_data, &stats_size)) {
         dbg.statistics = malloc(stats_size);
         memcpy(dbg.statistics, stats_data, stats_size);
      }

      dbg.ir_string = bin->llvm_ir_size ? (char *)(bin->data + bin->elf_size) : NULL;

      ac_rtld_close(&rtld_binary);
#endif
   } else {
      struct radv_shader_binary_legacy *bin = (struct radv_shader_binary_legacy *)binary;
      struct radv_shader_binary_layout layout = radv_shader_binary_get_layout(bin);

      if (bin->stats_size) {
         dbg.statistics = calloc(bin->stats_size, 1);
         memcpy(dbg.statistics, layout.stats, bin->stats_size);
      }

      dbg.ir_string = bin->ir_size ? layout.ir : NULL;
      dbg.disasm_string = bin->disasm_size ? layout.disasm : NULL;

      if (bin->debug_info_size) {
         dbg.debug_info = malloc(bin->debug_info_size);
         memcpy(dbg.debug_info, layout.debug_info, bin->debug_info_size);
         dbg.debug_info_count = bin->debug_info_size / sizeof(struct ac_shader_debug_info);
      }
   }

   return dbg;
}

VkResult
radv_shader_create_uncached(struct radv_device *device, const struct radv_shader_binary *binary, bool replayable,
                            struct radv_serialized_shader_arena_block *replay_block, struct radv_shader_debug_info *dbg,
                            struct radv_shader **out_shader)
{
   VkResult result = VK_SUCCESS;
   struct radv_shader *shader = calloc(1, sizeof(struct radv_shader));
   if (!shader) {
      *out_shader = NULL;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   simple_mtx_init(&shader->replay_mtx, mtx_plain);

   _mesa_blake3_compute(binary, binary->total_size, shader->hash);

   vk_pipeline_cache_object_init(&device->vk, &shader->base, &radv_shader_ops, shader->hash, sizeof(shader->hash));

   shader->info = binary->info;
   shader->config = binary->config;
   shader->max_waves = radv_get_max_waves(device, &shader->config, &shader->info);
   shader->binary = malloc(binary->total_size);
   if (!shader->binary) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto out;
   }
   memcpy(shader->binary, binary, binary->total_size);

   if (dbg)
      shader->dbg = *dbg;

   if (binary->type == RADV_BINARY_TYPE_RTLD) {
#if !defined(USE_LIBELF)
      goto out;
#else
      const struct radv_physical_device *pdev = radv_device_physical(device);
      struct ac_rtld_binary rtld_binary = {0};
      if (!radv_open_rtld_binary(pdev->info.gfx_level, binary, &rtld_binary)) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto out;
      }
      shader->code_size = rtld_binary.rx_size;
      shader->exec_size = rtld_binary.exec_size;
      ac_rtld_close(&rtld_binary);
#endif
   } else {
      struct radv_shader_binary_legacy *bin = (struct radv_shader_binary_legacy *)binary;
      shader->code_size = bin->code_size;
      shader->exec_size = bin->exec_size;
   }

   if (replay_block) {
      shader->alloc = radv_replay_shader_arena_block(device, replay_block, shader);
      if (!shader->alloc) {
         result = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
         goto out;
      }

      shader->has_replay_alloc = true;
   } else {
      shader->alloc = radv_alloc_shader_memory(device, shader->code_size, replayable, shader);
      if (!shader->alloc) {
         result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
         goto out;
      }
   }

   shader->bo = shader->alloc->arena->bo;
   shader->va = radv_buffer_get_va(shader->bo) + shader->alloc->offset;

   if (!radv_shader_upload(device, shader)) {
      result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
      goto out;
   }

   /* Precompute register values for faster emission. */
   radv_precompute_registers(device, shader);

   *out_shader = shader;

out:
   if (result != VK_SUCCESS) {
      radv_shader_unref(device, shader);
      *out_shader = NULL;
   }

   return result;
}

static bool
radv_shader_part_binary_upload(struct radv_device *device, const struct radv_shader_part_binary *bin,
                               struct radv_shader_part *shader_part)
{
   struct radv_shader_dma_submission *submission = NULL;
   void *dest_ptr;

   if (device->shader_use_invisible_vram) {
      uint64_t va = radv_buffer_get_va(shader_part->alloc->arena->bo) + shader_part->alloc->offset;
      submission = radv_shader_dma_get_submission(device, shader_part->alloc->arena->bo, va, bin->code_size);
      if (!submission)
         return false;

      dest_ptr = submission->ptr;
   } else {
      dest_ptr = shader_part->alloc->arena->ptr + shader_part->alloc->offset;
   }

   memcpy(dest_ptr, bin->data, bin->code_size);

   if (device->shader_use_invisible_vram) {
      if (!radv_shader_dma_submit(device, submission, &shader_part->upload_seq))
         return false;
   }

   return true;
}

struct radv_shader_part *
radv_shader_part_create(struct radv_device *device, struct radv_shader_part_binary *binary, unsigned wave_size)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_shader_part *shader_part;

   shader_part = calloc(1, sizeof(struct radv_shader_part));
   if (!shader_part)
      return NULL;

   shader_part->ref_count = 1;
   shader_part->code_size = binary->code_size;
   shader_part->rsrc1 =
      S_00B848_VGPRS((binary->num_vgprs - 1) / (wave_size == 32 ? 8 : 4)) | S_00B228_SGPRS((binary->num_sgprs - 1) / 8);
   shader_part->disasm_string = binary->disasm_size ? strdup((const char *)(binary->data + binary->code_size)) : NULL;

   shader_part->spi_shader_col_format = binary->info.spi_shader_col_format;
   shader_part->cb_shader_mask = binary->info.cb_shader_mask;
   shader_part->db_shader_control = binary->info.db_shader_control;
   shader_part->spi_shader_z_format = binary->info.spi_shader_z_format;

   if (pdev->info.gfx_level >= GFX11)
      shader_part->inst_pref_size = radv_get_inst_pref_size(&device->compiler_info, binary->exec_size);

   /* Allocate memory and upload. */
   shader_part->alloc = radv_alloc_shader_memory(device, shader_part->code_size, false, NULL);
   if (!shader_part->alloc)
      goto fail;

   shader_part->bo = shader_part->alloc->arena->bo;
   shader_part->va = radv_buffer_get_va(shader_part->bo) + shader_part->alloc->offset;

   if (!radv_shader_part_binary_upload(device, binary, shader_part))
      goto fail;

   return shader_part;

fail:
   radv_shader_part_destroy(device, shader_part);
   return NULL;
}

void
radv_shader_part_cache_init(struct radv_shader_part_cache *cache, struct radv_shader_part_cache_ops *ops)
{
   cache->ops = ops;
   _mesa_set_init(&cache->entries, NULL, cache->ops->hash, cache->ops->equals);
   simple_mtx_init(&cache->lock, mtx_plain);
}

void
radv_shader_part_cache_finish(struct radv_device *device, struct radv_shader_part_cache *cache)
{
   set_foreach (&cache->entries, entry)
      radv_shader_part_unref(device, radv_shader_part_from_cache_entry(entry->key));
   simple_mtx_destroy(&cache->lock);
   _mesa_set_fini(&cache->entries, NULL);
}

/*
 * A cache with atomics-free fast path for prolog / epilog lookups.
 *
 * VS prologs and PS/TCS epilogs are used to support dynamic states. In
 * particular dynamic blend state is heavily used by Zink. These are called
 * every frame as a part of command buffer building, so these functions are
 * on the hot path.
 *
 * Originally this was implemented with a rwlock, but this lead to high
 * overhead. To avoid locking altogether in the hot path, the cache is done
 * at two levels: one at device level, and another at each CS. Access to the
 * CS cache is externally synchronized and do not require a lock.
 */
struct radv_shader_part *
radv_shader_part_cache_get(struct radv_device *device, struct radv_shader_part_cache *cache, struct set *local_entries,
                           const void *key)
{
   struct set_entry *local, *global;
   bool local_found, global_found;
   uint32_t hash = cache->ops->hash(key);

   local = _mesa_set_search_or_add_pre_hashed(local_entries, hash, key, &local_found);
   if (local_found)
      return radv_shader_part_from_cache_entry(local->key);

   simple_mtx_lock(&cache->lock);
   global = _mesa_set_search_or_add_pre_hashed(&cache->entries, hash, key, &global_found);
   if (global_found) {
      simple_mtx_unlock(&cache->lock);
      local->key = global->key;
      return radv_shader_part_from_cache_entry(global->key);
   }

   struct radv_shader_part *shader_part = cache->ops->create(device, key);
   if (!shader_part) {
      _mesa_set_remove(&cache->entries, global);
      simple_mtx_unlock(&cache->lock);
      _mesa_set_remove(local_entries, local);
      return NULL;
   }

   /* Make the set entry a pointer to the key, so that the hash and equals
    * functions from radv_shader_part_cache_ops can be directly used.
    */
   global->key = &shader_part->key;
   simple_mtx_unlock(&cache->lock);
   local->key = &shader_part->key;
   return shader_part;
}

static char *
radv_gather_nir_debug_info(struct nir_shader *const *shaders, int shader_count)
{
   char **strings = malloc(shader_count * sizeof(char *));
   uint32_t total_size = 1;
   uint32_t first_line = 1;

   for (uint32_t i = 0; i < shader_count; i++) {
      char *string = nir_shader_gather_debug_info(shaders[i], "", first_line);
      strings[i] = string;

      uint32_t length = strlen(string);
      total_size += length;

      for (uint32_t c = 0; c < length; c++) {
         if (string[c] == '\n')
            first_line++;
      }
   }

   char *ret = calloc(total_size, sizeof(char));
   if (ret) {
      for (uint32_t i = 0; i < shader_count; i++)
         strcat(ret, strings[i]);
   }

   for (uint32_t i = 0; i < shader_count; i++)
      ralloc_free(strings[i]);
   free(strings);

   return ret;
}

char *
radv_dump_nir_shaders(const struct radv_compiler_info *compiler_info, struct nir_shader *const *shaders,
                      int shader_count)
{
   if (compiler_info->key.nir_debug_info)
      return radv_gather_nir_debug_info(shaders, shader_count);

   char *data = NULL;
   char *ret = NULL;
   size_t size = 0;
   struct u_memstream mem;
   if (u_memstream_open(&mem, &data, &size)) {
      FILE *const memf = u_memstream_get(&mem);
      for (int i = 0; i < shader_count; ++i)
         nir_print_shader(shaders[i], memf);
      u_memstream_close(&mem);
   }

   ret = malloc(size + 1);
   if (ret) {
      memcpy(ret, data, size);
      ret[size] = 0;
   }
   free(data);
   return ret;
}

static void
radv_aco_build_shader_binary(void **bin, const aco_callback_params *params)
{
   struct radv_shader_binary **binary = (struct radv_shader_binary **)bin;

   uint32_t debug_info_size = params->debug_info_count * sizeof(struct ac_shader_debug_info);
   uint32_t stats_size = params->stats ? sizeof(struct amd_stats) : 0;
   uint32_t symbols_size = params->num_symbols * sizeof(struct aco_symbol);

   size_t size = params->ir_size;
   size += debug_info_size;
   size += params->disasm_size;
   size += stats_size;
   size += symbols_size;
   size += params->code_dw * sizeof(uint32_t) + sizeof(struct radv_shader_binary_legacy);

   /* We need to calloc to prevent uninitialized data because this will be used
    * directly for the disk cache. Uninitialized data can appear because of
    * padding in the struct or because legacy_binary->data can be at an offset
    * from the start less than sizeof(radv_shader_binary_legacy). */
   struct radv_shader_binary_legacy *legacy_binary = (struct radv_shader_binary_legacy *)calloc(size, 1);
   legacy_binary->base.type = RADV_BINARY_TYPE_LEGACY;
   legacy_binary->base.total_size = size;
   legacy_binary->base.config = params->config;
   legacy_binary->stats_size = stats_size;
   legacy_binary->exec_size = params->exec_size;
   legacy_binary->code_size = params->code_dw * sizeof(uint32_t);
   legacy_binary->ir_size = params->ir_size;
   legacy_binary->disasm_size = params->disasm_size;
   legacy_binary->debug_info_size = debug_info_size;
   legacy_binary->symbols_size = symbols_size;

   struct radv_shader_binary_layout layout = radv_shader_binary_get_layout(legacy_binary);

   if (stats_size)
      amd_stats_serialize(layout.stats, params->stats);

   memcpy(layout.code, params->code, params->code_dw * sizeof(uint32_t));

   if (symbols_size)
      memcpy(layout.symbols, params->symbols, symbols_size);

   if (params->ir_size)
      memcpy(layout.ir, params->ir_str, params->ir_size);

   if (params->disasm_size)
      memcpy(layout.disasm, params->disasm_str, params->disasm_size);

   if (debug_info_size)
      memcpy(layout.debug_info, params->debug_info, debug_info_size);

   *binary = (struct radv_shader_binary *)legacy_binary;
}

struct build_binary_elf_args {
   const struct ac_compiler_info *compiler_info;
   struct radv_shader_binary_rtld *binary;
};

static void
radv_aco_build_shader_binary_elf(void **bin, const aco_callback_params *params)
{
   struct build_binary_elf_args *args = (struct build_binary_elf_args *)bin;

   size_t size = aco_create_elf(args->compiler_info, params, sizeof(struct radv_shader_binary_rtld), params->ir_size,
                                (void **)&args->binary);

   args->binary->base.type = RADV_BINARY_TYPE_RTLD;
   args->binary->base.total_size = sizeof(struct radv_shader_binary_rtld) + size + params->ir_size;
   args->binary->elf_size = size;
   args->binary->llvm_ir_size = params->ir_size;
   memcpy(args->binary->data + args->binary->elf_size, params->ir_str, params->ir_size);
}

#if RADV_LLVM_BACKEND
static void
radv_fill_llvm_compiler_options(struct radv_llvm_compiler_options *options,
                                const struct radv_compiler_info *compiler_info, bool should_use_wgp,
                                bool can_dump_shader, bool keep_shader_info)
{
   options->compiler_info = compiler_info->ac;
   options->family = compiler_info->key.family;
   options->address32_hi = compiler_info->hw.address32_hi;
   /* robust_buffer_access_llvm here used by LLVM only, robustness features are not exposed there. */
   options->robust_buffer_access = compiler_info->key.robust_buffer_access;
   options->wgp_mode = should_use_wgp;
   options->dump_shader = can_dump_shader;
   options->dump_preoptir = options->dump_shader && compiler_info->debug.dump_preopt_ir;
   options->record_ir = keep_shader_info;
   options->check_ir = compiler_info->debug.check_ir;
}
#endif

void
radv_aco_fill_compiler_options(struct aco_compiler_options *aco_info, const struct radv_compiler_info *compiler_info,
                               const struct radv_shader_stage_key *stage_key,
                               const struct radv_graphics_state_key *gfx_state, bool should_use_wgp,
                               bool can_dump_shader)
{
   aco_info->dump_ir = can_dump_shader && compiler_info->debug.dump_backend_ir;
   aco_info->dump_preoptir = can_dump_shader && compiler_info->debug.dump_preopt_ir;
   aco_info->record_asm = stage_key->keep_executable_info || can_dump_shader;
   aco_info->record_ir = stage_key->keep_executable_info;
   aco_info->record_stats = stage_key->keep_statistic_info;
   aco_info->enable_mrt_output_nan_fixup = gfx_state ? gfx_state->ps.epilog.enable_mrt_output_nan_fixup : false;
   aco_info->wgp_mode = should_use_wgp;
   aco_info->compiler_info = compiler_info->ac;
   aco_info->is_opengl = false;
   aco_info->optimisations_disabled = stage_key->optimisations_disabled;
   aco_info->gfx_level = compiler_info->ac->gfx_level;
   aco_info->family = compiler_info->debug.family;
   aco_info->address32_hi = compiler_info->hw.address32_hi;
}

static nir_component_mask_t
non_uniform_access_callback(const nir_src *src, void *_)
{
   if (src->ssa->num_components == 1)
      return 0x1;
   return nir_chase_binding(*src).success ? 0x2 : 0x3;
}

void
radv_postprocess_nir(const struct radv_compiler_info *compiler_info, const struct radv_graphics_state_key *gfx_state,
                     struct radv_shader_stage *stage)
{
   enum amd_gfx_level gfx_level = compiler_info->ac->gfx_level;
   const bool use_llvm = compiler_info->key.use_llvm;
   bool progress;

   if (stage->nir->info.uses_printf)
      NIR_PASS(_, stage->nir, radv_nir_lower_printf, compiler_info->debug.debug_nir);

   /* Wave and workgroup size should already be filled. */
   assert(stage->info.wave_size && stage->info.workgroup_size);

   if (stage->stage == MESA_SHADER_FRAGMENT) {
      if (!stage->key.optimisations_disabled) {
         NIR_PASS(_, stage->nir, nir_opt_cse);
      }
      NIR_PASS(_, stage->nir, radv_nir_lower_fs_intrinsics, stage, gfx_state);
   }

   /* LLVM could support more of these in theory. */
   radv_nir_opt_tid_function_options tid_options = {
      .use_masked_swizzle_amd = true,
      .use_dpp16_shift_amd = !use_llvm && gfx_level >= GFX8,
      .use_quad_swap_broadcast = true,
      .use_clustered_rotate = !use_llvm,
      .use_permute16_amd = !use_llvm && gfx_level >= GFX10,
      .use_dpp8_swizzle_amd = !use_llvm && gfx_level >= GFX10,
      .hw_ballot_bit_size = stage->info.wave_size,
      .hw_ballot_num_comp = 1,
   };
   NIR_PASS(_, stage->nir, radv_nir_opt_tid_function, &tid_options);

   NIR_PASS(_, stage->nir, ac_nir_flag_smem_for_loads, gfx_level, use_llvm);

   NIR_PASS(_, stage->nir, nir_lower_memory_model);

   nir_load_store_vectorize_options vectorize_opts = {
      .modes = nir_var_mem_ssbo | nir_var_mem_ubo | nir_var_mem_push_const | nir_var_mem_shared | nir_var_mem_global |
               nir_var_shader_temp,
      .callback = ac_nir_mem_vectorize_callback,
      .cb_data = &(struct ac_nir_config){gfx_level, !use_llvm},
      .robust_modes = 0,
      .bounds_checked_modes = nir_var_mem_ssbo | nir_var_mem_ubo | nir_var_mem_shared,
      /* Only vectorize shared2 during late optimizations. */
      .has_shared2_amd = false,
   };

   if (stage->key.uniform_robustness2)
      vectorize_opts.robust_modes |= nir_var_mem_ubo;

   if (stage->key.storage_robustness2)
      vectorize_opts.robust_modes |= nir_var_mem_ssbo;

   bool constant_fold_for_push_const = false;
   if (!stage->key.optimisations_disabled) {
      progress = false;
      NIR_PASS(progress, stage->nir, nir_opt_load_store_vectorize, &vectorize_opts);
      if (progress) {
         NIR_PASS(_, stage->nir, nir_opt_copy_prop);
         NIR_PASS(_, stage->nir, nir_opt_shrink_stores, !compiler_info->key.disable_shrink_image_store);

         constant_fold_for_push_const = true;
      }
   }

   enum nir_lower_non_uniform_access_type lower_non_uniform_access_types =
      nir_lower_non_uniform_ubo_access | nir_lower_non_uniform_ssbo_access | nir_lower_non_uniform_texture_access |
      nir_lower_non_uniform_image_access | nir_lower_non_uniform_texture_query | nir_lower_non_uniform_image_query;

   /* In practice, most shaders do not have non-uniform-qualified
    * accesses (see
    * https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/17558#note_1475069)
    * thus a cheaper and likely to fail check is run first.
    */
   if (nir_has_non_uniform_access(stage->nir, lower_non_uniform_access_types)) {
      if (!stage->key.optimisations_disabled) {
         NIR_PASS(_, stage->nir, nir_opt_non_uniform_access);
      }

      if (!use_llvm) {
         nir_lower_non_uniform_access_options options = {
            .types = lower_non_uniform_access_types,
            .callback = &non_uniform_access_callback,
            .callback_data = NULL,
         };
         NIR_PASS(_, stage->nir, nir_lower_non_uniform_access, &options);
      }
   }

   progress = false;
   NIR_PASS(progress, stage->nir, ac_nir_lower_mem_access_bit_sizes, gfx_level, use_llvm);
   if (progress)
      constant_fold_for_push_const = true;

   NIR_PASS(_, stage->nir, ac_nir_lower_tex_coords,
            &(ac_nir_lower_tex_coords_options){
               .lower_array_layer_round_even = !compiler_info->ac->conformant_trunc_coord,
               .fix_derivs_in_divergent_cf = stage->stage == MESA_SHADER_FRAGMENT && !use_llvm,
               .max_wqm_vgprs = 64, // TODO: improve spiller and RA support for linear VGPRs
            });

   NIR_PASS(_, stage->nir, ac_nir_lower_image_tex,
            &(ac_nir_lower_image_tex_options){
               .gfx_level = gfx_level,
            });

   if (stage->nir->info.uses_resource_info_query)
      NIR_PASS(_, stage->nir, ac_nir_lower_resinfo, gfx_level);

   /* Ensure split load_push_constant still have constant offsets, for radv_nir_lower_descriptors. */
   if (constant_fold_for_push_const && stage->args.ac.inline_push_const_mask)
      NIR_PASS(_, stage->nir, nir_opt_constant_folding);

   /* Optimize NIR before NGG culling */
   bool is_last_vgt_stage = radv_is_last_vgt_stage(stage);
   bool lowered_ngg = stage->info.is_ngg && is_last_vgt_stage;
   if (lowered_ngg && stage->nir->info.stage != MESA_SHADER_GEOMETRY && stage->info.has_ngg_culling)
      radv_optimize_nir_algebraic_early(stage->nir);

   /* This has to be done after nir_opt_algebraic for best descriptor vectorization, but also before
    * NGG culling.
    */
   NIR_PASS(_, stage->nir, radv_nir_lower_descriptors, compiler_info, stage);

   NIR_PASS(_, stage->nir, nir_lower_alu_width, ac_nir_opt_vectorize_cb, &gfx_level);

   nir_move_options sink_opts = nir_move_const_undef | nir_move_copies | nir_dont_move_byte_word_vecs;

   if (!stage->key.optimisations_disabled) {
      NIR_PASS(_, stage->nir, nir_opt_licm, NULL);

      if (stage->stage == MESA_SHADER_VERTEX) {
         /* Always load all VS inputs at the top to eliminate needless VMEM->s_wait->VMEM sequences.
          * Each s_wait can cost 1000 cycles, so make sure all VS input loads are grouped.
          */
         NIR_PASS(_, stage->nir, nir_opt_move_to_top, nir_move_to_top_input_loads_simple);
         NIR_PASS(_, stage->nir, nir_opt_sink, sink_opts);
         NIR_PASS(_, stage->nir, nir_opt_move, sink_opts);
      } else {
         if (stage->stage != MESA_SHADER_FRAGMENT || !compiler_info->key.disable_sinking_load_input_fs)
            sink_opts |= nir_move_load_input | nir_move_load_frag_coord;

         NIR_PASS(_, stage->nir, nir_opt_sink, sink_opts);
         NIR_PASS(_, stage->nir, nir_opt_move, sink_opts | nir_move_load_input | nir_move_load_frag_coord);
      }
   }

   /* Lower VS inputs. We need to do this after nir_opt_sink, because
    * load_input can be reordered, but buffer loads can't.
    */
   if (stage->stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_, stage->nir, radv_nir_lower_vs_inputs, compiler_info, stage, gfx_state);
   }

   /* Lower I/O intrinsics to memory instructions. */
   bool io_to_mem = radv_nir_lower_io_to_mem(compiler_info, stage);
   if (lowered_ngg) {
      radv_lower_ngg(compiler_info, stage, gfx_state);
   } else if (is_last_vgt_stage) {
      if (stage->stage != MESA_SHADER_GEOMETRY) {
         NIR_PASS(_, stage->nir, ac_nir_lower_legacy_vs, gfx_level,
                  stage->info.outinfo.clip_dist_mask | stage->info.outinfo.cull_dist_mask, false,
                  stage->info.outinfo.vs_output_param_offset, stage->info.outinfo.param_exports,
                  stage->info.outinfo.export_prim_id, false, stage->info.force_vrs_per_vertex);

      } else {
         ac_nir_lower_legacy_gs_options options = {
            .has_gen_prim_query = false,
            .has_pipeline_stats_query = false,
            .gfx_level = gfx_level,
            .export_clipdist_mask = stage->info.outinfo.clip_dist_mask | stage->info.outinfo.cull_dist_mask,
            .param_offsets = stage->info.outinfo.vs_output_param_offset,
            .has_param_exports = stage->info.outinfo.param_exports,
            .force_vrs = stage->info.force_vrs_per_vertex,
         };
         ac_nir_legacy_gs_info info = {0};

         NIR_PASS(_, stage->nir, ac_nir_lower_legacy_gs, &options, &stage->gs_copy_shader, &info);

         for (unsigned i = 0; i < 4; i++)
            stage->info.gs.num_components_per_stream[i] = info.num_components_per_stream[i];
      }
   } else if (stage->stage == MESA_SHADER_FRAGMENT) {
      ac_nir_lower_ps_late_options late_options = {
         .gfx_level = gfx_level,
         .use_aco = !use_llvm,
         .bc_optimize_for_persp = G_0286CC_PERSP_CENTER_ENA(stage->info.ps.spi_ps_input_ena) &&
                                  G_0286CC_PERSP_CENTROID_ENA(stage->info.ps.spi_ps_input_ena),
         .bc_optimize_for_linear = G_0286CC_LINEAR_CENTER_ENA(stage->info.ps.spi_ps_input_ena) &&
                                   G_0286CC_LINEAR_CENTROID_ENA(stage->info.ps.spi_ps_input_ena),
         .uses_discard = stage->info.ps.can_discard,
         .dcc_decompress_gfx11 = gfx_state->dcc_decompress_gfx11,
         .no_color_export = stage->info.ps.has_epilog,
         .no_depth_export = stage->info.ps.has_epilog,

      };

      if (!stage->info.ps.has_epilog) {
         late_options.dual_src_blend = gfx_state->ps.epilog.mrt0_is_dual_src;
         late_options.color_is_int8 = gfx_state->ps.epilog.color_is_int8;
         late_options.color_is_int10 = gfx_state->ps.epilog.color_is_int10;
         late_options.enable_mrt_output_nan_fixup =
            gfx_state->ps.epilog.enable_mrt_output_nan_fixup && !stage->nir->info.internal;
         /* Need to filter out unwritten color slots. */
         late_options.spi_shader_col_format =
            gfx_state->ps.epilog.spi_shader_col_format & stage->info.ps.colors_written;
         late_options.alpha_to_one = gfx_state->ps.epilog.alpha_to_one;
         late_options.alpha_to_coverage_via_mrtz = stage->info.ps.writes_mrt0_alpha_to_mrtz;
      }

      NIR_PASS(_, stage->nir, ac_nir_lower_ps_late, &late_options);
      NIR_PASS(_, stage->nir, ac_nir_lower_fs_input_loads, &stage->args.ac);
   }

   if (radv_shader_should_clear_lds(compiler_info, stage->nir)) {
      const unsigned chunk_size = 16; /* max single store size */
      const unsigned shared_size = align(stage->nir->info.shared_size, chunk_size);
      NIR_PASS(_, stage->nir, nir_clear_shared_memory, shared_size, chunk_size);
   }

   /* This must be after lowering resources to descriptor loads and before lowering intrinsics
    * to args and lowering int64.
    */
   if (!use_llvm)
      ac_nir_optimize_uniform_atomics(stage->nir);

   NIR_PASS(_, stage->nir, nir_opt_uniform_subgroup,
            &(struct nir_lower_subgroups_options){
               .subgroup_size = stage->info.wave_size,
               .ballot_bit_size = stage->info.wave_size,
               .ballot_components = 1,
               .lower_ballot_bit_count_to_mbcnt_amd = true,
            });

   NIR_PASS(_, stage->nir, nir_opt_idiv_const, 8);

   NIR_PASS(_, stage->nir, nir_lower_idiv,
            &(nir_lower_idiv_options){
               .allow_fp16 = gfx_level >= GFX9,
            });

   NIR_PASS(
      _, stage->nir, ac_nir_lower_intrinsics_to_args, &stage->args.ac,
      &(ac_nir_lower_intrinsics_to_args_options){
         .gfx_level = gfx_level,
         .has_ls_vgpr_init_bug = compiler_info->ac->has_ls_vgpr_init_bug && gfx_state && !gfx_state->vs.has_prolog,
         .hw_stage = radv_select_hw_stage(&stage->info, gfx_level),
         .wave_size = stage->info.wave_size,
         .workgroup_size = stage->info.workgroup_size,
         .use_llvm = use_llvm,
         .load_grid_size_from_user_sgpr = compiler_info->key.load_grid_size_from_user_sgpr,
      });
   NIR_PASS(_, stage->nir, radv_nir_lower_abi, gfx_level, stage, gfx_state, compiler_info->hw.address32_hi);

   if (!stage->key.optimisations_disabled) {
      NIR_PASS(_, stage->nir, nir_opt_dce);

      NIR_PASS(_, stage->nir, nir_opt_copy_prop);
      NIR_PASS(_, stage->nir, nir_opt_constant_folding);
      NIR_PASS(_, stage->nir, nir_opt_cse);
      NIR_PASS(_, stage->nir, nir_opt_if, nir_opt_if_optimize_phi_true_false);
      NIR_PASS(_, stage->nir, nir_opt_shrink_vectors, true);

      NIR_PASS(_, stage->nir, ac_nir_flag_smem_for_loads, gfx_level, use_llvm);
      NIR_PASS(_, stage->nir, ac_nir_lower_mem_access_bit_sizes, gfx_level, use_llvm);

      nir_load_store_vectorize_options late_vectorize_opts = {
         .modes =
            nir_var_mem_global | nir_var_mem_shared | nir_var_shader_out | nir_var_mem_task_payload | nir_var_shader_in,
         .callback = ac_nir_mem_vectorize_callback,
         .cb_data = &(struct ac_nir_config){gfx_level, !use_llvm},
         .robust_modes = 0,
         .bounds_checked_modes = nir_var_mem_ssbo | nir_var_mem_ubo | nir_var_mem_shared,
         .has_shared2_amd = true,
      };
      NIR_PASS(_, stage->nir, nir_opt_load_store_vectorize, &late_vectorize_opts);
   }

   NIR_PASS(_, stage->nir, ac_nir_lower_mem_access_bit_sizes, gfx_level, use_llvm);
   NIR_PASS(_, stage->nir, ac_nir_lower_global_access, gfx_level);
   NIR_PASS(_, stage->nir, nir_lower_int64);

   if (compiler_info->key.mitigate_smem_with_null_prt)
      NIR_PASS(_, stage->nir, ac_nir_fixup_smem_loads_null_prt, compiler_info->hw.address_prt_wa_control_bit);

   if (compiler_info->key.mitigate_smem_oob)
      NIR_PASS(_, stage->nir, ac_nir_fixup_mem_access_gfx6, &stage->args.ac, 4096, true, true);

   bool opt_intrinsics = false;
   if (gfx_level >= GFX11)
      NIR_PASS(opt_intrinsics, stage->nir, ac_nir_opt_flip_if_for_mem_loads);
   if (opt_intrinsics) /* optimize inot(inverse_ballot) */
      NIR_PASS(_, stage->nir, nir_opt_intrinsics);

   NIR_PASS(_, stage->nir, nir_opt_uub, &(nir_opt_uub_options){0});

   radv_optimize_nir_algebraic(
      stage->nir, io_to_mem || lowered_ngg || stage->stage == MESA_SHADER_COMPUTE || stage->stage == MESA_SHADER_TASK,
      gfx_level >= GFX8, gfx_level);

   if (stage->nir->info.cs.has_cooperative_matrix)
      NIR_PASS(_, stage->nir, radv_nir_opt_cooperative_matrix, gfx_level);

   NIR_PASS(_, stage->nir, nir_lower_fp16_casts, nir_lower_fp16_split_fp64);

   if (ac_nir_might_lower_bit_size(stage->nir)) {
      if (gfx_level >= GFX8)
         nir_divergence_analysis(stage->nir);

      NIR_PASS(_, stage->nir, nir_lower_bit_size, ac_nir_lower_bit_size_callback, &gfx_level);
   }
   if (gfx_level >= GFX8) {
      bool run_copy_prop = false;
      NIR_PASS(run_copy_prop, stage->nir, ac_nir_opt_16bit_tex_image, gfx_level);

      /* Optimizing 16bit texture/image dests leaves scalar moves that need to be removed
       * before the next alu nir_lower_alu_width, otherwise we might end up with invalid swizzles
       * in the backend.
       * It also allows nir_opt_vectorize to make more progress.
       */
      if (run_copy_prop) {
         NIR_PASS(_, stage->nir, nir_opt_copy_prop);
         NIR_PASS(_, stage->nir, nir_opt_dce);
      }

      if (!stage->key.optimisations_disabled && compiler_info->ac->has_packed_math_16bit) {
         NIR_PASS(_, stage->nir, nir_opt_vectorize, ac_nir_opt_vectorize_cb, &gfx_level);
      }
   }

   /* cleanup passes */
   NIR_PASS(_, stage->nir, nir_lower_alu_width, ac_nir_opt_vectorize_cb, &gfx_level);

   NIR_PASS(_, stage->nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, stage->nir, nir_opt_copy_prop);
   NIR_PASS(_, stage->nir, nir_opt_dce);

   if (!stage->key.optimisations_disabled) {
      sink_opts |= nir_move_comparisons | nir_move_load_ubo | nir_move_load_ssbo | nir_move_alu;
      NIR_PASS(_, stage->nir, nir_opt_sink, sink_opts);

      nir_move_options move_opts = nir_move_const_undef | nir_move_load_ubo | nir_move_load_input |
                                   nir_move_load_frag_coord | nir_move_comparisons | nir_move_copies |
                                   nir_dont_move_byte_word_vecs | nir_move_alu;
      NIR_PASS(_, stage->nir, nir_opt_move, move_opts);

      /* Run nir_opt_move again to make sure that comparision are as close as possible to the first use to prevent SCC
       * spilling.
       */
      NIR_PASS(_, stage->nir, nir_opt_move, nir_move_comparisons);
   }

   stage->info.nir_shared_size = stage->nir->info.shared_size;
}

bool
radv_shader_should_clear_lds(const struct radv_compiler_info *compiler_info, const nir_shader *shader)
{
   return (shader->info.stage == MESA_SHADER_COMPUTE || shader->info.stage == MESA_SHADER_MESH ||
           shader->info.stage == MESA_SHADER_TASK) &&
          shader->info.shared_size > 0 && compiler_info->key.clear_lds;
}

struct radv_shader_binary *
radv_shader_nir_to_asm(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *pl_stage,
                       struct nir_shader *const *shaders, int shader_count,
                       const struct radv_graphics_state_key *gfx_state)
{
   mesa_shader_stage stage = shaders[shader_count - 1]->info.stage;
   struct radv_shader_info *info = &pl_stage->info;
   const struct radv_shader_args *args = &pl_stage->args;

   bool wgp_mode = radv_should_use_wgp_mode(compiler_info->ac->gfx_level, stage, info);
   bool dump_shader = false;
   for (unsigned i = 0; i < shader_count; ++i)
      dump_shader |= radv_can_dump_shader(compiler_info, shaders[i]);

#if AMD_LLVM_AVAILABLE
   if (compiler_info->key.use_llvm || dump_shader || pl_stage->key.keep_executable_info)
      ac_init_llvm_once();
#endif

   struct radv_shader_binary *binary = NULL;
#if RADV_LLVM_BACKEND
   if (compiler_info->key.use_llvm) {
      struct radv_llvm_compiler_options options = {0};
      radv_fill_llvm_compiler_options(&options, compiler_info, wgp_mode, dump_shader,
                                      pl_stage->key.keep_executable_info);

      llvm_compile_shader(&options, info, shader_count, shaders, &binary, args);
#else
   if (false) {
#endif
   } else {
      struct aco_shader_info ac_info;
      struct aco_compiler_options ac_opts;
      radv_aco_fill_compiler_options(&ac_opts, compiler_info, &pl_stage->key, gfx_state, wgp_mode, dump_shader);
      radv_aco_convert_shader_info(&ac_info, info, args, compiler_info);

      if (compiler_info->key.use_elf) {
         struct build_binary_elf_args elf;
         elf.compiler_info = compiler_info->ac;
         aco_compile_shader(&ac_opts, &ac_info, shader_count, shaders, &args->ac, &radv_aco_build_shader_binary_elf,
                            (void **)&elf);
         binary = &elf.binary->base;
      } else {
         aco_compile_shader(&ac_opts, &ac_info, shader_count, shaders, &args->ac, &radv_aco_build_shader_binary,
                            (void **)&binary);
      }
   }

   binary->info = *info;

   if (!radv_postprocess_binary_config(compiler_info, binary, args)) {
      free(binary);
      return NULL;
   }

   return binary;
}

void
radv_shader_dump_asm(const struct radv_compiler_info *compiler_info, struct radv_shader_debug_info *debug,
                     const struct radv_shader_binary *binary, const struct radv_shader_info *info)
{
   if (debug->dump_shader && compiler_info->debug.dump_asm) {
      struct radv_shader_binary_debug_info dbg = radv_parse_binary_debug_info(compiler_info, binary);

      const char *sep = "";
      u_foreach_bit (stage, debug->stages) {
         fprintf(stderr, "%s%s", sep, radv_get_shader_name(info, stage));
         sep = " + ";
      }

      fprintf(stderr, "\ndisasm:\n");
      uint32_t code_size = 0;
      if (dbg.debug_info_count > 0) {
         struct radv_shader_binary_legacy *bin = (struct radv_shader_binary_legacy *)binary;
         code_size = bin->code_size;
      }
      radv_print_shader_disasm(&dbg, code_size, NULL, 0, 0x00000000, stderr);
      fprintf(stderr, "\n");

      radv_free_binary_debug_info(&dbg);
   }
}

struct radv_shader *
radv_create_trap_handler_shader(struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   struct radv_shader_stage_key stage_key = {0};
   struct radv_shader_stage stage = {0};
   const bool dump_shader = !!(RADV_DEBUG(instance, DUMP_TRAP_HANDLER));

   nir_builder b = radv_meta_nir_init_shader(MESA_SHADER_COMPUTE, "meta_trap_handler");

   stage.stage = MESA_SHADER_COMPUTE;
   stage.info.wave_size = 64;
   stage.info.workgroup_size = 64;
   stage.info.stage = MESA_SHADER_COMPUTE;
   stage.info.type = RADV_SHADER_TYPE_TRAP_HANDLER;

   struct radv_shader_debug_info debug = {0};
   radv_declare_shader_args(&device->compiler_info, NULL, &stage, MESA_SHADER_NONE, &debug);

#if AMD_LLVM_AVAILABLE
   if (dump_shader)
      ac_init_llvm_once();
#endif

   struct radv_shader_binary *binary = NULL;

   struct aco_shader_info ac_info;
   radv_aco_convert_shader_info(&ac_info, &stage.info, &stage.args, &device->compiler_info);

   struct aco_compiler_options ac_opts;
   bool wgp_mode = radv_should_use_wgp_mode(pdev->info.gfx_level, MESA_SHADER_COMPUTE, &stage.info);
   radv_aco_fill_compiler_options(&ac_opts, &device->compiler_info, &stage_key, NULL, wgp_mode, dump_shader);

   aco_compile_trap_handler(&ac_opts, &ac_info, &stage.args.ac, &radv_aco_build_shader_binary, (void **)&binary);
   binary->info = stage.info;

   radv_postprocess_binary_config(&device->compiler_info, binary, &stage.args);

   struct radv_shader *shader;
   radv_shader_create_uncached(device, binary, false, NULL, &debug, &shader);

   if (dump_shader) {
      struct radv_shader_binary_debug_info dbg = radv_parse_binary_debug_info(&device->compiler_info, shader->binary);
      fprintf(stderr, "Trap handler");
      fprintf(stderr, "\ndisasm:\n");
      radv_print_shader_disasm(&dbg, shader->code_size, NULL, 0, 0x00000000, stderr);
      fprintf(stderr, "\n");
      radv_free_binary_debug_info(&dbg);
   }

   ralloc_free(b.shader);
   free(binary);

   return shader;
}

void
radv_aco_build_shader_part(void **bin, const aco_callback_params *params)
{
   struct radv_shader_part_binary **binary = (struct radv_shader_part_binary **)bin;
   size_t size = params->code_dw * sizeof(uint32_t) + sizeof(struct radv_shader_part_binary);

   size += params->disasm_size;
   struct radv_shader_part_binary *part_binary = (struct radv_shader_part_binary *)calloc(size, 1);

   part_binary->num_sgprs = params->config.num_sgprs;
   part_binary->num_vgprs = params->config.num_vgprs;
   part_binary->total_size = size;
   part_binary->code_size = params->code_dw * sizeof(uint32_t);
   part_binary->exec_size = params->exec_size;
   memcpy(part_binary->data, params->code, part_binary->code_size);
   if (params->disasm_size) {
      memcpy((char *)part_binary->data + part_binary->code_size, params->disasm_str, params->disasm_size);
      part_binary->disasm_size = params->disasm_size;
   }

   *binary = part_binary;
}

struct radv_shader *
radv_compile_rt_prolog(struct radv_device *device, struct radv_shader_stage *stage,
                       struct radv_shader_debug_info *debug)
{
   const struct radv_compiler_info *compiler_info = &device->compiler_info;
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_instance *instance = radv_physical_device_instance(pdev);
   bool dump_shader = RADV_DEBUG(instance, DUMP_PROLOGS);
   bool keep_shader_info = radv_device_fault_detection_enabled(device);

   struct radv_shader *prolog;

   if (dump_shader) {
      simple_mtx_lock(compiler_info->debug.shader_dump_mtx);

      if (compiler_info->debug.dump_nir)
         nir_print_shader(stage->nir, stderr);
   }

#if AMD_LLVM_AVAILABLE
   if (dump_shader || keep_shader_info)
      ac_init_llvm_once();
#endif

   struct radv_shader_binary *binary = NULL;
   struct radv_shader_stage_key stage_key = {0};
   struct aco_shader_info ac_info;
   struct aco_compiler_options ac_opts;
   stage_key.keep_executable_info = keep_shader_info;
   radv_aco_convert_shader_info(&ac_info, &stage->info, &stage->args, compiler_info);
   radv_aco_fill_compiler_options(&ac_opts, compiler_info, &stage_key, NULL, false, dump_shader);
   aco_compile_shader(&ac_opts, &ac_info, 1, &stage->nir, &stage->args.ac, &radv_aco_build_shader_binary,
                      (void **)&binary);
   binary->info = stage->info;

   radv_postprocess_binary_config(compiler_info, binary, &stage->args);
   radv_shader_create_uncached(device, binary, false, NULL, debug, &prolog);
   if (!prolog)
      goto done;

   if (dump_shader) {
      struct radv_shader_binary_debug_info dbg = radv_parse_binary_debug_info(&device->compiler_info, binary);
      fprintf(stderr, "Raytracing prolog");
      fprintf(stderr, "\ndisasm:\n");
      radv_print_shader_disasm(&dbg, prolog->code_size, NULL, 0, 0x00000000, stderr);
      fprintf(stderr, "\n");
      simple_mtx_unlock(compiler_info->debug.shader_dump_mtx);
      radv_free_binary_debug_info(&dbg);
   }

done:
   free(binary);
   return prolog;
}

void
radv_shader_part_destroy(struct radv_device *device, struct radv_shader_part *shader_part)
{
   assert(shader_part->ref_count == 0);

   if (device->shader_use_invisible_vram) {
      /* Wait for any pending upload to complete, or we'll be writing into freed shader memory. */
      radv_shader_wait_for_upload(device, shader_part->upload_seq);
   }

   radv_free_shader_memory(device, shader_part->alloc);
   free(shader_part->disasm_string);
   free(shader_part);
}

uint64_t
radv_shader_get_va(const struct radv_shader *shader)
{
   return shader->va;
}

struct radv_shader *
radv_find_shader(struct radv_device *device, uint64_t pc)
{
   mtx_lock(&device->shader_arena_mutex);
   list_for_each_entry (struct radv_shader_arena, arena, &device->shader_arenas, list) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
      list_for_each_entry (union radv_shader_arena_block, block, &arena->entries, list) {
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
         uint64_t start = radv_buffer_get_va(block->arena->bo) + block->offset;
         start &= ((1ull << 48) - 1);
         if (!block->freelist.prev && pc >= start && pc < start + block->size) {
            mtx_unlock(&device->shader_arena_mutex);
            return (struct radv_shader *)block->freelist.next;
         }
      }
   }

   mtx_unlock(&device->shader_arena_mutex);
   return NULL;
}

const char *
radv_get_shader_name(const struct radv_shader_info *info, mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      if (info->vs.as_ls)
         return "Vertex Shader as LS";
      else if (info->vs.as_es)
         return "Vertex Shader as ES";
      else if (info->is_ngg)
         return "Vertex Shader as ESGS";
      else
         return "Vertex Shader as VS";
   case MESA_SHADER_TESS_CTRL:
      return "Tessellation Control Shader";
   case MESA_SHADER_TESS_EVAL:
      if (info->tes.as_es)
         return "Tessellation Evaluation Shader as ES";
      else if (info->is_ngg)
         return "Tessellation Evaluation Shader as ESGS";
      else
         return "Tessellation Evaluation Shader as VS";
   case MESA_SHADER_GEOMETRY:
      return "Geometry Shader";
   case MESA_SHADER_FRAGMENT:
      return "Pixel Shader";
   case MESA_SHADER_COMPUTE:
      return info->type == RADV_SHADER_TYPE_TRAP_HANDLER ? "Trap Handler Shader" : "Compute Shader";
   case MESA_SHADER_MESH:
      return "Mesh Shader as NGG";
   case MESA_SHADER_TASK:
      return "Task Shader as CS";
   case MESA_SHADER_RAYGEN:
      return "Ray Generation Shader as CS Function";
   case MESA_SHADER_CLOSEST_HIT:
      return "Closest Hit Shader as CS Function";
   case MESA_SHADER_INTERSECTION:
      return "Intersection Shader as CS Function";
   case MESA_SHADER_ANY_HIT:
      return "Any Hit Shader as CS Function";
   case MESA_SHADER_MISS:
      return "Miss Shader as CS Function";
   case MESA_SHADER_CALLABLE:
      return "Callable Shader as CS Function";
   default:
      return "Unknown shader";
   };
}

uint32_t
radv_get_user_sgpr(const struct radv_shader *shader, int idx)
{
   const uint32_t offset = radv_get_user_sgpr_loc(shader, idx);

   return offset ? ((offset - SI_SH_REG_OFFSET) >> 2) : 0;
}
