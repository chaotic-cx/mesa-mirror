/*
 * Copyright © 2024 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pco_nir.c
 *
 * \brief NIR-specific functions.
 */

#include "nir/nir_builder.h"
#include "pco.h"
#include "pco_internal.h"
#include "pvr_limits.h"

#include <stdio.h>

/** SPIR-V to NIR options. */
static const struct spirv_to_nir_options spirv_options = {
   .environment = NIR_SPIRV_VULKAN,

   .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,

   .min_ubo_alignment = PVR_UNIFORM_BUFFER_OFFSET_ALIGNMENT,
   .min_ssbo_alignment = PVR_STORAGE_BUFFER_OFFSET_ALIGNMENT,
};

/** NIR options. */
static const nir_shader_compiler_options nir_options = {
   .fuse_ffma32 = true,

   .has_fused_comp_and_csel = true,

   .instance_id_includes_base_index = true,

   .lower_fdiv = true,
   .lower_fquantize2f16 = true,
   .lower_layer_fs_input_to_sysval = true,
   .compact_arrays = true,
};

/**
 * \brief Returns the SPIR-V to NIR options.
 *
 * \return The SPIR-V to NIR options.
 */
const struct spirv_to_nir_options *pco_spirv_options(void)
{
   return &spirv_options;
}

/**
 * \brief Returns the NIR options for a PCO compiler
 * context.
 *
 * \return The NIR options.
 */
const nir_shader_compiler_options *pco_nir_options(void)
{
   return &nir_options;
}

/**
 * \brief Runs pre-processing passes on a NIR shader.
 *
 * \param[in] ctx PCO compiler context.
 * \param[in,out] nir NIR shader.
 */
void pco_preprocess_nir(pco_ctx *ctx, nir_shader *nir)
{
   if (nir->info.internal)
      NIR_PASS(_, nir, nir_lower_returns);

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_,
            nir,
            nir_split_struct_vars,
            nir_var_function_temp | nir_var_shader_temp);
   NIR_PASS(_,
            nir,
            nir_split_array_vars,
            nir_var_function_temp | nir_var_shader_temp);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      const struct nir_lower_sysvals_to_varyings_options sysvals_to_varyings = {
         .frag_coord = true,
         .point_coord = true,
      };
      NIR_PASS(_, nir, nir_lower_sysvals_to_varyings, &sysvals_to_varyings);
   }

   NIR_PASS(_, nir, nir_lower_system_values);

   NIR_PASS(_,
            nir,
            nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir),
            true,
            true);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_remove_dead_derefs);

   NIR_PASS(_,
            nir,
            nir_lower_indirect_derefs,
            nir_var_shader_in | nir_var_shader_out,
            UINT32_MAX);

   NIR_PASS(_,
            nir,
            nir_remove_dead_variables,
            nir_var_function_temp | nir_var_shader_temp,
            NULL);
   NIR_PASS(_, nir, nir_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_cse);

   if (pco_should_print_nir(nir)) {
      puts("after pco_preprocess_nir:");
      nir_print_shader(nir, stdout);
   }
}

/**
 * \brief Returns the GLSL type size.
 *
 * \param[in] type Type.
 * \param[in] bindless Whether the access is bindless.
 * \return The size.
 */
static int glsl_type_size(const struct glsl_type *type, UNUSED bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/**
 * \brief Returns the vectorization with for a given instruction.
 *
 * \param[in] instr Instruction.
 * \param[in] data User data.
 * \return The vectorization width.
 */
static uint8_t vectorize_filter(const nir_instr *instr, UNUSED const void *data)
{
   if (instr->type == nir_instr_type_load_const)
      return 1;

   if (instr->type != nir_instr_type_alu)
      return 0;

   /* TODO */
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   switch (alu->op) {
   default:
      break;
   }

   /* Basic for now. */
   return 2;
}

/**
 * \brief Filter for fragment shader inputs that need to be scalar.
 *
 * \param[in] instr Instruction.
 * \param[in] data User data.
 * \return True if the instruction was found.
 */
static bool frag_in_scalar_filter(const nir_instr *instr, const void *data)
{
   assert(instr->type == nir_instr_type_intrinsic);
   nir_shader *nir = (nir_shader *)data;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   gl_varying_slot location = nir_intrinsic_io_semantics(intr).location;
   if (location == VARYING_SLOT_POS)
      return true;

   nir_variable *var =
      nir_find_variable_with_location(nir, nir_var_shader_in, location);
   assert(var);

   if (var->data.interpolation == INTERP_MODE_FLAT)
      return true;

   return false;
}

/**
 * \brief Lowers a NIR shader.
 *
 * \param[in] ctx PCO compiler context.
 * \param[in,out] nir NIR shader.
 * \param[in,out] data Shader data.
 */
void pco_lower_nir(pco_ctx *ctx, nir_shader *nir, pco_data *data)
{
   NIR_PASS(_,
            nir,
            nir_lower_explicit_io,
            nir_var_mem_ubo,
            spirv_options.ubo_addr_format);

   NIR_PASS(_, nir, pco_nir_lower_vk, &data->common);

   NIR_PASS(_,
            nir,
            nir_lower_io,
            nir_var_shader_in | nir_var_shader_out,
            glsl_type_size,
            nir_lower_io_lower_64bit_to_32);

   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_,
            nir,
            nir_io_add_const_offset_to_base,
            nir_var_shader_in | nir_var_shader_out);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, pco_nir_pfo, &data->fs);
   } else if (nir->info.stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_,
               nir,
               nir_lower_point_size,
               PVR_POINT_SIZE_RANGE_MIN,
               PVR_POINT_SIZE_RANGE_MAX);

      if (!nir->info.internal)
         NIR_PASS(_, nir, pco_nir_point_size);

      NIR_PASS(_, nir, pco_nir_pvi, &data->vs);
   }

   /* TODO: this should happen in the linking stage to cull unused I/O. */
   NIR_PASS(_,
            nir,
            nir_lower_io_to_scalar,
            nir_var_shader_in | nir_var_shader_out,
            NULL,
            NULL);

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop_vars);
   NIR_PASS(_, nir, nir_opt_dead_write_vars);
   NIR_PASS(_, nir, nir_opt_combine_stores, nir_var_all);

   bool progress;
   NIR_PASS(_, nir, nir_lower_alu);
   NIR_PASS(_, nir, nir_lower_pack);
   NIR_PASS(_, nir, nir_opt_algebraic);
   NIR_PASS(_, nir, pco_nir_lower_algebraic);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_algebraic);
   NIR_PASS(_, nir, pco_nir_lower_algebraic);
   NIR_PASS(_, nir, nir_opt_constant_folding);

   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_algebraic_late);
      NIR_PASS(progress, nir, pco_nir_lower_algebraic_late);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_lower_load_const_to_scalar);
      NIR_PASS(progress, nir, nir_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
   } while (progress);

   nir_variable_mode vec_modes = nir_var_shader_in;
   /* Fragment shader needs scalar writes after pfo. */
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      vec_modes |= nir_var_shader_out;

   NIR_PASS(_, nir, nir_opt_vectorize_io, vec_modes, false);

   /* Special case for frag coords:
    * - x,y come from (non-consecutive) special regs - always scalar.
    * - z,w are iterated and driver will make sure they're consecutive.
    *   - TODO: keep scalar for now, but add pass to vectorize.
    */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_,
               nir,
               nir_lower_io_to_scalar,
               nir_var_shader_in,
               frag_in_scalar_filter,
               nir);
   }

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_undef);
   } while (progress);

   if (pco_should_print_nir(nir)) {
      puts("after pco_lower_nir:");
      nir_print_shader(nir, stdout);
   }
}

/**
 * \brief Gather fragment shader data pass.
 *
 * \param[in] b NIR builder.
 * \param[in] intr NIR intrinsic instruction.
 * \param[in,out] cb_data Callback data.
 * \return True if the shader was modified (always return false).
 */
static bool gather_fs_data_pass(UNUSED struct nir_builder *b,
                                nir_intrinsic_instr *intr,
                                void *cb_data)
{
   /* Check whether the shader accesses z/w. */
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   struct nir_io_semantics io_semantics = nir_intrinsic_io_semantics(intr);
   if (io_semantics.location != VARYING_SLOT_POS)
      return false;

   unsigned component = nir_intrinsic_component(intr);
   unsigned chans = intr->def.num_components;
   assert(component == 2 || chans == 1);

   pco_data *data = cb_data;

   data->fs.uses.z |= (component == 2);
   data->fs.uses.w |= (component + chans > 3);

   return false;
}

/**
 * \brief Gathers fragment shader data.
 *
 * \param[in] nir NIR shader.
 * \param[in,out] data Shader data.
 */
static void gather_fs_data(nir_shader *nir, pco_data *data)
{
   nir_shader_intrinsics_pass(nir, gather_fs_data_pass, nir_metadata_all, data);

   /* If any inputs use smooth shading, then w is needed. */
   if (!data->fs.uses.w) {
      nir_foreach_shader_in_variable (var, nir) {
         if (var->data.interpolation > INTERP_MODE_SMOOTH)
            continue;

         data->fs.uses.w = true;
         break;
      }
   }
}

/**
 * \brief Gathers shader data.
 *
 * \param[in] nir NIR shader.
 * \param[in,out] data Shader data.
 */
static void gather_data(nir_shader *nir, pco_data *data)
{
   switch (nir->info.stage) {
   case MESA_SHADER_FRAGMENT:
      return gather_fs_data(nir, data);

   case MESA_SHADER_VERTEX:
      /* TODO */
      break;

   default:
      UNREACHABLE("");
   }
}

/**
 * \brief Runs post-processing passes on a NIR shader.
 *
 * \param[in] ctx PCO compiler context.
 * \param[in,out] nir NIR shader.
 * \param[in,out] data Shader data.
 */
void pco_postprocess_nir(pco_ctx *ctx, nir_shader *nir, pco_data *data)
{
   NIR_PASS(_, nir, nir_opt_sink, ~0U);
   NIR_PASS(_, nir, nir_opt_move, ~0U);

   NIR_PASS(_, nir, nir_move_vec_src_uses_to_dest, false);

   /* Re-index everything. */
   nir_foreach_function_with_impl (_, impl, nir) {
      nir_index_blocks(impl);
      nir_index_instrs(impl);
      nir_index_ssa_defs(impl);
   }

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   gather_data(nir, data);

   if (pco_should_print_nir(nir)) {
      puts("after pco_postprocess_nir:");
      nir_print_shader(nir, stdout);
   }
}

/**
 * \brief Performs linking optimizations on consecutive NIR shader stages.
 *
 * \param[in] ctx PCO compiler context.
 * \param[in,out] producer NIR producer shader.
 * \param[in,out] consumer NIR consumer shader.
 */
void pco_link_nir(pco_ctx *ctx, nir_shader *producer, nir_shader *consumer)
{
   /* TODO */
   puts("finishme: pco_link_nir");

   if (pco_should_print_nir(producer)) {
      puts("producer after pco_link_nir:");
      nir_print_shader(producer, stdout);
   }

   if (pco_should_print_nir(consumer)) {
      puts("consumer after pco_link_nir:");
      nir_print_shader(consumer, stdout);
   }
}

/**
 * \brief Checks whether two varying variables are the same.
 *
 * \param[in] out_var The first varying being compared.
 * \param[in] in_var The second varying being compared.
 * \return True if the varyings match.
 */
static bool varyings_match(nir_variable *out_var, nir_variable *in_var)
{
   return in_var->data.location == out_var->data.location &&
          in_var->data.location_frac == out_var->data.location_frac &&
          in_var->type == out_var->type;
}

/**
 * \brief Performs reverse linking optimizations on consecutive NIR shader
 * stages.
 *
 * \param[in] ctx PCO compiler context.
 * \param[in,out] producer NIR producer shader.
 * \param[in,out] consumer NIR consumer shader.
 */
void pco_rev_link_nir(pco_ctx *ctx, nir_shader *producer, nir_shader *consumer)
{
   /* TODO */
   puts("finishme: pco_rev_link_nir");

   /* Propagate back/adjust the interpolation qualifiers. */
   nir_foreach_shader_in_variable (in_var, consumer) {
      if (in_var->data.location == VARYING_SLOT_POS ||
          in_var->data.location == VARYING_SLOT_PNTC) {
         in_var->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
      } else if (in_var->data.interpolation == INTERP_MODE_NONE) {
         in_var->data.interpolation = INTERP_MODE_SMOOTH;
      }

      nir_foreach_shader_out_variable (out_var, producer) {
         if (!varyings_match(out_var, in_var))
            continue;

         out_var->data.interpolation = in_var->data.interpolation;
         break;
      }
   }

   if (pco_should_print_nir(producer)) {
      puts("producer after pco_rev_link_nir:");
      nir_print_shader(producer, stdout);
   }

   if (pco_should_print_nir(consumer)) {
      puts("consumer after pco_rev_link_nir:");
      nir_print_shader(consumer, stdout);
   }
}
