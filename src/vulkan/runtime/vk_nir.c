/*
 * Copyright © 2015 Intel Corporation
 * Copyright © 2022 Collabora, LTD
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

#include "vk_nir.h"

#include "compiler/nir/nir_xfb_info.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/nir_spirv.h"
#include "vk_device.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "vk_util.h"

#define SPIR_V_MAGIC_NUMBER 0x07230203

uint32_t
vk_spirv_version(const uint32_t *spirv_data, size_t spirv_size_B)
{
   assert(spirv_size_B >= 8);
   assert(spirv_data[0] == SPIR_V_MAGIC_NUMBER);
   return spirv_data[1];
}

static void
spirv_nir_debug(void *private_data,
                enum nir_spirv_debug_level level,
                size_t spirv_offset,
                const char *message)
{
   const struct vk_object_base *log_obj = private_data;

   switch (level) {
   case NIR_SPIRV_DEBUG_LEVEL_INFO:
      //vk_logi(VK_LOG_OBJS(log_obj), "SPIR-V offset %lu: %s",
      //        (unsigned long) spirv_offset, message);
      break;
   case NIR_SPIRV_DEBUG_LEVEL_WARNING:
      vk_logw(VK_LOG_OBJS(log_obj), "SPIR-V offset %lu: %s",
              (unsigned long) spirv_offset, message);
      break;
   case NIR_SPIRV_DEBUG_LEVEL_ERROR:
      vk_loge(VK_LOG_OBJS(log_obj), "SPIR-V offset %lu: %s",
              (unsigned long) spirv_offset, message);
      break;
   default:
      break;
   }
}

bool
nir_vk_is_not_xfb_output(nir_variable *var, void *data)
{
   if (var->data.mode != nir_var_shader_out)
      return true;

   /* From the Vulkan 1.3.259 spec:
    *
    *    VUID-StandaloneSpirv-Offset-04716
    *
    *    "Only variables or block members in the output interface decorated
    *    with Offset can be captured for transform feedback, and those
    *    variables or block members must also be decorated with XfbBuffer
    *    and XfbStride, or inherit XfbBuffer and XfbStride decorations from
    *    a block containing them"
    *
    * glslang generates gl_PerVertex builtins when they are not declared,
    * enabled XFB should not prevent them from being DCE'd.
    *
    * The logic should match nir_gather_xfb_info_with_varyings
    */

   if (!var->data.explicit_xfb_buffer)
      return true;

   bool is_array_block = var->interface_type != NULL &&
      glsl_type_is_array(var->type) &&
      glsl_without_array(var->type) == var->interface_type;

   if (!is_array_block) {
      return !var->data.explicit_offset;
   } else {
      /* For array of blocks we have to check each element */
      unsigned aoa_size = glsl_get_aoa_size(var->type);
      const struct glsl_type *itype = var->interface_type;
      unsigned nfields = glsl_get_length(itype);
      for (unsigned b = 0; b < aoa_size; b++) {
         for (unsigned f = 0; f < nfields; f++) {
            if (glsl_get_struct_field_offset(itype, f) >= 0)
               return false;
         }
      }

      return true;
   }
}

/* Input attachment loads are OpImageRead in SPIR-V. Extract the input
 * attachment index from the deref chain and variable, and turn it into an
 * intrinsic which contains the index directly.
 */
static bool
lower_input_attachment_load(nir_builder *b, nir_instr *instr,
                            void *cb_data)
{
   nir_deref_instr *deref;

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (intrin->intrinsic != nir_intrinsic_image_deref_load)
         return false;

      deref = nir_src_as_deref(intrin->src[0]);

      assert(glsl_type_is_image(deref->type));
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      const int texture_src_idx =
         nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (texture_src_idx < 0)
         return false;

      deref = nir_src_as_deref(tex->src[texture_src_idx].src);
   } else {
      return false;
   }

   enum glsl_sampler_dim image_dim = glsl_get_sampler_dim(deref->type);
   if (image_dim != GLSL_SAMPLER_DIM_SUBPASS &&
       image_dim != GLSL_SAMPLER_DIM_SUBPASS_MS)
      return false;

   b->cursor = nir_before_instr(instr);

   nir_def *index;
   nir_variable *var;
   unsigned range = 1;
   if (deref->deref_type == nir_deref_type_array) {
      ASSERTED nir_deref_instr *parent = nir_deref_instr_parent(deref);
      assert(parent->deref_type == nir_deref_type_var);
      var = parent->var;
      index = deref->arr.index.ssa;
      range = glsl_array_size(parent->type);
   } else {
      assert(deref->deref_type == nir_deref_type_var);
      var = deref->var;
      index = nir_imm_int(b, 0);
   }

   /* Depth vs. stencil input attachment is determined by the sampler
    * result type.
    */
   bool is_stencil =
      var->data.index == NIR_VARIABLE_NO_INDEX &&
      glsl_base_type_is_integer(glsl_get_sampler_result_type(var->type));
   bool is_depth =
      var->data.index == NIR_VARIABLE_NO_INDEX &&
      !glsl_base_type_is_integer(glsl_get_sampler_result_type(var->type));

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      nir_def *new_def;
      nir_def *offset = nir_trim_vector(b, intrin->src[1].ssa, 2);
      nir_def *sample = intrin->src[2].ssa;
      if (is_stencil) {
         new_def =
            nir_image_deref_stencil_input_attachment_load(b,
               intrin->def.num_components, intrin->def.bit_size,
               intrin->src[0].ssa, offset, sample,
               .image_dim = nir_intrinsic_image_dim(intrin),
               .image_array = nir_intrinsic_image_array(intrin),
               .access = nir_intrinsic_access(intrin),
               .dest_type = nir_intrinsic_dest_type(intrin));
      } else if (is_depth) {
         new_def =
            nir_image_deref_depth_input_attachment_load(b,
               intrin->def.num_components, intrin->def.bit_size,
               intrin->src[0].ssa, offset, sample,
               .image_dim = nir_intrinsic_image_dim(intrin),
               .image_array = nir_intrinsic_image_array(intrin),
               .access = nir_intrinsic_access(intrin),
               .dest_type = nir_intrinsic_dest_type(intrin));
      } else {
         new_def =
            nir_image_deref_input_attachment_load(b,
               intrin->def.num_components, intrin->def.bit_size,
               intrin->src[0].ssa, offset, sample, index,
               .image_dim = nir_intrinsic_image_dim(intrin),
               .image_array = nir_intrinsic_image_array(intrin),
               .access = nir_intrinsic_access(intrin),
               .dest_type = nir_intrinsic_dest_type(intrin),
               .base = var->data.index,
               .range = range);
      }

      nir_def_rewrite_uses(&intrin->def, new_def);
      nir_instr_remove(instr);
   } else {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      tex->input_attachment_depth = is_depth;
      tex->input_attachment_stencil = is_stencil;
      tex->input_attachment_index = !is_depth && !is_stencil;

      if (tex->input_attachment_index) {
         tex->texture_index = var->index;
         if (deref->deref_type == nir_deref_type_array) {
            tex->texture_array_size = range;
            nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, index);
         }
      }
   }
   return true;
}

bool
vk_nir_lower_input_attachment_loads(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_input_attachment_load,
                                       nir_metadata_control_flow, NULL);
}

nir_shader *
vk_spirv_to_nir(struct vk_device *device,
                const uint32_t *spirv_data, size_t spirv_size_B,
                mesa_shader_stage stage, const char *entrypoint_name,
                const VkSpecializationInfo *spec_info,
                const struct spirv_to_nir_options *spirv_options,
                const struct nir_shader_compiler_options *nir_options,
                bool internal,
                void *mem_ctx)
{
   assert(spirv_size_B >= 4 && spirv_size_B % 4 == 0);
   assert(spirv_data[0] == SPIR_V_MAGIC_NUMBER);

   const struct spirv_capabilities spirv_caps =
      vk_physical_device_get_spirv_capabilities(device->physical);

   struct spirv_to_nir_options spirv_options_local = *spirv_options;
   spirv_options_local.capabilities = &spirv_caps;
   spirv_options_local.debug.func = spirv_nir_debug;
   spirv_options_local.debug.private_data = (void *)device;

   spirv_options_local.sampler_descriptor_size =
      device->physical->properties.samplerDescriptorSize;
   spirv_options_local.sampler_descriptor_alignment =
      device->physical->properties.samplerDescriptorAlignment;
   spirv_options_local.image_descriptor_size =
      device->physical->properties.imageDescriptorSize;
   spirv_options_local.image_descriptor_alignment =
      device->physical->properties.imageDescriptorAlignment;
   spirv_options_local.buffer_descriptor_size =
      device->physical->properties.bufferDescriptorSize;
   spirv_options_local.buffer_descriptor_alignment =
      device->physical->properties.bufferDescriptorAlignment;

   struct nir_spirv_specialization *spec =
      vk_spec_info_to_nir_spirv(spec_info);

   nir_shader *nir = spirv_to_nir(spirv_data, spirv_size_B / 4,
                                  spec, stage, entrypoint_name,
                                  &spirv_options_local, nir_options);
   vtn_free_specialization(spec);

   if (nir == NULL)
      return NULL;

   assert(nir->info.stage == stage);
   nir_validate_shader(nir, "after spirv_to_nir");
   if (mem_ctx != NULL)
      ralloc_steal(mem_ctx, nir);

   nir->info.internal = internal;

   /* We have to lower away local constant initializers right before we
    * inline functions.  That way they get properly initialized at the top
    * of the function and not at the top of its caller.
    */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_deref);

   /* Pick off the single entrypoint that we want */
   nir_remove_non_cmat_call_entrypoints(nir);

   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, vk_nir_lower_input_attachment_loads);

   /* Now that we've deleted all but the main function, we can go ahead and
    * lower the rest of the constant initializers.  We do this here so that
    * nir_remove_dead_variables and split_per_member_structs below see the
    * corresponding stores.
    */
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);

   /* Split member structs.  We do this before lower_io_to_temporaries so that
    * it doesn't lower system values to temporaries by accident.
    */
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_per_member_structs);

   nir_remove_dead_variables_options dead_vars_opts = {
      .can_remove_var = nir_vk_is_not_xfb_output,
   };
   NIR_PASS(_, nir, nir_remove_dead_variables,
              nir_var_shader_in | nir_var_shader_out | nir_var_system_value |
              nir_var_shader_call_data_in | nir_var_ray_hit_attrib,
              &dead_vars_opts);

   /* This needs to happen after remove_dead_vars because GLSLang likes to
    * insert dead clip/cull vars and we don't want to clip/cull based on
    * uninitialized garbage.
    */
   nir_gather_clip_cull_distance_sizes_from_vars(nir);
   NIR_PASS(_, nir, nir_merge_clip_cull_distance_vars);

   if (nir->info.stage == MESA_SHADER_VERTEX ||
       nir->info.stage == MESA_SHADER_TESS_EVAL ||
       nir->info.stage == MESA_SHADER_GEOMETRY)
      nir_shader_gather_xfb_info(nir);

   NIR_PASS(_, nir, nir_propagate_invariant, false);

   return nir;
}
