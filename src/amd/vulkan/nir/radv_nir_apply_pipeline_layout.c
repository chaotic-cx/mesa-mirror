/*
 * Copyright © 2020 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */
#include "ac_descriptors.h"
#include "ac_shader_util.h"
#include "nir.h"
#include "nir_builder.h"
#include "radv_descriptor_set.h"
#include "radv_descriptors.h"
#include "radv_device.h"
#include "radv_nir.h"
#include "radv_physical_device.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "sid.h"

typedef struct {
   enum amd_gfx_level gfx_level;
   uint32_t address32_hi;
   bool disable_aniso_single_level;
   bool has_image_load_dcc_bug;
   bool disable_tg4_trunc_coord;

   const struct radv_shader_args *args;
   const struct radv_shader_info *info;
   const struct radv_shader_layout *layout;
} apply_layout_state;

static nir_def *
get_scalar_arg(nir_builder *b, unsigned size, struct ac_arg arg)
{
   assert(arg.used);
   return nir_load_scalar_arg_amd(b, size, .base = arg.arg_index);
}

static nir_def *
convert_pointer_to_64_bit(nir_builder *b, apply_layout_state *state, nir_def *ptr)
{
   return nir_pack_64_2x32_split(b, ptr, nir_imm_int(b, state->address32_hi));
}

static nir_def *
load_desc_ptr(nir_builder *b, apply_layout_state *state, unsigned set)
{
   const struct radv_userdata_locations *user_sgprs_locs = &state->info->user_sgprs_locs;
   if (user_sgprs_locs->shader_data[AC_UD_INDIRECT_DESCRIPTOR_SETS].sgpr_idx != -1) {
      nir_def *addr = get_scalar_arg(b, 1, state->args->descriptor_sets[0]);
      addr = convert_pointer_to_64_bit(b, state, addr);
      return nir_load_smem_amd(b, 1, addr, nir_imm_int(b, set * 4));
   }

   assert(state->args->descriptor_sets[set].used);
   return get_scalar_arg(b, 1, state->args->descriptor_sets[set]);
}

static void
visit_vulkan_resource_index(nir_builder *b, apply_layout_state *state, nir_intrinsic_instr *intrin)
{
   unsigned desc_set = nir_intrinsic_desc_set(intrin);
   unsigned binding = nir_intrinsic_binding(intrin);
   struct radv_descriptor_set_layout *layout = state->layout->set[desc_set].layout;
   unsigned offset = layout->binding[binding].offset;
   unsigned stride;

   nir_def *set_ptr;
   if (vk_descriptor_type_is_dynamic(layout->binding[binding].type)) {
      unsigned idx = state->layout->set[desc_set].dynamic_offset_start + layout->binding[binding].dynamic_offset_offset;
      set_ptr = get_scalar_arg(b, 1, state->args->ac.push_constants);
      offset = state->layout->push_constant_size + idx * 16;
      stride = 16;
   } else {
      set_ptr = load_desc_ptr(b, state, desc_set);
      stride = layout->binding[binding].size;
   }

   nir_def *binding_ptr = nir_imul_imm(b, intrin->src[0].ssa, stride);
   nir_def_as_alu(binding_ptr)->no_unsigned_wrap = true;

   binding_ptr = nir_iadd_imm(b, binding_ptr, offset);
   nir_def_as_alu(binding_ptr)->no_unsigned_wrap = true;

   if (layout->binding[binding].type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
      assert(stride == 16);
      nir_def_rewrite_uses(&intrin->def, nir_pack_64_2x32_split(b, set_ptr, binding_ptr));
   } else {
      nir_def_rewrite_uses(&intrin->def, nir_vec3(b, set_ptr, binding_ptr, nir_imm_int(b, stride)));
   }
   nir_instr_remove(&intrin->instr);
}

static void
visit_vulkan_resource_reindex(nir_builder *b, apply_layout_state *state, nir_intrinsic_instr *intrin)
{
   VkDescriptorType desc_type = nir_intrinsic_desc_type(intrin);
   if (desc_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
      nir_def *set_ptr = nir_unpack_64_2x32_split_x(b, intrin->src[0].ssa);
      nir_def *binding_ptr = nir_unpack_64_2x32_split_y(b, intrin->src[0].ssa);

      nir_def *index = nir_imul_imm(b, intrin->src[1].ssa, 16);
      nir_def_as_alu(index)->no_unsigned_wrap = true;

      binding_ptr = nir_iadd_nuw(b, binding_ptr, index);

      nir_def_rewrite_uses(&intrin->def, nir_pack_64_2x32_split(b, set_ptr, binding_ptr));
   } else {
      assert(desc_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || desc_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

      nir_def *binding_ptr = nir_channel(b, intrin->src[0].ssa, 1);
      nir_def *stride = nir_channel(b, intrin->src[0].ssa, 2);

      nir_def *index = nir_imul(b, intrin->src[1].ssa, stride);
      nir_def_as_alu(index)->no_unsigned_wrap = true;

      binding_ptr = nir_iadd_nuw(b, binding_ptr, index);

      nir_def_rewrite_uses(&intrin->def, nir_vector_insert_imm(b, intrin->src[0].ssa, binding_ptr, 1));
   }
   nir_instr_remove(&intrin->instr);
}

static void
visit_load_vulkan_descriptor(nir_builder *b, apply_layout_state *state, nir_intrinsic_instr *intrin)
{
   if (nir_intrinsic_desc_type(intrin) == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
      nir_def *addr = convert_pointer_to_64_bit(b, state,
                                                nir_iadd(b, nir_unpack_64_2x32_split_x(b, intrin->src[0].ssa),
                                                         nir_unpack_64_2x32_split_y(b, intrin->src[0].ssa)));
      nir_def *desc = nir_build_load_global(b, 1, 64, addr, .access = ACCESS_NON_WRITEABLE);

      nir_def_rewrite_uses(&intrin->def, desc);
   } else {
      nir_def_rewrite_uses(&intrin->def, nir_vector_insert_imm(b, intrin->src[0].ssa, nir_imm_int(b, 0), 2));
   }
   nir_instr_remove(&intrin->instr);
}

static nir_def *
load_inline_buffer_descriptor(nir_builder *b, apply_layout_state *state, nir_def *rsrc)
{
   uint32_t desc[4];

   ac_build_raw_buffer_descriptor(state->gfx_level, (uint64_t)state->address32_hi << 32, 0xffffffff, desc);

   return nir_vec4(b, rsrc, nir_imm_int(b, desc[1]), nir_imm_int(b, desc[2]), nir_imm_int(b, desc[3]));
}

static nir_def *
load_buffer_descriptor(nir_builder *b, apply_layout_state *state, nir_def *rsrc, unsigned access)
{
   nir_binding binding = nir_chase_binding(nir_src_for_ssa(rsrc));

   /* If binding.success=false, then this is a variable pointer, which we don't support with
    * VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK.
    */
   if (binding.success) {
      struct radv_descriptor_set_layout *layout = state->layout->set[binding.desc_set].layout;
      if (layout->binding[binding.binding].type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
         rsrc = nir_iadd(b, nir_channel(b, rsrc, 0), nir_channel(b, rsrc, 1));
         return load_inline_buffer_descriptor(b, state, rsrc);
      }
   }

   if (access & ACCESS_NON_UNIFORM)
      return nir_iadd(b, nir_channel(b, rsrc, 0), nir_channel(b, rsrc, 1));

   nir_def *desc_set = convert_pointer_to_64_bit(b, state, nir_channel(b, rsrc, 0));
   return nir_load_smem_amd(b, 4, desc_set, nir_channel(b, rsrc, 1), .align_mul = 16);
}

static void
visit_get_ssbo_size(nir_builder *b, apply_layout_state *state, nir_intrinsic_instr *intrin)
{
   nir_def *rsrc = intrin->src[0].ssa;

   nir_def *size;
   if (nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM) {
      nir_def *ptr = nir_iadd(b, nir_channel(b, rsrc, 0), nir_channel(b, rsrc, 1));
      ptr = nir_iadd_imm(b, ptr, 8);
      ptr = convert_pointer_to_64_bit(b, state, ptr);
      size = nir_build_load_global(b, 4, 32, ptr, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER, .align_mul = 16,
                                   .align_offset = 4);
   } else {
      /* load the entire descriptor so it can be CSE'd */
      nir_def *ptr = convert_pointer_to_64_bit(b, state, nir_channel(b, rsrc, 0));
      nir_def *desc = nir_load_smem_amd(b, 4, ptr, nir_channel(b, rsrc, 1), .align_mul = 16);
      size = nir_channel(b, desc, 2);
   }

   nir_def_replace(&intrin->def, size);
}

static nir_def *
get_sampler_desc(nir_builder *b, apply_layout_state *state, nir_deref_instr *deref, enum ac_descriptor_type desc_type,
                 bool non_uniform, nir_tex_instr *tex, bool write)
{
   nir_variable *var = nir_deref_instr_get_variable(deref);
   assert(var);
   unsigned desc_set = var->data.descriptor_set;
   unsigned binding_index = var->data.binding;
   bool indirect = nir_deref_instr_has_indirect(deref);

   struct radv_descriptor_set_layout *layout = state->layout->set[desc_set].layout;
   struct radv_descriptor_set_binding_layout *binding = &layout->binding[binding_index];

   /* Handle immutable and embedded (compile-time) samplers
    * (VkDescriptorSetLayoutBinding::pImmutableSamplers) We can only do this for constant array
    * index. Note that indexing is forbidden with embedded samplers.
    */
   if (desc_type == AC_DESC_SAMPLER && binding->immutable_samplers_offset && !indirect) {
      unsigned constant_index = 0;

      while (deref->deref_type != nir_deref_type_var) {
         assert(deref->deref_type == nir_deref_type_array);
         unsigned array_size = MAX2(glsl_get_aoa_size(deref->type), 1);
         constant_index += nir_src_as_uint(deref->arr.index) * array_size;
         deref = nir_deref_instr_parent(deref);
      }

      uint32_t dword0_mask =
         tex->op == nir_texop_tg4 && state->disable_tg4_trunc_coord ? C_008F30_TRUNC_COORD : 0xffffffffu;
      const uint32_t *samplers = radv_immutable_samplers(layout, binding);
      return nir_imm_ivec4(b, samplers[constant_index * 4 + 0] & dword0_mask, samplers[constant_index * 4 + 1],
                           samplers[constant_index * 4 + 2], samplers[constant_index * 4 + 3]);
   }

   unsigned size = 8;
   unsigned offset = binding->offset;
   switch (desc_type) {
   case AC_DESC_IMAGE:
   case AC_DESC_PLANE_0:
      break;
   case AC_DESC_FMASK:
      offset += 32;
      break;
   case AC_DESC_PLANE_1:
      offset += RADV_COMBINED_IMAGE_SAMPLER_DESC_SIZE;
      break;
   case AC_DESC_SAMPLER:
      size = RADV_SAMPLER_DESC_SIZE / 4;
      if (binding->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         offset += RADV_COMBINED_IMAGE_SAMPLER_DESC_SAMPLER_OFFSET;
      break;
   case AC_DESC_BUFFER:
      size = RADV_BUFFER_DESC_SIZE / 4;
      break;
   case AC_DESC_PLANE_2:
      offset += 2 * RADV_COMBINED_IMAGE_SAMPLER_DESC_SIZE;
      break;
   }

   nir_def *index = NULL;
   while (deref->deref_type != nir_deref_type_var) {
      assert(deref->deref_type == nir_deref_type_array);
      unsigned array_size = MAX2(glsl_get_aoa_size(deref->type), 1);
      array_size *= binding->size;

      nir_def *tmp = nir_imul_imm(b, deref->arr.index.ssa, array_size);
      if (tmp != deref->arr.index.ssa)
         nir_def_as_alu(tmp)->no_unsigned_wrap = true;

      if (index) {
         index = nir_iadd(b, tmp, index);
         nir_def_as_alu(index)->no_unsigned_wrap = true;
      } else {
         index = tmp;
      }

      deref = nir_deref_instr_parent(deref);
   }

   nir_def *index_offset = index ? nir_iadd_imm(b, index, offset) : nir_imm_int(b, offset);
   if (index && index_offset != index)
      nir_def_as_alu(index_offset)->no_unsigned_wrap = true;

   if (non_uniform)
      return nir_iadd(b, load_desc_ptr(b, state, desc_set), index_offset);

   nir_def *addr = convert_pointer_to_64_bit(b, state, load_desc_ptr(b, state, desc_set));
   nir_def *desc = nir_load_smem_amd(b, size, addr, index_offset, .align_mul = size * 4u);

   if (desc_type == AC_DESC_IMAGE && state->has_image_load_dcc_bug && !tex && !write) {
      nir_def *comp[8];
      for (unsigned i = 0; i < 8; i++)
         comp[i] = nir_channel(b, desc, i);

      /* WRITE_COMPRESS_ENABLE must be 0 for all image loads to workaround a
       * hardware bug.
       */
      comp[6] = nir_iand_imm(b, comp[6], C_00A018_WRITE_COMPRESS_ENABLE);

      return nir_vec(b, comp, 8);
   } else if (desc_type == AC_DESC_SAMPLER && tex->op == nir_texop_tg4 && state->disable_tg4_trunc_coord) {
      nir_def *comp[4];
      for (unsigned i = 0; i < 4; i++)
         comp[i] = nir_channel(b, desc, i);

      /* We want to always use the linear filtering truncation behaviour for
       * nir_texop_tg4, even if the sampler uses nearest/point filtering.
       */
      comp[0] = nir_iand_imm(b, comp[0], C_008F30_TRUNC_COORD);

      return nir_vec(b, comp, 4);
   }

   return desc;
}

static void
update_image_intrinsic(nir_builder *b, apply_layout_state *state, nir_intrinsic_instr *intrin)
{
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   const enum glsl_sampler_dim dim = glsl_get_sampler_dim(deref->type);
   bool is_load =
      intrin->intrinsic == nir_intrinsic_image_deref_load || intrin->intrinsic == nir_intrinsic_image_deref_sparse_load;

   nir_def *desc = get_sampler_desc(b, state, deref, dim == GLSL_SAMPLER_DIM_BUF ? AC_DESC_BUFFER : AC_DESC_IMAGE,
                                    nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM, NULL, !is_load);

   if (intrin->intrinsic == nir_intrinsic_image_deref_descriptor_amd) {
      nir_def_replace(&intrin->def, desc);
   } else {
      nir_rewrite_image_intrinsic(intrin, desc, true);
   }
}

static bool
can_increase_load_size(nir_intrinsic_instr *intrin, unsigned offset, unsigned old, unsigned new)
{
   /* Only increase the size of loads if doing so won't extend into a new page/cache-line. */
   unsigned align_mul = MIN2(nir_intrinsic_align_mul(intrin), 64u);
   unsigned end = (nir_intrinsic_align_offset(intrin) + offset + old) & (align_mul - 1);
   return (new - old) <= (align_mul - end);
}

static nir_def *
load_push_constant(nir_builder *b, apply_layout_state *state, nir_intrinsic_instr *intrin)
{
   unsigned base = nir_intrinsic_base(intrin);
   unsigned bit_size = intrin->def.bit_size;
   unsigned count = intrin->def.num_components * (bit_size / 32u);
   assert(bit_size >= 32);

   nir_def *addr = NULL;
   nir_def *offset = NULL;
   unsigned const_offset = -1;
   if (nir_src_is_const(intrin->src[0]))
      const_offset = (base + nir_src_as_uint(intrin->src[0])) / 4u;

   const unsigned max_push_constant = sizeof(state->args->ac.inline_push_const_mask) * 8u;

   nir_component_mask_t comps_read = nir_def_components_read(&intrin->def);

   nir_def *data[NIR_MAX_VEC_COMPONENTS * 2];
   unsigned num_loads = 0;
   for (unsigned start = 0; start < count;) {
      if (!(comps_read & BITFIELD64_BIT(start >> (bit_size == 64 ? 1 : 0)))) {
         data[num_loads++] = nir_undef(b, 1, 32);
         start += 1;
         continue;
      }

      /* Try to use inline push constants when possible. */
      unsigned inline_idx = const_offset + start;
      if (const_offset != -1 && inline_idx < max_push_constant &&
          (state->args->ac.inline_push_const_mask & BITFIELD64_BIT(inline_idx))) {
         inline_idx = util_bitcount64(state->args->ac.inline_push_const_mask & BITFIELD64_MASK(inline_idx));
         data[num_loads++] = get_scalar_arg(b, 1, state->args->ac.inline_push_consts[inline_idx]);
         start += 1;
         continue;
      }

      if (!state->args->ac.push_constants.used) {
         /* Assume this is an inlined push constant load which was expanded to include dwords which are not inlined. */
         assert(const_offset != -1);
         data[num_loads++] = nir_undef(b, 1, 32);
         start += 1;
         continue;
      }

      if (!offset) {
         addr = get_scalar_arg(b, 1, state->args->ac.push_constants);
         addr = convert_pointer_to_64_bit(b, state, addr);
         offset = nir_iadd_imm_nuw(b, intrin->src[0].ssa, base);
      }

      /* Decrease to supported size. */
      unsigned size = count - start;
      size = state->gfx_level >= GFX12 && size == 3 ? 3 : (1 << (util_last_bit(size) - 1));

      /* Try to round up to power of two instead. */
      if (size < (count - start) && can_increase_load_size(intrin, start * 4, size, size * 2))
         size *= 2;

      data[num_loads++] = nir_load_smem_amd(b, size, addr, nir_iadd_imm_nuw(b, offset, start * 4));
      start += size;
   }
   return nir_extract_bits(b, data, num_loads, 0, intrin->def.num_components, bit_size);
}

static bool
apply_layout_to_intrin(nir_builder *b, apply_layout_state *state, nir_intrinsic_instr *intrin)
{
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *rsrc;
   switch (intrin->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      visit_vulkan_resource_index(b, state, intrin);
      break;
   case nir_intrinsic_vulkan_resource_reindex:
      visit_vulkan_resource_reindex(b, state, intrin);
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      visit_load_vulkan_descriptor(b, state, intrin);
      break;
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      rsrc = load_buffer_descriptor(b, state, intrin->src[0].ssa, nir_intrinsic_access(intrin));
      nir_src_rewrite(&intrin->src[0], rsrc);
      break;
   case nir_intrinsic_store_ssbo:
      rsrc = load_buffer_descriptor(b, state, intrin->src[1].ssa, nir_intrinsic_access(intrin));
      nir_src_rewrite(&intrin->src[1], rsrc);
      break;
   case nir_intrinsic_get_ssbo_size:
      visit_get_ssbo_size(b, state, intrin);
      break;
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_descriptor_amd:
      update_image_intrinsic(b, state, intrin);
      break;
   case nir_intrinsic_load_push_constant: {
      nir_def_replace(&intrin->def, load_push_constant(b, state, intrin));
      break;
   }
   default:
      return false;
   }

   return true;
}

static bool
apply_layout_to_tex(nir_builder *b, apply_layout_state *state, nir_tex_instr *tex)
{
   b->cursor = nir_before_instr(&tex->instr);

   nir_deref_instr *texture_deref_instr = NULL;
   nir_deref_instr *sampler_deref_instr = NULL;
   int plane = -1;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_deref:
         texture_deref_instr = nir_src_as_deref(tex->src[i].src);
         break;
      case nir_tex_src_sampler_deref:
         sampler_deref_instr = nir_src_as_deref(tex->src[i].src);
         break;
      case nir_tex_src_plane:
         plane = nir_src_as_int(tex->src[i].src);
         break;
      default:
         break;
      }
   }

   nir_def *image = NULL;
   nir_def *sampler = NULL;
   if (plane >= 0) {
      assert(tex->op != nir_texop_txf_ms && tex->op != nir_texop_samples_identical);
      assert(tex->sampler_dim != GLSL_SAMPLER_DIM_BUF);
      image =
         get_sampler_desc(b, state, texture_deref_instr, AC_DESC_PLANE_0 + plane, tex->texture_non_uniform, tex, false);
   } else if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
      image = get_sampler_desc(b, state, texture_deref_instr, AC_DESC_BUFFER, tex->texture_non_uniform, tex, false);
   } else if (tex->op == nir_texop_fragment_mask_fetch_amd) {
      image = get_sampler_desc(b, state, texture_deref_instr, AC_DESC_FMASK, tex->texture_non_uniform, tex, false);
   } else {
      image = get_sampler_desc(b, state, texture_deref_instr, AC_DESC_IMAGE, tex->texture_non_uniform, tex, false);
   }

   if (sampler_deref_instr) {
      sampler = get_sampler_desc(b, state, sampler_deref_instr, AC_DESC_SAMPLER, tex->sampler_non_uniform, tex, false);

      if (state->disable_aniso_single_level && tex->sampler_dim < GLSL_SAMPLER_DIM_RECT && state->gfx_level < GFX8) {
         /* Disable anisotropic filtering if BASE_LEVEL == LAST_LEVEL.
          *
          * GFX6-GFX7:
          *   If BASE_LEVEL == LAST_LEVEL, the shader must disable anisotropic
          *   filtering manually. The driver sets img7 to a mask clearing
          *   MAX_ANISO_RATIO if BASE_LEVEL == LAST_LEVEL. The shader must do:
          *     s_and_b32 samp0, samp0, img7
          *
          * GFX8:
          *   The ANISO_OVERRIDE sampler field enables this fix in TA.
          */
         /* TODO: This is unnecessary for combined image+sampler.
          * We can do this when updating the desc set. */
         nir_def *comp[4];
         for (unsigned i = 0; i < 4; i++)
            comp[i] = nir_channel(b, sampler, i);
         comp[0] = nir_iand(b, comp[0], nir_channel(b, image, 7));

         sampler = nir_vec(b, comp, 4);
      }
   }

   if (tex->op == nir_texop_descriptor_amd) {
      nir_def_replace(&tex->def, image);
      return true;
   }

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_deref:
         tex->src[i].src_type = nir_tex_src_texture_handle;
         nir_src_rewrite(&tex->src[i].src, image);
         break;
      case nir_tex_src_sampler_deref:
         tex->src[i].src_type = nir_tex_src_sampler_handle;
         nir_src_rewrite(&tex->src[i].src, sampler);
         break;
      default:
         break;
      }
   }

   return true;
}

bool
radv_nir_apply_pipeline_layout(nir_shader *shader, struct radv_device *device, const struct radv_shader_stage *stage)
{
   bool progress = false;
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   apply_layout_state state = {
      .gfx_level = pdev->info.gfx_level,
      .address32_hi = pdev->info.address32_hi,
      .disable_aniso_single_level = instance->drirc.disable_aniso_single_level,
      .has_image_load_dcc_bug = pdev->info.has_image_load_dcc_bug,
      .disable_tg4_trunc_coord = !pdev->info.conformant_trunc_coord && !instance->drirc.disable_trunc_coord,
      .args = &stage->args,
      .info = &stage->info,
      .layout = &stage->layout,
   };

   nir_foreach_function_impl (impl, shader) {
      bool impl_progress = false;
      nir_builder b = nir_builder_create(impl);

      /* Iterate in reverse so load_ubo lowering can look at
       * the vulkan_resource_index to tell if it's an inline
       * ubo.
       */
      nir_foreach_block_reverse (block, impl) {
         nir_foreach_instr_reverse_safe (instr, block) {
            if (instr->type == nir_instr_type_tex)
               impl_progress |= apply_layout_to_tex(&b, &state, nir_instr_as_tex(instr));
            else if (instr->type == nir_instr_type_intrinsic)
               impl_progress |= apply_layout_to_intrin(&b, &state, nir_instr_as_intrinsic(instr));
         }
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}
