/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates.
 * Copyright (C) 2019-2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "pan_nir.h"
#include "panfrost/model/pan_model.h"

static enum pipe_format
varying_format(nir_alu_type t, unsigned ncomps)
{
   assert(ncomps >= 1 && ncomps <= 4);

#define VARYING_FORMAT(ntype, nsz, ptype, psz)                                 \
   {                                                                           \
      .type = nir_type_##ntype##nsz, .formats = {                              \
         PIPE_FORMAT_R##psz##_##ptype,                                         \
         PIPE_FORMAT_R##psz##G##psz##_##ptype,                                 \
         PIPE_FORMAT_R##psz##G##psz##B##psz##_##ptype,                         \
         PIPE_FORMAT_R##psz##G##psz##B##psz##A##psz##_##ptype,                 \
      }                                                                        \
   }

   static const struct {
      nir_alu_type type;
      enum pipe_format formats[4];
   } conv[] = {
      VARYING_FORMAT(float, 32, FLOAT, 32),
      VARYING_FORMAT(uint, 32, UINT, 32),
      VARYING_FORMAT(float, 16, FLOAT, 16),
      VARYING_FORMAT(uint, 16, UINT, 16),
   };
#undef VARYING_FORMAT

   assert(ncomps > 0 && ncomps <= ARRAY_SIZE(conv[0].formats));

   for (unsigned i = 0; i < ARRAY_SIZE(conv); i++) {
      if (conv[i].type == t)
         return conv[i].formats[ncomps - 1];
   }

   UNREACHABLE("Invalid type");
}

struct slot_info {
   nir_alu_type type;
   bool any_highp;
   unsigned count;
   unsigned index;
};

struct walk_varyings_data {
   enum pan_mediump_vary mediump;
   bool quirk_no_auto32;
   struct slot_info *slots;
};

static bool
walk_varyings(UNUSED nir_builder *b, nir_instr *instr, void *data)
{
   struct walk_varyings_data *wv_data = data;
   struct slot_info *slots = wv_data->slots;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   unsigned count;
   unsigned size;

   /* Only consider intrinsics that access varyings */
   switch (intr->intrinsic) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_view_output:
      if (b->shader->info.stage != MESA_SHADER_VERTEX)
         return false;

      count = nir_src_num_components(intr->src[0]);
      size = nir_alu_type_get_type_size(nir_intrinsic_src_type(intr));
      break;

   case nir_intrinsic_load_input:
   case nir_intrinsic_load_interpolated_input:
      if (b->shader->info.stage != MESA_SHADER_FRAGMENT)
         return false;

      count = intr->def.num_components;
      size = intr->def.bit_size;
      break;

   default:
      return false;
   }

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);

   if (sem.no_varying)
      return false;

   /* In a fragment shader, flat shading is lowered to load_input but
    * interpolation is lowered to load_interpolated_input, so we can check
    * the intrinsic to distinguish.
    *
    * In a vertex shader, we consider everything flat, as the information
    * will not contribute to the final linked varyings -- flatness is used
    * only to determine the type, and the GL linker uses the type from the
    * fragment shader instead.
    */
   bool flat = intr->intrinsic != nir_intrinsic_load_interpolated_input;
   bool auto32 = !wv_data->quirk_no_auto32 && size == 32;
   nir_alu_type type = (flat && auto32) ? nir_type_uint : nir_type_float;

   if (sem.medium_precision) {
      /* Demote interpolated float varyings to fp16 where possible. We do not
       * demote flat varyings, including integer varyings, due to various
       * issues with the Midgard hardware behaviour and TGSI shaders, as well
       * as having no demonstrable benefit in practice.
       */
      if (wv_data->mediump == PAN_MEDIUMP_VARY_SMOOTH_16BIT)
         size = type == nir_type_float ? 16 : 32;

      if (wv_data->mediump == PAN_MEDIUMP_VARY_32BIT)
         size = 32;
   }

   assert(size == 32 || size == 16);
   type |= size;

   /* Count currently contains the number of components accessed by this
    * intrinsics. However, we may be accessing a fractional location,
    * indicating by the NIR component. Add that in. The final value be the
    * maximum (component + count), an upper bound on the number of
    * components possibly used.
    */
   count += nir_intrinsic_component(intr);

   /* Consider each slot separately */
   for (unsigned offset = 0; offset < sem.num_slots; ++offset) {
      unsigned location = sem.location + offset;
      unsigned index = pan_res_handle_get_index(nir_intrinsic_base(intr)) + offset;

      if (slots[location].type) {
         assert(slots[location].type == type);
         assert(slots[location].index == index);
      } else {
         slots[location].type = type;
         slots[location].index = index;
      }

      if (size == 32 && !sem.medium_precision)
         slots[location].any_highp = true;

      slots[location].count = MAX2(slots[location].count, count);
   }

   return false;
}

static bool
collect_noperspective_varyings_fs(UNUSED nir_builder *b,
                                  nir_intrinsic_instr *intr,
                                  void *data)
{
   uint32_t *noperspective_varyings = data;

   if (intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location < VARYING_SLOT_VAR0)
      return false;

   nir_intrinsic_instr *bary_instr = nir_src_as_intrinsic(intr->src[0]);
   assert(bary_instr);
   if (nir_intrinsic_interp_mode(bary_instr) == INTERP_MODE_NOPERSPECTIVE) {
      unsigned loc = sem.location - VARYING_SLOT_VAR0;
      *noperspective_varyings |= BITFIELD_RANGE(loc, sem.num_slots);
   }

   return false;
}

uint32_t
pan_nir_collect_noperspective_varyings_fs(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   uint32_t noperspective_varyings = 0;

   /* Collect from variables */
   nir_foreach_shader_in_variable(var, s) {
      if (var->data.location < VARYING_SLOT_VAR0)
         continue;

      if (var->data.interpolation != INTERP_MODE_NOPERSPECTIVE)
         continue;

      unsigned loc = var->data.location - VARYING_SLOT_VAR0;
      unsigned slots = glsl_count_attribute_slots(var->type, false);
      noperspective_varyings |= BITFIELD_RANGE(loc, slots);
   }

   /* And collect from load_interpolated_input intrinsics */
   nir_shader_intrinsics_pass(s, collect_noperspective_varyings_fs,
                              nir_metadata_all,
                              (void *)&noperspective_varyings);

   return noperspective_varyings;
}

void
pan_nir_collect_varyings(nir_shader *s, struct pan_shader_info *info,
                         enum pan_mediump_vary mediump)
{
   if (s->info.stage != MESA_SHADER_VERTEX &&
       s->info.stage != MESA_SHADER_FRAGMENT)
      return;

   struct slot_info slots[64] = {0};
   struct walk_varyings_data wv_data = {mediump, info->quirk_no_auto32, slots};
   nir_shader_instructions_pass(s, walk_varyings, nir_metadata_all, &wv_data);

   struct pan_shader_varying *varyings = (s->info.stage == MESA_SHADER_VERTEX)
                                            ? info->varyings.output
                                            : info->varyings.input;

   unsigned count = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(slots); ++i) {
      if (!slots[i].type)
         continue;

      enum pipe_format format = varying_format(slots[i].type, slots[i].count);
      assert(format != PIPE_FORMAT_NONE);

      unsigned index = slots[i].index;
      count = MAX2(count, index + 1);

      varyings[index].location = i;
      varyings[index].format = format;
   }

   if (s->info.stage == MESA_SHADER_VERTEX)
      info->varyings.output_count = count;
   else
      info->varyings.input_count = count;

   if (s->info.stage == MESA_SHADER_FRAGMENT)
      info->varyings.noperspective =
         pan_nir_collect_noperspective_varyings_fs(s);
}

/*
 * ABI: Special (desktop GL) slots come first, tightly packed. General varyings
 * come later, sparsely packed. This handles both linked and separable shaders
 * with a common code path, with minimal keying only for desktop GL. Each slot
 * consumes 16 bytes (TODO: fp16, partial vectors).
 */
static unsigned
bi_varying_base_bytes(gl_varying_slot slot, uint32_t fixed_varyings)
{
   if (slot >= VARYING_SLOT_VAR0) {
      unsigned nr_special = util_bitcount(fixed_varyings);
      unsigned general_index = (slot - VARYING_SLOT_VAR0);

      return 16 * (nr_special + general_index);
   } else {
      return 16 * (util_bitcount(fixed_varyings & BITFIELD_MASK(slot)));
   }
}

static const struct pan_varying_slot hw_varying_slots[] = {{
   .location = VARYING_SLOT_POS,
   .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
   .section = PAN_VARYING_SECTION_POSITION,
   .offset = 0,
}, {
   .location = VARYING_SLOT_PSIZ,
   .format = PIPE_FORMAT_R16_FLOAT,
   .section = PAN_VARYING_SECTION_ATTRIBS,
   .offset = 0,
}, {
   .location = VARYING_SLOT_LAYER,
   .format = PIPE_FORMAT_R8_UINT,
   .section = PAN_VARYING_SECTION_ATTRIBS,
   .offset = 2,
}, {
   .location = VARYING_SLOT_VIEWPORT,
   .format = PIPE_FORMAT_R8_UINT,
   .section = PAN_VARYING_SECTION_ATTRIBS,
   .offset = 2,
}, {
   .location = VARYING_SLOT_PRIMITIVE_ID,
   .format = PIPE_FORMAT_R32_UINT,
   .section = PAN_VARYING_SECTION_ATTRIBS,
   .offset = 12,
}};

static struct pan_varying_slot
hw_varying_slot(gl_varying_slot slot)
{
   for (unsigned i = 0; i < ARRAY_SIZE(hw_varying_slots); i++) {
      if (hw_varying_slots[i].location == slot)
         return hw_varying_slots[i];
   }
   UNREACHABLE("Invalid HW varying slot");
}

void
pan_build_varying_layout_sso_abi(struct pan_varying_layout *layout,
                                 nir_shader *nir, unsigned gpu_id,
                                 uint32_t fixed_varyings)
{
   /* TODO: Midgard */
   assert(pan_arch(gpu_id) >= 6);

   struct slot_info slots[64] = {0};
   struct walk_varyings_data wv_data = {PAN_MEDIUMP_VARY_32BIT, false, slots};
   nir_shader_instructions_pass(nir, walk_varyings, nir_metadata_all, &wv_data);

   memset(layout, 0, sizeof(*layout));

   unsigned generic_size_B = 0, count = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(slots); i++) {
      if (!slots[i].type)
         continue;

      /* It's possible that something has been dead code eliminated between
       * when the driver locations were set on variables and here.  Don't
       * trust our compaction to match the driver.  Just copy over the index
       * and accept that there's a hole in the mapping.
       */
      unsigned idx = slots[i].index;
      count = MAX2(count, idx + 1);
      assert(count <= ARRAY_SIZE(layout->slots));
      assert(layout->slots[idx].format == PIPE_FORMAT_NONE);

      if (BITFIELD64_BIT(i) & (VARYING_BIT_POS | PAN_ATTRIB_VARYING_BITS)) {
         layout->slots[idx] = hw_varying_slot(i);
      } else {
         unsigned offset = bi_varying_base_bytes(i, fixed_varyings);
         assert(offset < (1 << 11));

         const enum pipe_format format =
            varying_format(slots[i].type, slots[i].count);
         const unsigned size = util_format_get_blocksize(format);
         generic_size_B = MAX2(generic_size_B, offset + size);

         layout->slots[idx] = (struct pan_varying_slot) {
            .location = i,
            .format = format,
            .section = PAN_VARYING_SECTION_GENERIC,
            .offset = offset,
         };
      }
   }
   layout->count = count;
   layout->generic_size_B = generic_size_B;
}

void
pan_build_varying_layout_compact(struct pan_varying_layout *layout,
                                 nir_shader *nir, unsigned gpu_id)
{
   /* TODO: Midgard */
   assert(pan_arch(gpu_id) >= 6);

   struct slot_info slots[64] = {0};
   struct walk_varyings_data wv_data = {PAN_MEDIUMP_VARY_32BIT, false, slots};
   nir_shader_instructions_pass(nir, walk_varyings, nir_metadata_all, &wv_data);

   memset(layout, 0, sizeof(*layout));

   unsigned generic_size_B = 0, count = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(slots); i++) {
      if (!slots[i].type)
         continue;

      /* It's possible that something has been dead code eliminated between
       * when the driver locations were set on variables and here.  Don't
       * trust our compaction to match the driver.  Just copy over the index
       * and accept that there's a hole in the mapping.
       */
      unsigned idx = slots[i].index;
      count = MAX2(count, idx + 1);
      assert(count <= ARRAY_SIZE(layout->slots));
      assert(layout->slots[idx].format == PIPE_FORMAT_NONE);

      if (BITFIELD64_BIT(i) & (VARYING_BIT_POS | PAN_ATTRIB_VARYING_BITS)) {
         layout->slots[idx] = hw_varying_slot(i);
      } else {
         /* The Vulkan spec requires types to match across all uses of a
          * location but doesn't actually require RelaxedPrecision to match
          * for the whole location.  So we can only apply mediump if every use
          * of the location is mediump.
          */
         nir_alu_type type = nir_alu_type_get_base_type(slots[i].type);
         unsigned bit_size = nir_alu_type_get_type_size(slots[i].type);
         if (bit_size == 32 && !slots[i].any_highp)
            bit_size = 16;
         type |= bit_size;

         unsigned size = slots[i].count * (bit_size / 8);
         unsigned alignment = util_next_power_of_two(size);
         unsigned offset = align(generic_size_B, alignment);
         generic_size_B = offset + size;

         const enum pipe_format format = varying_format(type, slots[i].count);
         assert(size == util_format_get_blocksize(format));

         layout->slots[idx] = (struct pan_varying_slot) {
            .location = i,
            .format = format,
            .section = PAN_VARYING_SECTION_GENERIC,
            .offset = offset,
         };
      }
   }
   layout->count = count;
   layout->generic_size_B = generic_size_B;
}
