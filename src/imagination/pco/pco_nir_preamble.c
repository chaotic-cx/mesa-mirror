/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pco_nir_preamble.c
 *
 * \brief PCO NIR preamble lowering passes.
 */

#include "hwdef/rogue_hw_defs.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"
#include "pco.h"
#include "pco_builder.h"
#include "pco_common.h"
#include "pco_internal.h"
#include "usclib/pco_usclib.h"
#include "util/macros.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

/* TODO: tweak this value. */
#define MAX_PREAMBLE_REGS 128u

static void def_size(nir_def *def,
                     unsigned *size,
                     unsigned *align,
                     nir_preamble_class *class)
{
   /* Minimum 32-bit/DWORD granularity. */
   unsigned bit_size = def->bit_size < 32u ? 32u : def->bit_size;

   /* Bit -> DWORD granularity. */
   *size = DIV_ROUND_UP(bit_size, 32u) * def->num_components;
   *align = 1u;

   /* Use general for everything; all the image/sampler/etc. state would already
    * be present in the same regbank as preamble regs, so we don't have to
    * worry about them.
    */
   *class = nir_preamble_class_general;
}

static float instr_cost(UNUSED nir_instr *instr, UNUSED const void *data)
{
   /* TODO: add an actual cost calculation. */
   return 1.0f;
}

static float rewrite_cost(UNUSED nir_def *def, UNUSED const void *data)
{
   /* TODO: add an actual cost calculation. */
   return 0.0f;
}

static bool avoid_intr(const nir_intrinsic_instr *intr, UNUSED const void *data)
{
   switch (intr->intrinsic) {
   /* These just copy from coeff regs anyway, no point copying to shareds. */
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_workgroup_id:
      return true;

   default:
      break;
   }

   return false;
}

static bool avoid_instr(const nir_instr *instr, const void *data)
{
   switch (instr->type) {
   case nir_instr_type_intrinsic:
      return avoid_intr(nir_instr_as_intrinsic(instr), data);

   default:
      break;
   }

   return false;
}

bool pco_nir_opt_preamble(nir_shader *shader, pco_data *data)
{
   nir_opt_preamble_options options = {
      .drawid_uniform = true,
      .subgroup_size_uniform = true,
      .load_workgroup_size_allowed = true,
      .def_size = def_size,
      .preamble_storage_size = {
         [nir_preamble_class_general] = MAX_PREAMBLE_REGS,
         [nir_preamble_class_image] = 0u,
         [nir_preamble_class_sampler] = 0u,
      },
      .instr_cost_cb = instr_cost,
      .rewrite_cost_cb = rewrite_cost,
      .avoid_instr_cb = avoid_instr,
      .cb_data = data,
   };

   unsigned size[nir_preamble_num_classes] = { 0u };
   bool progress = false;

   NIR_PASS(progress, shader, nir_opt_preamble, &options, size);
   if (!progress)
      return false;

   assert(size[nir_preamble_class_general] > 0u);
   assert(!size[nir_preamble_class_image] && !size[nir_preamble_class_sampler]);

   data->common.uses.preamble = true;
   data->common.preamble = (pco_range){
      .count = size[nir_preamble_class_general],
   };

   return true;
}

static bool lower_vec_load_preamble(nir_builder *b, nir_intrinsic_instr *intr)
{
   unsigned num_components = nir_intrinsic_dest_components(intr);
   if (num_components == 1u)
      return false;

   unsigned base = nir_intrinsic_base(intr);
   nir_preamble_class preamble_class = nir_intrinsic_preamble_class(intr);

   nir_def *loads[NIR_MAX_VEC_COMPONENTS];

   for (unsigned c = 0u; c < num_components; ++c) {
      loads[c] = nir_load_preamble(b,
                                   1u,
                                   32u,
                                   .base = base + c,
                                   .preamble_class = preamble_class);
   }

   nir_def_replace(&intr->def, nir_vec(b, loads, num_components));

   return true;
}

static bool lower_vec_store_preamble(nir_builder *b, nir_intrinsic_instr *intr)
{
   unsigned num_components = nir_intrinsic_src_components(intr, 0);
   if (num_components == 1u)
      return false;

   unsigned base = nir_intrinsic_base(intr);
   nir_preamble_class preamble_class = nir_intrinsic_preamble_class(intr);

   for (unsigned c = 0u; c < num_components; ++c) {
      nir_def *chan = nir_channel(b, intr->src[0].ssa, c);
      nir_store_preamble(b,
                         chan,
                         .base = base + c,
                         .preamble_class = preamble_class);
   }

   nir_instr_remove(&intr->instr);

   return true;
}

static bool lower_vec_preamble(nir_builder *b,
                               nir_intrinsic_instr *intr,
                               UNUSED void *cb_data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_preamble:
      return lower_vec_load_preamble(b, intr);

   case nir_intrinsic_store_preamble:
      return lower_vec_store_preamble(b, intr);

   default:
      break;
   }

   return false;
}

bool pco_nir_lower_preamble_to_scalar(nir_shader *shader)
{
   /* Internal shaders can use the preamble for texture/sampler state words,
    * which shouldn't be scalarised.
    */
   if (shader->info.internal)
      return false;

   return nir_shader_intrinsics_pass(shader,
                                     lower_vec_preamble,
                                     nir_metadata_control_flow,
                                     NULL);
}
