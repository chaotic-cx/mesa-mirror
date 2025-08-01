/*
 * Copyright © 2023 Intel Corporation
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

/*
 * Lower non uniform at sample messages to the interpolator.
 *
 * This is pretty much identical to what nir_lower_non_uniform_access() does.
 * We do it here because otherwise GCM would undo this optimization. Also we
 * can assume divergence analysis here.
 */

#include "intel_nir.h"
#include "compiler/nir/nir_builder.h"

static bool
intel_nir_lower_non_uniform_barycentric_at_sample_instr(nir_builder *b,
                                                        nir_instr *instr,
                                                        void *cb_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   if (intrin->intrinsic != nir_intrinsic_load_barycentric_at_sample)
      return false;

   if (nir_src_is_always_uniform(intrin->src[0]) ||
       !nir_src_is_divergent(&intrin->src[0]))
      return false;

   if (intrin->def.parent_instr->pass_flags != 0)
      return false;

   nir_def *sample_id = intrin->src[0].ssa;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_push_loop(b);
   {
      nir_def *first_sample_id = nir_read_first_invocation(b, sample_id);

      nir_push_if(b, nir_ieq(b, sample_id, first_sample_id));
      {
         nir_builder_instr_insert(b, &intrin->instr);
         intrin->def.parent_instr->pass_flags = 1;

         nir_src_rewrite(&intrin->src[0], first_sample_id);

         nir_jump(b, nir_jump_break);
      }
   }

   return true;
}

static bool
intel_nir_lower_non_uniform_interpolated_input_instr(nir_builder *b,
                                                     nir_instr *instr,
                                                     void *cb_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *load_ii = nir_instr_as_intrinsic(instr);
   if (load_ii->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;

   assert(load_ii->src[0].ssa->parent_instr->type == nir_instr_type_intrinsic);

   nir_intrinsic_instr *bary =
      nir_def_as_intrinsic(load_ii->src[0].ssa);
   if (bary->intrinsic != nir_intrinsic_load_barycentric_at_sample)
      return false;

   if (nir_src_is_always_uniform(bary->src[0]) ||
       !nir_src_is_divergent(&bary->src[0]))
      return false;

   nir_def *sample_id = bary->src[0].ssa;

   b->cursor = nir_instr_remove(&load_ii->instr);

   nir_push_loop(b);
   {
      nir_def *first_sample_id = nir_read_first_invocation(b, sample_id);

      nir_push_if(b, nir_ieq(b, sample_id, first_sample_id));
      {
         nir_def *new_bary = nir_load_barycentric_at_sample(
            b, bary->def.bit_size, first_sample_id,
            .interp_mode = nir_intrinsic_interp_mode(bary));

         /* Set pass_flags so that the other lowering pass won't try to also
          * lower this new load_barycentric_at_sample.
          */
         new_bary->parent_instr->pass_flags = 1;

         nir_builder_instr_insert(b, &load_ii->instr);

         nir_src_rewrite(&load_ii->src[0], new_bary);

         nir_jump(b, nir_jump_break);
      }
   }

   return true;
}

bool
intel_nir_lower_non_uniform_barycentric_at_sample(nir_shader *nir)
{
   bool progress;

   nir_divergence_analysis(nir);
   nir_shader_clear_pass_flags(nir);

   progress = nir_shader_instructions_pass(
      nir,
      intel_nir_lower_non_uniform_interpolated_input_instr,
      nir_metadata_none,
      NULL);

   progress = nir_shader_instructions_pass(
      nir,
      intel_nir_lower_non_uniform_barycentric_at_sample_instr,
      nir_metadata_none,
      NULL) || progress;

   return progress;
}
