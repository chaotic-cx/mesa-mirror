/*
 * Copyright © 2015 Red Hat
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"

static bool
is_color_output(nir_shader *shader, int location)
{
   switch (shader->info.stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_TESS_EVAL:
      switch (location) {
      case VARYING_SLOT_COL0:
      case VARYING_SLOT_COL1:
      case VARYING_SLOT_BFC0:
      case VARYING_SLOT_BFC1:
         return true;
      default:
         return false;
      }
      break;
   case MESA_SHADER_FRAGMENT:
      return (location == FRAG_RESULT_COLOR ||
              location >= FRAG_RESULT_DATA0);
   default:
      return false;
   }
}

static bool
lower_intrinsic(nir_builder *b, nir_intrinsic_instr *intr, nir_shader *shader)
{
   int loc = -1;

   switch (intr->intrinsic) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_view_output:
      loc = nir_intrinsic_io_semantics(intr).location;
      break;
   default:
      return false;
   }

   if (is_color_output(shader, loc)) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *s = intr->src[0].ssa;
      s = nir_fsat(b, s);
      nir_src_rewrite(&intr->src[0], s);
      return true;
   }

   return false;
}

static bool
lower_instr(nir_builder *b, nir_instr *instr, void *cb_data)
{
   if (instr->type == nir_instr_type_intrinsic)
      return lower_intrinsic(b, nir_instr_as_intrinsic(instr), cb_data);
   return false;
}

bool
nir_lower_clamp_color_outputs(nir_shader *shader)
{
   assert(shader->info.io_lowered);
   return nir_shader_instructions_pass(shader, lower_instr,
                                       nir_metadata_control_flow,
                                       shader);
}
