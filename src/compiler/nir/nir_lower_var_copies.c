/*
 * Copyright © 2014 Intel Corporation
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

#include "compiler/glsl_types.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"

/*
 * Lowers all copy intrinsics to sequences of load/store intrinsics.
 */

static nir_deref_instr *
build_deref_to_next_wildcard(nir_builder *b,
                             nir_deref_instr *parent,
                             nir_deref_instr ***deref_arr)
{
   for (; **deref_arr; (*deref_arr)++) {
      if ((**deref_arr)->deref_type == nir_deref_type_array_wildcard)
         return parent;

      parent = nir_build_deref_follower(b, parent, **deref_arr);
   }

   assert(**deref_arr == NULL);
   *deref_arr = NULL;
   return parent;
}

static bool
array_copy_loop_allowed(
   const struct nir_lower_var_copies_options *options,
   nir_deref_instr *dst, nir_deref_instr *src)
{
   if (!options || options->loop_array_copy_modes == 0)
      return false;

   const nir_variable_mode modes = dst->modes | src->modes;
   return modes != 0 && !(modes & ~options->loop_array_copy_modes);
}

static void
emit_deref_copy_load_store(nir_builder *b,
                           nir_deref_instr *dst_deref,
                           nir_deref_instr **dst_deref_arr,
                           nir_deref_instr *src_deref,
                           nir_deref_instr **src_deref_arr,
                           enum gl_access_qualifier dst_access,
                           enum gl_access_qualifier src_access,
                           const struct nir_lower_var_copies_options *options)
{
   if (dst_deref_arr || src_deref_arr) {
      assert(dst_deref_arr && src_deref_arr);
      dst_deref = build_deref_to_next_wildcard(b, dst_deref, &dst_deref_arr);
      src_deref = build_deref_to_next_wildcard(b, src_deref, &src_deref_arr);
   }

   if (dst_deref_arr || src_deref_arr) {
      assert(dst_deref_arr && src_deref_arr);
      assert((*dst_deref_arr)->deref_type == nir_deref_type_array_wildcard);
      assert((*src_deref_arr)->deref_type == nir_deref_type_array_wildcard);

      unsigned length = glsl_get_length(src_deref->type);
      /* The wildcards should represent the same number of elements */
      assert(length == glsl_get_length(dst_deref->type));
      assert(length > 0);

      if (array_copy_loop_allowed(options, dst_deref, src_deref)) {
         unsigned dst_index_bit_size = dst_deref->def.bit_size;
         unsigned src_index_bit_size = src_deref->def.bit_size;
         assert((dst_index_bit_size == 32 || dst_index_bit_size == 64) &&
                (src_index_bit_size == 32 || src_index_bit_size == 64));
         nir_variable *idx_var = nir_local_variable_create(
            b->impl, glsl_uint_type(), "copy_idx");
         nir_store_var(b, idx_var, nir_imm_int(b, 0), 1);

         nir_loop *loop = nir_push_loop(b);
         nir_def *idx = nir_load_var(b, idx_var);
         nir_break_if(b, nir_uge_imm(b, idx, length));

         nir_def *dst_idx = nir_u2uN(b, idx, dst_index_bit_size);
         nir_def *src_idx = nir_u2uN(b, idx, src_index_bit_size);
         emit_deref_copy_load_store(
            b, nir_build_deref_array(b, dst_deref, dst_idx),
            dst_deref_arr + 1,
            nir_build_deref_array(b, src_deref, src_idx),
            src_deref_arr + 1, dst_access, src_access, options);

         nir_store_var(b, idx_var, nir_iadd_imm(b, idx, 1), 1);
         nir_pop_loop(b, loop);
      } else {
         for (unsigned i = 0; i < length; i++) {
            emit_deref_copy_load_store(b,
                                       nir_build_deref_array_imm(b, dst_deref, i),
                                       dst_deref_arr + 1,
                                       nir_build_deref_array_imm(b, src_deref, i),
                                       src_deref_arr + 1, dst_access, src_access, options);
         }
      }
   } else {
      assert(glsl_get_bare_type(dst_deref->type) ==
             glsl_get_bare_type(src_deref->type));
      assert(glsl_type_is_vector_or_scalar(dst_deref->type));

      nir_store_deref_with_access(b, dst_deref,
                                  nir_load_deref_with_access(b, src_deref, src_access),
                                  ~0, dst_access);
   }
}

void
nir_lower_deref_copy_instr(nir_builder *b, nir_intrinsic_instr *copy)
{
   /* Unfortunately, there's just no good way to handle wildcards except to
    * flip the chain around and walk the list from variable to final pointer.
    */
   nir_deref_instr *dst = nir_def_as_deref(copy->src[0].ssa);
   nir_deref_instr *src = nir_def_as_deref(copy->src[1].ssa);

   nir_deref_path dst_path, src_path;
   nir_deref_path_init(&dst_path, dst, NULL);
   nir_deref_path_init(&src_path, src, NULL);

   b->cursor = nir_before_instr(&copy->instr);
   emit_deref_copy_load_store(b, dst_path.path[0], &dst_path.path[1],
                              src_path.path[0], &src_path.path[1],
                              nir_intrinsic_dst_access(copy),
                              nir_intrinsic_src_access(copy), NULL);

   nir_deref_path_finish(&dst_path);
   nir_deref_path_finish(&src_path);
}

struct lower_var_copies_state {
   const struct nir_lower_var_copies_options *options;
};

static bool
lower_var_copies_instr(nir_builder *b, nir_intrinsic_instr *copy, void *data)
{
   if (copy->intrinsic != nir_intrinsic_copy_deref)
      return false;

   struct lower_var_copies_state *state = data;
   nir_deref_instr *dst = nir_src_as_deref(copy->src[0]);
   nir_deref_instr *src = nir_src_as_deref(copy->src[1]);
   nir_deref_path dst_path, src_path;
   nir_deref_path_init(&dst_path, dst, NULL);
   nir_deref_path_init(&src_path, src, NULL);

   b->cursor = nir_before_instr(&copy->instr);
   emit_deref_copy_load_store(
      b, dst_path.path[0], &dst_path.path[1],
      src_path.path[0], &src_path.path[1],
      nir_intrinsic_dst_access(copy), nir_intrinsic_src_access(copy),
      state->options);

   nir_deref_path_finish(&dst_path);
   nir_deref_path_finish(&src_path);

   nir_instr_remove(&copy->instr);
   nir_deref_instr_remove_if_unused(nir_src_as_deref(copy->src[0]));
   nir_deref_instr_remove_if_unused(nir_src_as_deref(copy->src[1]));

   nir_instr_free(&copy->instr);
   return true;
}

/* Lowers every copy_var instruction in the program to a sequence of
 * load/store instructions.
 */
bool
nir_lower_var_copies_with_options(
   nir_shader *shader,
   const struct nir_lower_var_copies_options *options)
{
   struct lower_var_copies_state state = {
      .options = options,
   };
   const nir_metadata preserved =
      options && options->loop_array_copy_modes != 0 ? nir_metadata_none : nir_metadata_control_flow;
   shader->info.var_copies_lowered =
      !options || !options->early_lowering;
   return nir_shader_intrinsics_pass(
      shader, lower_var_copies_instr, preserved, &state);
}

bool
nir_lower_var_copies(nir_shader *shader)
{
   return nir_lower_var_copies_with_options(shader, NULL);
}
