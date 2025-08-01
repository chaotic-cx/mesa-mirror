/*
 * Copyright 2018 Collabora Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "nir_opcodes.h"
#include "shader_enums.h"
#include "zink_context.h"
#include "zink_compiler.h"
#include "zink_descriptors.h"
#include "zink_program.h"
#include "zink_screen.h"
#include "nir_to_spirv/nir_to_spirv.h"

#include "pipe/p_state.h"

#include "nir.h"
#include "nir_xfb_info.h"
#include "nir/nir_draw_helpers.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "compiler/nir/nir_builtin_builder.h"

#include "nir/tgsi_to_nir.h"
#include "tgsi/tgsi_dump.h"

#include "util/u_memory.h"

#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "vk_util.h"

bool
zink_lower_cubemap_to_array(nir_shader *s, uint32_t nonseamless_cube_mask);


static void
copy_vars(nir_builder *b, nir_deref_instr *dst, nir_deref_instr *src)
{
   assert(glsl_get_bare_type(dst->type) == glsl_get_bare_type(src->type));
   if (glsl_type_is_struct_or_ifc(dst->type)) {
      for (unsigned i = 0; i < glsl_get_length(dst->type); ++i) {
         copy_vars(b, nir_build_deref_struct(b, dst, i), nir_build_deref_struct(b, src, i));
      }
   } else if (glsl_type_is_array_or_matrix(dst->type)) {
      unsigned count = glsl_type_is_array(dst->type) ? glsl_array_size(dst->type) : glsl_get_matrix_columns(dst->type);
      for (unsigned i = 0; i < count; i++) {
         copy_vars(b, nir_build_deref_array_imm(b, dst, i), nir_build_deref_array_imm(b, src, i));
      }
   } else {
      nir_def *load = nir_load_deref(b, src);
      nir_store_deref(b, dst, load, BITFIELD_MASK(load->num_components));
   }
}

static bool
is_clipcull_dist(int location)
{
   switch (location) {
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CLIP_DIST1:
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_CULL_DIST1:
      return true;
   default: break;
   }
   return false;
}

#define SIZEOF_FIELD(type, field) sizeof(((type *)0)->field)

static void
create_gfx_pushconst(nir_shader *nir)
{
#define PUSHCONST_MEMBER(member_idx, field)                                                                     \
fields[member_idx].type =                                                                                       \
   glsl_array_type(glsl_uint_type(), SIZEOF_FIELD(struct zink_gfx_push_constant, field) / sizeof(uint32_t), 0); \
fields[member_idx].name = ralloc_asprintf(nir, #field);                                                         \
fields[member_idx].offset = offsetof(struct zink_gfx_push_constant, field);

   nir_variable *pushconst;
   /* create compatible layout for the ntv push constant loader */
   struct glsl_struct_field *fields = rzalloc_array(nir, struct glsl_struct_field, ZINK_GFX_PUSHCONST_MAX);
   PUSHCONST_MEMBER(ZINK_GFX_PUSHCONST_DRAW_MODE_IS_INDEXED, draw_mode_is_indexed);
   PUSHCONST_MEMBER(ZINK_GFX_PUSHCONST_DRAW_ID, draw_id);
   PUSHCONST_MEMBER(ZINK_GFX_PUSHCONST_FRAMEBUFFER_IS_LAYERED, framebuffer_is_layered);
   PUSHCONST_MEMBER(ZINK_GFX_PUSHCONST_DEFAULT_INNER_LEVEL, default_inner_level);
   PUSHCONST_MEMBER(ZINK_GFX_PUSHCONST_DEFAULT_OUTER_LEVEL, default_outer_level);
   PUSHCONST_MEMBER(ZINK_GFX_PUSHCONST_LINE_STIPPLE_PATTERN, line_stipple_pattern);
   PUSHCONST_MEMBER(ZINK_GFX_PUSHCONST_VIEWPORT_SCALE, viewport_scale);
   PUSHCONST_MEMBER(ZINK_GFX_PUSHCONST_LINE_WIDTH, line_width);

   pushconst = nir_variable_create(nir, nir_var_mem_push_const,
                                   glsl_struct_type(fields, ZINK_GFX_PUSHCONST_MAX, "struct", false),
                                   "gfx_pushconst");
   pushconst->data.location = INT_MAX; //doesn't really matter

#undef PUSHCONST_MEMBER
}

static bool
lower_basevertex_instr(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   if (instr->intrinsic != nir_intrinsic_load_base_vertex)
      return false;

   b->cursor = nir_after_instr(&instr->instr);
   nir_intrinsic_instr *load = nir_intrinsic_instr_create(b->shader, nir_intrinsic_load_push_constant_zink);
   load->src[0] = nir_src_for_ssa(nir_imm_int(b, ZINK_GFX_PUSHCONST_DRAW_MODE_IS_INDEXED));
   load->num_components = 1;
   nir_def_init(&load->instr, &load->def, 1, 32);
   nir_builder_instr_insert(b, &load->instr);

   nir_def *composite = nir_build_alu(b, nir_op_bcsel,
                                          nir_build_alu(b, nir_op_ieq, &load->def, nir_imm_int(b, 1), NULL, NULL),
                                          &instr->def,
                                          nir_imm_int(b, 0),
                                          NULL);

   nir_def_rewrite_uses_after(&instr->def, composite);
   return true;
}

static bool
lower_basevertex(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_VERTEX)
      return false;

   if (!BITSET_TEST(shader->info.system_values_read, SYSTEM_VALUE_BASE_VERTEX))
      return false;

   return nir_shader_intrinsics_pass(shader, lower_basevertex_instr,
                                     nir_metadata_dominance, NULL);
}


static bool
lower_drawid_instr(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   if (instr->intrinsic != nir_intrinsic_load_draw_id)
      return false;

   b->cursor = nir_before_instr(&instr->instr);
   nir_intrinsic_instr *load = nir_intrinsic_instr_create(b->shader, nir_intrinsic_load_push_constant_zink);
   load->src[0] = nir_src_for_ssa(nir_imm_int(b, ZINK_GFX_PUSHCONST_DRAW_ID));
   load->num_components = 1;
   nir_def_init(&load->instr, &load->def, 1, 32);
   nir_builder_instr_insert(b, &load->instr);

   nir_def_rewrite_uses(&instr->def, &load->def);

   return true;
}

static bool
lower_drawid(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_VERTEX)
      return false;

   if (!BITSET_TEST(shader->info.system_values_read, SYSTEM_VALUE_DRAW_ID))
      return false;

   return nir_shader_intrinsics_pass(shader, lower_drawid_instr,
                                     nir_metadata_dominance, NULL);
}

struct lower_gl_point_state {
   nir_variable *gl_pos_out;
   nir_variable *gl_point_size;
};

static bool
lower_gl_point_gs_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_gl_point_state *state = data;
   nir_def *vp_scale, *pos;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   if (intrin->intrinsic != nir_intrinsic_emit_vertex_with_counter &&
       intrin->intrinsic != nir_intrinsic_emit_vertex)
      return false;

   if (nir_intrinsic_stream_id(intrin) != 0)
      return false;

   if (intrin->intrinsic == nir_intrinsic_end_primitive_with_counter ||
         intrin->intrinsic == nir_intrinsic_end_primitive) {
      nir_instr_remove(&intrin->instr);
      return true;
   }

   b->cursor = nir_before_instr(instr);

   // viewport-map endpoints
   nir_def *vp_const_pos = nir_imm_int(b, ZINK_GFX_PUSHCONST_VIEWPORT_SCALE);
   vp_scale = nir_load_push_constant_zink(b, 2, 32, vp_const_pos);

   // Load point info values
   nir_def *point_size = nir_load_var(b, state->gl_point_size);
   nir_def *point_pos = nir_load_var(b, state->gl_pos_out);

   // w_delta = gl_point_size / width_viewport_size_scale * gl_Position.w
   nir_def *w_delta = nir_fdiv(b, point_size, nir_channel(b, vp_scale, 0));
   w_delta = nir_fmul(b, w_delta, nir_channel(b, point_pos, 3));
   // halt_w_delta = w_delta / 2
   nir_def *half_w_delta = nir_fmul_imm(b, w_delta, 0.5);

   // h_delta = gl_point_size / height_viewport_size_scale * gl_Position.w
   nir_def *h_delta = nir_fdiv(b, point_size, nir_channel(b, vp_scale, 1));
   h_delta = nir_fmul(b, h_delta, nir_channel(b, point_pos, 3));
   // halt_h_delta = h_delta / 2
   nir_def *half_h_delta = nir_fmul_imm(b, h_delta, 0.5);

   nir_def *point_dir[4][2] = {
      { nir_imm_float(b, -1), nir_imm_float(b, -1) },
      { nir_imm_float(b, -1), nir_imm_float(b, 1) },
      { nir_imm_float(b, 1), nir_imm_float(b, -1) },
      { nir_imm_float(b, 1), nir_imm_float(b, 1) }
   };

   nir_def *point_pos_x = nir_channel(b, point_pos, 0);
   nir_def *point_pos_y = nir_channel(b, point_pos, 1);

   for (size_t i = 0; i < 4; i++) {
      pos = nir_vec4(b,
                     nir_ffma(b, half_w_delta, point_dir[i][0], point_pos_x),
                     nir_ffma(b, half_h_delta, point_dir[i][1], point_pos_y),
                     nir_channel(b, point_pos, 2),
                     nir_channel(b, point_pos, 3));

      nir_store_var(b, state->gl_pos_out, pos, 0xf);

      nir_emit_vertex(b);
   }

   nir_end_primitive(b);

   nir_instr_remove(&intrin->instr);

   return true;
}

static bool
lower_gl_point_gs(nir_shader *shader)
{
   struct lower_gl_point_state state;

   shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   shader->info.gs.vertices_out *= 4;

   // Gets the gl_Position in and out
   state.gl_pos_out =
      nir_find_variable_with_location(shader, nir_var_shader_out,
                                      VARYING_SLOT_POS);
   state.gl_point_size =
      nir_find_variable_with_location(shader, nir_var_shader_out,
                                      VARYING_SLOT_PSIZ);

   // if position in or gl_PointSize aren't written, we have nothing to do
   if (!state.gl_pos_out || !state.gl_point_size)
      return false;

   return nir_shader_instructions_pass(shader, lower_gl_point_gs_instr,
                                       nir_metadata_dominance, &state);
}

struct lower_pv_mode_state {
   nir_variable *varyings[VARYING_SLOT_MAX][4];
   nir_variable *pos_counter;
   nir_variable *out_pos_counter;
   nir_variable *ring_offset;
   unsigned ring_size;
   unsigned primitive_vert_count;
   unsigned prim;
};

static nir_def*
lower_pv_mode_gs_ring_index(nir_builder *b,
                            struct lower_pv_mode_state *state,
                            nir_def *index)
{
   nir_def *ring_offset = nir_load_var(b, state->ring_offset);
   return nir_imod_imm(b, nir_iadd(b, index, ring_offset),
                          state->ring_size);
}

/* Given the final deref of chain of derefs this function will walk up the chain
 * until it finds a var deref.
 *
 * It will then recreate an identical chain that ends with the provided deref.
 */
static nir_deref_instr*
replicate_derefs(nir_builder *b, nir_deref_instr *old, nir_deref_instr *new)
{
   nir_deref_instr *parent = nir_deref_instr_parent(old);
   if (!parent)
      return new;
   switch(old->deref_type) {
   case nir_deref_type_var:
      return new;
   case nir_deref_type_array:
      return nir_build_deref_array(b, replicate_derefs(b, parent, new), old->arr.index.ssa);
   case nir_deref_type_struct:
      return nir_build_deref_struct(b, replicate_derefs(b, parent, new), old->strct.index);
   case nir_deref_type_array_wildcard:
   case nir_deref_type_ptr_as_array:
   case nir_deref_type_cast:
      UNREACHABLE("unexpected deref type");
   }
   UNREACHABLE("impossible deref type");
}

static bool
lower_pv_mode_gs_store(nir_builder *b,
                       nir_intrinsic_instr *intrin,
                       struct lower_pv_mode_state *state)
{
   b->cursor = nir_before_instr(&intrin->instr);
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   if (nir_deref_mode_is(deref, nir_var_shader_out)) {
      nir_variable *var = nir_deref_instr_get_variable(deref);

      gl_varying_slot location = var->data.location;
      unsigned location_frac = var->data.location_frac;
      assert(state->varyings[location][location_frac]);
      nir_def *pos_counter = nir_load_var(b, state->pos_counter);
      nir_def *index = lower_pv_mode_gs_ring_index(b, state, pos_counter);
      nir_deref_instr *varying_deref = nir_build_deref_var(b, state->varyings[location][location_frac]);
      nir_deref_instr *ring_deref = nir_build_deref_array(b, varying_deref, index);
      // recreate the chain of deref that lead to the store.
      nir_deref_instr *new_top_deref = replicate_derefs(b, deref, ring_deref);
      nir_store_deref(b, new_top_deref, intrin->src[1].ssa, nir_intrinsic_write_mask(intrin));
      nir_instr_remove(&intrin->instr);
      return true;
   }

   return false;
}

static void
lower_pv_mode_emit_rotated_prim(nir_builder *b,
                                struct lower_pv_mode_state *state,
                                nir_def *current_vertex)
{
   nir_def *two = nir_imm_int(b, 2);
   nir_def *three = nir_imm_int(b, 3);
   bool is_triangle = state->primitive_vert_count == 3;
   /* This shader will always see the last three vertices emitted by the user gs.
    * The following table is used to to rotate primitives within a strip generated
    * by the user gs such that the last vertex becomes the first.
    *
    * [lines, tris][even/odd index][vertex mod 3]
    */
   static const unsigned vert_maps[2][2][3] = {
      {{1, 0, 0}, {1, 0, 0}},
      {{2, 0, 1}, {2, 1, 0}}
   };
   /* When the primive supplied to the gs comes from a strip, the last provoking vertex
    * is either the last or the second, depending on whether the triangle is at an odd
    * or even position within the strip.
    *
    * odd or even primitive within draw
    */
   nir_def *odd_prim = nir_imod(b, nir_load_primitive_id(b), two);
   for (unsigned i = 0; i < state->primitive_vert_count; i++) {
      /* odd or even triangle within strip emitted by user GS
       * this is handled using the table
       */
      nir_def *odd_user_prim = nir_imod(b, current_vertex, two);
      unsigned offset_even = vert_maps[is_triangle][0][i];
      unsigned offset_odd = vert_maps[is_triangle][1][i];
      nir_def *offset_even_value = nir_imm_int(b, offset_even);
      nir_def *offset_odd_value = nir_imm_int(b, offset_odd);
      nir_def *rotated_i = nir_bcsel(b, nir_b2b1(b, odd_user_prim),
                                            offset_odd_value, offset_even_value);
      /* Here we account for how triangles are provided to the gs from a strip.
       * For even primitives we rotate by 3, meaning we do nothing.
       * For odd primitives we rotate by 2, combined with the previous rotation this
       * means the second vertex becomes the last.
       */
      if (state->prim == ZINK_PVE_PRIMITIVE_TRISTRIP)
        rotated_i = nir_imod(b, nir_iadd(b, rotated_i,
                                            nir_isub(b, three,
                                                        odd_prim)),
                                            three);
      /* Triangles that come from fans are provided to the gs the same way as
       * odd triangles from a strip so always rotate by 2.
       */
      else if (state->prim == ZINK_PVE_PRIMITIVE_FAN)
        rotated_i = nir_imod(b, nir_iadd_imm(b, rotated_i, 2),
                                three);
      rotated_i = nir_iadd(b, rotated_i, current_vertex);
      nir_foreach_variable_with_modes(var, b->shader, nir_var_shader_out) {
         gl_varying_slot location = var->data.location;
         unsigned location_frac = var->data.location_frac;
         if (state->varyings[location][location_frac]) {
            nir_def *index = lower_pv_mode_gs_ring_index(b, state, rotated_i);
            nir_deref_instr *value = nir_build_deref_array(b, nir_build_deref_var(b, state->varyings[location][location_frac]), index);
            copy_vars(b, nir_build_deref_var(b, var), value);
         }
      }
      nir_emit_vertex(b);
   }
}

static bool
lower_pv_mode_gs_emit_vertex(nir_builder *b,
                             nir_intrinsic_instr *intrin,
                             struct lower_pv_mode_state *state)
{
   b->cursor = nir_before_instr(&intrin->instr);

   // increment pos_counter
   nir_def *pos_counter = nir_load_var(b, state->pos_counter);
   nir_store_var(b, state->pos_counter, nir_iadd_imm(b, pos_counter, 1), 1);

   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
lower_pv_mode_gs_end_primitive(nir_builder *b,
                               nir_intrinsic_instr *intrin,
                               struct lower_pv_mode_state *state)
{
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *pos_counter = nir_load_var(b, state->pos_counter);
   nir_push_loop(b);
   {
      nir_def *out_pos_counter = nir_load_var(b, state->out_pos_counter);
      nir_break_if(b, nir_ilt(b, nir_isub(b, pos_counter, out_pos_counter),
                                 nir_imm_int(b, state->primitive_vert_count)));

      lower_pv_mode_emit_rotated_prim(b, state, out_pos_counter);
      nir_end_primitive(b);

      nir_store_var(b, state->out_pos_counter, nir_iadd_imm(b, out_pos_counter, 1), 1);
   }
   nir_pop_loop(b, NULL);
   /* Set the ring offset such that when position 0 is
    * read we get the last value written
    */
   nir_store_var(b, state->ring_offset, pos_counter, 1);
   nir_store_var(b, state->pos_counter, nir_imm_int(b, 0), 1);
   nir_store_var(b, state->out_pos_counter, nir_imm_int(b, 0), 1);

   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
lower_pv_mode_gs_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   struct lower_pv_mode_state *state = data;
   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_store_deref:
      return lower_pv_mode_gs_store(b, intrin, state);
   case nir_intrinsic_copy_deref:
      UNREACHABLE("should be lowered");
   case nir_intrinsic_emit_vertex_with_counter:
   case nir_intrinsic_emit_vertex:
      return lower_pv_mode_gs_emit_vertex(b, intrin, state);
   case nir_intrinsic_end_primitive:
   case nir_intrinsic_end_primitive_with_counter:
      return lower_pv_mode_gs_end_primitive(b, intrin, state);
   default:
      return false;
   }
}

static bool
lower_pv_mode_gs(nir_shader *shader, unsigned prim)
{
   nir_builder b;
   struct lower_pv_mode_state state;
   memset(state.varyings, 0, sizeof(state.varyings));

   nir_function_impl *entry = nir_shader_get_entrypoint(shader);
   b = nir_builder_at(nir_before_impl(entry));

   state.primitive_vert_count =
      mesa_vertices_per_prim(shader->info.gs.output_primitive);
   state.ring_size = shader->info.gs.vertices_out;

   nir_foreach_variable_with_modes(var, shader, nir_var_shader_out) {
      gl_varying_slot location = var->data.location;
      unsigned location_frac = var->data.location_frac;

      char name[100];
      snprintf(name, sizeof(name), "__tmp_primverts_%d_%d", location, location_frac);
      state.varyings[location][location_frac] =
         nir_local_variable_create(entry,
                                   glsl_array_type(var->type,
                                                   state.ring_size,
                                                   false),
                                   name);
   }

   state.pos_counter = nir_local_variable_create(entry,
                                                 glsl_uint_type(),
                                                 "__pos_counter");

   state.out_pos_counter = nir_local_variable_create(entry,
                                                     glsl_uint_type(),
                                                     "__out_pos_counter");

   state.ring_offset = nir_local_variable_create(entry,
                                                 glsl_uint_type(),
                                                 "__ring_offset");

   state.prim = prim;

   // initialize pos_counter and out_pos_counter
   nir_store_var(&b, state.pos_counter, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, state.out_pos_counter, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, state.ring_offset, nir_imm_int(&b, 0), 1);

   shader->info.gs.vertices_out = (shader->info.gs.vertices_out -
                                   (state.primitive_vert_count - 1)) *
                                  state.primitive_vert_count;
   return nir_shader_instructions_pass(shader, lower_pv_mode_gs_instr,
                                       nir_metadata_dominance, &state);
}

struct lower_line_stipple_state {
   nir_variable *pos_out;
   nir_variable *stipple_out;
   nir_variable *prev_pos;
   nir_variable *pos_counter;
   nir_variable *stipple_counter;
   bool line_rectangular;
};

static nir_def *
viewport_map(nir_builder *b, nir_def *vert,
             nir_def *scale)
{
   nir_def *w_recip = nir_frcp(b, nir_channel(b, vert, 3));
   nir_def *ndc_point = nir_fmul(b, nir_trim_vector(b, vert, 2),
                                        w_recip);
   return nir_fmul(b, ndc_point, scale);
}

static bool
lower_line_stipple_gs_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_line_stipple_state *state = data;
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   if (intrin->intrinsic != nir_intrinsic_emit_vertex_with_counter &&
       intrin->intrinsic != nir_intrinsic_emit_vertex)
      return false;

   b->cursor = nir_before_instr(instr);

   nir_push_if(b, nir_ine_imm(b, nir_load_var(b, state->pos_counter), 0));
   // viewport-map endpoints
   nir_def *vp_scale = nir_load_push_constant_zink(b, 2, 32,
                                                       nir_imm_int(b, ZINK_GFX_PUSHCONST_VIEWPORT_SCALE));
   nir_def *prev = nir_load_var(b, state->prev_pos);
   nir_def *curr = nir_load_var(b, state->pos_out);
   prev = viewport_map(b, prev, vp_scale);
   curr = viewport_map(b, curr, vp_scale);

   // calculate length of line
   nir_def *len;
   if (state->line_rectangular)
      len = nir_fast_distance(b, prev, curr);
   else {
      nir_def *diff = nir_fabs(b, nir_fsub(b, prev, curr));
      len = nir_fmax(b, nir_channel(b, diff, 0), nir_channel(b, diff, 1));
   }
   // update stipple_counter
   nir_store_var(b, state->stipple_counter,
                    nir_fadd(b, nir_load_var(b, state->stipple_counter),
                                len), 1);
   nir_pop_if(b, NULL);
   // emit stipple out
   nir_copy_var(b, state->stipple_out, state->stipple_counter);
   nir_copy_var(b, state->prev_pos, state->pos_out);

   // update prev_pos and pos_counter for next vertex
   b->cursor = nir_after_instr(instr);
   nir_store_var(b, state->pos_counter,
                    nir_iadd_imm(b, nir_load_var(b, state->pos_counter),
                                    1), 1);

   return true;
}

static bool
lower_line_stipple_gs(nir_shader *shader, bool line_rectangular)
{
   nir_builder b;
   struct lower_line_stipple_state state;

   state.pos_out =
      nir_find_variable_with_location(shader, nir_var_shader_out,
                                      VARYING_SLOT_POS);

   // if position isn't written, we have nothing to do
   if (!state.pos_out)
      return false;

   state.stipple_out = nir_variable_create(shader, nir_var_shader_out,
                                           glsl_float_type(),
                                           "__stipple");
   state.stipple_out->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
   state.stipple_out->data.driver_location = shader->num_outputs++;
   state.stipple_out->data.location = MAX2(util_last_bit64(shader->info.outputs_written), VARYING_SLOT_VAR0);
   shader->info.outputs_written |= BITFIELD64_BIT(state.stipple_out->data.location);

   // create temp variables
   state.prev_pos = nir_variable_create(shader, nir_var_shader_temp,
                                        glsl_vec4_type(),
                                        "__prev_pos");
   state.pos_counter = nir_variable_create(shader, nir_var_shader_temp,
                                           glsl_uint_type(),
                                           "__pos_counter");
   state.stipple_counter = nir_variable_create(shader, nir_var_shader_temp,
                                               glsl_float_type(),
                                               "__stipple_counter");

   state.line_rectangular = line_rectangular;
   // initialize pos_counter and stipple_counter
   nir_function_impl *entry = nir_shader_get_entrypoint(shader);
   b = nir_builder_at(nir_before_impl(entry));
   nir_store_var(&b, state.pos_counter, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, state.stipple_counter, nir_imm_float(&b, 0), 1);

   return nir_shader_instructions_pass(shader, lower_line_stipple_gs_instr,
                                       nir_metadata_dominance, &state);
}

static bool
lower_line_stipple_fs(nir_shader *shader)
{
   nir_builder b;
   nir_function_impl *entry = nir_shader_get_entrypoint(shader);
   b = nir_builder_at(nir_after_impl(entry));

   // create stipple counter
   nir_variable *stipple = nir_variable_create(shader, nir_var_shader_in,
                                               glsl_float_type(),
                                               "__stipple");
   stipple->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
   stipple->data.driver_location = shader->num_inputs++;
   stipple->data.location = MAX2(util_last_bit64(shader->info.inputs_read), VARYING_SLOT_VAR0);
   shader->info.inputs_read |= BITFIELD64_BIT(stipple->data.location);

   nir_variable *sample_mask_out =
      nir_find_variable_with_location(shader, nir_var_shader_out,
                                      FRAG_RESULT_SAMPLE_MASK);
   if (!sample_mask_out) {
      sample_mask_out = nir_variable_create(shader, nir_var_shader_out,
                                        glsl_uint_type(), "sample_mask");
      sample_mask_out->data.driver_location = shader->num_outputs++;
      sample_mask_out->data.location = FRAG_RESULT_SAMPLE_MASK;
   }

   nir_def *pattern = nir_load_push_constant_zink(&b, 1, 32,
                                                      nir_imm_int(&b, ZINK_GFX_PUSHCONST_LINE_STIPPLE_PATTERN));
   nir_def *factor = nir_i2f32(&b, nir_ishr_imm(&b, pattern, 16));
   pattern = nir_iand_imm(&b, pattern, 0xffff);

   nir_def *sample_mask_in = nir_load_sample_mask_in(&b);
   nir_variable *v = nir_local_variable_create(entry, glsl_uint_type(), NULL);
   nir_variable *sample_mask = nir_local_variable_create(entry, glsl_uint_type(), NULL);
   nir_store_var(&b, v, sample_mask_in, 1);
   nir_store_var(&b, sample_mask, sample_mask_in, 1);
   nir_push_loop(&b);
   {
      nir_def *value = nir_load_var(&b, v);
      nir_def *index = nir_ufind_msb(&b, value);
      nir_def *index_mask = nir_ishl(&b, nir_imm_int(&b, 1), index);
      nir_def *new_value = nir_ixor(&b, value, index_mask);
      nir_store_var(&b, v, new_value,  1);
      nir_break_if(&b, nir_ieq_imm(&b, value, 0));

      nir_def *stipple_pos =
         nir_interp_deref_at_sample(&b, 1, 32,
            &nir_build_deref_var(&b, stipple)->def, index);
      stipple_pos = nir_fmod(&b, nir_fdiv(&b, stipple_pos, factor),
                                 nir_imm_float(&b, 16.0));
      stipple_pos = nir_f2i32(&b, stipple_pos);
      nir_def *bit =
         nir_iand_imm(&b, nir_ishr(&b, pattern, stipple_pos), 1);
      nir_push_if(&b, nir_ieq_imm(&b, bit, 0));
      {
         nir_def *sample_mask_value = nir_load_var(&b, sample_mask);
         sample_mask_value = nir_ixor(&b, sample_mask_value, index_mask);
         nir_store_var(&b, sample_mask, sample_mask_value, 1);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_loop(&b, NULL);
   nir_store_var(&b, sample_mask_out, nir_load_var(&b, sample_mask), 1);

   return true;
}

struct lower_line_smooth_state {
   nir_variable *pos_out;
   nir_variable *line_coord_out;
   nir_variable *prev_pos;
   nir_variable *pos_counter;
   nir_variable *prev_varyings[VARYING_SLOT_MAX][4],
                *varyings[VARYING_SLOT_MAX][4]; // location_frac
};

static bool
lower_line_smooth_gs_store(nir_builder *b,
                           nir_intrinsic_instr *intrin,
                           struct lower_line_smooth_state *state)
{
   b->cursor = nir_before_instr(&intrin->instr);
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   if (nir_deref_mode_is(deref, nir_var_shader_out)) {
      nir_variable *var = nir_deref_instr_get_variable(deref);

      // we take care of position elsewhere
      gl_varying_slot location = var->data.location;
      unsigned location_frac = var->data.location_frac;
      if (location != VARYING_SLOT_POS) {
         assert(state->varyings[location]);
         nir_store_var(b, state->varyings[location][location_frac],
                       intrin->src[1].ssa,
                       nir_intrinsic_write_mask(intrin));
         nir_instr_remove(&intrin->instr);
         return true;
      }
   }

   return false;
}

static bool
lower_line_smooth_gs_emit_vertex(nir_builder *b,
                                 nir_intrinsic_instr *intrin,
                                 struct lower_line_smooth_state *state)
{
   b->cursor = nir_before_instr(&intrin->instr);

   nir_push_if(b, nir_ine_imm(b, nir_load_var(b, state->pos_counter), 0));
   nir_def *vp_scale = nir_load_push_constant_zink(b, 2, 32,
                                                       nir_imm_int(b, ZINK_GFX_PUSHCONST_VIEWPORT_SCALE));
   nir_def *prev = nir_load_var(b, state->prev_pos);
   nir_def *curr = nir_load_var(b, state->pos_out);
   nir_def *prev_vp = viewport_map(b, prev, vp_scale);
   nir_def *curr_vp = viewport_map(b, curr, vp_scale);

   nir_def *width = nir_load_push_constant_zink(b, 1, 32,
                                                    nir_imm_int(b, ZINK_GFX_PUSHCONST_LINE_WIDTH));
   nir_def *half_width = nir_fadd_imm(b, nir_fmul_imm(b, width, 0.5), 0.5);

   const unsigned yx[2] = { 1, 0 };
   nir_def *vec = nir_fsub(b, curr_vp, prev_vp);
   nir_def *len = nir_fast_length(b, vec);
   nir_def *dir = nir_normalize(b, vec);
   nir_def *half_length = nir_fmul_imm(b, len, 0.5);
   half_length = nir_fadd_imm(b, half_length, 0.5);

   nir_def *vp_scale_rcp = nir_frcp(b, vp_scale);
   nir_def *tangent =
      nir_fmul(b,
               nir_fmul(b,
                        nir_swizzle(b, dir, yx, 2),
                        nir_imm_vec2(b, 1.0, -1.0)),
               vp_scale_rcp);
   tangent = nir_fmul(b, tangent, half_width);
   tangent = nir_pad_vector_imm_int(b, tangent, 0, 4);
   dir = nir_fmul_imm(b, nir_fmul(b, dir, vp_scale_rcp), 0.5);

   nir_def *line_offets[8] = {
      nir_fadd(b, tangent, nir_fneg(b, dir)),
      nir_fadd(b, nir_fneg(b, tangent), nir_fneg(b, dir)),
      tangent,
      nir_fneg(b, tangent),
      tangent,
      nir_fneg(b, tangent),
      nir_fadd(b, tangent, dir),
      nir_fadd(b, nir_fneg(b, tangent), dir),
   };
   nir_def *line_coord =
      nir_vec4(b, half_width, half_width, half_length, half_length);
   nir_def *line_coords[8] = {
      nir_fmul(b, line_coord, nir_imm_vec4(b, -1,  1,  -1,  1)),
      nir_fmul(b, line_coord, nir_imm_vec4(b,  1,  1,  -1,  1)),
      nir_fmul(b, line_coord, nir_imm_vec4(b, -1,  1,   0,  1)),
      nir_fmul(b, line_coord, nir_imm_vec4(b,  1,  1,   0,  1)),
      nir_fmul(b, line_coord, nir_imm_vec4(b, -1,  1,   0,  1)),
      nir_fmul(b, line_coord, nir_imm_vec4(b,  1,  1,   0,  1)),
      nir_fmul(b, line_coord, nir_imm_vec4(b, -1,  1,   1,  1)),
      nir_fmul(b, line_coord, nir_imm_vec4(b,  1,  1,   1,  1)),
   };

   /* emit first end-cap, and start line */
   for (int i = 0; i < 4; ++i) {
      nir_foreach_variable_with_modes(var, b->shader, nir_var_shader_out) {
         gl_varying_slot location = var->data.location;
         unsigned location_frac = var->data.location_frac;
         if (state->prev_varyings[location][location_frac])
            nir_copy_var(b, var, state->prev_varyings[location][location_frac]);
      }
      nir_store_var(b, state->pos_out,
                    nir_fadd(b, prev, nir_fmul(b, line_offets[i],
                             nir_channel(b, prev, 3))), 0xf);
      nir_store_var(b, state->line_coord_out, line_coords[i], 0xf);
      nir_emit_vertex(b);
   }

   /* finish line and emit last end-cap */
   for (int i = 4; i < 8; ++i) {
      nir_foreach_variable_with_modes(var, b->shader, nir_var_shader_out) {
         gl_varying_slot location = var->data.location;
         unsigned location_frac = var->data.location_frac;
         if (state->varyings[location][location_frac])
            nir_copy_var(b, var, state->varyings[location][location_frac]);
      }
      nir_store_var(b, state->pos_out,
                    nir_fadd(b, curr, nir_fmul(b, line_offets[i],
                             nir_channel(b, curr, 3))), 0xf);
      nir_store_var(b, state->line_coord_out, line_coords[i], 0xf);
      nir_emit_vertex(b);
   }
   nir_end_primitive(b);

   nir_pop_if(b, NULL);

   nir_copy_var(b, state->prev_pos, state->pos_out);
   nir_foreach_variable_with_modes(var, b->shader, nir_var_shader_out) {
      gl_varying_slot location = var->data.location;
      unsigned location_frac = var->data.location_frac;
      if (state->varyings[location][location_frac])
         nir_copy_var(b, state->prev_varyings[location][location_frac], state->varyings[location][location_frac]);
   }

   // update prev_pos and pos_counter for next vertex
   b->cursor = nir_after_instr(&intrin->instr);
   nir_store_var(b, state->pos_counter,
                    nir_iadd_imm(b, nir_load_var(b, state->pos_counter),
                                    1), 1);

   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
lower_line_smooth_gs_end_primitive(nir_builder *b,
                                   nir_intrinsic_instr *intrin,
                                   struct lower_line_smooth_state *state)
{
   b->cursor = nir_before_instr(&intrin->instr);

   // reset line counter
   nir_store_var(b, state->pos_counter, nir_imm_int(b, 0), 1);

   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
lower_line_smooth_gs_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   struct lower_line_smooth_state *state = data;
   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_store_deref:
      return lower_line_smooth_gs_store(b, intrin, state);
   case nir_intrinsic_copy_deref:
      UNREACHABLE("should be lowered");
   case nir_intrinsic_emit_vertex_with_counter:
   case nir_intrinsic_emit_vertex:
      return lower_line_smooth_gs_emit_vertex(b, intrin, state);
   case nir_intrinsic_end_primitive:
   case nir_intrinsic_end_primitive_with_counter:
      return lower_line_smooth_gs_end_primitive(b, intrin, state);
   default:
      return false;
   }
}

static bool
lower_line_smooth_gs(nir_shader *shader)
{
   nir_builder b;
   struct lower_line_smooth_state state;

   memset(state.varyings, 0, sizeof(state.varyings));
   memset(state.prev_varyings, 0, sizeof(state.prev_varyings));
   nir_foreach_variable_with_modes(var, shader, nir_var_shader_out) {
      gl_varying_slot location = var->data.location;
      unsigned location_frac = var->data.location_frac;
      if (location == VARYING_SLOT_POS)
         continue;

      char name[100];
      snprintf(name, sizeof(name), "__tmp_%d_%d", location, location_frac);
      state.varyings[location][location_frac] =
         nir_variable_create(shader, nir_var_shader_temp,
                              var->type, name);

      snprintf(name, sizeof(name), "__tmp_prev_%d_%d", location, location_frac);
      state.prev_varyings[location][location_frac] =
         nir_variable_create(shader, nir_var_shader_temp,
                              var->type, name);
   }

   state.pos_out =
      nir_find_variable_with_location(shader, nir_var_shader_out,
                                      VARYING_SLOT_POS);

   // if position isn't written, we have nothing to do
   if (!state.pos_out)
      return false;

   unsigned location = 0;
   nir_foreach_shader_in_variable(var, shader) {
     if (var->data.driver_location >= location)
         location = var->data.driver_location + 1;
   }

   state.line_coord_out =
      nir_variable_create(shader, nir_var_shader_out, glsl_vec4_type(),
                          "__line_coord");
   state.line_coord_out->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
   state.line_coord_out->data.driver_location = location;
   state.line_coord_out->data.location = MAX2(util_last_bit64(shader->info.outputs_written), VARYING_SLOT_VAR0);
   shader->info.outputs_written |= BITFIELD64_BIT(state.line_coord_out->data.location);
   shader->num_outputs++;

   // create temp variables
   state.prev_pos = nir_variable_create(shader, nir_var_shader_temp,
                                        glsl_vec4_type(),
                                        "__prev_pos");
   state.pos_counter = nir_variable_create(shader, nir_var_shader_temp,
                                           glsl_uint_type(),
                                           "__pos_counter");

   // initialize pos_counter
   nir_function_impl *entry = nir_shader_get_entrypoint(shader);
   b = nir_builder_at(nir_before_impl(entry));
   nir_store_var(&b, state.pos_counter, nir_imm_int(&b, 0), 1);

   shader->info.gs.vertices_out = 8 * shader->info.gs.vertices_out;
   shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;

   return nir_shader_instructions_pass(shader, lower_line_smooth_gs_instr,
                                       nir_metadata_dominance, &state);
}

static bool
lower_line_smooth_fs(nir_shader *shader, bool lower_stipple)
{
   int dummy;
   nir_builder b;

   nir_variable *stipple_counter = NULL, *stipple_pattern = NULL;
   if (lower_stipple) {
      stipple_counter = nir_variable_create(shader, nir_var_shader_in,
                                            glsl_float_type(),
                                            "__stipple");
      stipple_counter->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
      stipple_counter->data.driver_location = shader->num_inputs++;
      stipple_counter->data.location =
         MAX2(util_last_bit64(shader->info.inputs_read), VARYING_SLOT_VAR0);
      shader->info.inputs_read |= BITFIELD64_BIT(stipple_counter->data.location);

      stipple_pattern = nir_variable_create(shader, nir_var_shader_temp,
                                            glsl_uint_type(),
                                            "stipple_pattern");

      // initialize stipple_pattern
      nir_function_impl *entry = nir_shader_get_entrypoint(shader);
      b = nir_builder_at(nir_before_impl(entry));
      nir_def *pattern = nir_load_push_constant_zink(&b, 1, 32,
                                                         nir_imm_int(&b, ZINK_GFX_PUSHCONST_LINE_STIPPLE_PATTERN));
      nir_store_var(&b, stipple_pattern, pattern, 1);
   }

   nir_lower_aaline_fs(shader, &dummy, stipple_counter, stipple_pattern);
   return true;
}

static bool
lower_dual_blend(nir_shader *shader)
{
   bool progress = false;
   nir_variable *var = nir_find_variable_with_location(shader, nir_var_shader_out, FRAG_RESULT_DATA1);
   if (var) {
      var->data.location = FRAG_RESULT_DATA0;
      var->data.index = 1;
      progress = true;
   }
   nir_shader_preserve_all_metadata(shader);
   return progress;
}

static bool
lower_64bit_pack_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu_instr = (nir_alu_instr *) instr;
   if (alu_instr->op != nir_op_pack_64_2x32 &&
       alu_instr->op != nir_op_unpack_64_2x32)
      return false;
   b->cursor = nir_before_instr(&alu_instr->instr);
   nir_def *src = nir_ssa_for_alu_src(b, alu_instr, 0);
   nir_def *dest;
   switch (alu_instr->op) {
   case nir_op_pack_64_2x32:
      dest = nir_pack_64_2x32_split(b, nir_channel(b, src, 0), nir_channel(b, src, 1));
      break;
   case nir_op_unpack_64_2x32:
      dest = nir_vec2(b, nir_unpack_64_2x32_split_x(b, src), nir_unpack_64_2x32_split_y(b, src));
      break;
   default:
      UNREACHABLE("Impossible opcode");
   }
   nir_def_replace(&alu_instr->def, dest);
   return true;
}

static bool
lower_64bit_pack(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_64bit_pack_instr,
                                       nir_metadata_control_flow, NULL);
}

nir_shader *
zink_create_quads_emulation_gs(const nir_shader_compiler_options *options,
                               const nir_shader *prev_stage)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
                                                  options,
                                                  "filled quad gs");

   nir_shader *nir = b.shader;
   nir->info.gs.input_primitive = MESA_PRIM_LINES_ADJACENCY;
   nir->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   nir->info.gs.vertices_in = 4;
   nir->info.gs.vertices_out = 6;
   nir->info.gs.invocations = 1;
   nir->info.gs.active_stream_mask = 1;

   nir->info.has_transform_feedback_varyings = prev_stage->info.has_transform_feedback_varyings;
   memcpy(nir->info.xfb_stride, prev_stage->info.xfb_stride, sizeof(prev_stage->info.xfb_stride));
   if (prev_stage->xfb_info) {
      size_t size = nir_xfb_info_size(prev_stage->xfb_info->output_count);
      nir->xfb_info = ralloc_memdup(nir, prev_stage->xfb_info, size);
   }

   nir_variable *in_vars[VARYING_SLOT_MAX];
   nir_variable *out_vars[VARYING_SLOT_MAX];
   unsigned num_vars = 0;

   /* Create input/output variables. */
   nir_foreach_shader_out_variable(var, prev_stage) {
      assert(!var->data.patch);
      assert(var->data.location != VARYING_SLOT_PRIMITIVE_ID &&
            "not a VS output");

      /* input vars can't be created for those */
      if (var->data.location == VARYING_SLOT_LAYER ||
          var->data.location == VARYING_SLOT_VIEW_INDEX ||
          /* psiz not needed for quads */
          var->data.location == VARYING_SLOT_PSIZ)
         continue;

      char name[100];
      if (var->name)
         snprintf(name, sizeof(name), "in_%s", var->name);
      else
         snprintf(name, sizeof(name), "in_%d", var->data.driver_location);

      nir_variable *in = nir_variable_clone(var, nir);
      ralloc_free(in->name);
      in->name = ralloc_strdup(in, name);
      in->type = glsl_array_type(var->type, 4, false);
      in->data.mode = nir_var_shader_in;
      nir_shader_add_variable(nir, in);

      if (var->name)
         snprintf(name, sizeof(name), "out_%s", var->name);
      else
         snprintf(name, sizeof(name), "out_%d", var->data.driver_location);

      nir_variable *out = nir_variable_clone(var, nir);
      ralloc_free(out->name);
      out->name = ralloc_strdup(out, name);
      out->data.mode = nir_var_shader_out;
      nir_shader_add_variable(nir, out);

      in_vars[num_vars] = in;
      out_vars[num_vars++] = out;
   }

   /* When a geometry shader is not used, a fragment shader may read primitive
    * ID and get an implicit value without the vertex shader writing an ID. This
    * case needs to work even when we inject a GS internally.
    *
    * However, if a geometry shader precedes a fragment shader that reads
    * primitive ID, Vulkan requires that the geometry shader write primitive ID.
    * To handle this case correctly, we must write primitive ID, copying the
    * fixed-function gl_PrimitiveIDIn input which matches what the fragment
    * shader will expect.
    *
    * If the fragment shader doesn't read primitive ID, this copy will likely be
    * optimized out at link-time by the Vulkan driver. Unless this is
    * non-monolithic -- in which case we don't know whether the fragment shader
    * will read primitive ID either. In both cases, the right thing for Zink
    * to do is copy primitive ID unconditionally.
    */
   in_vars[num_vars] = nir_create_variable_with_location(
         nir, nir_var_shader_in, VARYING_SLOT_PRIMITIVE_ID, glsl_int_type());

   out_vars[num_vars] = nir_create_variable_with_location(
         nir, nir_var_shader_out, VARYING_SLOT_PRIMITIVE_ID, glsl_int_type());

   num_vars++;

   int mapping_first[] = {0, 1, 2, 0, 2, 3};
   int mapping_last[] = {0, 1, 3, 1, 2, 3};
   nir_def *last_pv_vert_def = nir_load_provoking_last(&b);
   last_pv_vert_def = nir_ine_imm(&b, last_pv_vert_def, 0);
   for (unsigned i = 0; i < 6; ++i) {
      /* swap indices 2 and 3 */
      nir_def *idx = nir_bcsel(&b, last_pv_vert_def,
                                   nir_imm_int(&b, mapping_last[i]),
                                   nir_imm_int(&b, mapping_first[i]));
      /* Copy inputs to outputs. */
      for (unsigned j = 0; j < num_vars; ++j) {
         if (in_vars[j]->data.location == VARYING_SLOT_EDGE) {
            continue;
         }

         /* gl_PrimitiveIDIn is not arrayed, all other inputs are */
         nir_deref_instr *in_value = nir_build_deref_var(&b, in_vars[j]);
         if (in_vars[j]->data.location != VARYING_SLOT_PRIMITIVE_ID)
            in_value = nir_build_deref_array(&b, in_value, idx);

         copy_vars(&b, nir_build_deref_var(&b, out_vars[j]), in_value);
      }
      nir_emit_vertex(&b, 0);
      if (i == 2)
        nir_end_primitive(&b, 0);
   }

   nir_end_primitive(&b, 0);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir_validate_shader(nir, "in zink_create_quads_emulation_gs");
   return nir;
}

static bool
lower_system_values_to_inlined_uniforms_instr(nir_builder *b,
                                              nir_intrinsic_instr *intrin,
                                              void *data)
{
   int inlined_uniform_offset;
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_flat_mask:
      inlined_uniform_offset = ZINK_INLINE_VAL_FLAT_MASK * sizeof(uint32_t);
      break;
   case nir_intrinsic_load_provoking_last:
      inlined_uniform_offset = ZINK_INLINE_VAL_PV_LAST_VERT * sizeof(uint32_t);
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intrin->instr);
   assert(intrin->def.bit_size == 32 || intrin->def.bit_size == 64);
   /* nir_inline_uniforms can't handle bit_size != 32 (it will silently ignore
    * anything with a different bit_size) so we need to split the load. */
   int num_dwords = intrin->def.bit_size / 32;
   nir_def *dwords[2] = {NULL};
   for (unsigned i = 0; i < num_dwords; i++)
      dwords[i] = nir_load_ubo(b, 1, 32, nir_imm_int(b, 0),
                                   nir_imm_int(b, inlined_uniform_offset + i),
                                   .align_mul = intrin->def.bit_size / 8,
                                   .align_offset = 0,
                                   .range_base = 0, .range = ~0);
   nir_def *new_dest_def;
   if (intrin->def.bit_size == 32)
      new_dest_def = dwords[0];
   else
      new_dest_def = nir_pack_64_2x32_split(b, dwords[0], dwords[1]);
   nir_def_replace(&intrin->def, new_dest_def);
   return true;
}

bool
zink_lower_system_values_to_inlined_uniforms(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir,
                                       lower_system_values_to_inlined_uniforms_instr,
                                       nir_metadata_dominance, NULL);
}

/* from radeonsi */
static unsigned
amd_varying_expression_max_cost(nir_shader *producer, nir_shader *consumer)
{
   /* TODO: maybe implement shader profiles to disable, cf. 39804ebf1766d38004259085e1fec4ed8db86f1c */

   switch (consumer->info.stage) {
   case MESA_SHADER_TESS_CTRL: /* VS->TCS */
      /* Non-amplifying shaders can always have their variyng expressions
       * moved into later shaders.
       */
      return UINT_MAX;

   case MESA_SHADER_GEOMETRY: /* VS->GS, TES->GS */
      return consumer->info.gs.vertices_in == 1 ? UINT_MAX :
             consumer->info.gs.vertices_in == 2 ? 20 : 14;

   case MESA_SHADER_TESS_EVAL: /* VS->TES, TCS->TES */
   case MESA_SHADER_FRAGMENT:
      /* Up to 3 uniforms and 5 ALUs. */
      return 14;

   default:
      UNREACHABLE("unexpected shader stage");
   }
}

void
zink_screen_init_compiler(struct zink_screen *screen)
{
   static const struct nir_shader_compiler_options
   default_options = {
      .io_options = nir_io_has_intrinsics | nir_io_separate_clip_cull_distance_arrays,
      .lower_ffma16 = true,
      .lower_ffma32 = true,
      .lower_ffma64 = true,
      .lower_scmp = true,
      .lower_fdph = true,
      .lower_flrp32 = true,
      .lower_fsat = true,
      .lower_hadd = true,
      .lower_iadd_sat = true,
      .lower_fisnormal = true,
      .lower_extract_byte = true,
      .lower_extract_word = true,
      .lower_insert_byte = true,
      .lower_insert_word = true,

      /* We can only support 32-bit ldexp, but NIR doesn't have a flag
       * distinguishing 64-bit ldexp support (radeonsi *does* support 64-bit
       * ldexp, so we don't just always lower it in NIR).  Given that ldexp is
       * effectively unused (no instances in shader-db), it's not worth the
       * effort to do so.
       * */
      .lower_ldexp = true,

      .lower_mul_high = true,
      .lower_to_scalar = true,
      .lower_uadd_carry = true,
      .compact_arrays = true,
      .lower_usub_borrow = true,
      .lower_uadd_sat = true,
      .lower_usub_sat = true,
      .lower_vector_cmp = true,
      .lower_int64_options =
         nir_lower_bit_count64 |
         nir_lower_find_lsb64 |
         nir_lower_ufind_msb64,
      .lower_doubles_options = nir_lower_dround_even,
      .lower_uniforms_to_ubo = true,
      .has_fsub = true,
      .has_isub = true,
      .lower_mul_2x32_64 = true,
      .support_16bit_alu = true, /* not quite what it sounds like */
      .max_unroll_iterations = 0,
   };

   screen->nir_options = default_options;

   if (!screen->info.feats.features.shaderInt64)
      screen->nir_options.lower_int64_options = ~0;

   if (!screen->info.feats.features.shaderFloat64) {
      screen->nir_options.lower_doubles_options = ~0;
      screen->nir_options.lower_flrp64 = true;
      screen->nir_options.lower_ffma64 = true;
      /* soft fp64 function inlining will blow up loop bodies and effectively
       * stop Vulkan drivers from unrolling the loops.
       */
      screen->nir_options.max_unroll_iterations_fp64 = 32;
   }

   /* XXX: do any drivers need different estimates? */
   screen->nir_options.varying_expression_max_cost = amd_varying_expression_max_cost;

   /*
       The OpFRem and OpFMod instructions use cheap approximations of remainder,
       and the error can be large due to the discontinuity in trunc() and floor().
       This can produce mathematically unexpected results in some cases, such as
       FMod(x,x) computing x rather than 0, and can also cause the result to have
       a different sign than the infinitely precise result.

       -Table 84. Precision of core SPIR-V Instructions
       * for drivers that are known to have imprecise fmod for doubles, lower dmod
    */
   if (zink_driverid(screen) == VK_DRIVER_ID_MESA_RADV ||
       zink_driverid(screen) == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
       zink_driverid(screen) == VK_DRIVER_ID_AMD_PROPRIETARY)
      screen->nir_options.lower_doubles_options = nir_lower_dmod;

   if (screen->info.have_EXT_shader_demote_to_helper_invocation &&
       !screen->driver_compiler_workarounds.broken_demote)
      screen->nir_options.discard_is_demote = true;

   if (!screen->info.have_KHR_maintenance9) {
      screen->nir_options.lower_bitfield_extract16 = true;
      screen->nir_options.lower_bitfield_extract8 = true;
      screen->nir_options.lower_int64_options |=
         nir_lower_bitfield_reverse64 | nir_lower_bitfield_extract64;
   }

   screen->nir_options.support_indirect_inputs = BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
                                                 BITFIELD_BIT(MESA_SHADER_TESS_EVAL) |
                                                 BITFIELD_BIT(MESA_SHADER_FRAGMENT);
   screen->nir_options.support_indirect_outputs = (uint8_t)BITFIELD_MASK(PIPE_SHADER_TYPES);
}

struct nir_shader *
zink_tgsi_to_nir(struct pipe_screen *screen, const struct tgsi_token *tokens)
{
   if (zink_debug & ZINK_DEBUG_TGSI) {
      fprintf(stderr, "TGSI shader:\n---8<---\n");
      tgsi_dump_to_file(tokens, 0, stderr);
      fprintf(stderr, "---8<---\n\n");
   }

   return tgsi_to_nir(tokens, screen, false);
}


static bool
def_is_64bit(nir_def *def, void *state)
{
   bool *lower = (bool *)state;
   if (def && (def->bit_size == 64)) {
      *lower = true;
      return false;
   }
   return true;
}

static bool
src_is_64bit(nir_src *src, void *state)
{
   bool *lower = (bool *)state;
   if (src && (nir_src_bit_size(*src) == 64)) {
      *lower = true;
      return false;
   }
   return true;
}

static bool
filter_64_bit_instr(const nir_instr *const_instr, UNUSED const void *data)
{
   bool lower = false;
   /* lower_alu_to_scalar required nir_instr to be const, but nir_foreach_*
    * doesn't have const variants, so do the ugly const_cast here. */
   nir_instr *instr = (nir_instr *)const_instr;

   nir_foreach_def(instr, def_is_64bit, &lower);
   if (lower)
      return true;
   nir_foreach_src(instr, src_is_64bit, &lower);
   return lower;
}

static bool
filter_pack_instr(const nir_instr *const_instr, UNUSED const void *data)
{
   nir_instr *instr = (nir_instr *)const_instr;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   switch (alu->op) {
   case nir_op_pack_64_2x32_split:
   case nir_op_pack_32_2x16_split:
   case nir_op_unpack_32_2x16_split_x:
   case nir_op_unpack_32_2x16_split_y:
   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_64_2x32_split_y:
      return true;
   default:
      break;
   }
   return false;
}


struct bo_vars {
   nir_variable *uniforms[5];
   nir_variable *ubo[5];
   nir_variable *ssbo[5];
   uint32_t first_ubo;
   uint32_t first_ssbo;
};

static struct bo_vars
get_bo_vars(struct zink_shader *zs, nir_shader *shader)
{
   struct bo_vars bo;
   memset(&bo, 0, sizeof(bo));
   if (zs->ubos_used)
      bo.first_ubo = ffs(zs->ubos_used & ~BITFIELD_BIT(0)) - 2;
   assert(bo.first_ssbo < PIPE_MAX_CONSTANT_BUFFERS);
   if (zs->ssbos_used)
      bo.first_ssbo = ffs(zs->ssbos_used) - 1;
   assert(bo.first_ssbo < PIPE_MAX_SHADER_BUFFERS);
   nir_foreach_variable_with_modes(var, shader, nir_var_mem_ssbo | nir_var_mem_ubo) {
      unsigned idx = glsl_get_explicit_stride(glsl_get_struct_field(glsl_without_array(var->type), 0)) >> 1;
      if (var->data.mode == nir_var_mem_ssbo) {
         assert(!bo.ssbo[idx]);
         bo.ssbo[idx] = var;
      } else {
         if (var->data.driver_location) {
            assert(!bo.ubo[idx]);
            bo.ubo[idx] = var;
         } else {
            assert(!bo.uniforms[idx]);
            bo.uniforms[idx] = var;
         }
      }
   }
   return bo;
}

static bool
bound_bo_access_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct bo_vars *bo = data;
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   nir_variable *var = NULL;
   nir_def *offset = NULL;
   bool is_load = true;
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_store_ssbo:
      var = bo->ssbo[intr->def.bit_size >> 4];
      offset = intr->src[2].ssa;
      is_load = false;
      break;
   case nir_intrinsic_load_ssbo:
      var = bo->ssbo[intr->def.bit_size >> 4];
      offset = intr->src[1].ssa;
      break;
   case nir_intrinsic_load_ubo:
      if (nir_src_is_const(intr->src[0]) && nir_src_as_const_value(intr->src[0])->u32 == 0)
         var = bo->uniforms[intr->def.bit_size >> 4];
      else
         var = bo->ubo[intr->def.bit_size >> 4];
      offset = intr->src[1].ssa;
      break;
   default:
      return false;
   }
   nir_src offset_src = nir_src_for_ssa(offset);
   if (!nir_src_is_const(offset_src))
      return false;

   unsigned offset_bytes = nir_src_as_const_value(offset_src)->u32;
   const struct glsl_type *strct_type = glsl_get_array_element(var->type);
   unsigned size = glsl_array_size(glsl_get_struct_field(strct_type, 0));
   bool has_unsized = glsl_array_size(glsl_get_struct_field(strct_type, glsl_get_length(strct_type) - 1)) == 0;
   if (has_unsized || offset_bytes + intr->num_components - 1 < size)
      return false;

   unsigned rewrites = 0;
   nir_def *result[2];
   for (unsigned i = 0; i < intr->num_components; i++) {
      if (offset_bytes + i >= size) {
         rewrites++;
         if (is_load)
            result[i] = nir_imm_zero(b, 1, intr->def.bit_size);
      }
   }
   assert(rewrites == intr->num_components);
   if (is_load) {
      nir_def *load = nir_vec(b, result, intr->num_components);
      nir_def_rewrite_uses(&intr->def, load);
   }
   nir_instr_remove(instr);
   return true;
}

static bool
bound_bo_access(nir_shader *shader, struct zink_shader *zs)
{
   struct bo_vars bo = get_bo_vars(zs, shader);
   return nir_shader_instructions_pass(shader, bound_bo_access_instr, nir_metadata_dominance, &bo);
}

static void
optimize_nir(struct nir_shader *s, struct zink_shader *zs, bool can_shrink)
{
   bool progress;
   do {
      progress = false;
      if (s->options->lower_int64_options)
         NIR_PASS(_, s, nir_lower_int64);
      if (s->options->lower_doubles_options & nir_lower_fp64_full_software)
         NIR_PASS(_, s, lower_64bit_pack);
      NIR_PASS(_, s, nir_lower_vars_to_ssa);
      NIR_PASS(progress, s, nir_lower_alu_to_scalar, filter_pack_instr, NULL);
      NIR_PASS(progress, s, nir_opt_copy_prop_vars);
      NIR_PASS(progress, s, nir_copy_prop);
      NIR_PASS(progress, s, nir_opt_remove_phis);
      if (s->options->lower_int64_options) {
         NIR_PASS(progress, s, nir_lower_64bit_phis);
         NIR_PASS(progress, s, nir_lower_alu_to_scalar, filter_64_bit_instr, NULL);
      }
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      NIR_PASS(progress, s, nir_lower_phis_to_scalar, NULL, NULL);
      NIR_PASS(progress, s, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 8,
         .indirect_load_ok = true,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, s, nir_opt_peephole_select, &peephole_select_options);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_undef);
      NIR_PASS(progress, s, zink_nir_lower_b2b);
      if (zs)
         NIR_PASS(progress, s, bound_bo_access, zs);
      if (can_shrink)
         NIR_PASS(progress, s, nir_opt_shrink_vectors, false);
   } while (progress);

   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_algebraic_late);
      if (progress) {
         NIR_PASS(_, s, nir_copy_prop);
         NIR_PASS(_, s, nir_opt_dce);
         NIR_PASS(_, s, nir_opt_cse);
      }
   } while (progress);
}

/* - copy the lowered fbfetch variable
 * - set the new one up as an input attachment for descriptor 0.6
 * - load it as an image
 * - overwrite the previous load
 */
static bool
lower_fbfetch_instr(nir_builder *b, nir_instr *instr, void *data)
{
   bool ms = data != NULL;
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var->data.fb_fetch_output)
      return false;
   b->cursor = nir_after_instr(instr);
   nir_variable *fbfetch = nir_variable_clone(var, b->shader);
   /* If Dim is SubpassData, ... Image Format must be Unknown
    * - SPIRV OpTypeImage specification
    */
   fbfetch->data.image.format = 0;
   fbfetch->data.index = 0; /* fix this if more than 1 fbfetch target is supported */
   fbfetch->data.mode = nir_var_uniform;
   fbfetch->data.binding = ZINK_FBFETCH_BINDING;
   fbfetch->data.binding = ZINK_FBFETCH_BINDING;
   fbfetch->data.sample = ms;
   enum glsl_sampler_dim dim = ms ? GLSL_SAMPLER_DIM_SUBPASS_MS : GLSL_SAMPLER_DIM_SUBPASS;
   fbfetch->type = glsl_image_type(dim, false, GLSL_TYPE_FLOAT);
   nir_shader_add_variable(b->shader, fbfetch);
   nir_def *deref = &nir_build_deref_var(b, fbfetch)->def;
   nir_def *sample = ms ? nir_load_sample_id(b) : nir_undef(b, 1, 32);
   nir_def *load = nir_image_deref_load(b, 4, 32, deref, nir_imm_vec4(b, 0, 0, 0, 1), sample, nir_imm_int(b, 0));
   nir_def_rewrite_uses(&intr->def, load);
   return true;
}

static bool
lower_fbfetch(nir_shader *shader, nir_variable **fbfetch, bool ms)
{
   nir_foreach_shader_out_variable(var, shader) {
      if (var->data.fb_fetch_output) {
         *fbfetch = var;
         break;
      }
   }
   assert(*fbfetch);
   if (!*fbfetch)
      return false;
   return nir_shader_instructions_pass(shader, lower_fbfetch_instr, nir_metadata_dominance, (void*)ms);
}

/*
 * Add a check for out of bounds LOD for every texel fetch op
 * It boils down to:
 * - if (lod < query_levels(tex))
 * -    res = txf(tex)
 * - else
 * -    res = (0, 0, 0, 1)
 */
static bool
lower_txf_lod_robustness_instr(nir_builder *b, nir_tex_instr *txf, void *data)
{
   if (txf->op != nir_texop_txf)
      return false;

   b->cursor = nir_before_instr(&txf->instr);
   int lod_idx = nir_tex_instr_src_index(txf, nir_tex_src_lod);
   assert(lod_idx >= 0);
   nir_src lod_src = txf->src[lod_idx].src;
   if (nir_src_is_const(lod_src) && nir_src_as_const_value(lod_src)->u32 == 0)
      return false;

   nir_def *lod = lod_src.ssa;

   int offset_idx = nir_tex_instr_src_index(txf, nir_tex_src_texture_offset);
   int handle_idx = nir_tex_instr_src_index(txf, nir_tex_src_texture_handle);
   int deref_idx = nir_tex_instr_src_index(txf, nir_tex_src_texture_deref);
   nir_tex_instr *levels = nir_tex_instr_create(b->shader,
                                                1 + !!(offset_idx >= 0) + !!(handle_idx >= 0));
   unsigned src_idx = 0;
   levels->op = nir_texop_query_levels;
   levels->dest_type = nir_type_int | lod->bit_size;
   if (deref_idx >= 0) {
      levels->src[src_idx].src_type = nir_tex_src_texture_deref;
      levels->src[src_idx++].src = nir_src_for_ssa(txf->src[deref_idx].src.ssa);
   }
   if (offset_idx >= 0) {
      levels->src[src_idx].src_type = nir_tex_src_texture_offset;
      levels->src[src_idx++].src = nir_src_for_ssa(txf->src[offset_idx].src.ssa);
   }
   if (handle_idx >= 0) {
      levels->src[src_idx].src_type = nir_tex_src_texture_handle;
      levels->src[src_idx++].src = nir_src_for_ssa(txf->src[handle_idx].src.ssa);
   }
   nir_def_init(&levels->instr, &levels->def,
                nir_tex_instr_dest_size(levels), 32);
   nir_builder_instr_insert(b, &levels->instr);

   nir_if *lod_oob_if = nir_push_if(b, nir_ilt(b, lod, &levels->def));
   nir_tex_instr *new_txf = nir_instr_as_tex(nir_instr_clone(b->shader, &txf->instr));
   nir_builder_instr_insert(b, &new_txf->instr);

   nir_if *lod_oob_else = nir_push_else(b, lod_oob_if);
   nir_const_value oob_values[4] = {0};
   unsigned bit_size = nir_alu_type_get_type_size(txf->dest_type);
   oob_values[3] = (txf->dest_type & nir_type_float) ?
                   nir_const_value_for_float(1.0, bit_size) : nir_const_value_for_uint(1, bit_size);
   nir_def *oob_val = nir_build_imm(b, nir_tex_instr_dest_size(txf), bit_size, oob_values);

   nir_pop_if(b, lod_oob_else);
   nir_def *robust_txf = nir_if_phi(b, &new_txf->def, oob_val);

   nir_def_rewrite_uses(&txf->def, robust_txf);
   nir_instr_remove_v(&txf->instr);
   return true;
}

/* This pass is used to workaround the lack of out of bounds LOD robustness
 * for texel fetch ops in VK_EXT_image_robustness.
 */
static bool
lower_txf_lod_robustness(nir_shader *shader)
{
   return nir_shader_tex_pass(shader, lower_txf_lod_robustness_instr,
                              nir_metadata_none, NULL);
}

/* check for a genuine gl_PointSize output vs one from nir_lower_point_size_mov */
static bool
check_psiz(struct nir_shader *s)
{
   bool have_psiz = false;
   nir_foreach_shader_out_variable(var, s) {
      if (var->data.location == VARYING_SLOT_PSIZ) {
         /* genuine PSIZ outputs will have this set */
         have_psiz |= !!var->data.explicit_location;
      }
   }
   return have_psiz;
}

static nir_variable *
find_var_with_location_frac(nir_shader *nir, unsigned location, unsigned location_frac, bool have_psiz, nir_variable_mode mode)
{
   assert((int)location >= 0);

   nir_foreach_variable_with_modes(var, nir, mode) {
      if (var->data.location == location && (location != VARYING_SLOT_PSIZ || !have_psiz || var->data.explicit_location)) {
         unsigned num_components = glsl_get_vector_elements(var->type);
         if (glsl_type_is_64bit(glsl_without_array(var->type)))
            num_components *= 2;
         if (is_clipcull_dist(var->data.location))
            num_components = glsl_get_aoa_size(var->type);
         if (var->data.location_frac <= location_frac &&
               var->data.location_frac + num_components > location_frac)
            return var;
      }
   }
   return NULL;
}

static bool
is_inlined(const bool *inlined, const nir_xfb_output_info *output)
{
   unsigned num_components = util_bitcount(output->component_mask);
   for (unsigned i = 0; i < num_components; i++)
      if (!inlined[output->component_offset + i])
         return false;
   return true;
}

static void
update_psiz_location(nir_shader *nir, nir_variable *psiz)
{
   uint32_t last_output = util_last_bit64(nir->info.outputs_written);
   if (last_output < VARYING_SLOT_VAR0)
      last_output = VARYING_SLOT_VAR0;
   else
      last_output++;
   /* this should get fixed up by slot remapping */
   psiz->data.location = last_output;
}

static const struct glsl_type *
clamp_slot_type(const struct glsl_type *type, unsigned slot)
{
   /* could be dvec/dmat/mat: each member is the same */
   const struct glsl_type *plain = glsl_without_array_or_matrix(type);
   /* determine size of each member type */
   unsigned slot_count = glsl_count_vec4_slots(plain, false, false);
   /* normalize slot idx to current type's size */
   slot %= slot_count;
   unsigned slot_components = glsl_get_components(plain);
   if (glsl_base_type_is_64bit(glsl_get_base_type(plain)))
      slot_components *= 2;
   /* create a vec4 mask of the selected slot's components out of all the components */
   uint32_t mask = BITFIELD_MASK(slot_components) & BITFIELD_RANGE(slot * 4, 4);
   /* return a vecN of the selected components */
   slot_components = util_bitcount(mask);
   return glsl_vec_type(slot_components);
}

static const struct glsl_type *
unroll_struct_type(const struct glsl_type *slot_type, unsigned *slot_idx)
{
   const struct glsl_type *type = slot_type;
   unsigned slot_count = 0;
   unsigned cur_slot = 0;
   /* iterate over all the members in the struct, stopping once the slot idx is reached */
   for (unsigned i = 0; i < glsl_get_length(slot_type) && cur_slot <= *slot_idx; i++, cur_slot += slot_count) {
      /* use array type for slot counting but return array member type for unroll */
      const struct glsl_type *arraytype = glsl_get_struct_field(slot_type, i);
      type = glsl_without_array(arraytype);
      slot_count = glsl_count_vec4_slots(arraytype, false, false);
   }
   *slot_idx -= (cur_slot - slot_count);
   if (!glsl_type_is_struct_or_ifc(type))
      /* this is a fully unrolled struct: find the number of vec components to output */
      type = clamp_slot_type(type, *slot_idx);
   return type;
}

static unsigned
get_slot_components(nir_variable *var, unsigned slot, unsigned so_slot)
{
   assert(var && slot < var->data.location + glsl_count_vec4_slots(var->type, false, false));
   const struct glsl_type *orig_type = var->type;
   const struct glsl_type *type = glsl_without_array(var->type);
   unsigned slot_idx = slot - so_slot;
   if (type != orig_type)
      slot_idx %= glsl_count_vec4_slots(type, false, false);
   /* need to find the vec4 that's being exported by this slot */
   while (glsl_type_is_struct_or_ifc(type))
      type = unroll_struct_type(type, &slot_idx);

   /* arrays here are already fully unrolled from their structs, so slot handling is implicit */
   unsigned num_components = glsl_get_components(glsl_without_array(type));
   /* special handling: clip/cull distance are arrays with vector semantics */
   if (is_clipcull_dist(var->data.location)) {
      num_components = glsl_array_size(type);
      if (slot_idx)
         /* this is the second vec4 */
         num_components %= 4;
      else
         /* this is the first vec4 */
         num_components = MIN2(num_components, 4);
   }
   assert(num_components);
   /* gallium handles xfb in terms of 32bit units */
   if (glsl_base_type_is_64bit(glsl_get_base_type(glsl_without_array(type))))
      num_components *= 2;
   return num_components;
}

static unsigned
get_var_slot_count(nir_shader *nir, nir_variable *var)
{
   assert(var->data.mode == nir_var_shader_in || var->data.mode == nir_var_shader_out);
   const struct glsl_type *type = var->type;
   if (nir_is_arrayed_io(var, nir->info.stage))
      type = glsl_get_array_element(type);
   unsigned slot_count = 0;
   if ((nir->info.stage == MESA_SHADER_VERTEX && var->data.mode == nir_var_shader_in && var->data.location >= VERT_ATTRIB_GENERIC0) ||
       var->data.location >= VARYING_SLOT_VAR0)
      slot_count = glsl_count_vec4_slots(type, false, false);
   else if (glsl_type_is_array(type))
      slot_count = DIV_ROUND_UP(glsl_get_aoa_size(type), 4);
   else
      slot_count = 1;
   return slot_count;
}


static const nir_xfb_output_info *
find_packed_output(const nir_xfb_info *xfb_info, unsigned slot)
{
   for (unsigned i = 0; i < xfb_info->output_count; i++) {
      const nir_xfb_output_info *packed_output = &xfb_info->outputs[i];
      if (packed_output->location == slot)
         return packed_output;
   }
   return NULL;
}

static void
update_so_info(struct zink_shader *zs, nir_shader *nir, uint64_t outputs_written, bool have_psiz)
{
   bool inlined[VARYING_SLOT_MAX][4] = {0};
   uint64_t packed = 0;
   uint8_t packed_components[VARYING_SLOT_MAX] = {0};
   uint8_t packed_streams[VARYING_SLOT_MAX] = {0};
   uint8_t packed_buffers[VARYING_SLOT_MAX] = {0};
   uint16_t packed_offsets[VARYING_SLOT_MAX][4] = {0};
   for (unsigned i = 0; i < nir->xfb_info->output_count; i++) {
      const nir_xfb_output_info *output = &nir->xfb_info->outputs[i];
      unsigned xfb_components = util_bitcount(output->component_mask);
      /* always set stride to be used during draw */
      zs->sinfo.stride[output->buffer] = nir->xfb_info->buffers[output->buffer].stride;
      for (unsigned c = 0; !is_inlined(inlined[output->location], output) && c < xfb_components; c++) {
         unsigned slot = output->location;
         if (inlined[slot][output->component_offset + c])
            continue;
         nir_variable *var = NULL;
         while (!var && slot < VARYING_SLOT_TESS_MAX)
            var = find_var_with_location_frac(nir, slot--, output->component_offset + c, have_psiz, nir_var_shader_out);
         slot = output->location;
         unsigned slot_count = var ? get_var_slot_count(nir, var) : 0;
         if (!var || var->data.location > slot || var->data.location + slot_count <= slot) {
            /* if no variable is found for the xfb output, no output exists */
            inlined[slot][c + output->component_offset] = true;
            continue;
         }
         if (var->data.explicit_xfb_buffer) {
            /* handle dvec3 where gallium splits streamout over 2 registers */
            for (unsigned j = 0; j < xfb_components; j++)
               inlined[slot][c + output->component_offset + j] = true;
         }
         if (is_inlined(inlined[slot], output))
            continue;
         assert(!glsl_type_is_array(var->type) || is_clipcull_dist(var->data.location));
         assert(!glsl_type_is_struct_or_ifc(var->type));
         unsigned num_components = glsl_type_is_array(var->type) ? glsl_get_aoa_size(var->type) : glsl_get_vector_elements(var->type);
         if (glsl_type_is_64bit(glsl_without_array(var->type)))
            num_components *= 2;
         /* if this is the entire variable, try to blast it out during the initial declaration
         * structs must be handled later to ensure accurate analysis
         */
         if ((num_components == xfb_components ||
               num_components < xfb_components ||
               (num_components > xfb_components && xfb_components == 4))) {
            var->data.explicit_xfb_buffer = 1;
            var->data.xfb.buffer = output->buffer;
            var->data.xfb.stride = zs->sinfo.stride[output->buffer];
            var->data.offset = (output->offset + c * sizeof(uint32_t));
            var->data.stream = nir->xfb_info->buffer_to_stream[output->buffer];
            for (unsigned j = 0; j < MIN2(num_components, xfb_components); j++)
               inlined[slot][c + output->component_offset + j] = true;
         } else {
            /* otherwise store some metadata for later */
            packed |= BITFIELD64_BIT(slot);
            packed_components[slot] += xfb_components;
            packed_streams[slot] |= BITFIELD_BIT(nir->xfb_info->buffer_to_stream[output->buffer]);
            packed_buffers[slot] |= BITFIELD_BIT(output->buffer);
            for (unsigned j = 0; j < xfb_components; j++)
               packed_offsets[output->location][j + output->component_offset + c] = output->offset + j * sizeof(uint32_t);
         }
      }
   }

   /* if this was flagged as a packed output before, and if all the components are
    * being output with the same stream on the same buffer with increasing offsets, this entire variable
    * can be consolidated into a single output to conserve locations
    */
   for (unsigned i = 0; i < nir->xfb_info->output_count; i++) {
      const nir_xfb_output_info *output = &nir->xfb_info->outputs[i];
      unsigned slot = output->location;
      if (is_inlined(inlined[slot], output))
         continue;
      nir_variable *var = NULL;
      while (!var)
         var = find_var_with_location_frac(nir, slot--, output->component_offset, have_psiz, nir_var_shader_out);
      slot = output->location;
      unsigned slot_count = var ? get_var_slot_count(nir, var) : 0;
      if (!var || var->data.location > slot || var->data.location + slot_count <= slot)
         continue;
      /* this is a lowered 64bit variable that can't be exported due to packing */
      if (var->data.is_xfb)
         goto out;

      unsigned num_slots = is_clipcull_dist(var->data.location) ?
                           glsl_array_size(var->type) / 4 :
                           glsl_count_vec4_slots(var->type, false, false);
      /* for each variable, iterate over all the variable's slots and inline the outputs */
      for (unsigned j = 0; j < num_slots; j++) {
         slot = var->data.location + j;
         const nir_xfb_output_info *packed_output = find_packed_output(nir->xfb_info, slot);
         if (!packed_output)
            goto out;

         /* if this slot wasn't packed or isn't in the same stream/buffer, skip consolidation */
         if (!(packed & BITFIELD64_BIT(slot)) ||
               util_bitcount(packed_streams[slot]) != 1 ||
               util_bitcount(packed_buffers[slot]) != 1)
            goto out;

         /* if all the components the variable exports to this slot aren't captured, skip consolidation */
         unsigned num_components = get_slot_components(var, slot, var->data.location);
         if (num_components != packed_components[slot])
            goto out;

         /* in order to pack the xfb output, all the offsets must be sequentially incrementing */
         uint32_t prev_offset = packed_offsets[packed_output->location][0];
         for (unsigned k = 1; k < num_components; k++) {
            /* if the offsets are not incrementing as expected, skip consolidation */
            if (packed_offsets[packed_output->location][k] != prev_offset + sizeof(uint32_t))
               goto out;
            prev_offset = packed_offsets[packed_output->location][k + packed_output->component_offset];
         }
      }
      /* this output can be consolidated: blast out all the data inlined */
      var->data.explicit_xfb_buffer = 1;
      var->data.xfb.buffer = output->buffer;
      var->data.xfb.stride = zs->sinfo.stride[output->buffer];
      var->data.offset = output->offset;
      var->data.stream = nir->xfb_info->buffer_to_stream[output->buffer];
      /* mark all slot components inlined to skip subsequent loop iterations */
      for (unsigned j = 0; j < num_slots; j++) {
         slot = var->data.location + j;
         for (unsigned k = 0; k < packed_components[slot]; k++)
            inlined[slot][k] = true;
         packed &= ~BITFIELD64_BIT(slot);
      }
      continue;
out:
      UNREACHABLE("xfb should be inlined by now!");
   }
}

struct decompose_state {
  nir_variable **split;
  bool needs_w;
};

static bool
lower_attrib(nir_builder *b, nir_instr *instr, void *data)
{
   struct decompose_state *state = data;
   nir_variable **split = state->split;
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (var != split[0])
      return false;
   unsigned num_components = glsl_get_vector_elements(split[0]->type);
   b->cursor = nir_after_instr(instr);
   nir_def *loads[4];
   for (unsigned i = 0; i < (state->needs_w ? num_components - 1 : num_components); i++)
      loads[i] = nir_load_deref(b, nir_build_deref_var(b, split[i+1]));
   if (state->needs_w) {
      /* oob load w comopnent to get correct value for int/float */
      loads[3] = nir_channel(b, loads[0], 3);
      loads[0] = nir_channel(b, loads[0], 0);
   }
   nir_def *new_load = nir_vec(b, loads, num_components);
   nir_def_rewrite_uses(&intr->def, new_load);
   nir_instr_remove_v(instr);
   return true;
}

static bool
decompose_attribs(nir_shader *nir, uint32_t decomposed_attrs, uint32_t decomposed_attrs_without_w)
{
   uint32_t bits = 0;
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in)
      bits |= BITFIELD_BIT(var->data.driver_location);
   bits = ~bits;
   u_foreach_bit(location, decomposed_attrs | decomposed_attrs_without_w) {
      nir_variable *split[5];
      struct decompose_state state;
      state.split = split;
      nir_variable *var = nir_find_variable_with_driver_location(nir, nir_var_shader_in, location);
      assert(var);
      split[0] = var;
      bits |= BITFIELD_BIT(var->data.driver_location);
      const struct glsl_type *new_type = glsl_type_is_scalar(var->type) ? var->type : glsl_get_array_element(var->type);
      unsigned num_components = glsl_get_vector_elements(var->type);
      state.needs_w = (decomposed_attrs_without_w & BITFIELD_BIT(location)) != 0 && num_components == 4;
      for (unsigned i = 0; i < (state.needs_w ? num_components - 1 : num_components); i++) {
         split[i+1] = nir_variable_clone(var, nir);
         split[i+1]->name = ralloc_asprintf(nir, "%s_split%u", var->name, i);
         if (decomposed_attrs_without_w & BITFIELD_BIT(location))
            split[i+1]->type = !i && num_components == 4 ? var->type : new_type;
         else
            split[i+1]->type = new_type;
         split[i+1]->data.driver_location = ffs(bits) - 1;
         bits &= ~BITFIELD_BIT(split[i+1]->data.driver_location);
         nir_shader_add_variable(nir, split[i+1]);
      }
      var->data.mode = nir_var_shader_temp;
      nir_shader_instructions_pass(nir, lower_attrib, nir_metadata_dominance, &state);
   }
   nir_fixup_deref_modes(nir);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_temp, NULL);
   optimize_nir(nir, NULL, true);
   return true;
}

static bool
rewrite_bo_access_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct zink_screen *screen = data;
   const bool has_int64 = screen->info.feats.features.shaderInt64;
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap: {
      /* convert offset to uintN_t[idx] */
      nir_def *offset = nir_udiv_imm(b, intr->src[1].ssa, intr->def.bit_size / 8);
      nir_src_rewrite(&intr->src[1], offset);
      return true;
   }
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ubo: {
      /* ubo0 can have unaligned 64bit loads, particularly for bindless texture ids */
      bool force_2x32 = intr->intrinsic == nir_intrinsic_load_ubo &&
                        nir_src_is_const(intr->src[0]) &&
                        nir_src_as_uint(intr->src[0]) == 0 &&
                        intr->def.bit_size == 64 &&
                        nir_intrinsic_align_offset(intr) % 8 != 0;
      force_2x32 |= intr->def.bit_size == 64 && !has_int64;
      nir_def *offset = nir_udiv_imm(b, intr->src[1].ssa, (force_2x32 ? 32 : intr->def.bit_size) / 8);
      nir_src_rewrite(&intr->src[1], offset);
      /* if 64bit isn't supported, 64bit loads definitely aren't supported, so rewrite as 2x32 with cast and pray */
      if (force_2x32) {
         /* this is always scalarized */
         assert(intr->def.num_components == 1);
         /* rewrite as 2x32 */
         nir_def *load[2];
         for (unsigned i = 0; i < 2; i++) {
            if (intr->intrinsic == nir_intrinsic_load_ssbo)
               load[i] = nir_load_ssbo(b, 1, 32, intr->src[0].ssa, nir_iadd_imm(b, intr->src[1].ssa, i), .align_mul = 4, .align_offset = 0);
            else
               load[i] = nir_load_ubo(b, 1, 32, intr->src[0].ssa, nir_iadd_imm(b, intr->src[1].ssa, i), .align_mul = 4, .align_offset = 0, .range = 4);
            nir_intrinsic_set_access(nir_instr_as_intrinsic(load[i]->parent_instr), nir_intrinsic_access(intr));
         }
         /* cast back to 64bit */
         nir_def *casted = nir_pack_64_2x32_split(b, load[0], load[1]);
         nir_def_rewrite_uses(&intr->def, casted);
         nir_instr_remove(instr);
      }
      return true;
   }
   case nir_intrinsic_load_scratch:
   case nir_intrinsic_load_shared: {
      b->cursor = nir_before_instr(instr);
      bool force_2x32 = intr->def.bit_size == 64 && !has_int64;
      nir_def *offset = nir_udiv_imm(b, intr->src[0].ssa, (force_2x32 ? 32 : intr->def.bit_size) / 8);
      nir_src_rewrite(&intr->src[0], offset);
      /* if 64bit isn't supported, 64bit loads definitely aren't supported, so rewrite as 2x32 with cast and pray */
      if (force_2x32) {
         /* this is always scalarized */
         assert(intr->def.num_components == 1);
         /* rewrite as 2x32 */
         nir_def *load[2];
         for (unsigned i = 0; i < 2; i++)
            load[i] = nir_load_shared(b, 1, 32, nir_iadd_imm(b, intr->src[0].ssa, i), .align_mul = 4, .align_offset = 0);
         /* cast back to 64bit */
         nir_def *casted = nir_pack_64_2x32_split(b, load[0], load[1]);
         nir_def_rewrite_uses(&intr->def, casted);
         nir_instr_remove(instr);
         return true;
      }
      break;
   }
   case nir_intrinsic_store_ssbo: {
      b->cursor = nir_before_instr(instr);
      bool force_2x32 = nir_src_bit_size(intr->src[0]) == 64 && !has_int64;
      nir_def *offset = nir_udiv_imm(b, intr->src[2].ssa, (force_2x32 ? 32 : nir_src_bit_size(intr->src[0])) / 8);
      nir_src_rewrite(&intr->src[2], offset);
      /* if 64bit isn't supported, 64bit loads definitely aren't supported, so rewrite as 2x32 with cast and pray */
      if (force_2x32) {
         /* this is always scalarized */
         assert(intr->src[0].ssa->num_components == 1);
         nir_def *vals[2] = {nir_unpack_64_2x32_split_x(b, intr->src[0].ssa), nir_unpack_64_2x32_split_y(b, intr->src[0].ssa)};
         for (unsigned i = 0; i < 2; i++)
            nir_store_ssbo(b, vals[i], intr->src[1].ssa, nir_iadd_imm(b, intr->src[2].ssa, i), .align_mul = 4, .align_offset = 0);
         nir_instr_remove(instr);
      }
      return true;
   }
   case nir_intrinsic_store_scratch:
   case nir_intrinsic_store_shared: {
      b->cursor = nir_before_instr(instr);
      bool force_2x32 = nir_src_bit_size(intr->src[0]) == 64 && !has_int64;
      nir_def *offset = nir_udiv_imm(b, intr->src[1].ssa, (force_2x32 ? 32 : nir_src_bit_size(intr->src[0])) / 8);
      nir_src_rewrite(&intr->src[1], offset);
      /* if 64bit isn't supported, 64bit loads definitely aren't supported, so rewrite as 2x32 with cast and pray */
      if (nir_src_bit_size(intr->src[0]) == 64 && !has_int64) {
         /* this is always scalarized */
         assert(intr->src[0].ssa->num_components == 1);
         nir_def *vals[2] = {nir_unpack_64_2x32_split_x(b, intr->src[0].ssa), nir_unpack_64_2x32_split_y(b, intr->src[0].ssa)};
         for (unsigned i = 0; i < 2; i++)
            nir_store_shared(b, vals[i], nir_iadd_imm(b, intr->src[1].ssa, i), .align_mul = 4, .align_offset = 0);
         nir_instr_remove(instr);
      }
      return true;
   }
   default:
      break;
   }
   return false;
}

static bool
rewrite_bo_access(nir_shader *shader, struct zink_screen *screen)
{
   return nir_shader_instructions_pass(shader, rewrite_bo_access_instr, nir_metadata_dominance, screen);
}

static nir_variable *
get_bo_var(nir_shader *shader, struct bo_vars *bo, bool ssbo, nir_src *src, unsigned bit_size)
{
   nir_variable *var, **ptr;
   unsigned idx = ssbo || (nir_src_is_const(*src) && !nir_src_as_uint(*src)) ? 0 : 1;

   if (ssbo)
      ptr = &bo->ssbo[bit_size >> 4];
   else {
      if (!idx) {
         ptr = &bo->uniforms[bit_size >> 4];
      } else
         ptr = &bo->ubo[bit_size >> 4];
   }
   var = *ptr;
   if (!var) {
      if (ssbo)
         var = bo->ssbo[32 >> 4];
      else {
         if (!idx)
            var = bo->uniforms[32 >> 4];
         else
            var = bo->ubo[32 >> 4];
      }
      var = nir_variable_clone(var, shader);
      if (ssbo)
         var->name = ralloc_asprintf(shader, "%s@%u", "ssbos", bit_size);
      else
         var->name = ralloc_asprintf(shader, "%s@%u", idx ? "ubos" : "uniform_0", bit_size);
      *ptr = var;
      nir_shader_add_variable(shader, var);

      struct glsl_struct_field *fields = rzalloc_array(shader, struct glsl_struct_field, 2);
      fields[0].name = ralloc_strdup(shader, "base");
      fields[1].name = ralloc_strdup(shader, "unsized");
      unsigned array_size = glsl_get_length(var->type);
      const struct glsl_type *bare_type = glsl_without_array(var->type);
      const struct glsl_type *array_type = glsl_get_struct_field(bare_type, 0);
      unsigned length = glsl_get_length(array_type);
      const struct glsl_type *type;
      const struct glsl_type *unsized = glsl_array_type(glsl_uintN_t_type(bit_size), 0, bit_size / 8);
      if (bit_size > 32) {
         assert(bit_size == 64);
         type = glsl_array_type(glsl_uintN_t_type(bit_size), length / 2, bit_size / 8);
      } else {
         type = glsl_array_type(glsl_uintN_t_type(bit_size), length * (32 / bit_size), bit_size / 8);
      }
      fields[0].type = type;
      fields[1].type = unsized;
      var->type = glsl_array_type(glsl_struct_type(fields, glsl_get_length(bare_type), "struct", false), array_size, 0);
      var->data.driver_location = idx;
   }
   return var;
}

static void
rewrite_atomic_ssbo_instr(nir_builder *b, nir_instr *instr, struct bo_vars *bo)
{
   nir_intrinsic_op op;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic == nir_intrinsic_ssbo_atomic)
      op = nir_intrinsic_deref_atomic;
   else if (intr->intrinsic == nir_intrinsic_ssbo_atomic_swap)
      op = nir_intrinsic_deref_atomic_swap;
   else
      UNREACHABLE("unknown intrinsic");
   nir_def *offset = intr->src[1].ssa;
   nir_src *src = &intr->src[0];
   nir_variable *var = get_bo_var(b->shader, bo, true, src,
                                  intr->def.bit_size);
   nir_deref_instr *deref_var = nir_build_deref_var(b, var);
   nir_def *idx = src->ssa;
   if (bo->first_ssbo)
      idx = nir_iadd_imm(b, idx, -bo->first_ssbo);
   nir_deref_instr *deref_array = nir_build_deref_array(b, deref_var, idx);
   nir_deref_instr *deref_struct = nir_build_deref_struct(b, deref_array, 0);

   /* generate new atomic deref ops for every component */
   nir_def *result[4];
   unsigned num_components = intr->def.num_components;
   for (unsigned i = 0; i < num_components; i++) {
      nir_deref_instr *deref_arr = nir_build_deref_array(b, deref_struct, offset);
      nir_intrinsic_instr *new_instr = nir_intrinsic_instr_create(b->shader, op);
      nir_def_init(&new_instr->instr, &new_instr->def, 1,
                   intr->def.bit_size);
      nir_intrinsic_set_atomic_op(new_instr, nir_intrinsic_atomic_op(intr));
      new_instr->src[0] = nir_src_for_ssa(&deref_arr->def);
      /* deref ops have no offset src, so copy the srcs after it */
      for (unsigned j = 2; j < nir_intrinsic_infos[intr->intrinsic].num_srcs; j++)
         new_instr->src[j - 1] = nir_src_for_ssa(intr->src[j].ssa);
      nir_builder_instr_insert(b, &new_instr->instr);

      result[i] = &new_instr->def;
      offset = nir_iadd_imm(b, offset, 1);
   }

   nir_def *load = nir_vec(b, result, num_components);
   nir_def_replace(&intr->def, load);
}

static bool
remove_bo_access_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct bo_vars *bo = data;
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   nir_variable *var = NULL;
   nir_def *offset = NULL;
   bool is_load = true;
   b->cursor = nir_before_instr(instr);
   nir_src *src;
   bool ssbo = true;
   switch (intr->intrinsic) {
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      rewrite_atomic_ssbo_instr(b, instr, bo);
      return true;
   case nir_intrinsic_store_ssbo:
      src = &intr->src[1];
      var = get_bo_var(b->shader, bo, true, src, nir_src_bit_size(intr->src[0]));
      offset = intr->src[2].ssa;
      is_load = false;
      break;
   case nir_intrinsic_load_ssbo:
      src = &intr->src[0];
      var = get_bo_var(b->shader, bo, true, src, intr->def.bit_size);
      offset = intr->src[1].ssa;
      break;
   case nir_intrinsic_load_ubo:
      src = &intr->src[0];
      var = get_bo_var(b->shader, bo, false, src, intr->def.bit_size);
      offset = intr->src[1].ssa;
      ssbo = false;
      break;
   default:
      return false;
   }
   assert(var);
   assert(offset);
   nir_deref_instr *deref_var = nir_build_deref_var(b, var);
   nir_def *idx = !ssbo && var->data.driver_location ? nir_iadd_imm(b, src->ssa, -1) : src->ssa;
   if (!ssbo && bo->first_ubo && var->data.driver_location)
      idx = nir_iadd_imm(b, idx, -bo->first_ubo);
   else if (ssbo && bo->first_ssbo)
      idx = nir_iadd_imm(b, idx, -bo->first_ssbo);
   nir_deref_instr *deref_array = nir_build_deref_array(b, deref_var,
                                                        nir_i2iN(b, idx, deref_var->def.bit_size));
   nir_deref_instr *deref_struct = nir_build_deref_struct(b, deref_array, 0);
   assert(intr->num_components <= 2);
   if (is_load) {
      nir_def *result[2];
      for (unsigned i = 0; i < intr->num_components; i++) {
         nir_deref_instr *deref_arr = nir_build_deref_array(b, deref_struct,
                                                            nir_i2iN(b, offset, deref_struct->def.bit_size));
         result[i] = nir_load_deref(b, deref_arr);
         if (intr->intrinsic == nir_intrinsic_load_ssbo)
            nir_intrinsic_set_access(nir_def_as_intrinsic(result[i]),
                                     nir_intrinsic_access(intr));
         offset = nir_iadd_imm(b, offset, 1);
      }
      nir_def *load = nir_vec(b, result, intr->num_components);
      nir_def_rewrite_uses(&intr->def, load);
   } else {
      nir_deref_instr *deref_arr = nir_build_deref_array(b, deref_struct,
                                                         nir_i2iN(b, offset, deref_struct->def.bit_size));
      nir_build_store_deref(b, &deref_arr->def, intr->src[0].ssa, BITFIELD_MASK(intr->num_components), nir_intrinsic_access(intr));
   }
   nir_instr_remove(instr);
   return true;
}

static bool
remove_bo_access(nir_shader *shader, struct zink_shader *zs)
{
   struct bo_vars bo = get_bo_vars(zs, shader);
   return nir_shader_instructions_pass(shader, remove_bo_access_instr, nir_metadata_dominance, &bo);
}

static bool
filter_io_instr(nir_intrinsic_instr *intr, bool *is_load, bool *is_input, bool *is_interp)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_interpolated_input:
      *is_interp = true;
      FALLTHROUGH;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
      *is_input = true;
      FALLTHROUGH;
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output:
      *is_load = true;
      FALLTHROUGH;
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_store_per_vertex_output:
      break;
   default:
      return false;
   }
   return true;
}

static bool
io_instr_is_arrayed(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output:
   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_store_per_vertex_output:
      return true;
   default:
      break;
   }
   return false;
}

static bool
find_var_deref(nir_shader *nir, nir_variable *var)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type == nir_deref_type_var && deref->var == var)
               return true;
         }
      }
   }
   return false;
}

static bool
find_var_io(nir_shader *nir, nir_variable *var)
{
   nir_foreach_function(function, nir) {
      if (!function->impl)
         continue;

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            bool is_load = false;
            bool is_input = false;
            bool is_interp = false;
            if (!filter_io_instr(intr, &is_load, &is_input, &is_interp))
               continue;
            if (var->data.mode == nir_var_shader_in && !is_input)
               continue;
            if (var->data.mode == nir_var_shader_out && is_input)
               continue;
            unsigned slot_offset = 0;
            if (var->data.fb_fetch_output && !is_load)
               continue;
            if (nir->info.stage == MESA_SHADER_FRAGMENT && !is_load && !is_input && nir_intrinsic_io_semantics(intr).dual_source_blend_index != var->data.index)
               continue;
            nir_src *src_offset = nir_get_io_offset_src(intr);
            if (src_offset && nir_src_is_const(*src_offset))
               slot_offset = nir_src_as_uint(*src_offset);
            unsigned slot_count = get_var_slot_count(nir, var);
            if (var->data.mode & (nir_var_shader_out | nir_var_shader_in) &&
                var->data.fb_fetch_output == nir_intrinsic_io_semantics(intr).fb_fetch_output &&
                var->data.location <= nir_intrinsic_io_semantics(intr).location + slot_offset &&
                var->data.location + slot_count > nir_intrinsic_io_semantics(intr).location + slot_offset)
               return true;
         }
      }
   }
   return false;
}

struct clamp_layer_output_state {
   nir_variable *original;
   nir_variable *clamped;
};

static void
clamp_layer_output_emit(nir_builder *b, struct clamp_layer_output_state *state)
{
   nir_def *is_layered = nir_load_push_constant_zink(b, 1, 32,
                                                         nir_imm_int(b, ZINK_GFX_PUSHCONST_FRAMEBUFFER_IS_LAYERED));
   nir_deref_instr *original_deref = nir_build_deref_var(b, state->original);
   nir_deref_instr *clamped_deref = nir_build_deref_var(b, state->clamped);
   nir_def *layer = nir_bcsel(b, nir_ieq_imm(b, is_layered, 1),
                                  nir_load_deref(b, original_deref),
                                  nir_imm_int(b, 0));
   nir_store_deref(b, clamped_deref, layer, 0);
}

static bool
clamp_layer_output_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct clamp_layer_output_state *state = data;
   switch (instr->type) {
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_emit_vertex_with_counter &&
          intr->intrinsic != nir_intrinsic_emit_vertex)
         return false;
      b->cursor = nir_before_instr(instr);
      clamp_layer_output_emit(b, state);
      return true;
   }
   default: return false;
   }
}

static bool
clamp_layer_output(nir_shader *vs, nir_shader *fs, unsigned *next_location)
{
   switch (vs->info.stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_TESS_EVAL:
      break;
   default:
      UNREACHABLE("invalid last vertex stage!");
   }
   struct clamp_layer_output_state state = {0};
   state.original = nir_find_variable_with_location(vs, nir_var_shader_out, VARYING_SLOT_LAYER);
   if (!state.original || (!find_var_deref(vs, state.original) && !find_var_io(vs, state.original)))
      return false;
   state.clamped = nir_variable_create(vs, nir_var_shader_out, glsl_int_type(), "layer_clamped");
   state.clamped->data.location = VARYING_SLOT_LAYER;
   nir_variable *fs_var = nir_find_variable_with_location(fs, nir_var_shader_in, VARYING_SLOT_LAYER);
   if ((state.original->data.explicit_xfb_buffer || fs_var) && *next_location < MAX_VARYING) {
      state.original->data.location = VARYING_SLOT_VAR0; // Anything but a built-in slot
      state.original->data.driver_location = (*next_location)++;
      if (fs_var) {
         fs_var->data.location = state.original->data.location;
         fs_var->data.driver_location = state.original->data.driver_location;
      }
   } else {
      if (state.original->data.explicit_xfb_buffer) {
         /* Will xfb the clamped output but still better than nothing */
         state.clamped->data.explicit_xfb_buffer = state.original->data.explicit_xfb_buffer;
         state.clamped->data.xfb.buffer = state.original->data.xfb.buffer;
         state.clamped->data.xfb.stride = state.original->data.xfb.stride;
         state.clamped->data.offset = state.original->data.offset;
         state.clamped->data.stream = state.original->data.stream;
      }
      state.original->data.mode = nir_var_shader_temp;
      nir_fixup_deref_modes(vs);
   }
   if (vs->info.stage == MESA_SHADER_GEOMETRY) {
      nir_shader_instructions_pass(vs, clamp_layer_output_instr, nir_metadata_dominance, &state);
   } else {
      nir_builder b;
      nir_function_impl *impl = nir_shader_get_entrypoint(vs);
      b = nir_builder_at(nir_after_impl(impl));
      assert(impl->end_block->predecessors->entries == 1);
      clamp_layer_output_emit(&b, &state);
      nir_progress(true, impl, nir_metadata_dominance);
   }
   optimize_nir(vs, NULL, true);
   NIR_PASS(_, vs, nir_remove_dead_variables, nir_var_shader_temp, NULL);
   return true;
}

struct io_slot_map {
   uint64_t *patch_slot_track;
   uint64_t *slot_track;
   unsigned char *slot_map;
   unsigned reserved;
   unsigned char *patch_slot_map;
   unsigned patch_reserved;
};

static void
assign_track_slot_mask(struct io_slot_map *io, nir_variable *var, unsigned slot, unsigned num_slots)
{
   uint64_t *track = var->data.patch ? io->patch_slot_track : io->slot_track;
   uint32_t mask = BITFIELD_MASK(glsl_get_vector_elements(glsl_without_array(var->type))) << var->data.location_frac;
   uint64_t slot_mask = BITFIELD64_RANGE(slot, num_slots);
   u_foreach_bit(c, mask) {
      assert((track[c] & slot_mask) == 0);
      track[c] |= slot_mask;
   }
}

static void
assign_slot_io(gl_shader_stage stage, struct io_slot_map *io, nir_variable *var, unsigned slot)
{
   unsigned num_slots;
   if (nir_is_arrayed_io(var, stage))
      num_slots = glsl_count_vec4_slots(glsl_get_array_element(var->type), false, false);
   else
      num_slots = glsl_count_vec4_slots(var->type, false, false);
   uint8_t *slot_map = var->data.patch ? io->patch_slot_map : io->slot_map;
   assign_track_slot_mask(io, var, slot, num_slots);
   if (slot_map[slot] != 0xff)
      return;
   unsigned *reserved = var->data.patch ? &io->patch_reserved : &io->reserved;
   assert(*reserved + num_slots <= MAX_VARYING);
   assert(*reserved < MAX_VARYING);
   for (unsigned i = 0; i < num_slots; i++)
      slot_map[slot + i] = (*reserved)++;
}

static void
assign_producer_var_io(gl_shader_stage stage, nir_variable *var, struct io_slot_map *io)
{
   unsigned slot = var->data.location;
   switch (slot) {
   case -1:
      UNREACHABLE("there should be no UINT32_MAX location variables!");
      break;
   case VARYING_SLOT_POS:
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_PRIMITIVE_ID:
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_FACE:
   case VARYING_SLOT_TESS_LEVEL_OUTER:
   case VARYING_SLOT_TESS_LEVEL_INNER:
      /* use a sentinel value to avoid counting later */
      var->data.driver_location = UINT32_MAX;
      return;

   default:
      break;
   }
   if (var->data.patch) {
      assert(slot >= VARYING_SLOT_PATCH0);
      slot -= VARYING_SLOT_PATCH0;
   }
   assign_slot_io(stage, io, var, slot);
   slot = var->data.patch ? io->patch_slot_map[slot] : io->slot_map[slot];
   assert(slot < MAX_VARYING);
   var->data.driver_location = slot;
}

ALWAYS_INLINE static bool
is_texcoord(gl_shader_stage stage, const nir_variable *var)
{
   if (stage != MESA_SHADER_FRAGMENT)
      return false;
   return var->data.location >= VARYING_SLOT_TEX0 && 
          var->data.location <= VARYING_SLOT_TEX7;
}

static bool
assign_consumer_var_io(gl_shader_stage stage, nir_variable *var, struct io_slot_map *io)
{
   unsigned slot = var->data.location;
   switch (slot) {
   case VARYING_SLOT_POS:
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_PRIMITIVE_ID:
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_FACE:
   case VARYING_SLOT_TESS_LEVEL_OUTER:
   case VARYING_SLOT_TESS_LEVEL_INNER:
      /* use a sentinel value to avoid counting later */
      var->data.driver_location = UINT_MAX;
      return true;
   default:
      break;
   }
   if (var->data.patch) {
      assert(slot >= VARYING_SLOT_PATCH0);
      slot -= VARYING_SLOT_PATCH0;
   }
   uint8_t *slot_map = var->data.patch ? io->patch_slot_map : io->slot_map;
   if (slot_map[slot] == (unsigned char)-1) {
      /* texcoords can't be eliminated in fs due to GL_COORD_REPLACE,
         * so keep for now and eliminate later
         */
      if (is_texcoord(stage, var)) {
         var->data.driver_location = UINT32_MAX;
         return true;
      }
      /* patch variables may be read in the workgroup */
      if (stage != MESA_SHADER_TESS_CTRL)
         /* dead io */
         return false;
      assign_slot_io(stage, io, var, slot);
   }
   var->data.driver_location = slot_map[slot];
   return true;
}


static bool
rewrite_read_as_0(nir_builder *b, nir_instr *instr, void *data)
{
   nir_variable *var = data;
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   if (!filter_io_instr(intr, &is_load, &is_input, &is_interp))
      return false;
   if (!is_load)
      return false;
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location != var->data.location)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *zero = nir_imm_zero(b, intr->def.num_components,
                                intr->def.bit_size);
   if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
      switch (location) {
      case VARYING_SLOT_COL0:
      case VARYING_SLOT_COL1:
      case VARYING_SLOT_BFC0:
      case VARYING_SLOT_BFC1:
         /* default color is 0,0,0,1 */
         if (intr->def.num_components == 4)
            zero = nir_vector_insert_imm(b, zero, nir_imm_float(b, 1.0), 3);
         break;
      default:
         break;
      }
   }
   nir_def_replace(&intr->def, zero);
   return true;
}



static bool
delete_psiz_store_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_store_per_vertex_output:
      break;
   default:
      return false;
   }
   if (nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PSIZ)
      return false;
   if (!data || (nir_src_is_const(intr->src[0]) && fabs(nir_src_as_float(intr->src[0]) - 1.0) < FLT_EPSILON)) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   return false;
}

static bool
delete_psiz_store(nir_shader *nir, bool one)
{
   bool progress = nir_shader_intrinsics_pass(nir, delete_psiz_store_instr,
                                              nir_metadata_dominance, one ? nir : NULL);
   if (progress)
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return progress;
}

struct write_components {
   unsigned slot;
   uint32_t component_mask;
};

static bool
fill_zero_reads(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct write_components *wc = data;
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   if (!filter_io_instr(intr, &is_load, &is_input, &is_interp))
      return false;
   if (!is_input)
      return false;
   nir_io_semantics s = nir_intrinsic_io_semantics(intr);
   if (wc->slot < s.location || wc->slot >= s.location + s.num_slots)
      return false;
   unsigned num_components = intr->num_components;
   unsigned c = nir_intrinsic_component(intr);
   if (intr->def.bit_size == 64)
      num_components *= 2;
   nir_src *src_offset = nir_get_io_offset_src(intr);
   if (!nir_src_is_const(*src_offset))
      return false;
   unsigned slot_offset = nir_src_as_uint(*src_offset);
   if (s.location + slot_offset != wc->slot)
      return false;
   uint32_t readmask = BITFIELD_MASK(intr->num_components) << c;
   if (intr->def.bit_size == 64)
      readmask |= readmask << (intr->num_components + c);
   /* handle dvec3/dvec4 */
   if (num_components + c > 4)
      readmask >>= 4;
   if ((wc->component_mask & readmask) == readmask)
      return false;
   uint32_t rewrite_mask = readmask & ~wc->component_mask;
   if (!rewrite_mask)
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *zero = nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);
   if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
      switch (wc->slot) {
      case VARYING_SLOT_COL0:
      case VARYING_SLOT_COL1:
      case VARYING_SLOT_BFC0:
      case VARYING_SLOT_BFC1:
         /* default color is 0,0,0,1 */
         if (intr->def.num_components == 4)
            zero = nir_vector_insert_imm(b, zero, nir_imm_float(b, 1.0), 3);
         break;
      default:
         break;
      }
   }
   rewrite_mask >>= c;
   nir_def *dest = &intr->def;
   u_foreach_bit(component, rewrite_mask)
      dest = nir_vector_insert_imm(b, dest, nir_channel(b, zero, component), component);
   nir_def_rewrite_uses_after(&intr->def, dest);
   return true;
}

static bool
find_max_write_components(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct write_components *wc = data;
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   if (!filter_io_instr(intr, &is_load, &is_input, &is_interp))
      return false;
   if (is_input || is_load)
      return false;
   nir_io_semantics s = nir_intrinsic_io_semantics(intr);
   if (wc->slot < s.location || wc->slot >= s.location + s.num_slots)
      return false;
   unsigned location = s.location;
   unsigned c = nir_intrinsic_component(intr);
   uint32_t wrmask = nir_intrinsic_write_mask(intr) << c;
   if ((nir_intrinsic_src_type(intr) & NIR_ALU_TYPE_SIZE_MASK) == 64) {
      unsigned num_components = intr->num_components * 2;
      nir_src *src_offset = nir_get_io_offset_src(intr);
      if (nir_src_is_const(*src_offset)) {
         if (location + nir_src_as_uint(*src_offset) != wc->slot && num_components + c < 4)
            return false;
      }
      wrmask |= wrmask << intr->num_components;
      /* handle dvec3/dvec4 */
      if (num_components + c > 4)
         wrmask >>= 4;
   }
   wc->component_mask |= wrmask;
   return false;
}

void
zink_compiler_assign_io(struct zink_screen *screen, nir_shader *producer, nir_shader *consumer)
{
   uint64_t slot_track[4] = {0};
   uint64_t patch_slot_track[4] = {0};
   unsigned char slot_map[VARYING_SLOT_MAX];
   memset(slot_map, -1, sizeof(slot_map));
   unsigned char patch_slot_map[VARYING_SLOT_MAX];
   memset(patch_slot_map, -1, sizeof(patch_slot_map));
   struct io_slot_map io = {
      .patch_slot_track = patch_slot_track,
      .slot_track = slot_track,
      .slot_map = slot_map,
      .patch_slot_map = patch_slot_map,
      .reserved = 0,
      .patch_reserved = 0,
   };
   bool do_fixup = false;
   nir_shader *nir = producer->info.stage == MESA_SHADER_TESS_CTRL ? producer : consumer;
   nir_variable *var = nir_find_variable_with_location(producer, nir_var_shader_out, VARYING_SLOT_PSIZ);
   if (var) {
      bool can_remove = false;
      if (!nir_find_variable_with_location(consumer, nir_var_shader_in, VARYING_SLOT_PSIZ)) {
         /* maintenance5 guarantees "A default size of 1.0 is used if PointSize is not written" */
         if (screen->info.have_KHR_maintenance5 && !var->data.explicit_xfb_buffer && delete_psiz_store(producer, true))
            can_remove = !(producer->info.outputs_written & VARYING_BIT_PSIZ);
         else if (consumer->info.stage != MESA_SHADER_FRAGMENT)
            can_remove = !var->data.explicit_location;
      }
      /* remove injected pointsize from all but the last vertex stage */
      if (can_remove) {
         var->data.mode = nir_var_shader_temp;
         nir_fixup_deref_modes(producer);
         delete_psiz_store(producer, false);
         NIR_PASS(_, producer, nir_remove_dead_variables, nir_var_shader_temp, NULL);
         optimize_nir(producer, NULL, true);
      }
   }
   if (consumer->info.stage != MESA_SHADER_FRAGMENT) {
      producer->info.has_transform_feedback_varyings = false;
      nir_foreach_shader_out_variable(var_out, producer)
         var_out->data.explicit_xfb_buffer = false;
   }
   if (producer->info.stage == MESA_SHADER_TESS_CTRL) {
      /* never assign from tcs -> tes, always invert */
      nir_foreach_variable_with_modes(var_in, consumer, nir_var_shader_in)
         assign_producer_var_io(consumer->info.stage, var_in, &io);
      nir_foreach_variable_with_modes_safe(var_out, producer, nir_var_shader_out) {
         if (!assign_consumer_var_io(producer->info.stage, var_out, &io))
            /* this is an output, nothing more needs to be done for it to be dropped */
            do_fixup = true;
      }
   } else {
      nir_foreach_variable_with_modes(var_out, producer, nir_var_shader_out)
         assign_producer_var_io(producer->info.stage, var_out, &io);
      nir_foreach_variable_with_modes_safe(var_in, consumer, nir_var_shader_in) {
         if (!assign_consumer_var_io(consumer->info.stage, var_in, &io)) {
            do_fixup = true;
            /* input needs to be rewritten */
            nir_shader_instructions_pass(consumer, rewrite_read_as_0, nir_metadata_dominance, var_in);
         }
      }
      if (consumer->info.stage == MESA_SHADER_FRAGMENT && screen->driver_compiler_workarounds.needs_sanitised_layer)
         do_fixup |= clamp_layer_output(producer, consumer, &io.reserved);
   }
   nir_shader_gather_info(producer, nir_shader_get_entrypoint(producer));
   if (producer->info.io_lowered && consumer->info.io_lowered) {
      u_foreach_bit64(slot, producer->info.outputs_written & BITFIELD64_RANGE(VARYING_SLOT_VAR0, 31)) {
         struct write_components wc = {slot, 0};
         nir_shader_intrinsics_pass(producer, find_max_write_components, nir_metadata_all, &wc);
         assert(wc.component_mask);
         if (wc.component_mask != BITFIELD_MASK(4))
            do_fixup |= nir_shader_intrinsics_pass(consumer, fill_zero_reads, nir_metadata_dominance, &wc);
      }
   }
   if (!do_fixup)
      return;
   nir_fixup_deref_modes(nir);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_temp, NULL);
   optimize_nir(nir, NULL, true);
}

/* all types that hit this function contain something that is 64bit */
static const struct glsl_type *
rewrite_64bit_type(nir_shader *nir, const struct glsl_type *type, nir_variable *var, bool doubles_only)
{
   if (glsl_type_is_array(type)) {
      const struct glsl_type *child = glsl_get_array_element(type);
      unsigned elements = glsl_array_size(type);
      unsigned stride = glsl_get_explicit_stride(type);
      return glsl_array_type(rewrite_64bit_type(nir, child, var, doubles_only), elements, stride);
   }
   /* rewrite structs recursively */
   if (glsl_type_is_struct_or_ifc(type)) {
      unsigned nmembers = glsl_get_length(type);
      struct glsl_struct_field *fields = rzalloc_array(nir, struct glsl_struct_field, nmembers * 2);
      unsigned xfb_offset = 0;
      for (unsigned i = 0; i < nmembers; i++) {
         const struct glsl_struct_field *f = glsl_get_struct_field_data(type, i);
         fields[i] = *f;
         xfb_offset += glsl_get_component_slots(fields[i].type) * 4;
         if (i < nmembers - 1 && xfb_offset % 8 &&
             (glsl_contains_double(glsl_get_struct_field(type, i + 1)) ||
              (glsl_type_contains_64bit(glsl_get_struct_field(type, i + 1)) && !doubles_only))) {
            var->data.is_xfb = true;
         }
         fields[i].type = rewrite_64bit_type(nir, f->type, var, doubles_only);
      }
      return glsl_struct_type(fields, nmembers, glsl_get_type_name(type), glsl_struct_type_is_packed(type));
   }
   if (!glsl_type_is_64bit(type) || (!glsl_contains_double(type) && doubles_only))
      return type;
   if (doubles_only && glsl_type_is_vector_or_scalar(type))
      return glsl_vector_type(GLSL_TYPE_UINT64, glsl_get_vector_elements(type));
   enum glsl_base_type base_type;
   switch (glsl_get_base_type(type)) {
   case GLSL_TYPE_UINT64:
      base_type = GLSL_TYPE_UINT;
      break;
   case GLSL_TYPE_INT64:
      base_type = GLSL_TYPE_INT;
      break;
   case GLSL_TYPE_DOUBLE:
      base_type = GLSL_TYPE_FLOAT;
      break;
   default:
      UNREACHABLE("unknown 64-bit vertex attribute format!");
   }
   if (glsl_type_is_scalar(type))
      return glsl_vector_type(base_type, 2);
   unsigned num_components;
   if (glsl_type_is_matrix(type)) {
      /* align to vec4 size: dvec3-composed arrays are arrays of dvec3s */
      unsigned vec_components = glsl_get_vector_elements(type);
      if (vec_components == 3)
         vec_components = 4;
      num_components = vec_components * 2 * glsl_get_matrix_columns(type);
   } else {
      num_components = glsl_get_vector_elements(type) * 2;
      if (num_components <= 4)
         return glsl_vector_type(base_type, num_components);
   }
   /* dvec3/dvec4/dmatX: rewrite as struct { vec4, vec4, vec4, ... [vec2] } */
   struct glsl_struct_field fields[8] = {0};
   unsigned remaining = num_components;
   unsigned nfields = 0;
   for (unsigned i = 0; remaining; i++, remaining -= MIN2(4, remaining), nfields++) {
      assert(i < ARRAY_SIZE(fields));
      fields[i].name = "";
      fields[i].offset = i * 16;
      fields[i].type = glsl_vector_type(base_type, MIN2(4, remaining));
   }
   char buf[64];
   snprintf(buf, sizeof(buf), "struct(%s)", glsl_get_type_name(type));
   return glsl_struct_type(fields, nfields, buf, true);
}

static const struct glsl_type *
deref_is_matrix(nir_deref_instr *deref)
{
   if (glsl_type_is_matrix(deref->type))
      return deref->type;
   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   if (parent)
      return deref_is_matrix(parent);
   return NULL;
}

static bool
lower_64bit_vars_function(nir_shader *shader, nir_function_impl *impl, nir_variable *var,
                          struct hash_table *derefs, struct set *deletes, bool doubles_only)
{
   bool func_progress = false;
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (!(deref->modes & var->data.mode))
               continue;
            if (nir_deref_instr_get_variable(deref) != var)
               continue;

            /* matrix types are special: store the original deref type for later use */
            const struct glsl_type *matrix = deref_is_matrix(deref);
            nir_deref_instr *parent = nir_deref_instr_parent(deref);
            if (!matrix) {
               /* if this isn't a direct matrix deref, it's maybe a matrix row deref */
               hash_table_foreach(derefs, he) {
                  /* propagate parent matrix type to row deref */
                  if (he->key == parent)
                     matrix = he->data;
               }
            }
            if (matrix)
               _mesa_hash_table_insert(derefs, deref, (void*)matrix);
            if (deref->deref_type == nir_deref_type_var)
               deref->type = var->type;
            else
               deref->type = rewrite_64bit_type(shader, deref->type, var, doubles_only);
         }
         break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_deref &&
                  intr->intrinsic != nir_intrinsic_load_deref)
               break;
            if (nir_intrinsic_get_var(intr, 0) != var)
               break;
            if ((intr->intrinsic == nir_intrinsic_store_deref && intr->src[1].ssa->bit_size != 64) ||
                  (intr->intrinsic == nir_intrinsic_load_deref && intr->def.bit_size != 64))
               break;
            b.cursor = nir_before_instr(instr);
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            unsigned num_components = intr->num_components * 2;
            nir_def *comp[NIR_MAX_VEC_COMPONENTS];
            /* this is the stored matrix type from the deref */
            struct hash_entry *he = _mesa_hash_table_search(derefs, deref);
            const struct glsl_type *matrix = he ? he->data : NULL;
            if (doubles_only && !matrix)
               break;
            func_progress = true;
            if (intr->intrinsic == nir_intrinsic_store_deref) {
               /* first, unpack the src data to 32bit vec2 components */
               for (unsigned i = 0; i < intr->num_components; i++) {
                  nir_def *ssa = nir_unpack_64_2x32(&b, nir_channel(&b, intr->src[1].ssa, i));
                  comp[i * 2] = nir_channel(&b, ssa, 0);
                  comp[i * 2 + 1] = nir_channel(&b, ssa, 1);
               }
               unsigned wrmask = nir_intrinsic_write_mask(intr);
               unsigned mask = 0;
               /* expand writemask for doubled components */
               for (unsigned i = 0; i < intr->num_components; i++) {
                  if (wrmask & BITFIELD_BIT(i))
                     mask |= BITFIELD_BIT(i * 2) | BITFIELD_BIT(i * 2 + 1);
               }
               if (matrix) {
                  /* matrix types always come from array (row) derefs */
                  assert(deref->deref_type == nir_deref_type_array);
                  nir_deref_instr *var_deref = nir_deref_instr_parent(deref);
                  /* let optimization clean up consts later */
                  nir_def *index = deref->arr.index.ssa;
                  /* this might be an indirect array index:
                     * - iterate over matrix columns
                     * - add if blocks for each column
                     * - perform the store in the block
                     */
                  for (unsigned idx = 0; idx < glsl_get_matrix_columns(matrix); idx++) {
                     nir_push_if(&b, nir_ieq_imm(&b, index, idx));
                     unsigned vec_components = glsl_get_vector_elements(matrix);
                     /* always clamp dvec3 to 4 components */
                     if (vec_components == 3)
                        vec_components = 4;
                     unsigned start_component = idx * vec_components * 2;
                     /* struct member */
                     unsigned member = start_component / 4;
                     /* number of components remaining */
                     unsigned remaining = num_components;
                     for (unsigned i = 0; i < num_components; member++) {
                        if (!(mask & BITFIELD_BIT(i)))
                           continue;
                        assert(member < glsl_get_length(var_deref->type));
                        /* deref the rewritten struct to the appropriate vec4/vec2 */
                        nir_deref_instr *strct = nir_build_deref_struct(&b, var_deref, member);
                        unsigned incr = MIN2(remaining, 4);
                        /* assemble the write component vec */
                        nir_def *val = nir_vec(&b, &comp[i], incr);
                        /* use the number of components being written as the writemask */
                        if (glsl_get_vector_elements(strct->type) > val->num_components)
                           val = nir_pad_vector(&b, val, glsl_get_vector_elements(strct->type));
                        nir_store_deref(&b, strct, val, BITFIELD_MASK(incr));
                        remaining -= incr;
                        i += incr;
                     }
                     nir_pop_if(&b, NULL);
                  }
                  _mesa_set_add(deletes, &deref->instr);
               } else if (num_components <= 4) {
                  /* simple store case: just write out the components */
                  nir_def *dest = nir_vec(&b, comp, num_components);
                  nir_store_deref(&b, deref, dest, mask);
               } else {
                  /* writing > 4 components: access the struct and write to the appropriate vec4 members */
                  for (unsigned i = 0; num_components; i++, num_components -= MIN2(num_components, 4)) {
                     if (!(mask & BITFIELD_MASK(4)))
                        continue;
                     nir_deref_instr *strct = nir_build_deref_struct(&b, deref, i);
                     nir_def *dest = nir_vec(&b, &comp[i * 4], MIN2(num_components, 4));
                     if (glsl_get_vector_elements(strct->type) > dest->num_components)
                        dest = nir_pad_vector(&b, dest, glsl_get_vector_elements(strct->type));
                     nir_store_deref(&b, strct, dest, mask & BITFIELD_MASK(4));
                     mask >>= 4;
                  }
               }
            } else {
               nir_def *dest = NULL;
               if (matrix) {
                  /* matrix types always come from array (row) derefs */
                  assert(deref->deref_type == nir_deref_type_array);
                  nir_deref_instr *var_deref = nir_deref_instr_parent(deref);
                  /* let optimization clean up consts later */
                  nir_def *index = deref->arr.index.ssa;
                  /* this might be an indirect array index:
                     * - iterate over matrix columns
                     * - add if blocks for each column
                     * - phi the loads using the array index
                     */
                  unsigned cols = glsl_get_matrix_columns(matrix);
                  nir_def *dests[4];
                  for (unsigned idx = 0; idx < cols; idx++) {
                     /* don't add an if for the final row: this will be handled in the else */
                     if (idx < cols - 1)
                        nir_push_if(&b, nir_ieq_imm(&b, index, idx));
                     unsigned vec_components = glsl_get_vector_elements(matrix);
                     /* always clamp dvec3 to 4 components */
                     if (vec_components == 3)
                        vec_components = 4;
                     unsigned start_component = idx * vec_components * 2;
                     /* struct member */
                     unsigned member = start_component / 4;
                     /* number of components remaining */
                     unsigned remaining = num_components;
                     /* component index */
                     unsigned comp_idx = 0;
                     for (unsigned i = 0; i < num_components; member++) {
                        assert(member < glsl_get_length(var_deref->type));
                        nir_deref_instr *strct = nir_build_deref_struct(&b, var_deref, member);
                        nir_def *load = nir_load_deref(&b, strct);
                        unsigned incr = MIN2(remaining, 4);
                        /* repack the loads to 64bit */
                        for (unsigned c = 0; c < incr / 2; c++, comp_idx++)
                           comp[comp_idx] = nir_pack_64_2x32(&b, nir_channels(&b, load, BITFIELD_RANGE(c * 2, 2)));
                        remaining -= incr;
                        i += incr;
                     }
                     dest = dests[idx] = nir_vec(&b, comp, intr->num_components);
                     if (idx < cols - 1)
                        nir_push_else(&b, NULL);
                  }
                  /* loop over all the if blocks that were made, pop them, and phi the loaded+packed results */
                  for (unsigned idx = cols - 1; idx >= 1; idx--) {
                     nir_pop_if(&b, NULL);
                     dest = nir_if_phi(&b, dests[idx - 1], dest);
                  }
                  _mesa_set_add(deletes, &deref->instr);
               } else if (num_components <= 4) {
                  /* simple load case */
                  nir_def *load = nir_load_deref(&b, deref);
                  /* pack 32bit loads into 64bit: this will automagically get optimized out later */
                  for (unsigned i = 0; i < intr->num_components; i++) {
                     comp[i] = nir_pack_64_2x32(&b, nir_channels(&b, load, BITFIELD_RANGE(i * 2, 2)));
                  }
                  dest = nir_vec(&b, comp, intr->num_components);
               } else {
                  /* writing > 4 components: access the struct and load the appropriate vec4 members */
                  for (unsigned i = 0; i < 2; i++, num_components -= 4) {
                     nir_deref_instr *strct = nir_build_deref_struct(&b, deref, i);
                     nir_def *load = nir_load_deref(&b, strct);
                     comp[i * 2] = nir_pack_64_2x32(&b,
                                                    nir_trim_vector(&b, load, 2));
                     if (num_components > 2)
                        comp[i * 2 + 1] = nir_pack_64_2x32(&b, nir_channels(&b, load, BITFIELD_RANGE(2, 2)));
                  }
                  dest = nir_vec(&b, comp, intr->num_components);
               }
               nir_def_rewrite_uses_after_instr(&intr->def, dest, instr);
            }
            _mesa_set_add(deletes, instr);
            break;
         }
         break;
         default: break;
         }
      }
   }
   nir_progress(func_progress, impl, nir_metadata_none);
   /* derefs must be queued for deletion to avoid deleting the same deref repeatedly */
   set_foreach_remove(deletes, he)
      nir_instr_remove((void*)he->key);
   return func_progress;
}

static bool
lower_64bit_vars_loop(nir_shader *shader, nir_variable *var, struct hash_table *derefs,
                      struct set *deletes, bool doubles_only)
{
   if (!glsl_type_contains_64bit(var->type) || (doubles_only && !glsl_contains_double(var->type)))
      return false;
   var->type = rewrite_64bit_type(shader, var->type, var, doubles_only);
   /* once type is rewritten, rewrite all loads and stores */
   nir_foreach_function_impl(impl, shader)
      lower_64bit_vars_function(shader, impl, var, derefs, deletes, doubles_only);
   return true;
}

/* rewrite all input/output variables using 32bit types and load/stores */
static bool
lower_64bit_vars(nir_shader *shader, bool doubles_only)
{
   bool progress = false;
   struct hash_table *derefs = _mesa_hash_table_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);
   struct set *deletes = _mesa_set_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_function_temp_variable(var, impl) {
         if (!glsl_type_contains_64bit(var->type) || (doubles_only && !glsl_contains_double(var->type)))
            continue;
         var->type = rewrite_64bit_type(shader, var->type, var, doubles_only);
         progress |= lower_64bit_vars_function(shader, impl, var, derefs, deletes, doubles_only);
      }
   }
   ralloc_free(deletes);
   ralloc_free(derefs);
   if (progress) {
      nir_lower_alu_to_scalar(shader, filter_64_bit_instr, NULL);
      nir_lower_phis_to_scalar(shader, NULL, NULL);
      optimize_nir(shader, NULL, true);
   }
   return progress;
}

static void
zink_shader_dump(const struct zink_shader *zs, void *words, size_t size, const char *file)
{
   FILE *fp = fopen(file, "wb");
   if (fp) {
      fwrite(words, 1, size, fp);
      fclose(fp);
      fprintf(stderr, "wrote %s shader '%s'...\n", _mesa_shader_stage_to_string(zs->info.stage), file);
   }
}

static VkShaderStageFlagBits
zink_get_next_stage(gl_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
             VK_SHADER_STAGE_GEOMETRY_BIT |
             VK_SHADER_STAGE_FRAGMENT_BIT;
   case MESA_SHADER_TESS_CTRL:
      return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
   case MESA_SHADER_TESS_EVAL:
      return VK_SHADER_STAGE_GEOMETRY_BIT |
             VK_SHADER_STAGE_FRAGMENT_BIT;
   case MESA_SHADER_GEOMETRY:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
   case MESA_SHADER_FRAGMENT:
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_KERNEL:
      return 0;
   default:
      UNREACHABLE("invalid shader stage");
   }
}

struct zink_shader_object
zink_shader_spirv_compile(struct zink_screen *screen, struct zink_shader *zs, struct spirv_shader *spirv, bool can_shobj, struct zink_program *pg)
{
   VkShaderModuleCreateInfo smci = {0};
   VkShaderCreateInfoEXT sci = {0};

   if (!spirv)
      spirv = zs->spirv;

   if (zink_debug & ZINK_DEBUG_SPIRV) {
      char buf[256];
      static int i;
      snprintf(buf, sizeof(buf), "dump%02d.spv", i++);
      zink_shader_dump(zs, spirv->words, spirv->num_words * sizeof(uint32_t), buf);
   }

   sci.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
   sci.stage = mesa_to_vk_shader_stage(zs->info.stage);
   sci.nextStage = zink_get_next_stage(zs->info.stage);
   sci.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
   sci.codeSize = spirv->num_words * sizeof(uint32_t);
   sci.pCode = spirv->words;
   sci.pName = "main";
   VkDescriptorSetLayout dsl[ZINK_GFX_SHADER_COUNT] = {0};
   if (pg) {
      sci.setLayoutCount = pg->num_dsl;
      sci.pSetLayouts = pg->dsl;
   } else {
      sci.setLayoutCount = zs->info.stage == MESA_SHADER_COMPUTE ? 1 : ZINK_GFX_SHADER_COUNT;
      dsl[zs->info.stage] = zs->precompile.dsl;;
      sci.pSetLayouts = dsl;
   }
   VkPushConstantRange pcr;
   pcr.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;
   pcr.offset = 0;
   pcr.size = sizeof(struct zink_gfx_push_constant);
   sci.pushConstantRangeCount = 1;
   sci.pPushConstantRanges = &pcr;

   smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
   smci.codeSize = spirv->num_words * sizeof(uint32_t);
   smci.pCode = spirv->words;

#ifndef NDEBUG
   if (zink_debug & ZINK_DEBUG_VALIDATION) {
      static const struct spirv_to_nir_options spirv_options = {
         .environment = NIR_SPIRV_VULKAN,
         .capabilities = NULL,
         .ubo_addr_format = nir_address_format_32bit_index_offset,
         .ssbo_addr_format = nir_address_format_32bit_index_offset,
         .phys_ssbo_addr_format = nir_address_format_64bit_global,
         .push_const_addr_format = nir_address_format_logical,
         .shared_addr_format = nir_address_format_32bit_offset,
      };
      uint32_t num_spec_entries = 0;
      struct nir_spirv_specialization *spec_entries = NULL;
      VkSpecializationInfo sinfo = {0};
      VkSpecializationMapEntry me[3];
      uint32_t size[3] = {1,1,1};
      if (!zs->info.workgroup_size[0]) {
         sinfo.mapEntryCount = 3;
         sinfo.pMapEntries = &me[0];
         sinfo.dataSize = sizeof(uint32_t) * 3;
         sinfo.pData = size;
         uint32_t ids[] = {ZINK_WORKGROUP_SIZE_X, ZINK_WORKGROUP_SIZE_Y, ZINK_WORKGROUP_SIZE_Z};
         for (int i = 0; i < 3; i++) {
            me[i].size = sizeof(uint32_t);
            me[i].constantID = ids[i];
            me[i].offset = i * sizeof(uint32_t);
         }
         spec_entries = vk_spec_info_to_nir_spirv(&sinfo, &num_spec_entries);
      }
      nir_shader *nir = spirv_to_nir(spirv->words, spirv->num_words,
                         spec_entries, num_spec_entries,
                         clamp_stage(&zs->info), "main", &spirv_options, &screen->nir_options);
      assert(nir);
      ralloc_free(nir);
      free(spec_entries);
   }
#endif

   VkResult ret;
   struct zink_shader_object obj = {0};
   if (!can_shobj || !screen->info.have_EXT_shader_object)
      ret = VKSCR(CreateShaderModule)(screen->dev, &smci, NULL, &obj.mod);
   else
      ret = VKSCR(CreateShadersEXT)(screen->dev, 1, &sci, NULL, &obj.obj);
   ASSERTED bool success = zink_screen_handle_vkresult(screen, ret);
   assert(success);
   return obj;
}

static void
prune_io(nir_shader *nir)
{
   nir_foreach_shader_in_variable_safe(var, nir) {
      if (!find_var_deref(nir, var) && !find_var_io(nir, var))
         var->data.mode = nir_var_shader_temp;
   }
   nir_foreach_shader_out_variable_safe(var, nir) {
      if (!find_var_deref(nir, var) && !find_var_io(nir, var))
         var->data.mode = nir_var_shader_temp;
   }
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_temp, NULL);
}

static void
flag_shadow_tex(nir_variable *var, struct zink_shader *zs)
{
   assert(var->data.driver_location < 32); //bitfield size for tracking
   zs->fs.legacy_shadow_mask |= BITFIELD_BIT(var->data.driver_location);
}

static void
flag_shadow_tex_instr(nir_builder *b, nir_tex_instr *tex, nir_variable *var, struct zink_shader *zs)
{
   assert(var);
   unsigned num_components = tex->def.num_components;
   bool rewrite_depth = tex->is_shadow && num_components > 1 && tex->op != nir_texop_tg4 && !tex->is_sparse;
   if (rewrite_depth && nir_def_components_read( &tex->def) & ~1) {
      /* this needs recompiles */
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT)
         flag_shadow_tex(var, zs);
      else
         mesa_loge("unhandled old-style shadow sampler in non-fragment stage!");
   }
}

static nir_def *
rewrite_tex_dest(nir_builder *b, nir_tex_instr *tex, nir_variable *var, struct zink_shader *zs)
{
   assert(var);
   const struct glsl_type *type = glsl_without_array(var->type);
   enum glsl_base_type ret_type = glsl_get_sampler_result_type(type);
   bool is_int = glsl_base_type_is_integer(ret_type);
   unsigned bit_size = glsl_base_type_get_bit_size(ret_type);
   unsigned dest_size = tex->def.bit_size;
   b->cursor = nir_after_instr(&tex->instr);
   unsigned num_components = tex->def.num_components;
   bool rewrite_depth = tex->is_shadow && num_components > 1 && tex->op != nir_texop_tg4 && !tex->is_sparse;
   if (bit_size == dest_size && !rewrite_depth)
      return NULL;
   nir_def *dest = &tex->def;
   if (rewrite_depth && zs) {
      if (nir_def_components_read(dest) & ~1) {
         /* handled above */
         return NULL;
      }
      /* If only .x is used in the NIR, then it's effectively not a legacy depth
       * sample anyway and we don't want to ask for shader recompiles.  This is
       * the typical path, since GL_DEPTH_TEXTURE_MODE defaults to either RED or
       * LUMINANCE, so apps just use the first channel.
       */
      tex->def.num_components = 1;
      tex->is_new_style_shadow = true;
   }
   if (bit_size != dest_size) {
      tex->def.bit_size = bit_size;
      tex->dest_type = nir_get_nir_type_for_glsl_base_type(ret_type);

      if (is_int) {
         if (glsl_unsigned_base_type_of(ret_type) == ret_type)
            dest = nir_u2uN(b, &tex->def, dest_size);
         else
            dest = nir_i2iN(b, &tex->def, dest_size);
      } else {
         dest = nir_f2fN(b, &tex->def, dest_size);
      }
      if (!rewrite_depth)
         nir_def_rewrite_uses_after(&tex->def, dest);
   }
   return dest;
}

struct lower_zs_swizzle_state {
   bool shadow_only;
   unsigned base_sampler_id;
   const struct zink_zs_swizzle_key *swizzle;
};

static bool
lower_zs_swizzle_tex_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_zs_swizzle_state *state = data;
   const struct zink_zs_swizzle_key *swizzle_key = state->swizzle;
   assert(state->shadow_only || swizzle_key);
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op == nir_texop_txs || tex->op == nir_texop_lod ||
       (!tex->is_shadow && state->shadow_only) || tex->is_new_style_shadow)
      return false;
   if (tex->is_shadow && tex->op == nir_texop_tg4)
      /* Will not even try to emulate the shadow comparison */
      return false;
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   nir_variable *var = NULL;
   if (handle != -1)
      /* gtfo bindless depth texture mode */
      return false;
   var = nir_deref_instr_get_variable(nir_def_as_deref(tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_deref)].src.ssa));
   assert(var);
   uint32_t sampler_id = var->data.binding - state->base_sampler_id;
   const struct glsl_type *type = glsl_without_array(var->type);
   enum glsl_base_type ret_type = glsl_get_sampler_result_type(type);
   bool is_int = glsl_base_type_is_integer(ret_type);
   unsigned num_components = tex->def.num_components;
   if (tex->is_shadow)
      tex->is_new_style_shadow = true;
   nir_def *dest = rewrite_tex_dest(b, tex, var, NULL);
   assert(dest || !state->shadow_only);
   if (!dest && !(swizzle_key->mask & BITFIELD_BIT(sampler_id)))
      return false;
   else if (!dest)
      dest = &tex->def;
   else
      tex->def.num_components = 1;
   if (swizzle_key && (swizzle_key->mask & BITFIELD_BIT(sampler_id))) {
      /* these require manual swizzles */
      if (tex->op == nir_texop_tg4) {
         assert(!tex->is_shadow);
         nir_def *swizzle;
         switch (swizzle_key->swizzle[sampler_id].s[tex->component]) {
         case PIPE_SWIZZLE_0:
            swizzle = nir_imm_zero(b, 4, tex->def.bit_size);
            break;
         case PIPE_SWIZZLE_1:
            if (is_int)
               swizzle = nir_imm_intN_t(b, 4, tex->def.bit_size);
            else
               swizzle = nir_imm_floatN_t(b, 4, tex->def.bit_size);
            break;
         default:
            if (!tex->component)
               return false;
            tex->component = 0;
            return true;
         }
         nir_def_rewrite_uses_after(dest, swizzle);
         return true;
      }
      nir_def *vec[4];
      for (unsigned i = 0; i < ARRAY_SIZE(vec); i++) {
         switch (swizzle_key->swizzle[sampler_id].s[i]) {
         case PIPE_SWIZZLE_0:
            vec[i] = nir_imm_zero(b, 1, tex->def.bit_size);
            break;
         case PIPE_SWIZZLE_1:
            if (is_int)
               vec[i] = nir_imm_intN_t(b, 1, tex->def.bit_size);
            else
               vec[i] = nir_imm_floatN_t(b, 1, tex->def.bit_size);
            break;
         default:
            vec[i] = dest->num_components == 1 ? dest : nir_channel(b, dest, i);
            break;
         }
      }
      nir_def *swizzle = nir_vec(b, vec, num_components);
      nir_def_rewrite_uses_after(dest, swizzle);
   } else {
      assert(tex->is_shadow);
      nir_def *vec[4] = {dest, dest, dest, dest};
      nir_def *splat = nir_vec(b, vec, num_components);
      nir_def_rewrite_uses_after(dest, splat);
   }
   return true;
}

/* Applies in-shader swizzles when necessary for depth/shadow sampling.
 *
 * SPIRV only has new-style (scalar result) shadow sampling, so to emulate
 * !is_new_style_shadow (vec4 result) shadow sampling we lower to a
 * new-style-shadow sample, and apply GL_DEPTH_TEXTURE_MODE swizzles in the NIR
 * shader to expand out to vec4.  Since this depends on sampler state, it's a
 * draw-time shader recompile to do so.
 *
 * We may also need to apply shader swizzles for
 * driver_compiler_workarounds.needs_zs_shader_swizzle.
 */
static bool
lower_zs_swizzle_tex(nir_shader *nir, const void *swizzle, bool shadow_only)
{
   /* We don't use nir_lower_tex to do our swizzling, because of this base_sampler_id. */
   unsigned base_sampler_id = gl_shader_stage_is_compute(nir->info.stage) ? 0 : PIPE_MAX_SAMPLERS * nir->info.stage;
   struct lower_zs_swizzle_state state = {shadow_only, base_sampler_id, swizzle};
   return nir_shader_instructions_pass(nir, lower_zs_swizzle_tex_instr,
                                       nir_metadata_control_flow,
                                       (void*)&state);
}

static bool
invert_point_coord_instr(nir_builder *b, nir_intrinsic_instr *intr,
                         void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_point_coord)
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *def = nir_vec2(b, nir_channel(b, &intr->def, 0),
                                  nir_fsub_imm(b, 1.0, nir_channel(b, &intr->def, 1)));
   nir_def_rewrite_uses_after(&intr->def, def);
   return true;
}

static bool
invert_point_coord(nir_shader *nir)
{
   if (!BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_POINT_COORD))
      return false;
   return nir_shader_intrinsics_pass(nir, invert_point_coord_instr,
                                     nir_metadata_dominance, NULL);
}

static bool
lower_sparse_instr(nir_builder *b, nir_instr *instr, void *data)
{
   b->cursor = nir_after_instr(instr);

   switch (instr->type) {
   case nir_instr_type_tex: {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      if (!tex->is_sparse)
         return false;

      nir_def *res = nir_b2i32(b, nir_is_sparse_resident_zink(b, &tex->def));
      nir_def *vec = nir_vector_insert_imm(b, &tex->def, res,
                                           tex->def.num_components - 1);
      nir_def_rewrite_uses_after(&tex->def, vec);
      return true;
   }

   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      switch (intrin->intrinsic) {
      case nir_intrinsic_image_deref_sparse_load: {
         nir_def *res = nir_b2i32(b, nir_is_sparse_resident_zink(b, &intrin->def));
         nir_def *vec = nir_vector_insert_imm(b, &intrin->def, res, 4);
         nir_def_rewrite_uses_after(&intrin->def, vec);
         return true;
      }

      case nir_intrinsic_sparse_residency_code_and: {
         nir_def *res = nir_iand(b, intrin->src[0].ssa, intrin->src[1].ssa);
         nir_def_rewrite_uses(&intrin->def, res);
         return true;
      }

      case nir_intrinsic_is_sparse_texels_resident: {
         nir_def *res = nir_i2b(b, intrin->src[0].ssa);
         nir_def_rewrite_uses(&intrin->def, res);
         return true;
      }

      default:
         return false;
      }
   }

   default:
      return false;
   }
}

static bool
lower_sparse(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_sparse_instr,
                                       nir_metadata_dominance, NULL);
}

static bool
add_derefs_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   if (!filter_io_instr(intr, &is_load, &is_input, &is_interp))
      return false;
   bool is_special_io = (b->shader->info.stage == MESA_SHADER_VERTEX && is_input) ||
                        (b->shader->info.stage == MESA_SHADER_FRAGMENT && !is_input);
   unsigned loc = nir_intrinsic_io_semantics(intr).location;
   nir_src *src_offset = nir_get_io_offset_src(intr);
   const unsigned slot_offset = src_offset && nir_src_is_const(*src_offset) ? nir_src_as_uint(*src_offset) : 0;
   unsigned location = loc + slot_offset;
   unsigned frac = nir_intrinsic_component(intr);
   unsigned bit_size = is_load ? intr->def.bit_size : nir_src_bit_size(intr->src[0]);
   /* set c aligned/rounded down to dword */
   unsigned c = frac;
   if (frac && bit_size < 32)
      c = frac * bit_size / 32;
   /* loop over all the variables and rewrite corresponding access */
   nir_foreach_variable_with_modes(var, b->shader, is_input ? nir_var_shader_in : nir_var_shader_out) {
      const struct glsl_type *type = var->type;
      if (nir_is_arrayed_io(var, b->shader->info.stage))
         type = glsl_get_array_element(type);
      unsigned slot_count = get_var_slot_count(b->shader, var);
      /* filter access that isn't specific to this variable */
      if (var->data.location > location || var->data.location + slot_count <= location)
         continue;
      if (var->data.fb_fetch_output != nir_intrinsic_io_semantics(intr).fb_fetch_output)
         continue;
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT && !is_load && nir_intrinsic_io_semantics(intr).dual_source_blend_index != var->data.index)
         continue;

      unsigned size = 0;
      bool is_struct = glsl_type_is_struct(glsl_without_array(type));
      if (is_struct)
         size = get_slot_components(var, var->data.location + slot_offset, var->data.location);
      else if (!is_special_io && var->data.compact)
         size = glsl_get_aoa_size(type);
      else
         size = glsl_get_vector_elements(glsl_without_array(type));
      assert(size);
      if (glsl_type_is_64bit(glsl_without_array(var->type)))
         size *= 2;
      if (var->data.location != location && size > 4 && size % 4 && !is_struct) {
         /* adjust for dvec3-type slot overflow */
         assert(location > var->data.location);
         size -= (location - var->data.location) * 4;
      }
      assert(size);
      if (var->data.location_frac + size <= c || var->data.location_frac > c)
         continue;

      b->cursor = nir_before_instr(&intr->instr);
      nir_deref_instr *deref = nir_build_deref_var(b, var);
      if (nir_is_arrayed_io(var, b->shader->info.stage)) {
         assert(intr->intrinsic != nir_intrinsic_store_output);
         deref = nir_build_deref_array(b, deref, intr->src[!is_load].ssa);
      }
      if (glsl_type_is_array(type)) {
         /* unroll array derefs */
         unsigned idx = var->data.compact ? (frac - var->data.location_frac) : 0;
         assert(src_offset);
         if (var->data.location < VARYING_SLOT_VAR0) {
            if (src_offset) {
               /* clip/cull dist and tess levels use different array offset semantics */
               bool is_clipdist = (b->shader->info.stage != MESA_SHADER_VERTEX || var->data.mode == nir_var_shader_out) &&
                                  is_clipcull_dist(var->data.location);
               bool is_tess_level = b->shader->info.stage == MESA_SHADER_TESS_CTRL &&
                                    var->data.location >= VARYING_SLOT_TESS_LEVEL_INNER && var->data.location >= VARYING_SLOT_TESS_LEVEL_OUTER;
               bool is_builtin_array = is_clipdist || is_tess_level;
               /* this is explicit for ease of debugging but could be collapsed at some point in the future*/
               if (nir_src_is_const(*src_offset)) {
                  unsigned offset = slot_offset;
                  if (is_builtin_array)
                     offset *= 4;
                  if (is_clipdist) {
                     if (loc == VARYING_SLOT_CLIP_DIST1 || loc == VARYING_SLOT_CULL_DIST1)
                        offset += 4;
                  }
                  deref = nir_build_deref_array_imm(b, deref, offset + idx);
               } else {
                  nir_def *offset = src_offset->ssa;
                  if (is_builtin_array)
                     nir_imul_imm(b, offset, 4);
                  deref = nir_build_deref_array(b, deref, idx ? nir_iadd_imm(b, offset, idx) : src_offset->ssa);
               }
            } else {
               deref = nir_build_deref_array_imm(b, deref, idx);
            }
            type = glsl_get_array_element(type);
         } else {
            idx += location - var->data.location;
            /* need to convert possible N*M to [N][M] */
            nir_def *nm = idx ? nir_iadd_imm(b, src_offset->ssa, idx) : src_offset->ssa;
            while (glsl_type_is_array(type)) {
               const struct glsl_type *elem = glsl_get_array_element(type);
               unsigned type_size = glsl_count_vec4_slots(elem, false, false);
               nir_def *n = glsl_type_is_array(elem) ? nir_udiv_imm(b, nm, type_size) : nm;
               if (glsl_type_is_vector_or_scalar(elem) && glsl_type_is_64bit(elem) && glsl_get_vector_elements(elem) > 2)
                  n = nir_udiv_imm(b, n, 2);
               deref = nir_build_deref_array(b, deref, n);
               nm = nir_umod_imm(b, nm, type_size);
               type = glsl_get_array_element(type);
            }
         }
      } else if (glsl_type_is_struct(type)) {
         deref = nir_build_deref_struct(b, deref, slot_offset);
      }
      assert(!glsl_type_is_array(type));
      unsigned num_components = glsl_get_vector_elements(type);
      if (is_load) {
         nir_def *load;
         if (is_interp) {
            nir_def *interp = intr->src[0].ssa;
            nir_intrinsic_instr *interp_intr = nir_def_as_intrinsic(interp);
            assert(interp_intr);
            var->data.interpolation = nir_intrinsic_interp_mode(interp_intr);
            switch (interp_intr->intrinsic) {
            case nir_intrinsic_load_barycentric_centroid:
               load = nir_interp_deref_at_centroid(b, num_components, bit_size, &deref->def);
               break;
            case nir_intrinsic_load_barycentric_sample:
               var->data.sample = 1;
               load = nir_load_deref(b, deref);
               break;
            case nir_intrinsic_load_barycentric_pixel:
               load = nir_load_deref(b, deref);
               break;
            case nir_intrinsic_load_barycentric_at_sample:
               load = nir_interp_deref_at_sample(b, num_components, bit_size, &deref->def, interp_intr->src[0].ssa);
               break;
            case nir_intrinsic_load_barycentric_at_offset:
               load = nir_interp_deref_at_offset(b, num_components, bit_size, &deref->def, interp_intr->src[0].ssa);
               break;
            default:
               UNREACHABLE("unhandled interp!");
            }
         } else {
            load = nir_load_deref(b, deref);
         }
         /* filter needed components */
         if (intr->num_components < load->num_components)
            load = nir_channels(b, load, BITFIELD_MASK(intr->num_components) << (c - var->data.location_frac));
         nir_def_rewrite_uses(&intr->def, load);
      } else {
         nir_def *store = intr->src[0].ssa;
         /* pad/filter components to match deref type */
         if (intr->num_components < num_components) {
            nir_def *zero = nir_imm_zero(b, 1, bit_size);
            nir_def *vec[4] = {zero, zero, zero, zero};
            u_foreach_bit(i, nir_intrinsic_write_mask(intr))
               vec[c - var->data.location_frac + i] = nir_channel(b, store, i);
            store = nir_vec(b, vec, num_components);
         } if (store->num_components > num_components) {
            store = nir_channels(b, store, nir_intrinsic_write_mask(intr));
         }
         if (store->bit_size != glsl_get_bit_size(type)) {
            /* this should be some weird bindless io conversion */
            assert(store->bit_size == 64 && glsl_get_bit_size(type) == 32);
            assert(num_components != store->num_components);
            store = nir_unpack_64_2x32(b, store);
         }
         nir_store_deref(b, deref, store, BITFIELD_RANGE(c - var->data.location_frac, intr->num_components));
      }
      nir_instr_remove(&intr->instr);
      return true;
   }
   UNREACHABLE("failed to find variable for explicit io!");
   return true;
}

static bool
add_derefs(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, add_derefs_instr,
                                     nir_metadata_dominance, NULL);
}

static struct zink_shader_object
compile_module(struct zink_screen *screen, struct zink_shader *zs, nir_shader *nir, bool can_shobj, struct zink_program *pg)
{
   struct zink_shader_info *sinfo = &zs->sinfo;
   prune_io(nir);

   NIR_PASS(_, nir, nir_convert_from_ssa, true, false);

   if (zink_debug & (ZINK_DEBUG_NIR | ZINK_DEBUG_SPIRV))
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   if (zink_debug & ZINK_DEBUG_NIR) {
      fprintf(stderr, "NIR shader:\n---8<---\n");
      nir_print_shader(nir, stderr);
      fprintf(stderr, "---8<---\n");
   }

   struct zink_shader_object obj = {0};
   struct spirv_shader *spirv = nir_to_spirv(nir, sinfo, screen);
   if (spirv)
      obj = zink_shader_spirv_compile(screen, zs, spirv, can_shobj, pg);

   /* TODO: determine if there's any reason to cache spirv output? */
   if (zs->info.stage == MESA_SHADER_TESS_CTRL && zs->non_fs.is_generated)
      zs->spirv = spirv;
   else
      obj.spirv = spirv;
   return obj;
}

static bool
remove_interpolate_at_sample(struct nir_builder *b, nir_intrinsic_instr *interp, void *data)
{
   if (interp->intrinsic != nir_intrinsic_interp_deref_at_sample)
      return false;

   b->cursor = nir_before_instr(&interp->instr);
   nir_def *res = nir_load_deref(b, nir_src_as_deref(interp->src[0]));
   nir_def_rewrite_uses(&interp->def, res);

   return true;
}

struct zink_shader_object
zink_shader_compile(struct zink_screen *screen, bool can_shobj, struct zink_shader *zs,
                    nir_shader *nir, const struct zink_shader_key *key, const void *extra_data, struct zink_program *pg)
{
   bool need_optimize = true;
   bool inlined_uniforms = false;

   NIR_PASS(_, nir, add_derefs);
   NIR_PASS(_, nir, nir_lower_fragcolor, nir->info.fs.color_is_dual_source ? 1 : 8);
   if (key) {
      if (key->inline_uniforms) {
         NIR_PASS(_, nir, nir_inline_uniforms,
                    nir->info.num_inlinable_uniforms,
                    key->base.inlined_uniform_values,
                    nir->info.inlinable_uniform_dw_offsets);

         inlined_uniforms = true;
      }

      /* TODO: use a separate mem ctx here for ralloc */

      if (!screen->optimal_keys) {
         switch (zs->info.stage) {
         case MESA_SHADER_VERTEX: {
            uint32_t decomposed_attrs = 0, decomposed_attrs_without_w = 0;
            const struct zink_vs_key *vs_key = zink_vs_key(key);
            switch (vs_key->size) {
            case 4:
               decomposed_attrs = vs_key->u32.decomposed_attrs;
               decomposed_attrs_without_w = vs_key->u32.decomposed_attrs_without_w;
               break;
            case 2:
               decomposed_attrs = vs_key->u16.decomposed_attrs;
               decomposed_attrs_without_w = vs_key->u16.decomposed_attrs_without_w;
               break;
            case 1:
               decomposed_attrs = vs_key->u8.decomposed_attrs;
               decomposed_attrs_without_w = vs_key->u8.decomposed_attrs_without_w;
               break;
            default: break;
            }
            if (decomposed_attrs || decomposed_attrs_without_w)
               NIR_PASS(_, nir, decompose_attribs, decomposed_attrs, decomposed_attrs_without_w);
            break;
         }

         case MESA_SHADER_GEOMETRY:
            if (zink_gs_key(key)->lower_line_stipple) {
               NIR_PASS(_, nir, lower_line_stipple_gs, zink_gs_key(key)->line_rectangular);
               NIR_PASS(_, nir, nir_lower_var_copies);
               need_optimize = true;
            }

            if (zink_gs_key(key)->lower_line_smooth) {
               NIR_PASS(_, nir, lower_line_smooth_gs);
               NIR_PASS(_, nir, nir_lower_var_copies);
               need_optimize = true;
            }

            if (zink_gs_key(key)->lower_gl_point) {
               NIR_PASS(_, nir, lower_gl_point_gs);
               need_optimize = true;
            }

            if (zink_gs_key(key)->lower_pv_mode) {
               NIR_PASS(_, nir, lower_pv_mode_gs, zink_gs_key(key)->lower_pv_mode);
               need_optimize = true; //TODO verify that this is required
            }
            break;

         default:
            break;
         }
      }

      switch (zs->info.stage) {
      case MESA_SHADER_VERTEX:
      case MESA_SHADER_TESS_EVAL:
      case MESA_SHADER_GEOMETRY:
         if (zink_vs_key_base(key)->last_vertex_stage) {
            if (!zink_vs_key_base(key)->clip_halfz && !screen->info.have_EXT_depth_clip_control) {
               NIR_PASS(_, nir, nir_lower_clip_halfz);
            }
            if (zink_vs_key_base(key)->push_drawid) {
               NIR_PASS(_, nir, lower_drawid);
            }
         } else {
            nir->xfb_info = NULL;
         }
         if (zink_vs_key_base(key)->robust_access)
            NIR_PASS(need_optimize, nir, lower_txf_lod_robustness);
         break;
      case MESA_SHADER_FRAGMENT:
         if (zink_fs_key(key)->lower_line_smooth) {
            NIR_PASS(_, nir, lower_line_smooth_fs,
                       zink_fs_key(key)->lower_line_stipple);
            need_optimize = true;
         } else if (zink_fs_key(key)->lower_line_stipple)
               NIR_PASS(_, nir, lower_line_stipple_fs);

         if (zink_fs_key(key)->lower_point_smooth) {
            NIR_PASS(_, nir, nir_lower_point_smooth, false);
            NIR_PASS(_, nir, nir_lower_discard_if, nir_lower_demote_if_to_cf | nir_lower_terminate_if_to_cf);
            nir->info.fs.uses_discard = true;
            need_optimize = true;
         }

         if (zink_fs_key(key)->robust_access)
            NIR_PASS(need_optimize, nir, lower_txf_lod_robustness);

         if (!zink_fs_key_base(key)->samples && zink_shader_uses_samples(zs)) {
            /* VK will always use gl_SampleMask[] values even if sample count is 0,
            * so we need to skip this write here to mimic GL's behavior of ignoring it
            */
            nir_foreach_shader_out_variable(var, nir) {
               if (var->data.location == FRAG_RESULT_SAMPLE_MASK)
                  var->data.mode = nir_var_shader_temp;
            }
            nir_fixup_deref_modes(nir);
            NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_temp, NULL);
            NIR_PASS(_, nir, nir_shader_intrinsics_pass, remove_interpolate_at_sample,
                       nir_metadata_control_flow, NULL);

            need_optimize = true;
         }
         if (zink_fs_key_base(key)->force_dual_color_blend && nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DATA1)) {
            NIR_PASS(_, nir, lower_dual_blend);
         }
         if (zink_fs_key_base(key)->coord_replace_bits)
            NIR_PASS(_, nir, nir_lower_texcoord_replace, zink_fs_key_base(key)->coord_replace_bits, true, false);
         if (zink_fs_key_base(key)->point_coord_yinvert)
            NIR_PASS(_, nir, invert_point_coord);
         if (zink_fs_key_base(key)->force_persample_interp || zink_fs_key_base(key)->fbfetch_ms) {
            nir_foreach_shader_in_variable(var, nir)
               var->data.sample = true;
            nir->info.fs.uses_sample_qualifier = true;
            nir->info.fs.uses_sample_shading = true;
         }
         if (zs->fs.legacy_shadow_mask && !key->base.needs_zs_shader_swizzle)
            NIR_PASS(need_optimize, nir, lower_zs_swizzle_tex, zink_fs_key_base(key)->shadow_needs_shader_swizzle ? extra_data : NULL, true);
         if (nir->info.fs.uses_fbfetch_output) {
            nir_variable *fbfetch = NULL;
            NIR_PASS(_, nir, lower_fbfetch, &fbfetch, zink_fs_key_base(key)->fbfetch_ms);
            /* old variable must be deleted to avoid spirv errors */
            fbfetch->data.mode = nir_var_shader_temp;
            nir_fixup_deref_modes(nir);
            NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_temp, NULL);
            need_optimize = true;
         }
         nir_foreach_shader_in_variable_safe(var, nir) {
            if (!is_texcoord(MESA_SHADER_FRAGMENT, var) || var->data.driver_location != -1)
               continue;
            nir_shader_instructions_pass(nir, rewrite_read_as_0, nir_metadata_dominance, var);
            var->data.mode = nir_var_shader_temp;
            nir_fixup_deref_modes(nir);
            NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_temp, NULL);
            need_optimize = true;
         }
         break;
      case MESA_SHADER_COMPUTE:
         if (zink_cs_key(key)->robust_access)
            NIR_PASS(need_optimize, nir, lower_txf_lod_robustness);
         break;
      default: break;
      }
      if (key->base.needs_zs_shader_swizzle) {
         assert(extra_data);
         NIR_PASS(need_optimize, nir, lower_zs_swizzle_tex, extra_data, false);
      }
      if (key->base.nonseamless_cube_mask) {
         NIR_PASS(_, nir, zink_lower_cubemap_to_array, key->base.nonseamless_cube_mask);
         need_optimize = true;
      }
   }
   if (screen->driconf.inline_uniforms) {
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_mem_global | nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_mem_shared, NULL, NULL);
      NIR_PASS(_, nir, rewrite_bo_access, screen);
      NIR_PASS(_, nir, remove_bo_access, zs);
      need_optimize = true;
   }
   if (inlined_uniforms) {
      optimize_nir(nir, zs, true);

      /* This must be done again. */
      NIR_PASS(_, nir, nir_io_add_const_offset_to_base, nir_var_shader_in |
                                                       nir_var_shader_out);

      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      if (impl->ssa_alloc > ZINK_ALWAYS_INLINE_LIMIT)
         zs->can_inline = false;
   } else if (need_optimize)
      optimize_nir(nir, zs, true);
   bool has_sparse = false;
   NIR_PASS(has_sparse, nir, lower_sparse);
   if (has_sparse)
      optimize_nir(nir, zs, false);
   
   struct zink_shader_object obj = compile_module(screen, zs, nir, can_shobj, pg);
   ralloc_free(nir);
   return obj;
}

struct zink_shader_object
zink_shader_compile_separate(struct zink_screen *screen, struct zink_shader *zs)
{
   nir_shader *nir = zs->nir;
   /* TODO: maybe compile multiple variants for different set counts for compact mode? */
   int set = zs->info.stage == MESA_SHADER_FRAGMENT;
   if (screen->info.have_EXT_shader_object)
      set = zs->info.stage;
   unsigned offsets[4];
   zink_descriptor_shader_get_binding_offsets(zs, offsets);
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_uniform | nir_var_image) {
      if (var->data.descriptor_set == screen->desc_set_id[ZINK_DESCRIPTOR_BINDLESS])
         continue;
      var->data.descriptor_set = set;
      switch (var->data.mode) {
      case nir_var_mem_ubo:
            var->data.binding = !!var->data.driver_location;
            break;
      case nir_var_uniform:
         if (glsl_type_is_sampler(glsl_without_array(var->type)))
            var->data.binding += offsets[1];
         break;
      case nir_var_mem_ssbo:
         var->data.binding += offsets[2];
         break;
      case nir_var_image:
         var->data.binding += offsets[3];
         break;
      default: break;
      }
   }
   NIR_PASS(_, nir, add_derefs);
   NIR_PASS(_, nir, nir_lower_fragcolor, nir->info.fs.color_is_dual_source ? 1 : 8);
   if (screen->driconf.inline_uniforms) {
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_mem_global | nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_mem_shared, NULL, NULL);
      NIR_PASS(_, nir, rewrite_bo_access, screen);
      NIR_PASS(_, nir, remove_bo_access, zs);
   }
   optimize_nir(nir, zs, true);
   zink_descriptor_shader_init(screen, zs);
   nir_shader *nir_clone = NULL;
   if (screen->info.have_EXT_shader_object)
      nir_clone = nir_shader_clone(nir, nir);
   struct zink_shader_object obj = compile_module(screen, zs, nir, true, NULL);
   if (screen->info.have_EXT_shader_object && !zs->info.internal) {
      /* always try to pre-generate a tcs in case it's needed */
      if (zs->info.stage == MESA_SHADER_TESS_EVAL) {
         nir_shader *nir_tcs = NULL;
         /* use max pcp for compat */
         zs->non_fs.generated_tcs = zink_shader_tcs_create(screen, 32);
         zink_shader_tcs_init(screen, zs->non_fs.generated_tcs, nir_clone, &nir_tcs);
         nir_tcs->info.separate_shader = true;
         zs->non_fs.generated_tcs->precompile.obj = zink_shader_compile_separate(screen, zs->non_fs.generated_tcs);
         ralloc_free(nir_tcs);
         zs->non_fs.generated_tcs->nir = NULL;
      }
   }
   spirv_shader_delete(obj.spirv);
   obj.spirv = NULL;
   return obj;
}

static bool
lower_baseinstance_instr(nir_builder *b, nir_intrinsic_instr *intr,
                         void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_instance_id)
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *def = nir_isub(b, &intr->def, nir_load_base_instance(b));
   nir_def_rewrite_uses_after(&intr->def, def);
   return true;
}

static bool
lower_baseinstance(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_VERTEX)
      return false;
   return nir_shader_intrinsics_pass(shader, lower_baseinstance_instr,
                                     nir_metadata_dominance, NULL);
}

/* gl_nir_lower_buffers makes variables unusable for all UBO/SSBO access
 * so instead we delete all those broken variables and just make new ones
 */
static bool
unbreak_bos(nir_shader *shader, struct zink_shader *zs, bool needs_size)
{
   uint64_t max_ssbo_size = 0;
   uint64_t max_ubo_size = 0;
   uint64_t max_uniform_size = 0;

   if (!shader->info.num_ssbos && !shader->info.num_ubos)
      return false;

   nir_foreach_variable_with_modes(var, shader, nir_var_mem_ssbo | nir_var_mem_ubo) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (type_is_counter(type))
         continue;
      /* be conservative: use the bigger of the interface and variable types to ensure in-bounds access */
      unsigned size = glsl_count_attribute_slots(glsl_type_is_array(var->type) ? var->type : type, false);
      const struct glsl_type *interface_type = var->interface_type ? glsl_without_array(var->interface_type) : NULL;
      if (interface_type) {
         unsigned block_size = glsl_get_explicit_size(interface_type, true);
         if (glsl_get_length(interface_type) == 1) {
            /* handle bare unsized ssbo arrays: glsl_get_explicit_size always returns type-aligned sizes */
            const struct glsl_type *f = glsl_get_struct_field(interface_type, 0);
            if (glsl_type_is_array(f) && !glsl_array_size(f))
               block_size = 0;
         }
         if (block_size) {
            block_size = DIV_ROUND_UP(block_size, sizeof(float) * 4);
            size = MAX2(size, block_size);
         }
      }
      if (var->data.mode == nir_var_mem_ubo) {
         if (var->data.driver_location)
            max_ubo_size = MAX2(max_ubo_size, size);
         else
            max_uniform_size = MAX2(max_uniform_size, size);
      } else {
         max_ssbo_size = MAX2(max_ssbo_size, size);
         if (interface_type) {
            if (glsl_type_is_unsized_array(glsl_get_struct_field(interface_type, glsl_get_length(interface_type) - 1)))
               needs_size = true;
         }
      }
      var->data.mode = nir_var_shader_temp;
   }
   nir_fixup_deref_modes(shader);
   NIR_PASS(_, shader, nir_remove_dead_variables, nir_var_shader_temp, NULL);
   optimize_nir(shader, NULL, true);

   struct glsl_struct_field field = {0};
   field.name = ralloc_strdup(shader, "base");
   if (shader->info.num_ubos) {
      if (shader->num_uniforms && zs->ubos_used & BITFIELD_BIT(0)) {
         field.type = glsl_array_type(glsl_uint_type(), max_uniform_size * 4, 4);
         nir_variable *var = nir_variable_create(shader, nir_var_mem_ubo,
                                                 glsl_array_type(glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD430, false, "struct"), 1, 0),
                                                 "uniform_0@32");
         var->interface_type = var->type;
         var->data.mode = nir_var_mem_ubo;
         var->data.driver_location = 0;
      }

      unsigned num_ubos = shader->info.num_ubos - !!shader->info.first_ubo_is_default_ubo;
      uint32_t ubos_used = zs->ubos_used & ~BITFIELD_BIT(0);
      if (num_ubos && ubos_used) {
         field.type = glsl_array_type(glsl_uint_type(), max_ubo_size * 4, 4);
         /* shrink array as much as possible */
         unsigned first_ubo = ffs(ubos_used) - 2;
         assert(first_ubo < PIPE_MAX_CONSTANT_BUFFERS);
         num_ubos -= first_ubo;
         assert(num_ubos);
         nir_variable *var = nir_variable_create(shader, nir_var_mem_ubo,
                                   glsl_array_type(glsl_struct_type(&field, 1, "struct", false), num_ubos, 0),
                                   "ubos@32");
         var->interface_type = var->type;
         var->data.mode = nir_var_mem_ubo;
         var->data.driver_location = first_ubo + !!shader->info.first_ubo_is_default_ubo;
      }
   }
   if (shader->info.num_ssbos && zs->ssbos_used) {
      /* shrink array as much as possible */
      unsigned first_ssbo = ffs(zs->ssbos_used) - 1;
      assert(first_ssbo < PIPE_MAX_SHADER_BUFFERS);
      unsigned num_ssbos = shader->info.num_ssbos - first_ssbo;
      assert(num_ssbos);
      const struct glsl_type *ssbo_type = glsl_array_type(glsl_uint_type(), needs_size ? 0 : max_ssbo_size * 4, 4);
      field.type = ssbo_type;
      nir_variable *var = nir_variable_create(shader, nir_var_mem_ssbo,
                                              glsl_array_type(glsl_struct_type(&field, 1, "struct", false), num_ssbos, 0),
                                              "ssbos@32");
      var->interface_type = var->type;
      var->data.mode = nir_var_mem_ssbo;
      var->data.driver_location = first_ssbo;
   }
   return true;
}

static uint32_t
get_src_mask_ssbo(unsigned total, nir_src src)
{
   if (nir_src_is_const(src))
      return BITFIELD_BIT(nir_src_as_uint(src));
   return BITFIELD_MASK(total);
}

static uint32_t
get_src_mask_ubo(unsigned total, nir_src src)
{
   if (nir_src_is_const(src))
      return BITFIELD_BIT(nir_src_as_uint(src));
   return BITFIELD_MASK(total) & ~BITFIELD_BIT(0);
}

static bool
analyze_io(struct zink_shader *zs, nir_shader *shader)
{
   bool ret = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (shader->info.stage != MESA_SHADER_KERNEL && instr->type == nir_instr_type_tex) {
            /* gl_nir_lower_samplers_as_deref is where this would normally be set, but zink doesn't use it */
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            int deref_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
            if (deref_idx >= 0) {
               nir_variable *img = nir_deref_instr_get_variable(nir_def_as_deref(tex->src[deref_idx].src.ssa));
               unsigned size = glsl_type_is_array(img->type) ? glsl_get_aoa_size(img->type) : 1;
               BITSET_SET_RANGE(shader->info.textures_used, img->data.driver_location, img->data.driver_location + (size - 1));
            }
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
 
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         switch (intrin->intrinsic) {
         case nir_intrinsic_store_ssbo:
            zs->ssbos_used |= get_src_mask_ssbo(shader->info.num_ssbos, intrin->src[1]);
            break;
 
         case nir_intrinsic_get_ssbo_size: {
            zs->ssbos_used |= get_src_mask_ssbo(shader->info.num_ssbos, intrin->src[0]);
            ret = true;
            break;
         }
         case nir_intrinsic_ssbo_atomic:
         case nir_intrinsic_ssbo_atomic_swap:
         case nir_intrinsic_load_ssbo:
            zs->ssbos_used |= get_src_mask_ssbo(shader->info.num_ssbos, intrin->src[0]);
            break;
         case nir_intrinsic_load_ubo:
         case nir_intrinsic_load_ubo_vec4:
            zs->ubos_used |= get_src_mask_ubo(shader->info.num_ubos, intrin->src[0]);
            break;
         default:
            break;
         }
      }
   }
   return ret;
}

struct zink_bindless_info {
   nir_variable *bindless[4];
   unsigned bindless_set;
};

/* this is a "default" bindless texture used if the shader has no texture variables */
static nir_variable *
create_bindless_texture(nir_shader *nir, nir_tex_instr *tex, unsigned descriptor_set)
{
   unsigned binding = tex->sampler_dim == GLSL_SAMPLER_DIM_BUF ? 1 : 0;
   nir_variable *var;

   const struct glsl_type *sampler_type = glsl_sampler_type(tex->sampler_dim, tex->is_shadow, tex->is_array, GLSL_TYPE_FLOAT);
   var = nir_variable_create(nir, nir_var_uniform, glsl_array_type(sampler_type, ZINK_MAX_BINDLESS_HANDLES, 0), "bindless_texture");
   var->data.descriptor_set = descriptor_set;
   var->data.driver_location = var->data.binding = binding;
   return var;
}

/* this is a "default" bindless image used if the shader has no image variables */
static nir_variable *
create_bindless_image(nir_shader *nir, enum glsl_sampler_dim dim, unsigned descriptor_set)
{
   unsigned binding = dim == GLSL_SAMPLER_DIM_BUF ? 3 : 2;
   nir_variable *var;

   const struct glsl_type *image_type = glsl_image_type(dim, false, GLSL_TYPE_FLOAT);
   var = nir_variable_create(nir, nir_var_image, glsl_array_type(image_type, ZINK_MAX_BINDLESS_HANDLES, 0), "bindless_image");
   var->data.descriptor_set = descriptor_set;
   var->data.driver_location = var->data.binding = binding;
   var->data.image.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   return var;
}

/* rewrite bindless instructions as array deref instructions */
static bool
lower_bindless_instr(nir_builder *b, nir_instr *in, void *data)
{
   struct zink_bindless_info *bindless = data;

   if (in->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(in);
      int idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
      if (idx == -1)
         return false;

      nir_variable *var = tex->sampler_dim == GLSL_SAMPLER_DIM_BUF ? bindless->bindless[1] : bindless->bindless[0];
      if (!var) {
         var = create_bindless_texture(b->shader, tex, bindless->bindless_set);
         if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
            bindless->bindless[1] = var;
         else
            bindless->bindless[0] = var;
      }
      b->cursor = nir_before_instr(in);
      nir_deref_instr *deref = nir_build_deref_var(b, var);
      if (glsl_type_is_array(var->type))
         deref = nir_build_deref_array(b, deref, nir_u2uN(b, tex->src[idx].src.ssa, 32));
      nir_src_rewrite(&tex->src[idx].src, &deref->def);

      /* bindless sampling uses the variable type directly, which means the tex instr has to exactly
       * match up with it in contrast to normal sampler ops where things are a bit more flexible;
       * this results in cases where a shader is passed with sampler2DArray but the tex instr only has
       * 2 components, which explodes spirv compilation even though it doesn't trigger validation errors
       *
       * to fix this, pad the coord src here and fix the tex instr so that ntv will do the "right" thing
       * - Warhammer 40k: Dawn of War III
       */
      unsigned needed_components = glsl_get_sampler_coordinate_components(glsl_without_array(var->type));
      unsigned c = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      unsigned coord_components = nir_src_num_components(tex->src[c].src);
      if (coord_components < needed_components) {
         nir_def *def = nir_pad_vector(b, tex->src[c].src.ssa, needed_components);
         nir_src_rewrite(&tex->src[c].src, def);
         tex->coord_components = needed_components;
      }
      return true;
   }
   if (in->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *instr = nir_instr_as_intrinsic(in);

   nir_intrinsic_op op;
#define OP_SWAP(OP) \
   case nir_intrinsic_bindless_image_##OP: \
      op = nir_intrinsic_image_deref_##OP; \
      break;


   /* convert bindless intrinsics to deref intrinsics */
   switch (instr->intrinsic) {
   OP_SWAP(atomic)
   OP_SWAP(atomic_swap)
   OP_SWAP(format)
   OP_SWAP(load)
   OP_SWAP(order)
   OP_SWAP(samples)
   OP_SWAP(size)
   OP_SWAP(store)
   default:
      return false;
   }

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(instr);
   nir_variable *var = dim == GLSL_SAMPLER_DIM_BUF ? bindless->bindless[3] : bindless->bindless[2];
   if (!var)
      var = create_bindless_image(b->shader, dim, bindless->bindless_set);
   instr->intrinsic = op;
   b->cursor = nir_before_instr(in);
   nir_deref_instr *deref = nir_build_deref_var(b, var);
   if (glsl_type_is_array(var->type))
      deref = nir_build_deref_array(b, deref, nir_u2uN(b, instr->src[0].ssa, 32));
   nir_src_rewrite(&instr->src[0], &deref->def);
   return true;
}

static bool
lower_bindless(nir_shader *shader, struct zink_bindless_info *bindless)
{
   if (!nir_shader_instructions_pass(shader, lower_bindless_instr, nir_metadata_dominance, bindless))
      return false;
   nir_fixup_deref_modes(shader);
   NIR_PASS(_, shader, nir_remove_dead_variables, nir_var_shader_temp, NULL);
   optimize_nir(shader, NULL, true);
   return true;
}

/* convert shader image/texture io variables to int64 handles for bindless indexing */
static bool
lower_bindless_io_instr(nir_builder *b, nir_intrinsic_instr *instr,
                        void *data)
{
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   if (!filter_io_instr(instr, &is_load, &is_input, &is_interp))
      return false;

   nir_variable *var = find_var_with_location_frac(b->shader, nir_intrinsic_io_semantics(instr).location, nir_intrinsic_component(instr), false, is_input ? nir_var_shader_in : nir_var_shader_out);
   if (var->data.bindless)
      return false;
   if (var->data.mode != nir_var_shader_in && var->data.mode != nir_var_shader_out)
      return false;
   if (!glsl_type_is_image(var->type) && !glsl_type_is_sampler(var->type))
      return false;

   var->type = glsl_vector_type(GLSL_TYPE_INT, 2);
   var->data.bindless = 1;
   return true;
}

static bool
lower_bindless_io(nir_shader *shader)
{
   return nir_shader_intrinsics_pass(shader, lower_bindless_io_instr,
                                     nir_metadata_dominance, NULL);
}

static uint32_t
zink_binding(gl_shader_stage stage, VkDescriptorType type, int index, bool compact_descriptors)
{
   if (stage == MESA_SHADER_NONE) {
      UNREACHABLE("not supported");
   } else {
      unsigned base = stage;
      /* clamp compute bindings for better driver efficiency */
      if (gl_shader_stage_is_compute(stage))
         base = 0;
      switch (type) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
         return base * 2 + !!index;

      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
         assert(stage == MESA_SHADER_KERNEL);
         FALLTHROUGH;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
         if (stage == MESA_SHADER_KERNEL) {
            assert(index < PIPE_MAX_SHADER_SAMPLER_VIEWS);
            return index + PIPE_MAX_SAMPLERS;
         }
         FALLTHROUGH;
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         assert(index < PIPE_MAX_SAMPLERS);
         assert(stage != MESA_SHADER_KERNEL);
         return (base * PIPE_MAX_SAMPLERS) + index;

      case VK_DESCRIPTOR_TYPE_SAMPLER:
         assert(index < PIPE_MAX_SAMPLERS);
         assert(stage == MESA_SHADER_KERNEL);
         return index;

      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         return base + (compact_descriptors * (ZINK_GFX_SHADER_COUNT * 2));

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         assert(index < ZINK_MAX_SHADER_IMAGES);
         if (stage == MESA_SHADER_KERNEL)
            return index + (compact_descriptors ? (PIPE_MAX_SAMPLERS + PIPE_MAX_SHADER_SAMPLER_VIEWS) : 0);
         return (base * ZINK_MAX_SHADER_IMAGES) + index + (compact_descriptors * (ZINK_GFX_SHADER_COUNT * PIPE_MAX_SAMPLERS));

      default:
         UNREACHABLE("unexpected type");
      }
   }
}

static void
handle_bindless_var(nir_shader *nir, nir_variable *var, const struct glsl_type *type, struct zink_bindless_info *bindless)
{
   if (glsl_type_is_struct(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++)
         handle_bindless_var(nir, var, glsl_get_struct_field(type, i), bindless);
      return;
   }

   /* just a random scalar in a struct */
   if (!glsl_type_is_image(type) && !glsl_type_is_sampler(type))
      return;

   VkDescriptorType vktype = glsl_type_is_image(type) ? zink_image_type(type) : zink_sampler_type(type);
   unsigned binding;
   switch (vktype) {
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         binding = 0;
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
         binding = 1;
         break;
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         binding = 2;
         break;
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         binding = 3;
         break;
      default:
         UNREACHABLE("unknown");
   }
   if (!bindless->bindless[binding]) {
      bindless->bindless[binding] = nir_variable_clone(var, nir);
      bindless->bindless[binding]->data.bindless = 0;
      bindless->bindless[binding]->data.descriptor_set = bindless->bindless_set;
      bindless->bindless[binding]->type = glsl_array_type(type, ZINK_MAX_BINDLESS_HANDLES, 0);
      bindless->bindless[binding]->data.driver_location = bindless->bindless[binding]->data.binding = binding;
      if (!bindless->bindless[binding]->data.image.format)
         bindless->bindless[binding]->data.image.format = PIPE_FORMAT_R8G8B8A8_UNORM;
      nir_shader_add_variable(nir, bindless->bindless[binding]);
   } else {
      assert(glsl_get_sampler_dim(glsl_without_array(bindless->bindless[binding]->type)) == glsl_get_sampler_dim(glsl_without_array(var->type)));
   }
   var->data.mode = nir_var_shader_temp;
}

static bool
convert_1d_shadow_tex(nir_builder *b, nir_instr *instr, void *data)
{
   struct zink_screen *screen = data;
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_1D || !tex->is_shadow)
      return false;
   if (tex->is_sparse && screen->need_2D_sparse) {
      /* no known case of this exists: only nvidia can hit it, and nothing uses it */
      mesa_loge("unhandled/unsupported 1D sparse texture!");
      abort();
   }
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   b->cursor = nir_before_instr(instr);
   tex->coord_components++;
   unsigned srcs[] = {
      nir_tex_src_coord,
      nir_tex_src_offset,
      nir_tex_src_ddx,
      nir_tex_src_ddy,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(srcs); i++) {
      unsigned c = nir_tex_instr_src_index(tex, srcs[i]);
      if (c == -1)
         continue;
      if (tex->src[c].src.ssa->num_components == tex->coord_components)
         continue;
      nir_def *def;
      nir_def *zero = nir_imm_zero(b, 1, tex->src[c].src.ssa->bit_size);
      if (tex->src[c].src.ssa->num_components == 1)
         def = nir_vec2(b, tex->src[c].src.ssa, zero);
      else
         def = nir_vec3(b, nir_channel(b, tex->src[c].src.ssa, 0), zero, nir_channel(b, tex->src[c].src.ssa, 1));
      nir_src_rewrite(&tex->src[c].src, def);
   }
   b->cursor = nir_after_instr(instr);
   unsigned needed_components = nir_tex_instr_dest_size(tex);
   unsigned num_components = tex->def.num_components;
   if (needed_components > num_components) {
      tex->def.num_components = needed_components;
      assert(num_components < 3);
      /* take either xz or just x since this is promoted to 2D from 1D */
      uint32_t mask = num_components == 2 ? (1|4) : 1;
      nir_def *dst = nir_channels(b, &tex->def, mask);
      nir_def_rewrite_uses_after(&tex->def, dst);
   }
   return true;
}

static bool
lower_1d_shadow(nir_shader *shader, struct zink_screen *screen)
{
   bool found = false;
   nir_foreach_variable_with_modes(var, shader, nir_var_uniform | nir_var_image) {
      const struct glsl_type *type = glsl_without_array(var->type);
      unsigned length = glsl_get_length(var->type);
      if (!glsl_type_is_sampler(type) || !glsl_sampler_type_is_shadow(type) || glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_1D)
         continue;
      const struct glsl_type *sampler = glsl_sampler_type(GLSL_SAMPLER_DIM_2D, true, glsl_sampler_type_is_array(type), glsl_get_sampler_result_type(type));
      var->type = type != var->type ? glsl_array_type(sampler, length, glsl_get_explicit_stride(var->type)) : sampler;

      found = true;
   }
   if (found) {
      nir_shader_instructions_pass(shader, convert_1d_shadow_tex, nir_metadata_dominance, screen);
      nir_fixup_deref_types(shader);
   }
   return found;
}

static void
scan_nir(struct zink_screen *screen, nir_shader *shader, struct zink_shader *zs)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block_safe(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               zs->sinfo.have_sparse |= tex->is_sparse;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_image_deref_load ||
                intr->intrinsic == nir_intrinsic_image_deref_sparse_load ||
                intr->intrinsic == nir_intrinsic_image_deref_store ||
                intr->intrinsic == nir_intrinsic_image_deref_atomic ||
                intr->intrinsic == nir_intrinsic_image_deref_atomic_swap ||
                intr->intrinsic == nir_intrinsic_image_deref_size ||
                intr->intrinsic == nir_intrinsic_image_deref_samples ||
                intr->intrinsic == nir_intrinsic_image_deref_format ||
                intr->intrinsic == nir_intrinsic_image_deref_order) {

                nir_variable *var = nir_intrinsic_get_var(intr, 0);

                /* Structs have been lowered already, so get_aoa_size is sufficient. */
                const unsigned size =
                   glsl_type_is_array(var->type) ? glsl_get_aoa_size(var->type) : 1;
                BITSET_SET_RANGE(shader->info.images_used, var->data.binding,
                                 var->data.binding + (MAX2(size, 1) - 1));
            }
            if (intr->intrinsic == nir_intrinsic_is_sparse_texels_resident ||
                intr->intrinsic == nir_intrinsic_image_deref_sparse_load)
               zs->sinfo.have_sparse = true;

            bool is_load = false;
            bool is_input = false;
            bool is_interp = false;
            if (filter_io_instr(intr, &is_load, &is_input, &is_interp)) {
               nir_io_semantics s = nir_intrinsic_io_semantics(intr);
               if (io_instr_is_arrayed(intr) && s.location < VARYING_SLOT_PATCH0) {
                  if (is_input)
                     zs->arrayed_inputs |= BITFIELD64_BIT(s.location);
                  else
                     zs->arrayed_outputs |= BITFIELD64_BIT(s.location);
               }
               /* TODO: delete this once #10826 is fixed */
               if (!(is_input && shader->info.stage == MESA_SHADER_VERTEX)) {
                  if (is_clipcull_dist(s.location)) {
                     unsigned frac = nir_intrinsic_component(intr) + 1;
                     if (s.location < VARYING_SLOT_CULL_DIST0) {
                        if (s.location == VARYING_SLOT_CLIP_DIST1)
                           frac += 4;
                        shader->info.clip_distance_array_size = MAX3(shader->info.clip_distance_array_size, frac, s.num_slots);
                     } else {
                        if (s.location == VARYING_SLOT_CULL_DIST1)
                           frac += 4;
                        shader->info.cull_distance_array_size = MAX3(shader->info.cull_distance_array_size, frac, s.num_slots);
                     }
                  }
               }
            }

            static bool warned = false;
            if (!screen->info.have_EXT_shader_atomic_float && !screen->is_cpu && !warned) {
               switch (intr->intrinsic) {
               case nir_intrinsic_image_deref_atomic: {
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (nir_intrinsic_atomic_op(intr) == nir_atomic_op_iadd &&
                      util_format_is_float(var->data.image.format))
                     fprintf(stderr, "zink: Vulkan driver missing VK_EXT_shader_atomic_float but attempting to do atomic ops!\n");
                  break;
               }
               default:
                  break;
               }
            }
         }
      }
   }
}

static bool
match_tex_dests_instr(nir_builder *b, nir_tex_instr *tex, void *data,
                      bool pre)
{
   if (tex->op == nir_texop_txs || tex->op == nir_texop_lod)
      return false;
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   nir_variable *var = NULL;
   if (handle != -1) {
      if (pre)
         return false;
      var = nir_deref_instr_get_variable(nir_src_as_deref(tex->src[handle].src));
   } else {
      var = nir_deref_instr_get_variable(nir_def_as_deref(tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_deref)].src.ssa));
   }
   if (pre) {
      flag_shadow_tex_instr(b, tex, var, data);
      return false;
   }
   return !!rewrite_tex_dest(b, tex, var, data);
}

static bool
match_tex_dests_instr_pre(nir_builder *b, nir_tex_instr *in, void *data)
{
   return match_tex_dests_instr(b, in, data, true);
}

static bool
match_tex_dests_instr_post(nir_builder *b, nir_tex_instr *in, void *data)
{
   return match_tex_dests_instr(b, in, data, false);
}

static bool
match_tex_dests(nir_shader *shader, struct zink_shader *zs, bool pre_mangle)
{
   return nir_shader_tex_pass(shader, pre_mangle ? match_tex_dests_instr_pre : match_tex_dests_instr_post, nir_metadata_dominance, zs);
}

static bool
split_bitfields_instr(nir_builder *b, nir_alu_instr *alu, void *data)
{
   switch (alu->op) {
   case nir_op_ubitfield_extract:
   case nir_op_ibitfield_extract:
   case nir_op_bitfield_insert:
      break;
   default:
      return false;
   }
   unsigned num_components = alu->def.num_components;
   if (num_components == 1)
      return false;
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *dests[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < num_components; i++) {
      if (alu->op == nir_op_bitfield_insert)
         dests[i] = nir_bitfield_insert(b,
                                        nir_channel(b, alu->src[0].src.ssa, alu->src[0].swizzle[i]),
                                        nir_channel(b, alu->src[1].src.ssa, alu->src[1].swizzle[i]),
                                        nir_channel(b, alu->src[2].src.ssa, alu->src[2].swizzle[i]),
                                        nir_channel(b, alu->src[3].src.ssa, alu->src[3].swizzle[i]));
      else if (alu->op == nir_op_ubitfield_extract)
         dests[i] = nir_ubitfield_extract(b,
                                          nir_channel(b, alu->src[0].src.ssa, alu->src[0].swizzle[i]),
                                          nir_channel(b, alu->src[1].src.ssa, alu->src[1].swizzle[i]),
                                          nir_channel(b, alu->src[2].src.ssa, alu->src[2].swizzle[i]));
      else
         dests[i] = nir_ibitfield_extract(b,
                                          nir_channel(b, alu->src[0].src.ssa, alu->src[0].swizzle[i]),
                                          nir_channel(b, alu->src[1].src.ssa, alu->src[1].swizzle[i]),
                                          nir_channel(b, alu->src[2].src.ssa, alu->src[2].swizzle[i]));
   }
   nir_def *dest = nir_vec(b, dests, num_components);
   nir_def_rewrite_uses_after_instr(&alu->def, dest, &alu->instr);
   nir_instr_remove(&alu->instr);
   return true;
}


static bool
split_bitfields(nir_shader *shader)
{
   return nir_shader_alu_pass(shader, split_bitfields_instr,
                              nir_metadata_dominance, NULL);
}

static bool
strip_tex_ms_instr(nir_builder *b, nir_instr *in, void *data)
{
   if (in->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(in);
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_samples:
      b->cursor = nir_before_instr(in);
      nir_def_rewrite_uses_after_instr(&intr->def, nir_imm_zero(b, 1, intr->def.bit_size), in);
      nir_instr_remove(in);
      break;
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
      break;
   default:
      return false;
   }
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   if (dim != GLSL_SAMPLER_DIM_MS)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   if (parent) {
      parent->type = var->type;
      deref->type = glsl_without_array(var->type);
   } else {
      deref->type = var->type;
   }
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   return true;
}


static bool
strip_tex_ms(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_image_variable(var, shader) {
      const struct glsl_type *bare_type = glsl_without_array(var->type);
      if (glsl_get_sampler_dim(bare_type) != GLSL_SAMPLER_DIM_MS)
         continue;
      unsigned array_size = 0;
      if (glsl_type_is_array(var->type))
         array_size = glsl_array_size(var->type);

      const struct glsl_type *new_type = glsl_image_type(GLSL_SAMPLER_DIM_2D, glsl_sampler_type_is_array(bare_type), glsl_get_sampler_result_type(bare_type));
      if (array_size)
         new_type = glsl_array_type(new_type, array_size, glsl_get_explicit_stride(var->type));
      var->type = new_type;
      progress = true;
   }
   if (!progress)
      return false;
   return nir_shader_instructions_pass(shader, strip_tex_ms_instr, nir_metadata_all, NULL);
}

static void
rewrite_cl_derefs(nir_shader *nir, nir_variable *var)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            nir_variable *img = nir_deref_instr_get_variable(deref);
            if (img != var)
               continue;
            if (glsl_type_is_array(var->type)) {
               if (deref->deref_type == nir_deref_type_array)
                  deref->type = glsl_without_array(var->type);
               else
                  deref->type = var->type;
            } else {
               deref->type = var->type;
            }
         }
      }
   }
}

static void
type_image(nir_shader *nir, nir_variable *var)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_image_deref_load ||
               intr->intrinsic == nir_intrinsic_image_deref_sparse_load ||
               intr->intrinsic == nir_intrinsic_image_deref_store ||
               intr->intrinsic == nir_intrinsic_image_deref_atomic ||
               intr->intrinsic == nir_intrinsic_image_deref_atomic_swap ||
               intr->intrinsic == nir_intrinsic_image_deref_samples ||
               intr->intrinsic == nir_intrinsic_image_deref_format ||
               intr->intrinsic == nir_intrinsic_image_deref_order) {
               nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
               nir_variable *img = nir_deref_instr_get_variable(deref);
               if (img != var)
                  continue;

               nir_alu_type alu_type;
               if (nir_intrinsic_has_src_type(intr))
                  alu_type = nir_intrinsic_src_type(intr);
               else
                  alu_type = nir_intrinsic_dest_type(intr);

               const struct glsl_type *type = glsl_without_array(var->type);
               if (glsl_get_sampler_result_type(type) != GLSL_TYPE_VOID) {
                  assert(glsl_get_sampler_result_type(type) == nir_get_glsl_base_type_for_nir_type(alu_type));
                  continue;
               }
               const struct glsl_type *img_type = glsl_image_type(glsl_get_sampler_dim(type), glsl_sampler_type_is_array(type), nir_get_glsl_base_type_for_nir_type(alu_type));
               if (glsl_type_is_array(var->type))
                  img_type = glsl_array_type(img_type, glsl_array_size(var->type), glsl_get_explicit_stride(var->type));
               var->type = img_type;
               rewrite_cl_derefs(nir, var);
               return;
            }
         }
      }
   }
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_image_deref_size)
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            nir_variable *img = nir_deref_instr_get_variable(deref);
            if (img != var)
               continue;
            nir_alu_type alu_type = nir_type_uint32;
            const struct glsl_type *type = glsl_without_array(var->type);
            if (glsl_get_sampler_result_type(type) != GLSL_TYPE_VOID) {
               continue;
            }
            const struct glsl_type *img_type = glsl_image_type(glsl_get_sampler_dim(type), glsl_sampler_type_is_array(type), nir_get_glsl_base_type_for_nir_type(alu_type));
            if (glsl_type_is_array(var->type))
               img_type = glsl_array_type(img_type, glsl_array_size(var->type), glsl_get_explicit_stride(var->type));
            var->type = img_type;
            rewrite_cl_derefs(nir, var);
            return;
         }
      }
   }
   var->data.mode = nir_var_shader_temp;
}

static bool
type_sampler_vars(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            nir_variable *var = nir_deref_instr_get_variable(nir_def_as_deref(tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_deref)].src.ssa));
            assert(var);
            if (glsl_get_sampler_result_type(glsl_without_array(var->type)) != GLSL_TYPE_VOID &&
                nir_tex_instr_is_query(tex))
               continue;
            const struct glsl_type *img_type = glsl_sampler_type(glsl_get_sampler_dim(glsl_without_array(var->type)), tex->is_shadow, tex->is_array, nir_get_glsl_base_type_for_nir_type(tex->dest_type));
            unsigned size = glsl_type_is_array(var->type) ? glsl_array_size(var->type) : 1;
            if (size > 1)
               img_type = glsl_array_type(img_type, size, 0);
            var->type = img_type;
            progress = true;
         }
      }
   }
   return progress;
}

static bool
type_images(nir_shader *nir)
{
   bool progress = false;
   progress |= type_sampler_vars(nir);
   nir_foreach_variable_with_modes(var, nir, nir_var_image) {
      type_image(nir, var);
      progress = true;
   }
   if (progress) {
      nir_fixup_deref_types(nir);
      nir_fixup_deref_modes(nir);
   }
   return progress;
}

/* attempt to assign io for separate shaders */
static bool
fixup_io_locations(nir_shader *nir)
{
   nir_variable_mode modes;
   if (nir->info.stage != MESA_SHADER_FRAGMENT && nir->info.stage != MESA_SHADER_VERTEX)
      modes = nir_var_shader_in | nir_var_shader_out;
   else
      modes = nir->info.stage == MESA_SHADER_FRAGMENT ? nir_var_shader_in : nir_var_shader_out;
   u_foreach_bit(mode, modes) {
      nir_variable_mode m = BITFIELD_BIT(mode);
      if ((m == nir_var_shader_in && ((nir->info.inputs_read & BITFIELD64_MASK(VARYING_SLOT_VAR1)) == nir->info.inputs_read)) ||
          (m == nir_var_shader_out && ((nir->info.outputs_written | nir->info.outputs_read) & BITFIELD64_MASK(VARYING_SLOT_VAR1)) == (nir->info.outputs_written | nir->info.outputs_read))) {
         /* this is a special heuristic to catch ARB/fixedfunc shaders which have different rules:
          * - i/o interface blocks don't need to match
          * - any location can be present or not
          * - it just has to work
          *
          * VAR0 is the only user varying that mesa can produce in this case, so overwrite POS
          * since it's a builtin and yolo it with all the other legacy crap
          */
         nir_foreach_variable_with_modes(var, nir, m) {
            if (nir_slot_is_sysval_output(var->data.location, MESA_SHADER_NONE))
               continue;
            if (var->data.location == VARYING_SLOT_VAR0)
               var->data.driver_location = 0;
            else if (var->data.patch)
               var->data.driver_location = var->data.location - VARYING_SLOT_PATCH0;
            else
               var->data.driver_location = var->data.location;
         }
         continue;
      }
      /* i/o interface blocks are required to be EXACT matches between stages:
      * iterate over all locations and set locations incrementally
      */
      unsigned slot = 0;
      for (unsigned i = 0; i < VARYING_SLOT_TESS_MAX; i++) {
         if (nir_slot_is_sysval_output(i, MESA_SHADER_NONE))
            continue;
         bool found = false;
         unsigned size = 0;
         nir_foreach_variable_with_modes(var, nir, m) {
            if (var->data.location != i)
               continue;
            /* only add slots for non-component vars or first-time component vars */
            if (!var->data.location_frac || !size) {
               /* ensure variable is given enough slots */
               if (nir_is_arrayed_io(var, nir->info.stage))
                  size += glsl_count_vec4_slots(glsl_get_array_element(var->type), false, false);
               else
                  size += glsl_count_vec4_slots(var->type, false, false);
            }
            if (var->data.patch)
               var->data.driver_location = var->data.location - VARYING_SLOT_PATCH0;
            else
               var->data.driver_location = slot;
            found = true;
         }
         slot += size;
         if (found) {
            /* ensure the consumed slots aren't double iterated */
            i += size - 1;
         } else {
            /* locations used between stages are not required to be contiguous */
            if (i >= VARYING_SLOT_VAR0)
               slot++;
         }
      }
   }

   nir_shader_preserve_all_metadata(nir);

   return false;
}

static uint64_t
zink_flat_flags(struct nir_shader *shader)
{
   uint64_t flat_flags = 0;
   nir_foreach_shader_in_variable(var, shader) {
      if (var->data.interpolation == INTERP_MODE_FLAT)
         flat_flags |= BITFIELD64_BIT(var->data.location);
   }

   return flat_flags;
}

struct rework_io_state {
   /* these are search criteria */
   bool indirect_only;
   unsigned location;
   nir_variable_mode mode;
   gl_shader_stage stage;
   nir_shader *nir;
   const char *name;

   /* these are found by scanning */
   bool arrayed_io;
   bool medium_precision;
   bool fb_fetch_output;
   bool dual_source_blend_index;
   uint32_t component_mask;
   uint32_t ignored_component_mask;
   unsigned array_size;
   unsigned bit_size;
   unsigned base;
   nir_alu_type type;
   /* must be last */
   char *newname;
};

/* match an existing variable against the rework state */
static nir_variable *
find_rework_var(nir_shader *nir, struct rework_io_state *ris)
{
   nir_foreach_variable_with_modes(var, nir, ris->mode) {
      const struct glsl_type *type = var->type;
      if (nir_is_arrayed_io(var, nir->info.stage))
         type = glsl_get_array_element(type);
      if (var->data.fb_fetch_output != ris->fb_fetch_output)
         continue;
      if (nir->info.stage == MESA_SHADER_FRAGMENT && ris->mode == nir_var_shader_out && ris->dual_source_blend_index != var->data.index)
         continue;
      unsigned num_slots = var->data.compact ? DIV_ROUND_UP(glsl_array_size(type), 4) : glsl_count_attribute_slots(type, false);
      if (var->data.location > ris->location + ris->array_size || var->data.location + num_slots <= ris->location)
         continue;
      unsigned num_components = glsl_get_vector_elements(glsl_without_array(type));
      assert(!glsl_type_contains_64bit(type));
      uint32_t component_mask = ris->component_mask ? ris->component_mask : BITFIELD_MASK(4);
      if (BITFIELD_RANGE(var->data.location_frac, num_components) & component_mask)
         return var;
   }
   return NULL;
}

static void
update_io_var_name(struct rework_io_state *ris, const char *name)
{
   if (!(zink_debug & (ZINK_DEBUG_NIR | ZINK_DEBUG_SPIRV)))
      return;
   if (!name)
      return;
   if (ris->name && !strcmp(ris->name, name))
      return;
   if (ris->newname && !strcmp(ris->newname, name))
      return;
   if (ris->newname) {
      ris->newname = ralloc_asprintf(ris->nir, "%s_%s", ris->newname, name);
   } else if (ris->name) {
      ris->newname = ralloc_asprintf(ris->nir, "%s_%s", ris->name, name);
   } else {
      ris->newname = ralloc_strdup(ris->nir, name);
   }
}

/* check/update tracking state for variable info */
static void
update_io_var_state(nir_intrinsic_instr *intr, struct rework_io_state *ris)
{
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   filter_io_instr(intr, &is_load, &is_input, &is_interp);
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned frac = nir_intrinsic_component(intr);
   /* the mask of components for the instruction */
   uint32_t cmask = is_load ? BITFIELD_RANGE(frac, intr->num_components) : (nir_intrinsic_write_mask(intr) << frac);

   /* always check for existing variables first */
   struct rework_io_state test = {
      .location = ris->location,
      .mode = ris->mode,
      .stage = ris->stage,
      .arrayed_io = io_instr_is_arrayed(intr),
      .medium_precision = sem.medium_precision,
      .fb_fetch_output = sem.fb_fetch_output,
      .dual_source_blend_index = sem.dual_source_blend_index,
      .component_mask = cmask,
      .array_size = sem.num_slots > 1 ? sem.num_slots : 0,
   };
   if (find_rework_var(ris->nir, &test))
      return;

   /* filter ignored components to scan later:
    * - ignore no-overlapping-components case
    * - always match fbfetch and dual src blend
    */
   if (ris->component_mask &&
       (!(ris->component_mask & cmask) || ris->fb_fetch_output != sem.fb_fetch_output || ris->dual_source_blend_index != sem.dual_source_blend_index)) {
      ris->ignored_component_mask |= cmask;
      return;
   }

   assert(!ris->indirect_only || sem.num_slots > 1);
   if (sem.num_slots > 1)
      ris->array_size = MAX2(ris->array_size, sem.num_slots);

   assert(!ris->component_mask || ris->arrayed_io == io_instr_is_arrayed(intr));
   ris->arrayed_io = io_instr_is_arrayed(intr);

   ris->component_mask |= cmask;

   unsigned bit_size = is_load ? intr->def.bit_size : nir_src_bit_size(intr->src[0]);
   assert(!ris->bit_size || ris->bit_size == bit_size);
   ris->bit_size = bit_size;

   nir_alu_type type = is_load ? nir_intrinsic_dest_type(intr) : nir_intrinsic_src_type(intr);
   if (ris->type) {
      /* in the case of clashing types, this heuristic guarantees some semblance of a match */
      if (ris->type & nir_type_float || type & nir_type_float) {
         ris->type = nir_type_float | bit_size;
      } else if (ris->type & nir_type_int || type & nir_type_int) {
         ris->type = nir_type_int | bit_size;
      } else if (ris->type & nir_type_uint || type & nir_type_uint) {
         ris->type = nir_type_uint | bit_size;
      } else {
         assert(bit_size == 1);
         ris->type = nir_type_bool;
      }
   } else {
      ris->type = type;
   }

   update_io_var_name(ris, intr->name);

   ris->medium_precision |= sem.medium_precision;
   ris->fb_fetch_output |= sem.fb_fetch_output;
   ris->dual_source_blend_index |= sem.dual_source_blend_index;
   if (ris->stage == MESA_SHADER_VERTEX && ris->mode == nir_var_shader_in)
      ris->base = nir_intrinsic_base(intr);
}

/* instruction-level scanning for variable data */
static bool
scan_io_var_usage(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct rework_io_state *ris = data;
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   /* mode-based filtering */
   if (!filter_io_instr(intr, &is_load, &is_input, &is_interp))
      return false;
   if (ris->mode == nir_var_shader_in) {
      if (!is_input)
         return false;
   } else {
      if (is_input)
         return false;
   }
   /* location-based filtering */
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != ris->location && (ris->location > sem.location || ris->location + ris->array_size <= sem.location))
      return false;

   /* only scan indirect i/o when indirect_only is set */
   nir_src *src_offset = nir_get_io_offset_src(intr);
   if (!nir_src_is_const(*src_offset)) {
      if (!ris->indirect_only)
         return false;
      update_io_var_state(intr, ris);
      return false;
   }

   /* don't scan direct i/o when indirect_only is set */
   if (ris->indirect_only)
      return false;

   update_io_var_state(intr, ris);
   return false;
}

/* scan a given i/o slot for state info */
static struct rework_io_state
scan_io_var_slot(nir_shader *nir, nir_variable_mode mode, unsigned location, bool scan_indirects)
{
   struct rework_io_state ris = {
      .location = location,
      .mode = mode,
      .stage = nir->info.stage,
      .nir = nir,
   };

   struct rework_io_state test;
   do {
      update_io_var_name(&test, ris.newname ? ris.newname : ris.name);
      test = ris;
      /* always run indirect scan first to detect potential overlaps */
      if (scan_indirects) {
         ris.indirect_only = true;
         nir_shader_intrinsics_pass(nir, scan_io_var_usage, nir_metadata_all, &ris);
      }
      ris.indirect_only = false;
      nir_shader_intrinsics_pass(nir, scan_io_var_usage, nir_metadata_all, &ris);
      /* keep scanning until no changes found */
   } while (memcmp(&ris, &test, offsetof(struct rework_io_state, newname)));
   return ris;
}

/* create a variable using explicit/scan info */
static void
create_io_var(nir_shader *nir, struct rework_io_state *ris)
{
   char name[1024];
   assert(ris->component_mask);
   if (ris->newname || ris->name) {
      snprintf(name, sizeof(name), "%s", ris->newname ? ris->newname : ris->name);
   /* always use builtin name where possible */
   } else if (nir->info.stage == MESA_SHADER_VERTEX && ris->mode == nir_var_shader_in) {
      snprintf(name, sizeof(name), "%s", gl_vert_attrib_name(ris->location));
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT && ris->mode == nir_var_shader_out) {
      snprintf(name, sizeof(name), "%s", gl_frag_result_name(ris->location));
   } else if (nir_slot_is_sysval_output(ris->location, nir->info.stage)) {
      snprintf(name, sizeof(name), "%s", gl_varying_slot_name_for_stage(ris->location, nir->info.stage));
   } else {
      int c = ffs(ris->component_mask) - 1;
      if (c)
         snprintf(name, sizeof(name), "slot_%u_c%u", ris->location, c);
      else
         snprintf(name, sizeof(name), "slot_%u", ris->location);
   }
   /* calculate vec/array type */
   int frac = ffs(ris->component_mask) - 1;
   int num_components = util_last_bit(ris->component_mask) - frac;
   assert(ris->component_mask == BITFIELD_RANGE(frac, num_components));
   const struct glsl_type *vec_type = glsl_vector_type(nir_get_glsl_base_type_for_nir_type(ris->type), num_components);
   if (ris->array_size)
      vec_type = glsl_array_type(vec_type, ris->array_size, glsl_get_explicit_stride(vec_type));
   if (ris->arrayed_io) {
      /* tess size may be unknown with generated tcs */
      unsigned arrayed = nir->info.stage == MESA_SHADER_GEOMETRY ?
                         nir->info.gs.vertices_in : 32 /* MAX_PATCH_VERTICES */;
      vec_type = glsl_array_type(vec_type, arrayed, glsl_get_explicit_stride(vec_type));
   }
   nir_variable *var = nir_variable_create(nir, ris->mode, vec_type, name);
   var->data.location_frac = frac;
   var->data.location = ris->location;
   /* gallium vertex inputs use intrinsic 'base' indexing */
   if (nir->info.stage == MESA_SHADER_VERTEX && ris->mode == nir_var_shader_in)
      var->data.driver_location = ris->base;
   var->data.patch = ris->location >= VARYING_SLOT_PATCH0 ||
                     ((nir->info.stage == MESA_SHADER_TESS_CTRL || nir->info.stage == MESA_SHADER_TESS_EVAL) &&
                      (ris->location == VARYING_SLOT_TESS_LEVEL_INNER || ris->location == VARYING_SLOT_TESS_LEVEL_OUTER));
   /* set flat by default: add_derefs will fill this in later after more shader passes */
   if (nir->info.stage == MESA_SHADER_FRAGMENT && ris->mode == nir_var_shader_in)
      var->data.interpolation = INTERP_MODE_FLAT;
   var->data.fb_fetch_output = ris->fb_fetch_output;
   var->data.index = ris->dual_source_blend_index;
   var->data.precision = ris->medium_precision;
   /* only clip/cull dist and tess levels are compact */
   if (nir->info.stage != MESA_SHADER_VERTEX || ris->mode != nir_var_shader_in)
      var->data.compact = is_clipcull_dist(ris->location) || (ris->location == VARYING_SLOT_TESS_LEVEL_INNER || ris->location == VARYING_SLOT_TESS_LEVEL_OUTER);
}

/* loop the i/o mask and generate variables for specified locations */
static void
loop_io_var_mask(nir_shader *nir, nir_variable_mode mode, bool indirect, bool patch, uint64_t mask)
{
   ASSERTED bool is_vertex_input = nir->info.stage == MESA_SHADER_VERTEX && mode == nir_var_shader_in;
   u_foreach_bit64(slot, mask) {
      unsigned location = slot;
      if (patch)
         location += VARYING_SLOT_PATCH0;

      /* this should've been handled explicitly */
      assert(is_vertex_input || !is_clipcull_dist(location));

      unsigned remaining = 0;
      do {
         /* scan the slot for usage */
         struct rework_io_state ris = scan_io_var_slot(nir, mode, location, indirect);
         /* one of these must be true or things have gone very wrong */
         assert(indirect || ris.component_mask || find_rework_var(nir, &ris) || remaining);
         /* release builds only */
         if (!ris.component_mask)
            break;

         /* whatever reaches this point is either enough info to create a variable or an existing variable */
         if (!find_rework_var(nir, &ris))
            create_io_var(nir, &ris);
         /* scanning may detect multiple potential variables per location at component offsets: process again */
         remaining = ris.ignored_component_mask;
      } while (remaining);
   }
}

/* for a given mode, generate variables */
static void
rework_io_vars(nir_shader *nir, nir_variable_mode mode, struct zink_shader *zs)
{
   assert(mode == nir_var_shader_out || mode == nir_var_shader_in);
   assert(util_bitcount(mode) == 1);
   bool found = false;
   /* if no i/o, skip */
   if (mode == nir_var_shader_out)
      found = nir->info.outputs_written || nir->info.outputs_read || nir->info.patch_outputs_written || nir->info.patch_outputs_read;
   else
      found = nir->info.inputs_read || nir->info.patch_inputs_read;
   if (!found)
      return;

   /* use local copies to enable incremental processing */
   uint64_t inputs_read = nir->info.inputs_read;
   uint64_t inputs_read_indirectly = nir->info.inputs_read_indirectly;
   uint64_t outputs_accessed = nir->info.outputs_written | nir->info.outputs_read;
   uint64_t outputs_accessed_indirectly = nir->info.outputs_read_indirectly |
                                          nir->info.outputs_written_indirectly;

   /* fragment outputs are special: handle separately */
   if (mode == nir_var_shader_out && nir->info.stage == MESA_SHADER_FRAGMENT) {
      assert(!outputs_accessed_indirectly);
      u_foreach_bit64(slot, outputs_accessed) {
         struct rework_io_state ris = {
            .location = slot,
            .mode = mode,
            .stage = nir->info.stage,
         };
         /* explicitly handle builtins */
         switch (slot) {
         case FRAG_RESULT_DEPTH:
         case FRAG_RESULT_STENCIL:
         case FRAG_RESULT_SAMPLE_MASK:
            ris.bit_size = 32;
            ris.component_mask = 0x1;
            ris.type = slot == FRAG_RESULT_DEPTH ? nir_type_float32 : nir_type_uint32;
            create_io_var(nir, &ris);
            outputs_accessed &= ~BITFIELD64_BIT(slot);
            break;
         default:
            break;
         }
      }
      /* the rest of the outputs can be generated normally */
      loop_io_var_mask(nir, mode, false, false, outputs_accessed);
      return;
   }

   /* vertex inputs are special: handle separately */
   if (nir->info.stage == MESA_SHADER_VERTEX && mode == nir_var_shader_in) {
      assert(!inputs_read_indirectly);
      u_foreach_bit64(slot, inputs_read) {
         /* explicitly handle builtins */
         if (slot != VERT_ATTRIB_POS && slot != VERT_ATTRIB_POINT_SIZE)
            continue;

         uint32_t component_mask = slot == VERT_ATTRIB_POINT_SIZE ? 0x1 : 0xf;
         struct rework_io_state ris = {
            .location = slot,
            .mode = mode,
            .stage = nir->info.stage,
            .bit_size = 32,
            .component_mask = component_mask,
            .type = nir_type_float32,
            .newname = scan_io_var_slot(nir, nir_var_shader_in, slot, false).newname,
         };
         create_io_var(nir, &ris);
         inputs_read &= ~BITFIELD64_BIT(slot);
      }
      /* the rest of the inputs can be generated normally */
      loop_io_var_mask(nir, mode, false, false, inputs_read);
      return;
   }

   /* these are the masks to process based on the mode: nothing "special" as above */
   uint64_t mask = mode == nir_var_shader_in ? inputs_read : outputs_accessed;
   uint64_t indirect_mask = mode == nir_var_shader_in ? inputs_read_indirectly : outputs_accessed_indirectly;
   u_foreach_bit64(slot, mask) {
      struct rework_io_state ris = {
         .location = slot,
         .mode = mode,
         .stage = nir->info.stage,
         .arrayed_io = (mode == nir_var_shader_in ? zs->arrayed_inputs : zs->arrayed_outputs) & BITFIELD64_BIT(slot),
      };
      /* explicitly handle builtins */
      unsigned max_components = 0;
      switch (slot) {
      case VARYING_SLOT_FOGC:
         /* use intr components */
         break;
      case VARYING_SLOT_POS:
      case VARYING_SLOT_CLIP_VERTEX:
      case VARYING_SLOT_PNTC:
      case VARYING_SLOT_BOUNDING_BOX0:
      case VARYING_SLOT_BOUNDING_BOX1:
         max_components = 4;
         ris.type = nir_type_float32;
         break;
      case VARYING_SLOT_CLIP_DIST0:
         max_components = nir->info.clip_distance_array_size;
         assert(max_components);
         ris.type = nir_type_float32;
         break;
      case VARYING_SLOT_CULL_DIST0:
         max_components = nir->info.cull_distance_array_size;
         assert(max_components);
         ris.type = nir_type_float32;
         break;
      case VARYING_SLOT_CLIP_DIST1:
      case VARYING_SLOT_CULL_DIST1:
         mask &= ~BITFIELD64_BIT(slot);
         indirect_mask &= ~BITFIELD64_BIT(slot);
         continue;
      case VARYING_SLOT_TESS_LEVEL_OUTER:
         max_components = 4;
         ris.type = nir_type_float32;
         break;
      case VARYING_SLOT_TESS_LEVEL_INNER:
         max_components = 2;
         ris.type = nir_type_float32;
         break;
      case VARYING_SLOT_PRIMITIVE_ID:
      case VARYING_SLOT_LAYER:
      case VARYING_SLOT_VIEWPORT:
      case VARYING_SLOT_FACE:
      case VARYING_SLOT_VIEW_INDEX:
      case VARYING_SLOT_VIEWPORT_MASK:
         ris.type = nir_type_int32;
         max_components = 1;
         break;
      case VARYING_SLOT_PSIZ:
         max_components = 1;
         ris.type = nir_type_float32;
         break;
      default:
         break;
      }
      if (!max_components)
         continue;
      switch (slot) {
      case VARYING_SLOT_CLIP_DIST0:
      case VARYING_SLOT_CLIP_DIST1:
      case VARYING_SLOT_CULL_DIST0:
      case VARYING_SLOT_CULL_DIST1:
      case VARYING_SLOT_TESS_LEVEL_OUTER:
      case VARYING_SLOT_TESS_LEVEL_INNER:
         /* compact arrays */
         ris.component_mask = 0x1;
         ris.array_size = max_components;
         break;
      default:
         ris.component_mask = BITFIELD_MASK(max_components);
         break;
      }
      ris.bit_size = 32;
      create_io_var(nir, &ris);
      mask &= ~BITFIELD64_BIT(slot);
      /* eliminate clip/cull distance scanning early */
      indirect_mask &= ~BITFIELD64_BIT(slot);
   }

   /* patch i/o */
   if ((nir->info.stage == MESA_SHADER_TESS_CTRL && mode == nir_var_shader_out) ||
       (nir->info.stage == MESA_SHADER_TESS_EVAL && mode == nir_var_shader_in)) {
      uint64_t patch_outputs_accessed = nir->info.patch_outputs_read | nir->info.patch_outputs_written;
      uint64_t indirect_patch_mask =
         mode == nir_var_shader_in ? nir->info.patch_inputs_read_indirectly
                                   : (nir->info.patch_outputs_read_indirectly |
                                      nir->info.patch_outputs_written_indirectly);
      uint64_t patch_mask = mode == nir_var_shader_in ? nir->info.patch_inputs_read : patch_outputs_accessed;

      loop_io_var_mask(nir, mode, true, true, indirect_patch_mask);
      loop_io_var_mask(nir, mode, false, true, patch_mask);
   }

   /* regular i/o */
   loop_io_var_mask(nir, mode, true, false, indirect_mask);
   loop_io_var_mask(nir, mode, false, false, mask);
}

static int
zink_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static nir_mem_access_size_align
mem_access_size_align_cb(nir_intrinsic_op intrin, uint8_t bytes,
                         uint8_t bit_size, uint32_t align,
                         uint32_t align_offset, bool offset_is_const,
                         enum gl_access_qualifier access, const void *cb_data)
{
   align = nir_combined_align(align, align_offset);

   assert(util_is_power_of_two_nonzero(align));

   /* simply drop the bit_size for unaligned load/stores */
   if (align < (bit_size / 8)) {
      return (nir_mem_access_size_align){
         .num_components = MIN2(bytes / align, 4),
         .bit_size = align * 8,
         .align = align,
         .shift = nir_mem_access_shift_method_scalar,
      };
   } else {
      return (nir_mem_access_size_align){
         .num_components = MIN2(bytes / (bit_size / 8), 4),
         .bit_size = bit_size,
         .align = bit_size / 8,
         .shift = nir_mem_access_shift_method_scalar,
      };
   }
}

static nir_mem_access_size_align
mem_access_scratch_size_align_cb(nir_intrinsic_op intrin, uint8_t bytes,
                                 uint8_t bit_size, uint32_t align,
                                 uint32_t align_offset, bool offset_is_const,
                                 enum gl_access_qualifier access, const void *cb_data)
{
   bit_size = *(const uint8_t *)cb_data;
   align = nir_combined_align(align, align_offset);

   assert(util_is_power_of_two_nonzero(align));

   return (nir_mem_access_size_align){
      .num_components = MIN2(bytes / (bit_size / 8), 4),
      .bit_size = bit_size,
      .align = bit_size / 8,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

static bool
alias_scratch_memory_scan_bit_size(struct nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   uint8_t *bit_size = data;
   switch (instr->intrinsic) {
   case nir_intrinsic_load_scratch:
      *bit_size = MIN2(*bit_size, instr->def.bit_size);
      return false;
   case nir_intrinsic_store_scratch:
      *bit_size = MIN2(*bit_size, instr->src[0].ssa->bit_size);
      return false;
   default:
      return false;
   }
}

static bool
alias_scratch_memory(nir_shader *nir)
{
   uint8_t bit_size = 64;

   nir_shader_intrinsics_pass(nir, alias_scratch_memory_scan_bit_size, nir_metadata_all, &bit_size);
   nir_lower_mem_access_bit_sizes_options lower_scratch_mem_access_options = {
      .modes = nir_var_function_temp,
      .may_lower_unaligned_stores_to_atomics = true,
      .callback = mem_access_scratch_size_align_cb,
      .cb_data = &bit_size,
   };
   return nir_lower_mem_access_bit_sizes(nir, &lower_scratch_mem_access_options);
}

static uint8_t
lower_vec816_alu(const nir_instr *instr, const void *cb_data)
{
   return 4;
}

static unsigned
zink_lower_bit_size_cb(const nir_instr *instr, void *data)
{
   struct zink_screen *screen = data;

   switch (instr->type) {
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_bit_count:
      case nir_op_find_lsb:
      case nir_op_ifind_msb:
      case nir_op_ufind_msb:
         return alu->src[0].src.ssa->bit_size == 32 ? 0 : 32;
      case nir_op_bitfield_insert:
      case nir_op_bitfield_reverse:
         if (!screen->info.have_KHR_maintenance9)
            return alu->src[0].src.ssa->bit_size == 32 ? 0 : 32;
         return 0;
      default:
         return 0;
      }
   }
   default:
      return 0;
   }
}

static bool
fix_vertex_input_locations_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   if (!filter_io_instr(intr, &is_load, &is_input, &is_interp) || !is_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location < VERT_ATTRIB_GENERIC0)
      return false;
   sem.location = VERT_ATTRIB_GENERIC0 + nir_intrinsic_base(intr);
   nir_intrinsic_set_io_semantics(intr, sem);
   return true;
}

static bool
fix_vertex_input_locations(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_VERTEX)
      return false;

   return nir_shader_intrinsics_pass(nir, fix_vertex_input_locations_instr, nir_metadata_all, NULL);
}

struct trivial_revectorize_state {
   bool has_xfb;
   uint32_t component_mask;
   nir_intrinsic_instr *base;
   nir_intrinsic_instr *next_emit_vertex;
   nir_intrinsic_instr *merge[NIR_MAX_VEC_COMPONENTS];
   struct set *deletions;
};

/* always skip xfb; scalarized xfb is preferred */
static bool
intr_has_xfb(nir_intrinsic_instr *intr)
{
   if (!nir_intrinsic_has_io_xfb(intr))
      return false;
   for (unsigned i = 0; i < 2; i++) {
      if (nir_intrinsic_io_xfb(intr).out[i].num_components || nir_intrinsic_io_xfb2(intr).out[i].num_components) {
         return true;
      }
   }
   return false;
}

/* helper to avoid vectorizing i/o for different vertices */
static nir_intrinsic_instr *
find_next_emit_vertex(nir_intrinsic_instr *intr)
{
   bool found = false;
   nir_foreach_instr_safe(instr, intr->instr.block) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *test_intr = nir_instr_as_intrinsic(instr);
         if (!found && test_intr != intr)
            continue;
         if (!found) {
            assert(intr == test_intr);
            found = true;
            continue;
         }
         if (test_intr->intrinsic == nir_intrinsic_emit_vertex)
            return test_intr;
      }
   }
   return NULL;
}

/* scan for vectorizable instrs on a given location */
static bool
trivial_revectorize_intr_scan(nir_shader *nir, nir_intrinsic_instr *intr, struct trivial_revectorize_state *state)
{
   nir_intrinsic_instr *base = state->base;

   if (intr == base)
      return false;

   if (intr->intrinsic != base->intrinsic)
      return false;

   if (_mesa_set_search(state->deletions, intr))
      return false;

   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   filter_io_instr(intr, &is_load, &is_input, &is_interp);

   nir_io_semantics base_sem = nir_intrinsic_io_semantics(base);
   nir_io_semantics test_sem = nir_intrinsic_io_semantics(intr);
   nir_alu_type base_type = is_load ? nir_intrinsic_dest_type(base) : nir_intrinsic_src_type(base);
   nir_alu_type test_type = is_load ? nir_intrinsic_dest_type(intr) : nir_intrinsic_src_type(intr);
   int c = nir_intrinsic_component(intr);
   /* already detected */
   if (state->component_mask & BITFIELD_BIT(c))
      return false;
   /* not a match */
   if (base_sem.location != test_sem.location || base_sem.num_slots != test_sem.num_slots || base_type != test_type)
      return false;
   /* only vectorize when all srcs match */
   for (unsigned i = !is_input; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; i++) {
      if (!nir_srcs_equal(intr->src[i], base->src[i]))
         return false;
   }
   /* never match xfb */
   state->has_xfb |= intr_has_xfb(intr);
   if (state->has_xfb)
      return false;
   if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      /* only match same vertex */
      if (state->next_emit_vertex != find_next_emit_vertex(intr))
         return false;
   }
   uint32_t mask = is_load ? BITFIELD_RANGE(c, intr->num_components) : (nir_intrinsic_write_mask(intr) << c);
   state->component_mask |= mask;
   u_foreach_bit(component, mask)
      state->merge[component] = intr;

   return true;
}

static bool
trivial_revectorize_scan(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool is_load = false;
   bool is_input = false;
   bool is_interp = false;
   if (!filter_io_instr(intr, &is_load, &is_input, &is_interp))
      return false;
   if (intr->num_components != 1)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (!is_input || b->shader->info.stage != MESA_SHADER_VERTEX) {
      /* always ignore compact arrays */
      switch (sem.location) {
      case VARYING_SLOT_CLIP_DIST0:
      case VARYING_SLOT_CLIP_DIST1:
      case VARYING_SLOT_CULL_DIST0:
      case VARYING_SLOT_CULL_DIST1:
      case VARYING_SLOT_TESS_LEVEL_INNER:
      case VARYING_SLOT_TESS_LEVEL_OUTER:
         return false;
      default: break;
      }
   }
   /* always ignore to-be-deleted instrs */
   if (_mesa_set_search(data, intr))
      return false;

   /* never vectorize xfb */
   if (intr_has_xfb(intr))
      return false;

   int ic = nir_intrinsic_component(intr);
   uint32_t mask = is_load ? BITFIELD_RANGE(ic, intr->num_components) : (nir_intrinsic_write_mask(intr) << ic);
   /* already vectorized */
   if (util_bitcount(mask) == 4)
      return false;
   struct trivial_revectorize_state state = {
      .component_mask = mask,
      .base = intr,
      /* avoid clobbering i/o for different vertices */
      .next_emit_vertex = b->shader->info.stage == MESA_SHADER_GEOMETRY ? find_next_emit_vertex(intr) : NULL,
      .deletions = data,
   };
   u_foreach_bit(bit, mask)
      state.merge[bit] = intr;
   bool progress = false;
   nir_foreach_instr(instr, intr->instr.block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *test_intr = nir_instr_as_intrinsic(instr);
      /* no matching across vertex emission */
      if (test_intr->intrinsic == nir_intrinsic_emit_vertex)
         break;
      progress |= trivial_revectorize_intr_scan(b->shader, test_intr, &state);
   }
   if (!progress || state.has_xfb)
      return false;

   /* verify nothing crazy happened */
   assert(state.component_mask);
   for (unsigned i = 0; i < 4; i++) {
      assert(!state.merge[i] || !intr_has_xfb(state.merge[i]));
   }

   unsigned first_component = ffs(state.component_mask) - 1;
   unsigned num_components = util_bitcount(state.component_mask);
   unsigned num_contiguous = 0;
   uint32_t contiguous_mask = 0;
   for (unsigned i = 0; i < num_components; i++) {
      unsigned c = i + first_component;
      /* calc mask of contiguous components to vectorize */
      if (state.component_mask & BITFIELD_BIT(c)) {
         num_contiguous++;
         contiguous_mask |= BITFIELD_BIT(c);
      }
      /* on the first gap or the the last component, vectorize */
      if (!(state.component_mask & BITFIELD_BIT(c)) || i == num_components - 1) {
         if (num_contiguous > 1) {
            /* reindex to enable easy src/dest index comparison */
            nir_index_ssa_defs(nir_shader_get_entrypoint(b->shader));
            /* determine the first/last instr to use for the base (vectorized) load/store */
            unsigned first_c = ffs(contiguous_mask) - 1;
            nir_intrinsic_instr *base = NULL;
            unsigned test_idx = is_load ? UINT32_MAX : 0;
            for (unsigned j = 0; j < num_contiguous; j++) {
               unsigned merge_c = j + first_c;
               nir_intrinsic_instr *merge_intr = state.merge[merge_c];
               /* avoid breaking ssa ordering by using:
                * - first instr for vectorized load
                * - last instr for vectorized store
                * this guarantees all srcs have been seen
                */
               if ((is_load && merge_intr->def.index < test_idx) ||
                   (!is_load && merge_intr->src[0].ssa->index >= test_idx)) {
                  test_idx = is_load ? merge_intr->def.index : merge_intr->src[0].ssa->index;
                  base = merge_intr;
               }
            }
            assert(base);
            /* update instr components */
            nir_intrinsic_set_component(base, nir_intrinsic_component(state.merge[first_c]));
            unsigned orig_components = base->num_components;
            base->num_components = num_contiguous;
            /* do rewrites after loads and before stores */
            b->cursor = is_load ? nir_after_instr(&base->instr) : nir_before_instr(&base->instr);
            if (is_load) {
               base->def.num_components = num_contiguous;
               /* iterate the contiguous loaded components and rewrite merged dests */
               for (unsigned j = 0; j < num_contiguous; j++) {
                  unsigned merge_c = j + first_c;
                  nir_intrinsic_instr *merge_intr = state.merge[merge_c];
                  /* detect if the merged instr loaded multiple components and use swizzle mask for rewrite */
                  unsigned use_components = merge_intr == base ? orig_components : merge_intr->def.num_components;
                  nir_def *swiz = nir_channels(b, &base->def, BITFIELD_RANGE(j, use_components));
                  nir_def_rewrite_uses_after_instr(&merge_intr->def, swiz, merge_intr == base ? swiz->parent_instr : &merge_intr->instr);
                  j += use_components - 1;
               }
            } else {
               nir_def *comp[NIR_MAX_VEC_COMPONENTS];
               /* generate swizzled vec of store components and rewrite store src */
               for (unsigned j = 0; j < num_contiguous; j++) {
                  unsigned merge_c = j + first_c;
                  nir_intrinsic_instr *merge_intr = state.merge[merge_c];
                  /* detect if the merged instr stored multiple components and extract them for rewrite */
                  unsigned use_components = merge_intr == base ? orig_components : merge_intr->num_components;
                  for (unsigned k = 0; k < use_components; k++)
                     comp[j + k] = nir_channel(b, merge_intr->src[0].ssa, k);
                  j += use_components - 1;
               }
               nir_def *val = nir_vec(b, comp, num_contiguous);
               nir_src_rewrite(&base->src[0], val);
               nir_intrinsic_set_write_mask(base, BITFIELD_MASK(num_contiguous));
            }
            /* deleting instructions during a foreach explodes the compiler, so delete later */
            for (unsigned j = 0; j < num_contiguous; j++) {
               unsigned merge_c = j + first_c;
               nir_intrinsic_instr *merge_intr = state.merge[merge_c];
               if (merge_intr != base)
                  _mesa_set_add(data, &merge_intr->instr);
            }
         }
         contiguous_mask = 0;
         num_contiguous = 0;
      }
   }

   return true;
}

/* attempt to revectorize scalar i/o, ignoring xfb and "hard stuff" */
static bool
trivial_revectorize(nir_shader *nir)
{
   struct set deletions;

   if (nir->info.stage > MESA_SHADER_FRAGMENT)
      return false;

   _mesa_set_init(&deletions, NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);
   bool progress = nir_shader_intrinsics_pass(nir, trivial_revectorize_scan, nir_metadata_dominance, &deletions);
   /* now it's safe to delete */
   set_foreach_remove(&deletions, entry) {
      nir_instr *instr = (void*)entry->key;
      nir_instr_remove(instr);
   }
   ralloc_free(deletions.table);
   return progress;
}

static bool
flatten_image_arrays_intr(struct nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_deref)
      return false;

   nir_deref_instr *deref = nir_instr_as_deref(instr);
   if (deref->deref_type != nir_deref_type_array)
      return false;
   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   if (!parent || parent->deref_type != nir_deref_type_array)
      return false;
   nir_variable *var = nir_deref_instr_get_variable(deref);
   const struct glsl_type *type = glsl_without_array(var->type);
   if (type == var->type || (!glsl_type_is_sampler(type) && !glsl_type_is_image(type)))
      return false;

   nir_deref_instr *parent_parent = nir_deref_instr_parent(parent);
   int parent_size = glsl_array_size(parent->type);
   b->cursor = nir_after_instr(instr);
   nir_deref_instr *new_deref = nir_build_deref_array(b, parent_parent, nir_iadd(b, nir_imul_imm(b, parent->arr.index.ssa, parent_size), deref->arr.index.ssa));
   nir_def_rewrite_uses_after(&deref->def, &new_deref->def);
   _mesa_set_add(data, instr);
   _mesa_set_add(data, &parent->instr);
   return true;
}

static bool
flatten_image_arrays(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform | nir_var_image) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_sampler(type) && !glsl_type_is_image(type))
         continue;
      if (type == var->type)
         continue;
      var->type = glsl_array_type(type, glsl_get_aoa_size(var->type), sizeof(void*));
      progress = true;
   }
   struct set *deletions = _mesa_set_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);
   progress |= nir_shader_instructions_pass(nir, flatten_image_arrays_intr, nir_metadata_dominance, deletions);
   set_foreach_remove(deletions, he) {
      nir_instr *instr = (void*)he->key;
      nir_instr_remove_v(instr);
   }
   _mesa_set_destroy(deletions, NULL);
   if (progress)
      nir_fixup_deref_types(nir);
   return progress;
}

static bool
bound_image_arrays_instr(struct nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_deref)
      return false;

   nir_deref_instr *deref = nir_instr_as_deref(instr);
   if (deref->deref_type != nir_deref_type_array)
      return false;

   if (!nir_src_is_const(deref->arr.index))
      return false;
   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   int parent_size = glsl_array_size(parent->type);
   unsigned idx = nir_src_as_uint(deref->arr.index);
   if (idx >= parent_size) {
      b->cursor = nir_before_instr(instr);
      nir_src_rewrite(&deref->arr.index, nir_imm_zero(b, 1, 32));
      return true;
   }
   return false;
}

static bool
bound_image_arrays(nir_shader *nir)
{
   return nir_shader_instructions_pass(nir, bound_image_arrays_instr, nir_metadata_dominance, NULL);
}

struct zink_shader *
zink_shader_create(struct zink_screen *screen, struct nir_shader *nir)
{
   struct zink_shader *zs = rzalloc(NULL, struct zink_shader);

   zs->has_edgeflags = nir->info.stage == MESA_SHADER_VERTEX &&
                       nir->info.outputs_written & VARYING_BIT_EDGE;

   zs->sinfo.have_vulkan_memory_model = screen->info.have_KHR_vulkan_memory_model;
   zs->sinfo.have_workgroup_memory_explicit_layout = screen->info.have_KHR_workgroup_memory_explicit_layout;
   zs->sinfo.broken_arbitary_type_const = screen->driver_compiler_workarounds.broken_const;
   if (screen->info.have_KHR_shader_float_controls) {
      if (screen->info.props12.shaderDenormFlushToZeroFloat16)
         zs->sinfo.float_controls.flush_denorms |= 0x1;
      if (screen->info.props12.shaderDenormFlushToZeroFloat32)
         zs->sinfo.float_controls.flush_denorms |= 0x2;
      if (screen->info.props12.shaderDenormFlushToZeroFloat64)
         zs->sinfo.float_controls.flush_denorms |= 0x4;

      if (screen->info.props12.shaderDenormPreserveFloat16)
         zs->sinfo.float_controls.preserve_denorms |= 0x1;
      if (screen->info.props12.shaderDenormPreserveFloat32)
         zs->sinfo.float_controls.preserve_denorms |= 0x2;
      if (screen->info.props12.shaderDenormPreserveFloat64)
         zs->sinfo.float_controls.preserve_denorms |= 0x4;

      zs->sinfo.float_controls.denorms_all_independence =
         screen->info.props12.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL;

      zs->sinfo.float_controls.denorms_32_bit_independence =
         zs->sinfo.float_controls.denorms_all_independence ||
         screen->info.props12.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY;
   }
   zs->sinfo.bindless_set_idx = screen->desc_set_id[ZINK_DESCRIPTOR_BINDLESS];

   util_queue_fence_init(&zs->precompile.fence);
   util_dynarray_init(&zs->pipeline_libs, zs);
   zs->hash = _mesa_hash_pointer(zs);

   zs->programs = _mesa_pointer_set_create(NULL);
   simple_mtx_init(&zs->lock, mtx_plain);
   memcpy(&zs->info, &nir->info, sizeof(nir->info));
   zs->info.name = ralloc_strdup(zs, nir->info.name);

   zs->can_inline = true;
   zs->nir = nir;

   if (nir->info.stage != MESA_SHADER_KERNEL)
      match_tex_dests(nir, zs, true);

   return zs;
}

void
zink_shader_init(struct zink_screen *screen, struct zink_shader *zs)
{
   bool have_psiz = false;
   nir_shader *nir = zs->nir;

   if (nir->info.stage == MESA_SHADER_KERNEL) {
      nir_lower_mem_access_bit_sizes_options lower_mem_access_options = {
         .modes = nir_var_all ^ nir_var_function_temp,
         .may_lower_unaligned_stores_to_atomics = true,
         .callback = mem_access_size_align_cb,
         .cb_data = screen,
      };
      NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &lower_mem_access_options);
      NIR_PASS(_, nir, nir_lower_bit_size, zink_lower_bit_size_cb, screen);
      NIR_PASS(_, nir, alias_scratch_memory);
      NIR_PASS(_, nir, nir_lower_alu_width, lower_vec816_alu, NULL);
      NIR_PASS(_, nir, nir_lower_alu_vec8_16_srcs);
   }

   NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_in | nir_var_shader_out, NULL, NULL);
   optimize_nir(nir, NULL, true);
   NIR_PASS(_, nir, bound_image_arrays);
   NIR_PASS(_, nir, flatten_image_arrays);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out) {
      if (glsl_type_is_image(var->type) || glsl_type_is_sampler(var->type)) {
         NIR_PASS(_, nir, lower_bindless_io);
         break;
      }
   }
   if (nir->info.stage < MESA_SHADER_FRAGMENT)
      nir_gather_xfb_info_from_intrinsics(nir);
   NIR_PASS(_, nir, fix_vertex_input_locations);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   scan_nir(screen, nir, zs);
   NIR_PASS(_, nir, nir_opt_vectorize, NULL, NULL);
   NIR_PASS(_, nir, trivial_revectorize);
   if (nir->info.io_lowered) {
      rework_io_vars(nir, nir_var_shader_in, zs);
      rework_io_vars(nir, nir_var_shader_out, zs);
      nir_sort_variables_by_location(nir, nir_var_shader_in);
      nir_sort_variables_by_location(nir, nir_var_shader_out);
   }

   if (nir->info.stage < MESA_SHADER_COMPUTE)
      create_gfx_pushconst(nir);

   if (nir->info.stage == MESA_SHADER_TESS_CTRL ||
            nir->info.stage == MESA_SHADER_TESS_EVAL)
      NIR_PASS(_, nir, nir_lower_io_array_vars_to_elements_no_indirects, false);

   if (nir->info.stage < MESA_SHADER_FRAGMENT)
      have_psiz = check_psiz(nir);
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      zs->flat_flags = zink_flat_flags(nir);

   if (!gl_shader_stage_is_compute(nir->info.stage) && nir->info.separate_shader)
      NIR_PASS(_, nir, fixup_io_locations);

   NIR_PASS(_, nir, lower_basevertex);
   NIR_PASS(_, nir, lower_baseinstance);
   NIR_PASS(_, nir, split_bitfields);
   if (!screen->info.feats.features.shaderStorageImageMultisample)
      NIR_PASS(_, nir, strip_tex_ms);
   NIR_PASS(_, nir, nir_lower_frexp); /* TODO: Use the spirv instructions for this. */

   if (screen->need_2D_zs)
      NIR_PASS(_, nir, lower_1d_shadow, screen);

   {
      nir_lower_subgroups_options subgroup_options = {0};
      subgroup_options.lower_to_scalar = true;
      subgroup_options.subgroup_size = screen->info.props11.subgroupSize;
      subgroup_options.ballot_bit_size = 32;
      subgroup_options.ballot_components = 4;
      subgroup_options.lower_subgroup_masks = true;
      if (!(screen->info.subgroup.supportedStages & mesa_to_vk_shader_stage(clamp_stage(&nir->info)))) {
         subgroup_options.subgroup_size = 1;
         subgroup_options.lower_vote_trivial = true;
      }
      subgroup_options.lower_inverse_ballot = true;
      NIR_PASS(_, nir, nir_lower_subgroups, &subgroup_options);
   }

   optimize_nir(nir, NULL, true);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
   NIR_PASS(_, nir, nir_lower_discard_if, nir_lower_demote_if_to_cf | nir_lower_terminate_if_to_cf);

   bool needs_size = analyze_io(zs, nir);
   NIR_PASS(_, nir, unbreak_bos, zs, needs_size);
   /* run in compile if there could be inlined uniforms */
   if (!screen->driconf.inline_uniforms && !nir->info.num_inlinable_uniforms) {
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_mem_global | nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_mem_shared, NULL, NULL);
      NIR_PASS(_, nir, rewrite_bo_access, screen);
      NIR_PASS(_, nir, remove_bo_access, zs);
   }

   struct zink_bindless_info bindless = {0};
   bindless.bindless_set = screen->desc_set_id[ZINK_DESCRIPTOR_BINDLESS];
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out)
      var->data.is_xfb = false;

   optimize_nir(nir, NULL, true);
   prune_io(nir);

   if (nir->info.stage == MESA_SHADER_KERNEL) {
      NIR_PASS(_, nir, type_images);
   }

   unsigned ubo_binding_mask = 0;
   unsigned ssbo_binding_mask = 0;
   foreach_list_typed_reverse_safe(nir_variable, var, node, &nir->variables) {
      if (_nir_shader_variable_has_mode(var, nir_var_uniform |
                                        nir_var_image |
                                        nir_var_mem_ubo |
                                        nir_var_mem_ssbo)) {
         enum zink_descriptor_type ztype;
         const struct glsl_type *type = glsl_without_array(var->type);
         if (var->data.mode == nir_var_mem_ubo) {
            ztype = ZINK_DESCRIPTOR_TYPE_UBO;
            /* buffer 0 is a push descriptor */
            var->data.descriptor_set = !!var->data.driver_location;
            var->data.binding = !var->data.driver_location ? clamp_stage(&nir->info) :
                                zink_binding(nir->info.stage,
                                             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                             var->data.driver_location,
                                             screen->compact_descriptors);
            assert(var->data.driver_location || var->data.binding < 10);
            VkDescriptorType vktype = !var->data.driver_location ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            int binding = var->data.binding;

            if (!var->data.driver_location) {
               zs->has_uniforms = true;
            } else if (!(ubo_binding_mask & BITFIELD_BIT(binding))) {
               zs->bindings[ztype][zs->num_bindings[ztype]].index = var->data.driver_location;
               zs->bindings[ztype][zs->num_bindings[ztype]].binding = binding;
               zs->bindings[ztype][zs->num_bindings[ztype]].type = vktype;
               zs->bindings[ztype][zs->num_bindings[ztype]].size = glsl_get_length(var->type);
               assert(zs->bindings[ztype][zs->num_bindings[ztype]].size);
               zs->num_bindings[ztype]++;
               ubo_binding_mask |= BITFIELD_BIT(binding);
            }
         } else if (var->data.mode == nir_var_mem_ssbo) {
            ztype = ZINK_DESCRIPTOR_TYPE_SSBO;
            var->data.descriptor_set = screen->desc_set_id[ztype];
            var->data.binding = zink_binding(clamp_stage(&nir->info),
                                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                             var->data.driver_location,
                                             screen->compact_descriptors);
            if (!(ssbo_binding_mask & BITFIELD_BIT(var->data.binding))) {
               zs->bindings[ztype][zs->num_bindings[ztype]].index = var->data.driver_location;
               zs->bindings[ztype][zs->num_bindings[ztype]].binding = var->data.binding;
               zs->bindings[ztype][zs->num_bindings[ztype]].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
               zs->bindings[ztype][zs->num_bindings[ztype]].size = glsl_get_length(var->type);
               assert(zs->bindings[ztype][zs->num_bindings[ztype]].size);
               zs->num_bindings[ztype]++;
               ssbo_binding_mask |= BITFIELD_BIT(var->data.binding);
            }
         } else {
            assert(var->data.mode == nir_var_uniform ||
                   var->data.mode == nir_var_image);
            if (var->data.bindless) {
               zs->bindless = true;
               handle_bindless_var(nir, var, type, &bindless);
            } else if (glsl_type_is_sampler(type) || glsl_type_is_image(type)) {
               VkDescriptorType vktype = glsl_type_is_image(type) ? zink_image_type(type) : glsl_type_is_bare_sampler(type) ? VK_DESCRIPTOR_TYPE_SAMPLER : zink_sampler_type(type);
               if (nir->info.stage == MESA_SHADER_KERNEL && vktype == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                  vktype = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
               ztype = zink_desc_type_from_vktype(vktype);
               var->data.driver_location = var->data.binding;
               var->data.descriptor_set = screen->desc_set_id[ztype];
               var->data.binding = zink_binding(nir->info.stage, vktype, var->data.driver_location, screen->compact_descriptors);
               zs->bindings[ztype][zs->num_bindings[ztype]].index = var->data.driver_location;
               zs->bindings[ztype][zs->num_bindings[ztype]].binding = var->data.binding;
               zs->bindings[ztype][zs->num_bindings[ztype]].type = vktype;
               if (glsl_type_is_array(var->type))
                  zs->bindings[ztype][zs->num_bindings[ztype]].size = glsl_get_aoa_size(var->type);
               else
                  zs->bindings[ztype][zs->num_bindings[ztype]].size = 1;
               zs->num_bindings[ztype]++;
            } else if (var->data.mode == nir_var_uniform) {
               /* this is a dead uniform */
               var->data.mode = 0;
               exec_node_remove(&var->node);
            }
         }
      }
   }
   bool bindless_lowered = false;
   NIR_PASS(bindless_lowered, nir, lower_bindless, &bindless);
   zs->bindless |= bindless_lowered;

   if (!screen->info.feats.features.shaderInt64 || !screen->info.feats.features.shaderFloat64)
      NIR_PASS(_, nir, lower_64bit_vars, screen->info.feats.features.shaderInt64);
   if (nir->info.stage != MESA_SHADER_KERNEL)
      NIR_PASS(_, nir, match_tex_dests, zs, false);

   if (!nir->info.internal)
      nir_foreach_shader_out_variable(var, nir)
         var->data.explicit_xfb_buffer = 0;
   if (nir->xfb_info && nir->xfb_info->output_count && nir->info.outputs_written)
      update_so_info(zs, nir, nir->info.outputs_written, have_psiz);
   zink_shader_serialize_blob(nir, &zs->blob);
   memcpy(&zs->info, &nir->info, sizeof(nir->info));
}

void
zink_shader_finalize(struct pipe_screen *pscreen, struct nir_shader *nir)
{
   struct zink_screen *screen = zink_screen(pscreen);

   nir_lower_tex_options tex_opts = {
      .lower_invalid_implicit_lod = true,
   };
   /*
      Sampled Image must be an object whose type is OpTypeSampledImage.
      The Dim operand of the underlying OpTypeImage must be 1D, 2D, 3D,
      or Rect, and the Arrayed and MS operands must be 0.
      - SPIRV, OpImageSampleProj* opcodes
    */
   tex_opts.lower_txp = BITFIELD_BIT(GLSL_SAMPLER_DIM_CUBE) |
                        BITFIELD_BIT(GLSL_SAMPLER_DIM_MS);
   tex_opts.lower_txp_array = true;
   if (!screen->info.feats.features.shaderImageGatherExtended)
      tex_opts.lower_tg4_offsets = true;
   NIR_PASS(_, nir, nir_lower_tex, &tex_opts);
   optimize_nir(nir, NULL, false);
   if (nir->info.stage == MESA_SHADER_VERTEX)
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   if (screen->driconf.inline_uniforms)
      nir_find_inlinable_uniforms(nir);
}

void
zink_shader_free(struct zink_screen *screen, struct zink_shader *shader)
{
   _mesa_set_destroy(shader->programs, NULL);
   util_queue_fence_wait(&shader->precompile.fence);
   util_queue_fence_destroy(&shader->precompile.fence);
   zink_descriptor_shader_deinit(screen, shader);
   if (screen->info.have_EXT_shader_object) {
      VKSCR(DestroyShaderEXT)(screen->dev, shader->precompile.obj.obj, NULL);
   } else {
      if (shader->precompile.obj.mod)
         VKSCR(DestroyShaderModule)(screen->dev, shader->precompile.obj.mod, NULL);
      if (shader->precompile.gpl)
         VKSCR(DestroyPipeline)(screen->dev, shader->precompile.gpl, NULL);
   }
   blob_finish(&shader->blob);
   ralloc_free(shader->spirv);
   free(shader->precompile.bindings);
   ralloc_free(shader);
}

static bool
gfx_shader_prune(struct zink_screen *screen, struct zink_shader *shader)
{
   /* this shader may still be precompiling, so access here must be locked and singular */
   simple_mtx_lock(&shader->lock);
   struct set_entry *entry = _mesa_set_next_entry(shader->programs, NULL);
   struct zink_gfx_program *prog = (void*)(entry ? entry->key : NULL);
   if (entry)
      _mesa_set_remove(shader->programs, entry);
   simple_mtx_unlock(&shader->lock);
   if (!prog)
      return false;
   gl_shader_stage stage = shader->info.stage;
   assert(stage < ZINK_GFX_SHADER_COUNT);
   util_queue_fence_wait(&prog->base.cache_fence);
   unsigned stages_present = prog->stages_present;
   unsigned stages_remaining = prog->stages_remaining;
   if (prog->shaders[MESA_SHADER_TESS_CTRL] &&
         prog->shaders[MESA_SHADER_TESS_CTRL]->non_fs.is_generated) {
      stages_present &= ~BITFIELD_BIT(MESA_SHADER_TESS_CTRL);
      stages_remaining &= ~BITFIELD_BIT(MESA_SHADER_TESS_CTRL);
   }
   unsigned idx = zink_program_cache_stages(stages_present);
   if (!prog->base.removed && stages_present == stages_remaining &&
         (stage == MESA_SHADER_FRAGMENT || !shader->non_fs.is_generated)) {
      struct hash_table *ht = &prog->base.ctx->program_cache[idx];
      simple_mtx_lock(&prog->base.ctx->program_lock[idx]);
      struct hash_entry *he = _mesa_hash_table_search(ht, prog->shaders);
      assert(he && he->data == prog);
      _mesa_hash_table_remove(ht, he);
      prog->base.removed = true;
      simple_mtx_unlock(&prog->base.ctx->program_lock[idx]);

      for (int i = 0; i < ARRAY_SIZE(prog->pipelines); ++i) {
         hash_table_foreach(&prog->pipelines[i], table_entry) {
            struct zink_gfx_pipeline_cache_entry *pc_entry = table_entry->data;

            util_queue_fence_wait(&pc_entry->fence);
         }
      }
   }
   if (stage == MESA_SHADER_FRAGMENT || !shader->non_fs.is_generated) {
      prog->shaders[stage] = NULL;
      prog->stages_remaining &= ~BITFIELD_BIT(stage);
   }
   /* only remove generated tcs during parent tes destruction */
   if (stage == MESA_SHADER_TESS_EVAL && shader->non_fs.generated_tcs)
      prog->shaders[MESA_SHADER_TESS_CTRL] = NULL;
   if (stage != MESA_SHADER_FRAGMENT &&
      prog->shaders[MESA_SHADER_GEOMETRY] &&
      prog->shaders[MESA_SHADER_GEOMETRY]->non_fs.parent ==
      shader) {
      prog->shaders[MESA_SHADER_GEOMETRY] = NULL;
   }
   zink_gfx_program_reference(screen, &prog, NULL);
   return true;
}

void
zink_gfx_shader_free(struct zink_screen *screen, struct zink_shader *shader)
{
   assert(shader->info.stage != MESA_SHADER_COMPUTE);
   util_queue_fence_wait(&shader->precompile.fence);

   /* if the shader is still precompiling, the program set must be pruned under lock */
   while (gfx_shader_prune(screen, shader));

   while (util_dynarray_contains(&shader->pipeline_libs, struct zink_gfx_lib_cache*)) {
      struct zink_gfx_lib_cache *libs = util_dynarray_pop(&shader->pipeline_libs, struct zink_gfx_lib_cache*);
      if (!libs->removed) {
         libs->removed = true;
         unsigned idx = zink_program_cache_stages(libs->stages_present);
         simple_mtx_lock(&screen->pipeline_libs_lock[idx]);
         _mesa_set_remove_key(&screen->pipeline_libs[idx], libs);
         simple_mtx_unlock(&screen->pipeline_libs_lock[idx]);
      }
      zink_gfx_lib_cache_unref(screen, libs);
   }
   if (shader->info.stage == MESA_SHADER_TESS_EVAL &&
       shader->non_fs.generated_tcs) {
      /* automatically destroy generated tcs shaders when tes is destroyed */
      zink_gfx_shader_free(screen, shader->non_fs.generated_tcs);
      shader->non_fs.generated_tcs = NULL;
   }
   if (shader->info.stage != MESA_SHADER_FRAGMENT) {
      for (unsigned int i = 0; i < ARRAY_SIZE(shader->non_fs.generated_gs); i++) {
         for (int j = 0; j < ARRAY_SIZE(shader->non_fs.generated_gs[0]); j++) {
            if (shader->non_fs.generated_gs[i][j]) {
               /* automatically destroy generated gs shaders when owner is destroyed */
               zink_gfx_shader_free(screen, shader->non_fs.generated_gs[i][j]);
               shader->non_fs.generated_gs[i][j] = NULL;
            }
         }
      }
   }
   zink_shader_free(screen, shader);
}


struct zink_shader_object
zink_shader_tcs_compile(struct zink_screen *screen, struct zink_shader *zs, unsigned patch_vertices, bool can_shobj, struct zink_program *pg)
{
   assert(zs->info.stage == MESA_SHADER_TESS_CTRL);
   /* shortcut all the nir passes since we just have to change this one word */
   zs->spirv->words[zs->spirv->tcs_vertices_out_word] = patch_vertices;
   return zink_shader_spirv_compile(screen, zs, NULL, can_shobj, pg);
}

/* creating a passthrough tcs shader that's roughly:

#version 150
#extension GL_ARB_tessellation_shader : require

in vec4 some_var[gl_MaxPatchVertices];
out vec4 some_var_out;

layout(push_constant) uniform tcsPushConstants {
    layout(offset = 0) float TessLevelInner[2];
    layout(offset = 8) float TessLevelOuter[4];
} u_tcsPushConstants;
layout(vertices = $vertices_per_patch) out;
void main()
{
  gl_TessLevelInner = u_tcsPushConstants.TessLevelInner;
  gl_TessLevelOuter = u_tcsPushConstants.TessLevelOuter;
  some_var_out = some_var[gl_InvocationID];
}

*/
void
zink_shader_tcs_init(struct zink_screen *screen, struct zink_shader *zs, nir_shader *tes, nir_shader **nir_ret)
{
   nir_shader *nir = zs->nir;

   nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));

   nir_def *invocation_id = nir_load_invocation_id(&b);

   nir_foreach_shader_in_variable(var, tes) {
      if (var->data.location == VARYING_SLOT_TESS_LEVEL_INNER || var->data.location == VARYING_SLOT_TESS_LEVEL_OUTER)
         continue;
      const struct glsl_type *in_type = var->type;
      const struct glsl_type *out_type = var->type;
      char buf[1024];
      snprintf(buf, sizeof(buf), "%s_out", var->name);
      if (!nir_is_arrayed_io(var, MESA_SHADER_TESS_EVAL)) {
         const struct glsl_type *type = var->type;
         in_type = glsl_array_type(type, 32 /* MAX_PATCH_VERTICES */, 0);
         out_type = glsl_array_type(type, nir->info.tess.tcs_vertices_out, 0);
      }

      nir_variable *in = nir_variable_create(nir, nir_var_shader_in, in_type, var->name);
      nir_variable *out = nir_variable_create(nir, nir_var_shader_out, out_type, buf);
      out->data.location = in->data.location = var->data.location;
      out->data.location_frac = in->data.location_frac = var->data.location_frac;

      /* gl_in[] receives values from equivalent built-in output
         variables written by the vertex shader (section 2.14.7).  Each array
         element of gl_in[] is a structure holding values for a specific vertex of
         the input patch.  The length of gl_in[] is equal to the
         implementation-dependent maximum patch size (gl_MaxPatchVertices).
         - ARB_tessellation_shader
       */
      /* we need to load the invocation-specific value of the vertex output and then store it to the per-patch output */
      nir_deref_instr *in_value = nir_build_deref_array(&b, nir_build_deref_var(&b, in), invocation_id);
      nir_deref_instr *out_value = nir_build_deref_array(&b, nir_build_deref_var(&b, out), invocation_id);
      copy_vars(&b, out_value, in_value);
   }
   nir_variable *gl_TessLevelInner = nir_variable_create(nir, nir_var_shader_out, glsl_array_type(glsl_float_type(), 2, 0), "gl_TessLevelInner");
   gl_TessLevelInner->data.location = VARYING_SLOT_TESS_LEVEL_INNER;
   gl_TessLevelInner->data.patch = 1;
   nir_variable *gl_TessLevelOuter = nir_variable_create(nir, nir_var_shader_out, glsl_array_type(glsl_float_type(), 4, 0), "gl_TessLevelOuter");
   gl_TessLevelOuter->data.location = VARYING_SLOT_TESS_LEVEL_OUTER;
   gl_TessLevelOuter->data.patch = 1;

   create_gfx_pushconst(nir);

   nir_def *load_inner = nir_load_push_constant_zink(&b, 2, 32,
                                                         nir_imm_int(&b, ZINK_GFX_PUSHCONST_DEFAULT_INNER_LEVEL));
   nir_def *load_outer = nir_load_push_constant_zink(&b, 4, 32,
                                                         nir_imm_int(&b, ZINK_GFX_PUSHCONST_DEFAULT_OUTER_LEVEL));

   for (unsigned i = 0; i < 2; i++) {
      nir_deref_instr *store_idx = nir_build_deref_array_imm(&b, nir_build_deref_var(&b, gl_TessLevelInner), i);
      nir_store_deref(&b, store_idx, nir_channel(&b, load_inner, i), 0xff);
   }
   for (unsigned i = 0; i < 4; i++) {
      nir_deref_instr *store_idx = nir_build_deref_array_imm(&b, nir_build_deref_var(&b, gl_TessLevelOuter), i);
      nir_store_deref(&b, store_idx, nir_channel(&b, load_outer, i), 0xff);
   }

   nir_validate_shader(nir, "created");

   optimize_nir(nir, NULL, true);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
   NIR_PASS(_, nir, nir_convert_from_ssa, true, false);

   *nir_ret = nir;
   zink_shader_serialize_blob(nir, &zs->blob);
}

struct zink_shader *
zink_shader_tcs_create(struct zink_screen *screen, unsigned vertices_per_patch)
{
   struct zink_shader *zs = rzalloc(NULL, struct zink_shader);
   util_queue_fence_init(&zs->precompile.fence);
   zs->hash = _mesa_hash_pointer(zs);
   zs->programs = _mesa_pointer_set_create(NULL);
   simple_mtx_init(&zs->lock, mtx_plain);

   nir_shader *nir = nir_shader_create(NULL, MESA_SHADER_TESS_CTRL, &screen->nir_options, NULL);
   nir_function *fn = nir_function_create(nir, "main");
   fn->is_entrypoint = true;
   nir_function_impl_create(fn);
   zs->nir = nir;

   nir->info.tess.tcs_vertices_out = vertices_per_patch;
   memcpy(&zs->info, &nir->info, sizeof(nir->info));
   zs->non_fs.is_generated = true;
   return zs;
}

bool
zink_shader_has_cubes(nir_shader *nir)
{
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (glsl_type_is_sampler(type) && glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_CUBE)
         return true;
   }
   return false;
}

nir_shader *
zink_shader_blob_deserialize(struct zink_screen *screen, struct blob *blob)
{
   struct blob_reader blob_reader;
   blob_reader_init(&blob_reader, blob->data, blob->size);
   return nir_deserialize(NULL, &screen->nir_options, &blob_reader);
}

nir_shader *
zink_shader_deserialize(struct zink_screen *screen, struct zink_shader *zs)
{
   return zink_shader_blob_deserialize(screen, &zs->blob);
}

void
zink_shader_serialize_blob(nir_shader *nir, struct blob *blob)
{
   blob_init(blob);
#ifndef NDEBUG
   bool strip = !(zink_debug & (ZINK_DEBUG_NIR | ZINK_DEBUG_SPIRV | ZINK_DEBUG_TGSI));
#else
   bool strip = false;
#endif
   nir_serialize(blob, nir, strip);
}

void
zink_print_shader(struct zink_screen *screen, struct zink_shader *zs, FILE *fp)
{
   nir_shader *nir = zink_shader_deserialize(screen, zs);
   nir_print_shader(nir, fp);
   ralloc_free(nir);
}
