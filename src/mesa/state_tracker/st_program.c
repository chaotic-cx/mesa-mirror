/**************************************************************************
 *
 * Copyright 2007 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/
 /*
  * Authors:
  *   Keith Whitwell <keithw@vmware.com>
  *   Brian Paul
  */


#include "nir_builder.h"
#include "main/errors.h"

#include "main/hash.h"
#include "main/mtypes.h"
#include "nir/nir_xfb_info.h"
#include "nir/pipe_nir.h"
#include "program/prog_parameter.h"
#include "program/prog_print.h"
#include "program/prog_to_nir.h"

#include "compiler/glsl/gl_nir.h"
#include "compiler/glsl/gl_nir_linker.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_serialize.h"
#include "draw/draw_context.h"

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_shader_tokens.h"
#include "draw/draw_context.h"

#include "util/u_dump.h"
#include "util/u_memory.h"

#include "st_debug.h"
#include "st_cb_bitmap.h"
#include "st_cb_drawpixels.h"
#include "st_context.h"
#include "st_program.h"
#include "st_atifs_to_nir.h"
#include "st_nir.h"
#include "st_shader_cache.h"
#include "st_util.h"
#include "cso_cache/cso_context.h"


static void
destroy_program_variants(struct st_context *st, struct gl_program *target);

static void
set_affected_state_flags(uint64_t *states,
                         struct gl_program *prog,
                         uint64_t new_constants,
                         uint64_t new_sampler_views,
                         uint64_t new_samplers,
                         uint64_t new_images,
                         uint64_t new_ubos,
                         uint64_t new_ssbos,
                         uint64_t new_atomics)
{
   if (prog->Parameters->NumParameters)
      *states |= new_constants;

   if (prog->info.num_textures)
      *states |= new_sampler_views | new_samplers;

   if (prog->info.num_images)
      *states |= new_images;

   if (prog->info.num_ubos)
      *states |= new_ubos;

   if (prog->info.num_ssbos)
      *states |= new_ssbos;

   if (prog->info.num_abos)
      *states |= new_atomics;
}

/**
 * This determines which states will be updated when the shader is bound.
 */
void
st_set_prog_affected_state_flags(struct gl_program *prog)
{
   uint64_t *states;

   switch (prog->info.stage) {
   case MESA_SHADER_VERTEX:
      states = &prog->affected_states;

      *states = ST_NEW_VS_STATE |
                ST_NEW_RASTERIZER |
                ST_NEW_VERTEX_ARRAYS;

      set_affected_state_flags(states, prog,
                               ST_NEW_VS_CONSTANTS,
                               ST_NEW_VS_SAMPLER_VIEWS,
                               ST_NEW_VS_SAMPLERS,
                               ST_NEW_VS_IMAGES,
                               ST_NEW_VS_UBOS,
                               ST_NEW_VS_SSBOS,
                               ST_NEW_VS_ATOMICS);
      break;

   case MESA_SHADER_TESS_CTRL:
      states = &prog->affected_states;

      *states = ST_NEW_TCS_STATE;

      set_affected_state_flags(states, prog,
                               ST_NEW_TCS_CONSTANTS,
                               ST_NEW_TCS_SAMPLER_VIEWS,
                               ST_NEW_TCS_SAMPLERS,
                               ST_NEW_TCS_IMAGES,
                               ST_NEW_TCS_UBOS,
                               ST_NEW_TCS_SSBOS,
                               ST_NEW_TCS_ATOMICS);
      break;

   case MESA_SHADER_TESS_EVAL:
      states = &prog->affected_states;

      *states = ST_NEW_TES_STATE |
                ST_NEW_RASTERIZER;

      set_affected_state_flags(states, prog,
                               ST_NEW_TES_CONSTANTS,
                               ST_NEW_TES_SAMPLER_VIEWS,
                               ST_NEW_TES_SAMPLERS,
                               ST_NEW_TES_IMAGES,
                               ST_NEW_TES_UBOS,
                               ST_NEW_TES_SSBOS,
                               ST_NEW_TES_ATOMICS);
      break;

   case MESA_SHADER_GEOMETRY:
      states = &prog->affected_states;

      *states = ST_NEW_GS_STATE |
                ST_NEW_RASTERIZER;

      set_affected_state_flags(states, prog,
                               ST_NEW_GS_CONSTANTS,
                               ST_NEW_GS_SAMPLER_VIEWS,
                               ST_NEW_GS_SAMPLERS,
                               ST_NEW_GS_IMAGES,
                               ST_NEW_GS_UBOS,
                               ST_NEW_GS_SSBOS,
                               ST_NEW_GS_ATOMICS);
      break;

   case MESA_SHADER_FRAGMENT:
      states = &prog->affected_states;

      /* gl_FragCoord and glDrawPixels always use constants. */
      *states = ST_NEW_FS_STATE |
                ST_NEW_SAMPLE_SHADING |
                ST_NEW_FS_CONSTANTS;

      set_affected_state_flags(states, prog,
                               ST_NEW_FS_CONSTANTS,
                               ST_NEW_FS_SAMPLER_VIEWS,
                               ST_NEW_FS_SAMPLERS,
                               ST_NEW_FS_IMAGES,
                               ST_NEW_FS_UBOS,
                               ST_NEW_FS_SSBOS,
                               ST_NEW_FS_ATOMICS);
      break;

   case MESA_SHADER_COMPUTE:
      states = &prog->affected_states;

      *states = ST_NEW_CS_STATE;

      set_affected_state_flags(states, prog,
                               ST_NEW_CS_CONSTANTS,
                               ST_NEW_CS_SAMPLER_VIEWS,
                               ST_NEW_CS_SAMPLERS,
                               ST_NEW_CS_IMAGES,
                               ST_NEW_CS_UBOS,
                               ST_NEW_CS_SSBOS,
                               ST_NEW_CS_ATOMICS);
      break;

   default:
      UNREACHABLE("unhandled shader stage");
   }
}


/**
 * Delete a shader variant.  Note the caller must unlink the variant from
 * the linked list.
 */
static void
delete_variant(struct st_context *st, struct st_variant *v, unsigned stage)
{
   if (v->driver_shader) {
      if (stage == MESA_SHADER_VERTEX &&
          ((struct st_common_variant*)v)->key.is_draw_shader) {
         /* Draw shader. */
         draw_delete_vertex_shader(st->draw, v->driver_shader);
      } else if (st->has_shareable_shaders || v->st == st) {
         /* The shader's context matches the calling context, or we
          * don't care.
          */
         switch (stage) {
         case MESA_SHADER_VERTEX:
            st->pipe->delete_vs_state(st->pipe, v->driver_shader);
            break;
         case MESA_SHADER_TESS_CTRL:
            st->pipe->delete_tcs_state(st->pipe, v->driver_shader);
            break;
         case MESA_SHADER_TESS_EVAL:
            st->pipe->delete_tes_state(st->pipe, v->driver_shader);
            break;
         case MESA_SHADER_GEOMETRY:
            st->pipe->delete_gs_state(st->pipe, v->driver_shader);
            break;
         case MESA_SHADER_FRAGMENT:
            st->pipe->delete_fs_state(st->pipe, v->driver_shader);
            break;
         case MESA_SHADER_COMPUTE:
            st->pipe->delete_compute_state(st->pipe, v->driver_shader);
            break;
         default:
            UNREACHABLE("bad shader type in delete_basic_variant");
         }
      } else {
         /* We can't delete a shader with a context different from the one
          * that created it.  Add it to the creating context's zombie list.
          */
         st_save_zombie_shader(v->st, stage, v->driver_shader);
      }
   }

   FREE(v);
}

static void
st_unbind_program(struct st_context *st, struct gl_program *p)
{
   struct gl_context *ctx = st->ctx;

   /* Unbind the shader in cso_context and re-bind in st/mesa. */
   switch (p->info.stage) {
   case MESA_SHADER_VERTEX:
      cso_set_vertex_shader_handle(st->cso_context, NULL);
      ctx->NewDriverState |= ST_NEW_VS_STATE;
      break;
   case MESA_SHADER_TESS_CTRL:
      cso_set_tessctrl_shader_handle(st->cso_context, NULL);
      ctx->NewDriverState |= ST_NEW_TCS_STATE;
      break;
   case MESA_SHADER_TESS_EVAL:
      cso_set_tesseval_shader_handle(st->cso_context, NULL);
      ctx->NewDriverState |= ST_NEW_TES_STATE;
      break;
   case MESA_SHADER_GEOMETRY:
      cso_set_geometry_shader_handle(st->cso_context, NULL);
      ctx->NewDriverState |= ST_NEW_GS_STATE;
      break;
   case MESA_SHADER_FRAGMENT:
      cso_set_fragment_shader_handle(st->cso_context, NULL);
      ctx->NewDriverState |= ST_NEW_FS_STATE;
      break;
   case MESA_SHADER_COMPUTE:
      cso_set_compute_shader_handle(st->cso_context, NULL);
      ctx->NewDriverState |= ST_NEW_CS_STATE;
      break;
   default:
      UNREACHABLE("invalid shader type");
   }
}

/**
 * Free all basic program variants.
 */
void
st_release_variants(struct st_context *st, struct gl_program *p)
{
   struct st_variant *v;

   /* If we are releasing shaders, re-bind them, because we don't
    * know which shaders are bound in the driver.
    */
   if (p->variants)
      st_unbind_program(st, p);

   for (v = p->variants; v; ) {
      struct st_variant *next = v->next;
      delete_variant(st, v, p->info.stage);
      v = next;
   }

   p->variants = NULL;

   /* Note: Any setup of ->ir.nir that has had pipe->create_*_state called on
    * it has resulted in the driver taking ownership of the NIR.  Those
    * callers should be NULLing out the nir field in any pipe_shader_state
    * that might have this called in order to indicate that.
    *
    * GLSL IR and ARB programs will have set gl_program->nir to the same
    * shader as ir->ir.nir, so it will be freed by _mesa_delete_program().
    */
}

/**
 * Free all basic program variants and unref program.
 */
void
st_release_program(struct st_context *st, struct gl_program **p)
{
   if (!*p)
      return;

   destroy_program_variants(st, *p);
   _mesa_reference_program(st->ctx, p, NULL);
}

static void
st_prog_to_nir_postprocess(struct st_context *st, nir_shader *nir,
                           struct gl_program *prog)
{
   struct pipe_screen *screen = st->screen;

   NIR_PASS(_, nir, nir_lower_reg_intrinsics_to_ssa);
   nir_validate_shader(nir, "after st/ptn lower_reg_intrinsics_to_ssa");

   NIR_PASS(_, nir, st_nir_lower_wpos_ytransform, prog, screen);
   NIR_PASS(_, nir, nir_lower_system_values);

   struct nir_lower_compute_system_values_options cs_options = {
      .has_base_global_invocation_id = false,
      .has_base_workgroup_id = false,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &cs_options);

   /* Optimise NIR */
   NIR_PASS(_, nir, nir_opt_constant_folding);
   gl_nir_opts(nir);

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   /* This must be done after optimizations to assign IO bases. */
   nir_recompute_io_bases(nir, nir_var_shader_in | nir_var_shader_out);

   if (st->allow_st_finalize_nir_twice) {
      st_serialize_base_nir(prog, nir);
      st_finalize_nir(st, prog, NULL, nir, true, false);

      if (screen->finalize_nir)
         screen->finalize_nir(screen, nir);
   }

   nir_validate_shader(nir, "after st/glsl finalize_nir");
}

/**
 * Prepare st_vertex_program info.
 *
 * attrib_to_index is an optional mapping from a vertex attrib to a shader
 * input index.
 */
void
st_prepare_vertex_program(struct gl_program *prog)
{
   struct gl_vertex_program *stvp = (struct gl_vertex_program *)prog;

   stvp->num_inputs = util_bitcount64(prog->info.inputs_read);
   stvp->vert_attrib_mask = prog->info.inputs_read;

   /* Compute mapping of vertex program outputs to slots. */
   memset(stvp->result_to_output, ~0, sizeof(stvp->result_to_output));
   unsigned num_outputs = 0;
   for (unsigned attr = 0; attr < VARYING_SLOT_MAX; attr++) {
      if (prog->info.outputs_written & BITFIELD64_BIT(attr))
         stvp->result_to_output[attr] = num_outputs++;
   }
   /* pre-setup potentially unused edgeflag output */
   stvp->result_to_output[VARYING_SLOT_EDGE] = num_outputs;
}

void
st_translate_stream_output_info(struct gl_program *prog)
{
   struct gl_transform_feedback_info *info = prog->sh.LinkedTransformFeedback;
   if (!info)
      return;

   /* Determine the (default) output register mapping for each output. */
   unsigned num_outputs = 0;
   uint8_t output_mapping[VARYING_SLOT_TESS_MAX];
   memset(output_mapping, 0, sizeof(output_mapping));

   for (unsigned attr = 0; attr < VARYING_SLOT_MAX; attr++) {
      /* this output was added by mesa/st and should not be tracked for xfb:
       * drivers must check var->data.explicit_location to find the original output
       * and only emit that one for xfb
       */
      if (prog->skip_pointsize_xfb && attr == VARYING_SLOT_PSIZ)
         continue;
      if (prog->info.outputs_written & BITFIELD64_BIT(attr))
         output_mapping[attr] = num_outputs++;
   }

   /* Translate stream output info. */
   struct pipe_stream_output_info *so_info =
      &prog->state.stream_output;

   if (!num_outputs) {
      so_info->num_outputs = 0;
      return;
   }

   for (unsigned i = 0; i < info->NumOutputs; i++) {
      so_info->output[i].register_index =
         output_mapping[info->Outputs[i].OutputRegister];
      so_info->output[i].start_component = info->Outputs[i].ComponentOffset;
      so_info->output[i].num_components = info->Outputs[i].NumComponents;
      so_info->output[i].output_buffer = info->Outputs[i].OutputBuffer;
      so_info->output[i].dst_offset = info->Outputs[i].DstOffset;
      so_info->output[i].stream = info->Outputs[i].StreamId;
   }

   for (unsigned i = 0; i < PIPE_MAX_SO_BUFFERS; i++) {
      so_info->stride[i] = info->Buffers[i].Stride;
   }
   so_info->num_outputs = info->NumOutputs;
}

/**
 * Creates a driver shader from a NIR shader.  Takes ownership of the
 * passed nir_shader.
 */
void *
st_create_nir_shader(struct st_context *st, struct pipe_shader_state *state)
{
   struct pipe_context *pipe = st->pipe;

   assert(state->type == PIPE_SHADER_IR_NIR);
   nir_shader *nir = state->ir.nir;
   gl_shader_stage stage = nir->info.stage;

   /* Renumber SSA defs to make it easier to run diff on printed NIR. */
   nir_foreach_function_impl(impl, nir) {
      nir_index_ssa_defs(impl);
   }

   if (ST_DEBUG & DEBUG_PRINT_IR) {
      fprintf(stderr, "NIR before handing off to driver:\n");
      nir_print_shader(nir, stderr);
   }

   if (ST_DEBUG & DEBUG_PRINT_XFB) {
      if (nir->info.io_lowered) {
         if (nir->xfb_info && nir->xfb_info->output_count) {
            fprintf(stderr, "XFB info before handing off to driver:\n");
            fprintf(stderr, "stride = {%u, %u, %u, %u}\n",
                    nir->info.xfb_stride[0], nir->info.xfb_stride[1],
                    nir->info.xfb_stride[2], nir->info.xfb_stride[3]);
            nir_print_xfb_info(nir->xfb_info, stderr);
         }
      } else {
         struct pipe_stream_output_info *so = &state->stream_output;

         if (so->num_outputs) {
            fprintf(stderr, "XFB info before handing off to driver:\n");
            fprintf(stderr, "stride = {%u, %u, %u, %u}\n",
                    so->stride[0], so->stride[1], so->stride[2],
                    so->stride[3]);

            for (unsigned i = 0; i < so->num_outputs; i++) {
               fprintf(stderr, "output%u: buffer=%u offset=%u, location=%u, "
                               "component_offset=%u, component_mask=0x%x, "
                               "stream=%u\n",
                       i, so->output[i].output_buffer,
                       so->output[i].dst_offset * 4,
                       so->output[i].register_index,
                       so->output[i].start_component,
                       BITFIELD_RANGE(so->output[i].start_component,
                                      so->output[i].num_components),
                       so->output[i].stream);
            }
         }
      }
   }

   void *shader;
   switch (stage) {
   case MESA_SHADER_VERTEX:
      shader = pipe->create_vs_state(pipe, state);
      break;
   case MESA_SHADER_TESS_CTRL:
      shader = pipe->create_tcs_state(pipe, state);
      break;
   case MESA_SHADER_TESS_EVAL:
      shader = pipe->create_tes_state(pipe, state);
      break;
   case MESA_SHADER_GEOMETRY:
      shader = pipe->create_gs_state(pipe, state);
      break;
   case MESA_SHADER_FRAGMENT:
      shader = pipe->create_fs_state(pipe, state);
      break;
   case MESA_SHADER_COMPUTE: {
      /* We'd like to use this for all stages but we need to rework streamout in
       * gallium first.
       */
      shader = pipe_shader_from_nir(pipe, nir);
      break;
   }
   default:
      UNREACHABLE("unsupported shader stage");
      return NULL;
   }

   return shader;
}

/**
 * Translate a vertex program.
 */
static bool
st_translate_vertex_program(struct st_context *st,
                            struct gl_program *prog)
{
   /* This determines which states will be updated when the assembly
      * shader is bound.
      */
   prog->affected_states = ST_NEW_VS_STATE |
                           ST_NEW_RASTERIZER |
                           ST_NEW_VERTEX_ARRAYS;

   if (prog->Parameters->NumParameters)
      prog->affected_states |= ST_NEW_VS_CONSTANTS;

   if (prog->arb.Instructions && prog->nir)
      ralloc_free(prog->nir);

   if (prog->serialized_nir) {
      free(prog->serialized_nir);
      prog->serialized_nir = NULL;
   }
   free(prog->base_serialized_nir);

   prog->state.type = PIPE_SHADER_IR_NIR;
   if (prog->arb.Instructions)
      prog->nir = prog_to_nir(st->ctx, prog);
   st_prog_to_nir_postprocess(st, prog->nir, prog);
   prog->info = prog->nir->info;

   st_prepare_vertex_program(prog);
   return true;
}

static const struct nir_shader_compiler_options draw_nir_options = {
   .lower_scmp = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_fsat = true,
   .lower_bitfield_insert = true,
   .lower_bitfield_extract = true,
   .lower_fdph = true,
   .lower_ffma16 = true,
   .lower_ffma32 = true,
   .lower_ffma64 = true,
   .lower_flrp16 = true,
   .lower_fmod = true,
   .lower_hadd = true,
   .lower_uadd_sat = true,
   .lower_usub_sat = true,
   .lower_iadd_sat = true,
   .lower_ldexp = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_split = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_half_2x16 = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_mul_2x32_64 = true,
   .lower_ifind_msb = true,
   .lower_int64_options = nir_lower_imul_2x32_64,
   .lower_doubles_options = nir_lower_dround_even,
   .max_unroll_iterations = 32,
   .lower_to_scalar = true,
   .lower_uniforms_to_ubo = true,
   .lower_vector_cmp = true,
   .lower_device_index_to_zero = true,
   .support_16bit_alu = true,
   .lower_fisnormal = true,
   .lower_fquantize2f16 = true,
   .driver_functions = true,
};

static struct nir_shader *
get_nir_shader(struct st_context *st, struct gl_program *prog, bool is_draw)
{
   if ((!is_draw || !st->ctx->Const.PackedDriverUniformStorage) && prog->nir) {
      nir_shader *nir = prog->nir;

      /* The first shader variant takes ownership of NIR, so that there is
       * no cloning. Additional shader variants are always generated from
       * serialized NIR to save memory.
       */
      prog->nir = NULL;
      assert(prog->serialized_nir && prog->serialized_nir_size);
      return nir;
   }

   struct blob_reader blob_reader;
   const struct nir_shader_compiler_options *options =
      is_draw ? &draw_nir_options : st->screen->nir_options[prog->info.stage];

   if (is_draw && st->ctx->Const.PackedDriverUniformStorage &&
       (!prog->shader_program || prog->shader_program->data->LinkStatus != LINKING_SKIPPED)) {
      assert(prog->base_serialized_nir);
      blob_reader_init(&blob_reader, prog->base_serialized_nir, prog->base_serialized_nir_size);
   } else {
      assert(prog->serialized_nir);
      blob_reader_init(&blob_reader, prog->serialized_nir, prog->serialized_nir_size);
   }
   return nir_deserialize(NULL, options, &blob_reader);
}

static void
lower_ucp(struct st_context *st,
          struct nir_shader *nir,
          unsigned ucp_enables,
          struct gl_program_parameter_list *params)
{
   if (nir->info.outputs_written & VARYING_BIT_CLIP_DIST0)
      NIR_PASS(_, nir, nir_lower_clip_disable, ucp_enables);
   else {
      bool can_compact = nir->options->compact_arrays;
      bool use_eye = st->ctx->_Shader->CurrentProgram[MESA_SHADER_VERTEX] != NULL;

      gl_state_index16 clipplane_state[MAX_CLIP_PLANES][STATE_LENGTH] = {{0}};
      for (int i = 0; i < MAX_CLIP_PLANES; ++i) {
         if (use_eye) {
            clipplane_state[i][0] = STATE_CLIPPLANE;
            clipplane_state[i][1] = i;
         } else {
            clipplane_state[i][0] = STATE_CLIP_INTERNAL;
            clipplane_state[i][1] = i;
         }
         _mesa_add_state_reference(params, clipplane_state[i]);
      }

      if (nir->info.stage == MESA_SHADER_VERTEX ||
          nir->info.stage == MESA_SHADER_TESS_EVAL) {
         NIR_PASS(_, nir, nir_lower_clip_vs, ucp_enables,
                    true, can_compact, clipplane_state);
      } else if (nir->info.stage == MESA_SHADER_GEOMETRY) {
         NIR_PASS(_, nir, nir_lower_clip_gs, ucp_enables,
                    can_compact, clipplane_state);
      }
   }
}

static bool
force_persample_shading(struct nir_builder *b, nir_intrinsic_instr *intr,
                        void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_barycentric_pixel ||
       intr->intrinsic == nir_intrinsic_load_barycentric_centroid) {
      intr->intrinsic = nir_intrinsic_load_barycentric_sample;
      return true;
   }

   return false;
}

static int
xfb_compare_dst_offset(const void *a, const void *b)
{
   const struct pipe_stream_output *var0 = (const struct pipe_stream_output*)a;
   const struct pipe_stream_output *var1 = (const struct pipe_stream_output*)b;

   if (var0->output_buffer != var1->output_buffer)
      return var0->output_buffer > var1->output_buffer ? 1 : -1;

   return var0->dst_offset - var1->dst_offset;
}

static void
get_stream_output_info_from_nir(nir_shader *nir,
                                struct pipe_stream_output_info *info)
{
   /* Get pipe_stream_output_info from NIR. Only used by IO variables. */
   nir_xfb_info *xfb = nir->xfb_info;
   memset(info, 0, sizeof(*info));

   if (!xfb)
      return;

   info->num_outputs = xfb->output_count;

   for (unsigned i = 0; i < 4; i++)
      info->stride[i] = nir->info.xfb_stride[i];

   for (unsigned i = 0; i < xfb->output_count; i++) {
      struct pipe_stream_output *out = &info->output[i];

      assert(!xfb->outputs[i].high_16bits);

      out->register_index =
         util_bitcount64(nir->info.outputs_written &
                         BITFIELD64_MASK(xfb->outputs[i].location));
      out->start_component = xfb->outputs[i].component_offset;
      out->num_components = util_bitcount(xfb->outputs[i].component_mask);
      out->output_buffer = xfb->outputs[i].buffer;
      out->dst_offset = xfb->outputs[i].offset / 4;
      out->stream = xfb->buffer_to_stream[out->output_buffer];
   }

   /* Intel requires that xfb outputs are sorted by dst_offset. */
   qsort(info->output, info->num_outputs, sizeof(info->output[0]),
         xfb_compare_dst_offset);
}

static struct st_common_variant *
st_create_common_variant(struct st_context *st,
                         struct gl_program *prog,
                         const struct st_common_variant_key *key,
                         bool report_compile_error, char **error)
{
   MESA_TRACE_FUNC();

   struct st_common_variant *v = CALLOC_STRUCT(st_common_variant);
   struct pipe_shader_state state = {0};

   static const gl_state_index16 point_size_state[STATE_LENGTH] =
      { STATE_POINT_SIZE_CLAMPED, 0 };
   struct gl_program_parameter_list *params = prog->Parameters;

   v->key = *key;

   state.stream_output = prog->state.stream_output;

   bool finalize = false;

   state.type = PIPE_SHADER_IR_NIR;
   state.report_compile_error = report_compile_error;
   state.ir.nir = get_nir_shader(st, prog, key->is_draw_shader);
   const nir_shader_compiler_options *options = state.ir.nir->options;

   if (key->clamp_color) {
      NIR_PASS(_, state.ir.nir, nir_lower_clamp_color_outputs);
      finalize = true;
   }
   if (key->passthrough_edgeflags) {
      NIR_PASS(_, state.ir.nir, nir_lower_passthrough_edgeflags);
      finalize = true;
   }

   if (key->export_point_size) {
      /* if flag is set, shader must export psiz */
      _mesa_add_state_reference(params, point_size_state);
      NIR_PASS(_, state.ir.nir, nir_lower_point_size_mov,
                  point_size_state);

      finalize = true;
   }

   if (key->lower_ucp) {
      assert(!options->unify_interfaces);
      lower_ucp(st, state.ir.nir, key->lower_ucp, params);
      finalize = true;
   }

   if (st->emulate_gl_clamp &&
         (key->gl_clamp[0] || key->gl_clamp[1] || key->gl_clamp[2])) {
      nir_lower_tex_options tex_opts = {0};
      tex_opts.saturate_s = key->gl_clamp[0];
      tex_opts.saturate_t = key->gl_clamp[1];
      tex_opts.saturate_r = key->gl_clamp[2];
      NIR_PASS(_, state.ir.nir, nir_lower_tex, &tex_opts);
   }

   if (finalize || !st->allow_st_finalize_nir_twice || key->is_draw_shader) {
      st_finalize_nir(st, prog, prog->shader_program, state.ir.nir, false,
                      key->is_draw_shader);
   }

   assert(state.ir.nir->info.stage == MESA_SHADER_COMPUTE ||
          state.ir.nir->info.io_lowered);

   /* This should be after all passes that touch IO. */
   if (state.ir.nir->info.io_lowered &&
       (!(state.ir.nir->options->io_options & nir_io_has_intrinsics) ||
        key->is_draw_shader)) {
      assert(!state.stream_output.num_outputs || state.ir.nir->xfb_info);
      get_stream_output_info_from_nir(state.ir.nir, &state.stream_output);
      /* Some lowering passes can leave dead code behind, but dead IO intrinsics
       * are still counted as enabled IO, which breaks things.
       */
      NIR_PASS(_, state.ir.nir, nir_opt_dce);
      NIR_PASS(_, state.ir.nir, st_nir_unlower_io_to_vars);

      if (state.ir.nir->info.stage == MESA_SHADER_TESS_CTRL &&
          state.ir.nir->options->compact_arrays &&
          state.ir.nir->options->vectorize_tess_levels)
         NIR_PASS(_, state.ir.nir, nir_lower_tess_level_array_vars_to_vec);

      gl_nir_opts(state.ir.nir);
      finalize = true;
   }

   if (finalize || !st->allow_st_finalize_nir_twice || key->is_draw_shader) {
      struct pipe_screen *screen = st->screen;
      if (!key->is_draw_shader && screen->finalize_nir)
         screen->finalize_nir(screen, state.ir.nir);

      /* Clip lowering and edgeflags may have introduced new varyings, so
       * update the inputs_read/outputs_written. However, with
       * unify_interfaces set (aka iris) the non-SSO varyings layout is
       * decided at link time with outputs_written updated so the two line
       * up.  A driver with this flag set may not use any of the lowering
       * passes that would change the varyings, so skip to make sure we don't
       * break its linkage.
       */
      if (!options->unify_interfaces) {
         nir_shader_gather_info(state.ir.nir,
                                 nir_shader_get_entrypoint(state.ir.nir));
      }
   }

   if (key->is_draw_shader) {
      NIR_PASS(_, state.ir.nir, gl_nir_lower_images, false);
      v->base.driver_shader = draw_create_vertex_shader(st->draw, &state);
   }
   else
      v->base.driver_shader = st_create_nir_shader(st, &state);

   if (report_compile_error && state.error_message) {
      *error = state.error_message;
      FREE(v);
      return NULL;
   }

   if (error)
      *error = NULL;
   return v;
}

static void
st_add_variant(struct st_variant **list, struct st_variant *v)
{
   struct st_variant *first = *list;

   /* Make sure that the default variant stays the first in the list, and insert
    * any later variants in as the second entry.
    */
   if (first) {
      v->next = first->next;
      first->next = v;
   } else {
      *list = v;
   }
}

/**
 * Find/create a vertex program variant.
 */
struct st_common_variant *
st_get_common_variant(struct st_context *st,
                      struct gl_program *prog,
                      const struct st_common_variant_key *key,
                      bool report_compile_error, char **error)
{
   struct st_common_variant *v;

   /* Search for existing variant */
   for (v = st_common_variant(prog->variants); v;
        v = st_common_variant(v->base.next)) {
      if (memcmp(&v->key, key, sizeof(*key)) == 0) {
         break;
      }
   }

   if (!v) {
      if (prog->variants != NULL) {
         _mesa_perf_debug(st->ctx, MESA_DEBUG_SEVERITY_MEDIUM,
                          "Compiling %s shader variant (%s%s%s%s%s%s)",
                          _mesa_shader_stage_to_string(prog->info.stage),
                          key->passthrough_edgeflags ? "edgeflags," : "",
                          key->clamp_color ? "clamp_color," : "",
                          key->export_point_size ? "point_size," : "",
                          key->lower_ucp ? "ucp," : "",
                          key->is_draw_shader ? "draw," : "",
                          key->gl_clamp[0] || key->gl_clamp[1] || key->gl_clamp[2] ? "GL_CLAMP," : "");
      }

      /* create now */
      v = st_create_common_variant(st, prog, key, report_compile_error, error);
      if (v) {
         v->base.st = key->st;

         if (prog->info.stage == MESA_SHADER_VERTEX) {
            struct gl_vertex_program *vp = (struct gl_vertex_program *)prog;

            v->vert_attrib_mask =
               vp->vert_attrib_mask |
               (key->passthrough_edgeflags ? VERT_BIT_EDGEFLAG : 0);
         }

         st_add_variant(&prog->variants, &v->base);
      }
   }

   return v;
}


/**
 * Translate a non-GLSL Mesa fragment shader into a NIR shader.
 */
static bool
st_translate_fragment_program(struct st_context *st,
                              struct gl_program *prog)
{
   /* This determines which states will be updated when the assembly
    * shader is bound.
    *
    * fragment.position and glDrawPixels always use constants.
    */
   prog->affected_states = ST_NEW_FS_STATE |
                           ST_NEW_SAMPLE_SHADING |
                           ST_NEW_FS_CONSTANTS;

   if (prog->ati_fs) {
      /* Just set them for ATI_fs unconditionally. */
      prog->affected_states |= ST_NEW_FS_SAMPLER_VIEWS |
                               ST_NEW_FS_SAMPLERS;
   } else {
      /* ARB_fp */
      if (prog->SamplersUsed)
         prog->affected_states |= ST_NEW_FS_SAMPLER_VIEWS |
                                  ST_NEW_FS_SAMPLERS;
   }

   /* Translate to NIR. */
   if (prog->nir && prog->arb.Instructions)
      ralloc_free(prog->nir);

   if (prog->serialized_nir) {
      free(prog->serialized_nir);
      prog->serialized_nir = NULL;
   }

   prog->state.type = PIPE_SHADER_IR_NIR;
   if (prog->arb.Instructions) {
      prog->nir = prog_to_nir(st->ctx, prog);
   } else if (prog->ati_fs) {
      const struct nir_shader_compiler_options *options =
         st->screen->nir_options[MESA_SHADER_FRAGMENT];

      assert(!prog->nir);
      prog->nir = st_translate_atifs_program(prog->ati_fs, prog, options);
   }
   st_prog_to_nir_postprocess(st, prog->nir, prog);

   prog->info = prog->nir->info;
   if (prog->ati_fs) {
      /* ATI_fs will lower fixed function fog at variant time, after the FF vertex
       * prog has been generated.  So we have to always declare a read of FOGC so
       * that FF vp feeds it to us just in case.
       */
      prog->info.inputs_read |= VARYING_BIT_FOGC;
   }

   return true;
}

static struct st_fp_variant *
st_create_fp_variant(struct st_context *st,
                     struct gl_program *fp,
                     const struct st_fp_variant_key *key,
                     bool report_compile_error, char **error)
{
   struct st_fp_variant *variant = CALLOC_STRUCT(st_fp_variant);
   struct pipe_shader_state state = {0};
   struct gl_program_parameter_list *params = fp->Parameters;
   static const gl_state_index16 texcoord_state[STATE_LENGTH] =
      { STATE_CURRENT_ATTRIB, VERT_ATTRIB_TEX0 };
   static const gl_state_index16 scale_state[STATE_LENGTH] =
      { STATE_PT_SCALE };
   static const gl_state_index16 bias_state[STATE_LENGTH] =
      { STATE_PT_BIAS };
   static const gl_state_index16 alpha_ref_state[STATE_LENGTH] =
      { STATE_ALPHA_REF };

   if (!variant)
      return NULL;

   MESA_TRACE_FUNC();

   /* Translate ATI_fs to NIR at variant time because that's when we have the
    * texture types.
    */
   state.ir.nir = get_nir_shader(st, fp, false);
   state.type = PIPE_SHADER_IR_NIR;
   state.report_compile_error = report_compile_error;

   bool finalize = false;

   if (fp->ati_fs) {
      if (key->fog) {
         NIR_PASS(_, state.ir.nir, st_nir_lower_fog, key->fog, fp->Parameters);
      }

      NIR_PASS(_, state.ir.nir, st_nir_lower_atifs_samplers, key->texture_index);

      finalize = true;
   }

   if (key->clamp_color) {
      NIR_PASS(_, state.ir.nir, nir_lower_clamp_color_outputs);
      finalize = true;
   }

   if (key->lower_flatshade) {
      NIR_PASS(_, state.ir.nir, nir_lower_flatshade);
      finalize = true;
   }

   if (key->lower_alpha_func != COMPARE_FUNC_ALWAYS) {
      _mesa_add_state_reference(params, alpha_ref_state);
      NIR_PASS(_, state.ir.nir, nir_lower_alpha_test, key->lower_alpha_func,
                  false, alpha_ref_state);
      finalize = true;
   }

   if (key->lower_two_sided_color) {
      bool face_sysval = st->ctx->Const.GLSLFrontFacingIsSysVal;
      NIR_PASS(_, state.ir.nir, nir_lower_two_sided_color, face_sysval);
      finalize = true;
   }

   if (key->persample_shading) {
      nir_shader *shader = state.ir.nir;
      nir_shader_intrinsics_pass(shader, force_persample_shading,
                                 nir_metadata_all, NULL);

      /* In addition to requiring per-sample interpolation, sample shading
       * changes the behaviour of gl_SampleMaskIn, so we need per-sample shading
       * even if there are no shader-in variables at all. In that case,
       * uses_sample_shading won't be set by glsl_to_nir. We need to do so here.
       */
      shader->info.fs.uses_sample_shading = true;

      finalize = true;
   }

   if (st->emulate_gl_clamp &&
         (key->gl_clamp[0] || key->gl_clamp[1] || key->gl_clamp[2])) {
      nir_lower_tex_options tex_opts = {0};
      tex_opts.saturate_s = key->gl_clamp[0];
      tex_opts.saturate_t = key->gl_clamp[1];
      tex_opts.saturate_r = key->gl_clamp[2];
      NIR_PASS(_, state.ir.nir, nir_lower_tex, &tex_opts);
      finalize = true;
   }

   assert(!(key->bitmap && key->drawpixels));

   /* glBitmap */
   if (key->bitmap) {
      nir_lower_bitmap_options options = {0};

      variant->bitmap_sampler = ffs(~fp->SamplersUsed) - 1;
      options.sampler = variant->bitmap_sampler;
      options.swizzle_xxxx = st->bitmap.tex_format == PIPE_FORMAT_R8_UNORM;

      NIR_PASS(_, state.ir.nir, nir_lower_bitmap, &options);
      finalize = true;
   }

   /* glDrawPixels (color only) */
   if (key->drawpixels) {
      nir_lower_drawpixels_options options = {{0}};
      unsigned samplers_used = fp->SamplersUsed;

      /* Find the first unused slot. */
      variant->drawpix_sampler = ffs(~samplers_used) - 1;
      options.drawpix_sampler = variant->drawpix_sampler;
      samplers_used |= (1 << variant->drawpix_sampler);

      options.pixel_maps = key->pixelMaps;
      if (key->pixelMaps) {
         variant->pixelmap_sampler = ffs(~samplers_used) - 1;
         options.pixelmap_sampler = variant->pixelmap_sampler;
      }

      options.scale_and_bias = key->scaleAndBias;
      if (key->scaleAndBias) {
         _mesa_add_state_reference(params, scale_state);
         memcpy(options.scale_state_tokens, scale_state,
                  sizeof(options.scale_state_tokens));
         _mesa_add_state_reference(params, bias_state);
         memcpy(options.bias_state_tokens, bias_state,
                  sizeof(options.bias_state_tokens));
      }

      _mesa_add_state_reference(params, texcoord_state);
      memcpy(options.texcoord_state_tokens, texcoord_state,
               sizeof(options.texcoord_state_tokens));

      NIR_PASS(_, state.ir.nir, nir_lower_drawpixels, &options);
      finalize = true;
   }

   bool need_lower_tex_src_plane = false;

   if (unlikely(key->external.lower_nv12 || key->external.lower_nv21 ||
                  key->external.lower_iyuv ||
                  key->external.lower_xy_uxvx || key->external.lower_yx_xuxv ||
                  key->external.lower_yx_xvxu || key->external.lower_xy_vxux ||
                  key->external.lower_ayuv || key->external.lower_xyuv ||
                  key->external.lower_yuv || key->external.lower_yu_yv ||
                  key->external.lower_yv_yu || key->external.lower_y41x)) {

      st_nir_lower_samplers(st->screen, state.ir.nir,
                              fp->shader_program, fp);

      nir_lower_tex_options options = {0};
      options.lower_y_uv_external = key->external.lower_nv12;
      options.lower_y_vu_external = key->external.lower_nv21;
      options.lower_y_u_v_external = key->external.lower_iyuv;
      options.lower_xy_uxvx_external = key->external.lower_xy_uxvx;
      options.lower_xy_vxux_external = key->external.lower_xy_vxux;
      options.lower_yx_xuxv_external = key->external.lower_yx_xuxv;
      options.lower_yx_xvxu_external = key->external.lower_yx_xvxu;
      options.lower_ayuv_external = key->external.lower_ayuv;
      options.lower_xyuv_external = key->external.lower_xyuv;
      options.lower_yuv_external = key->external.lower_yuv;
      options.lower_yu_yv_external = key->external.lower_yu_yv;
      options.lower_yv_yu_external = key->external.lower_yv_yu;
      options.lower_y41x_external = key->external.lower_y41x;
      options.lower_sx10_external = key->external.lower_sx10;
      options.lower_sx12_external = key->external.lower_sx12;
      options.bt709_external = key->external.bt709;
      options.bt2020_external = key->external.bt2020;
      options.yuv_full_range_external = key->external.yuv_full_range;
      NIR_PASS(_, state.ir.nir, nir_lower_tex, &options);
      finalize = true;
      need_lower_tex_src_plane = true;
   }

   if (finalize || !st->allow_st_finalize_nir_twice)
      st_finalize_nir(st, fp, fp->shader_program, state.ir.nir, false, false);

   /* This pass needs to happen *after* nir_lower_sampler */
   if (unlikely(need_lower_tex_src_plane)) {
      NIR_PASS(_, state.ir.nir, st_nir_lower_tex_src_plane,
                  ~fp->SamplersUsed,
                  key->external.lower_nv12 | key->external.lower_nv21 |
                     key->external.lower_xy_uxvx | key->external.lower_xy_vxux |
                     key->external.lower_yx_xuxv | key->external.lower_yx_xvxu,
                  key->external.lower_iyuv);
      finalize = true;
   }

   /* It is undefined behavior when an ARB assembly uses SHADOW2D target
    * with a texture in not depth format. In this case NVIDIA automatically
    * replaces SHADOW sampler with a normal sampler and some games like
    * Penumbra Overture which abuses this UB (issues/8425) works fine but
    * breaks with mesa. Replace the shadow sampler with a normal one here
    */
   if (!fp->shader_program && ~key->depth_textures & fp->ShadowSamplers) {
      NIR_PASS(_, state.ir.nir, nir_remove_tex_shadow,
                 ~key->depth_textures & fp->ShadowSamplers);
      finalize = true;
   }

   assert(state.ir.nir->info.io_lowered);

   /* This should be after all passes that touch IO. */
   if (!(state.ir.nir->options->io_options & nir_io_has_intrinsics)) {
      /* Some lowering passes can leave dead code behind, but dead IO intrinsics
       * are still counted as enabled IO, which breaks things.
       */
      NIR_PASS(_, state.ir.nir, nir_opt_dce);
      NIR_PASS(_, state.ir.nir, st_nir_unlower_io_to_vars);
      gl_nir_opts(state.ir.nir);
      finalize = true;
   }

   if (finalize || !st->allow_st_finalize_nir_twice) {
      /* Some of the lowering above may have introduced new varyings */
      nir_shader_gather_info(state.ir.nir,
                              nir_shader_get_entrypoint(state.ir.nir));

      struct pipe_screen *screen = st->screen;
      if (screen->finalize_nir)
         screen->finalize_nir(screen, state.ir.nir);
   }

   variant->base.driver_shader = st_create_nir_shader(st, &state);
   if (report_compile_error && state.error_message) {
      *error = state.error_message;
      FREE(variant);
      return NULL;
   }

   variant->key = *key;
   if (error)
      *error = NULL;
   return variant;
}

/**
 * Translate fragment program if needed.
 */
struct st_fp_variant *
st_get_fp_variant(struct st_context *st,
                  struct gl_program *fp,
                  const struct st_fp_variant_key *key,
                  bool report_compile_error, char **error)
{
   struct st_fp_variant *fpv;

   /* Search for existing variant */
   for (fpv = st_fp_variant(fp->variants); fpv;
        fpv = st_fp_variant(fpv->base.next)) {
      if (memcmp(&fpv->key, key, sizeof(*key)) == 0) {
         break;
      }
   }

   if (!fpv) {
      /* create new */

      if (fp->variants != NULL) {
         _mesa_perf_debug(st->ctx, MESA_DEBUG_SEVERITY_MEDIUM,
                          "Compiling fragment shader variant (%s%s%s%s%s%s%s%s%s%s%s%s%s%d)",
                          key->bitmap ? "bitmap," : "",
                          key->drawpixels ? "drawpixels," : "",
                          key->scaleAndBias ? "scale_bias," : "",
                          key->pixelMaps ? "pixel_maps," : "",
                          key->clamp_color ? "clamp_color," : "",
                          key->persample_shading ? "persample_shading," : "",
                          key->fog ? "fog," : "",
                          key->lower_two_sided_color ? "twoside," : "",
                          key->lower_flatshade ? "flatshade," : "",
                          key->lower_alpha_func != COMPARE_FUNC_ALWAYS ? "alpha_compare," : "",
                          /* skipped ATI_fs targets */
                          fp->ExternalSamplersUsed ? "external?," : "",
                          key->gl_clamp[0] || key->gl_clamp[1] || key->gl_clamp[2] ? "GL_CLAMP," : "",
                          "depth_textures=", key->depth_textures);
      }

      fpv = st_create_fp_variant(st, fp, key, report_compile_error, error);
      if (fpv) {
         fpv->base.st = key->st;

         st_add_variant(&fp->variants, &fpv->base);
      }
   }

   return fpv;
}

/**
 * Vert/Geom/Frag programs have per-context variants.  Free all the
 * variants attached to the given program which match the given context.
 */
static void
destroy_program_variants(struct st_context *st, struct gl_program *p)
{
   if (!p || p == &_mesa_DummyProgram)
      return;

   struct st_variant *v, **prevPtr = &p->variants;
   bool unbound = false;

   for (v = p->variants; v; ) {
      struct st_variant *next = v->next;
      if (v->st == st) {
         if (!unbound) {
            st_unbind_program(st, p);
            unbound = true;
         }

         /* unlink from list */
         *prevPtr = next;
         /* destroy this variant */
         delete_variant(st, v, p->info.stage);
      }
      else {
         prevPtr = &v->next;
      }
      v = next;
   }
}


/**
 * Callback for _mesa_HashWalk.  Free all the shader's program variants
 * which match the given context.
 */
static void
destroy_shader_program_variants_cb(void *data, void *userData)
{
   struct st_context *st = (struct st_context *) userData;
   struct gl_shader *shader = (struct gl_shader *) data;

   switch (shader->Type) {
   case GL_SHADER_PROGRAM_MESA:
      {
         struct gl_shader_program *shProg = (struct gl_shader_program *) data;
         GLuint i;

         for (i = 0; i < ARRAY_SIZE(shProg->_LinkedShaders); i++) {
            if (shProg->_LinkedShaders[i])
               destroy_program_variants(st, shProg->_LinkedShaders[i]->Program);
         }
      }
      break;
   case GL_VERTEX_SHADER:
   case GL_FRAGMENT_SHADER:
   case GL_GEOMETRY_SHADER:
   case GL_TESS_CONTROL_SHADER:
   case GL_TESS_EVALUATION_SHADER:
   case GL_COMPUTE_SHADER:
      break;
   default:
      assert(0);
   }
}


/**
 * Callback for _mesa_HashWalk.  Free all the program variants which match
 * the given context.
 */
static void
destroy_program_variants_cb(void *data, void *userData)
{
   struct st_context *st = (struct st_context *) userData;
   struct gl_program *program = (struct gl_program *) data;
   destroy_program_variants(st, program);
}


/**
 * Walk over all shaders and programs to delete any variants which
 * belong to the given context.
 * This is called during context tear-down.
 */
void
st_destroy_program_variants(struct st_context *st)
{
   /* If shaders can be shared with other contexts, the last context will
    * call DeleteProgram on all shaders, releasing everything.
    */
   if (st->has_shareable_shaders)
      return;

   /* ARB vert/frag program */
   _mesa_HashWalk(&st->ctx->Shared->Programs,
                  destroy_program_variants_cb, st);

   /* GLSL vert/frag/geom shaders */
   _mesa_HashWalk(&st->ctx->Shared->ShaderObjects,
                  destroy_shader_program_variants_cb, st);
}

/**
 * Compile one shader variant.
 */
static char *
st_precompile_shader_variant(struct st_context *st,
                             struct gl_program *prog,
                             bool report_compile_error)
{
   char *error = NULL;

   switch (prog->info.stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_COMPUTE: {
      struct st_common_variant_key key;

      memset(&key, 0, sizeof(key));

      if (_mesa_is_desktop_gl_compat(st->ctx) &&
          st->clamp_vert_color_in_shader &&
          (prog->info.outputs_written & (VARYING_SLOT_COL0 |
                                         VARYING_SLOT_COL1 |
                                         VARYING_SLOT_BFC0 |
                                         VARYING_SLOT_BFC1))) {
         key.clamp_color = true;
      }

      key.st = st->has_shareable_shaders ? NULL : st;
      st_get_common_variant(st, prog, &key, report_compile_error, &error);
      return error;
   }

   case MESA_SHADER_FRAGMENT: {
      struct st_fp_variant_key key;

      memset(&key, 0, sizeof(key));

      key.st = st->has_shareable_shaders ? NULL : st;
      key.lower_alpha_func = COMPARE_FUNC_ALWAYS;
      if (prog->ati_fs) {
         for (int i = 0; i < ARRAY_SIZE(key.texture_index); i++)
            key.texture_index[i] = TEXTURE_2D_INDEX;
      }

      /* Shadow samplers require texture in depth format, which we lower to
       * non-shadow if necessary for ARB programs
       */
      if (!prog->shader_program)
         key.depth_textures = prog->ShadowSamplers;

      st_get_fp_variant(st, prog, &key, report_compile_error, &error);
      return error;
   }

   default:
      UNREACHABLE("invalid shader stage");
   }
}

void
st_serialize_nir(struct gl_program *prog)
{
   if (!prog->serialized_nir) {
      struct blob blob;
      size_t size;

      blob_init(&blob);
      nir_serialize(&blob, prog->nir, false);
      blob_finish_get_buffer(&blob, &prog->serialized_nir, &size);
      prog->serialized_nir_size = size;
   }
}

void
st_serialize_base_nir(struct gl_program *prog, nir_shader *nir)
{
   if (!prog->base_serialized_nir && nir->info.stage == MESA_SHADER_VERTEX) {
      struct blob blob;
      size_t size;

      blob_init(&blob);
      nir_serialize(&blob, nir, false);
      blob_finish_get_buffer(&blob, &prog->base_serialized_nir, &size);
      prog->base_serialized_nir_size = size;
   }
}

char *
st_finalize_program(struct st_context *st, struct gl_program *prog,
                    bool report_compile_error)
{
   struct gl_context *ctx = st->ctx;
   bool is_bound = false;

   MESA_TRACE_FUNC();

   if (prog->info.stage == MESA_SHADER_VERTEX)
      is_bound = prog == ctx->VertexProgram._Current;
   else if (prog->info.stage == MESA_SHADER_TESS_CTRL)
      is_bound = prog == ctx->TessCtrlProgram._Current;
   else if (prog->info.stage == MESA_SHADER_TESS_EVAL)
      is_bound = prog == ctx->TessEvalProgram._Current;
   else if (prog->info.stage == MESA_SHADER_GEOMETRY)
      is_bound = prog == ctx->GeometryProgram._Current;
   else if (prog->info.stage == MESA_SHADER_FRAGMENT)
      is_bound = prog == ctx->FragmentProgram._Current;
   else if (prog->info.stage == MESA_SHADER_COMPUTE)
      is_bound = prog == ctx->ComputeProgram._Current;

   if (is_bound) {
      if (prog->info.stage == MESA_SHADER_VERTEX) {
         ctx->Array.NewVertexElements = true;
         ctx->NewDriverState |= ST_NEW_VERTEX_PROGRAM(ctx, prog);
      } else {
         ctx->NewDriverState |= prog->affected_states;
      }
   }

   if (prog->nir) {
      nir_sweep(prog->nir);

      /* This is only needed for ARB_vp/fp programs and when the disk cache
       * is disabled. If the disk cache is enabled, GLSL programs are
       * serialized in write_nir_to_cache.
       */
      st_serialize_base_nir(prog, prog->nir);
      st_serialize_nir(prog);
   }

   /* Always create the default variant of the program. */
   return st_precompile_shader_variant(st, prog, report_compile_error);
}

/**
 * Called when the program's text/code is changed.  We have to free
 * all shader variants and corresponding gallium shaders when this happens.
 */
GLboolean
st_program_string_notify( struct gl_context *ctx,
                          GLenum target,
                          struct gl_program *prog )
{
   struct st_context *st = st_context(ctx);

   /* GLSL-to-NIR should not end up here. */
   assert(!prog->shader_program);

   st_release_variants(st, prog);

   if (target == GL_FRAGMENT_PROGRAM_ARB ||
       target == GL_FRAGMENT_SHADER_ATI) {
      if (!st_translate_fragment_program(st, prog))
         return false;
   } else if (target == GL_VERTEX_PROGRAM_ARB) {
      if (!st_translate_vertex_program(st, prog))
         return false;
      if (st->add_point_size &&
          gl_nir_can_add_pointsize_to_program(&st->ctx->Const, prog)) {
         prog->skip_pointsize_xfb = true;
         NIR_PASS(_, prog->nir, gl_nir_add_point_size);
      }
   }

   st_finalize_program(st, prog, false);
   return GL_TRUE;
}
