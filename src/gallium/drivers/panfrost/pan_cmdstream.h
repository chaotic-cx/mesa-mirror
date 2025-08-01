/*
 * Copyright (C) 2018 Alyssa Rosenzweig
 * Copyright (C) 2020 Collabora Ltd.
 * Copyright © 2017 Intel Corporation
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

#ifndef __PAN_CMDSTREAM_H__
#define __PAN_CMDSTREAM_H__

#ifndef PAN_ARCH
#error "PAN_ARCH undefined!"
#endif

#include "genxml/gen_macros.h"

#include "pan_context.h"
#include "pan_job.h"

#include "pipe/p_defines.h"
#include "pipe/p_state.h"

#include "util/u_prim.h"

#define PAN_GPU_SUPPORTS_DISPATCH_INDIRECT (PAN_ARCH >= 6)
#define PAN_GPU_SUPPORTS_DRAW_INDIRECT     (PAN_ARCH >= 10)

struct panfrost_rasterizer {
   struct pipe_rasterizer_state base;

#if PAN_ARCH <= 7
   /* Partially packed RSD words */
   struct mali_multisample_misc_packed multisample;
   struct mali_stencil_mask_misc_packed stencil_misc;
#endif
};

struct panfrost_zsa_state {
   struct pipe_depth_stencil_alpha_state base;

   /* Is any depth, stencil, or alpha testing enabled? */
   bool enabled;

   /* Does the depth and stencil tests always pass? This ignores write
    * masks, we are only interested in whether pixels may be killed.
    */
   bool zs_always_passes;

   /* Are depth or stencil writes possible? */
   bool writes_zs;

#if PAN_ARCH <= 7
   /* Prepacked words from the RSD */
   struct mali_multisample_misc_packed rsd_depth;
   struct mali_stencil_mask_misc_packed rsd_stencil;
   struct mali_stencil_packed stencil_front, stencil_back;
#else
   /* Depth/stencil descriptor template */
   struct mali_depth_stencil_packed desc;
#endif
};

struct panfrost_vertex_state {
   unsigned num_elements;
   struct pipe_vertex_element pipe[PIPE_MAX_ATTRIBS];
   uint16_t strides[PIPE_MAX_ATTRIBS];

#if PAN_ARCH >= 9
   /* Packed attribute descriptors */
   struct mali_attribute_packed attributes[PIPE_MAX_ATTRIBS];
#else
   /* buffers corresponds to attribute buffer, element_buffers corresponds
    * to an index in buffers for each vertex element */
   struct pan_vertex_buffer buffers[PIPE_MAX_ATTRIBS];
   unsigned element_buffer[PIPE_MAX_ATTRIBS];
   unsigned nr_bufs;

   /* Bitmask flagging attributes with a non-zero instance divisor which
    * require an attribute offset adjustment when base_instance != 0.
    * This is used to force attributes re-emission even if the vertex state
    * isn't dirty to take the new base instance into account. */
   uint32_t attr_depends_on_base_instance_mask;

   unsigned formats[PIPE_MAX_ATTRIBS];
#endif
};

static inline bool
panfrost_is_implicit_prim_restart(const struct pipe_draw_info *info)
{
   /* As a reminder primitive_restart should always be checked before any
      access to restart_index. */
   return info->primitive_restart &&
          info->restart_index == (unsigned)BITFIELD_MASK(info->index_size * 8);
}

static inline bool
pan_allow_forward_pixel_to_kill(struct panfrost_context *ctx,
                                struct panfrost_compiled_shader *fs)
{
   /* Track if any colour buffer is reused across draws, either
    * from reading it directly, or from failing to write it
    */
   unsigned rt_mask = ctx->fb_rt_mask;
   uint64_t rt_written = (fs->info.outputs_written >> FRAG_RESULT_DATA0) &
                         ctx->blend->enabled_mask;
   bool blend_reads_dest = (ctx->blend->load_dest_mask & rt_mask);
   bool alpha_to_coverage = ctx->blend->base.alpha_to_coverage;

   return fs->info.fs.can_fpk && !(rt_mask & ~rt_written) &&
          !alpha_to_coverage && !blend_reads_dest;
}

/*
 * Determine whether to set the respective overdraw alpha flag.
 *
 * The overdraw alpha=1 flag should be set when alpha=1 implies full overdraw,
 * equivalently, all enabled render targets have alpha_one_store set. Likewise,
 * overdraw alpha=0 should be set when alpha=0 implies no overdraw,
 * equivalently, all enabled render targets have alpha_zero_nop set.
 */
#if PAN_ARCH >= 6
static inline bool
panfrost_overdraw_alpha(const struct panfrost_context *ctx, bool zero)
{
   const struct panfrost_blend_state *so = ctx->blend;

   for (unsigned i = 0; i < ctx->pipe_framebuffer.nr_cbufs; ++i) {
      const struct pan_blend_info info = so->info[i];

      bool enabled = ctx->pipe_framebuffer.cbufs[i].texture && !info.enabled;
      bool flag = zero ? info.alpha_zero_nop : info.alpha_one_store;

      if (enabled && !flag)
         return false;
   }

   return true;
}
#endif

#if PAN_ARCH < 13
static inline void
panfrost_emit_primitive_size(struct panfrost_context *ctx, bool points,
                             uint64_t size_array,
                             struct mali_primitive_size_packed *prim_size)
{
   struct panfrost_rasterizer *rast = ctx->rasterizer;

   pan_pack(prim_size, PRIMITIVE_SIZE, cfg) {
      if (panfrost_writes_point_size(ctx)) {
         cfg.size_array = size_array;
      } else {
         cfg.fixed_sized = points ? rast->base.point_size : rast->base.line_width;
      }
   }
}
#endif

static inline uint8_t
pan_draw_mode(enum mesa_prim mode)
{
   switch (mode) {

#define DEFINE_CASE(c)                                                         \
   case MESA_PRIM_##c:                                                         \
      return MALI_DRAW_MODE_##c;

      DEFINE_CASE(POINTS);
      DEFINE_CASE(LINES);
      DEFINE_CASE(LINE_LOOP);
      DEFINE_CASE(LINE_STRIP);
      DEFINE_CASE(TRIANGLES);
      DEFINE_CASE(TRIANGLE_STRIP);
      DEFINE_CASE(TRIANGLE_FAN);
      DEFINE_CASE(QUADS);
      DEFINE_CASE(POLYGON);
#if PAN_ARCH <= 6
      DEFINE_CASE(QUAD_STRIP);
#endif

#undef DEFINE_CASE

   default:
      UNREACHABLE("Invalid draw mode");
   }
}

static inline enum mali_index_type
panfrost_translate_index_size(unsigned size)
{
   STATIC_ASSERT(MALI_INDEX_TYPE_NONE == 0);
   STATIC_ASSERT(MALI_INDEX_TYPE_UINT8 == 1);
   STATIC_ASSERT(MALI_INDEX_TYPE_UINT16 == 2);

   return (size == 4) ? MALI_INDEX_TYPE_UINT32 : size;
}

static inline bool
panfrost_fs_required(struct panfrost_compiled_shader *fs,
                     struct panfrost_blend_state *blend,
                     struct pipe_framebuffer_state *state,
                     const struct panfrost_zsa_state *zsa)
{
   /* If we generally have side effects. This inclues use of discard,
    * which can affect the results of an occlusion query. */
   if (fs->info.fs.sidefx)
      return true;

   /* Using an empty FS requires early-z to be enabled, but alpha test
    * needs it disabled. Alpha test is only native on Midgard, so only
    * check there.
    */
   if (PAN_ARCH <= 5 && zsa->base.alpha_func != PIPE_FUNC_ALWAYS)
      return true;

   /* If colour is written we need to execute */
   for (unsigned i = 0; i < state->nr_cbufs; ++i) {
      if (state->cbufs[i].texture && blend->info[i].enabled)
         return true;
   }

   /* If depth is written and not implied we need to execute.
    * TODO: Predicate on Z/S writes being enabled */
   return (fs->info.fs.writes_depth || fs->info.fs.writes_stencil);
}

#if PAN_ARCH >= 9
static inline uint64_t
panfrost_get_position_shader(struct panfrost_batch *batch,
                             const struct pipe_draw_info *info)
{
   /* IDVS/points vertex shader */
   uint64_t vs_ptr = batch->rsd[PIPE_SHADER_VERTEX];

   /* IDVS/triangle vertex shader */
   if (vs_ptr && info->mode != MESA_PRIM_POINTS)
      vs_ptr += pan_size(SHADER_PROGRAM);

   return vs_ptr;
}

#if PAN_ARCH < 12
static inline uint64_t
panfrost_get_varying_shader(struct panfrost_batch *batch)
{
   return batch->rsd[PIPE_SHADER_VERTEX] + (2 * pan_size(SHADER_PROGRAM));
}
#endif

static inline unsigned
panfrost_vertex_attribute_stride(struct panfrost_compiled_shader *vs,
                                 struct panfrost_compiled_shader *fs)
{
   unsigned v = vs->info.varyings.output_count;
   unsigned f = fs->info.varyings.input_count;
   unsigned slots = MAX2(v, f);
   slots += util_bitcount(vs->info.varyings.fixed_varyings);

   /* Assumes 16 byte slots. We could do better. */
   return slots * 16;
}

static inline uint64_t
panfrost_emit_resources(struct panfrost_batch *batch,
                        enum pipe_shader_type stage)
{
   struct panfrost_context *ctx = batch->ctx;
   struct pan_ptr T;
   unsigned nr_tables =
      ALIGN_POT(PAN_NUM_RESOURCE_TABLES, MALI_RESOURCE_TABLE_SIZE_ALIGNMENT);

   /* Although individual resources need only 16 byte alignment, the
    * resource table as a whole must be 64-byte aligned.
    */
   T = pan_pool_alloc_aligned(&batch->pool.base, nr_tables * pan_size(RESOURCE),
                              64);
   if (!T.cpu)
      return 0;

   memset(T.cpu, 0, nr_tables * pan_size(RESOURCE));

   pan_make_resource_table(T, PAN_TABLE_UBO, batch->uniform_buffers[stage],
                           batch->nr_uniform_buffers[stage]);

   pan_make_resource_table(T, PAN_TABLE_TEXTURE, batch->textures[stage],
                           ctx->sampler_view_count[stage]);

   /* We always need at least 1 sampler for txf to work */
   pan_make_resource_table(T, PAN_TABLE_SAMPLER, batch->samplers[stage],
                           MAX2(ctx->sampler_count[stage], 1));

   pan_make_resource_table(T, PAN_TABLE_IMAGE, batch->images[stage],
                           util_last_bit(ctx->image_mask[stage]));

   if (stage == PIPE_SHADER_FRAGMENT) {
      pan_make_resource_table(T, PAN_TABLE_ATTRIBUTE, batch->attribs[stage],
                              batch->nr_varying_attribs[PIPE_SHADER_FRAGMENT]);
   } else if (stage == PIPE_SHADER_VERTEX) {
      pan_make_resource_table(T, PAN_TABLE_ATTRIBUTE, batch->attribs[stage],
                              ctx->vertex->num_elements);

      pan_make_resource_table(T, PAN_TABLE_ATTRIBUTE_BUFFER,
                              batch->attrib_bufs[stage],
                              util_last_bit(ctx->vb_mask));
   }

   pan_make_resource_table(T, PAN_TABLE_SSBO, batch->ssbos[stage],
                           util_last_bit(ctx->ssbo_mask[stage]));

   return T.gpu | nr_tables;
}
#endif /* PAN_ARCH >= 9 */

static bool
allow_rotating_primitives(const struct panfrost_compiled_shader *fs,
                          const struct pipe_draw_info *info)
{
   return u_reduced_prim(info->mode) != MESA_PRIM_LINES &&
          !fs->info.bifrost.uses_flat_shading;
}

#endif
