/*
 * Copyright © 2016 Rob Clark <robclark@freedesktop.org>
 * Copyright © 2018 Google, Inc.
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#define FD_BO_NO_HARDPIN 1

#include "freedreno_query_acc.h"
#include "freedreno_state.h"

#include "fd6_barrier.h"
#include "fd6_blend.h"
#include "fd6_blitter.h"
#include "fd6_compute.h"
#include "fd6_context.h"
#include "fd6_draw.h"
#include "fd6_emit.h"
#include "fd6_gmem.h"
#include "fd6_image.h"
#include "fd6_pack.h"
#include "fd6_program.h"
#include "fd6_query.h"
#include "fd6_rasterizer.h"
#include "fd6_resource.h"
#include "fd6_texture.h"
#include "fd6_zsa.h"

static void
fd6_context_destroy(struct pipe_context *pctx) in_dt
{
   struct fd6_context *fd6_ctx = fd6_context(fd_context(pctx));

   fd6_descriptor_set_invalidate(&fd6_ctx->cs_descriptor_set);
   for (unsigned i = 0; i < ARRAY_SIZE(fd6_ctx->descriptor_sets); i++)
      fd6_descriptor_set_invalidate(&fd6_ctx->descriptor_sets[i]);

   if (fd6_ctx->streamout_disable_stateobj)
      fd_ringbuffer_del(fd6_ctx->streamout_disable_stateobj);

   if (fd6_ctx->sample_locations_disable_stateobj)
      fd_ringbuffer_del(fd6_ctx->sample_locations_disable_stateobj);

   if (fd6_ctx->preamble)
      fd_ringbuffer_del(fd6_ctx->preamble);

   if (fd6_ctx->restore)
      fd_ringbuffer_del(fd6_ctx->restore);

   fd_context_destroy(pctx);

   if (fd6_ctx->vsc_draw_strm)
      fd_bo_del(fd6_ctx->vsc_draw_strm);
   if (fd6_ctx->vsc_prim_strm)
      fd_bo_del(fd6_ctx->vsc_prim_strm);
   fd_bo_del(fd6_ctx->control_mem);

   fd_context_cleanup_common_vbos(&fd6_ctx->base);

   fd6_texture_fini(pctx);

   free(fd6_ctx);
}

static void *
fd6_vertex_state_create(struct pipe_context *pctx, unsigned num_elements,
                        const struct pipe_vertex_element *elements)
{
   struct fd_context *ctx = fd_context(pctx);

   struct fd6_vertex_stateobj *state = CALLOC_STRUCT(fd6_vertex_stateobj);
   memcpy(state->base.pipe, elements, sizeof(*elements) * num_elements);
   state->base.num_elements = num_elements;
   state->stateobj =
      fd_ringbuffer_new_object(ctx->pipe, 4 * (num_elements * 4 + 1));
   struct fd_ringbuffer *ring = state->stateobj;

   OUT_PKT4(ring, REG_A6XX_VFD_FETCH_INSTR(0), 2 * num_elements);
   for (int32_t i = 0; i < num_elements; i++) {
      const struct pipe_vertex_element *elem = &elements[i];
      enum pipe_format pfmt = (enum pipe_format)elem->src_format;
      enum a6xx_format fmt = fd6_vertex_format(pfmt);
      bool isint = util_format_is_pure_integer(pfmt);
      assert(fmt != FMT6_NONE);

      OUT_RING(ring, A6XX_VFD_FETCH_INSTR_INSTR_IDX(elem->vertex_buffer_index) |
                        A6XX_VFD_FETCH_INSTR_INSTR_OFFSET(elem->src_offset) |
                        A6XX_VFD_FETCH_INSTR_INSTR_FORMAT(fmt) |
                        COND(elem->instance_divisor,
                             A6XX_VFD_FETCH_INSTR_INSTR_INSTANCED) |
                        A6XX_VFD_FETCH_INSTR_INSTR_SWAP(fd6_vertex_swap(pfmt)) |
                        A6XX_VFD_FETCH_INSTR_INSTR_UNK30 |
                        COND(!isint, A6XX_VFD_FETCH_INSTR_INSTR_FLOAT));
      OUT_RING(ring,
               MAX2(1, elem->instance_divisor)); /* VFD_FETCH_INSTR[j].STEP_RATE */
   }

   for (int32_t i = 0; i < num_elements; i++) {
      const struct pipe_vertex_element *elem = &elements[i];

      OUT_PKT4(ring, REG_A6XX_VFD_VERTEX_BUFFER_STRIDE(elem->vertex_buffer_index), 1);
      OUT_RING(ring, elem->src_stride);
   }

   return state;
}

static void
fd6_vertex_state_delete(struct pipe_context *pctx, void *hwcso)
{
   struct fd6_vertex_stateobj *so = (struct fd6_vertex_stateobj *)hwcso;

   fd_ringbuffer_del(so->stateobj);
   FREE(hwcso);
}

static void
validate_surface(struct pipe_context *pctx, const struct pipe_surface *psurf)
   assert_dt
{
   fd6_validate_format(fd_context(pctx), fd_resource(psurf->texture),
                       psurf->format);
}

static void
fd6_set_framebuffer_state(struct pipe_context *pctx,
                          const struct pipe_framebuffer_state *pfb)
   in_dt
{
   if (pfb->zsbuf.texture)
      validate_surface(pctx, &pfb->zsbuf);

   for (unsigned i = 0; i < pfb->nr_cbufs; i++) {
      if (!pfb->cbufs[i].texture)
         continue;
      validate_surface(pctx, &pfb->cbufs[i]);
   }

   fd_set_framebuffer_state(pctx, pfb);
}


static void
setup_state_map(struct fd_context *ctx)
{
   STATIC_ASSERT(FD6_GROUP_NON_GROUP < 32);

   fd_context_add_map(ctx, FD_DIRTY_VTXSTATE, BIT(FD6_GROUP_VTXSTATE));
   fd_context_add_map(ctx, FD_DIRTY_VTXBUF, BIT(FD6_GROUP_VBO));
   fd_context_add_map(ctx, FD_DIRTY_ZSA | FD_DIRTY_RASTERIZER,
                      BIT(FD6_GROUP_ZSA));
   fd_context_add_map(ctx, FD_DIRTY_ZSA | FD_DIRTY_BLEND | FD_DIRTY_PROG,
                      BIT(FD6_GROUP_LRZ));
   fd_context_add_map(ctx, FD_DIRTY_PROG | FD_DIRTY_RASTERIZER_CLIP_PLANE_ENABLE,
                      BIT(FD6_GROUP_PROG) | BIT(FD6_GROUP_PROG_KEY));
   fd_context_add_map(ctx, FD_DIRTY_RASTERIZER | FD_DIRTY_FRAMEBUFFER,
                      BIT(FD6_GROUP_PROG_KEY));
   if (ctx->screen->driconf.dual_color_blend_by_location) {
      fd_context_add_map(ctx, FD_DIRTY_BLEND_DUAL,
                         BIT(FD6_GROUP_PROG_KEY));
   }
   fd_context_add_map(ctx, FD_DIRTY_RASTERIZER, BIT(FD6_GROUP_RASTERIZER));
   fd_context_add_map(ctx,
                      FD_DIRTY_FRAMEBUFFER | FD_DIRTY_RASTERIZER_DISCARD |
                         FD_DIRTY_PROG | FD_DIRTY_BLEND_DUAL,
                      BIT(FD6_GROUP_PROG_FB_RAST));
   fd_context_add_map(ctx, FD_DIRTY_BLEND | FD_DIRTY_SAMPLE_MASK,
                      BIT(FD6_GROUP_BLEND));
   fd_context_add_map(ctx, FD_DIRTY_SAMPLE_LOCATIONS, BIT(FD6_GROUP_SAMPLE_LOCATIONS));
   fd_context_add_map(ctx, FD_DIRTY_BLEND_COLOR, BIT(FD6_GROUP_BLEND_COLOR));
   fd_context_add_map(ctx, FD_DIRTY_PROG | FD_DIRTY_CONST,
                      BIT(FD6_GROUP_CONST));
   fd_context_add_map(ctx, FD_DIRTY_STREAMOUT, BIT(FD6_GROUP_SO));
   fd_context_add_map(ctx, FD_DIRTY_BLEND_COHERENT,
      BIT(FD6_GROUP_PRIM_MODE_SYSMEM) | BIT(FD6_GROUP_PRIM_MODE_GMEM));

   fd_context_add_shader_map(ctx, PIPE_SHADER_VERTEX, FD_DIRTY_SHADER_TEX,
                             BIT(FD6_GROUP_VS_TEX));
   fd_context_add_shader_map(ctx, PIPE_SHADER_TESS_CTRL, FD_DIRTY_SHADER_TEX,
                             BIT(FD6_GROUP_HS_TEX));
   fd_context_add_shader_map(ctx, PIPE_SHADER_TESS_EVAL, FD_DIRTY_SHADER_TEX,
                             BIT(FD6_GROUP_DS_TEX));
   fd_context_add_shader_map(ctx, PIPE_SHADER_GEOMETRY, FD_DIRTY_SHADER_TEX,
                             BIT(FD6_GROUP_GS_TEX));
   fd_context_add_shader_map(ctx, PIPE_SHADER_FRAGMENT, FD_DIRTY_SHADER_TEX,
                             BIT(FD6_GROUP_FS_TEX));
   fd_context_add_shader_map(ctx, PIPE_SHADER_COMPUTE, FD_DIRTY_SHADER_TEX,
                             BIT(FD6_GROUP_CS_TEX));

   fd_context_add_shader_map(ctx, PIPE_SHADER_VERTEX,
                             FD_DIRTY_SHADER_SSBO | FD_DIRTY_SHADER_IMAGE,
                             BIT(FD6_GROUP_VS_BINDLESS));
   fd_context_add_shader_map(ctx, PIPE_SHADER_TESS_CTRL,
                             FD_DIRTY_SHADER_SSBO | FD_DIRTY_SHADER_IMAGE,
                             BIT(FD6_GROUP_HS_BINDLESS));
   fd_context_add_shader_map(ctx, PIPE_SHADER_TESS_EVAL,
                             FD_DIRTY_SHADER_SSBO | FD_DIRTY_SHADER_IMAGE,
                             BIT(FD6_GROUP_DS_BINDLESS));
   fd_context_add_shader_map(ctx, PIPE_SHADER_GEOMETRY,
                             FD_DIRTY_SHADER_SSBO | FD_DIRTY_SHADER_IMAGE,
                             BIT(FD6_GROUP_GS_BINDLESS));
   /* NOTE: FD6_GROUP_FS_BINDLESS has a weak dependency on the program
    * state (ie. it needs to be re-generated with fb-read descriptor
    * patched in) but this special case is handled in fd6_emit_3d_state()
    */
   fd_context_add_shader_map(ctx, PIPE_SHADER_FRAGMENT,
                             FD_DIRTY_SHADER_SSBO | FD_DIRTY_SHADER_IMAGE,
                             BIT(FD6_GROUP_FS_BINDLESS));
   fd_context_add_shader_map(ctx, PIPE_SHADER_COMPUTE,
                             FD_DIRTY_SHADER_SSBO | FD_DIRTY_SHADER_IMAGE,
                             BIT(FD6_GROUP_CS_BINDLESS));
   fd_context_add_shader_map(ctx, PIPE_SHADER_FRAGMENT,
                             FD_DIRTY_SHADER_PROG,
                             BIT(FD6_GROUP_PRIM_MODE_SYSMEM) | BIT(FD6_GROUP_PRIM_MODE_GMEM));

   /* NOTE: scissor enabled bit is part of rasterizer state, but
    * fd_rasterizer_state_bind() will mark scissor dirty if needed:
    */
   fd_context_add_map(ctx, FD_DIRTY_SCISSOR | FD_DIRTY_PROG,
                      BIT(FD6_GROUP_SCISSOR));

   /* Stuff still emit in IB2
    *
    * NOTE: viewport state doesn't seem to change frequently, so possibly
    * move it into FD6_GROUP_RASTERIZER?
    */
   fd_context_add_map(
      ctx, FD_DIRTY_STENCIL_REF | FD_DIRTY_VIEWPORT | FD_DIRTY_RASTERIZER | FD_DIRTY_PROG,
      BIT(FD6_GROUP_NON_GROUP));
}

template <chip CHIP>
struct pipe_context *
fd6_context_create(struct pipe_screen *pscreen, void *priv,
                   unsigned flags) disable_thread_safety_analysis
{
   struct fd_screen *screen = fd_screen(pscreen);
   struct fd6_context *fd6_ctx = CALLOC_STRUCT(fd6_context);
   struct pipe_context *pctx;

   if (!fd6_ctx)
      return NULL;

   pctx = &fd6_ctx->base.base;
   pctx->screen = pscreen;

   fd6_ctx->base.flags = flags;
   fd6_ctx->base.dev = fd_device_ref(screen->dev);
   fd6_ctx->base.screen = fd_screen(pscreen);
   fd6_ctx->base.last.key = &fd6_ctx->last_key;

   pctx->destroy = fd6_context_destroy;
   pctx->create_blend_state = fd6_blend_state_create;
   pctx->create_rasterizer_state = fd6_rasterizer_state_create;
   pctx->create_depth_stencil_alpha_state = fd6_zsa_state_create<CHIP>;
   pctx->create_vertex_elements_state = fd6_vertex_state_create;

   fd6_draw_init<CHIP>(pctx);
   fd6_compute_init<CHIP>(pctx);
   fd6_gmem_init<CHIP>(pctx);
   fd6_texture_init(pctx);
   fd6_prog_init<CHIP>(pctx);
   fd6_query_context_init<CHIP>(pctx);

   setup_state_map(&fd6_ctx->base);

   pctx = fd_context_init(&fd6_ctx->base, pscreen, priv, flags);
   if (!pctx) {
      free(fd6_ctx);
      return NULL;
   }

   pctx->set_framebuffer_state = fd6_set_framebuffer_state;

   /* after fd_context_init() to override set_shader_images() */
   fd6_image_init(pctx);

   /* after fd_context_init() to override memory_barrier/texture_barrier(): */
   fd6_barrier_init(pctx);

   util_blitter_set_texture_multisample(fd6_ctx->base.blitter, true);

   pctx->delete_vertex_elements_state = fd6_vertex_state_delete;

   /* fd_context_init overwrites delete_rasterizer_state, so set this
    * here. */
   pctx->delete_rasterizer_state = fd6_rasterizer_state_delete;
   pctx->delete_blend_state = fd6_blend_state_delete;
   pctx->delete_depth_stencil_alpha_state = fd6_zsa_state_delete;

   /* initial sizes for VSC buffers (or rather the per-pipe sizes
    * which is used to derive entire buffer size:
    */
   fd6_ctx->vsc_draw_strm_pitch = 0x440;
   fd6_ctx->vsc_prim_strm_pitch = 0x1040;

   fd6_ctx->control_mem =
      fd_bo_new(screen->dev, 0x1000, 0, "control");

   fd_context_add_private_bo(&fd6_ctx->base, fd6_ctx->control_mem);

   memset(fd_bo_map(fd6_ctx->control_mem), 0, sizeof(struct fd6_control));

   fd_context_setup_common_vbos(&fd6_ctx->base);

   fd6_blitter_init<CHIP>(pctx);

   struct fd_ringbuffer *ring =
      fd_ringbuffer_new_object(fd6_ctx->base.pipe, 6 * 4);

   OUT_REG(ring, A6XX_GRAS_SC_MSAA_SAMPLE_POS_CNTL());
   OUT_REG(ring, A6XX_RB_MSAA_SAMPLE_POS_CNTL());
   OUT_REG(ring, A6XX_TPL1_MSAA_SAMPLE_POS_CNTL());

   fd6_ctx->sample_locations_disable_stateobj = ring;

   fd6_ctx->preamble = fd6_build_preemption_preamble<CHIP>(&fd6_ctx->base);

   ring = fd_ringbuffer_new_object(fd6_ctx->base.pipe, 0x1000);
   fd6_emit_static_regs<CHIP>(&fd6_ctx->base, ring);
   fd6_ctx->restore = ring;

   return fd_context_init_tc(pctx, flags);
}
FD_GENX(fd6_context_create);
