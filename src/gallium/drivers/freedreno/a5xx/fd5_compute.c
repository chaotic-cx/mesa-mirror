/*
 * Copyright © 2017 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "pipe/p_state.h"

#include "freedreno_resource.h"

#include "fd5_compute.h"
#include "fd5_context.h"
#include "fd5_emit.h"

/* maybe move to fd5_program? */
static void
cs_program_emit(struct fd_context *ctx, struct fd_ringbuffer *ring, struct ir3_shader_variant *v) assert_dt
{
   const struct ir3_info *i = &v->info;
   enum a3xx_threadsize thrsz = i->double_threadsize ? FOUR_QUADS : TWO_QUADS;
   unsigned instrlen = v->instrlen;

   /* if shader is more than 32*16 instructions, don't preload it.  Similar
    * to the combined restriction of 64*16 for VS+FS
    */
   if (instrlen > 32)
      instrlen = 0;

   OUT_PKT4(ring, REG_A5XX_SP_SP_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* SP_SP_CNTL */

   OUT_PKT4(ring, REG_A5XX_HLSQ_CONTROL_0_REG, 1);
   OUT_RING(ring, A5XX_HLSQ_CONTROL_0_REG_FSTHREADSIZE(TWO_QUADS) |
                     A5XX_HLSQ_CONTROL_0_REG_CSTHREADSIZE(thrsz) |
                     0x00000880 /* XXX */);

   OUT_PKT4(ring, REG_A5XX_SP_CS_CTRL_REG0, 1);
   OUT_RING(ring,
            A5XX_SP_CS_CTRL_REG0_THREADSIZE(thrsz) |
               A5XX_SP_CS_CTRL_REG0_HALFREGFOOTPRINT(i->max_half_reg + 1) |
               A5XX_SP_CS_CTRL_REG0_FULLREGFOOTPRINT(i->max_reg + 1) |
               A5XX_SP_CS_CTRL_REG0_BRANCHSTACK(ir3_shader_branchstack_hw(v)) |
               COND(instrlen != 0, A5XX_SP_CS_CTRL_REG0_BUFFER) |
               0x2 /* XXX */);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONFIG, 1);
   OUT_RING(ring, A5XX_HLSQ_CS_CONFIG_CONSTOBJECTOFFSET(0) |
                     A5XX_HLSQ_CS_CONFIG_SHADEROBJOFFSET(0) |
                     A5XX_HLSQ_CS_CONFIG_ENABLED);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CNTL, 1);
   OUT_RING(ring, A5XX_HLSQ_CS_CNTL_INSTRLEN(instrlen) |
                     COND(v->has_ssbo, A5XX_HLSQ_CS_CNTL_SSBO_ENABLE));

   OUT_PKT4(ring, REG_A5XX_SP_CS_CONFIG, 1);
   OUT_RING(ring, A5XX_SP_CS_CONFIG_CONSTOBJECTOFFSET(0) |
                     A5XX_SP_CS_CONFIG_SHADEROBJOFFSET(0) |
                     A5XX_SP_CS_CONFIG_ENABLED);

   assert(v->constlen % 4 == 0);
   unsigned constlen = v->constlen / 4;
   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONSTLEN, 2);
   OUT_RING(ring, constlen); /* HLSQ_CS_CONSTLEN */
   OUT_RING(ring, instrlen); /* HLSQ_CS_INSTRLEN */

   fd5_emit_shader_obj(ctx, ring, v, REG_A5XX_SP_CS_OBJ_START_LO);

   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0x1f00000);

   uint32_t local_invocation_id, work_group_id;
   local_invocation_id =
      ir3_find_sysval_regid(v, SYSTEM_VALUE_LOCAL_INVOCATION_ID);
   work_group_id = ir3_find_sysval_regid(v, SYSTEM_VALUE_WORKGROUP_ID);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CNTL_0, 2);
   OUT_RING(ring, A5XX_HLSQ_CS_CNTL_0_WGIDCONSTID(work_group_id) |
                     A5XX_HLSQ_CS_CNTL_0_UNK0(regid(63, 0)) |
                     A5XX_HLSQ_CS_CNTL_0_UNK1(regid(63, 0)) |
                     A5XX_HLSQ_CS_CNTL_0_LOCALIDREGID(local_invocation_id));
   OUT_RING(ring, 0x1); /* HLSQ_CS_CNTL_1 */

   if (instrlen > 0)
      fd5_emit_shader(ring, v);
}

static void
fd5_launch_grid(struct fd_context *ctx,
                const struct pipe_grid_info *info) assert_dt
{
   struct ir3_shader_key key = {};
   struct ir3_shader_variant *v;
   struct fd_ringbuffer *ring = ctx->batch->draw;
   unsigned nglobal = 0;

   v =
      ir3_shader_variant(ir3_get_shader(ctx->compute), key, false, &ctx->debug);
   if (!v)
      return;

   if (ctx->dirty_shader[PIPE_SHADER_COMPUTE] & FD_DIRTY_SHADER_PROG)
      cs_program_emit(ctx, ring, v);

   fd5_emit_cs_state(ctx, ring, v);
   fd5_emit_cs_consts(v, ring, ctx, info);

   util_dynarray_foreach (&ctx->global_bindings, struct pipe_resource *, res)
      nglobal++;

   if (nglobal > 0) {
      /* global resources don't otherwise get an OUT_RELOC(), since
       * the raw ptr address is emitted ir ir3_emit_cs_consts().
       * So to make the kernel aware that these buffers are referenced
       * by the batch, emit dummy reloc's as part of a no-op packet
       * payload:
       */
      OUT_PKT7(ring, CP_NOP, 2 * nglobal);
      util_dynarray_foreach (&ctx->global_bindings, struct pipe_resource *, res)
         OUT_RELOC(ring, fd_resource(*res)->bo, 0, 0, 0);
   }

   const unsigned *local_size =
      info->block; // v->shader->nir->info->workgroup_size;
   const unsigned *num_groups = info->grid;
   /* for some reason, mesa/st doesn't set info->work_dim, so just assume 3: */
   const unsigned work_dim = info->work_dim ? info->work_dim : 3;
   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_NDRANGE_0, 7);
   OUT_RING(ring, A5XX_HLSQ_CS_NDRANGE_0_KERNELDIM(work_dim) |
                     A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEX(local_size[0] - 1) |
                     A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEY(local_size[1] - 1) |
                     A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEZ(local_size[2] - 1));
   OUT_RING(ring,
            A5XX_HLSQ_CS_NDRANGE_1_GLOBALSIZE_X(local_size[0] * num_groups[0]));
   OUT_RING(ring, 0); /* HLSQ_CS_NDRANGE_2_GLOBALOFF_X */
   OUT_RING(ring,
            A5XX_HLSQ_CS_NDRANGE_3_GLOBALSIZE_Y(local_size[1] * num_groups[1]));
   OUT_RING(ring, 0); /* HLSQ_CS_NDRANGE_4_GLOBALOFF_Y */
   OUT_RING(ring,
            A5XX_HLSQ_CS_NDRANGE_5_GLOBALSIZE_Z(local_size[2] * num_groups[2]));
   OUT_RING(ring, 0); /* HLSQ_CS_NDRANGE_6_GLOBALOFF_Z */

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_KERNEL_GROUP_X, 3);
   OUT_RING(ring, 1); /* HLSQ_CS_KERNEL_GROUP_X */
   OUT_RING(ring, 1); /* HLSQ_CS_KERNEL_GROUP_Y */
   OUT_RING(ring, 1); /* HLSQ_CS_KERNEL_GROUP_Z */

   if (info->indirect) {
      struct fd_resource *rsc = fd_resource(info->indirect);

      fd5_emit_flush(ctx, ring);

      OUT_PKT7(ring, CP_EXEC_CS_INDIRECT, 4);
      OUT_RING(ring, 0x00000000);
      OUT_RELOC(ring, rsc->bo, info->indirect_offset, 0, 0); /* ADDR_LO/HI */
      OUT_RING(ring,
               A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEX(local_size[0] - 1) |
                  A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEY(local_size[1] - 1) |
                  A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEZ(local_size[2] - 1));
   } else {
      OUT_PKT7(ring, CP_EXEC_CS, 4);
      OUT_RING(ring, 0x00000000);
      OUT_RING(ring, CP_EXEC_CS_1_NGROUPS_X(info->grid[0]));
      OUT_RING(ring, CP_EXEC_CS_2_NGROUPS_Y(info->grid[1]));
      OUT_RING(ring, CP_EXEC_CS_3_NGROUPS_Z(info->grid[2]));
   }
}

void
fd5_compute_init(struct pipe_context *pctx) disable_thread_safety_analysis
{
   struct fd_context *ctx = fd_context(pctx);
   ctx->launch_grid = fd5_launch_grid;
   pctx->create_compute_state = ir3_shader_compute_state_create;
   pctx->delete_compute_state = ir3_shader_state_delete;
}
