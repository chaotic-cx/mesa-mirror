/*
 * Copyright © 2016 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 */

#include "ir3/ir3_compiler.h"

#include "util/u_math.h"

#include "common/fd5_hw.h"
#include "common/freedreno_dev_info.h"

#include "ir3_asm.h"
#include "main.h"

struct a5xx_backend {
   struct backend base;

   struct ir3_compiler *compiler;
   struct fd_device *dev;

   const struct fd_dev_id *dev_id;
   const struct fd_dev_info *info;

   unsigned seqno;
   struct fd_bo *control_mem;

   struct fd_bo *query_mem;
   const struct perfcntr *perfcntrs;
   unsigned num_perfcntrs;
};
define_cast(backend, a5xx_backend);

/*
 * Data structures shared with GPU:
 */

/* This is a stripped down version of fd5_context::blit_mem, used to
 * synchronize with the CP:
 */
struct fd5_control {
   uint32_t seqno;
   uint32_t _pad[7];
};

#define control_ptr(a5xx_backend, member)                                      \
   (a5xx_backend)->control_mem, offsetof(struct fd5_control, member)

struct PACKED fd5_query_sample {
   uint64_t start;
   uint64_t result;
   uint64_t stop;
};

/* offset of a single field of an array of fd5_query_sample: */
#define query_sample_idx(a5xx_backend, idx, field)                             \
   (a5xx_backend)->query_mem, (idx * sizeof(struct fd5_query_sample)) +        \
                                 offsetof(struct fd5_query_sample, field)

/*
 * Backend implementation:
 */

static struct kernel *
a5xx_assemble(struct backend *b, FILE *in)
{
   struct a5xx_backend *a5xx_backend = to_a5xx_backend(b);
   struct ir3_kernel *ir3_kernel = ir3_asm_assemble(a5xx_backend->compiler, in);
   ir3_kernel->backend = b;

   for (unsigned i = 0; i < ir3_kernel->base.num_bufs; i++) {
      if (ir3_kernel->base.buf_types[i] == KERNEL_BUF_UBO)
         errx(-1, "@ubo is not supported on a5xx");
   }

   return &ir3_kernel->base;
}

static void
a5xx_disassemble(struct kernel *kernel, struct ir3_disasm_options *options)
{
   ir3_asm_disassemble(to_ir3_kernel(kernel), options);
}

/* The compute-relevant subset of fd5_emit_restore(). */
static void
cs_restore_emit(struct fd_ringbuffer *ring, struct a5xx_backend *a5xx_backend)
{
   OUT_PKT7(ring, CP_SET_RENDER_MODE, 5);
   OUT_RING(ring, CP_SET_RENDER_MODE_0_MODE(BYPASS));
   OUT_RING(ring, 0x00000000); /* ADDR_LO */
   OUT_RING(ring, 0x00000000); /* ADDR_HI */
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UCHE_CACHE_INVALIDATE_MIN, 5);
   OUT_RING(ring, 0x00000000); /* UCHE_CACHE_INVALIDATE_MIN_LO */
   OUT_RING(ring, 0x00000000); /* UCHE_CACHE_INVALIDATE_MIN_HI */
   OUT_RING(ring, 0x00000000); /* UCHE_CACHE_INVALIDATE_MAX_LO */
   OUT_RING(ring, 0x00000000); /* UCHE_CACHE_INVALIDATE_MAX_HI */
   OUT_RING(ring, 0x00000012); /* UCHE_CACHE_INVALIDATE */

   OUT_WFI5(ring);

   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0x000fffff);

   OUT_PKT4(ring, REG_A5XX_RB_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000044);

   OUT_PKT4(ring, REG_A5XX_RB_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x00100000);

   OUT_PKT4(ring, REG_A5XX_PC_MODE_CNTL, 1);
   OUT_RING(ring, 0x0000001f);

   OUT_PKT4(ring, REG_A5XX_SP_MODE_CNTL, 1);
   OUT_RING(ring, 0x0000001e);

   if (fd_dev_gpu_id(a5xx_backend->dev_id) == 540) {
      OUT_PKT4(ring, REG_A5XX_SP_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x00000800);

      OUT_PKT4(ring, REG_A5XX_HLSQ_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x00000000);
   } else {
      OUT_PKT4(ring, REG_A5XX_SP_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x40000800);
   }

   OUT_PKT4(ring, REG_A5XX_TPL1_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000544);

   OUT_PKT4(ring, REG_A5XX_HLSQ_TIMEOUT_THRESHOLD_0, 2);
   OUT_RING(ring, 0x00000080); /* HLSQ_TIMEOUT_THRESHOLD_0 */
   OUT_RING(ring, 0x00000000); /* HLSQ_TIMEOUT_THRESHOLD_1 */

   OUT_PKT4(ring, REG_A5XX_VPC_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x00000400);

   OUT_PKT4(ring, REG_A5XX_HLSQ_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000001);

   OUT_PKT4(ring, REG_A5XX_VPC_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000000);
}

static void
cs_program_emit(struct fd_ringbuffer *ring, struct kernel *kernel)
{
   struct ir3_kernel *ir3_kernel = to_ir3_kernel(kernel);
   struct a5xx_backend *a5xx_backend = to_a5xx_backend(ir3_kernel->backend);
   struct ir3_shader_variant *v = ir3_kernel->v;
   const struct ir3_info *i = &v->info;
   enum a3xx_threadsize thrsz = i->double_threadsize ? FOUR_QUADS : TWO_QUADS;

   /* if shader is more than 32*16 instructions, don't preload it.  Similar
    * to the combined restriction of 64*16 for VS+FS
    */
   unsigned instrlen = (v->instrlen > 32) ? 0 : v->instrlen;

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
                     COND(kernel->num_bufs > 0, A5XX_HLSQ_CS_CNTL_SSBO_ENABLE));

   OUT_PKT4(ring, REG_A5XX_SP_CS_CONFIG, 1);
   OUT_RING(ring, A5XX_SP_CS_CONFIG_CONSTOBJECTOFFSET(0) |
                     A5XX_SP_CS_CONFIG_SHADEROBJOFFSET(0) |
                     A5XX_SP_CS_CONFIG_ENABLED);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONSTLEN, 2);
   OUT_RING(ring, align(v->constlen, 4) / 4); /* HLSQ_CS_CONSTLEN */
   OUT_RING(ring, instrlen);                  /* HLSQ_CS_INSTRLEN */

   OUT_PKT4(ring, REG_A5XX_TPL1_CS_TEX_COUNT, 1);
   OUT_RING(ring, v->num_samp);

   uint32_t per_fiber_size = v->pvtmem_size;
   uint32_t per_sp_size =
      align(per_fiber_size * a5xx_backend->info->fibers_per_sp, 1 << 12);

   OUT_PKT4(ring, REG_A5XX_SP_CS_OBJ_START, 6);
   OUT_RELOC(ring, v->bo, 0, 0, 0); /* SP_CS_OBJ_START */
   OUT_RING(ring, A5XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM(per_fiber_size) |
                     A5XX_SP_CS_PVT_MEM_PARAM_HWSTACKOFFSET(per_sp_size));
   if (v->pvtmem_size > 0) { /* SP_CS_PVT_MEM_ADDR */
      uint32_t total_size = per_sp_size * a5xx_backend->info->num_sp_cores;
      struct fd_bo *pvtmem =
         fd_bo_new(a5xx_backend->dev, total_size, FD_BO_NOMAP, "pvtmem");
      OUT_RELOC(ring, pvtmem, 0, 0, 0);
   } else {
      OUT_RING(ring, 0);
      OUT_RING(ring, 0);
   }
   OUT_RING(ring, A5XX_SP_CS_PVT_MEM_SIZE_TOTALPVTMEMSIZE(per_sp_size));

   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0x01f00000);

   uint32_t local_invocation_id, work_group_id;
   local_invocation_id =
      ir3_find_sysval_regid(v, SYSTEM_VALUE_LOCAL_INVOCATION_ID);
   work_group_id = ir3_find_sysval_regid(v, SYSTEM_VALUE_WORKGROUP_ID);

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CNTL_0, 2);
   OUT_RING(ring, A5XX_HLSQ_CS_CNTL_0_WGIDCONSTID(work_group_id) |
                     A5XX_HLSQ_CS_CNTL_0_UNK0(INVALID_REG) |
                     A5XX_HLSQ_CS_CNTL_0_UNK1(INVALID_REG) |
                     A5XX_HLSQ_CS_CNTL_0_LOCALIDREGID(local_invocation_id));
   OUT_RING(ring, A5XX_HLSQ_CS_CNTL_1_WORKITEMRASTORDER(WORKITEMRASTORDER_LINEAR));

   if (instrlen > 0) {
      /* preload the shader into the instruction cache: */
      OUT_PKT7(ring, CP_LOAD_STATE4, 3);
      OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(0) |
                        CP_LOAD_STATE4_0_STATE_SRC(SS4_INDIRECT) |
                        CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SHADER) |
                        CP_LOAD_STATE4_0_NUM_UNIT(instrlen));
      OUT_RELOC(ring, v->bo, 0, CP_LOAD_STATE4_1_STATE_TYPE(ST4_SHADER), 0);
   }
}

static void
emit_const(struct fd_ringbuffer *ring, uint32_t constid, uint32_t sizedwords,
           const uint32_t *dwords)
{
   uint32_t align_sz;

   assert((constid % 4) == 0);

   align_sz = align(sizedwords, 4);

   OUT_PKT7(ring, CP_LOAD_STATE4, 3 + align_sz);
   OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(constid / 4) |
                     CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                     CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SHADER) |
                     CP_LOAD_STATE4_0_NUM_UNIT(DIV_ROUND_UP(sizedwords, 4)));
   OUT_RING(ring, CP_LOAD_STATE4_1_EXT_SRC_ADDR(0) |
                     CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
   OUT_RING(ring, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   for (unsigned i = 0; i < sizedwords; i++) {
      OUT_RING(ring, dwords[i]);
   }

   /* Zero-pad to multiple of 4 dwords */
   for (uint32_t i = sizedwords; i < align_sz; i++) {
      OUT_RING(ring, 0);
   }
}

static void
cs_const_emit(struct fd_ringbuffer *ring, struct kernel *kernel,
              uint32_t grid[3])
{
   struct ir3_kernel *ir3_kernel = to_ir3_kernel(kernel);
   struct ir3_shader_variant *v = ir3_kernel->v;

   const struct ir3_const_state *const_state = ir3_const_state(v);
   uint32_t base = const_state->allocs.max_const_offset_vec4;
   const struct ir3_imm_const_state *imm_state = &v->imm_state;
   int size = DIV_ROUND_UP(imm_state->count, 4);

   /* Same as 6xx, patch in numwg as a const. */
   if (ir3_kernel->info.numwg != INVALID_REG) {
      assert((ir3_kernel->info.numwg & 0x3) == 0);
      int idx = ir3_kernel->info.numwg >> 2;
      imm_state->values[idx * 4 + 0] = grid[0];
      imm_state->values[idx * 4 + 1] = grid[1];
      imm_state->values[idx * 4 + 2] = grid[2];
   }

   for (int i = 0; i < MAX_BUFS; i++) {
      if (kernel->buf_addr_regs[i] != INVALID_REG) {
         assert((kernel->buf_addr_regs[i] & 0x3) == 0);
         int idx = kernel->buf_addr_regs[i] >> 2;

         uint64_t iova = fd_bo_get_iova(kernel->bufs[i]);

         imm_state->values[idx * 4 + 1] = iova >> 32;
         imm_state->values[idx * 4 + 0] = (iova << 32) >> 32;
      }
   }

   /* truncate size to avoid writing constants that shader
    * does not use:
    */
   size = MIN2(size + base, v->constlen) - base;

   /* convert out of vec4: */
   base *= 4;
   size *= 4;

   if (size > 0) {
      emit_const(ring, base, size, imm_state->values);
   }
}

static void
cs_ibo_emit(struct fd_ringbuffer *ring, struct kernel *kernel)
{
   if (kernel->num_bufs == 0)
      return;

   OUT_PKT7(ring, CP_LOAD_STATE4, 3 + (2 * kernel->num_bufs));
   OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(0) |
                     CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                     CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SSBO) |
                     CP_LOAD_STATE4_0_NUM_UNIT(kernel->num_bufs));
   OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) |
                     CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
   OUT_RING(ring, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   for (unsigned i = 0; i < kernel->num_bufs; i++) {
      /* Unlike a4xx/a6xx, the size is in bytes, and overflows from width
       * into height:
       */
      unsigned sz = kernel->buf_sizes[i] * 4;

      OUT_RING(ring, A5XX_SSBO_1_0_WIDTH(sz));
      OUT_RING(ring, A5XX_SSBO_1_1_HEIGHT(sz >> 16));
   }

   OUT_PKT7(ring, CP_LOAD_STATE4, 3 + (2 * kernel->num_bufs));
   OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(0) |
                     CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                     CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SSBO) |
                     CP_LOAD_STATE4_0_NUM_UNIT(kernel->num_bufs));
   OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(ST4_UBO) |
                     CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
   OUT_RING(ring, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   for (unsigned i = 0; i < kernel->num_bufs; i++) {
      OUT_RELOC(ring, kernel->bufs[i], 0, 0, 0);
   }
}

static void
cache_flush(struct fd_ringbuffer *ring, struct kernel *kernel)
{
   struct ir3_kernel *ir3_kernel = to_ir3_kernel(kernel);
   struct a5xx_backend *a5xx_backend = to_a5xx_backend(ir3_kernel->backend);
   unsigned seqno = ++a5xx_backend->seqno;

   OUT_PKT7(ring, CP_EVENT_WRITE, 4);
   OUT_RING(ring, CACHE_FLUSH_TS);
   OUT_RELOC(ring, control_ptr(a5xx_backend, seqno));
   OUT_RING(ring, seqno);

   OUT_PKT7(ring, CP_WAIT_REG_MEM, 6);
   OUT_RING(ring, CP_WAIT_REG_MEM_0_FUNCTION(WRITE_EQ) |
                     CP_WAIT_REG_MEM_0_POLL(POLL_MEMORY));
   OUT_RELOC(ring, control_ptr(a5xx_backend, seqno));
   OUT_RING(ring, CP_WAIT_REG_MEM_3_REF(seqno));
   OUT_RING(ring, CP_WAIT_REG_MEM_4_MASK(~0));
   OUT_RING(ring, CP_WAIT_REG_MEM_5_DELAY_LOOP_CYCLES(16));
}

static void
a5xx_emit_grid(struct kernel *kernel, uint32_t grid[3],
               struct fd_submit *submit)
{
   struct ir3_kernel *ir3_kernel = to_ir3_kernel(kernel);
   struct a5xx_backend *a5xx_backend = to_a5xx_backend(ir3_kernel->backend);
   struct fd_ringbuffer *ring = fd_submit_new_ringbuffer(
      submit, 0,
      (enum fd_ringbuffer_flags)(FD_RINGBUFFER_PRIMARY |
                                 FD_RINGBUFFER_GROWABLE));

   cs_restore_emit(ring, a5xx_backend);
   cs_program_emit(ring, kernel);
   cs_const_emit(ring, kernel, grid);
   cs_ibo_emit(ring, kernel);

   const unsigned *local_size = kernel->local_size;
   const unsigned *num_groups = grid;

   unsigned work_dim = 0;
   for (int i = 0; i < 3; i++) {
      if (!grid[i])
         break;
      work_dim++;
   }

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

   if (a5xx_backend->num_perfcntrs > 0) {
      a5xx_backend->query_mem = fd_bo_new(
         a5xx_backend->dev,
         a5xx_backend->num_perfcntrs * sizeof(struct fd5_query_sample), 0,
         "query");

      /* configure the performance counters to count the requested
       * countables:
       */
      for (unsigned i = 0; i < a5xx_backend->num_perfcntrs; i++) {
         const struct perfcntr *counter = &a5xx_backend->perfcntrs[i];

         OUT_PKT4(ring, counter->select_reg, 1);
         OUT_RING(ring, counter->selector);
      }

      OUT_WFI5(ring);

      /* and snapshot the start values: */
      for (unsigned i = 0; i < a5xx_backend->num_perfcntrs; i++) {
         const struct perfcntr *counter = &a5xx_backend->perfcntrs[i];

         OUT_PKT7(ring, CP_REG_TO_MEM, 3);
         OUT_RING(ring, CP_REG_TO_MEM_0_REG(counter->counter_reg_lo) |
                           CP_REG_TO_MEM_0_IS_64B);
         OUT_RELOC(ring, query_sample_idx(a5xx_backend, i, start));
      }
   }

   OUT_PKT7(ring, CP_EXEC_CS, 4);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, CP_EXEC_CS_1_NGROUPS_X(grid[0]));
   OUT_RING(ring, CP_EXEC_CS_2_NGROUPS_Y(grid[1]));
   OUT_RING(ring, CP_EXEC_CS_3_NGROUPS_Z(grid[2]));

   OUT_WFI5(ring);

   if (a5xx_backend->num_perfcntrs > 0) {
      /* snapshot the end values: */
      for (unsigned i = 0; i < a5xx_backend->num_perfcntrs; i++) {
         const struct perfcntr *counter = &a5xx_backend->perfcntrs[i];

         OUT_PKT7(ring, CP_REG_TO_MEM, 3);
         OUT_RING(ring, CP_REG_TO_MEM_0_REG(counter->counter_reg_lo) |
                           CP_REG_TO_MEM_0_IS_64B);
         OUT_RELOC(ring, query_sample_idx(a5xx_backend, i, stop));
      }

      OUT_PKT7(ring, CP_WAIT_MEM_WRITES, 0);

      /* and compute the result: */
      for (unsigned i = 0; i < a5xx_backend->num_perfcntrs; i++) {
         /* result += stop - start: */
         OUT_PKT7(ring, CP_MEM_TO_MEM, 9);
         OUT_RING(ring, CP_MEM_TO_MEM_0_DOUBLE | CP_MEM_TO_MEM_0_NEG_C);
         OUT_RELOC(ring, query_sample_idx(a5xx_backend, i, result)); /* dst */
         OUT_RELOC(ring, query_sample_idx(a5xx_backend, i, result)); /* srcA */
         OUT_RELOC(ring, query_sample_idx(a5xx_backend, i, stop));   /* srcB */
         OUT_RELOC(ring, query_sample_idx(a5xx_backend, i, start));  /* srcC */
      }
   }

   cache_flush(ring, kernel);
}

static void
a5xx_set_perfcntrs(struct backend *b, const struct perfcntr *perfcntrs,
                   unsigned num_perfcntrs)
{
   struct a5xx_backend *a5xx_backend = to_a5xx_backend(b);

   a5xx_backend->perfcntrs = perfcntrs;
   a5xx_backend->num_perfcntrs = num_perfcntrs;
}

static void
a5xx_read_perfcntrs(struct backend *b, uint64_t *results)
{
   struct a5xx_backend *a5xx_backend = to_a5xx_backend(b);

   fd_bo_cpu_prep(a5xx_backend->query_mem, NULL, FD_BO_PREP_READ);
   struct fd5_query_sample *samples =
      (struct fd5_query_sample *)fd_bo_map(a5xx_backend->query_mem);

   for (unsigned i = 0; i < a5xx_backend->num_perfcntrs; i++) {
      results[i] = samples[i].result;
   }
}

struct backend *
a5xx_init(struct fd_device *dev, const struct fd_dev_id *dev_id)
{
   struct a5xx_backend *a5xx_backend =
      (struct a5xx_backend *)calloc(1, sizeof(*a5xx_backend));

   a5xx_backend->base = (struct backend){
      .assemble = a5xx_assemble,
      .disassemble = a5xx_disassemble,
      .emit_grid = a5xx_emit_grid,
      .set_perfcntrs = a5xx_set_perfcntrs,
      .read_perfcntrs = a5xx_read_perfcntrs,
   };

   struct ir3_compiler_options compiler_options = {
      .disable_cache = true,
   };
   a5xx_backend->compiler = ir3_compiler_create(
      dev, dev_id, fd_dev_info_raw(dev_id), &compiler_options);
   a5xx_backend->dev = dev;

   a5xx_backend->dev_id = dev_id;
   a5xx_backend->info = fd_dev_info_raw(dev_id);

   a5xx_backend->control_mem = fd_bo_new(dev, 0x1000, 0, "control");

   return &a5xx_backend->base;
}
