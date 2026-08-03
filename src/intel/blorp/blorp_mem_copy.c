/* Copyright © 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "blorp_nir_builder.h"

#include "blorp_priv.h"
#include "blorp_shaders.h"

/* We have 3 types of shaders:
 * - Aligned ones, where both addresses and the size are aligned. These are
 *   the faster shaders.
 * - Semi-aligned ones, where both addresses are aligned, but the size is not.
 *   This allows us to do properly-aligned memory reads and writes, for all
 *   invocations except the last one.
 * - Unaligned ones, where the addresses are not aligned. For these, we use
 *   vload/vstore, which works on unaligned access and may end up being
 *   translated to ugm loads and stores that do support unaligned access as
 *   well. Still, they are a little slower.
 */
enum mem2mem_cl_shader_version {
   MEM2MEM_SHADER_UNALIGNED_1,
   MEM2MEM_SHADER_UNALIGNED_8,
   MEM2MEM_SHADER_UNALIGNED_16,

   MEM2MEM_SHADER_SEMI_ALIGNED_4,
   MEM2MEM_SHADER_SEMI_ALIGNED_8,
   MEM2MEM_SHADER_SEMI_ALIGNED_16,

   MEM2MEM_SHADER_ALIGNED_4,
   MEM2MEM_SHADER_ALIGNED_8,
   MEM2MEM_SHADER_ALIGNED_16,
};

static int
mem2mem_cl_shader_version_get_divisor(enum mem2mem_cl_shader_version version)
{
   switch (version) {
   case MEM2MEM_SHADER_UNALIGNED_1:     return 1;
   case MEM2MEM_SHADER_UNALIGNED_8:     return 8;
   case MEM2MEM_SHADER_UNALIGNED_16:    return 16;

   case MEM2MEM_SHADER_SEMI_ALIGNED_4:  return 8; /* '8' is not a typo. */
   case MEM2MEM_SHADER_SEMI_ALIGNED_8:  return 8;
   case MEM2MEM_SHADER_SEMI_ALIGNED_16: return 16;

   case MEM2MEM_SHADER_ALIGNED_4:       return 8; /* '8' is not a typo. */
   case MEM2MEM_SHADER_ALIGNED_8:       return 8;
   case MEM2MEM_SHADER_ALIGNED_16:      return 16;
   default:
      UNREACHABLE("missing cl_shader_version case");
   }
}

#pragma pack(push, 1)
struct blorp_copy_mem2mem_key {
   struct blorp_base_key base;

   enum mem2mem_cl_shader_version cl_shader_version;
};
#pragma pack(pop)

/* Refer to struct blorp_wm_inputs_mem_copy. */
struct blorp_copy_mem2mem_vars {
   nir_variable *src_addr;
   nir_variable *dst_addr;
   nir_variable *size;
   nir_variable *width; /* Only used in the FS version. */
};

static void
blorp_copy_mem2mem_vars_init(nir_builder *b,
                             struct blorp_copy_mem2mem_vars *v)
{
   v->src_addr =
      BLORP_CREATE_NIR_INPUT(b->shader, mem_copy.src_addr,
                             glsl_uint64_t_type());
   v->dst_addr =
      BLORP_CREATE_NIR_INPUT(b->shader, mem_copy.dst_addr,
                             glsl_uint64_t_type());
   v->size =
      BLORP_CREATE_NIR_INPUT(b->shader, mem_copy.size,
                             glsl_uint64_t_type());
   v->width =
      BLORP_CREATE_NIR_INPUT(b->shader, mem_copy.width,
                             glsl_uint_type());
}

static nir_shader *
blorp_build_copy_mem2mem_shader(struct blorp_batch *batch, void *mem_ctx,
                                const struct blorp_copy_mem2mem_key *key)
{
   struct blorp_context *blorp = batch->blorp;
   bool is_compute =
      key->base.shader_pipeline == BLORP_SHADER_PIPELINE_COMPUTE;

   mesa_shader_stage stage = is_compute ?
      MESA_SHADER_COMPUTE : MESA_SHADER_FRAGMENT;

   const nir_shader_compiler_options *nir_options =
      blorp->compiler->nir_options(blorp, stage);
   nir_builder b = nir_builder_init_simple_shader(stage, nir_options,
                                                  "copy mem2mem");
   ralloc_steal(mem_ctx, b.shader);

   if (is_compute) {
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
   }

   struct blorp_copy_mem2mem_vars v;
   blorp_copy_mem2mem_vars_init(&b, &v);

   nir_def *src = nir_load_var(&b, v.src_addr);
   nir_def *dst = nir_load_var(&b, v.dst_addr);
   nir_def *size = nir_load_var(&b, v.size);

   nir_def *global_id;
   if (is_compute) {
      global_id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   } else {
      nir_def *width = nir_load_var(&b, v.width);
      nir_def *pos_in = nir_f2i32(&b, nir_build_frag_coord(&b, 2));
      global_id = nir_iadd(&b,
                           nir_imul(&b, nir_channel(&b, pos_in, 1), width),
                           nir_channel(&b, pos_in, 0));
   }

   switch (key->cl_shader_version) {
   case MEM2MEM_SHADER_UNALIGNED_1:
      blorp_copy_mem2mem_shader_aligned_uchar(&b, src, dst, size, global_id);
      break;
   case MEM2MEM_SHADER_UNALIGNED_8:
      blorp_copy_mem2mem_shader_unaligned_8(&b, src, dst, size, global_id);
      break;
   case MEM2MEM_SHADER_UNALIGNED_16:
      blorp_copy_mem2mem_shader_unaligned_16(&b, src, dst, size, global_id);
      break;
   case MEM2MEM_SHADER_SEMI_ALIGNED_4:
      blorp_copy_mem2mem_shader_semi_aligned_uint(&b, src, dst, size, global_id);
      break;
   case MEM2MEM_SHADER_SEMI_ALIGNED_8:
      blorp_copy_mem2mem_shader_semi_aligned_ulong(&b, src, dst, size, global_id);
      break;
   case MEM2MEM_SHADER_SEMI_ALIGNED_16:
      blorp_copy_mem2mem_shader_semi_aligned_ulong2(&b, src, dst, size, global_id);
      break;
   case MEM2MEM_SHADER_ALIGNED_4:
      blorp_copy_mem2mem_shader_aligned_uint(&b, src, dst, size, global_id);
      break;
   case MEM2MEM_SHADER_ALIGNED_8:
      blorp_copy_mem2mem_shader_aligned_ulong(&b, src, dst, size, global_id);
      break;
   case MEM2MEM_SHADER_ALIGNED_16:
      blorp_copy_mem2mem_shader_aligned_ulong2(&b, src, dst, size, global_id);
      break;
   default:
      UNREACHABLE("invalid cl_shader_version");
      break;
   }

   return b.shader;
}

static bool
blorp_get_copy_mem2mem_kernel_cs(struct blorp_batch *batch,
                                 struct blorp_params *params,
                                 const struct blorp_copy_mem2mem_key *key)
{
   struct blorp_context *blorp = batch->blorp;

   if (blorp->lookup_shader(batch, key, sizeof(*key), &params->cs_prog_kernel,
                            &params->cs_prog_data)) {
      return true;
   }

   void *mem_ctx = ralloc_context(NULL);

   nir_shader *nir =
      blorp_build_copy_mem2mem_shader(batch, mem_ctx, key);

   const struct blorp_program prog =
      blorp_compile_cs(blorp, mem_ctx, nir, key, sizeof(*key));

   bool result = blorp->upload_shader(batch, MESA_SHADER_COMPUTE,
                                      key, sizeof(*key),
                                      prog.kernel, prog.kernel_size,
                                      prog.prog_data, prog.prog_data_size,
                                      &params->cs_prog_kernel,
                                      &params->cs_prog_data);
   assert(result);

   ralloc_free(mem_ctx);
   return result;
}

static bool
blorp_get_copy_mem2mem_kernel_fs(struct blorp_batch *batch,
                                 struct blorp_params *params,
                                 const struct blorp_copy_mem2mem_key *key)
{
   struct blorp_context *blorp = batch->blorp;

   if (blorp->lookup_shader(batch, key, sizeof(*key), &params->wm_prog_kernel,
                            &params->fs_prog_data)) {
      return true;
   }

   void *mem_ctx = ralloc_context(NULL);

   nir_shader *nir =
      blorp_build_copy_mem2mem_shader(batch, mem_ctx, key);

   const struct blorp_program prog =
      blorp_compile_fs(blorp, mem_ctx, nir, false, false, false, key,
                       sizeof(*key));

   bool result = blorp->upload_shader(batch, MESA_SHADER_FRAGMENT,
                                      key, sizeof(*key),
                                      prog.kernel, prog.kernel_size,
                                      prog.prog_data, prog.prog_data_size,
                                      &params->wm_prog_kernel,
                                      &params->fs_prog_data);
   assert(result);

   ralloc_free(mem_ctx);
   return result;
}

static enum mem2mem_cl_shader_version
mem2mem_pick_shader_version(uint64_t src_addr, uint64_t dst_addr,
                            uint64_t size, bool is_compute,
                            unsigned max_img_dimension)
{
   /* The "MIN2(x, 6)" is here to prevent overflow/ub in the shifting in case
    * ffsll returns a number bigger than 31. So the max aligment we'll ever
    * get here is 2^6 (64).
    */
   unsigned int src_addr_alignment = 1 << MIN2((ffsll(src_addr) - 1), 6);
   unsigned int dst_addr_alignment = 1 << MIN2((ffsll(dst_addr) - 1), 6);
   unsigned int addr_alignment = MIN2(src_addr_alignment, dst_addr_alignment);
   unsigned int size_alignment = 1 << MIN2((ffsll(size) - 1), 6);
   unsigned int min_alignment = MIN2(addr_alignment, size_alignment);

   if (!is_compute && size > max_img_dimension * max_img_dimension) {
      /* Fragment shaders have limited maximum dimensions. If we exceed that,
       * we have to write multiple bytes per invocation. At most, we'll write
       * 16 bytes (16384 * 16484 * 16 == 2^32) for the platforms we currently
       * support.
       */
      assert(max_img_dimension >= 16384);
      if (min_alignment >= 16)
         return MEM2MEM_SHADER_ALIGNED_16;
      else if (addr_alignment >= 16)
         return MEM2MEM_SHADER_SEMI_ALIGNED_16;
      return MEM2MEM_SHADER_UNALIGNED_16;

   } else if (min_alignment >= 16) {
      return MEM2MEM_SHADER_ALIGNED_16;
   } else if (min_alignment >= 8) {
      return MEM2MEM_SHADER_ALIGNED_8;
   } else if (min_alignment >= 4) {
      return MEM2MEM_SHADER_ALIGNED_4;
   } else if (addr_alignment >= 16) {
      return MEM2MEM_SHADER_SEMI_ALIGNED_16;
   } else if (addr_alignment >= 8) {
      return MEM2MEM_SHADER_SEMI_ALIGNED_8;
   } else if (addr_alignment >= 4) {
      return MEM2MEM_SHADER_SEMI_ALIGNED_4;
   } else if (size <= 32) {
      return MEM2MEM_SHADER_UNALIGNED_1;
   } else {
      return MEM2MEM_SHADER_UNALIGNED_8;
   }
}

void
blorp_mem_copy(struct blorp_batch *batch,
               struct blorp_address src,
               struct blorp_address dst,
               uint64_t size)
{
   const struct isl_device *isl_dev = batch->blorp->isl_dev;

   /* We use global pointers pointers for accessing buffers, which get
    * translated into A64 messages (Gfx8+), and the ELK side of blorp was not
    * wired with the changes we need for this type of operation (Gfx8-).
    */
   assert(isl_dev->info->ver >= 9);

   /* We could add a blitter path for this, but it would be mostly the same
    * thing as what blorp_blit already does, so just defer to it.
    */
   if (batch->flags & BLORP_BATCH_USE_BLITTER)
      return blorp_blit_buffer_copy(batch, src, dst, size);

   uint64_t src_addr = batch->blorp->get_surface_address(batch, src);
   uint64_t dst_addr = batch->blorp->get_surface_address(batch, dst);

   /* Should match Vulkan's maxImageDimension2D. */
   const unsigned max_img_dimension = 16384;

   /* Just sanity checks. */
   assert(src_addr != 0 && dst_addr != 0 && size != 0);
   assert(size <= isl_dev->max_buffer_size);
   /* If we ever have to support bigger buffers, we'll have to adjust the
    * math that sets params.x1/y1 and everything that depends on that.
    */
   assert(size <= (1ULL << 32));

   bool is_compute = batch->flags & BLORP_BATCH_USE_COMPUTE;
   enum blorp_shader_pipeline blorp_pipeline = is_compute ?
      BLORP_SHADER_PIPELINE_COMPUTE : BLORP_SHADER_PIPELINE_RENDER;

   struct blorp_copy_mem2mem_key key;
   BLORP_KEY_INIT(key, batch->blorp, BLORP_SHADER_TYPE_MEM_COPY,
                  blorp_pipeline);

   key.cl_shader_version =
      mem2mem_pick_shader_version(src_addr, dst_addr, size, is_compute,
                                  max_img_dimension);

   struct blorp_params params;
   blorp_params_init(&params, batch->blorp);

   params.op = BLORP_OP_MEM_COPY;
   params.shader_type = BLORP_SHADER_TYPE_MEM_COPY;
   params.shader_pipeline = blorp_pipeline;

   params.wm_inputs.mem_copy.src_addr = src_addr;
   params.wm_inputs.mem_copy.dst_addr = dst_addr;
   params.wm_inputs.mem_copy.size = size;

   params.x0 = 0;
   params.y0 = 0;

   int divisor =
      mem2mem_cl_shader_version_get_divisor(key.cl_shader_version);
   uint32_t invocations = DIV_ROUND_UP(size, divisor);
   if (is_compute || invocations < max_img_dimension) {
      params.x1 = invocations;
      params.y1 = 1;
   } else {
      /* Here we want to break down the 1D size into a 2D shape that fits. We
       * want to do this in a way that minimizes the number of EUs that do
       * nothing.
       *
       * Pixels are dispatched as 2x2 quads. Performance is worse when
       * params.x1 is not a multiple of the divisor.
       */
      const int x1_step = divisor;
      int x1_ = MAX2(ROUND_DOWN_TO(invocations / max_img_dimension, x1_step),
                     x1_step);
      int y1_ = DIV_ROUND_UP(invocations, x1_);

      while (y1_ > max_img_dimension) {
         x1_ += x1_step;
         y1_ = DIV_ROUND_UP(invocations, x1_);
      }

      params.x1 = x1_;
      params.y1 = y1_;

      assert(params.x1 <= max_img_dimension);
      assert(params.y1 <= max_img_dimension);
      assert((uint64_t)params.x1 * params.y1 >= invocations);
   }
   params.wm_inputs.mem_copy.width = params.x1;

   if (is_compute) {
      if (!blorp_get_copy_mem2mem_kernel_cs(batch, &params, &key)) {
         mesa_loge("%s: failed to get CS kernel", __func__);
         return;
      }
   } else {
      if (!blorp_get_copy_mem2mem_kernel_fs(batch, &params, &key)) {
         mesa_loge("%s: failed to get FS kernel", __func__);
         return;
      }
   }

   batch->blorp->exec(batch, &params);
}
