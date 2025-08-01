/*
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "panvk_utrace.h"

#include "drm-uapi/panthor_drm.h"

#include "genxml/cs_builder.h"
#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_priv_bo.h"

static void
cmd_write_timestamp(const struct panvk_device *dev, struct cs_builder *b,
                    uint64_t addr, uint32_t wait_mask)
{
   const struct cs_index addr_reg = cs_scratch_reg64(b, 0);
   /* abuse DEFERRED_SYNC */
   const struct cs_async_op async = cs_defer(wait_mask, SB_ID(DEFERRED_SYNC));

   cs_move64_to(b, addr_reg, addr);
   cs_store_state(b, addr_reg, 0, MALI_CS_STATE_TIMESTAMP, async);
}

static void
cmd_copy_data(struct cs_builder *b, uint64_t dst_addr, uint64_t src_addr,
              uint32_t size)
{
   assert((dst_addr | src_addr | size) % sizeof(uint32_t) == 0);

   /* wait for timestamp writes */
   cs_wait_slot(b, SB_ID(DEFERRED_SYNC));

   /* Depending on where this is called from, we could potentially use SR
    * registers or copy with a compute job.
    */
   const struct cs_index dst_addr_reg = cs_scratch_reg64(b, 0);
   const struct cs_index src_addr_reg = cs_scratch_reg64(b, 2);
   const uint32_t temp_count = CS_REG_SCRATCH_COUNT - 4;

   while (size) {
      cs_move64_to(b, dst_addr_reg, dst_addr);
      cs_move64_to(b, src_addr_reg, src_addr);

      const uint32_t max_offset = 1 << 16;
      uint32_t copy_count = MIN2(size, max_offset) / sizeof(uint32_t);
      uint32_t offset = 0;
      while (copy_count) {
         const uint32_t count = MIN2(copy_count, temp_count);
         const struct cs_index reg = cs_scratch_reg_tuple(b, 4, count);

         cs_load_to(b, reg, src_addr_reg, BITFIELD_MASK(count), offset);
         cs_wait_slot(b, SB_ID(LS));
         cs_store(b, reg, dst_addr_reg, BITFIELD_MASK(count), offset);

         copy_count -= count;
         offset += count * sizeof(uint32_t);
      }

      dst_addr += offset;
      src_addr += offset;
      size -= offset;
   }

   cs_wait_slot(b, SB_ID(LS));
}

static struct cs_builder *
get_builder(struct panvk_cmd_buffer *cmdbuf, struct u_trace *ut)
{
   const uint32_t subqueue = ut - cmdbuf->utrace.uts;
   assert(subqueue < PANVK_SUBQUEUE_COUNT);

   return panvk_get_cs_builder(cmdbuf, subqueue);
}

static void
panvk_utrace_record_ts(struct u_trace *ut, void *cs, void *timestamps,
                       uint64_t offset_B, uint32_t flags)
{
   /* Here the input type for void *cs is panvk_utrace_cs_info instead of
    * panvk_cmd_buffer so we can pass additional parameters. */
   struct panvk_utrace_cs_info *cs_info = cs;
   struct panvk_cmd_buffer *cmdbuf = cs_info->cmdbuf;
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct cs_builder *b = get_builder(cmdbuf, ut);
   const struct panvk_priv_bo *bo = timestamps;
   const uint64_t addr = bo->addr.dev + offset_B;

   cmd_write_timestamp(dev, b, addr, cs_info->ts_wait_mask);
}

static void
panvk_utrace_capture_data(struct u_trace *ut, void *cs, void *dst_buffer,
                          uint64_t dst_offset_B, void *src_buffer,
                          uint64_t src_offset_B, uint32_t size_B)
{
   /* Here the input type for void *cs is panvk_utrace_cs_info instead of
    * panvk_cmd_buffer so we can pass additional parameters. */
   struct panvk_utrace_cs_info *cs_info = cs;
   struct cs_builder *b = get_builder(cs_info->cmdbuf, ut);
   const struct panvk_priv_bo *dst_bo = dst_buffer;
   const uint64_t dst_addr = dst_bo->addr.dev + dst_offset_B;
   const uint64_t src_addr = src_offset_B;

   /* src_offset_B is absolute */
   assert(!src_buffer);

   cmd_copy_data(b, dst_addr, src_addr, size_B);
}

void
panvk_per_arch(utrace_context_init)(struct panvk_device *dev)
{
   u_trace_context_init(&dev->utrace.utctx, dev, sizeof(uint64_t),
                        sizeof(VkDispatchIndirectCommand),
                        panvk_utrace_create_buffer, panvk_utrace_delete_buffer,
                        panvk_utrace_record_ts, panvk_utrace_read_ts,
                        panvk_utrace_capture_data, panvk_utrace_get_data,
                        panvk_utrace_delete_flush_data);
}

void
panvk_per_arch(utrace_context_fini)(struct panvk_device *dev)
{
   u_trace_context_fini(&dev->utrace.utctx);
}

void
panvk_per_arch(utrace_copy_buffer)(struct u_trace_context *utctx,
                                   void *cmdstream, void *ts_from,
                                   uint64_t from_offset, void *ts_to,
                                   uint64_t to_offset, uint64_t size_B)
{
   struct cs_builder *b = cmdstream;
   const struct panvk_priv_bo *src_bo = ts_from;
   const struct panvk_priv_bo *dst_bo = ts_to;
   const uint64_t src_addr = src_bo->addr.dev + from_offset;
   const uint64_t dst_addr = dst_bo->addr.dev + to_offset;

   cmd_copy_data(b, dst_addr, src_addr, size_B);
}

void
panvk_per_arch(utrace_clone_init_pool)(struct panvk_pool *pool,
                                       struct panvk_device *dev)
{
   const struct panvk_pool_properties pool_props = {
      .slab_size = 64 * 1024,
      .label = "utrace clone pool",
      .owns_bos = true,
   };
   panvk_pool_init(pool, dev, NULL, &pool_props);
}

static struct cs_buffer
alloc_clone_buffer(void *cookie)
{
   struct panvk_pool *pool = cookie;
   const uint32_t size = 4 * 1024;
   const uint32_t alignment = 64;

   struct pan_ptr ptr = pan_pool_alloc_aligned(&pool->base, size, alignment);

   return (struct cs_buffer){
      .cpu = ptr.cpu,
      .gpu = ptr.gpu,
      .capacity = size,
   };
}

void
panvk_per_arch(utrace_clone_init_builder)(struct cs_builder *b,
                                          struct panvk_pool *pool)
{
   const struct drm_panthor_csif_info *csif_info =
      panthor_kmod_get_csif_props(pool->dev->kmod.dev);
   const struct cs_builder_conf builder_conf = {
      .nr_registers = csif_info->cs_reg_count,
      .nr_kernel_registers = MAX2(csif_info->unpreserved_cs_reg_count, 4),
      .alloc_buffer = alloc_clone_buffer,
      .cookie = pool,
      .ls_sb_slot = SB_ID(LS),
   };
   cs_builder_init(b, &builder_conf, (struct cs_buffer){0});
}

void
panvk_per_arch(utrace_clone_finish_builder)(struct cs_builder *b)
{
   const struct cs_index flush_id = cs_scratch_reg32(b, 0);

   cs_move32_to(b, flush_id, 0);
   cs_flush_caches(b, MALI_CS_FLUSH_MODE_CLEAN, MALI_CS_FLUSH_MODE_NONE,
                   MALI_CS_OTHER_FLUSH_MODE_NONE, flush_id,
                   cs_defer(SB_IMM_MASK, SB_ID(IMM_FLUSH)));
   cs_wait_slot(b, SB_ID(IMM_FLUSH));

   cs_finish(b);
}
