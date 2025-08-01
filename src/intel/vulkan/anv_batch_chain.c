/*
 * Copyright © 2015 Intel Corporation
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

#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include <xf86drm.h>

#include "anv_private.h"
#include "anv_measure.h"

#include "common/intel_debug_identifier.h"

#include "genxml/gen90_pack.h"
#include "genxml/genX_bits.h"

#include "util/perf/u_trace.h"

/** \file anv_batch_chain.c
 *
 * This file contains functions related to anv_cmd_buffer as a data
 * structure.  This involves everything required to create and destroy
 * the actual batch buffers as well as link them together.
 *
 * It specifically does *not* contain any handling of actual vkCmd calls
 * beyond vkCmdExecuteCommands.
 */

/*-----------------------------------------------------------------------*
 * Functions related to anv_reloc_list
 *-----------------------------------------------------------------------*/

VkResult
anv_reloc_list_init(struct anv_reloc_list *list,
                    const VkAllocationCallbacks *alloc,
                    bool uses_relocs)
{
   assert(alloc != NULL);
   memset(list, 0, sizeof(*list));
   list->uses_relocs = uses_relocs;
   list->alloc = alloc;
   return VK_SUCCESS;
}

static VkResult
anv_reloc_list_init_clone(struct anv_reloc_list *list,
                          const struct anv_reloc_list *other_list)
{
   list->dep_words = other_list->dep_words;

   if (list->dep_words > 0) {
      list->deps =
         vk_alloc(list->alloc, list->dep_words * sizeof(BITSET_WORD), 8,
                  VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      memcpy(list->deps, other_list->deps,
             list->dep_words * sizeof(BITSET_WORD));
   } else {
      list->deps = NULL;
   }

   return VK_SUCCESS;
}

void
anv_reloc_list_finish(struct anv_reloc_list *list)
{
   vk_free(list->alloc, list->deps);
}

static VkResult
anv_reloc_list_grow_deps(struct anv_reloc_list *list,
                         uint32_t min_num_words)
{
   if (min_num_words <= list->dep_words)
      return VK_SUCCESS;

   uint32_t new_length = MAX2(32, list->dep_words * 2);
   while (new_length < min_num_words)
      new_length *= 2;

   BITSET_WORD *new_deps =
      vk_realloc(list->alloc, list->deps, new_length * sizeof(BITSET_WORD), 8,
                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (new_deps == NULL)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   list->deps = new_deps;

   /* Zero out the new data */
   memset(list->deps + list->dep_words, 0,
          (new_length - list->dep_words) * sizeof(BITSET_WORD));
   list->dep_words = new_length;

   return VK_SUCCESS;
}

VkResult
anv_reloc_list_add_bo_impl(struct anv_reloc_list *list,
                           struct anv_bo *target_bo)
{
   /* This can happen with sparse resources. */
   if (!target_bo)
      return VK_SUCCESS;

   uint32_t idx = target_bo->gem_handle;
   VkResult result = anv_reloc_list_grow_deps(list,
                                              (idx / BITSET_WORDBITS) + 1);
   if (unlikely(result != VK_SUCCESS))
      return result;

   BITSET_SET(list->deps, idx);

   return VK_SUCCESS;
}

static void
anv_reloc_list_clear(struct anv_reloc_list *list)
{
   if (list->dep_words > 0)
      memset(list->deps, 0, list->dep_words * sizeof(BITSET_WORD));
}

VkResult
anv_reloc_list_append(struct anv_reloc_list *list,
                      struct anv_reloc_list *other)
{
   anv_reloc_list_grow_deps(list, other->dep_words);
   for (uint32_t w = 0; w < other->dep_words; w++)
      list->deps[w] |= other->deps[w];

   return VK_SUCCESS;
}

/*-----------------------------------------------------------------------*
 * Functions related to anv_batch
 *-----------------------------------------------------------------------*/

static VkResult
anv_extend_batch(struct anv_batch *batch, uint32_t size)
{
   assert(batch->extend_cb != NULL);
   VkResult result = batch->extend_cb(batch, size, batch->user_data);
   if (result != VK_SUCCESS)
      return anv_batch_set_error(batch, result);
   return result;
}

void *
anv_batch_emit_dwords(struct anv_batch *batch, int num_dwords)
{
   uint32_t size = num_dwords * 4;
   if (batch->next + size > batch->end) {
      if (anv_extend_batch(batch, size) != VK_SUCCESS)
         return NULL;
   }

   void *p = batch->next;

   batch->next += num_dwords * 4;
   assert(batch->next <= batch->end);

   return p;
}

/* Ensure enough contiguous space is available */
VkResult
anv_batch_emit_ensure_space(struct anv_batch *batch, uint32_t size)
{
   if (batch->next + size > batch->end) {
      VkResult result = anv_extend_batch(batch, size);
      if (result != VK_SUCCESS)
         return result;
   }

   assert(batch->next + size <= batch->end);

   return VK_SUCCESS;
}

void
anv_batch_advance(struct anv_batch *batch, uint32_t size)
{
   assert(batch->next + size <= batch->end);

   batch->next += size;
}

struct anv_address
anv_batch_address(struct anv_batch *batch, void *batch_location)
{
   assert(batch->start <= batch_location);

   /* Allow a jump at the current location of the batch. */
   assert(batch->next >= batch_location);

   return anv_address_add(batch->start_addr, batch_location - batch->start);
}

void
anv_batch_emit_batch(struct anv_batch *batch, struct anv_batch *other)
{
   uint32_t size = other->next - other->start;
   assert(size % 4 == 0);

   if (batch->next + size > batch->end) {
      if (anv_extend_batch(batch, size) != VK_SUCCESS)
         return;
   }

   assert(batch->next + size <= batch->end);

   VG(VALGRIND_CHECK_MEM_IS_DEFINED(other->start, size));
   memcpy(batch->next, other->start, size);

   VkResult result = anv_reloc_list_append(batch->relocs, other->relocs);
   if (result != VK_SUCCESS) {
      anv_batch_set_error(batch, result);
      return;
   }

   batch->next += size;
}

/*-----------------------------------------------------------------------*
 * Functions related to anv_batch_bo
 *-----------------------------------------------------------------------*/

static VkResult
anv_batch_bo_create(struct anv_cmd_buffer *cmd_buffer,
                    uint32_t size,
                    struct anv_batch_bo **bbo_out)
{
   VkResult result;

   struct anv_batch_bo *bbo = vk_zalloc(&cmd_buffer->vk.pool->alloc, sizeof(*bbo),
                                        8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bbo == NULL)
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = anv_bo_pool_alloc(&cmd_buffer->device->batch_bo_pool,
                              size, &bbo->bo);
   ANV_DMR_BO_ALLOC(&cmd_buffer->vk.base, bbo->bo, result);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   const bool uses_relocs = cmd_buffer->device->physical->uses_relocs;
   result = anv_reloc_list_init(&bbo->relocs, &cmd_buffer->vk.pool->alloc, uses_relocs);
   if (result != VK_SUCCESS)
      goto fail_bo_alloc;

   *bbo_out = bbo;

   return VK_SUCCESS;

 fail_bo_alloc:
   ANV_DMR_BO_FREE(&cmd_buffer->vk.base, bbo->bo);
   anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool, bbo->bo);
 fail_alloc:
   vk_free(&cmd_buffer->vk.pool->alloc, bbo);

   return result;
}

static VkResult
anv_batch_bo_clone(struct anv_cmd_buffer *cmd_buffer,
                   const struct anv_batch_bo *other_bbo,
                   struct anv_batch_bo **bbo_out)
{
   VkResult result;

   struct anv_batch_bo *bbo = vk_alloc(&cmd_buffer->vk.pool->alloc, sizeof(*bbo),
                                        8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bbo == NULL)
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = anv_bo_pool_alloc(&cmd_buffer->device->batch_bo_pool,
                              other_bbo->bo->size, &bbo->bo);
   ANV_DMR_BO_ALLOC(&cmd_buffer->vk.base, bbo->bo, result);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   result = anv_reloc_list_init_clone(&bbo->relocs, &other_bbo->relocs);
   if (result != VK_SUCCESS)
      goto fail_bo_alloc;

   bbo->length = other_bbo->length;
   memcpy(bbo->bo->map, other_bbo->bo->map, other_bbo->length);
   *bbo_out = bbo;

   return VK_SUCCESS;

 fail_bo_alloc:
   ANV_DMR_BO_FREE(&cmd_buffer->vk.base, bbo->bo);
   anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool, bbo->bo);

 fail_alloc:
   vk_free(&cmd_buffer->vk.pool->alloc, bbo);

   return result;
}

static void
anv_batch_bo_start(struct anv_batch_bo *bbo, struct anv_batch *batch,
                   size_t batch_padding)
{
   anv_batch_set_storage(batch, (struct anv_address) { .bo = bbo->bo, },
                         bbo->bo->map, bbo->bo->size - batch_padding);
   batch->relocs = &bbo->relocs;
   anv_reloc_list_clear(&bbo->relocs);
}

static void
anv_batch_bo_continue(struct anv_batch_bo *bbo, struct anv_batch *batch,
                      size_t batch_padding)
{
   batch->start_addr = (struct anv_address) { .bo = bbo->bo, };
   batch->start = bbo->bo->map;
   batch->next = bbo->bo->map + bbo->length;
   batch->end = bbo->bo->map + bbo->bo->size - batch_padding;
   batch->relocs = &bbo->relocs;
}

static void
anv_batch_bo_finish(struct anv_batch_bo *bbo, struct anv_batch *batch)
{
   assert(batch->start == bbo->bo->map);
   bbo->length = batch->next - batch->start;
   VG(VALGRIND_CHECK_MEM_IS_DEFINED(batch->start, bbo->length));
}

static void
anv_batch_bo_link(struct anv_cmd_buffer *cmd_buffer,
                  struct anv_batch_bo *prev_bbo,
                  struct anv_batch_bo *next_bbo,
                  uint32_t next_bbo_offset)
{
   const uint32_t bb_start_offset =
      prev_bbo->length - GFX9_MI_BATCH_BUFFER_START_length * 4;
   ASSERTED const uint32_t *bb_start = prev_bbo->bo->map + bb_start_offset;

   /* Make sure we're looking at a MI_BATCH_BUFFER_START */
   assert(((*bb_start >> 29) & 0x07) == 0);
   assert(((*bb_start >> 23) & 0x3f) == 49);

   uint64_t *map = prev_bbo->bo->map + bb_start_offset + 4;
   *map = intel_canonical_address(next_bbo->bo->offset + next_bbo_offset);

#ifdef SUPPORT_INTEL_INTEGRATED_GPUS
   if (cmd_buffer->device->physical->memory.need_flush &&
       anv_bo_needs_host_cache_flush(prev_bbo->bo->alloc_flags))
      intel_flush_range(map, sizeof(uint64_t));
#endif
}

static void
anv_batch_bo_destroy(struct anv_batch_bo *bbo,
                     struct anv_cmd_buffer *cmd_buffer)
{
   anv_reloc_list_finish(&bbo->relocs);
   ANV_DMR_BO_FREE(&cmd_buffer->vk.base, bbo->bo);
   anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool, bbo->bo);
   vk_free(&cmd_buffer->vk.pool->alloc, bbo);
}

static VkResult
anv_batch_bo_list_clone(const struct list_head *list,
                        struct anv_cmd_buffer *cmd_buffer,
                        struct list_head *new_list)
{
   VkResult result = VK_SUCCESS;

   list_inithead(new_list);

   struct anv_batch_bo *prev_bbo = NULL;
   list_for_each_entry(struct anv_batch_bo, bbo, list, link) {
      struct anv_batch_bo *new_bbo = NULL;
      result = anv_batch_bo_clone(cmd_buffer, bbo, &new_bbo);
      if (result != VK_SUCCESS)
         break;
      list_addtail(&new_bbo->link, new_list);

      if (prev_bbo)
         anv_batch_bo_link(cmd_buffer, prev_bbo, new_bbo, 0);

      prev_bbo = new_bbo;
   }

   if (result != VK_SUCCESS) {
      list_for_each_entry_safe(struct anv_batch_bo, bbo, new_list, link) {
         list_del(&bbo->link);
         anv_batch_bo_destroy(bbo, cmd_buffer);
      }
   }

   return result;
}

/*-----------------------------------------------------------------------*
 * Functions related to anv_batch_bo
 *-----------------------------------------------------------------------*/

static struct anv_batch_bo *
anv_cmd_buffer_current_batch_bo(struct anv_cmd_buffer *cmd_buffer)
{
   return list_entry(cmd_buffer->batch_bos.prev, struct anv_batch_bo, link);
}

static struct anv_batch_bo *
anv_cmd_buffer_current_generation_batch_bo(struct anv_cmd_buffer *cmd_buffer)
{
   return list_entry(cmd_buffer->generation.batch_bos.prev, struct anv_batch_bo, link);
}

struct anv_address
anv_cmd_buffer_surface_base_address(struct anv_cmd_buffer *cmd_buffer)
{
   /* Only graphics & compute queues need binding tables. */
   if (!(cmd_buffer->queue_family->queueFlags & (VK_QUEUE_GRAPHICS_BIT |
                                                 VK_QUEUE_COMPUTE_BIT)))
      return ANV_NULL_ADDRESS;

   /* If we've never allocated a binding table block, do it now. Otherwise we
    * would trigger another STATE_BASE_ADDRESS emission which would require an
    * additional bunch of flushes/stalls.
    */
   if (u_vector_length(&cmd_buffer->bt_block_states) == 0) {
      VkResult result = anv_cmd_buffer_new_binding_table_block(cmd_buffer);
      if (result != VK_SUCCESS) {
         anv_batch_set_error(&cmd_buffer->batch, result);
         return ANV_NULL_ADDRESS;
      }
   }

   struct anv_state_pool *pool = &cmd_buffer->device->binding_table_pool;
   struct anv_state *bt_block = u_vector_head(&cmd_buffer->bt_block_states);
   return (struct anv_address) {
      .bo = pool->block_pool.bo,
      .offset = bt_block->offset - pool->start_offset,
   };
}

static void
emit_batch_buffer_start(struct anv_batch *batch,
                        struct anv_bo *bo, uint32_t offset)
{
   anv_batch_emit(batch, GFX9_MI_BATCH_BUFFER_START, bbs) {
      bbs.DWordLength               = GFX9_MI_BATCH_BUFFER_START_length -
                                      GFX9_MI_BATCH_BUFFER_START_length_bias;
      bbs.SecondLevelBatchBuffer    = Firstlevelbatch;
      bbs.AddressSpaceIndicator     = ASI_PPGTT;
      bbs.BatchBufferStartAddress   = (struct anv_address) { bo, offset };
   }
}

enum anv_cmd_buffer_batch {
   ANV_CMD_BUFFER_BATCH_MAIN,
   ANV_CMD_BUFFER_BATCH_GENERATION,
};

static void
cmd_buffer_chain_to_batch_bo(struct anv_cmd_buffer *cmd_buffer,
                             struct anv_batch_bo *bbo,
                             enum anv_cmd_buffer_batch batch_type)
{
   struct anv_batch *batch =
      batch_type == ANV_CMD_BUFFER_BATCH_GENERATION ?
      &cmd_buffer->generation.batch : &cmd_buffer->batch;
   struct anv_batch_bo *current_bbo =
      batch_type == ANV_CMD_BUFFER_BATCH_GENERATION ?
      anv_cmd_buffer_current_generation_batch_bo(cmd_buffer) :
      anv_cmd_buffer_current_batch_bo(cmd_buffer);

   /* We set the end of the batch a little short so we would be sure we
    * have room for the chaining command.  Since we're about to emit the
    * chaining command, let's set it back where it should go.
    */
   batch->end += GFX9_MI_BATCH_BUFFER_START_length * 4;
   assert(batch->end == current_bbo->bo->map + current_bbo->bo->size);

   emit_batch_buffer_start(batch, bbo->bo, 0);

   anv_batch_bo_finish(current_bbo, batch);

   /* Add the current amount of data written in the current_bbo to the command
    * buffer.
    */
   cmd_buffer->total_batch_size += current_bbo->length;
}

static void
anv_cmd_buffer_record_chain_submit(struct anv_cmd_buffer *cmd_buffer_from,
                                   struct anv_cmd_buffer *cmd_buffer_to)
{
   uint32_t *bb_start = cmd_buffer_from->batch_end;

   struct anv_batch_bo *last_bbo =
      list_last_entry(&cmd_buffer_from->batch_bos, struct anv_batch_bo, link);
   struct anv_batch_bo *first_bbo =
      list_first_entry(&cmd_buffer_to->batch_bos, struct anv_batch_bo, link);

   struct GFX9_MI_BATCH_BUFFER_START gen_bb_start = {
      __anv_cmd_header(GFX9_MI_BATCH_BUFFER_START),
      .SecondLevelBatchBuffer    = Firstlevelbatch,
      .AddressSpaceIndicator     = ASI_PPGTT,
      .BatchBufferStartAddress   = (struct anv_address) { first_bbo->bo, 0 },
   };
   struct anv_batch local_batch = {
      .start  = last_bbo->bo->map,
      .end    = last_bbo->bo->map + last_bbo->bo->size,
      .relocs = &last_bbo->relocs,
      .alloc  = &cmd_buffer_from->vk.pool->alloc,
   };

   __anv_cmd_pack(GFX9_MI_BATCH_BUFFER_START)(&local_batch, bb_start, &gen_bb_start);

   last_bbo->chained = true;
}

static void
anv_cmd_buffer_record_end_submit(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_batch_bo *last_bbo =
      list_last_entry(&cmd_buffer->batch_bos, struct anv_batch_bo, link);
   last_bbo->chained = false;

   uint32_t *batch = cmd_buffer->batch_end;
   anv_pack_struct(batch, GFX9_MI_BATCH_BUFFER_END,
                   __anv_cmd_header(GFX9_MI_BATCH_BUFFER_END));
}

static VkResult
anv_cmd_buffer_chain_batch(struct anv_batch *batch, uint32_t size, void *_data)
{
   /* The caller should not need that much space. Otherwise it should split
    * its commands.
    */
   assert(size <= ANV_MAX_CMD_BUFFER_BATCH_SIZE);

   struct anv_cmd_buffer *cmd_buffer = _data;
   struct anv_batch_bo *new_bbo = NULL;
   /* Amount of reserved space at the end of the batch to account for the
    * chaining instruction.
    */
   const uint32_t batch_padding = GFX9_MI_BATCH_BUFFER_START_length * 4;
   /* Cap reallocation to chunk. */
   uint32_t alloc_size = MIN2(
      MAX2(batch->allocated_batch_size, size + batch_padding),
      ANV_MAX_CMD_BUFFER_BATCH_SIZE);

   VkResult result = anv_batch_bo_create(cmd_buffer, alloc_size, &new_bbo);
   if (result != VK_SUCCESS)
      return result;

   batch->allocated_batch_size += alloc_size;

   struct anv_batch_bo **seen_bbo = u_vector_add(&cmd_buffer->seen_bbos);
   if (seen_bbo == NULL) {
      anv_batch_bo_destroy(new_bbo, cmd_buffer);
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   *seen_bbo = new_bbo;

   cmd_buffer_chain_to_batch_bo(cmd_buffer, new_bbo, ANV_CMD_BUFFER_BATCH_MAIN);

   list_addtail(&new_bbo->link, &cmd_buffer->batch_bos);

   anv_batch_bo_start(new_bbo, batch, batch_padding);

   return VK_SUCCESS;
}

static VkResult
anv_cmd_buffer_chain_generation_batch(struct anv_batch *batch, uint32_t size, void *_data)
{
   /* The caller should not need that much space. Otherwise it should split
    * its commands.
    */
   assert(size <= ANV_MAX_CMD_BUFFER_BATCH_SIZE);

   struct anv_cmd_buffer *cmd_buffer = _data;
   struct anv_batch_bo *new_bbo = NULL;
   /* Cap reallocation to chunk. */
   uint32_t alloc_size = MIN2(
      MAX2(batch->allocated_batch_size, size),
      ANV_MAX_CMD_BUFFER_BATCH_SIZE);

   VkResult result = anv_batch_bo_create(cmd_buffer, alloc_size, &new_bbo);
   if (result != VK_SUCCESS)
      return result;

   batch->allocated_batch_size += alloc_size;

   struct anv_batch_bo **seen_bbo = u_vector_add(&cmd_buffer->seen_bbos);
   if (seen_bbo == NULL) {
      anv_batch_bo_destroy(new_bbo, cmd_buffer);
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   *seen_bbo = new_bbo;

   if (!list_is_empty(&cmd_buffer->generation.batch_bos)) {
      cmd_buffer_chain_to_batch_bo(cmd_buffer, new_bbo,
                                   ANV_CMD_BUFFER_BATCH_GENERATION);
   }

   list_addtail(&new_bbo->link, &cmd_buffer->generation.batch_bos);

   anv_batch_bo_start(new_bbo, batch, GFX9_MI_BATCH_BUFFER_START_length * 4);

   return VK_SUCCESS;
}

/** Allocate a binding table
 *
 * This function allocates a binding table.  This is a bit more complicated
 * than one would think due to a combination of Vulkan driver design and some
 * unfortunate hardware restrictions.
 *
 * The 3DSTATE_BINDING_TABLE_POINTERS_* packets only have a 16-bit field for
 * the binding table pointer which means that all binding tables need to live
 * in the bottom 64k of surface state base address.  The way the GL driver has
 * classically dealt with this restriction is to emit all surface states
 * on-the-fly into the batch and have a batch buffer smaller than 64k.  This
 * isn't really an option in Vulkan for a couple of reasons:
 *
 *  1) In Vulkan, we have growing (or chaining) batches so surface states have
 *     to live in their own buffer and we have to be able to re-emit
 *     STATE_BASE_ADDRESS as needed which requires a full pipeline stall.  In
 *     order to avoid emitting STATE_BASE_ADDRESS any more often than needed
 *     (it's not that hard to hit 64k of just binding tables), we allocate
 *     surface state objects up-front when VkImageView is created.  In order
 *     for this to work, surface state objects need to be allocated from a
 *     global buffer.
 *
 *  2) We tried to design the surface state system in such a way that it's
 *     already ready for bindless texturing.  The way bindless texturing works
 *     on our hardware is that you have a big pool of surface state objects
 *     (with its own state base address) and the bindless handles are simply
 *     offsets into that pool.  With the architecture we chose, we already
 *     have that pool and it's exactly the same pool that we use for regular
 *     surface states so we should already be ready for bindless.
 *
 *  3) For render targets, we need to be able to fill out the surface states
 *     later in vkBeginRenderPass so that we can assign clear colors
 *     correctly.  One way to do this would be to just create the surface
 *     state data and then repeatedly copy it into the surface state BO every
 *     time we have to re-emit STATE_BASE_ADDRESS.  While this works, it's
 *     rather annoying and just being able to allocate them up-front and
 *     re-use them for the entire render pass.
 *
 * While none of these are technically blockers for emitting state on the fly
 * like we do in GL, the ability to have a single surface state pool is
 * simplifies things greatly.  Unfortunately, it comes at a cost...
 *
 * Because of the 64k limitation of 3DSTATE_BINDING_TABLE_POINTERS_*, we can't
 * place the binding tables just anywhere in surface state base address.
 * Because 64k isn't a whole lot of space, we can't simply restrict the
 * surface state buffer to 64k, we have to be more clever.  The solution we've
 * chosen is to have a block pool with a maximum size of 2G that starts at
 * zero and grows in both directions.  All surface states are allocated from
 * the top of the pool (positive offsets) and we allocate blocks (< 64k) of
 * binding tables from the bottom of the pool (negative offsets).  Every time
 * we allocate a new binding table block, we set surface state base address to
 * point to the bottom of the binding table block.  This way all of the
 * binding tables in the block are in the bottom 64k of surface state base
 * address.  When we fill out the binding table, we add the distance between
 * the bottom of our binding table block and zero of the block pool to the
 * surface state offsets so that they are correct relative to out new surface
 * state base address at the bottom of the binding table block.
 *
 * \param[in]  entries        The number of surface state entries the binding
 *                            table should be able to hold.
 *
 * \param[out] state_offset   The offset surface surface state base address
 *                            where the surface states live.  This must be
 *                            added to the surface state offset when it is
 *                            written into the binding table entry.
 *
 * \return                    An anv_state representing the binding table
 */
struct anv_state
anv_cmd_buffer_alloc_binding_table(struct anv_cmd_buffer *cmd_buffer,
                                   uint32_t entries, uint32_t *state_offset)
{
   if (u_vector_length(&cmd_buffer->bt_block_states) == 0)
      return (struct anv_state) { 0 };

   struct anv_state *bt_block = u_vector_head(&cmd_buffer->bt_block_states);

   uint32_t bt_size = align(entries * 4, 32);

   struct anv_state state = cmd_buffer->bt_next;
   if (bt_size > state.alloc_size)
      return (struct anv_state) { 0 };

   state.alloc_size = bt_size;
   cmd_buffer->bt_next.offset += bt_size;
   cmd_buffer->bt_next.map += bt_size;
   cmd_buffer->bt_next.alloc_size -= bt_size;

   if (cmd_buffer->device->info->verx10 >= 125) {
      /* We're using 3DSTATE_BINDING_TABLE_POOL_ALLOC to change the binding
       * table address independently from surface state base address.  We no
       * longer need any sort of offsetting.
       */
      *state_offset = 0;
   } else {
      assert(bt_block->offset < 0);
      *state_offset = -bt_block->offset;
   }

   return state;
}

struct anv_state
anv_cmd_buffer_alloc_surface_states(struct anv_cmd_buffer *cmd_buffer,
                                    uint32_t count)
{
   if (count == 0)
      return ANV_STATE_NULL;
   struct isl_device *isl_dev = &cmd_buffer->device->isl_dev;
   struct anv_state state =
      anv_state_stream_alloc(&cmd_buffer->surface_state_stream,
                             count * isl_dev->ss.size,
                             isl_dev->ss.align);
   if (state.map == NULL)
      anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return state;
}

struct anv_state
anv_cmd_buffer_alloc_dynamic_state(struct anv_cmd_buffer *cmd_buffer,
                                   uint32_t size, uint32_t alignment)
{
   if (size == 0)
      return ANV_STATE_NULL;
   struct anv_state state =
      anv_state_stream_alloc(&cmd_buffer->dynamic_state_stream,
                             size, alignment);
   if (state.map == NULL)
      anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return state;
}

struct anv_state
anv_cmd_buffer_alloc_general_state(struct anv_cmd_buffer *cmd_buffer,
                                   uint32_t size, uint32_t alignment)
{
   if (size == 0)
      return ANV_STATE_NULL;
   struct anv_state state =
      anv_state_stream_alloc(&cmd_buffer->general_state_stream,
                             size, alignment);
   if (state.map == NULL)
      anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return state;
}

/** Allocate space associated with a command buffer
 *
 * Some commands like vkCmdBuildAccelerationStructuresKHR() can end up needing
 * large amount of temporary buffers. This function is here to deal with those
 * potentially larger allocations, using a side BO if needed.
 *
 */
struct anv_cmd_alloc
anv_cmd_buffer_alloc_space(struct anv_cmd_buffer *cmd_buffer,
                           size_t size, uint32_t alignment,
                           bool mapped)
{
   /* Below 16k, source memory from dynamic state, otherwise allocate a BO. */
   if (size < 16 * 1024) {
      struct anv_state state =
         anv_state_stream_alloc(&cmd_buffer->dynamic_state_stream,
                                size, alignment);
      if (state.map == NULL) {
         anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return (struct anv_cmd_alloc) {
            .address = ANV_NULL_ADDRESS,
         };
      }

      return (struct anv_cmd_alloc) {
         .address = anv_state_pool_state_address(
            &cmd_buffer->device->dynamic_state_pool,
            state),
         .map = state.map,
         .size = size,
      };
   }

   assert(alignment <= 4096);

   struct anv_bo *bo = NULL;
   VkResult result =
      anv_bo_pool_alloc(mapped ?
                        &cmd_buffer->device->batch_bo_pool :
                        &cmd_buffer->device->bvh_bo_pool,
                        align(size, 4096), &bo);
   ANV_DMR_BO_ALLOC(&cmd_buffer->vk.base, bo, result);
   if (result != VK_SUCCESS) {
      anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return ANV_EMPTY_ALLOC;
   }

   struct anv_bo **bo_entry =
      u_vector_add(&cmd_buffer->dynamic_bos);
   if (bo_entry == NULL) {
      anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
      ANV_DMR_BO_FREE(&cmd_buffer->vk.base, bo);
      anv_bo_pool_free(bo->map != NULL ?
                       &cmd_buffer->device->batch_bo_pool :
                       &cmd_buffer->device->bvh_bo_pool, bo);

      return ANV_EMPTY_ALLOC;
   }
   *bo_entry = bo;

   return (struct anv_cmd_alloc) {
      .address = (struct anv_address) { .bo = bo },
      .map = bo->map,
      .size = size,
   };
}

VkResult
anv_cmd_buffer_new_binding_table_block(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_state *bt_block = u_vector_add(&cmd_buffer->bt_block_states);
   if (bt_block == NULL) {
      anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *bt_block = anv_binding_table_pool_alloc(cmd_buffer->device);

   /* The bt_next state is a rolling state (we update it as we suballocate
    * from it) which is relative to the start of the binding table block.
    */
   cmd_buffer->bt_next = *bt_block;
   cmd_buffer->bt_next.offset = 0;

   return VK_SUCCESS;
}

VkResult
anv_cmd_buffer_init_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_batch_bo *batch_bo = NULL;
   VkResult result;

   list_inithead(&cmd_buffer->batch_bos);

   cmd_buffer->total_batch_size = 0;

   result = anv_batch_bo_create(cmd_buffer,
                                ANV_MIN_CMD_BUFFER_BATCH_SIZE,
                                &batch_bo);
   if (result != VK_SUCCESS)
      return result;

   list_addtail(&batch_bo->link, &cmd_buffer->batch_bos);

   cmd_buffer->batch.alloc = &cmd_buffer->vk.pool->alloc;
   cmd_buffer->batch.user_data = cmd_buffer;
   cmd_buffer->batch.allocated_batch_size = ANV_MIN_CMD_BUFFER_BATCH_SIZE;

   cmd_buffer->batch.extend_cb = anv_cmd_buffer_chain_batch;
   cmd_buffer->batch.engine_class = cmd_buffer->queue_family->engine_class;
   cmd_buffer->batch.trace = &cmd_buffer->trace;

   anv_batch_bo_start(batch_bo, &cmd_buffer->batch,
                      GFX9_MI_BATCH_BUFFER_START_length * 4);

   /* Generation batch is initialized empty since it's possible it won't be
    * used.
    */
   list_inithead(&cmd_buffer->generation.batch_bos);

   cmd_buffer->generation.batch.alloc = &cmd_buffer->vk.pool->alloc;
   cmd_buffer->generation.batch.user_data = cmd_buffer;
   cmd_buffer->generation.batch.allocated_batch_size = 0;
   cmd_buffer->generation.batch.extend_cb = anv_cmd_buffer_chain_generation_batch;
   cmd_buffer->generation.batch.engine_class =
      cmd_buffer->queue_family->engine_class;

   int success = u_vector_init_pow2(&cmd_buffer->seen_bbos, 8,
                                    sizeof(struct anv_bo *));
   if (!success)
      goto fail_batch_bo;

   *(struct anv_batch_bo **)u_vector_add(&cmd_buffer->seen_bbos) = batch_bo;

   success = u_vector_init(&cmd_buffer->bt_block_states, 8,
                           sizeof(struct anv_state));
   if (!success)
      goto fail_seen_bbos;

   const bool uses_relocs = cmd_buffer->device->physical->uses_relocs;
   result = anv_reloc_list_init(&cmd_buffer->surface_relocs,
                                &cmd_buffer->vk.pool->alloc, uses_relocs);
   if (result != VK_SUCCESS)
      goto fail_bt_blocks;

   return VK_SUCCESS;

 fail_bt_blocks:
   u_vector_finish(&cmd_buffer->bt_block_states);
 fail_seen_bbos:
   u_vector_finish(&cmd_buffer->seen_bbos);
 fail_batch_bo:
   anv_batch_bo_destroy(batch_bo, cmd_buffer);

   return result;
}

void
anv_cmd_buffer_fini_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_state *bt_block;
   u_vector_foreach(bt_block, &cmd_buffer->bt_block_states)
      anv_binding_table_pool_free(cmd_buffer->device, *bt_block);
   u_vector_finish(&cmd_buffer->bt_block_states);

   anv_reloc_list_finish(&cmd_buffer->surface_relocs);

   u_vector_finish(&cmd_buffer->seen_bbos);

   /* Destroy all of the batch buffers */
   list_for_each_entry_safe(struct anv_batch_bo, bbo,
                            &cmd_buffer->batch_bos, link) {
      list_del(&bbo->link);
      anv_batch_bo_destroy(bbo, cmd_buffer);
   }
   /* Also destroy all generation batch buffers */
   list_for_each_entry_safe(struct anv_batch_bo, bbo,
                            &cmd_buffer->generation.batch_bos, link) {
      list_del(&bbo->link);
      anv_batch_bo_destroy(bbo, cmd_buffer);
   }

   if (cmd_buffer->generation.ring_bo) {
      ANV_DMR_BO_FREE(&cmd_buffer->vk.base, cmd_buffer->generation.ring_bo);
      anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool,
                       cmd_buffer->generation.ring_bo);
   }
}

void
anv_cmd_buffer_reset_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer)
{
   /* Delete all but the first batch bo */
   assert(!list_is_empty(&cmd_buffer->batch_bos));
   while (cmd_buffer->batch_bos.next != cmd_buffer->batch_bos.prev) {
      struct anv_batch_bo *bbo = anv_cmd_buffer_current_batch_bo(cmd_buffer);
      list_del(&bbo->link);
      anv_batch_bo_destroy(bbo, cmd_buffer);
   }
   assert(!list_is_empty(&cmd_buffer->batch_bos));

   anv_batch_bo_start(anv_cmd_buffer_current_batch_bo(cmd_buffer),
                      &cmd_buffer->batch,
                      GFX9_MI_BATCH_BUFFER_START_length * 4);

   while (u_vector_length(&cmd_buffer->bt_block_states) > 0) {
      struct anv_state *bt_block = u_vector_remove(&cmd_buffer->bt_block_states);
      anv_binding_table_pool_free(cmd_buffer->device, *bt_block);
   }
   cmd_buffer->bt_next = ANV_STATE_NULL;

   anv_reloc_list_clear(&cmd_buffer->surface_relocs);

   /* Reset the list of seen buffers */
   cmd_buffer->seen_bbos.head = 0;
   cmd_buffer->seen_bbos.tail = 0;

   struct anv_batch_bo *first_bbo = anv_cmd_buffer_current_batch_bo(cmd_buffer);

   *(struct anv_batch_bo **)u_vector_add(&cmd_buffer->seen_bbos) = first_bbo;

   assert(first_bbo->bo->size == ANV_MIN_CMD_BUFFER_BATCH_SIZE);
   cmd_buffer->batch.allocated_batch_size = first_bbo->bo->size;

   /* Delete all generation batch bos */
   list_for_each_entry_safe(struct anv_batch_bo, bbo,
                            &cmd_buffer->generation.batch_bos, link) {
      list_del(&bbo->link);
      anv_batch_bo_destroy(bbo, cmd_buffer);
   }

   /* And reset generation batch */
   cmd_buffer->generation.batch.allocated_batch_size = 0;
   cmd_buffer->generation.batch.start = NULL;
   cmd_buffer->generation.batch.end   = NULL;
   cmd_buffer->generation.batch.next  = NULL;

   if (cmd_buffer->generation.ring_bo) {
      ANV_DMR_BO_FREE(&cmd_buffer->vk.base, cmd_buffer->generation.ring_bo);
      anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool,
                       cmd_buffer->generation.ring_bo);
      cmd_buffer->generation.ring_bo = NULL;
   }

   cmd_buffer->total_batch_size = 0;
}

void
anv_cmd_buffer_end_batch_buffer(struct anv_cmd_buffer *cmd_buffer)
{
   const struct intel_device_info *devinfo = cmd_buffer->device->info;
   struct anv_batch_bo *batch_bo = anv_cmd_buffer_current_batch_bo(cmd_buffer);

   if (cmd_buffer->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
      /* When we start a batch buffer, we subtract a certain amount of
       * padding from the end to ensure that we always have room to emit a
       * BATCH_BUFFER_START to chain to the next BO.  We need to remove
       * that padding before we end the batch; otherwise, we may end up
       * with our BATCH_BUFFER_END in another BO.
       */
      cmd_buffer->batch.end += GFX9_MI_BATCH_BUFFER_START_length * 4;
      assert(cmd_buffer->batch.start == batch_bo->bo->map);
      assert(cmd_buffer->batch.end == batch_bo->bo->map + batch_bo->bo->size);

      /* Save end instruction location to override it later. */
      cmd_buffer->batch_end = cmd_buffer->batch.next;

      /* If we can chain this command buffer to another one, leave some place
       * for the jump instruction.
       */
      batch_bo->chained = anv_cmd_buffer_is_chainable(cmd_buffer);
      if (batch_bo->chained)
         emit_batch_buffer_start(&cmd_buffer->batch, batch_bo->bo, 0);
      else
         anv_batch_emit(&cmd_buffer->batch, GFX9_MI_BATCH_BUFFER_END, bbe);

      /* Round batch up to an even number of dwords. */
      if ((cmd_buffer->batch.next - cmd_buffer->batch.start) & 4)
         anv_batch_emit(&cmd_buffer->batch, GFX9_MI_NOOP, noop);

      cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_PRIMARY;
   } else {
      assert(cmd_buffer->vk.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY);
      /* If this is a secondary command buffer, we need to determine the
       * mode in which it will be executed with vkExecuteCommands.  We
       * determine this statically here so that this stays in sync with the
       * actual ExecuteCommands implementation.
       */
      const uint32_t length = cmd_buffer->batch.next - cmd_buffer->batch.start;
      if (!(cmd_buffer->device->physical->instance->debug & ANV_DEBUG_NO_SECONDARY_CALL)) {
         cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_CALL_AND_RETURN;

         void *jump_addr =
            anv_genX(devinfo, batch_emit_return)(&cmd_buffer->batch) +
            (GFX9_MI_BATCH_BUFFER_START_BatchBufferStartAddress_start / 8);
         cmd_buffer->return_addr = anv_batch_address(&cmd_buffer->batch, jump_addr);

         /* The emit above may have caused us to chain batch buffers which
          * would mean that batch_bo is no longer valid.
          */
         batch_bo = anv_cmd_buffer_current_batch_bo(cmd_buffer);
      } else if ((cmd_buffer->batch_bos.next == cmd_buffer->batch_bos.prev) &&
                 (length < ANV_MIN_CMD_BUFFER_BATCH_SIZE / 2)) {
         /* If the secondary has exactly one batch buffer in its list *and*
          * that batch buffer is less than half of the maximum size, we're
          * probably better of simply copying it into our batch.
          */
         cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_EMIT;
      } else if (!(cmd_buffer->usage_flags &
                   VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)) {
         cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_CHAIN;

         /* In order to chain, we need this command buffer to contain an
          * MI_BATCH_BUFFER_START which will jump back to the calling batch.
          * It doesn't matter where it points now so long as has a valid
          * relocation.  We'll adjust it later as part of the chaining
          * process.
          *
          * We set the end of the batch a little short so we would be sure we
          * have room for the chaining command.  Since we're about to emit the
          * chaining command, let's set it back where it should go.
          */
         cmd_buffer->batch.end += GFX9_MI_BATCH_BUFFER_START_length * 4;
         assert(cmd_buffer->batch.start == batch_bo->bo->map);
         assert(cmd_buffer->batch.end == batch_bo->bo->map + batch_bo->bo->size);

         emit_batch_buffer_start(&cmd_buffer->batch, batch_bo->bo, 0);
         assert(cmd_buffer->batch.start == batch_bo->bo->map);
      } else {
         cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_COPY_AND_CHAIN;
      }
   }

   anv_batch_bo_finish(batch_bo, &cmd_buffer->batch);

   /* Add the current amount of data written in the current_bbo to the command
    * buffer.
    */
   cmd_buffer->total_batch_size += batch_bo->length;
}

static VkResult
anv_cmd_buffer_add_seen_bbos(struct anv_cmd_buffer *cmd_buffer,
                             struct list_head *list)
{
   list_for_each_entry(struct anv_batch_bo, bbo, list, link) {
      struct anv_batch_bo **bbo_ptr = u_vector_add(&cmd_buffer->seen_bbos);
      if (bbo_ptr == NULL)
         return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);

      *bbo_ptr = bbo;
   }

   return VK_SUCCESS;
}

void
anv_cmd_buffer_add_secondary(struct anv_cmd_buffer *primary,
                             struct anv_cmd_buffer *secondary)
{
   anv_measure_add_secondary(primary, secondary);
   switch (secondary->exec_mode) {
   case ANV_CMD_BUFFER_EXEC_MODE_EMIT:
      anv_batch_emit_batch(&primary->batch, &secondary->batch);
      break;
   case ANV_CMD_BUFFER_EXEC_MODE_CHAIN: {
      struct anv_batch_bo *first_bbo =
         list_first_entry(&secondary->batch_bos, struct anv_batch_bo, link);
      struct anv_batch_bo *last_bbo =
         list_last_entry(&secondary->batch_bos, struct anv_batch_bo, link);

      emit_batch_buffer_start(&primary->batch, first_bbo->bo, 0);

      struct anv_batch_bo *this_bbo = anv_cmd_buffer_current_batch_bo(primary);
      assert(primary->batch.start == this_bbo->bo->map);
      uint32_t offset = primary->batch.next - primary->batch.start;

      /* Make the tail of the secondary point back to right after the
       * MI_BATCH_BUFFER_START in the primary batch.
       */
      anv_batch_bo_link(primary, last_bbo, this_bbo, offset);

      anv_cmd_buffer_add_seen_bbos(primary, &secondary->batch_bos);
      break;
   }
   case ANV_CMD_BUFFER_EXEC_MODE_COPY_AND_CHAIN: {
      struct list_head copy_list;
      VkResult result = anv_batch_bo_list_clone(&secondary->batch_bos,
                                                secondary,
                                                &copy_list);
      if (result != VK_SUCCESS)
         return; /* FIXME */

      anv_cmd_buffer_add_seen_bbos(primary, &copy_list);

      struct anv_batch_bo *first_bbo =
         list_first_entry(&copy_list, struct anv_batch_bo, link);
      struct anv_batch_bo *last_bbo =
         list_last_entry(&copy_list, struct anv_batch_bo, link);

      cmd_buffer_chain_to_batch_bo(primary, first_bbo,
                                   ANV_CMD_BUFFER_BATCH_MAIN);

      list_splicetail(&copy_list, &primary->batch_bos);

      anv_batch_bo_continue(last_bbo, &primary->batch,
                            GFX9_MI_BATCH_BUFFER_START_length * 4);
      break;
   }
   case ANV_CMD_BUFFER_EXEC_MODE_CALL_AND_RETURN: {
      struct anv_batch_bo *first_bbo =
         list_first_entry(&secondary->batch_bos, struct anv_batch_bo, link);

      anv_genX(primary->device->info, batch_emit_secondary_call)(
         &primary->batch, primary->device,
         (struct anv_address) { .bo = first_bbo->bo },
         secondary->return_addr);

      anv_cmd_buffer_add_seen_bbos(primary, &secondary->batch_bos);
      break;
   }
   default:
      UNREACHABLE("Invalid execution mode");
   }

   anv_reloc_list_append(&primary->surface_relocs, &secondary->surface_relocs);

   /* Add the amount of data written in the secondary buffer to the primary
    * command buffer.
    */
   primary->total_batch_size += secondary->total_batch_size;
}

void
anv_cmd_buffer_chain_command_buffers(struct anv_cmd_buffer **cmd_buffers,
                                     uint32_t num_cmd_buffers)
{
   if (!anv_cmd_buffer_is_chainable(cmd_buffers[0])) {
      assert(num_cmd_buffers == 1);
      return;
   }

   /* Chain the N-1 first batch buffers */
   for (uint32_t i = 0; i < (num_cmd_buffers - 1); i++) {
      assert(cmd_buffers[i]->companion_rcs_cmd_buffer == NULL);
      anv_cmd_buffer_record_chain_submit(cmd_buffers[i], cmd_buffers[i + 1]);
   }

   /* Put an end to the last one */
   anv_cmd_buffer_record_end_submit(cmd_buffers[num_cmd_buffers - 1]);
}

static void
anv_print_batch(struct anv_device *device,
                struct anv_queue *queue,
                struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_batch_bo *bbo =
      list_first_entry(&cmd_buffer->batch_bos, struct anv_batch_bo, link);
   device->cmd_buffer_being_decoded = cmd_buffer;
   struct intel_batch_decode_ctx *ctx = queue->decoder;

   if (cmd_buffer->is_companion_rcs_cmd_buffer) {
      int render_queue_idx =
         anv_get_first_render_queue_index(device->physical);
      ctx = &device->decoder[render_queue_idx];
   }

   if (INTEL_DEBUG(DEBUG_BATCH)) {
      intel_print_batch(ctx, bbo->bo->map,
                        bbo->bo->size, bbo->bo->offset, false);
   }
   if (INTEL_DEBUG(DEBUG_BATCH_STATS)) {
      intel_batch_stats(ctx, bbo->bo->map,
                        bbo->bo->size, bbo->bo->offset, false);
   }
   device->cmd_buffer_being_decoded = NULL;
}

void
anv_cmd_buffer_exec_batch_debug(struct anv_queue *queue,
                                uint32_t cmd_buffer_count,
                                struct anv_cmd_buffer **cmd_buffers,
                                struct anv_query_pool *perf_query_pool,
                                uint32_t perf_query_pass)
{
   if (!INTEL_DEBUG(DEBUG_BATCH) && !INTEL_DEBUG(DEBUG_BATCH_STATS))
      return;

   struct anv_device *device = queue->device;
   const bool has_perf_query = perf_query_pool && cmd_buffer_count;
   uint64_t frame_id = device->debug_frame_desc->frame_id;

   if (!intel_debug_batch_in_range(device->debug_frame_desc->frame_id))
      return;
   fprintf(stderr, "Batch for frame %"PRIu64" on queue %d\n",
      frame_id, (int)(queue - device->queues));

   if (cmd_buffer_count) {
      if (has_perf_query) {
         struct anv_bo *pass_batch_bo = perf_query_pool->bo;
         uint64_t pass_batch_offset =
            khr_perf_query_preamble_offset(perf_query_pool, perf_query_pass);

         if (INTEL_DEBUG(DEBUG_BATCH)) {
            intel_print_batch(queue->decoder,
                              pass_batch_bo->map + pass_batch_offset, 64,
                              pass_batch_bo->offset + pass_batch_offset, false);
         }
      }

      for (uint32_t i = 0; i < cmd_buffer_count; i++)
         anv_print_batch(device, queue, cmd_buffers[i]);
   } else if (INTEL_DEBUG(DEBUG_BATCH)) {
      intel_print_batch(queue->decoder, device->trivial_batch_bo->map,
                        device->trivial_batch_bo->size,
                        device->trivial_batch_bo->offset, false);
   }
}

/* We lock around execbuf for three main reasons:
 *
 *  1) When a block pool is resized, we create a new gem handle with a
 *     different size and, in the case of surface states, possibly a different
 *     center offset but we re-use the same anv_bo struct when we do so. If
 *     this happens in the middle of setting up an execbuf, we could end up
 *     with our list of BOs out of sync with our list of gem handles.
 *
 *  2) The algorithm we use for building the list of unique buffers isn't
 *     thread-safe. While the client is supposed to synchronize around
 *     QueueSubmit, this would be extremely difficult to debug if it ever came
 *     up in the wild due to a broken app. It's better to play it safe and
 *     just lock around QueueSubmit.
 *
 * Since the only other things that ever take the device lock such as block
 * pool resize only rarely happen, this will almost never be contended so
 * taking a lock isn't really an expensive operation in this case.
 */
static inline VkResult
anv_queue_exec_locked(struct anv_queue *queue,
                      uint32_t wait_count,
                      const struct vk_sync_wait *waits,
                      uint32_t cmd_buffer_count,
                      struct anv_cmd_buffer **cmd_buffers,
                      uint32_t signal_count,
                      const struct vk_sync_signal *signals,
                      struct anv_query_pool *perf_query_pool,
                      uint32_t perf_query_pass,
                      struct anv_utrace_submit *utrace_submit)
{
   struct anv_device *device = queue->device;
   VkResult result = VK_SUCCESS;

   /* We only need to synchronize the main & companion command buffers if we
    * have a companion command buffer somewhere in the list of command
    * buffers.
    */
   bool needs_companion_sync = false;
   for (uint32_t i = 0; i < cmd_buffer_count; i++) {
      if (cmd_buffers[i]->companion_rcs_cmd_buffer != NULL) {
         needs_companion_sync = true;
         break;
      }
   }

   if (perf_query_pool && device->perf_queue != queue)
      debug_warn_once("Mismatch between queue that OA stream was open and "
                      "queue were query will be executed.");

   result =
      device->kmd_backend->queue_exec_locked(
         queue,
         wait_count, waits,
         cmd_buffer_count, cmd_buffers,
         needs_companion_sync ? 0 : signal_count, signals,
         perf_query_pool,
         perf_query_pass,
         utrace_submit);
   if (result != VK_SUCCESS)
      return result;

   if (needs_companion_sync) {
      struct vk_sync_wait companion_sync = {
         .sync = queue->companion_sync,
      };
      /* If any of the command buffer had a companion batch, the submission
       * backend will signal queue->companion_sync, so to ensure completion,
       * we just need to wait on that fence.
       */
      result =
         device->kmd_backend->queue_exec_locked(queue,
                                                1, &companion_sync,
                                                0, NULL,
                                                signal_count, signals,
                                                NULL, 0,
                                                NULL);
   }

   return result;
}

static inline bool
can_chain_query_pools(struct anv_query_pool *p1, struct anv_query_pool *p2)
{
   return (!p1 || !p2 || p1 == p2);
}

static VkResult
anv_queue_submit_sparse_bind_locked(struct anv_queue *queue,
                                    struct vk_queue_submit *submit)
{
   struct anv_device *device = queue->device;
   VkResult result;

   /* When fake sparse is enabled, while we do accept creating "sparse"
    * resources we can't really handle sparse submission. Fake sparse is
    * supposed to be used by applications that request sparse to be enabled
    * but don't actually *use* it.
    */
   if (device->physical->sparse_type == ANV_SPARSE_TYPE_NOT_SUPPORTED) {
      if (INTEL_DEBUG(DEBUG_SPARSE))
         fprintf(stderr, "=== application submitting sparse operations: "
               "buffer_bind:%d image_opaque_bind:%d image_bind:%d\n",
               submit->buffer_bind_count, submit->image_opaque_bind_count,
               submit->image_bind_count);
      return vk_queue_set_lost(&queue->vk, "Sparse binding not supported");
   }

   assert(submit->command_buffer_count == 0);

   if (INTEL_DEBUG(DEBUG_SPARSE)) {
      fprintf(stderr, "[sparse submission, buffers:%u opaque_images:%u "
              "images:%u waits:%u signals:%u]\n",
              submit->buffer_bind_count,
              submit->image_opaque_bind_count,
              submit->image_bind_count,
              submit->wait_count, submit->signal_count);
   }

   struct anv_sparse_submission sparse_submit = {
      .queue = queue,
      .binds = NULL,
      .binds_len = 0,
      .binds_capacity = 0,
      .wait_count = submit->wait_count,
      .signal_count = submit->signal_count,
      .waits = submit->waits,
      .signals = submit->signals,
   };

   for (uint32_t i = 0; i < submit->buffer_bind_count; i++) {
      VkSparseBufferMemoryBindInfo *bind_info = &submit->buffer_binds[i];
      ANV_FROM_HANDLE(anv_buffer, buffer, bind_info->buffer);

      assert(anv_buffer_is_sparse(buffer));

      for (uint32_t j = 0; j < bind_info->bindCount; j++) {
         result = anv_sparse_bind_buffer(device, buffer,
                                         &bind_info->pBinds[j],
                                         &sparse_submit);
         if (result != VK_SUCCESS)
            goto out_free_submit;
      }
   }

   for (uint32_t i = 0; i < submit->image_bind_count; i++) {
      VkSparseImageMemoryBindInfo *bind_info = &submit->image_binds[i];
      ANV_FROM_HANDLE(anv_image, image, bind_info->image);

      assert(anv_image_is_sparse(image));
      assert(image->vk.create_flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT);

      for (uint32_t j = 0; j < bind_info->bindCount; j++) {
         result = anv_sparse_bind_image_memory(queue, image,
                                               &bind_info->pBinds[j],
                                               &sparse_submit);
         if (result != VK_SUCCESS)
            goto out_free_submit;
      }
   }

   for (uint32_t i = 0; i < submit->image_opaque_bind_count; i++) {
      VkSparseImageOpaqueMemoryBindInfo *bind_info =
         &submit->image_opaque_binds[i];
      ANV_FROM_HANDLE(anv_image, image, bind_info->image);

      assert(anv_image_is_sparse(image));

      for (uint32_t j = 0; j < bind_info->bindCount; j++) {
         result = anv_sparse_bind_image_opaque(device, image,
                                               &bind_info->pBinds[j],
                                               &sparse_submit);
         if (result != VK_SUCCESS)
            goto out_free_submit;
      }
   }

   result = anv_sparse_bind(device, &sparse_submit);

out_free_submit:
   vk_free(&device->vk.alloc, sparse_submit.binds);
   return result;
}

static VkResult
anv_queue_submit_cmd_buffers_locked(struct anv_queue *queue,
                                    struct vk_queue_submit *submit,
                                    struct anv_utrace_submit *utrace_submit)
{
   VkResult result;

   /* It's not safe to access submit->signals[] elements after submit because
    * the elements might signal through the kernel before this function
    * returns and another thread could wake up and destroy any of those
    * elements.
    *
    * Build a list of anv_bo_sync elements here and put them in the signal
    * state after without looking at any other element.
    */
   STACK_ARRAY(struct anv_bo_sync *, bo_signals, submit->signal_count);
   uint32_t bo_signal_count = 0;
   for (uint32_t i = 0; i < submit->signal_count; i++) {
      if (!vk_sync_is_anv_bo_sync(submit->signals[i].sync))
         continue;

      bo_signals[bo_signal_count++] =
         container_of(submit->signals[i].sync, struct anv_bo_sync, sync);
   }

   if (submit->command_buffer_count == 0) {
      result = anv_queue_exec_locked(queue, submit->wait_count, submit->waits,
                                     0 /* cmd_buffer_count */,
                                     NULL /* cmd_buffers */,
                                     submit->signal_count, submit->signals,
                                     NULL /* perf_query_pool */,
                                     0 /* perf_query_pass */,
                                     utrace_submit);
      if (result != VK_SUCCESS)
         goto fail;
   } else {
      /* Everything's easier if we don't have to bother with container_of() */
      STATIC_ASSERT(offsetof(struct anv_cmd_buffer, vk) == 0);
      struct vk_command_buffer **vk_cmd_buffers = submit->command_buffers;
      struct anv_cmd_buffer **cmd_buffers = (void *)vk_cmd_buffers;
      uint32_t start = 0;
      uint32_t end = submit->command_buffer_count;
      struct anv_query_pool *perf_query_pool =
         cmd_buffers[start]->perf_query_pool;
      for (uint32_t n = 0; n < end; n++) {
         bool can_chain = false;
         uint32_t next = n + 1;
         /* Can we chain the last buffer into the next one? */
         if (next < end &&
             anv_cmd_buffer_is_chainable(cmd_buffers[n]) &&
             anv_cmd_buffer_is_chainable(cmd_buffers[next]) &&
             can_chain_query_pools
             (cmd_buffers[next]->perf_query_pool, perf_query_pool)) {
            can_chain = true;
            perf_query_pool =
               perf_query_pool ? perf_query_pool :
               cmd_buffers[next]->perf_query_pool;
         }
         if (!can_chain) {
            /* The next buffer cannot be chained, or we have reached the
             * last buffer, submit what have been chained so far.
             */
            result =
               anv_queue_exec_locked(queue,
                                     start == 0 ? submit->wait_count : 0,
                                     start == 0 ? submit->waits : NULL,
                                     next - start, &cmd_buffers[start],
                                     next == end ? submit->signal_count : 0,
                                     next == end ? submit->signals : NULL,
                                     perf_query_pool,
                                     submit->perf_pass_index,
                                     next == end ? utrace_submit : NULL);
            if (result != VK_SUCCESS)
               goto fail;
            if (next < end) {
               start = next;
               perf_query_pool = cmd_buffers[start]->perf_query_pool;
            }
         }
      }
   }
   for (uint32_t i = 0; i < bo_signal_count; i++) {
      struct anv_bo_sync *bo_sync = bo_signals[i];

      /* Once the execbuf has returned, we need to set the fence state to
       * SUBMITTED.  We can't do this before calling execbuf because
       * anv_GetFenceStatus does take the global device lock before checking
       * fence->state.
       *
       * We set the fence state to SUBMITTED regardless of whether or not the
       * execbuf succeeds because we need to ensure that vkWaitForFences() and
       * vkGetFenceStatus() return a valid result (VK_ERROR_DEVICE_LOST or
       * VK_SUCCESS) in a finite amount of time even if execbuf fails.
       */
      assert(bo_sync->state == ANV_BO_SYNC_STATE_RESET);
      bo_sync->state = ANV_BO_SYNC_STATE_SUBMITTED;
   }

   pthread_cond_broadcast(&queue->device->queue_submit);

 fail:
   STACK_ARRAY_FINISH(bo_signals);
   return result;
}

static inline void
anv_queue_free_initial_submission(struct anv_queue *queue)
{
   if (queue->init_submit &&
       anv_async_submit_done(queue->init_submit)) {
      anv_async_submit_destroy(queue->init_submit);
      queue->init_submit = NULL;
   }
   if (queue->init_companion_submit &&
       anv_async_submit_done(queue->init_companion_submit)) {
      anv_async_submit_destroy(queue->init_companion_submit);
      queue->init_companion_submit = NULL;
   }
}

VkResult
anv_queue_submit(struct vk_queue *vk_queue,
                 struct vk_queue_submit *submit)
{
   struct anv_queue *queue = container_of(vk_queue, struct anv_queue, vk);
   struct anv_device *device = queue->device;
   VkResult result;

   anv_queue_free_initial_submission(queue);

   if (u_trace_should_process(&device->ds.trace_context)) {
      /* Refresh perfetto's knowledge of any easily accessible, long-lived
       * object names referenced by this submit -- this helps provide better
       * information in traces when tracing starts after application launch.
       */
      for (int i = 0; i < submit->command_buffer_count; i++)
         intel_ds_perfetto_refresh_debug_utils_object_name(&device->ds,
                                                           &submit->command_buffers[i]->base);
   }

   if (queue->device->info->no_hw) {
      for (uint32_t i = 0; i < submit->signal_count; i++) {
         result = vk_sync_signal(&device->vk,
                                 submit->signals[i].sync,
                                 submit->signals[i].signal_value);
         if (result != VK_SUCCESS)
            return vk_queue_set_lost(&queue->vk, "vk_sync_signal failed");
      }
      return VK_SUCCESS;
   }

   /* Flush the trace points first before taking the lock as the flushing
    * might try to take that same lock.
    */
   struct anv_utrace_submit *utrace_submit = NULL;
   result = anv_device_utrace_flush_cmd_buffers(
      queue,
      submit->command_buffer_count,
      (struct anv_cmd_buffer **)submit->command_buffers,
      &utrace_submit);
   if (result != VK_SUCCESS)
      return result;

   pthread_mutex_lock(&device->mutex);

   uint64_t start_ts = intel_ds_begin_submit(&queue->ds);

   if (submit->buffer_bind_count ||
       submit->image_opaque_bind_count ||
       submit->image_bind_count) {
      result = anv_queue_submit_sparse_bind_locked(queue, submit);
   } else {
      result = anv_queue_submit_cmd_buffers_locked(queue, submit,
                                                   utrace_submit);
   }

   /* Take submission ID under lock */
   intel_ds_end_submit(&queue->ds, start_ts);

   pthread_mutex_unlock(&device->mutex);

   intel_ds_device_process(&device->ds, false);

   return result;
}

void
anv_cmd_buffer_clflush(struct anv_cmd_buffer **cmd_buffers,
                       uint32_t num_cmd_buffers)
{
#ifdef SUPPORT_INTEL_INTEGRATED_GPUS
   struct anv_batch_bo **bbo;

   __builtin_ia32_mfence();

   for (uint32_t i = 0; i < num_cmd_buffers; i++) {
      u_vector_foreach(bbo, &cmd_buffers[i]->seen_bbos) {
         intel_flush_range_no_fence((*bbo)->bo->map, (*bbo)->length);
      }
   }

   __builtin_ia32_mfence();
#endif
}

static VkResult
anv_async_submit_extend_batch(struct anv_batch *batch, uint32_t size,
                              void *user_data)
{
   struct anv_async_submit *submit = user_data;
   struct anv_queue *queue = submit->queue;

   uint32_t alloc_size = 0;
   util_dynarray_foreach(&submit->batch_bos, struct anv_bo *, bo)
      alloc_size += (*bo)->size;
   alloc_size = MAX2(alloc_size * 2, 8192);

   struct anv_bo *bo;
   VkResult result = anv_bo_pool_alloc(submit->bo_pool,
                                       align(alloc_size, 4096),
                                       &bo);
   ANV_DMR_BO_ALLOC(&queue->vk.base, bo, result);
   if (result != VK_SUCCESS)
      return result;

   util_dynarray_append(&submit->batch_bos, struct anv_bo *, bo);

   batch->end += 4 * GFX9_MI_BATCH_BUFFER_START_length;

   anv_batch_emit(batch, GFX9_MI_BATCH_BUFFER_START, bbs) {
      bbs.DWordLength               = GFX9_MI_BATCH_BUFFER_START_length -
                                      GFX9_MI_BATCH_BUFFER_START_length_bias;
      bbs.SecondLevelBatchBuffer    = Firstlevelbatch;
      bbs.AddressSpaceIndicator     = ASI_PPGTT;
      bbs.BatchBufferStartAddress   = (struct anv_address) { bo, 0 };
   }

   anv_batch_set_storage(batch,
                         (struct anv_address) { .bo = bo, },
                         bo->map,
                         bo->size - 4 * GFX9_MI_BATCH_BUFFER_START_length);

   return VK_SUCCESS;
}

VkResult
anv_async_submit_init(struct anv_async_submit *submit,
                      struct anv_queue *queue,
                      struct anv_bo_pool *bo_pool,
                      bool use_companion_rcs,
                      bool create_signal_sync)
{
   struct anv_device *device = queue->device;

   memset(submit, 0, sizeof(*submit));

   submit->use_companion_rcs = use_companion_rcs;
   submit->queue = queue;
   submit->bo_pool = bo_pool;

   const bool uses_relocs = device->physical->uses_relocs;
   VkResult result =
      anv_reloc_list_init(&submit->relocs, &device->vk.alloc, uses_relocs);
   if (result != VK_SUCCESS)
      return result;

   submit->batch = (struct anv_batch) {
      .alloc = &device->vk.alloc,
      .relocs = &submit->relocs,
      .user_data = submit,
      .extend_cb = anv_async_submit_extend_batch,
   };

   util_dynarray_init(&submit->batch_bos, NULL);

   if (create_signal_sync) {
      result = vk_sync_create(&device->vk,
                              &device->physical->sync_syncobj_type,
                              0, 0, &submit->signal.sync);
      if (result != VK_SUCCESS) {
         anv_reloc_list_finish(&submit->relocs);
         util_dynarray_fini(&submit->batch_bos);
         return result;
      }
      submit->owns_sync = true;
   }

   return VK_SUCCESS;
}

void
anv_async_submit_fini(struct anv_async_submit *submit)
{
   struct anv_device *device = submit->queue->device;
   struct anv_queue *queue = submit->queue;

   if (submit->owns_sync)
      vk_sync_destroy(&device->vk, submit->signal.sync);

   util_dynarray_foreach(&submit->batch_bos, struct anv_bo *, bo) {
      ANV_DMR_BO_FREE(&queue->vk.base, *bo);
      anv_bo_pool_free(submit->bo_pool, *bo);
   }

   util_dynarray_fini(&submit->batch_bos);
   anv_reloc_list_finish(&submit->relocs);
}

VkResult
anv_async_submit_create(struct anv_queue *queue,
                        struct anv_bo_pool *bo_pool,
                        bool use_companion_rcs,
                        bool create_signal_sync,
                        struct anv_async_submit **out_submit)
{
   struct anv_device *device = queue->device;

   *out_submit =
      vk_alloc(&device->vk.alloc, sizeof(struct anv_async_submit), 8,
               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (*out_submit == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result = anv_async_submit_init(*out_submit, queue,
                                           bo_pool,
                                           use_companion_rcs,
                                           create_signal_sync);
   if (result != VK_SUCCESS)
      vk_free(&device->vk.alloc, *out_submit);

   return result;
}

void
anv_async_submit_destroy(struct anv_async_submit *submit)
{
   struct anv_device *device = submit->queue->device;
   anv_async_submit_fini(submit);
   vk_free(&device->vk.alloc, submit);
}

bool
anv_async_submit_done(struct anv_async_submit *submit)
{
   struct anv_device *device = submit->queue->device;

   return vk_sync_wait(&device->vk,
                       submit->signal.sync,
                       submit->signal.signal_value,
                       VK_SYNC_WAIT_COMPLETE, 0) == VK_SUCCESS;
}

bool
anv_async_submit_wait(struct anv_async_submit *submit)
{
   struct anv_device *device = submit->queue->device;

   return vk_sync_wait(&device->vk,
                       submit->signal.sync,
                       submit->signal.signal_value,
                       VK_SYNC_WAIT_COMPLETE,
                       os_time_get_absolute_timeout(OS_TIMEOUT_INFINITE)) == VK_SUCCESS;
}

void
anv_async_submit_print_batch(struct anv_async_submit *submit)
{
   if ((INTEL_DEBUG(DEBUG_BATCH) | INTEL_DEBUG(DEBUG_BATCH_STATS)) == 0)
      return;

   struct anv_bo *batch_bo =
      *util_dynarray_element(&submit->batch_bos, struct anv_bo *, 0);
   struct intel_batch_decode_ctx *ctx = submit->queue->decoder;

   if (submit->use_companion_rcs) {
      struct anv_device *device = submit->queue->device;

      ctx = NULL;
      for (unsigned i = 0; i < device->physical->queue.family_count; i++) {
         struct intel_batch_decode_ctx *decoder = &device->decoder[i];

         if (decoder->engine == INTEL_ENGINE_CLASS_RENDER) {
            ctx = decoder;
            break;
         }
      }
   }

   assert(ctx);
   submit->queue->device->cmd_buffer_being_decoded = NULL;

   if (INTEL_DEBUG(DEBUG_BATCH))
      intel_print_batch(ctx, batch_bo->map, batch_bo->size, batch_bo->offset, false);

   if (INTEL_DEBUG(DEBUG_BATCH_STATS))
      intel_batch_stats(ctx, batch_bo->map, batch_bo->size, batch_bo->offset, false);
}
