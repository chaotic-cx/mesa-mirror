/*
 * Copyright (C) 2022 Collabora Ltd.
 * Copyright (C) 2025 Arm Ltd.
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

#pragma once

#if !defined(PAN_ARCH) || PAN_ARCH < 10
#error "cs_builder.h requires PAN_ARCH >= 10"
#endif

#include "gen_macros.h"

#include "util/bitset.h"
#include "util/u_dynarray.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Before Avalon, RUN_IDVS could use a selector but as we only hardcode the same
 * configuration, we match v12+ naming here */

#if PAN_ARCH <= 11
#define MALI_IDVS_SR_VERTEX_SRT      MALI_IDVS_SR_SRT_0
#define MALI_IDVS_SR_FRAGMENT_SRT    MALI_IDVS_SR_SRT_2
#define MALI_IDVS_SR_VERTEX_FAU      MALI_IDVS_SR_FAU_0
#define MALI_IDVS_SR_FRAGMENT_FAU    MALI_IDVS_SR_FAU_2
#define MALI_IDVS_SR_VERTEX_POS_SPD  MALI_IDVS_SR_SPD_0
#define MALI_IDVS_SR_VERTEX_VARY_SPD MALI_IDVS_SR_SPD_1
#define MALI_IDVS_SR_FRAGMENT_SPD    MALI_IDVS_SR_SPD_2
#endif

#if PAN_ARCH == 10
#define CS_MAX_SB_COUNT 8
#else
#define CS_MAX_SB_COUNT 16
#endif

/*
 * cs_builder implements a builder for CSF command streams. It manages the
 * allocation and overflow behaviour of queues and provides helpers for emitting
 * commands to run on the CSF pipe.
 *
 * Users are responsible for the CS buffer allocation and must initialize the
 * command stream with an initial buffer using cs_builder_init(). The CS can
 * be extended with new buffers allocated with cs_builder_conf::alloc_buffer()
 * if the builder runs out of memory.
 */

struct cs_buffer {
   /* CPU pointer */
   uint64_t *cpu;

   /* GPU pointer */
   uint64_t gpu;

   /* Capacity in number of 64-bit instructions */
   uint32_t capacity;
};

/**
 * This is used to check that:
 * 1. registers are not used as a source after being loaded without a
 *    WAIT(<ls_scoreboard>) in the middle
 * 2. registers are not reused (used as a destination) after they served as a
 *    STORE() source without a WAIT(<ls_scoreboard>) in the middle
 */
struct cs_load_store_tracker {
   BITSET_DECLARE(pending_loads, 256);
   bool pending_stores;
};

/**
 * This is used to determine which registers as been written to (a.k.a. used
 * as an instruction's destination).
 */
struct cs_dirty_tracker {
   BITSET_DECLARE(regs, 256);
};

enum cs_reg_perm {
   CS_REG_NO_ACCESS = 0,
   CS_REG_RD = BITFIELD_BIT(1),
   CS_REG_WR = BITFIELD_BIT(2),
   CS_REG_RW = CS_REG_RD | CS_REG_WR,
};

struct cs_builder;

typedef enum cs_reg_perm (*reg_perm_cb_t)(struct cs_builder *b, unsigned reg);

struct cs_builder_conf {
   /* Number of 32-bit registers in the hardware register file */
   uint8_t nr_registers;

   /* Number of 32-bit registers used by the kernel at submission time */
   uint8_t nr_kernel_registers;

   /* CS buffer allocator */
   struct cs_buffer (*alloc_buffer)(void *cookie);

   /* Optional dirty registers tracker. */
   struct cs_dirty_tracker *dirty_tracker;

   /* Optional register access checker. */
   reg_perm_cb_t reg_perm;

   /* Cookie passed back to alloc_buffer() */
   void *cookie;

   /* SB slot used for load/store instructions. */
   uint8_t ls_sb_slot;
};

/* The CS is formed of one or more CS chunks linked with JUMP instructions.
 * The builder keeps track of the current chunk and the position inside this
 * chunk, so it can emit new instructions, and decide when a new chunk needs
 * to be allocated.
 */
struct cs_chunk {
   /* CS buffer object backing this chunk */
   struct cs_buffer buffer;

   union {
      /* Current position in the buffer object when the chunk is active. */
      uint32_t pos;

      /* Chunk size when the chunk was wrapped. */
      uint32_t size;
   };
};

/* Monolithic sequence of instruction. Must live in a virtually contiguous
 * portion of code.
 */
struct cs_block {
   /* Used to insert the block in the block stack. */
   struct cs_block *next;
};

#define CS_LABEL_INVALID_POS ~0u

/* Labels can only be used inside a cs_block. They can be defined and
 * referenced before they are set to point to a specific position
 * in the block. */
struct cs_label {
   /* The last reference we have seen pointing to this block before
    * it was set. If set to CS_LABEL_INVALID_POS, no forward reference
    * pointing to this label exist.
    */
   uint32_t last_forward_ref;

   /* The label target. If set to CS_LABEL_INVALID_POS, the label has
    * not been set yet.
    */
   uint32_t target;
};

/* CS if/else block. */
struct cs_if_else {
   struct cs_block block;
   struct cs_label end_label;
   struct cs_load_store_tracker *orig_ls_state;
   struct cs_load_store_tracker ls_state;
};

struct cs_maybe {
   /* Link to the next pending cs_maybe for the block stack */
   struct cs_maybe *next_pending;
   /* Position of patch block relative to blocks.instrs */
   uint32_t patch_pos;

   /* CPU address of patch block in the chunk */
   uint64_t *patch_addr;
   /* Original contents of the patch block, before replacing with NOPs */
   uint32_t num_instrs;
   uint64_t instrs[];
};

struct cs_builder {
   /* CS builder configuration */
   struct cs_builder_conf conf;

   /* True if an allocation failed, making the whole CS invalid. */
   bool invalid;

   /* Initial (root) CS chunk. */
   struct cs_chunk root_chunk;

   /* Current CS chunk. */
   struct cs_chunk cur_chunk;

   /* Current load/store tracker. */
   struct cs_load_store_tracker *cur_ls_tracker;

   struct cs_load_store_tracker root_ls_tracker;

   /* ralloc context used for cs_maybe allocations */
   void *maybe_ctx;

   /* Temporary storage for inner blocks that need to be built
    * and copied in one monolithic sequence of instructions with no
    * jump in the middle.
    */
   struct {
      struct cs_block *stack;
      struct util_dynarray instrs;
      struct cs_if_else pending_if;
      /* Linked list of cs_maybe that were emitted inside the current stack */
      struct cs_maybe *pending_maybes;
      unsigned last_load_ip_target;
   } blocks;

   /* Move immediate instruction at the end of the last CS chunk that needs to
    * be patched with the final length of the current CS chunk in order to
    * facilitate correct overflow behaviour.
    */
   uint32_t *length_patch;

   /* Used as temporary storage when the allocator couldn't allocate a new
    * CS chunk.
    */
   uint64_t discard_instr_slot;
};

static inline void
cs_builder_init(struct cs_builder *b, const struct cs_builder_conf *conf,
                struct cs_buffer root_buffer)
{
   memset(b, 0, sizeof(*b));
   b->conf = *conf;
   b->root_chunk.buffer = root_buffer;
   b->cur_chunk.buffer = root_buffer;
   b->cur_ls_tracker = &b->root_ls_tracker,

   memset(b->cur_ls_tracker, 0, sizeof(*b->cur_ls_tracker));

   /* We need at least 3 registers for CS chunk linking. Assume the kernel needs
    * at least that too.
    */
   b->conf.nr_kernel_registers = MAX2(b->conf.nr_kernel_registers, 3);

   util_dynarray_init(&b->blocks.instrs, NULL);
}

static inline bool
cs_is_valid(struct cs_builder *b)
{
   return !b->invalid;
}

static inline bool
cs_is_empty(struct cs_builder *b)
{
   return b->cur_chunk.pos == 0 &&
          b->root_chunk.buffer.gpu == b->cur_chunk.buffer.gpu;
}

static inline uint64_t
cs_root_chunk_gpu_addr(struct cs_builder *b)
{
   return b->root_chunk.buffer.gpu;
}

static inline uint32_t
cs_root_chunk_size(struct cs_builder *b)
{
   /* Make sure cs_finish() was called. */
   struct cs_chunk empty_chunk;
   memset(&empty_chunk, 0, sizeof(empty_chunk));
   assert(!memcmp(&b->cur_chunk, &empty_chunk, sizeof(b->cur_chunk)));

   return b->root_chunk.size * sizeof(uint64_t);
}

/*
 * Wrap the current queue. External users shouldn't call this function
 * directly, they should call cs_finish() when they are done building
 * the command stream, which will in turn call cs_wrap_queue().
 *
 * Internally, this is also used to finalize internal CS chunks when
 * allocating new sub-chunks. See cs_alloc_chunk() for details.
 *
 * This notably requires patching the previous chunk with the length
 * we ended up emitting for this chunk.
 */
static inline void
cs_wrap_chunk(struct cs_builder *b)
{
   if (!cs_is_valid(b))
      return;

   if (b->length_patch) {
      *b->length_patch = (b->cur_chunk.pos * 8);
      b->length_patch = NULL;
   }

   if (b->root_chunk.buffer.gpu == b->cur_chunk.buffer.gpu)
      b->root_chunk.size = b->cur_chunk.size;
}

enum cs_index_type {
   CS_INDEX_REGISTER = 0,
   CS_INDEX_UNDEF,
};

struct cs_index {
   enum cs_index_type type;

   /* Number of 32-bit words in the index, must be nonzero */
   uint8_t size;

   union {
      uint64_t imm;
      uint8_t reg;
   };
};

static inline struct cs_index
cs_undef(void)
{
   return (struct cs_index){
      .type = CS_INDEX_UNDEF,
   };
}

static inline uint8_t
cs_to_reg_tuple(struct cs_index idx, ASSERTED unsigned expected_size)
{
   assert(idx.type == CS_INDEX_REGISTER);
   assert(idx.size == expected_size);

   return idx.reg;
}

static inline void cs_flush_load_to(struct cs_builder *b, struct cs_index to,
                                    uint16_t mask);

static inline unsigned
cs_src_tuple(struct cs_builder *b, struct cs_index src, ASSERTED unsigned count,
             uint16_t mask)
{
   unsigned reg = cs_to_reg_tuple(src, count);

   if (unlikely(b->conf.reg_perm)) {
      for (unsigned i = reg; i < reg + count; i++) {
         if (mask & BITFIELD_BIT(i - reg)) {
            assert((b->conf.reg_perm(b, i) & CS_REG_RD) ||
                   !"Trying to read a restricted register");
         }
      }
   }

   cs_flush_load_to(b, src, mask);

   return reg;
}

static inline unsigned
cs_src32(struct cs_builder *b, struct cs_index src)
{
   return cs_src_tuple(b, src, 1, BITFIELD_MASK(1));
}

static inline unsigned
cs_src64(struct cs_builder *b, struct cs_index src)
{
   return cs_src_tuple(b, src, 2, BITFIELD_MASK(2));
}

static inline unsigned
cs_dst_tuple(struct cs_builder *b, struct cs_index dst, ASSERTED unsigned count,
             uint16_t mask)
{
   unsigned reg = cs_to_reg_tuple(dst, count);

   /* A load followed by another op with the same dst register can overwrite
    * the result of that following op if there is no wait. For example:
    * load(dst, addr)
    * move(dst, v)
    */
   cs_flush_load_to(b, dst, mask);

   if (unlikely(b->conf.reg_perm)) {
      for (unsigned i = reg; i < reg + count; i++) {
         if (mask & BITFIELD_BIT(i - reg)) {
            assert((b->conf.reg_perm(b, i) & CS_REG_WR) ||
                   !"Trying to write a restricted register");
         }
      }
   }

   if (unlikely(b->conf.dirty_tracker)) {
      for (unsigned i = reg; i < reg + count; i++) {
         if (mask & BITFIELD_BIT(i - reg))
            BITSET_SET(b->conf.dirty_tracker->regs, i);
      }
   }

   return reg;
}

static inline unsigned
cs_dst32(struct cs_builder *b, struct cs_index dst)
{
   return cs_dst_tuple(b, dst, 1, BITFIELD_MASK(1));
}

static inline unsigned
cs_dst64(struct cs_builder *b, struct cs_index dst)
{
   return cs_dst_tuple(b, dst, 2, BITFIELD_MASK(2));
}

static inline struct cs_index
cs_reg_tuple(ASSERTED struct cs_builder *b, uint8_t reg, uint8_t size)
{
   assert(reg + size <= b->conf.nr_registers - b->conf.nr_kernel_registers &&
          "overflowed register file");
   assert(size <= 16 && "unsupported");

   return (struct cs_index){
      .type = CS_INDEX_REGISTER,
      .size = size,
      .reg = reg,
   };
}

static inline struct cs_index
cs_reg32(struct cs_builder *b, unsigned reg)
{
   return cs_reg_tuple(b, reg, 1);
}

static inline struct cs_index
cs_reg64(struct cs_builder *b, unsigned reg)
{
   assert((reg % 2) == 0 && "unaligned 64-bit reg");
   return cs_reg_tuple(b, reg, 2);
}

#define cs_sr_reg_tuple(__b, __cmd, __name, __size)                            \
   cs_reg_tuple((__b), MALI_##__cmd##_SR_##__name, (__size))
#define cs_sr_reg32(__b, __cmd, __name)                                        \
   cs_reg32((__b), MALI_##__cmd##_SR_##__name)
#define cs_sr_reg64(__b, __cmd, __name)                                        \
   cs_reg64((__b), MALI_##__cmd##_SR_##__name)

/*
 * The top of the register file is reserved for cs_builder internal use. We
 * need 3 spare registers for handling command queue overflow. These are
 * available here.
 */
static inline uint8_t
cs_overflow_address_reg(struct cs_builder *b)
{
   return b->conf.nr_registers - 2;
}

static inline uint8_t
cs_overflow_length_reg(struct cs_builder *b)
{
   return b->conf.nr_registers - 3;
}

static inline struct cs_index
cs_extract32(struct cs_builder *b, struct cs_index idx, unsigned word)
{
   assert(idx.type == CS_INDEX_REGISTER && "unsupported");
   assert(word < idx.size && "overrun");

   return cs_reg32(b, idx.reg + word);
}

static inline struct cs_index
cs_extract64(struct cs_builder *b, struct cs_index idx, unsigned word)
{
   assert(idx.type == CS_INDEX_REGISTER && "unsupported");
   assert(word + 1 < idx.size && "overrun");

   return cs_reg64(b, idx.reg + word);
}

static inline struct cs_index
cs_extract_tuple(struct cs_builder *b, struct cs_index idx, unsigned word,
                 unsigned size)
{
   assert(idx.type == CS_INDEX_REGISTER && "unsupported");
   assert(word + size < idx.size && "overrun");

   return cs_reg_tuple(b, idx.reg + word, size);
}

static inline struct cs_block *
cs_cur_block(struct cs_builder *b)
{
   return b->blocks.stack;
}

#define JUMP_SEQ_INSTR_COUNT 4

static inline bool
cs_reserve_instrs(struct cs_builder *b, uint32_t num_instrs)
{
   /* Don't call this function with num_instrs=0. */
   assert(num_instrs > 0);
   assert(cs_cur_block(b) == NULL);

   /* If an allocation failure happened before, we just discard all following
    * instructions.
    */
   if (unlikely(!cs_is_valid(b)))
      return false;

   /* Make sure we have sufficient capacity if we wont allocate more */
   if (b->conf.alloc_buffer == NULL) {
      if (unlikely(b->cur_chunk.size + num_instrs > b->cur_chunk.buffer.capacity)) {
         assert(!"Out of CS space");
         b->invalid = true;
         return false;
      }

      return true;
   }

   /* Lazy root chunk allocation. */
   if (unlikely(!b->root_chunk.buffer.cpu)) {
      b->root_chunk.buffer = b->conf.alloc_buffer(b->conf.cookie);
      b->cur_chunk.buffer = b->root_chunk.buffer;
      if (!b->cur_chunk.buffer.cpu) {
         b->invalid = true;
         return false;
      }
   }

   /* Make sure the instruction sequence fits in a single chunk. */
   assert(b->cur_chunk.buffer.capacity >= num_instrs);

   /* If the current chunk runs out of space, allocate a new one and jump to it.
    * We actually do this a few instructions before running out, because the
    * sequence to jump to a new queue takes multiple instructions.
    */
   bool jump_to_next_chunk =
      (b->cur_chunk.size + num_instrs + JUMP_SEQ_INSTR_COUNT) >
      b->cur_chunk.buffer.capacity;
   if (unlikely(jump_to_next_chunk)) {
      /* Now, allocate a new chunk */
      struct cs_buffer newbuf = b->conf.alloc_buffer(b->conf.cookie);

      /* Allocation failure, from now on, all new instructions will be
       * discarded.
       */
      if (unlikely(!newbuf.cpu)) {
         b->invalid = true;
         return false;
      }

      uint64_t *ptr = b->cur_chunk.buffer.cpu + (b->cur_chunk.pos++);

      pan_cast_and_pack(ptr, CS_MOVE48, I) {
         I.destination = cs_overflow_address_reg(b);
         I.immediate = newbuf.gpu;
      }

      ptr = b->cur_chunk.buffer.cpu + (b->cur_chunk.pos++);

      pan_cast_and_pack(ptr, CS_MOVE32, I) {
         I.destination = cs_overflow_length_reg(b);
      }

      /* The length will be patched in later */
      uint32_t *length_patch = (uint32_t *)ptr;

      ptr = b->cur_chunk.buffer.cpu + (b->cur_chunk.pos++);

      pan_cast_and_pack(ptr, CS_JUMP, I) {
         I.length = cs_overflow_length_reg(b);
         I.address = cs_overflow_address_reg(b);
      }

      /* Now that we've emitted everything, finish up the previous queue */
      cs_wrap_chunk(b);

      /* And make this one current */
      b->length_patch = length_patch;
      b->cur_chunk.buffer = newbuf;
      b->cur_chunk.pos = 0;
   }

   return true;
}

static inline void *
cs_alloc_ins_block(struct cs_builder *b, uint32_t num_instrs)
{
   if (cs_cur_block(b))
      return util_dynarray_grow(&b->blocks.instrs, uint64_t, num_instrs);

   if (!cs_reserve_instrs(b, num_instrs))
      return NULL;

   assert(b->cur_chunk.size + num_instrs - 1 < b->cur_chunk.buffer.capacity);
   uint32_t pos = b->cur_chunk.pos;
   b->cur_chunk.pos += num_instrs;
   return b->cur_chunk.buffer.cpu + pos;
}

static inline void
cs_flush_block_instrs(struct cs_builder *b)
{
   if (cs_cur_block(b) != NULL)
      return;

   uint32_t num_instrs =
      util_dynarray_num_elements(&b->blocks.instrs, uint64_t);
   if (!num_instrs)
      return;

   /* If LOAD_IP is the last instruction in the block, we reserve one more
    * slot to make sure the next instruction won't point to a CS chunk linking
    * sequence. */
   if (unlikely(b->blocks.last_load_ip_target >= num_instrs)) {
      if (!cs_reserve_instrs(b, num_instrs + 1))
         return;
   }

   void *buffer = cs_alloc_ins_block(b, num_instrs);

   if (likely(buffer != NULL)) {
      /* We wait until block instrs are copied to the chunk buffer to calculate
       * patch_addr, in case we end up allocating a new chunk */
      while (b->blocks.pending_maybes) {
         b->blocks.pending_maybes->patch_addr =
            (uint64_t *) buffer + b->blocks.pending_maybes->patch_pos;
         b->blocks.pending_maybes = b->blocks.pending_maybes->next_pending;
      }

      /* If we have a LOAD_IP chain, we need to patch each LOAD_IP
       * instruction before we copy the block to the final memory
       * region. */
      while (unlikely(b->blocks.last_load_ip_target)) {
         uint64_t *instr = util_dynarray_element(
            &b->blocks.instrs, uint64_t, b->blocks.last_load_ip_target - 1);
         unsigned prev_load_ip_target = *instr & BITFIELD_MASK(32);
         uint64_t ip =
            b->cur_chunk.buffer.gpu +
            ((b->cur_chunk.pos - num_instrs + b->blocks.last_load_ip_target) *
             sizeof(uint64_t));

         /* Drop the prev_load_ip_target value and replace it by the final
          * IP. */
         *instr &= ~BITFIELD64_MASK(32);
         *instr |= ip;

         b->blocks.last_load_ip_target = prev_load_ip_target;
      }

      memcpy(buffer, b->blocks.instrs.data, b->blocks.instrs.size);
   }

   util_dynarray_clear(&b->blocks.instrs);
}

static inline uint32_t
cs_block_next_pos(struct cs_builder *b)
{
   assert(cs_cur_block(b) != NULL);

   return util_dynarray_num_elements(&b->blocks.instrs, uint64_t);
}

static inline void
cs_label_init(struct cs_label *label)
{
   label->last_forward_ref = CS_LABEL_INVALID_POS;
   label->target = CS_LABEL_INVALID_POS;
}

static inline void
cs_set_label(struct cs_builder *b, struct cs_label *label)
{
   assert(label->target == CS_LABEL_INVALID_POS);
   label->target = cs_block_next_pos(b);

   for (uint32_t next_forward_ref, forward_ref = label->last_forward_ref;
        forward_ref != CS_LABEL_INVALID_POS; forward_ref = next_forward_ref) {
      uint64_t *ins =
         util_dynarray_element(&b->blocks.instrs, uint64_t, forward_ref);

      assert(forward_ref < label->target);
      assert(label->target - forward_ref <= INT16_MAX);

      /* Save the next forward reference to this target before overwritting
       * it with the final offset.
       */
      int16_t offset = *ins & BITFIELD64_MASK(16);

      next_forward_ref =
         offset > 0 ? forward_ref - offset : CS_LABEL_INVALID_POS;

      assert(next_forward_ref == CS_LABEL_INVALID_POS ||
             next_forward_ref < forward_ref);

      *ins &= ~BITFIELD64_MASK(16);
      *ins |= label->target - forward_ref - 1;
   }
}

static inline void
cs_flush_pending_if(struct cs_builder *b)
{
   if (likely(cs_cur_block(b) != &b->blocks.pending_if.block))
      return;

   cs_set_label(b, &b->blocks.pending_if.end_label);
   b->blocks.stack = b->blocks.pending_if.block.next;
   cs_flush_block_instrs(b);
}

static inline void *
cs_alloc_ins(struct cs_builder *b)
{
   /* If an instruction is emitted after an if_end(), it flushes the pending if,
    * causing further cs_else_start() instructions to be invalid. */
   cs_flush_pending_if(b);

   return cs_alloc_ins_block(b, 1) ?: &b->discard_instr_slot;
}

/* Call this when you are done building a command stream and want to prepare
 * it for submission.
 */
static inline void
cs_finish(struct cs_builder *b)
{
   if (!cs_is_valid(b))
      return;

   cs_flush_pending_if(b);
   cs_wrap_chunk(b);

   /* This prevents adding instructions after that point. */
   memset(&b->cur_chunk, 0, sizeof(b->cur_chunk));

   util_dynarray_fini(&b->blocks.instrs);
   ralloc_free(b->maybe_ctx);
}

/*
 * Helper to emit a new instruction into the command queue. The allocation needs
 * to be separated out being pan_pack can evaluate its argument multiple times,
 * yet cs_alloc has side effects.
 */
#define cs_emit(b, T, cfg) pan_cast_and_pack(cs_alloc_ins(b), CS_##T, cfg)

/* Asynchronous operations take a mask of scoreboard slots to wait on
 * before executing the instruction, and signal a scoreboard slot when
 * the operation is complete.
 * On v11 and later, asynchronous operations can also wait on a scoreboard
 * mask and signal a scoreboard slot indirectly instead (set via SET_STATE)
 * A wait_mask of zero means the operation is synchronous, and signal_slot
 * is ignored in that case.
 */
struct cs_async_op {
   uint16_t wait_mask;
   uint8_t signal_slot;
#if PAN_ARCH >= 11
   bool indirect;
#endif
};

static inline struct cs_async_op
cs_defer(uint16_t wait_mask, uint8_t signal_slot)
{
   /* The scoreboard slot to signal is incremented before the wait operation,
    * waiting on it would cause an infinite wait.
    */
   assert(!(wait_mask & BITFIELD_BIT(signal_slot)));

   return (struct cs_async_op){
      .wait_mask = wait_mask,
      .signal_slot = signal_slot,
   };
}

static inline struct cs_async_op
cs_now(void)
{
   return (struct cs_async_op){
      .wait_mask = 0,
      .signal_slot = 0xff,
   };
}

#if PAN_ARCH >= 11
static inline struct cs_async_op
cs_defer_indirect(void)
{
   return (struct cs_async_op){
      .wait_mask = 0xff,
      .signal_slot = 0xff,
      .indirect = true,
   };
}
#endif

static inline bool
cs_instr_is_asynchronous(enum mali_cs_opcode opcode, uint16_t wait_mask)
{
   switch (opcode) {
   case MALI_CS_OPCODE_FLUSH_CACHE2:
   case MALI_CS_OPCODE_FINISH_TILING:
   case MALI_CS_OPCODE_LOAD_MULTIPLE:
   case MALI_CS_OPCODE_STORE_MULTIPLE:
   case MALI_CS_OPCODE_RUN_COMPUTE:
   case MALI_CS_OPCODE_RUN_COMPUTE_INDIRECT:
   case MALI_CS_OPCODE_RUN_FRAGMENT:
   case MALI_CS_OPCODE_RUN_FULLSCREEN:
#if PAN_ARCH >= 12
   case MALI_CS_OPCODE_RUN_IDVS2:
#else
   case MALI_CS_OPCODE_RUN_IDVS:
#if PAN_ARCH == 10
   case MALI_CS_OPCODE_RUN_TILING:
#endif
#endif

      /* Always asynchronous. */
      return true;

   case MALI_CS_OPCODE_FINISH_FRAGMENT:
   case MALI_CS_OPCODE_SYNC_ADD32:
   case MALI_CS_OPCODE_SYNC_SET32:
   case MALI_CS_OPCODE_SYNC_ADD64:
   case MALI_CS_OPCODE_SYNC_SET64:
   case MALI_CS_OPCODE_STORE_STATE:
   case MALI_CS_OPCODE_TRACE_POINT:
   case MALI_CS_OPCODE_HEAP_OPERATION:
#if PAN_ARCH >= 11
   case MALI_CS_OPCODE_SHARED_SB_INC:
#endif
      /* Asynchronous only if wait_mask != 0. */
      return wait_mask != 0;

   default:
      return false;
   }
}

/* TODO: was the signal_slot comparison bugged? */
#if PAN_ARCH == 10
#define cs_apply_async(I, async)                                               \
   do {                                                                        \
      I.wait_mask = async.wait_mask;                                           \
      I.signal_slot = cs_instr_is_asynchronous(I.opcode, I.wait_mask)          \
                         ? async.signal_slot                                   \
                         : 0;                                                  \
      assert(I.signal_slot != 0xff ||                                          \
             !"Can't use cs_now() on pure async instructions");                \
   } while (0)
#else
#define cs_apply_async(I, async)                                               \
   do {                                                                        \
      if (async.indirect) {                                                    \
         I.defer_mode = MALI_CS_DEFER_MODE_DEFER_INDIRECT;                     \
      } else {                                                                 \
         I.defer_mode = MALI_CS_DEFER_MODE_DEFER_IMMEDIATE;                    \
         I.wait_mask = async.wait_mask;                                        \
         I.signal_slot = cs_instr_is_asynchronous(I.opcode, I.wait_mask)       \
                            ? async.signal_slot                                \
                            : 0;                                               \
         assert(I.signal_slot != 0xff ||                                       \
                !"Can't use cs_now() on pure async instructions");             \
      }                                                                        \
   } while (0)
#endif

static inline void
cs_move32_to(struct cs_builder *b, struct cs_index dest, unsigned imm)
{
   cs_emit(b, MOVE32, I) {
      I.destination = cs_dst32(b, dest);
      I.immediate = imm;
   }
}

static inline void
cs_move48_to(struct cs_builder *b, struct cs_index dest, uint64_t imm)
{
   cs_emit(b, MOVE48, I) {
      I.destination = cs_dst64(b, dest);
      I.immediate = imm;
   }
}

static inline void
cs_load_ip_to(struct cs_builder *b, struct cs_index dest)
{
   /* If a load_ip instruction is emitted after an if_end(), it flushes the
    * pending if, causing further cs_else_start() instructions to be invalid.
    */
   cs_flush_pending_if(b);

   if (likely(cs_cur_block(b) == NULL)) {
      if (!cs_reserve_instrs(b, 2))
         return;

      /* We make IP point to the instruction right after our MOVE. */
      uint64_t ip =
         b->cur_chunk.buffer.gpu + (sizeof(uint64_t) * (b->cur_chunk.pos + 1));
      cs_move48_to(b, dest, ip);
   } else {
      cs_move48_to(b, dest, b->blocks.last_load_ip_target);
      b->blocks.last_load_ip_target =
         util_dynarray_num_elements(&b->blocks.instrs, uint64_t);
   }
}

static inline void
cs_wait_slots(struct cs_builder *b, unsigned wait_mask)
{
   struct cs_load_store_tracker *ls_tracker = b->cur_ls_tracker;
   assert(ls_tracker != NULL);

   cs_emit(b, WAIT, I) {
      I.wait_mask = wait_mask;
   }

   /* We don't do advanced tracking of cs_defer(), and assume that
    * load/store will be flushed with an explicit wait on the load/store
    * scoreboard. */
   if (wait_mask & BITFIELD_BIT(b->conf.ls_sb_slot)) {
      BITSET_CLEAR_RANGE(ls_tracker->pending_loads, 0, 255);
      ls_tracker->pending_stores = false;
   }
}

static inline void
cs_wait_slot(struct cs_builder *b, unsigned slot)
{
   assert(slot < CS_MAX_SB_COUNT && "invalid slot");

   cs_wait_slots(b, BITFIELD_BIT(slot));
}

static inline void
cs_flush_load_to(struct cs_builder *b, struct cs_index to, uint16_t mask)
{
   struct cs_load_store_tracker *ls_tracker = b->cur_ls_tracker;
   assert(ls_tracker != NULL);

   unsigned count = util_last_bit(mask);
   unsigned reg = cs_to_reg_tuple(to, count);

   for (unsigned i = reg; i < reg + count; i++) {
      if ((mask & BITFIELD_BIT(i - reg)) &&
          BITSET_TEST(ls_tracker->pending_loads, i)) {
         cs_wait_slots(b, BITFIELD_BIT(b->conf.ls_sb_slot));
         break;
      }
   }
}

static inline void
cs_flush_loads(struct cs_builder *b)
{
   struct cs_load_store_tracker *ls_tracker = b->cur_ls_tracker;
   assert(ls_tracker != NULL);

   if (!BITSET_IS_EMPTY(ls_tracker->pending_loads))
      cs_wait_slots(b, BITFIELD_BIT(b->conf.ls_sb_slot));
}

static inline void
cs_flush_stores(struct cs_builder *b)
{
   struct cs_load_store_tracker *ls_tracker = b->cur_ls_tracker;
   assert(ls_tracker != NULL);

   if (ls_tracker->pending_stores)
      cs_wait_slots(b, BITFIELD_BIT(b->conf.ls_sb_slot));
}

static inline void
cs_block_start(struct cs_builder *b, struct cs_block *block)
{
   cs_flush_pending_if(b);
   block->next = b->blocks.stack;
   b->blocks.stack = block;
}

static inline void
cs_block_end(struct cs_builder *b, struct cs_block *block)
{
   cs_flush_pending_if(b);

   assert(cs_cur_block(b) == block);

   b->blocks.stack = block->next;

   cs_flush_block_instrs(b);
}

static inline void
cs_branch(struct cs_builder *b, int offset, enum mali_cs_condition cond,
          struct cs_index val)
{
   cs_emit(b, BRANCH, I) {
      I.offset = offset;
      I.condition = cond;
      I.value = cs_src32(b, val);
   }
}

static inline void
cs_branch_label_cond32(struct cs_builder *b, struct cs_label *label,
                       enum mali_cs_condition cond, struct cs_index val)
{
   assert(cs_cur_block(b) != NULL);

   /* Call cs_src before cs_block_next_pos because cs_src can emit an extra
    * WAIT instruction if there is a pending load.
    */
   uint32_t val_reg = cond != MALI_CS_CONDITION_ALWAYS ? cs_src32(b, val) : 0;

   if (label->target == CS_LABEL_INVALID_POS) {
      uint32_t branch_ins_pos = cs_block_next_pos(b);

      /* Instead of emitting a BRANCH with the final offset, we record the
       * diff between the current branch, and the previous branch that was
       * referencing this unset label. This way we build a single link list
       * that can be walked when the label is set with cs_set_label().
       * We use -1 as the end-of-list marker.
       */
      int16_t offset = -1;
      if (label->last_forward_ref != CS_LABEL_INVALID_POS) {
         assert(label->last_forward_ref < branch_ins_pos);
         assert(branch_ins_pos - label->last_forward_ref <= INT16_MAX);
         offset = branch_ins_pos - label->last_forward_ref;
      }

      cs_emit(b, BRANCH, I) {
         I.offset = offset;
         I.condition = cond;
         I.value = val_reg;
      }

      label->last_forward_ref = branch_ins_pos;
   } else {
      int32_t offset = label->target - cs_block_next_pos(b) - 1;

      /* The branch target is encoded in a 16-bit signed integer, make sure we
       * don't underflow.
       */
      assert(offset >= INT16_MIN);

      /* Backward references are easy, we can emit them immediately. */
      cs_emit(b, BRANCH, I) {
         I.offset = offset;
         I.condition = cond;
         I.value = val_reg;
      }
   }
}

static inline enum mali_cs_condition
cs_invert_cond(enum mali_cs_condition cond)
{
   switch (cond) {
   case MALI_CS_CONDITION_LEQUAL:
      return MALI_CS_CONDITION_GREATER;
   case MALI_CS_CONDITION_EQUAL:
      return MALI_CS_CONDITION_NEQUAL;
   case MALI_CS_CONDITION_LESS:
      return MALI_CS_CONDITION_GEQUAL;
   case MALI_CS_CONDITION_GREATER:
      return MALI_CS_CONDITION_LEQUAL;
   case MALI_CS_CONDITION_NEQUAL:
      return MALI_CS_CONDITION_EQUAL;
   case MALI_CS_CONDITION_GEQUAL:
      return MALI_CS_CONDITION_LESS;
   case MALI_CS_CONDITION_ALWAYS:
      UNREACHABLE("cannot invert ALWAYS");
   default:
      UNREACHABLE("invalid cond");
   }
}

static inline void
cs_branch_label_cond64(struct cs_builder *b, struct cs_label *label,
                       enum mali_cs_condition cond, struct cs_index val)
{
   struct cs_label false_label;
   cs_label_init(&false_label);

   struct cs_index val_lo = cs_extract32(b, val, 0);
   struct cs_index val_hi = cs_extract32(b, val, 1);

   switch (cond) {
   case MALI_CS_CONDITION_ALWAYS:
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_ALWAYS, cs_undef());
      break;
   case MALI_CS_CONDITION_LEQUAL:
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_LESS, val_hi);
      cs_branch_label_cond32(b, &false_label, MALI_CS_CONDITION_NEQUAL, val_hi);
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_EQUAL, val_lo);
      break;
   case MALI_CS_CONDITION_GREATER:
      cs_branch_label_cond32(b, &false_label, MALI_CS_CONDITION_LESS, val_hi);
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_NEQUAL, val_hi);
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_NEQUAL, val_lo);
      break;
   case MALI_CS_CONDITION_GEQUAL:
      cs_branch_label_cond32(b, &false_label, MALI_CS_CONDITION_LESS, val_hi);
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_ALWAYS, cs_undef());
      break;
   case MALI_CS_CONDITION_LESS:
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_LESS, val_hi);
      break;
   case MALI_CS_CONDITION_NEQUAL:
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_NEQUAL, val_lo);
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_NEQUAL, val_hi);
      break;
   case MALI_CS_CONDITION_EQUAL:
      cs_branch_label_cond32(b, &false_label, MALI_CS_CONDITION_NEQUAL, val_lo);
      cs_branch_label_cond32(b, label, MALI_CS_CONDITION_EQUAL, val_hi);
      break;
   default:
      UNREACHABLE("unsupported 64bit condition");
   }

   cs_set_label(b, &false_label);
}

static inline void
cs_branch_label(struct cs_builder *b, struct cs_label *label,
                enum mali_cs_condition cond, struct cs_index val)
{
   if (val.size == 2) {
      cs_branch_label_cond64(b, label, cond, val);
   } else {
      cs_branch_label_cond32(b, label, cond, val);
   }
}

static inline struct cs_if_else *
cs_if_start(struct cs_builder *b, struct cs_if_else *if_else,
            enum mali_cs_condition cond, struct cs_index val)
{
   cs_block_start(b, &if_else->block);
   cs_label_init(&if_else->end_label);
   cs_branch_label(b, &if_else->end_label, cs_invert_cond(cond), val);

   if_else->orig_ls_state = b->cur_ls_tracker;
   if_else->ls_state = *if_else->orig_ls_state;
   b->cur_ls_tracker = &if_else->ls_state;

   return if_else;
}

static inline void
cs_if_end(struct cs_builder *b, struct cs_if_else *if_else)
{
   assert(cs_cur_block(b) == &if_else->block);

   b->blocks.pending_if.block.next = if_else->block.next;
   b->blocks.stack = &b->blocks.pending_if.block;
   b->blocks.pending_if.end_label = if_else->end_label;

   b->blocks.pending_if.orig_ls_state = if_else->orig_ls_state;
   b->blocks.pending_if.ls_state = if_else->ls_state;

   BITSET_OR(if_else->orig_ls_state->pending_loads,
             if_else->orig_ls_state->pending_loads,
             if_else->ls_state.pending_loads);
   if_else->orig_ls_state->pending_stores |= if_else->ls_state.pending_stores;
   b->cur_ls_tracker = if_else->orig_ls_state;
}

static inline struct cs_if_else *
cs_else_start(struct cs_builder *b, struct cs_if_else *if_else)
{
   assert(cs_cur_block(b) == &b->blocks.pending_if.block);

   if_else->block.next = b->blocks.pending_if.block.next;
   b->blocks.stack = &if_else->block;
   cs_label_init(&if_else->end_label);
   cs_branch_label(b, &if_else->end_label, MALI_CS_CONDITION_ALWAYS,
                   cs_undef());
   cs_set_label(b, &b->blocks.pending_if.end_label);
   cs_label_init(&b->blocks.pending_if.end_label);

   /* Restore the ls_tracker state from before the if block. */
   if_else->orig_ls_state = b->blocks.pending_if.orig_ls_state;
   if_else->ls_state = *if_else->orig_ls_state;
   b->cur_ls_tracker = &if_else->ls_state;

   return if_else;
}

static inline void
cs_else_end(struct cs_builder *b, struct cs_if_else *if_else)
{
   struct cs_load_store_tracker if_ls_state = b->blocks.pending_if.ls_state;

   cs_set_label(b, &if_else->end_label);
   cs_block_end(b, &if_else->block);

   BITSET_OR(if_else->orig_ls_state->pending_loads, if_ls_state.pending_loads,
             if_else->ls_state.pending_loads);
   if_else->orig_ls_state->pending_stores =
      if_ls_state.pending_stores | if_else->ls_state.pending_stores;
   b->cur_ls_tracker = if_else->orig_ls_state;
}

#define cs_if(__b, __cond, __val)                                              \
   for (struct cs_if_else __storage,                                           \
        *__if_else = cs_if_start(__b, &__storage, __cond, __val);              \
        __if_else != NULL; cs_if_end(__b, __if_else), __if_else = NULL)

#define cs_else(__b)                                                           \
   for (struct cs_if_else __storage,                                           \
        *__if_else = cs_else_start(__b, &__storage);                           \
        __if_else != NULL; cs_else_end(__b, __if_else), __if_else = NULL)

struct cs_loop {
   struct cs_label start, end;
   struct cs_block block;
   enum mali_cs_condition cond;
   struct cs_index val;
   struct cs_load_store_tracker *orig_ls_state;
   /* On continue we need to compare the original loads to the current ones.
    * orig_ls_state can get updated from inside the loop. */
   struct cs_load_store_tracker orig_ls_state_copy;
   struct cs_load_store_tracker ls_state;
};

static inline void
cs_loop_diverge_ls_update(struct cs_builder *b, struct cs_loop *loop)
{
   assert(loop->orig_ls_state);

   BITSET_OR(loop->orig_ls_state->pending_loads,
             loop->orig_ls_state->pending_loads,
             b->cur_ls_tracker->pending_loads);
   loop->orig_ls_state->pending_stores |= b->cur_ls_tracker->pending_stores;
}

static inline bool
cs_loop_continue_need_flush(struct cs_builder *b, struct cs_loop *loop)
{
   assert(loop->orig_ls_state);

   /* We need to flush on continue/loop-again, if there are loads to registers
    * that did not already have loads in-flight before the loop.
    * Those would have already gotten WAITs inserted because they were marked
    * already.
    */
   BITSET_DECLARE(new_pending_loads, 256);
   BITSET_ANDNOT(new_pending_loads, b->cur_ls_tracker->pending_loads,
                 loop->orig_ls_state_copy.pending_loads);
   return !BITSET_IS_EMPTY(new_pending_loads);
}

static inline struct cs_loop *
cs_loop_init(struct cs_builder *b, struct cs_loop *loop,
             enum mali_cs_condition cond, struct cs_index val)
{
   *loop = (struct cs_loop){
      .cond = cond,
      .val = val,
   };

   cs_block_start(b, &loop->block);
   cs_label_init(&loop->start);
   cs_label_init(&loop->end);
   return loop;
}

static inline struct cs_loop *
cs_while_start(struct cs_builder *b, struct cs_loop *loop,
               enum mali_cs_condition cond, struct cs_index val)
{
   cs_loop_init(b, loop, cond, val);

   /* Do an initial check on the condition, and if it's false, jump to
    * the end of the loop block. For 'while(true)' loops, skip the
    * conditional branch.
    */
   if (cond != MALI_CS_CONDITION_ALWAYS)
      cs_branch_label(b, &loop->end, cs_invert_cond(cond), val);

   /* The loops ls tracker only needs to track the actual loop body, not the
    * check to skip the whole body. */
   loop->orig_ls_state = b->cur_ls_tracker;
   loop->orig_ls_state_copy = *loop->orig_ls_state;
   loop->ls_state = *loop->orig_ls_state;
   b->cur_ls_tracker = &loop->ls_state;

   cs_set_label(b, &loop->start);

   return loop;
}

static inline void
cs_loop_conditional_continue(struct cs_builder *b, struct cs_loop *loop,
                             enum mali_cs_condition cond, struct cs_index val)
{
   cs_flush_pending_if(b);

   if (cs_loop_continue_need_flush(b, loop))
      cs_flush_loads(b);

   cs_branch_label(b, &loop->start, cond, val);
   cs_loop_diverge_ls_update(b, loop);
}

static inline void
cs_loop_conditional_break(struct cs_builder *b, struct cs_loop *loop,
                          enum mali_cs_condition cond, struct cs_index val)
{
   cs_flush_pending_if(b);
   cs_branch_label(b, &loop->end, cond, val);
   cs_loop_diverge_ls_update(b, loop);
}

static inline void
cs_while_end(struct cs_builder *b, struct cs_loop *loop)
{
   cs_flush_pending_if(b);

   if (cs_loop_continue_need_flush(b, loop))
      cs_flush_loads(b);

   cs_branch_label(b, &loop->start, loop->cond, loop->val);
   cs_set_label(b, &loop->end);
   cs_block_end(b, &loop->block);

   if (unlikely(loop->orig_ls_state)) {
      BITSET_OR(loop->orig_ls_state->pending_loads,
                loop->orig_ls_state->pending_loads,
                loop->ls_state.pending_loads);
      loop->orig_ls_state->pending_stores |= loop->ls_state.pending_stores;
      b->cur_ls_tracker = loop->orig_ls_state;
   }
}

#define cs_while(__b, __cond, __val)                                           \
   for (struct cs_loop __loop_storage,                                         \
        *__loop = cs_while_start(__b, &__loop_storage, __cond, __val);         \
        __loop != NULL; cs_while_end(__b, __loop), __loop = NULL)

#define cs_continue(__b)                                                       \
   cs_loop_conditional_continue(__b, __loop, MALI_CS_CONDITION_ALWAYS,         \
                                cs_undef())

#define cs_break(__b)                                                          \
   cs_loop_conditional_break(__b, __loop, MALI_CS_CONDITION_ALWAYS, cs_undef())

/* cs_maybe is an abstraction for retroactively patching cs contents. When the
 * block is closed, its original contents are recorded and then replaced with
 * NOP instructions. The caller can then use the cs_patch_maybe to restore the
 * original contents at a later point. This can be useful in situations where
 * not enough information is available during recording to determine what
 * instructions should be emitted at the time, but it will be known at some
 * point before submission. */

struct cs_maybe_state {
   struct cs_block block;
   uint32_t patch_pos;
   struct cs_load_store_tracker ls_state;
   struct cs_load_store_tracker *orig_ls_state;
};

static inline struct cs_maybe_state *
cs_maybe_start(struct cs_builder *b, struct cs_maybe_state *state)
{
   cs_block_start(b, &state->block);
   state->patch_pos = cs_block_next_pos(b);

   /* store the original ls_tracker state so that we can revert to it after
    * the maybe-block */
   state->orig_ls_state = b->cur_ls_tracker;
   state->ls_state = *b->cur_ls_tracker;
   b->cur_ls_tracker = &state->ls_state;

   return state;
}

static inline void
cs_maybe_end(struct cs_builder *b, struct cs_maybe_state *state,
             struct cs_maybe **maybe)
{
   assert(cs_cur_block(b) == &state->block);

   /* Flush any new loads and stores */
   BITSET_DECLARE(new_loads, 256);
   BITSET_ANDNOT(new_loads, b->cur_ls_tracker->pending_loads,
                 state->orig_ls_state->pending_loads);
   bool new_stores =
      b->cur_ls_tracker->pending_stores && !state->orig_ls_state->pending_stores;
   if (!BITSET_IS_EMPTY(new_loads) || new_stores)
      cs_wait_slots(b, BITFIELD_BIT(b->conf.ls_sb_slot));

   /* Restore the original ls tracker state */
   b->cur_ls_tracker = state->orig_ls_state;

   uint32_t num_instrs = cs_block_next_pos(b) - state->patch_pos;
   size_t size = num_instrs * sizeof(uint64_t);
   uint64_t *instrs = (uint64_t *) b->blocks.instrs.data + state->patch_pos;

   if (!b->maybe_ctx)
      b->maybe_ctx = ralloc_context(NULL);

   *maybe = (struct cs_maybe *)
      ralloc_size(b->maybe_ctx, sizeof(struct cs_maybe) + size);
   (*maybe)->next_pending = b->blocks.pending_maybes;
   b->blocks.pending_maybes = *maybe;
   (*maybe)->patch_pos = state->patch_pos;
   (*maybe)->num_instrs = num_instrs;
   /* patch_addr will be computed later in cs_flush_block_instrs, when the
    * outermost block is closed */
   (*maybe)->patch_addr = NULL;

   /* Save the emitted instructions in the patch block */
   memcpy((*maybe)->instrs, instrs, size);
   /* Replace instructions in the patch block with NOPs */
   memset(instrs, 0, size);

   cs_block_end(b, &state->block);
}

#define cs_maybe(__b, __maybe)                                                 \
   for (struct cs_maybe_state __storage,                                       \
        *__state = cs_maybe_start(__b, &__storage);                            \
        __state != NULL; cs_maybe_end(__b, __state, __maybe),                  \
        __state = NULL)

/* Must be called before cs_finish */
static inline void
cs_patch_maybe(struct cs_builder *b, struct cs_maybe *maybe)
{
   if (maybe->patch_addr) {
      /* Called after outer block was closed */
      memcpy(maybe->patch_addr, maybe->instrs,
             maybe->num_instrs * sizeof(uint64_t));
   } else {
      /* Called before outer block was closed */
      memcpy((uint64_t *)b->blocks.instrs.data + maybe->patch_pos,
             maybe->instrs, maybe->num_instrs * sizeof(uint64_t));
   }
}

struct cs_single_link_list_node {
   uint64_t next;
};

struct cs_single_link_list {
   uint64_t head;
   uint64_t tail;
};

/* Pseudoinstructions follow */

static inline void
cs_move64_to(struct cs_builder *b, struct cs_index dest, uint64_t imm)
{
   if (imm < (1ull << 48)) {
      /* Zero extends */
      cs_move48_to(b, dest, imm);
   } else {
      cs_move32_to(b, cs_extract32(b, dest, 0), imm);
      cs_move32_to(b, cs_extract32(b, dest, 1), imm >> 32);
   }
}

struct cs_shader_res_sel {
   uint8_t srt, fau, spd, tsd;
};

static inline struct cs_shader_res_sel
cs_shader_res_sel(uint8_t srt, uint8_t fau, uint8_t spd, uint8_t tsd)
{
   return (struct cs_shader_res_sel){
      .srt = srt,
      .fau = fau,
      .spd = spd,
      .tsd = tsd,
   };
}

static inline void
cs_run_compute(struct cs_builder *b, unsigned task_increment,
               enum mali_task_axis task_axis, struct cs_shader_res_sel res_sel)
{
   /* Staging regs */
   cs_flush_loads(b);

   cs_emit(b, RUN_COMPUTE, I) {
      I.task_increment = task_increment;
      I.task_axis = task_axis;
      I.srt_select = res_sel.srt;
      I.spd_select = res_sel.spd;
      I.tsd_select = res_sel.tsd;
      I.fau_select = res_sel.fau;
   }
}

#if PAN_ARCH == 10
static inline void
cs_run_tiling(struct cs_builder *b, uint32_t flags_override,
              struct cs_shader_res_sel res_sel)
{
   /* Staging regs */
   cs_flush_loads(b);

   cs_emit(b, RUN_TILING, I) {
      I.flags_override = flags_override;
      I.srt_select = res_sel.srt;
      I.spd_select = res_sel.spd;
      I.tsd_select = res_sel.tsd;
      I.fau_select = res_sel.fau;
   }
}
#endif

#if PAN_ARCH >= 12
static inline void
cs_run_idvs2(struct cs_builder *b, uint32_t flags_override, bool malloc_enable,
             struct cs_index draw_id,
             enum mali_idvs_shading_mode vertex_shading_mode)
{
   /* Staging regs */
   cs_flush_loads(b);

   cs_emit(b, RUN_IDVS2, I) {
      I.flags_override = flags_override;
      I.malloc_enable = malloc_enable;
      I.vertex_shading_mode = vertex_shading_mode;

      if (draw_id.type == CS_INDEX_UNDEF) {
         I.draw_id_register_enable = false;
      } else {
         I.draw_id_register_enable = true;
         I.draw_id = cs_src32(b, draw_id);
      }
   }
}
#else
static inline void
cs_run_idvs(struct cs_builder *b, uint32_t flags_override, bool malloc_enable,
            struct cs_shader_res_sel varying_sel,
            struct cs_shader_res_sel frag_sel, struct cs_index draw_id)
{
   /* Staging regs */
   cs_flush_loads(b);

   cs_emit(b, RUN_IDVS, I) {
      I.flags_override = flags_override;
      I.malloc_enable = malloc_enable;

      if (draw_id.type == CS_INDEX_UNDEF) {
         I.draw_id_register_enable = false;
      } else {
         I.draw_id_register_enable = true;
         I.draw_id = cs_src32(b, draw_id);
      }

      assert(varying_sel.spd == 1);
      assert(varying_sel.fau == 0 || varying_sel.fau == 1);
      assert(varying_sel.srt == 0 || varying_sel.srt == 1);
      assert(varying_sel.tsd == 0 || varying_sel.tsd == 1);
      I.varying_fau_select = varying_sel.fau == 1;
      I.varying_srt_select = varying_sel.srt == 1;
      I.varying_tsd_select = varying_sel.tsd == 1;

      assert(frag_sel.spd == 2);
      assert(frag_sel.fau == 2);
      assert(frag_sel.srt == 2 || frag_sel.srt == 0);
      assert(frag_sel.tsd == 2 || frag_sel.tsd == 0);
      I.fragment_srt_select = frag_sel.srt == 2;
      I.fragment_tsd_select = frag_sel.tsd == 2;
   }
}
#endif

static inline void
cs_run_fragment(struct cs_builder *b, bool enable_tem,
                enum mali_tile_render_order tile_order)
{
   /* Staging regs */
   cs_flush_loads(b);

   cs_emit(b, RUN_FRAGMENT, I) {
      I.enable_tem = enable_tem;
      I.tile_order = tile_order;
   }
}

static inline void
cs_run_fullscreen(struct cs_builder *b, uint32_t flags_override,
                  struct cs_index dcd)
{
   /* Staging regs */
   cs_flush_loads(b);

   cs_emit(b, RUN_FULLSCREEN, I) {
      I.flags_override = flags_override;
      I.dcd = cs_src64(b, dcd);
   }
}

static inline void
cs_finish_tiling(struct cs_builder *b)
{
   cs_emit(b, FINISH_TILING, I)
      ;
}

static inline void
cs_finish_fragment(struct cs_builder *b, bool increment_frag_completed,
                   struct cs_index first_free_heap_chunk,
                   struct cs_index last_free_heap_chunk,
                   struct cs_async_op async)
{
   cs_emit(b, FINISH_FRAGMENT, I) {
      I.increment_fragment_completed = increment_frag_completed;
      cs_apply_async(I, async);
      I.first_heap_chunk = cs_src64(b, first_free_heap_chunk);
      I.last_heap_chunk = cs_src64(b, last_free_heap_chunk);
   }
}

static inline void
cs_add32(struct cs_builder *b, struct cs_index dest, struct cs_index src,
         int32_t imm)
{
   cs_emit(b, ADD_IMM32, I) {
      I.destination = cs_dst32(b, dest);
      I.source = cs_src32(b, src);
      I.immediate = imm;
   }
}

static inline void
cs_add64(struct cs_builder *b, struct cs_index dest, struct cs_index src,
         int32_t imm)
{
   cs_emit(b, ADD_IMM64, I) {
      I.destination = cs_dst64(b, dest);
      I.source = cs_src64(b, src);
      I.immediate = imm;
   }
}

static inline void
cs_umin32(struct cs_builder *b, struct cs_index dest, struct cs_index src1,
          struct cs_index src2)
{
   cs_emit(b, UMIN32, I) {
      I.destination = cs_dst32(b, dest);
      I.source_1 = cs_src32(b, src1);
      I.source_0 = cs_src32(b, src2);
   }
}

#if PAN_ARCH >= 11
static inline void
cs_and32(struct cs_builder *b, struct cs_index dest, struct cs_index src1,
         struct cs_index src2)
{
   cs_emit(b, AND32, I) {
      I.destination = cs_dst32(b, dest);
      I.source_1 = cs_src32(b, src1);
      I.source_0 = cs_src32(b, src2);
   }
}

static inline void
cs_or32(struct cs_builder *b, struct cs_index dest, struct cs_index src1,
        struct cs_index src2)
{
   cs_emit(b, OR32, I) {
      I.destination = cs_dst32(b, dest);
      I.source_1 = cs_src32(b, src1);
      I.source_0 = cs_src32(b, src2);
   }
}

static inline void
cs_xor32(struct cs_builder *b, struct cs_index dest, struct cs_index src1,
         struct cs_index src2)
{
   cs_emit(b, XOR32, I) {
      I.destination = cs_dst32(b, dest);
      I.source_1 = cs_src32(b, src1);
      I.source_0 = cs_src32(b, src2);
   }
}

static inline void
cs_not32(struct cs_builder *b, struct cs_index dest, struct cs_index src)
{
   cs_emit(b, NOT32, I) {
      I.destination = cs_dst32(b, dest);
      I.source = cs_src32(b, src);
   }
}

static inline void
cs_bit_set32(struct cs_builder *b, struct cs_index dest, struct cs_index src1,
             struct cs_index src2)
{
   cs_emit(b, BIT_SET32, I) {
      I.destination = cs_dst32(b, dest);
      I.source_0 = cs_src32(b, src1);
      I.source_1 = cs_src32(b, src2);
   }
}

static inline void
cs_bit_clear32(struct cs_builder *b, struct cs_index dest, struct cs_index src1,
               struct cs_index src2)
{
   cs_emit(b, BIT_CLEAR32, I) {
      I.destination = cs_dst32(b, dest);
      I.source_1 = cs_src32(b, src1);
      I.source_0 = cs_src32(b, src2);
   }
}

static inline void
cs_move_reg32(struct cs_builder *b, struct cs_index dest, struct cs_index src)
{
   cs_emit(b, MOVE_REG32, I) {
      I.destination = cs_dst32(b, dest);
      I.source = cs_src32(b, src);
   }
}

static inline void
cs_set_state(struct cs_builder *b, enum mali_cs_set_state_type state,
             struct cs_index src)
{
   cs_emit(b, SET_STATE, I) {
      I.state = state;
      I.source = cs_src32(b, src);
   }
}

static inline void
cs_next_sb_entry(struct cs_builder *b, struct cs_index dest,
                 enum mali_cs_scoreboard_type sb_type,
                 enum mali_cs_next_sb_entry_format format)
{
   cs_emit(b, NEXT_SB_ENTRY, I) {
      I.destination = cs_dst32(b, dest);
      I.sb_type = sb_type;
      I.format = format;
   }
}

/*
 * Wait indirectly on a scoreboard (set via SET_STATE.SB_MASK_WAIT)
 */
static inline void
cs_wait_indirect(struct cs_builder *b)
{
   cs_emit(b, WAIT, I) {
      I.wait_mode = MALI_CS_WAIT_MODE_INDIRECT;
   }
}
#endif

static inline void
cs_load_to(struct cs_builder *b, struct cs_index dest, struct cs_index address,
           unsigned mask, int offset)
{
   unsigned count = util_last_bit(mask);
   unsigned base_reg = cs_dst_tuple(b, dest, count, mask);

   cs_emit(b, LOAD_MULTIPLE, I) {
      I.base_register = base_reg;
      I.address = cs_src64(b, address);
      I.mask = mask;
      I.offset = offset;
   }

   for (unsigned i = 0; i < count; i++) {
      if (mask & BITFIELD_BIT(i))
         BITSET_SET(b->cur_ls_tracker->pending_loads, base_reg + i);
   }
}

static inline void
cs_load32_to(struct cs_builder *b, struct cs_index dest,
             struct cs_index address, int offset)
{
   cs_load_to(b, dest, address, BITFIELD_MASK(1), offset);
}

static inline void
cs_load64_to(struct cs_builder *b, struct cs_index dest,
             struct cs_index address, int offset)
{
   cs_load_to(b, dest, address, BITFIELD_MASK(2), offset);
}

static inline void
cs_store(struct cs_builder *b, struct cs_index data, struct cs_index address,
         unsigned mask, int offset)
{
   unsigned count = util_last_bit(mask);
   unsigned base_reg = cs_src_tuple(b, data, count, mask);

   cs_emit(b, STORE_MULTIPLE, I) {
      I.base_register = base_reg;
      I.address = cs_src64(b, address);
      I.mask = mask;
      I.offset = offset;
   }

   for (unsigned i = 0; i < count; i++) {
      b->cur_ls_tracker->pending_stores |= mask & BITFIELD_BIT(i);
   }
}

static inline void
cs_store32(struct cs_builder *b, struct cs_index data, struct cs_index address,
           int offset)
{
   cs_store(b, data, address, BITFIELD_MASK(1), offset);
}

static inline void
cs_store64(struct cs_builder *b, struct cs_index data, struct cs_index address,
           int offset)
{
   cs_store(b, data, address, BITFIELD_MASK(2), offset);
}

#if PAN_ARCH < 11
/*
 * Select which scoreboard entry will track endpoint tasks and other tasks
 * respectively. Pass to cs_wait to wait later.
 */
static inline void
cs_set_scoreboard_entry(struct cs_builder *b, unsigned ep, unsigned other)
{
   assert(ep < CS_MAX_SB_COUNT && "invalid slot");
   assert(other < CS_MAX_SB_COUNT && "invalid slot");

   cs_emit(b, SET_SB_ENTRY, I) {
      I.endpoint_entry = ep;
      I.other_entry = other;
   }

   /* We assume the load/store scoreboard entry is static to keep things
    * simple. */
   assert(b->conf.ls_sb_slot == other);
}
#else
static inline void
cs_set_state_imm32(struct cs_builder *b, enum mali_cs_set_state_type state,
                   unsigned value)
{
   cs_emit(b, SET_STATE_IMM32, I) {
      I.state = state;
      I.value = value;
   }

   /* We assume the load/store scoreboard entry is static to keep things
    * simple. */
   if (state == MALI_CS_SET_STATE_TYPE_SB_SEL_OTHER)
      assert(b->conf.ls_sb_slot == value);
}
#endif

/*
 * Select which scoreboard entry will track endpoint tasks.
 * On v10, this also set other endpoint to SB0.
 * Pass to cs_wait to wait later.
 */
static inline void
cs_select_sb_entries_for_async_ops(struct cs_builder *b, unsigned ep)
{
#if PAN_ARCH == 10
   cs_set_scoreboard_entry(b, ep, 0);
#else
   cs_set_state_imm32(b, MALI_CS_SET_STATE_TYPE_SB_SEL_ENDPOINT, ep);
#endif
}

static inline void
cs_set_exception_handler(struct cs_builder *b,
                         enum mali_cs_exception_type exception_type,
                         struct cs_index address, struct cs_index length)
{
   cs_emit(b, SET_EXCEPTION_HANDLER, I) {
      I.exception_type = exception_type;
      I.address = cs_src64(b, address);
      I.length = cs_src32(b, length);
   }
}

static inline void
cs_call(struct cs_builder *b, struct cs_index address, struct cs_index length)
{
   cs_emit(b, CALL, I) {
      I.address = cs_src64(b, address);
      I.length = cs_src32(b, length);
   }
}

static inline void
cs_jump(struct cs_builder *b, struct cs_index address, struct cs_index length)
{
   cs_emit(b, JUMP, I) {
      I.address = cs_src64(b, address);
      I.length = cs_src32(b, length);
   }
}

enum cs_res_id {
   CS_COMPUTE_RES = BITFIELD_BIT(0),
   CS_FRAG_RES = BITFIELD_BIT(1),
   CS_TILER_RES = BITFIELD_BIT(2),
   CS_IDVS_RES = BITFIELD_BIT(3),
};

static inline void
cs_req_res(struct cs_builder *b, uint32_t res_mask)
{
   cs_emit(b, REQ_RESOURCE, I) {
      I.compute = res_mask & CS_COMPUTE_RES;
      I.tiler = res_mask & CS_TILER_RES;
      I.idvs = res_mask & CS_IDVS_RES;
      I.fragment = res_mask & CS_FRAG_RES;
   }
}

static inline void
cs_flush_caches(struct cs_builder *b, enum mali_cs_flush_mode l2,
                enum mali_cs_flush_mode lsc,
                enum mali_cs_other_flush_mode others, struct cs_index flush_id,
                struct cs_async_op async)
{
   cs_emit(b, FLUSH_CACHE2, I) {
      I.l2_flush_mode = l2;
      I.lsc_flush_mode = lsc;
      I.other_flush_mode = others;
      I.latest_flush_id = cs_src32(b, flush_id);
      cs_apply_async(I, async);
   }
}

#define CS_SYNC_OPS(__cnt_width)                                               \
   static inline void cs_sync##__cnt_width##_set(                              \
      struct cs_builder *b, bool propagate_error,                              \
      enum mali_cs_sync_scope scope, struct cs_index val,                      \
      struct cs_index addr, struct cs_async_op async)                          \
   {                                                                           \
      cs_emit(b, SYNC_SET##__cnt_width, I) {                                   \
         I.error_propagate = propagate_error;                                  \
         I.scope = scope;                                                      \
         I.data = cs_src##__cnt_width(b, val);                                 \
         I.address = cs_src64(b, addr);                                        \
         cs_apply_async(I, async);                                             \
      }                                                                        \
   }                                                                           \
                                                                               \
   static inline void cs_sync##__cnt_width##_add(                              \
      struct cs_builder *b, bool propagate_error,                              \
      enum mali_cs_sync_scope scope, struct cs_index val,                      \
      struct cs_index addr, struct cs_async_op async)                          \
   {                                                                           \
      cs_emit(b, SYNC_ADD##__cnt_width, I) {                                   \
         I.error_propagate = propagate_error;                                  \
         I.scope = scope;                                                      \
         I.data = cs_src##__cnt_width(b, val);                                 \
         I.address = cs_src64(b, addr);                                        \
         cs_apply_async(I, async);                                             \
      }                                                                        \
   }                                                                           \
                                                                               \
   static inline void cs_sync##__cnt_width##_wait(                             \
      struct cs_builder *b, bool reject_error, enum mali_cs_condition cond,    \
      struct cs_index ref, struct cs_index addr)                               \
   {                                                                           \
      assert(cond == MALI_CS_CONDITION_LEQUAL ||                               \
             cond == MALI_CS_CONDITION_GREATER);                               \
      cs_emit(b, SYNC_WAIT##__cnt_width, I) {                                  \
         I.error_reject = reject_error;                                        \
         I.condition = cond;                                                   \
         I.data = cs_src##__cnt_width(b, ref);                                 \
         I.address = cs_src64(b, addr);                                        \
      }                                                                        \
   }

CS_SYNC_OPS(32)
CS_SYNC_OPS(64)

static inline void
cs_store_state(struct cs_builder *b, struct cs_index address, int offset,
               enum mali_cs_state state, struct cs_async_op async)
{
   cs_emit(b, STORE_STATE, I) {
      I.offset = offset;
      I.state = state;
      I.address = cs_src64(b, address);
      cs_apply_async(I, async);
   }
}

static inline void
cs_prot_region(struct cs_builder *b, unsigned size)
{
   cs_emit(b, PROT_REGION, I) {
      I.size = size;
   }
}

static inline void
cs_run_compute_indirect(struct cs_builder *b, unsigned wg_per_task,
                        struct cs_shader_res_sel res_sel)
{
   /* Staging regs */
   cs_flush_loads(b);

   cs_emit(b, RUN_COMPUTE_INDIRECT, I) {
      I.workgroups_per_task = wg_per_task;
      I.srt_select = res_sel.srt;
      I.spd_select = res_sel.spd;
      I.tsd_select = res_sel.tsd;
      I.fau_select = res_sel.fau;
   }
}

static inline void
cs_error_barrier(struct cs_builder *b)
{
   cs_emit(b, ERROR_BARRIER, _)
      ;
}

static inline void
cs_heap_set(struct cs_builder *b, struct cs_index address)
{
   cs_emit(b, HEAP_SET, I) {
      I.address = cs_src64(b, address);
   }
}

static inline void
cs_heap_operation(struct cs_builder *b, enum mali_cs_heap_operation operation,
                  struct cs_async_op async)
{
   cs_emit(b, HEAP_OPERATION, I) {
      I.operation = operation;
      cs_apply_async(I, async);
   }
}

static inline void
cs_vt_start(struct cs_builder *b, struct cs_async_op async)
{
   cs_heap_operation(b, MALI_CS_HEAP_OPERATION_VERTEX_TILER_STARTED, async);
}

static inline void
cs_vt_end(struct cs_builder *b, struct cs_async_op async)
{
   cs_heap_operation(b, MALI_CS_HEAP_OPERATION_VERTEX_TILER_COMPLETED, async);
}

static inline void
cs_frag_end(struct cs_builder *b, struct cs_async_op async)
{
   cs_heap_operation(b, MALI_CS_HEAP_OPERATION_FRAGMENT_COMPLETED, async);
}

static inline void
cs_trace_point(struct cs_builder *b, struct cs_index regs,
               struct cs_async_op async)
{
   cs_emit(b, TRACE_POINT, I) {
      I.base_register =
         cs_src_tuple(b, regs, regs.size, (uint16_t)BITFIELD_MASK(regs.size));
      I.register_count = regs.size;
      cs_apply_async(I, async);
   }
}

struct cs_match {
   struct cs_block block;
   struct cs_label break_label;
   struct cs_block case_block;
   struct cs_label next_case_label;
   struct cs_index val;
   struct cs_index scratch_reg;
   struct cs_load_store_tracker case_ls_state;
   struct cs_load_store_tracker ls_state;
   struct cs_load_store_tracker *orig_ls_state;
   bool default_emitted;
};

static inline struct cs_match *
cs_match_start(struct cs_builder *b, struct cs_match *match,
               struct cs_index val, struct cs_index scratch_reg)
{
   *match = (struct cs_match){
      .val = val,
      .scratch_reg = scratch_reg,
      .orig_ls_state = b->cur_ls_tracker,
   };

   cs_block_start(b, &match->block);
   cs_label_init(&match->break_label);
   cs_label_init(&match->next_case_label);

   return match;
}

static inline void
cs_match_case_ls_set(struct cs_builder *b, struct cs_match *match)
{
   if (unlikely(match->orig_ls_state)) {
      match->case_ls_state = *match->orig_ls_state;
      b->cur_ls_tracker = &match->case_ls_state;
   }
}

static inline void
cs_match_case_ls_get(struct cs_match *match)
{
   if (unlikely(match->orig_ls_state)) {
      BITSET_OR(match->ls_state.pending_loads,
                match->case_ls_state.pending_loads,
                match->ls_state.pending_loads);
      match->ls_state.pending_stores |= match->case_ls_state.pending_stores;
   }
}

static inline void
cs_match_case(struct cs_builder *b, struct cs_match *match, uint32_t id)
{
   assert(!match->default_emitted || !"default case must be last");
   if (match->next_case_label.last_forward_ref != CS_LABEL_INVALID_POS) {
      cs_branch_label(b, &match->break_label, MALI_CS_CONDITION_ALWAYS,
                      cs_undef());
      cs_block_end(b, &match->case_block);
      cs_match_case_ls_get(match);
      cs_set_label(b, &match->next_case_label);
      cs_label_init(&match->next_case_label);
   }

   if (id)
      cs_add32(b, match->scratch_reg, match->val, -id);

   cs_branch_label(b, &match->next_case_label, MALI_CS_CONDITION_NEQUAL,
                   id ? match->scratch_reg : match->val);

   cs_match_case_ls_set(b, match);
   cs_block_start(b, &match->case_block);
}

static inline void
cs_match_default(struct cs_builder *b, struct cs_match *match)
{
   assert(match->next_case_label.last_forward_ref != CS_LABEL_INVALID_POS ||
          !"default case requires at least one other case");
   cs_branch_label(b, &match->break_label, MALI_CS_CONDITION_ALWAYS,
                   cs_undef());

   if (cs_cur_block(b) == &match->case_block) {
      cs_block_end(b, &match->case_block);
      cs_match_case_ls_get(match);
   }

   cs_set_label(b, &match->next_case_label);
   cs_label_init(&match->next_case_label);
   cs_match_case_ls_set(b, match);
   cs_block_start(b, &match->case_block);
   match->default_emitted = true;
}

static inline void
cs_match_end(struct cs_builder *b, struct cs_match *match)
{
   if (cs_cur_block(b) == &match->case_block) {
      cs_match_case_ls_get(match);
      cs_block_end(b, &match->case_block);
   }

   if (unlikely(match->orig_ls_state)) {
      if (!match->default_emitted) {
         /* If we don't have a default, assume we don't handle all possible cases
          * and the match load/store state with the original load/store state.
          */
         BITSET_OR(match->orig_ls_state->pending_loads,
                   match->ls_state.pending_loads,
                   match->orig_ls_state->pending_loads);
         match->orig_ls_state->pending_stores |= match->ls_state.pending_stores;
      } else {
         *match->orig_ls_state = match->ls_state;
      }

      b->cur_ls_tracker = match->orig_ls_state;
   }

   cs_set_label(b, &match->next_case_label);
   cs_set_label(b, &match->break_label);

   cs_block_end(b, &match->block);
}

#define cs_match(__b, __val, __scratch)                                        \
   for (struct cs_match __match_storage,                                       \
        *__match = cs_match_start(__b, &__match_storage, __val, __scratch);    \
        __match != NULL; cs_match_end(__b, &__match_storage), __match = NULL)

#define cs_case(__b, __ref)                                                    \
   for (bool __case_defined = ({                                               \
           cs_match_case(__b, __match, __ref);                                 \
           false;                                                              \
        });                                                                    \
        !__case_defined; __case_defined = true)

#define cs_default(__b)                                                        \
   for (bool __default_defined = ({                                            \
           cs_match_default(__b, __match);                                     \
           false;                                                              \
        });                                                                    \
        !__default_defined; __default_defined = true)

static inline void
cs_nop(struct cs_builder *b)
{
   cs_emit(b, NOP, I)
      ;
}

struct cs_function_ctx {
   struct cs_index ctx_reg;
   unsigned dump_addr_offset;
};

struct cs_function {
   struct cs_block block;
   struct cs_dirty_tracker dirty;
   struct cs_function_ctx ctx;
   unsigned dump_size;
   uint64_t address;
   uint32_t length;
};

static inline struct cs_function *
cs_function_start(struct cs_builder *b,
                  struct cs_function *function,
                  struct cs_function_ctx ctx)
{
   assert(cs_cur_block(b) == NULL);
   assert(b->conf.dirty_tracker == NULL);

   *function = (struct cs_function){
      .ctx = ctx,
   };

   cs_block_start(b, &function->block);

   b->conf.dirty_tracker = &function->dirty;

   return function;
}

#define SAVE_RESTORE_MAX_OPS (256 / 16)

static inline void
cs_function_end(struct cs_builder *b,
                struct cs_function *function)
{
   struct cs_index ranges[SAVE_RESTORE_MAX_OPS];
   uint16_t masks[SAVE_RESTORE_MAX_OPS];
   unsigned num_ranges = 0;
   uint32_t num_instrs =
      util_dynarray_num_elements(&b->blocks.instrs, uint64_t);
   struct cs_index addr_reg = {
      .type = CS_INDEX_REGISTER,
      .size = 2,
      .reg = (uint8_t) (b->conf.nr_registers - 2),
   };

   /* Manual cs_block_end() without an instruction flush. We do that to insert
    * the preamble without having to move memory in b->blocks.instrs. The flush
    * will be done after the preamble has been emitted. */
   assert(cs_cur_block(b) == &function->block);
   assert(function->block.next == NULL);
   b->blocks.stack = NULL;

   if (!num_instrs)
      return;

   /* Try to minimize number of load/store by grouping them */
   unsigned nregs = b->conf.nr_registers - b->conf.nr_kernel_registers;
   unsigned pos, last = 0;

   BITSET_FOREACH_SET(pos, function->dirty.regs, nregs) {
      unsigned range = MIN2(nregs - pos, 16);
      unsigned word = BITSET_BITWORD(pos);
      unsigned bit = pos % BITSET_WORDBITS;
      unsigned remaining_bits = BITSET_WORDBITS - bit;

      if (pos < last)
         continue;

      masks[num_ranges] = function->dirty.regs[word] >> bit;
      if (remaining_bits < range)
         masks[num_ranges] |= function->dirty.regs[word + 1] << remaining_bits;
      masks[num_ranges] &= BITFIELD_MASK(range);

      ranges[num_ranges] =
         cs_reg_tuple(b, pos, util_last_bit(masks[num_ranges]));
      num_ranges++;
      last = pos + range;
   }

   function->dump_size = BITSET_COUNT(function->dirty.regs) * sizeof(uint32_t);

   /* Make sure the current chunk is able to accommodate the block
    * instructions as well as the preamble and postamble.
    * Adding 4 instructions (2x wait_slot and the move for the address) as
    * the move might actually be translated to two MOVE32 instructions. */
   num_instrs += (num_ranges * 2) + 4;

   /* Align things on a cache-line in case the buffer contains more than one
    * function (64 bytes = 8 instructions). */
   uint32_t padded_num_instrs = ALIGN_POT(num_instrs, 8);

   if (!cs_reserve_instrs(b, padded_num_instrs))
      return;

   function->address =
      b->cur_chunk.buffer.gpu + (b->cur_chunk.pos * sizeof(uint64_t));

   /* Preamble: backup modified registers */
   if (num_ranges > 0) {
      unsigned offset = 0;

      cs_load64_to(b, addr_reg, function->ctx.ctx_reg,
                   function->ctx.dump_addr_offset);

      for (unsigned i = 0; i < num_ranges; ++i) {
         unsigned reg_count = util_bitcount(masks[i]);

         cs_store(b, ranges[i], addr_reg, masks[i], offset);
         offset += reg_count * 4;
      }

      cs_flush_stores(b);
   }

   /* Now that the preamble is emitted, we can flush the instructions we have in
    * our function block. */
   cs_flush_block_instrs(b);

   /* Postamble: restore modified registers */
   if (num_ranges > 0) {
      unsigned offset = 0;

      cs_load64_to(b, addr_reg, function->ctx.ctx_reg,
                   function->ctx.dump_addr_offset);

      for (unsigned i = 0; i < num_ranges; ++i) {
         unsigned reg_count = util_bitcount(masks[i]);

         cs_load_to(b, ranges[i], addr_reg, masks[i], offset);
         offset += reg_count * 4;
      }

      cs_flush_loads(b);
   }

   /* Fill the rest of the buffer with NOPs. */
   for (; num_instrs < padded_num_instrs; num_instrs++)
      cs_nop(b);

   function->length = padded_num_instrs;
}

#define cs_function_def(__b, __function, __ctx)                                \
   for (struct cs_function *__tmp = cs_function_start(__b, __function, __ctx); \
        __tmp != NULL;                                                         \
        cs_function_end(__b, __function), __tmp = NULL)

struct cs_tracing_ctx {
   bool enabled;
   struct cs_index ctx_reg;
   unsigned tracebuf_addr_offset;
};

static inline void
cs_trace_preamble(struct cs_builder *b, const struct cs_tracing_ctx *ctx,
                  struct cs_index scratch_regs, unsigned trace_size)
{
   assert(trace_size > 0 && ALIGN_POT(trace_size, 64) == trace_size &&
          trace_size < INT16_MAX);
   assert(scratch_regs.size >= 4 && !(scratch_regs.reg & 1));

   struct cs_index tracebuf_addr = cs_reg64(b, scratch_regs.reg);

   /* We always update the tracebuf position first, so we can easily detect OOB
    * access. Use cs_trace_field_offset() to get an offset taking this
    * pre-increment into account. */
   cs_load64_to(b, tracebuf_addr, ctx->ctx_reg, ctx->tracebuf_addr_offset);
   cs_add64(b, tracebuf_addr, tracebuf_addr, trace_size);
   cs_store64(b, tracebuf_addr, ctx->ctx_reg, ctx->tracebuf_addr_offset);
   cs_flush_stores(b);
}

#define cs_trace_field_offset(__type, __field)                                 \
   (int16_t)(offsetof(struct cs_##__type##_trace, __field) -                   \
             sizeof(struct cs_##__type##_trace))

struct cs_run_fragment_trace {
   uint64_t ip;
   uint32_t sr[7];
} __attribute__((aligned(64)));

static inline void
cs_trace_run_fragment(struct cs_builder *b, const struct cs_tracing_ctx *ctx,
                      struct cs_index scratch_regs, bool enable_tem,
                      enum mali_tile_render_order tile_order)
{
   if (likely(!ctx->enabled)) {
      cs_run_fragment(b, enable_tem, tile_order);
      return;
   }

   struct cs_index tracebuf_addr = cs_reg64(b, scratch_regs.reg);
   struct cs_index data = cs_reg64(b, scratch_regs.reg + 2);

   cs_trace_preamble(b, ctx, scratch_regs,
                     sizeof(struct cs_run_fragment_trace));

   /* cs_run_xx() must immediately follow cs_load_ip_to() otherwise the IP
    * won't point to the right instruction. */
   cs_load_ip_to(b, data);
   cs_run_fragment(b, enable_tem, tile_order);
   cs_store64(b, data, tracebuf_addr, cs_trace_field_offset(run_fragment, ip));

   cs_store(b, cs_reg_tuple(b, 40, 7), tracebuf_addr, BITFIELD_MASK(7),
            cs_trace_field_offset(run_fragment, sr));
   cs_flush_stores(b);
}

#if PAN_ARCH >= 12
struct cs_run_idvs2_trace {
   uint64_t ip;
   uint32_t draw_id;
   uint32_t pad;
   uint32_t sr[66];
} __attribute__((aligned(64)));

static inline void
cs_trace_run_idvs2(struct cs_builder *b, const struct cs_tracing_ctx *ctx,
                   struct cs_index scratch_regs, uint32_t flags_override,
                   bool malloc_enable, struct cs_index draw_id,
                   enum mali_idvs_shading_mode vertex_shading_mode)
{
   if (likely(!ctx->enabled)) {
      cs_run_idvs2(b, flags_override, malloc_enable, draw_id,
                   vertex_shading_mode);
      return;
   }

   struct cs_index tracebuf_addr = cs_reg64(b, scratch_regs.reg);
   struct cs_index data = cs_reg64(b, scratch_regs.reg + 2);

   cs_trace_preamble(b, ctx, scratch_regs, sizeof(struct cs_run_idvs2_trace));

   /* cs_run_xx() must immediately follow cs_load_ip_to() otherwise the IP
    * won't point to the right instruction. */
   cs_load_ip_to(b, data);
   cs_run_idvs2(b, flags_override, malloc_enable, draw_id, vertex_shading_mode);
   cs_store64(b, data, tracebuf_addr, cs_trace_field_offset(run_idvs2, ip));

   if (draw_id.type != CS_INDEX_UNDEF)
      cs_store32(b, draw_id, tracebuf_addr,
                 cs_trace_field_offset(run_idvs2, draw_id));

   for (unsigned i = 0; i < 64; i += 16)
      cs_store(b, cs_reg_tuple(b, i, 16), tracebuf_addr, BITFIELD_MASK(16),
               cs_trace_field_offset(run_idvs2, sr[0]) + i * sizeof(uint32_t));
   cs_store(b, cs_reg_tuple(b, 64, 2), tracebuf_addr, BITFIELD_MASK(2),
            cs_trace_field_offset(run_idvs2, sr[64]));
   cs_flush_stores(b);
}
#else
struct cs_run_idvs_trace {
   uint64_t ip;
   uint32_t draw_id;
   uint32_t pad;
   uint32_t sr[61];
} __attribute__((aligned(64)));

static inline void
cs_trace_run_idvs(struct cs_builder *b, const struct cs_tracing_ctx *ctx,
                  struct cs_index scratch_regs, uint32_t flags_override,
                  bool malloc_enable, struct cs_shader_res_sel varying_sel,
                  struct cs_shader_res_sel frag_sel, struct cs_index draw_id)
{
   if (likely(!ctx->enabled)) {
      cs_run_idvs(b, flags_override, malloc_enable, varying_sel, frag_sel,
                  draw_id);
      return;
   }

   struct cs_index tracebuf_addr = cs_reg64(b, scratch_regs.reg);
   struct cs_index data = cs_reg64(b, scratch_regs.reg + 2);

   cs_trace_preamble(b, ctx, scratch_regs, sizeof(struct cs_run_idvs_trace));

   /* cs_run_xx() must immediately follow cs_load_ip_to() otherwise the IP
    * won't point to the right instruction. */
   cs_load_ip_to(b, data);
   cs_run_idvs(b, flags_override, malloc_enable, varying_sel, frag_sel,
               draw_id);
   cs_store64(b, data, tracebuf_addr, cs_trace_field_offset(run_idvs, ip));

   if (draw_id.type != CS_INDEX_UNDEF)
      cs_store32(b, draw_id, tracebuf_addr,
                 cs_trace_field_offset(run_idvs, draw_id));

   for (unsigned i = 0; i < 48; i += 16)
      cs_store(b, cs_reg_tuple(b, i, 16), tracebuf_addr, BITFIELD_MASK(16),
               cs_trace_field_offset(run_idvs, sr[0]) + i * sizeof(uint32_t));
   cs_store(b, cs_reg_tuple(b, 48, 13), tracebuf_addr, BITFIELD_MASK(13),
            cs_trace_field_offset(run_idvs, sr[48]));
   cs_flush_stores(b);
}
#endif

struct cs_run_compute_trace {
   uint64_t ip;
   uint32_t sr[40];
} __attribute__((aligned(64)));

static inline void
cs_trace_run_compute(struct cs_builder *b, const struct cs_tracing_ctx *ctx,
                     struct cs_index scratch_regs, unsigned task_increment,
                     enum mali_task_axis task_axis,
                     struct cs_shader_res_sel res_sel)
{
   if (likely(!ctx->enabled)) {
      cs_run_compute(b, task_increment, task_axis, res_sel);
      return;
   }

   struct cs_index tracebuf_addr = cs_reg64(b, scratch_regs.reg);
   struct cs_index data = cs_reg64(b, scratch_regs.reg + 2);

   cs_trace_preamble(b, ctx, scratch_regs, sizeof(struct cs_run_compute_trace));

   /* cs_run_xx() must immediately follow cs_load_ip_to() otherwise the IP
    * won't point to the right instruction. */
   cs_load_ip_to(b, data);
   cs_run_compute(b, task_increment, task_axis, res_sel);
   cs_store64(b, data, tracebuf_addr, cs_trace_field_offset(run_compute, ip));

   for (unsigned i = 0; i < 32; i += 16)
      cs_store(b, cs_reg_tuple(b, i, 16), tracebuf_addr, BITFIELD_MASK(16),
               cs_trace_field_offset(run_compute, sr[0]) + i * sizeof(uint32_t));
   cs_store(b, cs_reg_tuple(b, 32, 8), tracebuf_addr, BITFIELD_MASK(8),
            cs_trace_field_offset(run_compute, sr[32]));
   cs_flush_stores(b);
}

static inline void
cs_trace_run_compute_indirect(struct cs_builder *b,
                              const struct cs_tracing_ctx *ctx,
                              struct cs_index scratch_regs,
                              unsigned wg_per_task,
                              struct cs_shader_res_sel res_sel)
{
   if (likely(!ctx->enabled)) {
      cs_run_compute_indirect(b, wg_per_task, res_sel);
      return;
   }

   struct cs_index tracebuf_addr = cs_reg64(b, scratch_regs.reg);
   struct cs_index data = cs_reg64(b, scratch_regs.reg + 2);

   cs_trace_preamble(b, ctx, scratch_regs, sizeof(struct cs_run_compute_trace));

   /* cs_run_xx() must immediately follow cs_load_ip_to() otherwise the IP
    * won't point to the right instruction. */
   cs_load_ip_to(b, data);
   cs_run_compute_indirect(b, wg_per_task, res_sel);
   cs_store64(b, data, tracebuf_addr, cs_trace_field_offset(run_compute, ip));

   for (unsigned i = 0; i < 32; i += 16)
      cs_store(b, cs_reg_tuple(b, i, 16), tracebuf_addr, BITFIELD_MASK(16),
               cs_trace_field_offset(run_compute, sr[0]) + i * sizeof(uint32_t));
   cs_store(b, cs_reg_tuple(b, 32, 8), tracebuf_addr, BITFIELD_MASK(8),
            cs_trace_field_offset(run_compute, sr[32]));
   cs_flush_stores(b);
}

#define cs_single_link_list_for_each_from(b, current, node_type, base_name)    \
   cs_while(b, MALI_CS_CONDITION_NEQUAL, current)                              \
      for (bool done = false; !done;                                           \
           cs_load64_to(b, current, current,                                   \
                        offsetof(node_type, base_name.next)),                  \
                done = true)

/**
 * Append an item to a cs_single_link_list.
 *
 * @param b The current cs_builder.
 * @param list_base CS register with the base address for the list pointer.
 * @param list_offset Offset added to list_base.
 * @param new_node_gpu GPU address of the node to insert.
 * @param base_offset Offset of cs_single_link_list_node in the new node.
 * @param head_tail Temporary register pair used in the function.
 */
static inline void
cs_single_link_list_add_tail(struct cs_builder *b, struct cs_index list_base,
                             int list_offset, struct cs_index new_node_gpu,
                             int base_offset, struct cs_index head_tail)
{
   assert(head_tail.size == 4);
   assert(head_tail.reg % 2 == 0);

   STATIC_ASSERT(offsetof(struct cs_single_link_list, tail) ==
                 offsetof(struct cs_single_link_list, head) + sizeof(uint64_t));
   struct cs_index head = cs_reg64(b, head_tail.reg);
   struct cs_index tail = cs_reg64(b, head_tail.reg + 2);

   /* Offset of the next pointer inside the node pointed to by new_node_gpu. */
   const int offset_next =
      base_offset + offsetof(struct cs_single_link_list_node, next);

   STATIC_ASSERT(offsetof(struct cs_single_link_list, head) == 0);
   cs_load_to(b, head_tail, list_base, BITFIELD_MASK(4), list_offset);

   /* If the list is empty (head == NULL), set the head, otherwise append to the
    * last node. */
   cs_if(b, MALI_CS_CONDITION_EQUAL, head)
      cs_add64(b, head, new_node_gpu, 0);
   cs_else(b)
      cs_store64(b, new_node_gpu, tail, offset_next);

   cs_add64(b, tail, new_node_gpu, 0);
   cs_store(b, head_tail, list_base, BITFIELD_MASK(4), list_offset);
   cs_flush_stores(b);
}

#ifdef __cplusplus
}
#endif
