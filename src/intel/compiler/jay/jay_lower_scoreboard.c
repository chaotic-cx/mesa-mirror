/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include <limits.h>
#include "compiler/brw/brw_eu_defines.h"
#include "util/bitscan.h"
#include "util/bitset.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_worklist.h"
#include "jay_builder.h"
#include "jay_ir.h"
#include "jay_opcodes.h"
#include "jay_private.h"

#define NUM_TOKENS (32)

static inline struct jay_footprint
def_to_sbid_key(jay_function *func, jay_inst *I, jay_def x)
{
   if (x.file == GPR) {
      return jay_def_to_footprint(func, I, x, JAY_TYPE_U32);
   } else if (x.file == UGPR) {
      /* SEND instructions can only use GRF-aligned multiples of whole
       * registers, so there's no point tracking UGPRs at a finer granularity.
       */
      return (struct jay_footprint){
         .base = jay_footprint_base(func->shader, UGPR) +
                 x.reg / jay_ugpr_per_grf(func->shader),
         .count =
            DIV_ROUND_UP(jay_num_values(x), jay_ugpr_per_grf(func->shader)),
         .width = 1,
         .stride = 1,
      };
   } else {
      return (struct jay_footprint){ 0 };
   }
}

enum sbid_dep_type { SRC, DST, MAX_SBID_DEP_TYPES };

#define jay_foreach_sbid_dep_type(type)                                        \
   for (enum sbid_dep_type type = SRC; type < MAX_SBID_DEP_TYPES; ++type)

struct swsb_sbid_state;

struct swsb_sbid_edge {
   struct swsb_sbid_state *ctx;
   uint32_t tokens_busy[MAX_SBID_DEP_TYPES];
   BITSET_WORD *tokens_bitset[NUM_TOKENS];
   bool tdr_state;
};

/** SBID scoreboarding */
struct swsb_sbid_state {
   unsigned words;
   unsigned max_sbids;

   void *mem_ctx;
   linear_ctx *lin_ctx;
   struct util_dynarray bitset_pool;

   BITSET_WORD *assigned;
   struct swsb_sbid_edge *edges;
   struct swsb_sbid_edge tmp;
};

static inline BITSET_WORD *
bitset_for(const struct swsb_sbid_edge *edge,
           unsigned sbid,
           enum sbid_dep_type type)
{
   assert(edge->tokens_bitset[sbid] != NULL);
   return &edge->tokens_bitset[sbid][type * edge->ctx->words];
}

/** Initializes the swsb_sbid_state struct */
static inline void
init_sbid_state(struct swsb_sbid_state *sbid_state,
                unsigned num_blocks,
                uint32_t nr_sbid_keys,
                unsigned max_sbids)
{
   *sbid_state = (struct swsb_sbid_state){
      .words = BITSET_WORDS(nr_sbid_keys),
      .max_sbids = max_sbids,
      .mem_ctx = ralloc_context(NULL),
   };

   sbid_state->lin_ctx = linear_context(sbid_state->mem_ctx);
   util_dynarray_init(&sbid_state->bitset_pool, sbid_state->mem_ctx);

   sbid_state->assigned =
      linear_zalloc_array(sbid_state->lin_ctx,
                          BITSET_WORD, BITSET_WORDS(num_blocks));
   sbid_state->edges =
      linear_zalloc_array(sbid_state->lin_ctx,
                          struct swsb_sbid_edge, num_blocks);

   for (unsigned i = 0; i < num_blocks; ++i)
      sbid_state->edges[i].ctx = sbid_state;

   sbid_state->tmp.ctx = sbid_state;
}

static inline void store_sbid_edge(struct swsb_sbid_edge *edge,
                                   const struct swsb_sbid_edge *src);

/** Resets the swsb_sbid_state struct */
static inline void
clear_sbid_state(struct swsb_sbid_state *sbid_state, unsigned dirty_blocks)
{
   if (dirty_blocks > 0) {
      for (unsigned i = 0; i < dirty_blocks; ++i) {
         store_sbid_edge(&sbid_state->edges[i], NULL);
      }

      BITSET_CLEAR_RANGE(sbid_state->assigned, 0, dirty_blocks);
   }
}

/** Allocate a zero'd bitset from the pool */
static inline BITSET_WORD *
alloc_sbid_bitset(struct swsb_sbid_state *ctx)
{
   if (!ctx->bitset_pool.size) {
      return linear_zalloc_child(ctx->lin_ctx,
                                 ctx->words * sizeof(BITSET_WORD) * 2);
   }
   BITSET_WORD *ret = util_dynarray_pop(&ctx->bitset_pool, BITSET_WORD *);
   assert(__bitset_is_empty(ret, ctx->words * 2));
   return ret;
}

/** Release a zero'd bitset back to the pool */
static inline void
release_sbid_bitset(struct swsb_sbid_state *ctx, BITSET_WORD **bitset)
{
   assert(__bitset_is_empty(*bitset, ctx->words * 2));
   util_dynarray_append(&ctx->bitset_pool, *bitset);
   *bitset = NULL;
}

/** Validates that a swsb_sbid_edge isn't in a bad state */
#define validate_edge(edge)                                                    \
   do {                                                                        \
      if (!(edge))                                                             \
         break;                                                                \
      uint32_t tokens_alloced =                                                \
         (edge)->tokens_busy[SRC] | (edge)->tokens_busy[DST];                  \
      assert(!((edge)->tokens_busy[SRC] & ~(edge)->tokens_busy[DST]));         \
      for (unsigned sbid = 0; sbid < NUM_TOKENS; ++sbid) {                     \
         assert(((edge)->tokens_bitset[sbid] != NULL) ==                       \
                !!(tokens_alloced & BITFIELD_BIT(sbid)));                      \
      }                                                                        \
      jay_foreach_sbid_dep_type(type) {                                        \
         u_foreach_bit(sbid, tokens_alloced & ~(edge)->tokens_busy[type]) {    \
            assert(__bitset_is_empty(bitset_for((edge), sbid, type),           \
                                     (edge)->ctx->words));                     \
         }                                                                     \
      }                                                                        \
   } while (false)

/** Copies the state of an swsb_sbid_edge struct */
static inline void
store_sbid_edge(struct swsb_sbid_edge *dst, const struct swsb_sbid_edge *src)
{
   validate_edge(src);
   uint32_t src_alloced = 0, dst_alloced = 0;

   jay_foreach_sbid_dep_type(type) {
      src_alloced |= src ? src->tokens_busy[type] : 0;
      dst_alloced |= dst->tokens_busy[type];
   }

   u_foreach_bit(sbid, src_alloced & ~dst_alloced) {
      dst->tokens_bitset[sbid] = alloc_sbid_bitset(dst->ctx);
   }

   jay_foreach_sbid_dep_type(type) {
      uint32_t src_busy = src ? src->tokens_busy[type] : 0;
      uint32_t dst_busy = dst->tokens_busy[type];

      u_foreach_bit(sbid, src_busy) {
         __bitset_copy(bitset_for(dst, sbid, type), bitset_for(src, sbid, type),
                       dst->ctx->words);
      }

      u_foreach_bit(sbid, ~src_busy & dst_busy) {
         __bitset_zero(bitset_for(dst, sbid, type), dst->ctx->words);
      }

      dst->tokens_busy[type] = src_busy;
   }

   u_foreach_bit(sbid, dst_alloced & ~src_alloced) {
      release_sbid_bitset(dst->ctx, &dst->tokens_bitset[sbid]);
   }

   dst->tdr_state = src ? src->tdr_state : false;

   validate_edge(dst);
}

/** Merges bits from src into dst, returns true if dst changed */
static inline bool
merge_bitsets(BITSET_WORD *dst, const BITSET_WORD *src, unsigned words)
{
   BITSET_WORD changed = 0;

   for (unsigned i = 0; i < words; ++i) {
      BITSET_WORD prev = dst[i];
      dst[i] = src[i] | prev;
      changed |= dst[i] ^ prev;
   }

   return changed;
}

/** Merges dependencies from src into dst, returns true if dst changed */
static inline bool
merge_sbid_edges(struct swsb_sbid_edge *dst, const struct swsb_sbid_edge *src)
{
   validate_edge(src);
   validate_edge(dst);
   uint32_t src_alloced = 0, dst_alloced = 0;
   bool changed = false;

   jay_foreach_sbid_dep_type(type) {
      src_alloced |= src->tokens_busy[type];
      dst_alloced |= dst->tokens_busy[type];
   }

   u_foreach_bit(sbid, src_alloced & ~dst_alloced) {
      dst->tokens_bitset[sbid] = alloc_sbid_bitset(dst->ctx);
   }

   jay_foreach_sbid_dep_type(type) {
      uint32_t src_busy = src->tokens_busy[type];
      uint32_t dst_busy = dst->tokens_busy[type];
      changed |= src_busy & ~dst_busy;

      u_foreach_bit(sbid, src_busy & ~dst_busy) {
         __bitset_copy(bitset_for(dst, sbid, type), bitset_for(src, sbid, type),
                       dst->ctx->words);
      }

      u_foreach_bit(sbid, src_busy & dst_busy) {
         changed |=
            merge_bitsets(bitset_for(dst, sbid, type),
                          bitset_for(src, sbid, type), dst->ctx->words);
      }

      dst->tokens_busy[type] = src_busy | dst_busy;
   }

   changed |= src->tdr_state & !dst->tdr_state;
   dst->tdr_state |= src->tdr_state;

   validate_edge(dst);
   return changed;
}

static inline unsigned
hash_block_index(unsigned index)
{
   index ^= index >> 16;
   index *= 0x85ebca6b;
   index ^= index >> 13;
   index *= 0xc2b2ae35;
   index ^= index >> 16;
   return index;
}

static inline void
sync_sbids(jay_builder *b, uint32_t mask, gen_sbid_mode mode)
{
   if (util_is_power_of_two_nonzero(mask)) {
      jay_SYNC(b, jay_null(), TGL_SYNC_NOP)->dep =
         gen_swsb_sbid(mode, util_logbase2(mask));
   } else if (mask) {
      jay_SYNC(b, mask, mode == GEN_SBID_DST ? TGL_SYNC_ALLWR : TGL_SYNC_ALLRD);
   }
}

static inline bool
jay_inst_has_sbid(const struct intel_device_info *devinfo, const jay_inst *I)
{
   return jay_inst_is_unordered(devinfo, I) &&
          !(I->op == JAY_OPCODE_SEND && jay_send_eot(I));
}

/**
 * Returns the index of the Nth zero-indexed bit set in the bitfield. Such a bit
 * must exist. For example, using this function to find with bit=1 bit set in
 * the bitfield 0b1100010 would return 5.
 */
static inline unsigned
find_nth_bit(uint32_t bitfield, unsigned bit)
{
   assert(bit < util_bitcount(bitfield) && "must exist");

   /* Repeatedly clear the bottom bit */
   for (unsigned i = 0; i < bit; ++i) {
      bitfield &= (bitfield - 1);
   }

   return ffs(bitfield) - 1;
}

static void
lower_sbid_local(jay_function *func,
                 jay_block *block,
                 struct swsb_sbid_edge *edge,
                 bool commit)
{
   validate_edge(edge);

   uint32_t busy_src = edge->tokens_busy[SRC];
   uint32_t busy_dst = edge->tokens_busy[DST];
   bool tdr_state = edge->tdr_state;

   unsigned roundrobin = 0;

   /* Try to reduce conflicts across control flow edges by selecting a random
    * number to XOR with the roundrobin counter.
    */
   unsigned hash = hash_block_index(block->index);

   /* We assign SBIDs as soon as we first reach the block, this avoids any
    * unnecessary SBID conflicts between edges of a block and it's dominators,
    * but the back edges of loops are left untended.
    */
   bool assign_sbid = !BITSET_TEST(edge->ctx->assigned, block->index);
   BITSET_SET(edge->ctx->assigned, block->index);

   jay_foreach_inst_in_block_safe(block, I) {
      jay_builder b = jay_init_builder(func, jay_before_inst(I));
      uint32_t sync_src = 0, sync_dst = 0;

      /* Read-after-write */
      jay_foreach_src(I, s) {
         struct jay_footprint src = def_to_sbid_key(func, I, I->src[s]);

         u_foreach_bit(sbid, busy_dst) {
            if (jay_footprint_test(bitset_for(edge, sbid, DST), src)) {
               sync_dst |= BITFIELD_BIT(sbid);
               busy_dst &= ~BITFIELD_BIT(sbid);
               busy_src &= ~BITFIELD_BIT(sbid);
            }
         }
      }

      /* Write-after-write & write-after-read */
      jay_foreach_dst(I, d) {
         struct jay_footprint dst = def_to_sbid_key(func, I, d);

         u_foreach_bit(sbid, busy_dst) {
            if (jay_footprint_test(bitset_for(edge, sbid, DST), dst)) {
               sync_dst |= BITFIELD_BIT(sbid);
               busy_dst &= ~BITFIELD_BIT(sbid);
               busy_src &= ~BITFIELD_BIT(sbid);
            }
         }

         u_foreach_bit(sbid, busy_src) {
            if (jay_footprint_test(bitset_for(edge, sbid, SRC), dst)) {
               sync_src |= BITFIELD_BIT(sbid);
               busy_src &= ~BITFIELD_BIT(sbid);
            }
         }
      }

      if (jay_inst_has_sbid(func->shader->devinfo, I)) {
         unsigned sbid;

         if (!assign_sbid) {
            sbid = I->dep.sbid;
         } else {
            if (sync_dst) {
               /* If we depend on $N.dst, there's no extra cost to $N.set */
               sbid = ffs(sync_dst) - 1;
            } else {
               /* Otherwise, select a random SBID that's not already busy */
               unsigned max_sbids = edge->ctx->max_sbids;
               uint32_t free_sbids = ~busy_dst & BITFIELD_MASK(max_sbids);
               if (free_sbids) {
                  sbid = (roundrobin++ ^ hash) % util_bitcount(free_sbids);
                  sbid = find_nth_bit(free_sbids, sbid);
               } else {
                  sbid = (roundrobin++ ^ hash) % max_sbids;
               }
            }

            I->dep.sbid = sbid;
         }

         if (edge->tokens_bitset[sbid] == NULL) {
            edge->tokens_bitset[sbid] = alloc_sbid_bitset(edge->ctx);
         } else {
            /* Dispose of the bitset's previous contents.  Waiting on $N.dst
             * also retires $N's reads.
             */
            jay_foreach_sbid_dep_type(type) {
               uint32_t mask = type == DST ? (busy_dst | sync_dst) :
                                             (busy_src | sync_src | sync_dst);
               if (mask & BITFIELD_BIT(sbid)) {
                  __bitset_zero(bitset_for(edge, sbid, type), edge->ctx->words);
               }
            }
         }

         /* SBID.set implies SBID.dst (which implies SBID.src), so elide */
         sync_dst &= ~BITFIELD_BIT(sbid);
         sync_src &= ~BITFIELD_BIT(sbid);
         busy_dst |= BITFIELD_BIT(sbid);
         busy_src |= BITFIELD_BIT(sbid);

         jay_footprint_set(bitset_for(edge, sbid, DST),
                           def_to_sbid_key(func, I, I->dst));

         jay_foreach_src(I, s) {
            jay_footprint_set(bitset_for(edge, sbid, SRC),
                              def_to_sbid_key(func, I, I->src[s]));
         }

         /* Barriers are non-EOT gateway messages. Insert the needed SYNC */
         if (commit &&
             I->op == JAY_OPCODE_SEND &&
             jay_send_sfid(I) == GEN_SFID_MESSAGE_GATEWAY) {
            b.cursor = jay_after_inst(I);
            jay_SYNC(&b, jay_null(), TGL_SYNC_BAR);
         }
      } else if (I->op == JAY_OPCODE_SCHEDULE_BARRIER) {
         sync_dst |= busy_dst;
         sync_src |= busy_src & ~busy_dst;
         busy_dst = 0;
         busy_src = 0;
      } else if (I->op == JAY_OPCODE_CHECK_TDR) {
         tdr_state = true;
      }

      /* Dispose of the bitsets for any synced sbids */
      jay_foreach_sbid_dep_type(type) {
         u_foreach_bit(sbid, sync_dst | (type == SRC ? sync_src : 0)) {
            __bitset_zero(bitset_for(edge, sbid, type), edge->ctx->words);
         }
      }

      u_foreach_bit(sbid, (sync_src | sync_dst) & ~busy_dst) {
         release_sbid_bitset(edge->ctx, &edge->tokens_bitset[sbid]);
      }

      if (!commit)
         continue;

      b.cursor = jay_before_inst(I);
      assert(((sync_dst & sync_src) == 0) && "by construction");

      sync_sbids(&b, sync_dst, GEN_SBID_DST);
      sync_sbids(&b, sync_src, GEN_SBID_SRC);

      /* Convert all memory volatile SENDs to SENDCs if we have a pending thread
       * dependency check. This can lead to some unnecessary SENDCs, but it
       * shouldn't matter for performance unless apps abuse interlocks. SENDC
       * does not stall the EU on its own, so we would also have to track when
       * its SBID has cleared if we ever wanted to implement proper elision.
       */
      if (tdr_state && I->op == JAY_OPCODE_SEND && !jay_send_pure(I)) {
         jay_set_send_check_tdr(I, true);
      }

      if (I->op == JAY_OPCODE_SCHEDULE_BARRIER ||
          I->op == JAY_OPCODE_CHECK_TDR) {
         /* Lowered above into a sync, but removed late to keep the cursor */
         jay_remove_instruction(I);
      }
   }

   edge->tokens_busy[SRC] = busy_src;
   edge->tokens_busy[DST] = busy_dst;
   edge->tdr_state = tdr_state;
   validate_edge(edge);
}

/**
 * Regdist scoreboarding
 *
 * Register access is tracked per pipe, with 0 (NONE) having data on the writer
 * packed into a u32 with the following macros.
 */
#define make_writer(pipe, ip) (((uint32_t) ip << 3) | (uint32_t) (pipe))
#define writer_ip(writer)     (writer >> 3)
#define writer_pipe(writer)   (gen_pipe)(writer & BITFIELD_MASK(3))

#define GEN_NUM_PIPES (GEN_PIPE_ALL)
typedef uint32_t u32_per_pipe[GEN_NUM_PIPES];

struct swsb_regdist_state {
   jay_shader *shader;
   unsigned ip[GEN_NUM_PIPES];

   /* finished_ip[X] = ip means ip on pipe X has already been waited on. */
   unsigned finished_ip[GEN_NUM_PIPES];
   u32_per_pipe *access;

   jay_inst *last_sync;
};

/*
 * Return the maximum ALU distance to consider. Anything further is guaranteed
 * to have already written its result by the time we issue. These values are not
 * in the bspec but are #define'd in IGC as SWSB_MAX_*_DEPENDENCE_DISTANCE.
 *
 * Confusingly, IGC also defines SWSB_MAX_ALU_DEPENDENCE_DISTANCE_VALUE as 7.
 * There is a discrepency between what the hardware does and what we can encode.
 * Any writes from 11 instructions ago are guaranteed to have landed, whereas if
 * you need to sync, you can only sync with something up to 7 instructions ago
 * (and implicitly, everything in-order before that).
 *
 * These are conservative values. Some archeology suggests the real values may
 * be lower on some platforms but for now we match IGC to be safe.
 */
static inline unsigned
max_dependence(gen_pipe pipe)
{
   return pipe == GEN_PIPE_SCALAR ? 2 :
          pipe == GEN_PIPE_MATH   ? 18 :
          pipe == GEN_PIPE_LONG   ? 15 :
                                    11;
}

static void
depend_on_writer(struct swsb_regdist_state *state,
                 struct jay_footprint r,
                 unsigned *dep,
                 gen_pipe exec,
                 bool except_exec)
{
   jay_foreach_in_footprint(r, key) {
      assert(key < jay_footprint_base(state->shader, ~0));
      uint32_t w = state->access[key][0];
      gen_pipe write = writer_pipe(w);

      /* We omit write-after-{read,write} dependencies (except_exec) within a
       * single execution pipe, since each pipe is internally in-order. We also
       * omit dependencies on the same pipe that are too far to be relevant.
       */
      if (write != exec ||
          (!except_exec &&
           writer_ip(w) + max_dependence(exec) > state->ip[write])) {

         dep[write] = MAX2(dep[write], writer_ip(w));
      }
   }
}

#define jay_foreach_pipe(pipe)                                                 \
   for (unsigned pipe = 1; pipe < GEN_NUM_PIPES; ++pipe)

static void
lower_regdist(jay_function *func, jay_inst *I, struct swsb_regdist_state *ctx)
{
   gen_pipe exec_pipe = jay_inst_exec_pipe(func->shader->devinfo, I);
   unsigned dep[GEN_NUM_PIPES] = { 0 };
   jay_def dsts[3] = { I->dst, I->cond_flag };

   for (unsigned i = 0; i < ARRAY_SIZE(dsts); ++i) {
      struct jay_footprint r = jay_def_to_footprint(func, I, dsts[i], I->type);
      depend_on_writer(ctx, r, dep, exec_pipe, true /* except_pipe */);

      jay_foreach_in_footprint(r, key) {
         jay_foreach_pipe(p) {
            if (p != exec_pipe) {
               dep[p] = MAX2(dep[p], ctx->access[key][p]);
            }
         }
      }
   }

   /* Read-after-write. The hardware scoreboards accumulators/flags within a
    * pipe, so we set except_pipe for that to omit those annotations. The
    * hardware does *not* scoreboard accumulator/flags  across pipes so we can't
    * just ignore accumulator/flags when scoreboarding. For example, the I@1
    * annotation is required in the following code:
    *
    * (16)        mul.s32 acc0, g26, g24<16,8,2>:u16                  │
    * (32)        mad.f32 acc0, u8.6, u8.8, g20                       │ I@1
    */
   jay_foreach_src(I, s) {
      bool except_pipe = I->src[s].file == ACCUM || I->src[s].file == FLAG;
      depend_on_writer(ctx,
                       jay_def_to_footprint(func, I, I->src[s],
                                            jay_src_type(I, s)),
                       dep, exec_pipe, except_pipe);
   }

   uint32_t wait_pipes = 0;
   unsigned min_delta = 7;

   jay_foreach_pipe(p) {
      if (dep[p] && dep[p] > ctx->finished_ip[p]) {
         min_delta = MIN2(min_delta, ctx->ip[p] - dep[p] + 1);
         wait_pipes |= BITFIELD_BIT(p);
      }
   }

   /* We'll wait on the unioned dependency. Update the tracking for that. */
   u_foreach_bit(p, wait_pipes) {
      ctx->finished_ip[p] = ctx->ip[p] + 1 - min_delta;
   }

   uint32_t last_pipe = util_logbase2(wait_pipes);
   bool single_wait = wait_pipes == BITFIELD_BIT(last_pipe);

   bool has_sbid = jay_inst_has_sbid(func->shader->devinfo, I);
   I->dep = (gen_swsb){
      .sbid = I->dep.sbid,
      .mode = has_sbid ? GEN_SBID_SET : GEN_SBID_NULL,
      .regdist = wait_pipes ? min_delta : 0,
      .pipe = single_wait && (!has_sbid ||
                              last_pipe == GEN_PIPE_FLOAT ||
                              last_pipe == GEN_PIPE_INT) ?
                 last_pipe :
              wait_pipes ? GEN_PIPE_ALL :
                           GEN_PIPE_NONE,
   };

   /* DPAS and slow ALU on Gfx12 can only represent in-order dependency for its.
    * inferred pipe, so if it depends on something else, add an extra SYNC.nop.
    */
   if (I->op != JAY_OPCODE_SEND &&
       jay_inst_is_unordered(func->shader->devinfo, I) &&
       wait_pipes &&
       (!single_wait ||
        last_pipe != jay_inferred_sync_pipe(func->shader->devinfo, I))) {
      assert(I->dep.regdist > 0);
      jay_builder b = jay_init_builder(func, jay_before_inst(I));

      jay_inst *sync = jay_SYNC(&b, jay_null(), TGL_SYNC_NOP);
      sync->dep.regdist = I->dep.regdist;
      sync->dep.pipe = I->dep.pipe;

      I->dep.regdist = 0;
      I->dep.pipe = GEN_PIPE_NONE;
   }

   /* Fold the immediate preceding SYNC.nop into this instruction, allowing
    * us to wait on both ALU and a SBID in the same annotation. We cannot do
    * this safely in the presence of predication or SIMD splitting that could
    * cause any part of the instruction to get shot down, skipping the sync
    * for future instructions (at least not without more tricky logic).
    */
   if (ctx->last_sync &&
       jay_sync_op(ctx->last_sync) == TGL_SYNC_NOP &&
       I->dep.mode == GEN_SBID_NULL &&
       !I->predication &&
       !I->simd_split &&
       (I->dep.regdist == 0 ||
        jay_inferred_sync_pipe(func->shader->devinfo, I) == I->dep.pipe)) {

      assert(ctx->last_sync->dep.regdist == 0);
      assert(ctx->last_sync->dep.pipe == GEN_PIPE_NONE);

      I->dep.mode = ctx->last_sync->dep.mode;
      I->dep.sbid = ctx->last_sync->dep.sbid;

      jay_remove_instruction(ctx->last_sync);
   }

   if (exec_pipe != GEN_PIPE_NONE) {
      ctx->ip[exec_pipe]++;

      uint32_t now = make_writer(exec_pipe, ctx->ip[exec_pipe]);

      for (unsigned i = 0; i < ARRAY_SIZE(dsts); ++i) {
         struct jay_footprint r =
            jay_def_to_footprint(func, I, dsts[i], I->type);

         jay_foreach_in_footprint(r, key) {
            ctx->access[key][0] = now;
         }
      }

      jay_foreach_src(I, s) {
         struct jay_footprint r =
            jay_def_to_footprint(func, I, I->src[s], jay_src_type(I, s));

         jay_foreach_in_footprint(r, key) {
            ctx->access[key][exec_pipe] = ctx->ip[exec_pipe];
         }
      }
   }

   ctx->last_sync = NULL;
}

/*
 * Trivial scoreboard lowering pass for debugging use. Stalls after every
 * instruction and assigns SBID zero to all messages.
 */
void
jay_lower_scoreboard_trivial(jay_shader *shader)
{
   bool any_check_tdr = false;
   jay_foreach_inst_in_shader_safe(shader, func, I) {
      if (jay_inst_has_sbid(shader->devinfo, I)) {
         /* DPAS can't have an A@1, so insert an extra SYNC.nop. */
         jay_builder before = jay_init_builder(func, jay_before_inst(I));
         jay_SYNC(&before, jay_null(), TGL_SYNC_NOP)->dep = gen_swsb_regdist(1);
         I->dep = gen_swsb_sbid(GEN_SBID_SET, 0);

         jay_builder b = jay_init_builder(func, jay_after_inst(I));
         sync_sbids(&b, BITFIELD_BIT(0), GEN_SBID_DST);

         /* Barriers are non-EOT gateway messages. Insert the needed SYNC */
         if (I->op == JAY_OPCODE_SEND &&
             jay_send_sfid(I) == GEN_SFID_MESSAGE_GATEWAY) {
            b.cursor = jay_after_inst(I);
            jay_SYNC(&b, jay_null(), TGL_SYNC_BAR);
         }

      } else if (I->op == JAY_OPCODE_CHECK_TDR) {
         any_check_tdr = true;
         jay_remove_instruction(I);
      } else if (I->op == JAY_OPCODE_SCHEDULE_BARRIER) {
         jay_remove_instruction(I);
      } else {
         I->dep = gen_swsb_regdist(1);
      }
   }

   jay_foreach_inst_in_shader(shader, func, I) {
      if (I->op == JAY_OPCODE_SEND) {
         jay_set_send_check_tdr(I, any_check_tdr);
      }
   }
}

void
jay_lower_scoreboard(jay_shader *shader)
{
   u32_per_pipe *regdists =
      malloc(sizeof(*regdists) * jay_footprint_base(shader, ~0));

   unsigned max_blocks = 0;
   jay_foreach_function(shader, f)
      max_blocks = MAX2(max_blocks, f->num_blocks);

   uint32_t nr_sbid_keys =
      (shader->num_regs[GPR] * jay_grf_per_gpr(shader)) +
      DIV_ROUND_UP(shader->num_regs[UGPR], jay_ugpr_per_grf(shader));

   unsigned max_sbids = intel_device_info_max_sbids(shader->devinfo);

   struct swsb_sbid_state sbid_state;
   init_sbid_state(&sbid_state, max_blocks, nr_sbid_keys, max_sbids);

   u_worklist sbid_worklist;
   u_worklist_init(&sbid_worklist, max_blocks, sbid_state.mem_ctx);

   unsigned dirty_blocks = 0;
   jay_foreach_function(shader, f) {
      memset(regdists, 0, sizeof(*regdists) * jay_footprint_base(shader, ~0));
      clear_sbid_state(&sbid_state, dirty_blocks);
      dirty_blocks = f->num_blocks;

      jay_foreach_block_rev(f, block) {
         jay_worklist_push_tail(&sbid_worklist, block);
      }

      /* Propagate SBID dependencies until a fixed point is reached */
      while (!u_worklist_is_empty(&sbid_worklist)) {
         jay_block *block = jay_worklist_pop_tail(&sbid_worklist);
         store_sbid_edge(&sbid_state.tmp, &sbid_state.edges[block->index]);
         lower_sbid_local(f, block, &sbid_state.tmp, false);

         jay_foreach_successor(block, succ, UGPR) {
            struct swsb_sbid_edge *edge = &sbid_state.edges[(*succ)->index];
            if (merge_sbid_edges(edge, &sbid_state.tmp)) {
               jay_worklist_push_tail(&sbid_worklist, *succ);
            }
         }
      }

      /* Apply the finished SBID dependencies to the program */
      jay_foreach_block(f, block) {
         lower_sbid_local(f, block, &sbid_state.edges[block->index], true);
      }

      struct swsb_regdist_state regdist_state = {
         .shader = shader,
         .access = regdists,
      };

      /* RegDist scoreboarding is global but requires no dataflow analysis,
       * because taking a branch stalls all ALU pipelines. Therefore, it
       * suffices to propagate scoreboard state along fallthrough edges. We
       * implement that backwards: state is preserved (correctness), except we
       * clear regdists[] when entering blocks that are unreachable by falling
       * through from the previous source-order block and hence must be branch
       * targets coming in with a clear scoreboard.
       */
      jay_block *next = NULL;

      jay_foreach_block(f, block) {
         if (!list_is_empty(&block->instructions) && next != block) {
            memset(regdists, 0,
                   sizeof(*regdists) * jay_footprint_base(shader, ~0));
         }

         next = jay_first_successor(block, UGPR);

         jay_foreach_inst_in_block_safe(block, I) {
            if (I->op == JAY_OPCODE_SYNC) {
               regdist_state.last_sync = I;

               /* RegDist-only syncs are added only by lower_regdist, before
                * the instruction, so are not seen here.
                */
               assert(jay_sync_op(I) != TGL_SYNC_NOP ||
                      I->dep.mode != GEN_SBID_NULL);
            } else {
               lower_regdist(f, I, &regdist_state);
            }
         }
      }
   }

   free(regdists);
   ralloc_free(sbid_state.mem_ctx);
}
