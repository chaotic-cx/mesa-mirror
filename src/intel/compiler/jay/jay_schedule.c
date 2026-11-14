/*
 * Copyright 2026 Intel Corporation
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * Copyright 2019 Broadcom
 * SPDX-License-Identifier: MIT
 */

/*
 * This file implements a simple list scheduler. It can run early (to decrease
 * register pressure and spilling), pre-RA but after spilling (to improve
 * latency), or post-RA (to improve latency).
 *
 * SSA form allows us to estimate register demand cheaply and accurately, which
 * theoretically [1] gives the pre-RA pass the two Hippocratic properties:
 *
 * 1. Shaders with low register pressure are unaffected by pressure scheduling.
 * 2. Register pressure can never be increased beyond the target.
 *
 * In other words: first, do no harm.
 *
 * We use simple greedy heuristics to choose instructions.
 *
 * For pressure, we choose instructions that decrease liveness. This is far from
 * optimal! But thanks to the above properties, even a heuristic that picked
 * random instructions would be a win overall - by construction, we can only
 * ever win.
 *
 * For latency, we use the standard textbook techniques. We support both
 * forwards & backwards scheduling.
 *
 * [1] In reality, neither property is strictly satisfied by the pre-spilling
 * scheduler due to the messy details of mapping our clean logical model onto
 * Intel's many weird physical register files. Nevertheless, the algorithm is
 * well-motivated and the empirical results on Xe2 are excellent.
 */

#include "util/bitset.h"
#include "util/ralloc.h"
#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "gen_enums.h"
#include "jay_builder.h"
#include "jay_dag.h"
#include "jay_ir.h"
#include "jay_opcodes.h"
#include "jay_private.h"

/* Aggression is stored in units of 1/AGGRESSION_SCALE. */
#define AGGRESSION_SCALE 8

/* Bitfield describing a scheduler policy */
enum sched_mode {
   BACKWARD = 0x1, /* Forward if unset */
   LATENCY = 0x2,
   PRESSURE = 0x4,
};

struct sched_block {
   uint32_t first, last;
   uint32_t latency;
   int32_t max_pressure;
};

struct sched_ctx {
   /* Compilation phase. This induces a scheduler mode. */
   enum { EARLY, POSTSPILL, POSTRA, POSTACC } phase;

   /* Function we are currently scheduling */
   jay_function *func;

   struct jay_dag dag, dag_t;
   struct jay_dag_iterator it;

   jay_inst **insts;
   struct u_sparse_bitset live;
   BITSET_WORD *seen;

   /* Array of node indices for the schedule we're building */
   struct util_dynarray schedule;

   struct sched_block *blocks;
   uint32_t *cycle_ready;

   /* For post-spill scheduling, register demand limits */
   int32_t demand_limit[JAY_NUM_SSA_FILES];

   /* Current time in cycles */
   uint32_t cycle;

   /* Current register demand */
   int32_t demand[JAY_NUM_SSA_FILES];

   /* For pressure-informed latency scheduling, a parameter controlling how
    * aggressive we are. Higher reduces latency more while costing more
    * pressure. Fixed-point, see AGGRESSION_SCALE.
    */
   unsigned aggression;
   /* Register headroom for the pressure-versus-latency rule. */
   unsigned headroom;
   /* Select the headroom rule instead of fixed-point aggression. */
   bool use_headroom;

   /* Pooled allocations used for DAG construction */
   struct util_dynarray helper_reads;

   union {
      struct {
         uint32_t *def;
      } prera;

      struct {
         uint32_t *writer;
         struct util_dynarray *readers;
      } postra;
   };
};

/* Cut down version of the function in jay_liveness.c */
static void
liveness_update(struct u_sparse_bitset *live, jay_inst *I)
{
   jay_foreach_dst_index(I, _, def) {
      u_sparse_bitset_clear(live, def);
   }

   jay_foreach_src_index(I, _, comp, index) {
      u_sparse_bitset_set(live, index);
   }
}

static void
add_edge(struct sched_ctx *ctx, uint32_t edge, uint32_t first_node, bool strong)
{
   if (edge && edge >= first_node) {
      jay_dag_add_edge(&ctx->dag, edge, strong);
   }
}

static void
add_reader(struct sched_ctx *ctx,
           uint32_t writer,
           struct util_dynarray *readers,
           uint32_t first_node)
{
   /* Read-after-write */
   add_edge(ctx, writer, first_node, true);

   /* Track for write-after-read but do not add a dependency, we
    * want to reorder readers freely.
    */
   util_dynarray_append(readers, ctx->dag.node);
}

static void
add_writer(struct sched_ctx *ctx,
           uint32_t *writer,
           struct util_dynarray *readers,
           uint32_t first_node)
{
   /* Write-after-write */
   add_edge(ctx, *writer, first_node, true);
   *writer = ctx->dag.node;

   /* Write-after-read, this is a weak edge */
   util_dynarray_foreach(readers, uint32_t, it) {
      add_edge(ctx, *it, first_node, false);
   }

   util_dynarray_clear(readers);
}

static void
populate_dag(struct sched_ctx *ctx, jay_block *block)
{
   uint32_t first_node = ctx->dag.node;

   /* TODO: Reorder memory instructions */
   uint32_t sidefx = 0, address = 0;
   uint32_t helper = 0;

   jay_foreach_inst_in_block(block, I) {
      if (jay_op_starts_block(I->op)) {
         continue;
      } else if (jay_op_ends_block(I->op)) {
         break;
      }

      if (ctx->phase < POSTRA) {
         /* Uses depend on definitions. SSA form forbids WaR and WaW hazards */
         jay_foreach_src_index(I, s, c, index) {
            add_edge(ctx, ctx->prera.def[index], first_node, true);
         }

         jay_foreach_dst_index(I, d, index) {
            ctx->prera.def[index] = ctx->dag.node;
         }
      } else {
         jay_def dsts[3] = { I->dst, I->cond_flag };

         /* MUL_32 is a macro implicitly clobbering integer acc0 */
         if (I->op == JAY_OPCODE_MUL_32) {
            dsts[2] = jay_bare_reg(ACCUM, 32, 0);
         }

         for (unsigned d = 0; d < ARRAY_SIZE(dsts); ++d) {
            struct jay_footprint fp =
               jay_def_to_footprint(ctx->func, I, dsts[d], I->type);

            jay_foreach_in_footprint(fp, key) {
               add_writer(ctx, &ctx->postra.writer[key],
                          &ctx->postra.readers[key], first_node);
            }
         }

         jay_foreach_src(I, s) {
            struct jay_footprint fp =
               jay_def_to_footprint(ctx->func, I, I->src[s],
                                    jay_src_type(I, s));

            jay_foreach_in_footprint(fp, key) {
               add_reader(ctx, ctx->postra.writer[key],
                          &ctx->postra.readers[key], first_node);
            }
         }
      }

      /* Serialize address register access until we have an address RA */
      bool use_a0 = I->dst.file == J_ADDRESS || I->op == JAY_OPCODE_SHUFFLE;
      jay_foreach_src(I, s) {
         use_a0 |= I->src[s].file == J_ADDRESS;
      }

      if (use_a0) {
         jay_dag_add_edge(&ctx->dag, address, false);
         address = ctx->dag.node;
      }

      /* Serialize side effects for now, including SENDs which need to be
       * predicated away after a demote.
       */
      if ((I->op == JAY_OPCODE_SEND && !jay_send_pure(I)) ||
          I->op == JAY_OPCODE_SCHEDULE_BARRIER ||
          I->op == JAY_OPCODE_CHECK_TDR ||
          (I->op == JAY_OPCODE_MOV &&
           I->src[0].file == J_ARF &&
           jay_base_index(I->src[0]) == GEN_ARF_TIMESTAMP)) {

         jay_dag_add_edge(&ctx->dag, sidefx, true);
         sidefx = ctx->dag.node;
      }

      /* The "is helper?" flag is a hidden register, track as if post-RA */
      if (I->op == JAY_OPCODE_IS_HELPER ||
          (I->op == JAY_OPCODE_SEND &&
           ctx->func->shader->helpers_tracked &&
           jay_send_skip_helpers(I))) {

         add_reader(ctx, helper, &ctx->helper_reads, first_node);
      } else if (I->op == JAY_OPCODE_DEMOTE) {
         add_writer(ctx, &helper, &ctx->helper_reads, first_node);
      }

      ctx->insts[ctx->dag.node] = I;
      jay_dag_next_node(&ctx->dag);
   }

   ctx->blocks[block->index].first = first_node;
   ctx->blocks[block->index].last = ctx->dag.node - 1;
}

/*
 * Due to multiple register files, register demand is a vector. Our dynamic
 * register file partitioning sometimes justifies modelling demand as a single
 * scalar, where each file has a weight determined here.
 */
static signed
weighted_demand(struct sched_ctx *ctx, signed *demands)
{
   return (demands[GPR] * ctx->func->shader->dispatch_width) +
          (demands[UGPR] + demands[FLAG]);
}

static bool
near_register_limit(struct sched_ctx *s, unsigned headroom)
{
   jay_shader *shader = s->func->shader;

   return s->demand[GPR] >= s->demand_limit[GPR] - (int32_t) headroom ||
          s->demand[UGPR] >= s->demand_limit[UGPR] -
                                (int32_t) (headroom * jay_ugpr_per_gpr(shader));
}

/*
 * Calculate the change in register pressure from scheduling a given
 * instuction. Based on jay_calculate_register_demands, but without the use of
 * kill-bits since we are reordering instructions.
 */
static void
adjust_demand_before(struct sched_ctx *ctx, jay_inst *I, signed *demand)
{
   assert(ctx->phase < POSTRA);

   /* Make destinations live */
   jay_foreach_dst(I, dst) {
      demand[dst.file] -= jay_num_values(dst);
   }
}

static void
adjust_demand_after(struct sched_ctx *ctx, jay_inst *I, signed *demand)
{
   assert(ctx->phase < POSTRA);

   /* Dead destinations are those written by the instruction but killed
    * immediately after the instruction finishes.
    */
   jay_foreach_dst_index(I, dst, index) {
      assert(dst.file < JAY_NUM_SSA_FILES);
      demand[dst.file] += !u_sparse_bitset_test(&ctx->live, index);
   }

   /* Late-kill sources. TODO: Is there a better way to deduplicate? */
   jay_foreach_src_index(I, s, c, index) {
      if (!BITSET_TEST(ctx->seen, index)) {
         demand[I->src[s].file] += !u_sparse_bitset_test(&ctx->live, index);
      }

      BITSET_SET(ctx->seen, index);
   }

   jay_foreach_src_index(I, _, c, index) {
      BITSET_CLEAR(ctx->seen, index);
   }
}

static inline unsigned
ready_cycle(struct sched_ctx *s, bool backward, uint32_t node)
{
   unsigned cycle = s->cycle;
   uint32_t lat = backward ? jay_latency(s->func->shader, s->insts[node],
                                         s->phase < POSTACC) :
                             0;
   struct jay_dag *dag = backward ? &s->dag_t : &s->dag;

   jay_dag_foreach_edge(dag, node, it) {
      if (it->strong) {
         cycle = MAX2(cycle, s->cycle_ready[it->node] + lat);
      }
   }

   return cycle;
}

static int32_t
choose_inst(struct sched_ctx *s, enum sched_mode mode)
{
   unsigned latency_weight = s->phase > EARLY ? 1 : 0;
   unsigned pressure_weight = (mode & PRESSURE) ? 1 : 0;

   if (s->phase > EARLY && (mode & PRESSURE)) {
      if ((mode & LATENCY) && s->use_headroom) {
         pressure_weight = near_register_limit(s, s->headroom);
         latency_weight = !pressure_weight;
      } else {
         latency_weight = s->aggression;
         pressure_weight = AGGRESSION_SCALE;
      }
   }

   int32_t min_score = INT32_MAX;
   int32_t best = -1;

   util_dynarray_foreach(&s->it.heads, uint32_t, head) {
      jay_inst *I = s->insts[*head];
      int32_t score = 0;
      if (mode & PRESSURE) {
         assert(s->phase < POSTRA);

         /* To minimize pressure, consider the effect on liveness. */
         int32_t deltas[JAY_NUM_SSA_FILES] = { 0 };
         adjust_demand_after(s, I, deltas);
         adjust_demand_before(s, I, deltas);
         int32_t delta = weighted_demand(s, deltas);

         /* As a tiebreaker (only), sink flag writes to reduce specifically flag
          * pressure, because spilling flags costs extra instructions and GPR
          * pressure. This is a mildly positive heuristic.
          */
         delta *= 2;
         if (jay_is_null(I->cond_flag)) {
            delta++;
         }

         score += delta * pressure_weight;

         if (s->phase > EARLY) {
            memcpy(deltas, s->demand, JAY_NUM_SSA_FILES * sizeof(deltas[0]));
            adjust_demand_after(s, I, deltas);
            bool skip = false;
            jay_foreach_ssa_file(file) {
               skip |= (deltas[file] > s->demand_limit[file]);
            }
            if (skip) {
               continue;
            }
         }
      }

      if (latency_weight) {
         score += latency_weight *
                  (ready_cycle(s, mode & BACKWARD, *head) - s->cycle);
      }

      if (score < min_score) {
         best = *head;
         min_score = score;
      } else if (score == min_score) {
         if (s->phase != EARLY) {
            if ((mode & BACKWARD && *head > best) ||
                (!(mode & BACKWARD) && *head < best)) {
               best = *head;
            }
         } else {
            best = *head;
         }
      }
   }

   return best;
}

static void
gather_block_info(struct sched_ctx *s, jay_block *block, void *memctx)
{
   int32_t demand[JAY_NUM_SSA_FILES] = { 0 };
   signed max_pressure = 0;
   unsigned q = 0;
   s->cycle = 0;

   if (s->phase == EARLY) {
      u_sparse_bitset_free(&s->live);
      u_sparse_bitset_dup_with_ctx(&s->live, &block->live_out, memctx);
   }

   jay_foreach_inst_in_block_rev(block, I) {
      if (jay_op_starts_block(I->op)) {
         break;
      } else if (jay_op_ends_block(I->op) && s->phase == EARLY) {
         continue;
      }

      if (s->phase == EARLY) {
         adjust_demand_after(s, I, demand);
         max_pressure = MAX2(weighted_demand(s, demand), max_pressure);
         adjust_demand_before(s, I, demand);
         liveness_update(&s->live, I);
      }

      /* Ignore branches for latency calculations */
      if (s->phase > EARLY && !jay_op_ends_block(I->op)) {
         uint32_t node = s->blocks[block->index].last - (q++);
         s->cycle = ready_cycle(s, true /* backward */, node);
         s->cycle_ready[node] = s->cycle;
         s->cycle++;
      }
   }

   s->blocks[block->index].max_pressure = max_pressure;
   s->blocks[block->index].latency = s->cycle;
}

enum sched_result { SUCCESS, FAIL_PRESSURE, FAIL_LATENCY };

static enum sched_result
schedule_block(jay_block *block,
               struct sched_ctx *s,
               void *memctx,
               enum sched_mode mode)
{
   /* The DAG we constructed corresponds to a backwards walk, so for a
    * forwards walk just iterate the transposed DAG instead.
    */
   s->it.dag = mode & BACKWARD ? &s->dag : &s->dag_t;
   jay_dag_iterate(&s->it, s->blocks[block->index].first,
                   s->blocks[block->index].last);

   util_dynarray_clear(&s->schedule);
   s->cycle = 0;

   if (s->phase < POSTRA) {
      memset(s->demand, 0, JAY_NUM_SSA_FILES * sizeof(s->demand[0]));
      u_sparse_bitset_free(&s->live);

      if (s->phase == EARLY) {
         memset(s->demand, 0, JAY_NUM_SSA_FILES * sizeof(s->demand[0]));
      } else {
         typed_memcpy(s->demand, block->demand_out, JAY_NUM_SSA_FILES);
      }

      u_sparse_bitset_dup_with_ctx(&s->live, &block->live_out, memctx);
   }

   if (s->phase > EARLY && s->phase < POSTRA) {
      jay_foreach_inst_in_block_rev(block, I) {
         if (!jay_op_ends_block(I->op)) {
            break;
         } else if (I->op != JAY_OPCODE_PHI_SRC) {
            adjust_demand_after(s, I, s->demand);
            adjust_demand_before(s, I, s->demand);
            liveness_update(&s->live, I);
         }
      }
   }

   while (s->it.heads.size) {
      int32_t node = choose_inst(s, mode);

      /* If there is no instruction picked, bail on the schedule. */
      if (node < 0) {
         jay_dag_iterator_reset(&s->it);
         return FAIL_PRESSURE;
      }

      if (s->phase < POSTRA && (mode & BACKWARD)) {
         adjust_demand_after(s, s->insts[node], s->demand);

         /* Toss schedules that blow up register pressure as we go */
         if (s->phase == EARLY) {
            if (weighted_demand(s, s->demand) >=
                s->blocks[block->index].max_pressure) {
               jay_dag_iterator_reset(&s->it);
               return FAIL_PRESSURE;
            }
         } else {
            jay_foreach_ssa_file(file) {
               if (s->demand[file] > s->demand_limit[file]) {
                  while (s->it.heads.size) {
                     jay_dag_iterator_reset(&s->it);
                     return FAIL_PRESSURE;
                  }
               }
            }
         }

         adjust_demand_before(s, s->insts[node], s->demand);
         liveness_update(&s->live, s->insts[node]);
      }

      jay_dag_take_head(&s->it, node);
      util_dynarray_append(&s->schedule, node);

      if (s->phase > EARLY) {
         s->cycle = ready_cycle(s, mode & BACKWARD, node);
         s->cycle_ready[node] =
            s->cycle + ((mode & BACKWARD) ?
                           0 :
                           jay_latency(s->func->shader, s->insts[node],
                                       s->phase < POSTACC));
         s->cycle++;
      }
   }

   /* We don't have pressure information available during the forward pass, so
    * determine it after scheduling (iterating backwards).
    */
   if (s->phase < POSTRA && !(mode & BACKWARD)) {
      util_dynarray_foreach_reverse(&s->schedule, uint32_t, node) {
         adjust_demand_after(s, s->insts[*node], s->demand);

         jay_foreach_ssa_file(file) {
            if (s->demand[file] > s->demand_limit[file]) {
               jay_dag_iterator_reset(&s->it);
               return FAIL_PRESSURE;
            }
         }

         adjust_demand_before(s, s->insts[*node], s->demand);
         liveness_update(&s->live, s->insts[*node]);
      }
   }

   /* During latency scheduling, only take schedules that improve latency */
   if (s->phase > EARLY && s->cycle >= s->blocks[block->index].latency) {
      return FAIL_LATENCY;
   }

   /* Apply schedule */
   util_dynarray_foreach(&s->schedule, uint32_t, node) {
      jay_remove_instruction(s->insts[*node]);
   }

   jay_builder b = jay_init_builder(s->func, jay_before_block(block));

   if (mode & BACKWARD) {
      util_dynarray_foreach_reverse(&s->schedule, uint32_t, node) {
         jay_builder_insert(&b, s->insts[*node]);
      }
   } else {
      util_dynarray_foreach(&s->schedule, uint32_t, node) {
         jay_builder_insert(&b, s->insts[*node]);
      }
   }

   s->blocks[block->index].latency = s->cycle;
   return SUCCESS;
}

static void
pass(jay_function *f)
{
   if (jay_debug & JAY_DBG_NOSCHED) {
      return;
   }

   struct sched_ctx sctx = { .func = f };
   sctx.phase = f->shader->post_acc               ? POSTACC :
                f->shader->post_ra                ? POSTRA :
                f->shader->partition.units_x16[0] ? POSTSPILL :
                                                    EARLY;

   void *memctx = ralloc_context(NULL);
   void *linctx = linear_context(memctx);

   uint32_t nr_inst = 1;
   jay_foreach_inst_in_func(f, _, I) {
      ++nr_inst;
   }

   if (sctx.phase >= POSTRA) {
      uint32_t keys = jay_footprint_base(f->shader, ~0);
      sctx.postra.writer = linear_zalloc_array(linctx, uint32_t, keys);
      sctx.postra.readers =
         linear_zalloc_array(linctx, struct util_dynarray, keys);

      for (unsigned i = 0; i < keys; ++i) {
         util_dynarray_init(&sctx.postra.readers[i], memctx);
      }
   } else {
      jay_compute_liveness(f);
      jay_calculate_last_use(f);
      jay_calculate_register_demands(f);
      sctx.seen = BITSET_LINEAR_ZALLOC(linctx, f->ssa_alloc);
      sctx.prera.def = linear_zalloc_array(linctx, uint32_t, f->ssa_alloc);
   }

   util_dynarray_init(&sctx.helper_reads, memctx);

   sctx.insts = linear_alloc_array(linctx, jay_inst *, nr_inst);
   sctx.cycle_ready = linear_zalloc_array(linctx, uint32_t, nr_inst);
   sctx.blocks = linear_zalloc_array(linctx, struct sched_block, f->num_blocks);
   jay_dag_init(&sctx.dag, memctx, nr_inst);
   jay_dag_iterator_init(&sctx.it, &sctx.dag);

   /* Build the DAG for the whole program and transpose it */
   jay_foreach_block(f, block) {
      populate_dag(&sctx, block);
   }

   jay_dag_transpose(&sctx.dag_t, &sctx.dag);

   if (sctx.phase == POSTSPILL) {
      jay_foreach_ssa_file(file) {
         sctx.demand_limit[file] = f->shader->num_regs[file];
      }

      for (unsigned i = 0; i < f->shader->partition.nr_blocks[UGPR]; ++i) {
         if (f->shader->partition.blocks[UGPR][i].type == JAY_BLOCK_SPILL) {
            sctx.demand_limit[UGPR] -=
               f->shader->partition.blocks[UGPR][i].len_gpr;
         }
      }

      /* XXX: common code */
      if (f->shader->helpers_tracked)
         sctx.demand_limit[FLAG]--;
   }

   jay_foreach_block(f, block) {
      /* Gather reference statistics about the program performance */
      gather_block_info(&sctx, block, memctx);

      /* Do pressure-only scheduling only on blocks that might spill, to
       * minimize harm. We use a conservative threshold to leave wiggle room for
       * late lowerings.
       */
      if (sctx.phase == EARLY) {
         unsigned demand_ugpr =
            block->demand_max[UGPR] + block->demand_max[FLAG];
         unsigned demand_gpr = block->demand_max[GPR];

         unsigned sched_at = f->shader->devinfo->ver >= 30 ? 64 : 120;
         unsigned prioritize_at = f->shader->devinfo->ver >= 30 ? 90 : 120;

         unsigned demand =
            (demand_gpr * jay_ugpr_per_gpr(f->shader)) + demand_ugpr;

         if (demand >= sched_at * jay_ugpr_per_grf(f->shader)) {
            f->prioritize_pressure =
               demand >= prioritize_at * jay_ugpr_per_grf(f->shader);
            schedule_block(block, &sctx, memctx, BACKWARD | PRESSURE);
         }
      } else if (sctx.phase == POSTSPILL) {
         /* Try to schedule forwards & backwards */
         enum sched_result fwd = schedule_block(block, &sctx, memctx, LATENCY);
         enum sched_result bw =
            schedule_block(block, &sctx, memctx, BACKWARD | LATENCY);

         /* Compare pressure-aware aggression and headroom schedules. */
         if (fwd != SUCCESS && bw == FAIL_PRESSURE) {
            /* 20 down to 2 in steps of 2, then 1.5, 1, 1/2, 1/4, 1/8 */
            static const uint8_t ladder[] = { 160, 144, 128, 112, 96,
                                              80,  64,  48,  32,  16,
                                              12,  8,   4,   2,   1 };
            static const unsigned headrooms[] = { 0, 1, 2, 4, 8 };

            sctx.use_headroom = false;
            for (unsigned i = 0; i < ARRAY_SIZE(ladder) && bw != SUCCESS; i++) {
               sctx.aggression = ladder[i];
               bw = schedule_block(block, &sctx, memctx,
                                   BACKWARD | PRESSURE | LATENCY);
            }

            if (sctx.aggression <= AGGRESSION_SCALE) {
               sctx.use_headroom = true;
               for (unsigned i = 0; i < ARRAY_SIZE(headrooms); i++) {
                  sctx.headroom = headrooms[i];
                  schedule_block(block, &sctx, memctx,
                                 BACKWARD | PRESSURE | LATENCY);
               }
            }
         }
      } else {
         schedule_block(block, &sctx, memctx, LATENCY);
      }
   }

   util_dynarray_fini(&sctx.schedule);
   ralloc_free(memctx);
}

JAY_DEFINE_FUNCTION_PASS(jay_schedule, pass)
