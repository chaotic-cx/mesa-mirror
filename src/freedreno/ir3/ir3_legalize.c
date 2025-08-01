/*
 * Copyright © 2014 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "util/ralloc.h"
#include "util/u_math.h"

#include "ir3.h"
#include "ir3_shader.h"

/*
 * Legalize:
 *
 * The legalize pass handles ensuring sufficient nop's and sync flags for
 * correct execution.
 *
 * 1) Iteratively determine where sync ((sy)/(ss)) flags are needed,
 *    based on state flowing out of predecessor blocks until there is
 *    no further change.  In some cases this requires inserting nops.
 * 2) Mark (ei) on last varying input
 * 3) Final nop scheduling for instruction latency
 * 4) Resolve jumps and schedule blocks, marking potential convergence
 *    points with (jp)
 */

struct ir3_legalize_ctx {
   struct ir3_compiler *compiler;
   struct ir3_shader_variant *so;
   gl_shader_stage type;
   int max_bary;
   bool early_input_release;
   bool has_inputs;
   bool has_tex_prefetch;
};

struct ir3_legalize_block_data {
   bool valid;
   struct ir3_legalize_state begin_state;
   struct ir3_legalize_state state;
};

static inline bool
needs_ss_war(struct ir3_legalize_state *state, struct ir3_register *dst,
             bool is_scalar_alu, enum ir3_instruction_flags cur_flags)
{
   if (regmask_get(&state->needs_ss_war, dst))
      return true;
   if (!(cur_flags & IR3_INSTR_SY) &&
       regmask_get(&state->needs_ss_or_sy_war, dst)) {
      return true;
   }

   if (!is_scalar_alu) {
      if (regmask_get(&state->needs_ss_scalar_war, dst))
         return true;
      if (!(cur_flags & IR3_INSTR_SY) &&
          regmask_get(&state->needs_ss_or_sy_scalar_war, dst)) {
         return true;
      }
   }

   return false;
}

static inline bool
needs_sy_war(struct ir3_legalize_state *state, struct ir3_register *dst)
{
   return regmask_get(&state->needs_sy_war, dst);
}

enum ir3_instruction_flags
ir3_required_sync_flags(struct ir3_legalize_state *state,
                        struct ir3_compiler *compiler,
                        struct ir3_instruction *n)
{
   if (n->opc == OPC_SHPE) {
      return IR3_INSTR_SS | IR3_INSTR_SY;
   }

   enum ir3_instruction_flags flags = 0;
   bool n_is_scalar_alu = is_scalar_alu(n, compiler);

   if (state->force_ss) {
      flags |= IR3_INSTR_SS;
   }

   if (state->force_sy) {
      flags |= IR3_INSTR_SY;
   }

   /* NOTE: consider dst register too.. it could happen that
    * texture sample instruction (for example) writes some
    * components which are unused.  A subsequent instruction
    * that writes the same register can race w/ the sam instr
    * resulting in undefined results:
    */
   for (unsigned i = 0; i < n->dsts_count + n->srcs_count; i++) {
      struct ir3_register *reg;
      if (i < n->dsts_count)
         reg = n->dsts[i];
      else
         reg = n->srcs[i - n->dsts_count];

      if (reg->flags & IR3_REG_DUMMY)
         continue;

      if (is_reg_gpr(reg)) {

         /* TODO: we probably only need (ss) for alu
          * instr consuming sfu result.. need to make
          * some tests for both this and (sy)..
          */
         if (regmask_get(&state->needs_ss, reg)) {
            flags |= IR3_INSTR_SS;
         }

         /* There is a fast feedback path for scalar ALU instructions which
          * only takes 1 cycle of latency, similar to the normal 3 cycle
          * latency path for ALU instructions. For this fast path the
          * producer and consumer must use the same register size (i.e. no
          * writing a full register and then reading half of it or vice
          * versa). If we don't hit this path, either because of a mismatched
          * size or a read via the regular ALU, then the write latency is
          * variable and we must use (ss) to wait for the scalar ALU. This is
          * different from the fixed 6 cycle latency for mismatched vector
          * ALU accesses.
          */
         if (n_is_scalar_alu) {
            /* Check if we have a mismatched size RaW dependency */
            if (regmask_get((reg->flags & IR3_REG_HALF)
                               ? &state->needs_ss_scalar_half
                               : &state->needs_ss_scalar_full,
                            reg)) {
               flags |= IR3_INSTR_SS;
            }
         } else {
            /* check if we have a scalar -> vector RaW dependency */
            if (regmask_get(&state->needs_ss_scalar_half, reg) ||
                regmask_get(&state->needs_ss_scalar_full, reg)) {
               flags |= IR3_INSTR_SS;
            }
         }

         if (regmask_get(&state->needs_sy, reg)) {
            flags |= IR3_INSTR_SY;
         }
      } else if ((reg->flags & IR3_REG_CONST)) {
         if (state->needs_ss_for_const) {
            flags |= IR3_INSTR_SS;
         }
         if (state->needs_sy_for_const) {
            flags |= IR3_INSTR_SY;
         }
      } else if (reg_is_addr1(reg) && n->block->in_early_preamble) {
         if (regmask_get(&state->needs_ss, reg)) {
            flags |= IR3_INSTR_SS;
         }
      }
   }

   foreach_dst (reg, n) {
      if (reg->flags & (IR3_REG_RT | IR3_REG_DUMMY))
         continue;
      if (needs_sy_war(state, reg)) {
         flags |= IR3_INSTR_SY;
      }
      if (needs_ss_war(state, reg, n_is_scalar_alu, flags)) {
         flags |= IR3_INSTR_SS;
      }
   }

   return flags;
}

static inline void
apply_ss(struct ir3_legalize_state *state, bool mergedregs)
{
   regmask_init(&state->needs_ss_war, mergedregs);
   regmask_init(&state->needs_ss_or_sy_war, mergedregs);
   regmask_init(&state->needs_ss, mergedregs);
   regmask_init(&state->needs_ss_scalar_war, mergedregs);
   regmask_init(&state->needs_ss_or_sy_scalar_war, mergedregs);
   regmask_init(&state->needs_ss_scalar_full, mergedregs);
   regmask_init(&state->needs_ss_scalar_half, mergedregs);
   state->needs_ss_for_const = false;
   state->force_ss = false;
}

static inline void
apply_sy(struct ir3_legalize_state *state, bool mergedregs)
{
   regmask_init(&state->needs_sy, mergedregs);
   regmask_init(&state->needs_sy_war, mergedregs);
   regmask_init(&state->needs_ss_or_sy_war, mergedregs);
   regmask_init(&state->needs_ss_or_sy_scalar_war, mergedregs);
   state->needs_sy_for_const = false;
   state->force_sy = false;
}

static void
sync_update(struct ir3_legalize_state *state, struct ir3_compiler *compiler,
            struct ir3_instruction *n)
{
   if (n->flags & IR3_INSTR_SS)
      apply_ss(state, compiler->mergedregs);
   if (n->flags & IR3_INSTR_SY)
      apply_sy(state, compiler->mergedregs);

   if (is_barrier(n)) {
      state->force_ss = true;
      state->force_sy = true;
   } else {
      state->force_ss = false;
      state->force_sy = false;
   }

   bool n_is_scalar_alu = is_scalar_alu(n, compiler);

   if (is_sfu(n) || n->opc == OPC_SHFL)
      regmask_set(&state->needs_ss, n->dsts[0]);

   foreach_dst (dst, n) {
      if (dst->flags & IR3_REG_SHARED) {
         if (n_is_scalar_alu) {
            if (dst->flags & IR3_REG_HALF)
               regmask_set(&state->needs_ss_scalar_full, dst);
            else
               regmask_set(&state->needs_ss_scalar_half, dst);
         } else {
            regmask_set(&state->needs_ss, dst);
         }
      } else if (reg_is_addr1(dst) && n->block->in_early_preamble) {
         regmask_set(&state->needs_ss, dst);
      }
   }

   if (is_tex_or_prefetch(n) && !has_dummy_dst(n)) {
      regmask_set(&state->needs_sy, n->dsts[0]);
   } else if (n->opc == OPC_RESINFO && !has_dummy_dst(n)) {
      regmask_set(&state->needs_ss, n->dsts[0]);
   } else if (is_load(n)) {
      if (is_local_mem_load(n))
         regmask_set(&state->needs_ss, n->dsts[0]);
      else
         regmask_set(&state->needs_sy, n->dsts[0]);
   } else if (is_atomic(n->opc)) {
      if (is_bindless_atomic(n->opc)) {
         regmask_set(&state->needs_sy, n->srcs[2]);
      } else if (is_global_a3xx_atomic(n->opc) ||
                 is_global_a6xx_atomic(n->opc)) {
         regmask_set(&state->needs_sy, n->dsts[0]);
      } else {
         regmask_set(&state->needs_ss, n->dsts[0]);
      }
   } else if (n->opc == OPC_PUSH_CONSTS_LOAD_MACRO || n->opc == OPC_STC) {
      state->needs_ss_for_const = true;
   } else if (n->opc == OPC_LDC_K) {
      state->needs_sy_for_const = true;
   }

   /* both tex/sfu appear to not always immediately consume
    * their src register(s):
    */
   if (is_war_hazard_producer(n)) {
      /* These WAR hazards can always be resolved with (ss). However, when
       * the reader is a sy-producer, they can also be resolved using (sy)
       * because once we have synced the reader's results using (sy), its
       * sources have definitely been consumed. We track the two cases
       * separately so that we don't add an unnecessary (ss) if a (sy) sync
       * already happened.
       * For example, this prevents adding the unnecessary (ss) in the
       * following sequence:
       * sam rd, rs, ...
       * (sy)... ; sam synced so consumed its sources
       * (ss)write rs ; (ss) unnecessary since rs has been consumed already
       */
      bool needs_ss = is_ss_producer(n) || is_store(n) || n->opc == OPC_STC;

      /* It seems like ray_intersection WAR hazards cannot be resolved using
       * (ss) and need a (sy) sync instead.
       */
      bool needs_sy = n->opc == OPC_RAY_INTERSECTION;

      if (n_is_scalar_alu) {
         /* Scalar ALU also does not immediately read its source because it
          * is not executed right away, but scalar ALU instructions are
          * executed in-order so subsequent scalar ALU instructions don't
          * need to wait for previous ones.
          */
         regmask_t *mask = needs_ss ? &state->needs_ss_scalar_war
                                    : &state->needs_ss_or_sy_scalar_war;

         foreach_src (reg, n) {
            if ((reg->flags & IR3_REG_SHARED) || is_reg_a0(reg)) {
               regmask_set(mask, reg);
            }
         }
      } else {
         regmask_t *mask = needs_sy   ? &state->needs_sy_war
                           : needs_ss ? &state->needs_ss_war
                                      : &state->needs_ss_or_sy_war;

         foreach_src (reg, n) {
            if (!(reg->flags & (IR3_REG_IMMED | IR3_REG_CONST))) {
               regmask_set(mask, reg);
            }
         }
      }
   }
}

void
ir3_init_legalize_state(struct ir3_legalize_state *state,
                        struct ir3_compiler *compiler)
{
   regmask_init(&state->needs_ss_war, compiler->mergedregs);
   regmask_init(&state->needs_sy_war, compiler->mergedregs);
   regmask_init(&state->needs_ss_or_sy_war, compiler->mergedregs);
   regmask_init(&state->needs_ss_scalar_war, compiler->mergedregs);
   regmask_init(&state->needs_ss_or_sy_scalar_war, compiler->mergedregs);
   regmask_init(&state->needs_ss_scalar_full, compiler->mergedregs);
   regmask_init(&state->needs_ss_scalar_half, compiler->mergedregs);
   regmask_init(&state->needs_ss, compiler->mergedregs);
   regmask_init(&state->needs_sy, compiler->mergedregs);
}

static struct ir3_legalize_state *
get_block_legalize_state(struct ir3_block *block)
{
   struct ir3_legalize_block_data *bd = block->data;
   return &bd->state;
}

void
ir3_merge_pred_legalize_states(struct ir3_legalize_state *state,
                               struct ir3_block *block,
                               ir3_get_block_legalize_state_cb get_state)
{
   /* Our input state is the OR of all predecessor blocks' state.
    *
    * Why don't we just zero the state at the beginning before merging in the
    * predecessors? Because otherwise updates may not be a "lattice refinement",
    * i.e. needs_ss may go from true to false for some register due to a (ss) we
    * inserted the second time around (and the same for (sy)). This means that
    * there's no solid guarantee the algorithm will converge, and in theory
    * there may be infinite loops where we fight over the placment of an (ss).
    */
   for (unsigned i = 0; i < block->predecessors_count; i++) {
      struct ir3_block *predecessor = block->predecessors[i];
      struct ir3_legalize_state *pstate = get_state(predecessor);

      if (!pstate) {
         continue;
      }

      /* Our input (ss)/(sy) state is based on OR'ing the output
       * state of all our predecessor blocks
       */
      regmask_or(&state->needs_ss, &state->needs_ss, &pstate->needs_ss);
      regmask_or(&state->needs_ss_war, &state->needs_ss_war,
                 &pstate->needs_ss_war);
      regmask_or(&state->needs_sy_war, &state->needs_sy_war,
                 &pstate->needs_sy_war);
      regmask_or(&state->needs_ss_or_sy_war, &state->needs_ss_or_sy_war,
                 &pstate->needs_ss_or_sy_war);
      regmask_or(&state->needs_sy, &state->needs_sy, &pstate->needs_sy);
      state->needs_ss_for_const |= pstate->needs_ss_for_const;
      state->needs_sy_for_const |= pstate->needs_sy_for_const;
      state->force_ss |= pstate->force_ss;
      state->force_sy |= pstate->force_sy;

      /* Our nop state is the max of the predecessor blocks. The predecessor nop
       * state contains the cycle offset from the start of its block when each
       * register becomes ready. But successor blocks need the cycle offset from
       * their start, which is the predecessor's block's end. Translate the
       * cycle offset.
       */
      for (unsigned i = 0; i < ARRAY_SIZE(state->pred_ready); i++)
         state->pred_ready[i] =
            MAX2(state->pred_ready[i],
                 MAX2(pstate->pred_ready[i], pstate->cycle) - pstate->cycle);
      for (unsigned i = 0; i < ARRAY_SIZE(state->alu_nop.full_ready); i++) {
         state->alu_nop.full_ready[i] = MAX2(
            state->alu_nop.full_ready[i],
            MAX2(pstate->alu_nop.full_ready[i], pstate->cycle) - pstate->cycle);
         state->alu_nop.half_ready[i] = MAX2(
            state->alu_nop.half_ready[i],
            MAX2(pstate->alu_nop.half_ready[i], pstate->cycle) - pstate->cycle);
         state->non_alu_nop.full_ready[i] =
            MAX2(state->non_alu_nop.full_ready[i],
                 MAX2(pstate->non_alu_nop.full_ready[i], pstate->cycle) -
                    pstate->cycle);
         state->non_alu_nop.half_ready[i] =
            MAX2(state->non_alu_nop.half_ready[i],
                 MAX2(pstate->non_alu_nop.half_ready[i], pstate->cycle) -
                    pstate->cycle);
      }
   }

   /* We need to take phsyical-only edges into account when tracking shared
    * registers.
    */
   for (unsigned i = 0; i < block->physical_predecessors_count; i++) {
      struct ir3_block *predecessor = block->physical_predecessors[i];
      struct ir3_legalize_state *pstate = get_state(predecessor);

      if (!pstate) {
         continue;
      }

      regmask_or_shared(&state->needs_ss, &state->needs_ss, &pstate->needs_ss);
      regmask_or_shared(&state->needs_ss_scalar_full,
                        &state->needs_ss_scalar_full,
                        &pstate->needs_ss_scalar_full);
      regmask_or_shared(&state->needs_ss_scalar_half,
                        &state->needs_ss_scalar_half,
                        &pstate->needs_ss_scalar_half);
      regmask_or_shared(&state->needs_ss_scalar_war,
                        &state->needs_ss_scalar_war,
                        &pstate->needs_ss_scalar_war);
      regmask_or_shared(&state->needs_ss_or_sy_scalar_war,
                        &state->needs_ss_or_sy_scalar_war,
                        &pstate->needs_ss_or_sy_scalar_war);
   }

   gl_shader_stage stage = block->shader->type;

   if (stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_GEOMETRY) {
      if (block == ir3_start_block(block->shader)) {
         state->force_ss = true;
         state->force_sy = true;
      }
   }
}

static bool
count_instruction(struct ir3_instruction *n, struct ir3_compiler *compiler)
{
   /* NOTE: don't count branch/jump since we don't know yet if they will
    * be eliminated later in resolve_jumps().. really should do that
    * earlier so we don't have this constraint.
    */
   return (is_alu(n) && !is_scalar_alu(n, compiler)) ||
      (is_flow(n) && (n->opc != OPC_JUMP) && (n->opc != OPC_BR) &&
           (n->opc != OPC_BRAA) && (n->opc != OPC_BRAO));
}

static unsigned *
get_ready_slot(struct ir3_legalize_state *state,
               struct ir3_register *reg, unsigned num,
               bool consumer_alu, bool matching_size)
{
   if (reg->flags & IR3_REG_PREDICATE) {
      assert(num == reg->num);
      assert(reg_num(reg) == REG_P0);
      return &state->pred_ready[reg_comp(reg)];
   }
   if (reg->num == regid(REG_A0, 0))
      return &state->addr_ready[0];
   if (reg->num == regid(REG_A0, 1))
      return &state->addr_ready[1];
   struct ir3_nop_state *nop =
      consumer_alu ? &state->alu_nop : &state->non_alu_nop;
   assert(!(reg->flags & IR3_REG_SHARED));
   if (reg->flags & IR3_REG_HALF) {
      if (matching_size) {
         assert(num < ARRAY_SIZE(nop->half_ready));
         return &nop->half_ready[num];
      } else {
         assert(num / 2 < ARRAY_SIZE(nop->full_ready));
         return &nop->full_ready[num / 2];
      }
   } else {
      if (matching_size) {
         assert(num < ARRAY_SIZE(nop->full_ready));
         return &nop->full_ready[num];
      }
      /* If "num" is large enough, then it can't alias a half-reg because only
       * the first half of the full reg speace aliases half regs. Return NULL in
       * this case.
       */
      else if (num * 2 < ARRAY_SIZE(nop->half_ready))
         return &nop->half_ready[num * 2];
      else
         return NULL;
   }
}

unsigned
ir3_required_delay(struct ir3_legalize_state *state,
                   struct ir3_compiler *compiler, struct ir3_instruction *instr)
{
   /* As far as we know, shader outputs don't need any delay. */
   if (instr->opc == OPC_END || instr->opc == OPC_CHMASK)
      return 0;

   unsigned delay = 0;
   foreach_src_n (src, n, instr) {
      if (src->flags & (IR3_REG_CONST | IR3_REG_IMMED | IR3_REG_SHARED))
         continue;

      unsigned elems = post_ra_reg_elems(src);
      unsigned num = post_ra_reg_num(src);
      unsigned src_cycle =
         state->cycle + ir3_src_read_delay(compiler, instr, n);

      for (unsigned elem = 0; elem < elems; elem++, num++) {
         unsigned ready_cycle =
            *get_ready_slot(state, src, num, is_alu(instr), true);
         delay = MAX2(delay, MAX2(ready_cycle, src_cycle) - src_cycle);

         /* Increment cycle for ALU instructions with (rptN) where sources are
          * read each subsequent cycle.
          */
         if (instr->repeat && !(src->flags & IR3_REG_RELATIV))
            src_cycle++;
      }
   }

   return delay;
}

static void
delay_update(struct ir3_compiler *compiler,
             struct ir3_legalize_state *state,
             struct ir3_instruction *instr)
{
   if (writes_addr1(instr) && instr->block->in_early_preamble)
      return;

   foreach_dst_n (dst, n, instr) {
      if (dst->flags & (IR3_REG_RT | IR3_REG_DUMMY))
         continue;

      unsigned elems = post_ra_reg_elems(dst);
      unsigned num = post_ra_reg_num(dst);
      unsigned dst_cycle = state->cycle;

      /* sct and swz have scalar destinations and each destination is written in
       * a subsequent cycle.
       */
      if (instr->opc == OPC_SCT || instr->opc == OPC_SWZ)
         dst_cycle += n;

      /* For relative accesses with (rptN), we have no way of knowing which
       * component is accessed when, so we have to assume the worst and mark
       * every array member as being written at the end.
       */
      if (dst->flags & IR3_REG_RELATIV)
         dst_cycle += instr->repeat;

      if (dst->flags & IR3_REG_SHARED)
         continue;

      for (unsigned elem = 0; elem < elems; elem++, num++) {
         /* Don't update delays for registers that aren't actually written. */
         if (!(dst->flags & IR3_REG_RELATIV) && !(dst->wrmask & (1 << elem)))
            continue;

         for (unsigned consumer_alu = 0; consumer_alu < 2; consumer_alu++) {
            for (unsigned matching_size = 0; matching_size < 2; matching_size++) {
               unsigned *ready_slot =
                  get_ready_slot(state, dst, num, consumer_alu, matching_size);

               if (!ready_slot)
                  continue;

               bool reset_ready_slot = false;
               unsigned delay = 0;
               if (!is_alu(instr)) {
                  /* Apparently writes that require (ss) or (sy) are
                   * synchronized against previous writes, so consumers don't
                   * have to wait for any previous overlapping ALU instructions
                   * to complete.
                   */
                  reset_ready_slot = true;
               } else if ((dst->flags & IR3_REG_PREDICATE) ||
                          reg_num(dst) == REG_A0) {
                  delay = compiler->delay_slots.non_alu;
                  if (!matching_size)
                     continue;
               } else {
                  delay = (consumer_alu && matching_size)
                             ? compiler->delay_slots.alu_to_alu
                             : compiler->delay_slots.non_alu;
               }

               if (!matching_size) {
                  for (unsigned i = 0; i < reg_elem_size(dst); i++) {
                     ready_slot[i] =
                        reset_ready_slot ? 0 :
                        MAX2(ready_slot[i], dst_cycle + delay);
                  }
               } else {
                  *ready_slot =
                     reset_ready_slot ? 0 :
                     MAX2(*ready_slot, dst_cycle + delay);
               }
            }
         }

         /* Increment cycle for ALU instructions with (rptN) where destinations
          * are written each subsequent cycle.
          */
         if (instr->repeat && !(dst->flags & IR3_REG_RELATIV))
            dst_cycle++;
      }
   }
}

void
ir3_update_legalize_state(struct ir3_legalize_state *state,
                          struct ir3_compiler *compiler,
                          struct ir3_instruction *n)
{
   sync_update(state, compiler, n);

   bool count = count_instruction(n, compiler);
   if (count)
      state->cycle += 1;

   delay_update(compiler, state, n);

   if (count)
      state->cycle += n->repeat + n->nop;
}

/* We want to evaluate each block from the position of any other
 * predecessor block, in order that the flags set are the union of
 * all possible program paths.
 *
 * To do this, we need to know the output state (needs_ss/ss_war/sy)
 * of all predecessor blocks.  The tricky thing is loops, which mean
 * that we can't simply recursively process each predecessor block
 * before legalizing the current block.
 *
 * How we handle that is by looping over all the blocks until the
 * results converge.  If the output state of a given block changes
 * in a given pass, this means that all successor blocks are not
 * yet fully legalized.
 */

static bool
legalize_block(struct ir3_legalize_ctx *ctx, struct ir3_block *block)
{
   struct ir3_legalize_block_data *bd = block->data;

   if (bd->valid)
      return false;

   struct ir3_instruction *last_n = NULL;
   struct list_head instr_list;
   struct ir3_legalize_state prev_state = bd->state;
   struct ir3_legalize_state *state = &bd->begin_state;
   bool last_input_needs_ss = false;
   struct ir3_builder build = ir3_builder_at(ir3_after_block(block));

   ir3_merge_pred_legalize_states(state, block, get_block_legalize_state);

   memcpy(&bd->state, state, sizeof(*state));
   state = &bd->state;

   unsigned input_count = 0;

   foreach_instr (n, &block->instr_list) {
      if (is_input(n)) {
         input_count++;
      }
   }

   unsigned inputs_remaining = input_count;

   /* Either inputs are in the first block or we expect inputs to be released
    * with the end of the program.
    */
   assert(input_count == 0 || !ctx->early_input_release ||
          block == ir3_after_preamble(block->shader));

   /* remove all the instructions from the list, we'll be adding
    * them back in as we go
    */
   list_replace(&block->instr_list, &instr_list);
   list_inithead(&block->instr_list);

   state->cycle = 0;

   foreach_instr_safe (n, &instr_list) {
      unsigned i;

      n->flags &= ~(IR3_INSTR_SS | IR3_INSTR_SY);

      /* _meta::tex_prefetch instructions removed later in
       * collect_tex_prefetches()
       */
      if (is_meta(n) && (n->opc != OPC_META_TEX_PREFETCH))
         continue;

      if (is_input(n)) {
         struct ir3_register *inloc = n->srcs[0];
         assert(inloc->flags & IR3_REG_IMMED);

         int last_inloc =
            inloc->iim_val + ((inloc->flags & IR3_REG_R) ? n->repeat : 0);
         ctx->max_bary = MAX2(ctx->max_bary, last_inloc);
      }

      bool n_is_scalar_alu = is_scalar_alu(n, ctx->compiler);

      enum ir3_instruction_flags sync_flags =
         ir3_required_sync_flags(state, ctx->compiler, n);
      n->flags |= sync_flags;

      if (sync_flags & IR3_INSTR_SS) {
         last_input_needs_ss = false;
      }

      /* I'm not exactly what this is for, but it seems we need this on every
       * mova1 in early preambles.
       */
      if (writes_addr1(n) && block->in_early_preamble)
         n->srcs[0]->flags |= IR3_REG_R;

      /* cat5+ does not have an (ss) bit, if needed we need to
       * insert a nop to carry the sync flag.  Would be kinda
       * clever if we were aware of this during scheduling, but
       * this should be a pretty rare case:
       */
      if ((n->flags & IR3_INSTR_SS) && !supports_ss(n)) {
         if (last_n && last_n->opc == OPC_NOP) {
            /* Note that reusing the previous nop isn't just an optimization
             * but prevents infinitely adding nops when this block is in a loop
             * and needs to be legalized more than once.
             */
            last_n->flags |= IR3_INSTR_SS;

            /* If we reuse the last nop, we shouldn't do a full state update as
             * its delay has already been taken into account.
             */
            sync_update(state, ctx->compiler, last_n);
         } else {
            struct ir3_instruction *nop = ir3_NOP(&build);
            nop->flags |= IR3_INSTR_SS;
            ir3_update_legalize_state(state, ctx->compiler, nop);
            last_n = nop;
         }

         n->flags &= ~IR3_INSTR_SS;
      }

      unsigned delay = ir3_required_delay(state, ctx->compiler, n);

      /* NOTE: I think the nopN encoding works for a5xx and
       * probably a4xx, but not a3xx.  So far only tested on
       * a6xx.
       */

      if ((delay > 0) && (ctx->compiler->gen >= 6) && last_n &&
          !n_is_scalar_alu &&
          ((opc_cat(last_n->opc) == 2) || (opc_cat(last_n->opc) == 3)) &&
          (last_n->repeat == 0)) {
         /* the previous cat2/cat3 instruction can encode at most 3 nop's: */
         unsigned transfer = MIN2(delay, 3 - last_n->nop);
         last_n->nop += transfer;
         delay -= transfer;
         state->cycle += transfer;
      }

      if ((delay > 0) && last_n && (last_n->opc == OPC_NOP)) {
         /* the previous nop can encode at most 5 repeats: */
         unsigned transfer = MIN2(delay, 5 - last_n->repeat);
         last_n->repeat += transfer;
         delay -= transfer;
         state->cycle += transfer;
      }

      if (delay > 0) {
         assert(delay <= 6);
         ir3_NOP(&build)->repeat = delay - 1;
         state->cycle += delay;
      }

      if (ctx->compiler->samgq_workaround &&
          ctx->type != MESA_SHADER_FRAGMENT &&
          ctx->type != MESA_SHADER_COMPUTE && n->opc == OPC_SAMGQ) {
         struct ir3_instruction *samgp;

         list_delinit(&n->node);

         for (i = 0; i < 4; i++) {
            samgp = ir3_instr_clone(n);
            samgp->opc = OPC_SAMGP0 + i;
            if (i > 1)
               samgp->flags |= IR3_INSTR_SY;
         }
      } else {
         list_delinit(&n->node);
         list_addtail(&n->node, &block->instr_list);
      }

      if (n->opc == OPC_META_TEX_PREFETCH) {
         assert(n->dsts_count > 0);
         ctx->has_tex_prefetch = true;
      }

      if (n->opc == OPC_RESINFO && !has_dummy_dst(n)) {
         ir3_update_legalize_state(state, ctx->compiler, n);

         n = ir3_NOP(&build);
         n->flags |= IR3_INSTR_SS;
         last_input_needs_ss = false;
      }

      if (is_ssbo(n->opc) || is_global_a3xx_atomic(n->opc) ||
          is_bindless_atomic(n->opc))
         ctx->so->has_ssbo = true;

      if (ctx->early_input_release && is_input(n)) {
         last_input_needs_ss |= (n->opc == OPC_LDLV);

         assert(inputs_remaining > 0);
         inputs_remaining--;
         if (inputs_remaining == 0) {
            /* This is the last input. We add the (ei) flag to release
             * varying memory after this executes. If it's an ldlv,
             * however, we need to insert a dummy bary.f on which we can
             * set the (ei) flag. We may also need to insert an (ss) to
             * guarantee that all ldlv's have finished fetching their
             * results before releasing the varying memory.
             */
            struct ir3_instruction *last_input = n;
            if (n->opc == OPC_LDLV) {
               struct ir3_instruction *baryf;

               /* (ss)bary.f (ei)r63.x, 0, r0.x */
               baryf = ir3_build_instr(&build, OPC_BARY_F, 1, 2);
               ir3_dst_create(baryf, regid(63, 0), 0);
               ir3_src_create(baryf, 0, IR3_REG_IMMED)->iim_val = 0;
               ir3_src_create(baryf, regid(0, 0), 0);

               last_input = baryf;
            }

            last_input->dsts[0]->flags |= IR3_REG_EI;
            if (last_input_needs_ss) {
               last_input->flags |= IR3_INSTR_SS;
            }
         }
      }

      ir3_update_legalize_state(state, ctx->compiler, n);
      last_n = n;
   }

   assert(inputs_remaining == 0 || !ctx->early_input_release);

   if (block == ir3_after_preamble(ctx->so->ir) &&
       ctx->has_tex_prefetch && !ctx->has_inputs) {
      /* texture prefetch, but *no* inputs.. we need to insert a
       * dummy bary.f at the top of the shader to unblock varying
       * storage:
       */
      struct ir3_instruction *baryf;

      /* (ss)bary.f (ei)r63.x, 0, r0.x */
      baryf = ir3_build_instr(&build, OPC_BARY_F, 1, 2);
      ir3_dst_create(baryf, regid(63, 0), 0)->flags |= IR3_REG_EI;
      ir3_src_create(baryf, 0, IR3_REG_IMMED)->iim_val = 0;
      ir3_src_create(baryf, regid(0, 0), 0);

      /* insert the dummy bary.f at head: */
      list_delinit(&baryf->node);
      list_add(&baryf->node, &block->instr_list);
   }

   bd->valid = true;

   if (memcmp(&prev_state, state, sizeof(*state))) {
      /* our output state changed, this invalidates all of our
       * successors:
       */
      for (unsigned i = 0; i < ARRAY_SIZE(block->successors); i++) {
         if (!block->successors[i])
            break;
         struct ir3_legalize_block_data *pbd = block->successors[i]->data;
         pbd->valid = false;
      }
   }

   return true;
}

/* Expands dsxpp and dsypp macros to:
 *
 * dsxpp.1 dst, src
 * dsxpp.1.p dst, src
 *
 * We apply this after flags syncing, as we don't want to sync in between the
 * two (which might happen if dst == src).
 */
static bool
apply_fine_deriv_macro(struct ir3_legalize_ctx *ctx, struct ir3_block *block)
{
   struct list_head instr_list;

   /* remove all the instructions from the list, we'll be adding
    * them back in as we go
    */
   list_replace(&block->instr_list, &instr_list);
   list_inithead(&block->instr_list);

   foreach_instr_safe (n, &instr_list) {
      list_addtail(&n->node, &block->instr_list);

      if (n->opc == OPC_DSXPP_MACRO || n->opc == OPC_DSYPP_MACRO) {
         n->opc = (n->opc == OPC_DSXPP_MACRO) ? OPC_DSXPP_1 : OPC_DSYPP_1;

         struct ir3_instruction *op_p = ir3_instr_clone(n);
         op_p->flags = IR3_INSTR_P;

         ctx->so->need_full_quad = true;
      }
   }

   return true;
}

static void
apply_push_consts_load_macro(struct ir3_legalize_ctx *ctx,
                             struct ir3_block *block)
{
   foreach_instr (n, &block->instr_list) {
      if (n->opc == OPC_PUSH_CONSTS_LOAD_MACRO) {
         struct ir3_instruction *stsc =
            ir3_instr_create_at(ir3_after_instr(n), OPC_STSC, 0, 2);
         ir3_src_create(stsc, 0, IR3_REG_IMMED)->iim_val =
            n->push_consts.dst_base;
         ir3_src_create(stsc, 0, IR3_REG_IMMED)->iim_val =
            n->push_consts.src_base;
         stsc->cat6.iim_val = n->push_consts.src_size;
         stsc->cat6.type = TYPE_U32;

         if (ctx->compiler->stsc_duplication_quirk) {
            struct ir3_builder build = ir3_builder_at(ir3_after_instr(stsc));
            struct ir3_instruction *nop = ir3_NOP(&build);
            nop->flags |= IR3_INSTR_SS;
            ir3_instr_move_after(ir3_instr_clone(stsc), nop);
         }

         list_delinit(&n->node);
         break;
      } else if (!is_meta(n)) {
         break;
      }
   }
}

/* NOTE: branch instructions are always the last instruction(s)
 * in the block.  We take advantage of this as we resolve the
 * branches, since "if (foo) break;" constructs turn into
 * something like:
 *
 *   block3 {
 *   	...
 *   	0029:021: mov.s32s32 r62.x, r1.y
 *   	0082:022: br !p0.x, target=block5
 *   	0083:023: br p0.x, target=block4
 *   	// succs: if _[0029:021: mov.s32s32] block4; else block5;
 *   }
 *   block4 {
 *   	0084:024: jump, target=block6
 *   	// succs: block6;
 *   }
 *   block5 {
 *   	0085:025: jump, target=block7
 *   	// succs: block7;
 *   }
 *
 * ie. only instruction in block4/block5 is a jump, so when
 * resolving branches we can easily detect this by checking
 * that the first instruction in the target block is itself
 * a jump, and setup the br directly to the jump's target
 * (and strip back out the now unreached jump)
 *
 * TODO sometimes we end up with things like:
 *
 *    br !p0.x, #2
 *    br p0.x, #12
 *    add.u r0.y, r0.y, 1
 *
 * If we swapped the order of the branches, we could drop one.
 */
static struct ir3_block *
resolve_dest_block(struct ir3_block *block)
{
   /* special case for last block: */
   if (!block->successors[0])
      return block;

   /* NOTE that we may or may not have inserted the jump
    * in the target block yet, so conditions to resolve
    * the dest to the dest block's successor are:
    *
    *   (1) successor[1] == NULL &&
    *   (2) (block-is-empty || only-instr-is-jump)
    */
   if (block->successors[1] == NULL) {
      if (list_is_empty(&block->instr_list)) {
         return block->successors[0];
      } else if (list_length(&block->instr_list) == 1) {
         struct ir3_instruction *instr =
            list_first_entry(&block->instr_list, struct ir3_instruction, node);
         if (instr->opc == OPC_JUMP) {
            /* If this jump is backwards, then we will probably convert
             * the jump being resolved to a backwards jump, which will
             * change a loop-with-continue or loop-with-if into a
             * doubly-nested loop and change the convergence behavior.
             * Disallow this here.
             */
            if (block->successors[0]->index <= block->index)
               return block;
            return block->successors[0];
         }
      }
   }
   return block;
}

static void
remove_unused_block(struct ir3_block *old_target)
{
   list_delinit(&old_target->node);

   /* cleanup dangling predecessors: */
   for (unsigned i = 0; i < ARRAY_SIZE(old_target->successors); i++) {
      if (old_target->successors[i]) {
         struct ir3_block *succ = old_target->successors[i];
         ir3_block_remove_predecessor(succ, old_target);
      }
   }
}

static bool
retarget_jump(struct ir3_instruction *instr, struct ir3_block *new_target)
{
   struct ir3_block *old_target = instr->cat0.target;
   struct ir3_block *cur_block = instr->block;

   /* update current blocks successors to reflect the retargetting: */
   if (cur_block->successors[0] == old_target) {
      cur_block->successors[0] = new_target;
   } else {
      assert(cur_block->successors[1] == old_target);
      cur_block->successors[1] = new_target;
   }

   /* update new target's predecessors: */
   ir3_block_add_predecessor(new_target, cur_block);

   /* and remove old_target's predecessor: */
   ir3_block_remove_predecessor(old_target, cur_block);

   instr->cat0.target = new_target;

   if (old_target->predecessors_count == 0) {
      remove_unused_block(old_target);
      return true;
   }

   return false;
}

static bool
is_invertible_branch(struct ir3_instruction *instr)
{
   switch (instr->opc) {
   case OPC_BR:
   case OPC_BRAA:
   case OPC_BRAO:
   case OPC_BANY:
   case OPC_BALL:
      return true;
   default:
      return false;
   }
}

static bool
opt_jump(struct ir3 *ir)
{
   bool progress = false;

   unsigned index = 0;
   foreach_block (block, &ir->block_list)
      block->index = index++;

   foreach_block (block, &ir->block_list) {
      /* This pass destroys the physical CFG so don't keep it around to avoid
       * validation errors.
       */
      block->physical_successors_count = 0;
      block->physical_predecessors_count = 0;

      foreach_instr (instr, &block->instr_list) {
         if (!is_flow(instr) || !instr->cat0.target)
            continue;

         struct ir3_block *tblock = resolve_dest_block(instr->cat0.target);
         if (tblock != instr->cat0.target) {
            progress = true;

            /* Exit early if we deleted a block to avoid iterator
             * weirdness/assert fails
             */
            if (retarget_jump(instr, tblock))
               return true;
         }
      }

      /* Detect the case where the block ends either with:
       * - A single unconditional jump to the next block.
       * - Two jump instructions with opposite conditions, and one of the
       *   them jumps to the next block.
       * We can remove the one that jumps to the next block in either case.
       */
      if (list_is_empty(&block->instr_list))
         continue;

      struct ir3_instruction *jumps[2] = {NULL, NULL};
      jumps[0] =
         list_last_entry(&block->instr_list, struct ir3_instruction, node);
      if (!list_is_singular(&block->instr_list))
         jumps[1] =
            list_last_entry(&jumps[0]->node, struct ir3_instruction, node);

      if (jumps[0]->opc == OPC_JUMP)
         jumps[1] = NULL;
      else if (!is_invertible_branch(jumps[0]) || !jumps[1] ||
               !is_invertible_branch(jumps[1])) {
         continue;
      }

      for (unsigned i = 0; i < 2; i++) {
         if (!jumps[i])
            continue;
         struct ir3_block *tblock = jumps[i]->cat0.target;
         if (&tblock->node == block->node.next) {
            list_delinit(&jumps[i]->node);
            progress = true;
            break;
         }
      }
   }

   return progress;
}

static void
resolve_jumps(struct ir3 *ir)
{
   foreach_block (block, &ir->block_list)
      foreach_instr (instr, &block->instr_list)
         if (is_flow(instr) && instr->cat0.target) {
            struct ir3_instruction *target = list_first_entry(
               &instr->cat0.target->instr_list, struct ir3_instruction, node);

            instr->cat0.immed = (int)target->ip - (int)instr->ip;
         }
}

static void
mark_jp(struct ir3_block *block)
{
   /* We only call this on the end block (in kill_sched) or after retargeting
    * all jumps to empty blocks (in mark_xvergence_points) so there's no need to
    * worry about empty blocks.
    */
   assert(!list_is_empty(&block->instr_list));

   struct ir3_instruction *target =
      list_first_entry(&block->instr_list, struct ir3_instruction, node);

   /* Add nop instruction for (jp) flag since it has no effect on a5xx when set
    * on the end instruction.
    */
   if (target->opc == OPC_END && block->shader->compiler->gen == 5) {
      struct ir3_builder build = ir3_builder_at(ir3_before_instr(target));
      target = ir3_NOP(&build);
   }

   target->flags |= IR3_INSTR_JP;
}

/* Mark points where control flow reconverges.
 *
 * Re-convergence points are where "parked" threads are reconverged with threads
 * that took the opposite path last time around. We already calculated them, we
 * just need to mark them with (jp).
 */
static void
mark_xvergence_points(struct ir3 *ir)
{
   foreach_block (block, &ir->block_list) {
      if (block->reconvergence_point)
         mark_jp(block);
   }
}

static void
invert_branch(struct ir3_instruction *branch)
{
   switch (branch->opc) {
   case OPC_BR:
      break;
   case OPC_BALL:
      branch->opc = OPC_BANY;
      break;
   case OPC_BANY:
      branch->opc = OPC_BALL;
      break;
   case OPC_BRAA:
      branch->opc = OPC_BRAO;
      break;
   case OPC_BRAO:
      branch->opc = OPC_BRAA;
      break;
   default:
      UNREACHABLE("can't get here");
   }

   branch->cat0.inv1 = !branch->cat0.inv1;
   branch->cat0.inv2 = !branch->cat0.inv2;
   branch->cat0.target = branch->block->successors[1];
}

/* Insert the branch/jump instructions for flow control between blocks.
 * Initially this is done naively, without considering if the successor
 * block immediately follows the current block (ie. so no jump required),
 * but that is cleaned up in opt_jump().
 */
static void
block_sched(struct ir3 *ir)
{
   foreach_block (block, &ir->block_list) {
      struct ir3_instruction *terminator = ir3_block_get_terminator(block);

      if (block->successors[1]) {
         /* if/else, conditional branches to "then" or "else": */
         struct ir3_instruction *br1, *br2;

         assert(terminator);
         unsigned opc = terminator->opc;

         if (opc == OPC_GETONE || opc == OPC_SHPS || opc == OPC_GETLAST) {
            /* getone/shps can't be inverted, and it wouldn't even make sense
             * to follow it with an inverted branch, so follow it by an
             * unconditional branch.
             */
            assert(terminator->srcs_count == 0);
            br1 = terminator;
            br1->cat0.target = block->successors[1];

            struct ir3_builder build = ir3_builder_at(ir3_after_block(block));
            br2 = ir3_JUMP(&build);
            br2->cat0.target = block->successors[0];
         } else if (opc == OPC_BR || opc == OPC_BRAA || opc == OPC_BRAO ||
                    opc == OPC_BALL || opc == OPC_BANY) {
            /* create "else" branch first (since "then" block should
             * frequently/always end up being a fall-thru):
             */
            br1 = terminator;
            br2 = ir3_instr_clone(br1);
            invert_branch(br1);
            br2->cat0.target = block->successors[0];
         } else {
            assert(opc == OPC_PREDT || opc == OPC_PREDF);

            /* Handled by prede_sched. */
            terminator->cat0.target = block->successors[0];
            continue;
         }

         /* Creating br2 caused it to be moved before the terminator b1, move it
          * back.
          */
         ir3_instr_move_after(br2, br1);
      } else if (block->successors[0]) {
         /* otherwise unconditional jump or predt/predf to next block which
          * should already have been inserted.
          */
         assert(terminator);
         assert(terminator->opc == OPC_JUMP || terminator->opc == OPC_PREDT ||
                terminator->opc == OPC_PREDF || terminator->opc == OPC_SHPE);

         /* shpe is not a branch (we just treat it as a terminator to make sure
          * it stays at the end of its block) so add one now.
          */
         if (terminator->opc == OPC_SHPE) {
            struct ir3_builder b = ir3_builder_at(ir3_after_instr(terminator));
            terminator = ir3_JUMP(&b);
         }

         terminator->cat0.target = block->successors[0];
      }
   }
}

static void
add_nop_before_block(struct ir3_block *block, unsigned repeat)
{
   struct ir3_instruction *nop = ir3_block_get_first_instr(block);

   if (!nop || nop->opc != OPC_NOP) {
      struct ir3_builder build = ir3_builder_at(ir3_before_block(block));
      nop = ir3_NOP(&build);
   }

   nop->repeat = MAX2(nop->repeat, repeat);
}

/* Some gens have a hardware issue that needs to be worked around by 1)
 * inserting 4 nops after the second pred[tf] of a pred[tf]/pred[ft] pair and/or
 * inserting 6 nops after prede.
 *
 * This function should be called with the second pred[tf] of such a pair and
 * NULL if there is only one pred[tf].
 */
static void
add_predication_workaround(struct ir3_compiler *compiler,
                           struct ir3_instruction *predtf,
                           struct ir3_instruction *prede)
{
   if (predtf && compiler->predtf_nop_quirk) {
      add_nop_before_block(predtf->block->predecessors[0]->successors[1], 4);
   }

   if (compiler->prede_nop_quirk) {
      add_nop_before_block(prede->block->successors[0], 6);
   }
}

static void
prede_sched(struct ir3 *ir)
{
   unsigned index = 0;
   foreach_block (block, &ir->block_list)
      block->index = index++;

   foreach_block (block, &ir->block_list) {
      /* Look for the following pattern generated by NIR lowering. The numbers
       * at the top of blocks are their index.
       *        |--- i ----|
       *        |   ...    |
       *        | pred[tf] |
       *        |----------|
       *      succ0 /   \ succ1
       * |-- i+1 ---| |-- i+2 ---|
       * |    ...   | |   ...    |
       * | pred[ft] | |   ...    |
       * |----------| |----------|
       *     succ0 \   / succ0
       *        |--- j ----|
       *        |   ...    |
       *        |----------|
       */
      struct ir3_block *succ0 = block->successors[0];
      struct ir3_block *succ1 = block->successors[1];

      if (!succ1)
         continue;

      struct ir3_instruction *terminator = ir3_block_get_terminator(block);
      if (!terminator)
         continue;
      if (terminator->opc != OPC_PREDT && terminator->opc != OPC_PREDF)
         continue;

      assert(!succ0->successors[1] && !succ1->successors[1]);
      assert(succ0->successors[0] == succ1->successors[0]);
      assert(succ0->predecessors_count == 1 && succ1->predecessors_count == 1);
      assert(succ0->index == (block->index + 1));
      assert(succ1->index == (block->index + 2));

      struct ir3_instruction *succ0_terminator =
         ir3_block_get_terminator(succ0);
      assert(succ0_terminator);
      assert(succ0_terminator->opc ==
             (terminator->opc == OPC_PREDT ? OPC_PREDF : OPC_PREDT));

      ASSERTED struct ir3_instruction *succ1_terminator =
         ir3_block_get_terminator(succ1);
      assert(!succ1_terminator || (succ1_terminator->opc == OPC_JUMP));

      /* Simple case: both successors contain instructions. Keep both blocks and
       * insert prede before the second successor's terminator:
       *        |--- i ----|
       *        |   ...    |
       *        | pred[tf] |
       *        |----------|
       *      succ0 /   \ succ1
       * |-- i+1 ---| |-- i+2 ---|
       * |    ...   | |   ...    |
       * | pred[ft] | | prede    |
       * |----------| |----------|
       *     succ0 \   / succ0
       *        |--- j ----|
       *        |   ...    |
       *        |----------|
       */
      if (!list_is_empty(&succ1->instr_list)) {
         struct ir3_builder build =
            ir3_builder_at(ir3_before_terminator(succ1));
         struct ir3_instruction *prede = ir3_PREDE(&build);
         add_predication_workaround(ir->compiler, succ0_terminator, prede);
         continue;
      }

      /* Second successor is empty so we can remove it:
       *        |--- i ----|
       *        |   ...    |
       *        | pred[tf] |
       *        |----------|
       *      succ0 /   \ succ1
       * |-- i+1 ---|   |
       * |    ...   |   |
       * |   prede  |   |
       * |----------|   |
       *     succ0 \    /
       *        |--- j ----|
       *        |   ...    |
       *        |----------|
       */
      list_delinit(&succ0_terminator->node);
      struct ir3_builder build = ir3_builder_at(ir3_before_terminator(succ0));
      struct ir3_instruction *prede = ir3_PREDE(&build);
      add_predication_workaround(ir->compiler, NULL, prede);
      remove_unused_block(succ1);
      block->successors[1] = succ0->successors[0];
      ir3_block_add_predecessor(succ0->successors[0], block);
   }
}

/* Here we workaround the fact that kill doesn't actually kill the thread as
 * GL expects. The last instruction always needs to be an end instruction,
 * which means that if we're stuck in a loop where kill is the only way out,
 * then we may have to jump out to the end. kill may also have the d3d
 * semantics of converting the thread to a helper thread, rather than setting
 * the exec mask to 0, in which case the helper thread could get stuck in an
 * infinite loop.
 *
 * We do this late, both to give the scheduler the opportunity to reschedule
 * kill instructions earlier and to avoid having to create a separate basic
 * block.
 *
 * TODO: Assuming that the wavefront doesn't stop as soon as all threads are
 * killed, we might benefit by doing this more aggressively when the remaining
 * part of the program after the kill is large, since that would let us
 * skip over the instructions when there are no non-killed threads left.
 */
static void
kill_sched(struct ir3 *ir, struct ir3_shader_variant *so)
{
   ir3_count_instructions(ir);

   /* True if we know that this block will always eventually lead to the end
    * block:
    */
   bool always_ends = true;
   bool added = false;
   struct ir3_block *last_block =
      list_last_entry(&ir->block_list, struct ir3_block, node);

   foreach_block_rev (block, &ir->block_list) {
      for (unsigned i = 0; i < 2 && block->successors[i]; i++) {
         if (block->successors[i]->start_ip < block->end_ip)
            always_ends = false;
      }

      if (always_ends)
         continue;

      foreach_instr_safe (instr, &block->instr_list) {
         if (instr->opc != OPC_KILL)
            continue;

         struct ir3_instruction *br =
            ir3_instr_create_at(ir3_after_instr(instr), OPC_BR, 0, 1);
         ir3_src_create(br, instr->srcs[0]->num, instr->srcs[0]->flags)->wrmask =
            1;
         br->cat0.target =
            list_last_entry(&ir->block_list, struct ir3_block, node);

         added = true;
      }
   }

   if (added) {
      /* I'm not entirely sure how the branchstack works, but we probably
       * need to add at least one entry for the divergence which is resolved
       * at the end:
       */
      so->branchstack++;

      /* We don't update predecessors/successors, so we have to do this
       * manually:
       */
      mark_jp(last_block);
   }
}

static void
dbg_sync_sched(struct ir3 *ir, struct ir3_shader_variant *so)
{
   foreach_block (block, &ir->block_list) {
      foreach_instr_safe (instr, &block->instr_list) {
         if (is_ss_producer(instr) || is_sy_producer(instr)) {
            struct ir3_builder build = ir3_builder_at(ir3_after_instr(instr));
            struct ir3_instruction *nop = ir3_NOP(&build);
            nop->flags |= IR3_INSTR_SS | IR3_INSTR_SY;
         }
      }
   }
}

static void
dbg_nop_sched(struct ir3 *ir, struct ir3_shader_variant *so)
{
   foreach_block (block, &ir->block_list) {
      foreach_instr_safe (instr, &block->instr_list) {
         struct ir3_builder build = ir3_builder_at(ir3_before_instr(instr));
         struct ir3_instruction *nop = ir3_NOP(&build);
         nop->repeat = 5;
      }
   }
}

static void
dbg_expand_rpt(struct ir3 *ir)
{
   foreach_block (block, &ir->block_list) {
      foreach_instr_safe (instr, &block->instr_list) {
         if (instr->repeat == 0 || instr->opc == OPC_NOP ||
             instr->opc == OPC_SWZ || instr->opc == OPC_GAT ||
             instr->opc == OPC_SCT) {
            continue;
         }

         for (unsigned i = 0; i <= instr->repeat; ++i) {
            struct ir3_instruction *rpt = ir3_instr_clone(instr);
            ir3_instr_move_before(rpt, instr);
            rpt->repeat = 0;

            foreach_dst (dst, rpt) {
               dst->num += i;
               dst->wrmask = 1;
            }

            foreach_src (src, rpt) {
               if (!(src->flags & IR3_REG_R))
                  continue;

               src->num += i;
               src->uim_val += i;
               src->wrmask = 1;
               src->flags &= ~IR3_REG_R;
            }
         }

         list_delinit(&instr->node);
      }
   }
}

struct ir3_helper_block_data {
   /* Whether helper invocations may be used on any path starting at the
    * beginning of the block.
    */
   bool uses_helpers_beginning;

   /* Whether helper invocations may be used by the end of the block. Branch
    * instructions are considered to be "between" blocks, because (eq) has to be
    * inserted after them in the successor blocks, so branch instructions using
    * helpers will result in uses_helpers_end = true for their block.
    */
   bool uses_helpers_end;
};

/* Insert (eq) after the last instruction using the results of helper
 * invocations. Use a backwards dataflow analysis to determine at which points
 * in the program helper invocations are definitely never used, and then insert
 * (eq) at the point where we cross from a point where they may be used to a
 * point where they are never used.
 */
static void
helper_sched(struct ir3_legalize_ctx *ctx, struct ir3 *ir,
             struct ir3_shader_variant *so)
{
   bool non_prefetch_helpers = false;

   foreach_block (block, &ir->block_list) {
      struct ir3_helper_block_data *bd =
         rzalloc(ctx, struct ir3_helper_block_data);
      foreach_instr (instr, &block->instr_list) {
         if (uses_helpers(instr)) {
            bd->uses_helpers_beginning = true;
            if (instr->opc != OPC_META_TEX_PREFETCH) {
               non_prefetch_helpers = true;
            }
         }

         if (instr->opc == OPC_SHPE) {
            /* (eq) is not allowed in preambles, mark the whole preamble as
             * requiring helpers to avoid putting it there.
             */
            bd->uses_helpers_beginning = true;
            bd->uses_helpers_end = true;
         }
      }

      struct ir3_instruction *terminator = ir3_block_get_terminator(block);
      if (terminator) {
         if (terminator->opc == OPC_BALL || terminator->opc == OPC_BANY ||
             (terminator->opc == OPC_GETONE &&
              (terminator->flags & IR3_INSTR_NEEDS_HELPERS))) {
            bd->uses_helpers_beginning = true;
            bd->uses_helpers_end = true;
            non_prefetch_helpers = true;
         }
      }

      block->data = bd;
   }

   /* If only prefetches use helpers then we can disable them in the shader via
    * a register setting.
    */
   if (!non_prefetch_helpers) {
      so->prefetch_end_of_quad = true;
      return;
   }

   bool progress;
   do {
      progress = false;
      foreach_block_rev (block, &ir->block_list) {
         struct ir3_helper_block_data *bd = block->data;

         if (!bd->uses_helpers_beginning)
            continue;

         for (unsigned i = 0; i < block->physical_predecessors_count; i++) {
            struct ir3_block *pred = block->physical_predecessors[i];
            struct ir3_helper_block_data *pred_bd = pred->data;
            if (!pred_bd->uses_helpers_end) {
               pred_bd->uses_helpers_end = true;
            }
            if (!pred_bd->uses_helpers_beginning) {
               pred_bd->uses_helpers_beginning = true;
               progress = true;
            }
         }
      }
   } while (progress);

   /* Now, we need to determine the points where helper invocations become
    * unused.
    */
   foreach_block (block, &ir->block_list) {
      struct ir3_helper_block_data *bd = block->data;
      if (bd->uses_helpers_end)
         continue;

      /* We need to check the predecessors because of situations with critical
       * edges like this that can occur after optimizing jumps:
       *
       *    br p0.x, #endif
       *    ...
       *    sam ...
       *    ...
       *    endif:
       *    ...
       *    end
       *
       * The endif block will have uses_helpers_beginning = false and
       * uses_helpers_end = false, but because we jump to there from the
       * beginning of the if where uses_helpers_end = true, we still want to
       * add an (eq) at the beginning of the block:
       *
       *    br p0.x, #endif
       *    ...
       *    sam ...
       *    (eq)nop
       *    ...
       *    endif:
       *    (eq)nop
       *    ...
       *    end
       *
       * This an extra nop in the case where the branch isn't taken, but that's
       * probably preferable to adding an extra jump instruction which is what
       * would happen if we ran this pass before optimizing jumps:
       *
       *    br p0.x, #else
       *    ...
       *    sam ...
       *    (eq)nop
       *    ...
       *    jump #endif
       *    else:
       *    (eq)nop
       *    endif:
       *    ...
       *    end
       *
       * We also need this to make sure we insert (eq) after branches which use
       * helper invocations.
       */
      bool pred_uses_helpers = bd->uses_helpers_beginning;
      for (unsigned i = 0; i < block->physical_predecessors_count; i++) {
         struct ir3_block *pred = block->physical_predecessors[i];
         struct ir3_helper_block_data *pred_bd = pred->data;
         if (pred_bd->uses_helpers_end) {
            pred_uses_helpers = true;
            break;
         }
      }

      if (!pred_uses_helpers)
         continue;

      /* The last use of helpers is somewhere between the beginning and the
       * end. first_instr will be the first instruction where helpers are no
       * longer required, or NULL if helpers are not required just at the end.
       */
      struct ir3_instruction *first_instr = NULL;
      foreach_instr_rev (instr, &block->instr_list) {
         /* Skip prefetches because they actually execute before the block
          * starts and at this stage they aren't guaranteed to be at the start
          * of the block.
          */
         if (uses_helpers(instr) && instr->opc != OPC_META_TEX_PREFETCH)
            break;
         first_instr = instr;
      }

      bool killed = false;
      bool expensive_instruction_in_block = false;
      if (first_instr) {
         foreach_instr_from (instr, first_instr, &block->instr_list) {
            /* If there's already a nop, we don't have to worry about whether to
             * insert one.
             */
            if (instr->opc == OPC_NOP) {
               instr->flags |= IR3_INSTR_EQ;
               killed = true;
               break;
            }

            /* ALU and SFU instructions probably aren't going to benefit much
             * from killing helper invocations, because they complete at least
             * an entire quad in a cycle and don't access any quad-divergent
             * memory, so delay emitting (eq) in the hopes that we find a nop
             * afterwards.
             */
            if (is_alu(instr) || is_sfu(instr))
               continue;
            if (instr->opc == OPC_PREDE)
               continue;

            expensive_instruction_in_block = true;
            break;
         }
      }

      /* If this block isn't the last block before the end instruction, assume
       * that there may be expensive instructions in later blocks so it's worth
       * it to insert a nop.
       */
      if (!killed && (expensive_instruction_in_block ||
                      block->successors[0] != ir3_end_block(ir))) {
         struct ir3_cursor cursor = first_instr ? ir3_before_instr(first_instr)
                                                : ir3_before_terminator(block);
         struct ir3_builder build = ir3_builder_at(cursor);
         struct ir3_instruction *nop = ir3_NOP(&build);
         nop->flags |= IR3_INSTR_EQ;
      }
   }
}

struct ir3_last_block_data {
   /* Whether a read will be done on a register at a later point, it is
    * considered safe to set (last) when this is false for a particular
    * register.
    */
   regmask_t will_read;
};

/* Use a backwards dataflow analysis to determine when a certain register is
 * always written to prior to being read in a similar manner to SSA liveness
 * which we can use to determine when we can insert (last) on src regs.
 */
static void
track_last(struct ir3_legalize_ctx *ctx, struct ir3 *ir,
           struct ir3_shader_variant *so)
{
   foreach_block (block, &ir->block_list) {
      struct ir3_last_block_data *bd = rzalloc(ctx, struct ir3_last_block_data);

      regmask_init(&bd->will_read, so->mergedregs);

      block->data = bd;
   }

   bool progress;
   do {
      progress = false;
      regmask_t will_read;
      regmask_init(&will_read, so->mergedregs);

      foreach_block_rev (block, &ir->block_list) {
         struct ir3_last_block_data *bd = block->data;

         for (unsigned i = 0; i < ARRAY_SIZE(block->successors); i++) {
            struct ir3_block *succ = block->successors[i];
            if (!succ)
               continue;

            struct ir3_last_block_data *succ_bd = succ->data;
            regmask_or(&will_read, &will_read, &succ_bd->will_read);
         }

         foreach_instr_rev (instr, &block->instr_list) {
            for (unsigned i = 0; i < instr->dsts_count; i++) {
               struct ir3_register *dst = instr->dsts[i];
               if (dst->flags & (IR3_REG_IMMED | IR3_REG_CONST |
                                 IR3_REG_SHARED | IR3_REG_RT)) {
                  continue;
               }

               regmask_clear(&will_read, dst);
            }

            for (unsigned i = 0; i < instr->srcs_count; i++) {
               struct ir3_register *src = instr->srcs[i];
               if (src->flags &
                   (IR3_REG_IMMED | IR3_REG_CONST | IR3_REG_SHARED)) {
                  continue;
               }

               if (!regmask_get(&will_read, src)) {
                  if (!(src->flags & IR3_REG_LAST_USE)) {
                     progress = true;
                     src->flags |= IR3_REG_LAST_USE;
                  }
               } else if (src->flags & IR3_REG_LAST_USE) {
                  progress = true;
                  src->flags &= ~IR3_REG_LAST_USE;
               }

               /* Setting will_read immediately ensures that only the first src
                * of multiple identical srcs will have (last) set. This matches
                * the blob's behavior.
                */
               regmask_set(&will_read, src);
            }
         }

         bd->will_read = will_read;
      }
   } while (progress);
}

bool
ir3_legalize(struct ir3 *ir, struct ir3_shader_variant *so, int *max_bary)
{
   struct ir3_legalize_ctx *ctx = rzalloc(ir, struct ir3_legalize_ctx);
   bool progress;

   ctx->so = so;
   ctx->max_bary = -1;
   ctx->compiler = ir->compiler;
   ctx->type = ir->type;

   /* allocate per-block data: */
   foreach_block (block, &ir->block_list) {
      struct ir3_legalize_block_data *bd =
         rzalloc(ctx, struct ir3_legalize_block_data);

      ir3_init_legalize_state(&bd->state, so->compiler);
      ir3_init_legalize_state(&bd->begin_state, so->compiler);

      block->data = bd;
   }

   /* We may have failed to pull all input loads into the first block.
    * In such case at the moment we aren't able to find a better place
    * to for (ei) than the end of the program.
    * a5xx and a6xx do automatically release varying storage at the end.
    */
   ctx->early_input_release = true;

   struct ir3_block *start_block = ir3_after_preamble(ir);

   /* Gather information to determine whether we can enable early preamble.
    */
   bool gpr_in_preamble = false;
   bool pred_in_preamble = false;
   bool relative_in_preamble = false;
   bool in_preamble = start_block != ir3_start_block(ir);
   bool has_preamble = start_block != ir3_start_block(ir);

   foreach_block (block, &ir->block_list) {
      if (block == start_block)
         in_preamble = false;

      foreach_instr (instr, &block->instr_list) {
         if (is_input(instr)) {
            ctx->has_inputs = true;
            if (block != start_block) {
               ctx->early_input_release = false;
            }
         }

         if (is_meta(instr))
            continue;

         foreach_src (reg, instr) {
            if (in_preamble) {
               if (!(reg->flags & IR3_REG_SHARED) && is_reg_gpr(reg))
                  gpr_in_preamble = true;
               if (reg->flags & IR3_REG_RELATIV)
                  relative_in_preamble = true;
            }
         }

         foreach_dst (reg, instr) {
            if (is_dest_gpr(reg)) {
               if (in_preamble) {
                  if (!(reg->flags & IR3_REG_SHARED))
                     gpr_in_preamble = true;
                  if (reg->flags & IR3_REG_RELATIV)
                     relative_in_preamble = true;
               }
            }
         }

         if (in_preamble && writes_pred(instr)) {
            pred_in_preamble = true;
         }
      }
   }

   so->early_preamble = has_preamble && !gpr_in_preamble &&
      !pred_in_preamble && !relative_in_preamble &&
      ir->compiler->has_early_preamble &&
      !(ir3_shader_debug & IR3_DBG_NOEARLYPREAMBLE);

   /* On a7xx, sync behavior for a1.x is different in the early preamble. RaW
    * dependencies must be synchronized with (ss) there must be an extra
    * (r) on the source of the mova1 instruction.
    */
   if (so->early_preamble && ir->compiler->gen >= 7) {
      foreach_block (block, &ir->block_list) {
         if (block == start_block)
            break;
         block->in_early_preamble = true;
      }
   }

   assert(ctx->early_input_release || ctx->compiler->gen >= 5);

   if (ir3_shader_debug & IR3_DBG_EXPANDRPT) {
      dbg_expand_rpt(ir);
   }

   /* process each block: */
   do {
      progress = false;
      foreach_block (block, &ir->block_list) {
         progress |= legalize_block(ctx, block);
      }
   } while (progress);

   *max_bary = ctx->max_bary;

   foreach_block (block, &ir->block_list) {
      struct ir3_instruction *terminator = ir3_block_get_terminator(block);
      if (terminator && terminator->opc == OPC_GETONE) {
         apply_push_consts_load_macro(ctx, block->successors[0]);
         break;
      }
   }

   block_sched(ir);

   foreach_block (block, &ir->block_list) {
      progress |= apply_fine_deriv_macro(ctx, block);
   }

   if (ir3_shader_debug & IR3_DBG_FULLSYNC) {
      dbg_sync_sched(ir, so);
   }

   if (ir3_shader_debug & IR3_DBG_FULLNOP) {
      dbg_nop_sched(ir, so);
   }

   bool cfg_changed = false;
   while (opt_jump(ir))
      cfg_changed = true;

   prede_sched(ir);

   if (cfg_changed)
      ir3_calc_reconvergence(so);

   if (so->type == MESA_SHADER_FRAGMENT)
      kill_sched(ir, so);

   /* TODO: does (eq) exist before a6xx? */
   if (so->type == MESA_SHADER_FRAGMENT && so->need_pixlod &&
       so->compiler->gen >= 6)
      helper_sched(ctx, ir, so);

   /* Note: insert (last) before alias.tex to have the sources that are actually
    * read by instructions (as opposed to alias registers) more easily
    * available.
    */
   if (so->compiler->gen >= 7)
      track_last(ctx, ir, so);

   ir3_insert_alias_tex(ir);
   ir3_count_instructions(ir);
   resolve_jumps(ir);

   mark_xvergence_points(ir);

   ralloc_free(ctx);

   return true;
}
