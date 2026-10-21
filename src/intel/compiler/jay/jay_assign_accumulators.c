/*
 * Copyright 2026 Intel Corporation
 * Copyright 2025 Valve Corporation
 * Copyright 2019-2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/bitset.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "util/u_worklist.h"
#include "jay_ir.h"
#include "jay_opcodes.h"
#include "jay_private.h"

static bool
tracked(jay_def x)
{
   return x.file == GPR || x.file == ACCUM;
}

/*
 * Track individual 16-bit GPRs but whole 32-bit accumulators. This is
 * convenient but should probably be unified at some point.
 *
 * We do not track UGPRs in this pass, as uniform accumulators introduce
 * considerable complexity for marginal benefit.
 */
static unsigned
liveness_key(jay_def x)
{
   return x.file == GPR ? (x.reg + JAY_MAX_ACCUMS) : (x.reg / 2);
}

static unsigned
liveness_width(jay_def x)
{
   return x.file == GPR ? jay_num_regs(x) : jay_num_values(x);
}

static void
postra_liveness_ins(BITSET_WORD *live, jay_inst *I)
{
   if (tracked(I->dst) && !I->predication) {
      BITSET_CLEAR_COUNT(live, liveness_key(I->dst), liveness_width(I->dst));
   }

   jay_foreach_src(I, s) {
      if (tracked(I->src[s])) {
         BITSET_SET_COUNT(live, liveness_key(I->src[s]),
                          liveness_width(I->src[s]));
      }
   }
}

/*
 * Globally, liveness analysis uses a fixed-point algorithm based on a
 * worklist. We initialize a work list with the exit block. We iterate the work
 * list to compute live_in from live_out for each block on the work list,
 * adding the predecessors of the block to the work list if we made progress.
 */
static void
postra_liveness(jay_function *func)
{
   u_worklist worklist;
   u_worklist_init(&worklist, func->num_blocks, NULL);

   jay_foreach_block(func, block) {
      BITSET_ZERO(block->postra_gpr_live_in);
      BITSET_ZERO(block->postra_gpr_live_out);

      jay_worklist_push_tail(&worklist, block);
   }

   while (!u_worklist_is_empty(&worklist)) {
      /* Pop off in reverse order since liveness is backwards */
      jay_block *blk = jay_worklist_pop_tail(&worklist);

      /* Calculate liveness locally */
      jay_foreach_successor(blk, succ, GPR) {
         BITSET_OR(blk->postra_gpr_live_out, blk->postra_gpr_live_out,
                   (*succ)->postra_gpr_live_in);
      }

      BITSET_DECLARE(live, JAY_NR_POSTRA_LIVE);
      memcpy(live, blk->postra_gpr_live_out, sizeof(live));

      jay_foreach_inst_in_block_rev(blk, ins) {
         postra_liveness_ins(live, ins);
      }

      /* If we made progress, we need to reprocess the predecessors */
      if (!BITSET_EQUAL(blk->postra_gpr_live_in, live)) {
         memcpy(blk->postra_gpr_live_in, live, sizeof(live));

         jay_foreach_predecessor(blk, pred, GPR) {
            jay_worklist_push_head(&worklist, *pred);
         }
      }
   }

   u_worklist_fini(&worklist);
}

/*
 * Check if a source is killed by the instruction. If the register is dead after
 * this instuction, it's the last use / killed.  That also includes if the
 * register is overwritten this cycle, but that won't show up in the liveness.
 */
static bool
source_killed(BITSET_WORD *live, const jay_inst *I, unsigned s)
{
   return !BITSET_TEST(live, liveness_key(I->src[s])) ||
          (I->dst.file == GPR &&
           I->src[s].reg >= I->dst.reg &&
           (I->src[s].reg - I->dst.reg) < jay_num_regs(I->dst));
}

/* We assign accumulators with a simple heuristic: promote registers with the
 * shortest live range. This is pretty naive but it is well-motivated:
 *
 * 1. Short live ranges reduce interfere with other potentially promotable
 *    registers, allowing for more overall accumulator usage. This is a builtin
 *    defense against being too greedy.
 *
 * 2. Short live ranges necessarily have the first read of the register shortly
 *    after the write. That situation benefits greatly from promoting to an
 *    accumulator as such sequences are GRF latency bound.
 *
 * There are lots of ways to do better in the future, but this is good for now.
 */
struct candidate {
   uint32_t def_ip, last_use_ip;
   bool mac_candidate;
};

static int
score(struct candidate c)
{
   assert(c.def_ip < c.last_use_ip);
   return (int) (c.last_use_ip - c.def_ip);
}

static int
cmp_candidates(const void *left_, const void *right_)
{
   const struct candidate *left = left_;
   const struct candidate *right = right_;
   int l = score(*left), r = score(*right);

   return (l > r) - (l < r);
}

/*
 * Query whether an instruction can access accumulators. Comments are quoted
 * from bspec 56619 as the rules are complex.
 */
static inline bool
can_access_accum(jay_shader *shader, jay_inst *I, signed src)
{
   /* "No Accumulator usage for Control Flow, Math, Send, DPAS instructions." */
   if (jay_op_is_control_flow(I->op) ||
       I->op == JAY_OPCODE_MATH ||
       I->op == JAY_OPCODE_SEND ||
       I->op == JAY_OPCODE_DPAS ||
       I->op == JAY_OPCODE_SLICE_REPACK) {
      return false;
   }

   if (shader->devinfo->verx10 < 125) {
      /* TGL PRM Vol. 9, p. 371 limits explicit ACC sources to src0, although
       * p. 493 appears to allow src1 for MUL/MAC. IGC restricts two-source
       * instructions to src0 before XeHP; RPL MUL/ADD src1 tests also fail.
       * For three-source instructions, p. 493 rule 5 forbids ACC in src2.
       */
      unsigned nr_srcs = jay_num_isa_srcs(I);
      if ((nr_srcs == 2 || nr_srcs == 3) && src == nr_srcs - 1)
         return false;
   }

   /* TODO: Many, many more restrictions on non-f32 */
   if (!(jay_operand_type(I, src) == JAY_TYPE_F32 && I->type == JAY_TYPE_F32)) {
      return false;
   }

   /* "When destination is accumulator with offset 0, destination horizontal
    * stride must be 1."
    */
   if (I->dst.file == GPR && jay_def_stride(shader, I->dst) != JAY_STRIDE_4) {
      return false;
   }

   /* "Register Regioning patterns where register data bit locations are changed
    * between source and destination are not supported when an accumulator is
    * used as an implicit source or an explicit source in an instruction.."
    */
   jay_foreach_src(I, s) {
      if (I->src[s].file == GPR &&
          jay_def_stride(shader, I->src[s]) != JAY_STRIDE_4) {
         return false;
      }
   }

   /* Jay's predication requires tying the destination to the source which is
    * too complicated to model here. It's also only dubiously useful.
    */
   if (src < 0 && I->predication) {
      return false;
   }

   /* This copies only part of a GRF so can't be accumulatored */
   if (I->op == JAY_OPCODE_DESWIZZLE_EVEN) {
      return false;
   }

   return true;
}

static inline bool
could_be_mac(const jay_shader *s, const jay_inst *I)
{
   /* The bspec says "Instructions that specify an implicit accumulator
    * source cannot specify an explicit accumulator source operand.". But
    * it works fine on Lunar Lake so ¯\_(ツ)_/¯ ... gate on !strict.
    *
    * Older platforms seem to actually need that restriction.
    */
   return (I->op == JAY_OPCODE_MAD && I->type == JAY_TYPE_F32) &&
          !jay_simd_split(s, I) &&
          !(I->src[0].negate || I->src[0].abs) &&
          !(jay_debug & JAY_DBG_STRICT) &&
          (s->devinfo->verx10 >= 125 ||
           (I->src[1].file != ACCUM && I->src[2].file != ACCUM));
}

static inline void
substitute_acc(jay_def *x, unsigned acc_p1)
{
   if (acc_p1) {
      assert(x->file == GPR && (acc_p1 - 1) < JAY_MAX_ACCUMS);

      x->file = ACCUM;
      x->reg = (acc_p1 - 1) * 2;
   }
}

/* Do not mix different accumulator sources before XeHP. Repeated uses of
 * the same definition may share an accumulator. Do this before rewriting
 * operands, so canceling an assignment also preserves earlier uses.
 *
 * TODO: Consider operand restrictions during allocation so rejected
 * assignments do not reserve accumulator slots. When candidates conflict,
 * consider their expected benefit instead of keeping the first source.
 */
static void
limit_accumulator_sources(jay_block *block, uint8_t *ra, uint32_t ip)
{
   uint32_t def_ip[JAY_MAX_PHYS_GRF] = { 0 };

   jay_foreach_inst_in_block(block, I) {
      bool has_acc = false;
      uint32_t acc_def = 0;

      jay_foreach_src(I, s) {
         if (I->src[s].file == ACCUM) {
            /* Keep pre-existing sources, including implicit MAC sources. */
            ra[acc_def] = 0;
            acc_def = 0;
            has_acc = true;
         } else if (I->src[s].file == GPR) {
            uint32_t def = def_ip[I->src[s].reg / 2];
            if (ra[def]) {
               if (has_acc && def != acc_def) {
                  ra[def] = 0;
               } else {
                  has_acc = true;
                  acc_def = def;
               }
            }
         }
      }

      if (I->dst.file == GPR) {
         jay_foreach_reg(I->dst, r) {
            def_ip[r / 2] = ip;
         }
      }

      ++ip;
   }
}

static void
pass(jay_function *func)
{
   unsigned nr_accums = jay_num_accums(func->shader);
   if (!nr_accums)
      return;

   void *memctx = ralloc_context(NULL);
   void *linctx = linear_context(memctx);

   /* Analyze the shader globally */
   postra_liveness(func);
   struct util_dynarray candidates = UTIL_DYNARRAY_INIT;

   /* Find the longest block so we can size our allocations & count IPs */
   uint32_t ip_bound = 0;
   jay_foreach_block(func, block) {
      ip_bound = MAX2(ip_bound, list_length(&block->instructions) + 1);
   }

   /* in_use[acc][IP] set if acc is in-use /before/ executing instruction IP */
   BITSET_WORD *in_use[JAY_MAX_ACCUMS];
   for (unsigned i = 0; i < nr_accums; ++i) {
      in_use[i] = BITSET_LINEAR_ZALLOC(linctx, ip_bound);
   }

   /* acc+1 if the instruction writes acc, 0 if no accumulator written */
   uint8_t *ra = linear_zalloc_array(linctx, uint8_t, ip_bound);

   jay_foreach_block(func, block) {
      util_dynarray_clear(&candidates);

      /* Live-set at each point in the program */
      BITSET_DECLARE(live, JAY_NR_POSTRA_LIVE);
      memcpy(live, block->postra_gpr_live_out, sizeof(live));

      uint32_t ip = ip_bound;
      uint32_t last_use_ip[JAY_MAX_PHYS_GRF] = { 0 };
      bool mac_candidates[JAY_MAX_PHYS_GRF] = { false };

      jay_foreach_inst_in_block_rev(block, I) {
         --ip;
         assert(ip > 0 && "invariant");

         /* Collect candidates */
         if (I->dst.file == GPR && last_use_ip[I->dst.reg / 2]) {
            if (can_access_accum(func->shader, I, -1)) {
               struct candidate c = { ip, last_use_ip[I->dst.reg / 2],
                                      mac_candidates[I->dst.reg / 2] };
               util_dynarray_append(&candidates, c);
            }

            last_use_ip[I->dst.reg / 2] = 0;
            mac_candidates[I->dst.reg / 2] = 0;
         }

         jay_foreach_src(I, s) {
            if (I->src[s].file == GPR && source_killed(live, I, s)) {
               last_use_ip[I->src[s].reg / 2] = ip;
               mac_candidates[I->src[s].reg / 2] |=
                  s == 0 && could_be_mac(func->shader, I);
            }
         }

         /* Prune candidates (in a second loop in case of duplicated sources) */
         jay_foreach_src(I, s) {
            if (I->src[s].file == GPR &&
                !can_access_accum(func->shader, I, s)) {

               jay_foreach_reg(I->src[s], r) {
                  last_use_ip[r / 2] = 0;
                  mac_candidates[r / 2] = false;
               }
            }
         }

         /* Implicit use of the integer accumulator acc0 corrupts acc0/acc1,
          * which coresponds to virtual acc0 in SIMD32 mode (a pair) or virtual
          * acc0/acc1 in SIMD16 (two registers). Model interference.
          */
         if (I->op == JAY_OPCODE_MUL_32) {
            unsigned n = func->shader->dispatch_width < 32 ? 2 : 1;

            for (unsigned i = 0; i < MIN2(n, nr_accums); ++i) {
               BITSET_SET(in_use[i], ip);
            }
         }

         postra_liveness_ins(live, I);

         unsigned i;
         BITSET_FOREACH_SET(i, live, nr_accums) {
            BITSET_SET(in_use[i], ip);
         }
      }

      /* If there are no candidates, there's nothing to do.
       *
       * Also avoids potential UB from calling qsort with a null pointer.
       */
      if (candidates.size == 0)
         continue;

      qsort(candidates.data,
            util_dynarray_num_elements(&candidates, struct candidate),
            sizeof(struct candidate), cmp_candidates);

      /* Greedily assign candidates */
      util_dynarray_foreach(&candidates, struct candidate, c) {
         /* MAC can only use acc0 but other instructions have no such
          * restriction. If we know this isn't a candidate for MAC, bias away
          * from acc0 to keep acc0 for some other MAC.
          */
         unsigned bias = c->mac_candidate ? 0 : 1;

         for (unsigned j = 0; j < nr_accums; ++j) {
            unsigned i = j + bias;
            i = i >= nr_accums ? i - nr_accums : i;

            if (!BITSET_TEST_RANGE(in_use[i], c->def_ip + 1, c->last_use_ip)) {
               BITSET_SET_RANGE(in_use[i], c->def_ip + 1, c->last_use_ip);
               ra[c->def_ip] = i + 1;
               break;
            }
         }
      }

      uint32_t min_ip = ip;
      if (func->shader->devinfo->verx10 < 125)
         limit_accumulator_sources(block, ra, min_ip);

      uint8_t gpr_to_acc_p1[JAY_MAX_PHYS_GRF] = { 0 };

      jay_foreach_inst_in_block_safe(block, I) {
         /* Rewrite operands using accumulators */
         jay_foreach_src(I, s) {
            if (I->src[s].file == GPR) {
               substitute_acc(&I->src[s], gpr_to_acc_p1[I->src[s].reg / 2]);
            }
         }

         if (I->dst.file == GPR) {
            /* TGL PRM Vol. 9, p. 371: Explicit accumulator sources and
             * destination must use the same accumulator register.
             * p. 495 forbids simultaneous ACC source/destination for MOV,
             * which is also the native instruction for MODIFIER.
             */
            if (func->shader->devinfo->verx10 < 125 && ra[ip]) {
               jay_foreach_src(I, s) {
                  if (I->src[s].file == ACCUM &&
                      (I->src[s].reg != (ra[ip] - 1) * 2 ||
                       I->op == JAY_OPCODE_MODIFIER)) {
                     ra[ip] = 0;
                     break;
                  }
               }
            }

            jay_foreach_reg(I->dst, r) {
               gpr_to_acc_p1[r / 2] = ra[ip];
            }

            substitute_acc(&I->dst, ra[ip]);
         }

         /* Rewrite MAD->MAC where possible to improve code density. */
         if (could_be_mac(func->shader, I) &&
             I->src[0].file == ACCUM &&
             I->src[0].reg == 0) {
            I->op = JAY_OPCODE_MAC;
            SWAP(I->src[0], I->src[2]);
         }

         /* Sometimes this algorithm turns nontrivial GPR->GPR copies into
          * trivial accumulator->accumulator copies, which can be coalesced now.
          */
         if (I->op == JAY_OPCODE_MOV && jay_regs_equal(I->dst, I->src[0])) {
            jay_remove_instruction(I);
         }

         ++ip;
      }

      assert(ip == ip_bound);

      /* Zero per-block allocation */
      for (unsigned i = 0; i < nr_accums; ++i) {
         BITSET_CLEAR_RANGE(in_use[i], min_ip, ip);
      }

      memset(ra + min_ip, 0, (ip - min_ip) * sizeof(*ra));
   }

   util_dynarray_fini(&candidates);
   ralloc_free(memctx);
}

JAY_DEFINE_FUNCTION_PASS(jay_assign_accumulators, pass)
