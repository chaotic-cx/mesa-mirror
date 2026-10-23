/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "util/lut.h"
#include "jay_builder.h"
#include "jay_ir.h"
#include "jay_opcodes.h"
#include "jay_private.h"

static bool
is_candidate(const jay_inst *I)
{
   return (I->op == JAY_OPCODE_AND ||
           I->op == JAY_OPCODE_OR ||
           I->op == JAY_OPCODE_XOR ||
           I->op == JAY_OPCODE_NOT ||
           I->op == JAY_OPCODE_BFN) &&
          (I->type == JAY_TYPE_U16 || I->type == JAY_TYPE_U32) &&
          !I->saturate &&
          !I->predication &&
          !jay_is_flag(I->dst) &&
          (jay_num_values(I->dst) == 1 ||
           (jay_is_null(I->dst) && !jay_is_null(I->cond_flag))) &&
          (I->conditional_mod == GEN_CONDITION_NONE ||
           I->conditional_mod == GEN_CONDITION_NE ||
           I->conditional_mod == GEN_CONDITION_EQ ||
           I->conditional_mod == GEN_CONDITION_GT ||
           I->conditional_mod == GEN_CONDITION_LT);
}

static util_lut3
apply_boolean(const jay_inst *I, const util_lut3 *src_luts)
{
   switch (I->op) {
   case JAY_OPCODE_AND:
      return src_luts[0] & src_luts[1];
   case JAY_OPCODE_OR:
      return src_luts[0] | src_luts[1];
   case JAY_OPCODE_XOR:
      return src_luts[0] ^ src_luts[1];
   case JAY_OPCODE_NOT:
      return ~src_luts[0];
   case JAY_OPCODE_BFN: {
      util_lut3 result = 0;
      for (unsigned i = 0; i < 8; ++i) {
         unsigned index = ((src_luts[0] >> i) & 1) |
                          (((src_luts[1] >> i) & 1) << 1) |
                          (((src_luts[2] >> i) & 1) << 2);
         result |= ((jay_bfn_ctrl(I) >> index) & 1) << i;
      }
      return result;
   }
   default:
      UNREACHABLE("not a boolean operation");
   }
}

struct bfn_expression {
   jay_def src[3];
   unsigned count;
};

/* Normalize inversion before looking up a source so x and ~x share a LUT
 * variable. Do the same for complemented immediates.
 */
static bool
add_source(struct bfn_expression *expr, jay_def src, util_lut3 *lut)
{
   if (jay_is_flag(src))
      return false;

   if (src.abs || (!jay_is_imm(src) && jay_num_values(src) != 1))
      return false;

   bool invert = src.negate;
   src.negate = false;

   /* Try the complement if the immediate does not fit in 16 bits. */
   if (jay_is_imm(src) && jay_as_uint(src) > UINT16_MAX) {
      uint32_t inverted = ~jay_as_uint(src);
      if (inverted > UINT16_MAX)
         return false;

      src = jay_imm(inverted);
      invert = !invert;
   }

   unsigned s;
   for (s = 0; s < expr->count; ++s) {
      if (jay_defs_equivalent(src, expr->src[s]))
         break;
   }

   if (s == 3)
      return false;

   if (s == expr->count)
      expr->src[expr->count++] = src;

   const util_lut3 vars[] = { UTIL_LUT3(a), UTIL_LUT3(b), UTIL_LUT3(c) };
   *lut = invert ? ~vars[s] : vars[s];
   return true;
}

/* Use the same input variables to combine the LUTs of I and inner. */
static bool
compose_bfn(const jay_inst *I,
            const jay_inst *inner,
            unsigned s,
            struct bfn_expression *expr,
            util_lut3 *lut)
{
   util_lut3 src_luts[3] = { 0 };
   jay_foreach_src(inner, j) {
      assume(j < ARRAY_SIZE(src_luts));
      if (!add_source(expr, inner->src[j], &src_luts[j]))
         return false;
   }

   util_lut3 inner_lut = apply_boolean(inner, src_luts);
   if (I->src[s].negate)
      inner_lut = ~inner_lut;

   jay_foreach_src(I, j) {
      assume(j < ARRAY_SIZE(src_luts));
      if (j == s) {
         src_luts[j] = inner_lut;
      } else if (!add_source(expr, I->src[j], &src_luts[j])) {
         return false;
      }
   }

   *lut = apply_boolean(I, src_luts);

   /* Pad unused sources with zero.  The LUT does not depend on them. */
   for (unsigned j = expr->count; j < 3; ++j)
      expr->src[j] = jay_imm(0);

   /* Only src0 and src2 can encode immediates. */
   unsigned immediates = 0;
   for (unsigned j = 0; j < 3; ++j)
      immediates += jay_is_imm(expr->src[j]);
   if (immediates > 2)
      return false;

   if (jay_is_imm(expr->src[1])) {
      unsigned other = jay_is_imm(expr->src[0]) ? 2 : 0;
      SWAP(expr->src[other], expr->src[1]);
      *lut = util_lut3_swap_sources(*lut, MIN2(other, 1), MAX2(other, 1));
   }

   return true;
}

/* Push NOT into the sources instead of using BFN.  Flag lowering can
 * expose these patterns after propagation has already run.
 */
static bool
compose_not(const jay_inst *inner,
            enum jay_opcode *op,
            struct bfn_expression *expr)
{
   switch (inner->op) {
   case JAY_OPCODE_AND:
      *op = JAY_OPCODE_OR;
      break;
   case JAY_OPCODE_OR:
      *op = JAY_OPCODE_AND;
      break;
   case JAY_OPCODE_XOR:
      *op = JAY_OPCODE_XOR;
      break;
   default:
      return false;
   }

   expr->count = 2;
   for (unsigned s = 0; s < 2; ++s) {
      jay_def src = inner->src[s];
      if (src.abs || (!jay_is_imm(src) && jay_num_values(src) != 1))
         return false;

      if (s == 0 || (inner->op != JAY_OPCODE_XOR)) {
         if (jay_is_imm(src)) {
            src = jay_imm(~jay_as_uint(src) &
                          BITFIELD_MASK(jay_type_size_bits(inner->type)));
         } else {
            src = jay_negate(src);
         }
      }

      expr->src[s] = src;
   }

   return true;
}

/* Combine single-use boolean operations.  Run after propagation so earlier
 * passes can still optimize the individual operations.
 */
static jay_inst *
fuse_bfn(jay_function *f, jay_inst *I, jay_inst **defs, unsigned *uses)
{
   if (!is_candidate(I))
      return I;

   /* Pick the last encodable producer in source order.  When this patch
    * was written that has shown better results in fossil-db than other
    * heuristics.  There's definitely room to improve here, both in what
    * to pick but also how many instructions to consider.
    */
   jay_foreach_ssa_src_rev(I, s) {
      jay_def src = I->src[s];
      if (jay_num_values(src) != 1 || src.abs || uses[jay_index(src)] != 1)
         continue;

      jay_inst *inner = defs[jay_index(src)];
      if (!inner ||
          !is_candidate(inner) ||
          jay_uses_flag(inner) ||
          inner->type != I->type ||
          !jay_defs_equivalent(inner->dst, src))
         continue;

      enum jay_opcode op = JAY_OPCODE_BFN;
      struct bfn_expression expr = { 0 };
      util_lut3 lut = 0;
      if (I->op == JAY_OPCODE_NOT && inner->op != JAY_OPCODE_BFN) {
         if (!compose_not(inner, &op, &expr))
            continue;
      } else if (!compose_bfn(I, inner, s, &expr, &lut)) {
         continue;
      }

      jay_builder b = jay_init_builder(f, jay_before_inst(I));
      jay_inst *J;
      if (op == JAY_OPCODE_BFN) {
         J = jay_BFN(&b, I->type, I->dst, expr.src[0], expr.src[1], expr.src[2],
                     lut);
      } else {
         J = jay_clone_inst(&b, inner, inner->num_srcs);
         J->op = op;
         memcpy(J->src, expr.src, J->num_srcs * sizeof(jay_def));
         jay_builder_insert(&b, J);
      }

      J->dst = I->dst;
      J->cond_flag = I->cond_flag;
      J->conditional_mod = I->conditional_mod;
      /* BFN cannot encode NE.  Use unsigned GT instead. */
      if (J->op == JAY_OPCODE_BFN && J->conditional_mod == GEN_CONDITION_NE)
         J->conditional_mod = GEN_CONDITION_GT;
      J->uniform = I->uniform;
      J->broadcast_flag = I->broadcast_flag;
      J->zero_inactive = I->zero_inactive;

      /* Update use counts so deduplicated sources can become single-use. */
      jay_foreach_src_index(I, _, c, index) {
         assert(uses[index] > 0);
         uses[index]--;
      }
      jay_foreach_src_index(inner, _, c, index) {
         assert(uses[index] > 0);
         uses[index]--;
      }
      jay_foreach_src_index(J, _, c, index) {
         uses[index]++;
      }
      jay_remove_instruction(I);
      jay_remove_instruction(inner);
      return J;
   }

   return I;
}

static void
opt_bfn(jay_function *f)
{
   jay_inst **defs = calloc(f->ssa_alloc, sizeof(*defs));
   unsigned *uses = calloc(f->ssa_alloc, sizeof(*uses));

   jay_foreach_inst_in_func(f, block, I) {
      jay_foreach_src_index(I, s, c, index) {
         uses[index]++;
      }
   }

   jay_foreach_inst_in_func_safe(f, block, I) {
      jay_inst *def = I;
      while (true) {
         jay_inst *J = fuse_bfn(f, def, defs, uses);
         if (J == def)
            break;
         def = J;
      }
      jay_foreach_dst_index(def, _, index) {
         defs[index] = def;
      }
   }

   free(uses);
   free(defs);
}

JAY_DEFINE_FUNCTION_PASS(jay_opt_bfn, opt_bfn)
