/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "jay_builder.h"
#include "jay_ir.h"
#include "jay_opcodes.h"

static void
set_cr0(jay_function *f, jay_cursor cursor, uint32_t *existing, uint32_t desired)
{
   /* Only touch cr0 if we are changing bits */
   if ((*existing) != desired) {
      jay_builder b = jay_init_builder(f, cursor);
      jay_def cr0 = jay_scalar(J_ARF, 32, GEN_ARF_CONTROL);

      jay_XOR(&b, JAY_TYPE_U32, cr0, cr0, (*existing) ^ desired);
      *existing = desired;
   }
}

static bool
is_mul(jay_inst *I)
{
   return I->op == JAY_OPCODE_MUL_32_PART ||
          I->op == JAY_OPCODE_MUL_32X16 ||
          I->op == JAY_OPCODE_MUL;
}

static bool
is_maclh(jay_inst *I)
{
   return I->op == JAY_OPCODE_MACL || I->op == JAY_OPCODE_MACH;
}

/* Wa_18035690555
 *
 * Issue 1: If we have mul <-> mac or macl <-> mach and src1 is
 * the same in current and previous inst, we need to insert a
 * dummy mov in between.
 *
 * Other conditions listed in the issue for mul <-> mac case:
 *    "prev instruction src1 has regioning/scalar" (not flat)
 *    "current instruction src1 is flat and shares the same src1 as prev"
 *
 * Issue 2: prev inst is non-mul or non-macl and src1 is
 * the same in current and previous inst, we need to insert a
 * dummy mov in between.
 */
static void
insert_dummy_mov(jay_builder *b, jay_inst *u, jay_inst *v)
{
   bool issue_1 = ((is_mul(u) && is_mul(v)) || (is_maclh(u) && is_maclh(v))) &&
                  (u->src[1].file == UGPR && jay_num_values(u->src[1]) > 1) &&
                  (v->src[1].file == UGPR && jay_num_values(v->src[1]) == 1) &&
                  (u->src[1].reg >> 1) == (v->src[1].reg >> 1);

   bool issue_2 = !(is_mul(u) || is_maclh(u)) &&
                  jay_num_isa_srcs(u) > 1 && is_maclh(v) &&
                  u->src[1].file == v->src[1].file &&
                  (u->src[1].reg >> 1) == (v->src[1].reg >> 1);

   if (issue_1 || issue_2) {
      jay_MOV(b, jay_null(), 0);
   }
}

void
jay_lower_post_sched(jay_shader *shader, uint32_t api, uint32_t float_sizes)
{
   /* First, work out the global float control mode for the shader */
   uint32_t global = 0x0;

   /* Initially fp16 denorms are flushed-to-zero, handle preserve. */
   if ((api & FLOAT_CONTROLS_DENORM_PRESERVE_FP16) && (float_sizes & 16)) {
      global |= BRW_CR0_FP16_DENORM_PRESERVE;
   }

   /* Initially fp32 denorms are flushed-to-zero, handle preserve.
    *
    * TODO: Optimize this, we have a dispatch bit.
    */
   if ((api & FLOAT_CONTROLS_DENORM_PRESERVE_FP32) && (float_sizes & 32)) {
      global |= BRW_CR0_FP32_DENORM_PRESERVE;
   }

   /* Initially fp64 denorms are flushed to zero, handle preserve. */
   if ((api & FLOAT_CONTROLS_DENORM_PRESERVE_FP64) && (float_sizes & 64)) {
      global |= BRW_CR0_FP64_DENORM_PRESERVE;
   }

   /* By default, we are in round-to-even mode. Note we do not permit setting
    * round mode separately by bitsize but this is ok for current APIs. The
    * Vulkan driver sets roundingModeIndependence = NONE.
    *
    * TODO: Optimize this, there is a command buffer bit for it.
    */
   if (((api & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP16) && (float_sizes & 16)) ||
       ((api & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32) && (float_sizes & 32)) ||
       ((api & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP64) && (float_sizes & 64))) {
      global |= (BRW_RND_MODE_RTZ << BRW_CR0_RND_MODE_SHIFT);
   }

   uint32_t cr0 = 0;
   jay_function *entrypoint = jay_shader_get_entrypoint(shader);
   set_cr0(entrypoint, jay_before_function(entrypoint), &cr0, global);

   /* Now handle per-instruction deltas to the global mode */
   jay_foreach_function(shader, func) {
      jay_foreach_block(func, block) {
         uint32_t current = cr0;
         jay_inst *last = NULL;

         jay_foreach_inst_in_block(block, I) {
            uint32_t required = cr0;
            enum jay_rounding_mode round =
               (I->op == JAY_OPCODE_CVT)      ? jay_cvt_rounding_mode(I) :
               (I->op == JAY_OPCODE_ADD_RTNE) ? JAY_RNE :
                                                JAY_ROUND;

            if (round != JAY_ROUND) {
               required &= ~BRW_CR0_RND_MODE_MASK;
               required |= ((round - JAY_RNE) << BRW_CR0_RND_MODE_SHIFT);
            }

            if (jay_type_is_any_float(I->type)) {
               set_cr0(func, jay_before_inst(I), &current, required);
            }

            jay_builder b = jay_init_builder(func, jay_before_inst(I));

            if (intel_needs_workaround(shader->devinfo, 18035690555) && last) {
               insert_dummy_mov(&b, last, I);
            }

            last = I;
         }

         /* Restore to global state on block boundaries */
         if (jay_num_successors(block, GPR) > 0) {
            set_cr0(func, jay_after_block_logical(block), &current, cr0);
         }
      }
   }
}
