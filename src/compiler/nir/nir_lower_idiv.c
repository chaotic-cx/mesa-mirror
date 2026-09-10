/*
 * Copyright © 2015 Red Hat
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
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "nir.h"
#include "nir_builder.h"

/* ported from LLVM's AMDGPUTargetLowering::LowerUDIVREM */
static nir_def *
emit_udiv(nir_builder *bld, nir_def *numer, nir_def *denom, bool modulo)
{
   nir_def *rcp = nir_frcp(bld, nir_u2f32(bld, denom));
   rcp = nir_f2u32(bld, nir_fmul_imm(bld, rcp, 4294966784.0));

   nir_def *neg_rcp_times_denom =
      nir_imul(bld, rcp, nir_ineg(bld, denom));
   rcp = nir_iadd(bld, rcp, nir_umul_high(bld, rcp, neg_rcp_times_denom));

   /* Get initial estimate for quotient/remainder, then refine the estimate
    * in two iterations after */
   nir_def *quotient = nir_umul_high(bld, numer, rcp);
   nir_def *num_s_remainder = nir_imul(bld, quotient, denom);
   nir_def *remainder = nir_isub(bld, numer, num_s_remainder);

   /* First refinement step */
   nir_def *remainder_ge_den = nir_uge(bld, remainder, denom);
   if (!modulo) {
      quotient = nir_bcsel(bld, remainder_ge_den,
                           nir_iadd_imm(bld, quotient, 1), quotient);
   }
   remainder = nir_bcsel(bld, remainder_ge_den,
                         nir_isub(bld, remainder, denom), remainder);

   /* Second refinement step */
   remainder_ge_den = nir_uge(bld, remainder, denom);
   if (modulo) {
      return nir_bcsel(bld, remainder_ge_den, nir_isub(bld, remainder, denom),
                       remainder);
   } else {
      return nir_bcsel(bld, remainder_ge_den, nir_iadd_imm(bld, quotient, 1),
                       quotient);
   }
}

/* 32-bit udiv when fp32 rcp and fma are cheaper than umul_high.
 *
 * emit_udiv() builds a 32.32 reciprocal and applies it with two
 * umul_high.  Where that expands, it is most of the sequence.  This
 * path stays in fp32.  Integer multiplies left are low-32.
 *
 * Needs an under-estimate.  Otherwise the two steps do not converge.
 * Step one can recover any shortfall.  Step two can only add 1.
 *
 *   1. Reciprocal nudged down 8 ulp.  Holds for rcp error in
 *      [-128, +8] ulp. Vulkan allows 2.5.
 *   2. -512.0 covers rounding of u2f32(numer).  ulp reaches 512
 *      near 2^32.
 */
static nir_def *
emit_udiv_fp32(nir_builder *bld, nir_def *numer, nir_def *denom, bool modulo)
{
   nir_def *rcp = nir_frcp(bld, nir_u2f32(bld, denom));

   /* An integer subtract on a positive float's bit pattern steps it down by
    * that many ulp.  The same trick convert_instr_small() uses.
    */
   rcp = nir_iadd_imm(bld, rcp, -8);

   nir_def *quotient =
      nir_f2u32(bld, nir_ffma_weak_imm2(bld, rcp, nir_u2f32(bld, numer),
                                        -512.0));

   nir_def *remainder = nir_isub(bld, numer, nir_imul(bld, quotient, denom));

   /* First correction: divide the remainder the same way.  The bias is one
    * ulp of a small remainder in fp32 and keeps this step from overshooting
    * by one.
    */
   nir_def *step =
      nir_f2u32(bld, nir_ffma_weak_imm2(bld, rcp, nir_u2f32(bld, remainder),
                                        -6.103515625e-05));
   quotient = nir_uadd_sat(bld, quotient, step);
   remainder = nir_isub(bld, numer, nir_imul(bld, quotient, denom));

   /* Second correction: at most one off now. */
   nir_def *ge = nir_uge(bld, remainder, denom);
   if (modulo)
      return nir_bcsel(bld, ge, nir_isub(bld, remainder, denom), remainder);

   return nir_uadd_sat(bld, quotient, nir_b2i32(bld, ge));
}

/* ported from LLVM's AMDGPUTargetLowering::LowerSDIVREM */
static nir_def *
emit_idiv(nir_builder *bld, nir_def *numer, nir_def *denom, nir_op op,
          const nir_lower_idiv_options *options)
{
   nir_def *lhs = nir_iabs(bld, numer);
   nir_def *rhs = nir_iabs(bld, denom);

   if (op == nir_op_idiv) {
      /* We want (numer < 0) ^ (denom < 0). This is the XOR of the sign bits,
       * and since XOR is bitwise, that's the sign bit of the XOR.
       */
      nir_def *d_sign = nir_ilt_imm(bld, nir_ixor(bld, numer, denom), 0);
      nir_def *res = options->fp32_udiv_correction
                        ? emit_udiv_fp32(bld, lhs, rhs, false)
                        : emit_udiv(bld, lhs, rhs, false);
      return nir_bcsel(bld, d_sign, nir_ineg(bld, res), res);
   } else {
      nir_def *lh_sign = nir_ilt_imm(bld, numer, 0);
      nir_def *rh_sign = nir_ilt_imm(bld, denom, 0);

      nir_def *res = options->fp32_udiv_correction
                        ? emit_udiv_fp32(bld, lhs, rhs, true)
                        : emit_udiv(bld, lhs, rhs, true);
      res = nir_bcsel(bld, lh_sign, nir_ineg(bld, res), res);
      if (op == nir_op_imod) {
         nir_def *cond = nir_ieq_imm(bld, res, 0);
         cond = nir_ior(bld, nir_ieq(bld, lh_sign, rh_sign), cond);
         res = nir_bcsel(bld, cond, res, nir_iadd(bld, res, denom));
      }
      return res;
   }
}

static nir_def *
convert_instr_small(nir_builder *b, nir_op op,
                    nir_def *numer, nir_def *denom,
                    const nir_lower_idiv_options *options)
{
   unsigned sz = numer->bit_size;
   nir_alu_type int_type = nir_op_infos[op].output_type | sz;
   nir_alu_type float_type = nir_type_float | (options->allow_fp16 ? sz * 2 : 32);

   nir_def *p = nir_type_convert(b, numer, int_type, float_type, nir_rounding_mode_undef);
   nir_def *q = nir_type_convert(b, denom, int_type, float_type, nir_rounding_mode_undef);

   /* Take 1/q but offset mantissa by 1 to correct for rounding. This is
    * needed for correct results and has been checked exhaustively for
    * all pairs of 16-bit integers */
   nir_def *rcp = nir_iadd_imm(b, nir_frcp(b, q), 1);

   /* Divide by multiplying by adjusted reciprocal */
   nir_def *res = nir_fmul(b, p, rcp);

   /* Convert back to integer space with rounding inferred by type */
   res = nir_type_convert(b, res, float_type, int_type, nir_rounding_mode_undef);

   /* Get remainder given the quotient */
   if (op == nir_op_umod || op == nir_op_imod || op == nir_op_irem)
      res = nir_isub(b, numer, nir_imul(b, denom, res));

   /* Adjust for sign, see constant folding definition */
   if (op == nir_op_imod) {
      nir_def *zero = nir_imm_zero(b, 1, sz);
      nir_def *diff_sign =
         nir_ine(b, nir_ige(b, numer, zero), nir_ige(b, denom, zero));

      nir_def *adjust = nir_iand(b, diff_sign, nir_ine(b, res, zero));
      res = nir_iadd(b, res, nir_bcsel(b, adjust, denom, zero));
   }

   return res;
}

static nir_def *
lower_idiv(nir_builder *b, nir_instr *instr, void *_data)
{
   const nir_lower_idiv_options *options = _data;
   nir_alu_instr *alu = nir_instr_as_alu(instr);

   nir_def *numer = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *denom = nir_ssa_for_alu_src(b, alu, 1);

   b->fp_math_ctrl |= nir_fp_exact;

   if (numer->bit_size < 32)
      return convert_instr_small(b, alu->op, numer, denom, options);
   else if (alu->op == nir_op_udiv || alu->op == nir_op_umod)
      return options->fp32_udiv_correction
                ? emit_udiv_fp32(b, numer, denom, alu->op == nir_op_umod)
                : emit_udiv(b, numer, denom, alu->op == nir_op_umod);
   else
      return emit_idiv(b, numer, denom, alu->op, options);
}

static bool
inst_is_idiv(const nir_instr *instr, UNUSED const void *_state)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);

   if (alu->def.bit_size > 32)
      return false;

   switch (alu->op) {
   case nir_op_idiv:
   case nir_op_udiv:
   case nir_op_imod:
   case nir_op_umod:
   case nir_op_irem:
      return true;
   default:
      return false;
   }
}

bool
nir_lower_idiv(nir_shader *shader, const nir_lower_idiv_options *options)
{
   return nir_shader_lower_instructions(shader,
                                        inst_is_idiv,
                                        lower_idiv,
                                        (void *)options);
}
