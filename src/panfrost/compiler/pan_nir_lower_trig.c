/*
 * Copyright (C) 2021,2026 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "pan_nir.h"

/* The hardware has extremely coarse tables for approximating sin/cos,
 * accessible as FSIN/COS_TABLE.u6, which multiplies the bottom 6-bits by pi/32
 * and calculates the results. We use them to calculate sin/cos via a Taylor
 * approximation:
 *
 * f(x + e) = f(x) + e f'(x) + (e^2)/2 f''(x)
 * sin(x + e) = sin(x) + e cos(x) - (e^2)/2 sin(x)
 * cos(x + e) = cos(x) - e sin(x) - (e^2)/2 cos(x)
 */
static bool
lower_trig_impl(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_fsin && alu->op != nir_op_fcos)
      return false;

   assert(alu->def.bit_size == 32);
   b->cursor = nir_before_instr(&alu->instr);

   nir_def *two_over_pi  = nir_imm_float(b, 2.0f / M_PI);
   nir_def *mpi_over_two = nir_imm_float(b, -M_PI / 2.0);
   nir_def *sincos_bias  = nir_imm_int(b, 0x49400000);
   nir_def *negzero      = nir_imm_float(b, -0.0f);

   bool is_cos = alu->op == nir_op_fcos;
   nir_def *src = nir_ssa_for_alu_src(b, alu, 0);

   /* We require precise rounding */
   b->fp_math_ctrl = nir_fp_exact;

   /* sincos_bias forces ULP=1/16, this with the 2/PI mul makes the lower 6 bits
    * of x_u6 = round(x * 32 / pi)
    */
   nir_def *x_u6 = nir_ffma(b, src, two_over_pi, sincos_bias);
   /* Approximate domain error (small) */
   nir_def *e = nir_ffma(b, nir_fadd(b, x_u6, nir_fneg(b, sincos_bias)),
                         mpi_over_two, src);

   /* Lookup sin(x), cos(x) */
   nir_def *sinx = nir_fsin_table_pan(b, x_u6);
   nir_def *cosx = nir_fcos_table_pan(b, x_u6);
   nir_def *fx = is_cos ? cosx : sinx;

   /* e^2 / 2 */
   nir_def *e2_over_2 =
      nir_fma_rscale_pan(b, e, e, negzero, nir_imm_int(b, -1));

   /* (-e^2)/2 f''(x) */
   nir_def *quadratic = nir_fmul(b, nir_fneg(b, e2_over_2), fx);
   /* e f'(x) - (e^2/2) f''(x) */
   nir_def *lin = nir_ffma(b, e, is_cos ? nir_fneg(b, sinx) : cosx, quadratic);
   lin = nir_fsat_signed(b, lin);

   /* f(x) + e f'(x) - (e^2/2) f''(x) */
   nir_def_replace(&alu->def, nir_fadd(b, lin, fx));
   return true;
}

/* The hardware does not have native SIN/COS instructions, it instead has tables
 * with very approximate results.  What we do have is Taylor series and a lot of
 * good will.
 */
bool
pan_nir_lower_trig(nir_shader *nir)
{
   return nir_shader_alu_pass(nir, lower_trig_impl, nir_metadata_control_flow,
                              NULL);
}
