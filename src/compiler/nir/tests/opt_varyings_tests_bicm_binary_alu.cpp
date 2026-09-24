/*
 * Copyright 2023 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

/* Tests for Backward Inter-Shader Code Motion with binary ALU opcodes. */

#include "nir_opt_varyings_test.h"

namespace {

class nir_opt_varyings_test_bicm_binary_alu : public nir_opt_varyings_test {
public:
   unsigned pslot[2], cslot[2], interp[3], test_interp[2];
   nir_intrinsic_instr *store[2] = { NULL };
   nir_def *load[2] = { NULL };
   bool divergent[3];
   nir_op op;
   unsigned bitsize;

   void setup(unsigned slot0, unsigned slot1,
              unsigned in_interp0, unsigned in_interp1,
              mesa_shader_stage producer, mesa_shader_stage consumer,
              nir_alu_type ty, unsigned in_bitsize, nir_op in_op)
   {
      pslot[0] = slot0;
      pslot[1] = slot1;
      cslot[0] = slot0;
      cslot[1] = slot1;
      test_interp[0] = in_interp0;
      test_interp[1] = in_interp1;
      interp[0] = in_interp0;
      interp[1] = in_interp1;
      interp[2] = 0;
      op = in_op;
      bitsize = in_bitsize;

      divergent[0] = interp[0] != INTERP_CONVERGENT;
      divergent[1] = interp[1] != INTERP_CONVERGENT;
      divergent[2] = false;

      /* Choose a random TES interpolation mode, but it must be the same for both inputs. */
      if (interp[0] == INTERP_TES_TRIANGLE && interp[1] == INTERP_TES_TRIANGLE)
         interp[0] = interp[1] = INTERP_TES_TRIANGLE + 1 + rand() % 4;
      if (interp[0] == INTERP_TES_TRIANGLE)
         interp[0] = INTERP_TES_TRIANGLE + 1 + rand() % 4;
      if (interp[1] == INTERP_TES_TRIANGLE)
         interp[1] = INTERP_TES_TRIANGLE + 1 + rand() % 4;

      if (!divergent[0])
         interp[0] = INTERP_LINEAR_CENTROID;
      if (!divergent[1])
         interp[1] = INTERP_LINEAR_CENTROID;
      options.varying_expression_max_cost = NULL; /* don't propagate uniforms */
      create_shaders(producer, consumer);

      for (unsigned s = 0; s < 2; s++) {
         nir_def *input;
         if (!divergent[s])
            input = load_uniform(b1, bitsize, 0);
         else
            input = load_input(b1, (gl_varying_slot)0, s, ty, -1, 0);
         store[s] = store_output(b1, (gl_varying_slot)pslot[s], s,
                                 ty, input, -1, false);
      }

      for (unsigned s = 0; s < 2; s++)
         load[s] = load_input(b2, (gl_varying_slot)cslot[s], s, ty, 0, interp[s]);

      nir_def *value = nir_build_alu2(b2, op, load[0], load[1]);
      if (value->bit_size == 1)
         value = nir_u2uN(b2, value, bitsize);

      store_output(b2, VARYING_SLOT_VAR0, 0, ty, value, 0, false);

      divergent[0] &= !is_patch((gl_varying_slot)pslot[0]);
      divergent[1] &= !is_patch((gl_varying_slot)pslot[1]);
   }

   void check()
   {
      if ((test_interp[0] == test_interp[1] || !divergent[0] || !divergent[1]) &&
          movable_across_interp(b2, op, interp, divergent, bitsize)) {
         ASSERT_EQ(opt_varyings(), (nir_progress_producer | nir_progress_consumer));
         /* An opcode with a convergent non-float result isn't moved into */
         /* the previous shader because a non-float result can't be interpolated. */
         if (!divergent[0] && !divergent[1] && interp[0] != INTERP_FLAT && interp[1] != INTERP_FLAT &&
             !(nir_op_infos[op].output_type & nir_type_float)) {
            ASSERT_TRUE(!shader_contains_alu_op(b1, op, bitsize));
            ASSERT_TRUE(shader_contains_alu_op(b2, op, bitsize));
         } else {
            mesa_shader_stage consumer = b2->shader->info.stage;
            ASSERT_TRUE(shader_contains_alu_op(b1, op, bitsize));
            /* TES uses fadd and fmul for interpolation, so it's always present. */
            if (consumer != MESA_SHADER_TESS_EVAL ||
                (op != nir_op_fadd && op != nir_op_fmul &&
                 op != nir_op_fmad && op != nir_op_ffma)) {
               ASSERT_TRUE(!shader_contains_alu_op(b2, op, bitsize));
            }
         }
         ASSERT_TRUE(shader_contains_instr(b1, &store[0]->instr));
         ASSERT_TRUE(!shader_contains_instr(b1, &store[1]->instr));
         ASSERT_TRUE(!shader_contains_def(b2, load[0]));
         ASSERT_TRUE(!shader_contains_def(b2, load[1]));
      } else {
         ASSERT_EQ(opt_varyings() & nir_progress_consumer, 0);
         ASSERT_TRUE(!shader_contains_alu_op(b1, op, bitsize));
         ASSERT_TRUE(shader_contains_alu_op(b2, op, bitsize));
         ASSERT_TRUE(shader_contains_instr(b1, &store[0]->instr));
         ASSERT_TRUE(shader_contains_instr(b1, &store[1]->instr));
         ASSERT_TRUE(shader_contains_def(b2, load[0]));
         ASSERT_TRUE(shader_contains_def(b2, load[1]));
      }
   }
};

#define TEST_ALU_BINARY(producer_stage, consumer_stage, type, bitsize, slot1, slot2, interp1, interp2, alu) \
TEST_F(nir_opt_varyings_test_bicm_binary_alu, \
       alu##_##type##bitsize##_##producer_stage##_##consumer_stage##_##slot1##_##slot2##_##interp1##_##interp2) \
{ \
   setup(VARYING_SLOT_##slot1, VARYING_SLOT_##slot2, INTERP_##interp1, INTERP_##interp2, \
         MESA_SHADER_##producer_stage, MESA_SHADER_##consumer_stage, \
         nir_type_##type##bitsize, bitsize, nir_op_##alu); \
   check(); \
}

#define TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, interp1, interp2) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 16, slot1, slot2, interp1, interp2, fadd) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fadd) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fdiv) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, feq) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fge) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fmin) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fmax) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fmod) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fmul) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fmulz) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fneu) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fpow) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, frem) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, fsub) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, seq) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, float, 32, slot1, slot2, interp1, interp2, sge)

#define TEST_ALU_BINARY_INT_OPS(producer_stage, consumer_stage, slot1, slot2) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 16, slot1, slot2, FLAT, FLAT, iadd) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, iadd) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, iand) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, idiv) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, ieq) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, ige) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, imax) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, ishl) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, udiv) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, uge) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, umin) \
   TEST_ALU_BINARY(producer_stage, consumer_stage, int, 32, slot1, slot2, FLAT, FLAT, umul_high)

#define TEST_ALU_BINARY_OPS(producer_stage, consumer_stage, slot1, slot2) \
   TEST_ALU_BINARY_INT_OPS(producer_stage, consumer_stage, slot1, slot2) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, FLAT, FLAT)

#define TEST_ALU_BINARY_OPS_FS_INTERP(producer_stage, consumer_stage, slot1, slot2) \
   TEST_ALU_BINARY_OPS(producer_stage, consumer_stage, slot1, slot2) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, FLAT, PERSP_PIXEL) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, FLAT, CONVERGENT) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, CONVERGENT, FLAT) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, PERSP_PIXEL, PERSP_PIXEL) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, PERSP_PIXEL, PERSP_CENTROID) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, PERSP_PIXEL, CONVERGENT) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, CONVERGENT, PERSP_PIXEL) \
   TEST_ALU_BINARY_FP_OPS(producer_stage, consumer_stage, slot1, slot2, CONVERGENT, CONVERGENT)

TEST_ALU_BINARY_FP_OPS(VERTEX, TESS_EVAL, VAR0, VAR1, TES_TRIANGLE, TES_TRIANGLE)
TEST_ALU_BINARY_FP_OPS(TESS_CTRL, TESS_EVAL, VAR0, VAR1, TES_TRIANGLE, TES_TRIANGLE)
TEST_ALU_BINARY_FP_OPS(TESS_CTRL, TESS_EVAL, VAR0, PATCH0, TES_TRIANGLE, FLAT)
TEST_ALU_BINARY_OPS(TESS_CTRL, TESS_EVAL, PATCH0, PATCH1)

TEST_ALU_BINARY_OPS_FS_INTERP(VERTEX, FRAGMENT, VAR0, VAR1)
TEST_ALU_BINARY_OPS_FS_INTERP(TESS_EVAL, FRAGMENT, VAR0, VAR1)

// TODO: unary/ternary, uniform/UBO load/constant

}
