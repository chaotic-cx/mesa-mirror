/*
 * Copyright © 2016 Intel Corporation
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

#include <gtest/gtest.h>
#include "brw_disasm_info.h"
#include "brw_eu.h"
#include "brw_eu_defines.h"
#include "util/bitset.h"
#include "util/ralloc.h"

static const struct intel_gfx_info {
   const char *name;
} gfx_names[] = {
   { "skl", },
   { "bxt", },
   { "kbl", },
   { "aml", },
   { "glk", },
   { "cfl", },
   { "whl", },
   { "cml", },
   { "icl", },
   { "ehl", },
   { "jsl", },
   { "tgl", },
   { "rkl", },
   { "dg1", },
   { "adl", },
   { "sg1", },
   { "rpl", },
   { "dg2", },
   { "mtl", },
   { "lnl", },
   { "bmg", },
   { "ptl", },
};

class validation_test: public ::testing::TestWithParam<struct intel_gfx_info> {
   virtual void SetUp();

public:
   validation_test();
   virtual ~validation_test();

   struct brw_isa_info isa;
   struct brw_codegen *p;
   struct intel_device_info devinfo;
};

validation_test::validation_test()
{
   p = rzalloc(NULL, struct brw_codegen);
   memset(&devinfo, 0, sizeof(devinfo));
}

validation_test::~validation_test()
{
   ralloc_free(p);
}

void validation_test::SetUp()
{
   struct intel_gfx_info info = GetParam();
   int devid = intel_device_name_to_pci_device_id(info.name);

   intel_get_device_info_from_pci_id(devid, &devinfo);

   brw_init_isa_info(&isa, &devinfo);

   brw_init_codegen(&isa, p, p);
}

struct gfx_name {
   template <class ParamType>
   std::string
   operator()(const ::testing::TestParamInfo<ParamType>& info) const {
      return info.param.name;
   }
};

INSTANTIATE_TEST_SUITE_P(
   eu_assembly, validation_test,
   ::testing::ValuesIn(gfx_names),
   gfx_name()
);

static bool
validate(struct brw_codegen *p, char **error = nullptr)
{
   const bool print = getenv("TEST_DEBUG");
   struct disasm_info *disasm = disasm_initialize(p->isa, NULL);

   struct inst_group *group = disasm_new_inst_group(disasm, 0);
   disasm_new_inst_group(disasm, p->next_insn_offset);

   bool ret = brw_validate_instructions(p->isa, p->store, 0,
                                        p->next_insn_offset, disasm);

   if (print) {
      dump_assembly(p->store, 0, p->next_insn_offset, disasm, NULL);
   }

   if (error)
      *error = ralloc_strdup(p, group->error);

   ralloc_free(disasm);

   return ret;
}

#define last_inst    (&p->store[p->nr_insn - 1])
#define g0           brw_vec8_grf(0, 0)
#define acc0         brw_acc_reg(8)
#define null         brw_null_reg()
#define zero         brw_imm_f(0.0f)

static void
clear_instructions(struct brw_codegen *p)
{
   p->next_insn_offset = 0;
   p->nr_insn = 0;
}

TEST_P(validation_test, sanity)
{
   brw_ADD(p, g0, g0, g0);

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, src0_null_reg)
{
   brw_MOV(p, g0, null);

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, src1_null_reg)
{
   brw_ADD(p, g0, g0, null);

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, math_src0_null_reg)
{
   gfx6_math(p, g0, BRW_MATH_FUNCTION_SIN, null, null);

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, math_src1_null_reg)
{
   gfx6_math(p, g0, BRW_MATH_FUNCTION_POW, g0, null);
   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, opcode46)
{
   /* opcode 46 is "push" on Gen 4 and 5
    *              "fork" on Gen 6
    *              reserved on Gen 7
    *              "goto" on Gfx8+
    */
   brw_next_insn(p, brw_opcode_decode(&isa, 46));

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, invalid_exec_size_encoding)
{
   const struct {
      enum brw_execution_size exec_size;
      bool expected_result;
   } test_case[] = {
      { BRW_EXECUTE_1,      true  },
      { BRW_EXECUTE_2,      true  },
      { BRW_EXECUTE_4,      true  },
      { BRW_EXECUTE_8,      true  },
      { BRW_EXECUTE_16,     true  },
      { BRW_EXECUTE_32,     true  },

      { (enum brw_execution_size)((int)BRW_EXECUTE_32 + 1), false },
      { (enum brw_execution_size)((int)BRW_EXECUTE_32 + 2), false },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(test_case); i++) {
      brw_MOV(p, g0, g0);

      brw_eu_inst_set_exec_size(&devinfo, last_inst, test_case[i].exec_size);
      brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
      brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);

      if (test_case[i].exec_size == BRW_EXECUTE_1) {
         brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
         brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
         brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);
      } else {
         brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_2);
         brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_2);
         brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
      }

      EXPECT_EQ(test_case[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, invalid_type_encoding)
{
   enum brw_reg_file files[2] = {
      FIXED_GRF,
      IMM,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(files); i++) {
      const enum brw_reg_file file = files[i];
      const int num_bits = 4;
      const int num_encodings = 1 << num_bits;

      /* The data types are encoded into <num_bits> bits to be used in hardware
       * instructions, so keep a record in a bitset the invalid patterns so
       * they can be verified to be invalid when used.
       */
      BITSET_DECLARE(invalid_encodings, num_encodings);

      const struct {
         enum brw_reg_type type;
         bool expected_result;
      } test_case[] = {
         { BRW_TYPE_DF, devinfo.has_64bit_float },
         { BRW_TYPE_F,  true },
         { BRW_TYPE_HF, true },
         { BRW_TYPE_VF, file == IMM },
         { BRW_TYPE_Q,  devinfo.has_64bit_int },
         { BRW_TYPE_UQ, devinfo.has_64bit_int },
         { BRW_TYPE_D,  true },
         { BRW_TYPE_UD, true },
         { BRW_TYPE_W,  true },
         { BRW_TYPE_UW, true },
         { BRW_TYPE_B,  file == FIXED_GRF },
         { BRW_TYPE_UB, file == FIXED_GRF },
         { BRW_TYPE_V,  file == IMM },
         { BRW_TYPE_UV, file == IMM },
      };

      /* Initially assume all hardware encodings are invalid */
      BITSET_ONES(invalid_encodings);

      brw_set_default_exec_size(p, BRW_EXECUTE_4);

      for (unsigned i = 0; i < ARRAY_SIZE(test_case); i++) {
         if (test_case[i].expected_result) {
            unsigned hw_type = brw_type_encode(&devinfo, file, test_case[i].type);
            if (hw_type != INVALID_HW_REG_TYPE) {
               /* ... and remove valid encodings from the set */
               assert(BITSET_TEST(invalid_encodings, hw_type));
               BITSET_CLEAR(invalid_encodings, hw_type);
            }

            if (file == FIXED_GRF) {
               struct brw_reg g = retype(g0, test_case[i].type);
               brw_MOV(p, g, g);
               brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
               brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
               brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
            } else {
               enum brw_reg_type t;

               switch (test_case[i].type) {
               case BRW_TYPE_V:
                  t = BRW_TYPE_W;
                  break;
               case BRW_TYPE_UV:
                  t = BRW_TYPE_UW;
                  break;
               case BRW_TYPE_VF:
                  t = BRW_TYPE_F;
                  break;
               default:
                  t = test_case[i].type;
                  break;
               }

               struct brw_reg g = retype(g0, t);
               brw_MOV(p, g, retype(brw_imm_w(0), test_case[i].type));
            }

            EXPECT_TRUE(validate(p));

            clear_instructions(p);
         }
      }

      /* The remaining encodings in invalid_encodings do not have a mapping
       * from BRW_TYPE_* and must be invalid. Verify that invalid
       * encodings are rejected by the validator.
       */
      int e;
      BITSET_FOREACH_SET(e, invalid_encodings, num_encodings) {
         if (file == FIXED_GRF) {
            brw_MOV(p, g0, g0);
            brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
            brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
            brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
         } else {
            brw_MOV(p, g0, brw_imm_w(0));
         }
         brw_eu_inst_set_dst_reg_hw_type(&devinfo, last_inst, e);
         brw_eu_inst_set_src0_reg_hw_type(&devinfo, last_inst, e);

         EXPECT_FALSE(validate(p));

         clear_instructions(p);
      }
   }
}

TEST_P(validation_test, invalid_type_encoding_3src_a16)
{
   /* 3-src instructions in align16 mode only supported on Gfx6-9. */
   if (devinfo.ver != 9)
      return;

   const int num_bits = 3;
   const int num_encodings = 1 << num_bits;

   /* The data types are encoded into <num_bits> bits to be used in hardware
    * instructions, so keep a record in a bitset the invalid patterns so
    * they can be verified to be invalid when used.
    */
   BITSET_DECLARE(invalid_encodings, num_encodings);

   const struct {
      enum brw_reg_type type;
      bool expected_result;
   } test_case[] = {
      { BRW_TYPE_DF, true },
      { BRW_TYPE_F,  true },
      { BRW_TYPE_HF, true },
      { BRW_TYPE_D,  true },
      { BRW_TYPE_UD, true },
   };

   /* Initially assume all hardware encodings are invalid */
   BITSET_ONES(invalid_encodings);

   brw_set_default_access_mode(p, BRW_ALIGN_16);
   brw_set_default_exec_size(p, BRW_EXECUTE_4);

   for (unsigned i = 0; i < ARRAY_SIZE(test_case); i++) {
      if (test_case[i].expected_result) {
         unsigned hw_type =
            brw_type_encode_for_3src(&devinfo, test_case[i].type);
         if (hw_type != INVALID_HW_REG_TYPE) {
            /* ... and remove valid encodings from the set */
            assert(BITSET_TEST(invalid_encodings, hw_type));
            BITSET_CLEAR(invalid_encodings, hw_type);
         }

         struct brw_reg g = retype(g0, test_case[i].type);
         if (!brw_type_is_int(test_case[i].type)) {
            brw_MAD(p, g, g, g, g);
         } else {
            brw_BFE(p, g, g, g, g);
         }

         EXPECT_TRUE(validate(p));

         clear_instructions(p);
      }
   }

   /* The remaining encodings in invalid_encodings do not have a mapping
    * from BRW_TYPE_* and must be invalid. Verify that invalid
    * encodings are rejected by the validator.
    */
   int e;
   BITSET_FOREACH_SET(e, invalid_encodings, num_encodings) {
      for (unsigned i = 0; i < 2; i++) {
         if (i == 0) {
            brw_MAD(p, g0, g0, g0, g0);
         } else {
            brw_BFE(p, g0, g0, g0, g0);
         }

         brw_eu_inst_set_3src_a16_dst_hw_type(&devinfo, last_inst, e);
         brw_eu_inst_set_3src_a16_src_hw_type(&devinfo, last_inst, e);

         EXPECT_FALSE(validate(p));

         clear_instructions(p);
      }
   }
}

TEST_P(validation_test, invalid_type_encoding_3src_a1)
{
   /* 3-src instructions in align1 mode only supported on Gfx11+ */
   if (devinfo.ver == 9)
      return;

   const int num_bits = 3 + 1 /* for exec_type */;
   const int num_encodings = 1 << num_bits;

   /* The data types are encoded into <num_bits> bits to be used in hardware
    * instructions, so keep a record in a bitset the invalid patterns so
    * they can be verified to be invalid when used.
    */
   BITSET_DECLARE(invalid_encodings, num_encodings);

   const struct {
      enum brw_reg_type type;
      unsigned exec_type;
      bool expected_result;
   } test_case[] = {
#define E(x) ((unsigned)BRW_ALIGN1_3SRC_EXEC_TYPE_##x)
      { BRW_TYPE_DF, E(FLOAT), devinfo.has_64bit_float },
      { BRW_TYPE_F,  E(FLOAT), true  },
      { BRW_TYPE_HF, E(FLOAT), true  },
      { BRW_TYPE_Q,  E(INT),   devinfo.has_64bit_int },
      { BRW_TYPE_UQ, E(INT),   devinfo.has_64bit_int },
      { BRW_TYPE_D,  E(INT),   true  },
      { BRW_TYPE_UD, E(INT),   true  },
      { BRW_TYPE_W,  E(INT),   true  },
      { BRW_TYPE_UW, E(INT),   true  },
      { BRW_TYPE_BF, E(FLOAT), devinfo.has_bfloat16 },

      /* There are no ternary instructions that can operate on B-type sources
       * on Gfx11-12. Src1/Src2 cannot be B-typed either.
       */
      { BRW_TYPE_B,  E(INT),   false },
      { BRW_TYPE_UB, E(INT),   false },
   };

   /* Initially assume all hardware encodings are invalid */
   BITSET_ONES(invalid_encodings);

   brw_set_default_access_mode(p, BRW_ALIGN_1);
   brw_set_default_exec_size(p, BRW_EXECUTE_4);

   for (unsigned i = 0; i < ARRAY_SIZE(test_case); i++) {
      if (test_case[i].expected_result) {
         unsigned hw_type =
            brw_type_encode_for_3src(&devinfo, test_case[i].type);
         unsigned hw_exec_type = hw_type | (test_case[i].exec_type << 3);
         if (hw_type != INVALID_HW_REG_TYPE) {
            /* ... and remove valid encodings from the set */
            assert(BITSET_TEST(invalid_encodings, hw_exec_type));
            BITSET_CLEAR(invalid_encodings, hw_exec_type);
         }

         struct brw_reg g = retype(g0, test_case[i].type);
         if (brw_type_is_bfloat(test_case[i].type)) {
            /* BF is more restrictive, so ensure the instruction is valid. */
            brw_MAD(p, retype(g, BRW_TYPE_F), g, g, retype(g, BRW_TYPE_F));
         } else if (!brw_type_is_int(test_case[i].type)) {
            brw_MAD(p, g, g, g, g);
         } else {
            brw_BFE(p, g, g, g, g);
         }

         char *error = NULL;
         EXPECT_TRUE(validate(p, &error)) << "Unexpected validation failure: " << error;

         clear_instructions(p);
      }
   }

   /* The remaining encodings in invalid_encodings do not have a mapping
    * from BRW_TYPE_* and must be invalid. Verify that invalid
    * encodings are rejected by the validator.
    */
   int e;
   BITSET_FOREACH_SET(e, invalid_encodings, num_encodings) {
      const unsigned hw_type = e & 0x7;
      const unsigned exec_type = e >> 3;

      for (unsigned i = 0; i < 2; i++) {
         if (i == 0) {
            brw_MAD(p, g0, g0, g0, g0);
            brw_eu_inst_set_3src_a1_exec_type(&devinfo, last_inst, BRW_ALIGN1_3SRC_EXEC_TYPE_FLOAT);
         } else {
            brw_CSEL(p, g0, g0, g0, g0);
            brw_eu_inst_set_3src_cond_modifier(&devinfo, last_inst, BRW_CONDITIONAL_NZ);
            brw_eu_inst_set_3src_a1_exec_type(&devinfo, last_inst, BRW_ALIGN1_3SRC_EXEC_TYPE_INT);
         }

         brw_eu_inst_set_3src_a1_exec_type(&devinfo, last_inst, exec_type);
         brw_eu_inst_set_3src_a1_dst_hw_type (&devinfo, last_inst, hw_type);
         brw_eu_inst_set_3src_a1_src0_hw_type(&devinfo, last_inst, hw_type);
         brw_eu_inst_set_3src_a1_src1_hw_type(&devinfo, last_inst, hw_type);
         brw_eu_inst_set_3src_a1_src2_hw_type(&devinfo, last_inst, hw_type);

         EXPECT_FALSE(validate(p));

         clear_instructions(p);
      }
   }
}

TEST_P(validation_test, 3src_inst_access_mode)
{
   /* No access mode bit on Gfx12+ */
   if (devinfo.ver >= 12)
      return;

   const struct {
      unsigned mode;
      bool expected_result;
   } test_case[] = {
      { BRW_ALIGN_1,  devinfo.ver != 9 },
      { BRW_ALIGN_16, devinfo.ver == 9 },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(test_case); i++) {
      if (devinfo.ver == 9)
         brw_set_default_access_mode(p, BRW_ALIGN_16);

      brw_MAD(p, g0, g0, g0, g0);
      brw_eu_inst_set_access_mode(&devinfo, last_inst, test_case[i].mode);

      EXPECT_EQ(test_case[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

/* When the Execution Data Type is wider than the destination data type, the
 * destination must [...] specify a HorzStride equal to the ratio in sizes of
 * the two data types.
 */
TEST_P(validation_test, dest_stride_must_be_equal_to_the_ratio_of_exec_size_to_dest_size)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);

   EXPECT_TRUE(validate(p));
}

/* When the Execution Data Type is wider than the destination data type, the
 * destination must be aligned as required by the wider execution data type
 * [...]
 */
TEST_P(validation_test, dst_subreg_must_be_aligned_to_exec_type_size)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 2);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 8);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));
}

/* ExecSize must be greater than or equal to Width. */
TEST_P(validation_test, exec_size_less_than_width)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_16);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_16);

   EXPECT_FALSE(validate(p));
}

/* If ExecSize = Width and HorzStride ≠ 0,
 * VertStride must be set to Width * HorzStride.
 */
TEST_P(validation_test, vertical_stride_is_width_by_horizontal_stride)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);

   EXPECT_FALSE(validate(p));
}

/* If Width = 1, HorzStride must be 0 regardless of the values
 * of ExecSize and VertStride.
 */
TEST_P(validation_test, horizontal_stride_must_be_0_if_width_is_1)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_FALSE(validate(p));
}

/* If ExecSize = Width = 1, both VertStride and HorzStride must be 0. */
TEST_P(validation_test, scalar_region_must_be_0_1_0)
{
   struct brw_reg g0_0 = brw_vec1_grf(0, 0);

   brw_ADD(p, g0, g0, g0_0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_1);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_1);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0_0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_1);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_1);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));
}

/* If VertStride = HorzStride = 0, Width must be 1 regardless of the value
 * of ExecSize.
 */
TEST_P(validation_test, zero_stride_implies_0_1_0)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));
}

/* Dst.HorzStride must not be 0. */
TEST_P(validation_test, dst_horizontal_stride_0)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   /* Align16 does not exist on Gfx11+ */
   if (devinfo.ver >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));
}

/* VertStride must be used to cross FIXED_GRF register boundaries. This rule implies
 * that elements within a 'Width' cannot cross FIXED_GRF boundaries.
 */
TEST_P(validation_test, must_not_cross_grf_boundary_in_a_width)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, 4);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src1_da1_subreg_nr(&devinfo, last_inst, 4);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_FALSE(validate(p));
}

/* Destination Horizontal must be 1 in Align16 */
TEST_P(validation_test, dst_hstride_on_align16_must_be_1)
{
   /* Align16 does not exist on Gfx11+ */
   if (devinfo.ver >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));
}

/* VertStride must be 0 or 4 in Align16 */
TEST_P(validation_test, vstride_on_align16_must_be_0_or_4)
{
   /* Align16 does not exist on Gfx11+ */
   if (devinfo.ver >= 11)
      return;

   const struct {
      enum brw_vertical_stride vstride;
      bool expected_result;
   } vstride[] = {
      { BRW_VERTICAL_STRIDE_0, true },
      { BRW_VERTICAL_STRIDE_1, false },
      { BRW_VERTICAL_STRIDE_2, true },
      { BRW_VERTICAL_STRIDE_4, true },
      { BRW_VERTICAL_STRIDE_8, false },
      { BRW_VERTICAL_STRIDE_16, false },
      { BRW_VERTICAL_STRIDE_32, false },
      { BRW_VERTICAL_STRIDE_ONE_DIMENSIONAL, false },
   };

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   for (unsigned i = 0; i < ARRAY_SIZE(vstride); i++) {
      brw_ADD(p, g0, g0, g0);
      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, vstride[i].vstride);

      EXPECT_EQ(vstride[i].expected_result, validate(p));

      clear_instructions(p);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(vstride); i++) {
      brw_ADD(p, g0, g0, g0);
      brw_eu_inst_set_src1_vstride(&devinfo, last_inst, vstride[i].vstride);

      EXPECT_EQ(vstride[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

/* In Direct Addressing mode, a source cannot span more than 2 adjacent FIXED_GRF
 * registers.
 */
TEST_P(validation_test, source_cannot_span_more_than_2_registers)
{
   enum brw_reg_type type = devinfo.ver >= 20 ? BRW_TYPE_D : BRW_TYPE_W;

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_32);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_8);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_8);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_eu_inst_set_src1_da1_subreg_nr(&devinfo, last_inst, 2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);

   EXPECT_TRUE(validate(p));
}

/* A destination cannot span more than 2 adjacent FIXED_GRF registers. */
TEST_P(validation_test, destination_cannot_span_more_than_2_registers)
{
   unsigned invalid_stride = devinfo.ver >= 20 ? 4 : 2;

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_32);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, cvt(invalid_stride));
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_8);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 6);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_4);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, src_region_spans_two_regs_dst_region_spans_one)
{
   const enum brw_reg_type type = devinfo.ver >= 20 ? BRW_TYPE_D : BRW_TYPE_W;

   /* Writes to dest are to the lower OWord */
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   /* Writes to dest are to the upper OWord */
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 16);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   /* Writes to dest are evenly split between OWords */
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_8);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   /* Writes to dest are uneven between OWords */
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 10);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, type);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, dst_elements_must_be_evenly_split_between_registers)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 4);

   if (devinfo.verx10 < 125) {
      EXPECT_TRUE(validate(p));
   } else {
      EXPECT_FALSE(validate(p));
   }

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   gfx6_math(p, g0, BRW_MATH_FUNCTION_SIN, g0, null);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   gfx6_math(p, g0, BRW_MATH_FUNCTION_SIN, g0, null);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 4);

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, two_src_two_dst_source_offsets_must_be_same)
{
   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_4);
   brw_eu_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, 16);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_2);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   if (devinfo.verx10 >= 125) {
      EXPECT_FALSE(validate(p));
   } else {
      EXPECT_TRUE(validate(p));
   }

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_4);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_8);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   if (devinfo.verx10 >= 125)
      EXPECT_FALSE(validate(p));
   else
      EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, two_src_two_dst_each_dst_must_be_derived_from_one_src)
{
   brw_MOV(p, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, 8);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_MOV(p, g0, g0);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 16);
   brw_eu_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, 8);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_2);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   if (devinfo.verx10 >= 125) {
      EXPECT_FALSE(validate(p));
   } else {
      EXPECT_TRUE(validate(p));
   }
}

TEST_P(validation_test, one_src_two_dst)
{
   struct brw_reg g0_0 = brw_vec1_grf(0, 0);

   brw_ADD(p, g0, g0_0, g0_0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_D);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_eu_inst_set_dst_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);
   brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);
   brw_eu_inst_set_src1_file_type(&devinfo, last_inst, FIXED_GRF, BRW_TYPE_W);

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, packed_byte_destination)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;
      bool neg, abs, sat;
      bool expected_result;
   } move[] = {
      { BRW_TYPE_UB, BRW_TYPE_UB, 0, 0, 0, true },
      { BRW_TYPE_B , BRW_TYPE_B , 0, 0, 0, true },
      { BRW_TYPE_UB, BRW_TYPE_B , 0, 0, 0, true },
      { BRW_TYPE_B , BRW_TYPE_UB, 0, 0, 0, true },

      { BRW_TYPE_UB, BRW_TYPE_UB, 1, 0, 0, false },
      { BRW_TYPE_B , BRW_TYPE_B , 1, 0, 0, false },
      { BRW_TYPE_UB, BRW_TYPE_B , 1, 0, 0, false },
      { BRW_TYPE_B , BRW_TYPE_UB, 1, 0, 0, false },

      { BRW_TYPE_UB, BRW_TYPE_UB, 0, 1, 0, false },
      { BRW_TYPE_B , BRW_TYPE_B , 0, 1, 0, false },
      { BRW_TYPE_UB, BRW_TYPE_B , 0, 1, 0, false },
      { BRW_TYPE_B , BRW_TYPE_UB, 0, 1, 0, false },

      { BRW_TYPE_UB, BRW_TYPE_UB, 0, 0, 1, false },
      { BRW_TYPE_B , BRW_TYPE_B , 0, 0, 1, false },
      { BRW_TYPE_UB, BRW_TYPE_B , 0, 0, 1, false },
      { BRW_TYPE_B , BRW_TYPE_UB, 0, 0, 1, false },

      { BRW_TYPE_UB, BRW_TYPE_UW, 0, 0, 0, false },
      { BRW_TYPE_B , BRW_TYPE_W , 0, 0, 0, false },
      { BRW_TYPE_UB, BRW_TYPE_UD, 0, 0, 0, false },
      { BRW_TYPE_B , BRW_TYPE_D , 0, 0, 0, false },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(move); i++) {
      brw_MOV(p, retype(g0, move[i].dst_type), retype(g0, move[i].src_type));
      brw_eu_inst_set_src0_negate(&devinfo, last_inst, move[i].neg);
      brw_eu_inst_set_src0_abs(&devinfo, last_inst, move[i].abs);
      brw_eu_inst_set_saturate(&devinfo, last_inst, move[i].sat);

      EXPECT_EQ(move[i].expected_result, validate(p));

      clear_instructions(p);
   }

   brw_SEL(p, retype(g0, BRW_TYPE_UB),
              retype(g0, BRW_TYPE_UB),
              retype(g0, BRW_TYPE_UB));
   brw_eu_inst_set_pred_control(&devinfo, last_inst, BRW_PREDICATE_NORMAL);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_SEL(p, retype(g0, BRW_TYPE_B),
              retype(g0, BRW_TYPE_B),
              retype(g0, BRW_TYPE_B));
   brw_eu_inst_set_pred_control(&devinfo, last_inst, BRW_PREDICATE_NORMAL);

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, byte_destination_relaxed_alignment)
{
   brw_SEL(p, retype(g0, BRW_TYPE_B),
              retype(g0, BRW_TYPE_W),
              retype(g0, BRW_TYPE_W));
   brw_eu_inst_set_pred_control(&devinfo, last_inst, BRW_PREDICATE_NORMAL);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_SEL(p, retype(g0, BRW_TYPE_B),
              retype(g0, BRW_TYPE_W),
              retype(g0, BRW_TYPE_W));
   brw_eu_inst_set_pred_control(&devinfo, last_inst, BRW_PREDICATE_NORMAL);
   brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 1);

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, byte_64bit_conversion)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;
      unsigned dst_stride;
      bool expected_result;
   } inst[] = {
#define INST(dst_type, src_type, dst_stride, expected_result)             \
      {                                                                   \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src_type,                                             \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                              \
         expected_result,                                                 \
      }

      INST(B,   Q, 1, false),
      INST(B,  UQ, 1, false),
      INST(B,  DF, 1, false),
      INST(UB,  Q, 1, false),
      INST(UB, UQ, 1, false),
      INST(UB, DF, 1, false),

      INST(B,   Q, 2, false),
      INST(B,  UQ, 2, false),
      INST(B , DF, 2, false),
      INST(UB,  Q, 2, false),
      INST(UB, UQ, 2, false),
      INST(UB, DF, 2, false),

      INST(B,   Q, 4, false),
      INST(B,  UQ, 4, false),
      INST(B,  DF, 4, false),
      INST(UB,  Q, 4, false),
      INST(UB, UQ, 4, false),
      INST(UB, DF, 4, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      if (!devinfo.has_64bit_float &&
          inst[i].src_type == BRW_TYPE_DF)
         continue;

      if (!devinfo.has_64bit_int &&
          (inst[i].src_type == BRW_TYPE_Q ||
           inst[i].src_type == BRW_TYPE_UQ))
         continue;

      brw_MOV(p, retype(g0, inst[i].dst_type), retype(g0, inst[i].src_type));
      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);
      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, half_float_conversion)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;
      unsigned dst_stride;
      unsigned dst_subnr;
      bool expected_result_gfx9;
      bool expected_result_gfx125;
   } inst[] = {
#define INST(dst_type, src_type, dst_stride, dst_subnr,                     \
             expected_result_gfx9,                                          \
             expected_result_gfx125)                                        \
      {                                                                     \
         BRW_TYPE_##dst_type,                                               \
         BRW_TYPE_##src_type,                                               \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                \
         dst_subnr,                                                         \
         expected_result_gfx9,                                              \
         expected_result_gfx125,                                            \
      }

      /* MOV to half-float destination */
      INST(HF,  B, 1, 0, false, false), /* 0 */
      INST(HF,  W, 1, 0, false, false),
      INST(HF, HF, 1, 0, true,  true),
      INST(HF, HF, 1, 2, true,  false),
      INST(HF,  D, 1, 0, false, false),
      INST(HF,  F, 1, 0, true,  false),
      INST(HF,  Q, 1, 0, false, false),
      INST(HF,  B, 2, 0, true,  false),
      INST(HF,  B, 2, 2, false, false),
      INST(HF,  W, 2, 0, true,  false),
      INST(HF,  W, 2, 2, false, false), /* 10 */
      INST(HF, HF, 2, 0, true,  false),
      INST(HF, HF, 2, 2, true,  false),
      INST(HF,  D, 2, 0, true,  true),
      INST(HF,  D, 2, 2, false, false),
      INST(HF,  F, 2, 0, true,  true),
      INST(HF,  F, 2, 2, true,  false),
      INST(HF,  Q, 2, 0, false, false),
      INST(HF, DF, 2, 0, false, false),
      INST(HF,  B, 4, 0, false, false),
      INST(HF,  W, 4, 0, false, false), /* 20 */
      INST(HF, HF, 4, 0, true,  false),
      INST(HF, HF, 4, 2, true,  false),
      INST(HF,  D, 4, 0, false, false),
      INST(HF,  F, 4, 0, false, false),
      INST(HF,  Q, 4, 0, false, false),
      INST(HF, DF, 4, 0, false, false),

      /* MOV from half-float source */
      INST( B, HF, 1, 0, false, false),
      INST( W, HF, 1, 0, false, false),
      INST( D, HF, 1, 0, true,  true),
      INST( D, HF, 1, 4, true,  true),  /* 30 */
      INST( F, HF, 1, 0, true,  false),
      INST( F, HF, 1, 4, true,  false),
      INST( Q, HF, 1, 0, false, false),
      INST(DF, HF, 1, 0, false, false),
      INST( B, HF, 2, 0, false, false),
      INST( W, HF, 2, 0, true,  true),
      INST( W, HF, 2, 2, false, false),
      INST( D, HF, 2, 0, false, false),
      INST( F, HF, 2, 0, true,  false),
      INST( B, HF, 4, 0, true,  true),  /* 40 */
      INST( B, HF, 4, 1, false, false),
      INST( W, HF, 4, 0, false, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      if (!devinfo.has_64bit_float &&
          (inst[i].dst_type == BRW_TYPE_DF ||
           inst[i].src_type == BRW_TYPE_DF))
         continue;

      if (!devinfo.has_64bit_int &&
          (inst[i].dst_type == BRW_TYPE_Q ||
           inst[i].dst_type == BRW_TYPE_UQ ||
           inst[i].src_type == BRW_TYPE_Q ||
           inst[i].src_type == BRW_TYPE_UQ))
         continue;

      brw_MOV(p, retype(g0, inst[i].dst_type), retype(g0, inst[i].src_type));

      brw_eu_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);
      brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, inst[i].dst_subnr);

      if (inst[i].src_type == BRW_TYPE_B) {
         brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
         brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_2);
         brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
      } else {
         brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
         brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
         brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
      }

      if (devinfo.verx10 >= 125) {
         EXPECT_EQ(inst[i].expected_result_gfx125, validate(p)) <<
            "Failing test is: " << i;
      } else {
         EXPECT_EQ(inst[i].expected_result_gfx9, validate(p)) <<
            "Failing test is: " << i;
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_source_indirect_addressing)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      unsigned dst_stride;
      bool dst_indirect;
      bool src0_indirect;
      bool expected_result;
      bool gfx125_expected_result;
   } inst[] = {
#define INST(dst_type, src0_type, src1_type,                              \
             dst_stride, dst_indirect, src0_indirect, expected_result,    \
             gfx125_expected_result)                                      \
      {                                                                   \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                              \
         dst_indirect,                                                    \
         src0_indirect,                                                   \
         expected_result,                                                 \
         gfx125_expected_result,                                          \
      }

      /* Source and dest are mixed float: indirect src addressing not allowed */
      INST(HF,  F,  F, 2, false, false, true,  true),
      INST(HF,  F,  F, 2, true,  false, true,  true),
      INST(HF,  F,  F, 2, false, true,  false, false),
      INST(HF,  F,  F, 2, true,  true,  false, false),
      INST( F, HF,  F, 1, false, false, true,  false),
      INST( F, HF,  F, 1, true,  false, true,  false),
      INST( F, HF,  F, 1, false, true,  false, false),
      INST( F, HF,  F, 1, true,  true,  false, false),

      INST(HF, HF,  F, 2, false, false, true,  false),
      INST(HF, HF,  F, 2, true,  false, true,  false),
      INST(HF, HF,  F, 2, false, true,  false, false),
      INST(HF, HF,  F, 2, true,  true,  false, false),
      INST( F,  F, HF, 1, false, false, true,  false),
      INST( F,  F, HF, 1, true,  false, true,  false),
      INST( F,  F, HF, 1, false, true,  false, false),
      INST( F,  F, HF, 1, true,  true,  false, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD(p, retype(g0, inst[i].dst_type),
                 retype(g0, inst[i].src0_type),
                 retype(g0, inst[i].src1_type));

      brw_eu_inst_set_dst_address_mode(&devinfo, last_inst, inst[i].dst_indirect);
      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);
      brw_eu_inst_set_src0_address_mode(&devinfo, last_inst, inst[i].src0_indirect);

      if (devinfo.verx10 >= 125) {
         EXPECT_EQ(inst[i].gfx125_expected_result, validate(p));
      } else {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_align1_simd16)
{
   static const struct {
      unsigned exec_size;
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      unsigned dst_stride;
      bool expected_result;
      bool gfx125_expected_result;
   } inst[] = {
#define INST(exec_size, dst_type, src0_type, src1_type,                   \
             dst_stride, expected_result, gfx125_expected_result)         \
      {                                                                   \
         BRW_EXECUTE_##exec_size,                                         \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                              \
         expected_result,                                                 \
         gfx125_expected_result,                                          \
      }

      /* No SIMD16 in mixed mode when destination is packed f16 */
      INST( 8, HF,  F, HF, 2, true,  false),
      INST(16, HF, HF,  F, 2, true,  false),
      INST(16, HF, HF,  F, 1, false, false),
      INST(16, HF,  F, HF, 1, false, false),

      /* No SIMD16 in mixed mode when destination is f32 */
      INST( 8,  F, HF,  F, 1, true,  false),
      INST( 8,  F,  F, HF, 1, true,  false),
      INST(16,  F, HF,  F, 1, false, false),
      INST(16,  F,  F, HF, 1, false, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD(p, retype(g0, inst[i].dst_type),
                 retype(g0, inst[i].src0_type),
                 retype(g0, inst[i].src1_type));

      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      if (devinfo.verx10 >= 125) {
         EXPECT_EQ(inst[i].gfx125_expected_result, validate(p));
      } else {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_align1_packed_fp16_dst_acc_read_offset_0)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      unsigned dst_stride;
      bool read_acc;
      unsigned subnr;
      bool expected_result_skl;
      bool expected_result_gfx125;
   } inst[] = {
#define INST(dst_type, src0_type, src1_type, dst_stride, read_acc, subnr,   \
             expected_result_skl, expected_result_gfx125)                   \
      {                                                                     \
         BRW_TYPE_##dst_type,                                               \
         BRW_TYPE_##src0_type,                                              \
         BRW_TYPE_##src1_type,                                              \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                \
         read_acc,                                                          \
         subnr,                                                             \
         expected_result_skl,                                               \
         expected_result_gfx125,                                            \
      }

      /* Destination is not packed */
      INST(HF, HF,  F, 2, true,  0, true, false),
      INST(HF, HF,  F, 2, true,  2, true, false),
      INST(HF, HF,  F, 2, true,  4, true, false),
      INST(HF, HF,  F, 2, true,  8, true, false),
      INST(HF, HF,  F, 2, true, 16, true, false),

      /* Destination is packed, we don't read acc */
      INST(HF, HF,  F, 1, false,  0, true, false),
      INST(HF, HF,  F, 1, false,  2, true, false),
      INST(HF, HF,  F, 1, false,  4, true, false),
      INST(HF, HF,  F, 1, false,  8, true, false),
      INST(HF, HF,  F, 1, false, 16, true, false),

      /* Destination is packed, we read acc */
      INST(HF, HF,  F, 1, true,  0, false, false),
      INST(HF, HF,  F, 1, true,  2, false, false),
      INST(HF, HF,  F, 1, true,  4, false, false),
      INST(HF, HF,  F, 1, true,  8, false, false),
      INST(HF, HF,  F, 1, true, 16, false, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD(p, retype(g0, inst[i].dst_type),
                 retype(inst[i].read_acc ? acc0 : g0, inst[i].src0_type),
                 retype(g0, inst[i].src1_type));

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_eu_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, inst[i].subnr);

      if (devinfo.verx10 >= 125)
         EXPECT_EQ(inst[i].expected_result_gfx125, validate(p));
      else
         EXPECT_EQ(inst[i].expected_result_skl, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_fp16_dest_with_acc)
{
   static const struct {
      unsigned exec_size;
      unsigned opcode;
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      unsigned dst_stride;
      bool read_acc;
      bool expected_result_skl;
      bool expected_result_gfx125;
   } inst[] = {
#define INST(exec_size, opcode, dst_type, src0_type, src1_type,           \
             dst_stride, read_acc,                                        \
             expected_result_skl, expected_result_gfx125)                 \
      {                                                                   \
         BRW_EXECUTE_##exec_size,                                         \
         BRW_OPCODE_##opcode,                                             \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                              \
         read_acc,                                                        \
         expected_result_skl,                                             \
         expected_result_gfx125,                                          \
      }

      /* Packed fp16 dest with implicit acc needs hstride=2 */
      INST(8, MAC, HF, HF,  F, 1, false, false, false),
      INST(8, MAC, HF, HF,  F, 2, false, true,  false),
      INST(8, MAC, HF,  F, HF, 1, false, false, false),
      INST(8, MAC, HF,  F, HF, 2, false, true,  false),

      /* Packed fp16 dest with explicit acc needs hstride=2 */
      INST(8, ADD, HF, HF,  F, 1, true,  false, false),
      INST(8, ADD, HF, HF,  F, 2, true,  true,  false),
      INST(8, ADD, HF,  F, HF, 1, true,  false, false),
      INST(8, ADD, HF,  F, HF, 2, true,  true,  false),

      /* If destination is not fp16, restriction doesn't apply */
      INST(8, MAC,  F, HF,  F, 1, false, true, false),
      INST(8, MAC,  F, HF,  F, 2, false, true, false),

      /* If there is no implicit/explicit acc, restriction doesn't apply */
      INST(8, ADD, HF, HF,  F, 1, false, true, false),
      INST(8, ADD, HF, HF,  F, 2, false, true, false),
      INST(8, ADD, HF,  F, HF, 1, false, true, false),
      INST(8, ADD, HF,  F, HF, 2, false, true, false),
      INST(8, ADD,  F, HF,  F, 1, false, true, false),
      INST(8, ADD,  F, HF,  F, 2, false, true, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      if (inst[i].opcode == BRW_OPCODE_MAC) {
         brw_MAC(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src0_type),
                    retype(g0, inst[i].src1_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_ADD);
         brw_ADD(p, retype(g0, inst[i].dst_type),
                    retype(inst[i].read_acc ? acc0: g0, inst[i].src0_type),
                    retype(g0, inst[i].src1_type));
      }

      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      if (devinfo.verx10 >= 125)
         EXPECT_EQ(inst[i].expected_result_gfx125, validate(p));
      else
         EXPECT_EQ(inst[i].expected_result_skl, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_align1_math_strided_fp16_inputs)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      unsigned dst_stride;
      unsigned src0_stride;
      unsigned src1_stride;
      bool expected_result;
      bool expected_result_gfx125;
   } inst[] = {
#define INST(dst_type, src0_type, src1_type,                              \
             dst_stride, src0_stride, src1_stride, expected_result,       \
             expected_result_125)                                         \
      {                                                                   \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                              \
         BRW_HORIZONTAL_STRIDE_##src0_stride,                             \
         BRW_HORIZONTAL_STRIDE_##src1_stride,                             \
         expected_result,                                                 \
         expected_result_125,                                             \
      }

      INST(HF, HF,  F, 2, 2, 1, true,  false),
      INST(HF,  F, HF, 2, 1, 2, true,  false),
      INST(HF,  F, HF, 1, 1, 2, true,  false),
      INST(HF,  F, HF, 2, 1, 1, false, false),
      INST(HF, HF,  F, 2, 1, 1, false, false),
      INST(HF, HF,  F, 1, 1, 1, false, false),
      INST(HF, HF,  F, 2, 1, 1, false, false),
      INST( F, HF,  F, 1, 1, 1, false, false),
      INST( F,  F, HF, 1, 1, 2, true,  false),
      INST( F, HF, HF, 1, 2, 1, false, false),
      INST( F, HF, HF, 1, 2, 2, true,  false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      gfx6_math(p, retype(g0, inst[i].dst_type),
                   BRW_MATH_FUNCTION_POW,
                   retype(g0, inst[i].src0_type),
                   retype(g0, inst[i].src1_type));

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
      brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
      brw_eu_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src0_stride);

      brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
      brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
      brw_eu_inst_set_src1_hstride(&devinfo, last_inst, inst[i].src1_stride);

      if (devinfo.verx10 >= 125)
         EXPECT_EQ(inst[i].expected_result_gfx125, validate(p));
      else
         EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_align1_packed_fp16_dst)
{
   static const struct {
      unsigned exec_size;
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      unsigned dst_stride;
      unsigned dst_subnr;
      bool expected_result_skl;
      bool expected_result_gfx125;
   } inst[] = {
#define INST(exec_size, dst_type, src0_type, src1_type, dst_stride, dst_subnr, \
             expected_result_skl, expected_result_gfx125)                      \
      {                                                                        \
         BRW_EXECUTE_##exec_size,                                              \
         BRW_TYPE_##dst_type,                                                  \
         BRW_TYPE_##src0_type,                                                 \
         BRW_TYPE_##src1_type,                                                 \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                   \
         dst_subnr,                                                            \
         expected_result_skl,                                                  \
         expected_result_gfx125                                                \
      }

      /* SIMD8 packed fp16 dst won't cross oword boundaries if region is
       * oword-aligned
       */
      INST( 8, HF, HF,  F, 1,  0, true,  false),
      INST( 8, HF, HF,  F, 1,  2, false, false),
      INST( 8, HF, HF,  F, 1,  4, false, false),
      INST( 8, HF, HF,  F, 1,  8, false, false),
      INST( 8, HF, HF,  F, 1, 16, true,  false),

      /* SIMD16 packed fp16 always crosses oword boundaries */
      INST(16, HF, HF,  F, 1,  0, false, false),
      INST(16, HF, HF,  F, 1,  2, false, false),
      INST(16, HF, HF,  F, 1,  4, false, false),
      INST(16, HF, HF,  F, 1,  8, false, false),
      INST(16, HF, HF,  F, 1, 16, false, false),

      /* If destination is not packed (or not fp16) we can cross oword
       * boundaries
       */
      INST( 8, HF, HF,  F, 2,  0, true, false),
      INST( 8,  F, HF,  F, 1,  0, true, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD(p, retype(g0, inst[i].dst_type),
                 retype(g0, inst[i].src0_type),
                 retype(g0, inst[i].src1_type));

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);
      brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, inst[i].dst_subnr);

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
      brw_eu_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
      brw_eu_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

      brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
      brw_eu_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
      brw_eu_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      if (devinfo.verx10 >= 125)
         EXPECT_EQ(inst[i].expected_result_gfx125, validate(p));
      else
         EXPECT_EQ(inst[i].expected_result_skl, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_align16_packed_data)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      unsigned src0_vstride;
      unsigned src1_vstride;
      bool expected_result;
   } inst[] = {
#define INST(dst_type, src0_type, src1_type,                              \
             src0_vstride, src1_vstride, expected_result)                 \
      {                                                                   \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         BRW_VERTICAL_STRIDE_##src0_vstride,                              \
         BRW_VERTICAL_STRIDE_##src1_vstride,                              \
         expected_result,                                                 \
      }

      /* We only test with F destination because there is a restriction
       * by which F->HF conversions need to be DWord aligned but Align16 also
       * requires that destination horizontal stride is 1.
       */
      INST(F,  F, HF, 4, 4, true),
      INST(F,  F, HF, 2, 4, false),
      INST(F,  F, HF, 4, 2, false),
      INST(F,  F, HF, 0, 4, false),
      INST(F,  F, HF, 4, 0, false),
      INST(F, HF,  F, 4, 4, true),
      INST(F, HF,  F, 4, 2, false),
      INST(F, HF,  F, 2, 4, false),
      INST(F, HF,  F, 0, 4, false),
      INST(F, HF,  F, 4, 0, false),

#undef INST
   };

   if (devinfo.ver >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD(p, retype(g0, inst[i].dst_type),
                 retype(g0, inst[i].src0_type),
                 retype(g0, inst[i].src1_type));

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src0_vstride);
      brw_eu_inst_set_src1_vstride(&devinfo, last_inst, inst[i].src1_vstride);

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_align16_no_simd16)
{
   static const struct {
      unsigned exec_size;
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      bool expected_result;
   } inst[] = {
#define INST(exec_size, dst_type, src0_type, src1_type, expected_result)  \
      {                                                                   \
         BRW_EXECUTE_##exec_size,                                         \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         expected_result,                                                 \
      }

      /* We only test with F destination because there is a restriction
       * by which F->HF conversions need to be DWord aligned but Align16 also
       * requires that destination horizontal stride is 1.
       */
      INST( 8,  F,  F, HF, true),
      INST( 8,  F, HF,  F, true),
      INST( 8,  F,  F, HF, true),
      INST(16,  F,  F, HF, false),
      INST(16,  F, HF,  F, false),
      INST(16,  F,  F, HF, false),

#undef INST
   };

   if (devinfo.ver >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD(p, retype(g0, inst[i].dst_type),
                 retype(g0, inst[i].src0_type),
                 retype(g0, inst[i].src1_type));

      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
      brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_align16_no_acc_read)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      bool read_acc;
      bool expected_result;
   } inst[] = {
#define INST(dst_type, src0_type, src1_type, read_acc, expected_result)   \
      {                                                                   \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         read_acc,                                                        \
         expected_result,                                                 \
      }

      /* We only test with F destination because there is a restriction
       * by which F->HF conversions need to be DWord aligned but Align16 also
       * requires that destination horizontal stride is 1.
       */
      INST( F,  F, HF, false, true),
      INST( F,  F, HF, true,  false),
      INST( F, HF,  F, false, true),
      INST( F, HF,  F, true,  false),

#undef INST
   };

   if (devinfo.ver >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD(p, retype(g0, inst[i].dst_type),
                 retype(inst[i].read_acc ? acc0 : g0, inst[i].src0_type),
                 retype(g0, inst[i].src1_type));

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
      brw_eu_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, mixed_float_align16_math_packed_format)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      unsigned src0_vstride;
      unsigned src1_vstride;
      bool expected_result;
   } inst[] = {
#define INST(dst_type, src0_type, src1_type,                              \
             src0_vstride, src1_vstride, expected_result)                 \
      {                                                                   \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         BRW_VERTICAL_STRIDE_##src0_vstride,                              \
         BRW_VERTICAL_STRIDE_##src1_vstride,                              \
         expected_result,                                                 \
      }

      /* We only test with F destination because there is a restriction
       * by which F->HF conversions need to be DWord aligned but Align16 also
       * requires that destination horizontal stride is 1.
       */
      INST( F, HF,  F, 4, 0, false),
      INST( F, HF, HF, 4, 4, true),
      INST( F,  F, HF, 4, 0, false),
      INST( F,  F, HF, 2, 4, false),
      INST( F,  F, HF, 4, 2, false),
      INST( F, HF, HF, 0, 4, false),

#undef INST
   };

   /* Align16 Math for mixed float mode is not supported in Gfx11+ */
   if (devinfo.ver >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      gfx6_math(p, retype(g0, inst[i].dst_type),
                   BRW_MATH_FUNCTION_POW,
                   retype(g0, inst[i].src0_type),
                   retype(g0, inst[i].src1_type));

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src0_vstride);
      brw_eu_inst_set_src1_vstride(&devinfo, last_inst, inst[i].src1_vstride);

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, vector_immediate_destination_alignment)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;
      unsigned subnr;
      unsigned exec_size;
      bool expected_result;
   } move[] = {
      { BRW_TYPE_F, BRW_TYPE_VF,  0, BRW_EXECUTE_4, true  },
      { BRW_TYPE_F, BRW_TYPE_VF, 16, BRW_EXECUTE_4, true  },
      { BRW_TYPE_F, BRW_TYPE_VF,  1, BRW_EXECUTE_4, false },

      { BRW_TYPE_W, BRW_TYPE_V,   0, BRW_EXECUTE_8, true  },
      { BRW_TYPE_W, BRW_TYPE_V,  16, BRW_EXECUTE_8, true  },
      { BRW_TYPE_W, BRW_TYPE_V,   1, BRW_EXECUTE_8, false },

      { BRW_TYPE_W, BRW_TYPE_UV,  0, BRW_EXECUTE_8, true  },
      { BRW_TYPE_W, BRW_TYPE_UV, 16, BRW_EXECUTE_8, true  },
      { BRW_TYPE_W, BRW_TYPE_UV,  1, BRW_EXECUTE_8, false },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(move); i++) {
      brw_MOV(p, retype(g0, move[i].dst_type), retype(zero, move[i].src_type));
      brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, move[i].subnr);
      brw_eu_inst_set_exec_size(&devinfo, last_inst, move[i].exec_size);

      EXPECT_EQ(move[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, vector_immediate_destination_stride)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;
      unsigned stride;
      bool expected_result;
   } move[] = {
      { BRW_TYPE_F, BRW_TYPE_VF, BRW_HORIZONTAL_STRIDE_1, true  },
      { BRW_TYPE_F, BRW_TYPE_VF, BRW_HORIZONTAL_STRIDE_2, false },
      { BRW_TYPE_D, BRW_TYPE_VF, BRW_HORIZONTAL_STRIDE_1, true  },
      { BRW_TYPE_D, BRW_TYPE_VF, BRW_HORIZONTAL_STRIDE_2, false },
      { BRW_TYPE_W, BRW_TYPE_VF, BRW_HORIZONTAL_STRIDE_2, true  },
      { BRW_TYPE_B, BRW_TYPE_VF, BRW_HORIZONTAL_STRIDE_4, true  },

      { BRW_TYPE_W, BRW_TYPE_V,  BRW_HORIZONTAL_STRIDE_1, true  },
      { BRW_TYPE_W, BRW_TYPE_V,  BRW_HORIZONTAL_STRIDE_2, false },
      { BRW_TYPE_W, BRW_TYPE_V,  BRW_HORIZONTAL_STRIDE_4, false },
      { BRW_TYPE_B, BRW_TYPE_V,  BRW_HORIZONTAL_STRIDE_2, true  },

      { BRW_TYPE_W, BRW_TYPE_UV, BRW_HORIZONTAL_STRIDE_1, true  },
      { BRW_TYPE_W, BRW_TYPE_UV, BRW_HORIZONTAL_STRIDE_2, false },
      { BRW_TYPE_W, BRW_TYPE_UV, BRW_HORIZONTAL_STRIDE_4, false },
      { BRW_TYPE_B, BRW_TYPE_UV, BRW_HORIZONTAL_STRIDE_2, true  },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(move); i++) {
      brw_MOV(p, retype(g0, move[i].dst_type), retype(zero, move[i].src_type));
      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, move[i].stride);

      EXPECT_EQ(move[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, qword_low_power_align1_regioning_restrictions)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      enum brw_reg_type dst_type;
      unsigned dst_subreg;
      unsigned dst_stride;

      enum brw_reg_type src_type;
      unsigned src_subreg;
      unsigned src_vstride;
      unsigned src_width;
      unsigned src_hstride;

      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst_type, dst_subreg, dst_stride, src_type,    \
             src_subreg, src_vstride, src_width, src_hstride, expected_result) \
      {                                                                        \
         BRW_OPCODE_##opcode,                                                  \
         BRW_EXECUTE_##exec_size,                                              \
         BRW_TYPE_##dst_type,                                                  \
         dst_subreg,                                                           \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                   \
         BRW_TYPE_##src_type,                                                  \
         src_subreg,                                                           \
         BRW_VERTICAL_STRIDE_##src_vstride,                                    \
         BRW_WIDTH_##src_width,                                                \
         BRW_HORIZONTAL_STRIDE_##src_hstride,                                  \
         expected_result,                                                      \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 4, DF, 0, 1, DF, 0, 4, 4, 1, true ),
      INST(MOV, 4, Q,  0, 1, Q,  0, 4, 4, 1, true ),
      INST(MOV, 4, UQ, 0, 1, UQ, 0, 4, 4, 1, true ),

      INST(MOV, 4, DF, 0, 1, F,  0, 8, 4, 2, true ),
      INST(MOV, 4, Q,  0, 1, D,  0, 8, 4, 2, true ),
      INST(MOV, 4, UQ, 0, 1, UD, 0, 8, 4, 2, true ),

      INST(MOV, 4, F,  0, 2, DF, 0, 4, 4, 1, true ),
      INST(MOV, 4, D,  0, 2, Q,  0, 4, 4, 1, true ),
      INST(MOV, 4, UD, 0, 2, UQ, 0, 4, 4, 1, true ),

      INST(MUL, 8, D,  0, 2, D,  0, 8, 4, 2, true ),
      INST(MUL, 8, UD, 0, 2, UD, 0, 8, 4, 2, true ),

      /* Something with subreg nrs */
      INST(MOV, 2, DF, 8, 1, DF, 8, 2, 2, 1, true ),
      INST(MOV, 2, Q,  8, 1, Q,  8, 2, 2, 1, true ),
      INST(MOV, 2, UQ, 8, 1, UQ, 8, 2, 2, 1, true ),

      INST(MUL, 2, D,  4, 2, D,  4, 4, 2, 2, true ),
      INST(MUL, 2, UD, 4, 2, UD, 4, 4, 2, 2, true ),

      /* The PRMs say that for CHV, BXT:
       *
       *    When source or destination datatype is 64b or operation is integer
       *    DWord multiply, regioning in Align1 must follow these rules:
       *
       *    1. Source and Destination horizontal stride must be aligned to the
       *       same qword.
       */
      INST(MOV, 4, DF, 0, 2, DF, 0, 4, 4, 1, false),
      INST(MOV, 4, Q,  0, 2, Q,  0, 4, 4, 1, false),
      INST(MOV, 4, UQ, 0, 2, UQ, 0, 4, 4, 1, false),

      INST(MOV, 4, DF, 0, 2, F,  0, 8, 4, 2, false),
      INST(MOV, 4, Q,  0, 2, D,  0, 8, 4, 2, false),
      INST(MOV, 4, UQ, 0, 2, UD, 0, 8, 4, 2, false),

      INST(MOV, 4, DF, 0, 2, F,  0, 4, 4, 1, false),
      INST(MOV, 4, Q,  0, 2, D,  0, 4, 4, 1, false),
      INST(MOV, 4, UQ, 0, 2, UD, 0, 4, 4, 1, false),

      INST(MUL, 4, D,  0, 2, D,  0, 4, 4, 1, false),
      INST(MUL, 4, UD, 0, 2, UD, 0, 4, 4, 1, false),

      INST(MUL, 4, D,  0, 1, D,  0, 8, 4, 2, false),
      INST(MUL, 4, UD, 0, 1, UD, 0, 8, 4, 2, false),

      /*    2. Regioning must ensure Src.Vstride = Src.Width * Src.Hstride. */
      INST(MOV, 4, DF, 0, 1, DF, 0, 0, 2, 1, false),
      INST(MOV, 4, Q,  0, 1, Q,  0, 0, 2, 1, false),
      INST(MOV, 4, UQ, 0, 1, UQ, 0, 0, 2, 1, false),

      INST(MOV, 4, DF, 0, 1, F,  0, 0, 2, 2, false),
      INST(MOV, 4, Q,  0, 1, D,  0, 0, 2, 2, false),
      INST(MOV, 4, UQ, 0, 1, UD, 0, 0, 2, 2, false),

      INST(MOV, 8, F,  0, 2, DF, 0, 0, 2, 1, false),
      INST(MOV, 8, D,  0, 2, Q,  0, 0, 2, 1, false),
      INST(MOV, 8, UD, 0, 2, UQ, 0, 0, 2, 1, false),

      INST(MUL, 8, D,  0, 2, D,  0, 0, 4, 2, false),
      INST(MUL, 8, UD, 0, 2, UD, 0, 0, 4, 2, false),

      INST(MUL, 8, D,  0, 2, D,  0, 0, 4, 2, false),
      INST(MUL, 8, UD, 0, 2, UD, 0, 0, 4, 2, false),

      /*    3. Source and Destination offset must be the same, except the case
       *       of scalar source.
       */
      INST(MOV, 2, DF, 8, 1, DF, 0, 2, 2, 1, false),
      INST(MOV, 2, Q,  8, 1, Q,  0, 2, 2, 1, false),
      INST(MOV, 2, UQ, 8, 1, UQ, 0, 2, 2, 1, false),

      INST(MOV, 2, DF, 0, 1, DF, 8, 2, 2, 1, false),
      INST(MOV, 2, Q,  0, 1, Q,  8, 2, 2, 1, false),
      INST(MOV, 2, UQ, 0, 1, UQ, 8, 2, 2, 1, false),

      INST(MUL, 4, D,  4, 2, D,  0, 4, 2, 2, false),
      INST(MUL, 4, UD, 4, 2, UD, 0, 4, 2, 2, false),

      INST(MUL, 4, D,  0, 2, D,  4, 4, 2, 2, false),
      INST(MUL, 4, UD, 0, 2, UD, 4, 4, 2, 2, false),

      INST(MOV, 2, DF, 8, 1, DF, 0, 0, 1, 0, true ),
      INST(MOV, 2, Q,  8, 1, Q,  0, 0, 1, 0, true ),
      INST(MOV, 2, UQ, 8, 1, UQ, 0, 0, 1, 0, true ),

      INST(MOV, 2, DF, 8, 1, F,  4, 0, 1, 0, true ),
      INST(MOV, 2, Q,  8, 1, D,  4, 0, 1, 0, true ),
      INST(MOV, 2, UQ, 8, 1, UD, 4, 0, 1, 0, true ),

      INST(MUL, 4, D,  4, 1, D,  0, 0, 1, 0, true ),
      INST(MUL, 4, UD, 4, 1, UD, 0, 0, 1, 0, true ),

      INST(MUL, 4, D,  0, 1, D,  4, 0, 1, 0, true ),
      INST(MUL, 4, UD, 0, 1, UD, 4, 0, 1, 0, true ),

#undef INST
   };

   /* NoDDChk/NoDDClr does not exist on Gfx12+ */
   if (devinfo.ver >= 12)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      if (!devinfo.has_64bit_float &&
          (inst[i].dst_type == BRW_TYPE_DF ||
           inst[i].src_type == BRW_TYPE_DF))
         continue;

      if (!devinfo.has_64bit_int &&
          (inst[i].dst_type == BRW_TYPE_Q ||
           inst[i].dst_type == BRW_TYPE_UQ ||
           inst[i].src_type == BRW_TYPE_Q ||
           inst[i].src_type == BRW_TYPE_UQ))
         continue;

      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_MUL);
         brw_MUL(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type),
                    retype(zero, inst[i].src_type));
      }
      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_eu_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, inst[i].dst_subreg);
      brw_eu_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, inst[i].src_subreg);

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src_vstride);
      brw_eu_inst_set_src0_width(&devinfo, last_inst, inst[i].src_width);
      brw_eu_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src_hstride);

      if (intel_device_info_is_9lp(&devinfo)) {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      } else {
         EXPECT_TRUE(validate(p));
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, qword_low_power_no_indirect_addressing)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      enum brw_reg_type dst_type;
      bool dst_is_indirect;
      unsigned dst_stride;

      enum brw_reg_type src_type;
      bool src_is_indirect;
      unsigned src_vstride;
      unsigned src_width;
      unsigned src_hstride;

      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst_type, dst_is_indirect, dst_stride,       \
             src_type, src_is_indirect, src_vstride, src_width, src_hstride, \
             expected_result)                                                \
      {                                                                      \
         BRW_OPCODE_##opcode,                                                \
         BRW_EXECUTE_##exec_size,                                            \
         BRW_TYPE_##dst_type,                                                \
         dst_is_indirect,                                                    \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                 \
         BRW_TYPE_##src_type,                                                \
         src_is_indirect,                                                    \
         BRW_VERTICAL_STRIDE_##src_vstride,                                  \
         BRW_WIDTH_##src_width,                                              \
         BRW_HORIZONTAL_STRIDE_##src_hstride,                                \
         expected_result,                                                    \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 4, DF, 0, 1, DF, 0, 4, 4, 1, true ),
      INST(MOV, 4, Q,  0, 1, Q,  0, 4, 4, 1, true ),
      INST(MOV, 4, UQ, 0, 1, UQ, 0, 4, 4, 1, true ),

      INST(MUL, 8, D,  0, 2, D,  0, 8, 4, 2, true ),
      INST(MUL, 8, UD, 0, 2, UD, 0, 8, 4, 2, true ),

      INST(MOV, 4, F,  1, 1, F,  0, 4, 4, 1, true ),
      INST(MOV, 4, F,  0, 1, F,  1, 4, 4, 1, true ),
      INST(MOV, 4, F,  1, 1, F,  1, 4, 4, 1, true ),

      /* The PRMs say that for CHV, BXT:
       *
       *    When source or destination datatype is 64b or operation is integer
       *    DWord multiply, indirect addressing must not be used.
       */
      INST(MOV, 4, DF, 1, 1, DF, 0, 4, 4, 1, false),
      INST(MOV, 4, Q,  1, 1, Q,  0, 4, 4, 1, false),
      INST(MOV, 4, UQ, 1, 1, UQ, 0, 4, 4, 1, false),

      INST(MOV, 4, DF, 0, 1, DF, 1, 4, 4, 1, false),
      INST(MOV, 4, Q,  0, 1, Q,  1, 4, 4, 1, false),
      INST(MOV, 4, UQ, 0, 1, UQ, 1, 4, 4, 1, false),

      INST(MOV, 4, DF, 1, 1, F,  0, 8, 4, 2, false),
      INST(MOV, 4, Q,  1, 1, D,  0, 8, 4, 2, false),
      INST(MOV, 4, UQ, 1, 1, UD, 0, 8, 4, 2, false),

      INST(MOV, 4, DF, 0, 1, F,  1, 8, 4, 2, false),
      INST(MOV, 4, Q,  0, 1, D,  1, 8, 4, 2, false),
      INST(MOV, 4, UQ, 0, 1, UD, 1, 8, 4, 2, false),

      INST(MOV, 4, F,  1, 2, DF, 0, 4, 4, 1, false),
      INST(MOV, 4, D,  1, 2, Q,  0, 4, 4, 1, false),
      INST(MOV, 4, UD, 1, 2, UQ, 0, 4, 4, 1, false),

      INST(MOV, 4, F,  0, 2, DF, 1, 4, 4, 1, false),
      INST(MOV, 4, D,  0, 2, Q,  1, 4, 4, 1, false),
      INST(MOV, 4, UD, 0, 2, UQ, 1, 4, 4, 1, false),

      INST(MUL, 8, D,  1, 2, D,  0, 8, 4, 2, false),
      INST(MUL, 8, UD, 1, 2, UD, 0, 8, 4, 2, false),

      INST(MUL, 8, D,  0, 2, D,  1, 8, 4, 2, false),
      INST(MUL, 8, UD, 0, 2, UD, 1, 8, 4, 2, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      if (!devinfo.has_64bit_float &&
          (inst[i].dst_type == BRW_TYPE_DF ||
           inst[i].src_type == BRW_TYPE_DF))
         continue;

      if (!devinfo.has_64bit_int &&
          (inst[i].dst_type == BRW_TYPE_Q ||
           inst[i].dst_type == BRW_TYPE_UQ ||
           inst[i].src_type == BRW_TYPE_Q ||
           inst[i].src_type == BRW_TYPE_UQ))
         continue;

      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_MUL);
         brw_MUL(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type),
                    retype(zero, inst[i].src_type));
      }
      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_eu_inst_set_dst_address_mode(&devinfo, last_inst, inst[i].dst_is_indirect);
      brw_eu_inst_set_src0_address_mode(&devinfo, last_inst, inst[i].src_is_indirect);

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src_vstride);
      brw_eu_inst_set_src0_width(&devinfo, last_inst, inst[i].src_width);
      brw_eu_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src_hstride);

      if (intel_device_info_is_9lp(&devinfo)) {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      } else {
         char *error = nullptr;
         EXPECT_TRUE(validate(p, &error))
            << "Test index = " << i << " failed to validate: " << error;
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, qword_low_power_no_64bit_arf)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      struct brw_reg dst;
      enum brw_reg_type dst_type;
      unsigned dst_stride;

      struct brw_reg src;
      enum brw_reg_type src_type;
      unsigned src_vstride;
      unsigned src_width;
      unsigned src_hstride;

      bool acc_wr;
      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst, dst_type, dst_stride,        \
             src, src_type, src_vstride, src_width, src_hstride,  \
             acc_wr, expected_result)                             \
      {                                                           \
         BRW_OPCODE_##opcode,                                     \
         BRW_EXECUTE_##exec_size,                                 \
         dst,                                                     \
         BRW_TYPE_##dst_type,                                     \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                      \
         src,                                                     \
         BRW_TYPE_##src_type,                                     \
         BRW_VERTICAL_STRIDE_##src_vstride,                       \
         BRW_WIDTH_##src_width,                                   \
         BRW_HORIZONTAL_STRIDE_##src_hstride,                     \
         acc_wr,                                                  \
         expected_result,                                         \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 4, g0,   DF, 1, g0,   F,  4, 2, 2, 0, true ),
      INST(MOV, 4, g0,   F,  2, g0,   DF, 4, 4, 1, 0, true ),

      INST(MOV, 4, g0,   Q,  1, g0,   D,  4, 2, 2, 0, true ),
      INST(MOV, 4, g0,   D,  2, g0,   Q,  4, 4, 1, 0, true ),

      INST(MOV, 4, g0,   UQ, 1, g0,   UD, 4, 2, 2, 0, true ),
      INST(MOV, 4, g0,   UD, 2, g0,   UQ, 4, 4, 1, 0, true ),

      INST(MOV, 4, null, F,  1, g0,   F,  4, 4, 1, 0, true ),
      INST(MOV, 4, acc0, F,  1, g0,   F,  4, 4, 1, 0, true ),
      INST(MOV, 4, g0,   F,  1, acc0, F,  4, 4, 1, 0, true ),

      INST(MOV, 4, null, D,  1, g0,   D,  4, 4, 1, 0, true ),
      INST(MOV, 4, acc0, D,  1, g0,   D,  4, 4, 1, 0, true ),
      INST(MOV, 4, g0,   D,  1, acc0, D,  4, 4, 1, 0, true ),

      INST(MOV, 4, null, UD, 1, g0,   UD, 4, 4, 1, 0, true ),
      INST(MOV, 4, acc0, UD, 1, g0,   UD, 4, 4, 1, 0, true ),
      INST(MOV, 4, g0,   UD, 1, acc0, UD, 4, 4, 1, 0, true ),

      INST(MUL, 4, g0,   D,  2, g0,   D,  4, 2, 2, 0, true ),
      INST(MUL, 4, g0,   UD, 2, g0,   UD, 4, 2, 2, 0, true ),

      /* The PRMs say that for CHV, BXT:
       *
       *    ARF registers must never be used with 64b datatype or when
       *    operation is integer DWord multiply.
       */
      INST(MOV, 4, acc0, DF, 1, g0,   F,  4, 2, 2, 0, false),
      INST(MOV, 4, g0,   DF, 1, acc0, F,  4, 2, 2, 0, false),

      INST(MOV, 4, acc0, Q,  1, g0,   D,  4, 2, 2, 0, false),
      INST(MOV, 4, g0,   Q,  1, acc0, D,  4, 2, 2, 0, false),

      INST(MOV, 4, acc0, UQ, 1, g0,   UD, 4, 2, 2, 0, false),
      INST(MOV, 4, g0,   UQ, 1, acc0, UD, 4, 2, 2, 0, false),

      INST(MOV, 4, acc0, F,  2, g0,   DF, 4, 4, 1, 0, false),
      INST(MOV, 4, g0,   F,  2, acc0, DF, 4, 4, 1, 0, false),

      INST(MOV, 4, acc0, D,  2, g0,   Q,  4, 4, 1, 0, false),
      INST(MOV, 4, g0,   D,  2, acc0, Q,  4, 4, 1, 0, false),

      INST(MOV, 4, acc0, UD, 2, g0,   UQ, 4, 4, 1, 0, false),
      INST(MOV, 4, g0,   UD, 2, acc0, UQ, 4, 4, 1, 0, false),

      INST(MUL, 4, acc0, D,  2, g0,   D,  4, 2, 2, 0, false),
      INST(MUL, 4, acc0, UD, 2, g0,   UD, 4, 2, 2, 0, false),
      /* MUL cannot have integer accumulator sources, so don't test that */

      /* We assume that the restriction does not apply to the null register */
      INST(MOV, 4, null, DF, 1, g0,   F,  4, 2, 2, 0, true ),
      INST(MOV, 4, null, Q,  1, g0,   D,  4, 2, 2, 0, true ),
      INST(MOV, 4, null, UQ, 1, g0,   UD, 4, 2, 2, 0, true ),

      /* Check implicit accumulator write control */
      INST(MOV, 4, null, DF, 1, g0,   F,  4, 2, 2, 1, false),
      INST(MUL, 4, null, DF, 1, g0,   F,  4, 2, 2, 1, false),

#undef INST
   };

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      if (!devinfo.has_64bit_float &&
          (inst[i].dst_type == BRW_TYPE_DF ||
           inst[i].src_type == BRW_TYPE_DF))
         continue;

      if (!devinfo.has_64bit_int &&
          (inst[i].dst_type == BRW_TYPE_Q ||
           inst[i].dst_type == BRW_TYPE_UQ ||
           inst[i].src_type == BRW_TYPE_Q ||
           inst[i].src_type == BRW_TYPE_UQ))
         continue;

      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(inst[i].dst, inst[i].dst_type),
                    retype(inst[i].src, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_MUL);
         brw_MUL(p, retype(inst[i].dst, inst[i].dst_type),
                    retype(inst[i].src, inst[i].src_type),
                    retype(zero, inst[i].src_type));
         brw_eu_inst_set_opcode(&isa, last_inst, inst[i].opcode);
      }
      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);
      if (devinfo.ver < 20)
         brw_eu_inst_set_acc_wr_control(&devinfo, last_inst, inst[i].acc_wr);

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src_vstride);
      brw_eu_inst_set_src0_width(&devinfo, last_inst, inst[i].src_width);
      brw_eu_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src_hstride);

      if (intel_device_info_is_9lp(&devinfo)) {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      } else {
         EXPECT_TRUE(validate(p));
      }

      clear_instructions(p);
   }

   if (!devinfo.has_64bit_float)
      return;

   /* MAC implicitly reads the accumulator */
   brw_MAC(p, retype(g0, BRW_TYPE_DF),
              retype(stride(g0, 4, 4, 1), BRW_TYPE_DF),
              retype(stride(g0, 4, 4, 1), BRW_TYPE_DF));
   if (intel_device_info_is_9lp(&devinfo)) {
      EXPECT_FALSE(validate(p));
   } else {
      EXPECT_TRUE(validate(p));
   }
}

TEST_P(validation_test, align16_64_bit_integer)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;

      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst_type, src_type, expected_result)  \
      {                                                               \
         BRW_OPCODE_##opcode,                                         \
         BRW_EXECUTE_##exec_size,                                     \
         BRW_TYPE_##dst_type,                                         \
         BRW_TYPE_##src_type,                                         \
         expected_result,                                             \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 2, Q,  D,  true ),
      INST(MOV, 2, UQ, UD, true ),
      INST(MOV, 2, DF, F,  true ),

      INST(ADD, 2, Q,  D,  true ),
      INST(ADD, 2, UQ, UD, true ),
      INST(ADD, 2, DF, F,  true ),

      /* The PRMs say that for BDW, SKL:
       *
       *    If Align16 is required for an operation with QW destination and non-QW
       *    source datatypes, the execution size cannot exceed 2.
       */

      INST(MOV, 4, Q,  D,  false),
      INST(MOV, 4, UQ, UD, false),
      INST(MOV, 4, DF, F,  false),

      INST(ADD, 4, Q,  D,  false),
      INST(ADD, 4, UQ, UD, false),
      INST(ADD, 4, DF, F,  false),

#undef INST
   };

   /* Align16 does not exist on Gfx11+ */
   if (devinfo.ver >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_ADD);
         brw_ADD(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type),
                    retype(g0, inst[i].src_type));
      }
      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, qword_low_power_no_depctrl)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      enum brw_reg_type dst_type;
      unsigned dst_stride;

      enum brw_reg_type src_type;
      unsigned src_vstride;
      unsigned src_width;
      unsigned src_hstride;

      bool no_dd_check;
      bool no_dd_clear;

      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst_type, dst_stride,        \
             src_type, src_vstride, src_width, src_hstride,  \
             no_dd_check, no_dd_clear, expected_result)      \
      {                                                      \
         BRW_OPCODE_##opcode,                                \
         BRW_EXECUTE_##exec_size,                            \
         BRW_TYPE_##dst_type,                                \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                 \
         BRW_TYPE_##src_type,                                \
         BRW_VERTICAL_STRIDE_##src_vstride,                  \
         BRW_WIDTH_##src_width,                              \
         BRW_HORIZONTAL_STRIDE_##src_hstride,                \
         no_dd_check,                                        \
         no_dd_clear,                                        \
         expected_result,                                    \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 4, DF, 1, F,  8, 4, 2, 0, 0, true ),
      INST(MOV, 4, Q,  1, D,  8, 4, 2, 0, 0, true ),
      INST(MOV, 4, UQ, 1, UD, 8, 4, 2, 0, 0, true ),

      INST(MOV, 4, F,  2, DF, 4, 4, 1, 0, 0, true ),
      INST(MOV, 4, D,  2, Q,  4, 4, 1, 0, 0, true ),
      INST(MOV, 4, UD, 2, UQ, 4, 4, 1, 0, 0, true ),

      INST(MUL, 8, D,  2, D,  8, 4, 2, 0, 0, true ),
      INST(MUL, 8, UD, 2, UD, 8, 4, 2, 0, 0, true ),

      INST(MOV, 4, F,  1, F,  4, 4, 1, 1, 1, true ),

      /* The PRMs say that for CHV, BXT:
       *
       *    When source or destination datatype is 64b or operation is integer
       *    DWord multiply, DepCtrl must not be used.
       */
      INST(MOV, 4, DF, 1, F,  8, 4, 2, 1, 0, false),
      INST(MOV, 4, Q,  1, D,  8, 4, 2, 1, 0, false),
      INST(MOV, 4, UQ, 1, UD, 8, 4, 2, 1, 0, false),

      INST(MOV, 4, F,  2, DF, 4, 4, 1, 1, 0, false),
      INST(MOV, 4, D,  2, Q,  4, 4, 1, 1, 0, false),
      INST(MOV, 4, UD, 2, UQ, 4, 4, 1, 1, 0, false),

      INST(MOV, 4, DF, 1, F,  8, 4, 2, 0, 1, false),
      INST(MOV, 4, Q,  1, D,  8, 4, 2, 0, 1, false),
      INST(MOV, 4, UQ, 1, UD, 8, 4, 2, 0, 1, false),

      INST(MOV, 4, F,  2, DF, 4, 4, 1, 0, 1, false),
      INST(MOV, 4, D,  2, Q,  4, 4, 1, 0, 1, false),
      INST(MOV, 4, UD, 2, UQ, 4, 4, 1, 0, 1, false),

      INST(MUL, 8, D,  2, D,  8, 4, 2, 1, 0, false),
      INST(MUL, 8, UD, 2, UD, 8, 4, 2, 1, 0, false),

      INST(MUL, 8, D,  2, D,  8, 4, 2, 0, 1, false),
      INST(MUL, 8, UD, 2, UD, 8, 4, 2, 0, 1, false),

#undef INST
   };

   /* NoDDChk/NoDDClr does not exist on Gfx12+ */
   if (devinfo.ver >= 12)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      if (!devinfo.has_64bit_float &&
          (inst[i].dst_type == BRW_TYPE_DF ||
           inst[i].src_type == BRW_TYPE_DF))
         continue;

      if (!devinfo.has_64bit_int &&
          (inst[i].dst_type == BRW_TYPE_Q ||
           inst[i].dst_type == BRW_TYPE_UQ ||
           inst[i].src_type == BRW_TYPE_Q ||
           inst[i].src_type == BRW_TYPE_UQ))
         continue;

      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_MUL);
         brw_MUL(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type),
                    retype(zero, inst[i].src_type));
      }
      brw_eu_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_eu_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src_vstride);
      brw_eu_inst_set_src0_width(&devinfo, last_inst, inst[i].src_width);
      brw_eu_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src_hstride);

      brw_eu_inst_set_no_dd_check(&devinfo, last_inst, inst[i].no_dd_check);
      brw_eu_inst_set_no_dd_clear(&devinfo, last_inst, inst[i].no_dd_clear);

      if (intel_device_info_is_9lp(&devinfo)) {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      } else {
         EXPECT_TRUE(validate(p));
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, gfx11_no_byte_src_1_2)
{
   static const struct {
      enum opcode opcode;
      unsigned access_mode;

      enum brw_reg_type dst_type;
      struct {
         enum brw_reg_type type;
         unsigned vstride;
         unsigned width;
         unsigned hstride;
      } srcs[3];

      int  gfx_ver;
      bool expected_result;
   } inst[] = {
#define INST(opcode, access_mode, dst_type,                             \
             src0_type, src0_vstride, src0_width, src0_hstride,         \
             src1_type, src1_vstride, src1_width, src1_hstride,         \
             src2_type,                                                 \
             gfx_ver, expected_result)                                  \
      {                                                                 \
         BRW_OPCODE_##opcode,                                           \
         BRW_ALIGN_##access_mode,                                       \
         BRW_TYPE_##dst_type,                                           \
         {                                                              \
            {                                                           \
               BRW_TYPE_##src0_type,                                    \
               BRW_VERTICAL_STRIDE_##src0_vstride,                      \
               BRW_WIDTH_##src0_width,                                  \
               BRW_HORIZONTAL_STRIDE_##src0_hstride,                    \
            },                                                          \
            {                                                           \
               BRW_TYPE_##src1_type,                                    \
               BRW_VERTICAL_STRIDE_##src1_vstride,                      \
               BRW_WIDTH_##src1_width,                                  \
               BRW_HORIZONTAL_STRIDE_##src1_hstride,                    \
            },                                                          \
            {                                                           \
               BRW_TYPE_##src2_type,                                    \
            },                                                          \
         },                                                             \
         gfx_ver,                                                       \
         expected_result,                                               \
      }

      /* Passes on < 11 */
      INST(MOV, 16,  F, B, 2, 4, 0, UD, 0, 4, 0,  D,  8, true ),
      INST(ADD, 16, UD, F, 0, 4, 0, UB, 0, 1, 0,  D,  7, true ),
      INST(MAD, 16,  D, B, 0, 4, 0, UB, 0, 1, 0,  B, 10, true ),

      /* Fails on 11+ */
      INST(MAD,  1, UB, W, 1, 1, 0,  D, 0, 4, 0,  B, 11, false ),
      INST(MAD,  1, UB, W, 1, 1, 1, UB, 1, 1, 0,  W, 11, false ),
      INST(ADD,  1,  W, W, 1, 4, 1,  B, 1, 1, 0,  D, 11, false ),

      /* Passes on 11+ */
      INST(MOV,  1,  W, B, 8, 8, 1,  D, 8, 8, 1,  D, 11, true ),
      INST(ADD,  1, UD, B, 8, 8, 1,  W, 8, 8, 1,  D, 11, true ),
      INST(MAD,  1,  B, B, 0, 1, 0,  D, 0, 4, 0,  W, 11, true ),

#undef INST
   };


   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      /* Skip instruction not meant for this gfx_ver. */
      if (devinfo.ver != inst[i].gfx_ver)
         continue;

      brw_push_insn_state(p);

      brw_set_default_exec_size(p, BRW_EXECUTE_8);
      brw_set_default_access_mode(p, inst[i].access_mode);

      switch (inst[i].opcode) {
      case BRW_OPCODE_MOV:
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].srcs[0].type));
         brw_eu_inst_set_src0_vstride(&devinfo, last_inst, inst[i].srcs[0].vstride);
         brw_eu_inst_set_src0_hstride(&devinfo, last_inst, inst[i].srcs[0].hstride);
         break;
      case BRW_OPCODE_ADD:
         brw_ADD(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].srcs[0].type),
                    retype(g0, inst[i].srcs[1].type));
         brw_eu_inst_set_src0_vstride(&devinfo, last_inst, inst[i].srcs[0].vstride);
         brw_eu_inst_set_src0_width(&devinfo, last_inst, inst[i].srcs[0].width);
         brw_eu_inst_set_src0_hstride(&devinfo, last_inst, inst[i].srcs[0].hstride);
         brw_eu_inst_set_src1_vstride(&devinfo, last_inst, inst[i].srcs[1].vstride);
         brw_eu_inst_set_src1_width(&devinfo, last_inst, inst[i].srcs[1].width);
         brw_eu_inst_set_src1_hstride(&devinfo, last_inst, inst[i].srcs[1].hstride);
         break;
      case BRW_OPCODE_MAD:
         brw_MAD(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].srcs[0].type),
                    retype(g0, inst[i].srcs[1].type),
                    retype(g0, inst[i].srcs[2].type));
         brw_eu_inst_set_3src_a1_src0_vstride(&devinfo, last_inst, inst[i].srcs[0].vstride);
         brw_eu_inst_set_3src_a1_src0_hstride(&devinfo, last_inst, inst[i].srcs[0].hstride);
         brw_eu_inst_set_3src_a1_src1_vstride(&devinfo, last_inst, inst[i].srcs[0].vstride);
         brw_eu_inst_set_3src_a1_src1_hstride(&devinfo, last_inst, inst[i].srcs[0].hstride);
         break;
      default:
         UNREACHABLE("invalid opcode");
      }

      brw_eu_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

      brw_eu_inst_set_src0_width(&devinfo, last_inst, inst[i].srcs[0].width);
      brw_eu_inst_set_src1_width(&devinfo, last_inst, inst[i].srcs[1].width);

      brw_pop_insn_state(p);

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, add3_source_types)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src0_type;
      enum brw_reg_type src1_type;
      enum brw_reg_type src2_type;
      bool expected_result;
   } inst[] = {
#define INST(dst_type, src0_type, src1_type, src2_type, expected_result)  \
      {                                                                   \
         BRW_TYPE_##dst_type,                                             \
         BRW_TYPE_##src0_type,                                            \
         BRW_TYPE_##src1_type,                                            \
         BRW_TYPE_##src2_type,                                            \
         expected_result,                                                 \
      }

      INST( F,  F,  F,  F, false),
      INST(HF, HF, HF, HF, false),
      INST( B,  B,  B,  B, false),
      INST(UB, UB, UB, UB, false),

      INST( W,  W,  W,  W, true),
      INST(UW, UW, UW, UW, true),
      INST( D,  D,  D,  D, true),
      INST(UD, UD, UD, UD, true),

      INST( W,  D,  W,  W, true),
      INST(UW, UW, UD, UW, true),
      INST( D,  D,  W,  D, true),
      INST(UD, UD, UD, UW, true),
#undef INST
   };


   if (devinfo.verx10 < 125)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD3(p,
               retype(g0, inst[i].dst_type),
               retype(g0, inst[i].src0_type),
               retype(g0, inst[i].src1_type),
               retype(g0, inst[i].src2_type));

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, add3_immediate_types)
{
   static const struct {
      enum brw_reg_type reg_type;
      enum brw_reg_type imm_type;
      unsigned imm_src;
      bool expected_result;
   } inst[] = {
#define INST(reg_type, imm_type, imm_src, expected_result)  \
      {                                                     \
         BRW_TYPE_##reg_type,                               \
         BRW_TYPE_##imm_type,                               \
         imm_src,                                           \
         expected_result,                                   \
      }

      INST( W,  W,  0, true),
      INST( W,  W,  2, true),
      INST(UW, UW,  0, true),
      INST(UW, UW,  2, true),
      INST( D,  W,  0, true),
      INST(UD,  W,  2, true),
      INST( D, UW,  0, true),
      INST(UW, UW,  2, true),

      INST( W,  D,  0, false),
      INST( W,  D,  2, false),
      INST(UW, UD,  0, false),
      INST(UW, UD,  2, false),
      INST( D,  D,  0, false),
      INST(UD,  D,  2, false),
      INST( D, UD,  0, false),
      INST(UW, UD,  2, false),
#undef INST
   };


   if (devinfo.verx10 < 125)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(inst); i++) {
      brw_ADD3(p,
               retype(g0, inst[i].reg_type),
               inst[i].imm_src == 0 ? retype(brw_imm_d(0x1234), inst[i].imm_type)
                                    : retype(g0, inst[i].reg_type),
               retype(g0, inst[i].reg_type),
               inst[i].imm_src == 2 ? retype(brw_imm_d(0x2143), inst[i].imm_type)
                                    : retype(g0, inst[i].reg_type));

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, dpas_sdepth)
{
   if (!devinfo.has_systolic)
      return;

   static const enum gfx12_systolic_depth depth[] = {
      BRW_SYSTOLIC_DEPTH_16,
      BRW_SYSTOLIC_DEPTH_2,
      BRW_SYSTOLIC_DEPTH_4,
      BRW_SYSTOLIC_DEPTH_8,
   };

   brw_set_default_exec_size(p, devinfo.ver >= 20 ? BRW_EXECUTE_16
                                                  : BRW_EXECUTE_8);

   for (unsigned i = 0; i < ARRAY_SIZE(depth); i++) {
      brw_DPAS(p,
               depth[i],
               8,
               retype(brw_vec8_grf(0, 0), BRW_TYPE_F),
               null,
               retype(brw_vec8_grf(16, 0), BRW_TYPE_HF),
               retype(brw_vec8_grf(32, 0), BRW_TYPE_HF));

      const bool expected_result = depth[i] == BRW_SYSTOLIC_DEPTH_8;

      EXPECT_EQ(expected_result, validate(p)) <<
         "Encoded systolic depth value is: " << depth[i];

      clear_instructions(p);
   }
}

TEST_P(validation_test, dpas_exec_size)
{
   if (!devinfo.has_systolic)
      return;

   static const enum brw_execution_size test_vectors[] = {
      BRW_EXECUTE_1,
      BRW_EXECUTE_2,
      BRW_EXECUTE_4,
      BRW_EXECUTE_8,
      BRW_EXECUTE_16,
      BRW_EXECUTE_32,
   };

   const brw_execution_size valid_exec_size =
      devinfo.ver >= 20 ? BRW_EXECUTE_16 : BRW_EXECUTE_8;

   for (unsigned i = 0; i < ARRAY_SIZE(test_vectors); i++) {
      brw_set_default_exec_size(p, test_vectors[i]);

      brw_DPAS(p,
               BRW_SYSTOLIC_DEPTH_8,
               8,
               retype(brw_vec8_grf(0, 0), BRW_TYPE_F),
               null,
               retype(brw_vec8_grf(16, 0), BRW_TYPE_HF),
               retype(brw_vec8_grf(32, 0), BRW_TYPE_HF));

      const bool expected_result = test_vectors[i] == valid_exec_size;

      EXPECT_EQ(expected_result, validate(p)) <<
         "Exec size = " << (1u << test_vectors[i]);

      clear_instructions(p);
   }

   brw_set_default_exec_size(p, BRW_EXECUTE_8);
}

TEST_P(validation_test, dpas_sub_byte_precision)
{
   if (!devinfo.has_systolic)
      return;

   static const struct {
      brw_reg_type dst_type;
      brw_reg_type src0_type;
      brw_reg_type src1_type;
      enum gfx12_sub_byte_precision src1_prec;
      brw_reg_type src2_type;
      enum gfx12_sub_byte_precision src2_prec;
      bool expected_result;
   } test_vectors[] = {
      {
         BRW_TYPE_F,
         BRW_TYPE_F,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_NONE,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_NONE,
         true,
      },
      {
         BRW_TYPE_F,
         BRW_TYPE_F,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_NONE,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_4BIT,
         false,
      },
      {
         BRW_TYPE_F,
         BRW_TYPE_F,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_NONE,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_2BIT,
         false,
      },
      {
         BRW_TYPE_F,
         BRW_TYPE_F,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_4BIT,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_NONE,
         false,
      },
      {
         BRW_TYPE_F,
         BRW_TYPE_F,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_2BIT,
         BRW_TYPE_HF, BRW_SUB_BYTE_PRECISION_NONE,
         false,
      },

      {
         BRW_TYPE_UD,
         BRW_TYPE_UD,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_NONE,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_NONE,
         true,
      },
      {
         BRW_TYPE_UD,
         BRW_TYPE_UD,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_NONE,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_4BIT,
         true,
      },
      {
         BRW_TYPE_UD,
         BRW_TYPE_UD,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_NONE,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_2BIT,
         true,
      },
      {
         BRW_TYPE_UD,
         BRW_TYPE_UD,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_NONE,
         BRW_TYPE_UB, (enum gfx12_sub_byte_precision) 3,
         false,
      },
      {
         BRW_TYPE_UD,
         BRW_TYPE_UD,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_4BIT,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_NONE,
         true,
      },
      {
         BRW_TYPE_UD,
         BRW_TYPE_UD,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_2BIT,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_NONE,
         true,
      },
      {
         BRW_TYPE_UD,
         BRW_TYPE_UD,
         BRW_TYPE_UB, (enum gfx12_sub_byte_precision) 3,
         BRW_TYPE_UB, BRW_SUB_BYTE_PRECISION_NONE,
         false,
      },
   };

   brw_set_default_exec_size(p, devinfo.ver >= 20 ? BRW_EXECUTE_16
                                                  : BRW_EXECUTE_8);

   for (unsigned i = 0; i < ARRAY_SIZE(test_vectors); i++) {
      brw_eu_inst *inst =
         brw_DPAS(p,
                  BRW_SYSTOLIC_DEPTH_8,
                  8,
                  retype(brw_vec8_grf(0, 0), test_vectors[i].dst_type),
                  retype(brw_vec8_grf(16, 0), test_vectors[i].src0_type),
                  retype(brw_vec8_grf(32, 0), test_vectors[i].src1_type),
                  retype(brw_vec8_grf(48, 0), test_vectors[i].src2_type));

      brw_eu_inst_set_dpas_3src_src1_subbyte(&devinfo, inst,
                                          test_vectors[i].src1_prec);
      brw_eu_inst_set_dpas_3src_src2_subbyte(&devinfo, inst,
                                          test_vectors[i].src2_prec);

      EXPECT_EQ(test_vectors[i].expected_result, validate(p)) <<
         "test vector index = " << i;

      clear_instructions(p);
   }
}

TEST_P(validation_test, dpas_types)
{
   if (!devinfo.has_systolic)
      return;

   if (devinfo.ver >= 20)
      assert(devinfo.has_bfloat16);

#define TV(a, b, c, d, r) \
   { BRW_TYPE_ ## a, BRW_TYPE_ ## b, BRW_TYPE_ ## c, BRW_TYPE_ ## d, r }

   const struct {
      brw_reg_type dst_type;
      brw_reg_type src0_type;
      brw_reg_type src1_type;
      brw_reg_type src2_type;

      bool expected_result;
   } test_vectors[] = {
      TV( F,  F, HF, HF, true),
      TV(HF, HF, HF, HF, devinfo.ver >= 20),
      TV( F, HF, HF, HF, devinfo.ver >= 20),
      TV(HF,  F, HF, HF, devinfo.ver >= 20),
      TV( F,  F,  F, HF, false),
      TV( F,  F, HF,  F, false),

      TV( F,  F, BF, BF, true),
      TV(BF, BF, BF, BF, devinfo.ver >= 20),
      TV(BF,  F, BF, BF, devinfo.ver >= 20),
      TV( F, BF, BF, BF, devinfo.ver >= 20),

      TV(DF, DF, DF, DF, false),
      TV(DF, DF, DF,  F, false),
      TV(DF, DF,  F, DF, false),
      TV(DF,  F, DF, DF, false),
      TV(DF, DF, DF, HF, false),
      TV(DF, DF, HF, DF, false),
      TV(DF, HF, DF, DF, false),

      TV(UD, UD, UB, UB, true),
      TV(UD, UD, UB, UD, false),
      TV(UD, UD, UD, UB, false),
      TV(UD, UD, UB, UW, false),
      TV(UD, UD, UW, UB, false),

      TV(UD, UB, UB, UB, false),
      TV(UD, UW, UB, UB, false),

      TV(UQ, UQ, UB, UB, false),
      TV(UQ, UQ, UB, UQ, false),
      TV(UQ, UQ, UQ, UB, false),
      TV(UQ, UQ, UB, UW, false),
      TV(UQ, UQ, UW, UB, false),

      TV( D,  D,  B,  B, true),
      TV( D,  D,  B, UB, true),
      TV( D,  D, UB,  B, true),
      TV( D, UD,  B,  B, true),

      TV( D,  D,  B,  D, false),
      TV( D,  D,  D,  B, false),
      TV( D,  D,  B,  W, false),
      TV( D,  D,  W,  B, false),

      TV( D,  B,  B,  B, false),
      TV( D,  W,  B,  B, false),

      TV( Q,  Q,  B,  B, false),
      TV( Q,  Q,  B,  Q, false),
      TV( Q,  Q,  Q,  B, false),
      TV( Q,  Q,  B,  W, false),
      TV( Q,  Q,  W,  B, false),

      TV(UD, UD, UB,  B, false),
      TV(UD, UD,  B, UB, false),
      TV(UD,  D, UB, UB, false),
   };

#undef TV

   brw_set_default_exec_size(p, devinfo.ver >= 20 ? BRW_EXECUTE_16
                                                  : BRW_EXECUTE_8);

   for (unsigned i = 0; i < ARRAY_SIZE(test_vectors); i++) {
      const auto &t = test_vectors[i];

      /* We encode the instruction and then decode to validate.  But our
       * encode reasonably asserts BF types when unsupported.  So skip those.
       *
       * TODO: Promote up the brw_eu_decoded_inst so that validation test can
       * use those types too instead of encoding/decoding.
       */
      if (!devinfo.has_bfloat16 &&
          (t.dst_type == BRW_TYPE_BF ||
           t.src0_type == BRW_TYPE_BF ||
           t.src1_type == BRW_TYPE_BF ||
           t.src2_type == BRW_TYPE_BF))
         continue;

      brw_DPAS(p,
               BRW_SYSTOLIC_DEPTH_8,
               8,
               retype(brw_vec8_grf(0, 0),  t.dst_type),
               retype(brw_vec8_grf(16, 0), t.src0_type),
               retype(brw_vec8_grf(32, 0), t.src1_type),
               retype(brw_vec8_grf(48, 0), t.src2_type));

      char *error = nullptr;
      bool valid = validate(p, &error);

      if (t.expected_result) {
         EXPECT_TRUE(valid)
            << "Test vector index = " << i << " expected to succeed "
            << "but failed validation with error: '" << error << "'.";
      } else {
         EXPECT_FALSE(valid)
            << "Test vector index = " << i << " expected to "
            << "fail validation but succeeded.";
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, dpas_src_subreg_nr)
{
   if (!devinfo.has_systolic)
      return;

#define TV(dt, od, t0, o0, t1, o1, o2, r) \
   { BRW_TYPE_ ## dt, od, BRW_TYPE_ ## t0, o0, BRW_TYPE_ ## t1, o1, o2, r }

   static const struct {
      brw_reg_type dst_type;
      unsigned dst_subnr;
      brw_reg_type src0_type;
      unsigned src0_subnr;
      brw_reg_type src1_src2_type;
      unsigned src1_subnr;
      unsigned src2_subnr;
      bool expected_result;
   } test_vectors[] = {
      TV( F,  0,  F,  0, HF,  0,  0, true),
      TV( D,  0,  D,  0,  B,  0,  0, true),
      TV( D,  0,  D,  0, UB,  0,  0, true),
      TV( D,  0, UD,  0,  B,  0,  0, true),

      TV( F,  1,  F,  0, HF,  0,  0, false),
      TV( F,  2,  F,  0, HF,  0,  0, false),
      TV( F,  3,  F,  0, HF,  0,  0, false),
      TV( F,  4,  F,  0, HF,  0,  0, false),
      TV( F,  5,  F,  0, HF,  0,  0, false),
      TV( F,  6,  F,  0, HF,  0,  0, false),
      TV( F,  7,  F,  0, HF,  0,  0, false),

      TV( F,  0,  F,  1, HF,  0,  0, false),
      TV( F,  0,  F,  2, HF,  0,  0, false),
      TV( F,  0,  F,  3, HF,  0,  0, false),
      TV( F,  0,  F,  4, HF,  0,  0, false),
      TV( F,  0,  F,  5, HF,  0,  0, false),
      TV( F,  0,  F,  6, HF,  0,  0, false),
      TV( F,  0,  F,  7, HF,  0,  0, false),

      TV( F,  0,  F,  0, HF,  1,  0, false),
      TV( F,  0,  F,  0, HF,  2,  0, false),
      TV( F,  0,  F,  0, HF,  3,  0, false),
      TV( F,  0,  F,  0, HF,  4,  0, false),
      TV( F,  0,  F,  0, HF,  5,  0, false),
      TV( F,  0,  F,  0, HF,  6,  0, false),
      TV( F,  0,  F,  0, HF,  7,  0, false),
      TV( F,  0,  F,  0, HF,  8,  0, false),
      TV( F,  0,  F,  0, HF,  9,  0, false),
      TV( F,  0,  F,  0, HF, 10,  0, false),
      TV( F,  0,  F,  0, HF, 11,  0, false),
      TV( F,  0,  F,  0, HF, 12,  0, false),
      TV( F,  0,  F,  0, HF, 13,  0, false),
      TV( F,  0,  F,  0, HF, 14,  0, false),
      TV( F,  0,  F,  0, HF, 15,  0, false),

      TV( F,  0,  F,  0, HF,  0,  1, false),
      TV( F,  0,  F,  0, HF,  0,  2, false),
      TV( F,  0,  F,  0, HF,  0,  3, false),
      TV( F,  0,  F,  0, HF,  0,  4, false),
      TV( F,  0,  F,  0, HF,  0,  5, false),
      TV( F,  0,  F,  0, HF,  0,  6, false),
      TV( F,  0,  F,  0, HF,  0,  7, false),
      TV( F,  0,  F,  0, HF,  0,  8, false),
      TV( F,  0,  F,  0, HF,  0,  9, false),
      TV( F,  0,  F,  0, HF,  0, 10, false),
      TV( F,  0,  F,  0, HF,  0, 11, false),
      TV( F,  0,  F,  0, HF,  0, 12, false),
      TV( F,  0,  F,  0, HF,  0, 13, false),
      TV( F,  0,  F,  0, HF,  0, 14, false),
      TV( F,  0,  F,  0, HF,  0, 15, false),

      /* These meet the requirements, but they specify a subnr that is part of
       * the next register. It is currently not possible to specify a subnr of
       * 32 for the B and UB values because brw_reg::subnr is only 5 bits.
       */
      TV( F, 16,  F,  0, HF,  0,  0, false),
      TV( F,  0,  F, 16, HF,  0,  0, false),
      TV( F,  0,  F,  0, HF,  0, 16, false),

      TV( D, 16,  D,  0,  B,  0,  0, false),
      TV( D,  0,  D, 16,  B,  0,  0, false),
   };

#undef TV

   brw_set_default_exec_size(p, devinfo.ver >= 20 ? BRW_EXECUTE_16
                                                  : BRW_EXECUTE_8);

   for (unsigned i = 0; i < ARRAY_SIZE(test_vectors); i++) {
      struct brw_reg dst =
         retype(brw_vec8_grf( 0, 0), test_vectors[i].dst_type);
      struct brw_reg src0 =
         retype(brw_vec8_grf(16, 0), test_vectors[i].src0_type);
      struct brw_reg src1 =
         retype(brw_vec8_grf(32, 0), test_vectors[i].src1_src2_type);
      struct brw_reg src2 =
         retype(brw_vec8_grf(48, 0), test_vectors[i].src1_src2_type);

      /* subnr for DPAS is in units of datatype precision instead of bytes as
       * it is for every other instruction. Set the value by hand instead of
       * using byte_offset() or similar.
       */
      dst.subnr = test_vectors[i].dst_subnr;
      src0.subnr = test_vectors[i].src0_subnr;
      src1.subnr = test_vectors[i].src1_subnr;
      src2.subnr = test_vectors[i].src2_subnr;

      brw_DPAS(p, BRW_SYSTOLIC_DEPTH_8, 8, dst, src0, src1, src2);

      EXPECT_EQ(test_vectors[i].expected_result, validate(p)) <<
         "test vector index = " << i;

      clear_instructions(p);
   }
}

TEST_P(validation_test, xe2_register_region_special_restrictions_for_src0_and_src1)
{
   if (devinfo.verx10 < 200)
      return;

   /* See "Src0 Restrictions" and "Src1 Restrictions" in "Special Restrictions"
    * in Bspec 56640 (r57070).
    */

   const unsigned V = 0xF;

#define DST(t, s, h)             { BRW_TYPE_ ## t, s, h }
#define SRC(t, s, v, w, h, ...)  { BRW_TYPE_ ## t, s, v, w, h, __VA_ARGS__ }
#define INDIRECT true

   static const struct {
      struct {
         brw_reg_type type;
         unsigned     subnr;
         unsigned     h;
      } dst;

      struct {
         brw_reg_type type;
         unsigned     subnr;
         unsigned     v;
         unsigned     w;
         unsigned     h;
         bool         indirect;
      } src0, src1;

      bool         expected_result;
   } test_vectors[] = {
      /* Source 0.  One element per dword channel. */
      { DST( D, 0, 1 ), SRC( D, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },
      { DST( D, 0, 1 ), SRC( W, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },
      { DST( D, 0, 1 ), SRC( B, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },

      { DST( W, 0, 2 ), SRC( D, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },
      { DST( W, 0, 2 ), SRC( W, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },
      { DST( W, 0, 2 ), SRC( B, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },

      { DST( B, 0, 4 ), SRC( D, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },
      { DST( B, 0, 4 ), SRC( W, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },
      { DST( B, 0, 4 ), SRC( B, 0,  1,1,0 ),  SRC( D, 0, 1,1,0 ), true },

      { DST( D, 0, 1 ), SRC( D, 0,  V,8,1, INDIRECT ),  SRC( D, 0, 1,1,0 ), true },
      { DST( D, 0, 1 ), SRC( D, 0,  V,1,0, INDIRECT ),  SRC( D, 0, 1,1,0 ), true },

      /* Source 0.  Uniform stride W->W cases. */
      { DST( W, 1, 1 ), SRC( W, 0,  1,1,0 ),  SRC( W, 0, 1,1,0 ), true },
      { DST( W, 1, 1 ), SRC( W, 2,  1,1,0 ),  SRC( W, 0, 1,1,0 ), true },
      { DST( W, 1, 1 ), SRC( W, 0,  2,1,0 ),  SRC( W, 0, 1,1,0 ), false },
      { DST( W, 1, 1 ), SRC( W, 2,  2,1,0 ),  SRC( W, 0, 1,1,0 ), true },
      { DST( W, 1, 1 ), SRC( W, 0,  4,1,0 ),  SRC( W, 0, 1,1,0 ), false },
      { DST( W, 1, 1 ), SRC( W, 2,  4,1,0 ),  SRC( W, 0, 1,1,0 ), false },

      /* Source 0.  Dword aligned W->W cases. */
      { DST( W, 2, 1 ), SRC( W, 0,  8,4,1 ),   SRC( W, 0, 1,1,0 ), true },
      { DST( W, 2, 1 ), SRC( W, 4,  8,4,1 ),   SRC( W, 0, 1,1,0 ), true },
      { DST( W, 2, 1 ), SRC( W, 0,  8,4,2 ),   SRC( W, 0, 1,1,0 ), false },
      { DST( W, 2, 1 ), SRC( W, 4,  8,4,2 ),   SRC( W, 0, 1,1,0 ), true },
      { DST( W, 2, 1 ), SRC( W, 0,  16,2,4 ),  SRC( W, 0, 1,1,0 ), false },
      { DST( W, 2, 1 ), SRC( W, 4,  16,2,4 ),  SRC( W, 0, 1,1,0 ), false },

      /* Source 0. Uniform stride W->B cases. */
      { DST( B, 2, 2 ), SRC( W, 0,  1,1,0),  SRC( W, 0, 1,1,0 ), true },
      { DST( B, 2, 2 ), SRC( W, 1,  1,1,0),  SRC( W, 0, 1,1,0 ), true },
      { DST( B, 2, 2 ), SRC( W, 0,  2,1,0),  SRC( W, 0, 1,1,0 ), false },
      { DST( B, 2, 2 ), SRC( W, 1,  2,1,0),  SRC( W, 0, 1,1,0 ), false },
      { DST( B, 2, 2 ), SRC( W, 0,  4,1,0),  SRC( W, 0, 1,1,0 ), false },
      { DST( B, 2, 2 ), SRC( W, 1,  4,1,0),  SRC( W, 0, 1,1,0 ), false },

      /* Source 0.  Dword aligned W->B cases. */
      { DST( B, 4, 2 ), SRC( W, 0,  8,4,1 ),   SRC( W, 0, 1,1,0 ), true },
      { DST( B, 4, 2 ), SRC( W, 2,  8,4,1 ),   SRC( W, 0, 1,1,0 ), true },
      { DST( B, 4, 2 ), SRC( W, 0,  8,4,2 ),   SRC( W, 0, 1,1,0 ), false },
      { DST( B, 4, 2 ), SRC( W, 2,  8,4,2 ),   SRC( W, 0, 1,1,0 ), false },
      { DST( B, 4, 2 ), SRC( W, 0,  16,2,4 ),  SRC( W, 0, 1,1,0 ), false },
      { DST( B, 4, 2 ), SRC( W, 2,  16,2,4 ),  SRC( W, 0, 1,1,0 ), false },

      /* TODO: Add B->W and B->B cases. */

      /* Source 1.  One element per dword channel. */
      { DST( D, 0, 1 ),  SRC( D, 0, 1,1,0 ),  SRC( D, 0,  1,1,0 ), true },
      { DST( D, 0, 1 ),  SRC( D, 0, 1,1,0 ),  SRC( W, 0,  1,1,0 ), true },
      { DST( W, 0, 2 ),  SRC( D, 0, 1,1,0 ),  SRC( D, 0,  1,1,0 ), true },
      { DST( W, 0, 2 ),  SRC( D, 0, 1,1,0 ),  SRC( W, 0,  1,1,0 ), true },

      /* Source 1.  Uniform stride W->W cases. */
      { DST( W, 1, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  1,1,0 ), true },
      { DST( W, 1, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 2,  1,1,0 ), true },
      { DST( W, 1, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  2,1,0 ), false },
      { DST( W, 1, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 2,  2,1,0 ), true },
      { DST( W, 1, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  4,1,0 ), false },
      { DST( W, 1, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 2,  4,1,0 ), false },

      /* Source 1.  Dword aligned W->W cases. */
      { DST( W, 2, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  8,4,1 ),  true },
      { DST( W, 2, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 4,  8,4,1 ),  true },
      { DST( W, 2, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  8,4,2 ),  false },
      { DST( W, 2, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 4,  8,4,2 ),  true },
      { DST( W, 2, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  16,2,4 ), false },
      { DST( W, 2, 1 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 4,  16,2,4 ), false },

      /* Source 1. Uniform stride W->B cases. */
      { DST( B, 2, 2 ),  SRC( B, 0, 1,1,0 ),  SRC( W, 0,  1,1,0), true },
      { DST( B, 2, 2 ),  SRC( B, 0, 1,1,0 ),  SRC( W, 1,  1,1,0), true },
      { DST( B, 2, 2 ),  SRC( B, 0, 1,1,0 ),  SRC( W, 0,  2,1,0), false },
      { DST( B, 2, 2 ),  SRC( B, 0, 1,1,0 ),  SRC( W, 1,  2,1,0), false },
      { DST( B, 2, 2 ),  SRC( B, 0, 1,1,0 ),  SRC( W, 0,  4,1,0), false },
      { DST( B, 2, 2 ),  SRC( B, 0, 1,1,0 ),  SRC( W, 1,  4,1,0), false },

      /* Source 1.  Dword aligned W->B cases. */
      { DST( B, 4, 2 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  8,4,1 ),  true },
      { DST( B, 4, 2 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 2,  8,4,1 ),  true },
      { DST( B, 4, 2 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  8,4,2 ),  false },
      { DST( B, 4, 2 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 2,  8,4,2 ),  false },
      { DST( B, 4, 2 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 0,  16,2,4 ), false },
      { DST( B, 4, 2 ),  SRC( W, 0, 1,1,0 ),  SRC( W, 2,  16,2,4 ), false },
   };

#undef DST
#undef SRC
#undef SOME
#undef INDIRECT

   for (unsigned i = 0; i < ARRAY_SIZE(test_vectors); i++) {
      struct brw_reg dst =
         brw_make_reg(FIXED_GRF,
                      0,
                      test_vectors[i].dst.subnr,
                      0,
                      0,
                      test_vectors[i].dst.type,
                      cvt(test_vectors[i].dst.h),
                      BRW_WIDTH_1,
                      cvt(test_vectors[i].dst.h),
                      BRW_SWIZZLE_XYZW,
                      WRITEMASK_XYZW);

      struct brw_reg src0 =
         brw_make_reg(FIXED_GRF,
                      2,
                      test_vectors[i].src0.subnr,
                      0,
                      0,
                      test_vectors[i].src0.type,
                      test_vectors[i].src0.v == V ? 0xF : cvt(test_vectors[i].src0.v),
                      cvt(test_vectors[i].src0.w) - 1,
                      cvt(test_vectors[i].src0.h),
                      BRW_SWIZZLE_XYZW,
                      WRITEMASK_XYZW);
      if (test_vectors[i].src0.indirect)
         src0.address_mode = BRW_ADDRESS_REGISTER_INDIRECT_REGISTER;

      struct brw_reg src1 =
         brw_make_reg(FIXED_GRF,
                      4,
                      test_vectors[i].src1.subnr,
                      0,
                      0,
                      test_vectors[i].src1.type,
                      test_vectors[i].src1.v == V ? 0xF : cvt(test_vectors[i].src1.v),
                      cvt(test_vectors[i].src1.w) - 1,
                      cvt(test_vectors[i].src1.h),
                      BRW_SWIZZLE_XYZW,
                      WRITEMASK_XYZW);
      if (test_vectors[i].src1.indirect)
         src1.address_mode = BRW_ADDRESS_REGISTER_INDIRECT_REGISTER;

      brw_ADD(p, dst, src0, src1);

      EXPECT_EQ(test_vectors[i].expected_result, validate(p)) <<
         "test vector index = " << i;

      clear_instructions(p);
   }
}

static brw_reg
brw_s0(enum brw_reg_type type, unsigned subnr)
{
   return brw_make_reg(ARF,
                       BRW_ARF_SCALAR,
                       subnr,
                       0,
                       0,
                       type,
                       BRW_VERTICAL_STRIDE_0,
                       BRW_WIDTH_1,
                       BRW_HORIZONTAL_STRIDE_0,
                       BRW_SWIZZLE_XYZW,
                       WRITEMASK_XYZW);
}

static brw_reg
brw_s0_with_region(enum brw_reg_type type, unsigned subnr, unsigned v, unsigned w, unsigned h)
{
   return brw_make_reg(ARF,
                       BRW_ARF_SCALAR,
                       subnr,
                       0,
                       0,
                       type,
                       cvt(v),
                       cvt(w)-1,
                       cvt(h),
                       BRW_SWIZZLE_XYZW,
                       WRITEMASK_XYZW);
}

static brw_reg
brw_grf(enum brw_reg_type type, unsigned nr, unsigned subnr, unsigned v, unsigned w, unsigned h)
{
   return brw_make_reg(FIXED_GRF,
                       nr,
                       subnr,
                       0,
                       0,
                       type,
                       cvt(v),
                       cvt(w)-1,
                       cvt(h),
                       BRW_SWIZZLE_XYZW,
                       WRITEMASK_XYZW);
}

TEST_P(validation_test, scalar_register_restrictions)
{
   /* Restrictions from BSpec 71168 (r55736). */

   if (devinfo.ver < 30)
      return;

   const brw_reg null_ud = retype(brw_null_reg(), BRW_TYPE_UD);

   struct test {
      enum opcode opcode;
      unsigned exec_size;
      struct {
         enum brw_conditional_mod cmod;
      } opts;
      brw_reg dst, src0, src1;
      bool expected_result;
   };

   static const struct test tests[] = {
      { BRW_OPCODE_MOV,  8, {}, brw_s0(BRW_TYPE_UD, 0), brw_grf(BRW_TYPE_UD, 1, 0, 1,1,0), {}, true },

      /* When destination, opcode must be MOV. */
      { BRW_OPCODE_NOT,  8, {}, brw_s0(BRW_TYPE_UD, 0), brw_grf(BRW_TYPE_UD, 1, 0, 1,1,0), {}, false },
      { BRW_OPCODE_ADD,  8, {}, brw_s0(BRW_TYPE_UD, 0), brw_grf(BRW_TYPE_UD, 1, 0, 1,1,0), brw_imm_ud(1), false },

      /* Source and destination types must match. */
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UQ, 0), brw_imm_uq(0x000036161836341E), {}, true },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UQ, 0), brw_imm_ud(0x1836341E),         {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UQ, 0), brw_imm_uw(0x341E),             {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UD, 0), brw_imm_uq(0x000036161836341E), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UD, 0), brw_imm_ud(0x1836341E),         {}, true },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UD, 0), brw_imm_uw(0x341E),             {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UW, 0), brw_imm_uq(0x000036161836341E), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UW, 0), brw_imm_ud(0x1836341E),         {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UW, 0), brw_imm_uw(0x341E),             {}, true  },

      /* When destination, must be integers of size 16, 32 or 64. */
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_B, 0),  brw_grf(BRW_TYPE_B,  1, 0, 0,1,0), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UB, 0), brw_grf(BRW_TYPE_UB, 1, 0, 0,1,0), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_W, 0),  brw_grf(BRW_TYPE_W,  1, 0, 0,1,0), {}, true },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UW, 0), brw_grf(BRW_TYPE_UW, 1, 0, 0,1,0), {}, true },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_D, 0),  brw_grf(BRW_TYPE_D,  1, 0, 0,1,0), {}, true },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UD, 0), brw_grf(BRW_TYPE_UD, 1, 0, 0,1,0), {}, true },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_Q, 0),  brw_grf(BRW_TYPE_Q,  1, 0, 0,1,0), {}, true },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UQ, 0), brw_grf(BRW_TYPE_UQ, 1, 0, 0,1,0), {}, true },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_F, 0),  brw_grf(BRW_TYPE_F,  1, 0, 0,1,0), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_HF, 0), brw_grf(BRW_TYPE_HF, 1, 0, 0,1,0), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_DF, 0), brw_grf(BRW_TYPE_DF, 1, 0, 0,1,0), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_V, 0),  brw_grf(BRW_TYPE_V,  1, 0, 0,1,0), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UV, 0), brw_grf(BRW_TYPE_UV, 1, 0, 0,1,0), {}, false },
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_VF, 0), brw_grf(BRW_TYPE_VF, 1, 0, 0,1,0), {}, false },

      /* When destination with immediate source, execution size must be 1. */
      { BRW_OPCODE_MOV,  8, {}, brw_s0(BRW_TYPE_UW, 0), brw_imm_uw(0x1234), {}, false },

      /* When destination with with immediate source, conditional modifier cannot be used. */
      { BRW_OPCODE_MOV,  1, {.cmod = BRW_CONDITIONAL_Z}, brw_s0(BRW_TYPE_UW, 0), brw_imm_uw(0x341E),                {}, false  },
      { BRW_OPCODE_MOV,  1, {.cmod = BRW_CONDITIONAL_Z}, brw_s0(BRW_TYPE_UW, 0), brw_grf(BRW_TYPE_UW, 1, 0, 0,1,0), {}, true },

      /* When source is scalar, destination must not be scalar. */
      { BRW_OPCODE_MOV,  1, {}, brw_s0(BRW_TYPE_UW, 0), brw_s0(BRW_TYPE_UW, 4), {}, false },

      /* When source of MOV is scalar, it must be a broadcast. */
      { BRW_OPCODE_MOV,  8, {}, brw_grf(BRW_TYPE_UW, 1, 0, 1,1,0), brw_s0_with_region(BRW_TYPE_UW, 0, 0,1,0), {}, true },
      { BRW_OPCODE_MOV,  8, {}, brw_grf(BRW_TYPE_UW, 1, 0, 1,1,0), brw_s0_with_region(BRW_TYPE_UW, 0, 1,1,0), {}, false },
      { BRW_OPCODE_MOV,  8, {}, brw_grf(BRW_TYPE_UW, 1, 0, 1,1,0), brw_s0_with_region(BRW_TYPE_UW, 0, 8,8,1), {}, false },

      /* When source 0 of SEND/SENDC is scalar, source 1 must be null. */
      { BRW_OPCODE_SEND,  16, {}, null_ud, brw_s0(BRW_TYPE_UD, 0), null_ud, true },
      { BRW_OPCODE_SENDC, 16, {}, null_ud, brw_s0(BRW_TYPE_UD, 0), null_ud, true },
      { BRW_OPCODE_SEND,  16, {}, null_ud, brw_s0(BRW_TYPE_UD, 0), brw_grf(BRW_TYPE_UD, 1, 0, 0,1,0), false },
      { BRW_OPCODE_SENDC, 16, {}, null_ud, brw_s0(BRW_TYPE_UD, 0), brw_grf(BRW_TYPE_UD, 1, 0, 0,1,0), false },

      /* When source is a scalar register, it must be on source 0. */
      { BRW_OPCODE_SEND,  16, {}, null_ud, brw_grf(BRW_TYPE_UD, 0, 0, 0,1,0), brw_grf(BRW_TYPE_UD, 2, 0, 0,1,0), true },
      { BRW_OPCODE_SEND,  16, {}, null_ud, brw_grf(BRW_TYPE_UD, 0, 0, 0,1,0), brw_s0(BRW_TYPE_UD, 0),            false },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(tests); i++) {
      const struct test &t = tests[i];

      switch (tests[i].opcode) {
      case BRW_OPCODE_ADD:
         brw_ADD(p, t.dst, t.src0, t.src1);
         break;
      case BRW_OPCODE_NOT:
         brw_NOT(p, t.dst, t.src0);
         break;
      case BRW_OPCODE_MOV:
         brw_MOV(p, t.dst, t.src0);
         break;
      case BRW_OPCODE_SEND:
      case BRW_OPCODE_SENDC: {
         brw_eu_inst *send = brw_next_insn(p, tests[i].opcode);
         brw_set_dest(p, send, t.dst);
         brw_set_src0(p, send, t.src0);
         brw_set_src1(p, send, t.src1);
         break;
      }
      default:
         UNREACHABLE("unexpected opcode in tests");
      }

      brw_eu_inst_set_exec_size(&devinfo, last_inst, cvt(t.exec_size) - 1);
      brw_eu_inst_set_cond_modifier(&devinfo, last_inst, t.opts.cmod);

      EXPECT_EQ(t.expected_result, validate(p)) <<
         "test vector index = " << i;

      clear_instructions(p);
   }
}

TEST_P(validation_test, bfloat_restrictions)
{
   /* Restrictions from ACM PRM, vol. 9, section "Register Region
    * Restrictions", sub-section 7.
    */

   if (!devinfo.has_bfloat16)
      return;

   struct test {
      const char *error_pattern;
      enum opcode opcode;
      unsigned exec_size;
      brw_reg dst, src0, src1, src2;
   };

   const char *PASS = nullptr;

   const struct test tests[] = {
      { PASS,
        BRW_OPCODE_MOV, 8, brw_grf(BRW_TYPE_BF, 10, 0, 2,1,2),
                           brw_grf(BRW_TYPE_F,  20, 0, 1,1,0) },

      { "pure bfloat16 operands are not supported",
        BRW_OPCODE_MOV, 8, brw_grf(BRW_TYPE_BF, 10, 0, 2,1,2),
                           brw_grf(BRW_TYPE_BF, 20, 0, 1,1,0) },

      { "Execution size must not be greater than",
        BRW_OPCODE_MOV, 16 * reg_unit(&devinfo),
                        brw_grf(BRW_TYPE_BF, 10, 0, 2,1,2),
                        brw_grf(BRW_TYPE_F,  20, 0, 1,1,0) },

      { PASS,
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 0, 2,1,2),
                           brw_grf(BRW_TYPE_F,  20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },

      { "pure bfloat16 operands are not supported",
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 0, 2,1,2),
                           brw_grf(BRW_TYPE_BF, 20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 30, 0, 1,1,0) },

      { "Broadcast of bfloat16 scalar is not supported",
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 0, 2,1,2),
                           brw_grf(BRW_TYPE_F,  20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 30, 0, 0,1,0) },

      { PASS,
        BRW_OPCODE_MUL, 8, brw_grf(BRW_TYPE_BF, 10, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },

      { "Bfloat16 not allowed in Src1 of 2-source instructions involving multiplier",
        BRW_OPCODE_MUL, 8, brw_grf(BRW_TYPE_BF, 10, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 30, 0, 1,1,0) },

      { PASS,
        BRW_OPCODE_MAD, 8, brw_grf(BRW_TYPE_BF, 10, 0, 2,1,2),
                           brw_grf(BRW_TYPE_BF, 20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 30, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  40, 0, 1,1,0) },

      { "Bfloat16 not allowed in Src2 of 3-source instructions involving multiplier",
        BRW_OPCODE_MAD, 8, brw_grf(BRW_TYPE_BF, 10, 0, 2,1,2),
                           brw_grf(BRW_TYPE_BF, 20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 40, 0, 1,1,0) },

      { PASS,
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 1, 2,1,2),
                           brw_grf(BRW_TYPE_F,  20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },

      { "Unpacked bfloat16 destination must have stride 2 and register offset 0 or 1",
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 3, 2,1,2),
                           brw_grf(BRW_TYPE_F,  20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },

      { PASS,
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 8 * reg_unit(&devinfo), 1,1,0),
                           brw_grf(BRW_TYPE_BF, 20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },

      { "Packed bfloat16 destination must have register offset 0 or half of GRF register",
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 1, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 20, 0, 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },

      { "Bfloat16 source must be packed",
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 20, 0, 2,1,2),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },

      { PASS,
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 20, 8 * reg_unit(&devinfo), 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },

      { "Bfloat16 source must have register offset 0 or half of GRF register",
        BRW_OPCODE_ADD, 8, brw_grf(BRW_TYPE_BF, 10, 0, 1,1,0),
                           brw_grf(BRW_TYPE_BF, 20, 5, 1,1,0),
                           brw_grf(BRW_TYPE_F,  30, 0, 1,1,0) },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(tests); i++) {
      const struct test &t = tests[i];

      switch (tests[i].opcode) {
      case BRW_OPCODE_MOV:
         brw_MOV(p, t.dst, t.src0);
         break;
      case BRW_OPCODE_ADD:
         brw_ADD(p, t.dst, t.src0, t.src1);
         break;
      case BRW_OPCODE_MUL:
         brw_MUL(p, t.dst, t.src0, t.src1);
         break;
      case BRW_OPCODE_MAD:
         brw_MAD(p, t.dst, t.src0, t.src1, t.src2);
         break;
      default:
         UNREACHABLE("unexpected opcode in tests");
      }

      if (tests[i].opcode == BRW_OPCODE_MAD) {
         brw_eu_inst_set_3src_exec_size(&devinfo, last_inst, cvt(t.exec_size) - 1);
      } else {
         brw_eu_inst_set_exec_size(&devinfo, last_inst, cvt(t.exec_size) - 1);
      }

      /* TODO: Expand this test logic to check validation error to other
       * tests.
       */

      char *error = nullptr;
      bool valid = validate(p, &error);

      if (t.error_pattern) {
         EXPECT_FALSE(valid)
            << "Test vector index = " << i << " expected to "
            << "fail validation with error containing: '" << t.error_pattern << "' "
            << "but succeeded instead.";

         if (error) {
            EXPECT_TRUE(strstr(error, t.error_pattern))
               << "Test vector index = " << i << " expected to "
               << "fail validation with error containing: '" << t.error_pattern << "' "
               << "but error was: '" << error << "'.";
         }
      } else {
         EXPECT_TRUE(valid)
            << "Test vector index = " << i << " expected to succeed "
            << "but failed validation with error: '" << error << "'.";
      }

      clear_instructions(p);
   }
}
