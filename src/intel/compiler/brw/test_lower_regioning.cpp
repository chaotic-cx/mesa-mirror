/*
 * Copyright © 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "brw_builder.h"

class lower_regioning_test : public brw_shader_pass_test {
};

TEST_F(lower_regioning_test, sel_ud_d_d)
{
   brw_builder bld = make_shader();

   brw_reg dest = vgrf(bld, BRW_TYPE_UD);
   brw_reg src0 = vgrf(bld, BRW_TYPE_D);
   brw_reg src1 = vgrf(bld, BRW_TYPE_D);

   bld.SEL(dest, src0, src1)
      ->saturate = true;

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_test, bf_to_f_accumulator)
{
   set_gfx_platform("dg2");

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 32);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 32);

   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_BF, 4);
   brw_reg tmp0 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg tmp1 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   bld.exec_all().MOV(acc0, src0);

   EXPECT_PROGRESS(brw_lower_simd_width, bld);
   EXPECT_PROGRESS(brw_lower_regioning, bld);

   brw_reg acc1 = acc0;
   brw_reg acc2 = acc0;
   brw_reg acc3 = acc0;

   acc1.nr = BRW_ARF_ACCUMULATOR + 1;
   acc2.nr = BRW_ARF_ACCUMULATOR + 2;
   acc3.nr = BRW_ARF_ACCUMULATOR + 3;

   exp.exec_all().group(8, 0).MOV(acc0, src0);
   exp.exec_all().group(8, 1).MOV(acc1, byte_offset(src0, 16));
   exp.exec_all().group(8, 2).UNDEF(tmp0);
   exp.exec_all().group(8, 2).MOV(retype(tmp0, BRW_TYPE_UW),
                                  retype(byte_offset(src0, 32), BRW_TYPE_UW));
   exp.exec_all().group(8, 2).MOV(acc2, retype(tmp0, BRW_TYPE_BF));
   exp.exec_all().group(8, 3).UNDEF(tmp1);
   exp.exec_all().group(8, 3).MOV(retype(tmp1, BRW_TYPE_UW),
                                  retype(byte_offset(src0, 48), BRW_TYPE_UW));
   exp.exec_all().group(8, 3).MOV(acc3, retype(tmp1, BRW_TYPE_BF));

   EXPECT_SHADERS_MATCH(bld, exp);
}

class lower_regioning_gfx35_test : public lower_regioning_test {
protected:
   void
   SetUp() override
   {
      set_gfx_platform("nvl");
      ASSERT_GE(devinfo->ver, 35);
      ASSERT_TRUE(devinfo->has_bfloat16);
   }

   unsigned
   grf_offset(const brw_reg &reg) const
   {
      return reg_offset(reg) % (reg_unit(devinfo) * REG_SIZE);
   }

   /* bf16 aligned to the second lowest bf16 of each fp32 execution
    * channel, i.e. a <2> region starting at .1.
    */
   brw_reg
   upper_bf(brw_builder &bld)
   {
      return byte_offset(horiz_stride(vgrf(bld, BRW_TYPE_BF, 4), 2), 2);
   }
};

TEST_F(lower_regioning_gfx35_test, bf_mixed_mode_strided_src_valid)
{
   brw_builder bld = make_shader();

   brw_reg dst = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = upper_bf(bld);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   bld.ADD(dst, src0, src1);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_mixed_mode_strided_dst_valid)
{
   brw_builder bld = make_shader();

   brw_reg dst = upper_bf(bld);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = upper_bf(bld);

   bld.ADD(dst, src0, src1);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_mixed_mode_packed_src)
{
   brw_builder bld = make_shader();

   brw_reg dst = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_BF);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   brw_inst *add = bld.ADD(dst, src0, src1);

   EXPECT_PROGRESS(brw_lower_regioning, bld);

   /* "bf16 on source must be strided to align all to the second lowest
    *  bf16 of the fp32 execution channel."
    */
   EXPECT_EQ(byte_stride(add->src[0]), 4u);
   EXPECT_EQ(grf_offset(add->src[0]), 2u);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_mixed_mode_packed_dst)
{
   brw_builder bld = make_shader();

   brw_reg dst = vgrf(bld, BRW_TYPE_BF);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   brw_inst *add = bld.ADD(dst, src0, src1);

   EXPECT_PROGRESS(brw_lower_regioning, bld);

   /* "bf16 on destination must be strided to align all to the second
    *  lowest bf16 of the fp32 execution channel."
    */
   EXPECT_EQ(byte_stride(add->dst), 4u);
   EXPECT_EQ(grf_offset(add->dst), 2u);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_mixed_mode_f_src_channel_aligned)
{
   brw_builder bld = make_shader();

   brw_reg dst = upper_bf(bld);
   brw_reg src0 = byte_offset(horiz_stride(vgrf(bld, BRW_TYPE_F, 3), 2), 4);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   brw_inst *add = bld.ADD(dst, src0, src1);

   EXPECT_PROGRESS(brw_lower_regioning, bld);

   /* fp32 sources start at the fp32 execution channel, not at the offset
    * of the bf16 destination within the channel.
    */
   EXPECT_EQ(byte_stride(add->src[0]), 4u);
   EXPECT_EQ(grf_offset(add->src[0]), 0u);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_mul_not_lowered)
{
   brw_builder bld = make_shader();

   brw_reg dst = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_BF);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* bf16 operands of MUL and MAD are converted to F later by
    * brw_lower_alu_restrictions().
    */
   bld.MUL(dst, src0, src1);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_packed_down_conversion_valid)
{
   brw_builder bld = make_shader();

   brw_reg dst = vgrf(bld, BRW_TYPE_BF, 2);
   brw_reg src = vgrf(bld, BRW_TYPE_F);

   /* "16bits elements must use offsets .0 or .16." */
   bld.MOV(dst, src);
   bld.MOV(byte_offset(dst, 16 * brw_type_size_bytes(BRW_TYPE_BF)), src);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_packed_down_conversion_src_offset)
{
   brw_builder bld = make_shader();

   brw_reg dst = vgrf(bld, BRW_TYPE_BF);
   brw_reg src = byte_offset(vgrf(bld, BRW_TYPE_F, 2), 4);

   brw_inst *mov = bld.MOV(dst, src);

   EXPECT_PROGRESS(brw_lower_regioning, bld);

   /* "Src0 must be packed GRF register (stride is 1) with [..] fp32 offset
    *  to be .0 only"
    */
   EXPECT_EQ(byte_stride(mov->src[0]), 4u);
   EXPECT_EQ(grf_offset(mov->src[0]), 0u);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_packed_down_conversion_dst_offset)
{
   brw_builder bld = make_shader();

   brw_reg dst = byte_offset(vgrf(bld, BRW_TYPE_BF, 2),
                             8 * brw_type_size_bytes(BRW_TYPE_BF));
   brw_reg src = vgrf(bld, BRW_TYPE_F);

   brw_inst *mov = bld.MOV(dst, src);

   EXPECT_PROGRESS(brw_lower_regioning, bld);

   /* "16bits elements must use offsets .0 or .16." */
   EXPECT_EQ(byte_stride(mov->dst), 2u);
   EXPECT_EQ(grf_offset(mov->dst), 0u);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}

TEST_F(lower_regioning_gfx35_test, bf_strided_down_conversion_valid)
{
   brw_builder bld = make_shader();

   brw_reg src = vgrf(bld, BRW_TYPE_F);

   /* Strided down conversions follow the integer regioning rules: a word
    * destination can be aligned to word 0 or 1 of the dword channel.
    */
   bld.MOV(horiz_stride(vgrf(bld, BRW_TYPE_BF, 2), 2), src);
   bld.MOV(upper_bf(bld), src);

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}
