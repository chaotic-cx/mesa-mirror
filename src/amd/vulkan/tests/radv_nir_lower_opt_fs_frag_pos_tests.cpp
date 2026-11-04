/*
 * Copyright © 2026 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "nir/radv_nir.h"
#include "tests/nir_test.h"

class radv_nir_lower_opt_fs_frag_pos_test : public nir_test {
protected:
   radv_nir_lower_opt_fs_frag_pos_test(): nir_test("radv_nir_lower_opt_fs_frag_pos_test")
   {
      b->shader->info.stage = MESA_SHADER_FRAGMENT;
   }

   void run(bool vrs, bool sample_shading, bool expect_progress = true);
};

void
radv_nir_lower_opt_fs_frag_pos_test::run(bool vrs, bool sample_shading, bool expect_progress)
{
   nir_opt_copy_prop(b->shader);

   EXPECT_EQ(radv_nir_lower_opt_fs_frag_pos(b->shader, vrs, sample_shading), expect_progress);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, b->shader, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(progress, b->shader, nir_opt_copy_prop);
      NIR_PASS(progress, b->shader, nir_opt_dce);
      NIR_PASS(progress, b->shader, nir_opt_constant_folding);
      NIR_PASS(progress, b->shader, nir_opt_cse);
      NIR_PASS(progress, b->shader, nir_opt_algebraic);
   } while (progress);

   nir_validate_shader(b->shader, "After radv_nir_lower_opt_fs_frag_pos");
}

TEST_F(radv_nir_lower_opt_fs_frag_pos_test, pixelx_bit0_fragy_bit0_vrs)
{
   nir_unit_test_output(b, nir_iand_imm(b, nir_channel(b, nir_load_pixel_coord(b), 0), 0x1), 0);
   nir_unit_test_output(b, nir_iand_imm(b, nir_f2u32(b, nir_channel(b, nir_load_frag_coord_xy(b), 1)), 0x1), 1);

   run(true, false, true);

   check_nir_string(NIR_REFERENCE_SHADER(R"(
      shader: MESA_SHADER_FRAGMENT
      name: radv_nir_lower_opt_fs_frag_pos_test
      max_subgroup_size: 128
      min_subgroup_size: 1
      decl_function main () (entrypoint)

      impl main {
         block b0:   // preds:
         16x2   %0 = @load_pixel_coord
         1      %1 = @load_use_quad_pos_amd
         32     %2 = @load_subgroup_invocation
         16     %3 = u2u16 %2
         16     %4 = bcsel %1, %3, %0.x
         16     %5 = load_const (0x0001)
         16     %6 = iand %4, %5 (0x1)
                     @unit_test_output (%6) (base=0)
         32     %7 = u2f32 %0.y
         32     %8 = load_const (0x00000001)
         32     %9 = ushr %2, %8 (0x1)
         32    %10 = u2f32 %9
         32    %11 = bcsel %1, %10, %7
         32    %12 = f2u32 %11
         32    %13 = iand %12, %8 (0x1)
                     @unit_test_output (%13) (base=1)
                     // succs: b1
         block b1:
      }
   )"));
}

TEST_F(radv_nir_lower_opt_fs_frag_pos_test, fragx_bit0_sampleposy)
{
   nir_unit_test_output(b, nir_iand_imm(b, nir_f2u32(b, nir_channel(b, nir_load_frag_coord_xy(b), 0)), 0x1), 0);
   nir_unit_test_output(b, nir_channel(b, nir_load_sample_pos(b), 1), 1);

   run(false, true, true);

   check_nir_string(NIR_REFERENCE_SHADER(R"(
      shader: MESA_SHADER_FRAGMENT
      name: radv_nir_lower_opt_fs_frag_pos_test
      max_subgroup_size: 128
      min_subgroup_size: 1
      decl_function main () (entrypoint)

      impl main {
         block b0:  // preds:
         32x2  %0 = @load_frag_coord_xy
         32    %1 = f2u32 %0.x
         32    %2 = load_const (0x00000001)
         32    %3 = iand %1, %2 (0x1)
                    @unit_test_output (%3) (base=0)
         32    %4 = ffract %0.y
                    @unit_test_output (%4) (base=1)
                    // succs: b1
         block b1:
      }
   )"));
}

TEST_F(radv_nir_lower_opt_fs_frag_pos_test, fragx_bit0_sampleposy_vrs)
{
   nir_unit_test_output(b, nir_iand_imm(b, nir_f2u32(b, nir_channel(b, nir_load_frag_coord_xy(b), 0)), 0x1), 0);
   nir_unit_test_output(b, nir_channel(b, nir_load_sample_pos(b), 1), 1);

   run(true, true, true);

   check_nir_string(NIR_REFERENCE_SHADER(R"(
      shader: MESA_SHADER_FRAGMENT
      name: radv_nir_lower_opt_fs_frag_pos_test
      max_subgroup_size: 128
      min_subgroup_size: 1
      decl_function main () (entrypoint)

      impl main {
         block b0:  // preds:
         32x2  %0 = @load_frag_coord_xy
         32    %1 = f2u32 %0.x
         32    %2 = load_const (0x00000001)
         32    %3 = iand %1, %2 (0x1)
                    @unit_test_output (%3) (base=0)
         32    %4 = ffract %0.y
                    @unit_test_output (%4) (base=1)
                    // succs: b1
         block b1:
      }
   )"));
}

TEST_F(radv_nir_lower_opt_fs_frag_pos_test, fragx_bit0_pixelx_vrs)
{
   nir_unit_test_output(b, nir_iand_imm(b, nir_f2u32(b, nir_channel(b, nir_load_frag_coord_xy(b), 0)), 0x1), 0);
   nir_unit_test_output(b, nir_channel(b, nir_load_pixel_coord(b), 0), 1);

   run(true, false, true);

   check_nir_string(NIR_REFERENCE_SHADER(R"(
      shader: MESA_SHADER_FRAGMENT
      name: radv_nir_lower_opt_fs_frag_pos_test
      max_subgroup_size: 128
      min_subgroup_size: 1
      decl_function main () (entrypoint)

      impl main {
         block b0:  // preds:
         16x2  %0 = @load_pixel_coord
         32    %1 = u2u32 %0.x
         32    %2 = load_const (0x00000001)
         32    %3 = iand %1, %2 (0x1)
                    @unit_test_output (%3) (base=0)
         16    %4 = mov %0.x
                    @unit_test_output (%4) (base=1)
                    // succs: b1
         block b1:
      }
   )"));
}
