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
