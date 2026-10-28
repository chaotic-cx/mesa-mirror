/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2023 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"
#include "radv_nir.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "radv_shader_info.h"

struct ctx {
   const struct radv_shader_stage *fs_stage;
   const struct radv_graphics_state_key *gfx_state;
};

static bool
pass(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   b->cursor = nir_after_instr(&intrin->instr);

   const struct ctx *ctx = data;
   const struct radv_graphics_state_key *gfx_state = ctx->gfx_state;
   const struct radv_shader_args *args = &ctx->fs_stage->args;

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_frag_coord_z: {
      if (!gfx_state->adjust_frag_coord_z)
         return false;

      b->fp_math_ctrl = nir_fp_no_fast_math;

      nir_def *frag_z = &intrin->def;

      /* VRS Rate X = Ancillary[2:3] */
      nir_def *ancillary = nir_load_vector_arg_amd(b, 1, .base = args->ac.ancillary.arg_index);
      nir_def *x_rate = nir_ubfe_imm(b, ancillary, 2, 2);

      /* xRate = xRate == 0x1 ? adjusted_frag_z : frag_z. */
      nir_def *cond = nir_ieq_imm(b, x_rate, 1);
      nir_def *mul = nir_bcsel(b, cond, nir_imm_float(b, 0.0625f), nir_imm_float(b, -0.0));

      /* adjusted_frag_z = dFdxFine(frag_z) * 0.0625 + frag_z */
      frag_z = nir_ffma_weak(b, nir_ddx_fine(b, frag_z), mul, frag_z);

      nir_def_rewrite_uses_after(&intrin->def, frag_z);

      b->fp_math_ctrl = 0;
      return true;
   }
   default:
      return false;
   }
}

bool
radv_nir_lower_fs_intrinsics(nir_shader *nir, const struct radv_shader_stage *fs_stage,
                             const struct radv_graphics_state_key *gfx_state)
{
   struct ctx ctx = {.fs_stage = fs_stage, .gfx_state = gfx_state};
   return nir_shader_intrinsics_pass(nir, pass, nir_metadata_none, &ctx);
}

static bool
lower_load_input_attachment(nir_builder *b, nir_intrinsic_instr *intrin, void *state)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_input_attachment_coord:
   case nir_intrinsic_load_depth_input_attachment_coord:
   case nir_intrinsic_load_stencil_input_attachment_coord: {
      b->cursor = nir_before_instr(&intrin->instr);

      nir_def *pos = nir_u2f32(b, nir_load_pixel_coord(b));
      nir_def *layer = nir_load_layer_id(b);
      nir_def *coord = nir_vec3(b, nir_f2i32(b, nir_channel(b, pos, 0)), nir_f2i32(b, nir_channel(b, pos, 1)), layer);

      nir_def_replace(&intrin->def, coord);
      return true;
   }
   default:
      return false;
   }
}

bool
radv_nir_lower_fs_input_attachment(nir_shader *nir)
{
   if (!nir->info.fs.uses_fbfetch_output)
      return false;

   return nir_shader_intrinsics_pass(nir, lower_load_input_attachment, nir_metadata_control_flow, NULL);
}
