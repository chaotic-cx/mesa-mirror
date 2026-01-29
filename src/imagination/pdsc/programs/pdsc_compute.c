/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include "pdsc_compute.h"

#include "pdsc/pdsc_internal.h"
#include "pdsc/pdsc_isa_pack.h"

#include "common/pvr_iface.h"
#include "util/macros.h"

#include <stdbool.h>
#include <stdint.h>

/* PDSC compute_shader patch IDs. */

enum pvr_compute_shader_id {
   /** Patch ID for CCS base wgid x. */
   PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_X,

   /** Patch ID for CCS base wgid y. */
   PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_Y,

   /** Patch ID for CCS base wgid z. */
   PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_Z,

   /** Patch ID for CCS doutd zero wgmem/coeffs src0. */
   PVR_COMPUTE_SHADER_ID_CCS_ZERO_SRC0,

   /** Patch ID for CCS doutd zero wgmem/coeffs src1. */
   PVR_COMPUTE_SHADER_ID_CCS_ZERO_SRC1,

   /** Patch ID for CCS doutd num wgs src0. */
   PVR_COMPUTE_SHADER_ID_CCS_NUM_WGS_SRC0,

   /** Patch ID for CCS doutd num wgs src1. */
   PVR_COMPUTE_SHADER_ID_CCS_NUM_WGS_SRC1,

   /** Patch ID for CCS doutw wgid src1: wgid x and/or y. */
   PVR_COMPUTE_SHADER_ID_CCS_WGID_XY,

   /** Patch ID for CCS doutw wgid src1: wgid z. */
   PVR_COMPUTE_SHADER_ID_CCS_WGID_Z,

   /** Patch ID for CCS doutw lid src1: lid x and/or y. */
   PVR_COMPUTE_SHADER_ID_CCS_LID_XY,

   /** Patch ID for CCS doutw lid src1: lid z. */
   PVR_COMPUTE_SHADER_ID_CCS_LID_Z,

   /** Patch ID for CCS doutu src0. */
   PVR_COMPUTE_SHADER_ID_CCS_DOUTU,

   /** Patch ID for CS doutw lid src1: lid x and/or y. */
   PVR_COMPUTE_SHADER_ID_CS_LID_XY,

   /** Patch ID for CS doutw lid src1: lid z. */
   PVR_COMPUTE_SHADER_ID_CS_LID_Z,

   /** Patch ID for CS conditional rendering ld src0. */
   PVR_COMPUTE_SHADER_ID_CS_CR_LD_SRC0,

   /** Patch ID for CS conditional rendering ld address 32-bit elem. */
   PVR_COMPUTE_SHADER_ID_CS_CR_LD_SEL_ELEM,

   /** Patch ID for CS conditional rendering negate. */
   PVR_COMPUTE_SHADER_ID_CS_CR_NEG,

   /** Patch ID for CS doutu src0. */
   PVR_COMPUTE_SHADER_ID_CS_DOUTU,
};

/* PDSC compute_shader saved ref IDs. */

enum pvr_compute_shader_ref_id {
   /** Ref ID for CS zero load dest. */
   PVR_COMPUTE_SHADER_REF_ID_ZERO_LD_DEST,
};

/* We could potentially have a variant for doutw_ext, but that would
 * always waste 1 register in 128-bit mode as we're only ever writing up to
 * three IDs, so let's not bother.
 */

pdsc_program *pdsc_compute_shader_gen(void *mem_ctx, enum pdsc_feature features)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, features, "compute_shader");

   /* First four temps in ccs. */
   DECL_TEMP32_ALIAS(p, ccs_wgid_x, PDSC_PRE_TEMP_CCS_WGID_X);
   DECL_TEMP32_ALIAS(p, ccs_wgid_y, PDSC_PRE_TEMP_CCS_WGID_Y);
   DECL_TEMP32_ALIAS(p, ccs_lid_xyz, PDSC_PRE_TEMP_CCS_LID_XYZ);
   DECL_TEMP32_ALIAS(p, ccs_wgid_z, PDSC_PRE_TEMP_CCS_WGID_Z);

   /* Don't need this for now. */

   /* For unpacking local id xyz in ccs. */
   DECL_TEMP32_ALIAS(p, ccs_lid_x, PDSC_PRE_TEMP_CCS_WGID_X);
   DECL_TEMP32_ALIAS(p, ccs_lid_y, PDSC_PRE_TEMP_CCS_WGID_Y);
   DECL_TEMP32_ALIAS(p, ccs_lid_z, PDSC_PRE_TEMP_CCS_WGID_Z);

   /* First two temps in cs. */
   DECL_TEMP32_ALIAS(p, cs_lid_x, PDSC_PRE_TEMP_CS_LID_X);
   DECL_TEMP32_ALIAS(p, cs_lid_yz, PDSC_PRE_TEMP_CS_LID_YZ);

   /* For unpacking local id y/z in cs. */
   DECL_TEMP32_ALIAS(p, cs_lid_y, PDSC_PRE_TEMP_CS_LID_YZ);
   DECL_TEMP32_ALIAS(p, cs_pad, PDSC_PRE_TEMP_CS_LID_YZ + 1);
   DECL_TEMP32_ALIAS(p, cs_lid_z, PDSC_PRE_TEMP_CS_LID_YZ + 2);

   static_assert(PDSC_PRE_TEMP_CS_LID_YZ + 1 == PDSC_PRE_TEMP_CCS_LID_XYZ,
                 "PDSC_PRE_TEMP_CS_LID_YZ + 1 != PDSC_PRE_TEMP_CCS_LID_XYZ");

   static_assert(PDSC_PRE_TEMP_CS_LID_YZ + 2 == PDSC_PRE_TEMP_CCS_WGID_Z,
                 "PDSC_PRE_TEMP_CS_LID_YZ + 2 != PDSC_PRE_TEMP_CCS_WGID_Z");

   /* Setup a zero. */
   ALLOC_TEMP64(p, tzero64);
   pdsc_limm(p, pdsc_ref_lo32(tzero64), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(tzero64), pdsc_ref_imm(0));

   /* For comparison. */
   ALLOC_TEMP64(p, tcmp);

   /* This same PDS program is used twice, once for the compute coefficient
    * sync task, once for the compute shader task.
    *
    * The preloaded temps are different in each instance, and will require
    * aliasing.
    *
    * If the if1 predicate is set, then we're running as the ccs task.
    */

   pdsc_instr_t not_if1 = pdsc_branch_push(p, PDSC_PRED_IF1, true);
   {
      /* if1 is set, didn't branch: ccs task. */

      /* Base workgroup IDs. */
      ALLOC_CONST32(p, base_wgid_x, PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_X);
      ALLOC_CONST32(p, base_wgid_y, PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_Y);
      ALLOC_CONST32(p, base_wgid_z, PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_Z);

      /* Zero out coefficients/local/workgroup memory
       * (if requested by the driver).
       */
      pdsc_doutd(p,
                 PVR_COMPUTE_SHADER_ID_CCS_ZERO_SRC0,
                 PVR_COMPUTE_SHADER_ID_CCS_ZERO_SRC1,
                 false);

      /* Write out num workgroups (if requested by the driver). */
      pdsc_doutd(p,
                 PVR_COMPUTE_SHADER_ID_CCS_NUM_WGS_SRC0,
                 PVR_COMPUTE_SHADER_ID_CCS_NUM_WGS_SRC1,
                 false);

      /* Offset the workgroup IDs. */
      pdsc_add32(p, ccs_wgid_x, ccs_wgid_x, base_wgid_x);
      pdsc_add32(p, ccs_wgid_y, ccs_wgid_y, base_wgid_y);
      pdsc_add32(p, ccs_wgid_z, ccs_wgid_z, base_wgid_z);

      pdsc_ref doutw_ccs_wgid_xy = pdsc_ref_64_2x32(ccs_wgid_x, ccs_wgid_y);
      pdsc_ref doutw_ccs_wgid__z = pdsc_ref_64_2x32(ccs_lid_xyz, ccs_wgid_z);

      /* Write out the workgroup ID(s) (if requested by the driver). */
      pdsc_doutw(p,
                 doutw_ccs_wgid_xy,
                 PVR_COMPUTE_SHADER_ID_CCS_WGID_XY,
                 pdsc_ref_null(),
                 false);
      pdsc_doutw(p,
                 doutw_ccs_wgid__z,
                 PVR_COMPUTE_SHADER_ID_CCS_WGID_Z,
                 pdsc_ref_null(),
                 false);

      /* Unpack the local ID(s). */
      /* We just do multiple shifts so we don't need to allocate consts. */
      pdsc_shl32(p, ccs_lid_z, ccs_lid_xyz, 2);
      pdsc_shr32(p, ccs_lid_z, ccs_lid_z, 22);

      pdsc_shl32(p, ccs_lid_y, ccs_lid_xyz, 12);
      pdsc_shr32(p, ccs_lid_y, ccs_lid_y, 22);

      pdsc_shl32(p, ccs_lid_x, ccs_lid_xyz, 22);
      pdsc_shr32(p, ccs_lid_x, ccs_lid_x, 22);

      pdsc_ref doutw_ccs_lid_xy = pdsc_ref_64_2x32(ccs_lid_x, ccs_lid_y);
      pdsc_ref doutw_ccs_lid__z = pdsc_ref_64_2x32(ccs_lid_xyz, ccs_lid_z);

      /* Write out the local IDs (if requested by the driver). */
      pdsc_doutw(p,
                 doutw_ccs_lid_xy,
                 PVR_COMPUTE_SHADER_ID_CCS_LID_XY,
                 pdsc_ref_null(),
                 false);
      pdsc_doutw(p,
                 doutw_ccs_lid__z,
                 PVR_COMPUTE_SHADER_ID_CCS_LID_Z,
                 pdsc_ref_null(),
                 false);

      /* Skip doutu if src0 is zero/uninitialised. */

      pdsc_instr_t cmp =
         pdsc_cmp(p,
                  tzero64,
                  pdsc_ref_null() /* To be replaced after the doutu. */,
                  .cop = PDSC_COP_NE);
      /* P0 is true if doutu_src0 is non-zero. */

      pdsc_instr_t not_p0 = pdsc_branch_push(p, PDSC_PRED_P0, true);
      {
         pdsc_instr_t doutu = pdsc_doutu(p,
                                         PVR_COMPUTE_SHADER_ID_CCS_DOUTU,
                                         false,
                                         .mods.cc = true);
         pdsc_src_set(p, cmp, 1, pdsc_src_get(p, doutu, 0));
      }
      pdsc_branch_pop(p, not_p0);

      pdsc_halt(p);
   }
   pdsc_branch_pop(p, not_if1);
   {
      /* if1 isn't set, branched: cs task. */

      /* Allocate a temp array for loading the conditional rendering buffer. */
      ALLOC_TEMP64_ARRAY(p, ld_data64x2, 2);

      pdsc_ref ld_data64[] = {
         pdsc_ref_elem64(ld_data64x2, 0),
         pdsc_ref_elem64(ld_data64x2, 1),
      };

      pdsc_ref ld_data32[] = {
         pdsc_ref_lo32(ld_data64[0]),
         pdsc_ref_hi32(ld_data64[0]),
         pdsc_ref_lo32(ld_data64[1]),
         pdsc_ref_hi32(ld_data64[1]),
      };

      /* Initialize to zero in case we aren't conditionally executing. */
      pdsc_limm(p, ld_data32[0], pdsc_ref_imm(0));
      /* pdsc_limm(p, ld_data32[1], pdsc_ref_imm(0)); */
      /* pdsc_limm(p, ld_data32[2], pdsc_ref_imm(0)); */
      /* pdsc_limm(p, ld_data32[3], pdsc_ref_imm(0)); */

      ALLOC_CONST32(p, cr_neg, PVR_COMPUTE_SHADER_ID_CS_CR_NEG);

      /* Unpack the local IDs. */
      /* We just do multiple shifts so we don't need to allocate consts. */
      pdsc_shr32(p, cs_lid_z, cs_lid_yz, 16);

      pdsc_shl32(p, cs_lid_y, cs_lid_yz, 16);
      pdsc_shr32(p, cs_lid_y, cs_lid_y, 16);

      pdsc_ref doutw_cs_lid_xy = pdsc_ref_64_2x32(cs_lid_x, cs_lid_y);
      pdsc_ref doutw_cs_lid_z_ = pdsc_ref_64_2x32(cs_pad, cs_lid_z);

      /* Write out the local IDs (if requested by the driver). */
      pdsc_doutw(p,
                 doutw_cs_lid_xy,
                 PVR_COMPUTE_SHADER_ID_CS_LID_XY,
                 pdsc_ref_null(),
                 false);
      pdsc_doutw(p,
                 doutw_cs_lid_z_,
                 PVR_COMPUTE_SHADER_ID_CS_LID_Z,
                 pdsc_ref_null(),
                 false);

      /* P0 is true if ld_src0 is non-zero. */
      pdsc_instr_t cmp =
         pdsc_cmp(p,
                  tzero64,
                  pdsc_ref_null() /* To be replaced after the ld. */,
                  .cop = PDSC_COP_NE);

      /* Perform load if ld_src0 is non-zero. */
      pdsc_setc(p, PDSC_PRED_P0);
      pdsc_instr_t ld = pdsc_ld(p,
                                ld_data64x2,
                                PVR_COMPUTE_SHADER_REF_ID_ZERO_LD_DEST,
                                PVR_COMPUTE_SHADER_ID_CS_CR_LD_SRC0,
                                false,
                                .mods.cc = true);
      pdsc_src_set(p, cmp, 1, pdsc_src_get(p, ld, 0));

      /* At this point, ld_data64x2 will either be all zeroes, or the result of
       * the load.
       */

      ALLOC_CONST32(p, ld_sel_elem, PVR_COMPUTE_SHADER_ID_CS_CR_LD_SEL_ELEM);
      pdsc_mov32(p, pdsc_ref_lo32(tcmp), ld_sel_elem);

      pdsc_setc(p, PDSC_PRED_P0);

      /* Set ld_data32[0] (ld_data64[0].lo32) to the appropriate data word. */
      {
         /* if (ld_sel_elem == 0) ld_data32[0] = ld_data32[0]; */
         /* pdsc_cmp(p, tcmp, pdsc_ref_imm(0), .cop = PDSC_COP_EQ); */
         /* pdsc_mov32(p, ld_data32[0], ld_data32[0], .cc = true); */

         /* if (ld_sel_elem == 1) ld_data32[0] = ld_data32[1]; */
         pdsc_cmp(p, tcmp, pdsc_ref_imm(1), .cop = PDSC_COP_EQ);
         pdsc_mov32(p, ld_data32[0], ld_data32[1], .cc = true);

         /* if (ld_sel_elem == 2) ld_data32[0] = ld_data32[2]; */
         pdsc_cmp(p, tcmp, pdsc_ref_imm(2), .cop = PDSC_COP_EQ);
         pdsc_mov32(p, ld_data32[0], ld_data32[2], .cc = true);

         /* if (ld_sel_elem == 3) ld_data32[0] = ld_data32[3]; */
         pdsc_cmp(p, tcmp, pdsc_ref_imm(3), .cop = PDSC_COP_EQ);
         pdsc_mov32(p, ld_data32[0], ld_data32[3], .cc = true);
      }

      /* Conditionally invert ld_data32[0] (ld_data64[0].lo32). */
      pdsc_xor32(p, ld_data32[0], ld_data32[0], cr_neg);

      /* Set ld_data32[1] (ld_data64[0].hi32) to zero. */
      pdsc_limm(p, ld_data32[1], pdsc_ref_imm(0));

      /* P0 is true if should conditionally render/execute. */
      pdsc_cmp(p, ld_data64[0], pdsc_ref_imm(0), .cop = PDSC_COP_NE);

      pdsc_instr_t not_p0 = pdsc_branch_push(p, PDSC_PRED_P0, true);
      {
         /* Only doutu if if0 is set (last in task). */
         pdsc_setc(p, PDSC_PRED_IF0);
         pdsc_doutu(p, PVR_COMPUTE_SHADER_ID_CS_DOUTU, false, .mods.cc = true);
      }
      pdsc_branch_pop(p, not_p0);

      pdsc_halt(p);
   }

   return p;
}

void pdsc_patch_compute_shader(pdsc_program *p,
                               struct pdsc_compute_arguments *args,
                               uint32_t *const_buffer)
{
   struct pdsc_doutd_src0 ccs_zero_src0 = {
      .slcmode = PDSC_SLC_MODE_LD_CACHED,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .doffset = 0,
      .sbase = args->zero_addr,
   };

   struct pdsc_doutd_src1 ccs_zero_src1 = {
      .last_issue = false,
      .store_dest = PDSC_STORE_DEST_COMMON,
      .cmode = PDSC_CMODE_LD_CACHED,
      .ao = args->zero_start,
      .bsize = args->zero_count,
   };

   struct pdsc_doutd_src0 ccs_num_wgs_src0 = {
      .slcmode = PDSC_SLC_MODE_LD_CACHED,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .doffset = 0,
      .sbase = args->num_wgs_addr,
   };

   struct pdsc_doutd_src1 ccs_num_wgs_src1 = {
      .last_issue = false,
      .store_dest = PDSC_STORE_DEST_COMMON,
      .cmode = PDSC_CMODE_LD_CACHED,
      .ao = args->num_wgs_start,
      .bsize = args->num_wgs_count,
   };

   enum pdsc_doutw_bsize bsize_lut_xy[] = {
      [0] = PDSC_DOUTW_BSIZE_NONE,
      [1] = PDSC_DOUTW_BSIZE_LOWER,
      [2] = PDSC_DOUTW_BSIZE_ALL64,
      [3] = PDSC_DOUTW_BSIZE_ALL64,
   };

   enum pdsc_doutw_bsize bsize_lut_z[] = {
      [0] = PDSC_DOUTW_BSIZE_NONE,
      [1] = PDSC_DOUTW_BSIZE_NONE,
      [2] = PDSC_DOUTW_BSIZE_NONE,
      [3] = PDSC_DOUTW_BSIZE_UPPER,
   };

   struct pdsc_doutw_src1 ccs_wgid_xy = {
      .last_issue = false,
      .store_dest = PDSC_STORE_DEST_COMMON,
      .cmode = PDSC_CMODE_LD_CACHED,
      .ao = args->wgid_start,
      .bsize = bsize_lut_xy[args->wgid_count],
   };

   struct pdsc_doutw_src1 ccs_wgid_z = {
      .last_issue = false,
      .store_dest = PDSC_STORE_DEST_COMMON,
      .cmode = PDSC_CMODE_LD_CACHED,
      .ao = args->wgid_start + 2,
      .bsize = bsize_lut_z[args->wgid_count],
   };

   struct pdsc_doutw_src1 ccs_lid_xy = {
      .last_issue = false,
      .store_dest = PDSC_STORE_DEST_COMMON,
      .cmode = PDSC_CMODE_LD_CACHED,
      .ao = args->ccs_lid_start,
      .bsize = bsize_lut_xy[args->ccs_lid_count],
   };

   struct pdsc_doutw_src1 ccs_lid_z = {
      .last_issue = true,
      .store_dest = PDSC_STORE_DEST_COMMON,
      .cmode = PDSC_CMODE_LD_CACHED,
      .ao = args->ccs_lid_start + 2,
      .bsize = bsize_lut_z[args->ccs_lid_count],
   };

   struct pdsc_doutu_src0 ccs_doutu = {
      .dual_phase = false,
      .temps = args->coeff_update_shader_temps,
      .sample_rate = PDSC_DOUTU_SAMPLE_RATE_INSTANCE,
      .exe_off = args->coeff_update_shader_addr,
   };

   struct pdsc_doutw_src1 cs_lid_xy = {
      .last_issue = false,
      .store_dest = PDSC_STORE_DEST_UNIFIED,
      .cmode = PDSC_CMODE_LD_CACHED,
      .ao = args->cs_lid_start,
      .bsize = bsize_lut_xy[args->cs_lid_count],
   };

   struct pdsc_doutw_src1 cs_lid_z = {
      .last_issue = true,
      .store_dest = PDSC_STORE_DEST_UNIFIED,
      .cmode = PDSC_CMODE_LD_CACHED,
      .ao = args->cs_lid_start + 2,
      .bsize = bsize_lut_z[args->cs_lid_count],
   };

   pdsc_ref zero_ld_dest =
      pdsc_load_ref(p, PVR_COMPUTE_SHADER_REF_ID_ZERO_LD_DEST);
   struct pdsc_ld64_src0 cs_cr = {
      .slcmode = PDSC_SLC_MODE_LD_CACHED,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,

      .dest = zero_ld_dest,

      .cmode = PDSC_CMODE_LD_CACHED,
      .count8 = pdsc_ref_get_chans(zero_ld_dest),
      .srcadd = args->cond_render_addr & ~0b1111,
   };

   struct pdsc_doutu_src0 cs_doutu = {
      .dual_phase = false,
      .temps = args->shader_temps,
      .sample_rate = PDSC_DOUTU_SAMPLE_RATE_INSTANCE,
      .exe_off = args->shader_addr,
   };

   pdsc_patch patches[] = {
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_X,
         .value = args->global_base_group[0],
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_Y,
         .value = args->global_base_group[1],
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_BASE_WGID_Z,
         .value = args->global_base_group[2],
      },

      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_ZERO_SRC0,
         .value = pdsc_doutd_src0_pack(p, &ccs_zero_src0),
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_ZERO_SRC1,
         .value = pdsc_doutd_src1_pack(p, &ccs_zero_src1),
      },

      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_NUM_WGS_SRC0,
         .value = pdsc_doutd_src0_pack(p, &ccs_num_wgs_src0),
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_NUM_WGS_SRC1,
         .value = pdsc_doutd_src1_pack(p, &ccs_num_wgs_src1),
      },

      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_WGID_XY,
         .value = pdsc_doutw_src1_pack(p, &ccs_wgid_xy),
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_WGID_Z,
         .value = pdsc_doutw_src1_pack(p, &ccs_wgid_z),
      },

      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_LID_XY,
         .value = pdsc_doutw_src1_pack(p, &ccs_lid_xy),
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_LID_Z,
         .value = pdsc_doutw_src1_pack(p, &ccs_lid_z),
      },

      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CCS_DOUTU,
         .value = !args->coeff_update_shader_addr
                     ? 0
                     : pdsc_doutu_src0_pack(p, &ccs_doutu),
      },

      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CS_LID_XY,
         .value = pdsc_doutw_src1_pack(p, &cs_lid_xy),
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CS_LID_Z,
         .value = pdsc_doutw_src1_pack(p, &cs_lid_z),
      },

      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CS_CR_LD_SRC0,
         .value = !args->cond_render_addr ? 0 : pdsc_ld64_src0_pack(p, &cs_cr),
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CS_CR_LD_SEL_ELEM,
         .value = (args->cond_render_addr >> 2) % 4,
      },
      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CS_CR_NEG,
         .value = (args->cond_render_invert || !args->cond_render_addr)
                     ? UINT32_MAX
                     : 0,
      },

      {
         .patch_id = PVR_COMPUTE_SHADER_ID_CS_DOUTU,
         .value = pdsc_doutu_src0_pack(p, &cs_doutu),
      },
   };
   pdsc_patch_program(p, ARRAY_SIZE(patches), patches, const_buffer);
}
