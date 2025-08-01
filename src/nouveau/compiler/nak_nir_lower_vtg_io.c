/*
 * Copyright © 2023 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "nak_private.h"
#include "nir_builder.h"

static nir_def *
tess_ctrl_output_vtx(nir_builder *b, nir_def *vtx)
{
   /* This is the pattern we see emitted by the blob driver:
    *
    *    S2R R0, SR_LANEID
    *    S2R R6, SR_INVOCATION_ID
    *    # R3 is our vertex index
    *    SULD.P.2D.CTA.R.IGN R3, [R2], 0x1d, 0x0
    *    IMAD.IADD R5, R0, 0x1, -R6
    *    IMAD.SHL.U32 R0, R3, 0x4, RZ
    *    LEA.HI.SX32 R4, R0, R5, 0x1e
    *    ALD.O R4, a[0x88], R4
    *
    * Translating the MADs and re-naming registers, this is
    *
    *    %r0 = iadd %lane -%invoc
    *    %r1 = imul %vtx 0x4
    *    %r2 = lea.hi.sx32 %r1 %r0 0x1e
    *    %out = ald.o a[%r2][0x88]
    *
    * But `lea.hi.sx32 %r1 %r0 0x1e` is just `(%r1 >> (32 - 0x1e)) + %r0`.
    * Since %r1 is just `%vtx * 4` and 0x1e is 30, the whole bit on the left
    * is `(%vtx * 4) >> 2 = %vtx`, assuming no overflow.  So, this means
    *
    *    %r0 = iadd %lane -%invoc
    *    %r2 = iadd %vtx %r0
    *    %out = ald.o a[%r2][0x88]
    *
    * In other words, the hardware actually indexes them by lane index with
    * all of the invocations for a given TCS dispatch going in a sequential
    * range of lanes.  We have to compute the lane index of the requested
    * invocation from the invocation index.
    */
   nir_def *lane = nak_nir_load_sysval(b, NAK_SV_LANE_ID,
                                       ACCESS_CAN_REORDER);
   nir_def *invoc = nak_nir_load_sysval(b, NAK_SV_INVOCATION_ID,
                                        ACCESS_CAN_REORDER);

   return nir_iadd(b, lane, nir_iadd(b, vtx, nir_ineg(b, invoc)));
}

static bool
lower_vtg_io_intrin(nir_builder *b,
                    nir_intrinsic_instr *intrin,
                    void *cb_data)
{
   const struct nak_compiler *nak = cb_data;
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *vtx = NULL, *offset = NULL, *data = NULL;
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_output:
      offset = intrin->src[0].ssa;
      break;

   case nir_intrinsic_load_per_vertex_input:
      vtx = intrin->src[0].ssa;
      offset = intrin->src[1].ssa;
      break;

   case nir_intrinsic_load_per_vertex_output:
      if (b->shader->info.stage == MESA_SHADER_TESS_CTRL)
         vtx = tess_ctrl_output_vtx(b, intrin->src[0].ssa);
      else
         vtx = intrin->src[0].ssa;
      offset = intrin->src[1].ssa;
      break;

   case nir_intrinsic_store_output:
      data = intrin->src[0].ssa;
      offset = intrin->src[1].ssa;
      break;

   case nir_intrinsic_store_per_vertex_output:
      data = intrin->src[0].ssa;
      vtx = intrin->src[1].ssa;
      offset = intrin->src[2].ssa;
      break;

   default:
      return false;
   }
   const bool offset_is_const = nir_src_is_const(nir_src_for_ssa(offset));

   const bool is_store = data != NULL;

   bool is_output;
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
      is_output = false;
      break;

   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
      is_output = true;
      break;

   default:
      UNREACHABLE("Unknown NIR I/O intrinsic");
   }

   bool is_patch;
   switch (b->shader->info.stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_GEOMETRY:
      is_patch = false;
      break;

   case MESA_SHADER_TESS_CTRL:
      is_patch = is_output && vtx == NULL;
      break;

   case MESA_SHADER_TESS_EVAL:
      is_patch = !is_output && vtx == NULL;
      break;

   default:
      UNREACHABLE("Unknown shader stage");
   }

   nir_component_mask_t mask;
   if (is_store)
      mask = nir_intrinsic_write_mask(intrin);
   else
      mask = nir_component_mask(intrin->num_components);

   if (vtx != NULL && !is_output) {
      if (nak->sm >= 50) {
         nir_def *info = nak_nir_load_sysval(b, NAK_SV_INVOCATION_INFO,
                                             ACCESS_CAN_REORDER);
         nir_def *lo = nir_extract_u8_imm(b, info, 0);
         nir_def *hi = nir_extract_u8_imm(b, info, 2);
         nir_def *idx = nir_iadd(b, nir_imul(b, lo, hi), vtx);
         vtx = nir_isberd_nv(b, idx);
      } else {
         vtx = nir_vild_nv(b, vtx);
      }
   }

   if (vtx == NULL)
      vtx = nir_imm_int(b, 0);

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intrin);
   unsigned component = nir_intrinsic_component(intrin);

   uint32_t base_addr;
   if (b->shader->info.stage == MESA_SHADER_VERTEX && !is_output)
      base_addr = nak_attribute_attr_addr(nak, sem.location);
   else
      base_addr = nak_varying_attr_addr(nak, sem.location);
   base_addr += 4 * component;

   uint32_t range;
   if (offset_is_const) {
      unsigned const_offset = nir_src_as_uint(nir_src_for_ssa(offset));

      /* Tighten the range */
      base_addr += const_offset * 16;
      range = 4 * intrin->num_components;

      if (const_offset != 0)
         offset = nir_imm_int(b, 0);
   } else {
      /* Offsets from NIR are in vec4's */
      offset = nir_imul_imm(b, offset, 16);
      range = (sem.num_slots - 1) * 16 + intrin->num_components * 4;
   }

   const struct nak_nir_attr_io_flags flags = {
      .output = is_output,
      .patch = is_patch,
      .phys = !offset_is_const && !is_patch,
   };

   nir_def *dst_comps[NIR_MAX_VEC_COMPONENTS];
   while (mask) {
      const unsigned c = ffs(mask) - 1;
      unsigned comps = ffs(~(mask >> c)) - 1;
      assert(comps > 0);

      unsigned c_addr = base_addr + 4 * c;

      /* vec2 has to be vec2 aligned, vec3/4 have to be vec4 aligned.  We
       * don't have actual alignment information on these intrinsics but we
       * can assume that the indirect offset (if any) is a multiple of 16 so
       * we don't need to worry about that and can just look at c_addr.
       */
      comps = MIN2(comps, 4);
      if (c_addr & 0xf)
         comps = MIN2(comps, 2);
      if (c_addr & 0x7)
         comps = 1;
      assert(!(c_addr & 0x3));

      /* Vector load/store with non-constant offsets don't work pre-Maxwell.
       * They encode fine and they don't throw any shader exceptions but the
       * seem to get stuck in the hardware and we get context timeouts.
       */
      if (nak->sm < 50 && !offset_is_const)
         comps = 1;

      nir_def *c_offset = offset;
      if (flags.phys) {
         /* Physical addressing has to be scalar */
         comps = 1;

         /* Use al2p to compute a physical address */
         c_offset = nir_al2p_nv(b, offset, .base = c_addr,
                                .flags = NAK_AS_U32(flags));
         c_addr = 0;
      }

      if (is_store) {
         nir_def *c_data = nir_channels(b, data, BITFIELD_RANGE(c, comps));
         nir_ast_nv(b, c_data, vtx, c_offset,
                    .base = c_addr,
                    .flags = NAK_AS_U32(flags),
                    .range_base = base_addr,
                    .range = range);
      } else {
         uint32_t access = flags.output ? 0 : ACCESS_CAN_REORDER;
         nir_def *c_data = nir_ald_nv(b, comps, vtx, c_offset,
                                      .base = c_addr,
                                      .flags = NAK_AS_U32(flags),
                                      .range_base = base_addr,
                                      .range = range,
                                      .access = access);
         for (unsigned i = 0; i < comps; i++)
            dst_comps[c + i] = nir_channel(b, c_data, i);
      }

      mask &= ~BITFIELD_RANGE(c, comps);
   }

   if (!is_store) {
      nir_def *dst = nir_vec(b, dst_comps, intrin->num_components);
      nir_def_rewrite_uses(&intrin->def, dst);
   }

   nir_instr_remove(&intrin->instr);

   return true;
}

bool
nak_nir_lower_vtg_io(nir_shader *nir, const struct nak_compiler *nak)
{
   return nir_shader_intrinsics_pass(nir, lower_vtg_io_intrin,
                                     nir_metadata_control_flow,
                                     (void *)nak);
}
