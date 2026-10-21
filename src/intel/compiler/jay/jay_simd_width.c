/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/brw/brw_eu_defines.h"
#include "jay_builder.h"
#include "jay_builder_opcodes.h"
#include "jay_ir.h"
#include "jay_opcodes.h"

#include "intel/compiler/brw/brw_rt.h"

/*
 * Return the maximum SIMD width that can operate on a given operand.
 */
static inline unsigned
operand_width(const jay_shader *shader, jay_def def, enum jay_type type)
{
   /* Look up the block, synthesizing a trivial one for non-GPRs */
   struct jay_register_block B = { 0 };
   if (def.file == GPR) {
      B = jay_lookup_block(&shader->partition, def.reg, GPR);
   }

   /* Determine the effective stride in bits. This needs to also account for
    * 64-bit UGPR-only instructions for the I->broadcast_flag case.
    */
   unsigned bits = MAX2(jay_stride_to_bits(B.stride), jay_type_size_bits(type));

   /* We must fit within 2 GRFs and within the SIMDness of the register block */
   return MIN2((512 * reg_unit(shader->devinfo)) / bits,
               shader->dispatch_width >> B.split);
}

static unsigned
max_simd_width(const jay_shader *shader, const jay_inst *I)
{
   /* Only certain "complex" quad swizzles require splitting down to SIMD4 */
   if (I->op == JAY_OPCODE_QUAD_SWIZZLE &&
       (jay_quad_swizzle_swizzle(I) == JAY_QUAD_SWIZZLE_XYXY ||
        jay_quad_swizzle_swizzle(I) == JAY_QUAD_SWIZZLE_ZWZW)) {
      return 4;
   }

   unsigned grf_simd = shader->devinfo->ver >= 20 ? 16 : 8;

   /* These special instructions need to be split for various reasons. */
   if (I->op == JAY_OPCODE_DESWIZZLE_ODD ||
       I->op == JAY_OPCODE_MUL_32 ||
       I->op == JAY_OPCODE_ZIP ||
       jay_clobbers_address_reg(I)) {
      return grf_simd;
   }

   unsigned width = 32;

   /* These special instructions need to be split for various reasons. */
   if (I->op == JAY_OPCODE_EXPAND_QUAD ||
       I->op == JAY_OPCODE_EXTRACT_SUBSPAN_INFO ||
       I->op == JAY_OPCODE_EXTRACT_BYTE_PER_8LANES ||
       I->op == JAY_OPCODE_OFFSET_PACKED_PIXEL_COORDS) {
      width = MIN2(width, 16);
   }

   if (I->op != JAY_OPCODE_SEND) {
      width = MIN2(width, operand_width(shader, I->dst, I->type));

      jay_foreach_src(I, s) {
         width = MIN2(width, operand_width(shader, I->src[s], I->type));
      }
   } else {
      /* Sends are split elsewhere */
      return 32;
   }

   /* Bspec 56797 (r62012):
    *
    *    Math operation rules when half-floats are used on both source and
    *    destination operands and both source and destinations are packed.
    *    The execution size must be 16.
    */
   if (I->op == JAY_OPCODE_MATH &&
       I->type == JAY_TYPE_F16 &&
       (I->dst.file == GPR && jay_def_stride(shader, I->dst) == JAY_STRIDE_2) &&
       (I->src[0].file == GPR &&
        jay_def_stride(shader, I->src[0]) == JAY_STRIDE_2)) {
      width = MIN2(width, 16);
   }

   /* BSpec 56640 requires that execution size be no greater than 16
    * for mixed-mode operations involving bfloats.
    */
   if (I->type == JAY_TYPE_BF16) {
      width = MIN2(width, 16);
   }
   jay_foreach_src(I, s) {
      if (jay_src_type(I, s) == JAY_TYPE_BF16)
         width = MIN2(width, 16);
   }

   return width;
}

unsigned
jay_simd_split(const jay_shader *s, const jay_inst *I)
{
   unsigned actual = jay_simd_width_logical(s, I);
   unsigned max = max_simd_width(s, I);

   return (actual > max) ? (util_logbase2(actual) - util_logbase2(max)) : 0;
}

static void
pass(jay_function *func)
{
   jay_builder b = jay_init_builder(func, jay_before_function(func));

   jay_foreach_inst_in_func_safe(func, block, I) {
      unsigned split = I->op == JAY_OPCODE_SEND ?
                          I->simd_split :
                          jay_simd_split(func->shader, I);
      if (split) {
         I->simd_split = split;
         b.cursor = jay_after_inst(I);

         for (unsigned i = 1; i < (1 << split); ++i) {
            jay_inst *clone = jay_clone_inst(&b, I, I->num_srcs);
            clone->simd_offs = i;
            jay_builder_insert(&b, clone);

            if (I->op == JAY_OPCODE_SEND) {
               if (jay_send_sfid(I) == GEN_SFID_RENDER_CACHE) {
                  clone->src[0] = jay_imm(jay_as_uint(clone->src[0]) | 0x800);
               } else if (jay_send_sfid(I) == GEN_SFID_RAY_TRACE_ACCELERATOR) {
                  assert(clone->src[2].file == UGPR);
                  jay_def global_lo = jay_bare_reg(UGPR, clone->src[2].reg);
                  signed offset_B = align(BRW_RT_DISPATCH_GLOBALS_SIZE, 64);

                  assert(clone->src[2].reg != clone->dst.reg &&
                         "must not be tied");
                  assert(clone->src[2].file != clone->src[3].file &&
                         "header + payload");

                  b.cursor = jay_before_inst(clone);
                  jay_ADD(&b, JAY_TYPE_U32, global_lo, global_lo, offset_B);

                  b.cursor = jay_after_inst(clone);
                  jay_ADD(&b, JAY_TYPE_U32, global_lo, global_lo, -offset_B);
               }

               /* Only the last SEND gets EOT */
               jay_set_send_eot(clone, jay_send_eot(clone) &&
                                          ((i + 1) == (1 << split)));
            }
         }

         if (I->op == JAY_OPCODE_SEND) {
            /* The first SEND can't possibly EOT */
            jay_set_send_eot(I, false);
         }
      }
   }

   /* Expand macros after SIMD splitting */
   jay_foreach_inst_in_func_safe(func, block, I) {
      b.cursor = jay_after_inst(I);

      if (I->op == JAY_OPCODE_MUL_32) {
         jay_def acc = jay_bare_reg(ACCUM, 0);

         jay_inst *mac =
            jay_MACL(&b, I->type, I->dst, I->src[0], I->src[1], acc);
         mac->simd_offs = I->simd_offs;
         mac->simd_split = I->simd_split;

         if (I->predication) {
            jay_add_predicate(&b, mac, *jay_inst_get_predicate(I), jay_null());
         }

         if (jay_mul_32_high(I)) {
            mac->op = JAY_OPCODE_MACH;
         }

         I->dst = acc;
         I->op = JAY_OPCODE_MUL_32_PART;
      } else if (I->op == JAY_OPCODE_SHUFFLE) {
         /* Use a dedicated address register for 1x1 indirects to avoid
          * interfering with a0.0 and a0.2 users. This affects UGPR spilling.
          */
         bool VxH = !jay_is_uniform(I->src[1]);
         unsigned addr = VxH ? 0 : 4;
         unsigned nr = VxH ? jay_simd_width_physical(func->shader, I) : 1;
         jay_def a0 = jay_bare_regs(J_ADDRESS, addr, DIV_ROUND_UP(nr, 2));

         struct jay_register_block block =
            jay_lookup_block(&func->shader->partition, I->src[0].reg,
                             I->src[0].file);
         unsigned reg_width =
            2 * (I->src[0].file == UGPR ? 1 : func->shader->dispatch_width);
         unsigned offset_B = block.start_grf * func->shader->devinfo->grf_size +
                             (I->src[0].reg - block.start_gpr) * reg_width;

         b.cursor = jay_before_inst(I);
         jay_inst *add = jay_ADD(&b, JAY_TYPE_U16, a0, I->src[1], offset_B);
         add->uniform = !VxH;
         add->simd_split = I->simd_split;
         add->simd_offs = I->simd_offs;

         I->op = JAY_OPCODE_MOV_INDIRECT;
         I->src[1] = a0;
      } else if (I->op == JAY_OPCODE_SLICE_REPACK) {
         b.cursor = jay_after_inst(I);
         for (unsigned i = 1; i < (1 << jay_slice_repack_factor_log2(I)); ++i) {
            jay_inst *clone = jay_clone_inst(&b, I, I->num_srcs);
            jay_set_slice_repack_index(clone, i);
            jay_builder_insert(&b, clone);
         }
      }
   }
}

JAY_DEFINE_FUNCTION_PASS(jay_lower_simd_width, pass)
