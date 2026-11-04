/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "jay_builder.h"
#include "jay_builder_opcodes.h"
#include "jay_ir.h"
#include "jay_opcodes.h"
#include "jay_private.h"

/* We reserve an address register for spilling by ABI */
#define ADDRESS_REG jay_bare_reg(J_ADDRESS, 2)

static void
insert_spill_fill(jay_builder *b,
                  jay_def mem,
                  jay_def gpr,
                  jay_def sp,
                  bool load,
                  unsigned *sp_delta_B)
{
   assert(mem.file == MEM && gpr.file != MEM);
   const struct intel_device_info *devinfo = b->shader->devinfo;
   const enum lsc_opcode op = load ? LSC_OP_LOAD : LSC_OP_STORE;
   const enum lsc_addr_surface_type surf_type = LSC_ADDR_SURFTYPE_SS;
   const unsigned max_imm_bits =
      brw_max_immediate_offset_bits(devinfo, surf_type, false, false);

   unsigned offs_B = mem.reg * 4;
   unsigned mem_reg_B = offs_B * b->shader->dispatch_width;

   /* The stack pointer needs to be offset to the desired offset */
   signed sp_adjust_B = mem_reg_B - (*sp_delta_B);

   if (devinfo->ver < 20 ||
       sp_adjust_B < u_intN_min(max_imm_bits) ||
       sp_adjust_B > u_intN_max(max_imm_bits)) {
      /* Fall back to the ALU when the offset overflows the immediate */
      jay_ADD(b, JAY_TYPE_U32, sp, sp, sp_adjust_B);
      *sp_delta_B = mem_reg_B;
      sp_adjust_B = 0;
   }

   unsigned cache = load ? LSC_CACHE(devinfo, LOAD, L1STATE_L3MOCS) :
                           LSC_CACHE(devinfo, STORE, L1STATE_L3MOCS);
   uint64_t desc = lsc_msg_desc(devinfo, op, surf_type, LSC_ADDR_SIZE_A32,
                                LSC_DATA_SIZE_D32, 1, false, cache);
   gen_lsc_ex_desc ex_desc = {
      .addr_type = surf_type,
      .surface_state = {
         .base_offset = sp_adjust_B,
      },
   };
   desc |= (uint64_t) gen_lsc_ex_desc_imm_encode(devinfo, op, &ex_desc) << 32;

   jay_def srcs[] = { sp, gpr };

   jay_SEND(b, .sfid = GEN_SFID_UGM, .msg_desc = desc, .srcs = srcs,
            .nr_srcs = load ? 1 : 2, .dst = load ? gpr : jay_null(),
            .type = JAY_TYPE_U32, .ex_desc = ADDRESS_REG, .bindless = true);
}

static void
lower_mem_copy_swap(jay_function *f)
{
   unsigned nr_mem = 0;
   jay_foreach_inst_in_func(f, block, I) {
      if (I->dst.file == MEM) {
         nr_mem = MAX2(nr_mem, I->dst.reg + 1);
      }
   }

   jay_foreach_inst_in_func_safe(f, block, I) {
      jay_builder b = jay_init_builder(f, jay_before_inst(I));

      jay_def t[] = { jay_bare_reg(MEM, nr_mem + 1),
                      jay_bare_reg(MEM, nr_mem + 2) };
      jay_def r[] = { jay_bare_reg(GPR, 0), jay_bare_reg(GPR, 1) };

      if (I->op == JAY_OPCODE_SWAP && I->src[0].file == MEM) {
         jay_MOV(&b, t[0], r[0]);
         jay_MOV(&b, t[1], r[1]);
         jay_MOV(&b, r[0], I->src[0]);
         jay_MOV(&b, r[1], I->src[1]);
         jay_MOV(&b, I->src[0], r[1]);
         jay_MOV(&b, I->src[1], r[0]);
         jay_MOV(&b, r[0], t[0]);
         jay_MOV(&b, r[1], t[1]);
         jay_remove_instruction(I);
      } else if (I->op == JAY_OPCODE_MOV &&
                 (I->dst.file == MEM && I->src[0].file == MEM)) {

         jay_MOV(&b, t[0], r[0]);
         jay_MOV(&b, r[0], I->src[0]);
         jay_MOV(&b, I->dst, r[0]);
         jay_MOV(&b, r[0], t[0]);
         jay_remove_instruction(I);
      }
   }
}

void
jay_lower_spill(jay_function *func)
{
   /* Lower memory copies/swaps to load/store upfront since this burns more
    * spill slots. Caller doesn't need to worry about it this way.
    */
   lower_mem_copy_swap(func);

   jay_builder b = jay_init_builder(func, jay_before_function(func));
   signed ugpr_reservation = -1;

   /* We reserved a block of UGPRs for our use */
   for (unsigned i = 0; i < func->shader->partition.nr_blocks[UGPR]; ++i) {
      struct jay_register_block B = func->shader->partition.blocks[UGPR][i];

      if (B.type == JAY_BLOCK_SPILL) {
         ugpr_reservation = B.start_gpr;
      }
   }

   assert(ugpr_reservation >= 0 && "must have reserved something");

   jay_def sp_0 = jay_bare_reg(UGPR, ugpr_reservation);
   jay_def sp =
      jay_bare_regs(UGPR, ugpr_reservation, func->shader->dispatch_width);

   /* Calculate how much stack space we need */
   unsigned nr_mem = 0;
   jay_foreach_inst_in_func(func, block, I) {
      if (I->op == JAY_OPCODE_MOV && jay_is_send_like(I)) {
         jay_def mem = I->dst.file == MEM ? I->dst : I->src[0];
         nr_mem = MAX2(nr_mem, mem.reg + 1);
      }
   }

   assert(nr_mem > 0);

   /* We burn the address & stack pointer registers for all spills/fills in a
    * shader. Preinitialize at the top using a scratch register.
    *
    * TODO: Need ABI for multi-function.
    */
   assert(func->is_entrypoint);
   jay_AND(&b, JAY_TYPE_U32, sp_0, jay_bare_reg(UGPR, 5), ~BITFIELD_MASK(10));
   jay_SHR(&b, JAY_TYPE_U32, ADDRESS_REG, sp_0, 4);

   /* We use a 32-bit strided stack: SP = scratch + (lane ID * 4) */
   unsigned disp_width = b.shader->dispatch_width;
   unsigned native_width = 8 * reg_unit(b.shader->devinfo);
   jay_LANE_ID_8(&b, jay_bare_regs(UGPR, ugpr_reservation, 4), 0);

   if (disp_width >= 16) {
      jay_LANE_ID_8(&b, jay_bare_regs(UGPR, ugpr_reservation + 4, 4), 8);
   }

   if (disp_width >= 32) {
      jay_ADD(&b, JAY_TYPE_U16, jay_bare_regs(UGPR, ugpr_reservation + 8, 8),
              jay_bare_regs(UGPR, ugpr_reservation, 8), 16);
   }

   jay_def lid = jay_bare_regs(UGPR, ugpr_reservation, disp_width / 2);
   jay_SHL(&b, JAY_TYPE_U16, lid, lid, util_logbase2(4));

   /* We want to convert U16 -> U32 in place. However, doing so with a
    * compressed SIMD32 instruction has a hazard since the first half's write
    * clobbers the second half's read. Instead, use uncompressed native
    * instructions in reverse-order to avoid all the read-after-write hazards.
    * This emits code like:
    *
    *    (16&W) mov.u32 u127, r126.16:u16
    *    (16&W) mov.u32 u126, r126:u16
    */
   for (signed i = disp_width - native_width; i >= 0; i -= native_width) {
      jay_CVT(&b, JAY_TYPE_U32,
              jay_bare_regs(UGPR, ugpr_reservation + i, native_width),
              jay_bare_regs(UGPR, ugpr_reservation + (i / 2), native_width / 2),
              JAY_TYPE_U16, JAY_ROUND, 0);
   }

   if (b.shader->scratch_size) {
      jay_ADD(&b, JAY_TYPE_U32, sp, sp, b.shader->scratch_size);
   }

   jay_foreach_block(func, block) {
      /* We offset the stack pointer locally within a block to form offsets. By
       * contract keep it in its canonical (unoffset) form at block boundaries.
       */
      unsigned sp_delta_B = 0;
      bool address_valid = true;

      jay_foreach_inst_in_block_safe(block, I) {
         b.cursor = jay_before_inst(I);

         if (I->op == JAY_OPCODE_MOV && jay_is_send_like(I)) {
            if (!address_valid) {
               jay_MOV(&b, ADDRESS_REG, sp_0);
               jay_MOV(&b, sp_0, b.shader->scratch_size + sp_delta_B);
               address_valid = true;
            }

            if (I->dst.file == MEM) {
               insert_spill_fill(&b, I->dst, I->src[0], sp, false, &sp_delta_B);
               func->shader->spills++;
            } else {
               insert_spill_fill(&b, I->src[0], I->dst, sp, true, &sp_delta_B);
               func->shader->fills++;
            }

            jay_remove_instruction(I);
         } else if (address_valid && jay_clobbers_address_reg(I)) {
            /* Shuffles implicitly clobber the address register. Spill it. */
            jay_MOV(&b, sp_0, ADDRESS_REG);
            address_valid = false;
         }
      }

      /* Canonicalize our internal registers at block boundaries */
      if (jay_num_successors(block, GPR) > 0) {
         b.cursor = jay_after_block_logical(block);

         if (!address_valid) {
            jay_MOV(&b, ADDRESS_REG, sp_0);
            jay_MOV(&b, sp_0, b.shader->scratch_size + sp_delta_B);
         }

         if (sp_delta_B > 0) {
            jay_ADD(&b, JAY_TYPE_U32, sp, sp, -sp_delta_B);
         }
      }
   }

   /* Note this is bogus with recursion, but recursion is not supported on any
    * current graphics/compute API.
    */
   func->shader->scratch_size += func->shader->dispatch_width * nr_mem * 4;
}
