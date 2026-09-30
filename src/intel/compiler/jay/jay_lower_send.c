/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "util/macros.h"
#include "jay_builder.h"
#include "jay_builder_opcodes.h"
#include "jay_ir.h"
#include "jay_opcodes.h"
#include "jay_private.h"

static void
lower_send_desc_mlen(jay_function *func,
                     jay_inst *I,
                     struct util_dynarray *addr_desc_imm)
{
   if (I->op == JAY_OPCODE_PACK_MLEN) {
      assert(jay_index(I->dst) != JAY_SENTINEL);
      assert(I->dst.file == J_ADDRESS);

      /* Add the instruction to the array for quick access */
      util_dynarray_append(addr_desc_imm, I);

   } else if (I->op == JAY_OPCODE_SEND) {
      jay_send_info *info = jay_get_send_info(I);
      unsigned simd = func->shader->dispatch_width >> I->simd_split;

      /* For message headers we pack a UGPR vector as a single GRF */
      unsigned len[3];
      for (unsigned i = 0; i < 3; ++i) {
         jay_def x = i == 0 ? I->dst : I->src[1 + i];
         len[i] = jay_num_values(x);

         if (x.file == UGPR) {
            len[i] = DIV_ROUND_UP(len[i], jay_ugpr_per_grf(func->shader));
         } else {
            len[i] *= jay_grf_per_gpr_at_width(func->shader, simd);
         }
      }

      /* Combine the message lengths with the other immediate bits */
      info->desc_imm |=
         gen_message_desc_encode(func->shader->devinfo, &(gen_message_desc) {
            .response_length = len[0],
            .msg_length = len[1],
            .header_present = info->has_header,
         });
      info->ex_desc_imm |=
         gen_message_ex_desc(func->shader->devinfo, len[2]);

      for (unsigned i = 0; i < 2; ++i) {
         if (i != 0 && jay_send_bindless(I)) {
            /* We have both an address register and an immediate for extended
             * descriptor, the ex_mlen is packed into the immediate and we can
             * leave the address register alone.
             */
            assert(I->src[i].file == J_ADDRESS);
            continue;
         }

         uint32_t bits = i != 0 ? info->ex_desc_imm : info->desc_imm;

         if (jay_is_null(I->src[i])) {
            /* The whole descriptor is an immediate */
            I->src[i] = jay_imm(bits);
         } else if (jay_index(I->src[i]) != JAY_SENTINEL) {
            /* A portion of the descriptor is dynamic and gets OR'd with the
             * immediate portion by a PACK_MLEN instruction.
             */
            assert(I->src[i].file == J_ADDRESS);
            jay_inst *pack_mlen = NULL;

            util_dynarray_foreach(addr_desc_imm, jay_inst *, etr) {
               if (jay_index((*etr)->dst) == jay_index(I->src[i])) {
                  pack_mlen = *etr;
                  util_dynarray_delete_unordered_ptr(addr_desc_imm,
                                                     jay_inst *, etr);
                  break;
               }
            }

            assert(pack_mlen && "should've found something");
            assert(jay_is_null(pack_mlen->src[1]));

            /* Convert the PACK_MLEN to an instruction that ORs the dynamic and
             * immediate portions and stores it in the register.
             *
             * XXX: This obviously produces some duplicated instructions
             */
            if (jay_is_imm(pack_mlen->src[0])) {
               bits |= jay_as_uint(pack_mlen->src[0]);
               pack_mlen->op = JAY_OPCODE_MOV;
               pack_mlen->src[0] = jay_imm(bits);
               jay_shrink_isa_sources(pack_mlen, 1);
            } else if (bits) {
               pack_mlen->op = JAY_OPCODE_OR;
               pack_mlen->src[1] = jay_imm(bits);
            } else {
               pack_mlen->op = JAY_OPCODE_MOV;
               jay_shrink_isa_sources(pack_mlen, 1);
            }
         }
      }
   }
}

static void
pass(jay_function *func)
{
   struct util_dynarray addr_desc_imm = { 0 };

   jay_foreach_block(func, block) {
      jay_foreach_inst_in_block_safe(block, I) {
         lower_send_desc_mlen(func, I, &addr_desc_imm);
      }
   }

   assert(!addr_desc_imm.size);
   util_dynarray_fini(&addr_desc_imm);
}

JAY_DEFINE_FUNCTION_PASS(jay_lower_send, pass)
