/*
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

TEST(RegisterAllocate, ChainedPhiEviction)
{
   void *mem_ctx = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem_ctx);
   agx_context *ctx = b->shader;
   struct agx_shader_key key = {};
   nir_shader nir = {};

   /* Limit allocation to 32 half-registers, with r0l reserved for control flow. */
   key.is_helper = true;
   nir.info.stage = MESA_SHADER_COMPUTE;
   nir.info.workgroup_size[0] = 1;
   nir.info.workgroup_size[1] = 1;
   nir.info.workgroup_size[2] = 1;
   ctx->nir = &nir;
   ctx->key = &key;
   ctx->stage = MESA_SHADER_COMPUTE;
   ctx->any_cf = true;

   agx_block *entry = agx_start_block(ctx);
   agx_block *left = agx_test_block(ctx);
   agx_block *right = agx_test_block(ctx);
   agx_block *merge = agx_test_block(ctx);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, merge);
   agx_block_add_successor(right, merge);

   agx_index values[AGX_NUM_REGS] = {};
   for (unsigned r = 2; r < 32; r += 2) {
      if (r == 6)
         continue;

      enum agx_size size = r == 8 ? AGX_SIZE_16 : AGX_SIZE_32;
      values[r] = agx_preload(b, agx_register(r, size));
   }

   b->cursor = agx_after_block(merge);
   agx_index value = agx_temp(ctx, AGX_SIZE_64);
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = agx_immediate(0);
   phi->src[1] = agx_immediate(0);
   agx_unit_test(b, value);

   agx_instr *uses[AGX_NUM_REGS] = {};
   for (unsigned r = 2; r < 32; r += 2) {
      if (r != 6)
         uses[r] = agx_unit_test(b, values[r]);
   }

   /* The 64-bit phi needs h4..h7. Evict the 32-bit value at h4 to h8,
    * which first requires moving the 16-bit value originally at h8 to h1.
    * Updating the first phi too early makes the second move match it again.
    */
   agx_ra(ctx);

   /* Check the data carried by both incoming edges, independently of the
    * chosen registers. Each initial half-register contains its own number.
    */
   for (agx_block *pred : {left, right}) {
      unsigned regs[AGX_NUM_REGS];
      for (unsigned r = 0; r < AGX_NUM_REGS; ++r)
         regs[r] = r;

      agx_foreach_instr_in_block(pred, I) {
         if (I->op == AGX_OPCODE_MOV) {
            ASSERT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
            unsigned src = I->src[0].value;
            unsigned dst = I->dest[0].value;
            unsigned saved[4];
            unsigned count = agx_index_size_16(I->dest[0]);
            ASSERT_LE(count, 4u);
            for (unsigned i = 0; i < count; ++i)
               saved[i] = regs[src + i];
            for (unsigned i = 0; i < count; ++i)
               regs[dst + i] = saved[i];
         } else if (I->op == AGX_OPCODE_MOV_IMM) {
            for (unsigned i = 0; i < agx_index_size_16(I->dest[0]); ++i)
               regs[I->dest[0].value + i] = 0;
         } else {
            FAIL() << "Unexpected instruction in phi copies";
         }
      }

      for (unsigned r = 2; r < 32; r += 2) {
         if (!uses[r])
            continue;

         agx_index src = uses[r]->src[0];
         ASSERT_EQ(src.type, AGX_INDEX_REGISTER);
         for (unsigned i = 0; i < agx_index_size_16(src); ++i)
            EXPECT_EQ(regs[src.value + i], r + i);
      }
   }

   ralloc_free(mem_ctx);
}
