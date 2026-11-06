/*
 * Copyright 2011 Tom Stellard <tstellar@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include "radeon_compiler.h"
#include "radeon_compiler_util.h"
#include "radeon_opcodes.h"
#include "radeon_program_pair.h"

static void
mark_used_presub(struct rc_pair_sub_instruction *sub)
{
   if (sub->Src[RC_PAIR_PRESUB_SRC].Used) {
      unsigned int presub_reg_count =
         rc_presubtract_src_reg_count(sub->Src[RC_PAIR_PRESUB_SRC].Index);
      unsigned int i;
      for (i = 0; i < presub_reg_count; i++) {
         sub->Src[i].Used = 1;
      }
   }
}

static void
mark_used(struct rc_pair_instruction *pair, struct rc_pair_sub_instruction *sub)
{
   unsigned int i;
   const struct rc_opcode_info *info = rc_get_opcode_info(sub->Opcode);
   for (i = 0; i < info->NumSrcRegs; i++) {
      unsigned int src_type = rc_source_type_swz(sub->Arg[i].Swizzle);
      if (src_type & RC_SOURCE_RGB) {
         pair->RGB.Src[sub->Arg[i].Source].Used = 1;
      }

      if (src_type & RC_SOURCE_ALPHA) {
         pair->Alpha.Src[sub->Arg[i].Source].Used = 1;
      }
   }
}

/**
 * Find sources that are not used by the instruction and mark them as unused.
 */
void
rc_pair_update_source_usage(struct rc_pair_instruction *pair)
{
   /* Mark all sources as unused */
   for (unsigned i = 0; i < 4; i++) {
      pair->RGB.Src[i].Used = 0;
      pair->Alpha.Src[i].Used = 0;
   }
   mark_used(pair, &pair->RGB);
   mark_used(pair, &pair->Alpha);

   mark_used_presub(&pair->RGB);
   mark_used_presub(&pair->Alpha);
}

void
rc_pair_remove_dead_sources(struct radeon_compiler *c, void *user)
{
   struct rc_instruction *inst;
   for (inst = c->Program.Instructions.Next; inst != &c->Program.Instructions; inst = inst->Next) {
      if (inst->Type != RC_INSTRUCTION_NORMAL)
         rc_pair_update_source_usage(&inst->U.P);
   }
}
