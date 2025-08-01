/*
 * Copyright © 2021 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "aco_ir.h"

#include <vector>

namespace aco {
namespace {

struct idx_ctx {
   std::vector<RegClass> temp_rc = {s1};
   std::vector<uint32_t> renames;
};

inline void
reindex_defs(idx_ctx& ctx, aco_ptr<Instruction>& instr)
{
   for (Definition& def : instr->definitions) {
      if (!def.isTemp())
         continue;
      uint32_t new_id = ctx.temp_rc.size();
      RegClass rc = def.regClass();
      ctx.renames[def.tempId()] = new_id;
      ctx.temp_rc.emplace_back(rc);
      def.setTemp(Temp(new_id, rc));
   }
}

inline void
reindex_ops(idx_ctx& ctx, aco_ptr<Instruction>& instr)
{
   for (Operand& op : instr->operands) {
      if (!op.isTemp())
         continue;
      uint32_t new_id = ctx.renames[op.tempId()];
      assert(op.regClass() == ctx.temp_rc[new_id]);
      op.setTemp(Temp(new_id, op.regClass()));
   }
}

void
reindex_program(idx_ctx& ctx, Program* program)
{
   ctx.renames.resize(program->peekAllocationId());

   for (Block& block : program->blocks) {
      auto it = block.instructions.begin();
      /* for phis, only reindex the definitions */
      while (is_phi(*it)) {
         reindex_defs(ctx, *it++);
      }
      /* reindex all other instructions */
      while (it != block.instructions.end()) {
         reindex_defs(ctx, *it);
         reindex_ops(ctx, *it);
         ++it;
      }
   }
   /* update the phi operands */
   for (Block& block : program->blocks) {
      auto it = block.instructions.begin();
      while (is_phi(*it)) {
         reindex_ops(ctx, *it++);
      }
   }

   /* update program members */
   for (auto& private_segment_buffer : program->private_segment_buffers) {
      private_segment_buffer =
         Temp(ctx.renames[private_segment_buffer.id()], private_segment_buffer.regClass());
   }
   for (auto& scratch_offset : program->scratch_offsets) {
      scratch_offset =
         Temp(ctx.renames[scratch_offset.id()], scratch_offset.regClass());
   }
   program->stack_ptr = Temp(ctx.renames[program->stack_ptr.id()], program->stack_ptr.regClass());
   program->static_scratch_rsrc =
      Temp(ctx.renames[program->static_scratch_rsrc.id()], program->static_scratch_rsrc.regClass());
   program->temp_rc = ctx.temp_rc;
}

} /* end namespace */

void
reindex_ssa(Program* program)
{
   idx_ctx ctx;
   reindex_program(ctx, program);

   monotonic_buffer_resource old_memory = std::move(program->live.memory);
   for (IDSet& set : program->live.live_in) {
      IDSet new_set(program->live.memory);
      for (uint32_t id : set)
         new_set.insert(ctx.renames[id]);
      set = std::move(new_set);
   }
}

} // namespace aco
