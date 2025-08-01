/*
 * Copyright 2021 Collabora LTD
 * Author: Gert Wollny <gert.wollny@collabora.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef SFN_GEOMETRYSHADER_H
#define SFN_GEOMETRYSHADER_H

#include "sfn_instr_export.h"
#include "sfn_shader.h"

namespace r600 {

class GeometryShader : public Shader {
public:
   GeometryShader(const r600_shader_key& key);

private:
   bool do_scan_instruction(nir_instr *instr) override;
   int do_allocate_reserved_registers() override;

   bool process_stage_intrinsic(nir_intrinsic_instr *intr) override;

   bool process_store_output(nir_intrinsic_instr *intr);
   bool process_load_input(nir_intrinsic_instr *intr);

   void do_finalize() override;

   void do_get_shader_info(r600_shader *sh_info) override;

   bool read_prop(std::istream& is) override;
   void do_print_properties(std::ostream& os) const override;

   void emit_adj_fix();

   bool emit_indirect_vertex_at_index(nir_intrinsic_instr *instr);

   bool emit_load_per_vertex_input_direct(nir_intrinsic_instr *instr);

   bool emit_load_per_vertex_input_indirect(nir_intrinsic_instr *instr);

   bool load_per_vertex_input_at_addr(nir_intrinsic_instr *instr, PRegister addr);

   bool load_input(UNUSED nir_intrinsic_instr *intr) override
   {
      UNREACHABLE("load_input must be lowered in GS");
   };
   bool store_output(nir_intrinsic_instr *instr) override;
   bool emit_vertex(nir_intrinsic_instr *instr, bool cut);

   std::array<PRegister, R600_GS_VERTEX_INDIRECT_TOTAL> m_per_vertex_offsets{nullptr};
   PRegister m_primitive_id{nullptr};
   PRegister m_invocation_id{nullptr};
   std::array<PRegister, 4> m_export_base{nullptr};

   unsigned m_ring_item_sizes[4]{0};

   bool m_tri_strip_adj_fix{false};
   int m_next_input_ring_offset{0};
   int m_cc_dist_mask{0};
   int m_clip_dist_write{0};
   uint64_t m_input_mask{0};
   unsigned m_noutputs{0};
   bool m_out_viewport{false};
   bool m_out_misc_write{false};

   std::map<int, MemRingOutInstr *> m_streamout_data;
};

} // namespace r600

#endif // GEOMETRYSHADER_H
