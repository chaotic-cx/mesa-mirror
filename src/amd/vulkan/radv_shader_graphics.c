/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_shader_graphics.h"
#include "nir/nir.h"
#include "nir/nir_xfb_info.h"
#include "nir/radv_nir.h"
#include "tools/radv_debug_hang.h"
#include "util/os_time.h"
#include "radv_aco_shader_info.h"
#include "radv_device.h"
#include "radv_formats.h"
#include "radv_instance.h"
#include "radv_physical_device.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "vk_graphics_state.h"

#include "ac_formats.h"
#include "ac_nir.h"
#include "ac_shader_util.h"
#include "aco_interface.h"
#include "sid.h"

#if AMD_LLVM_AVAILABLE
#include "ac_llvm_util.h"
#endif

bool
radv_consider_culling(const struct radv_compiler_info *compiler_info, struct nir_shader *nir, uint64_t ps_inputs_read,
                      unsigned num_vertices_per_primitive, const struct radv_shader_info *info,
                      const struct radv_graphics_state_key *gfx_state)
{
   /* Culling doesn't make sense for meta shaders. */
   if (radv_is_meta_shader(nir))
      return false;

   /* We don't support culling with vertex shader prologs. */
   if (info->vs.has_prolog)
      return false;

   if (!compiler_info->key.use_ngg_culling)
      return false;

   if (gfx_state->rs.skip_all_ngg_culling)
      return false;

   /* TODO: consider other heuristics here, such as PS execution time */
   assert(compiler_info->key.nggc_max_ps_params);
   if (util_bitcount64(ps_inputs_read) > compiler_info->key.nggc_max_ps_params)
      return false;

   /* Only triangle culling is supported. */
   if (num_vertices_per_primitive != 3)
      return false;

   /* When the shader writes memory, it is difficult to guarantee correctness.
    * Future work:
    * - if only write-only SSBOs are used
    * - if we can prove that non-position outputs don't rely on memory stores
    * then may be okay to keep the memory stores in the 1st shader part, and delete them from the 2nd.
    */
   if (nir->info.writes_memory)
      return false;

   /* NGG lowering exports 0 primitives and vertices with rasterizer discard. There's nothing
    * to cull.
    */
   if (gfx_state->rs.rasterizer_discard)
      return false;

   /* When the shader relies on the subgroup invocation ID, we'd break it, because the ID changes after the culling.
    * Future work: try to save this to LDS and reload, but it can still be broken in subtle ways.
    */
   if (BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_SUBGROUP_INVOCATION))
      return false;

   /* When re-using values that depend on subgroup operations, we'd break convergence guarantees.
    * Since we only re-use uniform values, the only subgroup operations we really care about are
    * ballot, reductions and vote intrinsics.
    */
   if (nir->info.maximally_reconverges && nir->info.uses_wide_subgroup_intrinsics)
      return false;

   return true;
}

void
radv_lower_ngg(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *ngg_stage,
               const struct radv_graphics_state_key *gfx_state)
{
   const struct radv_shader_info *info = &ngg_stage->info;
   nir_shader *nir = ngg_stage->nir;

   assert(nir->info.stage == MESA_SHADER_VERTEX || nir->info.stage == MESA_SHADER_TESS_EVAL ||
          nir->info.stage == MESA_SHADER_GEOMETRY || nir->info.stage == MESA_SHADER_MESH);

   unsigned num_vertices_per_prim = 3;

   /* Get the number of vertices per input primitive */
   if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      if (nir->info.tess.point_mode)
         num_vertices_per_prim = 1;
      else if (nir->info.tess._primitive_mode == TESS_PRIMITIVE_ISOLINES)
         num_vertices_per_prim = 2;

      /* Manually mark the primitive ID used, so the shader can repack it. */
      if (info->outinfo.export_prim_id)
         BITSET_SET(nir->info.system_values_read, SYSTEM_VALUE_PRIMITIVE_ID);

   } else if (nir->info.stage == MESA_SHADER_VERTEX) {
      num_vertices_per_prim = radv_get_num_vertices_per_prim(compiler_info->ac->gfx_level, gfx_state);

      /* Manually mark the instance ID used, so the shader can repack it. */
      if (gfx_state->vi.instance_rate_inputs)
         BITSET_SET(nir->info.system_values_read, SYSTEM_VALUE_INSTANCE_ID);

   } else if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      num_vertices_per_prim = nir->info.gs.vertices_in;
   } else if (nir->info.stage == MESA_SHADER_MESH) {
      if (nir->info.mesh.primitive_type == MESA_PRIM_POINTS)
         num_vertices_per_prim = 1;
      else if (nir->info.mesh.primitive_type == MESA_PRIM_LINES)
         num_vertices_per_prim = 2;
      else
         assert(nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES);
   } else {
      UNREACHABLE("NGG needs to be VS, TES or GS.");
   }

   ac_nir_lower_ngg_options options = {0};
   options.compiler_info = compiler_info->ac;
   options.max_workgroup_size = info->workgroup_size;
   options.wave_size = info->wave_size;
   options.export_clipdist_mask = info->outinfo.clip_dist_mask | info->outinfo.cull_dist_mask;
   options.cull_clipdist_mask = options.export_clipdist_mask;
   options.vs_output_param_offset = info->outinfo.vs_output_param_offset;
   options.has_param_exports = info->outinfo.param_exports || info->outinfo.prim_param_exports;
   options.can_cull = info->has_ngg_culling;
   options.disable_streamout = compiler_info->ac->gfx_level < GFX11;
   options.has_xfb_prim_query = info->has_xfb_query;
   options.has_gs_primitives_query = compiler_info->ac->gfx_level < GFX11;
   options.force_vrs = info->force_vrs_per_vertex;
   options.skip_face_culling = gfx_state->rs.skip_ngg_cull_face;
   options.rasterizer_discard = gfx_state->rs.rasterizer_discard;

   if (nir->info.stage == MESA_SHADER_VERTEX || nir->info.stage == MESA_SHADER_TESS_EVAL) {
      assert(info->is_ngg);

      if (info->has_ngg_culling)
         radv_optimize_nir_algebraic_late(nir);

      options.has_gen_prim_query = info->has_prim_query;
      options.num_vertices_per_primitive = num_vertices_per_prim;
      options.early_prim_export = info->has_ngg_early_prim_export;
      options.passthrough = info->is_ngg_passthrough;
      options.export_primitive_id = info->outinfo.export_prim_id;
      options.export_primitive_id_per_prim = info->outinfo.export_prim_id_per_primitive;
      options.instance_rate_inputs = gfx_state->vi.instance_rate_inputs << VERT_ATTRIB_GENERIC0;

      NIR_PASS(_, nir, ac_nir_lower_ngg_nogs, &options, &ngg_stage->info.ngg_lds_vertex_size,
               &ngg_stage->info.ngg_lds_scratch_size);
   } else if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      assert(info->is_ngg);

      options.has_gen_prim_query = info->has_prim_query;
      options.has_ms_gs_invocations_query = compiler_info->ac->gfx_level < GFX11;

      NIR_PASS(_, nir, ac_nir_lower_ngg_gs, &options, &ngg_stage->info.ngg_lds_vertex_size,
               &ngg_stage->info.ngg_lds_scratch_size);
   } else if (nir->info.stage == MESA_SHADER_MESH) {
      /* ACO aligns the workgroup size to the wave size. */
      options.max_workgroup_size = align(info->workgroup_size, info->wave_size);

      options.has_gen_prim_query = info->ms.has_query;
      options.has_ms_gs_invocations_query = info->ms.has_query;
      options.multiview = ngg_stage->key.multiview;

      bool scratch_ring = false;
      NIR_PASS(_, nir, ac_nir_lower_ngg_mesh, &options, &scratch_ring);
      ngg_stage->info.ms.needs_ms_scratch_ring = scratch_ring;
      ngg_stage->info.ngg_wave_id_en = scratch_ring;
   } else {
      UNREACHABLE("invalid SW stage passed to radv_lower_ngg");
   }
}

static unsigned
radv_get_num_pos_exports(const struct radv_shader_info *info, unsigned *clip_dist_mask, unsigned *cull_dist_mask)
{
   unsigned num = 1;

   if (info->outinfo.writes_pointsize || info->outinfo.writes_viewport_index || info->outinfo.writes_layer ||
       info->outinfo.writes_primitive_shading_rate)
      num++;

   /* Clip and cull distances are packed by ac_nir_export_position. */
   unsigned num_clip_dist_comps = util_bitcount(info->outinfo.clip_dist_mask);
   /* Cull distances are not exported if the shader culls against them. */
   unsigned num_cull_dist_comps = info->has_ngg_culling ? 0 : util_bitcount(info->outinfo.cull_dist_mask);
   unsigned clip_cull_mask = BITFIELD_MASK(num_clip_dist_comps + num_cull_dist_comps);

   if (clip_cull_mask & 0x0f)
      num++;
   if (clip_cull_mask & 0xf0)
      num++;

   *clip_dist_mask = BITFIELD_MASK(num_clip_dist_comps);
   *cull_dist_mask = BITFIELD_RANGE(num_clip_dist_comps, num_cull_dist_comps);
   return num;
}

void
radv_precompute_registers_hw_vs(struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader_info *info = &shader->info;
   struct radv_shader_regs *regs = &shader->regs;
   unsigned clip_dist_mask, cull_dist_mask;
   unsigned num_pos_exports = radv_get_num_pos_exports(info, &clip_dist_mask, &cull_dist_mask);

   /* VS is required to export at least one param. */
   const uint32_t nparams = MAX2(info->outinfo.param_exports, 1);
   regs->spi_vs_out_config = S_0286C4_VS_EXPORT_COUNT(nparams - 1);
   if (pdev->info.gfx_level >= GFX10) {
      regs->spi_vs_out_config |= S_0286C4_NO_PC_EXPORT(info->outinfo.param_exports == 0);
   }

   regs->spi_shader_pos_format =
      S_02870C_POS0_EXPORT_FORMAT(V_02870C_SPI_SHADER_4COMP) |
      S_02870C_POS1_EXPORT_FORMAT(num_pos_exports > 1 ? V_02870C_SPI_SHADER_4COMP : V_02870C_SPI_SHADER_NONE) |
      S_02870C_POS2_EXPORT_FORMAT(num_pos_exports > 2 ? V_02870C_SPI_SHADER_4COMP : V_02870C_SPI_SHADER_NONE) |
      S_02870C_POS3_EXPORT_FORMAT(num_pos_exports > 3 ? V_02870C_SPI_SHADER_4COMP : V_02870C_SPI_SHADER_NONE);

   const bool misc_vec_ena = info->outinfo.writes_pointsize || info->outinfo.writes_layer ||
                             info->outinfo.writes_viewport_index || info->outinfo.writes_primitive_shading_rate;
   const unsigned total_mask = clip_dist_mask | cull_dist_mask;

   regs->pa_cl_vs_out_cntl =
      S_02881C_USE_VTX_POINT_SIZE(info->outinfo.writes_pointsize) |
      S_02881C_USE_VTX_RENDER_TARGET_INDX(info->outinfo.writes_layer) |
      S_02881C_USE_VTX_VIEWPORT_INDX(info->outinfo.writes_viewport_index) |
      S_02881C_USE_VTX_VRS_RATE(info->outinfo.writes_primitive_shading_rate) |
      S_02881C_VS_OUT_MISC_VEC_ENA(misc_vec_ena) |
      S_02881C_VS_OUT_MISC_SIDE_BUS_ENA(misc_vec_ena || (pdev->info.gfx_level >= GFX10_3 && num_pos_exports > 1)) |
      S_02881C_VS_OUT_CCDIST0_VEC_ENA((total_mask & 0x0f) != 0) |
      S_02881C_VS_OUT_CCDIST1_VEC_ENA((total_mask & 0xf0) != 0) | total_mask << 8 | clip_dist_mask;

   if (pdev->info.gfx_level <= GFX8)
      regs->vs.vgt_reuse_off = info->outinfo.writes_viewport_index;

   unsigned late_alloc_wave64, cu_mask;
   ac_compute_late_alloc(&pdev->info, false, false, shader->config.scratch_bytes_per_wave > 0, &late_alloc_wave64,
                         &cu_mask);

   if (pdev->info.gfx_level >= GFX7) {
      regs->vs.spi_shader_pgm_rsrc3_vs =
         ac_apply_cu_en(S_00B118_CU_EN(cu_mask) | S_00B118_WAVE_LIMIT(0x3F), C_00B118_CU_EN, 0, &pdev->info);
      regs->vs.spi_shader_late_alloc_vs = S_00B11C_LIMIT(late_alloc_wave64);

      if (pdev->info.gfx_level >= GFX10) {
         const uint32_t oversub_pc_lines = late_alloc_wave64 ? pdev->info.pc_lines / 4 : 0;

         regs->ge_pc_alloc = S_030980_OVERSUB_EN(oversub_pc_lines > 0) | S_030980_NUM_PC_LINES(oversub_pc_lines - 1);

         /* Required programming for tessellation (legacy GS only). */
         if (shader->info.stage == MESA_SHADER_TESS_EVAL) {
            regs->vgt_gs_onchip_cntl = S_028A44_ES_VERTS_PER_SUBGRP(250) | S_028A44_GS_PRIMS_PER_SUBGRP(126) |
                                       S_028A44_GS_INST_PRIMS_IN_SUBGRP(126);
         }
      }
   }
}

void
radv_precompute_registers_hw_hs(struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_shader_regs *regs = &shader->regs;

   if (pdev->info.gfx_level < GFX11)
      return;

   unsigned inst_pref_size = radv_get_inst_pref_size(&device->compiler_info, shader->exec_size);

   if (pdev->info.gfx_level >= GFX12) {
      regs->spi_shader_pgm_rsrc4_gs_hs =
         S_00B420_WAVE_LIMIT(0x3ff) | S_00B420_GLG_FORCE_DISABLE(1) | S_00B420_INST_PREF_SIZE(inst_pref_size);
   } else {
      regs->spi_shader_pgm_rsrc4_gs_hs = ac_apply_cu_en(
         S_00B404_INST_PREF_SIZE(inst_pref_size) | S_00B404_CU_EN(0xffff), C_00B404_CU_EN, 16, &pdev->info);
   }
}

void
radv_precompute_registers_hw_gs(struct radv_device *device, const struct radv_shader_info *es_info,
                                struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader_info *gs_info = &shader->info;
   struct radv_shader_regs *regs = &shader->regs;

   regs->gs.vgt_esgs_ring_itemsize = es_info ? es_info->esgs_itemsize / 4 : gs_info->legacy_gs_info.esgs_itemsize / 4;

   regs->gs.vgt_gs_max_prims_per_subgroup =
      S_028A94_MAX_PRIMS_PER_SUBGROUP(gs_info->legacy_gs_info.gs_inst_prims_in_subgroup);

   regs->vgt_gs_onchip_cntl = S_028A44_ES_VERTS_PER_SUBGRP(gs_info->legacy_gs_info.es_verts_per_subgroup) |
                              S_028A44_GS_PRIMS_PER_SUBGRP(gs_info->legacy_gs_info.gs_prims_per_subgroup) |
                              S_028A44_GS_INST_PRIMS_IN_SUBGRP(gs_info->legacy_gs_info.gs_inst_prims_in_subgroup);

   const uint32_t gs_max_out_vertices = gs_info->gs.vertices_out;
   const uint8_t max_stream = gs_info->gs.num_components_per_stream[3]   ? 3
                              : gs_info->gs.num_components_per_stream[2] ? 2
                              : gs_info->gs.num_components_per_stream[1] ? 1
                                                                         : 0;
   const uint8_t *num_components = gs_info->gs.num_components_per_stream;

   uint32_t offset = num_components[0] * gs_max_out_vertices;
   regs->gs.vgt_gsvs_ring_offset[0] = offset;

   if (max_stream >= 1)
      offset += num_components[1] * gs_max_out_vertices;
   regs->gs.vgt_gsvs_ring_offset[1] = offset;

   if (max_stream >= 2)
      offset += num_components[2] * gs_max_out_vertices;
   regs->gs.vgt_gsvs_ring_offset[2] = offset;

   if (max_stream >= 3)
      offset += num_components[3] * gs_max_out_vertices;
   regs->gs.vgt_gsvs_ring_itemsize = offset;

   for (uint32_t i = 0; i < 4; i++)
      regs->gs.vgt_gs_vert_itemsize[i] = (max_stream >= i) ? num_components[i] : 0;

   const uint32_t gs_num_invocations = gs_info->gs.invocations;
   regs->gs.vgt_gs_instance_cnt = S_028B90_CNT(MIN2(gs_num_invocations, 127)) | S_028B90_ENABLE(gs_num_invocations > 0);

   regs->spi_shader_pgm_rsrc3_gs =
      ac_apply_cu_en(S_00B21C_CU_EN(0xffff) | S_00B21C_WAVE_LIMIT(0x3F), C_00B21C_CU_EN, 0, &pdev->info);

   if (pdev->info.gfx_level >= GFX10) {
      regs->spi_shader_pgm_rsrc4_gs_hs =
         ac_apply_cu_en(S_00B204_CU_EN_GFX10(0xffff) | S_00B204_SPI_SHADER_LATE_ALLOC_GS_GFX10(0), C_00B204_CU_EN_GFX10,
                        16, &pdev->info);
   }

   regs->vgt_gs_max_vert_out = gs_info->gs.vertices_out;

   radv_get_esgs_gsvs_ring_size(device, regs, NULL, gs_info);
}

void
radv_precompute_registers_hw_ngg(struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader_info *info = &shader->info;
   const struct ac_shader_config *config = &shader->config;
   struct radv_shader_regs *regs = &shader->regs;

   const bool no_pc_export = info->outinfo.param_exports == 0 && info->outinfo.prim_param_exports == 0;
   const unsigned num_prim_params = info->outinfo.prim_param_exports;

   if (pdev->info.gfx_level >= GFX12) {
      const unsigned num_params = MAX2(info->outinfo.param_exports, 1);
      unsigned inst_pref_size = radv_get_inst_pref_size(&device->compiler_info, shader->exec_size);

      regs->spi_vs_out_config = S_00B0C4_VS_EXPORT_COUNT(num_params - 1) | S_00B0C4_PRIM_EXPORT_COUNT(num_prim_params) |
                                S_00B0C4_NO_PC_EXPORT(no_pc_export);

      regs->spi_shader_pgm_rsrc4_gs_hs = S_00B220_SPI_SHADER_LATE_ALLOC_GS(127) | S_00B220_GLG_FORCE_DISABLE(1) |
                                         S_00B220_WAVE_LIMIT(0x3ff) | S_00B220_INST_PREF_SIZE(inst_pref_size);
   } else {
      const unsigned num_params = MAX2(info->outinfo.param_exports, 1);

      regs->spi_vs_out_config = S_0286C4_VS_EXPORT_COUNT(num_params - 1) | S_0286C4_PRIM_EXPORT_COUNT(num_prim_params) |
                                S_0286C4_NO_PC_EXPORT(no_pc_export);

      unsigned late_alloc_wave64, cu_mask;
      ac_compute_late_alloc(&pdev->info, true, info->has_ngg_culling, config->scratch_bytes_per_wave > 0,
                            &late_alloc_wave64, &cu_mask);

      regs->spi_shader_pgm_rsrc3_gs =
         ac_apply_cu_en(S_00B21C_CU_EN(cu_mask) | S_00B21C_WAVE_LIMIT(0x3F), C_00B21C_CU_EN, 0, &pdev->info);

      if (pdev->info.gfx_level >= GFX11) {
         unsigned inst_pref_size = radv_get_inst_pref_size(&device->compiler_info, shader->exec_size);

         regs->spi_shader_pgm_rsrc4_gs_hs =
            ac_apply_cu_en(S_00B204_CU_EN_GFX11(0x1) | S_00B204_SPI_SHADER_LATE_ALLOC_GS_GFX10(late_alloc_wave64) |
                              S_00B204_INST_PREF_SIZE(inst_pref_size),
                           C_00B204_CU_EN_GFX11, 16, &pdev->info);
      } else {
         regs->spi_shader_pgm_rsrc4_gs_hs =
            ac_apply_cu_en(S_00B204_CU_EN_GFX10(0xffff) | S_00B204_SPI_SHADER_LATE_ALLOC_GS_GFX10(late_alloc_wave64),
                           C_00B204_CU_EN_GFX10, 16, &pdev->info);
      }

      uint32_t oversub_pc_lines = late_alloc_wave64 ? pdev->info.pc_lines / 4 : 0;
      if (info->has_ngg_culling) {
         unsigned oversub_factor = 2;

         if (info->outinfo.param_exports > 4)
            oversub_factor = 4;
         else if (info->outinfo.param_exports > 2)
            oversub_factor = 3;

         oversub_pc_lines *= oversub_factor;
      }

      regs->ge_pc_alloc = S_030980_OVERSUB_EN(oversub_pc_lines > 0) | S_030980_NUM_PC_LINES(oversub_pc_lines - 1);
   }

   unsigned idx_format = V_028708_SPI_SHADER_1COMP;
   if (info->outinfo.writes_layer_per_primitive || info->outinfo.writes_viewport_index_per_primitive ||
       info->outinfo.writes_primitive_shading_rate_per_primitive)
      idx_format = V_028708_SPI_SHADER_2COMP;
   unsigned clip_dist_mask, cull_dist_mask;
   unsigned num_pos_exports = radv_get_num_pos_exports(info, &clip_dist_mask, &cull_dist_mask);

   regs->ngg.spi_shader_idx_format = S_028708_IDX0_EXPORT_FORMAT(idx_format);

   regs->spi_shader_pos_format =
      S_02870C_POS0_EXPORT_FORMAT(V_02870C_SPI_SHADER_4COMP) |
      S_02870C_POS1_EXPORT_FORMAT(num_pos_exports > 1 ? V_02870C_SPI_SHADER_4COMP : V_02870C_SPI_SHADER_NONE) |
      S_02870C_POS2_EXPORT_FORMAT(num_pos_exports > 2 ? V_02870C_SPI_SHADER_4COMP : V_02870C_SPI_SHADER_NONE) |
      S_02870C_POS3_EXPORT_FORMAT(num_pos_exports > 3 ? V_02870C_SPI_SHADER_4COMP : V_02870C_SPI_SHADER_NONE);

   const bool misc_vec_ena = info->outinfo.writes_pointsize || info->outinfo.writes_layer ||
                             info->outinfo.writes_viewport_index || info->outinfo.writes_primitive_shading_rate;
   const unsigned total_mask = clip_dist_mask | cull_dist_mask;

   regs->pa_cl_vs_out_cntl =
      S_02881C_USE_VTX_POINT_SIZE(info->outinfo.writes_pointsize) |
      S_02881C_USE_VTX_RENDER_TARGET_INDX(info->outinfo.writes_layer) |
      S_02881C_USE_VTX_VIEWPORT_INDX(info->outinfo.writes_viewport_index) |
      S_02881C_USE_VTX_VRS_RATE(info->outinfo.writes_primitive_shading_rate) |
      S_02881C_VS_OUT_MISC_VEC_ENA(misc_vec_ena) |
      S_02881C_VS_OUT_MISC_SIDE_BUS_ENA(misc_vec_ena || (pdev->info.gfx_level >= GFX10_3 && num_pos_exports > 1)) |
      S_02881C_VS_OUT_CCDIST0_VEC_ENA((total_mask & 0x0f) != 0) |
      S_02881C_VS_OUT_CCDIST1_VEC_ENA((total_mask & 0xf0) != 0) | total_mask << 8 | clip_dist_mask;

   regs->ngg.vgt_primitiveid_en =
      S_028A84_NGG_DISABLE_PROVOK_REUSE(info->stage == MESA_SHADER_VERTEX && info->outinfo.export_prim_id);

   const uint32_t gs_num_invocations = info->stage == MESA_SHADER_GEOMETRY ? info->gs.invocations : 1;

   regs->ngg.ge_max_output_per_subgroup = S_0287FC_MAX_VERTS_PER_SUBGROUP(info->ngg_info.max_out_verts);

   regs->ngg.ge_ngg_subgrp_cntl =
      S_028B4C_PRIM_AMP_FACTOR(info->ngg_info.prim_amp_factor) | S_028B4C_THDS_PER_SUBGRP(0); /* for fast launch */

   regs->vgt_gs_instance_cnt = S_028B90_CNT(gs_num_invocations) | S_028B90_ENABLE(gs_num_invocations > 1) |
                               S_028B90_EN_MAX_VERT_OUT_PER_GS_INSTANCE(info->ngg_info.max_vert_out_per_gs_instance);

   if (pdev->info.gfx_level >= GFX11) {
      /* This should be <= 252 for performance on Gfx11. 256 works too but is slower. */
      const uint32_t max_prim_grp_size = pdev->info.gfx_level >= GFX12 ? 256 : 252;

      regs->ngg.ge_cntl = S_03096C_PRIMS_PER_SUBGRP(info->ngg_info.max_gsprims) |
                          S_03096C_VERTS_PER_SUBGRP(info->ngg_info.hw_max_esverts) |
                          S_03096C_PRIM_GRP_SIZE_GFX11(max_prim_grp_size) |
                          S_03096C_DIS_PG_SIZE_ADJUST_FOR_STRIP(pdev->info.gfx_level >= GFX12);
   } else {
      regs->ngg.ge_cntl = S_03096C_PRIM_GRP_SIZE_GFX10(info->ngg_info.max_gsprims) |
                          S_03096C_VERT_GRP_SIZE(info->ngg_info.hw_max_esverts);

      regs->vgt_gs_onchip_cntl = S_028A44_ES_VERTS_PER_SUBGRP(info->ngg_info.hw_max_esverts) |
                                 S_028A44_GS_PRIMS_PER_SUBGRP(info->ngg_info.max_gsprims) |
                                 S_028A44_GS_INST_PRIMS_IN_SUBGRP(info->ngg_info.max_gsprims * gs_num_invocations);
   }

   regs->vgt_gs_max_vert_out = info->gs.vertices_out;
}

void
radv_precompute_registers_hw_ms(struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader_info *info = &shader->info;
   struct radv_shader_regs *regs = &shader->regs;

   radv_precompute_registers_hw_ngg(device, shader);

   regs->vgt_gs_max_vert_out = pdev->info.gfx_level >= GFX11 ? info->ngg_info.max_out_verts : info->workgroup_size;

   regs->ngg.ms.spi_shader_gs_meshlet_dim = S_00B2B0_MESHLET_NUM_THREAD_X(info->cs.block_size[0] - 1) |
                                            S_00B2B0_MESHLET_NUM_THREAD_Y(info->cs.block_size[1] - 1) |
                                            S_00B2B0_MESHLET_NUM_THREAD_Z(info->cs.block_size[2] - 1) |
                                            S_00B2B0_MESHLET_THREADGROUP_SIZE(info->workgroup_size - 1);

   regs->ngg.ms.spi_shader_gs_meshlet_exp_alloc =
      S_00B2B4_MAX_EXP_VERTS(info->ngg_info.max_out_verts) | S_00B2B4_MAX_EXP_PRIMS(info->ngg_info.prim_amp_factor);

   if (pdev->info.gfx_level >= GFX12) {
      const bool derivative_group_quads = info->cs.derivative_group == DERIVATIVE_GROUP_QUADS;

      regs->ngg.ms.spi_shader_gs_meshlet_ctrl =
         S_00B2B8_INTERLEAVE_BITS_X(derivative_group_quads) | S_00B2B8_INTERLEAVE_BITS_Y(derivative_group_quads);
   }
}

static uint32_t
radv_ps_input_type_bits(const struct radv_shader_info *info, unsigned index, uint32_t default_bits)
{
   if (info->ps.explicit_shaded_mask & BITFIELD_BIT(index)) {
      /* Force parameter cache to be read in passthrough mode. */
      return S_028644_OFFSET(1 << 5) | S_028644_FLAT_SHADE(1);
   } else if (info->ps.explicit_strict_shaded_mask & BITFIELD_BIT(index)) {
      /* Rotate parameter cache contents to strict vertex order. */
      return S_028644_ROTATE_PC_PTR(1) | S_028644_OFFSET(1 << 5) | S_028644_FLAT_SHADE(1);
   } else if (info->ps.float16_hi_shaded_mask & BITFIELD_BIT(index)) {
      /* These must be set even if only the high 16 bits are used. */
      return S_028644_ATTR1_VALID(1) | S_028644_FP16_INTERP_MODE(1) | S_028644_ATTR0_VALID(1);
   } else if (info->ps.float16_shaded_mask & BITFIELD_BIT(index)) {
      return S_028644_FP16_INTERP_MODE(1) | S_028644_ATTR0_VALID(1);
   } else if (info->ps.float32_shaded_mask & BITFIELD_BIT(index)) {
      return 0;
   }

   return default_bits;
}

static void
radv_ps_add_input(struct radv_ps_inputs_regs *in, uint8_t slot, uint32_t cntl)
{
   assert(in->num < ARRAY_SIZE(in->cntl));
   in->slot[in->num] = slot;
   in->cntl[in->num] = cntl;
   in->num++;
}

static void
radv_precompute_ps_inputs(const struct radv_physical_device *pdev, struct radv_shader *shader)
{
   const struct radv_shader_info *info = &shader->info;
   const uint32_t per_prim_bits = pdev->info.gfx_level >= GFX11 ? S_028644_PRIM_ATTR(1) : 0;
   const uint32_t flat_bits = S_028644_FLAT_SHADE(1);
   struct radv_ps_inputs_regs *in = &shader->regs.ps.inputs;

   in->num = 0;
   in->viewport_index = RADV_PS_INPUT_NONE;
   in->prim_id_index = RADV_PS_INPUT_NONE;

   if (info->ps.has_pcoord)
      radv_ps_add_input(in, VARYING_SLOT_PNTC, 0);

   if (info->ps.input_clips_culls_mask & 0x0f)
      radv_ps_add_input(in, VARYING_SLOT_CLIP_DIST0, 0);

   if (info->ps.input_clips_culls_mask & 0xf0)
      radv_ps_add_input(in, VARYING_SLOT_CLIP_DIST1, 0);

   u_foreach_bit (i, info->ps.input_mask)
      radv_ps_add_input(in, VARYING_SLOT_VAR0 + i, radv_ps_input_type_bits(info, in->num, flat_bits));

   /* Potentially per-primitive PS inputs, decided at emit time. */
   if (info->ps.viewport_index_input) {
      in->viewport_index = in->num;
      radv_ps_add_input(in, VARYING_SLOT_VIEWPORT, flat_bits);
   }

   if (info->ps.prim_id_input) {
      in->prim_id_index = in->num;
      radv_ps_add_input(in, VARYING_SLOT_PRIMITIVE_ID, flat_bits);
   }

   /* Per-primitive PS inputs: the HW needs these to be last. */
   in->num_per_primitive = util_bitcount(info->ps.input_per_primitive_mask);
   u_foreach_bit (i, info->ps.input_per_primitive_mask)
      radv_ps_add_input(in, VARYING_SLOT_VAR0 + i, radv_ps_input_type_bits(info, in->num, per_prim_bits));
}

void
radv_precompute_registers_hw_fs(struct radv_device *device, struct radv_shader *shader)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_shader_info *info = &shader->info;
   struct radv_shader_regs *regs = &shader->regs;

   unsigned conservative_z_export = V_02880C_EXPORT_ANY_Z;
   if (info->ps.depth_layout == FRAG_DEPTH_LAYOUT_GREATER)
      conservative_z_export = V_02880C_EXPORT_GREATER_THAN_Z;
   else if (info->ps.depth_layout == FRAG_DEPTH_LAYOUT_LESS)
      conservative_z_export = V_02880C_EXPORT_LESS_THAN_Z;

   const unsigned z_order =
      info->ps.early_fragment_test || !info->ps.writes_memory ? V_02880C_EARLY_Z_THEN_LATE_Z : V_02880C_LATE_Z;

   /* It shouldn't be needed to export gl_SampleMask when MSAA is disabled, but this appears to break Project Cars
    * (DXVK). See https://bugs.freedesktop.org/show_bug.cgi?id=109401
    */
   const bool mask_export_enable = info->ps.writes_sample_mask;
   const bool disable_rbplus = pdev->info.has_rbplus && !pdev->info.rbplus_allowed;

   regs->ps.db_shader_control =
      S_02880C_KILL_ENABLE(info->ps.can_discard) | S_02880C_CONSERVATIVE_Z_EXPORT(conservative_z_export) |
      S_02880C_Z_ORDER(z_order) | S_02880C_DEPTH_BEFORE_SHADER(info->ps.early_fragment_test) |
      S_02880C_PRE_SHADER_DEPTH_COVERAGE_ENABLE(info->ps.post_depth_coverage) |
      S_02880C_EXEC_ON_HIER_FAIL(info->ps.writes_memory) | S_02880C_EXEC_ON_NOOP(info->ps.writes_memory) |
      S_02880C_DUAL_QUAD_DISABLE(disable_rbplus) | S_02880C_PRIMITIVE_ORDERED_PIXEL_SHADER(info->ps.pops);

   if (!info->ps.has_epilog) {
      regs->ps.db_shader_control |= S_02880C_Z_EXPORT_ENABLE(info->ps.writes_z) |
                                    S_02880C_STENCIL_TEST_VAL_EXPORT_ENABLE(info->ps.writes_stencil) |
                                    S_02880C_MASK_EXPORT_ENABLE(mask_export_enable);
   }

   if (pdev->info.gfx_level >= GFX12) {
      regs->ps.spi_ps_in_control = S_028640_PS_W32_EN(info->wave_size == 32);
      regs->ps.spi_gs_out_config_ps = S_00B0C4_NUM_INTERP(info->ps.num_inputs);

      unsigned inst_pref_size = radv_get_inst_pref_size(&device->compiler_info, shader->exec_size);
      regs->ps.spi_shader_pgm_rsrc4_ps =
         S_00B01C_WAVE_LIMIT_GFX12(0x3FF) | S_00B01C_LDS_GROUP_SIZE_GFX12(1) | S_00B01C_INST_PREF_SIZE(inst_pref_size);

      if (info->ps.depth_layout == FRAG_DEPTH_LAYOUT_GREATER)
         regs->ps.pa_sc_hisz_control |= S_028BBC_CONSERVATIVE_Z_EXPORT(V_028BBC_EXPORT_GREATER_THAN_Z);
      else if (info->ps.depth_layout == FRAG_DEPTH_LAYOUT_LESS)
         regs->ps.pa_sc_hisz_control |= S_028BBC_CONSERVATIVE_Z_EXPORT(V_028BBC_EXPORT_LESS_THAN_Z);
   } else {
      /* GFX11 workaround when there are no PS inputs but LDS is used. */
      const bool param_gen = pdev->info.gfx_level == GFX11 && !info->ps.num_inputs && shader->config.lds_size;

      regs->ps.spi_ps_in_control = S_0286D8_PS_W32_EN(info->wave_size == 32) | S_0286D8_PARAM_GEN(param_gen);

      /* Can't precompute NUM_INTERP on GFX10.3 because per-primititve attributes
       * are tracked separately in NUM_PRIM_INTERP.
       */
      if (pdev->info.gfx_level != GFX10_3) {
         regs->ps.spi_ps_in_control |= S_0286D8_NUM_INTERP(info->ps.num_inputs);
      }

      if (pdev->info.gfx_level >= GFX11) {
         unsigned inst_pref_size = radv_get_inst_pref_size(&device->compiler_info, shader->exec_size);

         unsigned cu_mask_ps = ac_gfx103_get_cu_mask_ps(&pdev->info);

         regs->ps.spi_shader_pgm_rsrc4_ps =
            ac_apply_cu_en(S_00B004_CU_EN(cu_mask_ps >> 16) | S_00B004_INST_PREF_SIZE(inst_pref_size), C_00B004_CU_EN,
                           16, &pdev->info);
      }

      if (pdev->info.gfx_level >= GFX9 && pdev->info.gfx_level < GFX11)
         regs->ps.pa_sc_shader_control = S_028C40_LOAD_COLLISION_WAVEID(info->ps.pops);
   }

   regs->ps.spi_shader_z_format = ac_get_spi_shader_z_format(
      info->ps.writes_z, info->ps.writes_stencil, info->ps.writes_sample_mask, info->ps.writes_mrt0_alpha_to_mrtz);

   radv_precompute_ps_inputs(pdev, shader);
}

void
radv_precompute_vgt_ps_inputs(const struct radv_physical_device *pdev, struct radv_shader *shader)
{
   const struct radv_vs_output_info *outinfo = &shader->info.outinfo;
   const uint32_t per_prim_bits = pdev->info.gfx_level >= GFX11 ? S_028644_PRIM_ATTR(1) : 0;
   struct radv_vgt_ps_inputs_regs *r = &shader->regs.ps_inputs;

   r->exported_mask = 0;

   for (unsigned slot = 0; slot < VARYING_SLOT_MAX; slot++) {
      const uint32_t offset = outinfo->vs_output_param_offset[slot];

      if (offset == AC_EXP_PARAM_UNDEFINED) {
         /* The input is UNDEFINED, use zero. */
         r->offset[slot] = S_028644_OFFSET(0x20) | S_028644_DEFAULT_VAL(0);
      } else if (offset >= AC_EXP_PARAM_DEFAULT_VAL_0000 && offset <= AC_EXP_PARAM_DEFAULT_VAL_1111) {
         /* The input is a DEFAULT_VAL constant. */
         r->offset[slot] = S_028644_OFFSET(0x20) | S_028644_DEFAULT_VAL(offset - AC_EXP_PARAM_DEFAULT_VAL_0000);
      } else {
         r->offset[slot] = S_028644_OFFSET(offset);
         r->exported_mask |= BITFIELD64_BIT(slot);
      }
   }

   /* Point coord is FS-only and doesn't depend on the last pre-rasterization shader. */
   r->offset[VARYING_SLOT_PNTC] = S_028644_PT_SPRITE_TEX(1) | S_028644_OFFSET(0x20);

   /* Per-primitive outputs for viewport index and primitive ID. */
   if (outinfo->writes_viewport_index_per_primitive) {
      r->viewport_per_primitive = true;
      r->viewport_cntl = r->offset[VARYING_SLOT_VIEWPORT];
      if (r->exported_mask & BITFIELD64_BIT(VARYING_SLOT_VIEWPORT))
         r->viewport_cntl |= per_prim_bits;
   }

   if (outinfo->export_prim_id_per_primitive) {
      r->prim_id_per_primitive = true;
      r->prim_id_cntl = r->offset[VARYING_SLOT_PRIMITIVE_ID];
      if (r->exported_mask & BITFIELD64_BIT(VARYING_SLOT_PRIMITIVE_ID))
         r->prim_id_cntl |= per_prim_bits;
   }
}

void
radv_shader_combine_cfg_vs_tcs(const struct radv_device *device, const struct radv_shader *vs,
                               const struct radv_shader *tcs, uint32_t *rsrc1_out, uint32_t *rsrc2_out)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   if (rsrc1_out) {
      uint32_t rsrc1 = vs->config.rsrc1;

      if (G_00B848_VGPRS(tcs->config.rsrc1) > G_00B848_VGPRS(rsrc1))
         rsrc1 = (rsrc1 & C_00B848_VGPRS) | (tcs->config.rsrc1 & ~C_00B848_VGPRS);
      if (G_00B228_SGPRS(tcs->config.rsrc1) > G_00B228_SGPRS(rsrc1))
         rsrc1 = (rsrc1 & C_00B228_SGPRS) | (tcs->config.rsrc1 & ~C_00B228_SGPRS);
      if (G_00B428_LS_VGPR_COMP_CNT(tcs->config.rsrc1) > G_00B428_LS_VGPR_COMP_CNT(rsrc1))
         rsrc1 = (rsrc1 & C_00B428_LS_VGPR_COMP_CNT) | (tcs->config.rsrc1 & ~C_00B428_LS_VGPR_COMP_CNT);

      if (pdev->info.gfx_level >= GFX10 && pdev->info.gfx_level < GFX12)
         rsrc1 |= tcs->config.rsrc1 & ~C_00B428_MEM_ORDERED;

      *rsrc1_out = rsrc1;
   }

   if (rsrc2_out) {
      uint32_t rsrc2 = vs->config.rsrc2;

      rsrc2 |= tcs->config.rsrc2 & ~C_00B12C_SCRATCH_EN;

      *rsrc2_out = rsrc2;
   }
}

void
radv_shader_combine_cfg_vs_gs(const struct radv_device *device, const struct radv_shader *vs,
                              const struct radv_shader *gs, uint32_t *rsrc1_out, uint32_t *rsrc2_out,
                              uint32_t *rsrc4_out)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   assert(G_00B12C_USER_SGPR(vs->config.rsrc2) == G_00B12C_USER_SGPR(gs->config.rsrc2));

   if (rsrc1_out) {
      uint32_t rsrc1 = vs->config.rsrc1;

      if (G_00B848_VGPRS(gs->config.rsrc1) > G_00B848_VGPRS(rsrc1))
         rsrc1 = (rsrc1 & C_00B848_VGPRS) | (gs->config.rsrc1 & ~C_00B848_VGPRS);
      if (G_00B228_SGPRS(gs->config.rsrc1) > G_00B228_SGPRS(rsrc1))
         rsrc1 = (rsrc1 & C_00B228_SGPRS) | (gs->config.rsrc1 & ~C_00B228_SGPRS);
      if (G_00B228_GS_VGPR_COMP_CNT(gs->config.rsrc1) > G_00B228_GS_VGPR_COMP_CNT(rsrc1))
         rsrc1 = (rsrc1 & C_00B228_GS_VGPR_COMP_CNT) | (gs->config.rsrc1 & ~C_00B228_GS_VGPR_COMP_CNT);

      if (pdev->info.gfx_level >= GFX10 && pdev->info.gfx_level < GFX12)
         rsrc1 |= gs->config.rsrc1 & ~C_00B228_MEM_ORDERED;

      *rsrc1_out = rsrc1;
   }

   if (rsrc2_out) {
      uint32_t rsrc2 = vs->config.rsrc2;
      uint32_t lds_size;

      if (G_00B22C_ES_VGPR_COMP_CNT(gs->config.rsrc2) > G_00B22C_ES_VGPR_COMP_CNT(rsrc2))
         rsrc2 = (rsrc2 & C_00B22C_ES_VGPR_COMP_CNT) | (gs->config.rsrc2 & ~C_00B22C_ES_VGPR_COMP_CNT);

      rsrc2 |= gs->config.rsrc2 & ~(C_00B12C_SCRATCH_EN & C_00B12C_SO_EN & C_00B12C_SO_BASE0_EN & C_00B12C_SO_BASE1_EN &
                                    C_00B12C_SO_BASE2_EN & C_00B12C_SO_BASE3_EN);
      if (gs->info.is_ngg) {
         lds_size = gs->info.ngg_info.lds_size;
      } else {
         lds_size = gs->info.legacy_gs_info.lds_size;
      }

      rsrc2 |= S_00B22C_LDS_SIZE(ac_shader_encode_lds_size(lds_size, pdev->info.gfx_level, MESA_SHADER_VERTEX));

      *rsrc2_out = rsrc2;
   }

   if (rsrc4_out && pdev->info.gfx_level >= GFX11) {
      uint32_t inst_pref_mask = pdev->info.gfx_level >= GFX12 ? C_00B220_INST_PREF_SIZE : C_00B204_INST_PREF_SIZE;

      *rsrc4_out = (vs->regs.spi_shader_pgm_rsrc4_gs_hs & ~inst_pref_mask) |
                   (gs->regs.spi_shader_pgm_rsrc4_gs_hs & inst_pref_mask);
   }
}

void
radv_shader_combine_cfg_tes_gs(const struct radv_device *device, const struct radv_shader *tes,
                               const struct radv_shader *gs, uint32_t *rsrc1_out, uint32_t *rsrc2_out,
                               uint32_t *rsrc4_out)
{
   radv_shader_combine_cfg_vs_gs(device, tes, gs, rsrc1_out, rsrc2_out, rsrc4_out);

   if (rsrc2_out) {
      *rsrc2_out |= S_00B22C_OC_LDS_EN(1);
   }
}

unsigned
radv_compute_spi_ps_input(enum amd_gfx_level gfx_level, const struct radv_graphics_state_key *gfx_state,
                          const struct radv_shader_info *info)
{
   unsigned spi_ps_input;

   spi_ps_input = S_0286CC_PERSP_CENTER_ENA(info->ps.reads_persp_center) |
                  S_0286CC_PERSP_CENTROID_ENA(info->ps.reads_persp_centroid) |
                  S_0286CC_PERSP_SAMPLE_ENA(info->ps.reads_persp_sample) |
                  S_0286CC_LINEAR_CENTER_ENA(info->ps.reads_linear_center) |
                  S_0286CC_LINEAR_CENTROID_ENA(info->ps.reads_linear_centroid) |
                  S_0286CC_LINEAR_SAMPLE_ENA(info->ps.reads_linear_sample) |
                  S_0286CC_PERSP_PULL_MODEL_ENA(info->ps.reads_barycentric_model) |
                  S_0286CC_FRONT_FACE_ENA(info->ps.reads_front_face) |
                  S_0286CC_POS_FIXED_PT_ENA(info->ps.reads_pixel_coord);

   if (info->ps.reads_frag_coord_mask) {
      uint8_t mask = info->ps.reads_frag_coord_mask;

      for (unsigned i = 0; i < 4; i++) {
         if (mask & (1 << i))
            spi_ps_input |= S_0286CC_POS_X_FLOAT_ENA(1) << i;
      }

      if (gfx_state->adjust_frag_coord_z && info->ps.reads_frag_coord_mask & (1 << 2)) {
         spi_ps_input |= S_0286CC_ANCILLARY_ENA(1);
      }
   }

   if (info->ps.reads_sample_id || info->ps.reads_frag_shading_rate || info->ps.reads_layer) {
      spi_ps_input |= S_0286CC_ANCILLARY_ENA(1);
   }

   if (info->ps.reads_sample_mask_in || info->ps.reads_fully_covered || info->ps.needs_poly_line_smooth_coverage) {
      spi_ps_input |= S_0286CC_SAMPLE_COVERAGE_ENA(1) |
                      S_02865C_COVERAGE_TO_SHADER_SELECT(gfx_level >= GFX12 && info->ps.reads_fully_covered);
   }

   /* POW_W_FLOAT requires that one set of perspective barycentric coordinates is enabled. */
   if (G_0286CC_POS_W_FLOAT_ENA(spi_ps_input) && !G_0286CC_PERSP_SAMPLE_ENA(spi_ps_input) &&
       !G_0286CC_PERSP_CENTER_ENA(spi_ps_input) && !G_0286CC_PERSP_CENTROID_ENA(spi_ps_input) &&
       !G_0286CC_PERSP_PULL_MODEL_ENA(spi_ps_input))
      spi_ps_input |= S_0286CC_PERSP_CENTER_ENA(1);

   if (!G_0286CC_PERSP_SAMPLE_ENA(spi_ps_input) && !G_0286CC_PERSP_CENTER_ENA(spi_ps_input) &&
       !G_0286CC_PERSP_CENTROID_ENA(spi_ps_input) && !G_0286CC_PERSP_PULL_MODEL_ENA(spi_ps_input) &&
       !G_0286CC_LINEAR_SAMPLE_ENA(spi_ps_input) && !G_0286CC_LINEAR_CENTER_ENA(spi_ps_input) &&
       !G_0286CC_LINEAR_CENTROID_ENA(spi_ps_input) && !G_0286CC_LINE_STIPPLE_TEX_ENA(spi_ps_input)) {
      /* At least one of PERSP_*, LINEAR_*, or LINE_STIPPLE_TEX must be enabled.
       * LINE_STIPPLE_TEX uses the least number of initialized VGPRs, so let's use it because
       * pixel throughput is limited by the number of initialized VGPRs.
       *
       * We can't set LINE_STIPPLE_TEX on GFX12 because it reduces primitive throughput to only
       * 1 SE. Other gens are fine (tested on Navi10, Navi21, Navi31).
       * TODO: Test Strix Halo.
       */
      if (gfx_level == GFX12)
         spi_ps_input |= S_0286CC_PERSP_SAMPLE_ENA(1);
      else
         spi_ps_input |= S_0286CC_LINE_STIPPLE_TEX_ENA(1);
   }

   return spi_ps_input;
}

void
radv_get_tess_wg_info(const struct radv_compiler_info *compiler_info, const ac_nir_tess_io_info *io_info,
                      unsigned tcs_vertices_out, unsigned tcs_num_input_vertices, unsigned tcs_num_lds_inputs,
                      unsigned *num_patches_per_wg, unsigned *lds_size)
{
   const uint32_t lds_input_vertex_size = get_tcs_input_vertex_stride(tcs_num_lds_inputs);

   ac_nir_compute_tess_wg_info(compiler_info->ac, io_info, tcs_vertices_out, compiler_info->key.ge_wave_size, false,
                               tcs_num_input_vertices, lds_input_vertex_size, 0, num_patches_per_wg, lds_size);
}

static const mesa_shader_stage graphics_shader_order[] = {
   MESA_SHADER_VERTEX,   MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL, MESA_SHADER_GEOMETRY,

   MESA_SHADER_TASK,     MESA_SHADER_MESH,

   MESA_SHADER_FRAGMENT,
};

static bool
format_is_int8(const struct util_format_description *desc)
{
   int channel = util_format_get_first_non_void_channel(desc->format);

   return channel >= 0 && desc->channel[channel].pure_integer && desc->channel[channel].size == 8;
}

static bool
format_is_int10(const struct util_format_description *desc)
{
   if (desc->nr_channels != 4)
      return false;
   for (unsigned i = 0; i < 4; i++) {
      if (desc->channel[i].pure_integer && desc->channel[i].size == 10)
         return true;
   }
   return false;
}

static bool
format_is_float32(const struct util_format_description *desc)
{
   int channel = util_format_get_first_non_void_channel(desc->format);

   return channel >= 0 && desc->channel[channel].type == UTIL_FORMAT_TYPE_FLOAT && desc->channel[channel].size == 32;
}

static bool
format_ignores_signed_zero(const struct util_format_description *desc)
{
   /* Unsigned float formats don't care about signed zeros. */
   if (desc->format == PIPE_FORMAT_R11G11B10_FLOAT || desc->format == PIPE_FORMAT_R9G9B9E5_FLOAT)
      return true;

   for (unsigned i = 0; i < desc->nr_channels; i++) {
      if (desc->channel[i].pure_integer || desc->channel[i].type == UTIL_FORMAT_TYPE_FLOAT)
         return false;
   }

   return true;
}

struct radv_ia_multi_vgt_param_helpers
radv_compute_ia_multi_vgt_param(const struct radv_device *device, struct radv_shader *const *shaders)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   struct radv_ia_multi_vgt_param_helpers ia_multi_vgt_param = {0};

   if (shaders[MESA_SHADER_TESS_CTRL]) {
      const struct radv_shader *tes = radv_get_shader(shaders, MESA_SHADER_TESS_EVAL);

      /* SWITCH_ON_EOI must be set if PrimID is used. */
      if (shaders[MESA_SHADER_TESS_CTRL]->info.uses_prim_id || tes->info.uses_prim_id ||
          (tes->info.merged_shader_compiled_separately && shaders[MESA_SHADER_GEOMETRY]->info.uses_prim_id))
         ia_multi_vgt_param.ia_switch_on_eoi = true;
      if (shaders[MESA_SHADER_FRAGMENT] && shaders[MESA_SHADER_FRAGMENT]->info.ps.prim_id_input)
         ia_multi_vgt_param.ia_switch_on_eoi = true;
      if (shaders[MESA_SHADER_GEOMETRY] && shaders[MESA_SHADER_GEOMETRY]->info.uses_prim_id)
         ia_multi_vgt_param.ia_switch_on_eoi = true;
   }

   if (shaders[MESA_SHADER_TESS_CTRL]) {
      /* Bug with tessellation and GS on Bonaire and older 2 SE chips. */
      if ((pdev->info.family == CHIP_TAHITI || pdev->info.family == CHIP_PITCAIRN ||
           pdev->info.family == CHIP_BONAIRE) &&
          shaders[MESA_SHADER_GEOMETRY])
         ia_multi_vgt_param.partial_vs_wave = true;
      /* Needed for 028B6C_DISTRIBUTION_MODE != 0 */
      if (pdev->info.has_distributed_tess) {
         if (shaders[MESA_SHADER_GEOMETRY]) {
            if (pdev->info.gfx_level <= GFX8)
               ia_multi_vgt_param.partial_es_wave = true;
         } else {
            ia_multi_vgt_param.partial_vs_wave = true;
         }
      }
   }

   if (shaders[MESA_SHADER_GEOMETRY]) {
      /* On these chips there is the possibility of a hang if a GS is used and partial_vs_wave is
       * not set.
       *
       * This mostly does not hit 4-SE chips, as those typically set ia_switch_on_eoi and then
       * partial_vs_wave is set for GS shaders due to another workaround.
       *
       * Reproducer: https://bugs.freedesktop.org/show_bug.cgi?id=109242
       */
      if (pdev->info.family == CHIP_TONGA || pdev->info.family == CHIP_FIJI || pdev->info.family == CHIP_POLARIS10 ||
          pdev->info.family == CHIP_POLARIS11 || pdev->info.family == CHIP_POLARIS12 ||
          pdev->info.family == CHIP_VEGAM) {
         ia_multi_vgt_param.partial_vs_wave = true;
      }
   }

   ia_multi_vgt_param.base =
      /* The following field was moved to VGT_SHADER_STAGES_EN in GFX9. */
      S_028AA8_MAX_PRIMGRP_IN_WAVE(pdev->info.gfx_level == GFX8 ? 2 : 0) |
      S_030960_EN_INST_OPT_BASIC(pdev->info.gfx_level >= GFX9) | S_030960_EN_INST_OPT_ADV(pdev->info.gfx_level >= GFX9);

   return ia_multi_vgt_param;
}

static bool
radv_should_export_multiview(const struct radv_shader_stage *stage)
{
   /* Export the layer in the last VGT stage if multiview is used.
    * Also checks for NONE stage, which happens when we have depth-only rendering.
    * When the next stage is unknown (with GPL or ESO), the layer is exported unconditionally.
    */
   return stage->key.multiview && radv_is_last_vgt_stage(stage) &&
          !(stage->nir->info.outputs_written & VARYING_BIT_LAYER);
}

static void
merge_tess_info(struct shader_info *tes_info, struct shader_info *tcs_info)
{
   /* The Vulkan 1.0.38 spec, section 21.1 Tessellator says:
    *
    *    "PointMode. Controls generation of points rather than triangles
    *     or lines. This functionality defaults to disabled, and is
    *     enabled if either shader stage includes the execution mode.
    *
    * and about Triangles, Quads, IsoLines, VertexOrderCw, VertexOrderCcw,
    * PointMode, SpacingEqual, SpacingFractionalEven, SpacingFractionalOdd,
    * and OutputVertices, it says:
    *
    *    "One mode must be set in at least one of the tessellation
    *     shader stages."
    *
    * So, the fields can be set in either the TCS or TES, but they must
    * agree if set in both.  Our backend looks at TES, so bitwise-or in
    * the values from the TCS.
    */
   assert(tcs_info->tess.tcs_vertices_out == 0 || tes_info->tess.tcs_vertices_out == 0 ||
          tcs_info->tess.tcs_vertices_out == tes_info->tess.tcs_vertices_out);
   tes_info->tess.tcs_vertices_out |= tcs_info->tess.tcs_vertices_out;

   assert(tcs_info->tess.spacing == TESS_SPACING_UNSPECIFIED || tes_info->tess.spacing == TESS_SPACING_UNSPECIFIED ||
          tcs_info->tess.spacing == tes_info->tess.spacing);
   tes_info->tess.spacing |= tcs_info->tess.spacing;

   assert(tcs_info->tess._primitive_mode == TESS_PRIMITIVE_UNSPECIFIED ||
          tes_info->tess._primitive_mode == TESS_PRIMITIVE_UNSPECIFIED ||
          tcs_info->tess._primitive_mode == tes_info->tess._primitive_mode);
   tes_info->tess._primitive_mode |= tcs_info->tess._primitive_mode;
   tes_info->tess.ccw |= tcs_info->tess.ccw;
   tes_info->tess.point_mode |= tcs_info->tess.point_mode;

   /* Copy the merged info back to the TCS */
   tcs_info->tess.tcs_vertices_out = tes_info->tess.tcs_vertices_out;
   tcs_info->tess._primitive_mode = tes_info->tess._primitive_mode;
}

static void
radv_graphics_shaders_fill_linked_vs_io_info(struct radv_shader_stage *vs_stage,
                                             struct radv_shader_stage *consumer_stage)
{
   const unsigned num_reserved_slots = util_bitcount64(consumer_stage->nir->info.inputs_read);
   vs_stage->info.vs.num_linked_outputs = num_reserved_slots;
   vs_stage->info.outputs_linked = true;

   switch (consumer_stage->stage) {
   case MESA_SHADER_TESS_CTRL: {
      consumer_stage->info.tcs.num_linked_inputs = num_reserved_slots;
      consumer_stage->info.inputs_linked = true;
      break;
   }
   case MESA_SHADER_GEOMETRY: {
      consumer_stage->info.gs.num_linked_inputs = num_reserved_slots;
      consumer_stage->info.inputs_linked = true;
      break;
   }
   default:
      UNREACHABLE("invalid next stage for VS");
   }
}

static void
radv_graphics_shaders_fill_linked_tcs_tes_io_info(struct radv_shader_stage *tcs_stage,
                                                  struct radv_shader_stage *tes_stage)
{
   assume(tes_stage->stage == MESA_SHADER_TESS_EVAL);

   /* Count the number of per-vertex output slots we need to reserve for the TCS and TES. */
   const uint64_t per_vertex_mask =
      tes_stage->nir->info.inputs_read & ~(VARYING_BIT_TESS_LEVEL_OUTER | VARYING_BIT_TESS_LEVEL_INNER);
   const unsigned num_reserved_slots = util_bitcount64(per_vertex_mask);

   /* Count the number of per-patch output slots we need to reserve for the TCS and TES.
    * This is necessary because we need it to determine the patch size in VRAM.
    */
   const uint64_t tess_lvl_mask =
      tes_stage->nir->info.inputs_read & (VARYING_BIT_TESS_LEVEL_OUTER | VARYING_BIT_TESS_LEVEL_INNER);
   const unsigned num_reserved_patch_slots =
      util_bitcount64(tess_lvl_mask) + util_bitcount64(tes_stage->nir->info.patch_inputs_read);

   tcs_stage->info.outputs_linked = true;

   tes_stage->info.tes.num_linked_inputs = num_reserved_slots;
   tes_stage->info.tes.num_linked_patch_inputs = num_reserved_patch_slots;
   tes_stage->info.inputs_linked = true;
}

static void
radv_graphics_shaders_fill_linked_tes_gs_io_info(struct radv_shader_stage *tes_stage,
                                                 struct radv_shader_stage *gs_stage)
{
   assume(gs_stage->stage == MESA_SHADER_GEOMETRY);

   const unsigned num_reserved_slots = util_bitcount64(gs_stage->nir->info.inputs_read);
   tes_stage->info.tes.num_linked_outputs = num_reserved_slots;
   tes_stage->info.outputs_linked = true;
   gs_stage->info.gs.num_linked_inputs = num_reserved_slots;
   gs_stage->info.inputs_linked = true;
}

static void
radv_graphics_shaders_fill_linked_io_info(struct radv_shader_stage *producer_stage,
                                          struct radv_shader_stage *consumer_stage)
{
   /* We don't need to fill this info for the last pre-rasterization stage. */
   if (consumer_stage->stage == MESA_SHADER_FRAGMENT)
      return;

   switch (producer_stage->stage) {
   case MESA_SHADER_VERTEX:
      radv_graphics_shaders_fill_linked_vs_io_info(producer_stage, consumer_stage);
      break;

   case MESA_SHADER_TESS_CTRL:
      radv_graphics_shaders_fill_linked_tcs_tes_io_info(producer_stage, consumer_stage);
      break;

   case MESA_SHADER_TESS_EVAL:
      radv_graphics_shaders_fill_linked_tes_gs_io_info(producer_stage, consumer_stage);
      break;

   default:
      break;
   }
}

/**
 * Varying optimizations performed on lowered shader I/O.
 *
 * We do this after lowering shader I/O because this is more effective
 * than running the same optimizations on I/O derefs.
 */
static void
radv_graphics_shaders_link_varyings(struct radv_shader_stage *stages, enum amd_gfx_level gfx_level)
{
   bool fs_layer_lowered_to_input = false;

   /* Prepare shaders before running nir_opt_varyings. */
   for (int i = 0; i < ARRAY_SIZE(graphics_shader_order); ++i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      if (!stages[s].nir)
         continue;

      if (stages[s].key.optimisations_disabled)
         continue;

      nir_shader *shader = stages[s].nir;

      /* It is expected by nir_opt_varyings that no undefined stores are present in the shader. */
      NIR_PASS(_, shader, nir_opt_undef);

      /* Update load/store alignments because inter-stage code motion may move instructions used to deduce this info. */
      NIR_PASS(_, shader, nir_opt_load_store_update_alignments);

      if (s == MESA_SHADER_FRAGMENT) {
         /* LAYER_ID must be an input to be optimizable by nir_opt_varyings.
          * PRIMITIVE_ID too, but that's already an input.
          */
         NIR_PASS(fs_layer_lowered_to_input, shader, nir_lower_sysvals_to_varyings,
                  &(nir_lower_sysvals_to_varyings_options){.layer_id = true});
      }
   }

   int highest_changed_producer = -1;

   /* Optimize varyings from first to last stage. */
   for (int i = 0; i < ARRAY_SIZE(graphics_shader_order); ++i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      const mesa_shader_stage next = stages[s].info.next_stage;
      if (!stages[s].nir || next == MESA_SHADER_NONE || !stages[next].nir)
         continue;

      if (stages[s].key.optimisations_disabled || stages[next].key.optimisations_disabled)
         continue;

      nir_shader *producer = stages[s].nir;
      nir_shader *consumer = stages[next].nir;

      const nir_opt_varyings_progress p = nir_opt_varyings(producer, consumer, true, 0, 0, false);

      /* Run algebraic optimizations on shaders that changed. */
      if (p & nir_progress_producer) {
         radv_optimize_nir_algebraic(producer, false, false, gfx_level);
         NIR_PASS(_, producer, nir_opt_undef);

         highest_changed_producer = i;
      }
      if (p & nir_progress_consumer) {
         radv_optimize_nir_algebraic(consumer, false, false, gfx_level);
         NIR_PASS(_, consumer, nir_opt_undef);
      }
   }

   /* Optimize varyings from last to first stage. */
   for (int i = highest_changed_producer; i >= 0; --i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      const mesa_shader_stage next = stages[s].info.next_stage;
      if (!stages[s].nir || next == MESA_SHADER_NONE || !stages[next].nir)
         continue;

      if (stages[s].key.optimisations_disabled || stages[next].key.optimisations_disabled)
         continue;

      nir_shader *producer = stages[s].nir;
      nir_shader *consumer = stages[next].nir;

      const nir_opt_varyings_progress p = nir_opt_varyings(producer, consumer, true, 0, 0, false);

      /* Run algebraic optimizations on shaders that changed. */
      if (p & nir_progress_producer) {
         radv_optimize_nir_algebraic(producer, true, false, gfx_level);
         NIR_PASS(_, producer, nir_opt_undef);
      }
      if (p & nir_progress_consumer) {
         radv_optimize_nir_algebraic(consumer, true, false, gfx_level);
         NIR_PASS(_, consumer, nir_opt_undef);
      }
   }

   /* Run optimizations and fixups after linking. */
   for (int i = 0; i < ARRAY_SIZE(graphics_shader_order); ++i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      if (!stages[s].nir)
         continue;

      nir_shader *shader = stages[s].nir;

      if (shader->info.stage == MESA_SHADER_FRAGMENT && fs_layer_lowered_to_input)
         NIR_PASS(_, shader, nir_lower_system_values);

      /* Re-vectorize I/O for stages that use memory for I/O (LDS or VRAM).
       * Don't vectorize FS I/O, doing so just regresses shader stats without any benefit.
       */
      if (s != MESA_SHADER_FRAGMENT && !stages[s].key.optimisations_disabled) {
         /* Delete dead instructions to prevent them from being vectorized. */
         NIR_PASS(_, shader, nir_opt_dce);

         /* Vectorize inputs. Non-FS inputs are always read from memory. */
         nir_variable_mode vec_mode = nir_var_shader_in;

         /* There is also no benefit from re-vectorizing the outputs of the last pre-rasterization
          * stage here, because ac_nir_lower_ngg/legacy already takes care of that.
          */
         if (!radv_is_last_vgt_stage(&stages[s]))
            vec_mode |= nir_var_shader_out;

         /* Scalarize and revectorize VS inputs to make sure every VS input is loaded by a
          * single *_load_format_* instruction. Those instructions can't skip loading unused
          * components before the last used component, so loading X, Y, Z, W separately
          * actually loads X, XY, XYZ, XYZW, which unnecessarily increases VMEM return data
          * transfers between the VMEM cache and the SIMDs, which wastes SIMD<->VMEM cache bandwidth.
          * By allowing holes during VS input vectorization, VS input loads loading different
          * components are always merged, so that no used or unused component is ever loaded twice.
          */
         if (s == MESA_SHADER_VERTEX) {
            NIR_PASS(_, shader, nir_opt_vectorize_io, nir_var_shader_in, true);
            vec_mode &= ~nir_var_shader_in;
         }

         if (vec_mode)
            NIR_PASS(_, shader, nir_opt_vectorize_io, vec_mode, false);
      }

      /* Gather shader info; at least the I/O info likely changed
       * and changes to only the I/O info are not reflected in nir_opt_varyings_progress.
       */
      nir_shader_gather_info(shader, nir_shader_get_entrypoint(shader));

      /* Recreate XFB info from intrinsics (nir_opt_varyings may have changed it). */
      if (shader->xfb_info) {
         nir_gather_xfb_info_from_intrinsics(shader);
      }
   }

   /* Fill linked I/O info.
    * This needs to be done after all optimizations are done and shader info gathered.
    */
   for (int i = 0; i < ARRAY_SIZE(graphics_shader_order); ++i) {
      const mesa_shader_stage s = graphics_shader_order[i];
      const mesa_shader_stage next = stages[s].info.next_stage;
      if (!stages[s].nir || next == MESA_SHADER_NONE || !stages[next].nir)
         continue;

      radv_graphics_shaders_fill_linked_io_info(&stages[s], &stages[next]);
   }
}

void
radv_init_ps_epilog_format_infos(struct radv_compiler_info *compiler_info)
{
   const enum amd_gfx_level gfx_level = compiler_info->ac->gfx_level;

   for (unsigned i = 0; i < PIPE_FORMAT_COUNT; i++) {
      const enum pipe_format format = i;
      const struct util_format_description *desc = util_format_description(format);
      struct radv_ps_epilog_format_info *info = &compiler_info->ps_epilog_formats[format];

      if (!ac_is_colorbuffer_format_supported(gfx_level, format))
         continue;

      ac_choose_spi_color_formats(ac_get_cb_format(gfx_level, format), ac_translate_colorswap(gfx_level, format, false),
                                  ac_get_cb_number_type(format), false, compiler_info->hw.rbplus_allowed,
                                  &info->spi_color_formats);

      info->colormask = util_format_colormask(desc);
      info->ignores_signed_zero = format_ignores_signed_zero(desc);
      info->is_int8 = format_is_int8(desc);
      info->is_int10 = format_is_int10(desc);
      info->is_float32 = format_is_float32(desc);
   }
}

struct radv_ps_epilog_key
radv_generate_ps_epilog_key(const struct radv_compiler_info *compiler_info, const struct radv_ps_epilog_state *state)
{
   unsigned col_format = 0, is_int8 = 0, is_int10 = 0, is_float32 = 0, z_format = 0, no_signed_zero = 0;
   struct radv_ps_epilog_key key;

   memset(&key, 0, sizeof(key));
   memset(key.color_map, MESA_VK_ATTACHMENT_UNUSED, sizeof(key.color_map));

   for (unsigned i = 0; i < state->color_attachment_count; ++i) {
      unsigned cf;
      unsigned cb_idx = state->color_attachment_mappings[i];
      VkFormat fmt = state->color_attachment_formats[i];

      if (fmt == VK_FORMAT_UNDEFINED || !(state->color_write_mask & (0xfu << (i * 4))) ||
          cb_idx == MESA_VK_ATTACHMENT_UNUSED) {
         cf = V_028714_SPI_SHADER_ZERO;
      } else {
         const struct radv_ps_epilog_format_info *fmt_info =
            &compiler_info->ps_epilog_formats[radv_format_to_pipe_format(fmt)];
         const bool blend_enable = (state->color_blend_enable >> i) & 0x1u;
         const bool need_src_alpha = (state->need_src_alpha >> i) & 0x1u;

         if (blend_enable && need_src_alpha)
            cf = fmt_info->spi_color_formats.blend_alpha;
         else if (need_src_alpha)
            cf = fmt_info->spi_color_formats.alpha;
         else if (blend_enable)
            cf = fmt_info->spi_color_formats.blend;
         else
            cf = fmt_info->spi_color_formats.normal;

         uint32_t comp_used = fmt_info->colormask;

         comp_used &= (state->color_write_mask >> (i * 4));
         comp_used |= need_src_alpha << 3;

         key.colors_needed |= comp_used << (4 * i);

         if (fmt_info->ignores_signed_zero || blend_enable)
            no_signed_zero |= 1 << i;
         if (fmt_info->is_int8)
            is_int8 |= 1 << i;
         if (fmt_info->is_int10)
            is_int10 |= 1 << i;
         if (fmt_info->is_float32)
            is_float32 |= 1 << i;
      }

      col_format |= cf << (4 * i);

      key.color_map[i] = state->color_attachment_mappings[i];
   }

   if (!(col_format & 0xf) && state->need_src_alpha & (1 << 0)) {
      /* When a subpass doesn't have any color attachments, write the alpha channel of MRT0 when
       * alpha coverage is enabled because the depth attachment needs it.
       */
      col_format |= V_028714_SPI_SHADER_32_AR;
      key.color_map[0] = 0;
      key.colors_needed |= 0x8;
   }

   /* The output for dual source blending should have the same format as the first output. */
   if (state->mrt0_is_dual_src) {
      assert(!(col_format >> 4));
      col_format |= (col_format & 0xf) << 4;
      key.color_map[1] = 1;
      key.colors_needed |= (key.colors_needed & 0xf) << 4;
      no_signed_zero |= 0x2;
   }

   z_format = ac_get_spi_shader_z_format(state->has_depth_output && !state->ignore_depth_output,
                                         state->has_stencil_output && !state->ignore_stencil_output,
                                         state->has_sample_mask_output && !state->lower_1bit_sample_mask_to_discard,
                                         state->alpha_to_coverage_via_mrtz);

   key.spi_shader_col_format = col_format;
   key.color_is_int8 = compiler_info->ac->has_cb_lt16bit_int_clamp_bug ? is_int8 : 0;
   key.color_is_int10 = compiler_info->ac->has_cb_lt16bit_int_clamp_bug ? is_int10 : 0;
   key.enable_mrt_output_nan_fixup = compiler_info->key.enable_mrt_output_nan_fixup ? is_float32 : 0;
   key.no_signed_zero = no_signed_zero;
   key.colors_written = state->colors_written;
   key.mrt0_is_dual_src = state->mrt0_is_dual_src && key.colors_needed & 0xf;
   key.has_depth_output = state->has_depth_output;
   key.has_stencil_output = state->has_stencil_output;
   key.has_sample_mask_output = state->has_sample_mask_output;
   key.alpha_to_coverage_via_mrtz = state->alpha_to_coverage_via_mrtz;
   key.spi_shader_z_format = z_format;
   key.alpha_to_one = state->alpha_to_one;
   key.ignore_depth_output = state->ignore_depth_output;
   key.ignore_stencil_output = state->ignore_stencil_output;
   key.lower_1bit_sample_mask_to_discard = state->lower_1bit_sample_mask_to_discard;

   return key;
}

static void
radv_fill_shader_info_ngg(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                          VkShaderStageFlagBits active_nir_stages)
{
   if (!compiler_info->key.use_ngg)
      return;

   if (stages[MESA_SHADER_VERTEX].nir && stages[MESA_SHADER_VERTEX].info.next_stage != MESA_SHADER_TESS_CTRL) {
      stages[MESA_SHADER_VERTEX].info.is_ngg = true;
   } else if (stages[MESA_SHADER_TESS_EVAL].nir) {
      stages[MESA_SHADER_TESS_EVAL].info.is_ngg = true;
   } else if (stages[MESA_SHADER_MESH].nir) {
      stages[MESA_SHADER_MESH].info.is_ngg = true;
   }

   if (compiler_info->ac->gfx_level >= GFX11) {
      if (stages[MESA_SHADER_GEOMETRY].nir)
         stages[MESA_SHADER_GEOMETRY].info.is_ngg = true;
   } else {
      /* GFX10/GFX10.3 can't always enable NGG due to HW bugs/limitations. */
      if (stages[MESA_SHADER_TESS_EVAL].nir && stages[MESA_SHADER_GEOMETRY].nir &&
          stages[MESA_SHADER_GEOMETRY].nir->info.gs.invocations *
                stages[MESA_SHADER_GEOMETRY].nir->info.gs.vertices_out >
             256) {
         /* Fallback to the legacy path if tessellation is
          * enabled with extreme geometry because
          * EN_MAX_VERT_OUT_PER_GS_INSTANCE doesn't work and it
          * might hang.
          */
         stages[MESA_SHADER_TESS_EVAL].info.is_ngg = false;
      }

      struct radv_shader_stage *last_vgt_stage = NULL;
      radv_foreach_stage (i, active_nir_stages) {
         if (radv_is_last_vgt_stage(&stages[i])) {
            last_vgt_stage = &stages[i];
         }
      }

      if ((last_vgt_stage && last_vgt_stage->nir->xfb_info) ||
          (compiler_info->key.no_ngg_gs && stages[MESA_SHADER_GEOMETRY].nir)) {
         /* NGG needs to be disabled on GFX10/GFX10.3 when:
          * - streamout is used because NGG streamout isn't supported
          * - NGG GS is explictly disabled to workaround performance issues
          */
         if (stages[MESA_SHADER_TESS_EVAL].nir)
            stages[MESA_SHADER_TESS_EVAL].info.is_ngg = false;
         else
            stages[MESA_SHADER_VERTEX].info.is_ngg = false;
      }

      if (stages[MESA_SHADER_GEOMETRY].nir) {
         if (stages[MESA_SHADER_TESS_EVAL].nir)
            stages[MESA_SHADER_GEOMETRY].info.is_ngg = stages[MESA_SHADER_TESS_EVAL].info.is_ngg;
         else
            stages[MESA_SHADER_GEOMETRY].info.is_ngg = stages[MESA_SHADER_VERTEX].info.is_ngg;
      }

      /* When pre-rasterization stages are compiled separately with shader objects, NGG GS needs to
       * be disabled because if the next stage of VS/TES is GS and GS is unknown, it might use
       * streamout but it's not possible to know that when compiling VS or TES only.
       */
      if (stages[MESA_SHADER_VERTEX].nir && stages[MESA_SHADER_VERTEX].info.next_stage == MESA_SHADER_GEOMETRY &&
          !stages[MESA_SHADER_GEOMETRY].nir) {
         stages[MESA_SHADER_VERTEX].info.is_ngg = false;
      } else if (stages[MESA_SHADER_TESS_EVAL].nir &&
                 stages[MESA_SHADER_TESS_EVAL].info.next_stage == MESA_SHADER_GEOMETRY &&
                 !stages[MESA_SHADER_GEOMETRY].nir) {
         stages[MESA_SHADER_TESS_EVAL].info.is_ngg = false;
      } else if (stages[MESA_SHADER_GEOMETRY].nir &&
                 (!stages[MESA_SHADER_VERTEX].nir && !stages[MESA_SHADER_TESS_EVAL].nir)) {
         stages[MESA_SHADER_GEOMETRY].info.is_ngg = false;
      }
   }

   /* Now that we know if ngg is used for geometry shaders, determine the subgroup size. */
   if (stages[MESA_SHADER_GEOMETRY].nir) {
      unsigned wave_size = stages[MESA_SHADER_GEOMETRY].info.is_ngg
                              ? stages[MESA_SHADER_GEOMETRY].nir->info.min_subgroup_size
                              : stages[MESA_SHADER_GEOMETRY].nir->info.max_subgroup_size;
      stages[MESA_SHADER_GEOMETRY].nir->info.max_subgroup_size = wave_size;
      stages[MESA_SHADER_GEOMETRY].nir->info.min_subgroup_size = wave_size;
   }
}

static bool
radv_consider_force_vrs(const struct radv_graphics_state_key *gfx_state, const struct radv_shader_stage *last_vgt_stage,
                        const struct radv_shader_stage *fs_stage)
{
   if (!gfx_state->ps.force_vrs_enabled)
      return false;

   /* Mesh shaders aren't considered. */
   if (last_vgt_stage->info.stage == MESA_SHADER_MESH)
      return false;

   if (last_vgt_stage->nir->info.outputs_written & VARYING_BIT_PRIMITIVE_SHADING_RATE)
      return false;

   /* VRS has no effect if there is no pixel shader. */
   if (last_vgt_stage->info.next_stage == MESA_SHADER_NONE)
      return false;

   return true;
}

static mesa_shader_stage
radv_get_next_stage(mesa_shader_stage stage, VkShaderStageFlagBits active_nir_stages)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      if (active_nir_stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
         return MESA_SHADER_TESS_CTRL;
      } else if (active_nir_stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
         return MESA_SHADER_GEOMETRY;
      } else if (active_nir_stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
         return MESA_SHADER_FRAGMENT;
      } else {
         return MESA_SHADER_NONE;
      }
   case MESA_SHADER_TESS_CTRL:
      return MESA_SHADER_TESS_EVAL;
   case MESA_SHADER_TESS_EVAL:
      if (active_nir_stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
         return MESA_SHADER_GEOMETRY;
      } else if (active_nir_stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
         return MESA_SHADER_FRAGMENT;
      } else {
         return MESA_SHADER_NONE;
      }
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_MESH:
      if (active_nir_stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
         return MESA_SHADER_FRAGMENT;
      } else {
         return MESA_SHADER_NONE;
      }
   case MESA_SHADER_TASK:
      return MESA_SHADER_MESH;
   case MESA_SHADER_FRAGMENT:
      return MESA_SHADER_NONE;
   default:
      UNREACHABLE("invalid graphics shader stage");
   }
}

static void
radv_fill_shader_info(const struct radv_compiler_info *compiler_info, const struct radv_graphics_state_key *gfx_state,
                      struct radv_shader_stage *stages, VkShaderStageFlagBits active_nir_stages)
{
   radv_foreach_stage (i, active_nir_stages) {
      bool consider_force_vrs = false;

      if (radv_is_last_vgt_stage(&stages[i])) {
         consider_force_vrs = radv_consider_force_vrs(gfx_state, &stages[i], &stages[MESA_SHADER_FRAGMENT]);
      }

      if (i == MESA_SHADER_FRAGMENT)
         NIR_PASS(_, stages[i].nir, ac_nir_assign_fs_input_locations);

      radv_nir_shader_info_pass(compiler_info, stages[i].nir, &stages[i].layout, &stages[i].key, gfx_state,
                                consider_force_vrs, &stages[i].info);
   }

   radv_nir_shader_info_link(compiler_info, gfx_state, stages);
}

static void
radv_declare_graphics_shaders_args(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                                   const struct radv_graphics_state_key *gfx_state,
                                   VkShaderStageFlagBits active_nir_stages, struct radv_shader_debug_info *debug)
{
   enum amd_gfx_level gfx_level = compiler_info->ac->gfx_level;

   if (gfx_level >= GFX9 && stages[MESA_SHADER_TESS_CTRL].nir) {
      radv_declare_shader_args(compiler_info, gfx_state, &stages[MESA_SHADER_TESS_CTRL], MESA_SHADER_VERTEX,
                               &debug[MESA_SHADER_TESS_CTRL]);
      stages[MESA_SHADER_TESS_CTRL].info.user_sgprs_locs = stages[MESA_SHADER_TESS_CTRL].args.user_sgprs_locs;
      stages[MESA_SHADER_TESS_CTRL].info.inline_push_constant_mask =
         stages[MESA_SHADER_TESS_CTRL].args.ac.inline_push_const_mask;

      stages[MESA_SHADER_VERTEX].info.user_sgprs_locs = stages[MESA_SHADER_TESS_CTRL].info.user_sgprs_locs;
      stages[MESA_SHADER_VERTEX].info.inline_push_constant_mask =
         stages[MESA_SHADER_TESS_CTRL].info.inline_push_constant_mask;
      stages[MESA_SHADER_VERTEX].args = stages[MESA_SHADER_TESS_CTRL].args;

      active_nir_stages &= ~(1 << MESA_SHADER_VERTEX);
      active_nir_stages &= ~(1 << MESA_SHADER_TESS_CTRL);
   }

   if (gfx_level >= GFX9 && stages[MESA_SHADER_GEOMETRY].nir) {
      mesa_shader_stage pre_stage = stages[MESA_SHADER_TESS_EVAL].nir ? MESA_SHADER_TESS_EVAL : MESA_SHADER_VERTEX;
      radv_declare_shader_args(compiler_info, gfx_state, &stages[MESA_SHADER_GEOMETRY], pre_stage,
                               &debug[MESA_SHADER_GEOMETRY]);
      stages[MESA_SHADER_GEOMETRY].info.user_sgprs_locs = stages[MESA_SHADER_GEOMETRY].args.user_sgprs_locs;
      stages[MESA_SHADER_GEOMETRY].info.inline_push_constant_mask =
         stages[MESA_SHADER_GEOMETRY].args.ac.inline_push_const_mask;

      stages[pre_stage].info.user_sgprs_locs = stages[MESA_SHADER_GEOMETRY].info.user_sgprs_locs;
      stages[pre_stage].info.inline_push_constant_mask = stages[MESA_SHADER_GEOMETRY].info.inline_push_constant_mask;
      stages[pre_stage].args = stages[MESA_SHADER_GEOMETRY].args;
      active_nir_stages &= ~(1 << pre_stage);
      active_nir_stages &= ~(1 << MESA_SHADER_GEOMETRY);
   }

   u_foreach_bit (i, active_nir_stages) {
      radv_declare_shader_args(compiler_info, gfx_state, &stages[i], MESA_SHADER_NONE, &debug[i]);
      stages[i].info.user_sgprs_locs = stages[i].args.user_sgprs_locs;
      stages[i].info.inline_push_constant_mask = stages[i].args.ac.inline_push_const_mask;
   }
}

static struct radv_shader_binary *
radv_create_gs_copy_shader(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *gs_stage,
                           const struct radv_graphics_state_key *gfx_state,
                           struct radv_shader_debug_info *gs_copy_debug)
{

   const struct radv_shader_info *gs_info = &gs_stage->info;
   nir_shader *nir = gs_stage->gs_copy_shader;

   nir_validate_shader(nir, "after ac_nir_create_gs_copy_shader");
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   struct radv_shader_stage gs_copy_stage = {
      .stage = MESA_SHADER_VERTEX,
      .shader_blake3 = {0},
      .key =
         {
            .optimisations_disabled = gs_stage->key.optimisations_disabled,
            .keep_statistic_info = gs_stage->key.keep_statistic_info,
            .keep_executable_info = gs_stage->key.keep_executable_info,
         },
   };
   radv_nir_shader_info_init(gs_copy_stage.stage, MESA_SHADER_FRAGMENT, &gs_copy_stage.info);
   radv_nir_shader_info_pass(compiler_info, nir, &gs_stage->layout, &gs_stage->key, gfx_state, false,
                             &gs_copy_stage.info);
   gs_copy_stage.info.wave_size = 64;      /* Wave32 not supported. */
   gs_copy_stage.info.workgroup_size = 64; /* HW VS: separate waves, no workgroups */
   gs_copy_stage.info.so = gs_info->so;
   gs_copy_stage.info.outinfo = gs_info->outinfo;
   gs_copy_stage.info.force_vrs_per_vertex = gs_info->force_vrs_per_vertex;
   gs_copy_stage.info.type = RADV_SHADER_TYPE_GS_COPY;

   radv_declare_shader_args(compiler_info, gfx_state, &gs_copy_stage, MESA_SHADER_NONE, gs_copy_debug);
   gs_copy_stage.info.user_sgprs_locs = gs_copy_stage.args.user_sgprs_locs;
   gs_copy_stage.info.inline_push_constant_mask = gs_copy_stage.args.ac.inline_push_const_mask;

   NIR_PASS(_, nir, ac_nir_lower_intrinsics_to_args, &gs_copy_stage.args.ac,
            &(ac_nir_lower_intrinsics_to_args_options){.gfx_level = compiler_info->ac->gfx_level,
                                                       .has_ls_vgpr_init_bug = compiler_info->ac->has_ls_vgpr_init_bug,
                                                       .hw_stage = AC_HW_VERTEX_SHADER,
                                                       .wave_size = 64,
                                                       .workgroup_size = 64,
                                                       .use_llvm = compiler_info->key.use_llvm});
   NIR_PASS(_, nir, radv_nir_lower_abi, compiler_info->ac->gfx_level, &gs_copy_stage, gfx_state,
            compiler_info->hw.address32_hi);

   NIR_PASS(_, nir, ac_nir_lower_global_access, compiler_info->ac->gfx_level);
   NIR_PASS(_, nir, nir_lower_int64);

   struct radv_graphics_state_key empty_gfx_state = {0};
   gs_copy_debug->dump_shader = radv_can_dump_shader(compiler_info, nir);

   if (gs_copy_debug->dump_shader)
      simple_mtx_lock(compiler_info->debug.shader_dump_mtx);

   struct radv_shader_binary *gs_copy_binary =
      radv_shader_nir_to_asm(compiler_info, &gs_copy_stage, &nir, 1, &empty_gfx_state);

   char *nir_string = NULL;
   if (gs_copy_stage.key.keep_executable_info)
      nir_string = radv_dump_nir_shaders(compiler_info, &nir, 1);

   gs_copy_debug->nir_string = nir_string;
   gs_copy_debug->stages = 1 << MESA_SHADER_VERTEX;
   radv_shader_dump_asm(compiler_info, gs_copy_debug, gs_copy_binary, &gs_copy_stage.info);

   if (gs_copy_debug->dump_shader)
      simple_mtx_unlock(compiler_info->debug.shader_dump_mtx);

   return gs_copy_binary;
}

static void
radv_graphics_shaders_nir_to_asm(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                                 const struct radv_graphics_state_key *gfx_state,
                                 VkShaderStageFlagBits active_nir_stages, struct radv_shader_debug_info *debug,
                                 struct radv_shader_binary **binaries, struct radv_shader_debug_info *gs_copy_debug,
                                 struct radv_shader_binary **gs_copy_binary)
{
   for (int s = MESA_VULKAN_SHADER_STAGES - 1; s >= 0; s--) {
      if (!(active_nir_stages & (1 << s)))
         continue;

      nir_shader *nir_shaders[2] = {stages[s].nir, NULL};
      unsigned shader_count = 1;

      /* On GFX9+, TES is merged with GS and VS is merged with TCS or GS. */
      if (compiler_info->ac->gfx_level >= GFX9 &&
          ((s == MESA_SHADER_GEOMETRY &&
            (active_nir_stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT))) ||
           (s == MESA_SHADER_TESS_CTRL && (active_nir_stages & VK_SHADER_STAGE_VERTEX_BIT)))) {
         mesa_shader_stage pre_stage;

         if (s == MESA_SHADER_GEOMETRY && (active_nir_stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)) {
            pre_stage = MESA_SHADER_TESS_EVAL;
         } else {
            pre_stage = MESA_SHADER_VERTEX;
         }

         nir_shaders[0] = stages[pre_stage].nir;
         nir_shaders[1] = stages[s].nir;
         shader_count = 2;
      }

      int64_t stage_start = os_time_get_nano();

      for (unsigned i = 0; i < shader_count; ++i)
         debug[s].dump_shader |= radv_can_dump_shader(compiler_info, nir_shaders[i]);

      bool dump_nir = debug[s].dump_shader && compiler_info->debug.dump_nir;

      if (debug[s].dump_shader) {
         simple_mtx_lock(compiler_info->debug.shader_dump_mtx);

         if (dump_nir) {
            for (uint32_t i = 0; i < shader_count; i++)
               nir_print_shader(nir_shaders[i], stderr);
         }
      }

      binaries[s] = radv_shader_nir_to_asm(compiler_info, &stages[s], nir_shaders, shader_count, gfx_state);

      /* Dump NIR after nir_to_asm, because ACO modifies it. */
      char *nir_string = NULL;
      if (stages[s].key.keep_executable_info)
         nir_string = radv_dump_nir_shaders(compiler_info, nir_shaders, shader_count);

      debug[s].nir_string = nir_string;
      for (uint32_t i = 0; i < shader_count; i++)
         debug[s].stages |= 1 << nir_shaders[i]->info.stage;

      radv_shader_dump_asm(compiler_info, &debug[s], binaries[s], &stages[s].info);

      if (debug[s].dump_shader)
         simple_mtx_unlock(compiler_info->debug.shader_dump_mtx);

      if (stages[s].key.keep_executable_info && stages[s].spirv.size) {
         debug[s].spirv = malloc(stages[s].spirv.size);
         memcpy(debug[s].spirv, stages[s].spirv.data, stages[s].spirv.size);
         debug[s].spirv_size = stages[s].spirv.size;
      }

      if (s == MESA_SHADER_GEOMETRY && !stages[s].info.is_ngg) {
         *gs_copy_binary =
            radv_create_gs_copy_shader(compiler_info, &stages[MESA_SHADER_GEOMETRY], gfx_state, gs_copy_debug);
      }

      stages[s].feedback.duration += os_time_get_nano() - stage_start;

      active_nir_stages &= ~(1 << nir_shaders[0]->info.stage);
      if (nir_shaders[1])
         active_nir_stages &= ~(1 << nir_shaders[1]->info.stage);
   }
}

static unsigned
radv_get_num_raster_vertices_per_prim(enum amd_gfx_level gfx_level, const struct radv_shader_stage *stages,
                                      const struct radv_graphics_state_key *gfx_state)
{
   unsigned vgt_outprim_type;

   /* If VS or MS is present, it means we have all pre-rasterization shaders. We can't determine
    * the raster primitive type without them.
    */
   if (!stages[MESA_SHADER_VERTEX].nir && !stages[MESA_SHADER_MESH].nir)
      return 0; /* unknown */

   /* The pre-raster primitive type is determined from enabled shaders and the input topology. */
   if (stages[MESA_SHADER_GEOMETRY].nir) {
      vgt_outprim_type = radv_conv_gl_prim_to_gs_out(stages[MESA_SHADER_GEOMETRY].nir->info.gs.output_primitive);
   } else if (stages[MESA_SHADER_TESS_EVAL].nir) {
      if (stages[MESA_SHADER_TESS_EVAL].nir->info.tess.point_mode) {
         vgt_outprim_type = V_028A6C_POINTLIST;
      } else {
         vgt_outprim_type = radv_conv_tess_prim_to_gs_out(stages[MESA_SHADER_TESS_EVAL].nir->info.tess._primitive_mode);
      }
   } else if (stages[MESA_SHADER_MESH].nir) {
      vgt_outprim_type = radv_conv_gl_prim_to_gs_out(stages[MESA_SHADER_MESH].nir->info.mesh.primitive_type);
   } else {
      if (gfx_state->ia.topology == V_008958_DI_PT_NONE)
         return 0; /* unknown */

      vgt_outprim_type = radv_conv_prim_to_gs_out(gfx_level, gfx_state->ia.topology, false);
   }

   /* The rasterized primitive type is determined from the pre-raster primitive type and the polygon mode. */
   switch (vgt_outprim_type) {
   case V_028A6C_POINTLIST:
      return 1;

   case V_028A6C_LINESTRIP:
      return 2;

   case V_028A6C_TRISTRIP:
      if (!gfx_state->rs.polygon_mode_unknown) {
         switch (gfx_state->rs.polygon_mode) {
         case VK_POLYGON_MODE_POINT:
            return 1;
         case VK_POLYGON_MODE_LINE:
            return 2;
         default:
            return 3;
         }
      }
      break;

   default:
      UNREACHABLE("invalid vgt_outprim_type");
   }

   return 0;
}

void
radv_graphics_shaders_spirv_to_nir(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                                   bool is_internal)
{
   for (unsigned s = 0; s < MESA_VULKAN_SHADER_STAGES; s++) {
      /* NIR might already have been imported from a library or found in the cache. */
      if (stages[s].stage == MESA_SHADER_NONE || stages[s].nir)
         continue;

      int64_t stage_start = os_time_get_nano();

      const struct radv_spirv_to_nir_options options = radv_get_graphics_spirv_to_nir_options(&stages[s]);
      stages[s].nir = radv_shader_spirv_to_nir(compiler_info, &stages[s], &options, is_internal);

      stages[s].feedback.duration += os_time_get_nano() - stage_start;
   }
}

void
radv_graphics_shaders_compile(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                              const struct radv_graphics_state_key *gfx_state, bool noop_fs,
                              struct radv_shader_debug_info *debug, struct radv_shader_binary **binaries,
                              struct radv_shader_debug_info *gs_copy_debug, struct radv_shader_binary **gs_copy_binary)
{
   VkShaderStageFlagBits active_nir_stages = 0;
   for (int i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      if (stages[i].nir)
         active_nir_stages |= mesa_to_vk_shader_stage(i);
   }

   if (compiler_info->ac->gfx_level < GFX11 && stages[MESA_SHADER_MESH].nir &&
       BITSET_TEST(stages[MESA_SHADER_MESH].nir->info.system_values_read, SYSTEM_VALUE_WORKGROUP_ID)) {
      nir_shader *mesh = stages[MESA_SHADER_MESH].nir;
      nir_shader *task = stages[MESA_SHADER_TASK].nir;

      /* Mesh shaders only have a 1D "vertex index" which we use
       * as "workgroup index" to emulate the 3D workgroup ID.
       */
      nir_lower_compute_system_values_options o = {
         .lower_workgroup_id_to_index = true,
         .shortcut_1d_workgroup_id = true,
         .num_workgroups[0] = task ? task->info.mesh.ts_mesh_dispatch_dimensions[0] : 0,
         .num_workgroups[1] = task ? task->info.mesh.ts_mesh_dispatch_dimensions[1] : 0,
         .num_workgroups[2] = task ? task->info.mesh.ts_mesh_dispatch_dimensions[2] : 0,
      };

      NIR_PASS(_, mesh, nir_lower_compute_system_values, &o);
   }

   radv_foreach_stage (i, active_nir_stages) {
      mesa_shader_stage next_stage;

      if (stages[i].next_stage != MESA_SHADER_NONE) {
         next_stage = stages[i].next_stage;
      } else {
         next_stage = radv_get_next_stage(i, active_nir_stages);
      }

      radv_nir_shader_info_init(i, next_stage, &stages[i].info);
   }

   /* Determine if shaders uses NGG before linking because it's needed for some NIR pass. */
   radv_fill_shader_info_ngg(compiler_info, stages, active_nir_stages);

   if (stages[MESA_SHADER_GEOMETRY].nir) {
      unsigned nir_gs_flags = nir_lower_gs_intrinsics_per_stream;

      if (stages[MESA_SHADER_GEOMETRY].info.is_ngg) {
         nir_gs_flags |= nir_lower_gs_intrinsics_count_primitives |
                         nir_lower_gs_intrinsics_count_vertices_per_primitive |
                         nir_lower_gs_intrinsics_overwrite_incomplete;
      }

      NIR_PASS(_, stages[MESA_SHADER_GEOMETRY].nir, nir_lower_gs_intrinsics, nir_gs_flags);
      NIR_PASS(_, stages[MESA_SHADER_GEOMETRY].nir, nir_lower_vars_to_ssa);
   }

   if (stages[MESA_SHADER_TESS_CTRL].nir && stages[MESA_SHADER_TESS_EVAL].nir) {
      /* Copy TCS info into the TES info */
      merge_tess_info(&stages[MESA_SHADER_TESS_EVAL].nir->info, &stages[MESA_SHADER_TESS_CTRL].nir->info);
   }

   unsigned num_raster_vertices_per_prim =
      radv_get_num_raster_vertices_per_prim(compiler_info->ac->gfx_level, stages, gfx_state);

   if (stages[MESA_SHADER_FRAGMENT].nir) {
      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_lower_fs_barycentric, gfx_state,
               num_raster_vertices_per_prim);

      /* frag_depth = gl_FragCoord.z broadcasts to all samples of the fragment shader invocation,
       * so only optimize it away if we know there is only one sample per invocation.
       * Because we don't know if sample shading is used with factor 1.0f, this means
       * we only optimize single sampled shaders.
       */
      if ((gfx_state->lib_flags & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT) &&
          !gfx_state->dynamic_rasterization_samples && gfx_state->ms.rasterization_samples == 0)
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_fragdepth);
   }

   if (stages[MESA_SHADER_VERTEX].nir && !gfx_state->vs.has_prolog)
      NIR_PASS(_, stages[MESA_SHADER_VERTEX].nir, radv_nir_optimize_vs_inputs_to_const, gfx_state);

   radv_foreach_stage (i, active_nir_stages) {
      int64_t stage_start = os_time_get_nano();

      if (i == MESA_SHADER_FRAGMENT && stages[MESA_SHADER_MESH].nir) {
         nir_foreach_shader_in_variable (var, stages[i].nir) {
            /* These variables are implicitly per-primitive when used with mesh->fragment stages
             * and this can't be determined with only the FS.
             * nir_opt_varyings relies on inputs and outputs agreeing on per-primitive.
             */
            if (var->data.location == VARYING_SLOT_PRIMITIVE_ID || var->data.location == VARYING_SLOT_VIEWPORT ||
                var->data.location == VARYING_SLOT_LAYER) {
               var->data.per_primitive = true;
            }
         }
      }

      radv_nir_lower_io(stages[i].nir);

      /* Scalarize all I/O, because nir_opt_varyings and nir_opt_vectorize_io expect all I/O to be scalarized.
       * Scalar IO is also required by ac_nir_lower_fs_input_loads.
       */
      if (i == MESA_SHADER_FRAGMENT || !stages[i].key.optimisations_disabled) {
         NIR_PASS(_, stages[i].nir, nir_lower_io_to_scalar, nir_var_shader_in | nir_var_shader_out, NULL, NULL);

         /* Eliminate useless vec->mov copies resulting from scalarization. */
         NIR_PASS(_, stages[i].nir, nir_opt_copy_prop);
         NIR_PASS(_, stages[i].nir, nir_opt_constant_folding);
      }

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   if (stages[MESA_SHADER_FRAGMENT].nir) {
      /* Inter-shader code motion in nir_opt_varyings only works with FS inputs that are loaded only once,
       * so move all input loads to the entry block, so that CSE can deduplicate them, which increases
       * the likelihood of there being only one load per FS input component.
       *
       * This only moves FS input loads to the end of the entry block.
       */
      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_move_to_top,
               nir_move_to_entry_block_only | nir_move_to_top_input_loads_simple);

      if ((!num_raster_vertices_per_prim || num_raster_vertices_per_prim == 2) &&
          gfx_state->smooth_lines_may_be_enabled)
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_lower_poly_line_smooth, false, RADV_NUM_SMOOTH_AA_SAMPLES);

      if (!gfx_state->ps.color_outputs_need_epilog)
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_remap_color_attachment, gfx_state);

      const bool vrs_may_be_enabled =
         gfx_state->vrs_may_be_enabled &&
         !radv_force_disable_vrs(compiler_info->ac->gfx_level, stages[MESA_SHADER_FRAGMENT].nir, gfx_state);
      const bool max_sample_shading =
         gfx_state->ms.max_sample_shading_enable || stages[MESA_SHADER_FRAGMENT].nir->info.fs.uses_sample_shading;

      ac_nir_lower_ps_early_options opts = {0};
      opts.msaa_disabled = !gfx_state->dynamic_rasterization_samples && !gfx_state->ms.rasterization_samples;
      opts.vrs_disabled = !vrs_may_be_enabled;
      opts.custom_sample_locations_disabled = !gfx_state->custom_sample_locs_may_be_enabled;
      opts.dynamic_rasterization_samples = gfx_state->dynamic_rasterization_samples;
      opts.sample_shading = max_sample_shading;
      opts.alpha_func = COMPARE_FUNC_ALWAYS;

      opts.color_mask = 0xffffffff;
      if (!gfx_state->ps.color_outputs_need_epilog) {
         opts.color_mask = gfx_state->ps.epilog.colors_needed;
         if (gfx_state->ps.mrt0_alpha_is_dead)
            opts.color_mask &= ~0x8u;
         opts.color_no_signed_zero = gfx_state->ps.epilog.no_signed_zero;
         opts.kill_z = gfx_state->ps.epilog.ignore_depth_output;
         opts.kill_stencil = gfx_state->ps.epilog.ignore_stencil_output;
         opts.lower_1bit_sample_mask_to_discard = gfx_state->ps.epilog.lower_1bit_sample_mask_to_discard;
      }

      switch (num_raster_vertices_per_prim) {
      case 1:
      case 2:
         opts.force_front_face = 1;
         break;
      case 3:
         if (gfx_state->rs.cull_mode == VK_CULL_MODE_FRONT_BIT)
            opts.force_front_face = -1;
         else if (gfx_state->rs.cull_mode == VK_CULL_MODE_BACK_BIT)
            opts.force_front_face = 1;
         break;
      }
      opts.has_front_face_select = true;

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, ac_nir_lower_ps_early, &opts);

      if (!gfx_state->ps.color_outputs_need_epilog) {
         /* Cleanup after ac_nir_lower_ps_early might have removed outputs. */
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_copy_prop);
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_dce);
         NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_dead_cf);
      }

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_lower_fs_input_attachment);

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_copy_prop);
      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_dce);
      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, nir_opt_cse);

      /* nir_opt_dce must be run before this pass. */
      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, radv_nir_lower_opt_fs_frag_pos, vrs_may_be_enabled,
               gfx_state->ms.sample_shading_enable || stages[MESA_SHADER_FRAGMENT].nir->info.fs.uses_sample_shading);

      ac_nir_lower_sample_mask_in_options lower_sample_mask_in_options = {0};

      if (max_sample_shading) {
         lower_sample_mask_in_options.behavior = ac_nir_lower_samplemask_sample_shading_max;
      } else if (gfx_state->ms.sample_shading_enable) {
         lower_sample_mask_in_options.behavior = ac_nir_lower_samplemask_sample_shading_partial;
         lower_sample_mask_in_options.ps_iter_samples = gfx_state->ms.ps_iter_samples;
      } else if (!gfx_state->dynamic_rasterization_samples && gfx_state->ms.rasterization_samples == 0) {
         lower_sample_mask_in_options.behavior = ac_nir_lower_samplemask_1sample_no_vrs;
      } else {
         lower_sample_mask_in_options.behavior = ac_nir_lower_samplemask_unknown_states_no_sample_shading;
      }

      NIR_PASS(_, stages[MESA_SHADER_FRAGMENT].nir, ac_nir_lower_sample_mask_in, &lower_sample_mask_in_options);
   }

   radv_foreach_stage (i, active_nir_stages) {
      if (!radv_is_last_vgt_stage(&stages[i]))
         continue;

      /* Lower mesh shader draw ID to zero prevent app bugs from triggering undefined behaviour. */
      if (i == MESA_SHADER_MESH && stages[i].info.ms.has_task &&
          BITSET_TEST(stages[i].nir->info.system_values_read, SYSTEM_VALUE_DRAW_ID))
         radv_nir_lower_draw_id_to_zero(stages[i].nir);

      if (i != MESA_SHADER_MESH && radv_should_export_multiview(&stages[i]))
         NIR_PASS(_, stages[i].nir, radv_nir_export_multiview);

      uint64_t remove_as_varying = VARYING_BIT_PSIZ | VARYING_BIT_LAYER;
      uint64_t remove_as_sysval = 0;

      /* Remove all varyings when the fragment shader is a noop. */
      if (noop_fs)
         remove_as_varying |= ~0ull;

      /* Remove PSIZ from shaders when it's not needed.
       * This is typically produced by translation layers like Zink or d3d9 DXVK.
       */
      if (num_raster_vertices_per_prim > 1)
         remove_as_sysval |= VARYING_BIT_PSIZ;

      if (gfx_state->rs.rasterizer_discard) {
         remove_as_sysval |= ~0ull;
         remove_as_varying |= ~0ull;
      }

      NIR_PASS(_, stages[i].nir, nir_remove_outputs, MESA_SHADER_FRAGMENT, remove_as_varying, remove_as_sysval);
      break;
   }

   radv_foreach_stage (i, active_nir_stages) {
      int64_t stage_start = os_time_get_nano();

      ac_nir_lower_indirect_derefs_early(stages[i].nir);
      NIR_PASS(_, stages[i].nir, nir_lower_vars_to_ssa);

      radv_optimize_nir(stages[i].nir, stages[i].key.optimisations_disabled);

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   /* Optimize varyings on lowered shader I/O (more efficient than optimizing I/O derefs). */
   radv_graphics_shaders_link_varyings(stages, compiler_info->ac->gfx_level);

   /* Optimize constant clip/cull distance after linking to operate on scalar io in the last
    * pre raster stage.
    */
   radv_foreach_stage (
      i, active_nir_stages &
            (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT)) {
      if (stages[i].key.optimisations_disabled)
         continue;

      int64_t stage_start = os_time_get_nano();

      NIR_PASS(_, stages[i].nir, nir_opt_clip_cull_const);

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   radv_foreach_stage (i, active_nir_stages) {
      int64_t stage_start = os_time_get_nano();

      /* Indirect lowering must be called after the radv_optimize_nir() loop
       * has been called at least once. Otherwise indirect lowering can
       * bloat the instruction count of the loop and cause it to be
       * considered too large for unrolling.
       *
       * We want to do this as late as possible because scratch access isn't
       * very optimizable. We lower smaller arrays to SSA earlier with
       * ac_nir_lower_indirect_derefs_early, because that can actually enable
       * optimizations.
       */
      bool indirect_derefs_lowered = false;
      NIR_PASS(indirect_derefs_lowered, stages[i].nir, ac_nir_lower_indirect_derefs);
      NIR_PASS(_, stages[i].nir, nir_lower_vars_to_ssa);
      if (indirect_derefs_lowered && !stages[i].key.optimisations_disabled)
         radv_optimize_nir(stages[i].nir, false);

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   radv_fill_shader_info(compiler_info, gfx_state, stages, active_nir_stages);

   /* Remove the primitive shading rate output if VRS flat shading overrides it. */
   radv_foreach_stage (i, active_nir_stages) {
      if (!radv_is_last_vgt_stage(&stages[i]))
         continue;

      struct radv_shader_stage *fs_stage = &stages[MESA_SHADER_FRAGMENT];

      if (fs_stage->nir && (fs_stage->info.ps.allow_flat_shading || fs_stage->info.ps.force_disable_vrs)) {
         stages[i].info.force_vrs_per_vertex = false;

         if (stages[i].info.outinfo.writes_primitive_shading_rate ||
             stages[i].info.outinfo.writes_primitive_shading_rate_per_primitive) {
            NIR_PASS(_, stages[i].nir, nir_remove_outputs, MESA_SHADER_FRAGMENT, 0, VARYING_BIT_PRIMITIVE_SHADING_RATE);

            stages[i].info.outinfo.writes_primitive_shading_rate = false;
            stages[i].info.outinfo.writes_primitive_shading_rate_per_primitive = false;
            stages[i].nir->info.outputs_written &= ~VARYING_BIT_PRIMITIVE_SHADING_RATE;
            stages[i].nir->info.per_primitive_outputs &= ~VARYING_BIT_PRIMITIVE_SHADING_RATE;
         }
      } else if (fs_stage->nir && fs_stage->info.ps.disallow_force_vrs_per_vertex &&
                 stages[i].info.force_vrs_per_vertex) {
         stages[i].info.force_vrs_per_vertex = false;
         stages[i].info.outinfo.writes_primitive_shading_rate = false;

         assert(!(stages[i].nir->info.outputs_written & ~stages[i].nir->info.per_primitive_outputs &
                  VARYING_BIT_PRIMITIVE_SHADING_RATE));
      }
      break;
   }

   radv_declare_graphics_shaders_args(compiler_info, stages, gfx_state, active_nir_stages, debug);

   radv_foreach_stage (i, active_nir_stages) {
      int64_t stage_start = os_time_get_nano();

      radv_postprocess_nir(compiler_info, gfx_state, &stages[i]);

      stages[i].feedback.duration += os_time_get_nano() - stage_start;
   }

   if (stages[MESA_SHADER_VERTEX].nir || stages[MESA_SHADER_TESS_EVAL].nir) {
      struct radv_shader_stage *es_stage =
         stages[MESA_SHADER_TESS_EVAL].nir ? &stages[MESA_SHADER_TESS_EVAL] : &stages[MESA_SHADER_VERTEX];
      struct radv_shader_stage *gs_stage = stages[MESA_SHADER_GEOMETRY].nir ? &stages[MESA_SHADER_GEOMETRY] : NULL;
      struct radv_shader_stage *stage = gs_stage ? gs_stage : es_stage;

      if ((gs_stage ? gs_stage : es_stage)->info.is_ngg) {
         gfx10_get_ngg_info(compiler_info, &es_stage->info, gs_stage ? &gs_stage->info : NULL, &stage->info.ngg_info);
         stage->info.nir_shared_size = stage->info.ngg_info.lds_size;
      }
   }

   if (stages[MESA_SHADER_GEOMETRY].nir && !stages[MESA_SHADER_GEOMETRY].info.is_ngg)
      radv_get_legacy_gs_info(compiler_info, NULL, &stages[MESA_SHADER_GEOMETRY].info);

   /* Compile NIR shaders to AMD assembly. */
   radv_graphics_shaders_nir_to_asm(compiler_info, stages, gfx_state, active_nir_stages, debug, binaries, gs_copy_debug,
                                    gs_copy_binary);
}

struct radv_vgt_shader_key
radv_get_vgt_shader_key(const struct radv_device *device, struct radv_shader **shaders,
                        const struct radv_shader *gs_copy_shader)
{
   uint8_t hs_size = 64, gs_size = 64, vs_size = 64;
   struct radv_shader *last_vgt_shader = NULL;
   struct radv_vgt_shader_key key;

   memset(&key, 0, sizeof(key));

   if (shaders[MESA_SHADER_GEOMETRY]) {
      last_vgt_shader = shaders[MESA_SHADER_GEOMETRY];
   } else if (shaders[MESA_SHADER_TESS_EVAL]) {
      last_vgt_shader = shaders[MESA_SHADER_TESS_EVAL];
   } else if (shaders[MESA_SHADER_VERTEX]) {
      last_vgt_shader = shaders[MESA_SHADER_VERTEX];
   } else {
      assert(shaders[MESA_SHADER_MESH]);
      last_vgt_shader = shaders[MESA_SHADER_MESH];
   }

   vs_size = gs_size = last_vgt_shader->info.wave_size;
   if (gs_copy_shader)
      vs_size = gs_copy_shader->info.wave_size;

   if (shaders[MESA_SHADER_TESS_CTRL])
      hs_size = shaders[MESA_SHADER_TESS_CTRL]->info.wave_size;

   key.tess = !!shaders[MESA_SHADER_TESS_CTRL];
   key.gs = !!shaders[MESA_SHADER_GEOMETRY];
   key.mesh = !!shaders[MESA_SHADER_MESH];
   if (last_vgt_shader->info.is_ngg) {
      key.ngg = 1;
      key.ngg_passthrough = last_vgt_shader->info.is_ngg_passthrough;
      key.ngg_wave_id_en = last_vgt_shader->info.ngg_wave_id_en;
   }

   key.hs_wave32 = hs_size == 32;
   key.vs_wave32 = vs_size == 32;
   key.gs_wave32 = gs_size == 32;

   return key;
}

bool
radv_needs_null_export_workaround(const struct radv_device *device, const struct radv_shader *ps,
                                  unsigned custom_blend_mode)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const enum amd_gfx_level gfx_level = pdev->info.gfx_level;

   if (!ps)
      return false;

   /* Ensure that some export memory is always allocated, for two reasons:
    *
    * 1) Correctness: The hardware ignores the EXEC mask if no export
    *    memory is allocated, so KILL and alpha test do not work correctly
    *    without this.
    * 2) Performance: Every shader needs at least a NULL export, even when
    *    it writes no color/depth output. The NULL export instruction
    *    stalls without this setting.
    *
    * Don't add this to CB_SHADER_MASK.
    *
    * GFX10 supports pixel shaders without exports by setting both the
    * color and Z formats to SPI_SHADER_ZERO. The hw will skip export
    * instructions if any are present.
    *
    * GFX11 requires one color output, otherwise the DCC decompression does nothing.
    *
    * Primitive Ordered Pixel Shading also requires an export, otherwise interlocking doesn't work
    * correctly before GFX11, and a hang happens on GFX11.
    *
    * fbfetch Z/S reads via input attachments expect pre-depth values (late-Z) to be returned but
    * Z_ORDER has no effect without any exports.
    */
   return (gfx_level <= GFX9 || ps->info.ps.can_discard || ps->info.ps.pops || ps->info.ps.uses_fbfetch_output ||
           (custom_blend_mode == V_028808_CB_DCC_DECOMPRESS_GFX11 && gfx_level >= GFX11)) &&
          !radv_ps_writes_mrtz(&ps->info);
}

struct radv_shader_part *
radv_create_vs_prolog(struct radv_device *device, const struct radv_vs_prolog_key *key)
{
   const struct radv_compiler_info *compiler_info = &device->compiler_info;
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   bool dump_shader = RADV_DEBUG(instance, DUMP_PROLOGS);
   bool keep_shader_info = radv_device_fault_detection_enabled(device);

   struct radv_shader_part *prolog;

   struct radv_shader_stage stage = {0};
   stage.stage = key->next_stage;
   stage.info.stage = MESA_SHADER_VERTEX;
   stage.info.wave_size = key->wave32 ? 32 : 64;
   stage.info.vs.needs_instance_id = true;
   stage.info.vs.needs_base_instance = true;
   stage.info.vs.use_per_attribute_vb_descs = true;
   stage.info.vs.vb_desc_usage_mask = BITFIELD_MASK(key->num_attributes);
   stage.info.vs.has_prolog = true;
   stage.info.vs.as_ls = key->as_ls;
   stage.info.is_ngg = key->is_ngg;

   struct radv_graphics_state_key gfx_state = {0};

   radv_declare_shader_args(compiler_info, &gfx_state, &stage,
                            key->next_stage != MESA_SHADER_VERTEX ? MESA_SHADER_VERTEX : MESA_SHADER_NONE, NULL);

   stage.info.user_sgprs_locs = stage.args.user_sgprs_locs;
   stage.info.inline_push_constant_mask = stage.args.ac.inline_push_const_mask;

#if AMD_LLVM_AVAILABLE
   if (dump_shader || keep_shader_info)
      ac_init_llvm_once();
#endif

   struct radv_shader_part_binary *binary = NULL;
   struct radv_shader_stage_key stage_key = {0};
   struct aco_shader_info ac_info;
   struct aco_vs_prolog_info ac_prolog_info;
   struct aco_compiler_options ac_opts;
   stage_key.keep_executable_info = keep_shader_info;
   radv_aco_convert_shader_info(&ac_info, &stage.info, &stage.args, compiler_info);
   radv_aco_fill_compiler_options(&ac_opts, compiler_info, &stage_key, &gfx_state, false, dump_shader);
   radv_aco_convert_vs_prolog_key(&ac_prolog_info, key, &stage.args);
   aco_compile_vs_prolog(&ac_opts, &ac_info, &ac_prolog_info, &stage.args.ac, &radv_aco_build_shader_part,
                         (void **)&binary);

   prolog = radv_shader_part_create(device, binary, stage.info.wave_size);
   if (!prolog)
      goto fail;

   prolog->key.vs = *key;
   prolog->nontrivial_divisors = key->nontrivial_divisors;

   if (dump_shader) {
      fprintf(stderr, "Vertex prolog");
      fprintf(stderr, "\ndisasm:\n%s\n", prolog->disasm_string);
   }

   free(binary);

   return prolog;

fail:
   free(binary);
   return NULL;
}

struct radv_shader_part *
radv_create_ps_epilog(struct radv_device *device, const struct radv_ps_epilog_key *key,
                      struct radv_shader_part_binary **binary_out)
{
   const struct radv_compiler_info *compiler_info = &device->compiler_info;
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   bool dump_shader = RADV_DEBUG(instance, DUMP_EPILOGS);
   bool keep_shader_info = radv_device_fault_detection_enabled(device);

   struct radv_shader_part *epilog;
   struct radv_shader_args args = {0};

   struct radv_shader_info info = {0};
   info.stage = MESA_SHADER_FRAGMENT;
   info.wave_size = pdev->ps_wave_size;
   info.workgroup_size = 64;

   radv_declare_ps_epilog_args(compiler_info, key, &args);

#if AMD_LLVM_AVAILABLE
   if (dump_shader || keep_shader_info)
      ac_init_llvm_once();
#endif

   struct radv_shader_part_binary *binary = NULL;
   struct radv_shader_stage_key stage_key = {0};
   struct aco_shader_info ac_info;
   struct aco_ps_epilog_info ac_epilog_info = {0};
   struct aco_compiler_options ac_opts;
   stage_key.keep_executable_info = keep_shader_info;
   radv_aco_convert_shader_info(&ac_info, &info, &args, compiler_info);
   radv_aco_fill_compiler_options(&ac_opts, compiler_info, &stage_key, NULL, false, dump_shader);
   radv_aco_convert_ps_epilog_key(&ac_epilog_info, key, &args);
   aco_compile_ps_epilog(&ac_opts, &ac_info, &ac_epilog_info, &args.ac, &radv_aco_build_shader_part, (void **)&binary);

   binary->info.spi_shader_col_format = key->spi_shader_col_format;
   binary->info.cb_shader_mask = ac_get_cb_shader_mask(key->spi_shader_col_format);
   binary->info.db_shader_control =
      S_02880C_Z_EXPORT_ENABLE(key->has_depth_output && !key->ignore_depth_output) |
      S_02880C_STENCIL_TEST_VAL_EXPORT_ENABLE(key->has_stencil_output && !key->ignore_stencil_output) |
      S_02880C_MASK_EXPORT_ENABLE(key->has_sample_mask_output && !key->lower_1bit_sample_mask_to_discard) |
      S_02880C_KILL_ENABLE(key->lower_1bit_sample_mask_to_discard);
   binary->info.spi_shader_z_format = key->spi_shader_z_format;

   epilog = radv_shader_part_create(device, binary, info.wave_size);
   if (!epilog)
      goto fail;

   epilog->key.ps = *key;

   if (dump_shader) {
      fprintf(stderr, "Fragment epilog");
      fprintf(stderr, "\ndisasm:\n%s\n", epilog->disasm_string);
   }

   if (binary_out) {
      *binary_out = binary;
   } else {
      free(binary);
   }

   return epilog;

fail:
   free(binary);
   return NULL;
}
