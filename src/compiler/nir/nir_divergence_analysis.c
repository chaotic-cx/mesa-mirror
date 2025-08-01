/*
 * Copyright © 2018 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

#include "nir.h"

/* This pass computes for each ssa definition if it is uniform.
 * That is, the variable has the same value for all invocations
 * of the group.
 *
 * If the shader is not in LCSSA-form, passes need to use nir_src_is_divergent()
 * instead of reading the value from src->ssa->divergent as without LCSSA a src
 * can have a different divergence than the corresponding SSA-def.
 *
 * This algorithm implements "The Simple Divergence Analysis" from
 * Diogo Sampaio, Rafael De Souza, Sylvain Collange, Fernando Magno Quintão Pereira.
 * Divergence Analysis.  ACM Transactions on Programming Languages and Systems (TOPLAS),
 * ACM, 2013, 35 (4), pp.13:1-13:36. <10.1145/2523815>. <hal-00909072v2>
 */

struct divergence_state {
   const gl_shader_stage stage;
   nir_shader *shader;
   nir_function_impl *impl;
   nir_divergence_options options;
   nir_loop *loop;
   bool loop_all_invariant;

   /** current control flow state */
   /* True if some loop-active invocations might take a different control-flow path.
    * A divergent break does not cause subsequent control-flow to be considered
    * divergent because those invocations are no longer active in the loop.
    * For a divergent if, both sides are considered divergent flow because
    * the other side is still loop-active. */
   bool divergent_loop_cf;
   /* True if a divergent continue happened since the loop header */
   bool divergent_loop_continue;
   /* True if a divergent break happened since the loop header */
   bool divergent_loop_break;

   /* True if we visit the block for the fist time */
   bool first_visit;
   /* True if we visit a block that is dominated by a loop with a divergent break */
   bool consider_loop_invariance;
};

static bool
visit_cf_list(struct exec_list *list, struct divergence_state *state);

bool
nir_src_is_divergent(nir_src *src)
{
   if (src->ssa->divergent)
      return true;

   nir_cf_node *use_node = nir_src_get_block(src)->cf_node.parent;
   nir_cf_node *def_node = nir_def_block(src->ssa)->cf_node.parent;

   /* Short-cut the common case. */
   if (def_node == use_node)
      return false;

   /* If the source was computed in a divergent loop, and is not
    * loop-invariant, then it must also be considered divergent.
    */
   bool loop_invariant = src->ssa->loop_invariant;
   while (def_node) {
      if (def_node->type == nir_cf_node_loop) {
         /* Check whether the use is inside this loop. */
         for (nir_cf_node *node = use_node; node != NULL; node = node->parent) {
            if (def_node == node)
               return false;
         }

         /* Because the use is outside of this loop, it is divergent. */
         if (nir_cf_node_as_loop(def_node)->divergent_break && !loop_invariant)
            return true;

         /* For outer loops, consider this variable not loop invariant. */
         loop_invariant = false;
      }

      def_node = def_node->parent;
   }

   return false;
}

static inline bool
src_divergent(nir_src src, struct divergence_state *state)
{
   if (!state->consider_loop_invariance)
      return src.ssa->divergent;

   return nir_src_is_divergent(&src);
}

static inline bool
src_invariant(nir_src *src, void *loop)
{
   nir_block *first_block = nir_loop_first_block(loop);

   /* Invariant if SSA is defined before the current loop. */
   if (nir_def_block(src->ssa)->index < first_block->index)
      return true;

   if (!src->ssa->loop_invariant)
      return false;

   /* The value might be defined in a nested loop. */
   nir_cf_node *cf_node = nir_def_block(src->ssa)->cf_node.parent;
   while (cf_node->type != nir_cf_node_loop)
      cf_node = cf_node->parent;

   return nir_cf_node_as_loop(cf_node) == loop;
}

static bool
visit_alu(nir_alu_instr *instr, struct divergence_state *state)
{
   if (instr->def.divergent)
      return false;

   unsigned num_src = nir_op_infos[instr->op].num_inputs;

   for (unsigned i = 0; i < num_src; i++) {
      if (src_divergent(instr->src[i].src, state)) {
         instr->def.divergent = true;
         return true;
      }
   }

   return false;
}

/* On some HW uniform loads where there is a pending store/atomic from another
 * wave can "tear" so that different invocations see the pre-store value and
 * the post-store value even though they are loading from the same location.
 * This means we have to assume it's not uniform unless it's readonly.
 *
 * TODO The Vulkan memory model is much more strict here and requires an
 * atomic or volatile load for the data race to be valid, which could allow us
 * to do better if it's in use, however we currently don't have that
 * information plumbed through.
 */
static bool
load_may_tear(struct divergence_state *state, nir_intrinsic_instr *instr)
{
   return (state->options & nir_divergence_uniform_load_tears) &&
          !(nir_intrinsic_access(instr) & ACCESS_NON_WRITEABLE);
}

static bool
visit_intrinsic(nir_intrinsic_instr *instr, struct divergence_state *state)
{
   if (!nir_intrinsic_infos[instr->intrinsic].has_dest)
      return false;

   if (instr->def.divergent)
      return false;

   nir_divergence_options options = state->options;
   gl_shader_stage stage = state->stage;
   bool is_divergent = false;
   switch (instr->intrinsic) {
   case nir_intrinsic_shader_clock:
   case nir_intrinsic_ballot:
   case nir_intrinsic_ballot_relaxed:
   case nir_intrinsic_as_uniform:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_read_invocation_cond_ir3:
   case nir_intrinsic_read_getlast_ir3:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_first_invocation:
   case nir_intrinsic_last_invocation:
   case nir_intrinsic_load_subgroup_id:
   case nir_intrinsic_shared_append_amd:
   case nir_intrinsic_shared_consume_amd:
   case nir_intrinsic_load_sm_id_nv:
   case nir_intrinsic_load_warp_id_nv:
      /* VS/TES/GS invocations of the same primitive can be in different
       * subgroups, so subgroup ops are always divergent between vertices of
       * the same primitive.
       */
      is_divergent = state->options & nir_divergence_vertex;
      break;

   /* Intrinsics which are always uniform */
   case nir_intrinsic_load_preamble:
   case nir_intrinsic_load_push_constant:
   case nir_intrinsic_load_push_constant_zink:
   case nir_intrinsic_load_work_dim:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_workgroup_size:
   case nir_intrinsic_load_num_subgroups:
   case nir_intrinsic_load_ray_launch_size:
   case nir_intrinsic_load_sbt_base_amd:
   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_subgroup_id_shift_ir3:
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_draw_id:
   case nir_intrinsic_load_is_indexed_draw:
   case nir_intrinsic_load_viewport_scale:
   case nir_intrinsic_load_user_clip_plane:
   case nir_intrinsic_load_viewport_x_scale:
   case nir_intrinsic_load_viewport_y_scale:
   case nir_intrinsic_load_viewport_z_scale:
   case nir_intrinsic_load_viewport_offset:
   case nir_intrinsic_load_viewport_x_offset:
   case nir_intrinsic_load_viewport_y_offset:
   case nir_intrinsic_load_viewport_z_offset:
   case nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd:
   case nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd:
   case nir_intrinsic_load_blend_const_color_a_float:
   case nir_intrinsic_load_blend_const_color_b_float:
   case nir_intrinsic_load_blend_const_color_g_float:
   case nir_intrinsic_load_blend_const_color_r_float:
   case nir_intrinsic_load_blend_const_color_rgba:
   case nir_intrinsic_load_blend_const_color_aaaa8888_unorm:
   case nir_intrinsic_load_blend_const_color_rgba8888_unorm:
   case nir_intrinsic_load_line_width:
   case nir_intrinsic_load_aa_line_width:
   case nir_intrinsic_load_xfb_address:
   case nir_intrinsic_load_rasterization_stream:
   case nir_intrinsic_load_num_vertices:
   case nir_intrinsic_load_fb_layers_v3d:
   case nir_intrinsic_load_fep_w_v3d:
   case nir_intrinsic_load_tcs_num_patches_amd:
   case nir_intrinsic_load_tcs_tess_levels_to_tes_amd:
   case nir_intrinsic_load_tcs_primitive_mode_amd:
   case nir_intrinsic_load_patch_vertices_in:
   case nir_intrinsic_load_ring_tess_factors_amd:
   case nir_intrinsic_load_ring_tess_offchip_amd:
   case nir_intrinsic_load_ring_tess_factors_offset_amd:
   case nir_intrinsic_load_ring_tess_offchip_offset_amd:
   case nir_intrinsic_load_ring_mesh_scratch_amd:
   case nir_intrinsic_load_ring_mesh_scratch_offset_amd:
   case nir_intrinsic_load_ring_esgs_amd:
   case nir_intrinsic_load_ring_es2gs_offset_amd:
   case nir_intrinsic_load_ring_task_draw_amd:
   case nir_intrinsic_load_ring_task_payload_amd:
   case nir_intrinsic_load_sample_positions_amd:
   case nir_intrinsic_load_rasterization_samples_amd:
   case nir_intrinsic_load_ring_gsvs_amd:
   case nir_intrinsic_load_ring_gs2vs_offset_amd:
   case nir_intrinsic_load_streamout_config_amd:
   case nir_intrinsic_load_streamout_write_index_amd:
   case nir_intrinsic_load_streamout_offset_amd:
   case nir_intrinsic_load_task_ring_entry_amd:
   case nir_intrinsic_load_ring_attr_amd:
   case nir_intrinsic_load_ring_attr_offset_amd:
   case nir_intrinsic_load_provoking_vtx_amd:
   case nir_intrinsic_load_sample_positions_pan:
   case nir_intrinsic_load_shader_output_pan:
   case nir_intrinsic_load_workgroup_num_input_vertices_amd:
   case nir_intrinsic_load_workgroup_num_input_primitives_amd:
   case nir_intrinsic_load_pipeline_stat_query_enabled_amd:
   case nir_intrinsic_load_prim_gen_query_enabled_amd:
   case nir_intrinsic_load_prim_xfb_query_enabled_amd:
   case nir_intrinsic_load_merged_wave_info_amd:
   case nir_intrinsic_load_clamp_vertex_color_amd:
   case nir_intrinsic_load_cull_front_face_enabled_amd:
   case nir_intrinsic_load_cull_back_face_enabled_amd:
   case nir_intrinsic_load_cull_ccw_amd:
   case nir_intrinsic_load_cull_small_triangles_enabled_amd:
   case nir_intrinsic_load_cull_small_lines_enabled_amd:
   case nir_intrinsic_load_cull_any_enabled_amd:
   case nir_intrinsic_load_cull_small_triangle_precision_amd:
   case nir_intrinsic_load_cull_small_line_precision_amd:
   case nir_intrinsic_load_user_data_amd:
   case nir_intrinsic_load_force_vrs_rates_amd:
   case nir_intrinsic_load_tess_level_inner_default:
   case nir_intrinsic_load_tess_level_outer_default:
   case nir_intrinsic_load_scalar_arg_amd:
   case nir_intrinsic_load_smem_amd:
   case nir_intrinsic_load_resume_shader_address_amd:
   case nir_intrinsic_load_reloc_const_intel:
   case nir_intrinsic_load_btd_global_arg_addr_intel:
   case nir_intrinsic_load_btd_local_arg_addr_intel:
   case nir_intrinsic_load_inline_data_intel:
   case nir_intrinsic_load_ray_num_dss_rt_stacks_intel:
   case nir_intrinsic_load_lshs_vertex_stride_amd:
   case nir_intrinsic_load_esgs_vertex_stride_amd:
   case nir_intrinsic_load_hs_out_patch_data_offset_amd:
   case nir_intrinsic_load_clip_half_line_width_amd:
   case nir_intrinsic_load_num_vertices_per_primitive_amd:
   case nir_intrinsic_load_streamout_buffer_amd:
   case nir_intrinsic_load_ordered_id_amd:
   case nir_intrinsic_load_gs_wave_id_amd:
   case nir_intrinsic_load_provoking_vtx_in_prim_amd:
   case nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd:
   case nir_intrinsic_load_btd_shader_type_intel:
   case nir_intrinsic_load_base_global_invocation_id:
   case nir_intrinsic_load_base_workgroup_id:
   case nir_intrinsic_load_alpha_reference_amd:
   case nir_intrinsic_load_ubo_uniform_block_intel:
   case nir_intrinsic_load_ssbo_uniform_block_intel:
   case nir_intrinsic_load_shared_uniform_block_intel:
   case nir_intrinsic_load_barycentric_optimize_amd:
   case nir_intrinsic_load_poly_line_smooth_enabled:
   case nir_intrinsic_load_rasterization_primitive_amd:
   case nir_intrinsic_unit_test_uniform_amd:
   case nir_intrinsic_load_global_constant_uniform_block_intel:
   case nir_intrinsic_load_debug_log_desc_amd:
   case nir_intrinsic_load_xfb_state_address_gfx12_amd:
   case nir_intrinsic_cmat_length:
   case nir_intrinsic_load_vs_primitive_stride_ir3:
   case nir_intrinsic_load_vs_vertex_stride_ir3:
   case nir_intrinsic_load_hs_patch_stride_ir3:
   case nir_intrinsic_load_tess_factor_base_ir3:
   case nir_intrinsic_load_tess_param_base_ir3:
   case nir_intrinsic_load_primitive_location_ir3:
   case nir_intrinsic_preamble_start_ir3:
   case nir_intrinsic_optimization_barrier_sgpr_amd:
   case nir_intrinsic_load_fbfetch_image_fmask_desc_amd:
   case nir_intrinsic_load_fbfetch_image_desc_amd:
   case nir_intrinsic_load_polygon_stipple_buffer_amd:
   case nir_intrinsic_load_tcs_mem_attrib_stride:
   case nir_intrinsic_load_printf_buffer_address:
   case nir_intrinsic_load_printf_buffer_size:
   case nir_intrinsic_load_core_id_agx:
   case nir_intrinsic_load_samples_log2_agx:
   case nir_intrinsic_load_active_subgroup_count_agx:
   case nir_intrinsic_load_root_agx:
   case nir_intrinsic_load_descriptor_set_agx:
   case nir_intrinsic_load_sm_count_nv:
   case nir_intrinsic_load_warps_per_sm_nv:
   case nir_intrinsic_load_fs_msaa_intel:
   case nir_intrinsic_load_constant_base_ptr:
   case nir_intrinsic_load_const_buf_base_addr_lvp:
   case nir_intrinsic_load_max_polygon_intel:
   case nir_intrinsic_load_ray_base_mem_addr_intel:
   case nir_intrinsic_load_ray_hw_stack_size_intel:
   case nir_intrinsic_load_per_primitive_remap_intel:
      is_divergent = false;
      break;

   /* This is divergent because it specifically loads sequential values into
    * successive SIMD lanes.
    */
   case nir_intrinsic_load_global_block_intel:
      is_divergent = true;
      break;

   case nir_intrinsic_decl_reg:
   case nir_intrinsic_load_sysval_nv:
      is_divergent = nir_intrinsic_divergent(instr);
      break;

   /* Intrinsics with divergence depending on shader stage and hardware */
   case nir_intrinsic_load_shader_record_ptr:
      is_divergent = !(options & nir_divergence_shader_record_ptr_uniform);
      break;
   case nir_intrinsic_load_frag_shading_rate:
      is_divergent = !(options & nir_divergence_single_frag_shading_rate_per_subgroup);
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_primitive_input:
      is_divergent = src_divergent(instr->src[0], state);

      if (stage == MESA_SHADER_FRAGMENT) {
         is_divergent |= !(options & nir_divergence_single_prim_per_subgroup);
      } else if (stage == MESA_SHADER_TESS_EVAL) {
         /* Patch input loads are uniform between vertices of the same primitive. */
         if (state->options & nir_divergence_vertex)
            is_divergent = false;
         else
            is_divergent |= !(options & nir_divergence_single_patch_per_tes_subgroup);
      } else {
         is_divergent = true;
      }
      break;
   case nir_intrinsic_load_attribute_pan:
      assert(stage == MESA_SHADER_VERTEX);
      is_divergent = src_divergent(instr->src[0], state) ||
                     src_divergent(instr->src[1], state) ||
                     src_divergent(instr->src[2], state);
      break;
   case nir_intrinsic_load_per_vertex_input:
      is_divergent = src_divergent(instr->src[0], state) ||
                     src_divergent(instr->src[1], state);
      if (stage == MESA_SHADER_TESS_CTRL)
         is_divergent |= !(options & nir_divergence_single_patch_per_tcs_subgroup);
      if (stage == MESA_SHADER_TESS_EVAL)
         is_divergent |= !(options & nir_divergence_single_patch_per_tes_subgroup);
      else
         is_divergent = true;
      break;
   case nir_intrinsic_load_input_vertex:
      is_divergent = src_divergent(instr->src[1], state);
      assert(stage == MESA_SHADER_FRAGMENT);
      is_divergent |= !(options & nir_divergence_single_prim_per_subgroup);
      break;
   case nir_intrinsic_load_output:
      is_divergent = src_divergent(instr->src[0], state);
      switch (stage) {
      case MESA_SHADER_TESS_CTRL:
         is_divergent |= !(options & nir_divergence_single_patch_per_tcs_subgroup);
         break;
      case MESA_SHADER_FRAGMENT:
         is_divergent = true;
         break;
      case MESA_SHADER_TASK:
      case MESA_SHADER_MESH:
         /* NV_mesh_shader only (EXT_mesh_shader does not allow loading outputs).
          * Divergent if src[0] is, so nothing else to do.
          */
         break;
      default:
         UNREACHABLE("Invalid stage for load_output");
      }
      break;
   case nir_intrinsic_load_per_view_output:
      is_divergent = instr->src[0].ssa->divergent ||
                     instr->src[1].ssa->divergent ||
                     (stage == MESA_SHADER_TESS_CTRL &&
                      !(options & nir_divergence_single_patch_per_tcs_subgroup));
      break;
   case nir_intrinsic_load_per_vertex_output:
      /* TCS and NV_mesh_shader only (EXT_mesh_shader does not allow loading outputs). */
      assert(stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_MESH);
      is_divergent = src_divergent(instr->src[0], state) ||
                     src_divergent(instr->src[1], state) ||
                     (stage == MESA_SHADER_TESS_CTRL &&
                      !(options & nir_divergence_single_patch_per_tcs_subgroup));
      break;
   case nir_intrinsic_load_per_primitive_output:
      /* NV_mesh_shader only (EXT_mesh_shader does not allow loading outputs). */
      assert(stage == MESA_SHADER_MESH);
      is_divergent = src_divergent(instr->src[0], state) ||
                     src_divergent(instr->src[1], state);
      break;
   case nir_intrinsic_load_layer_id:
   case nir_intrinsic_load_front_face:
   case nir_intrinsic_load_front_face_fsign:
   case nir_intrinsic_load_back_face_agx:
      assert(stage == MESA_SHADER_FRAGMENT || state->shader->info.internal);
      is_divergent = !(options & nir_divergence_single_prim_per_subgroup);
      break;
   case nir_intrinsic_load_view_index:
      assert(stage != MESA_SHADER_COMPUTE && stage != MESA_SHADER_KERNEL);
      if (options & nir_divergence_view_index_uniform)
         is_divergent = false;
      else if (stage == MESA_SHADER_FRAGMENT)
         is_divergent = !(options & nir_divergence_single_prim_per_subgroup);
      else
         is_divergent = true;
      break;
   case nir_intrinsic_load_fs_input_interp_deltas:
      assert(stage == MESA_SHADER_FRAGMENT);
      is_divergent = src_divergent(instr->src[0], state);
      is_divergent |= !(options & nir_divergence_single_prim_per_subgroup);
      break;
   case nir_intrinsic_load_instance_id:
      is_divergent = !(state->options & nir_divergence_vertex);
      break;
   case nir_intrinsic_load_primitive_id:
      if (stage == MESA_SHADER_FRAGMENT)
         is_divergent = !(options & nir_divergence_single_prim_per_subgroup);
      else if (stage == MESA_SHADER_TESS_CTRL)
         is_divergent = !(state->options & nir_divergence_vertex) &&
                        !(options & nir_divergence_single_patch_per_tcs_subgroup);
      else if (stage == MESA_SHADER_TESS_EVAL)
         is_divergent = !(state->options & nir_divergence_vertex) &&
                        !(options & nir_divergence_single_patch_per_tes_subgroup);
      else if (stage == MESA_SHADER_GEOMETRY || stage == MESA_SHADER_VERTEX)
         is_divergent = !(state->options & nir_divergence_vertex);
      else if (stage == MESA_SHADER_ANY_HIT ||
               stage == MESA_SHADER_CLOSEST_HIT ||
               stage == MESA_SHADER_INTERSECTION)
         is_divergent = true;
      else
         UNREACHABLE("Invalid stage for load_primitive_id");
      break;
   case nir_intrinsic_load_tess_level_inner:
   case nir_intrinsic_load_tess_level_outer:
      if (stage == MESA_SHADER_TESS_CTRL)
         is_divergent = !(options & nir_divergence_single_patch_per_tcs_subgroup);
      else if (stage == MESA_SHADER_TESS_EVAL)
         is_divergent = !(options & nir_divergence_single_patch_per_tes_subgroup);
      else
         UNREACHABLE("Invalid stage for load_primitive_tess_level_*");
      break;

   case nir_intrinsic_load_workgroup_index:
   case nir_intrinsic_load_workgroup_id:
      assert(gl_shader_stage_uses_workgroup(stage) || stage == MESA_SHADER_TESS_CTRL);
      if (stage == MESA_SHADER_COMPUTE)
         is_divergent |= (options & nir_divergence_multiple_workgroup_per_compute_subgroup);
      break;

   /* Clustered reductions are uniform if cluster_size == subgroup_size or
    * the source is uniform and the operation is invariant.
    * Inclusive scans are uniform if
    * the source is uniform and the operation is invariant
    */
   case nir_intrinsic_reduce:
      if (nir_intrinsic_cluster_size(instr) == 0) {
         /* Cluster size of 0 means the subgroup size.
          * This is uniform within a subgroup, but divergent between
          * vertices of the same primitive because they may be in
          * different subgroups.
          */
         is_divergent = state->options & nir_divergence_vertex;
         break;
      }
      FALLTHROUGH;
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_inclusive_scan_clusters_ir3: {
      nir_op op = nir_intrinsic_reduction_op(instr);
      is_divergent = src_divergent(instr->src[0], state) ||
                     state->options & nir_divergence_vertex;
      if (op != nir_op_umin && op != nir_op_imin && op != nir_op_fmin &&
          op != nir_op_umax && op != nir_op_imax && op != nir_op_fmax &&
          op != nir_op_iand && op != nir_op_ior)
         is_divergent = true;
      break;
   }

   case nir_intrinsic_reduce_clusters_ir3:
      /* This reduces the last invocations in all 8-wide clusters. It should
       * behave the same as reduce with cluster_size == subgroup_size.
       */
      is_divergent = state->options & nir_divergence_vertex;
      break;

   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ubo_vec4:
   case nir_intrinsic_ldc_nv:
   case nir_intrinsic_ldcx_nv:
      is_divergent = (src_divergent(instr->src[0], state) &&
                      (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     src_divergent(instr->src[1], state);
      break;

   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ssbo_ir3:
   case nir_intrinsic_load_uav_ir3:
   case nir_intrinsic_load_ssbo_intel:
      is_divergent = (src_divergent(instr->src[0], state) &&
                      (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     src_divergent(instr->src[1], state) ||
                     load_may_tear(state, instr);
      break;

   case nir_intrinsic_load_shared:
   case nir_intrinsic_load_shared_ir3:
      is_divergent = src_divergent(instr->src[0], state) ||
                     (options & nir_divergence_uniform_load_tears);
      break;

   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_2x32:
   case nir_intrinsic_load_global_ir3:
   case nir_intrinsic_load_deref: {
      if (load_may_tear(state, instr)) {
         is_divergent = true;
         break;
      }

      unsigned num_srcs = nir_intrinsic_infos[instr->intrinsic].num_srcs;
      for (unsigned i = 0; i < num_srcs; i++) {
         if (src_divergent(instr->src[i], state)) {
            is_divergent = true;
            break;
         }
      }
      break;
   }

   case nir_intrinsic_get_ssbo_size:
   case nir_intrinsic_deref_buffer_array_length:
      is_divergent = src_divergent(instr->src[0], state) &&
                     (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM);
      break;

   case nir_intrinsic_image_samples_identical:
   case nir_intrinsic_image_deref_samples_identical:
   case nir_intrinsic_bindless_image_samples_identical:
   case nir_intrinsic_image_fragment_mask_load_amd:
   case nir_intrinsic_image_deref_fragment_mask_load_amd:
   case nir_intrinsic_bindless_image_fragment_mask_load_amd:
      is_divergent = (src_divergent(instr->src[0], state) &&
                      (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     src_divergent(instr->src[1], state) ||
                     load_may_tear(state, instr);
      break;

   case nir_intrinsic_image_texel_address:
   case nir_intrinsic_image_deref_texel_address:
   case nir_intrinsic_bindless_image_texel_address:
      is_divergent = (src_divergent(instr->src[0], state) &&
                      (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     src_divergent(instr->src[1], state) ||
                     src_divergent(instr->src[2], state);
      break;

   case nir_intrinsic_image_load:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_bindless_image_sparse_load:
      is_divergent = (src_divergent(instr->src[0], state) &&
                      (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     src_divergent(instr->src[1], state) ||
                     src_divergent(instr->src[2], state) ||
                     src_divergent(instr->src[3], state) ||
                     load_may_tear(state, instr);
      break;

   case nir_intrinsic_load_converted_output_pan:
   case nir_intrinsic_load_readonly_output_pan:
      is_divergent = ((src_divergent(instr->src[0], state) ||
                       src_divergent(instr->src[2], state)) &&
                      (nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM)) ||
                     src_divergent(instr->src[1], state);
      break;

   case nir_intrinsic_optimization_barrier_vgpr_amd:
      is_divergent = src_divergent(instr->src[0], state);
      break;

   /* Intrinsics with divergence depending on sources */
   case nir_intrinsic_convert_alu_types:
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddy_coarse:
   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_rotate:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_shuffle_xor_uniform_ir3:
   case nir_intrinsic_shuffle_up_uniform_ir3:
   case nir_intrinsic_shuffle_down_uniform_ir3:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_quad_vote_any:
   case nir_intrinsic_quad_vote_all:
   case nir_intrinsic_load_shared2_amd:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_global_amd:
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_constant:
   case nir_intrinsic_load_sample_pos_from_id:
   case nir_intrinsic_load_kernel_input:
   case nir_intrinsic_load_task_payload:
   case nir_intrinsic_load_buffer_amd:
   case nir_intrinsic_load_typed_buffer_amd:
   case nir_intrinsic_image_levels:
   case nir_intrinsic_image_deref_levels:
   case nir_intrinsic_bindless_image_levels:
   case nir_intrinsic_image_samples:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_bindless_image_samples:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_image_descriptor_amd:
   case nir_intrinsic_image_deref_descriptor_amd:
   case nir_intrinsic_bindless_image_descriptor_amd:
   case nir_intrinsic_strict_wqm_coord_amd:
   case nir_intrinsic_copy_deref:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_load_input_attachment_target_pan:
   case nir_intrinsic_load_input_attachment_conv_pan:
   case nir_intrinsic_atomic_counter_read:
   case nir_intrinsic_atomic_counter_read_deref:
   case nir_intrinsic_quad_swizzle_amd:
   case nir_intrinsic_masked_swizzle_amd:
   case nir_intrinsic_is_sparse_texels_resident:
   case nir_intrinsic_is_sparse_resident_zink:
   case nir_intrinsic_sparse_residency_code_and:
   case nir_intrinsic_bvh64_intersect_ray_amd:
   case nir_intrinsic_bvh8_intersect_ray_amd:
   case nir_intrinsic_image_deref_load_param_intel:
   case nir_intrinsic_image_load_raw_intel:
   case nir_intrinsic_get_ubo_size:
   case nir_intrinsic_load_ssbo_address:
   case nir_intrinsic_load_global_bounded:
   case nir_intrinsic_load_global_constant_bounded:
   case nir_intrinsic_load_global_constant_offset:
   case nir_intrinsic_load_reg:
   case nir_intrinsic_load_constant_agx:
   case nir_intrinsic_load_texture_handle_agx:
   case nir_intrinsic_load_from_texture_handle_agx:
   case nir_intrinsic_load_vbo_base_agx:
   case nir_intrinsic_load_attrib_clamp_agx:
   case nir_intrinsic_bindless_image_agx:
   case nir_intrinsic_bindless_sampler_agx:
   case nir_intrinsic_load_reg_indirect:
   case nir_intrinsic_load_const_ir3:
   case nir_intrinsic_load_frag_size_ir3:
   case nir_intrinsic_load_frag_offset_ir3:
   case nir_intrinsic_bindless_resource_ir3:
   case nir_intrinsic_ray_intersection_ir3:
   case nir_intrinsic_read_attribute_payload_intel: {
      unsigned num_srcs = nir_intrinsic_infos[instr->intrinsic].num_srcs;
      for (unsigned i = 0; i < num_srcs; i++) {
         if (src_divergent(instr->src[i], state)) {
            is_divergent = true;
            break;
         }
      }
      break;
   }

   case nir_intrinsic_resource_intel:
      /* Not having the non_uniform flag with divergent sources is undefined
       * behavior. The Intel driver defines it pick the lowest numbered live
       * SIMD lane (via emit_uniformize).
       */
      if ((nir_intrinsic_resource_access_intel(instr) &
           nir_resource_intel_non_uniform) != 0) {
         unsigned num_srcs = nir_intrinsic_infos[instr->intrinsic].num_srcs;
         for (unsigned i = 0; i < num_srcs; i++) {
            if (src_divergent(instr->src[i], state)) {
               is_divergent = true;
               break;
            }
         }
      }
      break;

   case nir_intrinsic_shuffle:
      is_divergent = src_divergent(instr->src[0], state) &&
                     src_divergent(instr->src[1], state);
      break;

   case nir_intrinsic_load_param:
      is_divergent =
         !state->impl->function->params[nir_intrinsic_param_idx(instr)].is_uniform;
      break;

   /* Intrinsics which are always divergent */
   case nir_intrinsic_inverse_ballot:
   case nir_intrinsic_load_color0:
   case nir_intrinsic_load_color1:
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_sample_mask_in:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_point_coord_maybe_flipped:
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_barycentric_model:
   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_barycentric_at_offset:
   case nir_intrinsic_load_barycentric_at_offset_nv:
   case nir_intrinsic_load_barycentric_coord_pixel:
   case nir_intrinsic_load_barycentric_coord_centroid:
   case nir_intrinsic_load_barycentric_coord_sample:
   case nir_intrinsic_load_barycentric_coord_at_sample:
   case nir_intrinsic_load_barycentric_coord_at_offset:
   case nir_intrinsic_load_persp_center_rhw_ir3:
   case nir_intrinsic_load_input_attachment_coord:
   case nir_intrinsic_interp_deref_at_offset:
   case nir_intrinsic_interp_deref_at_sample:
   case nir_intrinsic_interp_deref_at_centroid:
   case nir_intrinsic_interp_deref_at_vertex:
   case nir_intrinsic_load_tess_coord:
   case nir_intrinsic_load_tess_coord_xy:
   case nir_intrinsic_load_point_coord:
   case nir_intrinsic_load_line_coord:
   case nir_intrinsic_load_frag_coord:
   case nir_intrinsic_load_frag_coord_z:
   case nir_intrinsic_load_frag_coord_w:
   case nir_intrinsic_load_frag_coord_zw_pan:
   case nir_intrinsic_load_frag_coord_unscaled_ir3:
   case nir_intrinsic_load_pixel_coord:
   case nir_intrinsic_load_fully_covered:
   case nir_intrinsic_load_sample_pos:
   case nir_intrinsic_load_sample_pos_or_center:
   case nir_intrinsic_load_vertex_id_zero_base:
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_invocation_id:
   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_local_invocation_index:
   case nir_intrinsic_load_global_invocation_id:
   case nir_intrinsic_load_global_invocation_index:
   case nir_intrinsic_load_subgroup_invocation:
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_is_helper_invocation:
   case nir_intrinsic_load_scratch:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_ssbo_atomic_ir3:
   case nir_intrinsic_ssbo_atomic_swap_ir3:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   case nir_intrinsic_task_payload_atomic:
   case nir_intrinsic_task_payload_atomic_swap:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_alpha_to_coverage:
   case nir_intrinsic_global_atomic_amd:
   case nir_intrinsic_global_atomic_agx:
   case nir_intrinsic_global_atomic_swap_amd:
   case nir_intrinsic_global_atomic_swap_agx:
   case nir_intrinsic_global_atomic_2x32:
   case nir_intrinsic_global_atomic_swap_2x32:
   case nir_intrinsic_atomic_counter_add:
   case nir_intrinsic_atomic_counter_min:
   case nir_intrinsic_atomic_counter_max:
   case nir_intrinsic_atomic_counter_and:
   case nir_intrinsic_atomic_counter_or:
   case nir_intrinsic_atomic_counter_xor:
   case nir_intrinsic_atomic_counter_inc:
   case nir_intrinsic_atomic_counter_pre_dec:
   case nir_intrinsic_atomic_counter_post_dec:
   case nir_intrinsic_atomic_counter_exchange:
   case nir_intrinsic_atomic_counter_comp_swap:
   case nir_intrinsic_atomic_counter_add_deref:
   case nir_intrinsic_atomic_counter_min_deref:
   case nir_intrinsic_atomic_counter_max_deref:
   case nir_intrinsic_atomic_counter_and_deref:
   case nir_intrinsic_atomic_counter_or_deref:
   case nir_intrinsic_atomic_counter_xor_deref:
   case nir_intrinsic_atomic_counter_inc_deref:
   case nir_intrinsic_atomic_counter_pre_dec_deref:
   case nir_intrinsic_atomic_counter_post_dec_deref:
   case nir_intrinsic_atomic_counter_exchange_deref:
   case nir_intrinsic_atomic_counter_comp_swap_deref:
   case nir_intrinsic_exclusive_scan:
   case nir_intrinsic_exclusive_scan_clusters_ir3:
   case nir_intrinsic_ballot_bit_count_exclusive:
   case nir_intrinsic_ballot_bit_count_inclusive:
   case nir_intrinsic_write_invocation_amd:
   case nir_intrinsic_mbcnt_amd:
   case nir_intrinsic_lane_permute_16_amd:
   case nir_intrinsic_dpp16_shift_amd:
   case nir_intrinsic_elect:
   case nir_intrinsic_elect_any_ir3:
   case nir_intrinsic_load_tlb_color_brcm:
   case nir_intrinsic_load_tess_rel_patch_id_amd:
   case nir_intrinsic_load_gs_vertex_offset_amd:
   case nir_intrinsic_is_subgroup_invocation_lt_amd:
   case nir_intrinsic_load_packed_passthrough_primitive_amd:
   case nir_intrinsic_load_initial_edgeflags_amd:
   case nir_intrinsic_gds_atomic_add_amd:
   case nir_intrinsic_load_rt_arg_scratch_offset_amd:
   case nir_intrinsic_load_intersection_opaque_amd:
   case nir_intrinsic_load_vector_arg_amd:
   case nir_intrinsic_load_btd_stack_id_intel:
   case nir_intrinsic_load_topology_id_intel:
   case nir_intrinsic_load_scratch_base_ptr:
   case nir_intrinsic_ordered_xfb_counter_add_gfx11_amd:
   case nir_intrinsic_ordered_add_loop_gfx12_amd:
   case nir_intrinsic_xfb_counter_sub_gfx11_amd:
   case nir_intrinsic_unit_test_divergent_amd:
   case nir_intrinsic_load_stack:
   case nir_intrinsic_load_ray_launch_id:
   case nir_intrinsic_load_ray_instance_custom_index:
   case nir_intrinsic_load_ray_geometry_index:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_t_max:
   case nir_intrinsic_load_ray_object_to_world:
   case nir_intrinsic_load_ray_world_to_object:
   case nir_intrinsic_load_ray_hit_kind:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_cull_mask:
   case nir_intrinsic_emit_vertex_nv:
   case nir_intrinsic_end_primitive_nv:
   case nir_intrinsic_report_ray_intersection:
   case nir_intrinsic_rq_proceed:
   case nir_intrinsic_rq_load:
   case nir_intrinsic_load_ray_triangle_vertex_positions:
   case nir_intrinsic_cmat_extract:
   case nir_intrinsic_cmat_muladd_amd:
   case nir_intrinsic_dpas_intel:
   case nir_intrinsic_convert_cmat_intel:
   case nir_intrinsic_isberd_nv:
   case nir_intrinsic_vild_nv:
   case nir_intrinsic_al2p_nv:
   case nir_intrinsic_ald_nv:
   case nir_intrinsic_suclamp_nv:
   case nir_intrinsic_subfm_nv:
   case nir_intrinsic_sueau_nv:
   case nir_intrinsic_imadsp_nv:
   case nir_intrinsic_suldga_nv:
   case nir_intrinsic_sustga_nv:
   case nir_intrinsic_ipa_nv:
   case nir_intrinsic_ldtram_nv:
   case nir_intrinsic_cmat_muladd_nv:
   case nir_intrinsic_printf:
   case nir_intrinsic_load_gs_header_ir3:
   case nir_intrinsic_load_tcs_header_ir3:
   case nir_intrinsic_load_rel_patch_id_ir3:
   case nir_intrinsic_brcst_active_ir3:
   case nir_intrinsic_load_helper_op_id_agx:
   case nir_intrinsic_load_helper_arg_lo_agx:
   case nir_intrinsic_load_helper_arg_hi_agx:
   case nir_intrinsic_stack_map_agx:
   case nir_intrinsic_stack_unmap_agx:
   case nir_intrinsic_load_exported_agx:
   case nir_intrinsic_load_local_pixel_agx:
   case nir_intrinsic_load_coefficients_agx:
   case nir_intrinsic_load_active_subgroup_invocation_agx:
   case nir_intrinsic_load_sample_mask:
   case nir_intrinsic_quad_ballot_agx:
   case nir_intrinsic_load_agx:
   case nir_intrinsic_load_shared_lock_nv:
   case nir_intrinsic_store_shared_unlock_nv:
   case nir_intrinsic_bvh_stack_rtn_amd:
      is_divergent = true;
      break;

   default:
#ifdef NDEBUG
      is_divergent = true;
      break;
#else
      nir_print_instr(&instr->instr, stderr);
      UNREACHABLE("\nNIR divergence analysis: Unhandled intrinsic.");
#endif
   }

   instr->def.divergent = is_divergent;
   return is_divergent;
}

static bool
visit_tex(nir_tex_instr *instr, struct divergence_state *state)
{
   if (instr->def.divergent)
      return false;

   bool is_divergent = false;

   for (unsigned i = 0; i < instr->num_srcs; i++) {
      switch (instr->src[i].src_type) {
      case nir_tex_src_sampler_deref:
      case nir_tex_src_sampler_handle:
      case nir_tex_src_sampler_offset:
         is_divergent |= src_divergent(instr->src[i].src, state) &&
                         instr->sampler_non_uniform;
         break;
      case nir_tex_src_texture_deref:
      case nir_tex_src_texture_handle:
      case nir_tex_src_texture_offset:
         is_divergent |= src_divergent(instr->src[i].src, state) &&
                         instr->texture_non_uniform;
         break;
      case nir_tex_src_offset:
         instr->offset_non_uniform = src_divergent(instr->src[i].src, state);
         is_divergent |= instr->offset_non_uniform;
         break;
      default:
         is_divergent |= src_divergent(instr->src[i].src, state);
         break;
      }
   }

   /* If the texture instruction skips helpers, that may add divergence even
    * if none of the sources of the texture op diverge.
    */
   if (instr->skip_helpers)
      is_divergent = true;

   instr->def.divergent = is_divergent;
   return is_divergent;
}

static bool
visit_def(nir_def *def, struct divergence_state *state)
{
   return false;
}

static bool
nir_variable_mode_is_uniform(nir_variable_mode mode)
{
   switch (mode) {
   case nir_var_uniform:
   case nir_var_mem_ubo:
   case nir_var_mem_ssbo:
   case nir_var_mem_shared:
   case nir_var_mem_task_payload:
   case nir_var_mem_global:
   case nir_var_image:
      return true;
   default:
      return false;
   }
}

static bool
nir_variable_is_uniform(nir_shader *shader, nir_variable *var,
                        struct divergence_state *state)
{
   if (nir_variable_mode_is_uniform(var->data.mode))
      return true;

   /* Handle system value variables. */
   if (var->data.mode == nir_var_system_value) {
      /* Fake the instruction to reuse visit_intrinsic for all sysvals. */
      nir_intrinsic_instr fake_instr;

      memset(&fake_instr, 0, sizeof(fake_instr));
      fake_instr.intrinsic =
         nir_intrinsic_from_system_value(var->data.location);

      visit_intrinsic(&fake_instr, state);
      return !fake_instr.def.divergent;
   }

   nir_divergence_options options = state->options;
   gl_shader_stage stage = shader->info.stage;

   if (stage == MESA_SHADER_FRAGMENT &&
       (options & nir_divergence_single_prim_per_subgroup) &&
       var->data.mode == nir_var_shader_in &&
       var->data.interpolation == INTERP_MODE_FLAT)
      return true;

   if (stage == MESA_SHADER_TESS_CTRL &&
       (options & nir_divergence_single_patch_per_tcs_subgroup) &&
       var->data.mode == nir_var_shader_out && var->data.patch)
      return true;

   if (stage == MESA_SHADER_TESS_EVAL &&
       (options & nir_divergence_single_patch_per_tes_subgroup) &&
       var->data.mode == nir_var_shader_in && var->data.patch)
      return true;

   return false;
}

static bool
visit_deref(nir_shader *shader, nir_deref_instr *deref,
            struct divergence_state *state)
{
   if (deref->def.divergent)
      return false;

   bool is_divergent = false;
   switch (deref->deref_type) {
   case nir_deref_type_var:
      is_divergent = !nir_variable_is_uniform(shader, deref->var, state);
      break;
   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array:
      is_divergent = src_divergent(deref->arr.index, state);
      FALLTHROUGH;
   case nir_deref_type_struct:
   case nir_deref_type_array_wildcard:
      is_divergent |= src_divergent(deref->parent, state);
      break;
   case nir_deref_type_cast:
      is_divergent = !nir_variable_mode_is_uniform(deref->var->data.mode) ||
                     src_divergent(deref->parent, state);
      break;
   }

   deref->def.divergent = is_divergent;
   return is_divergent;
}

static bool
visit_jump(nir_jump_instr *jump, struct divergence_state *state)
{
   switch (jump->type) {
   case nir_jump_continue:
      if (state->divergent_loop_continue)
         return false;
      if (state->divergent_loop_cf)
         state->divergent_loop_continue = true;
      return state->divergent_loop_continue;
   case nir_jump_break:
      if (state->divergent_loop_break)
         return false;
      if (state->divergent_loop_cf)
         state->divergent_loop_break = true;
      return state->divergent_loop_break;
   case nir_jump_halt:
      /* This totally kills invocations so it doesn't add divergence */
      break;
   case nir_jump_return:
      UNREACHABLE("NIR divergence analysis: Unsupported return instruction.");
      break;
   case nir_jump_goto:
   case nir_jump_goto_if:
      UNREACHABLE("NIR divergence analysis: Unsupported goto_if instruction.");
      break;
   }
   return false;
}

static bool
set_ssa_def_not_divergent(nir_def *def, void *invariant)
{
   def->divergent = false;
   def->loop_invariant = *(bool *)invariant;
   return true;
}

static bool
instr_is_loop_invariant(nir_instr *instr, struct divergence_state *state)
{
   if (!state->loop)
      return false;

   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
   case nir_instr_type_jump:
      return true;
   case nir_instr_type_intrinsic:
      if (!nir_intrinsic_can_reorder(nir_instr_as_intrinsic(instr)))
         return false;
      FALLTHROUGH;
   case nir_instr_type_alu:
   case nir_instr_type_deref:
   case nir_instr_type_tex:
      return nir_foreach_src(instr, src_invariant, state->loop);
   case nir_instr_type_call:
      return false;
   case nir_instr_type_phi:
   case nir_instr_type_parallel_copy:
   default:
      UNREACHABLE("NIR divergence analysis: Unsupported instruction type.");
   }
}

static bool
update_instr_divergence(nir_instr *instr, struct divergence_state *state)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return visit_alu(nir_instr_as_alu(instr), state);
   case nir_instr_type_intrinsic:
      return visit_intrinsic(nir_instr_as_intrinsic(instr), state);
   case nir_instr_type_tex:
      return visit_tex(nir_instr_as_tex(instr), state);
   case nir_instr_type_load_const:
      return visit_def(&nir_instr_as_load_const(instr)->def, state);
   case nir_instr_type_undef:
      return visit_def(&nir_instr_as_undef(instr)->def, state);
   case nir_instr_type_deref:
      return visit_deref(state->shader, nir_instr_as_deref(instr), state);
   case nir_instr_type_call:
      return false;
   case nir_instr_type_jump:
   case nir_instr_type_phi:
   case nir_instr_type_parallel_copy:
   default:
      UNREACHABLE("NIR divergence analysis: Unsupported instruction type.");
   }
}

static bool
visit_block(nir_block *block, struct divergence_state *state)
{
   bool has_changed = false;

   nir_foreach_instr(instr, block) {
      /* phis are handled when processing the branches */
      if (instr->type == nir_instr_type_phi)
         continue;

      if (state->first_visit) {
         bool invariant = state->loop_all_invariant || instr_is_loop_invariant(instr, state);
         nir_foreach_def(instr, set_ssa_def_not_divergent, &invariant);
      }

      if (instr->type == nir_instr_type_jump) {
         has_changed |= visit_jump(nir_instr_as_jump(instr), state);
      } else {
         has_changed |= update_instr_divergence(instr, state);
      }
   }

   bool divergent = state->divergent_loop_cf ||
                    state->divergent_loop_continue ||
                    state->divergent_loop_break;
   if (divergent != block->divergent) {
      block->divergent = divergent;
      has_changed = true;
   }

   return has_changed;
}

/* There are 3 types of phi instructions:
 * (1) gamma: represent the joining point of different paths
 *     created by an “if-then-else” branch.
 *     The resulting value is divergent if the branch condition
 *     or any of the source values is divergent. */
static bool
visit_if_merge_phi(nir_phi_instr *phi, bool if_cond_divergent, bool ignore_undef)
{
   if (phi->def.divergent)
      return false;

   unsigned defined_srcs = 0;
   nir_foreach_phi_src(src, phi) {
      /* if any source value is divergent, the resulting value is divergent */
      if (nir_src_is_divergent(&src->src)) {
         phi->def.divergent = true;
         return true;
      }
      if (src->src.ssa->parent_instr->type != nir_instr_type_undef) {
         defined_srcs++;
      }
   }

   if (!(ignore_undef && defined_srcs <= 1) && if_cond_divergent) {
      phi->def.divergent = true;
      return true;
   }

   return false;
}

/* There are 3 types of phi instructions:
 * (2) mu: which only exist at loop headers,
 *     merge initial and loop-carried values.
 *     The resulting value is divergent if any source value
 *     is divergent or a divergent loop continue condition
 *     is associated with a different ssa-def. */
static bool
visit_loop_header_phi(nir_phi_instr *phi, nir_block *preheader, bool divergent_continue)
{
   if (phi->def.divergent)
      return false;

   nir_def *same = NULL;
   nir_foreach_phi_src(src, phi) {
      /* if any source value is divergent, the resulting value is divergent */
      if (nir_src_is_divergent(&src->src)) {
         phi->def.divergent = true;
         return true;
      }
      /* if this loop is uniform, we're done here */
      if (!divergent_continue)
         continue;
      /* skip the loop preheader */
      if (src->pred == preheader)
         continue;

      /* check if all loop-carried values are from the same ssa-def */
      if (!same)
         same = src->src.ssa;
      else if (same != src->src.ssa) {
         phi->def.divergent = true;
         return true;
      }
   }

   return false;
}

/* There are 3 types of phi instructions:
 * (3) eta: represent values that leave a loop.
 *     The resulting value is divergent if the source value is divergent
 *     or any loop exit condition is divergent for a value which is
 *     not loop-invariant (see nir_src_is_divergent()).
 */
static bool
visit_loop_exit_phi(nir_phi_instr *phi, nir_loop *loop)
{
   if (phi->def.divergent)
      return false;

   nir_def *same = NULL;
   nir_foreach_phi_src(src, phi) {
      /* If any loop exit condition is divergent and this value is not loop
       * invariant, or if the source value is divergent, then the resulting
       * value is divergent.
       */
      if ((loop->divergent_break && !src_invariant(&src->src, loop)) ||
          nir_src_is_divergent(&src->src)) {
         phi->def.divergent = true;
         return true;
      }

      /* if this loop is uniform, we're done here */
      if (!loop->divergent_break)
         continue;

      /* check if all loop-exit values are from the same ssa-def */
      if (!same)
         same = src->src.ssa;
      else if (same != src->src.ssa) {
         phi->def.divergent = true;
         return true;
      }
   }

   return false;
}

static bool
visit_if(nir_if *if_stmt, struct divergence_state *state)
{
   bool progress = false;
   bool cond_divergent = src_divergent(if_stmt->condition, state);

   struct divergence_state then_state = *state;
   then_state.divergent_loop_cf |= cond_divergent;
   progress |= visit_cf_list(&if_stmt->then_list, &then_state);

   struct divergence_state else_state = *state;
   else_state.divergent_loop_cf |= cond_divergent;
   progress |= visit_cf_list(&if_stmt->else_list, &else_state);

   /* handle phis after the IF */
   bool invariant = state->loop && src_invariant(&if_stmt->condition, state->loop);
   nir_foreach_phi(phi, nir_cf_node_cf_tree_next(&if_stmt->cf_node)) {
      if (state->first_visit) {
         phi->def.divergent = false;
         phi->def.loop_invariant =
            invariant && nir_foreach_src(&phi->instr, src_invariant, state->loop);
      }

      /* The only user of this option (ACO) only supports it for non-boolean phis. */
      bool ignore_undef =
         (state->options & nir_divergence_ignore_undef_if_phi_srcs) && phi->def.bit_size != 1;

      progress |= visit_if_merge_phi(phi, cond_divergent, ignore_undef);
   }

   /* join loop divergence information from both branch legs */
   state->divergent_loop_continue |= then_state.divergent_loop_continue ||
                                     else_state.divergent_loop_continue;
   state->divergent_loop_break |= then_state.divergent_loop_break ||
                                  else_state.divergent_loop_break;

   /* A divergent continue makes succeeding loop CF divergent:
    * not all loop-active invocations participate in the remaining loop-body
    * which means that a following break might be taken by some invocations, only */
   state->divergent_loop_cf |= state->divergent_loop_continue;

   state->consider_loop_invariance |= then_state.consider_loop_invariance ||
                                      else_state.consider_loop_invariance;

   return progress;
}

static bool
visit_loop(nir_loop *loop, struct divergence_state *state)
{
   assert(!nir_loop_has_continue_construct(loop));
   bool progress = false;
   nir_block *loop_header = nir_loop_first_block(loop);
   nir_block *loop_preheader = nir_block_cf_tree_prev(loop_header);

   /* handle loop header phis first: we have no knowledge yet about
    * the loop's control flow or any loop-carried sources. */
   nir_foreach_phi(phi, loop_header) {
      if (!state->first_visit && phi->def.divergent)
         continue;

      phi->def.loop_invariant = false;
      nir_foreach_phi_src(src, phi) {
         if (src->pred == loop_preheader) {
            phi->def.divergent = nir_src_is_divergent(&src->src);
            break;
         }
      }
      progress |= phi->def.divergent;
   }

   /* setup loop state */
   struct divergence_state loop_state = *state;
   loop_state.loop = loop;
   loop_state.loop_all_invariant = loop_header->predecessors->entries == 1;
   loop_state.divergent_loop_cf = false;
   loop_state.divergent_loop_continue = false;
   loop_state.divergent_loop_break = false;

   /* process loop body until no further changes are made */
   bool repeat;
   do {
      progress |= visit_cf_list(&loop->body, &loop_state);
      repeat = false;

      /* revisit loop header phis to see if something has changed */
      nir_foreach_phi(phi, loop_header) {
         repeat |= visit_loop_header_phi(phi, loop_preheader,
                                         loop_state.divergent_loop_continue);
      }

      loop_state.divergent_loop_cf = false;
      loop_state.first_visit = false;
   } while (repeat);

   loop->divergent_continue = loop_state.divergent_loop_continue;
   loop->divergent_break = loop_state.divergent_loop_break;

   /* handle phis after the loop */
   nir_foreach_phi(phi, nir_cf_node_cf_tree_next(&loop->cf_node)) {
      if (state->first_visit) {
         phi->def.divergent = false;
         phi->def.loop_invariant = false;
      }
      progress |= visit_loop_exit_phi(phi, loop);
   }

   state->consider_loop_invariance |= loop_state.consider_loop_invariance ||
                                      loop->divergent_break;
   return progress;
}

static bool
visit_cf_list(struct exec_list *list, struct divergence_state *state)
{
   bool has_changed = false;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         has_changed |= visit_block(nir_cf_node_as_block(node), state);
         break;
      case nir_cf_node_if:
         has_changed |= visit_if(nir_cf_node_as_if(node), state);
         break;
      case nir_cf_node_loop:
         has_changed |= visit_loop(nir_cf_node_as_loop(node), state);
         break;
      case nir_cf_node_function:
         UNREACHABLE("NIR divergence analysis: Unsupported cf_node type.");
      }
   }

   return has_changed;
}

void
nir_divergence_analysis_impl(nir_function_impl *impl, nir_divergence_options options)
{
   nir_metadata_require(impl, nir_metadata_block_index);

   struct divergence_state state = {
      .stage = impl->function->shader->info.stage,
      .shader = impl->function->shader,
      .impl = impl,
      .options = options,
      .loop = NULL,
      .loop_all_invariant = false,
      .divergent_loop_cf = false,
      .divergent_loop_continue = false,
      .divergent_loop_break = false,
      .first_visit = true,
   };

   visit_cf_list(&impl->body, &state);

   /* Unless this pass is called with shader->options->divergence_analysis_options,
    * it invalidates nir_metadata_divergence.
    */
   nir_progress(true, impl, ~nir_metadata_divergence);
}

void
nir_divergence_analysis(nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_divergence);
   }
}

/* Compute divergence between vertices of the same primitive. This uses
 * the same divergent field in nir_def and nir_loop as the regular divergence
 * pass.
 */
void
nir_vertex_divergence_analysis(nir_shader *shader)
{
   nir_divergence_options options =
      shader->options->divergence_analysis_options | nir_divergence_vertex;

   nir_foreach_function_impl(impl, shader) {
      nir_divergence_analysis_impl(impl, options);
   }
}

bool
nir_has_divergent_loop(nir_shader *shader)
{
   nir_function_impl *func = nir_shader_get_entrypoint(shader);

   foreach_list_typed(nir_cf_node, node, node, &func->body) {
      if (node->type == nir_cf_node_loop) {
         if (nir_cf_node_as_loop(node)->divergent_break)
            return true;
      }
   }

   return false;
}
