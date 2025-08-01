/*
 * Copyright © 2018 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "aco_instruction_selection.h"
#include "aco_interface.h"

#include "nir_builder.h"
#include "nir_control_flow.h"

#include "ac_nir.h"
#include <vector>

namespace aco {

namespace {

/* Check whether the given SSA def is only used by cross-lane instructions. */
bool
only_used_by_cross_lane_instrs(nir_def* ssa, bool follow_phis = true)
{
   nir_foreach_use (src, ssa) {
      switch (nir_src_parent_instr(src)->type) {
      case nir_instr_type_alu: {
         nir_alu_instr* alu = nir_instr_as_alu(nir_src_parent_instr(src));
         if (alu->op != nir_op_unpack_64_2x32_split_x && alu->op != nir_op_unpack_64_2x32_split_y)
            return false;
         if (!only_used_by_cross_lane_instrs(&alu->def, follow_phis))
            return false;

         continue;
      }
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr* intrin = nir_instr_as_intrinsic(nir_src_parent_instr(src));
         if (intrin->intrinsic != nir_intrinsic_read_invocation &&
             intrin->intrinsic != nir_intrinsic_read_first_invocation &&
             intrin->intrinsic != nir_intrinsic_lane_permute_16_amd)
            return false;

         continue;
      }
      case nir_instr_type_phi: {
         /* Don't follow more than 1 phis, this avoids infinite loops. */
         if (!follow_phis)
            return false;

         nir_phi_instr* phi = nir_instr_as_phi(nir_src_parent_instr(src));
         if (!only_used_by_cross_lane_instrs(&phi->def, false))
            return false;

         continue;
      }
      default: return false;
      }
   }

   return true;
}

/* If one side of a divergent IF ends in a branch and the other doesn't, we
 * might have to emit the contents of the side without the branch at the merge
 * block instead. This is so that we can use any SGPR live-out of the side
 * without the branch without creating a linear phi in the invert or merge block.
 *
 * This also removes any unreachable merge blocks.
 */
bool
sanitize_if(nir_function_impl* impl, nir_if* nif)
{
   nir_block* then_block = nir_if_last_then_block(nif);
   nir_block* else_block = nir_if_last_else_block(nif);
   bool then_jump = nir_block_ends_in_jump(then_block);
   bool else_jump = nir_block_ends_in_jump(else_block);
   if (!then_jump && !else_jump)
      return false;

   /* If the continue from block is empty then return as there is nothing to
    * move.
    */
   if (nir_cf_list_is_empty_block(then_jump ? &nif->else_list : &nif->then_list))
      return false;

   /* Even though this if statement has a jump on one side, we may still have
    * phis afterwards.  Single-source phis can be produced by loop unrolling
    * or dead control-flow passes and are perfectly legal.  Run a quick phi
    * removal on the block after the if to clean up any such phis.
    */
   nir_remove_single_src_phis_block(nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node)));

   /* Finally, move the continue from branch after the if-statement. */
   nir_block* last_continue_from_blk = then_jump ? else_block : then_block;
   nir_block* first_continue_from_blk =
      then_jump ? nir_if_first_else_block(nif) : nir_if_first_then_block(nif);

   /* We don't need to repair SSA. nir_remove_after_cf_node() replaces any uses with undef. */
   if (then_jump && else_jump)
      nir_remove_after_cf_node(&nif->cf_node);

   nir_cf_list tmp;
   nir_cf_extract(&tmp, nir_before_block(first_continue_from_blk),
                  nir_after_block(last_continue_from_blk));
   nir_cf_reinsert(&tmp, nir_after_cf_node(&nif->cf_node));

   return true;
}

bool
sanitize_cf_list(nir_function_impl* impl, struct exec_list* cf_list)
{
   bool progress = false;
   foreach_list_typed (nir_cf_node, cf_node, node, cf_list) {
      switch (cf_node->type) {
      case nir_cf_node_block: break;
      case nir_cf_node_if: {
         nir_if* nif = nir_cf_node_as_if(cf_node);
         progress |= sanitize_cf_list(impl, &nif->then_list);
         progress |= sanitize_cf_list(impl, &nif->else_list);
         progress |= sanitize_if(impl, nif);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop* loop = nir_cf_node_as_loop(cf_node);
         assert(!nir_loop_has_continue_construct(loop));
         progress |= sanitize_cf_list(impl, &loop->body);

         /* NIR seems to allow this, and even though the loop exit has no predecessors, SSA defs
          * from the loop header are live. Handle this without complicating the ACO IR by creating a
          * dummy break.
          */
         if (nir_cf_node_cf_tree_next(&loop->cf_node)->predecessors->entries == 0) {
            nir_builder b = nir_builder_create(impl);
            b.cursor = nir_after_block_before_jump(nir_loop_last_block(loop));

            nir_def* cond = nir_imm_false(&b);
            /* We don't use block divergence information, so just this is enough. */
            cond->divergent = false;

            nir_break_if(&b, cond);

            progress = true;
         }
         break;
      }
      case nir_cf_node_function: UNREACHABLE("Invalid cf type");
      }
   }

   return progress;
}

void
apply_nuw_to_ssa(isel_context* ctx, nir_def* ssa)
{
   nir_scalar scalar;
   scalar.def = ssa;
   scalar.comp = 0;

   if (!nir_scalar_is_alu(scalar) || nir_scalar_alu_op(scalar) != nir_op_iadd)
      return;

   nir_alu_instr* add = nir_instr_as_alu(ssa->parent_instr);

   if (add->no_unsigned_wrap)
      return;

   nir_scalar src0 = nir_scalar_chase_alu_src(scalar, 0);
   nir_scalar src1 = nir_scalar_chase_alu_src(scalar, 1);

   if (nir_scalar_is_const(src0)) {
      std::swap(src0, src1);
   }

   uint32_t src1_ub = nir_unsigned_upper_bound(ctx->shader, ctx->range_ht, src1, &ctx->ub_config);
   add->no_unsigned_wrap =
      !nir_addition_might_overflow(ctx->shader, ctx->range_ht, src0, src1_ub, &ctx->ub_config);
}

void
apply_nuw_to_offsets(isel_context* ctx, nir_function_impl* impl)
{
   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr* intrin = nir_instr_as_intrinsic(instr);

         switch (intrin->intrinsic) {
         case nir_intrinsic_load_constant:
         case nir_intrinsic_load_uniform:
         case nir_intrinsic_load_push_constant:
            if (!nir_src_is_divergent(&intrin->src[0]))
               apply_nuw_to_ssa(ctx, intrin->src[0].ssa);
            break;
         case nir_intrinsic_load_ubo:
         case nir_intrinsic_load_ssbo:
            if (!nir_src_is_divergent(&intrin->src[1]))
               apply_nuw_to_ssa(ctx, intrin->src[1].ssa);
            break;
         case nir_intrinsic_store_ssbo:
            if (!nir_src_is_divergent(&intrin->src[2]))
               apply_nuw_to_ssa(ctx, intrin->src[2].ssa);
            break;
         case nir_intrinsic_load_scratch: apply_nuw_to_ssa(ctx, intrin->src[0].ssa); break;
         case nir_intrinsic_store_scratch:
         case nir_intrinsic_load_smem_amd: apply_nuw_to_ssa(ctx, intrin->src[1].ssa); break;
         default: break;
         }
      }
   }
}

RegClass
get_reg_class(isel_context* ctx, RegType type, unsigned components, unsigned bitsize)
{
   if (bitsize == 1)
      return RegClass(RegType::sgpr, ctx->program->lane_mask.size() * components);
   else
      return RegClass::get(type, components * bitsize / 8u);
}

void
setup_tcs_info(isel_context* ctx)
{
   ctx->tcs_in_out_eq = ctx->program->info.vs.tcs_in_out_eq;
   ctx->any_tcs_inputs_via_lds = ctx->program->info.vs.any_tcs_inputs_via_lds;
}

void
setup_lds_size(isel_context* ctx, nir_shader* nir)
{
   /* TCS and GFX9 GS are special cases, already in units of the allocation granule. */
   if (ctx->stage.has(SWStage::TCS))
      ctx->program->config->lds_size = ctx->program->info.tcs.num_lds_blocks;
   else if (ctx->stage.hw == AC_HW_LEGACY_GEOMETRY_SHADER && ctx->options->gfx_level >= GFX9)
      ctx->program->config->lds_size = ctx->program->info.gfx9_gs_ring_lds_size;
   else
      ctx->program->config->lds_size =
         DIV_ROUND_UP(nir->info.shared_size, ctx->program->dev.lds_encoding_granule);

   /* Make sure we fit the available LDS space. */
   assert((ctx->program->config->lds_size * ctx->program->dev.lds_encoding_granule) <=
          ctx->program->dev.lds_limit);
}

void
setup_nir(isel_context* ctx, nir_shader* nir)
{
   nir_convert_to_lcssa(nir, true, false);
   if (nir_lower_phis_to_scalar(nir, ac_nir_lower_phis_to_scalar_cb, NULL)) {
      nir_copy_prop(nir);
      nir_opt_dce(nir);
   }

   nir_function_impl* func = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(func);
}

/* Returns true if we can skip uniformization of a merge phi. This makes the destination divergent,
 * and so is only safe if the inconsistency it introduces into the divergence analysis won't break
 * code generation. If we unsafely skip uniformization, later instructions (such as SSBO loads,
 * some subgroup intrinsics and certain conversions) can use divergence analysis information which
 * is no longer correct.
 */
bool
skip_uniformize_merge_phi(nir_def* ssa, unsigned depth)
{
   if (depth >= 16)
      return false;

   nir_foreach_use (src, ssa) {
      switch (nir_src_parent_instr(src)->type) {
      case nir_instr_type_alu: {
         nir_alu_instr* alu = nir_instr_as_alu(nir_src_parent_instr(src));
         if (alu->def.divergent)
            break;

         switch (alu->op) {
         case nir_op_f2i16:
         case nir_op_f2u16:
         case nir_op_f2i32:
         case nir_op_f2u32:
         case nir_op_b2i8:
         case nir_op_b2i16:
         case nir_op_b2i32:
         case nir_op_b2b32:
         case nir_op_b2f16:
         case nir_op_b2f32:
         case nir_op_b2f64:
         case nir_op_mov:
            /* These opcodes p_as_uniform or vote_any() the source, so fail immediately. We don't
             * need to do this for non-nir_op_b2 if we know we'll move it back into a VGPR,
             * in which case the p_as_uniform would be eliminated. This would be way too fragile,
             * though.
             */
            return false;
         default:
            if (!skip_uniformize_merge_phi(&alu->def, depth + 1))
               return false;
            break;
         }
         break;
      }
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr* intrin = nir_instr_as_intrinsic(nir_src_parent_instr(src));
         unsigned src_idx = src - intrin->src;
         /* nir_intrinsic_lane_permute_16_amd is only safe because we don't use divergence analysis
          * for it's instruction selection. We use that intrinsic for NGG culling. All others are
          * stores with VGPR sources.
          */
         if (intrin->intrinsic == nir_intrinsic_lane_permute_16_amd ||
             intrin->intrinsic == nir_intrinsic_export_amd ||
             intrin->intrinsic == nir_intrinsic_export_dual_src_blend_amd ||
             (intrin->intrinsic == nir_intrinsic_export_row_amd && src_idx == 0) ||
             (intrin->intrinsic == nir_intrinsic_store_buffer_amd && src_idx == 0) ||
             (intrin->intrinsic == nir_intrinsic_store_ssbo && src_idx == 0) ||
             (intrin->intrinsic == nir_intrinsic_store_global && src_idx == 0) ||
             (intrin->intrinsic == nir_intrinsic_store_scratch && src_idx == 0) ||
             (intrin->intrinsic == nir_intrinsic_store_shared && src_idx == 0))
            break;
         return false;
      }
      case nir_instr_type_phi: {
         nir_phi_instr* phi = nir_instr_as_phi(nir_src_parent_instr(src));
         if (phi->def.divergent || skip_uniformize_merge_phi(&phi->def, depth + 1))
            break;
         return false;
      }
      case nir_instr_type_tex: {
         /* This is either used as a VGPR source or it's a (potentially undef) descriptor. */
         break;
      }
      default: {
         return false;
      }
      }
   }

   return true;
}

} /* end namespace */

void
init_context(isel_context* ctx, nir_shader* shader)
{
   nir_function_impl* impl = nir_shader_get_entrypoint(shader);
   ctx->shader = shader;

   /* Init NIR range analysis. */
   ctx->range_ht = _mesa_pointer_hash_table_create(NULL);
   ctx->ub_config.min_subgroup_size = ctx->program->wave_size;
   ctx->ub_config.max_subgroup_size = ctx->program->wave_size;
   ctx->ub_config.max_workgroup_invocations = 2048;
   ctx->ub_config.max_workgroup_count[0] = 4294967295;
   ctx->ub_config.max_workgroup_count[1] = 65535;
   ctx->ub_config.max_workgroup_count[2] = 65535;
   ctx->ub_config.max_workgroup_size[0] = 1024;
   ctx->ub_config.max_workgroup_size[1] = 1024;
   ctx->ub_config.max_workgroup_size[2] = 1024;

   uint32_t options =
      shader->options->divergence_analysis_options | nir_divergence_ignore_undef_if_phi_srcs;
   nir_divergence_analysis_impl(impl, (nir_divergence_options)options);

   apply_nuw_to_offsets(ctx, impl);
   ac_nir_flag_smem_for_loads(shader, ctx->program->gfx_level, false, true);

   /* sanitize control flow */
   sanitize_cf_list(impl, &impl->body);
   nir_progress(true, impl, nir_metadata_none);

   /* we'll need these for isel */
   nir_metadata_require(impl, nir_metadata_block_index);

   /* Our definition of divergence is slightly different, but we still want nir to print it. */
   impl->valid_metadata |= nir_metadata_divergence;

   if (ctx->options->dump_preoptir) {
      fprintf(stderr, "NIR shader before instruction selection:\n");
      nir_print_shader(shader, stderr);
   }

   ctx->first_temp_id = ctx->program->peekAllocationId();
   ctx->program->allocateRange(impl->ssa_alloc);
   RegClass* regclasses = ctx->program->temp_rc.data() + ctx->first_temp_id;

   /* TODO: make this recursive to improve compile times */
   bool done = false;
   while (!done) {
      done = true;
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            switch (instr->type) {
            case nir_instr_type_alu: {
               nir_alu_instr* alu_instr = nir_instr_as_alu(instr);
               RegType type = RegType::sgpr;

               /* Packed 16-bit instructions have to be VGPR. */
               if (alu_instr->def.num_components == 2 &&
                   aco_nir_op_supports_packed_math_16bit(alu_instr))
                  type = RegType::vgpr;

               switch (alu_instr->op) {
               case nir_op_f2i16:
               case nir_op_f2u16:
               case nir_op_f2i32:
               case nir_op_f2u32:
               case nir_op_mov:
                  if (alu_instr->def.divergent &&
                      regclasses[alu_instr->src[0].src.ssa->index].type() == RegType::vgpr)
                     type = RegType::vgpr;
                  break;
               case nir_op_f2e4m3fn:
               case nir_op_f2e4m3fn_sat:
               case nir_op_f2e4m3fn_satfn:
               case nir_op_f2e5m2:
               case nir_op_f2e5m2_sat:
               case nir_op_e4m3fn2f:
               case nir_op_e5m22f:
               case nir_op_fmulz:
               case nir_op_ffmaz:
               case nir_op_f2f64:
               case nir_op_u2f64:
               case nir_op_i2f64:
               case nir_op_pack_unorm_2x16:
               case nir_op_pack_snorm_2x16:
               case nir_op_pack_uint_2x16:
               case nir_op_pack_sint_2x16:
               case nir_op_ldexp:
               case nir_op_frexp_sig:
               case nir_op_frexp_exp:
               case nir_op_cube_amd:
               case nir_op_msad_4x8:
               case nir_op_mqsad_4x8:
               case nir_op_udot_4x8_uadd:
               case nir_op_sdot_4x8_iadd:
               case nir_op_sudot_4x8_iadd:
               case nir_op_udot_4x8_uadd_sat:
               case nir_op_sdot_4x8_iadd_sat:
               case nir_op_sudot_4x8_iadd_sat:
               case nir_op_udot_2x16_uadd:
               case nir_op_sdot_2x16_iadd:
               case nir_op_udot_2x16_uadd_sat:
               case nir_op_sdot_2x16_iadd_sat:
               case nir_op_bfdot2_bfadd:
               case nir_op_byte_perm_amd:
               case nir_op_alignbyte_amd: type = RegType::vgpr; break;
               case nir_op_fmul:
               case nir_op_ffma:
               case nir_op_fadd:
               case nir_op_fsub:
               case nir_op_fmax:
               case nir_op_fmin:
               case nir_op_fsat:
               case nir_op_fneg:
               case nir_op_fabs:
               case nir_op_fsign:
               case nir_op_i2f16:
               case nir_op_i2f32:
               case nir_op_u2f16:
               case nir_op_u2f32:
               case nir_op_f2f16:
               case nir_op_f2f16_rtz:
               case nir_op_f2f16_rtne:
               case nir_op_f2f32:
               case nir_op_fquantize2f16:
               case nir_op_ffract:
               case nir_op_ffloor:
               case nir_op_fceil:
               case nir_op_ftrunc:
               case nir_op_fround_even:
               case nir_op_frcp:
               case nir_op_frsq:
               case nir_op_fsqrt:
               case nir_op_fexp2:
               case nir_op_flog2:
               case nir_op_fsin_amd:
               case nir_op_fcos_amd:
               case nir_op_pack_half_2x16_rtz_split:
               case nir_op_pack_half_2x16_split:
               case nir_op_unpack_half_2x16_split_x:
               case nir_op_unpack_half_2x16_split_y: {
                  if (ctx->program->gfx_level < GFX11_5 ||
                      alu_instr->src[0].src.ssa->bit_size > 32) {
                     type = RegType::vgpr;
                     break;
                  }
                  FALLTHROUGH;
               }
               default:
                  for (unsigned i = 0; i < nir_op_infos[alu_instr->op].num_inputs; i++) {
                     if (alu_instr->src[i].src.ssa->bit_size == 1
                            ? nir_src_is_divergent(&alu_instr->src[i].src)
                            : regclasses[alu_instr->src[i].src.ssa->index].type() == RegType::vgpr)
                        type = RegType::vgpr;
                  }
                  break;
               }

               RegClass rc =
                  get_reg_class(ctx, type, alu_instr->def.num_components, alu_instr->def.bit_size);
               regclasses[alu_instr->def.index] = rc;
               break;
            }
            case nir_instr_type_load_const: {
               unsigned num_components = nir_instr_as_load_const(instr)->def.num_components;
               unsigned bit_size = nir_instr_as_load_const(instr)->def.bit_size;
               RegClass rc = get_reg_class(ctx, RegType::sgpr, num_components, bit_size);
               regclasses[nir_instr_as_load_const(instr)->def.index] = rc;
               break;
            }
            case nir_instr_type_intrinsic: {
               nir_intrinsic_instr* intrinsic = nir_instr_as_intrinsic(instr);
               if (!nir_intrinsic_infos[intrinsic->intrinsic].has_dest)
                  break;
               if (intrinsic->intrinsic == nir_intrinsic_strict_wqm_coord_amd) {
                  regclasses[intrinsic->def.index] =
                     RegClass::get(RegType::vgpr, intrinsic->def.num_components * 4 +
                                                     nir_intrinsic_base(intrinsic))
                        .as_linear();
                  break;
               }
               RegType type = RegType::sgpr;
               switch (intrinsic->intrinsic) {
               case nir_intrinsic_load_push_constant:
               case nir_intrinsic_load_workgroup_id:
               case nir_intrinsic_load_num_workgroups:
               case nir_intrinsic_load_sbt_base_amd:
               case nir_intrinsic_load_subgroup_id:
               case nir_intrinsic_load_num_subgroups:
               case nir_intrinsic_vote_all:
               case nir_intrinsic_vote_any:
               case nir_intrinsic_read_first_invocation:
               case nir_intrinsic_as_uniform:
               case nir_intrinsic_read_invocation:
               case nir_intrinsic_first_invocation:
               case nir_intrinsic_ballot:
               case nir_intrinsic_ballot_relaxed:
               case nir_intrinsic_bindless_image_samples:
               case nir_intrinsic_load_scalar_arg_amd:
               case nir_intrinsic_load_smem_amd:
               case nir_intrinsic_unit_test_uniform_amd: type = RegType::sgpr; break;
               case nir_intrinsic_load_input:
               case nir_intrinsic_load_per_primitive_input:
               case nir_intrinsic_load_output:
               case nir_intrinsic_load_input_vertex:
               case nir_intrinsic_load_per_vertex_input:
               case nir_intrinsic_load_per_vertex_output:
               case nir_intrinsic_load_interpolated_input:
               case nir_intrinsic_write_invocation_amd:
               case nir_intrinsic_mbcnt_amd:
               case nir_intrinsic_lane_permute_16_amd:
               case nir_intrinsic_dpp16_shift_amd:
               case nir_intrinsic_ssbo_atomic:
               case nir_intrinsic_ssbo_atomic_swap:
               case nir_intrinsic_global_atomic_amd:
               case nir_intrinsic_global_atomic_swap_amd:
               case nir_intrinsic_bindless_image_atomic:
               case nir_intrinsic_bindless_image_atomic_swap:
               case nir_intrinsic_bindless_image_size:
               case nir_intrinsic_shared_atomic:
               case nir_intrinsic_shared_atomic_swap:
               case nir_intrinsic_load_scratch:
               case nir_intrinsic_load_typed_buffer_amd:
               case nir_intrinsic_load_buffer_amd:
               case nir_intrinsic_load_initial_edgeflags_amd:
               case nir_intrinsic_gds_atomic_add_amd:
               case nir_intrinsic_bvh64_intersect_ray_amd:
               case nir_intrinsic_bvh8_intersect_ray_amd:
               case nir_intrinsic_load_vector_arg_amd:
               case nir_intrinsic_ordered_xfb_counter_add_gfx11_amd:
               case nir_intrinsic_cmat_muladd_amd:
               case nir_intrinsic_unit_test_divergent_amd: type = RegType::vgpr; break;
               case nir_intrinsic_load_shared:
               case nir_intrinsic_load_shared2_amd:
                  /* When the result of these loads is only used by cross-lane instructions,
                   * it is beneficial to use a VGPR destination. This is because this allows
                   * to put the s_waitcnt further down, which decreases latency.
                   */
                  if (only_used_by_cross_lane_instrs(&intrinsic->def)) {
                     type = RegType::vgpr;
                     break;
                  }
                  FALLTHROUGH;
               case nir_intrinsic_shuffle:
               case nir_intrinsic_quad_broadcast:
               case nir_intrinsic_quad_swap_horizontal:
               case nir_intrinsic_quad_swap_vertical:
               case nir_intrinsic_quad_swap_diagonal:
               case nir_intrinsic_quad_swizzle_amd:
               case nir_intrinsic_masked_swizzle_amd:
               case nir_intrinsic_rotate:
               case nir_intrinsic_inclusive_scan:
               case nir_intrinsic_exclusive_scan:
               case nir_intrinsic_reduce:
               case nir_intrinsic_load_ubo:
               case nir_intrinsic_load_ssbo:
               case nir_intrinsic_load_global_amd:
                  type = intrinsic->def.divergent ? RegType::vgpr : RegType::sgpr;
                  break;
               case nir_intrinsic_ddx:
               case nir_intrinsic_ddy:
               case nir_intrinsic_ddx_fine:
               case nir_intrinsic_ddy_fine:
               case nir_intrinsic_ddx_coarse:
               case nir_intrinsic_ddy_coarse: type = RegType::vgpr; break;
               default:
                  for (unsigned i = 0; i < nir_intrinsic_infos[intrinsic->intrinsic].num_srcs;
                       i++) {
                     if (regclasses[intrinsic->src[i].ssa->index].type() == RegType::vgpr)
                        type = RegType::vgpr;
                  }
                  break;
               }
               RegClass rc =
                  get_reg_class(ctx, type, intrinsic->def.num_components, intrinsic->def.bit_size);
               regclasses[intrinsic->def.index] = rc;
               break;
            }
            case nir_instr_type_tex: {
               nir_tex_instr* tex = nir_instr_as_tex(instr);
               RegType type = tex->def.divergent ? RegType::vgpr : RegType::sgpr;

               if (tex->op == nir_texop_texture_samples) {
                  assert(!tex->def.divergent);
               }

               RegClass rc = get_reg_class(ctx, type, tex->def.num_components, tex->def.bit_size);
               regclasses[tex->def.index] = rc;
               break;
            }
            case nir_instr_type_undef: {
               unsigned num_components = nir_instr_as_undef(instr)->def.num_components;
               unsigned bit_size = nir_instr_as_undef(instr)->def.bit_size;
               RegClass rc = get_reg_class(ctx, RegType::sgpr, num_components, bit_size);
               regclasses[nir_instr_as_undef(instr)->def.index] = rc;
               break;
            }
            case nir_instr_type_phi: {
               nir_phi_instr* phi = nir_instr_as_phi(instr);
               RegType type = RegType::sgpr;
               unsigned num_components = phi->def.num_components;
               assert((phi->def.bit_size != 1 || num_components == 1) &&
                      "Multiple components not supported on boolean phis.");

               if (phi->def.divergent) {
                  type = RegType::vgpr;
               } else {
                  bool vgpr_src = false;
                  nir_foreach_phi_src (src, phi)
                     vgpr_src |= regclasses[src->src.ssa->index].type() == RegType::vgpr;

                  if (vgpr_src) {
                     type = RegType::vgpr;

                     /* This might be the case because of nir_divergence_ignore_undef_if_phi_srcs. */
                     bool divergent_merge = false;
                     if (nir_cf_node_prev(&block->cf_node) &&
                         nir_cf_node_prev(&block->cf_node)->type == nir_cf_node_if) {
                        nir_if* nif = nir_cf_node_as_if(nir_cf_node_prev(&block->cf_node));
                        divergent_merge = nir_src_is_divergent(&nif->condition);
                     }

                     /* In case of uniform phis after divergent merges, ensure that the dst is an
                      * SGPR and does not contain undefined values for some invocations.
                      */
                     if (divergent_merge && !skip_uniformize_merge_phi(&phi->def, 0))
                        type = RegType::sgpr;
                  }
               }

               RegClass rc = get_reg_class(ctx, type, num_components, phi->def.bit_size);
               if (rc != regclasses[phi->def.index])
                  done = false;
               regclasses[phi->def.index] = rc;
               break;
            }
            default: break;
            }
         }
      }
   }

   ctx->program->config->spi_ps_input_ena = ctx->program->info.ps.spi_ps_input_ena;
   ctx->program->config->spi_ps_input_addr = ctx->program->info.ps.spi_ps_input_addr;

   /* align and copy constant data */
   while (ctx->program->constant_data.size() % 4u)
      ctx->program->constant_data.push_back(0);
   ctx->constant_data_offset = ctx->program->constant_data.size();
   ctx->program->constant_data.insert(ctx->program->constant_data.end(),
                                      (uint8_t*)shader->constant_data,
                                      (uint8_t*)shader->constant_data + shader->constant_data_size);

   BITSET_CLEAR_RANGE(ctx->output_args, 0, BITSET_SIZE(ctx->output_args));
}

void
cleanup_context(isel_context* ctx)
{
   _mesa_hash_table_destroy(ctx->range_ht, NULL);
}

isel_context
setup_isel_context(Program* program, unsigned shader_count, struct nir_shader* const* shaders,
                   ac_shader_config* config, const struct aco_compiler_options* options,
                   const struct aco_shader_info* info, const struct ac_shader_args* args,
                   SWStage sw_stage)
{
   for (unsigned i = 0; i < shader_count; i++) {
      switch (shaders[i]->info.stage) {
      case MESA_SHADER_VERTEX: sw_stage = sw_stage | SWStage::VS; break;
      case MESA_SHADER_TESS_CTRL: sw_stage = sw_stage | SWStage::TCS; break;
      case MESA_SHADER_TESS_EVAL: sw_stage = sw_stage | SWStage::TES; break;
      case MESA_SHADER_GEOMETRY: sw_stage = sw_stage | SWStage::GS; break;
      case MESA_SHADER_FRAGMENT: sw_stage = sw_stage | SWStage::FS; break;
      case MESA_SHADER_KERNEL:
      case MESA_SHADER_COMPUTE: sw_stage = sw_stage | SWStage::CS; break;
      case MESA_SHADER_TASK: sw_stage = sw_stage | SWStage::TS; break;
      case MESA_SHADER_MESH: sw_stage = sw_stage | SWStage::MS; break;
      case MESA_SHADER_RAYGEN:
      case MESA_SHADER_CLOSEST_HIT:
      case MESA_SHADER_MISS:
      case MESA_SHADER_CALLABLE:
      case MESA_SHADER_INTERSECTION:
      case MESA_SHADER_ANY_HIT: sw_stage = SWStage::RT; break;
      default: UNREACHABLE("Shader stage not implemented");
      }
   }

   init_program(program, Stage{info->hw_stage, sw_stage}, info, options->gfx_level, options->family,
                options->wgp_mode, config);

   isel_context ctx = {};
   ctx.program = program;
   ctx.args = args;
   ctx.options = options;
   ctx.stage = program->stage;

   program->workgroup_size = program->info.workgroup_size;
   assert(program->workgroup_size);

   /* Mesh shading only works on GFX10.3+. */
   ASSERTED bool mesh_shading = ctx.stage.has(SWStage::TS) || ctx.stage.has(SWStage::MS);
   assert(!mesh_shading || ctx.program->gfx_level >= GFX10_3);

   setup_tcs_info(&ctx);

   calc_min_waves(program);

   unsigned scratch_size = 0;
   for (unsigned i = 0; i < shader_count; i++) {
      nir_shader* nir = shaders[i];
      setup_nir(&ctx, nir);
      setup_lds_size(&ctx, nir);
   }

   for (unsigned i = 0; i < shader_count; i++)
      scratch_size = std::max(scratch_size, shaders[i]->scratch_size);

   ctx.program->config->scratch_bytes_per_wave = scratch_size * ctx.program->wave_size;

   unsigned nir_num_blocks = 0;
   for (unsigned i = 0; i < shader_count; i++)
      nir_num_blocks += nir_shader_get_entrypoint(shaders[i])->num_blocks;
   ctx.program->blocks.reserve(nir_num_blocks * 2);
   ctx.block = ctx.program->create_and_insert_block();
   ctx.block->kind = block_kind_top_level;

   return ctx;
}

} // namespace aco
