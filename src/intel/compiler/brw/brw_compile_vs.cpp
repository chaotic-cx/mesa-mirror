/*
 * Copyright © 2011 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "brw_shader.h"
#include "brw_eu.h"
#include "brw_nir.h"
#include "brw_private.h"
#include "dev/intel_debug.h"

static bool
run_vs(brw_shader &s)
{
   assert(s.stage == MESA_SHADER_VERTEX);

   s.payload_ = new brw_vs_thread_payload(s);

   brw_from_nir(&s);

   if (s.failed)
      return false;

   brw_calculate_cfg(s);

   ASSERTED bool eot = s.mark_last_urb_write_with_eot();
   assert(eot);

   brw_optimize(s);

   s.assign_curb_setup();
   brw_assign_urb_setup(s);

   brw_lower_3src_null_dest(s);
   brw_workaround_emit_dummy_mov_instruction(s);

   brw_allocate_registers(s, true /* allow_spilling */);

   brw_workaround_source_arf_before_eot(s);

   return !s.failed;
}

extern "C" intel_shader_bin *
brw_compile_vs(const struct brw_compiler *compiler,
               struct brw_compile_params *params)
{
   struct nir_shader *nir = params->nir;
   const struct brw_vs_prog_key *key =
      (const struct brw_vs_prog_key *)params->key;
   struct brw_vs_prog_data *prog_data =
      (struct brw_vs_prog_data *)params->prog_data;
   const bool debug_enabled =
      brw_should_print_shader(nir, params->debug_flag ?
                                   params->debug_flag : DEBUG_VS,
                                   params->source_hash);
   const unsigned dispatch_width = brw_geometry_stage_dispatch_width(compiler->devinfo);

   /* We only expect slot compaction to be disabled when using device
    * generated commands, to provide an independent 3DSTATE_VERTEX_ELEMENTS
    * programming. This should always be enabled together with VF component
    * packing to minimize the size of the payload.
    */
   assert(!key->no_vf_slot_compaction || key->vf_component_packing);

   brw_pass_tracker pt_ = {
      .nir = nir,
      .dispatch_width = dispatch_width,
      .compiler = compiler,
      .key = &key->base,
      .archiver = params->archiver,
   }, *pt = &pt_;

   BRW_NIR_SNAPSHOT("first");

   brw_prog_data_init(&prog_data->base.base, params);

   const unsigned ray_queries = nir->info.ray_queries;

   /* When using Primitive Replication for multiview, each view gets its own
    * position slot.
    */
   const uint32_t pos_slots =
      (nir->info.per_view_outputs & VARYING_BIT_POS) ?
      MAX2(1, util_bitcount(key->base.view_mask)) : 1;

   /* Only position is allowed to be per-view */
   assert(!(nir->info.per_view_outputs & ~VARYING_BIT_POS));

   struct intel_vue_map vue_map;
   brw_compute_vue_map(compiler->devinfo,
                       &vue_map, nir->info.outputs_written,
                       key->base.vue_layout, pos_slots);

   brw_nir_apply_key(pt, &key->base, dispatch_width);

   const uint64_t inputs_read = nir->info.inputs_read;
   const uint64_t double_inputs_read = nir->info.vs.double_inputs;

   unsigned nr_input_components = 0;
   unsigned urb_read_length = 0;
   uint32_t vf_component_packing[4] = {};
   brw_nir_lower_vs_inputs(nir, compiler->devinfo, key,
                           &params->push_layout, vf_component_packing,
                           &nr_input_components,
                           &urb_read_length);
   brw_nir_lower_vue_outputs(nir);
   BRW_NIR_SNAPSHOT("after_lower_io");

   brw_postprocess_nir(pt, debug_enabled);

   BRW_NIR_PASS(brw_nir_lower_deferred_urb_writes, compiler->devinfo,
                &vue_map, 0, 0);

   /* Since vertex shaders reuse the same VUE entry for inputs and outputs
    * (overwriting the original contents), we need to make sure the size is
    * the larger of the two.
    */
   const unsigned vue_entries =
      MAX2(DIV_ROUND_UP(nr_input_components, 4),
           (unsigned)vue_map.num_slots);

   const unsigned urb_entry_size = DIV_ROUND_UP(vue_entries, 4);

   if (unlikely(debug_enabled)) {
      fprintf(stderr, "VS Output ");
      brw_print_vue_map(stderr, &vue_map, MESA_SHADER_VERTEX);
   }

   brw_shader_params shader_params = {
      .compiler                = compiler,
      .compile_params          = params,
      .nir                     = nir,
      .info = {
         .source_hash = params->source_hash,
         .ray_queries = ray_queries,
         .vue = {
            .vue_map = vue_map,
            .urb_read_length = urb_read_length,
            .urb_entry_size = urb_entry_size,
            .dispatch_mode = INTEL_DISPATCH_MODE_SIMD8,
         },
         .vs = {
            .inputs_read = inputs_read,
            .double_inputs_read = double_inputs_read,
            .uses_vertexid = BITSET_TEST(nir->info.system_values_read,
                                         SYSTEM_VALUE_VERTEX_ID_ZERO_BASE),
            .uses_instanceid = BITSET_TEST(nir->info.system_values_read,
                                           SYSTEM_VALUE_INSTANCE_ID),
            .uses_firstvertex = BITSET_TEST(nir->info.system_values_read,
                                            SYSTEM_VALUE_FIRST_VERTEX),
            .uses_baseinstance = BITSET_TEST(nir->info.system_values_read,
                                             SYSTEM_VALUE_BASE_INSTANCE),
            .uses_drawid = BITSET_TEST(nir->info.system_values_read,
                                       SYSTEM_VALUE_DRAW_ID),
         },
      },
      .dispatch_width          = dispatch_width,
      .needs_register_pressure = params->collect_register_pressure,
      .debug_enabled           = debug_enabled,
   };
   memcpy(shader_params.info.vs.vf_component_packing, vf_component_packing,
          sizeof(vf_component_packing));
   brw_shader v(&shader_params);
   if (!run_vs(v)) {
      params->error_str =
         ralloc_strdup(params->mem_ctx, v.fail_msg);
      return NULL;
   }

   const brw_to_binary_params to_binary_params = {
      .compiler = compiler,
      .params = params,
      .prog_data = &prog_data->base.base,
      .shaders = { &v },
   };
   return brw_to_binary(&to_binary_params);
}
