/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "compiler/brw/brw_compiler.h"
#include "compiler/brw/brw_nir.h"
#include "compiler/intel_nir.h"
#include "compiler/intel_prim.h"
#include "jay_private.h"
#include "nir.h"

static inline enum intel_barycentric_mode
brw_barycentric_mode(nir_intrinsic_instr *intr)
{
   const enum glsl_interp_mode mode = nir_intrinsic_interp_mode(intr);

   /* Barycentric modes don't make sense for flat inputs. */
   assert(mode != INTERP_MODE_FLAT);

   unsigned bary;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_at_offset:
      bary = INTEL_BARYCENTRIC_PERSPECTIVE_PIXEL;
      break;
   case nir_intrinsic_load_barycentric_centroid:
      bary = INTEL_BARYCENTRIC_PERSPECTIVE_CENTROID;
      break;
   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_barycentric_at_sample:
      bary = INTEL_BARYCENTRIC_PERSPECTIVE_SAMPLE;
      break;
   default:
      UNREACHABLE("invalid intrinsic");
   }

   if (mode == INTERP_MODE_NOPERSPECTIVE)
      bary += 3;

   return (enum intel_barycentric_mode) bary;
}

struct fs_info_ctx {
   const struct intel_device_info *devinfo;
   struct intel_shader_info *info;
   unsigned interp_modes;
   unsigned offset_interp_modes;
};

static bool
gather_fs_info(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct fs_info_ctx *ctx = data;
   struct intel_shader_info *info = ctx->info;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample:
      ctx->interp_modes |= BITFIELD_BIT(brw_barycentric_mode(intr));
      break;

   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_barycentric_at_offset:
      ctx->offset_interp_modes |= BITFIELD_BIT(brw_barycentric_mode(intr));
      info->fs.uses_src_xy = true;
      break;

   case nir_intrinsic_load_frag_coord_z:
      info->fs.uses_src_depth = true;
      break;

   case nir_intrinsic_load_frag_coord_w:
      info->fs.uses_src_w = true;
      break;

   case nir_intrinsic_load_coverage_mask_intel:
      info->fs.uses_sample_mask = true;
      break;

   case nir_intrinsic_load_pixel_coord:
      info->fs.uses_src_xy = true;
      break;

   case nir_intrinsic_load_sample_pos_from_id:
      info->fs.uses_sample_offsets = true;
      break;

   default:
      break;
   }

   return false;
}

static void
brw_compute_flat_inputs(struct intel_shader_info *info,
                        const nir_shader *shader)
{
   info->fs.flat_inputs = 0;

   nir_foreach_shader_in_variable(var, shader) {
      if (var->data.interpolation != INTERP_MODE_FLAT ||
          var->data.per_primitive)
         continue;

      unsigned slots = glsl_count_attribute_slots(var->type, false);
      for (unsigned s = 0; s < slots; s++) {
         int input_index = info->fs.urb_setup[var->data.location + s];

         if (input_index >= 0)
            info->fs.flat_inputs |= 1 << input_index;
      }
   }
}

static uint8_t
computed_depth_mode(const nir_shader *shader)
{
   if (shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)) {
      switch (shader->info.fs.depth_layout) {
      case FRAG_DEPTH_LAYOUT_NONE:
      case FRAG_DEPTH_LAYOUT_ANY:
         return BRW_PSCDEPTH_ON;
      case FRAG_DEPTH_LAYOUT_GREATER:
         return BRW_PSCDEPTH_ON_GE;
      case FRAG_DEPTH_LAYOUT_LESS:
         return BRW_PSCDEPTH_ON_LE;
      case FRAG_DEPTH_LAYOUT_UNCHANGED:
         /* We initially set this to OFF, but having the shader write the
          * depth means we allocate register space in the SEND message. The
          * difference between the SEND register count and the OFF state
          * programming makes the HW hang.
          *
          * Removing the depth writes also leads to test failures. So use
          * LesserThanOrEqual, which fits writing the same value
          * (unchanged/equal).
          *
          */
         return BRW_PSCDEPTH_ON_LE;
      }
   }
   return BRW_PSCDEPTH_OFF;
}

/*
 * Build up an array of indices into the urb_setup array that
 * references the active entries of the urb_setup array.
 * Used to accelerate walking the active entries of the urb_setup array
 * on each upload.
 */
static void
brw_compute_urb_setup_index(struct intel_shader_info *info)
{
   /* TODO(mesh): Review usage of this in the context of Mesh, we may want to
    * skip per-primitive attributes here.
    */

   /* Make sure uint8_t is sufficient */
   static_assert(VARYING_SLOT_MAX <= 0xff);
   uint8_t index = 0;
   for (uint8_t attr = 0; attr < VARYING_SLOT_MAX; attr++) {
      if (info->fs.urb_setup[attr] >= 0) {
         info->fs.urb_setup_attribs[index++] = attr;
      }
   }
   info->fs.urb_setup_attribs_count = index;
}

static void
calculate_urb_setup(const struct intel_device_info *devinfo,
                    const struct brw_fs_prog_key *key,
                    struct intel_shader_info *info,
                    nir_shader *nir,
                    const struct brw_mue_map *mue_map,
                    int *per_primitive_offsets)
{
   memset(info->fs.urb_setup, -1, sizeof(info->fs.urb_setup));
   int urb_next = 0; /* in vec4s */

   /* Figure out where the PrimitiveID lives, either in the per-vertex block
    * or in the per-primitive block or both.
    */
   const uint64_t per_vert_primitive_id =
      key->mesh_input == INTEL_ALWAYS ? 0 : VARYING_BIT_PRIMITIVE_ID;
   const uint64_t per_prim_primitive_id =
      key->mesh_input == INTEL_NEVER ? 0 : VARYING_BIT_PRIMITIVE_ID;
   const uint64_t inputs_read =
      nir->info.inputs_read &
      (~nir->info.per_primitive_inputs | per_vert_primitive_id);
   const uint64_t per_primitive_header_bits =
      VARYING_BIT_PRIMITIVE_SHADING_RATE |
      VARYING_BIT_LAYER |
      VARYING_BIT_VIEWPORT |
      VARYING_BIT_CULL_PRIMITIVE;
   const uint64_t per_primitive_inputs =
      nir->info.inputs_read &
      (nir->info.per_primitive_inputs | per_prim_primitive_id) &
      ~per_primitive_header_bits;
   struct intel_vue_map vue_map;
   uint32_t per_primitive_stride = 0, first_read_offset = UINT32_MAX;

   if (mue_map != NULL) {
      memcpy(&vue_map, &mue_map->vue_map, sizeof(vue_map));
      memcpy(per_primitive_offsets, mue_map->per_primitive_offsets,
             sizeof(mue_map->per_primitive_offsets));

      if (!mue_map->wa_18019110168_active) {
         u_foreach_bit64(location, per_primitive_inputs) {
            assert(per_primitive_offsets[location] != -1);

            first_read_offset =
               MIN2(first_read_offset,
                    (uint32_t) per_primitive_offsets[location]);
            per_primitive_stride =
               MAX2((uint32_t) per_primitive_offsets[location] + 16,
                    per_primitive_stride);
         }
      } else {
         first_read_offset = per_primitive_stride = 0;
      }
   } else {
      brw_compute_vue_map(devinfo, &vue_map, inputs_read, key->base.vue_layout,
                          1 /* pos_slots, TODO */);
      brw_compute_per_primitive_map(per_primitive_offsets,
                                    &per_primitive_stride, &first_read_offset,
                                    0, nir, nir_var_shader_in,
                                    per_primitive_inputs,
                                    true /* separate_shader */);
   }

   if (per_primitive_stride > first_read_offset) {
      first_read_offset = ROUND_DOWN_TO(first_read_offset, 32);

      /* Remove the first few unused registers */
      for (uint32_t i = 0; i < VARYING_SLOT_MAX; i++) {
         if (per_primitive_offsets[i] == -1)
            continue;
         per_primitive_offsets[i] -= first_read_offset;
      }

      info->fs.num_per_primitive_inputs =
         2 * DIV_ROUND_UP(per_primitive_stride - first_read_offset, 32);
   } else {
      info->fs.num_per_primitive_inputs = 0;
   }

   /* Now do the per-vertex stuff (what used to be legacy pipeline) */

   /* If Mesh is involved, we cannot do any packing. Documentation doesn't say
    * anything about this but 3DSTATE_SBE_SWIZ does not appear to work when
    * using Mesh.
    */
   if (util_bitcount64(inputs_read) <= 16 && key->mesh_input == INTEL_NEVER) {
      /* When not in Mesh pipeline mode, the SF/SBE pipeline stage can do
       * arbitrary rearrangement of the first 16 varying inputs, so we can put
       * them wherever we want. Just put them in order.
       *
       * This is useful because it means that (a) inputs not used by the
       * fragment shader won't take up valuable register space, and (b) we
       * won't have to recompile the fragment shader if it gets paired with a
       * different vertex (or geometry) shader.
       */
      for (unsigned int i = 0; i < VARYING_SLOT_MAX; i++) {
         if (inputs_read & BITFIELD64_BIT(i)) {
            info->fs.urb_setup[i] = urb_next++;
         }
      }
   } else {
      /* We have enough input varyings that the SF/SBE pipeline stage can't
       * arbitrarily rearrange them to suit our whim; we have to put them in
       * an order that matches the output of the previous pipeline stage
       * (geometry or vertex shader).
       */
      int first_slot = 0;
      for (int i = 0; i < vue_map.num_slots; i++) {
         int varying = vue_map.slot_to_varying[i];
         if (varying > 0 && (inputs_read & BITFIELD64_BIT(varying)) != 0) {
            first_slot = ROUND_DOWN_TO(i, 2);
            break;
         }
      }

      for (int slot = first_slot; slot < vue_map.num_slots; slot++) {
         int varying = vue_map.slot_to_varying[slot];
         if (varying > 0 && (inputs_read & BITFIELD64_BIT(varying))) {
            info->fs.urb_setup[varying] = slot - first_slot;
         }
      }
      urb_next = vue_map.num_slots - first_slot;
   }

   info->fs.num_varying_inputs = urb_next;
   info->fs.inputs = inputs_read;
   info->fs.per_primitive_inputs = per_primitive_inputs;

   brw_compute_urb_setup_index(info);
}

static void
populate_fs_prog_data(nir_shader *shader,
                      const struct intel_device_info *devinfo,
                      const struct brw_fs_prog_key *key,
                      const struct brw_mue_map *mue_map,
                      int *per_primitive_offsets,
                      struct intel_shader_info *info)
{
   /* Sanity check, the driver should be able to be sensible here. */
   assert(key->multisample_fbo != INTEL_NEVER || !key->persample_interp);

   struct fs_info_ctx ctx = {
      .devinfo = devinfo,
      .info = info,
   };
   nir_shader_intrinsics_pass(shader, gather_fs_info, nir_metadata_all, &ctx);

   /* Prior to Gfx20, HW has a pixel interpolator which needs to be configured
    * appropriately for interpolation at offset/sample. Gfx20+ does that
    * calculation in software so we only need to look at offset_interp_modes.
    */
   if (devinfo->ver < 20)
      ctx.interp_modes |= ctx.offset_interp_modes;

   info->fs.barycentric_interp_modes |= ctx.interp_modes;
   const bool sample_shading = shader->info.fs.uses_sample_shading;
   info->fs.persample_interp = sample_shading || key->persample_interp;

   /* If we have both pixel & sample interpolation we need both to per sample
    * dispatch, otherwise spec allows us to fallback to pixel in non MSAA
    * cases.
    */
   const unsigned interp_at_pixel_and_sample_bits =
      BITFIELD_BIT(INTEL_BARYCENTRIC_PERSPECTIVE_PIXEL) |
      BITFIELD_BIT(INTEL_BARYCENTRIC_PERSPECTIVE_SAMPLE);
   const bool interp_at_pixel_and_sample =
      (ctx.interp_modes & interp_at_pixel_and_sample_bits) ==
      interp_at_pixel_and_sample_bits;
   info->fs.persample_dispatch = (info->fs.persample_interp &&
                                  key->multisample_fbo >= INTEL_SOMETIMES) ||
                                 interp_at_pixel_and_sample;

   /* Move sample barycentric modes to pixel when persample dispatch is always
    * disabled.
    */
   {
      unsigned tmp = 0;
      u_foreach_bit(b, ctx.interp_modes) {
         tmp |= BITFIELD_BIT(intel_fs_barycentric_mode_for_persample_dispatch(
            info->fs.persample_dispatch, (enum intel_barycentric_mode) b));
      }
      ctx.interp_modes = tmp;
   }

   info->fs.uses_kill = shader->info.fs.uses_discard;
   info->fs.uses_omask =
      !key->ignore_sample_mask_out &&
      (shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK));
   info->fs.max_polygons = 1;
   info->fs.computed_depth_mode = computed_depth_mode(shader);
   info->fs.computed_stencil =
      shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_STENCIL);

   info->fs.dual_src_blend =
      shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DUAL_SRC_BLEND);

   /* Gate alpha to coverage with the draw buffer 0 being written. */
   info->fs.alpha_to_coverage = (shader->info.outputs_written &
                                 (BITFIELD64_BIT(FRAG_RESULT_COLOR) |
                                  BITFIELD64_BIT(FRAG_RESULT_DATA0))) != 0 ?
                                   key->alpha_to_coverage :
                                   INTEL_NEVER;

   assert(devinfo->verx10 >= 125 || key->mesh_input == INTEL_NEVER);

   assert(devinfo->verx10 >= 200 || key->provoking_vertex_last == INTEL_NEVER);
   info->fs.provoking_vertex_last = key->provoking_vertex_last;

   /* From the Ivy Bridge PRM documentation for 3DSTATE_PS:
    *
    *    "MSDISPMODE_PERSAMPLE is required in order to select
    *    POSOFFSET_SAMPLE"
    *
    * So we can only really get sample positions if we are doing real
    * per-sample dispatch.  If we need gl_SamplePosition and we don't have
    * persample dispatch, we hard-code it to 0.5.
    */
   info->fs.uses_pos_offset =
      info->fs.persample_dispatch &&
      (BITSET_TEST(shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_POS) ||
       BITSET_TEST(shader->info.system_values_read,
                   SYSTEM_VALUE_SAMPLE_POS_OR_CENTER));

   info->fs.early_fragment_tests = shader->info.fs.early_fragment_tests;
   info->fs.post_depth_coverage = shader->info.fs.post_depth_coverage;
   info->fs.inner_coverage = shader->info.fs.inner_coverage;

   if (devinfo->ver >= 20) {
      info->fs.vertex_attributes_bypass =
         brw_needs_vertex_attributes_bypass(shader);

      info->fs.uses_npc_bary_coefficients =
         ctx.offset_interp_modes & INTEL_BARYCENTRIC_NONPERSPECTIVE_BITS;
      info->fs.uses_pc_bary_coefficients =
         ctx.offset_interp_modes & ~INTEL_BARYCENTRIC_NONPERSPECTIVE_BITS;
      info->fs.uses_sample_offsets |=
         ctx.offset_interp_modes &
         ((1 << INTEL_BARYCENTRIC_PERSPECTIVE_SAMPLE) |
          (1 << INTEL_BARYCENTRIC_NONPERSPECTIVE_SAMPLE));
   }

   info->fs.uses_nonperspective_interp_modes =
      (info->fs.barycentric_interp_modes &
       INTEL_BARYCENTRIC_NONPERSPECTIVE_BITS) ||
      info->fs.uses_npc_bary_coefficients;

   /* Variable rate shading & sample shading are 2 features that are known at
    * compile time in Vulkan GPL & ESO scenarios, so a driver should not be
    * setting both at the same time.
    */
   assert(!key->coarse_pixel || !key->persample_interp);

   info->fs.coarse_pixel_dispatch = !info->fs.persample_dispatch;
   if (!key->coarse_pixel ||
       /* DG2 should support this, but Wa_22012766191 says there are issues
        * with CPS 1x1 + MSAA + FS writing to oMask.
        */
       (devinfo->verx10 < 200 &&
        (info->fs.uses_omask || info->fs.uses_sample_mask)) ||
       sample_shading ||
       (info->fs.computed_depth_mode != BRW_PSCDEPTH_OFF) ||
       info->fs.computed_stencil ||
       devinfo->ver < 11) {
      info->fs.coarse_pixel_dispatch = false;
   }

   /* ICL PRMs, Volume 9: Render Engine, Shared Functions Pixel Interpolater,
    * Message Descriptor :
    *
    *    "Message Type. Specifies the type of message being sent when
    *     pixel-rate evaluation is requested :
    *
    *     Format = U2
    *       0: Per Message Offset (eval_snapped with immediate offset)
    *       1: Sample Position Offset (eval_sindex)
    *       2: Centroid Position Offset (eval_centroid)
    *       3: Per Slot Offset (eval_snapped with register offset)
    *
    *     Message Type. Specifies the type of message being sent when
    *     coarse-rate evaluation is requested :
    *
    *     Format = U2
    *       0: Coarse to Pixel Mapping Message (internal message)
    *       1: Reserved
    *       2: Coarse Centroid Position (eval_centroid)
    *       3: Per Slot Coarse Pixel Offset (eval_snapped with register offset)"
    *
    * The Sample Position Offset is marked as reserved for coarse rate
    * evaluation and leads to hangs if we try to use it. So disable coarse
    * pixel shading if we have any intrinsic that will result in a pixel
    * interpolater message at sample.
    */
   if (intel_nir_pulls_at_sample(shader))
      info->fs.coarse_pixel_dispatch = false;

   /* TODO: Proprietary compilers are probably bloated with passes to
    * aggressively eliminate any instruction that computes derivatives from a
    * convergent source, we need to factor that into our decision here if we
    * want to guarantee that helper lane behavior matches the expectations of
    * broken apps.
    */
   info->fs.uses_vmask =
      shader->info.fs.needs_coarse_quad_helper_invocations ||
      shader->info.uses_wide_subgroup_intrinsics;

   info->fs.uses_depth_w_coefficients = info->fs.uses_pc_bary_coefficients;

   if (info->fs.coarse_pixel_dispatch) {
      info->fs.uses_depth_w_coefficients |= info->fs.uses_src_depth;
      info->fs.uses_src_xy |= info->fs.uses_src_depth;
      info->fs.uses_src_depth = false;
   }

   calculate_urb_setup(devinfo, key, info, shader, mue_map,
                       per_primitive_offsets);
   brw_compute_flat_inputs(info, shader);

   info->fs.has_side_effects = shader->info.writes_memory;
}

static void
populate_vs_prog_data(nir_shader *nir,
                      const struct intel_device_info *devinfo,
                      struct intel_shader_info *info)
{
   BITSET_WORD *sysvals = nir->info.system_values_read;

   const struct {
      bool *data;
      gl_system_value val;
   } bool_sysvals[] = {
      { &info->vs.uses_firstvertex,     SYSTEM_VALUE_FIRST_VERTEX        },
      { &info->vs.uses_baseinstance,    SYSTEM_VALUE_BASE_INSTANCE       },
      { &info->vs.uses_vertexid,        SYSTEM_VALUE_VERTEX_ID_ZERO_BASE },
      { &info->vs.uses_instanceid,      SYSTEM_VALUE_INSTANCE_ID         },
      { &info->vs.uses_drawid,          SYSTEM_VALUE_DRAW_ID             },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(bool_sysvals); ++i) {
      *bool_sysvals[i].data = BITSET_TEST(sysvals, bool_sysvals[i].val);
   }

   info->vue.dispatch_mode = INTEL_DISPATCH_MODE_SIMD8;
}

static void
populate_tcs_prog_data(nir_shader *nir,
                       const struct brw_tcs_prog_key *key,
                       struct intel_shader_info *info)
{
   brw_fill_tess_info_from_shader_info(&info->tcs.tess_info, &nir->info);

   info->tcs.input_vertices = key->input_vertices;
   info->tcs.output_vertices = nir->info.tess.tcs_vertices_out;

   info->tcs.instances = nir->info.tess.tcs_vertices_out;
   info->tcs.include_primitive_id =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_PRIMITIVE_ID);
   info->vue.dispatch_mode = INTEL_DISPATCH_MODE_TCS_MULTI_PATCH;

   /* We don't use push inputs for TCS. */
   info->vue.urb_read_length = 0;
}

/* TODO: this is copied in two places right now. probably dedup it? */
static const uint32_t
   gl_prim_to_hw_prim[MESA_PRIM_TRIANGLE_STRIP_ADJACENCY + 1] = {
      [MESA_PRIM_POINTS] = _3DPRIM_POINTLIST,
      [MESA_PRIM_LINES] = _3DPRIM_LINELIST,
      [MESA_PRIM_LINE_LOOP] = _3DPRIM_LINELOOP,
      [MESA_PRIM_LINE_STRIP] = _3DPRIM_LINESTRIP,
      [MESA_PRIM_TRIANGLES] = _3DPRIM_TRILIST,
      [MESA_PRIM_TRIANGLE_STRIP] = _3DPRIM_TRISTRIP,
      [MESA_PRIM_TRIANGLE_FAN] = _3DPRIM_TRIFAN,
      [MESA_PRIM_QUADS] = _3DPRIM_QUADLIST,
      [MESA_PRIM_QUAD_STRIP] = _3DPRIM_QUADSTRIP,
      [MESA_PRIM_POLYGON] = _3DPRIM_POLYGON,
      [MESA_PRIM_LINES_ADJACENCY] = _3DPRIM_LINELIST_ADJ,
      [MESA_PRIM_LINE_STRIP_ADJACENCY] = _3DPRIM_LINESTRIP_ADJ,
      [MESA_PRIM_TRIANGLES_ADJACENCY] = _3DPRIM_TRILIST_ADJ,
      [MESA_PRIM_TRIANGLE_STRIP_ADJACENCY] = _3DPRIM_TRISTRIP_ADJ,
   };

static void
populate_gs_prog_data(nir_shader *nir,
                      const struct brw_gs_prog_key *key,
                      struct intel_shader_info *info)
{
   unsigned output_vertex_size_bytes = info->vue.vue_map.num_slots * 16;
   assert(output_vertex_size_bytes <= GFX7_MAX_GS_OUTPUT_VERTEX_SIZE_BYTES);

   info->gs.output_vertex_size_hwords =
      align(output_vertex_size_bytes, 32) / 32;

   info->gs.include_primitive_id =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_PRIMITIVE_ID);
   info->gs.invocations = nir->info.gs.invocations;

   uint32_t control_data_bits_per_vertex;

   if (nir->info.gs.output_primitive == MESA_PRIM_POINTS) {
      info->gs.control_data_format = GFX7_GS_CONTROL_DATA_FORMAT_GSCTL_SID;

      if (nir->info.gs.active_stream_mask != (1 << 0)) {
         control_data_bits_per_vertex = 2;
      } else {
         control_data_bits_per_vertex = 0;
      }
   } else {
      info->gs.control_data_format = GFX7_GS_CONTROL_DATA_FORMAT_GSCTL_CUT;
      control_data_bits_per_vertex = nir->info.gs.uses_end_primitive ? 1 : 0;
   }

   uint32_t control_data_header_size_bits =
      nir->info.gs.vertices_out * control_data_bits_per_vertex;

   info->gs.control_data_header_size_hwords =
      align(control_data_header_size_bits, 256) / 256;

   info->gs.output_topology =
      gl_prim_to_hw_prim[nir->info.gs.output_primitive];

   info->gs.vertices_in = nir->info.gs.vertices_in;
}

void
jay_populate_prog_data(const struct intel_device_info *devinfo,
                       nir_shader *nir,
                       struct intel_shader_info *info,
                       union brw_any_prog_key *key,
                       struct jay_fs_perprim_data *fs_perprim)
{
   info->is_jay = true;

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      populate_vs_prog_data(nir, devinfo, info);
   } else if (nir->info.stage == MESA_SHADER_TESS_CTRL) {
      populate_tcs_prog_data(nir, &key->tcs, info);
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      int32_t *per_primitive_offsets = fs_perprim->per_primitive_offsets;
      memset(per_primitive_offsets, -1, VARYING_SLOT_MAX * sizeof(int));

      populate_fs_prog_data(nir, devinfo, &key->fs,
                            fs_perprim->mue, per_primitive_offsets, info);
   } else if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      populate_gs_prog_data(nir, &key->gs, info);
   }

   if (nir->info.stage == MESA_SHADER_VERTEX ||
       nir->info.stage == MESA_SHADER_TESS_EVAL ||
       nir->info.stage == MESA_SHADER_GEOMETRY ||
       nir->info.stage == MESA_SHADER_MESH) {

      uint32_t clip_mask = BITFIELD_MASK(nir->info.clip_distance_array_size);
      uint32_t cull_mask = BITFIELD_RANGE(nir->info.clip_distance_array_size,
                                          nir->info.cull_distance_array_size);

      if (nir->info.stage == MESA_SHADER_MESH) {
         info->mesh.clip_distance_mask = clip_mask;
         info->mesh.cull_distance_mask = cull_mask;
      } else {
         info->vue.clip_distance_mask = clip_mask;
         info->vue.cull_distance_mask = cull_mask;
      }
   }
}
