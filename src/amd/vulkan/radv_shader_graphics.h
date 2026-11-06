/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_SHADER_GRAPHICS_H
#define RADV_SHADER_GRAPHICS_H

#include "amdgfxregs.h"
#include "radv_shader.h"

struct radv_ia_multi_vgt_param_helpers {
   uint32_t base;
   bool partial_es_wave;
   bool ia_switch_on_eoi;
   bool partial_vs_wave;
};

static inline uint32_t
radv_conv_gl_prim_to_gs_out(unsigned gl_prim)
{
   switch (gl_prim) {
   case MESA_PRIM_POINTS:
      return V_028A6C_POINTLIST;
   case MESA_PRIM_LINES:
   case MESA_PRIM_LINE_STRIP:
   case MESA_PRIM_LINES_ADJACENCY:
      return V_028A6C_LINESTRIP;

   case MESA_PRIM_TRIANGLES:
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
   case MESA_PRIM_TRIANGLE_STRIP:
   case MESA_PRIM_QUADS:
      return V_028A6C_TRISTRIP;
   default:
      assert(0);
      return 0;
   }
}

static inline uint32_t
radv_conv_tess_prim_to_gs_out(enum tess_primitive_mode prim)
{
   switch (prim) {
   case TESS_PRIMITIVE_TRIANGLES:
   case TESS_PRIMITIVE_QUADS:
      return V_028A6C_TRISTRIP;
   case TESS_PRIMITIVE_ISOLINES:
      return V_028A6C_LINESTRIP;
   default:
      assert(0);
      return 0;
   }
}

static inline uint32_t
radv_conv_prim_to_gs_out(enum amd_gfx_level gfx_level, uint32_t topology, bool is_ngg)
{
   static_assert(V_028A6C_POINTLIST == V_030998_POINTLIST && V_028A6C_LINESTRIP == V_030998_LINESTRIP &&
                    V_028A6C_TRISTRIP == V_030998_TRISTRIP,
                 "Some VGT_GS_OUTPRIM_TYPE values don't match");

   switch (topology) {
   case V_008958_DI_PT_POINTLIST:
   case V_008958_DI_PT_PATCH:
      return V_028A6C_POINTLIST;
   case V_008958_DI_PT_LINELIST:
   case V_008958_DI_PT_LINESTRIP:
   case V_008958_DI_PT_LINELIST_ADJ:
   case V_008958_DI_PT_LINESTRIP_ADJ:
      return V_028A6C_LINESTRIP;
   case V_008958_DI_PT_TRILIST:
   case V_008958_DI_PT_TRISTRIP:
   case V_008958_DI_PT_TRIFAN:
   case V_008958_DI_PT_TRILIST_ADJ:
   case V_008958_DI_PT_TRISTRIP_ADJ:
      return V_028A6C_TRISTRIP;
   case V_008958_DI_PT_RECTLIST:
      return is_ngg ? (gfx_level >= GFX11 ? V_030998_RECT_2D : V_028A6C_RECTLIST) : V_028A6C_TRISTRIP;
   default:
      assert(0);
      return 0;
   }
}

static inline unsigned
radv_get_num_vertices_per_prim(enum amd_gfx_level gfx_level, const struct radv_graphics_state_key *gfx_state)
{
   if (gfx_state->ia.topology == V_008958_DI_PT_NONE) {
      /* When the topology is unknown, return the maximum number of vertices per primitives for VS.
       * This is used to lower NGG (the HW will ignore the extra bits for points/lines) and also to
       * enable NGG culling unconditionally (it will be disabled dynamically for points/lines).
       */
      return 3;
   } else {
      /* Need to add 1, because: V_028A6C_POINTLIST=0, V_028A6C_LINESTRIP=1, V_028A6C_TRISTRIP=2, etc. */
      return radv_conv_prim_to_gs_out(gfx_level, gfx_state->ia.topology, false) + 1;
   }
}

struct radv_ia_multi_vgt_param_helpers radv_compute_ia_multi_vgt_param(const struct radv_device *device,
                                                                       struct radv_shader *const *shaders);

static inline struct radv_shader *
radv_get_shader(struct radv_shader *const *shaders, mesa_shader_stage stage)
{
   if (stage == MESA_SHADER_VERTEX) {
      if (shaders[MESA_SHADER_VERTEX])
         return shaders[MESA_SHADER_VERTEX];
      if (shaders[MESA_SHADER_TESS_CTRL])
         return shaders[MESA_SHADER_TESS_CTRL];
      if (shaders[MESA_SHADER_GEOMETRY])
         return shaders[MESA_SHADER_GEOMETRY];
   } else if (stage == MESA_SHADER_TESS_EVAL) {
      if (!shaders[MESA_SHADER_TESS_CTRL])
         return NULL;
      if (shaders[MESA_SHADER_TESS_EVAL])
         return shaders[MESA_SHADER_TESS_EVAL];
      if (shaders[MESA_SHADER_GEOMETRY])
         return shaders[MESA_SHADER_GEOMETRY];
   }
   return shaders[stage];
}

struct radv_ps_epilog_state {
   VkFormat color_attachment_formats[MAX_RTS];
   uint32_t color_write_mask;
   uint32_t colors_written;

   uint8_t color_attachment_count;
   uint8_t color_attachment_mappings[MAX_RTS];
   uint8_t color_blend_enable;
   uint8_t need_src_alpha;

   bool mrt0_is_dual_src;
   bool has_depth_output;
   bool has_stencil_output;
   bool has_sample_mask_output;
   bool ignore_depth_output;
   bool ignore_stencil_output;
   bool lower_1bit_sample_mask_to_discard;
   bool alpha_to_coverage_via_mrtz;
   bool alpha_to_one;
};

void radv_init_ps_epilog_format_infos(struct radv_compiler_info *compiler_info);

struct radv_ps_epilog_key radv_generate_ps_epilog_key(const struct radv_compiler_info *compiler_info,
                                                      const struct radv_ps_epilog_state *state);

void radv_graphics_shaders_spirv_to_nir(const struct radv_compiler_info *compiler_info,
                                        struct radv_shader_stage *stages,
                                        bool is_internal);

void radv_graphics_shaders_compile(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                                   const struct radv_graphics_state_key *gfx_state, bool noop_fs,
                                   struct radv_shader_debug_info *debug, struct radv_shader_binary **binaries,
                                   struct radv_shader_debug_info *gs_copy_debug,
                                   struct radv_shader_binary **gs_copy_binary);

struct radv_vgt_shader_key {
   uint8_t tess : 1;
   uint8_t gs : 1;
   uint8_t mesh : 1;
   uint8_t ngg_passthrough : 1;
   uint8_t ngg : 1; /* gfx10+ */
   uint8_t ngg_wave_id_en : 1;
   uint8_t hs_wave32 : 1;
   uint8_t gs_wave32 : 1;
   uint8_t vs_wave32 : 1;
};

struct radv_vgt_shader_key radv_get_vgt_shader_key(const struct radv_device *device, struct radv_shader **shaders,
                                                   const struct radv_shader *gs_copy_shader);

bool radv_needs_null_export_workaround(const struct radv_device *device, const struct radv_shader *ps,
                                       unsigned custom_blend_mode);

unsigned radv_compute_spi_ps_input(enum amd_gfx_level gfx_level, const struct radv_graphics_state_key *gfx_state,
                                   const struct radv_shader_info *info);

void radv_get_tess_wg_info(const struct radv_compiler_info *compiler_info, const ac_nir_tess_io_info *io_info,
                           unsigned tcs_vertices_out, unsigned tcs_num_input_vertices, unsigned tcs_num_lds_inputs,
                           unsigned *num_patches_per_wg, unsigned *lds_size);

void radv_lower_ngg(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *ngg_stage,
                    const struct radv_graphics_state_key *gfx_state);

bool radv_consider_culling(const struct radv_compiler_info *compiler_info, struct nir_shader *nir,
                           uint64_t ps_inputs_read, unsigned num_vertices_per_primitive,
                           const struct radv_shader_info *info, const struct radv_graphics_state_key *gfx_state);

void radv_shader_combine_cfg_vs_tcs(const struct radv_device *device, const struct radv_shader *vs,
                                    const struct radv_shader *tcs, uint32_t *rsrc1_out, uint32_t *rsrc2_out);

void radv_shader_combine_cfg_vs_gs(const struct radv_device *device, const struct radv_shader *vs,
                                   const struct radv_shader *gs, uint32_t *rsrc1_out, uint32_t *rsrc2_out,
                                   uint32_t *rsrc4_out);

void radv_shader_combine_cfg_tes_gs(const struct radv_device *device, const struct radv_shader *tes,
                                    const struct radv_shader *gs, uint32_t *rsrc1_out, uint32_t *rsrc2_out,
                                    uint32_t *rsrc4_out);

void radv_precompute_registers_hw_gs(struct radv_device *device, const struct radv_shader_info *es_info,
                                     struct radv_shader *shader);

void radv_precompute_registers_hw_ngg(struct radv_device *device, struct radv_shader *shader);

void radv_precompute_registers_hw_vs(struct radv_device *device, struct radv_shader *shader);

void radv_precompute_registers_hw_hs(struct radv_device *device, struct radv_shader *shader);

void radv_precompute_registers_hw_ms(struct radv_device *device, struct radv_shader *shader);

void radv_precompute_registers_hw_fs(struct radv_device *device, struct radv_shader *shader);

void radv_precompute_vgt_ps_inputs(const struct radv_physical_device *pdev, struct radv_shader *shader);

struct radv_shader_part *radv_create_vs_prolog(struct radv_device *device, const struct radv_vs_prolog_key *key);

struct radv_shader_part *radv_create_ps_epilog(struct radv_device *device, const struct radv_ps_epilog_key *key,
                                               struct radv_shader_part_binary **binary_out);

#endif /* RADV_SHADER_GRAPHICS_H */
