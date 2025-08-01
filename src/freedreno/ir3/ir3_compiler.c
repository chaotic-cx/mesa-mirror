/*
 * Copyright © 2015 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "util/ralloc.h"

#include "freedreno_dev_info.h"

#include "ir3_compiler.h"

static const struct debug_named_value shader_debug_options[] = {
   /* clang-format off */
   {"vs",         IR3_DBG_SHADER_VS,  "Print shader disasm for vertex shaders"},
   {"tcs",        IR3_DBG_SHADER_TCS, "Print shader disasm for tess ctrl shaders"},
   {"tes",        IR3_DBG_SHADER_TES, "Print shader disasm for tess eval shaders"},
   {"gs",         IR3_DBG_SHADER_GS,  "Print shader disasm for geometry shaders"},
   {"fs",         IR3_DBG_SHADER_FS,  "Print shader disasm for fragment shaders"},
   {"cs",         IR3_DBG_SHADER_CS,  "Print shader disasm for compute shaders"},
   {"internal",   IR3_DBG_SHADER_INTERNAL, "Print shader disasm for internal shaders (normally not included in vs/fs/cs/etc)"},
   {"disasm",     IR3_DBG_DISASM,     "Dump NIR and adreno shader disassembly"},
   {"optmsgs",    IR3_DBG_OPTMSGS,    "Enable optimizer debug messages"},
   {"forces2en",  IR3_DBG_FORCES2EN,  "Force s2en mode for tex sampler instructions"},
   {"nouboopt",   IR3_DBG_NOUBOOPT,   "Disable lowering UBO to uniform"},
   {"nofp16",     IR3_DBG_NOFP16,     "Don't lower mediump to fp16"},
   {"nocache",    IR3_DBG_NOCACHE,    "Disable shader cache"},
   {"spillall",   IR3_DBG_SPILLALL,   "Spill as much as possible to test the spiller"},
   {"nopreamble", IR3_DBG_NOPREAMBLE, "Disable the preamble pass"},
   {"fullsync",   IR3_DBG_FULLSYNC,   "Add (sy) + (ss) after each cat5/cat6"},
   {"fullnop",    IR3_DBG_FULLNOP,    "Add nops before each instruction"},
   {"noearlypreamble", IR3_DBG_NOEARLYPREAMBLE, "Disable early preambles"},
   {"nodescprefetch", IR3_DBG_NODESCPREFETCH, "Disable descriptor prefetch optimization"},
   {"expandrpt",  IR3_DBG_EXPANDRPT,  "Expand rptN instructions"},
   {"noaliastex", IR3_DBG_NOALIASTEX, "Don't use alias.tex"},
   {"noaliasrt",  IR3_DBG_NOALIASRT,  "Don't use alias.rt"},
   {"asmroundtrip", IR3_DBG_ASM_ROUNDTRIP, "Disassemble, reassemble and compare every shader"},
#if MESA_DEBUG
   /* MESA_DEBUG-only options: */
   {"schedmsgs",  IR3_DBG_SCHEDMSGS,  "Enable scheduler debug messages"},
   {"ramsgs",     IR3_DBG_RAMSGS,     "Enable register-allocation debug messages"},
#endif
   DEBUG_NAMED_VALUE_END
   /* clang-format on */
};

DEBUG_GET_ONCE_FLAGS_OPTION(ir3_shader_debug, "IR3_SHADER_DEBUG",
                            shader_debug_options, 0)
DEBUG_GET_ONCE_OPTION(ir3_shader_override_path, "IR3_SHADER_OVERRIDE_PATH",
                      NULL)

enum ir3_shader_debug ir3_shader_debug = 0;
const char *ir3_shader_override_path = NULL;

void
ir3_compiler_destroy(struct ir3_compiler *compiler)
{
   disk_cache_destroy(compiler->disk_cache);
   ralloc_free(compiler);
}

static const nir_shader_compiler_options ir3_base_options = {
   .compact_arrays = true,
   .lower_fpow = true,
   .lower_scmp = true,
   .lower_flrp16 = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_ffract = true,
   .lower_fmod = true,
   .lower_fdiv = true,
   .lower_isign = true,
   .lower_ldexp = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_mul_high = true,
   .lower_mul_2x32_64 = true,
   .fuse_ffma16 = true,
   .fuse_ffma32 = true,
   .fuse_ffma64 = true,
   .vertex_id_zero_based = false,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_helper_invocation = true,
   .lower_bitfield_insert = true,
   .lower_bitfield_extract = true,
   .lower_bitfield_extract8 = true,
   .lower_bitfield_extract16 = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_unpack_half_2x16 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_pack_split = true,
   .lower_pack_64_4x16 = true,
   .lower_to_scalar = true,
   .has_imul24 = true,
   .has_icsel_eqz32 = true,
   .has_icsel_eqz16 = true,
   .has_fsub = true,
   .has_isub = true,
   .force_indirect_unrolling_sampler = true,
   .lower_uniforms_to_ubo = true,
   .max_unroll_iterations = 32,

   .lower_cs_local_index_to_id = true,
   .lower_wpos_pntc = true,

   .lower_hadd = true,
   .lower_hadd64 = true,
   .lower_fisnormal = true,

   .lower_int64_options = (nir_lower_int64_options)~0,
   .lower_doubles_options = (nir_lower_doubles_options)~0,

   .divergence_analysis_options = nir_divergence_uniform_load_tears,
   .scalarize_ddx = true,

   .per_view_unique_driver_locations = true,
   .compact_view_index = true,
};

struct ir3_compiler *
ir3_compiler_create(struct fd_device *dev, const struct fd_dev_id *dev_id,
                    const struct fd_dev_info *dev_info,
                    const struct ir3_compiler_options *options)
{
   struct ir3_compiler *compiler = rzalloc(NULL, struct ir3_compiler);

   ir3_shader_debug = debug_get_option_ir3_shader_debug();
   ir3_shader_override_path =
      __normal_user() ? debug_get_option_ir3_shader_override_path() : NULL;

   if (ir3_shader_override_path) {
      ir3_shader_debug |= IR3_DBG_NOCACHE;
   }

   compiler->dev = dev;
   compiler->dev_id = dev_id;
   compiler->gen = fd_dev_gen(dev_id);
   compiler->is_64bit = fd_dev_64b(dev_id);
   compiler->options = *options;

   /* TODO see if older GPU's were different here */
   compiler->branchstack_size = 64;
   compiler->wave_granularity = dev_info->wave_granularity;
   compiler->max_waves = dev_info->max_waves;

   compiler->max_variable_workgroup_size = 1024;

   compiler->local_mem_size = dev_info->cs_shared_mem_size;

   compiler->num_predicates = 1;
   compiler->bitops_can_write_predicates = false;
   compiler->has_branch_and_or = false;
   compiler->has_rpt_bary_f = false;
   compiler->has_alias_tex = false;
   compiler->delay_slots.alu_to_alu = 3;
   compiler->delay_slots.non_alu = 6;
   compiler->delay_slots.cat3_src2_read = 2;

   if (compiler->gen >= 6) {
      compiler->samgq_workaround = true;
      /* a6xx split the pipeline state into geometry and fragment state, in
       * order to let the VS run ahead of the FS. As a result there are now
       * separate const files for the the fragment shader and everything
       * else, and separate limits. There seems to be a shared limit, but
       * it's higher than the vert or frag limits.
       *
       * Also, according to the observation on a630/a650/a660, max_const_pipeline
       * has to be 512 when all geometry stages are present. Otherwise a gpu hang
       * happens. Accordingly maximum safe size for each stage should be under
       * (max_const_pipeline / 5 (stages)) with 4 vec4's alignment considered for
       * const files.
       *
       * Only when VS and FS stages are present, the limit is 640.
       *
       * TODO: The shared limit seems to be different on different models.
       */
      compiler->max_const_pipeline = 512;
      compiler->max_const_frag = 512;
      compiler->max_const_geom = 512;
      compiler->max_const_safe = 100;

      /* Compute shaders don't share a const file with the FS. Instead they
       * have their own file, which is smaller than the FS one. On a7xx the size
       * was doubled, although this doesn't work on X1-85.
       *
       * TODO: is this true on earlier gen's?
       */
      compiler->max_const_compute = compiler->gen >= 7 ? 512 : 256;

      if (dev_info->a6xx.is_a702) {
         /* No GS/tess, 128 per stage otherwise: */
         compiler->max_const_compute = 128;
         compiler->max_const_pipeline = 256;
         compiler->max_const_frag = 128;
         compiler->max_const_geom = 128;
         compiler->max_const_safe = 128;
      }

      /* TODO: implement clip+cull distances on earlier gen's */
      compiler->has_clip_cull = true;

      compiler->has_preamble = true;

      compiler->tess_use_shared = dev_info->a6xx.tess_use_shared;

      compiler->has_getfiberid = dev_info->a6xx.has_getfiberid;
      compiler->mov_half_shared_quirk = dev_info->a6xx.mov_half_shared_quirk;
      compiler->has_movs = dev_info->a6xx.has_movs;

      compiler->has_dp2acc = dev_info->a6xx.has_dp2acc;
      compiler->has_dp4acc = dev_info->a6xx.has_dp4acc;
      compiler->has_compliant_dp4acc = dev_info->a7xx.has_compliant_dp4acc;

      if (compiler->gen == 6 && options->shared_push_consts) {
         compiler->shared_consts_base_offset = 504;
         compiler->shared_consts_size = 8;
         compiler->geom_shared_consts_size_quirk = 16;
      } else {
         compiler->shared_consts_base_offset = -1;
         compiler->shared_consts_size = 0;
         compiler->geom_shared_consts_size_quirk = 0;
      }

      compiler->has_fs_tex_prefetch = dev_info->a6xx.has_fs_tex_prefetch;
      compiler->stsc_duplication_quirk = dev_info->a7xx.stsc_duplication_quirk;
      compiler->load_shader_consts_via_preamble = dev_info->a7xx.load_shader_consts_via_preamble;
      compiler->load_inline_uniforms_via_preamble_ldgk = dev_info->a7xx.load_inline_uniforms_via_preamble_ldgk;
      compiler->num_predicates = 4;
      compiler->bitops_can_write_predicates = true;
      compiler->has_branch_and_or = true;
      compiler->has_predication = true;
      compiler->predtf_nop_quirk = dev_info->a6xx.predtf_nop_quirk;
      compiler->prede_nop_quirk = dev_info->a6xx.prede_nop_quirk;
      compiler->has_scalar_alu = dev_info->a6xx.has_scalar_alu;
      compiler->has_isam_v = dev_info->a6xx.has_isam_v;
      compiler->has_ssbo_imm_offsets = dev_info->a6xx.has_ssbo_imm_offsets;
      compiler->fs_must_have_non_zero_constlen_quirk = dev_info->a7xx.fs_must_have_non_zero_constlen_quirk;
      compiler->has_early_preamble = dev_info->a6xx.has_early_preamble;
      compiler->has_rpt_bary_f = true;
      compiler->has_shfl = true;
      compiler->reading_shading_rate_requires_smask_quirk =
         dev_info->a7xx.reading_shading_rate_requires_smask_quirk;
      compiler->has_alias_rt = dev_info->a7xx.has_alias_rt;
      compiler->mergedregs = true;

      if (compiler->gen >= 7) {
         compiler->has_alias_tex = true;
         compiler->delay_slots.alu_to_alu = 2;
         compiler->delay_slots.non_alu = 5;
         compiler->delay_slots.cat3_src2_read = 1;
      }
   } else {
      compiler->max_const_pipeline = 512;
      compiler->max_const_geom = 512;
      compiler->max_const_frag = 512;
      compiler->max_const_compute = 512;

      /* Note: this will have to change if/when we support tess+GS on
       * earlier gen's.
       */
      compiler->max_const_safe = 256;

      compiler->has_scalar_alu = false;
      compiler->has_isam_v = false;
      compiler->has_ssbo_imm_offsets = false;
      compiler->has_early_preamble = false;
   }

   if (dev_info->compute_lb_size) {
      compiler->compute_lb_size = dev_info->compute_lb_size;
   } else {
      compiler->compute_lb_size =
         compiler->max_const_compute * 16 /* bytes/vec4 */ *
         compiler->wave_granularity + compiler->local_mem_size;
   }

   /* This is just a guess for a4xx. */
   compiler->pvtmem_per_fiber_align = compiler->gen >= 4 ? 512 : 128;
   /* TODO: implement private memory on earlier gen's */
   compiler->has_pvtmem = compiler->gen >= 5;

   compiler->has_isam_ssbo = compiler->gen >= 6;

   if (compiler->gen >= 6) {
      compiler->reg_size_vec4 = dev_info->a6xx.reg_size_vec4;
   } else if (compiler->gen >= 4) {
      /* On a4xx-a5xx, using r24.x and above requires using the smallest
       * threadsize.
       */
      compiler->reg_size_vec4 = 48;
   } else {
      /* TODO: confirm this */
      compiler->reg_size_vec4 = 96;
   }

   compiler->threadsize_base = dev_info->threadsize_base;

   if (compiler->gen >= 4) {
      /* need special handling for "flat" */
      compiler->flat_bypass = true;
      compiler->levels_add_one = false;
      compiler->unminify_coords = false;
      compiler->txf_ms_with_isaml = false;
      compiler->array_index_add_half = true;
      compiler->instr_align = 16;
      compiler->const_upload_unit = 4;
   } else {
      /* no special handling for "flat" */
      compiler->flat_bypass = false;
      compiler->levels_add_one = true;
      compiler->unminify_coords = true;
      compiler->txf_ms_with_isaml = true;
      compiler->array_index_add_half = false;
      compiler->instr_align = 4;
      compiler->const_upload_unit = 8;
   }

   compiler->bool_type = (compiler->gen >= 5) ? TYPE_U16 : TYPE_U32;
   compiler->has_shared_regfile = compiler->gen >= 5;
   compiler->has_bitwise_triops = compiler->gen >= 5;
   compiler->cat3_rel_offset_0_quirk = compiler->gen <= 5;

   /* The driver can't request this unless preambles are supported. */
   if (options->push_ubo_with_preamble)
      assert(compiler->has_preamble);

   /* Set up nir shader compiler options, using device-specific overrides of our base settings. */
   compiler->nir_options = ir3_base_options;
   compiler->nir_options.has_iadd3 = dev_info->a6xx.has_sad;

   if (compiler->gen >= 6) {
      compiler->nir_options.force_indirect_unrolling = nir_var_all,
      compiler->nir_options.lower_device_index_to_zero = true;
      compiler->nir_options.instance_id_includes_base_index = true;

      if (dev_info->a6xx.has_dp2acc || dev_info->a6xx.has_dp4acc) {
         compiler->nir_options.has_udot_4x8 =
            compiler->nir_options.has_udot_4x8_sat = true;
         compiler->nir_options.has_sudot_4x8 =
            compiler->nir_options.has_sudot_4x8_sat = true;
      }

      if (dev_info->a6xx.has_dp4acc && dev_info->a7xx.has_compliant_dp4acc) {
         compiler->nir_options.has_sdot_4x8 =
            compiler->nir_options.has_sdot_4x8_sat = true;
      }
   } else if (compiler->gen >= 3 && compiler->gen <= 5) {
      compiler->nir_options.vertex_id_zero_based = true;
   } else if (compiler->gen <= 2) {
      /* a2xx compiler doesn't handle indirect: */
      compiler->nir_options.force_indirect_unrolling = nir_var_all;
   }

   if (options->lower_base_vertex) {
      compiler->nir_options.lower_base_vertex = true;
   }

   /* 16-bit ALU op generation is mostly controlled by frontend compiler options, but
    * this core NIR option enables some optimizations of 16-bit operations.
    */
   if (compiler->gen >= 5 && !(ir3_shader_debug & IR3_DBG_NOFP16))
      compiler->nir_options.support_16bit_alu = true;

   compiler->nir_options.support_indirect_inputs =
      BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
      BITFIELD_BIT(MESA_SHADER_TESS_EVAL) | BITFIELD_BIT(MESA_SHADER_FRAGMENT);
   compiler->nir_options.support_indirect_outputs = (uint8_t)BITFIELD_MASK(PIPE_SHADER_TYPES);

   if (!options->disable_cache)
      ir3_disk_cache_init(compiler);

   return compiler;
}

const nir_shader_compiler_options *
ir3_get_compiler_options(struct ir3_compiler *compiler)
{
   return &compiler->nir_options;
}

const char *
ir3_shader_debug_as_string()
{
   return debug_dump_flags(shader_debug_options, ir3_shader_debug);
}
