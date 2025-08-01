/*
 * Copyright © 2014 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#ifndef IR3_SHADER_H_
#define IR3_SHADER_H_

#include <stdio.h>

#include "c11/threads.h"
#include "compiler/nir/nir.h"
#include "compiler/shader_enums.h"
#include "util/bitscan.h"
#include "util/disk_cache.h"

#include "ir3_compiler.h"

BEGINC;

#define dword_offsetof(type, name) DIV_ROUND_UP(offsetof(type, name), 4)
#define dword_sizeof(type)         DIV_ROUND_UP(sizeof(type), 4)

/**
 * Driver params for compute shaders.
 *
 * Note, driver param structs should be size aligned to vec4
 */
struct ir3_driver_params_cs {
   /* NOTE: gl_NumWorkGroups should be vec4 aligned because
    * glDispatchComputeIndirect() needs to load these from
    * the info->indirect buffer.  Keep that in mind when/if
    * adding any addition CS driver params.
    */
   uint32_t num_work_groups_x;
   uint32_t num_work_groups_y;
   uint32_t num_work_groups_z;
   uint32_t work_dim;
   uint32_t base_group_x;
   uint32_t base_group_y;
   uint32_t base_group_z;
   uint32_t subgroup_size;
   uint32_t local_group_size_x;
   uint32_t local_group_size_y;
   uint32_t local_group_size_z;
   uint32_t subgroup_id_shift;
   uint32_t workgroup_id_x;
   uint32_t workgroup_id_y;
   uint32_t workgroup_id_z;
   uint32_t __pad;
};
#define IR3_DP_CS(name) dword_offsetof(struct ir3_driver_params_cs, name)

/**
 * Driver params for vertex shaders.
 *
 * Note, driver param structs should be size aligned to vec4
 */
struct ir3_driver_params_vs {
   uint32_t draw_id;
   uint32_t vtxid_base;
   uint32_t instid_base;
   uint32_t vtxcnt_max;
   uint32_t is_indexed_draw;  /* Note: boolean, ie. 0 or ~0 */
   /* user-clip-plane components, up to 8x vec4's: */
   struct {
      uint32_t x;
      uint32_t y;
      uint32_t z;
      uint32_t w;
   } ucp[8];
   uint32_t __pad_37_39[3];
};
#define IR3_DP_VS(name) dword_offsetof(struct ir3_driver_params_vs, name)

/**
 * Driver params for TCS shaders.
 *
 * Note, driver param structs should be size aligned to vec4
 */
struct ir3_driver_params_tcs {
   uint32_t default_outer_level_x;
   uint32_t default_outer_level_y;
   uint32_t default_outer_level_z;
   uint32_t default_outer_level_w;
   uint32_t default_inner_level_x;
   uint32_t default_inner_level_y;
   uint32_t __pad_06_07[2];
};
#define IR3_DP_TCS(name) dword_offsetof(struct ir3_driver_params_tcs, name)

/**
 * Driver params for fragment shaders.
 *
 * Note, driver param structs should be size aligned to vec4
 */
struct ir3_driver_params_fs {
   uint32_t subgroup_size;
   uint32_t __pad_01_03[3];
   /* Dynamic params (that aren't known when compiling the shader) */
#define IR3_DP_FS_DYNAMIC dword_offsetof(struct ir3_driver_params_fs, frag_invocation_count)
   uint32_t frag_invocation_count;
   uint32_t __pad_05_07[3];
   uint32_t frag_size;
   uint32_t __pad_09;
   uint32_t frag_offset;
   uint32_t __pad_11_12[2];
};
#define IR3_DP_FS(name) dword_offsetof(struct ir3_driver_params_fs, name)

#define IR3_MAX_SHADER_BUFFERS  32
#define IR3_MAX_SHADER_IMAGES   32
#define IR3_MAX_SO_BUFFERS      4
#define IR3_MAX_SO_STREAMS      4
#define IR3_MAX_SO_OUTPUTS      128
#define IR3_MAX_UBO_PUSH_RANGES 32

/* mirrors SYSTEM_VALUE_BARYCENTRIC_ but starting from 0 */
enum ir3_bary {
   IJ_PERSP_PIXEL,
   IJ_PERSP_SAMPLE,
   IJ_PERSP_CENTROID,
   IJ_PERSP_CENTER_RHW,
   IJ_LINEAR_PIXEL,
   IJ_LINEAR_CENTROID,
   IJ_LINEAR_SAMPLE,
   IJ_COUNT,
};

/* Description of what wavesizes are allowed. */
enum ir3_wavesize_option {
   IR3_SINGLE_ONLY,
   IR3_SINGLE_OR_DOUBLE,
   IR3_DOUBLE_ONLY,
};

/**
 * Description of a lowered UBO.
 */
struct nir_def;

struct ir3_ubo_info {
   struct nir_def *global_base; /* For global loads, the base address */
   uint32_t block;         /* Which constant block */
   uint16_t bindless_base; /* For bindless, which base register is used */
   bool bindless;
   bool global;
};

/**
 * Description of a range of a lowered UBO access.
 *
 * Drivers should not assume that there are not multiple disjoint
 * lowered ranges of a single UBO.
 */
struct ir3_ubo_range {
   struct ir3_ubo_info ubo;
   uint32_t offset;     /* start offset to push in the const register file */
   uint32_t start, end; /* range of block that's actually used */
};

struct ir3_ubo_analysis_state {
   struct ir3_ubo_range range[IR3_MAX_UBO_PUSH_RANGES];
   uint32_t num_enabled;
   uint32_t size;
};

enum ir3_push_consts_type {
   IR3_PUSH_CONSTS_NONE,
   IR3_PUSH_CONSTS_PER_STAGE,
   IR3_PUSH_CONSTS_SHARED,
   IR3_PUSH_CONSTS_SHARED_PREAMBLE,
};

/* This represents an internal UBO filled out by the driver. There are a few
 * common UBOs that must be filled out identically by all drivers, for example
 * for shader linkage, but drivers can also add their own that they manage
 * themselves.
 */
struct ir3_driver_ubo {
   int32_t idx;
   uint32_t size;
};

enum ir3_const_alloc_type {
   /* Vulkan, push consts. */
   IR3_CONST_ALLOC_PUSH_CONSTS = 0,
   /* Vulkan, offsets required to calculate offsets of descriptors with dynamic
    * offsets.
    */
   IR3_CONST_ALLOC_DYN_DESCRIPTOR_OFFSET = 1,
   /* Vulkan, addresses of inline uniform buffers, to which we fallback when
    * their size is unknown.
    */
   IR3_CONST_ALLOC_INLINE_UNIFORM_ADDRS = 2,
   /* Common, stage-specific params uploaded by the driver/HW. */
   IR3_CONST_ALLOC_DRIVER_PARAMS = 3,
   /* Common, UBOs lowered to consts. */
   IR3_CONST_ALLOC_UBO_RANGES = 4,
   /* Common, consts produced by a preamble to be used in a main shader. */
   IR3_CONST_ALLOC_PREAMBLE = 5,
   /* Vulkan, inline uniforms loaded into consts in the preamble.*/
   IR3_CONST_ALLOC_GLOBAL = 6,
   /* OpenGL, pre-a6xx; pointers to UBOs */
   IR3_CONST_ALLOC_UBO_PTRS = 7,
   /* OpenGL, a5xx only; needed to calculate pixel offset, but only
    * for images that have image_{load,store,size,atomic*} intrinsics.
    */
   IR3_CONST_ALLOC_IMAGE_DIMS = 8,
   /* OpenGL, TFBO addresses only for vs on a3xx/a4xx */
   IR3_CONST_ALLOC_TFBO = 9,
   /* Common, stage-dependent primitive params:
    *  vs, gs: uvec4(primitive_stride, vertex_stride, 0, 0)
    *  hs, ds: uvec4(primitive_stride, vertex_stride,
    *                patch_stride, patch_vertices_in)
    *          uvec4(tess_param_base, tess_factor_base)
    */
   IR3_CONST_ALLOC_PRIMITIVE_PARAM = 10,
   /* Common, mapping from varying location to offset. */
   IR3_CONST_ALLOC_PRIMITIVE_MAP = 11,
   IR3_CONST_ALLOC_MAX = 12,
};

struct ir3_const_allocation {
   uint32_t offset_vec4;
   uint32_t size_vec4;

   uint32_t reserved_size_vec4;
   uint32_t reserved_align_vec4;
};

struct ir3_const_allocations {
   struct ir3_const_allocation consts[IR3_CONST_ALLOC_MAX];
   uint32_t max_const_offset_vec4;
   uint32_t reserved_vec4;
};

static inline bool
ir3_const_can_upload(const struct ir3_const_allocations *const_alloc,
                     enum ir3_const_alloc_type type,
                     uint32_t shader_const_size_vec4)
{
   return const_alloc->consts[type].size_vec4 > 0 &&
          const_alloc->consts[type].offset_vec4 < shader_const_size_vec4;
}

struct ir3_const_image_dims {
   uint32_t mask;  /* bitmask of images that have image_store */
   uint32_t count; /* number of consts allocated */
   /* three const allocated per image which has image_store:
      *  + cpp         (bytes per pixel)
      *  + pitch       (y pitch)
      *  + array_pitch (z pitch)
      */
   uint32_t off[IR3_MAX_SHADER_IMAGES];
};

struct ir3_imm_const_state {
   unsigned size;
   unsigned count;
   uint32_t *values;
};

/**
 * Describes the layout of shader consts in the const register file
 * and additional info about individual allocations.
 *
 * Each consts section is aligned to vec4. Note that pointer
 * size (ubo, etc) changes depending on generation.
 *
 * The consts allocation flow is as follows:
 * 1) Turnip/Freedreno allocates consts required by corresponding API,
 *    e.g. push const, inline uniforms, etc. Then passes ir3_const_allocations
 *    into IR3.
 * 2) ir3_setup_const_state allocates consts with non-negotiable size.
 * 3) IR3 lowerings afterwards allocate from the free space left.
 *
 * Note UBO size in bytes should be aligned to vec4
 */
struct ir3_const_state {
   unsigned num_ubos;
   unsigned num_app_ubos;      /* # of UBOs not including driver UBOs */
   unsigned num_driver_params; /* scalar */

   struct ir3_driver_ubo consts_ubo;
   struct ir3_driver_ubo driver_params_ubo;
   struct ir3_driver_ubo primitive_map_ubo, primitive_param_ubo;

   struct ir3_const_allocations allocs;

   struct ir3_const_image_dims image_dims;

   /* State of ubo access lowered to push consts: */
   struct ir3_ubo_analysis_state ubo_state;
   enum ir3_push_consts_type push_consts_type;
};

/**
 * A single output for vertex transform feedback.
 */
struct ir3_stream_output {
   unsigned register_index  : 6;  /**< 0 to 63 (OUT index) */
   unsigned start_component : 2;  /** 0 to 3 */
   unsigned num_components  : 3;  /** 1 to 4 */
   unsigned output_buffer   : 3;  /**< 0 to PIPE_MAX_SO_BUFFERS */
   unsigned dst_offset      : 16; /**< offset into the buffer in dwords */
   unsigned stream          : 2;  /**< 0 to 3 */
};

/**
 * Stream output for vertex transform feedback.
 */
struct ir3_stream_output_info {
   unsigned num_outputs;
   /** stride for an entire vertex for each buffer in dwords */
   uint16_t stride[IR3_MAX_SO_BUFFERS];

   /* These correspond to the VPC_SO_STREAM_CNTL fields */
   uint8_t streams_written;
   uint8_t buffer_to_stream[IR3_MAX_SO_BUFFERS];

   /**
    * Array of stream outputs, in the order they are to be written in.
    * Selected components are tightly packed into the output buffer.
    */
   struct ir3_stream_output output[IR3_MAX_SO_OUTPUTS];
};

/**
 * Starting from a4xx, HW supports pre-dispatching texture sampling
 * instructions prior to scheduling a shader stage, when the
 * coordinate maps exactly to an output of the previous stage.
 */

/**
 * There is a limit in the number of pre-dispatches allowed for any
 * given stage.
 */
#define IR3_MAX_SAMPLER_PREFETCH 4

/**
 * This is the output stream value for 'cmd', as used by blob. It may
 * encode the return type (in 3 bits) but it hasn't been verified yet.
 */
#define IR3_SAMPLER_PREFETCH_CMD          0x4
#define IR3_SAMPLER_BINDLESS_PREFETCH_CMD 0x6

/**
 * Stream output for texture sampling pre-dispatches.
 */
struct ir3_sampler_prefetch {
   uint8_t src;
   bool bindless;
   uint8_t samp_id;
   uint8_t tex_id;
   uint16_t samp_bindless_id;
   uint16_t tex_bindless_id;
   uint8_t dst;
   uint8_t wrmask;
   uint8_t half_precision;
   opc_t tex_opc;
};

/* Configuration key used to identify a shader variant.. different
 * shader variants can be used to implement features not supported
 * in hw (two sided color), binning-pass vertex shader, etc.
 *
 * When adding to this struct, please update ir3_shader_variant()'s debug
 * output.
 */
struct ir3_shader_key {
   union {
      struct {
         /*
          * Combined Vertex/Fragment shader parameters:
          */
         unsigned ucp_enables : 8;

         /* do we need to check {v,f}saturate_{s,t,r}? */
         unsigned has_per_samp : 1;

         /*
          * Fragment shader variant parameters:
          */
         unsigned msaa           : 1;
         /* used when shader needs to handle flat varyings (a4xx)
          * for front/back color inputs to frag shader:
          */
         unsigned rasterflat : 1;

         /* Indicates that this is a tessellation pipeline which requires a
          * whole different kind of vertex shader.  In case of
          * tessellation, this field also tells us which kind of output
          * topology the TES uses, which the TCS needs to know.
          */
#define IR3_TESS_NONE      0
#define IR3_TESS_QUADS     1
#define IR3_TESS_TRIANGLES 2
#define IR3_TESS_ISOLINES  3
         unsigned tessellation : 2;

         unsigned has_gs : 1;

         /* Whether stages after TCS read gl_PrimitiveID, used to determine
          * whether the TCS has to store it in the tess factor BO.
          */
         unsigned tcs_store_primid : 1;

         /* Whether this variant sticks to the "safe" maximum constlen,
          * which guarantees that the combined stages will never go over
          * the limit:
          */
         unsigned safe_constlen : 1;

         /* Whether driconf "dual_color_blend_by_location" workaround is
          * enabled
          */
         unsigned force_dual_color_blend : 1;
      };
      uint32_t global;
   };

   /* bitmask of ms shifts (a3xx) */
   uint32_t vsamples, fsamples;

   /* bitmask of samplers which need astc srgb workaround (a4xx): */
   uint16_t vastc_srgb, fastc_srgb;

   /* per-component (3-bit) swizzles of each sampler (a4xx tg4): */
   uint16_t vsampler_swizzles[16];
   uint16_t fsampler_swizzles[16];
};

static inline unsigned
ir3_tess_mode(enum tess_primitive_mode tess_mode)
{
   switch (tess_mode) {
   case TESS_PRIMITIVE_ISOLINES:
      return IR3_TESS_ISOLINES;
   case TESS_PRIMITIVE_TRIANGLES:
      return IR3_TESS_TRIANGLES;
   case TESS_PRIMITIVE_QUADS:
      return IR3_TESS_QUADS;
   default:
      UNREACHABLE("bad tessmode");
   }
}

static inline uint32_t
ir3_tess_factor_stride(unsigned patch_type)
{
   /* note: this matches the stride used by ir3's build_tessfactor_base */
   switch (patch_type) {
   case IR3_TESS_ISOLINES:
      return 12;
   case IR3_TESS_TRIANGLES:
      return 20;
   case IR3_TESS_QUADS:
      return 28;
   default:
      UNREACHABLE("bad tessmode");
   }
}

static inline bool
ir3_shader_key_equal(const struct ir3_shader_key *a,
                     const struct ir3_shader_key *b)
{
   /* slow-path if we need to check {v,f}saturate_{s,t,r} */
   if (a->has_per_samp || b->has_per_samp)
      return memcmp(a, b, sizeof(struct ir3_shader_key)) == 0;
   return a->global == b->global;
}

/* will the two keys produce different lowering for a fragment shader? */
static inline bool
ir3_shader_key_changes_fs(struct ir3_shader_key *key,
                          struct ir3_shader_key *last_key)
{
   if (last_key->has_per_samp || key->has_per_samp) {
      if ((last_key->fsamples != key->fsamples) ||
          (last_key->fastc_srgb != key->fastc_srgb) ||
          memcmp(last_key->fsampler_swizzles, key->fsampler_swizzles,
                sizeof(key->fsampler_swizzles)))
         return true;
   }

   if (last_key->rasterflat != key->rasterflat)
      return true;

   if (last_key->ucp_enables != key->ucp_enables)
      return true;

   if (last_key->safe_constlen != key->safe_constlen)
      return true;

   return false;
}

/* will the two keys produce different lowering for a vertex shader? */
static inline bool
ir3_shader_key_changes_vs(struct ir3_shader_key *key,
                          struct ir3_shader_key *last_key)
{
   if (last_key->has_per_samp || key->has_per_samp) {
      if ((last_key->vsamples != key->vsamples) ||
          (last_key->vastc_srgb != key->vastc_srgb) ||
          memcmp(last_key->vsampler_swizzles, key->vsampler_swizzles,
                sizeof(key->vsampler_swizzles)))
         return true;
   }

   if (last_key->ucp_enables != key->ucp_enables)
      return true;

   if (last_key->safe_constlen != key->safe_constlen)
      return true;

   return false;
}

/**
 * On a4xx+a5xx, Images share state with textures and SSBOs:
 *
 *   + Uses texture (cat5) state/instruction (isam) to read
 *   + Uses SSBO state and instructions (cat6) to write and for atomics
 *
 * Starting with a6xx, Images and SSBOs are basically the same thing,
 * with texture state and isam also used for SSBO reads.
 *
 * On top of that, gallium makes the SSBO (shader_buffers) state semi
 * sparse, with the first half of the state space used for atomic
 * counters lowered to atomic buffers.  We could ignore this, but I
 * don't think we could *really* handle the case of a single shader
 * that used the max # of textures + images + SSBOs.  And once we are
 * offsetting images by num_ssbos (or visa versa) to map them into
 * the same hardware state, the hardware state has become coupled to
 * the shader state, so at this point we might as well just use a
 * mapping table to remap things from image/SSBO idx to hw idx.
 *
 * To make things less (more?) confusing, for the hw "SSBO" state
 * (since it is really both SSBO and Image) I'll use the name "UAV"
 */
struct ir3_ibo_mapping {
#define UAV_INVALID 0xff
   /* Maps logical SSBO state to hw tex state: */
   uint8_t ssbo_to_tex[IR3_MAX_SHADER_BUFFERS];

   /* Maps logical Image state to hw tex state: */
   uint8_t image_to_tex[IR3_MAX_SHADER_IMAGES];

   /* Maps hw state back to logical SSBO or Image state:
    *
    * note UAV_SSBO ORd into values to indicate that the
    * hw slot is used for SSBO state vs Image state.
    */
#define UAV_SSBO 0x80
   uint8_t tex_to_image[32];

   /* including real textures */
   uint8_t num_tex;
   /* the number of real textures, ie. image/ssbo start here */
   uint8_t tex_base;
};

struct ir3_disasm_info {
   bool write_disasm;
   char *nir;
   char *disasm;
};

/* Represents half register in regid */
#define HALF_REG_ID 0x100

/* Options for common NIR optimization passes done in ir3. This is used for both
 * finalize and post-finalize (where it has to be in the shader).
 */
struct ir3_shader_nir_options {
   /* For the modes specified, accesses are assumed to be bounds-checked as
    * defined by VK_EXT_robustness2 and optimizations may have to be more
    * conservative.
    */
   nir_variable_mode robust_modes;
};

struct ir3_shader_options {
   /* What API-visible wavesizes are allowed. Even if only double wavesize is
    * allowed, we may still use the smaller wavesize "under the hood" and the
    * application simply sees the upper half as always disabled.
    */
   enum ir3_wavesize_option api_wavesize;
   /* What wavesizes we're allowed to actually use. If the API wavesize is
    * single-only, then this must be single-only too.
    */
   enum ir3_wavesize_option real_wavesize;
   enum ir3_push_consts_type push_consts_type;

   uint32_t push_consts_base;
   uint32_t push_consts_dwords;

   /* Some const allocations are required at API level. */
   struct ir3_const_allocations const_allocs;

   struct ir3_shader_nir_options nir_options;

   /* Whether FRAG_RESULT_DATAi slots may be dynamically remapped by the driver.
    * If true, ir3 will assume it cannot statically use the value of such slots
    * anywhere (e.g., as the target of alias.rt).
    */
   bool fragdata_dynamic_remap;
};

struct ir3_shader_output {
   uint8_t slot;
   uint8_t regid;
   uint8_t view;
   uint8_t aliased_components : 4;
   bool half : 1;
};

/**
 * Shader variant which contains the actual hw shader instructions,
 * and necessary info for shader state setup.
 */
struct ir3_shader_variant {
   struct fd_bo *bo;

   /* variant id (for debug) */
   uint32_t id;

   /* id of the shader the variant came from (for debug) */
   uint32_t shader_id;

   struct ir3_shader_key key;

   /* vertex shaders can have an extra version for hwbinning pass,
    * which is pointed to by so->binning:
    */
   bool binning_pass;
   //	union {
   struct ir3_shader_variant *binning;
   struct ir3_shader_variant *nonbinning;
   //	};

   struct ir3 *ir; /* freed after assembling machine instructions */

   /* shader variants form a linked list: */
   struct ir3_shader_variant *next;

   /* replicated here to avoid passing extra ptrs everywhere: */
   gl_shader_stage type;
   struct ir3_compiler *compiler;

   char *name;

   /* variant's copy of nir->constant_data (since we don't track the NIR in
    * the variant, and shader->nir is before the opt pass).  Moves to v->bin
    * after assembly.
    */
   void *constant_data;

   struct ir3_disasm_info disasm_info;

   /*
    * Below here is serialized when written to disk cache:
    */

   /* The actual binary shader instructions, size given by info.sizedwords: */
   uint32_t *bin;

   struct ir3_const_state *const_state;

   /* Immediate values that will be lowered to const registers. Before a7xx,
    * this will be uploaded together with the const_state. From a7xx on (where
    * load_shader_consts_via_preamble is true), this will be lowered to const
    * stores in the preamble.
    */
   struct ir3_imm_const_state imm_state;

   /*
    * The following macros are used by the shader disk cache save/
    * restore paths to serialize/deserialize the variant.  Any
    * pointers that require special handling in store_variant()
    * and retrieve_variant() should go above here.
    */
#define VARIANT_CACHE_START  offsetof(struct ir3_shader_variant, info)
#define VARIANT_CACHE_PTR(v) (((char *)v) + VARIANT_CACHE_START)
#define VARIANT_CACHE_SIZE                                                     \
   (sizeof(struct ir3_shader_variant) - VARIANT_CACHE_START)

   struct ir3_info info;

   char sha1_str[SHA1_DIGEST_STRING_LENGTH];

   struct ir3_shader_options shader_options;

   uint32_t constant_data_size;

   /* Levels of nesting of flow control:
    */
   unsigned branchstack;

   unsigned loops;

   /* the instructions length is in units of instruction groups
    * (4 instructions for a3xx, 16 instructions for a4xx.. each
    * instruction is 2 dwords):
    */
   unsigned instrlen;

   /* the constants length is in units of vec4's, and is the sum of
    * the uniforms and the built-in compiler constants
    */
   unsigned constlen;

   /* The private memory size in bytes per fiber */
   unsigned pvtmem_size;
   /* Whether we should use the new per-wave layout rather than per-fiber. */
   bool pvtmem_per_wave;

   /* Whether multi-position output is enabled. */
   bool multi_pos_output;

   /* Whether dual-source blending is enabled. */
   bool dual_src_blend;

   /* Whether early preamble is enabled. */
   bool early_preamble;

   /* Size in bytes of required shared memory */
   unsigned shared_size;

   /* About Linkage:
    *   + Let the frag shader determine the position/compmask for the
    *     varyings, since it is the place where we know if the varying
    *     is actually used, and if so, which components are used.  So
    *     what the hw calls "outloc" is taken from the "inloc" of the
    *     frag shader.
    *   + From the vert shader, we only need the output regid
    */

   bool frag_face, color0_mrt;
   uint8_t fragcoord_compmask;

   /* NOTE: for input/outputs, slot is:
    *   gl_vert_attrib  - for VS inputs
    *   gl_varying_slot - for VS output / FS input
    *   gl_frag_result  - for FS output
    */

   /* varyings/outputs: */
   unsigned outputs_count;
   struct ir3_shader_output outputs[32 + 2]; /* +POSITION +PSIZE */
   bool writes_pos, writes_smask, writes_psize, writes_viewport, writes_stencilref;
   bool writes_shading_rate;

   /* Size in dwords of all outputs for VS, size of entire patch for HS. */
   uint32_t output_size;

   /* Expected size of incoming output_loc for HS, DS, and GS */
   uint32_t input_size;

   /* Map from location to offset in per-primitive storage. In dwords for
    * HS, where varyings are read in the next stage via ldg with a dword
    * offset, and in bytes for all other stages.
    * +POSITION, +PSIZE, ... - see shader_io_get_unique_index
    */
   unsigned output_loc[13 + 32];

   /* attributes (VS) / varyings (FS):
    * Note that sysval's should come *after* normal inputs.
    */
   unsigned inputs_count;
   struct {
      uint8_t slot;
      uint8_t regid;
      uint8_t compmask;
      /* location of input (ie. offset passed to bary.f, etc).  This
       * matches the SP_VS_VPC_DST_REG.OUTLOCn value (a3xx and a4xx
       * have the OUTLOCn value offset by 8, presumably to account
       * for gl_Position/gl_PointSize)
       */
      uint8_t inloc;
      /* vertex shader specific: */
      bool sysval : 1; /* slot is a gl_system_value */
      /* fragment shader specific: */
      bool bary       : 1; /* fetched varying (vs one loaded into reg) */
      bool rasterflat : 1; /* special handling for emit->rasterflat */
      bool half       : 1;
      bool flat       : 1;
   } inputs[32 + 2]; /* +POSITION +FACE */
   bool reads_primid;
   bool reads_shading_rate;
   bool reads_smask;

   /* sum of input components (scalar).  For frag shaders, it only counts
    * the varying inputs:
    */
   unsigned total_in;

   /* sum of sysval input components (scalar). */
   unsigned sysval_in;

   /* For frag shaders, the total number of inputs (not scalar,
    * ie. SP_VS_PARAM_REG.TOTALVSOUTVAR)
    */
   unsigned varying_in;

   /* Remapping table to map Image and SSBO to hw state: */
   struct ir3_ibo_mapping image_mapping;

   /* number of samplers/textures (which are currently 1:1): */
   int num_samp;

   /* is there an implicit sampler to read framebuffer (FS only).. if
    * so the sampler-idx is 'num_samp - 1' (ie. it is appended after
    * the last "real" texture)
    */
   bool fb_read;

   /* do we have one or more SSBO instructions: */
   bool has_ssbo;

   /* Which bindless resources are used, for filling out sp_xs_config */
   bool bindless_tex;
   bool bindless_samp;
   bool bindless_ibo;
   bool bindless_ubo;

   /* do we need derivatives: */
   bool need_pixlod;

   bool need_full_quad;

   /* do we need VS driver params? */
   bool need_driver_params;

   /* do we have image write, etc (which prevents early-z): */
   bool no_earlyz;

   /* do we have kill, which also prevents early-z, but not necessarily
    * early-lrz (as long as lrz-write is disabled, which must be handled
    * outside of ir3.  Unlike other no_earlyz cases, kill doesn't have
    * side effects that prevent early-lrz discard.
    */
   bool has_kill;

   /* Whether the shader should run at sample rate (set by
    * info->fs.uses_sample_shading, which is set when using a variable that
    * implicitly enables it, or glMinSampleShading() or
    * VkPipelineMultisampleStateCreateInfo->sampleShadingEnable forcing it.
    */
   bool sample_shading;

   bool post_depth_coverage;

   bool empty;
   /* Doesn't have side-effects, no kill, no D/S write, etc. */
   bool writes_only_color;

   /* Are we using split or merged register file? */
   bool mergedregs;

   uint8_t clip_mask, cull_mask;

   /* for astc srgb workaround, the number/base of additional
    * alpha tex states we need, and index of original tex states
    */
   struct {
      unsigned base, count;
      unsigned orig_idx[16];
   } astc_srgb;

   /* for tg4 workaround, the number/base of additional
    * unswizzled tex states we need, and index of original tex states
    */
   struct {
      unsigned base, count;
      unsigned orig_idx[16];
   } tg4;

   /* texture sampler pre-dispatches */
   uint32_t num_sampler_prefetch;
   struct ir3_sampler_prefetch sampler_prefetch[IR3_MAX_SAMPLER_PREFETCH];
   enum ir3_bary prefetch_bary_type;

   /* If true, the last use of helper invocations is the texture prefetch and
    * they should be disabled for the actual shader. Equivalent to adding
    * (eq)nop at the beginning of the shader.
    */
   bool prefetch_end_of_quad;

   uint16_t local_size[3];
   bool local_size_variable;

   /* Important for compute shader to determine max reg footprint */
   bool has_barrier;

   /* The offset where images start in the UAV array. */
   unsigned num_ssbos;

   /* The total number of SSBOs and images, i.e. the number of hardware UAVs. */
   unsigned num_uavs;

   union {
      struct {
         enum tess_primitive_mode primitive_mode;

         /** The number of vertices in the TCS output patch. */
         uint8_t tcs_vertices_out;
         enum gl_tess_spacing spacing:2; /*gl_tess_spacing*/

         /** Is the vertex order counterclockwise? */
         bool ccw:1;
         bool point_mode:1;
      } tess;
      struct {
         /** The output primitive type */
         uint16_t output_primitive;

         /** The maximum number of vertices the geometry shader might write. */
         uint16_t vertices_out;

         /** 1 .. MAX_GEOMETRY_SHADER_INVOCATIONS */
         uint8_t invocations;

         /** The number of vertices received per input primitive (max. 6) */
         uint8_t vertices_in:3;
      } gs;
      struct {
         bool early_fragment_tests : 1;
         bool color_is_dual_source : 1;
         bool uses_fbfetch_output  : 1;
         bool fbfetch_coherent     : 1;
         enum gl_frag_depth_layout depth_layout;
      } fs;
      struct {
         unsigned req_local_mem;
         bool force_linear_dispatch;
         uint32_t local_invocation_id;
         uint32_t work_group_id;
      } cs;
   };

   uint32_t vtxid_base;

   /* For when we don't have a shader, variant's copy of streamout state */
   struct ir3_stream_output_info stream_output;
};

static inline const char *
ir3_shader_stage(struct ir3_shader_variant *v)
{
   switch (v->type) {
   case MESA_SHADER_VERTEX:
      return v->binning_pass ? "BVERT" : "VERT";
   case MESA_SHADER_TESS_CTRL:
      return "TCS";
   case MESA_SHADER_TESS_EVAL:
      return "TES";
   case MESA_SHADER_GEOMETRY:
      return "GEOM";
   case MESA_SHADER_FRAGMENT:
      return "FRAG";
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_KERNEL:
      return "CL";
   default:
      UNREACHABLE("invalid type");
      return NULL;
   }
}

/* Currently we do not do binning for tess.  And for GS there is no
 * cross-stage VS+GS optimization, so the full VS+GS is used in
 * the binning pass.
 */
static inline bool
ir3_has_binning_vs(const struct ir3_shader_key *key)
{
   if (key->tessellation || key->has_gs)
      return false;
   return true;
}

/**
 * Represents a shader at the API level, before state-specific variants are
 * generated.
 */
struct ir3_shader {
   gl_shader_stage type;

   /* shader id (for debug): */
   uint32_t id;
   uint32_t variant_count;

   /* Set by freedreno after shader_state_create, so we can emit debug info
    * when recompiling a shader at draw time.
    */
   bool initial_variants_done;

   struct ir3_compiler *compiler;

   struct ir3_shader_options options;

   bool nir_finalized;
   struct nir_shader *nir;
   struct ir3_stream_output_info stream_output;

   /* per shader stage specific info: */
   union {
      /* for compute shaders: */
      struct {
         unsigned req_local_mem;
         bool force_linear_dispatch;
      } cs;
      /* For vertex shaders: */
      struct {
         /* If we need to generate a passthrough TCS, it will be a function of
          * (a) the VS and (b) the # of patch_vertices (max 32), so cache them
          * in the VS keyed by # of patch_vertices-1.
          */
         unsigned passthrough_tcs_compiled;
         struct ir3_shader *passthrough_tcs[32];
      } vs;
   };

   struct ir3_shader_variant *variants;
   mtx_t variants_lock;

   cache_key cache_key; /* shader disk-cache key */

   /* Bitmask of bits of the shader key used by this shader.  Used to avoid
    * recompiles for GL NOS that doesn't actually apply to the shader.
    */
   struct ir3_shader_key key_mask;
};

/**
 * In order to use the same cmdstream, in particular constlen setup and const
 * emit, for both binning and draw pass (a6xx+), the binning pass re-uses it's
 * corresponding draw pass shaders const_state.
 */
static inline const struct ir3_const_state *
ir3_const_state(const struct ir3_shader_variant *v)
{
   if (v->binning_pass)
      return v->nonbinning->const_state;
   return v->const_state;
}

static inline struct ir3_const_state *
ir3_const_state_mut(const struct ir3_shader_variant *v)
{
   assert(!v->binning_pass);
   return v->const_state;
}

static inline unsigned
ir3_max_const_compute(const struct ir3_shader_variant *v,
                      const struct ir3_compiler *compiler)
{
   unsigned lm_size = v->local_size_variable ? compiler->local_mem_size :
      v->cs.req_local_mem;

   /* The LB is divided between consts and local memory. LB is split into
    * wave_granularity banks, to make it possible for different ALUs to access
    * it at the same time, and consts are duplicated into each bank so that they
    * always take constant time to access while LM is spread across the banks.
    *
    * We cannot arbitrarily divide LB. Instead only certain configurations, as
    * defined by the CONSTANTRAMMODE register field, are allowed. Not sticking
    * with the right configuration can result in hangs when multiple compute
    * shaders are in flight. We have to limit the constlen so that we can pick a
    * configuration where there is enough space for LM.
    */
   unsigned lb_const_size =
      ((compiler->compute_lb_size - lm_size) / compiler->wave_granularity) /
      16 /* bytes per vec4 */;
   if (lb_const_size < compiler->max_const_compute) {
      const uint32_t lb_const_sizes[] = { 128, 192, 256, 512 };

      assert(lb_const_size >= lb_const_sizes[0]);
      for (unsigned i = 0; i < ARRAY_SIZE(lb_const_sizes) - 1; i++) {
         if (lb_const_size < lb_const_sizes[i + 1])
            return lb_const_sizes[i];
      }
      return lb_const_sizes[ARRAY_SIZE(lb_const_sizes) - 1];
   } else {
      return compiler->max_const_compute;
   }
}

static inline unsigned
_ir3_max_const(const struct ir3_shader_variant *v, bool safe_constlen)
{
   if (v->binning_pass) {
      return v->nonbinning->constlen;
   }

   const struct ir3_compiler *compiler = v->compiler;
   bool shared_consts_enable =
      ir3_const_state(v)->push_consts_type == IR3_PUSH_CONSTS_SHARED;

   /* Shared consts size for CS and FS matches with what's acutally used,
    * but the size of shared consts for geomtry stages doesn't.
    * So we use a hw quirk for geometry shared consts.
    */
   uint32_t shared_consts_size = shared_consts_enable ?
         compiler->shared_consts_size : 0;

   uint32_t shared_consts_size_geom = shared_consts_enable ?
         compiler->geom_shared_consts_size_quirk : 0;

   uint32_t safe_shared_consts_size = shared_consts_enable ?
      ALIGN_POT(MAX2(DIV_ROUND_UP(shared_consts_size_geom, 4),
                     DIV_ROUND_UP(shared_consts_size, 5)), 4) : 0;

   if ((v->type == MESA_SHADER_COMPUTE) ||
       (v->type == MESA_SHADER_KERNEL)) {
      return ir3_max_const_compute(v, compiler) - shared_consts_size;
   } else if (safe_constlen) {
      return compiler->max_const_safe - safe_shared_consts_size;
   } else if (v->type == MESA_SHADER_FRAGMENT) {
      return compiler->max_const_frag - shared_consts_size;
   } else {
      return compiler->max_const_geom - shared_consts_size_geom;
   }
}

/* Given a variant, calculate the maximum constlen it can have.
 */
static inline unsigned
ir3_max_const(const struct ir3_shader_variant *v)
{
   return _ir3_max_const(v, v->key.safe_constlen);
}

bool ir3_const_ensure_imm_size(struct ir3_shader_variant *v, unsigned size);
uint16_t ir3_const_imm_index_to_reg(const struct ir3_const_state *const_state,
                                    unsigned i);
uint16_t ir3_const_find_imm(struct ir3_shader_variant *v, uint32_t imm);
uint16_t ir3_const_add_imm(struct ir3_shader_variant *v, uint32_t imm);

static inline unsigned
ir3_const_reg(const struct ir3_const_state *const_state,
              enum ir3_const_alloc_type type,
              unsigned offset)
{
   unsigned n = const_state->allocs.consts[type].offset_vec4;
   assert(const_state->allocs.consts[type].size_vec4 != 0);
   return regid(n + offset / 4, offset % 4);
}

/* Return true if a variant may need to be recompiled due to exceeding the
 * maximum "safe" constlen.
 */
static inline bool
ir3_exceeds_safe_constlen(const struct ir3_shader_variant *v)
{
   return v->constlen > _ir3_max_const(v, true);
}

void *ir3_shader_assemble(struct ir3_shader_variant *v);
struct ir3_shader_variant *
ir3_shader_create_variant(struct ir3_shader *shader,
                          const struct ir3_shader_key *key,
                          bool keep_ir);
struct ir3_shader_variant *
ir3_shader_get_variant(struct ir3_shader *shader,
                       const struct ir3_shader_key *key, bool binning_pass,
                       bool keep_ir, bool *created);

struct ir3_shader *
ir3_shader_from_nir(struct ir3_compiler *compiler, nir_shader *nir,
                    const struct ir3_shader_options *options,
                    struct ir3_stream_output_info *stream_output);
uint32_t ir3_trim_constlen(const struct ir3_shader_variant **variants,
                           const struct ir3_compiler *compiler);
struct ir3_shader *
ir3_shader_passthrough_tcs(struct ir3_shader *vs, unsigned patch_vertices);
void ir3_shader_destroy(struct ir3_shader *shader);
void ir3_shader_disasm(struct ir3_shader_variant *so, uint32_t *bin, FILE *out);
uint64_t ir3_shader_outputs(const struct ir3_shader *so);

int ir3_glsl_type_size(const struct glsl_type *type, bool bindless);

void ir3_shader_get_subgroup_size(const struct ir3_compiler *compiler,
                                  const struct ir3_shader_options *options,
                                  gl_shader_stage stage,
                                  unsigned *subgroup_size,
                                  unsigned *max_subgroup_size);

/*
 * Helper/util:
 */

/* clears shader-key flags which don't apply to the given shader.
 */
static inline void
ir3_key_clear_unused(struct ir3_shader_key *key, struct ir3_shader *shader)
{
   uint32_t *key_bits = (uint32_t *)key;
   uint32_t *key_mask = (uint32_t *)&shader->key_mask;
   STATIC_ASSERT(sizeof(*key) % 4 == 0);
   for (unsigned i = 0; i < sizeof(*key) >> 2; i++)
      key_bits[i] &= key_mask[i];
}

static inline int
ir3_find_output(const struct ir3_shader_variant *so, gl_varying_slot slot)
{
   for (unsigned j = 0; j < so->outputs_count; j++)
      if (so->outputs[j].slot == slot)
         return j;

   /* it seems optional to have a OUT.BCOLOR[n] for each OUT.COLOR[n]
    * in the vertex shader.. but the fragment shader doesn't know this
    * so  it will always have both IN.COLOR[n] and IN.BCOLOR[n].  So
    * at link time if there is no matching OUT.BCOLOR[n], we must map
    * OUT.COLOR[n] to IN.BCOLOR[n].  And visa versa if there is only
    * a OUT.BCOLOR[n] but no matching OUT.COLOR[n]
    */
   if (slot == VARYING_SLOT_BFC0) {
      slot = VARYING_SLOT_COL0;
   } else if (slot == VARYING_SLOT_BFC1) {
      slot = VARYING_SLOT_COL1;
   } else if (slot == VARYING_SLOT_COL0) {
      slot = VARYING_SLOT_BFC0;
   } else if (slot == VARYING_SLOT_COL1) {
      slot = VARYING_SLOT_BFC1;
   } else {
      return -1;
   }

   for (unsigned j = 0; j < so->outputs_count; j++)
      if (so->outputs[j].slot == slot)
         return j;

   return -1;
}

static inline int
ir3_next_varying(const struct ir3_shader_variant *so, int i)
{
   assert(so->inputs_count <= (unsigned)INT_MAX);
   while (++i < (int)so->inputs_count)
      if (so->inputs[i].compmask && so->inputs[i].bary)
         break;
   return i;
}

static inline int
ir3_find_input(const struct ir3_shader_variant *so, gl_varying_slot slot)
{
   int j = -1;

   while (true) {
      j = ir3_next_varying(so, j);

      assert(so->inputs_count <= (unsigned)INT_MAX);
      if (j >= (int)so->inputs_count)
         return -1;

      if (so->inputs[j].slot == slot)
         return j;
   }
}

static inline unsigned
ir3_find_input_loc(const struct ir3_shader_variant *so, gl_varying_slot slot)
{
   int var = ir3_find_input(so, slot);
   return var == -1 ? 0xff : so->inputs[var].inloc;
}

struct ir3_shader_linkage {
   /* Maximum location either consumed by the fragment shader or produced by
    * the last geometry stage, i.e. the size required for each vertex in the
    * VPC in DWORD's.
    */
   uint8_t max_loc;

   /* Number of entries in var. */
   uint8_t cnt;

   /* Bitset of locations used, including ones which are only used by the FS.
    */
   uint32_t varmask[4];

   /* Map from VS output to location. */
   struct {
      uint8_t slot;
      uint8_t regid;
      uint8_t compmask;
      uint8_t loc;
   } var[32];

   /* location for fixed-function gl_PrimitiveID passthrough */
   uint8_t primid_loc;

   /* location for fixed-function gl_ViewIndex passthrough */
   uint8_t viewid_loc;

   /* location for combined clip/cull distance arrays */
   uint8_t clip0_loc, clip1_loc;
};

static inline void
ir3_link_add(struct ir3_shader_linkage *l, uint8_t slot, uint8_t regid_,
             uint8_t compmask, uint8_t loc)
{
   for (unsigned j = 0; j < util_last_bit(compmask); j++) {
      uint8_t comploc = loc + j;
      l->varmask[comploc / 32] |= 1 << (comploc % 32);
   }

   l->max_loc = MAX2(l->max_loc, loc + util_last_bit(compmask));

   if (regid_ != regid(63, 0)) {
      int i = l->cnt++;
      assert(i < ARRAY_SIZE(l->var));

      l->var[i].slot = slot;
      l->var[i].regid = regid_;
      l->var[i].compmask = compmask;
      l->var[i].loc = loc;
   }
}

static inline void
ir3_link_shaders(struct ir3_shader_linkage *l,
                 const struct ir3_shader_variant *vs,
                 const struct ir3_shader_variant *fs, bool pack_vs_out)
{
   /* On older platforms, varmask isn't programmed at all, and it appears
    * that the hardware generates a mask of used VPC locations using the VS
    * output map, and hangs if a FS bary instruction references a location
    * not in the list. This means that we need to have a dummy entry in the
    * VS out map for things like gl_PointCoord which aren't written by the
    * VS. Furthermore we can't use r63.x, so just pick a random register to
    * use if there is no VS output.
    */
   const unsigned default_regid = pack_vs_out ? regid(63, 0) : regid(0, 0);
   int j = -1, k;

   l->primid_loc = 0xff;
   l->viewid_loc = 0xff;
   l->clip0_loc = 0xff;
   l->clip1_loc = 0xff;

   while (l->cnt < ARRAY_SIZE(l->var)) {
      j = ir3_next_varying(fs, j);

      assert(fs->inputs_count <= (unsigned)INT_MAX);
      if (j >= (int)fs->inputs_count)
         break;

      if (fs->inputs[j].inloc >= fs->total_in)
         continue;

      k = ir3_find_output(vs, (gl_varying_slot)fs->inputs[j].slot);

      if (fs->inputs[j].slot == VARYING_SLOT_PRIMITIVE_ID) {
         l->primid_loc = fs->inputs[j].inloc;
      }

      if (fs->inputs[j].slot == VARYING_SLOT_VIEW_INDEX) {
         assert(k < 0);
         l->viewid_loc = fs->inputs[j].inloc;
      }

      if (fs->inputs[j].slot == VARYING_SLOT_CLIP_DIST0)
         l->clip0_loc = fs->inputs[j].inloc;

      if (fs->inputs[j].slot == VARYING_SLOT_CLIP_DIST1)
         l->clip1_loc = fs->inputs[j].inloc;

      ir3_link_add(l, fs->inputs[j].slot,
                   k >= 0 ? vs->outputs[k].regid : default_regid,
                   fs->inputs[j].compmask, fs->inputs[j].inloc);
   }
}

static inline uint32_t
ir3_get_output_regid(const struct ir3_shader_output *output)
{
   return output->regid | (output->half ? HALF_REG_ID : 0);
}

static inline uint32_t
ir3_find_output_regid(const struct ir3_shader_variant *so, unsigned slot)
{
   int output_idx = ir3_find_output(so, (gl_varying_slot)slot);

   if (output_idx < 0) {
      return INVALID_REG;
   }

   return ir3_get_output_regid(&so->outputs[output_idx]);
}

void print_raw(FILE *out, const BITSET_WORD *data, size_t size);

void ir3_link_stream_out(struct ir3_shader_linkage *l,
                         const struct ir3_shader_variant *v);

#define VARYING_SLOT_GS_HEADER_IR3       (VARYING_SLOT_MAX + 0)
#define VARYING_SLOT_GS_VERTEX_FLAGS_IR3 (VARYING_SLOT_MAX + 1)
#define VARYING_SLOT_TCS_HEADER_IR3      (VARYING_SLOT_MAX + 2)
#define VARYING_SLOT_REL_PATCH_ID_IR3    (VARYING_SLOT_MAX + 3)

static inline uint32_t
ir3_find_sysval_regid(const struct ir3_shader_variant *so, unsigned slot)
{
   if (!so)
      return regid(63, 0);
   for (unsigned j = 0; j < so->inputs_count; j++)
      if (so->inputs[j].sysval && (so->inputs[j].slot == slot))
         return so->inputs[j].regid;
   return regid(63, 0);
}

/* calculate register footprint in terms of half-regs (ie. one full
 * reg counts as two half-regs).
 */
static inline uint32_t
ir3_shader_halfregs(const struct ir3_shader_variant *v)
{
   return (2 * (v->info.max_reg + 1)) + (v->info.max_half_reg + 1);
}

static inline uint32_t
ir3_shader_num_uavs(const struct ir3_shader_variant *v)
{
   return v->num_uavs;
}

static inline uint32_t
ir3_shader_branchstack_hw(const struct ir3_shader_variant *v)
{
   /* Dummy shader */
   if (!v->compiler)
      return 0;

   if (v->compiler->gen < 5)
      return v->branchstack;

   return DIV_ROUND_UP(MIN2(v->branchstack, v->compiler->branchstack_size), 2);
}

ENDC;

#endif /* IR3_SHADER_H_ */
