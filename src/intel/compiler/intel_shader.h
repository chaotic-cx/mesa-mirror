/*
 * Copyright 2024 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifndef __OPENCL_VERSION__
#include <stdint.h>
#include "util/bitscan.h"
#endif

#include "compiler/shader_enums.h"
#include "util/bitscan.h"
#include "util/enum_operators.h"

#ifndef __OPENCL_VERSION__
#include "util/shader_stats.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** A tri-state value to track states that are potentially dynamic */
enum intel_sometimes {
   INTEL_NEVER = 0,
   INTEL_SOMETIMES,
   INTEL_ALWAYS
};

static inline enum intel_sometimes
intel_sometimes_invert(enum intel_sometimes x)
{
   return (enum intel_sometimes)((int)INTEL_ALWAYS - (int)x);
}

#define INTEL_TESS_CONFIG_INPUT_VERTICES_OFFSET   (0)
#define INTEL_TESS_CONFIG_INPUT_VERTICES_SIZE     (5)
#define INTEL_TESS_CONFIG_OUTPUT_VERTICES_OFFSET  (5)
#define INTEL_TESS_CONFIG_OUTPUT_VERTICES_SIZE    (5)
#define INTEL_TESS_CONFIG_BUILTINS_OFFSET         (10)
#define INTEL_TESS_CONFIG_BUILTINS_SIZE           (6)
#define INTEL_TESS_CONFIG_PER_VERTEX_SLOTS_OFFSET (16)
#define INTEL_TESS_CONFIG_PER_VERTEX_SLOTS_SIZE   (6)
#define INTEL_TESS_CONFIG_PER_PATCH_SLOTS_OFFSET  (22)
#define INTEL_TESS_CONFIG_PER_PATCH_SLOTS_SIZE    (6)

enum intel_tess_configs {
   /** Tessellation inputs vertices minus 1
    *
    * This field actually covers 5bits.
    */
   INTEL_TESS_CONFIG_INPUT_VERTICES  = BITFIELD_BIT(INTEL_TESS_CONFIG_INPUT_VERTICES_OFFSET),

   /** Tessellation outputs vertices minus 1
    *
    * This field actually covers 5bits.
    */
   INTEL_TESS_CONFIG_OUTPUT_VERTICES = BITFIELD_BIT(INTEL_TESS_CONFIG_OUTPUT_VERTICES_OFFSET),

   /** Tessellation builtins per-vertex offset
    *
    * This field actually covers 5bits.
    */
   INTEL_TESS_CONFIG_BUILTINS      = BITFIELD_BIT(INTEL_TESS_CONFIG_BUILTINS_OFFSET),

   /** Number of per vertex slots
    *
    * This field actually covers 6bits.
    */
   INTEL_TESS_PER_VERTEX_SLOTS     = BITFIELD_BIT(INTEL_TESS_CONFIG_PER_VERTEX_SLOTS_OFFSET),

   /** Number of per patch slots
    *
    * This field actually covers 6bits.
    */
   INTEL_TESS_PER_PATCH_SLOTS      = BITFIELD_BIT(INTEL_TESS_CONFIG_PER_PATCH_SLOTS_OFFSET),

   /** Tesselation primitive modes
    *
    * Only one of the following 3 bits should be set.
    */
   INTEL_TESS_CONFIG_QUADS           = BITFIELD_BIT(29),
   INTEL_TESS_CONFIG_TRIANGLES       = BITFIELD_BIT(30),
   INTEL_TESS_CONFIG_ISOLINES        = BITFIELD_BIT(31)
};

#define INTEL_FS_CONFIG_FIRST_VUE_SLOT_OFFSET     (19)
#define INTEL_FS_CONFIG_FIRST_VUE_SLOT_SIZE       (6)
#define INTEL_FS_CONFIG_PRIMITIVE_ID_INDEX_OFFSET (25)
#define INTEL_FS_CONFIG_PRIMITIVE_ID_INDEX_SIZE   (6)
#define INTEL_FS_CONFIG_PRIMITIVE_ID_INDEX_MESH   (32)

enum intel_fs_config {
   /** Must be set whenever any dynamic MSAA is used
    *
    * This flag mostly exists to let us assert that the driver understands
    * dynamic MSAA so we don't run into trouble with drivers that don't.
    */
   INTEL_FS_CONFIG_ENABLE_DYNAMIC = (1 << 0),

   /** True if the framebuffer is multisampled */
   INTEL_FS_CONFIG_MULTISAMPLE_FBO = (1 << 1),

   /** True if this shader has been dispatched per-sample */
   INTEL_FS_CONFIG_PERSAMPLE_DISPATCH = (1 << 2),

   /** True if this shader has been dispatched with alpha-to-coverage */
   INTEL_FS_CONFIG_ALPHA_TO_COVERAGE = (1 << 3),

   /** True if provoking vertex is last */
   INTEL_FS_CONFIG_PROVOKING_VERTEX_LAST = (1 << 4),

   /** True if we need to apply Wa_18019110168 remapping */
   INTEL_FS_CONFIG_PER_PRIMITIVE_REMAPPING = (1 << 5),

   /** True if conservative rasterization is enabled */
   INTEL_FS_CONFIG_CONSERVATIVE_RASTER = (1 << 6),

   /** True if this shader has been dispatched coarse
    *
    * This is intentionally chose to be bit 15 to correspond to the coarse bit
    * in the pixel interpolator messages.
    */
   INTEL_FS_CONFIG_COARSE_PI_MSG = (1 << 15),

   /** True if this shader has been dispatched coarse
    *
    * This is intentionally chose to be bit 18 to correspond to the coarse bit
    * in the render target messages.
    */
   INTEL_FS_CONFIG_COARSE_RT_WRITES = (1 << 18),

   /** First slot read in the VUE
    *
    * This is not a flag but a value that cover 6bits.
    */
   INTEL_FS_CONFIG_FIRST_VUE_SLOT = (1 << INTEL_FS_CONFIG_FIRST_VUE_SLOT_OFFSET),

   /** Index of the PrimitiveID attribute relative to the first read
    * attribute.
    *
    * This is not a flag but a value that cover 6bits. Value 32 means the
    * PrimitiveID is coming from the PerPrimitive block, written by the Mesh
    * shader.
    */
   INTEL_FS_CONFIG_PRIMITIVE_ID_INDEX = (1 << INTEL_FS_CONFIG_PRIMITIVE_ID_INDEX_OFFSET),
};
MESA_DEFINE_CPP_ENUM_BITFIELD_OPERATORS(intel_fs_config)

/**
 * @defgroup Tessellator parameter enumerations.
 *
 * These correspond to the hardware values in 3DSTATE_TE, and are provided
 * as part of the tessellation evaluation shader.
 *
 * @{
 */
enum intel_tess_partitioning {
   INTEL_TESS_PARTITIONING_INTEGER         = 0,
   INTEL_TESS_PARTITIONING_ODD_FRACTIONAL  = 1,
   INTEL_TESS_PARTITIONING_EVEN_FRACTIONAL = 2,
};

enum intel_tess_output_topology {
   INTEL_TESS_OUTPUT_TOPOLOGY_POINT   = 0,
   INTEL_TESS_OUTPUT_TOPOLOGY_LINE    = 1,
   INTEL_TESS_OUTPUT_TOPOLOGY_TRI_CW  = 2,
   INTEL_TESS_OUTPUT_TOPOLOGY_TRI_CCW = 3,
};

enum intel_tess_domain {
   INTEL_TESS_DOMAIN_QUAD    = 0,
   INTEL_TESS_DOMAIN_TRI     = 1,
   INTEL_TESS_DOMAIN_ISOLINE = 2,
};
/** @} */

enum intel_shader_dispatch_mode {
   INTEL_DISPATCH_MODE_4X1_SINGLE = 0,
   INTEL_DISPATCH_MODE_4X2_DUAL_INSTANCE = 1,
   INTEL_DISPATCH_MODE_4X2_DUAL_OBJECT = 2,
   INTEL_DISPATCH_MODE_SIMD8 = 3,

   INTEL_DISPATCH_MODE_TCS_SINGLE_PATCH = 0,
   INTEL_DISPATCH_MODE_TCS_MULTI_PATCH = 2,
};

enum intel_barycentric_mode {
   INTEL_BARYCENTRIC_PERSPECTIVE_PIXEL       = 0,
   INTEL_BARYCENTRIC_PERSPECTIVE_CENTROID    = 1,
   INTEL_BARYCENTRIC_PERSPECTIVE_SAMPLE      = 2,
   INTEL_BARYCENTRIC_NONPERSPECTIVE_PIXEL    = 3,
   INTEL_BARYCENTRIC_NONPERSPECTIVE_CENTROID = 4,
   INTEL_BARYCENTRIC_NONPERSPECTIVE_SAMPLE   = 5,
   INTEL_BARYCENTRIC_MODE_COUNT              = 6
};
#define INTEL_BARYCENTRIC_PERSPECTIVE_BITS \
   ((1 << INTEL_BARYCENTRIC_PERSPECTIVE_PIXEL) | \
    (1 << INTEL_BARYCENTRIC_PERSPECTIVE_CENTROID) | \
    (1 << INTEL_BARYCENTRIC_PERSPECTIVE_SAMPLE))
#define INTEL_BARYCENTRIC_NONPERSPECTIVE_BITS \
   ((1 << INTEL_BARYCENTRIC_NONPERSPECTIVE_PIXEL) | \
    (1 << INTEL_BARYCENTRIC_NONPERSPECTIVE_CENTROID) | \
    (1 << INTEL_BARYCENTRIC_NONPERSPECTIVE_SAMPLE))

enum intel_vue_layout {
   /**
    * Layout is fixed and shared by producer/consumer, allowing for tigh
    * packing
    */
   INTEL_VUE_LAYOUT_FIXED = 0,
   /**
    * Layout is separate, works for ARB_separate_shader_objects but without
    * Mesh support.
    */
   INTEL_VUE_LAYOUT_SEPARATE,
   /**
    * Layout is separate and works with Mesh shaders.
    */
   INTEL_VUE_LAYOUT_SEPARATE_MESH,
};

/**
 * Data structure recording the relationship between the gl_varying_slot enum
 * and "slots" within the vertex URB entry (VUE).  A "slot" is defined as a
 * single octaword within the VUE (128 bits).
 *
 * Note that each BRW register contains 256 bits (2 octawords), so when
 * accessing the VUE in URB_NOSWIZZLE mode, each register corresponds to two
 * consecutive VUE slots.  When accessing the VUE in URB_INTERLEAVED mode (as
 * in a vertex shader), each register corresponds to a single VUE slot, since
 * it contains data for two separate vertices.
 */
struct intel_vue_map {
   /**
    * Bitfield representing all varying slots that are (a) stored in this VUE
    * map, and (b) actually written by the shader.  Does not include any of
    * the additional varying slots defined in brw_varying_slot.
    */
   uint64_t slots_valid;

   /**
    * The layout of the VUE
    *
    * Separable programs (GL_ARB_separate_shader_objects) can be mixed and
    * matched without the linker having a chance to dead code eliminate unused
    * varyings.
    *
    * This means that we have to use a fixed slot layout, based on the output's
    * location field, rather than assigning slots in a compact contiguous block.
    *
    * When using Mesh, another constraint arises which is the HW limits for
    * loading per-primitive & per-vertex data, limited to 32 varying in total.
    * This requires us to be quite inventive with the way we lay things out.
    * Take a fragment shader loading the following data :
    *
    *    float gl_ClipDistance[];
    *    uint gl_PrimitiveID;
    *    vec4 someAppValue[29];
    *
    * According to the Vulkan spec, someAppValue will occupy 29 slots,
    * gl_PrimitiveID 1 slot, gl_ClipDistance[] up to 2 slots. If the input is
    * coming from a VS/DS/GS shader, we can load all of this through a single
    * block using 3DSTATE_SBE::VertexURBEntryReadLength = 16 (maximum
    * programmable value) and the layout with
    * BRW_VUE_MAP_LAYOUT_FIXED/BRW_VUE_MAP_LAYOUT_SEPARATE will be this :
    *
    *   -----------------------
    *   | gl_ClipDistance 0-3 |
    *   |---------------------|
    *   | gl_ClipDistance 4-7 |
    *   |---------------------|
    *   |   gl_PrimitiveID    |
    *   |---------------------|
    *   |   someAppValue[]    |
    *   |---------------------|
    *
    * This works nicely as everything is coming from the same location in the
    * URB.
    *
    * When mesh shaders are involved, gl_PrimitiveID is located in a different
    * place in the URB (the per-primitive block) and requires programming
    * 3DSTATE_SBE_MESH::PerPrimitiveURBEntryOutputReadLength to load some
    * additional data. The HW has a limit such that
    * 3DSTATE_SBE_MESH::PerPrimitiveURBEntryOutputReadLength +
    * 3DSTATE_SBE_MESH::PerVertexURBEntryOutputReadLength <= 16. With the
    * layout above, we would not be able to accomodate that HW limit.
    *
    * The solution to this is to lay the built-in varyings out
    * (gl_ClipDistance omitted since it's part of the VUE header and cannot
    * live any other place) at the end of the VUE like this :
    *
    *   -----------------------
    *   | gl_ClipDistance 0-3 |
    *   |---------------------|
    *   | gl_ClipDistance 4-7 |
    *   |---------------------|
    *   |   someAppValue[]    |
    *   |---------------------|
    *   |   gl_PrimitiveID    |
    *   |---------------------|
    *
    * This layout adds another challenge because with separate shader
    * compilations, we cannot tell in the consumer shader how many outputs the
    * producer has, so we don't know where the gl_PrimitiveID lives. The
    * solution to this other problem is to read the built-in with a
    * MOV_INDIRECT and have the offset of the MOV_INDIRECT loaded through a
    * push constant.
    */
   enum intel_vue_layout layout;

   /**
    * Map from gl_varying_slot value to VUE slot.  For gl_varying_slots that are
    * not stored in a slot (because they are not written, or because
    * additional processing is applied before storing them in the VUE), the
    * value is -1.
    */
   int8_t varying_to_slot[NUM_TOTAL_VARYING_SLOTS];

   /**
    * Map from VUE slot to gl_varying_slot value.  For slots that do not
    * directly correspond to a gl_varying_slot, the value comes from
    * elk_varying_slot.
    *
    * For slots that are not in use, the value is -1 (brw) or
    * ELK_VARYING_SLOT_PAD.
    */
   int8_t slot_to_varying[NUM_TOTAL_VARYING_SLOTS];

   /**
    * Total number of VUE slots in use
    */
   int num_slots;

   /**
    * Number of position VUE slots.  If num_pos_slots > 1, primitive
    * replication is being used.
    */
   int num_pos_slots;

   /**
    * Number of per-patch VUE slots. Only valid for tessellation control
    * shader outputs and tessellation evaluation shader inputs.
    */
   int num_per_patch_slots;

   /**
    * Number of per-vertex VUE slots. Only valid for tessellation control
    * shader outputs and tessellation evaluation shader inputs.
    */
   int num_per_vertex_slots;

   /**
    * Location at which the builtins live.
    */
   int builtins_slot_offset;
};

struct intel_cs_dispatch_info {
   uint32_t group_size;
   uint32_t simd_size;
   uint32_t threads;

   /* RightExecutionMask field used in GPGPU_WALKER. */
   uint32_t right_mask;
};

enum intel_compute_walk_order {
   INTEL_WALK_ORDER_XYZ = 0,
   INTEL_WALK_ORDER_XZY = 1,
   INTEL_WALK_ORDER_YXZ = 2,
   INTEL_WALK_ORDER_YZX = 3,
   INTEL_WALK_ORDER_ZXY = 4,
   INTEL_WALK_ORDER_ZYX = 5,
};

/** Compute dispatch configuration selected while lowering system values. */
struct intel_cs_dispatch_params {
   uint8_t generate_local_id;
   enum intel_compute_walk_order walk_order;
};

static inline uint32_t
intel_tess_config(uint32_t input_vertices,
                  uint32_t output_vertices,
                  enum intel_tess_domain tess_domain,
                  uint32_t num_per_patch_slots,
                  uint32_t num_per_vertex_slots,
                  uint32_t builtins_slot_offset)
{
   assert(num_per_patch_slots < (1u << INTEL_TESS_CONFIG_PER_PATCH_SLOTS_SIZE));
   assert(num_per_vertex_slots < (1u << INTEL_TESS_CONFIG_PER_VERTEX_SLOTS_SIZE));
   assert(builtins_slot_offset < (1u << INTEL_TESS_CONFIG_BUILTINS_SIZE));
   assert(input_vertices != 0);
   assert((input_vertices - 1) < (1u << INTEL_TESS_CONFIG_INPUT_VERTICES_SIZE));
   assert((output_vertices - 1) < (1u << INTEL_TESS_CONFIG_OUTPUT_VERTICES_SIZE));

   const uint32_t primitive_flags =
      tess_domain == INTEL_TESS_DOMAIN_TRI ? INTEL_TESS_CONFIG_TRIANGLES :
      tess_domain == INTEL_TESS_DOMAIN_QUAD ? INTEL_TESS_CONFIG_QUADS :
      INTEL_TESS_CONFIG_ISOLINES;

   return
      (((input_vertices - 1) & 0x1f) << INTEL_TESS_CONFIG_INPUT_VERTICES_OFFSET) |
      (((output_vertices - 1) & 0x1f) << INTEL_TESS_CONFIG_OUTPUT_VERTICES_OFFSET) |
      primitive_flags |
      (num_per_patch_slots << INTEL_TESS_CONFIG_PER_PATCH_SLOTS_OFFSET) |
      (num_per_vertex_slots << INTEL_TESS_CONFIG_PER_VERTEX_SLOTS_OFFSET) |
      (builtins_slot_offset << INTEL_TESS_CONFIG_BUILTINS_OFFSET);

}

static inline enum intel_barycentric_mode
intel_fs_barycentric_mode_for_persample_dispatch(bool persample_dispatch,
                                                 enum intel_barycentric_mode mode)
{
   /* From the BDW PRM documentation for 3DSTATE_WM:
    *
    *    "MSDISPMODE_PERSAMPLE is required in order to select Perspective
    *     Sample or Non- perspective Sample barycentric coordinates."
    *
    * So cleanup any potentially set sample barycentric mode when not in per
    * sample dispatch.
    */
   if (!persample_dispatch &&
       (mode == INTEL_BARYCENTRIC_PERSPECTIVE_SAMPLE ||
        mode == INTEL_BARYCENTRIC_NONPERSPECTIVE_SAMPLE))
      return (enum intel_barycentric_mode)(mode - 2);
   return mode;
}

struct intel_fs_params {
   bool persample_interp;
   uint32_t rasterization_samples;
   bool coarse_pixel;
   bool alpha_to_coverage;
   bool provoking_vertex_last;
   uint32_t first_vue_slot;
   uint32_t primitive_id_index;
   bool per_primitive_remapping;
   bool conservative_raster;
};

static inline enum intel_fs_config
intel_fs_config(struct intel_fs_params params)
{
   enum intel_fs_config fs_config = INTEL_FS_CONFIG_ENABLE_DYNAMIC;

   if (params.rasterization_samples > 1) {
      fs_config |= INTEL_FS_CONFIG_MULTISAMPLE_FBO;

      if (params.persample_interp)
         fs_config |= INTEL_FS_CONFIG_PERSAMPLE_DISPATCH;
   }

   if (!(fs_config & INTEL_FS_CONFIG_PERSAMPLE_DISPATCH) &&
       params.coarse_pixel) {
      fs_config |= INTEL_FS_CONFIG_COARSE_PI_MSG |
                   INTEL_FS_CONFIG_COARSE_RT_WRITES;
   }

   if (params.alpha_to_coverage)
      fs_config |= INTEL_FS_CONFIG_ALPHA_TO_COVERAGE;

   assert(params.first_vue_slot < (1 << INTEL_FS_CONFIG_FIRST_VUE_SLOT_SIZE));
   fs_config |= (enum intel_fs_config)(
      params.first_vue_slot << INTEL_FS_CONFIG_FIRST_VUE_SLOT_OFFSET);

   assert(params.primitive_id_index < (1u << INTEL_FS_CONFIG_PRIMITIVE_ID_INDEX_SIZE));
   fs_config |= (enum intel_fs_config)(
      params.primitive_id_index << INTEL_FS_CONFIG_PRIMITIVE_ID_INDEX_OFFSET);

   if (params.provoking_vertex_last)
      fs_config |= INTEL_FS_CONFIG_PROVOKING_VERTEX_LAST;

   if (params.per_primitive_remapping)
      fs_config |= INTEL_FS_CONFIG_PER_PRIMITIVE_REMAPPING;

   if (params.conservative_raster)
      fs_config |= INTEL_FS_CONFIG_CONSERVATIVE_RASTER;

   return fs_config;
}

#define INTEL_MAX_EMBEDDED_SAMPLERS (4096)

enum intel_shader_reloc_id {
   INTEL_SHADER_RELOC_CONST_DATA_ADDR_LOW,
   INTEL_SHADER_RELOC_CONST_DATA_ADDR_HIGH,
   INTEL_SHADER_RELOC_SHADER_START_OFFSET,

   INTEL_LAST_COMMON_SHADER_RELOC = INTEL_SHADER_RELOC_SHADER_START_OFFSET,

   BRW_SHADER_RELOC_RESUME_SBT_ADDR_LOW = INTEL_LAST_COMMON_SHADER_RELOC + 1,
   BRW_SHADER_RELOC_RESUME_SBT_ADDR_HIGH,
   BRW_SHADER_RELOC_DESCRIPTORS_ADDR_HIGH,
   BRW_SHADER_RELOC_DESCRIPTORS_BUFFER_ADDR_HIGH,
   BRW_SHADER_RELOC_PUSH_DESCRIPTORS_BUFFER_ADDR_HIGH,
   BRW_SHADER_RELOC_DESCRIPTORS_VIEW_HANDLE,
   BRW_SHADER_RELOC_DESCRIPTORS_BUFFERS_VIEW_HANDLE,
   BRW_SHADER_RELOC_INSTRUCTION_BASE_ADDR_HIGH,
   BRW_SHADER_RELOC_PRINTF_BUFFER_ADDR_LOW,
   BRW_SHADER_RELOC_PRINTF_BUFFER_ADDR_HIGH,
   BRW_SHADER_RELOC_PRINTF_BUFFER_SIZE,
   BRW_SHADER_RELOC_NULL_CACHELINE_ADDR_LOW,
   BRW_SHADER_RELOC_NULL_CACHELINE_ADDR_HIGH,
   BRW_SHADER_RELOC_SCRATCH64_SURFACE_HIGH,
   BRW_SHADER_RELOC_SCRATCH64_SURFACE_LOW,
   BRW_SHADER_RELOC_DESCRIPTORS_APP_HIGH,
   BRW_SHADER_RELOC_DESCRIPTORS_INTERNAL_HIGH,
   /* Leave this entry last: */
   BRW_SHADER_RELOC_EMBEDDED_SAMPLER_HANDLE,
   BRW_SHADER_RELOC_LAST_EMBEDDED_SAMPLER_HANDLE =
   BRW_SHADER_RELOC_EMBEDDED_SAMPLER_HANDLE + INTEL_MAX_EMBEDDED_SAMPLERS - 1,
};

enum intel_shader_reloc_type {
   /** An arbitrary 32-bit value */
   INTEL_SHADER_RELOC_TYPE_U32,
   /** A MOV instruction with an immediate source */
   INTEL_SHADER_RELOC_TYPE_MOV_IMM,
};

/** Represents a code relocation
 *
 * Relocatable constants are immediates in the code which we want to be able
 * to replace post-compile with the actual value.
 */
struct intel_shader_reloc {
   /** The 32-bit ID of the relocatable constant */
   uint32_t id;

   /** Type of this relocation */
   enum intel_shader_reloc_type type;

   /** The offset in the shader to the relocated value
    *
    * For MOV_IMM relocs, this is an offset to the MOV instruction.  This
    * allows us to do some sanity checking while we update the value.
    */
   uint32_t offset;

   /** Value to be added to the relocated value before it is written */
   uint32_t delta;
};

/** A value to write to a relocation */
struct intel_shader_reloc_value {
   /** The 32-bit ID of the relocatable constant */
   uint32_t id;

   /** The value with which to replace the relocated immediate */
   uint32_t value;
};

struct intel_push_layout {
   /** Byte size of each of the four 32-byte-aligned push ranges. */
   uint16_t push_sizes[4];

   /** Bitmask of push ranges requiring robust UBO bounds checks. */
   uint8_t robust_ubo_ranges;

   /**
    * Dword index of four packed 8-bit UBO limits, in 16-byte units.
    * Valid when robust_ubo_ranges is nonzero.
    */
   unsigned push_reg_mask_param;

   /**
    * Byte offset of intel_tess_config in the pushed data.
    */
   unsigned tess_config_param;

   /**
    * Byte offset of intel_fs_config in the pushed data.
    */
   unsigned fs_config_param;
};

#ifndef __OPENCL_VERSION__
struct intel_shader_reloc;

struct brw_push_const_block {
   unsigned dwords;     /* Dword count, not reg aligned */
   unsigned regs;
   unsigned size;       /* Bytes, register aligned */
};

struct brw_cs_push_const_info {
   struct brw_push_const_block cross_thread;
   struct brw_push_const_block per_thread;
};

struct brw_tue_map {
   uint32_t size_dw;
};

struct brw_mue_map {
   /* Total size in bytes of the MUE (32B aligned) */
   uint32_t size;

   uint32_t max_primitives;
   uint32_t max_vertices;

   /* Stride in bytes between sets of primitive indices */
   uint32_t per_primitive_indices_stride;

   /* Per primitive offset from the start of the MUE (32B aligned) */
   uint32_t per_primitive_offset;

   /* Per primitive stride in bytes (32B aligned) */
   uint32_t per_primitive_stride;

   /* Per vertex offset in bytes from the start of the MUE (32B aligned) */
   uint32_t per_vertex_offset;

   /* Per vertex stride in bytes (32B aligned) */
   uint32_t per_vertex_stride;

   /* VUE map for the per vertex attributes */
   struct intel_vue_map vue_map;

   /* Offset in bytes of each per primitive relative to
    * per_primitive_offset (-1 if unused)
    */
   int per_primitive_offsets[VARYING_SLOT_MAX];

   /* Whether the per primitive block includes a header */
   bool has_per_primitive_header;

   bool wa_18019110168_active;
};

enum brw_mesh_index_format {
   BRW_INDEX_FORMAT_U32,
   BRW_INDEX_FORMAT_U888X,
};

struct brw_tess_info {
   enum tess_primitive_mode primitive_mode:8;
   uint8_t                         spacing:2;
   bool                                ccw:1;
   bool                         point_mode:1;
   uint32_t                            pad:20;
};

/**
 * Plain-old-data metadata about a compiled shader.
 *
 * Must not contain pointers, so it can be trivially serialized.
 */
struct intel_shader_info {
   uint64_t source_hash;
   mesa_shader_stage stage;

   /** How many ray query objects in this shader. */
   unsigned ray_queries;

   unsigned total_scratch;
   unsigned total_shared;

   /** Number of GRF registers used. */
   unsigned grf_used;

   /**
    * Register where the thread expects to find input data from the URB
    * (typically uniforms, followed by vertex or fragment attributes).
    */
   unsigned dispatch_grf_start_reg;

   unsigned const_data_offset;
   unsigned const_data_size;

   struct intel_push_layout push_layout;

   /* Was this shader compiled with Jay? */
   bool is_jay;

   /** Does this shader pull from any UBO or other constant buffers? */
   bool has_ubo_pull;

   /* Fields common to subset of stages */
   union {
      struct {
         struct intel_vue_map vue_map;
         uint32_t urb_read_length;
         uint32_t clip_distance_mask;
         uint32_t cull_distance_mask;

         /* Used for calculating urb partitions.  In the VS, this is the size of the
          * URB entry used for both input and output to the thread.  In the GS, this
          * is the size of the URB entry used for output.
          */
         uint32_t urb_entry_size;

         enum intel_shader_dispatch_mode dispatch_mode;

         /** Should the hardware deliver input VUE handles for URB pull loads? */
         bool include_vue_handles;
      } vue; /* VS, TES, TCS, GS */
      struct {
         uint32_t local_size[3];

         /* Program offsets for the 8/16/32 SIMD variants.  Multiple variants are
          * kept when using variable group size, and the right one can only be
          * decided at dispatch time.
          */
         uint32_t prog_offset[3];

         /* Bitmask indicating which program offsets are valid. */
         uint8_t prog_mask;

         /* Bitmask indicating which programs have spilled. */
         uint8_t prog_spilled;

         bool uses_barrier;
         bool uses_systolic;
      } wg; /* CS, KERNEL, TASK, MESH */
   };

   /* Extra stage-specific fields. */
   union {
      struct {
         uint64_t inputs_read;
         uint64_t double_inputs_read;
         uint32_t vf_component_packing[4];
         bool uses_vertexid;
         bool uses_instanceid;
         bool uses_firstvertex;
         bool uses_baseinstance;
         bool uses_drawid;
      } vs;
      struct {
         enum intel_compute_walk_order walk_order;
         struct brw_cs_push_const_info push;

         bool uses_fence;
         bool uses_btd_stack_ids;

         /* True if shader has any sample operation */
         bool uses_sampler;

         /* True if the shader was compiled with SIMD32 forced */
         bool force_simd32;

         uint8_t generate_local_id;
      } cs;
      struct {
         struct brw_tess_info tess_info;

         /** Number of input vertices, 0 means dynamic */
         uint32_t input_vertices;

         /** Number of output vertices */
         uint32_t output_vertices;

         /** Number vertices in output patch */
         uint32_t instances;

         /** Track patch count threshold */
         int32_t patch_count_threshold;

         /** Should the non-SINGLE_PATCH payload provide primitive ID? */
         bool include_primitive_id;
      } tcs;
      struct {
         struct brw_tess_info tess_info;
         bool include_primitive_id;
      } tes;
      struct {
         uint32_t vertices_in;

         /**
          * Size of an output vertex, measured in HWORDS (32 bytes).
          */
         uint32_t output_vertex_size_hwords;

         uint32_t output_topology;

         /**
          * Size of the control data (cut bits or StreamID bits), in hwords (32
          * bytes).  0 if there is no control data.
          */
         uint32_t control_data_header_size_hwords;

         /**
          * Format of the control data (either GFX7_GS_CONTROL_DATA_FORMAT_GSCTL_SID
          * if the control data is StreamID bits, or
          * GFX7_GS_CONTROL_DATA_FORMAT_GSCTL_CUT if the control data is cut bits).
          * Ignored if control_data_header_size is 0.
          */
         uint32_t control_data_format;

         /**
          * The number of vertices emitted, if constant - otherwise -1.
          */
         int32_t static_vertex_count;

         uint32_t invocations;
         bool include_primitive_id;
      } gs;
      struct {
         bool dispatch_8;
         bool dispatch_16;
         bool dispatch_32;

         /**
          * Dispatch width of the multi-polygon PS kernel, or 0 if no
          * multi-polygon kernel was built.
          */
         uint8_t dispatch_multi;

         /**
          * Number of polygons handled in parallel by the multi-polygon PS
          * kernel.
          */
         uint8_t max_polygons;

         uint8_t dispatch_grf_start_reg_16;
         uint8_t dispatch_grf_start_reg_32;

         uint8_t computed_depth_mode;

         uint32_t prog_offset_16;
         uint32_t prog_offset_32;

         /**
          * Number of slots (16B) chunks dedicated to per primitive payload.
          */
         unsigned num_per_primitive_inputs;

         /**
          * Number of slots (16B) chunks dedicated to per vertex payload.
          */
         unsigned num_varying_inputs;

         /**
          * Shader writes the SampleMask and this is AND-ed with the API's
          * SampleMask to generate a new coverage mask.
          */
         enum intel_sometimes alpha_to_coverage;

         /*
          * Provoking vertex may be dynamically set to last and we need to know
          * its value for fragment shader barycentrics and flat inputs.
          */
         enum intel_sometimes provoking_vertex_last;

         /**
          * Mask of which interpolation modes are required by the fragment shader.
          * Those interpolations are delivered as part of the thread payload. Used
          * in hardware setup on gfx6+.
          */
         uint32_t barycentric_interp_modes;

         /**
          * Mask of which FS inputs are marked flat by the shader source.  This is
          * needed for setting up 3DSTATE_SF/SBE.
          */
         uint32_t flat_inputs;

         /**
          * The FS inputs
          */
         uint64_t inputs;

         /**
          * The FS per-primitive inputs (some bits can be in both inputs &
          * per_primitive_inputs if the shader is compiled without being linked to
          * the previous stage)
          */
         uint64_t per_primitive_inputs;

         /**
          * Map from gl_varying_slot to the position within the FS setup data
          * payload where the varying's attribute vertex deltas should be
          * delivered.  For varying slots that are not used by the FS, the value
          * is -1.
          */
         int urb_setup[VARYING_SLOT_MAX];

         /**
          * Cache structure into the urb_setup array above that contains the
          * attribute numbers of active varyings out of urb_setup.
          * The actual count is stored in urb_setup_attribs_count.
          */
         uint8_t urb_setup_attribs[VARYING_SLOT_MAX];
         uint8_t urb_setup_attribs_count;

         bool computed_stencil;
         bool early_fragment_tests;
         bool post_depth_coverage;
         bool inner_coverage;
         bool dual_src_blend;
         bool uses_pos_offset;
         bool uses_omask;
         bool uses_kill;
         bool uses_src_xy;
         bool uses_src_depth;
         bool uses_src_w;
         bool uses_depth_w_coefficients;
         bool uses_pc_bary_coefficients;
         bool uses_npc_bary_coefficients;
         bool uses_sample_offsets;
         bool uses_sample_mask;
         bool uses_vmask;
         bool has_side_effects;
         bool pulls_bary;
         bool uses_fully_covered;

         /**
          * Whether nonperspective interpolation modes are used by the
          * barycentric_interp_modes or fragment shader through interpolator
          * messages.
          */
         bool uses_nonperspective_interp_modes;

         /** Fragment shader barycentrics
          *
          * Request that the HW delta computation of attributes be turned off. This
          * is needed when the FS has per-vertex inputs or reads
          * BaryCoordKHR/BaryCoordNoPerspKHR.
          */
         bool vertex_attributes_bypass;

         /** Shader is using per-sample interpolation */
         bool persample_interp;

         /** Whether this shader uses the FS config push data value */
         bool uses_fs_config;

         /** Whether this shader uses the FS color offset push data value
          *
          * Efficient 64bit mode only
          */
         bool uses_fs_color_offset;

         /** Whether this shader uses the FS color map push data value
          *
          * Efficient 64bit mode only
          */
         bool uses_fs_color_map;

         /** Should this shader be dispatched per-sample */
         bool persample_dispatch;

         /**
          * Shader is ran at the coarse pixel shading dispatch rate (3DSTATE_CPS).
          */
         bool coarse_pixel_dispatch;

         /** Whether the shader was compiled with a preference for SIMD32. */
         bool prefer_simd32;
      } fs;
      struct {
         struct brw_tue_map map;
         bool uses_drawid;
      } task;
      struct {
         struct brw_mue_map map;
         uint32_t clip_distance_mask;
         uint32_t cull_distance_mask;
         enum brw_mesh_index_format index_format;

         /**
          * Offset in the program where the remapping table for Wa_18019110168 is
          * located.
          *
          * The remapping table has the same format as
          * intel_vue_map::varying_to_slot[]
          */
         uint32_t wa_18019110168_mapping_offset;

         uint16_t primitive_type;
         bool uses_drawid;
         bool autostrip_enable;
      } mesh;
      struct {
         /** Maximum stack size of all shaders */
         uint32_t max_stack_size;

         /** Offset into the shader where the resume SBT is located */
         uint32_t resume_sbt_offset;

         /** SIMD size of the root shader */
         uint8_t simd_size;
      } bs;
   };
};

#define INTEL_SHADER_BIN_MAX_STATS 3

/**
 * A compiled shader binary and the metadata needed to use it.
 *
 * This is the output of the backend compilers (brw and jay).
 */
typedef struct intel_shader_bin {
   struct intel_shader_info info;

   const void *code;
   unsigned code_size;

   const struct intel_shader_reloc *relocs;
   unsigned num_relocs;

   struct genisa_stats stats[INTEL_SHADER_BIN_MAX_STATS];
   unsigned num_stats;
} intel_shader_bin;
#endif
#ifdef __cplusplus
} /* extern "C" */
#endif
