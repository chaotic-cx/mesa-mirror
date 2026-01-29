/*
 * Copyright © 2025 Imagination Technologies Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PVR_IFACE_H
#define PVR_IFACE_H

/**
 * \file pvr_iface.h
 *
 * \brief USC/PDS program interface.
 */

/** Query availability shader data; shared registers. */
enum pvr_query_availability_data {
   PVR_QUERY_AVAILABILITY_DATA_INDEX_COUNT,

   PVR_QUERY_AVAILABILITY_DATA_INDEX_BO_LO,
   PVR_QUERY_AVAILABILITY_DATA_INDEX_BO_HI,

   PVR_QUERY_AVAILABILITY_DATA_BO_LO,
   PVR_QUERY_AVAILABILITY_DATA_BO_HI,

   _PVR_QUERY_AVAILABILITY_DATA_COUNT,
};

/** Query copy shader data; shared registers. */
enum pvr_query_copy_data {
   PVR_QUERY_COPY_DATA_INDEX_COUNT,

   PVR_QUERY_COPY_DATA_DEST_BO_LO,
   PVR_QUERY_COPY_DATA_DEST_BO_HI,

   PVR_QUERY_COPY_DATA_AVAILABILITY_BO_LO,
   PVR_QUERY_COPY_DATA_AVAILABILITY_BO_HI,

   PVR_QUERY_COPY_DATA_RESULT_BO_LO,
   PVR_QUERY_COPY_DATA_RESULT_BO_HI,

   PVR_QUERY_COPY_DATA_DEST_STRIDE,

   PVR_QUERY_COPY_DATA_FLAGS,

   /** Stride in bytes between elements in the result buffer. */
   PVR_QUERY_COPY_DATA_RESULT_STRIDE,

   /** Number of 32-bit reports per query result. */
   PVR_QUERY_COPY_DATA_REPORT_COUNT,

   _PVR_QUERY_COPY_DATA_COUNT,
};

/** Query reset shader data; shared registers. */
enum pvr_query_reset_data {
   PVR_QUERY_RESET_DATA_INDEX_COUNT,

   PVR_QUERY_RESET_DATA_RESULT_BO_LO,
   PVR_QUERY_RESET_DATA_RESULT_BO_HI,

   PVR_QUERY_RESET_DATA_AVAILABILITY_BO_LO,
   PVR_QUERY_RESET_DATA_AVAILABILITY_BO_HI,

   /** Stride in bytes between elements in the result buffer. */
   PVR_QUERY_RESET_DATA_RESULT_STRIDE,

   /** Number of 32-bit reports per query result. */
   PVR_QUERY_RESET_DATA_REPORT_COUNT,

   _PVR_QUERY_RESET_DATA_COUNT,
};

/** Clear attachment shader data; shared registers. */
enum pvr_clear_attach_data {
   PVR_CLEAR_ATTACH_DATA_DWORD0,
   PVR_CLEAR_ATTACH_DATA_DWORD1,
   PVR_CLEAR_ATTACH_DATA_DWORD2,
   PVR_CLEAR_ATTACH_DATA_DWORD3,

   PVR_CLEAR_ATTACH_DATA_TILE_ADDR_LO,
   PVR_CLEAR_ATTACH_DATA_TILE_ADDR_HI,

   _PVR_CLEAR_ATTACH_DATA_COUNT,
};

/** SPM load shader data; shared registers. */
enum pvr_spm_load_data {
   PVR_SPM_LOAD_DATA_SMP = 0,

   PVR_SPM_LOAD_DATA_REG_TEX = 4,

   PVR_SPM_LOAD_DATA_BUF_TEX_0 = 8,
   PVR_SPM_LOAD_DATA_BUF_ADDR_0 = 12,

   PVR_SPM_LOAD_DATA_BUF_ADDR_1 = 14,
   PVR_SPM_LOAD_DATA_BUF_TEX_1 = 16,

   PVR_SPM_LOAD_DATA_BUF_TEX_2 = 20,
   PVR_SPM_LOAD_DATA_BUF_ADDR_2 = 24,

   PVR_SPM_LOAD_DATA_BUF_ADDR_3 = 26,
   PVR_SPM_LOAD_DATA_BUF_TEX_3 = 28,

   PVR_SPM_LOAD_DATA_BUF_TEX_4 = 32,
   PVR_SPM_LOAD_DATA_BUF_ADDR_4 = 36,

   PVR_SPM_LOAD_DATA_BUF_ADDR_5 = 38,
   PVR_SPM_LOAD_DATA_BUF_TEX_5 = 40,

   PVR_SPM_LOAD_DATA_BUF_TEX_6 = 44,
   PVR_SPM_LOAD_DATA_BUF_ADDR_6 = 48,

   _PVR_SPM_LOAD_DATA_COUNT = 50,
};

/** Load/store sr shader data; vertex input registers. */
enum pvr_load_store_sr_data {
   PVR_LOAD_STORE_SR_DATA_SIZE_ADDR_LO,
   PVR_LOAD_STORE_SR_DATA_SIZE_ADDR_HI,

   PVR_LOAD_STORE_SR_DATA_STORE_ADDR_LO,
   PVR_LOAD_STORE_SR_DATA_STORE_ADDR_HI,

   _PVR_LOAD_STORE_SR_DATA_COUNT,
};

/** IDFWDF shader data; shared registers. */
enum pvr_idfwdf_sr_data {
   PVR_IDFWDF_DATA_TEX = 0,
   PVR_IDFWDF_DATA_SMP = 4,

   PVR_IDFWDF_DATA_ADDR_LO,
   PVR_IDFWDF_DATA_ADDR_HI,

   _PVR_IDFWDF_DATA_COUNT,
};

#define PVR_IDFWDF_TEX_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT
#define PVR_IDFWDF_TEX_WIDTH 4U
#define PVR_IDFWDF_TEX_HEIGHT 2U
#define PVR_IDFWDF_TEX_STRIDE 4U

/** pck_info and pck_format bitfield helpers. */
#define PVR_PCK_FORMAT_INVALID 0b11111U
#define PVR_PCK_INFO_FORMAT_LENGTH 5U
#define PVR_PCK_INFO_FORMAT_OFFSET 0U
#define PVR_PCK_INFO_SPLIT_OFFSET 5U
#define PVR_PCK_INFO_SCALE_OFFSET 6U
#define PVR_PCK_INFO_ROUNDZERO_OFFSET 7U
#define PVR_PCK_INFO_UNORM_OFFSET 8U
#define PVR_PCK_INFO_SPLIT_BIT BITFIELD_BIT(PVR_PCK_INFO_SPLIT_OFFSET)
#define PVR_PCK_INFO_SCALE_BIT BITFIELD_BIT(PVR_PCK_INFO_SCALE_OFFSET)
#define PVR_PCK_INFO_ROUNDZERO_BIT BITFIELD_BIT(PVR_PCK_INFO_ROUNDZERO_OFFSET)
#define PVR_PCK_INFO_UNORM_BIT BITFIELD_BIT(PVR_PCK_INFO_UNORM_OFFSET)

/** Fragment shader metadata bit offsets. */
#define PVR_FS_META_ALPHA_TO_ONE_OFFSET 0U
#define PVR_FS_META_COLOR_WRITE_ENABLE_OFFSET 1U
#define PVR_FS_META_SAMPLE_MASK_OFFSET 9U
#define PVR_FS_META_ALPHA_TO_COVERAGE_OFFSET 25U
#define PVR_FS_META_SAMPLE_SHADING 26U

/** Fragment shader metadata lengths */
#define PVR_FS_META_ALPHA_TO_ONE_LENGTH 1U
#define PVR_FS_META_COLOR_WRITE_ENABLE_LENGTH 8U
#define PVR_FS_META_SAMPLE_MASK_LENGTH 16U
#define PVR_FS_META_ALPHA_TO_COVERAGE_LENGTH 1U
#define PVR_FS_META_SAMPLE_SHADING_LENGTH 1U

/** Sliced view of 3d helpers. */
#define PVR_SLICED_VIEW_OFFSET_OFFSET 0U
#define PVR_SLICED_VIEW_OFFSET_LENGTH 16U
#define PVR_SLICED_VIEW_COUNT_OFFSET 16U
#define PVR_SLICED_VIEW_COUNT_LENGTH 16U

/**/

/** VDM ptemps. */
enum pvr_vdm_ptemp {
   PVR_VDM_PTEMP_VIEW_INDEX_OUT = 4,
};

/* PDSC XFB main patch IDs */

enum pvr_xfb_main_patch_id {
   /** Patch ID for first STM instruction source 2 constant */
   PVR_XFB_MAIN_PATCH_ID_STM_SRC2_CONST_0 = 0,
};

/* PDSC XFB init patch IDs */

enum pvr_xfb_init_patch_id {
   /** Patch ID for transform feedback persistent count */
   PVR_XFB_INIT_PATCH_ID_PTEMP_LD_COUNT,

   /** Patch IDs for first varying STM instruction ptemp load */
   PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_0,

   /** Patch IDs for second varying STM instruction ptemp load */
   PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_1,

   /** Patch IDs for third varying STM instruction ptemp load */
   PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_2,

   /** Patch IDs for fourth varying STM instruction ptemp load */
   PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_3,

   /** Patch ID for transform feedback counter buffer load */
   PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_LD_SRC0,

   /** Patch ID for transform feedback counter buffer store */
   PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_ST_SRC0,

   /** Patch ID for transform feedback counter buffer alignment dword select */
   PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_DWORD_SELECT,

   /** Patch ID for transform feedback counter buffer count */
   PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_COUNT,
};

/* PDSC XFB init reference IDs */

enum pvr_xfb_init_ref_id {
   /** Ref ID for first ptemp destination */
   PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_0,

   /** Ref ID for second ptemp destination */
   PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_1,

   /** Ref ID for third ptemp destination */
   PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_2,

   /** Ref ID for fourth ptemp destination */
   PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_3,

   /** Ref ID for transform feedback counter buffer load */
   PVR_XFB_INIT_REF_ID_COUNTER_BUFFER_LD_DEST,

   /** Ref ID for transform feedback counter buffer store */
   PVR_XFB_INIT_REF_ID_COUNTER_BUFFER_ST_SRC0,
};

/* PDSC XFB end patch IDs */
enum pvr_xfb_end_patch_id {
   /** Patch ID for end transform feedback counter buffer count */
   PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_COUNT,

   /** Patch ID for end transform feedback counter buffer stride 0 */
   PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_0,

   /** Patch ID for end transform feedback counter buffer stride 1 */
   PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_1,

   /** Patch ID for end transform feedback counter buffer stride 2 */
   PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_2,

   /** Patch ID for end transform feedback counter buffer stride 3 */
   PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_3,

   /** Patch ID for end transform feedback counter buffer load source 0 */
   PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_LD_SRC0,

   /** Patch ID for end transform feedback counter buffer store source 0 */
   PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_ST_SRC0,

   /** Patch ID for stream query result load source 0 */
   PVR_XFB_END_PATCH_ID_QUERY_RESULT_LD_SRC0,

   /** Patch ID for stream query result store source 0 */
   PVR_XFB_END_PATCH_ID_QUERY_RESULT_ST_SRC0,
};

/* PDSC XFB end reference IDs */
enum pvr_xfb_end_ref_id {
   /** Ref ID for end transform feedback counter buffer load destination */
   PVR_XFB_END_REF_ID_COUNTER_BUFFER_LD_DEST,

   /** Ref ID for end transform feedback counter buffer store source 0 */
   PVR_XFB_END_REF_ID_COUNTER_BUFFER_ST_SRC0,

   /** Ref ID for stream query result load destination */
   PVR_XFB_END_REF_ID_QUERY_RESULT_LD_DEST,

   /** Ref ID for stream query result store source 0 */
   PVR_XFB_END_REF_ID_QUERY_RESULT_ST_SRC0,
};

/* PDSC XFB vertex count patch IDs */
enum pvr_xfb_vertex_count_patch_id {
   /** Patch ID for vertex count negative counter offset */
   PVR_XFB_VERTEX_COUNT_PATCH_ID_NEGATIVE_COUNTER_OFFSET,

   /** Patch ID for vertex count vertex stride reciprocal */
   PVR_XFB_VERTEX_COUNT_PATCH_ID_VERTEX_STRIDE_RECIP,

   /** Patch ID for instance count */
   PVR_XFB_VERTEX_COUNT_PATCH_ID_INSTANCE_COUNT,

   /** Patch ID for first instance */
   PVR_XFB_VERTEX_COUNT_PATCH_ID_FIRST_INSTANCE,

   /** Patch ID for counter buffer load source 0 */
   PVR_XFB_VERTEX_COUNT_PATCH_ID_COUNTER_BUFFER_LD_SRC0,

   /** Patch ID for output count buffer store source 0 */
   PVR_XFB_VERTEX_COUNT_PATCH_ID_OUTPUT_COUNT_BUFFER_ST_SRC0,

   /** Patch ID for counter dword select within the 16-byte aligned load */
   PVR_XFB_VERTEX_COUNT_PATCH_ID_COUNTER_DWORD_SELECT,

   /** Patch ID for divide enable (0 when vertex stride == 1) */
   PVR_XFB_VERTEX_COUNT_PATCH_ID_DIVIDE_ENABLE,
};

/* PDSC XFB vertex count reference IDs */
enum pvr_xfb_vertex_count_ref_id {
   /** Ref ID for counter buffer load destination */
   PVR_XFB_VERTEX_COUNT_REF_ID_COUNTER_BUFFER_LD_DEST,

   /** Ref ID for output count buffer store source 0 */
   PVR_XFB_VERTEX_COUNT_REF_ID_OUTPUT_COUNT_BUFFER_ST_SRC0,
};

/* PDSC XFB query begin patch IDs */
enum pvr_xfb_query_begin_patch_id {
   /** Patch ID for query availability store source 0 */
   PVR_XFB_QUERY_BEGIN_PATCH_ID_AVAILABILITY_ST_SRC0,

   /** Patch ID for query result store source 0 */
   PVR_XFB_QUERY_BEGIN_PATCH_ID_RESULT_ST_SRC0,

   /** Patch ID for the open segment scale: -1 if transform feedback is
    * active, 0 otherwise.
    */
   PVR_XFB_QUERY_BEGIN_PATCH_ID_OPEN_SEGMENT_SCALE,
};

/* PDSC XFB query begin reference IDs */
enum pvr_xfb_query_begin_ref_id {
   /** Ref ID for query availability store source 0 */
   PVR_XFB_QUERY_BEGIN_REF_ID_AVAILABILITY_ST_SRC0,

   /** Ref ID for query result store source 0 */
   PVR_XFB_QUERY_BEGIN_REF_ID_RESULT_ST_SRC0,
};

/* PDSC XFB query end patch IDs */
enum pvr_xfb_query_end_patch_id {
   /** Patch ID for query count load source 0 */
   PVR_XFB_QUERY_END_PATCH_ID_QUERY_COUNT_LD_SRC0,

   /** Patch ID for query result written store source 0 */
   PVR_XFB_QUERY_END_PATCH_ID_RESULT_WRITTEN_ST_SRC0,

   /** Patch ID for query availability store source 0 */
   PVR_XFB_QUERY_END_PATCH_ID_AVAILABILITY_ST_SRC0,
};

/* PDSC XFB query end reference IDs */
enum pvr_xfb_query_end_ref_id {
   /** Ref ID for query count load destination */
   PVR_XFB_QUERY_END_REF_ID_QUERY_COUNT_LD_DEST,

   /** Ref ID for query result written store source 0 */
   PVR_XFB_QUERY_END_REF_ID_RESULT_WRITTEN_ST_SRC0,

   /** Ref ID for query availability store source 0 */
   PVR_XFB_QUERY_END_REF_ID_AVAILABILITY_ST_SRC0,
};

#endif /* PVR_IFACE_H */
