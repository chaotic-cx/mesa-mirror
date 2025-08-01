/**************************************************************************
 *
 * Copyright 2009 Younes Manton.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef PIPE_VIDEO_STATE_H
#define PIPE_VIDEO_STATE_H

#include "pipe/p_defines.h"
#include "util/format/u_formats.h"
#include "pipe/p_state.h"
#include "pipe/p_screen.h"
#include "util/u_hash_table.h"
#include "util/u_inlines.h"
#include "util/u_rect.h"
#include "util/u_dynarray.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIPE_H264_MAX_NUM_LIST_REF    32
#define PIPE_H264_MAX_DPB_SIZE        17
#define PIPE_H265_MAX_NUM_LIST_REF    15
#define PIPE_H265_MAX_DPB_SIZE        16
#define PIPE_H265_MAX_SLICES          600
#define PIPE_H264_MAX_REFERENCES      16
#define PIPE_H265_MAX_REFERENCES      15
#define PIPE_AV1_MAX_REFERENCES       8
#define PIPE_DEFAULT_FRAME_RATE_DEN   1
#define PIPE_DEFAULT_FRAME_RATE_NUM   30
#define PIPE_DEFAULT_INTRA_IDR_PERIOD 30
#define PIPE_H2645_EXTENDED_SAR       255
#define PIPE_ENC_ROI_REGION_NUM_MAX   32
#define PIPE_ENC_DIRTY_RECTS_NUM_MAX 256
#define PIPE_ENC_MOVE_RECTS_NUM_MAX 256
#define PIPE_ENC_MOVE_MAP_MAX_HINTS 31
#define PIPE_H2645_LIST_REF_INVALID_ENTRY 0xff
#define PIPE_H265_MAX_LONG_TERM_REF_PICS_SPS 32
#define PIPE_H265_MAX_LONG_TERM_PICS 16
#define PIPE_H265_MAX_DELTA_POC 48
#define PIPE_H265_MAX_NUM_LIST_REF 15
#define PIPE_H265_MAX_ST_REF_PIC_SETS 65
#define PIPE_H265_MAX_SUB_LAYERS 7
#define PIPE_AV1_MAX_DPB_SIZE 8
#define PIPE_AV1_REFS_PER_FRAME 7

/*
 * see table 6-12 in the spec
 */
enum pipe_mpeg12_picture_coding_type
{
   PIPE_MPEG12_PICTURE_CODING_TYPE_I = 0x01,
   PIPE_MPEG12_PICTURE_CODING_TYPE_P = 0x02,
   PIPE_MPEG12_PICTURE_CODING_TYPE_B = 0x03,
   PIPE_MPEG12_PICTURE_CODING_TYPE_D = 0x04
};

/*
 * see table 6-14 in the spec
 */
enum pipe_mpeg12_picture_structure
{
   PIPE_MPEG12_PICTURE_STRUCTURE_RESERVED = 0x00,
   PIPE_MPEG12_PICTURE_STRUCTURE_FIELD_TOP = 0x01,
   PIPE_MPEG12_PICTURE_STRUCTURE_FIELD_BOTTOM = 0x02,
   PIPE_MPEG12_PICTURE_STRUCTURE_FRAME = 0x03
};

/*
 * flags for macroblock_type, see section 6.3.17.1 in the spec
 */
enum pipe_mpeg12_macroblock_type
{
   PIPE_MPEG12_MB_TYPE_QUANT = 0x01,
   PIPE_MPEG12_MB_TYPE_MOTION_FORWARD = 0x02,
   PIPE_MPEG12_MB_TYPE_MOTION_BACKWARD = 0x04,
   PIPE_MPEG12_MB_TYPE_PATTERN = 0x08,
   PIPE_MPEG12_MB_TYPE_INTRA = 0x10
};

/*
 * flags for motion_type, see table 6-17 and 6-18 in the spec
 */
enum pipe_mpeg12_motion_type
{
   PIPE_MPEG12_MO_TYPE_RESERVED = 0x00,
   PIPE_MPEG12_MO_TYPE_FIELD = 0x01,
   PIPE_MPEG12_MO_TYPE_FRAME = 0x02,
   PIPE_MPEG12_MO_TYPE_16x8 = 0x02,
   PIPE_MPEG12_MO_TYPE_DUAL_PRIME = 0x03
};

/*
 * see section 6.3.17.1 and table 6-19 in the spec
 */
enum pipe_mpeg12_dct_type
{
   PIPE_MPEG12_DCT_TYPE_FRAME = 0,
   PIPE_MPEG12_DCT_TYPE_FIELD = 1
};

enum pipe_mpeg12_field_select
{
   PIPE_MPEG12_FS_FIRST_FORWARD = 0x01,
   PIPE_MPEG12_FS_FIRST_BACKWARD = 0x02,
   PIPE_MPEG12_FS_SECOND_FORWARD = 0x04,
   PIPE_MPEG12_FS_SECOND_BACKWARD = 0x08
};

enum pipe_h264_nal_unit_type
{
   PIPE_H264_NAL_SLICE = 1,
   PIPE_H264_NAL_IDR_SLICE= 5,
   PIPE_H264_NAL_SPS = 7,
   PIPE_H264_NAL_PPS = 8,
   PIPE_H264_NAL_AUD = 9,
   PIPE_H264_NAL_PREFIX = 14,
};

enum pipe_h264_slice_type
{
   PIPE_H264_SLICE_TYPE_P = 0x0,
   PIPE_H264_SLICE_TYPE_B = 0x1,
   PIPE_H264_SLICE_TYPE_I = 0x2,
   PIPE_H264_SLICE_TYPE_SP = 0x3,
   PIPE_H264_SLICE_TYPE_SI = 0x4
};

enum pipe_h265_nal_unit_type
{
   PIPE_H265_NAL_TRAIL_N = 0,
   PIPE_H265_NAL_TRAIL_R = 1,
   PIPE_H265_NAL_TSA_N = 2,
   PIPE_H265_NAL_TSA_R = 3,
   PIPE_H265_NAL_BLA_W_LP = 16,
   PIPE_H265_NAL_IDR_W_RADL = 19,
   PIPE_H265_NAL_IDR_N_LP = 20,
   PIPE_H265_NAL_CRA_NUT = 21,
   PIPE_H265_NAL_RSV_IRAP_VCL23 = 23,
   PIPE_H265_NAL_VPS = 32,
   PIPE_H265_NAL_SPS = 33,
   PIPE_H265_NAL_PPS = 34,
   PIPE_H265_NAL_AUD = 35,
   PIPE_H265_NAL_PREFIX_SEI = 39,
};

enum pipe_h265_slice_type
{
   /* Values match Table 7-7 in HEVC spec
    for Name association of slice_type */
   PIPE_H265_SLICE_TYPE_B = 0x0,
   PIPE_H265_SLICE_TYPE_P = 0x1,
   PIPE_H265_SLICE_TYPE_I = 0x2,
};

/* To be used on each encoding feature bit field */
enum pipe_enc_feature
{
   PIPE_ENC_FEATURE_NOT_SUPPORTED = 0x0,
   PIPE_ENC_FEATURE_SUPPORTED = 0x1,
   PIPE_ENC_FEATURE_REQUIRED = 0x2,
};

/* Same enum for h264/h265 */
enum pipe_h2645_enc_picture_type
{
   PIPE_H2645_ENC_PICTURE_TYPE_P = 0x00,
   PIPE_H2645_ENC_PICTURE_TYPE_B = 0x01,
   PIPE_H2645_ENC_PICTURE_TYPE_I = 0x02,
   PIPE_H2645_ENC_PICTURE_TYPE_IDR = 0x03,
   PIPE_H2645_ENC_PICTURE_TYPE_SKIP = 0x04
};

enum pipe_av1_enc_frame_type
{
   PIPE_AV1_ENC_FRAME_TYPE_KEY = 0x00,
   PIPE_AV1_ENC_FRAME_TYPE_INTER = 0x01,
   PIPE_AV1_ENC_FRAME_TYPE_INTRA_ONLY = 0x02,
   PIPE_AV1_ENC_FRAME_TYPE_SWITCH = 0x03
};

enum pipe_h2645_enc_rate_control_method
{
   PIPE_H2645_ENC_RATE_CONTROL_METHOD_DISABLE = 0x00,
   PIPE_H2645_ENC_RATE_CONTROL_METHOD_CONSTANT_SKIP = 0x01,
   PIPE_H2645_ENC_RATE_CONTROL_METHOD_VARIABLE_SKIP = 0x02,
   PIPE_H2645_ENC_RATE_CONTROL_METHOD_CONSTANT = 0x03,
   PIPE_H2645_ENC_RATE_CONTROL_METHOD_VARIABLE = 0x04,
   PIPE_H2645_ENC_RATE_CONTROL_METHOD_QUALITY_VARIABLE = 0x05
};

enum pipe_slice_buffer_placement_type
{
   /* whole slice is in the buffer */
   PIPE_SLICE_BUFFER_PLACEMENT_TYPE_WHOLE = 0x0,
   /* The beginning of the slice is in the buffer but the end is not */
   PIPE_SLICE_BUFFER_PLACEMENT_TYPE_BEGIN = 0x1,
   /* Neither beginning nor end of the slice is in the buffer */
   PIPE_SLICE_BUFFER_PLACEMENT_TYPE_MIDDLE = 0x2,
   /* end of the slice is in the buffer */
   PIPE_SLICE_BUFFER_PLACEMENT_TYPE_END = 0x3,
};

struct pipe_picture_desc
{
   enum pipe_video_profile profile;
   enum pipe_video_entrypoint entry_point;
   bool protected_playback;
   bool cenc;
   uint8_t *decrypt_key;
   uint32_t key_size;
   enum pipe_format input_format;
   bool input_full_range;
   enum pipe_format output_format;
   /* Flush flags for pipe_video_codec::end_frame */
   unsigned flush_flags;
   /* A fence for pipe_video_codec::begin_frame to wait on */
   struct pipe_fence_handle *in_fence;
   uint64_t in_fence_value;
   /* A fence for pipe_video_codec::end_frame to signal job completion */
   struct pipe_fence_handle **out_fence;
};

struct pipe_quant_matrix
{
   enum pipe_video_format codec;
};

struct pipe_macroblock
{
   enum pipe_video_format codec;
};

struct pipe_mpeg12_picture_desc
{
   struct pipe_picture_desc base;

   unsigned picture_coding_type;
   unsigned picture_structure;
   unsigned frame_pred_frame_dct;
   unsigned q_scale_type;
   unsigned alternate_scan;
   unsigned intra_vlc_format;
   unsigned concealment_motion_vectors;
   unsigned intra_dc_precision;
   unsigned f_code[2][2];
   unsigned top_field_first;
   unsigned full_pel_forward_vector;
   unsigned full_pel_backward_vector;
   unsigned num_slices;

   const uint8_t *intra_matrix;
   const uint8_t *non_intra_matrix;

   struct pipe_video_buffer *ref[2];
};

struct pipe_mpeg12_macroblock
{
   struct pipe_macroblock base;

   /* see section 6.3.17 in the spec */
   unsigned short x, y;

   /* see section 6.3.17.1 in the spec */
   unsigned char macroblock_type;

   union {
      struct {
         /* see table 6-17 in the spec */
         unsigned int frame_motion_type:2;

         /* see table 6-18 in the spec */
         unsigned int field_motion_type:2;

         /* see table 6-19 in the spec */
         unsigned int dct_type:1;
      } bits;
      unsigned int value;
   } macroblock_modes;

    /* see section 6.3.17.2 in the spec */
   unsigned char motion_vertical_field_select;

   /* see Table 7-7 in the spec */
   short PMV[2][2][2];

   /* see figure 6.10-12 in the spec */
   unsigned short coded_block_pattern;

   /* see figure 6.10-12 in the spec */
   short *blocks;

   /* Number of skipped macroblocks after this macroblock */
   unsigned short num_skipped_macroblocks;
};

struct pipe_mpeg4_picture_desc
{
   struct pipe_picture_desc base;

   int32_t trd[2];
   int32_t trb[2];
   uint16_t vop_time_increment_resolution;
   uint8_t vop_coding_type;
   uint8_t vop_fcode_forward;
   uint8_t vop_fcode_backward;
   uint8_t resync_marker_disable;
   uint8_t interlaced;
   uint8_t quant_type;
   uint8_t quarter_sample;
   uint8_t short_video_header;
   uint8_t rounding_control;
   uint8_t alternate_vertical_scan_flag;
   uint8_t top_field_first;

   const uint8_t *intra_matrix;
   const uint8_t *non_intra_matrix;

   struct pipe_video_buffer *ref[2];
};

struct pipe_vc1_picture_desc
{
   struct pipe_picture_desc base;

   uint32_t slice_count;
   uint8_t picture_type;
   uint8_t frame_coding_mode;
   uint8_t is_first_field;
   uint8_t postprocflag;
   uint8_t pulldown;
   uint8_t interlace;
   uint8_t tfcntrflag;
   uint8_t finterpflag;
   uint8_t psf;
   uint8_t dquant;
   uint8_t panscan_flag;
   uint8_t refdist_flag;
   uint8_t quantizer;
   uint8_t extended_mv;
   uint8_t extended_dmv;
   uint8_t overlap;
   uint8_t vstransform;
   uint8_t loopfilter;
   uint8_t fastuvmc;
   uint8_t range_mapy_flag;
   uint8_t range_mapy;
   uint8_t range_mapuv_flag;
   uint8_t range_mapuv;
   uint8_t multires;
   uint8_t syncmarker;
   uint8_t rangered;
   uint8_t maxbframes;
   uint8_t deblockEnable;
   uint8_t pquant;

   struct pipe_video_buffer *ref[2];
};

struct pipe_h264_sps
{
   uint8_t  level_idc;
   uint8_t  chroma_format_idc;
   uint8_t  separate_colour_plane_flag;
   uint8_t  bit_depth_luma_minus8;
   uint8_t  bit_depth_chroma_minus8;
   uint8_t  seq_scaling_matrix_present_flag;
   uint8_t  ScalingList4x4[6][16];
   uint8_t  ScalingList8x8[6][64];
   uint8_t  log2_max_frame_num_minus4;
   uint8_t  pic_order_cnt_type;
   uint8_t  log2_max_pic_order_cnt_lsb_minus4;
   uint8_t  delta_pic_order_always_zero_flag;
   int32_t  offset_for_non_ref_pic;
   int32_t  offset_for_top_to_bottom_field;
   uint8_t  num_ref_frames_in_pic_order_cnt_cycle;
   int32_t  offset_for_ref_frame[256];
   uint8_t  max_num_ref_frames;
   uint8_t  frame_mbs_only_flag;
   uint8_t  mb_adaptive_frame_field_flag;
   uint8_t  direct_8x8_inference_flag;
   uint8_t  MinLumaBiPredSize8x8;
   uint8_t  gaps_in_frame_num_value_allowed_flag;
   uint32_t pic_width_in_mbs_minus1;
   uint32_t pic_height_in_mbs_minus1;
};

struct pipe_h264_pps
{
   struct pipe_h264_sps *sps;

   uint8_t  entropy_coding_mode_flag;
   uint8_t  bottom_field_pic_order_in_frame_present_flag;
   uint8_t  num_slice_groups_minus1;
   uint8_t  slice_group_map_type;
   uint8_t  slice_group_change_rate_minus1;
   uint8_t  num_ref_idx_l0_default_active_minus1;
   uint8_t  num_ref_idx_l1_default_active_minus1;
   uint8_t  weighted_pred_flag;
   uint8_t  weighted_bipred_idc;
   int8_t   pic_init_qp_minus26;
   int8_t   pic_init_qs_minus26;
   int8_t   chroma_qp_index_offset;
   uint8_t  deblocking_filter_control_present_flag;
   uint8_t  constrained_intra_pred_flag;
   uint8_t  redundant_pic_cnt_present_flag;
   uint8_t  ScalingList4x4[6][16];
   uint8_t  ScalingList8x8[6][64];
   uint8_t  transform_8x8_mode_flag;
   int8_t   second_chroma_qp_index_offset;
};

struct pipe_h264_picture_desc
{
   struct pipe_picture_desc base;

   struct pipe_h264_pps *pps;

   /* slice header */
   uint32_t frame_num;
   uint8_t  field_pic_flag;
   uint8_t  bottom_field_flag;
   uint8_t  num_ref_idx_l0_active_minus1;
   uint8_t  num_ref_idx_l1_active_minus1;

   uint32_t slice_count;
   int32_t  field_order_cnt[2];
   bool     is_reference;
   uint8_t  num_ref_frames;

   bool     is_long_term[16];
   bool     top_is_reference[16];
   bool     bottom_is_reference[16];
   uint32_t field_order_cnt_list[16][2];
   uint32_t frame_num_list[16];

   struct pipe_video_buffer *ref[16];

   struct
   {
      bool slice_info_present;
      uint8_t slice_type[128];
      uint32_t slice_data_size[128];
      uint32_t slice_data_offset[128];
      enum pipe_slice_buffer_placement_type slice_data_flag[128];
   } slice_parameter;
};

struct pipe_enc_quality_modes
{
   unsigned int level;
   unsigned int preset_mode;
   unsigned int pre_encode_mode;
   unsigned int vbaq_mode;
};

/*
 * intra refresh supports row or column only, it doens't support
 * row and column mixed, if mixed it will pick up column mode.
 * Also the assumption is the first row/column since the offset
 * is zero, and it marks the start of intra-refresh, it will need
 * to have headers at this point.
 */
struct pipe_enc_intra_refresh
{
   unsigned int mode;
   unsigned int region_size;
   unsigned int offset;
   unsigned int need_sequence_header;
};

/*
 * In AVC, unit is MB, HEVC (CTB) and AV1(SB)
 */
enum
{
   INTRA_REFRESH_MODE_NONE,
   INTRA_REFRESH_MODE_UNIT_ROWS,
   INTRA_REFRESH_MODE_UNIT_COLUMNS,
};

/* All the values are in pixels, driver converts it into
 * different units for different codecs, for example: h264
 * is in 16x16 block, hevc/av1 is in 64x64 block.
 * x, y means the location of region start, width/height defines
 * the region size; the qp value carries the qp_delta.
 */
struct pipe_enc_region_in_roi
{
   bool    valid;
   int32_t qp_value;
   unsigned int x, y;
   unsigned int width, height;
};
/* It does not support prioirty only qp_delta.
 * The priority is implied by the region sequence number.
 * Region 0 is most significant one, and region 1 is less
 * significant, and lesser significant when region number
 * grows. It allows region overlapping, and lower
 * priority region would be overwritten by the higher one.
 */
struct pipe_enc_roi
{
   unsigned int num;
   struct pipe_enc_region_in_roi region[PIPE_ENC_ROI_REGION_NUM_MAX];
};

enum pipe_enc_dirty_info_type
{
   PIPE_ENC_DIRTY_INFO_TYPE_DIRTY = 0,
   PIPE_ENC_DIRTY_INFO_TYPE_SKIP = 1,
};

enum pipe_enc_dirty_info_input_mode
{
   /* Requires PIPE_VIDEO_CAP_ENC_DIRTY_RECTS */
   PIPE_ENC_DIRTY_INFO_INPUT_MODE_RECTS = 0,
   /* Requires PIPE_VIDEO_CAP_ENC_DIRTY_MAPS */
   PIPE_ENC_DIRTY_INFO_INPUT_MODE_MAP = 1,
};

struct pipe_enc_dirty_info
{
   uint8_t dpb_reference_index; /* index in dpb for the recon pic the info refers to */
   bool full_frame_skip;
   enum pipe_enc_dirty_info_type dirty_info_type;

   /* Selects the input mode and app sets different params below */
   enum pipe_enc_dirty_info_input_mode input_mode;

   /* Used with PIPE_ENC_DIRTY_INFO_INPUT_MODE_RECTS */
   unsigned int num_rects;
   struct
   {
      uint32_t top;
      uint32_t left;
      uint32_t right;
      uint32_t bottom;
   } rects[PIPE_ENC_DIRTY_RECTS_NUM_MAX];

   /* Used with PIPE_ENC_DIRTY_INFO_INPUT_MODE_MAP */
   struct pipe_resource *map;
};

enum pipe_enc_move_info_precision_unit
{
   PIPE_ENC_MOVE_INFO_PRECISION_UNIT_FULL_PIXEL = 0,
   PIPE_ENC_MOVE_INFO_PRECISION_UNIT_HALF_PIXEL = 1,
   PIPE_ENC_MOVE_INFO_PRECISION_UNIT_QUARTER_PIXEL = 2,
};

enum pipe_enc_move_info_input_mode
{
   PIPE_ENC_MOVE_INFO_INPUT_MODE_DISABLED = 0,
   /* Requires PIPE_VIDEO_CAP_ENC_MOVE_RECTS */
   PIPE_ENC_MOVE_INFO_INPUT_MODE_RECTS = 1,
   /* Requires PIPE_VIDEO_CAP_ENC_MOTION_VECTOR_MAPS */
   PIPE_ENC_MOVE_INFO_INPUT_MODE_MAP = 2,
};

struct pipe_enc_move_info
{
   enum pipe_enc_move_info_input_mode input_mode;

   /* Used with PIPE_ENC_MOVE_INFO_INPUT_MODE_MAP */
   /* Contains the motion hints */
   struct pipe_resource *gpu_motion_vectors_map[PIPE_ENC_MOVE_MAP_MAX_HINTS];
   /* contains the DPB index the motion hints apply to, or 255 if no motion hint */
   struct pipe_resource *gpu_motion_metadata_map[PIPE_ENC_MOVE_MAP_MAX_HINTS];
   /* Indicates how many entries to gpu_motion_vectors_map and gpu_motion_metadata_map */
   uint32_t num_hints;

   /* Used with PIPE_ENC_MOVE_INFO_INPUT_MODE_RECTS */
   unsigned int num_rects;
   struct
   {
      struct {
         uint32_t x;
         uint32_t y;
      } source_point;
      struct {
         uint32_t top;
         uint32_t left;
         uint32_t right;
         uint32_t bottom;
      } dest_rect;
   } rects[PIPE_ENC_MOVE_RECTS_NUM_MAX];
   uint8_t dpb_reference_index; /* index in dpb for the recon pic the rects refer to */
   enum pipe_enc_move_info_precision_unit precision;
   bool overlapping_rects;
};

struct pipe_enc_two_pass_encoder_config
{
   /*
    * Enables two pass encoding for all supported
      frames during the lifetime of the encoder
   */
   bool enable;

   /*
    * Indicates the downscaling factor for the first pass
    *
    * Please check pipe_cap_two_pass for the supported downscaling factors
    *
    * pow2_downscale_factor | 1st pass width   | 1st pass height
    * 0                     | InputWidth       | InputHeight
    * 1                     | InputWidth / 2   | InputHeight / 2
    * 2                     | InputWidth / 4   | InputHeight / 4
    * n                     | InputWidth / 2^n | InputHeight / 2^n
   */
   unsigned pow2_downscale_factor;

   /*
   * When set, the first pass (low resolution) dpb texture output
   * will not be written by the driver. It is responsability of the
   * app to downscale the 2nd pass (full resolution) dpb recon texture into
   * a downscaled recon dpb texture to be used as a downscaled dpb
   * reference in future encoded frames.
   *
   * When not set, the driver writes the 1st pass output recon dpb
   * texture and that can be directly used as a downscaled dpb
   * reference in future encoded frames.
   *
   * Please check pipe_cap_two_pass.supports_1pass_recon_writing_skip
   */
   bool skip_1st_dpb_texture;
};

/*
 * Used when pipe_enc_two_pass_encoder_config
 * is enabled in the encoder
*/
struct pipe_enc_two_pass_frame_config
{
   /*
    * Downscaled version of the source buffer to be encoded
    * input to the 1st encoder pass
   */
   struct pipe_video_buffer *downscaled_source;

   /*
    * Disabled two pass for this frame
    *
    * Requires pipe_cap_two_pass.supports_dynamic_1st_pass_skip
    *
   */
   bool skip_1st_pass;
};

struct pipe_enc_raw_header
{
   uint8_t type; /* nal_unit_type or obu_type */
   bool is_slice; /* slice or frame header */
   uint32_t size;
   uint8_t *buffer;
};

struct pipe_h264_enc_rate_control
{
   enum pipe_h2645_enc_rate_control_method rate_ctrl_method;
   unsigned target_bitrate;
   unsigned peak_bitrate;
   unsigned frame_rate_num;
   unsigned frame_rate_den;
   unsigned vbv_buffer_size;
   unsigned vbv_buf_lv;
   unsigned vbv_buf_initial_size;
   bool app_requested_hrd_buffer;
   unsigned fill_data_enable;
   unsigned skip_frame_enable;
   unsigned enforce_hrd;
   unsigned max_au_size;
   unsigned max_qp;
   unsigned min_qp;
   bool app_requested_qp_range;

   /* Used with PIPE_H2645_ENC_RATE_CONTROL_METHOD_QUALITY_VARIABLE */
   unsigned vbr_quality_factor;
};

struct pipe_h264_enc_motion_estimation
{
   unsigned motion_est_quarter_pixel;
   unsigned enc_disable_sub_mode;
   unsigned lsmvert;
   unsigned enc_en_ime_overw_dis_subm;
   unsigned enc_ime_overw_dis_subm_no;
   unsigned enc_ime2_search_range_x;
   unsigned enc_ime2_search_range_y;
};

struct pipe_h264_enc_pic_control
{
   unsigned enc_cabac_enable;
   unsigned enc_cabac_init_idc;
   struct {
      uint32_t entropy_coding_mode_flag : 1;
      uint32_t weighted_pred_flag : 1;
      uint32_t deblocking_filter_control_present_flag : 1;
      uint32_t constrained_intra_pred_flag : 1;
      uint32_t redundant_pic_cnt_present_flag : 1;
      uint32_t more_rbsp_data : 1;
      uint32_t transform_8x8_mode_flag : 1;
   };
   uint8_t nal_ref_idc;
   uint8_t nal_unit_type;
   uint8_t num_ref_idx_l0_default_active_minus1;
   uint8_t num_ref_idx_l1_default_active_minus1;
   uint8_t weighted_bipred_idc;
   int8_t pic_init_qp_minus26;
   int8_t pic_init_qs_minus26;
   int8_t chroma_qp_index_offset;
   int8_t second_chroma_qp_index_offset;
   uint8_t temporal_id;
};

struct pipe_h264_enc_dbk_param
{
   unsigned  disable_deblocking_filter_idc;
   signed   alpha_c0_offset_div2;
   signed   beta_offset_div2;
};

struct h264_slice_descriptor
{
   /** Starting MB address for this slice. */
   uint32_t    macroblock_address;
   /** Number of macroblocks in this slice. */
   uint32_t    num_macroblocks;
   /** slice type. */
   enum pipe_h264_slice_type slice_type;
};

struct h265_slice_descriptor
{
   /** Starting CTU address for this slice. */
   uint32_t    slice_segment_address;
   /** Number of CTUs in this slice. */
   uint32_t    num_ctu_in_slice;
   /** slice type. */
   enum pipe_h265_slice_type slice_type;
};

struct pipe_enc_hdr_cll {
   uint16_t max_cll;
   uint16_t max_fall;
};

struct pipe_enc_hdr_mdcv {
   uint16_t primary_chromaticity_x[3];
   uint16_t primary_chromaticity_y[3];
   uint16_t white_point_chromaticity_x;
   uint16_t white_point_chromaticity_y;
   uint32_t luminance_max;
   uint32_t luminance_min;
};

typedef struct pipe_h264_enc_hrd_params
{
   uint32_t cpb_cnt_minus1;
   uint32_t bit_rate_scale;
   uint32_t cpb_size_scale;
   uint32_t bit_rate_value_minus1[32];
   uint32_t cpb_size_value_minus1[32];
   uint32_t cbr_flag[32];
   uint32_t initial_cpb_removal_delay_length_minus1;
   uint32_t cpb_removal_delay_length_minus1;
   uint32_t dpb_output_delay_length_minus1;
   uint32_t time_offset_length;
} pipe_h264_enc_hrd_params;

struct pipe_h264_enc_seq_param
{
   struct {
      uint32_t enc_frame_cropping_flag : 1;
      uint32_t vui_parameters_present_flag : 1;
      uint32_t video_full_range_flag : 1;
      uint32_t direct_8x8_inference_flag : 1;
      uint32_t gaps_in_frame_num_value_allowed_flag : 1;
   };
   unsigned profile_idc;
   unsigned enc_constraint_set_flags;
   unsigned level_idc;
   unsigned bit_depth_luma_minus8;
   unsigned bit_depth_chroma_minus8;
   unsigned enc_frame_crop_left_offset;
   unsigned enc_frame_crop_right_offset;
   unsigned enc_frame_crop_top_offset;
   unsigned enc_frame_crop_bottom_offset;
   unsigned pic_order_cnt_type;
   unsigned log2_max_frame_num_minus4;
   unsigned log2_max_pic_order_cnt_lsb_minus4;
   unsigned num_temporal_layers;
   struct {
      uint32_t aspect_ratio_info_present_flag: 1;
      uint32_t timing_info_present_flag: 1;
      uint32_t video_signal_type_present_flag: 1;
      uint32_t colour_description_present_flag: 1;
      uint32_t chroma_loc_info_present_flag: 1;
      uint32_t overscan_info_present_flag: 1;
      uint32_t overscan_appropriate_flag: 1;
      uint32_t fixed_frame_rate_flag: 1;
      uint32_t nal_hrd_parameters_present_flag: 1;
      uint32_t vcl_hrd_parameters_present_flag: 1;
      uint32_t low_delay_hrd_flag: 1;
      uint32_t pic_struct_present_flag: 1;
      uint32_t bitstream_restriction_flag: 1;
      uint32_t motion_vectors_over_pic_boundaries_flag: 1;
   } vui_flags;
   uint32_t aspect_ratio_idc;
   uint32_t sar_width;
   uint32_t sar_height;
   uint32_t num_units_in_tick;
   uint32_t time_scale;
   uint32_t video_format;
   uint32_t colour_primaries;
   uint32_t transfer_characteristics;
   uint32_t matrix_coefficients;
   uint32_t chroma_sample_loc_type_top_field;
   uint32_t chroma_sample_loc_type_bottom_field;
   uint32_t max_num_reorder_frames;
   pipe_h264_enc_hrd_params nal_hrd_parameters;
   pipe_h264_enc_hrd_params vcl_hrd_parameters;
   uint32_t max_bytes_per_pic_denom;
   uint32_t max_bits_per_mb_denom;
   uint32_t log2_max_mv_length_vertical;
   uint32_t log2_max_mv_length_horizontal;
   uint32_t max_dec_frame_buffering;
   uint32_t max_num_ref_frames;
   uint32_t pic_width_in_mbs_minus1;
   uint32_t pic_height_in_map_units_minus1;
};

struct pipe_h264_ref_list_mod_entry
{
   uint8_t modification_of_pic_nums_idc;
   uint32_t abs_diff_pic_num_minus1;
   uint32_t long_term_pic_num;
};

struct pipe_h264_ref_pic_marking_entry
{
   uint8_t memory_management_control_operation;
   uint32_t difference_of_pic_nums_minus1;
   uint32_t long_term_pic_num;
   uint32_t long_term_frame_idx;
   uint32_t max_long_term_frame_idx_plus1;
};

struct pipe_h264_enc_slice_param
{
   struct {
      uint32_t direct_spatial_mv_pred_flag : 1;
      uint32_t num_ref_idx_active_override_flag : 1;
      uint32_t ref_pic_list_modification_flag_l0 : 1;
      uint32_t ref_pic_list_modification_flag_l1 : 1;
      uint32_t no_output_of_prior_pics_flag : 1;
      uint32_t long_term_reference_flag : 1;
      uint32_t adaptive_ref_pic_marking_mode_flag : 1;
   };
   uint8_t slice_type;
   uint8_t colour_plane_id;
   uint32_t frame_num;
   uint32_t idr_pic_id;
   uint32_t pic_order_cnt_lsb;
   uint8_t redundant_pic_cnt;
   uint8_t num_ref_idx_l0_active_minus1;
   uint8_t num_ref_idx_l1_active_minus1;
   uint8_t num_ref_list0_mod_operations;
   struct pipe_h264_ref_list_mod_entry ref_list0_mod_operations[PIPE_H264_MAX_NUM_LIST_REF];
   uint8_t num_ref_list1_mod_operations;
   struct pipe_h264_ref_list_mod_entry ref_list1_mod_operations[PIPE_H264_MAX_NUM_LIST_REF];
   uint8_t num_ref_pic_marking_operations;
   struct pipe_h264_ref_pic_marking_entry ref_pic_marking_operations[PIPE_H264_MAX_NUM_LIST_REF];
   uint8_t cabac_init_idc;
   int32_t slice_qp_delta;
   uint8_t disable_deblocking_filter_idc;
   int32_t slice_alpha_c0_offset_div2;
   int32_t slice_beta_offset_div2;
};

struct pipe_h264_enc_dpb_entry
{
   uint32_t id;
   uint32_t frame_idx;
   uint32_t pic_order_cnt;
   uint32_t temporal_id;
   bool is_ltr;
   struct pipe_video_buffer *buffer;
   struct pipe_video_buffer *downscaled_buffer; /* for two pass */
   bool evict;
   enum pipe_h2645_enc_picture_type picture_type;
};

struct pipe_h264_enc_picture_desc
{
   struct pipe_picture_desc base;

   struct pipe_h264_enc_seq_param seq;
   struct pipe_h264_enc_slice_param slice;
   struct pipe_h264_enc_pic_control pic_ctrl;
   struct pipe_h264_enc_rate_control rate_ctrl[4];

   struct pipe_h264_enc_motion_estimation motion_est;
   struct pipe_h264_enc_dbk_param dbk;

   unsigned intra_idr_period;
   unsigned ip_period;

   unsigned init_qp;
   unsigned quant_i_frames;
   unsigned quant_p_frames;
   unsigned quant_b_frames;

   enum pipe_h2645_enc_picture_type picture_type;
   unsigned frame_num;
   unsigned frame_num_cnt;
   unsigned p_remain;
   unsigned i_remain;
   unsigned idr_pic_id;
   unsigned gop_cnt;
   unsigned pic_order_cnt;
   unsigned num_ref_idx_l0_active_minus1;
   unsigned num_ref_idx_l1_active_minus1;
   unsigned ref_idx_l0_list[PIPE_H264_MAX_NUM_LIST_REF];
   bool l0_is_long_term[PIPE_H264_MAX_NUM_LIST_REF];
   unsigned ref_idx_l1_list[PIPE_H264_MAX_NUM_LIST_REF];
   bool l1_is_long_term[PIPE_H264_MAX_NUM_LIST_REF];
   unsigned gop_size;
   struct pipe_enc_quality_modes quality_modes;
   struct pipe_enc_intra_refresh intra_refresh;
   struct pipe_enc_roi roi;
   struct pipe_enc_dirty_info dirty_info;
   struct pipe_enc_move_info move_info;

   /* See PIPE_VIDEO_CAP_ENC_GPU_STATS_QP_MAP */
   struct pipe_resource *gpu_stats_qp_map;
   /* See PIPE_VIDEO_CAP_ENC_GPU_STATS_SATD_MAP */
   struct pipe_resource *gpu_stats_satd_map;
   /* See PIPE_VIDEO_CAP_ENC_GPU_STATS_RATE_CONTROL_BITS_MAP */
   struct pipe_resource *gpu_stats_rc_bitallocation_map;
   /* See PIPE_VIDEO_CAP_ENC_GPU_STATS_PSNR */
   struct pipe_resource *gpu_stats_psnr;

   /* See PIPE_VIDEO_CAP_ENC_QP_MAPS */
   struct pipe_resource *input_gpu_qpmap;

   bool not_referenced;
   bool is_ltr;
   unsigned ltr_index;
   bool enable_vui;
   struct hash_table *frame_idx;

   enum pipe_video_slice_mode slice_mode;

   /* Use with PIPE_VIDEO_SLICE_MODE_BLOCKS */
   unsigned num_slice_descriptors;
   struct h264_slice_descriptor slices_descriptors[128];

   /* Use with PIPE_VIDEO_SLICE_MODE_MAX_SLICE_SIZE */
   unsigned max_slice_bytes;

   enum pipe_video_feedback_metadata_type requested_metadata;

   struct pipe_h264_enc_dpb_entry dpb[PIPE_H264_MAX_DPB_SIZE];
   uint8_t dpb_size;
   uint8_t dpb_curr_pic; /* index in dpb */
   uint8_t ref_list0[PIPE_H264_MAX_NUM_LIST_REF]; /* index in dpb, PIPE_H2645_LIST_REF_INVALID_ENTRY invalid */
   uint8_t ref_list1[PIPE_H264_MAX_NUM_LIST_REF]; /* index in dpb, PIPE_H2645_LIST_REF_INVALID_ENTRY invalid */

   struct util_dynarray raw_headers; /* struct pipe_enc_raw_header */

   struct pipe_enc_two_pass_frame_config twopass_frame_config;
};

struct pipe_h265_st_ref_pic_set
{
   struct {
      uint32_t inter_ref_pic_set_prediction_flag : 1;
   };
   uint32_t delta_idx_minus1;
   uint8_t delta_rps_sign;
   uint16_t abs_delta_rps_minus1;
   uint8_t used_by_curr_pic_flag[PIPE_H265_MAX_DPB_SIZE];
   uint8_t use_delta_flag[PIPE_H265_MAX_DPB_SIZE];
   uint8_t num_negative_pics;
   uint8_t num_positive_pics;
   uint16_t delta_poc_s0_minus1[PIPE_H265_MAX_DPB_SIZE];
   uint8_t used_by_curr_pic_s0_flag[PIPE_H265_MAX_DPB_SIZE];
   uint16_t delta_poc_s1_minus1[PIPE_H265_MAX_DPB_SIZE];
   uint8_t used_by_curr_pic_s1_flag[PIPE_H265_MAX_DPB_SIZE];
};

struct pipe_h265_ref_pic_lists_modification
{
   struct {
      uint32_t ref_pic_list_modification_flag_l0 : 1;
      uint32_t ref_pic_list_modification_flag_l1 : 1;
   };
   uint8_t list_entry_l0[PIPE_H265_MAX_NUM_LIST_REF];
   uint8_t list_entry_l1[PIPE_H265_MAX_NUM_LIST_REF];
};

struct pipe_h265_enc_sublayer_hrd_params
{
    uint32_t bit_rate_value_minus1[32];
    uint32_t cpb_size_value_minus1[32];
    uint32_t cpb_size_du_value_minus1[32];
    uint32_t bit_rate_du_value_minus1[32];
    uint32_t cbr_flag[32];
};

struct pipe_h265_enc_hrd_params
{
   uint32_t nal_hrd_parameters_present_flag;
   uint32_t vcl_hrd_parameters_present_flag;
   uint32_t sub_pic_hrd_params_present_flag;
   uint32_t tick_divisor_minus2;
   uint32_t du_cpb_removal_delay_increment_length_minus1;
   uint32_t sub_pic_cpb_params_in_pic_timing_sei_flag;
   uint32_t dpb_output_delay_du_length_minus1;
   uint32_t bit_rate_scale;
   uint32_t cpb_rate_scale;
   uint32_t cpb_size_du_scale;
   uint32_t initial_cpb_removal_delay_length_minus1;
   uint32_t au_cpb_removal_delay_length_minus1;
   uint32_t dpb_output_delay_length_minus1;
   uint32_t fixed_pic_rate_general_flag[PIPE_H265_MAX_SUB_LAYERS];
   uint32_t fixed_pic_rate_within_cvs_flag[PIPE_H265_MAX_SUB_LAYERS];
   uint32_t elemental_duration_in_tc_minus1[PIPE_H265_MAX_SUB_LAYERS];
   uint32_t low_delay_hrd_flag[PIPE_H265_MAX_SUB_LAYERS];
   uint32_t cpb_cnt_minus1[PIPE_H265_MAX_SUB_LAYERS];
   struct pipe_h265_enc_sublayer_hrd_params nal_hrd_parameters[PIPE_H265_MAX_SUB_LAYERS];
   struct pipe_h265_enc_sublayer_hrd_params vlc_hrd_parameters[PIPE_H265_MAX_SUB_LAYERS];
};

struct pipe_h265_profile_tier
{
   struct {
      uint32_t general_tier_flag : 1;
      uint32_t general_progressive_source_flag : 1;
      uint32_t general_interlaced_source_flag : 1;
      uint32_t general_non_packed_constraint_flag : 1;
      uint32_t general_frame_only_constraint_flag : 1;
   };
   uint8_t general_profile_space;
   uint8_t general_profile_idc;
   uint32_t general_profile_compatibility_flag;
};

struct pipe_h265_profile_tier_level
{
   uint8_t general_level_idc;
   uint8_t sub_layer_profile_present_flag[PIPE_H265_MAX_SUB_LAYERS];
   uint8_t sub_layer_level_present_flag[PIPE_H265_MAX_SUB_LAYERS];
   uint8_t sub_layer_level_idc[PIPE_H265_MAX_SUB_LAYERS];
   struct pipe_h265_profile_tier profile_tier;
   struct pipe_h265_profile_tier sub_layer_profile_tier[PIPE_H265_MAX_SUB_LAYERS];
};

struct pipe_h265_enc_vid_param
{
   struct {
      uint32_t vps_base_layer_internal_flag : 1;
      uint32_t vps_base_layer_available_flag : 1;
      uint32_t vps_temporal_id_nesting_flag : 1;
      uint32_t vps_sub_layer_ordering_info_present_flag : 1;
      uint32_t vps_timing_info_present_flag : 1;
      uint32_t vps_poc_proportional_to_timing_flag : 1;
   };
   uint8_t vps_max_layers_minus1;
   uint8_t vps_max_sub_layers_minus1;
   uint8_t vps_max_dec_pic_buffering_minus1[PIPE_H265_MAX_SUB_LAYERS];
   uint8_t vps_max_num_reorder_pics[PIPE_H265_MAX_SUB_LAYERS];
   uint32_t vps_max_latency_increase_plus1[PIPE_H265_MAX_SUB_LAYERS];
   uint8_t vps_max_layer_id;
   uint32_t vps_num_layer_sets_minus1;
   uint32_t vps_num_units_in_tick;
   uint32_t vps_time_scale;
   uint32_t vps_num_ticks_poc_diff_one_minus1;
   struct pipe_h265_profile_tier_level profile_tier_level;
};

struct pipe_h265_enc_seq_param
{
   struct {
      uint32_t sps_temporal_id_nesting_flag : 1;
      uint32_t strong_intra_smoothing_enabled_flag : 1;
      uint32_t amp_enabled_flag : 1;
      uint32_t sample_adaptive_offset_enabled_flag : 1;
      uint32_t pcm_enabled_flag : 1;
      uint32_t sps_temporal_mvp_enabled_flag : 1;
      uint32_t conformance_window_flag : 1;
      uint32_t vui_parameters_present_flag : 1;
      uint32_t video_full_range_flag : 1;
      uint32_t long_term_ref_pics_present_flag : 1;
      uint32_t sps_sub_layer_ordering_info_present_flag : 1;
   };
   uint8_t  general_profile_idc;
   uint8_t  general_level_idc;
   uint8_t  general_tier_flag;
   uint32_t intra_period;
   uint32_t ip_period;
   uint16_t pic_width_in_luma_samples;
   uint16_t pic_height_in_luma_samples;
   uint32_t chroma_format_idc;
   uint32_t bit_depth_luma_minus8;
   uint32_t bit_depth_chroma_minus8;
   uint8_t  log2_max_pic_order_cnt_lsb_minus4;
   uint8_t  log2_min_luma_coding_block_size_minus3;
   uint8_t  log2_diff_max_min_luma_coding_block_size;
   uint8_t  log2_min_transform_block_size_minus2;
   uint8_t  log2_diff_max_min_transform_block_size;
   uint8_t  max_transform_hierarchy_depth_inter;
   uint8_t  max_transform_hierarchy_depth_intra;
   uint16_t conf_win_left_offset;
   uint16_t conf_win_right_offset;
   uint16_t conf_win_top_offset;
   uint16_t conf_win_bottom_offset;
   struct {
      uint32_t aspect_ratio_info_present_flag: 1;
      uint32_t timing_info_present_flag: 1;
      uint32_t video_signal_type_present_flag: 1;
      uint32_t colour_description_present_flag: 1;
      uint32_t chroma_loc_info_present_flag: 1;
      uint32_t overscan_info_present_flag: 1;
      uint32_t overscan_appropriate_flag: 1;
      uint32_t neutral_chroma_indication_flag: 1;
      uint32_t field_seq_flag: 1;
      uint32_t frame_field_info_present_flag: 1;
      uint32_t default_display_window_flag: 1;
      uint32_t poc_proportional_to_timing_flag: 1;
      uint32_t hrd_parameters_present_flag: 1;
      uint32_t bitstream_restriction_flag: 1;
      uint32_t tiles_fixed_structure_flag: 1;
      uint32_t motion_vectors_over_pic_boundaries_flag: 1;
      uint32_t restricted_ref_pic_lists_flag: 1;
   } vui_flags;
   uint32_t aspect_ratio_idc;
   uint32_t sar_width;
   uint32_t sar_height;
   uint32_t num_units_in_tick;
   uint32_t time_scale;
   uint32_t video_format;
   uint32_t colour_primaries;
   uint32_t transfer_characteristics;
   uint32_t matrix_coefficients;
   uint32_t chroma_sample_loc_type_top_field;
   uint32_t chroma_sample_loc_type_bottom_field;
   uint32_t def_disp_win_left_offset;
   uint32_t def_disp_win_right_offset;
   uint32_t def_disp_win_top_offset;
   uint32_t def_disp_win_bottom_offset;
   uint32_t num_ticks_poc_diff_one_minus1;
   uint32_t min_spatial_segmentation_idc;
   uint32_t max_bytes_per_pic_denom;
   uint32_t max_bits_per_min_cu_denom;
   uint32_t log2_max_mv_length_horizontal;
   uint32_t log2_max_mv_length_vertical;
   uint32_t num_temporal_layers;
   uint32_t num_short_term_ref_pic_sets;
   uint32_t num_long_term_ref_pics_sps;
   uint32_t lt_ref_pic_poc_lsb_sps[PIPE_H265_MAX_LONG_TERM_REF_PICS_SPS];
   uint8_t used_by_curr_pic_lt_sps_flag[PIPE_H265_MAX_LONG_TERM_REF_PICS_SPS];
   uint8_t sps_max_sub_layers_minus1;
   uint8_t sps_max_dec_pic_buffering_minus1[PIPE_H265_MAX_SUB_LAYERS];
   uint8_t sps_max_num_reorder_pics[PIPE_H265_MAX_SUB_LAYERS];
   uint32_t sps_max_latency_increase_plus1[PIPE_H265_MAX_SUB_LAYERS];
   struct pipe_h265_profile_tier_level profile_tier_level;
   struct pipe_h265_enc_hrd_params hrd_parameters;
   struct pipe_h265_st_ref_pic_set st_ref_pic_set[PIPE_H265_MAX_ST_REF_PIC_SETS];
   struct {
      uint32_t sps_range_extension_flag;
      uint32_t transform_skip_rotation_enabled_flag: 1;
      uint32_t transform_skip_context_enabled_flag: 1;
      uint32_t implicit_rdpcm_enabled_flag: 1;
      uint32_t explicit_rdpcm_enabled_flag: 1;
      uint32_t extended_precision_processing_flag: 1;
      uint32_t intra_smoothing_disabled_flag: 1;
      uint32_t high_precision_offsets_enabled_flag: 1;
      uint32_t persistent_rice_adaptation_enabled_flag: 1;
      uint32_t cabac_bypass_alignment_enabled_flag: 1;
   } sps_range_extension;
   uint8_t separate_colour_plane_flag;
};

struct pipe_h265_enc_pic_param
{
   struct {
      uint32_t dependent_slice_segments_enabled_flag : 1;
      uint32_t output_flag_present_flag : 1;
      uint32_t sign_data_hiding_enabled_flag : 1;
      uint32_t cabac_init_present_flag : 1;
      uint32_t constrained_intra_pred_flag : 1;
      uint32_t transform_skip_enabled_flag : 1;
      uint32_t cu_qp_delta_enabled_flag : 1;
      uint32_t weighted_pred_flag : 1;
      uint32_t weighted_bipred_flag : 1;
      uint32_t transquant_bypass_enabled_flag : 1;
      uint32_t entropy_coding_sync_enabled_flag : 1;
      uint32_t pps_slice_chroma_qp_offsets_present_flag : 1;
      uint32_t pps_loop_filter_across_slices_enabled_flag : 1;
      uint32_t deblocking_filter_control_present_flag : 1;
      uint32_t deblocking_filter_override_enabled_flag : 1;
      uint32_t pps_deblocking_filter_disabled_flag : 1;
      uint32_t lists_modification_present_flag : 1;
   };
   uint8_t log2_parallel_merge_level_minus2;
   uint8_t nal_unit_type;
   uint8_t temporal_id;
   uint8_t num_extra_slice_header_bits;
   uint8_t num_ref_idx_l0_default_active_minus1;
   uint8_t num_ref_idx_l1_default_active_minus1;
   int8_t init_qp_minus26;
   uint8_t diff_cu_qp_delta_depth;
   int8_t pps_cb_qp_offset;
   int8_t pps_cr_qp_offset;
   int8_t pps_beta_offset_div2;
   int8_t pps_tc_offset_div2;
   struct {
      uint8_t pps_range_extension_flag;
      uint32_t log2_max_transform_skip_block_size_minus2;
      uint32_t cross_component_prediction_enabled_flag: 1;
      uint32_t chroma_qp_offset_list_enabled_flag: 1;
      uint32_t diff_cu_chroma_qp_offset_depth;
      uint32_t chroma_qp_offset_list_len_minus1;
      int32_t cb_qp_offset_list[6];
      int32_t cr_qp_offset_list[6];
      uint32_t log2_sao_offset_scale_luma;
      uint32_t log2_sao_offset_scale_chroma;
   } pps_range_extension;
};

struct pipe_h265_enc_slice_param
{
   struct {
      uint32_t no_output_of_prior_pics_flag : 1;
      uint32_t dependent_slice_segment_flag : 1;
      uint32_t pic_output_flag : 1;
      uint32_t short_term_ref_pic_set_sps_flag : 1;
      uint32_t slice_sao_luma_flag : 1;
      uint32_t slice_sao_chroma_flag : 1;
      uint32_t slice_temporal_mvp_enabled_flag : 1;
      uint32_t num_ref_idx_active_override_flag : 1;
      uint32_t mvd_l1_zero_flag : 1;
      uint32_t cabac_init_flag : 1;
      uint32_t collocated_from_l0_flag : 1;
      uint32_t cu_chroma_qp_offset_enabled_flag : 1;
      uint32_t deblocking_filter_override_flag : 1;
      uint32_t slice_deblocking_filter_disabled_flag : 1;
      uint32_t slice_loop_filter_across_slices_enabled_flag : 1;
   };
   uint8_t slice_type;
   uint32_t slice_pic_order_cnt_lsb;
   uint8_t colour_plane_id;
   uint8_t short_term_ref_pic_set_idx;
   uint8_t num_long_term_sps;
   uint8_t num_long_term_pics;
   uint8_t lt_idx_sps[PIPE_H265_MAX_LONG_TERM_REF_PICS_SPS];
   uint8_t poc_lsb_lt[PIPE_H265_MAX_LONG_TERM_PICS];
   uint8_t used_by_curr_pic_lt_flag[PIPE_H265_MAX_LONG_TERM_PICS];
   uint8_t delta_poc_msb_present_flag[PIPE_H265_MAX_DELTA_POC];
   uint8_t delta_poc_msb_cycle_lt[PIPE_H265_MAX_DELTA_POC];
   uint8_t num_ref_idx_l0_active_minus1;
   uint8_t num_ref_idx_l1_active_minus1;
   uint8_t collocated_ref_idx;
   uint8_t max_num_merge_cand;
   int8_t slice_qp_delta;
   int8_t slice_cb_qp_offset;
   int8_t slice_cr_qp_offset;
   int8_t slice_beta_offset_div2;
   int8_t slice_tc_offset_div2;
   struct pipe_h265_ref_pic_lists_modification ref_pic_lists_modification;
};

struct pipe_h265_enc_rate_control
{
   enum pipe_h2645_enc_rate_control_method rate_ctrl_method;
   unsigned target_bitrate;
   unsigned peak_bitrate;
   unsigned frame_rate_num;
   unsigned frame_rate_den;
   unsigned init_qp;
   unsigned quant_i_frames;
   unsigned quant_p_frames;
   unsigned quant_b_frames;
   unsigned vbv_buffer_size;
   unsigned vbv_buf_lv;
   unsigned vbv_buf_initial_size;
   bool app_requested_hrd_buffer;
   unsigned fill_data_enable;
   unsigned skip_frame_enable;
   unsigned enforce_hrd;
   unsigned max_au_size;
   unsigned max_qp;
   unsigned min_qp;
   bool app_requested_qp_range;

   /* Used with PIPE_H2645_ENC_RATE_CONTROL_METHOD_QUALITY_VARIABLE */
   unsigned vbr_quality_factor;
};

struct pipe_h265_enc_dpb_entry
{
   uint32_t id;
   uint32_t pic_order_cnt;
   uint32_t temporal_id;
   bool is_ltr;
   struct pipe_video_buffer *buffer;
   struct pipe_video_buffer *downscaled_buffer; /* for two pass */
   bool evict;
};

struct pipe_h265_enc_picture_desc
{
   struct pipe_picture_desc base;

   struct pipe_h265_enc_vid_param vid;
   struct pipe_h265_enc_seq_param seq;
   struct pipe_h265_enc_pic_param pic;
   struct pipe_h265_enc_slice_param slice;
   struct pipe_h265_enc_rate_control rc[4];

   enum pipe_h2645_enc_picture_type picture_type;
   unsigned decoded_curr_pic;
   unsigned reference_frames[16];
   unsigned frame_num;
   unsigned pic_order_cnt;
   unsigned pic_order_cnt_type;
   struct pipe_enc_quality_modes quality_modes;
   struct pipe_enc_intra_refresh intra_refresh;
   struct pipe_enc_roi roi;
   struct pipe_enc_dirty_info dirty_info;
   struct pipe_enc_move_info move_info;

   /* See PIPE_VIDEO_CAP_ENC_GPU_STATS_QP_MAP */
   struct pipe_resource *gpu_stats_qp_map;
   /* See PIPE_VIDEO_CAP_ENC_GPU_STATS_SATD_MAP */
   struct pipe_resource *gpu_stats_satd_map;
   /* See PIPE_VIDEO_CAP_ENC_GPU_STATS_RATE_CONTROL_BITS_MAP */
   struct pipe_resource *gpu_stats_rc_bitallocation_map;
   /* See PIPE_VIDEO_CAP_ENC_GPU_STATS_PSNR */
   struct pipe_resource *gpu_stats_psnr;

   /* See PIPE_VIDEO_CAP_ENC_QP_MAPS */
   struct pipe_resource *input_gpu_qpmap;

   unsigned num_ref_idx_l0_active_minus1;
   unsigned num_ref_idx_l1_active_minus1;
   unsigned ref_idx_l0_list[PIPE_H265_MAX_NUM_LIST_REF];
   unsigned ref_idx_l1_list[PIPE_H265_MAX_NUM_LIST_REF];
   bool not_referenced;
   struct hash_table *frame_idx;

   enum pipe_video_slice_mode slice_mode;

   /* Use with PIPE_VIDEO_SLICE_MODE_BLOCKS */
   unsigned num_slice_descriptors;
   struct h265_slice_descriptor slices_descriptors[128];

   /* Use with PIPE_VIDEO_SLICE_MODE_MAX_SLICE_SIZE */
   unsigned max_slice_bytes;
   enum pipe_video_feedback_metadata_type requested_metadata;

   struct pipe_enc_hdr_cll metadata_hdr_cll;
   struct pipe_enc_hdr_mdcv metadata_hdr_mdcv;

   struct pipe_h265_enc_dpb_entry dpb[PIPE_H265_MAX_DPB_SIZE];
   uint8_t dpb_size;
   uint8_t dpb_curr_pic; /* index in dpb */
   uint8_t ref_list0[PIPE_H265_MAX_NUM_LIST_REF]; /* index in dpb, PIPE_H2645_LIST_REF_INVALID_ENTRY invalid */
   uint8_t ref_list1[PIPE_H265_MAX_NUM_LIST_REF]; /* index in dpb, PIPE_H2645_LIST_REF_INVALID_ENTRY invalid */

   struct util_dynarray raw_headers; /* struct pipe_enc_raw_header */

   struct pipe_enc_two_pass_frame_config twopass_frame_config;
};

struct pipe_av1_enc_rate_control
{
   enum pipe_h2645_enc_rate_control_method rate_ctrl_method;
   unsigned target_bitrate;
   unsigned peak_bitrate;
   unsigned frame_rate_num;
   unsigned frame_rate_den;
   unsigned vbv_buffer_size;
   unsigned vbv_buf_lv;
   unsigned vbv_buf_initial_size;
   bool app_requested_hrd_buffer;
   unsigned fill_data_enable;
   unsigned skip_frame_enable;
   unsigned enforce_hrd;
   unsigned max_au_size;
   unsigned qp; /* Initial QP */
   unsigned qp_inter;
   unsigned max_qp;
   unsigned min_qp;
   bool app_requested_qp_range;
   bool app_requested_initial_qp;

   /* Used with PIPE_H2645_ENC_RATE_CONTROL_METHOD_QUALITY_VARIABLE */
   unsigned vbr_quality_factor;
};

struct pipe_av1_enc_decoder_model_info
{
   uint32_t buffer_delay_length_minus1;
   uint32_t num_units_in_decoding_tick;
   uint32_t buffer_removal_time_length_minus1;
   uint32_t frame_presentation_time_length_minus1;
};

struct pipe_av1_enc_color_description
{
   uint32_t color_primaries;
   uint32_t transfer_characteristics;
   uint32_t matrix_coefficients;
   uint32_t color_range;
   uint32_t chroma_sample_position;
};
struct pipe_av1_enc_seq_param
{
   uint32_t profile;
   uint32_t level;
   uint32_t tier;
   uint32_t num_temporal_layers;
   uint32_t intra_period;
   uint32_t ip_period;
   uint32_t bit_depth_minus8;
   uint32_t pic_width_in_luma_samples;
   uint32_t pic_height_in_luma_samples;
   struct
   {
      uint32_t use_128x128_superblock:1;
      uint32_t enable_filter_intra :1;
      uint32_t enable_intra_edge_filter :1;
      uint32_t enable_interintra_compound :1;
      uint32_t enable_masked_compound :1;
      uint32_t enable_warped_motion :1;
      uint32_t enable_dual_filter :1;
      uint32_t enable_cdef:1;
      uint32_t enable_restoration:1;
      uint32_t enable_superres:1;
      uint32_t enable_order_hint:1;
      uint32_t enable_jnt_comp:1;
      uint32_t color_description_present_flag:1;
      uint32_t enable_ref_frame_mvs:1;
      uint32_t frame_id_number_present_flag:1;
      uint32_t disable_screen_content_tools:1;
      uint32_t timing_info_present_flag:1;
      uint32_t equal_picture_interval:1;
      uint32_t decoder_model_info_present_flag:1;
      uint32_t force_screen_content_tools:2;
      uint32_t force_integer_mv:2;
      uint32_t initial_display_delay_present_flag:1;
      uint32_t choose_integer_mv:1;
      uint32_t still_picture:1;
      uint32_t reduced_still_picture_header:1;
   } seq_bits;

   /* timing info params */
   uint32_t num_units_in_display_tick;
   uint32_t time_scale;
   uint32_t num_tick_per_picture_minus1;
   uint32_t delta_frame_id_length;
   uint32_t additional_frame_id_length;
   uint32_t order_hint_bits;
   struct pipe_av1_enc_decoder_model_info decoder_model_info;
   struct pipe_av1_enc_color_description color_config;
   uint16_t frame_width_bits_minus1;
   uint16_t frame_height_bits_minus1;
   uint16_t operating_point_idc[32];
   uint8_t seq_level_idx[32];
   uint8_t seq_tier[32];
   uint8_t decoder_model_present_for_this_op[32];
   uint32_t decoder_buffer_delay[32];
   uint32_t encoder_buffer_delay[32];
   uint8_t low_delay_mode_flag[32];
   uint8_t initial_display_delay_present_for_this_op[32];
   uint8_t initial_display_delay_minus_1[32];
};

struct pipe_av1_tile_group {
   uint8_t tile_group_start;
   uint8_t tile_group_end;
};

struct pipe_av1_enc_dpb_entry
{
   uint32_t id;
   uint32_t order_hint;
   struct pipe_video_buffer *buffer;
};

struct pipe_av1_enc_picture_desc
{
   struct pipe_picture_desc base;
   enum pipe_av1_enc_frame_type frame_type;
   struct pipe_av1_enc_seq_param seq;
   struct pipe_av1_enc_rate_control rc[4];
   struct {
      uint32_t obu_extension_flag:1;
      uint32_t enable_frame_obu:1;
      uint32_t error_resilient_mode:1;
      uint32_t disable_cdf_update:1;
      uint32_t frame_size_override_flag:1;
      uint32_t allow_screen_content_tools:1;
      uint32_t allow_intrabc:1;
      uint32_t force_integer_mv:1;
      uint32_t disable_frame_end_update_cdf:1;
      uint32_t palette_mode_enable:1;
      uint32_t allow_high_precision_mv:1;
      uint32_t use_ref_frame_mvs;
      uint32_t show_existing_frame:1;
      uint32_t show_frame:1;
      uint32_t showable_frame:1;
      uint32_t enable_render_size:1;
      uint32_t use_superres:1;
      uint32_t reduced_tx_set:1;
      uint32_t skip_mode_present:1;
      uint32_t long_term_reference:1;
      uint32_t uniform_tile_spacing:1;
      uint32_t frame_refs_short_signaling:1;
      uint32_t is_motion_mode_switchable:1;
   };
   struct pipe_enc_quality_modes quality_modes;
   struct pipe_enc_intra_refresh intra_refresh;
   struct pipe_enc_roi roi;
   /* See PIPE_VIDEO_CAP_ENC_QP_MAPS */
   struct pipe_resource *input_gpu_qpmap;
   uint32_t tile_rows;
   uint32_t tile_cols;
   unsigned num_tile_groups;
   struct pipe_av1_tile_group tile_groups[256];
   uint32_t context_update_tile_id;
   uint16_t width_in_sbs_minus_1[63];
   uint16_t height_in_sbs_minus_1[63];
   uint32_t frame_num;
   uint32_t last_key_frame_num;
   uint32_t number_of_skips;
   uint32_t temporal_id;
   uint32_t spatial_id;
   uint16_t frame_width;
   uint16_t frame_height;
   uint16_t frame_width_sb;
   uint16_t frame_height_sb;
   uint16_t upscaled_width;
   uint16_t render_width_minus_1;
   uint16_t render_height_minus_1;
   uint32_t interpolation_filter;
   uint8_t tx_mode;
   uint8_t compound_reference_mode;
   uint32_t order_hint;
   uint8_t superres_scale_denominator;
   uint32_t primary_ref_frame;
   uint8_t refresh_frame_flags;
   uint8_t ref_frame_idx[7];
   uint32_t delta_frame_id_minus_1[7];
   uint32_t frame_presentation_time;
   uint32_t current_frame_id;
   uint32_t ref_order_hint[8];
   uint8_t last_frame_idx;
   uint8_t gold_frame_idx;

   struct {
      uint8_t cdef_damping_minus_3;
      uint8_t cdef_bits;
      uint8_t cdef_y_strengths[8];
      uint8_t cdef_uv_strengths[8];
   } cdef;

   struct {
      uint8_t yframe_restoration_type;
      uint8_t cbframe_restoration_type;
      uint8_t crframe_restoration_type;
      uint8_t lr_unit_shift;
      uint8_t lr_uv_shift;
   } restoration;

   struct {
      uint8_t filter_level[2];
      uint8_t filter_level_u;
      uint8_t filter_level_v;
      uint8_t sharpness_level;
      uint8_t mode_ref_delta_enabled;
      uint8_t mode_ref_delta_update;
      int8_t  ref_deltas[8];
      int8_t  mode_deltas[2];
      uint8_t delta_lf_present;
      uint8_t delta_lf_res;
      uint8_t delta_lf_multi;
   } loop_filter;

   struct {
      uint8_t base_qindex;
      int8_t y_dc_delta_q;
      int8_t u_dc_delta_q;
      int8_t u_ac_delta_q;
      int8_t v_dc_delta_q;
      int8_t v_ac_delta_q;
      uint8_t min_base_qindex;
      uint8_t max_base_qindex;
      uint8_t using_qmatrix;
      uint8_t qm_y;
      uint8_t qm_u;
      uint8_t qm_v;
      uint8_t delta_q_present;
      uint8_t delta_q_res;
   } quantization;

   struct {
      uint8_t obu_extension_flag;
      uint8_t obu_has_size_field;
      uint8_t temporal_id;
      uint8_t spatial_id;
   } tg_obu_header;

   enum pipe_video_feedback_metadata_type requested_metadata;

   union {
      struct {
         uint32_t hdr_cll:1;
         uint32_t hdr_mdcv:1;
      };
      uint32_t value;
   } metadata_flags;

   struct pipe_enc_hdr_cll metadata_hdr_cll;
   struct pipe_enc_hdr_mdcv metadata_hdr_mdcv;

   struct pipe_av1_enc_dpb_entry dpb[PIPE_AV1_MAX_DPB_SIZE + 1];
   uint8_t dpb_size;
   uint8_t dpb_curr_pic; /* index in dpb */
   uint8_t dpb_ref_frame_idx[PIPE_AV1_REFS_PER_FRAME]; /* index in dpb, PIPE_H2645_LIST_REF_INVALID_ENTRY invalid */
   uint8_t ref_list0[PIPE_AV1_REFS_PER_FRAME]; /* index in dpb_ref_frame_idx, PIPE_H2645_LIST_REF_INVALID_ENTRY invalid */
   uint8_t ref_list1[PIPE_AV1_REFS_PER_FRAME]; /* index in dpb_ref_frame_idx, PIPE_H2645_LIST_REF_INVALID_ENTRY invalid */

   struct util_dynarray raw_headers; /* struct pipe_enc_raw_header */
};

struct pipe_h265_sps
{
   uint8_t chroma_format_idc;
   uint8_t separate_colour_plane_flag;
   uint32_t pic_width_in_luma_samples;
   uint32_t pic_height_in_luma_samples;
   uint8_t bit_depth_luma_minus8;
   uint8_t bit_depth_chroma_minus8;
   uint8_t log2_max_pic_order_cnt_lsb_minus4;
   uint8_t sps_max_dec_pic_buffering_minus1;
   uint8_t log2_min_luma_coding_block_size_minus3;
   uint8_t log2_diff_max_min_luma_coding_block_size;
   uint8_t log2_min_transform_block_size_minus2;
   uint8_t log2_diff_max_min_transform_block_size;
   uint8_t max_transform_hierarchy_depth_inter;
   uint8_t max_transform_hierarchy_depth_intra;
   uint8_t scaling_list_enabled_flag;
   uint8_t ScalingList4x4[6][16];
   uint8_t ScalingList8x8[6][64];
   uint8_t ScalingList16x16[6][64];
   uint8_t ScalingList32x32[2][64];
   uint8_t ScalingListDCCoeff16x16[6];
   uint8_t ScalingListDCCoeff32x32[2];
   uint8_t amp_enabled_flag;
   uint8_t sample_adaptive_offset_enabled_flag;
   uint8_t pcm_enabled_flag;
   uint8_t pcm_sample_bit_depth_luma_minus1;
   uint8_t pcm_sample_bit_depth_chroma_minus1;
   uint8_t log2_min_pcm_luma_coding_block_size_minus3;
   uint8_t log2_diff_max_min_pcm_luma_coding_block_size;
   uint8_t pcm_loop_filter_disabled_flag;
   uint8_t num_short_term_ref_pic_sets;
   uint8_t long_term_ref_pics_present_flag;
   uint8_t num_long_term_ref_pics_sps;
   uint8_t sps_temporal_mvp_enabled_flag;
   uint8_t strong_intra_smoothing_enabled_flag;
   uint8_t no_pic_reordering_flag;
   uint8_t no_bi_pred_flag;
};

struct pipe_h265_pps
{
   struct pipe_h265_sps *sps;

   uint8_t dependent_slice_segments_enabled_flag;
   uint8_t output_flag_present_flag;
   uint8_t num_extra_slice_header_bits;
   uint8_t sign_data_hiding_enabled_flag;
   uint8_t cabac_init_present_flag;
   uint8_t num_ref_idx_l0_default_active_minus1;
   uint8_t num_ref_idx_l1_default_active_minus1;
   int8_t init_qp_minus26;
   uint8_t constrained_intra_pred_flag;
   uint8_t transform_skip_enabled_flag;
   uint8_t cu_qp_delta_enabled_flag;
   uint8_t diff_cu_qp_delta_depth;
   int8_t pps_cb_qp_offset;
   int8_t pps_cr_qp_offset;
   uint8_t pps_slice_chroma_qp_offsets_present_flag;
   uint8_t weighted_pred_flag;
   uint8_t weighted_bipred_flag;
   uint8_t transquant_bypass_enabled_flag;
   uint8_t tiles_enabled_flag;
   uint8_t entropy_coding_sync_enabled_flag;
   uint8_t num_tile_columns_minus1;
   uint8_t num_tile_rows_minus1;
   uint8_t uniform_spacing_flag;
   uint16_t column_width_minus1[20];
   uint16_t row_height_minus1[22];
   uint8_t loop_filter_across_tiles_enabled_flag;
   uint8_t pps_loop_filter_across_slices_enabled_flag;
   uint8_t deblocking_filter_control_present_flag;
   uint8_t deblocking_filter_override_enabled_flag;
   uint8_t pps_deblocking_filter_disabled_flag;
   int8_t pps_beta_offset_div2;
   int8_t pps_tc_offset_div2;
   uint8_t lists_modification_present_flag;
   uint8_t log2_parallel_merge_level_minus2;
   uint8_t slice_segment_header_extension_present_flag;
};

struct pipe_h265_picture_desc
{
   struct pipe_picture_desc base;

   struct pipe_h265_pps *pps;

   uint8_t IDRPicFlag;
   uint8_t RAPPicFlag;
   /*
      When the current picture is an IRAP picture, IntraPicFlag shall be equal to 1.
      When the current picture is not an IRAP picture, the host software decoder is
      not required to determine whether all slices of the current picture are I slices
      – i.e. it may simply set IntraPicFlag to 0 in this case....

      Some frontends have IntraPicFlag defined (ie. VAPictureParameterBufferHEVC)
      and some others like VDPAU/OMX can derive it from RAPPicFlag
   */
   uint8_t IntraPicFlag;
   uint8_t CurrRpsIdx;
   uint32_t NumPocTotalCurr;
   uint32_t NumDeltaPocsOfRefRpsIdx;
   uint32_t NumShortTermPictureSliceHeaderBits;
   uint32_t NumLongTermPictureSliceHeaderBits;

   int32_t CurrPicOrderCntVal;
   struct pipe_video_buffer *ref[16];
   int32_t PicOrderCntVal[16];
   uint8_t IsLongTerm[16];
   uint8_t NumPocStCurrBefore;
   uint8_t NumPocStCurrAfter;
   uint8_t NumPocLtCurr;
   uint8_t RefPicSetStCurrBefore[8];
   uint8_t RefPicSetStCurrAfter[8];
   uint8_t RefPicSetLtCurr[8];
   uint8_t RefPicList[2][15];
   bool LtCurrDone;

   struct
   {
      bool slice_info_present;
      uint32_t slice_count;
      uint32_t slice_data_size[PIPE_H265_MAX_SLICES];
      uint32_t slice_data_offset[PIPE_H265_MAX_SLICES];
      enum pipe_slice_buffer_placement_type slice_data_flag[PIPE_H265_MAX_SLICES];
   } slice_parameter;
};

struct pipe_mjpeg_picture_desc
{
   struct pipe_picture_desc base;

   struct
   {
      uint16_t picture_width;
      uint16_t picture_height;

      struct {
         uint8_t component_id;
         uint8_t h_sampling_factor;
         uint8_t v_sampling_factor;
         uint8_t quantiser_table_selector;
      } components[255];

      uint8_t num_components;
      uint16_t crop_x;
      uint16_t crop_y;
      uint16_t crop_width;
      uint16_t crop_height;
      uint32_t sampling_factor;
   } picture_parameter;

   struct
   {
      uint8_t load_quantiser_table[4];
      uint8_t quantiser_table[4][64];
   } quantization_table;

   struct
   {
      uint8_t load_huffman_table[2];

      struct {
         uint8_t   num_dc_codes[16];
         uint8_t   dc_values[12];
         uint8_t   num_ac_codes[16];
         uint8_t   ac_values[162];
         uint8_t   pad[2];
      } table[2];
   } huffman_table;

   struct
   {
      unsigned slice_data_size;
      unsigned slice_data_offset;
      unsigned slice_data_flag;
      unsigned slice_horizontal_position;
      unsigned slice_vertical_position;

      struct {
         uint8_t component_selector;
         uint8_t dc_table_selector;
         uint8_t ac_table_selector;
      } components[4];

      uint8_t num_components;

      uint16_t restart_interval;
      unsigned num_mcus;
   } slice_parameter;
};

struct vp9_segment_parameter
{
   struct {
      uint16_t segment_reference_enabled:1;
      uint16_t segment_reference:2;
      uint16_t segment_reference_skipped:1;
   } segment_flags;

   bool alt_quant_enabled;
   int16_t alt_quant;

   bool alt_lf_enabled;
   int16_t alt_lf;

   uint8_t filter_level[4][2];

   int16_t luma_ac_quant_scale;
   int16_t luma_dc_quant_scale;

   int16_t chroma_ac_quant_scale;
   int16_t chroma_dc_quant_scale;
};

struct pipe_vp9_picture_desc
{
   struct pipe_picture_desc base;

   struct pipe_video_buffer *ref[16];

   struct {
      uint16_t frame_width;
      uint16_t frame_height;
      uint16_t prev_frame_width;
      uint16_t prev_frame_height;

      struct {
         uint32_t  subsampling_x:1;
         uint32_t  subsampling_y:1;
         uint32_t  frame_type:1;
         uint32_t  show_frame:1;
         uint32_t  prev_show_frame:1;
         uint32_t  error_resilient_mode:1;
         uint32_t  intra_only:1;
         uint32_t  allow_high_precision_mv:1;
         uint32_t  mcomp_filter_type:3;
         uint32_t  frame_parallel_decoding_mode:1;
         uint32_t  reset_frame_context:2;
         uint32_t  refresh_frame_context:1;
         uint32_t  frame_context_idx:2;
         uint32_t  segmentation_enabled:1;
         uint32_t  segmentation_temporal_update:1;
         uint32_t  segmentation_update_map:1;
         uint32_t  last_ref_frame:3;
         uint32_t  last_ref_frame_sign_bias:1;
         uint32_t  golden_ref_frame:3;
         uint32_t  golden_ref_frame_sign_bias:1;
         uint32_t  alt_ref_frame:3;
         uint32_t  alt_ref_frame_sign_bias:1;
         uint32_t  lossless_flag:1;
      } pic_fields;

      uint8_t filter_level;
      uint8_t sharpness_level;

      uint8_t log2_tile_rows;
      uint8_t log2_tile_columns;

      uint8_t frame_header_length_in_bytes;

      uint16_t first_partition_size;

      uint8_t mb_segment_tree_probs[7];
      uint8_t segment_pred_probs[3];

      uint8_t profile;

      uint8_t bit_depth;

      bool mode_ref_delta_enabled;
      bool mode_ref_delta_update;

      uint8_t base_qindex;
      int8_t y_dc_delta_q;
      int8_t uv_ac_delta_q;
      int8_t uv_dc_delta_q;
      uint8_t abs_delta;
      uint8_t ref_deltas[4];
      uint8_t mode_deltas[2];
   } picture_parameter;

   struct {
      bool slice_info_present;
      uint32_t slice_count;
      uint32_t slice_data_size[128];
      uint32_t slice_data_offset[128];
      enum pipe_slice_buffer_placement_type slice_data_flag[128];
      struct vp9_segment_parameter seg_param[8];
   } slice_parameter;
};

struct pipe_av1_picture_desc
{
   struct pipe_picture_desc base;

   struct pipe_video_buffer *ref[16];
   struct pipe_video_buffer *film_grain_target;
   struct {
      uint8_t profile;
      uint8_t order_hint_bits_minus_1;
      uint8_t bit_depth_idx;

      struct {
         uint32_t use_128x128_superblock:1;
         uint32_t enable_filter_intra:1;
         uint32_t enable_intra_edge_filter:1;
         uint32_t enable_interintra_compound:1;
         uint32_t enable_masked_compound:1;
         uint32_t enable_dual_filter:1;
         uint32_t enable_order_hint:1;
         uint32_t enable_jnt_comp:1;
         uint32_t enable_cdef:1;
         uint32_t mono_chrome:1;
         uint32_t ref_frame_mvs:1;
         uint32_t film_grain_params_present:1;
         uint32_t subsampling_x:1;
         uint32_t subsampling_y:1;
      } seq_info_fields;

      uint32_t current_frame_id;

      uint16_t frame_width;
      uint16_t frame_height;
      uint16_t max_width;
      uint16_t max_height;

      uint8_t ref_frame_idx[7];
      uint8_t primary_ref_frame;
      uint8_t order_hint;

      struct {
         struct {
            uint32_t enabled:1;
            uint32_t update_map:1;
            uint32_t update_data:1;
            uint32_t temporal_update:1;
         } segment_info_fields;

         int16_t feature_data[8][8];
         uint8_t feature_mask[8];
      } seg_info;

      struct {
         struct {
            uint32_t apply_grain:1;
            uint32_t chroma_scaling_from_luma:1;
            uint32_t grain_scaling_minus_8:2;
            uint32_t ar_coeff_lag:2;
            uint32_t ar_coeff_shift_minus_6:2;
            uint32_t grain_scale_shift:2;
            uint32_t overlap_flag:1;
            uint32_t clip_to_restricted_range:1;
         } film_grain_info_fields;

         uint16_t grain_seed;
         uint8_t num_y_points;
         uint8_t point_y_value[14];
         uint8_t point_y_scaling[14];
         uint8_t num_cb_points;
         uint8_t point_cb_value[10];
         uint8_t point_cb_scaling[10];
         uint8_t num_cr_points;
         uint8_t point_cr_value[10];
         uint8_t point_cr_scaling[10];
         int8_t ar_coeffs_y[24];
         int8_t ar_coeffs_cb[25];
         int8_t ar_coeffs_cr[25];
         uint8_t cb_mult;
         uint8_t cb_luma_mult;
         uint16_t cb_offset;
         uint8_t cr_mult;
         uint8_t cr_luma_mult;
         uint16_t cr_offset;
      } film_grain_info;

      uint8_t tile_cols;
      uint8_t tile_rows;
      uint32_t tile_col_start_sb[65];
      uint32_t tile_row_start_sb[65];
      uint16_t width_in_sbs[64];
      uint16_t height_in_sbs[64];
      uint16_t context_update_tile_id;

      struct {
         uint32_t frame_type:2;
         uint32_t show_frame:1;
         uint32_t showable_frame:1;
         uint32_t error_resilient_mode:1;
         uint32_t disable_cdf_update:1;
         uint32_t allow_screen_content_tools:1;
         uint32_t force_integer_mv:1;
         uint32_t allow_intrabc:1;
         uint32_t use_superres:1;
         uint32_t allow_high_precision_mv:1;
         uint32_t is_motion_mode_switchable:1;
         uint32_t use_ref_frame_mvs:1;
         uint32_t disable_frame_end_update_cdf:1;
         uint32_t uniform_tile_spacing_flag:1;
         uint32_t allow_warped_motion:1;
         uint32_t large_scale_tile:1;
      } pic_info_fields;

      uint8_t superres_scale_denominator;

      uint8_t interp_filter;
      uint8_t filter_level[2];
      uint8_t filter_level_u;
      uint8_t filter_level_v;
      struct {
         uint8_t sharpness_level:3;
         uint8_t mode_ref_delta_enabled:1;
         uint8_t mode_ref_delta_update:1;
      } loop_filter_info_fields;

      int8_t ref_deltas[8];
      int8_t mode_deltas[2];

      uint8_t base_qindex;
      int8_t y_dc_delta_q;
      int8_t u_dc_delta_q;
      int8_t u_ac_delta_q;
      int8_t v_dc_delta_q;
      int8_t v_ac_delta_q;

      struct {
         uint16_t using_qmatrix:1;
         uint16_t qm_y:4;
         uint16_t qm_u:4;
         uint16_t qm_v:4;
      } qmatrix_fields;

      struct {
         uint32_t delta_q_present_flag:1;
         uint32_t log2_delta_q_res:2;
         uint32_t delta_lf_present_flag:1;
         uint32_t log2_delta_lf_res:2;
         uint32_t delta_lf_multi:1;
         uint32_t tx_mode:2;
         uint32_t reference_select:1;
         uint32_t reduced_tx_set_used:1;
         uint32_t skip_mode_present:1;
      } mode_control_fields;

      uint8_t cdef_damping_minus_3;
      uint8_t cdef_bits;
      uint8_t cdef_y_strengths[8];
      uint8_t cdef_uv_strengths[8];

      struct {
         uint16_t yframe_restoration_type:2;
         uint16_t cbframe_restoration_type:2;
         uint16_t crframe_restoration_type:2;
         uint16_t lr_unit_shift:2;
         uint16_t lr_uv_shift:1;
      } loop_restoration_fields;

      uint16_t lr_unit_size[3];

      struct {
         uint32_t wmtype;
         uint8_t invalid;
         int32_t wmmat[8];
      } wm[7];

      uint32_t refresh_frame_flags;
      uint8_t matrix_coefficients;
   } picture_parameter;

   struct {
      uint32_t slice_data_size[256];
      uint32_t slice_data_offset[256];
      uint16_t slice_data_row[256];
      uint16_t slice_data_col[256];
      uint8_t slice_data_anchor_frame_idx[256];
      uint16_t slice_count;
   } slice_parameter;
};

struct pipe_vpp_blend
{
   enum pipe_video_vpp_blend_mode mode;
   /* To be used with PIPE_VIDEO_VPP_BLEND_MODE_GLOBAL_ALPHA */
   float global_alpha;
};

struct pipe_vpp_desc
{
   struct pipe_picture_desc base;
   struct u_rect src_region;
   struct u_rect dst_region;
   enum pipe_video_vpp_orientation orientation;
   struct pipe_vpp_blend blend;

   uint32_t background_color;
   enum pipe_video_vpp_color_standard_type in_colors_standard;
   enum pipe_video_vpp_color_range in_color_range;
   enum pipe_video_vpp_chroma_siting in_chroma_siting;
   enum pipe_video_vpp_color_standard_type out_colors_standard;
   enum pipe_video_vpp_color_range out_color_range;
   enum pipe_video_vpp_chroma_siting out_chroma_siting;

   enum pipe_video_vpp_color_primaries in_color_primaries;
   enum pipe_video_vpp_transfer_characteristic in_transfer_characteristics;
   enum pipe_video_vpp_matrix_coefficients in_matrix_coefficients;

   enum pipe_video_vpp_color_primaries out_color_primaries;
   enum pipe_video_vpp_transfer_characteristic out_transfer_characteristics;
   enum pipe_video_vpp_matrix_coefficients out_matrix_coefficients;

   enum pipe_video_vpp_filter_flag filter_flags;
};


/* To be used with PIPE_VIDEO_CAP_ENC_HEVC_PREDICTION_DIRECTION */
enum pipe_h265_enc_pred_direction
{
   /* No restrictions*/
   PIPE_H265_PRED_DIRECTION_ALL = 0x0,
   /* P Frame*/
   PIPE_H265_PRED_DIRECTION_PREVIOUS = 0x1,
   /* Same reference lists for B Frame*/
   PIPE_H265_PRED_DIRECTION_FUTURE = 0x2,
   /* Low delay B frames */
   PIPE_H265_PRED_DIRECTION_BI_NOT_EMPTY = 0x4,
};

/* To be used with PIPE_VIDEO_CAP_ENC_HEVC_FEATURE_FLAGS
   the config_supported bit is used to differenciate a supported
   config with all bits as zero and unsupported by driver with value=0
*/
union pipe_h265_enc_cap_features {
   struct {
      /** Separate colour planes.
      *
      * Allows setting separate_colour_plane_flag in the SPS.
      */
      uint32_t separate_colour_planes    : 2;
      /** Scaling lists.
      *
      * Allows scaling_list() elements to be present in both the SPS
      * and the PPS.  The decoded form of the scaling lists must also
      * be supplied in a VAQMatrixBufferHEVC buffer when scaling lists
      * are enabled.
      */
      uint32_t scaling_lists             : 2;
      /** Asymmetric motion partitions.
      *
      * Allows setting amp_enabled_flag in the SPS.
      */
      uint32_t amp                       : 2;
      /** Sample adaptive offset filter.
      *
      * Allows setting slice_sao_luma_flag and slice_sao_chroma_flag
      * in slice headers.
      */
      uint32_t sao                       : 2;
      /** PCM sample blocks.
      *
      * Allows setting pcm_enabled_flag in the SPS.  When enabled
      * PCM parameters must be supplied with the sequence parameters,
      * including block sizes which may be further constrained as
      * noted in the VAConfigAttribEncHEVCBlockSizes attribute.
      */
      uint32_t pcm                       : 2;
      /** Temporal motion vector Prediction.
      *
      * Allows setting slice_temporal_mvp_enabled_flag in slice
      * headers.
      */
      uint32_t temporal_mvp              : 2;
      /** Strong intra smoothing.
      *
      * Allows setting strong_intra_smoothing_enabled_flag in the SPS.
      */
      uint32_t strong_intra_smoothing    : 2;
      /** Dependent slices.
      *
      * Allows setting dependent_slice_segment_flag in slice headers.
      */
      uint32_t dependent_slices          : 2;
      /** Sign data hiding.
      *
      * Allows setting sign_data_hiding_enable_flag in the PPS.
      */
      uint32_t sign_data_hiding          : 2;
      /** Constrained intra prediction.
      *
      * Allows setting constrained_intra_pred_flag in the PPS.
      */
      uint32_t constrained_intra_pred    : 2;
      /** Transform skipping.
      *
      * Allows setting transform_skip_enabled_flag in the PPS.
      */
      uint32_t transform_skip            : 2;
      /** QP delta within coding units.
      *
      * Allows setting cu_qp_delta_enabled_flag in the PPS.
      */
      uint32_t cu_qp_delta               : 2;
      /** Weighted prediction.
      *
      * Allows setting weighted_pred_flag and weighted_bipred_flag in
      * the PPS.  The pred_weight_table() data must be supplied with
      * every slice header when weighted prediction is enabled.
      */
      uint32_t weighted_prediction       : 2;
      /** Transform and quantisation bypass.
      *
      * Allows setting transquant_bypass_enabled_flag in the PPS.
      */
      uint32_t transquant_bypass         : 2;
      /** Deblocking filter disable.
      *
      * Allows setting slice_deblocking_filter_disabled_flag.
      */
      uint32_t deblocking_filter_disable : 2;
      /** Flag indicating this is a supported configuration
      *
      *  It could be possible all the bits above are set to zero
      *  and this is a valid configuration, so we distinguish
      *  between get_video_param returning 0 for no support
      *  and this case with this bit flag.
      */
      uint32_t config_supported                          : 1;
   } bits;
   uint32_t value;
};

/* To be used with PIPE_VIDEO_CAP_ENC_HEVC_BLOCK_SIZES
   the config_supported bit is used to differenciate a supported
   config with all bits as zero and unsupported by driver with value=0 */
union pipe_h265_enc_cap_block_sizes {
   struct {
      /** Largest supported size of coding tree blocks.
      *
      * CtbLog2SizeY must not be larger than this.
      */
      uint32_t log2_max_coding_tree_block_size_minus3    : 2;
      /** Smallest supported size of coding tree blocks.
      *
      * CtbLog2SizeY must not be smaller than this.
      *
      * This may be the same as the maximum size, indicating that only
      * one CTB size is supported.
      */
      uint32_t log2_min_coding_tree_block_size_minus3    : 2;

      /** Smallest supported size of luma coding blocks.
      *
      * MinCbLog2SizeY must not be smaller than this.
      */
      uint32_t log2_min_luma_coding_block_size_minus3    : 2;

      /** Largest supported size of luma transform blocks.
      *
      * MaxTbLog2SizeY must not be larger than this.
      */
      uint32_t log2_max_luma_transform_block_size_minus2 : 2;
      /** Smallest supported size of luma transform blocks.
      *
      * MinTbLog2SizeY must not be smaller than this.
      */
      uint32_t log2_min_luma_transform_block_size_minus2 : 2;

      /** Largest supported transform hierarchy depth in inter
      *  coding units.
      *
      * max_transform_hierarchy_depth_inter must not be larger
      * than this.
      */
      uint32_t max_max_transform_hierarchy_depth_inter   : 2;
      /** Smallest supported transform hierarchy depth in inter
      *  coding units.
      *
      * max_transform_hierarchy_depth_inter must not be smaller
      * than this.
      */
      uint32_t min_max_transform_hierarchy_depth_inter   : 2;

      /** Largest supported transform hierarchy depth in intra
      *  coding units.
      *
      * max_transform_hierarchy_depth_intra must not be larger
      * than this.
      */
      uint32_t max_max_transform_hierarchy_depth_intra   : 2;
      /** Smallest supported transform hierarchy depth in intra
      *  coding units.
      *
      * max_transform_hierarchy_depth_intra must not be smaller
      * than this.
      */
      uint32_t min_max_transform_hierarchy_depth_intra   : 2;

      /** Largest supported size of PCM coding blocks.
      *
      *  Log2MaxIpcmCbSizeY must not be larger than this.
      */
      uint32_t log2_max_pcm_coding_block_size_minus3     : 2;
      /** Smallest supported size of PCM coding blocks.
      *
      *  Log2MinIpcmCbSizeY must not be smaller than this.
      */
      uint32_t log2_min_pcm_coding_block_size_minus3     : 2;
      /** Flag indicating this is a supported configuration
      *
      *  It could be possible all the bits above are set to zero
      *  and this is a valid configuration, so we distinguish
      *  between get_video_param returning 0 for no support
      *  and this case with this bit flag.
      */
      uint32_t config_supported                          : 1;
      } bits;
      uint32_t value;
};

union pipe_av1_enc_cap_features {
    struct {
        /**
         * Use 128x128 superblock.
         *
         * Allows setting use_128x128_superblock in the SPS.
         */
        uint32_t support_128x128_superblock     : 2;
        /**
         * Intra  filter.
         * Allows setting enable_filter_intra in the SPS.
         */
        uint32_t support_filter_intra           : 2;
        /**
         *  Intra edge filter.
         * Allows setting enable_intra_edge_filter in the SPS.
         */
        uint32_t support_intra_edge_filter      : 2;
        /**
         *  Interintra compound.
         * Allows setting enable_interintra_compound in the SPS.
         */
        uint32_t support_interintra_compound    : 2;
        /**
         *  Masked compound.
         * Allows setting enable_masked_compound in the SPS.
         */
        uint32_t support_masked_compound        : 2;
        /**
         *  Warped motion.
         * Allows setting enable_warped_motion in the SPS.
         */
        uint32_t support_warped_motion          : 2;
        /**
         *  Palette mode.
         * Allows setting palette_mode in the PPS.
         */
        uint32_t support_palette_mode           : 2;
        /**
         *  Dual filter.
         * Allows setting enable_dual_filter in the SPS.
         */
        uint32_t support_dual_filter            : 2;
        /**
         *  Jnt compound.
         * Allows setting enable_jnt_comp in the SPS.
         */
        uint32_t support_jnt_comp               : 2;
        /**
         *  Refrence frame mvs.
         * Allows setting enable_ref_frame_mvs in the SPS.
         */
        uint32_t support_ref_frame_mvs          : 2;
        /**
         *  Super resolution.
         * Allows setting enable_superres in the SPS.
         */
        uint32_t support_superres               : 2;
        /**
         *  Restoration.
         * Allows setting enable_restoration in the SPS.
         */
        uint32_t support_restoration            : 2;
        /**
         *  Allow intraBC.
         * Allows setting allow_intrabc in the PPS.
         */
        uint32_t support_allow_intrabc          : 2;
        /**
         *  Cdef channel strength.
         * Allows setting cdef_y_strengths and cdef_uv_strengths in PPS.
         */
        uint32_t support_cdef_channel_strength  : 2;
        /** Reserved bits for future, must be zero. */
        uint32_t reserved                       : 4;
    } bits;
    uint32_t value;
};

union pipe_av1_enc_cap_features_ext1 {
    struct {
        /**
         * Fields indicate which types of interpolation filter are supported.
         * (interpolation_filter & 0x01) == 1: eight_tap filter is supported, 0: not.
         * (interpolation_filter & 0x02) == 1: eight_tap_smooth filter is supported, 0: not.
         * (interpolation_filter & 0x04) == 1: eight_sharp filter is supported, 0: not.
         * (interpolation_filter & 0x08) == 1: bilinear filter is supported, 0: not.
         * (interpolation_filter & 0x10) == 1: switchable filter is supported, 0: not.
         */
        uint32_t interpolation_filter          : 5;
        /**
         * Min segmentId block size accepted.
         * Application need to send seg_id_block_size in PPS equal or larger than this value.
         */
        uint32_t min_segid_block_size_accepted : 8;
        /**
         * Type of segment feature supported.
         * (segment_feature_support & 0x01) == 1: SEG_LVL_ALT_Q is supported, 0: not.
         * (segment_feature_support & 0x02) == 1: SEG_LVL_ALT_LF_Y_V is supported, 0: not.
         * (segment_feature_support & 0x04) == 1: SEG_LVL_ALT_LF_Y_H is supported, 0: not.
         * (segment_feature_support & 0x08) == 1: SEG_LVL_ALT_LF_U is supported, 0: not.
         * (segment_feature_support & 0x10) == 1: SEG_LVL_ALT_LF_V is supported, 0: not.
         * (segment_feature_support & 0x20) == 1: SEG_LVL_REF_FRAME is supported, 0: not.
         * (segment_feature_support & 0x40) == 1: SEG_LVL_SKIP is supported, 0: not.
         * (segment_feature_support & 0x80) == 1: SEG_LVL_GLOBALMV is supported, 0: not.
         */
        uint32_t segment_feature_support       : 8;
        /** Reserved bits for future, must be zero. */
        uint32_t reserved                      : 11;
    } bits;
    uint32_t value;
};

union pipe_av1_enc_cap_features_ext2 {
    struct {
        /**
        * Tile size bytes minus1.
        * Specify the number of bytes needed to code tile size supported.
        * This value need to be set in frame header obu.
        */
        uint32_t tile_size_bytes_minus1        : 2;
        /**
        * Tile size bytes minus1.
        * Specify the fixed number of bytes needed to code syntax obu_size.
        */
        uint32_t obu_size_bytes_minus1         : 2;
        /**
         * tx_mode supported.
         * (tx_mode_support & 0x01) == 1: ONLY_4X4 is supported, 0: not.
         * (tx_mode_support & 0x02) == 1: TX_MODE_LARGEST is supported, 0: not.
         * (tx_mode_support & 0x04) == 1: TX_MODE_SELECT is supported, 0: not.
         */
        uint32_t tx_mode_support               : 3;
        /**
         * Max tile num minus1.
         * Specify the max number of tile supported by driver.
         */
        uint32_t max_tile_num_minus1           : 13;
        /** Reserved bits for future, must be zero. */
        uint32_t reserved                      : 12;
    } bits;
    uint32_t value;
};

struct codec_unit_location_t
{
   uint64_t offset;
   uint64_t size;
   enum codec_unit_location_flags flags;
};

struct pipe_enc_feedback_metadata
{
   /*
   * Driver writes the metadata types present in this struct
   */
   enum pipe_video_feedback_metadata_type present_metadata;

   /*
    * Driver writes the result of encoding the associated frame.
    * Requires PIPE_VIDEO_FEEDBACK_METADATA_TYPE_ENCODE_RESULT
    */
   enum pipe_video_feedback_encode_result_flags encode_result;

   /*
    * Driver fills in with coded headers information
    * and a number codec_unit_metadata_count of valid entries
    * Requires PIPE_VIDEO_FEEDBACK_METADATA_TYPE_CODEC_UNIT_LOCATION
    */
   struct codec_unit_location_t codec_unit_metadata[256];
   unsigned codec_unit_metadata_count;

   /*
   * Driver writes the average QP used to encode this frame
   */
   unsigned int average_frame_qp;
};

union pipe_enc_cap_roi {
   struct {
      /**
       * The number of ROI regions supported, 0 if ROI is not supported
       */
      uint32_t num_roi_regions                 : 8;
      /**
       * A flag indicates whether ROI priority is supported
       *
       * roi_rc_priority_support equal to 1 specifies the underlying driver supports
       * ROI priority when VAConfigAttribRateControl != VA_RC_CQP, user can use roi_value
       * in #VAEncROI to set ROI priority. roi_rc_priority_support equal to 0 specifies
       * the underlying driver doesn't support ROI priority.
       *
       * User should ignore roi_rc_priority_support when VAConfigAttribRateControl == VA_RC_CQP
       * because ROI delta QP is always required when VAConfigAttribRateControl == VA_RC_CQP.
       */
      uint32_t roi_rc_priority_support         : 1;
      /**
       * A flag indicates whether ROI delta QP is supported
       *
       * roi_rc_qp_delta_support equal to 1 specifies the underlying driver supports
       * ROI delta QP when VAConfigAttribRateControl != VA_RC_CQP, user can use roi_value
       * in #VAEncROI to set ROI delta QP. roi_rc_qp_delta_support equal to 0 specifies
       * the underlying driver doesn't support ROI delta QP.
       *
       * User should ignore roi_rc_qp_delta_support when VAConfigAttribRateControl == VA_RC_CQP
       * because ROI delta QP is always required when VAConfigAttribRateControl == VA_RC_CQP.
       */
      uint32_t roi_rc_qp_delta_support         : 1;
      /*
       * Indicates the minimum driver granularity to set a ROI region with a specific QP value.
       * In other words, the QP delta block granularity pixel size of the hardware encoder.
       */
      uint32_t log2_roi_min_block_pixel_size   : 4;
      uint32_t reserved                        : 18;

   } bits;
   uint32_t value;
};

union pipe_enc_cap_surface_alignment {
   struct {
      /**
       * log2_width_alignment
       */
      uint32_t log2_width_alignment                 : 4;
      /**
       * log2_height_alignment
       */
      uint32_t log2_height_alignment                : 4;
      uint32_t reserved                             : 24;
   } bits;
   uint32_t value;
};

/* To be used with PIPE_VIDEO_CAP_ENC_HEVC_RANGE_EXTENSION_SUPPORT */
union pipe_h265_enc_cap_range_extension {
   struct {
      /* Driver output. A bitmask indicating which values are allowed to be configured when encoding with diff_cu_chroma_qp_offset_depth.
      * Codec valid range for support for diff_cu_chroma_qp_offset_depth is [0, 3].
      * For driver to indicate that value is supported, it must set the following in the reported bitmask.
      * supported_diff_cu_chroma_qp_offset_depth_values |= (1 << value)
      */
      uint32_t supported_diff_cu_chroma_qp_offset_depth_values: 4;
      /* Driver output. A bitmask indicating which values are allowed to be configured when encoding with log2_sao_offset_scale_luma.
      * Codec valid range for support for log2_sao_offset_scale_luma is [0, 6].
      * For driver to indicate that value is supported, it must set the following in the reported bitmask.
      * supported_log2_sao_offset_scale_luma_values |= (1 << value)
      */
      uint32_t supported_log2_sao_offset_scale_luma_values: 7;
      /* Driver output. A bitmask indicating which values are allowed to be configured when encoding with log2_sao_offset_scale_chroma.
      * Codec valid range for support for log2_sao_offset_scale_chroma is [0, 6].
      * For driver to indicate that value is supported, it must set the following in the reported bitmask.
      * supported_log2_sao_offset_scale_chroma_values |= (1 << value)
      */
      uint32_t supported_log2_sao_offset_scale_chroma_values: 7;
      /* Driver output. A bitmask indicating which values are allowed to be configured when encoding with log2_max_transform_skip_block_size_minus2.
      * Codec valid range for support for log2_max_transform_skip_block_size_minus2 is [0, 3].
      * For driver to indicate that value is supported, it must set the following in the reported bitmask.
      * supported_log2_max_transform_skip_block_size_minus2_values |= (1 << value)
      */
      uint32_t supported_log2_max_transform_skip_block_size_minus2_values: 6;
      /* Driver output.The minimum value allowed to be configured when encoding with chroma_qp_offset_list_len_minus1.
      * Codec valid range for support for chroma_qp_offset_list_len_minus1 is [0, 5].
      */
      uint32_t min_chroma_qp_offset_list_len_minus1_values: 3;
      /* Driver output.The maximum value allowed to be configured when encoding with chroma_qp_offset_list_len_minus1.
      * Codec valid range for support for chroma_qp_offset_list_len_minus1 is [0, 5].
      */
      uint32_t max_chroma_qp_offset_list_len_minus1_values: 3;
   } bits;
  uint32_t value;
};

union pipe_h265_enc_cap_range_extension_flags {
   struct {
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting transform_skip_rotation_enabled_flag
       */
      uint32_t supports_transform_skip_rotation_enabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting transform_skip_context_enabled_flag
       */
      uint32_t supports_transform_skip_context_enabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting implicit_rdpcm_enabled_flag
       */
      uint32_t supports_implicit_rdpcm_enabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting explicit_rdpcm_enabled_flag
       */
      uint32_t supports_explicit_rdpcm_enabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting extended_precision_processing_flag
       */
      uint32_t supports_extended_precision_processing_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting intra_smoothing_disabled_flag
       */
      uint32_t supports_intra_smoothing_disabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting high_precision_offsets_enabled_flag
       */
      uint32_t supports_high_precision_offsets_enabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting persistent_rice_adaptation_enabled_flag
       */
      uint32_t supports_persistent_rice_adaptation_enabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting cabac_bypass_alignment_enabled_flag
       */
      uint32_t supports_cabac_bypass_alignment_enabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting cross_component_prediction_enabled_flag
       */
      uint32_t supports_cross_component_prediction_enabled_flag: 2;
      /*
       * Driver Output. Indicates pipe_enc_feature values for setting chroma_qp_offset_list_enabled_flag
       */
      uint32_t supports_chroma_qp_offset_list_enabled_flag: 2;
   } bits;
  uint32_t value;
};

/* Used with PIPE_VIDEO_CAP_ENC_DIRTY_RECTS */
/* Used with PIPE_VIDEO_CAP_ENC_DIRTY_MAPS */
union pipe_enc_cap_dirty_info {
   struct {
      /*
       * Driver Output. Indicates support values for setting pipe_enc_dirty_info.full_frame_skip
       */
      uint32_t supports_full_frame_skip: 1;
      /*
       * Driver Output. Indicates support values for setting pipe_enc_dirty_info.dirty_info_type = PIPE_ENC_DIRTY_INFO_TYPE_SKIP
       */
      uint32_t supports_info_type_skip: 1;
      /*
       * Driver Output. Indicates support values for setting pipe_enc_dirty_info.dirty_info_type = PIPE_ENC_DIRTY_INFO_TYPE_DIRTY
       */
      uint32_t supports_info_type_dirty: 1;
      /*
       * Driver Output. Indicates pipe_enc_dirty_info.rects/input_map must cover an entire row of the frame
       */
      uint32_t supports_require_full_row: 1;
      /*
       * Driver Output. Indicates that to use pipe_enc_dirty_info.rects/input_map the frame must be encoded with PIPE_VIDEO_SLICE_MODE_AUTO
       * where the driver decides the slice partition for the frame.
       */
      uint32_t supports_require_auto_slice_mode: 1;
      /*
       * Driver Output. Indicates that to use pipe_enc_dirty_info.rects/input_map the frame must be encoded with SAO filter disabled
       * For example for HEVC would correspond to pipe_h265_enc_picture_desc.seq.sample_adaptive_offset_enabled_flag)
       */
      uint32_t supports_require_sao_filter_disabled: 1;
      /*
       * Driver Output. Indicates that to use pipe_enc_dirty_info.rects/input_map the frame must be encoded with LOOP filter disabled
       * For example for HEVC would correspond to pipe_h265_enc_picture_desc.pic.pps_loop_filter_across_slices_enabled_flag)
       */
      uint32_t supports_require_loop_filter_disabled: 1;
   } bits;
  uint32_t value;
};

/* Used with PIPE_VIDEO_CAP_ENC_MOVE_RECTS */
union pipe_enc_cap_move_rect {
   struct {
      /*
       * Driver Output. Indicates support for setting up to max_motion_hints in pipe_enc_move_info.rects when max_motion_hints > 0.
       */
      uint32_t max_motion_hints: 16;
      /*
       * Driver Output. Indicates support for sending overlapped rects in pipe_enc_move_info.rects
       */
      uint32_t supports_overlapped_rects: 1;
      /*
       * Driver Output. Indicates support for setting in pipe_enc_move_info.precision = PIPE_ENC_MOVE_INFO_PRECISION_UNIT_FULL_PIXEL
       */
      uint32_t supports_precision_full_pixel: 1;
      /*
       * Driver Output. Indicates support for setting in pipe_enc_move_info.precision = PIPE_ENC_MOVE_INFO_PRECISION_UNIT_HALF_PIXEL
       */
      uint32_t supports_precision_half_pixel: 1;
      /*
       * Driver Output. Indicates support for setting in pipe_enc_move_info.precision = PIPE_ENC_MOVE_INFO_PRECISION_UNIT_QUARTER_PIXEL
       */
      uint32_t supports_precision_quarter_pixel: 1;
   } bits;
  uint32_t value;
};

/* Used with PIPE_VIDEO_CAP_ENC_GPU_STATS_QP_MAP */
/* Used with PIPE_VIDEO_CAP_ENC_GPU_STATS_SATD_MAP */
/* Used with PIPE_VIDEO_CAP_ENC_GPU_STATS_RATE_CONTROL_BITS_MAP */
union pipe_enc_cap_gpu_stats_map {
   struct {
      /*
       * Driver Output. Indicates support for writing this map
         into a GPU resource during encode frame execution
       */
      uint32_t supported: 1;
      /*
       * Driver Output. Indicates the pipe_format required for
         the pipe_resource allocation passed to the driver
       */
      uint32_t pipe_pixel_format: 9; /* 9 bits for pipe_format < PIPE_FORMAT_COUNT */
      /*
       * Driver Output. Indicates the pixel size of the blocks containing
         the stats. For example log2_values_block_size=4 indicates that
         the stats blocks will correspond to 16x16 blocks. This also indicates
         the dimensions of the pipe_resource allocation passed to the driver
         as the encoded frame dimensions (rounded up to codec block size) divided
         by 2^log2_values_block_size
       */
      uint32_t log2_values_block_size: 4;
   } bits;
  uint32_t value;
};

/* Used with PIPE_VIDEO_CAP_ENC_GPU_STATS_PSNR */
union pipe_enc_cap_gpu_stats_psnr {
   struct {
      /*
       * Driver Output. Indicates support for writing the frame's PSNR into a
       * PIPE_BUFFER during encode frame execution as 32 bit float components
       *
       * The components are always written in Y, U, V order as packed
       * 32 bit floats in the gpu_stats_psnr buffer sent down to the driver
       * in the picture parameters. The unsupported channels are written as zero.
       */
      uint32_t supports_y_channel: 1;
      uint32_t supports_u_channel: 1;
      uint32_t supports_v_channel: 1;
   } bits;
  uint32_t value;
};

/* Used with PIPE_VIDEO_CAP_ENC_SLICED_NOTIFICATIONS */
union pipe_enc_cap_sliced_notifications {
   struct {
      /*
       * Driver Output. Indicates support for sliced notifications
       */
      uint32_t supported: 1;
      /*
       * Driver Output. When reported indicates that encode_bitstream_sliced
       * parameter slice_destinations[i] must contain different pipe_resource
       * allocations for each slice i.
       *
       * When NOT reported, indicates that encode_bitstream_sliced
       * parameter slice_destinations[i] must contain the same pipe_resource
       * allocation for each slice i. In this case the driver will take care
       * of suballocating and calculating offsets within the single allocation
       */
      uint32_t multiple_buffers_required: 1;
   } bits;
  uint32_t value;
};

/* Used with PIPE_VIDEO_CAP_ENC_QP_MAPS */
union pipe_enc_cap_qpmap {
   struct {
      /*
       * Driver Output. Indicates support for accepting a QP map input_gpu_qpmap
         as a GPU resource input during encode frame execution
       */
      uint32_t supported: 1;
      /*
       * Driver Output. Indicates the pipe_format required for
         the pipe_resource allocation passed to the driver
       */
      uint32_t pipe_pixel_format: 9; /* 9 bits for pipe_format < PIPE_FORMAT_COUNT */
      /*
       * Driver Output. Indicates the pixel size of the blocks containing
         the QP values. For example log2_values_block_size=4 indicates that
         the QP blocks will correspond to 16x16 blocks. This also indicates
         the dimensions of the pipe_resource allocation passed to the driver
         as the encoded frame dimensions (rounded up to codec block size) divided
         by 2^log2_values_block_size
       */
      uint32_t log2_values_block_size: 4;
   } bits;
  uint32_t value;
};

/* Used with PIPE_VIDEO_CAP_ENC_TWO_PASS */
union pipe_enc_cap_two_pass {
   struct {
      /*
       * Driver Output. Indicates support for setting pipe_enc_two_pass_encoder_config.enable
       */
      uint32_t supports_two_pass: 1;
      /*
       * Driver Output. Indicates minimum value supported for pipe_enc_two_pass_encoder_config.pow2_downscale_factor
       */
      uint32_t min_pow2_downscale_factor: 3;
      /*
       * Driver Output. Indicates maximum value supported for pipe_enc_two_pass_encoder_config.pow2_downscale_factor
       */
      uint32_t max_pow2_downscale_factor: 3;
      /*
       * Driver output. Indicates support for skipping the first pass
       * dpb output generation by setting pipe_enc_two_pass_encoder_config.skip_1st_dpb_texture = true
       * and instead having the caller externally
       * downscaling the dpb textures from the 2nd pass recon dpb texture.
       *
       * Using external downscaling may provide better quality at a higher
       * performance cost, apps can choose this as a trade-off.
       *
       * When this mode is enabled, the driver will only generate the 2nd pass recon
       * if the frame is used as reference and the app will be responsible for
       * generating the downscaled dpb texture from the 2nd pass recon dpb texture
       * to pass to future frames that use the reference.
       *
       * If not supported, the 1st pass recon pic output will always
       * be written by the driver
       */
      uint32_t supports_1pass_recon_writing_skip: 1;
      /*
       * Driver output. Indicates support for skipping the first pass
       * on arbitrary frames by setting pipe_enc_two_pass_frame_config.skip_1st_pass = true
       * in the picture params.
       *
       * If not supported, all supported frame types by the driver will have
       * two pass enabled during the encoder lifetime, ignoring the value in
       * pipe_enc_two_pass_frame_config.skip_1st_pass
       */
      uint32_t supports_dynamic_1st_pass_skip: 1;
   } bits;
  uint32_t value;
};

/* Used with PIPE_VIDEO_CAP_ENC_MOTION_VECTOR_MAPS */
union pipe_enc_cap_motion_vector_map {
   struct {
      /*
       * Driver Output. Indicates how many hint maps are supported in the
       * motion maps array. Zero indicates no support
       * for the feature.
       */
      uint32_t max_motion_hints: 5; /* Max 31 hints. */
      /*
       * Driver Output. Indicates the precision of the motion vectors passed in the motion vectors map
       */
      /*
       * Driver Output. Indicates support for setting in pipe_enc_move_info.precision = PIPE_ENC_MOVE_INFO_PRECISION_UNIT_FULL_PIXEL
       */
      uint32_t supports_precision_full_pixel: 1;
      /*
       * Driver Output. Indicates support for setting in pipe_enc_move_info.precision = PIPE_ENC_MOVE_INFO_PRECISION_UNIT_HALF_PIXEL
       */
      uint32_t supports_precision_half_pixel: 1;
      /*
       * Driver Output. Indicates support for setting in pipe_enc_move_info.precision = PIPE_ENC_MOVE_INFO_PRECISION_UNIT_QUARTER_PIXEL
       */
      uint32_t supports_precision_quarter_pixel: 1;
      /*
       * Driver Output. Indicates if the different motion vectors
       * in the maps can point to different references in the DPB
       *  or all must point to the same one on a given frame.
       */
      uint32_t support_multiple_dpb_refs: 1;
      /*
       * Driver Output. Indicates the pipe_format required for
       * the pipe_resource allocation with the motion vectors map
       *  passed to the driver
       */
      uint32_t pipe_pixel_vectors_map_format: 9; /* 9 bits for pipe_format < PIPE_FORMAT_COUNT */
      /*
       * Driver Output. Indicates the pipe_format required for
       * the pipe_resource allocation with the motion vectors metadata map
       * passed to the driver
       */
      uint32_t pipe_pixel_vectors_metadata_map_format: 9; /* 9 bits for pipe_format < PIPE_FORMAT_COUNT */
   } bits;
  uint32_t value;
};

#ifdef __cplusplus
}
#endif

#endif /* PIPE_VIDEO_STATE_H */
