/*
 * Copyright (c) 2026 Igalia S.L.
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
 */

#ifndef GENX_HUC_BRC_H
#define GENX_HUC_BRC_H

#include <math.h>

#include "genX_huc.h"
#include "anv_video_brc_tables.h"

/* AVC (H.264) VDENC HuC BRC firmware ABI. Unlike HEVC, the AVC BRC DMEM
 * layout is identical on gen12 (ADL-P) and gen12.5 (DG2): DG2 reuses the
 * gen12 encode class, so a single definition serves both. Init DMEM is 192
 * bytes, update DMEM is 448 bytes. Kernel descriptors and BRC flags differ
 * from the HEVC ones.
 */
#define ANV_HUC_AVC_BRC_INIT_KERNEL_DESCRIPTOR    4
#define ANV_HUC_AVC_BRC_UPDATE_KERNEL_DESCRIPTOR  5
#define ANV_HUC_AVC_BRC_FLAG_ICQ                  0x0000
#define ANV_HUC_AVC_BRC_FLAG_CBR                  0x0010
#define ANV_HUC_AVC_BRC_FLAG_VBR                  0x0020
#define ANV_HUC_AVC_BRC_FLAG_VCM                  0x0040
#define ANV_HUC_AVC_BRC_FLAG_LOWDELAY             0x0080

struct anv_huc_avc_brc_init_dmem {
   uint8_t  BRCFunc_U8;
   uint8_t  OpenSourceEnable_U8;
   uint8_t  RVSD[2];
   uint16_t INIT_BRCFlag_U16;
   uint16_t Reserved;
   uint16_t INIT_FrameWidth_U16;
   uint16_t INIT_FrameHeight_U16;
   uint32_t INIT_TargetBitrate_U32;
   uint32_t INIT_MinRate_U32;
   uint32_t INIT_MaxRate_U32;
   uint32_t INIT_BufSize_U32;
   uint32_t INIT_InitBufFull_U32;
   uint32_t INIT_ProfileLevelMaxFrame_U32;
   uint32_t INIT_FrameRateM_U32;
   uint32_t INIT_FrameRateD_U32;
   uint16_t INIT_GopP_U16;
   uint16_t INIT_GopB_U16;
   uint16_t INIT_MinQP_U16;
   uint16_t INIT_MaxQP_U16;
   int8_t   INIT_DevThreshPB0_S8[8];
   int8_t   INIT_DevThreshVBR0_S8[8];
   int8_t   INIT_DevThreshI0_S8[8];
   uint8_t  INIT_InitQPIP;
   uint8_t  INIT_NotUseRhoDm_U8;
   uint8_t  INIT_InitQPB;
   uint8_t  INIT_MbQpCtrl_U8;
   uint8_t  INIT_SliceSizeCtrlEn_U8;
   int8_t   INIT_IntraQPDelta_I8[3];
   int8_t   INIT_SkipQPDelta_I8;
   int8_t   INIT_DistQPDelta_I8[4];
   uint8_t  INIT_OscillationQpDelta_U8;
   uint8_t  INIT_HRDConformanceCheckDisable_U8;
   uint8_t  INIT_SkipFrameEnableFlag;
   uint8_t  INIT_TopQPDeltaThrForAdapt2Pass_U8;
   uint8_t  INIT_TopFrmSzThrForAdapt2Pass_U8;
   uint8_t  INIT_BotFrmSzThrForAdapt2Pass_U8;
   uint8_t  INIT_QPSelectForFirstPass_U8;
   uint8_t  INIT_MBHeaderCompensation_U8;
   uint8_t  INIT_OverShootCarryFlag_U8;
   uint8_t  INIT_OverShootSkipFramePct_U8;
   uint8_t  INIT_EstRateThreshP0_U8[7];
   uint8_t  INIT_EstRateThreshB0_U8[7];
   uint8_t  INIT_EstRateThreshI0_U8[7];
   uint8_t  INIT_FracQPEnable_U8;
   uint8_t  INIT_ScenarioInfo_U8;
   uint8_t  INIT_StaticRegionStreamIn_U8;
   uint8_t  INIT_DeltaQP_Adaptation_U8;
   uint8_t  INIT_MaxCRFQualityFactor_U8;
   uint8_t  INIT_CRFQualityFactor_U8;
   uint8_t  INIT_BotQPDeltaThrForAdapt2Pass_U8;
   uint8_t  INIT_SlidingWindowSize_U8;
   uint8_t  INIT_SlidingWidowRCEnable_U8;
   uint8_t  INIT_SlidingWindowMaxRateRatio_U8;
   uint8_t  INIT_LowDelayGoldenFrameBoost_U8;
   uint8_t  INIT_AdaptiveCostEnable_U8;
   uint8_t  INIT_AdaptiveHMEExtensionEnable_U8;
   uint8_t  INIT_ICQReEncode_U8;
   uint8_t  INIT_LookaheadDepth_U8;
   uint8_t  INIT_SinglePassOnly;
   uint8_t  INIT_New_DeltaQP_Adaptation_U8;
   uint8_t  RSVD2[55];
};

struct anv_huc_avc_brc_update_dmem {
   uint8_t  BRCFunc_U8;
   uint8_t  RSVD[3];
   uint32_t UPD_TARGETSIZE_U32;
   uint32_t UPD_FRAMENUM_U32;
   uint32_t UPD_PeakTxBitsPerFrame_U32;
   uint32_t UPD_FrameBudget_U32;
   uint32_t FrameByteCount;
   uint32_t TimingBudgetOverflow;
   uint32_t ImgStatusCtrl;
   uint32_t IPCMNonConformant;
   uint16_t UPD_startGAdjFrame_U16[4];
   uint16_t UPD_MBBudget_U16[52];
   uint16_t UPD_SLCSZ_TARGETSLCSZ_U16;
   uint16_t UPD_SLCSZ_UPD_THRDELTAI_U16[42];
   uint16_t UPD_SLCSZ_UPD_THRDELTAP_U16[42];
   uint16_t UPD_NumOfFramesSkipped_U16;
   uint16_t UPD_SkipFrameSize_U16;
   uint16_t UPD_StaticRegionPct_U16;
   uint8_t  UPD_gRateRatioThreshold_U8[7];
   uint8_t  UPD_CurrFrameType_U8;
   uint8_t  UPD_startGAdjMult_U8[5];
   uint8_t  UPD_startGAdjDiv_U8[5];
   uint8_t  UPD_gRateRatioThresholdQP_U8[8];
   uint8_t  UPD_PAKPassNum_U8;
   uint8_t  UPD_MaxNumPass_U8;
   uint8_t  UPD_SceneChgWidth_U8[2];
   uint8_t  UPD_SceneChgDetectEn_U8;
   uint8_t  UPD_SceneChgPrevIntraPctThreshold_U8;
   uint8_t  UPD_SceneChgCurIntraPctThreshold_U8;
   uint8_t  UPD_IPAverageCoeff_U8;
   uint8_t  UPD_MinQpAdjustment_U8;
   uint8_t  UPD_TimingBudgetCheck_U8;
   int8_t   reserved_I8[4];
   uint8_t  UPD_CQP_QpValue_U8;
   uint8_t  UPD_CQP_FracQp_U8;
   uint8_t  UPD_HMEDetectionEnable_U8;
   uint8_t  UPD_HMECostEnable_U8;
   uint8_t  UPD_DisablePFrame8x8Transform_U8;
   uint8_t  RSVD3;
   uint8_t  UPD_ROISource_U8;
   uint8_t  RSVD4;
   uint16_t UPD_TargetSliceSize_U16;
   uint16_t UPD_MaxNumSliceAllowed_U16;
   uint16_t UPD_SLBB_Size_U16;
   uint16_t UPD_SLBB_B_Offset_U16;
   uint16_t UPD_AvcImgStateOffset_U16;
   uint16_t reserved_u16;
   uint32_t NumOfSlice;
   uint16_t AveHmeDist_U16;
   uint8_t  HmeDistAvailable_U8;
   uint8_t  DisableDMA;
   uint16_t AdditionalFrameSize_U16;
   uint8_t  AddNALHeaderSizeInternally_U8;
   uint8_t  UPD_RoiQpViaForceQp_U8;
   uint32_t CABACZeroInsertionSize_U32;
   uint32_t MiniFramePaddingSize_U32;
   uint16_t UPD_WidthInMB_U16;
   uint16_t UPD_HeightInMB_U16;
   int8_t   UPD_ROIQpDelta_I8[8];
   int8_t   HME0XOffset_I8;
   int8_t   HME0YOffset_I8;
   int8_t   HME1XOffset_I8;
   int8_t   HME1YOffset_I8;
   uint8_t  MOTION_ADAPTIVE_G4;
   uint8_t  EnableLookAhead;
   uint8_t  UPD_LA_Data_Offset_U8;
   uint8_t  UPD_CQMEnabled_U8;
   uint32_t UPD_LA_TargetSize_U32;
   uint32_t UPD_LA_TargetFulness_U32;
   uint8_t  UPD_Delta_U8;
   uint8_t  UPD_ROM_CURRENT_U8;
   uint8_t  UPD_ROM_ZERO_U8;
   uint8_t  UPD_TCBRC_SCENARIO_U8;
   uint8_t  UPD_EnableFineGrainLA;
   int8_t   UPD_DeltaQpDcOffset;
   uint16_t UPD_NumSlicesForRounding;
   uint32_t UPD_UserMaxFramePB;
   uint8_t  RSVD2[4];
};

static_assert(sizeof(struct anv_huc_avc_brc_init_dmem) == 192, "huc avc brc init dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_init_dmem, INIT_BRCFlag_U16) == 4, "huc avc brc init dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_init_dmem, INIT_DevThreshPB0_S8) == 52, "huc avc brc init dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_init_dmem, INIT_InitQPIP) == 76, "huc avc brc init dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_init_dmem, INIT_EstRateThreshP0_U8) == 99, "huc avc brc init dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_init_dmem, INIT_New_DeltaQP_Adaptation_U8) == 136, "huc avc brc init dmem layout");

static_assert(sizeof(struct anv_huc_avc_brc_update_dmem) == 448, "huc avc brc update dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_update_dmem, UPD_TARGETSIZE_U32) == 4, "huc avc brc update dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_update_dmem, UPD_MBBudget_U16) == 44, "huc avc brc update dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_update_dmem, NumOfSlice) == 384, "huc avc brc update dmem layout");
static_assert(offsetof(struct anv_huc_avc_brc_update_dmem, UPD_UserMaxFramePB) == 440, "huc avc brc update dmem layout");


/* H.264 AVC BRC region-5 constant-data buffer (1728 bytes) and the source
 * tables the driver copies into it, extracted verbatim from the media-driver
 * (CBR/VBR scope). Field names follow the media layout for cross-reference.
 */
struct anv_huc_avc_brc_const_data {
   uint8_t UPD_GlobalRateQPAdjTabI_U8[64];
   uint8_t UPD_GlobalRateQPAdjTabP_U8[64];
   uint8_t UPD_GlobalRateQPAdjTabB_U8[64];
   uint8_t UPD_DistThreshldI_U8[10];
   uint8_t UPD_DistThreshldP_U8[10];
   uint8_t UPD_DistThreshldB_U8[10];
   uint8_t UPD_DistQPAdjTabI_U8[81];
   uint8_t UPD_DistQPAdjTabP_U8[81];
   uint8_t UPD_DistQPAdjTabB_U8[81];
   int8_t  UPD_BufRateAdjTabI_S8[72];
   int8_t  UPD_BufRateAdjTabP_S8[72];
   int8_t  UPD_BufRateAdjTabB_S8[72];
   uint8_t UPD_FrmSzMinTabP_U8[9];
   uint8_t UPD_FrmSzMinTabB_U8[9];
   uint8_t UPD_FrmSzMinTabI_U8[9];
   uint8_t UPD_FrmSzMaxTabP_U8[9];
   uint8_t UPD_FrmSzMaxTabB_U8[9];
   uint8_t UPD_FrmSzMaxTabI_U8[9];
   uint8_t UPD_FrmSzSCGTabP_U8[9];
   uint8_t UPD_FrmSzSCGTabB_U8[9];
   uint8_t UPD_FrmSzSCGTabI_U8[9];
   uint8_t UPD_I_IntraNonPred[42];
   uint8_t UPD_I_Intra16x16[42];
   uint8_t UPD_I_Intra8x8[42];
   uint8_t UPD_I_Intra4x4[42];
   uint8_t UPD_I_IntraChroma[42];
   uint8_t UPD_P_IntraNonPred[42];
   uint8_t UPD_P_Intra16x16[42];
   uint8_t UPD_P_Intra8x8[42];
   uint8_t UPD_P_Intra4x4[42];
   uint8_t UPD_P_IntraChroma[42];
   uint8_t UPD_P_Inter16x8[42];
   uint8_t UPD_P_Inter8x8[42];
   uint8_t UPD_P_Inter16x16[42];
   uint8_t UPD_P_RefId[42];
#if GFX_VERx10 >= 125
   uint8_t Reserved[630];
#else
   uint8_t UPD_HMEMVCost[8][42];
   uint8_t RSVD[42];
#endif
};

#if GFX_VERx10 >= 125
static_assert(sizeof(struct anv_huc_avc_brc_const_data) == 1980, "huc avc brc const data layout");
#else
static_assert(sizeof(struct anv_huc_avc_brc_const_data) == 1728, "huc avc brc const data layout");
#endif

static int32_t
anv_brc_estimate_init_qp(double frame_size, double frame_rate, double bitrate)
{
   const double x0 = 0.0, y0 = 1.19, x1 = 1.75, y1 = 1.75;
   double x = log10(frame_size * 2.0 / 3.0 * frame_rate / bitrate);

   return (int32_t)(1.0 / 1.2 *
                    pow(10.0, (x - x0) * (y1 - y0) / (x1 - x0) + y0) + 0.5) + 2;
}

#if GFX_VER >= 12
/* H.264 (AVC) HuC BRC building blocks. gen12+ only. Mirrors the HEVC BRC
 * helpers but targets the MFX PAK second level batch buffer and the AVC DMEM.
 */
/* Second level batch buffer (SLBB) image-state group size. gen120 is
 * MFX_AVC_IMG_STATE (84) + VDENC_IMG_STATE (140) + MI_BATCH_BUFFER_END (4) =
 * 228 bytes. gen125 replaces VDENC_IMG_STATE with VDENC_CMD3 +
 * VDENC_AVC_IMG_STATE. The HuC-patched SLBB carries only this image-state
 * group on both generations; the PAK slices stay in the primary batch.
 */
#if GFX_VERx10 < 125
#define ANV_H264_BRC_SLB_SIZE \
   ((GENX(MFX_AVC_IMG_STATE_length) + GENX(VDENC_IMG_STATE_length) + \
     GENX(MI_BATCH_BUFFER_END_length)) * 4)
#else
#define ANV_H264_BRC_SLB_SIZE \
   ((GENX(MFX_AVC_IMG_STATE_length) + GENX(VDENC_CMD3_length) + \
     GENX(VDENC_AVC_IMG_STATE_length) + GENX(MI_BATCH_BUFFER_END_length)) * 4)
#endif

/* 4-4 (exp/mantissa) LUT compression, from media-driver */
static uint8_t
anv_h264_brc_map44_lut(uint32_t v, uint8_t max)
{
   if (v == 0)
      return 0;
   uint32_t max_cost = (uint32_t)(max & 0xf) << (max >> 4);
   if (v >= max_cost)
      return max;
   int d = (int)(31 - __builtin_clz(v)) - 3;
   if (d < 0)
      d = 0;
   uint32_t ret = ((uint32_t)d << 4) +
                  ((v + (d ? (1u << (d - 1)) : 0)) >> d);
   if ((ret & 0xf) == 0)
      ret |= 8;
   return (uint8_t)ret;
}

static void
anv_h264_brc_fill_const_data(struct anv_video_session *vid,
                             StdVideoH264PictureType pic_type,
                             struct anv_huc_avc_brc_const_data *cd)
{
   bool is_cbr = vid->rc_mode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR;

   memcpy(cd->UPD_GlobalRateQPAdjTabI_U8, anv_h264_brc_global_rate_qp_adj_i,
          sizeof(cd->UPD_GlobalRateQPAdjTabI_U8));
   memcpy(cd->UPD_GlobalRateQPAdjTabP_U8, anv_h264_brc_global_rate_qp_adj_p,
          sizeof(cd->UPD_GlobalRateQPAdjTabP_U8));
   memcpy(cd->UPD_GlobalRateQPAdjTabB_U8, anv_h264_brc_global_rate_qp_adj_b,
          sizeof(cd->UPD_GlobalRateQPAdjTabB_U8));

   memcpy(cd->UPD_DistThreshldI_U8, anv_h264_brc_dist_threshld_i,
          sizeof(cd->UPD_DistThreshldI_U8));
   memcpy(cd->UPD_DistThreshldP_U8, anv_h264_brc_dist_threshld_p,
          sizeof(cd->UPD_DistThreshldP_U8));
   memcpy(cd->UPD_DistThreshldB_U8, anv_h264_brc_dist_threshld_p,
          sizeof(cd->UPD_DistThreshldB_U8));

   if (is_cbr) {
      memcpy(cd->UPD_DistQPAdjTabI_U8, anv_h264_brc_cbr_dist_qp_adj_i,
             sizeof(cd->UPD_DistQPAdjTabI_U8));
      memcpy(cd->UPD_DistQPAdjTabP_U8, anv_h264_brc_cbr_dist_qp_adj_p,
             sizeof(cd->UPD_DistQPAdjTabP_U8));
      memcpy(cd->UPD_DistQPAdjTabB_U8, anv_h264_brc_cbr_dist_qp_adj_b,
             sizeof(cd->UPD_DistQPAdjTabB_U8));
      memcpy(cd->UPD_BufRateAdjTabI_S8, anv_h264_brc_cbr_frm_sz_adj_i,
             sizeof(cd->UPD_BufRateAdjTabI_S8));
      memcpy(cd->UPD_BufRateAdjTabP_S8, anv_h264_brc_cbr_frm_sz_adj_p,
             sizeof(cd->UPD_BufRateAdjTabP_S8));
      memcpy(cd->UPD_BufRateAdjTabB_S8, anv_h264_brc_cbr_frm_sz_adj_b,
             sizeof(cd->UPD_BufRateAdjTabB_S8));
   } else {
      memcpy(cd->UPD_DistQPAdjTabI_U8, anv_h264_brc_vbr_dist_qp_adj_i,
             sizeof(cd->UPD_DistQPAdjTabI_U8));
      memcpy(cd->UPD_DistQPAdjTabP_U8, anv_h264_brc_vbr_dist_qp_adj_p,
             sizeof(cd->UPD_DistQPAdjTabP_U8));
      memcpy(cd->UPD_DistQPAdjTabB_U8, anv_h264_brc_vbr_dist_qp_adj_b,
             sizeof(cd->UPD_DistQPAdjTabB_U8));
      memcpy(cd->UPD_BufRateAdjTabI_S8, anv_h264_brc_vbr_frm_sz_adj_i,
             sizeof(cd->UPD_BufRateAdjTabI_S8));
      memcpy(cd->UPD_BufRateAdjTabP_S8, anv_h264_brc_vbr_frm_sz_adj_p,
             sizeof(cd->UPD_BufRateAdjTabP_S8));
      memcpy(cd->UPD_BufRateAdjTabB_S8, anv_h264_brc_vbr_frm_sz_adj_b,
             sizeof(cd->UPD_BufRateAdjTabB_S8));
   }

   memcpy(cd->UPD_FrmSzMinTabP_U8, anv_h264_brc_frm_sz_min_p,
          sizeof(cd->UPD_FrmSzMinTabP_U8));
   memcpy(cd->UPD_FrmSzMinTabI_U8, anv_h264_brc_frm_sz_min_i,
          sizeof(cd->UPD_FrmSzMinTabI_U8));
   memcpy(cd->UPD_FrmSzMaxTabP_U8, anv_h264_brc_frm_sz_max_p,
          sizeof(cd->UPD_FrmSzMaxTabP_U8));
   memcpy(cd->UPD_FrmSzMaxTabI_U8, anv_h264_brc_frm_sz_max_i,
          sizeof(cd->UPD_FrmSzMaxTabI_U8));
   memcpy(cd->UPD_FrmSzSCGTabP_U8, anv_h264_brc_frm_sz_scg_p,
          sizeof(cd->UPD_FrmSzSCGTabP_U8));
   memcpy(cd->UPD_FrmSzSCGTabI_U8, anv_h264_brc_frm_sz_scg_i,
          sizeof(cd->UPD_FrmSzSCGTabI_U8));

   memcpy(cd->UPD_I_IntraNonPred, anv_h264_brc_cost_i_intra_non_pred,
          sizeof(cd->UPD_I_IntraNonPred));
   memcpy(cd->UPD_I_Intra8x8, anv_h264_brc_cost_i_intra8x8,
          sizeof(cd->UPD_I_Intra8x8));
   memcpy(cd->UPD_I_Intra4x4, anv_h264_brc_cost_i_intra4x4,
          sizeof(cd->UPD_I_Intra4x4));
   memcpy(cd->UPD_P_IntraNonPred, anv_h264_brc_cost_p_intra_non_pred,
          sizeof(cd->UPD_P_IntraNonPred));
   memcpy(cd->UPD_P_Intra16x16, anv_h264_brc_cost_p_intra16x16,
          sizeof(cd->UPD_P_Intra16x16));
   memcpy(cd->UPD_P_Intra8x8, anv_h264_brc_cost_p_intra8x8,
          sizeof(cd->UPD_P_Intra8x8));
   memcpy(cd->UPD_P_Intra4x4, anv_h264_brc_cost_p_intra4x4,
          sizeof(cd->UPD_P_Intra4x4));
   memcpy(cd->UPD_P_Inter16x8, anv_h264_brc_cost_p_inter16x8,
          sizeof(cd->UPD_P_Inter16x8));
   memcpy(cd->UPD_P_Inter8x8, anv_h264_brc_cost_p_inter8x8,
          sizeof(cd->UPD_P_Inter8x8));
   memcpy(cd->UPD_P_Inter16x16, anv_h264_brc_cost_p_inter16x16,
          sizeof(cd->UPD_P_Inter16x16));
   memcpy(cd->UPD_P_RefId, anv_h264_brc_cost_p_ref_id, sizeof(cd->UPD_P_RefId));

#if GFX_VERx10 >= 125
   uint32_t gop_ref_dist = MAX2(vid->rc.consecutive_b_frames + 1, 1);
   uint32_t type;
   if (pic_type == STD_VIDEO_H264_PICTURE_TYPE_I ||
       pic_type == STD_VIDEO_H264_PICTURE_TYPE_IDR)
      type = 0;
   else if (pic_type == STD_VIDEO_H264_PICTURE_TYPE_P)
      type = gop_ref_dist == 1 ? 2 : 1;
   else
      type = 3;
   memcpy(cd->Reserved, h264_vdenc_brc_huc_const_reserved[type],
          sizeof(cd->Reserved));
#else
   for (uint32_t i = 0; i < 8; i++)
      for (uint32_t j = 0; j < 42; j++)
         cd->UPD_HMEMVCost[i][j] =
            anv_h264_brc_map44_lut(anv_h264_brc_hme_cost[i][j + 10], 0x6f);
#endif
}

static void
anv_h264_brc_fill_init_dmem(struct anv_cmd_buffer *cmd,
                            const StdVideoH264SequenceParameterSet *sps,
                            struct anv_huc_avc_brc_init_dmem *dmem)
{
   struct anv_video_session *vid = cmd->video.vid;
   struct anv_video_rc_state *rc = &vid->rc;
   bool is_cbr = vid->rc_mode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR;
   uint32_t width = (sps->pic_width_in_mbs_minus1 + 1) * 16;
   uint32_t height = (sps->pic_height_in_map_units_minus1 + 1) * 16;
   uint32_t frame_rate_num = rc->frame_rate_num ? rc->frame_rate_num : 30;
   uint32_t frame_rate_den = rc->frame_rate_den ? rc->frame_rate_den : 1;
   uint32_t frames_per_100s = frame_rate_num * 100 / frame_rate_den;
   if (frames_per_100s == 0)
      frames_per_100s = 3000;
   uint32_t target_bps = DIV_ROUND_UP(rc->average_bitrate, 1000) * 1000;
   uint32_t max_bps = is_cbr ? target_bps :
                      DIV_ROUND_UP(rc->max_bitrate, 1000) * 1000;
   uint32_t buf_size = rc->vbv_size_bits;
   uint32_t gop = rc->gop_frame_count ? rc->gop_frame_count : 30;
   uint32_t gop_ref_dist = MAX2(rc->consecutive_b_frames + 1, 1);

   dmem->BRCFunc_U8 = rc->frame_counter ? 2 : 0;
   dmem->INIT_BRCFlag_U16 =
      is_cbr ? ANV_HUC_AVC_BRC_FLAG_CBR : ANV_HUC_AVC_BRC_FLAG_VBR;
   dmem->INIT_FrameWidth_U16 = width;
   dmem->INIT_FrameHeight_U16 = height;
   dmem->INIT_TargetBitrate_U32 = target_bps;
   dmem->INIT_MaxRate_U32 = max_bps;
   /* Set the BRC min rate to the target bitrate for both CBR
    * (min = max = target) and VBR (min = target, max = maxrate). */
   dmem->INIT_MinRate_U32 = target_bps;
   dmem->INIT_BufSize_U32 = buf_size;
   dmem->INIT_InitBufFull_U32 = MIN2(rc->vbv_initial_fullness_bits, buf_size);
   /* TODO: proper profile/level based max frame size
    * This MB-count heuristic matches the media value at 720p/1080p. */
   dmem->INIT_ProfileLevelMaxFrame_U32 =
      (sps->pic_width_in_mbs_minus1 + 1) *
      (sps->pic_height_in_map_units_minus1 + 1) * 96;
   dmem->INIT_FrameRateM_U32 = frames_per_100s;
   dmem->INIT_FrameRateD_U32 = 100;

   uint32_t gop_p = gop > 1 ? (gop - 1) / gop_ref_dist : 0;
   uint32_t gop_b = gop_ref_dist > 1 ? (gop_ref_dist - 1) * gop_p : 0;
   dmem->INIT_GopP_U16 = gop_p;
   dmem->INIT_GopB_U16 = gop_b;
   dmem->INIT_MinQP_U16 = 10;
   dmem->INIT_MaxQP_U16 = 51;

   double bps_ratio = buf_size ? (double)max_bps / buf_size : 1.0;
   bps_ratio = CLAMP(bps_ratio, 0.1, 3.5);
   static const double pb_neg[4] = { 0.90, 0.66, 0.46, 0.3 };
   static const double pb_pos[4] = { 0.3, 0.46, 0.70, 0.90 };
   static const double i_neg[4] = { 0.80, 0.60, 0.34, 0.2 };
   static const double i_pos[4] = { 0.2, 0.4, 0.66, 0.9 };
   static const double vbr_neg[4] = { 0.90, 0.70, 0.50, 0.3 };
   static const double vbr_pos[4] = { 0.4, 0.5, 0.75, 0.90 };
   for (uint32_t i = 0; i < 4; i++) {
      dmem->INIT_DevThreshPB0_S8[i] = (int8_t)(-50.0 * pow(pb_neg[i], bps_ratio));
      dmem->INIT_DevThreshPB0_S8[i + 4] = (int8_t)(50.0 * pow(pb_pos[i], bps_ratio));
      dmem->INIT_DevThreshI0_S8[i] = (int8_t)(-50.0 * pow(i_neg[i], bps_ratio));
      dmem->INIT_DevThreshI0_S8[i + 4] = (int8_t)(50.0 * pow(i_pos[i], bps_ratio));
      dmem->INIT_DevThreshVBR0_S8[i] = (int8_t)(-50.0 * pow(vbr_neg[i], bps_ratio));
      dmem->INIT_DevThreshVBR0_S8[i + 4] = (int8_t)(100.0 * pow(vbr_pos[i], bps_ratio));
   }

   static const uint8_t est_rate_thresh[7] = { 4, 8, 12, 16, 20, 24, 28 };
   memcpy(dmem->INIT_EstRateThreshP0_U8, est_rate_thresh,
          sizeof(dmem->INIT_EstRateThreshP0_U8));
   memcpy(dmem->INIT_EstRateThreshB0_U8, est_rate_thresh,
          sizeof(dmem->INIT_EstRateThreshB0_U8));
   memcpy(dmem->INIT_EstRateThreshI0_U8, est_rate_thresh,
          sizeof(dmem->INIT_EstRateThreshI0_U8));

   int32_t qp = anv_brc_estimate_init_qp((double)(width * height * 3 / 2),
                                         frames_per_100s,
                                         (double)target_bps * 100.0);
   int32_t delta_q =
      (int32_t)(9 - (double)buf_size * frames_per_100s / ((double)target_bps * 100.0));
   qp += delta_q < 0 ? 0 : delta_q;
   qp = CLAMP(qp, 1, 51);
   qp--;
   if (qp < 0)
      qp = 1;
   dmem->INIT_InitQPIP = qp;

   /* Enable fractional QP (extended rho domain).
    * The BRC update's rate model relies on it.
    */
   dmem->INIT_FracQPEnable_U8 = 1;
   dmem->INIT_QPSelectForFirstPass_U8 = 1;
   dmem->INIT_MBHeaderCompensation_U8 = 1;
   dmem->INIT_DeltaQP_Adaptation_U8 = 1;
   dmem->INIT_MaxCRFQualityFactor_U8 = 52;
   dmem->INIT_TopQPDeltaThrForAdapt2Pass_U8 = 2;
   dmem->INIT_BotQPDeltaThrForAdapt2Pass_U8 = 1;
   dmem->INIT_TopFrmSzThrForAdapt2Pass_U8 = 32;
   dmem->INIT_BotFrmSzThrForAdapt2Pass_U8 = 24;
}

static void
anv_h264_brc_fill_update_dmem(struct anv_cmd_buffer *cmd,
                              const StdVideoH264SequenceParameterSet *sps,
                              StdVideoH264PictureType pic_type,
                              uint32_t pass,
                              struct anv_huc_avc_brc_update_dmem *dmem)
{
   struct anv_video_session *vid = cmd->video.vid;
   struct anv_video_rc_state *rc = &vid->rc;
   uint32_t frame_rate_num = rc->frame_rate_num ? rc->frame_rate_num : 30;
   uint32_t frame_rate_den = rc->frame_rate_den ? rc->frame_rate_den : 1;
   uint32_t frames_per_100s = frame_rate_num * 100 / frame_rate_den;
   if (frames_per_100s == 0)
      frames_per_100s = 3000;
   uint32_t target_bps = DIV_ROUND_UP(rc->average_bitrate, 1000) * 1000;
   uint32_t input_bits_per_frame = frames_per_100s ?
      (uint32_t)((uint64_t)target_bps * 100 / frames_per_100s) : 0;
   uint32_t gop = rc->gop_frame_count ? rc->gop_frame_count : 30;
   uint32_t gop_ref_dist = MAX2(rc->consecutive_b_frames + 1, 1);
   uint32_t num_p = gop > 1 ? (gop - 1) / gop_ref_dist : 0;
   uint8_t scene_chg_width = MIN2((num_p + 1) / 5, 6);

   dmem->BRCFunc_U8 = 1;
   uint64_t target_acc =
      (uint64_t)MIN2(rc->vbv_initial_fullness_bits, rc->vbv_size_bits) +
      (uint64_t)(rc->frame_counter - rc->brc_init_frame) * input_bits_per_frame;
   dmem->UPD_TARGETSIZE_U32 = rc->vbv_size_bits ?
      (uint32_t)(target_acc % rc->vbv_size_bits) : (uint32_t)target_acc;

   dmem->UPD_FRAMENUM_U32 = rc->frame_counter;
   dmem->UPD_PeakTxBitsPerFrame_U32 = input_bits_per_frame;

   dmem->UPD_CurrFrameType_U8 =
      (pic_type == STD_VIDEO_H264_PICTURE_TYPE_I ||
       pic_type == STD_VIDEO_H264_PICTURE_TYPE_IDR) ? 2 :
      (pic_type == STD_VIDEO_H264_PICTURE_TYPE_P) ? 0 : 1;

   dmem->UPD_WidthInMB_U16 = sps->pic_width_in_mbs_minus1 + 1;
   dmem->UPD_HeightInMB_U16 = sps->pic_height_in_map_units_minus1 + 1;
   dmem->UPD_SLBB_Size_U16 = ANV_H264_BRC_SLB_SIZE;
   dmem->UPD_AvcImgStateOffset_U16 = 0;
   dmem->UPD_PAKPassNum_U8 = pass;
   dmem->UPD_MaxNumPass_U8 = 2;

   static const uint16_t start_gadj_frame[4] = { 10, 50, 100, 150 };
   static const uint8_t start_gadj_mult[5] = { 1, 1, 3, 2, 1 };
   static const uint8_t start_gadj_div[5] = { 40, 5, 5, 3, 1 };
   static const uint8_t rate_ratio_threshold[7] = { 80, 90, 95, 101, 105, 115, 130 };
   static const int8_t rate_ratio_threshold_qp[8] = { -3, -2, -1, 0, 1, 1, 2, 3 };
   memcpy(dmem->UPD_startGAdjFrame_U16, start_gadj_frame, sizeof(start_gadj_frame));
   memcpy(dmem->UPD_startGAdjMult_U8, start_gadj_mult, sizeof(start_gadj_mult));
   memcpy(dmem->UPD_startGAdjDiv_U8, start_gadj_div, sizeof(start_gadj_div));
   memcpy(dmem->UPD_gRateRatioThreshold_U8, rate_ratio_threshold, sizeof(rate_ratio_threshold));
   memcpy(dmem->UPD_gRateRatioThresholdQP_U8, rate_ratio_threshold_qp, sizeof(rate_ratio_threshold_qp));

   dmem->UPD_SceneChgWidth_U8[0] = scene_chg_width;
   dmem->UPD_SceneChgWidth_U8[1] = scene_chg_width;
   dmem->UPD_SceneChgDetectEn_U8 = 1;
   dmem->UPD_SceneChgPrevIntraPctThreshold_U8 = 96;
   dmem->UPD_SceneChgCurIntraPctThreshold_U8 = 192;
   dmem->UPD_IPAverageCoeff_U8 = 128;
   dmem->UPD_HMECostEnable_U8 = 1;
}

/* Defined in genX_cmd_video_enc.c; the BRC input SLB reuses the same image
 * state emission as the CQP encode path.
 */
static void
anv_video_emit_mi_flush_dw(struct anv_cmd_buffer *cmd, bool cache_invalidate);
static void
anv_vdenc_emit_vd_pipeline_flush(struct anv_cmd_buffer *cmd);

static void
anv_h264_emit_mfx_avc_img_state(struct anv_cmd_buffer *cmd,
                                struct anv_batch *batch,
                                const VkVideoEncodeInfoKHR *enc_info,
                                bool brc_enabled);
static void
anv_h264_emit_vdenc_img_state(struct anv_cmd_buffer *cmd,
                              struct anv_batch *batch,
                              const VkVideoEncodeInfoKHR *enc_info);

#if GFX_VERx10 >= 125
static void
anv_h264_emit_vdenc_cmd3(struct anv_cmd_buffer *cmd,
                         struct anv_batch *batch,
                         const VkVideoEncodeInfoKHR *enc_info);

static void
anv_h264_emit_vdenc_avc_img_state(struct anv_cmd_buffer *cmd,
                                  struct anv_batch *batch,
                                  const VkVideoEncodeInfoKHR *enc_info,
                                  const uint8_t *dpb_idx);
#endif

/* Build the SLBB input read buffer the HuC BRC Update kernel parses and
 * patches: the H.264 image-state group (MFX_AVC_IMG_STATE + VDENC image state
 * + MI_BATCH_BUFFER_END). The HuC copies it to the output SLBB while writing
 * the BRC-computed frame QP into VDENC_IMG_STATE.QPPRIMEY.
 */
static void
anv_h264_brc_build_input_slb(struct anv_cmd_buffer *cmd,
                             const VkVideoEncodeInfoKHR *enc_info,
                             const uint8_t *dpb_idx,
                             struct anv_state slb_state)
{
   struct anv_batch slb = { 0 };

   anv_batch_set_storage(&slb,
                         anv_cmd_buffer_temporary_state_address(cmd, slb_state),
                         slb_state.map, slb_state.alloc_size);
   memset(slb_state.map, 0, slb_state.alloc_size);

   anv_h264_emit_mfx_avc_img_state(cmd, &slb, enc_info, true);
#if GFX_VERx10 >= 125
   anv_h264_emit_vdenc_cmd3(cmd, &slb, enc_info);
   anv_h264_emit_vdenc_avc_img_state(cmd, &slb, enc_info, dpb_idx);
#else
   anv_h264_emit_vdenc_img_state(cmd, &slb, enc_info);
#endif
   anv_batch_emit(&slb, GENX(MI_BATCH_BUFFER_END), bbe);

   assert((uint32_t)(slb.next - slb.start) == ANV_H264_BRC_SLB_SIZE);
}

#define ANV_HUC_AVC_BRC_REENCODE_MASK  (1u << 31)
#define ANV_HUC_AVC_BRC_ERROR_MASK     (1u << 26)

/* MFC PAK output MMIO register offsets (from the CS MMIO base, used with
 * MI_STORE_REGISTER_MEM AddCSMMIOStartOffset). DG2 (gen125) inherits the gen12
 * MFX class and does not remap these, so the same values apply to both
 * generations. */
#define ANV_H264_MFC_BITSTREAM_BYTECOUNT_FRAME_OFFSET 0x08A0
#define ANV_H264_MFC_IMAGE_STATUS_CTRL_OFFSET         0x08B8
#define ANV_H264_MFC_AVC_NUM_SLICES_OFFSET            0x0954

/* Update DMEM byte offsets of the PAK-output fields (see anv_huc_avc_brc_update_dmem). */
#define ANV_H264_BRC_DMEM_FRAME_BYTE_COUNT_OFFSET 20
#define ANV_H264_BRC_DMEM_IMAGE_STATUS_CTRL_OFFSET 28
#define ANV_H264_BRC_DMEM_NUM_OF_SLICE_OFFSET     384

static void
anv_h264_brc_emit_huc_common(struct anv_cmd_buffer *cmd,
                             uint32_t kernel_descriptor,
                             struct anv_address dmem_addr,
                             uint32_t dmem_size)
{
   struct anv_device *device = cmd->device;

   anv_batch_emit(&cmd->batch, GENX(HUC_IMEM_STATE), imem) {
      imem.HUCFirmwareDescriptor = kernel_descriptor;
   }

   anv_batch_emit(&cmd->batch, GENX(MFX_WAIT), mfx) {
      mfx.MFXSyncControlFlag = 1;
   }
   anv_batch_emit(&cmd->batch, GENX(MFX_WAIT), mfx) {
      mfx.MFXSyncControlFlag = 1;
   }

   anv_batch_emit(&cmd->batch, GENX(HUC_PIPE_MODE_SELECT), sel);

   anv_batch_emit(&cmd->batch, GENX(MFX_WAIT), mfx) {
      mfx.MFXSyncControlFlag = 1;
   }

   anv_batch_emit(&cmd->batch, GENX(HUC_DMEM_STATE), dmem) {
      dmem.HUCDataSourceAddress = dmem_addr;
      dmem.HUCDataSourceAddressAttributes = (struct GENX(MEMORYADDRESSATTRIBUTES)) {
         .MOCS = anv_mocs(device, dmem_addr.bo, 0),
      };
      dmem.HUCDataDestinationAddress = (struct anv_address) {
         NULL, ANV_HUC_DMEM_DEST_OFFSET,
      };
      dmem.HUCDataLength = align(dmem_size, 64) / 64;
   }
}

static void
anv_h264_brc_emit_huc_flush_and_status(struct anv_cmd_buffer *cmd)
{
   struct anv_video_session *vid = cmd->video.vid;
   struct anv_address pak_mmio_addr =
      ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_PAK_MMIO_SEM);

   anv_batch_emit(&cmd->batch, GENX(VD_PIPELINE_FLUSH), flush) {
      flush.HEVCPipelineDone = true;
      flush.HEVCPipelineCommandFlush = true;
      flush.VDCommandMessageParserDone = true;
   }

   anv_video_emit_mi_flush_dw(cmd, true);

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_DATA_IMM), sdi) {
      sdi.Address = anv_address_add(pak_mmio_addr, 4);
      sdi.ImmediateData = ANV_HUC_AVC_BRC_REENCODE_MASK;
   }

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_REGISTER_MEM), srm) {
      srm.RegisterAddress = ANV_HUC_STATUS_MMIO_OFFSET;
      srm.AddCSMMIOStartOffset = 1;
      srm.MemoryAddress = pak_mmio_addr;
   }

   struct anv_address huc_err_addr =
      ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_HUC_ERR_SEM);

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_DATA_IMM), sdi) {
      sdi.Address = anv_address_add(huc_err_addr, 4);
      sdi.ImmediateData = ANV_HUC_AVC_BRC_ERROR_MASK;
   }

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_REGISTER_MEM), srm) {
      srm.RegisterAddress = ANV_HUC_STATUS_MMIO_OFFSET;
      srm.AddCSMMIOStartOffset = 1;
      srm.MemoryAddress = huc_err_addr;
   }

   anv_batch_emit(&cmd->batch, GENX(MI_CONDITIONAL_BATCH_BUFFER_END), cbbe) {
      cbbe.CompareOperation = MADEqualIDD;
      cbbe.CompareSemaphore = 1;
      cbbe.CompareMaskMode = CompareMaskModeEnabled;
      cbbe.CompareDataDword = 0;
      cbbe.CompareAddress = huc_err_addr;
   }
}

static void
anv_h264_brc_emit_huc_tail(struct anv_cmd_buffer *cmd)
{
   struct anv_state status_state =
      anv_cmd_buffer_alloc_temporary_state(cmd, 8, 8);

   if (status_state.map == NULL)
      return;

   struct anv_address status_addr =
      anv_cmd_buffer_temporary_state_address(cmd, status_state);

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_REGISTER_MEM), srm) {
      srm.RegisterAddress = ANV_HUC_STATUS2_MMIO_OFFSET;
      srm.AddCSMMIOStartOffset = 1;
      srm.MemoryAddress = status_addr;
   }

   anv_batch_emit(&cmd->batch, GENX(HUC_START), start) {
      start.LastStreamObject = true;
   }

   anv_h264_brc_emit_huc_flush_and_status(cmd);
}

static void
anv_h264_brc_emit_huc_init(struct anv_cmd_buffer *cmd,
                           const struct VkVideoEncodeH264PictureInfoKHR *frame_info)
{
   const StdVideoH264SequenceParameterSet *sps = vk_video_find_h264_enc_std_sps(
         cmd->video.params,
         frame_info->pStdPictureInfo->seq_parameter_set_id);
   struct anv_device *device = cmd->device;
   struct anv_video_session *vid = cmd->video.vid;
   uint32_t dmem_size = sizeof(struct anv_huc_avc_brc_init_dmem);
   struct anv_state dmem_state =
      anv_cmd_buffer_alloc_temporary_state(cmd, dmem_size, 4096);

   if (dmem_state.map == NULL)
      return;

   memset(dmem_state.map, 0, dmem_size);
   anv_h264_brc_fill_init_dmem(cmd, sps, dmem_state.map);
   vid->rc.brc_init_frame = vid->rc.frame_counter;
   vid->rc.brc_reset = false;

   anv_h264_brc_emit_huc_common(cmd, ANV_HUC_AVC_BRC_INIT_KERNEL_DESCRIPTOR,
                                anv_cmd_buffer_temporary_state_address(cmd, dmem_state),
                                dmem_size);

   anv_batch_emit(&cmd->batch, GENX(HUC_VIRTUAL_ADDR_STATE), va) {
      va.HUCVirtualAddressRegion[0] = (struct GENX(HUC_VIRTUAL_ADDR_REGION)) {
         .Address = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_HISTORY),
         .AddressAttributes = {
            .MOCS = anv_mocs(device, NULL, 0),
         },
      };

      for (uint32_t i = 1; i < 16; i++) {
         va.HUCVirtualAddressRegion[i].AddressAttributes =
            (struct GENX(MEMORYADDRESSATTRIBUTES)) {
            .MOCS = anv_mocs(device, NULL, 0),
         };
      }
   }

   anv_h264_brc_emit_huc_tail(cmd);
}

static void
anv_h264_brc_emit_huc_update(struct anv_cmd_buffer *cmd,
                             const struct VkVideoEncodeH264PictureInfoKHR *frame_info,
                             uint32_t pass,
                             struct anv_state input_slb_state)
{
   const StdVideoH264SequenceParameterSet *sps = vk_video_find_h264_enc_std_sps(
         cmd->video.params,
         frame_info->pStdPictureInfo->seq_parameter_set_id);
   const uint32_t pic_type = frame_info->pStdPictureInfo->primary_pic_type;

   struct anv_device *device = cmd->device;
   struct anv_video_session *vid = cmd->video.vid;
   uint32_t dmem_size = sizeof(struct anv_huc_avc_brc_update_dmem);
   uint32_t cd_size = sizeof(struct anv_huc_avc_brc_const_data);

   struct anv_state dmem_state =
      anv_cmd_buffer_alloc_temporary_state(cmd, dmem_size, 4096);
   struct anv_state cd_state =
      anv_cmd_buffer_alloc_temporary_state(cmd, cd_size, 4096);

   if (dmem_state.map == NULL || cd_state.map == NULL)
      return;

   memset(dmem_state.map, 0, dmem_size);
   anv_h264_brc_fill_update_dmem(cmd, sps, pic_type, pass, dmem_state.map);

   /* Overlay the previous frame's PAK MMIO outputs (frame byte count, image
    * status, slice count) that were staged into the persistent PAK info
    * buffer after the previous PAK, so the HuC BRC Update sees the real prior
    * frame size and can drive the bitrate. */
   struct mi_builder b;
   mi_builder_init(&b, cmd->device->info, &cmd->batch);
   struct anv_address dmem_addr =
      anv_cmd_buffer_temporary_state_address(cmd, dmem_state);
   struct anv_address pak_info =
      ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_PAK_INFO);
   mi_memcpy(&b, anv_address_add(dmem_addr, ANV_H264_BRC_DMEM_FRAME_BYTE_COUNT_OFFSET),
             pak_info, 4);
   mi_memcpy(&b, anv_address_add(dmem_addr, ANV_H264_BRC_DMEM_IMAGE_STATUS_CTRL_OFFSET),
             anv_address_add(pak_info, 4), 4);
   mi_memcpy(&b, anv_address_add(dmem_addr, ANV_H264_BRC_DMEM_NUM_OF_SLICE_OFFSET),
             anv_address_add(pak_info, 8), 4);

   /* The mi_memcpy above patches the update DMEM via the command streamer.
    * Flush so those writes are visible to the HuC DMEM DMA read that follows. */
   anv_video_emit_mi_flush_dw(cmd, true);

   memset(cd_state.map, 0, cd_size);
   anv_h264_brc_fill_const_data(vid, pic_type, cd_state.map);

   anv_h264_brc_emit_huc_common(cmd, ANV_HUC_AVC_BRC_UPDATE_KERNEL_DESCRIPTOR,
                                anv_cmd_buffer_temporary_state_address(cmd, dmem_state),
                                dmem_size);

   anv_batch_emit(&cmd->batch, GENX(HUC_VIRTUAL_ADDR_STATE), va) {
      struct anv_address regions[16] = { 0 };

      regions[0] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_HISTORY);
      regions[1] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_VDENC_STATS);
      regions[2] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_PAK_STATS);
      regions[3] = anv_cmd_buffer_temporary_state_address(cmd, input_slb_state);
      regions[5] = anv_cmd_buffer_temporary_state_address(cmd, cd_state);
      regions[6] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_EXEC_SLB);
      regions[15] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_DEBUG);

      for (uint32_t i = 0; i < 16; i++) {
         va.HUCVirtualAddressRegion[i] = (struct GENX(HUC_VIRTUAL_ADDR_REGION)) {
            .Address = regions[i],
            .AddressAttributes = {
               .MOCS = anv_mocs(device, regions[i].bo, 0),
            },
         };
      }
   }

   anv_h264_brc_emit_huc_tail(cmd);
}
#endif /* GFX_VER >= 12 */

/* HEVC encoding from here */
#define ANV_HUC_HEVC_BRC_INIT_KERNEL_DESCRIPTOR   8
#define ANV_HUC_HEVC_BRC_UPDATE_KERNEL_DESCRIPTOR 9
#define ANV_HUC_HEVC_BRC_MAX_NUM_SLICES           70
#define ANV_HUC_HEVC_BRC_REENCODE_MASK            (1u << 31)
#define ANV_HUC_HEVC_BRC_ERROR_MASK               (1u << 28)
#define ANV_HUC_HEVC_BRC_FLAG_ACQP                0
#define ANV_HUC_HEVC_BRC_FLAG_CBR                 1
#define ANV_HUC_HEVC_BRC_FLAG_VBR                 2

#if GFX_VERx10 >= 125
#define ANV_H265_BRC_SLB_GROUP1_SIZE      64
#define ANV_H265_BRC_SLB_PIC_STATE_OFFSET 196
#define ANV_H265_BRC_SLB_CMD2_OFFSET      360
#define ANV_H265_BRC_SLB_GROUP3_OFFSET    640
#else
#define ANV_H265_BRC_SLB_GROUP1_SIZE      40
#define ANV_H265_BRC_SLB_PIC_STATE_OFFSET 164
#define ANV_H265_BRC_SLB_CMD2_OFFSET      328
#define ANV_H265_BRC_SLB_GROUP3_OFFSET    536
#endif

#define ANV_H265_HCP_BITSTREAM_BYTECOUNT_FRAME_OFFSET          0x28a0
#define ANV_H265_HCP_BITSTREAM_BYTECOUNT_FRAME_NOHEADER_OFFSET 0x28a4
#define ANV_H265_HCP_IMAGE_STATUS_CONTROL_OFFSET               0x28bc

struct anv_huc_hevc_brc_init_dmem {
   uint32_t BRCFunc_U32;
   uint32_t UserMaxFrame;
   uint32_t InitBufFull_U32;
   uint32_t BufSize_U32;
   uint32_t TargetBitrate_U32;
   uint32_t MaxRate_U32;
   uint32_t MinRate_U32;
   uint32_t FrameRateM_U32;
   uint32_t FrameRateD_U32;
   uint32_t LumaLog2WeightDenom_U32;
   uint32_t ChromaLog2WeightDenom_U32;
   uint8_t  BRCFlag;
   uint8_t  Reserved0;
   uint16_t GopP_U16;
   uint16_t GopB_U16;
   uint16_t FrameWidth_U16;
   uint16_t FrameHeight_U16;
   uint16_t GopB1_U16;
   uint16_t GopB2_U16;
   uint8_t  MinQP_U8;
   uint8_t  MaxQP_U8;
   uint8_t  MaxBRCLevel_U8;
   uint8_t  LumaBitDepth_U8;
   uint8_t  ChromaBitDepth_U8;
   uint8_t  CuQpCtrl_U8;
   uint8_t  RSVD0[4];
   int8_t   DevThreshPB0_S8[8];
   int8_t   DevThreshVBR0_S8[8];
   int8_t   DevThreshI0_S8[8];
   int8_t   InstRateThreshP0_S8[4];
   int8_t   InstRateThreshB0_S8[4];
   int8_t   InstRateThreshI0_S8[4];
   uint8_t  LowDelayMode_U8;
   uint8_t  InitQPIP_U8;
   uint8_t  InitQPB_U8;
   uint8_t  QPDeltaThrForAdapt2Pass_U8;
   uint8_t  TopFrmSzThrForAdapt2Pass_U8;
   uint8_t  BotFrmSzThrForAdapt2Pass_U8;
   uint8_t  QPSelectForFirstPass_U8;
   uint8_t  MBHeaderCompensation_U8;
   uint8_t  OverShootCarryFlag_U8;
   uint8_t  OverShootSkipFramePct_U8;
   uint8_t  EstRateThreshP0_U8[7];
   uint8_t  EstRateThreshB0_U8[7];
   uint8_t  EstRateThreshI0_U8[7];
   uint8_t  QPP_U8;
   uint8_t  StreamInSurfaceEnable_U8;
   uint8_t  StreamInROIEnable_U8;
   uint8_t  TimingBudget_Enable_U8;
   uint8_t  TopQPDeltaThrForAdapt2Pass_U8;
   uint8_t  BotQPDeltaThrForAdapt2Pass_U8;
   uint8_t  Reserved1;
   uint8_t  NetworkTraceEnable_U8;
   uint8_t  LowDelaySceneChangeXFrameSizeEnable_U8;
   uint32_t ACQP_U32;
   uint32_t SlidingWindow_Size_U32;
   uint8_t  SLIDINGWINDOW_MaxRateRatio;
   uint8_t  LookaheadDepth_U8;
   int8_t   CbQPOffset;
   int8_t   CrQPOffset;
   uint32_t ProfileLevelMaxFramePB_U32;
   uint16_t SlideWindowRC;
   uint16_t MaxLogCUSize;
   uint16_t FrameWidthInLCU;
   uint16_t FrameHeightInLCU;
   uint8_t  BRCPyramidEnable_U8;
   uint8_t  LongTermRefEnable_U8;
   uint16_t LongTermRefInterval_U16;
   uint8_t  LongTermRefMsdk_U8;
   uint8_t  IsLowDelay_U8;
   uint16_t RSVD2;
   uint32_t RSVD3[4];
};

struct anv_huc_hevc_brc_update_dmem {
   uint32_t TARGETSIZE_U32;
   uint32_t FrameID_U32;
   uint32_t Ref_L0_FrameID_U32[8];
   uint32_t Ref_L1_FrameID_U32[8];
   uint16_t startGAdjFrame_U16[4];
   uint16_t TargetSliceSize_U16;
   uint16_t SLB_Data_SizeInBytes;
   uint16_t PIC_STATE_StartInBytes;
   uint16_t CMD2_StartInBytes;
   uint16_t CMD1_StartInBytes;
   uint16_t PIPE_MODE_SELECT_StartInBytes;
   uint16_t Current_Data_Offset;
   uint16_t Ref_Data_Offset[5];
   uint16_t MaxNumSliceAllowed_U16;
   uint8_t  OpMode_U8;
   uint8_t  CurrentFrameType_U8;
   uint8_t  Num_Ref_L0_U8;
   uint8_t  Num_Ref_L1_U8;
   uint8_t  Num_Slices;
   uint8_t  CQP_QPValue_U8;
   uint8_t  CQP_FracQP_U8;
   uint8_t  MaxNumPass_U8;
   uint8_t  gRateRatioThreshold_U8[7];
   uint8_t  startGAdjMult_U8[5];
   uint8_t  startGAdjDiv_U8[5];
   uint8_t  gRateRatioThresholdQP_U8[8];
   uint8_t  SceneChgPrevIntraPctThreshold_U8;
   uint8_t  SceneChgCurIntraPctThreshold_U8;
   uint8_t  IPAverageCoeff_U8;
   uint8_t  CurrentPass_U8;
   int8_t   DeltaQPForMvZero_S8;
   int8_t   DeltaQPForMvZone0_S8;
   int8_t   DeltaQPForMvZone1_S8;
   int8_t   DeltaQPForMvZone2_S8;
   int8_t   DeltaQPForSadZone0_S8;
   int8_t   DeltaQPForSadZone1_S8;
   int8_t   DeltaQPForSadZone2_S8;
   int8_t   DeltaQPForSadZone3_S8;
   int8_t   DeltaQPForROI0_S8;
   int8_t   DeltaQPForROI1_S8;
   int8_t   DeltaQPForROI2_S8;
   int8_t   DeltaQPForROI3_S8;
   int8_t   LumaLog2WeightDenom_S8;
   int8_t   ChromaLog2WeightDenom_S8;
   uint8_t  DisabledFeature_U8;
   uint8_t  SlidingWindow_Enable_U8;
   uint8_t  LOG_LCU_Size_U8;
   uint16_t NetworkTraceEntry_U16;
   uint16_t LowDelaySceneChangeXFrameSize_U16;
   int8_t   ReEncodePositiveQPDeltaThr_S8;
   int8_t   ReEncodeNegativeQPDeltaThr_S8;
   uint8_t  MaxNumTileHuCCallMinus1;
   uint8_t  TileHucCallIndex;
   uint8_t  TileHuCCallPassIndex;
   uint8_t  TileHuCCallPassMax;
   uint16_t TileSizeInLCU;
   uint32_t TxSizeInBitsPerFrame;
   uint8_t  StartTileIdx;
   uint8_t  EndTileIdx;
   uint16_t NumFrameSkipped;
   uint32_t SkipFrameSize;
   uint32_t SliceHeaderSize;
   uint8_t  IsLongTermRef;
#if GFX_VERx10 >= 125
   uint8_t  FrameSizeBoostForSceneChange;
   uint8_t  ROMCurrent;
   uint8_t  ROMZero;
   uint32_t TargetFrameSize;
   uint32_t TargetFulness;
   uint8_t  Delta;
   uint8_t  CqmEnable;
   uint8_t  UPD_TempCurrentlayer;
   uint8_t  UPD_TempScalable;
   uint32_t UPD_UserMaxFrame;
   uint32_t UPD_UserMaxFramePB;
   uint8_t  UPD_Randomaccess;
   uint8_t  UPD_AdaptiveTUEnabled;
   uint16_t UPD_GopPicSize;
   uint8_t  UPD_LADsRatio;
   uint8_t  RSVD[39];
#else
   uint8_t  EnableMotionAdaptive;
   uint8_t  EnableLookAhead;
   uint8_t  UPD_CQMEnabled_U8;
#endif
};

struct anv_huc_hevc_brc_mode_costs {
   uint8_t I_INTRA_64X64DC;
   uint8_t I_INTRA_32x32;
   uint8_t I_INTRA_16x16;
   uint8_t I_INTRA_8x8;
   uint8_t I_INTRA_SADMPM;
   uint8_t I_INTRA_RDEMPM;
   uint8_t I_INTRA_NxN;
   uint8_t INTRA_64X64DC;
   uint8_t INTRA_32x32;
   uint8_t INTRA_16x16;
   uint8_t INTRA_8x8;
   uint8_t INTRA_SADMPM;
   uint8_t INTRA_RDEMPM;
   uint8_t INTRA_NxN;
   uint8_t INTER_32x32;
   uint8_t INTER_32x16;
   uint8_t INTER_16x16;
   uint8_t INTER_16x8;
   uint8_t INTER_8x8;
   uint8_t REF_ID;
   uint8_t MERGE_64X64;
   uint8_t MERGE_32X32;
   uint8_t MERGE_16x16;
   uint8_t MERGE_8x8;
   uint8_t SKIP_64X64;
   uint8_t SKIP_32X32;
   uint8_t SKIP_16x16;
   uint8_t SKIP_8x8;
};

struct anv_huc_hevc_brc_slice_offsets {
   uint16_t SizeOfCMDs;
   uint16_t HcpWeightOffsetL0_StartInBytes;
   uint16_t HcpWeightOffsetL1_StartInBytes;
   uint16_t SliceState_StartInBytes;
   uint16_t SliceHeaderPIO_StartInBytes;
   uint16_t VdencWeightOffset_StartInBytes;
   uint16_t SliceHeader_SizeInBits;
   uint16_t WeightTable_StartInBits;
   uint16_t WeightTable_EndInBits;
};

struct anv_huc_hevc_brc_const_data {
   uint16_t SADQPLambdaI[52];
   uint16_t SADQPLambdaP[52];
   uint16_t RDQPLambdaI[52];
   uint16_t RDQPLambdaP[52];
   uint16_t SLCSZ_THRDELTAI_U16[52];
   uint16_t SLCSZ_THRDELTAP_U16[52];
   uint8_t  DistThreshldI[9];
   uint8_t  DistThreshldP[9];
   uint8_t  DistThreshldB[9];
   int8_t   DistQPAdjTabI[81];
   int8_t   DistQPAdjTabP[81];
   int8_t   DistQPAdjTabB[81];
   int8_t   FrmSzAdjTabI_S8[72];
   int8_t   FrmSzAdjTabP_S8[72];
   int8_t   FrmSzAdjTabB_S8[72];
   uint8_t  FrmSzMaxTabI[9];
   uint8_t  FrmSzMaxTabP[9];
   uint8_t  FrmSzMaxTabB[9];
   uint8_t  FrmSzMinTabI[9];
   uint8_t  FrmSzMinTabP[9];
   uint8_t  FrmSzMinTabB[9];
   int8_t   QPAdjTabI[45];
   int8_t   QPAdjTabP[45];
   int8_t   QPAdjTabB[45];
   struct anv_huc_hevc_brc_mode_costs ModeCosts[52];
   struct anv_huc_hevc_brc_slice_offsets Slice[ANV_HUC_HEVC_BRC_MAX_NUM_SLICES];
#if GFX_VERx10 >= 125
   uint8_t  RSVD[61];
#else
   uint8_t  PenaltyForIntraNonDC32x32PredMode[52];
   uint32_t UPD_TR_TargetSize_U32;
   uint32_t UPD_LA_TargetFulness_U32;
   uint8_t  UPD_deltaQP;
   uint8_t  UPD_TCBRC_SCENARIO_U8;
   uint8_t  UPD_ROM_CURRENT_U8;
   uint8_t  UPD_ROM_ZERO_U8;
#endif
};

static_assert(sizeof(struct anv_huc_hevc_brc_init_dmem) == 192, "huc brc init dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_init_dmem, BRCFlag) == 44, "huc brc init dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_init_dmem, DevThreshPB0_S8) == 68, "huc brc init dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_init_dmem, InitQPIP_U8) == 105, "huc brc init dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_init_dmem, ACQP_U32) == 144, "huc brc init dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_update_dmem, startGAdjFrame_U16) == 72, "huc brc update dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_update_dmem, OpMode_U8) == 106, "huc brc update dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_update_dmem, CurrentPass_U8) == 142, "huc brc update dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_update_dmem, IsLongTermRef) == 188, "huc brc update dmem layout");
#if GFX_VERx10 >= 125
static_assert(sizeof(struct anv_huc_hevc_brc_update_dmem) == 256, "huc brc update dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_update_dmem, TargetFrameSize) == 192, "huc brc update dmem layout");
static_assert(offsetof(struct anv_huc_hevc_brc_update_dmem, UPD_GopPicSize) == 214, "huc brc update dmem layout");
#else
static_assert(sizeof(struct anv_huc_hevc_brc_update_dmem) == 192, "huc brc update dmem layout");
#endif
static_assert(sizeof(struct anv_huc_hevc_brc_mode_costs) == 28, "huc brc const data layout");
static_assert(sizeof(struct anv_huc_hevc_brc_slice_offsets) == 18, "huc brc const data layout");
static_assert(offsetof(struct anv_huc_hevc_brc_const_data, ModeCosts) == 1299, "huc brc const data layout");
static_assert(offsetof(struct anv_huc_hevc_brc_const_data, Slice) == 2756, "huc brc const data layout");
#if GFX_VERx10 >= 125
static_assert(sizeof(struct anv_huc_hevc_brc_const_data) == 4078, "huc brc const data layout");
#else
static_assert(sizeof(struct anv_huc_hevc_brc_const_data) == 4080, "huc brc const data layout");
#endif

typedef struct anv_huc_hevc_brc_update_dmem anv_h265_brc_update_dmem;

struct anv_h265_brc_slb_layout {
   uint32_t num_slices;
   uint32_t data_size;
   uint32_t slice_start[ANV_HUC_HEVC_BRC_MAX_NUM_SLICES];
   uint32_t slice_data_start[ANV_HUC_HEVC_BRC_MAX_NUM_SLICES];
   uint32_t slice_size[ANV_HUC_HEVC_BRC_MAX_NUM_SLICES];
   uint32_t header_bits[ANV_HUC_HEVC_BRC_MAX_NUM_SLICES];
   uint8_t  header_last_byte[ANV_HUC_HEVC_BRC_MAX_NUM_SLICES];
};

static void
anv_video_emit_mfx_wait(struct anv_cmd_buffer *cmd, struct anv_batch *batch);
static void
anv_h265_emit_hcp_pipe_mode_select(struct anv_cmd_buffer *cmd,
                                   struct anv_batch *batch,
                                   const VkVideoEncodeInfoKHR *enc_info,
                                   bool brc_enabled);
static void
anv_h265_emit_vdenc_cmd1(struct anv_cmd_buffer *cmd,
                         struct anv_batch *batch,
                         const VkVideoEncodeInfoKHR *enc_info);
static void
anv_h265_emit_hcp_pic_state(struct anv_cmd_buffer *cmd,
                            struct anv_batch *batch,
                            const VkVideoEncodeInfoKHR *enc_info);
static void
anv_h265_emit_vdenc_cmd2(struct anv_cmd_buffer *cmd,
                         struct anv_batch *batch,
                         const VkVideoEncodeInfoKHR *enc_info,
                         const uint8_t *dpb_idx,
                         bool is_low_delay);
static void
anv_h265_emit_hcp_weightoffset_state(struct anv_cmd_buffer *cmd,
                                     struct anv_batch *batch,
                                     const VkVideoEncodeInfoKHR *enc_info,
                                     uint32_t slice_id);
static void
anv_h265_emit_hcp_slice_state(struct anv_cmd_buffer *cmd,
                              struct anv_batch *batch,
                              const VkVideoEncodeInfoKHR *enc_info,
                              uint32_t slice_id,
                              const uint8_t *dpb_idx,
                              bool pak_only,
                              bool is_low_delay);
static void
anv_h265_emit_slice_header(struct anv_cmd_buffer *cmd,
                           struct anv_batch *batch,
                           const VkVideoEncodeInfoKHR *enc_info,
                           uint32_t slice_id);
static void
anv_vdenc_emit_weightsoffsets_state(struct anv_cmd_buffer *cmd,
                                    struct anv_batch *batch,
                                    bool chroma);

static void
anv_h265_brc_compute_slb_layout(const VkVideoEncodeH265PictureInfoKHR *frame_info,
                                const StdVideoH265VideoParameterSet *vps,
                                const StdVideoH265SequenceParameterSet *sps,
                                const StdVideoH265PictureParameterSet *pps,
                                struct anv_h265_brc_slb_layout *layout)
{
   uint32_t offset = ANV_H265_BRC_SLB_GROUP3_OFFSET;

   layout->num_slices = frame_info->naluSliceSegmentEntryCount;
   assert(layout->num_slices <= ANV_HUC_HEVC_BRC_MAX_NUM_SLICES);

   for (uint32_t slice_id = 0; slice_id < layout->num_slices; slice_id++) {
      const VkVideoEncodeH265NaluSliceSegmentInfoKHR *nalu =
         &frame_info->pNaluSliceSegmentEntries[slice_id];
      StdVideoEncodeH265SliceSegmentHeader *slice_header =
         (StdVideoEncodeH265SliceSegmentHeader *)nalu->pStdSliceSegmentHeader;
      uint8_t slice_header_data[256] = { 0, };
      size_t slice_header_data_len_in_bytes = 0;

      if (slice_header->slice_type % 5 == STD_VIDEO_H265_SLICE_TYPE_P)
         slice_header->slice_type = STD_VIDEO_H265_SLICE_TYPE_B;

      vk_video_encode_h265_slice_header(frame_info->pStdPictureInfo,
                                        vps, sps, pps, slice_header, 0,
                                        &slice_header_data_len_in_bytes,
                                        &slice_header_data);

      uint32_t header_bits = slice_header_data_len_in_bytes * 8;
      uint32_t payload = (align(header_bits, 32) >> 5) * 4;

      layout->header_bits[slice_id] = header_bits;
      layout->header_last_byte[slice_id] =
         slice_header_data[slice_header_data_len_in_bytes - 1];
      layout->slice_start[slice_id] = offset;

#if GFX_VERx10 >= 125
      layout->slice_data_start[slice_id] = offset + GENX(HCP_SLICE_STATE_length) * 4 +
                                           GENX(MI_BATCH_BUFFER_END_length) * 4;
      uint32_t end = layout->slice_data_start[slice_id] + 8 + payload +
                     GENX(VDENC_WEIGHTSOFFSETS_STATE_length) * 4 +
                     GENX(MI_BATCH_BUFFER_END_length) * 4;
      layout->slice_size[slice_id] = align(end, 64) - offset;
#else
      layout->slice_data_start[slice_id] = offset + GENX(HCP_SLICE_STATE_length) * 4;
      layout->slice_size[slice_id] = GENX(HCP_SLICE_STATE_length) * 4 + 8 + payload +
                                     GENX(VDENC_WEIGHTSOFFSETS_STATE_length) * 4 +
                                     GENX(MI_BATCH_BUFFER_END_length) * 4 + 32;
#endif

      offset += layout->slice_size[slice_id];
   }

   layout->data_size = offset;
}

static void
anv_h265_brc_fill_init_dmem(struct anv_cmd_buffer *cmd,
                            const StdVideoH265SequenceParameterSet *sps,
                            bool is_low_delay,
                            struct anv_huc_hevc_brc_init_dmem *dmem)
{
   struct anv_video_session *vid = cmd->video.vid;
   struct anv_video_rc_state *rc = &vid->rc;
   bool is_cbr = vid->rc_mode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR;
   uint32_t width = sps->pic_width_in_luma_samples;
   uint32_t height = sps->pic_height_in_luma_samples;
   uint32_t frame_rate_m = rc->frame_rate_num ? rc->frame_rate_num : 30;
   uint32_t frame_rate_d = rc->frame_rate_den ? rc->frame_rate_den : 1;
   uint32_t target_bps = DIV_ROUND_UP(rc->average_bitrate, 1000) * 1000;
   uint32_t max_bps = is_cbr ? target_bps :
                      DIV_ROUND_UP(rc->max_bitrate, 1000) * 1000;
   uint32_t gop = rc->gop_frame_count ? rc->gop_frame_count : 30;
   uint32_t gop_ref_dist = MAX2(rc->consecutive_b_frames + 1, 1);

   dmem->BRCFunc_U32 = 0;
   /* TODO: derive the profile and level based maximum frame size */
   dmem->UserMaxFrame = width * height;
   dmem->ProfileLevelMaxFramePB_U32 = dmem->UserMaxFrame;
   dmem->InitBufFull_U32 = MIN2(rc->vbv_initial_fullness_bits, rc->vbv_size_bits);
   dmem->BufSize_U32 = rc->vbv_size_bits;
   dmem->TargetBitrate_U32 = target_bps;
   dmem->MaxRate_U32 = max_bps;
   dmem->MinRate_U32 = 0;
   dmem->FrameRateM_U32 = frame_rate_m;
   dmem->FrameRateD_U32 = frame_rate_d;
   dmem->BRCFlag = is_cbr ? ANV_HUC_HEVC_BRC_FLAG_CBR : ANV_HUC_HEVC_BRC_FLAG_VBR;
   dmem->CuQpCtrl_U8 = 0;

   uint32_t intra_period = MIN2(gop, 4001) - 1;
   intra_period = DIV_ROUND_UP(intra_period, gop_ref_dist) * gop_ref_dist;
   dmem->GopP_U16 = intra_period / gop_ref_dist;
   dmem->GopB_U16 = intra_period - dmem->GopP_U16;
   dmem->GopB1_U16 = 0;
   dmem->GopB2_U16 = 0;
   dmem->MaxBRCLevel_U8 = dmem->GopB_U16 == 0 ? 0 : 1;
   dmem->BRCPyramidEnable_U8 = 0;

   dmem->FrameWidth_U16 = width;
   dmem->FrameHeight_U16 = height;
   dmem->MinQP_U8 = rc->min_qp < 10 ? 10 : rc->min_qp;
   dmem->MaxQP_U8 = rc->max_qp < 10 ? 51 : MIN2(rc->max_qp, 51);
   dmem->LumaBitDepth_U8 = sps->bit_depth_luma_minus8 + 8;
   dmem->ChromaBitDepth_U8 = sps->bit_depth_chroma_minus8 + 8;
   dmem->LowDelayMode_U8 = 0;
   dmem->IsLowDelay_U8 = is_low_delay;

   double fps = (double)frame_rate_m / frame_rate_d;
   double input_bits_per_frame = (double)max_bps / fps;
   double bps_ratio = input_bits_per_frame / ((double)dmem->BufSize_U32 / 30.0);
   bps_ratio = CLAMP(bps_ratio, 0.1, 3.5);

   for (uint32_t i = 0; i < 4; i++) {
      dmem->DevThreshPB0_S8[i] =
         (int8_t)(-50.0 * pow(anv_hevc_brc_dev_thresh_pb_neg[i], bps_ratio));
      dmem->DevThreshPB0_S8[i + 4] =
         (int8_t)(50.0 * pow(anv_hevc_brc_dev_thresh_pb_pos[i], bps_ratio));
      dmem->DevThreshI0_S8[i] =
         (int8_t)(-50.0 * pow(anv_hevc_brc_dev_thresh_i_neg[i], bps_ratio));
      dmem->DevThreshI0_S8[i + 4] =
         (int8_t)(50.0 * pow(anv_hevc_brc_dev_thresh_i_pos[i], bps_ratio));
      dmem->DevThreshVBR0_S8[i] =
         (int8_t)(-50.0 * pow(anv_hevc_brc_dev_thresh_vbr_neg[i], bps_ratio));
      dmem->DevThreshVBR0_S8[i + 4] =
         (int8_t)(100.0 * pow(anv_hevc_brc_dev_thresh_vbr_pos[i], bps_ratio));
   }

   memcpy(dmem->InstRateThreshP0_S8, anv_hevc_brc_inst_rate_thresh_p,
          sizeof(dmem->InstRateThreshP0_S8));
   memcpy(dmem->InstRateThreshB0_S8, anv_hevc_brc_inst_rate_thresh_b,
          sizeof(dmem->InstRateThreshB0_S8));
   memcpy(dmem->InstRateThreshI0_S8, anv_hevc_brc_inst_rate_thresh_i,
          sizeof(dmem->InstRateThreshI0_S8));
   memcpy(dmem->EstRateThreshP0_U8, anv_hevc_brc_est_rate_thresh,
          sizeof(dmem->EstRateThreshP0_U8));
   memcpy(dmem->EstRateThreshB0_U8, anv_hevc_brc_est_rate_thresh,
          sizeof(dmem->EstRateThreshB0_U8));
   memcpy(dmem->EstRateThreshI0_U8, anv_hevc_brc_est_rate_thresh,
          sizeof(dmem->EstRateThreshI0_U8));

   int32_t qp_ip = anv_brc_estimate_init_qp(width * height * 3 / 2, fps,
                                            target_bps);

   if (gop == 1)
      qp_ip += 12;
   else if (gop < 15)
      qp_ip += (14 - gop) >> 1;
   qp_ip = CLAMP(qp_ip, dmem->MinQP_U8, dmem->MaxQP_U8);
   qp_ip--;
   if (qp_ip < 0)
      qp_ip = 1;

   int32_t qp_b = ((qp_ip * 2) * 563 >> 10) + 1;
   qp_b = CLAMP(qp_b, dmem->MinQP_U8, dmem->MaxQP_U8);

   if (gop > 300) {
      qp_ip -= 8;
      qp_b -= 8;
   } else {
      qp_ip -= 2;
      qp_b -= 2;
   }
   dmem->InitQPIP_U8 = CLAMP(qp_ip, dmem->MinQP_U8, dmem->MaxQP_U8);
   dmem->InitQPB_U8 = CLAMP(qp_b, dmem->MinQP_U8, dmem->MaxQP_U8);

   dmem->TopFrmSzThrForAdapt2Pass_U8 = 32;
   dmem->BotFrmSzThrForAdapt2Pass_U8 = 24;
   dmem->TopQPDeltaThrForAdapt2Pass_U8 = 2;
   dmem->BotQPDeltaThrForAdapt2Pass_U8 = 1;

   dmem->SlidingWindow_Size_U32 = MIN2(frame_rate_m / frame_rate_d, 60);
   dmem->SLIDINGWINDOW_MaxRateRatio = 120;

   dmem->LongTermRefEnable_U8 = 1;
   dmem->LongTermRefMsdk_U8 = 1;
}

static void
anv_h265_brc_fill_update_dmem(struct anv_cmd_buffer *cmd,
                              const VkVideoEncodeH265PictureInfoKHR *frame_info,
                              const StdVideoH265SequenceParameterSet *sps,
                              const StdVideoEncodeH265ReferenceListsInfo *ref_lists,
                              const struct anv_h265_brc_slb_layout *layout,
                              uint32_t frame_qp,
                              uint32_t pass,
                              uint32_t pic_type,
                              bool is_low_delay,
                              anv_h265_brc_update_dmem *dmem)
{
   struct anv_video_session *vid = cmd->video.vid;
   struct anv_video_rc_state *rc = &vid->rc;

   dmem->TARGETSIZE_U32 = MIN2(rc->vbv_initial_fullness_bits, rc->vbv_size_bits);
   dmem->FrameID_U32 = rc->frame_counter;

   memcpy(dmem->startGAdjFrame_U16, anv_hevc_brc_start_gadj_frame,
          sizeof(dmem->startGAdjFrame_U16));
   memcpy(dmem->gRateRatioThreshold_U8, anv_hevc_brc_rate_ratio_threshold,
          sizeof(dmem->gRateRatioThreshold_U8));
   memcpy(dmem->startGAdjMult_U8, anv_hevc_brc_start_gadj_mult,
          sizeof(dmem->startGAdjMult_U8));
   memcpy(dmem->startGAdjDiv_U8, anv_hevc_brc_start_gadj_div,
          sizeof(dmem->startGAdjDiv_U8));
   memcpy(dmem->gRateRatioThresholdQP_U8, anv_hevc_brc_rate_ratio_threshold_qp,
          sizeof(dmem->gRateRatioThresholdQP_U8));

   dmem->SLB_Data_SizeInBytes = layout->data_size;
   dmem->PIC_STATE_StartInBytes = ANV_H265_BRC_SLB_PIC_STATE_OFFSET;
   dmem->CMD2_StartInBytes = ANV_H265_BRC_SLB_CMD2_OFFSET;
   dmem->CMD1_StartInBytes = ANV_H265_BRC_SLB_GROUP1_SIZE;
   dmem->PIPE_MODE_SELECT_StartInBytes = 0xFFFF;

   /* TODO: derive from the level */
   dmem->MaxNumSliceAllowed_U16 = ANV_HUC_HEVC_BRC_MAX_NUM_SLICES;
   dmem->OpMode_U8 = 1;
   dmem->CurrentFrameType_U8 = pic_type == 0 ? 2 : (is_low_delay ? 0 : 1);
   if (ref_lists && pic_type != 0) {
      dmem->Num_Ref_L0_U8 = ref_lists->num_ref_idx_l0_active_minus1 + 1;
      dmem->Num_Ref_L1_U8 = ref_lists->num_ref_idx_l1_active_minus1 + 1;
   }
   dmem->Num_Slices = layout->num_slices;
   dmem->CQP_QPValue_U8 = frame_qp;
   dmem->MaxNumPass_U8 = 2;
   dmem->SceneChgPrevIntraPctThreshold_U8 = 96;
   dmem->SceneChgCurIntraPctThreshold_U8 = 192;
   dmem->IPAverageCoeff_U8 = 64;
   dmem->CurrentPass_U8 = pass;
   dmem->LOG_LCU_Size_U8 = 6;
   dmem->ReEncodePositiveQPDeltaThr_S8 = 4;
   dmem->ReEncodeNegativeQPDeltaThr_S8 = -5;

#if GFX_VERx10 >= 125
   /* TODO: verify the softlet only fields against a DMEM dump */
   dmem->UPD_UserMaxFrame = sps->pic_width_in_luma_samples *
                            sps->pic_height_in_luma_samples;
   dmem->UPD_UserMaxFramePB = dmem->UPD_UserMaxFrame;
   dmem->UPD_Randomaccess = !is_low_delay;
   dmem->UPD_GopPicSize = rc->gop_frame_count ? rc->gop_frame_count : 30;
#endif
}

static void
anv_h265_brc_fill_const_data(const struct anv_h265_brc_slb_layout *layout,
                             uint32_t pic_type,
                             uint32_t pass,
                             struct anv_huc_hevc_brc_const_data *cd)
{
   memcpy(cd->SLCSZ_THRDELTAI_U16, anv_hevc_brc_const_tables,
          sizeof(anv_hevc_brc_const_tables));
   memcpy(cd->RDQPLambdaI, anv_hevc_brc_rd_qp_lambda_i, sizeof(cd->RDQPLambdaI));
   memcpy(cd->RDQPLambdaP, anv_hevc_brc_rd_qp_lambda_p, sizeof(cd->RDQPLambdaP));
   memcpy(cd->SADQPLambdaI, anv_hevc_brc_sad_qp_lambda_i, sizeof(cd->SADQPLambdaI));
#if GFX_VERx10 < 125
   memcpy(cd->PenaltyForIntraNonDC32x32PredMode, anv_hevc_brc_penalty_intra_non_dc32,
          sizeof(cd->PenaltyForIntraNonDC32x32PredMode));
#endif
   memcpy(cd->SADQPLambdaP, anv_hevc_brc_sad_qp_lambda_p, sizeof(cd->SADQPLambdaP));

   if (pic_type == 0)
      memcpy(cd->ModeCosts, anv_hevc_brc_mode_costs_i, sizeof(cd->ModeCosts));
   else
      memcpy(cd->ModeCosts, anv_hevc_brc_mode_costs_pb, sizeof(cd->ModeCosts));

   for (uint32_t slice_id = 0; slice_id < layout->num_slices; slice_id++) {
      struct anv_huc_hevc_brc_slice_offsets *slice = &cd->Slice[slice_id];

      slice->SizeOfCMDs = layout->slice_size[slice_id];
      slice->HcpWeightOffsetL0_StartInBytes = 0xFFFF;
      slice->HcpWeightOffsetL1_StartInBytes = 0xFFFF;
      slice->SliceState_StartInBytes = layout->slice_start[slice_id];
      slice->SliceHeaderPIO_StartInBytes = layout->slice_data_start[slice_id];
#if GFX_VERx10 >= 125
      slice->VdencWeightOffset_StartInBytes =
         layout->slice_data_start[slice_id] + 8 +
         (align(layout->header_bits[slice_id], 32) >> 5) * 4;
#else
      slice->VdencWeightOffset_StartInBytes =
         layout->slice_start[slice_id] + layout->slice_size[slice_id] -
         GENX(VDENC_WEIGHTSOFFSETS_STATE_length) * 4 -
         GENX(MI_BATCH_BUFFER_END_length) * 4 - 32;
#endif

      slice->SliceHeader_SizeInBits = layout->header_bits[slice_id];
      if (pass > 0) {
         uint8_t last_byte = layout->header_last_byte[slice_id];
         for (uint32_t i = 0; i < 8; i++) {
            if (last_byte & (1 << i)) {
               slice->SliceHeader_SizeInBits -= i + 1;
               break;
            }
         }
      }

      slice->WeightTable_StartInBits = 0xFFFF;
      slice->WeightTable_EndInBits = 0xFFFF;
   }
}

static void
anv_h265_brc_slb_pad_to(struct anv_batch *slb, uint32_t target)
{
   assert((uint32_t)(slb->next - slb->start) <= target);
   while ((uint32_t)(slb->next - slb->start) < target)
      anv_batch_emit(slb, GENX(MI_NOOP), noop);
}

static void
anv_h265_brc_build_input_slb(struct anv_cmd_buffer *cmd,
                             const VkVideoEncodeInfoKHR *enc_info,
                             const uint8_t *dpb_idx,
                             const struct anv_h265_brc_slb_layout *layout,
                             bool is_low_delay,
                             bool pak_only,
                             struct anv_state slb_state)
{
   struct anv_batch slb = { 0 };

   anv_batch_set_storage(&slb,
                         anv_cmd_buffer_temporary_state_address(cmd, slb_state),
                         slb_state.map, slb_state.alloc_size);
   memset(slb_state.map, 0, slb_state.alloc_size);

   anv_video_emit_mfx_wait(cmd, &slb);
   anv_h265_emit_hcp_pipe_mode_select(cmd, &slb, enc_info, true);
   anv_video_emit_mfx_wait(cmd, &slb);
   anv_batch_emit(&slb, GENX(MI_BATCH_BUFFER_END), bbe);
   anv_h265_brc_slb_pad_to(&slb, ANV_H265_BRC_SLB_GROUP1_SIZE);

   anv_h265_emit_vdenc_cmd1(cmd, &slb, enc_info);
   assert((uint32_t)(slb.next - slb.start) == ANV_H265_BRC_SLB_PIC_STATE_OFFSET);

   anv_h265_emit_hcp_pic_state(cmd, &slb, enc_info);
   assert((uint32_t)(slb.next - slb.start) == ANV_H265_BRC_SLB_CMD2_OFFSET);

   anv_h265_emit_vdenc_cmd2(cmd, &slb, enc_info, dpb_idx, is_low_delay);
   anv_batch_emit(&slb, GENX(MI_BATCH_BUFFER_END), bbe);
   anv_h265_brc_slb_pad_to(&slb, ANV_H265_BRC_SLB_GROUP3_OFFSET);

   for (uint32_t slice_id = 0; slice_id < layout->num_slices; slice_id++) {
      assert((uint32_t)(slb.next - slb.start) == layout->slice_start[slice_id]);

      anv_h265_emit_hcp_weightoffset_state(cmd, &slb, enc_info, slice_id);
      anv_h265_emit_hcp_slice_state(cmd, &slb, enc_info, slice_id, dpb_idx, pak_only, is_low_delay);
#if GFX_VERx10 >= 125
      anv_batch_emit(&slb, GENX(MI_BATCH_BUFFER_END), bbe);
#endif
      assert((uint32_t)(slb.next - slb.start) == layout->slice_data_start[slice_id]);

      anv_h265_emit_slice_header(cmd, &slb, enc_info, slice_id);
      anv_vdenc_emit_weightsoffsets_state(cmd, &slb, false);
      anv_batch_emit(&slb, GENX(MI_BATCH_BUFFER_END), bbe);
#if GFX_VERx10 >= 125
      anv_h265_brc_slb_pad_to(&slb, layout->slice_start[slice_id] +
                                    layout->slice_size[slice_id]);
#else
      for (uint32_t i = 0; i < 8; i++)
         anv_batch_emit(&slb, GENX(MI_NOOP), noop);
#endif
      assert((uint32_t)(slb.next - slb.start) ==
             layout->slice_start[slice_id] + layout->slice_size[slice_id]);
   }

   assert((uint32_t)(slb.next - slb.start) == layout->data_size);
}

static void
anv_h265_brc_emit_huc_flush_and_status(struct anv_cmd_buffer *cmd)
{
   struct anv_video_session *vid = cmd->video.vid;
   struct anv_address pak_mmio_addr =
      ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_PAK_MMIO_SEM);

   anv_batch_emit(&cmd->batch, GENX(VD_PIPELINE_FLUSH), flush) {
      flush.HEVCPipelineDone = true;
      flush.HEVCPipelineCommandFlush = true;
      flush.VDCommandMessageParserDone = true;
   }

   anv_video_emit_mi_flush_dw(cmd, true);

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_DATA_IMM), sdi) {
      sdi.Address = anv_address_add(pak_mmio_addr, 4);
      sdi.ImmediateData = ANV_HUC_HEVC_BRC_REENCODE_MASK;
   }

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_REGISTER_MEM), srm) {
      srm.RegisterAddress = ANV_HUC_STATUS_MMIO_OFFSET;
      srm.AddCSMMIOStartOffset = 1;
      srm.MemoryAddress = pak_mmio_addr;
   }

#if GFX_VERx10 < 125
   struct anv_address huc_err_addr =
      ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_HUC_ERR_SEM);

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_DATA_IMM), sdi) {
      sdi.Address = anv_address_add(huc_err_addr, 4);
      sdi.ImmediateData = ANV_HUC_HEVC_BRC_ERROR_MASK;
   }

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_REGISTER_MEM), srm) {
      srm.RegisterAddress = ANV_HUC_STATUS_MMIO_OFFSET;
      srm.AddCSMMIOStartOffset = 1;
      srm.MemoryAddress = huc_err_addr;
   }

   anv_batch_emit(&cmd->batch, GENX(MI_CONDITIONAL_BATCH_BUFFER_END), cbbe) {
      cbbe.CompareOperation = MADEqualIDD;
      cbbe.CompareSemaphore = 1;
      cbbe.CompareMaskMode = CompareMaskModeEnabled;
      cbbe.CompareDataDword = 0;
      cbbe.CompareAddress = huc_err_addr;
   }
#endif
}

static void
anv_h265_brc_emit_huc_common(struct anv_cmd_buffer *cmd,
                             uint32_t kernel_descriptor,
                             struct anv_address dmem_addr,
                             uint32_t dmem_size)
{
   struct anv_device *device = cmd->device;

   anv_batch_emit(&cmd->batch, GENX(HUC_IMEM_STATE), imem) {
      imem.HUCFirmwareDescriptor = kernel_descriptor;
   }

   anv_video_emit_mfx_wait(cmd, NULL);
   anv_batch_emit(&cmd->batch, GENX(HUC_PIPE_MODE_SELECT), sel);
   anv_video_emit_mfx_wait(cmd, NULL);

   anv_batch_emit(&cmd->batch, GENX(HUC_DMEM_STATE), dmem) {
      dmem.HUCDataSourceAddress = dmem_addr;
      dmem.HUCDataSourceAddressAttributes = (struct GENX(MEMORYADDRESSATTRIBUTES)) {
         .MOCS = anv_mocs(device, dmem_addr.bo, 0),
      };
      dmem.HUCDataDestinationAddress = (struct anv_address) {
         NULL, ANV_HUC_DMEM_DEST_OFFSET,
      };
      dmem.HUCDataLength = align(dmem_size, 64) / 64;
   }
}

static void
anv_h265_brc_emit_huc_tail(struct anv_cmd_buffer *cmd)
{
   struct anv_state status_state =
      anv_cmd_buffer_alloc_temporary_state(cmd, 8, 8);

   if (status_state.map == NULL)
      return;

   struct anv_address status_addr =
      anv_cmd_buffer_temporary_state_address(cmd, status_state);

   anv_batch_emit(&cmd->batch, GENX(MI_STORE_REGISTER_MEM), srm) {
      srm.RegisterAddress = ANV_HUC_STATUS2_MMIO_OFFSET;
      srm.AddCSMMIOStartOffset = 1;
      srm.MemoryAddress = status_addr;
   }

   anv_batch_emit(&cmd->batch, GENX(HUC_START), start) {
      start.LastStreamObject = true;
   }

   anv_h265_brc_emit_huc_flush_and_status(cmd);
}

static void
anv_h265_brc_emit_huc_init(struct anv_cmd_buffer *cmd,
                           const StdVideoH265SequenceParameterSet *sps,
                           bool is_low_delay)
{
   struct anv_device *device = cmd->device;
   struct anv_video_session *vid = cmd->video.vid;
   uint32_t dmem_size = sizeof(struct anv_huc_hevc_brc_init_dmem);
   struct anv_state dmem_state =
      anv_cmd_buffer_alloc_temporary_state(cmd, dmem_size, 4096);

   if (dmem_state.map == NULL)
      return;

   memset(dmem_state.map, 0, dmem_size);
   anv_h265_brc_fill_init_dmem(cmd, sps, is_low_delay, dmem_state.map);
   vid->rc.brc_reset = false;

   anv_h265_brc_emit_huc_common(cmd, ANV_HUC_HEVC_BRC_INIT_KERNEL_DESCRIPTOR,
                                anv_cmd_buffer_temporary_state_address(cmd, dmem_state),
                                dmem_size);

   anv_batch_emit(&cmd->batch, GENX(HUC_VIRTUAL_ADDR_STATE), va) {
      va.HUCVirtualAddressRegion[0] = (struct GENX(HUC_VIRTUAL_ADDR_REGION)) {
         .Address = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_HISTORY),
         .AddressAttributes = {
            .MOCS = anv_mocs(device, NULL, 0),
         },
      };

      for (uint32_t i = 1; i < 16; i++) {
         va.HUCVirtualAddressRegion[i].AddressAttributes = (struct GENX(MEMORYADDRESSATTRIBUTES)) {
            .MOCS = anv_mocs(device, NULL, 0),
         };
      }
   }

   anv_h265_brc_emit_huc_tail(cmd);
}

static void
anv_h265_brc_emit_huc_update(struct anv_cmd_buffer *cmd,
                             struct anv_state dmem_state,
                             uint32_t dmem_size,
                             struct anv_state input_slb_state,
                             struct anv_state const_data_state)
{
   struct anv_device *device = cmd->device;
   struct anv_video_session *vid = cmd->video.vid;

   anv_h265_brc_emit_huc_common(cmd, ANV_HUC_HEVC_BRC_UPDATE_KERNEL_DESCRIPTOR,
                                anv_cmd_buffer_temporary_state_address(cmd, dmem_state),
                                dmem_size);

   anv_batch_emit(&cmd->batch, GENX(HUC_VIRTUAL_ADDR_STATE), va) {
      struct anv_address regions[16] = { 0 };

      regions[0] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_HISTORY);
      regions[1] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_H265_VDENC_STATS_STREAMOUT);
      regions[2] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_H265_FRAME_STATS_STREAMOUT);
      regions[3] = anv_cmd_buffer_temporary_state_address(cmd, input_slb_state);
      regions[4] = anv_cmd_buffer_temporary_state_address(cmd, const_data_state);
      regions[5] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_EXEC_SLB);
      regions[6] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_WP_DATA);
      regions[7] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_H265_LCU_BASE_ADDR);
      regions[8] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_PAK_INFO);
      regions[15] = ANV_VID_MEM_ADDR(vid, ANV_VID_MEM_BRC_DEBUG);

      for (uint32_t i = 0; i < 16; i++) {
         va.HUCVirtualAddressRegion[i] = (struct GENX(HUC_VIRTUAL_ADDR_REGION)) {
            .Address = regions[i],
            .AddressAttributes = {
               .MOCS = anv_mocs(device, regions[i].bo, 0),
            },
         };
      }
   }

   anv_h265_brc_emit_huc_tail(cmd);
}

#endif /* GENX_HUC_BRC_H */
