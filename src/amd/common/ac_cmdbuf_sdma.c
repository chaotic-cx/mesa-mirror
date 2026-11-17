/*
 * Copyright 2012 Advanced Micro Devices, Inc.
 * Copyright 2025 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "ac_cmdbuf.h"
#include "ac_cmdbuf_sdma.h"
#include "ac_formats.h"
#include "ac_surface.h"
#include "sid.h"

#include "util/u_math.h"

static const uint32_t ac_sdma_t2t_alignment_2d_and_planar[][3] = {
   {16, 16, 1}, /* 1 bpp */
   {16, 8, 1},  /* 2 bpp */
   {8, 8, 1},   /* 4 bpp */
   {8, 4, 1},   /* 8 bpp */
   {4, 4, 1},   /* 16 bpp */
};

/* SDMA 4.x only. (1KB blocks) */
static const uint32_t ac_sdma4_t2t_alignment_3d[][3] = {
   {16, 8, 8}, /* 1 bpp */
   {8, 8, 8},  /* 2 bpp */
   {8, 8, 4},  /* 4 bpp */
   {8, 4, 4},  /* 8 bpp */
   {4, 4, 4},  /* 16 bpp */
};

/* SDMA 5+. (256B blocks) */
static const uint32_t ac_sdma5_t2t_alignment_3d[][3] = {
   {8, 4, 8}, /* 1 bpp */
   {4, 4, 8}, /* 2 bpp */
   {4, 4, 4}, /* 4 bpp */
   {4, 2, 4}, /* 8 bpp */
   {2, 2, 4}, /* 16 bpp */
};

static const uint32_t ac_sdma_t2t_thick_z_alignment_sdma2[] = {
   1, /* LINEAR_GENERAL */
   1, /* LINEAR_ALIGNED */
   1, /* 1D_TILED_THIN1 */
   4, /* 1D_TILED_THICK */
   1, /* 2D_TILED_THIN1 */
   1, /* PRT_TILED_THIN1 */
   1, /* PRT_2D_TILED_THIN1 */
   4, /* 2D_TILED_THICK */
   8, /* 2D_TILED_XTHICK */
   4, /* PRT_TILED_THICK */
   4, /* PRT_2D_TILED_THICK */
   1, /* PRT_3D_TILED_THIN1 */
   1, /* 3D_TILED_THIN1 */
   4, /* 3D_TILED_THICK */
   8, /* 3D_TILED_XTHICK */
   4, /* PRT_3D_TILED_THICK */
};

static bool
ac_sdma_5_2_uses_cpv(enum sdma_version sdma_ip_version)
{
   return sdma_ip_version >= SDMA_5_2 && sdma_ip_version < SDMA_7_0;
}

static uint32_t
ac_sdma_5_2_get_cache_policy_rd(enum sdma_version sdma_ip_version)
{
   /* Same cache policy as the kernel. */
   if (ac_sdma_5_2_uses_cpv(sdma_ip_version))
      return SDMA_5_2_CP_GL2_NOA | SDMA_5_2_CP_LLC_NOALLOC;

   return 0;
}

static uint32_t
ac_sdma_5_2_get_cache_policy_wr(enum sdma_version sdma_ip_version)
{
   /* Same cache policy as the kernel. */
   if (ac_sdma_5_2_uses_cpv(sdma_ip_version))
      return SDMA_5_2_CP_GL2_BYPASS | SDMA_5_2_CP_LLC_NOALLOC;

   return 0;
}

static uint32_t
ac_sdma_7_0_get_mall_policy(enum sdma_version sdma_ip_version, bool is_copy)
{
   /* Use Non-Temporal policy for all copies because re-use isn't expected and
    * it's supposedly better.
    */
   if (sdma_ip_version == SDMA_7_0)
      return is_copy ? SDMA_7_0_MALL_POLICY_NT : SDMA_7_0_MALL_POLICY_RT;

   return 0;
}

uint32_t
ac_sdma_max_img_extent(const enum sdma_version ver)
{
   if (ver >= SDMA_7_0)
      return 65536;
   else if (ver >= SDMA_2_4)
      return 16384;
   else
      return 16383; /* Just 1 pixel smaller than maximum supported image size... */
}

void
ac_emit_sdma_nop(struct ac_cmdbuf *cs)
{
   /* SDMA NOP acts as a fence command and causes the SDMA engine to wait for pending copy operations. */
   ac_cmdbuf_begin(cs);
   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_NOP, 0, 0));
   ac_cmdbuf_end();
}

void
ac_emit_sdma_write_timestamp(struct ac_cmdbuf *cs, enum sdma_version sdma_ip_version, uint64_t va)
{
   const bool cpv = ac_sdma_5_2_uses_cpv(sdma_ip_version);
   const uint32_t cp = ac_sdma_5_2_get_cache_policy_wr(sdma_ip_version);
   const uint32_t mp = ac_sdma_7_0_get_mall_policy(sdma_ip_version, false);

   ac_cmdbuf_begin(cs);
   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_TIMESTAMP, SDMA_TS_SUB_OPCODE_GET_GLOBAL_TIMESTAMP, 0) |
                  SDMA_5_2_TIMESTAMP_CPV(cpv) | SDMA_5_2_TIMESTAMP_CP(cp) | SDMA_7_0_TIMESTAMP_MALL_POLICY(mp));
   ac_cmdbuf_emit(va);
   ac_cmdbuf_emit(va >> 32);
   ac_cmdbuf_end();
}

void
ac_emit_sdma_fence(struct ac_cmdbuf *const cs,
                   const enum sdma_version sdma_ip_version,
                   const uint64_t va, const uint32_t fence)
{
   const bool cpv = ac_sdma_5_2_uses_cpv(sdma_ip_version);
   const uint32_t cp = ac_sdma_5_2_get_cache_policy_wr(sdma_ip_version);
   const uint32_t mp = ac_sdma_7_0_get_mall_policy(sdma_ip_version, false);

   ac_cmdbuf_begin(cs);

   /* Use NOP before FENCE on SDMA 2.4-3.1 to prevent hangs.
    * We noticed the hang on the Polaris 10 job in the Mesa CI.
    * Note that this issue is not documented or acknowledged by AMD.
    */
   if (sdma_ip_version >= SDMA_2_4 && sdma_ip_version <= SDMA_3_1)
      ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_NOP, 0, 0));

   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_FENCE, 0, SDMA_FENCE_MTYPE_UC) |
                  SDMA_5_2_FENCE_CPV(cpv) | SDMA_5_2_FENCE_CP(cp) | SDMA_7_0_FENCE_MALL_POLICY(mp));
   ac_cmdbuf_emit(va);
   ac_cmdbuf_emit(va >> 32);
   ac_cmdbuf_emit(fence);

   ac_cmdbuf_end();
}

void
ac_emit_sdma_wait_mem(struct ac_cmdbuf *const cs,
                      const enum sdma_version sdma_ip_version,
                      const uint32_t op, const uint64_t va,
                      const uint32_t ref, const uint32_t mask)
{
   const bool cpv = ac_sdma_5_2_uses_cpv(sdma_ip_version);
   const uint32_t cp = ac_sdma_5_2_get_cache_policy_rd(sdma_ip_version);
   const uint32_t mp = ac_sdma_7_0_get_mall_policy(sdma_ip_version, false);

   ac_cmdbuf_begin(cs);

   /* Use NOP before POLL_REGMEM on SDMA 2.4-3.1 to prevent hangs.
    * We noticed the hang on the Polaris 10 job in the Mesa CI.
    * Note that this issue is not documented or acknowledged by AMD.
    */
   if (sdma_ip_version >= SDMA_2_4 && sdma_ip_version <= SDMA_3_1)
      ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_NOP, 0, 0));

   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_POLL_REGMEM, 0, 0) | op << 28 | SDMA_POLL_MEM |
                  SDMA_5_2_POLL_REGMEM_CPV(cpv) | SDMA_5_2_POLL_REGMEM_CP(cp) | SDMA_7_0_POLL_REGMEM_MALL_POLICY(mp));
   ac_cmdbuf_emit(va);
   ac_cmdbuf_emit(va >> 32);
   ac_cmdbuf_emit(ref);
   ac_cmdbuf_emit(mask);
   ac_cmdbuf_emit(SDMA_POLL_INTERVAL_160_CLK | SDMA_POLL_RETRY_INDEFINITELY << 16);

   ac_cmdbuf_end();
}

void
ac_emit_sdma_write_data_head(struct ac_cmdbuf *cs, enum sdma_version ver, uint64_t va, uint32_t count)
{
   const bool cpv = ac_sdma_5_2_uses_cpv(ver);
   const uint32_t cp = ac_sdma_5_2_get_cache_policy_wr(ver);
   const uint32_t mp = ac_sdma_7_0_get_mall_policy(ver, false);

   ac_cmdbuf_begin(cs);
   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_WRITE, SDMA_WRITE_SUB_OPCODE_LINEAR, 0) |
                  SDMA_5_2_WRITE_LINEAR_CPV(cpv));
   ac_cmdbuf_emit(va);
   ac_cmdbuf_emit(va >> 32);
   if (ver >= SDMA_4_0)
      ac_cmdbuf_emit((count - 1) | SDMA_5_2_WRITE_LINEAR_CP(cp) | SDMA_7_0_WRITE_LINEAR_MALL_POLICY(mp));
   else
      ac_cmdbuf_emit(count);
   ac_cmdbuf_end();
}

uint64_t
ac_emit_sdma_constant_fill(struct ac_cmdbuf *cs, enum sdma_version sdma_ip_version,
                           uint64_t va, uint64_t size, uint32_t value)
{
   const uint32_t fill_size = 2; /* This means that count is in DWORDS. */

   const uint64_t max_fill_size = BITFIELD64_MASK(sdma_ip_version >= SDMA_6_0 ? 30 : 22) & ~0x3;
   const uint64_t bytes_written = MIN2(size, max_fill_size);
   const bool cpv = ac_sdma_5_2_uses_cpv(sdma_ip_version);
   const uint32_t cp = ac_sdma_5_2_get_cache_policy_wr(sdma_ip_version);
   const uint32_t mp = ac_sdma_7_0_get_mall_policy(sdma_ip_version, false);

   ac_cmdbuf_begin(cs);
   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_CONSTANT_FILL, 0, 0) | (fill_size << 30) |
                  SDMA_5_2_CONSTANT_FILL_CPV(cpv) | SDMA_5_2_CONSTANT_FILL_CP(cp) |
                  SDMA_7_0_CONSTANT_FILL_MALL_POLICY(mp));
   ac_cmdbuf_emit(va);
   ac_cmdbuf_emit(va >> 32);
   ac_cmdbuf_emit(value);
   if (sdma_ip_version >= SDMA_4_0)
      ac_cmdbuf_emit(bytes_written - 1); /* Must be programmed in bytes, even if the fill is done in dwords. */
   else
      ac_cmdbuf_emit(bytes_written);
   ac_cmdbuf_end();

   return bytes_written;
}

uint64_t
ac_emit_sdma_copy_linear(struct ac_cmdbuf *cs, enum sdma_version sdma_ip_version,
                         uint64_t src_va, uint64_t dst_va, uint64_t size,
                         bool tmz)
{
   const unsigned max_size_per_packet =
      sdma_ip_version >= SDMA_5_2 ? SDMA_5_2_COPY_MAX_BYTES : SDMA_2_0_COPY_MAX_BYTES;
   uint32_t align = ~0u;

   assert(sdma_ip_version >= SDMA_2_0);

   /* SDMA FW automatically enables a faster dword copy mode when
    * source, destination and size are all dword-aligned.
    *
    * When source and destination are dword-aligned, round down the size to
    * take advantage of faster dword copy, and copy the remaining few bytes
    * with the last copy packet.
    */
   if ((src_va & 0x3) == 0 && (dst_va & 0x3) == 0 && size > 4 && (size & 0x3) != 0) {
      align = ~0x3u;
   }

   const uint64_t bytes_written = size >= 4 ? MIN2(size & align, max_size_per_packet) : size;
   const bool cpv = ac_sdma_5_2_uses_cpv(sdma_ip_version);
   const uint32_t src_cp = ac_sdma_5_2_get_cache_policy_rd(sdma_ip_version);
   const uint32_t dst_cp = ac_sdma_5_2_get_cache_policy_wr(sdma_ip_version);
   const uint32_t mp = ac_sdma_7_0_get_mall_policy(sdma_ip_version, true);

   ac_cmdbuf_begin(cs);
   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_COPY, SDMA_COPY_SUB_OPCODE_LINEAR, (tmz ? 4 : 0)) |
                  SDMA_5_2_COPY_LINEAR_CPV(cpv));
   ac_cmdbuf_emit(sdma_ip_version >= SDMA_4_0 ? bytes_written - 1 : bytes_written);
   ac_cmdbuf_emit(SDMA_5_2_COPY_LINEAR_SRC_CP(src_cp) | SDMA_5_2_COPY_LINEAR_DST_CP(dst_cp) |
                  SDMA_7_0_COPY_LINEAR_SRC_MALL_POLICY(mp) |
                  SDMA_7_0_COPY_LINEAR_DST_MALL_POLICY(mp));
   ac_cmdbuf_emit(src_va);
   ac_cmdbuf_emit(src_va >> 32);
   ac_cmdbuf_emit(dst_va);
   ac_cmdbuf_emit(dst_va >> 32);
   ac_cmdbuf_end();

   return bytes_written;
}

static void
ac_sdma_check_pitches(enum sdma_version sdma_ip_version, uint32_t pitch,
                      uint32_t slice_pitch, uint32_t bpp, bool uses_depth)
{
   ASSERTED const uint32_t pitch_alignment = MAX2(1, 4 / bpp);
   assert(pitch);
   assert(pitch <= (1 << (sdma_ip_version >= SDMA_7_0 ? 16 : 14)));
   assert(util_is_aligned(pitch, pitch_alignment));

   if (uses_depth) {
      ASSERTED const uint32_t slice_pitch_alignment =
         sdma_ip_version >= SDMA_5_0 ? MAX2(1, 4 / bpp) : 4;
      assert(slice_pitch);
      assert(slice_pitch <= (1 << 28));
      assert(util_is_aligned(slice_pitch, slice_pitch_alignment));
   }
}

void
ac_emit_sdma_copy_linear_sub_window(struct ac_cmdbuf *cs, enum sdma_version sdma_ip_version,
                                    const struct ac_sdma_surf *src,
                                    const struct ac_sdma_surf *dst,
                                    uint32_t width, uint32_t height, uint32_t depth)
{
   assert(width <= ac_sdma_max_img_extent(sdma_ip_version));
   assert(height <= ac_sdma_max_img_extent(sdma_ip_version));

   /* Bitfield sizes between different SDMA versions:
    *
    * v2.4 - src/dst_pitch: 14 bits (shift: 16), rect_z: 11 bits
    * v4.0 - src/dst_pitch: 19 bits (shift: 13), rect_z: 11 bits
    * v5.0 - src/dst_pitch: 19 bits (shift: 13), rect_z: 13 bits
    *
    * We currently use the smallest limits (from SDMA v2.4).
    */
   uint32_t pitch_shift = (sdma_ip_version >= SDMA_7_0 || sdma_ip_version < SDMA_4_0) ? 16 : 13;
   assert(src->bpp == dst->bpp);
   assert(util_is_power_of_two_nonzero(src->bpp));
   ac_sdma_check_pitches(sdma_ip_version, src->pitch, src->slice_pitch, src->bpp, false);
   ac_sdma_check_pitches(sdma_ip_version, dst->pitch, dst->slice_pitch, dst->bpp, false);

   const bool cpv = ac_sdma_5_2_uses_cpv(sdma_ip_version);
   const uint32_t src_cp = ac_sdma_5_2_get_cache_policy_rd(sdma_ip_version);
   const uint32_t dst_cp = ac_sdma_5_2_get_cache_policy_wr(sdma_ip_version);
   const uint32_t mp = ac_sdma_7_0_get_mall_policy(sdma_ip_version, true);

   ac_cmdbuf_begin(cs);
   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_COPY, SDMA_COPY_SUB_OPCODE_LINEAR_SUB_WINDOW, 0) |
                  util_logbase2(src->bpp) << 29 | SDMA_5_2_COPY_LINEAR_SUB_WINDOW_CPV(cpv));
   ac_cmdbuf_emit(src->va);
   ac_cmdbuf_emit(src->va >> 32);
   ac_cmdbuf_emit(src->offset.x | src->offset.y << 16);
   ac_cmdbuf_emit(src->offset.z | (src->pitch - 1) << pitch_shift);
   ac_cmdbuf_emit(src->slice_pitch - 1);
   ac_cmdbuf_emit(dst->va);
   ac_cmdbuf_emit(dst->va >> 32);
   ac_cmdbuf_emit(dst->offset.x | dst->offset.y << 16);
   ac_cmdbuf_emit(dst->offset.z | (dst->pitch - 1) << pitch_shift);
   ac_cmdbuf_emit(dst->slice_pitch - 1);
   if (sdma_ip_version >= SDMA_2_4) {
      ac_cmdbuf_emit((width - 1) | (height - 1) << 16);
      ac_cmdbuf_emit((depth - 1) | SDMA_5_2_COPY_LINEAR_SUB_WINDOW_SRC_CP(src_cp) |
                     SDMA_5_2_COPY_LINEAR_SUB_WINDOW_DST_CP(dst_cp) |
                     SDMA_7_0_COPY_LINEAR_SUB_WINDOW_SRC_MALL_POLICY(mp) |
                     SDMA_7_0_COPY_LINEAR_SUB_WINDOW_DST_MALL_POLICY(mp));
   } else {
      ac_cmdbuf_emit(width | (height << 16));
      ac_cmdbuf_emit(depth);
   }
   ac_cmdbuf_end();
}

static uint32_t
ac_sdma_get_tiled_header_dword(enum sdma_version sdma_ip_version,
                               const struct ac_sdma_surf *tiled)
{
   if (sdma_ip_version >= SDMA_5_0) {
      return 0;
   } else if (sdma_ip_version >= SDMA_4_0) {
      const uint32_t mip_max = MAX2(tiled->num_levels, 1);
      const uint32_t mip_id = tiled->first_level;

      return (mip_max - 1) << 20 | mip_id << 24;
   } else {
      return 0;
   }
}

static enum gfx9_resource_type
ac_sdma_get_tiled_resource_dim_gfx9(enum sdma_version sdma_ip_version,
                                    const struct ac_sdma_surf *tiled)
{
   if (sdma_ip_version >= SDMA_5_0) {
      /* Use the 2D resource type for rotated or Z swizzles. */
      if ((tiled->surf->u.gfx9.resource_type == RADEON_RESOURCE_1D ||
           tiled->surf->u.gfx9.resource_type == RADEON_RESOURCE_3D) &&
          (tiled->surf->micro_tile_mode == RADEON_MICRO_MODE_RENDER ||
           tiled->surf->micro_tile_mode == RADEON_MICRO_MODE_DEPTH))
         return RADEON_RESOURCE_2D;
   }

   return tiled->surf->u.gfx9.resource_type;
}

static uint32_t
ac_sdma_get_tiled_info_dword(const struct radeon_info *info,
                             const struct ac_sdma_surf *tiled)
{
   const uint32_t mip_max = MAX2(tiled->num_levels, 1);
   const uint32_t mip_id = tiled->first_level;
   const uint32_t element_size = util_logbase2(tiled->bpp);
   uint32_t info_dword = 0;

   if (info->sdma_ip_version >= SDMA_4_0) {
      const uint32_t swizzle_mode =
         tiled->is_stencil ? tiled->surf->u.gfx9.zs.stencil_swizzle_mode
                           : tiled->surf->u.gfx9.swizzle_mode;
      const uint16_t epitch =
         tiled->is_stencil ? tiled->surf->u.gfx9.zs.stencil_epitch
                           : tiled->surf->u.gfx9.epitch;
      const enum gfx9_resource_type dimension =
         ac_sdma_get_tiled_resource_dim_gfx9(info->sdma_ip_version, tiled);

      info_dword |= element_size;
      info_dword |= swizzle_mode << 3;

      if (info->sdma_ip_version >= SDMA_7_0) {
         return info_dword | (mip_max - 1) << 16 | mip_id << 24;
      } else if (info->sdma_ip_version >= SDMA_5_0) {
         return info_dword | dimension << 9 | (mip_max - 1) << 16 | mip_id << 20;
      } else {
         return info_dword | dimension << 9 | epitch << 16;
      }
   } else {
      const uint32_t tile_index = ac_surface_get_legacy_tiling_index(tiled->surf, tiled->first_level, tiled->is_stencil);
      const uint32_t macro_tile_index = tiled->surf->u.legacy.macro_tile_index;
      const uint32_t tile_mode = info->si_tile_mode_array[tile_index];
      const uint32_t macro_tile_mode =
         G_009910_ARRAY_MODE(tile_mode) >= V_009910_ARRAY_2D_TILED_THIN1
            ? info->cik_macrotile_mode_array[macro_tile_index]
            : 0;
      const uint32_t tile_split_bytes =
         tiled->is_stencil ? tiled->surf->u.legacy.stencil_tile_split
                           : tiled->surf->u.legacy.tile_split;

      return element_size |
             (G_009910_ARRAY_MODE(tile_mode) << 3) |
             (G_009910_MICRO_TILE_MODE_NEW(tile_mode) << 8) |
             (util_logbase2(tile_split_bytes / 64) << 11) |
             (G_009990_BANK_WIDTH(macro_tile_mode) << 15) |
             (G_009990_BANK_HEIGHT(macro_tile_mode) << 18) |
             (G_009990_NUM_BANKS(macro_tile_mode) << 21) |
             (G_009990_MACRO_TILE_ASPECT(macro_tile_mode) << 24) |
             (G_009910_PIPE_CONFIG(tile_mode) << 26);
   }
}

static uint32_t
ac_sdma7_get_metadata_config(const struct radeon_info *info,
                             const struct ac_sdma_surf *src,
                             const struct ac_sdma_surf *dst)
{
   uint32_t meta_config = 0;

   if (src->is_compressed) {
      meta_config |= SDMA_7_0_DCC_READ_CM(2);
   }

   if (dst->is_compressed) {
      const uint32_t data_format = ac_get_cb_format(info->gfx_level, dst->format);
      const uint32_t number_type = ac_get_cb_number_type(dst->format);
      const uint32_t dcc_max_compressed_block_size =
         dst->surf->u.gfx9.color.dcc.max_compressed_block_size;

      meta_config |= SDMA_7_0_DCC_DATA_FORMAT(data_format) |
                     SDMA_7_0_DCC_NUM_TYPE(number_type) |
                     SDMA_7_0_DCC_MAX_COM(dcc_max_compressed_block_size) |
                     SDMA_7_0_DCC_MAX_UCOM(1) |
                     SDMA_7_0_DCC_WRITE_CM(1);
   }

   return meta_config;
}

static uint32_t
ac_sdma5_get_metadata_config(const struct radeon_info *info,
                             const struct ac_sdma_surf *tiled,
                             bool detile, bool tmz)
{
   const uint32_t data_format = ac_get_cb_format(info->gfx_level, tiled->format);
   const uint32_t number_type = ac_get_cb_number_type(tiled->format);
   const bool alpha_is_on_msb = ac_alpha_is_on_msb(info, tiled->format);
   const uint32_t dcc_max_compressed_block_size =
      tiled->surf->u.gfx9.color.dcc.max_compressed_block_size;
   const bool dcc_pipe_aligned = tiled->htile_enabled ||
                                 tiled->surf->u.gfx9.color.dcc.pipe_aligned;
   const bool dcc_write_compress = !detile && !tiled->htile_enabled;

   return SDMA_5_0_DCC_DATA_FORMAT(data_format) |
          SDMA_5_0_DCC_ALPHA_IS_ON_MSB(alpha_is_on_msb) |
          SDMA_5_0_DCC_NUM_TYPE(number_type) |
          SDMA_5_0_DCC_SURF_TYPE(tiled->surf_type) |
          SDMA_5_0_DCC_MAX_COM(dcc_max_compressed_block_size) |
          SDMA_5_0_DCC_PIPE_ALIGNED(dcc_pipe_aligned) |
          SDMA_5_0_DCC_MAX_UCOM(V_028C78_MAX_BLOCK_SIZE_256B) |
          SDMA_5_0_DCC_WRITE_COMPRESS(dcc_write_compress) |
          SDMA_5_0_DCC_TMZ(tmz);
}

/**
 * Return the maximum pitch of a tiled surface in units of tiles.
 */
static uint32_t
ac_sdma2_get_pitch_in_tile_max(const struct ac_sdma_surf *tiled)
{
   const uint32_t tile_width = 8;
   const struct legacy_surf_level *legacy_level =
      ac_surface_get_legacy_level(tiled->surf, tiled->first_level, tiled->is_stencil);

   return legacy_level->nblk_x / tile_width;
}

/**
 * Return the maximum slice pitch of a tiled surface in units of tiles.
 */
static uint32_t
ac_sdma2_get_slice_pitch_in_tile_max(const struct ac_sdma_surf *tiled)
{
   const uint32_t tile_width = 8;
   const uint32_t tile_height = 8;
   const uint32_t tile_pixels = tile_width * tile_height;
   const struct legacy_surf_level *legacy_level =
      ac_surface_get_legacy_level(tiled->surf, tiled->first_level, tiled->is_stencil);

   return (uint64_t)legacy_level->slice_size_dw * 4 / tiled->bpp / tile_pixels;
}

void
ac_emit_sdma_copy_tiled_sub_window(struct ac_cmdbuf *cs, const struct radeon_info *info,
                                   const struct ac_sdma_surf *linear,
                                   const struct ac_sdma_surf *tiled,
                                   bool detile, uint32_t width, uint32_t height,
                                   uint32_t depth, bool tmz)
{
   const enum sdma_version sdma_ip_version = info->sdma_ip_version;

   assert(width <= ac_sdma_max_img_extent(sdma_ip_version));
   assert(height <= ac_sdma_max_img_extent(sdma_ip_version));

   const uint32_t header_dword =
      ac_sdma_get_tiled_header_dword(sdma_ip_version, tiled);
   const uint32_t info_dword =
      ac_sdma_get_tiled_info_dword(info, tiled);
   const bool dcc =
      sdma_ip_version >= SDMA_7_0 ? (linear->is_compressed || tiled->is_compressed)
                                        : tiled->is_compressed;
   const bool cpv = ac_sdma_5_2_uses_cpv(info->sdma_ip_version);
   uint32_t rd_cp = ac_sdma_5_2_get_cache_policy_rd(info->sdma_ip_version);
   uint32_t wr_cp = ac_sdma_5_2_get_cache_policy_wr(info->sdma_ip_version);

   /* SDMA 5.2 (GFX10.3) seems to have a cache coherency issue with GL2 and
    * DCC which causes random corruption. Overwrite the default cache policy
    * and force NOA for writes and LRU for reads which seems the only
    * combination to workaround it.
    */
   if (info->sdma_ip_version == SDMA_5_2 && dcc) {
      rd_cp &= ~SDMA_5_2_CP_GL2_NOA;
      rd_cp |= SDMA_5_2_CP_GL2_LRU;

      wr_cp &= ~SDMA_5_2_CP_GL2_BYPASS;
      wr_cp |= SDMA_5_2_CP_GL2_NOA;
   }

   const uint32_t tiled_cp = detile ? rd_cp : wr_cp;
   const uint32_t linear_cp = detile ? wr_cp : rd_cp;

   const uint32_t mall_policy = ac_sdma_7_0_get_mall_policy(info->sdma_ip_version, true);

   /* Sanity checks. */
   const bool uses_depth = linear->offset.z != 0 || tiled->offset.z != 0 || depth != 1;
   assert(util_is_power_of_two_nonzero(tiled->bpp));
   ac_sdma_check_pitches(sdma_ip_version, linear->pitch, linear->slice_pitch,
                         tiled->bpp, uses_depth);
   if (!info->sdma_supports_compression)
      assert(!tiled->is_compressed);

   ac_cmdbuf_begin(cs);
   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_COPY, SDMA_COPY_SUB_OPCODE_TILED_SUB_WINDOW, (tmz ? 4 : 0)) |
                  dcc << 19 | detile << 31 | header_dword |
                  SDMA_5_2_COPY_TILED_SUB_WINDOW_CPV(cpv));
   ac_cmdbuf_emit(tiled->va);
   ac_cmdbuf_emit(tiled->va >> 32);
   ac_cmdbuf_emit(tiled->offset.x | tiled->offset.y << 16);
   if (sdma_ip_version >= SDMA_4_0) {
      ac_cmdbuf_emit(tiled->offset.z | (tiled->extent.width - 1) << 16);
      ac_cmdbuf_emit((tiled->extent.height - 1) | (tiled->extent.depth - 1) << 16);
   } else {
      ac_cmdbuf_emit(tiled->offset.z | (ac_sdma2_get_pitch_in_tile_max(tiled) - 1) << 16);
      ac_cmdbuf_emit(ac_sdma2_get_slice_pitch_in_tile_max(tiled) - 1);
   }
   ac_cmdbuf_emit(info_dword);
   ac_cmdbuf_emit(linear->va);
   ac_cmdbuf_emit(linear->va >> 32);
   ac_cmdbuf_emit(linear->offset.x | linear->offset.y << 16);
   ac_cmdbuf_emit(linear->offset.z | (linear->pitch - 1) << 16);
   ac_cmdbuf_emit(linear->slice_pitch - 1);
   if (sdma_ip_version >= SDMA_2_4) {
      ac_cmdbuf_emit((width - 1) | (height - 1) << 16);
      ac_cmdbuf_emit((depth - 1) |
                     SDMA_5_2_COPY_TILED_SUB_WINDOW_TILED_CP(tiled_cp) |
                     SDMA_5_2_COPY_TILED_SUB_WINDOW_LINEAR_CP(linear_cp) |
                     SDMA_7_0_COPY_TILED_SUB_WINDOW_TILED_MALL_POLICY(mall_policy) |
                     SDMA_7_0_COPY_TILED_SUB_WINDOW_LINEAR_MALL_POLICY(mall_policy));
   } else {
      ac_cmdbuf_emit(width | (height << 16));
      ac_cmdbuf_emit(depth);
   }

   if (dcc) {
      if (sdma_ip_version >= SDMA_7_0) {
         const struct ac_sdma_surf *src = detile ? tiled : linear;
         const struct ac_sdma_surf *dst = detile ? linear : tiled;
         const uint32_t meta_config = ac_sdma7_get_metadata_config(info, src, dst);

         ac_cmdbuf_emit(meta_config);
      } else {
         const uint32_t meta_config = ac_sdma5_get_metadata_config(info, tiled, detile, tmz);

         ac_cmdbuf_emit(tiled->meta_va);
         ac_cmdbuf_emit(tiled->meta_va >> 32);
         ac_cmdbuf_emit(meta_config);
      }
   }

   ac_cmdbuf_end();
}

void
ac_emit_sdma_copy_t2t_sub_window(struct ac_cmdbuf *cs, const struct radeon_info *info,
                                 const struct ac_sdma_surf *src,
                                 const struct ac_sdma_surf *dst,
                                 uint32_t width, uint32_t height, uint32_t depth)
{
   const enum sdma_version sdma_ip_version = info->sdma_ip_version;

   assert(width <= ac_sdma_max_img_extent(sdma_ip_version));
   assert(height <= ac_sdma_max_img_extent(sdma_ip_version));

   const uint32_t src_header_dword =
      ac_sdma_get_tiled_header_dword(sdma_ip_version, src);
   const uint32_t src_info_dword = ac_sdma_get_tiled_info_dword(info, src);
   const uint32_t dst_info_dword = ac_sdma_get_tiled_info_dword(info, dst);

   /* On SDMA 5-6.1, DCC is supported but cannot copy a compressed surface to
    * another compressed surface.
    */
   assert(sdma_ip_version >= SDMA_7_0 || !src->is_compressed || !dst->is_compressed);

   if (sdma_ip_version < SDMA_5_0) {
      /* SDMA v2-v4 doesn't support mip_id selection in the T2T copy packet. */
      assert(src_header_dword >> 24 == 0);
      /* SDMA v2-v4 doesn't support any image metadata. */
      assert(!src->is_compressed);
      assert(!dst->is_compressed);
   }
   assert(util_is_power_of_two_nonzero(src->bpp));
   assert(util_is_power_of_two_nonzero(dst->bpp));

   /* Despite the name, this can indicate DCC or HTILE metadata. */
   const uint32_t dcc = src->is_compressed || dst->is_compressed;
   /* 0 = compress (src is uncompressed), 1 = decompress (src is compressed). */
   const uint32_t dcc_dir = src->is_compressed && !dst->is_compressed;

   const bool cpv = ac_sdma_5_2_uses_cpv(info->sdma_ip_version);
   uint32_t src_cp = ac_sdma_5_2_get_cache_policy_rd(info->sdma_ip_version);
   uint32_t dst_cp = ac_sdma_5_2_get_cache_policy_wr(info->sdma_ip_version);

   /* SDMA 5.2 (GFX10.3) seems to have a cache coherency issue with GL2 and
    * DCC which causes random corruption. Overwrite the default cache policy
    * and force NOA for writes and LRU for reads which seems the only
    * combination to workaround it.
    */
   if (info->sdma_ip_version == SDMA_5_2 && dcc) {
      src_cp &= ~SDMA_5_2_CP_GL2_NOA;
      src_cp |= SDMA_5_2_CP_GL2_LRU;

      dst_cp &= ~SDMA_5_2_CP_GL2_BYPASS;
      dst_cp |= SDMA_5_2_CP_GL2_NOA;
   }

   const uint32_t mp = ac_sdma_7_0_get_mall_policy(info->sdma_ip_version, true);

   ac_cmdbuf_begin(cs);
   ac_cmdbuf_emit(SDMA_PACKET(SDMA_OPCODE_COPY, SDMA_COPY_SUB_OPCODE_T2T_SUB_WINDOW, 0) |
                  dcc << 19 | dcc_dir << 31 | src_header_dword |
                  SDMA_5_2_COPY_T2T_SUB_WINDOW_CPV(cpv));
   ac_cmdbuf_emit(src->va);
   ac_cmdbuf_emit(src->va >> 32);
   ac_cmdbuf_emit(src->offset.x | src->offset.y << 16);
   if (sdma_ip_version >= SDMA_4_0) {
      ac_cmdbuf_emit(src->offset.z | (src->extent.width - 1) << 16);
      ac_cmdbuf_emit((src->extent.height - 1) | (src->extent.depth - 1) << 16);
   } else {
      ac_cmdbuf_emit(src->offset.z | (ac_sdma2_get_pitch_in_tile_max(src) - 1) << 16);
      ac_cmdbuf_emit(ac_sdma2_get_slice_pitch_in_tile_max(src) - 1);
   }
   ac_cmdbuf_emit(src_info_dword);
   ac_cmdbuf_emit(dst->va);
   ac_cmdbuf_emit(dst->va >> 32);
   ac_cmdbuf_emit(dst->offset.x | dst->offset.y << 16);
   if (sdma_ip_version >= SDMA_4_0) {
      ac_cmdbuf_emit(dst->offset.z | (dst->extent.width - 1) << 16);
      ac_cmdbuf_emit((dst->extent.height - 1) | (dst->extent.depth - 1) << 16);
   } else {
      ac_cmdbuf_emit(dst->offset.z | (ac_sdma2_get_pitch_in_tile_max(dst) - 1) << 16);
      ac_cmdbuf_emit(ac_sdma2_get_slice_pitch_in_tile_max(dst) - 1);
   }
   ac_cmdbuf_emit(dst_info_dword);

   if (sdma_ip_version >= SDMA_4_0)
      ac_cmdbuf_emit((width - 1) | (height - 1) << 16);
   else if (sdma_ip_version >= SDMA_2_4)
      ac_cmdbuf_emit((width - 8) | ((height - 8) << 16));
   else
      ac_cmdbuf_emit(width | height << 16);

   if (sdma_ip_version >= SDMA_2_4)
      ac_cmdbuf_emit((depth - 1) |
                     SDMA_5_2_COPY_T2T_SUB_WINDOW_SRC_CP(src_cp) |
                     SDMA_5_2_COPY_T2T_SUB_WINDOW_DST_CP(dst_cp) |
                     SDMA_7_0_COPY_T2T_SUB_WINDOW_SRC_MALL_POLICY(mp) |
                     SDMA_7_0_COPY_T2T_SUB_WINDOW_DST_MALL_POLICY(mp));
   else
      ac_cmdbuf_emit(depth);

   if (dcc) {
      if (sdma_ip_version >= SDMA_7_0) {
         const uint32_t meta_config = ac_sdma7_get_metadata_config(info, src, dst);

         ac_cmdbuf_emit(meta_config);
      } else {
         if (dst->is_compressed) {
            const uint32_t dst_meta_config =
               ac_sdma5_get_metadata_config(info, dst, false, false);

            ac_cmdbuf_emit(dst->meta_va);
            ac_cmdbuf_emit(dst->meta_va >> 32);
            ac_cmdbuf_emit(dst_meta_config);
         } else {
            assert(src->is_compressed);
            const uint32_t src_meta_config =
               ac_sdma5_get_metadata_config(info, src, true, false);

            ac_cmdbuf_emit(src->meta_va);
            ac_cmdbuf_emit(src->meta_va >> 32);
            ac_cmdbuf_emit(src_meta_config);
         }
      }
   }

   ac_cmdbuf_end();
}

static bool
ac_sdma_check_t2t_alignment(const struct radeon_info *const info,
                            const struct ac_sdma_surf *const src,
                            const struct ac_sdma_surf *const dst,
                            const uint32_t width,
                            const uint32_t height,
                            const uint32_t depth)
{
   const enum sdma_version ver = info->sdma_ip_version;
   const unsigned log2bpp = util_logbase2(src->bpp);
   uint32_t x_alignment = 1;
   uint32_t y_alignment = 1;
   uint32_t z_alignment = 1;

   if (ver >= SDMA_4_0) {
      bool needs_3d_alignment = src->surf->u.gfx9.resource_type == RADEON_RESOURCE_3D;
      const uint32_t (*alignment_3d)[3] = ac_sdma5_t2t_alignment_3d;

      if (ver < SDMA_5_0) {
         /* 3D displayable swizzle modes are thin on SDMA4-4.4 and use the 2D
          * alignments. All other 3D swizzle modes (Z/S) are thick and need
          * larger alignments than on SDMA 5+.
          */
         needs_3d_alignment &= src->surf->micro_tile_mode != RADEON_MICRO_MODE_DISPLAY;
         alignment_3d = ac_sdma4_t2t_alignment_3d;
      } else if (ver < SDMA_7_0) {
         /* Only for 3D standard/displayable swizzle modes. */
         needs_3d_alignment &= (src->surf->micro_tile_mode == RADEON_MICRO_MODE_DISPLAY ||
                                src->surf->micro_tile_mode == RADEON_MICRO_MODE_STANDARD);
      }

      const uint32_t *alignment = needs_3d_alignment ? alignment_3d[log2bpp]
                                                     : ac_sdma_t2t_alignment_2d_and_planar[log2bpp];
      x_alignment = alignment[0];
      y_alignment = alignment[1];
      z_alignment = alignment[2];
   } else {
      const uint32_t src_tile_index = ac_surface_get_legacy_tiling_index(src->surf, src->first_level, src->is_stencil);
      const uint32_t src_tile_mode = info->si_tile_mode_array[src_tile_index];
      const uint32_t src_array_mode = G_009910_ARRAY_MODE(src_tile_mode);

      x_alignment = 8;
      y_alignment = 8;

      if (G_009910_MICRO_TILE_MODE_NEW(src_tile_mode) == V_009910_ADDR_SURF_THICK_MICRO_TILING)
         z_alignment = ac_sdma_t2t_thick_z_alignment_sdma2[src_array_mode];
   }

   return util_is_aligned(width, x_alignment) &&
          util_is_aligned(height, y_alignment) &&
          util_is_aligned(depth, z_alignment) &&
          util_is_aligned(src->offset.x, x_alignment) &&
          util_is_aligned(src->offset.y, y_alignment) &&
          util_is_aligned(src->offset.z, z_alignment) &&
          util_is_aligned(dst->offset.x, x_alignment) &&
          util_is_aligned(dst->offset.y, y_alignment) &&
          util_is_aligned(dst->offset.z, z_alignment);
}

/**
 * The SDMA has severe limitations on every AMD GPU generation
 * when copying a tiled image to another tiled image.
 *
 * When the HW requirements are met, we can use the built-in
 * SDMA T2T subwindow copy packet which should be the
 * most optimal.
 *
 * When the limitations are NOT met, we must use a workaround:
 * Copy the tiled image to a temporary linear buffer,
 * then copy the linear buffer to the destination image.
 * This is done by small chunks ("scanlines") so that
 * the temporary buffer doesn't have to be arbitrarily big.
 * The workaround is called a T2T scanline copy.
 *
 * Return true if the T2T scanline copy workaround is required.
 * Return false if the T2T subwindow packet can be used.
 */
bool
ac_sdma_use_t2t_scanline_copy(const struct radeon_info *const info,
                              const struct ac_sdma_surf *const src,
                              const struct ac_sdma_surf *const dst,
                              const uint32_t width,
                              const uint32_t height,
                              const uint32_t depth)
{
   /* These need a linear-to-linear / linear-to-tiled copy. */
   if (src->surf->is_linear || dst->surf->is_linear)
      return false;

   /* SDMA can't do format conversion. */
   assert(src->bpp == dst->bpp);

   const enum sdma_version ver = info->sdma_ip_version;
   if (ver < SDMA_5_0) {
      /* SDMA v4.x and older doesn't support proper mip level selection. */
      if (src->num_levels > 1 || dst->num_levels > 1)
         return true;
   }

   /* Before SDMA 7, the T2T subwindow copy packet only has fields for one metadata configuration.
    * It can either compress or decompress, or copy uncompressed images, but it can't copy from a
    * compressed image to another.
    */
   if (ver < SDMA_7_0 && src->is_compressed && dst->is_compressed)
      return true;

   if (ver >= SDMA_7_0) {
      /* No support for tiling format transformation at all. */
      if (src->surf->u.gfx9.swizzle_mode != dst->surf->u.gfx9.swizzle_mode)
         return true;
   } else {
      /* The two images can have a different block size, but must have the same swizzle mode. */
      if (src->surf->micro_tile_mode != dst->surf->micro_tile_mode)
         return true;
   }

   if (!ac_sdma_check_t2t_alignment(info, src, dst, width, height, depth))
      return true;

   if (ver < SDMA_6_0 &&
       ((src->img_format == PIPE_FORMAT_S8_UINT && dst->is_color) ||
        (src->is_color && dst->img_format == PIPE_FORMAT_S8_UINT))) {
      /* Color<->stencil only T2T subwindow copies don't work on SDMA5 and older.
       * Fallback to scanline copies as a workaround.
       */
      return true;
   }

   return false;
}

/**
 * The SDMA has some alignment requirements on
 * L2T (linear to tiled, ie. buffer to image) and
 * T2L (tiled to linear, ie. image to buffer) copies.
 *
 * When the HW requirements are met, we can use a
 * single tiled subwindow copy packet to perform the copy,
 * which performs best.
 *
 * When the HW requirements are NOT met, we must use a workaround.
 * The workaround is to do an aligned copy to a temporary buffer first,
 * and another aligned copy from the temporary to the destination.
 *
 * Return true if the unaligned copy workaround is required.
 * Return false if a single tiled subwindow packet can be used.
 */
bool
ac_sdma_use_unaligned_buffer_image_copy(const struct radeon_info *const info,
                                        const struct ac_sdma_surf *const buf,
                                        const struct ac_sdma_surf *const img,
                                        const uint32_t width,
                                        const uint32_t height,
                                        const uint32_t depth)
{
   const enum sdma_version ver = info->sdma_ip_version;

   if (ver < SDMA_4_0) {
      /* SDMA2-3 seems to access memory before the linear buffer's VA
       * when the image is tiled and the offset isn't aligned to
       * microtile width (8 pixels) or 16 bytes, causing page faults.
       * This issue isn't documented or acknowledged by AMD.
       */
      if (!img->surf->is_linear &&
          !(util_is_aligned(img->offset.x, 8) ||
            util_is_aligned(img->offset.x * img->bpp, 16)))
         return true;

      /* SDMA2-3 HW requirements for L2L/L2T/T2L copies:
       * Tiled surface must be 256-byte aligned.
       * Linear surface needs to be dword-aligned.
       * Linear pitch (and slice pitch) needs to be dword-aligned.
       * Linear and tiled offset X need to be dword-aligned
       * (multiple of 4 for 8 bits per pixel, multiple of 2 for 16 bits per pixel).
       */
      return !util_is_aligned(img->offset.x * img->bpp, 4) ||
             !util_is_aligned(buf->va, 4) ||
             !util_is_aligned(buf->pitch, 4) ||
             !util_is_aligned(buf->slice_pitch, 4) ||
             !util_is_aligned(width * img->bpp, 4);
   }

   const uint32_t required_pitch_alignment = ver >= SDMA_5_0 ? MAX2(1, 4 / img->bpp) : 4;

   if (!util_is_aligned(buf->pitch, required_pitch_alignment))
      return true;

   const bool uses_depth = img->offset.z != 0 || depth != 1;
   if (!img->surf->is_linear && uses_depth) {
      if (!util_is_aligned(buf->slice_pitch, required_pitch_alignment))
         return true;
   }

   return false;
}
