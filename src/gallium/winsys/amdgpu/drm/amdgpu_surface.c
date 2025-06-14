/*
 * Copyright © 2011 Red Hat All Rights Reserved.
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "amdgpu_winsys.h"
#include "util/format/u_format.h"

static int amdgpu_surface_sanity(const struct pipe_resource *tex)
{
   switch (tex->target) {
   case PIPE_TEXTURE_1D:
      if (tex->height0 > 1)
         return -EINVAL;
      FALLTHROUGH;
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_RECT:
      if (tex->depth0 > 1 || tex->array_size > 1)
         return -EINVAL;
      break;
   case PIPE_TEXTURE_3D:
      if (tex->array_size > 1)
         return -EINVAL;
      break;
   case PIPE_TEXTURE_1D_ARRAY:
      if (tex->height0 > 1)
         return -EINVAL;
      FALLTHROUGH;
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE_ARRAY:
      if (tex->depth0 > 1)
         return -EINVAL;
      break;
   default:
      return -EINVAL;
   }
   return 0;
}

static int amdgpu_surface_init(struct radeon_winsys *rws,
                               const struct radeon_info *info,
                               const struct pipe_resource *tex,
                               uint64_t flags, unsigned bpe,
                               enum radeon_surf_mode mode,
                               struct radeon_surf *surf)
{
   struct amdgpu_winsys *aws = amdgpu_winsys(rws);
   int r;

   r = amdgpu_surface_sanity(tex);
   if (r)
      return r;

   surf->blk_w = util_format_get_blockwidth(tex->format);
   surf->blk_h = util_format_get_blockheight(tex->format);
   surf->bpe = bpe;
   surf->flags = flags;

   struct ac_surf_config config;

   config.info.width = tex->width0;
   config.info.height = tex->height0;
   config.info.depth = tex->depth0;
   config.info.array_size = tex->array_size;
   config.info.samples = tex->nr_samples;
   config.info.storage_samples = tex->nr_storage_samples;
   config.info.levels = tex->last_level + 1;
   config.info.num_channels = util_format_get_nr_components(tex->format);
   config.is_1d = tex->target == PIPE_TEXTURE_1D ||
                  tex->target == PIPE_TEXTURE_1D_ARRAY;
   config.is_3d = tex->target == PIPE_TEXTURE_3D;
   config.is_cube = tex->target == PIPE_TEXTURE_CUBE;
   config.is_array = tex->target == PIPE_TEXTURE_1D_ARRAY ||
                     tex->target == PIPE_TEXTURE_2D_ARRAY ||
                     tex->target == PIPE_TEXTURE_CUBE_ARRAY;

   /* Use different surface counters for color and FMASK, so that MSAA MRTs
    * always use consecutive surface indices when FMASK is allocated between
    * them.
    */
   config.info.surf_index = &aws->surf_index_color;
   config.info.fmask_surf_index = &aws->surf_index_fmask;

   if (flags & RADEON_SURF_Z_OR_SBUFFER)
      config.info.surf_index = NULL;

   /* Use radeon_info from the driver, not the winsys. The driver is allowed to change it. */
   return ac_compute_surface(aws->addrlib, info, &config, mode, surf);
}

static uint64_t
amdgpu_surface_offset_from_coord(struct radeon_winsys *rws,
                                 const struct radeon_info *info,
                                 const struct radeon_surf *surf,
                                 const struct pipe_resource *tex,
                                 unsigned level, unsigned x,
                                 unsigned y, unsigned layer)
{
   struct amdgpu_winsys *aws = amdgpu_winsys(rws);
   unsigned samples = MAX2(1, tex->nr_samples);

   const struct ac_surf_info surf_info = {
      .width = tex->width0,
      .height = tex->height0,
      .depth = tex->depth0,
      .samples = samples,
      .storage_samples = samples,
      .levels = tex->last_level + 1,
      .num_channels = util_format_get_last_component(tex->format) + 1,
      .array_size = tex->array_size,
   };

   return ac_surface_addr_from_coord(
      aws->addrlib, info, surf, &surf_info,
      level, x, y, layer, tex->target == PIPE_TEXTURE_3D);
}

void amdgpu_surface_init_functions(struct amdgpu_screen_winsys *sws)
{
   sws->base.surface_init = amdgpu_surface_init;
   sws->base.surface_offset_from_coord = amdgpu_surface_offset_from_coord;
}
