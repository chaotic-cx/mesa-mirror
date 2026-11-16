/*
 * Copyright (c) 2016 Etnaviv Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Christian Gmeiner <christian.gmeiner@gmail.com>
 */

#include "etnaviv_format.h"
#include "etnaviv_screen.h"

#include "hw/common_3d.xml.h"
#include "hw/state.xml.h"
#include "hw/state_3d.xml.h"

#include "pipe/p_defines.h"
#include "util/compiler.h"

static inline bool
pe_format_is_native_rgba(enum pipe_format fmt, const struct etna_screen *screen)
{
   return fmt == PIPE_FORMAT_R8G8B8A8_UNORM &&
          VIV_FEATURE(screen, ETNA_FEATURE_PE_A8B8G8R8);
}

uint32_t
translate_texture_format(enum pipe_format fmt, const struct etna_screen *screen)
{
   fmt = util_format_linear(fmt);
   assert(eil_format_supports_sampling(screen->info, fmt));

   fmt = eil_format_emulated(fmt);

   const struct eil_format *f = eil_get_format_entry(fmt);
   uint32_t format = f->tex;

   if (f->tex_ext)
      format |= EXT_FORMAT;

   if (f->tex_astc)
      format |= ASTC_FORMAT;

   if (pe_format_is_native_rgba(fmt, screen))
      format = remap_texture_format_rb_swap(format);

   if (screen->info->halti >= 5) {
      if (fmt == PIPE_FORMAT_R32_SINT || fmt == PIPE_FORMAT_R32_UINT)
         format = TEXTURE_FORMAT_EXT_R32I | EXT_FORMAT;
      else if (fmt == PIPE_FORMAT_R32G32_SINT || fmt == PIPE_FORMAT_R32G32_UINT)
         format = TEXTURE_FORMAT_EXT_G32R32I | EXT_FORMAT;
   }

   return format;
}

bool
texture_use_int_filter(const struct pipe_sampler_view *sv,
                       const struct pipe_sampler_state *ss,
                       bool tex_desc)
{
   switch (sv->target) {
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D_ARRAY:
      if (tex_desc)
         break;
      FALLTHROUGH;
   case PIPE_TEXTURE_3D:
      return false;
   default:
      break;
   }

   /* only unorm formats can use int filter */
   if (!util_format_is_unorm(sv->format))
      return false;

   if (util_format_is_srgb(sv->format))
      return false;

   if (util_format_is_depth_or_stencil(sv->format))
      return false;

   if (util_format_description(sv->format)->layout == UTIL_FORMAT_LAYOUT_ASTC)
      return false;

   if (ss->max_anisotropy > 1)
      return false;

   switch (sv->format) {
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R10G10B10X2_UNORM:
   case PIPE_FORMAT_ETC2_R11_UNORM:
   case PIPE_FORMAT_ETC2_RG11_UNORM:
      return false;
   default:
      return true;
   }
}

uint32_t
get_texture_swiz(enum pipe_format fmt, unsigned swizzle_r,
                 unsigned swizzle_g, unsigned swizzle_b, unsigned swizzle_a)
{
   unsigned char swiz[4] = {
      swizzle_r, swizzle_g, swizzle_b, swizzle_a,
   };

   /* 128-bit texels are swizzled in the shader once the halves are joined. */
   if (eil_format_is_emulated_128bit(fmt)) {
      swiz[0] = PIPE_SWIZZLE_X;
      swiz[1] = PIPE_SWIZZLE_Y;
      swiz[2] = PIPE_SWIZZLE_Z;
      swiz[3] = PIPE_SWIZZLE_W;
   }

   if (unlikely(fmt == PIPE_FORMAT_DXT1_RGB)) {
      /* The HW uses the same decompression scheme for RGB and RGBA DXT1
       * textures, tell it to 1-fill the alpha channel for plain RGB.
       */
      for (unsigned i = 0; i < 4; i++) {
         if (swiz[i] == PIPE_SWIZZLE_W)
            swiz[i] = PIPE_SWIZZLE_1;
      }
   }

   if (util_format_linear(fmt) == PIPE_FORMAT_R8_UNORM) {
      /* R8 is emulated with L8, needs yz channels set to zero */
      for (unsigned i = 0; i < 4; i++) {
         if (swiz[i] == PIPE_SWIZZLE_Y || swiz[i] == PIPE_SWIZZLE_Z)
            swiz[i] = PIPE_SWIZZLE_0;
      }
   }

   /* PIPE_SWIZZLE_ maps 1:1 to TEXTURE_SWIZZLE_ */
   STATIC_ASSERT(PIPE_SWIZZLE_X == TEXTURE_SWIZZLE_RED);
   STATIC_ASSERT(PIPE_SWIZZLE_Y == TEXTURE_SWIZZLE_GREEN);
   STATIC_ASSERT(PIPE_SWIZZLE_Z == TEXTURE_SWIZZLE_BLUE);
   STATIC_ASSERT(PIPE_SWIZZLE_W == TEXTURE_SWIZZLE_ALPHA);
   STATIC_ASSERT(PIPE_SWIZZLE_0 == TEXTURE_SWIZZLE_ZERO);
   STATIC_ASSERT(PIPE_SWIZZLE_1 == TEXTURE_SWIZZLE_ONE);

   return VIVS_TE_SAMPLER_CONFIG1_SWIZZLE_R(swiz[0]) |
          VIVS_TE_SAMPLER_CONFIG1_SWIZZLE_G(swiz[1]) |
          VIVS_TE_SAMPLER_CONFIG1_SWIZZLE_B(swiz[2]) |
          VIVS_TE_SAMPLER_CONFIG1_SWIZZLE_A(swiz[3]);
}

uint32_t
translate_pe_format(enum pipe_format fmt, const struct etna_screen *screen)
{
   fmt = util_format_linear(fmt);
   assert(eil_format_supports_rendering(screen->info, fmt));

   if (pe_format_is_native_rgba(fmt, screen))
      return PE_FORMAT_A8B8G8R8;

   return eil_get_format_entry(fmt)->pe;
}

bool
translate_pe_format_rb_swap(enum pipe_format fmt, const struct etna_screen *screen)
{
   fmt = util_format_linear(fmt);

   if (pe_format_is_native_rgba(fmt, screen))
      return false;

   return eil_get_format_entry(fmt)->pe_rb_swap;
}

/* For RB_SWAP formats, remaps the HW texture format to the one matching
 * native byte order in memory (e.g., A8B8G8R8 for RGBA data). Normally we
 * use A8R8G8B8 to match PE-internal BGRA byte order, but shared resources
 * that have been flushed store data in the standard byte order.
 */
uint32_t
remap_texture_format_rb_swap(uint32_t format)
{
   switch (format) {
   case TEXTURE_FORMAT_A8R8G8B8: return TEXTURE_FORMAT_A8B8G8R8;
   case TEXTURE_FORMAT_X8R8G8B8: return TEXTURE_FORMAT_X8B8G8R8;
   default: return format;
   }
}

enum pipe_format
translate_pe_internal_format(enum pipe_format fmt, const struct etna_screen *screen)
{
   if (!translate_pe_format_rb_swap(fmt, screen))
      return fmt;

   switch (fmt) {
   case PIPE_FORMAT_R8G8B8A8_UNORM: return PIPE_FORMAT_B8G8R8A8_UNORM;
   case PIPE_FORMAT_R8G8B8X8_UNORM: return PIPE_FORMAT_B8G8R8X8_UNORM;
   case PIPE_FORMAT_R8G8B8A8_SRGB:  return PIPE_FORMAT_B8G8R8A8_SRGB;
   case PIPE_FORMAT_R8G8B8X8_SRGB:  return PIPE_FORMAT_B8G8R8X8_SRGB;
   default:
      UNREACHABLE("unexpected rb_swap format");
   }
}
