/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2010 LunarG Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Chia-I Wu <olv@lunarg.com>
 */

#include "mesa_interface.h"
#include "main/errors.h"
#include "main/texobj.h"
#include "main/teximage.h"
#include "util/hash_table.h"
#include "util/u_inlines.h"
#include "util/format/u_format.h"
#include "cso_cache/cso_context.h"
#include "nir_builder.h"
#include "st_cb_drawpixels.h"
#include "st_cb_eglimage.h"
#include "st_cb_texture.h"
#include "st_context.h"
#include "st_draw.h"
#include "st_nir.h"
#include "st_program.h"
#include "st_texture.h"
#include "st_format.h"
#include "st_manager.h"
#include "st_sampler_view.h"
#include "util/u_surface.h"

static bool
is_format_supported(struct pipe_screen *screen, enum pipe_format format,
                    unsigned nr_samples, unsigned nr_storage_samples,
                    unsigned usage, bool *native_supported)
{
   bool supported = screen->is_format_supported(screen, format, PIPE_TEXTURE_2D,
                                                nr_samples, nr_storage_samples,
                                                usage);

   if (!supported && (usage & PIPE_BIND_SAMPLER_VIEW)) {
      supported = screen->is_format_supported(screen, format, PIPE_TEXTURE_2D,
                                              nr_samples, nr_storage_samples,
                                              usage | PIPE_BIND_SAMPLER_VIEW_SUBOPTIMAL);
   }

   *native_supported = supported;

   /* for sampling, some formats can be emulated.. it doesn't matter that
    * the surface will have a format that the driver can't cope with because
    * we'll give it sampler view formats that it can deal with and generate
    * a shader variant that converts.
    */
   if ((usage == PIPE_BIND_SAMPLER_VIEW) && !supported) {
      switch (format) {
      case PIPE_FORMAT_IYUV:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage);
         break;
      case PIPE_FORMAT_NV12:
      case PIPE_FORMAT_NV21:
      case PIPE_FORMAT_NV16:
      case PIPE_FORMAT_NV61:
      case PIPE_FORMAT_NV24:
      case PIPE_FORMAT_NV42:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) &&
                     screen->is_format_supported(screen, PIPE_FORMAT_R8G8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage);
         break;
      case PIPE_FORMAT_P010:
      case PIPE_FORMAT_P012:
      case PIPE_FORMAT_P016:
      case PIPE_FORMAT_P030:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R16_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) &&
                     screen->is_format_supported(screen, PIPE_FORMAT_R16G16_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage);
         break;
      case PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM:
      case PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM:
      case PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM:
      case PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM:
      case PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM:
      case PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM:
      case PIPE_FORMAT_Y16_U16_V16_420_UNORM:
      case PIPE_FORMAT_Y16_U16_V16_422_UNORM:
      case PIPE_FORMAT_Y16_U16_V16_444_UNORM:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R16_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage);
         break;
      case PIPE_FORMAT_Y210:
         /* in principle this could be emulated by R16G16_R16B16_UNORM, but in
          * practice we won't pick that format for Y210, only the 10 bit variant
          */
         supported = screen->is_format_supported(screen, PIPE_FORMAT_X6R10X6G10_X6R10X6B10_422_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) ||
                     (screen->is_format_supported(screen, PIPE_FORMAT_R16G16_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) &&
                     screen->is_format_supported(screen, PIPE_FORMAT_R16G16B16A16_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage));
         break;
      case PIPE_FORMAT_Y212:
      case PIPE_FORMAT_Y216:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R16G16_R16B16_422_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) ||
                     (screen->is_format_supported(screen, PIPE_FORMAT_R16G16_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) &&
                     screen->is_format_supported(screen, PIPE_FORMAT_R16G16B16A16_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage));
         break;
      case PIPE_FORMAT_Y410:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R10G10B10A2_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage);
         break;
      case PIPE_FORMAT_Y412:
      case PIPE_FORMAT_Y416:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R16G16B16A16_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage);
         break;
      case PIPE_FORMAT_YUYV:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R8G8_R8B8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) ||
                     (screen->is_format_supported(screen, PIPE_FORMAT_R8G8_UNORM,
                                                  PIPE_TEXTURE_2D, nr_samples,
                                                  nr_storage_samples, usage) &&
                      screen->is_format_supported(screen, PIPE_FORMAT_B8G8R8A8_UNORM,
                                                  PIPE_TEXTURE_2D, nr_samples,
                                                  nr_storage_samples, usage));
         break;
      case PIPE_FORMAT_YVYU:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R8B8_R8G8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) ||
                     (screen->is_format_supported(screen, PIPE_FORMAT_R8G8_UNORM,
                                                  PIPE_TEXTURE_2D, nr_samples,
                                                  nr_storage_samples, usage) &&
                      screen->is_format_supported(screen, PIPE_FORMAT_B8G8R8A8_UNORM,
                                                  PIPE_TEXTURE_2D, nr_samples,
                                                  nr_storage_samples, usage));
         break;
      case PIPE_FORMAT_UYVY:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_G8R8_B8R8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) ||
                     (screen->is_format_supported(screen, PIPE_FORMAT_R8G8_UNORM,
                                                  PIPE_TEXTURE_2D, nr_samples,
                                                  nr_storage_samples, usage) &&
                      screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_UNORM,
                                                  PIPE_TEXTURE_2D, nr_samples,
                                                  nr_storage_samples, usage));
         break;
      case PIPE_FORMAT_VYUY:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_B8R8_G8R8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage) ||
                     (screen->is_format_supported(screen, PIPE_FORMAT_R8G8_UNORM,
                                                  PIPE_TEXTURE_2D, nr_samples,
                                                  nr_storage_samples, usage) &&
                      screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_UNORM,
                                                  PIPE_TEXTURE_2D, nr_samples,
                                                  nr_storage_samples, usage));
         break;
      case PIPE_FORMAT_AYUV:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage);
         break;
      case PIPE_FORMAT_XYUV:
         supported = screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8X8_UNORM,
                                                 PIPE_TEXTURE_2D, nr_samples,
                                                 nr_storage_samples, usage);
         break;
       default:
         break;
      }
   }

   return supported;
}

static bool
is_fmt_as_r8_g8b8_supported(struct pipe_screen *screen, struct st_egl_image *out,
                             unsigned usage, bool *native_supported)
{
   if (out->format == PIPE_FORMAT_NV12 &&
       out->texture->format == PIPE_FORMAT_R8_G8B8_420_UNORM &&
       screen->is_format_supported(screen, PIPE_FORMAT_R8_G8B8_420_UNORM,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }

   if (out->format == PIPE_FORMAT_NV21 &&
       out->texture->format == PIPE_FORMAT_R8_B8G8_420_UNORM &&
       screen->is_format_supported(screen, PIPE_FORMAT_R8_B8G8_420_UNORM,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }
   if (out->format == PIPE_FORMAT_NV16 &&
       out->texture->format == PIPE_FORMAT_R8_G8B8_422_UNORM &&
       screen->is_format_supported(screen, PIPE_FORMAT_R8_G8B8_422_UNORM,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }

   return false;
}

static bool
is_fmt_as_r10_g10b10_supported(struct pipe_screen *screen, struct st_egl_image *out,
                             unsigned usage, bool *native_supported)
{
   if (out->format == PIPE_FORMAT_NV15 &&
       out->texture->format == PIPE_FORMAT_R10_G10B10_420_UNORM &&
       screen->is_format_supported(screen, PIPE_FORMAT_R10_G10B10_420_UNORM,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }
   if (out->format == PIPE_FORMAT_NV20 &&
       out->texture->format == PIPE_FORMAT_R10_G10B10_422_UNORM &&
       screen->is_format_supported(screen, PIPE_FORMAT_R10_G10B10_422_UNORM,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }
   return false;
}

static bool
is_yuv420_as_r8g8b8_supported(struct pipe_screen *screen,
                              struct st_egl_image *out,
                              unsigned usage, bool *native_supported)
{
   if (out->format == PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED &&
       out->texture->format == PIPE_FORMAT_R8G8B8_420_UNORM_PACKED &&
       screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8_420_UNORM_PACKED,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }
   return false;
}

static bool
is_yuv420_as_r10g10b10_supported(struct pipe_screen *screen,
                                 struct st_egl_image *out,
                                 unsigned usage, bool *native_supported)
{
   if (out->format == PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED &&
       out->texture->format == PIPE_FORMAT_R10G10B10_420_UNORM_PACKED &&
       screen->is_format_supported(screen, PIPE_FORMAT_R10G10B10_420_UNORM_PACKED,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }
   return false;
}

static bool
is_i420_as_r8_g8_b8_420_supported(struct pipe_screen *screen,
                                  struct st_egl_image *out,
                                  unsigned usage, bool *native_supported)
{
   if (out->format == PIPE_FORMAT_IYUV &&
       out->texture->format == PIPE_FORMAT_R8_G8_B8_420_UNORM &&
       screen->is_format_supported(screen, PIPE_FORMAT_R8_G8_B8_420_UNORM,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }

   if (out->format == PIPE_FORMAT_IYUV &&
       out->texture->format == PIPE_FORMAT_R8_B8_G8_420_UNORM &&
       screen->is_format_supported(screen, PIPE_FORMAT_R8_B8_G8_420_UNORM,
                                   PIPE_TEXTURE_2D,
                                   out->texture->nr_samples,
                                   out->texture->nr_storage_samples,
                                   usage)) {
      *native_supported = false;
      return true;
   }

   return false;
}

/* whether the screen can use an EGLImage, natively or through lowering */
static bool
is_egl_image_supported(struct pipe_screen *screen, struct st_egl_image *out,
                       unsigned usage, bool *native_supported)
{
   return is_fmt_as_r8_g8b8_supported(screen, out, usage, native_supported) ||
          is_fmt_as_r10_g10b10_supported(screen, out, usage, native_supported) ||
          is_i420_as_r8_g8_b8_420_supported(screen, out, usage, native_supported) ||
          is_yuv420_as_r8g8b8_supported(screen, out, usage, native_supported) ||
          is_yuv420_as_r10g10b10_supported(screen, out, usage, native_supported) ||
          is_format_supported(screen, out->format, out->texture->nr_samples,
                              out->texture->nr_storage_samples, usage,
                              native_supported);
}

/**
 * Return the gallium texture of an EGLImage.
 */
bool
st_get_egl_image(struct gl_context *ctx, GLeglImageOES image_handle,
                 unsigned usage, bool tex_compression, const char *error,
                 struct st_egl_image *out, bool *native_supported)
{
   struct st_context *st = st_context(ctx);
   struct pipe_screen *screen = st->screen;
   struct pipe_frontend_screen *fscreen = st->frontend_screen;

   if (!fscreen || !fscreen->get_egl_image)
      return false;

   memset(out, 0, sizeof(*out));
   if (!fscreen->get_egl_image(fscreen, (void *) image_handle, out)) {
      /* image_handle does not refer to a valid EGL image object */
      _mesa_error(ctx, GL_INVALID_VALUE, "%s(image handle not found)", error);
      return false;
   }

   if (!is_egl_image_supported(screen, out, usage, native_supported)) {
      /* unable to specify a texture object using the specified EGL image */
      pipe_resource_reference(&out->texture, NULL);
      _mesa_error(ctx, GL_INVALID_OPERATION, "%s(format not supported)", error);
      return false;
   }

   if (out->texture->compression_rate != PIPE_COMPRESSION_FIXED_RATE_NONE &&
       !tex_compression) {
      /* texture is fixed-rate compressed but a uncompressed one is expected */
      pipe_resource_reference(&out->texture, NULL);
      _mesa_error(ctx, GL_INVALID_OPERATION, "%s(fixed-rate compression not enabled)", error);
      return false;
   }

   ctx->Shared->HasExternallySharedImages = true;
   return true;
}

/**
 * Return the base format just like _mesa_base_fbo_format does.
 */
static GLenum
st_pipe_format_to_base_format(enum pipe_format format)
{
   GLenum base_format;

   if (util_format_is_depth_or_stencil(format)) {
      if (util_format_is_depth_and_stencil(format)) {
         base_format = GL_DEPTH_STENCIL;
      }
      else {
         if (format == PIPE_FORMAT_S8_UINT)
            base_format = GL_STENCIL_INDEX;
         else
            base_format = GL_DEPTH_COMPONENT;
      }
   }
   else {
      /* is this enough? */
      if (util_format_has_alpha(format))
         base_format = GL_RGBA;
      else
         base_format = GL_RGB;
   }

   return base_format;
}

void
st_egl_image_target_renderbuffer_storage(struct gl_context *ctx,
                                         struct gl_renderbuffer *rb,
                                         GLeglImageOES image_handle)
{
   struct st_egl_image stimg;
   bool native_supported;

   if (st_get_egl_image(ctx, image_handle, PIPE_BIND_RENDER_TARGET, false,
                        "glEGLImageTargetRenderbufferStorage",
                        &stimg, &native_supported)) {
      struct pipe_surface surf_tmpl;

      u_surface_default_template(&surf_tmpl, stimg.texture);
      surf_tmpl.level = stimg.level;
      surf_tmpl.first_layer = stimg.layer;
      surf_tmpl.last_layer = stimg.layer;

      /* an image with the fixed little-endian variant of its format */
      enum pipe_format format = surf_tmpl.format;
      if (util_format_le_byte_order(stimg.format) == format)
         format = stimg.format;

      rb->Format = st_pipe_format_to_mesa_format(format);
      rb->_BaseFormat = st_pipe_format_to_base_format(format);
      rb->InternalFormat = rb->_BaseFormat;

      st_set_ws_renderbuffer_surface(rb, &surf_tmpl);
      pipe_resource_reference(&stimg.texture, NULL);
   }
}

void
st_bind_egl_image(struct gl_context *ctx,
                  struct gl_texture_object *texObj,
                  struct gl_texture_image *texImage,
                  struct st_egl_image *stimg,
                  bool tex_storage,
                  bool native_supported)
{
   struct st_context *st = st_context(ctx);
   GLenum internalFormat;
   mesa_format texFormat;

   if (stimg->texture->target != gl_target_to_pipe(texObj->Target)) {
      _mesa_error(ctx, GL_INVALID_OPERATION, __func__);
      return;
   }

   if (stimg->internalformat) {
      internalFormat = stimg->internalformat;
   } else {
      /* map pipe format to base format */
      if (util_format_get_component_bits(stimg->format,
                                         UTIL_FORMAT_COLORSPACE_RGB, 3) > 0)
         internalFormat = GL_RGBA;
      else
         internalFormat = GL_RGB;
   }

   /* switch to surface based */
   if (!texObj->surface_based) {
      _mesa_clear_texture_object(ctx, texObj, NULL);
      texObj->surface_based = GL_TRUE;
   }

   /* TODO RequiredTextureImageUnits should probably be reset back
    * to 1 somewhere if different texture is bound??
    */
   if (!native_supported) {
      switch (stimg->format) {
      case PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED:
         assert(stimg->texture->format == PIPE_FORMAT_R8G8B8_420_UNORM_PACKED);
         texFormat = MESA_FORMAT_R8G8B8X8_UNORM;
         texObj->RequiredTextureImageUnits = 1;
         break;
      case PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED:
         assert(stimg->texture->format == PIPE_FORMAT_R10G10B10_420_UNORM_PACKED);
         texFormat = MESA_FORMAT_R10G10B10X2_UNORM;
         texObj->RequiredTextureImageUnits = 1;
         break;
      case PIPE_FORMAT_NV12:
      case PIPE_FORMAT_NV21:
         if (stimg->texture->format == PIPE_FORMAT_R8_G8B8_420_UNORM ||
             stimg->texture->format == PIPE_FORMAT_R8_B8G8_420_UNORM) {
            texFormat = MESA_FORMAT_R8G8B8X8_UNORM;
            texObj->RequiredTextureImageUnits = 1;
         } else {
            texFormat = MESA_FORMAT_R_UNORM8;
            texObj->RequiredTextureImageUnits = 2;
         }
         break;
      case PIPE_FORMAT_NV16:
         if (stimg->texture->format == PIPE_FORMAT_R8_G8B8_422_UNORM ||
             stimg->texture->format == PIPE_FORMAT_R8_B8G8_422_UNORM) {
            texFormat = MESA_FORMAT_R8G8B8X8_UNORM;
            texObj->RequiredTextureImageUnits = 1;
         } else {
            texFormat = MESA_FORMAT_R_UNORM8;
            texObj->RequiredTextureImageUnits = 2;
         }
         break;
      case PIPE_FORMAT_NV61:
      case PIPE_FORMAT_NV24:
      case PIPE_FORMAT_NV42:
         texFormat = MESA_FORMAT_R_UNORM8;
         texObj->RequiredTextureImageUnits = 2;
         break;
      case PIPE_FORMAT_NV15:
         if (stimg->texture->format == PIPE_FORMAT_R10_G10B10_420_UNORM) {
            texFormat = MESA_FORMAT_R10G10B10X2_UNORM;
            texObj->RequiredTextureImageUnits = 1;
         } else {
            UNREACHABLE("NV15 emulation requires R10_G10B10_420_UNORM support");
         }
         break;
      case PIPE_FORMAT_NV20:
         if (stimg->texture->format == PIPE_FORMAT_R10_G10B10_422_UNORM) {
            texFormat = MESA_FORMAT_R10G10B10X2_UNORM;
            texObj->RequiredTextureImageUnits = 1;
         } else {
            UNREACHABLE("NV20 emulation requires R10_G10B10_422_UNORM support");
         }
         break;
      case PIPE_FORMAT_P010:
      case PIPE_FORMAT_P012:
      case PIPE_FORMAT_P016:
      case PIPE_FORMAT_P030:
         texFormat = MESA_FORMAT_R_UNORM16;
         texObj->RequiredTextureImageUnits = 2;
         break;
      case PIPE_FORMAT_Y210:
         if (stimg->texture->format == PIPE_FORMAT_X6R10X6G10_X6R10X6B10_422_UNORM) {
            texFormat = MESA_FORMAT_RG_RB_UNORM10;
            texObj->RequiredTextureImageUnits = 1;
         } else {
            texFormat = MESA_FORMAT_RG_UNORM16;
            texObj->RequiredTextureImageUnits = 2;
         }
         break;
      case PIPE_FORMAT_Y212:
      case PIPE_FORMAT_Y216:
         if (stimg->texture->format == PIPE_FORMAT_R16G16_R16B16_422_UNORM) {
            texFormat = MESA_FORMAT_RG_RB_UNORM16;
            texObj->RequiredTextureImageUnits = 1;
         } else {
            texFormat = MESA_FORMAT_RG_UNORM16;
            texObj->RequiredTextureImageUnits = 2;
         }
         break;
      case PIPE_FORMAT_Y410:
         texFormat = MESA_FORMAT_B10G10R10A2_UNORM;
         internalFormat = GL_RGBA;
         texObj->RequiredTextureImageUnits = 1;
         break;
      case PIPE_FORMAT_Y412:
      case PIPE_FORMAT_Y416:
         texFormat = MESA_FORMAT_RGBA_UNORM16;
         internalFormat = GL_RGBA;
         texObj->RequiredTextureImageUnits = 1;
         break;
      case PIPE_FORMAT_IYUV:
         if (stimg->texture->format == PIPE_FORMAT_R8_G8_B8_420_UNORM ||
             stimg->texture->format == PIPE_FORMAT_R8_B8_G8_420_UNORM) {
            texFormat = MESA_FORMAT_R8G8B8X8_UNORM;
            texObj->RequiredTextureImageUnits = 1;
         } else {
            texFormat = MESA_FORMAT_R_UNORM8;
            texObj->RequiredTextureImageUnits = 3;
         }
         break;
      case PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM:
      case PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM:
      case PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM:
      case PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM:
      case PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM:
      case PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM:
      case PIPE_FORMAT_Y16_U16_V16_420_UNORM:
      case PIPE_FORMAT_Y16_U16_V16_422_UNORM:
      case PIPE_FORMAT_Y16_U16_V16_444_UNORM:
         texFormat = MESA_FORMAT_R_UNORM16;
         texObj->RequiredTextureImageUnits = 3;
         break;
      case PIPE_FORMAT_YUYV:
      case PIPE_FORMAT_YVYU:
      case PIPE_FORMAT_UYVY:
      case PIPE_FORMAT_VYUY:
         if (stimg->texture->format == PIPE_FORMAT_R8G8_R8B8_UNORM) {
            texFormat = MESA_FORMAT_RG_RB_UNORM8;
            texObj->RequiredTextureImageUnits = 1;
         } else if (stimg->texture->format == PIPE_FORMAT_R8B8_R8G8_UNORM) {
            texFormat = MESA_FORMAT_RB_RG_UNORM8;
            texObj->RequiredTextureImageUnits = 1;
         } else if (stimg->texture->format == PIPE_FORMAT_G8R8_B8R8_UNORM) {
            texFormat = MESA_FORMAT_GR_BR_UNORM8;
            texObj->RequiredTextureImageUnits = 1;
         } else if (stimg->texture->format == PIPE_FORMAT_B8R8_G8R8_UNORM) {
            texFormat = MESA_FORMAT_BR_GR_UNORM8;
            texObj->RequiredTextureImageUnits = 1;
         } else {
            texFormat = MESA_FORMAT_RG_UNORM8;
            texObj->RequiredTextureImageUnits = 2;
         }
         break;
      case PIPE_FORMAT_AYUV:
         texFormat = MESA_FORMAT_R8G8B8A8_UNORM;
         internalFormat = GL_RGBA;
         texObj->RequiredTextureImageUnits = 1;
         break;
      case PIPE_FORMAT_XYUV:
         texFormat = MESA_FORMAT_R8G8B8X8_UNORM;
         texObj->RequiredTextureImageUnits = 1;
         break;
      default:
         UNREACHABLE("unexpected emulated format");
         break;
      }
      assert(texFormat != MESA_FORMAT_NONE);
   } else {
      texFormat = st_pipe_format_to_mesa_format(stimg->format);

      /* Use previously derived internalformat as specified by
       * EXT_EGL_image_storage.
       */
      if (tex_storage && texObj->Target == GL_TEXTURE_2D
          && stimg->internalformat) {
         internalFormat = stimg->internalformat;
         if (internalFormat == GL_NONE) {
            _mesa_error(ctx, GL_INVALID_OPERATION, __func__);
            return;
         }
      }

      assert(texFormat != MESA_FORMAT_NONE || util_format_is_yuv(stimg->format));
   }


   /* Minify texture size based on level set on the EGLImage. */
   uint32_t width = u_minify(stimg->texture->width0, stimg->level);
   uint32_t height = u_minify(stimg->texture->height0, stimg->level);

   _mesa_init_teximage_fields(ctx, texImage, width, height,
                              1, 0, internalFormat, texFormat);

   pipe_resource_reference(&texObj->pt, stimg->texture);
   st_texture_release_all_sampler_views(st, texObj);
   pipe_resource_reference(&texImage->pt, texObj->pt);
   if (st->screen->resource_changed)
      st->screen->resource_changed(st->screen, texImage->pt);

   texObj->surface_format = st_get_surface_format(stimg->format, texObj->pt);

   /* Cache whether the extra YUV plane-view / lowering setup in
    * st_get_sampler_views() and st_get_external_sampler_key() must run for
    * this texture. st_finalize_texture() returns early for surface_based
    * textures without computing this.
    */
   texObj->needs_yuv_plane_views = (stimg->format != texObj->pt->format ||
                                    util_format_is_yuv(stimg->format));

   switch (stimg->yuv_color_space) {
   case __DRI_YUV_COLOR_SPACE_ITU_REC709:
      texObj->yuv_color_space = GL_TEXTURE_YUV_COLOR_SPACE_REC709;
      break;
   case __DRI_YUV_COLOR_SPACE_ITU_REC2020:
      texObj->yuv_color_space = GL_TEXTURE_YUV_COLOR_SPACE_REC2020;
      break;
   default:
      texObj->yuv_color_space = GL_TEXTURE_YUV_COLOR_SPACE_REC601;
      break;
   }

   if (stimg->yuv_range == __DRI_YUV_FULL_RANGE)
      texObj->yuv_full_range = true;

   texObj->CompressionRate = stimg->texture->compression_rate;

   texObj->level_override = stimg->level;
   texObj->layer_override = stimg->layer;
   _mesa_update_texture_object_swizzle(ctx, texObj);

   _mesa_dirty_texobj(ctx, texObj);
}

bool
st_validate_egl_image(struct gl_context *ctx, GLeglImageOES image_handle)
{
   struct st_context *st = st_context(ctx);
   struct pipe_frontend_screen *fscreen = st->frontend_screen;

   return fscreen->validate_egl_image(fscreen, (void *)image_handle);
}

/* look up an EGLImage without setting GL errors */
static bool
lookup_egl_image(struct gl_context *ctx, GLeglImageOES image_handle,
                 struct st_egl_image *stimg)
{
   struct pipe_frontend_screen *fscreen = st_context(ctx)->frontend_screen;

   return fscreen && fscreen->validate_egl_image && fscreen->get_egl_image &&
          st_validate_egl_image(ctx, image_handle) &&
          fscreen->get_egl_image(fscreen, (void *)image_handle, stimg);
}

/**
 * Return the size and format of an EGLImage without binding it or setting GL
 * errors, for tools that capture its contents. format may be NULL.
 */
bool
st_get_egl_image_info(struct gl_context *ctx, GLeglImageOES image_handle,
                      unsigned *width, unsigned *height,
                      enum pipe_format *format)
{
   struct st_egl_image stimg = {0};

   if (!lookup_egl_image(ctx, image_handle, &stimg))
      return false;

   *width = u_minify(stimg.texture->width0, stimg.level);
   *height = u_minify(stimg.texture->height0, stimg.level);
   if (format)
      *format = stimg.format;
   pipe_resource_reference(&stimg.texture, NULL);
   return true;
}

/* convert the whole of res, a single-level 2D resource, to format in data */
static bool
map_resource(struct pipe_context *pipe, struct pipe_resource *res,
             enum pipe_format format, void *data, unsigned stride)
{
   struct pipe_transfer *transfer;
   const void *map = pipe_texture_map(pipe, res, 0, 0, PIPE_MAP_READ, 0, 0,
                                      res->width0, res->height0, &transfer);
   if (!map)
      return false;

   bool read = util_format_translate(format, data, stride, 0, 0, res->format,
                                     map, transfer->stride, 0, 0, res->width0,
                                     res->height0);
   pipe_texture_unmap(pipe, transfer);
   return read;
}

/*
 * copy one level and layer of src through a linear copy in its own format,
 * which any layout copies to, and convert it to format in data
 */
static bool
read_resource(struct pipe_context *pipe, struct pipe_resource *src,
              unsigned level, unsigned layer, enum pipe_format format,
              void *data, unsigned stride)
{
   struct pipe_screen *screen = pipe->screen;
   unsigned width = u_minify(src->width0, level);
   unsigned height = u_minify(src->height0, level);
   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = src->format,
      .width0 = width,
      .height0 = height,
      .depth0 = 1,
      .array_size = 1,
      .usage = PIPE_USAGE_STAGING,
   };
   struct pipe_resource *staging = screen->resource_create(screen, &templ);
   if (!staging)
      return false;

   struct pipe_box box;
   u_box_3d(0, 0, layer, width, height, 1, &box);
   pipe->resource_copy_region(pipe, staging, 0, 0, 0, 0, src, level, &box);

   bool read = map_resource(pipe, staging, format, data, stride);
   pipe_resource_reference(&staging, NULL);
   return read;
}

/*
 * the lowering of external sampler unit 0 in key, as bits; every field of
 * the key is a mask of sampler units
 */
static uint32_t
external_sampler_key_bits(const struct st_external_sampler_key *key)
{
   const GLuint *fields = (const GLuint *)key;
   uint32_t bits = 0;

   STATIC_ASSERT(sizeof(*key) % sizeof(GLuint) == 0);
   STATIC_ASSERT(sizeof(*key) / sizeof(GLuint) <= 32);
   for (unsigned i = 0; i < sizeof(*key) / sizeof(GLuint); i++)
      bits |= (fields[i] & 1) << i;
   return bits;
}

/*
 * a fragment shader that samples external sampler unit 0 at the TEX0
 * coordinates, lowered with key as the shaders of apps are
 */
static void *
get_external_sampler_fs(struct st_context *st,
                        const struct st_external_sampler_key *key)
{
   void *bits = (void *)(uintptr_t)external_sampler_key_bits(key);

   if (!st->egl_image_read_shaders)
      st->egl_image_read_shaders = _mesa_hash_table_create_u32_keys(NULL);
   struct hash_entry *he =
      _mesa_hash_table_search(st->egl_image_read_shaders, bits);
   if (he)
      return he->data;

   const nir_shader_compiler_options *options =
      st->screen->nir_options[MESA_SHADER_FRAGMENT];
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, options,
                                                  "st/eglimage read FS");
   b.shader->info.io_lowered = true;

   nir_def *baryc = nir_load_barycentric_pixel(&b, 32,
                                              .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *coord =
      nir_load_interpolated_input(&b, 2, 32, baryc, nir_imm_int(&b, 0),
                                  .io_semantics.location = VARYING_SLOT_TEX0);
   const struct glsl_type *type =
      glsl_sampler_type(GLSL_SAMPLER_DIM_EXTERNAL, false, false,
                        GLSL_TYPE_FLOAT);
   nir_variable *var =
      nir_variable_create(b.shader, nir_var_uniform, type, "image");
   var->data.binding = 0;
   var->data.explicit_binding = true;
   nir_deref_instr *deref = nir_build_deref_var(&b, var);
   nir_def *color = nir_tex(&b, coord, .texture_deref = deref,
                            .sampler_deref = deref);
   nir_store_output(&b, color, nir_imm_int(&b, 0),
                    .io_semantics.location = FRAG_RESULT_COLOR,
                    .src_type = nir_type_float32);

   st_nir_lower_external_samplers(b.shader, key);
   st_nir_lower_external_sampler_planes(b.shader, key, ~BITFIELD_BIT(0));

   void *fs = st_nir_finish_builtin_shader(st, b.shader);
   if (fs)
      _mesa_hash_table_insert(st->egl_image_read_shaders, bits, fs);
   return fs;
}

void
st_destroy_egl_image_read(struct st_context *st)
{
   if (!st->egl_image_read_shaders)
      return;

   hash_table_foreach(st->egl_image_read_shaders, entry)
      st->pipe->delete_fs_state(st->pipe, entry->data);
   _mesa_hash_table_destroy(st->egl_image_read_shaders, NULL);
}

/* draw fs, sampling views with nearest filtering, over dst */
static bool
draw_fs(struct st_context *st, void *fs, struct pipe_sampler_view **views,
        unsigned num_views, struct pipe_resource *dst)
{
   struct pipe_context *pipe = st->pipe;
   struct cso_context *cso = st->cso_context;

   /*
    * save only the states that have no st_atom and invalidate the rest
    * afterwards, as try_pbo_upload_common does
    */
   cso_save_state(cso, CSO_BIT_STREAM_OUTPUTS |
                       (st->active_queries ? CSO_BIT_PAUSE_QUERIES : 0) |
                       CSO_BIT_RENDER_CONDITION);
   cso_set_sample_mask(cso, ~0);
   cso_set_sample_coverage(cso, 1.0f, false);
   cso_set_min_samples(cso, 1);
   cso_set_render_condition(cso, NULL, false, 0);

   struct pipe_framebuffer_state fb = {
      .width = dst->width0,
      .height = dst->height0,
      .nr_cbufs = 1,
   };
   u_surface_default_template(&fb.cbufs[0], dst);
   cso_set_framebuffer(cso, &fb);
   cso_set_viewport_dims(cso, dst->width0, dst->height0, false);

   struct pipe_blend_state blend = {0};
   blend.rt[0].colormask = PIPE_MASK_RGBA;
   cso_set_blend(cso, &blend);
   struct pipe_depth_stencil_alpha_state dsa = {0};
   cso_set_depth_stencil_alpha(cso, &dsa);
   struct pipe_rasterizer_state rasterizer = {.half_pixel_center = 1};
   cso_set_rasterizer(cso, &rasterizer);
   /* window rectangles have no cso state, so turn them off by hand */
   bool window_rects = st->state.window_rects.num ||
                       st->state.window_rects.include;
   if (window_rects)
      pipe->set_window_rectangles(pipe, false, 0, NULL);

   struct pipe_sampler_state sampler = {
      .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .min_img_filter = PIPE_TEX_FILTER_NEAREST,
      .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
      .min_mip_filter = PIPE_TEX_MIPFILTER_NONE,
   };
   const struct pipe_sampler_state *samplers[PIPE_MAX_SAMPLERS];
   for (unsigned i = 0; i < num_views; i++)
      samplers[i] = &sampler;
   cso_set_samplers(cso, MESA_SHADER_FRAGMENT, num_views, samplers);
   pipe->set_sampler_views(pipe, MESA_SHADER_FRAGMENT, 0, num_views, 0,
                           views);
   st->state.num_sampler_views[MESA_SHADER_FRAGMENT] =
      MAX2(st->state.num_sampler_views[MESA_SHADER_FRAGMENT], num_views);

   st_make_passthrough_vertex_shader(st);
   cso_set_vertex_shader_handle(cso, st->passthrough_vs);
   cso_set_tessctrl_shader_handle(cso, NULL);
   cso_set_tesseval_shader_handle(cso, NULL);
   cso_set_geometry_shader_handle(cso, NULL);
   cso_set_mesh_shader_handle(cso, NULL);
   cso_set_fragment_shader_handle(cso, fs);
   st->util_velems.count = 3;
   cso_set_vertex_elements(cso, &st->util_velems);
   cso_set_stream_outputs(cso, 0, NULL, NULL, 0);

   /* the first row, at y = -1, samples the first row of the image */
   static const float color[4];
   bool drawn = st_draw_quad(st, -1.0f, -1.0f, 1.0f, 1.0f, 0.0f,
                             0.0f, 1.0f, 1.0f, 0.0f, color, 0);

   cso_restore_state(cso, CSO_UNBIND_FS_SAMPLERVIEWS);
   if (window_rects)
      pipe->set_window_rectangles(pipe, st->state.window_rects.include,
                                  st->state.window_rects.num,
                                  st->state.window_rects.rects);
   st->state.num_sampler_views[MESA_SHADER_FRAGMENT] = 0;
   st_context_invalidate_state(st, ST_INVALIDATE_VERTEX_BUFFERS |
                                   ST_INVALIDATE_FB_STATE |
                                   ST_INVALIDATE_VIEWPORT |
                                   ST_INVALIDATE_BLEND |
                                   ST_INVALIDATE_DSA |
                                   ST_INVALIDATE_RASTERIZER |
                                   ST_INVALIDATE_SAMPLE_MASK |
                                   ST_INVALIDATE_SAMPLE_SHADING |
                                   ST_INVALIDATE_FS_SAMPLERS |
                                   ST_INVALIDATE_FS_SAMPLER_VIEWS |
                                   ST_INVALIDATE_VS_STATE |
                                   ST_INVALIDATE_FS_STATE |
                                   ST_INVALIDATE_GS_STATE |
                                   ST_INVALIDATE_TCS_STATE |
                                   ST_INVALIDATE_TES_STATE |
                                   ST_INVALIDATE_MESH_STATE);
   return drawn;
}

/* draw texObj, bound to an EGLImage, over dst as st samples it for apps */
static bool
draw_texture(struct st_context *st, struct gl_texture_object *texObj,
             struct pipe_resource *dst)
{
   struct pipe_context *pipe = st->pipe;
   struct pipe_sampler_view *views[PIPE_MAX_SAMPLERS] = {0};

   views[0] = st_get_texture_sampler_view_from_stobj(st, texObj,
                                                     &texObj->Sampler, true,
                                                     false, 0, NULL);
   if (!views[0])
      return false;

   struct st_external_sampler_key key = {0};
   unsigned extra_views = 0;
   st_add_yuv_plane_views(pipe, texObj, views[0], BITFIELD_BIT(0), views,
                          &extra_views);
   st_add_external_sampler_key(&key, 0, texObj);
   void *fs = get_external_sampler_fs(st, &key);
   bool drawn = fs && draw_fs(st, fs, views, util_last_bit(extra_views | 1),
                              dst);

   /* release YUV views back to driver */
   u_foreach_bit (i, extra_views)
      pipe->sampler_view_release(pipe, views[i]);
   return drawn;
}

/* draw stimg over dst through a private texture object */
static bool
draw_egl_image(struct st_context *st, struct st_egl_image *stimg,
               bool native_supported, struct pipe_resource *dst)
{
   struct gl_context *ctx = st->ctx;
   struct gl_texture_object *texObj =
      _mesa_new_texture_object(ctx, 0, GL_TEXTURE_2D);
   if (!texObj)
      return false;

   struct gl_texture_image *texImage =
      _mesa_get_tex_image(ctx, texObj, GL_TEXTURE_2D, 0);
   bool drawn = false;
   if (texImage) {
      st_bind_egl_image(ctx, texObj, texImage, stimg, false, native_supported);
      drawn = draw_texture(st, texObj, dst);
   }
   _mesa_reference_texobj(&texObj, NULL);
   return drawn;
}

/* blit stimg, resolving its samples, over dst */
static void
blit_egl_image(struct pipe_context *pipe, struct st_egl_image *stimg,
               struct pipe_resource *dst)
{
   struct pipe_blit_info blit = {
      .src.resource = stimg->texture,
      .src.format = stimg->format,
      .src.level = stimg->level,
      .dst.resource = dst,
      .dst.format = dst->format,
      .mask = PIPE_MASK_RGBA,
      .filter = PIPE_TEX_FILTER_NEAREST,
   };

   u_box_2d_zslice(0, 0, stimg->layer, dst->width0, dst->height0,
                   &blit.src.box);
   u_box_2d(0, 0, dst->width0, dst->height0, &blit.dst.box);
   pipe->blit(pipe, &blit);
}

/*
 * read stimg through a render target in format, which YUV images are drawn
 * to, converting as sampling them does, and other images are blitted to,
 * resolving their samples
 */
static bool
read_through_target(struct st_context *st, struct st_egl_image *stimg,
                    enum pipe_format format, void *data, unsigned stride)
{
   struct pipe_screen *screen = st->screen;
   struct pipe_resource *src = stimg->texture;
   bool draw = util_format_is_yuv(stimg->format);
   bool native_supported = true;

   if (!screen->is_format_supported(screen, format, PIPE_TEXTURE_2D, 0, 0,
                                    PIPE_BIND_RENDER_TARGET))
      return false;
   /*
    * st_bind_egl_image sets GL errors for targets other than 2D, and the
    * draw writes floats
    */
   if (draw && (src->target != PIPE_TEXTURE_2D ||
                util_format_is_pure_integer(format) ||
                !is_egl_image_supported(screen, stimg, PIPE_BIND_SAMPLER_VIEW,
                                        &native_supported)))
      return false;
   /*
    * blits sample the source, and convert neither depth nor between
    * integer and other formats
    */
   if (!draw && (!screen->is_format_supported(screen, stimg->format,
                                              src->target, src->nr_samples,
                                              src->nr_storage_samples,
                                              PIPE_BIND_SAMPLER_VIEW) ||
                 util_format_is_depth_or_stencil(stimg->format) ||
                 util_format_is_pure_sint(stimg->format) !=
                 util_format_is_pure_sint(format) ||
                 util_format_is_pure_uint(stimg->format) !=
                 util_format_is_pure_uint(format)))
      return false;

   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = format,
      .width0 = u_minify(src->width0, stimg->level),
      .height0 = u_minify(src->height0, stimg->level),
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   struct pipe_resource *dst = screen->resource_create(screen, &templ);
   if (!dst)
      return false;

   bool filled = true;
   if (draw)
      filled = draw_egl_image(st, stimg, native_supported, dst);
   else
      blit_egl_image(st->pipe, stimg, dst);
   bool read = filled && map_resource(st->pipe, dst, format, data, stride);
   pipe_resource_reference(&dst, NULL);
   return read;
}

/**
 * Copy the contents of an EGLImage, converted to format, into data with
 * stride bytes per row, without binding it or touching GL state, for tools
 * that capture it. YUV images convert to RGB as sampling them does.
 */
bool
st_read_egl_image(struct gl_context *ctx, GLeglImageOES image_handle,
                  enum pipe_format format, void *data, unsigned stride)
{
   struct st_context *st = st_context(ctx);
   struct st_egl_image stimg = {0};

   if (!lookup_egl_image(ctx, image_handle, &stimg))
      return false;

   /*
    * formats that can't be rendered to convert on the CPU, from the format
    * of the resource
    */
   bool read = read_through_target(st, &stimg, format, data, stride) ||
               (stimg.format == stimg.texture->format &&
                !util_format_is_yuv(stimg.format) &&
                stimg.texture->nr_samples <= 1 &&
                read_resource(st->pipe, stimg.texture, stimg.level,
                              stimg.layer, format, data, stride));
   pipe_resource_reference(&stimg.texture, NULL);
   return read;
}
