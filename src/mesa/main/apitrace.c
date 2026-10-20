/*
 * SPDX-License-Identifier: MIT
 */

#include "main/apitrace.h"

#include "glapi/glapi/glapi.h"
#include "dispatch.h"
#include "main/context.h"
#include "main/debug_output.h"
#include "main/extensions.h"
#include "main/glformats.h"
#include "main/hash.h"
#include "main/mtypes.h"
#include "main/texobj.h"
#include "main/texstorage.h"
#include "state_tracker/st_cb_eglimage.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "util/detect_os.h"
#include "util/format/u_format.h"
#include "util/hash_table.h"
#include "util/log.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "util/u_process.h"
#define XXH_INLINE_ALL
#include "util/xxhash.h"

static simple_mtx_t apitrace_mutex = SIMPLE_MTX_INITIALIZER;
static int apitrace_pid;
static bool apitrace_enabled;

static bool apitrace_image_dedup;
static simple_mtx_t apitrace_image_mutex = SIMPLE_MTX_INITIALIZER;
static struct hash_table_u64 *apitrace_images;

static bool
apitrace_process_is_zygote(void)
{
#if DETECT_OS_ANDROID
   char name[32];
   if (!util_get_command_line(name, sizeof(name)))
      return false;
   return !strncmp(name, "zygote", 6) || !strncmp(name, "webview_zygote", 14) ||
          !strncmp(name, "usap", 4);
#else
   return false;
#endif
}

bool
_mesa_apitrace_enabled(void)
{
   int pid = getpid();
   bool enabled;

   simple_mtx_lock(&apitrace_mutex);
   if (apitrace_pid != pid) {
      apitrace_pid = pid;
      apitrace_enabled = debug_get_bool_option("MESA_APITRACE", false) &&
                         !apitrace_process_is_zygote();
      if (apitrace_enabled) {
         /* the writer keeps the copy */
         const char *option = os_get_option("MESA_GL_TRACE_FILE");
         char *file = option ? strdup(option) : NULL;
         bool signals = debug_get_bool_option("MESA_APITRACE_SIGNALS", true);
         unsigned flush_ms =
            debug_get_num_option("MESA_APITRACE_FLUSH_MS",
                                 DETECT_OS_ANDROID ? 100 : 0);

         apitrace_image_dedup =
            debug_get_bool_option("MESA_APITRACE_IMAGE_DEDUP", true);
         simple_mtx_lock(&apitrace_image_mutex);
         if (apitrace_images)
            _mesa_hash_table_u64_clear(apitrace_images);
         simple_mtx_unlock(&apitrace_image_mutex);
         bool present = debug_get_bool_option("MESA_APITRACE_PRESENT", true);

         apitrace_enabled =
            _mesa_apitrace_configure(signals, file, flush_ms, present);
         if (!apitrace_enabled) {
            mesa_loge("apitrace: could not open a trace (MESA_GL_TRACE_FILE=%s)",
                      file ? file : "");
         }
      }
   }
   enabled = apitrace_enabled;
   simple_mtx_unlock(&apitrace_mutex);

   return enabled;
}

/* the EGLImage size, for an upload without contents */
bool
_mesa_apitrace_image_size(void *image, int *width, int *height)
{
   GET_CURRENT_CONTEXT(ctx);
   unsigned w, h;

   if (!ctx || !st_get_egl_image_info(ctx, image, &w, &h, NULL))
      return false;

   *width = w;
   *height = h;
   return true;
}

/* the current context, once the calls that glthread queued have run */
static struct gl_context *
apitrace_finished_context(void)
{
   GET_CURRENT_CONTEXT(ctx);

   if (ctx)
      _mesa_glthread_finish(ctx);
   return ctx;
}

/* the formats that EGLImage captures upload as they are */
struct apitrace_read_format {
   enum pipe_format format;
   GLenum internalformat;
};

static const struct apitrace_read_format apitrace_read_formats[] = {
   {PIPE_FORMAT_R8_UNORM, GL_R8},
   {PIPE_FORMAT_R8G8_UNORM, GL_RG8},
   {PIPE_FORMAT_R8G8B8_UNORM, GL_RGB8},
   {PIPE_FORMAT_R8G8B8A8_UNORM, GL_RGBA8},
   {PIPE_FORMAT_R8G8B8_SRGB, GL_SRGB8},
   {PIPE_FORMAT_R8G8B8A8_SRGB, GL_SRGB8_ALPHA8},
   {PIPE_FORMAT_R8_SNORM, GL_R8_SNORM},
   {PIPE_FORMAT_R8G8_SNORM, GL_RG8_SNORM},
   {PIPE_FORMAT_R8G8B8A8_SNORM, GL_RGBA8_SNORM},
   {PIPE_FORMAT_R16_UNORM, GL_R16},
   {PIPE_FORMAT_R16G16_UNORM, GL_RG16},
   {PIPE_FORMAT_R16G16B16A16_UNORM, GL_RGBA16},
   {PIPE_FORMAT_R16_SNORM, GL_R16_SNORM},
   {PIPE_FORMAT_R16G16_SNORM, GL_RG16_SNORM},
   {PIPE_FORMAT_R16G16B16A16_SNORM, GL_RGBA16_SNORM},
   {PIPE_FORMAT_R16_FLOAT, GL_R16F},
   {PIPE_FORMAT_R16G16_FLOAT, GL_RG16F},
   {PIPE_FORMAT_R16G16B16A16_FLOAT, GL_RGBA16F},
   {PIPE_FORMAT_R32_FLOAT, GL_R32F},
   {PIPE_FORMAT_R32G32_FLOAT, GL_RG32F},
   {PIPE_FORMAT_R32G32B32A32_FLOAT, GL_RGBA32F},
   {PIPE_FORMAT_R8_UINT, GL_R8UI},
   {PIPE_FORMAT_R8G8_UINT, GL_RG8UI},
   {PIPE_FORMAT_R8G8B8A8_UINT, GL_RGBA8UI},
   {PIPE_FORMAT_R8_SINT, GL_R8I},
   {PIPE_FORMAT_R8G8_SINT, GL_RG8I},
   {PIPE_FORMAT_R8G8B8A8_SINT, GL_RGBA8I},
   {PIPE_FORMAT_R16_UINT, GL_R16UI},
   {PIPE_FORMAT_R16G16_UINT, GL_RG16UI},
   {PIPE_FORMAT_R16G16B16A16_UINT, GL_RGBA16UI},
   {PIPE_FORMAT_R16_SINT, GL_R16I},
   {PIPE_FORMAT_R16G16_SINT, GL_RG16I},
   {PIPE_FORMAT_R16G16B16A16_SINT, GL_RGBA16I},
   {PIPE_FORMAT_R32_UINT, GL_R32UI},
   {PIPE_FORMAT_R32G32_UINT, GL_RG32UI},
   {PIPE_FORMAT_R32G32B32A32_UINT, GL_RGBA32UI},
   {PIPE_FORMAT_R32_SINT, GL_R32I},
   {PIPE_FORMAT_R32G32_SINT, GL_RG32I},
   {PIPE_FORMAT_R32G32B32A32_SINT, GL_RGBA32I},
   {PIPE_FORMAT_R10G10B10A2_UNORM, GL_RGB10_A2},
   {PIPE_FORMAT_R10G10B10A2_UINT, GL_RGB10_A2UI},
   {PIPE_FORMAT_R11G11B10_FLOAT, GL_R11F_G11F_B10F},
   {PIPE_FORMAT_R9G9B9E5_FLOAT, GL_RGB9_E5},
   {PIPE_FORMAT_B5G6R5_UNORM, GL_RGB565},
};

/*
 * whether plain format b gives the components of a from channels of the same
 * kind, or holds a's constant 0 or 1, so that a converts to b without loss
 */
static bool
apitrace_same_components(enum pipe_format a, enum pipe_format b)
{
   const struct util_format_description *da = util_format_description(a);
   const struct util_format_description *db = util_format_description(b);

   if (da->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       db->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       da->colorspace != db->colorspace)
      return false;
   for (unsigned i = 0; i < 4; i++) {
      unsigned sa = da->swizzle[i], sb = db->swizzle[i];
      if (sb > PIPE_SWIZZLE_W) {
         if (sa != sb)
            return false;
         continue;
      }
      if (sa > PIPE_SWIZZLE_W)
         continue;
      const struct util_format_channel_description *ca = &da->channel[sa];
      const struct util_format_channel_description *cb = &db->channel[sb];
      if (ca->type != cb->type || ca->normalized != cb->normalized ||
          ca->pure_integer != cb->pure_integer || ca->size != cb->size)
         return false;
   }
   return true;
}

static const struct apitrace_read_format *
apitrace_find_read_format(enum pipe_format format)
{
   for (unsigned i = 0; i < ARRAY_SIZE(apitrace_read_formats); i++) {
      if (apitrace_read_formats[i].format == format)
         return &apitrace_read_formats[i];
   }
   for (unsigned i = 0; i < ARRAY_SIZE(apitrace_read_formats); i++) {
      if (apitrace_same_components(format, apitrace_read_formats[i].format))
         return &apitrace_read_formats[i];
   }
   return NULL;
}

/*
 * read an EGLImage into a format that the current context uploads it in
 * exactly, through the driver, so the app's GL state stays as it is; YUV
 * images convert to RGBA8 as sampling them does, and GLES 2, which has no
 * sized formats, takes RGBA8 as GL_RGBA
 */
bool
_mesa_apitrace_image_read(void *image, struct _mesa_apitrace_image_read *read)
{
   struct gl_context *ctx = apitrace_finished_context();
   unsigned width, height;
   enum pipe_format format;

   if (!ctx || !st_get_egl_image_info(ctx, image, &width, &height, &format))
      return false;

   bool gles2 = _mesa_is_gles(ctx) && ctx->Version < 30;
   const struct apitrace_read_format *row =
      apitrace_find_read_format(gles2 || util_format_is_yuv(format) ?
                                PIPE_FORMAT_R8G8B8A8_UNORM : format);
   if (!row)
      return false;

   GLenum internalformat = gles2 ? GL_RGBA : row->internalformat;
   GLenum gl_format = _mesa_get_format_base_format(row->format);
   if (_mesa_is_format_integer(row->format))
      gl_format = _mesa_base_format_to_integer_format(gl_format);
   GLenum type = _mesa_uncompressed_format_to_type(row->format);
   GLenum error = _mesa_is_gles(ctx) ?
      _mesa_gles_error_check_format_and_type(ctx, gl_format, type,
                                             internalformat) :
      _mesa_error_check_format_and_type(ctx, gl_format, type);
   if (error != GL_NO_ERROR || _mesa_base_tex_format(ctx, internalformat) < 0)
      return false;

   unsigned stride = util_format_get_stride(row->format, width);
   size_t size = util_format_get_2d_size(row->format, stride, height);
   void *pixels = malloc(size);
   if (!pixels ||
       !st_read_egl_image(ctx, image, row->format, pixels, stride)) {
      free(pixels);
      return false;
   }

   read->width = width;
   read->height = height;
   read->internalformat = internalformat;
   read->format = gl_format;
   read->type = type;
   read->pixels = pixels;
   read->size = size;
   return true;
}

/* the target of a texture, which compatibility contexts can't query */
unsigned
_mesa_apitrace_texture_target(unsigned texture)
{
   struct gl_context *ctx = apitrace_finished_context();
   struct gl_texture_object *obj =
      ctx ? _mesa_lookup_texture(ctx, texture) : NULL;

   return obj ? obj->Target : GL_NONE;
}

/* the texture bound to target, as GLES fakes of DSA calls bind theirs */
unsigned
_mesa_apitrace_bound_texture(unsigned target)
{
   struct gl_context *ctx = apitrace_finished_context();
   struct gl_texture_object *obj =
      ctx ? _mesa_get_current_tex_object(ctx, target) : NULL;

   return obj ? obj->Name : 0;
}

/*
 * the storage that an EGLImage gave texture, or the texture bound to target
 * if texture is 0, and the format that uploads a read of format and type to it
 */
bool
_mesa_apitrace_image_storage(unsigned target, unsigned texture,
                             unsigned read_format, unsigned type,
                             struct _mesa_apitrace_image_storage *storage)
{
   struct gl_context *ctx = apitrace_finished_context();
   struct gl_texture_object *obj = NULL;

   if (ctx)
      obj = texture ? _mesa_lookup_texture(ctx, texture) :
                      _mesa_get_current_tex_object(ctx, target);
   if (!obj || !obj->Image[0][0])
      return false;

   const struct gl_texture_image *image = obj->Image[0][0];
   GLenum internalformat = image->InternalFormat;
   GLenum base = _mesa_base_tex_format(ctx, internalformat);
   bool color = base == GL_RED || base == GL_RG || base == GL_RGB ||
                base == GL_RGBA;
   bool depth_stencil = base == GL_DEPTH_COMPONENT || base == GL_STENCIL_INDEX ||
                        base == GL_DEPTH_STENCIL;
   GLenum format = _mesa_is_enum_format_integer(internalformat) ?
                   _mesa_base_format_to_integer_format(base) : base;
   GLenum error = _mesa_is_gles(ctx) ?
      _mesa_gles_error_check_format_and_type(ctx, format, type,
                                             internalformat) :
      _mesa_error_check_format_and_type(ctx, format, type);

   storage->target = obj->Target;
   storage->width = image->Width;
   storage->height = image->Height;
   storage->depth = image->Depth;
   storage->base = base;
   storage->color = color;
   /*
    * images whose creator gave no format only have a base format, and the
    * DRI formats include desktop ones (GL_RGB10) that the API may lack;
    * depth and stencil images keep theirs, though a read can't upload to
    * them
    */
   storage->internalformat =
      (color || depth_stencil) && internalformat != base &&
      _mesa_is_legal_tex_storage_format(ctx, internalformat) ?
      internalformat : GL_NONE;
   storage->format = color && storage->internalformat && error == GL_NO_ERROR ?
                     format : GL_NONE;
   storage->texel = _mesa_bytes_per_pixel(format, type);
   storage->read_texel = _mesa_bytes_per_pixel(read_format, type);
   return true;
}

/*
 * the draw framebuffer binding as glGetIntegerv returns it, without waiting
 * for glthread
 */
unsigned
_mesa_apitrace_draw_framebuffer(void)
{
   GET_CURRENT_CONTEXT(ctx);

   if (!ctx)
      return 0;
   return ctx->GLThread.enabled ? ctx->GLThread.CurrentDrawFramebuffer :
                                  ctx->DrawBuffer->Name;
}

/*
 * the walk over the share group can see the flag before the images that
 * another context is still storing
 */
static bool
apitrace_is_present_texture(const struct gl_texture_object *obj)
{
   return obj->FromMemoryObject && obj->Image[0][0] &&
          (obj->Target == GL_TEXTURE_2D || obj->Target == GL_TEXTURE_2D_ARRAY);
}

/* of level 0, the size of the storage, which is where presents shift from */
static int64_t
apitrace_texture_area(const struct gl_texture_object *obj)
{
   const struct gl_texture_image *image = obj->Image[0][0];

   return (int64_t)image->Width * image->Height;
}

static void
apitrace_max_area(void *data, void *user_data)
{
   const struct gl_texture_object *obj = data;
   int64_t *area = user_data;

   if (apitrace_is_present_texture(obj))
      *area = MAX2(*area, apitrace_texture_area(obj));
}

/*
 * what a synthetic present needs from the context: the texture it would
 * show, which is the draw framebuffer's first color attachment if that is
 * the largest texture of the share group with memory object storage, the
 * framebuffers to bind back, and the state that its blit turns off;
 * GL_FRAMEBUFFER_SRGB only where the context can set it, though GLES
 * without EXT_sRGB_write_control can't query it
 */
bool
_mesa_apitrace_get_present_source(struct _mesa_apitrace_present_source *src)
{
   struct gl_context *ctx = apitrace_finished_context();

   if (!ctx)
      return false;

   const struct gl_renderbuffer_attachment *att =
      &ctx->DrawBuffer->Attachment[BUFFER_COLOR0];
   struct gl_texture_object *obj =
      att->Type == GL_TEXTURE ? att->Texture : NULL;
   /* not a texture that was deleted while its framebuffer was unbound */
   if (!obj || !apitrace_is_present_texture(obj) ||
       _mesa_lookup_texture(ctx, obj->Name) != obj)
      return false;

   src->multisampled_window = ctx->WinSysDrawBuffer &&
                              ctx->WinSysDrawBuffer->Visual.samples > 0;
   src->gles = _mesa_is_gles(ctx);
   src->texture = obj->Name;
   /* without the walk over the share group, as these presents are skipped */
   if (src->multisampled_window && src->gles)
      return true;

   int64_t area = 0;
   _mesa_HashWalk(&ctx->Shared->TexObjects, apitrace_max_area, &area);
   if (apitrace_texture_area(obj) < area)
      return false;

   const struct gl_texture_image *image = obj->Image[0][0];
   src->draw = ctx->DrawBuffer->Name;
   src->read = ctx->ReadBuffer->Name;
   src->target = obj->Target;
   src->width = image->Width;
   src->height = image->Height;
   src->depth = image->Depth;
   src->level = att->TextureLevel;
   src->layer = att->Zoffset;
   src->views = att->Layered ? 0 : MAX2(att->NumViews, 1);
   src->scissor = ctx->Scissor.EnableFlags & 1;
   src->scissor_indexed = _mesa_has_ARB_viewport_array(ctx) ||
                          _mesa_has_OES_viewport_array(ctx);
   src->scissor_indexed_ext = _mesa_is_gles(ctx) && ctx->Version < 32;
   src->srgb = ctx->Color.sRGBEnabled &&
               (_mesa_has_EXT_framebuffer_sRGB(ctx) ||
                _mesa_has_EXT_sRGB_write_control(ctx));
   return true;
}

/*
 * turn off GL_DEBUG_OUTPUT, so that errors of the tracer's own calls don't
 * reach the app's debug callback, and return whether it was on; the app's
 * calls that glthread queued run first, with its setting
 */
bool
_mesa_apitrace_debug_output_off(void)
{
   struct gl_context *ctx = apitrace_finished_context();

   if (!ctx || !_mesa_get_debug_state_int(ctx, GL_DEBUG_OUTPUT))
      return false;

   _mesa_set_debug_state_int(ctx, GL_DEBUG_OUTPUT, GL_FALSE);
   return true;
}

/* turn GL_DEBUG_OUTPUT back on once the tracer's queued calls have run */
void
_mesa_apitrace_debug_output_on(void)
{
   struct gl_context *ctx = apitrace_finished_context();

   if (ctx)
      _mesa_set_debug_state_int(ctx, GL_DEBUG_OUTPUT, GL_TRUE);
}

/*
 * whether the image changed since its last upload; the image lock is held
 * until _mesa_apitrace_image_end, so the upload that this decides is recorded
 * before any other thread can rely on it
 */
bool
_mesa_apitrace_image_begin(void *image, unsigned width, unsigned height,
                           const void *pixels, size_t size)
{
   const unsigned dims[2] = {width, height};
   /* never 0, so that a stored hash is distinct from a missing entry */
   uintptr_t hash = apitrace_image_dedup ?
      (uintptr_t)XXH64(pixels, size, XXH64(dims, sizeof(dims), 0)) | 1 : 0;

   simple_mtx_lock(&apitrace_image_mutex);
   if (!hash)
      return true;

   if (!apitrace_images)
      apitrace_images = _mesa_hash_table_u64_create(NULL);
   if ((uintptr_t)_mesa_hash_table_u64_search(apitrace_images,
                                              (uintptr_t)image) == hash)
      return false;

   _mesa_hash_table_u64_insert(apitrace_images, (uintptr_t)image,
                               (void *)hash);
   return true;
}

void
_mesa_apitrace_image_end(void)
{
   simple_mtx_unlock(&apitrace_image_mutex);
}

void
_mesa_apitrace_image_forget(void *image)
{
   simple_mtx_lock(&apitrace_image_mutex);
   if (apitrace_images)
      _mesa_hash_table_u64_remove(apitrace_images, (uintptr_t)image);
   simple_mtx_unlock(&apitrace_image_mutex);
}

void
_mesa_apitrace_init_dispatch(struct gl_context *ctx)
{
   _mesa_apitrace_install((_mesa_apitrace_proc *)ctx->Dispatch.Trace,
                          _mesa_glapi_get_dispatch_table_size());
}

static struct _glapi_table *
apitrace_next_table(void)
{
   GET_CURRENT_CONTEXT(ctx);

   if (ctx && ctx->Dispatch.Trace)
      return ctx->Dispatch.RealPublished;
   return _mesa_glapi_get_dispatch();
}

_mesa_apitrace_proc
_mesa_apitrace_next(int slot)
{
   return ((_glapi_proc *)apitrace_next_table())[slot];
}

static thread_local unsigned apitrace_suspended;

void
_mesa_apitrace_suspend(void)
{
   GET_CURRENT_CONTEXT(ctx);

   if (apitrace_suspended++)
      return;

   if (ctx && ctx->Dispatch.Trace &&
       _mesa_glapi_get_dispatch() == ctx->Dispatch.Trace)
      _mesa_glapi_set_dispatch(ctx->Dispatch.RealPublished);
}

void
_mesa_apitrace_resume(void)
{
   GET_CURRENT_CONTEXT(ctx);

   if (--apitrace_suspended)
      return;

   if (ctx && ctx->Dispatch.Trace &&
       _mesa_glapi_get_dispatch() == ctx->Dispatch.RealPublished)
      _mesa_glapi_set_dispatch(ctx->Dispatch.Trace);
}

/*
 * restore an error that the tracer consumed with glGetError; the app already
 * got its debug message, so don't go through _mesa_error
 */
void
_mesa_apitrace_set_error(unsigned error)
{
   struct gl_context *ctx = apitrace_finished_context();

   if (ctx && ctx->ErrorValue == GL_NO_ERROR)
      ctx->ErrorValue = error;
}

int
_mesa_apitrace_proc_offset(const char *name)
{
   return _mesa_glapi_get_proc_offset(name);
}

_mesa_apitrace_proc
_mesa_apitrace_public_stub(const char *name)
{
   return _mesa_glapi_get_proc_address(name);
}

_mesa_apitrace_proc
_mesa_apitrace_current(int slot)
{
   return ((_glapi_proc *)_mesa_glapi_get_dispatch())[slot];
}
