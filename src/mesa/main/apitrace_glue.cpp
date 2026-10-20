/*
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#ifndef _WIN32
#include <pthread.h>
#endif

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "glproc.hpp"
#include "gltrace.hpp"
#include "trace_writer_local.hpp"

#include "main/apitrace.h"
#include "util/u_atomic.h"

struct winsys_table {
   _mesa_apitrace_proc *active;
   unsigned count;
};

static std::mutex &winsys_mutex = *new std::mutex;
static std::vector<winsys_table> &winsys_tables = *new std::vector<winsys_table>;
static std::map<std::string, _mesa_apitrace_proc> &winsys_real =
   *new std::map<std::string, _mesa_apitrace_proc>;

#ifndef _WIN32
static void
winsys_fork_prepare(void)
{
   winsys_mutex.lock();
}

static void
winsys_fork_parent(void)
{
   winsys_mutex.unlock();
}

static void
winsys_fork_child(void)
{
   /* the child has a single thread */
   for (const winsys_table &table : winsys_tables)
      memset(table.active, 0, table.count * sizeof(*table.active));
   winsys_mutex.unlock();
}
#endif

void *
_getPublicProcAddress(const char *procName)
{
   return _mesa_apitrace_lookup(procName);
}

void *
_getPrivateProcAddress(const char *procName)
{
   return _mesa_apitrace_lookup(procName);
}

static _mesa_apitrace_proc
winsys_find(const char *name)
{
   std::lock_guard<std::mutex> lock(winsys_mutex);
   auto it = winsys_real.find(name);
   return it != winsys_real.end() ? it->second : NULL;
}

_mesa_apitrace_proc
_mesa_apitrace_winsys_real(const char *name)
{
   _mesa_apitrace_proc proc = winsys_find(name);
   if (proc)
      return proc;

   typedef _mesa_apitrace_proc (APIENTRY *get_proc_address)(const char *);
   get_proc_address get = NULL;
   if (!strncmp(name, "egl", 3))
      get = (get_proc_address)winsys_find("eglGetProcAddress");
   else if (!strncmp(name, "glX", 3))
      get = (get_proc_address)winsys_find("glXGetProcAddressARB");

   return get ? get(name) : NULL;
}

void
_mesa_apitrace_winsys_init(const char *const *names,
                           const _mesa_apitrace_proc *real,
                           _mesa_apitrace_proc *active, unsigned count)
{
   bool enabled = _mesa_apitrace_enabled();

   {
      std::lock_guard<std::mutex> lock(winsys_mutex);
#ifndef _WIN32
      static bool atfork_registered = false;
      if (!atfork_registered) {
         atfork_registered = true;
         pthread_atfork(winsys_fork_prepare, winsys_fork_parent,
                        winsys_fork_child);
      }
#endif
      if (std::none_of(winsys_tables.begin(), winsys_tables.end(),
                       [active](const winsys_table &table) {
                          return table.active == active;
                       }))
         winsys_tables.push_back({active, count});

      if (enabled) {
         for (unsigned i = 0; i < count; i++) {
            if (real[i])
               winsys_real[names[i]] = real[i];
         }
      }
   }

   for (unsigned i = 0; i < count; i++) {
      _mesa_apitrace_proc tracer =
         enabled ? _mesa_apitrace_tracer(names[i]) : NULL;
      _mesa_apitrace_proc proc = tracer    ? tracer
                                 : real[i] ? real[i]
                                           : _mesa_apitrace_public_stub(names[i]);
      /* read without a lock by _mesa_apitrace_get */
      p_atomic_set(&active[i], proc);
   }
}

_mesa_apitrace_proc
_mesa_apitrace_winsys_wrap(const char *name, _mesa_apitrace_proc real)
{
   if (!_mesa_apitrace_enabled())
      return real;

   /* GLVND asks for these here, not through the traced GetProcAddress */
   _mesa_apitrace_proc only = _mesa_apitrace_tracer_only(name, !real);
   if (only || !real)
      return only;

   _mesa_apitrace_proc tracer = _mesa_apitrace_tracer(name);
   if (!tracer)
      return real;

   {
      std::lock_guard<std::mutex> lock(winsys_mutex);
      winsys_real[name] = real;
   }
   return tracer;
}

void APIENTRY _fake_glBindFramebuffer(GLenum target, GLuint framebuffer);
void APIENTRY _fake_glBlitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                                      GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
                                      GLbitfield mask, GLenum filter);
void APIENTRY _fake_glDisable(GLenum cap);
void APIENTRY _fake_glDisablei(GLenum target, GLuint index);
void APIENTRY _fake_glDisableiEXT(GLenum target, GLuint index);
void APIENTRY _fake_glEnable(GLenum cap);
void APIENTRY _fake_glEnablei(GLenum target, GLuint index);
void APIENTRY _fake_glEnableiEXT(GLenum target, GLuint index);
void APIENTRY _fake_glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget,
                                           GLuint texture, GLint level);
void APIENTRY _fake_glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture,
                                              GLint level, GLint layer);
void APIENTRY _fake_glTexStorage1D(GLenum target, GLsizei levels, GLenum internalformat,
                                   GLsizei width);
void APIENTRY _fake_glTexStorage2D(GLenum target, GLsizei levels, GLenum internalformat,
                                   GLsizei width, GLsizei height);
void APIENTRY _fake_glTexStorage3D(GLenum target, GLsizei levels, GLenum internalformat,
                                   GLsizei width, GLsizei height, GLsizei depth);
void APIENTRY _fake_glTextureStorage1D(GLuint texture, GLsizei levels, GLenum internalformat,
                                       GLsizei width);
void APIENTRY _fake_glTextureStorage2D(GLuint texture, GLsizei levels, GLenum internalformat,
                                       GLsizei width, GLsizei height);
void APIENTRY _fake_glTextureStorage3D(GLuint texture, GLsizei levels, GLenum internalformat,
                                       GLsizei width, GLsizei height, GLsizei depth);
void APIENTRY _fake_glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset,
                                    GLsizei width, GLsizei height, GLenum format, GLenum type,
                                    const void *pixels);
void APIENTRY _fake_glTextureSubImage2D(GLuint texture, GLint level, GLint xoffset,
                                        GLint yoffset, GLsizei width, GLsizei height,
                                        GLenum format, GLenum type, const void *pixels);

/*
 * The storage that an EGLImage gave a texture, in the format that Mesa gave it
 * where the captured read can fill it, and the read, if pixels has one,
 * which only GL_TEXTURE_2D images have. With texture, the faked calls are
 * those of glEGLImageTargetTextureStorageEXT, and otherwise those of target.
 */
void
_mesa_apitrace_fake_image_storage(unsigned target, unsigned texture,
                                  unsigned internalformat, unsigned format,
                                  unsigned type, int width, int height,
                                  const void *pixels)
{
   struct _mesa_apitrace_image_storage storage;
   if (!_mesa_apitrace_image_storage(target, texture, format,
                                     pixels ? type : GL_NONE, &storage))
      return;

   /* nothing of a read of another kind of image fits depth or stencil storage */
   if (!storage.color)
      pixels = NULL;

   std::vector<uint8_t> narrow;
   if (pixels && storage.format) {
      /* keep the components of the read that the format has */
      size_t texel = storage.texel, read_texel = storage.read_texel;
      if (texel < read_texel) {
         narrow.resize((size_t)width * height * texel);
         for (size_t i = 0; i < (size_t)width * height; i++)
            memcpy(&narrow[i * texel], (const uint8_t *)pixels + i * read_texel, texel);
         pixels = narrow.data();
      }
      internalformat = storage.internalformat;
      format = storage.format;
   } else if (!pixels) {
      internalformat = storage.internalformat ? storage.internalformat :
                       storage.base == GL_RGB ? GL_RGB8 : GL_RGBA8;
   } else if (internalformat == GL_RGBA) {
      /* reads in GLES 2 contexts have no sized format */
      internalformat = GL_RGBA8;
   }

   GLsizei depth = storage.depth;
   switch (storage.target) {
   case GL_TEXTURE_1D:
      if (texture)
         _fake_glTextureStorage1D(texture, 1, internalformat, storage.width);
      else
         _fake_glTexStorage1D(target, 1, internalformat, storage.width);
      break;
   case GL_TEXTURE_CUBE_MAP_ARRAY:
      /* Mesa only sets up the first face, and the storage needs a cube */
      depth = 6;
      [[fallthrough]];
   case GL_TEXTURE_2D_ARRAY:
   case GL_TEXTURE_3D:
      if (texture)
         _fake_glTextureStorage3D(texture, 1, internalformat, storage.width,
                                  storage.height, depth);
      else
         _fake_glTexStorage3D(target, 1, internalformat, storage.width,
                              storage.height, depth);
      break;
   default:
      if (texture)
         _fake_glTextureStorage2D(texture, 1, internalformat, storage.width,
                                  storage.height);
      else
         _fake_glTexStorage2D(target, 1, internalformat, storage.width,
                              storage.height);
      break;
   }

   if (!pixels)
      return;
   if (texture)
      _fake_glTextureSubImage2D(texture, 0, 0, 0, width, height, format, type,
                                pixels);
   else
      _fake_glTexSubImage2D(target, 0, 0, 0, width, height, format, type,
                            pixels);
}

/* per gltrace context, freed with it */
struct present_context {
   GLuint helper = 0;
   /* since the last semaphore signal, which then needs no present of its own */
   bool presented = false;
};

static std::atomic<bool> present_enabled;
/* set by the first 2D or 3D texture storage in a memory object, the only
 * textures that presents show */
static std::atomic<bool> present_memory_objects;

static present_context &
present_current(void)
{
   std::shared_ptr<void> &slot = gltrace::getContext()->userData;
   if (!slot)
      slot = std::make_shared<present_context>();
   return *static_cast<present_context *>(slot.get());
}

/* the EGL display and draw surface that a present goes to */
static bool
present_surface(void **dpy, void **draw)
{
   typedef void *(*get_display)(void);
   typedef void *(*get_surface)(EGLint);
   get_display display =
      (get_display)_mesa_apitrace_winsys_real("eglGetCurrentDisplay");
   get_surface surface =
      (get_surface)_mesa_apitrace_winsys_real("eglGetCurrentSurface");
   if (!display || !surface)
      return false;

   *dpy = display();
   *draw = surface(EGL_DRAW);
   return *dpy && *draw;
}

/* the scissor test of viewport 0, which the blit uses, and no other */
static void
present_scissor(const struct _mesa_apitrace_present_source &src, bool enable)
{
   if (!src.scissor_indexed) {
      if (enable)
         _fake_glEnable(GL_SCISSOR_TEST);
      else
         _fake_glDisable(GL_SCISSOR_TEST);
   } else if (src.scissor_indexed_ext) {
      if (enable)
         _fake_glEnableiEXT(GL_SCISSOR_TEST, 0);
      else
         _fake_glDisableiEXT(GL_SCISSOR_TEST, 0);
   } else if (enable) {
      _fake_glEnablei(GL_SCISSOR_TEST, 0);
   } else {
      _fake_glDisablei(GL_SCISSOR_TEST, 0);
   }
}

/* blit and swap so replays show apps that only render to shared textures */
static void
present_emit(present_context &ctx,
             const struct _mesa_apitrace_present_source &src, void *dpy,
             void *draw)
{
   if (src.multisampled_window && src.gles) {
      static std::atomic<bool> warned;
      if (!warned.exchange(true))
         os::log("apitrace: warning: GLES can't blit into multisampled windows, "
                 "so the trace has no presents\n");
      return;
   }

   if (!ctx.helper) {
      _glGenFramebuffers(1, &ctx.helper);
      if (!ctx.helper)
         return;
      _mesa_apitrace_fake_gen_framebuffer(ctx.helper);
   }

   if (src.scissor)
      present_scissor(src, false);

   /* keep sRGB values encoded, as a compositor shows them */
   if (src.srgb)
      _fake_glDisable(GL_FRAMEBUFFER_SRGB);

   _fake_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
   _fake_glBindFramebuffer(GL_READ_FRAMEBUFFER, ctx.helper);

   bool array = src.target == GL_TEXTURE_2D_ARRAY;
   GLsizei views = 1;
   /* blits into multisampled windows can't scale the views side by side */
   if (array && !src.multisampled_window) {
      GLsizei layers = src.views ? src.views : src.depth - src.layer;
      views = std::max(std::min(layers, 2), 1);
   }
   GLsizei src_width = std::max(src.width >> src.level, 1);
   GLsizei src_height = std::max(src.height >> src.level, 1);
   GLsizei width = views > 1 ? src_width / 2 : src_width;
   GLsizei height = views > 1 ? src_height / 2 : src_height;
   for (GLsizei view = 0; view < views; view++) {
      if (array) {
         _fake_glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                         src.texture, src.level,
                                         src.layer + view);
      } else {
         _fake_glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, src.texture, src.level);
      }
      _fake_glBlitFramebuffer(0, 0, src_width, src_height,
                              view * width, 0, (view + 1) * width, height,
                              GL_COLOR_BUFFER_BIT, GL_LINEAR);
   }

   _fake_glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
   _mesa_apitrace_fake_swap(dpy, draw);
   ctx.presented = true;

   _fake_glBindFramebuffer(GL_READ_FRAMEBUFFER, src.read);
   _fake_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, src.draw);
   if (src.srgb)
      _fake_glEnable(GL_FRAMEBUFFER_SRGB);
   if (src.scissor)
      present_scissor(src, true);
}

static void
present_draw_framebuffer(present_context &ctx)
{
   void *dpy, *draw;
   struct _mesa_apitrace_present_source src;
   if (present_surface(&dpy, &draw) &&
       _mesa_apitrace_get_present_source(&src))
      present_emit(ctx, src, dpy, draw);
}

void
_mesa_apitrace_present_bind(unsigned target, unsigned framebuffer)
{
   if (!present_enabled || !present_memory_objects)
      return;

   if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER)
      return;

   /* the checks that need no glthread sync go first */
   GLuint current = _mesa_apitrace_draw_framebuffer();
   if (!current || current == framebuffer)
      return;

   present_draw_framebuffer(present_current());
}

/*
 * for apps that keep their framebuffer bound and mark frames with signals,
 * of the texture that the present shows if they list textures, so that
 * signals of other textures in the same frame don't present again
 */
void
_mesa_apitrace_present_signal(unsigned count, const unsigned *textures)
{
   if (!present_enabled || !present_memory_objects ||
       !_mesa_apitrace_draw_framebuffer())
      return;

   present_context &ctx = present_current();
   void *dpy, *draw;
   struct _mesa_apitrace_present_source src;
   if (!ctx.presented && present_surface(&dpy, &draw) &&
       _mesa_apitrace_get_present_source(&src) &&
       (!count || std::find(textures, textures + count, src.texture) !=
                  textures + count))
      present_emit(ctx, src, dpy, draw);
   ctx.presented = false;
}

void
_mesa_apitrace_present_memory_objects(void)
{
   present_memory_objects = true;
}

void
_mesa_apitrace_present_real_swap(void)
{
   present_enabled = false;
}

bool
_mesa_apitrace_configure(bool signals, const char *file, unsigned flush_ms,
                         bool present)
{
   present_enabled = present;
   trace::localWriterExceptionCallback = signals;
   trace::localWriterFileName = file;
   trace::localWriterFlushIntervalMs = flush_ms;

   return trace::localWriter.start();
}
