/*
 * SPDX-License-Identifier: MIT
 */

#ifndef MESA_APITRACE_H
#define MESA_APITRACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gl_context;

typedef void (*_mesa_apitrace_proc)(void);

#ifdef HAVE_APITRACE

bool
_mesa_apitrace_enabled(void);

void
_mesa_apitrace_init_dispatch(struct gl_context *ctx);

_mesa_apitrace_proc
_mesa_apitrace_next(int slot);

void
_mesa_apitrace_set_error(unsigned error);

void
_mesa_apitrace_suspend(void);

void
_mesa_apitrace_resume(void);

_mesa_apitrace_proc
_mesa_apitrace_current(int slot);

int
_mesa_apitrace_proc_offset(const char *name);

_mesa_apitrace_proc
_mesa_apitrace_public_stub(const char *name);

void
_mesa_apitrace_install(_mesa_apitrace_proc *table, unsigned size);

void *
_mesa_apitrace_lookup(const char *name);

_mesa_apitrace_proc
_mesa_apitrace_tracer(const char *name);

/*
 * the tracer's own entry points of extensions that it advertises, and debug
 * functions that it provides when Mesa doesn't (missing)
 */
_mesa_apitrace_proc
_mesa_apitrace_tracer_only(const char *name, bool missing);

_mesa_apitrace_proc
_mesa_apitrace_winsys_real(const char *name);

bool
_mesa_apitrace_configure(bool signals, const char *file, unsigned flush_ms,
                         bool present);

void
_mesa_apitrace_fake_gen_framebuffer(unsigned name);

void
_mesa_apitrace_fake_swap(void *dpy, void *surface);

void
_mesa_apitrace_present_bind(unsigned target, unsigned framebuffer);

void
_mesa_apitrace_present_memory_objects(void);

void
_mesa_apitrace_present_signal(unsigned count, const unsigned *textures);

void
_mesa_apitrace_present_real_swap(void);

struct _mesa_apitrace_present_source {
   unsigned draw;
   unsigned read;
   unsigned texture;
   unsigned target;
   int width;
   int height;
   int depth;
   int level;
   int layer;
   /* 0 for all layers */
   int views;
   bool scissor;
   /* whether scissor tests are per viewport, and with the EXT entry points */
   bool scissor_indexed;
   bool scissor_indexed_ext;
   bool srgb;
   /* which blits into the window GL allows */
   bool multisampled_window;
   bool gles;
};

unsigned
_mesa_apitrace_draw_framebuffer(void);

bool
_mesa_apitrace_get_present_source(struct _mesa_apitrace_present_source *src);

bool
_mesa_apitrace_debug_output_off(void);

void
_mesa_apitrace_debug_output_on(void);

unsigned
_mesa_apitrace_texture_target(unsigned texture);

unsigned
_mesa_apitrace_bound_texture(unsigned target);

struct _mesa_apitrace_image_storage {
   unsigned target;
   int width;
   int height;
   int depth;
   unsigned base;
   /* whether the storage has color, unlike depth and stencil */
   bool color;
   /* GL_NONE where Mesa only has the base format */
   unsigned internalformat;
   /* GL_NONE where the read can't upload internalformat */
   unsigned format;
   /* bytes per texel of format and of the read */
   int texel;
   int read_texel;
};

bool
_mesa_apitrace_image_storage(unsigned target, unsigned texture,
                             unsigned read_format, unsigned type,
                             struct _mesa_apitrace_image_storage *storage);

void
_mesa_apitrace_fake_image_storage(unsigned target, unsigned texture,
                                  unsigned internalformat, unsigned format,
                                  unsigned type, int width, int height,
                                  const void *pixels);

bool
_mesa_apitrace_image_size(void *image, int *width, int *height);

/* an EGLImage's contents, tightly packed in a format that keeps them exactly */
struct _mesa_apitrace_image_read {
   int width;
   int height;
   unsigned internalformat;
   unsigned format;
   unsigned type;
   /* malloc'd, size bytes */
   void *pixels;
   size_t size;
};

bool
_mesa_apitrace_image_read(void *image, struct _mesa_apitrace_image_read *read);

bool
_mesa_apitrace_image_begin(void *image, unsigned width, unsigned height,
                           const void *pixels, size_t size);

void
_mesa_apitrace_image_end(void);

void
_mesa_apitrace_image_forget(void *image);

PUBLIC void
_mesa_apitrace_winsys_init(const char *const *names,
                           const _mesa_apitrace_proc *real,
                           _mesa_apitrace_proc *active, unsigned count);

PUBLIC _mesa_apitrace_proc
_mesa_apitrace_winsys_wrap(const char *name, _mesa_apitrace_proc real);

#else

static inline bool
_mesa_apitrace_enabled(void)
{
   return false;
}

static inline void
_mesa_apitrace_init_dispatch(struct gl_context *ctx)
{
}

#endif

#ifdef __cplusplus
}
#endif

#endif
