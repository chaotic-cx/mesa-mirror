/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * Copyright 2009-2010 Chia-I Wu <olvaffe@gmail.com>
 * Copyright 2010-2011 LunarG, Inc.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef EGLCONTEXT_INCLUDED
#define EGLCONTEXT_INCLUDED

#include <EGL/egl.h>

#include "egldisplay.h"
#include "egltypedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * "Base" class for device driver contexts.
 */
struct _egl_context {
   /* A context is a display resource */
   struct _egl_resource Resource;

   /* The bound status of the context */
   _EGLThreadInfo *Binding;
   _EGLSurface *DrawSurface;
   _EGLSurface *ReadSurface;

   struct _egl_config *Config;

   EGLint ClientAPI; /**< EGL_OPENGL_ES_API, EGL_OPENGL_API, EGL_OPENVG_API */
   EGLint ClientMajorVersion;
   EGLint ClientMinorVersion;
   EGLint Flags;
   EGLint Profile;
   EGLint ResetNotificationStrategy;
   EGLint ContextPriority;
   EGLBoolean NoError;
   EGLint ReleaseBehavior;
   EGLBoolean Protected; /* EGL_EXT_protected_content */
};

extern EGLBoolean
_eglInitContext(struct _egl_context *ctx, struct _egl_display *disp, struct _egl_config *config,
                struct _egl_context *share_list, const EGLint *attrib_list);

extern EGLBoolean
_eglQueryContext(struct _egl_context *ctx, EGLint attribute, EGLint *value);

extern EGLBoolean
_eglBindContext(struct _egl_context *ctx, _EGLSurface *draw, _EGLSurface *read,
                struct _egl_context **old_ctx, _EGLSurface **old_draw,
                _EGLSurface **old_read);

extern struct _egl_context *
_eglBindContextToThread(struct _egl_context *ctx, _EGLThreadInfo *t);

/**
 * Increment reference count for the context.
 */
static inline struct _egl_context *
_eglGetContext(struct _egl_context *ctx)
{
   if (ctx)
      _eglGetResource(&ctx->Resource);
   return ctx;
}

/**
 * Decrement reference count for the context.
 */
static inline EGLBoolean
_eglPutContext(struct _egl_context *ctx)
{
   return (ctx) ? _eglPutResource(&ctx->Resource) : EGL_FALSE;
}

/**
 * Link a context to its display and return the handle of the link.
 * The handle can be passed to client directly.
 */
static inline EGLContext
_eglLinkContext(struct _egl_context *ctx)
{
   _eglLinkResource(&ctx->Resource, _EGL_RESOURCE_CONTEXT);
   return (EGLContext)ctx;
}

/**
 * Unlink a linked context from its display.
 * Accessing an unlinked context should generate EGL_BAD_CONTEXT error.
 */
static inline void
_eglUnlinkContext(struct _egl_context *ctx)
{
   _eglUnlinkResource(&ctx->Resource, _EGL_RESOURCE_CONTEXT);
}

/**
 * Lookup a handle to find the linked context.
 * Return NULL if the handle has no corresponding linked context.
 */
static inline struct _egl_context *
_eglLookupContext(EGLContext context, struct _egl_display *disp)
{
   struct _egl_context *ctx = (struct _egl_context *)context;
   if (!disp || !_eglCheckResource((void *)ctx, _EGL_RESOURCE_CONTEXT, disp))
      ctx = NULL;
   return ctx;
}

/**
 * Return the handle of a linked context, or EGL_NO_CONTEXT.
 */
static inline EGLContext
_eglGetContextHandle(struct _egl_context *ctx)
{
   struct _egl_resource *res = (struct _egl_resource *)ctx;
   return (res && _eglIsResourceLinked(res)) ? (EGLContext)ctx : EGL_NO_CONTEXT;
}

#ifdef __cplusplus
}
#endif

#endif /* EGLCONTEXT_INCLUDED */
