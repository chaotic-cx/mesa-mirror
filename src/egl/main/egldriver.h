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

#ifndef EGLDRIVER_INCLUDED
#define EGLDRIVER_INCLUDED

#include <stdbool.h>
#include <stddef.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglext_angle.h>
#include <EGL/eglmesaext.h>

#include "egltypedefs.h"
#include "eglconfig.h"
#include "egldisplay.h"
#include "eglsurface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Define an inline driver typecast function.
 *
 * Note that this macro defines a function and should not be ended with a
 * semicolon when used.
 */
#define _EGL_DRIVER_TYPECAST(drvtype, egltype, code)                           \
   static inline struct drvtype *drvtype(const egltype *obj)                   \
   {                                                                           \
      return (struct drvtype *)code;                                           \
   }

/**
 * Define the driver typecast functions for struct _egl_display,
 * struct _egl_context, _EGLSurface, and struct _egl_config.
 *
 * Note that this macro defines several functions and should not be ended with
 * a semicolon when used.
 */
#define _EGL_DRIVER_STANDARD_TYPECASTS(drvname)                                \
   /* note that this is not a direct cast */                                   \
   _EGL_DRIVER_TYPECAST(drvname##_display, struct _egl_display, obj->DriverDisplay) \
   _EGL_DRIVER_TYPECAST(drvname##_context, struct _egl_context, obj)           \
   _EGL_DRIVER_TYPECAST(drvname##_surface, _EGLSurface, obj)                   \
   _EGL_DRIVER_TYPECAST(drvname##_config, struct _egl_config, obj)

/**
 * A generic function ptr type
 */
typedef void (*_EGLProc)(void);

struct wl_display;
struct mesa_glinterop_device_info;
struct mesa_glinterop_export_in;
struct mesa_glinterop_export_out;
struct mesa_glinterop_flush_out;
typedef struct __GLsync *GLsync;

#define EGL_UUID_SIZE 16

/**
 * For use in EGL_EXT_device_query_name and EGL_EXT_device_persistent_id
 */
struct egl_device_info {
   uint8_t device_uuid[EGL_UUID_SIZE];
   uint8_t driver_uuid[EGL_UUID_SIZE];

   /* Data below needs to be freed manually */
   char *vendor_name;
   char *renderer_name;
   char *driver_name;

   EGLAttrib device_type;
};

/**
 * The API dispatcher jumps through these functions
 */
extern const struct _egl_driver Driver;
struct _egl_driver {
   /* driver funcs */
   EGLBoolean (*Initialize)(struct _egl_display *disp);
   EGLBoolean (*Terminate)(struct _egl_display *disp);

   /* context funcs */
   struct _egl_context *(*CreateContext)(struct _egl_display *disp, struct _egl_config *config,
                                 struct _egl_context *share_list,
                                 const EGLint *attrib_list);
   EGLBoolean (*DestroyContext)(struct _egl_display *disp, struct _egl_context *ctx);
   /* this is the only function (other than Initialize) that may be called
    * with an uninitialized display
    */
   EGLBoolean (*MakeCurrent)(struct _egl_display *disp, _EGLSurface *draw,
                             _EGLSurface *read, struct _egl_context *ctx);

   /* surface funcs */
   _EGLSurface *(*CreateWindowSurface)(struct _egl_display *disp, struct _egl_config *config,
                                       void *native_window,
                                       const EGLint *attrib_list);
   _EGLSurface *(*CreatePixmapSurface)(struct _egl_display *disp, struct _egl_config *config,
                                       void *native_pixmap,
                                       const EGLint *attrib_list);
   _EGLSurface *(*CreatePbufferSurface)(struct _egl_display *disp, struct _egl_config *config,
                                        const EGLint *attrib_list);
   EGLBoolean (*DestroySurface)(struct _egl_display *disp, _EGLSurface *surface);
   EGLBoolean (*QuerySurface)(struct _egl_display *disp, _EGLSurface *surface,
                              EGLint attribute, EGLint *value);
   EGLBoolean (*BindTexImage)(struct _egl_display *disp, _EGLSurface *surface,
                              EGLint buffer);
   EGLBoolean (*ReleaseTexImage)(struct _egl_display *disp, _EGLSurface *surface,
                                 EGLint buffer);
   EGLBoolean (*SwapInterval)(struct _egl_display *disp, _EGLSurface *surf,
                              EGLint interval);
   EGLBoolean (*SwapBuffers)(struct _egl_display *disp, _EGLSurface *draw);
   EGLBoolean (*CopyBuffers)(struct _egl_display *disp, _EGLSurface *surface,
                             void *native_pixmap_target);

   /* for EGL_KHR_partial_update */
   EGLBoolean (*SetDamageRegion)(struct _egl_display *disp, _EGLSurface *surface,
                                 EGLint *rects, EGLint n_rects);

   /* misc functions */
   EGLBoolean (*WaitClient)(struct _egl_display *disp, struct _egl_context *ctx);
   EGLBoolean (*WaitNative)(EGLint engine);

   /* for EGL_KHR_image_base */
   struct _egl_image *(*CreateImageKHR)(struct _egl_display *disp, struct _egl_context *ctx,
                                EGLenum target, EGLClientBuffer buffer,
                                const EGLint *attr_list);
   EGLBoolean (*DestroyImageKHR)(struct _egl_display *disp, struct _egl_image *image);

   /* for EGL_KHR_reusable_sync/EGL_KHR_fence_sync */
   _EGLSync *(*CreateSyncKHR)(struct _egl_display *disp, EGLenum type,
                              const EGLAttrib *attrib_list);
   EGLBoolean (*DestroySyncKHR)(struct _egl_display *disp, _EGLSync *sync);
   EGLint (*ClientWaitSyncKHR)(struct _egl_display *disp, _EGLSync *sync, EGLint flags,
                               EGLTime timeout);
   EGLint (*WaitSyncKHR)(struct _egl_display *disp, _EGLSync *sync);
   /* for EGL_KHR_reusable_sync */
   EGLBoolean (*SignalSyncKHR)(struct _egl_display *disp, _EGLSync *sync, EGLenum mode);

   /* for EGL_ANDROID_native_fence_sync */
   EGLint (*DupNativeFenceFDANDROID)(struct _egl_display *disp, _EGLSync *sync);

#ifdef HAVE_BIND_WL_DISPLAY
   /* for EGL_WL_bind_wayland_display */
   EGLBoolean (*BindWaylandDisplayWL)(struct _egl_display *disp,
                                      struct wl_display *display);
   EGLBoolean (*UnbindWaylandDisplayWL)(struct _egl_display *disp,
                                        struct wl_display *display);
   EGLBoolean (*QueryWaylandBufferWL)(struct _egl_display *displ,
                                      struct wl_resource *buffer,
                                      EGLint attribute, EGLint *value);

   /* for EGL_WL_create_wayland_buffer_from_image */
   struct wl_buffer *(*CreateWaylandBufferFromImageWL)(struct _egl_display *disp,
                                                       struct _egl_image *img);
#endif

   /* for EGL_EXT_swap_buffers_with_damage */
   EGLBoolean (*SwapBuffersWithDamageEXT)(struct _egl_display *disp,
                                          _EGLSurface *surface,
                                          const EGLint *rects, EGLint n_rects);

   /* for EGL_EXT_buffer_age/EGL_KHR_partial_update */
   EGLint (*QueryBufferAge)(struct _egl_display *disp, _EGLSurface *surface);

   /* for EGL_CHROMIUM_sync_control */
   EGLBoolean (*GetSyncValuesCHROMIUM)(struct _egl_display *disp, _EGLSurface *surface,
                                       EGLuint64KHR *ust, EGLuint64KHR *msc,
                                       EGLuint64KHR *sbc);

   EGLBoolean (*GetMscRateANGLE)(struct _egl_display *disp, _EGLSurface *surface,
                                 EGLint *numerator, EGLint *denominator);

   /* for EGL_MESA_image_dma_buf_export */
   EGLBoolean (*ExportDMABUFImageQueryMESA)(struct _egl_display *disp, struct _egl_image *img,
                                            EGLint *fourcc, EGLint *nplanes,
                                            EGLuint64KHR *modifiers);
   EGLBoolean (*ExportDMABUFImageMESA)(struct _egl_display *disp, struct _egl_image *img,
                                       EGLint *fds, EGLint *strides,
                                       EGLint *offsets);

   /* for EGL_MESA_query_driver */
   const char *(*QueryDriverName)(struct _egl_display *disp);
   char *(*QueryDriverConfig)(struct _egl_display *disp);

   /* for EGL_EXT_device_query_name and EGL_EXT_device_persistent_id */
   bool (*QueryDeviceInfo)(const void* driver_device_identifier,
                           struct egl_device_info *device_info);

   /* for OpenGL-OpenCL interop; see include/GL/mesa_glinterop.h */
   int (*GLInteropQueryDeviceInfo)(struct _egl_display *disp, struct _egl_context *ctx,
                                   struct mesa_glinterop_device_info *out);
   int (*GLInteropExportObject)(struct _egl_display *disp, struct _egl_context *ctx,
                                struct mesa_glinterop_export_in *in,
                                struct mesa_glinterop_export_out *out);
   int (*GLInteropFlushObjects)(struct _egl_display *disp, struct _egl_context *ctx,
                                unsigned count,
                                struct mesa_glinterop_export_in *in,
                                struct mesa_glinterop_flush_out *out);

   /* for EGL_EXT_image_dma_buf_import_modifiers */
   EGLBoolean (*QueryDmaBufFormatsEXT)(struct _egl_display *disp, EGLint max_formats,
                                       EGLint *formats, EGLint *num_formats);
   EGLBoolean (*QueryDmaBufModifiersEXT)(struct _egl_display *disp, EGLint format,
                                         EGLint max_modifiers,
                                         EGLuint64KHR *modifiers,
                                         EGLBoolean *external_only,
                                         EGLint *num_modifiers);

   /* for EGL_ANDROID_blob_cache */
   void (*SetBlobCacheFuncsANDROID)(struct _egl_display *disp,
                                    EGLSetBlobFuncANDROID set,
                                    EGLGetBlobFuncANDROID get);

   /* for EGL_EXT_surface_compression */
   EGLBoolean (*QuerySupportedCompressionRatesEXT)(struct _egl_display *disp,
                                                   struct _egl_config *config,
                                                   const EGLAttrib *attr_list,
                                                   EGLint *rates,
                                                   EGLint rate_size,
                                                   EGLint *num_rates);
};

#ifdef __cplusplus
}
#endif

#endif /* EGLDRIVER_INCLUDED */
