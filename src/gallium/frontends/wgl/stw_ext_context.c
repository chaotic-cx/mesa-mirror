/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2011 Morgan Armand <morgan.devel@gmail.com>
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <windows.h>

#define WGL_WGLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/wglext.h>

#include "stw_gdishim.h"
#include "gldrv.h"
#include "stw_context.h"
#include "stw_device.h"
#include "stw_ext_context.h"
#include "stw_apitrace.h"

#include "util/u_debug.h"


static wglCreateContext_t g_pfnwglCreateContext = NULL;
static wglDeleteContext_t g_pfnwglDeleteContext = NULL;

/* Optional hook for external opengl32.dll drop-in wrappers that want
 * wglCreateContextAttribsARB() to allocate/free public HGLRC handles through
 * custom wglCreateContext/wglDeleteContext entry points.
 *
 * When neither ICD callbacks (pfnGetDhglrc) nor these overrides are set,
 * public HGLRC handles are identical to driver DHGLRC handles, so
 * wglCreateContextAttribsARB() creates the context and handle directly.
 */
void
stw_override_opengl32_entry_points(wglCreateContext_t create, wglDeleteContext_t delete)
{
   g_pfnwglCreateContext = create;
   g_pfnwglDeleteContext = delete;
}


/**
 * When used as a Windows ICD, OPENGL32.DLL remaps the driver handles (DHGLRC)
 * returned by DrvCreateContext() to public HGLRC handles.  In that mode we
 * call OPENGL32.DLL's wglCreateContext() to allocate a public HGLRC ID and
 * then replace its stw_context with a new one reflecting the requested
 * attributes.
 *
 * When used as a drop-in opengl32.dll (without ICD callbacks or an explicit
 * entry-point override), HGLRC and DHGLRC are identical, so we create the
 * stw_context and handle directly without round-tripping through a throwaway
 * legacy wglCreateContext() call.
 *
 * The share context and the result are driver handles, see stw_apitrace.h.
 */
HGLRC WINAPI
stw_wglCreateContextAttribsARB(HDC hDC, HGLRC hShareContext, const int *attribList)
{
   HGLRC context = NULL;

   int majorVersion = 1, minorVersion = 0, layerPlane = 0;
   int contextFlags = 0x0;
   int profileMask = WGL_CONTEXT_CORE_PROFILE_BIT_ARB;
   int resetStrategy = WGL_NO_RESET_NOTIFICATION_ARB;
   int i;
   BOOL done = false;
   const int contextFlagsAll = (WGL_CONTEXT_DEBUG_BIT_ARB |
                                WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB |
                                WGL_CONTEXT_ROBUST_ACCESS_BIT_ARB);

   if (!stw_dev)
      return NULL;

   /* parse attrib_list */
   if (attribList) {
      for (i = 0; !done && attribList[i]; i++) {
         switch (attribList[i]) {
         case WGL_CONTEXT_MAJOR_VERSION_ARB:
            majorVersion = attribList[++i];
            break;
         case WGL_CONTEXT_MINOR_VERSION_ARB:
            minorVersion = attribList[++i];
            break;
         case WGL_CONTEXT_LAYER_PLANE_ARB:
            layerPlane = attribList[++i];
            break;
         case WGL_CONTEXT_FLAGS_ARB:
            contextFlags = attribList[++i];
            break;
         case WGL_CONTEXT_PROFILE_MASK_ARB:
            profileMask = attribList[++i];
            break;
         case WGL_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB:
            resetStrategy = attribList[++i];
            break;
         case 0:
            /* end of list */
            done = true;
            break;
         default:
            /* bad attribute */
            SetLastError(ERROR_INVALID_PARAMETER);
            return 0;
         }
      }
   }

   /* check contextFlags */
   if (contextFlags & ~contextFlagsAll) {
      SetLastError(ERROR_INVALID_PARAMETER);
      return NULL;
   }

   /* check profileMask */
   if (profileMask != WGL_CONTEXT_CORE_PROFILE_BIT_ARB &&
       profileMask != WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB &&
       profileMask != WGL_CONTEXT_ES_PROFILE_BIT_EXT) {
      SetLastError(ERROR_INVALID_PROFILE_ARB);
      return NULL;
   }

   /* check version (generate ERROR_INVALID_VERSION_ARB if bad) */
   if (majorVersion <= 0 ||
       minorVersion < 0 ||
       (profileMask != WGL_CONTEXT_ES_PROFILE_BIT_EXT &&
        ((majorVersion == 1 && minorVersion > 5) ||
         (majorVersion == 2 && minorVersion > 1) ||
         (majorVersion == 3 && minorVersion > 3) ||
         (majorVersion == 4 && minorVersion > 6) ||
         majorVersion > 4)) ||
       (profileMask == WGL_CONTEXT_ES_PROFILE_BIT_EXT &&
        ((majorVersion == 1 && minorVersion > 1) ||
         (majorVersion == 2 && minorVersion > 0) ||
         (majorVersion == 3 && minorVersion > 1) ||
         majorVersion > 3))) {
      SetLastError(ERROR_INVALID_VERSION_ARB);
      return NULL;
   }

   if ((contextFlags & WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB) &&
       majorVersion < 3) {
      SetLastError(ERROR_INVALID_VERSION_ARB);
      return 0;
   }

   if (resetStrategy != WGL_NO_RESET_NOTIFICATION_ARB &&
       resetStrategy != WGL_LOSE_CONTEXT_ON_RESET_ARB) {
      SetLastError(ERROR_INVALID_PARAMETER);
      return NULL;
   }

   wglCreateContext_t pfnwglCreateContext = NULL;
   wglDeleteContext_t pfnwglDeleteContext = NULL;

   if (stw_dev->callbacks.pfnGetDhglrc) {
      /* Used as an ICD.
       *
       * Get pointers to OPENGL32.DLL's wglCreate/DeleteContext() functions
       */
      HMODULE opengl_lib = GetModuleHandleA("opengl32.dll");
      if (!opengl_lib) {
         _debug_printf("wgl: GetModuleHandleA(\"opengl32.dll\") failed\n");
         return NULL;
      }

      pfnwglCreateContext = (wglCreateContext_t)
         GetProcAddress(opengl_lib, "wglCreateContext");
      if (!pfnwglCreateContext) {
         _debug_printf("wgl: failed to get wglCreateContext()\n");
         return NULL;
      }

      pfnwglDeleteContext = (wglDeleteContext_t)
         GetProcAddress(opengl_lib, "wglDeleteContext");
      if (!pfnwglDeleteContext) {
         _debug_printf("wgl: failed to get wglDeleteContext()\n");
         return NULL;
      }
   } else {
      pfnwglCreateContext = g_pfnwglCreateContext;
      pfnwglDeleteContext = g_pfnwglDeleteContext;

      if ((pfnwglCreateContext != NULL) != (pfnwglDeleteContext != NULL)) {
         _debug_printf("wgl: incomplete opengl32 entry point override for "
                       "wglCreateContextAttribsARB()\n");
         return NULL;
      }
   }

   DHGLRC dhglrc = 0;
   if (pfnwglCreateContext) {
      /* Call wglCreateContext to get a valid public HGLRC ID */
      context = pfnwglCreateContext(hDC);
      if (!context)
         return NULL;

      dhglrc = stw_dhglrc(context);
      if (!dhglrc) {
         pfnwglDeleteContext(context);
         return NULL;
      }
   }

   DHGLRC share_dhglrc = (DHGLRC)(UINT_PTR)hShareContext;
   struct stw_context *share_stw = stw_lookup_context(share_dhglrc);

   const struct stw_pixelformat_info *pfi = stw_pixelformat_get_info_from_hdc(hDC);
   if (!pfi) {
      if (context)
         pfnwglDeleteContext(context);
      return NULL;
   }

   struct stw_context *stw_ctx = stw_create_context_attribs(hDC, layerPlane, share_stw,
                                                            stw_dev->fscreen,
                                                            majorVersion, minorVersion,
                                                            contextFlags, profileMask, pfi,
                                                            resetStrategy);

   if (!stw_ctx) {
      if (context)
         pfnwglDeleteContext(context);
      return NULL;
   }

   stw_ctx->hglrc = context;
   DHGLRC c = stw_create_context_handle(stw_ctx, dhglrc);
   if (!c) {
      stw_destroy_context(stw_ctx);
      if (context)
         pfnwglDeleteContext(context);
      return NULL;
   }

   return (HGLRC)(UINT_PTR)c;
}


HGLRC WINAPI
wglCreateContextAttribsARB(HDC hDC, HGLRC hShareContext, const int *attribList)
{
   HGLRC hglrc = STW_WGL(wglCreateContextAttribsARB)(
      hDC, (HGLRC)(UINT_PTR)stw_dhglrc(hShareContext), attribList);
   if (!hglrc)
      return NULL;

   struct stw_context *ctx = stw_lookup_context((DHGLRC)(UINT_PTR)hglrc);
   return ctx ? ctx->hglrc : NULL;
}


/* the context is a driver handle, see stw_apitrace.h */
BOOL WINAPI
stw_wglMakeContextCurrentARB(HDC hDrawDC, HDC hReadDC, HGLRC hglrc)
{
   return stw_make_current_by_handles(hDrawDC, hReadDC,
                                      (DHGLRC)(UINT_PTR)hglrc);
}


/** Defined by WGL_ARB_make_current_read */
BOOL APIENTRY
wglMakeContextCurrentARB(HDC hDrawDC, HDC hReadDC, HGLRC hglrc)
{
   return STW_WGL(wglMakeContextCurrentARB)(
      hDrawDC, hReadDC, (HGLRC)(UINT_PTR)stw_dhglrc(hglrc));
}

HDC APIENTRY
wglGetCurrentReadDCARB(VOID)
{
   return stw_get_current_read_dc();
}
