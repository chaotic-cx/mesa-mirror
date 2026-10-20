/*
 * SPDX-License-Identifier: MIT
 */

#ifndef STW_APITRACE_H
#define STW_APITRACE_H

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * the WGL calls that the ICD entry points and the ARB context functions make,
 * with Mesa's context handles, so that apitrace records the same handles in
 * ICD mode, where opengl32 gives apps its own
 */
HGLRC WINAPI
stw_wglCreateContext(HDC hdc);

HGLRC WINAPI
stw_wglCreateContextAttribsARB(HDC hDC, HGLRC hShareContext,
                               const int *attribList);

HGLRC WINAPI
stw_wglCreateLayerContext(HDC hdc, int iLayerPlane);

BOOL WINAPI
stw_wglDeleteContext(HGLRC hglrc);

HGLRC WINAPI
stw_wglGetCurrentContext(void);

HDC WINAPI
stw_wglGetCurrentDC(void);

PROC WINAPI
stw_wglGetProcAddress(LPCSTR lpszProc);

BOOL WINAPI
stw_wglMakeContextCurrentARB(HDC hDrawDC, HDC hReadDC, HGLRC hglrc);

BOOL WINAPI
stw_wglMakeCurrent(HDC hdc, HGLRC hglrc);

BOOL WINAPI
stw_wglShareLists(HGLRC hglrc1, HGLRC hglrc2);

BOOL WINAPI
stw_wglSwapBuffers(HDC hdc);

BOOL WINAPI
stw_wglSwapLayerBuffers(HDC hdc, UINT fuPlanes);

#ifdef HAVE_APITRACE

/* generated, they call apitrace's tracers or the functions above */
HGLRC WINAPI
stw_apitrace_wglCreateContext(HDC hdc);

HGLRC WINAPI
stw_apitrace_wglCreateContextAttribsARB(HDC hDC, HGLRC hShareContext,
                                        const int *attribList);

HGLRC WINAPI
stw_apitrace_wglCreateLayerContext(HDC hdc, int iLayerPlane);

BOOL WINAPI
stw_apitrace_wglDeleteContext(HGLRC hglrc);

PROC WINAPI
stw_apitrace_wglGetProcAddress(LPCSTR lpszProc);

BOOL WINAPI
stw_apitrace_wglMakeContextCurrentARB(HDC hDrawDC, HDC hReadDC, HGLRC hglrc);

BOOL WINAPI
stw_apitrace_wglMakeCurrent(HDC hdc, HGLRC hglrc);

BOOL WINAPI
stw_apitrace_wglShareLists(HGLRC hglrc1, HGLRC hglrc2);

BOOL WINAPI
stw_apitrace_wglSwapBuffers(HDC hdc);

BOOL WINAPI
stw_apitrace_wglSwapLayerBuffers(HDC hdc, UINT fuPlanes);

#define STW_WGL(name) stw_apitrace_##name

#else

#define STW_WGL(name) stw_##name

#endif

#ifdef __cplusplus
}
#endif

#endif
