/*
 * Copyright © 2021 Google, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _UTIL_PERFETTO_H
#define _UTIL_PERFETTO_H

/* Mesa's perfetto C API plus the perfetto C++ SDK, for code that defines
 * perfetto data sources or emits trace packets directly.
 */
#include "util/perf/u_perfetto_api.h"

#if defined(__cplusplus) && defined(HAVE_PERFETTO)
/* perfetto's use of STL triggers GCC warnings
 * in libstdc++ that appear to be spurious. See:
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109717
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106093
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#if defined(__has_warning)
   #if __has_warning("-Wstringop-overflow")
      // warning group not present on Android NDK
      #pragma GCC diagnostic ignored "-Wstringop-overflow"
   #endif
#else
   #pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#ifndef ANDROID_LIBPERFETTO
#pragma push_macro("minor")
#undef minor
#include <perfetto.h>
#pragma pop_macro("minor")
#else
#include <perfetto/tracing.h>
#endif

#pragma GCC diagnostic pop
#endif /* __cplusplus && HAVE_PERFETTO */

#endif /* _UTIL_PERFETTO_H */
