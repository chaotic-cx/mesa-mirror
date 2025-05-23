/*
 * Copyright (c) 2018-2019 Lima Project
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
 */
#ifndef H_LIMA_FORMAT
#define H_LIMA_FORMAT

#include <stdbool.h>

#include <util/format/u_formats.h>
#include "lima_pack.h"

bool lima_format_texel_supported(enum pipe_format f);
bool lima_format_pixel_supported(enum pipe_format f);
int lima_format_get_texel(enum pipe_format f);
int lima_format_get_pixel(enum pipe_format f);
int lima_format_get_texel_reload(enum pipe_format f);
bool lima_format_get_texel_swap_rb(enum pipe_format f);
bool lima_format_get_pixel_swap_rb(enum pipe_format f);
const uint8_t *lima_format_get_texel_swizzle(enum pipe_format f);
struct LIMA_TILEBUFFER_CHANNEL_LAYOUT lima_format_get_channel_layout(enum pipe_format f);

#endif
