/*
 * Copyright © 2015 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef GENX_VIDEO_PACK_H
#define GENX_VIDEO_PACK_H

#ifndef GFX_VERx10
#  error "The GFX_VERx10 macro must be defined"
#endif

#if (GFX_VERx10 == 40)
#  include "genxml/gen40_video_pack.h"
#elif (GFX_VERx10 == 45)
#  include "genxml/gen45_video_pack.h"
#elif (GFX_VERx10 == 50)
#  include "genxml/gen50_video_pack.h"
#elif (GFX_VERx10 == 60)
#  include "genxml/gen60_video_pack.h"
#elif (GFX_VERx10 == 70)
#  include "genxml/gen70_video_pack.h"
#elif (GFX_VERx10 == 75)
#  include "genxml/gen75_video_pack.h"
#elif (GFX_VERx10 == 80)
#  include "genxml/gen80_video_pack.h"
#elif (GFX_VERx10 == 90)
#  include "genxml/gen90_video_pack.h"
#elif (GFX_VERx10 == 110)
#  include "genxml/gen110_video_pack.h"
#elif (GFX_VERx10 == 120)
#  include "genxml/gen120_video_pack.h"
#elif (GFX_VERx10 == 125)
#  include "genxml/gen125_video_pack.h"
#elif (GFX_VERx10 == 200)
#  include "genxml/gen200_video_pack.h"
#elif (GFX_VERx10 == 300)
#  include "genxml/gen300_video_pack.h"
#else
#  error "Need to add a pack header include for this gen"
#endif

#endif /* GENX_VIDEO_PACK_H */
