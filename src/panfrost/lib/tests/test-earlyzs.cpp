/*
 * Copyright (C) 2022 Collabora, Ltd.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "util/pan_ir.h"
#include "pan_earlyzs.h"

#include <gtest/gtest.h>

/*
 * Test the early-ZS helpers used on Bifrost and Valhall. Early-ZS state depends
 * on both shader state and draw-time API state. As such, there are two helpers
 * -- analzye and get -- that separate the link-time analysis of a fragment
 * shader from the draw-time classification. The internal data structure is not
 * under test, only the external API. So we test only the composition.
 */

#define ZS_WRITEMASK             BITFIELD_BIT(0)
#define ALPHA2COV                BITFIELD_BIT(1)
#define ZS_ALWAYS_PASSES         BITFIELD_BIT(2)
#define DISCARD                  BITFIELD_BIT(3)
#define WRITES_Z                 BITFIELD_BIT(4)
#define WRITES_S                 BITFIELD_BIT(5)
#define WRITES_COV               BITFIELD_BIT(6)
#define SIDEFX                   BITFIELD_BIT(7)
#define API_EARLY                BITFIELD_BIT(8)
#define SHADER_READS_ZS          BITFIELD_BIT(9)
#define ARCH_HAS_READONLY_ZS_OPT BITFIELD_BIT(10)
#define ARCH_HAS_STATE_TRACK_OPT BITFIELD_BIT(11)

static void
test(enum pan_earlyzs expected_update, enum pan_earlyzs expected_kill,
     bool expected_shader_readonly_zs, uint32_t flags)
{
   enum pan_earlyzs_zs_tilebuf_read zs_read = PAN_EARLYZS_ZS_TILEBUF_NOT_READ;
   struct pan_shader_info info = {};
   info.fs.can_discard = !!(flags & DISCARD);
   info.fs.writes_depth = !!(flags & WRITES_Z);
   info.fs.writes_stencil = !!(flags & WRITES_S);
   info.fs.writes_coverage = !!(flags & WRITES_COV);
   info.fs.early_fragment_tests = !!(flags & API_EARLY);
   info.writes_global = !!(flags & SIDEFX);

   unsigned arch = 9;

   if (flags & ARCH_HAS_STATE_TRACK_OPT)
      arch = 11;
   else if (flags & ARCH_HAS_READONLY_ZS_OPT)
      arch = 10;

   if (flags & SHADER_READS_ZS) {
      if (flags & (WRITES_Z | WRITES_S))
         zs_read = PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT;
      else
         zs_read = PAN_EARLYZS_ZS_TILEBUF_READ_OPT;
   }

   struct pan_earlyzs_state result = pan_earlyzs_get(
      pan_earlyzs_analyze(&info, arch), !!(flags & ZS_WRITEMASK),
      !!(flags & ALPHA2COV), !!(flags & ZS_ALWAYS_PASSES), zs_read);

   ASSERT_EQ(result.update, expected_update);
   ASSERT_EQ(result.kill, expected_kill);
   ASSERT_EQ(result.shader_readonly_zs, expected_shader_readonly_zs);
}

#define CASE(expected_update, expected_kill, flags)                            \
   test(PAN_EARLYZS_##expected_update, PAN_EARLYZS_##expected_kill, false,     \
        flags)

#define CASE_RO_ZS(expected_update, expected_kill, expected_ro_zs, flags)      \
   test(PAN_EARLYZS_##expected_update, PAN_EARLYZS_##expected_kill,            \
        expected_ro_zs, flags)

TEST(EarlyZS, APIForceEarly)
{
   CASE(FORCE_EARLY, FORCE_EARLY, API_EARLY);
   CASE(FORCE_EARLY, FORCE_EARLY, API_EARLY | WRITES_Z | WRITES_S);
   CASE(FORCE_EARLY, FORCE_EARLY, API_EARLY | ALPHA2COV | DISCARD);
}

TEST(EarlyZS, ShaderCalculatesZS)
{
   CASE(FORCE_LATE, FORCE_LATE, WRITES_Z);
   CASE(FORCE_LATE, FORCE_LATE, WRITES_S);
   CASE(FORCE_LATE, FORCE_LATE, WRITES_Z | WRITES_S);
   CASE(FORCE_LATE, FORCE_LATE, WRITES_Z | WRITES_S | SIDEFX);
   CASE(FORCE_LATE, FORCE_LATE, WRITES_Z | WRITES_S | ZS_ALWAYS_PASSES);
   CASE(FORCE_LATE, FORCE_LATE, WRITES_Z | ZS_ALWAYS_PASSES | ALPHA2COV);
}

TEST(EarlyZS, ModifiesCoverageWritesZSNoSideFX)
{
   CASE(FORCE_LATE, FORCE_EARLY, ZS_WRITEMASK | WRITES_COV);
   CASE(FORCE_LATE, FORCE_EARLY, ZS_WRITEMASK | DISCARD);
   CASE(FORCE_LATE, FORCE_EARLY, ZS_WRITEMASK | ALPHA2COV);
   CASE(FORCE_LATE, FORCE_EARLY,
        ZS_WRITEMASK | WRITES_COV | DISCARD | ALPHA2COV);
}

TEST(EarlyZS, ModifiesCoverageWritesZSNoSideFXAlt)
{
   CASE(FORCE_LATE, WEAK_EARLY, ZS_ALWAYS_PASSES | ZS_WRITEMASK | WRITES_COV);
   CASE(FORCE_LATE, WEAK_EARLY, ZS_ALWAYS_PASSES | ZS_WRITEMASK | DISCARD);
   CASE(FORCE_LATE, WEAK_EARLY, ZS_ALWAYS_PASSES | ZS_WRITEMASK | ALPHA2COV);
   CASE(FORCE_LATE, WEAK_EARLY,
        ZS_ALWAYS_PASSES | ZS_WRITEMASK | WRITES_COV | DISCARD | ALPHA2COV);
}

TEST(EarlyZS, ModifiesCoverageWritesZSSideFX)
{
   CASE(FORCE_LATE, FORCE_LATE, ZS_WRITEMASK | SIDEFX | WRITES_COV);
   CASE(FORCE_LATE, FORCE_LATE, ZS_WRITEMASK | SIDEFX | DISCARD);
   CASE(FORCE_LATE, FORCE_LATE, ZS_WRITEMASK | SIDEFX | ALPHA2COV);
   CASE(FORCE_LATE, FORCE_LATE,
        ZS_WRITEMASK | SIDEFX | WRITES_COV | DISCARD | ALPHA2COV);
}

TEST(EarlyZS, SideFXNoShaderZS)
{
   CASE(FORCE_EARLY, FORCE_LATE, SIDEFX);
   CASE(FORCE_EARLY, FORCE_LATE, SIDEFX | DISCARD);
   CASE(FORCE_EARLY, FORCE_LATE, SIDEFX | WRITES_COV);
   CASE(FORCE_LATE, FORCE_LATE, SIDEFX | ALPHA2COV);
}

TEST(EarlyZS, SideFXNoShaderZSAlt)
{
   CASE(WEAK_EARLY, FORCE_LATE, ZS_ALWAYS_PASSES | SIDEFX);
   CASE(WEAK_EARLY, FORCE_LATE, ZS_ALWAYS_PASSES | SIDEFX | DISCARD);
   CASE(WEAK_EARLY, FORCE_LATE, ZS_ALWAYS_PASSES | SIDEFX | WRITES_COV);
   CASE(FORCE_LATE, FORCE_LATE, ZS_ALWAYS_PASSES | SIDEFX | ALPHA2COV);
}

TEST(EarlyZS, NoSideFXNoShaderZS)
{
   CASE(FORCE_EARLY, FORCE_EARLY, 0);
   CASE(FORCE_LATE, FORCE_EARLY, ALPHA2COV | DISCARD | WRITES_COV);
   CASE(FORCE_EARLY, FORCE_EARLY, ZS_WRITEMASK);
}

TEST(EarlyZS, ShaderReadZS)
{
   CASE_RO_ZS(FORCE_LATE, FORCE_LATE, false, SIDEFX | SHADER_READS_ZS);
   CASE_RO_ZS(FORCE_EARLY, FORCE_LATE, true,
              SIDEFX | SHADER_READS_ZS | ARCH_HAS_READONLY_ZS_OPT);
   CASE_RO_ZS(FORCE_LATE, FORCE_LATE, false,
              SIDEFX | SHADER_READS_ZS | WRITES_Z | ARCH_HAS_READONLY_ZS_OPT);
   CASE_RO_ZS(FORCE_EARLY, FORCE_EARLY, true,
              SHADER_READS_ZS | ARCH_HAS_READONLY_ZS_OPT);
   CASE_RO_ZS(FORCE_LATE, FORCE_LATE, false,
              SHADER_READS_ZS | WRITES_Z | ARCH_HAS_READONLY_ZS_OPT);
   CASE_RO_ZS(FORCE_LATE, WEAK_EARLY, false,
              SHADER_READS_ZS | ZS_ALWAYS_PASSES);
   CASE_RO_ZS(FORCE_LATE, FORCE_EARLY, false, SHADER_READS_ZS);
}

TEST(EarlyZS, NoSideFXNoShaderZSAlt)
{
   CASE(WEAK_EARLY, WEAK_EARLY, ZS_ALWAYS_PASSES);
   CASE(FORCE_EARLY, WEAK_EARLY, ZS_ALWAYS_PASSES | ARCH_HAS_STATE_TRACK_OPT);
   CASE(FORCE_LATE, WEAK_EARLY,
        ZS_ALWAYS_PASSES | ALPHA2COV | DISCARD | WRITES_COV);
   CASE(WEAK_EARLY, WEAK_EARLY, ZS_ALWAYS_PASSES | ZS_WRITEMASK);
}
