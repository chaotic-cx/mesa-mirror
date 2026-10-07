/*
 * Copyright © 2021 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "brw_private.h"
#include "compiler/shader_info.h"
#include "intel/dev/intel_debug.h"
#include "intel/dev/intel_device_info.h"
#include "util/ralloc.h"

#include <gtest/gtest.h>

enum {
   SIMD8  = 0,
   SIMD16 = 1,
   SIMD32 = 2,
};

const bool spilled = true;
const bool not_spilled = false;

class SIMDSelectionTest : public ::testing::Test {
protected:
   SIMDSelectionTest()
   : mem_ctx(ralloc_context(NULL))
   , devinfo(rzalloc(mem_ctx, intel_device_info))
   , info(rzalloc(mem_ctx, struct intel_shader_info))
   , simd_state{
      .devinfo = devinfo,
      .stage = MESA_SHADER_COMPUTE,
      .cs_local_size = info->wg.local_size,
     }
   {
      info->stage = MESA_SHADER_COMPUTE;
      process_intel_debug_variable();
   }

   ~SIMDSelectionTest() {
      ralloc_free(mem_ctx);
   };

   uint8_t
   simd_mask(const bool values[SIMD_COUNT]) const
   {
      uint8_t mask = 0;
      for (unsigned i = 0; i < SIMD_COUNT; i++) {
         if (values[i])
            mask |= BITFIELD_BIT(i);
      }
      return mask;
   }

   void *mem_ctx;
   intel_device_info *devinfo;
   struct intel_shader_info *info;
   brw_simd_selection_state simd_state;
};

class SIMDSelectionCS : public SIMDSelectionTest {
protected:
   SIMDSelectionCS() {
      info->wg.local_size[0] = 32;
      info->wg.local_size[1] = 1;
      info->wg.local_size[2] = 1;

      devinfo->max_cs_workgroup_threads = 64;
   }
};

TEST_F(SIMDSelectionCS, DefaultsToSIMD16)
{
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), SIMD16);
}

TEST_F(SIMDSelectionCS, TooBigFor16)
{
   info->wg.local_size[0] = devinfo->max_cs_workgroup_threads;
   info->wg.local_size[1] = 32;
   info->wg.local_size[2] = 1;

   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, spilled);

   ASSERT_EQ(brw_simd_select(simd_state), SIMD32);
}

TEST_F(SIMDSelectionCS, WorkgroupSize1)
{
   info->wg.local_size[0] = 1;
   info->wg.local_size[1] = 1;
   info->wg.local_size[2] = 1;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), SIMD8);
}

TEST_F(SIMDSelectionCS, WorkgroupSize8)
{
   info->wg.local_size[0] = 8;
   info->wg.local_size[1] = 1;
   info->wg.local_size[2] = 1;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), SIMD8);
}

TEST_F(SIMDSelectionCS, WorkgroupSizeVariable)
{
   info->wg.local_size[0] = 0;
   info->wg.local_size[1] = 0;
   info->wg.local_size[2] = 0;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, not_spilled);

   const uint8_t prog_mask =
      1u << SIMD8 | 1u << SIMD16 | 1u << SIMD32;
   ASSERT_EQ(simd_mask(simd_state.compiled), prog_mask);
   info->wg.prog_mask = prog_mask;
   info->wg.prog_spilled = 0;

   const unsigned wg_8_1_1[] = { 8, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_8_1_1), SIMD8);

   const unsigned wg_16_1_1[] = { 16, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_16_1_1), SIMD16);

   const unsigned wg_32_1_1[] = { 32, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_32_1_1), SIMD16);
}

TEST_F(SIMDSelectionCS, WorkgroupSizeVariableSpilled)
{
   info->wg.local_size[0] = 0;
   info->wg.local_size[1] = 0;
   info->wg.local_size[2] = 0;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, spilled);

   const uint8_t prog_mask =
      1u << SIMD8 | 1u << SIMD16 | 1u << SIMD32;
   ASSERT_EQ(simd_mask(simd_state.compiled), prog_mask);
   info->wg.prog_mask = prog_mask;
   info->wg.prog_spilled = prog_mask;

   const unsigned wg_8_1_1[] = { 8, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_8_1_1), SIMD8);

   const unsigned wg_16_1_1[] = { 16, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_16_1_1), SIMD8);

   const unsigned wg_32_1_1[] = { 32, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_32_1_1), SIMD8);
}

TEST_F(SIMDSelectionCS, WorkgroupSizeVariableNoSIMD8)
{
   info->wg.local_size[0] = 0;
   info->wg.local_size[1] = 0;
   info->wg.local_size[2] = 0;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, not_spilled);

   const uint8_t prog_mask = 1u << SIMD16 | 1u << SIMD32;
   ASSERT_EQ(simd_mask(simd_state.compiled), prog_mask);
   info->wg.prog_mask = prog_mask;
   info->wg.prog_spilled = 0;

   const unsigned wg_8_1_1[] = { 8, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_8_1_1), -1);

   const unsigned wg_16_1_1[] = { 16, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_16_1_1), SIMD16);

   const unsigned wg_32_1_1[] = { 32, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_32_1_1), SIMD16);
}

TEST_F(SIMDSelectionCS, WorkgroupSizeVariableNoSIMD16)
{
   info->wg.local_size[0] = 0;
   info->wg.local_size[1] = 0;
   info->wg.local_size[2] = 0;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, not_spilled);

   const uint8_t prog_mask = 1u << SIMD8 | 1u << SIMD32;
   ASSERT_EQ(simd_mask(simd_state.compiled), prog_mask);
   info->wg.prog_mask = prog_mask;
   info->wg.prog_spilled = 0;

   const unsigned wg_8_1_1[] = { 8, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_8_1_1), SIMD8);

   const unsigned wg_16_1_1[] = { 16, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_16_1_1), SIMD8);

   const unsigned wg_32_1_1[] = { 32, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_32_1_1), SIMD8);
}

TEST_F(SIMDSelectionCS, WorkgroupSizeVariableNoSIMD8NoSIMD16)
{
   info->wg.local_size[0] = 0;
   info->wg.local_size[1] = 0;
   info->wg.local_size[2] = 0;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, not_spilled);

   const uint8_t prog_mask = 1u << SIMD32;
   ASSERT_EQ(simd_mask(simd_state.compiled), prog_mask);
   info->wg.prog_mask = prog_mask;
   info->wg.prog_spilled = 0;

   const unsigned wg_8_1_1[] = { 8, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_8_1_1), -1);

   const unsigned wg_16_1_1[] = { 16, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_16_1_1), -1);

   const unsigned wg_32_1_1[] = { 32, 1, 1 };
   ASSERT_EQ(brw_simd_select_for_workgroup_size(devinfo, info, wg_32_1_1), SIMD32);
}

TEST_F(SIMDSelectionCS, SpillAtSIMD8)
{
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, spilled);
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), SIMD8);
}

TEST_F(SIMDSelectionCS, SpillAtSIMD16)
{
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, spilled);
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), SIMD8);
}

TEST_F(SIMDSelectionCS, FailAtSIMD8)
{
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_failed(simd_state, SIMD8, "fail");
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), -1);
}

TEST_F(SIMDSelectionCS, FailAtSIMD8Forced)
{
   const uint64_t saved_intel_simd = intel_simd;
   const uint32_t saved_intel_simd_overridden = intel_simd_overridden;
   intel_simd |= DEBUG_CS_SIMD;
   intel_simd_overridden |= BITFIELD_BIT(MESA_SHADER_COMPUTE);

   EXPECT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_failed(simd_state, SIMD8, "fail");
   EXPECT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   EXPECT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   EXPECT_EQ(brw_simd_select(simd_state), -1);

   intel_simd = saved_intel_simd;
   intel_simd_overridden = saved_intel_simd_overridden;
}

TEST_F(SIMDSelectionCS, FailAtSIMD16)
{
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_failed(simd_state, SIMD16, "fail");
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), SIMD8);
}

TEST_F(SIMDSelectionCS, FailAtSIMD32Xe3)
{
   /* Xe3 attempts the widths in decreasing order, so a failure says
    * nothing about the narrower ones.
    */
   devinfo->ver = 30;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_failed(simd_state, SIMD32, "fail");
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);

   ASSERT_EQ(brw_simd_select(simd_state), SIMD16);
}

TEST_F(SIMDSelectionCS, EnvironmentVariable16Only)
{
   const uint64_t saved_intel_simd = intel_simd;
   const uint32_t saved_intel_simd_overridden = intel_simd_overridden;
   intel_simd = (intel_simd & ~DEBUG_CS_SIMD) | DEBUG_CS_SIMD16;
   intel_simd_overridden |= BITFIELD_BIT(MESA_SHADER_COMPUTE);

   EXPECT_FALSE(brw_simd_should_compile(simd_state, SIMD8));
   EXPECT_STREQ(simd_state.error[SIMD8],
                "Disabled by INTEL_SIMD_DEBUG environment variable");

   EXPECT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);
   EXPECT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   EXPECT_EQ(brw_simd_select(simd_state), SIMD16);

   intel_simd = saved_intel_simd;
   intel_simd_overridden = saved_intel_simd_overridden;
}

TEST_F(SIMDSelectionCS, EnvironmentVariable32)
{
   intel_simd_overridden |= BITFIELD_BIT(MESA_SHADER_COMPUTE);

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, not_spilled);

   ASSERT_EQ(brw_simd_select(simd_state), SIMD32);
}

TEST_F(SIMDSelectionCS, EnvironmentVariable32ButSpills)
{
   intel_simd_overridden |= BITFIELD_BIT(MESA_SHADER_COMPUTE);

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, spilled);

   ASSERT_EQ(brw_simd_select(simd_state), SIMD16);
}

TEST_F(SIMDSelectionCS, Require8)
{
   simd_state.required_width = 8;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), SIMD8);
}

TEST_F(SIMDSelectionCS, Require8ErrorWhenNotCompile)
{
   simd_state.required_width = 8;

   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), -1);
}

TEST_F(SIMDSelectionCS, Require16)
{
   simd_state.required_width = 16;

   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), SIMD16);
}

TEST_F(SIMDSelectionCS, Require16ErrorWhenNotCompile)
{
   simd_state.required_width = 16;

   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), -1);
}

TEST_F(SIMDSelectionCS, Require32)
{
   simd_state.required_width = 32;

   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, not_spilled);

   ASSERT_EQ(brw_simd_select(simd_state), SIMD32);
}

TEST_F(SIMDSelectionCS, Require32ErrorWhenNotCompile)
{
   simd_state.required_width = 32;

   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_FALSE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));

   ASSERT_EQ(brw_simd_select(simd_state), -1);
}

TEST_F(SIMDSelectionCS, FirstCompiledIsSIMD8)
{
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   brw_simd_mark_compiled(simd_state, SIMD8, not_spilled);

   ASSERT_TRUE(brw_simd_any_compiled(simd_state));
   ASSERT_EQ(brw_simd_first_compiled(simd_state), SIMD8);
}

TEST_F(SIMDSelectionCS, FirstCompiledIsSIMD16)
{
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   brw_simd_mark_compiled(simd_state, SIMD16, not_spilled);

   ASSERT_TRUE(brw_simd_any_compiled(simd_state));
   ASSERT_EQ(brw_simd_first_compiled(simd_state), SIMD16);
}

TEST_F(SIMDSelectionCS, FirstCompiledIsSIMD32)
{
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD8));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD16));
   ASSERT_TRUE(brw_simd_should_compile(simd_state, SIMD32));
   brw_simd_mark_compiled(simd_state, SIMD32, not_spilled);

   ASSERT_TRUE(brw_simd_any_compiled(simd_state));
   ASSERT_EQ(brw_simd_first_compiled(simd_state), SIMD32);
}
