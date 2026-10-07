/*
 * Copyright © 2021 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "brw_private.h"
#include "compiler/shader_info.h"
#include "intel/dev/intel_debug.h"
#include "intel/dev/intel_device_info.h"
#include "util/ralloc.h"

unsigned
brw_required_dispatch_width(const struct shader_info *info)
{
   if (info->min_subgroup_size == info->max_subgroup_size) {
      return info->max_subgroup_size;
   } else {
      return 0;
   }
}

unsigned
brw_geometry_stage_dispatch_width(const struct intel_device_info *devinfo)
{
   if (devinfo->ver >= 20)
      return 16;
   return 8;
}

static inline bool
test_bit(unsigned mask, unsigned bit) {
   return mask & (1u << bit);
}

/**
 * Return true if INTEL_SIMD_DEBUG allows the given SIMD mode.
 */
static bool
simd_debug_allowed(mesa_shader_stage stage, unsigned simd)
{
   return intel_simd_debug_allowed_modes(stage) & (1 << simd);
}

/**
 * Return true if INTEL_SIMD_DEBUG force-enables the given SIMD mode.
 */
static bool
simd_debug_forced(mesa_shader_stage stage, unsigned simd)
{
   return intel_simd_debug_forced(stage) && simd_debug_allowed(stage, simd);
}

bool
brw_simd_should_compile(brw_simd_selection_state &state, unsigned simd)
{
   assert(simd < SIMD_COUNT);
   assert(!state.compiled[simd]);
   assert(!state.failed[simd]);

   const unsigned width = 8u << simd;

   /* Hard requirements */
   if (width == 8 && state.devinfo->ver >= 20) {
      state.error[simd] = "SIMD8 not supported on Xe2+";
      return false;
   }

   if (width == 32 && state.uses_btd_stack_ids) {
      state.error[simd] = "Bindless shader calls not supported";
      return false;
   }

   /* A wider width will fail if a narrower width failed, even when forced. */
   for (unsigned i = 0; i < simd; i++) {
      if (state.failed[i]) {
         state.error[simd] = "Narrower SIMD width failed to compile";
         return false;
      }
   }

   if (state.required_width) {
      if (state.required_width == width)
         return true;

      state.error[simd] = "Different than required dispatch width";
      return false;
   }

   /* For shaders with variable size workgroup, in most cases we can compile
    * all the variants (exceptions are bindless dispatch & ray queries), since
    * the choice will happen only at dispatch time.
    */
   const bool workgroup_size_variable =
      state.cs_local_size && state.cs_local_size[0] == 0;

   if (!workgroup_size_variable &&
       likely(!simd_debug_forced(state.stage, simd))) {
      if (state.spilled[simd]) {
         state.error[simd] = "Would spill";
         return false;
      }

      const unsigned min_simd = state.devinfo->ver >= 20 ? 1 : 0;

      if (simd > min_simd &&
          state.beyond_threshold[simd] &&
          brw_simd_any_compiled(state)) {
         state.error[simd] = "estimated to be beyond the pressure threshold";
         return false;
      }

      if (state.cs_local_size != NULL) {
         const unsigned workgroup_size = state.cs_local_size[0] *
                                         state.cs_local_size[1] *
                                         state.cs_local_size[2];

         unsigned max_threads = state.devinfo->max_cs_workgroup_threads;

         if (simd > min_simd && workgroup_size <= (width / 2)) {
            state.error[simd] = "Workgroup size already fits in smaller SIMD";
            return false;
         }

         if (DIV_ROUND_UP(workgroup_size, width) > max_threads) {
            state.error[simd] = "Would need more than max_threads to fit all invocations";
            return false;
         }
      }

      /* The SIMD32 is only enabled for cases it is needed unless forced.
       *
       * TODO: Use performance_analysis and drop this rule.
       */
      if (width == 32 && state.devinfo->ver < 20 &&
          (state.compiled[0] || state.compiled[1])) {
         state.error[simd] = "SIMD32 not required (use INTEL_SIMD_DEBUG to force)";
         return false;
      }
   }

   if (unlikely(!simd_debug_allowed(state.stage, simd))) {
      state.error[simd] = "Disabled by INTEL_SIMD_DEBUG environment variable";
      return false;
   }

   return true;
}

void
brw_simd_mark_compiled(brw_simd_selection_state &state, unsigned simd, bool spilled)
{
   assert(simd < SIMD_COUNT);
   assert(!state.compiled[simd]);

   state.compiled[simd] = true;

   /* If a SIMD spilled, all the larger ones would spill too. */
   if (spilled) {
      for (unsigned i = simd; i < SIMD_COUNT; i++)
         state.spilled[i] = true;
   }
}

void
brw_simd_mark_failed(brw_simd_selection_state &state, unsigned simd, const char *error)
{
   assert(simd < SIMD_COUNT);
   assert(!state.compiled[simd]);

   state.error[simd] = error;
   state.failed[simd] = true;
}

int
brw_simd_select(const struct brw_simd_selection_state &state)
{
   for (int i = SIMD_COUNT - 1; i >= 0; i--) {
      if (state.compiled[i] && !state.spilled[i])
         return i;
   }
   for (int i = SIMD_COUNT - 1; i >= 0; i--) {
      if (state.compiled[i])
         return i;
   }
   return -1;
}

int
brw_simd_select_for_workgroup_size(const struct intel_device_info *devinfo,
                                   const struct intel_shader_info *info,
                                   const unsigned *sizes)
{
   const bool uses_btd_stack_ids =
      mesa_shader_stage_is_compute(info->stage) &&
      info->cs.uses_btd_stack_ids;

   if (!sizes || (info->wg.local_size[0] == sizes[0] &&
                  info->wg.local_size[1] == sizes[1] &&
                  info->wg.local_size[2] == sizes[2])) {
      brw_simd_selection_state simd_state{
         .stage = info->stage,
         .cs_local_size = info->wg.local_size,
         .uses_btd_stack_ids = uses_btd_stack_ids,
      };

      /* Propagate the bin information back to the simd_state, so we can use
       * select() directly.
       */
      for (int i = 0; i < SIMD_COUNT; i++) {
         simd_state.compiled[i] = test_bit(info->wg.prog_mask, i);
         simd_state.spilled[i] = test_bit(info->wg.prog_spilled, i);
      }

      return brw_simd_select(simd_state);
   }

   brw_simd_selection_state simd_state{
      .devinfo = devinfo,
      .stage = info->stage,
      .cs_local_size = sizes,
      .uses_btd_stack_ids = uses_btd_stack_ids,
   };

   for (unsigned simd = 0; simd < SIMD_COUNT; simd++) {
      /* We are not recompiling, so use original results of prog_mask and
       * prog_spilled as they will already contain all possible compilations.
       */
      if (brw_simd_should_compile(simd_state, simd) &&
          test_bit(info->wg.prog_mask, simd)) {
         brw_simd_mark_compiled(simd_state, simd,
                                test_bit(info->wg.prog_spilled, simd));
      }
   }

   return brw_simd_select(simd_state);
}
