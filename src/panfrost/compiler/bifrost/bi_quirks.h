/*
 * Copyright (C) 2019-2020 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef __BI_QUIRKS_H
#define __BI_QUIRKS_H

#include "panfrost/model/pan_model.h"

/* Model-specific quirks requiring compiler workarounds/etc. Quirks
 * may be errata requiring a workaround, or features. We're trying to be
 * quirk-positive here; quirky is the best! */

/* Whether this GPU lacks support for fp32 transcendentals, requiring backend
 * lowering to low-precision lookup tables and polynomial approximation */

#define BIFROST_NO_FP32_TRANSCENDENTALS (1 << 0)

/* Whether this GPU lacks support for the full form of the CLPER instruction.
 * These GPUs use a simple encoding of CLPER that does not support
 * inactive_result, subgroup_size, or lane_op. Using those features requires
 * lowering to additional ALU instructions. The encoding forces inactive_result
 * = zero, subgroup_size = subgroup4, and lane_op = none. */

#define BIFROST_LIMITED_CLPER (1 << 1)

static inline unsigned
bifrost_get_quirks(unsigned arch, const struct pan_model *model)
{
   /* If we don't know the model, be conservative and use arch_minor = 0.
    *
    * All Bifrost quirks are really just features added in subarchitecture
    * versions.
    */
   unsigned arch_minor = 0;
   if (model)
      arch_minor = PAN_PROD_ID_ARCH_MINOR(model->gpu_prod_id);

   /* Applies to archs before 6.2 */
   unsigned quirks = 0;
   if (arch < 7 && arch_minor < 2)
      quirks |= BIFROST_NO_FP32_TRANSCENDENTALS;

   /* Applies to archs before 7.2 */
   if (arch < 7 || (arch == 7 && arch_minor < 2))
      quirks |= BIFROST_LIMITED_CLPER;

   return quirks;
}

#endif
