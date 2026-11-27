/*
 * Copyright 2026 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl_vk.h"

void
nvk_clear_r32g32b32(__global uint32_t* restrict out,
                    uintptr_t row_stride_components, uint32_t width,
                    uint3 color)
{
   uint x = get_sub_group_local_id() + cl_group_id.x * 32;
   out += row_stride_components * cl_group_id.y;

   out += x;
   uint value = color[x % 3];
   if (x < width)
      *out = value;
}
