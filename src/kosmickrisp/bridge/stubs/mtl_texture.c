/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_texture.h"

/* Utils*/
uint64_t
mtl_texture_get_gpu_resource_id(mtl_texture *texture)
{
   return 0u;
}

uint32_t
mtl_texture_first_mipmap_in_tail(mtl_texture *texture)
{
   return 0u;
}

/* Texture view creation */
mtl_texture *
mtl_new_texture_view_with(mtl_texture *texture,
                          const struct kk_view_layout *layout, bool swizzle)
{
   return NULL;
}

void
mtl_texture_get_bytes(mtl_texture *texture, void *host_ptr,
                      struct mtl_texture_memory_copy *data)
{
}

void
mtl_texture_replace_region(mtl_texture *texture, const void *host_ptr,
                           struct mtl_texture_memory_copy *data)
{
}
