/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_TEXTURE_H
#define MTL_TEXTURE_H 1

#include "mtl_types.h"

#include <inttypes.h>
#include <stdbool.h>

/* TODO_KOSMICKRISP Move this to bridge. */
struct kk_view_layout;

/* Utils */
uint64_t mtl_texture_get_gpu_resource_id(mtl_texture *texture);

uint32_t mtl_texture_first_mipmap_in_tail(mtl_texture *texture);

/* Texture view creation */
mtl_texture *mtl_new_texture_view_with(mtl_texture *texture,
                                       const struct kk_view_layout *layout,
                                       bool swizzle);

void mtl_texture_get_bytes(mtl_texture *texture, void *host_ptr,
                           struct mtl_texture_memory_copy *data);

void mtl_texture_replace_region(mtl_texture *texture, const void *host_ptr,
                                struct mtl_texture_memory_copy *data);

#endif /* MTL_TEXTURE_H */
