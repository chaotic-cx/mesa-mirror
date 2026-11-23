/*
 * Copyright © 2020 Intel Corporation
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
#ifndef VK_MEMORY_H
#define VK_MEMORY_H

#include <stdbool.h>
#include <vulkan/vulkan_core.h>

#include "util/list.h"
#include "util/macros.h"
#include "util/u_atomic.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vk_memory_budget {
   struct list_head link;
   uint32_t ref_count;
   uint64_t key;

   /* Align to 64 bits to make atomic operations faster on 32 bit platforms. */
   alignas(8) uint64_t heap_used[VK_MAX_MEMORY_HEAPS];
};

/* Returns NULL on allocation failure. */
struct vk_memory_budget *vk_memory_budget_acquire(uint64_t key);

void vk_memory_budget_release(struct vk_memory_budget *budget);

static inline uint64_t
vk_memory_budget_used(const struct vk_memory_budget *budget, uint32_t heap)
{
   assert(heap < VK_MAX_MEMORY_HEAPS);
   return p_atomic_read(&budget->heap_used[heap]);
}

static inline void
vk_memory_budget_add(struct vk_memory_budget *budget, uint32_t heap,
                     uint64_t size)
{
   assert(heap < VK_MAX_MEMORY_HEAPS);
   p_atomic_add(&budget->heap_used[heap], size);
}

static inline bool
vk_memory_budget_try_add(struct vk_memory_budget *budget, uint32_t heap,
                         uint64_t size, uint64_t heap_size)
{
   assert(heap < VK_MAX_MEMORY_HEAPS);
   uint64_t used = p_atomic_add_return(&budget->heap_used[heap], size);
   if (used > heap_size) {
      p_atomic_add(&budget->heap_used[heap], -(int64_t)size);
      return false;
   }

   return true;
}

static inline void
vk_memory_budget_sub(struct vk_memory_budget *budget, uint32_t heap,
                     uint64_t size)
{
   assert(heap < VK_MAX_MEMORY_HEAPS);
   p_atomic_add(&budget->heap_used[heap], -(int64_t)size);
}

#ifdef __cplusplus
}
#endif

#endif /* VK_MEMORY_H */
