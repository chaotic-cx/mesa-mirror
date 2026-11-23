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
#include "vk_memory.h"

#include <stdlib.h>

#include "util/simple_mtx.h"

static simple_mtx_t vk_memory_budgets_mutex = SIMPLE_MTX_INITIALIZER;
static struct list_head vk_memory_budgets = {
   .prev = &vk_memory_budgets,
   .next = &vk_memory_budgets,
};

struct vk_memory_budget *
vk_memory_budget_acquire(uint64_t key)
{
   struct vk_memory_budget *budget = NULL;

   simple_mtx_lock(&vk_memory_budgets_mutex);

   list_for_each_entry(struct vk_memory_budget, b, &vk_memory_budgets, link) {
      if (b->key == key) {
         budget = b;
         break;
      }
   }

   if (budget) {
      budget->ref_count++;
   } else {
      budget = calloc(1, sizeof(*budget));
      if (budget) {
         budget->key = key;
         budget->ref_count = 1;
         list_addtail(&budget->link, &vk_memory_budgets);
      }
   }

   simple_mtx_unlock(&vk_memory_budgets_mutex);

   return budget;
}

void
vk_memory_budget_release(struct vk_memory_budget *budget)
{
   if (!budget)
      return;

   simple_mtx_lock(&vk_memory_budgets_mutex);

   assert(budget->ref_count > 0);
   if (--budget->ref_count == 0) {
      list_del(&budget->link);
      free(budget);
   }

   simple_mtx_unlock(&vk_memory_budgets_mutex);
}
