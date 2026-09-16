/*
 * Copyright 2026 Arm Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/u_inlines.h"

#include "ethosu_alloc.h"
#include "ethosu_lower.h"
#include "ethosu_ml.h"

unsigned
ethosu_allocate_feature_map(struct ethosu_subgraph *subgraph,
                            const unsigned *required_sizes,
                            struct ethosu_feature_map *fm)
{
   struct ethosu_tensor *tensor = fm->tensor;
   uint64_t size;

   assert(tensor);

   if (tensor->size > 0)
      return tensor->offset;

   size = (uint64_t)required_sizes[tensor - util_dynarray_element(&subgraph->tensors,
                                                                  struct ethosu_tensor, 0)] *
          tensor->batches;
   assert(size <= UINT_MAX);

   tensor->offset = subgraph->io_used;
   tensor->size = size;
   subgraph->io_used += ALIGN_POT(size, 16);

   return tensor->offset;
}
