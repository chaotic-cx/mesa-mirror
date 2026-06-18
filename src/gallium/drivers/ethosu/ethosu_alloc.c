/*
 * Copyright 2026 Arm Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/u_inlines.h"

#include "ethosu_alloc.h"
#include "ethosu_lower.h"
#include "ethosu_ml.h"

static unsigned
ethosu_tensor_storage_size(const struct ethosu_subgraph *subgraph,
                           const unsigned *required_sizes,
                           struct ethosu_tensor *tensor)
{
   uint64_t size;

   assert(tensor);

   if (tensor->layout == ETHOSU_LAYOUT_NHWC) {
      size = (uint64_t)tensor->shape.width * tensor->shape.height *
             tensor->shape.depth * tensor->type_size;
   } else if (tensor->layout == ETHOSU_LAYOUT_NHCWB16) {
      size = (uint64_t)tensor->shape.width * tensor->shape.height *
             align(tensor->shape.depth, 16) * tensor->type_size;
   } else {
      UNREACHABLE("Unsupported layout");
   }

   unsigned index = tensor - util_dynarray_element(&subgraph->tensors,
                                                   struct ethosu_tensor, 0);
   size = MAX2(size, required_sizes[index]);
   size *= tensor->batches;
   assert(size <= UINT_MAX);
   return size;
}

struct ethosu_io_interval {
   struct ethosu_tensor *tensor;
   unsigned parent;
   unsigned first_use;
   unsigned last_use;
   unsigned size;
   unsigned offset;
   bool active;
   bool has_producer;
   bool has_consumer;
};

struct ethosu_io_free_block {
   unsigned offset;
   unsigned size;
};

static unsigned
ethosu_find_interval(struct ethosu_io_interval *intervals, unsigned count,
                     struct ethosu_tensor *tensor)
{
   for (unsigned i = 0; i < count; i++) {
      if (intervals[i].tensor == tensor)
         return i;
   }

   UNREACHABLE("Tensor missing from IO allocator");
}

static unsigned
ethosu_interval_root(struct ethosu_io_interval *intervals, unsigned idx)
{
   while (intervals[idx].parent != idx) {
      intervals[idx].parent = intervals[intervals[idx].parent].parent;
      idx = intervals[idx].parent;
   }

   return idx;
}

static void
ethosu_union_intervals(struct ethosu_io_interval *intervals, unsigned a,
                       unsigned b)
{
   a = ethosu_interval_root(intervals, a);
   b = ethosu_interval_root(intervals, b);

   if (a == b)
      return;

   intervals[b].parent = a;
   intervals[a].first_use = MIN2(intervals[a].first_use, intervals[b].first_use);
   intervals[a].last_use = MAX2(intervals[a].last_use, intervals[b].last_use);
   intervals[a].size = MAX2(intervals[a].size, intervals[b].size);
   intervals[a].has_producer |= intervals[b].has_producer;
   intervals[a].has_consumer |= intervals[b].has_consumer;
}

static void
ethosu_mark_interval_access(struct ethosu_io_interval *intervals,
                            unsigned count,
                            struct ethosu_tensor *tensor,
                            unsigned op_idx,
                            bool write)
{
   unsigned idx = ethosu_find_interval(intervals, count, tensor);
   unsigned root = ethosu_interval_root(intervals, idx);

   intervals[root].first_use = MIN2(intervals[root].first_use, op_idx);
   intervals[root].last_use = MAX2(intervals[root].last_use, op_idx);

   if (write)
      intervals[root].has_producer = true;
   else
      intervals[root].has_consumer = true;
}

static unsigned
ethosu_alloc_io_block(struct ethosu_io_free_block *free_blocks,
                      unsigned *free_count, unsigned *high_water,
                      unsigned size)
{
   for (unsigned i = 0; i < *free_count; i++) {
      if (free_blocks[i].size >= size) {
         unsigned offset = free_blocks[i].offset;

         free_blocks[i].offset += size;
         free_blocks[i].size -= size;
         if (free_blocks[i].size == 0) {
            memmove(&free_blocks[i], &free_blocks[i + 1],
                    (*free_count - i - 1) * sizeof(free_blocks[0]));
            (*free_count)--;
         }

         return offset;
      }
   }

   unsigned offset = *high_water;
   *high_water += size;
   return offset;
}

static void
ethosu_free_io_block(struct ethosu_io_free_block *free_blocks,
                     unsigned *free_count, unsigned offset, unsigned size)
{
   unsigned insert = 0;

   while (insert < *free_count && free_blocks[insert].offset < offset)
      insert++;

   memmove(&free_blocks[insert + 1], &free_blocks[insert],
           (*free_count - insert) * sizeof(free_blocks[0]));
   free_blocks[insert].offset = offset;
   free_blocks[insert].size = size;
   (*free_count)++;

   if (insert > 0 &&
       free_blocks[insert - 1].offset + free_blocks[insert - 1].size ==
          free_blocks[insert].offset) {
      free_blocks[insert - 1].size += free_blocks[insert].size;
      memmove(&free_blocks[insert], &free_blocks[insert + 1],
              (*free_count - insert - 1) * sizeof(free_blocks[0]));
      (*free_count)--;
      insert--;
   }

   if (insert + 1 < *free_count &&
       free_blocks[insert].offset + free_blocks[insert].size ==
          free_blocks[insert + 1].offset) {
      free_blocks[insert].size += free_blocks[insert + 1].size;
      memmove(&free_blocks[insert + 1], &free_blocks[insert + 2],
              (*free_count - insert - 2) * sizeof(free_blocks[0]));
      (*free_count)--;
   }
}

static void
ethosu_add_feature_map_offset(struct ethosu_feature_map *fm)
{
   if (ethosu_is_feature_map_in_io(fm))
      fm->tiles.addresses[0] += fm->tensor->offset;
}

static void
ethosu_validate_io_feature_map(struct ethosu_feature_map *fm)
{
   unsigned offset;

   if (!ethosu_is_feature_map_in_io(fm))
      return;

   assert(fm->tensor->size > 0);
   assert(fm->tiles.addresses[0] >= fm->tensor->offset);

   offset = fm->tiles.addresses[0] - fm->tensor->offset;
   assert(offset <= fm->tensor->size);
   assert(ethosu_feature_map_span(fm) <= fm->tensor->size - offset);
}

void
ethosu_allocate_tensors(struct ethosu_subgraph *subgraph,
                        const unsigned *required_sizes)
{
   const unsigned op_count =
      util_dynarray_num_elements(&subgraph->operations, struct ethosu_operation);
   const unsigned tensor_count =
      util_dynarray_num_elements(&subgraph->tensors, struct ethosu_tensor);
   struct ethosu_io_interval *intervals;
   struct ethosu_io_free_block *free_blocks;
   unsigned free_count = 0;
   unsigned high_water = 0;

   if (!tensor_count)
      return;

   intervals = calloc(tensor_count, sizeof(*intervals));
   free_blocks = calloc(tensor_count + 1, sizeof(*free_blocks));
   assert(intervals && free_blocks);

   for (unsigned i = 0; i < tensor_count; i++) {
      struct ethosu_tensor *tensor =
         util_dynarray_element(&subgraph->tensors, struct ethosu_tensor, i);

      intervals[i].tensor = tensor;
      intervals[i].parent = i;
      intervals[i].first_use = UINT_MAX;
      intervals[i].last_use = 0;
      intervals[i].size = ALIGN_POT(ethosu_tensor_storage_size(subgraph, required_sizes, tensor), 16);
      intervals[i].offset = UINT_MAX;
      intervals[i].active = false;
      tensor->size = 0;
   }

   for (unsigned i = 0; i < op_count; i++) {
      struct ethosu_operation *operation =
         util_dynarray_element(&subgraph->operations, struct ethosu_operation, i);

      if (operation->type == ETHOSU_OPERATION_TYPE_NONE &&
          operation->ifm.tensor && operation->ofm.tensor) {
         unsigned ifm_idx = ethosu_find_interval(intervals, tensor_count,
                                                 operation->ifm.tensor);
         unsigned ofm_idx = ethosu_find_interval(intervals, tensor_count,
                                                 operation->ofm.tensor);
         ethosu_union_intervals(intervals, ifm_idx, ofm_idx);
      }
   }

   for (unsigned i = 0; i < op_count; i++) {
      struct ethosu_operation *operation =
         util_dynarray_element(&subgraph->operations, struct ethosu_operation, i);

      if (operation->type == ETHOSU_OPERATION_TYPE_NONE) {
         if (operation->ifm.region == IO_REGION && operation->ifm.tensor)
            ethosu_mark_interval_access(intervals, tensor_count,
                                        operation->ifm.tensor, i, false);

         if (operation->ofm.region == IO_REGION && operation->ofm.tensor)
            ethosu_mark_interval_access(intervals, tensor_count,
                                        operation->ofm.tensor, i, true);

         continue;
      }

      if (operation->ifm.region == IO_REGION && operation->ifm.tensor)
         ethosu_mark_interval_access(intervals, tensor_count,
                                     operation->ifm.tensor, i, false);

      if (operation->ifm2.region == IO_REGION && operation->ifm2.tensor)
         ethosu_mark_interval_access(intervals, tensor_count,
                                     operation->ifm2.tensor, i, false);

      if (operation->ofm.region == IO_REGION && operation->ofm.tensor)
         ethosu_mark_interval_access(intervals, tensor_count,
                                     operation->ofm.tensor, i, true);
   }

   for (unsigned i = 0; i < tensor_count; i++) {
      unsigned root = ethosu_interval_root(intervals, i);

      if (root == i)
         continue;

      intervals[root].first_use =
         MIN2(intervals[root].first_use, intervals[i].first_use);
      intervals[root].last_use =
         MAX2(intervals[root].last_use, intervals[i].last_use);
      intervals[root].size = MAX2(intervals[root].size, intervals[i].size);
      intervals[root].has_producer |= intervals[i].has_producer;
      intervals[root].has_consumer |= intervals[i].has_consumer;
   }

   for (unsigned i = 0; i < tensor_count; i++) {
      if (ethosu_interval_root(intervals, i) != i ||
          intervals[i].first_use == UINT_MAX)
         continue;

      if (!intervals[i].has_producer && intervals[i].has_consumer)
         intervals[i].first_use = 0;

      if (intervals[i].has_producer && !intervals[i].has_consumer)
         intervals[i].last_use = op_count;
   }

   for (unsigned op_idx = 0; op_idx <= op_count; op_idx++) {
      for (unsigned i = 0; i < tensor_count; i++) {
         if (ethosu_interval_root(intervals, i) != i ||
             intervals[i].offset == UINT_MAX ||
             !intervals[i].active ||
             intervals[i].last_use >= op_idx)
            continue;

         ethosu_free_io_block(free_blocks, &free_count, intervals[i].offset,
                              intervals[i].size);
         intervals[i].active = false;
      }

      for (unsigned i = 0; i < tensor_count; i++) {
         if (ethosu_interval_root(intervals, i) != i ||
             intervals[i].first_use != op_idx)
            continue;

         intervals[i].offset =
            ethosu_alloc_io_block(free_blocks, &free_count, &high_water,
                                  intervals[i].size);
         intervals[i].active = true;
      }
   }

   for (unsigned i = 0; i < tensor_count; i++) {
      unsigned root = ethosu_interval_root(intervals, i);

      if (intervals[root].offset == UINT_MAX) {
         intervals[i].tensor->offset = 0;
         intervals[i].tensor->size =
            ethosu_tensor_storage_size(subgraph, required_sizes,
                                       intervals[i].tensor);
      } else {
         intervals[i].tensor->offset = intervals[root].offset;
         intervals[i].tensor->size = ethosu_tensor_storage_size(subgraph, required_sizes,
                                                                intervals[i].tensor);
      }
   }

   util_dynarray_foreach (&subgraph->operations, struct ethosu_operation, operation) {
      if (operation->type == ETHOSU_OPERATION_TYPE_NONE) {
         ethosu_add_feature_map_offset(&operation->ifm);
         operation->ofm.tiles.addresses[0] = operation->ifm.tiles.addresses[0];
      } else {
         ethosu_add_feature_map_offset(&operation->ifm);
         ethosu_add_feature_map_offset(&operation->ifm2);
         ethosu_add_feature_map_offset(&operation->ofm);
      }

      ethosu_validate_io_feature_map(&operation->ifm);
      ethosu_validate_io_feature_map(&operation->ifm2);
      ethosu_validate_io_feature_map(&operation->ofm);
   }

   subgraph->io_used = high_water;

   free(free_blocks);
   free(intervals);
}
