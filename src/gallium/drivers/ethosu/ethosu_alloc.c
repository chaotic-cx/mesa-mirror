/*
 * Copyright 2026 Arm Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/u_inlines.h"

#include "ethosu_alloc.h"
#include "ethosu_device.h"
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
   bool is_external_output;
};

struct ethosu_io_free_block {
   unsigned offset;
   unsigned size;
};

#define ETHOSU_SRAM_REUSE_DELAY 1

struct ethosu_sram_interval {
   struct ethosu_tensor *tensor;
   unsigned parent;
   unsigned first_use;
   unsigned last_use;
   unsigned size;
   unsigned offset;
   unsigned limit;
   unsigned producers;
   unsigned consumers;
   bool active;
   bool is_external_output;
};

struct ethosu_sram_coefficient {
   struct ethosu_operation *dma;
   struct ethosu_operation *conv;
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
   intervals[a].is_external_output |= intervals[b].is_external_output;
}

static bool
ethosu_feature_maps_match(const struct ethosu_subgraph *subgraph,
                          const unsigned *required_sizes,
                          const struct ethosu_feature_map *ifm,
                          const struct ethosu_feature_map *ofm)
{
   if (!ethosu_is_feature_map_in_io(ifm) || !ethosu_is_feature_map_in_io(ofm))
      return false;

   return ifm->tensor->layout == ofm->tensor->layout &&
          ifm->tensor->type_size == ofm->tensor->type_size &&
          ifm->precision == ofm->precision && ifm->is_signed == ofm->is_signed &&
          ifm->shape.height == ofm->shape.height &&
          ifm->shape.width == ofm->shape.width &&
          ifm->shape.depth == ofm->shape.depth &&
          ifm->stride.x == ofm->stride.x && ifm->stride.y == ofm->stride.y &&
          ifm->stride.c == ofm->stride.c &&
          ifm->tiles.height_0 == ofm->tiles.height_0 &&
          ifm->tiles.height_1 == ofm->tiles.height_1 &&
          ifm->tiles.width_0 == ofm->tiles.width_0 &&
          ethosu_feature_map_span(ifm) == ethosu_feature_map_span(ofm) &&
          ethosu_tensor_storage_size(subgraph, required_sizes, ifm->tensor) ==
             ethosu_tensor_storage_size(subgraph, required_sizes, ofm->tensor);
}

static bool
ethosu_operation_is_one_to_one(const struct ethosu_operation *operation)
{
   if (operation->type == ETHOSU_OPERATION_TYPE_ELTWISE)
      return true;

   if (operation->type != ETHOSU_OPERATION_TYPE_POOLING)
      return false;

   if (operation->pooling.nop)
      return true;

   return operation->kernel.height == 1 && operation->kernel.width == 1 &&
          operation->kernel.stride_y == 1 && operation->kernel.stride_x == 1 &&
          operation->pad.top == 0 && operation->pad.bottom == 0 &&
          operation->pad.left == 0 && operation->pad.right == 0 &&
          operation->upscale == ETHOSU_UPSCALE_NONE;
}

static void
ethosu_tensor_operation_counts(struct ethosu_subgraph *subgraph,
                               const struct ethosu_tensor *tensor,
                               unsigned *producers, unsigned *consumers)
{
   *producers = 0;
   *consumers = 0;

   util_dynarray_foreach (&subgraph->operations, struct ethosu_operation, operation) {
      if (operation->type == ETHOSU_OPERATION_TYPE_DMA)
         continue;

      if (operation->ofm.tensor == tensor)
         (*producers)++;

      if (operation->ifm.tensor == tensor)
         (*consumers)++;

      if (operation->ifm2.tensor == tensor)
         (*consumers)++;
   }
}

static bool
ethosu_can_reuse_ifm(struct ethosu_subgraph *subgraph,
                     const unsigned *required_sizes,
                     struct ethosu_io_interval *intervals,
                     unsigned interval_count,
                     const struct ethosu_operation *operation,
                     const struct ethosu_feature_map *ifm)
{
   unsigned ifm_producers, ifm_consumers;
   unsigned ofm_producers, ofm_consumers;
   unsigned ifm_idx;

   /* Elementwise OFM writes can overtake reads from IFM2.  Reusing the
    * second input would overwrite data needed by a later block. */
   if (ifm == &operation->ifm2)
      return false;

   if (!ethosu_operation_is_one_to_one(operation) ||
       !ethosu_feature_maps_match(subgraph, required_sizes, ifm, &operation->ofm) ||
       ifm->tensor == operation->ofm.tensor)
      return false;

   ifm_idx = ethosu_find_interval(intervals, interval_count, ifm->tensor);
   if (intervals[ethosu_interval_root(intervals, ifm_idx)].is_external_output)
      return false;

   ethosu_tensor_operation_counts(subgraph, ifm->tensor,
                                  &ifm_producers, &ifm_consumers);
   ethosu_tensor_operation_counts(subgraph, operation->ofm.tensor,
                                  &ofm_producers, &ofm_consumers);

   return ifm_consumers == 1 && ofm_producers == 1;
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

static bool
ethosu_alloc_sram_block(struct ethosu_io_free_block *free_blocks,
                        unsigned *free_count, unsigned *high_water,
                        unsigned limit, unsigned size, unsigned *offset)
{
   if (size > limit)
      return false;

   for (unsigned i = 0; i < *free_count; i++) {
      if (free_blocks[i].size >= size &&
          free_blocks[i].offset <= limit - size) {
         *offset = free_blocks[i].offset;
         free_blocks[i].offset += size;
         free_blocks[i].size -= size;
         if (free_blocks[i].size == 0) {
            memmove(&free_blocks[i], &free_blocks[i + 1],
                    (*free_count - i - 1) * sizeof(free_blocks[0]));
            (*free_count)--;
         }

         return true;
      }
   }

   if (*high_water > limit || size > limit - *high_water)
      return false;

   *offset = *high_water;
   *high_water += size;
   return true;
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

unsigned
ethosu_coefficient_size(const struct ethosu_operation *operation)
{
   assert(operation->type == ETHOSU_OPERATION_TYPE_CONVOLUTION);

   return ALIGN_POT(operation->conv.scales.size + operation->conv.weights.size,
                    16);
}

static void
ethosu_allocate_sram_coefficients(struct ethosu_subgraph *subgraph)
{
   struct ethosu_ml_device *device = ethosu_ml_device(subgraph->base.device);
   unsigned sram_size = device->sram_size;
   unsigned op_count =
      util_dynarray_num_elements(&subgraph->operations, struct ethosu_operation);
   struct ethosu_sram_coefficient *coefficients;
   unsigned coefficient_count = 0;
   bool double_buffered = true;

   if (!op_count)
      return;

   coefficients = calloc(op_count, sizeof(*coefficients));
   assert(coefficients);

   for (unsigned i = 0; i < op_count; i++) {
      struct ethosu_operation *dma =
         util_dynarray_element(&subgraph->operations, struct ethosu_operation, i);
      struct ethosu_operation *conv = NULL;

      if (dma->type != ETHOSU_OPERATION_TYPE_DMA ||
          dma->dma.dst_region != SCRATCH_REGION)
         continue;

      /* A fused LUT upload is emitted after the coefficient DMA, before the
       * convolution that consumes both.  It does not break the association
       * between the coefficient DMA and that convolution. */
      for (unsigned j = i + 1; j < op_count; j++) {
         struct ethosu_operation *operation =
            util_dynarray_element(&subgraph->operations,
                                  struct ethosu_operation, j);

         if (operation->type == ETHOSU_OPERATION_TYPE_NONE)
            continue;

         if (operation->type == ETHOSU_OPERATION_TYPE_DMA) {
            if (operation->dma.dst_region == LUT_REGION)
               continue;

            break;
         }

         if (operation->type == ETHOSU_OPERATION_TYPE_CONVOLUTION)
            conv = operation;

         break;
      }

      if (!conv)
         continue;

      unsigned size = ethosu_coefficient_size(conv);

      assert(size <= sram_size);

      coefficients[coefficient_count++] = (struct ethosu_sram_coefficient){
         .dma = dma,
         .conv = conv,
         .size = size,
      };
   }

   for (unsigned i = 0; i < coefficient_count; i++) {
      unsigned adjacent_size = 0;

      if (i > 0)
         adjacent_size = coefficients[i - 1].size;
      if (i + 1 < coefficient_count)
         adjacent_size = MAX2(adjacent_size, coefficients[i + 1].size);

      if (coefficients[i].size + adjacent_size > sram_size) {
         double_buffered = false;
         break;
      }
   }

   for (unsigned i = 0; i < coefficient_count; i++) {
      struct ethosu_sram_coefficient *coefficient = &coefficients[i];
      unsigned sram_address;

      if (!double_buffered) {
         sram_address = sram_size - coefficient->size;
      } else if (i & 1) {
         sram_address = sram_size - coefficient->size;
      } else {
         unsigned adjacent_size = 0;

         if (i > 0)
            adjacent_size = coefficients[i - 1].size;
         if (i + 1 < coefficient_count)
            adjacent_size = MAX2(adjacent_size, coefficients[i + 1].size);

         sram_address = sram_size - coefficient->size - adjacent_size;
      }

      coefficient->dma->dma.dst_address = sram_address;
      coefficient->dma->dma.dst_region = SCRATCH_REGION;

      coefficient->conv->conv.scales.region = SCRATCH_REGION;
      coefficient->conv->conv.scales.address = sram_address;

      coefficient->conv->conv.weights.region = SCRATCH_REGION;
      coefficient->conv->conv.weights.address =
         coefficient->conv->conv.scales.address +
         coefficient->conv->conv.scales.size;
   }

   /* Unrolled convolutions share the coefficient upload of their owner. */
   struct ethosu_operation *owner = NULL;
   util_dynarray_foreach (&subgraph->operations, struct ethosu_operation,
                          operation) {
      if (operation->type != ETHOSU_OPERATION_TYPE_CONVOLUTION)
         continue;

      if (!operation->reuses_coefs) {
         owner = operation;
         continue;
      }

      assert(owner);
      operation->conv.scales = owner->conv.scales;
      operation->conv.weights = owner->conv.weights;
   }

   free(coefficients);
}

static unsigned
ethosu_find_sram_interval(struct ethosu_sram_interval *intervals,
                          unsigned count, struct ethosu_tensor *tensor)
{
   for (unsigned i = 0; i < count; i++) {
      if (intervals[i].tensor == tensor)
         return i;
   }

   UNREACHABLE("Tensor missing from SRAM allocator");
}

static unsigned
ethosu_sram_interval_root(struct ethosu_sram_interval *intervals,
                          unsigned idx)
{
   while (intervals[idx].parent != idx) {
      intervals[idx].parent = intervals[intervals[idx].parent].parent;
      idx = intervals[idx].parent;
   }

   return idx;
}

static void
ethosu_union_sram_intervals(struct ethosu_sram_interval *intervals,
                            unsigned a, unsigned b)
{
   a = ethosu_sram_interval_root(intervals, a);
   b = ethosu_sram_interval_root(intervals, b);

   if (a == b)
      return;

   intervals[b].parent = a;
   intervals[a].size = MAX2(intervals[a].size, intervals[b].size);
   intervals[a].is_external_output |= intervals[b].is_external_output;
}

static void
ethosu_mark_sram_interval_access(struct ethosu_sram_interval *intervals,
                                 unsigned count,
                                 const struct ethosu_feature_map *fm,
                                 unsigned op_idx, bool write)
{
   unsigned idx;

   if (!ethosu_is_feature_map_in_io(fm))
      return;

   idx = ethosu_find_sram_interval(intervals, count, fm->tensor);
   idx = ethosu_sram_interval_root(intervals, idx);
   intervals[idx].first_use = MIN2(intervals[idx].first_use, op_idx);
   intervals[idx].last_use = MAX2(intervals[idx].last_use, op_idx);
   if (write)
      intervals[idx].producers++;
   else
      intervals[idx].consumers++;
}

static void
ethosu_mark_sram_feature_map(struct ethosu_sram_interval *intervals,
                             unsigned count, struct ethosu_feature_map *fm)
{
   unsigned idx;

   if (!ethosu_is_feature_map_in_io(fm))
      return;

   idx = ethosu_find_sram_interval(intervals, count, fm->tensor);
   idx = ethosu_sram_interval_root(intervals, idx);
   if (intervals[idx].offset == UINT_MAX)
      return;

   fm->region = SCRATCH_REGION;
   fm->tiles.addresses[0] += intervals[idx].offset;
}

static void
ethosu_allocate_sram_feature_maps(struct ethosu_subgraph *subgraph,
                                  const unsigned *required_sizes)
{
   const unsigned op_count =
      util_dynarray_num_elements(&subgraph->operations,
                                 struct ethosu_operation);
   const unsigned tensor_count =
      util_dynarray_num_elements(&subgraph->tensors, struct ethosu_tensor);
   struct ethosu_sram_interval *intervals;
   struct ethosu_io_free_block *free_blocks;
   unsigned *coefficient_limits;
   unsigned free_count = 0;
   unsigned high_water = 0;
   unsigned npu_index = 0;
   unsigned npu_count = 0;
   unsigned sram_size = ethosu_ml_device(subgraph->base.device)->sram_size;

   if (!op_count || !tensor_count)
      return;

   util_dynarray_foreach (&subgraph->operations, struct ethosu_operation,
                          operation) {
      if (operation->type != ETHOSU_OPERATION_TYPE_NONE &&
          operation->type != ETHOSU_OPERATION_TYPE_DMA)
         npu_count++;
   }

   if (!npu_count)
      return;

   intervals = calloc(tensor_count, sizeof(*intervals));
   free_blocks = calloc(tensor_count + 1, sizeof(*free_blocks));
   assert(intervals && free_blocks);

   coefficient_limits = malloc(npu_count * sizeof(*coefficient_limits));
   assert(coefficient_limits);
   for (unsigned i = 0; i < npu_count; i++)
      coefficient_limits[i] = sram_size;

   for (unsigned i = 0; i < tensor_count; i++) {
      struct ethosu_tensor *tensor =
         util_dynarray_element(&subgraph->tensors, struct ethosu_tensor, i);

      intervals[i].tensor = tensor;
      intervals[i].parent = i;
      intervals[i].first_use = UINT_MAX;
      intervals[i].offset = UINT_MAX;
      intervals[i].size =
         ALIGN_POT(ethosu_tensor_storage_size(subgraph, required_sizes, tensor),
                   16);
      intervals[i].is_external_output = tensor->is_external_output;
   }

   for (unsigned i = 0; i < op_count; i++) {
      struct ethosu_operation *operation =
         util_dynarray_element(&subgraph->operations,
                               struct ethosu_operation, i);

      if (operation->type != ETHOSU_OPERATION_TYPE_NONE ||
          !operation->ifm.tensor || !operation->ofm.tensor)
         continue;

      ethosu_union_sram_intervals(
         intervals,
         ethosu_find_sram_interval(intervals, tensor_count,
                                   operation->ifm.tensor),
         ethosu_find_sram_interval(intervals, tensor_count,
                                   operation->ofm.tensor));
   }

   for (unsigned i = 0; i < tensor_count; i++) {
      unsigned root = ethosu_sram_interval_root(intervals, i);

      if (root == i)
         continue;

      intervals[root].size = MAX2(intervals[root].size, intervals[i].size);
      intervals[root].is_external_output |= intervals[i].is_external_output;
   }

   for (unsigned i = 0; i < op_count; i++) {
      struct ethosu_operation *operation =
         util_dynarray_element(&subgraph->operations,
                               struct ethosu_operation, i);

      if (operation->type == ETHOSU_OPERATION_TYPE_NONE ||
          operation->type == ETHOSU_OPERATION_TYPE_DMA)
         continue;

      if (operation->type == ETHOSU_OPERATION_TYPE_CONVOLUTION &&
          operation->conv.scales.region == SCRATCH_REGION) {
         coefficient_limits[npu_index] =
            MIN2(coefficient_limits[npu_index], operation->conv.scales.address);
         if (npu_index > 0)
            coefficient_limits[npu_index - 1] =
               MIN2(coefficient_limits[npu_index - 1],
                    operation->conv.scales.address);
      }

      ethosu_mark_sram_interval_access(intervals, tensor_count,
                                       &operation->ifm, npu_index, false);
      ethosu_mark_sram_interval_access(intervals, tensor_count,
                                       &operation->ifm2, npu_index, false);
      ethosu_mark_sram_interval_access(intervals, tensor_count,
                                       &operation->ofm, npu_index, true);
      npu_index++;
   }

   for (unsigned i = 0; i < tensor_count; i++) {
      struct ethosu_sram_interval *interval = &intervals[i];
      unsigned last_use;

      if (ethosu_sram_interval_root(intervals, i) != i ||
          interval->first_use == UINT_MAX)
         continue;

      last_use = MIN2(interval->last_use + ETHOSU_SRAM_REUSE_DELAY,
                      npu_count - 1);
      interval->limit = sram_size;
      for (unsigned time = interval->first_use; time <= last_use; time++)
         interval->limit = MIN2(interval->limit,
                                coefficient_limits[time]);
   }

   for (unsigned time = 0; time < npu_count; time++) {
      for (unsigned i = 0; i < tensor_count; i++) {
         struct ethosu_sram_interval *interval = &intervals[i];

         if (ethosu_sram_interval_root(intervals, i) != i ||
             !interval->active ||
             interval->last_use + ETHOSU_SRAM_REUSE_DELAY >= time)
            continue;

         ethosu_free_io_block(free_blocks, &free_count, interval->offset,
                              interval->size);
         interval->active = false;
      }

      for (unsigned i = 0; i < tensor_count; i++) {
         struct ethosu_sram_interval *interval = &intervals[i];
         unsigned offset;

         if (ethosu_sram_interval_root(intervals, i) != i ||
             interval->first_use != time || interval->is_external_output ||
             interval->producers != 1 || !interval->consumers)
            continue;

         if (!ethosu_alloc_sram_block(free_blocks, &free_count, &high_water,
                                      interval->limit, interval->size,
                                      &offset))
            continue;

         interval->offset = offset;
         interval->active = true;
      }
   }

   util_dynarray_foreach (&subgraph->operations, struct ethosu_operation,
                          operation) {
      ethosu_mark_sram_feature_map(intervals, tensor_count, &operation->ifm);
      ethosu_mark_sram_feature_map(intervals, tensor_count, &operation->ifm2);
      ethosu_mark_sram_feature_map(intervals, tensor_count, &operation->ofm);
   }

   free(free_blocks);
   free(intervals);
   free(coefficient_limits);
}

void
ethosu_allocate_tensors(struct ethosu_subgraph *subgraph,
                        const unsigned *required_sizes)
{
   ethosu_allocate_sram_coefficients(subgraph);
   ethosu_allocate_sram_feature_maps(subgraph, required_sizes);

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
      intervals[i].is_external_output = tensor->is_external_output;
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

      if (!DBG_ENABLED(ETHOSU_DBG_DISABLE_IO_REUSE) &&
          ethosu_can_reuse_ifm(subgraph, required_sizes, intervals, tensor_count, operation,
                               &operation->ifm)) {
         unsigned ifm_idx = ethosu_find_interval(intervals, tensor_count,
                                                 operation->ifm.tensor);
         unsigned ofm_idx = ethosu_find_interval(intervals, tensor_count,
                                                 operation->ofm.tensor);
         ethosu_union_intervals(intervals, ifm_idx, ofm_idx);
      } else if (!DBG_ENABLED(ETHOSU_DBG_DISABLE_IO_REUSE) &&
                 ethosu_can_reuse_ifm(subgraph, required_sizes, intervals, tensor_count,
                                      operation, &operation->ifm2)) {
         unsigned ifm_idx = ethosu_find_interval(intervals, tensor_count,
                                                 operation->ifm2.tensor);
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

      if (intervals[i].is_external_output)
         intervals[i].last_use = op_count;

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

         if (!DBG_ENABLED(ETHOSU_DBG_DISABLE_IO_REUSE))
            ethosu_free_io_block(free_blocks, &free_count,
                                 intervals[i].offset, intervals[i].size);
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
         operation->ofm.tensor->offset = operation->ifm.tensor->offset;
         operation->ofm.tensor->size = operation->ifm.tensor->size;
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
