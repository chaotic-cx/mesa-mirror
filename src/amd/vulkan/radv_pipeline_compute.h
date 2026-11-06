/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_PIPELINE_COMPUTE_H
#define RADV_PIPELINE_COMPUTE_H

#include "radv_pipeline.h"

struct radv_compute_pipeline {
   struct radv_pipeline base;
};

RADV_DECL_PIPELINE_DOWNCAST(compute, RADV_PIPELINE_COMPUTE)

VkResult radv_compute_pipeline_create(VkDevice _device, VkPipelineCache _cache,
                                      const VkComputePipelineCreateInfo *pCreateInfo,
                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipeline);

void radv_destroy_compute_pipeline(struct radv_device *device, struct radv_compute_pipeline *pipeline);

void radv_compute_pipeline_hash(const struct radv_device *device, const VkComputePipelineCreateInfo *pCreateInfo,
                                unsigned char *hash);

#endif /* RADV_PIPELINE_COMPUTE_H */
