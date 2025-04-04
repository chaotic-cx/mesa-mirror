/*
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: MIT
 *
 * based in part on anv and radv which are:
 * Copyright © 2015 Intel Corporation
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 */

#ifndef VN_QUERY_POOL_H
#define VN_QUERY_POOL_H

#include "vn_common.h"

struct vn_feedback_buffer;

struct vn_query_pool {
   struct vn_object_base base;

   VkAllocationCallbacks allocator;

   uint32_t query_count;
   /* synchronize lazy init of qfb buffer */
   simple_mtx_t mutex;
   /* non-NULL if VN_PERF_NO_QUERY_FEEDBACK is disabled */
   struct vn_feedback_buffer *fb_buf;
   uint32_t result_array_size;
   bool saturate_on_overflow;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(vn_query_pool,
                               base.vk,
                               VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

VkResult
vn_query_feedback_buffer_init_once(struct vn_device *dev,
                                   struct vn_query_pool *pool);

#endif /* VN_QUERY_POOL_H */
