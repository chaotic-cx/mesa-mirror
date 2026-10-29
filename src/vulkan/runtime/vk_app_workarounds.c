/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "vk_app_workarounds.h"

#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_queue.h"
#include "vk_util.h"

#include "util/log.h"
#include "util/stack_array.h"

#include <string.h>

/* This layer implements application workarounds that are common to all drivers. */

/* Metro Exodus (Linux native) */
static VKAPI_ATTR VkResult VKAPI_CALL
metro_exodus_GetSemaphoreCounterValue(VkDevice _device, VkSemaphore _semaphore, uint64_t *pValue)
{
   VK_FROM_HANDLE(vk_device, device, _device);

   /* See https://gitlab.freedesktop.org/mesa/mesa/-/issues/5119. */
   if (_semaphore == VK_NULL_HANDLE) {
      mesa_logw("Ignoring vkGetSemaphoreCounterValue() with NULL semaphore (game bug)!");
      return VK_SUCCESS;
   }

   return device->app_workarounds_dispatch_table->GetSemaphoreCounterValue(_device, _semaphore, pValue);
}

static VKAPI_ATTR VkResult VKAPI_CALL
metro_exodus_QueueSubmit(VkQueue _queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence)
{
   VK_FROM_HANDLE(vk_queue, queue, _queue);
   struct vk_device *device = queue->base.device;

   bool has_null_wait_semaphores = false;
   uint32_t n_wait_semaphores = 0;
   for (uint32_t s = 0; s < submitCount; s++) {
      for (uint32_t i = 0; i < pSubmits[s].waitSemaphoreCount; i++)
         has_null_wait_semaphores |= pSubmits[s].pWaitSemaphores[i] == VK_NULL_HANDLE;
      n_wait_semaphores += pSubmits[s].waitSemaphoreCount;
   }

   if (!has_null_wait_semaphores)
      return device->app_workarounds_dispatch_table->QueueSubmit(_queue, submitCount, pSubmits, fence);

   /* Metro Exodus sometimes passes VK_NULL_HANDLE in pWaitSemaphores (game bug).
    * See https://gitlab.freedesktop.org/mesa/mesa/-/work_items/10018.
    */
   mesa_logw("Ignoring NULL wait semaphores in vkQueueSubmit() (game bug)!");

   STACK_ARRAY(VkSubmitInfo, submit_info, submitCount);
   STACK_ARRAY(VkTimelineSemaphoreSubmitInfo, timeline_submit_info, submitCount);
   STACK_ARRAY(VkDeviceGroupSubmitInfo, group_submit_info, submitCount);
   STACK_ARRAY(VkProtectedSubmitInfo, protected_submit_info, submitCount);
   STACK_ARRAY(VkPerformanceQuerySubmitInfoKHR, perf_query_submit_info, submitCount);

   STACK_ARRAY(VkSemaphore, wait_semaphores, n_wait_semaphores);
   STACK_ARRAY(VkPipelineStageFlags, wait_dst_stage_masks, n_wait_semaphores);
   STACK_ARRAY(uint64_t, wait_semaphore_values, n_wait_semaphores);
   STACK_ARRAY(uint32_t, wait_semaphore_device_indices, n_wait_semaphores);

   n_wait_semaphores = 0;

   for (uint32_t s = 0; s < submitCount; s++) {
      const VkTimelineSemaphoreSubmitInfo *timeline_info =
         vk_find_struct_const(pSubmits[s].pNext, TIMELINE_SEMAPHORE_SUBMIT_INFO);
      const VkDeviceGroupSubmitInfo *group_info =
         vk_find_struct_const(pSubmits[s].pNext, DEVICE_GROUP_SUBMIT_INFO);

      uint32_t wait_count = 0;
      for (uint32_t i = 0; i < pSubmits[s].waitSemaphoreCount; i++) {
         if (pSubmits[s].pWaitSemaphores[i] == VK_NULL_HANDLE)
            continue;

         const uint32_t w = n_wait_semaphores + wait_count++;

         wait_semaphores[w] = pSubmits[s].pWaitSemaphores[i];
         wait_dst_stage_masks[w] = pSubmits[s].pWaitDstStageMask[i];
         wait_semaphore_values[w] =
            timeline_info && i < timeline_info->waitSemaphoreValueCount ? timeline_info->pWaitSemaphoreValues[i] : 0;
         wait_semaphore_device_indices[w] = group_info ? group_info->pWaitSemaphoreDeviceIndices[i] : 0;
      }

      submit_info[s] = pSubmits[s];
      submit_info[s].pNext = NULL;
      submit_info[s].waitSemaphoreCount = wait_count;
      submit_info[s].pWaitSemaphores = &wait_semaphores[n_wait_semaphores];
      submit_info[s].pWaitDstStageMask = &wait_dst_stage_masks[n_wait_semaphores];

      if (group_info) {
         group_submit_info[s] = *group_info;
         group_submit_info[s].pNext = submit_info[s].pNext;
         group_submit_info[s].waitSemaphoreCount = wait_count;
         group_submit_info[s].pWaitSemaphoreDeviceIndices = &wait_semaphore_device_indices[n_wait_semaphores];
         submit_info[s].pNext = &group_submit_info[s];
      }

      if (timeline_info) {
         timeline_submit_info[s] = *timeline_info;
         timeline_submit_info[s].pNext = submit_info[s].pNext;
         timeline_submit_info[s].waitSemaphoreValueCount = timeline_info->waitSemaphoreValueCount ? wait_count : 0;
         timeline_submit_info[s].pWaitSemaphoreValues = &wait_semaphore_values[n_wait_semaphores];
         submit_info[s].pNext = &timeline_submit_info[s];
      }

      const VkProtectedSubmitInfo *protected_info =
         vk_find_struct_const(pSubmits[s].pNext, PROTECTED_SUBMIT_INFO);
      if (protected_info) {
         protected_submit_info[s] = *protected_info;
         protected_submit_info[s].pNext = NULL;
         __vk_append_struct(&submit_info[s], &protected_submit_info[s]);
      }

      const VkPerformanceQuerySubmitInfoKHR *query_info =
         vk_find_struct_const(pSubmits[s].pNext, PERFORMANCE_QUERY_SUBMIT_INFO_KHR);
      if (query_info) {
         perf_query_submit_info[s] = *query_info;
         perf_query_submit_info[s].pNext = NULL;
         __vk_append_struct(&submit_info[s], &perf_query_submit_info[s]);
      }

      n_wait_semaphores += wait_count;
   }

   VkResult result = device->app_workarounds_dispatch_table->QueueSubmit(_queue, submitCount, submit_info, fence);

   STACK_ARRAY_FINISH(wait_semaphores);
   STACK_ARRAY_FINISH(wait_dst_stage_masks);
   STACK_ARRAY_FINISH(wait_semaphore_values);
   STACK_ARRAY_FINISH(wait_semaphore_device_indices);
   STACK_ARRAY_FINISH(submit_info);
   STACK_ARRAY_FINISH(timeline_submit_info);
   STACK_ARRAY_FINISH(group_submit_info);
   STACK_ARRAY_FINISH(protected_submit_info);
   STACK_ARRAY_FINISH(perf_query_submit_info);

   return result;
}

VkResult
vk_device_init_app_workarounds(struct vk_device *device, const char *app_layer)
{
   struct vk_device_dispatch_table *table = &device->dispatch_table;

   /* Must be called only once, after the driver has set up device->dispatch_table. */
   assert(!device->app_workarounds_dispatch_table);

   if (!app_layer || !app_layer[0])
      return VK_SUCCESS;

   device->app_workarounds_dispatch_table =
      vk_alloc(&device->alloc, sizeof(*device->app_workarounds_dispatch_table), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device->app_workarounds_dispatch_table)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* The workarounds call down to the driver through this copy. */
   *device->app_workarounds_dispatch_table = device->dispatch_table;

#define SET_ENTRYPOINT(app_layer, entrypoint) table->entrypoint = app_layer##_##entrypoint;

   if (!strcmp(app_layer, "metroexodus")) {
      SET_ENTRYPOINT(metro_exodus, GetSemaphoreCounterValue);
      SET_ENTRYPOINT(metro_exodus, QueueSubmit);
   } else {
      mesa_logw("Unknown vk_app_layer '%s', ignoring", app_layer);
      vk_free(&device->alloc, device->app_workarounds_dispatch_table);
      device->app_workarounds_dispatch_table = NULL;
   }

#undef SET_ENTRYPOINT

   return VK_SUCCESS;
}
