/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_event.h"

#include "kosmickrisp/bridge/mtl_encoder.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"
#include "kk_bo.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"

#define KK_EVENT_MEM_SIZE sizeof(VkResult)

static_assert(sizeof(uint32_t) == KK_EVENT_MEM_SIZE, "Events are 32 bits");

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateEvent(VkDevice device, const VkEventCreateInfo *pCreateInfo,
               const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   struct kk_event *event;
   VkResult result = VK_SUCCESS;

   event = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*event),
                            VK_OBJECT_TYPE_EVENT);
   if (!event)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* TODO_KOSMICKRISP Bring back the heap. */
   result = kk_alloc_bo(dev, &dev->vk.base, KK_EVENT_MEM_SIZE,
                        KK_EVENT_MEM_SIZE, &event->bo);
   if (result != VK_SUCCESS) {
      vk_object_free(&dev->vk, pAllocator, event);
      return result;
   }

   event->status = event->bo->cpu;
   event->addr = event->bo->gpu;
   *event->status = VK_EVENT_RESET;

   *pEvent = kk_event_to_handle(event);

   return result;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyEvent(VkDevice device, VkEvent _event,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_event, event, _event);

   if (!event)
      return;

   kk_destroy_bo(dev, event->bo);

   vk_object_free(&dev->vk, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_GetEventStatus(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(kk_event, event, _event);

   return *event->status;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_SetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(kk_event, event, _event);

   *event->status = VK_EVENT_SET;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_ResetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(kk_event, event, _event);

   *event->status = VK_EVENT_RESET;

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(kk_event, event, _event);
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   enum mtl_stages after_stages = 0u;
   enum mtl_stages before_stages = 0u;
   vk_dependency_info_to_mtl(pDependencyInfo, &after_stages, &before_stages,
                             dev->vk.enabled_features.conditionalRendering);

   if (pDependencyInfo->dependencyFlags &
          VK_DEPENDENCY_ASYMMETRIC_EVENT_BIT_KHR ||
       before_stages == 0u) {
      /* Second half runs in CmdWaitEvents with the write in the middle. */
      before_stages = MTL_STAGE_DISPATCH;
   }

   after_stages |=
      kk_cmd_add_chained_barrier_stages(cmd, after_stages, before_stages);
   kk_cmd_barrier(cmd, after_stages, before_stages);

   libkk_write_u32(cmd, kk_grid_1d(1), true, event->addr, VK_EVENT_SET);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                  VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(kk_event, event, _event);
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   enum mtl_stages stages = vk_pipeline_stages_to_mtl_stages(
      stageMask, true, dev->vk.enabled_features.conditionalRendering);
   stages |= kk_cmd_add_chained_barrier_stages(cmd, stages, MTL_STAGE_DISPATCH);
   kk_cmd_barrier(cmd, stages, MTL_STAGE_DISPATCH);

   libkk_write_u32(cmd, kk_grid_1d(1), true, event->addr, VK_EVENT_RESET);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount,
                  const VkEvent *pEvents,
                  const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   enum mtl_stages before_stages = 0u;

   for (uint i = 0u; i < eventCount; i++) {
      if (!(pDependencyInfos[i].dependencyFlags &
            VK_DEPENDENCY_ASYMMETRIC_EVENT_BIT_KHR))
         continue;

      enum mtl_stages before_stages_event = 0u;
      vk_dependency_info_to_mtl(&pDependencyInfos[i], NULL,
                                &before_stages_event,
                                dev->vk.enabled_features.conditionalRendering);
      before_stages |= before_stages_event;
   }

   if (before_stages != 0u)
      kk_cmd_barrier(cmd, MTL_STAGE_DISPATCH, before_stages);
}
