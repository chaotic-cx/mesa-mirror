/* Based on anv:
 * Copyright © 2015 Intel Corporation
 *
 * Copyright © 2016 Red Hat Inc.
 * Copyright © 2025 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir/radv_meta_nir.h"
#include "tools/radv_debug_hang.h"
#include "radv_cp_dma.h"
#include "radv_meta.h"
#include "radv_sdma.h"
#include "radv_tracepoints.h"
#include "vk_shader_module.h"

#include "radv_cs.h"

#include "ac_nir_meta.h"
#include "vk_command_pool.h"

struct radv_fill_or_copy_memory_key {
   enum radv_meta_object_key_type type;
   ac_cs_clear_copy_buffer_key ac;
};

static VkResult
get_fill_or_copy_memory_pipeline(struct radv_device *device, const ac_cs_clear_copy_buffer_options *const options,
                                 const ac_cs_clear_copy_buffer_dispatch *const dispatch, VkPipeline *pipeline_out,
                                 VkPipelineLayout *layout_out)
{
   struct radv_fill_or_copy_memory_key key;
   VkResult result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_FILL_OR_COPY_MEMORY;
   key.ac.key = dispatch->shader_key.key;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = dispatch->num_user_data * sizeof(uint32_t),
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                        layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = ac_create_clear_copy_buffer_cs(options, &key.ac);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static ac_cs_clear_copy_buffer_options
radv_clear_copy_buffer_options(const struct radv_cmd_buffer *const cmd_buffer)
{
   const struct radv_device *const device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *const pdev = radv_device_physical(device);
   const ac_cs_clear_copy_buffer_options options = {
      .info = &pdev->info,
      .nir_options = &device->compiler_info.nir_options[MESA_SHADER_COMPUTE],
      .addr_user_data = true,
      .is_compute_queue = cmd_buffer->qf == RADV_QUEUE_COMPUTE,
      /* Don't fail if CP DMA can't use L2. We prefer to use the compute path in that case.
       * CP DMA (the fallback) doesn't support conditional rendering.
       */
      .fail_if_slow =
         (pdev->drirc.performance.buffer_meta_path == RADV_BUFFER_META_PATH_AUTO && pdev->info.cp_dma_use_L2) &&
         (!cmd_buffer->state.cond_render.enabled || cmd_buffer->state.cond_render.suspended),
   };

   return options;
}

static ac_cs_clear_copy_buffer_info
radv_clear_copy_buffer_info(const struct radv_cmd_buffer *const cmd_buffer, const uint64_t src_va,
                            const uint64_t dst_va, const uint64_t size, const VkAddressCopyFlagsKHR src_copy_flags,
                            const VkAddressCopyFlagsKHR dst_copy_flags)
{
   const struct radv_device *const device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *const pdev = radv_device_physical(device);
   const ac_cs_clear_copy_buffer_info info = {
      .dst_offset = dst_va,
      .src_offset = src_va,
      .size = size,
      .dwords_per_thread = radv_buffer_meta_path_get_dwords_per_thread(pdev->drirc.performance.buffer_meta_path),
      .dst_is_vram = dst_copy_flags & VK_ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR,
      .src_is_vram = src_copy_flags & VK_ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR,
      .dst_is_sparse = dst_copy_flags & VK_ADDRESS_COPY_SPARSE_BIT_KHR,
      .src_is_sparse = src_copy_flags & VK_ADDRESS_COPY_SPARSE_BIT_KHR,
   };

   return info;
}

static void
radv_compute_fill_memory(struct radv_cmd_buffer *cmd_buffer, const ac_cs_clear_copy_buffer_options *const options,
                         const ac_cs_clear_copy_buffer_dispatch *const dispatch)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result = get_fill_or_copy_memory_pipeline(device, options, dispatch, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, dispatch->num_user_data * 4,
                            dispatch->user_data);

   radv_unaligned_dispatch(cmd_buffer, dispatch->num_threads, 1, 1);
}

static void
radv_compute_copy_memory(struct radv_cmd_buffer *cmd_buffer, const ac_cs_clear_copy_buffer_options *const options,
                         const ac_cs_clear_copy_buffer_dispatch *const dispatch, const uint64_t size)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result = get_fill_or_copy_memory_pipeline(device, options, dispatch, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_utrace_begin_compute_copy_memory(cmd_buffer, size);

   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, dispatch->num_user_data * 4,
                            dispatch->user_data);

   radv_unaligned_dispatch(cmd_buffer, dispatch->num_threads, 1, 1);

   radv_utrace_end_compute_copy_memory(cmd_buffer);
}

static bool
radv_prepare_cs_clear_copy_buffer(const struct radv_cmd_buffer *cmd_buffer,
                                  const ac_cs_clear_copy_buffer_options *options,
                                  const ac_cs_clear_copy_buffer_info *info, ac_cs_clear_copy_buffer_dispatch *out)
{
   const struct radv_device *const device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *const pdev = radv_device_physical(device);
   const bool use_compute = ac_prepare_cs_clear_copy_buffer(options, info, out);

   if (pdev->drirc.performance.buffer_meta_path == RADV_BUFFER_META_PATH_CP_DMA && out->cpdma_supported &&
       (!cmd_buffer->state.cond_render.enabled || cmd_buffer->state.cond_render.suspended))
      return false;

   assert(use_compute || !radv_buffer_meta_path_forces_compute(pdev->drirc.performance.buffer_meta_path));
   assert(use_compute || out->cpdma_supported);

   return use_compute;
}

static uint32_t
radv_fill_memory_internal(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image, uint64_t dst_va,
                          uint64_t size, uint32_t value, VkAddressCopyFlagsKHR dst_copy_flags, bool cp_coherent)
{
   assert(!(dst_va & 3));
   assert(!(size & 3));

   const struct radv_device *const device = radv_cmd_buffer_device(cmd_buffer);

   if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
      radv_sdma_fill_memory(device, cmd_buffer->cs, dst_va, size, value);
      return 0;
   }

   const ac_cs_clear_copy_buffer_options options = radv_clear_copy_buffer_options(cmd_buffer);
   ac_cs_clear_copy_buffer_dispatch dispatch = {0};
   ac_cs_clear_copy_buffer_info info = radv_clear_copy_buffer_info(cmd_buffer, 0, dst_va, size, 0, dst_copy_flags);

   info.clear_value[0] = value;
   info.clear_value_size = sizeof(uint32_t);

   const bool use_compute = radv_prepare_cs_clear_copy_buffer(cmd_buffer, &options, &info, &dispatch);
   uint32_t flush_bits = 0;

   if (use_compute) {
      radv_compute_fill_memory(cmd_buffer, &options, &dispatch);

      flush_bits = AC_BARRIER_SYNC_CS | AC_BARRIER_INV_VMEM |
                   radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_WRITE_BIT, 0, image, NULL);
   } else if (size) {
      assert(!cmd_buffer->state.cond_render.enabled || cmd_buffer->state.cond_render.suspended);
      radv_cp_dma_fill_memory(cmd_buffer, dst_va, size, value, image || cp_coherent);
      flush_bits = AC_BARRIER_SYNC_CP_DMA;
   }

   return flush_bits;
}

uint32_t
radv_fill_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t va, uint64_t size, uint32_t value,
                 VkAddressCopyFlagsKHR copy_flags, bool cp_coherent)
{
   return radv_fill_memory_internal(cmd_buffer, NULL, va, size, value, copy_flags, cp_coherent);
}

uint32_t
radv_fill_image(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *image, uint64_t offset, uint64_t size,
                uint32_t value)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const uint64_t va = image->bindings[0].addr + offset;
   struct radeon_winsys_bo *bo = image->bindings[0].bo;
   const VkAddressCopyFlagsKHR copy_flags = radv_get_copy_flags_from_bo(bo);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   radv_cs_add_buffer(device->ws, cs->b, bo);

   return radv_fill_memory_internal(cmd_buffer, image, va, size, value, copy_flags, false);
}

uint32_t
radv_fill_buffer(struct radv_cmd_buffer *cmd_buffer, struct radeon_winsys_bo *bo, uint64_t va, uint64_t size,
                 uint32_t value, bool cp_coherent)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const VkAddressCopyFlagsKHR copy_flags = radv_get_copy_flags_from_bo(bo);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   radv_cs_add_buffer(device->ws, cs->b, bo);

   return radv_fill_memory(cmd_buffer, va, size, value, copy_flags, cp_coherent);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize fillSize,
                   uint32_t data)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_buffer, dst_buffer, dstBuffer);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   fillSize = vk_buffer_range(&dst_buffer->vk, dstOffset, fillSize) & ~3ull;

   radv_fill_buffer(cmd_buffer, dst_buffer->bo, vk_buffer_address(&dst_buffer->vk, dstOffset), fillSize, data, false);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdFillMemoryKHR(VkCommandBuffer commandBuffer, const VkDeviceAddressRangeKHR *pDstRange,
                      VkAddressCommandFlagsKHR dstFlags, uint32_t data)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_command_flags(dstFlags);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_fill_memory(cmd_buffer, pDstRange->address, pDstRange->size, data, dst_copy_flags, false);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

void
radv_copy_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t src_va, uint64_t dst_va, uint64_t size,
                 VkAddressCopyFlagsKHR src_copy_flags, VkAddressCopyFlagsKHR dst_copy_flags)
{
   const struct radv_device *const device = radv_cmd_buffer_device(cmd_buffer);

   if (cmd_buffer->qf == RADV_QUEUE_TRANSFER) {
      radv_sdma_copy_memory(device, cmd_buffer->cs, src_va, dst_va, size,
                            cmd_buffer->vk.pool->flags & VK_COMMAND_POOL_CREATE_PROTECTED_BIT);
      return;
   }

   const ac_cs_clear_copy_buffer_options options = radv_clear_copy_buffer_options(cmd_buffer);
   ac_cs_clear_copy_buffer_dispatch dispatch = {0};
   ac_cs_clear_copy_buffer_info info =
      radv_clear_copy_buffer_info(cmd_buffer, src_va, dst_va, size, src_copy_flags, dst_copy_flags);

   const bool use_compute = radv_prepare_cs_clear_copy_buffer(cmd_buffer, &options, &info, &dispatch);

   if (use_compute) {
      radv_compute_copy_memory(cmd_buffer, &options, &dispatch, size);
   } else if (size) {
      assert(!cmd_buffer->state.cond_render.enabled || cmd_buffer->state.cond_render.suspended);
      radv_cp_dma_copy_memory(cmd_buffer, src_va, dst_va, size);
   }
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *pCopyBufferInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_buffer, src_buffer, pCopyBufferInfo->srcBuffer);
   VK_FROM_HANDLE(radv_buffer, dst_buffer, pCopyBufferInfo->dstBuffer);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   const VkAddressCopyFlagsKHR src_copy_flags = radv_get_copy_flags_from_bo(src_buffer->bo);
   const VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_bo(dst_buffer->bo);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_cs_add_buffer(device->ws, cs->b, src_buffer->bo);
   radv_cs_add_buffer(device->ws, cs->b, dst_buffer->bo);

   for (unsigned r = 0; r < pCopyBufferInfo->regionCount; r++) {
      const VkBufferCopy2 *region = &pCopyBufferInfo->pRegions[r];
      const uint64_t src_va = vk_buffer_address(&src_buffer->vk, region->srcOffset);
      const uint64_t dst_va = vk_buffer_address(&dst_buffer->vk, region->dstOffset);

      radv_copy_memory(cmd_buffer, src_va, dst_va, region->size, src_copy_flags, dst_copy_flags);
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdCopyMemoryKHR(VkCommandBuffer commandBuffer, const VkCopyDeviceMemoryInfoKHR *pCopyMemoryInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   for (unsigned r = 0; r < pCopyMemoryInfo->regionCount; r++) {
      const VkDeviceMemoryCopyKHR *region = &pCopyMemoryInfo->pRegions[r];

      VkAddressCopyFlagsKHR src_copy_flags = radv_get_copy_flags_from_command_flags(region->srcFlags);
      VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_command_flags(region->dstFlags);

      radv_copy_memory(cmd_buffer, region->srcRange.address, region->dstRange.address, region->srcRange.size,
                       src_copy_flags, dst_copy_flags);
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

void
radv_update_memory_cp(struct radv_cmd_buffer *cmd_buffer, uint64_t va, const void *data, uint64_t size)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   uint64_t words = size / 4;
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   assert(size < RADV_BUFFER_UPDATE_THRESHOLD);

   radv_emit_cache_flush(cmd_buffer, false);
   radeon_check_space(device->ws, cs->b, words + 4);

   ac_emit_cp_write_data(cs->b, V_371_MICRO_ENGINE, cmd_buffer->is_mec ? V_371_MEMORY : V_371_MEM_GRBM, va, words, data,
                         false);

   if (radv_device_fault_detection_enabled(device))
      radv_cmd_buffer_trace_emit(cmd_buffer);
}

void
radv_update_memory(struct radv_cmd_buffer *cmd_buffer, uint64_t va, uint64_t size, const void *data,
                   VkAddressCopyFlagsKHR dst_copy_flags)
{
   assert(!(size & 3));
   assert(!(va & 3));

   if (!size)
      return;

   if (size < RADV_BUFFER_UPDATE_THRESHOLD && cmd_buffer->qf != RADV_QUEUE_TRANSFER) {
      radv_update_memory_cp(cmd_buffer, va, data, size);
   } else {
      uint32_t buf_offset;

      radv_cmd_buffer_upload_data(cmd_buffer, size, data, &buf_offset);

      const VkAddressCopyFlagsKHR src_copy_flags = radv_get_copy_flags_from_bo(cmd_buffer->upload.upload_bo);
      const uint64_t src_va = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + buf_offset;

      radv_copy_memory(cmd_buffer, src_va, va, size, src_copy_flags, dst_copy_flags);
   }
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize,
                     const void *pData)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_buffer, dst_buffer, dstBuffer);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const uint64_t dst_va = vk_buffer_address(&dst_buffer->vk, dstOffset);
   struct radv_cmd_stream *cs = cmd_buffer->cs;

   const VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_bo(dst_buffer->bo);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_cs_add_buffer(device->ws, cs->b, dst_buffer->bo);

   radv_update_memory(cmd_buffer, dst_va, dataSize, pData, dst_copy_flags);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdUpdateMemoryKHR(VkCommandBuffer commandBuffer, const VkDeviceAddressRangeKHR *pDstRange,
                        VkAddressCommandFlagsKHR dstFlags, VkDeviceSize dataSize, const void *pData)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);

   VkAddressCopyFlagsKHR dst_copy_flags = radv_get_copy_flags_from_command_flags(dstFlags);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   radv_update_memory(cmd_buffer, pDstRange->address, dataSize, pData, dst_copy_flags);

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}
