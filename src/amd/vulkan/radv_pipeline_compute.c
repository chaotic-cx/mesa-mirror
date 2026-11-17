/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "meta/radv_meta.h"
#include "tools/radv_rmv.h"
#include "util/mesa-blake3.h"
#include "util/os_time.h"
#include "radv_pipeline_binary.h"
#include "radv_pipeline_cache.h"
#include "radv_shader.h"
#include "radv_shader_compute.h"
#include "vk_pipeline.h"
#include "vk_util.h"

void
radv_compute_pipeline_hash(const struct radv_device *device, const VkComputePipelineCreateInfo *pCreateInfo,
                           unsigned char *hash)
{
   VkPipelineCreateFlags2 create_flags = vk_compute_pipeline_create_flags(pCreateInfo);
   VK_FROM_HANDLE(radv_pipeline_layout, pipeline_layout, pCreateInfo->layout);
   const VkPipelineShaderStageCreateInfo *sinfo = &pCreateInfo->stage;
   blake3_hasher ctx;

   struct radv_shader_stage_key stage_key =
      radv_pipeline_get_shader_key(&device->compiler_info, sinfo, create_flags, pCreateInfo->pNext);

   _mesa_blake3_init(&ctx);
   radv_pipeline_hash(device, pipeline_layout, &ctx);
   radv_pipeline_hash_shader_stage(create_flags, sinfo, &stage_key, &ctx);
   _mesa_blake3_final(&ctx, hash);
}

static VkResult
radv_compute_pipeline_compile(const VkComputePipelineCreateInfo *pCreateInfo, struct radv_compute_pipeline *pipeline,
                              struct radv_pipeline_layout *pipeline_layout, struct radv_device *device,
                              struct vk_pipeline_cache *cache, const VkPipelineShaderStageCreateInfo *pStage,
                              const VkPipelineCreationFeedbackCreateInfo *creation_feedback)
{
   const struct radv_compiler_info *compiler_info = &device->compiler_info;
   const bool skip_shaders_cache = radv_pipeline_skip_shaders_cache(device, &pipeline->base);
   struct radv_shader_stage cs_stage = {0};
   VkPipelineCreationFeedback pipeline_feedback = {
      .flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT,
   };
   VkResult result = VK_SUCCESS;

   int64_t pipeline_start = os_time_get_nano();

   radv_compute_pipeline_hash(device, pCreateInfo, pipeline->base.blake3);

   pipeline->base.pipeline_hash = *(uint64_t *)pipeline->base.blake3;

   bool found_in_application_cache = true;
   if (!skip_shaders_cache &&
       radv_compute_pipeline_cache_search(device, cache, pipeline, &found_in_application_cache)) {
      if (found_in_application_cache)
         pipeline_feedback.flags |= VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;
      result = VK_SUCCESS;
      goto done;
   }

   if (pipeline->base.create_flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)
      return VK_PIPELINE_COMPILE_REQUIRED;

   int64_t stage_start = os_time_get_nano();

   const struct radv_shader_stage_key stage_key =
      radv_pipeline_get_shader_key(compiler_info, &pCreateInfo->stage, pipeline->base.create_flags, pCreateInfo->pNext);

   radv_pipeline_stage_init(pipeline->base.create_flags, pStage, pipeline_layout, &stage_key, &cs_stage);

   struct radv_shader_debug_info cs_dbg = {0};
   struct radv_shader_binary *cs_binary =
      radv_compile_cs(compiler_info, &cs_stage, pipeline->base.is_internal, &cs_dbg);
   pipeline->base.shaders[MESA_SHADER_COMPUTE] =
      radv_shader_create(device, cache, cs_binary, skip_shaders_cache, &cs_dbg);

   cs_stage.feedback.duration += os_time_get_nano() - stage_start;

   if (!skip_shaders_cache) {
      radv_pipeline_cache_insert(device, cache, &pipeline->base);
   }

   free(cs_binary);
   if (radv_can_dump_shader_stats(&device->compiler_info, cs_stage.nir)) {
      radv_dump_shader_stats(device, &pipeline->base, pipeline->base.shaders[MESA_SHADER_COMPUTE], stderr);
   }
   radv_pipeline_stage_finish(&cs_stage);

done:
   pipeline_feedback.duration = os_time_get_nano() - pipeline_start;

   if (creation_feedback) {
      *creation_feedback->pPipelineCreationFeedback = pipeline_feedback;

      if (creation_feedback->pipelineStageCreationFeedbackCount) {
         assert(creation_feedback->pipelineStageCreationFeedbackCount == 1);
         creation_feedback->pPipelineStageCreationFeedbacks[0] = cs_stage.feedback;
      }
   }

   return result;
}

static VkResult
radv_compute_pipeline_import_binary(struct radv_device *device, struct radv_compute_pipeline *pipeline,
                                    const VkPipelineBinaryInfoKHR *binary_info)
{
   VK_FROM_HANDLE(radv_pipeline_binary, pipeline_binary, binary_info->pPipelineBinaries[0]);
   struct radv_shader *shader;
   struct blob_reader blob;

   assert(binary_info->binaryCount == 1);

   blob_reader_init(&blob, pipeline_binary->data, pipeline_binary->size);

   shader = radv_shader_deserialize(device, pipeline_binary->key, sizeof(pipeline_binary->key), &blob);
   if (!shader)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   pipeline->base.shaders[MESA_SHADER_COMPUTE] = shader;

   pipeline->base.pipeline_hash = *(uint64_t *)pipeline_binary->key;

   return VK_SUCCESS;
}

VkResult
radv_compute_pipeline_create(VkDevice _device, VkPipelineCache _cache, const VkComputePipelineCreateInfo *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator, VkPipeline *pPipeline)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(vk_pipeline_cache, cache, _cache);
   VK_FROM_HANDLE(radv_pipeline_layout, pipeline_layout, pCreateInfo->layout);
   struct radv_compute_pipeline *pipeline;
   VkResult result;

   pipeline = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pipeline), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pipeline == NULL) {
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   radv_pipeline_init(device, &pipeline->base, RADV_PIPELINE_COMPUTE);
   pipeline->base.create_flags = vk_compute_pipeline_create_flags(pCreateInfo);
   pipeline->base.is_internal = _cache == device->meta_state.cache;

   const VkPipelineCreationFeedbackCreateInfo *creation_feedback =
      vk_find_struct_const(pCreateInfo->pNext, PIPELINE_CREATION_FEEDBACK_CREATE_INFO);

   const VkPipelineBinaryInfoKHR *binary_info = vk_find_struct_const(pCreateInfo->pNext, PIPELINE_BINARY_INFO_KHR);

   if (binary_info && binary_info->binaryCount > 0) {
      result = radv_compute_pipeline_import_binary(device, pipeline, binary_info);
   } else {
      result = radv_compute_pipeline_compile(pCreateInfo, pipeline, pipeline_layout, device, cache, &pCreateInfo->stage,
                                             creation_feedback);
   }

   if (result != VK_SUCCESS) {
      radv_pipeline_destroy(device, &pipeline->base, pAllocator);
      return result;
   }

   radv_pipeline_report_pso_history(device, &pipeline->base);

   *pPipeline = radv_pipeline_to_handle(&pipeline->base);
   radv_rmv_log_compute_pipeline_create(device, &pipeline->base, pipeline->base.is_internal);
   return VK_SUCCESS;
}

static VkResult
radv_create_compute_pipelines(VkDevice _device, VkPipelineCache pipelineCache, uint32_t count,
                              const VkComputePipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator,
                              VkPipeline *pPipelines)
{
   VkResult result = VK_SUCCESS;

   unsigned i = 0;
   for (; i < count; i++) {
      VkResult r;
      r = radv_compute_pipeline_create(_device, pipelineCache, &pCreateInfos[i], pAllocator, &pPipelines[i]);
      if (r != VK_SUCCESS) {
         result = r;
         pPipelines[i] = VK_NULL_HANDLE;

         VkPipelineCreateFlagBits2 create_flags = vk_compute_pipeline_create_flags(&pCreateInfos[i]);
         if (create_flags & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT)
            break;
      }
   }

   for (; i < count; ++i)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

void
radv_destroy_compute_pipeline(struct radv_device *device, struct radv_compute_pipeline *pipeline)
{
   if (pipeline->base.shaders[MESA_SHADER_COMPUTE])
      radv_shader_unref(device, pipeline->base.shaders[MESA_SHADER_COMPUTE]);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_CreateComputePipelines(VkDevice _device, VkPipelineCache pipelineCache, uint32_t count,
                            const VkComputePipelineCreateInfo *pCreateInfos, const VkAllocationCallbacks *pAllocator,
                            VkPipeline *pPipelines)
{
   return radv_create_compute_pipelines(_device, pipelineCache, count, pCreateInfos, pAllocator, pPipelines);
}
