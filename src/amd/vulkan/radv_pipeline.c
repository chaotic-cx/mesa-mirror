/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_pipeline.h"
#include "meta/radv_meta.h"
#include "nir/nir.h"
#include "nir/radv_nir.h"
#include "tools/radv_rmv.h"
#include "util/memstream.h"
#include "util/u_atomic.h"
#include "radv_pipeline_rt.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "vk_pipeline.h"
#include "vk_util.h"

#include "ac_binary.h"
#include "ac_nir.h"
#include "ac_shader_util.h"
#include "aco_interface.h"
#include "vk_shader_module.h"

bool
radv_pipeline_skip_shaders_cache(const struct radv_device *device, const struct radv_pipeline *pipeline)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   /* Skip the shaders cache when any of the below are true:
    * - shaders are dumped for debugging (RADV_DEBUG=shaders)
    * - binaries are captured (driver shouldn't store data to an internal cache)
    */
   return RADV_DEBUG_DUMP_SHADERS(instance) || (pipeline->create_flags & VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR);
}

void
radv_pipeline_init(struct radv_device *device, struct radv_pipeline *pipeline, enum radv_pipeline_type type)
{
   vk_object_base_init(&device->vk, &pipeline->base, VK_OBJECT_TYPE_PIPELINE);

   pipeline->type = type;
}

void
radv_pipeline_destroy(struct radv_device *device, struct radv_pipeline *pipeline,
                      const VkAllocationCallbacks *allocator)
{
   if (pipeline->cache_object)
      vk_pipeline_cache_object_unref(&device->vk, pipeline->cache_object);

   switch (pipeline->type) {
   case RADV_PIPELINE_GRAPHICS:
      radv_destroy_graphics_pipeline(device, radv_pipeline_to_graphics(pipeline));
      break;
   case RADV_PIPELINE_GRAPHICS_LIB:
      radv_destroy_graphics_lib_pipeline(device, radv_pipeline_to_graphics_lib(pipeline));
      break;
   case RADV_PIPELINE_COMPUTE:
      radv_destroy_compute_pipeline(device, radv_pipeline_to_compute(pipeline));
      break;
   case RADV_PIPELINE_RAY_TRACING:
      radv_destroy_ray_tracing_pipeline(device, radv_pipeline_to_ray_tracing(pipeline));
      break;
   default:
      UNREACHABLE("invalid pipeline type");
   }

   radv_rmv_log_resource_destroy(device, (uint64_t)radv_pipeline_to_handle(pipeline));
   vk_object_base_finish(&pipeline->base);
   vk_free2(&device->vk.alloc, allocator, pipeline);
}

VKAPI_ATTR void VKAPI_CALL
radv_DestroyPipeline(VkDevice _device, VkPipeline _pipeline, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_pipeline, pipeline, _pipeline);

   if (!_pipeline)
      return;

   radv_pipeline_destroy(device, pipeline, pAllocator);
}

static void
radv_set_stage_key_robustness(const struct vk_pipeline_robustness_state *rs, mesa_shader_stage stage,
                              struct radv_shader_stage_key *key)
{
   if (rs->storage_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2)
      key->storage_robustness2 = 1;
   if (rs->uniform_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2)
      key->uniform_robustness2 = 1;
   if (stage == MESA_SHADER_VERTEX &&
       (rs->vertex_inputs == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS ||
        rs->vertex_inputs == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2))
      key->vertex_robustness1 = 1u;
}

struct radv_shader_stage_key
radv_pipeline_get_shader_key(const struct radv_compiler_info *compiler_info,
                             const VkPipelineShaderStageCreateInfo *stage, VkPipelineCreateFlags2 flags,
                             const void *pNext)
{
   mesa_shader_stage s = vk_to_mesa_shader_stage(stage->stage);
   struct vk_pipeline_robustness_state rs;
   struct radv_shader_stage_key key = {0};

   key.keep_shader_arg_info = compiler_info->debug.keep_shader_info;
   key.keep_statistic_info =
      (flags & VK_PIPELINE_CREATE_2_CAPTURE_STATISTICS_BIT_KHR) || compiler_info->debug.capture_shader_stats;
   key.keep_executable_info =
      (flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR) || compiler_info->debug.capture_shaders;

   if (flags & VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT)
      key.optimisations_disabled = 1;

   if (flags & VK_PIPELINE_CREATE_2_VIEW_INDEX_FROM_DEVICE_INDEX_BIT)
      key.view_index_from_device_index = 1;

   if (flags & VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT)
      key.indirect_bindable = 1;

   if (flags & VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT)
      key.descriptor_heap = 1;

   if (stage->stage & RADV_GRAPHICS_STAGE_BITS) {
      key.version = compiler_info->override_graphics_shader_version;
   } else if (stage->stage & RADV_RT_STAGE_BITS) {
      key.version = compiler_info->override_ray_tracing_shader_version;
   } else {
      assert(stage->stage == VK_SHADER_STAGE_COMPUTE_BIT);
      key.version = compiler_info->override_compute_shader_version;
   }

   vk_pipeline_robustness_state_fill(&compiler_info->device_robustness_state, &rs, pNext, stage->pNext);

   radv_set_stage_key_robustness(&rs, s, &key);

   const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo *const subgroup_size =
      vk_find_struct_const(stage->pNext, PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);

   if (subgroup_size) {
      if (subgroup_size->requiredSubgroupSize == 32)
         key.subgroup_required_size = RADV_REQUIRED_WAVE32;
      else if (subgroup_size->requiredSubgroupSize == 64)
         key.subgroup_required_size = RADV_REQUIRED_WAVE64;
      else
         UNREACHABLE("Unsupported required subgroup size.");
   }

   if (stage->flags & VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) {
      key.subgroup_require_full = 1;
   }

   if (stage->flags & VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT) {
      key.subgroup_allow_varying = 1;
   }

   return key;
}

void
radv_merge_shader_stage_key(struct radv_shader_stage_key *dst, const struct radv_shader_stage_key *src)
{
   assert(dst->subgroup_required_size == src->subgroup_required_size);
   assert(dst->view_index_from_device_index == src->view_index_from_device_index);
   assert(dst->descriptor_heap == src->descriptor_heap);
   assert(dst->version == src->version);
   assert(dst->has_task_shader == src->has_task_shader);

   dst->subgroup_require_full |= src->subgroup_require_full;
   dst->subgroup_allow_varying &= src->subgroup_allow_varying;

   dst->storage_robustness2 |= src->storage_robustness2;
   dst->uniform_robustness2 |= src->uniform_robustness2;
   dst->vertex_robustness1 |= src->vertex_robustness1;

   dst->optimisations_disabled |= src->optimisations_disabled;
   dst->keep_statistic_info |= src->keep_statistic_info;
   dst->keep_executable_info |= src->keep_executable_info;
   dst->keep_shader_arg_info |= src->keep_shader_arg_info;
   dst->indirect_bindable |= src->indirect_bindable;
}

void
radv_pipeline_stage_init(VkPipelineCreateFlags2 pipeline_flags, const VkPipelineShaderStageCreateInfo *sinfo,
                         const struct radv_pipeline_layout *pipeline_layout,
                         const struct radv_shader_stage_key *stage_key, struct radv_shader_stage *out_stage)
{
   const VkShaderModuleCreateInfo *minfo = vk_find_struct_const(sinfo->pNext, SHADER_MODULE_CREATE_INFO);
   const VkPipelineShaderStageModuleIdentifierCreateInfoEXT *iinfo =
      vk_find_struct_const(sinfo->pNext, PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT);

   if (sinfo->module == VK_NULL_HANDLE && !minfo && !iinfo)
      return;

   memset(out_stage, 0, sizeof(*out_stage));

   out_stage->stage = vk_to_mesa_shader_stage(sinfo->stage);
   out_stage->next_stage = MESA_SHADER_NONE;
   out_stage->entrypoint = sinfo->pName;
   out_stage->spec_info = sinfo->pSpecializationInfo;
   out_stage->feedback.flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT;
   out_stage->key = *stage_key;

   if (sinfo->module != VK_NULL_HANDLE) {
      struct vk_shader_module *module = vk_shader_module_from_handle(sinfo->module);

      out_stage->spirv.data = module->data;
      out_stage->spirv.size = module->size;
      out_stage->spirv.object = &module->base;

      if (module->nir)
         out_stage->internal_nir = module->nir;
   } else if (minfo) {
      out_stage->spirv.data = (const char *)minfo->pCode;
      out_stage->spirv.size = minfo->codeSize;
   }

   const VkShaderDescriptorSetAndBindingMappingInfoEXT *mapping =
      vk_find_struct_const(sinfo->pNext, SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT);
   out_stage->layout.mapping = mapping;

   radv_shader_layout_init(pipeline_layout, out_stage->stage, &out_stage->layout);

   vk_pipeline_hash_shader_stage(pipeline_flags, sinfo, NULL, out_stage->shader_blake3);
}

void
radv_pipeline_stage_finish(struct radv_shader_stage *stage)
{
   ralloc_free(stage->nir);
}

void
radv_shader_layout_init(const struct radv_pipeline_layout *pipeline_layout, mesa_shader_stage stage,
                        struct radv_shader_layout *layout)
{
   if (!pipeline_layout)
      return;

   layout->num_sets = pipeline_layout->num_sets;
   for (unsigned i = 0; i < pipeline_layout->num_sets; i++) {
      layout->set[i].layout = pipeline_layout->set[i].layout;
      layout->set[i].dynamic_offset_start = pipeline_layout->set[i].dynamic_offset_start;
   }

   layout->dynamic_offset_count = pipeline_layout->dynamic_offset_count;
   layout->use_dynamic_descriptors = pipeline_layout->dynamic_offset_count &&
                                     (pipeline_layout->dynamic_shader_stages & mesa_to_vk_shader_stage(stage));
   layout->independent_sets = pipeline_layout->independent_sets;
}

static uint32_t
radv_get_executable_count(struct radv_pipeline *pipeline)
{
   uint32_t ret = 0;

   if (pipeline->type == RADV_PIPELINE_RAY_TRACING) {
      struct radv_ray_tracing_pipeline *rt_pipeline = radv_pipeline_to_ray_tracing(pipeline);
      for (uint32_t i = 0; i < rt_pipeline->stage_count; i++)
         ret += rt_pipeline->stages[i].shader ? 1 : 0;
      for (uint32_t i = 0; i < rt_pipeline->group_count; i++)
         ret += rt_pipeline->groups[i].ahit_isec_shader ? 1 : 0;
   }

   for (int i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      if (!pipeline->shaders[i])
         continue;

      ret += 1u;
      if (i == MESA_SHADER_GEOMETRY && pipeline->gs_copy_shader) {
         ret += 1u;
      }
   }

   return ret;
}

struct radv_shader *
radv_get_shader_from_executable_index(struct radv_pipeline *pipeline, int index, mesa_shader_stage *stage)
{
   if (pipeline->type == RADV_PIPELINE_RAY_TRACING) {
      struct radv_ray_tracing_pipeline *rt_pipeline = radv_pipeline_to_ray_tracing(pipeline);
      for (uint32_t i = 0; i < rt_pipeline->stage_count; i++) {
         struct radv_ray_tracing_stage *rt_stage = &rt_pipeline->stages[i];
         if (!rt_stage->shader)
            continue;

         if (!index) {
            *stage = rt_stage->stage;
            return rt_stage->shader;
         }

         index--;
      }
      for (uint32_t i = 0; i < rt_pipeline->group_count; i++) {
         struct radv_ray_tracing_group *rt_group = &rt_pipeline->groups[i];
         if (!rt_group->ahit_isec_shader)
            continue;

         if (!index) {
            *stage =
               rt_group->intersection_shader != VK_SHADER_UNUSED_KHR ? MESA_SHADER_INTERSECTION : MESA_SHADER_ANY_HIT;
            return rt_group->ahit_isec_shader;
         }

         index--;
      }
   }

   for (int i = 0; i < MESA_VULKAN_SHADER_STAGES; ++i) {
      if (!pipeline->shaders[i])
         continue;
      if (!index) {
         *stage = i;
         return pipeline->shaders[i];
      }

      --index;

      if (i == MESA_SHADER_GEOMETRY && pipeline->gs_copy_shader) {
         if (!index) {
            *stage = i;
            return pipeline->gs_copy_shader;
         }
         --index;
      }
   }

   *stage = -1;
   return NULL;
}

VkResult
radv_dump_shader_stats(struct radv_device *device, struct radv_pipeline *pipeline, struct radv_shader *shader,
                       FILE *output)
{
   VkPipelineExecutablePropertiesKHR *props = NULL;
   uint32_t prop_count = 0;
   VkResult result;

   VkPipelineInfoKHR pipeline_info = {0};
   pipeline_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
   pipeline_info.pipeline = radv_pipeline_to_handle(pipeline);

   result = radv_GetPipelineExecutablePropertiesKHR(radv_device_to_handle(device), &pipeline_info, &prop_count, NULL);
   if (result != VK_SUCCESS)
      return result;

   props = calloc(prop_count, sizeof(*props));
   if (!props)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   result = radv_GetPipelineExecutablePropertiesKHR(radv_device_to_handle(device), &pipeline_info, &prop_count, props);
   if (result != VK_SUCCESS)
      goto fail;

   for (unsigned exec_idx = 0; exec_idx < prop_count; exec_idx++) {
      mesa_shader_stage stage;
      if (radv_get_shader_from_executable_index(pipeline, exec_idx, &stage) != shader)
         continue;

      VkPipelineExecutableStatisticKHR *stats = NULL;
      uint32_t stat_count = 0;

      VkPipelineExecutableInfoKHR exec_info = {0};
      exec_info.pipeline = radv_pipeline_to_handle(pipeline);
      exec_info.executableIndex = exec_idx;

      result = radv_GetPipelineExecutableStatisticsKHR(radv_device_to_handle(device), &exec_info, &stat_count, NULL);
      if (result != VK_SUCCESS)
         goto fail;

      stats = calloc(stat_count, sizeof(*stats));
      if (!stats) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }

      result = radv_GetPipelineExecutableStatisticsKHR(radv_device_to_handle(device), &exec_info, &stat_count, stats);
      if (result != VK_SUCCESS) {
         free(stats);
         goto fail;
      }

      fprintf(output, "\n%s:\n", radv_get_shader_name(&shader->info, stage));
      fprintf(output, "*** SHADER STATS ***\n");

      for (unsigned i = 0; i < stat_count; i++) {
         fprintf(output, "%s: ", stats[i].name);
         switch (stats[i].format) {
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
            fprintf(output, "%s", stats[i].value.b32 == VK_TRUE ? "true" : "false");
            break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
            fprintf(output, "%" PRIi64, stats[i].value.i64);
            break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
            fprintf(output, "%" PRIu64, stats[i].value.u64);
            break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
            fprintf(output, "%f", stats[i].value.f64);
            break;
         default:
            UNREACHABLE("Invalid pipeline statistic format");
         }
         fprintf(output, "\n");
      }

      fprintf(output, "********************\n\n\n");

      free(stats);
   }

fail:
   free(props);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPipelineExecutablePropertiesKHR(VkDevice _device, const VkPipelineInfoKHR *pPipelineInfo,
                                        uint32_t *pExecutableCount, VkPipelineExecutablePropertiesKHR *pProperties)
{
   VK_FROM_HANDLE(radv_pipeline, pipeline, pPipelineInfo->pipeline);
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, pProperties, pExecutableCount);

   const uint32_t count = radv_get_executable_count(pipeline);
   for (uint32_t executable_idx = 0; executable_idx < count; executable_idx++) {
      VkPipelineExecutablePropertiesKHR *props = vk_outarray_next_typed(VkPipelineExecutablePropertiesKHR, &out);
      if (!props)
         continue;

      mesa_shader_stage stage;
      struct radv_shader *shader = radv_get_shader_from_executable_index(pipeline, executable_idx, &stage);

      props->stages = mesa_to_vk_shader_stage(stage);

      const char *name = _mesa_shader_stage_to_string(stage);
      const char *description = NULL;
      switch (stage) {
      case MESA_SHADER_VERTEX:
         description = "Vulkan Vertex Shader";
         break;
      case MESA_SHADER_TESS_CTRL:
         if (!pipeline->shaders[MESA_SHADER_VERTEX]) {
            props->stages |= VK_SHADER_STAGE_VERTEX_BIT;
            name = "vertex + tessellation control";
            description = "Combined Vulkan Vertex and Tessellation Control Shaders";
         } else {
            description = "Vulkan Tessellation Control Shader";
         }
         break;
      case MESA_SHADER_TESS_EVAL:
         description = "Vulkan Tessellation Evaluation Shader";
         break;
      case MESA_SHADER_GEOMETRY:
         if (shader->info.type == RADV_SHADER_TYPE_GS_COPY) {
            name = "geometry copy";
            description = "Extra shader stage that loads the GS output ringbuffer into the rasterizer";
            break;
         }

         if (pipeline->shaders[MESA_SHADER_TESS_CTRL] && !pipeline->shaders[MESA_SHADER_TESS_EVAL]) {
            props->stages |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
            name = "tessellation evaluation + geometry";
            description = "Combined Vulkan Tessellation Evaluation and Geometry Shaders";
         } else if (!pipeline->shaders[MESA_SHADER_TESS_CTRL] && !pipeline->shaders[MESA_SHADER_VERTEX]) {
            props->stages |= VK_SHADER_STAGE_VERTEX_BIT;
            name = "vertex + geometry";
            description = "Combined Vulkan Vertex and Geometry Shaders";
         } else {
            description = "Vulkan Geometry Shader";
         }
         break;
      case MESA_SHADER_FRAGMENT:
         description = "Vulkan Fragment Shader";
         break;
      case MESA_SHADER_COMPUTE:
         description = "Vulkan Compute Shader";
         break;
      case MESA_SHADER_MESH:
         description = "Vulkan Mesh Shader";
         break;
      case MESA_SHADER_TASK:
         description = "Vulkan Task Shader";
         break;
      case MESA_SHADER_RAYGEN:
         description = "Vulkan Ray Generation Shader";
         break;
      case MESA_SHADER_ANY_HIT:
         description = "Vulkan Any-Hit Shader";
         break;
      case MESA_SHADER_CLOSEST_HIT:
         description = "Vulkan Closest-Hit Shader";
         break;
      case MESA_SHADER_MISS:
         description = "Vulkan Miss Shader";
         break;
      case MESA_SHADER_INTERSECTION:
         if (shader->info.type == RADV_SHADER_TYPE_RT_TRAVERSAL)
            description = "Shader responsible for traversing the acceleration structure";
         else
            description = "Vulkan Intersection Shader";
         break;
      case MESA_SHADER_CALLABLE:
         description = "Vulkan Callable Shader";
         break;
      default:
         UNREACHABLE("Unsupported shader stage");
      }

      props->subgroupSize = shader->info.wave_size;
      VK_COPY_STR(props->name, name);
      VK_COPY_STR(props->description, description);
   }

   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPipelineExecutableStatisticsKHR(VkDevice _device, const VkPipelineExecutableInfoKHR *pExecutableInfo,
                                        uint32_t *pStatisticCount, VkPipelineExecutableStatisticKHR *pStatistics)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_pipeline, pipeline, pExecutableInfo->pipeline);
   mesa_shader_stage stage;
   struct radv_shader *shader =
      radv_get_shader_from_executable_index(pipeline, pExecutableInfo->executableIndex, &stage);

   const struct radv_physical_device *pdev = radv_device_physical(device);
   const enum amd_gfx_level gfx_level = pdev->info.gfx_level;

   struct radv_shader_binary_debug_info dbg = {0};
   if (pStatistics && *pStatisticCount)
      dbg = radv_parse_binary_debug_info(&device->compiler_info, shader->binary);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, pStatistics, pStatisticCount);

   struct amd_stats stats = {0};
   if (dbg.statistics)
      stats = *dbg.statistics;
   stats.driverhash = pipeline->pipeline_hash;
   stats.sgprs = shader->config.num_sgprs;
   stats.vgprs = shader->config.num_vgprs;
   stats.spillsgprs = shader->config.spilled_sgprs;
   stats.spillvgprs = shader->config.spilled_vgprs;
   stats.codesize = shader->exec_size;
   stats.lds = align(shader->config.lds_size, ac_shader_get_lds_alloc_granularity(gfx_level));
   stats.scratch = shader->config.scratch_bytes_per_wave;
   stats.maxwaves = shader->max_waves;

   radv_free_binary_debug_info(&dbg);

   switch (stage) {
   case MESA_SHADER_VERTEX:
      if (gfx_level <= GFX8 || (!shader->info.vs.as_es && !shader->info.vs.as_ls)) {
         /* VS inputs when VS is a separate stage */
         stats.inputs += util_bitcount(shader->info.vs.input_slot_usage_mask);
      }
      break;

   case MESA_SHADER_TESS_CTRL:
      if (gfx_level >= GFX9) {
         /* VS inputs when pipeline has tess */
         stats.inputs += util_bitcount(shader->info.vs.input_slot_usage_mask);
      }

      /* VS -> TCS inputs */
      stats.inputs += shader->info.tcs.num_linked_inputs;
      break;

   case MESA_SHADER_TESS_EVAL:
      if (gfx_level <= GFX8 || !shader->info.tes.as_es) {
         /* TCS -> TES inputs when TES is a separate stage */
         stats.inputs += shader->info.tes.num_linked_inputs + shader->info.tes.num_linked_patch_inputs;
      }
      break;

   case MESA_SHADER_GEOMETRY:
      /* The IO stats of the GS copy shader are already reflected by GS and FS, so leave it empty. */
      if (shader->info.type == RADV_SHADER_TYPE_GS_COPY)
         break;

      if (gfx_level >= GFX9) {
         if (shader->info.gs.es_type == MESA_SHADER_VERTEX) {
            /* VS inputs when pipeline has GS but no tess */
            stats.inputs += util_bitcount(shader->info.vs.input_slot_usage_mask);
         } else if (shader->info.gs.es_type == MESA_SHADER_TESS_EVAL) {
            /* TCS -> TES inputs when pipeline has GS */
            stats.inputs += shader->info.tes.num_linked_inputs + shader->info.tes.num_linked_patch_inputs;
         }
      }

      /* VS -> GS or TES -> GS inputs */
      stats.inputs += shader->info.gs.num_linked_inputs;
      break;

   case MESA_SHADER_FRAGMENT:
      stats.inputs += shader->info.ps.num_inputs;
      break;

   default:
      /* Other stages don't have IO or we are not interested in them. */
      break;
   }

   switch (stage) {
   case MESA_SHADER_VERTEX:
      if (!shader->info.vs.as_ls && !shader->info.vs.as_es) {
         /* VS -> FS outputs. */
         stats.outputs += shader->info.outinfo.param_exports + shader->info.outinfo.prim_param_exports;
      } else if (gfx_level <= GFX8) {
         /* VS -> TCS, VS -> GS outputs on GFX6-8 */
         stats.outputs += shader->info.vs.num_linked_outputs;
      }
      break;

   case MESA_SHADER_TESS_CTRL:
      if (gfx_level >= GFX9) {
         /* VS -> TCS outputs on GFX9+ */
         stats.outputs += shader->info.vs.num_linked_outputs;
      }

      /* TCS -> TES outputs */
      stats.outputs += shader->info.tcs.io_info.highest_remapped_vram_output +
                       shader->info.tcs.io_info.highest_remapped_vram_patch_output;
      break;

   case MESA_SHADER_TESS_EVAL:
      if (!shader->info.tes.as_es) {
         /* TES -> FS outputs */
         stats.outputs += shader->info.outinfo.param_exports + shader->info.outinfo.prim_param_exports;
      } else if (gfx_level <= GFX8) {
         /* TES -> GS outputs on GFX6-8 */
         stats.outputs += shader->info.tes.num_linked_outputs;
      }
      break;

   case MESA_SHADER_GEOMETRY:
      /* The IO stats of the GS copy shader are already reflected by GS and FS, so leave it empty. */
      if (shader->info.type == RADV_SHADER_TYPE_GS_COPY)
         break;

      if (gfx_level >= GFX9) {
         if (shader->info.gs.es_type == MESA_SHADER_VERTEX) {
            /* VS -> GS outputs on GFX9+ */
            stats.outputs += shader->info.vs.num_linked_outputs;
         } else if (shader->info.gs.es_type == MESA_SHADER_TESS_EVAL) {
            /* TES -> GS outputs on GFX9+ */
            stats.outputs += shader->info.tes.num_linked_outputs;
         }
      }

      if (shader->info.is_ngg) {
         /* GS -> FS outputs (GFX10+ NGG) */
         stats.outputs += shader->info.outinfo.param_exports + shader->info.outinfo.prim_param_exports;
      } else {
         /* GS -> FS outputs (GFX6-10.3 legacy) */
         stats.outputs += DIV_ROUND_UP(((uint32_t)shader->info.gs.num_components_per_stream[0] +
                                        (uint32_t)shader->info.gs.num_components_per_stream[1] +
                                        (uint32_t)shader->info.gs.num_components_per_stream[2] +
                                        (uint32_t)shader->info.gs.num_components_per_stream[3]) *
                                          4,
                                       16);
      }
      break;

   case MESA_SHADER_MESH:
      /* MS -> FS outputs */
      stats.outputs += shader->info.outinfo.param_exports + shader->info.outinfo.prim_param_exports;
      break;

   case MESA_SHADER_FRAGMENT:
      stats.outputs += DIV_ROUND_UP(util_bitcount(shader->info.ps.colors_written), 4) + !!shader->info.ps.writes_z +
                       !!shader->info.ps.writes_stencil + !!shader->info.ps.writes_sample_mask +
                       !!shader->info.ps.writes_mrt0_alpha_to_mrtz;
      break;

   default:
      /* Other stages don't have IO or we are not interested in them. */
      break;
   }

   vk_add_amd_stats(out, &stats);

   return vk_outarray_status(&out);
}

static VkResult
radv_copy_representation(void *data, size_t *data_size, const char *src)
{
   size_t total_size = strlen(src) + 1;

   if (!data) {
      *data_size = total_size;
      return VK_SUCCESS;
   }

   size_t size = MIN2(total_size, *data_size);

   memcpy(data, src, size);
   if (size)
      *((char *)data + size - 1) = 0;
   return size < total_size ? VK_INCOMPLETE : VK_SUCCESS;
}

static VkResult
radv_copy_assembly(void *data, size_t *data_size, const struct radv_shader *shader,
                   const struct radv_shader_binary_debug_info *dbg)
{
   char *memstream_data = NULL;
   size_t disasm_size = 0;
   struct u_memstream mem;

   if (!u_memstream_open(&mem, &memstream_data, &disasm_size)) {
      *data_size = 0;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   FILE *memf = u_memstream_get(&mem);
   radv_print_shader_disasm(dbg, shader->code_size, NULL, 0, radv_shader_get_va(shader), memf);
   fputc(0, memf);
   u_memstream_close(&mem);

   VkResult result = radv_copy_representation(data, data_size, memstream_data);
   free(memstream_data);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPipelineExecutableInternalRepresentationsKHR(
   VkDevice _device, const VkPipelineExecutableInfoKHR *pExecutableInfo, uint32_t *pInternalRepresentationCount,
   VkPipelineExecutableInternalRepresentationKHR *pInternalRepresentations)
{
   VK_FROM_HANDLE(radv_device, device, _device);
   VK_FROM_HANDLE(radv_pipeline, pipeline, pExecutableInfo->pipeline);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   mesa_shader_stage stage;
   struct radv_shader *shader =
      radv_get_shader_from_executable_index(pipeline, pExecutableInfo->executableIndex, &stage);

   struct radv_shader_binary_debug_info dbg = radv_parse_binary_debug_info(&device->compiler_info, shader->binary);

   VkPipelineExecutableInternalRepresentationKHR *p = pInternalRepresentations;
   VkPipelineExecutableInternalRepresentationKHR *end =
      p + (pInternalRepresentations ? *pInternalRepresentationCount : 0);
   VkResult result = VK_SUCCESS;
   /* optimized NIR */
   if (p < end) {
      p->isText = true;
      VK_COPY_STR(p->name, "NIR Shader(s)");
      VK_COPY_STR(p->description, "The optimized NIR shader(s)");
      if (radv_copy_representation(p->pData, &p->dataSize, shader->dbg.nir_string) != VK_SUCCESS)
         result = VK_INCOMPLETE;
   }
   ++p;

   /* backend IR */
   if (p < end) {
      p->isText = true;
      if (pdev->use_llvm) {
         VK_COPY_STR(p->name, "LLVM IR");
         VK_COPY_STR(p->description, "The LLVM IR after some optimizations");
      } else {
         VK_COPY_STR(p->name, "ACO IR");
         VK_COPY_STR(p->description, "The ACO IR after some optimizations");
      }
      if (radv_copy_representation(p->pData, &p->dataSize, dbg.ir_string) != VK_SUCCESS)
         result = VK_INCOMPLETE;
   }
   ++p;

   /* Disassembler */
   if (p < end && dbg.disasm_string) {
      p->isText = true;
      VK_COPY_STR(p->name, "Assembly");
      VK_COPY_STR(p->description, "Final Assembly");
      if (radv_copy_assembly(p->pData, &p->dataSize, shader, &dbg) != VK_SUCCESS)
         result = VK_INCOMPLETE;
   }
   ++p;

   radv_free_binary_debug_info(&dbg);

   if (!pInternalRepresentations)
      *pInternalRepresentationCount = p - pInternalRepresentations;
   else if (p > end) {
      result = VK_INCOMPLETE;
      *pInternalRepresentationCount = end - pInternalRepresentations;
   } else {
      *pInternalRepresentationCount = p - pInternalRepresentations;
   }

   return result;
}

static void
vk_shader_module_finish(void *_module)
{
   struct vk_shader_module *module = _module;
   vk_object_base_finish(&module->base);
}

VkShaderDescriptorSetAndBindingMappingInfoEXT *
radv_copy_descriptor_heap_mapping_info(const VkShaderDescriptorSetAndBindingMappingInfoEXT *mapping, void *mem_ctx)
{
   VkShaderDescriptorSetAndBindingMappingInfoEXT *new_mapping =
      ralloc(mem_ctx, VkShaderDescriptorSetAndBindingMappingInfoEXT);
   if (!new_mapping)
      return NULL;

   new_mapping->sType = mapping->sType;
   new_mapping->pNext = NULL;
   new_mapping->mappingCount = mapping->mappingCount;

   const uint32_t mappings_size = sizeof(VkDescriptorSetAndBindingMappingEXT) * mapping->mappingCount;
   new_mapping->pMappings = ralloc_size(mem_ctx, mappings_size);
   if (!new_mapping->pMappings)
      return NULL;

   memcpy((void *)new_mapping->pMappings, mapping->pMappings, mappings_size);

   return new_mapping;
}

VkPipelineShaderStageCreateInfo *
radv_copy_shader_stage_create_info(struct radv_device *device, uint32_t stageCount,
                                   const VkPipelineShaderStageCreateInfo *pStages, void *mem_ctx)
{
   VkPipelineShaderStageCreateInfo *new_stages;

   size_t size = sizeof(VkPipelineShaderStageCreateInfo) * stageCount;
   new_stages = ralloc_size(mem_ctx, size);
   if (!new_stages)
      return NULL;

   if (size)
      memcpy(new_stages, pStages, size);

   for (uint32_t i = 0; i < stageCount; i++) {
      VK_FROM_HANDLE(vk_shader_module, module, new_stages[i].module);

      const VkShaderModuleCreateInfo *minfo = vk_find_struct_const(pStages[i].pNext, SHADER_MODULE_CREATE_INFO);

      if (module) {
         struct vk_shader_module *new_module = ralloc_size(mem_ctx, sizeof(struct vk_shader_module) + module->size);
         if (!new_module)
            return NULL;

         ralloc_set_destructor(new_module, vk_shader_module_finish);
         vk_object_base_init(&device->vk, &new_module->base, VK_OBJECT_TYPE_SHADER_MODULE);

         new_module->nir = NULL;
         memcpy(new_module->hash, module->hash, sizeof(module->hash));
         new_module->size = module->size;
         memcpy(new_module->data, module->data, module->size);

         module = new_module;
      } else if (minfo) {
         module = ralloc_size(mem_ctx, sizeof(struct vk_shader_module) + minfo->codeSize);
         if (!module)
            return NULL;

         vk_shader_module_init(&device->vk, module, minfo);
      }

      if (module) {
         const VkSpecializationInfo *spec = new_stages[i].pSpecializationInfo;
         if (spec) {
            VkSpecializationInfo *new_spec = ralloc(mem_ctx, VkSpecializationInfo);
            if (!new_spec)
               return NULL;

            new_spec->mapEntryCount = spec->mapEntryCount;
            uint32_t map_entries_size = sizeof(VkSpecializationMapEntry) * spec->mapEntryCount;
            new_spec->pMapEntries = ralloc_size(mem_ctx, map_entries_size);
            if (!new_spec->pMapEntries)
               return NULL;
            memcpy((void *)new_spec->pMapEntries, spec->pMapEntries, map_entries_size);

            new_spec->dataSize = spec->dataSize;
            new_spec->pData = ralloc_size(mem_ctx, spec->dataSize);
            if (!new_spec->pData)
               return NULL;
            memcpy((void *)new_spec->pData, spec->pData, spec->dataSize);

            new_stages[i].pSpecializationInfo = new_spec;
         }

         new_stages[i].module = vk_shader_module_to_handle(module);
         new_stages[i].pName = ralloc_strdup(mem_ctx, new_stages[i].pName);
         if (!new_stages[i].pName)
            return NULL;
         new_stages[i].pNext = NULL;
      }

      const VkShaderDescriptorSetAndBindingMappingInfoEXT *mapping =
         vk_find_struct_const(pStages[i].pNext, SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT);
      if (mapping) {
         VkShaderDescriptorSetAndBindingMappingInfoEXT *copied_mapping =
            radv_copy_descriptor_heap_mapping_info(mapping, mem_ctx);
         if (!copied_mapping)
            return NULL;

         new_stages[i].pNext = copied_mapping;
      }
   }

   return new_stages;
}

void
radv_pipeline_hash(const struct radv_device *device, const struct radv_pipeline_layout *pipeline_layout,
                   blake3_hasher *ctx)
{
   _mesa_blake3_update(ctx, device->cache_hash, sizeof(device->cache_hash));
   if (pipeline_layout)
      _mesa_blake3_update(ctx, pipeline_layout->hash, sizeof(pipeline_layout->hash));
}

void
radv_pipeline_hash_shader_stage(VkPipelineCreateFlags2 pipeline_flags, const VkPipelineShaderStageCreateInfo *sinfo,
                                const struct radv_shader_stage_key *stage_key, blake3_hasher *ctx)
{
   unsigned char shader_blake3[BLAKE3_KEY_LEN];

   vk_pipeline_hash_shader_stage(pipeline_flags, sinfo, NULL, shader_blake3);

   _mesa_blake3_update(ctx, shader_blake3, sizeof(shader_blake3));
   _mesa_blake3_update(ctx, stage_key, sizeof(*stage_key));
}

static void
radv_print_pso_history(const struct radv_pipeline *pipeline, const struct radv_shader *shader, FILE *output)
{
   const uint64_t start_addr = radv_shader_get_va(shader) & ((1ull << 48) - 1);
   const uint64_t end_addr = start_addr + shader->code_size;

   fprintf(output, "pipeline_hash=%.16llx, VA=%.16llx-%.16llx, stage=%s\n", (long long)pipeline->pipeline_hash,
           (long long)start_addr, (long long)end_addr, _mesa_shader_stage_to_string(shader->info.stage));
   fflush(output);
}

void
radv_pipeline_report_pso_history(const struct radv_device *device, struct radv_pipeline *pipeline)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   FILE *output = instance->pso_history_logfile ? instance->pso_history_logfile : stderr;

   if (!(RADV_DEBUG(instance, PSO_HISTORY)))
      return;

   /* Only report PSO history for application pipelines. */
   if (pipeline->is_internal)
      return;

   switch (pipeline->type) {
   case RADV_PIPELINE_GRAPHICS:
      for (uint32_t i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         const struct radv_shader *shader = pipeline->shaders[i];

         if (shader)
            radv_print_pso_history(pipeline, shader, output);
      }

      if (pipeline->gs_copy_shader)
         radv_print_pso_history(pipeline, pipeline->gs_copy_shader, output);
      break;
   case RADV_PIPELINE_COMPUTE:
      radv_print_pso_history(pipeline, pipeline->shaders[MESA_SHADER_COMPUTE], output);
      break;
   case RADV_PIPELINE_RAY_TRACING: {
      struct radv_ray_tracing_pipeline *rt_pipeline = radv_pipeline_to_ray_tracing(pipeline);

      if (rt_pipeline->prolog)
         radv_print_pso_history(pipeline, rt_pipeline->prolog, output);

      if (pipeline->shaders[MESA_SHADER_INTERSECTION])
         radv_print_pso_history(pipeline, pipeline->shaders[MESA_SHADER_INTERSECTION], output);

      for (uint32_t i = 0; i < rt_pipeline->non_imported_stage_count; i++) {
         const struct radv_shader *shader = rt_pipeline->stages[i].shader;

         if (shader)
            radv_print_pso_history(pipeline, shader, output);
      }
      break;
   }
   default:
      break;
   }
}
