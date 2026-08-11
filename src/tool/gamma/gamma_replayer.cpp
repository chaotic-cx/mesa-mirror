/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */
#include "gamma_replayer.h"
#include "vulkan/vulkan_core.h"

gamma_acceleration_structure_build::gamma_acceleration_structure_build(gamma_replayer *replayer): replayer(replayer)
{
}

void
gamma_acceleration_structure_build::load(char *chunk_data, uint32_t chunk_size)
{
   gamma_acceleration_structure_build_info_header *header =
      (gamma_acceleration_structure_build_info_header *)chunk_data;
   char *geom_data = chunk_data + sizeof(gamma_acceleration_structure_build_info_header);

   range_infos.resize(header->geometry_count);
   geom_infos.resize(header->geometry_count);

   prim_data_size = 0;
   total_prim_count = 0;
   char *geom_iter = geom_data;
   for (uint32_t i = 0; i < header->geometry_count; ++i) {
      geom_infos[i].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geom_infos[i].geometryType = header->geometry_type;

      gamma_acceleration_structure_geometry_info *geom = (gamma_acceleration_structure_geometry_info *)geom_iter;
      geom_infos[i].flags = geom->geometry_flags;
      range_infos[i] = {
         .primitiveCount = geom->primitive_count,
      };

      geom_iter += sizeof(gamma_acceleration_structure_geometry_info);
      switch (header->geometry_type) {
      case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
         geom_infos[i].geometry.triangles = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData = {.deviceAddress = prim_data_size},
            .vertexStride = 3 * sizeof(float),
            .maxVertex = geom->primitive_count * 3,
            .indexType = VK_INDEX_TYPE_NONE_KHR,
         };
         geom_iter += geom->primitive_count * sizeof(gamma_acceleration_structure_triangle_primitive);
         prim_data_size += geom->primitive_count * 9 * sizeof(float);
         break;
      case VK_GEOMETRY_TYPE_AABBS_KHR:
         geom_infos[i].geometry.aabbs = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
            .data = {.deviceAddress = prim_data_size},
            .stride = 6 * sizeof(float),
         };
         geom_iter += geom->primitive_count * sizeof(gamma_acceleration_structure_aabb_primitive);
         prim_data_size += geom->primitive_count * 6 * sizeof(float);
         break;
      case VK_GEOMETRY_TYPE_INSTANCES_KHR:
         geom_infos[i].geometry.instances = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            .data = {.deviceAddress = prim_data_size},
         };
         geom_iter += geom->primitive_count * sizeof(VkAccelerationStructureInstanceKHR);
         prim_data_size += geom->primitive_count * sizeof(VkAccelerationStructureInstanceKHR);
         break;
      default:
         assert(!"gamma: Unhandled geometry type!\n");
      }

      total_prim_count += geom->primitive_count;
   }

   build_info = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .pNext = nullptr,
      .type = header->geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                                                                      : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = header->build_flags,
      .geometryCount = header->geometry_count,
      .pGeometries = geom_infos.data(),
   };

   prim_data_buffer = gamma_create_backed_buffer(
      (struct gamma_device *)replayer, MAX2(prim_data_size, 1), gamma_memory_type_device_local,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      false);

   std::vector<uint32_t> prim_counts;
   for (auto &range_info : range_infos)
      prim_counts.push_back(range_info.primitiveCount);

   VkAccelerationStructureBuildSizesInfoKHR size_info = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
   };
   replayer->vkGetAccelerationStructureBuildSizesKHR(replayer->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                     &build_info, prim_counts.data(), &size_info);

   as_storage = gamma_create_backed_buffer(
      (struct gamma_device *)replayer, size_info.accelerationStructureSize, gamma_memory_type_device_local,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false);
   build_scratch = gamma_create_backed_buffer(
      (struct gamma_device *)replayer, size_info.buildScratchSize, gamma_memory_type_device_local,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, false);

   VkAccelerationStructureCreateInfoKHR as_info = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .buffer = as_storage->buffer,
      .size = size_info.accelerationStructureSize,
      .type = build_info.type,
   };
   gamma_check_vk_result(replayer->vkCreateAccelerationStructureKHR(replayer->device, &as_info, NULL, &structure));
   build_info.dstAccelerationStructure = structure;
   build_info.scratchData.deviceAddress = build_scratch->address;

   for (auto &info : geom_infos) {
      switch (info.geometryType) {
      case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
         info.geometry.triangles.vertexData.deviceAddress += prim_data_buffer->address;
         break;
      case VK_GEOMETRY_TYPE_AABBS_KHR:
         info.geometry.aabbs.data.deviceAddress += prim_data_buffer->address;
         break;
      case VK_GEOMETRY_TYPE_INSTANCES_KHR:
         info.geometry.instances.data.deviceAddress += prim_data_buffer->address;
         break;
      default:
         assert(!"Unhandled geometry type!");
         break;
      }
   }
   build_info.pGeometries = geom_infos.data();

   if (prim_data_size == 0)
      return;

   prim_data = malloc(prim_data_size);

   geom_iter = geom_data;
   prim_data_size = 0;
   for (uint32_t i = 0; i < header->geometry_count; ++i) {
      gamma_acceleration_structure_geometry_info *geom = (gamma_acceleration_structure_geometry_info *)geom_iter;
      geom_iter += sizeof(gamma_acceleration_structure_geometry_info);
      void *src_prim_data = geom_iter;
      size_t prim_size = geom->primitive_count;

      switch (header->geometry_type) {
      case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
         prim_size *= sizeof(gamma_acceleration_structure_triangle_primitive);
         break;
      case VK_GEOMETRY_TYPE_AABBS_KHR:
         prim_size *= sizeof(gamma_acceleration_structure_aabb_primitive);
         break;
      case VK_GEOMETRY_TYPE_INSTANCES_KHR:
         prim_size *= sizeof(VkAccelerationStructureInstanceKHR);
         break;
      default:
         assert(!"gamma: Unhandled geometry type!\n");
      }
      memcpy((char *)prim_data + prim_data_size, src_prim_data, prim_size);
      geom_iter += prim_size;
      prim_data_size += prim_size;
   }
}

void
gamma_replayer::print_results(const char *dst_name, FILE *dst)
{
   fprintf(dst, "Filename,Run,ASIndex,PrimCount,BuildTime,CompactedSize\n");

   for (uint32_t i = 0; i < builds.size(); ++i) {
      uint32_t run_idx = 0;
      for (auto &result : results) {
         fprintf(dst, "%s,%u,%u,%u,%lu,%lu\n", dst_name, run_idx, i, builds[i]->total_prim_count, result[i].build_time,
                 result[i].compacted_size);
         run_idx += 1;
      }
   }
}

static bool
replay_pdev_supported(VkInstance instance, VkPhysicalDevice pdev)
{
   VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
   };

   VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &as_features,
   };

   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features12,
   };
   vkGetPhysicalDeviceFeatures2(pdev, &features2);

   if (!as_features.accelerationStructure || !features12.hostQueryReset)
      return false;
   return true;
}

int
gamma_replayer_init(gamma_replayer *replayer)
{
   const char *device_exts[] = {
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
   };

   VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_struct_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = NULL,
      .accelerationStructure = true,
   };

   VkPhysicalDeviceVulkan12Features vk12Features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &accel_struct_features,
      .hostQueryReset = true,
      .bufferDeviceAddress = true,
   };

   if (gamma_device_init(replayer, replay_pdev_supported, 0, nullptr, ARRAY_SIZE(device_exts), device_exts, nullptr,
                         &vk12Features)) {
      fprintf(stderr, "gamma: Failed to initialize base Vulkan structures\n");
      return 1;
   }

   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(replayer->pdev, &properties);
   replayer->timestamp_period = properties.limits.timestampPeriod;

   VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = replayer->queue_family_index,
   };
   gamma_check_vk_result(vkCreateCommandPool(replayer->device, &command_pool_info, NULL, &replayer->command_pool));
   VkCommandBufferAllocateInfo cmdbuf_alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = replayer->command_pool,
      .commandBufferCount = 1,
   };
   gamma_check_vk_result(vkAllocateCommandBuffers(replayer->device, &cmdbuf_alloc_info, &replayer->command_buffer));

   replayer->vkCreateAccelerationStructureKHR =
      (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(replayer->device, "vkCreateAccelerationStructureKHR");
   replayer->vkDestroyAccelerationStructureKHR =
      (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(replayer->device, "vkDestroyAccelerationStructureKHR");
   replayer->vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(
      replayer->device, "vkGetAccelerationStructureBuildSizesKHR");
   replayer->vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(
      replayer->device, "vkCmdBuildAccelerationStructuresKHR");
   replayer->vkCmdWriteAccelerationStructuresPropertiesKHR =
      (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)vkGetDeviceProcAddr(
         replayer->device, "vkCmdWriteAccelerationStructuresPropertiesKHR");

   return 0;
}

void
gamma_replayer_destroy(gamma_replayer *replayer)
{
   for (auto &build : replayer->builds)
      replayer->destroy_build(build.get());
   vkFreeCommandBuffers(replayer->device, replayer->command_pool, 1, &replayer->command_buffer);
   vkDestroyCommandPool(replayer->device, replayer->command_pool, NULL);
   vkDestroyQueryPool(replayer->device, replayer->time_query_pool, NULL);
}

void
gamma_replayer::load(FILE *file)
{
   gamma_header header;
   fread(&header, sizeof(header), 1, file);
   for (uint32_t i = 0; i < header.chunk_count; i++) {
      gamma_chunk_header chunk_header;
      fread(&chunk_header, sizeof(chunk_header), 1, file);

      if (chunk_header.type == gamma_chunk_type_acceleration_structure ||
          chunk_header.type >= gamma_chunk_type_driver_start) {
         continue;
      } else if (chunk_header.type == gamma_chunk_type_build_info) {
         std::unique_ptr<gamma_acceleration_structure_build> build =
            std::make_unique<gamma_acceleration_structure_build>(this);
         char *chunk_data = (char *)malloc(chunk_header.size);
         fread(chunk_data, 1, chunk_header.size, file);
         build->load(chunk_data, chunk_header.size);
         free(chunk_data);
         builds.push_back(std::move(build));
      } else {
         fprintf(stderr, "gamma: Invalid gamma_chunk_type\n");
         return;
      }
   }

   for (auto &build : builds)
      populate(build.get());
   flush_upload_memory();

   time_query_count = builds.size() * 2;
   VkQueryPoolCreateInfo query_pool_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = time_query_count,
   };
   vkCreateQueryPool(device, &query_pool_info, NULL, &time_query_pool);

   size_query_count = builds.size();
   query_pool_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
      .queryCount = size_query_count,
   };
   vkCreateQueryPool(device, &query_pool_info, NULL, &size_query_pool);
}

void
gamma_replayer::populate(gamma_acceleration_structure_build *build)
{
   void *prim_data_ptr = upload_memory(build->prim_data_buffer);
   if (build->build_info.type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) {
      memcpy(prim_data_ptr, build->prim_data, build->prim_data_size);
   } else {
      for (size_t instance_offset = 0; instance_offset < build->prim_data_size;
           instance_offset += sizeof(VkAccelerationStructureInstanceKHR)) {
         VkAccelerationStructureInstanceKHR instance;
         memcpy(&instance, (char *)build->prim_data + instance_offset, sizeof(VkAccelerationStructureInstanceKHR));
         if (instance.accelerationStructureReference == UINT32_MAX)
            instance.accelerationStructureReference = 0;
         else
            instance.accelerationStructureReference =
               builds[instance.accelerationStructureReference]->as_storage->address;
         memcpy((char *)prim_data_ptr + instance_offset, &instance, sizeof(VkAccelerationStructureInstanceKHR));
      }
   }
}

void
gamma_replayer::build(gamma_acceleration_structure_build *build)
{
   VkAccelerationStructureBuildRangeInfoKHR *range = build->range_infos.data();
   this->vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &build->build_info, &range);
}

void
gamma_replayer::destroy_build(gamma_acceleration_structure_build *build)
{
   this->vkDestroyAccelerationStructureKHR(device, build->structure, 0);
}

void
gamma_replayer_run(gamma_replayer *replayer, uint32_t permutation_seed, uint32_t num_test_runs)
{
   replayer->results.clear();
   replayer->results.resize(num_test_runs);

   std::vector<gamma_acceleration_structure_build *> permuted_builds;
   std::vector<uint32_t> inv_permutation;
   permuted_builds.reserve(replayer->builds.size());
   inv_permutation.reserve(replayer->builds.size());
   if (permutation_seed) {
      srand(permutation_seed);
      for (size_t i = 0; i < replayer->builds.size(); ++i) {
         if (!i)
            permuted_builds.push_back(replayer->builds[0].get());

         auto build_insertion_point = permuted_builds.begin();
         auto permute_insertion_point = inv_permutation.begin();
         size_t insertion_index = rand() % permuted_builds.size();

         std::advance(build_insertion_point, insertion_index);
         std::advance(permute_insertion_point, insertion_index);

         permuted_builds.insert(build_insertion_point, replayer->builds[i].get());
         inv_permutation.insert(permute_insertion_point, i);
      }
   } else {
      size_t i = 0;
      for (auto &build : replayer->builds) {
         permuted_builds.push_back(build.get());
         inv_permutation.push_back(i++);
      }
   }

   for (uint32_t run = 0; run < num_test_runs; ++run) {
      replayer->results[run].resize(replayer->builds.size());

      vkResetQueryPool(replayer->device, replayer->time_query_pool, 0, replayer->time_query_count);
      vkResetQueryPool(replayer->device, replayer->size_query_pool, 0, replayer->size_query_count);
      gamma_check_vk_result(vkResetCommandPool(replayer->device, replayer->command_pool, 0));

      VkCommandBufferBeginInfo begin_info = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      VkSubmitInfo submitInfo = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &replayer->command_buffer,
      };

      gamma_check_vk_result(vkBeginCommandBuffer(replayer->command_buffer, &begin_info));

      /* Synchronize uploading the input data */
      if (!run) {
         VkMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
         };
         vkCmdPipelineBarrier(replayer->command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0,
                              nullptr);
      }

      uint32_t timestamp_index = 0;
      size_t num_builds_pending = 0;
      for (bool blas = true;; blas = false) {
         for (unsigned i = 0; i < permuted_builds.size(); ++i) {
            auto build = permuted_builds[i];
            if ((build->build_info.type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) != blas)
               continue;

            vkCmdWriteTimestamp(replayer->command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, replayer->time_query_pool,
                                timestamp_index++);
            replayer->build(build);
            VkMemoryBarrier barrier = {
               .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
               .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
               .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            };
            vkCmdPipelineBarrier(replayer->command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            vkCmdWriteTimestamp(replayer->command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, replayer->time_query_pool,
                                timestamp_index++);
            replayer->vkCmdWriteAccelerationStructuresPropertiesKHR(
               replayer->command_buffer, 1, &build->structure, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
               replayer->size_query_pool, i);

            if (++num_builds_pending > 400) {
               gamma_check_vk_result(vkEndCommandBuffer(replayer->command_buffer));
               gamma_check_vk_result(vkQueueSubmit(replayer->queue, 1, &submitInfo, VK_NULL_HANDLE));
               gamma_check_vk_result(vkDeviceWaitIdle(replayer->device));

               gamma_check_vk_result(vkResetCommandPool(replayer->device, replayer->command_pool, 0));
               gamma_check_vk_result(vkBeginCommandBuffer(replayer->command_buffer, &begin_info));
            }
         }
         if (!blas)
            break;
      }

      gamma_check_vk_result(vkEndCommandBuffer(replayer->command_buffer));
      gamma_check_vk_result(vkQueueSubmit(replayer->queue, 1, &submitInfo, VK_NULL_HANDLE));
      gamma_check_vk_result(vkDeviceWaitIdle(replayer->device));

      assert(timestamp_index == replayer->time_query_count);

      std::vector<uint64_t> timestamps = std::vector<uint64_t>(replayer->time_query_count);
      std::vector<uint64_t> sizes = std::vector<uint64_t>(replayer->size_query_count);
      gamma_check_vk_result(vkGetQueryPoolResults(replayer->device, replayer->time_query_pool, 0,
                                                  replayer->time_query_count, timestamps.size() * sizeof(uint64_t),
                                                  timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
      gamma_check_vk_result(vkGetQueryPoolResults(replayer->device, replayer->size_query_pool, 0,
                                                  replayer->size_query_count, sizes.size() * sizeof(uint64_t),
                                                  sizes.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));

      timestamp_index = 0;
      /* iterate in the exact same order as timestamps were written */
      for (bool blas = true;; blas = false) {
         for (unsigned i = 0; i < permuted_builds.size(); ++i) {
            auto build = permuted_builds[i];
            if ((build->build_info.type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) != blas)
               continue;

            uint64_t build_time =
               (timestamps[timestamp_index + 1] - timestamps[timestamp_index]) * replayer->timestamp_period;
            replayer->results[run][inv_permutation[i]] = BenchmarkResult{
               .build_time = build_time,
               .compacted_size = sizes[i],
            };
            timestamp_index += 2;
         }
         if (!blas)
            break;
      }
   }
}
