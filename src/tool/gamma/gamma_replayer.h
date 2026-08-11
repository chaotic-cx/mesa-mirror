/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <filesystem>
#include "util/gamma_format.h"
#include "vulkan/vulkan_core.h"
#include "gamma_device.h"

enum class gamma_result_unit {
   nanoseconds,
   bytes,
};

struct gamma_replayer;

struct gamma_acceleration_structure_build {
   gamma_replayer *replayer = nullptr;

   VkAccelerationStructureBuildGeometryInfoKHR build_info;
   std::vector<VkAccelerationStructureGeometryKHR> geom_infos;
   std::vector<VkAccelerationStructureBuildRangeInfoKHR> range_infos;
   std::shared_ptr<gamma_backed_buffer> as_storage;
   std::shared_ptr<gamma_backed_buffer> build_scratch;
   std::shared_ptr<gamma_backed_buffer> prim_data_buffer;
   VkAccelerationStructureKHR structure;
   void *prim_data;
   size_t prim_data_size;
   uint32_t total_prim_count;

   gamma_acceleration_structure_build(gamma_replayer *replayer);

   void load(char *chunk_data, uint32_t chunk_size);
};

struct BenchmarkResult {
   uint64_t build_time;
   uint64_t compacted_size;
};

struct gamma_replayer : gamma_device {
   std::filesystem::path path;
   gamma_header header;
   std::vector<std::unique_ptr<gamma_acceleration_structure_build>> builds;

   double timestamp_period;
   VkQueryPool time_query_pool = VK_NULL_HANDLE;
   VkQueryPool size_query_pool = VK_NULL_HANDLE;
   VkCommandPool command_pool = VK_NULL_HANDLE;
   uint32_t time_query_count;
   uint32_t size_query_count;

   gamma_result_unit result_unit;
   bool export_csv;
   bool print_verbose;

   PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
   PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
   PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
   PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
   PFN_vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdWriteAccelerationStructuresPropertiesKHR;

   std::vector<std::vector<BenchmarkResult>> results;

   void print_results(const char *dst_name, FILE *dst);

   void load(FILE *in);
   void populate(gamma_acceleration_structure_build *build);
   void build(gamma_acceleration_structure_build *build);
   void destroy_build(gamma_acceleration_structure_build *build);
};

int gamma_replayer_init(gamma_replayer *replayer);
void gamma_replayer_destroy(gamma_replayer *replayer);

void gamma_replayer_run(gamma_replayer *replayer, uint32_t permutation_seed, uint32_t num_test_runs);
