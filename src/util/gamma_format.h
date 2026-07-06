/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>
#include "vulkan/vulkan_core.h"

enum gamma_driver {
   gamma_driver_radv,
};

struct gamma_header {
   uint32_t version;
   enum gamma_driver driver;
   uint64_t chunk_count;
};

enum gamma_chunk_type {
   gamma_chunk_type_acceleration_structure,
   gamma_chunk_type_build_info,
   gamma_chunk_type_driver_start = 0x80000000,

   gamma_chunk_type_trace_info_radv = gamma_chunk_type_driver_start,
   gamma_chunk_type_ray_history_radv,
};

struct gamma_chunk_header {
   enum gamma_chunk_type type;
   uint64_t size;
};

struct gamma_acceleration_structure_header {
   uint64_t address;
   uint64_t allocated_size;
   uint64_t compacted_size;
   VkAccelerationStructureTypeKHR type;
   VkGeometryTypeKHR geometry_type;
   VkBuildAccelerationStructureFlagsKHR build_flags;
   uint32_t geometry_count;
   uint32_t name_size;
};

/* Source data info for rebuilding the same acceleration
 * structure
 */

struct gamma_acceleration_structure_triangle_primitive {
   float coords[3][3];
};

struct gamma_acceleration_structure_aabb_primitive {
   float min[3];
   float max[3];
};

struct gamma_acceleration_structure_geometry_info {
   uint32_t primitive_count;
   VkGeometryFlagsKHR geometry_flags;
   /* If part of a build_info chunk, followed by primitive_count primitives */
};

struct gamma_acceleration_structure_build_info_header {
   uint32_t dst_acceleration_structure_id;
   VkGeometryTypeKHR geometry_type;
   VkBuildAccelerationStructureFlagsKHR build_flags;
   uint32_t geometry_count;
   /* followed by geometry_count instances of
    * gamma_acceleration_structure_geometry_data
    */
};
