/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "hwdef/rogue_hw_defs.h"
#include "hwdef/rogue_hw_utils.h"

#include "pdsc/pdsc_isa_pack.h"
#include "pdsc/pdsc_precomp_programs.h"
#include "pdsc/programs/pdsc_xfb_init.h"
#include "pdsc/programs/pdsc_xfb_main.h"

#include "pvr_buffer.h"
#include "pvr_cmd_buffer.h"
#include "pvr_device.h"
#include "pvr_entrypoints.h"
#include "pvr_limits.h"
#include "pvr_macros.h"
#include "pvr_physical_device.h"
#include "pvr_pipeline.h"
#include "pvr_query.h"
#include "pvr_types.h"
#include "pvr_xfb.h"

#include "util/u_math.h"

#include "vk_alloc.h"
#include "vk_command_buffer.h"

static uint32_t
pvr_get_output_primitive_vertex_count(VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return 1;

   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
      return 2;

   case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
      return 3;

   default:
      UNREACHABLE("Undefined primitive topology");
   }
}

static void
pvr_destroy_xfb_program_input(const VkAllocationCallbacks *allocator,
                              struct pvr_xfb_program_input *config)
{
   vk_free(allocator, config->data.varying_info_arr);
   config->data.varying_info_arr = NULL;

   vk_free(allocator, config->data.buffer_info);
   config->data.buffer_info = NULL;
}

static VkResult pvr_create_xfb_streamout_program_config(
   struct pvr_cmd_buffer *cmd_buffer,
   struct pvr_xfb_program_input *input_config)
{
   struct pvr_cmd_buffer_xfb_state *recording_state =
      &cmd_buffer->state.xfb_state;
   const struct pvr_graphics_pipeline *pipeline =
      cmd_buffer->state.gfx_pipeline;
   const struct pvr_xfb_shader_state *xfb_shader_state =
      &pipeline->shader_state.xfb_shader_state;
   const VkAllocationCallbacks *const allocator = &cmd_buffer->device->vk.alloc;
   struct pvr_xfb_binding *xfb_bindings;
   struct vk_dynamic_graphics_state *const dynamic_state =
      &cmd_buffer->vk.dynamic_graphics_state;
   const VkPrimitiveTopology topology = dynamic_state->ia.primitive_topology;

   input_config->primitive_type =
      pvr_get_output_primitive_vertex_count(topology);
   input_config->varying_count = xfb_shader_state->varying_count;

   input_config->data.buffer_info = vk_alloc(
      allocator,
      sizeof(*input_config->data.buffer_info) * xfb_shader_state->varying_count,
      8,
      VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!input_config->data.buffer_info)
      goto err_out_of_memory;

   input_config->data.varying_info_arr =
      vk_alloc(allocator,
               sizeof(*input_config->data.varying_info_arr) *
                  xfb_shader_state->varying_count,
               8,
               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!input_config->data.varying_info_arr)
      goto err_out_of_memory;

   struct pvr_xfb_varying *varyings = xfb_shader_state->varyings;
   uint32_t varying_count = xfb_shader_state->varying_count;

   if (varying_count) {
      for (uint32_t i = 0; i < varying_count; i++) {
         uint32_t vbuff_index = varyings[i].xfb_buffer;

         /* Varying info */
         input_config->data.varying_info_arr[i].varying_index =
            varyings[i].register_start;
         input_config->data.varying_info_arr[i].dma_count =
            varyings[i].register_count;
         input_config->data.varying_info_arr[i].output_offset =
            varyings[i].xfb_offset;
         input_config->data.varying_info_arr[i].varying_stride =
            varyings[i].xfb_stride;

         /* Buffer info */
         xfb_bindings = &recording_state->bindings[vbuff_index];
         input_config->data.buffer_info[i].buffer_address =
            xfb_bindings->buffer_addr;
         input_config->data.buffer_info[i].buffer_limit =
            xfb_bindings->buffer_limit;
      }
   }

   return VK_SUCCESS;

err_out_of_memory:
   pvr_destroy_xfb_program_input(allocator, input_config);

   return vk_error(&cmd_buffer->device->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
}

static VkResult
pvr_setup_init_config(struct pvr_cmd_buffer *cmd_buffer,
                      struct pvr_pds_xfb_init_program_input *init_input,
                      struct pvr_xfb_program_input *input_config)
{
   struct pvr_cmd_buffer_xfb_state *xfb_state = &cmd_buffer->state.xfb_state;
   const VkAllocationCallbacks *const allocator = &cmd_buffer->device->vk.alloc;
   bool has_valid_counter_buffer = false;

   init_input->device = cmd_buffer->device;

   if (xfb_state->counter_buffer_count) {
      init_input->counter_buffer_count = xfb_state->counter_buffer_count;
      init_input->data.counter_buffers_addr =
         vk_alloc(allocator,
                  sizeof(*init_input->data.counter_buffers_addr) *
                     init_input->counter_buffer_count,
                  8U,
                  VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);

      if (!init_input->data.counter_buffers_addr)
         return vk_error(&cmd_buffer->device->vk, VK_ERROR_OUT_OF_HOST_MEMORY);

      for (uint32_t i = 0; i < init_input->counter_buffer_count; i++) {
         uint64_t addr = xfb_state->bindings[i].counter_buffer_addr.addr;

         init_input->data.counter_buffers_addr[i].addr = addr;
         has_valid_counter_buffer |= !!addr;
      }

      init_input->load_from_counter_buffers = has_valid_counter_buffer;
      if (!has_valid_counter_buffer)
         init_input->counter_buffer_count = 0;
   } else {
      init_input->load_from_counter_buffers = false;
      init_input->counter_buffer_count = 0;
      init_input->data.counter_buffers_addr = NULL;
   }

   assert(input_config->data.buffer_info);
   assert(input_config->data.varying_info_arr);

   init_input->data.buffer_info = input_config->data.buffer_info;
   init_input->data.varying_info_arr = input_config->data.varying_info_arr;

   init_input->data.counter_buffer_offsets_addr =
      &cmd_buffer->state.gfx_pipeline->shader_state.xfb_shader_state
          .counter_buffer_offsets->dev_addr;
   init_input->primitive_type = input_config->primitive_type;

   return VK_SUCCESS;
}

static VkResult
pvr_setup_end_program_config(struct pvr_cmd_buffer *cmd_buffer,
                             struct pvr_pds_xfb_end_program_input *input)
{
   struct pvr_cmd_buffer_xfb_state *xfb_state = &cmd_buffer->state.xfb_state;
   const VkAllocationCallbacks *const allocator = &cmd_buffer->device->vk.alloc;
   bool has_valid_counter_buffer = false;

   const struct pvr_graphics_pipeline *pipeline =
      cmd_buffer->state.gfx_pipeline;
   const struct pvr_xfb_shader_state *xfb_shader_state =
      &pipeline->shader_state.xfb_shader_state;
   struct vk_dynamic_graphics_state *const dynamic_state =
      &cmd_buffer->vk.dynamic_graphics_state;
   const uint32_t prim_vert_count = pvr_get_output_primitive_vertex_count(
      dynamic_state->ia.primitive_topology);

   if (xfb_state->counter_buffer_count) {
      input->counter_buffer_count = xfb_state->counter_buffer_count;
      assert(input->counter_buffer_count <= PVR_XFB_MAX_BUFFERS);

      input->data.counter_buffers_addr =
         vk_alloc(allocator,
                  sizeof(*input->data.counter_buffers_addr) *
                     input->counter_buffer_count,
                  8U,
                  VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
      if (!input->data.counter_buffers_addr)
         return vk_error(&cmd_buffer->device->vk, VK_ERROR_OUT_OF_HOST_MEMORY);

      for (uint32_t i = 0; i < input->counter_buffer_count; i++) {
         pvr_dev_addr_t *counter_buffers_addr =
            &input->data.counter_buffers_addr[i];

         if (!xfb_state->bindings[i].last_counter_buffer_addr.addr) {
            counter_buffers_addr->addr = 0;
         } else {
            counter_buffers_addr->addr =
               xfb_state->bindings[i].last_counter_buffer_addr.addr;
            has_valid_counter_buffer = true;

            input->buffer_strides[i] =
               xfb_shader_state->buffer_strides[i] * prim_vert_count;
         }
      }

      input->has_store_counter_buffers = has_valid_counter_buffer;
      if (!has_valid_counter_buffer)
         input->counter_buffer_count = 0;
   } else {
      input->has_store_counter_buffers = false;
      input->counter_buffer_count = 0;
      input->data.counter_buffers_addr = NULL;
   }

   input->data.counter_buffer_offsets_addr =
      &xfb_shader_state->counter_buffer_offsets->dev_addr;

   return VK_SUCCESS;
}

static VkResult pvr_xfb_end(struct pvr_cmd_buffer *cmd_buffer)
{
   struct pvr_cmd_buffer_state *const state = &cmd_buffer->state;
   struct pvr_pds_xfb_end_program_input end_input = { 0 };
   struct pvr_program_info *end_program_info;
   struct pvr_pds_upload *end_pds_program;
   pdsc_program **end_program;
   VkResult result;

   const struct pvr_device_info *dev_info =
      &cmd_buffer->device->pdevice->dev_info;
   const uint32_t cache_line_size = pvr_get_slc_cache_line_size(dev_info);

   end_program_info = &cmd_buffer->state.xfb_state.pds_end_program;
   end_pds_program = &end_program_info->pds_program;

   result = pvr_setup_end_program_config(cmd_buffer, &end_input);
   if (result != VK_SUCCESS)
      return result;

   end_program = &cmd_buffer->device->pdsc_precomp_programs.xfb_end;
   *end_program = pdsc_precomp_xfb_end_variant(NULL, 0);
   pdsc_finalize_program(*end_program);

   end_program_info->temp_size = pdsc_program_temps_used(*end_program);

   pdsc_ref counter_buffer_ld_dest =
      pdsc_load_ref(*end_program, PVR_XFB_END_REF_ID_COUNTER_BUFFER_LD_DEST);
   struct pdsc_ld64_src0 counter_buffers = {
      .slcmode = PDSC_SLC_MODE_LD_CACHED,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,

      .dest = counter_buffer_ld_dest,

      .cmode = PDSC_CMODE_LD_CACHED,
      .count8 = end_input.counter_buffer_count,
      .srcadd = end_input.has_store_counter_buffers
                   ? end_input.data.counter_buffer_offsets_addr[0].addr
                   : 0,
   };

   struct pdsc_st32_src0 st_src0 = {
      .cmode = PDSC_CMODE_ST_WRITE_THROUGH,
      .slcmode = PDSC_SLC_MODE_ST_WRITE_THROUGH,

      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,

      .count4 = end_input.counter_buffer_count,

      .src = pdsc_load_ref(*end_program,
                           PVR_XFB_END_REF_ID_COUNTER_BUFFER_ST_SRC0),
      .dstadd = end_input.has_store_counter_buffers
                   ? end_input.data.counter_buffers_addr[0].addr
                   : 0,
   };

   /* While a stream query is active, add this segment's primitive counts to
    * its result elem, as the next transform feedback begin resets them.
    */
   const struct pvr_query_pool *const query_pool =
      state->query_pools[PVR_QUERY_POOL_INDEX_XFB_STREAM_0];
   const bool query_active = !!query_pool;
   pvr_dev_addr_t query_result_addr = PVR_DEV_ADDR_INVALID;

   if (query_active) {
      const uint32_t query_idx =
         state->query_pool_indices[PVR_QUERY_POOL_INDEX_XFB_STREAM_0];

      query_result_addr =
         PVR_DEV_ADDR_OFFSET(query_pool->result_buffer->dev_addr,
                             query_pool->result_elem_stride * query_idx);
   }

   const struct pdsc_ld64_src0 query_result_ld_src0 = {
      .slcmode = PDSC_SLC_MODE_LD_CACHED,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .dest = pdsc_load_ref(*end_program,
                            PVR_XFB_END_REF_ID_QUERY_RESULT_LD_DEST),
      .cmode = PDSC_CMODE_LD_CACHED,
      .count8 = 1, /* 1 × 8 bytes: both query result counters */
      .srcadd = query_result_addr.addr,
   };

   const struct pdsc_st32_src0 query_result_st_src0 = {
      .cmode = PDSC_CMODE_ST_WRITE_THROUGH,
      .slcmode = PDSC_SLC_MODE_ST_WRITE_THROUGH,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .count4 = 2, /* 2 × 4-byte words: query result counters */
      .src = pdsc_load_ref(*end_program,
                           PVR_XFB_END_REF_ID_QUERY_RESULT_ST_SRC0),
      .dstadd = query_result_addr.addr,
   };

   pdsc_patch patches_end[] = {
      { .patch_id = PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_COUNT,
        .value = end_input.counter_buffer_count },
      { .patch_id = PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_0,
        .value = end_input.buffer_strides[0] },
      { .patch_id = PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_1,
        .value = end_input.buffer_strides[1] },
      { .patch_id = PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_2,
        .value = end_input.buffer_strides[2] },
      { .patch_id = PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_STRIDE_3,
        .value = end_input.buffer_strides[3] },
      { .patch_id = PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_LD_SRC0,
        .value = end_input.has_store_counter_buffers
                    ? pdsc_ld64_src0_pack(*end_program, &counter_buffers)
                    : 0 },

      { .patch_id = PVR_XFB_END_PATCH_ID_COUNTER_BUFFER_ST_SRC0,
        .value = end_input.has_store_counter_buffers
                    ? pdsc_st32_src0_pack(*end_program, &st_src0)
                    : 0 },

      { .patch_id = PVR_XFB_END_PATCH_ID_QUERY_RESULT_LD_SRC0,
        .value = query_active
                    ? pdsc_ld64_src0_pack(*end_program, &query_result_ld_src0)
                    : 0 },

      { .patch_id = PVR_XFB_END_PATCH_ID_QUERY_RESULT_ST_SRC0,
        .value = query_active
                    ? pdsc_st32_src0_pack(*end_program, &query_result_st_src0)
                    : 0 },
   };

   assert((ARRAY_SIZE(patches_end) < pdsc_program_data_size(*end_program)));

   pdsc_patch_program(*end_program, ARRAY_SIZE(patches_end), patches_end, NULL);

   result = pvr_arch_cmd_buffer_upload_pds(
      cmd_buffer,
      pdsc_program_data(*end_program),
      pdsc_program_data_size(*end_program) / sizeof(uint32_t),
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      pdsc_program_code(*end_program),
      pdsc_program_code_size(*end_program) / sizeof(uint32_t),
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      cache_line_size,
      end_pds_program);
   if (result != VK_SUCCESS) {
      pvr_cmd_buffer_set_error_unwarned(cmd_buffer, result);
      goto out;
   }

   result = pvr_arch_emit_xfb_sync_state(cmd_buffer, end_program_info);
   if (result != VK_SUCCESS)
      goto out;

   if (state->current_sub_cmd &&
       state->current_sub_cmd->type == PVR_SUB_CMD_TYPE_GRAPHICS) {
      pvr_cmd_buffer_insert_mid_frag_barrier_event(cmd_buffer,
                                                   PVR_PIPELINE_STAGE_GEOM_BIT,
                                                   PVR_PIPELINE_STAGE_GEOM_BIT);
   }

out:
   vk_free(&cmd_buffer->device->vk.alloc, end_input.data.counter_buffers_addr);
   end_input.data.counter_buffers_addr = NULL;
   return result;
}

static VkResult pvr_pds_xfb_ptemp_buffer_init(
   struct pvr_pds_xfb_init_program_input *input,
   const struct pvr_xfb_shader_state *xfb_shader_state,
   const pdsc_program *p)
{
   struct pvr_device *const device = input->device;
   VkResult result;

   const uint32_t buffer_group_count = xfb_shader_state->buffer_group_count;
   const uint32_t ptemp_count =
      PVR_XFB_MAX_BUFFERS * PVR_XFB_PTEMP_ENTRIES_PER_VARYING;

   uint16_t primitive_type = input->primitive_type;

   uint64_t *ptemp_values = vk_zalloc(&device->vk.alloc,
                                      sizeof(*ptemp_values) * ptemp_count,
                                      16,
                                      VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!ptemp_values)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   /* Process each buffer group */
   uint32_t var_i = 0;
   for (uint32_t g = 0; g < buffer_group_count; g++) {
      uint32_t group_start = var_i;
      var_i += xfb_shader_state->group_varying_count[g];

      /* Get ptemp indices for this group (fixed stride-3 layout) */
      uint32_t src0_pt = g * PVR_XFB_PTEMP_ENTRIES_PER_VARYING;
      uint32_t src1_pt = g * PVR_XFB_PTEMP_ENTRIES_PER_VARYING + 1;
      uint32_t src3_pt = g * PVR_XFB_PTEMP_ENTRIES_PER_VARYING + 2;

      const uint32_t src0_ptemp64_idx_u32 = src0_pt * 2;
      const uint32_t src1_ptemp64_idx_u32 = src1_pt * 2;
      const uint16_t src0_ptemp64_idx = (uint16_t)src0_ptemp64_idx_u32;
      const uint16_t src1_ptemp64_idx = (uint16_t)src1_ptemp64_idx_u32;

      /* SRC0: address register — shared across all varyings in group.*/
      struct pdsc_stm_src0 src0 = {
         .ptemp_prim_needed_written = pdsc_ref_ptemp64(src1_ptemp64_idx),
         .ptemp_address = pdsc_ref_ptemp64(src0_ptemp64_idx),
         .address = input->data.buffer_info[group_start].buffer_address.addr,
      };
      ptemp_values[src0_pt] = pdsc_stm_src0_pack(p, &src0);

      /* SRC1: primwritten/primneeded — initialized to zero */
      ptemp_values[src1_pt] = 0ULL;

      /* SRC3 */
      uint32_t vo_size =
         (input->data.varying_info_arr[group_start].varying_stride) /
         sizeof(uint32_t);
      struct pdsc_stm_src3 src3 = {
         .primtype = (enum pdsc_primtype)primitive_type,
         .vosize = vo_size,
         .eop = false,
         .limit = input->data.buffer_info[group_start].buffer_limit.addr,
      };
      ptemp_values[src3_pt] = pdsc_stm_src3_pack(p, &src3);
   }

   for (uint32_t g = 0;
        g < ARRAY_SIZE(input->data.ptemp_memory_buffer_addr_array);
        g++) {
      result =
         pvr_gpu_upload(device,
                        device->heaps.general_heap,
                        &ptemp_values[g * PVR_XFB_PTEMP_ENTRIES_PER_VARYING],
                        sizeof(uint64_t) * PVR_XFB_PTEMP_ENTRIES_PER_VARYING,
                        16, /* 128-bit alignment */
                        &input->data.ptemp_memory_buffer_addr_array[g]);
      if (result != VK_SUCCESS)
         break;
   }

   vk_free(&device->vk.alloc, ptemp_values);
   return result;
}

static VkResult pvr_setup_xfb_event_program(struct pvr_cmd_buffer *cmd_buffer)
{
   struct pvr_device *device = cmd_buffer->device;
   struct pvr_xfb_program_input xfb_config = { 0 };
   struct pvr_pds_xfb_init_program_input init_input = { 0 };
   struct pvr_program_info *init_pds_upload =
      &cmd_buffer->state.xfb_state.pds_init_program;
   struct pvr_program_info *program_pds_upload =
      &cmd_buffer->state.xfb_state.pds_streamout_program;
   program_pds_upload->pds_program = (struct pvr_pds_upload){ 0 };
   init_pds_upload->pds_program = (struct pvr_pds_upload){ 0 };
   pdsc_program *xfb_main_prog = NULL;
   pdsc_program **p;

   const struct pvr_device_info *dev_info =
      &cmd_buffer->device->pdevice->dev_info;
   const uint32_t cache_line_size = pvr_get_slc_cache_line_size(dev_info);
   VkResult result;

   result = pvr_create_xfb_streamout_program_config(cmd_buffer, &xfb_config);
   if (result != VK_SUCCESS)
      goto out;

   const struct pvr_graphics_pipeline *pipeline =
      cmd_buffer->state.gfx_pipeline;
   const struct pvr_xfb_shader_state *xfb_shader_state =
      &pipeline->shader_state.xfb_shader_state;

   /* Build stream out program */
   uint32_t group_varying_counts[PVR_XFB_MAX_BUFFERS] = { 0 };

   assert(xfb_shader_state->buffer_group_count <=
          ARRAY_SIZE(xfb_shader_state->group_varying_count));

   for (uint32_t g = 0; g < xfb_shader_state->buffer_group_count; g++)
      group_varying_counts[g] = xfb_shader_state->group_varying_count[g];

   xfb_main_prog =
      pdsc_xfb_main_runtime_gen(NULL,
                                0,
                                xfb_shader_state->buffer_group_count,
                                group_varying_counts);

   program_pds_upload->temp_size = pdsc_program_temps_used(xfb_main_prog);

   /* Build SRC2 values for patching */
   assert(xfb_config.varying_count <= PVR_XFB_MAX_TOTAL_VARYINGS);
   struct pdsc_stm_src2 stm_src2[PVR_XFB_MAX_TOTAL_VARYINGS];
   for (uint32_t i = 0; i < xfb_config.varying_count; i++) {
      const struct pvr_pds_xfb_varying_info *varying_info =
         &xfb_config.data.varying_info_arr[i];

      stm_src2[i] = (struct pdsc_stm_src2){
         .vioff = varying_info->varying_index,
         .dmasize = varying_info->dma_count,
         .vooff = varying_info->output_offset / sizeof(uint32_t),
      };
   }

   pdsc_patch_xfb_main(xfb_main_prog, NULL, xfb_config.varying_count, stm_src2);

   result = pvr_arch_cmd_buffer_upload_pds(
      cmd_buffer,
      pdsc_program_data(xfb_main_prog),
      pdsc_program_data_size(xfb_main_prog) / sizeof(uint32_t),
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      pdsc_program_code(xfb_main_prog),
      pdsc_program_code_size(xfb_main_prog) / sizeof(uint32_t),
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      cache_line_size,
      &program_pds_upload->pds_program);
   if (result != VK_SUCCESS) {
      pvr_cmd_buffer_set_error_unwarned(cmd_buffer, result);
      goto out;
   }

   result = pvr_setup_init_config(cmd_buffer, &init_input, &xfb_config);
   if (result != VK_SUCCESS)
      goto out;

   p = &device->pdsc_precomp_programs.xfb_init;
   *p = pdsc_precomp_xfb_init_variant(NULL, 0);
   pdsc_finalize_program(*p);
   result = pvr_pds_xfb_ptemp_buffer_init(&init_input, xfb_shader_state, *p);
   if (result != VK_SUCCESS)
      goto out;

   for (uint32_t g = 0;
        g < ARRAY_SIZE(init_input.data.ptemp_memory_buffer_addr_array);
        g++) {
      if (init_input.data.ptemp_memory_buffer_addr_array[g])
         list_add(&init_input.data.ptemp_memory_buffer_addr_array[g]->link,
                  &cmd_buffer->bo_list);
   }

   init_pds_upload->temp_size = pdsc_program_temps_used(*p);

   uint32_t data_size = pdsc_program_data_size(*p) / sizeof(uint32_t);
   uint32_t code_size = pdsc_program_code_size(*p) / sizeof(uint32_t);

   pvr_dev_addr_t counter_buffer_addr = { 0 };
   uint32_t num_dwords = 0;

   if (init_input.load_from_counter_buffers &&
       init_input.counter_buffer_count > 0) {
      counter_buffer_addr = init_input.data.counter_buffers_addr[0].addr != 0
                               ? init_input.data.counter_buffers_addr[0]
                               : PVR_DEV_ADDR_INVALID;
      uint32_t misalignment = counter_buffer_addr.addr & 0b1111;
      num_dwords = misalignment / 4;
      counter_buffer_addr.addr &= ~0b1111;
   }

   pdsc_ref ptemp_ld_dest[PVR_XFB_MAX_BUFFERS] = {
      pdsc_load_ref(*p, PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_0),
      pdsc_load_ref(*p, PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_1),
      pdsc_load_ref(*p, PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_2),
      pdsc_load_ref(*p, PVR_XFB_INIT_REF_ID_PTEMPS_LD_DEST_3),
   };

   struct pdsc_ld64_src0 ptemp_buffers[PVR_XFB_MAX_BUFFERS];
   const uint32_t init_group_count = xfb_shader_state->buffer_group_count;
   for (uint32_t g = 0; g < ARRAY_SIZE(ptemp_buffers); g++) {
      ptemp_buffers[g] = (struct pdsc_ld64_src0){
         .slcmode = PDSC_SLC_MODE_LD_CACHED,
         .ccpri = 0,
         .ccslc = PDSC_CCSLC_LD_NORMAL,
         .ccmcu = PDSC_CCMCU_LD_NORMAL,
         .dest = ptemp_ld_dest[g],
         .cmode = PDSC_CMODE_LD_CACHED,
         .count8 = PVR_XFB_PTEMP_ENTRIES_PER_VARYING,
         .srcadd =
            init_input.data.ptemp_memory_buffer_addr_array[g]->dev_addr.addr,
      };
   }

   pdsc_ref counter_buffer_ld_dest =
      pdsc_load_ref(*p, PVR_XFB_INIT_REF_ID_COUNTER_BUFFER_LD_DEST);
   struct pdsc_ld64_src0 counter_buffers = {
      .slcmode = PDSC_SLC_MODE_LD_CACHED,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,

      .dest = counter_buffer_ld_dest,

      .cmode = PDSC_CMODE_LD_CACHED,
      .count8 = DIV_ROUND_UP(init_input.counter_buffer_count + num_dwords, 2),
      .srcadd = counter_buffer_addr.addr,
   };

   struct pdsc_st32_src0 st_src0 = {
      .cmode = PDSC_CMODE_ST_WRITE_THROUGH,
      .slcmode = PDSC_SLC_MODE_ST_WRITE_THROUGH,

      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .count4 = MAX2(cmd_buffer->state.xfb_state.counter_buffer_count, 1),

      .src = pdsc_load_ref(*p, PVR_XFB_INIT_REF_ID_COUNTER_BUFFER_ST_SRC0),
      .dstadd = init_input.data.counter_buffer_offsets_addr->addr,
   };

   pdsc_patch patches[] = {
      { .patch_id = PVR_XFB_INIT_PATCH_ID_PTEMP_LD_COUNT,
        .value = init_group_count },

      { .patch_id = PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_0,
        .value = (0 < init_group_count
                     ? pdsc_ld64_src0_pack(*p, &ptemp_buffers[0])
                     : 0) },

      { .patch_id = PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_1,
        .value = (1 < init_group_count
                     ? pdsc_ld64_src0_pack(*p, &ptemp_buffers[1])
                     : 0) },

      { .patch_id = PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_2,
        .value = (2 < init_group_count
                     ? pdsc_ld64_src0_pack(*p, &ptemp_buffers[2])
                     : 0) },

      { .patch_id = PVR_XFB_INIT_PATCH_ID_PTEMP_LD_SRC0_3,
        .value = (3 < init_group_count
                     ? pdsc_ld64_src0_pack(*p, &ptemp_buffers[3])
                     : 0) },

      { .patch_id = PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_LD_SRC0,
        .value = init_input.load_from_counter_buffers
                    ? pdsc_ld64_src0_pack(*p, &counter_buffers)
                    : 0 },

      { .patch_id = PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_ST_SRC0,
        .value = pdsc_st32_src0_pack(*p, &st_src0) },

      { .patch_id = PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_DWORD_SELECT,
        .value = num_dwords },

      { .patch_id = PVR_XFB_INIT_PATCH_ID_COUNTER_BUFFER_COUNT,
        .value = init_input.counter_buffer_count },
   };

   assert((ARRAY_SIZE(patches) < pdsc_program_data_size(*p)));
   pdsc_patch_program(*p, ARRAY_SIZE(patches), patches, NULL);

   result = pvr_arch_cmd_buffer_upload_pds(
      cmd_buffer,
      pdsc_program_data(*p),
      data_size,
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      pdsc_program_code(*p),
      code_size,
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      cache_line_size,
      &init_pds_upload->pds_program);

out:
   ralloc_free(xfb_main_prog);
   vk_free(&cmd_buffer->device->vk.alloc, init_input.data.counter_buffers_addr);
   vk_free(&cmd_buffer->device->vk.alloc, init_input.data.varying_info_arr);
   vk_free(&cmd_buffer->device->vk.alloc, init_input.data.buffer_info);

   init_input.data.counter_buffers_addr = NULL;
   init_input.data.varying_info_arr = NULL;
   init_input.data.buffer_info = NULL;
   return result;
}

/* The sequence of generating PDS programs -> Main, init, and sync */
static VkResult pvr_xfb_start(struct pvr_cmd_buffer *cmd_buffer)
{
   struct pvr_cmd_buffer_xfb_state *xfb_state = &cmd_buffer->state.xfb_state;
   struct pvr_program_info *init_program_info = &xfb_state->pds_init_program;
   VkResult result;

   /* Generate PDS programs */
   result = pvr_setup_xfb_event_program(cmd_buffer);
   if (result != VK_SUCCESS) {
      pvr_cmd_buffer_set_error_unwarned(cmd_buffer, result);
      return result;
   }

   /* Emit the sync state */
   result = pvr_arch_emit_xfb_sync_state(cmd_buffer, init_program_info);
   if (result != VK_SUCCESS)
      return result;

   /* Set XFB main program PPP state */
   pvr_arch_set_xfb_main_program_ppp_state(cmd_buffer);

   return VK_SUCCESS;
}

VkResult PVR_PER_ARCH(validate_xfb_state)(struct pvr_cmd_buffer *cmd_buffer)
{
   struct pvr_cmd_buffer_xfb_state *xfb_state = &cmd_buffer->state.xfb_state;
   const struct pvr_graphics_pipeline *pipeline =
      cmd_buffer->state.gfx_pipeline;
   const struct pvr_xfb_shader_state *xfb_shader_state;
   struct vk_dynamic_graphics_state *const dynamic_state =
      &cmd_buffer->vk.dynamic_graphics_state;
   uint32_t output_primitive_vertex_count;
   VkResult result;

   if (!xfb_state->is_active)
      return VK_SUCCESS;

   xfb_shader_state = &pipeline->shader_state.xfb_shader_state;

   if (!xfb_shader_state->varying_count)
      return VK_SUCCESS;

   output_primitive_vertex_count = pvr_get_output_primitive_vertex_count(
      dynamic_state->ia.primitive_topology);

   /* Topology may be unset at Begin; rebuild with the real primitive type. */
   if (xfb_state->output_primitive_vertex_count !=
       output_primitive_vertex_count) {
      /* TODO: handle class changes mid-segment
       * (dynamicPrimitiveTopologyUnrestricted).
       * TODO: this end may store to a stale last_counter_buffer_addr.
       */
      result = pvr_xfb_end(cmd_buffer);
      if (result != VK_SUCCESS)
         return result;

      /* Rebuild init/stream-out with the new primitive type.
       */
      xfb_state->output_primitive_vertex_count = output_primitive_vertex_count;

      result = pvr_xfb_start(cmd_buffer);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

void PVR_PER_ARCH(CmdBindTransformFeedbackBuffersEXT)(
   VkCommandBuffer commandBuffer,
   uint32_t firstBinding,
   uint32_t bindingCount,
   const VkBuffer *pBuffers,
   const VkDeviceSize *pOffsets,
   const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(pvr_cmd_buffer, cmd_buffer, commandBuffer);
   struct pvr_cmd_buffer_xfb_state *xfb_state = &cmd_buffer->state.xfb_state;

   PVR_CHECK_COMMAND_BUFFER_BUILDING_STATE(cmd_buffer);

   for (uint32_t i = 0; i < ARRAY_SIZE(xfb_state->bindings); i++) {
      struct pvr_xfb_binding *binding = &xfb_state->bindings[i];

      if (i < firstBinding || i >= firstBinding + bindingCount) {
         binding->buffer_addr = PVR_DEV_ADDR_INVALID;
         binding->buffer_limit = PVR_DEV_ADDR_INVALID;
         binding->buffer_size = 0;
         continue;
      }

      unsigned j = i - firstBinding;
      VK_FROM_HANDLE(pvr_buffer, buffer, pBuffers[j]);

      binding->buffer_addr = PVR_DEV_ADDR_OFFSET(buffer->dev_addr, pOffsets[j]);

      if (!pSizes || pSizes[j] == VK_WHOLE_SIZE) {
         binding->buffer_limit =
            PVR_DEV_ADDR_OFFSET(buffer->dev_addr, buffer->vk.size);
         binding->buffer_size = buffer->vk.size;
      } else {
         binding->buffer_limit =
            PVR_DEV_ADDR_OFFSET(binding->buffer_addr, pSizes[j]);
         binding->buffer_size = pSizes[j];
      }
   }
}

void PVR_PER_ARCH(CmdBeginTransformFeedbackEXT)(
   VkCommandBuffer commandBuffer,
   uint32_t firstCounterBuffer,
   uint32_t counterBufferCount,
   const VkBuffer *pCounterBuffers,
   const VkDeviceSize *pCounterBufferOffsets)
{
   VK_FROM_HANDLE(pvr_cmd_buffer, cmd_buffer, commandBuffer);
   struct pvr_cmd_buffer_state *state = &cmd_buffer->state;
   struct pvr_cmd_buffer_xfb_state *xfb_state = &state->xfb_state;

   PVR_CHECK_COMMAND_BUFFER_BUILDING_STATE(cmd_buffer);

   xfb_state->counter_buffer_count = counterBufferCount;

   for (uint32_t i = 0; i < ARRAY_SIZE(xfb_state->bindings); i++) {
      struct pvr_xfb_binding *binding = &xfb_state->bindings[i];

      if (i < firstCounterBuffer ||
          i >= firstCounterBuffer + counterBufferCount) {
         binding->counter_buffer_addr = PVR_DEV_ADDR_INVALID;
         continue;
      }

      unsigned j = i - firstCounterBuffer;

      if (pCounterBuffers[j] != VK_NULL_HANDLE) {
         VK_FROM_HANDLE(pvr_buffer, buffer, pCounterBuffers[j]);
         uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[j] : 0;

         binding->counter_buffer_addr =
            PVR_DEV_ADDR_OFFSET(buffer->dev_addr, offset);
      } else {
         binding->counter_buffer_addr = PVR_DEV_ADDR_INVALID;
      }
   }

   if (state->gfx_pipeline->shader_state.xfb_shader_state.varying_count) {
      xfb_state->is_active = true;
      xfb_state->output_primitive_vertex_count =
         pvr_get_output_primitive_vertex_count(
            cmd_buffer->vk.dynamic_graphics_state.ia.primitive_topology);
      pvr_xfb_start(cmd_buffer);
   }
}

void PVR_PER_ARCH(CmdEndTransformFeedbackEXT)(
   VkCommandBuffer commandBuffer,
   uint32_t firstCounterBuffer,
   uint32_t counterBufferCount,
   const VkBuffer *pCounterBuffers,
   const VkDeviceSize *pCounterBufferOffsets)
{
   VK_FROM_HANDLE(pvr_cmd_buffer, cmd_buffer, commandBuffer);
   struct pvr_cmd_buffer_state *state = &cmd_buffer->state;
   struct pvr_cmd_buffer_xfb_state *xfb_state = &state->xfb_state;
   struct pvr_sub_cmd *prev_sub_cmd = NULL;
   VkResult result;

   PVR_CHECK_COMMAND_BUFFER_BUILDING_STATE(cmd_buffer);

   xfb_state->counter_buffer_count = counterBufferCount;

   for (uint32_t i = 0; i < ARRAY_SIZE(xfb_state->bindings); i++) {
      struct pvr_xfb_binding *binding = &xfb_state->bindings[i];

      if (i < firstCounterBuffer ||
          i >= firstCounterBuffer + counterBufferCount) {
         binding->last_counter_buffer_addr = PVR_DEV_ADDR_INVALID;
         continue;
      }

      unsigned j = i - firstCounterBuffer;

      if (pCounterBuffers[j] != VK_NULL_HANDLE) {
         VK_FROM_HANDLE(pvr_buffer, buffer, pCounterBuffers[j]);
         uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[j] : 0;

         binding->last_counter_buffer_addr =
            PVR_DEV_ADDR_OFFSET(buffer->dev_addr, offset);
      } else {
         binding->last_counter_buffer_addr = PVR_DEV_ADDR_INVALID;
      }
   }

   xfb_state->is_active = false;

   if (cmd_buffer->state.current_sub_cmd) {
      if (cmd_buffer->state.current_sub_cmd->type ==
          PVR_SUB_CMD_TYPE_GRAPHICS) {
         /* Kick render. */
         cmd_buffer->state.current_sub_cmd->gfx.barrier_store = true;
      }

      prev_sub_cmd = cmd_buffer->state.current_sub_cmd;

      result = pvr_arch_cmd_buffer_end_sub_cmd(cmd_buffer);
      if (result != VK_SUCCESS)
         return;
   }

   if (prev_sub_cmd) {
      result =
         pvr_arch_cmd_buffer_start_sub_cmd(cmd_buffer, prev_sub_cmd->type);
      assert(result == VK_SUCCESS);

      if (prev_sub_cmd->type == PVR_SUB_CMD_TYPE_GRAPHICS) {
         cmd_buffer->state.current_sub_cmd->gfx.dr_info =
            prev_sub_cmd->gfx.dr_info;
         prev_sub_cmd->gfx.dr_info = NULL;

         /* Use existing render setup, but load color attachments from HW
          * BGOBJ.
          */
         cmd_buffer->state.current_sub_cmd->gfx.barrier_load = true;
         cmd_buffer->state.current_sub_cmd->gfx.barrier_store = false;
      }
   }

   if (state->gfx_pipeline->shader_state.xfb_shader_state.varying_count)
      pvr_xfb_end(cmd_buffer);
}

static void
pvr_xfb_query_program_input_init(struct pvr_cmd_buffer *cmd_buffer,
                                 struct pvr_query_pool *pool,
                                 uint32_t query_idx,
                                 struct pvr_pds_xfb_query_program_input *input)
{
   struct pvr_cmd_buffer_xfb_state *xfb_state = &cmd_buffer->state.xfb_state;

   const uint32_t elem_size = pool->result_elem_stride;
   const uint32_t result_offset = elem_size * query_idx;

   input->availability_addr =
      PVR_DEV_ADDR_OFFSET(pool->availability_buffer->dev_addr,
                          query_idx * sizeof(uint32_t));
   input->result_addr =
      PVR_DEV_ADDR_OFFSET(pool->result_buffer->dev_addr, result_offset);
   input->is_active = xfb_state->is_active;
}

static bool
pvr_xfb_query_begin_program_generate(struct pvr_cmd_buffer *cmd_buffer,
                                     struct pvr_query_pool *pool,
                                     uint32_t query_idx)
{
   struct pvr_program_info *pds_query_upload =
      &cmd_buffer->state.xfb_state.pds_query_begin_program;
   struct pvr_pds_xfb_query_program_input input = { 0 };
   const struct pvr_device_info *dev_info =
      &cmd_buffer->device->pdevice->dev_info;
   const uint32_t cache_line_size = pvr_get_slc_cache_line_size(dev_info);
   pdsc_program **query_program;
   VkResult result;

   pvr_xfb_query_program_input_init(cmd_buffer, pool, query_idx, &input);

   query_program = &cmd_buffer->device->pdsc_precomp_programs.xfb_query_begin;
   *query_program = pdsc_precomp_xfb_query_begin_variant(NULL, 0);
   pdsc_finalize_program(*query_program);

   pds_query_upload->temp_size = pdsc_program_temps_used(*query_program);

   const struct pdsc_st32_src0 availability_st_src0 = {
      .cmode = PDSC_CMODE_ST_WRITE_THROUGH,
      .slcmode = PDSC_SLC_MODE_ST_WRITE_THROUGH,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .count4 = 1, /* 1 × 4-byte word: availability flag */
      .src = pdsc_load_ref(*query_program,
                           PVR_XFB_QUERY_BEGIN_REF_ID_AVAILABILITY_ST_SRC0),
      .dstadd = input.availability_addr.addr,
   };
   const struct pdsc_st32_src0 result_st_src0 = {
      .cmode = PDSC_CMODE_ST_WRITE_THROUGH,
      .slcmode = PDSC_SLC_MODE_ST_WRITE_THROUGH,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .count4 = 2, /* 2 × 4-byte words: query result counters */
      .src = pdsc_load_ref(*query_program,
                           PVR_XFB_QUERY_BEGIN_REF_ID_RESULT_ST_SRC0),
      .dstadd = input.result_addr.addr,
   };

   pdsc_patch patches_begin[] = {
      { .patch_id = PVR_XFB_QUERY_BEGIN_PATCH_ID_AVAILABILITY_ST_SRC0,
        .value = pdsc_st32_src0_pack(*query_program, &availability_st_src0) },
      { .patch_id = PVR_XFB_QUERY_BEGIN_PATCH_ID_RESULT_ST_SRC0,
        .value = pdsc_st32_src0_pack(*query_program, &result_st_src0) },
      { .patch_id = PVR_XFB_QUERY_BEGIN_PATCH_ID_OPEN_SEGMENT_SCALE,
        .value = input.is_active ? UINT32_MAX : 0 },
   };

   assert((ARRAY_SIZE(patches_begin) < pdsc_program_data_size(*query_program)));
   pdsc_patch_program(*query_program,
                      ARRAY_SIZE(patches_begin),
                      patches_begin,
                      NULL);

   result = pvr_arch_cmd_buffer_upload_pds(
      cmd_buffer,
      pdsc_program_data(*query_program),
      pdsc_program_data_size(*query_program) / sizeof(uint32_t),
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      pdsc_program_code(*query_program),
      pdsc_program_code_size(*query_program) / sizeof(uint32_t),
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      cache_line_size,
      &pds_query_upload->pds_program);
   if (result != VK_SUCCESS) {
      pvr_cmd_buffer_set_error_unwarned(cmd_buffer, result);
      return false;
   }

   return true;
}

static bool
pvr_xfb_query_store_program_generate(struct pvr_cmd_buffer *cmd_buffer,
                                     struct pvr_query_pool *pool,
                                     uint32_t query_idx)
{
   struct pvr_program_info *pds_query_upload =
      &cmd_buffer->state.xfb_state.pds_query_end_program;
   struct pvr_pds_xfb_query_program_input input = { 0 };
   const struct pvr_device_info *dev_info =
      &cmd_buffer->device->pdevice->dev_info;
   const uint32_t cache_line_size = pvr_get_slc_cache_line_size(dev_info);
   pdsc_program **query_program;
   VkResult result;

   pvr_xfb_query_program_input_init(cmd_buffer, pool, query_idx, &input);

   query_program = &cmd_buffer->device->pdsc_precomp_programs.xfb_query_end;
   *query_program = pdsc_precomp_xfb_query_end_variant(NULL, 0);
   pdsc_finalize_program(*query_program);

   pds_query_upload->temp_size = pdsc_program_temps_used(*query_program);

   const struct pdsc_ld64_src0 result_ld_src0 = {
      .slcmode = PDSC_SLC_MODE_LD_CACHED,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .dest = pdsc_load_ref(*query_program,
                            PVR_XFB_QUERY_END_REF_ID_QUERY_COUNT_LD_DEST),
      .cmode = PDSC_CMODE_LD_CACHED,
      .count8 = 1, /* 1 × 8 bytes: both query result counters */
      .srcadd = input.result_addr.addr,
   };
   const struct pdsc_st32_src0 result_written_st_src0 = {
      .cmode = PDSC_CMODE_ST_WRITE_THROUGH,
      .slcmode = PDSC_SLC_MODE_ST_WRITE_THROUGH,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .count4 = 2, /* 2 × 4-byte words: query result counters */
      .src = pdsc_load_ref(*query_program,
                           PVR_XFB_QUERY_END_REF_ID_RESULT_WRITTEN_ST_SRC0),
      .dstadd = input.result_addr.addr,
   };
   const struct pdsc_st32_src0 availability_st_src0 = {
      .cmode = PDSC_CMODE_ST_WRITE_THROUGH,
      .slcmode = PDSC_SLC_MODE_ST_WRITE_THROUGH,
      .ccpri = 0,
      .ccslc = PDSC_CCSLC_LD_NORMAL,
      .ccmcu = PDSC_CCMCU_LD_NORMAL,
      .count4 = 1, /* 1 × 4-byte word: availability flag */
      .src = pdsc_load_ref(*query_program,
                           PVR_XFB_QUERY_END_REF_ID_AVAILABILITY_ST_SRC0),
      .dstadd = input.availability_addr.addr,
   };

   /* Segments that ended inside the query were already accumulated by the
    * transform feedback end program, so only a segment still open needs
    * adding. A zero patch value skips the load and store.
    */
   const bool add_open_segment = cmd_buffer->state.xfb_state.is_active;

   pdsc_patch patches_end[] = {
      { .patch_id = PVR_XFB_QUERY_END_PATCH_ID_QUERY_COUNT_LD_SRC0,
        .value = add_open_segment
                    ? pdsc_ld64_src0_pack(*query_program, &result_ld_src0)
                    : 0 },
      { .patch_id = PVR_XFB_QUERY_END_PATCH_ID_RESULT_WRITTEN_ST_SRC0,
        .value = add_open_segment
                    ? pdsc_st32_src0_pack(*query_program,
                                          &result_written_st_src0)
                    : 0 },
      { .patch_id = PVR_XFB_QUERY_END_PATCH_ID_AVAILABILITY_ST_SRC0,
        .value = pdsc_st32_src0_pack(*query_program, &availability_st_src0) },
   };

   assert((ARRAY_SIZE(patches_end) < pdsc_program_data_size(*query_program)));
   pdsc_patch_program(*query_program,
                      ARRAY_SIZE(patches_end),
                      patches_end,
                      NULL);

   result = pvr_arch_cmd_buffer_upload_pds(
      cmd_buffer,
      pdsc_program_data(*query_program),
      pdsc_program_data_size(*query_program) / sizeof(uint32_t),
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      pdsc_program_code(*query_program),
      pdsc_program_code_size(*query_program) / sizeof(uint32_t),
      ROGUE_TA_STATE_STREAM_OUT1_PDS_DATA_SIZE_UNIT_SIZE,
      cache_line_size,
      &pds_query_upload->pds_program);
   if (result != VK_SUCCESS) {
      pvr_cmd_buffer_set_error_unwarned(cmd_buffer, result);
      return false;
   }

   return true;
}

void PVR_PER_ARCH(xfb_query_pool_begin)(struct pvr_cmd_buffer *cmd_buffer,
                                        struct pvr_query_pool *pool,
                                        uint32_t query_idx)
{
   struct pvr_cmd_buffer_xfb_state *xfb_state = &cmd_buffer->state.xfb_state;
   struct pvr_cmd_buffer_state *state = &cmd_buffer->state;
   VkResult result;

   if (!pvr_xfb_query_begin_program_generate(cmd_buffer, pool, query_idx))
      return;

   const uint32_t query_pool_index = PVR_QUERY_POOL_INDEX_XFB_STREAM_0;

   state->query_pool_indices[query_pool_index] = query_idx;
   state->query_pools[query_pool_index] = pool;

   result = pvr_arch_emit_xfb_sync_state(cmd_buffer,
                                         &xfb_state->pds_query_begin_program);
   if (result != VK_SUCCESS)
      return;

   pvr_arch_set_xfb_main_program_ppp_state(cmd_buffer);
}

void PVR_PER_ARCH(xfb_query_pool_end)(struct pvr_cmd_buffer *cmd_buffer,
                                      struct pvr_query_pool *pool,
                                      uint32_t query_idx)
{
   struct pvr_cmd_buffer_xfb_state *xfb_state = &cmd_buffer->state.xfb_state;
   struct pvr_cmd_buffer_state *state = &cmd_buffer->state;
   VkResult result;

   if (!pvr_xfb_query_store_program_generate(cmd_buffer, pool, query_idx))
      return;

   state->query_pools[PVR_QUERY_POOL_INDEX_XFB_STREAM_0] = NULL;

   result = pvr_arch_emit_xfb_sync_state(cmd_buffer,
                                         &xfb_state->pds_query_end_program);
   if (result != VK_SUCCESS)
      return;

   pvr_arch_set_xfb_main_program_ppp_state(cmd_buffer);
}
