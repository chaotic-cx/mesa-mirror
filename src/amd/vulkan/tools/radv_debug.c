/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_debug.h"
#include "ac_debug.h"
#include "ac_shader_debug_info.h"
#include "radv_buffer.h"
#include "radv_device.h"
#include "radv_entrypoints.h"
#include "vk_common_entrypoints.h"

#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[1;32m"

static uint32_t
radv_find_memory_index(const struct radv_physical_device *pdev, VkMemoryPropertyFlags flags)
{
   const VkPhysicalDeviceMemoryProperties *mem_properties = &pdev->memory_properties;
   for (uint32_t i = 0; i < mem_properties->memoryTypeCount; ++i) {
      if (mem_properties->memoryTypes[i].propertyFlags == flags) {
         return i;
      }
   }
   UNREACHABLE("invalid memory properties");
}

VkResult
radv_backed_buffer_init(struct radv_device *device, struct radv_backed_buffer *buffer, uint64_t size,
                        enum radv_memory_type memory_type, VkBufferUsageFlags2 usage, bool map)
{
   VkMemoryPropertyFlags memory_property_flags = 0;
   switch (memory_type) {
   case radv_memory_type_invisible_vram:
      memory_property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      break;
   case radv_memory_type_visible_vram:
      memory_property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      break;
   case radv_memory_type_gtt:
      memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      break;
   }
   uint32_t memory_type_index = radv_find_memory_index(radv_device_physical(device), memory_property_flags);

   memset(buffer, 0, sizeof(*buffer));

   VkResult result;

   VkBufferCreateInfo buffer_create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext =
         &(VkBufferUsageFlags2CreateInfo){
            .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
            .usage = usage,
         },
      .size = size,
   };
   result = radv_CreateBuffer(radv_device_to_handle(device), &buffer_create_info, NULL, &buffer->buffer);
   if (result != VK_SUCCESS)
      return result;

   VkDeviceBufferMemoryRequirements buffer_mem_req_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
      .pCreateInfo = &buffer_create_info,
   };
   VkMemoryRequirements2 mem_reqs = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
   };
   radv_GetDeviceBufferMemoryRequirements(radv_device_to_handle(device), &buffer_mem_req_info, &mem_reqs);

   VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mem_reqs.memoryRequirements.size,
      .memoryTypeIndex = memory_type_index,
   };
   result = radv_AllocateMemory(radv_device_to_handle(device), &alloc_info, NULL, &buffer->memory);
   if (result != VK_SUCCESS)
      goto fail;

   VkBindBufferMemoryInfo bind_info = {
      .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
      .buffer = buffer->buffer,
      .memory = buffer->memory,
   };
   result = radv_BindBufferMemory2(radv_device_to_handle(device), 1, &bind_info);
   if (result != VK_SUCCESS)
      goto fail;

   if (map) {
      VkMemoryMapInfo mem_map_info = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO,
         .memory = buffer->memory,
         .size = VK_WHOLE_SIZE,
      };
      result = radv_MapMemory2(radv_device_to_handle(device), &mem_map_info, &buffer->map);
      if (result != VK_SUCCESS)
         goto fail;
   }

   return result;

fail:
   radv_backed_buffer_finish(device, buffer);
   return result;
}

void
radv_backed_buffer_finish(struct radv_device *device, struct radv_backed_buffer *buffer)
{
   if (buffer->map) {
      VkMemoryUnmapInfo unmap_info = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO,
         .memory = buffer->memory,
      };
      radv_UnmapMemory2(radv_device_to_handle(device), &unmap_info);
   }

   radv_DestroyBuffer(radv_device_to_handle(device), buffer->buffer, NULL);

   radv_FreeMemory(radv_device_to_handle(device), buffer->memory, NULL);

   memset(buffer, 0, sizeof(*buffer));
}

uint64_t
radv_backed_buffer_get_va(struct radv_device *device, struct radv_backed_buffer *buffer)
{
   if (!buffer->buffer)
      return 0;

   VkBufferDeviceAddressInfo addr_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = buffer->buffer,
   };
   return vk_common_GetBufferDeviceAddress(radv_device_to_handle(device), &addr_info);
}

void
radv_add_split_disasm(const char *disasm, uint64_t start_addr, unsigned *num, struct radv_shader_inst *instructions)
{
   struct radv_shader_inst *last_inst = *num ? &instructions[*num - 1] : NULL;
   const char *next;
   const char *repeat = strstr(disasm, "then repeated");

   while ((next = strchr(disasm, '\n'))) {
      struct radv_shader_inst *inst = &instructions[*num];
      unsigned len = next - disasm;

      if (repeat >= disasm && repeat < next) {
         uint32_t repeat_count;
         sscanf(repeat, "then repeated %u times", &repeat_count);

         for (uint32_t i = 0; i < repeat_count; i++) {
            inst = &instructions[*num];
            memcpy(inst, last_inst, sizeof(struct radv_shader_inst));
            inst->offset = last_inst->offset + last_inst->size * (i + 1);
            (*num)++;
         }

         last_inst = inst;

         disasm = next + 1;
         repeat = strstr(disasm, "then repeated");
         continue;
      }

      if (!memchr(disasm, ';', len)) {
         /* Ignore everything that is not an instruction. */
         disasm = next + 1;
         continue;
      }

      assert(len < ARRAY_SIZE(inst->text));
      memcpy(inst->text, disasm, len);
      inst->text[len] = 0;
      inst->offset = last_inst ? last_inst->offset + last_inst->size : 0;

      const char *semicolon = strchr(disasm, ';');
      assert(semicolon);
      /* 9 = 8 hex digits + a leading space */
      inst->size = (next - semicolon) / 9 * 4;

      snprintf(inst->text + len, ARRAY_SIZE(inst->text) - len, " [PC=0x%" PRIx64 ", off=%u, size=%u]",
               start_addr + inst->offset, inst->offset, inst->size);

      last_inst = inst;
      (*num)++;
      disasm = next + 1;
   }
}

void
radv_print_shader_disasm(const struct radv_shader_binary_debug_info *dbg, uint32_t code_size,
                         struct ac_wave_info *waves, unsigned num_waves, uint64_t start_addr, FILE *f)
{
   /* Nothing to do, output disassembly directly */
   if (dbg->debug_info_count == 0 && !waves) {
      fprintf(f, "%s", dbg->disasm_string);
      return;
   }

   /* Get the list of instructions.
    * Buffer size / 4 is the upper bound of the instruction count.
    */
   unsigned num_inst = 0;
   struct radv_shader_inst *instructions = calloc(code_size / 4, sizeof(struct radv_shader_inst));

   radv_add_split_disasm(dbg->disasm_string, start_addr, &num_inst, instructions);

   /* Maps a given offset inside the shader binary to a given debug_info index */
   struct ac_shader_debug_info **debug_info_mapping = NULL;
   if (dbg->debug_info_count > 0) {
      debug_info_mapping = calloc(code_size, sizeof(struct ac_shader_debug_info *));
      for (uint32_t debug_index = 0; debug_index < dbg->debug_info_count; debug_index++) {
         struct ac_shader_debug_info *debug_info = &dbg->debug_info[debug_index];
         if (debug_info->type != ac_shader_debug_info_src_loc)
            continue;

         /* Such a debug_info marks the end of a range of instructions with location info */
         if (!debug_info->src_loc.file)
            continue;

         uint32_t range_start = debug_info->offset;
         uint32_t range_end = code_size; /* Last debug_info spans until the end of the shader */

         if (debug_index + 1 < dbg->debug_info_count) { /* Only if there is another debug_info after the current one */
            struct ac_shader_debug_info *next_debug_info = &dbg->debug_info[debug_index + 1];
            range_end = next_debug_info->offset;
         }

         for (uint32_t offset = range_start; offset < range_end; offset++)
            debug_info_mapping[offset] = debug_info;
      }
   }

   /* Print instructions with annotations. */
   unsigned i;
   for (i = 0; i < num_inst; i++) {
      struct radv_shader_inst *inst = &instructions[i];

      /* Print source location, if we have it */
      if (debug_info_mapping) {
         struct ac_shader_debug_info *debug_info = debug_info_mapping[inst->offset];
         if (debug_info) {
            /* Found the source location, print it */
            fprintf(f, "// 0x%x %s:%u:%u\n", debug_info->offset, debug_info->src_loc.file, debug_info->src_loc.line,
                    debug_info->src_loc.column);
         }
      }

      fprintf(f, "%s\n", inst->text);

      /* Print which waves execute the instruction right now. */
      while (num_waves && start_addr + inst->offset == waves->pc) {
         fprintf(f,
                 "          " COLOR_GREEN "^ SE%u SH%u CU%u "
                 "SIMD%u WAVE%u  EXEC=%016" PRIx64 "  ",
                 waves->se, waves->sh, waves->cu, waves->simd, waves->wave, waves->exec);

         if (inst->size == 4) {
            fprintf(f, "INST32=%08X" COLOR_RESET "\n", waves->inst_dw0);
         } else {
            fprintf(f, "INST64=%08X %08X" COLOR_RESET "\n", waves->inst_dw0, waves->inst_dw1);
         }

         waves->matched = true;
         waves = &waves[1];
         num_waves--;
      }
   }

   free(debug_info_mapping);
   free(instructions);
}