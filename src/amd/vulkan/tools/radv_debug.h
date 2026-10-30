/*
 * Copyright © 2017 Google.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_DEBUG_H
#define RADV_DEBUG_H

#include "ac_debug.h"
#include "radv_physical_device.h"
#include "radv_shader.h"

struct radv_device;

struct radv_backed_buffer {
   VkDeviceMemory memory;
   VkBuffer buffer;
   void *map;
};

enum radv_memory_type {
   radv_memory_type_invisible_vram,
   radv_memory_type_visible_vram,
   radv_memory_type_gtt,
};

struct radv_shader_inst {
   char text[160];  /* one disasm line */
   unsigned offset; /* instruction offset */
   unsigned size;   /* instruction size >= 4 */
};

VkResult radv_backed_buffer_init(struct radv_device *device, struct radv_backed_buffer *buffer, uint64_t size,
                                 enum radv_memory_type memory_type, VkBufferUsageFlags2 usage, bool map);

void radv_backed_buffer_finish(struct radv_device *device, struct radv_backed_buffer *buffer);

uint64_t radv_backed_buffer_get_va(struct radv_device *device, struct radv_backed_buffer *buffer);

/* Split a disassembly string into lines and add them to the array pointed
 * to by "instructions". */
void radv_add_split_disasm(const char *disasm, uint64_t start_addr, unsigned *num,
                           struct radv_shader_inst *instructions);

void radv_print_shader_disasm(const struct radv_shader_binary_debug_info *shader_debug_info, uint32_t code_size,
                              struct ac_wave_info *waves, unsigned num_waves, uint64_t start_addr, FILE *f);

#endif /* RADV_DEBUG_H */
