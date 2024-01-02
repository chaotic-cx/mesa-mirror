/*
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vk_util_compiler.h"
#include "compiler/spirv/nir_spirv.h"

struct nir_spirv_specialization*
vk_spec_info_to_nir_spirv(const VkSpecializationInfo *spec_info,
                          uint32_t *out_num_spec_entries)
{
   if (spec_info == NULL || spec_info->mapEntryCount == 0)
      return NULL;

   uint32_t num_spec_entries = spec_info->mapEntryCount;
   struct nir_spirv_specialization *spec_entries =
      calloc(num_spec_entries, sizeof(*spec_entries));

   for (uint32_t i = 0; i < num_spec_entries; i++) {
      VkSpecializationMapEntry entry = spec_info->pMapEntries[i];
      const void *data = (uint8_t *)spec_info->pData + entry.offset;
      assert((uint8_t *)data + entry.size <=
             (uint8_t *)spec_info->pData + spec_info->dataSize);

      spec_entries[i].id = spec_info->pMapEntries[i].constantID;
      switch (entry.size) {
      case 8:
         spec_entries[i].value.u64 = *(const uint64_t *)data;
         break;
      case 4:
         spec_entries[i].value.u32 = *(const uint32_t *)data;
         break;
      case 2:
         spec_entries[i].value.u16 = *(const uint16_t *)data;
         break;
      case 1:
         spec_entries[i].value.u8 = *(const uint8_t *)data;
         break;
      case 0:
      default:
         /* The Vulkan spec says:
          *
          *    "For a constantID specialization constant declared in a
          *    shader, size must match the byte size of the constantID. If
          *    the specialization constant is of type boolean, size must be
          *    the byte size of VkBool32."
          *
          * Therefore, since only scalars can be decorated as
          * specialization constants, we can assume that if it doesn't have
          * a size of 1, 2, 4, or 8, any use in a shader would be invalid
          * usage.  The spec further says:
          *
          *    "If a constantID value is not a specialization constant ID
          *    used in the shader, that map entry does not affect the
          *    behavior of the pipeline."
          *
          * so we should ignore any invalid specialization constants rather
          * than crash or error out when we see one.
          */
         break;
      }
   }

   *out_num_spec_entries = num_spec_entries;
   return spec_entries;
}
