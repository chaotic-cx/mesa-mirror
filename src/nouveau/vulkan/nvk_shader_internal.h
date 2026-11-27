/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#ifndef NVK_SHADER_INTERNAL_H
#define NVK_SHADER_INTERNAL_H 1

#include "nvk_device.h"
#include "nvk_shader.h"

static inline struct nvk_shader *
atomic_set_or_destroy_shader(struct nvk_device *dev,
                             struct nvk_shader **shader_ptr,
                             struct nvk_shader *shader,
                             const VkAllocationCallbacks *alloc)
{
   struct nvk_shader *old_shader = p_atomic_cmpxchg(shader_ptr, NULL, shader);
   if (old_shader == NULL) {
      return shader;
   } else {
      vk_shader_destroy(&dev->vk, &shader->vk, alloc);
      return old_shader;
   }
}

typedef nir_shader* (*nvk_build_internal_shader_func)(void);

static inline VkResult
nvk_get_internal_shader(struct nvk_device *dev,
                        struct nvk_shader **shader_storage,
                        nvk_build_internal_shader_func build,
                        struct nvk_shader **shader_out)
{
   struct nvk_shader *shader = p_atomic_read(shader_storage);
   if (shader != NULL) {
      *shader_out = shader;
      return VK_SUCCESS;
   }

   nir_shader *nir = build();
   VkResult result = nvk_compile_nir_shader(dev, nir, &dev->vk.alloc, &shader);
   if (result != VK_SUCCESS)
      return result;

   *shader_out = atomic_set_or_destroy_shader(dev, shader_storage,
                                              shader, &dev->vk.alloc);

   return VK_SUCCESS;
}

#endif
