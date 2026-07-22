/*
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "util/blob.h"
#include "util/hash_table.h"
#include "util/u_debug.h"
#include "util/disk_cache.h"
#include "util/mesa-blake3.h"
#include "nir/nir_serialize.h"
#include "nir/nir.h"
#include "anv_private.h"
#include "anv_shader.h"
#include "nir/nir_xfb_info.h"
#include "vk_util.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/u_printf.h"


static bool
anv_shader_internal_serialize(struct vk_pipeline_cache_object *object,
                              struct blob *blob);

struct vk_pipeline_cache_object *
anv_shader_internal_deserialize(struct vk_pipeline_cache *cache,
                                const void *key_data, size_t key_size,
                                struct blob_reader *blob);

static void
anv_shader_internal_destroy(struct vk_device *_device,
                            struct vk_pipeline_cache_object *object)
{
   struct anv_device *device =
      container_of(_device, struct anv_device, vk);

   struct anv_shader_internal *shader =
      container_of(object, struct anv_shader_internal, base);

   anv_shader_heap_free(&device->shader_heap, shader->kernel);
   vk_pipeline_cache_object_finish(&shader->base);
   vk_free(&device->vk.alloc, shader);
}

static const struct vk_pipeline_cache_object_ops anv_shader_internal_ops = {
   .serialize = anv_shader_internal_serialize,
   .deserialize = anv_shader_internal_deserialize,
   .destroy = anv_shader_internal_destroy,
};

const struct vk_pipeline_cache_object_ops *const anv_cache_import_ops[2] = {
   &anv_shader_internal_ops,
   NULL
};

static struct anv_shader_internal *
anv_shader_internal_create(struct anv_device *device,
                           mesa_shader_stage stage,
                           const void *key_data, uint32_t key_size,
                           const intel_shader_bin *src_bin)
{
   assert(src_bin != NULL);
   assert(src_bin->info.stage == stage);
   assert(src_bin->num_stats <= ARRAY_SIZE(src_bin->stats));

   VK_MULTIALLOC(ma);
   VK_MULTIALLOC_DECL(&ma, struct anv_shader_internal, shader, 1);
   VK_MULTIALLOC_DECL_SIZE(&ma, void, obj_key_data, key_size);
   VK_MULTIALLOC_DECL(&ma, struct intel_shader_reloc, prog_data_relocs,
                           src_bin->num_relocs);
   VK_MULTIALLOC_DECL(&ma, void, code, src_bin->code_size);
   VK_MULTIALLOC_DECL(&ma, intel_shader_bin, bin, 1);

   if (!vk_multialloc_zalloc(&ma, &device->vk.alloc,
                             VK_SYSTEM_ALLOCATION_SCOPE_DEVICE))
      return NULL;

   memcpy(obj_key_data, key_data, key_size);
   vk_pipeline_cache_object_init(&device->vk, &shader->base,
                                 &anv_shader_internal_ops, obj_key_data, key_size);

   shader->stage = stage;

   shader->code = code;
   memcpy(shader->code, src_bin->code, src_bin->code_size);

   shader->bin = bin;
   *bin = *src_bin;
   bin->code = code;
   typed_memcpy(prog_data_relocs, src_bin->relocs, src_bin->num_relocs);
   bin->relocs = prog_data_relocs;

   if (anv_needs_printf_buffer()) {
      struct intel_shader_reloc_value reloc_values[3];
      uint32_t rv_count = 0;

      struct anv_bo *bo = device->printf.bo;
      assert(bo != NULL);

      reloc_values[rv_count++] = (struct intel_shader_reloc_value) {
         .id = BRW_SHADER_RELOC_PRINTF_BUFFER_ADDR_LOW,
         .value = bo->offset & 0xffffffff,
      };
      reloc_values[rv_count++] = (struct intel_shader_reloc_value) {
         .id = BRW_SHADER_RELOC_PRINTF_BUFFER_ADDR_HIGH,
         .value = bo->offset >> 32,
      };
      reloc_values[rv_count++] = (struct intel_shader_reloc_value) {
         .id = BRW_SHADER_RELOC_PRINTF_BUFFER_SIZE,
         .value = anv_printf_buffer_size(),
      };

      brw_write_shader_relocs(&device->physical->compiler->isa,
                              shader->code, bin,
                              reloc_values, rv_count);
   }

   shader->kernel = anv_shader_heap_alloc(&device->shader_heap,
                                          bin->code_size, 64, false, 0);
   if (shader->kernel.alloc_size == 0) {
      vk_pipeline_cache_object_finish(&shader->base);
      vk_free(&device->vk.alloc, shader);
      return NULL;
   }

   anv_shader_heap_upload(&device->shader_heap, shader->kernel,
                          shader->code, bin->code_size);

   return shader;
}

static bool
anv_shader_internal_serialize(struct vk_pipeline_cache_object *object,
                              struct blob *blob)
{
   struct anv_shader_internal *shader =
      container_of(object, struct anv_shader_internal, base);

   blob_write_uint32(blob, shader->stage);

   blob_write_bytes(blob, &shader->bin->info, sizeof(shader->bin->info));
   blob_write_uint32(blob, shader->bin->code_size);
   blob_write_bytes(blob, shader->bin->code, shader->bin->code_size);

   blob_write_uint32(blob, shader->bin->num_relocs);
   blob_write_bytes(blob, shader->bin->relocs,
                    shader->bin->num_relocs * sizeof(shader->bin->relocs[0]));

   blob_write_uint32(blob, shader->bin->num_stats);
   blob_write_bytes(blob, shader->bin->stats,
                    shader->bin->num_stats * sizeof(shader->bin->stats[0]));

   return !blob->out_of_memory;
}

struct vk_pipeline_cache_object *
anv_shader_internal_deserialize(struct vk_pipeline_cache *cache,
                                const void *key_data, size_t key_size,
                                struct blob_reader *blob)
{
   struct anv_device *device =
      container_of(cache->base.device, struct anv_device, vk);

   mesa_shader_stage stage = blob_read_uint32(blob);

   intel_shader_bin bin = {};
   blob_copy_bytes(blob, &bin.info, sizeof(bin.info));

   bin.code_size = blob_read_uint32(blob);
   bin.code = blob_read_bytes(blob, bin.code_size);

   bin.num_relocs = blob_read_uint32(blob);
   bin.relocs = (const struct intel_shader_reloc *)
      blob_read_bytes(blob, bin.num_relocs * sizeof(bin.relocs[0]));

   uint32_t num_stats = blob_read_uint32(blob);
   const struct genisa_stats *stats =
      blob_read_bytes(blob, num_stats * sizeof(stats[0]));

   if (bin.info.stage != stage ||
       num_stats > ARRAY_SIZE(bin.stats) || blob->overrun)
      return NULL;

   bin.num_stats = num_stats;
   typed_memcpy(bin.stats, stats, num_stats);

   struct anv_shader_internal *shader =
      anv_shader_internal_create(device, stage,
                                 key_data, key_size,
                                 &bin);

   if (shader == NULL)
      return NULL;

   return &shader->base;
}

struct anv_shader_internal *
anv_device_search_for_kernel(struct anv_device *device,
                             struct vk_pipeline_cache *cache,
                             const void *key_data, uint32_t key_size,
                             bool *user_cache_hit)
{
   /* Use the default pipeline cache if none is specified */
   if (cache == NULL)
      cache = device->vk.mem_cache;

   bool cache_hit = false;
   struct vk_pipeline_cache_object *object =
      vk_pipeline_cache_lookup_object(cache, key_data, key_size,
                                      &anv_shader_internal_ops, &cache_hit);
   if (user_cache_hit != NULL) {
      *user_cache_hit = object != NULL && cache_hit &&
                        cache != device->vk.mem_cache;
   }

   if (object == NULL)
      return NULL;

   return container_of(object, struct anv_shader_internal, base);
}

struct anv_shader_internal *
anv_device_upload_kernel(struct anv_device *device,
                         struct vk_pipeline_cache *cache,
                         const struct anv_shader_upload_params *params)
{
   /* Use the default pipeline cache if none is specified */
   if (cache == NULL)
      cache = device->vk.mem_cache;

   struct anv_shader_internal *shader =
      anv_shader_internal_create(device,
                                 params->stage,
                                 params->key_data,
                                 params->key_size,
                                 params->bin);
   if (shader == NULL)
      return NULL;

   struct vk_pipeline_cache_object *cached =
      vk_pipeline_cache_add_object(cache, &shader->base);

   return container_of(cached, struct anv_shader_internal, base);
}

struct nir_shader *
anv_device_search_for_nir(struct anv_device *device,
                          struct vk_pipeline_cache *cache,
                          const nir_shader_compiler_options *nir_options,
                          unsigned char blake3_key[BLAKE3_KEY_LEN],
                          void *mem_ctx)
{
   if (cache == NULL)
      cache = device->vk.mem_cache;

   return vk_pipeline_cache_lookup_nir(cache, blake3_key, BLAKE3_KEY_LEN,
                                       nir_options, NULL, mem_ctx);
}

void
anv_device_upload_nir(struct anv_device *device,
                      struct vk_pipeline_cache *cache,
                      const struct nir_shader *nir,
                      unsigned char blake3_key[BLAKE3_KEY_LEN])
{
   if (cache == NULL)
      cache = device->vk.mem_cache;

   vk_pipeline_cache_add_nir(cache, blake3_key, BLAKE3_KEY_LEN, nir);
}
