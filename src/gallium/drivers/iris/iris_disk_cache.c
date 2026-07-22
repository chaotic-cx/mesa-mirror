/*
 * Copyright © 2018 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * @file iris_disk_cache.c
 *
 * Functions for interacting with the on-disk shader cache.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "util/blob.h"
#include "util/build_id.h"
#include "util/disk_cache.h"
#include "util/mesa-blake3.h"
#include "intel/compiler/brw/brw_compiler.h"
#ifdef INTEL_USE_ELK
#include "intel/compiler/elk/elk_compiler.h"
#endif

#include "iris_context.h"

static bool debug = false;

/**
 * Compute a disk cache key for the given uncompiled shader and NOS key.
 */
static void
iris_disk_cache_compute_key(struct disk_cache *cache,
                            const struct iris_uncompiled_shader *ish,
                            const void *orig_prog_key,
                            uint32_t prog_key_size,
                            cache_key cache_key)
{
   /* Create a copy of the program key with program_string_id zeroed out.
    * It's essentially random data which we don't want to include in our
    * hashing and comparisons.  We'll set a proper value on a cache hit.
    */
   union iris_any_prog_key prog_key;
   memcpy(&prog_key, orig_prog_key, prog_key_size);
   prog_key.base.program_string_id = 0;

   uint8_t data[sizeof(prog_key) + sizeof(ish->nir_blake3)];
   uint32_t data_size = prog_key_size + sizeof(ish->nir_blake3);

   memcpy(data, ish->nir_blake3, sizeof(ish->nir_blake3));
   memcpy(data + sizeof(ish->nir_blake3), &prog_key, prog_key_size);

   disk_cache_compute_key(cache, data, data_size, cache_key);
}

#ifdef ENABLE_SHADER_CACHE
static void
iris_disk_cache_store_brw(struct disk_cache *cache,
                          const struct iris_uncompiled_shader *ish,
                          const struct iris_compiled_shader *shader,
                          const intel_shader_bin *bin,
                          const cache_key cache_key)
{
   mesa_shader_stage stage = ish->nir->info.stage;
   const struct brw_stage_prog_data *brw = shader->brw_prog_data;
   assert(bin && brw);

   struct blob blob;
   blob_init(&blob);

   /* We write the following data to the cache blob:
    *
    * 1. Prog data
    * 2. Shader info
    * 3. Assembly size
    * 4. Assembly code
    * 5. Number of entries in the system value array
    * 6. System value array
    * 7. Shader relocations
    * 8. UBO ranges
    * 9. Binding table
    */
   size_t prog_data_s = brw_prog_data_size(stage);
   blob_write_bytes(&blob, brw, prog_data_s);

   blob_write_bytes(&blob, &shader->info, sizeof(shader->info));
   blob_write_uint32(&blob, shader->program_size);
   blob_write_bytes(&blob, shader->map, shader->program_size);
   blob_write_uint32(&blob, shader->num_system_values);
   blob_write_bytes(&blob, shader->system_values,
                    shader->num_system_values * sizeof(uint32_t));
   blob_write_uint32(&blob, bin->num_relocs);
   blob_write_bytes(&blob, bin->relocs,
                    bin->num_relocs * sizeof(bin->relocs[0]));
   blob_write_bytes(&blob, shader->ubo_ranges, sizeof(shader->ubo_ranges));
   blob_write_bytes(&blob, &shader->bt, sizeof(shader->bt));

   disk_cache_put(cache, cache_key, blob.data, blob.size, NULL);
   blob_finish(&blob);
}

#ifdef INTEL_USE_ELK
static void
iris_disk_cache_store_elk(struct disk_cache *cache,
                          const struct iris_uncompiled_shader *ish,
                          const struct iris_compiled_shader *shader,
                          const cache_key cache_key)
{
   mesa_shader_stage stage = ish->nir->info.stage;
   const struct elk_stage_prog_data *elk = shader->elk_prog_data;
   assert(elk);

   struct blob blob;
   blob_init(&blob);

   /* We write the following data to the cache blob:
    *
    * 1. Prog data (must come first because it has the assembly size)
    *   - Zero out pointer values in prog data, so cache entries will be
    *     consistent.
    * 2. Assembly code
    * 3. Number of entries in the system value array
    * 4. System value array
    * 5. Shader relocations
    * 6. Legacy param array (only used for compute workgroup ID)
    * 7. Binding table
    */
   size_t prog_data_s = elk_prog_data_size(stage);
   union elk_any_prog_data serializable;
   assert(prog_data_s <= sizeof(serializable));
   memcpy(&serializable, shader->elk_prog_data, prog_data_s);
   serializable.base.param = NULL;
   serializable.base.relocs = NULL;
   blob_write_bytes(&blob, &serializable, prog_data_s);

   blob_write_bytes(&blob, shader->map, shader->program_size);
   blob_write_uint32(&blob, shader->num_system_values);
   blob_write_bytes(&blob, shader->system_values,
                    shader->num_system_values * sizeof(uint32_t));
   blob_write_bytes(&blob, elk->relocs,
                    elk->num_relocs * sizeof(struct intel_shader_reloc));
   blob_write_bytes(&blob, elk->param,
                    elk->nr_params * sizeof(uint32_t));
   blob_write_bytes(&blob, &shader->bt, sizeof(shader->bt));

   disk_cache_put(cache, cache_key, blob.data, blob.size, NULL);
   blob_finish(&blob);
}
#endif
#endif

/**
 * Store the given compiled shader in the disk cache.
 *
 * This should only be called on newly compiled shaders.  No checking is
 * done to prevent repeated stores of the same shader.
 */
void
iris_disk_cache_store(struct disk_cache *cache,
                      const struct iris_uncompiled_shader *ish,
                      const struct iris_compiled_shader *shader,
                      const intel_shader_bin *bin,
                      const void *prog_key,
                      uint32_t prog_key_size)
{
#ifdef ENABLE_SHADER_CACHE
   if (!cache)
      return;

   cache_key cache_key;
   iris_disk_cache_compute_key(cache, ish, prog_key, prog_key_size, cache_key);

   if (debug) {
      char blake3[BLAKE3_HEX_LEN];
      _mesa_blake3_format(blake3, cache_key);
      fprintf(stderr, "[mesa disk cache] storing %s\n", blake3);
   }

#ifdef INTEL_USE_ELK
   assert((shader->brw_prog_data == NULL) != (shader->elk_prog_data == NULL));
   if (shader->elk_prog_data)
      iris_disk_cache_store_elk(cache, ish, shader, cache_key);
   else
      iris_disk_cache_store_brw(cache, ish, shader, bin, cache_key);
#else
   assert(shader->brw_prog_data);
   iris_disk_cache_store_brw(cache, ish, shader, bin, cache_key);
#endif
#endif
}

static const enum iris_program_cache_id cache_id_for_stage[] = {
   [MESA_SHADER_VERTEX]    = IRIS_CACHE_VS,
   [MESA_SHADER_TESS_CTRL] = IRIS_CACHE_TCS,
   [MESA_SHADER_TESS_EVAL] = IRIS_CACHE_TES,
   [MESA_SHADER_GEOMETRY]  = IRIS_CACHE_GS,
   [MESA_SHADER_FRAGMENT]  = IRIS_CACHE_FS,
   [MESA_SHADER_COMPUTE]   = IRIS_CACHE_CS,
};

#ifdef ENABLE_SHADER_CACHE
static bool
iris_disk_cache_retrieve_brw(struct iris_screen *screen,
                             struct u_upload_mgr *uploader,
                             struct iris_uncompiled_shader *ish,
                             struct iris_compiled_shader *shader,
                             void *buffer,
                             size_t size,
                             const void *prog_key,
                             uint32_t key_size)
{
   assert(screen->brw);

   mesa_shader_stage stage = ish->nir->info.stage;
   const uint32_t prog_data_size = brw_prog_data_size(stage);
   struct brw_stage_prog_data *brw = ralloc_size(NULL, prog_data_size);
   intel_shader_bin bin = {};
   const void *assembly;
   uint32_t num_system_values;
   uint32_t *system_values = NULL;
   uint32_t *so_decls = NULL;

   struct blob_reader blob;
   blob_reader_init(&blob, buffer, size);
   blob_copy_bytes(&blob, brw, prog_data_size);

   blob_copy_bytes(&blob, &bin.info, sizeof(bin.info));
   bin.code_size = blob_read_uint32(&blob);
   assembly = blob_read_bytes(&blob, bin.code_size);
   bin.code = assembly;
   num_system_values = blob_read_uint32(&blob);
   if (num_system_values) {
      system_values =
         ralloc_array(NULL, uint32_t, num_system_values);
      blob_copy_bytes(&blob, system_values,
                      num_system_values * sizeof(uint32_t));
   }

   bin.num_relocs = blob_read_uint32(&blob);
   bin.relocs = bin.num_relocs ?
      (const struct intel_shader_reloc *)
      blob_read_bytes(&blob, bin.num_relocs * sizeof(bin.relocs[0])) :
      NULL;

   bin.prog_data = brw;

   blob_copy_bytes(&blob, shader->ubo_ranges, sizeof(shader->ubo_ranges));

   struct iris_binding_table bt;
   blob_copy_bytes(&blob, &bt, sizeof(bt));

   if (stage == MESA_SHADER_VERTEX ||
       stage == MESA_SHADER_TESS_EVAL ||
       stage == MESA_SHADER_GEOMETRY) {
      struct intel_vue_map *vue_map = &brw_vue_prog_data(brw)->vue_map;
      so_decls = screen->vtbl.create_so_decl_list(&ish->stream_output, vue_map);
   }

   /* System values and uniforms are stored in constant buffer 0, the
    * user-facing UBOs are indexed by one.  So if any constant buffer is
    * needed, the constant buffer 0 will be needed, so account for it.
    */
   unsigned num_cbufs = ish->nir->info.num_ubos;

   if (num_cbufs || ish->nir->num_uniforms)
      num_cbufs++;

   if (num_system_values)
      num_cbufs++;

   iris_apply_brw_prog_data(shader, &bin, NULL);

   iris_finalize_program(shader, so_decls, system_values,
                         num_system_values, num_cbufs,
                         &bt);

   assert(stage < ARRAY_SIZE(cache_id_for_stage));
   enum iris_program_cache_id cache_id = cache_id_for_stage[stage];

   /* Upload our newly read shader to the in-memory program cache. */
   iris_upload_shader(screen, ish, shader, NULL, uploader,
                      cache_id, key_size, prog_key, assembly, &bin);

   free(buffer);

   return true;
}

#ifdef INTEL_USE_ELK
static bool
iris_disk_cache_retrieve_elk(struct iris_screen *screen,
                             struct u_upload_mgr *uploader,
                             struct iris_uncompiled_shader *ish,
                             struct iris_compiled_shader *shader,
                             void *buffer,
                             size_t size,
                             const void *prog_key,
                             uint32_t key_size)
{
   assert(screen->elk);

   mesa_shader_stage stage = ish->nir->info.stage;
   const uint32_t prog_data_size = elk_prog_data_size(stage);
   struct elk_stage_prog_data *elk = ralloc_size(NULL, prog_data_size);
   const void *assembly;
   uint32_t num_system_values;
   uint32_t *system_values = NULL;
   uint32_t *so_decls = NULL;

   struct blob_reader blob;
   blob_reader_init(&blob, buffer, size);
   blob_copy_bytes(&blob, elk, prog_data_size);

   assembly = blob_read_bytes(&blob, elk->program_size);
   num_system_values = blob_read_uint32(&blob);
   if (num_system_values) {
      system_values =
         ralloc_array(NULL, uint32_t, num_system_values);
      blob_copy_bytes(&blob, system_values,
                      num_system_values * sizeof(uint32_t));
   }

   elk->relocs = NULL;
   if (elk->num_relocs) {
      struct intel_shader_reloc *relocs =
         ralloc_array(NULL, struct intel_shader_reloc, elk->num_relocs);
      blob_copy_bytes(&blob, relocs,
                      elk->num_relocs * sizeof(struct intel_shader_reloc));
      elk->relocs = relocs;
   }

   elk->param = NULL;
   if (elk->nr_params) {
      elk->param = ralloc_array(NULL, uint32_t, elk->nr_params);
      blob_copy_bytes(&blob, elk->param,
                      elk->nr_params * sizeof(uint32_t));
   }

   struct iris_binding_table bt;
   blob_copy_bytes(&blob, &bt, sizeof(bt));

   if (stage == MESA_SHADER_VERTEX ||
       stage == MESA_SHADER_TESS_EVAL ||
       stage == MESA_SHADER_GEOMETRY) {
      struct intel_vue_map *vue_map = &elk_vue_prog_data(elk)->vue_map;
      so_decls = screen->vtbl.create_so_decl_list(&ish->stream_output, vue_map);
   }

   /* System values and uniforms are stored in constant buffer 0, the
    * user-facing UBOs are indexed by one.  So if any constant buffer is
    * needed, the constant buffer 0 will be needed, so account for it.
    */
   unsigned num_cbufs = ish->nir->info.num_ubos;

   if (num_cbufs || ish->nir->num_uniforms)
      num_cbufs++;

   if (num_system_values)
      num_cbufs++;

   iris_apply_elk_prog_data(shader, elk);

   iris_finalize_program(shader, so_decls, system_values,
                         num_system_values, num_cbufs,
                         &bt);

   assert(stage < ARRAY_SIZE(cache_id_for_stage));
   enum iris_program_cache_id cache_id = cache_id_for_stage[stage];

   /* Upload our newly read shader to the in-memory program cache. */
   iris_upload_shader(screen, ish, shader, NULL, uploader,
                      cache_id, key_size, prog_key, assembly, NULL);

   free(buffer);

   return true;
}
#endif
#endif

/**
 * Search for a compiled shader in the disk cache.  If found, upload it
 * to the in-memory program cache so we can use it.
 */
bool
iris_disk_cache_retrieve(struct iris_screen *screen,
                         struct u_upload_mgr *uploader,
                         struct iris_uncompiled_shader *ish,
                         struct iris_compiled_shader *shader,
                         const void *prog_key,
                         uint32_t key_size)
{
#ifdef ENABLE_SHADER_CACHE
   struct disk_cache *cache = screen->disk_cache;

   if (!cache)
      return false;

   cache_key cache_key;
   iris_disk_cache_compute_key(cache, ish, prog_key, key_size, cache_key);

   if (debug) {
      char blake3[BLAKE3_HEX_LEN];
      _mesa_blake3_format(blake3, cache_key);
      fprintf(stderr, "[mesa disk cache] retrieving %s: ", blake3);
   }

   size_t size;
   void *buffer = disk_cache_get(screen->disk_cache, cache_key, &size);

   if (debug)
      fprintf(stderr, "%s\n", buffer ? "found" : "missing");

   if (!buffer)
      return false;

#ifdef INTEL_USE_ELK
   assert((screen->brw == NULL) != (screen->elk == NULL));
   if (screen->brw) {
      return iris_disk_cache_retrieve_brw(screen, uploader, ish, shader,
                                          buffer, size, prog_key, key_size);
   } else {
      return iris_disk_cache_retrieve_elk(screen, uploader, ish, shader,
                                          buffer, size, prog_key, key_size);
   }
#else
   assert(screen->brw);
   return iris_disk_cache_retrieve_brw(screen, uploader, ish, shader,
                                       buffer, size, prog_key, key_size);
#endif
#else
   return false;
#endif
}

/**
 * Initialize the on-disk shader cache.
 */
void
iris_disk_cache_init(struct iris_screen *screen)
{
#ifdef ENABLE_SHADER_CACHE
   if (INTEL_DEBUG(DEBUG_DISK_CACHE_DISABLE_MASK))
      return;

   /* array length = strlen("iris_") + blake3 + nul char */
   char renderer[5 + 40 + 1] = {0};

   if (screen->brw) {
      char device_info_sha[BLAKE3_HEX_LEN];
      brw_device_blake3(device_info_sha, screen->devinfo);
      memcpy(renderer, "iris_", 5);
      memcpy(renderer + 5, device_info_sha, 40);
   } else {
      /* For Gfx8, just use PCI ID. */
      ASSERTED int len = snprintf(renderer, sizeof(renderer),
                                  "iris_%04x", screen->devinfo->pci_device_id);
      assert(len < ARRAY_SIZE(renderer) - 1);
   }

   const struct build_id_note *note =
      build_id_find_nhdr_for_addr(iris_disk_cache_init);
   assert(note && build_id_length(note) == BUILD_ID_EXPECTED_HASH_LENGTH); /* sha1 */

   const uint8_t *id_sha1 = build_id_data(note);
   assert(id_sha1);

   char timestamp[BLAKE3_HEX_LEN];
   _mesa_sha1_format(timestamp, id_sha1);

   const uint64_t driver_flags =
#ifdef INTEL_USE_ELK
      screen->elk ? elk_get_compiler_config_value(screen->elk) :
#endif
      brw_get_compiler_config_value(screen->brw);

   screen->disk_cache = disk_cache_create(renderer, timestamp, driver_flags);
#endif
}
