/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 *
 * RDF RGP Trace Writer implementation.
 */

#include "ac_rdf_rgp_writer.h"
#include "ac_rdf_rgp_chunks.h"

#include "ac_gpu_info.h"
#include "ac_rdf.h"
#include "ac_rgp.h"
#include "ac_shader_util.h"
#include "ac_spm.h"
#include "ac_sqtt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

/* Map VRAM types to RDF memory types */
static enum rdf_memory_type
vram_type_to_rdf_memory_type(uint32_t vram_type)
{
   switch (vram_type) {
   case AMD_VRAM_TYPE_UNKNOWN:
      return RDF_MEMORY_TYPE_UNKNOWN;
   case AMD_VRAM_TYPE_DDR2:
      return RDF_MEMORY_TYPE_DDR2;
   case AMD_VRAM_TYPE_DDR3:
      return RDF_MEMORY_TYPE_DDR3;
   case AMD_VRAM_TYPE_DDR4:
      return RDF_MEMORY_TYPE_DDR4;
   case AMD_VRAM_TYPE_DDR5:
      return RDF_MEMORY_TYPE_DDR5;
   case AMD_VRAM_TYPE_GDDR3:
      return RDF_MEMORY_TYPE_GDDR3;
   case AMD_VRAM_TYPE_GDDR4:
      return RDF_MEMORY_TYPE_GDDR4;
   case AMD_VRAM_TYPE_GDDR5:
      return RDF_MEMORY_TYPE_GDDR5;
   case AMD_VRAM_TYPE_GDDR6:
      return RDF_MEMORY_TYPE_GDDR6;
   case AMD_VRAM_TYPE_HBM:
      return RDF_MEMORY_TYPE_HBM;
   case AMD_VRAM_TYPE_LPDDR4:
      return RDF_MEMORY_TYPE_LPDDR4;
   case AMD_VRAM_TYPE_LPDDR5:
      return RDF_MEMORY_TYPE_LPDDR5;
   default:
      return RDF_MEMORY_TYPE_UNKNOWN;
   }
}

/* Fill AsicInfo matching ac_sqtt_fill_asic_info from ac_rgp.c */
static void
rdf_fill_asic_info(const struct radeon_info *info, uint32_t trace_shader_core_clock,
                   uint32_t trace_memory_clock, struct rdf_asic_info *rdf_asic_info)
{
   bool has_wave32 = info->gfx_level >= GFX10;

   memset(rdf_asic_info, 0, sizeof(*rdf_asic_info));

   /* Clocks sampled at trace time - matching ac_rgp.c */
   rdf_asic_info->shader_core_clock_frequency = trace_shader_core_clock * 1000000ull;
   rdf_asic_info->memory_clock_frequency = trace_memory_clock * 1000000ull;

   /* Workaround for chips reporting 0 clocks (from ac_rgp.c) */
   if (!rdf_asic_info->shader_core_clock_frequency)
      rdf_asic_info->shader_core_clock_frequency = 1300000000;
   if (!rdf_asic_info->memory_clock_frequency)
      rdf_asic_info->memory_clock_frequency = 687000000;

   rdf_asic_info->max_shader_core_clock = info->max_gpu_freq_mhz * 1000000ull;
   rdf_asic_info->max_memory_clock = info->memory_freq_mhz * 1000000ull;
   rdf_asic_info->gpu_timestamp_frequency = info->clock_crystal_freq * 1000;

   /* Device identification */
   rdf_asic_info->pci_id = info->pci_id;
   rdf_asic_info->device_id = info->pci_id;
   rdf_asic_info->device_revision_id = info->pci_rev_id;

   /* Shader engine configuration - from radeon_info.compiler_info */
   rdf_asic_info->vgprs_per_simd =
      info->compiler_info.num_physical_wave64_vgprs_per_simd * (has_wave32 ? 2 : 1);
   rdf_asic_info->sgprs_per_simd = info->compiler_info.num_physical_sgprs_per_simd;
   rdf_asic_info->shader_engines = info->max_se;
   rdf_asic_info->compute_unit_per_shader_engine = info->min_good_cu_per_sa * info->max_sa_per_se;
   rdf_asic_info->simd_per_compute_unit = info->compiler_info.num_simd_per_compute_unit;
   rdf_asic_info->wavefronts_per_simd = info->compiler_info.max_waves_per_simd;

   /* VGPR/SGPR allocation - from radeon_info.compiler_info */
   rdf_asic_info->minimum_vgpr_alloc = info->compiler_info.min_wave64_vgpr_alloc;
   rdf_asic_info->vgpr_alloc_granularity =
      info->compiler_info.wave64_vgpr_alloc_granularity * (has_wave32 ? 2 : 1);
   rdf_asic_info->minimum_sgpr_alloc = info->compiler_info.min_sgpr_alloc;
   rdf_asic_info->sgpr_alloc_granularity = info->compiler_info.sgpr_alloc_granularity;

   rdf_asic_info->hardware_contexts = 8;

   /* GPU type */
   rdf_asic_info->gpu_type = info->has_dedicated_vram ? RDF_GPU_TYPE_DISCRETE
                                                      : RDF_GPU_TYPE_INTEGRATED;

   /* GFX IP level, as reported by the kernel */
   rdf_asic_info->gfx_ip_level.major = info->ip[AMD_IP_GFX].ver_major;
   rdf_asic_info->gfx_ip_level.minor = info->ip[AMD_IP_GFX].ver_minor;
   rdf_asic_info->gfx_ip_level.stepping = info->ip[AMD_IP_GFX].ver_rev;

   rdf_asic_info->gpu_index = 0;

   /* CE RAM - not exposed, set to 0 like ac_rgp.c */
   rdf_asic_info->ce_ram_size = 0;
   rdf_asic_info->ce_ram_size_graphics = 0;
   rdf_asic_info->ce_ram_size_compute = 0;
   rdf_asic_info->max_number_of_dedicated_cus = 0;

   /* Memory configuration */
   rdf_asic_info->vram_size = (int64_t)info->vram_size_kb * 1024;
   rdf_asic_info->vram_bus_width = info->memory_bus_width;
   rdf_asic_info->l2_cache_size = info->l2_cache_size;
   rdf_asic_info->l1_cache_size = info->tcp_cache_size;
   rdf_asic_info->lds_size = info->compiler_info.lds_size_per_workgroup;

   /* GPU name - use family name like ac_rgp.c */
   strncpy(rdf_asic_info->gpu_name, ac_get_family_name(info->family), RDF_GPU_NAME_MAX_SIZE - 1);
   rdf_asic_info->gpu_name[RDF_GPU_NAME_MAX_SIZE - 1] = '\0';

   /* Performance characteristics - matching ac_rgp.c */
   rdf_asic_info->alu_per_clock = 0.0f;
   rdf_asic_info->texture_per_clock = 0.0f;
   rdf_asic_info->prims_per_clock = (float)info->max_se;
   if (info->gfx_level == GFX10)
      rdf_asic_info->prims_per_clock *= 2;
   rdf_asic_info->pixels_per_clock = 0.0f;

   /* Memory configuration */
   rdf_asic_info->memory_ops_per_clock = ac_memory_ops_per_clock(info->vram_type);
   rdf_asic_info->memory_chip_type = vram_type_to_rdf_memory_type(info->vram_type);
   rdf_asic_info->lds_granularity = ac_shader_get_lds_alloc_granularity(info->gfx_level);

   /* CU mask */
   for (unsigned se = 0; se < AMD_MAX_SE && se < RDF_MAX_NUM_SE; se++) {
      for (unsigned sa = 0; sa < AMD_MAX_SA_PER_SE && sa < RDF_SA_PER_SE; sa++) {
         rdf_asic_info->cu_mask[se][sa] = info->cu_mask[se][sa];
      }
   }

   /* Additional cache info */
   rdf_asic_info->gl1_cache_size = info->l1_cache_size;
   rdf_asic_info->inst_cache_size = info->sqc_inst_cache_size;
   rdf_asic_info->scalar_cache_size = info->sqc_scalar_cache_size;
   rdf_asic_info->mall_cache_size = info->l3_cache_size_mb * 1024 * 1024;
}

/* Write AsicInfo chunk. */
static bool
write_asic_info_chunk(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                      uint32_t trace_shader_core_clock, uint32_t trace_memory_clock,
                      bool use_compression)
{
   struct rdf_asic_info rdf_asic_info;
   rdf_fill_asic_info(info, trace_shader_core_clock, trace_memory_clock, &rdf_asic_info);

   struct ac_rdf_chunk_info chunk = {
      .identifier = RDF_CHUNK_ID_ASIC_INFO,
      .version = RDF_ASIC_INFO_VERSION,
      .header = NULL,
      .header_size = 0,
      .data = &rdf_asic_info,
      .data_size = sizeof(rdf_asic_info),
      .use_compression = use_compression,
   };

   return ac_rdf_chunk_write(rdf_writer, &chunk);
}

/* Write ApiInfo chunk. */
static bool
write_api_info_chunk(struct ac_rdf_writer *rdf_writer, bool use_compression)
{
   struct rdf_api_info rdf_api_info = {
      .api_type = RDF_API_TYPE_VULKAN,
      .api_version_major = 1,
      .api_version_minor = 3,
   };

   struct ac_rdf_chunk_info chunk = {
      .identifier = RDF_CHUNK_ID_API_INFO,
      .version = RDF_API_INFO_VERSION,
      .header = NULL,
      .header_size = 0,
      .data = &rdf_api_info,
      .data_size = sizeof(rdf_api_info),
      .use_compression = use_compression,
   };

   return ac_rdf_chunk_write(rdf_writer, &chunk);
}

/* Write TraceConfig chunk (JSON format required by RGP). */
static bool
write_trace_config_chunk(struct ac_rdf_writer *rdf_writer, const char *config_json,
                         bool use_compression)
{
   /* TraceConfig is a JSON string describing the trace configuration.
    * RDP provides this config via ConfigureTraceParams. RGP requires this chunk to be present. */

   if (!config_json)
      return true;

   const char *json_data = config_json;
   size_t json_size = strlen(config_json);

   struct ac_rdf_chunk_info chunk = {
      .identifier = RDF_CHUNK_ID_TRACE_CONFIG,
      .version = RDF_TRACE_CONFIG_VERSION,
      .header = NULL,
      .header_size = 0,
      .data = json_data,
      .data_size = json_size,
      .use_compression = use_compression,
   };

   return ac_rdf_chunk_write(rdf_writer, &chunk);
}

/* Write ClockCalibration chunks. */
static bool
write_clock_calibration_chunks(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                               const struct rgp_clock_calibration *clock_calibration,
                               bool use_compression)
{
   if (!clock_calibration || clock_calibration->record_count == 0)
      return true;

   list_for_each_entry (struct rgp_clock_calibration_record, record, &clock_calibration->record, list) {
      struct rdf_clock_calibration rdf_clock_calibration = {
         .pci_id = info->pci_id,
         .cpu_timestamp = record->cpu_timestamp,
         .gpu_timestamp = record->gpu_timestamp,
      };

      struct ac_rdf_chunk_info chunk = {
         .identifier = RDF_CHUNK_ID_CLOCK_CALIB,
         .version = RDF_CLOCK_CALIB_VERSION,
         .header = NULL,
         .header_size = 0,
         .data = &rdf_clock_calibration,
         .data_size = sizeof(rdf_clock_calibration),
         .use_compression = use_compression,
      };

      if (!ac_rdf_chunk_write(rdf_writer, &chunk))
         return false;
   }
   return true;
}

/* Write SqttData chunks. */
static bool
write_sqtt_chunks(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                  const struct ac_sqtt_trace *sqtt_trace, bool instruction_timing_enabled,
                  uint32_t buffer_size, bool use_compression)
{
   uint32_t sqtt_version = ac_gfx_level_to_sqtt_version(info->gfx_level);

   for (unsigned i = 0; i < sqtt_trace->num_traces; i++) {
      const struct ac_sqtt_data_se *se_data = &sqtt_trace->traces[i];

      /* Data size is in units of 32 bytes */
      size_t data_size = se_data->info.cur_offset * 32;

      if (data_size == 0 || !se_data->data_ptr)
         continue;

      /* Fill SqttData header */
      struct rdf_sqtt_data_header rdf_header = {
         .pci_id = info->pci_id,
         .shader_engine = se_data->shader_engine,
         .sqtt_version = sqtt_version,
         .instrumentation_version_spec = 1,
         .instrumentation_version_api = 5,
         .wgp_index = se_data->compute_unit,
         .trace_buffer_size = buffer_size,
         .instruction_timing_enabled =
            instruction_timing_enabled &&
            (sqtt_trace->instruction_timing_se_mask & (1u << se_data->shader_engine)),
         .exec_pop_tokens_enabled = 0,
      };

      /* Write SqttData chunk with header */
      struct ac_rdf_chunk_info data_chunk = {
         .identifier = RDF_CHUNK_ID_SQTT_DATA,
         .version = RDF_SQTT_DATA_VERSION,
         .header = &rdf_header,
         .header_size = sizeof(rdf_header),
         .data = se_data->data_ptr,
         .data_size = data_size,
         .use_compression = use_compression,
      };

      if (!ac_rdf_chunk_write(rdf_writer, &data_chunk))
         return false;
   }
   return true;
}

/* Write PsoCorrelation chunk. */
static bool
write_pso_correlation_chunk(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                            const struct rgp_pso_correlation *pso_correlation)
{
   if (!pso_correlation || pso_correlation->record_count == 0)
      return true;

   /* Allocate records array */
   size_t rdf_records_size =
      pso_correlation->record_count * sizeof(struct rdf_pso_correlation_record);
   struct rdf_pso_correlation_record *rdf_records = (struct rdf_pso_correlation_record *)calloc(
      pso_correlation->record_count, sizeof(struct rdf_pso_correlation_record));
   if (!rdf_records)
      return false;

   /* Fill records */
   uint32_t i = 0;
   list_for_each_entry (struct rgp_pso_correlation_record, record, &pso_correlation->record, list) {
      rdf_records[i].pci_id = info->pci_id;
      rdf_records[i].api_pso_hash = record->api_pso_hash;
      rdf_records[i].pipeline_hash[0] = record->pipeline_hash[0];
      rdf_records[i].pipeline_hash[1] = record->pipeline_hash[1];
      memcpy(rdf_records[i].api_level_obj_name, record->api_level_obj_name, 64);
      i++;
   }

   struct rdf_pso_correlation_header rdf_header = {
      .count = pso_correlation->record_count,
   };

   struct ac_rdf_chunk_info chunk = {
      .identifier = RDF_CHUNK_ID_PSO_CORRELATION,
      .version = RDF_PSO_CORRELATION_VERSION,
      .header = &rdf_header,
      .header_size = sizeof(rdf_header),
      .data = rdf_records,
      .data_size = rdf_records_size,
      .use_compression = false, /* Compression hardcoded to false. It always must be disabled for
                                 * PsoCorrelation (known compression issues). */
   };

   bool ret = ac_rdf_chunk_write(rdf_writer, &chunk);
   free(rdf_records);

   return ret;
}

/* Write CodeObjectLoaderEvents chunk. */
static bool
write_loader_events_chunk(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                          const struct rgp_loader_events *loader_events)
{
   if (!loader_events || loader_events->record_count == 0)
      return true;

   /* Allocate records array */
   size_t rdf_records_size = loader_events->record_count * sizeof(struct rdf_loader_events_record);

   struct rdf_loader_events_record *rdf_records = (struct rdf_loader_events_record *)calloc(
      loader_events->record_count, sizeof(struct rdf_loader_events_record));
   if (!rdf_records)
      return false;

   /* Fill records */
   uint32_t i = 0;
   list_for_each_entry (struct rgp_loader_events_record, record, &loader_events->record, list) {
      rdf_records[i].pci_id = info->pci_id;
      rdf_records[i].loader_event_type = record->loader_event_type;
      rdf_records[i].base_address = record->base_address;
      rdf_records[i].code_object_hash[0] = record->code_object_hash[0];
      rdf_records[i].code_object_hash[1] = record->code_object_hash[1];
      rdf_records[i].time_stamp = record->time_stamp;
      i++;
   }

   struct rdf_loader_events_header rdf_header = {
      .count = loader_events->record_count,
   };

   struct ac_rdf_chunk_info chunk = {
      .identifier = RDF_CHUNK_ID_CO_LOAD_EVENT,
      .version = RDF_CO_LOAD_EVENT_VERSION,
      .header = &rdf_header,
      .header_size = sizeof(rdf_header),
      .data = rdf_records,
      .data_size = rdf_records_size,
      .use_compression = false, /* Compression hardcoded to false. It always must be disabled for
                                 * CoLoadEvent (known compression issues). */
   };

   bool ret = ac_rdf_chunk_write(rdf_writer, &chunk);
   free(rdf_records);

   return ret;
}

/* Write QueueInfo chunk. */
static bool
write_queue_info_chunks(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                        const struct rgp_queue_info *queue_info, bool use_compression)
{
   if (!queue_info || queue_info->record_count == 0)
      return true;

   list_for_each_entry (struct rgp_queue_info_record, record, &queue_info->record, list) {
      struct rdf_queue_info_record rdf_record = {
         .pci_id = info->pci_id,
         .queue_id = record->queue_id,
         .queue_context = record->queue_context,
         .queue_type = record->hardware_info.queue_type,
         .engine_type = record->hardware_info.engine_type,
      };

      struct ac_rdf_chunk_info chunk = {
         .identifier = RDF_CHUNK_ID_QUEUE_INFO,
         .version = RDF_QUEUE_INFO_VERSION,
         .header = NULL,
         .header_size = 0,
         .data = &rdf_record,
         .data_size = sizeof(rdf_record),
         .use_compression = use_compression,
      };

      if (!ac_rdf_chunk_write(rdf_writer, &chunk))
         return false;
   }
   return true;
}

/* Helper: Resolve queue_info_index to queue_id by walking the queue_info list. */
static uint64_t
resolve_queue_id(const struct rgp_queue_info *queue_info, uint32_t index)
{
   if (!queue_info)
      return 0;

   uint32_t i = 0;
   list_for_each_entry (struct rgp_queue_info_record, record, &queue_info->record, list) {
      if (i == index)
         return record->queue_id;
      i++;
   }
   return 0;
}

/* Write QueueEvent chunk. */
static bool
write_queue_event_chunks(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                         const struct rgp_queue_event *queue_event,
                         const struct rgp_queue_info *queue_info, bool use_compression)
{
   if (!queue_event || queue_event->record_count == 0)
      return true;

   list_for_each_entry (struct rgp_queue_event_record, record, &queue_event->record, list) {
      struct rdf_queue_event_record rdf_record = {
         .pci_id = info->pci_id,
         .queue_id = resolve_queue_id(queue_info, record->queue_info_index),
         .event_type = record->event_type,
         .sqtt_cb_id = record->sqtt_cb_id,
         .frame_index = record->frame_index,
         .submit_sub_index = record->submit_sub_index,
         .api_event_id = record->api_id,
         .cpu_timestamp = record->cpu_timestamp,
      };

      /* Dereference GPU timestamp pointers */
      if (record->event_type == SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT) {
         if (record->gpu_timestamps[0])
            rdf_record.gpu_timestamp_1 = *record->gpu_timestamps[0];
         if (record->gpu_timestamps[1])
            rdf_record.gpu_timestamp_2 = *record->gpu_timestamps[1];
      } else if (record->event_type == SQTT_QUEUE_TIMING_EVENT_PRESENT) {
         if (record->gpu_timestamps[0])
            rdf_record.gpu_timestamp_1 = *record->gpu_timestamps[0];
      }

      struct ac_rdf_chunk_info chunk = {
         .identifier = RDF_CHUNK_ID_QUEUE_EVENT,
         .version = RDF_QUEUE_EVENT_VERSION,
         .header = NULL,
         .header_size = 0,
         .data = &rdf_record,
         .data_size = sizeof(rdf_record),
         .use_compression = use_compression,
      };

      if (!ac_rdf_chunk_write(rdf_writer, &chunk))
         return false;
   }
   return true;
}

/* Write CodeObject chunks (one per code object, ELF binary). */
#if defined(USE_LIBELF)
static bool
write_code_object_chunks(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                         const struct rgp_code_object *code_object)
{
   if (!code_object || code_object->record_count == 0)
      return true;

   bool ret = true;
   uint32_t elf_gfxip_flags = ac_gfx_level_to_elf_gfxip_level(info->gfx_level);

   list_for_each_entry (struct rgp_code_object_record, record, &code_object->record, list) {
      /* Generate ELF into memory buffer.
       * To reuse ac_rgp_file_write_elf_object, use memfd_create + fdopen for now
       * (memstream has SEEK_END issues in ac_rgp_file_write_elf_object's seek-back patching).
       * TODO: Maybe change to something prettier in the future to avoid the extra malloc
       * and fread from the memfd. */
      int memfd = memfd_create("ac_elf", MFD_CLOEXEC);
      if (memfd < 0)
         continue;

      FILE *elf_file = fdopen(memfd, "w+b");
      if (!elf_file) {
         close(memfd);
         continue;
      }

      uint32_t written_size = 0;
      ac_rgp_file_write_elf_object(elf_file, 0, record, &written_size, elf_gfxip_flags);
      fflush(elf_file);

      if (written_size == 0) {
         fclose(elf_file);
         continue;
      }

      /* Read back the ELF from the memfd into a heap buffer */
      void *elf_buffer = malloc(written_size);
      if (!elf_buffer) {
         fclose(elf_file);
         continue;
      }

      rewind(elf_file);
      size_t bytes_read = fread(elf_buffer, 1, written_size, elf_file);
      fclose(elf_file); /* also closes memfd */

      if (bytes_read != written_size) {
         free(elf_buffer);
         continue;
      }

      /* Chunk header */
      struct rdf_code_object_header rdf_header = {
         .pci_id = info->pci_id,
         .code_object_hash = {record->pipeline_hash[0], record->pipeline_hash[1]},
      };

      struct ac_rdf_chunk_info chunk = {
         .identifier = RDF_CHUNK_ID_CODE_OBJECT,
         .version = RDF_CODE_OBJECT_VERSION,
         .header = &rdf_header,
         .header_size = sizeof(rdf_header),
         .data = elf_buffer,
         .data_size = written_size,
         .use_compression = false, /* Compression hardcoded to false. It always must be disabled for
                                    * CodeObject (known compression issues). */
      };

      ret = ac_rdf_chunk_write(rdf_writer, &chunk);
      free(elf_buffer);

      if (!ret)
         return ret; /* Abort the loop on first failure. */
   }

   return ret;
}
#endif /* USE_LIBELF */

/* Write SPM chunks (SpmSession + SpmCounterData). */
static bool
write_spm_chunks(struct ac_rdf_writer *rdf_writer, const struct radeon_info *info,
                 const struct ac_spm_trace *spm_trace, bool use_compression)
{
   if (!spm_trace || spm_trace->num_counters == 0)
      return true;

   uint32_t num_samples = spm_trace->num_samples;
   uint32_t num_counters = spm_trace->num_counters;

   /* Extract timestamps from SPM data */
   size_t timestamps_size = num_samples * sizeof(uint64_t);
   uint64_t *timestamps = (uint64_t *)calloc(num_samples, sizeof(uint64_t));
   if (!timestamps)
      return false;

   uint8_t *spm_data_ptr = (uint8_t *)spm_trace->ptr + 32; /* Skip reserved 32 bytes */
   uint64_t sample_size_in_qwords = spm_trace->sample_size_in_bytes / sizeof(uint64_t);
   uint64_t *timestamp_ptr = (uint64_t *)spm_data_ptr;

   for (uint32_t s = 0; s < num_samples; s++) {
      timestamps[s] = timestamp_ptr[s * sample_size_in_qwords];
   }

   /* Write SpmSession chunk: header = session info, data = timestamps */
   struct rdf_spm_session_header rdf_session_header = {
      .pci_id = info->pci_id,
      .flags = 0,
      .sampling_interval = spm_trace->sample_interval,
      .num_timestamps = num_samples,
      .num_spm_counters = num_counters,
   };

   struct ac_rdf_chunk_info session_chunk = {
      .identifier = RDF_CHUNK_ID_SPM_SESSION,
      .version = RDF_SPM_SESSION_VERSION,
      .header = &rdf_session_header,
      .header_size = sizeof(rdf_session_header),
      .data = timestamps,
      .data_size = timestamps_size,
      .use_compression = use_compression,
   };

   bool ret = ac_rdf_chunk_write(rdf_writer, &session_chunk);
   free(timestamps);

   if (!ret)
      return ret;

   /* Write SpmCounterData chunks: one per counter */
   uint32_t sample_size_in_hwords = spm_trace->sample_size_in_bytes / sizeof(uint16_t);
   uint16_t *counter_values_ptr = (uint16_t *)spm_data_ptr;

   for (uint32_t c = 0; c < num_counters; c++) {
      /* Extract counter values for this counter across all samples */
      uint16_t *values = (uint16_t *)calloc(num_samples, sizeof(uint16_t));
      if (!values)
         return false;

      uint64_t offset = spm_trace->counters[c].offset;
      for (uint32_t s = 0; s < num_samples; s++) {
         values[s] = counter_values_ptr[offset + s * sample_size_in_hwords];
      }

      struct rdf_spm_counter_data_header rdf_counter_header = {
         .pci_id = info->pci_id,
         .gpu_block = spm_trace->counters[c].gpu_block,
         .block_instance = spm_trace->counters[c].instance,
         .event_index = spm_trace->counters[c].event_id,
         .data_size = sizeof(uint16_t),
      };

      struct ac_rdf_chunk_info counter_chunk = {
         .identifier = RDF_CHUNK_ID_SPM_COUNTER,
         .version = RDF_SPM_COUNTER_VERSION,
         .header = &rdf_counter_header,
         .header_size = sizeof(rdf_counter_header),
         .data = values,
         .data_size = num_samples * sizeof(uint16_t),
         .use_compression = use_compression,
      };

      ret = ac_rdf_chunk_write(rdf_writer, &counter_chunk);
      free(values);

      if (!ret)
         return ret; /* Abort the loop on first failure. */
   }

   return ret;
}

/* Main function: convert SQTT trace and other data to RDF format. */
void *
ac_sqtt_to_rdf(const struct radeon_info *info, const struct ac_sqtt_trace *sqtt_trace,
               const struct ac_spm_trace *spm_trace, bool instruction_timing_enabled,
               uint32_t buffer_size, const char *trace_config_json, uint32_t chunk_mask,
               uint32_t compression_mask, size_t *out_size)
{
   if (!out_size)
      return NULL;

   *out_size = 0;

   if (!info || !sqtt_trace)
      return NULL;

   /* Create the RDF writer (buffer-backed) */
   struct ac_rdf_writer *rdf_writer = ac_rdf_writer_init_buffer();
   if (!rdf_writer) {
      fprintf(stderr, "ac/rdf: Failed to create RDF writer\n");
      return NULL;
   }

   bool result = true;

   /* Write AsicInfo chunk */
   if (chunk_mask & RDF_CHUNK_MASK_ASIC_INFO) {
      result = write_asic_info_chunk(rdf_writer, info, sqtt_trace->trace_shader_core_clock,
                                     sqtt_trace->trace_memory_clock,
                                     compression_mask & RDF_CHUNK_MASK_ASIC_INFO);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write AsicInfo chunk\n");
   }

   /* Write ApiInfo chunk */
   if (result && (chunk_mask & RDF_CHUNK_MASK_API_INFO)) {
      result = write_api_info_chunk(rdf_writer, compression_mask & RDF_CHUNK_MASK_API_INFO);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write ApiInfo chunk\n");
   }

   /* Write TraceConfig chunk (required by RGP) */
   if (result && (chunk_mask & RDF_CHUNK_MASK_TRACE_CONFIG)) {
      result = write_trace_config_chunk(rdf_writer, trace_config_json,
                                        compression_mask & RDF_CHUNK_MASK_TRACE_CONFIG);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write TraceConfig chunk\n");
   }

   /* Write ClockCalibration chunks */
   if (result && (chunk_mask & RDF_CHUNK_MASK_CLOCK_CALIBRATION) &&
       sqtt_trace->rgp_clock_calibration) {
      result = write_clock_calibration_chunks(rdf_writer, info, sqtt_trace->rgp_clock_calibration,
                                              compression_mask & RDF_CHUNK_MASK_CLOCK_CALIBRATION);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write ClockCalibration chunks\n");
   }

   /* Write SQTT desc/data chunk pairs */
   if (result && (chunk_mask & RDF_CHUNK_MASK_SQTT_DATA)) {
      result = write_sqtt_chunks(rdf_writer, info, sqtt_trace, instruction_timing_enabled,
                                 buffer_size, compression_mask & RDF_CHUNK_MASK_SQTT_DATA);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write SQTT chunks\n");
   }

   /* Write PsoCorrelation chunk */
   if (result && (chunk_mask & RDF_CHUNK_MASK_CODE_OBJECT) && sqtt_trace->rgp_pso_correlation) {
      result = write_pso_correlation_chunk(rdf_writer, info, sqtt_trace->rgp_pso_correlation);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write PsoCorrelation chunk\n");
   }

   /* Write CodeObjectLoaderEvents chunk */
   if (result && (chunk_mask & RDF_CHUNK_MASK_CODE_OBJECT) && sqtt_trace->rgp_loader_events) {
      result = write_loader_events_chunk(rdf_writer, info, sqtt_trace->rgp_loader_events);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write COLoadEvent chunk\n");
   }

   /* Write CodeObject chunks */
#if defined(USE_LIBELF)
   if (result && (chunk_mask & RDF_CHUNK_MASK_CODE_OBJECT) && sqtt_trace->rgp_code_object) {
      result = write_code_object_chunks(rdf_writer, info, sqtt_trace->rgp_code_object);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write CodeObject chunks\n");
   }
#endif

   /* Write QueueInfo chunks (one per queue) */
   if (result && (chunk_mask & RDF_CHUNK_MASK_QUEUE_TIMINGS) && sqtt_trace->rgp_queue_info) {
      result = write_queue_info_chunks(rdf_writer, info, sqtt_trace->rgp_queue_info,
                                       compression_mask & RDF_CHUNK_MASK_QUEUE_TIMINGS);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write QueueInfo chunks\n");
   }

   /* Write QueueEvent chunks (one per event) */
   if (result && (chunk_mask & RDF_CHUNK_MASK_QUEUE_TIMINGS) && sqtt_trace->rgp_queue_event) {
      result = write_queue_event_chunks(rdf_writer, info, sqtt_trace->rgp_queue_event,
                                        sqtt_trace->rgp_queue_info,
                                        compression_mask & RDF_CHUNK_MASK_QUEUE_TIMINGS);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write QueueEvent chunks\n");
   }

   /* Write SPM chunks if SPM data is available */
   if (result && (chunk_mask & RDF_CHUNK_MASK_SPM_DATA) && spm_trace) {
      result = write_spm_chunks(rdf_writer, info, spm_trace,
                                compression_mask & RDF_CHUNK_MASK_SPM_DATA);
      if (!result)
         fprintf(stderr, "ac/rdf: Failed to write SPM chunks\n");
   }

   void *out_rdf_trace = NULL;

   /* Write chunk index and back-patch the file header. */
   if (result && ac_rdf_finalize(rdf_writer)) {
      const void *rdf_data = NULL;
      uint64_t rdf_size = 0;

      /* Get RDF data owned by the writer and copy it; the writer will be destroyed. */
      ac_rdf_get_buffer(rdf_writer, &rdf_data, &rdf_size);

      out_rdf_trace = malloc(rdf_size);
      if (out_rdf_trace) {
         memcpy(out_rdf_trace, rdf_data, rdf_size);
         *out_size = rdf_size;
      }
   } else if (result) {
      fprintf(stderr, "ac/rdf: Failed to finalize RDF writer\n");
   }

   ac_rdf_writer_destroy(rdf_writer);

   return out_rdf_trace;
}
