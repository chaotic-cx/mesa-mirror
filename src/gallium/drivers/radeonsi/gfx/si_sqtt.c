/*
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "si_gfx.h"
#include "ac_cmdbuf_cp.h"
#include "ac_shader_util.h"
#include "amd_family.h"
#include "si_build_pm4.h"
#include "si_pipe.h"

#include "util/hash_table.h"
#include "util/u_debug.h"
#include "util/u_memory.h"
#include "ac_rgp.h"
#include "ac_sqtt.h"

#define SI_SQTT_TIMESTAMP_SIZE   8
#define SI_SQTT_QUEUE_GRAPHICS   1
#define SI_SQTT_QUEUE_COMPUTE    2

static void
si_emit_spi_config_cntl(struct si_context *sctx,
                        struct radeon_cmdbuf *cs, bool enable);

static bool si_sqtt_init_bo(struct si_context *sctx)
{
   unsigned max_se = sctx->screen->info.max_se;
   uint64_t size;

   /* The buffer size and address need to be aligned in HW regs. The size is
    * aligned when it is set. */
   assert(!(sctx->sqtt.data->buffer_size & ((1u << SQTT_BUFFER_ALIGN_SHIFT) - 1)));

   /* Compute total size of the thread trace BO for all SEs. */
   size = align64(sizeof(struct ac_sqtt_data_info) * max_se,
                  1ull << SQTT_BUFFER_ALIGN_SHIFT);
   size += sctx->sqtt.data->buffer_size * (uint64_t)max_se;

   if (size > UINT32_MAX)
      return false;

   sctx->sqtt.data->bo =
      pipe_aligned_buffer_create(&sctx->screen->b, SI_RESOURCE_FLAG_DRIVER_INTERNAL,
                                 PIPE_USAGE_DEFAULT, size, 4096);
   if (!sctx->sqtt.data->bo)
      return false;

   sctx->sqtt.data->buffer_va = si_resource(sctx->sqtt.data->bo)->gpu_address;

   return true;
}

static bool si_sqtt_add_queue_info_record(struct si_context *sctx)
{
   struct rgp_queue_info *queue_info = &sctx->sqtt.data->rgp_queue_info;
   struct rgp_queue_info_record *record;

   record = calloc(1, sizeof(*record));
   if (!record)
      return false;

   record->queue_id = (uintptr_t)sctx;
   record->queue_context = (uintptr_t)sctx->ctx;
   record->hardware_info.engine_type = sctx->is_gfx_queue ? SQTT_ENGINE_TYPE_UNIVERSAL
                                                          : SQTT_ENGINE_TYPE_COMPUTE;
   record->hardware_info.queue_type = sctx->is_gfx_queue ? SQTT_QUEUE_TYPE_UNIVERSAL
                                                         : SQTT_QUEUE_TYPE_COMPUTE;

   simple_mtx_lock(&queue_info->lock);
   list_addtail(&record->list, &queue_info->record);
   queue_info->record_count++;
   simple_mtx_unlock(&queue_info->lock);

   return true;
}

static void si_emit_sqtt_start(struct si_context *sctx,
                               struct radeon_cmdbuf *cs,
                               enum amd_ip_type ip_type)
{
   struct si_screen *sscreen = sctx->screen;
   const bool is_compute_queue = ip_type == AMD_IP_COMPUTE;
   struct ac_pm4_state *pm4;

   pm4 = ac_pm4_create_sized(&sscreen->info, false, 512, is_compute_queue);
   if (!pm4)
      return;

   ac_sqtt_emit_start(&sscreen->info, pm4, sctx->sqtt.data, is_compute_queue);
   ac_pm4_finalize(pm4);
   ac_pm4_emit_commands(&cs->current, pm4);
   ac_pm4_free_state(pm4);
}

static void si_emit_sqtt_stop(struct si_context *sctx, struct radeon_cmdbuf *cs,
                              enum amd_ip_type ip_type)
{
   struct si_screen *sscreen = sctx->screen;
   const bool is_compute_queue = ip_type == AMD_IP_COMPUTE;
   struct ac_pm4_state *pm4;

   pm4 = ac_pm4_create_sized(&sscreen->info, false, 512, is_compute_queue);
   if (!pm4)
      return;

   ac_sqtt_emit_stop(&sscreen->info, pm4, is_compute_queue);
   ac_pm4_finalize(pm4);
   ac_pm4_emit_commands(&cs->current, pm4);
   ac_pm4_clear_state(pm4, &sscreen->info, false, is_compute_queue);

   if (sctx->screen->info.has_sqtt_rb_harvest_bug) {
      /* Some chips with disabled RBs should wait for idle because FINISH_DONE
       * doesn't work. */
      sctx->barrier_flags |= AC_BARRIER_SYNC_AND_INV_CB | AC_BARRIER_SYNC_AND_INV_DB |
                             AC_BARRIER_SYNC_CS;
      sctx->emit_barrier(sctx, cs);
   }

   ac_sqtt_emit_wait(&sscreen->info, pm4, sctx->sqtt.data, is_compute_queue);
   ac_pm4_finalize(pm4);
   ac_pm4_emit_commands(&cs->current, pm4);
   ac_pm4_free_state(pm4);
}

static void si_sqtt_start(struct si_context *sctx, struct radeon_cmdbuf *cs)
{
   enum amd_ip_type ip_type = sctx->ws->cs_get_ip_type(cs);

   radeon_begin(cs);

   switch (ip_type) {
      case AMD_IP_GFX:
         radeon_emit(PKT3(PKT3_CONTEXT_CONTROL, 1, 0));
         radeon_emit(S_281_UPDATE_LOAD_ENABLES(1));
         radeon_emit(S_282_UPDATE_SHADOW_ENABLES(1));
         break;
      case AMD_IP_COMPUTE:
         radeon_emit(PKT3(PKT3_NOP, 0, 0));
         radeon_emit(0);
         break;
      default:
        /* Unsupported. */
        assert(false);
   }
   radeon_end();

   radeon_add_to_buffer_list(sctx, cs, si_resource(sctx->sqtt.data->bo), RADEON_USAGE_READWRITE);
   if (sctx->sqtt.spm.bo)
      radeon_add_to_buffer_list(sctx, cs, si_resource(sctx->sqtt.spm.bo), RADEON_USAGE_READWRITE);

   si_cp_dma_wait_for_idle(sctx, cs);

   /* Make sure to wait-for-idle before starting SQTT. */
   sctx->barrier_flags |= AC_BARRIER_SYNC_PS | AC_BARRIER_SYNC_CS |
                          AC_BARRIER_INV_ICACHE | AC_BARRIER_INV_SMEM |
                          AC_BARRIER_INV_VMEM | AC_BARRIER_INV_L2 |
                          AC_BARRIER_PFP_SYNC_ME;
   sctx->emit_barrier(sctx, cs);

   si_inhibit_clockgating(sctx, cs, true);

   /* Enable SQG events that collects thread trace data. */
   si_emit_spi_config_cntl(sctx, cs, true);

   if (sctx->sqtt.spm.bo) {
      ac_emit_spm_reset(&cs->current);
      ac_pc_emit_shaders(sctx->screen->info.gfx_level, &cs->current,
                         ac_sqtt_get_shader_mask(&sctx->screen->info));
      si_emit_spm_setup(sctx, &sctx->sqtt.spm, cs);
   }

   si_emit_sqtt_start(sctx, cs, ip_type);

   if (sctx->sqtt.spm.bo)
      ac_emit_spm_start(&cs->current, AMD_IP_GFX, &sctx->screen->info);
}

static void si_sqtt_stop(struct si_context *sctx, struct radeon_cmdbuf *cs)
{
   struct radeon_winsys *ws = sctx->ws;
   enum amd_ip_type ip_type = sctx->ws->cs_get_ip_type(cs);

   radeon_begin(cs);

   switch (ip_type) {
      case AMD_IP_GFX:
         radeon_emit(PKT3(PKT3_CONTEXT_CONTROL, 1, 0));
         radeon_emit(S_281_UPDATE_LOAD_ENABLES(1));
         radeon_emit(S_282_UPDATE_SHADOW_ENABLES(1));
         break;
      case AMD_IP_COMPUTE:
         radeon_emit(PKT3(PKT3_NOP, 0, 0));
         radeon_emit(0);
         break;
      default:
        /* Unsupported. */
        assert(false);
   }
   radeon_end();

   ws->cs_add_buffer(cs, si_resource(sctx->sqtt.data->bo)->buf, RADEON_USAGE_READWRITE,
                     RADEON_DOMAIN_VRAM);

   if (sctx->sqtt.spm.bo)
      ws->cs_add_buffer(cs, si_resource(sctx->sqtt.spm.bo)->buf, RADEON_USAGE_READWRITE,
                        RADEON_DOMAIN_VRAM);

   si_cp_dma_wait_for_idle(sctx, cs);

   if (sctx->sqtt.spm.bo)
      ac_emit_spm_stop(&cs->current, AMD_IP_GFX, &sctx->screen->info);

   /* Make sure to wait-for-idle before stopping SQTT. */
   sctx->barrier_flags |= AC_BARRIER_SYNC_PS | AC_BARRIER_SYNC_CS |
                          AC_BARRIER_INV_ICACHE | AC_BARRIER_INV_SMEM |
                          AC_BARRIER_INV_VMEM | AC_BARRIER_INV_L2 |
                          AC_BARRIER_PFP_SYNC_ME;
   sctx->emit_barrier(sctx, cs);

   si_emit_sqtt_stop(sctx, cs, ip_type);

   if (sctx->sqtt.spm.bo)
      ac_emit_spm_reset(&cs->current);

   /* Restore previous state by disabling SQG events. */
   si_emit_spi_config_cntl(sctx, cs, false);

   si_inhibit_clockgating(sctx, cs, false);
}

static void si_sqtt_init_cs(struct si_context *sctx)
{
   struct radeon_winsys *ws = sctx->ws;
   unsigned barriers;

   for (unsigned i = 0; i < ARRAY_SIZE(sctx->sqtt.start_cs); i++) {
      sctx->sqtt.start_cs[i] = CALLOC_STRUCT(radeon_cmdbuf);
      if (!ws->cs_create(sctx->sqtt.start_cs[i], sctx->ctx, (enum amd_ip_type)i,
                         NULL, NULL)) {
         free(sctx->sqtt.start_cs[i]);
         sctx->sqtt.start_cs[i] = NULL;
         return;
      }

      /* Save and restore global barrier_flags. */
      barriers = sctx->barrier_flags;
      sctx->barrier_flags = 0;

      si_sqtt_start(sctx, sctx->sqtt.start_cs[i]);

      sctx->sqtt.stop_cs[i] = CALLOC_STRUCT(radeon_cmdbuf);
      if (!ws->cs_create(sctx->sqtt.stop_cs[i], sctx->ctx, (enum amd_ip_type)i,
                         NULL, NULL)) {
         ws->cs_destroy(sctx->sqtt.start_cs[i]);
         free(sctx->sqtt.start_cs[i]);
         sctx->sqtt.start_cs[i] = NULL;
         free(sctx->sqtt.stop_cs[i]);
         sctx->sqtt.stop_cs[i] = NULL;
         sctx->barrier_flags = barriers;
         return;
      }

      si_sqtt_stop(sctx, sctx->sqtt.stop_cs[i]);
      sctx->barrier_flags = barriers;
   }
}

static void si_begin_sqtt(struct si_context *sctx, struct radeon_cmdbuf *rcs)
{
   struct radeon_cmdbuf *cs = sctx->sqtt.start_cs[sctx->ws->cs_get_ip_type(rcs)];
   sctx->ws->cs_flush(cs, 0, NULL);
}

static void si_end_sqtt(struct si_context *sctx, struct radeon_cmdbuf *rcs)
{
   struct radeon_cmdbuf *cs = sctx->sqtt.stop_cs[sctx->ws->cs_get_ip_type(rcs)];
   sctx->ws->cs_flush(cs, 0, &sctx->sqtt.last_fence);
}

static bool
si_sqtt_resize_bo(struct si_context *sctx)
{
   struct pipe_resource *old_bo = sctx->sqtt.data->bo;
   uint32_t old_size = sctx->sqtt.data->buffer_size;

   if (!ac_sqtt_update_bo_size(sctx->sqtt.data, "AMD"))
      return false;

   /* keep the current buffer when the bigger one can't be allocated */
   if (!si_sqtt_init_bo(sctx)) {
      sctx->sqtt.data->bo = old_bo;
      sctx->sqtt.data->buffer_size = old_size;
      return false;
   }

   pipe_resource_reference(&old_bo, NULL);
   return true;
}

static bool si_get_sqtt_trace(struct si_context *sctx,
                              struct ac_sqtt_trace *sqtt,
                              struct pipe_transfer **transfer,
                              bool *resized)
{
   struct pipe_resource *bo = sctx->sqtt.data->bo;

   memset(sqtt, 0, sizeof(*sqtt));
   *resized = false;

   sctx->sqtt.data->ptr = pipe_buffer_map(&sctx->b, bo, PIPE_MAP_READ, transfer);

   if (!sctx->sqtt.data->ptr)
      return false;

   if (!ac_sqtt_get_trace(sctx->sqtt.data, &sctx->screen->info, sqtt)) {
      pipe_buffer_unmap(&sctx->b, *transfer);
      *transfer = NULL;
      sctx->sqtt.data->ptr = NULL;

      if (!si_sqtt_resize_bo(sctx)) {
         mesa_loge("Failed to resize the SQTT buffer.");
      } else {
         for (int i = 0; i < ARRAY_SIZE(sctx->sqtt.start_cs); i++) {
            sctx->screen->ws->cs_destroy(sctx->sqtt.start_cs[i]);
            sctx->screen->ws->cs_destroy(sctx->sqtt.stop_cs[i]);
         }
         si_sqtt_init_cs(sctx);
         *resized = true;
      }
      return false;
   }

   return true;
}

bool si_init_sqtt(struct si_context *sctx)
{
   static bool warn_once = true;
   if (warn_once) {
      mesa_logw("Thread trace support is experimental *");
      warn_once = false;
   }

   sctx->sqtt.data = CALLOC_STRUCT(ac_sqtt);

   list_inithead(&sctx->sqtt.timestamp.list);

   if (sctx->gfx_level < GFX8) {
      mesa_loge("GPU hardware not supported: refer to "
                "the RGP documentation for the list of "
                "supported GPUs!");
      return false;
   }

   if (sctx->gfx_level > GFX12) {
      mesa_loge("Thread trace is not supported "
                "for that GPU!");
      return false;
   }

   if (!ac_sqtt_update_bo_size(sctx->sqtt.data, "AMD"))
      return false;

   sctx->sqtt.data->instruction_timing_enabled =
      debug_get_bool_option("AMD_THREAD_TRACE_INSTRUCTION_TIMING", true);
   sctx->sqtt.data->instruction_timing_se_mask =
      sctx->sqtt.data->instruction_timing_enabled
         ? (uint32_t)debug_get_num_option("AMD_THREAD_TRACE_INSTRUCTION_TIMING_SE_MASK", ~0u)
         : 0;
   sctx->sqtt.start_frame = 10;
   sctx->sqtt.end_frame = 10;
   sctx->sqtt.mode = AC_RGP_CAPTURE_MODE_FRAME;

   const char *trigger = os_get_option("AMD_THREAD_TRACE_TRIGGER");
   if (trigger) {
      char *endptr;
      sctx->sqtt.start_frame = strtol(trigger, &endptr, 10);
      sctx->sqtt.end_frame = sctx->sqtt.start_frame;
      if (trigger == endptr) {
         /* This isn't a frame number, must be a file */
         sctx->sqtt.trigger_file = strdup(trigger);
         sctx->sqtt.start_frame = -1;
      } else {
         if (*endptr == '-' && strlen(endptr) > 1) {
            sctx->sqtt.end_frame = strtol(endptr + 1, &endptr, 10);
            if (*endptr == 's')
               sctx->sqtt.mode = AC_RGP_CAPTURE_MODE_SUBMIT;
         }
      }
   }

   if (!si_sqtt_init_bo(sctx))
      return false;

   sctx->sqtt.data->pipeline_bos = _mesa_hash_table_u64_create(NULL);

   ac_sqtt_init(sctx->sqtt.data);

   if (sctx->gfx_level >= GFX10 &&
       debug_get_bool_option("AMD_THREAD_TRACE_SPM", sctx->gfx_level < GFX11)) {
      /* Limit SPM counters to GFX10 and GFX10_3 for now */
      ASSERTED bool r = si_spm_init(sctx, &sctx->sqtt.spm);
      assert(r);
   }

   si_sqtt_init_cs(sctx);
   if (!si_sqtt_add_queue_info_record(sctx))
      mesa_loge("Failed to add queue info record");

   sctx->sqtt.next_event = EventInvalid;
   sctx->sqtt.cb_id = 0;
   sctx->sqtt.device_id = ((uint64_t)sctx->screen->info.pci.domain << 48) |
                          ((uint64_t)sctx->screen->info.pci.bus << 40)    |
                          ((uint64_t)sctx->screen->info.pci.dev << 35)    |
                          ((uint64_t)sctx->screen->info.pci.func << 32)   |
                          ((uint64_t)sctx->screen->info.pci_id);

   /* TODO: Proper clock calibration */
   uint64_t cpu_timestamp = os_time_get_nano();
   uint64_t gpu_timestamp = sctx->ws->query_value(sctx->ws, RADEON_TIMESTAMP);   
   ac_sqtt_add_clock_calibration(sctx->sqtt.data, cpu_timestamp, gpu_timestamp);

   uint64_t shader_clock_freq, memory_clock_freq;
   shader_clock_freq = sctx->ws->query_value(sctx->ws, RADEON_CURRENT_SCLK);
   memory_clock_freq = sctx->ws->query_value(sctx->ws, RADEON_CURRENT_MCLK);
   ac_sqtt_set_gpu_trace_clocks(sctx->sqtt.data, shader_clock_freq, memory_clock_freq); 

   return true;
}

static void si_sqtt_reset_queue_state(struct si_context *sctx)
{
   struct rgp_clock_calibration *clock_calibration = &sctx->sqtt.data->rgp_clock_calibration;
   struct rgp_queue_event *queue_event = &sctx->sqtt.data->rgp_queue_event;

   simple_mtx_lock(&clock_calibration->lock);
   list_for_each_entry_safe (struct rgp_clock_calibration_record, record, &clock_calibration->record, list) {
      clock_calibration->record_count--;
      list_del(&record->list);
      free(record);
   }
   simple_mtx_unlock(&clock_calibration->lock);

   simple_mtx_lock(&queue_event->lock);
   list_for_each_entry_safe (struct rgp_queue_event_record, record,
                             &queue_event->record, list) {
      list_del(&record->list);
      queue_event->record_count--;
      free(record);
   }
   simple_mtx_unlock(&queue_event->lock);

   list_for_each_entry_safe (struct si_sqtt_timestamp, timestamp,
                             &sctx->sqtt.timestamp.list, list) {
      si_resource_reference(&timestamp->bo, NULL);
      list_del(&timestamp->list);
      free(timestamp);
   }

   sctx->sqtt.timestamp.offset = 0;
   sctx->sqtt.cb_id = 0;
}

bool si_is_sqtt_active(const struct si_context *sctx)
{
   return sctx->sqtt.status == SQTT_STATUS_TRACING;
}

void si_destroy_sqtt(struct si_context *sctx)
{
   struct si_screen *sscreen = sctx->screen;
   struct pipe_resource *bo = sctx->sqtt.data->bo;
   pipe_resource_reference(&bo, NULL);

   free(sctx->sqtt.trigger_file);

   for (int i = 0; i < ARRAY_SIZE(sctx->sqtt.start_cs); i++) {
      if (sctx->sqtt.start_cs[i])
         sscreen->ws->cs_destroy(sctx->sqtt.start_cs[i]);
      if (sctx->sqtt.stop_cs[i])
         sscreen->ws->cs_destroy(sctx->sqtt.stop_cs[i]);
   }

   struct rgp_pso_correlation *pso_correlation =
      &sctx->sqtt.data->rgp_pso_correlation;
   struct rgp_loader_events *loader_events = &sctx->sqtt.data->rgp_loader_events;
   struct rgp_code_object *code_object = &sctx->sqtt.data->rgp_code_object;
   struct rgp_queue_info *queue_info = &sctx->sqtt.data->rgp_queue_info;
   list_for_each_entry_safe (struct rgp_pso_correlation_record, record,
                             &pso_correlation->record, list) {
      list_del(&record->list);
      pso_correlation->record_count--;
      free(record);
   }

   list_for_each_entry_safe (struct rgp_loader_events_record, record,
                             &loader_events->record, list) {
      list_del(&record->list);
      loader_events->record_count--;
      free(record);
   }

   list_for_each_entry_safe (struct rgp_queue_info_record, record,
                             &queue_info->record, list) {
      list_del(&record->list);
      queue_info->record_count--;
      free(record);
   }

   list_for_each_entry_safe (struct rgp_code_object_record, record,
                             &code_object->record, list) {
      uint32_t mask = record->shader_stages_mask;
      int i;

      /* Free the disassembly. */
      while (mask) {
         i = u_bit_scan(&mask);
         free(record->shader_data[i].code);
      }
      list_del(&record->list);
      free(record);
      code_object->record_count--;
   }

   si_sqtt_reset_queue_state(sctx);
   ac_sqtt_finish(sctx->sqtt.data);

   si_resource_reference(&sctx->sqtt.timestamp.bo, NULL);

   hash_table_foreach (&sctx->sqtt.data->pipeline_bos->table, entry) {
      struct si_sqtt_fake_pipeline *pipeline =
         (struct si_sqtt_fake_pipeline *)entry->data;
      si_resource_reference(&pipeline->bo, NULL);
      FREE(pipeline);
   }

   free(sctx->sqtt.data);
   sctx->sqtt.data = NULL;

   if (sctx->sqtt.spm.bo)
      si_spm_finish(sctx, &sctx->sqtt.spm);
}

static uint64_t num_frames = 0, num_submits = 0;

static bool si_sqtt_must_start_capture(const struct si_context *sctx, bool end_of_frame) {
   if (sctx->sqtt.status != SQTT_STATUS_OFF)
      return false;
   if (sctx->sqtt.mode == AC_RGP_CAPTURE_MODE_SUBMIT)
      return num_submits == sctx->sqtt.start_frame;
   /* captures start where a frame begins: at the first submit or the previous frame's present */
   if (num_submits > 0 && !end_of_frame)
      return false;
   /* File based capture*/
   if (sctx->sqtt.trigger_file &&
       access(sctx->sqtt.trigger_file, W_OK) == 0) {
      if (unlink(sctx->sqtt.trigger_file) == 0) {
         return true;
      } else {
         /* Do not enable tracing if we cannot remove the file,
          * because by then we'll trace every frame.
          */
         mesa_logw("could not remove thread "
                     "trace trigger file, ignoring");
      }
   }
   return num_frames == sctx->sqtt.start_frame;
}

static bool si_sqtt_must_stop_capture(const struct si_context *sctx, bool end_of_frame) {
   if (sctx->sqtt.status != SQTT_STATUS_TRACING)
      return false;
   if (sctx->sqtt.mode == AC_RGP_CAPTURE_MODE_SUBMIT)
      return num_submits >= sctx->sqtt.end_frame;
   /* frame and file based captures cover whole frames */
   if (!end_of_frame)
      return false;
   /* File based capture*/
   if (sctx->sqtt.start_frame < 0)
      return true;
   return num_frames >= sctx->sqtt.end_frame;
}

static void si_sqtt_stop_capture(struct si_context *sctx, struct radeon_cmdbuf *rcs,
                                 bool allow_retry)
{
   struct ac_sqtt_trace sqtt_trace = {0};
   struct pipe_transfer *transfers[2];
   bool resized = false;

   /* reading the trace back flushes, which must not start another capture */
   sctx->sqtt.status = SQTT_STATUS_STOPPING;

   /* Stop SQTT */
   si_end_sqtt(sctx, rcs);
   assert(sctx->sqtt.last_fence);

   /* Wait for SQTT to finish and read back the bo */
   if (sctx->ws->fence_wait(sctx->ws, sctx->sqtt.last_fence,
                            OS_TIMEOUT_INFINITE) &&
       si_get_sqtt_trace(sctx, &sqtt_trace, &transfers[0], &resized)) {
      struct ac_spm_trace spm_trace;

      /* Map the SPM counter buffer */
      if (sctx->sqtt.spm.bo)
         sctx->sqtt.spm.ptr = pipe_buffer_map(&sctx->b, sctx->sqtt.spm.bo, PIPE_MAP_READ, &transfers[1]);

      if (sctx->sqtt.spm.ptr)
         ac_spm_get_trace(&sctx->sqtt.spm, &spm_trace);

      struct ac_rgp_capture_info info = {
         .mode = sctx->sqtt.mode,
         .frame_idx = sctx->sqtt.mode == AC_RGP_CAPTURE_MODE_FRAME ? num_frames : num_submits
      };

      ac_dump_rgp_capture(&sctx->screen->info, &sqtt_trace,
                          sctx->sqtt.spm.ptr ? &spm_trace : NULL, &info);

      if (sctx->sqtt.spm.ptr) {
         pipe_buffer_unmap(&sctx->b, transfers[1]);
         sctx->sqtt.spm.ptr = NULL;
      }

      pipe_buffer_unmap(&sctx->b, transfers[0]);
      sctx->sqtt.data->ptr = NULL;
      sctx->sqtt.status = SQTT_STATUS_OFF;
   } else if (allow_retry && resized) {
      if (!sctx->sqtt.trigger_file) {
         int end_frame = sctx->sqtt.end_frame;
         sctx->sqtt.end_frame += sctx->sqtt.end_frame - sctx->sqtt.start_frame;
         sctx->sqtt.start_frame = end_frame;
      }

      /* Restart SQTT to try to capture the next frame. */
      si_begin_sqtt(sctx, rcs);
      sctx->sqtt.status = SQTT_STATUS_TRACING;
   }
   si_sqtt_reset_queue_state(sctx);
}

void si_handle_sqtt(struct si_context *sctx, struct radeon_cmdbuf *rcs, bool end_of_frame)
{
   if (si_sqtt_must_stop_capture(sctx, end_of_frame))
      si_sqtt_stop_capture(sctx, rcs, true);

   if (end_of_frame)
      num_frames++;

   if (si_sqtt_must_start_capture(sctx, end_of_frame)) {
      /* Wait for last submission */
      if (sctx->last_gfx_fence)
         sctx->ws->fence_wait(sctx->ws, sctx->last_gfx_fence,
                              OS_TIMEOUT_INFINITE);

      /* Start SQTT */
      si_begin_sqtt(sctx, rcs);

      sctx->sqtt.status = SQTT_STATUS_TRACING;

      /* Force shader update to make sure si_sqtt_describe_pipeline_bind is
         * called for the current "pipeline".
         */
      sctx->dirty_shaders_mask |= SI_SQTT_STATE_DIRTY_BIT;
   }
   num_submits++;
}

/* ends a running capture without starting another one or counting a submit */
void si_sqtt_finish_capture(struct si_context *sctx)
{
   if (sctx->sqtt.status == SQTT_STATUS_TRACING)
      si_sqtt_stop_capture(sctx, &sctx->gfx_cs, false);
}

static void si_emit_sqtt_userdata(struct si_context *sctx,
                                  struct radeon_cmdbuf *cs, const void *data,
                                  uint32_t num_dwords)
{
   const enum amd_ip_type ip_type = sctx->ws->cs_get_ip_type(cs);
   const uint32_t *dwords = (uint32_t *)data;

   radeon_begin(cs);

   while (num_dwords > 0) {
      uint32_t count = MIN2(num_dwords, 2);

      radeon_set_uconfig_perfctr_reg_seq(sctx->gfx_level, ip_type,
                                         R_030D08_SQ_THREAD_TRACE_USERDATA_2,
                                         count);
      radeon_emit_array(dwords, count);

      dwords += count;
      num_dwords -= count;
   }
   radeon_end();
}

static void
si_emit_spi_config_cntl(struct si_context *sctx,
                        struct radeon_cmdbuf *cs, bool enable)
{
   ac_emit_cp_spi_config_cntl(&cs->current, sctx->gfx_level, enable);
}

static uint32_t num_events = 0;
void si_sqtt_write_event_marker(struct si_context *sctx, struct radeon_cmdbuf *rcs,
                                enum rgp_sqtt_marker_event_type api_type,
                                uint32_t vertex_offset_user_data,
                                uint32_t instance_offset_user_data,
                                uint32_t draw_index_user_data)
{
   struct rgp_sqtt_marker_event marker = {0};

   marker.identifier = RGP_SQTT_MARKER_IDENTIFIER_EVENT;
   marker.api_type = api_type == EventInvalid ? EventCmdDraw : api_type;
   marker.cmd_id = num_events++;
   marker.cb_id = sctx->sqtt.cb_id;

   if (vertex_offset_user_data == UINT_MAX ||
       instance_offset_user_data == UINT_MAX) {
      vertex_offset_user_data = 0;
      instance_offset_user_data = 0;
   }

   if (draw_index_user_data == UINT_MAX)
      draw_index_user_data = vertex_offset_user_data;

   marker.vertex_offset_reg_idx = vertex_offset_user_data;
   marker.instance_offset_reg_idx = instance_offset_user_data;
   marker.draw_index_reg_idx = draw_index_user_data;

   si_emit_sqtt_userdata(sctx, rcs, &marker, sizeof(marker) / 4);

   sctx->sqtt.next_event = EventInvalid;
}

void si_write_event_with_dims_marker(struct si_context *sctx, struct radeon_cmdbuf *rcs,
                                     enum rgp_sqtt_marker_event_type api_type,
                                     uint32_t x, uint32_t y, uint32_t z)
{
   struct rgp_sqtt_marker_event_with_dims marker = {0};

   marker.event.identifier = RGP_SQTT_MARKER_IDENTIFIER_EVENT;
   marker.event.api_type = api_type;
   marker.event.cmd_id = num_events++;
   marker.event.cb_id = sctx->sqtt.cb_id;
   marker.event.has_thread_dims = 1;

   marker.thread_x = x;
   marker.thread_y = y;
   marker.thread_z = z;

   si_emit_sqtt_userdata(sctx, rcs, &marker, sizeof(marker) / 4);
   sctx->sqtt.next_event = EventInvalid;
}

void si_sqtt_describe_barrier_start(struct si_context *sctx, struct radeon_cmdbuf *rcs)
{
   struct rgp_sqtt_marker_barrier_start marker = {0};

   marker.identifier = RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START;
   marker.cb_id = sctx->sqtt.cb_id;
   marker.dword02 = 0xC0000000 + 10; /* RGP_BARRIER_INTERNAL_BASE */

   si_emit_sqtt_userdata(sctx, rcs, &marker, sizeof(marker) / 4);
}

void si_sqtt_describe_barrier_end(struct si_context *sctx, struct radeon_cmdbuf *rcs,
                                  enum ac_rgp_flush_bits flush_bits)
{
   struct rgp_sqtt_marker_barrier_end marker = {0};

   marker.identifier = RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END;
   marker.cb_id = sctx->sqtt.cb_id;

   ac_rgp_flush_bits_to_barrier_marker(flush_bits, &marker);

   si_emit_sqtt_userdata(sctx, rcs, &marker, sizeof(marker) / 4);
}

void si_write_user_event(struct si_context *sctx, struct radeon_cmdbuf *rcs,
                         enum rgp_sqtt_marker_user_event_type type,
                         const char *str, int len)
{
   if (type == UserEventPop) {
      assert(str == NULL);
      struct rgp_sqtt_marker_user_event marker = {0};
      marker.identifier = RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT;
      marker.data_type = type;

      si_emit_sqtt_userdata(sctx, rcs, &marker, sizeof(marker) / 4);
   } else {
      assert(str != NULL);
      struct rgp_sqtt_marker_user_event_with_length marker = {0};
      marker.user_event.identifier = RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT;
      marker.user_event.data_type = type;
      len = MIN2(1024, len);
      marker.length = align(len, 4);

      uint8_t *buffer = alloca(sizeof(marker) + marker.length);
      memcpy(buffer, &marker, sizeof(marker));
      memcpy(buffer + sizeof(marker), str, len);
      memset(buffer + sizeof(marker) + len, 0, marker.length - len);

      si_emit_sqtt_userdata(sctx, rcs, buffer,
                            sizeof(marker) / 4 + marker.length / 4);
   }
}

bool si_sqtt_pipeline_is_registered(struct ac_sqtt *sqtt,
                                    uint64_t pipeline_hash)
{
   simple_mtx_lock(&sqtt->rgp_pso_correlation.lock);
   list_for_each_entry_safe (struct rgp_pso_correlation_record, record,
                             &sqtt->rgp_pso_correlation.record, list) {
      if (record->pipeline_hash[0] == pipeline_hash) {
         simple_mtx_unlock(&sqtt->rgp_pso_correlation.lock);
         return true;
      }
   }
   simple_mtx_unlock(&sqtt->rgp_pso_correlation.lock);

   return false;
}

static enum rgp_hardware_stages
si_sqtt_pipe_to_rgp_shader_stage(union si_shader_key *key, mesa_shader_stage stage)
{
   switch (stage) {
      case MESA_SHADER_VERTEX:
         if (key->ge.as_ls)
            return RGP_HW_STAGE_LS;
         else if (key->ge.as_es)
            return RGP_HW_STAGE_ES;
         else if (key->ge.as_ngg)
            return RGP_HW_STAGE_GS;
         else
            return RGP_HW_STAGE_VS;
      case MESA_SHADER_TESS_CTRL:
         return RGP_HW_STAGE_HS;
      case MESA_SHADER_TESS_EVAL:
         if (key->ge.as_es)
            return RGP_HW_STAGE_ES;
         else if (key->ge.as_ngg)
            return RGP_HW_STAGE_GS;
         else
            return RGP_HW_STAGE_VS;
      case MESA_SHADER_GEOMETRY:
      case MESA_SHADER_MESH:
         return RGP_HW_STAGE_GS;
      case MESA_SHADER_FRAGMENT:
         return RGP_HW_STAGE_PS;
      case MESA_SHADER_COMPUTE:
      case MESA_SHADER_TASK:
         return RGP_HW_STAGE_CS;
      default:
         UNREACHABLE("invalid mesa shader stage");
   }
}

static bool
si_sqtt_add_code_object(struct si_context *sctx,
                        struct si_sqtt_fake_pipeline *pipeline,
                        uint32_t *gfx_sh_offsets)
{
   struct rgp_code_object *code_object = &sctx->sqtt.data->rgp_code_object;
   struct rgp_code_object_record *record;
   bool is_compute = gfx_sh_offsets == NULL;

   record = calloc(1, sizeof(struct rgp_code_object_record));
   if (!record)
      return false;

   record->shader_stages_mask = 0;
   record->num_shaders_combined = 0;
   record->pipeline_hash[0] = pipeline->code_hash;
   record->pipeline_hash[1] = pipeline->code_hash;

   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      struct si_shader *shader;
      enum rgp_hardware_stages hw_stage;

      if (is_compute) {
         if (i != MESA_SHADER_COMPUTE)
            continue;
         shader = &sctx->cs_shader_state.program->shader;
         hw_stage = RGP_HW_STAGE_CS;
      } else if (i <= MESA_SHADER_FRAGMENT) {
         if (!sctx->shaders[i].cso || !sctx->shaders[i].current)
            continue;
         shader = sctx->shaders[i].current;
         hw_stage = si_sqtt_pipe_to_rgp_shader_stage(&shader->key, i);
      } else if (i == MESA_SHADER_MESH) {
         if (!sctx->ms_shader_state.cso)
            continue;
         shader = sctx->ms_shader_state.current;
         hw_stage = RGP_HW_STAGE_GS;
      } else if (i == MESA_SHADER_TASK) {
         if (!sctx->ts_shader_state.program)
            continue;
         shader = &sctx->ts_shader_state.program->shader;
         hw_stage = RGP_HW_STAGE_CS;
      } else {
         continue;
      }

      uint8_t *code = malloc(shader->binary.uploaded_code_size);
      if (!code) {
         free(record);
         return false;
      }
      memcpy(code, shader->binary.uploaded_code, shader->binary.uploaded_code_size);

      uint64_t va = pipeline->bo->gpu_address + (is_compute ? 0 : gfx_sh_offsets[i]);

      memset(record->shader_data[i].rt_shader_name, 0, sizeof(record->shader_data[i].rt_shader_name));
      record->shader_data[i].hash[0] = _mesa_hash_data(code, shader->binary.uploaded_code_size);
      record->shader_data[i].hash[1] = record->shader_data[i].hash[0];
      record->shader_data[i].code_size = shader->binary.uploaded_code_size;
      record->shader_data[i].code = code;
      record->shader_data[i].vgpr_count = shader->config.num_vgprs;
      record->shader_data[i].sgpr_count = shader->config.num_sgprs;
      record->shader_data[i].base_address = va & 0xffffffffffff;
      record->shader_data[i].elf_symbol_offset = 0;
      record->shader_data[i].hw_stage = hw_stage;
      record->shader_data[i].is_combined = false;
      record->shader_data[i].scratch_memory_size = shader->config.scratch_bytes_per_wave;
      record->shader_data[i].lds_size = align(shader->config.lds_size, ac_shader_get_lds_alloc_granularity(sctx->gfx_level));
      record->shader_data[i].wavefront_size = shader->wave_size;

      record->shader_stages_mask |= 1 << i;
      record->num_shaders_combined++;
   }

   simple_mtx_lock(&code_object->lock);
   list_addtail(&record->list, &code_object->record);
   code_object->record_count++;
   simple_mtx_unlock(&code_object->lock);

   return true;
}

bool si_sqtt_register_pipeline(struct si_context *sctx, struct si_sqtt_fake_pipeline *pipeline,
                               uint32_t *gfx_sh_offsets)
{
   assert(!si_sqtt_pipeline_is_registered(sctx->sqtt.data, pipeline->code_hash));

   bool result = ac_sqtt_add_pso_correlation(sctx->sqtt.data, pipeline->code_hash, pipeline->code_hash);
   if (!result)
      return false;

   result = ac_sqtt_add_code_object_loader_event(
      sctx->sqtt.data, pipeline->code_hash, pipeline->bo->gpu_address);
   if (!result)
      return false;

   return si_sqtt_add_code_object(sctx, pipeline, gfx_sh_offsets);
}

void si_sqtt_describe_pipeline_bind(struct si_context *sctx,
                                    uint64_t pipeline_hash,
                                    int bind_point)
{
   struct rgp_sqtt_marker_pipeline_bind marker = {0};
   struct radeon_cmdbuf *cs = &sctx->gfx_cs;

   if (likely(!si_is_sqtt_active(sctx)))
      return;

   marker.identifier = RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE;
   marker.cb_id = sctx->sqtt.cb_id;
   marker.bind_point = bind_point;
   marker.api_pso_hash[0] = pipeline_hash;
   marker.api_pso_hash[1] = pipeline_hash >> 32;

   si_emit_sqtt_userdata(sctx, cs, &marker, sizeof(marker) / 4);
}

void si_sqtt_describe_begin(struct si_context *sctx, struct radeon_cmdbuf *rcs)
{
   enum amd_ip_type ip_type = sctx->ws->cs_get_ip_type(rcs);
   
   if (!sctx->sqtt.data)
      return;

   unsigned bo_size = sctx->sqtt.timestamp.bo ? sctx->sqtt.timestamp.bo->bo_size : 0;

   if (sctx->sqtt.timestamp.offset + 2 * SI_SQTT_TIMESTAMP_SIZE > bo_size) {
      uint8_t *map;
      uint64_t new_size;
      new_size = MAX2(4096, 2 * bo_size);

      struct si_resource *ts_bo = NULL;
      ts_bo = si_aligned_buffer_create(&sctx->screen->b, SI_RESOURCE_FLAG_DRIVER_INTERNAL,
                                       PIPE_USAGE_STAGING, new_size, 4096);

      if (!ts_bo) {
         mesa_loge("Failed to create a timestamp buffer for SQTT.");
         goto fail;
      }

      map = si_buffer_map(sctx, ts_bo, PIPE_MAP_READ);
      if (!map) {
         si_resource_reference(&ts_bo, NULL);
         mesa_loge("Failed to map the timestamp buffer for SQTT.");
         goto fail;
      }

      if (sctx->sqtt.timestamp.bo) {
         struct si_sqtt_timestamp *new_timestamp;

         new_timestamp = malloc(sizeof(*new_timestamp));
         if (!new_timestamp) {
            si_resource_reference(&ts_bo, NULL);
            goto fail;
         }

         memcpy(new_timestamp, &sctx->sqtt.timestamp, sizeof(*new_timestamp));
         list_add(&new_timestamp->list, &sctx->sqtt.timestamp.list);
      }

      sctx->sqtt.timestamp.bo = ts_bo;
      sctx->sqtt.timestamp.offset = 0;
      sctx->sqtt.timestamp.map = map;
   }

   si_emit_ts(sctx, sctx->sqtt.timestamp.bo, sctx->sqtt.timestamp.offset);

   union rgp_sqtt_marker_cb_id cb_id = ac_sqtt_get_next_cmdbuf_id(sctx->sqtt.data, ip_type);
   sctx->sqtt.cb_id = cb_id.all;

   struct rgp_sqtt_marker_cb_start marker = {0};
   marker.identifier = RGP_SQTT_MARKER_IDENTIFIER_CB_START;
   marker.cb_id = sctx->sqtt.cb_id;
   marker.device_id_low = sctx->sqtt.device_id;
   marker.device_id_high = sctx->sqtt.device_id >> 32;
   marker.queue_flags = SI_SQTT_QUEUE_COMPUTE;
   if (sctx->is_gfx_queue)
      marker.queue_flags |= SI_SQTT_QUEUE_GRAPHICS;

   si_emit_sqtt_userdata(sctx, &sctx->gfx_cs, &marker, sizeof(marker) / 4);
   return;

fail:
   sctx->sqtt.cb_id = 0;
   return;
}

void si_sqtt_describe_flush(struct si_context *sctx)
{
   if (!sctx->sqtt.data || !sctx->sqtt.cb_id)
      return;

   struct rgp_queue_event *queue_event = &sctx->sqtt.data->rgp_queue_event;
   struct rgp_queue_event_record *record;

   unsigned pre_offset = sctx->sqtt.timestamp.offset;
   unsigned post_offset = pre_offset + SI_SQTT_TIMESTAMP_SIZE;

   sctx->sqtt.timestamp.offset = post_offset + SI_SQTT_TIMESTAMP_SIZE;
   si_emit_ts(sctx, sctx->sqtt.timestamp.bo, post_offset);

   struct rgp_sqtt_marker_cb_end marker = {0};
   marker.identifier = RGP_SQTT_MARKER_IDENTIFIER_CB_END;
   marker.cb_id = sctx->sqtt.cb_id;
   marker.device_id_low = sctx->sqtt.device_id;
   marker.device_id_high = sctx->sqtt.device_id >> 32;
   si_emit_sqtt_userdata(sctx, &sctx->gfx_cs, &marker, sizeof(marker) / 4);

   record = calloc(1, sizeof(*record));
   if (!record)
      return;

   simple_mtx_lock(&queue_event->lock);

   record->event_type = SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT;
   record->sqtt_cb_id = sctx->sqtt.cb_id;
   record->cpu_timestamp = os_time_get_nano();
   record->gpu_timestamps[0] = (uint64_t *)(sctx->sqtt.timestamp.map + pre_offset);
   record->gpu_timestamps[1] = (uint64_t *)(sctx->sqtt.timestamp.map + post_offset);
   record->frame_index = num_frames;
   record->api_id = queue_event->record_count;
   record->queue_info_index = 0;
   record->submit_sub_index = 0;

   list_addtail(&record->list, &queue_event->record);
   queue_event->record_count++;
   simple_mtx_unlock(&queue_event->lock);
}
