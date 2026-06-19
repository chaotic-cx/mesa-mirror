/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 *
 * UberTrace Service
 *
 * UberTrace RPC service is registered with DevDriver's RPC
 * server and receives RPC calls from tools like RDP and
 * forwards them to the driver's SQTT infrastructure.
 *
 * UberTrace Protocol Flow:
 *   1. Tool calls EnableTracing() to prepare the driver
 *   2. Tool calls ConfigureTraceParams() with JSON config
 *   3. Tool calls RequestTrace() to request a trace
 *   4. Driver executes the trace (on desired frame submission)
 *   5. Tool calls CollectTrace() to retrieve the trace data
 *   6. Tool calls CancelTrace() if needed
 */

#ifndef AC_UBERTRACE_SERVICE_H
#define AC_UBERTRACE_SERVICE_H

#include <ddRpcServer.h>
#include <pthread.h>
#include "ac_devdriver.h"
#include "ac_rdf_rgp_chunks.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Capture mode from RDP JSON "captureMode" field */
enum ac_ubertrace_capture_mode {
   AC_UBERTRACE_CAPTURE_MODE_RELATIVE = 0, /* "relative" - capture now (default) */
   AC_UBERTRACE_CAPTURE_MODE_ABSOLUTE,     /* "absolute" - frame index from app start */
};

/* Parsed trace configuration from RDP */
struct ac_ubertrace_trace_config {
   /* --- SQTT config (from sources[gpuperfexp].config.sqtt) --- */
   /* TODO: Apply sqtt_enabled - currently SQTT is always on during a trace. */
   bool sqtt_enabled;               /* Whether SQTT source is enabled (default: true) */
   /* TODO: Apply sqtt_memory_limit_mb before capture. */
   uint32_t sqtt_memory_limit_mb;   /* SQTT buffer size in MB (default: 32) */
   bool instruction_tokens_enabled; /* Enable instruction-level tokens */
   uint32_t se_mask;                /* Instruction timing shader engine mask
                                     * (bit set = SE enabled, but 0 = all) */

   /* --- SPM config (from sources[gpuperfexp].config.spm) --- */
   /* For now, it gates the RDF_CHUNK_MASK_SPM_DATA bit so SPM samples are only written when
    * the client requests them. The SPM BO is still allocated at device init.
    * TODO: Apply spm_enabled to conditionally init and allocate SPM */
   bool spm_enabled;              /* Whether SPM is enabled (default: true) */
   /* TODO: Apply spm_memory_limit_mb before capture */
   uint32_t spm_memory_limit_mb;  /* SPM buffer size in MB (default: 32) */
   /* TODO: Apply spm_sample_frequency before capture */
   uint32_t spm_sample_frequency; /* SPM sample interval in sclk (default: 4096) */

   /* TODO: Parse and apply perfCounters from RDP config for runtime counter selection.
    *       Currently counters are hardcoded per GFX gen in ac_spm.c.
    */

   /* --- Controller config (from controller.config) --- */
   /* TODO: Apply capture_frame_count / num_prep_frames.
    *       The driver always captures exactly 1 frame with no prep-frame logic. */
   uint32_t capture_frame_count;     /* Number of frames to capture (default: 1) */
   uint32_t num_prep_frames;         /* Preparation frames before capture (default: 0) */
   uint32_t preparation_start_index; /* 0-indexed frame offset for prep start (default: 0) */
   enum ac_ubertrace_capture_mode capture_mode; /* "relative" or "absolute" (default: relative) */

   /* --- RDF chunk/compression control (built from sources[]) --- */
   uint32_t rdf_chunk_mask;   /* Bitmask of which RDF chunks to emit */
   uint32_t compression_mask; /* Bitmask of which chunk groups use Zstd (default: 0) */

   /* --- Raw JSON for passthrough into the TraceConfig RDF chunk --- */
   char *raw_json;       /* Raw JSON string for TraceConfig chunk */
   size_t raw_json_size; /* Size of raw JSON (excluding null) */
};

enum ac_ubertrace_state {
   AC_UBERTRACE_STATE_DISABLED,
   AC_UBERTRACE_STATE_READY,
   AC_UBERTRACE_STATE_REQUESTED,
   AC_UBERTRACE_STATE_RUNNING, /* Currently not used */
   AC_UBERTRACE_STATE_COMPLETE
};

/* UberTrace service context */
struct ac_ubertrace_service {
   bool tracing_enabled;
   enum ac_ubertrace_state state;
   uint8_t *trace_data;
   size_t trace_data_size;
   struct ac_ubertrace_trace_config trace_config;
   /* Synchronization for state/data accessed from multiple threads */
   pthread_mutex_t mutex;
};

typedef struct ac_ubertrace_service ac_ubertrace_service;

/* Initialize the UberTrace service */
ac_ubertrace_service *ac_ubertrace_service_create(void);

/* Destroy the UberTrace service */
void ac_ubertrace_service_destroy(ac_ubertrace_service *service);

/* Register the UberTrace service with an RPC server */
DD_RESULT ac_ubertrace_register_service(ac_ubertrace_service *service, DDRpcServer server);

/* Get current trace state */
enum ac_ubertrace_state ac_ubertrace_get_state(ac_ubertrace_service *service);

/* Called by driver when trace data is ready (after capture completes) */
void ac_ubertrace_set_trace_data(ac_ubertrace_service *service, uint8_t *data, size_t size);

/* Get parsed trace configuration */
const struct ac_ubertrace_trace_config *
ac_ubertrace_get_trace_config(ac_ubertrace_service *service);

#ifdef __cplusplus
}
#endif

#endif /* AC_UBERTRACE_SERVICE_H */
