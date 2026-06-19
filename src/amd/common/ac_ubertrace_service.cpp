/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 *
 * UberTrace Service implementation.
 */

#include "ac_ubertrace_service.h"
#include <ddRpcServer.h>

#include "util/parson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* UberTrace service identifiers - must match what RDP expects */
#define AC_UBERTRACE_SERVICE_ID            0x63727461
#define AC_UBERTRACE_SERVICE_VERSION_MAJOR 0
#define AC_UBERTRACE_SERVICE_VERSION_MINOR 2
#define AC_UBERTRACE_SERVICE_VERSION_PATCH 0

/* Function IDs */
#define AC_UBERTRACE_FUNC_ID_ENABLE_TRACING         0x1
#define AC_UBERTRACE_FUNC_ID_QUERY_TRACE_PARAMS     0x2
#define AC_UBERTRACE_FUNC_ID_CONFIGURE_TRACE_PARAMS 0x3
#define AC_UBERTRACE_FUNC_ID_REQUEST_TRACE          0x4
#define AC_UBERTRACE_FUNC_ID_CANCEL_TRACE           0x5
#define AC_UBERTRACE_FUNC_ID_COLLECT_TRACE          0x6

/* Service info structure - matches DevDriver's expected format */
static DDRpcServerRegisterServiceInfo ubertrace_service_info = {
   .id = AC_UBERTRACE_SERVICE_ID,
   .version =
      {
         .major = AC_UBERTRACE_SERVICE_VERSION_MAJOR,
         .minor = AC_UBERTRACE_SERVICE_VERSION_MINOR,
         .patch = AC_UBERTRACE_SERVICE_VERSION_PATCH,
      },
   .pName = "UberTrace",
   .pDescription = "A service that provides generic trace functionality",
};

/* ---- Service Implementation Functions ---- */

static DD_RESULT
EnableTracingHandler(const DDRpcServerCallInfo *pCall)
{
   ac_ubertrace_service *service = (ac_ubertrace_service *)pCall->pUserdata;

   if (!service)
      return DD_RESULT_DD_GENERIC_INVALID_PARAMETER;

   service->tracing_enabled = true;

   return DD_RESULT_SUCCESS;
}

static DD_RESULT
QueryTraceParamsHandler(const DDRpcServerCallInfo *pCall)
{
   /* Not needed for now */
   (void)pCall;
   return DD_RESULT_SUCCESS;
}

static DD_RESULT
ConfigureTraceParamsHandler(const DDRpcServerCallInfo *pCall)
{
   ac_ubertrace_service *service = (ac_ubertrace_service *)pCall->pUserdata;

   if (!service)
      return DD_RESULT_DD_GENERIC_INVALID_PARAMETER;

   if (!pCall->pParameterData || pCall->parameterDataSize == 0)
      return DD_RESULT_SUCCESS;

   /* Store raw JSON for TraceConfig chunk emission */
   pthread_mutex_lock(&service->mutex);

   free(service->trace_config.raw_json);
   service->trace_config.raw_json = (char *)malloc(pCall->parameterDataSize + 1);
   if (service->trace_config.raw_json) {
      memcpy(service->trace_config.raw_json, pCall->pParameterData, pCall->parameterDataSize);
      service->trace_config.raw_json[pCall->parameterDataSize] = '\0';
      service->trace_config.raw_json_size = pCall->parameterDataSize;
   }

   /* Parse JSON config to extract trace parameters */
   JSON_Value *root = json_parse_string(service->trace_config.raw_json);
   if (root) {
      JSON_Object *root_obj = json_value_get_object(root);

      /* --- Controller config --- */
      JSON_Object *controller = json_object_get_object(root_obj, "controller");
      if (controller) {
         JSON_Object *ctrl_config = json_object_get_object(controller, "config");
         if (ctrl_config) {
            /* TODO: Apply these - for now driver always captures exactly 1 frame with no prep-frame
             * logic */
            double capture_frame_count = json_object_get_number(ctrl_config, "captureFrameCount");
            if (capture_frame_count > 0)
               service->trace_config.capture_frame_count = (uint32_t)capture_frame_count;

            double num_prep_frames = json_object_get_number(ctrl_config, "numPrepFrames");
            if (num_prep_frames > 0)
               service->trace_config.num_prep_frames = (uint32_t)num_prep_frames;

            double preparation_start_index =
               json_object_get_number(ctrl_config, "preparationStartIndex");
            if (preparation_start_index >= 0)
               service->trace_config.preparation_start_index = (uint32_t)preparation_start_index;

            const char *capture_mode = json_object_get_string(ctrl_config, "captureMode");
            if (capture_mode && strcmp(capture_mode, "relative") == 0)
               service->trace_config.capture_mode = AC_UBERTRACE_CAPTURE_MODE_RELATIVE;
            else if (capture_mode && strcmp(capture_mode, "absolute") == 0)
               service->trace_config.capture_mode = AC_UBERTRACE_CAPTURE_MODE_ABSOLUTE;
         }
      }

      /* --- Sources array --- */
      JSON_Array *sources = json_object_get_array(root_obj, "sources");
      if (sources) {
         service->trace_config.rdf_chunk_mask = 0;
         service->trace_config.compression_mask = 0;

         size_t num_sources = json_array_get_count(sources);
         for (size_t i = 0; i < num_sources; i++) {
            JSON_Object *source = json_array_get_object(sources, i);
            const char *name = json_object_get_string(source, "name");
            if (!name)
               continue;

            JSON_Object *source_config = json_object_get_object(source, "config");

            /* Map source names to RDF chunk mask bits */
            uint32_t chunk_mask = 0;
            if (strcmp(name, "asicinfo") == 0) {
               chunk_mask = RDF_CHUNK_MASK_ASIC_INFO;
            } else if (strcmp(name, "apiinfo") == 0) {
               chunk_mask = RDF_CHUNK_MASK_API_INFO;
            } else if (strcmp(name, "clockcalibration") == 0) {
               chunk_mask = RDF_CHUNK_MASK_CLOCK_CALIBRATION;
            } else if (strcmp(name, "codeobject") == 0) {
               chunk_mask = RDF_CHUNK_MASK_CODE_OBJECT;
            } else if (strcmp(name, "queuetimings") == 0) {
               chunk_mask = RDF_CHUNK_MASK_QUEUE_TIMINGS;
            } else if (strcmp(name, "traceconfig") == 0) {
               chunk_mask = RDF_CHUNK_MASK_TRACE_CONFIG;
            } else if (strcmp(name, "gpuperfexp") == 0) {
               chunk_mask = RDF_CHUNK_MASK_SQTT_DATA;

               /* --- SQTT config --- */
               JSON_Object *sqtt_config =
                  source_config ? json_object_get_object(source_config, "sqtt") : NULL;
               if (sqtt_config) {
                  if (json_object_has_value(sqtt_config, "enabled"))
                     service->trace_config.sqtt_enabled =
                        (bool)json_object_get_boolean(sqtt_config, "enabled");

                  double sqtt_memory_limit_mb =
                     json_object_get_number(sqtt_config, "memoryLimitInMb");
                  if (sqtt_memory_limit_mb > 0)
                     service->trace_config.sqtt_memory_limit_mb = (uint32_t)sqtt_memory_limit_mb;

                  if (json_object_has_value(sqtt_config, "enableInstructionTokens"))
                     service->trace_config.instruction_tokens_enabled =
                        (bool)json_object_get_boolean(sqtt_config, "enableInstructionTokens");

                  double se_mask = json_object_get_number(sqtt_config, "seMask");
                  if (se_mask > 0)
                     service->trace_config.se_mask = (uint32_t)se_mask;
               }

               /* --- SPM config --- */
               JSON_Object *spm_config =
                  source_config ? json_object_get_object(source_config, "spm") : NULL;
               if (spm_config) {
                  if (json_object_has_value(spm_config, "enabled"))
                     service->trace_config.spm_enabled =
                        (bool)json_object_get_boolean(spm_config, "enabled");

                  double spm_memory_limit_mb =
                     json_object_get_number(spm_config, "memoryLimitInMb");
                  if (spm_memory_limit_mb > 0)
                     service->trace_config.spm_memory_limit_mb = (uint32_t)spm_memory_limit_mb;

                  double spm_sample_frequency =
                     json_object_get_number(spm_config, "sampleFrequency");
                  if (spm_sample_frequency > 0)
                     service->trace_config.spm_sample_frequency = (uint32_t)spm_sample_frequency;

                  /* TODO: Parse perfCounters array - currently counters are hardcoded
                   * per GFX gen in ac_spm.c.
                   * JSON_Array *perf_counters =
                   *    json_object_get_array(spm_config, "perfCounters"); */
               }

               /* Only emit SPM data when the client requested it. For now, the SPM BO
                * is allocated at device init (gated by RADV_THREAD_TRACE_CACHE_COUNTERS
                * and GFX level), so this mask bit is only the control point for whether
                * captured SPM samples are written into the RDF stream. */
               if (service->trace_config.spm_enabled)
                  chunk_mask |= RDF_CHUNK_MASK_SPM_DATA;
            } else {
               fprintf(stderr, "ac/rdp: Unknown source name in received config: \"%s\" (ignored)\n",
                       name);
               continue;
            }

            service->trace_config.rdf_chunk_mask |= chunk_mask;
            if (source_config && json_object_has_value(source_config, "useCompression") &&
                json_object_get_boolean(source_config, "useCompression"))
               service->trace_config.compression_mask |= chunk_mask;
         }
      }

      json_value_free(root);
   }

   pthread_mutex_unlock(&service->mutex);

   return DD_RESULT_SUCCESS;
}

static DD_RESULT
RequestTraceHandler(const DDRpcServerCallInfo *pCall)
{
   ac_ubertrace_service *service = (ac_ubertrace_service *)pCall->pUserdata;

   if (!service)
      return DD_RESULT_DD_GENERIC_INVALID_PARAMETER;

   if (!service->tracing_enabled) {
      fprintf(stderr, "ac/rdp: RequestTrace rejected - tracing not enabled\n");
      return DD_RESULT_DD_GENERIC_UNAVAILABLE;
   }

   if (service->state != AC_UBERTRACE_STATE_READY) {
      fprintf(stderr, "ac/rdp: RequestTrace rejected - invalid state %d (not READY)\n",
              service->state);
      return DD_RESULT_DD_GENERIC_REJECTED;
   }

   service->state = AC_UBERTRACE_STATE_REQUESTED;

   return DD_RESULT_SUCCESS;
}

static DD_RESULT
CancelTraceHandler(const DDRpcServerCallInfo *pCall)
{
   ac_ubertrace_service *service = (ac_ubertrace_service *)pCall->pUserdata;

   if (!service)
      return DD_RESULT_DD_GENERIC_INVALID_PARAMETER;

   pthread_mutex_lock(&service->mutex);

   /* Free any pending trace data */
   if (service->trace_data) {
      free(service->trace_data);
      service->trace_data = NULL;
      service->trace_data_size = 0;
   }

   service->state = AC_UBERTRACE_STATE_READY;

   pthread_mutex_unlock(&service->mutex);
   return DD_RESULT_SUCCESS;
}

static DD_RESULT
CollectTraceHandler(const DDRpcServerCallInfo *pCall)
{
   ac_ubertrace_service *service = (ac_ubertrace_service *)pCall->pUserdata;

   if (!service || !pCall->pWriter) {
      fprintf(stderr, "ac/rdp: CollectTrace rejected - invalid parameter\n");
      return DD_RESULT_DD_GENERIC_INVALID_PARAMETER;
   }

   pthread_mutex_lock(&service->mutex);

   if (service->state != AC_UBERTRACE_STATE_COMPLETE) {
      pthread_mutex_unlock(&service->mutex);
      return DD_RESULT_DD_GENERIC_NOT_READY;
   }

   if (!service->trace_data || service->trace_data_size == 0) {
      pthread_mutex_unlock(&service->mutex);
      return DD_RESULT_DD_GENERIC_NOT_READY;
   }

   /* Take ownership of trace data and change service fields before the network transfer to avoid
    * doing it under the lock. */
   uint8_t *data = service->trace_data;
   size_t data_size = service->trace_data_size;
   service->trace_data = NULL;
   service->trace_data_size = 0;
   service->state = AC_UBERTRACE_STATE_READY;

   pthread_mutex_unlock(&service->mutex);

   /* Write trace data using the provided writer */
   const DDByteWriter *writer = pCall->pWriter;
   DD_RESULT result = writer->pfnBegin(writer->pUserdata, &data_size);

   if (result == DD_RESULT_SUCCESS) {
      result = writer->pfnWriteBytes(writer->pUserdata, data, data_size);
      writer->pfnEnd(writer->pUserdata, result);
   }

   if (result == DD_RESULT_SUCCESS) {
      fprintf(stderr, "ac/rdp: Trace transfer successful\n");
   } else {
      fprintf(stderr, "ac/rdp: Trace transfer failed (result=%d)\n", result);
   }

   free(data);
   return result;
}

/* ---- RPC handlers registration ---- */

static DD_RESULT
RegisterFunctions(DDRpcServer server, ac_ubertrace_service *service)
{
   DD_RESULT result = DD_RESULT_SUCCESS;

   /* EnableTracing */
   if (result == DD_RESULT_SUCCESS) {
      DDRpcServerRegisterFunctionInfo info = {0};
      info.serviceId = AC_UBERTRACE_SERVICE_ID;
      info.id = AC_UBERTRACE_FUNC_ID_ENABLE_TRACING;
      info.pName = "EnableTracing";
      info.pDescription = "Attempts to enable tracing";
      info.pFuncUserdata = service;
      info.pfnFuncCb = EnableTracingHandler;
      result = ddRpcServerRegisterFunction(server, &info);
   }

   /* QueryTraceParams */
   if (result == DD_RESULT_SUCCESS) {
      DDRpcServerRegisterFunctionInfo info = {0};
      info.serviceId = AC_UBERTRACE_SERVICE_ID;
      info.id = AC_UBERTRACE_FUNC_ID_QUERY_TRACE_PARAMS;
      info.pName = "QueryTraceParams";
      info.pDescription = "Queries the current set of trace parameters";
      info.pFuncUserdata = service;
      info.pfnFuncCb = QueryTraceParamsHandler;
      result = ddRpcServerRegisterFunction(server, &info);
   }

   /* ConfigureTraceParams */
   if (result == DD_RESULT_SUCCESS) {
      DDRpcServerRegisterFunctionInfo info = {0};
      info.serviceId = AC_UBERTRACE_SERVICE_ID;
      info.id = AC_UBERTRACE_FUNC_ID_CONFIGURE_TRACE_PARAMS;
      info.pName = "ConfigureTraceParams";
      info.pDescription = "Configures the current set of trace parameters";
      info.pFuncUserdata = service;
      info.pfnFuncCb = ConfigureTraceParamsHandler;
      result = ddRpcServerRegisterFunction(server, &info);
   }

   /* RequestTrace */
   if (result == DD_RESULT_SUCCESS) {
      DDRpcServerRegisterFunctionInfo info = {0};
      info.serviceId = AC_UBERTRACE_SERVICE_ID;
      info.id = AC_UBERTRACE_FUNC_ID_REQUEST_TRACE;
      info.pName = "RequestTrace";
      info.pDescription = "Requests execution of a trace";
      info.pFuncUserdata = service;
      info.pfnFuncCb = RequestTraceHandler;
      result = ddRpcServerRegisterFunction(server, &info);
   }

   /* CancelTrace */
   if (result == DD_RESULT_SUCCESS) {
      DDRpcServerRegisterFunctionInfo info = {0};
      info.serviceId = AC_UBERTRACE_SERVICE_ID;
      info.id = AC_UBERTRACE_FUNC_ID_CANCEL_TRACE;
      info.pName = "CancelTrace";
      info.pDescription = "Cancels a previously requested trace";
      info.pFuncUserdata = service;
      info.pfnFuncCb = CancelTraceHandler;
      result = ddRpcServerRegisterFunction(server, &info);
   }

   /* CollectTrace */
   if (result == DD_RESULT_SUCCESS) {
      DDRpcServerRegisterFunctionInfo info = {0};
      info.serviceId = AC_UBERTRACE_SERVICE_ID;
      info.id = AC_UBERTRACE_FUNC_ID_COLLECT_TRACE;
      info.pName = "CollectTrace";
      info.pDescription = "Collects the data created by a previously executed trace";
      info.pFuncUserdata = service;
      info.pfnFuncCb = CollectTraceHandler;
      result = ddRpcServerRegisterFunction(server, &info);
   }

   return result;
}

/* ---- Public API Implementation ---- */

ac_ubertrace_service *
ac_ubertrace_service_create(void)
{
   ac_ubertrace_service *service = (ac_ubertrace_service *)calloc(1, sizeof(ac_ubertrace_service));

   if (service) {
      service->state = AC_UBERTRACE_STATE_READY;
      service->trace_data = NULL;
      service->trace_data_size = 0;
      service->tracing_enabled = false;
      pthread_mutex_init(&service->mutex, NULL);
      /* Default trace config - matches what the direct .rgp dump path uses:
       * instruction timing enabled, SPM enabled (on GFX10+), se_mask=0 (all SEs),
       * 32MB buffers, 4096 sclk sample interval. */
      service->trace_config.sqtt_enabled = true;
      service->trace_config.sqtt_memory_limit_mb = 32;
      service->trace_config.instruction_tokens_enabled = true;
      service->trace_config.se_mask = 0;
      service->trace_config.spm_enabled = true;
      service->trace_config.spm_memory_limit_mb = 32;
      service->trace_config.spm_sample_frequency = 4096;
      service->trace_config.capture_frame_count = 1;
      service->trace_config.num_prep_frames = 0;
      service->trace_config.preparation_start_index = 0;
      service->trace_config.capture_mode = AC_UBERTRACE_CAPTURE_MODE_RELATIVE;
      service->trace_config.rdf_chunk_mask = RDF_CHUNK_MASK_ALL;
      service->trace_config.raw_json = NULL;
      service->trace_config.raw_json_size = 0;
   }

   return service;
}

void
ac_ubertrace_service_destroy(ac_ubertrace_service *service)
{
   if (service) {
      if (service->trace_data) {
         free(service->trace_data);
      }
      free(service->trace_config.raw_json);
      pthread_mutex_destroy(&service->mutex);
      free(service);
   }
}

DD_RESULT
ac_ubertrace_register_service(ac_ubertrace_service *service, DDRpcServer server)
{
   if (!server || !service)
      return DD_RESULT_DD_GENERIC_INVALID_PARAMETER;

   /* Register the service */
   DD_RESULT result = ddRpcServerRegisterService(server, &ubertrace_service_info);

   /* Register individual functions */
   if (result == DD_RESULT_SUCCESS) {
      result = RegisterFunctions(server, service);

      if (result != DD_RESULT_SUCCESS) {
         /* Unregister service if function registration fails */
         ddRpcServerUnregisterService(server, AC_UBERTRACE_SERVICE_ID);
      }
   }

   return result;
}

enum ac_ubertrace_state
ac_ubertrace_get_state(ac_ubertrace_service *service)
{
   if (!service)
      return AC_UBERTRACE_STATE_DISABLED;
   return service->state;
}

void
ac_ubertrace_set_trace_data(ac_ubertrace_service *service, uint8_t *data, size_t size)
{
   if (!service)
      return;

   pthread_mutex_lock(&service->mutex);

   /* Free any existing data */
   if (service->trace_data) {
      free(service->trace_data);
   }

   service->trace_data = data;
   service->trace_data_size = size;
   service->state = AC_UBERTRACE_STATE_COMPLETE;

   pthread_mutex_unlock(&service->mutex);
}

const struct ac_ubertrace_trace_config *
ac_ubertrace_get_trace_config(ac_ubertrace_service *service)
{
   if (!service)
      return NULL;
   return &service->trace_config;
}
