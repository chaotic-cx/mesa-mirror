/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 *
 * C interface for the DevDriver integration implementation.
 */

#include "ac_devdriver.h"
#include "ac_ubertrace_service.h"

#include "devDriverServer.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
/* driverControlProtocol.h uses constructor parameter names that shadow member fields. */
#include "protocols/driverControlServer.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include <cstdlib>
#include <cstring>
#include "ddNet.h"
#include "ddRpcServer.h"

using namespace DevDriver;

struct ac_devdriver_server {
   DevDriverServer *devdriver_server;
   AllocCb alloc_cb;

   /* UberTrace fields */
   DDRpcServer rpc_server;
   ac_ubertrace_service *ubertrace_service;
};

/* Default memory allocator */
static void *
DefaultAlloc(void *userdata, size_t size, size_t alignment, bool zero)
{
   (void)userdata;
   (void)alignment;

   void *memory = malloc(size);
   if (memory && zero) {
      memset(memory, 0, size);
   }

   return memory;
}

static void
DefaultFree(void *userdata, void *memory)
{
   (void)userdata;
   free(memory);
}

extern "C" {

/* Initialize the DevDriver server. Returns null on failure. */
ac_devdriver_server *
ac_devdriver_create(void)
{
   auto *server = new ac_devdriver_server();
   if (!server)
      return NULL;

   /* Initialize all fields */
   server->devdriver_server = NULL;
   server->rpc_server = NULL;
   server->ubertrace_service = NULL;

   /* Set up allocator */
   server->alloc_cb.pUserdata = NULL;
   server->alloc_cb.pfnAlloc = DefaultAlloc;
   server->alloc_cb.pfnFree = DefaultFree;

   /* Set up server creation info */
   ServerCreateInfo server_info = {};
   server_info.connectionInfo.type = TransportType::Local;
   /* Local transport uses Unix sockets, not hostname. */
   server_info.connectionInfo.pHostname = NULL;
   server_info.connectionInfo.port = 0;

   server_info.initialFlags = 0;
   server_info.componentType = Component::Driver;
   server_info.createUpdateThread = true;

   /* Always use "AMD Vulkan Driver" because RDP checks for allowed client descriptions. */
   Platform::Strncpy(server_info.clientDescription, "AMD Vulkan Driver",
                     sizeof(server_info.clientDescription));

   /* RGP server is not needed because tracing is handled via UberTrace RPC. */
   server_info.servers.rgp = 0;
   /* Required for RDP to detect the driver and halt or step it through init. */
   server_info.servers.driverControl = 1;
   server_info.servers.event = false;

   /* Create the DevDriver server */
   server->devdriver_server =
      DD_NEW(DevDriverServer, server->alloc_cb)(server->alloc_cb, server_info);
   if (!server->devdriver_server) {
      delete server;
      return NULL;
   }

   /* Initialize the server */
   Result result = server->devdriver_server->Initialize();
   if (result != Result::Success) {
      DD_DELETE(server->devdriver_server, server->alloc_cb);
      delete server;
      return NULL;
   }

   /* Create the RPC server and register the UberTrace service */
   IMsgChannel *msg_channel = server->devdriver_server->GetMessageChannel();
   if (msg_channel) {
      /* Create the RPC server */
      DDRpcServerCreateInfo rpc_create_info = {};
      rpc_create_info.hConnection = reinterpret_cast<DDNetConnection>(msg_channel);

      DD_RESULT rpc_result = ddRpcServerCreate(&rpc_create_info, &server->rpc_server);
      if (rpc_result == DD_RESULT_SUCCESS) {
         /* Create and register the UberTrace service */
         server->ubertrace_service = ac_ubertrace_service_create();
         if (server->ubertrace_service) {
            ac_ubertrace_register_service(server->ubertrace_service, server->rpc_server);
         } else {
            fprintf(stderr, "ac/rdp: Failed to create UberTrace service\n");
         }
      } else {
         fprintf(stderr, "ac/rdp: Failed to create RPC server for UberTrace\n");
      }
   }

   /* Attempt to locate a tool with DeveloperModeEnabled status flag.
    * If no tool is found, tear down the server.
    */
   if (msg_channel) {
      ClientId client_id = kBroadcastClientId;
      ClientMetadata filter = {};
      filter.clientType = Component::Tool;
      filter.status = static_cast<StatusFlags>(ClientStatusFlags::DeveloperModeEnabled);

      Result discover_result =
         msg_channel->FindFirstClient(filter, &client_id, kFindClientTimeout, &filter);
      if (discover_result != Result::Success) {
         fprintf(stderr, "ac/rdp: No tool found - tearing down DevDriver server\n");
         ac_devdriver_destroy(server);
         return NULL;
      }
   }

   /* Do not call any init functions here (StartEarlyDeviceInit, StartLateDeviceInit, Finalize).
    * They are called later during device initialization in the correct DevDriver sequence.
    */

   return server;
}

/* Destroy the DevDriver server. */
void
ac_devdriver_destroy(ac_devdriver_server *server)
{
   if (server) {
      /* Destroy the UberTrace service first. */
      if (server->ubertrace_service) {
         ac_ubertrace_service_destroy(server->ubertrace_service);
         server->ubertrace_service = NULL;
      }

      /* Destroy the RPC server. */
      if (server->rpc_server) {
         ddRpcServerDestroy(server->rpc_server);
         server->rpc_server = NULL;
      }

      /* Destroy the DevDriver server. */
      if (server->devdriver_server) {
         server->devdriver_server->Destroy();
         DD_DELETE(server->devdriver_server, server->alloc_cb);
      }
      delete server;
   }
}

/* Finalize the server (call after device initialization). */
void
ac_devdriver_finalize(ac_devdriver_server *server)
{
   if (server && server->devdriver_server) {
      server->devdriver_server->Finalize();
   }
}

/* ---- Driver Control API ---- */

/* Signal device init stages. */

void
ac_devdriver_start_early_device_init(ac_devdriver_server *server)
{
   if (server && server->devdriver_server) {
      auto *driver_control = server->devdriver_server->GetDriverControlServer();
      if (driver_control) {
         driver_control->StartEarlyDeviceInit();
      }
   }
}

void
ac_devdriver_start_late_device_init(ac_devdriver_server *server)
{
   if (server && server->devdriver_server) {
      auto *driver_control = server->devdriver_server->GetDriverControlServer();
      if (driver_control) {
         driver_control->StartLateDeviceInit();
      }
   }
}

void
ac_devdriver_finish_device_init(ac_devdriver_server *server)
{
   if (server && server->devdriver_server) {
      auto *driver_control = server->devdriver_server->GetDriverControlServer();
      if (driver_control) {
         driver_control->FinishDeviceInit();
         fprintf(stderr, "ac/rdp: Finished initialization - ready for tracing\n");
      }
   }
}

/* Check if the driver has completed initialization (reached Running state). */
bool
ac_devdriver_is_driver_initialized(ac_devdriver_server *server)
{
   if (server && server->devdriver_server) {
      auto *driver_control = server->devdriver_server->GetDriverControlServer();
      if (driver_control) {
         return driver_control->IsDriverInitialized();
      }
   }
   return false;
}

/* Set the number of GPUs managed by the driver. */
void
ac_devdriver_set_num_gpus(ac_devdriver_server *server, uint32_t num_gpus)
{
   if (server && server->devdriver_server) {
      auto *driver_control = server->devdriver_server->GetDriverControlServer();
      if (driver_control) {
         driver_control->SetNumGpus(num_gpus);
      }
   }
}

/* Called from render loop to handle driver stepping. */
void
ac_devdriver_driver_tick(ac_devdriver_server *server)
{
   if (server && server->devdriver_server) {
      auto *driver_control = server->devdriver_server->GetDriverControlServer();
      if (driver_control) {
         driver_control->DriverTick();
      }
   }
}

/* ---- UberTrace API ---- */

/* Check current UberTrace state. */
static enum ac_ubertrace_state
ac_devdriver_ubertrace_get_state(ac_devdriver_server *server)
{
   if (!server || !server->ubertrace_service) {
      return AC_UBERTRACE_STATE_DISABLED;
   }

   return ac_ubertrace_get_state(server->ubertrace_service);
}

/* Check if RDP requested a trace via UberTrace RPC. */
bool
ac_devdriver_ubertrace_is_trace_requested(ac_devdriver_server *server)
{
   return ac_devdriver_ubertrace_get_state(server) == AC_UBERTRACE_STATE_REQUESTED;
}

/* Submit completed trace data for UberTrace protocol.
 * Returns true on success. 'data' buffer ownership is transferred on success
 */
bool
ac_devdriver_ubertrace_set_trace_data(ac_devdriver_server *server, uint8_t *data, size_t size)
{
   if (server && server->ubertrace_service) {
      ac_ubertrace_set_trace_data(server->ubertrace_service, data, size);
      return true;
   }
   return false;
}

/* Get trace config values from the UberTrace service. These come from
 * RDP's ConfigureTraceParams JSON. Returns null if UberTrace is not active
 * or config is unavailable.
 */
const char *
ac_devdriver_ubertrace_get_trace_config_json(ac_devdriver_server *server)
{
   if (server && server->ubertrace_service) {
      const struct ac_ubertrace_trace_config *cfg =
         ac_ubertrace_get_trace_config(server->ubertrace_service);
      return (cfg && cfg->raw_json) ? cfg->raw_json : NULL;
   }
   return NULL;
}

/* Get instruction tokens setting from UberTrace trace config.
 * Defaults to enabled if unavailable.
 */
bool
ac_devdriver_ubertrace_get_instruction_tokens_enabled(ac_devdriver_server *server)
{
   if (server && server->ubertrace_service) {
      const struct ac_ubertrace_trace_config *cfg =
         ac_ubertrace_get_trace_config(server->ubertrace_service);
      return cfg ? cfg->instruction_tokens_enabled : true;
   }
   return true;
}

/* Get SE mask from UberTrace trace config. In RDP convention, set bits enable SEs, but 0 = all
 * enabled. Translate to the usual bitmask format (0 = all disabled). Return 0xFFFFFFFF (= all SEs
 * enabled) if unavailable.
 */
uint32_t
ac_devdriver_ubertrace_get_se_mask(ac_devdriver_server *server)
{
   if (server && server->ubertrace_service) {
      const struct ac_ubertrace_trace_config *cfg =
         ac_ubertrace_get_trace_config(server->ubertrace_service);
      return (cfg && cfg->se_mask) ? cfg->se_mask : ~0u;
   }
   return ~0u;
}

/* Get RDF chunk mask from UberTrace trace config.
 * Returns RDF_CHUNK_MASK_ALL if unavailable (emit all chunks).
 */
uint32_t
ac_devdriver_ubertrace_get_rdf_chunk_mask(ac_devdriver_server *server)
{
   if (server && server->ubertrace_service) {
      const struct ac_ubertrace_trace_config *cfg =
         ac_ubertrace_get_trace_config(server->ubertrace_service);
      return cfg ? cfg->rdf_chunk_mask : RDF_CHUNK_MASK_ALL;
   }
   return RDF_CHUNK_MASK_ALL;
}

/* Get per-source Zstd compression mask (same bit layout as chunk mask).
 * Returns 0 if unavailable (no compression).
 */
uint32_t
ac_devdriver_ubertrace_get_compression_mask(ac_devdriver_server *server)
{
   if (server && server->ubertrace_service) {
      const struct ac_ubertrace_trace_config *cfg =
         ac_ubertrace_get_trace_config(server->ubertrace_service);
      return cfg ? cfg->compression_mask : 0;
   }
   return 0;
}

/* Get target capture frame from UberTrace trace config.
 * Returns preparation_start_index + num_prep_frames.
 * 0 means "capture immediately" (default / capture-now).
 */
uint32_t
ac_devdriver_ubertrace_get_target_capture_frame(ac_devdriver_server *server)
{
   if (server && server->ubertrace_service) {
      const struct ac_ubertrace_trace_config *cfg =
         ac_ubertrace_get_trace_config(server->ubertrace_service);
      /* Only absolute mode has a real target frame.
       * Relative mode ("capture now") always returns 0 = immediate. */
      if (cfg && cfg->capture_mode == AC_UBERTRACE_CAPTURE_MODE_ABSOLUTE)
         return cfg->preparation_start_index + cfg->num_prep_frames;
   }
   return 0;
}

} /* extern "C" */
