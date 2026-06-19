/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 *
 * C interface for the DevDriver integration.
 *
 * Exposes the C-callable wrapper that lets the driver's C code interact with
 * the underlying C++ DevDriver implementation. DevDriver is used for
 * communication with Radeon Developer Panel (RDP) over the UberTrace RPC
 * protocol.
 *
 * The DevDriver library is imported from GPUOpen without modifications.
 */

#ifndef AC_DEVDRIVER_H
#define AC_DEVDRIVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to the DevDriver server */
typedef struct ac_devdriver_server ac_devdriver_server;

ac_devdriver_server *ac_devdriver_create(void);

void ac_devdriver_destroy(ac_devdriver_server *server);

void ac_devdriver_finalize(ac_devdriver_server *server);

/* ---- UberTrace ---- */

bool ac_devdriver_ubertrace_is_trace_requested(ac_devdriver_server *server);

bool ac_devdriver_ubertrace_set_trace_data(ac_devdriver_server *server, uint8_t *data, size_t size);

const char *ac_devdriver_ubertrace_get_trace_config_json(ac_devdriver_server *server);

bool ac_devdriver_ubertrace_get_instruction_tokens_enabled(ac_devdriver_server *server);

uint32_t ac_devdriver_ubertrace_get_se_mask(ac_devdriver_server *server);

uint32_t ac_devdriver_ubertrace_get_rdf_chunk_mask(ac_devdriver_server *server);

uint32_t ac_devdriver_ubertrace_get_compression_mask(ac_devdriver_server *server);

uint32_t ac_devdriver_ubertrace_get_target_capture_frame(ac_devdriver_server *server);

/* ---- Driver Control API ---- */
void ac_devdriver_start_early_device_init(ac_devdriver_server *server);

void ac_devdriver_start_late_device_init(ac_devdriver_server *server);

void ac_devdriver_finish_device_init(ac_devdriver_server *server);

bool ac_devdriver_is_driver_initialized(ac_devdriver_server *server);

void ac_devdriver_set_num_gpus(ac_devdriver_server *server, uint32_t num_gpus);

void ac_devdriver_driver_tick(ac_devdriver_server *server);

#ifdef __cplusplus
}
#endif

#endif /* AC_DEVDRIVER_H */
