/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 *
 * RDF RGP Trace Writer - converts SQTT trace to RDF (Radeon Data Format)
 * format. This format is expected by RDP when using the UberTrace
 * connection protocol.
 *
 * RDF vs Legacy SQTT Format:
 *   - Legacy SQTT: Sequential chunks with sqtt_file_header, sqtt_file_chunk_*.
 *   - RDF: Named chunks with header/data, orderless, indexed at end of file,
 *          optional compression.
 *
 * Required RDF chunks for RDP:
 *   - "AsicInfo"          - GPU hardware information
 *   - "ApiInfo"           - API (Vulkan) information
 *   - "ClockCalibration"  - CPU/GPU clock synchronization
 *   - "CodeObject"        - Shader ELF objects
 *   - "QueueTimings"      - Queue submission timings
 *   - "SqttData"          - Raw SQTT data per shader engine
 *   - "TraceConfig"       - Trace configuration
 */

#ifndef AC_RDF_RGP_WRITER_H
#define AC_RDF_RGP_WRITER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct radeon_info;
struct ac_sqtt_trace;
struct ac_spm_trace;

/* Convert SQTT trace and other data to RDF format.
 * The returned buffer contains the encoded RDF trace and must be freed by the
 * caller. Returns NULL on failure.
 */
void *ac_sqtt_to_rdf(const struct radeon_info *info, const struct ac_sqtt_trace *sqtt_trace,
                     const struct ac_spm_trace *spm_trace, bool instruction_timing_enabled,
                     uint32_t buffer_size, const char *trace_config_json, uint32_t chunk_mask,
                     uint32_t compression_mask, size_t *out_size);

#ifdef __cplusplus
}
#endif

#endif /* AC_RDF_RGP_WRITER_H */
