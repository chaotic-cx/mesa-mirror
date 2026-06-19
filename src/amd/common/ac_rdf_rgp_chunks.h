/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 *
 * RDF RGP Chunk Definitions for RDP/RGP integration.
 * They must match interface-defined definitions exactly.
 */

#ifndef AC_RDF_RGP_CHUNKS_H
#define AC_RDF_RGP_CHUNKS_H

#include <assert.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Constants ---- */

/* Maximum GPU name length */
#define RDF_GPU_NAME_MAX_SIZE 256

/* Maximum shader engines supported */
#define RDF_MAX_NUM_SE 32
#define RDF_SA_PER_SE  2

/* Bits per shader engine for packer mask */
#define RDF_BITS_PER_SHADER_ENGINE 4

/* ---- Chunk identifiers ---- */

/* Chunk type identifier strings.
 * Note: "ClockCalibration" is exactly 16 chars (the identifier field size).
 * Assigning these via a designated initializer (.identifier = ...) is safe:
 * C drops the terminating null when the literal exactly fills the array.
 */
#define RDF_CHUNK_ID_ASIC_INFO       "AsicInfo"
#define RDF_CHUNK_ID_API_INFO        "ApiInfo"
#define RDF_CHUNK_ID_CLOCK_CALIB     "ClockCalibration"
#define RDF_CHUNK_ID_SQTT_DATA       "SqttData"
#define RDF_CHUNK_ID_CODE_OBJECT     "CodeObject"
#define RDF_CHUNK_ID_CO_LOAD_EVENT   "COLoadEvent"
#define RDF_CHUNK_ID_PSO_CORRELATION "PsoCorrelation"
#define RDF_CHUNK_ID_QUEUE_INFO      "QueueInfo"
#define RDF_CHUNK_ID_QUEUE_EVENT     "QueueEvent"
#define RDF_CHUNK_ID_TRACE_CONFIG    "TraceConfig"
#define RDF_CHUNK_ID_SPM_SESSION     "SpmSession"
#define RDF_CHUNK_ID_SPM_COUNTER     "SpmCounterData"

/* Chunk versions */
#define RDF_ASIC_INFO_VERSION       3
#define RDF_API_INFO_VERSION        2
#define RDF_CLOCK_CALIB_VERSION     2
#define RDF_SQTT_DATA_VERSION       5
#define RDF_CODE_OBJECT_VERSION     2
#define RDF_CO_LOAD_EVENT_VERSION   3
#define RDF_PSO_CORRELATION_VERSION 3
#define RDF_QUEUE_INFO_VERSION      1
#define RDF_QUEUE_EVENT_VERSION     1
#define RDF_TRACE_CONFIG_VERSION    1
#define RDF_SPM_SESSION_VERSION     2
#define RDF_SPM_COUNTER_VERSION     2

/* Chunk bitmasks */
/* Bitmask values for rdf_chunk_mask (driver internal) - controls which RDF chunks are emitted.
 * Built from the "sources" array names in the TraceConfig JSON.
 */
#define RDF_CHUNK_MASK_ASIC_INFO         (1u << 0)
#define RDF_CHUNK_MASK_API_INFO          (1u << 1)
#define RDF_CHUNK_MASK_CLOCK_CALIBRATION (1u << 2)
#define RDF_CHUNK_MASK_CODE_OBJECT       (1u << 3)
#define RDF_CHUNK_MASK_SQTT_DATA         (1u << 4) /* from "gpuperfexp" source */
#define RDF_CHUNK_MASK_SPM_DATA          (1u << 5) /* from "gpuperfexp" source */
#define RDF_CHUNK_MASK_QUEUE_TIMINGS     (1u << 6)
#define RDF_CHUNK_MASK_TRACE_CONFIG      (1u << 7)
#define RDF_CHUNK_MASK_ALL               0xFFFFFFFFu /* default: emit everything */

/* ---- Enumerations ---- */

/* GPU types */
enum rdf_gpu_type {
   RDF_GPU_TYPE_UNKNOWN = 0,
   RDF_GPU_TYPE_INTEGRATED = 1,
   RDF_GPU_TYPE_DISCRETE = 2,
   RDF_GPU_TYPE_VIRTUAL = 3,
};

/* Memory types */
enum rdf_memory_type {
   RDF_MEMORY_TYPE_UNKNOWN = 0,
   RDF_MEMORY_TYPE_DDR = 1,
   RDF_MEMORY_TYPE_DDR2 = 2,
   RDF_MEMORY_TYPE_DDR3 = 3,
   RDF_MEMORY_TYPE_DDR4 = 4,
   RDF_MEMORY_TYPE_DDR5 = 5,
   RDF_MEMORY_TYPE_GDDR3 = 6,
   RDF_MEMORY_TYPE_GDDR4 = 7,
   RDF_MEMORY_TYPE_GDDR5 = 8,
   RDF_MEMORY_TYPE_GDDR6 = 9,
   RDF_MEMORY_TYPE_HBM = 10,
   RDF_MEMORY_TYPE_HBM2 = 11,
   RDF_MEMORY_TYPE_HBM3 = 12,
   RDF_MEMORY_TYPE_LPDDR4 = 13,
   RDF_MEMORY_TYPE_LPDDR5 = 14,
};

/* API types */
enum rdf_api_type {
   RDF_API_TYPE_VULKAN = 4,
};

/* ---- AsicInfo chunk (Version 3)---- */

/* GFX IP level */
struct rdf_gfx_ip_level {
   uint16_t major;
   uint16_t minor;
   uint16_t stepping;
};
static_assert(sizeof(struct rdf_gfx_ip_level) == 6, "rdf_gfx_ip_level doesn't match RDF-RGP spec");

/* AsicInfo chunk structure (version 3) */
struct rdf_asic_info {
   uint32_t pci_id;                                        /* GPU PCI ID */
   uint64_t shader_core_clock_frequency;                   /* GPU core clock in Hz */
   uint64_t memory_clock_frequency;                        /* Memory clock in Hz */
   uint64_t gpu_timestamp_frequency;                       /* GPU timestamp clock in Hz */
   uint64_t max_shader_core_clock;                         /* Max shader core clock in Hz */
   uint64_t max_memory_clock;                              /* Max memory clock in Hz */
   int32_t device_id;                                      /* PCIE device ID */
   int32_t device_revision_id;                             /* PCIE revision ID */
   int32_t vgprs_per_simd;                                 /* VGPRs per SIMD */
   int32_t sgprs_per_simd;                                 /* SGPRs per SIMD */
   int32_t shader_engines;                                 /* Number of shader engines */
   int32_t compute_unit_per_shader_engine;                 /* CUs per SE */
   int32_t simd_per_compute_unit;                          /* SIMDs per CU */
   int32_t wavefronts_per_simd;                            /* Wavefronts per SIMD */
   int32_t minimum_vgpr_alloc;                             /* Min VGPR allocation */
   int32_t vgpr_alloc_granularity;                         /* VGPR allocation granularity */
   int32_t minimum_sgpr_alloc;                             /* Min SGPR allocation */
   int32_t sgpr_alloc_granularity;                         /* SGPR allocation granularity */
   int32_t hardware_contexts;                              /* Hardware contexts */
   uint32_t gpu_type;                                      /* rdf_gpu_type */
   struct rdf_gfx_ip_level gfx_ip_level;                   /* GFX IP version */
   int32_t gpu_index;                                      /* GPU index */
   int32_t ce_ram_size;                                    /* CE RAM size in bytes */
   int32_t ce_ram_size_graphics;                           /* Graphics CE RAM size */
   int32_t ce_ram_size_compute;                            /* Compute CE RAM size */
   int32_t max_number_of_dedicated_cus;                    /* Dedicated CUs for RT audio */
   int64_t vram_size;                                      /* VRAM size in bytes */
   int32_t vram_bus_width;                                 /* VRAM bus width in bits */
   int32_t l2_cache_size;                                  /* L2 cache (TCC/GL2C) size */
   int32_t l1_cache_size;                                  /* L1 cache (TCP/L0) per CU */
   int32_t lds_size;                                       /* LDS size per CU */
   char gpu_name[RDF_GPU_NAME_MAX_SIZE];                   /* GPU name string */
   float alu_per_clock;                                    /* ALUs per clock */
   float texture_per_clock;                                /* Textures per clock */
   float prims_per_clock;                                  /* Primitives per clock */
   float pixels_per_clock;                                 /* Pixels per clock */
   uint32_t memory_ops_per_clock;                          /* Memory ops per clock cycle */
   uint32_t memory_chip_type;                              /* rdf_memory_type */
   uint32_t lds_granularity;                               /* LDS allocation granularity */
   uint16_t cu_mask[RDF_MAX_NUM_SE][RDF_SA_PER_SE];        /* CU presence mask */
   uint32_t pixel_packer_mask[RDF_BITS_PER_SHADER_ENGINE]; /* Pixel packer mask */
   uint32_t gl1_cache_size;                                /* GL1 cache per SA */
   uint32_t inst_cache_size;                               /* Instruction cache per CU */
   uint32_t scalar_cache_size;                             /* Scalar cache (K$) per CU */
   uint32_t mall_cache_size;                               /* MALL cache (Infinity cache) */
};
static_assert(sizeof(struct rdf_asic_info) == 608, "rdf_asic_info doesn't match RDF-RGP spec");

/* ---- ApiInfo chunk (version 2) ---- */
struct rdf_api_info {
   uint32_t api_type;          /* rdf_api_type */
   uint16_t api_version_major; /* Client API major version */
   uint16_t api_version_minor; /* Client API minor version */
};
static_assert(sizeof(struct rdf_api_info) == 8, "rdf_api_info doesn't match RDF-RGP spec");

/* ---- ClockCalibration chunk (for CPU/GPU clock synchronization) ---- */

struct rdf_clock_calibration {
   uint32_t pci_id;        /* GPU PCI ID */
   uint64_t cpu_timestamp; /* CPU timestamp (clock_monotonic) */
   uint64_t gpu_timestamp; /* GPU timestamp */
};
static_assert(sizeof(struct rdf_clock_calibration) == 24,
              "rdf_clock_calibration doesn't match RDF-RGP spec");

/* ---- SPM (Streaming Performance Monitors) chunks ---- */

/* SPM session header (SpmSession chunk) */
struct rdf_spm_session_header {
   uint32_t pci_id;            /* GPU PCI ID */
   uint32_t flags;             /* SPM trace flags (reserved) */
   uint32_t sampling_interval; /* Sampling interval in GPU clocks */
   uint32_t num_timestamps;    /* Number of sample timestamps */
   uint32_t num_spm_counters;  /* Number of SPM counters */
};
static_assert(sizeof(struct rdf_spm_session_header) == 20,
              "rdf_spm_session_header doesn't match RDF-RGP spec");

/* SPM counter data header (SpmCounterData chunk) */
struct rdf_spm_counter_data_header {
   uint32_t pci_id;         /* GPU PCI ID */
   uint32_t gpu_block;      /* ac_pc_gpu_block enum value */
   uint32_t block_instance; /* Instance index of the block */
   uint32_t event_index;    /* Event index within the block */
   uint32_t data_size;      /* Size of counter data in bytes */
};
static_assert(sizeof(struct rdf_spm_counter_data_header) == 20,
              "rdf_spm_counter_data_header doesn't match RDF-RGP spec");

/* ---- SqttData chunk header ---- */

struct rdf_sqtt_data_header {
   uint32_t pci_id;                       /* GPU PCI ID */
   uint32_t shader_engine;                /* Shader engine index */
   uint32_t sqtt_version;                 /* SQTT data version */
   uint32_t instrumentation_version_spec; /* Instrumentation spec version */
   uint32_t instrumentation_version_api;  /* Instrumentation API version */
   uint32_t wgp_index;                    /* WGP/CU index */
   uint64_t trace_buffer_size;            /* Size of trace buffer */
   uint32_t instruction_timing_enabled : 1;
   uint32_t exec_pop_tokens_enabled : 1;
   uint32_t reserved : 30;
};
static_assert(sizeof(struct rdf_sqtt_data_header) == 40,
              "rdf_sqtt_data_header doesn't match RDF-RGP spec");

/* ---- CodeObjectDatabase chunk ---- */

struct rdf_code_object_header {
   uint32_t pci_id;              /* GPU PCI ID */
   uint64_t code_object_hash[2]; /* 128-bit code object hash (low, high) */
};
static_assert(sizeof(struct rdf_code_object_header) == 24,
              "rdf_code_object_header doesn't match RDF-RGP spec");

/* ---- CodeObjectLoaderEvents chunk ---- */

enum rdf_loader_event_type {
   RDF_LOADER_EVENT_LOAD_TO_GPU = 0,
   RDF_LOADER_EVENT_UNLOAD = 1,
};

struct rdf_loader_events_record {
   uint32_t pci_id;              /* GPU PCI ID */
   uint32_t loader_event_type;   /* rdf_loader_event_type */
   uint64_t base_address;        /* GPU VA where code object loaded */
   uint64_t code_object_hash[2]; /* 128-bit code object hash (low, high) */
   uint64_t time_stamp;          /* Timestamp of load event */
};
static_assert(sizeof(struct rdf_loader_events_record) == 40,
              "rdf_loader_events_record doesn't match RDF-RGP spec");

struct rdf_loader_events_header {
   uint32_t count; /* Number of loader events */
};
static_assert(sizeof(struct rdf_loader_events_header) == 4,
              "rdf_loader_events_header doesn't match RDF-RGP spec");

/* ---- PsoCorrelation chunk ----  */

struct rdf_pso_correlation_record {
   uint32_t pci_id;             /* GPU PCI ID */
   uint64_t api_pso_hash;       /* API-level PSO hash */
   uint64_t pipeline_hash[2];   /* Internal pipeline hash (low, high) */
   char api_level_obj_name[64]; /* PSO name string */
};
static_assert(sizeof(struct rdf_pso_correlation_record) == 96,
              "rdf_pso_correlation_record doesn't match RDF-RGP spec");

struct rdf_pso_correlation_header {
   uint32_t count; /* Number of PSO correlations */
};
static_assert(sizeof(struct rdf_pso_correlation_header) == 4,
              "rdf_pso_correlation_header doesn't match RDF-RGP spec");

/* ---- QueueInfo and QueueEvent chunks ---- */

/* Queue types */
enum rdf_queue_type {
   RDF_QUEUE_TYPE_UNKNOWN = 0,
   RDF_QUEUE_TYPE_UNIVERSAL = 1,
   RDF_QUEUE_TYPE_COMPUTE = 2,
   RDF_QUEUE_TYPE_DMA = 3,
};

/* Engine types */
enum rdf_engine_type {
   RDF_ENGINE_TYPE_UNKNOWN = 0,
   RDF_ENGINE_TYPE_UNIVERSAL = 1,
   RDF_ENGINE_TYPE_COMPUTE = 2,
   RDF_ENGINE_TYPE_EXCLUSIVE_COMPUTE = 3,
   RDF_ENGINE_TYPE_DMA = 4,
   RDF_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL = 7,
   RDF_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS = 8,
};

struct rdf_queue_info_record {
   uint32_t pci_id;        /* GPU PCI ID */
   uint64_t queue_id;      /* Unique queue identifier */
   uint64_t queue_context; /* Queue context handle */
   uint8_t queue_type;     /* rdf_queue_type */
   uint8_t engine_type;    /* rdf_engine_type */
};
static_assert(sizeof(struct rdf_queue_info_record) == 32,
              "rdf_queue_info_record doesn't match RDF-RGP spec");

enum rdf_queue_event_type {
   RDF_QUEUE_EVENT_CMDBUF_SUBMIT = 0,
   RDF_QUEUE_EVENT_SIGNAL_SEMAPHORE = 1,
   RDF_QUEUE_EVENT_WAIT_SEMAPHORE = 2,
   RDF_QUEUE_EVENT_PRESENT = 3,
};

struct rdf_queue_event_record {
   uint32_t pci_id;           /* GPU PCI ID */
   uint64_t queue_id;         /* Queue that triggered the event */
   uint32_t event_type;       /* rdf_queue_event_type */
   uint32_t sqtt_cb_id;       /* Command buffer ID in SQTT */
   uint64_t frame_index;      /* Frame number */
   uint32_t submit_sub_index; /* Sub-index within submit */
   uint64_t api_event_id;     /* API-level identifier */
   uint64_t cpu_timestamp;    /* CPU time of event */
   uint64_t gpu_timestamp_1;  /* GPU start timestamp */
   uint64_t gpu_timestamp_2;  /* GPU end timestamp (CmdBufSubmit only) */
};
static_assert(sizeof(struct rdf_queue_event_record) == 72,
              "rdf_queue_event_record doesn't match RDF-RGP spec");

#ifdef __cplusplus
}
#endif

#endif /* AC_RDF_RGP_CHUNKS_H */
