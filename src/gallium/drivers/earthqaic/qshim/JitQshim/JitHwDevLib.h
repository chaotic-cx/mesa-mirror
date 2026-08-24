/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef _JITHWDEV_LIB_H_
#define _JITHWDEV_LIB_H_

#include "AICMetadataExecCtx.h"
#include "jit_common.h"
#include "jit_qshim_api.h"
#include "jit_reader.h"
#include "nsp_qshim_jit_interface.h"
#include <stdarg.h>
#include <stddef.h>

const uint32_t L2TCM_CMD_BUFFER_INPUT_SIZE = 0x800;
const uint32_t L2TCM_CMD_BUFFER_OUTPUT_SIZE = 0x800;
const uint32_t L2TCM_CMD_BUFFER_BASE_OFFSET = 0x20000;
const uint32_t L2TCM_CMD_BUFFER_OUTPUT_OFFSET =
    L2TCM_CMD_BUFFER_BASE_OFFSET + L2TCM_CMD_BUFFER_INPUT_SIZE;
const uint32_t SHDDR_CMD_BUFFER_OUTPUT_SIZE = 0x800000;
const uint32_t MAX_ERROR_REPONSE_STR_LENGTH = 112;
const uint32_t MAX_AIC_DEV_EXECUTABLE = 16;
const uint32_t MAX_LIB_NAME_STR_LEN = 32;
const uint8_t MAX_NUM_THREADS = 6;
const uint8_t NUM_WORKER_THREADS = 5;
const uint8_t MAX_AIC_CORES = 16;
/* Master thread processes command sets - As per design specification */
const uint8_t MASTER_THREAD = 5;
const uint8_t L2TCM_WORKER_THREADS_START_DB_OFFSET = 2;
const uint8_t L2TCM_WORKER_THREADS_FINISH_DB_OFFSET =
    L2TCM_WORKER_THREADS_START_DB_OFFSET + MAX_NUM_THREADS;
/* L2TCM Map for more info */
const uint16_t MAX_UDMA_REQS = 256;
const uint8_t U64_BIT_COUNT = 64;
const uint16_t L2TCM_UDMA_START_DB_OFFSET = 1024;
const uint8_t MASTER_SLAVE_NSP_INTER_CMD_SET_SYNC_DB_OFFSET = 32;
const uint8_t MASTER_SLAVE_NSP_INTER_CMD_SYNC_DB_OFFSET = 32;
const uint32_t NSP0_VTCM_MULTICAST_ID = 6;
const uint8_t SLAVE_NSP_ENTRY_SEM_SYNC_DB_OFFSET_BEGIN = 14;
const uint8_t SLAVE_NSP_ENTRY_SEM_SYNC_DB_OFFSET_LAST = 28;
const uint8_t L2TCM_ERR_DB_OFFSET = 48;

const uint8_t NNC_AIC_JIT_EXE_MAX_RW_SEG = 16;
const uint8_t NNC_AIC_JIT_EXE_RO_IO_MAX_BUF = 1;
const uint32_t JIT_EXE_BUF_ALLOC_NUM_ARGS = 3;
const uint32_t JIT_IO_BUF_ALLOC_NUM_ARGS = 1;
const uint32_t JIT_EXE_BUF_MMAP_NUM_ARGS = 6;
const uint32_t JIT_IO_BUF_MMAP_NUM_ARGS = 3;

const uint8_t NUM_HVX_THREADS = 4;

const uint8_t JIT_PROF_TABLE_MAX_SIZE = 15;

const uint32_t SHDDR_BUFFER_ALPS_BASE_ADDR = 0x800000;
const uint32_t SHDDR_BUFFER_ALPS_SIZE = 0x100000;
const uint32_t MAX_GOT_PLT_OFFSETS = 2048;
const uint32_t QURT_HVX_RELEASE_CONTEXT = 3;

const uint16_t JIT_CMD_SET_INPUT_DB_BASE_INDEX = 2048;
const uint16_t JIT_CMD_SET_MAX_OUTPUT_SEM_VAL = 4000;
const uint64_t JIT_MAX_BUF_DESC_SIZE = 0x100000000;
const uint64_t JIT_MMAP_BLOCK_SIZE = 0x1000000;

typedef enum {
  NNC_MMAP_JIT_ADDR_TYPE_EXE = 0,
  NNC_MMAP_JIT_ADDR_TYPE_IO = 1,
  NNC_MMAP_JIT_ADDR_TYPE_MAX
} JitMmapAddrType_t;

typedef enum {
  JIT_DDR_BUF_EXE,
  JIT_DDR_BUF_IO,
  JIT_DDR_BUF_INVALID
} JitBufAllocType_t;

typedef enum { JIT_ENTRY_SEM, JIT_EXIT_SEM } JitSemType_t;

typedef enum {
  JIT_QPC_SHIM_ALLOCATOR_INVALID = 0,
  JIT_QPC_SHIM_ALLOCATOR_NETWORK_HEAP = 1,
  JIT_QPC_SHIM_ALLOCATOR_DDR = 2,
  JIT_QPC_SHIM_ALLOCATOR_TCM = 3,
  JIT_QPC_SHIM_ALLOCATOR_MAX,
} JitQpcShimAllocatorType_t;

/** QuRT mutex type */
typedef union QurtMutex {
  struct {
    unsigned int holder;
    unsigned int count;
    unsigned int queue;
    unsigned int wait_count;
  };
  unsigned long long int raw;
} QurtMutex_t;

constexpr unsigned DMA_state_mask = 0x3;
constexpr unsigned DMA_state_idle = 0;
constexpr unsigned DMA_state_run = 1;
constexpr unsigned DMA_state_error = 2;

static inline bool isUserDMAError(uint32_t dm0) {
  return (dm0 & DMA_state_mask) == DMA_state_error;
}

typedef struct IOAddrSizeInfo {
  uint64_t buffAddr;
  uint64_t offsetAlpBase;
  uint64_t offset;
  uint32_t size;
  uint32_t bufId;
  uint32_t virtAddr;
  uint32_t nodeIndex;
  AicJit_AicMemType_enum_t bufferLocation;
  bool isMap;
  bool inUse;
} IOAddrSizeInfo_t;

typedef struct OpMmapNodeInfo {
  uint64_t mapAddr;
  uint64_t mapOffsetAlpBase;
  uint32_t blockOffset; // Mapped Buffer offset from 16MB block
  uint32_t virtAddr;    // Mapped virtual address of mapAddr
  uint32_t mapSize;     // Mapped Size
  uint32_t nodeIndex;
  uint16_t opMmapCount; // no of mappings count on this physical address block
  bool isMap;
} OpMmapNodeInfo_t;

typedef struct OpMmapNode {
  OpMmapNodeInfo_t opMmapNodeInfo;
  OpMmapNode *next;
} OpMmapNode_t;

typedef struct OpMmapBuffInfo {
  uint64_t buffAddr;      // The actual buffAddr passed from MLFW in buffDesc
  uint64_t offsetAlpBase; // offset of bufAddr from the base DDR base buffer
  uint64_t offset;        // offset of buffer as specified in buffer descriptor
  uint64_t buffSize;      // Total size of buffer from buffer descriptor
  uint32_t bufId;
  OpMmapNode_t *opMmapNode;
} OpMmapBuffInfo_t;

typedef struct ExeAddrSizeInfo {
  uint64_t roOffset;
  uint64_t rwOffset;
  uint64_t goldenRwOffset;
  uint64_t roSize;
  uint64_t rwSize;
  uint64_t goldenRwSize;
  uint32_t roBufId;
  uint32_t rwBufId;
  uint32_t goldenRwBufId;
  uint32_t virtAddrExe;
  uint32_t nodeIndexExe;
  uint32_t virtAddrGoldenRw;
  uint32_t nodeIndexGoldenRw;
  uint32_t gotPltOffsets[MAX_GOT_PLT_OFFSETS];
  uint32_t numGotPltOffsets;
  char exeName[MAX_LIB_NAME_STR_LEN];
  bool isCommonLib;
} ExeAddrSizeInfo_t;

typedef struct AicExeNode {
  ExeAddrSizeInfo_t exeAddrSizeInfo;
  AicExeNode *next;
} AicExeNode_t;

typedef struct AicBufferDescriptorNode {
  union {
    IOAddrSizeInfo_t ioAddrSizeInfo;
    OpMmapBuffInfo_t opMmapBufInfo;
  };
  AicJit_AicBufferMappingRule_enum_t mmapRule;
  AicBufferDescriptorNode *next;
} AicBufferDescriptorNode_t;

typedef struct AicParameterType {
  AicJit_AicParameterType_enum_t paramType;
  uint32_t index;
} AicParameterType_t;

typedef struct {
  uint64_t res[MAX_UDMA_REQS / U64_BIT_COUNT];
  uint16_t size;
  uint16_t available;
  uint16_t allocated;
} BitMaskResContext_t;

typedef struct {
  uint64_t start_ts;
  uint64_t stop_ts;
  char tag[16];
} Profiling_Datapoint_t;

typedef struct {
  int32_t entryPointFuncAddr;
  uint32_t ringBufferIdx;
  uint32_t ringBufferDepth;
  uint32_t cmdBufSize;
  char commonLibName[MAX_LIB_NAME_STR_LEN];
  char operatorLibName[MAX_LIB_NAME_STR_LEN];
} DebugInfo_t;

/** @cond rest_reg_dist*/
/** QuRT cache type; specifies data cache or instruction cache. */
typedef enum {
  QURT_MEM_ICACHE, /**< Instruction cache.*/
  QURT_MEM_DCACHE  /**< Data cache.*/
} qurt_mem_cache_type_t;

/** QuRT cache operation code type. */
typedef enum {
  QURT_MEM_CACHE_FLUSH,                  /**< Flush. */
  QURT_MEM_CACHE_INVALIDATE,             /**< Invalidate */
  QURT_MEM_CACHE_FLUSH_INVALIDATE,       /**< Flush invalidate. */
  QURT_MEM_CACHE_FLUSH_ALL,              /**< Flush all. */
  QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL,   /**< Flush invalidate all. */
  QURT_MEM_CACHE_TABLE_FLUSH_INVALIDATE, /**< Table flush invalidate. */
  QURT_MEM_CACHE_FLUSH_INVALIDATE_L2,    /**< L2 flush invalidate.*/
} qurt_mem_cache_op_t;

typedef enum {
  PRNG_WQ_SLOT_EMPTY = 0,
  PRNG_WQ_SLOT_PENDING = 1,
  PRNG_WQ_SLOT_COMPLETE = 2
} prng_work_item_status_t;

typedef struct {
  uint32_t *addr;
  uint32_t numElements;
  uint32_t status;
  uint8_t padding[4];
} prng_work_item_t;

typedef void *(*JitQpcShim_Malloc_t)(size_t);
typedef void *(*JitQpcShim_tcmMalloc_t)(uint32_t);
typedef int (*JitQpcShim_ddrMalloc_t)(jit_ddr_malloc_params *);
typedef void *(*JitQpcShim_AlignedMalloc_t)(uint32_t, size_t);
typedef void (*JitQpcShim_Free_t)(void *);
typedef int (*JitQpcShim_ddrFree_t)(uint32_t, uint32_t *, uint8_t);
typedef void *(*JitQpcShim_Memcpy_t)(void *, void *, size_t);
typedef void *(*JitQpcShim_Memset_t)(void *, int, size_t);
typedef int (*JitQpcShim_Snprintf_t)(char *str, size_t size, const char *format,
                                     ...);
typedef int (*JitQpcShim_Mmap_t)(void **, uint32_t *, JitMmapAddrType_t,
                                 uint32_t, ...);
typedef int (*JitQpcShim_Vsnprintf_t)(char *str, size_t size,
                                      const char *format, va_list args);
typedef void (*JitQpcShim_QurtMutexInit_t)(QurtMutex_t *);
typedef void (*JitQpcShim_QurtMutexLock_t)(QurtMutex_t *);
typedef void (*JitQpcShim_QurtMutexUnlock_t)(QurtMutex_t *);
typedef void (*JitQpcShim_QurtMutexDestroy_t)(QurtMutex_t *);
typedef uint32_t (*JitQpcShim_QurtHvxSet_t)(uint32_t);
typedef uint8_t (*JitQpcShim_getVtid_t)(void);
typedef void (*JitQpcShim_assert_t)(void);
typedef void (*JitQpcShim_UnmapExe_t)(uint32_t, uint32_t);
typedef int (*JitQpcShim_QurtMemCacheClean_t)(unsigned int, unsigned int,
                                              unsigned int, unsigned int);

void initializeQpcShim(AICExecContext *qctx, uint8_t tid, uint32_t stid);
void run(AICExecContext *qctx, uint8_t tid, uint32_t stid);

#endif
