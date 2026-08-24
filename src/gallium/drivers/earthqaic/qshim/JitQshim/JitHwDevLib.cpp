/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#include "JitHwDevLib.h"
#include "ComputeAPI.h"
#include "Log.h"
#include "SimpleAPI.h"
#include "hexagon_protos.h"
#include "jit_common.h"
#include "jit_dev_exe_function.h"
#include "jit_dev_status_codes.h"
#include "jit_qshim_api.h"
#include "jit_reader.h"
#include "jit_verifier.h"
#include "nnc_err_inf.h"

#ifdef __cplusplus
extern "C" {
#endif

void *memset(void *dst, int c, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  unsigned char v = (unsigned char)c;
  for (size_t i = 0; i < n; ++i) {
    d[i] = v;
  }
  return dst;
}

char *strcpy(char *dst, const char *src) {
  char *ret = dst;
  while ((*dst++ = *src++) != '\0') {
    /* copy including terminating NUL */
  }
  return ret;
}

#ifdef __cplusplus
}
#endif

using namespace qaic;

JitQpcShim_Malloc_t g_networkHeapMalloc = nullptr;
JitQpcShim_ddrMalloc_t g_ddrMalloc = nullptr;
JitQpcShim_Free_t g_networkHeapFree = nullptr;
JitQpcShim_ddrFree_t g_ddrFree = nullptr;
JitQpcShim_Memcpy_t g_memcpy = nullptr;
JitQpcShim_Snprintf_t g_snprintf = nullptr;
JitQpcShim_Memset_t g_memset = nullptr;
JitQpcShim_Mmap_t g_mmap = nullptr;
JitQpcShim_Vsnprintf_t g_vsnprintf = nullptr;
JitQpcShim_QurtMutexInit_t g_qurtMutexInit = nullptr;
JitQpcShim_QurtMutexLock_t g_qurtMutexLock = nullptr;
JitQpcShim_QurtMutexUnlock_t g_qurtMutexUnlock = nullptr;
JitQpcShim_QurtMutexDestroy_t g_qurtMutexDestroy = nullptr;
JitQpcShim_QurtHvxSet_t g_qurtHvxSet = nullptr;
AicJitDevExeFunction entryPointFunc = nullptr;
JitQpcShim_getVtid_t g_getVtid = nullptr;
JitQpcShim_assert_t g_assert = nullptr;
JitQpcShim_UnmapExe_t g_unmapExe = nullptr;
JitQpcShim_QurtMemCacheClean_t g_qurtMemCacheClean = nullptr;
nnc_exit_fp g_exitFp = nullptr;

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif /* MIN */
#define UTIMER_FREQ_MS (19200u)

/* Global buffer pointer that points to the flatbuffer command set */
static AicJit_AicCommandSet_table_t buffer = NULL;
static AicCommandSetResponseHeader *cmdBuffRspHdr = nullptr;
static JitDevStatusCode_t initStatus = JIT_DEV_STATUS_SUCCESS;

static uint8_t g_firstOpErrorReported = 0;
static uint8_t g_ErrorFatalReported = 0;

static AicBufferDescriptorNode_t *headAicBuffDescList = nullptr;
static AicBufferDescriptorNode_t *headAicOpBuffDescList = nullptr;
static AicBufferDescriptorNode_t **listAicDevBuffDescNodes = nullptr;
static uint8_t numAicDevBuffDescNodes = 0;

static AicExeNode_t *headAicDevExeList = nullptr;
static AicExeNode_t *commonDevExe = nullptr;
static AicExeNode_t *activeOpExe = nullptr;
static uint32_t *libAicDevExeBaseAddress = nullptr;
static uint8_t numLibAicDevExes = 0;

static uint8_t *localStorageBaseAddress = nullptr;
static uint8_t localStorageSize = 0;
static int32_t entryPointFuncAddr = 0;
static uint32_t threadIdBitMap = 0;
static AicJitPointerArray *listParamPtr = nullptr;
static uint8_t numExecInfo = 0;

/* Global used as atomic variable to synchronize across threads */
static uint32_t inputSync = 0;
static uint32_t outputSync = 0;
static uint32_t commandIdx = 0;
static uint32_t waitForDispatch = 0;

/* Global Response TLV */
static ResponseTLV_t *g_rspTlv = nullptr;
/* Global Profiling Response TLV */
static ResponseTLV_t *g_profTlv = nullptr;
/* Global Profiling Data Header */
static ProfilingData_t *g_profHeader = nullptr;
/* Global Profiling status Buffer pointer that points to the end of status
 * buffer */
static uint8_t *g_profStatusBuffer = nullptr;

/* Global response offset */
static AicCommandSetResponseInfo_t g_respInfo = {0};

static uint64_t g_rspOffset = 0;
static uint64_t g_rspSize = 0;
static jit_heap_ddr_alloc_alp_t g_allocAlps[JIT_HEAP_MAX_ALPS] = {0};
static jit_heap_ddr_alloc_alp_t g_vcQueueDdrAllocInfo = {0};
static prng_work_item_t prngWorkQueue[NUM_WORKER_THREADS] = {0};
static uint32_t prngWorkQueueCount = 0;

/* Global bitmap to store level of profiling */
static uint64_t profilingBitmap = 0;
static uint64_t g_startTs = 0;
static Profiling_Datapoint_t g_profTable[JIT_PROF_TABLE_MAX_SIZE];
static uint8_t g_profCount = 0;
static uint32_t g_profCmdId = 0;

/* Max size of profiling Status Buffer*/
static uint64_t g_profRspSize = 0;

static char *errRspBuffPtr[MAX_NUM_THREADS] = {nullptr};

static char msgBuffer[256] = {'\0'};
static char *msgBuffPtr = &msgBuffer[0];

#define CACHE_LINE_SIZE 128
__attribute__((aligned(CACHE_LINE_SIZE))) int32_t uDmaDbBitMaskResLock = 0;
__attribute__((aligned(CACHE_LINE_SIZE))) int32_t syncPointResLock = 0;
__attribute__((aligned(CACHE_LINE_SIZE))) int32_t opErrorReportLock = 0;
__attribute__((aligned(CACHE_LINE_SIZE))) int32_t profStatusBufferLock = 0;
__attribute__((aligned(CACHE_LINE_SIZE))) int32_t qshimMmapLock = 0;

static BitMaskResContext_t uDmaDbBitMask = {0};
static uint32_t uDmaDbVal = 1;

static uint32_t hostPid = 1;
static bool isHostPidSet = false;
static uint32_t g_numCores = 0;
static uint32_t g_numThreads = 0;

/* Ring buffer index */
static uint32_t ringBufferIdx = 0;

/* Counter to track output semaphore value */
static uint16_t opSemCounter = 0;

static AICExecContext *g_qctx = nullptr;

static AicBufferDescriptorNode_t tcmAicBuffDescNode = {0};

/* Host controlled request command buffer size */
static uint32_t g_cmdBufSize;

/* Host controlled ring buffer depth */
static uint32_t g_ringBufferDepth;

static uint64_t g_opTimeoutMs;

static uint8_t g_tidFinishDbWait;

static uint32_t g_opErrCommandId;
/* Need to modify the parser script if this offset changes */
// * Offset 0x40000 - uDMA dummy descriptor
// * Offset 0x40080 - uDMA descriptors per thread
//   + Start at 0x40080
//   + There are 16 cache lines reserved for uDMA descriptors per thread.
//   + The DMA descriptors are assumed to be 32-bit and that's how the cache
//   line is partitioned.
//   + Therefore, upto 64 uDMA descriptors per thread are supported.
// * Offset 0x4XXXX - Copy of uDMA descriptors - Copy is not done at the moment
// (future extension)
// * Offset 0x60000 - L2TCM Struct Debug Info

#define DEBUG_INFO_OFFSET 0x60000
static DebugInfo_t *debugInfo = nullptr;

void resetDebugInfoStruct(void) {
  memset((void *)debugInfo, 0, sizeof(DebugInfo_t));
}

void resetPerCommandBufferDebugInfo(void) {
  debugInfo->entryPointFuncAddr = 0;
  debugInfo->ringBufferIdx = 0;
  memset(&debugInfo->commonLibName[0], 0, sizeof(char) * MAX_LIB_NAME_STR_LEN);
  memset(&debugInfo->operatorLibName[0], 0,
         sizeof(char) * MAX_LIB_NAME_STR_LEN);
}

void setExeName(char *exeName, bool isCommonLib) {
  if (isCommonLib) {
    g_memcpy(&debugInfo->commonLibName[0], (void *)exeName,
             MAX_LIB_NAME_STR_LEN);
  } else {
    g_memcpy(&debugInfo->operatorLibName[0], (void *)exeName,
             MAX_LIB_NAME_STR_LEN);
  }
}

void setEntryPointFuncAddr(int32_t entryPointFuncAddr) {
  debugInfo->entryPointFuncAddr = entryPointFuncAddr;
}

void setRingBufferIdx(uint32_t ringBufferIdx) {
  debugInfo->ringBufferIdx = ringBufferIdx;
}

void setRingBufferDepth(uint32_t ringBufferDepth) {
  debugInfo->ringBufferDepth = ringBufferDepth;
}

void setCmdBufSize(uint32_t cmdBufSize) { debugInfo->cmdBufSize = cmdBufSize; }

uint8_t *getAndUpdate_profStatusBuffer(AICExecContext *qctx,
                                       JITProfilingEventType eventType) {
  LOG_DEBUG(qctx->logFuncPtr, "Profiling: Start Getting the PROF DATA address");
  uint8_t *target_address = nullptr;
  if (g_profStatusBuffer == nullptr) {
    LOG_ERROR(qctx->logFuncPtr,
              "Profiling Error: g_profStatusBuffer is nullptr");
  } else {
    int32_t expValue = 0;
    while (!qcAtomicCompareExchange(&profStatusBufferLock, &expValue, 1)) {
      expValue = 0; // Reset expected value for the next iteration
    }
    target_address = g_profStatusBuffer;
    uint8_t dataSize = 0;
    if (eventType == EVENT_TYPE_TIME_STAMP) {
      dataSize = sizeof(JITTimeStampEventData_t);
    } else if (eventType == EVENT_TYPE_PMU_DDR) {
      dataSize = sizeof(JITPMUDdrEventData_t);
    } else {
      dataSize = 0;
    }
    if (dataSize != 0) {
      if ((g_profHeader->size + sizeof(JITProfilingEventTrace_t) + dataSize) >
          g_profRspSize) {
        target_address = nullptr;
        g_profHeader->dataOverFlow = 1;
      } else {
        g_profStatusBuffer =
            (uint8_t *)((char *)g_profStatusBuffer +
                        sizeof(JITProfilingEventTrace_t) + dataSize);
        g_profHeader->numTraceEvent += 1;
        g_profHeader->size += sizeof(JITProfilingEventTrace_t) + dataSize;
      }
    } else {
      target_address = nullptr;
    }
    qcAtomicStore((uint32_t *)&profStatusBufferLock, 0);
    if (g_profHeader->dataOverFlow == 1) {
      LOG_ERROR(qctx->logFuncPtr,
                "Profiling Error: Profiling Status buffer dataOverFlow");
    } else if (target_address == nullptr) {
      LOG_ERROR(qctx->logFuncPtr, "Profiling Error: Wrong eventType");
    }
    LOG_DEBUG(
        qctx->logFuncPtr,
        "Profiling: End Getting the PROF DATA address, writing address: %p",
        target_address);
  }
  return target_address;
}

void write_Profiling_Event(AICExecContext *qctx, const char *eventName,
                           uint64_t eventUID, uint8_t threadId,
                           JITProfilingEventPhase eventPhase,
                           JITProfilingEventType eventType, void *eventData) {
  uint8_t *target_address = getAndUpdate_profStatusBuffer(qctx, eventType);

  if (target_address == nullptr) {
    LOG_ERROR(qctx->logFuncPtr,
              "Profiling Error: Failed to get status buffer writing address");
    return;
  }
  LOG_DEBUG(qctx->logFuncPtr,
            "Profiling: Start Writing the PROF DATA to address: %p",
            target_address);
  JITProfilingEventTrace_t *event = (JITProfilingEventTrace_t *)target_address;
  g_memcpy(&event->eventName, (void *)eventName, sizeof(event->eventName) - 1);
  event->eventName[sizeof(event->eventName) - 1] = '\0';
  event->eventUID = eventUID;
  event->threadId = threadId;
  event->eventPhase = eventPhase;
  event->eventType = eventType;

  if (eventType == EVENT_TYPE_TIME_STAMP) {
    JITTimeStampEventData_t *timeData =
        (JITTimeStampEventData_t *)event->eventData;
    timeData->deviceBootTimeUTC =
        ((JITTimeStampEventData_t *)eventData)->deviceBootTimeUTC;
    timeData->uCycleCounts =
        ((JITTimeStampEventData_t *)eventData)->uCycleCounts;
  } else if (eventType == EVENT_TYPE_PMU_DDR) {
    g_memcpy(event->eventData, eventData, sizeof(JITPMUDdrEventData_t));
  }
  LOG_DEBUG(qctx->logFuncPtr,
            "Profiling: END Writing the PROF DATA to address: %p",
            target_address);
}

AicJit_AicCommandSet_table_t
verify_AicJit_AicCommandSet(AICExecContext *qctx, uint8_t *buff, size_t size) {
#ifndef BLD_VAR_REL
  int ok = AicJit_AicCommandSet_verify_as_root((void *)buff, size);
  if (ok) {
    const char *s = flatcc_verify_error_string(ok);
    LOG_ERROR(qctx->logFuncPtr, "%s: buffer failed: %s", __func__, s);
    return NULL;
  } else {
    LOG_DEBUG(qctx->logFuncPtr, "%s: success: flatbuffer loaded", __func__);
  }
#endif
  AicJit_AicCommandSet_table_t buffer = AicJit_AicCommandSet_as_root(buff);
  return buffer;
}

void errExitThreadHandler() {
  uint16_t cmdSetOutputSem = 1;
  if ((g_qctx->virtualNSPId == 0) && (g_getVtid() == MASTER_THREAD) &&
      (qcDoorbellRead(L2TCM_ERR_DB_OFFSET) != 0)) {
    LOG_DEBUG(g_qctx->logFuncPtr, "Output buffers have been sent");
    qcSemaphoreInc(cmdSetOutputSem);
    g_assert();
  }
  /*call the original thread exit handler*/
  g_exitFp();
}

void setMasterNspErrorDb() {
  uint32_t mcId = 0;
  uint16_t exitDb = getExitDbNum();
  AicJit_AicSemaphoreSignalType_enum_t aicSemSignalType;
  if (AicJit_AicCommandSet_semaphoares_is_present(buffer)) {
    AicJit_AicCommandSetSemaphores_table_t semaphores =
        AicJit_AicCommandSet_semaphoares(buffer);
    uint32_t signalSemListSize = AicJit_AicSemaphoreSignal_vec_len(
        AicJit_AicCommandSetSemaphores_exitSignalSemList(semaphores));

    for (uint8_t i = 0; i < signalSemListSize; i++) {
      AicJit_AicSemaphoreSignal_struct_t signalSem;
      signalSem = AicJit_AicSemaphoreSignal_vec_at(
          AicJit_AicCommandSetSemaphores_exitSignalSemList(semaphores), i);

      if (AicJit_AicSemaphoreSignalType_is_known_value(
              AicJit_AicSemaphoreSignal_type(signalSem))) {
        aicSemSignalType = AicJit_AicSemaphoreSignal_type(signalSem);
      }

      if (aicSemSignalType == AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP) {
        mcId = AicJit_AicSemaphoreSignal_mcID(signalSem);
        LOG_DEBUG(g_qctx->logFuncPtr, "%s: mcId: %d", __func__, mcId);
      }
    }
    qcDoorbellUnicast(mcId, exitDb, 1);
    qcDoorbellUnicast(mcId, L2TCM_ERR_DB_OFFSET, 1);
  }
}

void resetErrRspStrPtr(uint8_t tid) { errRspBuffPtr[tid] = nullptr; }

void setErrRspStrPtr(uint8_t tid, char *ptr) { errRspBuffPtr[tid] = ptr; }

void qshimLog(uint32_t mask, uint32_t dataCount, const char *formatStr, ...) {

  if (dataCount > 10) {
    LOG_ERROR(g_qctx->logFuncPtr, "Max 10 arguments allowed");
    return;
  }

  char buf[256] = {'\0'};
  va_list args;
  va_start(args, formatStr);
  (void)g_vsnprintf(buf, sizeof(buf), formatStr, args);
  va_end(args);

  int tid = -1;
  if (g_getVtid != nullptr) {
    tid = (int)g_getVtid();
  }

  if (mask & NNC_LOG_MASK_INFO) {
    LOG_INFO(g_qctx->logFuncPtr, "%s", buf);
  } else if (mask & NNC_LOG_MASK_DEBUG) {
    LOG_DEBUG(g_qctx->logFuncPtr, "%s", buf);
  } else if (mask & NNC_LOG_MASK_WARN) {
    LOG_WARN(g_qctx->logFuncPtr, "%s", buf);
  } else if (mask & NNC_LOG_MASK_FATAL) {
    LOG_FATAL(g_qctx->logFuncPtr, "%s", buf);
  }

  if (mask & NNC_LOG_MASK_ERROR) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s", buf);

    if (tid != MASTER_THREAD) {
      int32_t expValue = 0;
      while (!qcAtomicCompareExchange(&opErrorReportLock, &expValue, 1)) {
        expValue = 0; // Reset expected value for the next iteration
      }
      if (g_firstOpErrorReported != 0 || g_ErrorFatalReported != 0) {
        qcAtomicStore((uint32_t *)&opErrorReportLock, 0);
        return;
      }
      g_firstOpErrorReported = 1;
      qcAtomicStore((uint32_t *)&opErrorReportLock, 0);
    }

    char *errorString = errRspBuffPtr[MASTER_THREAD];
    if (errorString == nullptr) {
      LOG_ERROR(g_qctx->logFuncPtr,
                "memory is not allocated for reporting error string");
      return;
    }

    // Prefix thread Id
    g_snprintf(errorString, MAX_ERROR_REPONSE_STR_LENGTH, "NSP %d: TID %d: %s",
               g_qctx->virtualNSPId, tid, buf);
    LOG_DEBUG(g_qctx->logFuncPtr, "error string reported: %s", errorString);
  }

  if (mask & NNC_LOG_MASK_FATAL) {
    int32_t expValue = 0;
    while (!qcAtomicCompareExchange(&opErrorReportLock, &expValue, 1)) {
      expValue = 0; // Reset expected value for the next iteration
    }
    if (g_ErrorFatalReported) {
      qcAtomicStore((uint32_t *)&opErrorReportLock, 0);
      g_exitFp();
    }
    g_ErrorFatalReported = 1;
    qcAtomicStore((uint32_t *)&opErrorReportLock, 0);

    cmdBuffRspHdr->status = JIT_DEV_ERROR_FATAL;
    if (g_rspTlv != nullptr) {
      ResponseTLV_t *rspTlv = g_rspTlv;
      CommandValue_t *cmdVal = (CommandValue_t *)&rspTlv->value[0];
      cmdVal->commandID = g_opErrCommandId;
      cmdBuffRspHdr->tlvCount = rspTlv->slotCount;
      cmdVal->status = JIT_DEV_ERROR_FATAL;
    }

    setErrRspStrPtr(tid, &cmdBuffRspHdr->errMsg[0]);
    char *errorString = errRspBuffPtr[tid];
    if (errorString == nullptr) {
      LOG_ERROR(g_qctx->logFuncPtr,
                "memory is not allocated for reporting error string");
      return;
    }

    g_snprintf(errorString, MAX_ERROR_REPONSE_STR_LENGTH, "NSP %d: TID %d: %s",
               g_qctx->virtualNSPId, tid, buf);
    LOG_DEBUG(g_qctx->logFuncPtr, "error string reported: %s", errorString);

    LOG_FATAL(g_qctx->logFuncPtr, "%s", buf);

    uintptr_t caller0 = (uintptr_t)__builtin_return_address(0);
    uintptr_t caller1 = (uintptr_t)__builtin_return_address(1);
    LOG_FATAL(g_qctx->logFuncPtr, "caller0: 0x%x, caller1: 0x%x", caller0,
              caller1);

    uint16_t exitDb = getExitDbNum();
    if (g_qctx->virtualNSPId == 0) {
      qcDoorbellWrite(L2TCM_ERR_DB_OFFSET, 1);
      qcDoorbellWrite(exitDb, 1);
      errExitThreadHandler();
    } else {
      setMasterNspErrorDb();
      qcDoorbellWrite(exitDb, 1);
      g_exitFp();
    }
  }
}

/* Function returns offset from the base of ALP corresponding to the bufId */
JitDevStatusCode_t getOffsetAlpBase(AICExecContext *qctx, uint64_t addr,
                                    uint32_t bufId, uint64_t *offsetAlpBase) {
  jit_heap_ddr_alloc_alp_t *chunkList =
      (jit_heap_ddr_alloc_alp_t *)(qctx->baseSharedDDR +
                                   SHDDR_BUFFER_ALPS_BASE_ADDR);
  for (uint8_t i = 0; i < JIT_HEAP_MAX_ALPS; i++) {
    if (chunkList->buf_id == bufId) {
      *offsetAlpBase = addr - chunkList->addr_base;
      return JIT_DEV_STATUS_SUCCESS;
    }
    chunkList++;
  }
  LOG_ERROR(qctx->logFuncPtr, "%s: bufID no match %d", __func__, bufId);
  return JIT_DEV_ERROR_INVALID_PARAMETER;
}

int mmapAicBuffDescriptor(AICExecContext *qctx,
                          AicBufferDescriptorNode_t *aicBuffDescNode) {
  int status = 0;
  void *virtAddr;
  uint32_t nodeIdx;

  uint64_t offset = aicBuffDescNode->ioAddrSizeInfo.offsetAlpBase;
  uint32_t bufId = aicBuffDescNode->ioAddrSizeInfo.bufId;
  uint32_t size = aicBuffDescNode->ioAddrSizeInfo.size +
                  aicBuffDescNode->ioAddrSizeInfo.offset;

  if (g_mmap) {
    int retVal = g_mmap(&virtAddr, &nodeIdx, NNC_MMAP_JIT_ADDR_TYPE_IO,
                        JIT_IO_BUF_MMAP_NUM_ARGS, offset, bufId, size);
    if (retVal) {
      LOG_ERROR(qctx->logFuncPtr,
                "aicBuffDescNode failed to mmap bufId: 0x%x, size: 0x%x, "
                "offset: 0x%llx, retVal: %d",
                bufId, size, LOG64_DATA(offset), retVal);
      status = JIT_DEV_ERROR_MMAP;
    } else {
      aicBuffDescNode->ioAddrSizeInfo.virtAddr = (uint32_t)virtAddr;
      aicBuffDescNode->ioAddrSizeInfo.nodeIndex = nodeIdx;
      aicBuffDescNode->ioAddrSizeInfo.isMap = true;
      aicBuffDescNode->ioAddrSizeInfo.inUse = true;
      LOG_DEBUG(qctx->logFuncPtr,
                "allocBufNode mmap successfully offset: 0x%llx, bufId: 0x%x, "
                "virtAddr: 0x%x, nodeIdx: %d",
                LOG64_DATA(offset), bufId,
                aicBuffDescNode->ioAddrSizeInfo.virtAddr, nodeIdx);
    }
  }

  return status;
}

JitDevStatusCode_t mmapAicDevExecutable(
    AICExecContext *qctx, const char *exeName, AicJit_DevDdrBuf_t *roSegment,
    AicJit_DevDdrBuf_t *rwSegment, AicJit_DevDdrBuf_t *goldenRwSegment,
    AicExeNode_t **newNode, bool isCommonLib) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  void *virtAddr = nullptr;
  void *virtAddrGolden = nullptr;
  uint32_t nodeIdx = 0;
  uint32_t nodeIdxGolden = 0;
  int retVal = 0;
  jit_ddr_buf_alp_info roSegmentInfo = {0};
  jit_ddr_buf_alp_info rwSegmentInfo = {0};
  jit_ddr_buf_alp_info goldenRwSegmentInfo = {0};

  AicExeNode_t *aicDevExeNode =
      (AicExeNode_t *)g_networkHeapMalloc(sizeof(AicExeNode_t));
  if (aicDevExeNode == nullptr) {
    LOG_ERROR(qctx->logFuncPtr,
              "%s: failed to allocate memory for AicDevExe node", __func__);
    status = JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE;
    goto exit;
  }

  roSegmentInfo.offset = AicJit_DevDdrBuf_offset(roSegment);
  roSegmentInfo.size = AicJit_DevDdrBuf_size(roSegment);
  roSegmentInfo.buf_id = AicJit_DevDdrBuf_bufId(roSegment);

  rwSegmentInfo.offset = AicJit_DevDdrBuf_offset(rwSegment);
  rwSegmentInfo.size = AicJit_DevDdrBuf_size(rwSegment);
  rwSegmentInfo.buf_id = AicJit_DevDdrBuf_bufId(rwSegment);

  goldenRwSegmentInfo.offset = AicJit_DevDdrBuf_offset(goldenRwSegment);
  goldenRwSegmentInfo.size = AicJit_DevDdrBuf_size(goldenRwSegment);
  goldenRwSegmentInfo.buf_id = AicJit_DevDdrBuf_bufId(goldenRwSegment);

  g_memset((void *)aicDevExeNode, 0, sizeof(AicExeNode_t));
  aicDevExeNode->exeAddrSizeInfo.roOffset = roSegmentInfo.offset;
  aicDevExeNode->exeAddrSizeInfo.roSize = roSegmentInfo.size;
  aicDevExeNode->exeAddrSizeInfo.roBufId = roSegmentInfo.buf_id;
  aicDevExeNode->exeAddrSizeInfo.rwOffset = rwSegmentInfo.offset;
  aicDevExeNode->exeAddrSizeInfo.rwSize = rwSegmentInfo.size;
  aicDevExeNode->exeAddrSizeInfo.rwBufId = rwSegmentInfo.buf_id;
  aicDevExeNode->exeAddrSizeInfo.goldenRwOffset = goldenRwSegmentInfo.offset;
  aicDevExeNode->exeAddrSizeInfo.goldenRwSize = goldenRwSegmentInfo.size;
  aicDevExeNode->exeAddrSizeInfo.goldenRwBufId = goldenRwSegmentInfo.buf_id;
  aicDevExeNode->exeAddrSizeInfo.isCommonLib = isCommonLib;

  /* Memcpy the executable name */
  g_memcpy(&aicDevExeNode->exeAddrSizeInfo.exeName[0], (void *)exeName,
           MAX_LIB_NAME_STR_LEN);

  if (g_mmap) {
    retVal = g_mmap(&virtAddr, &nodeIdx, NNC_MMAP_JIT_ADDR_TYPE_EXE,
                    JIT_EXE_BUF_MMAP_NUM_ARGS, &roSegmentInfo, &rwSegmentInfo,
                    &goldenRwSegmentInfo, (uint32_t)isCommonLib,
                    &virtAddrGolden, &nodeIdxGolden);
    if (retVal) {
      LOG_ERROR(qctx->logFuncPtr, "%s: exe %s failed to mmap, retVal: %d",
                __func__, aicDevExeNode->exeAddrSizeInfo.exeName, retVal);
      status = JIT_DEV_ERROR_MMAP;
    } else {
      aicDevExeNode->exeAddrSizeInfo.virtAddrExe = (uint32_t)virtAddr;
      aicDevExeNode->exeAddrSizeInfo.nodeIndexExe = nodeIdx;
      aicDevExeNode->exeAddrSizeInfo.virtAddrGoldenRw =
          (uint32_t)virtAddrGolden;
      aicDevExeNode->exeAddrSizeInfo.nodeIndexGoldenRw = nodeIdxGolden;
      if (isCommonLib) {
        commonDevExe = aicDevExeNode;
      }
      LOG_DEBUG(qctx->logFuncPtr,
                "%s: exe %s mmap successfully, virtAddr: 0x%x, nodeIdx: 0x%x "
                "goldenVirtAddr: 0x%x goldenNodeIdx: 0x%x, commonLib %d",
                __func__, aicDevExeNode->exeAddrSizeInfo.exeName, virtAddr,
                nodeIdx, virtAddrGolden, nodeIdxGolden, isCommonLib);
    }
  }

  aicDevExeNode->next = headAicDevExeList;
  headAicDevExeList = aicDevExeNode;

  *newNode = aicDevExeNode;

  /* FixMe: If mmap fails (running out of virtual address space),
     need to unmap an existing executable.*/

exit:
  return status;
}

void unMapAicDevExecutables(AICExecContext *qctx) {
  if (headAicDevExeList == nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "%s: No AicDevExecutables mapped", __func__);
    return;
  }
  AicExeNode_t *temp = headAicDevExeList;
  AicExeNode_t *prev = nullptr;
  while (temp != nullptr) {
    if (temp != activeOpExe && (temp->exeAddrSizeInfo.isCommonLib == false)) {
      if (temp == headAicDevExeList) {
        headAicDevExeList = headAicDevExeList->next;
      } else {
        prev->next = temp->next;
      }
      LOG_DEBUG(qctx->logFuncPtr,
                "unMapAicDevExe: roID: 0x%lx, rwID: 0x%lx, exeName: %s, "
                "nodeIndex: %lu",
                temp->exeAddrSizeInfo.roBufId, temp->exeAddrSizeInfo.roBufId,
                temp->exeAddrSizeInfo.exeName,
                temp->exeAddrSizeInfo.nodeIndexExe);
      g_unmapExe(temp->exeAddrSizeInfo.nodeIndexExe,
                 (uint32_t)temp->exeAddrSizeInfo.isCommonLib);
      qctx->munmapFuncPtr(temp->exeAddrSizeInfo.nodeIndexGoldenRw);
      temp->next = nullptr;
      g_networkHeapFree((void *)temp);

      temp = (prev) ? prev->next : headAicDevExeList;
    } else {
      prev = temp;
      temp = temp->next;
    }
  }
  return;
}

void unMapUnusedAicBuffDescriptors(AICExecContext *qctx) {
  for (AicBufferDescriptorNode_t *temp = headAicBuffDescList; temp != nullptr;
       temp = temp->next) {
    if (temp->ioAddrSizeInfo.isMap && temp->ioAddrSizeInfo.inUse == false) {
      LOG_DEBUG(qctx->logFuncPtr,
                "*VA Space Exhaust* Unmapped buffer addr: 0x%llx, "
                "mapSize: 0x%x, bufId: 0x%x, inUse: 0x%x",
                LOG64_DATA(temp->ioAddrSizeInfo.buffAddr),
                temp->ioAddrSizeInfo.size, temp->ioAddrSizeInfo.bufId,
                temp->ioAddrSizeInfo.inUse);

      qctx->munmapFuncPtr(temp->ioAddrSizeInfo.nodeIndex);
      temp->ioAddrSizeInfo.isMap = false;
      temp->ioAddrSizeInfo.virtAddr = 0;
      temp->ioAddrSizeInfo.nodeIndex = 0;
    }
  }
}

void unMapAicBuffDescriptors(AICExecContext *qctx) {
  for (AicBufferDescriptorNode_t *temp = headAicBuffDescList; temp != nullptr;
       temp = temp->next) {
    if (temp->ioAddrSizeInfo.isMap) {
      LOG_DEBUG(qctx->logFuncPtr,
                "*VA Space Exhaust* Unmapped buffer addr: "
                "0x%llx, mapSize: 0x%x, bufId: 0x%x, inUse: 0x%x",
                LOG64_DATA(temp->ioAddrSizeInfo.buffAddr),
                temp->ioAddrSizeInfo.size, temp->ioAddrSizeInfo.bufId,
                temp->ioAddrSizeInfo.inUse);

      qctx->munmapFuncPtr(temp->ioAddrSizeInfo.nodeIndex);
      temp->ioAddrSizeInfo.isMap = false;
      temp->ioAddrSizeInfo.virtAddr = 0;
      temp->ioAddrSizeInfo.nodeIndex = 0;
    }
  }
}

JitDevStatusCode_t isBuffDescMapAndValid(AICExecContext *qctx, uint64_t addr,
                                         uint32_t size, uint32_t bufId,
                                         uint64_t offset,
                                         AicBufferDescriptorNode_t **node,
                                         bool *isValid, bool *isNodePresent) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  AicBufferDescriptorNode_t *aicBuffDescNode = headAicBuffDescList;
  uint64_t P16MB_PAGE = 0x1000000;
  uint64_t addrToMap = addr & ~(P16MB_PAGE - 1);

  if (node == nullptr || isValid == nullptr || isNodePresent == nullptr) {
    LOG_ERROR(qctx->logFuncPtr, "%s: Invalid input args", __func__);
    status = JIT_DEV_ERROR_INVALID_PARAMETER;
    goto exit;
  }

  *isValid = false;
  *isNodePresent = false;
  *node = nullptr;

  while (aicBuffDescNode) {
    if (aicBuffDescNode->ioAddrSizeInfo.buffAddr == addrToMap &&
        aicBuffDescNode->ioAddrSizeInfo.bufId == bufId) {
      uint32_t mapSize = aicBuffDescNode->ioAddrSizeInfo.size;
      LOG_DEBUG(qctx->logFuncPtr,
                "Mapped buffer addr: 0x%llx, mapSize: 0x%x, bufId: 0x%x",
                LOG64_DATA(aicBuffDescNode->ioAddrSizeInfo.buffAddr), mapSize,
                bufId);

      if (aicBuffDescNode->ioAddrSizeInfo.isMap == true &&
          (((addr & (P16MB_PAGE - 1)) + offset + size) < mapSize)) {
        LOG_DEBUG(qctx->logFuncPtr,
                  "Buffer overlap addr: 0x%llx, offset: 0x%llx, size: 0x%x",
                  LOG64_DATA(addr), LOG64_DATA(offset), size);
        *isValid = true;
        aicBuffDescNode->ioAddrSizeInfo.inUse = true;
      } else {
        *isValid = false;
        *isNodePresent = true;
        LOG_DEBUG(qctx->logFuncPtr,
                  "Buffer overflows block or unmapped. addr: 0x%llx, offset: "
                  "0x%llx, "
                  "size: 0x%x, mapStatus: %d",
                  LOG64_DATA(addr), LOG64_DATA(offset), size,
                  aicBuffDescNode->ioAddrSizeInfo.isMap);
      }

      *node = aicBuffDescNode;
      break;
    }
    aicBuffDescNode = aicBuffDescNode->next;
  }

exit:
  return status;
}

JitDevStatusCode_t
createAicBuffDescNode(AICExecContext *qctx, uint64_t addr, uint64_t size,
                      uint32_t bufId, uint64_t offset,
                      AicBufferDescriptorNode_t **newNode,
                      AicJit_AicMemType_enum_t buffLoc,
                      AicJit_AicBufferMappingRule_enum_t mmapRule) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  AicBufferDescriptorNode_t *aicBuffDescNode = nullptr;
  uint64_t P16MB_PAGE = 0x1000000;
  uint64_t addrToMap = addr & ~(P16MB_PAGE - 1);
  uint32_t numBlocks = 0, blockOffset = 0;
  bool isValid = false, isNodePresent = false;
  jit_heap_ddr_alloc_alp_t *chunkList = nullptr;

  if (mmapRule == AicJit_AicBufferMappingRule_BUF_MAPPING_BY_OPS) {
    aicBuffDescNode = (AicBufferDescriptorNode_t *)g_networkHeapMalloc(
        sizeof(AicBufferDescriptorNode_t));
    if (aicBuffDescNode == nullptr) {
      LOG_ERROR(qctx->logFuncPtr,
                "%s: failed to allocate memory for AicBuffDesc node", __func__);
      status = JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE;
      goto exit;
    }
    g_memset((void *)aicBuffDescNode, 0, sizeof(AicBufferDescriptorNode_t));
    aicBuffDescNode->next = headAicOpBuffDescList;
    aicBuffDescNode->mmapRule = mmapRule;
    headAicOpBuffDescList = aicBuffDescNode;

    aicBuffDescNode->opMmapBufInfo.buffAddr = addr;
    aicBuffDescNode->opMmapBufInfo.offset = offset;
    aicBuffDescNode->opMmapBufInfo.buffSize = size;
    aicBuffDescNode->opMmapBufInfo.bufId = bufId;

    status =
        getOffsetAlpBase(qctx, aicBuffDescNode->opMmapBufInfo.buffAddr, bufId,
                         &aicBuffDescNode->opMmapBufInfo.offsetAlpBase);
    if (status != JIT_DEV_STATUS_SUCCESS) {
      goto exit;
    }
  } else if (mmapRule == AicJit_AicBufferMappingRule_BUF_MAPPING_AUTO) {
    if (buffLoc == AicJit_AicMemType_DDR) {
      status =
          isBuffDescMapAndValid(qctx, addr, size, bufId, offset,
                                &aicBuffDescNode, &isValid, &isNodePresent);
      if (status != JIT_DEV_STATUS_SUCCESS) {
        goto exit;
      }

      if (!isValid) {
        if (isNodePresent == false) {
          aicBuffDescNode = (AicBufferDescriptorNode_t *)g_networkHeapMalloc(
              sizeof(AicBufferDescriptorNode_t));
          if (aicBuffDescNode == nullptr) {
            LOG_ERROR(qctx->logFuncPtr,
                      "%s: failed to allocate memory for AicBuffDesc node",
                      __func__);
            status = JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE;
            goto exit;
          }

          g_memset((void *)aicBuffDescNode, 0,
                   sizeof(AicBufferDescriptorNode_t));
          aicBuffDescNode->next = headAicBuffDescList;
          aicBuffDescNode->mmapRule = mmapRule;
          headAicBuffDescList = aicBuffDescNode;
        } else {
          if (aicBuffDescNode->ioAddrSizeInfo.isMap) {
            uint32_t mapSize = aicBuffDescNode->ioAddrSizeInfo.size;
            LOG_DEBUG(
                qctx->logFuncPtr,
                "Unmapped buffer addr: 0x%llx, mapSize: 0x%x, bufId: 0x%x",
                LOG64_DATA(aicBuffDescNode->ioAddrSizeInfo.buffAddr), mapSize,
                aicBuffDescNode->ioAddrSizeInfo.bufId);

            qctx->munmapFuncPtr(aicBuffDescNode->ioAddrSizeInfo.nodeIndex);
            aicBuffDescNode->ioAddrSizeInfo.isMap = false;
            aicBuffDescNode->ioAddrSizeInfo.nodeIndex = 0;
            aicBuffDescNode->ioAddrSizeInfo.virtAddr = 0;
            aicBuffDescNode->ioAddrSizeInfo.inUse = false;
          }
        }
        blockOffset = addr & (P16MB_PAGE - 1);
        numBlocks = (size + offset + blockOffset) / P16MB_PAGE;
        numBlocks += (size + offset + blockOffset) % P16MB_PAGE ? 1 : 0;
        if (numBlocks >= (JIT_MAX_BUF_DESC_SIZE / JIT_MMAP_BLOCK_SIZE)) {
          LOG_ERROR(qctx->logFuncPtr,
                    "%s: MMAP size greater than 4GB due to 16MB alignment",
                    __func__);
          status = JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE;
          goto exit;
        }
        aicBuffDescNode->ioAddrSizeInfo.offset = 0;
        aicBuffDescNode->ioAddrSizeInfo.buffAddr = addrToMap;
        aicBuffDescNode->ioAddrSizeInfo.size = P16MB_PAGE * numBlocks;

        status = getOffsetAlpBase(
            qctx, aicBuffDescNode->ioAddrSizeInfo.buffAddr, bufId,
            &aicBuffDescNode->ioAddrSizeInfo.offsetAlpBase);
        if (status != JIT_DEV_STATUS_SUCCESS) {
          goto exit;
        }

        aicBuffDescNode->ioAddrSizeInfo.bufId = bufId;
        aicBuffDescNode->ioAddrSizeInfo.bufferLocation = buffLoc;

        // Check if the size overflow
        chunkList = (jit_heap_ddr_alloc_alp_t *)(qctx->baseSharedDDR +
                                                 SHDDR_BUFFER_ALPS_BASE_ADDR);
        uint64_t upperBoundSize =
            aicBuffDescNode->ioAddrSizeInfo.offsetAlpBase +
            aicBuffDescNode->ioAddrSizeInfo.size;
        for (uint8_t i = 0; i < JIT_HEAP_MAX_ALPS; i++, chunkList++) {
          if (chunkList->buf_id == bufId) {
            if (upperBoundSize >= chunkList->addr_size) {
              aicBuffDescNode->ioAddrSizeInfo.size =
                  MIN(aicBuffDescNode->ioAddrSizeInfo.size,
                      chunkList->addr_size -
                          aicBuffDescNode->ioAddrSizeInfo.offsetAlpBase - 1);
              LOG_DEBUG(
                  qctx->logFuncPtr,
                  "*Block exceed allowed size. Reduce size* offsetAlpBase: "
                  "0x%llx, "
                  "mapSize: 0x%x, bufId: 0x%x",
                  LOG64_DATA(aicBuffDescNode->ioAddrSizeInfo.offsetAlpBase),
                  aicBuffDescNode->ioAddrSizeInfo.size,
                  aicBuffDescNode->ioAddrSizeInfo.bufId);
            }
          }
        }

        if (mmapAicBuffDescriptor(qctx, aicBuffDescNode)) {
          unMapUnusedAicBuffDescriptors(qctx);
          unMapAicDevExecutables(qctx);
          if (mmapAicBuffDescriptor(qctx, aicBuffDescNode)) {
            /*unmap current cmdset buffers and return MMAP error, initiate mmap
             *one more time for all buffers in cmdset*/
            unMapAicBuffDescriptors(qctx);
            status = JIT_DEV_ERROR_MMAP;
          }
        }
      }

    } else {
      tcmAicBuffDescNode.ioAddrSizeInfo.buffAddr = addr;
      tcmAicBuffDescNode.ioAddrSizeInfo.offsetAlpBase = 0;
      tcmAicBuffDescNode.ioAddrSizeInfo.offset = offset;
      tcmAicBuffDescNode.ioAddrSizeInfo.size = size;
      tcmAicBuffDescNode.ioAddrSizeInfo.bufId = bufId;
      tcmAicBuffDescNode.ioAddrSizeInfo.virtAddr = (uint32_t)qctx->baseVTCM;
      tcmAicBuffDescNode.ioAddrSizeInfo.bufferLocation = buffLoc;
      tcmAicBuffDescNode.mmapRule = mmapRule;
    }
  }

  *newNode = (buffLoc == AicJit_AicMemType_DDR) ? aicBuffDescNode
                                                : &tcmAicBuffDescNode;

  /* FixMe: If mmap fails (running out of virtual address space),
     need to unmap an existing executable.*/

exit:
  return status;
}

static int compareAicDevExeNames(const char *exeName1, const char *exeName2) {
  if (exeName1 == nullptr || exeName2 == nullptr) {
    return -1;
  }

  while ((*exeName1 != '\0') && (*exeName2 != '\0')) {
    if (*exeName1 != *exeName2) {
      return 0;
    }
    exeName1++;
    exeName2++;
  }
  if (*exeName1 == *exeName2) {
    return 1;
  }
  return 0;
}

int compareAicDevExeListNodes(AicExeNode_t *node, AicJit_DevDdrBuf_t *roSegment,
                              AicJit_DevDdrBuf_t *rwSegment,
                              AicJit_DevDdrBuf_t *goldenRwSegment) {
  if (node->exeAddrSizeInfo.roOffset == roSegment->offset &&
      node->exeAddrSizeInfo.roSize == roSegment->size &&
      node->exeAddrSizeInfo.roBufId == roSegment->bufId &&
      node->exeAddrSizeInfo.rwOffset == rwSegment->offset &&
      node->exeAddrSizeInfo.rwSize == rwSegment->size &&
      node->exeAddrSizeInfo.rwBufId == rwSegment->bufId &&
      node->exeAddrSizeInfo.goldenRwOffset == goldenRwSegment->offset &&
      node->exeAddrSizeInfo.goldenRwSize == goldenRwSegment->size &&
      node->exeAddrSizeInfo.goldenRwBufId == goldenRwSegment->bufId) {
    return 1;
  }
  return 0;
}

void removeAicDevExecutable(AICExecContext *qctx, const char *exeName,
                            AicJit_DevDdrBuf_t *roSegment,
                            AicJit_DevDdrBuf_t *rwSegment,
                            AicJit_DevDdrBuf_t *goldenRwSegment,
                            bool isCommonLib) {
  if (headAicDevExeList == nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "%s: No AicDevExecutables mapped", __func__);
    return;
  }

  AicExeNode_t *temp = headAicDevExeList;
  AicExeNode_t *prev;
  while (temp != nullptr) {
    if (compareAicDevExeListNodes(temp, roSegment, rwSegment,
                                  goldenRwSegment) &&
        compareAicDevExeNames(temp->exeAddrSizeInfo.exeName, exeName)) {
      LOG_DEBUG(qctx->logFuncPtr,
                "RemoveAicDevExe: roID: 0x%x, rwID: 0x%x, exeName: %s",
                roSegment->bufId, rwSegment->bufId, exeName);
      if (temp == headAicDevExeList) {
        headAicDevExeList = headAicDevExeList->next;
        LOG_DEBUG(qctx->logFuncPtr, "%s: Removed from the head of list",
                  __func__);
      } else {
        prev->next = temp->next;
        LOG_DEBUG(qctx->logFuncPtr,
                  "%s: Removed AicDevExeNode is not head of the list",
                  __func__);
      }
      g_unmapExe(temp->exeAddrSizeInfo.nodeIndexExe, (uint32_t)isCommonLib);
      qctx->munmapFuncPtr(temp->exeAddrSizeInfo.nodeIndexGoldenRw);
      temp->next = nullptr;
      g_networkHeapFree((void *)temp);
      if (isCommonLib) {
        commonDevExe = nullptr;
      }
      return;
    }
    prev = temp;
    temp = temp->next;
  }

  return;
}

bool isAicDevExecutableMmap(AICExecContext *qctx, const char *exeName,
                            AicJit_DevDdrBuf_t *roSegment,
                            AicJit_DevDdrBuf_t *rwSegment,
                            AicJit_DevDdrBuf_t *goldenRwSegment,
                            AicExeNode_t **newNode, bool isCommonLib) {
  if (headAicDevExeList == nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "%s: No AicDevExecutables mapped", __func__);
    *newNode = nullptr;
    return false;
  }

  AicExeNode_t *temp = headAicDevExeList;
  if (isCommonLib) {
    if (commonDevExe == nullptr)
      goto exit;
    else {
      if (compareAicDevExeListNodes(commonDevExe, roSegment, rwSegment,
                                    goldenRwSegment) &&
          compareAicDevExeNames(commonDevExe->exeAddrSizeInfo.exeName,
                                exeName)) {
        *newNode = commonDevExe;
        return true;
      } else
        goto exit;
    }
  }

  while (temp != nullptr) {
    if (compareAicDevExeListNodes(temp, roSegment, rwSegment,
                                  goldenRwSegment) &&
        compareAicDevExeNames(temp->exeAddrSizeInfo.exeName, exeName)) {
      LOG_DEBUG(qctx->logFuncPtr,
                "isAicDevExecutableMmap: DevExecutable present. roOffset: "
                "0x%llx, rwOffset: 0x%llx, exeName: %s",
                LOG64_DATA(roSegment->offset), LOG64_DATA(rwSegment->offset),
                exeName);

      *newNode = temp;
      /* Bring the node to the head of list */

      return true;
    }
    temp = temp->next;
  }

exit:
  *newNode = nullptr;
  return false;
}

void notifyOpTimeoutFunc() {
  char buf[MAX_ERROR_REPONSE_STR_LENGTH] = {'\0'};
  int tid = -1;
  if (g_getVtid != nullptr) {
    tid = (int)g_getVtid();
  }

  setErrRspStrPtr(tid, &cmdBuffRspHdr->errMsg[0]);
  char *errorString = errRspBuffPtr[tid];

  LOG_ERROR(g_qctx->logFuncPtr, " Operator Timed out on tid %d",
            g_tidFinishDbWait);
  g_memcpy(buf, (void *)"Operator Timedout", MAX_ERROR_REPONSE_STR_LENGTH);
  cmdBuffRspHdr->status = JIT_DEV_ERROR_OPERATOR_KERNEL_TIMEOUT;

  if (g_rspTlv != nullptr) {
    ResponseTLV_t *rspTlv = g_rspTlv;
    CommandValue_t *cmdVal = (CommandValue_t *)&rspTlv->value[0];
    cmdVal->commandID = g_opErrCommandId;
    cmdBuffRspHdr->tlvCount = rspTlv->slotCount;
    cmdVal->status = JIT_DEV_ERROR_OPERATOR_KERNEL_TIMEOUT;
  }

  if (errorString == nullptr) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "memory is not allocated for reporting error string");
    return;
  }

  g_snprintf(errorString, MAX_ERROR_REPONSE_STR_LENGTH, "NSP %d: TID %d : %s",
             g_qctx->virtualNSPId, g_tidFinishDbWait, buf);
  if (g_qctx->virtualNSPId == 0) {
    uint16_t exitDb = getExitDbNum();
    LOG_DEBUG(g_qctx->logFuncPtr, "%s: set error db ", __func__);
    qcDoorbellWrite(L2TCM_ERR_DB_OFFSET, 1);
    qcDoorbellWrite(exitDb, 1);
    errExitThreadHandler();
  } else {
    setMasterNspErrorDb();
    g_exitFp();
  }
}

int mmapOpBufferNode(AICExecContext *qctx, OpMmapNode_t *opMmapNode,
                     uint32_t bufId) {
  int status = 0;
  void *virtAddr;
  uint32_t nodeIdx;

  uint64_t offset = opMmapNode->opMmapNodeInfo.mapOffsetAlpBase;
  uint32_t size = opMmapNode->opMmapNodeInfo.mapSize;

  if (g_mmap) {
    int retVal = g_mmap(&virtAddr, &nodeIdx, NNC_MMAP_JIT_ADDR_TYPE_IO,
                        JIT_IO_BUF_MMAP_NUM_ARGS, offset, bufId, size);
    if (retVal) {
      LOG_ERROR(qctx->logFuncPtr,
                "opBufNode failed to mmap bufId: 0x%x, size: 0x%x, "
                "offset: 0x%llx, retVal: %d",
                bufId, size, LOG64_DATA(offset), retVal);
      status = JIT_DEV_ERROR_MMAP;
    } else {
      opMmapNode->opMmapNodeInfo.virtAddr = (uint32_t)virtAddr;
      opMmapNode->opMmapNodeInfo.nodeIndex = nodeIdx;
      opMmapNode->opMmapNodeInfo.isMap = true;
      LOG_DEBUG(qctx->logFuncPtr,
                "opBufNode mmap successfully offset: 0x%llx, bufId: 0x%x, "
                "virtAddr: 0x%x, nodeIdx: %d",
                LOG64_DATA(offset), bufId, opMmapNode->opMmapNodeInfo.virtAddr,
                nodeIdx);
    }
  }

  return status;
}

void dispatchWorkerThreads(int32_t tid, AICExecContext *qctx) {
  int retVal = 0;
  qcDoorbellWrite(L2TCM_WORKER_THREADS_START_DB_OFFSET + tid, 0);
  AicJitEntryPointConfig entryConfig = {.coreID = qctx->virtualNSPId,
                                        .numCores = g_numCores,
                                        .threadID = (uint32_t)tid,
                                        .numThreads = g_numThreads};
  entryPointFunc = (AicJitDevExeFunction)entryPointFuncAddr;
  entryPointFunc(&entryConfig, listParamPtr);
  retVal = g_qurtHvxSet(QURT_HVX_RELEASE_CONTEXT);
  if (retVal) {
    LOG_ERROR(qctx->logFuncPtr,
              "%s:HVX release failure on tid: %lu with error: %lu", __func__,
              tid, retVal);
  }
  qcDoorbellWrite(L2TCM_WORKER_THREADS_FINISH_DB_OFFSET + tid, 1);
}

static uint32_t countSetBits(uint32_t n) {
  uint32_t count = 0;
  while (n) {
    n &= (n - 1);
    count++;
  }
  return count;
}

JitDevStatusCode_t
processAicCommandDispatch(AICExecContext *qctx, uint32_t commandID,
                          AicJit_AicCommandSet_table_t commandSetBuffer,
                          void *buffer, uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  uint64_t funcOffset = 0, buffAddr = 0;
  uint32_t exeIndex = 0, threadBitmap = 0, index = 0, blockOffset = 0;
  AicJit_AicParameterType_enum_t aicParamType;
  AicExeNode_t *exeNode = nullptr;

  AicJit_AicCmdDispatch_table_t aicCmdDispatch =
      (AicJit_AicCmdDispatch_table_t)buffer;
  if (AicJit_AicCmdDispatch_funcOffset_is_present(aicCmdDispatch)) {
    funcOffset = AicJit_AicCmdDispatch_funcOffset(aicCmdDispatch);
    LOG_DEBUG(qctx->logFuncPtr, "AicCmdDispatch: funcOffset: 0x%llx",
              LOG64_DATA(funcOffset));
  }

  if (AicJit_AicCmdDispatch_exeIndex_is_present(aicCmdDispatch)) {
    exeIndex = AicJit_AicCmdDispatch_exeIndex(aicCmdDispatch);
    LOG_DEBUG(qctx->logFuncPtr, "%s: exeIndex: %d", __func__, exeIndex);
  }

  if (AicJit_AicCmdDispatch_numCores_is_present(aicCmdDispatch)) {
    g_numCores = AicJit_AicCmdDispatch_numCores(aicCmdDispatch);
    LOG_DEBUG(qctx->logFuncPtr, "AicCmdDispatch: numCores: %d", g_numCores);
  }

  /* Calculate the entry point function address */
  if (exeIndex >= numLibAicDevExes) {
    qshimLog(NNC_LOG_MASK_ERROR, 3,
             "%s: Invalid exeIndex. index: %d, numLibAicDevExes: 0x%x",
             __func__, exeIndex, numLibAicDevExes);
    status = JIT_DEV_ERROR_INVALID_PARAMETER;
    goto exit;
  }

  exeNode = (AicExeNode_t *)libAicDevExeBaseAddress[exeIndex];
  if (exeNode->exeAddrSizeInfo.isCommonLib) {
    entryPointFuncAddr = (uint32_t)funcOffset;
  } else {
    entryPointFuncAddr =
        exeNode->exeAddrSizeInfo.virtAddrExe + (uint32_t)funcOffset;
  }
  LOG_DEBUG(qctx->logFuncPtr, "%s: entryPointFuncAddr: 0x%x", __func__,
            entryPointFuncAddr);

  setEntryPointFuncAddr(entryPointFuncAddr);

  if (AicJit_AicCmdDispatch_threadBitmap_is_present(aicCmdDispatch)) {
    threadBitmap = AicJit_AicCmdDispatch_threadBitmap(aicCmdDispatch);
    threadIdBitMap = threadBitmap;
    LOG_DEBUG(qctx->logFuncPtr, "%s: threadBitmap: 0x%x", __func__,
              threadBitmap);
    g_numThreads = countSetBits(threadBitmap);
    LOG_DEBUG(qctx->logFuncPtr, "%s: Num Threads: 0x%x", __func__,
              g_numThreads);
  }

  if (AicJit_AicCmdDispatch_argList_is_present(aicCmdDispatch)) {
    uint32_t sizeArgList = AicJit_AicParameter_vec_len(
        AicJit_AicCmdDispatch_argList(aicCmdDispatch));
    LOG_DEBUG(qctx->logFuncPtr, "%s: sizeArgList: 0x%x", __func__, sizeArgList);

    listParamPtr = (AicJitPointerArray *)g_networkHeapMalloc(
        sizeof(AicJitPointerArray) + sizeof(void *) * (sizeArgList + 1));
    if (listParamPtr == nullptr) {
      qshimLog(NNC_LOG_MASK_ERROR, 1,
               "%s: failed to allocate memory for list of param pointers",
               __func__);
      status = JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE;
      goto exit;
    }

    listParamPtr->numPointers = sizeArgList;

    for (uint32_t i = 0; i < sizeArgList; i++) {
      AicJit_AicParameter_struct_t aicParam = AicJit_AicParameter_vec_at(
          AicJit_AicCmdDispatch_argList(aicCmdDispatch), i);

      if (AicJit_AicParameterType_is_known_value(
              AicJit_AicParameter_type(aicParam))) {
        aicParamType = AicJit_AicParameter_type(aicParam);
        LOG_DEBUG(
            qctx->logFuncPtr, "%s: aicParamType: %s", __func__,
            AicJit_AicParameterType_name(AicJit_AicParameter_type(aicParam)));

      } else {
        qshimLog(NNC_LOG_MASK_ERROR, 1,
                 "Unknown aicParamType: %d in AicParameterType",
                 AicJit_AicParameter_type(aicParam));
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      /* For LocalStorage, the index is the first byte into the
         local storage for this parameter. */
      /* For BufferDescriptor, the index is the index of the descriptor. */
      index = AicJit_AicParameter_index(aicParam);
      LOG_DEBUG(qctx->logFuncPtr, "%s: index: 0x%x", __func__, index);

      if (aicParamType == AicJit_AicParameterType_BUFFER_DESCRIPTOR) {
        blockOffset = 0;
        AicJit_AicBufferDescriptor_struct_t bufferDesc =
            AicJit_AicBufferDescriptor_vec_at(
                AicJit_AicCommandSet_bufferDescriptors(commandSetBuffer),
                index);

        if (AicJit_AicBufferDescriptor_mappingRule(bufferDesc) ==
            AicJit_AicBufferMappingRule_BUF_MAPPING_AUTO) {
          buffAddr = AicJit_AicBufferDescriptor_addr(bufferDesc);
          if (listAicDevBuffDescNodes[index]->ioAddrSizeInfo.bufferLocation ==
              AicJit_AicMemType_DDR) {
            blockOffset = buffAddr & 0xFFFFFF;
          }
          listParamPtr->pointers[i] =
              (void *)(listAicDevBuffDescNodes[index]->ioAddrSizeInfo.virtAddr +
                       blockOffset +
                       AicJit_AicBufferDescriptor_offset(bufferDesc));
        } else {
          listParamPtr->pointers[i] = (void *)(listAicDevBuffDescNodes[index]);
        }
      } else if (aicParamType == AicJit_AicParameterType_LOCAL_STORAGE) {
        if (index >= localStorageSize) {
          qshimLog(NNC_LOG_MASK_ERROR, 3,
                   "%s: Invalid local storage index. index: %d, "
                   "localStorageSize: 0x%x",
                   __func__, index, localStorageSize);
          status = JIT_DEV_ERROR_INVALID_PARAMETER;
          goto exit;
        }
        listParamPtr->pointers[i] = (void *)(localStorageBaseAddress + index);
      }
      LOG_DEBUG(qctx->logFuncPtr, "%s: listParamPtr[%d]: 0x%x", __func__, i,
                listParamPtr->pointers[i]);
    }
  }

exit:
  return status;
}

JitDevStatusCode_t processAicCommandAllocBuf(AICExecContext *qctx,
                                             uint32_t commandID, void *buffer,
                                             uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  jit_ddr_malloc_params ddrMallocParams = {0};
  uint64_t size = 0;
  uint32_t bufId = 0;
  void *ptr = nullptr;
  uint8_t numAlps = 0;
  int retVal = 0;

  AicJit_AicCmdAllocBuf_table_t aicCmdAllocBuf =
      (AicJit_AicCmdAllocBuf_table_t)buffer;
  if (AicJit_AicCmdAllocBuf_size_is_present(aicCmdAllocBuf)) {
    size = AicJit_AicCmdAllocBuf_size(aicCmdAllocBuf);
    LOG_DEBUG(qctx->logFuncPtr, "%s: size: 0x%llx", __func__, size);
  } else {
    qshimLog(NNC_LOG_MASK_ERROR, 0, "size field not present in AicCmdAllocBuf");
    status = JIT_DEV_ERROR_INVALID_PARAMETER;
    goto exit;
  }

  if (AicJit_AicCmdAllocBuf_pid_is_present(aicCmdAllocBuf)) {
    if (isHostPidSet == false) {
      hostPid = AicJit_AicCmdAllocBuf_pid(aicCmdAllocBuf);
      isHostPidSet = true;
    }
    NN_LOG(qctx->logFuncPtr, NNC_LOG_MASK_INFO, "%s: hostPid: 0x%x", __func__,
           AicJit_AicCmdAllocBuf_pid(aicCmdAllocBuf));
  } else {
    qshimLog(NNC_LOG_MASK_ERROR, 0, "pid field not present in AicCmdAllocBuf");
    status = JIT_DEV_ERROR_INVALID_PARAMETER;
    goto exit;
  }

  ddrMallocParams.size = size;
  ddrMallocParams.host_pid = hostPid;
  ddrMallocParams.reserve_size = 0; // Switch to vcQueueMemsize when using DDR
  ddrMallocParams.num_alps = &numAlps;
  ddrMallocParams.reserve_alloc = &g_vcQueueDdrAllocInfo;
  ddrMallocParams.ddr_alloc_alps = &g_allocAlps[0];
  retVal = g_ddrMalloc(&ddrMallocParams);
  if (retVal) {
    qshimLog(NNC_LOG_MASK_ERROR, 2,
             "DDR Memory allocation failed size: 0x%llx, status: %d",
             LOG64_DATA(size), retVal);
    status = JIT_DEV_ERROR_MEM_ALLOC_FAILURE;
  } else {
    LOG_DEBUG(qctx->logFuncPtr,
              "DDR Memory allocation successful. Num ALPs: 0x%x",
              LOG64_DATA(numAlps));
  }

exit:
  if (status == JIT_DEV_STATUS_SUCCESS && g_rspTlv != nullptr) {
    ResponseTLV_t *rspTlv = g_rspTlv;
    rspTlv->slotCount = 2;
    CommandValue_t *cmdVal = (CommandValue_t *)&rspTlv->value[0];

    LOG_DEBUG(qctx->logFuncPtr, "%s: rspTlv: 0x%x, status: 0x%x", __func__,
              rspTlv, status);
    DevMemChunkList_t *allocBufData = (DevMemChunkList_t *)&cmdVal->data[0];
    LOG_DEBUG(qctx->logFuncPtr, "%s: val: 0x%x, allocBufData: 0x%x", __func__,
              cmdVal, allocBufData);
    allocBufData->numChunks = numAlps;
    DevMemChunk_t *chunk = allocBufData->chunks;
    for (uint8_t i = 0; i < numAlps; i++) {
      chunk->bufId = g_allocAlps[i].buf_id;
      chunk->phy_addr = g_allocAlps[i].addr_base;
      chunk->size = g_allocAlps[i].addr_size;
      chunk++;
    }
    // copy the ALPs to DDR to make it accessible to other NSPs
    g_memcpy((void *)(qctx->baseSharedDDR + SHDDR_BUFFER_ALPS_BASE_ADDR),
             (void *)g_allocAlps,
             (JIT_HEAP_MAX_ALPS * sizeof(jit_heap_ddr_alloc_alp_t)));
  }
  return status;
}

JitDevStatusCode_t processAicCommandFreeBuf(AICExecContext *qctx,
                                            uint32_t commandID, void *buffer,
                                            uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  uint8_t numAlps = 0;
  uint32_t bufIdList[JIT_HEAP_MAX_ALPS] = {0};

  AicJit_AicCmdFreeBuf_table_t aicCmdFreeBuf =
      (AicJit_AicCmdFreeBuf_table_t)buffer;
  if (AicJit_AicCmdFreeBuf_bufferIdList_is_present(aicCmdFreeBuf)) {
    numAlps = flatbuffers_uint32_vec_len(
        AicJit_AicCmdFreeBuf_bufferIdList(aicCmdFreeBuf));
    LOG_DEBUG(qctx->logFuncPtr, "AicCmdFreeBuf: Number of ALPs: 0x%x", numAlps);
    for (uint8_t i = 0; i < numAlps; i++) {
      bufIdList[i] = flatbuffers_uint32_vec_at(
          AicJit_AicCmdFreeBuf_bufferIdList(aicCmdFreeBuf), i);
    }
  } else {
    qshimLog(NNC_LOG_MASK_ERROR, 0,
             "bufferIdList field not present in AicCmdFreeBuf");
    status = JIT_DEV_ERROR_INVALID_PARAMETER;
    goto exit;
  }

  if (g_ddrFree != nullptr) {
    int retVal = g_ddrFree(hostPid, &bufIdList[0], numAlps);
    LOG_DEBUG(qctx->logFuncPtr, "%s: Buffer free status: 0x%x", __func__,
              retVal);
    if (retVal) {
      qshimLog(NNC_LOG_MASK_ERROR, 2, "%s: Failed to free buffer status: 0x%x",
               __func__, retVal);
      status = JIT_DEV_ERROR_BUFFER_FREE_FAILURE;
      goto exit;
    }
  }

exit:

  return status;
}

JitDevStatusCode_t processCommandSetAicBufferDescriptor(
    AICExecContext *qctx, AicJit_AicCommandSet_table_t buffer, uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  AicJit_AicMemType_enum_t memType = AicJit_AicMemType_NONE;
  AicBufferDescriptorNode_t *newNode = nullptr;
  AicBufferDescriptorNode_t *buffNode = nullptr;
  uint64_t addr = 0, offset = 0, size = 0;
  uint32_t bufId = 0, mmapFailCount = 0;
  void *virtAddr = nullptr;
  uint32_t nodeIdx = 0;
  AicBufferDescriptorNode_t *aicBuffDescNode = headAicBuffDescList;
  AicJit_AicBufferMappingRule_enum_t mappingRule =
      AicJit_AicBufferMappingRule_BUF_MAPPING_AUTO;
  bool opMapPresent = false;

  while (aicBuffDescNode) {
    aicBuffDescNode->ioAddrSizeInfo.inUse = false;
    aicBuffDescNode = aicBuffDescNode->next;
  }

  // Update the aicBuffDescNode to point to the head of operator
  // buffer descriptor list.
  aicBuffDescNode = headAicOpBuffDescList;
  while (aicBuffDescNode) {
    OpMmapNode_t *opMmapNode = aicBuffDescNode->opMmapBufInfo.opMmapNode;
    while (opMmapNode) {
      if (opMmapNode->opMmapNodeInfo.opMmapCount) {
        LOG_WARN(qctx->logFuncPtr,
                 "%s: Operator mmap count is not zero. Operator missed an "
                 "unmap call",
                 __func__);
      } else {
        // Error scenario where the operator has called unmap for all
        // mmap calls but the node is still in the list. Ideally, this
        // scenario should not be observed unless the node is not released
        // successfully in Qshim API once all the unmap calls are complete
        // from the operator.
        LOG_WARN(qctx->logFuncPtr,
                 "%s: Possible error in Qshim API. Qshim API shall release the "
                 "node if the opMmapCount is zero",
                 __func__);
      }
      if (opMmapNode->opMmapNodeInfo.isMap) {
        qctx->munmapFuncPtr(opMmapNode->opMmapNodeInfo.nodeIndex);
      }
      OpMmapNode_t *tempMapNode = opMmapNode;
      opMmapNode = opMmapNode->next;
      tempMapNode->next = nullptr;
      g_networkHeapFree(tempMapNode);
    }
    // Release the memory for the node
    AicBufferDescriptorNode_t *temp = aicBuffDescNode;
    aicBuffDescNode = aicBuffDescNode->next;
    temp->next = nullptr;
    g_networkHeapFree(temp);
  }

  uint32_t numAicBufferDescriptors = AicJit_AicBufferDescriptor_vec_len(
      AicJit_AicCommandSet_bufferDescriptors(buffer));

  if (listAicDevBuffDescNodes == nullptr) {
    listAicDevBuffDescNodes = (AicBufferDescriptorNode_t **)g_networkHeapMalloc(
        sizeof(AicBufferDescriptorNode_t *) * numAicBufferDescriptors);
    if (listAicDevBuffDescNodes == nullptr) {
      qshimLog(NNC_LOG_MASK_ERROR, 1,
               "%s: failed to allocate memory for list of AicBuffDesc nodes",
               __func__);
      status = JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE;
      goto exit;
    }
  }

node_create_loop:
  for (uint32_t i = 0; i < numAicBufferDescriptors; i++) {
    AicJit_AicBufferDescriptor_struct_t bufferDesc =
        AicJit_AicBufferDescriptor_vec_at(
            AicJit_AicCommandSet_bufferDescriptors(buffer), i);

    if (AicJit_AicMemType_is_known_value(
            AicJit_AicBufferDescriptor_type(bufferDesc)) &&
        AicJit_AicBufferDescriptor_type(bufferDesc) != AicJit_AicMemType_NONE &&
        AicJit_AicBufferDescriptor_type(bufferDesc) != AicJit_AicMemType_HOST) {
      LOG_DEBUG(
          qctx->logFuncPtr, "%s: memType: %s", __func__,
          AicJit_AicMemType_name(AicJit_AicBufferDescriptor_type(bufferDesc)));
      memType = AicJit_AicBufferDescriptor_type(bufferDesc);

    } else {
      qshimLog(NNC_LOG_MASK_ERROR, 1,
               "Unknown memType: %d in buffer descriptor",
               AicJit_AicBufferDescriptor_type(bufferDesc));
      status = JIT_DEV_ERROR_INVALID_PARAMETER;
      goto exit;
    }

    if (AicJit_AicBufferMappingRule_is_known_value(
            AicJit_AicBufferDescriptor_mappingRule(bufferDesc))) {
      LOG_DEBUG(qctx->logFuncPtr, "%s: mappingRule: %s", __func__,
                AicJit_AicBufferMappingRule_name(
                    AicJit_AicBufferDescriptor_mappingRule(bufferDesc)));
      mappingRule = AicJit_AicBufferDescriptor_mappingRule(bufferDesc);

    } else {
      qshimLog(NNC_LOG_MASK_ERROR, 1,
               "Unknown mappingRule: %d in buffer descriptor",
               AicJit_AicBufferDescriptor_mappingRule(bufferDesc));
      status = JIT_DEV_ERROR_INVALID_PARAMETER;
      goto exit;
    }
    if (mappingRule == AicJit_AicBufferMappingRule_BUF_MAPPING_BY_OPS) {
      if (memType == AicJit_AicMemType_TCM) {
        qshimLog(NNC_LOG_MASK_ERROR, 1, "MappingRule: %d not applicable to TCM",
                 AicJit_AicBufferDescriptor_mappingRule(bufferDesc));
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }
      if (opMapPresent == false) {
        unMapAicBuffDescriptors(qctx);
        opMapPresent = true;
        goto node_create_loop;
      }
    }

    addr = AicJit_AicBufferDescriptor_addr(bufferDesc);
    LOG_DEBUG(qctx->logFuncPtr, "AicBufferDescriptor: addr: 0x%llx",
              LOG64_DATA(addr));

    offset = AicJit_AicBufferDescriptor_offset(bufferDesc);
    LOG_DEBUG(qctx->logFuncPtr, "AicBufferDescriptor: offset: 0x%llx",
              LOG64_DATA(offset));

    size = AicJit_AicBufferDescriptor_size(bufferDesc);
    LOG_DEBUG(qctx->logFuncPtr, "AicBufferDescriptor: size: 0x%llx",
              LOG64_DATA(size));

    bufId = AicJit_AicBufferDescriptor_bufId(bufferDesc);
    LOG_DEBUG(qctx->logFuncPtr, "AicBufferDescriptor: bufId: 0x%x",
              LOG64_DATA(bufId));

    // check if buffer size is greater than 4GB
    if (size >= JIT_MAX_BUF_DESC_SIZE) {
      LOG_ERROR(qctx->logFuncPtr,
                "%s: Buffer descriptor MMAP size greater than 4GB", __func__);
      status = JIT_DEV_ERROR_INVALID_PARAMETER;
      goto exit;
    }

    status =
        createAicBuffDescNode(qctx, AicJit_AicBufferDescriptor_addr(bufferDesc),
                              AicJit_AicBufferDescriptor_size(bufferDesc),
                              AicJit_AicBufferDescriptor_bufId(bufferDesc),
                              AicJit_AicBufferDescriptor_offset(bufferDesc),
                              &newNode, memType, mappingRule);
    if (status == JIT_DEV_STATUS_SUCCESS) {
      listAicDevBuffDescNodes[i] = newNode;
      numAicDevBuffDescNodes++;
      LOG_DEBUG(qctx->logFuncPtr,
                "%s: buffer descriptor node added. node address: 0x%x",
                __func__, listAicDevBuffDescNodes[i]);
    } else if (status == JIT_DEV_ERROR_MMAP) {
      LOG_ERROR(qctx->logFuncPtr,
                "%s: MMAP error.retry mapping current cmdset buffers",
                __func__);
      if (!mmapFailCount) {
        mmapFailCount++;
        goto node_create_loop;
      } else {
        status = JIT_DEV_ERROR_MMAP;
        goto exit;
      }
    } else {
      qshimLog(NNC_LOG_MASK_ERROR, 1,
               "%s: failed to add buffer descriptor node", __func__);
      status = JIT_DEV_ERROR_INVALID_PARAMETER;
      goto exit;
    }
  }

exit:
  return status;
}

JitDevStatusCode_t processCommandSetLocalStorage(
    AICExecContext *qctx, AicJit_AicCommandSet_table_t buffer, uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;

  if (AicJit_AicCommandSet_localStorage_is_present(buffer)) {
    LOG_DEBUG(qctx->logFuncPtr, "%s: Local Storage", __func__);

    localStorageSize =
        flatbuffers_uint8_vec_len(AicJit_AicCommandSet_localStorage(buffer));

    if (localStorageSize) {
      LOG_DEBUG(qctx->logFuncPtr, "%s: localStorageSize: 0x%x", __func__,
                localStorageSize);

#if 0
      uint8_t *localBuffer = (uint8_t *)g_networkHeapMalloc(localStorageSize);
      if (localBuffer == nullptr)
      {
        LOG_ERROR(qctx->logFuncPtr,
               "%s: failed to allocate memory for local storage", __func__);
        status = JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE;
        goto exit;
      }

      for (uint32_t i = 0; i < localStorageSize; i++)
      {
        *(localBuffer + i) = flatbuffers_uint8_vec_at(
                                                      AicJit_AicCommandSet_localStorage(buffer), i);
        LOG_DEBUG(qctx->logFuncPtr,
               "%s: localStorage[%d]: 0x%x", __func__, i, *(localBuffer + i));
      }
#endif

      const unsigned char *localStorageBuffAddr =
          AicJit_AicCommandSet_localStorage_get(buffer);
      localStorageBaseAddress = (uint8_t *)localStorageBuffAddr;
      for (uint32_t i = 0; i < localStorageSize; i++) {
        LOG_DEBUG(qctx->logFuncPtr, "%s: localStorage[%d]: 0x%x", __func__, i,
                  localStorageBuffAddr[i]);
      }
    } else {
      LOG_DEBUG(qctx->logFuncPtr, "%s: Local Storage size is 0.", __func__);
    }
  } else {
    LOG_DEBUG(qctx->logFuncPtr, "%s: Local Storage not present", __func__);
  }

exit:
  return status;
}

static int devLibMemCacheClean(AICExecContext *qctx) {
  AicExeNode_t *temp = headAicDevExeList;
  int ret = 0;
  while (temp != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, " %s: Dev Library %s Cache clean up", __func__,
              temp->exeAddrSizeInfo.exeName);
    ret = g_qurtMemCacheClean(temp->exeAddrSizeInfo.virtAddrExe +
                                  temp->exeAddrSizeInfo.roSize,
                              temp->exeAddrSizeInfo.rwSize,
                              QURT_MEM_CACHE_INVALIDATE, QURT_MEM_ICACHE);
    if (ret != 0) {
      LOG_ERROR(qctx->logFuncPtr,
                " %s: Dev Lib RW segment icache clean failed with status %d",
                __func__, ret);
      return ret;
    }
    ret = g_qurtMemCacheClean(temp->exeAddrSizeInfo.virtAddrExe +
                                  temp->exeAddrSizeInfo.roSize,
                              temp->exeAddrSizeInfo.rwSize,
                              QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (ret != 0) {
      LOG_ERROR(qctx->logFuncPtr,
                " %s: Dev Lib RW segment dcache clean failed with status %d",
                __func__, ret);
      return ret;
    }

    ret = g_qurtMemCacheClean(temp->exeAddrSizeInfo.virtAddrExe,
                              temp->exeAddrSizeInfo.roSize,
                              QURT_MEM_CACHE_INVALIDATE, QURT_MEM_ICACHE);
    if (ret != 0) {
      LOG_ERROR(qctx->logFuncPtr,
                " %s: Dev Lib RO segment icache clean failed with status %d",
                ret);
      return ret;
    }
    ret = g_qurtMemCacheClean(temp->exeAddrSizeInfo.virtAddrExe,
                              temp->exeAddrSizeInfo.roSize,
                              QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (ret != 0) {
      LOG_ERROR(qctx->logFuncPtr,
                " %s: Dev Lib RO segment dcache clean failed with status %d",
                ret);
      return ret;
    }

    ret = g_qurtMemCacheClean(temp->exeAddrSizeInfo.virtAddrGoldenRw,
                              temp->exeAddrSizeInfo.goldenRwSize,
                              QURT_MEM_CACHE_INVALIDATE, QURT_MEM_ICACHE);
    if (ret != 0) {
      LOG_ERROR(
          qctx->logFuncPtr,
          "%s: Dev Lib GoldenRW segment icache clean failed with status %d",
          __func__, ret);
      return ret;
    }
    ret = g_qurtMemCacheClean(temp->exeAddrSizeInfo.virtAddrGoldenRw,
                              temp->exeAddrSizeInfo.goldenRwSize,
                              QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (ret != 0) {
      LOG_ERROR(
          qctx->logFuncPtr,
          "%s: Dev Lib GoldenRW segment dcache clean failed with status %d",
          __func__, ret);
      return ret;
    }
    temp = temp->next;
  }

  asm volatile("isync \n"
               "syncht \n");

  return ret;
}

JitDevStatusCode_t processCommandSetAicInit(AICExecContext *qctx,
                                            AicJit_AicCommandSet_table_t buffer,
                                            uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  AicJit_AicBuffer_struct_t alp;
  AicExeNode_t *newNode = nullptr;
  uint64_t aicExecInfoOffset = 0;
  bool isDevExeMmap = false;

  if (AicJit_AicCommandSet_init_is_present(buffer)) {
    LOG_DEBUG(qctx->logFuncPtr, "%s: AicInit", __func__);

    AicJit_AicInit_table_t aicInit = AicJit_AicCommandSet_init(buffer);

    if (aicInit) {
      if (AicJit_AicInit_execInfoList_is_present(aicInit)) {
        uint32_t numExes =
            AicJit_ExecInfo_vec_len(AicJit_AicInit_execInfoList(aicInit));

        if (libAicDevExeBaseAddress == nullptr) {
          libAicDevExeBaseAddress =
              (uint32_t *)g_networkHeapMalloc(numExes * sizeof(uint32_t));
          if (libAicDevExeBaseAddress == nullptr) {
            qshimLog(NNC_LOG_MASK_ERROR, 1,
                     "%s: failed to allocate memory for list of AicDevExes",
                     __func__);
            status = JIT_DEV_ERROR_LOCAL_MEM_ALLOC_FAILURE;
            goto exit;
          }
        }

        for (uint32_t i = 0; i < numExes; i++) {
          AicJit_ExecInfo_table_t execInfo =
              AicJit_ExecInfo_vec_at(AicJit_AicInit_execInfoList(aicInit), i);
          // Search if executable node already exists
          // If not, then add to the list
          // Will you have to maintain a record of all exe addresses
          // that are mapped until now ? - Probably yes

          AicJit_DevDdrBuf_t *roSegment =
              (AicJit_DevDdrBuf_t *)AicJit_ExecInfo_roSegmentBuf(execInfo);
          AicJit_DevDdrBuf_t *rwSegment =
              (AicJit_DevDdrBuf_t *)AicJit_ExecInfo_rwSegmentBuf(execInfo);
          AicJit_DevDdrBuf_t *goldenRwSegment =
              (AicJit_DevDdrBuf_t *)AicJit_ExecInfo_goldenRWsegmentBuf(
                  execInfo);
          bool isCommonLib = AicJit_ExecInfo_isCommonLib(execInfo);
          qshimLog(NNC_LOG_MASK_DEBUG, 1,
                   "%s: device executable is common lib %d", __func__,
                   isCommonLib);

          isDevExeMmap = isAicDevExecutableMmap(
              qctx, AicJit_ExecInfo_exeName(execInfo), roSegment, rwSegment,
              goldenRwSegment, &newNode, isCommonLib);

          setExeName((char *)AicJit_ExecInfo_exeName(execInfo), isCommonLib);

          if (!isDevExeMmap) {
            if (isCommonLib && commonDevExe) {
              AicExeNode_t *temp = commonDevExe;
              AicJit_DevDdrBuf_t cmnRoSeg = {
                  .offset = temp->exeAddrSizeInfo.roOffset,
                  .size = temp->exeAddrSizeInfo.roSize,
                  .bufId = temp->exeAddrSizeInfo.roBufId};
              AicJit_DevDdrBuf_t cmnRwSeg = {
                  .offset = temp->exeAddrSizeInfo.rwOffset,
                  .size = temp->exeAddrSizeInfo.rwSize,
                  .bufId = temp->exeAddrSizeInfo.rwBufId};
              AicJit_DevDdrBuf_t cmnGoldenRwSeg = {
                  .offset = temp->exeAddrSizeInfo.goldenRwOffset,
                  .size = temp->exeAddrSizeInfo.goldenRwSize,
                  .bufId = temp->exeAddrSizeInfo.goldenRwBufId};

              if (devLibMemCacheClean(qctx) != 0) {
                qshimLog(NNC_LOG_MASK_ERROR, 1,
                         "%s: Cache clean for Operator Library failed",
                         __func__);
                status = JIT_DEV_ERROR_CACHE_CLEANUP;
                goto exit;
              }
              removeAicDevExecutable(qctx, &(temp->exeAddrSizeInfo.exeName[0]),
                                     &cmnRoSeg, &cmnRwSeg, &cmnGoldenRwSeg,
                                     true);
            }

            if (mmapAicDevExecutable(qctx, AicJit_ExecInfo_exeName(execInfo),
                                     roSegment, rwSegment, goldenRwSegment,
                                     &newNode, isCommonLib)) {
              unMapAicBuffDescriptors(qctx);
              unMapAicDevExecutables(qctx);
              status = mmapAicDevExecutable(
                  qctx, AicJit_ExecInfo_exeName(execInfo), roSegment, rwSegment,
                  goldenRwSegment, &newNode, isCommonLib);
            }

            if (status != JIT_DEV_STATUS_SUCCESS) {
              qshimLog(NNC_LOG_MASK_ERROR, 1,
                       "%s: failed to mmap device executable", __func__);
              goto exit;
            }
          }

          if (!isCommonLib) {
            activeOpExe = newNode;
          }
          libAicDevExeBaseAddress[i] = (uint32_t)newNode;
          numLibAicDevExes++;

          if (!isCommonLib) {
            newNode->exeAddrSizeInfo.numGotPltOffsets =
                flatbuffers_uint32_vec_len(
                    AicJit_ExecInfo_gotPltOffsets(execInfo));

            if (newNode->exeAddrSizeInfo.numGotPltOffsets >
                MAX_GOT_PLT_OFFSETS) {
              qshimLog(NNC_LOG_MASK_ERROR, 1,
                       "%s: GotPlt offsets exceeds the allocated array size",
                       __func__);
              status = JIT_DEV_ERROR_GOTPLT_SIZE_OUT_OF_BOUND;
              goto exit;
            }

            for (uint32_t i = 0; i < newNode->exeAddrSizeInfo.numGotPltOffsets;
                 i++) {
              uint32_t execOffset = flatbuffers_uint32_vec_at(
                  AicJit_ExecInfo_gotPltOffsets(execInfo), i);
              newNode->exeAddrSizeInfo.gotPltOffsets[i] = execOffset;
            }
          }
        }

        for (AicExeNode_t *temp = headAicDevExeList; temp != nullptr;
             temp = temp->next) {
          if (temp == activeOpExe || temp == commonDevExe) {
            // refresh RW segment from golden RW
            g_memcpy((void *)(temp->exeAddrSizeInfo.virtAddrExe +
                              temp->exeAddrSizeInfo.roSize),
                     (void *)temp->exeAddrSizeInfo.virtAddrGoldenRw,
                     temp->exeAddrSizeInfo.goldenRwSize);

            if (!temp->exeAddrSizeInfo.isCommonLib) {
              for (uint32_t i = 0; i < temp->exeAddrSizeInfo.numGotPltOffsets;
                   i++) {
                uint32_t execOffset = temp->exeAddrSizeInfo.gotPltOffsets[i];
                uint32_t *devExeGotPltEntryAddr =
                    (uint32_t *)(temp->exeAddrSizeInfo.virtAddrExe +
                                 execOffset);
                *devExeGotPltEntryAddr += temp->exeAddrSizeInfo.virtAddrExe;
                LOG_DEBUG(qctx->logFuncPtr,
                          "%s: Updated address 0x%x with value 0x%x", __func__,
                          devExeGotPltEntryAddr, *devExeGotPltEntryAddr);
              }
            }
          }
        }
      } else {
        LOG_DEBUG(qctx->logFuncPtr, "%s: execInfoList not present", __func__);
      }

      if (AicJit_AicInit_initSemaphores_is_present(aicInit)) {
        uint32_t numInitSemaphores = AicJit_AicInitSemaphore_vec_len(
            AicJit_AicInit_initSemaphores(aicInit));

        if (numInitSemaphores == 0) {
          LOG_DEBUG(qctx->logFuncPtr, "%s: No AicInitSemaphore present",
                    __func__);
        }

        for (uint32_t i = 0; i < numInitSemaphores; i++) {
          AicJit_AicInitSemaphore_struct_t sem = AicJit_AicInitSemaphore_vec_at(
              AicJit_AicInit_initSemaphores(aicInit), i);
          uint32_t semIndex = AicJit_AicInitSemaphore_index(sem);
          uint32_t semInitVal = AicJit_AicInitSemaphore_initValue(sem);
          qcDoorbellWrite(semIndex, semInitVal);

          LOG_DEBUG(qctx->logFuncPtr, "%s: semIndex: 0x%x, semInitVal: 0x%x",
                    __func__, semIndex, semInitVal);
        }
      } else {
        LOG_DEBUG(qctx->logFuncPtr, "%s: No AicInitSemaphore present",
                  __func__);
      }
      // extract the profiling configuration
      if (AicJit_AicInit_profilingConfig_is_present(aicInit)) {
        profilingBitmap = AicJit_ProfilingConfiguration_profilingMetricBitMap(
            AicJit_AicInit_profilingConfig(aicInit));
        LOG_DEBUG(qctx->logFuncPtr, "Profiling bitmask: 0x%llx",
                  LOG64_DATA(profilingBitmap));
      } else {
        LOG_DEBUG(qctx->logFuncPtr, "Profiling bitmask not set");
        // Explicitly set profiling globals to 0
        profilingBitmap = 0;
        g_profTlv = nullptr;
        g_profHeader = nullptr;
      }
    }
  } else {
    LOG_ERROR(qctx->logFuncPtr, "%s: Aic Init not present", __func__);
  }

exit:
  return status;
}

JitDevStatusCode_t
processAicCommandSemaphoreSignal(AICExecContext *qctx,
                                 AicJit_AicCommand_table_t cmd, uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  AicJit_AicSemaphoreOp_enum_t aicSemOpType;
  AicJit_AicSemaphoreSignalType_enum_t aicSemSignalType;
  uint32_t semIndex = 0, semOpValue = 0, semValue = 0, mcId = 0;

  if (AicJit_AicCommand_signalSemList_is_present(cmd)) {
    LOG_DEBUG(qctx->logFuncPtr, "%s: signalSemList", __func__);
    uint32_t signalSemListSize =
        AicJit_AicSemaphoreSignal_vec_len(AicJit_AicCommand_signalSemList(cmd));

    LOG_DEBUG(qctx->logFuncPtr, "%s: signalSemListSize: 0x%x", __func__,
              signalSemListSize);

    for (uint8_t i = 0; i < signalSemListSize; i++) {
      AicJit_AicSemaphoreSignal_struct_t signalSem =
          AicJit_AicSemaphoreSignal_vec_at(AicJit_AicCommand_signalSemList(cmd),
                                           i);

      if (AicJit_AicSemaphoreOp_is_known_value(
              AicJit_AicSemaphoreSignal_op(signalSem))) {
        aicSemOpType = AicJit_AicSemaphoreSignal_op(signalSem);
        LOG_DEBUG(qctx->logFuncPtr, "%s: semaphoreOp: %s", __func__,
                  AicJit_AicSemaphoreOp_name(
                      AicJit_AicSemaphoreSignal_op(signalSem)));
      } else {
        qshimLog(NNC_LOG_MASK_ERROR, 1,
                 "Unknown semaphoreOp: %d in AicSemaphoreSignal",
                 AicJit_AicSemaphoreSignal_op(signalSem));
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      if (AicJit_AicSemaphoreSignalType_is_known_value(
              AicJit_AicSemaphoreSignal_type(signalSem))) {
        aicSemSignalType = AicJit_AicSemaphoreSignal_type(signalSem);
        LOG_DEBUG(qctx->logFuncPtr, "%s: semaphoreSignalType: %s", __func__,
                  AicJit_AicSemaphoreSignalType_name(
                      AicJit_AicSemaphoreSignal_type(signalSem)));
      } else {
        qshimLog(NNC_LOG_MASK_ERROR, 1,
                 "Unknown semaphoreSignalType: %d in AicSemaphoreSignal",
                 AicJit_AicSemaphoreSignal_type(signalSem));
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      if (aicSemSignalType == AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP &&
          aicSemOpType != AicJit_AicSemaphoreOp_SEM_OP_INIT) {
        qshimLog(NNC_LOG_MASK_ERROR, 2,
                 "SemOpType: %d not allowed for SemSignalType: %d",
                 aicSemOpType, aicSemSignalType);
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      semIndex = AicJit_AicSemaphoreSignal_index(signalSem);
      semOpValue = AicJit_AicSemaphoreSignal_opValue(signalSem);
      LOG_DEBUG(qctx->logFuncPtr, "%s: semIndex: %d, semOpValue: %d", __func__,
                semIndex, semOpValue);

      if (aicSemSignalType == AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP) {
        mcId = AicJit_AicSemaphoreSignal_mcID(signalSem);
        LOG_DEBUG(qctx->logFuncPtr, "%s: mcId: %d", __func__, mcId);
      }

      if (aicSemSignalType == AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP) {
        if (semIndex < MASTER_SLAVE_NSP_INTER_CMD_SYNC_DB_OFFSET) {
          qshimLog(NNC_LOG_MASK_ERROR, 1,
                   "Invalid semIndex: %d for inter nsp sync", semIndex);
          status = JIT_DEV_ERROR_INVALID_PARAMETER;
          goto exit;
        }
      }

      if (aicSemOpType == AicJit_AicSemaphoreOp_SEM_OP_INIT) {
        /* For semaphore init, semOpValue = semValue */
        semValue = semOpValue;
        if (aicSemSignalType ==
            AicJit_AicSemaphoreSignalType_SIGNAL_INTRA_NSP) {
          qcDoorbellWrite(semIndex, semValue);
        } else if (aicSemSignalType ==
                   AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP) {
          qcDoorbellUnicast(mcId, semIndex, semValue);
        }
      } else if (aicSemOpType == AicJit_AicSemaphoreOp_SEM_OP_INC) {
        semValue = qcDoorbellRead(semIndex);
        qcAtomicFetchAdd(&semValue, semOpValue);
        qcDoorbellWrite(semIndex, semValue);
      } else if (aicSemOpType == AicJit_AicSemaphoreOp_SEM_OP_DEC) {
        semValue = qcDoorbellRead(semIndex);
        qcAtomicFetchSub(&semValue, semOpValue);
        qcDoorbellWrite(semIndex, semValue);
      }
      LOG_DEBUG(qctx->logFuncPtr, "%s: semIndex: %d, semValue: %d", __func__,
                semIndex, semValue);
    }
  }

exit:
  return status;
}

JitDevStatusCode_t processAicCommandSemaphoreWait(AICExecContext *qctx,
                                                  AicJit_AicCommand_table_t cmd,
                                                  uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  AicJit_AicSemaphoreCheck_enum_t aicSemCheckType;
  uint32_t semIndex = 0, semWaitValue = 0;

  if (AicJit_AicCommand_waitSemList_is_present(cmd)) {
    LOG_DEBUG(qctx->logFuncPtr, "%s: waitSemList", __func__);
    uint32_t waitSemListSize =
        AicJit_AicSemaphoreWait_vec_len(AicJit_AicCommand_waitSemList(cmd));

    LOG_DEBUG(qctx->logFuncPtr, "%s: waitSemListSize: 0x%x", __func__,
              waitSemListSize);

    for (uint8_t i = 0; i < waitSemListSize; i++) {
      AicJit_AicSemaphoreWait_struct_t waitSem =
          AicJit_AicSemaphoreWait_vec_at(AicJit_AicCommand_waitSemList(cmd), i);

      if (AicJit_AicSemaphoreCheck_is_known_value(
              AicJit_AicSemaphoreWait_check(waitSem))) {
        aicSemCheckType = AicJit_AicSemaphoreWait_check(waitSem);
        LOG_DEBUG(qctx->logFuncPtr, "%s: semaphoreCheck: %s", __func__,
                  AicJit_AicSemaphoreCheck_name(
                      AicJit_AicSemaphoreWait_check(waitSem)));
      } else {
        qshimLog(NNC_LOG_MASK_ERROR, 1,
                 "Unknown semaphoreCheck: %d in AicSemaphoreWait",
                 AicJit_AicSemaphoreWait_check(waitSem));
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      semIndex = AicJit_AicSemaphoreWait_index(waitSem);
      semWaitValue = AicJit_AicSemaphoreWait_checkValue(waitSem);
      LOG_DEBUG(qctx->logFuncPtr, "%s: semIndex: %d, semWaitValue: %d",
                __func__, semIndex, semWaitValue);

      if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_EQ) {
        qcDoorbellWaitEq(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      } else if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_GT) {
        qcDoorbellWaitGt(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      } else if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_GE) {
        qcDoorbellWaitGe(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      } else if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_LT) {
        qcDoorbellWaitLt(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      } else if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_LE) {
        qcDoorbellWaitLe(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      }
    }
  }

exit:
  return status;
}

JitDevStatusCode_t
processAicCommandSetSemaphoreSignal(AICExecContext *qctx,
                                    AicJit_AicCommandSet_table_t buffer,
                                    uint8_t tid, JitSemType_t type) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  AicJit_AicSemaphoreOp_enum_t aicSemOpType;
  AicJit_AicSemaphoreSignalType_enum_t aicSemSignalType;
  uint32_t semIndex = 0, semOpValue = 0, semValue = 0, mcId = 0;

  if (AicJit_AicCommandSet_semaphoares_is_present(buffer)) {
    AicJit_AicCommandSetSemaphores_table_t semaphores =
        AicJit_AicCommandSet_semaphoares(buffer);
    uint32_t signalSemListSize = 0;
    if (type == JIT_EXIT_SEM) {
      LOG_DEBUG(qctx->logFuncPtr, "%s: exitSignalSemList", __func__);
      signalSemListSize = AicJit_AicSemaphoreSignal_vec_len(
          AicJit_AicCommandSetSemaphores_exitSignalSemList(semaphores));

      LOG_DEBUG(qctx->logFuncPtr, "%s: exitSignalSemListSize: 0x%x", __func__,
                signalSemListSize);
    } else if (type == JIT_ENTRY_SEM) {
      LOG_DEBUG(qctx->logFuncPtr, "%s: entrySignalSemList", __func__);
      signalSemListSize = AicJit_AicSemaphoreSignal_vec_len(
          AicJit_AicCommandSetSemaphores_entrySignalSemList(semaphores));

      LOG_DEBUG(qctx->logFuncPtr, "%s: entrySignalSemListSize: 0x%x", __func__,
                signalSemListSize);
    }
    for (uint8_t i = 0; i < signalSemListSize; i++) {
      AicJit_AicSemaphoreSignal_struct_t signalSem;
      if (type == JIT_EXIT_SEM) {
        signalSem = AicJit_AicSemaphoreSignal_vec_at(
            AicJit_AicCommandSetSemaphores_exitSignalSemList(semaphores), i);
      } else if (type == JIT_ENTRY_SEM) {
        signalSem = AicJit_AicSemaphoreSignal_vec_at(
            AicJit_AicCommandSetSemaphores_entrySignalSemList(semaphores), i);
      }
      if (AicJit_AicSemaphoreOp_is_known_value(
              AicJit_AicSemaphoreSignal_op(signalSem))) {
        aicSemOpType = AicJit_AicSemaphoreSignal_op(signalSem);
        LOG_DEBUG(qctx->logFuncPtr, "%s: semaphoreOp: %s", __func__,
                  AicJit_AicSemaphoreOp_name(
                      AicJit_AicSemaphoreSignal_op(signalSem)));
      } else {
        qshimLog(NNC_LOG_MASK_ERROR, 1,
                 "Unknown semaphoreOp: %d in AicSemaphoreSignal",
                 AicJit_AicSemaphoreSignal_op(signalSem));
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      if (AicJit_AicSemaphoreSignalType_is_known_value(
              AicJit_AicSemaphoreSignal_type(signalSem))) {
        aicSemSignalType = AicJit_AicSemaphoreSignal_type(signalSem);
        LOG_DEBUG(qctx->logFuncPtr, "%s: semaphoreSignalType: %s", __func__,
                  AicJit_AicSemaphoreSignalType_name(
                      AicJit_AicSemaphoreSignal_type(signalSem)));
      } else {
        qshimLog(NNC_LOG_MASK_ERROR, 1,
                 "Unknown semaphoreSignalType: %d in AicSemaphoreSignal",
                 AicJit_AicSemaphoreSignal_type(signalSem));
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      if (aicSemSignalType == AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP &&
          aicSemOpType != AicJit_AicSemaphoreOp_SEM_OP_INIT) {
        qshimLog(NNC_LOG_MASK_ERROR, 2,
                 "SemOpType: %d not allowed for SemSignalType: %d",
                 aicSemOpType, aicSemSignalType);
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      semIndex = AicJit_AicSemaphoreSignal_index(signalSem);
      semOpValue = AicJit_AicSemaphoreSignal_opValue(signalSem);
      LOG_DEBUG(qctx->logFuncPtr, "%s: semIndex: %d, semOpValue: %d", __func__,
                semIndex, semOpValue);

      if (aicSemSignalType == AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP) {
        mcId = AicJit_AicSemaphoreSignal_mcID(signalSem);
        LOG_DEBUG(qctx->logFuncPtr, "%s: mcId: %d", __func__, mcId);
      }

      if (aicSemSignalType == AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP) {
        if (semIndex < SLAVE_NSP_ENTRY_SEM_SYNC_DB_OFFSET_BEGIN ||
            semIndex > SLAVE_NSP_ENTRY_SEM_SYNC_DB_OFFSET_LAST) {
          if (semIndex < MASTER_SLAVE_NSP_INTER_CMD_SET_SYNC_DB_OFFSET ||
              semIndex >= (MASTER_SLAVE_NSP_INTER_CMD_SET_SYNC_DB_OFFSET +
                           MAX_AIC_CORES)) {
            qshimLog(NNC_LOG_MASK_ERROR, 1,
                     "Invalid semIndex: %d for inter nsp sync", semIndex);
            status = JIT_DEV_ERROR_INVALID_PARAMETER;
            goto exit;
          }
        }
      }

      if (aicSemOpType == AicJit_AicSemaphoreOp_SEM_OP_INIT) {
        /* For semaphore init, semOpValue = semValue */
        semValue = semOpValue;
        if (aicSemSignalType ==
            AicJit_AicSemaphoreSignalType_SIGNAL_INTRA_NSP) {
          qcDoorbellWrite(semIndex, semValue);
        } else if (aicSemSignalType ==
                   AicJit_AicSemaphoreSignalType_SIGNAL_INTER_NSP) {
          qcDoorbellUnicast(mcId, semIndex, semValue);
        }
      } else if (aicSemOpType == AicJit_AicSemaphoreOp_SEM_OP_INC) {
        semValue = qcDoorbellRead(semIndex);
        qcAtomicFetchAdd(&semValue, semOpValue);
        qcDoorbellWrite(semIndex, semValue);
      } else if (aicSemOpType == AicJit_AicSemaphoreOp_SEM_OP_DEC) {
        semValue = qcDoorbellRead(semIndex);
        qcAtomicFetchSub(&semValue, semOpValue);
        qcDoorbellWrite(semIndex, semValue);
      }
      LOG_DEBUG(qctx->logFuncPtr, "%s: semIndex: %d, semValue: %d", __func__,
                semIndex, semValue);
    }
  }

exit:
  return status;
}

JitDevStatusCode_t
processAicCommandSetSemaphoreWait(AICExecContext *qctx,
                                  AicJit_AicCommandSet_table_t buffer,
                                  uint8_t tid, JitSemType_t type) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  AicJit_AicSemaphoreCheck_enum_t aicSemCheckType;
  uint32_t semIndex = 0, semWaitValue = 0;

  if (AicJit_AicCommandSet_semaphoares_is_present(buffer)) {
    AicJit_AicCommandSetSemaphores_table_t semaphores =
        AicJit_AicCommandSet_semaphoares(buffer);
    uint32_t SemListSize = 0;
    if (type == JIT_EXIT_SEM) {
      LOG_DEBUG(qctx->logFuncPtr, "%s: exitWaitSemList", __func__);
      SemListSize = AicJit_AicSemaphoreWait_vec_len(
          AicJit_AicCommandSetSemaphores_exitWaitSemList(semaphores));
      LOG_DEBUG(qctx->logFuncPtr, "%s: exitWaitSemListSize: 0x%x", __func__,
                SemListSize);
    } else if (type == JIT_ENTRY_SEM) {
      LOG_DEBUG(qctx->logFuncPtr, "%s: entryWaitSemList", __func__);
      SemListSize = AicJit_AicSemaphoreWait_vec_len(
          AicJit_AicCommandSetSemaphores_entryWaitSemList(semaphores));
      LOG_DEBUG(qctx->logFuncPtr, "%s: entryWaitSemListSize: 0x%x", __func__,
                SemListSize);
    }

    for (uint8_t i = 0; i < SemListSize; i++) {
      AicJit_AicSemaphoreWait_struct_t waitSem;
      if (type == JIT_EXIT_SEM) {
        waitSem = AicJit_AicSemaphoreWait_vec_at(
            AicJit_AicCommandSetSemaphores_exitWaitSemList(semaphores), i);
      } else if (type == JIT_ENTRY_SEM) {
        waitSem = AicJit_AicSemaphoreWait_vec_at(
            AicJit_AicCommandSetSemaphores_entryWaitSemList(semaphores), i);
      }

      if (AicJit_AicSemaphoreCheck_is_known_value(
              AicJit_AicSemaphoreWait_check(waitSem))) {
        aicSemCheckType = AicJit_AicSemaphoreWait_check(waitSem);
        LOG_DEBUG(qctx->logFuncPtr, "%s: semaphoreCheck: %s", __func__,
                  AicJit_AicSemaphoreCheck_name(
                      AicJit_AicSemaphoreWait_check(waitSem)));
      } else {
        qshimLog(NNC_LOG_MASK_ERROR, 1,
                 "Unknown semaphoreCheck: %d in AicSemaphoreWait",
                 AicJit_AicSemaphoreWait_check(waitSem));
        status = JIT_DEV_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      semIndex = AicJit_AicSemaphoreWait_index(waitSem);
      semWaitValue = AicJit_AicSemaphoreWait_checkValue(waitSem);
      LOG_DEBUG(qctx->logFuncPtr, "%s: semIndex: %d, semWaitValue: %d",
                __func__, semIndex, semWaitValue);

      if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_EQ) {
        qcDoorbellWaitEq(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      } else if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_GT) {
        qcDoorbellWaitGt(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      } else if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_GE) {
        qcDoorbellWaitGe(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      } else if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_LT) {
        qcDoorbellWaitLt(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      } else if (aicSemCheckType == AicJit_AicSemaphoreCheck_SEM_CHK_LE) {
        qcDoorbellWaitLe(semIndex, semWaitValue);
        qcDoorbellWrite(semIndex, 0);
      }
      LOG_DEBUG(qctx->logFuncPtr,
                "%s: Doorbell semIndex: %d, semWaitValue: %d received",
                __func__, semIndex, semWaitValue);
    }
  }

exit:
  return status;
}

JitDevStatusCode_t
processAicCommandSetCommands(AICExecContext *qctx,
                             AicJit_AicCommandSet_table_t buffer,
                             uint8_t cmdIdx, uint8_t tid) {
  JitDevStatusCode_t status = JIT_DEV_STATUS_SUCCESS;
  uint32_t commandID = 0;

  AicJit_AicCommand_table_t cmd =
      AicJit_AicCommand_vec_at(AicJit_AicCommandSet_commands(buffer), cmdIdx);
  if (cmd) {
    if (AicJit_AicCommand_commandID_is_present(cmd)) {
      commandID = AicJit_AicCommand_commandID(cmd);
      g_opErrCommandId = commandID;
      LOG_DEBUG(qctx->logFuncPtr, "%s: tid: %d, commandID: %d", __func__, tid,
                commandID);
      if (profilingBitmap & (1 << DEV_GRANULAR_DATA)) {
        if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
          g_profTable[g_profCount].start_ts = qcGetTimestamp();
        } else {
          LOG_ERROR(qctx->logFuncPtr,
                    "Profiling Error: Profiling table index overflow");
        }
      }
      status = processAicCommandSemaphoreWait(qctx, cmd, tid);
      if (profilingBitmap & (1 << DEV_GRANULAR_DATA) &&
          g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
        g_profTable[g_profCount].stop_ts = qcGetTimestamp();
        strcpy(g_profTable[g_profCount].tag, "C-CmdSemWait");
        g_profCount++;
      }
      if (AicJit_AicCommand_cmd_is_present(cmd) &&
          status == JIT_DEV_STATUS_SUCCESS) {
        AicJit_AicCmdUnion_union_type_t cmdType =
            AicJit_AicCommand_cmd_type(cmd);
        if (AicJit_AicCmdUnion_is_known_type(cmdType)) {
          LOG_DEBUG(qctx->logFuncPtr, "%s: cmdType: %s", __func__,
                    AicJit_AicCmdUnion_type_name(cmdType));
          if (cmdType == AicJit_AicCmdUnion_AicCmdDispatch) {
            if (profilingBitmap & (1 << DEV_GRANULAR_DATA)) {
              if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
                g_profTable[g_profCount].start_ts = qcGetTimestamp();
              } else {
                LOG_ERROR(qctx->logFuncPtr,
                          "Profiling Error: Profiling table index overflow");
              }
            }
            status = processAicCommandDispatch(
                qctx, commandID, buffer, (void *)AicJit_AicCommand_cmd(cmd),
                tid);

            if (status != JIT_DEV_STATUS_SUCCESS)
              goto exit;

            if (threadIdBitMap) {
              qcAtomicStore(&waitForDispatch, 1);

              uint8_t numWorkerThreads = 0;
              for (uint8_t i = 0; i < MAX_NUM_THREADS; i++) {
                if (threadIdBitMap & (1 << i)) {
                  numWorkerThreads++;
                }
              }
              LOG_DEBUG(qctx->logFuncPtr, "numWorkerThreads: %d",
                        numWorkerThreads);
              if (profilingBitmap & (1 << DEV_GRANULAR_DATA &&
                                     g_profCount < JIT_PROF_TABLE_MAX_SIZE)) {
                g_profTable[g_profCount].stop_ts = qcGetTimestamp();
                strcpy(g_profTable[g_profCount].tag, "C-cmdDispatch");
                g_profCount++;
              }
              if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
                uint64_t ucycle = qcGetTimestamp();
                JITTimeStampEventData_t data = {
                    .uCycleCounts = ucycle,
                    .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
                write_Profiling_Event(qctx, "C-Prep",
                                      g_profHeader->commandSetID, tid,
                                      END_PHASE, EVENT_TYPE_TIME_STAMP, &data);
                write_Profiling_Event(qctx, "C-Exec", g_profCmdId, tid,
                                      BEGIN_PHASE, EVENT_TYPE_TIME_STAMP,
                                      &data);
              }

              for (uint8_t i = 0; i < MAX_NUM_THREADS; i++) {
                if (threadIdBitMap & (1 << i)) {
                  LOG_DEBUG(qctx->logFuncPtr, "%s: Dispatch tid: %d", __func__,
                            i);
                  qcDoorbellWrite(L2TCM_WORKER_THREADS_START_DB_OFFSET + i, 1);
                }
              }
              uint64_t waitStart = qcGetTimestamp();
              int32_t uTimerFreqMs = getUtimerFreqMs();
              uint32_t threadIdBitMapCpy = threadIdBitMap;
              uint64_t waitTimeCycles = g_opTimeoutMs * uTimerFreqMs -
                                        ((qcGetTimestamp() - waitStart));
              setWaitTimeOut(waitTimeCycles);

              while (threadIdBitMapCpy) {
                if (qcAtomicLoad(&prngWorkQueueCount) > 0) {
                  LOG_DEBUG(qctx->logFuncPtr, "%s: PRNG WQ Count: 0x%x",
                            __func__, prngWorkQueueCount);
                  for (uint8_t i = 0; i < NUM_WORKER_THREADS; i++) {
                    if (qcAtomicLoad(&prngWorkQueue[i].status) ==
                        PRNG_WQ_SLOT_PENDING) {
                      qctx->getRandNumFuncPtr(prngWorkQueue[i].addr,
                                              prngWorkQueue[i].numElements);
                      qcAtomicStore(&prngWorkQueue[i].status,
                                    PRNG_WQ_SLOT_COMPLETE);
                    }
                  }
                }

                for (uint8_t i = 0; i < MAX_NUM_THREADS; i++) {
                  if (threadIdBitMapCpy & (1 << i)) {
                    g_tidFinishDbWait = i;
                    if (qcDoorbellRead(L2TCM_WORKER_THREADS_FINISH_DB_OFFSET +
                                       i) == 1) {
                      qcDoorbellWrite(L2TCM_WORKER_THREADS_FINISH_DB_OFFSET + i,
                                      0);
                      LOG_DEBUG(qctx->logFuncPtr, "%s: Wait for tid: %d",
                                __func__, i);
                      threadIdBitMapCpy = threadIdBitMapCpy & ~(1 << i);
                    }
                    uint64_t time_elapsed_ms =
                        (qcGetTimestamp() - waitStart) / UTIMER_FREQ_MS;
                    if (time_elapsed_ms >= g_opTimeoutMs) {
                      threadIdBitMapCpy = threadIdBitMapCpy & ~(1 << i);
                      qcDoorbellWrite(L2TCM_WORKER_THREADS_FINISH_DB_OFFSET + i,
                                      0);
                      notifyOpTimeoutFunc();
                    }
                    qaic::quitIfExitDBSet(tid);
                  }
                }
              }

              // For now we don't check the return value of entry point
              // function. Compute function is not expected to return an
              // error.

              if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
                uint64_t ucycle = qcGetTimestamp();
                JITTimeStampEventData_t data = {
                    .uCycleCounts = ucycle,
                    .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
                write_Profiling_Event(qctx, "C-Exec", g_profCmdId, tid,
                                      END_PHASE, EVENT_TYPE_TIME_STAMP, &data);
                write_Profiling_Event(
                    qctx, "C-Cleanup", g_profHeader->commandSetID, tid,
                    BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &data);
              }
              if (listParamPtr) {
                g_networkHeapFree(listParamPtr);
                listParamPtr = nullptr;
              }
            }
          } else if (cmdType == AicJit_AicCmdUnion_AicCmdAllocBuf) {
            if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
              uint64_t ucycle = qcGetTimestamp();
              JITTimeStampEventData_t data = {
                  .uCycleCounts = ucycle,
                  .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
              write_Profiling_Event(qctx, "C-Prep", g_profHeader->commandSetID,
                                    tid, END_PHASE, EVENT_TYPE_TIME_STAMP,
                                    &data);
              write_Profiling_Event(qctx, "C-Exec", g_profCmdId, tid,
                                    BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &data);
            }
            status = processAicCommandAllocBuf(
                qctx, commandID, (void *)AicJit_AicCommand_cmd(cmd), tid);
            if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
              uint64_t ucycle = qcGetTimestamp();
              JITTimeStampEventData_t data = {
                  .uCycleCounts = ucycle,
                  .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
              write_Profiling_Event(qctx, "C-Exec", g_profCmdId, tid, END_PHASE,
                                    EVENT_TYPE_TIME_STAMP, &data);
              write_Profiling_Event(qctx, "C-Cleanup",
                                    g_profHeader->commandSetID, tid,
                                    BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &data);
            }
            if (status != JIT_DEV_STATUS_SUCCESS) {
              goto exit;
            }
          } else if (cmdType == AicJit_AicCmdUnion_AicCmdFreeBuf) {
            if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
              uint64_t ucycle = qcGetTimestamp();
              JITTimeStampEventData_t data = {
                  .uCycleCounts = ucycle,
                  .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
              write_Profiling_Event(qctx, "C-Prep", g_profHeader->commandSetID,
                                    tid, END_PHASE, EVENT_TYPE_TIME_STAMP,
                                    &data);
              write_Profiling_Event(qctx, "C-Exec", g_profCmdId, tid,
                                    BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &data);
            }
            status = processAicCommandFreeBuf(
                qctx, commandID, (void *)AicJit_AicCommand_cmd(cmd), tid);
            if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
              uint64_t ucycle = qcGetTimestamp();
              JITTimeStampEventData_t data = {
                  .uCycleCounts = ucycle,
                  .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
              write_Profiling_Event(qctx, "C-Exec", g_profCmdId, tid, END_PHASE,
                                    EVENT_TYPE_TIME_STAMP, &data);
              write_Profiling_Event(qctx, "C-Cleanup",
                                    g_profHeader->commandSetID, tid,
                                    BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &data);
            }
            if (status != JIT_DEV_STATUS_SUCCESS) {
              goto exit;
            }
          }
        } else {
          qshimLog(NNC_LOG_MASK_ERROR, 2, "%s: Unknown cmdType: %d", __func__,
                   cmdType);
          status = JIT_DEV_ERROR_INVALID_PARAMETER;
          goto exit;
        }
      }

      if (status == JIT_DEV_STATUS_SUCCESS) {
        status = processAicCommandSemaphoreSignal(qctx, cmd, tid);
      }
    } else {
      qshimLog(NNC_LOG_MASK_ERROR, 1,
               "%s: Command ID not present in AicCommand", __func__);
      status = JIT_DEV_ERROR_INVALID_PARAMETER;
      goto exit;
    }
  }

exit:
  if (g_rspTlv != nullptr) {
    ResponseTLV_t *rspTlv = g_rspTlv;
    CommandValue_t *cmdVal = (CommandValue_t *)&rspTlv->value[0];

    cmdVal->commandID = commandID;
    if (status != JIT_DEV_STATUS_SUCCESS) {
      // Command processing failure, status 1
      cmdVal->status = 1;
    } else if (status == JIT_DEV_STATUS_SUCCESS &&
               g_firstOpErrorReported != 0) {
      // Operator failure status 2
      cmdVal->status = 2;
    } else {
      cmdVal->status = status;
    }
  }
  return status;
}

JitDevStatusCode_t loadFwSymbols(AICExecContext *qctx) {
  // Extract malloc and free function pointers for network heap
  g_networkHeapMalloc =
      (JitQpcShim_Malloc_t)qctx->dlSymPtr(RTLD_DEFAULT, "__wrap_malloc");
  if (g_networkHeapMalloc != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted __wrap_malloc successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract __wrap_malloc");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_networkHeapFree =
      (JitQpcShim_Free_t)qctx->dlSymPtr(RTLD_DEFAULT, "__wrap_free");
  if (g_networkHeapFree != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted __wrap_free successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract __wrap_free");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  // Extract malloc and free function pointers for DDR
  g_ddrMalloc = (JitQpcShim_ddrMalloc_t)qctx->dlSymPtr(RTLD_DEFAULT,
                                                       "nnc_pd_ddr_alloc_req");
  if (g_ddrMalloc != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted nnc_pd_ddr_alloc_req successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract nnc_pd_ddr_alloc_req");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_ddrFree =
      (JitQpcShim_ddrFree_t)qctx->dlSymPtr(RTLD_DEFAULT, "nnc_pd_ddr_free_req");
  if (g_ddrFree != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted nnc_pd_ddr_free_req successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract nnc_pd_ddr_free_req");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  // Extract memcpy, memset and snprintf function pointers
  g_memcpy = (JitQpcShim_Memcpy_t)qctx->dlSymPtr(RTLD_DEFAULT, "memcpy");
  if (g_memcpy != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted memcpy successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract memcpy");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_memset = (JitQpcShim_Memset_t)qctx->dlSymPtr(RTLD_DEFAULT, "memset");
  if (g_memset != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted memset successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract memset");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_snprintf = (JitQpcShim_Snprintf_t)qctx->dlSymPtr(RTLD_DEFAULT, "snprintf");
  if (g_snprintf != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted snprintf successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract snprintf");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_mmap =
      (JitQpcShim_Mmap_t)qctx->dlSymPtr(RTLD_DEFAULT, "nnc_pd_mmap_jit_buffer");
  if (g_mmap != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr,
              "Extracted nnc_pd_mmap_jit_buffer successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract nnc_pd_mmap_jit_buffer");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_vsnprintf =
      (JitQpcShim_Vsnprintf_t)qctx->dlSymPtr(RTLD_DEFAULT, "vsnprintf");
  if (g_vsnprintf != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted vsnprintf successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract vsnprintf");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_qurtMutexInit = (JitQpcShim_QurtMutexInit_t)qctx->dlSymPtr(
      RTLD_DEFAULT, "qurt_mutex_init");
  if (g_qurtMutexInit != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted qurt_mutex_init successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract qurt_mutex_init");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_qurtMutexLock = (JitQpcShim_QurtMutexInit_t)qctx->dlSymPtr(
      RTLD_DEFAULT, "qurt_mutex_lock");
  if (g_qurtMutexLock != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted qurt_mutex_lock successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract qurt_mutex_lock");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_qurtMutexUnlock = (JitQpcShim_QurtMutexInit_t)qctx->dlSymPtr(
      RTLD_DEFAULT, "qurt_mutex_unlock");
  if (g_qurtMutexUnlock != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted qurt_mutex_unlock successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract qurt_mutex_unlock");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_qurtMutexDestroy = (JitQpcShim_QurtMutexInit_t)qctx->dlSymPtr(
      RTLD_DEFAULT, "qurt_mutex_destroy");
  if (g_qurtMutexDestroy != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted qurt_mutex_destroy successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract qurt_mutex_destroy");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_qurtHvxSet =
      (JitQpcShim_QurtHvxSet_t)qctx->dlSymPtr(RTLD_DEFAULT, "qurt_hvx_set");
  if (g_qurtHvxSet != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted qurt_hvx_set successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract qurt_hvx_set");
    return JIT_DEV_ERROR_LOADING_RTLD_SYMBOLS;
  }

  g_getVtid =
      (JitQpcShim_getVtid_t)qctx->dlSymPtr(RTLD_DEFAULT, "nnc_pd_get_vtid");
  if (g_getVtid != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted nnc_pd_get_vtid successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract nnc_pd_get_vtid");
  }

  g_assert = (JitQpcShim_assert_t)qctx->dlSymPtr(RTLD_DEFAULT, "nnc_pd_assert");
  if (g_assert != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted nnc_pd_assert successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract nnc_pd_assert");
  }

  g_unmapExe = (JitQpcShim_UnmapExe_t)qctx->dlSymPtr(RTLD_DEFAULT,
                                                     "nnc_pd_unmap_jit_exe");
  if (g_getVtid != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted nnc_pd_unmap_jit_exe successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract nnc_pd_unmap_jit_exe");
  }

  g_qurtMemCacheClean = (JitQpcShim_QurtMemCacheClean_t)qctx->dlSymPtr(
      RTLD_DEFAULT, "qurt_mem_cache_clean");
  if (g_qurtMemCacheClean != nullptr) {
    LOG_DEBUG(qctx->logFuncPtr, "Extracted qurt_mem_cache_clean successfully");
  } else {
    LOG_ERROR(qctx->logFuncPtr, "Failed to extract qurt_mem_cache_clean");
  }
  return JIT_DEV_STATUS_SUCCESS;
}

void write_ProfTable_To_StatusBuffer(AICExecContext *qctx, uint64_t eventUID,
                                     uint8_t threadId) {
  for (size_t i = 0; i < g_profCount; i++) {
    JITTimeStampEventData_t startData = {
        .uCycleCounts = g_profTable[i].start_ts,
        .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
    JITTimeStampEventData_t endData = {
        .uCycleCounts = g_profTable[i].stop_ts,
        .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
    write_Profiling_Event(qctx, g_profTable[i].tag, eventUID, threadId,
                          BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &startData);
    write_Profiling_Event(qctx, g_profTable[i].tag, eventUID, threadId,
                          END_PHASE, EVENT_TYPE_TIME_STAMP, &endData);
  }
}

void qshimProfilingEntryStart(uint32_t threadId, const char *eventName) {
  if ((profilingBitmap & (1 << DEV_GRANULAR_DATA)) &&
      (profilingBitmap & (1 << COMMAND_EXECUTION_TIME))) {
    JITTimeStampEventData_t data = {
        .uCycleCounts = qcGetTimestamp(),
        .deviceBootTimeUTC = *((uint64_t *)g_qctx->baseUtcOffsetDDR)};
    write_Profiling_Event(g_qctx, eventName, g_profHeader->commandSetID,
                          (uint8_t)threadId, BEGIN_PHASE, EVENT_TYPE_TIME_STAMP,
                          &data);
  }
}

void qshimProfilingEntryEnd(uint32_t threadId, const char *eventName) {
  if ((profilingBitmap & (1 << DEV_GRANULAR_DATA)) &&
      (profilingBitmap & (1 << COMMAND_EXECUTION_TIME))) {
    JITTimeStampEventData_t data = {
        .uCycleCounts = qcGetTimestamp(),
        .deviceBootTimeUTC = *((uint64_t *)g_qctx->baseUtcOffsetDDR)};
    write_Profiling_Event(g_qctx, eventName, g_profHeader->commandSetID,
                          (uint8_t)threadId, END_PHASE, EVENT_TYPE_TIME_STAMP,
                          &data);
  }
}

static uint64_t g_mstrStart = 0;
static uint64_t g_mstrStop = 0;

void run(AICExecContext *qctx, uint8_t tid, uint32_t stid) {

  resetErrRspStrPtr(tid);

  if (MASTER_THREAD == tid) {

    AicCommandSetResponse *rsp = nullptr;
    void *obuff = nullptr;

    buffer = nullptr;
    g_rspTlv = nullptr;
    g_startTs = 0;
    g_profCount = 0;
    g_firstOpErrorReported = 0;
    cmdBuffRspHdr = nullptr;
    initStatus = JIT_DEV_STATUS_SUCCESS;

    /* Reset the input sync variable */
    qcAtomicStore(&commandIdx, 0);
    qcAtomicStore(&waitForDispatch, 0);
    qcAtomicStore(&threadIdBitMap, 0);
    uint32_t *dbs = reinterpret_cast<uint32_t *>(g_qctx->baseL2TCM);
    // Reset all the uDMA doorbells before dispatch
    for (uint32_t i = 0; i < MAX_UDMA_REQS; i++) {
      dbs[L2TCM_UDMA_START_DB_OFFSET + i] = 0;
    }

    /* Get the first input */

    uint16_t cmdSetInputDB = JIT_CMD_SET_INPUT_DB_BASE_INDEX + ringBufferIdx;
    qcDoorbellWaitEq(cmdSetInputDB, 1);
    qcDoorbellWrite(cmdSetInputDB, 0);
    resetPerCommandBufferDebugInfo();

    if (qctx->virtualNSPId != 0) {
      g_mstrStart = qcGetTimestamp();
    } else {
      // PROF capture: command prep start time for master NSP
      g_startTs = qcGetTimestamp();
    }

    LOG_DEBUG(qctx->logFuncPtr, "Input buffer is ready");

    LOG_DEBUG(qctx->logFuncPtr, "Ring Buffer Index: %d", ringBufferIdx);

    /* Get input buffer address and size */
    uint32_t bufferOffset =
        L2TCM_CMD_BUFFER_BASE_OFFSET + (ringBufferIdx * g_cmdBufSize);
    setRingBufferIdx(ringBufferIdx);

    /* update ring buffer index */
    if (++ringBufferIdx == g_ringBufferDepth) {
      ringBufferIdx -= g_ringBufferDepth;
    }

    uint8_t *buff = (uint8_t *)(qctx->baseL2TCM + bufferOffset);
    AicCommandSetRequest_t *buff0 = (AicCommandSetRequest_t *)buff;

    uint32_t inSize = buff0->length;
    LOG_DEBUG(qctx->logFuncPtr, "buff0 addr: 0x%x, size: 0x%x", buff0, inSize);

    /* Check canary value and record any error in CmdSetHeaderRsp */
    if (buff0->canary != CommandSetRequestCanary) {
      ERR_FATAL(qctx->errFuncPtr,
                "Invalid CommandSet Canary: Act[0x%x] Exp[0x%x]", buff0->canary,
                CommandSetRequestCanary, 0);
    } else {
      g_respInfo = buff0->cmdSetRespInfo;
    }

    /* Get output pointer and memset to 0 */
    obuff = (void *)((char *)qctx->baseSharedDDR + g_respInfo.offset);
    memset(obuff, 0, g_respInfo.size);
    rsp = (AicCommandSetResponse *)obuff;
    cmdBuffRspHdr = &rsp->header;
    cmdBuffRspHdr->size = g_respInfo.size;
    setErrRspStrPtr(tid, &cmdBuffRspHdr->errMsg[0]);

    LOG_DEBUG(qctx->logFuncPtr,
              "g_respInfo.size: %llu, g_respInfo.offset: %llu",
              LOG64_DATA(g_respInfo.size), LOG64_DATA(g_respInfo.offset));

    // TBD
    // PROF capture: FlatBufferVerify
    if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
      g_profTable[g_profCount].start_ts = qcGetTimestamp();
    } else {
      LOG_ERROR(qctx->logFuncPtr,
                "Profiling Error: Profiling table index overflow");
    }

    /* Verify command set */
    buffer = verify_AicJit_AicCommandSet(
        qctx, (uint8_t *)buff0->body,
        inSize - offsetof(AicCommandSetRequest_t, body));
    if (!buffer) {
      qshimLog(NNC_LOG_MASK_ERROR, 0, "Flatbuffer verification failed");
      initStatus = JIT_DEV_ERROR_INVALID_FLAT_BUFFER;
    } else {
      if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
        g_profTable[g_profCount].stop_ts = qcGetTimestamp();
        strcpy(g_profTable[g_profCount].tag, "C-FBVerify");
        g_profCount++;
      }

      if (qctx->virtualNSPId == 0) {
        /* Signal all the Slave NSPs */
        processAicCommandSetSemaphoreSignal(qctx, buffer, tid, JIT_ENTRY_SEM);
        LOG_DEBUG(qctx->logFuncPtr, "Master NSP sync set");
      } else {
        /* Wait for master NSP signal to start command processing */
        processAicCommandSetSemaphoreWait(qctx, buffer, tid, JIT_ENTRY_SEM);
        LOG_DEBUG(qctx->logFuncPtr, "Slave NSP sync received");
        g_mstrStop = qcGetTimestamp();
        // PROF capture: command prep start time
        g_startTs = qcGetTimestamp();
      }

      if (AicJit_AicCommandSet_commandSetID_is_present(buffer)) {
        LOG_DEBUG(qctx->logFuncPtr, "CommandSetId: %d",
                  AicJit_AicCommandSet_commandSetID(buffer));

        // Update commandSetID in response header
        cmdBuffRspHdr->commandID = AicJit_AicCommandSet_commandSetID(buffer);

        // PROF capture: AicInit()
        if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
          g_profTable[g_profCount].start_ts = qcGetTimestamp();
        } else {
          LOG_ERROR(qctx->logFuncPtr,
                    "Profiling Error: Profiling table index overflow");
        }

        initStatus = processCommandSetAicInit(qctx, buffer, tid);

        /* AicInit is successful so any error in Buffer Descriptor
          initialization will be recorded in CmdSetHeaderRsp */
        if (initStatus == JIT_DEV_STATUS_SUCCESS) {
          if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
            g_profTable[g_profCount].stop_ts = qcGetTimestamp();
            strcpy(g_profTable[g_profCount].tag, "C-CmdSetInit");
            g_profCount++;
          }

          // PROF capture: SetLocalStorage()
          if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
            g_profTable[g_profCount].start_ts = qcGetTimestamp();
          } else {
            LOG_ERROR(qctx->logFuncPtr,
                      "Profiling Error: Profiling table index overflow");
          }

          initStatus = processCommandSetLocalStorage(qctx, buffer, tid);

          if (initStatus == JIT_DEV_STATUS_SUCCESS) {
            if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
              g_profTable[g_profCount].stop_ts = qcGetTimestamp();
              strcpy(g_profTable[g_profCount].tag, "C-LocalStor");
              g_profCount++;
            }

            // PROF capture: BufferDescriptor()
            if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
              g_profTable[g_profCount].start_ts = qcGetTimestamp();
            } else {
              LOG_ERROR(qctx->logFuncPtr,
                        "Profiling Error: Profiling table index overflow");
            }

            initStatus =
                processCommandSetAicBufferDescriptor(qctx, buffer, tid);

            if (initStatus == JIT_DEV_STATUS_SUCCESS) {
              if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
                g_profTable[g_profCount].stop_ts = qcGetTimestamp();
                strcpy(g_profTable[g_profCount].tag, "C-SetupBuf");
                g_profCount++;
              }
            }
          }
        }
      } else {
        initStatus = JIT_DEV_ERROR_CMDSET_VERIFICATION;
        qshimLog(NNC_LOG_MASK_ERROR, 0,
                 "CommandSetId not present in Aic Command Set");
      }
    }
    /* Tid5 sets the doorbell to 1 atomically */
    qcAtomicStore(&inputSync, 1);
  } else {
    // Other threads wait for doorbell value to be
    // equal to 1 - wait for Command Set buffer
    // verification and initialization
    qcDoorbellLocalWaitGe(&inputSync, tid, 1);
  }

  /* All threads atomically increment the doorbell */
  qcAtomicFetchAdd(&inputSync, 1);

  /* All threads wait for the doorbell value to be equal to 7 */
  qcDoorbellLocalWaitEq(&inputSync, tid, 1 + totalThreadCount());

  if (tid == MASTER_THREAD) {
    if (initStatus == JIT_DEV_STATUS_SUCCESS) {
      // PROF capture: C-InitTLV (inital res buffer and prof buffer)
      if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
        g_profTable[g_profCount].start_ts = qcGetTimestamp();
      } else {
        LOG_ERROR(qctx->logFuncPtr,
                  "Profiling Error: Profiling table index overflow");
      }

      if (AicJit_AicCommandSet_commands_is_present(buffer)) {
        uint32_t numCommands =
            AicJit_AicCommand_vec_len(AicJit_AicCommandSet_commands(buffer));

        uint32_t idx = qcAtomicFetchAdd(&commandIdx, 1);
        AicJit_AicCommand_table_t cmd = AicJit_AicCommand_vec_at(
            AicJit_AicCommandSet_commands(buffer), idx);

        if (AicJit_AicResponseInfo_vec_len(
                AicJit_AicCommand_cmdRespInfo(cmd)) != 0) {
          AicJit_AicResponseInfo_struct_t rspInfo =
              AicJit_AicResponseInfo_vec_at(AicJit_AicCommand_cmdRespInfo(cmd),
                                            0);
          g_rspOffset = AicJit_AicResponseInfo_offset(rspInfo);
          g_rspSize = AicJit_AicResponseInfo_size(rspInfo);
          LOG_DEBUG(qctx->logFuncPtr,
                    "Command Response Offset : %llu size : %llu",
                    LOG64_DATA(g_rspOffset), LOG64_DATA(g_rspSize));

          if (g_rspSize != 0) {
            if ((g_rspSize % 128) != 0 || g_rspOffset == 0) {
              initStatus = JIT_DEV_ERROR_COMMANDS_VERIFICATION;
              qshimLog(NNC_LOG_MASK_ERROR, 0,
                       "Invalid parameters in Command Response Info");
            } else {
              void *rbuff = (void *)((char *)qctx->baseSharedDDR +
                                     g_respInfo.offset + g_rspOffset);
              memset(rbuff, 0, g_rspSize);
              g_rspTlv = (ResponseTLV_t *)rbuff;
              g_rspTlv->type = CommandResponse;
              g_rspTlv->slotCount = 1;
              CommandValue_t *val = (CommandValue_t *)&g_rspTlv->value[0];
            }
          }

          if (initStatus == JIT_DEV_STATUS_SUCCESS) {
            static JitDevStatusCode_t cmdStatus = JIT_DEV_STATUS_SUCCESS;

            if (profilingBitmap) {
              // initialize the structure using bitmap offset
              AicJit_AicInit_table_t aicInit =
                  AicJit_AicCommandSet_init(buffer);
              AicJit_AicResponseInfo_struct_t profRspInfo =
                  AicJit_ProfilingConfiguration_profilingRespInfo(
                      AicJit_AicInit_profilingConfig(aicInit));
              uint32_t cmdSetId = AicJit_AicCommandSet_commandSetID(buffer);
              g_profCmdId = AicJit_AicCommand_commandID(cmd);
              uint64_t profRspOffset =
                  AicJit_AicResponseInfo_offset(profRspInfo);
              g_profRspSize = AicJit_AicResponseInfo_size(profRspInfo);
              LOG_DEBUG(qctx->logFuncPtr,
                        "Command Profiling Offset : %llu size : %llu",
                        LOG64_DATA(profRspOffset), LOG64_DATA(g_profRspSize));
              void *pbuff = (void *)((char *)qctx->baseSharedDDR +
                                     profRspOffset + g_respInfo.offset);
              // clean the status buffer
              memset(pbuff, 0, g_profRspSize);
              g_profTlv = (ResponseTLV_t *)pbuff;
              g_profHeader = (ProfilingData_t *)g_profTlv->value;
              g_profHeader->commandSetID = cmdSetId;
              g_profHeader->dataOverFlow = 0;
              g_profHeader->size = sizeof(ProfilingData_t);
              g_profStatusBuffer = g_profHeader->profilingData;
            }

            if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
              g_profTable[g_profCount].stop_ts = qcGetTimestamp();
              strcpy(g_profTable[g_profCount].tag, "C-InitResp");
              g_profCount++;
            }

            if (idx < numCommands) {
              qcAtomicStore(&waitForDispatch, 0);
              /* Wait for master signal only for slave NSPs*/
              if (qctx->virtualNSPId != 0) {
                if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
                  JITTimeStampEventData_t cmdPrepStartData = {
                      .uCycleCounts = g_mstrStart,
                      .deviceBootTimeUTC =
                          *((uint64_t *)qctx->baseUtcOffsetDDR)};
                  write_Profiling_Event(
                      qctx, "C-WaitMstrSig", g_profHeader->commandSetID, tid,
                      BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &cmdPrepStartData);
                }
                if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
                  JITTimeStampEventData_t cmdPrepStartData = {
                      .uCycleCounts = g_mstrStop,
                      .deviceBootTimeUTC =
                          *((uint64_t *)qctx->baseUtcOffsetDDR)};
                  write_Profiling_Event(
                      qctx, "C-WaitMstrSig", g_profHeader->commandSetID, tid,
                      END_PHASE, EVENT_TYPE_TIME_STAMP, &cmdPrepStartData);
                }
              }
              if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
                // PROF capture: C-Prep END for Master thread, write: CMD-Prep
                // BEGIN for Master thread
                JITTimeStampEventData_t cmdPrepStartData = {
                    .uCycleCounts = g_startTs,
                    .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
                write_Profiling_Event(
                    qctx, "C-Prep", g_profHeader->commandSetID, tid,
                    BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &cmdPrepStartData);
              }

              cmdStatus = processAicCommandSetCommands(qctx, buffer, idx, tid);

              if (cmdStatus == JIT_DEV_STATUS_SUCCESS) {
                // clean end
                if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
                  JITTimeStampEventData_t data = {
                      .uCycleCounts = qcGetTimestamp(),
                      .deviceBootTimeUTC =
                          *((uint64_t *)qctx->baseUtcOffsetDDR)};
                  write_Profiling_Event(
                      qctx, "C-Cleanup", g_profHeader->commandSetID, tid,
                      END_PHASE, EVENT_TYPE_TIME_STAMP, &data);
                }
                if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
                  JITTimeStampEventData_t data = {
                      .uCycleCounts = qcGetTimestamp(),
                      .deviceBootTimeUTC =
                          *((uint64_t *)qctx->baseUtcOffsetDDR)};
                  write_Profiling_Event(
                      qctx, "C-ExitPrep", g_profHeader->commandSetID, tid,
                      BEGIN_PHASE, EVENT_TYPE_TIME_STAMP, &data);
                }
              }
            }

            if (cmdStatus != JIT_DEV_STATUS_SUCCESS || !threadIdBitMap) {
              qcAtomicStore(&waitForDispatch, 1);
            }
          }
        } else {
          initStatus = JIT_DEV_ERROR_COMMANDS_VERIFICATION;
          qshimLog(NNC_LOG_MASK_ERROR, 0,
                   "Command Response Info is not present");
        }
      } else {
        initStatus = JIT_DEV_ERROR_COMMANDS_VERIFICATION;
        qshimLog(NNC_LOG_MASK_ERROR, 0, "AicCommand not present");
      }
    } else {
      // AicInit or Buffer descriptor setup failed
      LOG_WARN(qctx->logFuncPtr, "Init commands failed");
      qcAtomicStore(&waitForDispatch, 1);
    }

    if (profilingBitmap) {
      g_profTlv->type = ProfilingData;
      g_profTlv->slotCount = g_profRspSize / AIC_RESPONSE_SLOT_SIZE;
    }

    qcAtomicStore(&waitForDispatch, 1);
  } else {
    qcDoorbellLocalWaitEq(&waitForDispatch, tid, 1);
    if (initStatus == JIT_DEV_STATUS_SUCCESS) {
      if (threadIdBitMap & (1 << tid)) {
        qcDoorbellWaitEq(L2TCM_WORKER_THREADS_START_DB_OFFSET + tid, 1);
        if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
          // PROF capture: C-Exec Begin for worker thread
          uint64_t ucycle = 0, pcycle = 0;
          qcGetUandPcycles(&ucycle, &pcycle);
          JITTimeStampEventData_t cmdExeStartTimeStampData = {
              .uCycleCounts = ucycle,
              .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
          write_Profiling_Event(qctx, "C-Exec", g_profCmdId, tid, BEGIN_PHASE,
                                EVENT_TYPE_TIME_STAMP,
                                &cmdExeStartTimeStampData);
        }

        dispatchWorkerThreads((int32_t)tid, qctx);

        if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
          // PROF capture: C-Exec END for worker thread
          uint64_t ucycle = 0, pcycle = 0;
          qcGetUandPcycles(&ucycle, &pcycle);
          JITTimeStampEventData_t cmdExeEndTimeStampData = {
              .uCycleCounts = ucycle,
              .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
          write_Profiling_Event(qctx, "C-Exec", g_profCmdId, tid, END_PHASE,
                                EVENT_TYPE_TIME_STAMP, &cmdExeEndTimeStampData);
        }
      }
    }
  }

  // All threads atomically increment the post
  // calculation doorbell
  qcAtomicFetchAdd(&outputSync, 1);

  if (tid == MASTER_THREAD) {
    uint16_t cmdSetOutputSem = 1;
    // Tid0 waits for the local doorbell to be
    // greater than or equal to total thread count
    // Once the doorbell value is equal to number
    // of threads spawned, it means that all threads
    // are done with computation and the output can
    // be sent back to host
    qcDoorbellLocalWaitEq(&outputSync, tid, totalThreadCount());
    // PROF capture: calcualte response
    if (profilingBitmap & (1 << DEV_GRANULAR_DATA)) {
      if (g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
        g_profTable[g_profCount].start_ts = qcGetTimestamp();
      } else {
        LOG_ERROR(qctx->logFuncPtr,
                  "Profiling Error: Profiling table index overflow");
      }
    }

    cmdBuffRspHdr->status = initStatus;
    if (g_rspTlv) {
      ResponseTLV_t *rspTlv = g_rspTlv;
      cmdBuffRspHdr->tlvCount = rspTlv->slotCount;
    }
    if (g_profTlv) {
      ResponseTLV_t *rspTlv = g_profTlv;
      cmdBuffRspHdr->tlvCount += rspTlv->slotCount;
    }

    LOG_DEBUG(qctx->logFuncPtr,
              "Response for CommandSet ID: %d, status: %d and has TLVs: %d",
              cmdBuffRspHdr->commandID, cmdBuffRspHdr->status,
              cmdBuffRspHdr->tlvCount);

    if (cmdBuffRspHdr->tlvCount != 0) {
      if (g_rspTlv) {
        LOG_DEBUG(qctx->logFuncPtr, "tlv type: %d and slot count: %d",
                  g_rspTlv->type, g_rspTlv->slotCount);

        if (g_rspTlv->type == CommandResponse) {
          CommandValue_t *cmdVal = (CommandValue_t *)&g_rspTlv->value[0];
          LOG_DEBUG(qctx->logFuncPtr, "command ID: %d and status: %d",
                    cmdVal->commandID, cmdVal->status);
        }
      }

      if (g_profTlv) {
        LOG_DEBUG(qctx->logFuncPtr, "profTlv type: %d and profSlot count: %d",
                  g_profTlv->type, g_profTlv->slotCount);

        if (g_profTlv->type == ProfilingData) {
          ProfilingData_t *profData = (ProfilingData_t *)&g_profTlv->value[0];
          LOG_DEBUG(qctx->logFuncPtr, "command Set ID: %d",
                    profData->commandSetID);
        }
      }
    }

    if (profilingBitmap & (1 << DEV_GRANULAR_DATA) &&
        g_profCount < JIT_PROF_TABLE_MAX_SIZE) {
      g_profTable[g_profCount].stop_ts = qcGetTimestamp();
      strcpy(g_profTable[g_profCount].tag, "C-calRspSize");
      g_profCount++;
    }
    if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
      if (profilingBitmap & (1 << DEV_GRANULAR_DATA)) {
        JITTimeStampEventData_t dataStart = {
            .uCycleCounts = qcGetTimestamp(),
            .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
        write_Profiling_Event(qctx, "C-wProfTable", g_profHeader->commandSetID,
                              tid, BEGIN_PHASE, EVENT_TYPE_TIME_STAMP,
                              &dataStart);
        write_ProfTable_To_StatusBuffer(qctx, g_profHeader->commandSetID, tid);
        JITTimeStampEventData_t dataEnd = {
            .uCycleCounts = qcGetTimestamp(),
            .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
        write_Profiling_Event(qctx, "C-wProfTable", g_profHeader->commandSetID,
                              tid, END_PHASE, EVENT_TYPE_TIME_STAMP, &dataEnd);
      }
    }
    if (qctx->virtualNSPId == 0) {
      /* wait for response from all NSPs*/
      if (profilingBitmap & (1 << DEV_GRANULAR_DATA)) {
        JITTimeStampEventData_t data = {
            .uCycleCounts = qcGetTimestamp(),
            .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
        write_Profiling_Event(qctx, "C-MasterNspSem",
                              g_profHeader->commandSetID, tid, BEGIN_PHASE,
                              EVENT_TYPE_TIME_STAMP, &data);
      }
      processAicCommandSetSemaphoreWait(qctx, buffer, tid, JIT_EXIT_SEM);
      if (profilingBitmap & (1 << DEV_GRANULAR_DATA)) {
        JITTimeStampEventData_t data = {
            .uCycleCounts = qcGetTimestamp(),
            .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
        write_Profiling_Event(qctx, "C-MasterNspSem",
                              g_profHeader->commandSetID, tid, END_PHASE,
                              EVENT_TYPE_TIME_STAMP, &data);
      }

      if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
        JITTimeStampEventData_t data = {
            .uCycleCounts = qcGetTimestamp(),
            .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
        write_Profiling_Event(qctx, "C-ExitPrep", g_profHeader->commandSetID,
                              tid, END_PHASE, EVENT_TYPE_TIME_STAMP, &data);
      }

      /* Wait for any previous output to be sent*/
      if (opSemCounter == (JIT_CMD_SET_MAX_OUTPUT_SEM_VAL - 1)) {
        uint16_t cmdSetOutputDB =
            JIT_CMD_SET_INPUT_DB_BASE_INDEX + g_ringBufferDepth;
        qcSemaphoreInc(cmdSetOutputSem);
        qcDoorbellWaitEq(cmdSetOutputDB, 1);
        qcDoorbellWrite(cmdSetOutputDB, 0);
        // TODO: Check for outputsem value becoming 0 and assert.
        opSemCounter = 0;
        LOG_DEBUG(qctx->logFuncPtr, "Cmd Set output DB reset");
      } else {
        qcSemaphoreInc(cmdSetOutputSem);
        ++opSemCounter;
      }

      LOG_DEBUG(qctx->logFuncPtr, "Output buffers have been sent");
    } else {
      /* wait for response from all NSPs*/
      if (profilingBitmap & (1 << DEV_GRANULAR_DATA)) {
        JITTimeStampEventData_t data = {
            .uCycleCounts = qcGetTimestamp(),
            .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
        write_Profiling_Event(qctx, "C-SlveNspSemSig",
                              g_profHeader->commandSetID, tid, BEGIN_PHASE,
                              EVENT_TYPE_TIME_STAMP, &data);
      }
      /* Will be valid for slave NSPs */
      processAicCommandSetSemaphoreSignal(qctx, buffer, tid, JIT_EXIT_SEM);
      if (profilingBitmap & (1 << DEV_GRANULAR_DATA)) {
        JITTimeStampEventData_t data = {
            .uCycleCounts = qcGetTimestamp(),
            .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
        write_Profiling_Event(qctx, "C-SlveNspSemSig",
                              g_profHeader->commandSetID, tid, END_PHASE,
                              EVENT_TYPE_TIME_STAMP, &data);
      }
      if (profilingBitmap & (1 << COMMAND_EXECUTION_TIME)) {
        JITTimeStampEventData_t data = {
            .uCycleCounts = qcGetTimestamp(),
            .deviceBootTimeUTC = *((uint64_t *)qctx->baseUtcOffsetDDR)};
        write_Profiling_Event(qctx, "C-ExitPrep", g_profHeader->commandSetID,
                              tid, END_PHASE, EVENT_TYPE_TIME_STAMP, &data);
      }
    }

    if (libAicDevExeBaseAddress) {
      g_networkHeapFree(libAicDevExeBaseAddress);
      libAicDevExeBaseAddress = nullptr;
      numLibAicDevExes = 0;
    }

    if (listAicDevBuffDescNodes) {
      g_networkHeapFree(listAicDevBuffDescNodes);
      listAicDevBuffDescNodes = nullptr;
      numAicDevBuffDescNodes = 0;
    }

    // FixMe: Reset buffer residency in allocBufNodeList

    qcAtomicStore(&inputSync, 0);
    qcAtomicStore(&outputSync, 0);
  } else {
    // Other threads wait for the previous output
    // to be sent back to host
    qcDoorbellLocalWaitEq(&outputSync, tid, 0);
  }
}

void bitMaskResContextInit(BitMaskResContext_t *ctx, uint16_t maxSize) {
  for (uint16_t i = 0; i < (MAX_UDMA_REQS / U64_BIT_COUNT); i++) {
    ctx->res[i] = 0xFFFFFFFFFFFFFFFF;
  }
  ctx->size = maxSize;
  ctx->available = maxSize;
  ctx->allocated = 0;
}

JitDevStatusCode_t bitMaskResContextAlloc(BitMaskResContext_t *ctx,
                                          int32_t *dbNum) {
  uint64_t mBit = 0x1;
  uint8_t dbOffset = 0;

  if (ctx == NULL || dbNum == NULL) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s: Invalid parameter ctx: 0x%x, val: 0x%x",
              __func__, ctx, dbNum);
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  if (ctx->available == 0) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s: uDMA doorbell resource not available",
              __func__);
    return JIT_DEV_ERROR_UDMA_DB_RES_FAILURE;
  }

  for (uint16_t i = 0; i < (MAX_UDMA_REQS / U64_BIT_COUNT); i++) {
    if (ctx->res[i] != 0) {
      uint64_t pos = ctx->res[i] & ~(ctx->res[i] - 1);
      uint32_t idx = Q6_R_ct0_P(pos);
      ctx->res[i] &= ~pos;
      *dbNum = (i * U64_BIT_COUNT) + idx + L2TCM_UDMA_START_DB_OFFSET;
      break;
    }
  }
  ctx->allocated += 1;
  ctx->available -= 1;

  return JIT_DEV_STATUS_SUCCESS;
}

JitDevStatusCode_t bitMaskResContextFree(BitMaskResContext_t *ctx,
                                         int32_t *dbNum) {
  uint64_t mBit;
  uint64_t res;
  uint8_t resIdx = 0;

  if (ctx == NULL || dbNum == NULL) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s: Invalid parameter ctx: 0x%x, val: 0x%x",
              __func__, ctx, dbNum);
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  LOG_DEBUG(g_qctx->logFuncPtr, "%s: uDMA doorbell resource free dbNum: %d",
            __func__, *dbNum);

  int32_t dbResIdx = *dbNum - L2TCM_UDMA_START_DB_OFFSET;
  resIdx = dbResIdx / U64_BIT_COUNT;
  dbResIdx = dbResIdx % U64_BIT_COUNT;

  res = ctx->res[resIdx];

  mBit = ((uint64_t)0x1 << dbResIdx);
  if (res & mBit) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: Resource was already free dbNum: %d, dbResIdx: %d", __func__,
              *dbNum, dbResIdx);
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }
  /* Return the resource */
  res |= mBit;

  ctx->res[resIdx] = res;
  ctx->available += 1;
  ctx->allocated -= 1;

  return JIT_DEV_STATUS_SUCCESS;
}

void initializeQpcShim(AICExecContext *qctx, uint8_t tid, uint32_t stid) {
  if (tid == MASTER_THREAD) {
    g_qctx = qctx;
    uint16_t inputSem = 0;
    debugInfo = (DebugInfo_t *)(qctx->baseL2TCM + DEBUG_INFO_OFFSET);
    resetDebugInfoStruct();
    ringBufferIdx = 0;
    if (qctx->virtualNSPId == 0) {
      qcSemaphoreInc(inputSem);
    }

    if (loadFwSymbols(qctx) != JIT_DEV_STATUS_SUCCESS) {
      ERR_FATAL(qctx->errFuncPtr, "Failed to load fw symbols", 0, 0, 0);
    }

    bitMaskResContextInit(&uDmaDbBitMask, MAX_UDMA_REQS);
    AicQShimInitConfig_t *init_config =
        (AicQShimInitConfig_t *)(qctx->baseL2TCM + QSHIM_CONFIG_L2TCM_OFFSET);
    g_cmdBufSize = 8192;
    g_ringBufferDepth = 4;
    g_opTimeoutMs = 16000000;
    setCmdBufSize(g_cmdBufSize);
    setRingBufferDepth(g_ringBufferDepth);
    LOG_DEBUG(g_qctx->logFuncPtr,
              "Ring buffer size: %d Request buffer size: %d Operator "
              "timeout ms: %lld",
              g_ringBufferDepth, g_cmdBufSize, LOG64_DATA(g_opTimeoutMs));

    /* sanity check for configurable params */
    if (g_cmdBufSize == 0 || g_ringBufferDepth == 0) {
      ERR_FATAL(qctx->errFuncPtr, "Request buffer or buffer depth is 0", 0, 0,
                0);
    }

    // Reset L2TCM space to 0s
    init_config->reqBufferSize = 0;
    init_config->ringBufferSize = 0;
    init_config->opTimeoutMs = 0;

    g_exitFp = (nnc_exit_fp)registerExitFunc(&errExitThreadHandler);
    registerOpTimeoutPtr(&notifyOpTimeoutFunc);
  }
}

uint32_t qshimSyncThreads(uint32_t threadId,
                          AicJitThreadSyncStruct *kernelThreadSync) {

  if (kernelThreadSync == NULL ||
      (qcAtomicLoad((uint32_t *)&kernelThreadSync->numThreads) == 0)) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: kernelThreadSync is NULL or numThreads is %d. Return (-1)",
              __func__, kernelThreadSync->numThreads);
    return JIT_DEV_QSHIM_ERROR_SYNC_THREADS;
  }

  while ((qcAtomicLoad((uint32_t *)&kernelThreadSync->reserved) != 0) &&
         ((qcAtomicLoad((uint32_t *)&kernelThreadSync->reserved) !=
           ((1 << (qcAtomicLoad((uint32_t *)&kernelThreadSync->numThreads))) -
            1))))
    ;

  int32_t expValue = 0;
  while (!qcAtomicCompareExchange(&syncPointResLock, &expValue, 1)) {
    expValue = 0; // Reset expected value for the next iteration
  }

  if (qcAtomicLoad((uint32_t *)&kernelThreadSync->data) ==
      qcAtomicLoad((uint32_t *)&kernelThreadSync->numThreads)) {
    qcAtomicStore((uint32_t *)&kernelThreadSync->data, 0);
  }

  qcAtomicFetchAdd((uint32_t *)&kernelThreadSync->data, 1);

  qcAtomicStore((uint32_t *)&syncPointResLock, 0);
  qcDoorbellLocalWaitEq((uint32_t *)&kernelThreadSync->data, threadId,
                        kernelThreadSync->numThreads);

  qcAtomicFetchXor(&kernelThreadSync->reserved, (1 << threadId));

  return 0;
}

QShimUDmaHandle qshimLinearUdmaSubmit(uint32_t threadId, void *src,
                                      uint32_t size, void *dst,
                                      AicJitUdmaDescCommonAttrs *udmaDescAttrs,
                                      bool requireHandle, uint32_t *status) {
  /* Return invalid dbNum if failed to submit uDMA request
     or if a handle is not requested for a submit request */
  QShimUDmaHandle dbNum = INVALID_UDMA_HANDLE;

  if (status == nullptr) {
    return dbNum;
  }
  *status = JIT_DEV_STATUS_SUCCESS;

  if (size == 0) {
    LOG_DEBUG(g_qctx->logFuncPtr, "%s: Invalid size passed", __func__);
    *status = JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_SIZE;
    return dbNum;
  }

  int32_t expValue = 0;
  while (!qcAtomicCompareExchange(&uDmaDbBitMaskResLock, &expValue, 1)) {
    expValue = 0; // Reset expected value for the next iteration
  }

  if (udmaDescAttrs == nullptr) {
    LOG_DEBUG(g_qctx->logFuncPtr, "%s: Descriptor attributes not specified",
              __func__);
  }

  if (requireHandle) {
    if (bitMaskResContextAlloc(&uDmaDbBitMask, (int32_t *)&dbNum)) {
      qcAtomicStore((uint32_t *)&uDmaDbBitMaskResLock, 0);
      *status = JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_RES_ALLOC;
      return dbNum;
    }

    // Reset the uDMA doorbell
    qcDoorbellWrite(dbNum, 0);
  }

  LOG_DEBUG(g_qctx->logFuncPtr, "src: 0x%x, dst: 0x%x, size: 0x%x", src, dst,
            size);

  if (udmaDescAttrs) {
    LOG_DEBUG(g_qctx->logFuncPtr, "UdmaDescAttrs: order: 0x%x",
              udmaDescAttrs->order);
  }

  udmaSubmitLinearDesc(threadId, (const int8_t *)src, size, (const int8_t *)dst,
                       &uDmaDbVal, false /*waitForDone*/,
                       requireHandle /*updateDBs*/, dbNum,
                       udmaDescAttrs == nullptr ? 0 : udmaDescAttrs->order);

  qcAtomicStore((uint32_t *)&uDmaDbBitMaskResLock, 0);
  return dbNum;
}

QShimUDmaHandle qshim2DUdmaSubmit(uint32_t threadId, void *src, void *dst,
                                  AicJit2DUdmaDescAttrs *udmaDescAttrs,
                                  bool requireHandle, uint32_t *status) {
  /* Return invalid dbNum if failed to submit uDMA request
     or if a handle is not requested for a submit request */
  QShimUDmaHandle dbNum = INVALID_UDMA_HANDLE;

  if (status == nullptr) {
    return dbNum;
  }
  *status = JIT_DEV_STATUS_SUCCESS;

  if (udmaDescAttrs == nullptr) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: Descriptor attributes not specified. Return error",
              __func__);
    *status = JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_NO_DESC_ATTR;
    return dbNum;
  }

  if (udmaDescAttrs->width == 0) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: Invalid width passed in descriptor attributes", __func__);
    *status = JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_WIDTH;
    return dbNum;
  }

  if (udmaDescAttrs->height == 0) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: Invalid height passed in descriptor attributes", __func__);
    *status = JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_HEIGHT;
    return dbNum;
  }

  if (udmaDescAttrs->srcStride < udmaDescAttrs->width) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: Source stride (%d) is less than width (%d) in descriptor "
              "attributes",
              __func__, udmaDescAttrs->srcStride, udmaDescAttrs->width);
    *status = JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_SRC_STRIDE;
    return dbNum;
  }

  if (udmaDescAttrs->destStride < udmaDescAttrs->width) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: Destination stride (%d) is less than width (%d) in "
              "descriptor attributes",
              __func__, udmaDescAttrs->destStride, udmaDescAttrs->width);
    *status = JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_INVALID_DST_STRIDE;
    return dbNum;
  }

  int32_t expValue = 0;
  while (!qcAtomicCompareExchange(&uDmaDbBitMaskResLock, &expValue, 1)) {
    expValue = 0; // Reset expected value for the next iteration
  }

  if (requireHandle) {
    if (bitMaskResContextAlloc(&uDmaDbBitMask, (int32_t *)&dbNum)) {
      qcAtomicStore((uint32_t *)&uDmaDbBitMaskResLock, 0);
      *status = JIT_DEV_QSHIM_ERROR_UDMA_SUBMIT_RES_ALLOC;
      return dbNum;
    }

    // Reset the uDMA doorbell
    qcDoorbellWrite(dbNum, 0);
  }

  LOG_DEBUG(g_qctx->logFuncPtr, "src: 0x%x, dst: 0x%x, udmaDescAttrs: 0x%x",
            src, dst, udmaDescAttrs);
  LOG_DEBUG(g_qctx->logFuncPtr, "UdmaDescAttrs: height: %d, width: %d",
            udmaDescAttrs->height, udmaDescAttrs->width);
  LOG_DEBUG(g_qctx->logFuncPtr, "UdmaDescAttrs: destStride: %d, srcStride: %d",
            udmaDescAttrs->destStride, udmaDescAttrs->srcStride);
  if (udmaDescAttrs->udmaDescCommonAttrs) {
    LOG_DEBUG(g_qctx->logFuncPtr, "UdmaDescAttrs: order: 0x%x",
              udmaDescAttrs->udmaDescCommonAttrs->order);
  }

  udmaSubmit2DDesc(
      threadId, (const int8_t *)src, (const int8_t *)dst, udmaDescAttrs->height,
      udmaDescAttrs->width, udmaDescAttrs->destStride, udmaDescAttrs->srcStride,
      &uDmaDbVal, false /*waitForDone*/, requireHandle /*updateDBs*/, dbNum,
      udmaDescAttrs->udmaDescCommonAttrs == nullptr
          ? 0
          : udmaDescAttrs->udmaDescCommonAttrs->order);

  qcAtomicStore((uint32_t *)&uDmaDbBitMaskResLock, 0);

  return dbNum;
}

uint32_t qshimUDmaWait(QShimUDmaHandle dbNum) {
  if (dbNum == INVALID_UDMA_HANDLE) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s: Invalid handle received", __func__);
    return JIT_DEV_QSHIM_ERROR_UDMA_WAIT_INVALID_HANDLE;
  }

  auto *dbs = reinterpret_cast<uint32_t *>(g_qctx->baseL2TCM);
  LOG_DEBUG(g_qctx->logFuncPtr, "Wait for uDMA Doorbell: db[%d]: %d", dbNum,
            dbs[dbNum]);

  qcDoorbellWaitEq(dbNum, 1);

  // Check for uDMA error
  uint32_t dm0 = qcUdmaPoll();
  if (isUserDMAError(dm0)) {
    return JIT_DEV_QSHIM_ERROR_UDMA_WAIT_USER_DMA_ERROR;
  }
  LOG_DEBUG(g_qctx->logFuncPtr, "%s: dm0: 0x%x", __func__, dm0);

  qcDoorbellWrite(dbNum, 0);

  LOG_DEBUG(g_qctx->logFuncPtr, "Doorbell reset: db[%d]: %d", dbNum,
            dbs[dbNum]);

  int32_t expValue = 0;
  while (!qcAtomicCompareExchange(&uDmaDbBitMaskResLock, &expValue, 1)) {
    expValue = 0; // Reset expected value for the next iteration
  }

  if (bitMaskResContextFree(&uDmaDbBitMask, (int32_t *)&dbNum)) {
    qcAtomicStore((uint32_t *)&uDmaDbBitMaskResLock, 0);
    return JIT_DEV_QSHIM_ERROR_UDMA_WAIT_INVALID_PARAMETER;
  }

  qcAtomicStore((uint32_t *)&uDmaDbBitMaskResLock, 0);

  return 0;
}

uint32_t qshimGetRandomDataSubmit(uint32_t threadId, uint32_t *outPtr,
                                  uint32_t numElements) {
  if (outPtr == nullptr || numElements == 0 || threadId >= NUM_WORKER_THREADS) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s: Invalid paramters received", __func__);
    return JIT_DEV_QSHIM_ERROR_PRNG_INVALID_PARAMETER;
  }

  // Check for multiple submissions from single thread
  if (qcAtomicLoad(&prngWorkQueue[threadId].status) != PRNG_WQ_SLOT_EMPTY) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s: PRNG request in progress", __func__);
    return JIT_DEV_QSHIM_ERROR_PRNG_MULTIPLE_SUBMISSION;
  }

  prngWorkQueue[threadId].addr = outPtr;
  prngWorkQueue[threadId].numElements = numElements;
  qcAtomicStore(&prngWorkQueue[threadId].status, PRNG_WQ_SLOT_PENDING);
  qcAtomicFetchAdd(&prngWorkQueueCount, 1);

  LOG_DEBUG(g_qctx->logFuncPtr, "%s: Requested %d random numbers. TID 0x%x",
            __func__, numElements, threadId);

  return 0;
}

uint32_t qshimGetRandomDataWait(uint32_t threadId) {

  if (threadId >= NUM_WORKER_THREADS) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s: Invalid paramters received", __func__);
    return JIT_DEV_QSHIM_ERROR_PRNG_INVALID_PARAMETER;
  }

  while (qcAtomicLoad(&prngWorkQueue[threadId].status) != PRNG_WQ_SLOT_COMPLETE)
    ;
  qcAtomicFetchSub(&prngWorkQueueCount, 1);
  qcAtomicStore(&prngWorkQueue[threadId].status, PRNG_WQ_SLOT_EMPTY);
  LOG_DEBUG(g_qctx->logFuncPtr, "%s: PRNG TID 0x%x, status %d", __func__,
            threadId, prngWorkQueue[threadId].status);

  return 0;
}

uint8_t *qshimGetBaseVtcmAddr(void) { return g_qctx->baseVTCM; }

// Qshim Mmap API to be used by operators.
// bufHandle - Buffer handle received through operator API which is
//             passed as an input in the list of args to entry point
//             function.
// offset    - Offset into the buffer to start mapping at
// size      - Size to map
// flag      - To specify mapping attributes such as cacheable etc
// Return value is the virtual address. Will be NULL if the mmap call fails.
void *qshimMmap(void *bufHandle, uint64_t offset, uint32_t size, uint32_t flag,
                uint32_t *status) {
  AicBufferDescriptorNode_t *aicOpBuffDescNode = nullptr;
  void *virtAddr = nullptr;
  uint32_t blockOffset = 0, numBlocks = 0;
  uint64_t P16MB_PAGE = 0x1000000;
  uint64_t addrToMap = 0, addr = 0;
  OpMmapNode_t *opMmapNode = nullptr, *temp = nullptr;
  *status = JIT_DEV_STATUS_SUCCESS;

  LOG_DEBUG(g_qctx->logFuncPtr,
            "OperatorMmap: Buffer handle: 0x%lx, size: 0x%lx, offset: %llx",
            bufHandle, size, LOG64_DATA(offset));
  int32_t expValue = 0;
  while (!qcAtomicCompareExchange(&qshimMmapLock, &expValue, 1)) {
    expValue = 0; // Reset expected value for the next iteration
  }

  if (bufHandle == nullptr) {
    LOG_ERROR(g_qctx->logFuncPtr, "%s: Invalid buffer handle", __func__);
    *status = JIT_DEV_QSHIM_ERROR_MMAP_INVALID_PARAMETER;
    goto exit;
  }
  aicOpBuffDescNode = (AicBufferDescriptorNode_t *)bufHandle;

  if ((size > aicOpBuffDescNode->opMmapBufInfo.buffSize) ||
      ((size + offset) > aicOpBuffDescNode->opMmapBufInfo.buffSize)) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "OperatorMmap: buffer out of bounds, mmap size: 0x%lx, mmap "
              "offset: 0x%llx, buffer size: 0x%llx",
              size, LOG64_DATA(offset),
              LOG64_DATA(aicOpBuffDescNode->opMmapBufInfo.buffSize));
    *status = JIT_DEV_QSHIM_ERROR_MMAP_INVALID_PARAMETER;
    goto exit;
  }
  addr = aicOpBuffDescNode->opMmapBufInfo.buffAddr +
         aicOpBuffDescNode->opMmapBufInfo.offset + offset;

  LOG_DEBUG(g_qctx->logFuncPtr, "Bufaddr: 0x%llx, addr to map: 0x%llx ",
            LOG64_DATA(aicOpBuffDescNode->opMmapBufInfo.buffAddr),
            LOG64_DATA(addr));

  temp = aicOpBuffDescNode->opMmapBufInfo.opMmapNode;
  while (temp) {
    uint64_t upperBound =
        temp->opMmapNodeInfo.mapAddr + temp->opMmapNodeInfo.mapSize;
    uint64_t lowerBound = temp->opMmapNodeInfo.mapAddr;
    if (temp->opMmapNodeInfo.isMap == true &&
        (lowerBound <= addr && addr < upperBound) &&
        (lowerBound <= (addr + size) && (addr + size) < upperBound)) {

      temp->opMmapNodeInfo.opMmapCount++;
      virtAddr = (void *)(temp->opMmapNodeInfo.virtAddr +
                          (addr - temp->opMmapNodeInfo.mapAddr));
      goto exit;
    }
    temp = temp->next;
  }

  opMmapNode = (OpMmapNode_t *)g_networkHeapMalloc(sizeof(OpMmapNode_t));
  if (opMmapNode == nullptr) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: Failed to allocate memory for operator mmap node", __func__);
    *status = JIT_DEV_QSHIM_ERROR_MMAP;
    goto exit;
  }
  g_memset((void *)opMmapNode, 0, sizeof(OpMmapNode_t));

  addrToMap = addr & ~(P16MB_PAGE - 1);
  blockOffset = addr & (P16MB_PAGE - 1);
  numBlocks = (size + blockOffset) / P16MB_PAGE;
  numBlocks += (size + blockOffset) % P16MB_PAGE ? 1 : 0;

  if (numBlocks >= (JIT_MAX_BUF_DESC_SIZE / JIT_MMAP_BLOCK_SIZE)) {
    LOG_ERROR(g_qctx->logFuncPtr,
              "%s: MMAP size greater than 4GB due to 16MB alignment", __func__);
    *status = JIT_DEV_QSHIM_ERROR_MMAP;
    goto exit;
  }

  LOG_DEBUG(g_qctx->logFuncPtr,
            "OperatorMmap: aligned addr to map: 0x%llx, num blocks: %d, bufid: "
            "%d, blkoffset: 0x%lx",
            LOG64_DATA(addrToMap), numBlocks,
            aicOpBuffDescNode->opMmapBufInfo.bufId, blockOffset);

  opMmapNode->opMmapNodeInfo.mapSize = P16MB_PAGE * numBlocks;
  opMmapNode->opMmapNodeInfo.mapAddr = addrToMap;
  opMmapNode->opMmapNodeInfo.blockOffset = blockOffset;

  if (getOffsetAlpBase(g_qctx, opMmapNode->opMmapNodeInfo.mapAddr,
                       aicOpBuffDescNode->opMmapBufInfo.bufId,
                       &opMmapNode->opMmapNodeInfo.mapOffsetAlpBase)) {
    *status = JIT_DEV_QSHIM_ERROR_MMAP;
    goto exit;
  }

  if (mmapOpBufferNode(g_qctx, opMmapNode,
                       aicOpBuffDescNode->opMmapBufInfo.bufId)) {
    *status = JIT_DEV_QSHIM_ERROR_MMAP;
    goto exit;
  }
  opMmapNode->opMmapNodeInfo.opMmapCount++;
  virtAddr = (void *)(opMmapNode->opMmapNodeInfo.virtAddr +
                      opMmapNode->opMmapNodeInfo.blockOffset);

  opMmapNode->next = aicOpBuffDescNode->opMmapBufInfo.opMmapNode;
  aicOpBuffDescNode->opMmapBufInfo.opMmapNode = opMmapNode;

exit:
  if (*status != JIT_DEV_STATUS_SUCCESS && opMmapNode) {
    g_networkHeapFree(opMmapNode);
  }
  qcAtomicStore((uint32_t *)&qshimMmapLock, 0);
  return virtAddr;
}

// Qshim Unmap API to be used by operators.
// vaAddr - Takes in virtual address, returned by mmap API, as input
//          and unmaps the buffer.
void qshimUnmap(void *vaAddr) {
  LOG_DEBUG(g_qctx->logFuncPtr, "%s: VA addr 0x%x", __func__, vaAddr);
  int32_t expValue = 0;
  while (!qcAtomicCompareExchange(&qshimMmapLock, &expValue, 1)) {
    expValue = 0; // Reset expected value for the next iteration
  }
  AicBufferDescriptorNode_t *aicOpBuffDescNode = headAicOpBuffDescList;
  while (aicOpBuffDescNode) {
    OpMmapNode_t *temp = aicOpBuffDescNode->opMmapBufInfo.opMmapNode;
    OpMmapNode_t *prev = nullptr;
    while (temp) {
      if ((temp->opMmapNodeInfo.virtAddr + temp->opMmapNodeInfo.blockOffset <=
           (uint32_t)vaAddr) &&
          ((uint32_t)vaAddr) < (temp->opMmapNodeInfo.virtAddr +
                                temp->opMmapNodeInfo.blockOffset +
                                temp->opMmapNodeInfo.mapSize)) {
        LOG_DEBUG(g_qctx->logFuncPtr,
                  "%s: Node VA: 0x%x, map size: 0x%x, map count: %d", __func__,
                  temp->opMmapNodeInfo.virtAddr, temp->opMmapNodeInfo.mapSize,
                  temp->opMmapNodeInfo.opMmapCount);
        temp->opMmapNodeInfo.opMmapCount--;
        if (temp->opMmapNodeInfo.opMmapCount == 0) {
          g_qctx->munmapFuncPtr(temp->opMmapNodeInfo.nodeIndex);
          temp->opMmapNodeInfo.isMap = false;
          if (prev) {
            prev->next = temp->next;
          } else {
            aicOpBuffDescNode->opMmapBufInfo.opMmapNode = temp->next;
          }
          temp->next = nullptr;
          g_networkHeapFree(temp);
        }
        goto exit;
      } else {
        prev = temp;
        temp = temp->next;
      }
    }
    aicOpBuffDescNode = aicOpBuffDescNode->next;
  }
exit:
  qcAtomicStore((uint32_t *)&qshimMmapLock, 0);
  return;
}
