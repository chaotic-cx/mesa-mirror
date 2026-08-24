// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "ComputeAPI.h"

#include "NSPContext.h"
#include "SerializedProgramDesc.h"
#include "libdev/os-inlines.h"
#include "libdev/os.h"

namespace qaic {

bool isBufferValid(int buffNum) {
  const BufferDesc_t *buff = &_progBuffers[buffNum];
  const uint16_t waitDBNum = buff->waitDBNum;
  const uint32_t *dbs = (uint32_t *)getNSPContext()->baseL2TCM;
  return (buff->waitDBVal == dbs[waitDBNum]);
}

void clearBufferValid(int buffNum) {
  const BufferDesc_t *buff = &_progBuffers[buffNum];
  const uint16_t waitDBNum = buff->waitDBNum;
  uint32_t *dbs = (uint32_t *)getNSPContext()->baseL2TCM;
  os_doorbell_local_write4b(&dbs[waitDBNum], 0);
}

void waitForBuffer(int buffNum, uint32_t waitDBVal, bool clear) {
  const BufferDesc_t *buff = &_progBuffers[buffNum];
  uint16_t waitDBNum = buff->waitDBNum;
  uint32_t *dbs = (uint32_t *)getNSPContext()->baseL2TCM;
  os_doorbell_wait_eq((nsp_doorbell_t)&dbs[waitDBNum], waitDBVal,
                      /*doTimeoutCheck*/ false,
                      /*threadId*/ 0);
  if (clear) {
    os_doorbell_local_write4b(&dbs[waitDBNum], 0);
  }
}

void waitForBuffer(int buffNum, bool clear) {
  const BufferDesc_t *buff = &_progBuffers[buffNum];
  waitForBuffer(buffNum, buff->waitDBVal, clear);
}

void waitForAllInputsReady(bool clear) {
  const uint16_t numBuffs = _progDesc->numInputBuffs;
  const uint16_t numInputs = _progDesc->numInputBuffs;
  uint16_t numInputsReady = 0;

  for (int buffNum = 0; (numInputsReady < numInputs) && (buffNum < numBuffs);
       buffNum++) {
    const BufferDesc_t *buff = &_progBuffers[buffNum];
    if (buff->usage == USAGE_INPUT) {
      waitForBuffer(buffNum, clear);
      numInputsReady++;
    }
  }
  if (numInputsReady != numInputs) {
    ERR_FATAL(getNSPContext()->errFuncPtr,
              "Only found %d ready inputs, but expected to find %d",
              numInputsReady, numInputs, 0);
    __builtin_unreachable();
  }
}

void waitForAllOutputsReady(bool clear) {
  const uint16_t numBuffs = _progDesc->numBuffs;
  const uint16_t numOutputs = _progDesc->numOutputBuffs;
  uint16_t numOutputsReady = 0;

  for (int buffNum = 0; (numOutputsReady < numOutputs) && (buffNum < numBuffs);
       buffNum++) {
    const BufferDesc_t *buff = &_progBuffers[buffNum];
    if (buff->usage == USAGE_OUTPUT) {
      waitForBuffer(buffNum, clear);
      numOutputsReady++;
    }
  }
  if (numOutputsReady != numOutputs) {
    ERR_FATAL(getNSPContext()->errFuncPtr,
              "Only found %d ready outputs, but expected to find %d",
              numOutputsReady, numOutputs, 0);
    __builtin_unreachable();
  }
}

void readyForAllInputs(bool waitForArrival, bool clear) {
  // Input from host
  CoreInfo *ctx = getNSPContext();
  if ((0x1 << ctx->virtualNSPId) & _progDesc->hasInputsMask) {
    uint16_t inputSem = _progDesc->inputSem;

    hostsem_t semAddr =
        (hostsem_t *)getNSPContext()->semInfo[inputSem].semAddress;
    auto fwSemIdx = getNSPContext()->semInfo[inputSem].semNum;

    // Clear the input doorbells before doing the host increment
    const uint16_t numBuffs = _progDesc->numBuffs;
    const uint16_t numInputs = _progDesc->numInputBuffs;
    uint32_t *dbs = (uint32_t *)getNSPContext()->baseL2TCM;
    uint16_t numInputsCleared = 0;
    for (int buffNum = 0;
         (numInputsCleared < numInputs) && (buffNum < numBuffs); buffNum++) {
      const BufferDesc_t *buff = &_progBuffers[buffNum];
      if (buff->usage == USAGE_INPUT) {
        uint16_t waitDBNum = buff->waitDBNum;
        os_doorbell_local_write4b(&dbs[waitDBNum], 0);
        numInputsCleared++;
      }
    }

    // Make sure all reads are done before doing semaphore
    os_release_allthreads(&ctx->inputSemaphoreReleaseLoc);
    os_load_acquire(&ctx->inputSemaphoreReleaseLoc);

    os_hostsem_inc(semAddr, static_cast<uint32_t>(fwSemIdx));
    if (waitForArrival) {
      waitForAllInputsReady(clear);
    }
  } else {
    NN_LOG(ctx->logFuncPtr, NNC_LOG_MASK_WARN,
           "NSP %d called readyForAllInputs, but doesn't have inputs",
           ctx->virtualNSPId);
  }
}

void sendAllOutputs(bool waitForArrival, bool clear) {
  // Output to host
  CoreInfo *ctx = getNSPContext();
  if ((0x1 << ctx->virtualNSPId) & _progDesc->hasInputsMask) {
    uint16_t outputSem = _progDesc->outputSem;

    hostsem_t semAddr =
        (hostsem_t *)getNSPContext()->semInfo[outputSem].semAddress;
    int fwSemIdx = getNSPContext()->semInfo[outputSem].semNum;

    // Clear the output doorbells before doing the host decrement
    const uint16_t numBuffs = _progDesc->numBuffs;
    const uint16_t numOutputs = _progDesc->numOutputBuffs;
    uint32_t *dbs = (uint32_t *)getNSPContext()->baseL2TCM;
    uint16_t numOutputsCleared = 0;
    for (int buffNum = 0;
         (numOutputsCleared < numOutputs) && (buffNum < numBuffs); buffNum++) {
      const BufferDesc_t *buff = &_progBuffers[buffNum];
      if (buff->usage == USAGE_OUTPUT) {
        uint16_t waitDBNum = buff->waitDBNum;
        os_doorbell_local_write4b(&dbs[waitDBNum], 0);
        numOutputsCleared++;
      }
    }

    os_global_memsync();
    os_hostsem_dec(semAddr, fwSemIdx);
    if (waitForArrival) {
      waitForAllOutputsReady(clear);
    }
  } else {
    NN_LOG(ctx->logFuncPtr, NNC_LOG_MASK_WARN,
           "NSP %d called sendAllOutputs, but doesn't have outputs",
           ctx->virtualNSPId);
  }
}

void logActivate(uint8_t virtualThreadId) {
  CoreInfo *ctx = getNSPContext();

  NN_LOG(ctx->logFuncPtr, NNC_LOG_MASK_INFO,
        "NN_ACTIVATE_THREAD:  NSP %d Thread %d",
        ctx->virtualNSPId, virtualThreadId);
}
void logDeactivate(uint8_t virtualThreadId) {
  CoreInfo *ctx = getNSPContext();

  NN_LOG(ctx->logFuncPtr, NNC_LOG_MASK_INFO,
        "NN_DEACTIVATE_THREAD:  NSP %d Thread %d",
        ctx->virtualNSPId, virtualThreadId);
}

void *registerExitFunc(void (*exitFunc)()) {
  CoreInfo *ctx = getNSPContext();
  void *oldExitThreadFp = (void *)ctx->exitThread;
  ctx->exitThread = exitFunc;
  return oldExitThreadFp;
}

void quitIfExitDBSet(uint8_t virtualThreadId) {
  CoreInfo *ctx = getNSPContext();
  OSTimeoutCheckContext timeoutCtx(virtualThreadId, ctx->exitDB(),
                                   /* expectedVal */ 1);
  os_timeout_check(&timeoutCtx, /* dbval */ 0, /* doTimeoutCheck */ false,
                   /* debugLog */ false);
}

void qcAtomicStore(uint32_t *atomicVar, uint32_t val) {
  atomic_store_n(atomicVar, val);
}
uint32_t qcAtomicLoad(uint32_t *atomicVar) { return atomic_load_n(atomicVar); }

uint16_t getExitDbNum() { return _progDesc->exitDB; }

uint32_t qcAtomicFetchAdd(uint32_t *atomicVar, uint32_t val) {
  return atomic_fetch_add(atomicVar, val);
}

int qcAtomicFetchOr(int *atomicVar, int val) {
  return atomic_fetch_or(atomicVar, val);
}

int32_t qcAtomicFetchXor(int32_t *atomicVar, int32_t val) {
  return atomic_fetch_xor(atomicVar, val);
}

int32_t qcAtomicCompareExchange(int32_t *atomicVar, int32_t *expected_val,
                                int32_t val) {
  return atomic_compare_exchange_n(atomicVar, expected_val, val);
}

void qcAtomicFetchSub(uint32_t *atomicVar, uint32_t val) {
  atomic_fetch_sub(atomicVar, val);
}

int32_t getUtimerFreqMs() { return utimer_freq_ms(); }


int totalThreadCount() {
  // Hardcoded return value since there are 4 HVX threads spawned
  return 6;
}

void qcDoorbellLocalWaitGe(uint32_t *var, int virtualThreadId, uint32_t val) {
  os_doorbell_local_wait_ge(var, val,
                            /*doTimeoutCheck*/ false,
                            /*threadId*/ virtualThreadId);
}

void qcDoorbellLocalWaitEq(uint32_t *var, int virtualThreadId, uint32_t val) {
  os_doorbell_local_wait_eq(var, val,
                            /*doTimeoutCheck*/ false,
                            /*threadId*/ virtualThreadId);
}

void setWaitTimeOut(uint64_t value) {
  getNSPContext()->waitTimeoutLogMS = value / utimer_freq_ms();
}

void registerOpTimeoutPtr(void (*notifyTimeoutFunc)()) {
  getNSPContext()->opTimeoutPtr = notifyTimeoutFunc;
}

} // namespace qaic
