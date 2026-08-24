// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "SimpleAPI.h"
#include "libdev/os-inlines.h"
#include "libdev/os.h"

namespace qaic {

void qcDoorbellWrite(uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  os_doorbell_local_write4b(&dbs[dbNum], dbVal);
}

void qcDoorbellWaitEq(uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  os_doorbell_wait_eq(&dbs[dbNum], dbVal,
                      /*doTimeoutCheck*/ false,
                      /*threadId*/ 0);
}

void qcDoorbellWaitEqWithTimeoutCheck(uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  os_doorbell_wait_eq(&dbs[dbNum], dbVal,
                      /*doTimeoutCheck*/ true,
                      /*threadId*/ 0);
}

void qcDoorbellWaitGe(uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  os_doorbell_wait_ge(&dbs[dbNum], dbVal,
                      /*doTimeoutCheck*/ false,
                      /*threadId*/ 0);
}

void qcDoorbellWaitGt(uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  os_doorbell_wait_ge(&dbs[dbNum], dbVal,
                      /*doTimeoutCheck*/ false,
                      /*threadId*/ 0);
}

void qcDoorbellWaitLe(uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  os_doorbell_wait_le(&dbs[dbNum], dbVal,
                      /*doTimeoutCheck*/ false,
                      /*threadId*/ 0);
}

void qcDoorbellWaitLt(uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  os_doorbell_wait_lt(&dbs[dbNum], dbVal,
                      /*doTimeoutCheck*/ false,
                      /*threadId*/ 0);
}

bool qcDoorbellCheckGe(uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  if (os_doorbell_read4b_acquire(&dbs[dbNum]) >= dbVal) {
    return true;
  }
  return false;
}

uint32_t qcDoorbellRead(uint16_t dbNum) {
  auto *dbs = reinterpret_cast<uint32_t *>(getNSPContext()->baseL2TCM);
  return os_doorbell_read4b_acquire(&dbs[dbNum]);
}

void qcSemaphoreInit(uint16_t sem, uint32_t val) {
  auto semAddr =
      reinterpret_cast<hostsem_t *>(getNSPContext()->semInfo[sem].semAddress);
  int semIdx = getNSPContext()->semInfo[sem].semNum;
  os_hostsem_init(semAddr, semIdx, val);
}

void qcSemaphoreInc(uint16_t sem) {
  auto semAddr =
      reinterpret_cast<hostsem_t *>(getNSPContext()->semInfo[sem].semAddress);
  int semIdx = getNSPContext()->semInfo[sem].semNum;
  os_hostsem_inc(semAddr, semIdx);
}

void qcSemaphoreDec(uint16_t sem) {
  auto semAddr =
      reinterpret_cast<hostsem_t *>(getNSPContext()->semInfo[sem].semAddress);
  int semIdx = getNSPContext()->semInfo[sem].semNum;
  os_hostsem_dec(semAddr, semIdx);
}

void qcNanoSleep(int sleepCount) {
  auto *p = os_get_nanosleep_ptr();
  os_thread_nanosleep(sleepCount, p);
}

static nnc_exit_fp oldExitThreadFp = nullptr;
static CleanUpFp cleanUpFp = nullptr;

static volatile int threadCountDown = 4;
static void qcExitThread() {
  int val = atomic_fetch_sub<volatile int>(&threadCountDown, 1);
  if (val == 1 && cleanUpFp) {
    // last thread
    cleanUpFp();
  }
  if (oldExitThreadFp) {
    oldExitThreadFp();
  }
}

// CleanUpFunc is called by the last exiting thread
void qcSetCleanUpFunc(CleanUpFp cleanUpFunc) {
  auto *ctx = getNSPContext();
  if (ctx->exitThread) {
    oldExitThreadFp = ctx->exitThread;
  }
  cleanUpFp = cleanUpFunc;
  ctx->exitThread = qcExitThread;
}

uint64_t qcGetTimestamp() { return os_get_system_timestamp(); }

void qcGetUandPcycles(uint64_t *ucycle, uint64_t *pcycle) {
  os_get_ucycles_pcycles(ucycle, pcycle);
}

void qcDoorbellUnicast(uint32_t mcId, uint16_t dbNum, uint32_t dbVal) {
  auto *dbs = reinterpret_cast<void *>(getNSPContext()->mcAddresses[mcId]);
  os_mc_write_atomic4b(dbs, dbNum * 4, dbVal);
}

uint32_t qcHvxThreadCount(void) { return 4; }

uint32_t qcUdmaPoll(void) { return os_udma_poll(); }

} // namespace qaic
