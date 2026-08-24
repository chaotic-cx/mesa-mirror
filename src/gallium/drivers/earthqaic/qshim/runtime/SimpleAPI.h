// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#ifndef _QAIC_SIMPLEAPI_H_
#define _QAIC_SIMPLEAPI_H_

#include <stdint.h>

namespace qaic {
/* This set of APIs are direct access of the semaphores and doorbells */

using CleanUpFp = void (*)();

void qcDoorbellWrite(uint16_t dbNum, uint32_t dbVal);
void qcDoorbellWaitEq(uint16_t dbNum, uint32_t dbVal);
void qcDoorbellWaitEqWithTimeoutCheck(uint16_t dbNum, uint32_t dbVal);
void qcDoorbellWaitGe(uint16_t dbNum, uint32_t dbVal);
void qcDoorbellWaitGt(uint16_t dbNum, uint32_t dbVal);
void qcDoorbellWaitLe(uint16_t dbNum, uint32_t dbVal);
void qcDoorbellWaitLt(uint16_t dbNum, uint32_t dbVal);
bool qcDoorbellCheckGe(uint16_t dbNum, uint32_t dbVal);
void qcDoorbellUnicast(uint32_t mcId, uint16_t dbNum, uint32_t dbVal);
uint32_t qcDoorbellRead(uint16_t dbNum);
uint32_t qcHvxThreadCount(void);
uint32_t qcUdmaPoll(void);

void qcSemaphoreInit(uint16_t sem, uint32_t val);
void qcSemaphoreInc(uint16_t sem);
void qcSemaphoreDec(uint16_t sem);

void qcNanoSleep(int sleepCount);

void qcSetCleanUpFunc(CleanUpFp cleanUpFunc);

uint64_t qcGetTimestamp(void);

void qcGetUandPcycles(uint64_t *ucycle, uint64_t *pcycle);

} // namespace qaic
#endif
