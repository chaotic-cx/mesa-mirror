// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef OS_H
#define OS_H

#include "../AICDefsInternal.h"
#include "AICMetadataExecCtx.h"
#include <stdint.h>

extern "C" {

void os_global_memsync();

// Host/PCI semaphores.
typedef void *hostsem_t;
void os_hostsem_init(hostsem_t s, uint32_t semNum, uint32_t val);
void os_hostsem_inc(hostsem_t s, uint32_t semNum);
void os_hostsem_dec(hostsem_t s, uint32_t semNum);

// NSP TCM doorbells.
typedef void *nsp_doorbell_t;
void os_doorbell_local_write4b(nsp_doorbell_t db, uint32_t val);
uint8_t os_doorbell_read1b(nsp_doorbell_t db);
uint32_t os_doorbell_read4b_acquire(nsp_doorbell_t db);
void os_doorbell_wait_eq(nsp_doorbell_t db, uint32_t val, bool doTimeoutCheck,
                         int threadId);

// Acquire/release memory ordering
void os_release_allthreads(void *addr);
uint32_t os_load_acquire(uint32_t *addr);

// UserDMA
using aic::DMADescriptor;
void os_udma_link(DMADescriptor *tailDmaDescriptor, DMADescriptor *dmaDescriptor,
                  bool doRelease = true);
uint32_t os_udma_poll_nocheck();
uint32_t os_udma_poll();
void os_udma_report_error(uint32_t dm0);
uint32_t os_udma_wait();
void os_udma_wait_done(const DMADescriptor *dmaDescriptor, int threadId);
void debugPrintUdmaDesc(const aic::DMADescriptor *p);

uint64_t os_get_system_timestamp();

// Debugging
struct OSTimeoutCheckContext {
  OSTimeoutCheckContext(int threadId, nsp_doorbell_t db, uint32_t expectedVal)
      : threadId(threadId), db(db), expectedVal(expectedVal) {}
  int threadId;
  nsp_doorbell_t db;
  uint32_t expectedVal;
  uint64_t waitStart{0};
  int waitExceededCount{0};
};

void os_timeout_check(OSTimeoutCheckContext *timeoutCtx, uint32_t dbval,
                      bool doTimeoutCheck, bool debugLog);
}; // extern "C"

#endif // OS_H
