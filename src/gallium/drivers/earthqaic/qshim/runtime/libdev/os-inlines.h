// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef OS_INLINES_H
#define OS_INLINES_H

#include "../AICDefsInternal.h"
#include "AICMetadata.h"
#include "libdev_interface.h"
#include <inttypes.h>

using namespace aic;


template <class T> T atomic_fetch_sub(T *ptr, T val) {
  return __atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST);
}

template <class T> void atomic_store_n(T *ptr, T val) {
  __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST);
}

template <class T> T atomic_load_n(T *ptr) {
  return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
}

template <class T> T atomic_fetch_add(T *ptr, T val) {
  return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
}

template <class T> T atomic_fetch_and(T *ptr, T val) {
  return __atomic_fetch_and(ptr, val, __ATOMIC_SEQ_CST);
}

template <class T> T atomic_fetch_or(T *ptr, T val) {
  return __atomic_fetch_or(ptr, val, __ATOMIC_SEQ_CST);
}

template <class T> T atomic_fetch_xor(T *ptr, T val) {
  return __atomic_fetch_xor(ptr, val, __ATOMIC_SEQ_CST);
}

template <class T> T atomic_compare_exchange_n(T *ptr, T *expected, T desired) {
  // bool __atomic_compare_exchange_n (type *ptr, type *expected, type desired,
  // bool weak, int success_memorder, int failure_memorder)
  return __atomic_compare_exchange_n(ptr, expected, desired, false,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

/// Returns the value of \p x rounded up to the next mutliple of \p m.
inline constexpr uint64_t alignTo(uint64_t x, uint64_t m) {
  return (x + m - 1) / m * m;
}

// This is just a simple store, but we want to prevent the compiler
// from re-ordering it with other memory operations.  We could use
// an atomic store, but that does a mem_locked operation, which we
// don't need here since any required exclusive access to this
// address should be guarenteed at a higher level.
inline void hexagon_atomic_store_nolock1b(uint8_t *p, uint8_t val) {
  asm("memb(%0+#0) = %1" : : "r"(p), "r"(val) : "memory");
}
inline void hexagon_atomic_store_nolock2b(uint16_t *p, uint16_t val) {
  asm("memh(%0+#0) = %1" : : "r"(p), "r"(val) : "memory");
}
inline void hexagon_atomic_store_nolock4b(uint32_t *p, uint32_t val) {
  asm("memw(%0+#0) = %1" : : "r"(p), "r"(val) : "memory");
}
inline uint32_t hexagon_atomic_load_nolock4b_acquire(uint32_t *p) {
  uint32_t val;
  asm("%0 = memw_aq(%1)" : "=r"(val) : "r"(p) : "memory");
  return val;
}

// We can just use atomic load here, since that doesn't get compiled down to
// mem_locked and will prevent the compiler from re-ordering.
template <typename T> inline T os_doorbell_read(nsp_doorbell_t db) {
  T *dba = (T *)db;
  return atomic_load_n(dba);
}

inline void os_thread_nanosleep(const int sleepCount,
                                const volatile uint32_t *p) {
  switch (sleepCount) {
  case 0:
    // Errata workaround
    (void)p;
    os_udma_poll_nocheck();
    asm volatile("isync;\n");
    break;
  case 255:
    asm volatile("pause(#255);\n");
    break;
  default:
    asm volatile("pause(#8);\n");
  }
}

// Get short wait ptr for passing to os_thread_nanosleep
inline const volatile uint32_t *os_get_nanosleep_ptr(void) {
  return ((uint32_t *)libdev_getcontext()->baseL2TCM);
}

template <bool isLocal, bool isDMAPossiblyActive, typename Compare>
static void os_doorbell_wait(nsp_doorbell_t db, uint32_t val, Compare &comp,
                             bool doTimeoutCheck, int threadId) {
  // 1.	Synchronizing threads on the same NSP
  //    Loop waiting for a doorbell update with small pause
  // 2.	Synchronizing with other NSPs
  //    Other threads on NSP may still be active
  //    Loop waiting for a doorbell update with a large pause,
  //      doorbell write via slave port will wake paused threads
  // 3.	Synchronizing with PCIDMA
  //    All threads on NSP will be waiting
  //    This is the idle state of an active network between inferences
  //    Loop waiting for a doorbell update with a large pause,
  //      doorbell write via slave port will wake paused threads

  int sleepCount = isLocal ? 0 : 255;
  const volatile uint32_t *nanosleep_ptr = os_get_nanosleep_ptr();
  OSTimeoutCheckContext timeoutCtx(threadId, db, val);
  // timeoutCheckIter is selected to arrange for os_timeout_check to be called
  // around every 1ms
  int timeoutCheckIter = isLocal ? 10000 : 1000;
  int timeoutIter = 0;

  uint32_t dbval;
  // Use __builtin_expect to make the case where the doorbell is already met the
  // fast path since the other case is just waiting anyway.
  while (__builtin_expect(
      !comp(dbval = os_doorbell_read4b_acquire((uint32_t *)db), val), false)) {
    // A dmwait/dmpoll is required in order to expose page exceptions.
    // If this is a local wait, this poll will already happen in the next call
    // to os_thread_nanosleep().
    if (isDMAPossiblyActive && !isLocal)
      os_udma_poll();

    // If the doorbell arrives immediately before this sleep,
    // there is a chance of a missed wakeup that would cause us to pause
    // for the whole pause amount. We need to consider that likelihood
    // when selecting the sleepCount.
    os_thread_nanosleep(sleepCount, nanosleep_ptr);

    if (++timeoutIter == timeoutCheckIter) {
      os_timeout_check(&timeoutCtx, dbval, doTimeoutCheck, false);
      timeoutIter = 0;
    }
  }
}

extern "C" {

inline void os_doorbell_wait_eq(nsp_doorbell_t db, uint32_t val,
                                bool doTimeoutCheck, int threadId) {
  auto cmpEq = [](uint32_t dbval, uint32_t waitval) {
    return dbval == waitval;
  };
  os_doorbell_wait</*isLocal=*/false, /*isDMAPossiblyActive=*/false>(
      db, val, cmpEq, doTimeoutCheck, threadId);
}

inline void os_doorbell_wait_ge(nsp_doorbell_t db, uint32_t val,
                                bool doTimeoutCheck, int threadId) {
  auto cmpGe = [](uint32_t dbval, uint32_t waitval) {
    return dbval >= waitval;
  };
  os_doorbell_wait</*isLocal=*/false, /*isDMAPossiblyActive=*/false>(
      db, val, cmpGe, doTimeoutCheck, threadId);
}

inline void os_doorbell_wait_le(nsp_doorbell_t db, uint32_t val,
                                bool doTimeoutCheck, int threadId) {
  auto cmpLe = [](uint32_t dbval, uint32_t waitval) {
    return dbval <= waitval;
  };
  os_doorbell_wait</*isLocal=*/false, /*isDMAPossiblyActive=*/false>(
      db, val, cmpLe, doTimeoutCheck, threadId);
}

inline void os_doorbell_wait_lt(nsp_doorbell_t db, uint32_t val,
                                bool doTimeoutCheck, int threadId) {
  auto cmpLt = [](uint32_t dbval, uint32_t waitval) { return dbval < waitval; };
  os_doorbell_wait</*isLocal=*/false, /*isDMAPossiblyActive=*/false>(
      db, val, cmpLt, doTimeoutCheck, threadId);
}

inline void os_udma_wait_done(const DMADescriptor *desc, int threadId) {
  auto cmpDoneBit = [](uint32_t doneWord, uint32_t /*waitVal*/) {
    return doneWord & (1 << 31);
  };
  os_doorbell_wait</*isLocal=*/true, /*isDMAPossiblyActive=*/true>(
      &((uint32_t *)desc)[1], 0, cmpDoneBit, /*doTimeoutCheck=*/true, threadId);
}

// NOTE: We cannot safely use the single-thread release since it is specified in
// terms of HW thread, not SW thread, so is not safe to use if there is any
// OS thread scheduling/context switching going on.
inline void os_release_allthreads(void *addr) {
  asm("release(%0):at" : : "r"(addr) : "memory");
}

inline uint32_t os_load_acquire(uint32_t *addr) {
  return hexagon_atomic_load_nolock4b_acquire(addr);
}

// UserDMA
inline void os_udma_link(DMADescriptor *tailDmaDescriptor, DMADescriptor *dmaDescriptor,
                         bool doRelease) {
  // We need a release here to ensure that prior HVX/HMX operations have
  // completed in case that's what we're DMAing.  We could potentially avoid
  // this release in certain cases, e.g. if we are doing an in-bound DMA of
  // constants.
  if (doRelease)
    os_release_allthreads(dmaDescriptor);
  asm("dmlink(%0,%1)"
      :
      : "r"(tailDmaDescriptor), "r"(dmaDescriptor)
      : "memory");
}

inline uint32_t os_udma_poll() {
  uint32_t dm0;

  asm("%0 = dmpoll" : "=r"(dm0) : : "memory");
  if (isUserDMAError(dm0))
    os_udma_report_error(dm0);

  return dm0;
}

// os_udma_poll_nocheck does not do any UDMA error checking.
// Primarily intended for use by os_thread_nanosleep.
// Others should probably be using os_udma_poll instead.
inline uint32_t os_udma_poll_nocheck() {
  uint32_t dm0;
  asm("%0 = dmpoll" : "=r"(dm0) : : "memory");
  return dm0;
}

inline uint32_t os_udma_wait() {
  uint32_t dm0;
  asm("%0 = dmwait" : "=r"(dm0) : : "memory");

  if (isUserDMAError(dm0))
    os_udma_report_error(dm0);

  return dm0;
}
inline void os_global_memsync() {
  uint32_t sync;
  os_release_allthreads(&sync);
  os_load_acquire(&sync);
  asm("syncht" : : : "memory");
}

inline void os_doorbell_local_write4b(nsp_doorbell_t db, uint32_t val) {
  hexagon_atomic_store_nolock4b((uint32_t *)db, val);
}

inline uint8_t os_doorbell_read1b(nsp_doorbell_t db) {
  return os_doorbell_read<uint8_t>(db);
}
inline uint32_t os_doorbell_read4b_acquire(nsp_doorbell_t db) {
  return hexagon_atomic_load_nolock4b_acquire((uint32_t *)db);
}

inline void os_doorbell_local_wait_eq(nsp_doorbell_t db, uint32_t val,
                                      bool doTimeoutCheck, int threadId) {
  auto cmpEq = [](uint32_t dbval, uint32_t waitval) {
    return dbval == waitval;
  };
  os_doorbell_wait</*isLocal=*/true, /*isDMAPossiblyActive=*/false>(
      db, val, cmpEq, doTimeoutCheck, threadId);
}

inline void os_doorbell_local_wait_ge(nsp_doorbell_t db, uint32_t val,
                                      bool doTimeoutCheck, int threadId) {
  auto cmpGe = [](uint32_t dbval, uint32_t waitval) {
    return dbval >= waitval;
  };
  os_doorbell_wait</*isLocal=*/true, /*isDMAPossiblyActive=*/false>(
      db, val, cmpGe, doTimeoutCheck, threadId);
}

inline uint64_t os_get_system_timestamp() {
  uint64_t ts;
  asm volatile("%0=UTIMER" : "=r"(ts));
  return ts;
}

inline int32_t utimer_freq_ms() { return UTimerFreqMS; }

// Use this when getting ucycle and pcycle at the same time
// this reduce the chance of sampling error
inline void os_get_ucycles_pcycles(uint64_t *ucycle, uint64_t *pcycle) {
  asm volatile("%1=c15:14\n"
               "%0=UTIMER\n"
               : "=r"(*ucycle), "=r"(*pcycle));
  return;
}

inline void os_mc_write_atomic4b_impl(uint32_t *mcAddress, uint32_t val) {
  hexagon_atomic_store_nolock4b(mcAddress, val);
}

inline void os_mc_write_atomic4b(void *mcBaseAddress, int32_t offset,
                                 uint32_t val) {
  uint8_t *mcAddress = (uint8_t *)mcBaseAddress;
  os_mc_write_atomic4b_impl((uint32_t *)(mcAddress + offset), val);
}

} // extern "C"

#endif // OS_INLINES_H
