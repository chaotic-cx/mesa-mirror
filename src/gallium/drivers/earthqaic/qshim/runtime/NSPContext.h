// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef _QAIC_NSPCONTEXT_H_
#define _QAIC_NSPCONTEXT_H_

#include "AICMetadataExecCtx.h"
#include "Exit.h"
#include "libdev/os.h"

using namespace aic;

struct CoreInfo {
  // NOTE: This constructor must be constexpr to ensure no
  // runtime initializers are needed to initialize the global CoreCtxData since
  // these do not get run on the real root-image execution flow.
  constexpr CoreInfo() {}

  // ExecCtx
  uint8_t virtualNSPId{0};
  uint8_t *baseL2TCM{nullptr};
  uint8_t *baseVTCM{nullptr};
  union {
    uint8_t *baseConstantDataMem{nullptr};
    uint8_t *baseStaticConstantWeightVars;
  };
  uint8_t *baseSharedDDR{nullptr};
  union {
    uint8_t *baseL2CachedDDR{nullptr};
    uint8_t *baseL2PinnedLineForPause;
  };
  // Semaphore base address.
  const SemaphoreInfo *semInfo{nullptr};
  union {
    void **mcAddresses{nullptr};
    void **mcBaseAddresses;
  };
  uint64_t startTimeStamp{0};
  nnc_log_fp logFuncPtr{nullptr};
  nnc_exit_fp exitThread{nullptr};
  nnc_pmu_set setPMUReg{nullptr};
  nnc_err_fatal_fp errFuncPtr{nullptr};
  nnc_notify_hang_fp notifyHangPtr{nullptr};
  nnc_udma_read_fp udmaReadFuncPtr{nullptr};
  nnc_mmap_fp mmapFuncPtr{nullptr};
  nnc_munmap_fp munmapFuncPtr{nullptr};
  nnc_notify_hang_fp opTimeoutPtr{nullptr};
  union {
    uint8_t *qdss_stm_port_vaddr{nullptr};
    uint8_t *qdssSTMPortVaddr;
  };

  // DMA
  DMADescriptor *getNextFreeDMADesc(int threadId);
  void fillOutDMADesc(DMADescriptor *desc, DMADescriptor *next, int8_t *dst,
                      int32_t dstOffset, const int8_t *src, unsigned size,
                      bool destBypass, bool srcBypass, bool order,
                      bool isMulticast, int threadId);
  void fillOutDMADescLocalDBUpdate(DMADescriptor *desc, DMADescriptor *next,
                                   const uint32_t *dbVal, bool order,
                                   int threadId, uint32_t DBNum);
  void fillOutDMADescGlobalDBUpdate(DMADescriptor *desc, DMADescriptor *next,
                                    const uint32_t *dbVal, bool order,
                                    int threadId, uint32_t DBNum,
				    uint32_t mcId);
  void submitDMAs(DMADescriptor *desc, DMADescriptor *tail, bool doRelease, int threadId);
  void resetDMATail(int threadId);
  DMADescriptor *dmaTail[MAX_NUM_THREADS]{nullptr};
  DMADescriptor *dmaDescStart[MAX_NUM_THREADS]{nullptr};
  DMADescriptor *dmaDescEnd[MAX_NUM_THREADS]{nullptr};
  DMADescriptor *dmaDescNext[MAX_NUM_THREADS]{nullptr};

  // These need to be on separate cache lines from each other so they aren't
  // treated as the same acquire/release location.
  __attribute__((aligned(CACHE_LINE_SIZE)))
  uint32_t inputSemaphoreReleaseLoc{0};
  __attribute__((aligned(CACHE_LINE_SIZE)))
  uint32_t outputSemaphoreReleaseLoc{0};

public:
  nsp_doorbell_t exitDB();
  int waitTimeoutLogMS{1000};
};

void _nspContextInit(AICExecContext *ctx);
void _udmaContextInit(int threadId);
void _udmaContextCleanup(int threadId);
CoreInfo *getNSPContext();
#endif
