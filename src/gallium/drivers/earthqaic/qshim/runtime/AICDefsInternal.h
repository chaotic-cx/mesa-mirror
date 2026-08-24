// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef AICDEFSINTERNAL_H
#define AICDEFSINTERNAL_H

#include "AICMetadata.h"
#include <stdint.h>

//remove namespace and other C++ specifics so that C files can use some of the constants here.
#ifdef __cplusplus
namespace aic {
#endif

static const uint64_t KB = 1 << 10;
static const uint64_t MB = 1 << 20;
static const uint64_t GB = 1 << 30;

const int MAX_NUM_CORES = 16;
const int MAX_NUM_THREADS = 6;

const int UDMAMaxSize = ((1 << 24) - 1);

const int DB_SIZE = 4;

#define CACHE_LINE_SIZE 128
#define NUM_UDMA_CACHELINES_PER_THREAD 2

const int UTimerFreqMS = 19200;

const unsigned DMA_state_mask = 0x3;
const unsigned DMA_state_idle = 0;
const unsigned DMA_state_run = 1;
const unsigned DMA_state_error = 2;

#ifdef __cplusplus
static inline bool isUserDMAError(uint32_t dm0) {
  return (dm0 & DMA_state_mask) == DMA_state_error;
}

struct __attribute__((packed, aligned(16))) DMADescriptor {
  union {
    struct {
      uint64_t dword0;
      uint64_t dword1;
    };
#define DMA_DESC_DWORD0(NEXT, WORD1)                                           \
  ((uint32_t)(NEXT) | ((uint64_t)(WORD1) << 32))
#define DMA_DESC_DWORD1(SRC, DST) ((uint32_t)(SRC) | ((uint64_t)(DST) << 32))

    struct {
      uint32_t next;

      union {
        struct {
          unsigned int length : 24;
          unsigned reserved : 4; // must be 0
          unsigned destBypass : 1; // bypass cache hierarchy
          unsigned srcBypass : 1;
          unsigned order : 1; // enforce order relative to previous DMA commands
          unsigned done : 1;  // filled in when memory operations are complete
        };
#define DMA_LINEAR_DESC_WORD1(LEN, DSTBYPASS, SRCBYPASS, ORDER)                \
  ((LEN & 0xffffff) | ((DSTBYPASS & 1) << 28) |    \
   ((SRCBYPASS & 1) << 29) | ((ORDER & 1) << 30))
        uint32_t word1;
      };

      uint32_t src; // virtual address
      uint32_t dst; // virtual address
    };
  };
};

} // namespace aic
#endif // __cplusplus

#endif // AICDEFSINTERNAL_H
