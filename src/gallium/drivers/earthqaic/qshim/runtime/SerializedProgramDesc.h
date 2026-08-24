// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef _QAIC_SERIALIZEDPROGRAMDESC_H_
#define _QAIC_SERIALIZEDPROGRAMDESC_H_

#include "AICMetadataExecCtx.h"
#include "BufferDesc.h"

namespace qaic {

const uint16_t SERIALIZED_PROGRAMDESC_VERSION = 1;

typedef struct {
  uint16_t serialVersion;
  uint16_t exitDB;
  uint32_t size;
  uint32_t numThreads;
  uint16_t numBuffs;
  uint16_t numInputBuffs;
  uint16_t numOutputBuffs;
  uint16_t numInternalBuffs;
  uint16_t inputSem;
  uint16_t outputSem;
  uint16_t hasInputsMask;
  uint16_t hasOutputsMask;
  uint32_t buffersOffset;
  uint32_t udmaDescBuffNum;
  uint32_t udmaDummyStartDescOffset;
} SerializedProgramDesc_t;

#ifndef _QAIC_SERIALIZEDPROGRAMDESC_CPP_
extern const SerializedProgramDesc_t *_progDesc;
extern const BufferDesc_t *_progBuffers;
#endif

void _programDescInit(AICExecContext *ctx);
} // namespace qaic
#endif
