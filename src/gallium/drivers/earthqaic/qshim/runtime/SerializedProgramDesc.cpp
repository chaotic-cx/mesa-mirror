// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#define _QAIC_SERIALIZEDPROGRAMDESC_CPP_
#include "SerializedProgramDesc.h"
#include "NSPContext.h"
#include <inttypes.h>

namespace qaic {

SerializedProgramDesc_t *_progDesc;
BufferDesc_t *_progBuffers;

void _programDescInit(AICExecContext *ctx) {
  _progDesc = (SerializedProgramDesc_t *)getNSPContext()->baseConstantDataMem;
  if (_progDesc->serialVersion != SERIALIZED_PROGRAMDESC_VERSION) {
    ERR_FATAL(ctx->errFuncPtr,
              "Expected SERIALIZED_PROGRAMDESC_VERSION %d but found %d",
              SERIALIZED_PROGRAMDESC_VERSION, _progDesc->serialVersion, 0);
    exit(-1);
  }
  _progBuffers =
      (BufferDesc_t *)((uint8_t *)_progDesc + _progDesc->buffersOffset);
}

} // namespace qaic
