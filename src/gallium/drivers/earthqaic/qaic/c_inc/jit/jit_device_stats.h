/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef AICLINUXRUNTIME_JIT_DEVICE_STATS_H
#define AICLINUXRUNTIME_JIT_DEVICE_STATS_H

#ifdef __cplusplus
#include <stdint.h>
extern "C" {
#else
#include <stdint.h>
#endif

// Represents BIT position in AicInit::profilingConfig::profilingMetricBitMap
enum JITProfilingMetricBitPosition {
  COMMAND_EXECUTION_TIME = 0,
  DDR_TRAFFIC = 1,
  HVX_ACTIVE_PMU = 2,
  HMX_ACTIVE_PMU = 3,
  DMA_ACTIVE_PMU = 4,
  DEV_GRANULAR_DATA = 5,
};

enum JITProfilingEventPhase {
  BEGIN_PHASE,
  END_PHASE,
};

enum JITProfilingEventType {
  EVENT_TYPE_TIME_STAMP,
  EVENT_TYPE_PMU_DDR,
};

typedef struct JITTimeStampEventData {
  uint64_t uCycleCounts;
  uint64_t deviceBootTimeUTC;
} JITTimeStampEventData_t;

typedef struct JITPMUDdrEventData {
  uint64_t pCycleCounts;
  // NOTE: There can be 8 PMUs enabled at a given time, currently we are
  // supporting only 3. Might need revisit if we want to expand the scope.
  uint32_t pmuCounter[3];
  uint32_t DmaTrafficCount;
} JITPMUDdrEventData_t;

typedef struct JITProfilingEventTrace {
  char eventName[16];
  // For events associated with a command, this UID should be command id
  // otherwise command set id.
  uint64_t eventUID;
  uint8_t threadId;
  uint8_t eventPhase;
  uint8_t eventType;
  uint8_t pad[5];
  // Will be either JITPMUDdrEventData or JITTimeStampEventData
  uint8_t eventData[];
} JITProfilingEventTrace_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AICLINUXRUNTIME_JIT_DEVICE_STATS_H
