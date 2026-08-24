/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef JIT_QSHIM_API_H
#define JIT_QSHIM_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t QShimUDmaHandle;

#define LOG64_KERNEL(x) (uint32_t) x, (uint32_t)((uint64_t)x >> 32)
#define INVALID_UDMA_HANDLE 0xFFFFFFFF

/* Various log masks*/
typedef enum {
  LOG_MASK_NONE = 0x0,
  LOG_MASK_FATAL = 0x1,
  LOG_MASK_ERROR = 0x2,
  LOG_MASK_WARN = 0x4,
  LOG_MASK_INFO = 0x8,
  LOG_MASK_DEBUG = 0x10,
  LOG_MASK_INVALID = 0x7f
} QshimLogMask_t;

// Caller shall populate the struct for 1D descriptor API, if they choose
// to specify the enclosed fields. Else, the user need not to pass this struct
// as an argument in qshimLinearUdmaSubmit() API. Be default, API specifies the
// struct parameter as nullptr in 1D Udma submit API declaration. A pointer
// to this struct is embedded in 2D uDma descriptor struct as well and can
// be left as nullptr if user doesn't wish to specify enclosed fields.
typedef struct {
  uint32_t order : 1;              // Descriptor ordering point.
                                   // * 0 – DMA associated with this descriptor has no ordering
                                   // requirements with previous DMAs which are not reported "Done"
                                   // * 1 – Previous DMA in this descriptor chain must complete,
                                   // prior to starting this descriptor. NOTE: DMAs are not
                                   // guaranteed to execute in order. This bit must be set when
                                   // ordering between DMA descriptors is required.
                                   // By default, the bit is set to 0.

  uint32_t reserved : 31;
} AicJitUdmaDescCommonAttrs;

// Caller must initialize the struct to 0 and then populate the attributes for
// 2D descriptor API. User shall populate udmaDescCommonAttrs if they need to
// specify flags field. Else, the pointer can be set to nullptr.
typedef struct {
  uint32_t descType : 8;           // Descriptor Type - Only Type0 is supported for now
                                   // This field isn't required to be populated.

  uint32_t reserved1 : 24;

  uint32_t height : 16;            // Number of rows in the region.
                                   // Height cannot exceed (1 << 16) - 1.

  uint32_t width : 16;             // Width of the region (number of bytes in the row)
                                   // Width cannot exceed (1 << 16) - 1.

  uint32_t destStride : 16;        // Number of bytes between successive rows in the
                                   // destination. This field must be greater than or equal
                                   // to the width. A value of 0 is interpreted as follows:
                                   // * Version 1 – 0
                                   // * Version 2 – 0x1000
                                   // Dest Stride cannot exceed (1 << 16) - 1.

  uint32_t srcStride : 16;         // Number of bytes between successive rows in the source.
                                   // This field must be greater than or equal to the width.
                                   // A value of 0 is interpreted as follows:
                                   // * Version 1 – 0
                                   // * Version 2 – 0x1000
                                   // Dest Stride cannot exceed (1 << 16) - 1.

  uint32_t reserved2;

  AicJitUdmaDescCommonAttrs
      *udmaDescCommonAttrs;        // Common attributes between 1D & 2D descriptor
} AicJit2DUdmaDescAttrs;

// Caller should initialize the numThreads field to the number of threads
// to sync before the first call to qshimSyncThreads API and shall not touch
// the field till the synchronization is complete. At that point, the value
// of numThreads shall be the same as original value. User can reuse the struct
// with the same or different numThreads value after that. User shall not
// change any other fields of the struct at any time.
typedef struct AicJitThreadSyncStruct {
  int32_t numThreads;
  int32_t data;
  int32_t reserved;
} AicJitThreadSyncStruct_t;

// The syncPoint should come from global context, say a global variable defined
// in the operator shared library. It must not come from a thread's stack. The
// value of numThreads field in syncPoint struct must be initialized to the
// number of threads to sync before the first call to qshimSyncThreads.
// The callers block till there are enough thread callers. At that point all
// callers are unblocked. The numThreads field shall be the same as original
// value and the struct can be reused with same or differnt value of numThreads.
// This functions returns error (non-zero) if the syncPoint pointer is NULL or
// numThreads field is 0.
uint32_t qshimSyncThreads(uint32_t threadId, AicJitThreadSyncStruct *syncPoint);

// Qshim API for Flexible Profiling Operation level. This function writes a
// Start time stamp datapoint on the chrome trace. Please be aware that the
// delay of qshimProfilingEventStart to write the data to DDR is 2us, and it
// marks the beginning of a chrome trace event.
void qshimProfilingEntryStart (uint32_t threadId, const char *eventName);

// Qshim API for Flexible Profiling Operation level. This function writes a
// End time stamp datapoint on the chrome trace with the same event name used in
// qshimProfilingEntryStart. This funciton must be called after
// shimProfilingEntryStart has been called with the same event name otherwise
// the chrome trace time stamp can not display correctly. Please be aware that
// the delay of qshimProfilingEventStart to write the data to DDR is 2us, and it
// marks the ending of a chrome trace event.
void qshimProfilingEntryEnd (uint32_t threadId, const char *eventName);

// QShim API for logging
void qshimLog(uint32_t mask, uint32_t dataCount, const char *formatStr, ...);

// This API returns a valid handle, only if requireHandle flag is passed as
// true, which can be waited on for the completion of transfer in qshimUDmaWait().
// Size is the number of bytes to move.
// If a valid pointer to Udma descriptor struct is not passed, default values
// will be used for all attributes in AicJitUdmaDescCommonAttrs struct.
// The return value is INVALID_UDMA_HANDLE if the submission is not successful.
// status pointer returns success(zero) for a  successful submission. In case
// of submission failure, status indicates failure scenario as per JitDevStatusCode_t enum.
QShimUDmaHandle
qshimLinearUdmaSubmit(uint32_t threadID, void *src, uint32_t size, void *dst,
                      AicJitUdmaDescCommonAttrs *udmaDescAttrs, bool requireHandle,
                      uint32_t *status);

// This API returns a valid handle, only if requireHandle flag is passed as
// true, which can be waited on for the completion of transfer in qshimUDmaWait().
// Udma descriptor struct needs to be populated as per the rules mentioned
// in AicJit2DUdmaDescAttrs struct definition.
// The return value is INVALID_UDMA_HANDLE if the submission is not successful.
// status pointer returns success(zero) for a successful submission. In case
// of submission failure, status indicates failure scenario  as per JitDevStatusCode_t enum.
QShimUDmaHandle qshim2DUdmaSubmit(uint32_t threadID, void *src, void *dst,
                                  AicJit2DUdmaDescAttrs *udmaDescAttrs, bool requireHandle,
                                  uint32_t *status);

// Wait for the handle returned in qshimUDmaSubmit().
// Return 0 for success, non-zero for error.
// Non-zero returned if INVALID_UDMA_HANDLE is provided.
uint32_t qshimUDmaWait(QShimUDmaHandle handle);

// Returns base address of VTCM
uint8_t *qshimGetBaseVtcmAddr(void);

// This API is used to queue a random number generation request, which can be waited
// upon using the threadID of the thread submitting the request.
// numElements is the number of random numbers expected to be generated, each of size 4B.
// If a valid pointer is not passed or numElements is 0 or invalid threadId is provided,
// JIT_DEV_QSHIM_ERROR_INVALID_PARAMETER error code is returned.
// If the same thread submits a new request, before the completion of the previous request,
// JIT_DEV_QSHIM_ERROR_PRNG_MULTIPLE_SUBMISSION error code is returned.
uint32_t qshimGetRandomDataSubmit(uint32_t threadId, uint32_t *outPtr, uint32_t numElements);

// Wait for the submitted Random Data generation request
// Returns 0 for success.
// Returns JIT_DEV_QSHIM_ERROR_INVALID_PARAMETER when invalid threadID is provided.
uint32_t qshimGetRandomDataWait(uint32_t threadId);



// Qshim Mmap API to be used by operators.
// bufHandle - Buffer handle received through operator API which is
//             passed as an input in the list of args to entry point
//             function.
// offset    - Offset into the buffer to start mapping at
// size      - Size to map
// flag      - To specify mapping attributes such as cacheable etc
// status    - status pointer returns success(zero) for a successful
//             Mmap. In case of failure specifies error code as per
//             JitDevStatusCode_t.
// Return value is the virtual address. Will be NULL if the mmap call fails.
void* qshimMmap(void* bufHandle, uint64_t offset, uint32_t size, uint32_t flag, uint32_t *status);

// Qshim Unmap API to be used by operators.
// vaAddr - Takes in virtual address, returned by mmap API, as input
//          and unmaps the buffer.
void qshimUnmap(void* vaAddr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JIT_QSHIM_API_H
