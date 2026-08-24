/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef EXECCONTEXTGENERATED_32BITPOINTERS_H
#define EXECCONTEXTGENERATED_32BITPOINTERS_H

typedef struct AICExecContext_32bitPointers {

  /// ExecCtx Major and Minor Version for version compliance check
  uint16_t execContextMajorVersion;
  uint16_t execContextMinorVersion;

  uint8_t virtualNSPId;

  /// L2TCM base virtual address
  uint32_t baseL2TCM;
  ///< DirectApi: -aic-l2tcm-size DDR alloc\n
  ///<  r/w cacheable align=4kB CCCC=0x7\n
  ///<  AICMetadata.L2TCMSize\n
  ///<  Persistent across activations. Current recommendation is 512kB though
  ///<  this may change.

  /// VTCM base virtual address
  uint32_t baseVTCM;
  ///< r/w uncached align=2048 CCCC=0x6\n
  ///< AICMetadata.VTCMSize\n
  ///< DirectApi: All networks MUST map to the start of VTCM.
  ///<  Any carve-out for CV MUST be at the top end of VTCM space.
  ///<  This requirement derives from HMX HW requirements related to TLB entry
  ///<  crossings.

  /// Constant data base virtual address
  uint32_t baseConstantDataMem;
  ///< Base address for statically mapped constant weights memory block.
  ///< Includes constants and static placeholders.\n
  ///< r/o uncached align=2048 CCCC=0x6

  /// Shared DDR data base virtual address
  uint32_t baseSharedDDR;
  ///< Mutable weights memory block, Inputs and Outputs as well
  ///< as intermediate activations that don't end up in VTCM.\n
  ///< r/w uncached align=2048 CCCC=0x6

  /// L2 cacheable only base virtual address
  uint32_t baseL2CachedDDR;
  ///< cccc=0xf (L1 uncached, L2 cached WB). size=4kB\n
  ///< The first 64B cache line will be locked in the L2 to guarantee latency.

  /// Pointer to array of virtual addresses to use for multicasts.
  uint32_t mcAddresses;
  ///< Indexed by virtual multicast id.\n
  ///< DirectApi: nullptr

  uint64_t startTimeStamp;
  ///< DirectApi: startTimeStamp=0\n
  ///< NOTE: Firmware needs to sync UTIMER with system TOD

  /// Log function pointer. Should be set appropriately for each path.
  uint32_t logFuncPtr;
  /// Function used to exit thread execution
  uint32_t exitThread;
  ///< See 'exitDoorbellOffset description.\n
  ///< DirectApi: nullptr

  /// Function for setting PMU specific registers
  uint32_t setPMUReg;
  /// Error handling function
  uint32_t errFuncPtr;
  /// Report network hang
  uint32_t notifyHangPtr;
  /// UDMA register reader function
  uint32_t udmaReadFuncPtr;
  /// Memory Map functions
  uint32_t mmapFuncPtr;
  ///< DirectApi: nullptr
  uint32_t munmapFuncPtr;
  ///< DirectApi: nullptr

  /// Network QDSS STM Port virtual address
  uint32_t qdssSTMPortVaddr;

  /// Function to get all PMU count registers
  uint32_t readPMUCnt;

  /// DDR Bandwidth Monitor virtual address
  uint32_t ddrBWMonRegVaddr;

  /// Pointer to array[32] of virtual semaphore information
  uint32_t semaphoreListPtr;

  /// Network heap
  uint32_t networkHeapAddr;
  uint64_t networkHeapSize;

  /// Function to support MCID reprogramming
  uint32_t reprogMcidFuncPtr;

  /// RTLD APIs related to dynamic loading
  uint32_t dlOpenPtr;
  uint32_t dlOpenbufPtr;
  uint32_t dlClosePtr;
  uint32_t dlSymPtr;
  uint32_t dlAddrPtr;
  uint32_t dlErrorPtr;
  uint32_t dlInfoPtr;

  /// UTC offset DDR data (input) base virtual address
  uint32_t baseUtcOffsetDDR;

  /// Function network calls to update predicates
  uint32_t predicateUpdateFuncPtr;

  /// Pointer to the buffer of predicate update
  uint32_t predicateBufferPtr;

  /// Predicate update buffer size
  uint32_t predicateBufferSize;

  // Pointer to Userkey buffer
  uint32_t userkeyPtr;

  // Userkey size
  uint64_t userkeySize;

  // Secure Static shared DDR region
  uint32_t baseSecureStaticSharedDDR;

  // Function to get random numbers
  uint32_t getRandNumFuncPtr;

} AICExecContext32bitPointers;

#endif // EXECCONTEXTGENERATED_32BITPOINTERS_H