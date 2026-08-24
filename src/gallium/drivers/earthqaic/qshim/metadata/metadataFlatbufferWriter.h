// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef METADATAFLATBUFFERWRITER
#define METADATAFLATBUFFERWRITER

#include "AICMetadata.h"
#include "AicMetadataFlat_generated.h"
#include "execContextWriter.hpp"
#include <array>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

typedef void (*nnc_activate_fp)(void *ctx, uint8_t virtualThreadId,
                                uint32_t stid);
typedef AICMDSemaphoreOp SemaphoreOp;
typedef std::vector<SemaphoreOp> SemaphoreOps;
typedef AICMDDoorbellOp DoorbellOp;
typedef std::vector<DoorbellOp> DoorbellOps;

// END TEMPORARY DECLARATIONS.

using SemaphoreOpF = AicMetadataFlat::AICMDSemaphoreOpT;
using SemaphoreOpsF = std::vector<SemaphoreOpF>;
using DoorbellOpF = AicMetadataFlat::AICMDDoorbellOpT;
using DoorbellOpsF = std::vector<DoorbellOpF>;

class MetadataFlatbufferWriter {
public:
  enum class DynamicSharedDDRStatus { Disabled = 0, Enabled = 1 };

private:
  AicMetadataFlat::MetadataT metadata_;
  std::vector<uint8_t> metadataBuffer;
  DynamicSharedDDRStatus dynamicDDRStatus_{DynamicSharedDDRStatus::Disabled};
  // Constant mappings.
  uint32_t constMappingCores{0};
  static bool searchKnownFields(const std::string &requiredField);
  void addIntrospectionString(const std::string &element);
  void serialize();

public:
  MetadataFlatbufferWriter(
      uint16_t hwVersionMajor, uint16_t hwVersionMinor,
      uint16_t versionMajor = AIC_METADATA_MAJOR_VERSION,
      uint16_t versionMinor = AIC_METADATA_MINOR_VERSION,
      uint16_t execContextMajorVersion = AIC_METADATA_EXEC_CTX_MAJOR_VERSION,
      uint64_t exitDoorbellOffset = ~(uint64_t)0);
  void finalize();
  bool isFinal() const { return (metadataBuffer.size() > 0); }
  int getSize() const {
    assert(isFinal());
    return metadataBuffer.size();
  }
  // Returns true if write succeeded, false otherwise.
  bool writeMetadata(const char *outfile) const;
  const std::vector<uint8_t> &getMetadata() const {
    assert(isFinal());
    return metadataBuffer;
  }

  // Number of NSPs.
  int getNumNSPs() const { return metadata_.numNSPs; }
  void setNumNSPs(uint16_t numNSPs) {
    metadata_.numNSPs = numNSPs;
    while (metadata_.nspMulticastTables.size() < metadata_.numNSPs) {
      metadata_.nspMulticastTables.emplace_back(
          new AicMetadataFlat::AICMDNSPMulticastEntryTableT());
    }
  }

  // VTCM size in bytes.
  int getVTCMSize() const { return metadata_.VTCMSize; }
  void setVTCMSize(uint32_t vtcmSize) { metadata_.VTCMSize = vtcmSize; }

  // L2TCM size in bytes.
  int getL2TCMSize() const { return metadata_.L2TCMSize; }
  void setL2TCMSize(uint32_t l2TCMSize) { metadata_.L2TCMSize = l2TCMSize; }

  // Network name.
  std::string getNetworkName() const { return metadata_.networkName; }
  void setNetworkName(const std::string name) { metadata_.networkName = name; }

  // Number of virtual semaphores.
  uint16_t getNumSemaphores() const {
    return metadata_.semaphoreInitState.size();
  }
  void setNumSemaphores(uint16_t numSemaphores) {
    metadata_.semaphoreInitState.resize(numSemaphores);
    metadata_.numSemaphores = numSemaphores;
  }
  void initSemaphore(uint32_t semId, uint32_t val) {
    assert(semId < metadata_.semaphoreInitState.size());
    metadata_.semaphoreInitState[semId] = val;
  }
  // L2TCM init region size in bytes.
  int getL2TCMInitSize() const { return metadata_.L2TCMInitState.size(); }
  void initL2TCMResize(uint32_t newSize) {
    metadata_.L2TCMInitState.resize(newSize);
    metadata_.L2TCMInitSize = newSize;
  }
  void initL2TCMByte(uint64_t offset, uint8_t byte) {
    assert(offset < metadata_.L2TCMInitState.size());
    metadata_.L2TCMInitState[offset] = byte;
  }
  void initL2TCMWord(uint64_t offset, uint32_t word) {
    for (int i = 0; i < 4; i++, word >>= 8) {
      uint8_t b = word & 0xff;
      metadata_.L2TCMInitState[offset++] = b;
    }
  }

  // Shared DDR
  uint64_t getStaticSharedDDRSize() const {
    return metadata_.staticSharedDDRSize;
  }
  void setStaticSharedDDRSize(uint64_t size) {
    metadata_.staticSharedDDRSize = size;
  }

  uint64_t getDynamicSharedDDRSize() const {
    return metadata_.dynamicSharedDDRSize;
  }
  void setDynamicSharedDDRSize(uint64_t size) {
    metadata_.dynamicSharedDDRSize = size;
  }

  bool getStaticSharedDDRECCEnabled() const {
    return metadata_.staticSharedDDRECCEnabled;
  }
  void setStaticSharedDDRECCEnabled(bool v) {
    metadata_.staticSharedDDRECCEnabled = v;
  }

  bool getDynamicSharedDDRECCEnabled() const {
    return metadata_.dynamicSharedDDRECCEnabled;
  }
  void setDynamicSharedDDRECCEnabled(bool v) {
    metadata_.dynamicSharedDDRECCEnabled = v;
  }

  // Constants
  uint64_t getStaticConstantsSize() const {
    return metadata_.staticConstantsSize;
  }
  void setStaticConstantsSize(uint64_t size) {
    metadata_.staticConstantsSize = size;
  }

  uint64_t getDynamicConstantsSize() const {
    return metadata_.dynamicConstantsSize;
  }
  void setDynamicConstantsSize(uint64_t size) {
    metadata_.dynamicConstantsSize = size;
  }

  bool getStaticConstantsECCEnabled() const {
    return metadata_.staticConstantsECCEnabled;
  }
  void setStaticConstantsECCEnabled(bool v) {
    metadata_.staticConstantsECCEnabled = v;
  }

  bool getDynamicConstantsECCEnabled() const {
    return metadata_.dynamicConstantsECCEnabled;
  }
  void setDynamicConstantsECCEnabled(bool v) {
    metadata_.dynamicConstantsECCEnabled = v;
  }

  bool getSingleVTCMPage() const { return metadata_.singleVTCMPage; }
  void setSingleVTCMPage(bool v) { metadata_.singleVTCMPage = v; }

  // BEGIN TEMPORARY LEGACY SUPPORT
  // Host<->NSP DMA requests.
  void addSemaphoreOp(SemaphoreOps &semaphoreOps, uint8_t semOp,
                      uint16_t semNum, uint16_t semValue, uint8_t preOrPost,
                      uint8_t inSyncFence, uint8_t outSyncFence);
  void addDoorbellOp(DoorbellOps &doorbellsOps, uint8_t sizeEnum, uint16_t mcId,
                     uint64_t offset, uint32_t data);
  void addDMARequest(uint16_t num, uint64_t hostOffset, uint8_t devAddrSpace,
                     uint64_t devOffset, uint32_t size, uint8_t inOut,
                     uint16_t portID, uint16_t mcId, SemaphoreOps &semaphoreOps,
                     DoorbellOps &doorbellOps, uint32_t transactionId = 0);
  // END TEMPORARY LEGACY SUPPORT
  // Host<->NSP DMA requests.
  void addSemaphoreOp(SemaphoreOpsF &semaphoreOps, uint8_t semOp,
                      uint16_t semNum, uint16_t semValue, uint8_t preOrPost,
                      uint8_t inSyncFence, uint8_t outSyncFence);
  void addDoorbellOp(DoorbellOpsF &doorbellsOps, uint8_t sizeEnum,
                     uint16_t mcId, uint64_t offset, uint32_t data);
  void addDMARequest(uint16_t num, uint64_t hostOffset, uint8_t devAddrSpace,
                     uint64_t devOffset, uint32_t size, uint8_t inOut,
                     uint16_t portID, uint16_t mcId,
                     SemaphoreOpsF &semaphoreOps, DoorbellOpsF &doorbellOps,
                     uint32_t transactionId);
  // Add NSP/host multicast entries.
  void addNSPMulticastEntry(int core, uint8_t dynamic, uint32_t mask,
                            uint32_t size, uint8_t addrSpace,
                            uint64_t baseAddrOffset);
  void addHostMulticastEntry(uint32_t mask, uint32_t size);

  // Add thread descriptor.
  void addThreadDescriptor(nnc_activate_fp entryPoint, bool hasHMX,
                           bool hasHVX);
  void addThreadDescriptor(nnc_activate_fp entryPoint, uint8_t typeMask);
  // Add constant mapping.
  void addConstantMapping(uint32_t mask, uint64_t offset, uint32_t size);

  void setExitDoorbellOffset(uint64_t offset) {
    metadata_.exitDoorbellOffset = offset;
  }

  void addPort(uint16_t portId, AicMetadataFlat::AICMDPortType type);

  // Network heap
  uint64_t getNetworkHeapSize() const { return metadata_.networkHeapSize; }
  void setNetworkHeapSize(uint64_t size) { metadata_.networkHeapSize = size; }

  // set FP presence
  void setHasHvxFP(bool hasHvx) { metadata_.hasHvxFP = hasHvx ? 1 : 0; }
  void setHasHmxFP(bool hasHmx) { metadata_.hasHmxFP = hasHmx ? 1 : 0; }

  // add a field to requiredFields
  void addRequiredField(const std::string &field) {
    metadata_.requiredFields.push_back(field);
  }

  void
  setCacheableConstants(const AicMetadataFlat::cacheableConstants setting) {
    metadata_.QNNConfig->Constants = setting;
    if (setting !=
        AicMetadataFlat::cacheableConstants_CACHE_DISABLED) { // DISABLED is
                                                              // default behavior
      this->addIntrospectionString(
          "AicMetadataFlat_Metadata->QNNConfig->Constants");
    }
  }

  void enableDynamicSharedDDR() {
    if (dynamicDDRStatus_ == DynamicSharedDDRStatus::Disabled) {
      dynamicDDRStatus_ = DynamicSharedDDRStatus::Enabled;
      this->addIntrospectionString(
          "AicMetadataFlat_Metadata->dynamicSharedDDRSupported");
    }
  }
  void setNetworkHeapBehavior(
      const AicMetadataFlat::networkDeactivateAction setting) {
    metadata_.networkHeapBehavior->onNetworkDeactivate = setting;
    if (setting !=
        AicMetadataFlat::networkDeactivateAction_freeNetworkHeap) { // free is
                                                                    // default
                                                                    // behavior
      this->addIntrospectionString(
          "AicMetadataFlat_Metadata->networkHeapBehavior");
    }
  }

  // record the raw bytes taken up by struct version for firmware's use
  void set_raw_struct_version_length(const uint64_t length) {
    metadata_.raw_struct_version_length = length;
  }
  // input of a vector, set execContext to that vector
  void setExecContext(const std::vector<uint8_t> &execContext) {
    metadata_.execContext = execContext;
    this->addIntrospectionString("AicMetadataFlat_Metadata->execContext");
  }
  void PopulateL2TCMInitStateNonZeroRegions();
};

#endif // METADATAFLATBUFFERWRITER
