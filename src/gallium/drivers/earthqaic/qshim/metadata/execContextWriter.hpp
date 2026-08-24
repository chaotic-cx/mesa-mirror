// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifndef AICMETADATA_EXECCONTEXTWRITER_HPP
#define AICMETADATA_EXECCONTEXTWRITER_HPP
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "execContextGenerated_32bitPointers.h"
#include "execContext_generated.h"

// define a class execContext Writer
class ExecContextWriter final {
public:
  enum class defaultType { DEFAULT, NO_DEFAULT };
  explicit ExecContextWriter(
      const defaultType setting = defaultType::NO_DEFAULT) {
    if (setting == defaultType::DEFAULT) {
      usingExecContextVersion();
      usingExecContextVirtualAddresses();
      usingExecContextNNCfp();
      usingExecContextRTLDapi();
    }
  }
  // setters for each section, "using_<section_name>"
  void usingExecContextVersion() {
    addUsedElement(ExecContext::execContextVariables_execContextMajorVersion,
                   sizeof(placeholder.execContextMajorVersion),
                   offsetof(AICExecContext_32bitPointers, execContextMajorVersion));
    addUsedElement(ExecContext::execContextVariables_execContextMinorVersion,
                   sizeof(placeholder.execContextMinorVersion),
                   offsetof(AICExecContext_32bitPointers, execContextMinorVersion));
    addUsedElement(ExecContext::execContextVariables_virtualNSPId,
                   sizeof(placeholder.virtualNSPId),
                   offsetof(AICExecContext_32bitPointers, virtualNSPId));
  }
  // addresses section starting with baseL2TCM and ending with startTimeStamp
  void usingExecContextVirtualAddresses() {
    addUsedElement(ExecContext::execContextVariables_baseL2TCM,
                   sizeof(placeholder.baseL2TCM),
                   offsetof(AICExecContext_32bitPointers, baseL2TCM));
    addUsedElement(ExecContext::execContextVariables_baseVTCM,
                   sizeof(placeholder.baseVTCM),
                   offsetof(AICExecContext_32bitPointers, baseVTCM));
    addUsedElement(ExecContext::execContextVariables_baseConstantDataMem,
                   sizeof(placeholder.baseConstantDataMem),
                   offsetof(AICExecContext_32bitPointers, baseConstantDataMem));
    addUsedElement(ExecContext::execContextVariables_baseSharedDDR,
                   sizeof(placeholder.baseSharedDDR),
                   offsetof(AICExecContext_32bitPointers, baseSharedDDR));
    addUsedElement(ExecContext::execContextVariables_baseL2CachedDDR,
                   sizeof(placeholder.baseL2CachedDDR),
                   offsetof(AICExecContext_32bitPointers, baseL2CachedDDR));
    addUsedElement(ExecContext::execContextVariables_mcAddresses,
                   sizeof(placeholder.mcAddresses),
                   offsetof(AICExecContext_32bitPointers, mcAddresses));
    addUsedElement(ExecContext::execContextVariables_startTimeStamp,
                   sizeof(placeholder.startTimeStamp),
                   offsetof(AICExecContext_32bitPointers, startTimeStamp));
  }
  // section starting with logFuncPtr and ending with reprogMcidFuncPtr
  void usingExecContextNNCfp() {
    addUsedElement(ExecContext::execContextVariables_logFuncPtr,
                   sizeof(placeholder.logFuncPtr),
                   offsetof(AICExecContext_32bitPointers, logFuncPtr));
    addUsedElement(ExecContext::execContextVariables_exitThread,
                   sizeof(placeholder.exitThread),
                   offsetof(AICExecContext_32bitPointers, exitThread));
    addUsedElement(ExecContext::execContextVariables_setPMUReg,
                   sizeof(placeholder.setPMUReg),
                   offsetof(AICExecContext_32bitPointers, setPMUReg));
    addUsedElement(ExecContext::execContextVariables_errFuncPtr,
                   sizeof(placeholder.errFuncPtr),
                   offsetof(AICExecContext_32bitPointers, errFuncPtr));
    addUsedElement(ExecContext::execContextVariables_notifyHangPtr,
                   sizeof(placeholder.notifyHangPtr),
                   offsetof(AICExecContext_32bitPointers, notifyHangPtr));
    addUsedElement(ExecContext::execContextVariables_udmaReadFuncPtr,
                   sizeof(placeholder.udmaReadFuncPtr),
                   offsetof(AICExecContext_32bitPointers, udmaReadFuncPtr));
    addUsedElement(ExecContext::execContextVariables_mmapFuncPtr,
                   sizeof(placeholder.mmapFuncPtr),
                   offsetof(AICExecContext_32bitPointers, mmapFuncPtr));
    addUsedElement(ExecContext::execContextVariables_munmapFuncPtr,
                   sizeof(placeholder.munmapFuncPtr),
                   offsetof(AICExecContext_32bitPointers, munmapFuncPtr));
    addUsedElement(ExecContext::execContextVariables_qdssSTMPortVaddr,
                   sizeof(placeholder.qdssSTMPortVaddr),
                   offsetof(AICExecContext_32bitPointers, qdssSTMPortVaddr));
    addUsedElement(ExecContext::execContextVariables_readPMUCnt,
                   sizeof(placeholder.readPMUCnt),
                   offsetof(AICExecContext_32bitPointers, readPMUCnt));
    addUsedElement(ExecContext::execContextVariables_ddrBWMonRegVaddr,
                   sizeof(placeholder.ddrBWMonRegVaddr),
                   offsetof(AICExecContext_32bitPointers, ddrBWMonRegVaddr));
    addUsedElement(ExecContext::execContextVariables_semaphoreListPtr,
                   sizeof(placeholder.semaphoreListPtr),
                   offsetof(AICExecContext_32bitPointers, semaphoreListPtr));
    addUsedElement(ExecContext::execContextVariables_networkHeapAddr,
                   sizeof(placeholder.networkHeapAddr),
                   offsetof(AICExecContext_32bitPointers, networkHeapAddr));
    addUsedElement(ExecContext::execContextVariables_networkHeapSize,
                   sizeof(placeholder.networkHeapSize),
                   offsetof(AICExecContext_32bitPointers, networkHeapSize));
    addUsedElement(ExecContext::execContextVariables_reprogMcidFuncPtr,
                   sizeof(placeholder.reprogMcidFuncPtr),
                   offsetof(AICExecContext_32bitPointers, reprogMcidFuncPtr));
    addUsedElement(ExecContext::execContextVariables_getRandNumFuncPtr,
                   sizeof(placeholder.getRandNumFuncPtr),
                   offsetof(AICExecContext_32bitPointers, getRandNumFuncPtr));
  }
  // section starting with dlOpenPtr and ending with dlInfoPtr
  void usingExecContextRTLDapi() {
    addUsedElement(ExecContext::execContextVariables_dlOpenPtr,
                   sizeof(placeholder.dlOpenPtr),
                   offsetof(AICExecContext_32bitPointers, dlOpenPtr));
    addUsedElement(ExecContext::execContextVariables_dlOpenbufPtr,
                   sizeof(placeholder.dlOpenbufPtr),
                   offsetof(AICExecContext_32bitPointers, dlOpenbufPtr));
    addUsedElement(ExecContext::execContextVariables_dlClosePtr,
                   sizeof(placeholder.dlClosePtr),
                   offsetof(AICExecContext_32bitPointers, dlClosePtr));
    addUsedElement(ExecContext::execContextVariables_dlSymPtr,
                   sizeof(placeholder.dlSymPtr),
                   offsetof(AICExecContext_32bitPointers, dlSymPtr));
    addUsedElement(ExecContext::execContextVariables_dlAddrPtr,
                   sizeof(placeholder.dlAddrPtr),
                   offsetof(AICExecContext_32bitPointers, dlAddrPtr));
    addUsedElement(ExecContext::execContextVariables_dlErrorPtr,
                   sizeof(placeholder.dlErrorPtr),
                   offsetof(AICExecContext_32bitPointers, dlErrorPtr));
    addUsedElement(ExecContext::execContextVariables_dlInfoPtr,
                   sizeof(placeholder.dlInfoPtr),
                   offsetof(AICExecContext_32bitPointers, dlInfoPtr));
    addUsedElement(ExecContext::execContextVariables_baseUtcOffsetDDR,
                   sizeof(placeholder.baseUtcOffsetDDR),
                   offsetof(AICExecContext_32bitPointers, baseUtcOffsetDDR));
  }
  // a serialize function that will serialize the whole thing

  const std::vector<uint8_t> &getExecContext() {
    finalize();
    return ExecContextBytes;
  }

private:
  void finalize() {
    execContext_.execContextSize = maxsize;
    flatbuffers::FlatBufferBuilder builder;
    builder.ForceDefaults(true);
    builder.Finish(ExecContext::execContext::Pack(builder, &execContext_));
    std::vector<uint8_t> outvalue(builder.GetBufferPointer(),
                                  builder.GetBufferPointer() +
                                      builder.GetSize());
    ExecContextBytes.swap(outvalue);
  }
  void addUsedElement(const ExecContext::execContextVariables variableUsed,
                      const uint32_t sizeOfElement,
                      const uint32_t offsetOfElement,
                      const ExecContext::requiredDef isRequired =
                          ExecContext::requiredDef_required) {
    ExecContext::execContextFieldT afield;
    afield.variableNeeded = variableUsed;
    afield.variableSize = sizeOfElement;
    afield.execContextOffset = offsetOfElement;
    afield.isRequired = isRequired;
    execContext_.execContextFields.push_back(
        std::make_unique<ExecContext::execContextFieldT>(afield));
    // use std::max to find the max size
    maxsize = std::max(maxsize, offsetOfElement + sizeOfElement);
  }
  // private C++ variable that holds the execContext
  ExecContext::execContextT execContext_;
  AICExecContext_32bitPointers placeholder;
  uint32_t maxsize = 0;
  std::vector<uint8_t> ExecContextBytes;
};

#endif // AICMETADATA_EXECCONTEXTWRITER_HPP
