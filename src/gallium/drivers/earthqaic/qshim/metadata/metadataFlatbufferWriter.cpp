// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "metadataFlatbufferWriter.h"
#include "AicMetadataFlat_generated.h"
#include "CompressL2TCMInitState.h"
#include "metadata_flat_knownFields.h"
#include <cassert>
#include <cstring> // memset
#include <tuple>
#include <unordered_set>
namespace {
std::unordered_set<std::string>
    known_metadata_fields(std::begin(known_AicMetadataFlat_fields),
                          std::end(known_AicMetadataFlat_fields));

[[nodiscard]] bool isKnownField(const std::string &requiredField) {
  std::ignore = known_AicMetadataFlat_fields_names;
  return known_metadata_fields.find(requiredField) !=
         known_metadata_fields.end();
}
} // namespace

MetadataFlatbufferWriter::MetadataFlatbufferWriter(
    uint16_t hwVersionMajor, uint16_t hwVersionMinor, uint16_t versionMajor,
    uint16_t versionMinor, uint16_t execContextMajorVersion,
    uint64_t exitDoorbellOffset) {
  metadata_.versionMajor = versionMajor;
  metadata_.versionMinor = versionMinor;
  metadata_.execContextMajorVersion = execContextMajorVersion;
  metadata_.hwVersionMajor = hwVersionMajor;
  metadata_.hwVersionMinor = hwVersionMinor;
  metadata_.exitDoorbellOffset = exitDoorbellOffset;
  metadata_.hostMulticastTable =
      std::make_unique<AicMetadataFlat::AICMDHostMulticastEntryTableT>();
  metadata_.QNNConfig = std::make_unique<AicMetadataFlat::QNNConfigDefT>();
  metadata_.networkHeapBehavior =
      std::make_unique<AicMetadataFlat::networkHeapBehaviorDefT>();
}

void MetadataFlatbufferWriter::serialize() {
  flatbuffers::FlatBufferBuilder builder;
  builder.ForceDefaults(true);
  builder.Finish(AicMetadataFlat::Metadata::Pack(builder, &metadata_));
  std::vector<uint8_t> outvalue(builder.GetBufferPointer(),
                                builder.GetBufferPointer() + builder.GetSize());
  const auto begin = reinterpret_cast<const uint8_t *>(&termMetadata);
  const auto end = std::next(begin, sizeof(termMetadata));
  outvalue.insert(outvalue.begin(), begin, end);
  this->metadataBuffer = outvalue;
}

void MetadataFlatbufferWriter::finalize() {
  assert(getNumNSPs() && "Must set numNSPs");
  assert(getVTCMSize() && "Must set VTCMSize");
  assert(getL2TCMSize() && "Must set L2TCMSize");
  this->addIntrospectionString("AicMetadataFlat_Metadata->requiredFields");
  PopulateL2TCMInitStateNonZeroRegions();
  // Write network name.
  if (metadata_.networkName.empty())
    metadata_.networkName = "unnamed";
  if (metadata_.execContext.empty()) {
    ExecContextWriter context(ExecContextWriter::defaultType::DEFAULT);
    setExecContext(context.getExecContext());
  }
  addIntrospectionString("AicMetadataFlat_Metadata->networkName");
  if (metadata_.raw_struct_version_length == 0) {
    serialize();
    metadata_.raw_struct_version_length = 2 * this->metadataBuffer.size();
  }
  serialize();
}

bool MetadataFlatbufferWriter::writeMetadata(const char *outfile) const {
  assert(isFinal() && "Metadata not finalized.");
  assert(getNumNSPs() && "Must set numNSPs");
  assert(getVTCMSize() && "Must set VTCMSize");
  assert(getL2TCMSize() && "Must set L2TCMSize");
  assert(outfile && "null outfile");

  FILE *fd = fopen(outfile, "wb");
  if (fd == nullptr) {
    assert(!"writeMetadata: File open failed");
    return false;
  }
  size_t bytes = metadataBuffer.size();
  if (fwrite(metadataBuffer.data(), 1, bytes, fd) != bytes) {
    assert(!"writeMetadata: fwrite failed");
    fclose(fd);
    return false;
  }
  if (fclose(fd) != 0) {
    assert(!"writeMetadata: fclose failed");
    return false;
  }
  return true;
}

void MetadataFlatbufferWriter::addSemaphoreOp(SemaphoreOps &semaphoreOps,
                                              uint8_t semOp, uint16_t semNum,
                                              uint16_t semValue,
                                              uint8_t preOrPost,
                                              uint8_t inSyncFence,
                                              uint8_t outSyncFence) {
  bool validSemaphoreOp;
  switch (semOp) {
  default:
    validSemaphoreOp = false;
    break;
  case AICMDSemaphoreCmdNOP:    // nop.
  case AICMDSemaphoreCmdINIT:   // Set sem to val.
  case AICMDSemaphoreCmdINC:    // Increment sem.
  case AICMDSemaphoreCmdDEC:    // Decrement sem.
  case AICMDSemaphoreCmdWAITEQ: // Wait for sem to be == val.
  case AICMDSemaphoreCmdWAITGE: // Wait for sem to be >= val.
  case AICMDSemaphoreCmdP:      // Wait for sem to be > 0 and decrement.
    validSemaphoreOp = true;
    break;
  }
  assert(validSemaphoreOp && "Invalid DMA semaphore operation");
  (void)validSemaphoreOp;

  SemaphoreOp semaphoreOp;
  memset(&semaphoreOp, 0, sizeof(SemaphoreOp));
  semaphoreOp.semOp = semOp;
  semaphoreOp.semNum = semNum;
  semaphoreOp.semValue = semValue;
  semaphoreOp.preOrPost = preOrPost;
  semaphoreOp.inSyncFence = inSyncFence;
  semaphoreOp.outSyncFence = outSyncFence;
  semaphoreOps.push_back(semaphoreOp);
}

void MetadataFlatbufferWriter::addDoorbellOp(DoorbellOps &doorbellsOps,
                                             uint8_t sizeEnum, uint16_t mcId,
                                             uint64_t offset, uint32_t data) {
  bool validDoorbellOp;
  switch (sizeEnum) {
  default:
    validDoorbellOp = false;
    break;
  case AICMDDoorballOpSize8:
    [[fallthrough]];
  case AICMDDoorballOpSize16:
    [[fallthrough]];
  case AICMDDoorballOpSize32:
    validDoorbellOp = true;
    break;
  }
  assert(validDoorbellOp && "Invalid DMA doorbell operation");
  (void)validDoorbellOp;
  AICMDDoorbellOp doorbell;
  memset(&doorbell, 0, sizeof(AICMDDoorbellOp));
  doorbell.size = sizeEnum;
  doorbell.mcId = mcId;
  doorbell.offset = offset;
  doorbell.data = data;
  doorbellsOps.push_back(doorbell);
}
namespace {
static auto translateSemaphoreOp(const SemaphoreOp *s) {
  SemaphoreOpF AsemaphoreOp;
  AsemaphoreOp.semOp = s->semOp;
  AsemaphoreOp.semNum = s->semNum;
  AsemaphoreOp.semValue = s->semValue;
  AsemaphoreOp.preOrPost = s->preOrPost;
  AsemaphoreOp.inSyncFence = s->inSyncFence;
  AsemaphoreOp.outSyncFence = s->outSyncFence;
  return AsemaphoreOp;
}

auto translateDoorbellOp(const DoorbellOp *s) {
  DoorbellOpF AdoorbellOp;
  AdoorbellOp.size = s->size;
  AdoorbellOp.mcId = s->mcId;
  AdoorbellOp.offset = s->offset;
  AdoorbellOp.data = s->data;
  return AdoorbellOp;
}
} // namespace
void MetadataFlatbufferWriter::addDMARequest(
    uint16_t num, uint64_t hostOffset, uint8_t devAddrSpace, uint64_t devOffset,
    uint32_t size, uint8_t inOut, uint16_t portID, uint16_t mcId,
    SemaphoreOps &semaphoreOps, DoorbellOps &doorbellOps,
    uint32_t transactionId) {
  SemaphoreOpsF these_semaphoreOps;
  for (SemaphoreOp i : semaphoreOps) {
    these_semaphoreOps.push_back(translateSemaphoreOp(&i));
  }
  DoorbellOpsF these_doorbellOps;
  for (DoorbellOp i : doorbellOps) {
    these_doorbellOps.push_back(translateDoorbellOp(&i));
  }
  addDMARequest(num, hostOffset, devAddrSpace, devOffset, size, inOut, portID,
                mcId, these_semaphoreOps, these_doorbellOps, transactionId);
}

void MetadataFlatbufferWriter::addSemaphoreOp(SemaphoreOpsF &semaphoreOps,
                                              uint8_t semOp, uint16_t semNum,
                                              uint16_t semValue,
                                              uint8_t preOrPost,
                                              uint8_t inSyncFence,
                                              uint8_t outSyncFence) {
  bool validSemaphoreOp;
  switch (semOp) {
  default:
    validSemaphoreOp = false;
    break;
  case AicMetadataFlat::AICMDSemaphoreOpcode_AICMDSemaphoreCmdNOP:  // nop.
  case AicMetadataFlat::AICMDSemaphoreOpcode_AICMDSemaphoreCmdINIT: // Set sem
                                                                    // to val.
  case AicMetadataFlat::AICMDSemaphoreOpcode_AICMDSemaphoreCmdINC:  // Increment
                                                                    // sem.
  case AicMetadataFlat::AICMDSemaphoreOpcode_AICMDSemaphoreCmdDEC:  // Decrement
                                                                    // sem.
  case AicMetadataFlat::AICMDSemaphoreOpcode_AICMDSemaphoreCmdWAITEQ: // Wait
                                                                      // for sem
                                                                      // to be
                                                                      // ==
                                                                      // val.
  case AicMetadataFlat::AICMDSemaphoreOpcode_AICMDSemaphoreCmdWAITGE: // Wait
                                                                      // for sem
                                                                      // to be
                                                                      // >=
                                                                      // val.
  case AicMetadataFlat::AICMDSemaphoreOpcode_AICMDSemaphoreCmdP: // Wait for sem
                                                                 // to be > 0
                                                                 // and
                                                                 // decrement.
    validSemaphoreOp = true;
    break;
  }
  assert(validSemaphoreOp && "Invalid DMA semaphore operation");
  (void)validSemaphoreOp;

  SemaphoreOpF semaphoreOp;
  semaphoreOp.semOp = semOp;
  semaphoreOp.semNum = semNum;
  semaphoreOp.semValue = semValue;
  semaphoreOp.preOrPost = preOrPost;
  semaphoreOp.inSyncFence = inSyncFence;
  semaphoreOp.outSyncFence = outSyncFence;
  semaphoreOps.push_back(semaphoreOp);
}

void MetadataFlatbufferWriter::addDoorbellOp(DoorbellOpsF &doorbellsOps,
                                             uint8_t sizeEnum, uint16_t mcId,
                                             uint64_t offset, uint32_t data) {
  bool validDoorbellOp;
  switch (sizeEnum) {
  default:
    validDoorbellOp = false;
    break;
  case AicMetadataFlat::AICMDDoorbellOpSize_AICMDDoorballOpSize8:
    [[fallthrough]];
  case AicMetadataFlat::AICMDDoorbellOpSize_AICMDDoorballOpSize16:
    [[fallthrough]];
  case AicMetadataFlat::AICMDDoorbellOpSize_AICMDDoorballOpSize32:
    validDoorbellOp = true;
    break;
  }
  assert(validDoorbellOp && "Invalid DMA doorbell operation");
  (void)validDoorbellOp;
  AicMetadataFlat::AICMDDoorbellOpT doorbell;
  doorbell.size = sizeEnum;
  doorbell.mcId = mcId;
  doorbell.offset = offset;
  doorbell.data = data;
  doorbellsOps.push_back(doorbell);
}

void MetadataFlatbufferWriter::addDMARequest(
    uint16_t num, uint64_t hostOffset, uint8_t devAddrSpace, uint64_t devOffset,
    uint32_t size, uint8_t inOut, uint16_t portId, uint16_t mcId,
    SemaphoreOpsF &semaphoreOps, DoorbellOpsF &doorbellOps,
    uint32_t transactionId) {
  assert((devAddrSpace ==
              AicMetadataFlat::AICMDDMAEntryAddrSpace_AICMDDMAAddrSpaceMC ||
          devAddrSpace ==
              AicMetadataFlat::AICMDDMAEntryAddrSpace_AICMDDMAAddrSpaceDDR ||
          devAddrSpace ==
              AicMetadataFlat::
                  AICMDDMAEntryAddrSpace_AICMDDMAAddrSpaceDDRDynamicShared) &&
         "Invalid DMA address space.");
  // the below code is written this way to bypass bugs in flatbuffer 1.11.0
  metadata_.dmaRequests.push_back(
      std::make_unique<AicMetadataFlat::AICMDDMARequestT>());
  auto &lastDMARequest = metadata_.dmaRequests.back();
  lastDMARequest->num = num;
  lastDMARequest->hostOffset = hostOffset;
  lastDMARequest->devAddrSpace = devAddrSpace;
  lastDMARequest->devOffset = devOffset;
  lastDMARequest->size = size;
  lastDMARequest->inOut = inOut;
  lastDMARequest->portId = portId;
  lastDMARequest->mcId = mcId;
  lastDMARequest->transactionId = transactionId;

  for (auto semaphoreOp : semaphoreOps) {
    lastDMARequest->semaphoreOps.push_back(
        std::make_unique<AicMetadataFlat::AICMDSemaphoreOpT>(semaphoreOp));
  }
  for (auto doorbellOp : doorbellOps) {
    lastDMARequest->doorbellOps.push_back(
        std::make_unique<AicMetadataFlat::AICMDDoorbellOpT>(doorbellOp));
  }

  if (devAddrSpace ==
          AicMetadataFlat::
              AICMDDMAEntryAddrSpace_AICMDDMAAddrSpaceDDRDynamicShared &&
      size > 0) {
    enableDynamicSharedDDR();
  }
}

void MetadataFlatbufferWriter::addNSPMulticastEntry(int core, uint8_t dynamic,
                                                    uint32_t mask,
                                                    uint32_t size,
                                                    uint8_t addrSpace,
                                                    uint64_t baseAddrOffset) {
  assert(metadata_.numNSPs &&
         "Must specify the number of cores before adding a mc entry.");
  assert(core < metadata_.numNSPs);
  AicMetadataFlat::AICMDNSPMulticastEntryT mcEntry;
  mcEntry.dynamic = dynamic;
  mcEntry.mask = mask;
  mcEntry.size = size;
  mcEntry.addrSpace = addrSpace;
  mcEntry.baseAddrOffset = baseAddrOffset;
  metadata_.nspMulticastTables[core]->multicastEntries.push_back(
      std::make_unique<AicMetadataFlat::AICMDNSPMulticastEntryT>(mcEntry));
}

void MetadataFlatbufferWriter::addHostMulticastEntry(uint32_t mask,
                                                     uint32_t size) {
  AicMetadataFlat::AICMDHostMulticastEntryT mcEntry;
  mcEntry.mask = mask;
  mcEntry.size = size;
  metadata_.hostMulticastTable->multicastEntries.push_back(
      std::make_unique<AicMetadataFlat::AICMDHostMulticastEntryT>(mcEntry));
}

void MetadataFlatbufferWriter::addThreadDescriptor(nnc_activate_fp entryPoint,
                                                   bool hasHMX, bool hasHVX) {
  assert(entryPoint && "Invalid entryPoint.");
  AicMetadataFlat::AICMDThreadDescriptorT threadDesc;
  threadDesc.entryPoint = (uint64_t)entryPoint;
  threadDesc.typeMask = 0;
  if (hasHMX)
    threadDesc.typeMask |= AicMetadataFlat::AICMDThreadType_AICMDThreadHMX;
  if (hasHVX)
    threadDesc.typeMask |= AicMetadataFlat::AICMDThreadType_AICMDThreadHVX;
  metadata_.threadDescriptors.push_back(
      std::make_unique<AicMetadataFlat::AICMDThreadDescriptorT>(threadDesc));
}

void MetadataFlatbufferWriter::addThreadDescriptor(nnc_activate_fp entryPoint,
                                                   uint8_t typeMask) {
  assert(entryPoint && "Invalid entryPoint.");
  AicMetadataFlat::AICMDThreadDescriptorT threadDesc;
  threadDesc.entryPoint = (uint64_t)entryPoint;
  threadDesc.typeMask = typeMask;
  metadata_.threadDescriptors.push_back(
      std::make_unique<AicMetadataFlat::AICMDThreadDescriptorT>(threadDesc));
}

void MetadataFlatbufferWriter::addConstantMapping(uint32_t mask,
                                                  uint64_t offset,
                                                  uint32_t size) {
  assert((constMappingCores & mask) == 0 &&
         "NSPs are only allowed a single mapping.");

  constMappingCores |= mask;
  AicMetadataFlat::AICMDConstantMappingT mapping;
  mapping.coreMask = mask;
  mapping.constantDataBaseOffset = offset;
  mapping.size = size;
  metadata_.constantMappings.push_back(
      std::make_unique<AicMetadataFlat::AICMDConstantMappingT>(mapping));
}

void MetadataFlatbufferWriter::addIntrospectionString(
    const std::string &element) {

  if (!isKnownField(element)) {
    throw std::runtime_error("Attempted to add invalid field not found in "
                             "known fields.  Invalid field: " +
                             element);
  }
  // if the field is not already in the list, add it
  if (std::find(metadata_.requiredFields.begin(),
                metadata_.requiredFields.end(),
                element) == metadata_.requiredFields.end()) {
    metadata_.requiredFields.push_back(element);
  }
}

void MetadataFlatbufferWriter::addPort(uint16_t portId,
                                       AicMetadataFlat::AICMDPortType type) {
  metadata_.portTable.emplace_back(portId, type);
}

void MetadataFlatbufferWriter::PopulateL2TCMInitStateNonZeroRegions() {
  // clear the vector so we can call this function multiple times
  metadata_.L2TCMInitStateNonZeroRegions.clear();
  // from L2TCMInitState, find the zero regions and push to a vector
  auto zeroRegions = FindZeroRegions(metadata_.L2TCMInitState);
  // from the zero regions, find the non-zero regions and push to a vector
  auto nonZeroRegions =
      FindNonZeroRegions(zeroRegions, metadata_.L2TCMInitState.size());
  for (auto &region : nonZeroRegions) {
    AicMetadataFlat::NonZeroRegionT Region;
    Region.start = region.start;
    Region.end = region.end;
    Region.size = region.size;
    metadata_.L2TCMInitStateNonZeroRegions.push_back(
        std::make_unique<AicMetadataFlat::NonZeroRegionT>(Region));
  }
}
