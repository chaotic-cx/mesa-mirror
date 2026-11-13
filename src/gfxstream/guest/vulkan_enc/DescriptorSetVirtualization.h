/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "gfxstream_vk_private.h"
#include "util/simple_mtx.h"

namespace gfxstream {
namespace vk {

enum DescriptorWriteType {
    Empty = 0,
    ImageInfo = 1,
    BufferInfo = 2,
    BufferView = 3,
    AccelerationStructure = 4,
};

struct DescriptorBindingElementWrite {
    DescriptorWriteType type = DescriptorWriteType::Empty;
    bool pendingWrite = false;
    uint32_t dstArrayElement = 0;  // Only used for accelerationStructure.

    union {
        VkDescriptorImageInfo imageInfo;
        VkDescriptorBufferInfo bufferInfo;
        VkBufferView bufferView;
        VkWriteDescriptorSetAccelerationStructureKHR accelerationStructure;
    };

    AliveTokenRef alive;
    AliveTokenRef samplerAlive;
};

struct DescriptorBindingSparseWrite {
    uint32_t descriptorCount = 0;
    std::unordered_map<uint32_t, DescriptorBindingElementWrite> elements;
    std::vector<uint32_t> dirtyIndices;
};

struct DescriptorInlineUniformBlockWrite {
    bool pendingWrite = false;
    std::vector<uint8_t> bytes;
    VkWriteDescriptorSetInlineUniformBlock writeInfo;
};

struct DescriptorBindingWrite {
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    std::variant<
        // Unused / unconfigured binding.
        std::monostate,
        // The writes of non-inline uniform descriptors.
        DescriptorBindingSparseWrite,
        // The writes of inline uniform descriptors.
        DescriptorInlineUniformBlockWrite>
        elements;
};

// All of the writes for a given descriptor.
struct DescriptorWriteTable {
    std::vector<DescriptorBindingWrite> bindings;
};

struct DescriptorWriteArrayRange {
    uint32_t begin;
    uint32_t count;
};

using DescriptorWriteDstArrayRangeTable = std::vector<std::vector<DescriptorWriteArrayRange>>;

struct ReifiedDescriptorSet {
    mutable std::recursive_mutex lock;
    VkDescriptorPool pool;
    VkDescriptorSetLayout setLayout;
    uint64_t poolId;
    std::atomic<bool> allocationPending{false};
    bool allocatedFromPoolIds = false;
    std::atomic<bool> hasPendingWrites{false};

    // Indexed first by binding number
    DescriptorWriteTable allWrites;

    // Indexed first by binding number
    DescriptorWriteDstArrayRangeTable pendingWriteArrayRanges;

    // Indexed by binding number
    std::vector<bool> bindingIsImmutableSampler;

    // Copied from the descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> bindings;
};

struct DescriptorPoolAllocationInfo {
    DescriptorPoolAllocationInfo() { simple_mtx_init(&poolLock, mtx_plain); }
    ~DescriptorPoolAllocationInfo() { simple_mtx_destroy(&poolLock); }
    DescriptorPoolAllocationInfo(const DescriptorPoolAllocationInfo&) = delete;
    DescriptorPoolAllocationInfo& operator=(const DescriptorPoolAllocationInfo&) = delete;

    simple_mtx_t poolLock;
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorPoolCreateFlags createFlags = 0;

    // TODO: This should be in a single fancy data structure of some kind.
    std::vector<uint64_t> freePoolIds;
    std::unordered_set<uint64_t> allocedPoolIds;
    std::unordered_set<VkDescriptorSet> allocedSets;
    uint32_t maxSets = 0;
    uint32_t usedSets = 0;

    // Fine-grained tracking of descriptor counts in individual pools
    struct DescriptorCountInfo {
        VkDescriptorType type;
        uint32_t descriptorCount;
        uint32_t used;
    };
    std::vector<DescriptorCountInfo> descriptorCountInfo;
};

struct DescriptorSetLayoutInfo {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorBindingFlags> bindingFlags;
    std::optional<VkAllocationCallbacks> destroyAllocator;
    std::atomic<uint32_t> refcount{0};
};

void clearReifiedDescriptorSet(ReifiedDescriptorSet* set);

void initDescriptorWriteTable(const std::vector<VkDescriptorSetLayoutBinding>& layoutBindings,
                              DescriptorWriteTable& table);

bool isDescriptorTypeImageInfo(VkDescriptorType descType);
bool isDescriptorTypeBufferInfo(VkDescriptorType descType);
bool isDescriptorTypeBufferView(VkDescriptorType descType);
bool isDescriptorTypeInlineUniformBlock(VkDescriptorType descType);
bool isDescriptorTypeAccelerationStructure(VkDescriptorType descType);

void doEmulatedDescriptorWrite(const VkWriteDescriptorSet* write, ReifiedDescriptorSet* toWrite);
void doEmulatedDescriptorCopy(const VkCopyDescriptorSet* copy, const ReifiedDescriptorSet* src,
                              ReifiedDescriptorSet* dst);

void doEmulatedDescriptorImageInfoWriteFromTemplate(VkDescriptorType descType, uint32_t binding,
                                                    uint32_t dstArrayElement, uint32_t count,
                                                    const VkDescriptorImageInfo* imageInfos,
                                                    ReifiedDescriptorSet* set);

void doEmulatedDescriptorBufferInfoWriteFromTemplate(VkDescriptorType descType, uint32_t binding,
                                                     uint32_t dstArrayElement, uint32_t count,
                                                     const VkDescriptorBufferInfo* bufferInfos,
                                                     ReifiedDescriptorSet* set);

void doEmulatedDescriptorBufferViewWriteFromTemplate(VkDescriptorType descType, uint32_t binding,
                                                     uint32_t dstArrayElement, uint32_t count,
                                                     const VkBufferView* bufferViews,
                                                     ReifiedDescriptorSet* set);

void doEmulatedDescriptorInlineUniformBlockFromTemplate(VkDescriptorType descType, uint32_t binding,
                                                        uint32_t dstArrayElement, uint32_t count,
                                                        const void* pData,
                                                        ReifiedDescriptorSet* set);

bool hasVariableDescriptorCountLayout(const VkDescriptorSetAllocateInfo* pAllocateInfo);
VkResult validateDescriptorSetAllocation(const VkDescriptorSetAllocateInfo* pAllocateInfo);
void applyDescriptorSetAllocation(VkDescriptorPool pool, VkDescriptorSetLayout setLayout,
                                  const uint32_t* variableDescriptorCount = nullptr);
bool fillDescriptorSetInfoForPool(VkDescriptorPool pool, VkDescriptorSetLayout setLayout,
                                  VkDescriptorSet set, bool allocatedFromPoolIds = true,
                                  const uint32_t* variableDescriptorCount = nullptr);
VkResult validateAndApplyVirtualDescriptorSetAllocation(
    VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pSets);

// Returns false if set wasn't found in its pool.
bool removeDescriptorSetFromPool(VkDescriptorSet set, bool usePoolIds);

std::vector<VkDescriptorSet> clearDescriptorPool(VkDescriptorPool pool, bool usePoolIds);

}  // namespace vk
}  // namespace gfxstream
