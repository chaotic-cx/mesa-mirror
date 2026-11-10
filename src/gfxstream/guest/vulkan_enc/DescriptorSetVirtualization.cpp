/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "DescriptorSetVirtualization.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <type_traits>

#include "Resources.h"
#include "util/log.h"
#include "vk_util.h"

namespace gfxstream {
namespace vk {

void clearReifiedDescriptorSet(ReifiedDescriptorSet* set) {
    std::lock_guard<std::recursive_mutex> lock(set->lock);
    set->pool = VK_NULL_HANDLE;
    set->setLayout = VK_NULL_HANDLE;
    set->poolId = -1;
    set->allocationPending = false;
    set->allocatedFromPoolIds = false;
    set->allWrites.bindings.clear();
    set->pendingWriteArrayRanges.clear();
}

static std::vector<VkDescriptorSetLayoutBinding> getEffectiveLayoutBindings(
    const DescriptorSetLayoutInfo& layoutInfo, const uint32_t* variableDescriptorCount) {
    std::vector<VkDescriptorSetLayoutBinding> bindings = layoutInfo.bindings;
    for (size_t i = 0; i < bindings.size(); ++i) {
        if (i < layoutInfo.bindingFlags.size() &&
            (layoutInfo.bindingFlags[i] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)) {
            bindings[i].descriptorCount =
                variableDescriptorCount
                    ? std::min(*variableDescriptorCount, bindings[i].descriptorCount)
                    : 0;
        }
    }
    return bindings;
}

bool hasVariableDescriptorCountLayout(const VkDescriptorSetAllocateInfo* pAllocateInfo) {
    if (!pAllocateInfo || !pAllocateInfo->pSetLayouts) return false;
    for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; ++i) {
        if (!pAllocateInfo->pSetLayouts[i]) continue;
        const auto* dsl =
            gfxstream_vk_descriptor_set_layout_from_handle(pAllocateInfo->pSetLayouts[i]);
        if (!dsl || !dsl->layoutInfo) continue;
        for (VkDescriptorBindingFlags flags : dsl->layoutInfo->bindingFlags) {
            if (flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) {
                return true;
            }
        }
    }
    return false;
}

void initDescriptorWriteTable(const std::vector<VkDescriptorSetLayoutBinding>& layoutBindings,
                              DescriptorWriteTable& table) {
    uint32_t highestBindingNumber = 0;
    for (const VkDescriptorSetLayoutBinding& binding : layoutBindings) {
        highestBindingNumber = std::max(highestBindingNumber, binding.binding);
    }

    table.bindings.clear();
    table.bindings.resize(highestBindingNumber + 1);

    for (const VkDescriptorSetLayoutBinding& binding : layoutBindings) {
        DescriptorBindingWrite& bindingWrite = table.bindings[binding.binding];
        bindingWrite.descriptorType = binding.descriptorType;
        if (isDescriptorTypeInlineUniformBlock(binding.descriptorType)) {
            DescriptorInlineUniformBlockWrite blockWrite;
            blockWrite.pendingWrite = false;
            blockWrite.bytes.resize(binding.descriptorCount, 0);
            bindingWrite.elements = std::move(blockWrite);
        } else {
            DescriptorBindingSparseWrite sparseWrite;
            sparseWrite.descriptorCount = binding.descriptorCount;
            bindingWrite.elements = std::move(sparseWrite);
        }
    }
}

enum class CopyDirection {
    ReadFromTable,
    WriteToTable,
};

// Handles either reading bytes from or writing bytes to inline uniform block
// descriptors while handling spilling over into the next binding(s) as needed.
//
// See the `ConsecutiveBindingUpdates` section in
// https://docs.vulkan.org/refpages/latest/refpages/source/VkWriteDescriptorSet.html
template <CopyDirection Dir>
static void copyInlineUniformBlockBytes(
        // If reading from the table, `table` is const.
        std::conditional_t<Dir == CopyDirection::WriteToTable,  //
                           DescriptorWriteTable&,               //
                           const DescriptorWriteTable&> table,
        uint32_t startBinding,
        uint32_t startByteOffset,
        // If reading from the table, `data` is not const.
        std::conditional_t<Dir == CopyDirection::WriteToTable,  //
                           const void*,                         //
                           void*> data,
        uint32_t dataSize) {
    auto* dataPtr = static_cast<
        std::conditional_t<Dir == CopyDirection::WriteToTable, const uint8_t*, uint8_t*>>(data);
    uint32_t remainingBytes = dataSize;
    uint32_t currBinding = startBinding;
    uint32_t currBindingByteOffset = startByteOffset;

    while (remainingBytes > 0 && currBinding < table.bindings.size()) {
        auto* blockWrite =
            std::get_if<DescriptorInlineUniformBlockWrite>(&table.bindings[currBinding].elements);
        if (!blockWrite) {
            ++currBinding;
            continue;
        }

        if (currBindingByteOffset >= blockWrite->bytes.size()) {
            currBindingByteOffset -= static_cast<uint32_t>(blockWrite->bytes.size());
            ++currBinding;
            continue;
        }

        const uint32_t maxBytesInBinding =
            static_cast<uint32_t>(blockWrite->bytes.size()) - currBindingByteOffset;
        const uint32_t bytesToCopy = std::min(remainingBytes, maxBytesInBinding);

        if constexpr (Dir == CopyDirection::WriteToTable) {
            memcpy(blockWrite->bytes.data() + currBindingByteOffset, dataPtr, bytesToCopy);
            blockWrite->pendingWrite = true;
            table.bindings[currBinding].descriptorType = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
        } else {
            memcpy(dataPtr, blockWrite->bytes.data() + currBindingByteOffset, bytesToCopy);
        }

        dataPtr += bytesToCopy;
        remainingBytes -= bytesToCopy;
        currBindingByteOffset = 0;
        ++currBinding;
    }
}

static void readInlineUniformBlockBytes(const DescriptorWriteTable& table, uint32_t srcBinding,
                                        uint32_t srcArrayElement, void* pData, uint32_t dataSize) {
    copyInlineUniformBlockBytes<CopyDirection::ReadFromTable>(table, srcBinding, srcArrayElement,
                                                              pData, dataSize);
}

static void writeInlineUniformBlockBytes(DescriptorWriteTable& table, uint32_t dstBinding,
                                         uint32_t dstArrayElement, const void* pData,
                                         uint32_t dataSize) {
    copyInlineUniformBlockBytes<CopyDirection::WriteToTable>(table, dstBinding, dstArrayElement,
                                                             pData, dataSize);
}

// Handles either reading elements from or writing elements to non- inline uniform
// block descriptors while handling spilling over into the next binding(s) as needed.
//
// See the `ConsecutiveBindingUpdates` section in
// https://docs.vulkan.org/refpages/latest/refpages/source/VkWriteDescriptorSet.html
template <CopyDirection Dir, typename Fn>
static void forEachDescriptorBindingElement(
        std::conditional_t<Dir == CopyDirection::WriteToTable,  //
                           DescriptorWriteTable&,               //
                           const DescriptorWriteTable&> table,
        uint32_t startBinding,
        uint32_t startArrayElement,
        uint32_t count,
        Fn&& fn) {
    static const DescriptorBindingElementWrite kEmptyElement{};
    uint32_t currBinding = startBinding;
    uint32_t currArrayElement = startArrayElement;
    uint32_t i = 0;
    while (i < count && currBinding < table.bindings.size()) {
        auto* sparse =
            std::get_if<DescriptorBindingSparseWrite>(&table.bindings[currBinding].elements);
        if (!sparse) {
            ++currBinding;
            continue;
        }
        if (currArrayElement >= sparse->descriptorCount) {
            currArrayElement -= sparse->descriptorCount;
            ++currBinding;
            continue;
        }
        if constexpr (Dir == CopyDirection::WriteToTable) {
            auto& entry = sparse->elements[currArrayElement];
            const bool wasPending = entry.pendingWrite;
            fn(table.bindings[currBinding], entry, i);
            if (entry.type == DescriptorWriteType::Empty) {
                sparse->elements.erase(currArrayElement);
            } else {
                entry.pendingWrite = true;
                if (!wasPending) {
                    sparse->dirtyIndices.push_back(currArrayElement);
                }
            }
        } else {
            auto it = sparse->elements.find(currArrayElement);
            fn(table.bindings[currBinding],
               it != sparse->elements.end() ? it->second : kEmptyElement, i);
        }
        ++i;
        ++currArrayElement;
    }
}

static void readDescriptorBindingElements(const DescriptorWriteTable& table, uint32_t srcBinding,
                                          uint32_t srcArrayElement, uint32_t count,
                                          DescriptorBindingElementWrite* outElements) {
    forEachDescriptorBindingElement<CopyDirection::ReadFromTable>(
        table, srcBinding, srcArrayElement, count,
        [&](const DescriptorBindingWrite&,
            const DescriptorBindingElementWrite& element,
            uint32_t i) {
            outElements[i] = element;
        });
}

static void writeDescriptorBindingElements(DescriptorWriteTable& table, uint32_t dstBinding,
                                           uint32_t dstArrayElement, uint32_t count,
                                           const DescriptorBindingElementWrite* inElements) {
    forEachDescriptorBindingElement<CopyDirection::WriteToTable>(
        table, dstBinding, dstArrayElement, count,
        [&](DescriptorBindingWrite&,
            DescriptorBindingElementWrite& element,
            uint32_t i) {
            element = inElements[i];
        });
}

static void initializeReifiedDescriptorSet(VkDescriptorPool pool, VkDescriptorSetLayout setLayout,
                                           ReifiedDescriptorSet* set, bool allocationPending,
                                           const uint32_t* variableDescriptorCount = nullptr) {
    set->pendingWriteArrayRanges.clear();

    const auto& layoutInfo =
        *(gfxstream_vk_descriptor_set_layout_from_handle(setLayout)->layoutInfo);

    std::vector<VkDescriptorSetLayoutBinding> effectiveBindings =
        getEffectiveLayoutBindings(layoutInfo, variableDescriptorCount);

    initDescriptorWriteTable(effectiveBindings, set->allWrites);

    for (size_t i = 0; i < effectiveBindings.size(); ++i) {
        // Bindings can be sparsely defined
        const auto& binding = effectiveBindings[i];
        uint32_t bindingIndex = binding.binding;
        if (set->bindingIsImmutableSampler.size() <= bindingIndex) {
            set->bindingIsImmutableSampler.resize(bindingIndex + 1, false);
        }
        set->bindingIsImmutableSampler[bindingIndex] =
            binding.descriptorCount > 0 &&
            (binding.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
             binding.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) &&
            binding.pImmutableSamplers;
    }

    set->pool = pool;
    set->setLayout = setLayout;
    set->allocationPending = allocationPending;
    set->bindings = std::move(effectiveBindings);
}

bool isDescriptorTypeImageInfo(VkDescriptorType descType) {
    return (descType == VK_DESCRIPTOR_TYPE_SAMPLER) ||
           (descType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ||
           (descType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) ||
           (descType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) ||
           (descType == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
}

bool isDescriptorTypeBufferInfo(VkDescriptorType descType) {
    return (descType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) ||
           (descType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) ||
           (descType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ||
           (descType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
}

bool isDescriptorTypeBufferView(VkDescriptorType descType) {
    return (descType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) ||
           (descType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
}

bool isDescriptorTypeInlineUniformBlock(VkDescriptorType descType) {
    return descType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT;
}

bool isDescriptorTypeAccelerationStructure(VkDescriptorType descType) {
    return descType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
}

void doEmulatedDescriptorWrite(const VkWriteDescriptorSet* write, ReifiedDescriptorSet* toWrite) {
    std::lock_guard<std::recursive_mutex> lock(toWrite->lock);
    VkDescriptorType descType = write->descriptorType;
    uint32_t dstBinding = write->dstBinding;
    uint32_t dstArrayElement = write->dstArrayElement;
    uint32_t descriptorCount = write->descriptorCount;

    if (isDescriptorTypeImageInfo(descType)) {
        doEmulatedDescriptorImageInfoWriteFromTemplate(descType, dstBinding, dstArrayElement,
                                                       descriptorCount, write->pImageInfo, toWrite);
    } else if (isDescriptorTypeBufferInfo(descType)) {
        doEmulatedDescriptorBufferInfoWriteFromTemplate(
            descType, dstBinding, dstArrayElement, descriptorCount, write->pBufferInfo, toWrite);
    } else if (isDescriptorTypeBufferView(descType)) {
        doEmulatedDescriptorBufferViewWriteFromTemplate(descType, dstBinding, dstArrayElement,
                                                        descriptorCount, write->pTexelBufferView,
                                                        toWrite);
    } else if (isDescriptorTypeInlineUniformBlock(descType)) {
        const VkWriteDescriptorSetInlineUniformBlock* descInlineUniformBlock =
            static_cast<const VkWriteDescriptorSetInlineUniformBlock*>(write->pNext);
        while (descInlineUniformBlock &&
               descInlineUniformBlock->sType !=
                   VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK) {
            descInlineUniformBlock = static_cast<const VkWriteDescriptorSetInlineUniformBlock*>(
                descInlineUniformBlock->pNext);
        }
        if (!descInlineUniformBlock) {
            mesa_loge("%s: did not find inline uniform block\n", __func__);
            return;
        }
        writeInlineUniformBlockBytes(toWrite->allWrites, dstBinding, dstArrayElement,
                                     descInlineUniformBlock->pData,
                                     descInlineUniformBlock->dataSize);
    } else if (isDescriptorTypeAccelerationStructure(descType)) {
        // TODO
        // Look for pNext inline uniform block or acceleration structure.
        // Append new DescriptorWrite entry that holds the buffer
        mesa_logw("%s: Ignoring emulated write for descriptor type 0x%x\n", __func__, descType);
    }
}

void doEmulatedDescriptorCopy(const VkCopyDescriptorSet* copy, const ReifiedDescriptorSet* src,
                              ReifiedDescriptorSet* dst) {
    // Lock in total pointer order (matching commitDescriptorSetUpdates) instead of
    // std::lock, which is unavailable in libc++ under -fno-exceptions. Because
    // ReifiedDescriptorSet::lock is recursive, src == dst locks cleanly as well.
    const bool srcFirst = std::less<const ReifiedDescriptorSet*>{}(src, dst);
    const ReifiedDescriptorSet* first = srcFirst ? src : dst;
    const ReifiedDescriptorSet* second = srcFirst ? dst : src;
    std::lock_guard<std::recursive_mutex> firstLock(first->lock);
    std::lock_guard<std::recursive_mutex> secondLock(second->lock);
    const DescriptorWriteTable& srcTable = src->allWrites;
    DescriptorWriteTable& dstTable = dst->allWrites;

    if (copy->descriptorCount == 0 || copy->srcBinding >= srcTable.bindings.size()) {
        return;
    }

    // src/dst may be the same descriptor set, so we need to create a temporary array for that case.
    // (TODO: Maybe just notice the pointers are the same? can aliasing in any other way happen?)

    if (isDescriptorTypeInlineUniformBlock(srcTable.bindings[copy->srcBinding].descriptorType)) {
        std::vector<uint8_t> bytesToCopy(copy->descriptorCount, 0);
        readInlineUniformBlockBytes(srcTable, copy->srcBinding, copy->srcArrayElement,
                                    bytesToCopy.data(), copy->descriptorCount);

        writeInlineUniformBlockBytes(dstTable, copy->dstBinding, copy->dstArrayElement,
                                     bytesToCopy.data(), copy->descriptorCount);
    } else {
        std::vector<DescriptorBindingElementWrite> elementsToCopy(copy->descriptorCount);
        readDescriptorBindingElements(srcTable, copy->srcBinding, copy->srcArrayElement,
                                    copy->descriptorCount, elementsToCopy.data());

        writeDescriptorBindingElements(dstTable, copy->dstBinding, copy->dstArrayElement,
                                    copy->descriptorCount, elementsToCopy.data());
    }
}

void doEmulatedDescriptorImageInfoWriteFromTemplate(VkDescriptorType descType, uint32_t binding,
                                                    uint32_t dstArrayElement, uint32_t count,
                                                    const VkDescriptorImageInfo* imageInfos,
                                                    ReifiedDescriptorSet* set) {
    std::lock_guard<std::recursive_mutex> lock(set->lock);
    forEachDescriptorBindingElement<CopyDirection::WriteToTable>(
        set->allWrites, binding, dstArrayElement, count,
        [&](DescriptorBindingWrite& bindingWrite, DescriptorBindingElementWrite& entry,
            uint32_t i) {
            entry.imageInfo = imageInfos[i];
            entry.type = DescriptorWriteType::ImageInfo;
            bindingWrite.descriptorType = descType;
            auto* imageView = gfxstream_vk_image_view_from_handle(entry.imageInfo.imageView);
            entry.alive = imageView ? imageView->alive : std::weak_ptr<bool>();
            auto* sampler = gfxstream_vk_sampler_from_handle(entry.imageInfo.sampler);
            entry.samplerAlive = sampler ? sampler->alive : std::weak_ptr<bool>();
        });
}

void doEmulatedDescriptorBufferInfoWriteFromTemplate(VkDescriptorType descType, uint32_t binding,
                                                     uint32_t dstArrayElement, uint32_t count,
                                                     const VkDescriptorBufferInfo* bufferInfos,
                                                     ReifiedDescriptorSet* set) {
    std::lock_guard<std::recursive_mutex> lock(set->lock);
    forEachDescriptorBindingElement<CopyDirection::WriteToTable>(
        set->allWrites, binding, dstArrayElement, count,
        [&](DescriptorBindingWrite& bindingWrite, DescriptorBindingElementWrite& entry,
            uint32_t i) {
            entry.bufferInfo = bufferInfos[i];
            entry.type = DescriptorWriteType::BufferInfo;
            bindingWrite.descriptorType = descType;
            auto* buffer = gfxstream_vk_buffer_from_handle(entry.bufferInfo.buffer);
            entry.alive = buffer ? buffer->alive : std::weak_ptr<bool>();
        });
}

void doEmulatedDescriptorBufferViewWriteFromTemplate(VkDescriptorType descType, uint32_t binding,
                                                     uint32_t dstArrayElement, uint32_t count,
                                                     const VkBufferView* bufferViews,
                                                     ReifiedDescriptorSet* set) {
    std::lock_guard<std::recursive_mutex> lock(set->lock);
    forEachDescriptorBindingElement<CopyDirection::WriteToTable>(
        set->allWrites, binding, dstArrayElement, count,
        [&](DescriptorBindingWrite& bindingWrite, DescriptorBindingElementWrite& entry,
            uint32_t i) {
            entry.bufferView = bufferViews[i];
            entry.type = DescriptorWriteType::BufferView;
            bindingWrite.descriptorType = descType;
            auto* bufferView = gfxstream_vk_buffer_view_from_handle(entry.bufferView);
            entry.alive = bufferView ? bufferView->alive : std::weak_ptr<bool>();
        });
}

void doEmulatedDescriptorInlineUniformBlockFromTemplate(VkDescriptorType descType, uint32_t binding,
                                                        uint32_t dstArrayElement, uint32_t count,
                                                        const void* pData,
                                                        ReifiedDescriptorSet* set) {
    std::lock_guard<std::recursive_mutex> lock(set->lock);
    (void)descType;
    writeInlineUniformBlockBytes(set->allWrites, binding, dstArrayElement, pData, count);
}

static bool isBindingFeasibleForAlloc(
    const DescriptorPoolAllocationInfo::DescriptorCountInfo& countInfo,
    const VkDescriptorSetLayoutBinding& binding) {
    if (binding.descriptorCount && (countInfo.type != binding.descriptorType)) {
        return false;
    }

    uint32_t availDescriptorCount = countInfo.descriptorCount - countInfo.used;

    if (availDescriptorCount < binding.descriptorCount) {
        mesa_logd(
            "%s: Ran out of descriptors of type 0x%x. "
            "Wanted %u from layout but "
            "we only have %u free (total in pool: %u)\n",
            __func__, binding.descriptorType, binding.descriptorCount,
            countInfo.descriptorCount - countInfo.used, countInfo.descriptorCount);
        return false;
    }

    return true;
}

static bool isBindingFeasibleForFree(
    const DescriptorPoolAllocationInfo::DescriptorCountInfo& countInfo,
    const VkDescriptorSetLayoutBinding& binding) {
    if (countInfo.type != binding.descriptorType) return false;
    if (countInfo.used < binding.descriptorCount) {
        mesa_logd(
            "%s: Was a descriptor set double freed? "
            "Ran out of descriptors of type 0x%x. "
            "Wanted to free %u from layout but "
            "we only have %u used (total in pool: %u)\n",
            __func__, binding.descriptorType, binding.descriptorCount, countInfo.used,
            countInfo.descriptorCount);
        return false;
    }
    return true;
}

static void allocBindingFeasible(const VkDescriptorSetLayoutBinding& binding,
                                 DescriptorPoolAllocationInfo::DescriptorCountInfo& poolState) {
    poolState.used += binding.descriptorCount;
}

static void freeBindingFeasible(const VkDescriptorSetLayoutBinding& binding,
                                DescriptorPoolAllocationInfo::DescriptorCountInfo& poolState) {
    poolState.used -= binding.descriptorCount;
}

VkResult validateDescriptorSetAllocation(const VkDescriptorSetAllocateInfo* pAllocateInfo) {
    VkDescriptorPool pool = pAllocateInfo->descriptorPool;
    DescriptorPoolAllocationInfo* poolInfo =
        gfxstream_vk_descriptor_pool_from_handle(pool)->allocInfo;

    // Check the number of sets available.
    auto setsAvailable = poolInfo->maxSets - poolInfo->usedSets;

    if (setsAvailable < pAllocateInfo->descriptorSetCount) {
        mesa_logd(
            "%s: Error: VkDescriptorSetAllocateInfo wants %u sets "
            "but we only have %u available. "
            "Bailing with VK_ERROR_OUT_OF_POOL_MEMORY.\n",
            __func__, pAllocateInfo->descriptorSetCount, setsAvailable);
        return VK_ERROR_OUT_OF_POOL_MEMORY;
    }

    // Perform simulated allocation and error out with
    // VK_ERROR_OUT_OF_POOL_MEMORY if it fails.
    std::vector<DescriptorPoolAllocationInfo::DescriptorCountInfo> descriptorCountCopy =
        poolInfo->descriptorCountInfo;

    const auto* varDescCountAllocInfo = vk_find_struct_const(
        pAllocateInfo->pNext, DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

    for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; ++i) {
        if (!pAllocateInfo->pSetLayouts[i]) {
            mesa_logd("%s: Error: Tried to allocate a descriptor set with null set layout.\n",
                      __func__);
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto setLayoutInfo =
            gfxstream_vk_descriptor_set_layout_from_handle(pAllocateInfo->pSetLayouts[i])
                ->layoutInfo;
        if (!setLayoutInfo) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        const uint32_t* varCount =
            (varDescCountAllocInfo && varDescCountAllocInfo->descriptorSetCount > i &&
             varDescCountAllocInfo->pDescriptorCounts)
                ? &varDescCountAllocInfo->pDescriptorCounts[i]
                : nullptr;
        auto effectiveBindings = getEffectiveLayoutBindings(*setLayoutInfo, varCount);

        for (const auto& binding : effectiveBindings) {
            if (binding.descriptorCount == 0) continue;
            bool success = false;
            for (auto& pool : descriptorCountCopy) {
                if (!isBindingFeasibleForAlloc(pool, binding)) continue;

                success = true;
                allocBindingFeasible(binding, pool);
                break;
            }

            if (!success) {
                return VK_ERROR_OUT_OF_POOL_MEMORY;
            }
        }
    }
    return VK_SUCCESS;
}

void applyDescriptorSetAllocation(VkDescriptorPool pool, VkDescriptorSetLayout setLayout,
                                  const uint32_t* variableDescriptorCount) {
    auto allocInfo = gfxstream_vk_descriptor_pool_from_handle(pool)->allocInfo;
    auto setLayoutInfo = gfxstream_vk_descriptor_set_layout_from_handle(setLayout)->layoutInfo;

    ++allocInfo->usedSets;

    auto effectiveBindings = getEffectiveLayoutBindings(*setLayoutInfo, variableDescriptorCount);
    for (const auto& binding : effectiveBindings) {
        if (binding.descriptorCount == 0) continue;
        for (auto& countForPool : allocInfo->descriptorCountInfo) {
            if (!isBindingFeasibleForAlloc(countForPool, binding)) continue;
            allocBindingFeasible(binding, countForPool);
            break;
        }
    }
}

static void removeDescriptorSetAllocation(
    VkDescriptorPool pool, const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    auto allocInfo = gfxstream_vk_descriptor_pool_from_handle(pool)->allocInfo;

    if (0 == allocInfo->usedSets) {
        mesa_logd("%s: Warning: a descriptor set was double freed.\n", __func__);
        return;
    }

    --allocInfo->usedSets;

    for (const auto& binding : bindings) {
        if (binding.descriptorCount == 0) continue;
        for (auto& countForPool : allocInfo->descriptorCountInfo) {
            if (!isBindingFeasibleForFree(countForPool, binding)) continue;
            freeBindingFeasible(binding, countForPool);
            break;
        }
    }
}

void fillDescriptorSetInfoForPool(VkDescriptorPool pool, VkDescriptorSetLayout setLayout,
                                  VkDescriptorSet set, bool allocatedFromPoolIds,
                                  const uint32_t* variableDescriptorCount) {
    DescriptorPoolAllocationInfo* allocInfo =
        gfxstream_vk_descriptor_pool_from_handle(pool)->allocInfo;

    ReifiedDescriptorSet* newReified = new ReifiedDescriptorSet;
    newReified->poolId = gfxstream_vk_descriptor_set_to_host_u64(set);
    newReified->allocatedFromPoolIds = allocatedFromPoolIds;
    newReified->allocationPending = allocatedFromPoolIds;

    gfxstream_vk_descriptor_set_from_handle(set)->reified = newReified;

    if (allocatedFromPoolIds) {
        allocInfo->allocedPoolIds.insert(newReified->poolId);
    }
    allocInfo->allocedSets.insert(set);

    initializeReifiedDescriptorSet(pool, setLayout, newReified, allocatedFromPoolIds,
                                   variableDescriptorCount);
}

VkResult validateAndApplyVirtualDescriptorSetAllocation(
    const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pSets) {
    VkResult validateRes = validateDescriptorSetAllocation(pAllocateInfo);

    if (validateRes != VK_SUCCESS) return validateRes;

    const auto* varDescCountAllocInfo = vk_find_struct_const(
        pAllocateInfo->pNext, DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

    for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; ++i) {
        const uint32_t* varCount =
            (varDescCountAllocInfo && varDescCountAllocInfo->descriptorSetCount > i &&
             varDescCountAllocInfo->pDescriptorCounts)
                ? &varDescCountAllocInfo->pDescriptorCounts[i]
                : nullptr;
        applyDescriptorSetAllocation(pAllocateInfo->descriptorPool, pAllocateInfo->pSetLayouts[i],
                                     varCount);
    }

    VkDescriptorPool pool = pAllocateInfo->descriptorPool;
    DescriptorPoolAllocationInfo* allocInfo =
        gfxstream_vk_descriptor_pool_from_handle(pool)->allocInfo;

    if (allocInfo->freePoolIds.size() < pAllocateInfo->descriptorSetCount) {
        mesa_loge(
            "%s: FATAL: Somehow out of descriptor pool IDs. Wanted %u IDs but only have %u free "
            "IDs remaining. The count for maxSets was %u and used was %u\n",
            __func__, pAllocateInfo->descriptorSetCount, (uint32_t)allocInfo->freePoolIds.size(),
            allocInfo->maxSets, allocInfo->usedSets);
    }

    for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; ++i) {
        uint64_t id = allocInfo->freePoolIds.back();
        allocInfo->freePoolIds.pop_back();

        VkDescriptorSet newSet = create_gfxstream_vk_descriptor_set(id);
        pSets[i] = newSet;

        const uint32_t* varCount =
            (varDescCountAllocInfo && varDescCountAllocInfo->descriptorSetCount > i &&
             varDescCountAllocInfo->pDescriptorCounts)
                ? &varDescCountAllocInfo->pDescriptorCounts[i]
                : nullptr;
        fillDescriptorSetInfoForPool(pool, pAllocateInfo->pSetLayouts[i], newSet,
                                     /*allocatedFromPoolIds=*/true, varCount);
    }

    return VK_SUCCESS;
}

bool removeDescriptorSetFromPool(VkDescriptorSet set, bool usePoolIds) {
    ReifiedDescriptorSet* reified = gfxstream_vk_descriptor_set_from_handle(set)->reified;

    VkDescriptorPool pool = reified->pool;
    DescriptorPoolAllocationInfo* allocInfo =
        gfxstream_vk_descriptor_pool_from_handle(pool)->allocInfo;

    const bool fromPoolIds = usePoolIds && reified->allocatedFromPoolIds;
    if (fromPoolIds) {
        // Look for the set's pool Id in the pool. If not found, then this wasn't really allocated,
        // and bail.
        if (allocInfo->allocedPoolIds.find(reified->poolId) == allocInfo->allocedPoolIds.end()) {
            return false;
        }
    } else if (allocInfo->allocedSets.find(set) == allocInfo->allocedSets.end()) {
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding>& bindings = reified->bindings;
    removeDescriptorSetAllocation(pool, bindings);

    if (fromPoolIds) {
        allocInfo->freePoolIds.push_back(reified->poolId);
        allocInfo->allocedPoolIds.erase(reified->poolId);
    }
    allocInfo->allocedSets.erase(set);

    return true;
}

std::vector<VkDescriptorSet> clearDescriptorPool(VkDescriptorPool pool, bool usePoolIds) {
    std::vector<VkDescriptorSet> toClear;
    auto* dp = gfxstream_vk_descriptor_pool_from_handle(pool);
    if (!dp || !dp->allocInfo) return toClear;
    for (auto set : dp->allocInfo->allocedSets) {
        toClear.push_back(set);
    }

    for (auto set : toClear) {
        removeDescriptorSetFromPool(set, usePoolIds);
    }

    return toClear;
}

}  // namespace vk
}  // namespace gfxstream
