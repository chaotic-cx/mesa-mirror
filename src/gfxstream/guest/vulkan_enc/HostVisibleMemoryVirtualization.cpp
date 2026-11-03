/*
 * Copyright 2018 Google
 * SPDX-License-Identifier: MIT
 */
#include "HostVisibleMemoryVirtualization.h"

#include <set>

#include "ResourceTracker.h"
#include "Resources.h"
#include "VkEncoder.h"
#include "util/detect_os.h"
#include "util/log.h"

namespace gfxstream {
namespace vk {

CoherentMemory::CoherentMemory(VirtGpuResourceMappingPtr blobMapping, uint64_t size,
                               VkDevice device, VkDeviceMemory memory)
    : mSize(size), mBlobMapping(blobMapping), mDevice(device), mMemory(memory) {
    util_vma_heap_init(&mHeap, kLargestPageSize, mSize);
    mHeap.alloc_high = false;
    mBaseAddr = blobMapping->asRawPtr();
}

#if DETECT_OS_ANDROID
CoherentMemory::CoherentMemory(GoldfishAddressSpaceBlockPtr block, uint64_t gpuAddr, uint64_t size,
                               VkDevice device, VkDeviceMemory memory)
    : mSize(size), mBlock(block), mDevice(device), mMemory(memory) {
    util_vma_heap_init(&mHeap, kLargestPageSize, mSize);
    mHeap.alloc_high = false;
    mBaseAddr = (uint8_t*)block->mmap(gpuAddr);
}
#endif  // DETECT_OS_ANDROID

CoherentMemory::~CoherentMemory() {
    ResourceTracker::getThreadLocalEncoder()->vkFreeMemorySyncGOOGLE(mDevice, mMemory, nullptr,
                                                                     false);
    util_vma_heap_finish(&mHeap);
}

VkDeviceMemory CoherentMemory::getDeviceMemory() const { return mMemory; }

bool CoherentMemory::subAllocate(uint64_t size, uint8_t** ptr, uint64_t& offset) {
    if (!mBaseAddr || size == 0) return false;
    // 2^12 = 4096 (page size)
    uint64_t addr = util_vma_heap_alloc(&mHeap, size, 4096);
    if (!addr) return false;

    offset = addr - kLargestPageSize;
    mAllocations[offset] = size;
    *ptr = mBaseAddr + offset;
    return true;
}

bool CoherentMemory::release(uint64_t offset) {
    auto it = mAllocations.find(offset);
    if (it != mAllocations.end()) {
        util_vma_heap_free(&mHeap, offset + kLargestPageSize, it->second);
        mAllocations.erase(it);
        return true;
    } else {
        mesa_loge("unable to find block");
    }

    return false;
}

}  // namespace vk
}  // namespace gfxstream
