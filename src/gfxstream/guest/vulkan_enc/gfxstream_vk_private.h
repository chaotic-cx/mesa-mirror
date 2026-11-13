/*
 * Copyright © 2023 Google Inc.
 *
 * derived from panvk_private.h driver which is:
 * Copyright © 2021 Collabora Ltd.
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GFXSTREAM_VK_PRIVATE_H
#define GFXSTREAM_VK_PRIVATE_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vk_icd.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "HostVisibleMemoryVirtualization.h"
#include "VirtGpu.h"
#include "gfxstream_vk_entrypoints.h"
#include "util/detect_os.h"
#include "util/list.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_device.h"
#include "vk_device_memory.h"
#include "vk_extensions.h"
#include "vk_fence.h"
#include "vk_image.h"
#include "vk_instance.h"
#include "vk_log.h"
#include "vk_object.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_semaphore.h"
#include "vk_ycbcr_conversion.h"
#include "vulkan/wsi/wsi_common.h"

#ifdef VK_USE_PLATFORM_ANDROID_KHR
#include <android/hardware_buffer.h>
#endif

#ifdef VK_USE_PLATFORM_FUCHSIA
#include <fidl/fuchsia.hardware.goldfish/cpp/wire.h>
#include <fidl/fuchsia.sysmem/cpp/wire.h>
#include <lib/zx/channel.h>
#include <lib/zx/vmo.h>
#include <zircon/types.h>
#else
#ifndef ZX_HANDLE_INVALID
typedef uint32_t zx_handle_t;
typedef uint64_t zx_koid_t;
#define ZX_HANDLE_INVALID ((zx_handle_t)0)
#define ZX_KOID_INVALID ((zx_koid_t)0)
#endif
#endif

#define GFXSTREAM_DEFAULT_ALIGN 8

namespace gfxstream {
namespace guest {
class IOStream;
}  // namespace guest
namespace vk {
class VkEncoder;
struct DescriptorPoolAllocationInfo;
struct ReifiedDescriptorSet;
struct DescriptorSetLayoutInfo;

inline const VkAllocationCallbacks* resolveDeviceAllocator(struct vk_device* device) {
    return (device && device->alloc.pfnAllocation && device->alloc.pfnFree)
               ? &device->alloc
               : vk_default_allocator();
}

template <typename T, typename... Args>
T* allocDeviceSubobject(struct vk_device* device, const VkAllocationCallbacks* pAllocator,
                        VkSystemAllocationScope scope, Args&&... args) {
    const VkAllocationCallbacks* parentAlloc = resolveDeviceAllocator(device);
    void* mem = vk_alloc2(parentAlloc, pAllocator, sizeof(T),
                          std::max<size_t>(GFXSTREAM_DEFAULT_ALIGN, alignof(T)), scope);
    if (!mem) return nullptr;
    return new (mem) T(std::forward<Args>(args)...);
}

template <typename T>
void freeDeviceSubobject(struct vk_device* device, const VkAllocationCallbacks* pAllocator,
                         T* obj) {
    if (!obj) return;
    obj->~T();
    const VkAllocationCallbacks* parentAlloc = resolveDeviceAllocator(device);
    vk_free2(parentAlloc, pAllocator, obj);
}

// Ref-counted liveness token allocated via the VkDevice's VkAllocationCallbacks
// (VK_SYSTEM_ALLOCATION_SCOPE_DEVICE) so virtualized descriptor sets can check
// whether a referenced Vulkan resource is still alive without bypassing
// VkAllocationCallbacks or throwing/aborting on OOM.
struct AliveToken {
    VkAllocationCallbacks alloc;
    std::atomic<uint32_t> refcount{1};
    std::atomic<bool> alive{true};

    explicit AliveToken(const VkAllocationCallbacks& cb) : alloc(cb) {}

    static AliveToken* create(const VkAllocationCallbacks* alloc) {
        const VkAllocationCallbacks* cb =
            (alloc && alloc->pfnAllocation && alloc->pfnFree) ? alloc : vk_default_allocator();
        void* mem = vk_alloc(cb, sizeof(AliveToken),
                             std::max<size_t>(GFXSTREAM_DEFAULT_ALIGN, alignof(AliveToken)),
                             VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
        if (!mem) return nullptr;
        return new (mem) AliveToken(*cb);
    }

    void retain() { refcount.fetch_add(1, std::memory_order_relaxed); }

    void release() {
        if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            VkAllocationCallbacks cb = alloc;
            this->~AliveToken();
            vk_free(&cb, this);
        }
    }

    void markDead() { alive.store(false, std::memory_order_release); }

    bool isAlive() const { return alive.load(std::memory_order_acquire); }
};

class AliveTokenRef {
   public:
    AliveTokenRef() = default;
    explicit AliveTokenRef(AliveToken* token) : mToken(token) {
        if (mToken) mToken->retain();
    }
    ~AliveTokenRef() { reset(); }

    AliveTokenRef(const AliveTokenRef& other) : mToken(other.mToken) {
        if (mToken) mToken->retain();
    }
    AliveTokenRef& operator=(const AliveTokenRef& other) {
        if (this == &other || mToken == other.mToken) return *this;
        if (other.mToken) other.mToken->retain();
        reset();
        mToken = other.mToken;
        return *this;
    }
    AliveTokenRef(AliveTokenRef&& other) noexcept : mToken(other.mToken) { other.mToken = nullptr; }
    AliveTokenRef& operator=(AliveTokenRef&& other) noexcept {
        if (this == &other) return *this;
        reset();
        mToken = other.mToken;
        other.mToken = nullptr;
        return *this;
    }

    void reset() {
        if (mToken) {
            mToken->release();
            mToken = nullptr;
        }
    }

    bool expired() const { return !mToken || !mToken->isAlive(); }

   private:
    AliveToken* mToken = nullptr;
};

class AliveTokenOwner {
   public:
    AliveTokenOwner() = default;
    ~AliveTokenOwner() { reset(); }

    AliveTokenOwner(const AliveTokenOwner&) = delete;
    AliveTokenOwner& operator=(const AliveTokenOwner&) = delete;
    AliveTokenOwner(AliveTokenOwner&&) = delete;
    AliveTokenOwner& operator=(AliveTokenOwner&&) = delete;

    bool init(const VkAllocationCallbacks* alloc) {
        assert(!mToken);
        mToken = AliveToken::create(alloc);
        return mToken != nullptr;
    }

    void reset() {
        if (mToken) {
            mToken->markDead();
            mToken->release();
            mToken = nullptr;
        }
    }

    AliveTokenRef weakRef() const { return AliveTokenRef(mToken); }

   private:
    AliveToken* mToken = nullptr;
};

}  // namespace vk
}  // namespace gfxstream

#define GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE_ROOT(gfxstream_type, vk_type)                   \
    extern "C" vk_type create_##gfxstream_type(const VkAllocationCallbacks* pAllocator,      \
                                               uint64_t underlying);                         \
    extern "C" void delete_##gfxstream_type(vk_type toDelete,                                \
                                            const VkAllocationCallbacks* pAllocator);        \
    extern "C" uint64_t gfxstream_type##_to_host_u64(const vk_type obj);

#define GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE(gfxstream_type, vk_type)                        \
    extern "C" vk_type create_##gfxstream_type(                                              \
        VkInstance instance, const VkAllocationCallbacks* pAllocator, uint64_t underlying);  \
    extern "C" void delete_##gfxstream_type(VkInstance instance, vk_type toDelete,           \
                                            const VkAllocationCallbacks* pAllocator);        \
    extern "C" uint64_t gfxstream_type##_to_host_u64(const vk_type obj);

#define GFXSTREAM_DECLARE_VK_OBJECT_DEVICE_ROOT(gfxstream_type, vk_type)                     \
    extern "C" vk_type create_##gfxstream_type(VkPhysicalDevice physicalDevice,              \
                                               const VkAllocationCallbacks* pAllocator,      \
                                               uint64_t underlying);                         \
    extern "C" void delete_##gfxstream_type(vk_type toDelete,                                \
                                            const VkAllocationCallbacks* pAllocator);        \
    extern "C" uint64_t gfxstream_type##_to_host_u64(const vk_type obj);

#define GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_type, vk_type)                                 \
    extern "C" vk_type create_##gfxstream_type(                                              \
        VkDevice device, const VkAllocationCallbacks* pAllocator, uint64_t underlying);      \
    extern "C" void delete_##gfxstream_type(VkDevice device, vk_type toDelete,               \
                                            const VkAllocationCallbacks* pAllocator);        \
    extern "C" uint64_t gfxstream_type##_to_host_u64(const vk_type obj);

struct gfxstream_vk_object_common {
    // The untyped host handle.
    uint64_t underlying;
};

struct gfxstream_vk_instance {
    // Must be first member.
    struct vk_instance base;

    struct gfxstream_vk_object_common common;

    uint32_t api_version;
    uint32_t highest_api_version;
    bool init_failed;
    bool runtime_initialized;
    std::vector<VkPhysicalDevice> physical_devices_vec;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_instance, base.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)
GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE_ROOT(gfxstream_vk_instance, VkInstance)

struct gfxstream_vk_physical_device {
    // Must be first member.
    struct vk_physical_device base;

    struct gfxstream_vk_object_common common;

    struct gfxstream_vk_instance* instance;
    struct wsi_device wsi_device;
    const struct vk_sync_type* sync_types[2];
    bool doImageDrmFormatModifierEmulation;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_physical_device, base.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE(gfxstream_vk_physical_device, VkPhysicalDevice)

struct gfxstream_vk_device {
    // Must be first member.
    struct vk_device base;

    struct gfxstream_vk_object_common common;

    struct vk_device_dispatch_table cmd_dispatch;
    struct gfxstream_vk_physical_device* physical_device;

    /* unique queue family indices in which to create the device queues */
    uint32_t* queue_families;
    uint32_t queue_family_count;

    VkPhysicalDevice internal_physical_device;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t api_version;
    bool runtime_initialized;

    simple_mtx_t memory_mtx;
    struct list_head device_memories;
    std::vector<std::pair<uint32_t, std::weak_ptr<gfxstream::vk::CoherentMemory>>> coherent_pools;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_device, base.base, VkDevice, VK_OBJECT_TYPE_DEVICE)
GFXSTREAM_DECLARE_VK_OBJECT_DEVICE_ROOT(gfxstream_vk_device, VkDevice)

struct gfxstream_vk_queue {
    // Must be first member.
    struct vk_queue base;

    struct gfxstream_vk_object_common common;

    struct gfxstream_vk_device* device;

    gfxstream::vk::VkEncoder* lastUsedEncoder;
    uint32_t sequenceNumber;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_queue, base.base, VkQueue, VK_OBJECT_TYPE_QUEUE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_queue, VkQueue)

struct gfxstream_vk_buffer {
    // Must be first member.
    struct vk_buffer base;

    struct gfxstream_vk_object_common common;

    VkBufferCreateInfo createInfo;
    VkExternalMemoryBufferCreateInfo externalCreateInfo;
    VkMemoryRequirements baseRequirements;
    bool external;
    bool baseRequirementsKnown;
#ifdef VK_USE_PLATFORM_FUCHSIA
    bool isSysmemBackedMemory;
#endif
    gfxstream::vk::AliveTokenOwner alive;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_buffer, base.base, VkBuffer, VK_OBJECT_TYPE_BUFFER)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_buffer, VkBuffer)

struct gfxstream_vk_command_pool {
    // Must be first member.
    struct vk_command_pool base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_command_pool, base.base, VkCommandPool,
                               VK_OBJECT_TYPE_COMMAND_POOL)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_command_pool, VkCommandPool)

struct gfxstream_vk_command_buffer {
    // Must be first member.
    struct vk_command_buffer base;

    struct gfxstream_vk_object_common common;

    gfxstream::vk::VkEncoder* lastUsedEncoder;
    uint32_t sequenceNumber;
    gfxstream::vk::VkEncoder* privateEncoder;
    gfxstream::guest::IOStream* privateStream;
    uint32_t flags;
    struct util_dynarray subObjects;
    struct util_dynarray superObjects;
    void* userPtr;
    bool isSecondary;
    VkDevice device;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_command_buffer, base.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_command_buffer, VkCommandBuffer)

struct gfxstream_vk_fence {
    // Must be first member.
    struct vk_fence base;

    struct gfxstream_vk_object_common common;

    VkDevice device;
    VkExportFenceCreateInfo exportFenceCreateInfo;
    bool external;
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    std::optional<int> syncFd;
#endif
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_fence, base.base, VkFence, VK_OBJECT_TYPE_FENCE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_fence, VkFence)

struct gfxstream_vk_semaphore {
    // Must be first member.
    struct vk_semaphore base;

    struct gfxstream_vk_object_common common;

    VkDevice device;
    zx_handle_t eventHandle;
    zx_koid_t eventKoid;
    std::optional<int> syncFd;
    bool isTimeline;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_semaphore, base.base, VkSemaphore,
                               VK_OBJECT_TYPE_SEMAPHORE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_semaphore, VkSemaphore)

struct gfxstream_vk_descriptor_pool {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;

    gfxstream::vk::DescriptorPoolAllocationInfo* allocInfo;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_descriptor_pool, VkDescriptorPool)

struct gfxstream_vk_descriptor_set {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;

    gfxstream::vk::ReifiedDescriptorSet* reified;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_descriptor_set, VkDescriptorSet)

struct gfxstream_vk_descriptor_set_layout {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
    gfxstream::vk::DescriptorSetLayoutInfo* layoutInfo;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_descriptor_set_layout, base, VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_descriptor_set_layout, VkDescriptorSetLayout)

struct gfxstream_vk_device_memory {
    // Must be first member.
    struct vk_device_memory base;
    struct gfxstream_vk_object_common common;

    struct list_head link;
    VkDevice device;
    uint8_t* ptr;
    uint64_t blobId;
    uint64_t coherentMemorySize;
    uint64_t coherentMemoryOffset;
    zx_handle_t vmoHandle;
    int importedFd;
    bool dedicated;
    bool imported;
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    AHardwareBuffer* ahw;
#endif
#if DETECT_OS_ANDROID
    gfxstream::vk::GoldfishAddressSpaceBlockPtr goldfishBlock;
#endif
    gfxstream::vk::CoherentMemoryPtr coherentMemory;
    VirtGpuResourcePtr blobPtr;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_device_memory, base.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_device_memory, VkDeviceMemory)

static inline uint8_t* gfxstream_vk_device_memory_get_ptr(VkDeviceMemory memory) {
    VK_FROM_HANDLE(gfxstream_vk_device_memory, gfxstream_memory, memory);
    if (!gfxstream_memory) return nullptr;

    return gfxstream_memory->ptr;
}

static inline VkDeviceSize gfxstream_vk_device_memory_get_ptr_size(VkDeviceMemory memory) {
    VK_FROM_HANDLE(gfxstream_vk_device_memory, gfxstream_memory, memory);
    if (!gfxstream_memory) return 0;

    return gfxstream_memory->base.size;
}

struct gfxstream_vk_image {
    // Must be first member.
    struct vk_image base;
    struct gfxstream_vk_object_common common;

    VkDevice device;
    VkImageCreateInfo createInfo;
    VkExternalMemoryImageCreateInfo externalCreateInfo;
    VkMemoryRequirements baseRequirements;
    bool external;
    bool baseRequirementsKnown;
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    bool hasExternalFormat;
    unsigned externalFourccFormat;
    std::vector<int> pendingQsriSyncFds;
    bool hasAnb;
#endif
#ifdef VK_USE_PLATFORM_FUCHSIA
    bool isSysmemBackedMemory;
#endif
#ifdef LINUX_GUEST_BUILD
    bool emulatedDrmFormatModifier;
    bool hasExplicitDrmModifier;
    VkSubresourceLayout explicitPlaneLayout;
#endif
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_image, base.base, VkImage, VK_OBJECT_TYPE_IMAGE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_image, VkImage)

struct gfxstream_vk_descriptor_update_template {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;

    void* templateStorage;
    uint32_t templateEntryCount;
    VkDescriptorUpdateTemplateEntry* templateEntries;
    uint32_t imageInfoCount;
    uint32_t bufferInfoCount;
    uint32_t bufferViewCount;
    uint32_t inlineUniformBlockCount;
    uint32_t inlineUniformBlockBufferSize;
    uint32_t* imageInfoIndices;
    uint32_t* bufferInfoIndices;
    uint32_t* bufferViewIndices;
    uint32_t* inlineUniformBlockBytesPerBlocks;
    VkPipelineBindPoint pipelineBindPoint;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_descriptor_update_template, base,
                               VkDescriptorUpdateTemplate,
                               VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_descriptor_update_template, VkDescriptorUpdateTemplate)

struct gfxstream_vk_sampler {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
    gfxstream::vk::AliveTokenOwner alive;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_sampler, base, VkSampler, VK_OBJECT_TYPE_SAMPLER)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_sampler, VkSampler)

struct gfxstream_vk_private_data_slot {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_private_data_slot, base, VkPrivateDataSlot,
                               VK_OBJECT_TYPE_PRIVATE_DATA_SLOT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_private_data_slot, VkPrivateDataSlot)

#ifdef VK_USE_PLATFORM_FUCHSIA
struct gfxstream_vk_buffer_collection_fuchsia {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;

    fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>* sysmem_collection;
    std::optional<fuchsia_sysmem::wire::BufferCollectionConstraints> constraints;
    std::optional<VkBufferCollectionPropertiesFUCHSIA> properties;
    std::vector<uint32_t> createInfoIndex;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_buffer_collection_fuchsia, base,
                               VkBufferCollectionFUCHSIA, VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_buffer_collection_fuchsia, VkBufferCollectionFUCHSIA)
#endif

struct gfxstream_vk_buffer_view {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
    gfxstream::vk::AliveTokenOwner alive;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_buffer_view, base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_buffer_view, VkBufferView)

struct gfxstream_vk_image_view {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
    gfxstream::vk::AliveTokenOwner alive;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_image_view, base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_image_view, VkImageView)

struct gfxstream_vk_shader_module {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_shader_module, base, VkShaderModule,
                               VK_OBJECT_TYPE_SHADER_MODULE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_shader_module, VkShaderModule)

struct gfxstream_vk_pipeline {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_pipeline, base, VkPipeline, VK_OBJECT_TYPE_PIPELINE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_pipeline, VkPipeline)

struct gfxstream_vk_pipeline_cache {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_pipeline_cache, base, VkPipelineCache,
                               VK_OBJECT_TYPE_PIPELINE_CACHE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_pipeline_cache, VkPipelineCache)

struct gfxstream_vk_pipeline_layout {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
    std::vector<std::vector<bool>> immutableSamplers;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_pipeline_layout, base, VkPipelineLayout,
                               VK_OBJECT_TYPE_PIPELINE_LAYOUT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_pipeline_layout, VkPipelineLayout)

struct gfxstream_vk_render_pass {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_render_pass, base, VkRenderPass,
                               VK_OBJECT_TYPE_RENDER_PASS)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_render_pass, VkRenderPass)

struct gfxstream_vk_framebuffer {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_framebuffer, base, VkFramebuffer,
                               VK_OBJECT_TYPE_FRAMEBUFFER)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_framebuffer, VkFramebuffer)

struct gfxstream_vk_event {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_event, VkEvent)

struct gfxstream_vk_query_pool {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_query_pool, base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_query_pool, VkQueryPool)

struct gfxstream_vk_sampler_ycbcr_conversion {
    // Must be first member.
    struct vk_ycbcr_conversion base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_sampler_ycbcr_conversion, base.base,
                               VkSamplerYcbcrConversion, VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_sampler_ycbcr_conversion, VkSamplerYcbcrConversion)

struct gfxstream_vk_surface_khr {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_surface_khr, base, VkSurfaceKHR,
                               VK_OBJECT_TYPE_SURFACE_KHR)
GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE(gfxstream_vk_surface_khr, VkSurfaceKHR)

struct gfxstream_vk_swapchain_khr {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_swapchain_khr, base, VkSwapchainKHR,
                               VK_OBJECT_TYPE_SWAPCHAIN_KHR)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_swapchain_khr, VkSwapchainKHR)

struct gfxstream_vk_display_khr {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_display_khr, base, VkDisplayKHR,
                               VK_OBJECT_TYPE_DISPLAY_KHR)
GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE(gfxstream_vk_display_khr, VkDisplayKHR)

struct gfxstream_vk_display_mode_khr {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_display_mode_khr, base, VkDisplayModeKHR,
                               VK_OBJECT_TYPE_DISPLAY_MODE_KHR)
GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE(gfxstream_vk_display_mode_khr, VkDisplayModeKHR)

struct gfxstream_vk_validation_cache_ext {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_validation_cache_ext, base, VkValidationCacheEXT,
                               VK_OBJECT_TYPE_VALIDATION_CACHE_EXT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_validation_cache_ext, VkValidationCacheEXT)

struct gfxstream_vk_debug_report_callback_ext {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_debug_report_callback_ext, base,
                               VkDebugReportCallbackEXT, VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT)
GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE(gfxstream_vk_debug_report_callback_ext,
                                     VkDebugReportCallbackEXT)

struct gfxstream_vk_debug_utils_messenger_ext {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_debug_utils_messenger_ext, base,
                               VkDebugUtilsMessengerEXT, VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT)
GFXSTREAM_DECLARE_VK_OBJECT_INSTANCE(gfxstream_vk_debug_utils_messenger_ext,
                                     VkDebugUtilsMessengerEXT)

struct gfxstream_vk_micromap_ext {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_micromap_ext, base, VkMicromapEXT,
                               VK_OBJECT_TYPE_MICROMAP_EXT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_micromap_ext, VkMicromapEXT)

#ifdef VK_NVX_binary_import
struct gfxstream_vk_cu_module_nvx {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_cu_module_nvx, base, VkCuModuleNVX,
                               VK_OBJECT_TYPE_CU_MODULE_NVX)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_cu_module_nvx, VkCuModuleNVX)

struct gfxstream_vk_cu_function_nvx {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_cu_function_nvx, base, VkCuFunctionNVX,
                               VK_OBJECT_TYPE_CU_FUNCTION_NVX)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_cu_function_nvx, VkCuFunctionNVX)
#endif

#ifdef VK_NVX_device_generated_commands
struct gfxstream_vk_object_table_nvx {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_object_table_nvx, base, VkObjectTableNVX,
                               VK_OBJECT_TYPE_OBJECT_TABLE_NVX)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_object_table_nvx, VkObjectTableNVX)

struct gfxstream_vk_indirect_commands_layout_nvx {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_indirect_commands_layout_nvx, base,
                               VkIndirectCommandsLayoutNVX,
                               VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NVX)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_indirect_commands_layout_nvx, VkIndirectCommandsLayoutNVX)
#endif

#ifdef VK_NV_device_generated_commands
struct gfxstream_vk_indirect_commands_layout_nv {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_indirect_commands_layout_nv, base,
                               VkIndirectCommandsLayoutNV,
                               VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_indirect_commands_layout_nv, VkIndirectCommandsLayoutNV)
#endif

#ifdef VK_NV_ray_tracing
struct gfxstream_vk_acceleration_structure_nv {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_acceleration_structure_nv, base,
                               VkAccelerationStructureNV, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_acceleration_structure_nv, VkAccelerationStructureNV)
#endif

#ifdef VK_KHR_acceleration_structure
struct gfxstream_vk_acceleration_structure_khr {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_acceleration_structure_khr, base,
                               VkAccelerationStructureKHR,
                               VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_acceleration_structure_khr, VkAccelerationStructureKHR)
#endif

VkResult gfxstream_vk_wsi_init(struct gfxstream_vk_physical_device* physical_device);

void gfxstream_vk_wsi_finish(struct gfxstream_vk_physical_device* physical_device);

bool isNoopFence(gfxstream_vk_fence* fence);

bool HasNoopSemaphores(const VkSemaphore* pSemaphores, uint32_t semaphoreCount);

bool HasNoopSemaphoreSubmitInfos(const VkSemaphoreSubmitInfo* pSemaphoreSubmitInfos,
                                 uint32_t semaphoreSubmitInfoCount);

std::vector<VkSemaphore> FilterNoopSemaphores(const VkSemaphore* pSemaphores,
                                              uint32_t semaphoreCount);

std::vector<VkSemaphoreSubmitInfo> FilterNoopSemaphoreSubmitInfos(
    const VkSemaphoreSubmitInfo* pSemaphoreSubmitInfos, uint32_t semaphoreSubmitInfoCount);

float linearChannelToSRGB(float cl);
float srgbFormatNeedsConversionForClearColor(const VkFormat& format);

#endif /* GFXSTREAM_VK_PRIVATE_H */
