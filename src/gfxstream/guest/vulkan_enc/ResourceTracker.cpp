/*
 * Copyright 2018 Google
 * SPDX-License-Identifier: MIT
 */

#include "ResourceTracker.h"

#include "CommandBufferStagingStream.h"
#include "DescriptorSetVirtualization.h"
#include "HostVisibleMemoryVirtualization.h"
#include "VkEncoder.h"
#include "gfxstream_vk_private.h"
#include "git_sha1.h"
#include "goldfish_address_space.h"
#include "goldfish_vk_private_defs.h"
#include "util/anon_file.h"
#include "util/detect_os.h"
#include "util/log.h"
#include "util/macros.h"
#include "virtio/virtio-gpu/virgl_hw.h"
#include "vk_debug_utils.h"
#include "vulkan/util/vk_format.h"
#include "vulkan/vulkan_core.h"

#ifdef VK_USE_PLATFORM_ANDROID_KHR
#include <vndk/hardware_buffer.h>

#include "vk_format_info.h"
#endif
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef HAVE_PERFETTO
#include "util/perf/u_perfetto.h"

#define GFXSTREAM_TRACE_DEFAULT_CATEGORY "gfxstream.default"

PERFETTO_DEFINE_CATEGORIES(perfetto::Category(GFXSTREAM_TRACE_DEFAULT_CATEGORY));

#endif  // HAVE_PERFETTO

#include "vk_util.h"

#if DETECT_OS_LINUX
#include <drm_fourcc.h>
#endif

#ifndef VK_USE_PLATFORM_FUCHSIA
static void zx_handle_close(zx_handle_t) {}
#endif

struct vk_struct_chain_iterator {
    VkBaseOutStructure* value;
};

template <class T>
static vk_struct_chain_iterator vk_make_chain_iterator(T* vk_struct) {
    vk_struct_chain_iterator result = {reinterpret_cast<VkBaseOutStructure*>(vk_struct)};
    return result;
}

template <class T>
static void vk_append_struct(vk_struct_chain_iterator* i, T* vk_struct) {
    VkBaseOutStructure* p = i->value;
    if (p->pNext) {
        ::abort();
    }

    p->pNext = reinterpret_cast<VkBaseOutStructure*>(vk_struct);
    vk_struct->pNext = NULL;

    *i = vk_make_chain_iterator(vk_struct);
}

template <class T>
static T vk_make_orphan_copy(const T& vk_struct) {
    T copy = vk_struct;
    copy.pNext = NULL;
    return copy;
}

template <class T>
T* vk_extract_struct(void* start, VkStructureType sType) {
    auto* curr = reinterpret_cast<VkBaseOutStructure*>(start);
    while (curr && curr->pNext) {
        if (curr->pNext->sType == sType) {
            auto* extracted = curr->pNext;
            curr->pNext = extracted->pNext;
            extracted->pNext = nullptr;
            return reinterpret_cast<T*>(extracted);
        }
        curr = curr->pNext;
    }
    return nullptr;
}

namespace gfxstream {
namespace vk {
namespace {

#ifdef HAVE_PERFETTO
uint64_t GeneratePseudoUniqueId() {
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<uint64_t> distribution(0, std::numeric_limits<uint64_t>::max());
    return distribution(generator);
}
#endif

}  // namespace

static uint32_t* sSeqnoPtr = nullptr;

// static
uint32_t ResourceTracker::streamFeatureBits = 0;
ResourceTracker::ThreadingCallbacks ResourceTracker::threadingCallbacks;

static std::atomic<bool> sStagingAlive{true};

struct StagingInfo {
    std::mutex mLock;
    std::vector<CommandBufferStagingStream*> streams;
    std::vector<VkEncoder*> encoders;
    std::atomic<bool> mAllocInitDone{false};

    void ensureAllocFree() {
        if (unlikely(!mAllocInitDone.load(std::memory_order_acquire))) {
            std::lock_guard<std::mutex> lock(mLock);
            if (!mAllocInitDone.load(std::memory_order_relaxed)) {
                mAlloc = ResourceTracker::get()->getAlloc();
                mFree = ResourceTracker::get()->getFree();
                mAllocInitDone.store(true, std::memory_order_release);
            }
        }
    }

    ~StagingInfo() {
        sStagingAlive.store(false, std::memory_order_release);
        for (auto encoder : encoders) {
            delete encoder;
        }
        for (auto stream : streams) {
            delete stream;
        }
    }

    CommandBufferStagingStream* createStream() {
        if (mAlloc && mFree) {
            return new CommandBufferStagingStream(mAlloc, mFree);
        }
        return new CommandBufferStagingStream;
    }

    void returnBatch(CommandBufferStagingStream** batchStreams, VkEncoder** batchEncoders,
                     size_t count) {
        std::lock_guard<std::mutex> lock(mLock);
        for (size_t i = 0; i < count; ++i) {
            streams.push_back(batchStreams[i]);
            encoders.push_back(batchEncoders[i]);
        }
    }

    bool takeOneGlobal(CommandBufferStagingStream** streamOut, VkEncoder** encoderOut) {
        std::lock_guard<std::mutex> lock(mLock);
        if (streams.empty()) {
            return false;
        }
        *streamOut = streams.back();
        *encoderOut = encoders.back();
        streams.pop_back();
        encoders.pop_back();
        return true;
    }

    void pushStaging(CommandBufferStagingStream* stream, VkEncoder* encoder);
    void popStaging(CommandBufferStagingStream** streamOut, VkEncoder** encoderOut);

   private:
    CommandBufferStagingStream::Alloc mAlloc = nullptr;
    CommandBufferStagingStream::Free mFree = nullptr;
};

static StagingInfo sStaging;

struct ThreadLocalStagingCache {
    static constexpr size_t kCapacity = 8;
    CommandBufferStagingStream* streams[kCapacity];
    VkEncoder* encoders[kCapacity];
    size_t count = 0;

    ~ThreadLocalStagingCache() {
        if (count == 0) return;
        if (sStagingAlive.load(std::memory_order_acquire)) {
            sStaging.returnBatch(streams, encoders, count);
        } else {
            for (size_t i = 0; i < count; ++i) {
                delete encoders[i];
                delete streams[i];
            }
        }
        count = 0;
    }
};

static thread_local ThreadLocalStagingCache sTlsStaging;

void StagingInfo::pushStaging(CommandBufferStagingStream* stream, VkEncoder* encoder) {
    stream->reset();
    if (sTlsStaging.count < ThreadLocalStagingCache::kCapacity) {
        sTlsStaging.streams[sTlsStaging.count] = stream;
        sTlsStaging.encoders[sTlsStaging.count] = encoder;
        ++sTlsStaging.count;
        return;
    }
    constexpr size_t kSpill = ThreadLocalStagingCache::kCapacity / 2;
    returnBatch(sTlsStaging.streams, sTlsStaging.encoders, kSpill);
    for (size_t i = kSpill; i < sTlsStaging.count; ++i) {
        sTlsStaging.streams[i - kSpill] = sTlsStaging.streams[i];
        sTlsStaging.encoders[i - kSpill] = sTlsStaging.encoders[i];
    }
    sTlsStaging.count -= kSpill;
    sTlsStaging.streams[sTlsStaging.count] = stream;
    sTlsStaging.encoders[sTlsStaging.count] = encoder;
    ++sTlsStaging.count;
}

void StagingInfo::popStaging(CommandBufferStagingStream** streamOut, VkEncoder** encoderOut) {
    if (sTlsStaging.count > 0) {
        --sTlsStaging.count;
        *streamOut = sTlsStaging.streams[sTlsStaging.count];
        *encoderOut = sTlsStaging.encoders[sTlsStaging.count];
        return;
    }
    if (takeOneGlobal(streamOut, encoderOut)) {
        return;
    }
    CommandBufferStagingStream* stream = createStream();
    VkEncoder* encoder = new VkEncoder(stream);
    *streamOut = stream;
    *encoderOut = encoder;
}

struct CommandBufferPendingDescriptorSets {
    std::unordered_set<VkDescriptorSet> sets;
};

static uint32_t getWaitSemaphoreCount(const VkSubmitInfo& pSubmit) {
    return pSubmit.waitSemaphoreCount;
}

static uint32_t getWaitSemaphoreCount(const VkSubmitInfo2& pSubmit) {
    return pSubmit.waitSemaphoreInfoCount;
}

static uint32_t getCommandBufferCount(const VkSubmitInfo& pSubmit) {
    return pSubmit.commandBufferCount;
}

static uint32_t getCommandBufferCount(const VkSubmitInfo2& pSubmit) {
    return pSubmit.commandBufferInfoCount;
}

static uint32_t getSignalSemaphoreCount(const VkSubmitInfo& pSubmit) {
    return pSubmit.signalSemaphoreCount;
}

static uint32_t getSignalSemaphoreCount(const VkSubmitInfo2& pSubmit) {
    return pSubmit.signalSemaphoreInfoCount;
}

static VkSemaphore getWaitSemaphore(const VkSubmitInfo& pSubmit, int i) {
    return pSubmit.pWaitSemaphores[i];
}

static VkSemaphore getWaitSemaphore(const VkSubmitInfo2& pSubmit, int i) {
    return pSubmit.pWaitSemaphoreInfos[i].semaphore;
}

static VkSemaphore getSignalSemaphore(const VkSubmitInfo& pSubmit, int i) {
    return pSubmit.pSignalSemaphores[i];
}

static VkSemaphore getSignalSemaphore(const VkSubmitInfo2& pSubmit, int i) {
    return pSubmit.pSignalSemaphoreInfos[i].semaphore;
}

static VkCommandBuffer getCommandBuffer(const VkSubmitInfo& pSubmit, int i) {
    return pSubmit.pCommandBuffers[i];
}

static VkCommandBuffer getCommandBuffer(const VkSubmitInfo2& pSubmit, int i) {
    return pSubmit.pCommandBufferInfos[i].commandBuffer;
}

static bool isHostVisible(const VkPhysicalDeviceMemoryProperties* memoryProps, uint32_t index) {
    return memoryProps->memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
}

#ifdef VK_USE_PLATFORM_FUCHSIA
inline fuchsia_sysmem::wire::BufferCollectionConstraints defaultBufferCollectionConstraints(
    size_t minSizeBytes, size_t minBufferCount, size_t maxBufferCount = 0u,
    size_t minBufferCountForCamping = 0u, size_t minBufferCountForDedicatedSlack = 0u,
    size_t minBufferCountForSharedSlack = 0u) {
    fuchsia_sysmem::wire::BufferCollectionConstraints constraints = {};
    constraints.min_buffer_count = minBufferCount;
    if (maxBufferCount > 0) {
        constraints.max_buffer_count = maxBufferCount;
    }
    if (minBufferCountForCamping) {
        constraints.min_buffer_count_for_camping = minBufferCountForCamping;
    }
    if (minBufferCountForSharedSlack) {
        constraints.min_buffer_count_for_shared_slack = minBufferCountForSharedSlack;
    }
    constraints.has_buffer_memory_constraints = true;
    fuchsia_sysmem::wire::BufferMemoryConstraints& buffer_constraints =
        constraints.buffer_memory_constraints;

    buffer_constraints.min_size_bytes = minSizeBytes;
    buffer_constraints.max_size_bytes = 0xffffffff;
    buffer_constraints.physically_contiguous_required = false;
    buffer_constraints.secure_required = false;

    // No restrictions on coherency domain or Heaps.
    buffer_constraints.ram_domain_supported = true;
    buffer_constraints.cpu_domain_supported = true;
    buffer_constraints.inaccessible_domain_supported = true;
    buffer_constraints.heap_permitted_count = 2;
    buffer_constraints.heap_permitted[0] = fuchsia_sysmem::wire::HeapType::kGoldfishDeviceLocal;
    buffer_constraints.heap_permitted[1] = fuchsia_sysmem::wire::HeapType::kGoldfishHostVisible;

    return constraints;
}

uint32_t getBufferCollectionConstraintsVulkanImageUsage(const VkImageCreateInfo* pImageInfo) {
    uint32_t usage = 0u;
    VkImageUsageFlags imageUsage = pImageInfo->usage;

#define SetUsageBit(BIT, VALUE)                                  \
    if (imageUsage & VK_IMAGE_USAGE_##BIT##_BIT) {               \
        usage |= fuchsia_sysmem::wire::kVulkanImageUsage##VALUE; \
    }

    SetUsageBit(COLOR_ATTACHMENT, ColorAttachment);
    SetUsageBit(TRANSFER_SRC, TransferSrc);
    SetUsageBit(TRANSFER_DST, TransferDst);
    SetUsageBit(SAMPLED, Sampled);

#undef SetUsageBit
    return usage;
}

uint32_t getBufferCollectionConstraintsVulkanBufferUsage(VkBufferUsageFlags bufferUsage) {
    uint32_t usage = 0u;

#define SetUsageBit(BIT, VALUE)                                   \
    if (bufferUsage & VK_BUFFER_USAGE_##BIT##_BIT) {              \
        usage |= fuchsia_sysmem::wire::kVulkanBufferUsage##VALUE; \
    }

    SetUsageBit(TRANSFER_SRC, TransferSrc);
    SetUsageBit(TRANSFER_DST, TransferDst);
    SetUsageBit(UNIFORM_TEXEL_BUFFER, UniformTexelBuffer);
    SetUsageBit(STORAGE_TEXEL_BUFFER, StorageTexelBuffer);
    SetUsageBit(UNIFORM_BUFFER, UniformBuffer);
    SetUsageBit(STORAGE_BUFFER, StorageBuffer);
    SetUsageBit(INDEX_BUFFER, IndexBuffer);
    SetUsageBit(VERTEX_BUFFER, VertexBuffer);
    SetUsageBit(INDIRECT_BUFFER, IndirectBuffer);

#undef SetUsageBit
    return usage;
}

uint32_t getBufferCollectionConstraintsVulkanBufferUsage(
    const VkBufferConstraintsInfoFUCHSIA* pBufferConstraintsInfo) {
    VkBufferUsageFlags bufferUsage = pBufferConstraintsInfo->createInfo.usage;
    return getBufferCollectionConstraintsVulkanBufferUsage(bufferUsage);
}

static fuchsia_sysmem::wire::PixelFormatType vkFormatTypeToSysmem(VkFormat format) {
    switch (format) {
        case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_SSCALED:
        case VK_FORMAT_B8G8R8A8_USCALED:
            return fuchsia_sysmem::wire::PixelFormatType::kBgra32;
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_SSCALED:
        case VK_FORMAT_R8G8B8A8_USCALED:
            return fuchsia_sysmem::wire::PixelFormatType::kR8G8B8A8;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_USCALED:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8_SSCALED:
        case VK_FORMAT_R8_SRGB:
            return fuchsia_sysmem::wire::PixelFormatType::kR8;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8_USCALED:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8_SSCALED:
        case VK_FORMAT_R8G8_SRGB:
            return fuchsia_sysmem::wire::PixelFormatType::kR8G8;
        default:
            return fuchsia_sysmem::wire::PixelFormatType::kInvalid;
    }
}

static bool vkFormatMatchesSysmemFormat(VkFormat vkFormat,
                                        fuchsia_sysmem::wire::PixelFormatType sysmemFormat) {
    switch (vkFormat) {
        case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_SSCALED:
        case VK_FORMAT_B8G8R8A8_USCALED:
            return sysmemFormat == fuchsia_sysmem::wire::PixelFormatType::kBgra32;
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_SSCALED:
        case VK_FORMAT_R8G8B8A8_USCALED:
            return sysmemFormat == fuchsia_sysmem::wire::PixelFormatType::kR8G8B8A8;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_USCALED:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8_SSCALED:
        case VK_FORMAT_R8_SRGB:
            return sysmemFormat == fuchsia_sysmem::wire::PixelFormatType::kR8 ||
                   sysmemFormat == fuchsia_sysmem::wire::PixelFormatType::kL8;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8_USCALED:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8_SSCALED:
        case VK_FORMAT_R8G8_SRGB:
            return sysmemFormat == fuchsia_sysmem::wire::PixelFormatType::kR8G8;
        default:
            return false;
    }
}

static VkFormat sysmemPixelFormatTypeToVk(fuchsia_sysmem::wire::PixelFormatType format) {
    switch (format) {
        case fuchsia_sysmem::wire::PixelFormatType::kBgra32:
            return VK_FORMAT_B8G8R8A8_SRGB;
        case fuchsia_sysmem::wire::PixelFormatType::kR8G8B8A8:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case fuchsia_sysmem::wire::PixelFormatType::kL8:
        case fuchsia_sysmem::wire::PixelFormatType::kR8:
            return VK_FORMAT_R8_UNORM;
        case fuchsia_sysmem::wire::PixelFormatType::kR8G8:
            return VK_FORMAT_R8G8_UNORM;
        default:
            return VK_FORMAT_UNDEFINED;
    }
}

// TODO(fxbug.dev/42172354): This is currently only used for allocating
// memory for dedicated external images. It should be migrated to use
// SetBufferCollectionImageConstraintsFUCHSIA.
VkResult ResourceTracker::setBufferCollectionConstraintsFUCHSIA(
    VkEncoder* enc, VkDevice device,
    fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>* collection,
    const VkImageCreateInfo* pImageInfo) {
    if (pImageInfo == nullptr) {
        mesa_loge("setBufferCollectionConstraints: pImageInfo cannot be null.");
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    const VkSysmemColorSpaceFUCHSIA kDefaultColorSpace = {
        .sType = VK_STRUCTURE_TYPE_SYSMEM_COLOR_SPACE_FUCHSIA,
        .pNext = nullptr,
        .colorSpace = static_cast<uint32_t>(fuchsia_sysmem::wire::ColorSpaceType::kSrgb),
    };

    std::vector<VkImageFormatConstraintsInfoFUCHSIA> formatInfos;
    if (pImageInfo->format == VK_FORMAT_UNDEFINED) {
        const auto kFormats = {
            VK_FORMAT_B8G8R8A8_SRGB,
            VK_FORMAT_R8G8B8A8_SRGB,
        };
        for (auto format : kFormats) {
            // shallow copy, using pNext from pImageInfo directly.
            auto createInfo = *pImageInfo;
            createInfo.format = format;
            formatInfos.push_back(VkImageFormatConstraintsInfoFUCHSIA{
                .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_CONSTRAINTS_INFO_FUCHSIA,
                .pNext = nullptr,
                .imageCreateInfo = createInfo,
                .colorSpaceCount = 1,
                .pColorSpaces = &kDefaultColorSpace,
            });
        }
    } else {
        formatInfos.push_back(VkImageFormatConstraintsInfoFUCHSIA{
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_CONSTRAINTS_INFO_FUCHSIA,
            .pNext = nullptr,
            .imageCreateInfo = *pImageInfo,
            .colorSpaceCount = 1,
            .pColorSpaces = &kDefaultColorSpace,
        });
    }

    VkImageConstraintsInfoFUCHSIA imageConstraints = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CONSTRAINTS_INFO_FUCHSIA,
        .pNext = nullptr,
        .formatConstraintsCount = static_cast<uint32_t>(formatInfos.size()),
        .pFormatConstraints = formatInfos.data(),
        .bufferCollectionConstraints =
            VkBufferCollectionConstraintsInfoFUCHSIA{
                .sType = VK_STRUCTURE_TYPE_BUFFER_COLLECTION_CONSTRAINTS_INFO_FUCHSIA,
                .pNext = nullptr,
                .minBufferCount = 1,
                .maxBufferCount = 0,
                .minBufferCountForCamping = 0,
                .minBufferCountForDedicatedSlack = 0,
                .minBufferCountForSharedSlack = 0,
            },
        .flags = 0u,
    };

    return setBufferCollectionImageConstraintsFUCHSIA(enc, device, collection, VK_NULL_HANDLE,
                                                      &imageConstraints);
}

VkResult addImageBufferCollectionConstraintsFUCHSIA(
    VkEncoder* enc, VkDevice device, VkPhysicalDevice physicalDevice,
    const VkImageFormatConstraintsInfoFUCHSIA* formatConstraints,  // always non-zero
    VkImageTiling tiling, fuchsia_sysmem::wire::BufferCollectionConstraints* constraints) {
    // First check if the format, tiling and usage is supported on host.
    VkImageFormatProperties imageFormatProperties;
    auto createInfo = &formatConstraints->imageCreateInfo;
    auto result = enc->vkGetPhysicalDeviceImageFormatProperties(
        physicalDevice, createInfo->format, createInfo->imageType, tiling, createInfo->usage,
        createInfo->flags, &imageFormatProperties, true /* do lock */);
    if (result != VK_SUCCESS) {
        mesa_logd(
            "%s: Image format (%u) type (%u) tiling (%u) "
            "usage (%u) flags (%u) not supported by physical "
            "device",
            __func__, static_cast<uint32_t>(createInfo->format),
            static_cast<uint32_t>(createInfo->imageType), static_cast<uint32_t>(tiling),
            static_cast<uint32_t>(createInfo->usage), static_cast<uint32_t>(createInfo->flags));
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    // Check if format constraints contains unsupported format features.
    {
        VkFormatProperties formatProperties;
        enc->vkGetPhysicalDeviceFormatProperties(physicalDevice, createInfo->format,
                                                 &formatProperties, true /* do lock */);

        auto supportedFeatures = (tiling == VK_IMAGE_TILING_LINEAR)
                                     ? formatProperties.linearTilingFeatures
                                     : formatProperties.optimalTilingFeatures;
        auto requiredFeatures = formatConstraints->requiredFormatFeatures;
        if ((~supportedFeatures) & requiredFeatures) {
            mesa_logd(
                "%s: Host device support features for %s tiling: %08x, "
                "required features: %08x, feature bits %08x missing",
                __func__, tiling == VK_IMAGE_TILING_LINEAR ? "LINEAR" : "OPTIMAL",
                static_cast<uint32_t>(requiredFeatures), static_cast<uint32_t>(supportedFeatures),
                static_cast<uint32_t>((~supportedFeatures) & requiredFeatures));
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
    }

    fuchsia_sysmem::wire::ImageFormatConstraints imageConstraints;
    if (formatConstraints->sysmemPixelFormat != 0) {
        auto pixelFormat = static_cast<fuchsia_sysmem::wire::PixelFormatType>(
            formatConstraints->sysmemPixelFormat);
        if (createInfo->format != VK_FORMAT_UNDEFINED &&
            !vkFormatMatchesSysmemFormat(createInfo->format, pixelFormat)) {
            mesa_logd("%s: VkFormat %u doesn't match sysmem pixelFormat %lu", __func__,
                      static_cast<uint32_t>(createInfo->format),
                      formatConstraints->sysmemPixelFormat);
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        imageConstraints.pixel_format.type = pixelFormat;
    } else {
        auto pixel_format = vkFormatTypeToSysmem(createInfo->format);
        if (pixel_format == fuchsia_sysmem::wire::PixelFormatType::kInvalid) {
            mesa_logd("%s: Unsupported VkFormat %u", __func__,
                      static_cast<uint32_t>(createInfo->format));
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        imageConstraints.pixel_format.type = pixel_format;
    }

    imageConstraints.color_spaces_count = formatConstraints->colorSpaceCount;
    for (size_t i = 0; i < formatConstraints->colorSpaceCount; i++) {
        imageConstraints.color_space[0].type = static_cast<fuchsia_sysmem::wire::ColorSpaceType>(
            formatConstraints->pColorSpaces[i].colorSpace);
    }

    // Get row alignment from host GPU.
    VkDeviceSize offset = 0;
    VkDeviceSize rowPitchAlignment = 1u;

    if (tiling == VK_IMAGE_TILING_LINEAR) {
        VkImageCreateInfo createInfoDup = *createInfo;
        createInfoDup.pNext = nullptr;
        enc->vkGetLinearImageLayout2GOOGLE(device, &createInfoDup, &offset, &rowPitchAlignment,
                                           true /* do lock */);
        mesa_logd(
            "vkGetLinearImageLayout2GOOGLE: format %d offset %lu "
            "rowPitchAlignment = %lu",
            (int)createInfo->format, offset, rowPitchAlignment);
    }

    imageConstraints.min_coded_width = createInfo->extent.width;
    imageConstraints.max_coded_width = 0xfffffff;
    imageConstraints.min_coded_height = createInfo->extent.height;
    imageConstraints.max_coded_height = 0xffffffff;
    // The min_bytes_per_row can be calculated by sysmem using
    // |min_coded_width|, |bytes_per_row_divisor| and color format.
    imageConstraints.min_bytes_per_row = 0;
    imageConstraints.max_bytes_per_row = 0xffffffff;
    imageConstraints.max_coded_width_times_coded_height = 0xffffffff;

    imageConstraints.layers = 1;
    imageConstraints.coded_width_divisor = 1;
    imageConstraints.coded_height_divisor = 1;
    imageConstraints.bytes_per_row_divisor = rowPitchAlignment;
    imageConstraints.start_offset_divisor = 1;
    imageConstraints.display_width_divisor = 1;
    imageConstraints.display_height_divisor = 1;
    imageConstraints.pixel_format.has_format_modifier = true;
    imageConstraints.pixel_format.format_modifier.value =
        (tiling == VK_IMAGE_TILING_LINEAR)
            ? fuchsia_sysmem::wire::kFormatModifierLinear
            : fuchsia_sysmem::wire::kFormatModifierGoogleGoldfishOptimal;

    constraints->image_format_constraints[constraints->image_format_constraints_count++] =
        imageConstraints;
    return VK_SUCCESS;
}

SetBufferCollectionBufferConstraintsResult setBufferCollectionBufferConstraintsImpl(
    fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>* pCollection,
    const VkBufferConstraintsInfoFUCHSIA* pBufferConstraintsInfo) {
    const auto& collection = *pCollection;
    if (pBufferConstraintsInfo == nullptr) {
        mesa_loge(
            "setBufferCollectionBufferConstraints: "
            "pBufferConstraintsInfo cannot be null.");
        return {VK_ERROR_OUT_OF_DEVICE_MEMORY};
    }

    fuchsia_sysmem::wire::BufferCollectionConstraints constraints =
        defaultBufferCollectionConstraints(
            /* min_size_bytes */ pBufferConstraintsInfo->createInfo.size,
            /* buffer_count */ pBufferConstraintsInfo->bufferCollectionConstraints.minBufferCount);
    constraints.usage.vulkan =
        getBufferCollectionConstraintsVulkanBufferUsage(pBufferConstraintsInfo);

    constexpr uint32_t kVulkanPriority = 5;
    const char kName[] = "GoldfishBufferSysmemShared";
    collection->SetName(kVulkanPriority, fidl::StringView(kName));

    auto result = collection->SetConstraints(true, constraints);
    if (!result.ok()) {
        mesa_loge("setBufferCollectionConstraints: SetConstraints failed: %d", result.status());
        return {VK_ERROR_OUT_OF_DEVICE_MEMORY};
    }

    return {VK_SUCCESS, constraints};
}
#endif

#ifdef VK_USE_PLATFORM_ANDROID_KHR
uint64_t ResourceTracker::getAHardwareBufferId(AHardwareBuffer* ahw) {
    uint64_t id = 0;
    mGralloc->getId(ahw, &id);
    return id;
}
#endif

static void transformExternalResourceMemoryDedicatedRequirementsForGuest(
    VkMemoryDedicatedRequirements* dedicatedReqs) {
    dedicatedReqs->prefersDedicatedAllocation = VK_TRUE;
    dedicatedReqs->requiresDedicatedAllocation = VK_TRUE;
}

#if defined(LINUX_GUEST_BUILD)
static void fillEmulatedLinearSubresourceLayout(const VkImageCreateInfo& createInfo,
                                                VkSubresourceLayout* pLayout) {
    uint32_t bpp = vk_format_get_blocksize(createInfo.format);
    if (bpp == 0) bpp = 4;
    if (pLayout->rowPitch == 0) {
        pLayout->rowPitch = ALIGN_POT(createInfo.extent.width * bpp, 256);
    }
    pLayout->size = (VkDeviceSize)pLayout->rowPitch * createInfo.extent.height;
    pLayout->depthPitch = pLayout->size;
    pLayout->arrayPitch = pLayout->size;
}
#endif

void ResourceTracker::transformImageMemoryRequirementsForGuest(VkImage image,
                                                               VkMemoryRequirements* reqs) {
#ifdef VK_USE_PLATFORM_FUCHSIA
    VK_FROM_HANDLE(gfxstream_vk_image, info, image);
    if (!info) return;
    if (info->isSysmemBackedMemory) {
        auto width = info->createInfo.extent.width;
        auto height = info->createInfo.extent.height;
        reqs->size = width * height * 4;
    }
#elif defined(LINUX_GUEST_BUILD)
    VK_FROM_HANDLE(gfxstream_vk_image, info, image);
    if (!info) return;
    if (info->emulatedDrmFormatModifier && info->external &&
        (info->externalCreateInfo.handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)) {
        VkSubresourceLayout layout = {};
        if (info->hasExplicitDrmModifier && info->explicitPlaneLayout.size > 0) {
            layout = info->explicitPlaneLayout;
        } else {
            fillEmulatedLinearSubresourceLayout(info->createInfo, &layout);
        }
        VkDeviceSize minSize = ALIGN_POT(layout.size, 4096);
        if (reqs->size < minSize) {
            reqs->size = minSize;
        }
    }
#else
    (void)image;
    (void)reqs;
#endif
}

CoherentMemoryPtr ResourceTracker::freeCoherentMemory(VkDevice device, VkDeviceMemory memory,
                                                      const VkAllocationCallbacks* pAllocator) {
    VK_FROM_HANDLE(gfxstream_vk_device_memory, info, memory);
    if (!info || !info->coherentMemory) {
        return nullptr;
    }

    VkDevice effectiveDevice = (device != VK_NULL_HANDLE) ? device : info->device;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, effectiveDevice);

    CoherentMemoryPtr coherentMemory;
    bool isSuballocatedAlias = false;

    if (dev) {
        simple_mtx_lock(&dev->memory_mtx);
        coherentMemory = std::move(info->coherentMemory);
        isSuballocatedAlias = (coherentMemory->getDeviceMemory() != memory);
        if (!isSuballocatedAlias) {
            coherentMemory->setAllocator(pAllocator);
        }
        if (info->ptr) {
            coherentMemory->release(info->coherentMemoryOffset);
            info->ptr = nullptr;
        }
        simple_mtx_unlock(&dev->memory_mtx);
    } else {
        std::lock_guard<std::recursive_mutex> lock(mLock);
        coherentMemory = std::move(info->coherentMemory);
        isSuballocatedAlias = (coherentMemory->getDeviceMemory() != memory);
        if (!isSuballocatedAlias) {
            coherentMemory->setAllocator(pAllocator);
        }
        if (info->ptr) {
            coherentMemory->release(info->coherentMemoryOffset);
            info->ptr = nullptr;
        }
    }

    if (isSuballocatedAlias) {
        unregister_VkDeviceMemory(memory);
        delete_gfxstream_vk_device_memory(effectiveDevice, memory, pAllocator);
    }

    return coherentMemory;
}

void ResourceTracker::EmitGuestAndHostTraceMarker(VkEncoder* encoder) {
#ifdef HAVE_PERFETTO
    const uint64_t flowId = GeneratePseudoUniqueId();

    TRACE_EVENT_INSTANT(GFXSTREAM_TRACE_DEFAULT_CATEGORY, "vkTraceAsyncGOOGLE",
                        perfetto::Flow::Global(flowId), "flow id", flowId);

    if (mCaps.vulkanCapset.hasTraceAsyncCommand) {
        encoder->vkTraceAsyncGOOGLE(flowId, true /* do lock */);
    }
#else
    (void)encoder;
#endif  // HAVE_PERFETTO
}

static VkResult acquireSync(uint64_t syncId, int64_t& osHandle) {
    struct VirtGpuExecBuffer exec = {};
    struct gfxstreamAcquireSync acquireSync = {};
    VirtGpuDevice* instance = VirtGpuDevice::getInstance();

    acquireSync.hdr.opCode = GFXSTREAM_ACQUIRE_SYNC;
    acquireSync.syncId = syncId;

    exec.command = static_cast<void*>(&acquireSync);
    exec.command_size = sizeof(acquireSync);
    exec.flags = kFenceOut | kRingIdx | kShareableOut;

    if (instance->execBuffer(exec, nullptr)) return VK_ERROR_OUT_OF_HOST_MEMORY;

    osHandle = exec.handle.osHandle;
    return VK_SUCCESS;
}

static VkResult createFence(VkDevice device, uint64_t hostFenceHandle, int64_t& osHandle) {
    struct VirtGpuExecBuffer exec = {};
    struct gfxstreamCreateExportSyncVK exportSync = {};
    VirtGpuDevice* instance = VirtGpuDevice::getInstance();

    uint64_t hostDeviceHandle = gfxstream_vk_device_to_host_u64(device);

    exportSync.hdr.opCode = GFXSTREAM_CREATE_EXPORT_SYNC_VK;
    exportSync.deviceHandleLo = (uint32_t)hostDeviceHandle;
    exportSync.deviceHandleHi = (uint32_t)(hostDeviceHandle >> 32);
    exportSync.fenceHandleLo = (uint32_t)hostFenceHandle;
    exportSync.fenceHandleHi = (uint32_t)(hostFenceHandle >> 32);

    exec.command = static_cast<void*>(&exportSync);
    exec.command_size = sizeof(exportSync);
    exec.flags = kFenceOut | kRingIdx;
    if (instance->execBuffer(exec, nullptr)) return VK_ERROR_OUT_OF_HOST_MEMORY;

    osHandle = exec.handle.osHandle;
    return VK_SUCCESS;
}

static bool hasAnyDirtyPendingDescriptorSets(const VkCommandBuffer* workingSet,
                                             size_t workingSetCount) {
    for (size_t i = 0; i < workingSetCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, workingSet[i]);
        if (!cb) continue;

        if (cb->userPtr) {
            auto* pendingDescriptorSets =
                static_cast<CommandBufferPendingDescriptorSets*>(cb->userPtr);
            for (VkDescriptorSet set : pendingDescriptorSets->sets) {
                VK_FROM_HANDLE(gfxstream_vk_descriptor_set, ds, set);
                if (ds && ds->reified &&
                    (ds->reified->hasPendingWrites.load(std::memory_order_acquire) ||
                     ds->reified->allocationPending.load(std::memory_order_acquire))) {
                    return true;
                }
            }
        }

        unsigned subCount = util_dynarray_num_elements(&cb->subObjects, VkCommandBuffer);
        if (subCount > 0 &&
            hasAnyDirtyPendingDescriptorSets(
                static_cast<const VkCommandBuffer*>(cb->subObjects.data), subCount)) {
            return true;
        }
    }
    return false;
}

static void collectDirtyPendingDescriptorSetsBottomUp(const VkCommandBuffer* workingSet,
                                                      size_t workingSetCount,
                                                      std::unordered_set<VkDescriptorSet>& allDs) {
    if (workingSetCount == 0) return;

    for (size_t i = 0; i < workingSetCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, workingSet[i]);
        if (!cb) continue;
        unsigned subCount = util_dynarray_num_elements(&cb->subObjects, VkCommandBuffer);
        if (subCount > 0) {
            collectDirtyPendingDescriptorSetsBottomUp(
                static_cast<const VkCommandBuffer*>(cb->subObjects.data), subCount, allDs);
        }
    }

    for (size_t i = 0; i < workingSetCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, workingSet[i]);
        if (!cb || !cb->userPtr) {
            continue;
        }

        CommandBufferPendingDescriptorSets* pendingDescriptorSets =
            static_cast<CommandBufferPendingDescriptorSets*>(cb->userPtr);

        for (VkDescriptorSet set : pendingDescriptorSets->sets) {
            VK_FROM_HANDLE(gfxstream_vk_descriptor_set, ds, set);
            if (ds && ds->reified &&
                (ds->reified->hasPendingWrites.load(std::memory_order_acquire) ||
                 ds->reified->allocationPending.load(std::memory_order_acquire))) {
                allDs.insert(set);
            }
        }
    }
}

static void commitDescriptorSetUpdates(void* context, VkQueue queue,
                                       const std::unordered_set<VkDescriptorSet>& sets) {
    if (sets.empty()) return;

    VkEncoder* enc = (VkEncoder*)context;

    std::vector<ReifiedDescriptorSet*> sortedReified;
    sortedReified.reserve(sets.size());
    for (auto set : sets) {
        sortedReified.push_back(gfxstream_vk_descriptor_set_from_handle(set)->reified);
    }
    std::sort(sortedReified.begin(), sortedReified.end());
    sortedReified.erase(std::unique(sortedReified.begin(), sortedReified.end()),
                        sortedReified.end());
    std::vector<std::unique_lock<std::recursive_mutex>> reifiedLocks;
    reifiedLocks.reserve(sortedReified.size());
    for (auto* r : sortedReified) {
        reifiedLocks.emplace_back(r->lock);
    }

    std::unordered_map<VkDescriptorPool, uint32_t> poolSet;
    std::vector<VkDescriptorPool> pools;
    std::vector<VkDescriptorSetLayout> setLayouts;
    std::vector<uint64_t> poolIds;
    std::vector<uint32_t> descriptorSetWhichPool;
    std::vector<uint32_t> pendingAllocations;
    std::vector<uint32_t> writeStartingIndices;
    std::vector<VkWriteDescriptorSet> writesForHost;

    uint32_t poolIndex = 0;
    uint32_t currentWriteIndex = 0;
    bool hasPendingAllocations = false;
    for (auto set : sets) {
        ReifiedDescriptorSet* reified = gfxstream_vk_descriptor_set_from_handle(set)->reified;
        VkDescriptorPool pool = reified->pool;
        VkDescriptorSetLayout setLayout = reified->setLayout;

        auto it = poolSet.find(pool);
        if (it == poolSet.end()) {
            poolSet[pool] = poolIndex;
            descriptorSetWhichPool.push_back(poolIndex);
            pools.push_back(pool);
            ++poolIndex;
        } else {
            uint32_t savedPoolIndex = it->second;
            descriptorSetWhichPool.push_back(savedPoolIndex);
        }

        poolIds.push_back(reified->poolId);
        setLayouts.push_back(setLayout);
        if (reified->allocationPending) {
            hasPendingAllocations = true;
        }
        pendingAllocations.push_back(reified->allocationPending ? 1 : 0);
        writeStartingIndices.push_back(currentWriteIndex);

        auto& writes = reified->allWrites;

        for (size_t i = 0; i < writes.bindings.size(); ++i) {
            uint32_t binding = i;
            auto& bindingWrite = writes.bindings[i];

            if (auto* elementWrites =
                    std::get_if<DescriptorBindingSparseWrite>(&bindingWrite.elements)) {
                for (uint32_t j : elementWrites->dirtyIndices) {
                    auto elemIt = elementWrites->elements.find(j);
                    if (elemIt == elementWrites->elements.end()) continue;
                    auto& write = elemIt->second;
                    if (!write.pendingWrite) continue;
                    write.pendingWrite = false;

                    if (write.type == DescriptorWriteType::Empty) {
                        elementWrites->elements.erase(elemIt);
                        continue;
                    }

                    uint32_t dstArrayElement = 0;
                    uint32_t descriptorCount = 1;
                    const void* pNext = nullptr;

                    VkDescriptorImageInfo* imageInfo = nullptr;
                    VkDescriptorBufferInfo* bufferInfo = nullptr;
                    VkBufferView* bufferView = nullptr;

                    // VK_EXT_descriptor_indexing allows an application to update a descriptor
                    // set and then destroy the corresponding descriptor resource before
                    // submitting the command buffer if the descriptor is unused.
                    //
                    // VK_KHR_robustness2 allows an application to explicitly set VK_NULL_HANDLE
                    // for unused resources in descriptors.
                    //
                    // Gfxstream should send writes with VK_NULL_HANDLE (the `!= VK_NULL_HANDLE`
                    // checks below) but should not send writes for destroyed resources (the
                    // `alive.expired()` checks below) to support both without trying to
                    // dereference deleted handles on the host.
                    switch (write.type) {
                        case DescriptorWriteType::Empty:
                            break;
                        case DescriptorWriteType::ImageInfo: {
                            bool resourcesAlive = true;
                            if (bindingWrite.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER &&
                                write.imageInfo.imageView != VK_NULL_HANDLE) {
                                resourcesAlive &= !write.alive.expired();
                            }
                            if ((bindingWrite.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                                 bindingWrite.descriptorType ==
                                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) &&
                                (binding >= reified->bindingIsImmutableSampler.size() ||
                                 !reified->bindingIsImmutableSampler[binding]) &&
                                write.imageInfo.sampler != VK_NULL_HANDLE) {
                                resourcesAlive &= !write.samplerAlive.expired();
                            }
                            if (!resourcesAlive) {
                                elementWrites->elements.erase(elemIt);
                                continue;
                            }
                            dstArrayElement = j;
                            imageInfo = &write.imageInfo;
                            break;
                        }
                        case DescriptorWriteType::BufferInfo:
                            if (write.bufferInfo.buffer != VK_NULL_HANDLE &&
                                write.alive.expired()) {
                                elementWrites->elements.erase(elemIt);
                                continue;
                            }
                            dstArrayElement = j;
                            bufferInfo = &write.bufferInfo;
                            break;
                        case DescriptorWriteType::BufferView:
                            if (write.bufferView != VK_NULL_HANDLE && write.alive.expired()) {
                                elementWrites->elements.erase(elemIt);
                                continue;
                            }
                            dstArrayElement = j;
                            bufferView = &write.bufferView;
                            break;
                        case DescriptorWriteType::AccelerationStructure:
                            // TODO
                            mesa_loge(
                                "Encountered pending acceleration structure "
                                "desc write, abort (NYI)\n");
                            abort();
                        default:
                            break;
                    }

                    // TODO: Combine multiple writes into one VkWriteDescriptorSet.
                    VkWriteDescriptorSet forHost = {
                        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        pNext,
                        set,
                        binding,
                        dstArrayElement,
                        descriptorCount,
                        bindingWrite.descriptorType,
                        imageInfo,
                        bufferInfo,
                        bufferView,
                    };

                    writesForHost.push_back(forHost);
                    ++currentWriteIndex;
                }
                elementWrites->dirtyIndices.clear();
            } else if (auto* blockWrite =
                           std::get_if<DescriptorInlineUniformBlockWrite>(&bindingWrite.elements)) {
                if (!blockWrite->pendingWrite) continue;

                uint32_t descriptorCount = static_cast<uint32_t>(blockWrite->bytes.size());
                blockWrite->writeInfo.sType =
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK;
                blockWrite->writeInfo.pNext = nullptr;
                blockWrite->writeInfo.dataSize = descriptorCount;
                blockWrite->writeInfo.pData = blockWrite->bytes.data();

                VkWriteDescriptorSet forHost = {
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    &blockWrite->writeInfo,
                    set,
                    binding,
                    0,  // dstArrayElement
                    descriptorCount,
                    bindingWrite.descriptorType,
                    nullptr,
                    nullptr,
                    nullptr,
                };

                writesForHost.push_back(forHost);
                ++currentWriteIndex;

                blockWrite->pendingWrite = false;
            }
        }
        reified->hasPendingWrites.store(false, std::memory_order_release);
    }

    // Skip out if there's nothing to VkWriteDescriptorSet or allocate on the host.
    if (writesForHost.empty() && !hasPendingAllocations) {
        return;
    }

    enc->vkQueueCommitDescriptorSetUpdatesGOOGLE(
        queue, (uint32_t)pools.size(), pools.data(), (uint32_t)sets.size(), setLayouts.data(),
        poolIds.data(), descriptorSetWhichPool.data(), pendingAllocations.data(),
        writeStartingIndices.data(), (uint32_t)writesForHost.size(), writesForHost.data(),
        false /* no lock */);

    // If we got here, then we definitely serviced the allocations.
    for (auto set : sets) {
        ReifiedDescriptorSet* reified = gfxstream_vk_descriptor_set_from_handle(set)->reified;
        reified->allocationPending.store(false, std::memory_order_release);
    }
}

uint32_t ResourceTracker::syncEncodersForCommandBuffer(VkCommandBuffer commandBuffer,
                                                       VkEncoder* currentEncoder) {
    VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, commandBuffer);
    if (!cb) return 0;

    auto lastEncoder = cb->lastUsedEncoder;

    if (lastEncoder == currentEncoder) return 0;

    currentEncoder->incRef();

    cb->lastUsedEncoder = currentEncoder;

    if (!lastEncoder) return 0;

    auto oldSeq = cb->sequenceNumber;
    cb->sequenceNumber += 2;
    lastEncoder->vkCommandBufferHostSyncGOOGLE(commandBuffer, false, oldSeq + 1,
                                               true /* do lock */);
    lastEncoder->flush();
    currentEncoder->vkCommandBufferHostSyncGOOGLE(commandBuffer, true, oldSeq + 2,
                                                  true /* do lock */);

    if (lastEncoder->decRef()) {
        cb->lastUsedEncoder = nullptr;
    }
    return 0;
}

static void addPendingDescriptorSets(VkCommandBuffer commandBuffer, uint32_t descriptorSetCount,
                                     const VkDescriptorSet* pDescriptorSets) {
    VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, commandBuffer);

    if (!cb->userPtr) {
        CommandBufferPendingDescriptorSets* newPendingSets = new CommandBufferPendingDescriptorSets;
        cb->userPtr = newPendingSets;
    }

    CommandBufferPendingDescriptorSets* pendingSets =
        static_cast<CommandBufferPendingDescriptorSets*>(cb->userPtr);

    if (!pDescriptorSets) return;

    for (uint32_t i = 0; i < descriptorSetCount; ++i) {
        if (pDescriptorSets[i]) {
            pendingSets->sets.insert(pDescriptorSets[i]);
        }
    }
}

static void decDescriptorSetLayoutRef(void* context, VkDevice device,
                                      VkDescriptorSetLayout descriptorSetLayout,
                                      const VkAllocationCallbacks* pAllocator) {
    if (!descriptorSetLayout) return;

    VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, setLayout, descriptorSetLayout);
    if (!setLayout || !setLayout->layoutInfo) return;

    if (setLayout->layoutInfo->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::optional<VkAllocationCallbacks> savedAlloc = setLayout->layoutInfo->destroyAllocator;
        const VkAllocationCallbacks* effectiveAlloc =
            pAllocator ? pAllocator : (savedAlloc ? &(*savedAlloc) : nullptr);
        VkEncoder* enc = (VkEncoder*)context;
        enc->vkDestroyDescriptorSetLayout(device, descriptorSetLayout, effectiveAlloc,
                                          true /* do lock */);
    }
}

void ResourceTracker::ensureSyncDeviceFd() {
#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
    if (mSyncDeviceFd >= 0) return;
    mSyncDeviceFd = goldfish_sync_open();
    if (mSyncDeviceFd >= 0) {
        mesa_logd("%s: created sync device for current Vulkan process: %d\n", __func__,
                  mSyncDeviceFd);
    } else {
        mesa_logd("%s: failed to create sync device for current Vulkan process\n", __func__);
    }
#endif
}

void ResourceTracker::unregister_VkCommandPool(VkCommandPool pool) {
    if (!pool) return;
    clearCommandPool(pool);
}

void ResourceTracker::unregister_VkCommandBuffer(VkCommandBuffer commandBuffer) {
    resetCommandBufferStagingInfo(commandBuffer, true, true);

    VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, commandBuffer);
    if (!cb) return;
    if (cb->lastUsedEncoder) {
        cb->lastUsedEncoder->decRef();
        cb->lastUsedEncoder = nullptr;
    }

    if (cb->userPtr) {
        CommandBufferPendingDescriptorSets* pendingSets =
            static_cast<CommandBufferPendingDescriptorSets*>(cb->userPtr);
        delete pendingSets;
        cb->userPtr = nullptr;
    }
}

void ResourceTracker::unregister_VkQueue(VkQueue queue) {
    VK_FROM_HANDLE(gfxstream_vk_queue, q, queue);
    if (!q) return;
    if (q->lastUsedEncoder) {
        q->lastUsedEncoder->decRef();
        q->lastUsedEncoder = nullptr;
    }
}

void ResourceTracker::unregister_VkDeviceMemory(VkDeviceMemory mem) {
    VK_FROM_HANDLE(gfxstream_vk_device_memory, memInfo, mem);
    if (!memInfo) return;

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (memInfo->ahw) {
        mGralloc->release(memInfo->ahw);
        memInfo->ahw = nullptr;
    }
#endif

    if (memInfo->vmoHandle != ZX_HANDLE_INVALID) {
        zx_handle_close(memInfo->vmoHandle);
        memInfo->vmoHandle = ZX_HANDLE_INVALID;
    }
}

void ResourceTracker::unregister_VkSemaphore(VkSemaphore sem) {
    VK_FROM_HANDLE(gfxstream_vk_semaphore, semInfo, sem);
    if (!semInfo) return;

    if (semInfo->eventHandle != ZX_HANDLE_INVALID) {
        zx_handle_close(semInfo->eventHandle);
        semInfo->eventHandle = ZX_HANDLE_INVALID;
    }

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    if (semInfo->syncFd.value_or(-1) >= 0) {
        mSyncHelper->close(semInfo->syncFd.value());
        semInfo->syncFd.reset();
    }
#endif
}

void ResourceTracker::unregister_VkFence(VkFence fence) {
    VK_FROM_HANDLE(gfxstream_vk_fence, fenceInfo, fence);
    if (!fenceInfo) return;

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    if (fenceInfo->syncFd && *fenceInfo->syncFd >= 0) {
        mSyncHelper->close(*fenceInfo->syncFd);
        fenceInfo->syncFd.reset();
    }
#endif
}

void ResourceTracker::freeDescriptorSetsIfHostAllocated(VkEncoder* enc, VkDevice device,
                                                        uint32_t descriptorSetCount,
                                                        const VkDescriptorSet* sets) {
    std::vector<VkDescriptorSet> hostAllocatedSets;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < descriptorSetCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_descriptor_set, ds, sets[i]);
        if (ds->reified->allocationPending) {
            delete_gfxstream_vk_descriptor_set(device, sets[i], nullptr);
        } else {
            pool = ds->reified->pool;
            hostAllocatedSets.push_back(sets[i]);
        }
    }
    if (!hostAllocatedSets.empty()) {
        enc->vkFreeDescriptorSets(device, pool, (uint32_t)hostAllocatedSets.size(),
                                  hostAllocatedSets.data(), true /* do lock */);
    }
}

void ResourceTracker::clearDescriptorPoolAndUnregisterDescriptorSets(void* context, VkDevice device,
                                                                     VkDescriptorPool pool) {
    VK_FROM_HANDLE(gfxstream_vk_descriptor_pool, dp, pool);
    if (!dp || !dp->allocInfo) return;

    std::vector<VkDescriptorSet> toClear;
    simple_mtx_lock(&dp->allocInfo->poolLock);
    toClear = clearDescriptorPool(pool, mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate);
    simple_mtx_unlock(&dp->allocInfo->poolLock);

    for (auto set : toClear) {
        if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
            VkDescriptorSetLayout setLayout =
                gfxstream_vk_descriptor_set_from_handle(set)->reified->setLayout;
            decDescriptorSetLayoutRef(context, device, setLayout, nullptr);
        }
        delete_gfxstream_vk_descriptor_set(device, set, nullptr);
    }
}

void ResourceTracker::deviceMemoryTransform_fromhost(VkDeviceMemory* memory, uint32_t memoryCount,
                                                     VkDeviceSize* offset, uint32_t offsetCount,
                                                     VkDeviceSize* size, uint32_t sizeCount,
                                                     uint32_t* typeIndex, uint32_t typeIndexCount,
                                                     uint32_t* typeBits, uint32_t typeBitsCount) {
    (void)memory;
    (void)memoryCount;
    (void)offset;
    (void)offsetCount;
    (void)size;
    (void)sizeCount;
    (void)typeIndex;
    (void)typeIndexCount;
    (void)typeBits;
    (void)typeBitsCount;
}

void ResourceTracker::transformImpl_VkExternalMemoryProperties_fromhost(
    VkExternalMemoryProperties* pProperties, uint32_t) {
    VkExternalMemoryHandleTypeFlags supportedHandleType = 0u;
#ifdef VK_USE_PLATFORM_FUCHSIA
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA;
#endif  // VK_USE_PLATFORM_FUCHSIA
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
#endif  // VK_USE_PLATFORM_ANDROID_KHR
#ifdef LINUX_GUEST_BUILD
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif  // LINUX_GUEST_BUILD
    if (supportedHandleType) {
        pProperties->compatibleHandleTypes &= supportedHandleType;
        pProperties->exportFromImportedHandleTypes &= supportedHandleType;
    }
}

bool ResourceTracker::isValidMemoryRange(const VkMappedMemoryRange& range) {
    VK_FROM_HANDLE(gfxstream_vk_device_memory, info, range.memory);
    if (!info || !info->ptr) return false;

    VkDeviceSize offset = range.offset;
    VkDeviceSize size = range.size;

    if (size == VK_WHOLE_SIZE) {
        return offset <= info->base.size;
    }

    return offset <= info->base.size && size <= info->base.size - offset;
}

void ResourceTracker::setupCaps(uint32_t& noRenderControlEnc) {
    VirtGpuDevice* instance = VirtGpuDevice::getInstance(kCapsetGfxStreamVulkan);
    if (instance) {
        mCaps = instance->getCaps();
    }

    // Delete once goldfish Linux drivers are gone
    if (mCaps.vulkanCapset.protocolVersion == 0) {
        mCaps.vulkanCapset.colorBufferMemoryIndex = 0xFFFFFFFF;
    } else {
        // Don't query the render control encoder for features, since for virtio-gpu the
        // capabilities provide versioning. Set features to be unconditionally true, since
        // using virtio-gpu encompasses all prior goldfish features.  mFeatureInfo should be
        // deprecated in favor of caps.
        mFeatureInfo.hasVulkanNullOptionalStrings = true;
        mFeatureInfo.hasVulkanIgnoredHandles = true;
        mFeatureInfo.hasVulkanShaderFloat16Int8 = true;
        mFeatureInfo.hasVulkanQueueSubmitWithCommands = true;
        mFeatureInfo.hasDeferredVulkanCommands = true;
        mFeatureInfo.hasVulkanAsyncQueueSubmit = true;
        mFeatureInfo.hasVulkanCreateResourcesWithRequirements = true;
        mFeatureInfo.hasVirtioGpuNext = true;
        mFeatureInfo.hasVirtioGpuNativeSync = true;
        mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate =
            mCaps.vulkanCapset.vulkanBatchedDescriptorSetUpdate;
        mFeatureInfo.hasVulkanAsyncQsri = true;

        ResourceTracker::streamFeatureBits |= VULKAN_STREAM_FEATURE_NULL_OPTIONAL_STRINGS_BIT;
        ResourceTracker::streamFeatureBits |= VULKAN_STREAM_FEATURE_IGNORED_HANDLES_BIT;
        ResourceTracker::streamFeatureBits |= VULKAN_STREAM_FEATURE_SHADER_FLOAT16_INT8_BIT;
        ResourceTracker::streamFeatureBits |= VULKAN_STREAM_FEATURE_QUEUE_SUBMIT_WITH_COMMANDS_BIT;
    }

    noRenderControlEnc = mCaps.vulkanCapset.noRenderControlEnc;
}

void ResourceTracker::setupFeatures(const struct GfxStreamVkFeatureInfo* features) {
    if (mFeatureInfo.setupComplete) {
        return;
    }

    mFeatureInfo = *features;
#if DETECT_OS_ANDROID
    if (mFeatureInfo.hasDirectMem) {
        mGoldfishAddressSpaceBlockProvider.reset(
            new GoldfishAddressSpaceBlockProvider(GoldfishAddressSpaceSubdeviceType::NoSubdevice));
    }
#endif  // DETECT_OS_ANDROID

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (mFeatureInfo.hasVulkan) {
        fidl::ClientEnd<fuchsia_hardware_goldfish::ControlDevice> channel{zx::channel(
            GetConnectToServiceFunction()("/loader-gpu-devices/class/goldfish-control/000"))};
        if (!channel) {
            mesa_loge("failed to open control device");
            abort();
        }
        mControlDevice =
            fidl::WireSyncClient<fuchsia_hardware_goldfish::ControlDevice>(std::move(channel));

        fidl::ClientEnd<fuchsia_sysmem::Allocator> sysmem_channel{
            zx::channel(GetConnectToServiceFunction()("/svc/fuchsia.sysmem.Allocator"))};
        if (!sysmem_channel) {
            mesa_loge("failed to open sysmem connection");
        }
        mSysmemAllocator =
            fidl::WireSyncClient<fuchsia_sysmem::Allocator>(std::move(sysmem_channel));
        char name[ZX_MAX_NAME_LEN] = {};
        zx_object_get_property(zx_process_self(), ZX_PROP_NAME, name, sizeof(name));
        std::string client_name(name);
        client_name += "-goldfish";
        zx_info_handle_basic_t info;
        zx_object_get_info(zx_process_self(), ZX_INFO_HANDLE_BASIC, &info, sizeof(info), nullptr,
                           nullptr);
        mSysmemAllocator->SetDebugClientInfo(fidl::StringView::FromExternal(client_name),
                                             info.koid);
    }
#endif

    if (mFeatureInfo.hasVulkanNullOptionalStrings) {
        ResourceTracker::streamFeatureBits |= VULKAN_STREAM_FEATURE_NULL_OPTIONAL_STRINGS_BIT;
    }
    if (mFeatureInfo.hasVulkanIgnoredHandles) {
        ResourceTracker::streamFeatureBits |= VULKAN_STREAM_FEATURE_IGNORED_HANDLES_BIT;
    }
    if (mFeatureInfo.hasVulkanShaderFloat16Int8) {
        ResourceTracker::streamFeatureBits |= VULKAN_STREAM_FEATURE_SHADER_FLOAT16_INT8_BIT;
    }
    if (mFeatureInfo.hasVulkanQueueSubmitWithCommands) {
        ResourceTracker::streamFeatureBits |= VULKAN_STREAM_FEATURE_QUEUE_SUBMIT_WITH_COMMANDS_BIT;
    }

    mFeatureInfo.setupComplete = true;
}

void ResourceTracker::setupPlatformHelpers() {
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    VirtGpuDevice* instance = VirtGpuDevice::getInstance(kCapsetGfxStreamVulkan);
    auto deviceHandle = instance ? instance->getDeviceHandle() : INVALID_DESCRIPTOR;
    if (mGralloc == nullptr) {
        mGralloc.reset(gfxstream::createPlatformGralloc(deviceHandle));
    }
#endif

    if (mSyncHelper == nullptr) {
        mSyncHelper.reset(gfxstream::createPlatformSyncHelper());
    }
}

void ResourceTracker::setThreadingCallbacks(const ResourceTracker::ThreadingCallbacks& callbacks) {
    ResourceTracker::threadingCallbacks = callbacks;
}

bool ResourceTracker::usingDirectMapping() const { return true; }

bool ResourceTracker::supportsGuestCapabilities() const {
    return mCaps.vulkanCapset.hasGuestCapabilities;
}

uint32_t ResourceTracker::getStreamFeatures() const { return ResourceTracker::streamFeatureBits; }

bool ResourceTracker::supportsDeferredCommands() const {
    return mFeatureInfo.hasDeferredVulkanCommands;
}

bool ResourceTracker::supportsAsyncQueueSubmit() const {
    return mFeatureInfo.hasVulkanAsyncQueueSubmit;
}

bool ResourceTracker::supportsAsyncResets() const {
    // A reply-less command is ordered before later commands from other guest threads only by
    // the host's per-process seqno.
    return mCaps.vulkanCapset.hasAsyncResets &&
           (streamFeatureBits & VULKAN_STREAM_FEATURE_QUEUE_SUBMIT_WITH_COMMANDS_BIT);
}

bool ResourceTracker::supportsCreateResourcesWithRequirements() const {
    return mFeatureInfo.hasVulkanCreateResourcesWithRequirements;
}

int ResourceTracker::getHostInstanceExtensionIndex(const std::string& extName) const {
    int i = 0;
    for (const auto& prop : mHostInstanceExtensions) {
        if (extName == std::string(prop.extensionName)) {
            return i;
        }
        ++i;
    }
    return -1;
}

int ResourceTracker::getHostDeviceExtensionIndex(const std::string& extName) const {
    int i = 0;
    for (const auto& prop : mHostDeviceExtensions) {
        if (extName == std::string(prop.extensionName)) {
            return i;
        }
        ++i;
    }
    return -1;
}

void ResourceTracker::deviceMemoryTransform_tohost(VkDeviceMemory* memory, uint32_t memoryCount,
                                                   VkDeviceSize* offset, uint32_t offsetCount,
                                                   VkDeviceSize* size, uint32_t sizeCount,
                                                   uint32_t* typeIndex, uint32_t typeIndexCount,
                                                   uint32_t* typeBits, uint32_t typeBitsCount) {
    (void)offsetCount;
    (void)sizeCount;
    (void)typeIndex;
    (void)typeIndexCount;
    (void)typeBits;
    (void)typeBitsCount;

    if (memory) {
        for (uint32_t i = 0; i < memoryCount; ++i) {
            VkDeviceMemory mem = memory[i];
            if (!mem) {
                return;
            }
            VK_FROM_HANDLE(gfxstream_vk_device_memory, info, mem);
            if (!info->coherentMemory) continue;

            memory[i] = info->coherentMemory->getDeviceMemory();

            if (offset) {
                offset[i] = info->coherentMemoryOffset + offset[i];
            }

            if (size && size[i] == VK_WHOLE_SIZE) {
                size[i] = info->base.size;
            }
        }
    }
}

uint32_t ResourceTracker::getColorBufferMemoryIndex(void* context, VkDevice device) {
    // Create test image to get the memory requirements
    VkEncoder* enc = (VkEncoder*)context;
    VkImageCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {64, 64, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image = VK_NULL_HANDLE;
    VkResult res = enc->vkCreateImage(device, &createInfo, nullptr, &image, true /* do lock */);

    if (res != VK_SUCCESS) {
        return 0;
    }

    VkMemoryRequirements memReqs;
    enc->vkGetImageMemoryRequirements(device, image, &memReqs, true /* do lock */);
    enc->vkDestroyImage(device, image, nullptr, true /* do lock */);

    const VkPhysicalDeviceMemoryProperties& memProps =
        getPhysicalDeviceMemoryProperties(context, device, VK_NULL_HANDLE);

    // Currently, host looks for the last index that has with memory
    // property type VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    VkMemoryPropertyFlags memoryProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (int i = VK_MAX_MEMORY_TYPES - 1; i >= 0; --i) {
        if ((memReqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & memoryProperty)) {
            return i;
        }
    }

    return 0;
}

VkResult ResourceTracker::on_vkEnumerateInstanceExtensionProperties(
    void* context, VkResult, const char*, uint32_t* pPropertyCount,
    VkExtensionProperties* pProperties) {
    std::vector<const char*> allowedExtensionNames = {
        "VK_KHR_get_physical_device_properties2", "VK_KHR_sampler_ycbcr_conversion",
        "VK_KHR_device_group_creation",
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
        "VK_KHR_external_semaphore_capabilities", "VK_KHR_external_memory_capabilities",
        "VK_KHR_external_fence_capabilities",     "VK_EXT_debug_utils",
#endif
    };

    VkEncoder* enc = (VkEncoder*)context;

    // Only advertise a select set of extensions.
    if (mHostInstanceExtensions.empty()) {
        uint32_t hostPropCount = 0;
        enc->vkEnumerateInstanceExtensionProperties(nullptr, &hostPropCount, nullptr,
                                                    true /* do lock */);
        mHostInstanceExtensions.resize(hostPropCount);

        VkResult hostRes = enc->vkEnumerateInstanceExtensionProperties(
            nullptr, &hostPropCount, mHostInstanceExtensions.data(), true /* do lock */);

        if (hostRes != VK_SUCCESS) {
            return hostRes;
        }
    }

    std::vector<VkExtensionProperties> filteredExts;

    for (size_t i = 0; i < allowedExtensionNames.size(); ++i) {
        auto extIndex = getHostInstanceExtensionIndex(allowedExtensionNames[i]);
        if (extIndex != -1) {
            filteredExts.push_back(mHostInstanceExtensions[extIndex]);
        }
    }

    VkExtensionProperties anbExtProps[] = {
#ifdef VK_USE_PLATFORM_FUCHSIA
        {"VK_KHR_external_memory_capabilities", 1},
        {"VK_KHR_external_semaphore_capabilities", 1},
#endif
    };

    for (auto& anbExtProp : anbExtProps) {
        filteredExts.push_back(anbExtProp);
    }

    // Spec:
    //
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkEnumerateInstanceExtensionProperties.html
    //
    // If pProperties is NULL, then the number of extensions properties
    // available is returned in pPropertyCount. Otherwise, pPropertyCount
    // must point to a variable set by the user to the number of elements
    // in the pProperties array, and on return the variable is overwritten
    // with the number of structures actually written to pProperties. If
    // pPropertyCount is less than the number of extension properties
    // available, at most pPropertyCount structures will be written. If
    // pPropertyCount is smaller than the number of extensions available,
    // VK_INCOMPLETE will be returned instead of VK_SUCCESS, to indicate
    // that not all the available properties were returned.
    //
    // pPropertyCount must be a valid pointer to a uint32_t value
    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;

    if (!pProperties) {
        *pPropertyCount = (uint32_t)filteredExts.size();
        return VK_SUCCESS;
    } else {
        auto actualExtensionCount = (uint32_t)filteredExts.size();
        if (*pPropertyCount > actualExtensionCount) {
            *pPropertyCount = actualExtensionCount;
        }

        for (uint32_t i = 0; i < *pPropertyCount; ++i) {
            pProperties[i] = filteredExts[i];
        }

        if (actualExtensionCount > *pPropertyCount) {
            return VK_INCOMPLETE;
        }

        return VK_SUCCESS;
    }
}

VkResult ResourceTracker::on_vkEnumerateDeviceExtensionProperties(
    void* context, VkResult, VkPhysicalDevice physdev, const char*, uint32_t* pPropertyCount,
    VkExtensionProperties* pProperties) {
    std::vector<const char*> allowedExtensionNames = {
        "VK_KHR_vulkan_memory_model",
        "VK_KHR_buffer_device_address",
        "VK_KHR_maintenance1",
        "VK_KHR_maintenance2",
        "VK_KHR_maintenance3",
        "VK_KHR_bind_memory2",
        "VK_KHR_dedicated_allocation",
        "VK_KHR_get_memory_requirements2",
        "VK_KHR_sampler_ycbcr_conversion",
        "VK_KHR_shader_float16_int8",
        "VK_KHR_timeline_semaphore",
        "VK_KHR_push_descriptor",
        "VK_KHR_dynamic_rendering_local_read",
        "VK_AMD_gpu_shader_half_float",
        "VK_NV_shader_subgroup_partitioned",
        "VK_KHR_shader_subgroup_extended_types",
        "VK_EXT_subgroup_size_control",
        "VK_EXT_provoking_vertex",
        "VK_KHR_line_rasterization",
        "VK_EXT_line_rasterization",
        "VK_KHR_shader_terminate_invocation",
        "VK_EXT_transform_feedback",
        "VK_EXT_primitive_topology_list_restart",
        "VK_EXT_index_type_uint8",
        "VK_EXT_load_store_op_none",
        "VK_EXT_swapchain_colorspace",
        "VK_EXT_image_robustness",
        "VK_EXT_robustness2",
        "VK_EXT_custom_border_color",
        "VK_EXT_border_color_swizzle",
        "VK_EXT_shader_stencil_export",
        "VK_KHR_image_format_list",
        "VK_KHR_incremental_present",
        "VK_KHR_pipeline_executable_properties",
        "VK_EXT_queue_family_foreign",
        "VK_EXT_scalar_block_layout",
        "VK_KHR_descriptor_update_template",
        "VK_KHR_storage_buffer_storage_class",
        "VK_EXT_depth_clip_enable",
        "VK_KHR_create_renderpass2",
        "VK_KHR_vertex_attribute_divisor",
        "VK_EXT_vertex_attribute_divisor",
        "VK_EXT_host_query_reset",
        "VK_EXT_blend_operation_advanced",
        "VK_EXT_frame_boundary",
        "VK_EXT_primitives_generated_query",
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
        "VK_KHR_external_semaphore",
        "VK_KHR_external_semaphore_fd",
        "VK_KHR_external_memory",
        "VK_KHR_external_fence",
        "VK_KHR_external_fence_fd",
        "VK_EXT_device_memory_report",
#endif
#ifdef LINUX_GUEST_BUILD
        // Passthrough if available on host. Will otherwise be emulated by guest
        "VK_EXT_image_drm_format_modifier",
        "VK_KHR_external_memory_fd",
        "VK_EXT_robustness2",
        "VK_EXT_fragment_density_map",
        "VK_EXT_graphics_pipeline_library",
        "VK_EXT_image_compression_control",
        "VK_EXT_image_compression_control_swapchain",
        "VK_KHR_pipeline_library",
#endif
        // Vulkan 1.1
        "VK_KHR_16bit_storage",
        "VK_KHR_device_group",
        "VK_KHR_device_group_creation",
        "VK_KHR_external_fence_capabilities",
        "VK_KHR_external_memory_capabilities",
        "VK_KHR_external_semaphore_capabilities",
        "VK_KHR_get_physical_device_properties2",
        "VK_KHR_relaxed_block_layout",
        "VK_KHR_shader_draw_parameters",
        "VK_KHR_variable_pointers",
        "VK_EXT_color_write_enable",
        "VK_EXT_memory_budget",
        "VK_KHR_multiview",

        // Vulkan 1.2
        "VK_KHR_8bit_storage",
        "VK_KHR_depth_stencil_resolve",
        "VK_KHR_draw_indirect_count",
        "VK_KHR_driver_properties",
        "VK_KHR_imageless_framebuffer",
        "VK_KHR_sampler_mirror_clamp_to_edge",
        "VK_KHR_separate_depth_stencil_layouts",
        "VK_KHR_shader_atomic_int64",
        "VK_KHR_shader_float_controls",
        "VK_KHR_spirv_1_4",
        "VK_KHR_uniform_buffer_standard_layout",
        "VK_EXT_descriptor_indexing",
        "VK_EXT_sampler_filter_minmax",
        "VK_EXT_separate_stencil_usage",
        "VK_EXT_shader_viewport_index_layer",

        // Vulkan 1.3
        "VK_KHR_synchronization2",
        "VK_EXT_private_data",
        "VK_KHR_dynamic_rendering",
        "VK_KHR_copy_commands2",
        "VK_KHR_format_feature_flags2",
        "VK_KHR_maintenance4",
        "VK_KHR_shader_integer_dot_product",
        "VK_KHR_shader_non_semantic_info",
        "VK_KHR_zero_initialize_workgroup_memory",
        "VK_EXT_4444_formats",
        "VK_EXT_extended_dynamic_state",
        "VK_EXT_extended_dynamic_state2",
        "VK_EXT_inline_uniform_block",
        "VK_EXT_pipeline_creation_cache_control",
        "VK_EXT_pipeline_creation_feedback",
        "VK_EXT_shader_demote_to_helper_invocation",
        "VK_EXT_texel_buffer_alignment",
        "VK_EXT_texture_compression_astc_hdr",
        "VK_EXT_tooling_info",
        "VK_EXT_ycbcr_2plane_444_formats",

        // Vulkan 1.4
        "VK_KHR_global_priority",
        "VK_KHR_index_type_uint8",
        "VK_KHR_load_store_op_none",
        "VK_KHR_maintenance5",
        "VK_KHR_map_memory2",
        "VK_KHR_shader_expect_assume",
        "VK_KHR_shader_float_controls2",
        "VK_KHR_shader_subgroup_rotate",
        "VK_EXT_pipeline_robustness",

        // Android requirements
        "VK_EXT_pipeline_protected_access",
        "VK_KHR_maintenance6",
        "VK_KHR_maintenance7",
        "VK_KHR_maintenance8",
        "VK_KHR_maintenance9",
        "VK_GOOGLE_display_timing",
    };

    if (supportsGuestCapabilities()) {
        allowedExtensionNames.push_back("VK_EXT_host_image_copy");
    }

    VkEncoder* enc = (VkEncoder*)context;

    if (mHostDeviceExtensions.empty()) {
        uint32_t hostPropCount = 0;
        enc->vkEnumerateDeviceExtensionProperties(physdev, nullptr, &hostPropCount, nullptr,
                                                  true /* do lock */);
        mHostDeviceExtensions.resize(hostPropCount);

        VkResult hostRes = enc->vkEnumerateDeviceExtensionProperties(
            physdev, nullptr, &hostPropCount, mHostDeviceExtensions.data(), true /* do lock */);

        if (hostRes != VK_SUCCESS) {
            return hostRes;
        }
    }

    std::vector<VkExtensionProperties> filteredExts;

    for (size_t i = 0; i < allowedExtensionNames.size(); ++i) {
        auto extIndex = getHostDeviceExtensionIndex(allowedExtensionNames[i]);
        if (extIndex != -1) {
            filteredExts.push_back(mHostDeviceExtensions[extIndex]);
        }
    }

    VkExtensionProperties anbExtProps[] = {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
        {"VK_ANDROID_native_buffer", 7},
#endif
#ifdef VK_USE_PLATFORM_FUCHSIA
        {"VK_KHR_external_memory", 1},
        {"VK_KHR_external_semaphore", 1},
        {"VK_FUCHSIA_external_semaphore", 1},
#endif
    };

    for (auto& anbExtProp : anbExtProps) {
        filteredExts.push_back(anbExtProp);
    }

    /*
     * GfxstreamEnd2EndVkTest::DeviceMemoryReport always assumes the memory report
     * extension is present.  It's is filtered out when sent host side, since for a
     * virtual GPU this is quite difficult to implement.
     *
     * Mesa runtime checks physical device features.  So if the test tries to enable
     * device level extension without it definitely existing, the test will fail.
     *
     * The test can also be modified to check VkPhysicalDeviceDeviceMemoryReportFeaturesEXT,
     * but that's more involved.  Work around this by always advertising the extension.
     * Tracking bug: b/338270042
     */
    filteredExts.push_back(VkExtensionProperties{"VK_EXT_device_memory_report", 1});

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    bool hostSupportsExternalFenceFd =
        getHostDeviceExtensionIndex("VK_KHR_external_fence_fd") != -1;
    if (!hostSupportsExternalFenceFd) {
        filteredExts.push_back(VkExtensionProperties{"VK_KHR_external_fence_fd", 1});
    }
#endif

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    bool hostHasPosixExternalSemaphore =
        getHostDeviceExtensionIndex("VK_KHR_external_semaphore_fd") != -1;
    if (!hostHasPosixExternalSemaphore) {
        // Always advertise posix external semaphore capabilities on Android/Linux.
        // SYNC_FD handles will always work, regardless of host support. Support
        // for non-sync, opaque FDs, depends on host driver support, but will
        // be handled accordingly by host.
        filteredExts.push_back(VkExtensionProperties{"VK_KHR_external_semaphore_fd", 1});
    }
#endif

    // TODO: query ExternalMemoryMode from vkEmulation directly to compansate for the host platform
    // correctly, instead of host extension checks, or make this support mandatory for Android.
    bool win32ExtMemAvailable = getHostDeviceExtensionIndex("VK_KHR_external_memory_win32") != -1;
    bool posixExtMemAvailable = getHostDeviceExtensionIndex("VK_KHR_external_memory_fd") != -1;
    bool metalExtMemAvailable = getHostDeviceExtensionIndex("VK_EXT_external_memory_metal") != -1;
    bool hostAllocationExtMemAvailable =
        getHostDeviceExtensionIndex("VK_EXT_external_memory_host") != -1;
    bool qnxExtMemAvailable =
        getHostDeviceExtensionIndex("VK_QNX_external_memory_screen_buffer") != -1;

    bool hostHasExternalMemorySupport = win32ExtMemAvailable || posixExtMemAvailable ||
                                        metalExtMemAvailable || hostAllocationExtMemAvailable ||
                                        qnxExtMemAvailable;

    if (hostHasExternalMemorySupport) {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
        filteredExts.push_back(
            VkExtensionProperties{"VK_ANDROID_external_memory_android_hardware_buffer", 7});
#endif
#ifdef VK_USE_PLATFORM_FUCHSIA
        filteredExts.push_back(VkExtensionProperties{"VK_FUCHSIA_external_memory", 1});
        filteredExts.push_back(VkExtensionProperties{"VK_FUCHSIA_buffer_collection", 1});
#endif
        filteredExts.push_back(VkExtensionProperties{"VK_EXT_queue_family_foreign", 1});
    } else {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
        mesa_loge(
            "%s: Did not recognize any form of external memory support on the host device/driver. "
            "Guest won't support VK_ANDROID_external_memory_android_hardware_buffer!",
            __func__);
#endif
#ifdef LINUX_GUEST_BUILD
        // Note: Linux gfxstream-vulkan driver automatically assumes some form of external memory
        // support on the host, and advertises VK_KHR_external_memory_fd and
        // VK_EXT_external_memory_dma_buf unconditionally.
        mesa_logw(
            "%s: Did not recognize any form of external memory support on the host device/driver. "
            "This may result in some unexpected functionality, "
            "specifically when using external memory Vulkan extensions.",
            __func__);
#endif
    }

    // NOTE: the Vulkan Loader's trampoline functions will remove duplicates. This can lead
    // to lead errors if this function returns VK_SUCCESS with N elements (including a duplicate)
    // but the Vulkan Loader's trampoline function returns VK_INCOMPLETE with N-1 elements
    // (without the duplicate).
    std::sort(filteredExts.begin(), filteredExts.end(),
              [](const VkExtensionProperties& a, const VkExtensionProperties& b) {
                  return strcmp(a.extensionName, b.extensionName) < 0;
              });
    filteredExts.erase(
        std::unique(filteredExts.begin(), filteredExts.end(),
                    [](const VkExtensionProperties& a, const VkExtensionProperties& b) {
                        return strcmp(a.extensionName, b.extensionName) == 0;
                    }),
        filteredExts.end());

    // Spec:
    //
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkEnumerateDeviceExtensionProperties.html
    //
    // pPropertyCount is a pointer to an integer related to the number of
    // extension properties available or queried, and is treated in the
    // same fashion as the
    // vkEnumerateInstanceExtensionProperties::pPropertyCount parameter.
    //
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkEnumerateInstanceExtensionProperties.html
    //
    // If pProperties is NULL, then the number of extensions properties
    // available is returned in pPropertyCount. Otherwise, pPropertyCount
    // must point to a variable set by the user to the number of elements
    // in the pProperties array, and on return the variable is overwritten
    // with the number of structures actually written to pProperties. If
    // pPropertyCount is less than the number of extension properties
    // available, at most pPropertyCount structures will be written. If
    // pPropertyCount is smaller than the number of extensions available,
    // VK_INCOMPLETE will be returned instead of VK_SUCCESS, to indicate
    // that not all the available properties were returned.
    //
    // pPropertyCount must be a valid pointer to a uint32_t value

    if (!pPropertyCount) return VK_ERROR_INITIALIZATION_FAILED;

    if (!pProperties) {
        *pPropertyCount = (uint32_t)filteredExts.size();
        return VK_SUCCESS;
    } else {
        auto actualExtensionCount = (uint32_t)filteredExts.size();
        if (*pPropertyCount > actualExtensionCount) {
            *pPropertyCount = actualExtensionCount;
        }

        for (uint32_t i = 0; i < *pPropertyCount; ++i) {
            pProperties[i] = filteredExts[i];
        }

        if (actualExtensionCount > *pPropertyCount) {
            return VK_INCOMPLETE;
        }

        return VK_SUCCESS;
    }
}

VkResult ResourceTracker::on_vkEnumeratePhysicalDevices(void* context, VkResult,
                                                        VkInstance instance,
                                                        uint32_t* pPhysicalDeviceCount,
                                                        VkPhysicalDevice* pPhysicalDevices) {
    VkEncoder* enc = (VkEncoder*)context;

    if (!instance) return VK_ERROR_INITIALIZATION_FAILED;

    if (!pPhysicalDeviceCount) return VK_ERROR_INITIALIZATION_FAILED;

    VK_FROM_HANDLE(gfxstream_vk_instance, inst, instance);
    if (!inst) return VK_ERROR_INITIALIZATION_FAILED;

    std::unique_lock<std::recursive_mutex> lock(mLock);

    if (inst->physical_devices_vec.empty()) {
        uint32_t hostPhysicalDeviceCount = 0;

        lock.unlock();
        VkResult countRes = enc->vkEnumeratePhysicalDevices(instance, &hostPhysicalDeviceCount,
                                                            nullptr, false /* no lock */);
        lock.lock();

        if (countRes != VK_SUCCESS) {
            mesa_loge(
                "%s: failed: could not count host physical devices. "
                "Error %d\n",
                __func__, countRes);
            return countRes;
        }

        inst->physical_devices_vec.resize(hostPhysicalDeviceCount);

        lock.unlock();
        VkResult enumRes = enc->vkEnumeratePhysicalDevices(
            instance, &hostPhysicalDeviceCount, inst->physical_devices_vec.data(),
            false /* no lock */);
        lock.lock();

        if (enumRes != VK_SUCCESS) {
            inst->physical_devices_vec.clear();
            mesa_loge(
                "%s: failed: could not retrieve host physical devices. "
                "Error %d\n",
                __func__, enumRes);
            return enumRes;
        }
    }

    if (!pPhysicalDevices) {
        *pPhysicalDeviceCount = (uint32_t)inst->physical_devices_vec.size();
        return VK_SUCCESS;
    } else {
        uint32_t actualDeviceCount = (uint32_t)inst->physical_devices_vec.size();
        uint32_t toWrite =
            actualDeviceCount < *pPhysicalDeviceCount ? actualDeviceCount : *pPhysicalDeviceCount;

        for (uint32_t i = 0; i < toWrite; ++i) {
            pPhysicalDevices[i] = inst->physical_devices_vec[i];
        }

        *pPhysicalDeviceCount = toWrite;

        if (actualDeviceCount > *pPhysicalDeviceCount) {
            return VK_INCOMPLETE;
        }

        return VK_SUCCESS;
    }
}

void ResourceTracker::on_vkGetPhysicalDeviceFeatures2(void*, VkPhysicalDevice,
                                                      VkPhysicalDeviceFeatures2* pFeatures) {
    if (pFeatures) {
        VkPhysicalDeviceDeviceMemoryReportFeaturesEXT* memoryReportFeaturesEXT =
            vk_find_struct(pFeatures, PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT);
        if (memoryReportFeaturesEXT) {
            memoryReportFeaturesEXT->deviceMemoryReport = VK_TRUE;
        }

        if (!supportsGuestCapabilities()) {
            VkPhysicalDeviceHostImageCopyFeatures* hostImageCopyFeatures =
                vk_find_struct(pFeatures, PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES);
            if (hostImageCopyFeatures) {
                hostImageCopyFeatures->hostImageCopy = VK_FALSE;
            }
            VkPhysicalDeviceVulkan14Features* vulkan14Features =
                vk_find_struct(pFeatures, PHYSICAL_DEVICE_VULKAN_1_4_FEATURES);
            if (vulkan14Features) {
                vulkan14Features->hostImageCopy = VK_FALSE;
            }
        }
    }
}

void ResourceTracker::on_vkGetPhysicalDeviceProperties2KHR(
    void* context, VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties) {
    on_vkGetPhysicalDeviceProperties2(context, physicalDevice, pProperties);
}

void ResourceTracker::on_vkGetPhysicalDeviceProperties2(void* context,
                                                        VkPhysicalDevice physicalDevice,
                                                        VkPhysicalDeviceProperties2* pProperties) {
    VkEncoder* enc = (VkEncoder*)context;
    VirtGpuDevice* instance = VirtGpuDevice::getInstance();
    if (!pProperties) {
        return;
    }

    // Safely extract guest-only structures from the pNext chain before calling the host encoder.
    auto* drmProps = vk_extract_struct<VkPhysicalDeviceDrmPropertiesEXT>(
        pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT);
    auto* pciProps = vk_extract_struct<VkPhysicalDevicePCIBusInfoPropertiesEXT>(
        pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT);
    auto* layeredPropsList = vk_extract_struct<VkPhysicalDeviceLayeredApiPropertiesListKHR>(
        pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LAYERED_API_PROPERTIES_LIST_KHR);
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    auto* presentationProps = vk_extract_struct<VkPhysicalDevicePresentationPropertiesANDROID>(
        pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENTATION_PROPERTIES_ANDROID);
#endif

    // Always request VkPhysicalDeviceDriverProperties to name the driver correctly
    VkPhysicalDeviceDriverProperties driverPropsLocal = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceDriverProperties* hostDriverProps =
        vk_find_struct(pProperties, PHYSICAL_DEVICE_DRIVER_PROPERTIES);
    if (!hostDriverProps) {
        hostDriverProps = &driverPropsLocal;
        __vk_append_struct(pProperties, &driverPropsLocal);
    }

    VkPhysicalDeviceLayeredApiVulkanPropertiesKHR* layeredVkProps =
        (layeredPropsList && layeredPropsList->pLayeredApis &&
         layeredPropsList->layeredApiCount > 0)
            ? vk_find_struct(&layeredPropsList->pLayeredApis[0],
                             PHYSICAL_DEVICE_LAYERED_API_VULKAN_PROPERTIES_KHR)
            : nullptr;
    VkPhysicalDeviceIDProperties* layeredIdProps =
        layeredVkProps ? vk_find_struct(&layeredVkProps->properties, PHYSICAL_DEVICE_ID_PROPERTIES)
                       : nullptr;

    VkPhysicalDeviceIDProperties idPropsLocal = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties* hostVk11Props =
        vk_find_struct(pProperties, PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES);
    VkPhysicalDeviceIDProperties* hostIdProps =
        vk_find_struct(pProperties, PHYSICAL_DEVICE_ID_PROPERTIES);
    if (layeredIdProps && !hostIdProps && !hostVk11Props) {
        hostIdProps = &idPropsLocal;
        __vk_append_struct(pProperties, &idPropsLocal);
    }

    enc->vkGetPhysicalDeviceProperties2(physicalDevice, pProperties, false /* no lock */);

    // Extract driver and ID properties from the pNext chain if appended locally
    if (hostDriverProps == &driverPropsLocal) {
        vk_extract_struct<VkPhysicalDeviceDriverProperties>(
            pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
    }
    if (hostIdProps == &idPropsLocal) {
        vk_extract_struct<VkPhysicalDeviceIDProperties>(
            pProperties, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
    }

    // Re-append extracted guest-only structures
    if (drmProps) {
        __vk_append_struct(pProperties, drmProps);
    }
    if (pciProps) {
        __vk_append_struct(pProperties, pciProps);
    }
    if (layeredPropsList) {
        __vk_append_struct(pProperties, layeredPropsList);
    }
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    if (presentationProps) {
        __vk_append_struct(pProperties, presentationProps);
    }
#endif

    // Must populate before top-level pProperties and pNext structs are overwritten below.
    if (layeredPropsList) {
        if (!layeredPropsList->pLayeredApis) {
            layeredPropsList->layeredApiCount = 1;
        } else if (layeredPropsList->layeredApiCount > 0) {
            layeredPropsList->layeredApiCount = 1;
            VkPhysicalDeviceLayeredApiPropertiesKHR* layeredApiProps =
                &layeredPropsList->pLayeredApis[0];
            layeredApiProps->vendorID = pProperties->properties.vendorID;
            layeredApiProps->deviceID = pProperties->properties.deviceID;
            layeredApiProps->layeredAPI = VK_PHYSICAL_DEVICE_LAYERED_API_VULKAN_KHR;
            memcpy(layeredApiProps->deviceName, pProperties->properties.deviceName,
                   sizeof(layeredApiProps->deviceName));

            if (layeredVkProps) {
                layeredVkProps->properties.properties = pProperties->properties;
                memset(&layeredVkProps->properties.properties.limits, 0,
                       sizeof(layeredVkProps->properties.properties.limits));
                memset(&layeredVkProps->properties.properties.sparseProperties, 0,
                       sizeof(layeredVkProps->properties.properties.sparseProperties));

                if (auto* layeredDriverProps = vk_find_struct(&layeredVkProps->properties,
                                                              PHYSICAL_DEVICE_DRIVER_PROPERTIES)) {
                    layeredDriverProps->driverID = hostDriverProps->driverID;
                    memcpy(layeredDriverProps->driverName, hostDriverProps->driverName,
                           sizeof(layeredDriverProps->driverName));
                    memcpy(layeredDriverProps->driverInfo, hostDriverProps->driverInfo,
                           sizeof(layeredDriverProps->driverInfo));
                    layeredDriverProps->conformanceVersion = hostDriverProps->conformanceVersion;
                }

                if (layeredIdProps) {
                    auto copy_id_props = [](auto* dst, const auto* src) {
                        memcpy(dst->deviceUUID, src->deviceUUID, sizeof(dst->deviceUUID));
                        memcpy(dst->driverUUID, src->driverUUID, sizeof(dst->driverUUID));
                        memcpy(dst->deviceLUID, src->deviceLUID, sizeof(dst->deviceLUID));
                        dst->deviceNodeMask = src->deviceNodeMask;
                        dst->deviceLUIDValid = src->deviceLUIDValid;
                    };
                    if (hostIdProps) {
                        copy_id_props(layeredIdProps, hostIdProps);
                    } else if (hostVk11Props) {
                        copy_id_props(layeredIdProps, hostVk11Props);
                    }
                }
            }
        }
    }

    // Overwrite device type and driver properties
    pProperties->properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
    pProperties->properties.driverVersion = vk_get_driver_version();

    const char* transport_name = instance ? "Virtio-GPU GFXStream" : "Goldfish GFXStream";
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    int device_name_len = snprintf(device_name, sizeof(device_name), "%s (%s)", transport_name,
                                   pProperties->properties.deviceName);
    if (device_name_len >= (int)VK_MAX_PHYSICAL_DEVICE_NAME_SIZE) {
        memcpy(device_name + VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 5, "...)", 4);
        device_name_len = VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1;
    }
    memcpy(pProperties->properties.deviceName, device_name, device_name_len + 1);

    char gfxstreamDriverName[VK_MAX_DRIVER_NAME_SIZE] = "";
    snprintf(gfxstreamDriverName, sizeof(gfxstreamDriverName), "gfxstream (%s)",
             hostDriverProps->driverName);
    gfxstreamDriverName[VK_MAX_DRIVER_NAME_SIZE - 1] = '\0';

    auto overwrite_id_props = [](auto* props) {
        memset(props->deviceUUID, 0, sizeof(props->deviceUUID));
        memset(props->driverUUID, 0, sizeof(props->driverUUID));
        memset(props->deviceLUID, 0, sizeof(props->deviceLUID));
        props->deviceNodeMask = 0;
        props->deviceLUIDValid = VK_FALSE;
    };

    auto overwrite_driver_props = [&](auto* props) {
        props->driverID = VK_DRIVER_ID_MESA_GFXSTREAM;
        memcpy(props->driverName, gfxstreamDriverName, sizeof(props->driverName));
        snprintf(props->driverInfo, sizeof(props->driverInfo),
                 "Mesa " PACKAGE_VERSION MESA_GIT_SHA1);
    };

    auto overwrite_drm_props = [](VirtGpuDevice* instance, auto* props) {
        if (instance) {
            VkPhysicalDeviceDrmPropertiesEXT* drmProps =
                reinterpret_cast<VkPhysicalDeviceDrmPropertiesEXT*>(props);
            VirtGpuDrmInfo drmInfo;
            if (instance->getDrmInfo(&drmInfo)) {
                drmProps->hasPrimary = drmInfo.hasPrimary;
                drmProps->hasRender = drmInfo.hasRender;
                drmProps->primaryMajor = drmInfo.primaryMajor;
                drmProps->primaryMinor = drmInfo.primaryMinor;
                drmProps->renderMajor = drmInfo.renderMajor;
                drmProps->renderMinor = drmInfo.renderMinor;
            } else {
                mesa_logd(
                    "%s: encountered VkPhysicalDeviceDrmPropertiesEXT in pProperties::pNext chain, "
                    "but failed to query DrmInfo from the VirtGpuDevice",
                    __func__);
            }
        }
    };

    auto overwrite_pci_bus_info_props = [](VirtGpuDevice* instance, auto* props) {
        if (instance) {
            VkPhysicalDevicePCIBusInfoPropertiesEXT* pciBusInfoProps =
                reinterpret_cast<VkPhysicalDevicePCIBusInfoPropertiesEXT*>(props);
            VirtGpuPciBusInfo pciBusInfo;
            if (instance->getPciBusInfo(&pciBusInfo)) {
                pciBusInfoProps->pciDomain = pciBusInfo.domain;
                pciBusInfoProps->pciBus = pciBusInfo.bus;
                pciBusInfoProps->pciDevice = pciBusInfo.device;
                pciBusInfoProps->pciFunction = pciBusInfo.function;
            } else {
                mesa_logd(
                    "%s: encountered VkPhysicalDevicePCIBusInfoPropertiesEXT in pProperties::pNext "
                    "chain, but failed to query PciBusInfo from the VirtGpuDevice",
                    __func__);
            }
        }
    };

    // Overwrite all of the properties with gfxstream driver info.
    vk_foreach_struct(sType, ext, pProperties) {
        switch (sType) {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES: {
                overwrite_id_props(reinterpret_cast<VkPhysicalDeviceVulkan11Properties*>(ext));
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES: {
                overwrite_id_props(reinterpret_cast<VkPhysicalDeviceIDProperties*>(ext));
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES: {
                overwrite_driver_props(reinterpret_cast<VkPhysicalDeviceVulkan12Properties*>(ext));
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES: {
                overwrite_driver_props(reinterpret_cast<VkPhysicalDeviceDriverProperties*>(ext));
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT: {
                overwrite_drm_props(instance, ext);
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT: {
                overwrite_pci_bus_info_props(instance, ext);
                break;
            }
            default:
                break;
        }
    }
}

void ResourceTracker::on_vkGetPhysicalDeviceMemoryProperties(
    void* context, VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* out) {
    // gfxstream decides which physical device to expose to the guest on startup.
    // Otherwise, we would need a physical device to properties mapping.
    *out = getPhysicalDeviceMemoryProperties(context, VK_NULL_HANDLE, physicalDevice);
}

void ResourceTracker::on_vkGetPhysicalDeviceMemoryProperties2(
    void*, VkPhysicalDevice physdev, VkPhysicalDeviceMemoryProperties2* out) {
    on_vkGetPhysicalDeviceMemoryProperties(nullptr, physdev, &out->memoryProperties);
}

VkResult ResourceTracker::on_vkCreateInstance(void* context, VkResult input_result,
                                              const VkInstanceCreateInfo*,
                                              const VkAllocationCallbacks*, VkInstance* pInstance) {
    if (input_result != VK_SUCCESS) return input_result;

    VkEncoder* enc = (VkEncoder*)context;

    uint32_t apiVersion = VK_API_VERSION_1_0;
    input_result = enc->vkEnumerateInstanceVersion(&apiVersion, false /* no lock */);

    VK_FROM_HANDLE(gfxstream_vk_instance, inst, *pInstance);
    if (inst) {
        inst->api_version = apiVersion;
        inst->highest_api_version = apiVersion;
    }

    return input_result;
}

VkResult ResourceTracker::on_vkCreateDevice(void* context, VkResult input_result,
                                            VkPhysicalDevice physicalDevice,
                                            const VkDeviceCreateInfo* pCreateInfo,
                                            const VkAllocationCallbacks*, VkDevice* pDevice) {
    (void)pCreateInfo;
    if (input_result != VK_SUCCESS) return input_result;

    VkEncoder* enc = (VkEncoder*)context;

    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties memProps;
    enc->vkGetPhysicalDeviceProperties(physicalDevice, &props, false /* no lock */);
    enc->vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps, false /* no lock */);

    VK_FROM_HANDLE(gfxstream_vk_device, dev, *pDevice);
    if (dev) {
        dev->internal_physical_device = physicalDevice;
        dev->props = props;
        dev->mem_props = memProps;
        dev->api_version = props.apiVersion;
    }

    return input_result;
}

void ResourceTracker::on_vkDestroyDevice_pre(void* context, VkDevice device,
                                             const VkAllocationCallbacks*) {
    (void)context;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    if (!dev) return;

    std::vector<VkDeviceMemory> leakedMemories;
    simple_mtx_lock(&dev->memory_mtx);
    dev->coherent_pools.clear();
    list_for_each_entry_safe(struct gfxstream_vk_device_memory, memObj, &dev->device_memories,
                             link) {
        list_delinit(&memObj->link);
        leakedMemories.push_back(gfxstream_vk_device_memory_to_handle(memObj));
    }
    simple_mtx_unlock(&dev->memory_mtx);

    for (VkDeviceMemory mem : leakedMemories) {
        VK_FROM_HANDLE(gfxstream_vk_device_memory, memObj, mem);
        if (memObj->coherentMemory) {
            memObj->coherentMemory->detachDeviceMemory();
            memObj->coherentMemory = nullptr;
        }
        if (memObj->importedFd >= 0) {
            close(memObj->importedFd);
            memObj->importedFd = -1;
        }
        unregister_VkDeviceMemory(mem);
        delete_gfxstream_vk_device_memory(device, mem, nullptr);
    }
}

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
static void updateMemoryTypeBits(uint32_t* memoryTypeBits, uint32_t memoryIndex) {
    *memoryTypeBits = 1u << memoryIndex;
}
#endif

#ifdef VK_USE_PLATFORM_ANDROID_KHR

VkResult ResourceTracker::on_vkGetAndroidHardwareBufferPropertiesANDROID(
    void* context, VkResult, VkDevice device, const AHardwareBuffer* buffer,
    VkAndroidHardwareBufferPropertiesANDROID* pProperties) {
    // Delete once goldfish Linux drivers are gone
    if (mCaps.vulkanCapset.colorBufferMemoryIndex == 0xFFFFFFFF) {
        mCaps.vulkanCapset.colorBufferMemoryIndex = getColorBufferMemoryIndex(context, device);
    }

    updateMemoryTypeBits(&pProperties->memoryTypeBits, mCaps.vulkanCapset.colorBufferMemoryIndex);

    return getAndroidHardwareBufferPropertiesANDROID(mGralloc.get(), buffer, pProperties);
}

VkResult ResourceTracker::on_vkGetMemoryAndroidHardwareBufferANDROID(
    void*, VkResult, VkDevice device, const VkMemoryGetAndroidHardwareBufferInfoANDROID* pInfo,
    struct AHardwareBuffer** pBuffer) {
    if (!device || !pInfo || !pInfo->memory) return VK_ERROR_INITIALIZATION_FAILED;

    VK_FROM_HANDLE(gfxstream_vk_device_memory, info, pInfo->memory);
    if (!info) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult queryRes = getMemoryAndroidHardwareBufferANDROID(mGralloc.get(), &info->ahw);

    if (queryRes != VK_SUCCESS) return queryRes;

    *pBuffer = info->ahw;

    return queryRes;
}
#endif

#ifdef VK_USE_PLATFORM_FUCHSIA
VkResult ResourceTracker::on_vkGetMemoryZirconHandleFUCHSIA(
    void*, VkResult, VkDevice device, const VkMemoryGetZirconHandleInfoFUCHSIA* pInfo,
    uint32_t* pHandle) {
    if (!device || !pInfo || !pInfo->memory) return VK_ERROR_INITIALIZATION_FAILED;

    VK_FROM_HANDLE(gfxstream_vk_device_memory, info, pInfo->memory);
    if (!info) return VK_ERROR_INITIALIZATION_FAILED;

    if (info->vmoHandle == ZX_HANDLE_INVALID) {
        mesa_loge("%s: memory cannot be exported", __func__);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    *pHandle = ZX_HANDLE_INVALID;
    zx_handle_duplicate(info->vmoHandle, ZX_RIGHT_SAME_RIGHTS, pHandle);
    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkGetMemoryZirconHandlePropertiesFUCHSIA(
    void*, VkResult, VkDevice device, VkExternalMemoryHandleTypeFlagBits handleType,
    uint32_t handle, VkMemoryZirconHandlePropertiesFUCHSIA* pProperties) {
    using fuchsia_hardware_goldfish::wire::kMemoryPropertyDeviceLocal;
    using fuchsia_hardware_goldfish::wire::kMemoryPropertyHostVisible;

    if (handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    zx_info_handle_basic_t handleInfo;
    zx_status_t status = zx::unowned_vmo(handle)->get_info(ZX_INFO_HANDLE_BASIC, &handleInfo,
                                                           sizeof(handleInfo), nullptr, nullptr);
    if (status != ZX_OK || handleInfo.type != ZX_OBJ_TYPE_VMO) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }

    VK_FROM_HANDLE(gfxstream_vk_device, info, device);
    if (!info) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    zx::vmo vmo_dup;
    status = zx::unowned_vmo(handle)->duplicate(ZX_RIGHT_SAME_RIGHTS, &vmo_dup);
    if (status != ZX_OK) {
        mesa_loge("zx_handle_duplicate() error: %d", status);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    uint32_t memoryProperty = 0u;

    auto result = mControlDevice->GetBufferHandleInfo(std::move(vmo_dup));
    if (!result.ok()) {
        mesa_loge("mControlDevice->GetBufferHandleInfo fatal error: epitaph: %d", result.status());
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (result.value().is_ok()) {
        memoryProperty = result.value().value()->info.memory_property();
    } else if (result.value().error_value() == ZX_ERR_NOT_FOUND) {
        memoryProperty = kMemoryPropertyDeviceLocal;
    } else {
        mesa_logw("GetBufferHandleInfo failed: %d", result.value().error_value());
        pProperties->memoryTypeBits = 0;
        return VK_SUCCESS;
    }

    pProperties->memoryTypeBits = 0;
    for (uint32_t i = 0; i < info->mem_props.memoryTypeCount; ++i) {
        if (((memoryProperty & kMemoryPropertyDeviceLocal) &&
             (info->mem_props.memoryTypes[i].propertyFlags &
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ||
            ((memoryProperty & kMemoryPropertyHostVisible) &&
             (info->mem_props.memoryTypes[i].propertyFlags &
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) {
            pProperties->memoryTypeBits |= 1ull << i;
        }
    }
    return VK_SUCCESS;
}

zx_koid_t getEventKoid(zx_handle_t eventHandle) {
    if (eventHandle == ZX_HANDLE_INVALID) {
        return ZX_KOID_INVALID;
    }

    zx_info_handle_basic_t info;
    zx_status_t status = zx_object_get_info(eventHandle, ZX_INFO_HANDLE_BASIC, &info, sizeof(info),
                                            nullptr, nullptr);
    if (status != ZX_OK) {
        mesa_loge("Cannot get object info of handle %u: %d", eventHandle, status);
        return ZX_KOID_INVALID;
    }
    return info.koid;
}

VkResult ResourceTracker::on_vkImportSemaphoreZirconHandleFUCHSIA(
    void*, VkResult, VkDevice device, const VkImportSemaphoreZirconHandleInfoFUCHSIA* pInfo) {
    if (!device || !pInfo || !pInfo->semaphore) return VK_ERROR_INITIALIZATION_FAILED;

    VK_FROM_HANDLE(gfxstream_vk_semaphore, info, pInfo->semaphore);
    if (!info) return VK_ERROR_INITIALIZATION_FAILED;

    if (info->eventHandle != ZX_HANDLE_INVALID) {
        zx_handle_close(info->eventHandle);
    }
#if VK_HEADER_VERSION < 174
    info->eventHandle = pInfo->handle;
#else   // VK_HEADER_VERSION >= 174
    info->eventHandle = pInfo->zirconHandle;
#endif  // VK_HEADER_VERSION < 174
    if (info->eventHandle != ZX_HANDLE_INVALID) {
        info->eventKoid = getEventKoid(info->eventHandle);
    }

    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkGetSemaphoreZirconHandleFUCHSIA(
    void*, VkResult, VkDevice device, const VkSemaphoreGetZirconHandleInfoFUCHSIA* pInfo,
    uint32_t* pHandle) {
    if (!device || !pInfo || !pInfo->semaphore) return VK_ERROR_INITIALIZATION_FAILED;

    VK_FROM_HANDLE(gfxstream_vk_semaphore, info, pInfo->semaphore);
    if (!info || info->eventHandle == ZX_HANDLE_INVALID) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    *pHandle = ZX_HANDLE_INVALID;
    zx_handle_duplicate(info->eventHandle, ZX_RIGHT_SAME_RIGHTS, pHandle);
    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkCreateBufferCollectionFUCHSIA(
    void*, VkResult, VkDevice device, const VkBufferCollectionCreateInfoFUCHSIA* pInfo,
    const VkAllocationCallbacks* pAllocator, VkBufferCollectionFUCHSIA* pCollection) {
    fidl::ClientEnd<::fuchsia_sysmem::BufferCollectionToken> token_client;

    if (pInfo->collectionToken) {
        token_client = fidl::ClientEnd<::fuchsia_sysmem::BufferCollectionToken>(
            zx::channel(pInfo->collectionToken));
    } else {
        auto endpoints = fidl::CreateEndpoints<::fuchsia_sysmem::BufferCollectionToken>();
        if (!endpoints.is_ok()) {
            mesa_loge("zx_channel_create failed: %d", endpoints.status_value());
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto result = mSysmemAllocator->AllocateSharedCollection(std::move(endpoints->server));
        if (!result.ok()) {
            mesa_loge("AllocateSharedCollection failed: %d", result.status());
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        token_client = std::move(endpoints->client);
    }

    auto endpoints = fidl::CreateEndpoints<::fuchsia_sysmem::BufferCollection>();
    if (!endpoints.is_ok()) {
        mesa_loge("zx_channel_create failed: %d", endpoints.status_value());
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto [collection_client, collection_server] = std::move(endpoints.value());

    auto result = mSysmemAllocator->BindSharedCollection(std::move(token_client),
                                                         std::move(collection_server));
    if (!result.ok()) {
        mesa_loge("BindSharedCollection failed: %d", result.status());
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    auto* sysmem_collection =
        new fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>(std::move(collection_client));
    *pCollection = create_gfxstream_vk_buffer_collection_fuchsia(
        device, pAllocator, reinterpret_cast<uint64_t>(sysmem_collection));
    if (!*pCollection) {
        delete sysmem_collection;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return VK_SUCCESS;
}

void ResourceTracker::on_vkDestroyBufferCollectionFUCHSIA(void*, VkResult, VkDevice device,
                                                          VkBufferCollectionFUCHSIA collection,
                                                          const VkAllocationCallbacks* pAllocator) {
    if (!collection) return;
    auto sysmem_collection =
        reinterpret_cast<fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>*>(
            gfxstream_vk_buffer_collection_fuchsia_to_host_u64(collection));
    if (sysmem_collection) {
        (*sysmem_collection)->Close();
    }
    delete sysmem_collection;
    delete_gfxstream_vk_buffer_collection_fuchsia(device, collection, pAllocator);
}

SetBufferCollectionImageConstraintsResult ResourceTracker::setBufferCollectionImageConstraintsImpl(
    VkEncoder* enc, VkDevice device,
    fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>* pCollection,
    const VkImageConstraintsInfoFUCHSIA* pImageConstraintsInfo) {
    const auto& collection = *pCollection;
    if (!pImageConstraintsInfo ||
        pImageConstraintsInfo->sType != VK_STRUCTURE_TYPE_IMAGE_CONSTRAINTS_INFO_FUCHSIA) {
        mesa_loge("%s: invalid pImageConstraintsInfo", __func__);
        return {VK_ERROR_INITIALIZATION_FAILED};
    }

    if (pImageConstraintsInfo->formatConstraintsCount == 0) {
        mesa_loge("%s: formatConstraintsCount must be greater than 0", __func__);
        abort();
    }

    fuchsia_sysmem::wire::BufferCollectionConstraints constraints =
        defaultBufferCollectionConstraints(
            /* min_size_bytes */ 0,
            pImageConstraintsInfo->bufferCollectionConstraints.minBufferCount,
            pImageConstraintsInfo->bufferCollectionConstraints.maxBufferCount,
            pImageConstraintsInfo->bufferCollectionConstraints.minBufferCountForCamping,
            pImageConstraintsInfo->bufferCollectionConstraints.minBufferCountForDedicatedSlack,
            pImageConstraintsInfo->bufferCollectionConstraints.minBufferCountForSharedSlack);

    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    if (!dev) {
        return {VK_ERROR_INITIALIZATION_FAILED};
    }
    VkPhysicalDevice physicalDevice = dev->internal_physical_device;

    std::vector<uint32_t> createInfoIndex;

    bool hasOptimalTiling = false;
    for (uint32_t i = 0; i < pImageConstraintsInfo->formatConstraintsCount; i++) {
        const VkImageCreateInfo* createInfo =
            &pImageConstraintsInfo->pFormatConstraints[i].imageCreateInfo;
        const VkImageFormatConstraintsInfoFUCHSIA* formatConstraints =
            &pImageConstraintsInfo->pFormatConstraints[i];

        VkResult optimalResult = VK_ERROR_FORMAT_NOT_SUPPORTED;
        if (createInfo->tiling == VK_IMAGE_TILING_OPTIMAL) {
            optimalResult = addImageBufferCollectionConstraintsFUCHSIA(
                enc, device, physicalDevice, formatConstraints, VK_IMAGE_TILING_OPTIMAL,
                &constraints);
            if (optimalResult == VK_SUCCESS) {
                createInfoIndex.push_back(i);
                hasOptimalTiling = true;
            }
        }

        VkResult linearResult = addImageBufferCollectionConstraintsFUCHSIA(
            enc, device, physicalDevice, formatConstraints, VK_IMAGE_TILING_LINEAR, &constraints);
        if (linearResult == VK_SUCCESS) {
            createInfoIndex.push_back(i);
        }

        if (linearResult == VK_SUCCESS || optimalResult == VK_SUCCESS) {
            constraints.usage.vulkan |= getBufferCollectionConstraintsVulkanImageUsage(createInfo);

            if (formatConstraints && formatConstraints->flags) {
                mesa_logw(
                    "%s: Non-zero flags (%08x) in image format "
                    "constraints; this is currently not supported, see "
                    "fxbug.dev/42147900.",
                    __func__, formatConstraints->flags);
            }
        }
    }

    VkImageConstraintsInfoFlagsFUCHSIA flags = pImageConstraintsInfo->flags;
    if (flags & VK_IMAGE_CONSTRAINTS_INFO_CPU_READ_RARELY_FUCHSIA)
        constraints.usage.cpu |= fuchsia_sysmem::wire::kCpuUsageRead;
    if (flags & VK_IMAGE_CONSTRAINTS_INFO_CPU_READ_OFTEN_FUCHSIA)
        constraints.usage.cpu |= fuchsia_sysmem::wire::kCpuUsageReadOften;
    if (flags & VK_IMAGE_CONSTRAINTS_INFO_CPU_WRITE_RARELY_FUCHSIA)
        constraints.usage.cpu |= fuchsia_sysmem::wire::kCpuUsageWrite;
    if (flags & VK_IMAGE_CONSTRAINTS_INFO_CPU_WRITE_OFTEN_FUCHSIA)
        constraints.usage.cpu |= fuchsia_sysmem::wire::kCpuUsageWriteOften;

    constraints.has_buffer_memory_constraints = true;
    auto& memory_constraints = constraints.buffer_memory_constraints;
    memory_constraints.cpu_domain_supported = true;
    memory_constraints.ram_domain_supported = true;
    memory_constraints.inaccessible_domain_supported =
        hasOptimalTiling && !(flags & (VK_IMAGE_CONSTRAINTS_INFO_CPU_READ_RARELY_FUCHSIA |
                                       VK_IMAGE_CONSTRAINTS_INFO_CPU_READ_OFTEN_FUCHSIA |
                                       VK_IMAGE_CONSTRAINTS_INFO_CPU_WRITE_RARELY_FUCHSIA |
                                       VK_IMAGE_CONSTRAINTS_INFO_CPU_WRITE_OFTEN_FUCHSIA));

    if (memory_constraints.inaccessible_domain_supported) {
        memory_constraints.heap_permitted_count = 2;
        memory_constraints.heap_permitted[0] = fuchsia_sysmem::wire::HeapType::kGoldfishDeviceLocal;
        memory_constraints.heap_permitted[1] = fuchsia_sysmem::wire::HeapType::kGoldfishHostVisible;
    } else {
        memory_constraints.heap_permitted_count = 1;
        memory_constraints.heap_permitted[0] = fuchsia_sysmem::wire::HeapType::kGoldfishHostVisible;
    }

    if (constraints.image_format_constraints_count == 0) {
        mesa_loge("%s: none of the specified formats is supported by device", __func__);
        return {VK_ERROR_FORMAT_NOT_SUPPORTED};
    }

    constexpr uint32_t kVulkanPriority = 5;
    const char kName[] = "GoldfishSysmemShared";
    collection->SetName(kVulkanPriority, fidl::StringView(kName));

    auto result = collection->SetConstraints(true, constraints);
    if (!result.ok()) {
        mesa_loge("setBufferCollectionConstraints: SetConstraints failed: %d", result.status());
        return {VK_ERROR_INITIALIZATION_FAILED};
    }

    return {VK_SUCCESS, constraints, std::move(createInfoIndex)};
}

VkResult ResourceTracker::setBufferCollectionImageConstraintsFUCHSIA(
    VkEncoder* enc, VkDevice device,
    fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>* pCollection,
    VkBufferCollectionFUCHSIA collectionHandle,
    const VkImageConstraintsInfoFUCHSIA* pImageConstraintsInfo) {
    auto setConstraintsResult =
        setBufferCollectionImageConstraintsImpl(enc, device, pCollection, pImageConstraintsInfo);
    if (setConstraintsResult.result != VK_SUCCESS) {
        return setConstraintsResult.result;
    }

    if (collectionHandle) {
        VK_FROM_HANDLE(gfxstream_vk_buffer_collection_fuchsia, bc, collectionHandle);
        if (bc) {
            bc->constraints =
                std::make_optional<fuchsia_sysmem::wire::BufferCollectionConstraints>(
                    std::move(setConstraintsResult.constraints));
            bc->createInfoIndex = std::move(setConstraintsResult.createInfoIndex);
        }
    }

    return VK_SUCCESS;
}

VkResult ResourceTracker::setBufferCollectionBufferConstraintsFUCHSIA(
    fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>* pCollection,
    VkBufferCollectionFUCHSIA collectionHandle,
    const VkBufferConstraintsInfoFUCHSIA* pBufferConstraintsInfo) {
    auto setConstraintsResult =
        setBufferCollectionBufferConstraintsImpl(pCollection, pBufferConstraintsInfo);
    if (setConstraintsResult.result != VK_SUCCESS) {
        return setConstraintsResult.result;
    }

    if (collectionHandle) {
        VK_FROM_HANDLE(gfxstream_vk_buffer_collection_fuchsia, bc, collectionHandle);
        if (bc) {
            bc->constraints =
                std::make_optional<fuchsia_sysmem::wire::BufferCollectionConstraints>(
                    setConstraintsResult.constraints);
        }
    }

    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkSetBufferCollectionImageConstraintsFUCHSIA(
    void* context, VkResult, VkDevice device, VkBufferCollectionFUCHSIA collection,
    const VkImageConstraintsInfoFUCHSIA* pImageConstraintsInfo) {
    VkEncoder* enc = (VkEncoder*)context;
    auto sysmem_collection =
        reinterpret_cast<fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>*>(
            gfxstream_vk_buffer_collection_fuchsia_to_host_u64(collection));
    return setBufferCollectionImageConstraintsFUCHSIA(enc, device, sysmem_collection, collection,
                                                      pImageConstraintsInfo);
}

VkResult ResourceTracker::on_vkSetBufferCollectionBufferConstraintsFUCHSIA(
    void*, VkResult, VkDevice, VkBufferCollectionFUCHSIA collection,
    const VkBufferConstraintsInfoFUCHSIA* pBufferConstraintsInfo) {
    auto sysmem_collection =
        reinterpret_cast<fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>*>(
            gfxstream_vk_buffer_collection_fuchsia_to_host_u64(collection));
    return setBufferCollectionBufferConstraintsFUCHSIA(sysmem_collection, collection,
                                                       pBufferConstraintsInfo);
}

VkResult ResourceTracker::getBufferCollectionImageCreateInfoIndexLocked(
    VkBufferCollectionFUCHSIA collection, fuchsia_sysmem::wire::BufferCollectionInfo2& info,
    uint32_t* outCreateInfoIndex) {
    VK_FROM_HANDLE(gfxstream_vk_buffer_collection_fuchsia, bc, collection);
    if (!bc || !bc->constraints.has_value()) {
        mesa_loge("%s: constraints not set", __func__);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    if (!info.settings.has_image_format_constraints) {
        return VK_SUCCESS;
    }

    const auto& constraints = *bc->constraints;
    const auto& createInfoIndices = bc->createInfoIndex;
    const auto& out = info.settings.image_format_constraints;

    for (size_t imageFormatIndex = 0; imageFormatIndex < constraints.image_format_constraints_count;
         imageFormatIndex++) {
        const auto& in = constraints.image_format_constraints[imageFormatIndex];
        if ((out.pixel_format.type != in.pixel_format.type) ||
            (out.pixel_format.has_format_modifier != in.pixel_format.has_format_modifier) ||
            (out.pixel_format.format_modifier.value != in.pixel_format.format_modifier.value) ||
            (out.min_bytes_per_row < in.min_bytes_per_row) ||
            (out.required_max_coded_width < in.required_max_coded_width) ||
            (out.required_max_coded_height < in.required_max_coded_height) ||
            (in.bytes_per_row_divisor != 0 &&
             out.bytes_per_row_divisor % in.bytes_per_row_divisor != 0)) {
            continue;
        }
        bool all_color_spaces_found = true;
        for (uint32_t j = 0; j < out.color_spaces_count; j++) {
            bool found_matching_color_space = false;
            for (uint32_t k = 0; k < in.color_spaces_count; k++) {
                if (out.color_space[j].type == in.color_space[k].type) {
                    found_matching_color_space = true;
                    break;
                }
            }
            if (!found_matching_color_space) {
                all_color_spaces_found = false;
                break;
            }
        }
        if (!all_color_spaces_found) {
            continue;
        }

        *outCreateInfoIndex = createInfoIndices[imageFormatIndex];
        return VK_SUCCESS;
    }

    mesa_loge("%s: cannot find a valid image format in constraints", __func__);
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult ResourceTracker::on_vkGetBufferCollectionPropertiesFUCHSIA(
    void* context, VkResult, VkDevice device, VkBufferCollectionFUCHSIA collection,
    VkBufferCollectionPropertiesFUCHSIA* pProperties) {
    VkEncoder* enc = (VkEncoder*)context;
    const auto& sysmem_collection =
        *reinterpret_cast<fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>*>(
            gfxstream_vk_buffer_collection_fuchsia_to_host_u64(collection));

    auto result = sysmem_collection->WaitForBuffersAllocated();
    if (!result.ok() || result->status != ZX_OK) {
        mesa_loge("Failed wait for allocation: %d %d", result.status(),
                  GET_STATUS_SAFE(result, status));
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    fuchsia_sysmem::wire::BufferCollectionInfo2 info = std::move(result->buffer_collection_info);

    bool is_host_visible =
        info.settings.buffer_settings.heap == fuchsia_sysmem::wire::HeapType::kGoldfishHostVisible;
    bool is_device_local =
        info.settings.buffer_settings.heap == fuchsia_sysmem::wire::HeapType::kGoldfishDeviceLocal;
    if (!is_host_visible && !is_device_local) {
        mesa_loge("buffer collection uses a non-goldfish heap (type 0x%lu)",
                  static_cast<uint64_t>(info.settings.buffer_settings.heap));
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VK_FROM_HANDLE(gfxstream_vk_device, deviceInfo, device);
    if (!deviceInfo) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    pProperties->memoryTypeBits = 0;
    for (uint32_t i = 0; i < deviceInfo->mem_props.memoryTypeCount; ++i) {
        if ((is_device_local && (deviceInfo->mem_props.memoryTypes[i].propertyFlags &
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ||
            (is_host_visible && (deviceInfo->mem_props.memoryTypes[i].propertyFlags &
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) {
            pProperties->memoryTypeBits |= 1ull << i;
        }
    }

    pProperties->bufferCount = info.buffer_count;

    auto storeProperties = [collection, pProperties]() -> VkResult {
        VK_FROM_HANDLE(gfxstream_vk_buffer_collection_fuchsia, bc, collection);
        if (!bc) {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }

        bc->properties = std::make_optional<VkBufferCollectionPropertiesFUCHSIA>(*pProperties);
        bc->properties->pNext = nullptr;
        bc->properties->sysmemColorSpaceIndex.pNext = nullptr;
        return VK_SUCCESS;
    };

    if (!info.settings.has_image_format_constraints) {
        mesa_logd("%s: buffer collection doesn't have image format constraints", __func__);
        return storeProperties();
    }

    pProperties->sysmemPixelFormat =
        static_cast<uint64_t>(info.settings.image_format_constraints.pixel_format.type);

    if (info.settings.image_format_constraints.color_spaces_count == 0) {
        mesa_loge(
            "%s: color space missing from allocated buffer collection "
            "constraints",
            __func__);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    pProperties->sysmemColorSpaceIndex.colorSpace =
        static_cast<uint32_t>(info.settings.image_format_constraints.color_space[0].type);

    auto getIndexResult =
        getBufferCollectionImageCreateInfoIndexLocked(collection, info, &pProperties->createInfoIndex);
    if (getIndexResult != VK_SUCCESS) {
        return getIndexResult;
    }

    VkPhysicalDevice physicalDevice = deviceInfo->internal_physical_device;

    VkFormat vkFormat =
        sysmemPixelFormatTypeToVk(info.settings.image_format_constraints.pixel_format.type);
    VkFormatProperties formatProperties;
    enc->vkGetPhysicalDeviceFormatProperties(physicalDevice, vkFormat, &formatProperties,
                                             true /* do lock */);
    if (is_device_local) {
        pProperties->formatFeatures = formatProperties.optimalTilingFeatures;
    }
    if (is_host_visible) {
        pProperties->formatFeatures = formatProperties.linearTilingFeatures;
    }

    pProperties->samplerYcbcrConversionComponents.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    pProperties->samplerYcbcrConversionComponents.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    pProperties->samplerYcbcrConversionComponents.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    pProperties->samplerYcbcrConversionComponents.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    pProperties->suggestedYcbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY;
    pProperties->suggestedYcbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
    pProperties->suggestedXChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
    pProperties->suggestedYChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;

    return storeProperties();
}
#endif

static uint32_t getVirglFormat(VkFormat vkFormat) {
    switch (vkFormat) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
            return VIRGL_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            return VIRGL_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return VIRGL_FORMAT_R10G10B10A2_UNORM;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return VIRGL_FORMAT_R16G16B16A16_FLOAT;
        case VK_FORMAT_R5G6B5_UNORM_PACK16:
            return VIRGL_FORMAT_B5G6R5_UNORM;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SRGB:
            return VIRGL_FORMAT_R8_UNORM;
        default:
            return 0;
    }
}

CoherentMemoryPtr ResourceTracker::createCoherentMemory(
    VkDevice device, VkDeviceMemory mem, const VkMemoryAllocateInfo& hostAllocationInfo,
    const VirtGpuResourcePtr& guestBlob, VkEncoder* enc, VkResult& res) {
    CoherentMemoryPtr coherentMemory = nullptr;

    if (guestBlob) {
        auto mapping = guestBlob->createMapping();
        if (!mapping) {
            mesa_loge("Failed to create coherent memory: failed to create blob mapping.");
            res = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            return coherentMemory;
        }

        return std::make_shared<CoherentMemory>(mapping, hostAllocationInfo.allocationSize, device,
                                                mem);
    }

#if DETECT_OS_ANDROID
    if (mFeatureInfo.hasDirectMem) {
        uint64_t gpuAddr = 0;
        GoldfishAddressSpaceBlockPtr block = nullptr;
        res = enc->vkMapMemoryIntoAddressSpaceGOOGLE(device, mem, &gpuAddr, true);
        if (res != VK_SUCCESS) {
            mesa_loge(
                "Failed to create coherent memory: vkMapMemoryIntoAddressSpaceGOOGLE "
                "returned:%d.",
                res);
            return coherentMemory;
        }
        VK_FROM_HANDLE(gfxstream_vk_device_memory, info, mem);
        if (!info) {
            mesa_loge("Failed to create coherent memory: failed to find device memory.");
            res = VK_ERROR_OUT_OF_HOST_MEMORY;
            return coherentMemory;
        }
        block = info->goldfishBlock;
        info->goldfishBlock = nullptr;

        coherentMemory = std::make_shared<CoherentMemory>(
            block, gpuAddr, hostAllocationInfo.allocationSize, device, mem);
    } else
#endif  // DETECT_OS_ANDROID
        if (mFeatureInfo.hasVirtioGpuNext) {
            struct VirtGpuCreateBlob createBlob = {0};
            uint64_t hvaSizeId[3];
            res = enc->vkGetMemoryHostAddressInfoGOOGLE(device, mem, &hvaSizeId[0], &hvaSizeId[1],
                                                        &hvaSizeId[2], true /* do lock */);
            if (res != VK_SUCCESS) {
                mesa_loge(
                    "Failed to create coherent memory: vkMapMemoryIntoAddressSpaceGOOGLE "
                    "returned:%d.",
                    res);
                return coherentMemory;
            }
            VirtGpuDevice* instance = VirtGpuDevice::getInstance((enum VirtGpuCapset)3);
            createBlob.blobMem = kBlobMemHost3d;
            createBlob.flags = kBlobFlagMappable;
            createBlob.blobId = hvaSizeId[2];
            createBlob.size = hostAllocationInfo.allocationSize;

            auto blob = instance->createBlob(createBlob);
            if (!blob) {
                mesa_loge("Failed to create coherent memory: failed to create blob.");
                res = VK_ERROR_OUT_OF_DEVICE_MEMORY;
                return coherentMemory;
            }

            VirtGpuResourceMappingPtr mapping = blob->createMapping();
            if (!mapping) {
                mesa_loge("Failed to create coherent memory: failed to create blob mapping.");
                res = VK_ERROR_OUT_OF_DEVICE_MEMORY;
                return coherentMemory;
            }

            coherentMemory =
                std::make_shared<CoherentMemory>(mapping, createBlob.size, device, mem);
        } else {
            mesa_loge("FATAL: Unsupported virtual memory feature");
            abort();
        }
    return coherentMemory;
}

VkResult ResourceTracker::allocateCoherentMemory(VkDevice device,
                                                 const VkMemoryAllocateInfo* pAllocateInfo,
                                                 const VkAllocationCallbacks* pAllocator,
                                                 VkEncoder* enc, VkDeviceMemory* pMemory) {
    uint64_t offset = 0;
    uint8_t* ptr = nullptr;
    VkMemoryAllocateFlagsInfo allocFlagsInfo;
    VkMemoryOpaqueCaptureAddressAllocateInfo opaqueCaptureAddressAllocInfo;
    VkCreateBlobGOOGLE createBlobInfo;
    VirtGpuResourcePtr guestBlob = nullptr;

    memset(&createBlobInfo, 0, sizeof(struct VkCreateBlobGOOGLE));
    createBlobInfo.sType = VK_STRUCTURE_TYPE_CREATE_BLOB_GOOGLE;

    const VkMemoryAllocateFlagsInfo* allocFlagsInfoPtr =
        vk_find_struct_const(pAllocateInfo, MEMORY_ALLOCATE_FLAGS_INFO);
    const VkMemoryOpaqueCaptureAddressAllocateInfo* opaqueCaptureAddressAllocInfoPtr =
        vk_find_struct_const(pAllocateInfo, MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO);

    bool deviceAddressMemoryAllocation =
        allocFlagsInfoPtr &&
        ((allocFlagsInfoPtr->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
         (allocFlagsInfoPtr->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT));

    bool dedicated = deviceAddressMemoryAllocation;

    if (mCaps.vulkanCapset.deferredMapping || mCaps.params[kParamCreateGuestHandle])
        dedicated = true;

    VkMemoryAllocateInfo hostAllocationInfo = vk_make_orphan_copy(*pAllocateInfo);
    vk_struct_chain_iterator structChainIter = vk_make_chain_iterator(&hostAllocationInfo);

    if (mCaps.vulkanCapset.deferredMapping || mCaps.params[kParamCreateGuestHandle]) {
        hostAllocationInfo.allocationSize =
            ALIGN_POT(pAllocateInfo->allocationSize, mCaps.vulkanCapset.blobAlignment);
    } else if (dedicated) {
        // Over-aligning to kLargestSize to some Windows drivers (b:152769369).  Can likely
        // have host report the desired alignment.
        hostAllocationInfo.allocationSize =
            ALIGN_POT(pAllocateInfo->allocationSize, kLargestPageSize);
    } else {
        VkDeviceSize roundedUpAllocSize = ALIGN_POT(pAllocateInfo->allocationSize, kMegaByte);
        hostAllocationInfo.allocationSize = std::max(roundedUpAllocSize, kDefaultHostMemBlockSize);
    }

    // Support device address capture/replay allocations
    if (deviceAddressMemoryAllocation) {
        if (allocFlagsInfoPtr) {
            MESA_TRACE_SCOPE("%s: has alloc flags\n", __func__);
            allocFlagsInfo = *allocFlagsInfoPtr;
            vk_append_struct(&structChainIter, &allocFlagsInfo);
        }

        if (opaqueCaptureAddressAllocInfoPtr) {
            MESA_TRACE_SCOPE("%s: has opaque capture address\n", __func__);
            opaqueCaptureAddressAllocInfo = *opaqueCaptureAddressAllocInfoPtr;
            vk_append_struct(&structChainIter, &opaqueCaptureAddressAllocInfo);
        }
    }

    if (mCaps.params[kParamCreateGuestHandle]) {
        struct VirtGpuCreateBlob createBlob = {0};
        struct VirtGpuExecBuffer exec = {};
        VirtGpuDevice* instance = VirtGpuDevice::getInstance();
        struct gfxstreamPlaceholderCommandVk placeholderCmd = {};

        createBlobInfo.blobId = ++mAtomicId;
        createBlobInfo.blobMem = kBlobMemGuest;
        createBlobInfo.blobFlags = kBlobFlagCreateGuestHandle;
        vk_append_struct(&structChainIter, &createBlobInfo);

        createBlob.blobMem = kBlobMemGuest;
        createBlob.flags = kBlobFlagCreateGuestHandle;
        createBlob.blobId = createBlobInfo.blobId;
        createBlob.size = hostAllocationInfo.allocationSize;

        guestBlob = instance->createBlob(createBlob);
        if (!guestBlob) {
            mesa_loge("Failed to allocate coherent memory: failed to create blob.");
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }

        placeholderCmd.hdr.opCode = GFXSTREAM_PLACEHOLDER_COMMAND_VK;
        exec.command = static_cast<void*>(&placeholderCmd);
        exec.command_size = sizeof(placeholderCmd);
        exec.flags = kRingIdx;
        exec.ring_idx = 1;
        if (instance->execBuffer(exec, guestBlob.get())) {
            mesa_loge("Failed to allocate coherent memory: failed to execbuffer for wait.");
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        guestBlob->wait();
    } else if (mCaps.vulkanCapset.deferredMapping) {
        createBlobInfo.blobId = ++mAtomicId;
        createBlobInfo.blobMem = kBlobMemHost3d;
        vk_append_struct(&structChainIter, &createBlobInfo);
    }

    const VkAllocationCallbacks* hostHandleAlloc = dedicated ? pAllocator : nullptr;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult host_res = enc->vkAllocateMemory(device, &hostAllocationInfo, hostHandleAlloc, &mem,
                                              true /* do lock */);
    if (host_res != VK_SUCCESS) {
        mesa_loge("Failed to allocate coherent memory: failed to allocate on the host: %d.",
                  host_res);
        return host_res;
    }
    VK_FROM_HANDLE(gfxstream_vk_device_memory, memObj, mem);
    memObj->base.size = pAllocateInfo->allocationSize;
    memObj->blobId = createBlobInfo.blobId;
    memObj->coherentMemorySize = hostAllocationInfo.allocationSize;
    memObj->base.memory_type_index = hostAllocationInfo.memoryTypeIndex;
    memObj->device = device;
    memObj->dedicated = dedicated;

    // With deferred mapping, the coherent memory is created on the first vkMapMemory().
    if (mCaps.vulkanCapset.deferredMapping && !mCaps.params[kParamCreateGuestHandle]) {
        *pMemory = mem;
        return VK_SUCCESS;
    }

    auto coherentMemory =
        createCoherentMemory(device, mem, hostAllocationInfo, guestBlob, enc, host_res);
    if (!coherentMemory) {
        enc->vkFreeMemory(device, mem, hostHandleAlloc, true /* do lock */);
        return host_res;
    }
    coherentMemory->setAllocator(hostHandleAlloc);

    if (!coherentMemory->subAllocate(pAllocateInfo->allocationSize, &ptr, offset)) {
        mesa_loge(
            "Failed to allocate coherent memory: failed to sub-allocate %llu bytes from %llu-byte "
            "block.",
            static_cast<unsigned long long>(pAllocateInfo->allocationSize),
            static_cast<unsigned long long>(hostAllocationInfo.allocationSize));
        // |coherentMemory| owns |mem|: destroying it on return frees |mem| on the host and
        // deletes the guest wrapper (see ~CoherentMemory()).
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    if (dedicated) {
        memObj->coherentMemoryOffset = offset;
        memObj->coherentMemory = coherentMemory;
        memObj->ptr = ptr;
        *pMemory = mem;
        return VK_SUCCESS;
    }

    VkDeviceMemory appMem = create_gfxstream_vk_device_memory(device, pAllocator, 0);
    if (!appMem) {
        coherentMemory->release(offset);
        coherentMemory = nullptr;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VK_FROM_HANDLE(gfxstream_vk_device_memory, appMemObj, appMem);
    appMemObj->base.size = pAllocateInfo->allocationSize;
    appMemObj->coherentMemorySize = hostAllocationInfo.allocationSize;
    appMemObj->coherentMemoryOffset = offset;
    appMemObj->base.memory_type_index = hostAllocationInfo.memoryTypeIndex;
    appMemObj->coherentMemory = coherentMemory;
    appMemObj->ptr = ptr;
    appMemObj->device = device;
    appMemObj->dedicated = false;

    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    if (dev) {
        simple_mtx_lock(&dev->memory_mtx);
        dev->coherent_pools.emplace_back(hostAllocationInfo.memoryTypeIndex, coherentMemory);
        simple_mtx_unlock(&dev->memory_mtx);
    } else {
        std::lock_guard<std::recursive_mutex> lock(mLock);
        mAuxCoherentPools.emplace_back(hostAllocationInfo.memoryTypeIndex, coherentMemory);
    }

    *pMemory = appMem;
    return VK_SUCCESS;
}

VkResult ResourceTracker::getCoherentMemory(const VkMemoryAllocateInfo* pAllocateInfo,
                                            const VkAllocationCallbacks* pAllocator, VkEncoder* enc,
                                            VkDevice device, VkDeviceMemory* pMemory) {
    const VkMemoryAllocateFlagsInfo* allocFlagsInfoPtr =
        vk_find_struct_const(pAllocateInfo, MEMORY_ALLOCATE_FLAGS_INFO);

    bool dedicated =
        allocFlagsInfoPtr &&
        ((allocFlagsInfoPtr->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
         (allocFlagsInfoPtr->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT));

    if (mCaps.vulkanCapset.deferredMapping || mCaps.params[kParamCreateGuestHandle])
        dedicated = true;

    if (!dedicated) {
        CoherentMemoryPtr coherentMemory = nullptr;
        uint8_t* ptr = nullptr;
        uint64_t offset = 0;

        VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
        auto trySubAllocFromPools =
            [&](std::vector<std::pair<uint32_t, std::weak_ptr<CoherentMemory>>>& pools) {
                size_t writeIdx = 0;
                for (size_t i = 0; i < pools.size(); ++i) {
                    auto locked = pools[i].second.lock();
                    if (!locked) {
                        continue;
                    }
                    pools[writeIdx++] = pools[i];
                    if (!coherentMemory && pools[i].first == pAllocateInfo->memoryTypeIndex) {
                        if (locked->subAllocate(pAllocateInfo->allocationSize, &ptr, offset)) {
                            coherentMemory = std::move(locked);
                        }
                    }
                }
                pools.resize(writeIdx);
            };

        if (dev) {
            simple_mtx_lock(&dev->memory_mtx);
            trySubAllocFromPools(dev->coherent_pools);
            simple_mtx_unlock(&dev->memory_mtx);
        } else {
            std::lock_guard<std::recursive_mutex> lock(mLock);
            trySubAllocFromPools(mAuxCoherentPools);
        }

        if (coherentMemory) {
            auto mem = create_gfxstream_vk_device_memory(device, pAllocator, 0);
            if (!mem) {
                if (dev) {
                    simple_mtx_lock(&dev->memory_mtx);
                    coherentMemory->release(offset);
                    simple_mtx_unlock(&dev->memory_mtx);
                } else {
                    std::lock_guard<std::recursive_mutex> lock(mLock);
                    coherentMemory->release(offset);
                }
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            VK_FROM_HANDLE(gfxstream_vk_device_memory, info, mem);
            info->coherentMemoryOffset = offset;
            info->ptr = ptr;
            info->base.memory_type_index = pAllocateInfo->memoryTypeIndex;
            info->base.size = pAllocateInfo->allocationSize;
            info->coherentMemory = std::move(coherentMemory);
            info->device = device;
            *pMemory = mem;
            return VK_SUCCESS;
        }
    }

    return allocateCoherentMemory(device, pAllocateInfo, pAllocator, enc, pMemory);
}

VkResult ResourceTracker::on_vkAllocateMemory(void* context, VkResult input_result, VkDevice device,
                                              const VkMemoryAllocateInfo* pAllocateInfo,
                                              const VkAllocationCallbacks* pAllocator,
                                              VkDeviceMemory* pMemory) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    uint32_t heapIndex = (dev && pAllocateInfo->memoryTypeIndex < dev->mem_props.memoryTypeCount)
                             ? dev->mem_props.memoryTypes[pAllocateInfo->memoryTypeIndex].heapIndex
                             : 0;

#define _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(result)                                         \
    {                                                                                             \
        if (!dev) return result;                                                                  \
        mesa_logw("%s:%d: Allocation failure %d", __func__, __LINE__, result);                    \
        vk_device_memory_report_emit(&dev->base, result, true, false, 0,                          \
                                     pAllocateInfo->allocationSize, VK_OBJECT_TYPE_DEVICE_MEMORY, \
                                     0, heapIndex);                                               \
        return result;                                                                            \
    }

    if (input_result != VK_SUCCESS) _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(input_result);

    VkEncoder* enc = (VkEncoder*)context;

    bool hasDedicatedImage = false;
    bool hasDedicatedBuffer = false;

    VkMemoryAllocateInfo finalAllocInfo = vk_make_orphan_copy(*pAllocateInfo);
    vk_struct_chain_iterator structChainIter = vk_make_chain_iterator(&finalAllocInfo);

    VkMemoryAllocateFlagsInfo allocFlagsInfo;
    VkMemoryOpaqueCaptureAddressAllocateInfo opaqueCaptureAddressAllocInfo;

    // Add buffer device address capture structs
    const VkMemoryAllocateFlagsInfo* allocFlagsInfoPtr =
        vk_find_struct_const(pAllocateInfo, MEMORY_ALLOCATE_FLAGS_INFO);
    const VkMemoryOpaqueCaptureAddressAllocateInfo* opaqueCaptureAddressAllocInfoPtr =
        vk_find_struct_const(pAllocateInfo, MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO);

    if (allocFlagsInfoPtr) {
        MESA_TRACE_SCOPE("%s: has alloc flags\n", __func__);
        allocFlagsInfo = *allocFlagsInfoPtr;
        vk_append_struct(&structChainIter, &allocFlagsInfo);
    }

    if (opaqueCaptureAddressAllocInfoPtr) {
        MESA_TRACE_SCOPE("%s: has opaque capture address\n", __func__);
        opaqueCaptureAddressAllocInfo = *opaqueCaptureAddressAllocInfoPtr;
        vk_append_struct(&structChainIter, &opaqueCaptureAddressAllocInfo);
    }

    VkMemoryDedicatedAllocateInfo dedicatedAllocInfo;
    VkImportColorBufferGOOGLE importCbInfo = {
        VK_STRUCTURE_TYPE_IMPORT_COLOR_BUFFER_GOOGLE,
        0,
    };
    VkImportBufferGOOGLE importBufferInfo = {
        VK_STRUCTURE_TYPE_IMPORT_BUFFER_GOOGLE,
        0,
    };

    const VkExportMemoryAllocateInfo* exportAllocateInfoPtr =
        vk_find_struct_const(pAllocateInfo, EXPORT_MEMORY_ALLOCATE_INFO);

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    const VkImportAndroidHardwareBufferInfoANDROID* importAhbInfoPtr =
        vk_find_struct_const(pAllocateInfo, IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID);
    // Even if we export allocate, the underlying operation
    // for the host is always going to be an import operation.
    // This is also how Intel's implementation works,
    // and is generally simpler;
    // even in an export allocation,
    // we perform AHardwareBuffer allocation
    // on the guest side, at this layer,
    // and then we attach a new VkDeviceMemory
    // to the AHardwareBuffer on the host via an "import" operation.
    AHardwareBuffer* ahw = nullptr;
#else
    const void* importAhbInfoPtr = nullptr;
    void* ahw = nullptr;
#endif

#ifdef VK_USE_PLATFORM_FUCHSIA
    const VkImportMemoryBufferCollectionFUCHSIA* importBufferCollectionInfoPtr =
        vk_find_struct_const(pAllocateInfo, IMPORT_MEMORY_BUFFER_COLLECTION_FUCHSIA);

    const VkImportMemoryZirconHandleInfoFUCHSIA* importVmoInfoPtr =
        vk_find_struct_const(pAllocateInfo, IMPORT_MEMORY_ZIRCON_HANDLE_INFO_FUCHSIA);
#else
    const void* importBufferCollectionInfoPtr = nullptr;
    const void* importVmoInfoPtr = nullptr;
#endif  // VK_USE_PLATFORM_FUCHSIA

    const VkMemoryDedicatedAllocateInfo* dedicatedAllocInfoPtr =
        vk_find_struct_const(pAllocateInfo, MEMORY_DEDICATED_ALLOCATE_INFO);

    // Note for AHardwareBuffers, the Vulkan spec states:
    //
    //     Android hardware buffers have intrinsic width, height, format, and usage
    //     properties, so Vulkan images bound to memory imported from an Android
    //     hardware buffer must use dedicated allocations
    //
    // so any allocation requests with a VkImportAndroidHardwareBufferInfoANDROID
    // will necessarily have a VkMemoryDedicatedAllocateInfo. However, the host
    // may or may not actually use a dedicated allocation to emulate
    // AHardwareBuffers. As such, the VkMemoryDedicatedAllocateInfo is passed to the
    // host and the host will decide whether or not to use it.

    const VkPhysicalDeviceMemoryProperties& physicalDeviceMemoryProps =
        getPhysicalDeviceMemoryProperties(context, device, VK_NULL_HANDLE);

    const bool requestedMemoryIsHostVisible =
        isHostVisible(&physicalDeviceMemoryProps, pAllocateInfo->memoryTypeIndex);

    // State needed for import/export.
    bool exportAhb = false;
    bool exportVmo = false;
    bool exportDmabuf = false;
    bool importAhb = false;
    bool importBufferCollection = false;
    bool importVmo = false;
    bool importDmabuf = false;
    (void)exportVmo;
    (void)exportAhb;

    if (exportAllocateInfoPtr) {
        exportAhb = exportAllocateInfoPtr->handleTypes &
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
#ifdef VK_USE_PLATFORM_FUCHSIA
        exportVmo = exportAllocateInfoPtr->handleTypes &
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA;
#endif  // VK_USE_PLATFORM_FUCHSIA
        exportDmabuf =
            (exportAllocateInfoPtr->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    } else if (importAhbInfoPtr) {
        importAhb = true;
    } else if (importBufferCollectionInfoPtr) {
        importBufferCollection = true;
    } else if (importVmoInfoPtr) {
        importVmo = true;
    }

    const VkImportMemoryFdInfoKHR* importFdInfoPtr =
        vk_find_struct_const(pAllocateInfo, IMPORT_MEMORY_FD_INFO_KHR);
    if (importFdInfoPtr) {
        importDmabuf =
            (importFdInfoPtr->handleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    }
    bool isImport = importAhb || importBufferCollection || importVmo || importDmabuf;

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    if (exportAhb) {
        hasDedicatedImage =
            dedicatedAllocInfoPtr && (dedicatedAllocInfoPtr->image != VK_NULL_HANDLE);
        hasDedicatedBuffer =
            dedicatedAllocInfoPtr && (dedicatedAllocInfoPtr->buffer != VK_NULL_HANDLE);
        VkExtent3D imageExtent = {0, 0, 0};
        uint32_t imageLayers = 0;
        VkFormat imageFormat = VK_FORMAT_UNDEFINED;
        VkImageUsageFlags imageUsage = 0;
        VkImageCreateFlags imageCreateFlags = 0;
        VkDeviceSize bufferSize = 0;
        VkDeviceSize allocationInfoAllocSize = finalAllocInfo.allocationSize;

        if (hasDedicatedImage) {
            VK_FROM_HANDLE(gfxstream_vk_image, info, dedicatedAllocInfoPtr->image);
            if (!info) _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(VK_ERROR_INITIALIZATION_FAILED);
            const auto& imgCi = info->createInfo;

            imageExtent = imgCi.extent;
            imageLayers = imgCi.arrayLayers;
            imageFormat = imgCi.format;
            imageUsage = imgCi.usage;
            imageCreateFlags = imgCi.flags;
        }

        if (hasDedicatedBuffer) {
            VK_FROM_HANDLE(gfxstream_vk_buffer, info, dedicatedAllocInfoPtr->buffer);
            if (!info) _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(VK_ERROR_INITIALIZATION_FAILED);
            bufferSize = info->createInfo.size;
        }

        VkResult ahbCreateRes = createAndroidHardwareBuffer(
            mGralloc.get(), hasDedicatedImage, hasDedicatedBuffer, imageExtent, imageLayers,
            imageFormat, imageUsage, imageCreateFlags, bufferSize, allocationInfoAllocSize, &ahw);

        if (ahbCreateRes != VK_SUCCESS) {
            _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(ahbCreateRes);
        }
    }

    if (importAhb) {
        ahw = importAhbInfoPtr->buffer;
        // We still need to acquire the AHardwareBuffer.
        importAndroidHardwareBuffer(mGralloc.get(), importAhbInfoPtr, nullptr);
    }

    if (ahw) {
        const uint32_t hostHandle = mGralloc->getHostHandle(ahw);
        if (mGralloc->getFormat(ahw) == AHARDWAREBUFFER_FORMAT_BLOB &&
            !mGralloc->treatBlobAsImage()) {
            importBufferInfo.buffer = hostHandle;
            vk_append_struct(&structChainIter, &importBufferInfo);
        } else {
            importCbInfo.colorBuffer = hostHandle;
            vk_append_struct(&structChainIter, &importCbInfo);
        }
    }
#endif
    zx_handle_t vmo_handle = ZX_HANDLE_INVALID;

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (importBufferCollection) {
        const auto& collection =
            *reinterpret_cast<fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>*>(
                gfxstream_vk_buffer_collection_fuchsia_to_host_u64(
                    importBufferCollectionInfoPtr->collection));
        auto result = collection->WaitForBuffersAllocated();
        if (!result.ok() || result->status != ZX_OK) {
            mesa_loge("WaitForBuffersAllocated failed: %d %d", result.status(),
                      GET_STATUS_SAFE(result, status));
            _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(VK_ERROR_INITIALIZATION_FAILED);
        }
        fuchsia_sysmem::wire::BufferCollectionInfo2& info = result->buffer_collection_info;
        uint32_t index = importBufferCollectionInfoPtr->index;
        if (info.buffer_count < index) {
            mesa_loge("Invalid buffer index: %d", index);
            _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(VK_ERROR_INITIALIZATION_FAILED);
        }
        vmo_handle = info.buffers[index].vmo.release();
    }

    if (importVmo) {
        vmo_handle = importVmoInfoPtr->handle;
    }

    if (exportVmo) {
        hasDedicatedImage =
            dedicatedAllocInfoPtr && (dedicatedAllocInfoPtr->image != VK_NULL_HANDLE);
        hasDedicatedBuffer =
            dedicatedAllocInfoPtr && (dedicatedAllocInfoPtr->buffer != VK_NULL_HANDLE);

        if (hasDedicatedImage && hasDedicatedBuffer) {
            mesa_loge(
                "Invalid VkMemoryDedicatedAllocationInfo: At least one "
                "of image and buffer must be VK_NULL_HANDLE.");
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }

        const VkImageCreateInfo* pImageCreateInfo = nullptr;

        VkBufferConstraintsInfoFUCHSIA bufferConstraintsInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_COLLECTION_CREATE_INFO_FUCHSIA,
            .pNext = nullptr,
            .createInfo = {},
            .requiredFormatFeatures = 0,
            .bufferCollectionConstraints =
                VkBufferCollectionConstraintsInfoFUCHSIA{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_COLLECTION_CONSTRAINTS_INFO_FUCHSIA,
                    .pNext = nullptr,
                    .minBufferCount = 1,
                    .maxBufferCount = 0,
                    .minBufferCountForCamping = 0,
                    .minBufferCountForDedicatedSlack = 0,
                    .minBufferCountForSharedSlack = 0,
                },
        };
        const VkBufferConstraintsInfoFUCHSIA* pBufferConstraintsInfo = nullptr;

        if (hasDedicatedImage) {
            VK_FROM_HANDLE(gfxstream_vk_image, imageInfo, dedicatedAllocInfoPtr->image);
            if (!imageInfo) return VK_ERROR_INITIALIZATION_FAILED;
            pImageCreateInfo = &imageInfo->createInfo;
        }

        if (hasDedicatedBuffer) {
            VK_FROM_HANDLE(gfxstream_vk_buffer, bufferInfo, dedicatedAllocInfoPtr->buffer);
            if (!bufferInfo) return VK_ERROR_INITIALIZATION_FAILED;
            bufferConstraintsInfo.createInfo = bufferInfo->createInfo;
            pBufferConstraintsInfo = &bufferConstraintsInfo;
        }

        hasDedicatedImage =
            hasDedicatedImage && getBufferCollectionConstraintsVulkanImageUsage(pImageCreateInfo);
        hasDedicatedBuffer = hasDedicatedBuffer && getBufferCollectionConstraintsVulkanBufferUsage(
                                                       pBufferConstraintsInfo);

        if (hasDedicatedImage || hasDedicatedBuffer) {
            auto token_ends = fidl::CreateEndpoints<::fuchsia_sysmem::BufferCollectionToken>();
            if (!token_ends.is_ok()) {
                mesa_loge("zx_channel_create failed: %d", token_ends.status_value());
                abort();
            }

            {
                auto result =
                    mSysmemAllocator->AllocateSharedCollection(std::move(token_ends->server));
                if (!result.ok()) {
                    mesa_loge("AllocateSharedCollection failed: %d", result.status());
                    abort();
                }
            }

            auto collection_ends = fidl::CreateEndpoints<::fuchsia_sysmem::BufferCollection>();
            if (!collection_ends.is_ok()) {
                mesa_loge("zx_channel_create failed: %d", collection_ends.status_value());
                abort();
            }

            {
                auto result = mSysmemAllocator->BindSharedCollection(
                    std::move(token_ends->client), std::move(collection_ends->server));
                if (!result.ok()) {
                    mesa_loge("BindSharedCollection failed: %d", result.status());
                    abort();
                }
            }

            fidl::WireSyncClient<fuchsia_sysmem::BufferCollection> collection(
                std::move(collection_ends->client));
            if (hasDedicatedImage) {
                VkResult res = setBufferCollectionConstraintsFUCHSIA(enc, device, &collection,
                                                                     pImageCreateInfo);
                if (res == VK_ERROR_FORMAT_NOT_SUPPORTED) {
                    mesa_loge("setBufferCollectionConstraints failed: format %u is not supported",
                              pImageCreateInfo->format);
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                if (res != VK_SUCCESS) {
                    mesa_loge("setBufferCollectionConstraints failed: %d", res);
                    abort();
                }
            }

            if (hasDedicatedBuffer) {
                VkResult res = setBufferCollectionBufferConstraintsFUCHSIA(
                    &collection, VK_NULL_HANDLE, pBufferConstraintsInfo);
                if (res != VK_SUCCESS) {
                    mesa_loge("setBufferCollectionBufferConstraints failed: %d", res);
                    abort();
                }
            }

            {
                auto result = collection->WaitForBuffersAllocated();
                if (result.ok() && result->status == ZX_OK) {
                    fuchsia_sysmem::wire::BufferCollectionInfo2& info =
                        result->buffer_collection_info;
                    if (!info.buffer_count) {
                        mesa_loge(
                            "WaitForBuffersAllocated returned "
                            "invalid count: %d",
                            info.buffer_count);
                        abort();
                    }
                    vmo_handle = info.buffers[0].vmo.release();
                } else {
                    mesa_loge("WaitForBuffersAllocated failed: %d %d", result.status(),
                              GET_STATUS_SAFE(result, status));
                    abort();
                }
            }

            collection->Close();

            zx::vmo vmo_copy;
            zx_status_t status = zx_handle_duplicate(vmo_handle, ZX_RIGHT_SAME_RIGHTS,
                                                     vmo_copy.reset_and_get_address());
            if (status != ZX_OK) {
                mesa_loge("Failed to duplicate VMO: %d", status);
                abort();
            }

            if (pImageCreateInfo) {
                bool isLinear = pImageCreateInfo->tiling == VK_IMAGE_TILING_LINEAR;
                if (!isLinear) {
                    fuchsia_hardware_goldfish::wire::ColorBufferFormatType format;
                    switch (pImageCreateInfo->format) {
                        case VK_FORMAT_B8G8R8A8_SINT:
                        case VK_FORMAT_B8G8R8A8_UNORM:
                        case VK_FORMAT_B8G8R8A8_SRGB:
                        case VK_FORMAT_B8G8R8A8_SNORM:
                        case VK_FORMAT_B8G8R8A8_SSCALED:
                        case VK_FORMAT_B8G8R8A8_USCALED:
                            format = fuchsia_hardware_goldfish::wire::ColorBufferFormatType::kBgra;
                            break;
                        case VK_FORMAT_R8G8B8A8_SINT:
                        case VK_FORMAT_R8G8B8A8_UNORM:
                        case VK_FORMAT_R8G8B8A8_SRGB:
                        case VK_FORMAT_R8G8B8A8_SNORM:
                        case VK_FORMAT_R8G8B8A8_SSCALED:
                        case VK_FORMAT_R8G8B8A8_USCALED:
                            format = fuchsia_hardware_goldfish::wire::ColorBufferFormatType::kRgba;
                            break;
                        case VK_FORMAT_R8_UNORM:
                        case VK_FORMAT_R8_UINT:
                        case VK_FORMAT_R8_USCALED:
                        case VK_FORMAT_R8_SNORM:
                        case VK_FORMAT_R8_SINT:
                        case VK_FORMAT_R8_SSCALED:
                        case VK_FORMAT_R8_SRGB:
                            format =
                                fuchsia_hardware_goldfish::wire::ColorBufferFormatType::kLuminance;
                            break;
                        case VK_FORMAT_R8G8_UNORM:
                        case VK_FORMAT_R8G8_UINT:
                        case VK_FORMAT_R8G8_USCALED:
                        case VK_FORMAT_R8G8_SNORM:
                        case VK_FORMAT_R8G8_SINT:
                        case VK_FORMAT_R8G8_SSCALED:
                        case VK_FORMAT_R8G8_SRGB:
                            format = fuchsia_hardware_goldfish::wire::ColorBufferFormatType::kRg;
                            break;
                        default:
                            mesa_loge("Unsupported format: %d", pImageCreateInfo->format);
                            abort();
                    }

                    fidl::Arena arena;
                    fuchsia_hardware_goldfish::wire::CreateColorBuffer2Params createParams(arena);
                    createParams.set_width(pImageCreateInfo->extent.width)
                        .set_height(pImageCreateInfo->extent.height)
                        .set_format(format)
                        .set_memory_property(
                            fuchsia_hardware_goldfish::wire::kMemoryPropertyDeviceLocal);

                    auto result = mControlDevice->CreateColorBuffer2(std::move(vmo_copy),
                                                                     std::move(createParams));
                    if (!result.ok() || result->res != ZX_OK) {
                        if (result.ok() && result->res == ZX_ERR_ALREADY_EXISTS) {
                            mesa_logd(
                                "CreateColorBuffer: color buffer already "
                                "exists\n");
                        } else {
                            mesa_loge("CreateColorBuffer failed: %d:%d", result.status(),
                                      GET_STATUS_SAFE(result, res));
                            abort();
                        }
                    }
                }
            }

            if (pBufferConstraintsInfo) {
                fidl::Arena arena;
                fuchsia_hardware_goldfish::wire::CreateBuffer2Params createParams(arena);
                createParams.set_size(arena, pBufferConstraintsInfo->createInfo.size)
                    .set_memory_property(
                        fuchsia_hardware_goldfish::wire::kMemoryPropertyDeviceLocal);

                auto result =
                    mControlDevice->CreateBuffer2(std::move(vmo_copy), std::move(createParams));
                if (!result.ok() || result->is_error()) {
                    mesa_loge("CreateBuffer2 failed: %d:%d", result.status(),
                              GET_STATUS_SAFE(result, error_value()));
                    abort();
                }
            }
        } else {
            mesa_logw(
                "Dedicated image / buffer not available. Cannot create "
                "BufferCollection to export VMOs.");
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }

    if (vmo_handle != ZX_HANDLE_INVALID) {
        zx::vmo vmo_copy;
        zx_status_t status =
            zx_handle_duplicate(vmo_handle, ZX_RIGHT_SAME_RIGHTS, vmo_copy.reset_and_get_address());
        if (status != ZX_OK) {
            mesa_loge("Failed to duplicate VMO: %d", status);
            abort();
        }

        auto result = mControlDevice->GetBufferHandle(std::move(vmo_copy));
        if (!result.ok() || result->res != ZX_OK) {
            mesa_loge("GetBufferHandle failed: %d:%d", result.status(),
                      GET_STATUS_SAFE(result, res));
        } else {
            fuchsia_hardware_goldfish::wire::BufferHandleType handle_type = result->type;
            uint32_t buffer_handle = result->id;

            if (handle_type == fuchsia_hardware_goldfish::wire::BufferHandleType::kBuffer) {
                importBufferInfo.buffer = buffer_handle;
                vk_append_struct(&structChainIter, &importBufferInfo);
            } else {
                importCbInfo.colorBuffer = buffer_handle;
                vk_append_struct(&structChainIter, &importCbInfo);
            }
        }
    }
#endif

    VirtGpuResourcePtr bufferBlob = nullptr;
    int importedFd = -1;
#if defined(LINUX_GUEST_BUILD)
    hasDedicatedImage = dedicatedAllocInfoPtr && (dedicatedAllocInfoPtr->image != VK_NULL_HANDLE);
    hasDedicatedBuffer = dedicatedAllocInfoPtr && (dedicatedAllocInfoPtr->buffer != VK_NULL_HANDLE);

    // Check for import first; this takes precedence over exportDmabuf in creating the
    // VirtGpuResource
    if (exportDmabuf && hasDedicatedImage) {
        VK_FROM_HANDLE(gfxstream_vk_image, imgInfo, dedicatedAllocInfoPtr->image);
        if (!imgInfo || !imgInfo->external) {
            exportDmabuf = false;
        }
    }
    if (importDmabuf) {
        VirtGpuExternalHandle importHandle = {};
        // importBlob impl may close the receivedFd after it creates an GEM handle from it. dup()
        // the FD here for input to that impl, then manage the original FD at the Vulkan level
        importHandle.osHandle = dup(importFdInfoPtr->fd);
        importHandle.type = kMemHandleDmabuf;

        auto instance = VirtGpuDevice::getInstance();
        bufferBlob = instance->importBlob(importHandle);
        if (!bufferBlob) {
            mesa_loge("%s: Failed to import colorBuffer resource\n", __func__);
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        // As per the Vulkan spec, the ownership of this FD has been transferred
        // to the implementation
        importedFd = importFdInfoPtr->fd;
    } else if (exportDmabuf) {
        VirtGpuDevice* instance = VirtGpuDevice::getInstance();

        if (hasDedicatedImage) {
            VK_FROM_HANDLE(gfxstream_vk_image, imageInfo, dedicatedAllocInfoPtr->image);
            if (!imageInfo) return VK_ERROR_INITIALIZATION_FAILED;
            VkImageCreateInfo imageCreateInfo = imageInfo->createInfo;

            uint32_t virglFormat = gfxstream::vk::getVirglFormat(imageCreateInfo.format);
            if (!virglFormat) {
                mesa_loge("Unsupported VK format for VirtGpu resource, vkFormat: 0x%x",
                          imageCreateInfo.format);
                return VK_ERROR_FORMAT_NOT_SUPPORTED;
            }
            const uint32_t target = PIPE_TEXTURE_2D;
            uint32_t bind = VIRGL_BIND_RENDER_TARGET;

            if (mCaps.vulkanCapset.alwaysBlob) {
                struct gfxstreamResourceCreate3d create3d = {};
                struct VirtGpuExecBuffer exec = {};
                struct gfxstreamPlaceholderCommandVk placeholderCmd = {};
                struct VirtGpuCreateBlob createBlob = {};

                create3d.hdr.opCode = GFXSTREAM_RESOURCE_CREATE_3D;
                create3d.bind = bind;
                create3d.target = target;
                create3d.format = virglFormat;
                create3d.width = imageCreateInfo.extent.width;
                create3d.height = imageCreateInfo.extent.height;
                create3d.blobId = ++mAtomicId;

                createBlob.blobCmd = reinterpret_cast<uint8_t*>(&create3d);
                createBlob.blobCmdSize = sizeof(create3d);
                createBlob.blobMem = kBlobMemHost3d;
                createBlob.flags = kBlobFlagShareable | kBlobFlagCrossDevice;
                createBlob.blobId = create3d.blobId;
                createBlob.size = finalAllocInfo.allocationSize;

                bufferBlob = instance->createBlob(createBlob);
                if (!bufferBlob) return VK_ERROR_OUT_OF_DEVICE_MEMORY;

                placeholderCmd.hdr.opCode = GFXSTREAM_PLACEHOLDER_COMMAND_VK;
                exec.command = static_cast<void*>(&placeholderCmd);
                exec.command_size = sizeof(placeholderCmd);
                exec.flags = kRingIdx;
                exec.ring_idx = 1;
                if (instance->execBuffer(exec, bufferBlob.get())) {
                    mesa_loge("Failed to execbuffer placeholder command.");
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }

                if (bufferBlob->wait()) {
                    mesa_loge("Failed to wait for blob.");
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
            } else {
                const VkImageSubresource imageSubresource = {
                    .aspectMask =
                        (imageCreateInfo.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT)
                            ? VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT
                            : VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .arrayLayer = 0,
                };
                VkSubresourceLayout subResourceLayout = {};
                on_vkGetImageSubresourceLayout(context, device, dedicatedAllocInfoPtr->image,
                                               &imageSubresource, &subResourceLayout);
                if (!subResourceLayout.rowPitch) {
                    mesa_loge("Failed to query stride for VirtGpu resource creation.");
                    return VK_ERROR_INITIALIZATION_FAILED;
                }

                bufferBlob = instance->createResource(
                    imageCreateInfo.extent.width, imageCreateInfo.extent.height,
                    subResourceLayout.rowPitch,
                    subResourceLayout.rowPitch * imageCreateInfo.extent.height, virglFormat, target,
                    bind);
                if (!bufferBlob) {
                    mesa_loge("Failed to create colorBuffer resource for Image memory");
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                if (bufferBlob->wait()) {
                    mesa_loge("Failed to wait for colorBuffer resource for Image memory");
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
            }
        } else {
            uint32_t virglFormat = VIRGL_FORMAT_R8_UNORM;
            const uint32_t target = PIPE_BUFFER;
            uint32_t bind = VIRGL_BIND_LINEAR;
            uint32_t width = finalAllocInfo.allocationSize;
            uint32_t height = 1;

            if (mCaps.vulkanCapset.alwaysBlob) {
                struct gfxstreamResourceCreate3d create3d = {};
                struct VirtGpuExecBuffer exec = {};
                struct gfxstreamPlaceholderCommandVk placeholderCmd = {};
                struct VirtGpuCreateBlob createBlob = {};

                create3d.hdr.opCode = GFXSTREAM_RESOURCE_CREATE_3D;
                create3d.bind = bind;
                create3d.target = target;
                create3d.format = virglFormat;
                create3d.width = width;
                create3d.height = height;
                create3d.blobId = ++mAtomicId;

                createBlob.blobCmd = reinterpret_cast<uint8_t*>(&create3d);
                createBlob.blobCmdSize = sizeof(create3d);
                createBlob.blobMem = kBlobMemHost3d;
                createBlob.flags = kBlobFlagShareable | kBlobFlagCrossDevice;
                createBlob.blobId = create3d.blobId;
                createBlob.size = width;

                bufferBlob = instance->createBlob(createBlob);
                if (!bufferBlob) return VK_ERROR_OUT_OF_DEVICE_MEMORY;

                placeholderCmd.hdr.opCode = GFXSTREAM_PLACEHOLDER_COMMAND_VK;
                exec.command = static_cast<void*>(&placeholderCmd);
                exec.command_size = sizeof(placeholderCmd);
                exec.flags = kRingIdx;
                exec.ring_idx = 1;
                if (instance->execBuffer(exec, bufferBlob.get())) {
                    mesa_loge("Failed to allocate coherent memory: failed to execbuffer for wait.");
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }

                bufferBlob->wait();
            } else {
                bufferBlob = instance->createResource(width, height, width, width * height,
                                                      virglFormat, target, bind);
                if (!bufferBlob) {
                    mesa_loge("Failed to create colorBuffer resource for Image memory");
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                if (bufferBlob->wait()) {
                    mesa_loge("Failed to wait for colorBuffer resource for Image memory");
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
            }
        }
    }

    if (bufferBlob) {
        if (hasDedicatedImage) {
            importCbInfo.colorBuffer = bufferBlob->getResourceHandle();
            vk_append_struct(&structChainIter, &importCbInfo);
        } else {
            importBufferInfo.buffer = bufferBlob->getResourceHandle();
            vk_append_struct(&structChainIter, &importBufferInfo);
        }
    }
#endif

    bool shouldPassThroughDedicatedAllocInfo =
        (!exportAllocateInfoPtr || ahw != nullptr || bufferBlob != nullptr) &&
        !importBufferCollectionInfoPtr && !importVmoInfoPtr;
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    shouldPassThroughDedicatedAllocInfo &=
        (ahw != nullptr || bufferBlob != nullptr || !requestedMemoryIsHostVisible);
#endif

    if (shouldPassThroughDedicatedAllocInfo && dedicatedAllocInfoPtr) {
        dedicatedAllocInfo = vk_make_orphan_copy(*dedicatedAllocInfoPtr);
        vk_append_struct(&structChainIter, &dedicatedAllocInfo);
    }

    if (ahw || bufferBlob || !requestedMemoryIsHostVisible) {
        input_result =
            enc->vkAllocateMemory(device, &finalAllocInfo, pAllocator, pMemory, true /* do lock */);

        if (input_result != VK_SUCCESS) _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(input_result);

        VK_FROM_HANDLE(gfxstream_vk_device_memory, memObj, *pMemory);
        memObj->device = device;
        memObj->base.size = 0;
        memObj->ptr = nullptr;
        memObj->base.memory_type_index = finalAllocInfo.memoryTypeIndex;
#ifdef VK_USE_PLATFORM_ANDROID_KHR
        memObj->ahw = ahw;
#endif
        memObj->imported = isImport;
        memObj->vmoHandle = vmo_handle;
        memObj->blobPtr = bufferBlob;
        memObj->importedFd = importedFd;

        uint64_t memoryObjectId = (uint64_t)(uintptr_t)(*pMemory);
#ifdef VK_USE_PLATFORM_ANDROID_KHR
        if (ahw) {
            memoryObjectId = getAHardwareBufferId(ahw);
        }
#endif
        if (dev) {
            vk_device_memory_report_emit(&dev->base, VK_SUCCESS, true, isImport, memoryObjectId,
                                         pAllocateInfo->allocationSize,
                                         VK_OBJECT_TYPE_DEVICE_MEMORY,
                                         (uint64_t)(uintptr_t)(*pMemory), heapIndex);
        }
        return VK_SUCCESS;
    }

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (vmo_handle != ZX_HANDLE_INVALID) {
        input_result =
            enc->vkAllocateMemory(device, &finalAllocInfo, pAllocator, pMemory, true /* do lock */);
        if (input_result != VK_SUCCESS) _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(input_result);

        zx_info_handle_basic handle_info;
        zx_status_t status = zx_object_get_info(vmo_handle, ZX_INFO_HANDLE_BASIC, &handle_info,
                                                sizeof(handle_info), nullptr, nullptr);
        if (status != ZX_OK) {
            mesa_loge("%s: cannot get vmo object info: vmo = %u status: %d.", __func__, vmo_handle,
                      status);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        zx_vm_option_t vm_permission = 0u;
        vm_permission |= (handle_info.rights & ZX_RIGHT_READ) ? ZX_VM_PERM_READ : 0;
        vm_permission |= (handle_info.rights & ZX_RIGHT_WRITE) ? ZX_VM_PERM_WRITE : 0;

        zx_paddr_t addr;
        status = zx_vmar_map(zx_vmar_root_self(), vm_permission, 0, vmo_handle, 0,
                             finalAllocInfo.allocationSize, &addr);
        if (status != ZX_OK) {
            mesa_loge("%s: cannot map vmar: status %d.", __func__, status);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        VK_FROM_HANDLE(gfxstream_vk_device_memory, memObj, *pMemory);
        memObj->device = device;
        memObj->base.size = finalAllocInfo.allocationSize;
        memObj->ptr = reinterpret_cast<uint8_t*>(addr);
        memObj->base.memory_type_index = finalAllocInfo.memoryTypeIndex;
        memObj->imported = isImport;
        memObj->vmoHandle = vmo_handle;
        memObj->importedFd = -1;
        return VK_SUCCESS;
    }
#endif

    VkResult result = getCoherentMemory(&finalAllocInfo, pAllocator, enc, device, pMemory);
    if (result != VK_SUCCESS) _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT(result);

    uint64_t memoryObjectId = (uint64_t)(uintptr_t)(*pMemory);

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (ahw) {
        memoryObjectId = getAHardwareBufferId(ahw);
    }
#endif

    if (dev) {
        vk_device_memory_report_emit(&dev->base, VK_SUCCESS, true, isImport, memoryObjectId,
                                     pAllocateInfo->allocationSize, VK_OBJECT_TYPE_DEVICE_MEMORY,
                                     (uint64_t)(uintptr_t)(*pMemory), heapIndex);
    }
    return VK_SUCCESS;
#undef _RETURN_FAILURE_WITH_DEVICE_MEMORY_REPORT
}

void ResourceTracker::on_vkFreeMemory(void* context, VkDevice device, VkDeviceMemory memory,
                                      const VkAllocationCallbacks* pAllocateInfo) {
    if (!memory) return;

    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_device_memory, info, memory);
    if (!info) return;

    uint64_t memoryObjectId = (uint64_t)(uintptr_t)memory;
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (info->ahw) {
        memoryObjectId = getAHardwareBufferId(info->ahw);
    }
#endif
    if (info->importedFd >= 0) {
        close(info->importedFd);
        info->importedFd = -1;
    }

    if (dev) {
        uint32_t heapIndex = (info->base.memory_type_index < dev->mem_props.memoryTypeCount)
                                 ? dev->mem_props.memoryTypes[info->base.memory_type_index].heapIndex
                                 : 0;
        vk_device_memory_report_emit(&dev->base, VK_SUCCESS, false, info->imported, memoryObjectId,
                                     0, VK_OBJECT_TYPE_DEVICE_MEMORY,
                                     (uint64_t)(uintptr_t)memory, heapIndex);
    }

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (info->vmoHandle && info->ptr) {
        zx_status_t status = zx_vmar_unmap(
            zx_vmar_root_self(), reinterpret_cast<zx_paddr_t>(info->ptr), info->base.size);
        if (status != ZX_OK) {
            mesa_loge("%s: Cannot unmap ptr: status %d", __func__, status);
        }
        info->ptr = nullptr;
    }
#endif

    if (!info->coherentMemory) {
        VkEncoder* enc = (VkEncoder*)context;
        enc->vkFreeMemory(device, memory, pAllocateInfo, true /* do lock */);
        return;
    }

    auto coherentMemory = freeCoherentMemory(device, memory, pAllocateInfo);
    coherentMemory = nullptr;
}

VkResult ResourceTracker::on_vkMapMemory(void* context, VkResult host_result, VkDevice device,
                                         VkDeviceMemory memory, VkDeviceSize offset,
                                         VkDeviceSize size, VkMemoryMapFlags, void** ppData) {
    if (host_result != VK_SUCCESS) {
        mesa_loge("%s: Host failed to map", __func__);
        return host_result;
    }

    VK_FROM_HANDLE(gfxstream_vk_device_memory, deviceMemoryInfo, memory);
    if (!deviceMemoryInfo) {
        mesa_loge("%s: Failed to find VkDeviceMemory.", __func__);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    if (deviceMemoryInfo->blobId && !deviceMemoryInfo->coherentMemory &&
        !mCaps.params[kParamCreateGuestHandle]) {
        VkEncoder* enc = (VkEncoder*)context;
        VkResult vkResult = enc->vkGetBlobGOOGLE(device, memory, /*doLock*/ false);
        if (vkResult != VK_SUCCESS) {
            mesa_loge("%s: Failed to vkGetBlobGOOGLE().", __func__);
            return vkResult;
        }

        struct VirtGpuCreateBlob createBlob = {};
        createBlob.blobMem = kBlobMemHost3d;
        createBlob.flags = kBlobFlagMappable;
        createBlob.blobId = deviceMemoryInfo->blobId;
        createBlob.size = deviceMemoryInfo->coherentMemorySize;

        auto blob = VirtGpuDevice::getInstance()->createBlob(createBlob);
        if (!blob) {
            mesa_loge("%s: createBlob failed", __func__);
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }

        VirtGpuResourceMappingPtr mapping = blob->createMapping();
        if (!mapping) {
            mesa_loge("%s: createMapping failed", __func__);
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }

        auto coherentMemory =
            std::make_shared<CoherentMemory>(mapping, createBlob.size, device, memory);

        // |coherentMemory| owns |memory| and frees it on destruction, so keep it even if the
        // sub-allocation fails; it is released when the application frees |memory|.
        deviceMemoryInfo->coherentMemory = coherentMemory;

        uint8_t* ptr = nullptr;
        uint64_t subAllocOffset = 0;
        if (!coherentMemory->subAllocate(deviceMemoryInfo->base.size, &ptr, subAllocOffset)) {
            mesa_loge("%s: Failed to sub-allocate %llu bytes from %llu-byte coherent memory block.",
                      __func__, static_cast<unsigned long long>(deviceMemoryInfo->base.size),
                      static_cast<unsigned long long>(createBlob.size));
            return VK_ERROR_MEMORY_MAP_FAILED;
        }

        deviceMemoryInfo->coherentMemoryOffset = subAllocOffset;
        deviceMemoryInfo->ptr = ptr;
    }

    if (!deviceMemoryInfo->ptr) {
        mesa_loge("%s: VkDeviceMemory has nullptr.", __func__);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    if (size != VK_WHOLE_SIZE &&
        (offset > deviceMemoryInfo->base.size ||
         size > deviceMemoryInfo->base.size - offset)) {
        mesa_loge(
            "%s: size is too big. alloc size 0x%llx while we wanted offset 0x%llx size 0x%llx "
            "total 0x%llx",
            __func__, (unsigned long long)deviceMemoryInfo->base.size,
            (unsigned long long)offset, (unsigned long long)size, (unsigned long long)offset);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    *ppData = deviceMemoryInfo->ptr + offset;

    return host_result;
}

void ResourceTracker::on_vkUnmapMemory(void*, VkDevice, VkDeviceMemory) {
    // no-op
}

VkResult ResourceTracker::on_vkMapMemory2(void* context, VkResult host_result, VkDevice device,
                                          const VkMemoryMapInfo* pMemoryMapInfo, void** ppData) {
    return on_vkMapMemory(context, host_result, device, pMemoryMapInfo->memory,
                          pMemoryMapInfo->offset, pMemoryMapInfo->size, pMemoryMapInfo->flags,
                          ppData);
}

VkResult ResourceTracker::on_vkUnmapMemory2(void* context, VkResult host_result, VkDevice device,
                                            const VkMemoryUnmapInfo* pMemoryUnmapInfo) {
    on_vkUnmapMemory(context, device, pMemoryUnmapInfo->memory);
    return host_result;
}

VkResult ResourceTracker::on_vkFlushMappedMemoryRanges(
    void* context, VkResult, VkDevice device, uint32_t memoryRangeCount,
    const VkMappedMemoryRange* pMemoryRanges) {
    if (!memoryRangeCount) return VK_SUCCESS;
    if (!pMemoryRanges) return VK_ERROR_OUT_OF_HOST_MEMORY;

    for (uint32_t i = 0; i < memoryRangeCount; ++i) {
        if (!isValidMemoryRange(pMemoryRanges[i])) {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    if (usingDirectMapping()) {
        VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
        if (dev && dev->mem_props.memoryTypeCount > 0) {
            bool allHostCoherent = true;
            for (uint32_t i = 0; i < memoryRangeCount; ++i) {
                if (pMemoryRanges[i].pNext != nullptr) {
                    allHostCoherent = false;
                    break;
                }
                VK_FROM_HANDLE(gfxstream_vk_device_memory, mem, pMemoryRanges[i].memory);
                if (!mem || mem->base.memory_type_index >= dev->mem_props.memoryTypeCount ||
                    !(dev->mem_props.memoryTypes[mem->base.memory_type_index].propertyFlags &
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    allHostCoherent = false;
                    break;
                }
            }
            if (allHostCoherent) {
                return VK_SUCCESS;
            }
        }
    }

    VkEncoder* enc = (VkEncoder*)context;
    return enc->vkFlushMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges,
                                          true /* do lock */);
}

VkResult ResourceTracker::on_vkInvalidateMappedMemoryRanges(
    void* context, VkResult, VkDevice device, uint32_t memoryRangeCount,
    const VkMappedMemoryRange* pMemoryRanges) {
    if (!memoryRangeCount) return VK_SUCCESS;
    if (!pMemoryRanges) return VK_ERROR_OUT_OF_HOST_MEMORY;

    for (uint32_t i = 0; i < memoryRangeCount; ++i) {
        if (!isValidMemoryRange(pMemoryRanges[i])) {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    if (usingDirectMapping()) {
        VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
        if (dev && dev->mem_props.memoryTypeCount > 0) {
            bool allHostCoherent = true;
            for (uint32_t i = 0; i < memoryRangeCount; ++i) {
                if (pMemoryRanges[i].pNext != nullptr) {
                    allHostCoherent = false;
                    break;
                }
                VK_FROM_HANDLE(gfxstream_vk_device_memory, mem, pMemoryRanges[i].memory);
                if (!mem || mem->base.memory_type_index >= dev->mem_props.memoryTypeCount ||
                    !(dev->mem_props.memoryTypes[mem->base.memory_type_index].propertyFlags &
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    allHostCoherent = false;
                    break;
                }
            }
            if (allHostCoherent) {
                return VK_SUCCESS;
            }
        }
    }

    VkEncoder* enc = (VkEncoder*)context;
    return enc->vkInvalidateMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges,
                                               true /* do lock */);
}

void ResourceTracker::transformImageMemoryRequirements2ForGuest(VkImage image,
                                                                VkMemoryRequirements2* reqs2) {
    VK_FROM_HANDLE(gfxstream_vk_image, info, image);
    if (!info) return;

    if (!info->external || !info->externalCreateInfo.handleTypes) {
        transformImageMemoryRequirementsForGuest(image, &reqs2->memoryRequirements);
        return;
    }

    transformImageMemoryRequirementsForGuest(image, &reqs2->memoryRequirements);

    VkMemoryDedicatedRequirements* dedicatedReqs =
        vk_find_struct(reqs2, MEMORY_DEDICATED_REQUIREMENTS);

    if (!dedicatedReqs) return;

    transformExternalResourceMemoryDedicatedRequirementsForGuest(dedicatedReqs);
}

void ResourceTracker::transformBufferMemoryRequirements2ForGuest(VkBuffer buffer,
                                                                 VkMemoryRequirements2* reqs2) {
    VK_FROM_HANDLE(gfxstream_vk_buffer, info, buffer);
    if (!info || !info->external || !info->externalCreateInfo.handleTypes) {
        return;
    }

    VkMemoryDedicatedRequirements* dedicatedReqs =
        vk_find_struct(reqs2, MEMORY_DEDICATED_REQUIREMENTS);

    if (!dedicatedReqs) return;

    transformExternalResourceMemoryDedicatedRequirementsForGuest(dedicatedReqs);
}

VkResult ResourceTracker::on_vkCreateImage(void* context, VkResult, VkDevice device,
                                           const VkImageCreateInfo* pCreateInfo,
                                           const VkAllocationCallbacks* pAllocator,
                                           VkImage* pImage) {
    VkEncoder* enc = (VkEncoder*)context;

    VkImageCreateInfo localCreateInfo = vk_make_orphan_copy(*pCreateInfo);
    if (localCreateInfo.sharingMode != VK_SHARING_MODE_CONCURRENT) {
        localCreateInfo.queueFamilyIndexCount = 0;
        localCreateInfo.pQueueFamilyIndices = nullptr;
    }

    vk_struct_chain_iterator structChainIter = vk_make_chain_iterator(&localCreateInfo);
    VkExternalMemoryImageCreateInfo localExtImgCi = {};

    const VkExternalMemoryImageCreateInfo* extImgCiPtr =
        vk_find_struct_const(pCreateInfo, EXTERNAL_MEMORY_IMAGE_CREATE_INFO);

    if (extImgCiPtr) {
        localExtImgCi = vk_make_orphan_copy(*extImgCiPtr);
#if defined(LINUX_GUEST_BUILD)
        localExtImgCi.handleTypes &= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        if (localExtImgCi.handleTypes) {
            bool valid = getVirglFormat(pCreateInfo->format) &&
                         pCreateInfo->imageType == VK_IMAGE_TYPE_2D &&
                         pCreateInfo->mipLevels == 1 && pCreateInfo->arrayLayers == 1;
            if (valid && pCreateInfo->tiling == VK_IMAGE_TILING_LINEAR) {
                VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
                valid = !dev || !doImageDrmFormatModifierEmulation(dev->internal_physical_device);
            }
            if (!valid) {
                localExtImgCi.handleTypes &= ~VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            }
        }
#endif
        if (localExtImgCi.handleTypes) {
            vk_append_struct(&structChainIter, &localExtImgCi);
        }
    }

#if defined(LINUX_GUEST_BUILD)
    if (pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&
        !localExtImgCi.handleTypes) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VkImageDrmFormatModifierExplicitCreateInfoEXT localDrmFormatModifierInfo;
    VkImageDrmFormatModifierListCreateInfoEXT localDrmFormatModifierList;
    const VkImageDrmFormatModifierExplicitCreateInfoEXT* drmFmtMod =
        vk_find_struct_const(pCreateInfo, IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT);

    // If the VkImage will be bound to guest-dmabuf memory
    if (localExtImgCi.handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
        const wsi_image_create_info* wsiImageCi =
            vk_find_struct_const(pCreateInfo, WSI_IMAGE_CREATE_INFO_MESA);
        if (wsiImageCi) {
            localCreateInfo.flags &= ~VK_IMAGE_CREATE_ALIAS_BIT;
        }

        const VkImageDrmFormatModifierListCreateInfoEXT* drmFmtModList =
            vk_find_struct_const(pCreateInfo, IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT);
        if ((pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) &&
            (drmFmtMod || drmFmtModList)) {
            VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
            if (!dev) return VK_ERROR_UNKNOWN;
            VkPhysicalDevice physicalDevice = dev->internal_physical_device;
            if (doImageDrmFormatModifierEmulation(physicalDevice)) {
                bool canUseLinearModifier =
                    (drmFmtMod && drmFmtMod->drmFormatModifier == DRM_FORMAT_MOD_LINEAR) ||
                    (drmFmtModList &&
                     std::any_of(
                         drmFmtModList->pDrmFormatModifiers,
                         drmFmtModList->pDrmFormatModifiers + drmFmtModList->drmFormatModifierCount,
                         [](const uint64_t mod) { return mod == DRM_FORMAT_MOD_LINEAR; }));
                if (canUseLinearModifier) {
                    mesa_logd(
                        "vkCreateImage: emulating DRM_FORMAT_MOD_LINEAR with "
                        "VK_IMAGE_TILING_OPTIMAL");
                    localCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
                } else {
                    mesa_loge(
                        "Host does not support DRM format modifiers; DRM_FORMAT_MOD_LINEAR must be "
                        "provided, as it is the only format modifier that can be emulated");
                    return VK_ERROR_FORMAT_NOT_SUPPORTED;
                }
            } else {
                if (drmFmtMod) {
                    localDrmFormatModifierInfo = vk_make_orphan_copy(*drmFmtMod);
                    vk_append_struct(&structChainIter, &localDrmFormatModifierInfo);
                }
                if (drmFmtModList) {
                    localDrmFormatModifierList = vk_make_orphan_copy(*drmFmtModList);
                    vk_append_struct(&structChainIter, &localDrmFormatModifierList);
                }
            }
        }
    }
#endif

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    VkNativeBufferANDROID localAnb;
    const VkNativeBufferANDROID* anbInfoPtr =
        vk_find_struct_const(pCreateInfo, NATIVE_BUFFER_ANDROID);
    if (anbInfoPtr) {
        localAnb = vk_make_orphan_copy(*anbInfoPtr);
        vk_append_struct(&structChainIter, &localAnb);
    }

    VkExternalFormatANDROID localExtFormatAndroid;
    const VkExternalFormatANDROID* extFormatAndroidPtr =
        vk_find_struct_const(pCreateInfo, EXTERNAL_FORMAT_ANDROID);
    if (extFormatAndroidPtr) {
        localExtFormatAndroid = vk_make_orphan_copy(*extFormatAndroidPtr);

        // Do not append external format android;
        // instead, replace the local image localCreateInfo format
        // with the corresponding Vulkan format
        if (extFormatAndroidPtr->externalFormat) {
            localCreateInfo.format = vk_format_from_fourcc(extFormatAndroidPtr->externalFormat);
            if (localCreateInfo.format == VK_FORMAT_UNDEFINED)
                return VK_ERROR_VALIDATION_FAILED_EXT;
        }
    }
#endif

#ifdef VK_USE_PLATFORM_FUCHSIA
    const VkBufferCollectionImageCreateInfoFUCHSIA* extBufferCollectionPtr =
        vk_find_struct_const(pCreateInfo, BUFFER_COLLECTION_IMAGE_CREATE_INFO_FUCHSIA);

    bool isSysmemBackedMemory = false;

    if (extImgCiPtr &&
        (extImgCiPtr->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA)) {
        isSysmemBackedMemory = true;
    }

    if (extBufferCollectionPtr) {
        const auto& collection =
            *reinterpret_cast<fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>*>(
                gfxstream_vk_buffer_collection_fuchsia_to_host_u64(
                    extBufferCollectionPtr->collection));
        uint32_t index = extBufferCollectionPtr->index;
        zx::vmo vmo;

        fuchsia_sysmem::wire::BufferCollectionInfo2 info;

        auto result = collection->WaitForBuffersAllocated();
        if (result.ok() && result->status == ZX_OK) {
            info = std::move(result->buffer_collection_info);
            if (index < info.buffer_count && info.settings.has_image_format_constraints) {
                vmo = std::move(info.buffers[index].vmo);
            }
        } else {
            mesa_loge("WaitForBuffersAllocated failed: %d %d", result.status(),
                      GET_STATUS_SAFE(result, status));
        }

        if (vmo.is_valid()) {
            zx::vmo vmo_dup;
            if (zx_status_t status = vmo.duplicate(ZX_RIGHT_SAME_RIGHTS, &vmo_dup);
                status != ZX_OK) {
                mesa_loge("%s: zx_vmo_duplicate failed: %d", __func__, status);
                abort();
            }

            auto buffer_handle_result = mControlDevice->GetBufferHandle(std::move(vmo_dup));
            if (!buffer_handle_result.ok()) {
                mesa_loge("%s: GetBufferHandle FIDL error: %d", __func__,
                          buffer_handle_result.status());
                abort();
            }
            if (buffer_handle_result.value().res == ZX_OK) {
                if (buffer_handle_result.value().type !=
                    fuchsia_hardware_goldfish::wire::BufferHandleType::kColorBuffer) {
                    mesa_loge("%s: BufferHandle %u is not a ColorBuffer", __func__,
                              buffer_handle_result.value().id);
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
            } else if (buffer_handle_result.value().res == ZX_ERR_NOT_FOUND) {
                auto format = info.settings.image_format_constraints.pixel_format.type ==
                                      fuchsia_sysmem::wire::PixelFormatType::kR8G8B8A8
                                  ? fuchsia_hardware_goldfish::wire::ColorBufferFormatType::kRgba
                                  : fuchsia_hardware_goldfish::wire::ColorBufferFormatType::kBgra;

                uint32_t memory_property =
                    info.settings.buffer_settings.heap ==
                            fuchsia_sysmem::wire::HeapType::kGoldfishDeviceLocal
                        ? fuchsia_hardware_goldfish::wire::kMemoryPropertyDeviceLocal
                        : fuchsia_hardware_goldfish::wire::kMemoryPropertyHostVisible;

                fidl::Arena arena;
                fuchsia_hardware_goldfish::wire::CreateColorBuffer2Params createParams(arena);
                createParams.set_width(info.settings.image_format_constraints.min_coded_width)
                    .set_height(info.settings.image_format_constraints.min_coded_height)
                    .set_format(format)
                    .set_memory_property(memory_property);

                auto result =
                    mControlDevice->CreateColorBuffer2(std::move(vmo), std::move(createParams));
                if (result.ok() && result->res == ZX_ERR_ALREADY_EXISTS) {
                    mesa_logd("CreateColorBuffer: color buffer already exists\n");
                } else if (!result.ok() || result->res != ZX_OK) {
                    mesa_loge("CreateColorBuffer failed: %d:%d", result.status(),
                              GET_STATUS_SAFE(result, res));
                }
            }

            if (info.settings.buffer_settings.heap ==
                fuchsia_sysmem::wire::HeapType::kGoldfishHostVisible) {
                mesa_logd(
                    "%s: Image uses host visible memory heap; set tiling "
                    "to linear to match host ImageCreateInfo",
                    __func__);
                localCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
            }
        }
        isSysmemBackedMemory = true;
    }

    if (isSysmemBackedMemory) {
        localCreateInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    }
#endif

    VkResult res;
    VkMemoryRequirements memReqs;

    if (supportsCreateResourcesWithRequirements()) {
        res = enc->vkCreateImageWithRequirementsGOOGLE(device, &localCreateInfo, pAllocator, pImage,
                                                       &memReqs, true /* do lock */);
    } else {
        res = enc->vkCreateImage(device, &localCreateInfo, pAllocator, pImage, true /* do lock */);
    }

    if (res != VK_SUCCESS) return res;

    VK_FROM_HANDLE(gfxstream_vk_image, info, *pImage);
    if (!info) return VK_ERROR_INITIALIZATION_FAILED;

    info->device = device;
    info->createInfo = *pCreateInfo;
    info->createInfo.pNext = nullptr;

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (extFormatAndroidPtr && extFormatAndroidPtr->externalFormat) {
        info->hasExternalFormat = true;
        info->externalFourccFormat = extFormatAndroidPtr->externalFormat;
    }
    info->hasAnb = (anbInfoPtr != nullptr);
#endif  // VK_USE_PLATFORM_ANDROID_KHR

    if (supportsCreateResourcesWithRequirements()) {
        info->baseRequirementsKnown = true;
    }

    if (localExtImgCi.handleTypes) {
        info->external = true;
        info->externalCreateInfo = localExtImgCi;
    }

#if defined(LINUX_GUEST_BUILD)
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    if (dev) {
        info->emulatedDrmFormatModifier =
            (pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) &&
            doImageDrmFormatModifierEmulation(dev->internal_physical_device);
    }
    if (info->emulatedDrmFormatModifier && drmFmtMod && drmFmtMod->pPlaneLayouts &&
        drmFmtMod->drmFormatModifierPlaneCount > 0) {
        info->hasExplicitDrmModifier = true;
        info->explicitPlaneLayout = drmFmtMod->pPlaneLayouts[0];
        if (info->explicitPlaneLayout.size == 0) {
            fillEmulatedLinearSubresourceLayout(info->createInfo, &info->explicitPlaneLayout);
        }
    }
#endif

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (isSysmemBackedMemory) {
        info->isSysmemBackedMemory = true;
    }
#endif

// Delete `protocolVersion` check goldfish drivers are gone.
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    if (mCaps.vulkanCapset.colorBufferMemoryIndex == 0xFFFFFFFF) {
        mCaps.vulkanCapset.colorBufferMemoryIndex = getColorBufferMemoryIndex(context, device);
    }
    if ((extImgCiPtr && (extImgCiPtr->handleTypes &
                         VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID))) {
        updateMemoryTypeBits(&memReqs.memoryTypeBits, mCaps.vulkanCapset.colorBufferMemoryIndex);
    }
#endif
#if defined(LINUX_GUEST_BUILD)
    if (mCaps.vulkanCapset.colorBufferMemoryIndex == 0xFFFFFFFF) {
        mCaps.vulkanCapset.colorBufferMemoryIndex = getColorBufferMemoryIndex(context, device);
    }
    if (localExtImgCi.handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
        updateMemoryTypeBits(&memReqs.memoryTypeBits, mCaps.vulkanCapset.colorBufferMemoryIndex);
    }
#endif

    if (info->baseRequirementsKnown) {
        transformImageMemoryRequirementsForGuest(*pImage, &memReqs);
        info->baseRequirements = memReqs;
    }
    return res;
}

VkResult ResourceTracker::on_vkCreateSamplerYcbcrConversion(
    void* context, VkResult, VkDevice device, const VkSamplerYcbcrConversionCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSamplerYcbcrConversion* pYcbcrConversion) {
    VkSamplerYcbcrConversionCreateInfo localCreateInfo = vk_make_orphan_copy(*pCreateInfo);

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    const VkExternalFormatANDROID* extFormatAndroidPtr =
        vk_find_struct_const(pCreateInfo, EXTERNAL_FORMAT_ANDROID);
    if (extFormatAndroidPtr) {
        if (extFormatAndroidPtr->externalFormat == DRM_FORMAT_RGB565) {
            // We don't support external formats on host and it causes RGB565
            // to fail in CtsGraphicsTestCases android.graphics.cts.BasicVulkanGpuTest
            // when passed as an external format.
            // Allocate a guest-only vk_ycbcr_conversion wrapper (underlying == 0) so
            // Mesa's common runtime (e.g. private data and debug utils object names)
            // always sees a valid vk_object_base.
            // See b/134771579.
            *pYcbcrConversion = create_gfxstream_vk_sampler_ycbcr_conversion(device, pAllocator, 0);
            if (!*pYcbcrConversion) {
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            return VK_SUCCESS;
        } else if (extFormatAndroidPtr->externalFormat) {
            localCreateInfo.format = vk_format_from_fourcc(extFormatAndroidPtr->externalFormat);
        }
    }
#endif

    VkEncoder* enc = (VkEncoder*)context;
    return enc->vkCreateSamplerYcbcrConversion(device, &localCreateInfo, pAllocator,
                                               pYcbcrConversion, true /* do lock */);
}

void ResourceTracker::on_vkDestroySamplerYcbcrConversion(void* context, VkDevice device,
                                                         VkSamplerYcbcrConversion ycbcrConversion,
                                                         const VkAllocationCallbacks* pAllocator) {
    if (!ycbcrConversion) {
        return;
    }
    if (gfxstream_vk_sampler_ycbcr_conversion_to_host_u64(ycbcrConversion) == 0) {
        delete_gfxstream_vk_sampler_ycbcr_conversion(device, ycbcrConversion, pAllocator);
        return;
    }
    VkEncoder* enc = (VkEncoder*)context;
    enc->vkDestroySamplerYcbcrConversion(device, ycbcrConversion, pAllocator, true /* do lock */);
}

VkResult ResourceTracker::on_vkCreateSampler(void* context, VkResult, VkDevice device,
                                             const VkSamplerCreateInfo* pCreateInfo,
                                             const VkAllocationCallbacks* pAllocator,
                                             VkSampler* pSampler) {
    VkSamplerCreateInfo localCreateInfo = vk_make_orphan_copy(*pCreateInfo);

    vk_struct_chain_iterator structChainIter = vk_make_chain_iterator(&localCreateInfo);

    VkSamplerYcbcrConversionInfo localVkSamplerYcbcrConversionInfo;
    const VkSamplerYcbcrConversionInfo* samplerYcbcrConversionInfo =
        vk_find_struct_const(pCreateInfo, SAMPLER_YCBCR_CONVERSION_INFO);
    if (samplerYcbcrConversionInfo && gfxstream_vk_sampler_ycbcr_conversion_to_host_u64(
                                          samplerYcbcrConversionInfo->conversion) != 0) {
        localVkSamplerYcbcrConversionInfo = vk_make_orphan_copy(*samplerYcbcrConversionInfo);
        vk_append_struct(&structChainIter, &localVkSamplerYcbcrConversionInfo);
    }

    VkSamplerCustomBorderColorCreateInfoEXT localVkSamplerCustomBorderColorCreateInfo;
    const VkSamplerCustomBorderColorCreateInfoEXT* samplerCustomBorderColorCreateInfo =
        vk_find_struct_const(pCreateInfo, SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT);
    if (samplerCustomBorderColorCreateInfo) {
        localVkSamplerCustomBorderColorCreateInfo =
            vk_make_orphan_copy(*samplerCustomBorderColorCreateInfo);
        vk_append_struct(&structChainIter, &localVkSamplerCustomBorderColorCreateInfo);
    }

    VkSamplerReductionModeCreateInfo localVkSamplerReductionModeCreateInfo;
    const VkSamplerReductionModeCreateInfo* samplerReductionModeCreateInfo =
        vk_find_struct_const(pCreateInfo, SAMPLER_REDUCTION_MODE_CREATE_INFO);
    if (samplerReductionModeCreateInfo) {
        localVkSamplerReductionModeCreateInfo =
            vk_make_orphan_copy(*samplerReductionModeCreateInfo);
        vk_append_struct(&structChainIter, &localVkSamplerReductionModeCreateInfo);
    }

    VkSamplerBorderColorComponentMappingCreateInfoEXT
        localVkSamplerBorderColorComponentMappingCreateInfo;
    const VkSamplerBorderColorComponentMappingCreateInfoEXT*
        samplerBorderColorComponentMappingCreateInfo = vk_find_struct_const(
            pCreateInfo, SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT);
    if (samplerBorderColorComponentMappingCreateInfo) {
        localVkSamplerBorderColorComponentMappingCreateInfo =
            vk_make_orphan_copy(*samplerBorderColorComponentMappingCreateInfo);
        vk_append_struct(&structChainIter, &localVkSamplerBorderColorComponentMappingCreateInfo);
    }

    VkEncoder* enc = (VkEncoder*)context;
    return enc->vkCreateSampler(device, &localCreateInfo, pAllocator, pSampler, true /* do lock */);
}

void ResourceTracker::on_vkGetPhysicalDeviceExternalFenceProperties(
    void* context, VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceExternalFenceInfo* pExternalFenceInfo,
    VkExternalFenceProperties* pExternalFenceProperties) {
    (void)context;
    (void)physicalDevice;

    pExternalFenceProperties->exportFromImportedHandleTypes = 0;
    pExternalFenceProperties->compatibleHandleTypes = 0;
    pExternalFenceProperties->externalFenceFeatures = 0;

    bool syncFd = pExternalFenceInfo->handleType & VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;

    if (!syncFd) {
        return;
    }

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    pExternalFenceProperties->exportFromImportedHandleTypes =
        VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
    pExternalFenceProperties->compatibleHandleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
    pExternalFenceProperties->externalFenceFeatures =
        VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT;
#endif
}

VkResult ResourceTracker::on_vkCreateFence(void* context, VkResult input_result, VkDevice device,
                                           const VkFenceCreateInfo* pCreateInfo,
                                           const VkAllocationCallbacks* pAllocator,
                                           VkFence* pFence) {
    VkEncoder* enc = (VkEncoder*)context;
    VkFenceCreateInfo finalCreateInfo = *pCreateInfo;

    const VkExportFenceCreateInfo* exportFenceInfoPtr =
        vk_find_struct_const(pCreateInfo, EXPORT_FENCE_CREATE_INFO);

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    bool exportSyncFd = exportFenceInfoPtr && (exportFenceInfoPtr->handleTypes &
                                               VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT);
#endif

    input_result =
        enc->vkCreateFence(device, &finalCreateInfo, pAllocator, pFence, true /* do lock */);

    if (input_result != VK_SUCCESS) return input_result;

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    if (exportSyncFd) {
        if (!mFeatureInfo.hasVirtioGpuNativeSync) {
            MESA_TRACE_SCOPE("%s: ensure sync device\n", __func__);
            ensureSyncDeviceFd();
        }

        VK_FROM_HANDLE(gfxstream_vk_fence, info, *pFence);
        if (!info) return VK_ERROR_INITIALIZATION_FAILED;

        info->external = true;
        info->exportFenceCreateInfo = *exportFenceInfoPtr;
    }
#endif

    return input_result;
}

void ResourceTracker::on_vkDestroyFence(void* context, VkDevice device, VkFence fence,
                                        const VkAllocationCallbacks* pAllocator) {
    VkEncoder* enc = (VkEncoder*)context;
    enc->vkDestroyFence(device, fence, pAllocator, true /* do lock */);
}

VkResult ResourceTracker::on_vkResetFences(void* context, VkResult, VkDevice device,
                                           uint32_t fenceCount, const VkFence* pFences) {
    if (!fenceCount) return VK_SUCCESS;

    VkEncoder* enc = (VkEncoder*)context;

    VK_FROM_HANDLE(gfxstream_vk_device, gfxstream_device, device);
    VkFence stackFences[16];
    std::vector<VkFence> heapFences;
    VkFence* hostFences = stackFences;
    if (fenceCount > ARRAY_SIZE(stackFences)) {
        heapFences.resize(fenceCount);
        hostFences = heapFences.data();
    }
    uint32_t hostFenceCount = 0;
    for (uint32_t i = 0; i < fenceCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_fence, gfxstream_fence, pFences[i]);
        if (isNoopFence(gfxstream_fence)) {
            vk_fence_reset_temporary(&gfxstream_device->base, &gfxstream_fence->base);
        } else {
            hostFences[hostFenceCount++] = pFences[i];
        }
    }
    if (!hostFenceCount) return VK_SUCCESS;

    VkResult res = VK_SUCCESS;
    if (supportsAsyncResets()) {
        // vkResetFences() can only fail with out of memory errors: skip the host round trip for
        // the result.
        enc->vkResetFencesAsyncGOOGLE(device, hostFenceCount, hostFences, true /* do lock */);
    } else {
        res = enc->vkResetFences(device, hostFenceCount, hostFences, true /* do lock */);
    }

    if (res != VK_SUCCESS) return res;

    // Permanence: temporary
    // on fence reset, close the fence fd
    // and act like we need to GetFenceFdKHR/ImportFenceFdKHR again
#if DETECT_OS_LINUX
    for (uint32_t i = 0; i < hostFenceCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_fence, info, hostFences[i]);
        if (!info || !info->external) continue;

        if (info->syncFd && *info->syncFd >= 0) {
            MESA_TRACE_SCOPE("%s: resetting fence. make fd -1\n", __func__);

#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
            goldfish_sync_signal(*info->syncFd);
#endif
            mSyncHelper->close(*info->syncFd);
        }
        info->syncFd.reset();
    }
#endif

    return res;
}

VkResult ResourceTracker::on_vkImportFenceFdKHR(void* context, VkResult, VkDevice device,
                                                const VkImportFenceFdInfoKHR* pImportFenceFdInfo) {
    (void)context;
    (void)device;
    (void)pImportFenceFdInfo;

    // Transference: copy
    // meaning dup() the incoming fd

    bool hasFence = pImportFenceFdInfo->fence != VK_NULL_HANDLE;

    if (!hasFence) return VK_ERROR_OUT_OF_HOST_MEMORY;

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX

    bool syncFdImport = pImportFenceFdInfo->handleType & VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;

    if (!syncFdImport) {
        mesa_loge("%s: VK_ERROR_OUT_OF_HOST_MEMORY: no sync fd import\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VK_FROM_HANDLE(gfxstream_vk_fence, info, pImportFenceFdInfo->fence);
    if (!info) {
        mesa_loge("%s: VK_ERROR_OUT_OF_HOST_MEMORY: no fence info\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    if (info->syncFd && *info->syncFd >= 0) {
        MESA_TRACE_SCOPE("%s: previous sync fd exists, close it\n", __func__);

#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
        goldfish_sync_signal(*info->syncFd);
#endif

        mSyncHelper->close(*info->syncFd);
    }

    if (pImportFenceFdInfo->fd < 0) {
        MESA_TRACE_SCOPE("%s: import -1, set to -1 and exit\n", __func__);
        info->syncFd = -1;
    } else {
        MESA_TRACE_SCOPE("%s: import actual fd, dup and close()\n", __func__);

        int fenceCopy = mSyncHelper->dup(pImportFenceFdInfo->fd);
        if (fenceCopy < 0) {
            mesa_loge("Failed to dup() import sync fd.");
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        info->syncFd = fenceCopy;

        mSyncHelper->close(pImportFenceFdInfo->fd);
    }
    return VK_SUCCESS;
#else
    return VK_ERROR_OUT_OF_HOST_MEMORY;
#endif
}

VkResult ResourceTracker::on_vkGetFenceFdKHR(void* context, VkResult, VkDevice device,
                                             const VkFenceGetFdInfoKHR* pGetFdInfo, int* pFd) {
    // export operation.
    // first check if fence is signaled
    // then if so, return -1
    // else, queue work

    VkEncoder* enc = (VkEncoder*)context;

    bool hasFence = pGetFdInfo->fence != VK_NULL_HANDLE;

    if (!hasFence) {
        mesa_loge("%s: VK_ERROR_OUT_OF_HOST_MEMORY: no fence\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    bool syncFdExport = pGetFdInfo->handleType & VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;

    if (!syncFdExport) {
        mesa_loge("%s: VK_ERROR_OUT_OF_HOST_MEMORY: no sync fd fence\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VkResult currentFenceStatus =
        enc->vkGetFenceStatus(device, pGetFdInfo->fence, true /* do lock */);

    if (VK_ERROR_DEVICE_LOST == currentFenceStatus) {  // Other error
        mesa_loge("%s: VK_ERROR_DEVICE_LOST: Other error\n", __func__);
        *pFd = -1;
        return VK_ERROR_DEVICE_LOST;
    }

    if (VK_NOT_READY == currentFenceStatus || VK_SUCCESS == currentFenceStatus) {
        // Fence is valid. We also create a new sync fd for a signaled
        // fence, because ANGLE will use the returned fd directly to
        // implement eglDupNativeFenceFDANDROID, where -1 is only returned
        // when error occurs.
        VK_FROM_HANDLE(gfxstream_vk_fence, info, pGetFdInfo->fence);
        if (!info) {
            mesa_loge("%s: VK_ERROR_OUT_OF_HOST_MEMORY: no fence info\n", __func__);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        bool syncFdCreated = info->external && (info->exportFenceCreateInfo.handleTypes &
                                                VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT);

        if (!syncFdCreated) {
            mesa_loge("%s: VK_ERROR_OUT_OF_HOST_MEMORY: no sync fd created\n", __func__);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        if (mFeatureInfo.hasVirtioGpuNativeSync) {
            VkResult result;
            int64_t osHandle;
            uint64_t hostFenceHandle = gfxstream_vk_fence_to_host_u64(pGetFdInfo->fence);

            result = createFence(device, hostFenceHandle, osHandle);
            if (result != VK_SUCCESS) return result;

            *pFd = osHandle;
        } else {
#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
            std::lock_guard<std::recursive_mutex> lock(mLock);
            goldfish_sync_queue_work(
                mSyncDeviceFd, gfxstream_vk_fence_to_host_u64(pGetFdInfo->fence) /* the handle */,
                GOLDFISH_SYNC_VULKAN_SEMAPHORE_SYNC /* thread handle (doubling as type field) */,
                pFd);
#endif
        }

        // relinquish ownership
        info->syncFd.reset();

        MESA_TRACE_SCOPE("%s: got fd: %d\n", __func__, *pFd);
        return VK_SUCCESS;
    }
    return VK_ERROR_DEVICE_LOST;
#else
    return VK_ERROR_OUT_OF_HOST_MEMORY;
#endif
}

VkResult ResourceTracker::on_vkGetFenceStatus(void* context, VkResult input_result, VkDevice device,
                                              VkFence fence) {
    (void)input_result;
    VkEncoder* enc = (VkEncoder*)context;

    VK_FROM_HANDLE(gfxstream_vk_fence, gfxstream_fence, fence);
    if (isNoopFence(gfxstream_fence)) {
        return VK_SUCCESS;
    }

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    VK_FROM_HANDLE(gfxstream_vk_fence, fenceInfo, fence);
    if (!fenceInfo) {
        return VK_NOT_READY;
    }
    if (fenceInfo->syncFd) {
        if (*fenceInfo->syncFd == -1) {
            return VK_SUCCESS;
        }
        int syncFdSignaled = mSyncHelper->wait(*fenceInfo->syncFd, /*timeout=*/0) == 0;
        return syncFdSignaled ? VK_SUCCESS : VK_NOT_READY;
    }
#endif

    return enc->vkGetFenceStatus(device, fence, /*doLock=*/true);
}

VkResult ResourceTracker::on_vkWaitForFences(void* context, VkResult, VkDevice device,
                                             uint32_t fenceCount, const VkFence* pFences,
                                             VkBool32 waitAll, uint64_t timeout) {
    if (!fenceCount) return VK_SUCCESS;

    VkEncoder* enc = (VkEncoder*)context;

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    bool hasSpecialFence = false;
    for (uint32_t i = 0; i < fenceCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_fence, info, pFences[i]);
        if (isNoopFence(info)) {
            if (!waitAll) {
                return VK_SUCCESS;
            }
            hasSpecialFence = true;
        } else if (info && info->syncFd) {
            hasSpecialFence = true;
        }
    }

    if (!hasSpecialFence) {
        return enc->vkWaitForFences(device, fenceCount, pFences, waitAll, timeout,
                                    true /* do lock */);
    }

    std::vector<int> fencesExternalSyncFds;
    std::vector<VkFence> fencesNonExternal;
    fencesExternalSyncFds.reserve(fenceCount);
    fencesNonExternal.reserve(fenceCount);

    for (uint32_t i = 0; i < fenceCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_fence, info, pFences[i]);
        if (!info) continue;
        if (isNoopFence(info)) {
            if (!waitAll) {
                return VK_SUCCESS;
            }
            continue;
        }
        if (info->syncFd) {
            if (*info->syncFd >= 0) {
                fencesExternalSyncFds.push_back(*info->syncFd);
            }
        } else {
            fencesNonExternal.push_back(pFences[i]);
        }
    }

    for (auto fd : fencesExternalSyncFds) {
        MESA_TRACE_SCOPE("Waiting on sync fd: %d", fd);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        int waitMs = (timeout == UINT64_MAX)
                         ? -1
                         : static_cast<int>(
                               std::min<uint64_t>(DIV_ROUND_UP(timeout, 1000000ULL), INT32_MAX));
        mSyncHelper->wait(fd, waitMs);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        uint64_t timeTaken =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        if (timeout != UINT64_MAX) {
            if (timeTaken >= timeout) {
                return VK_TIMEOUT;
            }
            timeout -= timeTaken;
        }
        MESA_TRACE_SCOPE("Done waiting on sync fd: %d", fd);

#if GFXSTREAM_SYNC_DEBUG
        mSyncHelper->debugPrint(fd);
#endif
    }

    if (!fencesNonExternal.empty()) {
        MESA_TRACE_SCOPE("vkWaitForFences to host");
        return enc->vkWaitForFences(device, fencesNonExternal.size(), fencesNonExternal.data(),
                                    waitAll, timeout, true /* do lock */);
    }

    return VK_SUCCESS;
#else
    return enc->vkWaitForFences(device, fenceCount, pFences, waitAll, timeout, true /* do lock */);
#endif
}

VkResult ResourceTracker::on_vkCreateDescriptorPool(void* context, VkResult, VkDevice device,
                                                    const VkDescriptorPoolCreateInfo* pCreateInfo,
                                                    const VkAllocationCallbacks* pAllocator,
                                                    VkDescriptorPool* pDescriptorPool) {
    VkEncoder* enc = (VkEncoder*)context;

    VkResult res = enc->vkCreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool,
                                               true /* do lock */);

    if (res != VK_SUCCESS) return res;

    VkDescriptorPool pool = *pDescriptorPool;

    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_descriptor_pool, dp, pool);
    dp->allocInfo = allocDeviceSubobject<DescriptorPoolAllocationInfo>(
        dev ? &dev->base : nullptr, pAllocator, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    if (!dp->allocInfo) {
        enc->vkDestroyDescriptorPool(device, pool, pAllocator, true /* do lock */);
        *pDescriptorPool = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    dp->allocInfo->device = device;
    dp->allocInfo->createFlags = pCreateInfo->flags;
    dp->allocInfo->maxSets = pCreateInfo->maxSets;
    dp->allocInfo->usedSets = 0;

    dp->allocInfo->descriptorCountInfo.reserve(pCreateInfo->poolSizeCount);
    for (uint32_t i = 0; i < pCreateInfo->poolSizeCount; ++i) {
        dp->allocInfo->descriptorCountInfo.push_back({
            pCreateInfo->pPoolSizes[i].type,
            pCreateInfo->pPoolSizes[i].descriptorCount,
            0, /* used */
        });
    }

    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
        dp->allocInfo->freePoolIds.resize(pCreateInfo->maxSets);
        uint32_t count = pCreateInfo->maxSets;
        enc->vkCollectDescriptorPoolIdsGOOGLE(device, pool, &count,
                                              dp->allocInfo->freePoolIds.data(), true /* do lock */);
    }

    return res;
}

void ResourceTracker::on_vkDestroyDescriptorPool(void* context, VkDevice device,
                                                 VkDescriptorPool descriptorPool,
                                                 const VkAllocationCallbacks* pAllocator) {
    if (!descriptorPool) return;

    VkEncoder* enc = (VkEncoder*)context;

    clearDescriptorPoolAndUnregisterDescriptorSets(context, device, descriptorPool);

    enc->vkDestroyDescriptorPool(device, descriptorPool, pAllocator, true /* do lock */);
}

VkResult ResourceTracker::on_vkResetEvent(void* context, VkResult, VkDevice device, VkEvent event) {
    VkEncoder* enc = (VkEncoder*)context;
    if (supportsAsyncResets()) {
        // vkResetEvent() can only fail with out of memory errors: skip the host round trip.
        enc->vkResetEventAsyncGOOGLE(device, event, true /* do lock */);
        return VK_SUCCESS;
    }
    return enc->vkResetEvent(device, event, true /* do lock */);
}

VkResult ResourceTracker::on_vkResetDescriptorPool(void* context, VkResult, VkDevice device,
                                                   VkDescriptorPool descriptorPool,
                                                   VkDescriptorPoolResetFlags flags) {
    if (!descriptorPool) return VK_ERROR_INITIALIZATION_FAILED;

    VkEncoder* enc = (VkEncoder*)context;

    VkResult res = VK_SUCCESS;
    if (supportsAsyncResets()) {
        // vkResetDescriptorPool() always succeeds: skip the host round trip for the result.
        enc->vkResetDescriptorPoolAsyncGOOGLE(device, descriptorPool, flags, true /* do lock */);
    } else {
        res = enc->vkResetDescriptorPool(device, descriptorPool, flags, true /* do lock */);
    }

    if (res != VK_SUCCESS) return res;

    clearDescriptorPoolAndUnregisterDescriptorSets(context, device, descriptorPool);
    return res;
}

VkResult ResourceTracker::on_vkAllocateDescriptorSets(
    void* context, VkResult, VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
    VkDescriptorSet* pDescriptorSets) {
    VkEncoder* enc = (VkEncoder*)context;
    auto ci = pAllocateInfo;
    auto sets = pDescriptorSets;
    VK_FROM_HANDLE(gfxstream_vk_descriptor_pool, dp, ci->descriptorPool);
    DescriptorPoolAllocationInfo* allocInfo = dp ? dp->allocInfo : nullptr;
    if (!allocInfo) return VK_ERROR_INITIALIZATION_FAILED;

    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate && !hasVariableDescriptorCountLayout(ci)) {
        simple_mtx_lock(&allocInfo->poolLock);
        VkResult poolAllocResult = validateAndApplyVirtualDescriptorSetAllocation(device, ci, sets);
        simple_mtx_unlock(&allocInfo->poolLock);

        if (poolAllocResult != VK_SUCCESS) return poolAllocResult;

        for (uint32_t i = 0; i < ci->descriptorSetCount; ++i) {
            VkDescriptorSetLayout setLayout =
                gfxstream_vk_descriptor_set_from_handle(sets[i])->reified->setLayout;
            VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, dsl, setLayout);
            dsl->layoutInfo->refcount.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
            simple_mtx_lock(&allocInfo->poolLock);
            VkResult validateRes = validateDescriptorSetAllocation(ci);
            simple_mtx_unlock(&allocInfo->poolLock);
            if (validateRes != VK_SUCCESS) return validateRes;
        }

        VkResult allocRes = enc->vkAllocateDescriptorSets(device, ci, sets, true /* do lock */);

        if (allocRes != VK_SUCCESS) {
            for (uint32_t i = 0; i < ci->descriptorSetCount; ++i) {
                if (sets[i]) {
                    delete_gfxstream_vk_descriptor_set(device, sets[i], nullptr);
                    sets[i] = VK_NULL_HANDLE;
                }
            }
            return allocRes;
        }

        const auto* varDescCountAllocInfo =
            vk_find_struct_const(ci->pNext, DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

        simple_mtx_lock(&allocInfo->poolLock);
        for (uint32_t i = 0; i < ci->descriptorSetCount; ++i) {
            const uint32_t* varCount =
                (varDescCountAllocInfo && varDescCountAllocInfo->descriptorSetCount > i &&
                 varDescCountAllocInfo->pDescriptorCounts)
                    ? &varDescCountAllocInfo->pDescriptorCounts[i]
                    : nullptr;
            if (!fillDescriptorSetInfoForPool(ci->descriptorPool, ci->pSetLayouts[i], sets[i],
                                              /*allocatedFromPoolIds=*/false, varCount)) {
                for (uint32_t j = 0; j < i; ++j) {
                    removeDescriptorSetFromPool(sets[j], /*usePoolIds=*/false);
                    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
                        VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, dsl, ci->pSetLayouts[j]);
                        dsl->layoutInfo->refcount.fetch_sub(1, std::memory_order_acq_rel);
                    }
                }
                const bool canFree = (allocInfo->createFlags &
                                      VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT) != 0;
                simple_mtx_unlock(&allocInfo->poolLock);
                if (canFree) {
                    enc->vkFreeDescriptorSets(device, ci->descriptorPool, ci->descriptorSetCount,
                                              sets, true /* do lock */);
                } else {
                    for (uint32_t j = 0; j < ci->descriptorSetCount; ++j) {
                        delete_gfxstream_vk_descriptor_set(device, sets[j], nullptr);
                    }
                }
                for (uint32_t j = 0; j < ci->descriptorSetCount; ++j) {
                    sets[j] = VK_NULL_HANDLE;
                }
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            applyDescriptorSetAllocation(ci->descriptorPool, ci->pSetLayouts[i], varCount);
            if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
                VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, dsl, ci->pSetLayouts[i]);
                dsl->layoutInfo->refcount.fetch_add(1, std::memory_order_relaxed);
            }
        }
        simple_mtx_unlock(&allocInfo->poolLock);
    }

    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkFreeDescriptorSets(void* context, VkResult, VkDevice device,
                                                  VkDescriptorPool descriptorPool,
                                                  uint32_t descriptorSetCount,
                                                  const VkDescriptorSet* pDescriptorSets) {
    if (!descriptorPool || !descriptorSetCount || !pDescriptorSets) return VK_SUCCESS;

    VkEncoder* enc = (VkEncoder*)context;
    VK_FROM_HANDLE(gfxstream_vk_descriptor_pool, dp, descriptorPool);
    if (!dp || !dp->allocInfo) return VK_SUCCESS;

    DescriptorPoolAllocationInfo* allocInfo = dp->allocInfo;
    if (!(allocInfo->createFlags & VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT)) {
        return VK_SUCCESS;
    }

    VkDescriptorSet stackToFree[32];
    std::vector<VkDescriptorSet> heapToFree;
    VkDescriptorSet* toActuallyFree = stackToFree;
    if (descriptorSetCount > 32) {
        heapToFree.resize(descriptorSetCount);
        toActuallyFree = heapToFree.data();
    }
    uint32_t toFreeCount = 0;

    {
        simple_mtx_lock(&allocInfo->poolLock);
        const auto& allocedSets = allocInfo->allocedSets;
        for (uint32_t i = 0; i < descriptorSetCount; ++i) {
            VkDescriptorSet set = pDescriptorSets[i];
            if (!set) continue;
            if (allocedSets.find(set) == allocedSets.end()) {
                mesa_loge(
                    "%s: Warning: descriptor set %p not found in pool. Was this "
                    "double-freed?\n",
                    __func__, (void*)set);
                continue;
            }
            if (removeDescriptorSetFromPool(set,
                                            mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate)) {
                toActuallyFree[toFreeCount++] = set;
            }
        }
        simple_mtx_unlock(&allocInfo->poolLock);
    }

    if (!toFreeCount) return VK_SUCCESS;

    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
        for (uint32_t i = 0; i < toFreeCount; ++i) {
            VkDescriptorSetLayout setLayout =
                gfxstream_vk_descriptor_set_from_handle(toActuallyFree[i])->reified->setLayout;
            decDescriptorSetLayoutRef(context, device, setLayout, nullptr);
        }
        freeDescriptorSetsIfHostAllocated(enc, device, toFreeCount, toActuallyFree);
    } else {
        enc->vkFreeDescriptorSets(device, descriptorPool, toFreeCount, toActuallyFree,
                                  true /* do lock */);
    }
    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkCreateDescriptorSetLayout(
    void* context, VkResult, VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout) {
    VkEncoder* enc = (VkEncoder*)context;

    VkResult res = enc->vkCreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout,
                                                    true /* do lock */);

    if (res != VK_SUCCESS) return res;

    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, dsl, *pSetLayout);
    dsl->layoutInfo = allocDeviceSubobject<DescriptorSetLayoutInfo>(
        dev ? &dev->base : nullptr, pAllocator, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    if (!dsl->layoutInfo) {
        enc->vkDestroyDescriptorSetLayout(device, *pSetLayout, pAllocator, true /* do lock */);
        *pSetLayout = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    dsl->layoutInfo->bindings.reserve(pCreateInfo->bindingCount);
    for (uint32_t i = 0; i < pCreateInfo->bindingCount; ++i) {
        dsl->layoutInfo->bindings.push_back(pCreateInfo->pBindings[i]);
    }
    const auto* bindingFlagsCi =
        vk_find_struct_const(pCreateInfo->pNext, DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
    if (bindingFlagsCi && bindingFlagsCi->pBindingFlags) {
        const uint32_t flagsCount =
            std::min(bindingFlagsCi->bindingCount, pCreateInfo->bindingCount);
        dsl->layoutInfo->bindingFlags.assign(bindingFlagsCi->pBindingFlags,
                                             bindingFlagsCi->pBindingFlags + flagsCount);
    }
    dsl->layoutInfo->refcount.store(1, std::memory_order_relaxed);

    return res;
}

void ResourceTracker::on_vkUpdateDescriptorSets(void* context, VkDevice device,
                                                uint32_t descriptorWriteCount,
                                                const VkWriteDescriptorSet* pDescriptorWrites,
                                                uint32_t descriptorCopyCount,
                                                const VkCopyDescriptorSet* pDescriptorCopies) {
    VkEncoder* enc = (VkEncoder*)context;

    VkWriteDescriptorSet stackWrites[32];
    std::vector<VkWriteDescriptorSet> heapWrites;
    VkWriteDescriptorSet* transformedWrites = stackWrites;
    if (descriptorWriteCount > 32) {
        heapWrites.resize(descriptorWriteCount);
        transformedWrites = heapWrites.data();
    }

    size_t imageInfosNeeded = 0;
    size_t bufferInfosNeeded = 0;
    for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
        const VkWriteDescriptorSet& src = pDescriptorWrites[i];
        if (isDescriptorTypeImageInfo(src.descriptorType)) {
            imageInfosNeeded += src.descriptorCount;
        } else if (isDescriptorTypeBufferInfo(src.descriptorType)) {
            bufferInfosNeeded += src.descriptorCount;
        }
    }

    VkDescriptorImageInfo stackImageInfos[64];
    std::vector<VkDescriptorImageInfo> heapImageInfos;
    VkDescriptorImageInfo* transformedImageInfos = stackImageInfos;
    if (imageInfosNeeded > 64) {
        heapImageInfos.resize(imageInfosNeeded);
        transformedImageInfos = heapImageInfos.data();
    }

    VkDescriptorBufferInfo stackBufferInfos[64];
    std::vector<VkDescriptorBufferInfo> heapBufferInfos;
    VkDescriptorBufferInfo* transformedBufferInfos = stackBufferInfos;
    if (bufferInfosNeeded > 64) {
        heapBufferInfos.resize(bufferInfosNeeded);
        transformedBufferInfos = heapBufferInfos.data();
    }

    size_t imageInfoIndex = 0;
    size_t bufferInfoIndex = 0;
    for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
        VkWriteDescriptorSet dst = pDescriptorWrites[i];
        VkDescriptorType descType = dst.descriptorType;
        uint32_t count = dst.descriptorCount;

        if (isDescriptorTypeImageInfo(descType)) {
            dst.pBufferInfo = nullptr;
            dst.pTexelBufferView = nullptr;
            VkDescriptorImageInfo* outImg = &transformedImageInfos[imageInfoIndex];
            VK_FROM_HANDLE(gfxstream_vk_descriptor_set, ds, dst.dstSet);
            const ReifiedDescriptorSet* reified = ds ? ds->reified : nullptr;
            bool hasSampler = (descType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                               descType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
            uint32_t currBinding = dst.dstBinding;
            uint32_t arrOffset = dst.dstArrayElement;

            for (uint32_t j = 0; j < count; ++j) {
                if (reified) {
                    while (currBinding < reified->allWrites.bindings.size()) {
                        const auto* sparse = std::get_if<DescriptorBindingSparseWrite>(
                            &reified->allWrites.bindings[currBinding].elements);
                        if (!sparse) {
                            ++currBinding;
                            continue;
                        }
                        if (arrOffset >= sparse->descriptorCount) {
                            arrOffset -= sparse->descriptorCount;
                            ++currBinding;
                            continue;
                        }
                        break;
                    }
                }
                bool isImmutable =
                    reified && currBinding < reified->bindingIsImmutableSampler.size() &&
                    reified->bindingIsImmutableSampler[currBinding];

                VkDescriptorImageInfo info =
                    dst.pImageInfo ? dst.pImageInfo[j] : VkDescriptorImageInfo{};
                if (!hasSampler || isImmutable) {
                    info.sampler = VK_NULL_HANDLE;
                }
                if (descType == VK_DESCRIPTOR_TYPE_SAMPLER) {
                    info.imageView = VK_NULL_HANDLE;
                }
                outImg[j] = info;
                ++arrOffset;
            }
            dst.pImageInfo = outImg;
            imageInfoIndex += count;
        } else if (isDescriptorTypeBufferInfo(descType)) {
            dst.pImageInfo = nullptr;
            dst.pTexelBufferView = nullptr;
            if (!dst.pBufferInfo) {
                VkDescriptorBufferInfo* outBuf = &transformedBufferInfos[bufferInfoIndex];
                memset(outBuf, 0, sizeof(VkDescriptorBufferInfo) * count);
                dst.pBufferInfo = outBuf;
                bufferInfoIndex += count;
            }
        } else if (isDescriptorTypeBufferView(descType)) {
            dst.pImageInfo = nullptr;
            dst.pBufferInfo = nullptr;
        } else {
            dst.pImageInfo = nullptr;
            dst.pBufferInfo = nullptr;
            dst.pTexelBufferView = nullptr;
        }
        transformedWrites[i] = dst;
    }

    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
        for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
            VkDescriptorSet set = transformedWrites[i].dstSet;
            doEmulatedDescriptorWrite(&transformedWrites[i],
                                      gfxstream_vk_descriptor_set_from_handle(set)->reified);
        }

        for (uint32_t i = 0; i < descriptorCopyCount; ++i) {
            doEmulatedDescriptorCopy(
                &pDescriptorCopies[i],
                gfxstream_vk_descriptor_set_from_handle(pDescriptorCopies[i].srcSet)->reified,
                gfxstream_vk_descriptor_set_from_handle(pDescriptorCopies[i].dstSet)->reified);
        }
    } else {
        enc->vkUpdateDescriptorSets(device, descriptorWriteCount, transformedWrites,
                                    descriptorCopyCount, pDescriptorCopies, true /* do lock */);
    }
}

void ResourceTracker::on_vkDestroyImage(void* context, VkDevice device, VkImage image,
                                        const VkAllocationCallbacks* pAllocator) {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    {
        std::lock_guard<std::recursive_mutex> lock(mLock);  // do not guard encoder may cause
                                                            // deadlock b/243339973

        // Wait for any pending QSRIs to prevent a race between the Gfxstream host
        // potentially processing the below `vkDestroyImage()` from the VK encoder
        // command stream before processing a previously submitted
        // `VIRTIO_GPU_NATIVE_SYNC_VULKAN_QSRI_EXPORT` from the virtio-gpu command
        // stream which relies on the image existing.
        VK_FROM_HANDLE(gfxstream_vk_image, imageInfo, image);
        if (imageInfo) {
            for (int syncFd : imageInfo->pendingQsriSyncFds) {
                int syncWaitRet = mSyncHelper->wait(syncFd, 3000);
                if (syncWaitRet < 0) {
                    mesa_loge("%s: Failed to wait for pending QSRI sync: sterror: %s errno: %d",
                              __func__, strerror(errno), errno);
                }

#if GFXSTREAM_SYNC_DEBUG
                mSyncHelper->debugPrint(syncFd);
#endif
                mSyncHelper->close(syncFd);
            }
            imageInfo->pendingQsriSyncFds.clear();
        }
    }
#endif
    VkEncoder* enc = (VkEncoder*)context;
    enc->vkDestroyImage(device, image, pAllocator, true /* do lock */);
}

void ResourceTracker::on_vkGetImageMemoryRequirements(void* context, VkDevice device, VkImage image,
                                                      VkMemoryRequirements* pMemoryRequirements) {
    VK_FROM_HANDLE(gfxstream_vk_image, info, image);
    if (!info) return;

    if (info->baseRequirementsKnown) {
        *pMemoryRequirements = info->baseRequirements;
        return;
    }

    VkEncoder* enc = (VkEncoder*)context;
    enc->vkGetImageMemoryRequirements(device, image, pMemoryRequirements, true /* do lock */);

    transformImageMemoryRequirementsForGuest(image, pMemoryRequirements);

    info->baseRequirements = *pMemoryRequirements;
    info->baseRequirementsKnown = true;
}

void ResourceTracker::on_vkGetImageMemoryRequirements2(void* context, VkDevice device,
                                                       const VkImageMemoryRequirementsInfo2* pInfo,
                                                       VkMemoryRequirements2* pMemoryRequirements) {
    VkEncoder* enc = (VkEncoder*)context;
    enc->vkGetImageMemoryRequirements2(device, pInfo, pMemoryRequirements, true /* do lock */);
    transformImageMemoryRequirements2ForGuest(pInfo->image, pMemoryRequirements);
}

VkResult ResourceTracker::on_vkGetImageDrmFormatModifierPropertiesEXT(
    void* context, VkResult, VkDevice device, VkImage image,
    VkImageDrmFormatModifierPropertiesEXT* pProperties) {
#ifdef LINUX_GUEST_BUILD
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    if (!dev) return VK_ERROR_UNKNOWN;
    if (doImageDrmFormatModifierEmulation(dev->internal_physical_device)) {
        // This is the only format modifier that will be used in emulation
        pProperties->drmFormatModifier = DRM_FORMAT_MOD_LINEAR;
        return VK_SUCCESS;
    } else {
        // Passthrough to host
        VkEncoder* enc = (VkEncoder*)context;
        return enc->vkGetImageDrmFormatModifierPropertiesEXT(device, image, pProperties,
                                                             true /* do lock */);
    }
#else
    (void)context;
    (void)device;
    (void)image;
    (void)pProperties;
    return VK_ERROR_INCOMPATIBLE_DRIVER;
#endif
}

void ResourceTracker::on_vkGetImageSubresourceLayout(void* context, VkDevice device, VkImage image,
                                                     const VkImageSubresource* pSubresource,
                                                     VkSubresourceLayout* pLayout) {
    if (!pSubresource || !pLayout) return;

    VkEncoder* enc = (VkEncoder*)context;
#if defined(LINUX_GUEST_BUILD)
    VK_FROM_HANDLE(gfxstream_vk_image, img, image);
    if (img && img->emulatedDrmFormatModifier) {
        if (img->hasExplicitDrmModifier) {
            *pLayout = img->explicitPlaneLayout;
        } else {
            *pLayout = {};
            fillEmulatedLinearSubresourceLayout(img->createInfo, pLayout);
        }
        return;
    }
#endif
    enc->vkGetImageSubresourceLayout(device, image, pSubresource, pLayout, true /* do lock */);
}

VkResult ResourceTracker::on_vkBindImageMemory(void* context, VkResult, VkDevice device,
                                               VkImage image, VkDeviceMemory memory,
                                               VkDeviceSize memoryOffset) {
    VkEncoder* enc = (VkEncoder*)context;
    if (!memory || !image) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return enc->vkBindImageMemory(device, image, memory, memoryOffset, true /* do lock */);
}

VkResult ResourceTracker::on_vkBindImageMemory2(void* context, VkResult, VkDevice device,
                                                uint32_t bindingCount,
                                                const VkBindImageMemoryInfo* pBindInfos) {
    VkEncoder* enc = (VkEncoder*)context;

    if (bindingCount < 1 || !pBindInfos) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    for (uint32_t i = 0; i < bindingCount; i++) {
        const VkBindImageMemoryInfo& bimi = pBindInfos[i];
        if (!bimi.image) {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }

    VkResult res = enc->vkBindImageMemory2(device, bindingCount, pBindInfos, true /* do lock */);

    // The host decoder historically treated the `const VkBindMemoryStatus*` as
    // an input only parameter and did not return individual VkResult values in
    // the `VkBindMemoryStatus`s. It would be a backwards incompatible change
    // to update that now. Instead just propagate the overall result to each
    // individual `VkBindMemoryStatus::pResult` as the spec says that the
    // `vkBindImageMemory2()` call must return one of the failure results and
    // that all images are left in an indeterminate state and must not be used:
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkBindImageMemory2.html.
    for (uint32_t i = 0; i < bindingCount; i++) {
        const VkBindMemoryStatus* bindStatus =
            vk_find_struct_const(&pBindInfos[i], BIND_MEMORY_STATUS);
        if (bindStatus && bindStatus->pResult) {
            *bindStatus->pResult = res;
        }
    }
    return res;
}

VkResult ResourceTracker::on_vkBindImageMemory2KHR(void* context, VkResult input_result,
                                                   VkDevice device, uint32_t bindingCount,
                                                   const VkBindImageMemoryInfo* pBindInfos) {
    return on_vkBindImageMemory2(context, input_result, device, bindingCount, pBindInfos);
}

VkResult ResourceTracker::on_vkBindBufferMemory2(void* context, VkResult, VkDevice device,
                                                 uint32_t bindInfoCount,
                                                 const VkBindBufferMemoryInfo* pBindInfos) {
    VkEncoder* enc = (VkEncoder*)context;

    VkResult res = enc->vkBindBufferMemory2(device, bindInfoCount, pBindInfos, true /* do lock */);

    // The host decoder historically treated the `const VkBindMemoryStatus*` as
    // an input only parameter and did not return individual VkResult values in
    // the `VkBindMemoryStatus`s. It would be a backwards incompatible change
    // to update that now. Instead just propagate the overall result to each
    // individual `VkBindMemoryStatus::pResult` as the spec says that the
    // `vkBindBufferMemory2()` call must return one of the failure results and
    // that all buffers are left in an indeterminate state and must not be used:
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkBindBufferMemory2.html
    for (uint32_t i = 0; i < bindInfoCount; i++) {
        const VkBindMemoryStatus* bindStatus =
            vk_find_struct_const(&pBindInfos[i], BIND_MEMORY_STATUS);
        if (bindStatus && bindStatus->pResult) {
            *bindStatus->pResult = res;
        }
    }
    return res;
}

VkResult ResourceTracker::on_vkBindBufferMemory2KHR(void* context, VkResult input_result,
                                                    VkDevice device, uint32_t bindInfoCount,
                                                    const VkBindBufferMemoryInfo* pBindInfos) {
    return on_vkBindBufferMemory2(context, input_result, device, bindInfoCount, pBindInfos);
}

VkResult ResourceTracker::on_vkCreateBuffer(void* context, VkResult, VkDevice device,
                                            const VkBufferCreateInfo* pCreateInfo,
                                            const VkAllocationCallbacks* pAllocator,
                                            VkBuffer* pBuffer) {
    VkEncoder* enc = (VkEncoder*)context;

    VkBufferCreateInfo localCreateInfo = vk_make_orphan_copy(*pCreateInfo);
    vk_struct_chain_iterator structChainIter = vk_make_chain_iterator(&localCreateInfo);
    VkExternalMemoryBufferCreateInfo localExtBufCi = {};

    const VkExternalMemoryBufferCreateInfo* extBufCiPtr =
        vk_find_struct_const(pCreateInfo, EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
    if (extBufCiPtr) {
        localExtBufCi = vk_make_orphan_copy(*extBufCiPtr);
#if defined(LINUX_GUEST_BUILD)
        localExtBufCi.handleTypes &= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif
        if (localExtBufCi.handleTypes) {
            vk_append_struct(&structChainIter, &localExtBufCi);
        }
    }

    VkBufferOpaqueCaptureAddressCreateInfo localCapAddrCi;
    const VkBufferOpaqueCaptureAddressCreateInfo* pCapAddrCi =
        vk_find_struct_const(pCreateInfo, BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO);
    if (pCapAddrCi) {
        localCapAddrCi = vk_make_orphan_copy(*pCapAddrCi);
        vk_append_struct(&structChainIter, &localCapAddrCi);
    }

    VkBufferDeviceAddressCreateInfoEXT localDevAddrCi;
    const VkBufferDeviceAddressCreateInfoEXT* pDevAddrCi =
        vk_find_struct_const(pCreateInfo, BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT);
    if (pDevAddrCi) {
        localDevAddrCi = vk_make_orphan_copy(*pDevAddrCi);
        vk_append_struct(&structChainIter, &localDevAddrCi);
    }

    VkBufferUsageFlags2CreateInfo localBufUsageFlags2Ci;
    const VkBufferUsageFlags2CreateInfo* pBufUsageFlags2Ci =
        vk_find_struct_const(pCreateInfo, BUFFER_USAGE_FLAGS_2_CREATE_INFO);
    if (pBufUsageFlags2Ci) {
        localBufUsageFlags2Ci = vk_make_orphan_copy(*pBufUsageFlags2Ci);
        vk_append_struct(&structChainIter, &localBufUsageFlags2Ci);
    }

#ifdef VK_USE_PLATFORM_FUCHSIA
    std::optional<zx::vmo> vmo;
    bool isSysmemBackedMemory = false;

    if (extBufCiPtr &&
        (extBufCiPtr->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA)) {
        isSysmemBackedMemory = true;
    }

    const VkBufferCollectionBufferCreateInfoFUCHSIA* extBufferCollectionPtr =
        vk_find_struct_const(pCreateInfo, BUFFER_COLLECTION_BUFFER_CREATE_INFO_FUCHSIA);

    if (extBufferCollectionPtr) {
        const auto& collection =
            *reinterpret_cast<fidl::WireSyncClient<fuchsia_sysmem::BufferCollection>*>(
                gfxstream_vk_buffer_collection_fuchsia_to_host_u64(
                    extBufferCollectionPtr->collection));
        uint32_t index = extBufferCollectionPtr->index;

        auto result = collection->WaitForBuffersAllocated();
        if (result.ok() && result->status == ZX_OK) {
            auto& info = result->buffer_collection_info;
            if (index < info.buffer_count) {
                vmo = std::make_optional<zx::vmo>(std::move(info.buffers[index].vmo));
            }
        } else {
            mesa_loge("WaitForBuffersAllocated failed: %d %d", result.status(),
                      GET_STATUS_SAFE(result, status));
        }

        if (vmo && vmo->is_valid()) {
            fidl::Arena arena;
            fuchsia_hardware_goldfish::wire::CreateBuffer2Params createParams(arena);
            createParams.set_size(arena, pCreateInfo->size)
                .set_memory_property(fuchsia_hardware_goldfish::wire::kMemoryPropertyDeviceLocal);

            auto result = mControlDevice->CreateBuffer2(std::move(*vmo), createParams);
            if (!result.ok() ||
                (result->is_error() != ZX_OK && result->error_value() != ZX_ERR_ALREADY_EXISTS)) {
                mesa_loge("CreateBuffer2 failed: %d:%d", result.status(),
                          GET_STATUS_SAFE(result, error_value()));
            }
            isSysmemBackedMemory = true;
        }
    }
#endif  // VK_USE_PLATFORM_FUCHSIA

    VkResult res;
    VkMemoryRequirements memReqs;

    if (supportsCreateResourcesWithRequirements()) {
        res = enc->vkCreateBufferWithRequirementsGOOGLE(device, &localCreateInfo, pAllocator,
                                                        pBuffer, &memReqs, true /* do lock */);
    } else {
        res =
            enc->vkCreateBuffer(device, &localCreateInfo, pAllocator, pBuffer, true /* do lock */);
    }

    if (res != VK_SUCCESS) return res;

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    if (mCaps.vulkanCapset.colorBufferMemoryIndex == 0xFFFFFFFF) {
        mCaps.vulkanCapset.colorBufferMemoryIndex = getColorBufferMemoryIndex(context, device);
    }
    if ((localExtBufCi.handleTypes &
         (VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID |
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT))) {
        updateMemoryTypeBits(&memReqs.memoryTypeBits, mCaps.vulkanCapset.colorBufferMemoryIndex);
    }
#endif

    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_buffer, info, *pBuffer);
    if (!info) return VK_ERROR_INITIALIZATION_FAILED;

    vk_buffer_init(&dev->base, &info->base, pCreateInfo);

    info->createInfo = localCreateInfo;
    info->createInfo.pNext = nullptr;

    if (supportsCreateResourcesWithRequirements()) {
        info->baseRequirementsKnown = true;
        info->baseRequirements = memReqs;
    }

    if (localExtBufCi.handleTypes) {
        info->external = true;
        info->externalCreateInfo = localExtBufCi;
    }

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (isSysmemBackedMemory) {
        info->isSysmemBackedMemory = true;
    }
#endif

    return res;
}

void ResourceTracker::on_vkDestroyBuffer(void* context, VkDevice device, VkBuffer buffer,
                                         const VkAllocationCallbacks* pAllocator) {
    VkEncoder* enc = (VkEncoder*)context;
    enc->vkDestroyBuffer(device, buffer, pAllocator, true /* do lock */);
}

void ResourceTracker::on_vkGetBufferMemoryRequirements(void* context, VkDevice device,
                                                       VkBuffer buffer,
                                                       VkMemoryRequirements* pMemoryRequirements) {
    VK_FROM_HANDLE(gfxstream_vk_buffer, info, buffer);
    if (!info) return;

    if (info->baseRequirementsKnown) {
        *pMemoryRequirements = info->baseRequirements;
        return;
    }

    VkEncoder* enc = (VkEncoder*)context;
    enc->vkGetBufferMemoryRequirements(device, buffer, pMemoryRequirements, true /* do lock */);

    info->baseRequirements = *pMemoryRequirements;
    info->baseRequirementsKnown = true;
}

void ResourceTracker::on_vkGetBufferMemoryRequirements2(
    void* context, VkDevice device, const VkBufferMemoryRequirementsInfo2* pInfo,
    VkMemoryRequirements2* pMemoryRequirements) {
    VkEncoder* enc = (VkEncoder*)context;
    enc->vkGetBufferMemoryRequirements2(device, pInfo, pMemoryRequirements, true /* do lock */);
    transformBufferMemoryRequirements2ForGuest(pInfo->buffer, pMemoryRequirements);
}

VkResult ResourceTracker::on_vkCreateSemaphore(void* context, VkResult input_result,
                                               VkDevice device,
                                               const VkSemaphoreCreateInfo* pCreateInfo,
                                               const VkAllocationCallbacks* pAllocator,
                                               VkSemaphore* pSemaphore) {
    (void)input_result;
    VkEncoder* enc = (VkEncoder*)context;

    VkSemaphoreCreateInfo finalCreateInfo = *pCreateInfo;

    const VkExportSemaphoreCreateInfoKHR* exportSemaphoreInfoPtr =
        vk_find_struct_const(pCreateInfo, EXPORT_SEMAPHORE_CREATE_INFO);

    const VkSemaphoreTypeCreateInfo* typeCi =
        vk_find_struct_const(pCreateInfo, SEMAPHORE_TYPE_CREATE_INFO);

#ifdef VK_USE_PLATFORM_FUCHSIA
    bool exportEvent =
        exportSemaphoreInfoPtr && (exportSemaphoreInfoPtr->handleTypes &
                                   VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_ZIRCON_EVENT_BIT_FUCHSIA);

    if (exportEvent) {
        finalCreateInfo.pNext = nullptr;
        // If we have timeline semaphores externally, leave it there.
        if (typeCi) finalCreateInfo.pNext = typeCi;
    }
#elif defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    bool exportSyncFd = exportSemaphoreInfoPtr && (exportSemaphoreInfoPtr->handleTypes &
                                                   VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);

    if (exportSyncFd) {
        finalCreateInfo.pNext = nullptr;
        // If we have timeline semaphores externally, leave it there.
        if (typeCi) finalCreateInfo.pNext = typeCi;
    }
#endif
    input_result = enc->vkCreateSemaphore(device, &finalCreateInfo, pAllocator, pSemaphore,
                                          true /* do lock */);
    if (input_result != VK_SUCCESS) return input_result;

    zx_handle_t event_handle = ZX_HANDLE_INVALID;

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (exportEvent) {
        zx_event_create(0, &event_handle);
    }
#endif

    VK_FROM_HANDLE(gfxstream_vk_semaphore, info, *pSemaphore);
    if (!info) return VK_ERROR_INITIALIZATION_FAILED;

    info->device = device;
    info->eventHandle = event_handle;
#ifdef VK_USE_PLATFORM_FUCHSIA
    info->eventKoid = getEventKoid(info->eventHandle);
#endif
    info->isTimeline = typeCi && (typeCi->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE);

#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    if (exportSyncFd && !mFeatureInfo.hasVirtioGpuNativeSync) {
#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
        std::lock_guard<std::recursive_mutex> lock(mLock);
        ensureSyncDeviceFd();

        int syncFd = -1;
        goldfish_sync_queue_work(
            mSyncDeviceFd, gfxstream_vk_semaphore_to_host_u64(*pSemaphore) /* the handle */,
            GOLDFISH_SYNC_VULKAN_SEMAPHORE_SYNC /* thread handle (doubling as type field) */,
            &syncFd);
        info->syncFd.emplace(syncFd);
#endif
    }
#endif

    return VK_SUCCESS;
}

void ResourceTracker::on_vkDestroySemaphore(void* context, VkDevice device, VkSemaphore semaphore,
                                            const VkAllocationCallbacks* pAllocator) {
    VkEncoder* enc = (VkEncoder*)context;
    enc->vkDestroySemaphore(device, semaphore, pAllocator, true /* do lock */);
}

// https://docs.vulkan.org/refpages/latest/refpages/source/vkGetSemaphoreFdKHR.html
// Each call to vkGetSemaphoreFdKHR must create a new file descriptor and transfer ownership
// of it to the application. To avoid leaking resources, the application must release ownership
// of the file descriptor when it is no longer needed.
VkResult ResourceTracker::on_vkGetSemaphoreFdKHR(void* context, VkResult, VkDevice device,
                                                 const VkSemaphoreGetFdInfoKHR* pGetFdInfo,
                                                 int* pFd) {
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    VkEncoder* enc = (VkEncoder*)context;
    bool getSyncFd = pGetFdInfo->handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

    if (getSyncFd) {
        if (mCaps.params[kParamFencePassing] && mCaps.vulkanCapset.externalSync) {
            uint64_t syncId = ++mAtomicId;
            int64_t osHandle = -1;

            VkResult result = enc->vkGetSemaphoreGOOGLE(device, pGetFdInfo->semaphore, syncId,
                                                        true /* do lock */);
            if (result != VK_SUCCESS) {
                mesa_loge("unable to get the semaphore");
                return result;
            }

            result = acquireSync(syncId, osHandle);
            if (result != VK_SUCCESS) {
                mesa_loge("unable to create host sync object");
                return result;
            }

            *pFd = (int)osHandle;
            return VK_SUCCESS;
        } else if (mFeatureInfo.hasVirtioGpuNativeSync) {
            uint64_t syncId = ++mAtomicId;
            int64_t osHandle = -1;

            VkResult result = enc->vkGetSemaphoreGOOGLE(device, pGetFdInfo->semaphore, syncId,
                                                        true /* do lock */);
            if (result != VK_SUCCESS) {
                mesa_loge("unable to get the semaphore");
                return result;
            }

            result = createFence(device, syncId, osHandle);
            if (result != VK_SUCCESS) {
                mesa_loge("Failed to export sync handle for semaphore!");
                return result;
            }

            *pFd = (int)osHandle;
            return VK_SUCCESS;
        } else {
            VK_FROM_HANDLE(gfxstream_vk_semaphore, semInfo, pGetFdInfo->semaphore);
            if (!semInfo) return VK_ERROR_OUT_OF_HOST_MEMORY;
            *pFd = mSyncHelper->dup(semInfo->syncFd.value_or(-1));
            return VK_SUCCESS;
        }
    } else {
        // opaque fd
        int hostFd = 0;
        int32_t size = 0;
        VkResult result = enc->vkGetSemaphoreFdKHR(device, pGetFdInfo, &hostFd, true /* do lock */);
        if (result != VK_SUCCESS) {
            return result;
        }
        *pFd = os_create_anonymous_file(size, "vk_opaque_fd");
        int write_result = write(*pFd, &hostFd, sizeof(hostFd));
        (void)write_result;
        return VK_SUCCESS;
    }
#else
    (void)context;
    (void)device;
    (void)pGetFdInfo;
    (void)pFd;
    return VK_ERROR_INCOMPATIBLE_DRIVER;
#endif
}

VkResult ResourceTracker::on_vkImportSemaphoreFdKHR(
    void* context, VkResult input_result, VkDevice device,
    const VkImportSemaphoreFdInfoKHR* pImportSemaphoreFdInfo) {
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    VkEncoder* enc = (VkEncoder*)context;
    if (input_result != VK_SUCCESS) {
        return input_result;
    }

    if (pImportSemaphoreFdInfo->handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT) {
        VK_FROM_HANDLE(gfxstream_vk_semaphore, info, pImportSemaphoreFdInfo->semaphore);
        if (!info) return VK_ERROR_OUT_OF_HOST_MEMORY;

        if (info->syncFd.value_or(-1) >= 0) {
            mSyncHelper->close(info->syncFd.value());
        }

        info->syncFd.emplace(pImportSemaphoreFdInfo->fd);

        return VK_SUCCESS;
    } else {
        int fd = pImportSemaphoreFdInfo->fd;
        int err = lseek(fd, 0, SEEK_SET);
        if (err == -1) {
            mesa_loge("lseek fail on import semaphore");
        }
        int hostFd = 0;
        int read_result = read(fd, &hostFd, sizeof(hostFd));
        (void)read_result;
        VkImportSemaphoreFdInfoKHR tmpInfo = *pImportSemaphoreFdInfo;
        tmpInfo.fd = hostFd;
        VkResult result = enc->vkImportSemaphoreFdKHR(device, &tmpInfo, true /* do lock */);
        mSyncHelper->close(fd);
        return result;
    }
#else
    (void)context;
    (void)input_result;
    (void)device;
    (void)pImportSemaphoreFdInfo;
    return VK_ERROR_INCOMPATIBLE_DRIVER;
#endif
}

VkResult ResourceTracker::on_vkGetMemoryFdPropertiesKHR(
    void* context, VkResult, VkDevice device, VkExternalMemoryHandleTypeFlagBits handleType, int fd,
    VkMemoryFdPropertiesKHR* pMemoryFdProperties) {
#ifdef LINUX_GUEST_BUILD
    if (!(handleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)) {
        mesa_loge("%s: VK_KHR_external_memory_fd behavior not defined for handleType: 0x%x\n",
                  __func__, handleType);
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    if (!device) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    (void)fd;

    if (mCaps.vulkanCapset.colorBufferMemoryIndex == 0xFFFFFFFF) {
        mCaps.vulkanCapset.colorBufferMemoryIndex = getColorBufferMemoryIndex(context, device);
    }

    updateMemoryTypeBits(&pMemoryFdProperties->memoryTypeBits,
                         mCaps.vulkanCapset.colorBufferMemoryIndex);

    return VK_SUCCESS;
#else
    (void)context;
    (void)device;
    (void)handleType;
    (void)fd;
    (void)pMemoryFdProperties;
    return VK_ERROR_INCOMPATIBLE_DRIVER;
#endif
}

VkResult ResourceTracker::on_vkGetMemoryFdKHR(void* context, VkResult, VkDevice device,
                                              const VkMemoryGetFdInfoKHR* pGetFdInfo, int* pFd) {
#ifdef LINUX_GUEST_BUILD
    if (!pGetFdInfo) return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (!pGetFdInfo->memory) return VK_ERROR_OUT_OF_HOST_MEMORY;

    if (!(pGetFdInfo->handleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)) {
        mesa_loge("%s: Export operation not defined for handleType: 0x%x\n", __func__,
                  pGetFdInfo->handleType);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (!device) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VK_FROM_HANDLE(gfxstream_vk_device_memory, info, pGetFdInfo->memory);
    if (!info || !info->blobPtr) {
        mesa_loge("%s: VkDeviceMemory does not have a resource available for export.\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VirtGpuExternalHandle handle{};
    int ret = info->blobPtr->exportBlob(handle);
    if (ret != 0 || handle.osHandle < 0) {
        mesa_loge("%s: Failed to export host resource to FD.\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *pFd = handle.osHandle;
    return VK_SUCCESS;
#else
    (void)context;
    (void)device;
    (void)pGetFdInfo;
    (void)pFd;
    return VK_ERROR_INCOMPATIBLE_DRIVER;
#endif
}

void ResourceTracker::flushCommandBufferPendingCommandsBottomUp(
    void* context, VkQueue queue, const VkCommandBuffer* workingSet, size_t workingSetCount) {
    if (!workingSetCount) return;

    size_t nextLevelCount = 0;
    for (size_t idx = 0; idx < workingSetCount; ++idx) {
        VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, workingSet[idx]);
        if (cb) {
            nextLevelCount += util_dynarray_num_elements(&cb->subObjects, VkCommandBuffer);
        }
    }

    if (nextLevelCount > 0) {
        VkCommandBuffer stackNextLevel[32];
        std::vector<VkCommandBuffer> heapNextLevel;
        VkCommandBuffer* nextLevel = stackNextLevel;
        if (nextLevelCount > 32) {
            heapNextLevel.resize(nextLevelCount);
            nextLevel = heapNextLevel.data();
        }
        size_t pos = 0;
        for (size_t idx = 0; idx < workingSetCount; ++idx) {
            VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, workingSet[idx]);
            if (cb) {
                util_dynarray_foreach(&cb->subObjects, VkCommandBuffer, elem) {
                    nextLevel[pos++] = *elem;
                }
            }
        }
        flushCommandBufferPendingCommandsBottomUp(context, queue, nextLevel, pos);
    }

    for (size_t idx = 0; idx < workingSetCount; ++idx) {
        VkCommandBuffer cmdbuf = workingSet[idx];
        VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, cmdbuf);

        if (!cb->privateStream) continue;

        unsigned char* writtenPtr = 0;
        size_t written = 0;
        CommandBufferStagingStream* cmdBufStream =
            static_cast<CommandBufferStagingStream*>(cb->privateStream);
        cmdBufStream->getWritten(&writtenPtr, &written);

        if (!written) continue;

        VkEncoder* enc = (VkEncoder*)context;
        VkDeviceMemory deviceMemory = cmdBufStream->getDeviceMemory();
        VkDeviceSize dataOffset = 0;
        if (mFeatureInfo.hasVulkanAuxCommandMemory) {
            deviceMemoryTransform_tohost(&deviceMemory, 1 /*memoryCount*/, &dataOffset,
                                         1 /*offsetCount*/, nullptr /*size*/, 0 /*sizeCount*/,
                                         nullptr /*typeIndex*/, 0 /*typeIndexCount*/,
                                         nullptr /*typeBits*/, 0 /*typeBitCounts*/);

            cmdBufStream->markFlushing();
            enc->vkQueueFlushCommandsFromAuxMemoryGOOGLE(queue, cmdbuf, deviceMemory, dataOffset,
                                                         written, true /*do lock*/);
        } else {
            enc->vkQueueFlushCommandsGOOGLE(queue, cmdbuf, written, (const void*)writtenPtr,
                                            true /* do lock */);
        }
        cmdBufStream->reset();
    }
}

uint32_t ResourceTracker::syncEncodersForQueue(VkQueue queue, VkEncoder* currentEncoder) {
    if (!supportsAsyncQueueSubmit()) {
        return 0;
    }

    VK_FROM_HANDLE(gfxstream_vk_queue, q, queue);
    if (!q) return 0;

    auto lastEncoder = q->lastUsedEncoder;

    if (lastEncoder == currentEncoder) return 0;

    currentEncoder->incRef();

    q->lastUsedEncoder = currentEncoder;

    if (!lastEncoder) return 0;

    auto oldSeq = q->sequenceNumber;
    q->sequenceNumber += 2;
    lastEncoder->vkQueueHostSyncGOOGLE(queue, false, oldSeq + 1, true /* do lock */);
    lastEncoder->flush();
    currentEncoder->vkQueueHostSyncGOOGLE(queue, true, oldSeq + 2, true /* do lock */);

    if (lastEncoder->decRef()) {
        q->lastUsedEncoder = nullptr;
    }

    return 0;
}

template <class VkSubmitInfoType>
void ResourceTracker::flushStagingStreams(void* context, VkQueue queue, uint32_t submitCount,
                                          const VkSubmitInfoType* pSubmits) {
    size_t totalCmdBufs = 0;
    for (uint32_t i = 0; i < submitCount; ++i) {
        totalCmdBufs += getCommandBufferCount(pSubmits[i]);
    }
    if (!totalCmdBufs) return;

    VkCommandBuffer stackToFlush[32];
    std::vector<VkCommandBuffer> heapToFlush;
    VkCommandBuffer* toFlush = stackToFlush;
    if (totalCmdBufs > 32) {
        heapToFlush.resize(totalCmdBufs);
        toFlush = heapToFlush.data();
    }

    size_t idx = 0;
    for (uint32_t i = 0; i < submitCount; ++i) {
        uint32_t cbCount = getCommandBufferCount(pSubmits[i]);
        for (uint32_t j = 0; j < cbCount; ++j) {
            toFlush[idx++] = getCommandBuffer(pSubmits[i], j);
        }
    }

    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
        if (hasAnyDirtyPendingDescriptorSets(toFlush, totalCmdBufs)) {
            std::unordered_set<VkDescriptorSet> pendingSets;
            collectDirtyPendingDescriptorSetsBottomUp(toFlush, totalCmdBufs, pendingSets);
            if (!pendingSets.empty()) {
                commitDescriptorSetUpdates(context, queue, pendingSets);
            }
        }
    }

    flushCommandBufferPendingCommandsBottomUp(context, queue, toFlush, totalCmdBufs);

    for (size_t i = 0; i < totalCmdBufs; ++i) {
        resetCommandBufferPendingTopology(toFlush[i]);
    }
}

VkResult ResourceTracker::on_vkQueueSubmit(void* context, VkResult input_result, VkQueue queue,
                                           uint32_t submitCount, const VkSubmitInfo* pSubmits,
                                           VkFence fence) {
    MESA_TRACE_SCOPE("on_vkQueueSubmit");
    EmitGuestAndHostTraceMarker((VkEncoder*)context);

    /* From the Vulkan 1.3.204 spec:
     *
     *    VUID-VkSubmitInfo-pNext-03240
     *
     *    "If the pNext chain of this structure includes a VkTimelineSemaphoreSubmitInfo structure
     *    and any element of pSignalSemaphores was created with a VkSemaphoreType of
     *    VK_SEMAPHORE_TYPE_TIMELINE, then its signalSemaphoreValueCount member must equal
     *    signalSemaphoreCount"
     *
     * Internally, Mesa WSI creates placeholder semaphores/fences (see FilterNoop* functions
     * in gfxstream_vk_private.cpp).  We don't want to forward that to the host, since there is
     * no host side Vulkan object associated with the placeholder sync objects.
     *
     * The way to test this behavior is Zink + glxgears, on Linux hosts.  It should fail without
     * this check.
     */
    for (uint32_t i = 0; i < submitCount; i++) {
        VkTimelineSemaphoreSubmitInfo* tssi =
            vk_find_struct(const_cast<VkSubmitInfo*>(&pSubmits[i]), TIMELINE_SEMAPHORE_SUBMIT_INFO);

        if (tssi) {
            uint32_t count = getSignalSemaphoreCount(pSubmits[i]);
            if (count != tssi->signalSemaphoreValueCount) {
                tssi->signalSemaphoreValueCount = count;
            }
        }
    }

    return on_vkQueueSubmitTemplate<VkSubmitInfo, VkSemaphore>(context, input_result, queue,
                                                               submitCount, pSubmits, fence);
}

VkResult ResourceTracker::on_vkQueueSubmit2(void* context, VkResult input_result, VkQueue queue,
                                            uint32_t submitCount, const VkSubmitInfo2* pSubmits,
                                            VkFence fence) {
    MESA_TRACE_SCOPE("on_vkQueueSubmit2");
    EmitGuestAndHostTraceMarker((VkEncoder*)context);
    return on_vkQueueSubmitTemplate<VkSubmitInfo2, VkSemaphoreSubmitInfo>(
        context, input_result, queue, submitCount, pSubmits, fence);
}

VkResult ResourceTracker::vkQueueSubmitEnc(VkEncoder* enc, VkQueue queue, uint32_t submitCount,
                                           const VkSubmitInfo* pSubmits, VkFence fence) {
    if (supportsAsyncQueueSubmit()) {
        enc->vkQueueSubmitAsyncGOOGLE(queue, submitCount, pSubmits, fence, true /* do lock */);
        return VK_SUCCESS;
    } else {
        return enc->vkQueueSubmit(queue, submitCount, pSubmits, fence, true /* do lock */);
    }
}

VkResult ResourceTracker::vkQueueSubmitEnc(VkEncoder* enc, VkQueue queue, uint32_t submitCount,
                                           const VkSubmitInfo2* pSubmits, VkFence fence) {
    if (supportsAsyncQueueSubmit()) {
        enc->vkQueueSubmitAsync2GOOGLE(queue, submitCount, pSubmits, fence, true /* do lock */);
        return VK_SUCCESS;
    } else {
        return enc->vkQueueSubmit2(queue, submitCount, pSubmits, fence, true /* do lock */);
    }
}

static void pruneWaitSemaphores(const std::vector<VkSemaphore>& toRemove, VkSubmitInfo& submitInfo,
                                const VkTimelineSemaphoreSubmitInfo* currTssi,
                                std::vector<VkSemaphore>& newSemList,
                                std::vector<VkPipelineStageFlags>& newWaitDstStageMaskList,
                                VkTimelineSemaphoreSubmitInfo& newTssi,
                                std::vector<uint64_t>& newSemValueList) {
    newSemList.clear();
    newSemValueList.clear();
    newWaitDstStageMaskList.clear();

    if (!submitInfo.pWaitSemaphores) {
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = nullptr;
        submitInfo.pWaitDstStageMask = nullptr;
        return;
    }

    for (uint32_t i = 0; i < submitInfo.waitSemaphoreCount; i++) {
        auto it = std::find(toRemove.begin(), toRemove.end(), submitInfo.pWaitSemaphores[i]);
        if (it == toRemove.end()) {
            newSemList.push_back(submitInfo.pWaitSemaphores[i]);

            if (submitInfo.pWaitDstStageMask) {
                newWaitDstStageMaskList.push_back(submitInfo.pWaitDstStageMask[i]);
            }

            if (currTssi && currTssi->pWaitSemaphoreValues &&
                i < currTssi->waitSemaphoreValueCount) {
                newSemValueList.push_back(currTssi->pWaitSemaphoreValues[i]);
            }
        }
    }

    submitInfo.waitSemaphoreCount = static_cast<uint32_t>(newSemList.size());
    submitInfo.pWaitSemaphores = newSemList.empty() ? nullptr : newSemList.data();
    submitInfo.pWaitDstStageMask =
        newWaitDstStageMaskList.empty() ? nullptr : newWaitDstStageMaskList.data();

    if (newSemValueList.size() > 0) {
        newTssi.waitSemaphoreValueCount = static_cast<uint32_t>(newSemValueList.size());
        newTssi.pWaitSemaphoreValues = newSemValueList.data();
        submitInfo.pNext = &newTssi;
    }
}

static void pruneWaitSemaphores(const std::vector<VkSemaphore>& toRemove, VkSubmitInfo2& submitInfo,
                                const VkTimelineSemaphoreSubmitInfo* currTssi,
                                std::vector<VkSemaphoreSubmitInfo>& newSemList,
                                std::vector<VkPipelineStageFlags>& newWaitDstStageMaskList,
                                VkTimelineSemaphoreSubmitInfo& newTssi,
                                std::vector<uint64_t>& newSemValueList) {
    // All of this info is contained in VkSubmitInfo2, so goes unused in this implementation
    // of pruneWaitSemaphores
    (void)currTssi;
    (void)newWaitDstStageMaskList;
    (void)newTssi;
    (void)newSemValueList;
    newSemList.clear();
    for (uint32_t i = 0; i < submitInfo.waitSemaphoreInfoCount; i++) {
        auto it = std::find(toRemove.begin(), toRemove.end(),
                            submitInfo.pWaitSemaphoreInfos[i].semaphore);
        if (it == toRemove.end()) {
            newSemList.push_back(submitInfo.pWaitSemaphoreInfos[i]);
        }
    }
    submitInfo.waitSemaphoreInfoCount = static_cast<uint32_t>(newSemList.size());
    submitInfo.pWaitSemaphoreInfos = newSemList.data();
}

static void pruneSignalSemaphores(const std::vector<VkSemaphore>& toRemove,
                                  VkSubmitInfo& submitInfo,
                                  const VkTimelineSemaphoreSubmitInfo* currTssi,
                                  std::vector<VkSemaphore>& newSemList,
                                  VkTimelineSemaphoreSubmitInfo& newTssi,
                                  std::vector<uint64_t>& newSemValueList) {
    newSemList.clear();
    newSemValueList.clear();

    if (!submitInfo.pSignalSemaphores) {
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = nullptr;
        return;
    }

    for (uint32_t i = 0; i < submitInfo.signalSemaphoreCount; i++) {
        auto it = std::find(toRemove.begin(), toRemove.end(), submitInfo.pSignalSemaphores[i]);
        if (it == toRemove.end()) {
            newSemList.push_back(submitInfo.pSignalSemaphores[i]);

            if (currTssi && currTssi->pSignalSemaphoreValues &&
                i < currTssi->signalSemaphoreValueCount) {
                newSemValueList.push_back(currTssi->pSignalSemaphoreValues[i]);
            }
        }
    }

    submitInfo.signalSemaphoreCount = static_cast<uint32_t>(newSemList.size());
    submitInfo.pSignalSemaphores = newSemList.empty() ? nullptr : newSemList.data();

    if (newSemValueList.size() > 0) {
        newTssi.signalSemaphoreValueCount = static_cast<uint32_t>(newSemValueList.size());
        newTssi.pSignalSemaphoreValues = newSemValueList.data();
        submitInfo.pNext = &newTssi;
    }
}

static void pruneSignalSemaphores(const std::vector<VkSemaphore>& toRemove,
                                  VkSubmitInfo2& submitInfo,
                                  const VkTimelineSemaphoreSubmitInfo* currTssi,
                                  std::vector<VkSemaphoreSubmitInfo>& newSemList,
                                  VkTimelineSemaphoreSubmitInfo& newTssi,
                                  std::vector<uint64_t>& newSemValueList) {
    // All of this info is contained in VkSubmitInfo2, so goes unused in this implementation
    // of pruneSignalSemaphores
    (void)currTssi;
    (void)newTssi;
    (void)newSemValueList;
    newSemList.clear();
    for (uint32_t i = 0; i < submitInfo.signalSemaphoreInfoCount; i++) {
        auto it = std::find(toRemove.begin(), toRemove.end(),
                            submitInfo.pSignalSemaphoreInfos[i].semaphore);
        if (it == toRemove.end()) {
            newSemList.push_back(submitInfo.pSignalSemaphoreInfos[i]);
        }
    }
    submitInfo.signalSemaphoreInfoCount = static_cast<uint32_t>(newSemList.size());
    submitInfo.pSignalSemaphoreInfos = newSemList.data();
}

template <typename VkSubmitInfoType, typename VkSemaphoreInfoType>
VkResult ResourceTracker::on_vkQueueSubmitTemplate(void* context, VkResult input_result,
                                                   VkQueue queue, uint32_t submitCount,
                                                   const VkSubmitInfoType* pSubmits,
                                                   VkFence fence) {
    flushStagingStreams(context, queue, submitCount, pSubmits);

    VkEncoder* enc = (VkEncoder*)context;

    bool needsPruneOrExternalSync = false;
    bool hasTimelineSemaphores = false;
    bool hasPNext = false;

    for (uint32_t i = 0; i < submitCount; ++i) {
        if (pSubmits[i].pNext) {
            hasPNext = true;
        }
        uint32_t waitCount = getWaitSemaphoreCount(pSubmits[i]);
        for (uint32_t j = 0; j < waitCount; ++j) {
            VK_FROM_HANDLE(gfxstream_vk_semaphore, semInfo, getWaitSemaphore(pSubmits[i], j));
            if (!semInfo) continue;
            if (semInfo->isTimeline) hasTimelineSemaphores = true;
#ifdef VK_USE_PLATFORM_FUCHSIA
            if (semInfo->eventHandle) needsPruneOrExternalSync = true;
#endif
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
            if (semInfo->syncFd.has_value()) needsPruneOrExternalSync = true;
#endif
        }
        uint32_t signalCount = getSignalSemaphoreCount(pSubmits[i]);
        for (uint32_t j = 0; j < signalCount; ++j) {
            VK_FROM_HANDLE(gfxstream_vk_semaphore, semInfo, getSignalSemaphore(pSubmits[i], j));
            if (!semInfo) continue;
            if (semInfo->isTimeline) hasTimelineSemaphores = true;
#ifdef VK_USE_PLATFORM_FUCHSIA
            if (semInfo->eventHandle) needsPruneOrExternalSync = true;
#endif
#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
            if (semInfo->syncFd.value_or(-1) >= 0) needsPruneOrExternalSync = true;
#endif
        }
    }

    if (hasPNext) {
        for (uint32_t i = 0; i < submitCount; ++i) {
            if (pSubmits[i].pNext) {
                needsPruneOrExternalSync = true;
                break;
            }
        }
    }

    int externalFenceFdToSignal = -1;
#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
    if (fence != VK_NULL_HANDLE) {
        VK_FROM_HANDLE(gfxstream_vk_fence, fenceInfo, fence);
        if (fenceInfo && fenceInfo->syncFd && *fenceInfo->syncFd >= 0) {
            externalFenceFdToSignal = *fenceInfo->syncFd;
        }
    }
#endif

    if (!needsPruneOrExternalSync && externalFenceFdToSignal < 0) {
        return vkQueueSubmitEnc(enc, queue, submitCount, pSubmits, fence);
    }

#ifdef VK_USE_PLATFORM_FUCHSIA
    std::vector<VkSemaphore> pre_signal_semaphores;
    std::vector<zx_handle_t> pre_signal_events;
#endif
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    std::vector<int> preSignalSyncFds;
#endif

    std::vector<std::pair<zx_handle_t, zx_koid_t>> post_wait_events;
    std::vector<int> post_wait_sync_fds;

    std::vector<VkSubmitInfoType> prunedSubmitInfos(submitCount);
    std::vector<std::vector<VkSemaphoreInfoType>> prunedWaitSemaphoreLists(submitCount);
    std::vector<std::vector<VkPipelineStageFlags>> prunedWaitDstStageMaskFlagLists(submitCount);
    std::vector<std::vector<VkSemaphoreInfoType>> prunedSignalSemaphoreLists(submitCount);
    std::vector<VkTimelineSemaphoreSubmitInfo> prunedTssis(submitCount);
    std::vector<std::vector<uint64_t>> prunedWaitSemaphoreValueLists(submitCount);
    std::vector<std::vector<uint64_t>> prunedSignalSemaphoreValueLists(submitCount);

    for (uint32_t i = 0; i < submitCount; ++i) {
        std::vector<VkSemaphore> waitSemsToRemove;
        std::vector<VkSemaphore> signalSemsToRemove;
        for (uint32_t j = 0; j < getWaitSemaphoreCount(pSubmits[i]); ++j) {
            VkSemaphore semaphore = getWaitSemaphore(pSubmits[i], j);
            VK_FROM_HANDLE(gfxstream_vk_semaphore, semInfo, semaphore);
            if (semInfo) {
#ifdef VK_USE_PLATFORM_FUCHSIA
                if (semInfo->eventHandle) {
                    pre_signal_events.push_back(semInfo->eventHandle);
                    pre_signal_semaphores.push_back(semaphore);
                }
#endif
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
                if (semInfo->syncFd.has_value()) {
                    preSignalSyncFds.push_back(semInfo->syncFd.value());
                    waitSemsToRemove.push_back(semaphore);
                }
#endif
            }
        }
        for (uint32_t j = 0; j < getSignalSemaphoreCount(pSubmits[i]); ++j) {
            VkSemaphore semaphore = getSignalSemaphore(pSubmits[i], j);
            VK_FROM_HANDLE(gfxstream_vk_semaphore, semInfo, semaphore);
            if (semInfo) {
#ifdef VK_USE_PLATFORM_FUCHSIA
                if (semInfo->eventHandle) {
                    post_wait_events.push_back({semInfo->eventHandle, semInfo->eventKoid});
#ifndef FUCHSIA_NO_TRACE
                    if (semInfo->eventKoid != ZX_KOID_INVALID) {
                        TRACE_FLOW_END("gfx", "semaphore", semInfo->eventKoid);
                        TRACE_FLOW_BEGIN("gfx", "goldfish_post_wait_event", semInfo->eventKoid);
                    }
#endif
                }
#endif
#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
                if (semInfo->syncFd.value_or(-1) >= 0) {
                    post_wait_sync_fds.push_back(semInfo->syncFd.value());
                    signalSemsToRemove.push_back(semaphore);
                }
#endif
            }
        }

        const VkTimelineSemaphoreSubmitInfo* currTssi =
            hasTimelineSemaphores
                ? vk_find_struct_const(&pSubmits[i], TIMELINE_SEMAPHORE_SUBMIT_INFO)
                : nullptr;
        prunedSubmitInfos[i] = vk_make_orphan_copy(pSubmits[i]);
        prunedTssis[i] = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, .pNext = NULL};
        pruneWaitSemaphores(waitSemsToRemove, prunedSubmitInfos[i], currTssi,
                            prunedWaitSemaphoreLists[i], prunedWaitDstStageMaskFlagLists[i],
                            prunedTssis[i], prunedWaitSemaphoreValueLists[i]);
        pruneSignalSemaphores(signalSemsToRemove, prunedSubmitInfos[i], currTssi,
                              prunedSignalSemaphoreLists[i], prunedTssis[i],
                              prunedSignalSemaphoreValueLists[i]);
    }

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (!pre_signal_semaphores.empty()) {
        for (auto event : pre_signal_events) {
            preSignalTasks.push_back([event] {
                zx_object_wait_one(event, ZX_EVENT_SIGNALED, ZX_TIME_INFINITE, nullptr);
            });
        }
        VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .signalSemaphoreCount = static_cast<uint32_t>(pre_signal_semaphores.size()),
            .pSignalSemaphores = pre_signal_semaphores.data()};
        vkQueueSubmitEnc(enc, queue, 1, &submit_info, VK_NULL_HANDLE);
    }
#endif
#if defined(VK_USE_PLATFORM_ANDROID_KHR) || DETECT_OS_LINUX
    for (auto fd : preSignalSyncFds) {
        if (fd != -1) {
            mSyncHelper->wait(fd, 3000);
#if GFXSTREAM_SYNC_DEBUG
            mSyncHelper->debugPrint(fd);
#endif
        }
    }
#endif

    input_result = vkQueueSubmitEnc(enc, queue, submitCount, prunedSubmitInfos.data(), fence);
    if (input_result != VK_SUCCESS) return input_result;

    VkResult waitIdleRes = VK_SUCCESS;
    if (externalFenceFdToSignal >= 0 || !post_wait_events.empty() || !post_wait_sync_fds.empty()) {
        waitIdleRes = enc->vkQueueWaitIdle(queue, true /* do lock */);
        if (VK_SUCCESS == waitIdleRes) {
#ifdef VK_USE_PLATFORM_FUCHSIA
            MESA_TRACE_SCOPE("on_vkQueueSubmit::SignalSemaphores");
            (void)externalFenceFdToSignal;
            for (auto& [event, koid] : post_wait_events) {
#ifndef FUCHSIA_NO_TRACE
                if (koid != ZX_KOID_INVALID) {
                    TRACE_FLOW_END("gfx", "goldfish_post_wait_event", koid);
                    TRACE_FLOW_BEGIN("gfx", "event_signal", koid);
                }
#endif
                zx_object_signal(event, 0, ZX_EVENT_SIGNALED);
            }
#endif
#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
            for (auto& fd : post_wait_sync_fds) {
                goldfish_sync_signal(fd);
            }

            if (externalFenceFdToSignal >= 0) {
                MESA_TRACE_SCOPE("%s: external fence real signal: %d\n", __func__,
                                 externalFenceFdToSignal);
                goldfish_sync_signal(externalFenceFdToSignal);
            }
#endif
        }
    }
    return waitIdleRes;
}

VkResult ResourceTracker::on_vkQueueWaitIdle(void* context, VkResult, VkQueue queue) {
    VkEncoder* enc = (VkEncoder*)context;

    // now done waiting, get the host's opinion
    return enc->vkQueueWaitIdle(queue, true /* do lock */);
}

#ifdef VK_USE_PLATFORM_ANDROID_KHR
void ResourceTracker::unwrap_VkNativeBufferANDROID(const VkNativeBufferANDROID* inputNativeInfo,
                                                   VkNativeBufferANDROID* outputNativeInfo) {
    if (!inputNativeInfo || !inputNativeInfo->handle) {
        return;
    }

    if (!outputNativeInfo || !outputNativeInfo) {
        mesa_loge("FATAL: Local native buffer info not properly allocated!");
        abort();
    }

    const native_handle_t* nativeHandle = (const native_handle_t*)inputNativeInfo->handle;
    *(uint32_t*)(outputNativeInfo->handle) = mGralloc->getHostHandle(nativeHandle);
}

void ResourceTracker::unwrap_VkBindImageMemorySwapchainInfoKHR(
    const VkBindImageMemorySwapchainInfoKHR* inputBimsi,
    VkBindImageMemorySwapchainInfoKHR* outputBimsi) {
    if (!inputBimsi || !inputBimsi->swapchain) {
        return;
    }

    if (!outputBimsi || !outputBimsi->swapchain) {
        return;
    }

    // Android based swapchains are implemented by the Android framework's
    // libvulkan. The only exist within the guest and should not be sent to
    // the host.
    outputBimsi->swapchain = VK_NULL_HANDLE;
}
#endif

void ResourceTracker::unwrap_vkCreateImage_pCreateInfo(const VkImageCreateInfo* pCreateInfo,
                                                       VkImageCreateInfo* local_pCreateInfo) {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    const VkNativeBufferANDROID* inputNativeInfo =
        vk_find_struct_const(pCreateInfo, NATIVE_BUFFER_ANDROID);

    VkNativeBufferANDROID* outputNativeInfo =
        vk_find_struct(local_pCreateInfo, NATIVE_BUFFER_ANDROID);

    unwrap_VkNativeBufferANDROID(inputNativeInfo, outputNativeInfo);
#endif
}

void ResourceTracker::unwrap_vkAcquireImageANDROID_nativeFenceFd(int fd, int* fd_out) {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    (void)fd_out;
    if (fd != -1) {
        MESA_TRACE_SCOPE("waitNativeFenceInAcquire");
        // Implicit Synchronization
        mSyncHelper->wait(fd, 3000);
        // From libvulkan's swapchain.cpp:
        // """
        // NOTE: we're relying on AcquireImageANDROID to close fence_clone,
        // even if the call fails. We could close it ourselves on failure, but
        // that would create a race condition if the driver closes it on a
        // failure path: some other thread might create an fd with the same
        // number between the time the driver closes it and the time we close
        // it. We must assume one of: the driver *always* closes it even on
        // failure, or *never* closes it on failure.
        // """
        // Therefore, assume contract where we need to close fd in this driver

#if GFXSTREAM_SYNC_DEBUG
        mSyncHelper->debugPrint(fd);
#endif
        mSyncHelper->close(fd);
    }
#endif
}

void ResourceTracker::unwrap_VkBindImageMemory2_pBindInfos(
    uint32_t bindInfoCount, const VkBindImageMemoryInfo* inputBindInfos,
    VkBindImageMemoryInfo* outputBindInfos) {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    for (uint32_t i = 0; i < bindInfoCount; ++i) {
        const VkBindImageMemoryInfo* inputBindInfo = &inputBindInfos[i];
        VkBindImageMemoryInfo* outputBindInfo = &outputBindInfos[i];

        const VkNativeBufferANDROID* inputNativeInfo =
            vk_find_struct_const(inputBindInfo, NATIVE_BUFFER_ANDROID);

        VkNativeBufferANDROID* outputNativeInfo =
            vk_find_struct(outputBindInfo, NATIVE_BUFFER_ANDROID);

        unwrap_VkNativeBufferANDROID(inputNativeInfo, outputNativeInfo);

        const VkBindImageMemorySwapchainInfoKHR* inputBimsi =
            vk_find_struct_const(inputBindInfo, BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR);

        VkBindImageMemorySwapchainInfoKHR* outputBimsi =
            vk_find_struct(outputBindInfo, BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR);

        unwrap_VkBindImageMemorySwapchainInfoKHR(inputBimsi, outputBimsi);
    }
#endif
}

// Action of vkMapMemoryIntoAddressSpaceGOOGLE:
// 1. preprocess (on_vkMapMemoryIntoAddressSpaceGOOGLE_pre):
//    uses address space device to reserve the right size of
//    memory.
// 2. the reservation results in a physical address. the physical
//    address is set as |*pAddress|.
// 3. after pre, the API call is encoded to the host, where the
//    value of pAddress is also sent (the physical address).
// 4. the host will obtain the actual gpu pointer and send it
//    back out in |*pAddress|.
// 5. postprocess (on_vkMapMemoryIntoAddressSpaceGOOGLE) will run,
//    using the mmap() method of GoldfishAddressSpaceBlock to obtain
//    a pointer in guest userspace corresponding to the host pointer.
VkResult ResourceTracker::on_vkMapMemoryIntoAddressSpaceGOOGLE_pre(void*, VkResult, VkDevice,
                                                                   VkDeviceMemory memory,
                                                                   uint64_t* pAddress) {
    VK_FROM_HANDLE(gfxstream_vk_device_memory, memInfo, memory);
    if (!memInfo) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

#if DETECT_OS_ANDROID
    GoldfishAddressSpaceBlockPtr block = std::make_shared<GoldfishAddressSpaceBlock>();
    if (!block->allocate(mGoldfishAddressSpaceBlockProvider.get(), memInfo->coherentMemorySize)) {
        *pAddress = 0;
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    memInfo->goldfishBlock = block;
    *pAddress = block->physAddr();

    return VK_SUCCESS;
#else
    (void)pAddress;
    return VK_ERROR_MEMORY_MAP_FAILED;
#endif
}

VkResult ResourceTracker::on_vkMapMemoryIntoAddressSpaceGOOGLE(void*, VkResult input_result,
                                                               VkDevice, VkDeviceMemory memory,
                                                               uint64_t* pAddress) {
    (void)memory;
    (void)pAddress;

    if (input_result != VK_SUCCESS) {
        return input_result;
    }

    return input_result;
}

VkResult ResourceTracker::initDescriptorUpdateTemplateBuffers(
    VkDevice device, const VkAllocationCallbacks* pAllocator,
    const VkDescriptorUpdateTemplateCreateInfo* pCreateInfo,
    VkDescriptorUpdateTemplate descriptorUpdateTemplate) {
    VK_FROM_HANDLE(gfxstream_vk_descriptor_update_template, info, descriptorUpdateTemplate);
    if (!info) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    info->pipelineBindPoint = pCreateInfo->pipelineBindPoint;
    uint32_t inlineUniformBlockBufferSize = 0;

    for (uint32_t i = 0; i < pCreateInfo->descriptorUpdateEntryCount; ++i) {
        const auto& entry = pCreateInfo->pDescriptorUpdateEntries[i];
        uint32_t descCount = entry.descriptorCount;
        VkDescriptorType descType = entry.descriptorType;
        ++info->templateEntryCount;
        if (isDescriptorTypeInlineUniformBlock(descType)) {
            inlineUniformBlockBufferSize += descCount;
            ++info->inlineUniformBlockCount;
        } else {
            for (uint32_t j = 0; j < descCount; ++j) {
                if (isDescriptorTypeImageInfo(descType)) {
                    ++info->imageInfoCount;
                } else if (isDescriptorTypeBufferInfo(descType)) {
                    ++info->bufferInfoCount;
                } else if (isDescriptorTypeBufferView(descType)) {
                    ++info->bufferViewCount;
                } else {
                    mesa_loge("%s: FATAL: Unknown descriptor type %d\n", __func__, descType);
                }
            }
        }
    }

    info->inlineUniformBlockBufferSize = inlineUniformBlockBufferSize;

    struct vk_multialloc ma = {};
    ma.align = GFXSTREAM_DEFAULT_ALIGN;
    vk_multialloc_add(&ma, &info->templateEntries, VkDescriptorUpdateTemplateEntry,
                      info->templateEntryCount);
    vk_multialloc_add(&ma, &info->imageInfoIndices, uint32_t, info->imageInfoCount);
    vk_multialloc_add(&ma, &info->bufferInfoIndices, uint32_t, info->bufferInfoCount);
    vk_multialloc_add(&ma, &info->bufferViewIndices, uint32_t, info->bufferViewCount);
    vk_multialloc_add(&ma, &info->inlineUniformBlockBytesPerBlocks, uint32_t,
                      info->inlineUniformBlockCount);

    if (ma.size > 0) {
        VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
        const VkAllocationCallbacks* parentAlloc =
            resolveDeviceAllocator(dev ? &dev->base : info->base.device);
        info->templateStorage =
            vk_multialloc_zalloc2(&ma, parentAlloc, pAllocator, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
        if (!info->templateStorage) {
            info->templateEntries = nullptr;
            info->imageInfoIndices = nullptr;
            info->bufferInfoIndices = nullptr;
            info->bufferViewIndices = nullptr;
            info->inlineUniformBlockBytesPerBlocks = nullptr;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    uint32_t imageInfoIndex = 0;
    uint32_t bufferInfoIndex = 0;
    uint32_t bufferViewIndex = 0;
    uint32_t inlineUniformBlockIndex = 0;

    for (uint32_t i = 0; i < pCreateInfo->descriptorUpdateEntryCount; ++i) {
        const auto& entry = pCreateInfo->pDescriptorUpdateEntries[i];
        uint32_t descCount = entry.descriptorCount;
        VkDescriptorType descType = entry.descriptorType;

        info->templateEntries[i] = entry;

        if (isDescriptorTypeInlineUniformBlock(descType)) {
            info->inlineUniformBlockBytesPerBlocks[inlineUniformBlockIndex] = descCount;
            ++inlineUniformBlockIndex;
        } else {
            for (uint32_t j = 0; j < descCount; ++j) {
                if (isDescriptorTypeImageInfo(descType)) {
                    info->imageInfoIndices[imageInfoIndex] = i;
                    ++imageInfoIndex;
                } else if (isDescriptorTypeBufferInfo(descType)) {
                    info->bufferInfoIndices[bufferInfoIndex] = i;
                    ++bufferInfoIndex;
                } else if (isDescriptorTypeBufferView(descType)) {
                    info->bufferViewIndices[bufferViewIndex] = i;
                    ++bufferViewIndex;
                } else {
                    mesa_loge("%s: FATAL: Unknown descriptor type %d\n", __func__, descType);
                }
            }
        }
    }

    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkCreateDescriptorUpdateTemplate(
    void* context, VkResult input_result, VkDevice device,
    const VkDescriptorUpdateTemplateCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDescriptorUpdateTemplate* pDescriptorUpdateTemplate) {
    if (input_result != VK_SUCCESS) return input_result;

    VkResult res = initDescriptorUpdateTemplateBuffers(device, pAllocator, pCreateInfo,
                                                       *pDescriptorUpdateTemplate);
    if (res != VK_SUCCESS) {
        VkEncoder* enc = (VkEncoder*)context;
        enc->vkDestroyDescriptorUpdateTemplate(device, *pDescriptorUpdateTemplate, pAllocator,
                                               false /* do lock */);
        *pDescriptorUpdateTemplate = VK_NULL_HANDLE;
    }
    return res;
}

VkResult ResourceTracker::on_vkCreateDescriptorUpdateTemplateKHR(
    void* context, VkResult input_result, VkDevice device,
    const VkDescriptorUpdateTemplateCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDescriptorUpdateTemplate* pDescriptorUpdateTemplate) {
    if (input_result != VK_SUCCESS) return input_result;

    VkResult res = initDescriptorUpdateTemplateBuffers(device, pAllocator, pCreateInfo,
                                                       *pDescriptorUpdateTemplate);
    if (res != VK_SUCCESS) {
        VkEncoder* enc = (VkEncoder*)context;
        enc->vkDestroyDescriptorUpdateTemplateKHR(device, *pDescriptorUpdateTemplate, pAllocator,
                                                  false /* do lock */);
        *pDescriptorUpdateTemplate = VK_NULL_HANDLE;
    }
    return res;
}

void ResourceTracker::on_vkUpdateDescriptorSetWithTemplate(
    void* context, VkDevice device, VkDescriptorSet descriptorSet,
    VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void* pData) {
    VkEncoder* enc = (VkEncoder*)context;

    const uint8_t* userBuffer = static_cast<const uint8_t*>(pData);
    if (!userBuffer) return;

    VK_FROM_HANDLE(gfxstream_vk_descriptor_update_template, info, descriptorUpdateTemplate);
    if (!info) return;

    uint32_t templateEntryCount = info->templateEntryCount;
    const VkDescriptorUpdateTemplateEntry* templateEntries = info->templateEntries;

    uint32_t imageInfoCount = info->imageInfoCount;
    uint32_t bufferInfoCount = info->bufferInfoCount;
    uint32_t bufferViewCount = info->bufferViewCount;
    uint32_t inlineUniformBlockBufferSize = info->inlineUniformBlockBufferSize;

    VkDescriptorImageInfo stackImageInfos[64];
    VkDescriptorBufferInfo stackBufferInfos[64];
    VkBufferView stackBufferViews[64];
    uint8_t stackInlineUniformBlock[256];

    std::vector<VkDescriptorImageInfo> heapImageInfos;
    std::vector<VkDescriptorBufferInfo> heapBufferInfos;
    std::vector<VkBufferView> heapBufferViews;
    std::vector<uint8_t> heapInlineUniformBlock;

    VkDescriptorImageInfo* imageInfos = stackImageInfos;
    if (imageInfoCount > 64) {
        heapImageInfos.resize(imageInfoCount);
        imageInfos = heapImageInfos.data();
    }
    VkDescriptorBufferInfo* bufferInfos = stackBufferInfos;
    if (bufferInfoCount > 64) {
        heapBufferInfos.resize(bufferInfoCount);
        bufferInfos = heapBufferInfos.data();
    }
    VkBufferView* bufferViews = stackBufferViews;
    if (bufferViewCount > 64) {
        heapBufferViews.resize(bufferViewCount);
        bufferViews = heapBufferViews.data();
    }
    uint8_t* inlineUniformBlockBuffer = stackInlineUniformBlock;
    if (inlineUniformBlockBufferSize > 256) {
        heapInlineUniformBlock.resize(inlineUniformBlockBufferSize);
        inlineUniformBlockBuffer = heapInlineUniformBlock.data();
    }

    size_t currImageInfoIdx = 0;
    size_t currBufferInfoIdx = 0;
    size_t currBufferViewIdx = 0;
    size_t inlineUniformBlockOffset = 0;
    size_t inlineUniformBlockIdx = 0;

    VK_FROM_HANDLE(gfxstream_vk_descriptor_set, ds, descriptorSet);
    ReifiedDescriptorSet* reified = ds ? ds->reified : nullptr;

    bool batched = mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate;

    for (uint32_t i = 0; i < templateEntryCount; ++i) {
        const auto& entry = templateEntries[i];
        VkDescriptorType descType = entry.descriptorType;
        uint32_t dstBinding = entry.dstBinding;

        auto offset = entry.offset;
        auto stride = entry.stride;
        auto dstArrayElement = entry.dstArrayElement;

        uint32_t descCount = entry.descriptorCount;

        if (isDescriptorTypeImageInfo(descType)) {
            if (!stride) stride = sizeof(VkDescriptorImageInfo);

            const VkDescriptorImageInfo* currImageInfoBegin = &imageInfos[currImageInfoIdx];
            bool hasSampler = (descType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                               descType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
            bool isImmutableSampler =
                reified && dstBinding < reified->bindingIsImmutableSampler.size() &&
                reified->bindingIsImmutableSampler[dstBinding];

            for (uint32_t j = 0; j < descCount; ++j) {
                const VkDescriptorImageInfo* user =
                    reinterpret_cast<const VkDescriptorImageInfo*>(userBuffer + offset + j * stride);
                VkDescriptorImageInfo sanitized = *user;
                if (!hasSampler || isImmutableSampler) {
                    sanitized.sampler = VK_NULL_HANDLE;
                }
                if (descType == VK_DESCRIPTOR_TYPE_SAMPLER) {
                    sanitized.imageView = VK_NULL_HANDLE;
                }
                imageInfos[currImageInfoIdx++] = sanitized;
            }

            if (batched) {
                doEmulatedDescriptorImageInfoWriteFromTemplate(
                    descType, dstBinding, dstArrayElement, descCount, currImageInfoBegin, reified);
            }
        } else if (isDescriptorTypeBufferInfo(descType)) {
            if (!stride) stride = sizeof(VkDescriptorBufferInfo);

            const VkDescriptorBufferInfo* currBufferInfoBegin = &bufferInfos[currBufferInfoIdx];

            for (uint32_t j = 0; j < descCount; ++j) {
                const VkDescriptorBufferInfo* user =
                    reinterpret_cast<const VkDescriptorBufferInfo*>(userBuffer + offset +
                                                                    j * stride);
                bufferInfos[currBufferInfoIdx++] = *user;
            }

            if (batched) {
                doEmulatedDescriptorBufferInfoWriteFromTemplate(
                    descType, dstBinding, dstArrayElement, descCount, currBufferInfoBegin, reified);
            }
        } else if (isDescriptorTypeBufferView(descType)) {
            if (!stride) stride = sizeof(VkBufferView);

            const VkBufferView* currBufferViewBegin = &bufferViews[currBufferViewIdx];

            for (uint32_t j = 0; j < descCount; ++j) {
                const VkBufferView* user =
                    reinterpret_cast<const VkBufferView*>(userBuffer + offset + j * stride);
                bufferViews[currBufferViewIdx++] = *user;
            }

            if (batched) {
                doEmulatedDescriptorBufferViewWriteFromTemplate(
                    descType, dstBinding, dstArrayElement, descCount, currBufferViewBegin, reified);
            }
        } else if (isDescriptorTypeInlineUniformBlock(descType)) {
            uint32_t inlineUniformBlockBytesPerBlock =
                info->inlineUniformBlockBytesPerBlocks[inlineUniformBlockIdx];
            uint8_t* currInlineUniformBlockBufferBegin =
                inlineUniformBlockBuffer + inlineUniformBlockOffset;
            memcpy(currInlineUniformBlockBufferBegin, userBuffer + offset,
                   inlineUniformBlockBytesPerBlock);
            inlineUniformBlockIdx++;
            inlineUniformBlockOffset += inlineUniformBlockBytesPerBlock;

            if (batched) {
                doEmulatedDescriptorInlineUniformBlockFromTemplate(
                    descType, dstBinding, dstArrayElement, descCount,
                    currInlineUniformBlockBufferBegin, reified);
            }
        } else {
            mesa_loge("%s: FATAL: Unknown descriptor type %d\n", __func__, descType);
            abort();
        }
    }

    if (batched) return;

    enc->vkUpdateDescriptorSetWithTemplateSized2GOOGLE(
        device, descriptorSet, descriptorUpdateTemplate, imageInfoCount, bufferInfoCount,
        bufferViewCount, inlineUniformBlockBufferSize, info->imageInfoIndices,
        info->bufferInfoIndices, info->bufferViewIndices, imageInfos, bufferInfos, bufferViews,
        inlineUniformBlockBuffer, true /* do lock */);
}

void ResourceTracker::on_vkUpdateDescriptorSetWithTemplateKHR(
    void* context, VkDevice device, VkDescriptorSet descriptorSet,
    VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void* pData) {
    on_vkUpdateDescriptorSetWithTemplate(context, device, descriptorSet, descriptorUpdateTemplate,
                                         pData);
}

// VK_KHR_push_descriptor (core in VK_VERSION_1_4) allows applications to leave
// some fields of VkWriteDescriptorSet unset (e.g. `dstSet`) and some other
// fields unset based on the descriptor type (e.g. `pImageInfo` for
// `VK_DESCRIPTOR_TYPE_SAMPLER`). The pre-existing guest marshaling code for
// `VkWriteDescriptorSet` will try to lookup the host handles of those various
// fields if they are not VK_NULL_HANDLE, so we clear them out here.
void ResourceTracker::sanitizePushDescriptorWrites(
    VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
    const VkWriteDescriptorSet* pDescriptorWrites,
    std::vector<VkWriteDescriptorSet>& transformedWrites,
    std::vector<VkDescriptorImageInfo>& transformedImageInfos) {
    VK_FROM_HANDLE(gfxstream_vk_pipeline_layout, pl, layout);
    const std::vector<bool>* immutableBindings =
        (pl && set < pl->immutableSamplers.size() && !pl->immutableSamplers[set].empty())
            ? &pl->immutableSamplers[set]
            : nullptr;

    transformedWrites.resize(descriptorWriteCount);
    size_t imageInfosNeeded = 0;
    for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
        if (isDescriptorTypeImageInfo(pDescriptorWrites[i].descriptorType) &&
            pDescriptorWrites[i].pImageInfo) {
            imageInfosNeeded += pDescriptorWrites[i].descriptorCount;
        }
    }
    transformedImageInfos.resize(imageInfosNeeded);
    size_t imageInfoIndex = 0;
    for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
        VkWriteDescriptorSet& write = transformedWrites[i];
        write = pDescriptorWrites[i];
        write.dstSet = VK_NULL_HANDLE;

        VkDescriptorType descType = write.descriptorType;
        if (!isDescriptorTypeBufferInfo(descType)) {
            write.pBufferInfo = nullptr;
        }
        if (!isDescriptorTypeBufferView(descType)) {
            write.pTexelBufferView = nullptr;
        }
        if (!isDescriptorTypeImageInfo(descType)) {
            write.pImageInfo = nullptr;
            continue;
        }

        const bool isImmutable = immutableBindings &&
                                 write.dstBinding < immutableBindings->size() &&
                                 (*immutableBindings)[write.dstBinding];
        if (!write.pImageInfo || write.descriptorCount == 0) {
            write.pImageInfo = nullptr;
            continue;
        }

        VkDescriptorImageInfo* dstImageInfos = &transformedImageInfos[imageInfoIndex];
        for (uint32_t j = 0; j < write.descriptorCount; ++j) {
            VkDescriptorImageInfo imgInfo = write.pImageInfo[j];
            if (descType == VK_DESCRIPTOR_TYPE_SAMPLER) {
                imgInfo.imageView = VK_NULL_HANDLE;
                if (isImmutable) {
                    imgInfo.sampler = VK_NULL_HANDLE;
                }
            } else if (descType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                if (isImmutable) {
                    imgInfo.sampler = VK_NULL_HANDLE;
                }
            } else {
                imgInfo.sampler = VK_NULL_HANDLE;
            }
            dstImageInfos[j] = imgInfo;
        }
        write.pImageInfo = dstImageInfos;
        imageInfoIndex += write.descriptorCount;
    }
}

void ResourceTracker::on_vkCmdPushDescriptorSet(void* context, VkCommandBuffer commandBuffer,
                                                VkPipelineBindPoint pipelineBindPoint,
                                                VkPipelineLayout layout, uint32_t set,
                                                uint32_t descriptorWriteCount,
                                                const VkWriteDescriptorSet* pDescriptorWrites) {
    VkEncoder* enc = (VkEncoder*)context;
    std::vector<VkDescriptorImageInfo> transformedImageInfos;
    std::vector<VkWriteDescriptorSet> transformedWrites;
    sanitizePushDescriptorWrites(layout, set, descriptorWriteCount, pDescriptorWrites,
                                 transformedWrites, transformedImageInfos);
    enc->vkCmdPushDescriptorSet(commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount,
                                transformedWrites.data(), true /* do lock */);
}

void ResourceTracker::on_vkCmdPushDescriptorSetKHR(void* context, VkCommandBuffer commandBuffer,
                                                   VkPipelineBindPoint pipelineBindPoint,
                                                   VkPipelineLayout layout, uint32_t set,
                                                   uint32_t descriptorWriteCount,
                                                   const VkWriteDescriptorSet* pDescriptorWrites) {
    VkEncoder* enc = (VkEncoder*)context;
    std::vector<VkDescriptorImageInfo> transformedImageInfos;
    std::vector<VkWriteDescriptorSet> transformedWrites;
    sanitizePushDescriptorWrites(layout, set, descriptorWriteCount, pDescriptorWrites,
                                 transformedWrites, transformedImageInfos);
    enc->vkCmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set,
                                   descriptorWriteCount, transformedWrites.data(),
                                   true /* do lock */);
}

void ResourceTracker::on_vkCmdPushDescriptorSetWithTemplate(
    void* context, VkCommandBuffer commandBuffer,
    VkDescriptorUpdateTemplate descriptorUpdateTemplate, VkPipelineLayout layout, uint32_t set,
    const void* pData) {
    pushDescriptorSetWithTemplate(context, commandBuffer, descriptorUpdateTemplate, layout, set,
                                  pData, false /* khr */);
}

void ResourceTracker::on_vkCmdPushDescriptorSetWithTemplateKHR(
    void* context, VkCommandBuffer commandBuffer,
    VkDescriptorUpdateTemplate descriptorUpdateTemplate, VkPipelineLayout layout, uint32_t set,
    const void* pData) {
    pushDescriptorSetWithTemplate(context, commandBuffer, descriptorUpdateTemplate, layout, set,
                                  pData, true /* khr */);
}

void ResourceTracker::pushDescriptorSetWithTemplate(
    void* context, VkCommandBuffer commandBuffer,
    VkDescriptorUpdateTemplate descriptorUpdateTemplate, VkPipelineLayout layout, uint32_t set,
    const void* pData, bool khr) {
    VkPushDescriptorSetWithTemplateInfo pushInfo = {
        .sType = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_WITH_TEMPLATE_INFO,
        .pNext = nullptr,
        .descriptorUpdateTemplate = descriptorUpdateTemplate,
        .layout = layout,
        .set = set,
        .pData = pData,
    };
    pushDescriptorSetWithTemplate2(context, commandBuffer, &pushInfo, khr);
}

void ResourceTracker::on_vkCmdPushDescriptorSet2(
    void* context, VkCommandBuffer commandBuffer,
    const VkPushDescriptorSetInfo* pPushDescriptorSetInfo) {
    if (!pPushDescriptorSetInfo) return;
    VkEncoder* enc = (VkEncoder*)context;
    std::vector<VkDescriptorImageInfo> transformedImageInfos;
    std::vector<VkWriteDescriptorSet> transformedWrites;
    sanitizePushDescriptorWrites(
        pPushDescriptorSetInfo->layout, pPushDescriptorSetInfo->set,
        pPushDescriptorSetInfo->descriptorWriteCount, pPushDescriptorSetInfo->pDescriptorWrites,
        transformedWrites, transformedImageInfos);
    VkPushDescriptorSetInfo modifiedInfo = *pPushDescriptorSetInfo;
    modifiedInfo.pDescriptorWrites = transformedWrites.data();
    enc->vkCmdPushDescriptorSet2(commandBuffer, &modifiedInfo, true /* do lock */);
}

void ResourceTracker::on_vkCmdPushDescriptorSet2KHR(
    void* context, VkCommandBuffer commandBuffer,
    const VkPushDescriptorSetInfo* pPushDescriptorSetInfo) {
    if (!pPushDescriptorSetInfo) return;
    VkEncoder* enc = (VkEncoder*)context;
    std::vector<VkDescriptorImageInfo> transformedImageInfos;
    std::vector<VkWriteDescriptorSet> transformedWrites;
    sanitizePushDescriptorWrites(
        pPushDescriptorSetInfo->layout, pPushDescriptorSetInfo->set,
        pPushDescriptorSetInfo->descriptorWriteCount, pPushDescriptorSetInfo->pDescriptorWrites,
        transformedWrites, transformedImageInfos);
    VkPushDescriptorSetInfo modifiedInfo = *pPushDescriptorSetInfo;
    modifiedInfo.pDescriptorWrites = transformedWrites.data();
    enc->vkCmdPushDescriptorSet2KHR(commandBuffer, &modifiedInfo, true);
}

// VkPushDescriptorSetWithTemplateInfo has an opaque `const void* pData` field
// whose size is determined from the VkDescriptorUpdateTemplateEntry referenced.
// Instead of trying to make the marhsaling code more stateful, the template
// writes are expanded into vkCmdPushDescriptorSet() calls in the guest.
void ResourceTracker::on_vkCmdPushDescriptorSetWithTemplate2(
    void* context, VkCommandBuffer commandBuffer,
    const VkPushDescriptorSetWithTemplateInfo* pPushDescriptorSetWithTemplateInfo) {
    pushDescriptorSetWithTemplate2(context, commandBuffer, pPushDescriptorSetWithTemplateInfo,
                                   false /* khr */);
}

void ResourceTracker::on_vkCmdPushDescriptorSetWithTemplate2KHR(
    void* context, VkCommandBuffer commandBuffer,
    const VkPushDescriptorSetWithTemplateInfo* pPushDescriptorSetWithTemplateInfo) {
    pushDescriptorSetWithTemplate2(context, commandBuffer, pPushDescriptorSetWithTemplateInfo,
                                   true /* khr */);
}

void ResourceTracker::pushDescriptorSetWithTemplate2(
    void* context, VkCommandBuffer commandBuffer,
    const VkPushDescriptorSetWithTemplateInfo* pPushDescriptorSetWithTemplateInfo, bool khr) {
    if (!pPushDescriptorSetWithTemplateInfo) return;
    const uint8_t* userBuffer =
        static_cast<const uint8_t*>(pPushDescriptorSetWithTemplateInfo->pData);

    VK_FROM_HANDLE(gfxstream_vk_descriptor_update_template, info,
                   pPushDescriptorSetWithTemplateInfo->descriptorUpdateTemplate);
    if (!info) {
        return;
    }
    if (!userBuffer && info->templateEntryCount > 0) {
        return;
    }

    VkPipelineBindPoint pipelineBindPoint = info->pipelineBindPoint;
    std::vector<VkWriteDescriptorSet> writes(info->templateEntryCount);
    std::vector<VkDescriptorImageInfo> imageInfos(info->imageInfoCount);
    std::vector<VkDescriptorBufferInfo> bufferInfos(info->bufferInfoCount);
    std::vector<VkBufferView> bufferViews(info->bufferViewCount);
    std::vector<VkWriteDescriptorSetInlineUniformBlock> inlineUniformBlocks(
        info->inlineUniformBlockCount);

    uint32_t currImageInfoIndex = 0;
    uint32_t currBufferInfoIndex = 0;
    uint32_t currBufferViewIndex = 0;
    uint32_t currInlineUniformBlockIndex = 0;

    for (uint32_t i = 0; i < info->templateEntryCount; ++i) {
        const auto& entry = info->templateEntries[i];
        VkDescriptorType descType = entry.descriptorType;
        uint32_t descCount = entry.descriptorCount;
        size_t offset = entry.offset;
        size_t stride = entry.stride;

        VkWriteDescriptorSet& write = writes[i];
        write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = entry.dstBinding,
            .dstArrayElement = entry.dstArrayElement,
            .descriptorCount = descCount,
            .descriptorType = descType,
            .pImageInfo = nullptr,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        if (isDescriptorTypeImageInfo(descType)) {
            if (!stride) stride = sizeof(VkDescriptorImageInfo);
            write.pImageInfo = &imageInfos[currImageInfoIndex];
            for (uint32_t j = 0; j < descCount; ++j) {
                imageInfos[currImageInfoIndex++] =
                    *reinterpret_cast<const VkDescriptorImageInfo*>(userBuffer + offset +
                                                                    j * stride);
            }
        } else if (isDescriptorTypeBufferInfo(descType)) {
            if (!stride) stride = sizeof(VkDescriptorBufferInfo);
            write.pBufferInfo = &bufferInfos[currBufferInfoIndex];
            for (uint32_t j = 0; j < descCount; ++j) {
                bufferInfos[currBufferInfoIndex++] =
                    *reinterpret_cast<const VkDescriptorBufferInfo*>(userBuffer + offset +
                                                                     j * stride);
            }
        } else if (isDescriptorTypeBufferView(descType)) {
            if (!stride) stride = sizeof(VkBufferView);
            write.pTexelBufferView = &bufferViews[currBufferViewIndex];
            for (uint32_t j = 0; j < descCount; ++j) {
                bufferViews[currBufferViewIndex++] =
                    *reinterpret_cast<const VkBufferView*>(userBuffer + offset + j * stride);
            }
        } else if (isDescriptorTypeInlineUniformBlock(descType)) {
            VkWriteDescriptorSetInlineUniformBlock& iub =
                inlineUniformBlocks[currInlineUniformBlockIndex++];
            iub = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK,
                .pNext = nullptr,
                .dataSize = descCount,
                .pData = userBuffer + offset,
            };
            write.pNext = &iub;
        }
    }

    if (!pPushDescriptorSetWithTemplateInfo->pNext &&
        pPushDescriptorSetWithTemplateInfo->layout != VK_NULL_HANDLE) {
        if (khr) {
            on_vkCmdPushDescriptorSetKHR(context, commandBuffer, pipelineBindPoint,
                                         pPushDescriptorSetWithTemplateInfo->layout,
                                         pPushDescriptorSetWithTemplateInfo->set,
                                         static_cast<uint32_t>(writes.size()), writes.data());
        } else {
            on_vkCmdPushDescriptorSet(context, commandBuffer, pipelineBindPoint,
                                      pPushDescriptorSetWithTemplateInfo->layout,
                                      pPushDescriptorSetWithTemplateInfo->set,
                                      static_cast<uint32_t>(writes.size()), writes.data());
        }
        return;
    }

    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_ALL;
    if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;
    } else if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
        stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    } else if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
        stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    }

    VkPushDescriptorSetInfo pushInfo = {
        .sType = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO,
        .pNext = pPushDescriptorSetWithTemplateInfo->pNext,
        .stageFlags = stageFlags,
        .layout = pPushDescriptorSetWithTemplateInfo->layout,
        .set = pPushDescriptorSetWithTemplateInfo->set,
        .descriptorWriteCount = static_cast<uint32_t>(writes.size()),
        .pDescriptorWrites = writes.data(),
    };
    if (khr) {
        on_vkCmdPushDescriptorSet2KHR(context, commandBuffer, &pushInfo);
    } else {
        on_vkCmdPushDescriptorSet2(context, commandBuffer, &pushInfo);
    }
}

#ifdef LINUX_GUEST_BUILD
static void fillEmulatedDrmFormatModPropsList(
    VkFormat format, const VkFormatProperties* pFormatProperties,
    VkDrmFormatModifierPropertiesListEXT* emulatedDrmFmtModPropsList) {
    mesa_logd(
        "VkDrmFormatModifierPropertiesListEXT: emulating DRM_FORMAT_MOD_LINEAR with optimal tiling "
        "features");
    VkFormatFeatureFlags tilingFeatures = (pFormatProperties && getVirglFormat(format))
                                              ? pFormatProperties->optimalTilingFeatures
                                              : 0;
    if (!tilingFeatures) {
        emulatedDrmFmtModPropsList->drmFormatModifierCount = 0;
        return;
    }
    uint32_t count = emulatedDrmFmtModPropsList->drmFormatModifierCount;
    emulatedDrmFmtModPropsList->drmFormatModifierCount = 1;
    if (emulatedDrmFmtModPropsList->pDrmFormatModifierProperties && count > 0) {
        emulatedDrmFmtModPropsList->pDrmFormatModifierProperties[0] = {
            .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
            .drmFormatModifierPlaneCount = 1,
            .drmFormatModifierTilingFeatures = tilingFeatures,
        };
    }
}
#endif

void ResourceTracker::on_vkGetPhysicalDeviceFormatProperties2(
    void* context, VkPhysicalDevice physicalDevice, VkFormat format,
    VkFormatProperties2* pFormatProperties) {
    VkEncoder* enc = (VkEncoder*)context;

#ifdef LINUX_GUEST_BUILD
    VkDrmFormatModifierPropertiesListEXT* emulatedDrmFmtModPropsList =
        vk_find_struct(pFormatProperties, DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT);
    if (emulatedDrmFmtModPropsList && doImageDrmFormatModifierEmulation(physicalDevice)) {
        // Unlink DRM modifier properties from the chain sent to host so host unmarshaler
        // doesn't write out-of-bounds on unpadded guest buffers.
        emulatedDrmFmtModPropsList = vk_extract_struct<VkDrmFormatModifierPropertiesListEXT>(
            pFormatProperties, VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT);

        enc->vkGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties,
                                                  true /* do lock */);

        __vk_append_struct(pFormatProperties, emulatedDrmFmtModPropsList);

        fillEmulatedDrmFormatModPropsList(format, &pFormatProperties->formatProperties,
                                          emulatedDrmFmtModPropsList);
        return;
    }
#endif

    enc->vkGetPhysicalDeviceFormatProperties2(physicalDevice, format, pFormatProperties,
                                              true /* do lock */);
}

void ResourceTracker::on_vkGetPhysicalDeviceFormatProperties2KHR(
    void* context, VkPhysicalDevice physicalDevice, VkFormat format,
    VkFormatProperties2* pFormatProperties) {
    on_vkGetPhysicalDeviceFormatProperties2(context, physicalDevice, format, pFormatProperties);
}

VkResult ResourceTracker::on_vkGetPhysicalDeviceImageFormatProperties2(
    void* context, VkResult input_result, VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceImageFormatInfo2* pImageFormatInfo,
    VkImageFormatProperties2* pImageFormatProperties) {
    VkEncoder* enc = (VkEncoder*)context;
    (void)input_result;

    VkPhysicalDeviceImageFormatInfo2 localImageFormatInfo = *pImageFormatInfo;

    uint32_t supportedHandleType = 0;
    VkExternalImageFormatProperties* ext_img_properties =
        vk_find_struct(pImageFormatProperties, EXTERNAL_IMAGE_FORMAT_PROPERTIES);

#ifdef VK_USE_PLATFORM_FUCHSIA

    constexpr VkFormat kExternalImageSupportedFormats[] = {
        VK_FORMAT_B8G8R8A8_SINT,  VK_FORMAT_B8G8R8A8_UNORM,   VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_B8G8R8A8_SNORM, VK_FORMAT_B8G8R8A8_SSCALED, VK_FORMAT_B8G8R8A8_USCALED,
        VK_FORMAT_R8G8B8A8_SINT,  VK_FORMAT_R8G8B8A8_UNORM,   VK_FORMAT_R8G8R8A8_SRGB,
        VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_SSCALED, VK_FORMAT_R8G8B8A8_USCALED,
        VK_FORMAT_R8_UNORM,       VK_FORMAT_R8_UINT,          VK_FORMAT_R8_USCALED,
        VK_FORMAT_R8_SNORM,       VK_FORMAT_R8_SINT,          VK_FORMAT_R8_SSCALED,
        VK_FORMAT_R8_SRGB,        VK_FORMAT_R8G8_UNORM,       VK_FORMAT_R8G8_UINT,
        VK_FORMAT_R8G8_USCALED,   VK_FORMAT_R8G8_SNORM,       VK_FORMAT_R8G8_SINT,
        VK_FORMAT_R8G8_SSCALED,   VK_FORMAT_R8G8_SRGB,
    };

    if (ext_img_properties) {
        if (std::find(std::begin(kExternalImageSupportedFormats),
                      std::end(kExternalImageSupportedFormats),
                      pImageFormatInfo->format) == std::end(kExternalImageSupportedFormats)) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
    }
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA;
#endif

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    VkAndroidHardwareBufferUsageANDROID* output_ahw_usage =
        vk_find_struct(pImageFormatProperties, ANDROID_HARDWARE_BUFFER_USAGE_ANDROID);
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
#endif
#ifdef LINUX_GUEST_BUILD
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif
    const VkPhysicalDeviceExternalImageFormatInfo* ext_img_info =
        vk_find_struct_const(pImageFormatInfo, PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
    if (supportedHandleType && ext_img_info) {
        // 0 is a valid handleType so we don't check against 0
        if (ext_img_info->handleType != (ext_img_info->handleType & supportedHandleType)) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
    }

#ifdef LINUX_GUEST_BUILD
    const bool isLinuxDmaBufQuery =
        ext_img_info && (ext_img_info->handleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    if (isLinuxDmaBufQuery &&
        (!getVirglFormat(pImageFormatInfo->format) || pImageFormatInfo->type != VK_IMAGE_TYPE_2D)) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    const VkPhysicalDeviceImageDrmFormatModifierInfoEXT* drmFmtMod =
        vk_find_struct_const(pImageFormatInfo, PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT);
    VkDrmFormatModifierPropertiesListEXT* emulatedDrmFmtModPropsList = nullptr;
    if (drmFmtMod) {
        if (doImageDrmFormatModifierEmulation(physicalDevice)) {
            emulatedDrmFmtModPropsList =
                vk_find_struct(pImageFormatProperties, DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT);

            if (drmFmtMod->drmFormatModifier == DRM_FORMAT_MOD_LINEAR &&
                getVirglFormat(pImageFormatInfo->format)) {
                // Remove the drmFmtMod from the localImageFormatInfo
                vk_filter_struct(&localImageFormatInfo,
                                 PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT);
                mesa_logd(
                    "getPhysicalDeviceImageFormatProperties2: emulating DRM_FORMAT_MOD_LINEAR with "
                    "VK_IMAGE_TILING_OPTIMAL");
                localImageFormatInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            } else {
                return VK_ERROR_FORMAT_NOT_SUPPORTED;
            }
        } else {
            // Host supports DRM format modifiers => leave the input unchanged.
        }
    }
#endif  // LINUX_GUEST_BUILD

    VkResult hostRes;

    hostRes = enc->vkGetPhysicalDeviceImageFormatProperties2(
        physicalDevice, &localImageFormatInfo, pImageFormatProperties, true /* do lock */);

    if (hostRes != VK_SUCCESS) return hostRes;

#ifdef LINUX_GUEST_BUILD
    if (isLinuxDmaBufQuery && ext_img_properties &&
        !(ext_img_properties->externalMemoryProperties.externalMemoryFeatures &
          (VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
           VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    if (emulatedDrmFmtModPropsList) {
        VkFormatProperties formatProperties;
        enc->vkGetPhysicalDeviceFormatProperties(physicalDevice, localImageFormatInfo.format,
                                                 &formatProperties, true /* do lock */);
        fillEmulatedDrmFormatModPropsList(localImageFormatInfo.format, &formatProperties,
                                          emulatedDrmFmtModPropsList);
    }
    if (ext_img_properties) {
        if (ext_img_info) {
            if (ext_img_info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
                pImageFormatProperties->imageFormatProperties.maxMipLevels = 1;
                pImageFormatProperties->imageFormatProperties.maxArrayLayers = 1;
                ext_img_properties->externalMemoryProperties = {
                    .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                                              VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
                    .exportFromImportedHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                    .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                };
            }
        }
    }
#endif  // LINUX_GUEST_BUILD

#ifdef VK_USE_PLATFORM_FUCHSIA
    if (ext_img_properties) {
        if (ext_img_info) {
            if (static_cast<uint32_t>(ext_img_info->handleType) ==
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA) {
                ext_img_properties->externalMemoryProperties = {
                    .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                                              VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
                    .exportFromImportedHandleTypes =
                        VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA,
                    .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA,
                };
            }
        }
    }
#endif

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (output_ahw_usage) {
        output_ahw_usage->androidHardwareBufferUsage = getAndroidHardwareBufferUsageFromVkUsage(
            pImageFormatInfo->flags, pImageFormatInfo->usage);
    }
#endif
    if (ext_img_properties) {
        transformImpl_VkExternalMemoryProperties_fromhost(
            &ext_img_properties->externalMemoryProperties, 0);
    }
    return hostRes;
}

void ResourceTracker::on_vkGetPhysicalDeviceExternalBufferProperties(
    void* context, VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceExternalBufferInfo* pExternalBufferInfo,
    VkExternalBufferProperties* pExternalBufferProperties) {
    VkEncoder* enc = (VkEncoder*)context;

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    // Older versions of Goldfish's Gralloc did not support allocating AHARDWAREBUFFER_FORMAT_BLOB
    // with GPU usage (b/299520213).
    if (mGralloc->treatBlobAsImage() &&
        pExternalBufferInfo->handleType ==
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID) {
        pExternalBufferProperties->externalMemoryProperties.externalMemoryFeatures = 0;
        pExternalBufferProperties->externalMemoryProperties.exportFromImportedHandleTypes = 0;
        pExternalBufferProperties->externalMemoryProperties.compatibleHandleTypes = 0;
        return;
    }
#endif

    uint32_t supportedHandleType = 0;
#ifdef VK_USE_PLATFORM_FUCHSIA
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA;
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
#endif
#ifdef LINUX_GUEST_BUILD
    supportedHandleType |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif
    if (supportedHandleType) {
        // 0 is a valid handleType so we can't check against 0
        if (pExternalBufferInfo->handleType !=
            (pExternalBufferInfo->handleType & supportedHandleType)) {
            pExternalBufferProperties->externalMemoryProperties.externalMemoryFeatures = 0;
            pExternalBufferProperties->externalMemoryProperties.exportFromImportedHandleTypes = 0;
            pExternalBufferProperties->externalMemoryProperties.compatibleHandleTypes =
                pExternalBufferInfo->handleType;
            return;
        }
    }

    enc->vkGetPhysicalDeviceExternalBufferProperties(physicalDevice, pExternalBufferInfo,
                                                     pExternalBufferProperties, true /* do lock */);

    transformImpl_VkExternalMemoryProperties_fromhost(
        &pExternalBufferProperties->externalMemoryProperties, 0);
}

void ResourceTracker::on_vkGetPhysicalDeviceExternalSemaphoreProperties(
    void*, VkPhysicalDevice, const VkPhysicalDeviceExternalSemaphoreInfo* pExternalSemaphoreInfo,
    VkExternalSemaphoreProperties* pExternalSemaphoreProperties) {
    (void)pExternalSemaphoreInfo;
    (void)pExternalSemaphoreProperties;
#ifdef VK_USE_PLATFORM_FUCHSIA
    if (pExternalSemaphoreInfo->handleType ==
        static_cast<uint32_t>(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_ZIRCON_EVENT_BIT_FUCHSIA)) {
        pExternalSemaphoreProperties->compatibleHandleTypes |=
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_ZIRCON_EVENT_BIT_FUCHSIA;
        pExternalSemaphoreProperties->exportFromImportedHandleTypes |=
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_ZIRCON_EVENT_BIT_FUCHSIA;
        pExternalSemaphoreProperties->externalSemaphoreFeatures |=
            VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT |
            VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    }
#else
    const VkSemaphoreTypeCreateInfo* semaphoreTypeCi =
        vk_find_struct_const(pExternalSemaphoreInfo, SEMAPHORE_TYPE_CREATE_INFO);
    bool isSemaphoreTimeline =
        semaphoreTypeCi != nullptr && semaphoreTypeCi->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE;
    if (isSemaphoreTimeline) {
        // b/304373623
        // dEQP-VK.api.external.semaphore.sync_fd#info_timeline
        pExternalSemaphoreProperties->compatibleHandleTypes = 0;
        pExternalSemaphoreProperties->exportFromImportedHandleTypes = 0;
        pExternalSemaphoreProperties->externalSemaphoreFeatures = 0;
    } else if (pExternalSemaphoreInfo->handleType ==
               VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT) {
        pExternalSemaphoreProperties->compatibleHandleTypes |=
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        pExternalSemaphoreProperties->exportFromImportedHandleTypes |=
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        pExternalSemaphoreProperties->externalSemaphoreFeatures |=
            VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT |
            VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    }
#endif  // VK_USE_PLATFORM_FUCHSIA
}

void ResourceTracker::registerEncoderCleanupCallback(const VkEncoder* encoder, void* object,
                                                     CleanupCallback callback) {
    std::lock_guard<std::recursive_mutex> lock(mLock);
    auto& callbacks = mEncoderCleanupCallbacks[encoder];
    callbacks[object] = callback;
}

void ResourceTracker::unregisterEncoderCleanupCallback(const VkEncoder* encoder, void* object) {
    std::lock_guard<std::recursive_mutex> lock(mLock);
    mEncoderCleanupCallbacks[encoder].erase(object);
}

void ResourceTracker::onEncoderDeleted(const VkEncoder* encoder) {
    std::unique_lock<std::recursive_mutex> lock(mLock);
    if (mEncoderCleanupCallbacks.find(encoder) == mEncoderCleanupCallbacks.end()) return;

    std::unordered_map<void*, CleanupCallback> callbackCopies = mEncoderCleanupCallbacks[encoder];

    mEncoderCleanupCallbacks.erase(encoder);
    lock.unlock();

    for (auto it : callbackCopies) {
        it.second();
    }
}

CommandBufferStagingStream::Alloc ResourceTracker::getAlloc() {
    if (mFeatureInfo.hasVulkanAuxCommandMemory) {
        return [this](size_t size) -> CommandBufferStagingStream::Memory {
            VkMemoryAllocateInfo info{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = nullptr,
                .allocationSize = size,
                .memoryTypeIndex = VK_MAX_MEMORY_TYPES
            };

            auto enc = ResourceTracker::getThreadLocalEncoder();
            VkDevice device = VK_NULL_HANDLE;
            VkDeviceMemory vkDeviceMem = VK_NULL_HANDLE;
            VkResult result = getCoherentMemory(&info, nullptr, enc, device, &vkDeviceMem);
            if (result != VK_SUCCESS) {
                mesa_loge("Failed to get coherent memory %u", result);
                return {.deviceMemory = VK_NULL_HANDLE, .ptr = nullptr};
            }

            VK_FROM_HANDLE(gfxstream_vk_device_memory, memObj, vkDeviceMem);
            if (!memObj) {
                mesa_loge("Coherent memory allocated %u not found", result);
                return {.deviceMemory = VK_NULL_HANDLE, .ptr = nullptr};
            }

            return {.deviceMemory = vkDeviceMem, .ptr = memObj->ptr};
        };
    }
    return nullptr;
}

CommandBufferStagingStream::Free ResourceTracker::getFree() {
    if (mFeatureInfo.hasVulkanAuxCommandMemory) {
        return [this](const CommandBufferStagingStream::Memory& memory) {
            auto coherentMemory =
                freeCoherentMemory(VK_NULL_HANDLE, memory.deviceMemory, nullptr);
            coherentMemory = nullptr;
        };
    }
    return nullptr;
}

VkResult ResourceTracker::on_vkBeginCommandBuffer(void* context, VkResult input_result,
                                                  VkCommandBuffer commandBuffer,
                                                  const VkCommandBufferBeginInfo* pBeginInfo) {
    (void)context;

    resetCommandBufferStagingInfo(commandBuffer, true /* also reset primaries */,
                                  true /* also clear pending descriptor sets */);

    VkEncoder* enc = ResourceTracker::getCommandBufferEncoder(commandBuffer);
    (void)input_result;

    VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, commandBuffer);
    cb->flags = pBeginInfo->flags;

    VkCommandBufferBeginInfo modifiedBeginInfo;

    if (pBeginInfo->pInheritanceInfo && !cb->isSecondary) {
        modifiedBeginInfo = *pBeginInfo;
        modifiedBeginInfo.pInheritanceInfo = nullptr;
        pBeginInfo = &modifiedBeginInfo;
    }

    if (!supportsDeferredCommands()) {
        return enc->vkBeginCommandBuffer(commandBuffer, pBeginInfo, true /* do lock */);
    }

    enc->vkBeginCommandBufferAsyncGOOGLE(commandBuffer, pBeginInfo, true /* do lock */);

    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkEndCommandBuffer(void* context, VkResult input_result,
                                                VkCommandBuffer commandBuffer) {
    VkEncoder* enc = (VkEncoder*)context;
    (void)input_result;

    if (!supportsDeferredCommands()) {
        return enc->vkEndCommandBuffer(commandBuffer, true /* do lock */);
    }

    enc->vkEndCommandBufferAsyncGOOGLE(commandBuffer, true /* do lock */);

    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkResetCommandBuffer(void* context, VkResult input_result,
                                                  VkCommandBuffer commandBuffer,
                                                  VkCommandBufferResetFlags flags) {
    VkEncoder* enc = (VkEncoder*)context;
    (void)input_result;

    if (!supportsDeferredCommands()) {
        VkResult res = enc->vkResetCommandBuffer(commandBuffer, flags, true /* do lock */);
        resetCommandBufferStagingInfo(commandBuffer, true /* also reset primaries */,
                                      true /* also clear pending descriptor sets */);
        return res;
    }

    enc->vkResetCommandBufferAsyncGOOGLE(commandBuffer, flags, true /* do lock */);
    resetCommandBufferStagingInfo(commandBuffer, true /* also reset primaries */,
                                  true /* also clear pending descriptor sets */);
    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkCreateImageView(void* context, VkResult input_result,
                                               VkDevice device,
                                               const VkImageViewCreateInfo* pCreateInfo,
                                               const VkAllocationCallbacks* pAllocator,
                                               VkImageView* pView) {
    VkEncoder* enc = (VkEncoder*)context;
    (void)input_result;

    VkImageViewCreateInfo localCreateInfo = vk_make_orphan_copy(*pCreateInfo);

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    vk_struct_chain_iterator structChainIter = vk_make_chain_iterator(&localCreateInfo);
    if (pCreateInfo->format == VK_FORMAT_UNDEFINED) {
        VK_FROM_HANDLE(gfxstream_vk_image, img, pCreateInfo->image);
        if (img && img->hasExternalFormat) {
            localCreateInfo.format = vk_format_from_fourcc(img->externalFourccFormat);
        }
    }
    VkSamplerYcbcrConversionInfo localVkSamplerYcbcrConversionInfo;
    const VkSamplerYcbcrConversionInfo* samplerYcbcrConversionInfo =
        vk_find_struct_const(pCreateInfo, SAMPLER_YCBCR_CONVERSION_INFO);
    if (samplerYcbcrConversionInfo && gfxstream_vk_sampler_ycbcr_conversion_to_host_u64(
                                          samplerYcbcrConversionInfo->conversion) != 0) {
        localVkSamplerYcbcrConversionInfo = vk_make_orphan_copy(*samplerYcbcrConversionInfo);
        vk_append_struct(&structChainIter, &localVkSamplerYcbcrConversionInfo);
    }
#endif

    return enc->vkCreateImageView(device, &localCreateInfo, pAllocator, pView, true /* do lock */);
}

static void dynarrayAppendUniqueCmdBuf(struct util_dynarray* arr, VkCommandBuffer cmdBuf) {
    util_dynarray_foreach(arr, VkCommandBuffer, elem) {
        if (*elem == cmdBuf) return;
    }
    util_dynarray_append_typed(arr, VkCommandBuffer, cmdBuf);
}

static void dynarrayEraseCmdBuf(struct util_dynarray* arr, VkCommandBuffer cmdBuf) {
    unsigned count = util_dynarray_num_elements(arr, VkCommandBuffer);
    VkCommandBuffer* data = static_cast<VkCommandBuffer*>(arr->data);
    for (unsigned i = 0; i < count; ++i) {
        if (data[i] == cmdBuf) {
            data[i] = data[count - 1];
            arr->size -= sizeof(VkCommandBuffer);
            return;
        }
    }
}

void ResourceTracker::on_vkCmdExecuteCommands(void* context, VkCommandBuffer commandBuffer,
                                              uint32_t commandBufferCount,
                                              const VkCommandBuffer* pCommandBuffers) {
    VkEncoder* enc = (VkEncoder*)context;

    if (!mFeatureInfo.hasVulkanQueueSubmitWithCommands) {
        enc->vkCmdExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers,
                                  true /* do lock */);
        return;
    }

    VK_FROM_HANDLE(gfxstream_vk_command_buffer, primary, commandBuffer);
    for (uint32_t i = 0; i < commandBufferCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_command_buffer, secondary, pCommandBuffers[i]);
        dynarrayAppendUniqueCmdBuf(&secondary->superObjects, commandBuffer);
        dynarrayAppendUniqueCmdBuf(&primary->subObjects, pCommandBuffers[i]);
    }

    enc->vkCmdExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers,
                              true /* do lock */);
}

void ResourceTracker::on_vkCmdBindDescriptorSets(void* context, VkCommandBuffer commandBuffer,
                                                 VkPipelineBindPoint pipelineBindPoint,
                                                 VkPipelineLayout layout, uint32_t firstSet,
                                                 uint32_t descriptorSetCount,
                                                 const VkDescriptorSet* pDescriptorSets,
                                                 uint32_t dynamicOffsetCount,
                                                 const uint32_t* pDynamicOffsets) {
    VkEncoder* enc = (VkEncoder*)context;

    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate)
        addPendingDescriptorSets(commandBuffer, descriptorSetCount, pDescriptorSets);

    enc->vkCmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet,
                                 descriptorSetCount, pDescriptorSets, dynamicOffsetCount,
                                 pDynamicOffsets, true /* do lock */);
}

void ResourceTracker::on_vkCmdBindDescriptorSets2(
    void* context, VkCommandBuffer commandBuffer,
    const VkBindDescriptorSetsInfo* pBindDescriptorSetsInfo) {
    if (!pBindDescriptorSetsInfo) return;
    VkEncoder* enc = (VkEncoder*)context;

    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
        addPendingDescriptorSets(commandBuffer, pBindDescriptorSetsInfo->descriptorSetCount,
                                 pBindDescriptorSetsInfo->pDescriptorSets);
    }

    enc->vkCmdBindDescriptorSets2(commandBuffer, pBindDescriptorSetsInfo, true /* do lock */);
}

void ResourceTracker::on_vkCmdBindDescriptorSets2KHR(
    void* context, VkCommandBuffer commandBuffer,
    const VkBindDescriptorSetsInfo* pBindDescriptorSetsInfo) {
    if (!pBindDescriptorSetsInfo) return;
    VkEncoder* enc = (VkEncoder*)context;

    if (mFeatureInfo.hasVulkanBatchedDescriptorSetUpdate) {
        addPendingDescriptorSets(commandBuffer, pBindDescriptorSetsInfo->descriptorSetCount,
                                 pBindDescriptorSetsInfo->pDescriptorSets);
    }

    enc->vkCmdBindDescriptorSets2KHR(commandBuffer, pBindDescriptorSetsInfo, true);
}

void ResourceTracker::on_vkCmdPipelineBarrier(
    void* context, VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers) {
    VkEncoder* enc = (VkEncoder*)context;

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    std::vector<VkImageMemoryBarrier> updatedImageMemoryBarriers;
    updatedImageMemoryBarriers.reserve(imageMemoryBarrierCount);
    for (uint32_t i = 0; i < imageMemoryBarrierCount; i++) {
        VkImageMemoryBarrier barrier = pImageMemoryBarriers[i];

        // Unfortunetly, Android does not yet have a mechanism for sharing the expected
        // VkImageLayout when passing around AHardwareBuffer-s so many existing users
        // that import AHardwareBuffer-s into VkImage-s/VkDeviceMemory-s simply use
        // VK_IMAGE_LAYOUT_UNDEFINED. However, the Vulkan spec's image layout transition
        // sections says "If the old layout is VK_IMAGE_LAYOUT_UNDEFINED, the contents of
        // that range may be discarded." Some Vulkan drivers have been observed to actually
        // perform the discard which leads to AHardwareBuffer-s being unintentionally
        // cleared. See go/ahb-vkimagelayout for more information.
        if (barrier.srcQueueFamilyIndex != barrier.dstQueueFamilyIndex &&
            (barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL ||
             barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT) &&
            barrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            // This is not a complete solution as the Vulkan spec does not require that
            // Vulkan drivers perform a no-op in the case when oldLayout equals newLayout
            // but this has been observed to be enough to work for now to avoid clearing
            // out images.
            // TODO(b/236179843): figure out long term solution.
            barrier.oldLayout = barrier.newLayout;
        }

        updatedImageMemoryBarriers.push_back(barrier);
    }

    enc->vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags,
                              memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                              pBufferMemoryBarriers, updatedImageMemoryBarriers.size(),
                              updatedImageMemoryBarriers.data(), true /* do lock */);
#else
    enc->vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags,
                              memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                              pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers,
                              true /* do lock */);
#endif
}

void ResourceTracker::on_vkCmdClearColorImage(void* context, VkCommandBuffer commandBuffer,
                                              VkImage image, VkImageLayout imageLayout,
                                              const VkClearColorValue* pColor, uint32_t rangeCount,
                                              const VkImageSubresourceRange* pRanges) {
    VkEncoder* enc = (VkEncoder*)context;
    if (!pColor) {
        mesa_loge("%s: Null VkClearColorValue requested", __func__);
        return;
    }
    VK_FROM_HANDLE(gfxstream_vk_image, imageInfo, image);
    if (!imageInfo) {
        mesa_loge("%s: Failed to find image required for vkCmdClearColorImage", __func__);
        return;
    }

    VkClearColorValue convertedColor = *pColor;

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    // Color buffer image on the host will be created with UNORM format to ensure
    // it'll have the identical parameters, so we need to convert the linearized
    // clear color back to sRGB at this point.
    // TODO(b/420857458): revise the allocation logic to support mutable formats better
    VkFormat actualFormat = imageInfo->createInfo.format;
    if (imageInfo->hasAnb && srgbFormatNeedsConversionForClearColor(actualFormat)) {
        // Perform linear to srgb conversion
        // Backing image is UNORM for vkCmdClearColorImage so we convert pColor
        convertedColor.float32[0] = linearChannelToSRGB(convertedColor.float32[0]);
        convertedColor.float32[1] = linearChannelToSRGB(convertedColor.float32[1]);
        convertedColor.float32[2] = linearChannelToSRGB(convertedColor.float32[2]);
    }
#endif
    enc->vkCmdClearColorImage(commandBuffer, image, imageLayout, &convertedColor, rangeCount,
                              pRanges, true);
}

void ResourceTracker::on_vkDestroyDescriptorSetLayout(void* context, VkDevice device,
                                                      VkDescriptorSetLayout descriptorSetLayout,
                                                      const VkAllocationCallbacks* pAllocator) {
    if (!descriptorSetLayout) return;
    if (pAllocator) {
        VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, setLayout, descriptorSetLayout);
        if (setLayout && setLayout->layoutInfo) {
            setLayout->layoutInfo->destroyAllocator = *pAllocator;
        }
    }
    decDescriptorSetLayoutRef(context, device, descriptorSetLayout, pAllocator);
}

VkResult ResourceTracker::on_vkAllocateCommandBuffers(
    void* context, VkResult input_result, VkDevice device,
    const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers) {
    (void)input_result;

    VkEncoder* enc = (VkEncoder*)context;
    VkResult res =
        enc->vkAllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers, true /* do lock */);
    if (VK_SUCCESS != res) return res;

    for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; ++i) {
        VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, pCommandBuffers[i]);
        cb->isSecondary = pAllocateInfo->level == VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        cb->device = device;
    }

    return res;
}

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
VkResult ResourceTracker::exportSyncFdForQSRILocked(VkImage image, int* fd) {
    mesa_logd("%s: call for image %p host image handle 0x%llx\n", __func__, (void*)image,
              (unsigned long long)gfxstream_vk_image_to_host_u64(image));

    if (mFeatureInfo.hasVirtioGpuNativeSync) {
        struct VirtGpuExecBuffer exec = {};
        struct gfxstreamCreateQSRIExportVK exportQSRI = {};
        VirtGpuDevice* instance = VirtGpuDevice::getInstance();

        uint64_t hostImageHandle = gfxstream_vk_image_to_host_u64(image);

        exportQSRI.hdr.opCode = GFXSTREAM_CREATE_QSRI_EXPORT_VK;
        exportQSRI.imageHandleLo = (uint32_t)hostImageHandle;
        exportQSRI.imageHandleHi = (uint32_t)(hostImageHandle >> 32);

        exec.command = static_cast<void*>(&exportQSRI);
        exec.command_size = sizeof(exportQSRI);
        exec.flags = kFenceOut | kRingIdx;
        if (instance->execBuffer(exec, nullptr)) return VK_ERROR_OUT_OF_HOST_MEMORY;

        *fd = exec.handle.osHandle;
    } else {
#if GFXSTREAM_ENABLE_GUEST_GOLDFISH
        ensureSyncDeviceFd();
        goldfish_sync_queue_work(
            mSyncDeviceFd, gfxstream_vk_image_to_host_u64(image) /* the handle */,
            GOLDFISH_SYNC_VULKAN_QSRI /* thread handle (doubling as type field) */, fd);
#endif
    }

    MESA_TRACE_SCOPE("%s: got fd: %d\n", __func__, *fd);
    VK_FROM_HANDLE(gfxstream_vk_image, imageInfo, image);
    if (imageInfo) {
        // Remove any pending QSRI sync fds that are already signaled.
        auto syncFdIt = imageInfo->pendingQsriSyncFds.begin();
        while (syncFdIt != imageInfo->pendingQsriSyncFds.end()) {
            int syncFd = *syncFdIt;
            int syncWaitRet = mSyncHelper->wait(syncFd, /*timeout msecs*/ 0);
            if (syncWaitRet == 0) {
                // Sync fd is signaled.
                syncFdIt = imageInfo->pendingQsriSyncFds.erase(syncFdIt);
                mSyncHelper->close(syncFd);
            } else {
                if (errno != ETIME) {
                    mesa_loge("%s: Failed to wait for pending QSRI sync: sterror: %s errno: %d",
                              __func__, strerror(errno), errno);
                }
                break;
            }
        }

        int syncFdDup = mSyncHelper->dup(*fd);
        if (syncFdDup < 0) {
            mesa_loge("%s: Failed to dup() QSRI sync fd : sterror: %s errno: %d", __func__,
                      strerror(errno), errno);
        } else {
            imageInfo->pendingQsriSyncFds.push_back(syncFdDup);
        }
    }

    return VK_SUCCESS;
}

VkResult ResourceTracker::on_vkQueueSignalReleaseImageANDROID(void* context, VkResult input_result,
                                                              VkQueue queue,
                                                              uint32_t waitSemaphoreCount,
                                                              const VkSemaphore* pWaitSemaphores,
                                                              VkImage image, int* pNativeFenceFd) {
    (void)input_result;

    VkEncoder* enc = (VkEncoder*)context;

    if (!mFeatureInfo.hasVulkanAsyncQsri) {
        return enc->vkQueueSignalReleaseImageANDROID(queue, waitSemaphoreCount, pWaitSemaphores,
                                                     image, pNativeFenceFd, true /* lock */);
    }

    if (!image) {
        if (pNativeFenceFd) *pNativeFenceFd = -1;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    enc->vkQueueSignalReleaseImageANDROIDAsyncGOOGLE(queue, waitSemaphoreCount, pWaitSemaphores,
                                                     image, true /* lock */);

    std::lock_guard<std::recursive_mutex> lock(mLock);
    VkResult result;
    if (pNativeFenceFd) {
        result = exportSyncFdForQSRILocked(image, pNativeFenceFd);
    } else {
        int syncFd;
        result = exportSyncFdForQSRILocked(image, &syncFd);

        if (syncFd >= 0) {
            mSyncHelper->close(syncFd);
        }
    }

    return result;
}
#endif

VkResult ResourceTracker::on_vkCreateGraphicsPipelines(
    void* context, VkResult input_result, VkDevice device, VkPipelineCache pipelineCache,
    uint32_t createInfoCount, const VkGraphicsPipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
    (void)input_result;
    VkEncoder* enc = (VkEncoder*)context;
    std::vector<VkGraphicsPipelineCreateInfo> localCreateInfos(pCreateInfos,
                                                               pCreateInfos + createInfoCount);
    std::vector<VkPipelineRenderingCreateInfo> localRenderingInfos(createInfoCount);
    std::vector<VkPipelineRenderingCreateInfo*> extractedRenderingInfos(createInfoCount, nullptr);

    for (uint32_t i = 0; i < createInfoCount; ++i) {
        VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo = localCreateInfos[i];
        VkPipelineCreateFlags2 flags2 = graphicsPipelineCreateInfo.flags;
        if (const auto* flags2Info = vk_find_struct_const(&graphicsPipelineCreateInfo,
                                                          PIPELINE_CREATE_FLAGS_2_CREATE_INFO)) {
            flags2 = flags2Info->flags;
        }
        const auto* gplInfo = vk_find_struct_const(&graphicsPipelineCreateInfo,
                                                   GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
        const bool isPipelineLibrary = (flags2 & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) != 0;
        const bool hasDirectFragmentOutput =
            gplInfo ? (gplInfo->flags &
                       VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT) != 0
                    : !isPipelineLibrary;
        // dEQP-VK.api.pipeline.pipeline_invalid_pointers_unused_structs#graphics
        bool requireViewportState = false;
        // VUID-VkGraphicsPipelineCreateInfo-rasterizerDiscardEnable-00750
        requireViewportState |=
            graphicsPipelineCreateInfo.pRasterizationState != nullptr &&
            graphicsPipelineCreateInfo.pRasterizationState->rasterizerDiscardEnable == VK_FALSE;
        // VUID-VkGraphicsPipelineCreateInfo-pViewportState-04892
#ifdef VK_EXT_extended_dynamic_state2
        if (!requireViewportState && graphicsPipelineCreateInfo.pRasterizationState != nullptr &&
            graphicsPipelineCreateInfo.pDynamicState) {
            for (uint32_t i = 0; i < graphicsPipelineCreateInfo.pDynamicState->dynamicStateCount;
                 i++) {
                if (VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT ==
                    graphicsPipelineCreateInfo.pDynamicState->pDynamicStates[i]) {
                    requireViewportState = true;
                    break;
                }
            }
        }
#endif  // VK_EXT_extended_dynamic_state2
        if (!requireViewportState) {
            graphicsPipelineCreateInfo.pViewportState = nullptr;
        }

        // Partial pipeline libraries without pre-rasterization state have pRasterizationState == nullptr
        // but may still include fragment shader or fragment output state.
        bool shouldIncludeFragmentShaderState =
            (graphicsPipelineCreateInfo.pRasterizationState == nullptr) || requireViewportState;

        // VUID-VkGraphicsPipelineCreateInfo-rasterizerDiscardEnable-00751
        if (!shouldIncludeFragmentShaderState) {
            graphicsPipelineCreateInfo.pMultisampleState = nullptr;
        }

        bool forceDepthStencilState = false;
        bool forceColorBlendState = false;

        const VkPipelineRenderingCreateInfo* pipelineRenderingInfo =
            vk_find_struct_const(&graphicsPipelineCreateInfo, PIPELINE_RENDERING_CREATE_INFO);

        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06579 / 06580:
        // Attachment format fields in VkPipelineRenderingCreateInfo are ignored unless
        // the pipeline provides fragment output interface state directly and renderPass
        // is VK_NULL_HANDLE. Scrub ignored fields so the autogenerated marshaler never
        // dereferences uninitialized pointers.
        const bool ignoreRenderingFormats =
            !hasDirectFragmentOutput || graphicsPipelineCreateInfo.renderPass != VK_NULL_HANDLE;
        if (pipelineRenderingInfo &&
            ((ignoreRenderingFormats &&
              (pipelineRenderingInfo->colorAttachmentCount != 0 ||
               pipelineRenderingInfo->pColorAttachmentFormats != nullptr)) ||
             (pipelineRenderingInfo->colorAttachmentCount == 0 &&
              pipelineRenderingInfo->pColorAttachmentFormats != nullptr))) {
            auto* origInfo = const_cast<VkGraphicsPipelineCreateInfo*>(&pCreateInfos[i]);
            extractedRenderingInfos[i] = vk_extract_struct<VkPipelineRenderingCreateInfo>(
                origInfo, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
            if (extractedRenderingInfos[i]) {
                localRenderingInfos[i] = vk_make_orphan_copy(*extractedRenderingInfos[i]);
                localRenderingInfos[i].colorAttachmentCount = 0;
                localRenderingInfos[i].pColorAttachmentFormats = nullptr;
                __vk_append_struct(origInfo, &localRenderingInfos[i]);
                graphicsPipelineCreateInfo.pNext = origInfo->pNext;
                pipelineRenderingInfo = &localRenderingInfos[i];
            }
        }

        if (pipelineRenderingInfo) {
            forceDepthStencilState |=
                pipelineRenderingInfo->depthAttachmentFormat != VK_FORMAT_UNDEFINED;
            forceDepthStencilState |=
                pipelineRenderingInfo->stencilAttachmentFormat != VK_FORMAT_UNDEFINED;
            forceColorBlendState |= pipelineRenderingInfo->colorAttachmentCount != 0;
        }

        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06043
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06044
        if ((!isPipelineLibrary && graphicsPipelineCreateInfo.renderPass == VK_NULL_HANDLE) ||
            !shouldIncludeFragmentShaderState) {
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06053
            if (!forceDepthStencilState) {
                graphicsPipelineCreateInfo.pDepthStencilState = nullptr;
            }
            if (!forceColorBlendState) {
                graphicsPipelineCreateInfo.pColorBlendState = nullptr;
            }
        }
    }

    VkResult res = enc->vkCreateGraphicsPipelines(device, pipelineCache, localCreateInfos.size(),
                                                  localCreateInfos.data(), pAllocator, pPipelines,
                                                  true /* do lock */);

    for (uint32_t i = createInfoCount; i > 0; --i) {
        if (extractedRenderingInfos[i - 1]) {
            auto* origInfo = const_cast<VkGraphicsPipelineCreateInfo*>(&pCreateInfos[i - 1]);
            vk_extract_struct<VkPipelineRenderingCreateInfo>(
                origInfo, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
            __vk_append_struct(origInfo, extractedRenderingInfos[i - 1]);
        }
    }
    return res;
}

VkResult ResourceTracker::on_vkCreatePipelineLayout(void* context, VkResult input_result,
                                                    VkDevice device,
                                                    const VkPipelineLayoutCreateInfo* pCreateInfo,
                                                    const VkAllocationCallbacks* pAllocator,
                                                    VkPipelineLayout* pPipelineLayout) {
    (void)input_result;
    VkEncoder* enc = (VkEncoder*)context;
    VkResult res = enc->vkCreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout,
                                               true /* do lock */);
    if (res != VK_SUCCESS || !pCreateInfo || !pCreateInfo->pSetLayouts) {
        return res;
    }

    VK_FROM_HANDLE(gfxstream_vk_pipeline_layout, pl, *pPipelineLayout);
    if (!pl) {
        return res;
    }

    bool anyImmutable = false;
    for (uint32_t s = 0; s < pCreateInfo->setLayoutCount && !anyImmutable; ++s) {
        if (!pCreateInfo->pSetLayouts[s]) continue;
        VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, dsl, pCreateInfo->pSetLayouts[s]);
        if (!dsl || !dsl->layoutInfo) continue;
        for (const auto& b : dsl->layoutInfo->bindings) {
            if (b.descriptorCount > 0 && b.pImmutableSamplers &&
                (b.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                 b.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)) {
                anyImmutable = true;
                break;
            }
        }
    }

    if (anyImmutable) {
        pl->immutableSamplers.resize(pCreateInfo->setLayoutCount);
        for (uint32_t s = 0; s < pCreateInfo->setLayoutCount; ++s) {
            if (!pCreateInfo->pSetLayouts[s]) continue;
            VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, dsl, pCreateInfo->pSetLayouts[s]);
            if (!dsl || !dsl->layoutInfo) continue;
            uint32_t maxBinding = 0;
            bool setHasImmutable = false;
            for (const auto& b : dsl->layoutInfo->bindings) {
                if (b.descriptorCount > 0 && b.pImmutableSamplers &&
                    (b.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                     b.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)) {
                    maxBinding = std::max(maxBinding, b.binding);
                    setHasImmutable = true;
                }
            }
            if (setHasImmutable) {
                pl->immutableSamplers[s].resize(maxBinding + 1, false);
                for (const auto& b : dsl->layoutInfo->bindings) {
                    if (b.descriptorCount > 0 && b.pImmutableSamplers &&
                        (b.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                         b.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)) {
                        pl->immutableSamplers[s][b.binding] = true;
                    }
                }
            }
        }
    }

    return res;
}

#ifdef LINUX_GUEST_BUILD
bool ResourceTracker::doImageDrmFormatModifierEmulation(VkPhysicalDevice physicalDevice) {
    VK_FROM_HANDLE(gfxstream_vk_physical_device, physDev, physicalDevice);
    if (physDev && physDev->base.instance) {
        return physDev->doImageDrmFormatModifierEmulation;
    }

    std::lock_guard<std::recursive_mutex> lock(mLock);
    bool hostSupportsImageDrmFormatModifiers =
        getHostDeviceExtensionIndex(VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) != -1;
    return !hostSupportsImageDrmFormatModifiers;
}
#endif

VkDevice ResourceTracker::getDevice(VkCommandBuffer commandBuffer) const {
    VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, commandBuffer);
    if (!cb) {
        return nullptr;
    }
    return cb->device;
}

// Resets staging stream for this command buffer and primary command buffers
// where this command buffer has been recorded. If requested, also clears the pending
// descriptor sets.
void ResourceTracker::resetCommandBufferStagingInfo(VkCommandBuffer commandBuffer,
                                                    bool alsoResetPrimaries,
                                                    bool alsoClearPendingDescriptorSets) {
    VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, commandBuffer);
    if (!cb) {
        return;
    }
    if (cb->privateEncoder) {
        sStaging.pushStaging((CommandBufferStagingStream*)cb->privateStream, cb->privateEncoder);
        cb->privateEncoder = nullptr;
        cb->privateStream = nullptr;
    }

    if (alsoClearPendingDescriptorSets && cb->userPtr) {
        CommandBufferPendingDescriptorSets* pendingSets =
            (CommandBufferPendingDescriptorSets*)cb->userPtr;
        pendingSets->sets.clear();
    }

    if (alsoResetPrimaries) {
        unsigned superCount = util_dynarray_num_elements(&cb->superObjects, VkCommandBuffer);
        if (superCount > 0) {
            VkCommandBuffer stackSupers[16];
            std::vector<VkCommandBuffer> heapSupers;
            VkCommandBuffer* supers = stackSupers;
            if (superCount > 16) {
                heapSupers.resize(superCount);
                supers = heapSupers.data();
            }
            memcpy(supers, cb->superObjects.data, superCount * sizeof(VkCommandBuffer));
            util_dynarray_clear(&cb->superObjects);
            for (unsigned i = 0; i < superCount; ++i) {
                resetCommandBufferStagingInfo(supers[i], alsoResetPrimaries,
                                              alsoClearPendingDescriptorSets);
            }
        }
    }

    util_dynarray_foreach(&cb->subObjects, VkCommandBuffer, elem) {
        VK_FROM_HANDLE(gfxstream_vk_command_buffer, subCb, *elem);
        if (subCb) {
            dynarrayEraseCmdBuf(&subCb->superObjects, commandBuffer);
        }
    }
    util_dynarray_clear(&cb->subObjects);
}

// Unlike resetCommandBufferStagingInfo, this does not always erase its
// superObjects pointers because the command buffer has merely been
// submitted, not reset.  However, if the command buffer was recorded with
// ONE_TIME_SUBMIT_BIT, then it will also reset its primaries.
//
// Also, we save the set of descriptor sets referenced by this command
// buffer because we only submitted the command buffer and it's possible to
// update the descriptor set again and re-submit the same command without
// recording it (Update-after-bind descriptor sets)
void ResourceTracker::resetCommandBufferPendingTopology(VkCommandBuffer commandBuffer) {
    VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, commandBuffer);
    if (cb->flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) {
        resetCommandBufferStagingInfo(commandBuffer, true /* reset primaries */,
                                      true /* clear pending descriptor sets */);
    } else {
        resetCommandBufferStagingInfo(commandBuffer, false /* Don't reset primaries */,
                                      false /* Don't clear pending descriptor sets */);
    }
}

void ResourceTracker::resetCommandPoolStagingInfo(VkCommandPool commandPool) {
    VK_FROM_HANDLE(gfxstream_vk_command_pool, p, commandPool);
    if (!p) return;

    list_for_each_entry_safe(struct vk_command_buffer, vk_cb, &p->base.command_buffers, pool_link) {
        auto* cb = reinterpret_cast<struct gfxstream_vk_command_buffer*>(vk_cb);
        resetCommandBufferStagingInfo(gfxstream_vk_command_buffer_to_handle(cb),
                                      true /* also reset primaries */,
                                      true /* also clear pending descriptor sets */);
    }
}

void ResourceTracker::clearCommandPool(VkCommandPool commandPool) {
    VK_FROM_HANDLE(gfxstream_vk_command_pool, p, commandPool);
    if (!p) return;

    VkDevice device =
        p->base.base.device
            ? gfxstream_vk_device_to_handle(
                  reinterpret_cast<struct gfxstream_vk_device*>(p->base.base.device))
            : VK_NULL_HANDLE;

    list_for_each_entry_safe(struct vk_command_buffer, vk_cb, &p->base.command_buffers, pool_link) {
        auto* cb = reinterpret_cast<struct gfxstream_vk_command_buffer*>(vk_cb);
        VkCommandBuffer cbHandle = gfxstream_vk_command_buffer_to_handle(cb);
        unregister_VkCommandBuffer(cbHandle);
        delete_gfxstream_vk_command_buffer(device, cbHandle, nullptr);
    }
}

const VkPhysicalDeviceMemoryProperties& ResourceTracker::getPhysicalDeviceMemoryProperties(
    void* context, VkDevice device, VkPhysicalDevice physicalDevice) {
    if (device != VK_NULL_HANDLE) {
        VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
        if (dev && dev->mem_props.memoryTypeCount > 0) {
            return dev->mem_props;
        }
    }
    if (!mCachedPhysicalDeviceMemoryProps) {
        if (physicalDevice == VK_NULL_HANDLE) {
            VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
            if (!dev) {
                mesa_loge("Failed to pass device or physical device.");
                abort();
            }
            physicalDevice = dev->internal_physical_device;
        }

        VkEncoder* enc = (VkEncoder*)context;

        VkPhysicalDeviceMemoryProperties properties;
        enc->vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties, true /* no lock */);
        mCachedPhysicalDeviceMemoryProps.emplace(std::move(properties));
    }
    return *mCachedPhysicalDeviceMemoryProps;
}

ResourceTracker::ResourceTracker() {}

ResourceTracker::~ResourceTracker() {}

// static
ResourceTracker* ResourceTracker::get() {
    // Made on first use and never freed. A function local static is initialized
    // once even when several threads arrive at the same time.
    static ResourceTracker* const sTracker = new ResourceTracker;
    return sTracker;
}

// static
ALWAYS_INLINE_GFXSTREAM VkEncoder* ResourceTracker::getCommandBufferEncoder(
    VkCommandBuffer commandBuffer) {
    if (!(ResourceTracker::streamFeatureBits &
          VULKAN_STREAM_FEATURE_QUEUE_SUBMIT_WITH_COMMANDS_BIT)) {
        auto enc = ResourceTracker::getThreadLocalEncoder();
        ResourceTracker::get()->syncEncodersForCommandBuffer(commandBuffer, enc);
        return enc;
    }

    VK_FROM_HANDLE(gfxstream_vk_command_buffer, cb, commandBuffer);
    if (!cb->privateEncoder) {
        sStaging.ensureAllocFree();
        sStaging.popStaging((CommandBufferStagingStream**)&cb->privateStream, &cb->privateEncoder);
    }
    return cb->privateEncoder;
}

// static
ALWAYS_INLINE_GFXSTREAM VkEncoder* ResourceTracker::getQueueEncoder(VkQueue queue) {
    auto enc = ResourceTracker::getThreadLocalEncoder();
    if (!(ResourceTracker::streamFeatureBits &
          VULKAN_STREAM_FEATURE_QUEUE_SUBMIT_WITH_COMMANDS_BIT)) {
        ResourceTracker::get()->syncEncodersForQueue(queue, enc);
    }
    return enc;
}

// static
ALWAYS_INLINE_GFXSTREAM VkEncoder* ResourceTracker::getThreadLocalEncoder() {
    auto hostConn = ResourceTracker::threadingCallbacks.hostConnectionGetFunc();
    auto vkEncoder = ResourceTracker::threadingCallbacks.vkEncoderGetFunc(hostConn);
    return vkEncoder;
}

// static
void ResourceTracker::setSeqnoPtr(uint32_t* seqnoptr) { sSeqnoPtr = seqnoptr; }

// static
ALWAYS_INLINE_GFXSTREAM uint32_t ResourceTracker::nextSeqno() {
    uint32_t res = __atomic_add_fetch(sSeqnoPtr, 1, __ATOMIC_SEQ_CST);
    return res;
}

// static
ALWAYS_INLINE_GFXSTREAM uint32_t ResourceTracker::getSeqno() {
    uint32_t res = __atomic_load_n(sSeqnoPtr, __ATOMIC_SEQ_CST);
    return res;
}

void ResourceTracker::transformImpl_VkExternalMemoryProperties_tohost(VkExternalMemoryProperties*,
                                                                      uint32_t) {}

void ResourceTracker::transformImpl_VkImageCreateInfo_fromhost(const VkImageCreateInfo*, uint32_t) {
}
void ResourceTracker::transformImpl_VkImageCreateInfo_tohost(const VkImageCreateInfo*, uint32_t) {}

void ResourceTracker::transformImpl_VkMemoryToImageCopy_fromhost(const VkMemoryToImageCopy*,
                                                                 uint32_t) {}
void ResourceTracker::transformImpl_VkMemoryToImageCopy_tohost(const VkMemoryToImageCopy*,
                                                               uint32_t) {}

void ResourceTracker::transformImpl_VkImageToMemoryCopy_fromhost(const VkImageToMemoryCopy* regions,
                                                                 uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const VkImageToMemoryCopy& region = regions[i];
        const auto* hostPtrResolve =
            static_cast<const VkHostPointerResolveGOOGLE*>(__vk_find_struct(
                const_cast<void*>(region.pNext), VK_STRUCTURE_TYPE_HOST_POINTER_RESOLVE_GOOGLE));

        // The copy is not needed when both pointers point to the same buffer.
        if (hostPtrResolve && region.pHostPointer && hostPtrResolve->pHostPointer &&
            region.pHostPointer != hostPtrResolve->pHostPointer &&
            hostPtrResolve->pointerSize > 0) {
            memcpy(region.pHostPointer, hostPtrResolve->pHostPointer, hostPtrResolve->pointerSize);
        }
    }
}
void ResourceTracker::transformImpl_VkImageToMemoryCopy_tohost(const VkImageToMemoryCopy*,
                                                               uint32_t) {}

uint32_t ResourceTracker::getHostImageCopySize(VkEncoder* enc, bool isExt, VkDevice device,
                                               VkImage image, VkHostImageCopyFlags flags,
                                               const VkImageSubresourceLayers& subresource,
                                               uint32_t memoryRowLength, uint32_t memoryImageHeight,
                                               const VkExtent3D& imageExtent) {
    VK_FROM_HANDLE(gfxstream_vk_image, imgInfo, image);
    if (!imgInfo) {
        return 0;
    }
    const VkImageCreateInfo& imageCreateInfo = imgInfo->createInfo;

    const uint32_t layerCount = (subresource.layerCount == VK_REMAINING_ARRAY_LAYERS)
                                    ? (imageCreateInfo.arrayLayers - subresource.baseArrayLayer)
                                    : subresource.layerCount;
    if (layerCount == 0) {
        return 0;
    }

    if (flags & VK_HOST_IMAGE_COPY_MEMCPY_BIT) {
        uint32_t totalSize = 0;
        for (uint32_t i = 0; i < layerCount; ++i) {
            VkImageSubresource2 subres = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2,
                .pNext = nullptr,
                .imageSubresource =
                    {
                        .aspectMask = subresource.aspectMask,
                        .mipLevel = subresource.mipLevel,
                        .arrayLayer = subresource.baseArrayLayer + i,
                    },
            };
            VkSubresourceHostMemcpySize copySize = {
                .sType = VK_STRUCTURE_TYPE_SUBRESOURCE_HOST_MEMCPY_SIZE,
                .pNext = nullptr,
                .size = 0,
            };
            VkSubresourceLayout2 layout = {
                .sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2,
                .pNext = &copySize,
                .subresourceLayout = {},
            };
            if (isExt) {
                enc->vkGetImageSubresourceLayout2EXT(device, image, &subres, &layout,
                                                     true /* do lock */);
            } else {
                enc->vkGetImageSubresourceLayout2(device, image, &subres, &layout,
                                                  true /* do lock */);
            }
            totalSize += static_cast<uint32_t>(copySize.size);
        }
        return totalSize;
    }

    VkFormat format = imageCreateInfo.format;
    if (subresource.aspectMask &
        (VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT | VK_IMAGE_ASPECT_PLANE_2_BIT)) {
        format = vk_format_get_plane_aspect_format(format, subresource.aspectMask);
    } else {
        const bool copyDepth = (subresource.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
        const bool copyStencil = (subresource.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
        if (copyDepth && !copyStencil) {
            format = vk_format_depth_only(format);
        } else if (!copyDepth && copyStencil) {
            format = vk_format_stencil_only(format);
        }
    }

    if (imageExtent.width == 0 || imageExtent.height == 0 || imageExtent.depth == 0) {
        return 0;
    }

    const uint32_t width = memoryRowLength ? memoryRowLength : imageExtent.width;
    const uint32_t height = memoryImageHeight ? memoryImageHeight : imageExtent.height;
    const uint32_t bw = vk_format_get_blockwidth(format);
    const uint32_t bh = vk_format_get_blockheight(format);
    const uint32_t bs = vk_format_get_blocksize(format);

    const size_t rowExtent = static_cast<size_t>(DIV_ROUND_UP(width, bw)) * bs;
    const size_t sliceExtent = static_cast<size_t>(DIV_ROUND_UP(height, bh)) * rowExtent;
    const size_t layerExtent = static_cast<size_t>(imageExtent.depth) * sliceExtent;

    const size_t lastLayerOffset = static_cast<size_t>(layerCount - 1) * layerExtent;
    const size_t lastSliceOffset =
        static_cast<size_t>(imageExtent.depth - 1) * sliceExtent + lastLayerOffset;
    const size_t lastRowOffset =
        static_cast<size_t>(DIV_ROUND_UP(imageExtent.height, bh) - 1) * rowExtent + lastSliceOffset;
    const size_t lastRowSize = static_cast<size_t>(DIV_ROUND_UP(imageExtent.width, bw)) * bs;

    return static_cast<uint32_t>(lastRowSize + lastRowOffset);
}

VkResult ResourceTracker::on_vkCopyMemoryToImage_common(
    bool isExt, void* context, VkResult input_result, VkDevice device,
    const VkCopyMemoryToImageInfo* pCopyMemoryToImageInfo) {
    (void)input_result;
    VkEncoder* enc = (VkEncoder*)context;

    VkCopyMemoryToImageInfo localInfo = *pCopyMemoryToImageInfo;
    std::vector<VkMemoryToImageCopy> localRegions(
        pCopyMemoryToImageInfo->pRegions,
        pCopyMemoryToImageInfo->pRegions + pCopyMemoryToImageInfo->regionCount);
    std::vector<VkHostPointerResolveGOOGLE> hostPointerResolves(
        pCopyMemoryToImageInfo->regionCount);

    for (uint32_t i = 0; i < pCopyMemoryToImageInfo->regionCount; ++i) {
        VkMemoryToImageCopy& region = localRegions[i];
        const uint32_t copySize = getHostImageCopySize(
            enc, isExt, device, pCopyMemoryToImageInfo->dstImage, pCopyMemoryToImageInfo->flags,
            region.imageSubresource, region.memoryRowLength, region.memoryImageHeight,
            region.imageExtent);
        // Set the same host pointer for the resolve to avoid copying memory at
        // transformImpl_VkImageToMemoryCopy_fromhost
        const bool hasData = region.pHostPointer && copySize > 0;
        hostPointerResolves[i] = {
            .sType = VK_STRUCTURE_TYPE_HOST_POINTER_RESOLVE_GOOGLE,
            .pNext = region.pNext,
            .pointerSize = hasData ? copySize : 0u,
            .pHostPointer =
                hasData ? const_cast<uint8_t*>(static_cast<const uint8_t*>(region.pHostPointer))
                        : nullptr,
        };
        region.pNext = &hostPointerResolves[i];
    }
    localInfo.pRegions = localRegions.data();

    if (isExt) {
        return enc->vkCopyMemoryToImageEXT(device, &localInfo, true /* do lock */);
    } else {
        return enc->vkCopyMemoryToImage(device, &localInfo, true /* do lock */);
    }
}

VkResult ResourceTracker::on_vkCopyImageToMemory_common(
    bool isExt, void* context, VkResult input_result, VkDevice device,
    const VkCopyImageToMemoryInfo* pCopyImageToMemoryInfo) {
    (void)input_result;
    VkEncoder* enc = (VkEncoder*)context;

    VkCopyImageToMemoryInfo localInfo = *pCopyImageToMemoryInfo;
    std::vector<VkImageToMemoryCopy> localRegions(
        pCopyImageToMemoryInfo->pRegions,
        pCopyImageToMemoryInfo->pRegions + pCopyImageToMemoryInfo->regionCount);
    std::vector<VkHostPointerResolveGOOGLE> hostPointerResolves(
        pCopyImageToMemoryInfo->regionCount);

    for (uint32_t i = 0; i < pCopyImageToMemoryInfo->regionCount; ++i) {
        VkImageToMemoryCopy& region = localRegions[i];
        const uint32_t copySize = getHostImageCopySize(
            enc, isExt, device, pCopyImageToMemoryInfo->srcImage, pCopyImageToMemoryInfo->flags,
            region.imageSubresource, region.memoryRowLength, region.memoryImageHeight,
            region.imageExtent);
        const bool hasData = region.pHostPointer && copySize > 0;
        hostPointerResolves[i] = {
            .sType = VK_STRUCTURE_TYPE_HOST_POINTER_RESOLVE_GOOGLE,
            .pNext = region.pNext,
            .pointerSize = hasData ? copySize : 0u,
            .pHostPointer = hasData ? static_cast<uint8_t*>(region.pHostPointer) : nullptr,
        };
        region.pNext = &hostPointerResolves[i];
    }
    localInfo.pRegions = localRegions.data();

    if (isExt) {
        return enc->vkCopyImageToMemoryEXT(device, &localInfo, true /* do lock */);
    } else {
        return enc->vkCopyImageToMemory(device, &localInfo, true /* do lock */);
    }
}

VkResult ResourceTracker::on_vkCopyMemoryToImage(
    void* context, VkResult input_result, VkDevice device,
    const VkCopyMemoryToImageInfo* pCopyMemoryToImageInfo) {
    return on_vkCopyMemoryToImage_common(false, context, input_result, device,
                                         pCopyMemoryToImageInfo);
}

VkResult ResourceTracker::on_vkCopyMemoryToImageEXT(
    void* context, VkResult input_result, VkDevice device,
    const VkCopyMemoryToImageInfoEXT* pCopyMemoryToImageInfo) {
    return on_vkCopyMemoryToImage_common(true, context, input_result, device,
                                         pCopyMemoryToImageInfo);
}

VkResult ResourceTracker::on_vkCopyImageToMemory(
    void* context, VkResult input_result, VkDevice device,
    const VkCopyImageToMemoryInfo* pCopyImageToMemoryInfo) {
    return on_vkCopyImageToMemory_common(false, context, input_result, device,
                                         pCopyImageToMemoryInfo);
}

VkResult ResourceTracker::on_vkCopyImageToMemoryEXT(
    void* context, VkResult input_result, VkDevice device,
    const VkCopyImageToMemoryInfoEXT* pCopyImageToMemoryInfo) {
    return on_vkCopyImageToMemory_common(true, context, input_result, device,
                                         pCopyImageToMemoryInfo);
}

#define DEFINE_TRANSFORMED_TYPE_IMPL(type)                                  \
    void ResourceTracker::transformImpl_##type##_tohost(type*, uint32_t) {} \
    void ResourceTracker::transformImpl_##type##_fromhost(type*, uint32_t) {}

LIST_TRIVIAL_TRANSFORMED_TYPES(DEFINE_TRANSFORMED_TYPE_IMPL)

}  // namespace vk
}  // namespace gfxstream
