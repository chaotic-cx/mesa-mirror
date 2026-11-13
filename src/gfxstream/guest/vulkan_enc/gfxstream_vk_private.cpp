/*
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "gfxstream_vk_private.h"

#include <algorithm>
#include <new>
#include <type_traits>
#include <utility>

#include "DescriptorSetVirtualization.h"
#include "vk_debug_utils.h"
#include "vk_sync_dummy.h"
#include "vulkan/vulkan_core.h"

/* Under the assumption that Mesa VK runtime queue submission is used, WSI flow
 * sets this temporary state to a dummy sync type (when no explicit dma-buf
 * synchronization is available). For gfxstream, ignore this sync object when
 * this is the case. Synchronization will be done on the host.
 */

bool isNoopFence(gfxstream_vk_fence* fence) {
    return (fence && fence->base.temporary && vk_sync_type_is_dummy(fence->base.temporary->type));
}

static bool isNoopSemaphore(gfxstream_vk_semaphore* semaphore) {
    return (semaphore && semaphore->base.temporary &&
            vk_sync_type_is_dummy(semaphore->base.temporary->type));
}

bool HasNoopSemaphores(const VkSemaphore* pSemaphores, uint32_t semaphoreCount) {
    if (!pSemaphores) return false;
    for (uint32_t j = 0; j < semaphoreCount; ++j) {
        VK_FROM_HANDLE(gfxstream_vk_semaphore, gfxstream_semaphore, pSemaphores[j]);
        if (isNoopSemaphore(gfxstream_semaphore)) {
            return true;
        }
    }
    return false;
}

bool HasNoopSemaphoreSubmitInfos(const VkSemaphoreSubmitInfo* pSemaphoreSubmitInfos,
                                 uint32_t semaphoreSubmitInfoCount) {
    if (!pSemaphoreSubmitInfos) return false;
    for (uint32_t j = 0; j < semaphoreSubmitInfoCount; ++j) {
        VK_FROM_HANDLE(gfxstream_vk_semaphore, gfxstream_semaphore,
                       pSemaphoreSubmitInfos[j].semaphore);
        if (isNoopSemaphore(gfxstream_semaphore)) {
            return true;
        }
    }
    return false;
}

std::vector<VkSemaphore> FilterNoopSemaphores(const VkSemaphore* pSemaphores,
                                              uint32_t semaphoreCount) {
    std::vector<VkSemaphore> outSemaphores;
    outSemaphores.reserve(semaphoreCount);
    for (uint32_t j = 0; j < semaphoreCount; ++j) {
        VK_FROM_HANDLE(gfxstream_vk_semaphore, gfxstream_semaphore, pSemaphores[j]);
        if (isNoopSemaphore(gfxstream_semaphore)) {
            if (gfxstream_semaphore->base.base.device) {
                vk_semaphore_reset_temporary(gfxstream_semaphore->base.base.device,
                                             &gfxstream_semaphore->base);
            }
        } else {
            outSemaphores.push_back(pSemaphores[j]);
        }
    }
    return outSemaphores;
}

std::vector<VkSemaphoreSubmitInfo> FilterNoopSemaphoreSubmitInfos(
    const VkSemaphoreSubmitInfo* pSemaphoreSubmitInfos, uint32_t semaphoreSubmitInfoCount) {
    std::vector<VkSemaphoreSubmitInfo> outSemaphoreSubmitInfo;
    outSemaphoreSubmitInfo.reserve(semaphoreSubmitInfoCount);
    for (uint32_t j = 0; j < semaphoreSubmitInfoCount; ++j) {
        VkSemaphoreSubmitInfo outInfo = pSemaphoreSubmitInfos[j];
        VK_FROM_HANDLE(gfxstream_vk_semaphore, gfxstream_semaphore, outInfo.semaphore);
        if (isNoopSemaphore(gfxstream_semaphore)) {
            if (gfxstream_semaphore->base.base.device) {
                vk_semaphore_reset_temporary(gfxstream_semaphore->base.base.device,
                                             &gfxstream_semaphore->base);
            }
        } else {
            outSemaphoreSubmitInfo.push_back(outInfo);
        }
    }
    return outSemaphoreSubmitInfo;
}

float linearChannelToSRGB(float cl) {
    if (cl <= 0.0f)
        return 0.0f;
    else if (cl < 0.0031308f)
        return 12.92f * cl;
    else if (cl < 1.0f)
        return 1.055f * pow(cl, 0.41666f) - 0.055f;
    else
        return 1.0f;
}

float srgbFormatNeedsConversionForClearColor(const VkFormat& format) {
    return format == VK_FORMAT_R8G8B8A8_SRGB;
}

namespace {

template <typename T, typename = void>
struct has_alive_token : std::false_type {};

template <typename T>
struct has_alive_token<T, std::enable_if_t<std::is_same_v<decltype(std::declval<T&>().alive),
                                                          gfxstream::vk::AliveTokenOwner>>>
    : std::true_type {};

inline const VkAllocationCallbacks* resolve_instance_alloc(struct vk_instance* instance) {
    return (instance && instance->alloc.pfnAllocation && instance->alloc.pfnFree)
               ? &instance->alloc
               : vk_default_allocator();
}

template <typename T>
T* gfxstream_vk_object_alloc(struct vk_device* device, const VkAllocationCallbacks* pAllocator,
                             VkObjectType obj_type,
                             VkSystemAllocationScope scope = VK_SYSTEM_ALLOCATION_SCOPE_OBJECT) {
    static_assert(!std::is_polymorphic_v<T>, "Vulkan handle wrappers must not have virtual tables");
    const VkAllocationCallbacks* parent_alloc = gfxstream::vk::resolveDeviceAllocator(device);
    void* mem = vk_zalloc2(parent_alloc, pAllocator, sizeof(T),
                           std::max<size_t>(GFXSTREAM_DEFAULT_ALIGN, alignof(T)), scope);
    if (!mem) return nullptr;

    T* obj = new (mem) T();
    vk_object_base_init(device, reinterpret_cast<struct vk_object_base*>(obj), obj_type);

    if constexpr (has_alive_token<T>::value) {
        if (!obj->alive.init(parent_alloc)) {
            vk_object_base_finish(reinterpret_cast<struct vk_object_base*>(obj));
            obj->~T();
            vk_free2(parent_alloc, pAllocator, obj);
            return nullptr;
        }
    }

    return obj;
}

template <typename T, typename FinishFn>
void gfxstream_vk_object_free(struct vk_device* device, const VkAllocationCallbacks* pAllocator,
                              T* obj, FinishFn&& finish) {
    if (!obj) return;

    finish(obj);
    obj->~T();

    const VkAllocationCallbacks* parent_alloc = gfxstream::vk::resolveDeviceAllocator(device);
    vk_free2(parent_alloc, pAllocator, obj);
}

template <typename T>
void gfxstream_vk_object_free(struct vk_device* device, const VkAllocationCallbacks* pAllocator,
                              T* obj) {
    gfxstream_vk_object_free(device, pAllocator, obj, [](T* target) {
        vk_object_base_finish(reinterpret_cast<struct vk_object_base*>(target));
    });
}

template <typename T>
T* gfxstream_vk_instance_object_alloc(
    struct vk_instance* instance, const VkAllocationCallbacks* pAllocator, VkObjectType obj_type,
    VkSystemAllocationScope scope = VK_SYSTEM_ALLOCATION_SCOPE_OBJECT) {
    static_assert(!std::is_polymorphic_v<T>, "Vulkan handle wrappers must not have virtual tables");
    const VkAllocationCallbacks* parent_alloc = resolve_instance_alloc(instance);
    void* mem = vk_zalloc2(parent_alloc, pAllocator, sizeof(T),
                           std::max<size_t>(GFXSTREAM_DEFAULT_ALIGN, alignof(T)), scope);
    if (!mem) return nullptr;

    T* obj = new (mem) T();
    vk_object_base_instance_init(instance, reinterpret_cast<struct vk_object_base*>(obj), obj_type);

    return obj;
}

template <typename T, typename FinishFn>
void gfxstream_vk_instance_object_free(struct vk_instance* instance,
                                       const VkAllocationCallbacks* pAllocator, T* obj,
                                       FinishFn&& finish) {
    if (!obj) return;

    finish(obj);
    obj->~T();

    const VkAllocationCallbacks* parent_alloc = resolve_instance_alloc(instance);
    vk_free2(parent_alloc, pAllocator, obj);
}

template <typename T>
void gfxstream_vk_instance_object_free(struct vk_instance* instance,
                                       const VkAllocationCallbacks* pAllocator, T* obj) {
    gfxstream_vk_instance_object_free(instance, pAllocator, obj, [](T* target) {
        vk_object_base_finish(reinterpret_cast<struct vk_object_base*>(target));
    });
}

}  // namespace

extern "C" {

VkInstance create_gfxstream_vk_instance(const VkAllocationCallbacks* pAllocator,
                                        uint64_t underlying) {
    struct gfxstream_vk_instance* res =
        gfxstream_vk_instance_object_alloc<struct gfxstream_vk_instance>(
            nullptr, pAllocator, VK_OBJECT_TYPE_INSTANCE, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
    if (!res) return VK_NULL_HANDLE;
    res->base.alloc = pAllocator ? *pAllocator : *vk_default_allocator();
    res->common.underlying = underlying;
    return gfxstream_vk_instance_to_handle(res);
}

void delete_gfxstream_vk_instance(VkInstance toDelete, const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_instance, obj, toDelete);
    VkAllocationCallbacks alloc =
        obj->base.alloc.pfnFree ? obj->base.alloc : *vk_default_allocator();
    if (obj->runtime_initialized) {
        vk_instance_finish(&obj->base);
    } else {
        vk_free(&alloc, (char*)obj->base.app_info.app_name);
        vk_free(&alloc, (char*)obj->base.app_info.engine_name);
        if (obj->base.debug_utils.instance_callbacks.next) {
            list_for_each_entry_safe(struct vk_debug_utils_messenger, messenger,
                                     &obj->base.debug_utils.instance_callbacks, link) {
                list_del(&messenger->link);
                vk_object_base_finish(&messenger->base);
                vk_free2(&alloc, &messenger->alloc, messenger);
            }
        }
        vk_object_base_finish(&obj->base.base);
    }
    obj->~gfxstream_vk_instance();
    vk_free2(&alloc, pAllocator, obj);
}

uint64_t gfxstream_vk_instance_to_host_u64(const VkInstance obj) {
    if (!obj) return 0;
    return gfxstream_vk_instance_from_handle(obj)->common.underlying;
}

VkPhysicalDevice create_gfxstream_vk_physical_device(VkInstance instance,
                                                     const VkAllocationCallbacks* pAllocator,
                                                     uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_instance, inst, instance);
    struct gfxstream_vk_physical_device* res =
        gfxstream_vk_instance_object_alloc<struct gfxstream_vk_physical_device>(
            inst ? &inst->base : nullptr, pAllocator, VK_OBJECT_TYPE_PHYSICAL_DEVICE,
            VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
    if (!res) return VK_NULL_HANDLE;
    res->instance = inst;
    res->common.underlying = underlying;
    return gfxstream_vk_physical_device_to_handle(res);
}

void delete_gfxstream_vk_physical_device(VkInstance instance, VkPhysicalDevice toDelete,
                                         const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_physical_device, obj, toDelete);
    struct gfxstream_vk_instance* inst =
        instance ? gfxstream_vk_instance_from_handle(instance) : obj->instance;
    gfxstream_vk_instance_object_free(inst ? &inst->base : nullptr, pAllocator, obj,
                                      [](struct gfxstream_vk_physical_device* pd) {
                                          if (pd->base.instance) {
                                              vk_physical_device_finish(&pd->base);
                                          } else {
                                              vk_object_base_finish(&pd->base.base);
                                          }
                                      });
}

uint64_t gfxstream_vk_physical_device_to_host_u64(const VkPhysicalDevice obj) {
    if (!obj) return 0;
    return gfxstream_vk_physical_device_from_handle(obj)->common.underlying;
}

VkDevice create_gfxstream_vk_device(VkPhysicalDevice physicalDevice,
                                    const VkAllocationCallbacks* pAllocator, uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_physical_device, physDev, physicalDevice);
    struct vk_instance* instBase =
        (physDev && physDev->instance) ? &physDev->instance->base : nullptr;
    struct gfxstream_vk_device* res =
        gfxstream_vk_instance_object_alloc<struct gfxstream_vk_device>(
            instBase, pAllocator, VK_OBJECT_TYPE_DEVICE, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
    if (!res) return VK_NULL_HANDLE;
    res->base.alloc = pAllocator ? *pAllocator : *resolve_instance_alloc(instBase);
    res->physical_device = physDev;
    res->internal_physical_device = physicalDevice;
    simple_mtx_init(&res->memory_mtx, mtx_plain);
    list_inithead(&res->device_memories);
    res->common.underlying = underlying;
    return gfxstream_vk_device_to_handle(res);
}

void delete_gfxstream_vk_device(VkDevice toDelete, const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, obj, toDelete);
    struct vk_instance* instBase = (obj->physical_device && obj->physical_device->instance)
                                       ? &obj->physical_device->instance->base
                                       : nullptr;
    VkAllocationCallbacks alloc =
        obj->base.alloc.pfnFree ? obj->base.alloc : *resolve_instance_alloc(instBase);
    simple_mtx_destroy(&obj->memory_mtx);
    if (obj->runtime_initialized) {
        vk_device_finish(&obj->base);
    } else {
        vk_object_base_finish(&obj->base.base);
    }
    obj->~gfxstream_vk_device();
    vk_free2(&alloc, pAllocator, obj);
}

uint64_t gfxstream_vk_device_to_host_u64(const VkDevice obj) {
    if (!obj) return 0;
    return gfxstream_vk_device_from_handle(obj)->common.underlying;
}

#define GFXSTREAM_DEFINE_VK_OBJECT_INSTANCE(gfxstream_type, vk_type, vk_object_type)              \
    vk_type create_##gfxstream_type(VkInstance instance, const VkAllocationCallbacks* pAllocator, \
                                    uint64_t underlying) {                                        \
        VK_FROM_HANDLE(gfxstream_vk_instance, inst, instance);                                    \
        struct gfxstream_type* res = gfxstream_vk_instance_object_alloc<struct gfxstream_type>(   \
            inst ? &inst->base : nullptr, pAllocator, vk_object_type);                            \
        if (!res) return VK_NULL_HANDLE;                                                          \
        res->common.underlying = underlying;                                                      \
        return gfxstream_type##_to_handle(res);                                                   \
    }                                                                                             \
    void delete_##gfxstream_type(VkInstance instance, vk_type toDelete,                           \
                                 const VkAllocationCallbacks* pAllocator) {                       \
        if (!toDelete) return;                                                                    \
        VK_FROM_HANDLE(gfxstream_vk_instance, inst, instance);                                    \
        auto* obj = gfxstream_type##_from_handle(toDelete);                                       \
        gfxstream_vk_instance_object_free(inst ? &inst->base : nullptr, pAllocator, obj);         \
    }                                                                                             \
    uint64_t gfxstream_type##_to_host_u64(const vk_type obj) {                                    \
        if (!obj) return 0;                                                                       \
        return gfxstream_type##_from_handle(obj)->common.underlying;                              \
    }

#define GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_type, vk_type, vk_object_type)                   \
    vk_type create_##gfxstream_type(VkDevice device, const VkAllocationCallbacks* pAllocator, \
                                    uint64_t underlying) {                                    \
        VK_FROM_HANDLE(gfxstream_vk_device, dev, device);                                     \
        struct gfxstream_type* res = gfxstream_vk_object_alloc<struct gfxstream_type>(        \
            dev ? &dev->base : nullptr, pAllocator, vk_object_type);                          \
        if (!res) return VK_NULL_HANDLE;                                                      \
        res->common.underlying = underlying;                                                  \
        return gfxstream_type##_to_handle(res);                                               \
    }                                                                                         \
    void delete_##gfxstream_type(VkDevice device, vk_type toDelete,                           \
                                 const VkAllocationCallbacks* pAllocator) {                   \
        if (!toDelete) return;                                                                \
        VK_FROM_HANDLE(gfxstream_vk_device, dev, device);                                     \
        auto* obj = gfxstream_type##_from_handle(toDelete);                                   \
        gfxstream_vk_object_free(dev ? &dev->base : nullptr, pAllocator, obj);                \
    }                                                                                         \
    uint64_t gfxstream_type##_to_host_u64(const vk_type obj) {                                \
        if (!obj) return 0;                                                                   \
        return gfxstream_type##_from_handle(obj)->common.underlying;                          \
    }

VkQueue create_gfxstream_vk_queue(VkDevice device, const VkAllocationCallbacks* pAllocator,
                                  uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_queue* res = gfxstream_vk_object_alloc<struct gfxstream_vk_queue>(
        dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_QUEUE);
    if (!res) return VK_NULL_HANDLE;
    res->device = dev;
    res->common.underlying = underlying;
    return gfxstream_vk_queue_to_handle(res);
}

void delete_gfxstream_vk_queue(VkDevice device, VkQueue toDelete,
                               const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_queue, obj, toDelete);
    struct gfxstream_vk_device* dev =
        device ? gfxstream_vk_device_from_handle(device) : obj->device;
    gfxstream_vk_object_free(dev ? &dev->base : nullptr, pAllocator, obj,
                             [](struct gfxstream_vk_queue* queue) {
                                 if (queue->base.base.device && queue->base.link.next) {
                                     vk_queue_finish(&queue->base);
                                 } else {
                                     vk_object_base_finish(&queue->base.base);
                                 }
                             });
}

uint64_t gfxstream_vk_queue_to_host_u64(const VkQueue obj) {
    if (!obj) return 0;
    return gfxstream_vk_queue_from_handle(obj)->common.underlying;
}

VkCommandPool create_gfxstream_vk_command_pool(VkDevice device,
                                               const VkAllocationCallbacks* pAllocator,
                                               uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct vk_device* vkDev = dev ? &dev->base : nullptr;
    struct gfxstream_vk_command_pool* res =
        gfxstream_vk_object_alloc<struct gfxstream_vk_command_pool>(vkDev, pAllocator,
                                                                    VK_OBJECT_TYPE_COMMAND_POOL);
    if (!res) return VK_NULL_HANDLE;
    res->base.alloc = pAllocator ? *pAllocator : *gfxstream::vk::resolveDeviceAllocator(vkDev);
    list_inithead(&res->base.command_buffers);
    for (uint32_t i = 0; i < ARRAY_SIZE(res->base.free_command_buffers); ++i) {
        list_inithead(&res->base.free_command_buffers[i]);
    }
    res->common.underlying = underlying;
    return gfxstream_vk_command_pool_to_handle(res);
}

void delete_gfxstream_vk_command_pool(VkDevice device, VkCommandPool toDelete,
                                      const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_command_pool, obj, toDelete);
    gfxstream_vk_object_free(
        dev ? &dev->base : nullptr, pAllocator, obj,
        [](struct gfxstream_vk_command_pool* pool) { vk_command_pool_finish(&pool->base); });
}

uint64_t gfxstream_vk_command_pool_to_host_u64(const VkCommandPool obj) {
    if (!obj) return 0;
    return gfxstream_vk_command_pool_from_handle(obj)->common.underlying;
}

VkCommandBuffer create_gfxstream_vk_command_buffer(VkDevice device,
                                                   const VkAllocationCallbacks* pAllocator,
                                                   uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_command_buffer* res =
        gfxstream_vk_object_alloc<struct gfxstream_vk_command_buffer>(
            dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_COMMAND_BUFFER);
    if (!res) return VK_NULL_HANDLE;
    list_inithead(&res->base.pool_link);
    util_dynarray_init(&res->subObjects, NULL);
    util_dynarray_init(&res->superObjects, NULL);
    res->device = device;
    res->common.underlying = underlying;
    return gfxstream_vk_command_buffer_to_handle(res);
}

void delete_gfxstream_vk_command_buffer(VkDevice device, VkCommandBuffer toDelete,
                                        const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_command_buffer, obj, toDelete);
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device ? device : obj->device);
    const VkAllocationCallbacks* cbAlloc =
        pAllocator ? pAllocator : (obj->base.pool ? &obj->base.pool->alloc : nullptr);
    util_dynarray_fini(&obj->subObjects);
    util_dynarray_fini(&obj->superObjects);
    gfxstream_vk_object_free(dev ? &dev->base : nullptr, cbAlloc, obj,
                             [](struct gfxstream_vk_command_buffer* cb) {
                                 if (cb->base.pool) {
                                     vk_command_buffer_finish(&cb->base);
                                 } else {
                                     vk_object_base_finish(&cb->base.base);
                                 }
                             });
}

uint64_t gfxstream_vk_command_buffer_to_host_u64(const VkCommandBuffer obj) {
    if (!obj) return 0;
    return gfxstream_vk_command_buffer_from_handle(obj)->common.underlying;
}

VkBuffer create_gfxstream_vk_buffer(VkDevice device, const VkAllocationCallbacks* pAllocator,
                                    uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_buffer* res = gfxstream_vk_object_alloc<struct gfxstream_vk_buffer>(
        dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_BUFFER);
    if (!res) return VK_NULL_HANDLE;
    res->common.underlying = underlying;
    return gfxstream_vk_buffer_to_handle(res);
}

void delete_gfxstream_vk_buffer(VkDevice device, VkBuffer toDelete,
                                const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_buffer, obj, toDelete);
    gfxstream_vk_object_free(dev ? &dev->base : nullptr, pAllocator, obj,
                             [](struct gfxstream_vk_buffer* buf) { vk_buffer_finish(&buf->base); });
}

uint64_t gfxstream_vk_buffer_to_host_u64(const VkBuffer obj) {
    if (!obj) return 0;
    return gfxstream_vk_buffer_from_handle(obj)->common.underlying;
}

VkFence create_gfxstream_vk_fence(VkDevice device, const VkAllocationCallbacks* pAllocator,
                                  uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_fence* res = gfxstream_vk_object_alloc<struct gfxstream_vk_fence>(
        dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_FENCE);
    if (!res) return VK_NULL_HANDLE;
    res->device = device;
    res->common.underlying = underlying;
    return gfxstream_vk_fence_to_handle(res);
}

void delete_gfxstream_vk_fence(VkDevice device, VkFence toDelete,
                               const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_fence, obj, toDelete);
    if (dev && obj->base.temporary) {
        vk_fence_reset_temporary(&dev->base, &obj->base);
    }
    gfxstream_vk_object_free(dev ? &dev->base : nullptr, pAllocator, obj);
}

uint64_t gfxstream_vk_fence_to_host_u64(const VkFence obj) {
    if (!obj) return 0;
    return gfxstream_vk_fence_from_handle(obj)->common.underlying;
}

VkSemaphore create_gfxstream_vk_semaphore(VkDevice device, const VkAllocationCallbacks* pAllocator,
                                          uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_semaphore* res = gfxstream_vk_object_alloc<struct gfxstream_vk_semaphore>(
        dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_SEMAPHORE);
    if (!res) return VK_NULL_HANDLE;
    res->device = device;
    res->eventHandle = ZX_HANDLE_INVALID;
    res->eventKoid = ZX_KOID_INVALID;
    res->common.underlying = underlying;
    return gfxstream_vk_semaphore_to_handle(res);
}

void delete_gfxstream_vk_semaphore(VkDevice device, VkSemaphore toDelete,
                                   const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_semaphore, obj, toDelete);
    if (dev && obj->base.temporary) {
        vk_semaphore_reset_temporary(&dev->base, &obj->base);
    }
    gfxstream_vk_object_free(dev ? &dev->base : nullptr, pAllocator, obj);
}

uint64_t gfxstream_vk_semaphore_to_host_u64(const VkSemaphore obj) {
    if (!obj) return 0;
    return gfxstream_vk_semaphore_from_handle(obj)->common.underlying;
}

VkDeviceMemory create_gfxstream_vk_device_memory(VkDevice device,
                                                 const VkAllocationCallbacks* pAllocator,
                                                 uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_device_memory* res =
        gfxstream_vk_object_alloc<struct gfxstream_vk_device_memory>(
            dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_DEVICE_MEMORY);
    if (!res) return VK_NULL_HANDLE;
    list_inithead(&res->link);
    res->device = device;
    res->vmoHandle = ZX_HANDLE_INVALID;
    res->importedFd = -1;
    res->common.underlying = underlying;
    if (dev) {
        simple_mtx_lock(&dev->memory_mtx);
        list_addtail(&res->link, &dev->device_memories);
        simple_mtx_unlock(&dev->memory_mtx);
    }
    return gfxstream_vk_device_memory_to_handle(res);
}

void delete_gfxstream_vk_device_memory(VkDevice device, VkDeviceMemory toDelete,
                                       const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device_memory, obj, toDelete);
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device ? device : obj->device);
    if (dev) {
        simple_mtx_lock(&dev->memory_mtx);
        if (!list_is_empty(&obj->link)) {
            list_delinit(&obj->link);
        }
        simple_mtx_unlock(&dev->memory_mtx);
    }
    gfxstream_vk_object_free(dev ? &dev->base : nullptr, pAllocator, obj);
}

uint64_t gfxstream_vk_device_memory_to_host_u64(const VkDeviceMemory obj) {
    if (!obj) return 0;
    return gfxstream_vk_device_memory_from_handle(obj)->common.underlying;
}

VkImage create_gfxstream_vk_image(VkDevice device, const VkAllocationCallbacks* pAllocator,
                                  uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_image* res = gfxstream_vk_object_alloc<struct gfxstream_vk_image>(
        dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_IMAGE);
    if (!res) return VK_NULL_HANDLE;
    res->device = device;
    res->common.underlying = underlying;
    return gfxstream_vk_image_to_handle(res);
}

void delete_gfxstream_vk_image(VkDevice device, VkImage toDelete,
                               const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_image, obj, toDelete);
    gfxstream_vk_object_free(dev ? &dev->base : nullptr, pAllocator, obj,
                             [](struct gfxstream_vk_image* img) { vk_image_finish(&img->base); });
}

uint64_t gfxstream_vk_image_to_host_u64(const VkImage obj) {
    if (!obj) return 0;
    return gfxstream_vk_image_from_handle(obj)->common.underlying;
}

VkDescriptorUpdateTemplate create_gfxstream_vk_descriptor_update_template(
    VkDevice device, const VkAllocationCallbacks* pAllocator, uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_descriptor_update_template* res =
        gfxstream_vk_object_alloc<struct gfxstream_vk_descriptor_update_template>(
            dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE);
    if (!res) return VK_NULL_HANDLE;
    res->common.underlying = underlying;
    return gfxstream_vk_descriptor_update_template_to_handle(res);
}

void delete_gfxstream_vk_descriptor_update_template(VkDevice device,
                                                    VkDescriptorUpdateTemplate toDelete,
                                                    const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_descriptor_update_template, obj, toDelete);
    struct vk_device* vkDev = dev ? &dev->base : obj->base.device;
    vk_free2(gfxstream::vk::resolveDeviceAllocator(vkDev), pAllocator, obj->templateStorage);
    gfxstream_vk_object_free(vkDev, pAllocator, obj);
}

uint64_t gfxstream_vk_descriptor_update_template_to_host_u64(const VkDescriptorUpdateTemplate obj) {
    if (!obj) return 0;
    return gfxstream_vk_descriptor_update_template_from_handle(obj)->common.underlying;
}

VkDescriptorPool create_gfxstream_vk_descriptor_pool(VkDevice device,
                                                     const VkAllocationCallbacks* pAllocator,
                                                     uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_descriptor_pool* res =
        gfxstream_vk_object_alloc<struct gfxstream_vk_descriptor_pool>(
            dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
    if (!res) return VK_NULL_HANDLE;
    res->common.underlying = underlying;
    return gfxstream_vk_descriptor_pool_to_handle(res);
}

void delete_gfxstream_vk_descriptor_pool(VkDevice device, VkDescriptorPool toDelete,
                                         const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_descriptor_pool, obj, toDelete);
    struct vk_device* vkDev = dev ? &dev->base : obj->base.device;
    gfxstream::vk::freeDeviceSubobject(vkDev, pAllocator, obj->allocInfo);
    gfxstream_vk_object_free(vkDev, pAllocator, obj);
}

uint64_t gfxstream_vk_descriptor_pool_to_host_u64(const VkDescriptorPool obj) {
    if (!obj) return 0;
    return gfxstream_vk_descriptor_pool_from_handle(obj)->common.underlying;
}

VkDescriptorSet create_gfxstream_vk_descriptor_set(VkDevice device,
                                                   const VkAllocationCallbacks* pAllocator,
                                                   uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_descriptor_set* res =
        gfxstream_vk_object_alloc<struct gfxstream_vk_descriptor_set>(
            dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_DESCRIPTOR_SET);
    if (!res) return VK_NULL_HANDLE;
    res->common.underlying = underlying;
    return gfxstream_vk_descriptor_set_to_handle(res);
}

void delete_gfxstream_vk_descriptor_set(VkDevice device, VkDescriptorSet toDelete,
                                        const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_descriptor_set, obj, toDelete);
    struct vk_device* vkDev = dev ? &dev->base : obj->base.device;
    gfxstream::vk::freeDeviceSubobject(vkDev, pAllocator, obj->reified);
    gfxstream_vk_object_free(vkDev, pAllocator, obj);
}

uint64_t gfxstream_vk_descriptor_set_to_host_u64(const VkDescriptorSet obj) {
    if (!obj) return 0;
    return gfxstream_vk_descriptor_set_from_handle(obj)->common.underlying;
}

VkDescriptorSetLayout create_gfxstream_vk_descriptor_set_layout(
    VkDevice device, const VkAllocationCallbacks* pAllocator, uint64_t underlying) {
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    struct gfxstream_vk_descriptor_set_layout* res =
        gfxstream_vk_object_alloc<struct gfxstream_vk_descriptor_set_layout>(
            dev ? &dev->base : nullptr, pAllocator, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT);
    if (!res) return VK_NULL_HANDLE;
    res->common.underlying = underlying;
    return gfxstream_vk_descriptor_set_layout_to_handle(res);
}

void delete_gfxstream_vk_descriptor_set_layout(VkDevice device, VkDescriptorSetLayout toDelete,
                                               const VkAllocationCallbacks* pAllocator) {
    if (!toDelete) return;
    VK_FROM_HANDLE(gfxstream_vk_device, dev, device);
    VK_FROM_HANDLE(gfxstream_vk_descriptor_set_layout, obj, toDelete);
    struct vk_device* vkDev = dev ? &dev->base : obj->base.device;
    gfxstream::vk::freeDeviceSubobject(vkDev, pAllocator, obj->layoutInfo);
    gfxstream_vk_object_free(vkDev, pAllocator, obj);
}

uint64_t gfxstream_vk_descriptor_set_layout_to_host_u64(const VkDescriptorSetLayout obj) {
    if (!obj) return 0;
    return gfxstream_vk_descriptor_set_layout_from_handle(obj)->common.underlying;
}

GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_sampler_ycbcr_conversion, VkSamplerYcbcrConversion,
                           VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION)
#ifdef VK_USE_PLATFORM_FUCHSIA
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_buffer_collection_fuchsia, VkBufferCollectionFUCHSIA,
                           VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA)
#endif
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_sampler, VkSampler, VK_OBJECT_TYPE_SAMPLER)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_private_data_slot, VkPrivateDataSlot,
                           VK_OBJECT_TYPE_PRIVATE_DATA_SLOT)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_buffer_view, VkBufferView, VK_OBJECT_TYPE_BUFFER_VIEW)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_image_view, VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_shader_module, VkShaderModule, VK_OBJECT_TYPE_SHADER_MODULE)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_pipeline, VkPipeline, VK_OBJECT_TYPE_PIPELINE)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_pipeline_cache, VkPipelineCache,
                           VK_OBJECT_TYPE_PIPELINE_CACHE)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_pipeline_layout, VkPipelineLayout,
                           VK_OBJECT_TYPE_PIPELINE_LAYOUT)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_render_pass, VkRenderPass, VK_OBJECT_TYPE_RENDER_PASS)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_framebuffer, VkFramebuffer, VK_OBJECT_TYPE_FRAMEBUFFER)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_event, VkEvent, VK_OBJECT_TYPE_EVENT)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_query_pool, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL)
GFXSTREAM_DEFINE_VK_OBJECT_INSTANCE(gfxstream_vk_surface_khr, VkSurfaceKHR,
                                    VK_OBJECT_TYPE_SURFACE_KHR)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_swapchain_khr, VkSwapchainKHR, VK_OBJECT_TYPE_SWAPCHAIN_KHR)
GFXSTREAM_DEFINE_VK_OBJECT_INSTANCE(gfxstream_vk_display_khr, VkDisplayKHR,
                                    VK_OBJECT_TYPE_DISPLAY_KHR)
GFXSTREAM_DEFINE_VK_OBJECT_INSTANCE(gfxstream_vk_display_mode_khr, VkDisplayModeKHR,
                                    VK_OBJECT_TYPE_DISPLAY_MODE_KHR)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_validation_cache_ext, VkValidationCacheEXT,
                           VK_OBJECT_TYPE_VALIDATION_CACHE_EXT)
GFXSTREAM_DEFINE_VK_OBJECT_INSTANCE(gfxstream_vk_debug_report_callback_ext,
                                    VkDebugReportCallbackEXT,
                                    VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT)
GFXSTREAM_DEFINE_VK_OBJECT_INSTANCE(gfxstream_vk_debug_utils_messenger_ext,
                                    VkDebugUtilsMessengerEXT,
                                    VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_micromap_ext, VkMicromapEXT, VK_OBJECT_TYPE_MICROMAP_EXT)
#ifdef VK_NVX_binary_import
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_cu_module_nvx, VkCuModuleNVX, VK_OBJECT_TYPE_CU_MODULE_NVX)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_cu_function_nvx, VkCuFunctionNVX,
                           VK_OBJECT_TYPE_CU_FUNCTION_NVX)
#endif
#ifdef VK_NVX_device_generated_commands
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_object_table_nvx, VkObjectTableNVX,
                           VK_OBJECT_TYPE_OBJECT_TABLE_NVX)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_indirect_commands_layout_nvx, VkIndirectCommandsLayoutNVX,
                           VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NVX)
#endif
#ifdef VK_NV_device_generated_commands
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_indirect_commands_layout_nv, VkIndirectCommandsLayoutNV,
                           VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV)
#endif
#ifdef VK_NV_ray_tracing
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_acceleration_structure_nv, VkAccelerationStructureNV,
                           VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV)
#endif
#ifdef VK_KHR_acceleration_structure
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_acceleration_structure_khr, VkAccelerationStructureKHR,
                           VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR)
#endif

}  // extern "C"