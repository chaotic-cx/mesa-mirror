/*
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: MIT
 *
 * based in part on anv and radv which are:
 * Copyright © 2015 Intel Corporation
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 */

#ifndef VN_COMMON_H
#define VN_COMMON_H

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <vulkan/vulkan.h>

#include "c11/threads.h"
#include "drm-uapi/drm_fourcc.h"
#include "util/bitscan.h"
#include "util/bitset.h"
#include "util/compiler.h"
#include "util/detect_os.h"
#include "util/libsync.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/perf/cpu_trace.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "util/xmlconfig.h"
#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_debug_report.h"
#include "vk_device.h"
#include "vk_device_memory.h"
#include "vk_image.h"
#include "vk_instance.h"
#include "vk_object.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_util.h"

#include "vn_entrypoints.h"

#define VN_DEFAULT_ALIGN             8
#define VN_WATCHDOG_REPORT_PERIOD_US 3000000

#define VN_DEBUG(category) (unlikely(vn_env.debug & VN_DEBUG_##category))
#define VN_PERF(category)  (unlikely(vn_env.perf & VN_PERF_##category))

#define vn_error(instance, error)                                            \
   (VN_DEBUG(RESULT) ? vn_log_result((instance), (error), __func__) : (error))
#define vn_result(instance, result)                                          \
   ((result) >= VK_SUCCESS ? (result) : vn_error((instance), (result)))

#define VN_TRACE_SCOPE(name) MESA_TRACE_SCOPE(name)
#define VN_TRACE_FUNC()      MESA_TRACE_SCOPE(__func__)

struct vn_instance;
struct vn_physical_device;
struct vn_device;
struct vn_queue;
struct vn_fence;
struct vn_semaphore;
struct vn_device_memory;
struct vn_buffer;
struct vn_buffer_view;
struct vn_image;
struct vn_image_view;
struct vn_sampler;
struct vn_sampler_ycbcr_conversion;
struct vn_descriptor_set_layout;
struct vn_descriptor_pool;
struct vn_descriptor_set;
struct vn_descriptor_update_template;
struct vn_render_pass;
struct vn_framebuffer;
struct vn_event;
struct vn_query_pool;
struct vn_shader_module;
struct vn_pipeline_layout;
struct vn_pipeline_cache;
struct vn_pipeline;
struct vn_command_pool;
struct vn_command_buffer;

struct vn_cs_encoder;
struct vn_cs_decoder;
struct vn_ring;

struct vn_renderer;
struct vn_renderer_shmem;
struct vn_renderer_bo;
struct vn_renderer_sync;

enum vn_debug {
   VN_DEBUG_INIT = 1ull << 0,
   VN_DEBUG_RESULT = 1ull << 1,
   VN_DEBUG_VTEST = 1ull << 2,
   VN_DEBUG_WSI = 1ull << 3,
   VN_DEBUG_NO_ABORT = 1ull << 4,
   VN_DEBUG_LOG_CTX_INFO = 1ull << 5,
   VN_DEBUG_CACHE = 1ull << 6,
   VN_DEBUG_NO_SPARSE = 1ull << 7,
   VN_DEBUG_NO_GPL = 1ull << 8,
   VN_DEBUG_NO_SECOND_QUEUE = 1ull << 9,
   VN_DEBUG_NO_RAY_TRACING = 1ull << 10,
   VN_DEBUG_MEM_BUDGET = 1ull << 11,
};

enum vn_perf {
   VN_PERF_NO_ASYNC_SET_ALLOC = 1ull << 0,
   VN_PERF_NO_ASYNC_BUFFER_CREATE = 1ull << 1,
   VN_PERF_NO_ASYNC_QUEUE_SUBMIT = 1ull << 2,
   VN_PERF_NO_EVENT_FEEDBACK = 1ull << 3,
   VN_PERF_NO_FENCE_FEEDBACK = 1ull << 4,
   VN_PERF_NO_CMD_BATCHING = 1ull << 6,
   VN_PERF_NO_SEMAPHORE_FEEDBACK = 1ull << 7,
   VN_PERF_NO_QUERY_FEEDBACK = 1ull << 8,
   VN_PERF_NO_ASYNC_MEM_ALLOC = 1ull << 9,
   VN_PERF_NO_TILED_WSI_IMAGE = 1ull << 10,
   VN_PERF_NO_MULTI_RING = 1ull << 11,
   VN_PERF_NO_ASYNC_IMAGE_CREATE = 1ull << 12,
   VN_PERF_NO_ASYNC_IMAGE_FORMAT = 1ull << 13,
};

typedef uint64_t vn_object_id;

/* base class of vn_instance */
struct vn_instance_base {
   struct vk_instance vk;
   vn_object_id id;
};

/* base class of vn_physical_device */
struct vn_physical_device_base {
   struct vk_physical_device vk;
   vn_object_id id;
};

/* base class of vn_device */
struct vn_device_base {
   struct vk_device vk;
   vn_object_id id;
};

/* base class of vn_queue */
struct vn_queue_base {
   struct vk_queue vk;
   vn_object_id id;
};

/* base class of vn_command_pool */
struct vn_command_pool_base {
   struct vk_command_pool vk;
   vn_object_id id;
};

/* base class of vn_command_buffer */
struct vn_command_buffer_base {
   struct vk_command_buffer vk;
   vn_object_id id;
};

/* base class of vn_device_memory */
struct vn_device_memory_base {
   struct vk_device_memory vk;
   vn_object_id id;
};

/* base class of vn_image */
struct vn_image_base {
   struct vk_image vk;
   vn_object_id id;
};

/* base class of other driver objects */
struct vn_object_base {
   struct vk_object_base vk;
   vn_object_id id;
};

struct vn_refcount {
   atomic_int count;
};

struct vn_env {
   uint64_t debug;
   uint64_t perf;
};
extern struct vn_env vn_env;

/* Only one "waiting" thread may fulfill the "watchdog" role at a time. Every
 * VN_WATCHDOG_REPORT_PERIOD_US or longer, the watchdog tests the ring's ALIVE
 * status, updates the "alive" atomic, and resets the ALIVE status for the
 * next cycle. Other waiting threads just check the "alive" atomic. The
 * watchdog role may be released and acquired by another waiting thread
 * dynamically.
 *
 * Examples of "waiting" are to wait for:
 * - ring to reach a seqno
 * - ring space to be released
 * - sync primitives to signal
 * - query result being available
 */
struct vn_watchdog {
   mtx_t mutex;
   atomic_int tid;
   atomic_bool alive;
};

enum vn_relax_reason {
   VN_RELAX_REASON_RING_SEQNO,
   VN_RELAX_REASON_TLS_RING_SEQNO,
   VN_RELAX_REASON_RING_SPACE,
   VN_RELAX_REASON_FENCE,
   VN_RELAX_REASON_SEMAPHORE,
   VN_RELAX_REASON_QUERY,
};

/* vn_relax_profile defines the driver side polling behavior
 *
 * - base_sleep_us:
 *   - the minimum polling interval after initial busy waits
 *
 * - busy_wait_order:
 *   - initial 2 ^ busy_wait_order times thrd_yield()
 *
 * - warn_order:
 *   - number of polls at order N:
 *     - fn_cnt(N) = 2 ^ N
 *   - interval of poll at order N:
 *     - fn_step(N) = base_sleep_us * (2 ^ (N - busy_wait_order))
 *   - warn occasionally if we have slept at least:
 *     - for (i = busy_wait_order; i < warn_order; i++)
 *          total_sleep += fn_cnt(i) * fn_step(i)
 *
 * - abort_order:
 *   - similar to warn_order, but would abort() instead
 */
struct vn_relax_profile {
   uint32_t base_sleep_us;
   uint32_t busy_wait_order;
   uint32_t warn_order;
   uint32_t abort_order;
};

struct vn_relax_state {
   struct vn_instance *instance;
   uint32_t iter;
   const struct vn_relax_profile profile;
   const char *reason_str;
};

/* TLS ring
 * - co-owned by TLS and VkInstance
 * - initialized in TLS upon requested
 * - teardown happens upon thread exit or instance destroy
 * - teardown is split into 2 stages:
 *   1. one owner locks and destroys the ring and mark destroyed
 *   2. the other owner locks and frees up the tls ring storage
 */
struct vn_tls_ring {
   mtx_t mutex;
   struct vn_ring *ring;
   struct vn_instance *instance;
   struct list_head tls_head;
   struct list_head vk_head;
};

struct vn_tls {
   /* Track the threads on which swapchain and command pool creations occur.
    * Pipeline create on those threads are forced async via the primary ring.
    */
   bool async_pipeline_create;
   /* Track TLS rings owned across instances. */
   struct list_head tls_rings;
};

/* A cached storage for object internal usages with below constraints:
 * - It belongs to the object and shares the lifetime.
 * - The storage reuse is protected by external synchronization.
 * - The returned storage is not zero-initialized.
 * - It never shrinks unless being purged via fini.
 *
 * The current users are:
 * - VkCommandPool
 * - VkQueue
 */
struct vn_cached_storage {
   const VkAllocationCallbacks *alloc;
   size_t size;
   void *data;
};

void
vn_env_init(void);

void
vn_trace_init(void);

void
vn_log(struct vn_instance *instance, const char *format, ...)
   PRINTFLIKE(2, 3);

VkResult
vn_log_result(struct vn_instance *instance,
              VkResult result,
              const char *where);

#define VN_REFCOUNT_INIT(val)                                                \
   (struct vn_refcount)                                                      \
   {                                                                         \
      .count = (val),                                                        \
   }

static inline int
vn_refcount_load_relaxed(const struct vn_refcount *ref)
{
   return atomic_load_explicit(&ref->count, memory_order_relaxed);
}

static inline int
vn_refcount_fetch_add_relaxed(struct vn_refcount *ref, int val)
{
   return atomic_fetch_add_explicit(&ref->count, val, memory_order_relaxed);
}

static inline int
vn_refcount_fetch_sub_release(struct vn_refcount *ref, int val)
{
   return atomic_fetch_sub_explicit(&ref->count, val, memory_order_release);
}

static inline bool
vn_refcount_is_valid(const struct vn_refcount *ref)
{
   return vn_refcount_load_relaxed(ref) > 0;
}

static inline void
vn_refcount_inc(struct vn_refcount *ref)
{
   /* no ordering imposed */
   ASSERTED const int old = vn_refcount_fetch_add_relaxed(ref, 1);
   assert(old >= 1);
}

static inline bool
vn_refcount_dec(struct vn_refcount *ref)
{
   /* prior reads/writes cannot be reordered after this */
   const int old = vn_refcount_fetch_sub_release(ref, 1);
   assert(old >= 1);

   /* subsequent free cannot be reordered before this */
   if (old == 1)
      atomic_thread_fence(memory_order_acquire);

   return old == 1;
}

extern uint64_t vn_next_obj_id;

static inline uint64_t
vn_get_next_obj_id(void)
{
   return p_atomic_fetch_add(&vn_next_obj_id, 1);
}

uint32_t
vn_extension_get_spec_version(const char *name);

static inline void
vn_watchdog_init(struct vn_watchdog *watchdog)
{
#ifndef NDEBUG
   /* ensure minimum check period is greater than maximum renderer
    * reporting period (with margin of safety to ensure no false
    * positives).
    *
    * first_warn_time is pre-calculated based on parameters in vn_relax
    * and must update together.
    */
   static const uint32_t first_warn_time = 3481600;
   static const uint32_t safety_margin = 250000;
   assert(first_warn_time - safety_margin >= VN_WATCHDOG_REPORT_PERIOD_US);
#endif

   mtx_init(&watchdog->mutex, mtx_plain);

   watchdog->tid = 0;

   /* initialized to be alive to avoid vn_watchdog_timout false alarm */
   watchdog->alive = true;
}

static inline void
vn_watchdog_fini(struct vn_watchdog *watchdog)
{
   mtx_destroy(&watchdog->mutex);
}

struct vn_relax_state
vn_relax_init(struct vn_instance *instance, enum vn_relax_reason reason);

void
vn_relax(struct vn_relax_state *state);

void
vn_relax_fini(struct vn_relax_state *state);

static_assert(sizeof(vn_object_id) >= sizeof(uintptr_t), "");

static inline VkResult
vn_instance_base_init(
   struct vn_instance_base *instance,
   const struct vk_instance_extension_table *supported_extensions,
   const struct vk_instance_dispatch_table *dispatch_table,
   const VkInstanceCreateInfo *info,
   const VkAllocationCallbacks *alloc)
{
   VkResult result = vk_instance_init(&instance->vk, supported_extensions,
                                      dispatch_table, info, alloc);
   instance->id = vn_get_next_obj_id();
   return result;
}

static inline void
vn_instance_base_fini(struct vn_instance_base *instance)
{
   vk_instance_finish(&instance->vk);
}

static inline VkResult
vn_physical_device_base_init(
   struct vn_physical_device_base *physical_dev,
   struct vn_instance_base *instance,
   const struct vk_device_extension_table *supported_extensions,
   const struct vk_physical_device_dispatch_table *dispatch_table)
{
   VkResult result = vk_physical_device_init(&physical_dev->vk, &instance->vk,
                                             supported_extensions, NULL, NULL,
                                             dispatch_table);
   physical_dev->id = vn_get_next_obj_id();
   return result;
}

static inline void
vn_physical_device_base_fini(struct vn_physical_device_base *physical_dev)
{
   vk_physical_device_finish(&physical_dev->vk);
}

static inline VkResult
vn_device_base_init(struct vn_device_base *dev,
                    struct vn_physical_device_base *physical_dev,
                    const struct vk_device_dispatch_table *dispatch_table,
                    const VkDeviceCreateInfo *info,
                    const VkAllocationCallbacks *alloc)
{
   VkResult result = vk_device_init(&dev->vk, &physical_dev->vk,
                                    dispatch_table, info, alloc);
   dev->id = vn_get_next_obj_id();
   return result;
}

static inline void
vn_device_base_fini(struct vn_device_base *dev)
{
   vk_device_finish(&dev->vk);
}

static inline VkResult
vn_queue_base_init(struct vn_queue_base *queue,
                   struct vn_device_base *dev,
                   const VkDeviceQueueCreateInfo *queue_info,
                   uint32_t queue_index)
{
   VkResult result =
      vk_queue_init(&queue->vk, &dev->vk, queue_info, queue_index);
   queue->id = vn_get_next_obj_id();
   return result;
}

static inline void
vn_queue_base_fini(struct vn_queue_base *queue)
{
   vk_queue_finish(&queue->vk);
}

static inline VkResult
vn_command_pool_base_init(struct vn_command_pool_base *cmd_pool,
                          struct vn_device_base *dev,
                          const VkCommandPoolCreateInfo *info,
                          const VkAllocationCallbacks *alloc)
{
   VkResult result =
      vk_command_pool_init(&dev->vk, &cmd_pool->vk, info, alloc);
   cmd_pool->id = vn_get_next_obj_id();
   return result;
}

static inline void
vn_command_pool_base_fini(struct vn_command_pool_base *cmd_pool)
{
   vk_command_pool_finish(&cmd_pool->vk);
}

static inline VkResult
vn_command_buffer_base_init(struct vn_command_buffer_base *cmd,
                            struct vn_command_pool_base *cmd_pool,
                            const struct vk_command_buffer_ops *ops,
                            VkCommandBufferLevel level)
{
   VkResult result =
      vk_command_buffer_init(&cmd_pool->vk, &cmd->vk, ops, level);
   cmd->id = vn_get_next_obj_id();
   return result;
}

static inline void
vn_command_buffer_base_fini(struct vn_command_buffer_base *cmd)
{
   vk_command_buffer_finish(&cmd->vk);
}

static inline void
vn_object_base_init(struct vn_object_base *obj,
                    VkObjectType type,
                    struct vn_device_base *dev)
{
   vk_object_base_init(&dev->vk, &obj->vk, type);
   obj->id = vn_get_next_obj_id();
}

static inline void
vn_object_base_fini(struct vn_object_base *obj)
{
   vk_object_base_finish(&obj->vk);
}

static inline void
vn_object_set_id(void *obj, vn_object_id id, VkObjectType type)
{
   assert(((const struct vk_object_base *)obj)->type == type);
   switch (type) {
   case VK_OBJECT_TYPE_INSTANCE:
      ((struct vn_instance_base *)obj)->id = id;
      break;
   case VK_OBJECT_TYPE_PHYSICAL_DEVICE:
      ((struct vn_physical_device_base *)obj)->id = id;
      break;
   case VK_OBJECT_TYPE_DEVICE:
      ((struct vn_device_base *)obj)->id = id;
      break;
   case VK_OBJECT_TYPE_QUEUE:
      ((struct vn_queue_base *)obj)->id = id;
      break;
   case VK_OBJECT_TYPE_COMMAND_POOL:
      ((struct vn_command_pool_base *)obj)->id = id;
      break;
   case VK_OBJECT_TYPE_COMMAND_BUFFER:
      ((struct vn_command_buffer_base *)obj)->id = id;
      break;
   case VK_OBJECT_TYPE_DEVICE_MEMORY:
      ((struct vn_device_memory_base *)obj)->id = id;
      break;
   case VK_OBJECT_TYPE_IMAGE:
      ((struct vn_image_base *)obj)->id = id;
      break;
   default:
      ((struct vn_object_base *)obj)->id = id;
      break;
   }
}

static inline vn_object_id
vn_object_get_id(const void *obj, VkObjectType type)
{
   assert(((const struct vk_object_base *)obj)->type == type);
   switch (type) {
   case VK_OBJECT_TYPE_INSTANCE:
      return ((struct vn_instance_base *)obj)->id;
   case VK_OBJECT_TYPE_PHYSICAL_DEVICE:
      return ((struct vn_physical_device_base *)obj)->id;
   case VK_OBJECT_TYPE_DEVICE:
      return ((struct vn_device_base *)obj)->id;
   case VK_OBJECT_TYPE_QUEUE:
      return ((struct vn_queue_base *)obj)->id;
   case VK_OBJECT_TYPE_COMMAND_POOL:
      return ((struct vn_command_pool_base *)obj)->id;
   case VK_OBJECT_TYPE_COMMAND_BUFFER:
      return ((struct vn_command_buffer_base *)obj)->id;
   case VK_OBJECT_TYPE_DEVICE_MEMORY:
      return ((struct vn_device_memory_base *)obj)->id;
   case VK_OBJECT_TYPE_IMAGE:
      return ((struct vn_image_base *)obj)->id;
   default:
      return ((struct vn_object_base *)obj)->id;
   }
}

static inline pid_t
vn_gettid(void)
{
#if DETECT_OS_ANDROID
   return gettid();
#else
   return syscall(SYS_gettid);
#endif
}

struct vn_tls *
vn_tls_get(void);

static inline void
vn_tls_set_async_pipeline_create(void)
{
   struct vn_tls *tls = vn_tls_get();
   if (likely(tls))
      tls->async_pipeline_create = true;
}

static inline bool
vn_tls_get_async_pipeline_create(void)
{
   const struct vn_tls *tls = vn_tls_get();
   if (likely(tls))
      return tls->async_pipeline_create;
   return true;
}

struct vn_ring *
vn_tls_get_ring(struct vn_instance *instance);

void
vn_tls_destroy_ring(struct vn_tls_ring *tls_ring);

static inline uint32_t
vn_cache_key_hash_function(const void *key)
{
   return _mesa_hash_data(key, SHA1_DIGEST_LENGTH);
}

static inline bool
vn_cache_key_equal_function(const void *key1, const void *key2)
{
   return memcmp(key1, key2, SHA1_DIGEST_LENGTH) == 0;
}

static inline void
vn_cached_storage_init(struct vn_cached_storage *storage,
                       const VkAllocationCallbacks *alloc)
{
   storage->alloc = alloc;
   storage->size = 0;
   storage->data = NULL;
}

static inline void *
vn_cached_storage_get(struct vn_cached_storage *storage, size_t size)
{
   if (size > storage->size) {
      void *data =
         vk_realloc(storage->alloc, storage->data, size, VN_DEFAULT_ALIGN,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!data)
         return NULL;

      storage->size = size;
      storage->data = data;
   }
   return storage->data;
}

static inline void
vn_cached_storage_fini(struct vn_cached_storage *storage)
{
   vk_free(storage->alloc, storage->data);
}

#endif /* VN_COMMON_H */
