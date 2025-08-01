/*
 * Copyright © 2020 Google, Inc.
 * SPDX-License-Identifier: MIT
 */

#include "tu_knl.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-heap.h>

#define __user
#include "msm_kgsl.h"
#include "ion/ion.h"
#include "ion/ion_4.19.h"

#include "vk_util.h"

#include "util/os_file.h"
#include "util/u_debug.h"
#include "util/u_vector.h"
#include "util/libsync.h"
#include "util/timespec.h"

#include "tu_cmd_buffer.h"
#include "tu_cs.h"
#include "tu_device.h"
#include "tu_dynamic_rendering.h"
#include "tu_queue.h"
#include "tu_rmv.h"

/* ION_HEAP(ION_SYSTEM_HEAP_ID) */
#define KGSL_ION_SYSTEM_HEAP_MASK (1u << 25)


static int
safe_ioctl(int fd, unsigned long request, void *arg)
{
   int ret;

   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

   return ret;
}

static int
kgsl_submitqueue_new(struct tu_device *dev,
                     int priority,
                     uint32_t *queue_id)
{
   struct kgsl_drawctxt_create req = {
      .flags = KGSL_CONTEXT_SAVE_GMEM |
              KGSL_CONTEXT_NO_GMEM_ALLOC |
              KGSL_CONTEXT_PREAMBLE,
   };

   int ret = safe_ioctl(dev->physical_device->local_fd, IOCTL_KGSL_DRAWCTXT_CREATE, &req);
   if (ret)
      return ret;

   *queue_id = req.drawctxt_id;

   return 0;
}

static void
kgsl_submitqueue_close(struct tu_device *dev, uint32_t queue_id)
{
   struct kgsl_drawctxt_destroy req = {
      .drawctxt_id = queue_id,
   };

   safe_ioctl(dev->physical_device->local_fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &req);
}

static void kgsl_bo_finish(struct tu_device *dev, struct tu_bo *bo);

static VkResult
bo_init_new_dmaheap(struct tu_device *dev, struct tu_bo **out_bo, uint64_t size,
                enum tu_bo_alloc_flags flags)
{
   struct dma_heap_allocation_data alloc = {
      .len = size,
      .fd_flags = O_RDWR | O_CLOEXEC,
   };

   int ret;
   ret = safe_ioctl(dev->physical_device->kgsl_dma_fd, DMA_HEAP_IOCTL_ALLOC,
                    &alloc);

   if (ret) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "DMA_HEAP_IOCTL_ALLOC failed (%s)", strerror(errno));
   }

   return tu_bo_init_dmabuf(dev, out_bo, -1, alloc.fd);
}

static VkResult
bo_init_new_ion(struct tu_device *dev, struct tu_bo **out_bo, uint64_t size,
                enum tu_bo_alloc_flags flags)
{
   struct ion_new_allocation_data alloc = {
      .len = size,
      .heap_id_mask = KGSL_ION_SYSTEM_HEAP_MASK,
      .flags = 0,
      .fd = -1,
   };

   int ret;
   ret = safe_ioctl(dev->physical_device->kgsl_dma_fd, ION_IOC_NEW_ALLOC, &alloc);
   if (ret) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "ION_IOC_NEW_ALLOC failed (%s)", strerror(errno));
   }

   return tu_bo_init_dmabuf(dev, out_bo, -1, alloc.fd);
}

static VkResult
bo_init_new_ion_legacy(struct tu_device *dev, struct tu_bo **out_bo, uint64_t size,
                       enum tu_bo_alloc_flags flags)
{
   struct ion_allocation_data alloc = {
      .len = size,
      .align = 4096,
      .heap_id_mask = KGSL_ION_SYSTEM_HEAP_MASK,
      .flags = 0,
      .handle = -1,
   };

   int ret;
   ret = safe_ioctl(dev->physical_device->kgsl_dma_fd, ION_IOC_ALLOC, &alloc);
   if (ret) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "ION_IOC_ALLOC failed (%s)", strerror(errno));
   }

   struct ion_fd_data share = {
      .handle = alloc.handle,
      .fd = -1,
   };

   ret = safe_ioctl(dev->physical_device->kgsl_dma_fd, ION_IOC_SHARE, &share);
   if (ret) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "ION_IOC_SHARE failed (%s)", strerror(errno));
   }

   struct ion_handle_data free = {
      .handle = alloc.handle,
   };
   ret = safe_ioctl(dev->physical_device->kgsl_dma_fd, ION_IOC_FREE, &free);
   if (ret) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "ION_IOC_FREE failed (%s)", strerror(errno));
   }

   return tu_bo_init_dmabuf(dev, out_bo, -1, share.fd);
}

static VkResult
kgsl_bo_init(struct tu_device *dev,
             struct vk_object_base *base,
             struct tu_bo **out_bo,
             uint64_t size,
             uint64_t client_iova,
             VkMemoryPropertyFlags mem_property,
             enum tu_bo_alloc_flags flags,
             const char *name)
{
   if (flags & TU_BO_ALLOC_SHAREABLE) {
      /* The Vulkan spec doesn't forbid allocating exportable memory with a
       * fixed address, only imported memory, but on kgsl we can't sensibly
       * implement it so just always reject it.
       */
      if (client_iova) {
         return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                          "cannot allocate an exportable BO with a fixed address");
      }

      switch(dev->physical_device->kgsl_dma_type) {
      case TU_KGSL_DMA_TYPE_DMAHEAP:
         return bo_init_new_dmaheap(dev, out_bo, size, flags);
      case TU_KGSL_DMA_TYPE_ION:
         return bo_init_new_ion(dev, out_bo, size, flags);
      case TU_KGSL_DMA_TYPE_ION_LEGACY:
         return bo_init_new_ion_legacy(dev, out_bo, size, flags);
      }
   }

   struct kgsl_gpumem_alloc_id req = {
      .size = size,
   };

   if (mem_property & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
      if (mem_property & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
         req.flags |= KGSL_MEMFLAGS_IOCOHERENT;
      }

      req.flags |= KGSL_CACHEMODE_WRITEBACK << KGSL_CACHEMODE_SHIFT;
   } else {
      req.flags |= KGSL_CACHEMODE_WRITECOMBINE << KGSL_CACHEMODE_SHIFT;
   }

   if (flags & TU_BO_ALLOC_GPU_READ_ONLY)
      req.flags |= KGSL_MEMFLAGS_GPUREADONLY;

   if (flags & TU_BO_ALLOC_REPLAYABLE)
      req.flags |= KGSL_MEMFLAGS_USE_CPU_MAP;

   int ret;

   ret = safe_ioctl(dev->physical_device->local_fd,
                    IOCTL_KGSL_GPUMEM_ALLOC_ID, &req);
   if (ret) {
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "GPUMEM_ALLOC_ID failed (%s)", strerror(errno));
   }

   struct tu_bo* bo = tu_device_lookup_bo(dev, req.id);
   assert(bo && bo->gem_handle == 0);

   *bo = (struct tu_bo) {
      .gem_handle = req.id,
      .size = req.mmapsize,
      .iova = req.gpuaddr,
      .name = tu_debug_bos_add(dev, req.mmapsize, name),
      .refcnt = 1,
      .shared_fd = -1,
      .base = base,
   };

   if (flags & TU_BO_ALLOC_REPLAYABLE) {
      uint64_t offset = req.id << 12;
      void *map = mmap((void *)client_iova, bo->size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, dev->physical_device->local_fd, offset);
      if (map == MAP_FAILED) {
         kgsl_bo_finish(dev, bo);

         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "mmap failed (%s)", strerror(errno));
      }

      if (client_iova && (uint64_t)map != client_iova) {
         kgsl_bo_finish(dev, bo);

         return vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                          "mmap could not map the given address");
      }

      bo->map = map;
      bo->iova = (uint64_t)map;

      /* Because we're using SVM, the CPU mapping and GPU mapping are the same
       * and the CPU mapping must stay fixed for the lifetime of the BO.
       */
      bo->never_unmap = true;
   }

   tu_dump_bo_init(dev, bo);

   *out_bo = bo;

   TU_RMV(bo_allocate, dev, bo);
   if (flags & TU_BO_ALLOC_INTERNAL_RESOURCE) {
      TU_RMV(internal_resource_create, dev, bo);
      TU_RMV(resource_name, dev, bo, name);
   }

   return VK_SUCCESS;
}

static VkResult
kgsl_bo_init_dmabuf(struct tu_device *dev,
                    struct tu_bo **out_bo,
                    uint64_t size,
                    int fd)
{
   struct kgsl_gpuobj_import_dma_buf import_dmabuf = {
      .fd = fd,
   };
   struct kgsl_gpuobj_import req = {
      .priv = (uintptr_t)&import_dmabuf,
      .priv_len = sizeof(import_dmabuf),
      .flags = 0,
      .type = KGSL_USER_MEM_TYPE_DMABUF,
   };
   int ret;

   ret = safe_ioctl(dev->physical_device->local_fd,
                    IOCTL_KGSL_GPUOBJ_IMPORT, &req);
   if (ret)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Failed to import dma-buf (%s)\n", strerror(errno));

   struct kgsl_gpuobj_info info_req = {
      .id = req.id,
   };

   ret = safe_ioctl(dev->physical_device->local_fd,
                    IOCTL_KGSL_GPUOBJ_INFO, &info_req);
   if (ret)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Failed to get dma-buf info (%s)\n", strerror(errno));

   struct tu_bo* bo = tu_device_lookup_bo(dev, req.id);
   assert(bo && bo->gem_handle == 0);

   *bo = (struct tu_bo) {
      .gem_handle = req.id,
      .size = info_req.size,
      .iova = info_req.gpuaddr,
      .name = tu_debug_bos_add(dev, info_req.size, "dmabuf"),
      .refcnt = 1,
      .shared_fd = os_dupfd_cloexec(fd),
   };

   tu_dump_bo_init(dev, bo);

   *out_bo = bo;

   return VK_SUCCESS;
}

static int
kgsl_bo_export_dmabuf(struct tu_device *dev, struct tu_bo *bo)
{
   assert(bo->shared_fd != -1);
   return os_dupfd_cloexec(bo->shared_fd);
}

static VkResult
kgsl_bo_map(struct tu_device *dev, struct tu_bo *bo, void *placed_addr)
{
   void *map = MAP_FAILED;
   if (bo->shared_fd == -1) {
      uint64_t offset = bo->gem_handle << 12;
      map = mmap(placed_addr, bo->size, PROT_READ | PROT_WRITE,
                 MAP_SHARED | (placed_addr != NULL ? MAP_FIXED : 0),
                 dev->physical_device->local_fd, offset);
   } else {
      map = mmap(placed_addr, bo->size, PROT_READ | PROT_WRITE,
                 MAP_SHARED | (placed_addr != NULL ? MAP_FIXED : 0),
                 bo->shared_fd, 0);
   }

   if (map == MAP_FAILED)
      return vk_error(dev, VK_ERROR_MEMORY_MAP_FAILED);

   bo->map = map;
   TU_RMV(bo_map, dev, bo);

   return VK_SUCCESS;
}

static void
kgsl_bo_allow_dump(struct tu_device *dev, struct tu_bo *bo)
{
}

static void
kgsl_bo_finish(struct tu_device *dev, struct tu_bo *bo)
{
   assert(bo->gem_handle);

   if (!p_atomic_dec_zero(&bo->refcnt))
      return;

   if (bo->map) {
      TU_RMV(bo_unmap, dev, bo);
      munmap(bo->map, bo->size);
   }

   if (bo->shared_fd != -1)
      close(bo->shared_fd);

   TU_RMV(bo_destroy, dev, bo);
   tu_debug_bos_del(dev, bo);
   tu_dump_bo_del(dev, bo);

   struct kgsl_gpumem_free_id req = {
      .id = bo->gem_handle
   };

   /* Tell sparse array that entry is free */
   memset(bo, 0, sizeof(*bo));

   safe_ioctl(dev->physical_device->local_fd, IOCTL_KGSL_GPUMEM_FREE_ID, &req);
}

static VkResult
get_kgsl_prop(int fd, unsigned int type, void *value, size_t size)
{
   struct kgsl_device_getproperty getprop = {
      .type = type,
      .value = value,
      .sizebytes = size,
   };

   return safe_ioctl(fd, IOCTL_KGSL_DEVICE_GETPROPERTY, &getprop)
             ? VK_ERROR_UNKNOWN
             : VK_SUCCESS;
}

static bool
kgsl_is_memory_type_supported(int fd, uint32_t flags)
{
   struct kgsl_gpumem_alloc_id req_alloc = {
      .flags = flags,
      .size = 0x1000,
   };

   int ret = safe_ioctl(fd, IOCTL_KGSL_GPUMEM_ALLOC_ID, &req_alloc);
   if (ret) {
      return false;
   }

   struct kgsl_gpumem_free_id req_free = { .id = req_alloc.id };

   safe_ioctl(fd, IOCTL_KGSL_GPUMEM_FREE_ID, &req_free);

   return true;
}

enum kgsl_syncobj_state {
   KGSL_SYNCOBJ_STATE_UNSIGNALED,
   KGSL_SYNCOBJ_STATE_SIGNALED,
   KGSL_SYNCOBJ_STATE_TS,
   KGSL_SYNCOBJ_STATE_FD,
};

struct kgsl_syncobj
{
   struct vk_object_base base;
   enum kgsl_syncobj_state state;

   struct tu_queue *queue;
   uint32_t timestamp;

   int fd;
};

static void
kgsl_syncobj_init(struct kgsl_syncobj *s, bool signaled)
{
   s->state =
      signaled ? KGSL_SYNCOBJ_STATE_SIGNALED : KGSL_SYNCOBJ_STATE_UNSIGNALED;

   s->timestamp = UINT32_MAX;
   s->fd = -1;
}

static void
kgsl_syncobj_reset(struct kgsl_syncobj *s)
{
   if (s->state == KGSL_SYNCOBJ_STATE_FD && s->fd >= 0) {
      ASSERTED int ret = close(s->fd);
      assert(ret == 0);
   }

   s->state = KGSL_SYNCOBJ_STATE_UNSIGNALED;
   s->timestamp = UINT32_MAX;
   s->fd = -1;
}

static void
kgsl_syncobj_destroy(struct kgsl_syncobj *s)
{
   kgsl_syncobj_reset(s);
}

static struct kgsl_syncobj
kgsl_syncobj_dup(struct kgsl_syncobj *s)
{
   struct kgsl_syncobj dups = *s;
   if (s->state == KGSL_SYNCOBJ_STATE_FD && s->fd >= 0) {
      dups.fd = dup(s->fd);
      assert(dups.fd >= 0);
   }
   return dups;
}

static int
timestamp_to_fd(struct tu_queue *queue, uint32_t timestamp)
{
   int fd;
   struct kgsl_timestamp_event event = {
      .type = KGSL_TIMESTAMP_EVENT_FENCE,
      .timestamp = timestamp,
      .context_id = queue->msm_queue_id,
      .priv = &fd,
      .len = sizeof(fd),
   };

   int ret = safe_ioctl(queue->device->fd, IOCTL_KGSL_TIMESTAMP_EVENT, &event);
   if (ret)
      return -1;

   return fd;
}

static int
kgsl_syncobj_ts_to_fd(const struct kgsl_syncobj *syncobj)
{
   assert(syncobj->state == KGSL_SYNCOBJ_STATE_TS);
   return timestamp_to_fd(syncobj->queue, syncobj->timestamp);
}

/* return true if timestamp a is greater (more recent) then b
 * this relies on timestamps never having a difference > (1<<31)
 */
static inline bool
timestamp_cmp(uint32_t a, uint32_t b)
{
   return (int32_t) (a - b) >= 0;
}

static uint32_t
max_ts(uint32_t a, uint32_t b)
{
   return timestamp_cmp(a, b) ? a : b;
}

static uint32_t
min_ts(uint32_t a, uint32_t b)
{
   return timestamp_cmp(a, b) ? b : a;
}

static int
get_relative_ms(uint64_t abs_timeout_ns)
{
   if (abs_timeout_ns >= INT64_MAX)
      /* We can assume that a wait with a value this high is a forever wait
       * and return -1 here as it's the infinite timeout for ppoll() while
       * being the highest unsigned integer value for the wait KGSL IOCTL
       */
      return -1;

   uint64_t cur_time_ms = os_time_get_nano() / 1000000;
   uint64_t abs_timeout_ms = abs_timeout_ns / 1000000;
   if (abs_timeout_ms <= cur_time_ms)
      return 0;

   return abs_timeout_ms - cur_time_ms;
}

/* safe_ioctl is not enough as restarted waits would not adjust the timeout
 * which could lead to waiting substantially longer than requested
 */
static VkResult
wait_timestamp_safe(int fd,
                    unsigned int context_id,
                    unsigned int timestamp,
                    uint64_t abs_timeout_ns)
{
   struct kgsl_device_waittimestamp_ctxtid wait = {
      .context_id = context_id,
      .timestamp = timestamp,
      .timeout = get_relative_ms(abs_timeout_ns),
   };

   while (true) {
      int ret = ioctl(fd, IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, &wait);

      if (ret == -1 && (errno == EINTR || errno == EAGAIN)) {
         int timeout_ms = get_relative_ms(abs_timeout_ns);

         /* update timeout to consider time that has passed since the start */
         if (timeout_ms == 0)
            return VK_TIMEOUT;

         wait.timeout = timeout_ms;
      } else if (ret == -1) {
         assert(errno == ETIMEDOUT);
         return VK_TIMEOUT;
      } else {
         return VK_SUCCESS;
      }
   }
}

VkResult
kgsl_queue_wait_fence(struct tu_queue *queue, uint32_t fence,
                      uint64_t timeout_ns)
{
   uint64_t abs_timeout_ns = os_time_get_nano() + timeout_ns;

   return wait_timestamp_safe(queue->device->fd, queue->msm_queue_id,
                              fence, abs_timeout_ns);
}

static VkResult
kgsl_syncobj_wait(struct tu_device *device,
                  struct kgsl_syncobj *s,
                  uint64_t abs_timeout_ns)
{
   if (s->state == KGSL_SYNCOBJ_STATE_UNSIGNALED) {
      /* If this syncobj is unsignaled we need to wait for it to resolve to a
       * valid syncobj prior to letting the rest of the wait continue, this
       * avoids needing kernel support for wait-before-signal semantics.
       */

      if (abs_timeout_ns == 0)
         return VK_TIMEOUT; // If this is a simple poll then we can return early

      pthread_mutex_lock(&device->submit_mutex);
      struct timespec abstime;
      timespec_from_nsec(&abstime, abs_timeout_ns);

      while (s->state == KGSL_SYNCOBJ_STATE_UNSIGNALED) {
         int ret;
         if (abs_timeout_ns == UINT64_MAX) {
            ret = pthread_cond_wait(&device->timeline_cond,
                                    &device->submit_mutex);
         } else {
            ret = pthread_cond_timedwait(&device->timeline_cond,
                                         &device->submit_mutex, &abstime);
         }
         if (ret != 0) {
            assert(ret == ETIMEDOUT);
            pthread_mutex_unlock(&device->submit_mutex);
            return VK_TIMEOUT;
         }
      }

      pthread_mutex_unlock(&device->submit_mutex);
   }

   switch (s->state) {
   case KGSL_SYNCOBJ_STATE_SIGNALED:
      return VK_SUCCESS;

   case KGSL_SYNCOBJ_STATE_UNSIGNALED:
      return VK_TIMEOUT;

   case KGSL_SYNCOBJ_STATE_TS: {
      return wait_timestamp_safe(device->fd, s->queue->msm_queue_id,
                                 s->timestamp, abs_timeout_ns);
   }

   case KGSL_SYNCOBJ_STATE_FD: {
      int ret = sync_wait(s->fd, get_relative_ms(abs_timeout_ns));
      if (ret) {
         assert(errno == ETIME);
         return VK_TIMEOUT;
      } else {
         return VK_SUCCESS;
      }
   }

   default:
      UNREACHABLE("invalid syncobj state");
   }
}

#define kgsl_syncobj_foreach_state(syncobjs, filter) \
   for (uint32_t i = 0; sync = syncobjs[i], i < count; i++) \
      if (sync->state == filter)

static VkResult
kgsl_syncobj_wait_any(struct tu_device* device, struct kgsl_syncobj **syncobjs, uint32_t count, uint64_t abs_timeout_ns)
{
   if (count == 0)
      return VK_TIMEOUT;
   else if (count == 1)
      return kgsl_syncobj_wait(device, syncobjs[0], abs_timeout_ns);

   uint32_t num_fds = 0;
   struct tu_queue *queue = NULL;
   struct kgsl_syncobj *sync = NULL;

   /* Simple case, we already have a signal one */
   kgsl_syncobj_foreach_state(syncobjs, KGSL_SYNCOBJ_STATE_SIGNALED)
      return VK_SUCCESS;

   kgsl_syncobj_foreach_state(syncobjs, KGSL_SYNCOBJ_STATE_FD)
      num_fds++;

   /* If we have TS from different queues we cannot compare them and would
    * have to convert them into FDs
    */
   bool convert_ts_to_fd = false;
   kgsl_syncobj_foreach_state(syncobjs, KGSL_SYNCOBJ_STATE_TS) {
      if (queue != NULL && sync->queue != queue) {
         convert_ts_to_fd = true;
         break;
      }
      queue = sync->queue;
   }

   /* If we have no FD nor TS syncobjs then we can return immediately */
   if (num_fds == 0 && queue == NULL)
      return VK_TIMEOUT;

   VkResult result = VK_TIMEOUT;

   struct u_vector poll_fds = { 0 };
   uint32_t lowest_timestamp = 0;

   if (convert_ts_to_fd || num_fds > 0)
      u_vector_init(&poll_fds, 4, sizeof(struct pollfd));

   if (convert_ts_to_fd) {
      kgsl_syncobj_foreach_state(syncobjs, KGSL_SYNCOBJ_STATE_TS) {
         struct pollfd *poll_fd = (struct pollfd *) u_vector_add(&poll_fds);
         poll_fd->fd = timestamp_to_fd(sync->queue, sync->timestamp);
         poll_fd->events = POLLIN;
      }
   } else {
      /* TSs could be merged by finding the one with the lowest timestamp */
      bool first_ts = true;
      kgsl_syncobj_foreach_state(syncobjs, KGSL_SYNCOBJ_STATE_TS) {
         if (first_ts || timestamp_cmp(sync->timestamp, lowest_timestamp)) {
            first_ts = false;
            lowest_timestamp = sync->timestamp;
         }
      }

      if (num_fds) {
         struct pollfd *poll_fd = (struct pollfd *) u_vector_add(&poll_fds);
         poll_fd->fd = timestamp_to_fd(queue, lowest_timestamp);
         poll_fd->events = POLLIN;
      }
   }

   if (num_fds) {
      kgsl_syncobj_foreach_state(syncobjs, KGSL_SYNCOBJ_STATE_FD) {
         struct pollfd *poll_fd = (struct pollfd *) u_vector_add(&poll_fds);
         poll_fd->fd = sync->fd;
         poll_fd->events = POLLIN;
      }
   }

   if (u_vector_length(&poll_fds) == 0) {
      result = wait_timestamp_safe(device->fd, queue->msm_queue_id,
                                   lowest_timestamp, MIN2(abs_timeout_ns, INT64_MAX));
   } else {
      int ret, i;

      struct pollfd *fds = (struct pollfd *) poll_fds.data;
      uint32_t fds_count = u_vector_length(&poll_fds);
      do {
         ret = poll(fds, fds_count, get_relative_ms(abs_timeout_ns));
         if (ret > 0) {
            for (i = 0; i < fds_count; i++) {
               if (fds[i].revents & (POLLERR | POLLNVAL)) {
                  errno = EINVAL;
                  ret = -1;
                  break;
               }
            }
            break;
         } else if (ret == 0) {
            errno = ETIME;
            break;
         }
      } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

      for (uint32_t i = 0; i < fds_count - num_fds; i++)
         close(fds[i].fd);

      if (ret != 0) {
         assert(errno == ETIME);
         result = VK_TIMEOUT;
      } else {
         result = VK_SUCCESS;
      }
   }

   u_vector_finish(&poll_fds);
   return result;
}

static VkResult
kgsl_syncobj_export(struct kgsl_syncobj *s, int *pFd)
{
   if (!pFd)
      return VK_SUCCESS;

   switch (s->state) {
   case KGSL_SYNCOBJ_STATE_SIGNALED:
   case KGSL_SYNCOBJ_STATE_UNSIGNALED:
      /* Getting a sync FD from an unsignaled syncobj is UB in Vulkan */
      *pFd = -1;
      return VK_SUCCESS;

   case KGSL_SYNCOBJ_STATE_FD:
      if (s->fd < 0)
         *pFd = -1;
      else
         *pFd = os_dupfd_cloexec(s->fd);
      return VK_SUCCESS;

   case KGSL_SYNCOBJ_STATE_TS:
      *pFd = kgsl_syncobj_ts_to_fd(s);
      return VK_SUCCESS;

   default:
      UNREACHABLE("Invalid syncobj state");
   }
}

static VkResult
kgsl_syncobj_import(struct kgsl_syncobj *s, int fd)
{
   kgsl_syncobj_reset(s);
   if (fd >= 0) {
      s->state = KGSL_SYNCOBJ_STATE_FD;
      s->fd = fd;
   } else {
      s->state = KGSL_SYNCOBJ_STATE_SIGNALED;
   }

   return VK_SUCCESS;
}

static int
sync_merge_close(const char *name, int fd1, int fd2, bool close_fd2)
{
   int fd = sync_merge(name, fd1, fd2);
   if (fd < 0)
      return -1;

   close(fd1);
   if (close_fd2)
      close(fd2);

   return fd;
}

/* Merges multiple kgsl_syncobjs into a single one which is only signalled
 * after all submitted syncobjs are signalled
 */
static struct kgsl_syncobj
kgsl_syncobj_merge(const struct kgsl_syncobj **syncobjs, uint32_t count)
{
   struct kgsl_syncobj ret;
   kgsl_syncobj_init(&ret, true);

   if (count == 0)
      return ret;

   for (uint32_t i = 0; i < count; ++i) {
      const struct kgsl_syncobj *sync = syncobjs[i];

      switch (sync->state) {
      case KGSL_SYNCOBJ_STATE_SIGNALED:
         break;

      case KGSL_SYNCOBJ_STATE_UNSIGNALED:
         kgsl_syncobj_reset(&ret);
         return ret;

      case KGSL_SYNCOBJ_STATE_TS:
         if (ret.state == KGSL_SYNCOBJ_STATE_TS) {
            if (ret.queue == sync->queue) {
               ret.timestamp = max_ts(ret.timestamp, sync->timestamp);
            } else {
               ret.state = KGSL_SYNCOBJ_STATE_FD;
               int sync_fd = kgsl_syncobj_ts_to_fd(sync);
               ret.fd = sync_merge_close("tu_sync", ret.fd, sync_fd, true);
               assert(ret.fd >= 0);
            }
         } else if (ret.state == KGSL_SYNCOBJ_STATE_FD) {
            int sync_fd = kgsl_syncobj_ts_to_fd(sync);
            ret.fd = sync_merge_close("tu_sync", ret.fd, sync_fd, true);
            assert(ret.fd >= 0);
         } else {
            ret = *sync;
         }
         break;

      case KGSL_SYNCOBJ_STATE_FD:
         if (ret.state == KGSL_SYNCOBJ_STATE_FD) {
            ret.fd = sync_merge_close("tu_sync", ret.fd, sync->fd, false);
            assert(ret.fd >= 0);
         } else if (ret.state == KGSL_SYNCOBJ_STATE_TS) {
            ret.state = KGSL_SYNCOBJ_STATE_FD;
            int sync_fd = kgsl_syncobj_ts_to_fd(sync);
            ret.fd = sync_merge_close("tu_sync", ret.fd, sync_fd, true);
            assert(ret.fd >= 0);
         } else {
            ret = *sync;
            ret.fd = os_dupfd_cloexec(ret.fd);
            assert(ret.fd >= 0);
         }
         break;

      default:
         UNREACHABLE("invalid syncobj state");
      }
   }

   return ret;
}

struct vk_kgsl_syncobj
{
   struct vk_sync vk;
   struct kgsl_syncobj syncobj;
};

static VkResult
vk_kgsl_sync_init(struct vk_device *device,
                  struct vk_sync *sync,
                  uint64_t initial_value)
{
   struct vk_kgsl_syncobj *s = container_of(sync, struct vk_kgsl_syncobj, vk);
   kgsl_syncobj_init(&s->syncobj, initial_value != 0);
   return VK_SUCCESS;
}

static void
vk_kgsl_sync_finish(struct vk_device *device, struct vk_sync *sync)
{
   struct vk_kgsl_syncobj *s = container_of(sync, struct vk_kgsl_syncobj, vk);
   kgsl_syncobj_destroy(&s->syncobj);
}

static VkResult
vk_kgsl_sync_reset(struct vk_device *device, struct vk_sync *sync)
{
   struct vk_kgsl_syncobj *s = container_of(sync, struct vk_kgsl_syncobj, vk);
   kgsl_syncobj_reset(&s->syncobj);
   return VK_SUCCESS;
}

static VkResult
vk_kgsl_sync_move(struct vk_device *device,
                  struct vk_sync *dst,
                  struct vk_sync *src)
{
   struct vk_kgsl_syncobj *d = container_of(dst, struct vk_kgsl_syncobj, vk);
   struct vk_kgsl_syncobj *s = container_of(src, struct vk_kgsl_syncobj, vk);
   kgsl_syncobj_reset(&d->syncobj);
   d->syncobj = s->syncobj;
   kgsl_syncobj_init(&s->syncobj, false);
   return VK_SUCCESS;
}

static VkResult
vk_kgsl_sync_wait(struct vk_device *_device,
                  struct vk_sync *sync,
                  uint64_t wait_value,
                  enum vk_sync_wait_flags wait_flags,
                  uint64_t abs_timeout_ns)
{
   struct tu_device *device = container_of(_device, struct tu_device, vk);
   struct vk_kgsl_syncobj *s = container_of(sync, struct vk_kgsl_syncobj, vk);

   if (wait_flags & VK_SYNC_WAIT_PENDING)
      return VK_SUCCESS;

   return kgsl_syncobj_wait(device, &s->syncobj, abs_timeout_ns);
}

static VkResult
vk_kgsl_sync_wait_many(struct vk_device *_device,
                       uint32_t wait_count,
                       const struct vk_sync_wait *waits,
                       enum vk_sync_wait_flags wait_flags,
                       uint64_t abs_timeout_ns)
{
   struct tu_device *device = container_of(_device, struct tu_device, vk);

   if (wait_flags & VK_SYNC_WAIT_PENDING)
      return VK_SUCCESS;

   if (wait_flags & VK_SYNC_WAIT_ANY) {
      struct kgsl_syncobj *syncobjs[wait_count];
      for (uint32_t i = 0; i < wait_count; i++) {
         syncobjs[i] =
            &container_of(waits[i].sync, struct vk_kgsl_syncobj, vk)->syncobj;
      }

      return kgsl_syncobj_wait_any(device, syncobjs, wait_count,
                                   abs_timeout_ns);
   } else {
      for (uint32_t i = 0; i < wait_count; i++) {
         struct vk_kgsl_syncobj *s =
            container_of(waits[i].sync, struct vk_kgsl_syncobj, vk);

         VkResult result =
            kgsl_syncobj_wait(device, &s->syncobj, abs_timeout_ns);
         if (result != VK_SUCCESS)
            return result;
      }
      return VK_SUCCESS;
   }
}

static VkResult
vk_kgsl_sync_import_sync_file(struct vk_device *device,
                              struct vk_sync *sync,
                              int fd)
{
   struct vk_kgsl_syncobj *s = container_of(sync, struct vk_kgsl_syncobj, vk);
   if (fd >= 0) {
      fd = os_dupfd_cloexec(fd);
      if (fd < 0) {
         mesa_loge("vk_kgsl_sync_import_sync_file: dup failed: %s",
                   strerror(errno));
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }
   return kgsl_syncobj_import(&s->syncobj, fd);
}

static VkResult
vk_kgsl_sync_export_sync_file(struct vk_device *device,
                              struct vk_sync *sync,
                              int *pFd)
{
   struct vk_kgsl_syncobj *s = container_of(sync, struct vk_kgsl_syncobj, vk);
   return kgsl_syncobj_export(&s->syncobj, pFd);
}

const struct vk_sync_type vk_kgsl_sync_type = {
   .size = sizeof(struct vk_kgsl_syncobj),
   .features = (enum vk_sync_features)
               (VK_SYNC_FEATURE_BINARY |
                VK_SYNC_FEATURE_GPU_WAIT |
                VK_SYNC_FEATURE_GPU_MULTI_WAIT |
                VK_SYNC_FEATURE_CPU_WAIT |
                VK_SYNC_FEATURE_CPU_RESET |
                VK_SYNC_FEATURE_WAIT_ANY |
                VK_SYNC_FEATURE_WAIT_PENDING),
   .init = vk_kgsl_sync_init,
   .finish = vk_kgsl_sync_finish,
   .reset = vk_kgsl_sync_reset,
   .move = vk_kgsl_sync_move,
   .wait = vk_kgsl_sync_wait,
   .wait_many = vk_kgsl_sync_wait_many,
   .import_sync_file = vk_kgsl_sync_import_sync_file,
   .export_sync_file = vk_kgsl_sync_export_sync_file,
};

struct tu_kgsl_queue_submit {
   struct util_dynarray commands;
};

static void *
kgsl_submit_create(struct tu_device *device)
{
   return vk_zalloc(&device->vk.alloc, sizeof(struct tu_kgsl_queue_submit), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
}

static void
kgsl_submit_finish(struct tu_device *device,
                   void *_submit)
{
   struct tu_kgsl_queue_submit *submit =
      (struct tu_kgsl_queue_submit *)_submit;

   util_dynarray_fini(&submit->commands);
   vk_free(&device->vk.alloc, submit);
}

static void
kgsl_submit_add_entries(struct tu_device *device, void *_submit,
                        struct tu_cs_entry *entries, unsigned num_entries)
{
   struct tu_kgsl_queue_submit *submit =
      (struct tu_kgsl_queue_submit *)_submit;

   struct kgsl_command_object *cmds = (struct kgsl_command_object *)
      util_dynarray_grow(&submit->commands, struct kgsl_command_object,
                      num_entries);

   for (unsigned i = 0; i < num_entries; i++) {
      cmds[i] = (struct kgsl_command_object) {
         .gpuaddr = entries[i].bo->iova + entries[i].offset,
         .size = entries[i].size,
         .flags = KGSL_CMDLIST_IB,
         .id = entries[i].bo->gem_handle,
      };
   }
}

static VkResult
kgsl_queue_submit(struct tu_queue *queue, void *_submit,
                  struct vk_sync_wait *waits, uint32_t wait_count,
                  struct vk_sync_signal *signals, uint32_t signal_count,
                  struct tu_u_trace_submission_data *u_trace_submission_data)
{
   struct tu_kgsl_queue_submit *submit =
      (struct tu_kgsl_queue_submit *)_submit;

#if HAVE_PERFETTO
   uint64_t start_ts = tu_perfetto_begin_submit();
#endif

   if (submit->commands.size == 0) {
      /* This handles the case where we have a wait and no commands to submit.
       * It is necessary to handle this case separately as the kernel will not
       * advance the GPU timeline if a submit with no commands is made, even
       * though it will return an incremented fence timestamp (which will
       * never be signaled).
       */
      const struct kgsl_syncobj *wait_semaphores[wait_count + 1];
      for (uint32_t i = 0; i < wait_count; i++) {
         wait_semaphores[i] = &container_of(waits[i].sync,
                                            struct vk_kgsl_syncobj, vk)
                                  ->syncobj;
      }

      struct kgsl_syncobj last_submit_sync;
      if (queue->fence >= 0)
         last_submit_sync = (struct kgsl_syncobj) {
            .state = KGSL_SYNCOBJ_STATE_TS,
            .queue = queue,
            .timestamp = queue->fence,
         };
      else
         last_submit_sync = (struct kgsl_syncobj) {
            .state = KGSL_SYNCOBJ_STATE_SIGNALED,
         };

      wait_semaphores[wait_count] = &last_submit_sync;

      struct kgsl_syncobj wait_sync =
         kgsl_syncobj_merge(wait_semaphores, wait_count + 1);
      assert(wait_sync.state !=
             KGSL_SYNCOBJ_STATE_UNSIGNALED); // Would wait forever

      if (signal_count == 1) {
         /* Move instead of duplicating the syncobj, as we don't need to
          * keep the original one around.
          */
         struct kgsl_syncobj *signal_sync =
            &container_of(signals[0].sync, struct vk_kgsl_syncobj, vk)
                ->syncobj;

         kgsl_syncobj_reset(signal_sync);
         *signal_sync = wait_sync;
      } else {
         for (uint32_t i = 0; i < signal_count; i++) {
            struct kgsl_syncobj *signal_sync =
               &container_of(signals[i].sync, struct vk_kgsl_syncobj, vk)
                   ->syncobj;

            kgsl_syncobj_reset(signal_sync);
            *signal_sync = kgsl_syncobj_dup(&wait_sync);
         }

         kgsl_syncobj_destroy(&wait_sync);
      }

      kgsl_syncobj_destroy(&last_submit_sync);

      return VK_SUCCESS;
   }

   VkResult result = VK_SUCCESS;

   if (u_trace_submission_data) {
      mtx_lock(&queue->device->kgsl_profiling_mutex);
      tu_suballoc_bo_alloc(&u_trace_submission_data->kgsl_timestamp_bo,
                           &queue->device->kgsl_profiling_suballoc,
                           sizeof(struct kgsl_cmdbatch_profiling_buffer), 4);
      mtx_unlock(&queue->device->kgsl_profiling_mutex);
   }

   uint32_t obj_count = 0;
   if (u_trace_submission_data)
      obj_count++;

   struct kgsl_command_object *objs = (struct kgsl_command_object *)
      vk_alloc(&queue->device->vk.alloc, sizeof(*objs) * obj_count,
               alignof(*objs), VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);

   struct kgsl_cmdbatch_profiling_buffer *profiling_buffer = NULL;
   uint32_t obj_idx = 0;
   if (u_trace_submission_data) {
      struct tu_suballoc_bo *bo = &u_trace_submission_data->kgsl_timestamp_bo;

      objs[obj_idx++] = (struct kgsl_command_object) {
         .offset = bo->iova - bo->bo->iova,
         .gpuaddr = bo->bo->iova,
         .size = sizeof(struct kgsl_cmdbatch_profiling_buffer),
         .flags = KGSL_OBJLIST_MEMOBJ | KGSL_OBJLIST_PROFILE,
         .id = bo->bo->gem_handle,
      };
      profiling_buffer =
         (struct kgsl_cmdbatch_profiling_buffer *) tu_suballoc_bo_map(bo);
      memset(profiling_buffer, 0, sizeof(*profiling_buffer));
   }

   const struct kgsl_syncobj *wait_semaphores[wait_count];
   for (uint32_t i = 0; i < wait_count; i++) {
      wait_semaphores[i] =
         &container_of(waits[i].sync, struct vk_kgsl_syncobj, vk)
             ->syncobj;
   }

   struct kgsl_syncobj wait_sync =
      kgsl_syncobj_merge(wait_semaphores, wait_count);
   assert(wait_sync.state !=
          KGSL_SYNCOBJ_STATE_UNSIGNALED); // Would wait forever

   struct kgsl_cmd_syncpoint_timestamp ts;
   struct kgsl_cmd_syncpoint_fence fn;
   struct kgsl_command_syncpoint sync = { 0 };
   bool has_sync = false;
   switch (wait_sync.state) {
   case KGSL_SYNCOBJ_STATE_SIGNALED:
      break;

   case KGSL_SYNCOBJ_STATE_TS:
      ts.context_id = wait_sync.queue->msm_queue_id;
      ts.timestamp = wait_sync.timestamp;

      has_sync = true;
      sync.type = KGSL_CMD_SYNCPOINT_TYPE_TIMESTAMP;
      sync.priv = (uintptr_t) &ts;
      sync.size = sizeof(ts);
      break;

   case KGSL_SYNCOBJ_STATE_FD:
      fn.fd = wait_sync.fd;

      has_sync = true;
      sync.type = KGSL_CMD_SYNCPOINT_TYPE_FENCE;
      sync.priv = (uintptr_t) &fn;
      sync.size = sizeof(fn);
      break;

   default:
      UNREACHABLE("invalid syncobj state");
   }

   struct kgsl_gpu_command req = {
      .flags = KGSL_CMDBATCH_SUBMIT_IB_LIST,
      .cmdlist = (uintptr_t) submit->commands.data,
      .cmdsize = sizeof(struct kgsl_command_object),
      .numcmds = util_dynarray_num_elements(&submit->commands,
                                            struct kgsl_command_object),
      .synclist = (uintptr_t) &sync,
      .syncsize = sizeof(sync),
      .numsyncs = has_sync != 0 ? 1 : 0,
      .context_id = queue->msm_queue_id,
   };

   if (obj_idx) {
      req.flags |= KGSL_CMDBATCH_PROFILING;
      req.objlist = (uintptr_t) objs;
      req.objsize = sizeof(struct kgsl_command_object);
      req.numobjs = obj_idx;
   }

   int ret = safe_ioctl(queue->device->physical_device->local_fd,
                        IOCTL_KGSL_GPU_COMMAND, &req);

   uint64_t gpu_offset = 0;
#if HAVE_PERFETTO
   if (profiling_buffer) {
      /* We need to wait for KGSL to queue the GPU command before we can read
       * the timestamp. Since this is just for profiling and doesn't take too
       * long, we can just busy-wait for it.
       */
      while (p_atomic_read(&profiling_buffer->gpu_ticks_queued) == 0);

      struct kgsl_perfcounter_read_group perf = {
         .groupid = KGSL_PERFCOUNTER_GROUP_ALWAYSON,
         .countable = 0,
         .value = 0
      };

      struct kgsl_perfcounter_read req = {
         .reads = &perf,
         .count = 1,
      };

      ret = safe_ioctl(queue->device->fd, IOCTL_KGSL_PERFCOUNTER_READ, &req);
      /* Older KGSL has some kind of garbage in upper 32 bits */
      uint64_t offseted_gpu_ts = perf.value & 0xffffffff;

      gpu_offset = tu_device_ticks_to_ns(
         queue->device, offseted_gpu_ts - profiling_buffer->gpu_ticks_queued);

      struct tu_perfetto_clocks clocks = {
         .cpu = profiling_buffer->wall_clock_ns,
         .gpu_ts = tu_device_ticks_to_ns(queue->device,
                                         profiling_buffer->gpu_ticks_queued),
         .gpu_ts_offset = gpu_offset,
      };

      clocks = tu_perfetto_end_submit(queue, queue->device->submit_count,
                                      start_ts, &clocks);
      gpu_offset = clocks.gpu_ts_offset;
   }
#endif

   kgsl_syncobj_destroy(&wait_sync);

   if (ret) {
      result = vk_device_set_lost(&queue->device->vk, "submit failed: %s\n",
                                  strerror(errno));
      goto fail_submit;
   }

   p_atomic_set(&queue->fence, req.timestamp);

   for (uint32_t i = 0; i < signal_count; i++) {
      struct kgsl_syncobj *signal_sync =
         &container_of(signals[i].sync, struct vk_kgsl_syncobj, vk)
             ->syncobj;

      kgsl_syncobj_reset(signal_sync);
      signal_sync->state = KGSL_SYNCOBJ_STATE_TS;
      signal_sync->queue = queue;
      signal_sync->timestamp = req.timestamp;
   }

   if (u_trace_submission_data) {
      struct tu_u_trace_submission_data *submission_data =
         u_trace_submission_data;
      submission_data->gpu_ts_offset = gpu_offset;
   }

fail_submit:
   if (result != VK_SUCCESS && u_trace_submission_data) {
      mtx_lock(&queue->device->kgsl_profiling_mutex);
      tu_suballoc_bo_free(&queue->device->kgsl_profiling_suballoc,
                          &u_trace_submission_data->kgsl_timestamp_bo);
      mtx_unlock(&queue->device->kgsl_profiling_mutex);
   }

   return result;
}

static VkResult
kgsl_device_init(struct tu_device *dev)
{
   dev->fd = dev->physical_device->local_fd;
   return VK_SUCCESS;
}

static void
kgsl_device_finish(struct tu_device *dev)
{
   /* No-op */
}

static int
kgsl_device_get_gpu_timestamp(struct tu_device *dev, uint64_t *ts)
{
   UNREACHABLE("");
   return 0;
}

static int
kgsl_device_get_suspend_count(struct tu_device *dev, uint64_t *suspend_count)
{
   /* kgsl doesn't have a way to get it */
   *suspend_count = 0;
   return 0;
}

static VkResult
kgsl_device_check_status(struct tu_device *device)
{
   for (unsigned i = 0; i < TU_MAX_QUEUE_FAMILIES; i++) {
      for (unsigned q = 0; q < device->queue_count[i]; q++) {
         /* KGSL's KGSL_PROP_GPU_RESET_STAT takes the u32 msm_queue_id and returns a
         * KGSL_CTX_STAT_* for the worst reset that happened since the last time it
         * was queried on that queue.
         */
         uint32_t value = device->queues[i][q].msm_queue_id;
         VkResult status = get_kgsl_prop(device->fd, KGSL_PROP_GPU_RESET_STAT,
                                       &value, sizeof(value));
         if (status != VK_SUCCESS)
            return vk_device_set_lost(&device->vk, "Failed to get GPU reset status");

         if (value != KGSL_CTX_STAT_NO_ERROR &&
            value != KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT) {
            return vk_device_set_lost(&device->vk, "GPU faulted or hung");
         }
      }
   }

   return VK_SUCCESS;
}

static const struct tu_knl kgsl_knl_funcs = {
      .name = "kgsl",

      .device_init = kgsl_device_init,
      .device_finish = kgsl_device_finish,
      .device_get_gpu_timestamp = kgsl_device_get_gpu_timestamp,
      .device_get_suspend_count = kgsl_device_get_suspend_count,
      .device_check_status = kgsl_device_check_status,
      .submitqueue_new = kgsl_submitqueue_new,
      .submitqueue_close = kgsl_submitqueue_close,
      .bo_init = kgsl_bo_init,
      .bo_init_dmabuf = kgsl_bo_init_dmabuf,
      .bo_export_dmabuf = kgsl_bo_export_dmabuf,
      .bo_map = kgsl_bo_map,
      .bo_allow_dump = kgsl_bo_allow_dump,
      .bo_finish = kgsl_bo_finish,
      .submit_create = kgsl_submit_create,
      .submit_finish = kgsl_submit_finish,
      .submit_add_entries = kgsl_submit_add_entries,
      .queue_submit = kgsl_queue_submit,
      .queue_wait_fence = kgsl_queue_wait_fence,
};

static bool
tu_kgsl_get_raytracing(int fd)
{
   uint32_t value;
   int ret = get_kgsl_prop(fd, KGSL_PROP_IS_RAYTRACING_ENABLED, &value, sizeof(value));
   if (ret)
      return false;

   return value;
}

VkResult
tu_knl_kgsl_load(struct tu_instance *instance, int fd)
{
   if (instance->vk.enabled_extensions.KHR_display) {
      return vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                       "I can't KHR_display");
   }

   struct tu_physical_device *device = (struct tu_physical_device *)
      vk_zalloc(&instance->vk.alloc, sizeof(*device), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!device) {
      close(fd);
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   static const char dma_heap_path[] = "/dev/dma_heap/system";
   static const char ion_path[] = "/dev/ion";
   int dma_fd;

   dma_fd = open(dma_heap_path, O_RDONLY);
   if (dma_fd >= 0) {
      device->kgsl_dma_type = TU_KGSL_DMA_TYPE_DMAHEAP;
   } else {
      dma_fd = open(ion_path, O_RDONLY);
      if (dma_fd >= 0) {
         /* ION_IOC_FREE available only for legacy ION */
         struct ion_handle_data free = { .handle = 0 };
         if (safe_ioctl(dma_fd, ION_IOC_FREE, &free) >= 0 || errno != ENOTTY)
            device->kgsl_dma_type = TU_KGSL_DMA_TYPE_ION_LEGACY;
         else
            device->kgsl_dma_type = TU_KGSL_DMA_TYPE_ION;
      } else {
         mesa_logw(
            "Unable to open neither %s nor %s, VK_KHR_external_memory_fd would be "
            "unavailable: %s",
            dma_heap_path, ion_path, strerror(errno));
      }
   }

   VkResult result = VK_ERROR_INITIALIZATION_FAILED;

   struct kgsl_devinfo info;
   if (get_kgsl_prop(fd, KGSL_PROP_DEVICE_INFO, &info, sizeof(info)))
      goto fail;

   uint64_t gmem_iova;
   if (get_kgsl_prop(fd, KGSL_PROP_UCHE_GMEM_VADDR, &gmem_iova, sizeof(gmem_iova)))
      goto fail;

   uint32_t highest_bank_bit;
   if (get_kgsl_prop(fd, KGSL_PROP_HIGHEST_BANK_BIT, &highest_bank_bit,
                     sizeof(highest_bank_bit)))
      goto fail;

   uint32_t ubwc_version;
   if (get_kgsl_prop(fd, KGSL_PROP_UBWC_MODE, &ubwc_version,
                     sizeof(ubwc_version)))
      goto fail;

   if (get_kgsl_prop(fd, KGSL_PROP_UCHE_TRAP_BASE, &device->uche_trap_base,
                     sizeof(device->uche_trap_base))) {
      /* It is known to be hardcoded to */
      device->uche_trap_base = 0x1fffffffff000ull;
   }

   /* kgsl version check? */

   device->instance = instance;
   device->master_fd = -1;
   device->local_fd = fd;
   device->kgsl_dma_fd = dma_fd;

   device->dev_id.gpu_id =
      ((info.chip_id >> 24) & 0xff) * 100 +
      ((info.chip_id >> 16) & 0xff) * 10 +
      ((info.chip_id >>  8) & 0xff);
   device->dev_id.chip_id = info.chip_id;
   device->gmem_size = debug_get_num_option("TU_GMEM", info.gmem_sizebytes);
   device->gmem_base = gmem_iova;

   device->has_raytracing = tu_kgsl_get_raytracing(fd);

   device->submitqueue_priority_count = 1;
   
   device->timeline_type = vk_sync_timeline_get_type(&vk_kgsl_sync_type);

   device->sync_types[0] = &vk_kgsl_sync_type;
   device->sync_types[1] = &device->timeline_type.sync;
   device->sync_types[2] = NULL;

   device->heap.size = tu_get_system_heap_size(device);
   device->heap.used = 0u;
   device->heap.flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;

   device->has_set_iova = kgsl_is_memory_type_supported(
      fd, KGSL_MEMFLAGS_USE_CPU_MAP);

   /* Even if kernel is new enough, the GPU itself may not support it. */
   device->has_cached_coherent_memory = kgsl_is_memory_type_supported(
      fd, KGSL_MEMFLAGS_IOCOHERENT |
             (KGSL_CACHEMODE_WRITEBACK << KGSL_CACHEMODE_SHIFT));

   /* preemption is always supported on kgsl */
   device->has_preemption = true;

   device->ubwc_config.highest_bank_bit = highest_bank_bit;

   /* The other config values can be partially inferred from the UBWC version,
    * but kgsl also hardcodes overrides for specific a6xx versions that we
    * have to follow here. Yuck.
    */
   switch (ubwc_version) {
   case KGSL_UBWC_1_0:
      device->ubwc_config.bank_swizzle_levels = 0x7;
      device->ubwc_config.macrotile_mode = FDL_MACROTILE_4_CHANNEL;
      break;
   case KGSL_UBWC_2_0:
      device->ubwc_config.bank_swizzle_levels = 0x6;
      device->ubwc_config.macrotile_mode = FDL_MACROTILE_4_CHANNEL;
      break;
   case KGSL_UBWC_3_0:
      device->ubwc_config.bank_swizzle_levels = 0x6;
      device->ubwc_config.macrotile_mode = FDL_MACROTILE_4_CHANNEL;
      break;
   case KGSL_UBWC_4_0:
      device->ubwc_config.bank_swizzle_levels = 0x6;
      device->ubwc_config.macrotile_mode = FDL_MACROTILE_8_CHANNEL;
      break;
   default:
      return vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                       "unknown UBWC version 0x%x", ubwc_version);
   }

   /* kgsl unfortunately hardcodes some settings for certain GPUs and doesn't
    * expose them in the uAPI so hardcode them here to match.
    */
   if (device->dev_id.gpu_id == 663 || device->dev_id.gpu_id == 680) {
      device->ubwc_config.macrotile_mode = FDL_MACROTILE_8_CHANNEL;
   }
   if (device->dev_id.gpu_id == 663) {
      /* level2_swizzling_dis = 1 */
      device->ubwc_config.bank_swizzle_levels = 0x4;
   }

   instance->knl = &kgsl_knl_funcs;

   result = tu_physical_device_init(device, instance);
   if (result != VK_SUCCESS)
      goto fail;

   list_addtail(&device->vk.link, &instance->vk.physical_devices.list);

   return VK_SUCCESS;

fail:
   vk_free(&instance->vk.alloc, device);
   close(fd);
   if (dma_fd >= 0)
      close(dma_fd);
   return result;
}
