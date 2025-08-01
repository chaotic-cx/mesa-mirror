/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * SPDX-License-Identifier: MIT
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 */

#ifndef TU_DEVICE_H
#define TU_DEVICE_H

#include "tu_common.h"

#include "vk_device_memory.h"
#include "vk_meta.h"

#include "tu_autotune.h"
#include "tu_cs.h"
#include "tu_pass.h"
#include "tu_perfetto.h"
#include "tu_suballoc.h"
#include "tu_util.h"

#include "radix_sort/radix_sort_vk.h"

#include "common/freedreno_rd_output.h"
#include "util/vma.h"
#include "util/u_vector.h"

/* queue types */
#define TU_QUEUE_GENERAL 0

#define TU_MAX_QUEUE_FAMILIES 1

#define TU_BORDER_COLOR_COUNT 4096

#define TU_BLIT_SHADER_SIZE 4096

/* extra space in vsc draw/prim streams */
#define VSC_PAD 0x40

enum global_shader {
   GLOBAL_SH_VS_BLIT,
   GLOBAL_SH_VS_CLEAR,
   GLOBAL_SH_FS_BLIT,
   GLOBAL_SH_FS_BLIT_ZSCALE,
   GLOBAL_SH_FS_COPY_MS,
   GLOBAL_SH_FS_CLEAR0,
   GLOBAL_SH_FS_CLEAR_MAX = GLOBAL_SH_FS_CLEAR0 + MAX_RTS,
   GLOBAL_SH_COUNT,
};

struct tu_memory_heap {
   /* Standard bits passed on to the client */
   VkDeviceSize      size;
   VkMemoryHeapFlags flags;

   /** Copied from ANV:
    *
    * Driver-internal book-keeping.
    *
    * Align it to 64 bits to make atomic operations faster on 32 bit platforms.
    */
   alignas(8) VkDeviceSize used;
};

enum tu_kgsl_dma_type
{
   TU_KGSL_DMA_TYPE_ION_LEGACY,
   TU_KGSL_DMA_TYPE_ION,
   TU_KGSL_DMA_TYPE_DMAHEAP,
};

extern uint64_t os_page_size;

struct tu_physical_device
{
   struct vk_physical_device vk;

   struct tu_instance *instance;

   const char *name;
   uint8_t driver_uuid[VK_UUID_SIZE];
   uint8_t device_uuid[VK_UUID_SIZE];
   uint8_t cache_uuid[VK_UUID_SIZE];

   struct wsi_device wsi_device;

   char fd_path[20];
   int local_fd;
   bool has_local;
   int64_t local_major;
   int64_t local_minor;
   int master_fd;
   bool has_master;
   int64_t master_major;
   int64_t master_minor;

   int kgsl_dma_fd;
   enum tu_kgsl_dma_type kgsl_dma_type;

   uint32_t gmem_size;
   uint64_t gmem_base;

   uint32_t usable_gmem_size_gmem;
   uint32_t ccu_offset_gmem;
   uint32_t ccu_offset_bypass;
   uint32_t ccu_depth_offset_bypass;
   uint32_t vpc_attr_buf_offset_gmem;
   uint32_t vpc_attr_buf_size_gmem;
   uint32_t vpc_attr_buf_offset_bypass;
   uint32_t vpc_attr_buf_size_bypass;

   uint64_t uche_trap_base;

   /* Amount of usable descriptor sets, this excludes any reserved set */
   uint32_t usable_sets;
   /* Index of the reserved descriptor set, may be -1 if unset */
   int32_t reserved_set_idx;

   bool has_set_iova;
   bool has_raytracing;
   uint64_t va_start;
   uint64_t va_size;

   bool has_cached_coherent_memory;
   bool has_cached_non_coherent_memory;
   uintptr_t level1_dcache_size;

   struct fdl_ubwc_config ubwc_config;

   bool has_preemption;

   struct {
      uint32_t type_count;
      VkMemoryPropertyFlags types[VK_MAX_MEMORY_TYPES];
   } memory;

   struct fd_dev_id dev_id;
   struct fd_dev_info dev_info;
   const struct fd_dev_info *info;

   int msm_major_version;
   int msm_minor_version;

   /* with 0 being the highest priority */
   uint32_t submitqueue_priority_count;

   struct tu_memory_heap heap;

   struct vk_sync_type syncobj_type;
   struct vk_sync_timeline_type timeline_type;
   const struct vk_sync_type *sync_types[3];

   uint32_t device_count;
};
VK_DEFINE_HANDLE_CASTS(tu_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

struct tu_knl;

struct tu_instance
{
   struct vk_instance vk;

   const struct tu_knl *knl;

   uint32_t instance_idx;
   uint32_t api_version;

   struct driOptionCache dri_options;
   struct driOptionCache available_dri_options;

   bool dont_care_as_load;

   /* Conservative LRZ (default true) invalidates LRZ on draws with
    * blend and depth-write enabled, because this can lead to incorrect
    * rendering.  Driconf can be used to disable conservative LRZ for
    * games which do not have the problematic sequence of draws *and*
    * suffer a performance loss with conservative LRZ.
    */
   bool conservative_lrz;

   /* If to internally reserve a descriptor set for descriptor set
    * dynamic offsets, a descriptor set can be freed at the cost of
    * being unable to use the feature. As it is a part of the Vulkan
    * core, this is enabled by default.
    */
   bool reserve_descriptor_set;

   /* Allow out of bounds UBO access by disabling lowering of UBO loads for
    * indirect access, which rely on the UBO bounds specified in the shader,
    * rather than the bound UBO size which isn't known until draw time.
    *
    * See: https://github.com/doitsujin/dxvk/issues/3861
    */
   bool allow_oob_indirect_ubo_loads;

   /* DXVK and VKD3D-Proton use customBorderColorWithoutFormat
    * and have most of D24S8 images with USAGE_SAMPLED, in such case we
    * disable UBWC for correctness. However, games don't use border color for
    * depth-stencil images. So we elect to ignore this edge case and force
    * UBWC to be enabled.
    */
   bool disable_d24s8_border_color_workaround;

   /* D3D emulation requires texture coordinates to be rounded to nearest even value. */
   bool use_tex_coord_round_nearest_even_mode;

   /* Apps may be accidentally incorrect  */
   bool ignore_frag_depth_direction;
};
VK_DEFINE_HANDLE_CASTS(tu_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)

/* This struct defines the layout of the global_bo */
struct tu6_global
{
   /* clear/blit shaders */
   uint32_t shaders[TU_BLIT_SHADER_SIZE];

   uint32_t seqno_dummy;          /* dummy seqno for CP_EVENT_WRITE */
   uint32_t _pad0;
   volatile uint32_t vsc_draw_overflow;
   uint32_t _pad1;
   volatile uint32_t vsc_prim_overflow;
   uint32_t _pad2;
   uint64_t predicate;

   /* scratch space for VPC_SO[i].FLUSH_BASE_LO/HI, start on 32 byte boundary. */
   struct {
      uint32_t offset;
      uint32_t pad[7];
   } flush_base[4];

   alignas(16) uint32_t cs_indirect_xyz[12];

   uint32_t vsc_state[32];

   volatile uint32_t vtx_stats_query_not_running;

   /* To know when renderpass stats for autotune are valid */
   volatile uint32_t autotune_fence;

   /* For recycling command buffers for dynamic suspend/resume comamnds */
   volatile uint32_t dynamic_rendering_fence;

   volatile uint32_t dbg_one;
   volatile uint32_t dbg_gmem_total_loads;
   volatile uint32_t dbg_gmem_taken_loads;
   volatile uint32_t dbg_gmem_total_stores;
   volatile uint32_t dbg_gmem_taken_stores;

   /* Written from GPU */
   volatile uint32_t breadcrumb_gpu_sync_seqno;
   uint32_t _pad3;
   /* Written from CPU, acknowledges value written from GPU */
   volatile uint32_t breadcrumb_cpu_sync_seqno;
   uint32_t _pad4;

   volatile uint32_t userspace_fence;
   uint32_t _pad5;

   struct bcolor_entry bcolor[];
};
#define gb_offset(member) offsetof(struct tu6_global, member)
#define global_iova(cmd, member) ((cmd)->device->global_bo->iova + gb_offset(member))
#define global_iova_arr(cmd, member, idx)                                    \
   (global_iova(cmd, member) + sizeof_field(struct tu6_global, member[0]) * (idx))

struct tu_pvtmem_bo {
      mtx_t mtx;
      struct tu_bo *bo;
      uint32_t per_fiber_size, per_sp_size;
};

struct tu_virtio_device;
struct tu_queue;

struct tu_device
{
   struct vk_device vk;
   struct tu_instance *instance;

   struct tu_queue *queues[TU_MAX_QUEUE_FAMILIES];
   int queue_count[TU_MAX_QUEUE_FAMILIES];

   struct tu_physical_device *physical_device;
   uint32_t device_idx;
   int fd;

   struct ir3_compiler *compiler;

   /* Backup in-memory cache to be used if the app doesn't provide one */
   struct vk_pipeline_cache *mem_cache;

   struct vk_meta_device meta;

   radix_sort_vk_t *radix_sort;
   mtx_t radix_sort_mutex;

   struct util_sparse_array accel_struct_ranges;

#define MIN_SCRATCH_BO_SIZE_LOG2 12 /* A page */

   /* Currently the kernel driver uses a 32-bit GPU address space, but it
    * should be impossible to go beyond 48 bits.
    */
   struct {
      struct tu_bo *bo;
      mtx_t construct_mtx;
      bool initialized;
   } scratch_bos[48 - MIN_SCRATCH_BO_SIZE_LOG2];

   struct tu_pvtmem_bo fiber_pvtmem_bo, wave_pvtmem_bo;

   struct tu_bo *global_bo;
   struct tu6_global *global_bo_map;

   struct tu_bo *null_accel_struct_bo;

   uint32_t implicit_sync_bo_count;

   /* Device-global BO suballocator for reducing BO management overhead for
    * (read-only) pipeline state.  Synchronized by pipeline_mutex.
    */
   struct tu_suballocator pipeline_suballoc;
   mtx_t pipeline_mutex;

   /* Device-global BO suballocator for reducing BO management for small
    * gmem/sysmem autotune result buffers.  Synchronized by autotune_mutex.
    */
   struct tu_suballocator autotune_suballoc;
   mtx_t autotune_mutex;

   /* KGSL requires a small chunk of GPU mem to retrieve raw GPU time on
    * each submission.
    */
   struct tu_suballocator kgsl_profiling_suballoc;
   mtx_t kgsl_profiling_mutex;

   /* VkEvent BO suballocator.  Synchronized by event_mutex.
    */
   struct tu_suballocator event_suballoc;
   mtx_t event_mutex;

   struct tu_suballocator *trace_suballoc;
   mtx_t trace_mutex;

   /* the blob seems to always use 8K factor and 128K param sizes, copy them */
#define TU_TESS_FACTOR_SIZE (8 * 1024)
#define TU_TESS_PARAM_SIZE (128 * 1024)
#define TU_TESS_BO_SIZE (TU_TESS_FACTOR_SIZE + TU_TESS_PARAM_SIZE)
   /* Lazily allocated, protected by the device mutex. */
   struct tu_bo *tess_bo;

   struct ir3_shader_variant *global_shader_variants[GLOBAL_SH_COUNT];
   struct ir3_shader *global_shaders[GLOBAL_SH_COUNT];
   uint64_t global_shader_va[GLOBAL_SH_COUNT];

   struct tu_shader *empty_tcs, *empty_tes, *empty_gs, *empty_fs, *empty_fs_fdm;

   uint32_t vsc_draw_strm_pitch;
   uint32_t vsc_prim_strm_pitch;
   BITSET_DECLARE(custom_border_color, TU_BORDER_COLOR_COUNT);
   mtx_t mutex;

   mtx_t vma_mutex;
   struct util_vma_heap vma;

   /* bo list for submits: */
   struct drm_msm_gem_submit_bo *submit_bo_list;
   /* map bo handles to bo list index: */
   uint32_t submit_bo_count, submit_bo_list_size;
   /* bo list for dumping: */
   struct util_dynarray dump_bo_list;
   mtx_t bo_mutex;
   /* protects imported BOs creation/freeing */
   struct u_rwlock dma_bo_lock;

   /* Tracking of name -> size allocated for TU_DEBUG_BOS */
   struct hash_table *bo_sizes;

   /* This array holds all our 'struct tu_bo' allocations. We use this
    * so we can add a refcount to our BOs and check if a particular BO
    * was already allocated in this device using its GEM handle. This is
    * necessary to properly manage BO imports, because the kernel doesn't
    * refcount the underlying BO memory.
    *
    * Specifically, when self-importing (i.e. importing a BO into the same
    * device that created it), the kernel will give us the same BO handle
    * for both BOs and we must only free it once when  both references are
    * freed. Otherwise, if we are not self-importing, we get two different BO
    * handles, and we want to free each one individually.
    *
    * The refcount is also useful for being able to maintain BOs across
    * VK object lifetimes, such as pipelines suballocating out of BOs
    * allocated on the device.
    */
   struct util_sparse_array bo_map;

   /* We cannot immediately free VMA when freeing BO, kernel truly
    * frees BO when it stops being busy.
    * So we have to free our VMA only after the kernel does it.
    */
   struct u_vector zombie_vmas;

   struct tu_cs sub_cs;

   /* Command streams to set pass index to a scratch reg */
   struct tu_cs_entry *perfcntrs_pass_cs_entries;

   struct tu_cs_entry cmdbuf_start_a725_quirk_entry;

   struct tu_cs_entry bin_preamble_entry;

   struct util_dynarray dynamic_rendering_pending;
   VkCommandPool dynamic_rendering_pool;
   uint32_t dynamic_rendering_fence;

   /* Condition variable for timeline semaphore to notify waiters when a
    * new submit is executed. */
   pthread_cond_t timeline_cond;
   pthread_mutex_t submit_mutex;

   struct tu_autotune autotune;

   struct breadcrumbs_context *breadcrumbs_ctx;

   struct tu_cs *dbg_cmdbuf_stomp_cs;
   struct tu_cs *dbg_renderpass_stomp_cs;

#ifdef TU_HAS_VIRTIO
   struct tu_virtio_device *vdev;
#endif

   uint32_t submit_count;

   /* Address space and global fault count for this local_fd with DRM backend */
   uint64_t fault_count;

   struct u_trace_context trace_context;
   struct list_head copy_timestamp_cs_pool;

   #ifdef HAVE_PERFETTO
   struct tu_perfetto_state perfetto;
   #endif

   bool use_z24uint_s8uint;
   bool use_lrz;

   struct fd_rd_output rd_output;
};
VK_DEFINE_HANDLE_CASTS(tu_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

struct tu_device_memory
{
   struct vk_device_memory vk;

   struct tu_bo *bo;

   /* for dedicated allocations */
   struct tu_image *image;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(tu_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

struct tu_attachment_info
{
   struct tu_image_view *attachment;
};

struct tu_vsc_config {
   /* number of tiles */
   VkExtent2D tile_count;
   /* size of the first VSC pipe */
   VkExtent2D pipe0;
   /* number of VSC pipes */
   VkExtent2D pipe_count;

   /* Whether binning could be used for gmem rendering using this framebuffer. */
   bool binning_possible;

   /* Whether binning should be used for gmem rendering using this framebuffer. */
   bool binning;

   /* pipe register values */
   uint32_t pipe_config[MAX_VSC_PIPES];
   uint32_t pipe_sizes[MAX_VSC_PIPES];
};

struct tu_tiling_config {
   /* size of the first tile */
   VkExtent2D tile0;

   /* Whether using GMEM is even possible with this configuration */
   bool possible;

   struct tu_vsc_config vsc, fdm_offset_vsc;
};

struct tu_framebuffer
{
   struct vk_object_base base;

   uint32_t width;
   uint32_t height;
   uint32_t layers;

   struct tu_tiling_config tiling[TU_GMEM_LAYOUT_COUNT];

   uint32_t attachment_count;
   struct tu_attachment_info attachments[0];
};
VK_DEFINE_NONDISP_HANDLE_CASTS(tu_framebuffer, base, VkFramebuffer,
                               VK_OBJECT_TYPE_FRAMEBUFFER)

uint64_t
tu_get_system_heap_size(struct tu_physical_device *physical_device);

VkResult
tu_physical_device_init(struct tu_physical_device *device,
                        struct tu_instance *instance);

void
tu_physical_device_get_global_priority_properties(const struct tu_physical_device *pdevice,
                                                  VkQueueFamilyGlobalPriorityPropertiesKHR *props);

uint64_t
tu_device_ticks_to_ns(struct tu_device *dev, uint64_t ts);

static inline struct tu_bo *
tu_device_lookup_bo(struct tu_device *device, uint32_t handle)
{
   return (struct tu_bo *) util_sparse_array_get(&device->bo_map, handle);
}

struct u_trace_context *
tu_device_get_u_trace(struct tu_device *device);

/* Get a scratch bo for use inside a command buffer. This will always return
 * the same bo given the same size or similar sizes, so only one scratch bo
 * can be used at the same time. It's meant for short-lived things where we
 * need to write to some piece of memory, read from it, and then immediately
 * discard it.
 */
VkResult
tu_get_scratch_bo(struct tu_device *dev, uint64_t size, struct tu_bo **bo);

void tu_setup_dynamic_framebuffer(struct tu_cmd_buffer *cmd_buffer,
                                  const VkRenderingInfo *pRenderingInfo);

void
tu_copy_buffer(struct u_trace_context *utctx, void *cmdstream,
               void *ts_from, uint64_t from_offset_B,
               void *ts_to, uint64_t to_offset_B,
               uint64_t size_B);

VkResult
tu_create_copy_timestamp_cs(struct tu_u_trace_submission_data *submission_data,
                            struct tu_cmd_buffer **cmd_buffers,
                            uint32_t cmd_buffer_count,
                            uint32_t trace_chunks_to_copy);

struct tu_copy_timestamp_data {
   struct list_head node;
   struct tu_cs cs;
   struct u_trace trace;
};

/* Data necessary to retrieve timestamps and clean all
 * associated resources afterwards.
 */
struct tu_u_trace_submission_data
{
   uint32_t submission_id;

   /* We have to know when timestamps are available,
    * this queue and fence indicates it.
    */
   struct tu_queue *queue;
   uint32_t fence;

   uint32_t cmd_buffer_count;
   uint32_t last_buffer_with_tracepoints;
   void *mem_ctx;
   struct u_trace **trace_per_cmd_buffer;
   struct tu_copy_timestamp_data *timestamp_copy_data;

   /* GPU time is reset on GPU power cycle and the GPU time
    * offset may change between submissions due to power cycle.
    */
   uint64_t gpu_ts_offset;

   /* KGSL needs a GPU memory to write submission timestamps into */
   struct tu_suballoc_bo kgsl_timestamp_bo;
};

VkResult
tu_u_trace_submission_data_create(
   struct tu_device *device,
   struct tu_cmd_buffer **cmd_buffers,
   uint32_t cmd_buffer_count,
   struct tu_u_trace_submission_data **submission_data);

void
tu_u_trace_submission_data_finish(
   struct tu_device *device,
   struct tu_u_trace_submission_data *submission_data);

const char *
tu_debug_bos_add(struct tu_device *dev, uint64_t size, const char *name);
void
tu_debug_bos_del(struct tu_device *dev, struct tu_bo *bo);
void
tu_debug_bos_print_stats(struct tu_device *dev);

void
tu_dump_bo_init(struct tu_device *dev, struct tu_bo *bo);
void
tu_dump_bo_del(struct tu_device *dev, struct tu_bo *bo);

/* Use cached-coherent when available, for faster CPU readback.
 */
static inline VkResult
tu_bo_init_new_cached(struct tu_device *dev, struct vk_object_base *base,
                      struct tu_bo **out_bo, uint64_t size,
                      enum tu_bo_alloc_flags flags, const char *name)
{
   return tu_bo_init_new_explicit_iova(
      dev, base, out_bo, size, 0,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
         (dev->physical_device->has_cached_coherent_memory ? 
          VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0),
      flags, name);
}


#endif /* TU_DEVICE_H */
