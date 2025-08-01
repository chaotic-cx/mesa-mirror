/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <fcntl.h>

#ifdef MAJOR_IN_SYSMACROS
#include <sys/sysmacros.h>
#endif

#include "vk_log.h"
#include "vk_shader_module.h"

#include "util/disk_cache.h"
#include "util/hex.h"
#include "util/u_debug.h"
#include "radv_android.h"
#include "radv_debug.h"
#include "radv_entrypoints.h"
#include "radv_instance.h"
#include "radv_physical_device.h"
#include "radv_pipeline_rt.h"
#include "radv_video.h"
#include "radv_wsi.h"

#ifdef _WIN32
typedef void *drmDevicePtr;
#include <io.h>
#else
#include <amdgpu.h>
#include "drm-uapi/amdgpu_drm.h"
#include "util/os_drm.h"
#include "winsys/amdgpu/radv_amdgpu_winsys_public.h"
#endif
#include "winsys/null/radv_null_winsys_public.h"
#include "git_sha1.h"

#if AMD_LLVM_AVAILABLE
#include "ac_llvm_util.h"
#endif

#ifdef _WIN32
#define RADV_SUPPORT_CALIBRATED_TIMESTAMPS 0
#else
#define RADV_SUPPORT_CALIBRATED_TIMESTAMPS 1
#endif

static bool
radv_perf_query_supported(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   /* SQTT / SPM interfere with the register states for perf counters, and
    * the code has only been tested on GFX10.3 */
   return pdev->info.gfx_level == GFX10_3 && !(instance->vk.trace_mode & RADV_TRACE_MODE_RGP);
}

static bool
radv_taskmesh_enabled(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   if (instance->debug_flags & RADV_DEBUG_NO_MESH_SHADER)
      return false;

   return pdev->use_ngg && !pdev->use_llvm && pdev->info.gfx_level >= GFX10_3 && radv_compute_queue_enabled(pdev) &&
          pdev->info.has_gang_submit;
}

static bool
radv_transfer_queue_enabled(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   /* Check if the GPU has SDMA support and transfer queues are allowed. */
   if (pdev->info.sdma_ip_version == SDMA_UNKNOWN || !pdev->info.ip[AMD_IP_SDMA].num_queues ||
       !(instance->perftest_flags & RADV_PERFTEST_TRANSFER_QUEUE))
      return false;

   return pdev->info.gfx_level >= GFX9;
}

static bool
radv_video_decode_queue_enabled(const struct radv_physical_device *pdev)
{
   return pdev->video_decode_enabled && pdev->info.ip[pdev->vid_decode_ip].num_queues > 0;
}

static bool
radv_video_encode_queue_enabled(const struct radv_physical_device *pdev)
{
   return pdev->video_encode_enabled && pdev->info.ip[AMD_IP_VCN_ENC].num_queues > 0;
}

bool
radv_compute_queue_enabled(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   return pdev->info.ip[AMD_IP_COMPUTE].num_queues > 0 &&
          (!(instance->debug_flags & RADV_DEBUG_NO_COMPUTE_QUEUE) || !pdev->info.has_graphics);
}

static bool
radv_graphics_queue_enabled(const struct radv_physical_device *pdev)
{
   return pdev->info.ip[AMD_IP_GFX].num_queues > 0;
}

static bool
radv_vrs_attachment_enabled(const struct radv_physical_device *pdev)
{
   return pdev->info.gfx_level >= GFX11 || pdev->use_hiz;
}

static bool
radv_calibrated_timestamps_enabled(const struct radv_physical_device *pdev)
{
   return RADV_SUPPORT_CALIBRATED_TIMESTAMPS && !(pdev->info.family == CHIP_RAVEN || pdev->info.family == CHIP_RAVEN2);
}

static bool
radv_filter_minmax_enabled(const struct radv_physical_device *pdev)
{
   /* Tahiti and Verde only: reduction mode is unsupported due to a bug
    * (it might work sometimes, but that's not enough)
    */
   return !(pdev->info.family == CHIP_TAHITI || pdev->info.family == CHIP_VERDE);
}

static bool
radv_cooperative_matrix_enabled(const struct radv_physical_device *pdev)
{
   return pdev->info.gfx_level >= GFX11 && !pdev->use_llvm;
}

static bool
radv_cooperative_matrix2_nv_enabled(const struct radv_physical_device *pdev)
{
   if (!radv_cooperative_matrix_enabled(pdev))
      return false;

   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   return instance->drirc.cooperative_matrix2_nv;
}

bool
radv_host_image_copy_enabled(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   return pdev->info.gfx_level >= GFX10 && (instance->perftest_flags & RADV_PERFTEST_HIC);
}

bool
radv_enable_rt(const struct radv_physical_device *pdev)
{
   if (!pdev->info.has_image_bvh_intersect_ray && !radv_emulate_rt(pdev))
      return false;

   if (pdev->use_llvm)
      return false;

   return true;
}

bool
radv_emulate_rt(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   if (instance->perftest_flags & RADV_PERFTEST_EMULATE_RT)
      return true;

   /* Do not force emulated RT on GPUs that have native support. */
   return !pdev->info.has_image_bvh_intersect_ray && instance->drirc.emulate_rt;
}

bool
radv_use_bvh8(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   return pdev->info.gfx_level >= GFX12 && !radv_emulate_rt(pdev) && !(instance->debug_flags & RADV_DEBUG_BVH4);
}

static void
parse_hex(char *out, const char *in, unsigned length)
{
   for (unsigned i = 0; i < length; ++i)
      out[i] = 0;

   for (unsigned i = 0; i < 2 * length; ++i) {
      unsigned v = in[i] <= '9' ? in[i] - '0' : (in[i] >= 'a' ? (in[i] - 'a' + 10) : (in[i] - 'A' + 10));
      out[i / 2] |= v << (4 * (1 - i % 2));
   }
}

static void
radv_physical_device_init_cache_key(struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   struct radv_physical_device_cache_key *key = &pdev->cache_key;

   key->family = pdev->info.family;
   key->ptr_size = sizeof(void *);
   key->conformant_trunc_coord = pdev->info.conformant_trunc_coord;

   key->clear_lds = instance->drirc.clear_lds;
   key->cs_wave32 = pdev->cs_wave_size == 32;
   key->disable_aniso_single_level = instance->drirc.disable_aniso_single_level && pdev->info.gfx_level < GFX8;
   key->disable_shrink_image_store = instance->drirc.disable_shrink_image_store;
   key->disable_sinking_load_input_fs = instance->drirc.disable_sinking_load_input_fs;
   key->disable_trunc_coord = instance->drirc.disable_trunc_coord;
   key->emulate_rt = radv_emulate_rt(pdev);
   key->bvh8 = radv_use_bvh8(pdev);
   key->ge_wave32 = pdev->ge_wave_size == 32;
   key->invariant_geom = !!(instance->debug_flags & RADV_DEBUG_INVARIANT_GEOM);
   key->no_fmask = !!(instance->debug_flags & RADV_DEBUG_NO_FMASK);
   key->no_ngg_gs = !!(instance->debug_flags & RADV_DEBUG_NO_NGG_GS);
   key->no_rt = !!(instance->debug_flags & RADV_DEBUG_NO_RT);
   key->ps_wave32 = pdev->ps_wave_size == 32;
   key->rt_wave64 = pdev->rt_wave_size == 64;
   key->split_fma = !!(instance->debug_flags & RADV_DEBUG_SPLIT_FMA);
   key->ssbo_non_uniform = instance->drirc.ssbo_non_uniform;
   key->tex_non_uniform = instance->drirc.tex_non_uniform;
   key->lower_terminate_to_discard = instance->drirc.lower_terminate_to_discard;
   key->use_llvm = pdev->use_llvm;
   key->use_ngg = pdev->use_ngg;
   key->use_ngg_culling = pdev->use_ngg_culling;
}

static int
radv_device_get_cache_uuid(struct radv_physical_device *pdev, void *uuid)
{
   struct mesa_sha1 ctx;
   unsigned char sha1[20];

   memset(uuid, 0, VK_UUID_SIZE);
   _mesa_sha1_init(&ctx);

#ifdef RADV_BUILD_ID_OVERRIDE
   {
      unsigned size = strlen(RADV_BUILD_ID_OVERRIDE) / 2;
      char *data = alloca(size);
      parse_hex(data, RADV_BUILD_ID_OVERRIDE, size);
      _mesa_sha1_update(&ctx, data, size);
   }
#else
   if (!disk_cache_get_function_identifier(radv_device_get_cache_uuid, &ctx))
      return -1;
#endif

#if AMD_LLVM_AVAILABLE
   if (pdev->use_llvm && !disk_cache_get_function_identifier(LLVMInitializeAMDGPUTargetInfo, &ctx))
      return -1;
#endif

   _mesa_sha1_final(&ctx, sha1);

   memcpy(uuid, sha1, VK_UUID_SIZE);
   return 0;
}

static void
radv_get_driver_uuid(void *uuid)
{
   ac_compute_driver_uuid(uuid, VK_UUID_SIZE);
}

static void
radv_get_device_uuid(const struct radeon_info *gpu_info, void *uuid)
{
   ac_compute_device_uuid(gpu_info, uuid, VK_UUID_SIZE);
}

static void
radv_physical_device_init_queue_table(struct radv_physical_device *pdev)
{
   int idx = 0;

   for (unsigned i = 0; i < RADV_MAX_QUEUE_FAMILIES; i++)
      pdev->vk_queue_to_radv[i] = RADV_MAX_QUEUE_FAMILIES + 1;

   if (radv_graphics_queue_enabled(pdev)) {
      pdev->vk_queue_to_radv[idx] = RADV_QUEUE_GENERAL;
      idx++;
   }

   if (radv_compute_queue_enabled(pdev)) {
      pdev->vk_queue_to_radv[idx] = RADV_QUEUE_COMPUTE;
      idx++;
   }

   if (radv_video_decode_queue_enabled(pdev)) {
      pdev->vk_queue_to_radv[idx] = RADV_QUEUE_VIDEO_DEC;
      idx++;
   }

   if (radv_transfer_queue_enabled(pdev)) {
      pdev->vk_queue_to_radv[idx] = RADV_QUEUE_TRANSFER;
      idx++;
   }

   if (radv_video_encode_queue_enabled(pdev)) {
      pdev->vk_queue_to_radv[idx] = RADV_QUEUE_VIDEO_ENC;
      idx++;
   }

   if (radv_dedicated_sparse_queue_enabled(pdev)) {
      pdev->vk_queue_to_radv[idx] = RADV_QUEUE_SPARSE;
      idx++;
   }

   pdev->num_queues = idx;
}

enum radv_heap {
   RADV_HEAP_VRAM = 1 << 0,
   RADV_HEAP_GTT = 1 << 1,
   RADV_HEAP_VRAM_VIS = 1 << 2,
   RADV_HEAP_MAX = 1 << 3,
};

static uint64_t
radv_get_adjusted_vram_size(struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   int ov = instance->drirc.override_vram_size;
   if (ov >= 0)
      return MIN2((uint64_t)pdev->info.vram_size_kb * 1024, (uint64_t)ov << 20);
   return (uint64_t)pdev->info.vram_size_kb * 1024;
}

static uint64_t
radv_get_visible_vram_size(struct radv_physical_device *pdev)
{
   return MIN2(radv_get_adjusted_vram_size(pdev), (uint64_t)pdev->info.vram_vis_size_kb * 1024);
}

static uint64_t
radv_get_vram_size(struct radv_physical_device *pdev)
{
   uint64_t total_size = radv_get_adjusted_vram_size(pdev);
   return total_size - MIN2(total_size, (uint64_t)pdev->info.vram_vis_size_kb * 1024);
}

static void
radv_physical_device_init_mem_types(struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   uint64_t visible_vram_size = radv_get_visible_vram_size(pdev);
   uint64_t vram_size = radv_get_vram_size(pdev);
   uint64_t gtt_size = (uint64_t)pdev->info.gart_size_kb * 1024;
   int vram_index = -1, visible_vram_index = -1, gart_index = -1;

   pdev->memory_properties.memoryHeapCount = 0;
   pdev->heaps = 0;

   if (!pdev->info.has_dedicated_vram) {
      const uint64_t total_size = gtt_size + visible_vram_size;

      if (instance->drirc.enable_unified_heap_on_apu) {
         /* Some applications seem better when the driver exposes only one heap of VRAM on APUs. */
         visible_vram_size = total_size;
         gtt_size = 0;
      } else {
         /* On APUs, the carveout is usually too small for games that request a minimum VRAM size
          * greater than it. To workaround this, we compute the total available memory size (GTT +
          * visible VRAM size) and report 2/3 as VRAM and 1/3 as GTT.
          */
         visible_vram_size = align64((total_size * 2) / 3, pdev->info.gart_page_size);
         gtt_size = total_size - visible_vram_size;
      }

      vram_size = 0;
   }

   /* Only get a VRAM heap if it is significant, not if it is a 16 MiB
    * remainder above visible VRAM. */
   if (vram_size > 0 && vram_size * 9 >= visible_vram_size) {
      vram_index = pdev->memory_properties.memoryHeapCount++;
      pdev->heaps |= RADV_HEAP_VRAM;
      pdev->memory_properties.memoryHeaps[vram_index] = (VkMemoryHeap){
         .size = vram_size,
         .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
      };
   }

   if (gtt_size > 0) {
      gart_index = pdev->memory_properties.memoryHeapCount++;
      pdev->heaps |= RADV_HEAP_GTT;
      pdev->memory_properties.memoryHeaps[gart_index] = (VkMemoryHeap){
         .size = gtt_size,
         .flags = 0,
      };
   }

   if (visible_vram_size) {
      visible_vram_index = pdev->memory_properties.memoryHeapCount++;
      pdev->heaps |= RADV_HEAP_VRAM_VIS;
      pdev->memory_properties.memoryHeaps[visible_vram_index] = (VkMemoryHeap){
         .size = visible_vram_size,
         .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
      };
   }

   unsigned type_count = 0;

   if (vram_index >= 0 || visible_vram_index >= 0) {
      pdev->memory_domains[type_count] = RADEON_DOMAIN_VRAM;
      pdev->memory_flags[type_count] = RADEON_FLAG_NO_CPU_ACCESS;
      pdev->memory_properties.memoryTypes[type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
         .heapIndex = vram_index >= 0 ? vram_index : visible_vram_index,
      };

      pdev->memory_domains[type_count] = RADEON_DOMAIN_VRAM;
      pdev->memory_flags[type_count] = RADEON_FLAG_NO_CPU_ACCESS | RADEON_FLAG_32BIT;
      pdev->memory_properties.memoryTypes[type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
         .heapIndex = vram_index >= 0 ? vram_index : visible_vram_index,
      };
   }

   if (gart_index >= 0) {
      pdev->memory_domains[type_count] = RADEON_DOMAIN_GTT;
      pdev->memory_flags[type_count] = RADEON_FLAG_GTT_WC | RADEON_FLAG_CPU_ACCESS;
      pdev->memory_properties.memoryTypes[type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         .heapIndex = gart_index,
      };
   }
   if (visible_vram_index >= 0) {
      pdev->memory_domains[type_count] = RADEON_DOMAIN_VRAM;
      pdev->memory_flags[type_count] = RADEON_FLAG_CPU_ACCESS;
      pdev->memory_properties.memoryTypes[type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         .heapIndex = visible_vram_index,
      };

      pdev->memory_domains[type_count] = RADEON_DOMAIN_VRAM;
      pdev->memory_flags[type_count] = RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_32BIT;
      pdev->memory_properties.memoryTypes[type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         .heapIndex = visible_vram_index,
      };
   }

   if (gart_index >= 0) {
      pdev->memory_domains[type_count] = RADEON_DOMAIN_GTT;
      pdev->memory_flags[type_count] = RADEON_FLAG_CPU_ACCESS;
      pdev->memory_properties.memoryTypes[type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                          VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
         .heapIndex = gart_index,
      };

      pdev->memory_domains[type_count] = RADEON_DOMAIN_GTT;
      pdev->memory_flags[type_count] = RADEON_FLAG_CPU_ACCESS | RADEON_FLAG_32BIT;
      pdev->memory_properties.memoryTypes[type_count++] = (VkMemoryType){
         .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                          VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
         .heapIndex = gart_index,
      };
   }
   pdev->memory_properties.memoryTypeCount = type_count;

   if (pdev->info.has_l2_uncached) {
      for (int i = 0; i < pdev->memory_properties.memoryTypeCount; i++) {
         VkMemoryType mem_type = pdev->memory_properties.memoryTypes[i];

         if (((mem_type.propertyFlags & (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ||
              mem_type.propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
             !(pdev->memory_flags[i] & RADEON_FLAG_32BIT)) {

            VkMemoryPropertyFlags property_flags = mem_type.propertyFlags | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |
                                                   VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;

            pdev->memory_domains[type_count] = pdev->memory_domains[i];
            pdev->memory_flags[type_count] = pdev->memory_flags[i] | RADEON_FLAG_VA_UNCACHED;
            pdev->memory_properties.memoryTypes[type_count++] = (VkMemoryType){
               .propertyFlags = property_flags,
               .heapIndex = mem_type.heapIndex,
            };
         }
      }
      pdev->memory_properties.memoryTypeCount = type_count;
   }

   for (unsigned i = 0; i < type_count; ++i) {
      if (pdev->memory_flags[i] & RADEON_FLAG_32BIT)
         pdev->memory_types_32bit |= BITFIELD_BIT(i);
      if (pdev->memory_flags[i] & RADEON_FLAG_CPU_ACCESS)
         pdev->memory_types_host_visible |= BITFIELD_BIT(i);
   }
}

uint32_t
radv_find_memory_index(const struct radv_physical_device *pdev, VkMemoryPropertyFlags flags)
{
   const VkPhysicalDeviceMemoryProperties *mem_properties = &pdev->memory_properties;
   for (uint32_t i = 0; i < mem_properties->memoryTypeCount; ++i) {
      if (mem_properties->memoryTypes[i].propertyFlags == flags) {
         return i;
      }
   }
   UNREACHABLE("invalid memory properties");
}

static void
radv_get_binning_settings(const struct radv_physical_device *pdev, struct radv_binning_settings *settings)
{
   if ((pdev->info.has_dedicated_vram && pdev->info.max_render_backends > 4) || pdev->info.gfx_level >= GFX10) {
      /* Using higher settings on GFX10+ can cause random GPU hangs. */
      settings->context_states_per_bin = 1;
      settings->persistent_states_per_bin = 1;
   } else {
      settings->context_states_per_bin = pdev->info.has_gfx9_scissor_bug ? 1 : 3;
      settings->persistent_states_per_bin = 1;
   }

   settings->fpovs_per_batch = 63;
}

static void
radv_physical_device_get_supported_extensions(const struct radv_physical_device *pdev,
                                              struct vk_device_extension_table *out_ext)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   const struct vk_device_extension_table ext = {
      .KHR_8bit_storage = true,
      .KHR_16bit_storage = true,
      .KHR_acceleration_structure = radv_enable_rt(pdev),
      .KHR_calibrated_timestamps = radv_calibrated_timestamps_enabled(pdev),
      .KHR_compute_shader_derivatives = true,
      .KHR_cooperative_matrix = radv_cooperative_matrix_enabled(pdev),
      .KHR_bind_memory2 = true,
      .KHR_buffer_device_address = true,
      .KHR_copy_commands2 = true,
      .KHR_create_renderpass2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_deferred_host_operations = true,
      .KHR_depth_clamp_zero_one = true,
      .KHR_depth_stencil_resolve = true,
      .KHR_descriptor_update_template = true,
      .KHR_device_group = true,
      .KHR_draw_indirect_count = true,
      .KHR_driver_properties = true,
      .KHR_dynamic_rendering = true,
      .KHR_dynamic_rendering_local_read = true,
      .KHR_external_fence = true,
      .KHR_external_fence_fd = true,
      .KHR_external_memory = true,
      .KHR_external_memory_fd = true,
      .KHR_external_semaphore = true,
      .KHR_external_semaphore_fd = true,
      .KHR_format_feature_flags2 = true,
      .KHR_fragment_shader_barycentric = pdev->info.gfx_level >= GFX10_3,
      .KHR_fragment_shading_rate = pdev->info.gfx_level >= GFX10_3,
      .KHR_get_memory_requirements2 = true,
      .KHR_global_priority = true,
      .KHR_image_format_list = true,
      .KHR_imageless_framebuffer = true,
#ifdef RADV_USE_WSI_PLATFORM
      .KHR_incremental_present = true,
#endif
      .KHR_index_type_uint8 = pdev->info.gfx_level >= GFX8,
      .KHR_line_rasterization = true,
      .KHR_load_store_op_none = true,
      .KHR_maintenance1 = true,
      .KHR_maintenance2 = true,
      .KHR_maintenance3 = true,
      .KHR_maintenance4 = true,
      .KHR_maintenance5 = true,
      .KHR_maintenance6 = true,
      .KHR_maintenance7 = true,
      .KHR_maintenance8 = true,
      .KHR_maintenance9 = true,
      .KHR_map_memory2 = true,
      .KHR_multiview = true,
      .KHR_performance_query = radv_perf_query_supported(pdev),
      .KHR_pipeline_binary = true,
      .KHR_pipeline_executable_properties = true,
      .KHR_pipeline_library = !pdev->use_llvm,
      /* Hide these behind dri configs for now since we cannot implement it reliably on
       * all surfaces yet. There is no surface capability query for present wait/id,
       * but the feature is useful enough to hide behind an opt-in mechanism for now.
       * If the instance only enables surface extensions that unconditionally support present wait,
       * we can also expose the extension that way. */
      .KHR_present_id =
         instance->drirc.enable_khr_present_wait || wsi_common_vk_instance_supports_present_wait(&instance->vk),
      .KHR_present_id2 = true,
      .KHR_present_wait =
         (instance->drirc.enable_khr_present_wait || wsi_common_vk_instance_supports_present_wait(&instance->vk)) &&
         pdev->info.has_timeline_syncobj,
      .KHR_present_wait2 = true,
      .KHR_push_descriptor = true,
      .KHR_ray_query = radv_enable_rt(pdev),
      .KHR_ray_tracing_maintenance1 = radv_enable_rt(pdev),
      .KHR_ray_tracing_pipeline = radv_enable_rt(pdev),
      .KHR_ray_tracing_position_fetch = radv_enable_rt(pdev),
      .KHR_relaxed_block_layout = true,
      .KHR_robustness2 = true,
      .KHR_sampler_mirror_clamp_to_edge = true,
      .KHR_sampler_ycbcr_conversion = true,
      .KHR_separate_depth_stencil_layouts = true,
      .KHR_shader_atomic_int64 = true,
      .KHR_shader_bfloat16 = pdev->info.gfx_level >= GFX12, /* GFX11 has precision issues. */
      .KHR_shader_clock = true,
      .KHR_shader_draw_parameters = true,
      .KHR_shader_expect_assume = true,
      .KHR_shader_float16_int8 = true,
      .KHR_shader_float_controls = true,
      .KHR_shader_float_controls2 = true,
      .KHR_shader_integer_dot_product = true,
      .KHR_shader_maximal_reconvergence = true,
      .KHR_shader_non_semantic_info = true,
      .KHR_shader_quad_control = true,
      .KHR_shader_relaxed_extended_instruction = true,
      .KHR_shader_subgroup_extended_types = true,
      .KHR_shader_subgroup_rotate = true,
      .KHR_shader_subgroup_uniform_control_flow = true,
      .KHR_shader_terminate_invocation = true,
      .KHR_spirv_1_4 = true,
      .KHR_storage_buffer_storage_class = true,
#ifdef RADV_USE_WSI_PLATFORM
      .KHR_swapchain = true,
      .KHR_swapchain_mutable_format = true,
#endif
      .KHR_synchronization2 = true,
      .KHR_timeline_semaphore = pdev->info.has_timeline_syncobj,
      .KHR_unified_image_layouts = pdev->info.gfx_level >= GFX11,
      .KHR_uniform_buffer_standard_layout = true,
      .KHR_variable_pointers = true,
      .KHR_vertex_attribute_divisor = true,
      .KHR_video_maintenance1 = pdev->video_decode_enabled || pdev->video_encode_enabled,
      .KHR_video_maintenance2 = pdev->video_decode_enabled || pdev->video_encode_enabled,
      .KHR_video_queue = pdev->video_decode_enabled || pdev->video_encode_enabled,
      .KHR_video_decode_av1 = (pdev->info.vcn_ip_version >= VCN_3_0_0 && pdev->info.vcn_ip_version != VCN_3_0_33 &&
                               VIDEO_CODEC_AV1DEC && pdev->video_decode_enabled),
      .KHR_video_decode_queue = pdev->video_decode_enabled,
      .KHR_video_decode_h264 = VIDEO_CODEC_H264DEC && pdev->video_decode_enabled,
      .KHR_video_decode_h265 = VIDEO_CODEC_H265DEC && pdev->video_decode_enabled,
      .KHR_video_decode_vp9 =
         (radv_video_decode_vp9_supported(pdev) && VIDEO_CODEC_VP9DEC && pdev->video_decode_enabled),
      .KHR_video_encode_h264 = VIDEO_CODEC_H264ENC && pdev->video_encode_enabled,
      .KHR_video_encode_h265 = VIDEO_CODEC_H265ENC && pdev->video_encode_enabled,
      .KHR_video_encode_av1 =
         (radv_video_encode_av1_supported(pdev) && VIDEO_CODEC_AV1ENC && pdev->video_encode_enabled),
      .KHR_video_encode_queue = pdev->video_encode_enabled,
      .KHR_vulkan_memory_model = true,
      .KHR_workgroup_memory_explicit_layout = true,
      .KHR_zero_initialize_workgroup_memory = true,
      .EXT_4444_formats = true,
      .EXT_attachment_feedback_loop_dynamic_state = true,
      .EXT_attachment_feedback_loop_layout = true,
      .EXT_border_color_swizzle = pdev->info.gfx_level >= GFX10,
      .EXT_buffer_device_address = true,
      .EXT_calibrated_timestamps = radv_calibrated_timestamps_enabled(pdev),
      .EXT_color_write_enable = true,
      .EXT_conditional_rendering = true,
      .EXT_conservative_rasterization = pdev->info.gfx_level >= GFX9,
      .EXT_custom_border_color = true,
      .EXT_debug_marker = instance->vk.trace_mode & RADV_TRACE_MODE_RGP,
      .EXT_depth_bias_control = true,
      .EXT_depth_clamp_zero_one = true,
      .EXT_depth_clamp_control = true,
      .EXT_depth_clip_control = true,
      .EXT_depth_clip_enable = true,
      .EXT_depth_range_unrestricted = true,
      .EXT_descriptor_buffer = true,
      .EXT_descriptor_indexing = true,
      .EXT_device_address_binding_report = true,
      .EXT_device_fault = pdev->info.has_gpuvm_fault_query,
      .EXT_device_generated_commands = pdev->info.gfx_level >= GFX8,
      .EXT_device_memory_report = true,
      .EXT_discard_rectangles = true,
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
      .EXT_display_control = true,
#endif
      .EXT_dynamic_rendering_unused_attachments = true,
      .EXT_extended_dynamic_state = true,
      .EXT_extended_dynamic_state2 = true,
      .EXT_extended_dynamic_state3 = true,
      .EXT_external_memory_acquire_unmodified = true,
      .EXT_external_memory_dma_buf = true,
      .EXT_external_memory_host = pdev->info.has_userptr,
      .EXT_fragment_shader_interlock = radv_has_pops(pdev),
      .EXT_global_priority = true,
      .EXT_global_priority_query = true,
      .EXT_graphics_pipeline_library = !pdev->use_llvm && !(instance->debug_flags & RADV_DEBUG_NO_GPL),
      .EXT_hdr_metadata = true,
      .EXT_host_image_copy = radv_host_image_copy_enabled(pdev),
      .EXT_host_query_reset = true,
      .EXT_image_2d_view_of_3d = true,
      .EXT_image_compression_control = true,
      .EXT_image_drm_format_modifier = pdev->info.gfx_level >= GFX9,
      .EXT_image_robustness = true,
      .EXT_image_sliced_view_of_3d = pdev->info.gfx_level >= GFX10,
      .EXT_image_view_min_lod = true,
      .EXT_index_type_uint8 = pdev->info.gfx_level >= GFX8,
      .EXT_inline_uniform_block = true,
      .EXT_legacy_vertex_attributes = !pdev->use_llvm,
      .EXT_line_rasterization = true,
      .EXT_load_store_op_none = true,
      .EXT_map_memory_placed = true,
      .EXT_memory_budget = true,
      .EXT_memory_priority = true,
      .EXT_mesh_shader = radv_taskmesh_enabled(pdev),
      .EXT_multi_draw = true,
      .EXT_mutable_descriptor_type = true, /* Trivial promotion from VALVE. */
      .EXT_nested_command_buffer = true,
      .EXT_non_seamless_cube_map = true,
      .EXT_pci_bus_info = true,
#ifndef _WIN32
      .EXT_physical_device_drm = true,
#endif
      .EXT_pipeline_creation_cache_control = true,
      .EXT_pipeline_creation_feedback = true,
      .EXT_pipeline_library_group_handles = radv_enable_rt(pdev),
      .EXT_pipeline_robustness = !pdev->use_llvm,
      .EXT_post_depth_coverage = pdev->info.gfx_level >= GFX10,
      .EXT_primitive_topology_list_restart = true,
      .EXT_primitives_generated_query = true,
      .EXT_private_data = true,
      .EXT_provoking_vertex = true,
      .EXT_queue_family_foreign = true,
      .EXT_robustness2 = true,
      .EXT_sample_locations = true,
      .EXT_sampler_filter_minmax = radv_filter_minmax_enabled(pdev),
      .EXT_scalar_block_layout = true,
      .EXT_separate_stencil_usage = true,
      .EXT_shader_atomic_float = true,
      .EXT_shader_atomic_float2 = true,
      .EXT_shader_demote_to_helper_invocation = true,
      .EXT_shader_float8 = pdev->info.gfx_level >= GFX12 && !pdev->use_llvm,
      .EXT_shader_image_atomic_int64 = true,
      .EXT_shader_module_identifier = true,
      .EXT_shader_object = !pdev->use_llvm && !(instance->debug_flags & RADV_DEBUG_NO_ESO),
      .EXT_shader_replicated_composites = true,
      .EXT_shader_stencil_export = true,
      .EXT_shader_subgroup_ballot = true,
      .EXT_shader_subgroup_vote = true,
      .EXT_shader_viewport_index_layer = true,
      .EXT_subgroup_size_control = true,
#ifdef RADV_USE_WSI_PLATFORM
      .EXT_swapchain_maintenance1 = true,
#endif
      .EXT_texel_buffer_alignment = true,
      .EXT_tooling_info = true,
      .EXT_transform_feedback = true,
      .EXT_vertex_attribute_divisor = true,
      .EXT_vertex_input_dynamic_state = !pdev->use_llvm,
      .EXT_ycbcr_image_arrays = true,
      .EXT_zero_initialize_device_memory = true,
      .AMD_buffer_marker = true,
      .AMD_device_coherent_memory = true,
      .AMD_draw_indirect_count = true,
      .AMD_gcn_shader = true,
      .AMD_gpu_shader_half_float = pdev->info.has_packed_math_16bit,
      .AMD_gpu_shader_int16 = pdev->info.has_packed_math_16bit,
      .AMD_memory_overallocation_behavior = true,
      .AMD_mixed_attachment_samples = true,
      .AMD_rasterization_order = pdev->info.has_out_of_order_rast,
      .AMD_shader_ballot = true,
      .AMD_shader_core_properties = true,
      .AMD_shader_core_properties2 = true,
      .AMD_shader_early_and_late_fragment_tests = true,
      .AMD_shader_explicit_vertex_parameter = true,
      .AMD_shader_fragment_mask = pdev->use_fmask,
      .AMD_shader_image_load_store_lod = true,
      .AMD_shader_trinary_minmax = true,
      .AMD_texture_gather_bias_lod = pdev->info.gfx_level < GFX11,
#if DETECT_OS_ANDROID
      .ANDROID_external_memory_android_hardware_buffer = RADV_SUPPORT_ANDROID_HARDWARE_BUFFER,
      .ANDROID_native_buffer = true,
#endif
      .GOOGLE_decorate_string = true,
      .GOOGLE_hlsl_functionality1 = true,
      .GOOGLE_user_type = true,
      .INTEL_shader_integer_functions2 = true,
      .MESA_image_alignment_control = pdev->info.gfx_level >= GFX9,
      .NV_compute_shader_derivatives = true,
      .NV_cooperative_matrix2 = radv_cooperative_matrix2_nv_enabled(pdev),
      .VALVE_mutable_descriptor_type = true,
   };
   *out_ext = ext;
}

static void
radv_physical_device_get_features(const struct radv_physical_device *pdev, struct vk_features *features)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   bool taskmesh_en = radv_taskmesh_enabled(pdev);
   bool has_perf_query = radv_perf_query_supported(pdev);
   bool has_shader_image_float_minmax = pdev->info.gfx_level != GFX8 && pdev->info.gfx_level != GFX9 &&
                                        pdev->info.gfx_level != GFX11 && pdev->info.gfx_level != GFX11_5;
   bool has_fragment_shader_interlock = radv_has_pops(pdev);

   *features = (struct vk_features){
      /* Vulkan 1.0 */
      .robustBufferAccess = true,
      .fullDrawIndexUint32 = true,
      .imageCubeArray = true,
      .independentBlend = true,
      .geometryShader = true,
      .tessellationShader = true,
      .sampleRateShading = true,
      .dualSrcBlend = true,
      .logicOp = true,
      .multiDrawIndirect = true,
      .drawIndirectFirstInstance = true,
      .depthClamp = true,
      .depthBiasClamp = true,
      .fillModeNonSolid = true,
      .depthBounds = true,
      .wideLines = true,
      .largePoints = true,
      .alphaToOne = true,
      .multiViewport = true,
      .samplerAnisotropy = true,
      .textureCompressionETC2 = pdev->info.has_etc_support || pdev->emulate_etc2,
      .textureCompressionASTC_LDR = pdev->emulate_astc,
      .textureCompressionBC = true,
      .occlusionQueryPrecise = true,
      .pipelineStatisticsQuery = true,
      .vertexPipelineStoresAndAtomics = true,
      .fragmentStoresAndAtomics = true,
      .shaderTessellationAndGeometryPointSize = true,
      .shaderImageGatherExtended = true,
      .shaderStorageImageExtendedFormats = true,
      .shaderStorageImageMultisample = true,
      .shaderUniformBufferArrayDynamicIndexing = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,
      .shaderStorageImageReadWithoutFormat = true,
      .shaderStorageImageWriteWithoutFormat = true,
      .shaderClipDistance = true,
      .shaderCullDistance = true,
      .shaderFloat64 = true,
      .shaderInt64 = true,
      .shaderInt16 = true,
      .sparseBinding = true,
      .sparseResidencyBuffer = pdev->info.family >= CHIP_POLARIS10,
      .sparseResidencyImage2D = pdev->info.family >= CHIP_POLARIS10,
      .sparseResidencyImage3D = pdev->info.family >= CHIP_POLARIS10,
      .sparseResidencyAliased = pdev->info.family >= CHIP_POLARIS10,
      .variableMultisampleRate = true,
      .shaderResourceMinLod = true,
      .shaderResourceResidency = true,
      .inheritedQueries = true,

      /* Vulkan 1.1 */
      .storageBuffer16BitAccess = true,
      .uniformAndStorageBuffer16BitAccess = true,
      .storagePushConstant16 = true,
      .storageInputOutput16 = pdev->info.has_packed_math_16bit,
      .multiview = true,
      .multiviewGeometryShader = true,
      .multiviewTessellationShader = true,
      .variablePointersStorageBuffer = true,
      .variablePointers = true,
      .protectedMemory = false,
      .samplerYcbcrConversion = true,
      .shaderDrawParameters = true,

      /* Vulkan 1.2 */
      .samplerMirrorClampToEdge = true,
      .drawIndirectCount = true,
      .storageBuffer8BitAccess = true,
      .uniformAndStorageBuffer8BitAccess = true,
      .storagePushConstant8 = true,
      .shaderBufferInt64Atomics = true,
      .shaderSharedInt64Atomics = true,
      .shaderFloat16 =
         pdev->info.has_packed_math_16bit || (pdev->info.gfx_level == GFX8 && instance->drirc.expose_float16_gfx8),
      .shaderInt8 = true,

      .descriptorIndexing = pdev->info.has_vm_always_valid,
      .shaderInputAttachmentArrayDynamicIndexing = true,
      .shaderUniformTexelBufferArrayDynamicIndexing = true,
      .shaderStorageTexelBufferArrayDynamicIndexing = true,
      .shaderUniformBufferArrayNonUniformIndexing = true,
      .shaderSampledImageArrayNonUniformIndexing = true,
      .shaderStorageBufferArrayNonUniformIndexing = true,
      .shaderStorageImageArrayNonUniformIndexing = true,
      .shaderInputAttachmentArrayNonUniformIndexing = true,
      .shaderUniformTexelBufferArrayNonUniformIndexing = true,
      .shaderStorageTexelBufferArrayNonUniformIndexing = true,
      .descriptorBindingUniformBufferUpdateAfterBind = pdev->info.has_vm_always_valid,
      .descriptorBindingSampledImageUpdateAfterBind = pdev->info.has_vm_always_valid,
      .descriptorBindingStorageImageUpdateAfterBind = pdev->info.has_vm_always_valid,
      .descriptorBindingStorageBufferUpdateAfterBind = pdev->info.has_vm_always_valid,
      .descriptorBindingUniformTexelBufferUpdateAfterBind = pdev->info.has_vm_always_valid,
      .descriptorBindingStorageTexelBufferUpdateAfterBind = pdev->info.has_vm_always_valid,
      .descriptorBindingUpdateUnusedWhilePending = pdev->info.has_vm_always_valid,
      .descriptorBindingPartiallyBound = pdev->info.has_vm_always_valid,
      .descriptorBindingVariableDescriptorCount = true,
      .runtimeDescriptorArray = true,

      .samplerFilterMinmax = radv_filter_minmax_enabled(pdev),
      .scalarBlockLayout = true,
      .imagelessFramebuffer = true,
      .uniformBufferStandardLayout = true,
      .shaderSubgroupExtendedTypes = true,
      .separateDepthStencilLayouts = true,
      .hostQueryReset = true,
      .timelineSemaphore = pdev->info.has_timeline_syncobj,
      .bufferDeviceAddress = pdev->info.has_vm_always_valid,
      .bufferDeviceAddressCaptureReplay = true,
      .bufferDeviceAddressMultiDevice = false,
      .vulkanMemoryModel = true,
      .vulkanMemoryModelDeviceScope = true,
      .vulkanMemoryModelAvailabilityVisibilityChains = false,
      .shaderOutputViewportIndex = true,
      .shaderOutputLayer = true,
      .subgroupBroadcastDynamicId = true,

      /* Vulkan 1.3 */
      .robustImageAccess = true,
      .inlineUniformBlock = true,
      .descriptorBindingInlineUniformBlockUpdateAfterBind = true,
      .pipelineCreationCacheControl = true,
      .privateData = true,
      .shaderDemoteToHelperInvocation = true,
      .shaderTerminateInvocation = true,
      .subgroupSizeControl = true,
      .computeFullSubgroups = true,
      .synchronization2 = true,
      .textureCompressionASTC_HDR = false,
      .shaderZeroInitializeWorkgroupMemory = true,
      .dynamicRendering = true,
      .shaderIntegerDotProduct = true,
      .maintenance4 = true,

      /* Vulkan 1.4 */
      .globalPriorityQuery = true,
      .shaderSubgroupRotate = true,
      .shaderSubgroupRotateClustered = true,
      .shaderFloatControls2 = true,
      .shaderExpectAssume = true,
      .rectangularLines = true,
      .bresenhamLines = true,
      .smoothLines = true,
      .stippledRectangularLines = false,
      .stippledBresenhamLines = true,
      .stippledSmoothLines = false,
      .vertexAttributeInstanceRateDivisor = true,
      .vertexAttributeInstanceRateZeroDivisor = true,
      .indexTypeUint8 = pdev->info.gfx_level >= GFX8,
      .dynamicRenderingLocalRead = true,
      .maintenance5 = true,
      .maintenance6 = true,
      .pipelineProtectedAccess = false,
      .pipelineRobustness = true,
      .hostImageCopy = radv_host_image_copy_enabled(pdev),
      .pushDescriptor = true,

      /* VK_EXT_conditional_rendering */
      .conditionalRendering = true,
      .inheritedConditionalRendering = false,

      /* VK_KHR_vertex_attribute_divisor */
      .vertexAttributeInstanceRateDivisor = true,
      .vertexAttributeInstanceRateZeroDivisor = true,

      /* VK_EXT_transform_feedback */
      .transformFeedback = true,
      .geometryStreams = true,

      /* VK_EXT_memory_priority */
      .memoryPriority = true,

      /* VK_EXT_depth_clip_enable */
      .depthClipEnable = true,

      /* VK_KHR_compute_shader_derivatives */
      .computeDerivativeGroupQuads = pdev->info.gfx_level >= GFX12,
      .computeDerivativeGroupLinear = true,

      /* VK_EXT_ycbcr_image_arrays */
      .ycbcrImageArrays = true,

      /* VK_KHR_index_type_uint8 */
      .indexTypeUint8 = pdev->info.gfx_level >= GFX8,

      /* VK_KHR_pipeline_executable_properties */
      .pipelineExecutableInfo = true,

      /* VK_KHR_shader_clock */
      .shaderSubgroupClock = true,
      .shaderDeviceClock = pdev->info.gfx_level >= GFX8,

      /* VK_EXT_texel_buffer_alignment */
      .texelBufferAlignment = true,

      /* VK_AMD_device_coherent_memory */
      .deviceCoherentMemory = pdev->info.has_l2_uncached,

      /* VK_KHR_line_rasterization */
      .rectangularLines = true,
      .bresenhamLines = true,
      .smoothLines = true,
      .stippledRectangularLines = false,
      .stippledBresenhamLines = true,
      .stippledSmoothLines = false,

      /* VK_KHR_robustness2 */
      .robustBufferAccess2 = true,
      .robustImageAccess2 = true,
      .nullDescriptor = true,

      /* VK_EXT_custom_border_color */
      .customBorderColors = true,
      .customBorderColorWithoutFormat = true,

      /* VK_EXT_extended_dynamic_state */
      .extendedDynamicState = true,

      /* VK_EXT_shader_atomic_float */
      .shaderBufferFloat32Atomics = true,
      .shaderBufferFloat32AtomicAdd = pdev->info.gfx_level >= GFX11,
      .shaderBufferFloat64Atomics = true,
      .shaderBufferFloat64AtomicAdd = false,
      .shaderSharedFloat32Atomics = true,
      .shaderSharedFloat32AtomicAdd = pdev->info.gfx_level >= GFX8,
      .shaderSharedFloat64Atomics = true,
      .shaderSharedFloat64AtomicAdd = false,
      .shaderImageFloat32Atomics = true,
      .shaderImageFloat32AtomicAdd = pdev->info.gfx_level >= GFX12 && !pdev->use_llvm,
      .sparseImageFloat32Atomics = true,
      .sparseImageFloat32AtomicAdd = pdev->info.gfx_level >= GFX12 && !pdev->use_llvm,

      /* VK_EXT_4444_formats */
      .formatA4R4G4B4 = true,
      .formatA4B4G4R4 = true,

      /* VK_EXT_shader_image_atomic_int64 */
      .shaderImageInt64Atomics = true,
      .sparseImageInt64Atomics = true,

      /* VK_EXT_mutable_descriptor_type */
      .mutableDescriptorType = true,

      /* VK_KHR_fragment_shading_rate */
      .pipelineFragmentShadingRate = true,
      .primitiveFragmentShadingRate = true,
      .attachmentFragmentShadingRate = radv_vrs_attachment_enabled(pdev),

      /* VK_KHR_workgroup_memory_explicit_layout */
      .workgroupMemoryExplicitLayout = true,
      .workgroupMemoryExplicitLayoutScalarBlockLayout = true,
      .workgroupMemoryExplicitLayout8BitAccess = true,
      .workgroupMemoryExplicitLayout16BitAccess = true,

      /* VK_EXT_provoking_vertex */
      .provokingVertexLast = true,
      .transformFeedbackPreservesProvokingVertex = true,

      /* VK_EXT_extended_dynamic_state2 */
      .extendedDynamicState2 = true,
      .extendedDynamicState2LogicOp = true,
      .extendedDynamicState2PatchControlPoints = true,

      /* VK_EXT_global_priority_query */
      .globalPriorityQuery = true,

      /* VK_KHR_acceleration_structure */
      .accelerationStructure = true,
      .accelerationStructureCaptureReplay = true,
      .accelerationStructureIndirectBuild = false,
      .accelerationStructureHostCommands = false,
      .descriptorBindingAccelerationStructureUpdateAfterBind = true,

      /* VK_EXT_buffer_device_address */
      .bufferDeviceAddressCaptureReplayEXT = false,

      /* VK_KHR_shader_subgroup_uniform_control_flow */
      .shaderSubgroupUniformControlFlow = true,

      /* VK_EXT_map_memory_placed */
      .memoryMapPlaced = true,
      .memoryMapRangePlaced = false,
      .memoryUnmapReserve = true,

      /* VK_EXT_multi_draw */
      .multiDraw = true,

      /* VK_EXT_color_write_enable */
      .colorWriteEnable = true,

      /* VK_EXT_shader_atomic_float2 */
      .shaderBufferFloat16Atomics = false,
      .shaderBufferFloat16AtomicAdd = false,
      .shaderBufferFloat16AtomicMinMax = false,
      .shaderBufferFloat32AtomicMinMax = radv_has_shader_buffer_float_minmax(pdev, 32),
      .shaderBufferFloat64AtomicMinMax = radv_has_shader_buffer_float_minmax(pdev, 64),
      .shaderSharedFloat16Atomics = false,
      .shaderSharedFloat16AtomicAdd = false,
      .shaderSharedFloat16AtomicMinMax = false,
      .shaderSharedFloat32AtomicMinMax = true,
      .shaderSharedFloat64AtomicMinMax = true,
      .shaderImageFloat32AtomicMinMax = has_shader_image_float_minmax,
      .sparseImageFloat32AtomicMinMax = has_shader_image_float_minmax,

      /* VK_KHR_present_id */
      .presentId = pdev->vk.supported_extensions.KHR_present_id,

      /* VK_KHR_present_wait */
      .presentWait = pdev->vk.supported_extensions.KHR_present_wait,

      /* VK_EXT_primitive_topology_list_restart */
      .primitiveTopologyListRestart = true,
      .primitiveTopologyPatchListRestart = false,

      /* VK_KHR_ray_query */
      .rayQuery = true,

      /* VK_EXT_pipeline_library_group_handles */
      .pipelineLibraryGroupHandles = true,

      /* VK_KHR_ray_tracing_pipeline */
      .rayTracingPipeline = true,
      .rayTracingPipelineShaderGroupHandleCaptureReplay = true,
      .rayTracingPipelineShaderGroupHandleCaptureReplayMixed = false,
      .rayTracingPipelineTraceRaysIndirect = true,
      .rayTraversalPrimitiveCulling = true,

      /* VK_KHR_ray_tracing_maintenance1 */
      .rayTracingMaintenance1 = true,
      .rayTracingPipelineTraceRaysIndirect2 = radv_enable_rt(pdev),

      /* VK_KHR_ray_tracing_position_fetch */
      .rayTracingPositionFetch = true,

      /* VK_EXT_vertex_input_dynamic_state */
      .vertexInputDynamicState = true,

      /* VK_EXT_image_view_min_lod */
      .minLod = true,

      /* VK_EXT_mesh_shader */
      .meshShader = taskmesh_en,
      .taskShader = taskmesh_en,
      .multiviewMeshShader = taskmesh_en,
      .primitiveFragmentShadingRateMeshShader = taskmesh_en,
      .meshShaderQueries = false,

      /* VK_EXT_depth_clip_control */
      .depthClipControl = true,

      /* VK_EXT_image_2d_view_of_3d  */
      .image2DViewOf3D = true,
      .sampler2DViewOf3D = false,

      /* VK_INTEL_shader_integer_functions2 */
      .shaderIntegerFunctions2 = true,

      /* VK_EXT_primitives_generated_query */
      .primitivesGeneratedQuery = true,
      .primitivesGeneratedQueryWithRasterizerDiscard = true,
      .primitivesGeneratedQueryWithNonZeroStreams = true,

      /* VK_EXT_non_seamless_cube_map */
      .nonSeamlessCubeMap = true,

      /* VK_EXT_border_color_swizzle */
      .borderColorSwizzle = true,
      .borderColorSwizzleFromImage = true,

      /* VK_EXT_shader_module_identifier */
      .shaderModuleIdentifier = true,

      /* VK_KHR_performance_query */
      .performanceCounterQueryPools = has_perf_query,
      .performanceCounterMultipleQueryPools = has_perf_query,

      /* VK_EXT_attachment_feedback_loop_layout */
      .attachmentFeedbackLoopLayout = true,

      /* VK_EXT_graphics_pipeline_library */
      .graphicsPipelineLibrary = true,

      /* VK_EXT_extended_dynamic_state3 */
      .extendedDynamicState3TessellationDomainOrigin = true,
      .extendedDynamicState3PolygonMode = true,
      .extendedDynamicState3SampleMask = true,
      .extendedDynamicState3AlphaToCoverageEnable = !pdev->use_llvm,
      .extendedDynamicState3LogicOpEnable = true,
      .extendedDynamicState3LineStippleEnable = true,
      .extendedDynamicState3ColorBlendEnable = !pdev->use_llvm,
      .extendedDynamicState3DepthClipEnable = true,
      .extendedDynamicState3ConservativeRasterizationMode = pdev->info.gfx_level >= GFX9,
      .extendedDynamicState3DepthClipNegativeOneToOne = true,
      .extendedDynamicState3ProvokingVertexMode = true,
      .extendedDynamicState3DepthClampEnable = true,
      .extendedDynamicState3ColorWriteMask = !pdev->use_llvm,
      .extendedDynamicState3RasterizationSamples = true,
      .extendedDynamicState3ColorBlendEquation = !pdev->use_llvm,
      .extendedDynamicState3SampleLocationsEnable = true,
      .extendedDynamicState3LineRasterizationMode = true,
      .extendedDynamicState3ExtraPrimitiveOverestimationSize = false,
      .extendedDynamicState3AlphaToOneEnable = !pdev->use_llvm,
      .extendedDynamicState3RasterizationStream = false,
      .extendedDynamicState3ColorBlendAdvanced = false,
      .extendedDynamicState3ViewportWScalingEnable = false,
      .extendedDynamicState3ViewportSwizzle = false,
      .extendedDynamicState3CoverageToColorEnable = false,
      .extendedDynamicState3CoverageToColorLocation = false,
      .extendedDynamicState3CoverageModulationMode = false,
      .extendedDynamicState3CoverageModulationTableEnable = false,
      .extendedDynamicState3CoverageModulationTable = false,
      .extendedDynamicState3CoverageReductionMode = false,
      .extendedDynamicState3RepresentativeFragmentTestEnable = false,
      .extendedDynamicState3ShadingRateImageEnable = false,

      /* VK_EXT_descriptor_buffer */
      .descriptorBuffer = true,
      .descriptorBufferCaptureReplay = true,
      .descriptorBufferImageLayoutIgnored = true,
      .descriptorBufferPushDescriptors = true,

      /* VK_AMD_shader_early_and_late_fragment_tests */
      .shaderEarlyAndLateFragmentTests = true,

      /* VK_EXT_image_sliced_view_of_3d */
      .imageSlicedViewOf3D = true,

#ifdef RADV_USE_WSI_PLATFORM
      /* VK_EXT_swapchain_maintenance1 */
      .swapchainMaintenance1 = true,
#endif

      /* VK_EXT_attachment_feedback_loop_dynamic_state */
      .attachmentFeedbackLoopDynamicState = true,

      /* VK_EXT_dynamic_rendering_unused_attachments */
      .dynamicRenderingUnusedAttachments = true,

      /* VK_KHR_fragment_shader_barycentric */
      .fragmentShaderBarycentric = true,

      /* VK_EXT_depth_bias_control */
      .depthBiasControl = true,
      .leastRepresentableValueForceUnormRepresentation = true,
      .floatRepresentation = true,
      .depthBiasExact = true,

      /* VK_EXT_fragment_shader_interlock */
      .fragmentShaderSampleInterlock = has_fragment_shader_interlock,
      .fragmentShaderPixelInterlock = has_fragment_shader_interlock,
      .fragmentShaderShadingRateInterlock = false,

      /* VK_EXT_pipeline_robustness */
      .pipelineRobustness = true,

      /* VK_KHR_maintenance5 */
      .maintenance5 = true,

      /* VK_KHR_cooperative_matrix */
      .cooperativeMatrix = radv_cooperative_matrix_enabled(pdev),
      .cooperativeMatrixRobustBufferAccess = radv_cooperative_matrix_enabled(pdev),

      /* VK_EXT_image_compression_control */
      .imageCompressionControl = true,

      /* VK_EXT_device_fault */
      .deviceFault = true,
      .deviceFaultVendorBinary = instance->debug_flags & RADV_DEBUG_HANG,

      /* VK_KHR_depth_clamp_zero_one */
      .depthClampZeroOne = true,

      /* VK_KHR_maintenance6 */
      .maintenance6 = true,

      /* VK_KHR_shader_subgroup_rotate */
      .shaderSubgroupRotate = true,
      .shaderSubgroupRotateClustered = true,

      /* VK_EXT_shader_object */
      .shaderObject = true,

      /* VK_KHR_shader_expect_assume */
      .shaderExpectAssume = true,

      /* VK_KHR_shader_maximal_reconvergence */
      .shaderMaximalReconvergence = true,

      /* VK_KHR_shader_quad_control */
      .shaderQuadControl = true,

      /* VK_EXT_address_binding_report */
      .reportAddressBinding = true,

      /* VK_EXT_nested_command_buffer */
      .nestedCommandBuffer = true,
      .nestedCommandBufferRendering = true,
      .nestedCommandBufferSimultaneousUse = true,

      /* VK_KHR_dynamic_rendering_local_read */
      .dynamicRenderingLocalRead = true,

      /* VK_EXT_legacy_vertex_attributes */
      .legacyVertexAttributes = true,

      /* VK_MESA_image_alignment_control */
      .imageAlignmentControl = true,

      /* VK_EXT_shader_replicated_composites */
      .shaderReplicatedComposites = true,

      /* VK_KHR_maintenance7 */
      .maintenance7 = true,

      /* VK_KHR_video_maintenance1 */
      .videoMaintenance1 = true,

      /* VK_KHR_video_maintenance2 */
      .videoMaintenance2 = true,

      /* VK_KHR_pipeline_binary */
      .pipelineBinaries = true,

      /* VK_KHR_shader_relaxed_extended_instruction */
      .shaderRelaxedExtendedInstruction = true,

      /* VK_KHR_shader_float_controls2 */
      .shaderFloatControls2 = true,

      /* VK_EXT_depth_clamp_control */
      .depthClampControl = true,

      /* VK_EXT_device_generated_commands */
      .deviceGeneratedCommands = true,
      .dynamicGeneratedPipelineLayout = true,

      /* VK_KHR_maintenance8 */
      .maintenance8 = true,

      /* VK_EXT_device_memory_report */
      .deviceMemoryReport = true,

      /* VK_KHR_shader_bfloat16 */
      .shaderBFloat16Type = true,
      .shaderBFloat16DotProduct = true,
      .shaderBFloat16CooperativeMatrix = radv_cooperative_matrix_enabled(pdev),

      /* VK_EXT_zero_initialize_device_memory */
      .zeroInitializeDeviceMemory = true,

      /* VK_KHR_video_decode_vp9 */
      .videoDecodeVP9 = true,

      /* VK_KHR_maintenance9 */
      .maintenance9 = true,

      /* VK_KHR_unified_layouts */
      .unifiedImageLayouts = true,
      .unifiedImageLayoutsVideo = true,

      /* VK_EXT_shader_float8 */
      .shaderFloat8 = true,
      .shaderFloat8CooperativeMatrix = radv_cooperative_matrix_enabled(pdev),

      /* VK_NV_cooperative_matrix2 */
      .cooperativeMatrixConversions = true,

      /* VK_KHR_video_encode_av1 */
      .videoEncodeAV1 = true,
   };
}

static size_t
radv_max_descriptor_set_size()
{
   /* make sure that the entire descriptor set is addressable with a signed
    * 32-bit int. So the sum of all limits scaled by descriptor size has to
    * be at most 2 GiB. the combined image & samples object count as one of
    * both. This limit is for the pipeline layout, not for the set layout, but
    * there is no set limit, so we just set a pipeline limit. I don't think
    * any app is going to hit this soon. */
   return ((1ull << 31) - 16 * MAX_DYNAMIC_BUFFERS - MAX_INLINE_UNIFORM_BLOCK_SIZE * MAX_INLINE_UNIFORM_BLOCK_COUNT) /
          (32 /* uniform buffer, 32 due to potential space wasted on alignment */ +
           32 /* storage buffer, 32 due to potential space wasted on alignment */ +
           32 /* sampler, largest when combined with image */ + 64 /* sampled image */ + 64 /* storage image */);
}

static uint32_t
radv_uniform_buffer_offset_alignment(const struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   uint32_t uniform_offset_alignment = instance->drirc.override_uniform_offset_alignment;
   if (!util_is_power_of_two_or_zero(uniform_offset_alignment)) {
      fprintf(stderr,
              "ERROR: invalid radv_override_uniform_offset_alignment setting %d:"
              "not a power of two\n",
              uniform_offset_alignment);
      uniform_offset_alignment = 0;
   }

   /* Take at least the hardware limit. */
   return MAX2(uniform_offset_alignment, 4);
}

static const char *
radv_get_compiler_string(struct radv_physical_device *pdev)
{
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   if (!pdev->use_llvm) {
      /* Some games like SotTR apply shader workarounds if the LLVM
       * version is too old or if the LLVM version string is
       * missing. This gives 2-5% performance with SotTR and ACO.
       */
      if (instance->drirc.report_llvm9_version_string) {
         return " (LLVM 9.0.1)";
      }

      return "";
   }

#if AMD_LLVM_AVAILABLE
   return " (LLVM " MESA_LLVM_VERSION_STRING ")";
#else
   UNREACHABLE("LLVM is not available");
#endif
}

static void
radv_get_physical_device_properties(struct radv_physical_device *pdev)
{
   VkSampleCountFlags sample_counts = 0xf;

   size_t max_descriptor_set_size = radv_max_descriptor_set_size();

   VkPhysicalDeviceType device_type;
   if (pdev->info.has_dedicated_vram) {
      device_type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
   } else {
      device_type = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
   }

   bool has_fp16 = pdev->info.has_packed_math_16bit;

   VkShaderStageFlags taskmesh_stages =
      radv_taskmesh_enabled(pdev) ? VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_TASK_BIT_EXT : 0;
   VkShaderStageFlags rt_stages = radv_enable_rt(pdev) ? RADV_RT_STAGE_BITS : 0;

   bool accel_dot = pdev->info.has_accelerated_dot_product;
   bool gfx11plus = pdev->info.gfx_level >= GFX11;

   VkExtent2D vrs_texel_extent = radv_vrs_attachment_enabled(pdev) ? (VkExtent2D){8, 8} : (VkExtent2D){0, 0};
   const int32_t max_viewport_size = pdev->info.gfx_level >= GFX12 ? 32768 : 16384;

   uint64_t os_page_size = 4096;
   os_get_page_size(&os_page_size);

   pdev->vk.properties = (struct vk_properties){
#ifdef ANDROID_STRICT
      .apiVersion = RADV_API_VERSION,
#else
      .apiVersion = pdev->info.gfx_level >= GFX8 ? RADV_API_VERSION : RADV_API_VERSION_1_3,
#endif
      .driverVersion = vk_get_driver_version(),
      .vendorID = ATI_VENDOR_ID,
      .deviceID = pdev->info.pci_id,
      .deviceType = device_type,
      .maxImageDimension1D = (1 << 14),
      .maxImageDimension2D = (1 << 14),
      .maxImageDimension3D = (1 << 11),
      .maxImageDimensionCube = (1 << 14),
      .maxImageArrayLayers = (1 << 11),
      .maxTexelBufferElements = UINT32_MAX,
      .maxUniformBufferRange = UINT32_MAX,
      .maxStorageBufferRange = UINT32_MAX,
      .maxPushConstantsSize = MAX_PUSH_CONSTANTS_SIZE,
      .maxMemoryAllocationCount = UINT32_MAX,
      .maxSamplerAllocationCount = 64 * 1024,
      .bufferImageGranularity = 1,
      .sparseAddressSpaceSize = RADV_MAX_MEMORY_ALLOCATION_SIZE, /* buffer max size */
      .maxBoundDescriptorSets = MAX_SETS,
      .maxPerStageDescriptorSamplers = max_descriptor_set_size,
      .maxPerStageDescriptorUniformBuffers = max_descriptor_set_size,
      .maxPerStageDescriptorStorageBuffers = max_descriptor_set_size,
      .maxPerStageDescriptorSampledImages = max_descriptor_set_size,
      .maxPerStageDescriptorStorageImages = max_descriptor_set_size,
      .maxPerStageDescriptorInputAttachments = max_descriptor_set_size,
      .maxPerStageResources = max_descriptor_set_size,
      .maxDescriptorSetSamplers = max_descriptor_set_size,
      .maxDescriptorSetUniformBuffers = max_descriptor_set_size,
      .maxDescriptorSetUniformBuffersDynamic = MAX_DYNAMIC_UNIFORM_BUFFERS,
      .maxDescriptorSetStorageBuffers = max_descriptor_set_size,
      .maxDescriptorSetStorageBuffersDynamic = MAX_DYNAMIC_STORAGE_BUFFERS,
      .maxDescriptorSetSampledImages = max_descriptor_set_size,
      .maxDescriptorSetStorageImages = max_descriptor_set_size,
      .maxDescriptorSetInputAttachments = max_descriptor_set_size,
      .maxVertexInputAttributes = MAX_VERTEX_ATTRIBS,
      .maxVertexInputBindings = MAX_VBS,
      .maxVertexInputAttributeOffset = UINT32_MAX,
      .maxVertexInputBindingStride = 2048,
      .maxVertexOutputComponents = 128,
      .maxTessellationGenerationLevel = 64,
      .maxTessellationPatchSize = 32,
      .maxTessellationControlPerVertexInputComponents = 128,
      .maxTessellationControlPerVertexOutputComponents = 128,
      .maxTessellationControlPerPatchOutputComponents = 120,
      .maxTessellationControlTotalOutputComponents = 4096,
      .maxTessellationEvaluationInputComponents = 128,
      .maxTessellationEvaluationOutputComponents = 128,
      .maxGeometryShaderInvocations = 32,
      .maxGeometryInputComponents = 64,
      .maxGeometryOutputComponents = 128,
      .maxGeometryOutputVertices = 256,
      .maxGeometryTotalOutputComponents = 1024,
      .maxFragmentInputComponents = 128,
      .maxFragmentOutputAttachments = 8,
      .maxFragmentDualSrcAttachments = 1,
      .maxFragmentCombinedOutputResources = max_descriptor_set_size,
      .maxComputeSharedMemorySize = pdev->max_shared_size,
      .maxComputeWorkGroupCount = {4294967295, 65535, 65535},
      .maxComputeWorkGroupInvocations = 1024,
      .maxComputeWorkGroupSize = {1024, 1024, 1024},
      .subPixelPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .mipmapPrecisionBits = 8,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT32_MAX,
      .maxSamplerLodBias = 16,
      .maxSamplerAnisotropy = 16,
      .maxViewports = MAX_VIEWPORTS,
      .maxViewportDimensions = {max_viewport_size, max_viewport_size},
      .viewportBoundsRange = {-2 * max_viewport_size, 2 * max_viewport_size - 1},
      .viewportSubPixelBits = 8,
      .minMemoryMapAlignment = 4096, /* A page */
      .minTexelBufferOffsetAlignment = 4,
      .minUniformBufferOffsetAlignment = radv_uniform_buffer_offset_alignment(pdev),
      .minStorageBufferOffsetAlignment = 4,
      .minTexelOffset = -32,
      .maxTexelOffset = 31,
      .minTexelGatherOffset = -32,
      .maxTexelGatherOffset = 31,
      .minInterpolationOffset = -2,
      .maxInterpolationOffset = 2,
      .subPixelInterpolationOffsetBits = 8,
      .maxFramebufferWidth = MAX_FRAMEBUFFER_WIDTH,
      .maxFramebufferHeight = MAX_FRAMEBUFFER_HEIGHT,
      .maxFramebufferLayers = (1 << 10),
      .framebufferColorSampleCounts = sample_counts,
      .framebufferDepthSampleCounts = sample_counts,
      .framebufferStencilSampleCounts = sample_counts,
      .framebufferNoAttachmentsSampleCounts = sample_counts,
      .maxColorAttachments = MAX_RTS,
      .sampledImageColorSampleCounts = sample_counts,
      .sampledImageIntegerSampleCounts = sample_counts,
      .sampledImageDepthSampleCounts = sample_counts,
      .sampledImageStencilSampleCounts = sample_counts,
      .storageImageSampleCounts = sample_counts,
      .maxSampleMaskWords = 1,
      .timestampComputeAndGraphics = true,
      .timestampPeriod = 1000000.0 / pdev->info.clock_crystal_freq,
      .maxClipDistances = 8,
      .maxCullDistances = 8,
      .maxCombinedClipAndCullDistances = 8,
      .discreteQueuePriorities = 2,
      .pointSizeRange = {0.0, 8191.875},
      .lineWidthRange = {0.0, 8.0},
      .pointSizeGranularity = (1.0 / 8.0),
      .lineWidthGranularity = (1.0 / 8.0),
      .strictLines = false, /* FINISHME */
      .standardSampleLocations = true,
      .optimalBufferCopyOffsetAlignment = 1,
      .optimalBufferCopyRowPitchAlignment = 1,
      .nonCoherentAtomSize = 64,
      .sparseResidencyNonResidentStrict = pdev->info.family >= CHIP_POLARIS10,
      .sparseResidencyStandard2DBlockShape = pdev->info.family >= CHIP_POLARIS10,
      .sparseResidencyStandard3DBlockShape = pdev->info.gfx_level >= GFX9,

      /* Vulkan 1.1 */
      .driverID = VK_DRIVER_ID_MESA_RADV,
      .deviceLUIDValid = false, /* The LUID is for Windows. */
      .deviceNodeMask = 0,
      .subgroupSize = RADV_SUBGROUP_SIZE,
      .subgroupSupportedStages =
         VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT | taskmesh_stages | rt_stages,
      .subgroupSupportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
                                     VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                     VK_SUBGROUP_FEATURE_CLUSTERED_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT |
                                     VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
                                     VK_SUBGROUP_FEATURE_ROTATE_BIT | VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT,
      .subgroupQuadOperationsInAllStages = true,
      .pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES,
      .maxMultiviewViewCount = MAX_VIEWS,
      .maxMultiviewInstanceIndex = INT_MAX,
      .protectedNoFault = false,
      .maxPerSetDescriptors = RADV_MAX_PER_SET_DESCRIPTORS,
      .maxMemoryAllocationSize = RADV_MAX_MEMORY_ALLOCATION_SIZE,

      /* Vulkan 1.2 */
      /* On AMD hardware, denormals and rounding modes for fp16/fp64 are
       * controlled by the same config register.
       */
      .denormBehaviorIndependence =
         has_fp16 ? VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY : VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
      .roundingModeIndependence =
         has_fp16 ? VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY : VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
      /* With LLVM, do not allow both preserving and flushing denorms because
       * different shaders in the same pipeline can have different settings and
       * this won't work for merged shaders. To make it work, this requires LLVM
       * support for changing the register. The same logic applies for the
       * rounding modes because they are configured with the same config
       * register.
       */
      .shaderDenormFlushToZeroFloat32 = true,
      .shaderDenormPreserveFloat32 = !pdev->use_llvm,
      .shaderRoundingModeRTEFloat32 = true,
      .shaderRoundingModeRTZFloat32 = !pdev->use_llvm,
      .shaderSignedZeroInfNanPreserveFloat32 = true,
      .shaderDenormFlushToZeroFloat16 = has_fp16 && !pdev->use_llvm,
      .shaderDenormPreserveFloat16 = has_fp16,
      .shaderRoundingModeRTEFloat16 = has_fp16,
      .shaderRoundingModeRTZFloat16 = has_fp16 && !pdev->use_llvm,
      .shaderSignedZeroInfNanPreserveFloat16 = has_fp16,
      .shaderDenormFlushToZeroFloat64 = pdev->info.gfx_level >= GFX8 && !pdev->use_llvm,
      .shaderDenormPreserveFloat64 = pdev->info.gfx_level >= GFX8,
      .shaderRoundingModeRTEFloat64 = pdev->info.gfx_level >= GFX8,
      .shaderRoundingModeRTZFloat64 = pdev->info.gfx_level >= GFX8 && !pdev->use_llvm,
      .shaderSignedZeroInfNanPreserveFloat64 = pdev->info.gfx_level >= GFX8,
      .maxUpdateAfterBindDescriptorsInAllPools = UINT32_MAX / 64,
      .shaderUniformBufferArrayNonUniformIndexingNative = false,
      .shaderSampledImageArrayNonUniformIndexingNative = false,
      .shaderStorageBufferArrayNonUniformIndexingNative = false,
      .shaderStorageImageArrayNonUniformIndexingNative = false,
      .shaderInputAttachmentArrayNonUniformIndexingNative = false,
      .robustBufferAccessUpdateAfterBind = true,
      .quadDivergentImplicitLod = false,
      .maxPerStageDescriptorUpdateAfterBindSamplers = max_descriptor_set_size,
      .maxPerStageDescriptorUpdateAfterBindUniformBuffers = max_descriptor_set_size,
      .maxPerStageDescriptorUpdateAfterBindStorageBuffers = max_descriptor_set_size,
      .maxPerStageDescriptorUpdateAfterBindSampledImages = max_descriptor_set_size,
      .maxPerStageDescriptorUpdateAfterBindStorageImages = max_descriptor_set_size,
      .maxPerStageDescriptorUpdateAfterBindInputAttachments = max_descriptor_set_size,
      .maxPerStageUpdateAfterBindResources = max_descriptor_set_size,
      .maxDescriptorSetUpdateAfterBindSamplers = max_descriptor_set_size,
      .maxDescriptorSetUpdateAfterBindUniformBuffers = max_descriptor_set_size,
      .maxDescriptorSetUpdateAfterBindUniformBuffersDynamic = MAX_DYNAMIC_UNIFORM_BUFFERS,
      .maxDescriptorSetUpdateAfterBindStorageBuffers = max_descriptor_set_size,
      .maxDescriptorSetUpdateAfterBindStorageBuffersDynamic = MAX_DYNAMIC_STORAGE_BUFFERS,
      .maxDescriptorSetUpdateAfterBindSampledImages = max_descriptor_set_size,
      .maxDescriptorSetUpdateAfterBindStorageImages = max_descriptor_set_size,
      .maxDescriptorSetUpdateAfterBindInputAttachments = max_descriptor_set_size,
      /* We support all of the depth resolve modes */
      .supportedDepthResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_AVERAGE_BIT |
                                    VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT,
      /* Average doesn't make sense for stencil so we don't support that */
      .supportedStencilResolveModes =
         VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT,
      .independentResolveNone = true,
      .independentResolve = true,
      /* GFX6-8 only support single channel min/max filter. */
      .filterMinmaxImageComponentMapping = pdev->info.gfx_level >= GFX9,
      .filterMinmaxSingleComponentFormats = true,
      .maxTimelineSemaphoreValueDifference = UINT64_MAX,
      .framebufferIntegerColorSampleCounts = VK_SAMPLE_COUNT_1_BIT,

      /* Vulkan 1.3 */
      .minSubgroupSize = pdev->info.gfx_level >= GFX10 ? 32 : 64,
      .maxSubgroupSize = 64,
      .maxComputeWorkgroupSubgroups = UINT32_MAX,
      .requiredSubgroupSizeStages = pdev->info.gfx_level >= GFX10 ? VK_SHADER_STAGE_COMPUTE_BIT | taskmesh_stages : 0,
      .maxInlineUniformBlockSize = MAX_INLINE_UNIFORM_BLOCK_SIZE,
      .maxPerStageDescriptorInlineUniformBlocks = MAX_INLINE_UNIFORM_BLOCK_SIZE * MAX_SETS,
      .maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = MAX_INLINE_UNIFORM_BLOCK_SIZE * MAX_SETS,
      .maxDescriptorSetInlineUniformBlocks = MAX_INLINE_UNIFORM_BLOCK_COUNT,
      .maxDescriptorSetUpdateAfterBindInlineUniformBlocks = MAX_INLINE_UNIFORM_BLOCK_COUNT,
      .maxInlineUniformTotalSize = UINT16_MAX,
      .integerDotProduct8BitUnsignedAccelerated = accel_dot,
      .integerDotProduct8BitSignedAccelerated = accel_dot,
      .integerDotProduct8BitMixedSignednessAccelerated = accel_dot && gfx11plus,
      .integerDotProduct4x8BitPackedUnsignedAccelerated = accel_dot,
      .integerDotProduct4x8BitPackedSignedAccelerated = accel_dot,
      .integerDotProduct4x8BitPackedMixedSignednessAccelerated = accel_dot && gfx11plus,
      .integerDotProduct16BitUnsignedAccelerated = accel_dot && !gfx11plus,
      .integerDotProduct16BitSignedAccelerated = accel_dot && !gfx11plus,
      .integerDotProduct16BitMixedSignednessAccelerated = false,
      .integerDotProduct32BitUnsignedAccelerated = false,
      .integerDotProduct32BitSignedAccelerated = false,
      .integerDotProduct32BitMixedSignednessAccelerated = false,
      .integerDotProduct64BitUnsignedAccelerated = false,
      .integerDotProduct64BitSignedAccelerated = false,
      .integerDotProduct64BitMixedSignednessAccelerated = false,
      .integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = accel_dot,
      .integerDotProductAccumulatingSaturating8BitSignedAccelerated = accel_dot,
      .integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = accel_dot && gfx11plus,
      .integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = accel_dot,
      .integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = accel_dot,
      .integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = accel_dot && gfx11plus,
      .integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = accel_dot && !gfx11plus,
      .integerDotProductAccumulatingSaturating16BitSignedAccelerated = accel_dot && !gfx11plus,
      .integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = false,
      .integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = false,
      .integerDotProductAccumulatingSaturating32BitSignedAccelerated = false,
      .integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = false,
      .integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = false,
      .integerDotProductAccumulatingSaturating64BitSignedAccelerated = false,
      .integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = false,
      .storageTexelBufferOffsetAlignmentBytes = 4,
      .storageTexelBufferOffsetSingleTexelAlignment = true,
      .uniformTexelBufferOffsetAlignmentBytes = 4,
      .uniformTexelBufferOffsetSingleTexelAlignment = true,
      .maxBufferSize = RADV_MAX_MEMORY_ALLOCATION_SIZE,

      /* Vulkan 1.4 */
      .lineSubPixelPrecisionBits = 4,
      .maxVertexAttribDivisor = UINT32_MAX,
      .supportsNonZeroFirstInstance = true,
      .maxPushDescriptors = MAX_PUSH_DESCRIPTORS,
      .dynamicRenderingLocalReadDepthStencilAttachments = true,
      .dynamicRenderingLocalReadMultisampledAttachments = true,
      .earlyFragmentMultisampleCoverageAfterSampleCounting = true,
      .earlyFragmentSampleMaskTestBeforeSampleCounting = true,
      .depthStencilSwizzleOneSupport = true,
      .polygonModePointSize = true,
      .nonStrictSinglePixelWideLinesUseParallelogram = true,
      .nonStrictWideLinesUseParallelogram = true,
      .blockTexelViewCompatibleMultipleLayers = true,
      .maxCombinedImageSamplerDescriptorCount = 3,
      .fragmentShadingRateClampCombinerInputs = true,
      .defaultRobustnessStorageBuffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
      .defaultRobustnessUniformBuffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
      .defaultRobustnessVertexInputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .defaultRobustnessImages = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2,

      /* VK_EXT_discard_rectangles */
      .maxDiscardRectangles = MAX_DISCARD_RECTANGLES,

      /* VK_EXT_external_memory_host */
      .minImportedHostPointerAlignment = 4096,

      /* VK_AMD_shader_core_properties */
      /* Shader engines. */
      .shaderEngineCount = pdev->info.max_se,
      .shaderArraysPerEngineCount = pdev->info.max_sa_per_se,
      .computeUnitsPerShaderArray = pdev->info.min_good_cu_per_sa,
      .simdPerComputeUnit = pdev->info.num_simd_per_compute_unit,
      .wavefrontsPerSimd = pdev->info.max_waves_per_simd,
      .wavefrontSize = 64,

      /* SGPR. */
      .sgprsPerSimd = pdev->info.num_physical_sgprs_per_simd,
      .minSgprAllocation = pdev->info.min_sgpr_alloc,
      .maxSgprAllocation = pdev->info.max_sgpr_alloc,
      .sgprAllocationGranularity = pdev->info.sgpr_alloc_granularity,

      /* VGPR. */
      .vgprsPerSimd = pdev->info.num_physical_wave64_vgprs_per_simd,
      .minVgprAllocation = pdev->info.min_wave64_vgpr_alloc,
      .maxVgprAllocation = pdev->info.max_vgpr_alloc,
      .vgprAllocationGranularity = pdev->info.wave64_vgpr_alloc_granularity,

      /* VK_AMD_shader_core_properties2 */
      .shaderCoreFeatures = 0,
      .activeComputeUnitCount = pdev->info.num_cu,

      /* VK_EXT_conservative_rasterization */
      .primitiveOverestimationSize = 0,
      .maxExtraPrimitiveOverestimationSize = 0,
      .extraPrimitiveOverestimationSizeGranularity = 0,
      .primitiveUnderestimation = true,
      .conservativePointAndLineRasterization = false,
      .degenerateTrianglesRasterized = true,
      .degenerateLinesRasterized = false,
      .fullyCoveredFragmentShaderInputVariable = true,
      .conservativeRasterizationPostDepthCoverage = false,

#ifndef _WIN32
      /* VK_EXT_pci_bus_info */
      .pciDomain = pdev->bus_info.domain,
      .pciBus = pdev->bus_info.bus,
      .pciDevice = pdev->bus_info.dev,
      .pciFunction = pdev->bus_info.func,
#endif

      /* VK_EXT_transform_feedback */
      .maxTransformFeedbackStreams = MAX_SO_STREAMS,
      .maxTransformFeedbackBuffers = MAX_SO_BUFFERS,
      .maxTransformFeedbackBufferSize = UINT32_MAX,
      .maxTransformFeedbackStreamDataSize = 512,
      .maxTransformFeedbackBufferDataSize = 512,
      .maxTransformFeedbackBufferDataStride = 512,
      .transformFeedbackQueries = true,
      .transformFeedbackStreamsLinesTriangles = true,
      .transformFeedbackRasterizationStreamSelect = false,
      .transformFeedbackDraw = true,

      /* VK_EXT_sample_locations */
      .sampleLocationSampleCounts = (pdev->info.gfx_level >= GFX10 ? VK_SAMPLE_COUNT_1_BIT : 0) |
                                    VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT,
      .maxSampleLocationGridSize = (VkExtent2D){2, 2},
      .sampleLocationCoordinateRange = {0.0f, 0.9375f},
      .sampleLocationSubPixelBits = 4,
      .variableSampleLocations = true,

      /* VK_KHR_robustness2 */
      .robustStorageBufferAccessSizeAlignment = 4,
      .robustUniformBufferAccessSizeAlignment = 4,

      /* VK_EXT_custom_border_color */
      .maxCustomBorderColorSamplers = RADV_BORDER_COLOR_COUNT,

      /* VK_KHR_fragment_shading_rate */
      .minFragmentShadingRateAttachmentTexelSize = vrs_texel_extent,
      .maxFragmentShadingRateAttachmentTexelSize = vrs_texel_extent,
      .maxFragmentShadingRateAttachmentTexelSizeAspectRatio = 1,
      .primitiveFragmentShadingRateWithMultipleViewports = true,
      .layeredShadingRateAttachments = false, /* TODO */
      .fragmentShadingRateNonTrivialCombinerOps = true,
      .maxFragmentSize = (VkExtent2D){2, 2},
      .maxFragmentSizeAspectRatio = 2,
      .maxFragmentShadingRateCoverageSamples = pdev->info.gfx_level >= GFX12 ? 16 : 32,
      .maxFragmentShadingRateRasterizationSamples =
         pdev->info.gfx_level >= GFX12 ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_8_BIT,
      .fragmentShadingRateWithShaderDepthStencilWrites = !pdev->info.has_vrs_ds_export_bug,
      .fragmentShadingRateWithSampleMask = true,
      .fragmentShadingRateWithShaderSampleMask = false,
      .fragmentShadingRateWithConservativeRasterization = true,
      .fragmentShadingRateWithFragmentShaderInterlock = pdev->info.gfx_level >= GFX11 && radv_has_pops(pdev),
      .fragmentShadingRateWithCustomSampleLocations = true,
      .fragmentShadingRateStrictMultiplyCombiner = true,

      /* VK_EXT_provoking_vertex */
      .provokingVertexModePerPipeline = true,
      .transformFeedbackPreservesTriangleFanProvokingVertex = true,

      /* VK_KHR_acceleration_structure */
      .maxGeometryCount = (1 << 24) - 1,
      .maxInstanceCount = (1 << 24) - 1,
      .maxPrimitiveCount = (1 << 29) - 1,
      .maxPerStageDescriptorAccelerationStructures = max_descriptor_set_size,
      .maxPerStageDescriptorUpdateAfterBindAccelerationStructures = max_descriptor_set_size,
      .maxDescriptorSetAccelerationStructures = max_descriptor_set_size,
      .maxDescriptorSetUpdateAfterBindAccelerationStructures = max_descriptor_set_size,
      /* Technically we can work with 128-byte alignment, but DOOM: The Dark Ages breaks if
       * the alignment is lower than this.
       */
      .minAccelerationStructureScratchOffsetAlignment = 256,

      /* VK_EXT_multi_draw */
      .maxMultiDrawCount = 2048,

      /* VK_KHR_ray_tracing_pipeline */
      .shaderGroupHandleSize = RADV_RT_HANDLE_SIZE,
      .maxRayRecursionDepth = 31,    /* Minimum allowed for DXR. */
      .maxShaderGroupStride = 16384, /* dummy */
      /* This isn't strictly necessary, but Doom Eternal breaks if the
       * alignment is any lower. */
      .shaderGroupBaseAlignment = RADV_RT_HANDLE_SIZE,
      .shaderGroupHandleCaptureReplaySize = sizeof(struct radv_rt_capture_replay_handle),
      .maxRayDispatchInvocationCount = 1024 * 1024 * 64,
      .shaderGroupHandleAlignment = 16,
      .maxRayHitAttributeSize = RADV_MAX_HIT_ATTRIB_SIZE,

      /* VK_KHR_performance_query */
      .allowCommandBufferQueryCopies = false,

      /* VK_EXT_graphics_pipeline_library */
      .graphicsPipelineLibraryFastLinking = true,
      .graphicsPipelineLibraryIndependentInterpolationDecoration = true,

      /* VK_EXT_mesh_shader */
      .maxTaskWorkGroupTotalCount = 4194304, /* 2^22 min required */
      .maxTaskWorkGroupCount = {65535, 65535, 65535},
      .maxTaskWorkGroupInvocations = 1024,
      .maxTaskWorkGroupSize = {1024, 1024, 1024},
      .maxTaskPayloadSize = 16384, /* 16K min required */
      .maxTaskSharedMemorySize = 65536,
      .maxTaskPayloadAndSharedMemorySize = 65536,

      .maxMeshWorkGroupTotalCount = 4194304, /* 2^22 min required */
      .maxMeshWorkGroupCount = {65535, 65535, 65535},
      .maxMeshWorkGroupInvocations = 256, /* Max NGG HW limit */
      .maxMeshWorkGroupSize = {256, 256, 256},
      .maxMeshOutputMemorySize = 32 * 1024,                   /* 32K min required */
      .maxMeshSharedMemorySize = 28672,                       /* 28K min required */
      .maxMeshPayloadAndSharedMemorySize = 16384 + 28672,     /* 28K min required */
      .maxMeshPayloadAndOutputMemorySize = 16384 + 32 * 1024, /* 47K min required */
      .maxMeshOutputComponents = 128,                         /* 32x vec4 min required */
      .maxMeshOutputVertices = 256,
      .maxMeshOutputPrimitives = 256,
      .maxMeshOutputLayers = 8,
      .maxMeshMultiviewViewCount = MAX_VIEWS,
      .meshOutputPerVertexGranularity = 1,
      .meshOutputPerPrimitiveGranularity = 1,

      .maxPreferredTaskWorkGroupInvocations = 64,
      .maxPreferredMeshWorkGroupInvocations = 128,
      .prefersLocalInvocationVertexOutput = true,
      .prefersLocalInvocationPrimitiveOutput = true,
      .prefersCompactVertexOutput = true,
      .prefersCompactPrimitiveOutput = false,

      /* VK_EXT_extended_dynamic_state3 */
      .dynamicPrimitiveTopologyUnrestricted = false,

      /* VK_EXT_descriptor_buffer */
      .combinedImageSamplerDescriptorSingleArray = true,
      .bufferlessPushDescriptors = true,
      .allowSamplerImageViewPostSubmitCreation = false,
      .descriptorBufferOffsetAlignment = 4,
      .maxDescriptorBufferBindings = MAX_SETS,
      .maxResourceDescriptorBufferBindings = MAX_SETS,
      .maxSamplerDescriptorBufferBindings = MAX_SETS,
      .maxEmbeddedImmutableSamplerBindings = MAX_SETS,
      .maxEmbeddedImmutableSamplers = radv_max_descriptor_set_size(),
      /* No data required for capture/replay (except for sparse buffers/images) but these values
       * need to be non-zero.
       */
      .bufferCaptureReplayDescriptorDataSize = 8,
      .imageCaptureReplayDescriptorDataSize = 8,
      .imageViewCaptureReplayDescriptorDataSize = 1,
      .samplerCaptureReplayDescriptorDataSize = 1,
      .accelerationStructureCaptureReplayDescriptorDataSize = 1,
      .samplerDescriptorSize = RADV_SAMPLER_DESC_SIZE,
      .combinedImageSamplerDescriptorSize = RADV_COMBINED_IMAGE_SAMPLER_DESC_SIZE,
      .sampledImageDescriptorSize = radv_get_sampled_image_desc_size(pdev),
      .storageImageDescriptorSize = RADV_STORAGE_IMAGE_DESC_SIZE,
      .uniformTexelBufferDescriptorSize = RADV_BUFFER_DESC_SIZE,
      .robustUniformTexelBufferDescriptorSize = RADV_BUFFER_DESC_SIZE,
      .storageTexelBufferDescriptorSize = RADV_BUFFER_DESC_SIZE,
      .robustStorageTexelBufferDescriptorSize = RADV_BUFFER_DESC_SIZE,
      .uniformBufferDescriptorSize = RADV_BUFFER_DESC_SIZE,
      .robustUniformBufferDescriptorSize = RADV_BUFFER_DESC_SIZE,
      .storageBufferDescriptorSize = RADV_BUFFER_DESC_SIZE,
      .robustStorageBufferDescriptorSize = RADV_BUFFER_DESC_SIZE,
      .inputAttachmentDescriptorSize = radv_get_sampled_image_desc_size(pdev),
      .accelerationStructureDescriptorSize = RADV_ACCEL_STRUCT_DESC_SIZE,
      .maxSamplerDescriptorBufferRange = UINT32_MAX,
      .maxResourceDescriptorBufferRange = UINT32_MAX,
      .samplerDescriptorBufferAddressSpaceSize = RADV_MAX_MEMORY_ALLOCATION_SIZE,
      .resourceDescriptorBufferAddressSpaceSize = RADV_MAX_MEMORY_ALLOCATION_SIZE,
      .descriptorBufferAddressSpaceSize = RADV_MAX_MEMORY_ALLOCATION_SIZE,

      /* VK_KHR_fragment_shader_barycentric */
      .triStripVertexOrderIndependentOfProvokingVertex = false,

      /* VK_EXT_pipeline_robustness */
      .defaultRobustnessStorageBuffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
      .defaultRobustnessUniformBuffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
      .defaultRobustnessVertexInputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .defaultRobustnessImages = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2,

      /* VK_KHR_cooperative_matrix */
      .cooperativeMatrixSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT,

      /* VK_EXT_map_memory_placed */
      .minPlacedMemoryMapAlignment = os_page_size,

      /* VK_EXT_nested_command_buffer */
      .maxCommandBufferNestingLevel = UINT32_MAX,

      /* VK_EXT_legacy_vertex_attributes */
      .nativeUnalignedPerformance = false,

      /* VK_MESA_image_alignment_control */
      .supportedImageAlignmentMask = (4 * 1024) | (64 * 1024) | (gfx11plus ? 256 * 1024 : 0),

      /* VK_KHR_maintenance7 */
      .robustFragmentShadingRateAttachmentAccess = true,
      .separateDepthStencilAttachmentAccess = true,
      .maxDescriptorSetTotalUniformBuffersDynamic = MAX_DYNAMIC_UNIFORM_BUFFERS,
      .maxDescriptorSetTotalStorageBuffersDynamic = MAX_DYNAMIC_STORAGE_BUFFERS,
      .maxDescriptorSetTotalBuffersDynamic = MAX_DYNAMIC_BUFFERS,
      .maxDescriptorSetUpdateAfterBindTotalUniformBuffersDynamic = MAX_DYNAMIC_UNIFORM_BUFFERS,
      .maxDescriptorSetUpdateAfterBindTotalStorageBuffersDynamic = MAX_DYNAMIC_STORAGE_BUFFERS,
      .maxDescriptorSetUpdateAfterBindTotalBuffersDynamic = MAX_DYNAMIC_BUFFERS,

      /* VK_KHR_pipeline_binary */
      .pipelineBinaryInternalCache = true,
      .pipelineBinaryInternalCacheControl = true,
      .pipelineBinaryPrefersInternalCache = false,
      .pipelineBinaryPrecompiledInternalCache = false,
      .pipelineBinaryCompressedData = false,

      /* VK_KHR_compute_shader_derivatives */
      .meshAndTaskShaderDerivatives = radv_taskmesh_enabled(pdev),

      /* VK_EXT_device_generated_commands */
      .maxIndirectPipelineCount = 4096,
      .maxIndirectShaderObjectCount = 4096,
      .maxIndirectSequenceCount = 1048576,
      .maxIndirectCommandsTokenCount = 128,
      .maxIndirectCommandsTokenOffset = 2047,
      .maxIndirectCommandsIndirectStride = 2048,
      .supportedIndirectCommandsInputModes = VK_INDIRECT_COMMANDS_INPUT_MODE_VULKAN_INDEX_BUFFER_EXT |
                                             VK_INDIRECT_COMMANDS_INPUT_MODE_DXGI_INDEX_BUFFER_EXT,
      .supportedIndirectCommandsShaderStages =
         VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT | taskmesh_stages | rt_stages,
      .supportedIndirectCommandsShaderStagesPipelineBinding = VK_SHADER_STAGE_COMPUTE_BIT,
      .supportedIndirectCommandsShaderStagesShaderBinding = VK_SHADER_STAGE_COMPUTE_BIT,
      .deviceGeneratedCommandsTransformFeedback = true,
      .deviceGeneratedCommandsMultiDrawIndirectCount = true,

      /* VK_KHR_maintenance9 */
      .image2DViewOf3DSparse = pdev->info.gfx_level >= GFX8,
      .defaultVertexAttributeValue = VK_DEFAULT_VERTEX_ATTRIBUTE_VALUE_ZERO_ZERO_ZERO_ZERO_KHR,
   };

   struct vk_properties *p = &pdev->vk.properties;

   strcpy(p->deviceName, pdev->marketing_name);
   memcpy(p->pipelineCacheUUID, pdev->cache_uuid, VK_UUID_SIZE);

   memcpy(p->deviceUUID, pdev->device_uuid, VK_UUID_SIZE);
   memcpy(p->driverUUID, pdev->driver_uuid, VK_UUID_SIZE);
   memset(p->deviceLUID, 0, VK_LUID_SIZE);

   snprintf(p->driverName, VK_MAX_DRIVER_NAME_SIZE, "radv");
   snprintf(p->driverInfo, VK_MAX_DRIVER_INFO_SIZE, "Mesa " PACKAGE_VERSION MESA_GIT_SHA1 "%s",
            radv_get_compiler_string(pdev));

   p->conformanceVersion = (VkConformanceVersion){
      .major = 1,
      .minor = 4,
      .subminor = 0,
      .patch = 0,
   };

   /* VK_EXT_host_image_copy */
   static const VkImageLayout supported_layouts[] = {
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PREINITIALIZED,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
      VK_IMAGE_LAYOUT_VIDEO_DECODE_SRC_KHR,
      VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
      VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR,
      VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR,
      VK_IMAGE_LAYOUT_VIDEO_ENCODE_DST_KHR,
      VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
      VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
      VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
      VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT,
   };

   p->copySrcLayoutCount = ARRAY_SIZE(supported_layouts);
   p->pCopySrcLayouts = (VkImageLayout *)supported_layouts;
   p->copyDstLayoutCount = ARRAY_SIZE(supported_layouts);
   p->pCopyDstLayouts = (VkImageLayout *)supported_layouts;
   memcpy(p->optimalTilingLayoutUUID, pdev->driver_uuid, VK_UUID_SIZE);
   p->identicalMemoryTypeRequirements = false;

   /* VK_EXT_physical_device_drm */
#ifndef _WIN32
   if (pdev->available_nodes & (1 << DRM_NODE_PRIMARY)) {
      p->drmHasPrimary = true;
      p->drmPrimaryMajor = (int64_t)major(pdev->primary_devid);
      p->drmPrimaryMinor = (int64_t)minor(pdev->primary_devid);
   } else {
      p->drmHasPrimary = false;
   }
   if (pdev->available_nodes & (1 << DRM_NODE_RENDER)) {
      p->drmHasRender = true;
      p->drmRenderMajor = (int64_t)major(pdev->render_devid);
      p->drmRenderMinor = (int64_t)minor(pdev->render_devid);
   } else {
      p->drmHasRender = false;
   }
#endif

   /* VK_EXT_shader_module_identifier */
   STATIC_ASSERT(sizeof(vk_shaderModuleIdentifierAlgorithmUUID) == sizeof(p->shaderModuleIdentifierAlgorithmUUID));
   memcpy(p->shaderModuleIdentifierAlgorithmUUID, vk_shaderModuleIdentifierAlgorithmUUID,
          sizeof(p->shaderModuleIdentifierAlgorithmUUID));

   /* VK_EXT_shader_object */
   radv_device_get_cache_uuid(pdev, p->shaderBinaryUUID);
   p->shaderBinaryVersion = 1;
}

static bool
radv_is_gpu_supported(const struct radeon_info *info)
{
   /* AMD CDNA isn't supported. */
   if (info->gfx_level == GFX9 && !info->has_graphics)
      return false;

   /* Unknown GPU generations aren't supported. */
   if (info->gfx_level > GFX12)
      return false;

   return true;
}

static VkResult
radv_physical_device_try_create(struct radv_instance *instance, drmDevicePtr drm_device,
                                struct radv_physical_device **pdev_out)
{
   VkResult result;
   int fd = -1;
   int master_fd = -1;

#ifdef _WIN32
   assert(drm_device == NULL);
#else
   bool is_virtio = false;
   if (drm_device) {
      const char *path = drm_device->nodes[DRM_NODE_RENDER];
      drmVersionPtr version;

      fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd < 0) {
         return vk_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER, "Could not open device %s: %m", path);
      }

      version = drmGetVersion(fd);
      if (!version) {
         close(fd);

         return vk_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                          "Could not get the kernel driver version for device %s: %m", path);
      }

      if (!strcmp(version->name, "amdgpu")) {
#ifdef HAVE_AMDGPU_VIRTIO
         if (debug_get_bool_option("AMD_FORCE_VPIPE", false)) {
            is_virtio = true;
            close(fd);
            fd = -1;
         }
#endif
      } else
#ifdef HAVE_AMDGPU_VIRTIO
         if (!strcmp(version->name, "virtio_gpu")) {
         is_virtio = true;
      } else
#endif
      {
         drmFreeVersion(version);
         close(fd);

         return vk_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                          "Device '%s' is not using the AMDGPU kernel driver: %m", path);
      }
      drmFreeVersion(version);

      if (instance->debug_flags & RADV_DEBUG_STARTUP)
         fprintf(stderr, "radv: info: Found device '%s'.\n", path);
   }
#endif

   struct radv_physical_device *pdev =
      vk_zalloc2(&instance->vk.alloc, NULL, sizeof(*pdev), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!pdev) {
      result = vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_fd;
   }

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(&dispatch_table, &radv_physical_device_entrypoints, true);
   vk_physical_device_dispatch_table_from_entrypoints(&dispatch_table, &wsi_physical_device_entrypoints, false);

   result = vk_physical_device_init(&pdev->vk, &instance->vk, NULL, NULL, NULL, &dispatch_table);
   if (result != VK_SUCCESS) {
      goto fail_alloc;
   }

#ifdef _WIN32
   pdev->ws = radv_null_winsys_create();
   if (!pdev->ws)
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
#else
   if (drm_device) {
      bool reserve_vmid = instance->vk.trace_mode & RADV_TRACE_MODE_RGP;

      result = radv_amdgpu_winsys_create(fd, instance->debug_flags, instance->perftest_flags, reserve_vmid, is_virtio,
                                         &pdev->ws);
   } else {
      pdev->ws = radv_null_winsys_create();
      if (!pdev->ws)
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
   }
#endif

   if (result != VK_SUCCESS) {
      result = vk_errorf(instance, result, "failed to initialize winsys");
      goto fail_base;
   }

   pdev->vk.supported_sync_types = pdev->ws->get_sync_types(pdev->ws);

#ifndef _WIN32
   if (drm_device && instance->vk.enabled_extensions.KHR_display) {
      master_fd = open(drm_device->nodes[DRM_NODE_PRIMARY], O_RDWR | O_CLOEXEC);
      if (master_fd >= 0) {
         uint32_t accel_working = 0;
         struct drm_amdgpu_info request = {.return_pointer = (uintptr_t)&accel_working,
                                           .return_size = sizeof(accel_working),
                                           .query = AMDGPU_INFO_ACCEL_WORKING};

         if (drm_ioctl_write(master_fd, DRM_AMDGPU_INFO, &request, sizeof(struct drm_amdgpu_info)) < 0 ||
             !accel_working) {
            close(master_fd);
            master_fd = -1;
         }
      }
   }
#endif

   pdev->master_fd = master_fd;
   pdev->local_fd = fd;
   pdev->ws->query_info(pdev->ws, &pdev->info);
   pdev->info.family_overridden = drm_device == NULL;

   /* Allow all devices on a virtual winsys, otherwise do a basic support check. */
   if (!radv_is_gpu_supported(&pdev->info) && drm_device) {
      if (instance->debug_flags & RADV_DEBUG_STARTUP)
         fprintf(stderr, "radv: info: device '%s' is not supported by RADV.\n", pdev->info.name);
      result = VK_ERROR_INCOMPATIBLE_DRIVER;
      goto fail_wsi;
   }

   if (drm_device) {
      pdev->addrlib = ac_addrlib_create(&pdev->info, &pdev->info.max_alignment);
      if (!pdev->addrlib) {
         result = VK_ERROR_INITIALIZATION_FAILED;
         goto fail_wsi;
      }
   }

   pdev->use_llvm = instance->debug_flags & RADV_DEBUG_LLVM;
#if !AMD_LLVM_AVAILABLE
   if (pdev->use_llvm) {
      fprintf(stderr, "ERROR: LLVM compiler backend selected for radv, but LLVM support was not "
                      "enabled at build time.\n");
      abort();
   }
#endif

#if DETECT_OS_ANDROID
   pdev->emulate_etc2 = !pdev->info.has_etc_support;
   pdev->emulate_astc = true;
#else
   pdev->emulate_etc2 = !pdev->info.has_etc_support && instance->drirc.vk_require_etc2;
   pdev->emulate_astc = instance->drirc.vk_require_astc;
#endif

   snprintf(pdev->name, sizeof(pdev->name), "AMD RADV %s%s", pdev->info.name, radv_get_compiler_string(pdev));

   const char *marketing_name = pdev->ws->get_chip_name(pdev->ws);
   snprintf(pdev->marketing_name, sizeof(pdev->name), "%s (RADV %s%s)", marketing_name ? marketing_name : "AMD Unknown",
            pdev->info.name, radv_get_compiler_string(pdev));

   if (pdev->info.gfx_level >= GFX12)
      vk_warn_non_conformant_implementation("radv");

   radv_get_driver_uuid(&pdev->driver_uuid);
   radv_get_device_uuid(&pdev->info, &pdev->device_uuid);

   pdev->dcc_msaa_allowed = (instance->perftest_flags & RADV_PERFTEST_DCC_MSAA);

   pdev->use_fmask = pdev->info.gfx_level < GFX11 && !(instance->debug_flags & RADV_DEBUG_NO_FMASK);

   pdev->use_hiz = !(instance->debug_flags & RADV_DEBUG_NO_HIZ);
   if (pdev->info.gfx_level == GFX12 && instance->drirc.disable_hiz_his_gfx12)
      pdev->use_hiz = false;

   pdev->use_gfx12_hiz_his_event_wa =
      pdev->info.gfx_level == GFX12 && pdev->use_hiz; /* TODO: Implement the alternative solution. */

   pdev->use_ngg = (pdev->info.gfx_level >= GFX10 && pdev->info.family != CHIP_NAVI14 &&
                    !(instance->debug_flags & RADV_DEBUG_NO_NGG)) ||
                   pdev->info.gfx_level >= GFX11;

   /* TODO: Investigate if NGG culling helps on GFX11. */
   pdev->use_ngg_culling = pdev->use_ngg && pdev->info.max_render_backends > 1 &&
                           (pdev->info.gfx_level == GFX10_3 || pdev->info.gfx_level == GFX10 ||
                            (instance->perftest_flags & RADV_PERFTEST_NGGC)) &&
                           !(instance->debug_flags & RADV_DEBUG_NO_NGGC);

   pdev->use_ngg_streamout = pdev->info.gfx_level >= GFX11;

   pdev->emulate_ngg_gs_query_pipeline_stat = pdev->use_ngg && pdev->info.gfx_level < GFX11;

   pdev->emulate_mesh_shader_queries = pdev->info.gfx_level == GFX10_3;

   /* Determine the number of threads per wave for all stages. */
   pdev->cs_wave_size = 64;
   pdev->ps_wave_size = 64;
   pdev->ge_wave_size = 64;
   pdev->rt_wave_size = 64;

   if (pdev->info.gfx_level >= GFX10) {
      if (instance->perftest_flags & RADV_PERFTEST_CS_WAVE_32)
         pdev->cs_wave_size = 32;

      /* For pixel shaders, wave64 is recommended. */
      if (instance->perftest_flags & RADV_PERFTEST_PS_WAVE_32)
         pdev->ps_wave_size = 32;

      if (instance->perftest_flags & RADV_PERFTEST_GE_WAVE_32)
         pdev->ge_wave_size = 32;

      /* Default to 32 on RDNA1-2 as that gives better perf due to less issues with divergence.
       * However, on RDNA3+ default to wave64 as implicit dual issuing is likely better than
       * wave32 VOPD for VALU dependent code.
       * (as well as the SALU count becoming more problematic with wave32)
       */
      if (instance->perftest_flags & RADV_PERFTEST_RT_WAVE_32 || pdev->info.gfx_level < GFX11)
         pdev->rt_wave_size = 32;

      if (instance->perftest_flags & RADV_PERFTEST_RT_WAVE_64)
         pdev->rt_wave_size = 64;
   }

   radv_probe_video_decode(pdev);
   radv_probe_video_encode(pdev);

   pdev->max_shared_size = pdev->info.gfx_level >= GFX7 ? 65536 : 32768;

   radv_physical_device_init_mem_types(pdev);

   radv_physical_device_get_supported_extensions(pdev, &pdev->vk.supported_extensions);
   radv_physical_device_get_features(pdev, &pdev->vk.supported_features);

   radv_get_nir_options(pdev);

#ifndef _WIN32
   if (drm_device) {
      struct stat primary_stat = {0}, render_stat = {0};

      pdev->available_nodes = drm_device->available_nodes;
      pdev->bus_info = *drm_device->businfo.pci;

      if ((drm_device->available_nodes & (1 << DRM_NODE_PRIMARY)) &&
          stat(drm_device->nodes[DRM_NODE_PRIMARY], &primary_stat) != 0) {
         result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED, "failed to stat DRM primary node %s",
                            drm_device->nodes[DRM_NODE_PRIMARY]);
         goto fail_perfcounters;
      }
      pdev->primary_devid = primary_stat.st_rdev;

      if ((drm_device->available_nodes & (1 << DRM_NODE_RENDER)) &&
          stat(drm_device->nodes[DRM_NODE_RENDER], &render_stat) != 0) {
         result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED, "failed to stat DRM render node %s",
                            drm_device->nodes[DRM_NODE_RENDER]);
         goto fail_perfcounters;
      }
      pdev->render_devid = render_stat.st_rdev;
   }
#endif

   radv_physical_device_init_cache_key(pdev);

   if (radv_device_get_cache_uuid(pdev, pdev->cache_uuid)) {
      result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED, "cannot generate UUID");
      goto fail_wsi;
   }

   /* The gpu id is already embedded in the uuid so we just pass "radv"
    * when creating the cache.
    */
   char buf[VK_UUID_SIZE * 2 + 1];
   mesa_bytes_to_hex(buf, pdev->cache_uuid, VK_UUID_SIZE);
   pdev->vk.disk_cache = disk_cache_create(pdev->name, buf, 0);

   pdev->disk_cache_meta = disk_cache_create_custom(pdev->name, buf, 0, "radv_builtin_shaders", 1024 * 32 /* 32MiB */);

   radv_get_physical_device_properties(pdev);

   if ((instance->debug_flags & RADV_DEBUG_INFO))
      ac_print_gpu_info(&pdev->info, stdout);

   radv_init_physical_device_decoder(pdev);
   radv_init_physical_device_encoder(pdev);

   radv_physical_device_init_queue_table(pdev);

   /* We don't check the error code, but later check if it is initialized. */
   ac_init_perfcounters(&pdev->info, false, false, &pdev->ac_perfcounters);

   /* The WSI is structured as a layer on top of the driver, so this has
    * to be the last part of initialization (at least until we get other
    * semi-layers).
    */
   result = radv_init_wsi(pdev);
   if (result != VK_SUCCESS) {
      vk_error(instance, result);
      goto fail_perfcounters;
   }

   pdev->gs_table_depth = ac_get_gs_table_depth(pdev->info.gfx_level, pdev->info.family);

   ac_get_task_info(&pdev->info, &pdev->task_info);
   radv_get_binning_settings(pdev, &pdev->binning_settings);

   if (pdev->info.has_distributed_tess) {
      if (pdev->info.family == CHIP_FIJI || pdev->info.family >= CHIP_POLARIS10)
         pdev->tess_distribution_mode = V_028B6C_TRAPEZOIDS;
      else
         pdev->tess_distribution_mode = V_028B6C_DONUTS;
   } else {
      pdev->tess_distribution_mode = V_028B6C_NO_DIST;
   }

   *pdev_out = pdev;

   return VK_SUCCESS;

fail_perfcounters:
   ac_destroy_perfcounters(&pdev->ac_perfcounters);
   disk_cache_destroy(pdev->vk.disk_cache);
   disk_cache_destroy(pdev->disk_cache_meta);
fail_wsi:
   if (pdev->addrlib)
      ac_addrlib_destroy(pdev->addrlib);
   pdev->ws->destroy(pdev->ws);
fail_base:
   vk_physical_device_finish(&pdev->vk);
fail_alloc:
   vk_free(&instance->vk.alloc, pdev);
fail_fd:
   if (fd != -1)
      close(fd);
   if (master_fd != -1)
      close(master_fd);
   return result;
}

VkResult
create_null_physical_device(struct vk_instance *vk_instance)
{
   struct radv_instance *instance = container_of(vk_instance, struct radv_instance, vk);
   struct radv_physical_device *pdev;

   VkResult result = radv_physical_device_try_create(instance, NULL, &pdev);
   if (result != VK_SUCCESS)
      return result;

   list_addtail(&pdev->vk.link, &instance->vk.physical_devices.list);
   return VK_SUCCESS;
}

VkResult
create_drm_physical_device(struct vk_instance *vk_instance, struct _drmDevice *device, struct vk_physical_device **out)
{
#ifndef _WIN32
   bool supported_device = false;

   if (!(device->available_nodes & (1 << DRM_NODE_RENDER)) || device->bustype != DRM_BUS_PCI)
      return VK_ERROR_INCOMPATIBLE_DRIVER;

#ifdef HAVE_AMDGPU_VIRTIO
   supported_device |= device->deviceinfo.pci->vendor_id == VIRTGPU_PCI_VENDOR_ID;
#endif

   supported_device |= device->deviceinfo.pci->vendor_id == ATI_VENDOR_ID;

   if (!supported_device)
      return VK_ERROR_INCOMPATIBLE_DRIVER;

   return radv_physical_device_try_create((struct radv_instance *)vk_instance, device,
                                          (struct radv_physical_device **)out);
#else
   return VK_SUCCESS;
#endif
}

void
radv_physical_device_destroy(struct vk_physical_device *vk_device)
{
   struct radv_physical_device *pdev = container_of(vk_device, struct radv_physical_device, vk);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   radv_finish_wsi(pdev);
   ac_destroy_perfcounters(&pdev->ac_perfcounters);
   if (pdev->addrlib)
      ac_addrlib_destroy(pdev->addrlib);
   pdev->ws->destroy(pdev->ws);
   disk_cache_destroy(pdev->vk.disk_cache);
   disk_cache_destroy(pdev->disk_cache_meta);
   if (pdev->local_fd != -1)
      close(pdev->local_fd);
   if (pdev->master_fd != -1)
      close(pdev->master_fd);
   vk_physical_device_finish(&pdev->vk);
   vk_free(&instance->vk.alloc, pdev);
}

static void
radv_get_physical_device_queue_family_properties(struct radv_physical_device *pdev, uint32_t *pCount,
                                                 VkQueueFamilyProperties **pQueueFamilyProperties)
{
   int num_queue_families = 0;
   int idx;

   if (radv_graphics_queue_enabled(pdev))
      num_queue_families++;

   if (radv_compute_queue_enabled(pdev))
      num_queue_families++;

   if (radv_video_decode_queue_enabled(pdev))
      num_queue_families++;

   if (radv_transfer_queue_enabled(pdev)) {
      num_queue_families++;
   }

   if (radv_video_encode_queue_enabled(pdev))
      num_queue_families++;

   if (radv_dedicated_sparse_queue_enabled(pdev)) {
      num_queue_families++;
   }

   if (pQueueFamilyProperties == NULL) {
      *pCount = num_queue_families;
      return;
   }

   if (!*pCount)
      return;

   idx = 0;
   if (radv_graphics_queue_enabled(pdev)) {
      if (*pCount >= 1) {
         VkQueueFlags gfx_flags =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
         *pQueueFamilyProperties[idx] = (VkQueueFamilyProperties){
            .queueFlags = gfx_flags,
            .queueCount = 1,
            .timestampValidBits = 64,
            .minImageTransferGranularity = (VkExtent3D){1, 1, 1},
         };
         idx++;
      }
   }

   if (radv_compute_queue_enabled(pdev)) {
      VkQueueFlags compute_flags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
      if (*pCount > idx) {
         *pQueueFamilyProperties[idx] = (VkQueueFamilyProperties){
            .queueFlags = compute_flags,
            .queueCount = pdev->info.ip[AMD_IP_COMPUTE].num_queues,
            .timestampValidBits = 64,
            .minImageTransferGranularity = (VkExtent3D){1, 1, 1},
         };
         idx++;
      }
   }

   if (radv_video_decode_queue_enabled(pdev)) {
      if (*pCount > idx) {
         *pQueueFamilyProperties[idx] = (VkQueueFamilyProperties){
            .queueFlags = VK_QUEUE_VIDEO_DECODE_BIT_KHR,
            .queueCount = pdev->info.ip[pdev->vid_decode_ip].num_queues,
            .timestampValidBits = 0,
            .minImageTransferGranularity = (VkExtent3D){1, 1, 1},
         };
         idx++;
      }
   }

   if (radv_transfer_queue_enabled(pdev)) {
      if (*pCount > idx) {
         *pQueueFamilyProperties[idx] = (VkQueueFamilyProperties){
            .queueFlags = VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT,
            .queueCount = pdev->info.ip[AMD_IP_SDMA].num_queues,
            .timestampValidBits = 64,
            .minImageTransferGranularity = (VkExtent3D){16, 16, 8},
         };
         idx++;
      }
   }

   if (radv_video_encode_queue_enabled(pdev)) {
      if (*pCount > idx) {
         *pQueueFamilyProperties[idx] = (VkQueueFamilyProperties){
            .queueFlags = VK_QUEUE_VIDEO_ENCODE_BIT_KHR,
            .queueCount = pdev->info.ip[AMD_IP_VCN_ENC].num_queues,
            .timestampValidBits = 0,
            .minImageTransferGranularity = (VkExtent3D){1, 1, 1},
         };
         idx++;
      }
   }

   if (radv_dedicated_sparse_queue_enabled(pdev)) {
      if (*pCount > idx) {
         *pQueueFamilyProperties[idx] = (VkQueueFamilyProperties){
            .queueFlags = VK_QUEUE_SPARSE_BINDING_BIT,
            .queueCount = 1,
            .timestampValidBits = 0,
            .minImageTransferGranularity = (VkExtent3D){1, 1, 1},
         };
         idx++;
      }
   }

   *pCount = idx;
}

static const VkQueueGlobalPriority radv_global_queue_priorities[] = {
   VK_QUEUE_GLOBAL_PRIORITY_LOW,
   VK_QUEUE_GLOBAL_PRIORITY_MEDIUM,
   VK_QUEUE_GLOBAL_PRIORITY_HIGH,
   VK_QUEUE_GLOBAL_PRIORITY_REALTIME,
};

VKAPI_ATTR void VKAPI_CALL
radv_GetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                             VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(radv_physical_device, pdev, physicalDevice);
   if (!pQueueFamilyProperties) {
      radv_get_physical_device_queue_family_properties(pdev, pCount, NULL);
      return;
   }
   VkQueueFamilyProperties *properties[] = {
      &pQueueFamilyProperties[0].queueFamilyProperties, &pQueueFamilyProperties[1].queueFamilyProperties,
      &pQueueFamilyProperties[2].queueFamilyProperties, &pQueueFamilyProperties[3].queueFamilyProperties,
      &pQueueFamilyProperties[4].queueFamilyProperties, &pQueueFamilyProperties[5].queueFamilyProperties,
   };
   radv_get_physical_device_queue_family_properties(pdev, pCount, properties);
   assert(*pCount <= 6);

   for (uint32_t i = 0; i < *pCount; i++) {
      vk_foreach_struct (ext, pQueueFamilyProperties[i].pNext) {
         switch (ext->sType) {
         case VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES: {
            VkQueueFamilyGlobalPriorityProperties *prop = (VkQueueFamilyGlobalPriorityProperties *)ext;
            STATIC_ASSERT(ARRAY_SIZE(radv_global_queue_priorities) <= VK_MAX_GLOBAL_PRIORITY_SIZE);
            prop->priorityCount = ARRAY_SIZE(radv_global_queue_priorities);
            memcpy(&prop->priorities, radv_global_queue_priorities, sizeof(radv_global_queue_priorities));
            break;
         }
         case VK_STRUCTURE_TYPE_QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES_KHR: {
            VkQueueFamilyQueryResultStatusPropertiesKHR *prop = (VkQueueFamilyQueryResultStatusPropertiesKHR *)ext;
            prop->queryResultStatusSupport = VK_FALSE;
            break;
         }
         case VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR: {
            VkQueueFamilyVideoPropertiesKHR *prop = (VkQueueFamilyVideoPropertiesKHR *)ext;
            prop->videoCodecOperations = 0;
            if (pQueueFamilyProperties[i].queueFamilyProperties.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) {
               if (VIDEO_CODEC_H264DEC)
                  prop->videoCodecOperations |= VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
               if (VIDEO_CODEC_H265DEC)
                  prop->videoCodecOperations |= VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
               if (VIDEO_CODEC_AV1DEC && pdev->info.vcn_ip_version >= VCN_3_0_0 &&
                   pdev->info.vcn_ip_version != VCN_3_0_33)
                  prop->videoCodecOperations |= VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
               if (VIDEO_CODEC_VP9DEC)
                  prop->videoCodecOperations |= VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
            }
            if (pQueueFamilyProperties[i].queueFamilyProperties.queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
               if (VIDEO_CODEC_H264ENC)
                  prop->videoCodecOperations |= VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
               if (VIDEO_CODEC_H265ENC)
                  prop->videoCodecOperations |= VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR;
               if (VIDEO_CODEC_AV1ENC && radv_video_encode_av1_supported(pdev))
                  prop->videoCodecOperations |= VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR;
            }
            break;
         }
         case VK_STRUCTURE_TYPE_QUEUE_FAMILY_OWNERSHIP_TRANSFER_PROPERTIES_KHR: {
            VkQueueFamilyOwnershipTransferPropertiesKHR *prop = (VkQueueFamilyOwnershipTransferPropertiesKHR *)ext;
            prop->optimalImageTransferToQueueFamilies = ~0;
            break;
         }
         default:
            break;
         }
      }
   }
}

static void
radv_get_memory_budget_properties(VkPhysicalDevice physicalDevice,
                                  VkPhysicalDeviceMemoryBudgetPropertiesEXT *memoryBudget)
{
   VK_FROM_HANDLE(radv_physical_device, pdev, physicalDevice);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   VkPhysicalDeviceMemoryProperties *memory_properties = &pdev->memory_properties;

   /* For all memory heaps, the computation of budget is as follow:
    *	heap_budget = heap_size - global_heap_usage + app_heap_usage
    *
    * The Vulkan spec 1.1.97 says that the budget should include any
    * currently allocated device memory.
    *
    * Note that the application heap usages are not really accurate (eg.
    * in presence of shared buffers).
    */
   if (!pdev->info.has_dedicated_vram) {
      if (instance->drirc.enable_unified_heap_on_apu) {
         /* When the heaps are unified, only the visible VRAM heap is exposed on APUs. */
         assert(pdev->heaps == RADV_HEAP_VRAM_VIS);
         assert(pdev->memory_properties.memoryHeaps[0].flags == VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
         const uint8_t vram_vis_heap_idx = 0;

         /* Get the total heap size which is the visible VRAM heap size. */
         uint64_t total_heap_size = pdev->memory_properties.memoryHeaps[vram_vis_heap_idx].size;

         /* Get the different memory usages. */
         uint64_t vram_vis_internal_usage = pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_VRAM_VIS) +
                                            pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_VRAM);
         uint64_t gtt_internal_usage = pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_GTT);
         uint64_t total_internal_usage = vram_vis_internal_usage + gtt_internal_usage;
         uint64_t total_system_usage =
            pdev->ws->query_value(pdev->ws, RADEON_VRAM_VIS_USAGE) + pdev->ws->query_value(pdev->ws, RADEON_GTT_USAGE);
         uint64_t total_usage = MAX2(total_internal_usage, total_system_usage);

         /* Compute the total free space that can be allocated for this process across all heaps. */
         uint64_t total_free_space = total_heap_size - MIN2(total_heap_size, total_usage);

         memoryBudget->heapBudget[vram_vis_heap_idx] = total_free_space + total_internal_usage;
         memoryBudget->heapUsage[vram_vis_heap_idx] = total_internal_usage;
      } else {
         /* On APUs, the driver exposes fake heaps to the application because usually the carveout
          * is too small for games but the budgets need to be redistributed accordingly.
          */
         assert(pdev->heaps == (RADV_HEAP_GTT | RADV_HEAP_VRAM_VIS));
         assert(pdev->memory_properties.memoryHeaps[0].flags == 0); /* GTT */
         assert(pdev->memory_properties.memoryHeaps[1].flags == VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
         const uint8_t gtt_heap_idx = 0, vram_vis_heap_idx = 1;

         /* Get the visible VRAM/GTT heap sizes and internal usages. */
         uint64_t gtt_heap_size = pdev->memory_properties.memoryHeaps[gtt_heap_idx].size;
         uint64_t vram_vis_heap_size = pdev->memory_properties.memoryHeaps[vram_vis_heap_idx].size;

         uint64_t vram_vis_internal_usage = pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_VRAM_VIS) +
                                            pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_VRAM);
         uint64_t gtt_internal_usage = pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_GTT);

         /* Compute the total heap size, internal and system usage. */
         uint64_t total_heap_size = vram_vis_heap_size + gtt_heap_size;
         uint64_t total_internal_usage = vram_vis_internal_usage + gtt_internal_usage;
         uint64_t total_system_usage =
            pdev->ws->query_value(pdev->ws, RADEON_VRAM_VIS_USAGE) + pdev->ws->query_value(pdev->ws, RADEON_GTT_USAGE);

         uint64_t total_usage = MAX2(total_internal_usage, total_system_usage);

         /* Compute the total free space that can be allocated for this process across all heaps. */
         uint64_t total_free_space = total_heap_size - MIN2(total_heap_size, total_usage);

         /* Compute the remaining visible VRAM size for this process. */
         uint64_t vram_vis_free_space = vram_vis_heap_size - MIN2(vram_vis_heap_size, vram_vis_internal_usage);

         /* Distribute the total free space (2/3rd as VRAM and 1/3rd as GTT) to match the heap
          * sizes, and align down to the page size to be conservative.
          */
         vram_vis_free_space =
            ROUND_DOWN_TO(MIN2((total_free_space * 2) / 3, vram_vis_free_space), pdev->info.gart_page_size);
         uint64_t gtt_free_space = total_free_space - vram_vis_free_space;

         memoryBudget->heapBudget[vram_vis_heap_idx] = vram_vis_free_space + vram_vis_internal_usage;
         memoryBudget->heapUsage[vram_vis_heap_idx] = vram_vis_internal_usage;
         memoryBudget->heapBudget[gtt_heap_idx] = gtt_free_space + gtt_internal_usage;
         memoryBudget->heapUsage[gtt_heap_idx] = gtt_internal_usage;
      }
   } else {
      unsigned mask = pdev->heaps;
      unsigned heap = 0;
      while (mask) {
         uint64_t internal_usage = 0, system_usage = 0;
         unsigned type = 1u << u_bit_scan(&mask);

         switch (type) {
         case RADV_HEAP_VRAM:
            internal_usage = pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_VRAM);
            system_usage = pdev->ws->query_value(pdev->ws, RADEON_VRAM_USAGE);
            break;
         case RADV_HEAP_VRAM_VIS:
            internal_usage = pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_VRAM_VIS);
            if (!(pdev->heaps & RADV_HEAP_VRAM))
               internal_usage += pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_VRAM);
            system_usage = pdev->ws->query_value(pdev->ws, RADEON_VRAM_VIS_USAGE);
            break;
         case RADV_HEAP_GTT:
            internal_usage = pdev->ws->query_value(pdev->ws, RADEON_ALLOCATED_GTT);
            system_usage = pdev->ws->query_value(pdev->ws, RADEON_GTT_USAGE);
            break;
         }

         uint64_t total_usage = MAX2(internal_usage, system_usage);

         uint64_t free_space = pdev->memory_properties.memoryHeaps[heap].size -
                               MIN2(pdev->memory_properties.memoryHeaps[heap].size, total_usage);
         memoryBudget->heapBudget[heap] = free_space + internal_usage;
         memoryBudget->heapUsage[heap] = internal_usage;
         ++heap;
      }

      assert(heap == memory_properties->memoryHeapCount);
   }

   /* The heapBudget value must be less than or equal to VkMemoryHeap::size for each heap. */
   for (uint32_t i = 0; i < memory_properties->memoryHeapCount; i++) {
      memoryBudget->heapBudget[i] = MIN2(memory_properties->memoryHeaps[i].size, memoryBudget->heapBudget[i]);
   }

   /* The heapBudget and heapUsage values must be zero for array elements
    * greater than or equal to
    * VkPhysicalDeviceMemoryProperties::memoryHeapCount.
    */
   for (uint32_t i = memory_properties->memoryHeapCount; i < VK_MAX_MEMORY_HEAPS; i++) {
      memoryBudget->heapBudget[i] = 0;
      memoryBudget->heapUsage[i] = 0;
   }
}

VKAPI_ATTR void VKAPI_CALL
radv_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
                                        VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(radv_physical_device, pdev, physicalDevice);

   pMemoryProperties->memoryProperties = pdev->memory_properties;

   VkPhysicalDeviceMemoryBudgetPropertiesEXT *memory_budget =
      vk_find_struct(pMemoryProperties->pNext, PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT);
   if (memory_budget)
      radv_get_memory_budget_properties(physicalDevice, memory_budget);
}

VKAPI_ATTR void VKAPI_CALL
radv_GetPhysicalDeviceMultisamplePropertiesEXT(VkPhysicalDevice physicalDevice, VkSampleCountFlagBits samples,
                                               VkMultisamplePropertiesEXT *pMultisampleProperties)
{
   VK_FROM_HANDLE(radv_physical_device, pdev, physicalDevice);

   VkSampleCountFlagBits supported_samples = VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT;
   if (pdev->info.gfx_level >= GFX10)
      supported_samples |= VK_SAMPLE_COUNT_1_BIT;

   if (samples & supported_samples) {
      pMultisampleProperties->maxSampleLocationGridSize = (VkExtent2D){2, 2};
   } else {
      pMultisampleProperties->maxSampleLocationGridSize = (VkExtent2D){0, 0};
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPhysicalDeviceFragmentShadingRatesKHR(VkPhysicalDevice physicalDevice, uint32_t *pFragmentShadingRateCount,
                                              VkPhysicalDeviceFragmentShadingRateKHR *pFragmentShadingRates)
{
   VK_FROM_HANDLE(radv_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkPhysicalDeviceFragmentShadingRateKHR, out, pFragmentShadingRates,
                          pFragmentShadingRateCount);

#define append_rate(w, h, s)                                                                                           \
   {                                                                                                                   \
      VkPhysicalDeviceFragmentShadingRateKHR rate = {                                                                  \
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR,                              \
         .sampleCounts = s,                                                                                            \
         .fragmentSize = {.width = w, .height = h},                                                                    \
      };                                                                                                               \
      vk_outarray_append_typed(VkPhysicalDeviceFragmentShadingRateKHR, &out, r) *r = rate;                             \
   }

   for (uint32_t x = 2; x >= 1; x--) {
      for (uint32_t y = 2; y >= 1; y--) {
         VkSampleCountFlagBits samples;

         if (x == 1 && y == 1) {
            samples = ~0;
         } else {
            samples = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT;

            /* VRS coarse shading with 8x MSAA isn't supported on GFX12 and the
             * hw automatically clamps to 1x1.
             */
            if (pdev->info.gfx_level < GFX12)
               samples |= VK_SAMPLE_COUNT_8_BIT;
         }

         append_rate(x, y, samples);
      }
   }
#undef append_rate

   return vk_outarray_status(&out);
}

/* VK_EXT_tooling_info */
VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPhysicalDeviceToolProperties(VkPhysicalDevice physicalDevice, uint32_t *pToolCount,
                                     VkPhysicalDeviceToolProperties *pToolProperties)
{
   VK_FROM_HANDLE(radv_physical_device, pdev, physicalDevice);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);
   VK_OUTARRAY_MAKE_TYPED(VkPhysicalDeviceToolProperties, out, pToolProperties, pToolCount);
   bool rgp_enabled, rmv_enabled, rra_enabled;
   uint32_t tool_count = 0;

   /* RGP */
   rgp_enabled = instance->vk.trace_mode & RADV_TRACE_MODE_RGP;
   if (rgp_enabled)
      tool_count++;

   /* RMV */
   rmv_enabled = instance->vk.trace_mode & VK_TRACE_MODE_RMV;
   if (rmv_enabled)
      tool_count++;

   /* RRA */
   rra_enabled = instance->vk.trace_mode & RADV_TRACE_MODE_RRA;
   if (rra_enabled)
      tool_count++;

   if (!pToolProperties) {
      *pToolCount = tool_count;
      return VK_SUCCESS;
   }

   if (rgp_enabled) {
      VkPhysicalDeviceToolProperties tool = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES,
         .name = "Radeon GPU Profiler",
         .version = "1.15",
         .description = "A ground-breaking low-level optimization tool that provides detailed "
                        "information on Radeon GPUs.",
         .purposes = VK_TOOL_PURPOSE_PROFILING_BIT | VK_TOOL_PURPOSE_TRACING_BIT |
                     /* VK_EXT_debug_marker is only exposed if SQTT is enabled. */
                     VK_TOOL_PURPOSE_ADDITIONAL_FEATURES_BIT | VK_TOOL_PURPOSE_DEBUG_MARKERS_BIT_EXT,
      };
      vk_outarray_append_typed(VkPhysicalDeviceToolProperties, &out, t) *t = tool;
   }

   if (rmv_enabled) {
      VkPhysicalDeviceToolProperties tool = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES,
         .name = "Radeon Memory Visualizer",
         .version = "1.6",
         .description = "A tool to allow you to gain a deep understanding of how your application "
                        "uses memory for graphics resources.",
         .purposes = VK_TOOL_PURPOSE_PROFILING_BIT | VK_TOOL_PURPOSE_TRACING_BIT,
      };
      vk_outarray_append_typed(VkPhysicalDeviceToolProperties, &out, t) *t = tool;
   }

   if (rra_enabled) {
      VkPhysicalDeviceToolProperties tool = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES,
         .name = "Radeon Raytracing Analyzer",
         .version = "1.2",
         .description = "A tool to investigate the performance of your ray tracing applications and "
                        "highlight potential bottlenecks.",
         .purposes = VK_TOOL_PURPOSE_PROFILING_BIT | VK_TOOL_PURPOSE_TRACING_BIT,
      };
      vk_outarray_append_typed(VkPhysicalDeviceToolProperties, &out, t) *t = tool;
   }

   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPhysicalDeviceCooperativeMatrixPropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                                     VkCooperativeMatrixPropertiesKHR *pProperties)
{
   VK_FROM_HANDLE(radv_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkCooperativeMatrixPropertiesKHR, out, pProperties, pPropertyCount);

   if (pdev->info.gfx_level >= GFX12) {
      for (unsigned e5m2_a = 0; e5m2_a < 2; e5m2_a++) {
         for (unsigned e5m2_b = 0; e5m2_b < 2; e5m2_b++) {
            VkComponentTypeKHR a_type = e5m2_a ? VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT : VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT;
            VkComponentTypeKHR b_type = e5m2_b ? VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT : VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT;

            vk_outarray_append_typed(VkCooperativeMatrixPropertiesKHR, &out, p)
            {
               *p = (struct VkCooperativeMatrixPropertiesKHR){
                  .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                  .MSize = 16,
                  .NSize = 16,
                  .KSize = 16,
                  .AType = a_type,
                  .BType = b_type,
                  .CType = VK_COMPONENT_TYPE_FLOAT32_KHR,
                  .ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR,
                  .saturatingAccumulation = false,
                  .scope = VK_SCOPE_SUBGROUP_KHR};
            }
         }
      }
   }

   for (unsigned bfloat = 0; bfloat < 2; bfloat++) {
      for (unsigned fp32 = 0; fp32 < 2; fp32++) {
         VkComponentTypeKHR ab_type = bfloat ? VK_COMPONENT_TYPE_BFLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT16_KHR;
         VkComponentTypeKHR cd_type = fp32 ? VK_COMPONENT_TYPE_FLOAT32_KHR : ab_type;

         if (pdev->info.gfx_level < GFX12 && bfloat)
            continue; /* BF16 isn't working precisely on GFX11. */

         vk_outarray_append_typed(VkCooperativeMatrixPropertiesKHR, &out, p)
         {
            *p = (struct VkCooperativeMatrixPropertiesKHR){.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                                                           .MSize = 16,
                                                           .NSize = 16,
                                                           .KSize = 16,
                                                           .AType = ab_type,
                                                           .BType = ab_type,
                                                           .CType = cd_type,
                                                           .ResultType = cd_type,
                                                           .saturatingAccumulation = false,
                                                           .scope = VK_SCOPE_SUBGROUP_KHR};
         }
      }
   }

   for (unsigned asigned = 0; asigned < 2; asigned++) {
      for (unsigned bsigned = 0; bsigned < 2; bsigned++) {
         for (unsigned csigned = 0; csigned < 2; csigned++) {
            for (unsigned saturate = 0; saturate < 2; saturate++) {
               if (!csigned && saturate)
                  continue; /* The HW only supports signed acc. */
               vk_outarray_append_typed(VkCooperativeMatrixPropertiesKHR, &out, p)
               {
                  *p = (struct VkCooperativeMatrixPropertiesKHR){
                     .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                     .MSize = 16,
                     .NSize = 16,
                     .KSize = 16,
                     .AType = asigned ? VK_COMPONENT_TYPE_SINT8_KHR : VK_COMPONENT_TYPE_UINT8_KHR,
                     .BType = bsigned ? VK_COMPONENT_TYPE_SINT8_KHR : VK_COMPONENT_TYPE_UINT8_KHR,
                     .CType = csigned ? VK_COMPONENT_TYPE_SINT32_KHR : VK_COMPONENT_TYPE_UINT32_KHR,
                     .ResultType = csigned ? VK_COMPONENT_TYPE_SINT32_KHR : VK_COMPONENT_TYPE_UINT32_KHR,
                     .saturatingAccumulation = saturate,
                     .scope = VK_SCOPE_SUBGROUP_KHR};
               }
            }
         }
      }
   }

   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_GetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(
   VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
   VkCooperativeMatrixFlexibleDimensionsPropertiesNV *pProperties)
{
   *pPropertyCount = 0;
   return VK_SUCCESS;
}
