/*
 * Copyright 2018 Collabora Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "zink_screen.h"

#include "zink_kopper.h"
#include "zink_compiler.h"
#include "zink_context.h"
#include "zink_descriptors.h"
#include "zink_fence.h"
#include "vk_format.h"
#include "zink_format.h"
#include "zink_program.h"
#include "zink_public.h"
#include "zink_query.h"
#include "zink_resource.h"
#include "zink_state.h"
#include "nir_to_spirv/nir_to_spirv.h" // for SPIRV_VERSION

#include "util/u_debug.h"
#include "util/u_dl.h"
#include "util/os_file.h"
#include "util/u_memory.h"
#include "util/u_screen.h"
#include "util/u_string.h"
#include "util/perf/u_trace.h"
#include "util/u_transfer_helper.h"
#include "util/hex.h"
#include "util/xmlconfig.h"

#include "util/u_cpu_detect.h"

#ifdef HAVE_LIBDRM
#include <xf86drm.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef MAJOR_IN_MKDEV
#include <sys/mkdev.h>
#endif
#ifdef MAJOR_IN_SYSMACROS
#include <sys/sysmacros.h>
#endif
#endif

static int num_screens = 0;
bool zink_tracing = false;

#if DETECT_OS_WINDOWS
#include <io.h>
#define VK_LIBNAME "vulkan-1.dll"
#else
#include <unistd.h>
#if DETECT_OS_APPLE
#define VK_LIBNAME "libvulkan.1.dylib"
#elif DETECT_OS_ANDROID
#define VK_LIBNAME "libvulkan.so"
#else
#define VK_LIBNAME "libvulkan.so.1"
#endif
#endif

#ifdef __APPLE__
#include "MoltenVK/mvk_vulkan.h"
// Source of MVK_VERSION
#include "MoltenVK/mvk_config.h"
#define VK_NO_PROTOTYPES
#include "MoltenVK/mvk_deprecated_api.h"
#include "MoltenVK/mvk_private_api.h"
#endif /* __APPLE__ */

#ifdef HAVE_LIBDRM
#include "drm-uapi/dma-buf.h"
#include <xf86drm.h>
#endif

static const struct debug_named_value
zink_debug_options[] = {
   { "nir", ZINK_DEBUG_NIR, "Dump NIR during program compile" },
   { "spirv", ZINK_DEBUG_SPIRV, "Dump SPIR-V during program compile" },
   { "tgsi", ZINK_DEBUG_TGSI, "Dump TGSI during program compile" },
   { "validation", ZINK_DEBUG_VALIDATION, "Dump Validation layer output" },
   { "vvl", ZINK_DEBUG_VALIDATION, "Dump Validation layer output" },
   { "sync", ZINK_DEBUG_SYNC, "Force synchronization before draws/dispatches" },
   { "compact", ZINK_DEBUG_COMPACT, "Use only 4 descriptor sets" },
   { "noreorder", ZINK_DEBUG_NOREORDER, "Do not reorder command streams" },
   { "gpl", ZINK_DEBUG_GPL, "Force using Graphics Pipeline Library for all shaders" },
   { "shaderdb", ZINK_DEBUG_SHADERDB, "Do stuff to make shader-db work" },
   { "rp", ZINK_DEBUG_RP, "Enable renderpass tracking/optimizations" },
   { "norp", ZINK_DEBUG_NORP, "Disable renderpass tracking/optimizations" },
   { "map", ZINK_DEBUG_MAP, "Track amount of mapped VRAM" },
   { "flushsync", ZINK_DEBUG_FLUSHSYNC, "Force synchronous flushes/presents" },
   { "noshobj", ZINK_DEBUG_NOSHOBJ, "Disable EXT_shader_object" },
   { "optimal_keys", ZINK_DEBUG_OPTIMAL_KEYS, "Debug/use optimal_keys" },
   { "noopt", ZINK_DEBUG_NOOPT, "Disable async optimized pipeline compiles" },
   { "nobgc", ZINK_DEBUG_NOBGC, "Disable all async pipeline compiles" },
   { "mem", ZINK_DEBUG_MEM, "Debug memory allocations" },
   { "quiet", ZINK_DEBUG_QUIET, "Suppress warnings" },
   { "nopc", ZINK_DEBUG_NOPC, "No precompilation" },
   { "msaaopt", ZINK_DEBUG_MSAAOPT, "Optimize out loads/stores of MSAA attachments" },
   DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(zink_debug, "ZINK_DEBUG", zink_debug_options, 0)

uint32_t
zink_debug;


static const struct debug_named_value
zink_descriptor_options[] = {
   { "auto", ZINK_DESCRIPTOR_MODE_AUTO, "Automatically detect best mode" },
   { "lazy", ZINK_DESCRIPTOR_MODE_LAZY, "Don't cache, do least amount of updates" },
   { "db", ZINK_DESCRIPTOR_MODE_DB, "Use descriptor buffers" },
   DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(zink_descriptor_mode, "ZINK_DESCRIPTORS", zink_descriptor_options, ZINK_DESCRIPTOR_MODE_AUTO)

enum zink_descriptor_mode zink_descriptor_mode;

struct zink_device {
   unsigned refcount;
   VkPhysicalDevice pdev;
   VkDevice dev;
   struct zink_device_info *info;
};

static simple_mtx_t device_lock = SIMPLE_MTX_INITIALIZER;
static struct set device_table;

static simple_mtx_t instance_lock = SIMPLE_MTX_INITIALIZER;
static struct zink_instance_info instance_info;
static unsigned instance_refcount;
static VkInstance instance;

static const char *
zink_get_vendor(struct pipe_screen *pscreen)
{
   return "Mesa";
}

static const char *
zink_get_device_vendor(struct pipe_screen *pscreen)
{
   return zink_screen(pscreen)->vendor_name;
}

static const char *
zink_get_name(struct pipe_screen *pscreen)
{
   return zink_screen(pscreen)->device_name;
}

static int
zink_set_driver_strings(struct zink_screen *screen)
{
   char buf[1000];
   const char *driver_name = vk_DriverId_to_str(zink_driverid(screen)) + strlen("VK_DRIVER_ID_");
   int written = snprintf(buf, sizeof(buf), "zink Vulkan %d.%d(%s (%s))",
      VK_VERSION_MAJOR(screen->info.device_version),
      VK_VERSION_MINOR(screen->info.device_version),
      screen->info.props.deviceName,
      strstr(vk_DriverId_to_str(zink_driverid(screen)), "VK_DRIVER_ID_") ? driver_name : "Driver Unknown"
   );
   if (written < 0)
      return written;
   assert(written < sizeof(buf));
   screen->device_name = ralloc_strdup(screen, buf);

   written = snprintf(buf, sizeof(buf), "Unknown (vendor-id: 0x%04x)", screen->info.props.vendorID);
   if (written < 0)
      return written;
   assert(written < sizeof(buf));
   screen->vendor_name = ralloc_strdup(screen, buf);
   return 0;
}

static void
zink_get_driver_uuid(struct pipe_screen *pscreen, char *uuid)
{
   struct zink_screen *screen = zink_screen(pscreen);
   if (screen->vk_version >= VK_MAKE_VERSION(1,2,0)) {
      memcpy(uuid, screen->info.props11.driverUUID, VK_UUID_SIZE);
   } else {
      memcpy(uuid, screen->info.deviceid_props.driverUUID, VK_UUID_SIZE);
   }
}

static void
zink_get_device_uuid(struct pipe_screen *pscreen, char *uuid)
{
   struct zink_screen *screen = zink_screen(pscreen);
   if (screen->vk_version >= VK_MAKE_VERSION(1,2,0)) {
      memcpy(uuid, screen->info.props11.deviceUUID, VK_UUID_SIZE);
   } else {
      memcpy(uuid, screen->info.deviceid_props.deviceUUID, VK_UUID_SIZE);
   }
}

static void
zink_get_device_luid(struct pipe_screen *pscreen, char *luid)
{
   struct zink_screen *screen = zink_screen(pscreen);
   if (screen->info.have_vulkan12) {
      memcpy(luid, screen->info.props11.deviceLUID, VK_LUID_SIZE);
   } else {
      memcpy(luid, screen->info.deviceid_props.deviceLUID, VK_LUID_SIZE);
   }
}

static uint32_t
zink_get_device_node_mask(struct pipe_screen *pscreen)
{
   struct zink_screen *screen = zink_screen(pscreen);
   if (screen->info.have_vulkan12) {
      return screen->info.props11.deviceNodeMask;
   } else {
      return screen->info.deviceid_props.deviceNodeMask;
   }
}

static void
zink_set_max_shader_compiler_threads(struct pipe_screen *pscreen, unsigned max_threads)
{
   struct zink_screen *screen = zink_screen(pscreen);
   util_queue_adjust_num_threads(&screen->cache_get_thread, max_threads, false);
}

static bool
zink_is_parallel_shader_compilation_finished(struct pipe_screen *screen, void *shader, enum pipe_shader_type shader_type)
{
   if (shader_type == MESA_SHADER_COMPUTE) {
      struct zink_program *pg = shader;
      return !pg->can_precompile || util_queue_fence_is_signalled(&pg->cache_fence);
   }

   struct zink_shader *zs = shader;
   if (!util_queue_fence_is_signalled(&zs->precompile.fence))
      return false;
   bool finished = true;
   set_foreach(zs->programs, entry) {
      struct zink_gfx_program *prog = (void*)entry->key;
      finished &= util_queue_fence_is_signalled(&prog->base.cache_fence);
   }
   return finished;
}

static VkDeviceSize
get_video_mem(struct zink_screen *screen)
{
   VkDeviceSize size = 0;
   for (uint32_t i = 0; i < screen->info.mem_props.memoryHeapCount; ++i) {
      if (screen->info.mem_props.memoryHeaps[i].flags &
          VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
         size += screen->info.mem_props.memoryHeaps[i].size;
   }
   return size;
}

/**
 * Creates the disk cache used by mesa/st frontend for caching the GLSL -> NIR
 * path.
 *
 * The output that gets stored in the frontend's cache is the result of
 * zink_shader_finalize().  So, our blake3 cache key here needs to include
 * everything that would change the NIR we generate from a given set of GLSL
 * source, including our driver build, the Vulkan device and driver (which could
 * affect the pipe caps we show the frontend), and any debug flags that change
 * codegen.
 *
 * This disk cache also gets used by zink itself for storing its output from NIR
 * -> SPIRV translation.
 */
static bool
disk_cache_init(struct zink_screen *screen)
{
   if (zink_debug & ZINK_DEBUG_SHADERDB)
      return true;

#ifdef ENABLE_SHADER_CACHE
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);

#ifdef HAVE_DL_ITERATE_PHDR
   /* Hash in the zink driver build. */
   const struct build_id_note *note =
       build_id_find_nhdr_for_addr(disk_cache_init);
   unsigned build_id_len = build_id_length(note);
   assert(note && build_id_len == 20); /* blake3 */
   _mesa_blake3_update(&ctx, build_id_data(note), build_id_len);
#endif

   /* Hash in the Vulkan pipeline cache UUID to identify the combination of
   *  vulkan device and driver (or any inserted layer that would invalidate our
   *  cached pipelines).
   *
   * "Although they have identical descriptions, VkPhysicalDeviceIDProperties
   *  ::deviceUUID may differ from
   *  VkPhysicalDeviceProperties2::pipelineCacheUUID. The former is intended to
   *  identify and correlate devices across API and driver boundaries, while the
   *  latter is used to identify a compatible device and driver combination to
   *  use when serializing and de-serializing pipeline state."
   */
   _mesa_blake3_update(&ctx, screen->info.props.pipelineCacheUUID, VK_UUID_SIZE);

   /* Hash in our debug flags that affect NIR generation as of finalize_nir */
   unsigned shader_debug_flags = zink_debug & ZINK_DEBUG_COMPACT;
   _mesa_blake3_update(&ctx, &shader_debug_flags, sizeof(shader_debug_flags));

   /* add in these shader keys */
   _mesa_blake3_update(&ctx, &screen->driver_compiler_workarounds, sizeof(screen->driver_compiler_workarounds));

   /* Some of the driconf options change shaders.  Let's just hash the whole
    * thing to not forget any (especially as options get added).
    */
   _mesa_blake3_update(&ctx, &screen->driconf, sizeof(screen->driconf));

   /* EXT_shader_object causes different descriptor layouts for separate shaders */
   _mesa_blake3_update(&ctx, &screen->info.have_EXT_shader_object, sizeof(screen->info.have_EXT_shader_object));

   /* Finish the blake3 and format it as text. */
   blake3_hash blake3;
   _mesa_blake3_final(&ctx, blake3);

   char cache_id[20 * 2 + 1];
   mesa_bytes_to_hex(cache_id, blake3, 20);

   screen->disk_cache = disk_cache_create("zink", cache_id, 0);

   if (!screen->disk_cache)
      return true;

   if (!util_queue_init(&screen->cache_put_thread, "zcq", 8, 1, UTIL_QUEUE_INIT_RESIZE_IF_FULL, screen)) {
      mesa_loge("zink: Failed to create disk cache queue\n");

      disk_cache_destroy(screen->disk_cache);
      screen->disk_cache = NULL;

      return false;
   }
#endif

   return true;
}


static void
cache_put_job(void *data, void *gdata, int thread_index)
{
   struct zink_program *pg = data;
   struct zink_screen *screen = gdata;
   size_t size = 0;
   u_rwlock_rdlock(&pg->pipeline_cache_lock);
   VkResult result = VKSCR(GetPipelineCacheData)(screen->dev, pg->pipeline_cache, &size, NULL);
   if (result != VK_SUCCESS) {
      u_rwlock_rdunlock(&pg->pipeline_cache_lock);
      mesa_loge("ZINK: vkGetPipelineCacheData failed (%s)", vk_Result_to_str(result));
      return;
   }
   if (pg->pipeline_cache_size == size) {
      u_rwlock_rdunlock(&pg->pipeline_cache_lock);
      return;
   }
   void *pipeline_data = malloc(size);
   if (!pipeline_data) {
      u_rwlock_rdunlock(&pg->pipeline_cache_lock);
      return;
   }
   result = VKSCR(GetPipelineCacheData)(screen->dev, pg->pipeline_cache, &size, pipeline_data);
   u_rwlock_rdunlock(&pg->pipeline_cache_lock);
   if (result == VK_SUCCESS) {
      pg->pipeline_cache_size = size;

      cache_key key;
      disk_cache_compute_key(screen->disk_cache, pg->blake3, sizeof(pg->blake3), key);
      disk_cache_put_nocopy(screen->disk_cache, key, pipeline_data, size, NULL);
   } else {
      mesa_loge("ZINK: vkGetPipelineCacheData failed (%s)", vk_Result_to_str(result));
   }
}

void
zink_screen_update_pipeline_cache(struct zink_screen *screen, struct zink_program *pg, bool in_thread)
{
   if (!screen->disk_cache || !pg->pipeline_cache)
      return;

   if (in_thread)
      cache_put_job(pg, screen, 0);
   else if (util_queue_fence_is_signalled(&pg->cache_fence))
      util_queue_add_job(&screen->cache_put_thread, pg, &pg->cache_fence, cache_put_job, NULL, 0);
}

static void
cache_get_job(void *data, void *gdata, int thread_index)
{
   struct zink_program *pg = data;
   struct zink_screen *screen = gdata;

   VkPipelineCacheCreateInfo pcci;
   pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
   pcci.pNext = NULL;
   pcci.flags = screen->info.have_EXT_pipeline_creation_cache_control ? VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT : 0;
   pcci.initialDataSize = 0;
   pcci.pInitialData = NULL;

   cache_key key;
   disk_cache_compute_key(screen->disk_cache, pg->blake3, sizeof(pg->blake3), key);
   pcci.pInitialData = disk_cache_get(screen->disk_cache, key, &pg->pipeline_cache_size);
   pcci.initialDataSize = pg->pipeline_cache_size;

   VkResult res = VKSCR(CreatePipelineCache)(screen->dev, &pcci, NULL, &pg->pipeline_cache);
   if (res != VK_SUCCESS) {
      mesa_loge("ZINK: vkCreatePipelineCache failed (%s)", vk_Result_to_str(res));
   }
   free((void*)pcci.pInitialData);
}

void
zink_screen_get_pipeline_cache(struct zink_screen *screen, struct zink_program *pg, bool in_thread)
{
   if (!screen->disk_cache)
      return;

   if (in_thread)
      cache_get_job(pg, screen, 0);
   else
      util_queue_add_job(&screen->cache_get_thread, pg, &pg->cache_fence, cache_get_job, NULL, 0);
}

static uint32_t
get_smallest_buffer_heap(struct zink_screen *screen)
{
   enum zink_heap heaps[] = {
      ZINK_HEAP_DEVICE_LOCAL,
      ZINK_HEAP_DEVICE_LOCAL_VISIBLE,
      ZINK_HEAP_HOST_VISIBLE_COHERENT,
      ZINK_HEAP_HOST_VISIBLE_COHERENT
   };
   unsigned size = UINT32_MAX;
   for (unsigned i = 0; i < ARRAY_SIZE(heaps); i++) {
      for (unsigned j = 0; j < screen->heap_count[i]; j++) {
         unsigned heap_idx = screen->info.mem_props.memoryTypes[screen->heap_map[i][j]].heapIndex;
         size = MIN2(screen->info.mem_props.memoryHeaps[heap_idx].size, size);
      }
   }
   return size;
}

static inline bool
have_fp32_filter_linear(struct zink_screen *screen)
{
   const VkFormat fp32_formats[] = {
      VK_FORMAT_R32_SFLOAT,
      VK_FORMAT_R32G32_SFLOAT,
      VK_FORMAT_R32G32B32_SFLOAT,
      VK_FORMAT_R32G32B32A32_SFLOAT,
      VK_FORMAT_D32_SFLOAT,
   };
   for (int i = 0; i < ARRAY_SIZE(fp32_formats); ++i) {
      VkFormatProperties props;
      VKSCR(GetPhysicalDeviceFormatProperties)(screen->pdev,
                                               fp32_formats[i],
                                               &props);
      if (((props.linearTilingFeatures | props.optimalTilingFeatures) &
           (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) ==
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
         return false;
      }
   }
   return true;
}

static void
zink_init_shader_caps(struct zink_screen *screen)
{
   for (unsigned i = 0; i <= PIPE_SHADER_COMPUTE; i++) {
      struct pipe_shader_caps *caps =
         (struct pipe_shader_caps *)&screen->base.shader_caps[i];

      switch (i) {
      case MESA_SHADER_TESS_CTRL:
      case MESA_SHADER_TESS_EVAL:
         if (!screen->info.feats.features.tessellationShader ||
             !screen->info.have_KHR_maintenance2)
            continue;
         break;
      case MESA_SHADER_GEOMETRY:
         if (!screen->info.feats.features.geometryShader)
            continue;
         break;
      default:
         break;
      }

      caps->max_instructions =
      caps->max_alu_instructions =
      caps->max_tex_instructions =
      caps->max_tex_indirections =
      caps->max_control_flow_depth = INT_MAX;

      unsigned max_in = 0;
      unsigned max_out = 0;
      switch (i) {
      case MESA_SHADER_VERTEX:
         max_in = MIN2(screen->info.props.limits.maxVertexInputAttributes, PIPE_MAX_ATTRIBS);
         max_out = screen->info.props.limits.maxVertexOutputComponents / 4;
         break;
      case MESA_SHADER_TESS_CTRL:
         max_in = screen->info.props.limits.maxTessellationControlPerVertexInputComponents / 4;
         max_out = screen->info.props.limits.maxTessellationControlPerVertexOutputComponents / 4;
         break;
      case MESA_SHADER_TESS_EVAL:
         max_in = screen->info.props.limits.maxTessellationEvaluationInputComponents / 4;
         max_out = screen->info.props.limits.maxTessellationEvaluationOutputComponents / 4;
         break;
      case MESA_SHADER_GEOMETRY:
         max_in = screen->info.props.limits.maxGeometryInputComponents / 4;
         max_out = screen->info.props.limits.maxGeometryOutputComponents / 4;
         break;
      case MESA_SHADER_FRAGMENT:
         /* intel drivers report fewer components, but it's a value that's compatible
          * with what we need for GL, so we can still force a conformant value here
          */
         if (zink_driverid(screen) == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
             zink_driverid(screen) == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS)
            max_in = 32;
         else
            max_in = screen->info.props.limits.maxFragmentInputComponents / 4;

         max_out = screen->info.props.limits.maxColorAttachments;
         break;
      default:
         break;
      }

      switch (i) {
      case MESA_SHADER_VERTEX:
      case MESA_SHADER_TESS_EVAL:
      case MESA_SHADER_GEOMETRY:
         /* last vertex stage must support streamout, and this is capped in glsl compiler */
         caps->max_inputs = MIN2(max_in, MAX_VARYING);
         break;
      default:
         /* prevent overflowing struct shader_info::inputs_read */
         caps->max_inputs = MIN2(max_in, 64);
         break;
      }

      /* prevent overflowing struct shader_info::outputs_read/written */
      caps->max_outputs = MIN2(max_out, 64);

      /* At least 16384 is guaranteed by VK spec */
      assert(screen->info.props.limits.maxUniformBufferRange >= 16384);
      /* but Gallium can't handle values that are too big */
      caps->max_const_buffer0_size =
         MIN3(get_smallest_buffer_heap(screen),
              screen->info.props.limits.maxUniformBufferRange, BITFIELD_BIT(31));

      caps->max_const_buffers =
         MIN2(screen->info.props.limits.maxPerStageDescriptorUniformBuffers,
              PIPE_MAX_CONSTANT_BUFFERS);

      caps->max_temps = INT_MAX;
      caps->integers = true;
      caps->indirect_const_addr = true;
      caps->indirect_temp_addr = true;

      /* enabling this breaks GTF-GL46.gtf21.GL2Tests.glGetUniform.glGetUniform */
      caps->fp16_const_buffers = false;
         //screen->info.feats11.uniformAndStorageBuffer16BitAccess ||
         //(screen->info.have_KHR_16bit_storage && screen->info.storage_16bit_feats.uniformAndStorageBuffer16BitAccess);

      /* spirv requires 32bit derivative srcs and dests */
      caps->fp16_derivatives = false;

      caps->fp16 =
         screen->info.feats12.shaderFloat16 ||
         (screen->info.have_KHR_shader_float16_int8 &&
          screen->info.shader_float16_int8_feats.shaderFloat16);
      caps->glsl_16bit_load_dst = true;

      caps->int16 = screen->info.feats.features.shaderInt16;

      caps->max_texture_samplers =
      caps->max_sampler_views =
         MIN2(MIN2(screen->info.props.limits.maxPerStageDescriptorSamplers,
                   screen->info.props.limits.maxPerStageDescriptorSampledImages),
              PIPE_MAX_SAMPLERS);

      /* TODO: this limitation is dumb, and will need some fixes in mesa */
      caps->max_shader_buffers =
         MIN2(screen->info.props.limits.maxPerStageDescriptorStorageBuffers,
              PIPE_MAX_SHADER_BUFFERS);

      switch (i) {
      case MESA_SHADER_VERTEX:
      case MESA_SHADER_TESS_CTRL:
      case MESA_SHADER_TESS_EVAL:
      case MESA_SHADER_GEOMETRY:
         if (!screen->info.feats.features.vertexPipelineStoresAndAtomics)
            caps->max_shader_buffers = 0;
         break;
      case MESA_SHADER_FRAGMENT:
         if (!screen->info.feats.features.fragmentStoresAndAtomics)
            caps->max_shader_buffers = 0;
         break;
      default:
         break;
      }

      caps->supported_irs = (1 << PIPE_SHADER_IR_NIR) | (1 << PIPE_SHADER_IR_TGSI);

      if (screen->info.feats.features.shaderStorageImageExtendedFormats &&
          screen->info.feats.features.shaderStorageImageWriteWithoutFormat) {
         caps->max_shader_images =
            MIN2(screen->info.props.limits.maxPerStageDescriptorStorageImages,
                 ZINK_MAX_SHADER_IMAGES);
      }

      caps->cont_supported = true;
   }
}

static void
zink_init_compute_caps(struct zink_screen *screen)
{
   struct pipe_compute_caps *caps =
      (struct pipe_compute_caps *)&screen->base.compute_caps;

   caps->address_bits = 64;

   caps->grid_dimension = 3;

   caps->max_grid_size[0] = screen->info.props.limits.maxComputeWorkGroupCount[0];
   caps->max_grid_size[1] = screen->info.props.limits.maxComputeWorkGroupCount[1];
   caps->max_grid_size[2] = screen->info.props.limits.maxComputeWorkGroupCount[2];

   /* MaxComputeWorkGroupSize[0..2] */
   caps->max_block_size[0] = screen->info.props.limits.maxComputeWorkGroupSize[0];
   caps->max_block_size[1] = screen->info.props.limits.maxComputeWorkGroupSize[1];
   caps->max_block_size[2] = screen->info.props.limits.maxComputeWorkGroupSize[2];

   caps->max_threads_per_block =
   caps->max_variable_threads_per_block =
      screen->info.props.limits.maxComputeWorkGroupInvocations;

   caps->max_local_size =
      screen->info.props.limits.maxComputeSharedMemorySize;

   caps->subgroup_sizes = screen->info.props11.subgroupSize;
   caps->max_mem_alloc_size = screen->clamp_video_mem;
   caps->max_global_size = screen->total_video_mem;
   /* no way in vulkan to retrieve this information. */
   caps->max_compute_units = 1;
   caps->max_subgroups = screen->info.props13.maxComputeWorkgroupSubgroups;
}

static void
zink_init_screen_caps(struct zink_screen *screen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&screen->base.caps;

   u_init_pipe_screen_caps(&screen->base, screen->is_cpu ? 0 : 1);

   caps->null_textures = screen->info.rb_image_feats.robustImageAccess;
   /* support OVR_multiview and OVR_multiview2 */
   caps->multiview = screen->info.feats11.multiview;
   caps->texrect = false;
   caps->multi_draw_indirect_partial_stride = false;
   caps->anisotropic_filter = screen->info.feats.features.samplerAnisotropy;
   caps->emulate_nonfixed_primitive_restart = true;
   {
      uint32_t modes = BITFIELD_BIT(MESA_PRIM_LINE_STRIP) |
         BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP) |
         BITFIELD_BIT(MESA_PRIM_LINE_STRIP_ADJACENCY) |
         BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP_ADJACENCY);
      if (screen->have_triangle_fans)
         modes |= BITFIELD_BIT(MESA_PRIM_TRIANGLE_FAN);
      if (screen->info.have_EXT_primitive_topology_list_restart) {
         modes |= BITFIELD_BIT(MESA_PRIM_POINTS) |
            BITFIELD_BIT(MESA_PRIM_LINES) |
            BITFIELD_BIT(MESA_PRIM_LINES_ADJACENCY) |
            BITFIELD_BIT(MESA_PRIM_TRIANGLES) |
            BITFIELD_BIT(MESA_PRIM_TRIANGLES_ADJACENCY);
         if (screen->info.list_restart_feats.primitiveTopologyPatchListRestart)
            modes |= BITFIELD_BIT(MESA_PRIM_PATCHES);
      }
      caps->supported_prim_modes_with_restart = modes;
   }
   {
      uint32_t modes = BITFIELD_MASK(MESA_PRIM_COUNT);
      if (!screen->have_triangle_fans || !screen->info.feats.features.geometryShader)
         modes &= ~BITFIELD_BIT(MESA_PRIM_QUADS);
      modes &= ~BITFIELD_BIT(MESA_PRIM_QUAD_STRIP);
      modes &= ~BITFIELD_BIT(MESA_PRIM_POLYGON);
      modes &= ~BITFIELD_BIT(MESA_PRIM_LINE_LOOP);
      if (!screen->have_triangle_fans)
         modes &= ~BITFIELD_BIT(MESA_PRIM_TRIANGLE_FAN);
      caps->supported_prim_modes = modes;
   }
#if defined(MVK_VERSION)
   caps->fbfetch = 0;
#else
   caps->fbfetch = screen->info.have_KHR_dynamic_rendering_local_read;
#endif
   caps->fbfetch_coherent = caps->fbfetch && screen->info.have_EXT_rasterization_order_attachment_access;

   caps->memobj =
      screen->instance_info->have_KHR_external_memory_capabilities &&
      (screen->info.have_KHR_external_memory_fd ||
       screen->info.have_KHR_external_memory_win32);
   caps->fence_signal =
      screen->info.have_KHR_external_semaphore_fd ||
      screen->info.have_KHR_external_semaphore_win32;
   caps->native_fence_fd =
      screen->instance_info->have_KHR_external_semaphore_capabilities &&
      screen->info.have_KHR_external_semaphore_fd;
   caps->resource_from_user_memory = screen->info.have_EXT_external_memory_host;

   caps->surface_reinterpret_blocks =
      screen->info.have_vulkan11 || screen->info.have_KHR_maintenance2;
   caps->compressed_surface_reinterpret_blocks_layered = caps->surface_reinterpret_blocks &&
                                                         screen->info.maint6_props.blockTexelViewCompatibleMultipleLayers;

   caps->validate_all_dirty_states = true;
   caps->allow_mapped_buffers_during_execution = true;
   caps->map_unsynchronized_thread_safe = true;
   caps->shareable_shaders = true;
   caps->device_reset_status_query = true;
   caps->query_memory_info = true;
   caps->npot_textures = true;
   caps->tgsi_texcoord = true;
   caps->draw_indirect = true;
   caps->texture_query_lod = true;
   caps->glsl_tess_levels_as_inputs = true;
   caps->copy_between_compressed_and_plain_formats = true;
   caps->force_persample_interp = true;
   caps->framebuffer_no_attachment = true;
   caps->shader_array_components = true;
   caps->query_buffer_object = true;
   caps->conditional_render_inverted = true;
   caps->clip_halfz = true;
   caps->texture_query_samples = true;
   caps->texture_barrier = true;
   caps->query_so_overflow = true;
   caps->gl_spirv = true;
   caps->clear_scissored = true;
   caps->invalidate_buffer = true;
   caps->prefer_real_buffer_in_constbuf0 = true;
   caps->packed_uniforms = true;
   caps->shader_pack_half_float = true;
   caps->seamless_cube_map_per_texture = true;
   caps->load_constbuf = true;
   caps->multisample_z_resolve = true;
   caps->allow_glthread_buffer_subdata_opt = true;
   caps->nir_samplers_as_deref = true;
   caps->call_finalize_nir_in_linker = true;

   caps->draw_vertex_state = screen->info.have_EXT_vertex_input_dynamic_state;

   caps->surface_sample_count = screen->vk_version >= VK_MAKE_VERSION(1,2,0);

   caps->shader_group_vote =
      (screen->info.have_vulkan11 &&
       (screen->info.subgroup.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) &&
       (screen->info.subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) ||
      screen->info.have_EXT_shader_subgroup_vote;
   caps->quads_follow_provoking_vertex_convention = true;

   caps->texture_mirror_clamp_to_edge =
      screen->info.have_KHR_sampler_mirror_clamp_to_edge ||
      (screen->info.have_vulkan12 && screen->info.feats12.samplerMirrorClampToEdge);

   caps->polygon_offset_clamp = screen->info.feats.features.depthBiasClamp;

   caps->query_pipeline_statistics_single =
      screen->info.feats.features.pipelineStatisticsQuery;

   caps->robust_buffer_access_behavior =
      screen->info.feats.features.robustBufferAccess &&
      (screen->info.rb2_feats.robustImageAccess2 ||
       screen->driver_compiler_workarounds.lower_robustImageAccess2);

   caps->multi_draw_indirect = screen->info.feats.features.multiDrawIndirect;

   caps->image_atomic_float_add =
      (screen->info.have_EXT_shader_atomic_float &&
       screen->info.atomic_float_feats.shaderSharedFloat32AtomicAdd &&
       screen->info.atomic_float_feats.shaderBufferFloat32AtomicAdd);
   caps->shader_atomic_int64 =
      (screen->info.have_KHR_shader_atomic_int64 &&
       screen->info.atomic_int_feats.shaderSharedInt64Atomics &&
       screen->info.atomic_int_feats.shaderBufferInt64Atomics);

   caps->multi_draw_indirect_params = screen->info.have_KHR_draw_indirect_count;

   caps->start_instance =
   caps->draw_parameters =
      (screen->info.have_vulkan12 && screen->info.feats11.shaderDrawParameters) ||
      screen->info.have_KHR_shader_draw_parameters;

   caps->vertex_element_instance_divisor =
      screen->info.have_EXT_vertex_attribute_divisor;

   caps->max_vertex_streams = screen->info.tf_props.maxTransformFeedbackStreams;

   caps->compute_shader_derivatives = screen->info.have_NV_compute_shader_derivatives;

   caps->int64 = true;
   caps->doubles = true;

   caps->max_dual_source_render_targets =
      screen->info.feats.features.dualSrcBlend ?
      screen->info.props.limits.maxFragmentDualSrcAttachments : 0;

   caps->max_render_targets = screen->info.props.limits.maxColorAttachments;

   caps->occlusion_query = screen->info.feats.features.occlusionQueryPrecise;

   caps->programmable_sample_locations =
      screen->info.have_EXT_sample_locations &&
      screen->info.have_EXT_extended_dynamic_state;

   caps->query_time_elapsed = screen->timestamp_valid_bits > 0;

   caps->texture_multisample = true;

   caps->fragment_shader_interlock = screen->info.have_EXT_fragment_shader_interlock;

   caps->shader_clock = screen->info.have_KHR_shader_clock;

   caps->shader_ballot =
      screen->info.props11.subgroupSize <= 64 &&
      ((screen->info.have_vulkan11 &&
        screen->info.subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ||
       screen->info.have_EXT_shader_subgroup_ballot);

   caps->demote_to_helper_invocation =
      (screen->spirv_version >= SPIRV_VERSION(1, 6) ||
       screen->info.have_EXT_shader_demote_to_helper_invocation) &&
      !screen->driver_compiler_workarounds.broken_demote;

   caps->sample_shading = screen->info.feats.features.sampleRateShading;

   caps->texture_swizzle = true;

   caps->vertex_input_alignment =
      screen->info.have_EXT_legacy_vertex_attributes ?
      PIPE_VERTEX_INPUT_ALIGNMENT_NONE : PIPE_VERTEX_INPUT_ALIGNMENT_ELEMENT;

   caps->gl_clamp = false;

   /* Assume that the vk driver is capable of moving imm arrays to some sort of constant storage on its own. */
   caps->prefer_imm_arrays_as_constbuf = false;
   {
      enum pipe_quirk_texture_border_color_swizzle quirk =
         PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_ALPHA_NOT_W;
      if (!screen->info.border_color_feats.customBorderColorWithoutFormat)
         quirk |= PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_FREEDRENO;
      /* assume that if drivers don't implement this extension they either:
       * - don't support custom border colors
       * - handle things correctly
       * - hate border color accuracy
       */
      else if (screen->info.have_EXT_border_color_swizzle &&
               !screen->info.border_swizzle_feats.borderColorSwizzleFromImage)
         quirk |= PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_NV50;
      caps->texture_border_color_quirk = quirk;
   }
   caps->max_texture_2d_size =
      MIN2(screen->info.props.limits.maxImageDimension1D,
           screen->info.props.limits.maxImageDimension2D);
   caps->max_texture_3d_levels =
      1 + util_logbase2(screen->info.props.limits.maxImageDimension3D);
   caps->max_texture_cube_levels =
      1 + util_logbase2(screen->info.props.limits.maxImageDimensionCube);

   caps->fragment_shader_texture_lod = true;
   caps->fragment_shader_derivatives = true;

   caps->blend_equation_separate =
   caps->indep_blend_enable =
   caps->indep_blend_func = screen->info.feats.features.independentBlend;

   caps->dithering = false;

   caps->max_stream_output_buffers =
      screen->info.have_EXT_transform_feedback ?
      screen->info.tf_props.maxTransformFeedbackBuffers : 0;
   caps->stream_output_pause_resume =
   caps->stream_output_interleave_buffers = screen->info.have_EXT_transform_feedback;

   caps->max_texture_array_layers = screen->info.props.limits.maxImageArrayLayers;

   caps->depth_clip_disable = screen->info.have_EXT_depth_clip_enable;

   caps->shader_stencil_export = screen->info.have_EXT_shader_stencil_export;

   caps->vs_instanceid = true;
   caps->seamless_cube_map = true;

   caps->min_texel_offset = screen->info.props.limits.minTexelOffset;
   caps->max_texel_offset = screen->info.props.limits.maxTexelOffset;

   caps->max_timeline_semaphore_difference = screen->info.timeline_props.maxTimelineSemaphoreValueDifference;

   caps->vertex_color_unclamped = true;

   caps->conditional_render = true;

   caps->glsl_feature_level_compatibility =
   caps->glsl_feature_level = 460;

   caps->compute = true;

   caps->constant_buffer_offset_alignment =
      screen->info.props.limits.minUniformBufferOffsetAlignment;

   caps->query_timestamp = screen->timestamp_valid_bits > 0;

   caps->query_timestamp_bits = screen->timestamp_valid_bits;

   caps->timer_resolution = ceil(screen->info.props.limits.timestampPeriod);

   caps->min_map_buffer_alignment = 1 << MIN_SLAB_ORDER;

   caps->cube_map_array = screen->info.feats.features.imageCubeArray;

   caps->texture_buffer_objects = true;
   caps->primitive_restart = true;

   caps->bindless_texture =
      (zink_descriptor_mode != ZINK_DESCRIPTOR_MODE_DB ||
       (screen->info.db_props.maxDescriptorBufferBindings >= 2 &&
        screen->info.db_props.maxSamplerDescriptorBufferBindings >= 2)) &&
      screen->info.have_EXT_descriptor_indexing;

   caps->texture_buffer_offset_alignment =
      screen->info.props.limits.minTexelBufferOffsetAlignment;

   {
      enum pipe_texture_transfer_mode mode = PIPE_TEXTURE_TRANSFER_BLIT;
      if (!screen->is_cpu &&
          screen->info.have_KHR_8bit_storage &&
          screen->info.have_KHR_16bit_storage &&
          screen->info.have_KHR_shader_float16_int8)
         mode |= PIPE_TEXTURE_TRANSFER_COMPUTE;
      caps->texture_transfer_modes = mode;
   }

   caps->max_texel_buffer_elements =
      MIN2(get_smallest_buffer_heap(screen),
           screen->info.props.limits.maxTexelBufferElements);

   caps->endianness = PIPE_ENDIAN_NATIVE; /* unsure */

   caps->max_viewports =
      MIN2(screen->info.props.limits.maxViewports, PIPE_MAX_VIEWPORTS);

   caps->image_load_formatted =
      screen->info.feats.features.shaderStorageImageReadWithoutFormat;

   caps->image_store_formatted =
      screen->info.feats.features.shaderStorageImageWriteWithoutFormat;

   caps->mixed_framebuffer_sizes = true;

   caps->max_geometry_output_vertices =
      screen->info.props.limits.maxGeometryOutputVertices;
   caps->max_geometry_total_output_components =
      screen->info.props.limits.maxGeometryTotalOutputComponents;

   caps->max_texture_gather_components = 4;

   caps->min_texture_gather_offset = screen->info.props.limits.minTexelGatherOffset;
   caps->max_texture_gather_offset = screen->info.props.limits.maxTexelGatherOffset;

   caps->sampler_reduction_minmax_arb =
      screen->info.feats12.samplerFilterMinmax ||
      screen->info.have_EXT_sampler_filter_minmax;

   caps->opencl_integer_functions =
   caps->integer_multiply_32x16 = screen->info.have_INTEL_shader_integer_functions2;

   caps->fs_fine_derivative = true;

   caps->vendor_id = screen->info.props.vendorID;
   caps->device_id = screen->info.props.deviceID;

   caps->video_memory = get_video_mem(screen) >> 20;
   caps->uma = screen->info.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

   caps->max_vertex_attrib_stride =
      screen->info.props.limits.maxVertexInputBindingStride;

   caps->sampler_view_target = true;

   caps->vs_layer_viewport =
   caps->tes_layer_viewport =
      screen->info.have_EXT_shader_viewport_index_layer ||
      (screen->spirv_version >= SPIRV_VERSION(1, 5) &&
       screen->info.feats12.shaderOutputLayer &&
       screen->info.feats12.shaderOutputViewportIndex);

   caps->texture_float_linear = have_fp32_filter_linear(screen);

   caps->texture_half_float_linear = true;

   caps->shader_buffer_offset_alignment =
      screen->info.props.limits.minStorageBufferOffsetAlignment;

   caps->pci_group =
   caps->pci_bus =
   caps->pci_device =
   caps->pci_function = 0; /* TODO: figure these out */

   caps->cull_distance = screen->info.feats.features.shaderCullDistance;

   caps->sparse_buffer_page_size =
      screen->info.feats.features.sparseResidencyBuffer ? ZINK_SPARSE_BUFFER_PAGE_SIZE : 0;

   /* Sparse texture */
   caps->max_sparse_texture_size =
      screen->info.feats.features.sparseResidencyImage2D ?
      caps->max_texture_2d_size : 0;
   caps->max_sparse_3d_texture_size =
      screen->info.feats.features.sparseResidencyImage3D ?
      (1 << (caps->max_texture_3d_levels - 1)) : 0;
   caps->max_sparse_array_texture_layers =
      screen->info.feats.features.sparseResidencyImage2D ?
      caps->max_texture_array_layers : 0;
   caps->sparse_texture_full_array_cube_mipmaps =
      screen->info.feats.features.sparseResidencyImage2D;
   caps->query_sparse_texture_residency =
      screen->info.feats.features.sparseResidency2Samples &&
      screen->info.feats.features.shaderResourceResidency;
   caps->clamp_sparse_texture_lod =
      screen->info.feats.features.shaderResourceMinLod &&
      screen->info.feats.features.sparseResidency2Samples &&
      screen->info.feats.features.shaderResourceResidency;

   caps->viewport_subpixel_bits = screen->info.props.limits.viewportSubPixelBits;

   caps->max_gs_invocations = screen->info.props.limits.maxGeometryShaderInvocations;

   /* gallium handles this automatically */
   caps->max_combined_shader_buffers = 0;

   /* 1<<27 is required by VK spec */
   assert(screen->info.props.limits.maxStorageBufferRange >= 1 << 27);
   /* clamp to VK spec minimum */
   caps->max_shader_buffer_size =
      MIN2(get_smallest_buffer_heap(screen), screen->info.props.limits.maxStorageBufferRange);

   caps->fs_coord_origin_upper_left = true;
   caps->fs_coord_pixel_center_half_integer = true;

   caps->fs_coord_origin_lower_left = false;
   caps->fs_coord_pixel_center_integer = false;

   caps->fs_face_is_integer_sysval = true;
   caps->fs_point_is_sysval = true;

   caps->viewport_transform_lowered = true;

   caps->point_size_fixed =
      screen->info.have_KHR_maintenance5 ?
      PIPE_POINT_SIZE_LOWER_USER_ONLY : PIPE_POINT_SIZE_LOWER_ALWAYS;
   caps->flatshade = false;
   caps->alpha_test = false;
   caps->clip_planes = 0;
   caps->two_sided_color = false;

   caps->max_shader_patch_varyings =
      screen->info.props.limits.maxTessellationControlPerPatchOutputComponents / 4;
   /* need to reserve up to 60 of our varying components and 16 slots for streamout */
   caps->max_varyings =
      MIN2(screen->info.props.limits.maxVertexOutputComponents / 4 / 2, 16);

   caps->dmabuf =
#if defined(HAVE_LIBDRM) && (DETECT_OS_LINUX || DETECT_OS_BSD)
      screen->info.have_KHR_external_memory_fd &&
      screen->info.have_EXT_external_memory_dma_buf &&
      screen->info.have_EXT_queue_family_foreign
      ? DRM_PRIME_CAP_IMPORT | DRM_PRIME_CAP_EXPORT : 0;
#else
      0;
#endif

   caps->depth_bounds_test = screen->info.feats.features.depthBounds;

   caps->post_depth_coverage = screen->info.have_EXT_post_depth_coverage;

   caps->cl_gl_sharing = caps->dmabuf && screen->info.have_KHR_external_semaphore_fd;
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
      caps->linear_image_pitch_alignment = 1;
      break;
   /* AMD requires 256 */
   case VK_DRIVER_ID_AMD_PROPRIETARY:
   case VK_DRIVER_ID_MESA_RADV:
   default:
      caps->linear_image_pitch_alignment = 256;
      break;
   }
   caps->linear_image_base_address_alignment = 1;

   caps->string_marker = screen->instance_info->have_EXT_debug_utils;

   caps->min_line_width =
   caps->min_line_width_aa =
      screen->info.feats.features.wideLines ?
      MAX2(screen->info.props.limits.lineWidthRange[0], 0.01) : 1.0f;

   caps->min_point_size =
   caps->min_point_size_aa =
      screen->info.feats.features.largePoints ?
      MAX2(screen->info.props.limits.pointSizeRange[0], 0.01) : 1.0f;

   caps->line_width_granularity =
      screen->info.feats.features.wideLines ?
      screen->info.props.limits.lineWidthGranularity : 0.1f;

   caps->point_size_granularity =
      screen->info.feats.features.largePoints ?
      screen->info.props.limits.pointSizeGranularity : 0.1f;

   caps->max_line_width =
   caps->max_line_width_aa =
      screen->info.feats.features.wideLines ?
      screen->info.props.limits.lineWidthRange[1] : 1.0f;

   caps->max_point_size =
   caps->max_point_size_aa =
      screen->info.feats.features.largePoints ?
      screen->info.props.limits.pointSizeRange[1] : 1.0f;

   caps->max_texture_anisotropy =
      screen->info.feats.features.samplerAnisotropy ?
      screen->info.props.limits.maxSamplerAnisotropy : 1.0f;

   caps->max_texture_lod_bias = screen->info.props.limits.maxSamplerLodBias;

   if (screen->info.feats12.subgroupBroadcastDynamicId && screen->info.feats12.shaderSubgroupExtendedTypes && screen->info.feats.features.shaderFloat64) {
      caps->shader_subgroup_size = screen->info.subgroup.subgroupSize;
      caps->shader_subgroup_supported_stages = screen->info.subgroup.supportedStages & BITFIELD_MASK(MESA_SHADER_STAGES);
      caps->shader_subgroup_supported_features = screen->info.subgroup.supportedOperations & BITFIELD_MASK(PIPE_SHADER_SUBGROUP_NUM_FEATURES);
      caps->shader_subgroup_quad_all_stages = screen->info.subgroup.quadOperationsInAllStages;
   }
}

static VkSampleCountFlagBits
vk_sample_count_flags(uint32_t sample_count)
{
   switch (sample_count) {
   case 1: return VK_SAMPLE_COUNT_1_BIT;
   case 2: return VK_SAMPLE_COUNT_2_BIT;
   case 4: return VK_SAMPLE_COUNT_4_BIT;
   case 8: return VK_SAMPLE_COUNT_8_BIT;
   case 16: return VK_SAMPLE_COUNT_16_BIT;
   case 32: return VK_SAMPLE_COUNT_32_BIT;
   case 64: return VK_SAMPLE_COUNT_64_BIT;
   default:
      return 0;
   }
}

static bool
zink_is_compute_copy_faster(struct pipe_screen *pscreen,
                            enum pipe_format src_format,
                            enum pipe_format dst_format,
                            unsigned width,
                            unsigned height,
                            unsigned depth,
                            bool cpu)
{
   if (cpu)
      /* very basic for now, probably even worse for some cases,
       * but fixes lots of others
       */
      return width * height * depth > 64 * 64;
   return false;
}

static bool
zink_is_format_supported(struct pipe_screen *pscreen,
                         enum pipe_format format,
                         enum pipe_texture_target target,
                         unsigned sample_count,
                         unsigned storage_sample_count,
                         unsigned bind)
{
   struct zink_screen *screen = zink_screen(pscreen);

   if (storage_sample_count && !screen->info.feats.features.shaderStorageImageMultisample && bind & PIPE_BIND_SHADER_IMAGE)
      return false;

   if (format == PIPE_FORMAT_NONE)
      return screen->info.props.limits.framebufferNoAttachmentsSampleCounts &
             vk_sample_count_flags(sample_count);

   if (bind & PIPE_BIND_INDEX_BUFFER) {
      if (format == PIPE_FORMAT_R8_UINT &&
          !screen->info.have_EXT_index_type_uint8)
         return false;
      if (format != PIPE_FORMAT_R8_UINT &&
          format != PIPE_FORMAT_R16_UINT &&
          format != PIPE_FORMAT_R32_UINT)
         return false;
   }

   /* always use superset to determine feature support */
   VkFormat vkformat = zink_get_format(screen, PIPE_FORMAT_A8_UNORM ? zink_format_get_emulated_alpha(format) : format);
   if (vkformat == VK_FORMAT_UNDEFINED)
      return false;

   if (sample_count >= 1) {
      VkSampleCountFlagBits sample_mask = vk_sample_count_flags(sample_count);
      if (!sample_mask)
         return false;
      const struct util_format_description *desc = util_format_description(format);
      if (util_format_is_depth_or_stencil(format)) {
         if (util_format_has_depth(desc)) {
            if (bind & PIPE_BIND_DEPTH_STENCIL &&
                (screen->info.props.limits.framebufferDepthSampleCounts & sample_mask) != sample_mask)
               return false;
            if (bind & PIPE_BIND_SAMPLER_VIEW &&
                (screen->info.props.limits.sampledImageDepthSampleCounts & sample_mask) != sample_mask)
               return false;
         }
         if (util_format_has_stencil(desc)) {
            if (bind & PIPE_BIND_DEPTH_STENCIL &&
                (screen->info.props.limits.framebufferStencilSampleCounts & sample_mask) != sample_mask)
               return false;
            if (bind & PIPE_BIND_SAMPLER_VIEW &&
                (screen->info.props.limits.sampledImageStencilSampleCounts & sample_mask) != sample_mask)
               return false;
         }
      } else if (util_format_is_pure_integer(format)) {
         if (bind & PIPE_BIND_RENDER_TARGET &&
             !(screen->info.props.limits.framebufferColorSampleCounts & sample_mask))
            return false;
         if (bind & PIPE_BIND_SAMPLER_VIEW &&
             !(screen->info.props.limits.sampledImageIntegerSampleCounts & sample_mask))
            return false;
      } else {
         if (bind & PIPE_BIND_RENDER_TARGET &&
             !(screen->info.props.limits.framebufferColorSampleCounts & sample_mask))
            return false;
         if (bind & PIPE_BIND_SAMPLER_VIEW &&
             !(screen->info.props.limits.sampledImageColorSampleCounts & sample_mask))
            return false;
      }
      if (bind & PIPE_BIND_SHADER_IMAGE) {
          if (!(screen->info.props.limits.storageImageSampleCounts & sample_mask))
             return false;
      }
      VkResult ret;
      VkImageFormatProperties image_props;
      VkImageFormatProperties2 props2;
      props2.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
      props2.pNext = NULL;
      VkPhysicalDeviceImageFormatInfo2 info;
      info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
      info.pNext = NULL;
      info.format = vkformat;
      info.flags = 0;
      info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      switch (target) {
      case PIPE_TEXTURE_1D:
      case PIPE_TEXTURE_1D_ARRAY: {
         bool need_2D = false;
         if (util_format_is_depth_or_stencil(format))
            need_2D |= screen->need_2D_zs;
         info.type = need_2D ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_1D;
         break;
      }

      case PIPE_TEXTURE_CUBE:
      case PIPE_TEXTURE_CUBE_ARRAY:
         info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
         FALLTHROUGH;
      case PIPE_TEXTURE_2D:
      case PIPE_TEXTURE_2D_ARRAY:
      case PIPE_TEXTURE_RECT:
         info.type = VK_IMAGE_TYPE_2D;
         break;

      case PIPE_TEXTURE_3D:
         info.type = VK_IMAGE_TYPE_3D;
         if (bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_DEPTH_STENCIL))
            info.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
         if (screen->info.have_EXT_image_2d_view_of_3d)
            info.flags |= VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT;
         break;

      default:
         UNREACHABLE("unknown texture target");
      }
      u_foreach_bit(b, bind) {
         switch (1<<b) {
         case PIPE_BIND_RENDER_TARGET:
            info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            break;
         case PIPE_BIND_DEPTH_STENCIL:
            info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            break;
         case PIPE_BIND_SAMPLER_VIEW:
            info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            break;
         }
      }

      if (VKSCR(GetPhysicalDeviceImageFormatProperties2)) {
         ret = VKSCR(GetPhysicalDeviceImageFormatProperties2)(screen->pdev, &info, &props2);
         /* this is using VK_IMAGE_CREATE_EXTENDED_USAGE_BIT and can't be validated */
         if (vk_format_aspects(vkformat) & VK_IMAGE_ASPECT_PLANE_1_BIT)
            ret = VK_SUCCESS;
         image_props = props2.imageFormatProperties;
      } else {
         ret = VKSCR(GetPhysicalDeviceImageFormatProperties)(screen->pdev, vkformat, info.type,
                                                             info.tiling, info.usage, info.flags, &image_props);
      }
      if (ret != VK_SUCCESS)
         return false;
      if (!(sample_count & image_props.sampleCounts))
         return false;
   }

   const struct zink_format_props *props = zink_get_format_props(screen, format);

   if (target == PIPE_BUFFER) {
      if (bind & PIPE_BIND_VERTEX_BUFFER) {
         if (!(props->bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
            enum pipe_format new_format = zink_decompose_vertex_format(format);
            if (!new_format)
               return false;
            if (!(zink_get_format_props(screen, new_format)->bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT))
               return false;
         }
      }

      /* We can't swizzle buffer views */
      if (bind & (PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE) &&
          util_format_is_intensity(format))
          return false;

      if (bind & PIPE_BIND_SAMPLER_VIEW &&
         !(props->bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT))
            return false;

      if (bind & PIPE_BIND_SHADER_IMAGE &&
          !(props->bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT))
         return false;
   } else {
      /* all other targets are texture-targets */
      if (bind & PIPE_BIND_RENDER_TARGET &&
          !(props->optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
         return false;

      if (bind & PIPE_BIND_BLENDABLE &&
         !(props->optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT))
        return false;

      if (bind & PIPE_BIND_SAMPLER_VIEW &&
         !(props->optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
            return false;

      if (bind & PIPE_BIND_SAMPLER_REDUCTION_MINMAX &&
          !(props->optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT))
         return false;

      if ((bind & PIPE_BIND_SAMPLER_VIEW) || (bind & PIPE_BIND_RENDER_TARGET)) {
         /* if this is a 3-component texture, force gallium to give us 4 components by rejecting this one */
         const struct util_format_description *desc = util_format_description(format);
         if (desc->nr_channels == 3 &&
             (desc->block.bits == 24 || desc->block.bits == 48 || desc->block.bits == 96))
            return false;
      }

      if (bind & PIPE_BIND_DEPTH_STENCIL &&
          !(props->optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
         return false;

      if (bind & PIPE_BIND_SHADER_IMAGE &&
          !(props->optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
         return false;

      /* Can't swizzle storage images. */
      if (bind & PIPE_BIND_SHADER_IMAGE && util_format_is_intensity(format))
         return false;
   }

   return true;
}

static void
zink_set_damage_region(struct pipe_screen *pscreen, struct pipe_resource *pres, unsigned int nrects, const struct pipe_box *rects)
{
   struct zink_resource *res = zink_resource(pres);

   if (nrects == 0) {
      res->use_damage = false;
      return;
   }

   struct pipe_box damage = rects[0];
   for (unsigned i = 1; i < nrects; i++)
      u_box_union_2d(&damage, &damage, &rects[i]);

   /* The damage we get from EGL uses a lower-left origin but Vulkan uses
    * upper-left so we need to flip it.
    */
   damage.y = pres->height0 - (damage.y + damage.height);

   /* Intersect with the area of the resource */
   struct pipe_box res_area;
   u_box_origin_2d(pres->width0, pres->height0, &res_area);
   u_box_intersect_2d(&damage, &damage, &res_area);

   res->damage = (VkRect2D) {
      .offset.x = damage.x,
      .offset.y = damage.y,
      .extent.width = damage.width,
      .extent.height = damage.height,
   };
   res->use_damage = damage.x != 0 ||
                     damage.y != 0 ||
                     damage.width != res->base.b.width0 ||
                     damage.height != res->base.b.height0;
}

static void
zink_destroy_screen(struct pipe_screen *pscreen)
{
   struct zink_screen *screen = zink_screen(pscreen);

   if (screen->renderdoc_capture_all && p_atomic_dec_zero(&num_screens))
      screen->renderdoc_api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(screen->instance), NULL);

   hash_table_foreach(&screen->dts, entry)
      zink_kopper_deinit_displaytarget(screen, entry->data);

   if (screen->copy_context)
      screen->copy_context->base.destroy(&screen->copy_context->base);

   struct zink_batch_state *bs = screen->free_batch_states;
   while (bs) {
      struct zink_batch_state *bs_next = bs->next;
      zink_batch_state_destroy(screen, bs);
      bs = bs_next;
   }

   if (VK_NULL_HANDLE != screen->debugUtilsCallbackHandle) {
      VKSCR(DestroyDebugUtilsMessengerEXT)(screen->instance, screen->debugUtilsCallbackHandle, NULL);
   }

   util_vertex_state_cache_deinit(&screen->vertex_state_cache);

   if (screen->gfx_push_constant_layout)
      VKSCR(DestroyPipelineLayout)(screen->dev, screen->gfx_push_constant_layout, NULL);

   u_transfer_helper_destroy(pscreen->transfer_helper);
   if (util_queue_is_initialized(&screen->cache_get_thread)) {
      util_queue_finish(&screen->cache_get_thread);
      util_queue_destroy(&screen->cache_get_thread);
   }
#ifdef ENABLE_SHADER_CACHE
   if (screen->disk_cache && util_queue_is_initialized(&screen->cache_put_thread)) {
      util_queue_finish(&screen->cache_put_thread);
      disk_cache_wait_for_idle(screen->disk_cache);
      util_queue_destroy(&screen->cache_put_thread);
   }
#endif
   disk_cache_destroy(screen->disk_cache);

   /* we don't have an API to check if a set is already initialized */
   for (unsigned i = 0; i < ARRAY_SIZE(screen->pipeline_libs); i++)
      if (screen->pipeline_libs[i].table)
         _mesa_set_clear(&screen->pipeline_libs[i], NULL);

   zink_bo_deinit(screen);
   util_live_shader_cache_deinit(&screen->shaders);

   zink_descriptor_layouts_deinit(screen);

   if (screen->sem)
      VKSCR(DestroySemaphore)(screen->dev, screen->sem, NULL);

   if (screen->fence)
      VKSCR(DestroyFence)(screen->dev, screen->fence, NULL);

   if (util_queue_is_initialized(&screen->flush_queue))
      util_queue_destroy(&screen->flush_queue);

   while (util_dynarray_contains(&screen->semaphores, VkSemaphore))
      VKSCR(DestroySemaphore)(screen->dev, util_dynarray_pop(&screen->semaphores, VkSemaphore), NULL);
   while (util_dynarray_contains(&screen->fd_semaphores, VkSemaphore))
      VKSCR(DestroySemaphore)(screen->dev, util_dynarray_pop(&screen->fd_semaphores, VkSemaphore), NULL);
   if (screen->bindless_layout)
      VKSCR(DestroyDescriptorSetLayout)(screen->dev, screen->bindless_layout, NULL);

   if (screen->dev) {
      simple_mtx_lock(&device_lock);
      set_foreach(&device_table, entry) {
         struct zink_device *zdev = (void*)entry->key;
         if (zdev->pdev == screen->pdev) {
            zdev->refcount--;
            if (!zdev->refcount) {
               VKSCR(DestroyDevice)(zdev->dev, NULL);
               _mesa_set_remove(&device_table, entry);
               free(zdev);
               break;
            }
         }
      }
      if (!device_table.entries) {
         ralloc_free(device_table.table);
         device_table.table = NULL;
      }
      simple_mtx_unlock(&device_lock);
   }

   simple_mtx_lock(&instance_lock);
   if (screen->instance && --instance_refcount == 0)
      VKSCR(DestroyInstance)(instance, NULL);
   simple_mtx_unlock(&instance_lock);

   util_idalloc_mt_fini(&screen->buffer_ids);

   if (screen->loader_lib)
      util_dl_close(screen->loader_lib);

   if (screen->drm_fd != -1)
      close(screen->drm_fd);

   slab_destroy_parent(&screen->transfer_pool);
   ralloc_free(screen);
   glsl_type_singleton_decref();
}

static int
zink_get_display_device(const struct zink_screen *screen, uint32_t pdev_count,
                        const VkPhysicalDevice *pdevs, int64_t dev_major,
                        int64_t dev_minor)
{
   VkPhysicalDeviceDrmPropertiesEXT drm_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &drm_props,
   };

   for (uint32_t i = 0; i < pdev_count; ++i) {
      VKSCR(GetPhysicalDeviceProperties2)(pdevs[i], &props);
      if (drm_props.renderMajor == dev_major &&
          drm_props.renderMinor == dev_minor)
         return i;
   }

   return -1;
}

static int
zink_get_cpu_device_type(const struct zink_screen *screen, uint32_t pdev_count,
                         const VkPhysicalDevice *pdevs)
{
   VkPhysicalDeviceProperties props;

   for (uint32_t i = 0; i < pdev_count; ++i) {
      VKSCR(GetPhysicalDeviceProperties)(pdevs[i], &props);

      /* if user wants cpu, only give them cpu */
      if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
         return i;
   }

   mesa_loge("ZINK: CPU device requested but none found!");

   return -1;
}

static int
zink_match_adapter_luid(const struct zink_screen *screen, uint32_t pdev_count, const VkPhysicalDevice *pdevs, uint64_t adapter_luid)
{
   VkPhysicalDeviceVulkan11Properties props11 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES
   };
   VkPhysicalDeviceProperties2 props = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      &props11
   };

   for (uint32_t i = 0; i < pdev_count; ++i) {
      VKSCR(GetPhysicalDeviceProperties2)(pdevs[i], &props);

      if (!memcmp(props11.deviceLUID, &adapter_luid, sizeof(adapter_luid)))
         return i;
   }

   mesa_loge("ZINK: matching LUID not found!");

   return -1;
}

static void
choose_pdev(struct zink_screen *screen, int64_t dev_major, int64_t dev_minor, uint64_t adapter_luid)
{
   bool cpu = debug_get_bool_option("LIBGL_ALWAYS_SOFTWARE", false) ||
              debug_get_bool_option("D3D_ALWAYS_SOFTWARE", false);

   if (cpu || (dev_major > 0 && dev_major < 255) || adapter_luid) {
      uint32_t pdev_count;
      int idx;
      VkPhysicalDevice *pdevs;
      VkResult result = VKSCR(EnumeratePhysicalDevices)(screen->instance, &pdev_count, NULL);
      if (result != VK_SUCCESS) {
         if (!screen->driver_name_is_inferred)
            mesa_loge("ZINK: vkEnumeratePhysicalDevices failed (%s)", vk_Result_to_str(result));
         return;
      }

      if (!pdev_count)
         return;

      pdevs = malloc(sizeof(*pdevs) * pdev_count);
      if (!pdevs) {
         if (!screen->driver_name_is_inferred)
            mesa_loge("ZINK: failed to allocate pdevs!");
         return;
      }
      result = VKSCR(EnumeratePhysicalDevices)(screen->instance, &pdev_count, pdevs);
      assert(result == VK_SUCCESS);
      assert(pdev_count > 0);

      if (adapter_luid)
         idx = zink_match_adapter_luid(screen, pdev_count, pdevs, adapter_luid);
      else if (cpu)
         idx = zink_get_cpu_device_type(screen, pdev_count, pdevs);
      else
         idx = zink_get_display_device(screen, pdev_count, pdevs, dev_major,
                                       dev_minor);

      if (idx != -1)
         /* valid cpu device */
         screen->pdev = pdevs[idx];

      free(pdevs);

      if (idx == -1)
         return;

   } else {
      VkPhysicalDevice pdev;
      unsigned pdev_count = 1;
      VkResult result = VKSCR(EnumeratePhysicalDevices)(screen->instance, &pdev_count, &pdev);
      if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
         if (!screen->driver_name_is_inferred)
            mesa_loge("ZINK: vkEnumeratePhysicalDevices failed (%s)", vk_Result_to_str(result));
         return;
      }
      if (!pdev_count)
         return;
      screen->pdev = pdev;
   }
   VKSCR(GetPhysicalDeviceProperties)(screen->pdev, &screen->info.props);

   /* allow software rendering only if forced by the user */
   if (((!cpu || screen->driver_name_is_inferred) && screen->info.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)) {
      screen->pdev = VK_NULL_HANDLE;
      return;
   }

   screen->info.device_version = screen->info.props.apiVersion;

   /* runtime version is the lesser of the instance version and device version */
   screen->vk_version = MIN2(screen->info.device_version, screen->instance_info->loader_version);

   /* calculate SPIR-V version based on VK version */
   if (screen->vk_version >= VK_MAKE_VERSION(1, 3, 0))
      screen->spirv_version = SPIRV_VERSION(1, 6);
   else if (screen->vk_version >= VK_MAKE_VERSION(1, 2, 0))
      screen->spirv_version = SPIRV_VERSION(1, 5);
   else if (screen->vk_version >= VK_MAKE_VERSION(1, 1, 0))
      screen->spirv_version = SPIRV_VERSION(1, 3);
   else
      screen->spirv_version = SPIRV_VERSION(1, 0);
}

static void
update_queue_props(struct zink_screen *screen)
{
   uint32_t num_queues;
   VKSCR(GetPhysicalDeviceQueueFamilyProperties)(screen->pdev, &num_queues, NULL);
   assert(num_queues > 0);

   VkQueueFamilyProperties *props = malloc(sizeof(*props) * num_queues);
   if (!props) {
      mesa_loge("ZINK: failed to allocate props!");
      return;
   }
      
   VKSCR(GetPhysicalDeviceQueueFamilyProperties)(screen->pdev, &num_queues, props);

   bool found_gfx = false;
   uint32_t sparse_only = UINT32_MAX;
   screen->sparse_queue = UINT32_MAX;
   for (uint32_t i = 0; i < num_queues; i++) {
      if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         if (found_gfx)
            continue;
         screen->sparse_queue = screen->gfx_queue = i;
         screen->max_queues = props[i].queueCount;
         screen->timestamp_valid_bits = props[i].timestampValidBits;
         found_gfx = true;
      } else if (props[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)
         sparse_only = i;
   }
   if (sparse_only != UINT32_MAX)
      screen->sparse_queue = sparse_only;
   free(props);
}

static void
init_queue(struct zink_screen *screen)
{
   simple_mtx_init(&screen->queue_lock, mtx_plain);
   VKSCR(GetDeviceQueue)(screen->dev, screen->gfx_queue, 0, &screen->queue);
   if (screen->sparse_queue != screen->gfx_queue)
      VKSCR(GetDeviceQueue)(screen->dev, screen->sparse_queue, 0, &screen->queue_sparse);
   else
      screen->queue_sparse = screen->queue;
}

static void
zink_flush_frontbuffer(struct pipe_screen *pscreen,
                       struct pipe_context *pctx,
                       struct pipe_resource *pres,
                       unsigned level, unsigned layer,
                       void *winsys_drawable_handle,
                       unsigned nboxes,
                       struct pipe_box *sub_box)
{
   struct zink_screen *screen = zink_screen(pscreen);
   struct zink_resource *res = zink_resource(pres);
   struct zink_context *ctx = zink_context(pctx);

   /* if the surface is no longer a swapchain, this is a no-op */
   if (!zink_is_swapchain(res))
      return;

   ctx = zink_tc_context_unwrap(pctx);

   if (!zink_kopper_acquired(res->obj->dt, res->obj->dt_idx)) {
      /* swapbuffers to an undefined surface: acquire and present garbage */
      zink_kopper_acquire(ctx, res, UINT64_MAX);
      zink_resource_reference(&ctx->needs_present, res);
      /* set batch usage to submit acquire semaphore */
      zink_batch_resource_usage_set(ctx->bs, res, true, false);
      /* ensure the resource is set up to present garbage */
      ctx->base.flush_resource(&ctx->base, pres);
   }

   /* handle any outstanding acquire submits (not just from above) */
   if (ctx->swapchain || ctx->needs_present) {
      ctx->bs->has_work = true;
      pctx->flush(pctx, NULL, PIPE_FLUSH_END_OF_FRAME);
      if (ctx->last_batch_state && screen->threaded_submit) {
         struct zink_batch_state *bs = ctx->last_batch_state;
         util_queue_fence_wait(&bs->flush_completed);
      }
   }
   res->use_damage = false;

   /* always verify that this was acquired */
   assert(zink_kopper_acquired(res->obj->dt, res->obj->dt_idx));
   zink_kopper_present_queue(screen, res, nboxes, sub_box);
}

bool
zink_is_depth_format_supported(struct zink_screen *screen, VkFormat format)
{
   VkFormatProperties props;
   VKSCR(GetPhysicalDeviceFormatProperties)(screen->pdev, format, &props);
   return (props.linearTilingFeatures | props.optimalTilingFeatures) &
          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
}

VkFormat
zink_get_format(struct zink_screen *screen, enum pipe_format format)
{
   if (format == PIPE_FORMAT_A8_UNORM && !screen->driver_workarounds.missing_a8_unorm)
      return VK_FORMAT_A8_UNORM_KHR;
   else if (!screen->driver_workarounds.broken_l4a4 || format != PIPE_FORMAT_L4A4_UNORM)
      format = zink_format_get_emulated_alpha(format);

   VkFormat ret = vk_format_from_pipe_format(zink_format_emulate_x8(format));

   if (format == PIPE_FORMAT_X32_S8X24_UINT &&
       screen->have_D32_SFLOAT_S8_UINT)
      return VK_FORMAT_D32_SFLOAT_S8_UINT;

   if (format == PIPE_FORMAT_X24S8_UINT)
      /* valid when using aspects to extract stencil,
       * fails format test because it's emulated */
      ret = VK_FORMAT_D24_UNORM_S8_UINT;

   if (ret == VK_FORMAT_X8_D24_UNORM_PACK32 &&
       !screen->have_X8_D24_UNORM_PACK32) {
      assert(zink_is_depth_format_supported(screen, VK_FORMAT_D32_SFLOAT));
      return VK_FORMAT_D32_SFLOAT;
   }

   if (ret == VK_FORMAT_D24_UNORM_S8_UINT &&
       !screen->have_D24_UNORM_S8_UINT) {
      assert(screen->have_D32_SFLOAT_S8_UINT);
      return VK_FORMAT_D32_SFLOAT_S8_UINT;
   }

   if ((ret == VK_FORMAT_A4B4G4R4_UNORM_PACK16 &&
        !screen->info.format_4444_feats.formatA4B4G4R4) ||
       (ret == VK_FORMAT_A4R4G4B4_UNORM_PACK16 &&
        !screen->info.format_4444_feats.formatA4R4G4B4))
      return VK_FORMAT_UNDEFINED;

   if (format == PIPE_FORMAT_R4A4_UNORM)
      return VK_FORMAT_R4G4_UNORM_PACK8;

   return ret;
}

void
zink_convert_color(const struct zink_screen *screen, enum pipe_format format,
                   union pipe_color_union *dst,
                   const union pipe_color_union *src)
{
   const struct util_format_description *desc = util_format_description(format);
   union pipe_color_union tmp = *src;

   for (unsigned i = 0; i < 4; i++)
      zink_format_clamp_channel_color(desc, &tmp, src, i);

   if (zink_format_is_emulated_alpha(format) &&
       /* Don't swizzle colors if the driver supports real A8_UNORM */
       (format != PIPE_FORMAT_A8_UNORM ||
         screen->driver_workarounds.missing_a8_unorm)) {
      if (util_format_is_alpha(format)) {
         tmp.ui[0] = tmp.ui[3];
         tmp.ui[1] = 0;
         tmp.ui[2] = 0;
         tmp.ui[3] = 0;
      } else if (util_format_is_luminance(format)) {
         tmp.ui[1] = 0;
         tmp.ui[2] = 0;
         tmp.f[3] = 1.0;
      } else if (util_format_is_luminance_alpha(format)) {
         tmp.ui[1] = tmp.ui[3];
         tmp.ui[2] = 0;
         tmp.f[3] = 1.0;
      } else /* zink_format_is_red_alpha */ {
         tmp.ui[1] = tmp.ui[3];
         tmp.ui[2] = 0;
         tmp.ui[3] = 0;
      }
   }

   memcpy(dst, &tmp, sizeof(union pipe_color_union));
}

static bool
check_have_device_time(struct zink_screen *screen)
{
   uint32_t num_domains = 0;
   VkTimeDomainEXT domains[8]; //current max is 4
   VkResult result = VKSCR(GetPhysicalDeviceCalibrateableTimeDomainsEXT)(screen->pdev, &num_domains, NULL);
   if (result != VK_SUCCESS) {
      mesa_loge("ZINK: vkGetPhysicalDeviceCalibrateableTimeDomainsEXT failed (%s)", vk_Result_to_str(result));
   }
   assert(num_domains > 0);
   assert(num_domains < ARRAY_SIZE(domains));

   result = VKSCR(GetPhysicalDeviceCalibrateableTimeDomainsEXT)(screen->pdev, &num_domains, domains);
   if (result != VK_SUCCESS) {
      mesa_loge("ZINK: vkGetPhysicalDeviceCalibrateableTimeDomainsEXT failed (%s)", vk_Result_to_str(result));
   }

   /* VK_TIME_DOMAIN_DEVICE_EXT is used for the ctx->get_timestamp hook and is the only one we really need */
   for (unsigned i = 0; i < num_domains; i++) {
      if (domains[i] == VK_TIME_DOMAIN_DEVICE_EXT) {
         return true;
      }
   }

   return false;
}

static void
zink_error(const char *msg)
{
}

static void
zink_warn(const char *msg)
{
}

static void
zink_info(const char *msg)
{
}

static void
zink_msg(const char *msg)
{
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
zink_debug_util_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT                  messageType,
    const VkDebugUtilsMessengerCallbackDataEXT      *pCallbackData,
    void                                            *pUserData)
{
   // Pick message prefix and color to use.
   // Only MacOS and Linux have been tested for color support
   if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
      zink_error(pCallbackData->pMessage);
   } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
      zink_warn(pCallbackData->pMessage);
   } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
      zink_info(pCallbackData->pMessage);
   } else
      zink_msg(pCallbackData->pMessage);

   return VK_FALSE;
}

static bool
create_debug(struct zink_screen *screen)
{
   VkDebugUtilsMessengerCreateInfoEXT vkDebugUtilsMessengerCreateInfoEXT = {
       VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
       NULL,
       0,  // flags
       VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
       VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
       VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
       zink_debug_util_callback,
       NULL
   };

   VkDebugUtilsMessengerEXT vkDebugUtilsCallbackEXT = VK_NULL_HANDLE;

   VkResult result = VKSCR(CreateDebugUtilsMessengerEXT)(
           screen->instance,
           &vkDebugUtilsMessengerCreateInfoEXT,
           NULL,
           &vkDebugUtilsCallbackEXT);
   if (result != VK_SUCCESS) {
      mesa_loge("ZINK: vkCreateDebugUtilsMessengerEXT failed (%s)", vk_Result_to_str(result));
   }

   screen->debugUtilsCallbackHandle = vkDebugUtilsCallbackEXT;

   return true;
}

static bool
zink_internal_setup_moltenvk(struct zink_screen *screen)
{
#if defined(MVK_VERSION)
   // MoltenVK only supports VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE in newer Metal versions
   // disable unless we can get MoltenVK to confirm it is supported
   screen->have_dynamic_state_vertex_input_binding_stride = false;

   if (!screen->instance_info->have_MVK_moltenvk)
      return true;

   GET_PROC_ADDR_INSTANCE_LOCAL(screen, screen->instance, GetMoltenVKConfigurationMVK);
   GET_PROC_ADDR_INSTANCE_LOCAL(screen, screen->instance, SetMoltenVKConfigurationMVK);
   GET_PROC_ADDR_INSTANCE_LOCAL(screen, screen->instance, GetVersionStringsMVK);
   GET_PROC_ADDR_INSTANCE_LOCAL(screen, screen->instance, GetPhysicalDeviceMetalFeaturesMVK);

   if (vk_GetVersionStringsMVK) {
      char molten_version[64] = {0};
      char vulkan_version[64] = {0};

      vk_GetVersionStringsMVK(molten_version, sizeof(molten_version) - 1, vulkan_version, sizeof(vulkan_version) - 1);

      printf("zink: MoltenVK %s Vulkan %s \n", molten_version, vulkan_version);
   }

   if (vk_GetMoltenVKConfigurationMVK && vk_SetMoltenVKConfigurationMVK) {
      MVKConfiguration molten_config = {0};
      size_t molten_config_size = sizeof(molten_config);

      VkResult res = vk_GetMoltenVKConfigurationMVK(screen->instance, &molten_config, &molten_config_size);
      if (res == VK_SUCCESS || res == VK_INCOMPLETE) {
         // Needed to allow MoltenVK to accept VkImageView swizzles.
         // Encountered when using VK_FORMAT_R8G8_UNORM
         molten_config.fullImageViewSwizzle = VK_TRUE;
         vk_SetMoltenVKConfigurationMVK(screen->instance, &molten_config, &molten_config_size);
      }
   }

   if (vk_GetPhysicalDeviceMetalFeaturesMVK) {
      MVKPhysicalDeviceMetalFeatures metal_features={0};
      size_t metal_features_size = sizeof(metal_features);

      VkResult res = vk_GetPhysicalDeviceMetalFeaturesMVK(screen->pdev, &metal_features, &metal_features_size);
      if (res == VK_SUCCESS) {
        screen->have_dynamic_state_vertex_input_binding_stride = metal_features.dynamicVertexStride;
      }
   }
#endif // MVK_VERSION

   return true;
}

void
zink_init_format_props(struct zink_screen *screen, enum pipe_format pformat)
{
   VkFormat format;
retry:
   format = zink_get_format(screen, pformat);
   if (!format)
      return;
   if (VKSCR(GetPhysicalDeviceFormatProperties2)) {
      VkFormatProperties2 props = {0};
      props.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;

      VkDrmFormatModifierPropertiesListEXT mod_props;
      VkDrmFormatModifierPropertiesEXT mods[128];
      if (screen->info.have_EXT_image_drm_format_modifier) {
         mod_props.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT;
         mod_props.pNext = NULL;
         mod_props.drmFormatModifierCount = ARRAY_SIZE(mods);
         mod_props.pDrmFormatModifierProperties = mods;
         props.pNext = &mod_props;
      }
      VkFormatProperties3 props3 = {0};
      if (screen->info.have_KHR_format_feature_flags2 || screen->info.have_vulkan13) {
         props3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
         props3.pNext = props.pNext;
         props.pNext = &props3;
      }

      VKSCR(GetPhysicalDeviceFormatProperties2)(screen->pdev, format, &props);

      if (screen->info.have_KHR_format_feature_flags2 || screen->info.have_vulkan13) {
         screen->format_props[pformat].linearTilingFeatures = props3.linearTilingFeatures;
         screen->format_props[pformat].optimalTilingFeatures = props3.optimalTilingFeatures;
         screen->format_props[pformat].bufferFeatures = props3.bufferFeatures;

         if (props3.linearTilingFeatures & VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV)
            screen->format_props[pformat].linearTilingFeatures |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
      } else {
         // MoltenVk is 1.2 API
         screen->format_props[pformat].linearTilingFeatures = props.formatProperties.linearTilingFeatures;
         screen->format_props[pformat].optimalTilingFeatures = props.formatProperties.optimalTilingFeatures;
         screen->format_props[pformat].bufferFeatures = props.formatProperties.bufferFeatures;
      }

      if (screen->info.have_EXT_image_drm_format_modifier && mod_props.drmFormatModifierCount) {
         screen->modifier_props[pformat].drmFormatModifierCount = mod_props.drmFormatModifierCount;
         screen->modifier_props[pformat].pDrmFormatModifierProperties = ralloc_array(screen, VkDrmFormatModifierPropertiesEXT, mod_props.drmFormatModifierCount);
         if (mod_props.pDrmFormatModifierProperties) {
            for (unsigned j = 0; j < mod_props.drmFormatModifierCount; j++)
               screen->modifier_props[pformat].pDrmFormatModifierProperties[j] = mod_props.pDrmFormatModifierProperties[j];
         }
      }
   } else {
      VkFormatProperties props = {0};
      VKSCR(GetPhysicalDeviceFormatProperties)(screen->pdev, format, &props);
      screen->format_props[pformat].linearTilingFeatures = props.linearTilingFeatures;
      screen->format_props[pformat].optimalTilingFeatures = props.optimalTilingFeatures;
      screen->format_props[pformat].bufferFeatures = props.bufferFeatures;
   }
   if (pformat == PIPE_FORMAT_A8_UNORM && !screen->driver_workarounds.missing_a8_unorm) {
      if (!screen->format_props[pformat].linearTilingFeatures &&
            !screen->format_props[pformat].optimalTilingFeatures &&
            !screen->format_props[pformat].bufferFeatures) {
         screen->driver_workarounds.missing_a8_unorm = true;
         goto retry;
      }
   }
   if (zink_format_is_emulated_alpha(pformat)) {
      VkFormatFeatureFlags blocked = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
      screen->format_props[pformat].linearTilingFeatures &= ~blocked;
      screen->format_props[pformat].optimalTilingFeatures &= ~blocked;
      screen->format_props[pformat].bufferFeatures = 0;
   }
   screen->format_props_init[pformat] = true;
}

static void
check_vertex_formats(struct zink_screen *screen)
{
   /* from vbuf */
   enum pipe_format format_list[] = {
      /* not supported by vk
      PIPE_FORMAT_R32_FIXED,
      PIPE_FORMAT_R32G32_FIXED,
      PIPE_FORMAT_R32G32B32_FIXED,
      PIPE_FORMAT_R32G32B32A32_FIXED,
      */
      PIPE_FORMAT_R16_FLOAT,
      PIPE_FORMAT_R16G16_FLOAT,
      PIPE_FORMAT_R16G16B16_FLOAT,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      /* not supported by vk
      PIPE_FORMAT_R64_FLOAT,
      PIPE_FORMAT_R64G64_FLOAT,
      PIPE_FORMAT_R64G64B64_FLOAT,
      PIPE_FORMAT_R64G64B64A64_FLOAT,
      PIPE_FORMAT_R32_UNORM,
      PIPE_FORMAT_R32G32_UNORM,
      PIPE_FORMAT_R32G32B32_UNORM,
      PIPE_FORMAT_R32G32B32A32_UNORM,
      PIPE_FORMAT_R32_SNORM,
      PIPE_FORMAT_R32G32_SNORM,
      PIPE_FORMAT_R32G32B32_SNORM,
      PIPE_FORMAT_R32G32B32A32_SNORM,
      PIPE_FORMAT_R32_USCALED,
      PIPE_FORMAT_R32G32_USCALED,
      PIPE_FORMAT_R32G32B32_USCALED,
      PIPE_FORMAT_R32G32B32A32_USCALED,
      PIPE_FORMAT_R32_SSCALED,
      PIPE_FORMAT_R32G32_SSCALED,
      PIPE_FORMAT_R32G32B32_SSCALED,
      PIPE_FORMAT_R32G32B32A32_SSCALED,
      */
      PIPE_FORMAT_R16_UNORM,
      PIPE_FORMAT_R16G16_UNORM,
      PIPE_FORMAT_R16G16B16_UNORM,
      PIPE_FORMAT_R16G16B16A16_UNORM,
      PIPE_FORMAT_R16_SNORM,
      PIPE_FORMAT_R16G16_SNORM,
      PIPE_FORMAT_R16G16B16_SNORM,
      PIPE_FORMAT_R16G16B16_SINT,
      PIPE_FORMAT_R16G16B16_UINT,
      PIPE_FORMAT_R16G16B16A16_SNORM,
      PIPE_FORMAT_R16_USCALED,
      PIPE_FORMAT_R16G16_USCALED,
      PIPE_FORMAT_R16G16B16_USCALED,
      PIPE_FORMAT_R16G16B16A16_USCALED,
      PIPE_FORMAT_R16_SSCALED,
      PIPE_FORMAT_R16G16_SSCALED,
      PIPE_FORMAT_R16G16B16_SSCALED,
      PIPE_FORMAT_R16G16B16A16_SSCALED,
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8G8_UNORM,
      PIPE_FORMAT_R8G8B8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8_SNORM,
      PIPE_FORMAT_R8G8_SNORM,
      PIPE_FORMAT_R8G8B8_SNORM,
      PIPE_FORMAT_R8G8B8A8_SNORM,
      PIPE_FORMAT_R8_USCALED,
      PIPE_FORMAT_R8G8_USCALED,
      PIPE_FORMAT_R8G8B8_USCALED,
      PIPE_FORMAT_R8G8B8A8_USCALED,
      PIPE_FORMAT_R8_SSCALED,
      PIPE_FORMAT_R8G8_SSCALED,
      PIPE_FORMAT_R8G8B8_SSCALED,
      PIPE_FORMAT_R8G8B8A8_SSCALED,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(format_list); i++) {
      if (zink_is_format_supported(&screen->base, format_list[i], PIPE_BUFFER, 0, 0, PIPE_BIND_VERTEX_BUFFER))
         continue;
      if (util_format_get_nr_components(format_list[i]) == 1)
         continue;
      enum pipe_format decomposed = zink_decompose_vertex_format(format_list[i]);
      if (zink_is_format_supported(&screen->base, decomposed, PIPE_BUFFER, 0, 0, PIPE_BIND_VERTEX_BUFFER)) {
         screen->need_decompose_attrs = true;
         mesa_logw("zink: this application would be much faster if %s supported vertex format %s", screen->info.props.deviceName, util_format_name(format_list[i]));
      }
   }
}

static void
populate_format_props(struct zink_screen *screen)
{
   zink_init_format_props(screen, PIPE_FORMAT_A8_UNORM);
   check_vertex_formats(screen);
   VkImageFormatProperties image_props;
   VkResult ret = VKSCR(GetPhysicalDeviceImageFormatProperties)(screen->pdev, VK_FORMAT_D32_SFLOAT,
                                                                VK_IMAGE_TYPE_1D,
                                                                VK_IMAGE_TILING_OPTIMAL,
                                                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                0, &image_props);
   if (ret != VK_SUCCESS && ret != VK_ERROR_FORMAT_NOT_SUPPORTED) {
      mesa_loge("ZINK: vkGetPhysicalDeviceImageFormatProperties failed (%s)", vk_Result_to_str(ret));
   }
   screen->need_2D_zs = ret != VK_SUCCESS;

   if (screen->info.feats.features.sparseResidencyImage2D)
      screen->need_2D_sparse = !screen->base.get_sparse_texture_virtual_page_size(&screen->base, PIPE_TEXTURE_1D, false, PIPE_FORMAT_R32_FLOAT, 0, 16, NULL, NULL, NULL);
}

static void
setup_renderdoc(struct zink_screen *screen)
{
#ifndef _WIN32
   const char *capture_id = debug_get_option("ZINK_RENDERDOC", NULL);
   if (!capture_id)
      return;
#if DETECT_OS_ANDROID
   const char *libstr = "libVkLayer_GLES_RenderDoc.so";
#else
   const char *libstr = "librenderdoc.so";
#endif
   void *renderdoc = dlopen(libstr, RTLD_NOW | RTLD_NOLOAD);
   /* not loaded */
   if (!renderdoc)
      return;

   pRENDERDOC_GetAPI get_api = dlsym(renderdoc, "RENDERDOC_GetAPI");
   if (!get_api)
      return;

   /* need synchronous dispatch for renderdoc coherency */
   screen->threaded_submit = false;
   get_api(eRENDERDOC_API_Version_1_0_0, (void*)&screen->renderdoc_api);
   screen->renderdoc_api->SetActiveWindow(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(screen->instance), NULL);

   int count = sscanf(capture_id, "%u:%u", &screen->renderdoc_capture_start, &screen->renderdoc_capture_end);
   if (count != 2) {
      count = sscanf(capture_id, "%u", &screen->renderdoc_capture_start);
      if (!count) {
         if (!strcmp(capture_id, "all")) {
            screen->renderdoc_capture_all = true;
         } else {
            printf("`ZINK_RENDERDOC` usage: ZINK_RENDERDOC=all|frame_no[:end_frame_no]\n");
            abort();
         }
      }
      screen->renderdoc_capture_end = screen->renderdoc_capture_start;
   }
   p_atomic_set(&screen->renderdoc_frame, 1);
#endif
}

bool
zink_screen_init_semaphore(struct zink_screen *screen)
{
   VkSemaphoreCreateInfo sci = {0};
   VkSemaphoreTypeCreateInfo tci = {0};
   sci.pNext = &tci;
   sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
   tci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
   tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;

   return VKSCR(CreateSemaphore)(screen->dev, &sci, NULL, &screen->sem) == VK_SUCCESS;
}

VkSemaphore
zink_create_exportable_semaphore(struct zink_screen *screen)
{
   VkExportSemaphoreCreateInfo eci = {
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      NULL,
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT
   };
   VkSemaphoreCreateInfo sci = {
      VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      &eci,
      0
   };

   VkSemaphore sem = VK_NULL_HANDLE;
   if (util_dynarray_contains(&screen->fd_semaphores, VkSemaphore)) {
      simple_mtx_lock(&screen->semaphores_lock);
      if (util_dynarray_contains(&screen->fd_semaphores, VkSemaphore))
         sem = util_dynarray_pop(&screen->fd_semaphores, VkSemaphore);
      simple_mtx_unlock(&screen->semaphores_lock);
   }
   if (sem)
      return sem;
   VkResult ret = VKSCR(CreateSemaphore)(screen->dev, &sci, NULL, &sem);
   return ret == VK_SUCCESS ? sem : VK_NULL_HANDLE;
}

#if defined(HAVE_LIBDRM) && (DETECT_OS_LINUX || DETECT_OS_BSD)
static int
zink_resource_get_dma_buf(struct zink_screen *screen, struct zink_resource *res)
{
   if (res->obj->is_aux) {
      return os_dupfd_cloexec(res->obj->handle);
   } else {
      VkMemoryGetFdInfoKHR fd_info = {0};
      fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
      fd_info.memory = zink_bo_get_mem(res->obj->bo);
      fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

      int fd;
      if (VKSCR(GetMemoryFdKHR)(screen->dev, &fd_info, &fd) != VK_SUCCESS)
         return -1;

      return fd;
   }
}
#endif

VkSemaphore
zink_screen_export_dmabuf_semaphore(struct zink_screen *screen, struct zink_resource *res)
{
   VkSemaphore sem = VK_NULL_HANDLE;
#if defined(HAVE_LIBDRM) && (DETECT_OS_LINUX || DETECT_OS_BSD)
   struct dma_buf_export_sync_file export = {
      .flags = DMA_BUF_SYNC_RW,
      .fd = -1,
   };

   int fd = zink_resource_get_dma_buf(screen, res);
   if (unlikely(fd < 0)) {
      mesa_loge("MESA: Unable to get a valid memory fd");
      return VK_NULL_HANDLE;
   }

   int ret = drmIoctl(fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &export);
   close(fd);
   if (ret) {
      if (errno == ENOTTY || errno == EBADF || errno == ENOSYS) {
         assert(!"how did this fail?");
         return VK_NULL_HANDLE;
      } else {
         mesa_loge("MESA: failed to import sync file '%s'", strerror(errno));
         return VK_NULL_HANDLE;
      }
   }

   sem = zink_create_exportable_semaphore(screen);

   const VkImportSemaphoreFdInfoKHR sdi = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
      .semaphore = sem,
      .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
      .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
      .fd = export.fd,
   };
   bool success = VKSCR(ImportSemaphoreFdKHR)(screen->dev, &sdi) == VK_SUCCESS;
   if (!success) {
      close(export.fd);
      VKSCR(DestroySemaphore)(screen->dev, sem, NULL);
      return VK_NULL_HANDLE;
   }
#endif
   return sem;
}

bool
zink_screen_import_dmabuf_semaphore(struct zink_screen *screen, struct zink_resource *res, VkSemaphore sem)
{
#if defined(HAVE_LIBDRM) && (DETECT_OS_LINUX || DETECT_OS_BSD)
   const VkSemaphoreGetFdInfoKHR get_fd_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
      .semaphore = sem,
      .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
   };
   int sync_file_fd = -1;
   VkResult result = VKSCR(GetSemaphoreFdKHR)(screen->dev, &get_fd_info, &sync_file_fd);
   if (result != VK_SUCCESS) {
      return false;
   }

   bool ret = false;
   int fd = zink_resource_get_dma_buf(screen, res);
   if (fd != -1) {
      struct dma_buf_import_sync_file import = {
         .flags = DMA_BUF_SYNC_RW,
         .fd = sync_file_fd,
      };
      int ioctl_ret = drmIoctl(fd, DMA_BUF_IOCTL_IMPORT_SYNC_FILE, &import);
      if (ioctl_ret) {
         if (errno == ENOTTY || errno == EBADF || errno == ENOSYS) {
            assert(!"how did this fail?");
         } else {
            ret = true;
         }
      }
      close(fd);
   }
   close(sync_file_fd);
   return ret;
#else
   return true;
#endif
}

bool
zink_screen_timeline_wait(struct zink_screen *screen, uint64_t batch_id, uint64_t timeout)
{
   VkSemaphoreWaitInfo wi = {0};

   if (zink_screen_check_last_finished(screen, batch_id))
      return true;

   wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
   wi.semaphoreCount = 1;
   wi.pSemaphores = &screen->sem;
   wi.pValues = &batch_id;
   bool success = false;
   if (screen->device_lost)
      return true;
   VkResult ret = VKSCR(WaitSemaphores)(screen->dev, &wi, timeout);
   success = zink_screen_handle_vkresult(screen, ret);

   if (success)
      zink_screen_update_last_finished(screen, batch_id);

   return success;
}

static uint32_t
zink_get_loader_version(struct zink_screen *screen)
{

   uint32_t loader_version = VK_API_VERSION_1_0;

   // Get the Loader version
   GET_PROC_ADDR_INSTANCE_LOCAL(screen, NULL, EnumerateInstanceVersion);
   if (vk_EnumerateInstanceVersion) {
      uint32_t loader_version_temp = VK_API_VERSION_1_0;
      VkResult result = (*vk_EnumerateInstanceVersion)(&loader_version_temp);
      if (VK_SUCCESS == result) {
         loader_version = loader_version_temp;
      } else {
         mesa_loge("ZINK: vkEnumerateInstanceVersion failed (%s)", vk_Result_to_str(result));
      }
   }

   return loader_version;
}

static void
zink_query_memory_info(struct pipe_screen *pscreen, struct pipe_memory_info *info)
{
   struct zink_screen *screen = zink_screen(pscreen);
   memset(info, 0, sizeof(struct pipe_memory_info));
   if (screen->info.have_EXT_memory_budget && VKSCR(GetPhysicalDeviceMemoryProperties2)) {
      VkPhysicalDeviceMemoryProperties2 mem = {0};
      mem.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;

      VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {0};
      budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
      mem.pNext = &budget;
      VKSCR(GetPhysicalDeviceMemoryProperties2)(screen->pdev, &mem);

      for (unsigned i = 0; i < mem.memoryProperties.memoryHeapCount; i++) {
         if (mem.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            /* VRAM */
            info->total_device_memory += mem.memoryProperties.memoryHeaps[i].size / 1024;
            info->avail_device_memory += (mem.memoryProperties.memoryHeaps[i].size - budget.heapUsage[i]) / 1024;
         } else {
            /* GART */
            info->total_staging_memory += mem.memoryProperties.memoryHeaps[i].size / 1024;
            info->avail_staging_memory += (mem.memoryProperties.memoryHeaps[i].size - budget.heapUsage[i]) / 1024;
         }
      }
      /* evictions not yet supported in vulkan */
   } else {
      for (unsigned i = 0; i < screen->info.mem_props.memoryHeapCount; i++) {
         if (screen->info.mem_props.memoryHeaps[i].flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            /* VRAM */
            info->total_device_memory += screen->info.mem_props.memoryHeaps[i].size / 1024;
            /* free real estate! */
            info->avail_device_memory += info->total_device_memory;
         } else {
            /* GART */
            info->total_staging_memory += screen->info.mem_props.memoryHeaps[i].size / 1024;
            /* free real estate! */
            info->avail_staging_memory += info->total_staging_memory;
         }
      }
   }
}

static void
zink_query_dmabuf_modifiers(struct pipe_screen *pscreen, enum pipe_format format, int max, uint64_t *modifiers, unsigned int *external_only, int *count)
{
   struct zink_screen *screen = zink_screen(pscreen);
   const struct zink_modifier_props *props = zink_get_modifier_props(screen, format);
   *count = props->drmFormatModifierCount;
   for (int i = 0; i < MIN2(max, *count); i++) {
      modifiers[i] = props->pDrmFormatModifierProperties[i].drmFormatModifier;
      if (external_only)
         external_only[i] = !(props->pDrmFormatModifierProperties[i].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
   }
}

static bool
zink_is_dmabuf_modifier_supported(struct pipe_screen *pscreen, uint64_t modifier, enum pipe_format format, bool *external_only)
{
   struct zink_screen *screen = zink_screen(pscreen);
   const struct zink_modifier_props *props = zink_get_modifier_props(screen, format);
   for (unsigned i = 0; i < props->drmFormatModifierCount; i++)
      if (props->pDrmFormatModifierProperties[i].drmFormatModifier == modifier)
         return true;
   return false;
}

static unsigned
zink_get_dmabuf_modifier_planes(struct pipe_screen *pscreen, uint64_t modifier, enum pipe_format format)
{
   struct zink_screen *screen = zink_screen(pscreen);
   const struct zink_modifier_props *props = zink_get_modifier_props(screen, format);
   for (unsigned i = 0; i < props->drmFormatModifierCount; i++)
      if (props->pDrmFormatModifierProperties[i].drmFormatModifier == modifier)
         return props->pDrmFormatModifierProperties[i].drmFormatModifierPlaneCount;
   return util_format_get_num_planes(format);
}

static int
zink_get_sparse_texture_virtual_page_size(struct pipe_screen *pscreen,
                                          enum pipe_texture_target target,
                                          bool multi_sample,
                                          enum pipe_format pformat,
                                          unsigned offset, unsigned size,
                                          int *x, int *y, int *z)
{
   struct zink_screen *screen = zink_screen(pscreen);
   static const int page_size_2d[][3] = {
      { 256, 256, 1 }, /* 8bpp   */
      { 256, 128, 1 }, /* 16bpp  */
      { 128, 128, 1 }, /* 32bpp  */
      { 128, 64,  1 }, /* 64bpp  */
      { 64,  64,  1 }, /* 128bpp */
   };
   static const int page_size_3d[][3] = {
      { 64,  32,  32 }, /* 8bpp   */
      { 32,  32,  32 }, /* 16bpp  */
      { 32,  32,  16 }, /* 32bpp  */
      { 32,  16,  16 }, /* 64bpp  */
      { 16,  16,  16 }, /* 128bpp */
   };
   /* Only support one type of page size. */
   if (offset != 0)
      return 0;

   /* reject multisample if 2x isn't supported; assume none are */
   if (multi_sample && !screen->info.feats.features.sparseResidency2Samples)
      return 0;

   VkFormat format = zink_get_format(screen, pformat);
   bool is_zs = util_format_is_depth_or_stencil(pformat);
   VkImageType type;
   switch (target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
      type = (screen->need_2D_sparse || (screen->need_2D_zs && is_zs)) ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_1D;
      break;

   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_RECT:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE_ARRAY:
      type = VK_IMAGE_TYPE_2D;
      break;

   case PIPE_TEXTURE_3D:
      type = VK_IMAGE_TYPE_3D;
      break;

   case PIPE_BUFFER:
      goto hack_it_up;

   default:
      return 0;
   }

   VkImageUsageFlags use_flags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 VK_IMAGE_USAGE_STORAGE_BIT;
   use_flags |= is_zs ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   VkImageUsageFlags flags = zink_get_format_props(screen, pformat)->optimalTilingFeatures & use_flags;
   VkSparseImageFormatProperties props[4]; //planar?
   unsigned prop_count = ARRAY_SIZE(props);
   VKSCR(GetPhysicalDeviceSparseImageFormatProperties)(screen->pdev, format, type,
                                                       multi_sample ? VK_SAMPLE_COUNT_2_BIT : VK_SAMPLE_COUNT_1_BIT,
                                                       flags,
                                                       VK_IMAGE_TILING_OPTIMAL,
                                                       &prop_count, props);
   if (!prop_count) {
      /* format may not support storage; try without */
      flags &= ~VK_IMAGE_USAGE_STORAGE_BIT;
      prop_count = ARRAY_SIZE(props);
      VKSCR(GetPhysicalDeviceSparseImageFormatProperties)(screen->pdev, format, type,
                                                         multi_sample ? VK_SAMPLE_COUNT_2_BIT : VK_SAMPLE_COUNT_1_BIT,
                                                         flags,
                                                         VK_IMAGE_TILING_OPTIMAL,
                                                         &prop_count, props);
      if (!prop_count)
         return 0;
   }

   if (size) {
      if (x)
         *x = props[0].imageGranularity.width;
      if (y)
         *y = props[0].imageGranularity.height;
      if (z)
         *z = props[0].imageGranularity.depth;
   }

   return 1;
hack_it_up:
   {
      const int (*page_sizes)[3] = target == PIPE_TEXTURE_3D ? page_size_3d : page_size_2d;
      int blk_size = util_format_get_blocksize(pformat);

      if (size) {
         unsigned index = util_logbase2(blk_size);
         if (x) *x = page_sizes[index][0];
         if (y) *y = page_sizes[index][1];
         if (z) *z = page_sizes[index][2];
      }
   }
   return 1;
}

static VkDevice
get_device(struct zink_screen *screen, VkDeviceCreateInfo *dci)
{
   VkDevice dev = VK_NULL_HANDLE;

   simple_mtx_lock(&device_lock);

   if (!device_table.table)
      _mesa_set_init(&device_table, NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);

   set_foreach(&device_table, entry) {
      struct zink_device *zdev = (void*)entry->key;
      if (zdev->pdev != screen->pdev)
         continue;
      zdev->refcount++;
      simple_mtx_unlock(&device_lock);
      return zdev->dev;
   }

   VkResult result = VKSCR(CreateDevice)(screen->pdev, dci, NULL, &dev);
   if (result != VK_SUCCESS)
      mesa_loge("ZINK: vkCreateDevice failed (%s)", vk_Result_to_str(result));

   struct zink_device *zdev = malloc(sizeof(struct zink_device));
   zdev->refcount = 1;
   zdev->pdev = screen->pdev;
   zdev->dev = dev;
   _mesa_set_add(&device_table, zdev);
   simple_mtx_unlock(&device_lock);
   return dev;
}

static VkDevice
zink_create_logical_device(struct zink_screen *screen)
{
   VkDeviceQueueCreateInfo qci[2] = {0};
   uint32_t queues[3] = {
      screen->gfx_queue,
      screen->sparse_queue,
   };
   float dummy = 0.0f;
   for (unsigned i = 0; i < ARRAY_SIZE(qci); i++) {
      qci[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qci[i].queueFamilyIndex = queues[i];
      qci[i].queueCount = 1;
      qci[i].pQueuePriorities = &dummy;
   }

   unsigned num_queues = 1;
   if (screen->sparse_queue != screen->gfx_queue)
      num_queues++;

   VkDeviceCreateInfo dci = {0};
   dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
   dci.queueCreateInfoCount = num_queues;
   dci.pQueueCreateInfos = qci;
   /* extensions don't have bool members in pEnabledFeatures.
    * this requires us to pass the whole VkPhysicalDeviceFeatures2 struct
    */
   if (screen->info.feats.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
      dci.pNext = &screen->info.feats;
   } else {
      dci.pEnabledFeatures = &screen->info.feats.features;
   }

   dci.ppEnabledExtensionNames = screen->info.extensions;
   dci.enabledExtensionCount = screen->info.num_extensions;

   return get_device(screen, &dci);
}

static void
check_base_requirements(struct zink_screen *screen)
{
   if (zink_debug & ZINK_DEBUG_QUIET)
      return;
   if (zink_driverid(screen) == VK_DRIVER_ID_MESA_V3DV) {
      /* v3dv doesn't support straddling i/o, but zink doesn't do that so this is effectively supported:
       * don't spam errors in this case
       */
      screen->info.feats12.scalarBlockLayout = true;
      screen->info.have_EXT_scalar_block_layout = true;
   }
   if (!screen->info.feats.features.logicOp ||
       !screen->info.feats.features.fillModeNonSolid ||
       !screen->info.feats.features.shaderClipDistance ||
       !(screen->info.feats12.scalarBlockLayout ||
         screen->info.have_EXT_scalar_block_layout) ||
       !screen->info.have_KHR_maintenance1 ||
       !screen->info.have_EXT_custom_border_color ||
       !screen->info.have_EXT_line_rasterization) {
      fprintf(stderr, "WARNING: Some incorrect rendering "
              "might occur because the selected Vulkan device (%s) doesn't support "
              "base Zink requirements: ", screen->info.props.deviceName);
#define CHECK_OR_PRINT(X) \
      if (!screen->info.X) \
         fprintf(stderr, "%s ", #X)
      CHECK_OR_PRINT(feats.features.logicOp);
      CHECK_OR_PRINT(feats.features.fillModeNonSolid);
      CHECK_OR_PRINT(feats.features.shaderClipDistance);
      if (!screen->info.feats12.scalarBlockLayout && !screen->info.have_EXT_scalar_block_layout)
         fprintf(stderr, "scalarBlockLayout OR EXT_scalar_block_layout ");
      CHECK_OR_PRINT(have_KHR_maintenance1);
      CHECK_OR_PRINT(have_EXT_custom_border_color);
      CHECK_OR_PRINT(have_EXT_line_rasterization);
      fprintf(stderr, "\n");
   }
   if (zink_driverid(screen) == VK_DRIVER_ID_MESA_V3DV) {
      screen->info.feats12.scalarBlockLayout = false;
      screen->info.have_EXT_scalar_block_layout = false;
   }
}

static void
zink_get_sample_pixel_grid(struct pipe_screen *pscreen, unsigned sample_count,
                           unsigned *width, unsigned *height)
{
   struct zink_screen *screen = zink_screen(pscreen);
   unsigned idx = util_logbase2_ceil(MAX2(sample_count, 1));
   assert(idx < ARRAY_SIZE(screen->maxSampleLocationGridSize));
   *width = screen->maxSampleLocationGridSize[idx].width;
   *height = screen->maxSampleLocationGridSize[idx].height;
}

static void
init_driver_workarounds(struct zink_screen *screen)
{
   /* enable implicit sync for all non-mesa drivers */
   screen->driver_workarounds.implicit_sync = !zink_driver_is_venus(screen);
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_RADV:
   case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
   case VK_DRIVER_ID_MESA_LLVMPIPE:
   case VK_DRIVER_ID_MESA_TURNIP:
   case VK_DRIVER_ID_MESA_V3DV:
   case VK_DRIVER_ID_MESA_PANVK:
   case VK_DRIVER_ID_MESA_NVK:
      screen->driver_workarounds.implicit_sync = false;
      break;
   default:
      break;
   }
   /* TODO: maybe compile multiple variants for different set counts for compact mode? */
   if (screen->info.props.limits.maxBoundDescriptorSets < ZINK_DESCRIPTOR_ALL_TYPES ||
       zink_debug & (ZINK_DEBUG_COMPACT | ZINK_DEBUG_NOSHOBJ))
      screen->info.have_EXT_shader_object = false;
   /* EDS2 is only used with EDS1 */
   if (!screen->info.have_EXT_extended_dynamic_state) {
      screen->info.have_EXT_extended_dynamic_state2 = false;
      /* CWE usage needs EDS1 */
      screen->info.have_EXT_color_write_enable = false;
   }
   if (zink_driverid(screen) == VK_DRIVER_ID_AMD_PROPRIETARY)
      /* this completely breaks xfb somehow */
      screen->info.have_EXT_extended_dynamic_state2 = false;
   /* EDS3 is only used with EDS2 */
   if (!screen->info.have_EXT_extended_dynamic_state2)
      screen->info.have_EXT_extended_dynamic_state3 = false;
   /* EXT_vertex_input_dynamic_state is only used with EDS2 and above */
   if (!screen->info.have_EXT_extended_dynamic_state2)
      screen->info.have_EXT_vertex_input_dynamic_state = false;
   if (screen->info.line_rast_feats.stippledRectangularLines &&
       screen->info.line_rast_feats.stippledBresenhamLines &&
       screen->info.line_rast_feats.stippledSmoothLines &&
       !screen->info.dynamic_state3_feats.extendedDynamicState3LineStippleEnable)
      screen->info.have_EXT_extended_dynamic_state3 = false;
   if (!screen->info.dynamic_state3_feats.extendedDynamicState3PolygonMode ||
       !screen->info.dynamic_state3_feats.extendedDynamicState3DepthClampEnable ||
       !screen->info.dynamic_state3_feats.extendedDynamicState3DepthClipNegativeOneToOne ||
       !screen->info.dynamic_state3_feats.extendedDynamicState3DepthClipEnable ||
       !screen->info.dynamic_state3_feats.extendedDynamicState3ProvokingVertexMode ||
       !screen->info.dynamic_state3_feats.extendedDynamicState3LineRasterizationMode)
      screen->info.have_EXT_extended_dynamic_state3 = false;
   else if (screen->info.dynamic_state3_feats.extendedDynamicState3SampleMask &&
            screen->info.dynamic_state3_feats.extendedDynamicState3AlphaToCoverageEnable &&
            (!screen->info.feats.features.alphaToOne || screen->info.dynamic_state3_feats.extendedDynamicState3AlphaToOneEnable) &&
            screen->info.dynamic_state3_feats.extendedDynamicState3ColorBlendEnable &&
            screen->info.dynamic_state3_feats.extendedDynamicState3RasterizationSamples &&
            screen->info.dynamic_state3_feats.extendedDynamicState3ColorWriteMask &&
            screen->info.dynamic_state3_feats.extendedDynamicState3ColorBlendEquation &&
            screen->info.dynamic_state3_feats.extendedDynamicState3LogicOpEnable &&
            screen->info.dynamic_state2_feats.extendedDynamicState2LogicOp)
      screen->have_full_ds3 = true;
   if (screen->info.have_EXT_graphics_pipeline_library)
      screen->info.have_EXT_graphics_pipeline_library = screen->info.have_EXT_extended_dynamic_state &&
                                                        screen->info.have_EXT_extended_dynamic_state2 &&
                                                        ((zink_debug & ZINK_DEBUG_GPL) ||
                                                         screen->info.dynamic_state2_feats.extendedDynamicState2PatchControlPoints) &&
                                                        screen->info.have_EXT_extended_dynamic_state3 &&
                                                        screen->info.have_EXT_non_seamless_cube_map &&
                                                        (!(zink_debug & ZINK_DEBUG_GPL) ||
                                                         screen->info.gpl_props.graphicsPipelineLibraryFastLinking ||
                                                         screen->is_cpu);
   screen->driver_workarounds.broken_l4a4 = zink_driverid(screen) == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
   if (zink_driverid(screen) == VK_DRIVER_ID_MESA_TURNIP) {
      /* performance */
      screen->info.border_color_feats.customBorderColorWithoutFormat = VK_FALSE;
   }
   if (!screen->info.have_KHR_maintenance5)
      screen->driver_workarounds.missing_a8_unorm = true;

   if ((!screen->info.have_EXT_line_rasterization ||
        !screen->info.line_rast_feats.stippledBresenhamLines) &&
       screen->info.feats.features.geometryShader &&
       screen->info.feats.features.sampleRateShading) {
      /* we're using stippledBresenhamLines as a proxy for all of these, to
       * avoid accidentally changing behavior on VK-drivers where we don't
       * want to add emulation.
       */
      screen->driver_workarounds.no_linestipple = true;
   }

   if (zink_driverid(screen) ==
       VK_DRIVER_ID_IMAGINATION_PROPRIETARY &&
       screen->info.feats.features.geometryShader)
      screen->driver_workarounds.no_linesmooth = true;

   /* This is a workarround for the lack of
    * gl_PointSize + glPolygonMode(..., GL_LINE), in the imagination
    * proprietary driver.
    */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
      screen->driver_workarounds.no_hw_gl_point = true;
      break;
   default:
      screen->driver_workarounds.no_hw_gl_point = false;
      break;
   }

   /* these drivers don't use VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT, so it can always be set */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_RADV:
   case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
   case VK_DRIVER_ID_MESA_LLVMPIPE:
   case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
   case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
      screen->driver_workarounds.always_feedback_loop = screen->info.have_EXT_attachment_feedback_loop_layout;
      break;
   default:
      break;
   }
   /* these drivers don't use VK_PIPELINE_CREATE_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT, so it can always be set */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_LLVMPIPE:
   case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
      screen->driver_workarounds.always_feedback_loop_zs = screen->info.have_EXT_attachment_feedback_loop_layout;
      break;
   default:
      break;
   }
   /* use same mechanics if dynamic state is supported */
   screen->driver_workarounds.always_feedback_loop |= screen->info.have_EXT_attachment_feedback_loop_dynamic_state;
   screen->driver_workarounds.always_feedback_loop_zs |= screen->info.have_EXT_attachment_feedback_loop_dynamic_state;

   /* these drivers cannot handle OOB gl_Layer values, and therefore need clamping in shader.
    * TODO: Vulkan extension that details whether vulkan driver can handle OOB layer values
    */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
      screen->driver_compiler_workarounds.needs_sanitised_layer = true;
      break;
   default:
      screen->driver_compiler_workarounds.needs_sanitised_layer = false;
      break;
   }
   /* these drivers will produce undefined results when using swizzle 1 with combined z/s textures
    * TODO: use a future device property when available
    */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
   case VK_DRIVER_ID_IMAGINATION_OPEN_SOURCE_MESA:
      screen->driver_compiler_workarounds.needs_zs_shader_swizzle = true;
      break;
   default:
      screen->driver_compiler_workarounds.needs_zs_shader_swizzle = false;
      break;
   }

   /* these drivers cannot handle arbitary const value types */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
      screen->driver_compiler_workarounds.broken_const = true;
      break;
   default:
      screen->driver_compiler_workarounds.broken_const = false;
      break;
   }

   /* these drivers do not implement demote properly */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
      screen->driver_compiler_workarounds.broken_demote = true;
      break;
   default:
      screen->driver_compiler_workarounds.broken_demote = false;
      break;
   }

   /* When robust contexts are advertised but robustImageAccess2 is not available */
   screen->driver_compiler_workarounds.lower_robustImageAccess2 =
      !screen->info.rb2_feats.robustImageAccess2 &&
      screen->info.feats.features.robustBufferAccess &&
      screen->info.rb_image_feats.robustImageAccess;

   /* once more testing has been done, use the #if 0 block */
   unsigned illegal = ZINK_DEBUG_RP | ZINK_DEBUG_NORP;
   if ((zink_debug & illegal) == illegal) {
      mesa_loge("Cannot specify ZINK_DEBUG=rp and ZINK_DEBUG=norp");
      abort();
   }

   /* these drivers benefit from renderpass optimization */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_LLVMPIPE:
   case VK_DRIVER_ID_MESA_TURNIP:
   case VK_DRIVER_ID_MESA_PANVK:
   case VK_DRIVER_ID_MESA_V3DV:
   case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
   case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
   case VK_DRIVER_ID_BROADCOM_PROPRIETARY:
   case VK_DRIVER_ID_ARM_PROPRIETARY:
   case VK_DRIVER_ID_MESA_HONEYKRISP:
      screen->driver_workarounds.track_renderpasses = true; //screen->info.primgen_feats.primitivesGeneratedQueryWithRasterizerDiscard
      break;
   default:
      break;
   }
   if (zink_debug & ZINK_DEBUG_RP)
      screen->driver_workarounds.track_renderpasses = true;
   else if (zink_debug & ZINK_DEBUG_NORP)
      screen->driver_workarounds.track_renderpasses = false;

   /* these drivers can successfully do INVALID <-> LINEAR dri3 modifier swap */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_TURNIP:
   case VK_DRIVER_ID_MESA_NVK:
   case VK_DRIVER_ID_MESA_LLVMPIPE:
      screen->driver_workarounds.can_do_invalid_linear_modifier = true;
      break;
   default:
      break;
   }
   if (zink_driver_is_venus(screen))
      screen->driver_workarounds.can_do_invalid_linear_modifier = true;

   /* these drivers have no difference between unoptimized and optimized shader compilation */
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_LLVMPIPE:
      screen->driver_workarounds.disable_optimized_compile = true;
      break;
   default:
      if (zink_debug & ZINK_DEBUG_NOOPT)
         screen->driver_workarounds.disable_optimized_compile = true;
      break;
   }

   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_RADV:
   case VK_DRIVER_ID_AMD_OPEN_SOURCE:
   case VK_DRIVER_ID_AMD_PROPRIETARY:
      /* this has bad perf on AMD */
      screen->info.have_KHR_push_descriptor = false;
      /* Interpolation is not consistent between two triangles of a rectangle. */
      screen->driver_workarounds.inconsistent_interpolation = true;
      break;
   default:
      break;
   }

   screen->driver_workarounds.can_2d_view_sparse = true;
   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
   case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
      /* this does wild things to block shapes */
      screen->driver_workarounds.can_2d_view_sparse = false;
      break;
   default:
      break;
   }

   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_RADV:
   case VK_DRIVER_ID_MESA_NVK:
   case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
      screen->driver_workarounds.general_depth_layout = true;
      break;
   default:
      break;
   }

   switch (zink_driverid(screen)) {
   case VK_DRIVER_ID_MESA_LLVMPIPE:
   case VK_DRIVER_ID_MESA_NVK:
   case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
   case VK_DRIVER_ID_MESA_TURNIP:
   case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
      screen->driver_workarounds.general_layout = true;
      break;
   default:
      screen->driver_workarounds.general_layout = screen->info.have_KHR_unified_image_layouts;
      break;
   }

   if (!screen->resizable_bar)
      screen->info.have_EXT_host_image_copy = false;
}

static void
check_hic_shader_read(struct zink_screen *screen)
{
   if (screen->info.have_EXT_host_image_copy) {
      for (unsigned i = 0; i < screen->info.hic_props.copyDstLayoutCount; i++) {
         if (screen->info.hic_props.pCopyDstLayouts[i] == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            screen->can_hic_shader_read = true;
            break;
         }
      }
   }
}

static void
init_optimal_keys(struct zink_screen *screen)
{
   /* assume that anyone who knows enough to enable optimal_keys on turnip doesn't care about missing line stipple */
   if (zink_debug & ZINK_DEBUG_OPTIMAL_KEYS && zink_driverid(screen) == VK_DRIVER_ID_MESA_TURNIP)
      zink_debug |= ZINK_DEBUG_QUIET;
   screen->optimal_keys = !screen->need_decompose_attrs &&
                          screen->info.have_EXT_non_seamless_cube_map &&
                          screen->info.have_EXT_provoking_vertex &&
                          !screen->driconf.inline_uniforms &&
                          !screen->driver_workarounds.no_linestipple &&
                          !screen->driver_workarounds.no_linesmooth &&
                          !screen->driver_workarounds.no_hw_gl_point &&
                          !screen->driver_compiler_workarounds.lower_robustImageAccess2 &&
                          !screen->driconf.emulate_point_smooth &&
                          !screen->driver_compiler_workarounds.needs_zs_shader_swizzle;
   if (!screen->optimal_keys && zink_debug & ZINK_DEBUG_OPTIMAL_KEYS && !(zink_debug & ZINK_DEBUG_QUIET)) {
      fprintf(stderr, "The following criteria are preventing optimal_keys enablement:\n");
      if (screen->need_decompose_attrs)
         fprintf(stderr, "missing vertex attribute formats\n");
      if (screen->driconf.inline_uniforms)
         fprintf(stderr, "uniform inlining must be disabled (set ZINK_INLINE_UNIFORMS=0 in your env)\n");
      if (screen->driconf.emulate_point_smooth)
         fprintf(stderr, "smooth point emulation is enabled\n");
      if (screen->driver_compiler_workarounds.needs_zs_shader_swizzle)
         fprintf(stderr, "Z/S shader swizzle workaround is enabled\n");
      CHECK_OR_PRINT(have_EXT_line_rasterization);
      CHECK_OR_PRINT(line_rast_feats.stippledBresenhamLines);
      CHECK_OR_PRINT(feats.features.geometryShader);
      CHECK_OR_PRINT(feats.features.sampleRateShading);
      CHECK_OR_PRINT(have_EXT_non_seamless_cube_map);
      CHECK_OR_PRINT(have_EXT_provoking_vertex);
      if (screen->driver_workarounds.no_linesmooth)
         fprintf(stderr, "driver does not support smooth lines\n");
      if (screen->driver_workarounds.no_hw_gl_point)
         fprintf(stderr, "driver does not support hardware GL_POINT\n");
      CHECK_OR_PRINT(rb2_feats.robustImageAccess2);
      CHECK_OR_PRINT(feats.features.robustBufferAccess);
      CHECK_OR_PRINT(rb_image_feats.robustImageAccess);
      printf("\n");
      mesa_logw("zink: force-enabling optimal_keys despite missing features. Good luck!");
   }
   if (zink_debug & ZINK_DEBUG_OPTIMAL_KEYS)
      screen->optimal_keys = true;
   if (!screen->optimal_keys)
      screen->info.have_EXT_graphics_pipeline_library = false;

   if (!screen->optimal_keys ||
       !screen->info.have_KHR_maintenance5 ||
      /* EXT_shader_object needs either dynamic feedback loop or per-app enablement */
       (!screen->driconf.zink_shader_object_enable && !screen->info.have_EXT_attachment_feedback_loop_dynamic_state))
      screen->info.have_EXT_shader_object = false;
   if (screen->info.have_EXT_shader_object)
      screen->have_full_ds3 = true;
}

static struct disk_cache *
zink_get_disk_shader_cache(struct pipe_screen *_screen)
{
   struct zink_screen *screen = zink_screen(_screen);

   return screen->disk_cache;
}

VkSemaphore
zink_create_semaphore(struct zink_screen *screen)
{
   VkSemaphoreCreateInfo sci = {
      VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      NULL,
      0
   };
   VkSemaphore sem = VK_NULL_HANDLE;
   if (util_dynarray_contains(&screen->semaphores, VkSemaphore)) {
      simple_mtx_lock(&screen->semaphores_lock);
      if (util_dynarray_contains(&screen->semaphores, VkSemaphore))
         sem = util_dynarray_pop(&screen->semaphores, VkSemaphore);
      simple_mtx_unlock(&screen->semaphores_lock);
   }
   if (sem)
      return sem;
   VkResult ret = VKSCR(CreateSemaphore)(screen->dev, &sci, NULL, &sem);
   return ret == VK_SUCCESS ? sem : VK_NULL_HANDLE;
}

void
zink_screen_lock_context(struct zink_screen *screen)
{
   simple_mtx_lock(&screen->copy_context_lock);
   if (!screen->copy_context)
      screen->copy_context = zink_context(screen->base.context_create(&screen->base, NULL, ZINK_CONTEXT_COPY_ONLY));
   if (!screen->copy_context) {
      mesa_loge("zink: failed to create copy context");
      /* realistically there's nothing that can be done here */
   }
}

void
zink_screen_unlock_context(struct zink_screen *screen)
{
   simple_mtx_unlock(&screen->copy_context_lock);
}

static bool
init_layouts(struct zink_screen *screen)
{
   if (screen->info.have_EXT_descriptor_indexing) {
      VkDescriptorSetLayoutBinding bindings[4];
      const unsigned num_bindings = 4;
      VkDescriptorSetLayoutCreateInfo dcslci = {0};
      dcslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      dcslci.pNext = NULL;
      VkDescriptorSetLayoutBindingFlagsCreateInfo fci = {0};
      VkDescriptorBindingFlags flags[4];
      dcslci.pNext = &fci;
      if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_DB)
         dcslci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
      else
         dcslci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
      fci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
      fci.bindingCount = num_bindings;
      fci.pBindingFlags = flags;
      for (unsigned i = 0; i < num_bindings; i++) {
         flags[i] = VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
         if (zink_descriptor_mode != ZINK_DESCRIPTOR_MODE_DB)
            flags[i] |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
      }
      /* there is exactly 1 bindless descriptor set per context, and it has 4 bindings, 1 for each descriptor type */
      for (unsigned i = 0; i < num_bindings; i++) {
         bindings[i].binding = i;
         bindings[i].descriptorType = zink_descriptor_type_from_bindless_index(i);
         bindings[i].descriptorCount = ZINK_MAX_BINDLESS_HANDLES;
         bindings[i].stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT;
         bindings[i].pImmutableSamplers = NULL;
      }

      dcslci.bindingCount = num_bindings;
      dcslci.pBindings = bindings;
      VkResult result = VKSCR(CreateDescriptorSetLayout)(screen->dev, &dcslci, 0, &screen->bindless_layout);
      if (result != VK_SUCCESS) {
         mesa_loge("ZINK: vkCreateDescriptorSetLayout failed (%s)", vk_Result_to_str(result));
         return false;
      }
   }

   screen->gfx_push_constant_layout = zink_pipeline_layout_create(screen, NULL, 0, false, 0);
   return !!screen->gfx_push_constant_layout;
}

static int
zink_screen_get_fd(struct pipe_screen *pscreen)
{
   struct zink_screen *screen = zink_screen(pscreen);

   return screen->drm_fd;
}

static const char*
zink_cl_cts_version(struct pipe_screen *pscreen)
{
   /* https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_437 */
   return "v2024-08-08-00";
}

static struct zink_screen *
zink_internal_create_screen(const struct pipe_screen_config *config, int64_t dev_major, int64_t dev_minor, uint64_t adapter_luid)
{
   if (getenv("ZINK_USE_LAVAPIPE")) {
      mesa_loge("ZINK_USE_LAVAPIPE is obsolete. Use LIBGL_ALWAYS_SOFTWARE\n");
      return NULL;
   }

   struct zink_screen *screen = rzalloc(NULL, struct zink_screen);
   if (!screen) {
      if (!config || !config->driver_name_is_inferred)
         mesa_loge("ZINK: failed to allocate screen");
      return NULL;
   }

   screen->driver_name_is_inferred = config && config->driver_name_is_inferred;
   screen->drm_fd = -1;

   glsl_type_singleton_init_or_ref();
   zink_debug = debug_get_option_zink_debug();
   if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_AUTO)
      zink_descriptor_mode = debug_get_option_zink_descriptor_mode();

   screen->threaded = util_get_cpu_caps()->nr_cpus > 1 && debug_get_bool_option("GALLIUM_THREAD", util_get_cpu_caps()->nr_cpus > 1);
   if (zink_debug & ZINK_DEBUG_FLUSHSYNC)
      screen->threaded_submit = false;
   else
      screen->threaded_submit = screen->threaded;
   screen->abort_on_hang = debug_get_bool_option("ZINK_HANG_ABORT", false);


   u_trace_state_init();

   screen->loader_lib = util_dl_open(VK_LIBNAME);
   if (!screen->loader_lib) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("ZINK: failed to load "VK_LIBNAME);
      goto fail;
   }

   screen->vk_GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)util_dl_get_proc_address(screen->loader_lib, "vkGetInstanceProcAddr");
   screen->vk_GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)util_dl_get_proc_address(screen->loader_lib, "vkGetDeviceProcAddr");
   if (!screen->vk_GetInstanceProcAddr ||
       !screen->vk_GetDeviceProcAddr) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("ZINK: failed to get proc address");
      goto fail;
   }

   if (config) {
      driParseConfigFiles(config->options, config->options_info, 0, "zink",
                          NULL, NULL, NULL, 0, NULL, 0);
      screen->driconf.dual_color_blend_by_location = driQueryOptionb(config->options, "dual_color_blend_by_location");
      //screen->driconf.inline_uniforms = driQueryOptionb(config->options, "radeonsi_inline_uniforms");
      screen->driconf.emulate_point_smooth = driQueryOptionb(config->options, "zink_emulate_point_smooth");
      screen->driconf.zink_shader_object_enable = driQueryOptionb(config->options, "zink_shader_object_enable");
   }

   simple_mtx_lock(&instance_lock);
   if (++instance_refcount == 1) {
      instance_info.loader_version = zink_get_loader_version(screen);
      instance = zink_create_instance(screen, &instance_info);
   }
   if (!instance) {
      /* We don't decrement instance_refcount here. This prevents us from trying
       * to create another instance on subsequent calls.
       */
      simple_mtx_unlock(&instance_lock);
      goto fail;
   }
   screen->instance = instance;
   screen->instance_info = &instance_info;
   simple_mtx_unlock(&instance_lock);

   if (zink_debug & ZINK_DEBUG_VALIDATION) {
      if (!screen->instance_info->have_layer_KHRONOS_validation &&
          !screen->instance_info->have_layer_LUNARG_standard_validation) {
         if (!screen->driver_name_is_inferred)
            mesa_loge("Failed to load validation layer");
         goto fail;
      }
   }

   vk_instance_uncompacted_dispatch_table_load(&screen->vk.instance,
                                                screen->vk_GetInstanceProcAddr,
                                                screen->instance);
   vk_physical_device_uncompacted_dispatch_table_load(&screen->vk.physical_device,
                                                      screen->vk_GetInstanceProcAddr,
                                                      screen->instance);

   zink_verify_instance_extensions(screen);

   if (screen->instance_info->have_EXT_debug_utils &&
      (zink_debug & ZINK_DEBUG_VALIDATION) && !create_debug(screen)) {
      if (!screen->driver_name_is_inferred)
         debug_printf("ZINK: failed to setup debug utils\n");
   }

   choose_pdev(screen, dev_major, dev_minor, adapter_luid);
   if (screen->pdev == VK_NULL_HANDLE) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("ZINK: failed to choose pdev");
      goto fail;
   }
   screen->is_cpu = screen->info.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;

   update_queue_props(screen);

   screen->have_X8_D24_UNORM_PACK32 = zink_is_depth_format_supported(screen,
                                              VK_FORMAT_X8_D24_UNORM_PACK32);
   screen->have_D24_UNORM_S8_UINT = zink_is_depth_format_supported(screen,
                                              VK_FORMAT_D24_UNORM_S8_UINT);
   screen->have_D32_SFLOAT_S8_UINT = zink_is_depth_format_supported(screen,
                                              VK_FORMAT_D32_SFLOAT_S8_UINT);
   screen->have_dynamic_state_vertex_input_binding_stride = true;

   if (!zink_get_physical_device_info(screen)) {
      if (!screen->driver_name_is_inferred)
         debug_printf("ZINK: failed to detect features\n");
      goto fail;
   }

   if (!screen->info.rb2_feats.nullDescriptor) {
      mesa_loge("Zink requires the nullDescriptor feature of KHR/EXT robustness2.");
      goto fail;
   }

   if (zink_set_driver_strings(screen)) {
      mesa_loge("ZINK: failed to set driver strings\n");
      goto fail;
   }

   memset(&screen->heap_map, UINT8_MAX, sizeof(screen->heap_map));
   for (enum zink_heap i = 0; i < ZINK_HEAP_MAX; i++) {
      for (unsigned j = 0; j < screen->info.mem_props.memoryTypeCount; j++) {
         VkMemoryPropertyFlags domains = vk_domain_from_heap(i);
         if ((screen->info.mem_props.memoryTypes[j].propertyFlags & domains) == domains) {
            screen->heap_map[i][screen->heap_count[i]++] = j;
         }
      }
   }

   bool maybe_has_rebar = true;
   /* iterate again to check for missing heaps */
   for (enum zink_heap i = 0; i < ZINK_HEAP_MAX; i++) {
      /* not found: use compatible heap */
      if (screen->heap_map[i][0] == UINT8_MAX) {
         /* only cached mem has a failure case for now */
         assert(i == ZINK_HEAP_HOST_VISIBLE_COHERENT_CACHED || i == ZINK_HEAP_DEVICE_LOCAL_LAZY ||
                i == ZINK_HEAP_DEVICE_LOCAL_VISIBLE);
         if (i == ZINK_HEAP_HOST_VISIBLE_COHERENT_CACHED) {
            memcpy(screen->heap_map[i], screen->heap_map[ZINK_HEAP_HOST_VISIBLE_COHERENT], screen->heap_count[ZINK_HEAP_HOST_VISIBLE_COHERENT]);
            screen->heap_count[i] = screen->heap_count[ZINK_HEAP_HOST_VISIBLE_COHERENT];
         } else {
            memcpy(screen->heap_map[i], screen->heap_map[ZINK_HEAP_DEVICE_LOCAL], screen->heap_count[ZINK_HEAP_DEVICE_LOCAL]);
            screen->heap_count[i] = screen->heap_count[ZINK_HEAP_DEVICE_LOCAL];
            if (i == ZINK_HEAP_DEVICE_LOCAL_VISIBLE)
               maybe_has_rebar = false;
         }
      }
   }
   if (maybe_has_rebar) {
      uint64_t biggest_vis_vram = 0;
      for (unsigned i = 0; i < screen->heap_count[ZINK_HEAP_DEVICE_LOCAL_VISIBLE]; i++)
         biggest_vis_vram = MAX2(biggest_vis_vram, screen->info.mem_props.memoryHeaps[screen->info.mem_props.memoryTypes[screen->heap_map[ZINK_HEAP_DEVICE_LOCAL_VISIBLE][i]].heapIndex].size);
      uint64_t biggest_vram = 0;
      for (unsigned i = 0; i < screen->heap_count[ZINK_HEAP_DEVICE_LOCAL]; i++)
         biggest_vram = MAX2(biggest_vram, screen->info.mem_props.memoryHeaps[screen->info.mem_props.memoryTypes[screen->heap_map[ZINK_HEAP_DEVICE_LOCAL][i]].heapIndex].size);
      /* determine if vis vram is roughly equal to total vram */
      if (biggest_vis_vram > biggest_vram * 0.9)
         screen->resizable_bar = true;
      if (biggest_vis_vram >= 8ULL * 1024ULL * 1024ULL * 1024ULL)
         screen->always_cached_upload = true;
   }

   setup_renderdoc(screen);
   if (screen->threaded_submit && !util_queue_init(&screen->flush_queue, "zfq", 8, 1, UTIL_QUEUE_INIT_RESIZE_IF_FULL, screen)) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("zink: Failed to create flush queue.\n");
      goto fail;
   }

   zink_internal_setup_moltenvk(screen);
   if (!screen->info.have_KHR_timeline_semaphore && !screen->info.feats12.timelineSemaphore) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("zink: KHR_timeline_semaphore is required");
      goto fail;
   }

   /* Reject IMG blobs with DDK below 24.1@6554834 if not forced */
   if (zink_driverid(screen) == VK_DRIVER_ID_IMAGINATION_PROPRIETARY && screen->info.props.driverVersion < 6554834) {
      debug_printf("zink: Imagination proprietary driver is too old to be supported, expect failure\n");
      if (screen->driver_name_is_inferred)
         goto fail;
   }

   if (zink_debug & ZINK_DEBUG_MEM) {
      simple_mtx_init(&screen->debug_mem_lock, mtx_plain);
      screen->debug_mem_sizes = _mesa_hash_table_create(screen, _mesa_hash_string, _mesa_key_string_equal);
   }

   check_hic_shader_read(screen);

   init_driver_workarounds(screen);

   screen->dev = zink_create_logical_device(screen);
   if (!screen->dev)
      goto fail;

   vk_device_uncompacted_dispatch_table_load(&screen->vk.device,
                                             screen->vk_GetDeviceProcAddr,
                                             screen->dev);

   init_queue(screen);

   zink_verify_device_extensions(screen);

   /* descriptor set indexing is determined by 'compact' descriptor mode:
    * by default, 6 sets are used to provide more granular updating
    * in compact mode, a maximum of 4 sets are used, with like-types combined
    */
   if ((zink_debug & ZINK_DEBUG_COMPACT) ||
       screen->info.props.limits.maxBoundDescriptorSets < ZINK_MAX_DESCRIPTOR_SETS) {
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_UNIFORMS] = 0;
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_UBO] = 1;
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_SSBO] = 1;
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_SAMPLER_VIEW] = 2;
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_IMAGE] = 2;
      screen->desc_set_id[ZINK_DESCRIPTOR_BINDLESS] = 3;
      screen->compact_descriptors = true;
   } else {
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_UNIFORMS] = 0;
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_UBO] = 1;
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_SAMPLER_VIEW] = 2;
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_SSBO] = 3;
      screen->desc_set_id[ZINK_DESCRIPTOR_TYPE_IMAGE] = 4;
      screen->desc_set_id[ZINK_DESCRIPTOR_BINDLESS] = 5;
   }

   if (screen->info.have_EXT_calibrated_timestamps && !check_have_device_time(screen))
      goto fail;

   screen->have_triangle_fans = true;
#if defined(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)
   if (screen->info.have_KHR_portability_subset) {
      screen->have_triangle_fans = (VK_TRUE == screen->info.portability_subset_feats.triangleFans);
   }
#endif // VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME

   check_base_requirements(screen);
   util_live_shader_cache_init(&screen->shaders, zink_create_gfx_shader_state, zink_delete_shader_state);

   for (unsigned i = 0; i < ARRAY_SIZE(screen->base.nir_options); i++)
      screen->base.nir_options[i] = &screen->nir_options;

   screen->base.get_name = zink_get_name;
   if (screen->instance_info->have_KHR_external_memory_capabilities) {
      screen->base.get_device_uuid = zink_get_device_uuid;
      screen->base.get_driver_uuid = zink_get_driver_uuid;
   }
   if (screen->info.have_KHR_external_memory_win32) {
      screen->base.get_device_luid = zink_get_device_luid;
      screen->base.get_device_node_mask = zink_get_device_node_mask;
   }
   screen->base.set_max_shader_compiler_threads = zink_set_max_shader_compiler_threads;
   screen->base.is_parallel_shader_compilation_finished = zink_is_parallel_shader_compilation_finished;
   screen->base.get_vendor = zink_get_vendor;
   screen->base.get_device_vendor = zink_get_device_vendor;
   screen->base.get_timestamp = zink_get_timestamp;
   screen->base.query_memory_info = zink_query_memory_info;
   screen->base.get_sample_pixel_grid = zink_get_sample_pixel_grid;
   screen->base.is_compute_copy_faster = zink_is_compute_copy_faster;
   screen->base.is_format_supported = zink_is_format_supported;
   screen->base.driver_thread_add_job = zink_driver_thread_add_job;
   if (screen->info.have_EXT_image_drm_format_modifier && screen->info.have_EXT_external_memory_dma_buf) {
      screen->base.query_dmabuf_modifiers = zink_query_dmabuf_modifiers;
      screen->base.is_dmabuf_modifier_supported = zink_is_dmabuf_modifier_supported;
      screen->base.get_dmabuf_modifier_planes = zink_get_dmabuf_modifier_planes;
   }
#if defined(_WIN32)
   if (screen->info.have_KHR_external_memory_win32)
      screen->base.create_fence_win32 = zink_create_fence_win32;
#endif
   screen->base.context_create = zink_context_create;
   screen->base.flush_frontbuffer = zink_flush_frontbuffer;
   screen->base.destroy = zink_destroy_screen;
   screen->base.finalize_nir = zink_shader_finalize;
   screen->base.get_disk_shader_cache = zink_get_disk_shader_cache;
   screen->base.get_sparse_texture_virtual_page_size = zink_get_sparse_texture_virtual_page_size;
   screen->base.get_driver_query_group_info = zink_get_driver_query_group_info;
   screen->base.get_driver_query_info = zink_get_driver_query_info;
   screen->base.set_damage_region = zink_set_damage_region;
   screen->base.get_cl_cts_version = zink_cl_cts_version;

   screen->total_video_mem = get_video_mem(screen);
   screen->clamp_video_mem = screen->total_video_mem * 0.8;
   if (!os_get_total_physical_memory(&screen->total_mem)) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("ZINK: failed to get total physical memory");
      goto fail;
   }

   zink_init_shader_caps(screen);
   zink_init_compute_caps(screen);
   zink_init_screen_caps(screen);

   if (screen->info.have_EXT_sample_locations) {
      VkMultisamplePropertiesEXT prop;
      prop.sType = VK_STRUCTURE_TYPE_MULTISAMPLE_PROPERTIES_EXT;
      prop.pNext = NULL;
      for (unsigned i = 0; i < ARRAY_SIZE(screen->maxSampleLocationGridSize); i++) {
         if (screen->info.sample_locations_props.sampleLocationSampleCounts & (1 << i)) {
            VKSCR(GetPhysicalDeviceMultisamplePropertiesEXT)(screen->pdev, 1 << i, &prop);
            screen->maxSampleLocationGridSize[i] = prop.maxSampleLocationGridSize;
         }
      }
   }

   if (!zink_screen_resource_init(&screen->base))
      goto fail;
   if (!zink_bo_init(screen)) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("ZINK: failed to initialize suballocator");
      goto fail;
   }
   zink_screen_fence_init(&screen->base);

   zink_screen_init_compiler(screen);
   if (!disk_cache_init(screen)) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("ZINK: failed to initialize disk cache");
      goto fail;
   }
   if (!util_queue_init(&screen->cache_get_thread, "zcfq", 8, 4,
                        UTIL_QUEUE_INIT_RESIZE_IF_FULL, screen))
      goto fail;
   populate_format_props(screen);

   slab_create_parent(&screen->transfer_pool, sizeof(struct zink_transfer), 16);

   screen->driconf.inline_uniforms = debug_get_bool_option("ZINK_INLINE_UNIFORMS", screen->is_cpu);

   if (!zink_screen_init_semaphore(screen)) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("zink: failed to create timeline semaphore");
      goto fail;
   }

   bool can_db = true;
   {
      if (!screen->info.have_EXT_descriptor_buffer) {
         if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_DB) {
            if (!screen->driver_name_is_inferred)
               mesa_loge("Cannot use db descriptor mode without EXT_descriptor_buffer");
            goto fail;
         }
         can_db = false;
      }
      if (!screen->resizable_bar) {
         if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_DB) {
            if (!screen->driver_name_is_inferred)
               mesa_loge("Cannot use db descriptor mode without resizable bar");
            goto fail;
         }
         can_db = false;
      }
      if (!screen->info.have_EXT_non_seamless_cube_map) {
         if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_DB) {
            if (!screen->driver_name_is_inferred)
               mesa_loge("Cannot use db descriptor mode without EXT_non_seamless_cube_map");
            goto fail;
         }
         can_db = false;
      }
      if (ZINK_FBFETCH_DESCRIPTOR_SIZE < screen->info.db_props.inputAttachmentDescriptorSize) {
         if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_DB) {
            if (!screen->driver_name_is_inferred)
               mesa_loge("Cannot use db descriptor mode with inputAttachmentDescriptorSize(%u) > %u", (unsigned)screen->info.db_props.inputAttachmentDescriptorSize, ZINK_FBFETCH_DESCRIPTOR_SIZE);
            goto fail;
         }
         mesa_logw("zink: bug detected: inputAttachmentDescriptorSize(%u) > %u", (unsigned)screen->info.db_props.inputAttachmentDescriptorSize, ZINK_FBFETCH_DESCRIPTOR_SIZE);
         can_db = false;
      }
      if (screen->info.db_props.maxDescriptorBufferBindings < 2 || screen->info.db_props.maxSamplerDescriptorBufferBindings < 2) {
         if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_DB) {
            /* allow for testing, but disable bindless */
            mesa_logw("Cannot use bindless and db descriptor mode with (maxDescriptorBufferBindings||maxSamplerDescriptorBufferBindings) < 2");
         } else {
            can_db = false;
         }
      }
   }
   if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_AUTO) {
      switch(screen->info.driver_props.driverID) {
      /* descriptor buffer is not performant with virt yet */
      case VK_DRIVER_ID_MESA_VENUS:
      /* db descriptor mode is known to be broken on IMG proprietary drivers */
      case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
         zink_descriptor_mode = ZINK_DESCRIPTOR_MODE_LAZY;
	 break;
      default:
         zink_descriptor_mode = can_db ? ZINK_DESCRIPTOR_MODE_DB : ZINK_DESCRIPTOR_MODE_LAZY;
      }
   }
   if (zink_descriptor_mode == ZINK_DESCRIPTOR_MODE_DB) {
      const uint32_t sampler_size = MAX2(screen->info.db_props.combinedImageSamplerDescriptorSize, screen->info.db_props.robustUniformTexelBufferDescriptorSize);
      const uint32_t image_size = MAX2(screen->info.db_props.storageImageDescriptorSize, screen->info.db_props.robustStorageTexelBufferDescriptorSize);
      if (screen->compact_descriptors) {
         screen->db_size[ZINK_DESCRIPTOR_TYPE_UBO] = screen->info.db_props.robustUniformBufferDescriptorSize +
                                                     screen->info.db_props.robustStorageBufferDescriptorSize;
         screen->db_size[ZINK_DESCRIPTOR_TYPE_SAMPLER_VIEW] = sampler_size + image_size;
      } else {
         screen->db_size[ZINK_DESCRIPTOR_TYPE_UBO] = screen->info.db_props.robustUniformBufferDescriptorSize;
         screen->db_size[ZINK_DESCRIPTOR_TYPE_SAMPLER_VIEW] = sampler_size;
         screen->db_size[ZINK_DESCRIPTOR_TYPE_SSBO] = screen->info.db_props.robustStorageBufferDescriptorSize;
         screen->db_size[ZINK_DESCRIPTOR_TYPE_IMAGE] = image_size;
      }
      screen->db_size[ZINK_DESCRIPTOR_TYPE_UNIFORMS] = screen->info.db_props.robustUniformBufferDescriptorSize;
      screen->info.have_KHR_push_descriptor = false;
      screen->base_descriptor_size = MAX4(screen->db_size[0], screen->db_size[1], screen->db_size[2], screen->db_size[3]);
   }

   simple_mtx_init(&screen->free_batch_states_lock, mtx_plain);
   simple_mtx_init(&screen->dt_lock, mtx_plain);

   util_idalloc_mt_init_tc(&screen->buffer_ids);

   simple_mtx_init(&screen->semaphores_lock, mtx_plain);
   util_dynarray_init(&screen->semaphores, screen);
   util_dynarray_init(&screen->fd_semaphores, screen);

   util_vertex_state_cache_init(&screen->vertex_state_cache,
                                zink_create_vertex_state, zink_vertex_state_destroy);
   screen->base.create_vertex_state = zink_cache_create_vertex_state;
   screen->base.vertex_state_destroy = zink_cache_vertex_state_destroy;

   zink_synchronization_init(screen);

   zink_init_screen_pipeline_libs(screen);

   if (!init_layouts(screen)) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("ZINK: failed to initialize layouts");
      goto fail;
   }

   if (!zink_descriptor_layouts_init(screen)) {
      if (!screen->driver_name_is_inferred)
         mesa_loge("ZINK: failed to initialize descriptor layouts");
      goto fail;
   }

   simple_mtx_init(&screen->copy_context_lock, mtx_plain);

   init_optimal_keys(screen);

   screen->screen_id = p_atomic_inc_return(&num_screens);
   zink_tracing = screen->instance_info->have_EXT_debug_utils &&
                  (u_trace_is_enabled(U_TRACE_TYPE_PERFETTO) || u_trace_is_enabled(U_TRACE_TYPE_MARKERS));

   screen->frame_marker_emitted = zink_screen_debug_marker_begin(screen, "frame");

   return screen;

fail:
   zink_destroy_screen(&screen->base);
   return NULL;
}

struct pipe_screen *
zink_create_screen(struct sw_winsys *winsys, const struct pipe_screen_config *config)
{
   struct zink_screen *ret = zink_internal_create_screen(config, -1, -1, 0);
   if (ret) {
      ret->drm_fd = -1;
   }

   return &ret->base;
}

static inline int
zink_render_rdev(int fd, int64_t *dev_major, int64_t *dev_minor)
{
   int ret = 0;
   *dev_major = *dev_minor = -1;
#ifdef HAVE_LIBDRM
   struct stat stx;
   drmDevicePtr dev;

   if (fd == -1)
      return 0;

   if (drmGetDevice2(fd, 0, &dev))
      return -1;

   if(!(dev->available_nodes & (1 << DRM_NODE_RENDER))) {
      ret = -1;
      goto free_device;
   }

   if(stat(dev->nodes[DRM_NODE_RENDER], &stx)) {
      ret = -1;
      goto free_device;
   }

   *dev_major = major(stx.st_rdev);
   *dev_minor = minor(stx.st_rdev);

free_device:
   drmFreeDevice(&dev);
#endif //HAVE_LIBDRM

   return ret;
}

struct pipe_screen *
zink_drm_create_screen(int fd, const struct pipe_screen_config *config)
{
   int64_t dev_major, dev_minor;
   struct zink_screen *ret;

   if (zink_render_rdev(fd, &dev_major, &dev_minor))
      return NULL;

   ret = zink_internal_create_screen(config, dev_major, dev_minor, 0);

   if (ret)
      ret->drm_fd = os_dupfd_cloexec(fd);
   if (ret && !ret->info.have_KHR_external_memory_fd) {
      debug_printf("ZINK: KHR_external_memory_fd required!\n");
      zink_destroy_screen(&ret->base);
      return NULL;
   }

   return &ret->base;
}

struct pipe_screen *
zink_win32_create_screen(uint64_t adapter_luid)
{
   struct zink_screen *ret = zink_internal_create_screen(NULL, -1, -1, adapter_luid);
   return ret ? &ret->base : NULL;
}

void VKAPI_PTR zink_stub_function_not_loaded()
{
   /* this will be used by the zink_verify_*_extensions() functions on a
    * release build
    */
   mesa_loge("ZINK: a Vulkan function was called without being loaded");
   abort();
}

bool
zink_screen_debug_marker_begin(struct zink_screen *screen, const char *fmt, ...)
{
   if (!zink_tracing)
      return false;

   char *name;
   va_list va;
   va_start(va, fmt);
   int ret = vasprintf(&name, fmt, va);
   va_end(va);

   if (ret == -1)
      return false;

   VkDebugUtilsLabelEXT info = { 0 };
   info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
   info.pLabelName = name;

   VKSCR(QueueBeginDebugUtilsLabelEXT)(screen->queue, &info);

   free(name);
   return true;
}

void
zink_screen_debug_marker_end(struct zink_screen *screen, bool emitted)
{
   if (emitted)
      VKSCR(QueueEndDebugUtilsLabelEXT)(screen->queue);
}
