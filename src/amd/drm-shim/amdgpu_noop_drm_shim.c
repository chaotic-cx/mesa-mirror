/*
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "common/amdgpu_devices.h"
#include "drm-shim/amdgpu_noop_drm_shim.h"
#include "drm-shim/drm_shim.h"
#include "drm-uapi/amdgpu_drm.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/os_misc.h"
#include "util/u_debug.h"

static const struct amdgpu_device *amdgpu_dev;

static int
amdgpu_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
amdgpu_ioctl_gem_create(int fd, unsigned long request, void *_arg)
{
   union drm_amdgpu_gem_create *arg = _arg;
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct shim_bo *bo = calloc(1, sizeof(*bo));
   int ret;

   ret = drm_shim_bo_init(bo, arg->in.bo_size);
   if (ret) {
      free(bo);
      return ret;
   }

   arg->out.handle = drm_shim_bo_get_handle(shim_fd, bo);

   drm_shim_bo_put(bo);

   return 0;
}

static int
amdgpu_ioctl_gem_mmap(int fd, unsigned long request, void *_arg)
{
   union drm_amdgpu_gem_mmap *arg = _arg;
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, arg->in.handle);

   arg->out.addr_ptr = drm_shim_bo_get_mmap_offset(shim_fd, bo);

   drm_shim_bo_put(bo);

   return 0;
}

static void
amdgpu_info_hw_ip_info(uint32_t type, void *out, uint32_t return_size)
{
   const struct drm_amdgpu_info_hw_ip *ip_info;

   switch (type) {
   case AMDGPU_HW_IP_GFX:
      ip_info = &amdgpu_dev->hw_ip_gfx;
      break;
   case AMDGPU_HW_IP_COMPUTE:
      ip_info = &amdgpu_dev->hw_ip_compute;
      break;
   default:
      return;
   }

   memcpy(out, ip_info, MIN2(return_size, sizeof(*ip_info)));
}

static void
amdgpu_info_fw_version(uint32_t type, void *out, uint32_t return_size)
{
   const struct drm_amdgpu_info_firmware *firmware;

   switch (type) {
   case AMDGPU_INFO_FW_GFX_ME:
      firmware = &amdgpu_dev->fw_gfx_me;
      break;
   case AMDGPU_INFO_FW_GFX_PFP:
      firmware = &amdgpu_dev->fw_gfx_pfp;
      break;
   case AMDGPU_INFO_FW_GFX_MEC:
      firmware = &amdgpu_dev->fw_gfx_mec;
      break;
   default:
      return;
   }

   memcpy(out, firmware, MIN2(return_size, sizeof(*firmware)));
}

static void
amdgpu_info_read_mmr_reg(uint32_t reg, uint32_t count, uint32_t instance,
                        void *out, uint32_t return_size)
{
   uint8_t *dst = out;

   for (uint32_t i = 0; i < count && return_size; i++) {
      /* linear search */
      bool found = false;
      uint32_t val = 0;
      for (int j = 0; j < amdgpu_dev->mmr_reg_count; j++) {
         const uint32_t *triple = &amdgpu_dev->mmr_regs[j * 3];
         if (triple[0] == reg + i && triple[1] == instance) {
            val = triple[2];
            found = true;
            break;
         }
      }

      if (!found)
         mesa_logw("reg 0x%04x is unknown", reg + i);

      uint32_t copy_size = MIN2(return_size, sizeof(val));
      memcpy(dst, &val, copy_size);
      dst += copy_size;
      return_size -= copy_size;
   }
}

static void
amdgpu_info_dev_info(void *out, uint32_t return_size)
{
   memcpy(out, &amdgpu_dev->dev, MIN2(return_size, sizeof(amdgpu_dev->dev)));
}

static void
amdgpu_info_memory(void *out, uint32_t return_size)
{
   struct drm_amdgpu_memory_info memory = amdgpu_dev->mem;

   /* override all but total_heap_size */
   memory.vram.usable_heap_size = memory.vram.total_heap_size;
   memory.vram.heap_usage = 0;
   memory.vram.max_allocation = memory.vram.total_heap_size * 3 / 4;
   memory.cpu_accessible_vram.usable_heap_size = memory.cpu_accessible_vram.total_heap_size;
   memory.cpu_accessible_vram.heap_usage = 0;
   memory.cpu_accessible_vram.max_allocation = memory.cpu_accessible_vram.total_heap_size * 3 / 4;
   memory.gtt.usable_heap_size = memory.gtt.total_heap_size;
   memory.gtt.heap_usage = 0;
   memory.gtt.max_allocation = memory.gtt.total_heap_size * 3 / 4;

   memcpy(out, &memory, MIN2(return_size, sizeof(memory)));
}

static void
amdgpu_info_video_caps(uint32_t type, struct drm_amdgpu_info_video_caps *out)
{
   /* nop */
}

static int
amdgpu_ioctl_info(int fd, unsigned long request, void *arg)
{
   const struct drm_amdgpu_info *info = arg;
   void *out = (void *)info->return_pointer;
   const uint32_t one = 1;

   switch (info->query) {
   case AMDGPU_INFO_ACCEL_WORKING:
      memcpy(out, &one, MIN2(info->return_size, sizeof(one)));
      break;
   case AMDGPU_INFO_HW_IP_INFO:
      amdgpu_info_hw_ip_info(info->query_hw_ip.type, out, info->return_size);
      break;
   case AMDGPU_INFO_FW_VERSION:
      amdgpu_info_fw_version(info->query_fw.fw_type, out, info->return_size);
      break;
   case AMDGPU_INFO_READ_MMR_REG:
      amdgpu_info_read_mmr_reg(info->read_mmr_reg.dword_offset, info->read_mmr_reg.count,
                              info->read_mmr_reg.instance, out, info->return_size);
      break;
   case AMDGPU_INFO_DEV_INFO:
      amdgpu_info_dev_info(out, info->return_size);
      break;
   case AMDGPU_INFO_MEMORY:
      amdgpu_info_memory(out, info->return_size);
      break;
   case AMDGPU_INFO_VIDEO_CAPS:
      amdgpu_info_video_caps(info->video_cap.type, out);
      break;
   case AMDGPU_INFO_HW_IP_COUNT:
      memcpy(out, &one, MIN2(info->return_size, sizeof(one)));
      break;
   default:
      return -EINVAL;
   }

   return 0;
}

static ioctl_fn_t amdgpu_ioctls[] = {
   [DRM_AMDGPU_GEM_CREATE] = amdgpu_ioctl_gem_create,
   [DRM_AMDGPU_GEM_MMAP] = amdgpu_ioctl_gem_mmap,
   [DRM_AMDGPU_CTX] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_BO_LIST] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_CS] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_INFO] = amdgpu_ioctl_info,
   [DRM_AMDGPU_GEM_METADATA] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_GEM_WAIT_IDLE] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_GEM_VA] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_WAIT_CS] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_GEM_OP] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_GEM_USERPTR] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_WAIT_FENCES] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_VM] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_FENCE_TO_HANDLE] = amdgpu_ioctl_noop,
   [DRM_AMDGPU_SCHED] = amdgpu_ioctl_noop,
};

void
drm_shim_amdgpu_select_device(const char *gpu_id)
{
   if (gpu_id) {
      for (uint32_t i = 0; i < num_amdgpu_devices; i++) {
         const struct amdgpu_device *dev = &amdgpu_devices[i];
         if (!strcasecmp(dev->name, gpu_id)) {
            amdgpu_dev = &amdgpu_devices[i];
            break;
         }
      }
   } else {
      amdgpu_dev = &amdgpu_devices[0];
   }

   if (!amdgpu_dev) {
      mesa_loge("Failed to find amdgpu GPU named \"%s\"\n", gpu_id);
      abort();
   }
}

void
drm_shim_driver_init(void)
{
   const char *gpu_id = debug_get_option("AMDGPU_GPU_ID", "renoir");

   drm_shim_amdgpu_select_device(gpu_id);

   shim_device.driver_ioctls = amdgpu_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(amdgpu_ioctls);

   shim_device.version_major = AC_AMDGPU_DRM_MAJOR;
   shim_device.version_minor = AC_AMDGPU_DRM_MINOR;
   shim_device.version_patchlevel = 0;

   /* make drmGetDevices2 and drmProcessPciDevice happy */
   drm_shim_pci_device_setup(0x1002, 0x15E7, "0000:04:00.0", "amdgpu");
}
