/*
 * Copyright © 2019 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "drm-shim/drm_shim.h"
#include "drm-uapi/msm_drm.h"
#include <sys/ioctl.h>

#include <string.h>
#include "freedreno_dev_info.h"
#include "util/os_misc.h"
#include "util/u_math.h"

static const struct fd_dev_rec *device_rec;
static uint64_t device_chip_id;

static uint32_t
rec_gpu_id(const struct fd_dev_rec *rec)
{
   if (rec->id.gpu_id)
      return rec->id.gpu_id;

   /* Variants such as "FD305B" name no gpu_id of their own. */
   static const char *const prefixes[] = { "FD", "Adreno (TM) " };
   for (unsigned i = 0; i < ARRAY_SIZE(prefixes); i++) {
      const char *p = rec->name;
      size_t len = strlen(prefixes[i]);

      if (strncmp(p, prefixes[i], len))
         continue;
      p += len;
      if (!*p)
         continue;

      const char *digits = p;
      while (*p >= '0' && *p <= '9')
         p++;
      if (*p == '\0' && p != digits)
         return atoi(digits);
   }
   return 0;
}

/* What a part reports where the table's id only has to recognise it. */
static const struct {
   uint32_t gpu_id;
   uint8_t shim_patch;
} chip_id_fixups[] = {
   { .gpu_id = 225, .shim_patch = 0x06 },
   { .gpu_id = 305, .shim_patch = 0x20 },
   { .gpu_id = 430, .shim_patch = 0x02 },
   { .gpu_id = 530, .shim_patch = 0x02 },
   { .gpu_id = 540, .shim_patch = 0x01 },
   { .gpu_id = 630, .shim_patch = 0x01 },
   { .gpu_id = 640, .shim_patch = 0x01 },
   { .gpu_id = 650, .shim_patch = 0x02 },
   { .gpu_id = 660, .shim_patch = 0x01 },
   { .gpu_id = 680, .shim_patch = 0x01 },
};

/* The table names a whole part with a 0xff wildcard patch level, which no
 * real GPU reports. Report the patch level of the actual part instead of
 * that fake id or a patch level of earlier revisions, requiring more
 * workarounds.
 */
static uint64_t
rec_chip_id(const struct fd_dev_rec *rec)
{
   uint64_t chip_id = rec->id.chip_id;
   uint8_t shim_patch = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(chip_id_fixups); i++) {
      if (chip_id_fixups[i].gpu_id != rec_gpu_id(rec))
         continue;
      shim_patch = chip_id_fixups[i].shim_patch;
   }

   if ((chip_id & 0xff) == 0xff)
      chip_id = (chip_id & ~0xffull) | shim_patch;

   return chip_id;
}

/* Prefer drm/msm entries (fuse bits set) over KGSL ones. */
static unsigned
rec_rank(const struct fd_dev_rec *rec)
{
   return !!(rec->id.chip_id >> 32);
}

static bool
rec_better(const struct fd_dev_rec *rec, const struct fd_dev_rec *best)
{
   if (rec_rank(rec) != rec_rank(best))
      return rec_rank(rec) > rec_rank(best);

   /* Distinct parts can share a marketing name; only compare patch levels of one part. */
   return (rec->id.chip_id >> 8) == (best->id.chip_id >> 8) &&
          rec->id.chip_id > best->id.chip_id;
}

static bool
rec_usable(const struct fd_dev_rec *rec)
{
   return rec->info->gmem_size != 0;
}

static int
msm_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
msm_ioctl_gem_new(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_msm_gem_new *create = arg;
   size_t size = (size_t)align64(create->size, 4096);

   if (!size)
      return -EINVAL;

   struct shim_bo *bo = calloc(1, sizeof(*bo));
   int ret;

   ret = drm_shim_bo_init(bo, size);
   if (ret) {
      free(bo);
      return ret;
   }

   create->handle = drm_shim_bo_get_handle(shim_fd, bo);

   drm_shim_bo_put(bo);

   return 0;
}

static int
msm_ioctl_gem_info(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_msm_gem_info *args = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, args->handle);

   if (!bo)
      return -ENOENT;

   switch (args->info) {
   case MSM_INFO_GET_OFFSET:
      args->value = drm_shim_bo_get_mmap_offset(shim_fd, bo);
      break;
   case MSM_INFO_GET_IOVA:
      args->value = bo->mem_addr;
      break;
   case MSM_INFO_SET_IOVA:
   case MSM_INFO_SET_NAME:
   case MSM_INFO_SET_METADATA:
   case MSM_INFO_GET_METADATA:
      break;
   default:
      fprintf(stderr, "Unknown DRM_IOCTL_MSM_GEM_INFO %d\n", args->info);
      drm_shim_bo_put(bo);
      return -1;
   }

   drm_shim_bo_put(bo);

   return 0;
}

static int
msm_ioctl_get_param(int fd, unsigned long request, void *arg)
{
   struct drm_msm_param *gp = arg;

   switch (gp->param) {
   case MSM_PARAM_GPU_ID:
      gp->value = rec_gpu_id(device_rec);
      return 0;
   case MSM_PARAM_GMEM_SIZE:
      gp->value = device_rec->info->gmem_size;
      return 0;
   case MSM_PARAM_GMEM_BASE:
      gp->value = 0x100000;
      return 0;
   case MSM_PARAM_CHIP_ID:
      gp->value = device_chip_id;
      return 0;
   case MSM_PARAM_PRIORITIES:
      gp->value = 3;
      return 0;
   case MSM_PARAM_MAX_FREQ:
      gp->value = 1000000;
      return 0;
   case MSM_PARAM_TIMESTAMP:
      gp->value = 0;
      return 0;
   case MSM_PARAM_PP_PGTABLE:
      gp->value = 1;
      return 0;
   case MSM_PARAM_FAULTS:
   case MSM_PARAM_SUSPENDS:
      gp->value = 0;
      return 0;
   case MSM_PARAM_VA_START:
      gp->value = 0x100000000ULL;
      return 0;
   case MSM_PARAM_VA_SIZE:
      gp->value = (1ULL << 48) - 0x100000000ULL;
      return 0;
   case MSM_PARAM_RAYTRACING:
      gp->value = 1;
      return 0;
   case MSM_PARAM_UCHE_TRAP_BASE:
      gp->value = 0x1fffffffff000ull;
      return 0;

   case MSM_PARAM_HAS_PRR:
      gp->value = 1;
      return 0;

      /* UBWC config values from some GPU, but we don't actually maintain the
       * device list mapping because it doesn't matter to shader compiles.
       */
   case MSM_PARAM_UBWC_SWIZZLE:
      gp->value = 0x6;
      return 0;
   case MSM_PARAM_HIGHEST_BANK_BIT:
      gp->value = 15;
      return 0;
   case MSM_PARAM_MACROTILE_MODE:
      gp->value = 0;
      return 0;

   default:
      fprintf(stderr, "Unknown DRM_IOCTL_MSM_GET_PARAM %d\n", gp->param);
      return -1;
   }
}

static int
msm_ioctl_set_param(int fd, unsigned long request, void *arg)
{
   struct drm_msm_param *sp = arg;

   switch (sp->param) {
   case MSM_PARAM_EN_VM_BIND:
      /* Shim doesn't have to do anything with this -- it's binding iovas to
       * BOs, but since we don't exec the iovas don't matter.
       */
      return 0;
   default:
      return 0;
   }
}

static int
msm_ioctl_gem_madvise(int fd, unsigned long request, void *arg)
{
   struct drm_msm_gem_madvise *args = arg;

   args->retained = true;

   return 0;
}

static int
msm_ioctl_vm_bind(int fd, unsigned long request, void *arg)
{
   struct drm_msm_vm_bind *vm_bind = arg;
   vm_bind->fence_fd = -1;
   return 0;
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_MSM_GET_PARAM] = msm_ioctl_get_param,
   [DRM_MSM_SET_PARAM] = msm_ioctl_set_param,
   [DRM_MSM_GEM_NEW] = msm_ioctl_gem_new,
   [DRM_MSM_GEM_INFO] = msm_ioctl_gem_info,
   [DRM_MSM_GEM_CPU_PREP] = msm_ioctl_noop,
   [DRM_MSM_GEM_CPU_FINI] = msm_ioctl_noop,
   [DRM_MSM_GEM_SUBMIT] = msm_ioctl_noop,
   [DRM_MSM_WAIT_FENCE] = msm_ioctl_noop,
   [DRM_MSM_GEM_MADVISE] = msm_ioctl_gem_madvise,
   [DRM_MSM_SUBMITQUEUE_NEW] = msm_ioctl_noop,
   [DRM_MSM_SUBMITQUEUE_CLOSE] = msm_ioctl_noop,
   [DRM_MSM_SUBMITQUEUE_QUERY] = msm_ioctl_noop,
   [DRM_MSM_VM_BIND] = msm_ioctl_vm_bind,
   [DRM_MSM_PERFCNTR_CONFIG] = msm_ioctl_noop,
};

static void
print_supported_chips_and_abort(const char *unrecognized_env)
{
   fprintf(stderr, "%s unrecognized, shim supports:\n", unrecognized_env);

   for (unsigned i = 0; ; i++) {
      const struct fd_dev_rec *rec = fd_dev_rec_at(i);
      if (!rec)
         break;
      if (!rec_usable(rec))
         continue;
      fprintf(stderr, "- gpu_id=%" PRIu32 ", chip_id=0x%" PRIx64 " (%s)\n",
              rec_gpu_id(rec), rec_chip_id(rec), rec->name);
   }

   abort();
}

static void
msm_driver_get_device_info(void)
{
   const char *chip_id_env = os_get_option("FD_CHIP_ID");

   if (chip_id_env) {
      char *endptr;
      uint64_t chip_id = strtoul(chip_id_env, &endptr, 16);

      if (endptr == chip_id_env || *endptr != '\0') {
         fprintf(stderr, "Invalid hex value for FD_CHIP_ID\n");
         abort();
      }

      /* A patch level nobody has enumerated should still find its part. */
      struct fd_dev_id id = { .gpu_id = 0, .chip_id = chip_id };
      const struct fd_dev_rec *rec = fd_dev_rec_find(&id);

      if (rec && rec_usable(rec)) {
         device_rec = rec;
         device_chip_id = chip_id;
         return;
      }

      print_supported_chips_and_abort("FD_CHIP_ID");
   }

   const char *env = os_get_option("FD_GPU_ID");

   int gpu_id = env ? atoi(env) : 630;

   const struct fd_dev_rec *best = NULL;

   for (unsigned i = 0; ; i++) {
      const struct fd_dev_rec *rec = fd_dev_rec_at(i);
      if (!rec)
         break;
      if (!rec_usable(rec) || rec_gpu_id(rec) != gpu_id)
         continue;
      if (!best || rec_better(rec, best))
         best = rec;
   }

   if (best) {
      device_rec = best;
      device_chip_id = rec_chip_id(best);
      return;
   }

   if (!env) {
      fprintf(stderr, "no device info for the default gpu_id %d\n", gpu_id);
      abort();
   }

   print_supported_chips_and_abort("FD_GPU_ID");
}

void
drm_shim_driver_init(void)
{
   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);

   /* msm uses the DRM version to expose features, instead of getparam. */
   shim_device.version_major = 1;
   shim_device.version_minor = 9;
   shim_device.version_patchlevel = 0;

   msm_driver_get_device_info();

   drm_shim_platform_device_setup("msm", "/rdb/msm", "qcom,adreno");
}
