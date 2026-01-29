/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PDSC_COMPUTE_H
#define PDSC_COMPUTE_H

#include <stdbool.h>
#include <stdint.h>

typedef struct _pdsc_program pdsc_program;
enum pdsc_feature;

pdsc_program *pdsc_compute_shader_gen(void *mem_ctx,
                                      enum pdsc_feature features);

struct pdsc_compute_arguments {
   uint32_t global_base_group[3];

   uint64_t zero_addr;
   uint32_t zero_start;
   uint32_t zero_count;

   uint64_t num_wgs_addr;
   uint32_t num_wgs_start;
   uint32_t num_wgs_count;

   uint32_t wgid_start;
   uint32_t wgid_count;

   uint32_t ccs_lid_start;
   uint32_t ccs_lid_count;

   uint32_t coeff_update_shader_temps;
   uint64_t coeff_update_shader_addr;

   uint32_t cs_lid_start;
   uint32_t cs_lid_count;

   uint64_t cond_render_addr;
   bool cond_render_invert;

   uint32_t shader_temps;
   uint64_t shader_addr;
};

void pdsc_patch_compute_shader(pdsc_program *p,
                               struct pdsc_compute_arguments *args,
                               uint32_t *const_buffer);

#endif /* PDSC_COMPUTE_H */
