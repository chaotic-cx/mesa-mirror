/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PDSC_SET_VIEW_INDEX_H
#define PDSC_SET_VIEW_INDEX_H

#include <stdint.h>

typedef struct _pdsc_program pdsc_program;
enum pdsc_feature;

pdsc_program *pdsc_set_view_index_gen(void *mem_ctx,
                                      enum pdsc_feature features);
void pdsc_patch_set_view_index(pdsc_program *p,
                               unsigned index,
                               uint32_t *const_buffer);

#endif /* PDSC_SET_VIEW_INDEX_H */
