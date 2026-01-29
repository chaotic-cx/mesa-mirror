/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include "pdsc_set_view_index.h"

#include "pdsc/pdsc_internal.h"
#include "common/pvr_iface.h"

#include <stdint.h>

/* PDSC set_view_index patch IDs. */
enum pvr_set_view_index_id {
   /** Patch ID for view index input. */
   PVR_SET_VIEW_INDEX_ID_IN,
};

pdsc_program *pdsc_set_view_index_gen(void *mem_ctx, enum pdsc_feature features)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, features, "set_view_index");

   ALLOC_CONST32(p, view_index_in, PVR_SET_VIEW_INDEX_ID_IN);
   DECL_PTEMP32(p, view_index_out, PVR_VDM_PTEMP_VIEW_INDEX_OUT);

   pdsc_mov32(p, view_index_out, view_index_in);
   pdsc_halt(p);

   return p;
}

void pdsc_patch_set_view_index(pdsc_program *p,
                               unsigned index,
                               uint32_t *const_buffer)
{
   pdsc_patch patch = { .patch_id = PVR_SET_VIEW_INDEX_ID_IN, .value = index };
   pdsc_patch_program(p, 1, &patch, const_buffer);
}
