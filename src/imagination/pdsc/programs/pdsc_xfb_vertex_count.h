/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include "common/pvr_iface.h"
#include "pdsc/pdsc.h"
#include "pdsc/pdsc_internal.h"

static inline pdsc_program *
pdsc_xfb_vertex_count_gen(void *mem_ctx, enum pdsc_feature features)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, features, "xfb_vertex_count");

   /* Patched with -counterOffset (64-bit two's complement); added, not
    * subtracted.
    */
   ALLOC_CONST64(p,
                 counter_offset,
                 PVR_XFB_VERTEX_COUNT_PATCH_ID_NEGATIVE_COUNTER_OFFSET);
   /* Reciprocal of vertex_stride patched as ceil(2^32 / stride) by the CPU.
    * Unused when stride == 1 (it would not fit in 32 bits).
    */
   ALLOC_CONST32(p,
                 vertex_stride_recip,
                 PVR_XFB_VERTEX_COUNT_PATCH_ID_VERTEX_STRIDE_RECIP);
   ALLOC_CONST32(p,
                 instance_count,
                 PVR_XFB_VERTEX_COUNT_PATCH_ID_INSTANCE_COUNT);
   ALLOC_CONST32(p,
                 first_instance,
                 PVR_XFB_VERTEX_COUNT_PATCH_ID_FIRST_INSTANCE);
   ALLOC_CONST64(p,
                 dword_select,
                 PVR_XFB_VERTEX_COUNT_PATCH_ID_COUNTER_DWORD_SELECT);
   ALLOC_CONST64(p, divide_enable, PVR_XFB_VERTEX_COUNT_PATCH_ID_DIVIDE_ENABLE);

   ALLOC_TEMP64(p, tzero);
   pdsc_limm(p, pdsc_ref_lo32(tzero), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(tzero), pdsc_ref_imm(0));

   ALLOC_TEMP64(p, sel);
   ALLOC_TEMP64(p, byte_count);
   ALLOC_TEMP64(p, working_temp);
   ALLOC_TEMP32_ARRAY(p, draw_params, 4);

   /* 16-byte aligned load of the 16 bytes containing the 32-bit counter. */
   ALLOC_TEMP64_ARRAY(p, counter_buffer_value, 2);
   const pdsc_ref cb_words[] = {
      pdsc_ref_lo32(pdsc_ref_elem64(counter_buffer_value, 0)),
      pdsc_ref_hi32(pdsc_ref_elem64(counter_buffer_value, 0)),
      pdsc_ref_lo32(pdsc_ref_elem64(counter_buffer_value, 1)),
      pdsc_ref_hi32(pdsc_ref_elem64(counter_buffer_value, 1)),
   };

   pdsc_ld(p,
           counter_buffer_value,
           PVR_XFB_VERTEX_COUNT_REF_ID_COUNTER_BUFFER_LD_DEST,
           PVR_XFB_VERTEX_COUNT_PATCH_ID_COUNTER_BUFFER_LD_SRC0,
           false);
   pdsc_wdf(p);

   /* Pick the counter dword. */
   pdsc_mov64(p, sel, dword_select);
   pdsc_limm(p, pdsc_ref_lo32(byte_count), pdsc_ref_imm(0));
   pdsc_limm(p, pdsc_ref_hi32(byte_count), pdsc_ref_imm(0));

   for (uint32_t i = 0; i < ARRAY_SIZE(cb_words); i++) {
      pdsc_cmp(p, sel, pdsc_ref_imm(i), .cop = PDSC_COP_EQ);
      pdsc_setc(p, PDSC_PRED_P0);
      pdsc_mov32(p, pdsc_ref_lo32(byte_count), cb_words[i], .cc = true);
   }

   /* byte_count += -counterOffset */
   pdsc_add64(p, byte_count, byte_count, counter_offset);

   pdsc_mov32(p, pdsc_ref_elem32(draw_params, 0), pdsc_ref_lo32(byte_count));

   /* Divide by vertex_stride using magic-number multiply:
    * floor(n / stride) = hi32(n * ceil(2^32 / stride)).
    * Skipped for stride == 1, where n is already the vertex count.
    */
   pdsc_cmp(p, tzero, divide_enable, .cop = PDSC_COP_NE);
   pdsc_setc(p, PDSC_PRED_P0);
   pdsc_mad(p,
            working_temp,
            pdsc_ref_elem32(draw_params, 0),
            vertex_stride_recip,
            tzero,
            .cc = true);
   pdsc_mov32(p,
              pdsc_ref_elem32(draw_params, 0),
              pdsc_ref_hi32(working_temp),
              .cc = true);

   pdsc_mov32(p, pdsc_ref_elem32(draw_params, 1), instance_count);
   pdsc_limm(p, pdsc_ref_elem32(draw_params, 2), pdsc_ref_imm(0));
   pdsc_mov32(p, pdsc_ref_elem32(draw_params, 3), first_instance);

   pdsc_st(p,
           pdsc_ref_elem32(draw_params, 0),
           PVR_XFB_VERTEX_COUNT_REF_ID_OUTPUT_COUNT_BUFFER_ST_SRC0,
           PVR_XFB_VERTEX_COUNT_PATCH_ID_OUTPUT_COUNT_BUFFER_ST_SRC0,
           false);
   pdsc_wdf(p);

   pdsc_halt(p);

   return p;
}
