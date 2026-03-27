/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl_vk.h"

#include "poly/cl/restart.h"

/*
 * To implement drawIndirectCount generically, we dispatch a kernel to
 * clone-and-patch the indirect buffer, predicating out draws as appropriate.
 */
static uint
libkk_predicate_draw(global uint32_t *out, constant uint32_t *in,
                     uint32_t out_stride_el, uint32_t in_stride_el,
                     uint draw_id, bool enabled)
{
   out += draw_id * out_stride_el;
   in += draw_id * in_stride_el;

   /* Copy enabled draws, zero predicated draws. */
   for (uint i = 0; i < out_stride_el; ++i) {
      out[i] = enabled ? in[i] : 0;
   }
}

/*
 * Indirect Draw predicate: value > draw_id
 */
KERNEL(32)
libkk_predicate_indirect_gt_draw_id(global uint32_t *out,
                                    constant uint32_t *in,
                                    constant uint32_t *value,
                                    uint32_t out_stride_el,
                                    uint32_t in_stride_el)
{
   uint draw_id = cl_global_id.x;
   bool enabled = *value > draw_id;

   libkk_predicate_draw(out, in, out_stride_el, in_stride_el, draw_id, enabled);
}

/*
 * Indirect Draw predicate: value == 0
 */
KERNEL(32)
libkk_predicate_indirect_eq_zero(global uint32_t *out, constant uint32_t *in,
                                 constant uint32_t *value,
                                 uint32_t out_stride_el, uint32_t in_stride_el)
{
   uint draw_id = cl_global_id.x;
   bool enabled = *value == 0;

   libkk_predicate_draw(out, in, out_stride_el, in_stride_el, draw_id, enabled);
}

/*
 * Indirect Draw predicate: value != 0
 */
KERNEL(32)
libkk_predicate_indirect_neq_zero(global uint32_t *out, constant uint32_t *in,
                                  constant uint32_t *value,
                                  uint32_t out_stride_el,
                                  uint32_t in_stride_el)
{
   uint draw_id = cl_global_id.x;
   bool enabled = *value != 0;

   libkk_predicate_draw(out, in, out_stride_el, in_stride_el, draw_id, enabled);
}

KERNEL(1024)
libkk_unroll_geometry(
   uint64_t index_buffer, global struct poly_heap *heap,
   constant uint32_t *in_draw, global uint32_t *out_draw,
   uint32_t in_draw_stride_el, uint32_t restart_index,
   uint32_t index_buffer_size_el, uint32_t in_el_size_B,
   uint32_t out_el_size_B, uint32_t flatshade_first, uint32_t mode)
{
   uint gid = cl_group_id.x;
   in_draw += gid * in_draw_stride_el;
   out_draw += gid * 5;

   POLY_DECL_UNROLL_RESTART_SCRATCH(scratch, 1024);
   poly_unroll_geometry(out_draw, heap, in_draw, index_buffer,
                        index_buffer_size_el, in_el_size_B, out_el_size_B,
                        restart_index, flatshade_first, true, mode, scratch);
}

/* Unrolls restart without reordering vertices for a GS input. */
KERNEL(1024)
libkk_unroll_restart(uint64_t index_buffer, global struct poly_heap *heap,
                     constant uint32_t *in_draw, global uint32_t *out_draw,
                     uint32_t restart_index, uint32_t index_buffer_size_el,
                     uint32_t index_size_B, uint32_t flatshade_first,
                     uint32_t mode)
{
   POLY_DECL_UNROLL_RESTART_SCRATCH(scratch, 1024);
   poly_unroll_restart(out_draw, heap, in_draw, index_buffer,
                       index_buffer_size_el, index_size_B, restart_index,
                       flatshade_first, mode, scratch);
}

/* Rewrites adjacency primitives to the lines or triangles they rasterize for
 * draws without a GS. Restart must already be unrolled. */
KERNEL(1024)
libkk_unroll_adjacency(uint64_t index_buffer, global struct poly_heap *heap,
                       constant uint32_t *in_draw, global uint32_t *out_draw,
                       uint32_t in_draw_stride_el,
                       uint32_t index_buffer_size_el, uint32_t in_el_size_B,
                       uint32_t flatshade_first, uint32_t mode)
{
   uint tid = cl_local_id.x;
   in_draw += cl_group_id.x * in_draw_stride_el;
   out_draw += cl_group_id.x * 5;

   local uintptr_t out_ptr;
   if (tid == 0) {
      out_ptr = (uintptr_t)poly_setup_unroll_for_draw(
         heap, in_draw, out_draw, mode, in_el_size_B, sizeof(uint32_t));
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   uintptr_t in_ptr = (uintptr_t)(poly_index_buffer(
      index_buffer, index_buffer_size_el, in_draw[2], in_el_size_B,
      nir_load_ro_sink_address_poly()));
   uint in_range_el =
      poly_index_buffer_range_el(index_buffer_size_el, in_draw[2]);

   enum mesa_prim out_mode = u_reduced_prim(mode);
   uint out_per_prim = mesa_vertices_per_prim(out_mode);
   uint prims = u_decomposed_prims_for_vertices(mode, in_draw[0]);

   for (uint i = tid; i < prims; i += cl_local_size.x) {
      for (uint k = 0; k < out_per_prim; ++k) {
         /* Lines use vertices 1 and 2 of 4, triangles 0, 2 and 4 of 6 */
         uint vtx = out_mode == MESA_PRIM_LINES ? k + 1 : k * 2;
         uint id =
            poly_vertex_id_for_topology(mode, flatshade_first, i, vtx, prims);
         uint x = (i * out_per_prim) + poly_output_vertex_id_for_topology(
                                          out_mode, flatshade_first, true, k);
         poly_store_index(
            out_ptr, sizeof(uint32_t), x,
            poly_load_index(in_ptr, in_range_el, id, in_el_size_B));
      }
   }

   if (tid == 0)
      out_draw[0] = prims * out_per_prim;
}
