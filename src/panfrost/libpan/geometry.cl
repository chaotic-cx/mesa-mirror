/*
 * Copyright 2026 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "poly/cl/restart.h"
#include "poly/geometry.h"

#if PAN_ARCH >= 10
KERNEL(1)
panlib_gs_setup_indirect(global struct poly_vertex_params *vp /* output */,
                         global struct poly_geometry_params *gp /* output */,
                         global struct poly_heap *heap,
                         uint64_t index_buffer,
                         uint32_t index_size_B /* 0 if no index bffer */,
                         uint32_t index_buffer_range_el,
                         constant uint *draw,
                         uint64_t vs_outputs /* Vertex (TES) output mask */,
                         uint prim /* Input primitive type, mesa_prim */,
                         uint is_prefix_summing,
                         uint max_indices,
                         uint shape /* poly_gs_shape */)
{
   poly_gs_setup_indirect(index_buffer, draw, vp, gp, heap, vs_outputs,
                          index_size_B, index_buffer_range_el, prim,
                          is_prefix_summing, max_indices, shape);
}
#endif
