# Copyright © 2018 Rob Clark
# Copyright © 2019 Collabora
# SPDX-License-Identifier: MIT

libpanfrost_util_files = files(
  'pan_collect_varyings.c',
  'pan_ir.c',
  'pan_ir.h',
  'pan_lower_frag_coord_zw.c',
  'pan_lower_framebuffer.c',
  'pan_lower_helper_invocation.c',
  'pan_lower_image_index.c',
  'pan_lower_image_ms.c',
  'pan_lower_noperspective.c',
  'pan_lower_sample_position.c',
  'pan_lower_store_component.c',
  'pan_lower_vertex_id.c',
  'pan_lower_writeout.c',
  'pan_lower_xfb.c',
)

libpanfrost_util = static_library(
  'panfrost_util',
  [libpanfrost_util_files],
  include_directories : [inc_include, inc_src],
  dependencies: [idep_nir, idep_mesautil],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)
