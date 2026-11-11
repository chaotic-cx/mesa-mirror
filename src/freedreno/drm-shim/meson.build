# Copyright © 2019 Google LLC
# SPDX-License-Identifier: MIT

libfreedreno_noop_drm_shim = shared_library(
  'freedreno_noop_drm_shim',
  'freedreno_noop.c',
  include_directories: [inc_include, inc_src, inc_freedreno],
  dependencies: dep_drm_shim,
  link_with: libfreedreno_common,
  gnu_symbol_visibility : 'hidden',
  install : true,
)
