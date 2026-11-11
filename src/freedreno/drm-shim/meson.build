# Copyright © 2019 Google LLC
# SPDX-License-Identifier: MIT

# bin/drm-shim.py reads this to learn which parts the shim can stand in for
custom_target(
  'freedreno_shim_targets.py',
  input: '../common/freedreno_devices.py',
  output: 'freedreno_shim_targets.py',
  command: [
    prog_python, '@INPUT@', '--drm-shim-targets',
    '-p', join_paths(meson.project_build_root(), 'src/freedreno/registers/adreno/')
  ],
  depends: freedreno_py_header_files,
  depend_files: [files('../common/freedreno_dev_info.py')],
  capture: true,
  build_by_default: true,
)

libfreedreno_noop_drm_shim = shared_library(
  'freedreno_noop_drm_shim',
  'freedreno_noop.c',
  include_directories: [inc_include, inc_src, inc_freedreno],
  dependencies: dep_drm_shim,
  link_with: libfreedreno_common,
  gnu_symbol_visibility : 'hidden',
  install : true,
)
