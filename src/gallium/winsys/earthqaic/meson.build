# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

libearthqaicwinsys = static_library(
  'earthqaicwinsys',
  files('earthqaic_winsys.c'),
  include_directories : [
    inc_src, inc_include,
    inc_gallium, inc_gallium_aux, inc_gallium_drivers,
  ],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_mesautil, dep_libdrm],
)
