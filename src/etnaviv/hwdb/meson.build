# Copyright © 2024 Igalia S.L.
# SPDX-License-Identifier: MIT

prog_hwdb = files('hwdb.h.py')
hwdb_h = custom_target(
  'hwdb.h',
  input: [
    'amlogic/gc_feature_database.h',
    'd-robotics/gc_feature_database.h',
    'eys3d/gc_feature_database.h',
    'nxp/gc_feature_database.h',
    'st/gc_feature_database.h'
  ],
  output: 'hwdb.h',
  command: [prog_python, prog_hwdb, '--output', '@OUTPUT@', '@INPUT@'],
)

libetnaviv_hwdb_files = [
  hwdb_h,
  'etna_hwdb.c',
]

libetnaviv_hwdb = static_library(
  'etnaviv_hwdb',
  libetnaviv_hwdb_files,
  dependencies: idep_etna_common,
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)
