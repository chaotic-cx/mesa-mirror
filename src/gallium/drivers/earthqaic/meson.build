#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#

### Determine build and install directories
driver_install_dir = get_option('prefix') / get_option('datadir') / 'earthqaic'

conf = configuration_data()
conf.set_quoted('DRIVER_BUILD_DIR', meson.current_build_dir())
conf.set_quoted('DRIVER_INSTALL_DIR', driver_install_dir)

driver_dirs_rs = configure_file(
  input: 'driver_dirs.rs.in',
  output: 'driver_dirs.rs',
  configuration: conf,
)

libdriver_dirs = static_library(
  'earthqaic_driver_dirs',
  driver_dirs_rs,
  rust_abi : 'rust',
  gnu_symbol_visibility : 'hidden',
)

### Bindgen targets
_compiler_bindgen_args = [
  '--allowlist-type', 'pipe.*',
  '--allowlist-type', 'renderonly',
  '--allowlist-function', 'u_default.*',
  '--allowlist-function', 'util.*',
  '--allowlist-function', 'u_init_pipe_screen_caps',
  '--with-derive-default',
  '--wrap-static-fns',
  '--impl-debug',
]

_bindings_rs = rust.bindgen(
  input : ['bindings.h'],
  output : 'bindings.rs',
  dependencies : [
    idep_mesautil,
    idep_nir_headers,
    dep_libdrm,
  ],
  include_directories : [inc_gallium_aux, inc_gallium, inc_include, inc_src],
  args : _compiler_bindgen_args,
)

# To suppress warnings from bindgen
bindgen_args = [
  '-Anon_snake_case',
  '-Anon_upper_case_globals',
  '-Anon_camel_case_types',
  '-Aunnecessary_transmutes',
]

_libearthqaic_rs_bindings = static_library(
  'earthqaic_rs_bindings',
  _bindings_rs,
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : bindgen_args,
)

libearthqaic = static_library(
    'earthqaic',
    files('lib.rs'),
    gnu_symbol_visibility : 'hidden',
    rust_abi : 'c',
    dependencies : [],
    link_with : [_libearthqaic_rs_bindings, libdriver_dirs],
)

driver_earthqaic = declare_dependency(
  compile_args : '-DGALLIUM_EARTHQAIC',
  link_with : [libearthqaicwinsys, libearthqaic],
)
