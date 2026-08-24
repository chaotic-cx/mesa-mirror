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

### Qshim
qshim_src_dir = meson.current_source_dir() / 'qshim'
qshim_out_dir = meson.current_build_dir()
platform_defs_dir = meson.current_source_dir() / 'qaic/c_inc'
jit_fbs_dir = meson.current_source_dir() / 'qaic'

### Bindgen targets
_compiler_bindgen_args = [
  '--allowlist-type', 'drm_gem_close',
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

_qaic_bindgen_args = [
  '--allowlist-file', '.*/HostApiInfo.h',
  '--allowlist-file', '.*/NspCmdProtocol.h',
  '--allowlist-file', '.*/hostNncActivateDeactivateMessages.h',
  '--allowlist-file', '.*/hostNncLoadUnloadMessages.h',
  '--allowlist-file', '.*/hostNncResourceGroupMessages.h',
  '--allowlist-file', '.*/hostNncResourceReservationMessages.h',
  '--allowlist-file', '.*/hostNncTransactionMessages.h',
  '--allowlist-file', '.*/nncErrors.h',
  '--allowlist-file', '.*/qaic_accel.h',
  '--allowlist-var', '_DRM_.*',
  '--with-derive-default',
]

_qaic_bindings_rs = rust.bindgen(
  input : ['qaic/qaic_bindings.h'],
  output : 'qaic_bindings.rs',
  dependencies : [
    idep_mesautil,
    dep_libdrm,
  ],
  include_directories : [inc_gallium_aux, inc_gallium, inc_include, inc_src],
  args : _qaic_bindgen_args,
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

_libqaic_rs_bindings = static_library(
  'qaic_rs_bindings',
  _qaic_bindings_rs,
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : bindgen_args,
)

### Rust dependencies
dep_rustlibc = dependency('libc',
  version : '>= 0.2.168',
  fallback : ['libc-0.2-rs', 'dep_libc'],
  required : true,
)

# The flatc that ships with Ubuntu 26 (v23.5.26) is too outdated.
prog_flatc = find_program('flatc', version : '>= 25.2.10', required : true)

if 'clang' in cc.get_id()
  prog_clang = cc.cmd_array()
else
  prog_clang = find_program('clang')
endif

if 'clang' in cpp.get_id()
  prog_clangxx = cpp.cmd_array()
else
  prog_clangxx = find_program('clang++')
endif

prog_ar = find_program('ar', required: true)

### Qshim device executable + artifacts (network.elf, constants.bin,
### constantsdesc.bin, md_struct.bin). Needs clang/clang++/flatc above.
subdir('qshim')

### Earthqaic driver
libearthqaic = static_library(
    'earthqaic',
    files('lib.rs'),
    gnu_symbol_visibility : 'hidden',
    rust_abi : 'c',
    dependencies : [dep_rustlibc],
    link_with : [_libearthqaic_rs_bindings, _libqaic_rs_bindings, libdriver_dirs],
)

driver_earthqaic = declare_dependency(
  compile_args : '-DGALLIUM_EARTHQAIC',
  link_with : [libearthqaicwinsys, libearthqaic],
)
