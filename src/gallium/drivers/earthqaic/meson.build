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
  '--allowlist-type', 'nir.*',
  '--allowlist-type', 'pipe.*',
  '--allowlist-type', 'renderonly',
  '--allowlist-function', 'drmIoctl',
  '--allowlist-function', 'nir_block_cf_tree_next',
  '--allowlist-function', 'nir_index_blocks',
  '--allowlist-function', 'nir_index_ssa_defs',
  '--allowlist-function', 'nir_print_shader_stdout',
  '--allowlist-function', 'nir_shader_get_entrypoint',
  '--allowlist-function', 'pipe_buffer_write',
  '--allowlist-function', 'pipe_reference_init',
  '--allowlist-function', 'pipe_resource_reference',
  '--allowlist-function', 'u_default_buffer_subdata',
  '--allowlist-function', 'u_default_clear_buffer',
  '--allowlist-function', 'u_init_pipe_screen_caps',
  '--allowlist-function', 'util_resource_copy_region',
  '--allowlist-var', 'nir.*',
  '--with-derive-default',
  '--impl-debug',
  '--wrap-static-fns-suffix', '_earthqaic_extern',
]

_bindings_rs = rust.bindgen(
  input : ['bindings.h'],
  output : 'bindings.rs',
  output_inline_wrapper : 'inline_wrapper.c',
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

_libearthqaic_binding_wrappers = static_library(
  'earthqaic_binding_wrappers',
  ['bindings.h', _bindings_rs[1]],
  gnu_symbol_visibility : 'hidden',
  dependencies: [
    idep_mesautilformat,
    idep_nir_headers,
    dep_libdrm,
  ],
  include_directories : [inc_gallium_aux, inc_gallium, inc_include, inc_src],
  c_args: ['-Wno-missing-prototypes'],
)

_libearthqaic_rs_bindings = static_library(
  'earthqaic_rs_bindings',
  _bindings_rs[0],
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : bindgen_args,
  link_with : [_libearthqaic_binding_wrappers],
)

_libqaic_rs_bindings = static_library(
  'qaic_rs_bindings',
  _qaic_bindings_rs,
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : bindgen_args,
)

### Rust dependencies
dep_elf_rs = dependency('elf',
  version : '>= 0.8.0',
  fallback : ['elf-0.8-rs', 'dep_elf_rs'],
  required: true,
)

dep_flatbuffers_rust = dependency('flatbuffers_rust',
  version : '>= 25.2.10',
  fallback : ['flatbuffers-25-rs', 'dep_flatbuffers_rust'],
  required : true,
)

dep_llvm_sys = dependency('llvm_sys',
  version : '>= 201.0.1',
  fallback : ['llvm-sys-201-rs', 'dep_llvm_sys'],
  required : true,
)

dep_rustlibc = dependency('libc',
  version : '>= 0.2.168',
  fallback : ['libc-0.2-rs', 'dep_libc'],
  required : true,
)

# The flatc that ships with Ubuntu 26 (v23.5.26) is too outdated.
prog_flatc = find_program('flatc', version : '>= 25.2.10', required : true)

### JIT flatbuffers schema
_jit_generated_rs = custom_target(
  'jit_generated_rs',
  input : 'qaic/jit.fbs',
  output : 'jit_generated.rs',
  command : [
    prog_flatc,
    '--rust',
    '--no-warnings',
    '-o', '@OUTDIR@',
    '@INPUT@',
  ],
)

_libjit_generated = static_library(
  'jit_generated',
  _jit_generated_rs,
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : [
    '-Anon_snake_case',
    '-Amismatched_lifetime_syntaxes',
    '-Aunused_imports',
  ],
  dependencies : [dep_flatbuffers_rust],
  rust_dependency_map : {'flatbuffers_rust' : 'flatbuffers'},
)

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

### Hexagon library
hexagon_args = [
    '--target=hexagon-unknown-elf',
    '-ffreestanding',
    '-fno-builtin',
    '-nostdlib',
    '-mv68',
    '-mhvx=v68',
    '-mllvm', '--disable-hexagon-amodeopt',
    '-mllvm', '-hexagon-commgep=false',
    '-mhvx-ieee-fp',
    '-fPIC',
]

hexagon_dfdiv_o = custom_target(
  'hexagon_dfdiv_o',
  input: [
    'hexagon/dfdiv.S',
  ],
  output: 'dfdiv.o',
  command: [
    prog_clang,
    hexagon_args,
    '-c', '@INPUT@',
    '-o', '@OUTPUT@',
  ],
)

hexagon_libm_o = custom_target(
  'hexagon_libm_o',
  input: [
    'hexagon/libm.c',
  ],
  output: 'libm.o',
  command: [
    prog_clang,
    hexagon_args,
    '-c', '@INPUT@',
    '-o', '@OUTPUT@',
  ],
)

libhexagon_a = custom_target(
  'libhexagon_a',
  input: [hexagon_dfdiv_o, hexagon_libm_o],
  output: 'libhexagon.a',
  command: [prog_ar, 'rcs', '@OUTPUT@', '@INPUT@'],
  build_by_default: true,
  install: true,
  install_dir: driver_install_dir,
)

### Teflon operations
tensor_ops_o = custom_target(
  'tensor_ops_o',
  input: 'teflon_ops/tensor_ops.cpp',
  output: 'tensor_ops.o',
  command: [
    prog_clangxx,
    hexagon_args,
    '-I' + meson.current_source_dir() / 'qaic/c_inc/jit',
    '-I' + meson.current_source_dir() / 'teflon_ops',
    '-c', '@INPUT@',
    '-o', '@OUTPUT@',
  ],
  build_by_default: true,
  install: true,
  install_dir: driver_install_dir,
)

### Earthqaic driver
libearthqaic = static_library(
    'earthqaic',
    files('lib.rs'),
    gnu_symbol_visibility : 'hidden',
    rust_abi : 'c',
    dependencies : [dep_elf_rs, dep_flatbuffers_rust, dep_llvm_sys, dep_rustlibc],
    link_with : [_libearthqaic_rs_bindings, _libqaic_rs_bindings, _libjit_generated, libdriver_dirs],
)

driver_earthqaic = declare_dependency(
  compile_args : '-DGALLIUM_EARTHQAIC',
  link_with : [libearthqaicwinsys, libearthqaic],
)
