# Copyright © 2026 Collabora, Ltd.
# SPDX-License-Identifier: MIT

_util_rust_args = [
  rust_global_args,
  rust_2024_lint_args,
  '-Aclippy::identity_op',
  '-Aclippy::len_zero',
  '-Aclippy::manual_range_contains',
  # normally this is a good one, but we use it where the "better" code is worse
  '-Aclippy::needless_range_loop',
  '-Aclippy::redundant_field_names',
  '-Aclippy::upper_case_acronyms',
  '-Aclippy::vec_box',
  '-Aclippy::write_with_newline',
  # warns about public function might dereference a raw pointer, but nothing is
  # actually public here
  '-Aclippy::not_unsafe_ptr_arg_deref',
  '-Anon_snake_case',
]

_util_bindgen_args = [
    bindgen_output_args,
    '--allowlist-function', 'util_.*format_.*',
    '--rustified-enum', 'pipe_format',
    '--rustified-enum', 'util_format_layout',
    '--allowlist-var', 'PIPE_FORMAT_.*',
    '--no-prepend-enum-name',
    '--clang-macro-fallback',
    # The fallback's temp files default to fixed names in the cwd, which
    # races with other bindgen invocations in a parallel build.
    '--clang-macro-fallback-build-dir', '@PRIVATE_DIR@',
    '--wrap-static-fns-suffix', '_mesa_util_rs_extern',
]

_util_binding_types = [
  'pipe_format',
  'util_format_*',
]

foreach type : _util_binding_types
  _util_bindgen_args += ['--allowlist-type', type]
endforeach

_util_rust_bindgen = rust.bindgen(
  input : ['util_bindgen.h'],
  output : 'util_bindings.rs',
  output_inline_wrapper : 'util_bindings.c',
  include_directories : inc_util,
  dependencies : [idep_mesautil],
  args : _util_bindgen_args,
)

libutil_rust_bindgen_c = static_library(
  'util_rust_bindgen_c',
  _util_rust_bindgen[1],
  include_directories : inc_util,
  dependencies : [
    idep_mesautil.partial_dependency(compile_args : true, includes : true,
                                     sources : true),
  ],
  c_args : [
    c_msvc_compat_args,
    cc.get_supported_arguments('-Wno-missing-prototypes'),
  ],
  gnu_symbol_visibility : 'hidden',
)

# The generated bindings live in their own crate, so that the bulk of
# libutil_rust's code can be outside of a structured_sources() and get compiler
# errors that point at the original files (for a better IDE experience).
_libmesa_util_bindings_rs = static_library(
  'mesa_util_bindings',
  structured_sources([
    'bindings/lib.rs',
    _util_rust_bindgen,
  ]),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  link_with : [libutil_rust_bindgen_c],
)

_libmesa_util_rs = static_library(
  'mesa_util',
  files('lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : _util_rust_args,
  link_with : [_libmesa_util_bindings_rs],
)

dep_syn = dependency('syn',
  version : '>= 2.0.15',
  fallback : ['syn-2-rs', 'dep_syn'],
  required : true,
)

_libmesa_util_proc_rs = rust.proc_macro(
  'mesa_util_proc',
  'proc/lib.rs',
  dependencies : [dep_syn],
)

idep_mesa_util_rs = declare_dependency(
  link_with : [_libmesa_util_rs, _libmesa_util_proc_rs],
)

util_rs_bindgen_args = [
  '--raw-line', 'use mesa_util::format::pipe_format;'
]
foreach type : _util_binding_types
  util_rs_bindgen_args += ['--blocklist-type', type]
endforeach

if with_tests and get_option('b_sanitize') == 'none'
  rust.test(
    'util_rs',
    _libmesa_util_rs,
    suite : ['util'],
    dependencies : [
      idep_mesautil.partial_dependency(link_args : true, links : true),
    ],
    # This is needed to ensure we link against glibc
    # See also https://gitlab.freedesktop.org/mesa/mesa/-/issues/11632
    rust_args: ['-C', 'default-linker-libraries'],
  )
endif
