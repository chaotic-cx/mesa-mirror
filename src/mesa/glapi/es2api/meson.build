# Copyright © 2017 Intel Corporation
# SPDX-License-Identifier: MIT

es2_apitrace_input = []
es2_apitrace_skip = []
es2_apitrace_link_whole = []
# Windows traces the aliases by the names that Mesa dispatches them by
if with_apitrace and not with_glvnd and not with_platform_windows
  es2_apitrace_aliases = custom_target(
    'es2_apitrace_aliases.txt',
    input : [apitrace_gen_py, 'gles2-symbols.txt'],
    output : 'es2_apitrace_aliases.txt',
    command : [
      prog_python, '@INPUT0@', apitrace_source_dir, 'aliases',
      '--mesa-xml', apitrace_mesa_xml, '--symbols', '@INPUT1@',
    ],
    depend_files : glapi_xml_py_deps,
    capture : true,
  )
  es2_apitrace_wrap_cpp = custom_target(
    'es2_apitrace_wrap.cpp',
    input : [apitrace_gen_py, 'gles2-symbols.txt'],
    output : 'es2_apitrace_wrap.cpp',
    command : [
      prog_python, '@INPUT0@', apitrace_source_dir, 'wrap', '--gl-aliases',
      '--mesa-xml', apitrace_mesa_xml, '--symbols', '@INPUT1@',
    ],
    depend_files : glapi_xml_py_deps,
    capture : true,
  )
  es2_apitrace_input += es2_apitrace_aliases
  es2_apitrace_skip += '--skip=@INPUT2@'
  es2_apitrace_link_whole += static_library(
    'gles2_apitrace',
    [es2_apitrace_wrap_cpp, apitrace_glproc_h],
    cpp_args : apitrace_cpp_args,
    gnu_symbol_visibility : 'hidden',
    include_directories : [inc_include, inc_src, inc_mesa],
    dependencies : dep_apitrace_headers,
    override_options : apitrace_override_options,
  )
endif

es2_glapi_mapi_tmp_h = custom_target(
  'es2_glapi_mapi_tmp.h',
  input : ['../new/gen_gldispatch_mapi.py', '../glapi/registry/gl.xml', es2_apitrace_input],
  output : 'es2_glapi_mapi_tmp.h',
  command : [prog_python, '@INPUT0@', 'glesv2', '@INPUT1@', es2_apitrace_skip],
  depend_files : glapi_xml_py_deps,
  capture : true,
)

_es2_c_args = []
if with_platform_windows
  _es2_c_args += ['-D_GDI32_']
endif

gles2_def = custom_target(
  'gles2.def',
  input: 'gles2.def.in',
  output : 'gles2.def',
  command : gen_vs_module_defs_normal_command,
)

libgles2 = shared_library(
  'GLESv2' + get_option('gles-lib-suffix'),
  ['libgles2_public.c', es2_glapi_mapi_tmp_h],
  vs_module_defs : gles2_def,
  c_args : [
    c_msvc_compat_args,
    _es2_c_args,
    gcc_lto_quirk,
  ],
  gnu_symbol_visibility : 'hidden',
  link_args : [ld_args_gc_sections, ld_args_apitrace],
  include_directories : [inc_mesa, inc_src, inc_include],
  link_with : shared_glapi_lib,
  link_whole : es2_apitrace_link_whole,
  dependencies : [dep_thread, dep_libdrm, dep_m, dep_dl, idep_mesautilc11],
  soversion : host_machine.system() == 'windows' ? '' : '2',
  version : '2.0.0',
  darwin_versions : '3.0.0',
  name_prefix : host_machine.system() == 'windows' ? 'lib' : [],  # always use lib, but avoid warnings on !windows
  install : true,
)

pkg.generate(
  name : 'glesv2',
  filebase : 'glesv2',
  description : 'Mesa OpenGL ES 2.0 library',
  version : meson.project_version(),
  libraries : libgles2,
  libraries_private : gl_priv_libs,
)

if with_symbols_check
  test(
    'es2-ABI-check',
    symbols_check,
    args : [
      '--lib', libgles2,
      '--symbols-file', files('gles2-symbols.txt'),
      symbols_check_args,
    ],
    suite : ['glapi'],
  )
endif
