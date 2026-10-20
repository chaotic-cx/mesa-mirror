# Copyright © 2018 Intel Corporation
# SPDX-License-Identifier: MIT

inc_wgl = include_directories('.')

_wgl_deps = [idep_xmlconfig, driver_zink, idep_mesautilformat]
if with_gallium_zink
  _wgl_deps += idep_vulkan_wsi_defines
endif

files_libwgl = files(
  'stw_context.c',
  'stw_device.c',
  'stw_ext_context.c',
  'stw_ext_extensionsstring.c',
  'stw_ext_interop.c',
  'stw_ext_pbuffer.c',
  'stw_ext_pixelformat.c',
  'stw_ext_rendertexture.c',
  'stw_ext_swapinterval.c',
  'stw_framebuffer.c',
  'stw_getprocaddress.c',
  'stw_image.c',
  'stw_nopfuncs.c',
  'stw_nopfuncs.h',
  'stw_pixelformat.c',
  'stw_st.c',
  'stw_tls.c',
)

if target_machine.system().startswith('Gaming.Xbox')
  files_libwgl += files('stw_gdishim.c')
endif

libwgl_apitrace = []
if with_apitrace
  wgl_apitrace_wrap_cpp = custom_target(
    'wgl_apitrace_wrap.cpp',
    input : apitrace_gen_py,
    output : 'wgl_apitrace_wrap.cpp',
    command : [
      prog_python, '@INPUT@', apitrace_source_dir, 'wrap', '--wgl',
      '--real-prefix', 'stw_', '--wrap-prefix', 'stw_apitrace_',
      '--include', 'stw_apitrace.h',
    ],
    capture : true,
  )
  libwgl_apitrace = static_library(
    'wgl_apitrace',
    [wgl_apitrace_wrap_cpp, apitrace_glproc_h],
    cpp_args : apitrace_cpp_args,
    include_directories : [inc_include, inc_src, inc_mesa, inc_wgl],
    dependencies : dep_apitrace_headers,
    override_options : apitrace_override_options,
  )
endif

libwgl = static_library(
  'wgl',
  files_libwgl,
  c_args : [
    '-D_GDI32_',             # prevent wgl* being declared __declspec(dllimport)
  ],
  include_directories : [
    inc_include, inc_src, inc_gallium, inc_gallium_aux, inc_mesa,
  ],
  link_whole : libwgl_apitrace,
  dependencies : _wgl_deps,
)
