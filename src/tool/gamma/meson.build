# Copyright © 2026 Valve Corporation
#
# SPDX-License-Identifier: MIT

libgamma_files = files(
  'gamma_device.cpp',
  'gamma_util.cpp',
)

subdir('shaders')

dep_sdl3 = dependency('sdl3')

dep_vulkan = dependency('vulkan')

gamma_includes = [
  inc_include,
  inc_src,
  inc_util,
  inc_radv_bvh,
  inc_vulkan_bvh,
  inc_radv_tools,
]

gamma_lib = static_library(
  'gamma',
  [libgamma_files],
  cpp_args: [cpp_msvc_compat_args],
  link_with: [],
  include_directories: gamma_includes,
  dependencies: [idep_mesautil, libimgui_sdl3_vulkan_dep, dep_sdl3, dep_vulkan],
)

executable(
  'gamma',
  [
    'main_app.cpp',
    'radv/gamma_file_view_radv_bvh4.cpp',
    'radv/gamma_file_view_radv_bvh8.cpp',
    'radv/gamma_file_view_radv.cpp',
    'gamma_app.cpp',
    'gamma_file_view.cpp',
    spv_shaders,
  ],
  cpp_args: cpp_msvc_compat_args,
  gnu_symbol_visibility: 'hidden',
  link_with: [gamma_lib],
  include_directories: gamma_includes,
  dependencies: [idep_mesautil, libimgui_sdl3_vulkan_dep, dep_sdl3, dep_vulkan],
)
