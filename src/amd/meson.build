# Copyright © 2017 Intel Corporation
# SPDX-License-Identifier: MIT

inc_amd = include_directories('.')

if with_tools.contains('drm-shim')
  subdir('drm-shim')
endif

with_amd_rdp = with_amd_vk and amd_options.contains('rdp') and not with_platform_windows
if with_amd_rdp
  # DevDriver Meson subproject for RDP integration - it has to come before 'common'
  dep_devdriver = dependency(
    'devdriver',
    fallback : ['devdriver', 'dep_devdriver'],
    default_options : ['werror=false'],
  )
endif

if with_amd_vk or with_gallium_radeonsi
  subdir('addrlib')
  subdir('common')
  if amd_with_llvm
    subdir('llvm')
  else
    libamd_common_llvm = []
  endif
  subdir('compiler')
endif

if with_amd_vk
  subdir('vulkan')
  if with_aco_tests
    subdir('compiler/tests')
  endif
endif

if with_gallium_radeonsi
  subdir('vpelib')
  subdir('gmlib')
  subdir('lanczoslib')
endif
