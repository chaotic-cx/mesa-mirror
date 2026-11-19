# Copyright © 2022 Valve Corporation
# SPDX-License-Identifier: MIT

radv_annotate_layer = custom_target(
  'radv_annotate',
  input : [files('radv_annotate_layer_gen.py'), vk_api_xml],
  output : ['radv_annotate_layer.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', vulkan_gen_beta_arg,
  ],
  depend_files : [files('radv_annotate_layer_gen.py'), vk_dispatch_table_gen_depend_files],
)
