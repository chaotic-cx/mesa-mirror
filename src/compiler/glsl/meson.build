# Copyright © 2017 Intel Corporation
# SPDX-License-Identifier: MIT

libglsl_util = static_library(
  'glsl_util',
  glsl_util_files,
  include_directories : [inc_include, inc_src, inc_mapi, inc_mesa, inc_gallium, inc_gallium_aux],
  dependencies : [dep_valgrind, idep_mesautil],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

subdir('glcpp')

bison_command = []
if yacc_is_bison
  bison_command = [
    prog_bison, '-o', '@OUTPUT0@', '-p', '_mesa_glsl_', '--defines=@OUTPUT1@',
    '@INPUT@',
  ]
else
  bison_command = [
    prog_bison, '-o', '@OUTPUT0@', '-p', '_mesa_glsl_', '-H', '@OUTPUT1@',
    '@INPUT@',
  ]
endif

glsl_parser = custom_target(
  'glsl_parser',
  input : 'glsl_parser.yy',
  output : ['glsl_parser.cpp', 'glsl_parser.h'],
  command : bison_command
)

glsl_lexer_cpp = custom_target(
  'glsl_lexer_cpp',
  input : 'glsl_lexer.ll',
  output : 'glsl_lexer.cpp',
  command : [prog_flex_cpp, '-o', '@OUTPUT@', '@INPUT@'],
)

ir_expression_operation_constant_h = custom_target(
  'ir_expression_operation_constant.h',
  input : 'ir_expression_operation.py',
  output : 'ir_expression_operation_constant.h',
  command : [prog_python, '@INPUT@', 'constant'],
  capture : true,
)

ir_expression_operation_strings_h = custom_target(
  'ir_expression_operation_strings.h',
  input : 'ir_expression_operation.py',
  output : 'ir_expression_operation_strings.h',
  command : [prog_python, '@INPUT@', 'strings'],
  capture : true,
)

float64_glsl_h = custom_target(
  'float64_glsl.h',
  input : [files_xxd, 'float64.glsl'],
  output : 'float64_glsl.h',
  command : [prog_python, '@INPUT@', '@OUTPUT@', '-n', 'float64_source'],
)

cross_platform_settings_piece_all_h = custom_target(
  'cross_platform_settings_piece_all.h',
  input : [files_xxd, 'CrossPlatformSettings_piece_all.glsl'],
  output : 'cross_platform_settings_piece_all.h',
  command : [prog_python, '@INPUT@', '@OUTPUT@', '-n', 'cross_platform_settings_piece_all_header'],
)

bc1_glsl_h = custom_target(
  'bc1_glsl.h',
  input : [files_xxd, 'bc1.glsl'],
  output : 'bc1_glsl.h',
  command : [prog_python, '@INPUT@', '@OUTPUT@', '-n', 'bc1_source'],
)

bc4_glsl_h = custom_target(
  'bc4_glsl.h',
  input : [files_xxd, 'bc4.glsl'],
  output : 'bc4_glsl.h',
  command : [prog_python, '@INPUT@', '@OUTPUT@', '-n', 'bc4_source'],
)

etc2_rgba_stitch_glsl_h = custom_target(
  'etc2_rgba_stitch_glsl.h',
  input : [files_xxd, 'etc2_rgba_stitch.glsl'],
  output : 'etc2_rgba_stitch_glsl.h',
  command : [prog_python, '@INPUT@', '@OUTPUT@', '-n', 'etc2_rgba_stitch_source'],
)

astc_glsl_h = custom_target(
  'astc_glsl.h',
  input : [files_xxd, 'astc_decoder.glsl'],
  output : 'astc_glsl.h',
  command : [prog_python, '@INPUT@', '@OUTPUT@', '-n', 'astc_source'],
)

files_libglsl = files(
  'ast.h',
  'ast_array_index.cpp',
  'ast_expr.cpp',
  'ast_function.cpp',
  'ast_to_hir.cpp',
  'ast_type.cpp',
  'builtin_functions.cpp',
  'builtin_functions.h',
  'builtin_types.cpp',
  'builtin_variables.cpp',
  'gl_nir_detect_function_recursion.c',
  'gl_nir_lower_atomics.c',
  'gl_nir_lower_images.c',
  'gl_nir_lower_blend_equation_advanced.c',
  'gl_nir_lower_buffers.c',
  'gl_nir_lower_discard_flow.c',
  'gl_nir_lower_named_interface_blocks.c',
  'gl_nir_lower_packed_varyings.c',
  'gl_nir_lower_samplers.c',
  'gl_nir_lower_samplers_as_deref.c',
  'gl_nir_lower_xfb_varying.c',
  'gl_nir_link_atomics.c',
  'gl_nir_link_functions.c',
  'gl_nir_link_interface_blocks.c',
  'gl_nir_link_uniform_blocks.c',
  'gl_nir_link_uniform_initializers.c',
  'gl_nir_link_uniforms.c',
  'gl_nir_link_varyings.c',
  'gl_nir_link_varyings.h',
  'gl_nir_link_xfb.c',
  'gl_nir_linker.c',
  'gl_nir_linker.h',
  'gl_nir.h',
  'glsl_parser_extras.cpp',
  'glsl_parser_extras.h',
  'glsl_symbol_table.cpp',
  'glsl_symbol_table.h',
  'glsl_to_nir.cpp',
  'glsl_to_nir.h',
  'hir_field_selection.cpp',
  'ir_basic_block.cpp',
  'ir_basic_block.h',
  'ir_builder.cpp',
  'ir_builder.h',
  'ir_clone.cpp',
  'ir_constant_expression.cpp',
  'ir.cpp',
  'ir.h',
  'ir_expression_flattening.cpp',
  'ir_expression_flattening.h',
  'ir_function_detect_recursion.cpp',
  'ir_function.cpp',
  'ir_hierarchical_visitor.cpp',
  'ir_hierarchical_visitor.h',
  'ir_hv_accept.cpp',
  'ir_list.h',
  'ir_optimization.h',
  'ir_print_visitor.cpp',
  'ir_print_visitor.h',
  'ir_rvalue_visitor.cpp',
  'ir_rvalue_visitor.h',
  'ir_validate.cpp',
  'ir_variable_refcount.cpp',
  'ir_variable_refcount.h',
  'ir_visitor.h',
  'linker_util.h',
  'linker_util.cpp',
  'lower_builtins.cpp',
  'lower_instructions.cpp',
  'lower_jumps.cpp',
  'lower_mat_op_to_vec.cpp',
  'lower_packing_builtins.cpp',
  'lower_precision.cpp',
  'lower_subroutine.cpp',
  'lower_vec_index_to_cond_assign.cpp',
  'lower_vector_derefs.cpp',
  'opt_algebraic.cpp',
  'opt_dead_builtin_variables.cpp',
  'opt_dead_code.cpp',
  'opt_flatten_nested_if_blocks.cpp',
  'opt_function_inlining.cpp',
  'opt_if_simplification.cpp',
  'opt_minmax.cpp',
  'opt_rebalance_tree.cpp',
  'opt_tree_grafting.cpp',
  'propagate_invariance.cpp',
  'string_to_uint_map.cpp',
  'string_to_uint_map.h',
  'serialize.cpp',
  'serialize.h',
  'shader_cache.cpp',
  'shader_cache.h',
)

files_libglsl_standalone = files(
  'standalone_scaffolding.cpp',
  'standalone_scaffolding.h',
  'standalone.cpp',
  'standalone.h',
)

libglsl_headers = [
  ir_expression_operation_h, cross_platform_settings_piece_all_h, bc1_glsl_h,
  bc4_glsl_h, etc2_rgba_stitch_glsl_h, astc_glsl_h
]

libglsl = static_library(
  'glsl',
  [files_libglsl, glsl_parser, glsl_lexer_cpp, libglsl_headers,
   ir_expression_operation_strings_h, ir_expression_operation_constant_h,
   float64_glsl_h],
  c_args : [c_msvc_compat_args, no_override_init_args],
  cpp_args : [cpp_msvc_compat_args],
  gnu_symbol_visibility : 'hidden',
  link_with : libglcpp,
  include_directories : [inc_include, inc_src, inc_mapi, inc_mesa, inc_gallium, inc_gallium_aux],
  dependencies : [idep_nir, idep_mesautil],
  build_by_default : false,
)

idep_libglsl = declare_dependency(
  sources: libglsl_headers,
  link_with: libglsl,
)

libglsl_standalone = static_library(
  'glsl_standalone',
  [files_libglsl_standalone, ir_expression_operation_h],
  c_args : [c_msvc_compat_args, no_override_init_args],
  cpp_args : [cpp_msvc_compat_args],
  gnu_symbol_visibility : 'hidden',
  include_directories : [inc_include, inc_src, inc_mapi, inc_mesa, inc_gallium, inc_gallium_aux],
  link_with : [libglsl, libglsl_util, libglcpp_standalone],
  dependencies : [idep_mesautil, idep_getopt, idep_compiler, idep_nir],
  build_by_default : false,
)

glsl_compiler = executable(
  'glsl_compiler',
  'main.cpp',
  c_args : [c_msvc_compat_args, no_override_init_args],
  cpp_args : [cpp_msvc_compat_args],
  gnu_symbol_visibility : 'hidden',
  dependencies : [dep_clock, dep_thread, idep_getopt, idep_mesautil],
  include_directories : [inc_include, inc_src, inc_mapi, inc_mesa, inc_gallium, inc_gallium_aux],
  link_with : [libglsl_standalone],
  build_by_default : with_tools.contains('glsl'),
  install : with_tools.contains('glsl'),
)

if with_any_opengl and with_tests
  subdir('tests')
endif
