# Copyright © 2018 Rob Clark
# SPDX-License-Identifier: MIT

ir3_nir_trig_c = custom_target(
  'ir3_nir_trig.c',
  input : 'ir3_nir_trig.py',
  output : 'ir3_nir_trig.c',
  command : [
    prog_python, '@INPUT@', '-p', dir_compiler_nir,
  ],
  capture : true,
  depend_files : nir_algebraic_depends,
)

ir3_nir_imul_c = custom_target(
  'ir3_nir_imul.c',
  input : 'ir3_nir_imul.py',
  output : 'ir3_nir_imul.c',
  command : [
    prog_python, '@INPUT@', '-p', dir_compiler_nir,
  ],
  capture : true,
  depend_files : nir_algebraic_depends,
)

ir3_nir_branch_and_or_not_c = custom_target(
  'ir3_nir_branch_and_or_not.c',
  input : 'ir3_nir_branch_and_or_not.py',
  output : 'ir3_nir_branch_and_or_not.c',
  command : [
    prog_python, '@INPUT@', '-p', dir_compiler_nir,
  ],
  capture : true,
  depend_files : nir_algebraic_depends,
)

ir3_nir_triop_bitwise_c = custom_target(
  'ir3_nir_triop_bitwise.c',
  input : 'ir3_nir_triop_bitwise.py',
  output : 'ir3_nir_triop_bitwise.c',
  command : [
    prog_python, '@INPUT@', '-p', dir_compiler_nir,
  ],
  capture : true,
  depend_files : nir_algebraic_depends,
)

ir3_parser = custom_target(
  'ir3_parser.[ch]',
  input: 'ir3_parser.y',
  output: ['ir3_parser.c', 'ir3_parser.h'],
  command: [
    prog_bison, '@INPUT@', '--name-prefix=ir3_yy', '--defines=@OUTPUT1@', '--output=@OUTPUT0@'
  ]
)

ir3_lexer = custom_target(
  'ir3_lexer.c',
  input: 'ir3_lexer.l',
  output: 'ir3_lexer.c',
  command: [
    prog_flex, '-o', '@OUTPUT@', '@INPUT@'
  ]
)

libfreedreno_ir3_files = files(
  'disasm-a3xx.c',
  'instr-a3xx.h',
  'ir3.c',
  'ir3_a4xx.c',
  'ir3_a6xx.c',
  'ir3_alias.c',
  'ir3_array_to_ssa.c',
  'ir3_assembler.c',
  'ir3_assembler.h',
  'ir3_compiler_nir.c',
  'ir3_compiler.c',
  'ir3_compiler.h',
  'ir3_context.c',
  'ir3_context.h',
  'ir3_cf.c',
  'ir3_cp.c',
  'ir3_cse.c',
  'ir3_dce.c',
  'ir3_delay.c',
  'ir3_dominance.c',
  'ir3_disk_cache.c',
  'ir3_image.c',
  'ir3_image.h',
  'ir3.h',
  'ir3_legalize.c',
  'ir3_legalize_relative.c',
  'ir3_liveness.c',
  'ir3_lower_parallelcopy.c',
  'ir3_lower_shared_phi.c',
  'ir3_lower_spill.c',
  'ir3_lower_subgroups.c',
  'ir3_merge_regs.c',
  'ir3_nir.c',
  'ir3_nir.h',
  'ir3_nir_analyze_ubo_ranges.c',
  'ir3_nir_lower_64b.c',
  'ir3_nir_lower_driver_params_to_ubo.c',
  'ir3_nir_lower_load_barycentric_at_sample.c',
  'ir3_nir_lower_load_barycentric_at_offset.c',
  'ir3_nir_lower_push_consts_to_preamble.c',
  'ir3_nir_lower_shading_rate.c',
  'ir3_nir_lower_io_offsets.c',
  'ir3_nir_lower_tess.c',
  'ir3_nir_lower_tex_prefetch.c',
  'ir3_nir_move_varying_inputs.c',
  'ir3_nir_lower_layer_id.c',
  'ir3_nir_opt_preamble.c',
  'ir3_opt_predicates.c',
  'ir3_postsched.c',
  'ir3_preamble.c',
  'ir3_print.c',
  'ir3_ra.c',
  'ir3_ra.h',
  'ir3_ra_predicates.c',
  'ir3_ra_validate.c',
  'ir3_reconvergence.c',
  'ir3_remove_unreachable.c',
  'ir3_rpt.c',
  'ir3_sched.c',
  'ir3_shader.c',
  'ir3_shader.h',
  'ir3_shared_folding.c',
  'ir3_shared_ra.c',
  'ir3_spill.c',
  'ir3_validate.c',
)

libfreedreno_ir3 = static_library(
  'freedreno_ir3',
  [libfreedreno_ir3_files, ir3_nir_trig_c, ir3_nir_imul_c, ir3_nir_branch_and_or_not_c, ir3_nir_triop_bitwise_c, ir3_parser[0], ir3_parser[1], ir3_lexer],
  include_directories : [inc_freedreno, inc_include, inc_src],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  link_with: [libir3encode, libfreedreno_common],
  dependencies : [idep_libir3decode, idep_nir_headers, dep_dl, idep_mesautil],
  build_by_default : false,
)

test('ir3_disasm',
  executable(
    'ir3_disasm',
    'tests/disasm.c',
    link_args : [ld_args_build_id],
    link_with: [libfreedreno_ir3],
    dependencies: [idep_libir3decode, idep_mesautil, idep_nir],
    include_directories: [inc_freedreno, inc_include, inc_src],
  ),
  suite: ['freedreno'],
)

test('ir3_delay_test',
  executable(
    'ir3_delay_test',
    'tests/delay.c',
    link_with: libfreedreno_ir3,
    link_args: ld_args_build_id,
    dependencies: [idep_mesautil, idep_nir],
    include_directories: [inc_freedreno, inc_include, inc_src],
  ),
  suite: ['freedreno'],
)
