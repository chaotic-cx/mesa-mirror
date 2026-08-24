project('flatcc', 'c', version : '0.6.4', license : 'Apache-2.0')

### flatcc compiler (CLI codegen tool, native) -> bin/flatcc
flatcc_compiler_srcs = files(
  'external/hash/str_set.c',
  'external/hash/ptr_set.c',
  'src/compiler/hash_tables/symbol_table.c',
  'src/compiler/hash_tables/scope_table.c',
  'src/compiler/hash_tables/name_table.c',
  'src/compiler/hash_tables/schema_table.c',
  'src/compiler/hash_tables/value_set.c',
  'src/compiler/fileio.c',
  'src/compiler/parser.c',
  'src/compiler/semantics.c',
  'src/compiler/coerce.c',
  'src/compiler/flatcc.c',
  'src/compiler/codegen_c.c',
  'src/compiler/codegen_c_reader.c',
  'src/compiler/codegen_c_sort.c',
  'src/compiler/codegen_c_builder.c',
  'src/compiler/codegen_c_verifier.c',
  'src/compiler/codegen_c_sorter.c',
  'src/compiler/codegen_c_json_parser.c',
  'src/compiler/codegen_c_json_printer.c',
  'src/runtime/builder.c',
  'src/runtime/emitter.c',
  'src/runtime/refmap.c',
  'src/cli/flatcc_cli.c',
)

flatcc_exe = executable(
  'flatcc',
  flatcc_compiler_srcs,
  include_directories : include_directories('include', 'external', 'config'),
  native : true,
)

### flatcc runtime reader/verifier support (cross-compiled by consumers for
### their target). Only verifier.c is exposed: builder/emitter/refmap/
### json_parser/json_printer are part of upstream flatcc but are unused by
### a reader-only consumer (e.g. a freestanding device target that only
### decodes buffers, never builds or prints them).
flatcc_verifier_src = files('src/runtime/verifier.c')
flatcc_rt_include = meson.current_source_dir() / 'include'

meson.override_find_program('flatcc', flatcc_exe)
