project(
  'elf',
  'rust',
  version : '0.8.0',
  license : 'MIT or Apache-2.0',
)

lib_elf_rs = static_library(
  'elf',
  'src/lib.rs',
  override_options : ['rust_std=2021', 'build.rust_std=2021'],
  rust_abi : 'rust',
  native : true,
)

dep_elf_rs = declare_dependency(
  link_with : [lib_elf_rs]
)
