project(
  'llvm_sys',
  'rust',
  version : '201.0.1',
  license :'MIT',
)

rc = meson.get_compiler('rust')

libc = subproject('libc-0.2-rs').get_variable('lib')

lib_llvm_sys = static_library(
  'llvm_sys',
  'src/lib.rs',
  link_with : [libc],
  rust_abi : 'rust',
  native : true,
  rust_args : ['-Anon_snake_case'],
)

dep_llvm_sys = declare_dependency(
  link_with : [lib_llvm_sys, libc],
)
