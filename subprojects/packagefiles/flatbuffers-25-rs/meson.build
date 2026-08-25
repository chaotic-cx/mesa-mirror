project(
  'flatbuffers_rust',
  'rust',
  version : '25.2.10',
  license :'Apache-2.0',
)

rc = meson.get_compiler('rust')

rust_args = [
  # '--no-default-features',
]

bitflags = subproject('bitflags-2-rs').get_variable('lib')

lib_flatbuffers_rust = static_library(
  'flatbuffers_rust',
  'src/lib.rs',
  rust_args : rust_args,
  link_with : [bitflags],
  rust_abi : 'rust',
  native : true,
)

dep_flatbuffers_rust = declare_dependency(
  link_with : [lib_flatbuffers_rust, bitflags],
)
