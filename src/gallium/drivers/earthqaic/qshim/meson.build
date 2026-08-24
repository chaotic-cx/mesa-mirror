#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# In-tree build of the qshim device executable (network.elf) and the loose
# artifacts the driver consumes (metadata embedded in network.elf, plus
# constants.bin, constantsdesc.bin, md_struct.bin). No Docker, no qaic-cc,
# no QPC round-trip, no Python: upstream clang cross-compiles the vendored
# device runtime + qshim app for Hexagon, lld links network.elf, and the
# mdgen host tool generates the metadata/constants directly from
# jithwdev.json + the linked ELF's _qaic_start entrypoint.

fs = import('fs')

### Host toolchain discovery
# llvm-objcopy / llvm-ar / ld.lld ship beside clang; derive their paths so the
# qshim build tracks whichever clang meson already found.
prog_llvm_objcopy = find_program(
  run_command(prog_clang, '-print-prog-name=llvm-objcopy', check : true).stdout().strip(),
)
prog_llvm_ar = find_program(
  run_command(prog_clang, '-print-prog-name=llvm-ar', check : true).stdout().strip(),
)

# dvidelabs flatcc: generates the C reader/verifier API used by JitHwDevLib and
# provides the device-side verifier runtime. Pulled in as a meson subproject
# (subprojects/flatcc.wrap) rather than an external checkout.
flatcc_sub = subproject('flatcc')
prog_flatcc = find_program('flatcc')
flatcc_verifier_src = flatcc_sub.get_variable('flatcc_verifier_src')
flatcc_rt_include = flatcc_sub.get_variable('flatcc_rt_include')

# We also need flatbuffers for flatbuffers/flatbuffers.h referened
# during the qshim build.
# Note: this is distinct from the flatbuffers runtime rust crate,
# which is built as flatbuffers_rust and mapped to 'flatbuffers'
# when building the jit_generated.rs file to avoid name conflicts.
dep_flatbuffers = dependency(
  'flatbuffers',
  version : '>= 25.2.10',
  required : true,
  allow_fallback: false
)

### Hexagon cross-compile flags (freestanding device code)
qshim_hex_cflags = [
  '--target=hexagon-unknown-elf',
  '-mv68', '-mhvx=v68', '-mhvx-ieee-fp',
  '-g0', '-O2', '-G0',
  '-ffreestanding', '-fno-builtin', '-nostdlib',
]
qshim_hex_cxxflags = qshim_hex_cflags + [
  '-nostdinc++', '-fno-exceptions', '-fvisibility=default',
]

### flatcc C reader/verifier headers generated from jit.fbs
jit_flatcc_headers = custom_target(
  'jit_flatcc_headers',
  input : jit_fbs_dir / 'jit.fbs',
  output : [
    'flatbuffers_common_builder.h',
    'flatbuffers_common_reader.h',
    'jit_builder.h',
    'jit_reader.h',
    'jit_verifier.h',
  ],
  command : [prog_flatcc, '-a', '-o', '@OUTDIR@', '@INPUT@'],
)
jit_flatcc_incdir = meson.current_build_dir()

### Vendored device runtime -> libqaicrt.a + libdevRuntime.a
runtime_dir = meson.current_source_dir() / 'runtime'
runtime_inc = ['-I' + runtime_dir, '-I' + runtime_dir / 'libdev']

qaicrt_srcs = [
  'runtime/_qaic_start.cpp',
  'runtime/SerializedProgramDesc.cpp',
  'runtime/BufferDesc.cpp',
  'runtime/NSPContext.cpp',
  'runtime/Exit.cpp',
  'runtime/ComputeAPI.cpp',
  'runtime/SimpleAPI.cpp',
]

devrt_srcs = [
  'runtime/libdev/os_hexagon.cpp',
  'runtime/libdev/os_common.cpp',
  'runtime/libdev/libdev_interface.cpp',
  'runtime/libdev/libdev_udma.cpp',
]

gen_cpp_objs_rt = generator(
  prog_clangxx,
  arguments : [qshim_hex_cxxflags, runtime_inc, '@EXTRA_ARGS@', '-c', '@INPUT@', '-o', '@OUTPUT@', '-MD', '-MF', '@DEPFILE@'],
  depfile : '@PLAINNAME@.d',
  output : '@BASENAME@.o',
)

libqaicrt = custom_target(
  'libqaicrt_a',
  input : gen_cpp_objs_rt.process(qaicrt_srcs),
  output : 'libqaicrt.a',
  command : [prog_llvm_ar, 'rcs', '@OUTPUT@', '@INPUT@'],
)
libdevrt = custom_target(
  'libdevRuntime_a',
  input : gen_cpp_objs_rt.process(devrt_srcs),
  output : 'libdevRuntime.a',
  command : [prog_llvm_ar, 'rcs', '@OUTPUT@', '@INPUT@'],
)

### flatcc device runtime (reader/verifier used by the qshim) -> libflatccrt.a
# Only verifier.c is needed: JitHwDevLib.cpp decodes buffers via the
# header-only reader API and bounds-checks them via the verifier; it never
# builds, emits, or JSON-prints flatbuffers, so builder/emitter/refmap/
# json_parser/json_printer (part of upstream flatcc, unused here) are
# dropped from the device build entirely.
flatccrt_verifier_o = custom_target(
  'flatccrt_verifier_o',
  input : flatcc_verifier_src,
  output : 'verifier.o',
  command : [
    prog_clang, qshim_hex_cflags, '-I' + flatcc_rt_include, '-DFLATCC_NO_ASSERT',
    '-c', '@INPUT@', '-o', '@OUTPUT@',
  ],
)
libflatccrt = custom_target(
  'libflatccrt_a',
  input : flatccrt_verifier_o,
  output : 'libflatccrt.a',
  command : [prog_llvm_ar, 'rcs', '@OUTPUT@', '@INPUT@'],
)

### qshim application objects
qaic_c_inc = platform_defs_dir
qshim_app_inc = [
  '-I' + runtime_dir, '-I' + runtime_dir / 'libdev',
  '-I' + qaic_c_inc / 'jit', '-I' + qaic_c_inc,
  '-I' + meson.current_source_dir() / 'JitQshim/inc',
  '-I' + meson.current_source_dir() / 'JitQshim',
  '-I' + jit_flatcc_incdir,
  '-I' + meson.current_source_dir() / 'metadata',
  '-I' + flatcc_rt_include,
]

jitlib_srcs = [
  'JitQshim/JitHwDevLib.cpp',
  'JitQshim/JitHwDev.cpp',
]

gen_cpp_objs_jitlib = generator(
  prog_clangxx,
  arguments : [qshim_hex_cxxflags, '-DFLATCC_NO_ASSERT', qshim_app_inc, '-c', '@INPUT@', '-o', '@OUTPUT@'],
  depfile : '@PLAINNAME@.d',
  output : '@BASENAME@.o',
  depends : [jit_flatcc_headers[3]], # needs jit_reader.h
)

jitsrc_srcs = [
  'JitQshim/src/queue.c',
  'JitQshim/src/CorePool.c',
]

gen_c_objs_jitsrc = generator(
  prog_clang,
  arguments : [qshim_hex_cflags, qshim_app_inc, '-c', '@INPUT@', '-o', '@OUTPUT@'],
  depfile : '@PLAINNAME@.d',
  output : '@BASENAME@.o',
)

### Link network.elf (no metadata section yet)
# Archives in a --start-group so lld resolves cross-references between the
# runtime archives regardless of input order.
qshim_link_args = [
  '--target=hexagon-unknown-elf', '-fno-pic', '-nostdlib', '-nostdlib++',
  '-fuse-ld=lld',
  '-Wl,--image-base=0x2000000', '-Wl,-z,max-page-size=4096',
  '-Wl,--no-rosegment', '-Wl,-z,separate-code', '-Wl,-z,norelro',
  '-Wl,--no-eh-frame-hdr', '-Wl,-z,nognustack',
  '-Wl,--defsym,STACK_SIZE=12288', '-u', '_qaic_start',
  '-Wl,--entry=0'
]
network_elf_nomd = custom_target(
  'network_elf_nomd',
  input : [
    gen_cpp_objs_jitlib.process(jitlib_srcs),
    gen_c_objs_jitsrc.process(jitsrc_srcs),
    libqaicrt, libdevrt, libflatccrt,
  ],
  output : 'network.elf.nomd',
  command : [
    prog_clangxx, qshim_link_args,
    '-Wl,--start-group', '@INPUT@', '-Wl,--end-group',
    '-o', '@OUTPUT@',
  ],
)

### mdgen host tool: metadata flatbuffer C++ headers (built with flatc)
metadata_dir = meson.current_source_dir() / 'metadata'
mdgen_flatc_headers = custom_target(
  'mdgen_flatc_headers',
  input : [metadata_dir / 'AicMetadataFlat.fbs', metadata_dir / 'execContext.fbs'],
  output : ['AicMetadataFlat_generated.h', 'execContext_generated.h'],
  command : [prog_flatc, '--cpp', '--gen-object-api', '--no-warnings', '-o', '@OUTDIR@', '@INPUT@'],
)

# mdgen is an x86 host tool (runs at build time), so build it natively.
mdgen = executable(
  'mdgen',
  [metadata_dir / 'mdgen.cpp', metadata_dir / 'metadataFlatbufferWriter.cpp', mdgen_flatc_headers],
  include_directories : include_directories(
    'metadata',
  ),
  native : true,
  build_by_default : false,
)

### Generate metadata.bin / constants.bin / constantsdesc.bin / md_struct.bin
jithwdev_json = meson.current_source_dir() / 'JitQshim/jithwdev.json'
qshim_artifacts = custom_target(
  'qshim_artifacts',
  input : network_elf_nomd,
  output : ['metadata.bin', 'constants.bin', 'constantsdesc.bin', 'md_struct.bin'],
  command : [
    mdgen, jithwdev_json, '--elf', '@INPUT@',
    '-o', '@OUTPUT0@',
    '--constants', '@OUTPUT1@',
    '--constantsdesc', '@OUTPUT2@',
    '--md-struct', '@OUTPUT3@',
  ],
  install : true,
  install_dir : driver_install_dir / 'qshim',
)

### Embed the metadata flatbuffer as the `metadata` ELF section
network_elf = custom_target(
  'network_elf',
  input : [network_elf_nomd, qshim_artifacts[0]],
  output : 'network.elf',
  command : [
    prog_llvm_objcopy,
    '--add-section', 'metadata=@INPUT1@',
    '--set-section-flags', 'metadata=contents',
    '@INPUT0@', '@OUTPUT@',
  ],
  build_by_default : true,
  install : true,
  install_dir : driver_install_dir / 'qshim',
)
