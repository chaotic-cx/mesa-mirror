# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# This meson.build integrates the GPUOpen DevDriver library into Mesa's build
# system as a meson subproject. DevDriver enables RADV to communicate with the
# Radeon Developer Panel (RDP).
#
# Sourced from https://github.com/GPUOpen-Tools/devdriver, see
# subprojects/devdriver.wrap for the exact version. Original repo uses CMake
# which is why this Meson file is needed. It builds only the minimal set of
# required files.
#
# To use this file during compilation, Meson copies it into
# 'subprojects/devdriver-<version>/', but only when it extracts the sources.
# After any changes, run e.g. 'meson subprojects packagefiles --apply devdriver'
# (or reset/purge/rm extracted dir) to use the updated meson file.
#
# Dependencies:
#   - mpack: MessagePack library (included in third_party/)
#   - dd_crc32: CRC32 implementation (included in third_party/)
#   - metrohash: Hash library (included in third_party/)
#   - rapidjson: Header-only JSON library (included in third_party/)
#
# =============================================================================

project(
  'devdriver',
  'cpp',
  version : '26.07.23',
  license : 'MIT',
  default_options : ['cpp_std=c++17'],
)

cpp = meson.get_compiler('cpp')
dep_threads = dependency('threads')

# Include directories - public (visible to consumers)
devdriver_inc = include_directories(
  # Legacy devdriver headers
  'shared/legacy/inc',
  'shared/legacy/core/inc',
  'shared/legacy',
  # Common components
  'shared/ddCommon/inc',
  'shared/ddApi/inc',
  # APIs
  'apis/apis',
  'apis/common/inc',
  # RPC server API (for UberTrace)
  'apis/ddRpc/ddRpcServer/inc',
  'apis/ddRpc/ddRpcShared/inc',
  'apis/ddSocket/inc',           # ddSocket.h
  'apis/ddNet/inc',              # ddNet.h
  # Third-party dependencies
  'third_party/mpack',
  'third_party/stb_sprintf/inc',
  'third_party/dd_crc32/inc',
  'third_party/metrohash/src',
)

# Include directories - private
devdriver_inc_private = include_directories(
  'shared/legacy/src',
  'apis/ddRpc/ddRpcServer/src',  # rpcServer.h, rpcClientHandler.h
  'apis/ddRpc/ddRpcShared/src',
  'apis/ddSocket/src',           # socketServer.h
  'apis/ddNet/src',
  # Third-party dependencies
  'third_party/rapidjson/include',
)

# =============================================================================
# Source files
# =============================================================================

# Core platform abstraction layer
devdriver_core_sources = files(
  'shared/legacy/core/src/ddcMemory.cpp',
  'shared/legacy/core/src/ddcPlatform.cpp',
  'shared/legacy/core/src/platforms/ddcPosixPlatform.cpp',
)

# Common utilities
devdriver_common_sources = files(
  'shared/ddCommon/src/ddCommon.cpp',
  'shared/ddApi/src/ddApi.cpp',
  'apis/common/src/dd_timeout_constants.cpp',
)

# Legacy devdriver server core
devdriver_legacy_sources = files(
  # Core session and message handling
  'shared/legacy/src/session.cpp',
  'shared/legacy/src/sessionManager.cpp',
  'shared/legacy/src/messageChannel.cpp',
  'shared/legacy/src/baseProtocolServer.cpp',
  'shared/legacy/src/legacyProtocolClient.cpp',
  'shared/legacy/src/protocolSession.cpp',
  'shared/legacy/src/protocols/systemProtocols.cpp',
  'shared/legacy/src/gpuopen.cpp',

  # Server and client implementations
  'shared/legacy/src/devDriverServer.cpp',

  # Transfer and URI services
  'shared/legacy/src/ddTransferManager.cpp',
  'shared/legacy/src/ddURIRequestContext.cpp',
  'shared/legacy/src/ddClientURIService.cpp',
  'shared/legacy/src/ddUriInterface.cpp',

  # Transport - Linux uses Unix domain sockets
  'shared/legacy/src/posix/ddPosixSocket.cpp',
  'shared/legacy/src/socketMsgTransport.cpp',

  # Utility functions
  'shared/legacy/src/util/ddEventTimer.cpp',
  'shared/legacy/src/util/ddJsonWriter.cpp',
  'shared/legacy/src/util/ddTextWriter.cpp',
  'shared/legacy/src/util/ddByteReader.cpp',
  'shared/legacy/src/util/ddMetroHash.cpp',
  'shared/legacy/src/util/hashFunc.cpp',
  'shared/legacy/src/util/sharedptr.cpp',
  'shared/legacy/src/util/string.cpp',
  'shared/legacy/src/util/vector.cpp',
)

# Protocol implementations
devdriver_protocol_sources = files(
  # Driver Control Protocol - handles driver state machine, halting, stepping
  'shared/legacy/src/protocols/driverControlServer.cpp',

  # RGP Protocol - handles trace requests and data transfer
  'shared/legacy/src/protocols/rgpServer.cpp',

  # Transfer Protocol - handles large data transfers
  'shared/legacy/src/protocols/ddTransferServer.cpp',
  'shared/legacy/src/protocols/ddTransferClient.cpp',
  'shared/legacy/src/protocols/ddTransferProtocol.cpp',

  # URI Protocol - handles URI-based service requests
  'shared/legacy/src/protocols/ddURIServer.cpp',
  'shared/legacy/src/protocols/ddURIClient.cpp',

  # Event Protocol - handles event streaming
  'shared/legacy/src/protocols/ddEventServer.cpp',
  'shared/legacy/src/protocols/ddEventProvider.cpp',
  'shared/legacy/src/protocols/ddEventServerSession.cpp',
  'shared/legacy/src/protocols/ddEventProtocol.cpp',

  # Internal and Info services
  'shared/legacy/src/protocols/ddInternalService.cpp',
  'shared/legacy/src/protocols/ddInfoService.cpp',
)

# Third-party dependencies (compiled as part of devdriver)
devdriver_thirdparty_sources = files(
  'third_party/mpack/mpack.cpp',
  'third_party/stb_sprintf/src/stb_sprintf.cpp',
  'third_party/dd_crc32/src/dd_crc32.cpp',
  'third_party/metrohash/src/metrohash64.cpp',
)

# RPC server sources (for UberTrace support)
devdriver_rpc_sources = files(
  # RPC Server
  'apis/ddRpc/ddRpcServer/src/ddRpcServer.cpp',
  'apis/ddRpc/ddRpcServer/src/rpcServer.cpp',
  'apis/ddRpc/ddRpcServer/src/rpcClientHandler.cpp',
  # RPC Shared
  'apis/ddRpc/ddRpcShared/src/ddRpcShared.cpp',
  # Socket layer (required by RPC)
  'apis/ddSocket/src/ddSocket.cpp',
  'apis/ddSocket/src/socketServer.cpp',
  # Net layer (required by Socket)
  'apis/ddNet/src/ddNet.cpp',
)

# Combine all sources
devdriver_sources = \
  devdriver_core_sources + \
  devdriver_common_sources + \
  devdriver_legacy_sources + \
  devdriver_protocol_sources + \
  devdriver_thirdparty_sources + \
  devdriver_rpc_sources

# =============================================================================
# Build the static library
# =============================================================================

# DevDriver interface version, also exported to consumers, which must be built
# with the same value
devdriver_interface_args = ['-DGPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION=42']

libdevdriver = static_library(
  'devdriver',
  devdriver_sources,
  include_directories : [devdriver_inc, devdriver_inc_private],
  dependencies : dep_threads,
  cpp_args : devdriver_interface_args + cpp.get_supported_arguments(
    # As in devdriver:
    '-fno-exceptions',
    '-fno-rtti',
    '-fno-math-errno',
    '-fno-strict-aliasing',
    '-fvisibility-inlines-hidden',
    # Silence warnings
    '-Wno-maybe-uninitialized',
    # As in Mesa/RADV flags:
    '-flifetime-dse=1',
    # - Let linker drop unused code with --gc-sections
    '-ffunction-sections',
    '-fdata-sections',
  ) + [
    # Platform defines for Linux user-mode
    '-DDD_PLATFORM_LINUX_UM=1',
    '-DDD_PLATFORM_IS_UM=1',
    '-DDD_PLATFORM_IS_GNU=1',
    '-DDD_PLATFORM_IS_POSIX=1',
    '-DDD_TARGET_PLATFORM_LINUX=1',
    '-DDD_PLATFORM_STRING="Linux User Mode"',
    '-DDD_ARCH_BITS=@0@'.format(cpp.sizeof('void*') * 8),
    '-DDD_BRANCH_STRING="unknown"',
    '-DMPACK_WRITE_TRACKING=0',
  ],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

# Export dependency for build system to use
dep_devdriver = declare_dependency(
  include_directories : devdriver_inc,
  compile_args : devdriver_interface_args,
  link_with : libdevdriver,
  dependencies : dep_threads,
)
