#!/usr/bin/env bash
# shellcheck disable=SC2086 # we want word splitting

# When changing this file, you need to bump the following
# .gitlab-ci/image-tags.yml tags:
# DEBIAN_BUILD_TAG

set -e

. .gitlab-ci/setup-test-env.sh

set -o xtrace

export DEBIAN_FRONTEND=noninteractive
: "${LLVM_VERSION:?llvm version not set!}"

# Ephemeral packages (installed for this script and removed again at the end)
EPHEMERAL=(
    autoconf
    automake
    autotools-dev
    bzip2
    libtool
    libssl-dev
)

DEPS=(
    check
    "clang-${LLVM_VERSION}"
    libasan8
    libarchive-dev
    libdrm-dev
    "libclang-cpp${LLVM_VERSION}-dev"
    "libclang-rt-${LLVM_VERSION}-dev"
    libgbm-dev
    libglvnd-dev
    liblua5.3-dev
    libxcb-dri2-0-dev
    libxcb-dri3-dev
    libxcb-glx0-dev
    libxcb-present-dev
    libxcb-randr0-dev
    libxcb-shm0-dev
    libxcb-sync-dev
    libxcb-xfixes0-dev
    libxcb1-dev
    libxml2-dev
    "llvm-${LLVM_VERSION}-dev"
    ocl-icd-opencl-dev
    python3-pip
    procps
    spirv-tools
    strace
    time
    zstd
)

apt-get update

apt-get install -y --no-remove \
      "${DEPS[@]}" "${EPHEMERAL[@]}"


. .gitlab-ci/container/container_pre_build.sh

# dependencies where we want a specific version
export              XORG_RELEASES=https://xorg.freedesktop.org/releases/individual

export         XORGMACROS_VERSION=util-macros-1.19.0

. .gitlab-ci/container/build-mold.sh

curl -L --retry 4 -f --retry-all-errors --retry-delay 60 -O \
  $XORG_RELEASES/util/$XORGMACROS_VERSION.tar.bz2
tar -xvf $XORGMACROS_VERSION.tar.bz2 && rm $XORGMACROS_VERSION.tar.bz2
cd $XORGMACROS_VERSION; ./configure; make install; cd ..
rm -rf $XORGMACROS_VERSION

. .gitlab-ci/container/build-wayland.sh

. .gitlab-ci/container/build-shader-db.sh

. .gitlab-ci/container/build-directx-headers.sh

. .gitlab-ci/container/build-bindgen.sh

############### Uninstall the build software

apt-get purge -y "${EPHEMERAL[@]}"

. .gitlab-ci/container/container_post_build.sh
