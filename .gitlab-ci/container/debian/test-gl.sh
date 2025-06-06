#!/usr/bin/env bash
# shellcheck disable=SC2086 # we want word splitting

set -e

. .gitlab-ci/setup-test-env.sh

set -o xtrace

uncollapsed_section_start debian_setup "Base Debian system setup"

export DEBIAN_FRONTEND=noninteractive
: "${LLVM_VERSION:?llvm version not set!}"

apt-get install -y libelogind0  # this interfere with systemd deps, install separately

# Ephemeral packages (installed for this script and removed again at the end)
EPHEMERAL=(
    bzip2
    ccache
    "clang-${LLVM_VERSION}"
    cmake
    dpkg-dev
    g++
    glslang-tools
    libasound2-dev
    libcap-dev
    "libclang-cpp${LLVM_VERSION}-dev"
    libdrm-dev
    libfontconfig-dev
    libgl-dev
    libgles2-mesa-dev
    libglu1-mesa-dev
    libglx-dev
    libpciaccess-dev
    libpng-dev
    libudev-dev
    libwaffle-dev
    libwayland-dev
    libx11-xcb-dev
    libxcb-dri2-0-dev
    libxkbcommon-dev
    libxrandr-dev
    libxrender-dev
    "llvm-${LLVM_VERSION}-dev"
    "lld-${LLVM_VERSION}"
    make
    meson
    ocl-icd-opencl-dev
    patch
    pkgconf
    python-is-python3
    python3-distutils
    xz-utils
)

DEPS=(
    libfontconfig1
    libglu1-mesa
    libvulkan-dev
)

apt-get update

apt-get install -y --no-remove "${DEPS[@]}" "${EPHEMERAL[@]}" \
      $EXTRA_LOCAL_PACKAGES


. .gitlab-ci/container/container_pre_build.sh

section_end debian_setup

############### Build ANGLE

if [ "$DEBIAN_ARCH" != "armhf" ]; then
  ANGLE_TARGET=linux \
  . .gitlab-ci/container/build-angle.sh
fi

############### Build piglit

PIGLIT_OPTS="-DPIGLIT_USE_WAFFLE=ON
	     -DPIGLIT_USE_GBM=ON
	     -DPIGLIT_USE_WAYLAND=ON
	     -DPIGLIT_USE_X11=ON
	     -DPIGLIT_BUILD_GLX_TESTS=ON
	     -DPIGLIT_BUILD_EGL_TESTS=ON
	     -DPIGLIT_BUILD_WGL_TESTS=OFF
	     -DPIGLIT_BUILD_GL_TESTS=ON
	     -DPIGLIT_BUILD_GLES1_TESTS=ON
	     -DPIGLIT_BUILD_GLES2_TESTS=ON
	     -DPIGLIT_BUILD_GLES3_TESTS=ON
	     -DPIGLIT_BUILD_CL_TESTS=ON
	     -DPIGLIT_BUILD_VK_TESTS=ON
	     -DPIGLIT_BUILD_DMA_BUF_TESTS=ON" \
  . .gitlab-ci/container/build-piglit.sh

############### Build dEQP GL

DEQP_API=tools \
DEQP_TARGET=surfaceless \
. .gitlab-ci/container/build-deqp.sh

DEQP_API=GL \
DEQP_TARGET=surfaceless \
. .gitlab-ci/container/build-deqp.sh

DEQP_API=GLES \
DEQP_TARGET=surfaceless \
. .gitlab-ci/container/build-deqp.sh

rm -rf /VK-GL-CTS

############### Build validation layer for zink

. .gitlab-ci/container/build-vulkan-validation.sh


############### Build SKQP

if [ "$DEBIAN_ARCH" != "armhf" ]; then
  . .gitlab-ci/container/build-skqp.sh
fi

############### Uninstall the build software

uncollapsed_section_switch debian_cleanup "Cleaning up base Debian system"

apt-get purge -y "${EPHEMERAL[@]}"

. .gitlab-ci/container/container_post_build.sh

section_end debian_cleanup

############### Remove unused packages

. .gitlab-ci/container/strip-rootfs.sh
