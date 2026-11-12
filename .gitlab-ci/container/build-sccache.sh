#!/usr/bin/env bash
# shellcheck disable=SC2086 # we want word splitting

# When changing this file, you need to bump the following
# .gitlab-ci/image-tags.yml tags:
# ALPINE_X86_64_BUILD_TAG
# DEBIAN_BUILD_BASE_TAG
# FEDORA_X86_64_BUILD_TAG

set -ex

section_start sccache "Building sccache"

# sccache caches our rustc invocations the way ccache caches C/C++ ones.
#
# This is the head of https://github.com/mozilla/sccache/pull/2919, which is
# needed for meson's rustc invocations (several --emit kind=path arguments, no
# --out-dir) to be cacheable at all.  Switch to a release once it's included in
# one.
SCCACHE_REV=more-emit-review-2

# We only use a local disk cache, so leave out distributed compilation and the
# remote storage backends (and their dependencies).
#
# Keep it out of $PATH: meson autodetects sccache in preference to ccache and
# wraps every C/C++ compiler that isn't explicitly specified with it, while we
# only want it for rustc (see .gitlab-ci/meson/build.sh).
cargo install sccache \
  --git https://github.com/anholt/sccache \
  --rev ${SCCACHE_REV} \
  --no-default-features \
  --locked \
  -j ${FDO_CI_CONCURRENT:-4} \
  --root /usr/local/sccache

section_end sccache
