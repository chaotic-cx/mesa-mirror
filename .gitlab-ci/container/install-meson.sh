#!/usr/bin/env bash

# When changing this file, you need to bump the following
# .gitlab-ci/image-tags.yml tags:
# ALPINE_X86_64_BUILD_TAG
# DEBIAN_BUILD_BASE_TAG
# FEDORA_X86_64_BUILD_TAG

rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# rusticl tests need 1.9+
pip3 install 'meson==1.9.2'
