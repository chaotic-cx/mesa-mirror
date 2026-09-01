/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/brw/brw_compiler.h"
#include "util/shader_stats.h"
#include "nir.h"

struct intel_device_info;
struct nir_shader_compiler_options;

struct jay_shader_bin {
   const uint32_t *kernel;
   uint32_t size;
   struct genisa_stats stats[3];
};

struct jay_compile_params {
   void *mem_ctx;

   nir_shader *nir;

   union brw_any_prog_key *key;
   union brw_any_prog_data *prog_data;

   struct {
      const struct brw_mue_map *mue_map;
   } fs;

   debug_archiver *archiver;

   char **error_str;
};

struct jay_shader_bin *jay_compile(const struct intel_device_info *devinfo,
                                   const struct jay_compile_params *params);
