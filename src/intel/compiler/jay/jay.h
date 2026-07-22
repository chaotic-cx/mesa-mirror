/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/brw/brw_compiler.h"
#include "nir.h"

struct intel_device_info;
struct nir_shader_compiler_options;

struct jay_compile_params {
   void *mem_ctx;

   nir_shader *nir;

   union brw_any_prog_key *key;
   union brw_any_prog_data *prog_data;

   struct intel_push_layout push_layout;

   struct {
      /* Whether the compute shader uses BTD stack IDs, delivered in the
       * thread payload.
       */
      bool uses_btd_stack_ids;
      struct intel_cs_dispatch_params dispatch;
      struct brw_cs_push_const_info push;
      bool force_simd32;
   } cs;

   struct {
      const struct brw_mue_map *mue_map;
   } fs;

   debug_archiver *archiver;

   char **error_str;
};

intel_shader_bin *jay_compile(const struct intel_device_info *devinfo,
                              const struct jay_compile_params *params);
