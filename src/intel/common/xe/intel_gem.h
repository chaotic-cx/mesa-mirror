/*
 * Copyright © 2023 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include "common/intel_engine.h"

bool xe_gem_read_render_timestamp(int fd, uint64_t *value);
bool
xe_gem_read_correlate_cpu_gpu_timestamp(int fd,
                                        enum intel_engine_class engine_class,
                                        uint16_t engine_instance,
                                        clockid_t cpu_clock_id,
                                        uint64_t *cpu_timestamp,
                                        uint64_t *gpu_timestamp,
                                        uint64_t *cpu_delta);
bool xe_gem_can_render_on_fd(int fd);
bool xe_gem_supports_protected_exec_queue(int fd);

void intel_xe_gem_add_ext(uint64_t *ptr, uint32_t ext_name, void *data);
