/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use crate::earthqaic_device::*;
use crate::earthqaic_rs_bindings::*;
use crate::jit::device::*;

#[repr(C)]
#[derive(Default)]
pub struct EarthqaicComputeShader {
    pub base: pipe_shader_state,
    operator: Vec<u8>,
    elf_data: ElfData,
    operator_buf: JitBuffer,
    global_buffers: Vec<JitBuffer>,
    global_resources: Vec<*mut EarthqaicResource>,
    constant_buffer: JitBuffer,
    constant_resource: Vec<u8>,
    pub exe_name: String,
    alloc_base: u64,
}
