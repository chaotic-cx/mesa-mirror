/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::collections::HashMap;
use std::ffi::c_double;

use crate::jit::device::ElfData;
use crate::jit::device::*;
use earthqaic_rs_bindings::*;

#[derive(Default)]
#[repr(C)]
pub struct EarthqaicMLSubgraph {
    base: pipe_ml_subgraph,
    pub context: *mut pipe_context,

    pub operations: Vec<EarthqaicMLOperation>,
    pub num_tensors: usize,
    pub io_tensors: TensorList,
    pub weights: TensorList,
    pub biases: TensorList,

    // Jit specific
    pub buf_id: u32,
    pub elf_data: ElfData,
    pub exe_name: String,
    pub operator_addr: u64,
    pub output_addr: u64,
    pub mappings: HashMap<u32, u32>,
    pub alloc_base: u64,
}

// Must be 32b compliant as operator library is 32b
#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct EarthqaicMLOperation {
    pub type_: u32,

    // Track io, weight, and bias tensors with their teflon indices.
    // These can be used to index into the subgraph's TensorLists.
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub weight: u32,
    pub bias: u32,

    pub stride_x: u32,
    pub stride_y: u32,
    pub has_padding: bool,
}

#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct Tensor {
    // Device ID enumerated by driver
    pub dev_id: u32,

    // Output tensors dont need to write (non-quantization)
    // data to the device
    is_output: bool,

    // Quantization data
    size: u32,
    width: u32,
    height: u32,
    depth: u32,
    zero_point: i32,
    padding: u32,
    scale: c_double,

    data: *mut u8,
}

#[derive(Debug, Default)]
pub struct TensorList {
    // Number of tensors in the TensorList
    pub num: usize,

    // Mapping between a TensorFlow id and the tensor data.
    // Importantly, the Tensor object also contains the tensor's device
    // specific ID used to serialize the tensors on the device.
    //
    // TensorFlow subgraphs often contain tensor indices far above
    // the number of tensors in a given subgraph. To avoid wasting
    // space in the device buffers for unused tensors, we must normalize
    // these values.
    // E.g., 89 = ADD(87, 88) -> 2 = ADD(0, 1)
    pub tensors: HashMap<u32, Tensor>,

    // Location and size of tensor buffers on device.
    pub buf_sizes: JitBuffer,
    pub buf_widths: JitBuffer,
    pub buf_heights: JitBuffer,
    pub buf_depths: JitBuffer,
    pub buf_zero_points: JitBuffer,
    pub buf_scales: JitBuffer,
    pub buf_tensors: Vec<JitBuffer>,
}
