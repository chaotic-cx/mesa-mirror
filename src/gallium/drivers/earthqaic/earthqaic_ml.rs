/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::collections::HashMap;
use std::ffi::{c_double, c_void};

use crate::earthqaic_device::*;
use crate::get_earthqaic_screen;
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

impl EarthqaicMLSubgraph {
    pub fn serialize_device_buffers(&self) -> Vec<JitBuffer> {
        let mut dev_buffers: Vec<JitBuffer> = Vec::new();

        // Dont send device buffers for weights that dont exist
        let tensor_list = if self.weights.num > 0 && self.biases.num > 0 {
            vec![&self.weights, &self.biases, &self.io_tensors]
        } else {
            vec![&self.io_tensors]
        };

        for tlist in tensor_list {
            dev_buffers.push(tlist.buf_sizes);
            dev_buffers.push(tlist.buf_widths);
            dev_buffers.push(tlist.buf_heights);
            dev_buffers.push(tlist.buf_depths);
            dev_buffers.push(tlist.buf_zero_points);
            dev_buffers.push(tlist.buf_scales);
            for i in 0..tlist.num {
                dev_buffers.push(tlist.buf_tensors[i]);
            }
        }
        dev_buffers
    }
}

fn quantized_supported(tensor: &pipe_tensor) -> bool {
    tensor.scales.is_null() && tensor.zero_points.is_null()
}

#[allow(non_upper_case_globals)]
pub extern "C" fn earthqaic_ml_operation_supported(
    _ml_device: *mut pipe_ml_device,
    poperation: *const pipe_ml_operation,
) -> bool {
    let pop: pipe_ml_operation = unsafe { *poperation };
    let pop_type = pop.type_;
    let input_tensor;
    let weight_tensor;
    let bias_tensor;
    let output_tensor;
    let conv;
    match pop_type {
        pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_CONVOLUTION => {
            unsafe {
                input_tensor = **(pop.input_tensors.wrapping_add(0));
                weight_tensor = *pop.__bindgen_anon_1.conv.weight_tensor;
                bias_tensor = *pop.__bindgen_anon_1.conv.bias_tensor;
                output_tensor = **(pop.output_tensors.wrapping_add(0));
                conv = pop.__bindgen_anon_1.conv;
            }

            let has_padding = conv.padding_top > 0
                || conv.padding_bottom > 0
                || conv.padding_left > 0
                || conv.padding_right > 0;

            if conv.depthwise
                || !conv.pointwise
                || conv.relu
                || has_padding
                || conv.stride_x != 1
                || conv.stride_y != 1
            {
                return false;
            }

            if !quantized_supported(&input_tensor)
                || !quantized_supported(&weight_tensor)
                || !quantized_supported(&bias_tensor)
                || !quantized_supported(&output_tensor)
            {
                return false;
            }

            if input_tensor.is_signed {
                return false;
            }

            return true;
        }
        pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_ADD => {
            unsafe {
                input_tensor = **(pop.input_tensors.wrapping_add(0));
            }
            if input_tensor.is_signed {
                return false;
            }
            return true;
        }
        _ => {
            return false;
        }
    }
}

#[allow(non_upper_case_globals)]
unsafe fn count_tensors(poperations: *const pipe_ml_operation, count: usize) -> u32 {
    let mut max_idx: u32 = 0;
    for i in 0..count {
        let poperation: *const pipe_ml_operation = poperations.wrapping_add(i);
        for j in 0..(*poperation).input_count {
            max_idx = std::cmp::max(
                max_idx,
                (*(*(*poperation).input_tensors.wrapping_add(j as usize))).index,
            );
        }
        for j in 0..(*poperation).output_count {
            max_idx = std::cmp::max(
                max_idx,
                (*(*(*poperation).output_tensors.wrapping_add(j as usize))).index,
            );
        }
        match (*poperation).type_ {
            pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_CONVOLUTION => {
                max_idx = std::cmp::max(
                    max_idx,
                    (*(*poperation).__bindgen_anon_1.conv.weight_tensor).index,
                );
                max_idx = std::cmp::max(
                    max_idx,
                    (*(*poperation).__bindgen_anon_1.conv.bias_tensor).index,
                );
            }
            pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_POOLING => {}
            pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_ADD => {}
            _ => {
                eqaic_eprintln!("Unsupported ML operation type");
                break;
            }
        }
    }
    max_idx + 1
}

unsafe fn create_tensor(tensor: *const pipe_tensor, dev_id: u32, is_output: bool) -> Tensor {
    Tensor {
        dev_id: dev_id,
        is_output: is_output,
        size: (*tensor).dims[0] * (*tensor).dims[1] * (*tensor).dims[2] * (*tensor).dims[3],
        width: (*tensor).dims[1],
        height: (*tensor).dims[2],
        depth: (*tensor).dims[3],
        zero_point: (*tensor).zero_point,
        padding: 0,
        scale: (*tensor).scale as f64,
        data: (*tensor).data,
    }
}

// Biases have a unique tensor format which needs a unique function to generate
unsafe fn create_bias_tensor(
    tensor: *const pipe_tensor,
    dev_id: u32,
    output_channels: u32,
) -> Tensor {
    Tensor {
        dev_id: dev_id,
        is_output: false,
        size: output_channels * (*tensor).type_size as u32,
        width: 0,
        height: 0,
        depth: 0,
        zero_point: (*tensor).zero_point,
        padding: 0,
        scale: (*tensor).scale as f64,
        data: (*tensor).data,
    }
}

// There needs to be an a priori understanding of tensor ordering between the operator
// library (device) and host. This ordering is determined by the index mappings that
// map the absolute tensorflow tensor index onto the number of tensors actually required
// by the subgraph (device IDs). The tensors are allocated sequentially on device according
// to their mapped value, i.e., ordered by mappings[tensorflow_index], aka device IDs.
//
// This function first determines the ordering of tensors by sorting the mapped indices
// by their values. For example, a mapping [(89, 1), (88, 2), (87,0)] -> [87, 89, 88].
// This indicates that tensor 87 corresponds to tensor 0 on the device, 88 corresponds to
// tensor 1 on the device, and so on.
//
// Once this ordering is determined, we created an ordered Vec<Tensor> to match this ordering.
// This ordered vector is passed to init_tensor_list, which allocates device memory in
// accordance with this ordering principle.
fn sort_tensor_list(tensor_map: &HashMap<u32, Tensor>) -> Vec<Tensor> {
    let mut sorted_tensors: Vec<Tensor> = tensor_map.values().cloned().collect();
    sorted_tensors.sort_by_key(|tensor| tensor.dev_id);
    sorted_tensors
}

fn alloc_tensor_list(pcontext: *mut pipe_context, list: &mut TensorList) {
    let tensors = sort_tensor_list(&list.tensors);
    let num = tensors.len();

    // In the case where a subgraph contains no weights
    // or biases, dont allocate any space for them.
    if num == 0 {
        return;
    }

    let jitdev = get_jitdev!(pcontext);

    // Allocate space for the tensor's quantization data
    let sizes_bytes = num * size_of::<u32>();
    list.buf_sizes.size = sizes_bytes;
    list.buf_sizes.addr = jitdev
        .alloc(sizes_bytes, 8)
        .expect("Could not allocate space for sizes.");

    let widths_bytes = num * size_of::<u32>();
    list.buf_widths.size = widths_bytes;
    list.buf_widths.addr = jitdev
        .alloc(widths_bytes, 8)
        .expect("Could not allocate space for widths.");

    let heights_bytes = num * size_of::<u32>();
    list.buf_heights.size = heights_bytes;
    list.buf_heights.addr = jitdev
        .alloc(heights_bytes, 8)
        .expect("Could not allocate space for heights.");

    let depths_bytes = num * size_of::<u32>();
    list.buf_depths.size = depths_bytes;
    list.buf_depths.addr = jitdev
        .alloc(depths_bytes, 8)
        .expect("Could not allocate space for depths.");

    let zero_points_bytes = num * size_of::<i32>();
    list.buf_zero_points.size = zero_points_bytes;
    list.buf_zero_points.addr = jitdev
        .alloc(zero_points_bytes, 8)
        .expect("Could not allocate space for zero points.");

    let scales_bytes = num * size_of::<c_double>();
    list.buf_scales.size = scales_bytes;
    list.buf_scales.addr = jitdev
        .alloc(scales_bytes, 8)
        .expect("Could not allocate space for scales.");

    // Allocate space for the tensors themselves
    for tensor in tensors {
        let size = tensor.size as usize;
        let buf = JitBuffer {
            size: size,
            addr: jitdev
                .alloc(size, 8)
                .expect("Could not allocate space for tensor data."),
        };
        list.buf_tensors.push(buf);
    }
}

fn populate_tensor_list(pcontext: *mut pipe_context, list: &mut TensorList) {
    let tensors = sort_tensor_list(&list.tensors);
    let num = tensors.len();

    // In the case where a subgraph contains no weights
    // or biases, no memory has been allocated to write to.
    if num == 0 {
        return;
    }

    let jitdev = get_jitdev!(pcontext);
    assert_eq!(list.num, tensors.len());

    for i in 0..num {
        let tensor = &tensors[i];
        jitdev
            .write_buffer(
                list.buf_sizes.addr + (i * size_of::<u32>()) as u64,
                &tensor.size.to_le_bytes().to_vec(),
                size_of::<u32>() as u64,
            )
            .expect("Could not write tensor size data");
        jitdev
            .write_buffer(
                list.buf_widths.addr + (i * size_of::<u32>()) as u64,
                &tensor.width.to_le_bytes().to_vec(),
                size_of::<u32>() as u64,
            )
            .expect("Could not write tensor width data");
        jitdev
            .write_buffer(
                list.buf_heights.addr + (i * size_of::<u32>()) as u64,
                &tensor.height.to_le_bytes().to_vec(),
                size_of::<u32>() as u64,
            )
            .expect("Could not write tensor height data");
        jitdev
            .write_buffer(
                list.buf_depths.addr + (i * size_of::<u32>()) as u64,
                &tensor.depth.to_le_bytes().to_vec(),
                size_of::<u32>() as u64,
            )
            .expect("Could not write tensor depth data");
        jitdev
            .write_buffer(
                list.buf_zero_points.addr + (i * size_of::<i32>()) as u64,
                &tensor.zero_point.to_le_bytes().to_vec(),
                size_of::<i32>() as u64,
            )
            .expect("Could not write tensor zero point data");
        jitdev
            .write_buffer(
                list.buf_scales.addr + (i * size_of::<c_double>()) as u64,
                &tensor.scale.to_le_bytes().to_vec(),
                size_of::<c_double>() as u64,
            )
            .expect("Could not write tensor scale data");

        // Output buffers do not need to be written
        if !tensor.is_output {
            // Write the tensor contents into the JIT buffer
            let buf = &list.buf_tensors[i];
            let data_ptr: *mut u8 = tensor.data;
            let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, tensor.size as usize) };
            jitdev
                .write_buffer(buf.addr, &data_slice.to_vec(), buf.size as u64)
                .unwrap();
        }
    }
}

pub extern "C" fn earthqaic_ml_subgraph_create(
    ml_device: *mut pipe_ml_device,
    poperations: *const pipe_ml_operation,
    count: u32,
) -> *mut pipe_ml_subgraph {
    let eml = ml_device as *mut EarthqaicMlDev;

    // If ml context doesnt exist, create it
    unsafe {
        let escreen = (*eml).screen;
        if (*eml).context.is_null() {
            let pscreen = &mut (*(*escreen).ml_device.screen).pscreen;
            (*eml).context = (*pscreen).context_create.unwrap()(pscreen, std::ptr::null_mut(), 0)
                as *mut EarthqaicContext;
        }
    }
    let pcontext = unsafe { (*eml).context as *mut pipe_context };
    let jitdev = get_jitdev!(pcontext);
    let mut subgraph = Box::new(EarthqaicMLSubgraph {
        base: pipe_ml_subgraph { device: ml_device },
        context: pcontext,
        alloc_base: u64::MAX,
        ..Default::default()
    });
    let tensor_count = unsafe { count_tensors(poperations, count as usize) };

    subgraph.num_tensors = tensor_count as usize;

    // Each tensor has a distinct "device id", which is different than its teflon id.
    // Teflon IDs may span many subgraphs, and may exceed the number of tensors
    // sent to the device.
    // Device IDs enumerate the tensors sent to the device per subgraph.
    // Device IDs are allocated when inserting tensors into their respective TensorLists.
    let mut io_dev_id: u32 = 0;
    let mut weight_dev_id: u32 = 0;
    let mut bias_dev_id: u32 = 0;

    unsafe {
        for i in 0..count {
            let poperation: *const pipe_ml_operation = poperations.wrapping_add(i as usize);
            let mut operation = EarthqaicMLOperation {
                ..Default::default()
            };
            if (*poperation).type_ == pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_ADD {
                operation.type_ = (*poperation).type_;

                let input_pipe_tensor_0 = *((*poperation).input_tensors).wrapping_add(0);
                let input_pipe_index_0 = (*input_pipe_tensor_0).index;
                let input_pipe_tensor_1 = *((*poperation).input_tensors).wrapping_add(1);
                let input_pipe_index_1 = (*input_pipe_tensor_1).index;
                let output_pipe_tensor = *((*poperation).output_tensors).wrapping_add(0);
                let output_pipe_index = (*output_pipe_tensor).index;

                // If input or output is not already in TensorList, add it
                if !(*subgraph)
                    .io_tensors
                    .tensors
                    .contains_key(&input_pipe_index_0)
                {
                    (*subgraph).io_tensors.tensors.insert(
                        input_pipe_index_0,
                        create_tensor(input_pipe_tensor_0, io_dev_id, false),
                    );
                    io_dev_id += 1;
                }

                if !(*subgraph)
                    .io_tensors
                    .tensors
                    .contains_key(&input_pipe_index_1)
                {
                    (*subgraph).io_tensors.tensors.insert(
                        input_pipe_index_1,
                        create_tensor(input_pipe_tensor_1, io_dev_id, false),
                    );
                    io_dev_id += 1;
                }

                if !(*subgraph)
                    .io_tensors
                    .tensors
                    .contains_key(&output_pipe_index)
                {
                    (*subgraph).io_tensors.tensors.insert(
                        output_pipe_index,
                        create_tensor(output_pipe_tensor, io_dev_id, true),
                    );
                    io_dev_id += 1;
                }

                operation.inputs = vec![input_pipe_index_0, input_pipe_index_1];
                operation.outputs = vec![output_pipe_index];
                (*subgraph).operations.push(operation);
            } else if (*poperation).type_
                == pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_CONVOLUTION
            {
                operation.type_ = (*poperation).type_;

                let input_pipe_tensor = *((*poperation).input_tensors).wrapping_add(0);
                let input_pipe_index = (*input_pipe_tensor).index;
                let weight_pipe_tensor = (*poperation).__bindgen_anon_1.conv.weight_tensor;
                let weight_pipe_index = (*weight_pipe_tensor).index;
                let bias_pipe_tensor = (*poperation).__bindgen_anon_1.conv.bias_tensor;
                let bias_pipe_index = (*bias_pipe_tensor).index;
                let output_pipe_tensor = *((*poperation).output_tensors).wrapping_add(0);
                let output_pipe_index = (*output_pipe_tensor).index;

                if !(*subgraph).weights.tensors.contains_key(&weight_pipe_index) {
                    (*subgraph).weights.tensors.insert(
                        weight_pipe_index,
                        create_tensor(weight_pipe_tensor, weight_dev_id, false),
                    );
                    weight_dev_id += 1;
                }

                if !(*subgraph).biases.tensors.contains_key(&bias_pipe_index) {
                    let output_channels = (*output_pipe_tensor).dims[3];
                    (*subgraph).biases.tensors.insert(
                        bias_pipe_index,
                        create_bias_tensor(bias_pipe_tensor, bias_dev_id, output_channels),
                    );
                    bias_dev_id += 1;
                }

                if !(*subgraph)
                    .io_tensors
                    .tensors
                    .contains_key(&input_pipe_index)
                {
                    (*subgraph).io_tensors.tensors.insert(
                        input_pipe_index,
                        create_tensor(input_pipe_tensor, io_dev_id, false),
                    );
                    io_dev_id += 1;
                }

                if !(*subgraph)
                    .io_tensors
                    .tensors
                    .contains_key(&output_pipe_index)
                {
                    (*subgraph).io_tensors.tensors.insert(
                        output_pipe_index,
                        create_tensor(output_pipe_tensor, io_dev_id, true),
                    );
                    io_dev_id += 1;
                }

                operation.inputs = vec![input_pipe_index];
                operation.outputs = vec![output_pipe_index];
                operation.weight = weight_pipe_index;
                operation.bias = bias_pipe_index;
                operation.stride_x = (*poperation).__bindgen_anon_1.conv.stride_x;
                operation.stride_y = (*poperation).__bindgen_anon_1.conv.stride_y;
                operation.has_padding = (*poperation).__bindgen_anon_1.conv.padding_top > 0
                    || (*poperation).__bindgen_anon_1.conv.padding_bottom > 0
                    || (*poperation).__bindgen_anon_1.conv.padding_left > 0
                    || (*poperation).__bindgen_anon_1.conv.padding_right > 0;
                (*subgraph).operations.push(operation);
            } else {
                eqaic_eprintln!("Unsupported pipe operation!");
            }
        }
    }

    subgraph.weights.num = subgraph.weights.tensors.len();
    subgraph.biases.num = subgraph.biases.tensors.len();
    subgraph.io_tensors.num = subgraph.io_tensors.tensors.len();

    let (operator, elf_data, exe_name) = jitdev.compile_subgraph(&subgraph).unwrap();
    (*subgraph).elf_data = elf_data;
    (*subgraph).exe_name = exe_name;
    // JIT operators must be aligned to 16MB; reserve extra space for the live RW copy
    let rw_extra = align_up((*subgraph).elf_data.rw_segment_size, 4096) as usize;
    (*subgraph).alloc_base = jitdev.alloc_cursor();
    let operator_addr = jitdev
        .alloc(operator.len() + rw_extra, 16 * 1024 * 1024)
        .expect("Could not allocate space for operators.");
    subgraph.operator_addr = operator_addr;
    jitdev
        .write_buffer(operator_addr, &operator, operator.len() as u64)
        .unwrap();

    alloc_tensor_list(pcontext, &mut subgraph.weights);
    populate_tensor_list(pcontext, &mut subgraph.weights);

    alloc_tensor_list(pcontext, &mut subgraph.biases);
    populate_tensor_list(pcontext, &mut subgraph.biases);

    // IO tensors will be populated when the subgraph is invoked
    alloc_tensor_list(pcontext, &mut subgraph.io_tensors);

    unsafe { &mut (*Box::into_raw(subgraph)).base }
}

pub extern "C" fn earthqaic_ml_subgraph_invoke(
    pcontext: *mut pipe_context,
    psubgraph: *mut pipe_ml_subgraph,
    inputs_count: u32,
    input_idxs: *mut u32,
    inputs: *mut *mut c_void,
    _is_signed: *mut bool,
) {
    let subgraph: &mut EarthqaicMLSubgraph =
        unsafe { &mut *(psubgraph as *mut EarthqaicMLSubgraph) };
    let jitdev = get_jitdev!(pcontext);

    let io_tensors = &mut subgraph.io_tensors.tensors;

    // Update each input tensor's data pointer
    for i in 0..inputs_count {
        let index = unsafe { *input_idxs.wrapping_add(i as usize) };
        if let Some(tensor) = io_tensors.get_mut(&index) {
            tensor.data = unsafe { *inputs.wrapping_add(i as usize) as *mut u8 };
        }
    }

    populate_tensor_list(pcontext, &mut (*subgraph).io_tensors);

    // Invoke the subgraph!
    let nsp_count = get_earthqaic_screen!((*pcontext).screen)
        .network
        .metadata
        .nsp_count;
    let _ = jitdev
        .execute_operator(
            subgraph.operator_addr,
            &subgraph.serialize_device_buffers(),
            &subgraph.elf_data,
            subgraph.exe_name.clone(),
            1, // TODO: thread_bitmap, will need to crack it up at some point
            nsp_count,
        )
        .unwrap();
}

pub unsafe extern "C" fn earthqaic_ml_subgraph_read_output(
    pcontext: *mut pipe_context,
    psubgraph: *mut pipe_ml_subgraph,
    outputs_count: u32,
    output_idxs: *mut u32,
    outputs: *mut *mut c_void,
    _is_signed: *mut bool,
) {
    let jitdev = get_jitdev!(pcontext);
    let subgraph: *mut EarthqaicMLSubgraph = psubgraph as *mut EarthqaicMLSubgraph;

    for i in 0..outputs_count {
        let index = *output_idxs.wrapping_add(i as usize);
        let tensor = (*subgraph).io_tensors.tensors.get(&index).unwrap();
        let size = tensor.size as usize;
        let buf_idx = tensor.dev_id;

        // Ensure the buf_idx points to an output tensor
        assert_eq!(
            (*subgraph)
                .io_tensors
                .tensors
                .get(&index)
                .unwrap()
                .is_output,
            true
        );

        let addr = (&(*subgraph).io_tensors.buf_tensors)[buf_idx as usize].addr;
        let buf = jitdev.read_buffer(addr, size as u64).unwrap();
        let output_ptr = *outputs.wrapping_add(i as usize) as *mut u8;
        for j in 0..size {
            *output_ptr.wrapping_add(j as usize) = buf[j];
        }
    }
}

pub unsafe extern "C" fn earthqaic_ml_subgraph_destroy(
    _ml_device: *mut pipe_ml_device,
    psubgraph: *mut pipe_ml_subgraph,
) {
    let subgraph: *mut EarthqaicMLSubgraph = psubgraph as *mut EarthqaicMLSubgraph;
    let _ = drop(Box::from_raw(subgraph));
}
