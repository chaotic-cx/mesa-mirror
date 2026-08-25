/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::ffi::c_void;

use crate::earthqaic_device::*;
use crate::earthqaic_rs_bindings::*;
use crate::get_earthqaic_screen;
use crate::jit::device::*;

// Worker threads are 0-4; thread 5 is the master
const THREADS_PER_NSP: u32 = 5;
const THREAD_BITMAP_PER_NSP: u32 = 0x1f;

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

#[repr(C)]
#[derive(Default)]
struct EarthqaicFence {
    reference: pipe_reference,
    val: u32,
}

pub extern "C" fn earthqaic_create_compute_state(
    pctx: *mut pipe_context,
    cstate: *const pipe_compute_state,
) -> *mut c_void {
    let jitdev = get_jitdev!(pctx);

    let nir = unsafe { (*cstate).prog as *mut nir_shader };

    let base = pipe_shader_state {
        type_: pipe_shader_ir_PIPE_SHADER_IR_NIR,
        ir: pipe_shader_state__bindgen_ty_1 { nir: nir },
        ..Default::default()
    };

    let mut estate = Box::new(EarthqaicComputeShader {
        base: base,
        alloc_base: u64::MAX,
        ..Default::default()
    });

    let (operator, elf_data, exe_name) = jitdev.compile_nir(&estate).unwrap();
    estate.operator = operator;
    estate.elf_data = elf_data;
    estate.exe_name = exe_name;

    // If shader compiles successfuly, send it to the device
    if estate.operator.len() > 0 {
        let jitdev = get_jitdev!(pctx);
        estate.alloc_base = jitdev.alloc_cursor();
        let rw_extra = align_up(estate.elf_data.rw_segment_size, 4096) as usize;
        let operator_addr = jitdev
            .alloc(estate.operator.len() + rw_extra, 16 * 1024 * 1024)
            .expect("Could not allocate space for kernel!");

        estate.operator_buf.addr = operator_addr;
        estate.operator_buf.size = estate.operator.len();

        jitdev
            .write_buffer(
                estate.operator_buf.addr,
                &estate.operator,
                estate.operator_buf.size as u64,
            )
            .expect("Could not write operator to DDR");
    }

    Box::into_raw(estate) as *mut c_void
}

pub unsafe extern "C" fn earthqaic_bind_compute_state(pctx: *mut pipe_context, data: *mut c_void) {
    let ectx = pctx as *mut EarthqaicContext;
    let shader = data as *mut EarthqaicComputeShader;
    (*ectx).shader = shader;
}

pub unsafe extern "C" fn earthqaic_delete_compute_state(
    _pctx: *mut pipe_context,
    data: *mut c_void,
) {
    let shader = data as *mut EarthqaicComputeShader;

    // TODO: free memory once we have a proper allocator
    let _ = Box::from_raw(shader);
}

pub unsafe extern "C" fn earthqaic_get_compute_state_info(
    _pctx: *mut pipe_context,
    data: *mut c_void,
    info: *mut pipe_compute_state_object_info,
) {
    let _estate = data as *mut EarthqaicComputeShader;
    (*info).max_threads = 1;
}

pub unsafe extern "C" fn earthqaic_get_compute_state_subgroup_size(
    _pctx: *mut pipe_context,
    _data: *mut c_void,
    _block: *const u32,
) -> u32 {
    0
}

pub unsafe extern "C" fn earthqaic_set_global_binding(
    pctx: *mut pipe_context,
    first: u32,
    count: u32,
    presources: *mut *mut pipe_resource,
    handles: *mut *mut u32,
) {
    let ectx = pctx as *mut EarthqaicContext;
    let shader = (*ectx).shader;
    let jitdev = get_jitdev!(pctx);

    // If NIR cant be compiled, theres nothing to bind
    if (*shader).operator.len() == 0 {
        return;
    }

    if presources.is_null() {
        let start = first as usize;
        let end = (first + count) as usize;
        (*shader).global_buffers.drain(start..end);
        (*shader).global_resources.drain(start..end);
        return;
    }

    /*
     * Last threads may go beyond buffer allocation due to how we handle dispatch.
     * Adding enough padding here should handle those cases.
     * Worst case: (nsp_count * THREADS_PER_NSP - 1) float elements beyond the end.
     */
    let nsp_count = get_earthqaic_screen!((*pctx).screen)
        .network
        .metadata
        .nsp_count as usize;
    // TODO: 8 comes from sizeof double, 4 causes oob randomly
    let padding = (nsp_count * THREADS_PER_NSP as usize - 1) * 8;

    for i in 0..count {
        let prsc: *mut pipe_resource = *(presources.wrapping_add((first + i) as usize));
        let prsc_size = (*prsc).width0 as usize;
        let prsc_jit_addr = jitdev
            .alloc(prsc_size + padding, 8)
            .expect("Could not allocate space for kernel!");
        (*shader).global_buffers.push(JitBuffer {
            addr: prsc_jit_addr,
            size: prsc_size,
        });
        (*shader)
            .global_resources
            .push(prsc as *mut EarthqaicResource);

        /*
         * TODO: This handle cannot be passed opaquely to the operator
         * running on the device as it represents a physical address.
         * The qshim will map this buffer into the operator's address
         * space and give it a unique VA.
         */
        let handle_ptr = *(handles.wrapping_add((first + i) as usize));
        let lsb = prsc_jit_addr & 0xffffffff;
        *handle_ptr = lsb as u32;
        // If we need 64b:
        // let hsb = prsc_jit_addr >> 32;
        // *(handle_ptr.wrapping_add(1)) = hsb as u32;
    }
}

pub unsafe extern "C" fn earthqaic_set_constant_buffer(
    pctx: *mut pipe_context,
    _shader: i32,
    _index: u32,
    cb: *const pipe_constant_buffer,
) {
    let ectx = pctx as *mut EarthqaicContext;
    let shader = (*ectx).shader;
    let jitdev = get_jitdev!(pctx);

    // If NIR cant be compiled, theres nothing to bind
    if (*shader).operator.len() == 0 {
        return;
    }

    if !cb.is_null() {
        let size = (*cb).buffer_size as usize;
        let ptr = (*cb).user_buffer.wrapping_add((*cb).buffer_offset as usize) as *const u8;
        let buf_addr = jitdev
            .alloc(size, 8)
            .expect("Could not allocate space for kernel!");
        (*shader).constant_buffer = JitBuffer {
            addr: buf_addr,
            size: size,
        };

        let mut vec: Vec<u8> = Vec::new();
        for i in 0..size {
            vec.push(*(ptr.wrapping_add(i)));
        }
        (*shader).constant_resource = vec;
    }
    // Cleanup
    else {
        (*shader).constant_buffer = JitBuffer::default();
        (*shader).constant_resource.clear();
    }
}

pub unsafe extern "C" fn earthqaic_launch_grid(
    pctx: *mut pipe_context,
    info: *const pipe_grid_info,
) {
    let jitdev = get_jitdev!(pctx);

    let econtext = pctx as *mut EarthqaicContext;
    let shader = (*econtext).shader as *const EarthqaicComputeShader;
    let nsp_count = get_earthqaic_screen!((*pctx).screen)
        .network
        .metadata
        .nsp_count;

    let work_items = (*info).block[0] * (*info).grid[0];

    // do a ceil division so that we don't miss any work_items
    // overshoots now handled by padding during allocation
    let total_per_dispatch = THREADS_PER_NSP * nsp_count;
    let num_groups = (work_items + total_per_dispatch - 1) / total_per_dispatch;
    eqaic_debugln!(
        "launch_grid cores={}, block={} grid={} work_items={} groups={}",
        nsp_count,
        (*info).block[0],
        (*info).grid[0],
        work_items,
        num_groups,
    );

    // If NIR cant be compiled, theres nothing to execute
    if (*shader).operator.len() == 0 {
        return;
    }

    // Serialized device buffers
    let mut entries: Vec<JitBuffer> = Vec::new();

    // Write global buffers
    for i in 0..(*shader).global_buffers.len() {
        let prsc: *mut EarthqaicResource = (&((*shader).global_resources))[i];
        let buf: &JitBuffer = &(&(*shader).global_buffers)[i];
        jitdev
            .write_resource(buf.addr, prsc, buf.size as u64)
            .unwrap();
        entries.push(*buf);
    }

    // Write constant buffer
    jitdev
        .write_buffer(
            (*shader).constant_buffer.addr,
            &(*shader).constant_resource,
            (*shader).constant_buffer.size as u64,
        )
        .expect("Could not write constant buffer");
    entries.push((*shader).constant_buffer);

    // Allocate space for the work item ID (int)
    // This will need to be updated each iteration
    let wg_id_addr = jitdev
        .alloc(4, 8)
        .expect("Couldnt allocate space for work ID");
    let wg_id_buf = JitBuffer {
        addr: wg_id_addr,
        size: 4,
    };
    entries.push(wg_id_buf);

    for i in 0..num_groups {
        let wg_id_bytes = i.to_le_bytes().to_vec();
        assert!(wg_id_bytes.len() == 4);
        jitdev
            .write_buffer(wg_id_addr, &wg_id_bytes, 4)
            .expect("Could not write work group ID");

        if let Err(e) = jitdev.execute_operator(
            (*shader).operator_buf.addr,
            &entries,
            &(*shader).elf_data,
            (*shader).exe_name.clone(),
            THREAD_BITMAP_PER_NSP,
            nsp_count,
        ) {
            eqaic_eprintln!("launch_grid: execute_operator failed: {}", e);
            return;
        }
    }

    // Read global buffers
    for i in 0..(*shader).global_buffers.len() {
        let ersc: *mut EarthqaicResource = (&((*shader).global_resources))[i];
        let jitbuf: &JitBuffer = &(&(*shader).global_buffers)[i];
        // Why cant we read directly into the pipe_resource?
        // jitdev.read_resource(buf.addr, ersc, buf.size).unwrap();
        let buf = jitdev.read_buffer(jitbuf.addr, jitbuf.size as u64).unwrap();
        pipe_buffer_write(
            pctx,
            ersc as *mut pipe_resource,
            0,
            jitbuf.size as u32,
            buf.as_ptr() as *const c_void,
        );
    }
}

pub unsafe extern "C" fn earthqaic_fence_finish(
    _pscreen: *mut pipe_screen,
    _pctx: *mut pipe_context,
    fence_handle: *mut pipe_fence_handle,
    _timeout: u64,
) -> bool {
    let _efence = fence_handle as *mut EarthqaicFence;
    true
}

pub unsafe extern "C" fn earthqaic_fence_reference(
    _pscreen: *mut pipe_screen,
    ptr: *mut *mut pipe_fence_handle,
    fence: *mut pipe_fence_handle,
) {
    let _old = ptr as *mut *mut EarthqaicFence;
    let _f = fence as *mut EarthqaicFence;
}

pub unsafe extern "C" fn earthqaic_get_name(pscreen: *mut pipe_screen) -> *const i8 {
    let escreen: &EarthqaicScreen = get_earthqaic_screen!(pscreen);
    (*escreen).name.as_ptr() as *const i8
}

pub unsafe extern "C" fn earthqaic_get_vendor(pscreen: *mut pipe_screen) -> *const i8 {
    let escreen: &EarthqaicScreen = get_earthqaic_screen!(pscreen);
    (*escreen).vendor.as_ptr() as *const i8
}

pub unsafe extern "C" fn earthqaic_is_format_supported(
    _pscreen: *mut pipe_screen,
    _format: u32,
    _target: u32,
    _sample_count: u32,
    _storage_sample: u32,
    _bind: u32,
) -> bool {
    true
}

pub unsafe extern "C" fn earthqaic_bind_sampler_states(
    _pctx: *mut pipe_context,
    _shader: i32,
    _start: u32,
    _num: u32,
    _samplers: *mut *mut c_void,
) {
}

pub unsafe extern "C" fn earthqaic_create_query(
    _pctx: *mut pipe_context,
    _type_: u32,
    _index: u32,
) -> *mut pipe_query {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn earthqaic_delete_sampler_state(
    _pctx: *mut pipe_context,
    _sampler: *mut c_void,
) {
}

pub unsafe extern "C" fn earthqaic_destroy_query(
    _pctx: *mut pipe_context,
    _pquery: *mut pipe_query,
) {
}

pub unsafe extern "C" fn earthqaic_end_query(
    _pctx: *mut pipe_context,
    _pquery: *mut pipe_query,
) -> bool {
    true
}

pub unsafe extern "C" fn earthqaic_flush(
    _pctx: *mut pipe_context,
    fence: *mut *mut pipe_fence_handle,
    _flags: u32,
) {
    // TODO: Check that fence pointer is non null
    if !fence.is_null() {
        *fence = Box::into_raw(Box::new(EarthqaicFence::default())) as *mut pipe_fence_handle;
    }
}

pub unsafe extern "C" fn earthqaic_memory_barrier(_pctx: *mut pipe_context, _flags: u32) {}

pub unsafe extern "C" fn earthqaic_sampler_view_destroy(
    _pctx: *mut pipe_context,
    _view: *mut pipe_sampler_view,
) {
}

pub unsafe extern "C" fn earthqaic_set_sampler_views(
    _pctx: *mut pipe_context,
    _shader: i32,
    _start: u32,
    _num: u32,
    _unbind_num_trailing_slots: u32,
    _views: *mut *mut pipe_sampler_view,
) {
}

pub unsafe extern "C" fn earthqaic_sampler_view_release(
    _pctx: *mut pipe_context,
    _view: *mut pipe_sampler_view,
) {
}

pub unsafe extern "C" fn earthqaic_set_shader_images(
    _pctx: *mut pipe_context,
    _shader: i32,
    _start_slot: u32,
    _count: u32,
    _unbind_num_trailing_slots: u32,
    _images: *const pipe_image_view,
) {
}

pub unsafe extern "C" fn earthqaic_texture_map(
    _pctx: *mut pipe_context,
    _prsc: *mut pipe_resource,
    _level: u32,
    _usage: u32,
    _box_: *const pipe_box,
    _ptransfer: *mut *mut pipe_transfer,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn earthqaic_texture_subdata(
    _pctx: *mut pipe_context,
    _prsc: *mut pipe_resource,
    _level: u32,
    _usage: u32,
    _box_: *const pipe_box,
    _data: *const c_void,
    _stride: u32,
    _layer_stride: usize,
) {
}

pub unsafe extern "C" fn earthqaic_texture_unmap(
    _pctx: *mut pipe_context,
    _ptransfer: *mut pipe_transfer,
) {
}
