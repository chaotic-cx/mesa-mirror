/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::ffi::{c_void, CString};
use std::mem::offset_of;

use crate::earthqaic_rs_bindings::*;

#[repr(C)]
pub struct EarthqaicScreen {
    pub pscreen: pipe_screen,
    pub name: CString,
    pub vendor: CString,
    pub fd: i32,
}

#[repr(C)]
pub struct EarthqaicContext {
    pub base: pipe_context,
}

#[macro_export]
macro_rules! get_earthqaic_screen {
    ($pscreen:expr) => {
        unsafe { &mut *($pscreen as *const pipe_screen as *mut EarthqaicScreen) }
    };
}

macro_rules! ectx_to_pctx {
    ($ectx:expr) => {{
        unsafe {
            Box::into_raw($ectx).byte_add(offset_of!(EarthqaicContext, base)) as *mut pipe_context
        }
    }};
}

#[no_mangle]
pub extern "C" fn c_earthqaic_buffer_map(
    _pctx: *mut pipe_context,
    _prsc: *mut pipe_resource,
    _level: u32,
    _usage: u32,
    _pbox: *const pipe_box,
    _out_transfer: *mut *mut pipe_transfer,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn c_earthqaic_buffer_unmap(
    _pctx: *mut pipe_context,
    _ptransfer: *mut pipe_transfer,
) {
}

#[no_mangle]
pub extern "C" fn c_earthqaic_context_destroy(pcontext: *mut pipe_context) {
    let ectx = pcontext as *mut EarthqaicContext;
    let _ = unsafe { drop(Box::from_raw(ectx)) };
}

#[no_mangle]
pub extern "C" fn c_earthqaic_context_create(
    pscreen: *mut pipe_screen,
    private: *mut c_void,
    _flags: u32,
) -> *mut pipe_context {
    let ectx: Box<EarthqaicContext> = Box::new(EarthqaicContext {
        base: pipe_context {
            screen: pscreen,
            priv_: private,
            destroy: Some(c_earthqaic_context_destroy),
            buffer_map: Some(c_earthqaic_buffer_map),
            buffer_unmap: Some(c_earthqaic_buffer_unmap),
            resource_copy_region: Some(util_resource_copy_region),
            buffer_subdata: Some(u_default_buffer_subdata),
            clear_buffer: Some(u_default_clear_buffer),
            ..Default::default()
        },
    });

    ectx_to_pctx!(ectx)
}

#[no_mangle]
pub extern "C" fn c_earthqaic_resource_destroy(
    _pscreen: *mut pipe_screen,
    _presource: *mut pipe_resource,
) {
}

#[no_mangle]
pub extern "C" fn c_earthqaic_resource_create(
    _pscreen: *mut pipe_screen,
    _template: *const pipe_resource,
) -> *mut pipe_resource {
    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn c_earthqaic_screen_destroy(pscreen: *mut pipe_screen) {
    let escreen = unsafe { Box::from_raw(pscreen as *mut EarthqaicScreen) };
    drop(escreen);
}

#[no_mangle]
pub extern "C" fn c_earthqaic_get_screen_fd(pscreen: *mut pipe_screen) -> i32 {
    let escreen: &EarthqaicScreen = get_earthqaic_screen!(pscreen);
    (*escreen).fd
}

fn screen_create(fd: i32) -> Result<Box<EarthqaicScreen>, String> {
    let pscreen = pipe_screen {
        get_screen_fd: Some(c_earthqaic_get_screen_fd),
        destroy: Some(c_earthqaic_screen_destroy),
        context_create: Some(c_earthqaic_context_create),
        resource_create: Some(c_earthqaic_resource_create),
        resource_destroy: Some(c_earthqaic_resource_destroy),
        ..Default::default()
    };

    let escreen = Box::new(EarthqaicScreen {
        name: c"Qualcomm AI 100".into(),
        vendor: c"Qualcomm".into(),
        pscreen: pscreen,
        fd: fd,
    });

    Ok(escreen)
}

#[no_mangle]
pub unsafe extern "C" fn earthqaic_screen_create(
    fd: i32,
    _config: *const pipe_screen_config,
    _ro: *mut renderonly,
) -> *mut pipe_screen {
    match screen_create(fd) {
        Ok(escreen) => {
            let ptr = Box::into_raw(escreen);
            return &mut (*ptr).pscreen;
        }
        Err(_e) => return std::ptr::null_mut(),
    }
}
