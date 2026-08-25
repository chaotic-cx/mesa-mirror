/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::ffi::{c_void, CString};
use std::mem::offset_of;

use crate::earthqaic_rs_bindings::*;
use crate::qaic;
use crate::qaic::qaic_rs_bindings::*;

#[repr(C)]
pub struct EarthqaicScreen {
    pub pscreen: pipe_screen,
    pub name: CString,
    pub vendor: CString,
    pub fd: i32,
    pub network: qaic::network::Network,
}

#[repr(C)]
pub struct EarthqaicContext {
    pub base: pipe_context,
}

#[repr(C)]
pub struct EarthqaicResource {
    base: pipe_resource,
    pub handle: u32,
}

#[repr(C)]
struct EarthqaicPipeTransfer {
    base: pipe_transfer,
    mapping: *mut c_void,
}

#[macro_export]
macro_rules! get_earthqaic_screen {
    ($pscreen:expr) => {
        unsafe { &mut *($pscreen as *const pipe_screen as *mut EarthqaicScreen) }
    };
}

#[macro_export]
macro_rules! get_earthqaic_resource {
    ($presource:expr) => {
        unsafe { &*($presource as *const pipe_resource as *const EarthqaicResource) }
    };
}

macro_rules! ectx_to_pctx {
    ($ectx:expr) => {{
        unsafe {
            Box::into_raw($ectx).byte_add(offset_of!(EarthqaicContext, base)) as *mut pipe_context
        }
    }};
}

/*
 * The pipe_transfer C struct has a 24b bitfield for usage and an 8b one for level.
 * In the generated rust struct pipe_transfer, this is represented as a single array
 * of 4 u8 values [u8; 4] (I think).
 */
fn set_pipe_transfer_level_usage(transfer: &mut pipe_transfer, level: u32, usage: u32) {
    transfer._bitfield_1.set(0, 24, usage as u64);
    transfer._bitfield_1.set(24, 8, level as u64);
}

#[no_mangle]
pub extern "C" fn c_earthqaic_buffer_map(
    pctx: *mut pipe_context,
    prsc: *mut pipe_resource,
    level: u32,
    usage: u32,
    pbox: *const pipe_box,
    out_transfer: *mut *mut pipe_transfer,
) -> *mut c_void {
    let ersc: &EarthqaicResource = get_earthqaic_resource!(prsc);
    let fd = get_earthqaic_screen!((*pctx).screen).fd;

    let mut transfer = Box::new(EarthqaicPipeTransfer {
        mapping: std::ptr::null_mut(),
        base: pipe_transfer {
            ..Default::default()
        },
    });

    let mut arg = qaic_mmap_bo {
        ..Default::default()
    };
    arg.handle = (*ersc).handle;
    let arg_ptr: *mut c_void = &arg as *const qaic_mmap_bo as *mut c_void;
    let ret = unsafe { drmIoctl(fd, _DRM_IOCTL_QAIC_MMAP_BO, arg_ptr) };
    if ret != 0 {
        eqaic_eprintln!("MMAP BO ioctl failed!\n");
        return std::ptr::null_mut();
    }

    let map: *mut c_void = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            (*prsc).width0 as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            arg.offset as i64,
        )
    };
    if map == libc::MAP_FAILED {
        eqaic_eprintln!("Mapping BO failed!\n");
        return std::ptr::null_mut();
    }

    (*transfer).mapping = map;
    set_pipe_transfer_level_usage(&mut (*transfer).base, level, usage);
    (*transfer).base.box_ = unsafe { *pbox };

    let offset;
    unsafe {
        let raw_ptr: *mut pipe_transfer = &mut (*Box::into_raw(transfer)).base;
        let raw_rsc_ptr: *mut *mut pipe_resource = &mut (*raw_ptr).resource;
        pipe_resource_reference(raw_rsc_ptr, prsc);
        *out_transfer = raw_ptr;
        offset = (*pbox).x as usize;
    }

    map.wrapping_add(offset)
}

#[no_mangle]
pub extern "C" fn c_earthqaic_buffer_unmap(
    _pctx: *mut pipe_context,
    ptransfer: *mut pipe_transfer,
) {
    unsafe {
        let qtransfer = ptransfer as *mut EarthqaicPipeTransfer;
        let resource_ptr = (*ptransfer).resource;
        let size = (*resource_ptr).width0 as usize;
        libc::munmap((*qtransfer).mapping, size);
        pipe_resource_reference(
            &mut (*ptransfer).resource as *mut *mut pipe_resource,
            std::ptr::null_mut(),
        );
        drop(Box::from_raw(qtransfer));
    }
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
    pscreen: *mut pipe_screen,
    presource: *mut pipe_resource,
) {
    let escreen: &EarthqaicScreen = get_earthqaic_screen!(pscreen);
    let ersc: &EarthqaicResource = get_earthqaic_resource!(presource);
    let mut arg = drm_gem_close {
        ..Default::default()
    };
    arg.handle = (*ersc).handle;
    let arg_ptr: *mut c_void = &arg as *const drm_gem_close as *mut c_void;
    let ret = unsafe { drmIoctl((*escreen).fd, _DRM_IOCTL_GEM_CLOSE, arg_ptr) };
    if ret != 0 {
        eqaic_eprintln!("Close ioctl failed!\n");
    }
    let ersc = unsafe { Box::from_raw(presource as *mut EarthqaicResource) };
    drop(ersc);
}

#[no_mangle]
pub extern "C" fn c_earthqaic_resource_create(
    pscreen: *mut pipe_screen,
    template: *const pipe_resource,
) -> *mut pipe_resource {
    let escreen: &EarthqaicScreen = get_earthqaic_screen!(pscreen);

    let fd = (*escreen).fd;

    let mut arg = qaic_create_bo {
        ..Default::default()
    };
    arg.size = unsafe { (*template).width0 as u64 };

    let arg_ptr: *mut c_void = &arg as *const qaic_create_bo as *mut c_void;
    let ret = unsafe { drmIoctl(fd, _DRM_IOCTL_QAIC_CREATE_BO, arg_ptr) };
    if ret != 0 {
        eqaic_eprintln!("Create BO ioctl failed!\n");
        return std::ptr::null_mut();
    }

    let mut base: pipe_resource = unsafe { *template };
    base.screen = pscreen;

    let mut resource = Box::new(EarthqaicResource {
        base: base,
        handle: arg.handle,
    });
    unsafe {
        pipe_reference_init(&mut (resource.base.__bindgen_anon_1.reference), 1);
    }

    unsafe { &mut (*Box::into_raw(resource)).base }
}

#[no_mangle]
pub extern "C" fn c_earthqaic_screen_destroy(pscreen: *mut pipe_screen) {
    let escreen = unsafe { Box::from_raw(pscreen as *mut EarthqaicScreen) };
    qaic::network::deactivate_and_unload_network(escreen.fd, &escreen.network, true);
    // The fd is dup'd by earthqaic_winsys.c before earthqaic_screen_create is
    // called, so this driver owns it and must close it here.
    unsafe {
        libc::close(escreen.fd);
    }
    drop(escreen);
}

#[no_mangle]
pub extern "C" fn c_earthqaic_get_screen_fd(pscreen: *mut pipe_screen) -> i32 {
    let escreen: &EarthqaicScreen = get_earthqaic_screen!(pscreen);
    (*escreen).fd
}

fn screen_create(fd: i32) -> Result<Box<EarthqaicScreen>, String> {
    let mut status_resp = nnc_cmd_status_query_response_t {
        ..Default::default()
    };
    let mut status_resp_trans = nnc_transaction_status_response {
        ..Default::default()
    };

    let mut ret = qaic::control::status_query(fd, &mut status_resp);
    if ret != 0 {
        return Err(format!("Could not query device status! Err: {}", ret));
    }
    ret = qaic::control::status_query_transaction(fd, &mut status_resp_trans);
    if ret != 0 {
        return Err(format!(
            "Could not query device status with transaction! Err: {}",
            ret
        ));
    }

    let dev_str;
    match qaic::control::is_valid_fw(&status_resp) {
        Some("AIC 100") => dev_str = c"Qualcomm AIC 100",
        _ => return Err("Invalid fw!".to_string()),
    }

    let pscreen = pipe_screen {
        get_screen_fd: Some(c_earthqaic_get_screen_fd),
        destroy: Some(c_earthqaic_screen_destroy),
        context_create: Some(c_earthqaic_context_create),
        resource_create: Some(c_earthqaic_resource_create),
        resource_destroy: Some(c_earthqaic_resource_destroy),
        ..Default::default()
    };

    let network = qaic::network::load_and_activate_network(fd, true).unwrap();

    let escreen = Box::new(EarthqaicScreen {
        name: dev_str.into(),
        vendor: c"Qualcomm".into(),
        pscreen: pscreen,
        fd: fd,
        network: network,
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
        Err(e) => {
            eqaic_eprintln!("{}", e);
            return std::ptr::null_mut();
        }
    }
}
