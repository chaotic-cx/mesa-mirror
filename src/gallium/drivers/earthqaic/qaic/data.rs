/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::ffi::c_void;

use crate::earthqaic_rs_bindings::drm_gem_close;
use crate::qaic::qaic_rs_bindings::*;

#[allow(unused)]
#[derive(Clone, Copy)]
pub enum DataDirection {
    DirUndef,
    DmaToDevice = 1,
    DmaFromDevice = 2,
}

#[derive(Default)]
pub struct BOParams {
    pub handle: u32,
    pub offset: u64,
    pub size: u64,
}

#[allow(unused)]
pub fn create_bo(fd: i32, size: u64, handle: &mut u32) -> i32 {
    let mut msg = qaic_create_bo {
        size: size,
        handle: 0,
        pad: 0,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            _DRM_IOCTL_QAIC_CREATE_BO as libc::c_ulong,
            &mut msg as *mut qaic_create_bo as *mut c_void,
        )
    };
    *handle = msg.handle;
    ret
}

#[allow(unused)]
pub fn close_bo(fd: i32, handle: u32) -> i32 {
    let mut msg = drm_gem_close {
        handle: handle,
        pad: 0,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            _DRM_IOCTL_GEM_CLOSE as libc::c_ulong,
            &mut msg as *mut drm_gem_close as *mut c_void,
        )
    };
    ret
}

#[allow(unused)]
pub fn mmap_bo(fd: i32, handle: u32, offset: &mut u64) -> i32 {
    let mut msg = qaic_mmap_bo {
        handle: handle,
        pad: 0,
        offset: 0,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            _DRM_IOCTL_QAIC_MMAP_BO as libc::c_ulong,
            &mut msg as *mut qaic_mmap_bo as *mut c_void,
        )
    };
    *offset = msg.offset;
    ret
}

// RAII guard for a BO created and mmap'd via create_and_mmap_bo: munmap + close_bo
// run automatically on drop, so every early return in a caller gets cleanup for free
// instead of repeating it at each failure site.
pub struct MappedBo {
    fd: i32,
    pub handle: u32,
    pub ptr: *mut c_void,
    size: u64,
}

impl MappedBo {
    pub fn new(fd: i32, size: u64) -> Option<MappedBo> {
        let mut bo_params = BOParams::default();
        let ptr = create_and_mmap_bo(fd, size, &mut bo_params);
        if ptr.is_null() {
            return None;
        }
        Some(MappedBo {
            fd,
            handle: bo_params.handle,
            ptr,
            size,
        })
    }
}

impl Drop for MappedBo {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.size as usize);
        }
        close_bo(self.fd, self.handle);
    }
}

#[allow(unused)]
pub fn create_and_mmap_bo(fd: i32, size: u64, bo_params: &mut BOParams) -> *mut c_void {
    let mut handle: u32 = 0;
    let mut offset: u64 = 0;

    let ret = create_bo(fd, size, &mut handle);
    if ret != 0 {
        eqaic_eprintln!("Couldnt create BO with size {}", size);
        return std::ptr::null_mut();
    }

    let ret = mmap_bo(fd, handle, &mut offset);
    if ret != 0 {
        eqaic_eprintln!("Couldnt mmap BO with handle {}", handle);
        close_bo(fd, handle);
        return std::ptr::null_mut();
    }

    let ret_buf = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            offset as i64,
        )
    };
    if ret_buf == libc::MAP_FAILED {
        eqaic_eprintln!("Couldnt mmap BO with handle {}", handle);
        close_bo(fd, handle);
        return std::ptr::null_mut();
    }

    (*bo_params).size = size;
    (*bo_params).handle = handle;
    (*bo_params).offset = offset;
    ret_buf
}

#[allow(unused)]
pub fn attach_slice_bo(
    fd: i32,
    count: u32,
    dbc_id: u32,
    handle: u32,
    dir: DataDirection,
    size: u64,
    slices: &mut qaic_attach_slice_entry,
) -> i32 {
    let mut msg = qaic_attach_slice {
        hdr: qaic_attach_slice_hdr {
            count: count,
            dbc_id: dbc_id,
            handle: handle,
            dir: dir as u32,
            size: size,
        },
        data: slices as *mut qaic_attach_slice_entry as u64,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            _DRM_IOCTL_QAIC_ATTACH_SLICE_BO as libc::c_ulong,
            &mut msg as *mut qaic_attach_slice as *mut c_void,
        )
    };
    ret
}

#[allow(unused)]
pub fn partial_execute_bo(
    fd: i32,
    count: u32,
    handle: u32,
    dbc_id: u32,
    entries: *mut qaic_partial_execute_entry,
) -> i32 {
    let mut msg = qaic_execute {
        hdr: qaic_execute_hdr {
            count: count,
            dbc_id: dbc_id,
        },
        data: entries as *mut qaic_partial_execute_entry as u64,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            _DRM_IOCTL_QAIC_PARTIAL_EXECUTE_BO as libc::c_ulong,
            &mut msg as *mut qaic_execute as *mut c_void,
        )
    };
    ret
}

#[allow(unused)]
pub fn wait_bo(fd: i32, handle: u32, timeout: u32, dbc_id: u32) -> i32 {
    let mut msg = qaic_wait {
        handle: handle,
        timeout: timeout,
        dbc_id: dbc_id,
        pad: 0,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            _DRM_IOCTL_QAIC_WAIT_BO as libc::c_ulong,
            &mut msg as *mut qaic_wait as *mut c_void,
        )
    };
    ret
}

#[allow(unused)]
pub fn detach_slice_bo(fd: i32, handle: u32) -> i32 {
    let mut msg = qaic_detach_slice {
        handle: handle,
        pad: 0,
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            _DRM_IOCTL_QAIC_DETACH_SLICE_BO as libc::c_ulong,
            &mut msg as *mut qaic_detach_slice as *mut c_void,
        )
    };
    ret
}
