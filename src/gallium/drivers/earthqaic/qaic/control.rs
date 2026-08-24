/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::ffi::c_void;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;

use crate::qaic::qaic_rs_bindings::*;

const NNC_CONSTANTS_TAG: u32 = 300;
const NNC_IMAGE_TAG: u32 = 100;
const VCQUEUESIZE16K: u32 = 0x4000;

#[allow(unused)]
enum NncDmaOptionType {
    NncDmaOptionShared = 0,
    NncDmaOptionDedicatedPrefer = 1,
    NncDmaOptionDedicated = 2,
}

struct ActivateParams {
    elf_id: u32,
    constants_id: u32,
    res_type: nnc_reservation_type_t,
    resource_res_id: i32,
    opt_vc_setup: u32,
    vc_res_id: i32,
    init_state: nnc_activation_state_type_t,
    dma_type: u32,
    support_datapath_switching: u8,
}

// Return a nnc_transaction_passthrough_t reference pointing to msg.data
macro_rules! init_passthrough_header {
    ($msg:expr, $t:ty) => {{
        let pt_trans =
            unsafe { &mut *($msg.data.as_mut_ptr() as *mut nnc_transaction_passthrough_t) };
        pt_trans.header.type_ = nnc_transaction_type_NNC_TRANSACTION_TYPE_PASSTHROUGH_UK as i32;
        pt_trans.header.len = (std::mem::size_of::<nnc_transaction_passthrough_t>()
            + std::mem::size_of::<$t>()) as u32;
        pt_trans // return the reference
    }};
}

// Return a reference of NncCmd_ type ty that points to pt_trans.cmd
macro_rules! reference_passthrough_cmd {
    ($pt_trans:expr, $t:ty) => {{
        unsafe { &mut *(&mut $pt_trans.cmd as *mut nnc_cmd_t as *mut $t) }
    }};
}

// Return a reference of type T pointing to msg.data with offset pt_trans.header.len
macro_rules! reference_msg_data_plus_pt_hdr {
    ($msg:expr, $pt_trans:expr, $t:ty) => {{
        unsafe { &mut *($msg.data.as_mut_ptr().add((*$pt_trans).header.len as usize) as *mut $t) }
    }};
}

// Return a nnc_transaction_vc_setup_uk_t reference embedded in msg.data with offset pt_trans.header.len
macro_rules! init_trans_vc_setup {
    ($msg:expr, $pt_trans:expr) => {{
        let trans_vc_set =
            reference_msg_data_plus_pt_hdr!($msg, $pt_trans, nnc_transaction_vc_setup_uk_t);
        trans_vc_set.header.type_ = nnc_transaction_type_NNC_TRANSACTION_TYPE_SETUP_VC_UK as i32;
        trans_vc_set.header.len = size_of::<nnc_transaction_vc_setup_uk_t>() as u32;
        trans_vc_set
    }};
}

// Return a nnc_transaction_vc_teardown_uk_t reference embedded in msg.data with offset pt_trans.header.len
macro_rules! init_trans_vc_teardown {
    ($msg:expr, $pt_trans:expr) => {{
        let trans_vc_set =
            reference_msg_data_plus_pt_hdr!($msg, $pt_trans, nnc_transaction_vc_teardown_uk_t);
        trans_vc_set.header.type_ = nnc_transaction_type_NNC_TRANSACTION_TYPE_TEARDOWN_VC_UK as i32;
        trans_vc_set.header.len = size_of::<nnc_transaction_vc_teardown_uk_t>() as u32;
        trans_vc_set
    }};
}

// Return a nnc_transaction_dma_xfer_t reference embedded in msg.data with offset pt_trans.header.len
macro_rules! init_dma_xfer {
    ($msg:expr, $pt_trans:expr) => {{
        let trans_dma_xfer =
            reference_msg_data_plus_pt_hdr!($msg, $pt_trans, nnc_transaction_dma_xfer_t);
        trans_dma_xfer.header.type_ = nnc_transaction_type_NNC_TRANSACTION_TYPE_DMA_XFER_UK as i32;
        trans_dma_xfer.header.len = (std::mem::size_of::<nnc_transaction_dma_xfer_t>()
            + std::mem::size_of::<addr_len_pair_t>()) as u32;
        trans_dma_xfer
    }};
}

// Set address and len pair within the dma transfer
macro_rules! set_alp {
    ($trans_dma_xfer:expr, $buf:expr) => {{
        let alp_ptr = unsafe { &mut *($trans_dma_xfer.alp.as_mut_ptr()) };
        alp_ptr.addr = $buf.as_mut_ptr() as u64;
        alp_ptr.len = $buf.len() as u64;
    }};
}

// Set msg.header.{length,count}, invoke the QAIC manage ioctl, and on
// success copy the passthrough response out of pt_trans into resp. On
// failure, print err_msg and return the ioctl error code, leaving resp
// untouched.
macro_rules! send_passthrough {
    ($fd:expr, $msg:expr, $pt_trans:expr, $length:expr, $count:expr, $resp_ty:ty, $resp:expr, $err_msg:expr) => {{
        (*$msg).header.length = $length;
        (*$msg).header.count = $count;

        let ret = manage_ioctl($fd, &mut $msg);
        if ret != 0 {
            eqaic_eprintln!($err_msg);
            return ret;
        }

        let cmd_resp = reference_passthrough_cmd!($pt_trans, $resp_ty);
        unsafe {
            std::ptr::copy_nonoverlapping(cmd_resp, $resp, 1);
        }
    }};
}

fn manage_ioctl(fd: i32, user_msg: &mut manage_msg_t) -> i32 {
    let mut msg = qaic_manage_msg {
        len: (*user_msg).header.length,
        count: (*user_msg).header.count,
        data: &mut ((*user_msg).data) as *const u8 as u64,
    };

    // TODO: check message length and count are valid
    unsafe {
        libc::ioctl(
            fd,
            _DRM_IOCTL_QAIC_MANAGE as libc::c_ulong,
            &mut msg as *mut qaic_manage_msg as *mut c_void,
        )
    }
}

pub fn status_query(fd: i32, resp: &mut nnc_cmd_status_query_response_t) -> nnc_status_code_t {
    let mut msg = Box::new(manage_msg_t {
        header: manage_msg_header_t {
            length: (size_of::<nnc_transaction_passthrough_t>()
                + size_of::<nnc_cmd_status_query_request_t>()) as u32,
            count: 1,
        },
        ..Default::default()
    });

    let pt_trans = init_passthrough_header!(msg, nnc_cmd_status_query_request_t);
    let status_query_req = reference_passthrough_cmd!(pt_trans, nnc_cmd_status_query_request_t);
    (*status_query_req).type_ = nnc_command_type_NNC_COMMAND_TYPE_STATUS_REQ;

    send_passthrough!(
        fd,
        msg,
        pt_trans,
        (*pt_trans).header.len,
        1,
        nnc_cmd_status_query_response_t,
        resp,
        "Status query ioctl failed!"
    );

    resp.status_code
}

pub fn status_query_transaction(
    fd: i32,
    resp: &mut nnc_transaction_status_response,
) -> nnc_status_code_t {
    let mut user_msg = Box::new(manage_msg_t {
        header: manage_msg_header_t {
            length: size_of::<nnc_transaction_status_request>() as u32,
            count: 1,
        },
        ..Default::default()
    });
    let req: *mut nnc_transaction_status_request =
        &mut user_msg.data as *mut u8 as *mut nnc_transaction_status_request;
    unsafe {
        (*req).header.type_ = nnc_transaction_type_NNC_TRANSACTION_TYPE_STATUS_UK as i32;
        (*req).header.len = size_of::<nnc_transaction_status_request>() as u32;
    }
    let mut ret = manage_ioctl(fd, &mut user_msg);
    if ret != 0 {
        eqaic_eprintln!("Status query transaction failed!");
        return ret;
    }
    let status_query_resp = &mut user_msg.data as *mut u8 as *mut nnc_transaction_status_response;
    unsafe {
        std::ptr::copy_nonoverlapping(status_query_resp, resp, 1);
        ret = (*status_query_resp).status_code
    }
    ret
}

pub fn load_constants_ex(
    fd: i32,
    desc_path: &str,
    _const_name: Option<&str>,
    constants_id: &mut u32,
    resp: &mut nnc_cmd_load_constants_response_t,
) -> nnc_status_code_t {
    let desc_buf: Vec<u8> = std::fs::read(desc_path).expect("Unable to read binary file!");

    let mut msg = Box::new(manage_msg_t {
        ..Default::default()
    });

    let pt_trans = init_passthrough_header!(msg, nnc_cmd_load_constants_ex_request_t);
    // header.len and descriptor_len are u32 fields; descriptor files aren't
    // expected to exceed 4GB.
    (*pt_trans).header.len += desc_buf.len() as u32;

    let load_const_ex_req =
        reference_passthrough_cmd!(pt_trans, nnc_cmd_load_constants_ex_request_t);
    (*load_const_ex_req).type_ = nnc_command_type_NNC_COMMAND_TYPE_LOAD_CONSTANTS_EX_REQ;
    (*load_const_ex_req).process_id = std::process::id();
    (*load_const_ex_req).descriptor_len = desc_buf.len() as u32;
    unsafe {
        // Copy desc_buf into load_const_ex_req.descriptor
        let descriptor_ptr = load_const_ex_req.descriptor.as_mut_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(desc_buf.as_ptr(), descriptor_ptr, desc_buf.len());
    }

    send_passthrough!(
        fd,
        msg,
        pt_trans,
        (*pt_trans).header.len,
        1,
        nnc_cmd_load_constants_response_t,
        resp,
        "Load constants ex ioctl failed!"
    );

    *constants_id = resp.constants_id;

    resp.status_code
}

fn load_constants_payload_at_offset(
    fd: i32,
    constants_path: &str,
    constants_id: u32,
    offset: u64,
    _length: u64,
    resp: &mut nnc_cmd_load_constants_response_t,
) -> nnc_status_code_t {
    let mut binfile = std::fs::File::open(constants_path).expect("Unable to open binary file!");
    let _ = binfile.seek(SeekFrom::Start(offset));
    let mut constants_buf: Vec<u8> = Vec::new();
    let _ = binfile.read_to_end(&mut constants_buf);
    let constants_len = constants_buf.len();

    let mut msg = Box::new(manage_msg_t {
        ..Default::default()
    });

    let pt_trans = init_passthrough_header!(msg, nnc_cmd_load_constants_payload_request_t);

    let load_const_pl_req =
        reference_passthrough_cmd!(pt_trans, nnc_cmd_load_constants_payload_request_t);
    (*load_const_pl_req).type_ = nnc_command_type_NNC_COMMAND_TYPE_LOAD_CONSTANTS_PAYLOAD_REQ;
    (*load_const_pl_req).process_id = std::process::id();
    (*load_const_pl_req).tag = NNC_CONSTANTS_TAG;
    (*load_const_pl_req).constants_id = constants_id;
    (*load_const_pl_req).data_length = constants_len as u64;
    (*load_const_pl_req).offset = offset;

    let trans_dma_xfer = init_dma_xfer!(msg, pt_trans);
    (*trans_dma_xfer).tag = NNC_CONSTANTS_TAG;
    (*trans_dma_xfer).count = 1;
    set_alp!(trans_dma_xfer, constants_buf);

    send_passthrough!(
        fd,
        msg,
        pt_trans,
        (*pt_trans).header.len + (*trans_dma_xfer).header.len,
        2,
        nnc_cmd_load_constants_response_t,
        resp,
        "Load constants payload ioctl failed!"
    );

    resp.status_code
}

pub fn load_constants_payload(
    fd: i32,
    constants_path: &str,
    constants_id: u32,
    resp: &mut nnc_cmd_load_constants_response_t,
) -> nnc_status_code_t {
    load_constants_payload_at_offset(fd, constants_path, constants_id, 0, 0, resp)
}

pub fn unload_constants(
    fd: i32,
    constants_id: u32,
    resp: &mut nnc_cmd_unload_constants_response_t,
) -> nnc_status_code_t {
    let mut msg = Box::new(manage_msg_t {
        ..Default::default()
    });

    let pt_trans = init_passthrough_header!(msg, nnc_cmd_unload_constants_request_t);

    let unload_consts_req =
        reference_passthrough_cmd!(pt_trans, nnc_cmd_unload_constants_request_t);
    (*unload_consts_req).type_ = nnc_command_type_NNC_COMMAND_TYPE_UNLOAD_CONSTANTS_REQ;
    (*unload_consts_req).constants_id = constants_id;

    send_passthrough!(
        fd,
        msg,
        pt_trans,
        (*pt_trans).header.len,
        1,
        nnc_cmd_unload_constants_response_t,
        resp,
        "Unload constants ioctl failed!"
    );

    resp.status_code
}

pub fn load_elf(
    fd: i32,
    elf_path: &str,
    _network_name: Option<&str>,
    elf_id: &mut u32,
    resp: &mut nnc_cmd_load_elf_response_t,
) -> nnc_status_code_t {
    let mut elf_buf: Vec<u8> = std::fs::read(elf_path).expect("Unable to read binary file!");

    let mut msg = Box::new(manage_msg_t {
        ..Default::default()
    });

    let pt_trans = init_passthrough_header!(msg, nnc_cmd_load_elf_request_t);

    let load_elf_req = reference_passthrough_cmd!(pt_trans, nnc_cmd_load_elf_request_t);
    (*load_elf_req).type_ = nnc_command_type_NNC_COMMAND_TYPE_LOAD_ELF_REQ;
    (*load_elf_req).process_id = std::process::id();
    (*load_elf_req).tag = NNC_IMAGE_TAG;
    (*load_elf_req).data_length = elf_buf.len() as u64;
    (*load_elf_req).metadata_flatbuffer_crc = 0; // To be set
                                                 // TODO: check if network_name is set

    let trans_dma_xfer = init_dma_xfer!(msg, pt_trans);
    (*trans_dma_xfer).tag = NNC_IMAGE_TAG;
    (*trans_dma_xfer).count = 1;
    set_alp!(trans_dma_xfer, elf_buf);

    send_passthrough!(
        fd,
        msg,
        pt_trans,
        (*pt_trans).header.len + (*trans_dma_xfer).header.len,
        2,
        nnc_cmd_load_elf_response_t,
        resp,
        "Load elf ioctl failed!"
    );

    *elf_id = resp.elf_id;

    resp.status_code
}

pub fn unload_elf(
    fd: i32,
    elf_id: u32,
    resp: &mut nnc_cmd_unload_elf_response_t,
) -> nnc_status_code_t {
    let mut msg = Box::new(manage_msg_t {
        ..Default::default()
    });

    let pt_trans = init_passthrough_header!(msg, nnc_cmd_unload_elf_request_t);

    let unload_elf_req = reference_passthrough_cmd!(pt_trans, nnc_cmd_unload_elf_request_t);
    (*unload_elf_req).type_ = nnc_command_type_NNC_COMMAND_TYPE_UNLOAD_ELF_REQ;
    (*unload_elf_req).elf_id = elf_id;

    send_passthrough!(
        fd,
        msg,
        pt_trans,
        (*pt_trans).header.len,
        1,
        nnc_cmd_unload_elf_response_t,
        resp,
        "Unload elf ioctl failed!"
    );

    resp.status_code
}

fn activate(
    fd: i32,
    params: &ActivateParams,
    activ_id: &mut u32,
    dbc_id: &mut u32,
    static_ddr_base: &mut u64,
    dynamic_ddr_base: &mut u64,
    mcid_base: &mut u64,
    eventfd: &mut i32,
    resp: &mut nnc_cmd_activate_response_t,
) -> nnc_status_code_t {
    let mut msg = Box::new(manage_msg_t {
        ..Default::default()
    });

    let pt_trans = init_passthrough_header!(msg, nnc_cmd_activate_request_t);

    let activate_req = reference_passthrough_cmd!(pt_trans, nnc_cmd_activate_request_t);
    (*activate_req).type_ = nnc_command_type_NNC_COMMAND_TYPE_ACTIVATE_REQ;
    (*activate_req).process_id = std::process::id();
    (*activate_req).constants_id = params.constants_id;
    (*activate_req).elf_id = params.elf_id;
    (*activate_req).reservation_type = params.res_type;
    (*activate_req).resource_res_id = params.resource_res_id;
    (*activate_req).opt_vc_setup = params.opt_vc_setup;
    (*activate_req).vc_res_id = params.vc_res_id;
    (*activate_req).initial_activation_state = params.init_state;
    (*activate_req).dma_type = params.dma_type;
    (*activate_req).opt_support_datapath_switching = params.support_datapath_switching;

    // Owned by this process; the caller must close it once the VC is torn down.
    let efd = unsafe { libc::eventfd(0, 0) };
    if efd < 0 {
        eqaic_eprintln!("Activate: failed to create eventfd.");
        return nnc_status_code_NNC_STATUS_ERROR;
    }
    *eventfd = efd;

    let trans_vc_set = init_trans_vc_setup!(msg, pt_trans);
    (*trans_vc_set).q_size = VCQUEUESIZE16K;
    (*trans_vc_set).eventfd = efd;
    (*trans_vc_set).options = 0;

    send_passthrough!(
        fd,
        msg,
        pt_trans,
        (*pt_trans).header.len + (*trans_vc_set).header.len,
        2,
        nnc_cmd_activate_response_t,
        resp,
        "Activate ioctl failed!"
    );

    *activ_id = resp.activation_id;
    *dbc_id = resp.virtual_channel_id;
    *static_ddr_base = resp.static_ddr_base;
    *dynamic_ddr_base = resp.dynamic_ddr_base;
    *mcid_base = resp.mcid_base;

    resp.status_code
}

pub fn activate_default(
    fd: i32,
    elf_id: u32,
    constants_id: u32,
    activ_id: &mut u32,
    dbc_id: &mut u32,
    static_ddr_base: &mut u64,
    dynamic_ddr_base: &mut u64,
    mcid_base: &mut u64,
    eventfd: &mut i32,
    resp: &mut nnc_cmd_activate_response_t,
) -> nnc_status_code_t {
    let params = ActivateParams {
        elf_id,
        constants_id,
        res_type: nnc_reservation_type_NNC_RESERVATION_NONE,
        resource_res_id: 0,
        opt_vc_setup: 1,
        vc_res_id: NNC_VC_RESERVATION_ID_DEFAULT,
        init_state: nnc_activation_state_type_NNC_ACTIVATION_STATE_CMD_READY,
        dma_type: NncDmaOptionType::NncDmaOptionShared as u32,
        support_datapath_switching: 0,
    };
    activate(
        fd,
        &params,
        activ_id,
        dbc_id,
        static_ddr_base,
        dynamic_ddr_base,
        mcid_base,
        eventfd,
        resp,
    )
}

pub fn deactivate(
    fd: i32,
    activation_id: u32,
    dbc_id: u32,
    vc_teardown: u32,
    resp: &mut nnc_cmd_deactivate_response_t,
) -> nnc_status_code_t {
    let mut msg = Box::new(manage_msg_t {
        ..Default::default()
    });

    let pt_trans = init_passthrough_header!(msg, nnc_cmd_deactivate_request_t);

    let deactivate_req = reference_passthrough_cmd!(pt_trans, nnc_cmd_deactivate_request_t);
    (*deactivate_req).type_ = nnc_command_type_NNC_COMMAND_TYPE_DEACTIVATE_REQ;
    (*deactivate_req).activation_id = activation_id;
    (*deactivate_req).opt_vc_teardown = vc_teardown;

    let deactivate_trans = init_trans_vc_teardown!(msg, pt_trans);
    (*deactivate_trans).virtual_channel_id = dbc_id;

    send_passthrough!(
        fd,
        msg,
        pt_trans,
        (*pt_trans).header.len + (*deactivate_trans).header.len,
        2,
        nnc_cmd_deactivate_response_t,
        resp,
        "Deactivate ioctl failed!"
    );

    resp.status_code
}

//TODO: should probably create an enum for fw versions
pub fn is_valid_fw(status_resp: &nnc_cmd_status_query_response_t) -> Option<&str> {
    let fw_version_bytes = status_resp.info.nsp_data.nsp_fw_version.variant;
    let fw_version_str = unsafe {
        let slice = std::slice::from_raw_parts(
            fw_version_bytes.as_ptr() as *const u8,
            fw_version_bytes.len(),
        );
        std::str::from_utf8(slice)
            .expect("Could not parse FW version!")
            .trim_end_matches('\0')
    };
    if let Some(dev_name) = fw_version_str.split('.').next() {
        if dev_name != "aic100" {
            eqaic_eprintln!("Device is not supported!");
            return None;
        } else {
            eqaic_debugln!("Detected device: {}", dev_name);
            return Some("AIC 100");
        }
    } else {
        eqaic_eprintln!("Could not get device name!");
        return None;
    }
}
