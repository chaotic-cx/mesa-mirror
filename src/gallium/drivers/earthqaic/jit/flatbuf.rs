/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

extern crate elf;

use std::ffi::c_void;

use flatbuffers_rust::FlatBufferBuilder;
use jit_generated::aic_jit::*;

use crate::jit::device::JitBuffer;
use crate::jit::mem::align_up;
use crate::jit::operators::ElfData;
use crate::qaic;
use crate::qaic::data::DataDirection::*;
use crate::qaic::metadata::AicMdDoorbellOpSize::*;
use crate::qaic::network::Network;
use crate::qaic::qaic_rs_bindings::*;
use crate::qaic::semaphores::*;

pub const AIC_CMD_RESPONSE_ERR_MSG_LEN: usize = 112;
pub const COMMAND_SET_REQUEST_CANARY: u32 = 0x9c039a1c;

// Must match JIT_CMD_SET_MAX_OUTPUT_SEM_VAL in JitHwDevLib.h.
pub const JIT_CMD_SET_MAX_OUTPUT_SEM_VAL: u64 = 4000;

#[derive(Default)]
#[repr(C)]
pub struct AicCommandSetResponseInfo {
    pub offset: u64,
    pub size: u64,
}

#[repr(C)]
pub struct AicCommandSetRequestHeader {
    pub canary: u32, // Must be CommandSetRequestCanary value 0x9c039a1c
    pub length: u32, // Include this header
    pub cmd_set_resp_info: AicCommandSetResponseInfo,
}

#[repr(C)]
pub struct AicCommandSetRequest {
    pub header: AicCommandSetRequestHeader,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct AicCommandSetResponseHeader {
    pub size: u32,
    pub command_id: u32,
    pub status: u32,
    pub tlv_count: u32,
    pub err_msg: [u8; AIC_CMD_RESPONSE_ERR_MSG_LEN],
}

/*
 * The C structs make heavy use of flexible arrays which are
 * not available in rust. For now, encode nearly everything
 * into one big structure.
 *
 *.E.g., tlv.data points to CommandValue, commandValue.data
 * points to DevMemChunkList, and chunks is a flexible array
 * of DevMemChunks.
 */

#[derive(Debug)]
#[repr(C)]
pub struct DevMemChunk {
    pub phys_addr: u64,
    pub size: u64,
    pub buf_id: u32,
    padding: [u8; 4],
}

#[derive(Debug)]
#[repr(C)]
pub struct ResponseTLVHeader {
    pub type_: u32,
    pub slot_count: u32,
    pub command_id: u32,
    pub status: u32,
    pub num_chunks: u32,
    pub padding: [u8; 4],
}

#[derive(Debug)]
#[repr(C)]
pub struct ResponseTLV {
    pub header: ResponseTLVHeader,
    pub chunks: Vec<DevMemChunk>,
}

pub fn parse_tlvs(response_bo_ptr: *mut c_void) -> Vec<ResponseTLV> {
    // Copy response from BO
    let response: AicCommandSetResponseHeader = unsafe {
        let bo_ptr_cast = response_bo_ptr as *const AicCommandSetResponseHeader;
        std::ptr::read(bo_ptr_cast)
    };

    let mut tlv_start_ptr =
        unsafe { response_bo_ptr.add(size_of::<AicCommandSetResponseHeader>()) };
    let mut tlvs: Vec<ResponseTLV> = Vec::new();
    for _ in 0..response.tlv_count {
        let mut tlv = ResponseTLV {
            header: unsafe {
                let ptr_cast = tlv_start_ptr as *const ResponseTLVHeader;
                std::ptr::read(ptr_cast)
            },
            chunks: Vec::new(),
        };
        for _ in 0..tlv.header.num_chunks {
            unsafe {
                tlv_start_ptr = tlv_start_ptr.add(size_of::<ResponseTLVHeader>());
                tlv.chunks.push({
                    let ptr_cast = tlv_start_ptr as *const DevMemChunk;
                    std::ptr::read(ptr_cast)
                });
                tlv_start_ptr = tlv_start_ptr.add(size_of::<DevMemChunk>());
            }
        }
        tlvs.push(tlv);
    }
    tlvs
}

fn create_aic_command_set(
    bldr: &mut FlatBufferBuilder,
    dest: &mut Vec<u8>,
    args: &AicCommandSetArgs,
) {
    dest.clear();
    let cmdset_offset = AicCommandSet::create(bldr, args);
    finish_aic_command_set_buffer(bldr, cmdset_offset);
    let finished_data = bldr.finished_data();
    dest.extend_from_slice(finished_data);
}

const JIT_CMDSET_RESP_SIZE: u64 = 0x100000;

#[derive(PartialEq)]
pub enum CmdSetType {
    DdrAlloc,
    OperatorExecute,
}

pub struct CmdSetResponse {
    pub header: AicCommandSetResponseHeader,
    pub tlvs: Option<Vec<ResponseTLV>>,
}

pub fn send_cmdset(
    fd: i32,
    network: &Network,
    request: AicCommandSetRequest,
    type_: CmdSetType,
    ncmds: u64,
) -> Option<CmdSetResponse> {
    let static_ddr_base = (*network).static_ddr_base;
    let mcid_base = (*network).mcid_base;
    let dbc_id = (*network).dbc_id;

    let mut request_bo = qaic::data::BOParams {
        ..Default::default()
    };
    let request_bo_ptr = qaic::data::create_and_mmap_bo(fd, 8192, &mut request_bo);
    if request_bo_ptr.is_null() {
        eqaic_eprintln!("send_cmdset: create_and_mmap_bo failed for request bo!");
        return None;
    }

    let mut response_bo = qaic::data::BOParams {
        ..Default::default()
    };
    let response_bo_ptr = qaic::data::create_and_mmap_bo(fd, 4096, &mut response_bo);
    if response_bo_ptr.is_null() {
        eqaic_eprintln!("send_cmdset: create_and_mmap_bo failed for response bo!");
        return None;
    }

    let dma0 = &network.metadata.dma_requests[0];

    let cmdset_idx = ncmds;
    let db_base = mcid_base + 0x2000;
    let cmdset_base = mcid_base + 0x800000;
    let db_len = if dma0.db_ops.size == AicMdDoorbellOpSize8 {
        8
    } else if dma0.db_ops.size == AicMdDoorbellOpSize16 {
        16
    } else if dma0.db_ops.size == AicMdDoorbellOpSize32 {
        32
    } else {
        0
    };
    let mut request_slice = qaic_attach_slice_entry {
        size: 8192,
        sem0: default_pre_sem!(1, 0),
        sem1: empty_sem!(),
        sem2: empty_sem!(),
        sem3: empty_sem!(),
        dev_addr: cmdset_base + (cmdset_idx % 4) * 0x2000,
        db_addr: db_base + (cmdset_idx % 4) * 0x4,
        db_data: dma0.db_ops.data,
        db_len: db_len,
        offset: 0,
    };

    // When we execute 4000th cmd, the qshim opSemCounter hits its max and blocks
    // waiting for the host to write 1 to cmdSetOutputDB (doorbell at db_base+0x10)
    let is_rollover = ncmds % JIT_CMD_SET_MAX_OUTPUT_SEM_VAL == JIT_CMD_SET_MAX_OUTPUT_SEM_VAL - 1;
    let mut response_slice = qaic_attach_slice_entry {
        size: 4096,
        sem0: define_sem!(
            cmdset_idx as u16 + 1,
            1,
            1,
            qaic_rs_bindings::QAIC_SEM_WAIT_GT_EQ as u8,
            0
        ),
        sem1: if is_rollover {
            // needed to reset DBC sem[1]=0 so the driver can safely restart its counter from 0,
            // staying within the 12-bit semaphore limit.
            default_post_sem!(0, 1)
        } else {
            empty_sem!()
        },
        sem2: empty_sem!(),
        sem3: empty_sem!(),
        dev_addr: static_ddr_base + ((cmdset_idx % 4) * JIT_CMDSET_RESP_SIZE),
        db_addr: if is_rollover { db_base + 0x10 } else { 0 },
        db_data: if is_rollover { 1 } else { 0 },
        db_len: if is_rollover { db_len } else { 0 },
        offset: 0,
    };

    // Copy request into BO
    unsafe {
        let bo_ptr_cast = request_bo_ptr as *mut AicCommandSetRequestHeader;
        std::ptr::write(bo_ptr_cast, request.header);
        let bo_ptr_cast = request_bo_ptr.add(size_of::<AicCommandSetRequestHeader>()) as *mut u8;
        std::ptr::copy_nonoverlapping(request.body.as_ptr(), bo_ptr_cast, request.body.len());
    }

    let ret = qaic::data::attach_slice_bo(
        fd,
        1,
        dbc_id,
        request_bo.handle,
        DmaToDevice,
        8192,
        &mut request_slice,
    );
    if ret != 0 {
        eqaic_eprintln!("send_cmdset: Failed to slice BO! Err: {}", ret);
        return None;
    }

    let ret = qaic::data::attach_slice_bo(
        fd,
        1,
        dbc_id,
        response_bo.handle,
        DmaFromDevice,
        4096,
        &mut response_slice,
    );
    if ret != 0 {
        eqaic_eprintln!("send_cmdset: Failed to slice BO! Err: {}", ret);
        return None;
    }

    let mut exec_entries = Vec::new();
    exec_entries.push(qaic_partial_execute_entry {
        handle: request_bo.handle,
        dir: DmaToDevice as u32,
        resize: 8192,
    });
    exec_entries.push(qaic_partial_execute_entry {
        handle: response_bo.handle,
        dir: DmaFromDevice as u32,
        resize: 4096,
    });

    let ret =
        qaic::data::partial_execute_bo(fd, 2, request_bo.handle, dbc_id, exec_entries.as_mut_ptr());
    if ret != 0 {
        eqaic_eprintln!("send_cmdset: Failed to execute BO! Err: {}", ret);
        return None;
    }

    let ret = qaic::data::wait_bo(fd, response_bo.handle, 16000000, dbc_id);
    if ret != 0 {
        eqaic_eprintln!(
            "send_cmdset: Failed to wait on BO! ncmds={} is_rollover={} Err: {}",
            ncmds,
            is_rollover,
            ret
        );
        return None;
    }

    let mut response = CmdSetResponse {
        header: unsafe { (*(response_bo_ptr as *const AicCommandSetResponseHeader)).clone() },
        tlvs: None,
    };

    if type_ == CmdSetType::DdrAlloc {
        let tlvs = parse_tlvs(response_bo_ptr);
        response.tlvs = Some(tlvs);
    }

    unsafe {
        libc::munmap(request_bo_ptr, 8192);
        libc::munmap(response_bo_ptr, 4096);
    }

    qaic::data::close_bo(fd, request_bo.handle);
    qaic::data::close_bo(fd, response_bo.handle);

    Some(response)
}

pub fn encode_ddr_alloc_request(size: usize, cmdset_idx: u64) -> AicCommandSetRequest {
    let mut request = AicCommandSetRequest {
        header: AicCommandSetRequestHeader {
            canary: COMMAND_SET_REQUEST_CANARY,
            length: 0, // placeholder
            cmd_set_resp_info: AicCommandSetResponseInfo {
                offset: (cmdset_idx % 4) * JIT_CMDSET_RESP_SIZE,
                size: 256,
            },
        },
        body: Vec::new(),
    };

    let mut bldr = FlatBufferBuilder::new();
    bldr.force_defaults(true);

    let alloc_buf_args = AicCmdAllocBufArgs {
        size: size as u64,
        pid: std::process::id(),
    };
    let alloc_buf_offset = AicCmdAllocBuf::create(&mut bldr, &alloc_buf_args);

    let cmd_resp_info = AicResponseInfo::new(128, 256);
    let cmd_args = AicCommandArgs {
        // cmdset_idx is kept below JIT_CMD_SET_MAX_OUTPUT_SEM_VAL by
        // JitDevice::execute_operator, so it always fits in u32.
        commandID: cmdset_idx as u32 + 1,
        cmd_type: AicCmdUnion::AicCmdAllocBuf,
        cmd: Some(alloc_buf_offset.as_union_value()),
        cmdRespInfo: Some(&cmd_resp_info),
        ..Default::default()
    };
    let cmd_offset = AicCommand::create(&mut bldr, &cmd_args);
    let cmd_vec = bldr.create_vector(&[cmd_offset]);

    // Required, but has to be null
    let init = AicInit::create(&mut bldr, &AicInitArgs::default());
    let cmdset_args = AicCommandSetArgs {
        // cmdset_idx is kept below JIT_CMD_SET_MAX_OUTPUT_SEM_VAL by
        // JitDevice::execute_operator, so it always fits in u32.
        commandSetID: cmdset_idx as u32 + 1,
        init: Some(init),
        commands: Some(cmd_vec),
        ..Default::default()
    };

    create_aic_command_set(&mut bldr, &mut request.body, &cmdset_args);
    request.header.length = (request.body.len() + size_of::<AicCommandSetRequestHeader>()) as u32;

    if crate::debug_enabled() {
        let cmdset = root_as_aic_command_set(&request.body);
        eqaic_debugln!("{:?}", cmdset);
    }

    request
}

pub fn encode_execute_request(
    buf_id: u32,
    entries: &Vec<JitBuffer>,
    elf_data: &ElfData,
    operator_ddr_offset: u64,
    operator_name: String,
    cmd_set_idx: u64,
    thread_bitmap: u32,
    num_cores: u32,
) -> AicCommandSetRequest {
    let resp_size = 0x100000;
    let mut request = AicCommandSetRequest {
        header: AicCommandSetRequestHeader {
            canary: COMMAND_SET_REQUEST_CANARY,
            length: 0, // placeholder
            cmd_set_resp_info: AicCommandSetResponseInfo {
                offset: (cmd_set_idx % 4) * resp_size,
                size: 256,
            },
        },
        body: Vec::new(),
    };
    let mut bldr = FlatBufferBuilder::new();
    bldr.force_defaults(true);

    let ro_buf = DevDdrBuf::new(operator_ddr_offset, elf_data.rw_segment_offset, buf_id);
    let golden_buf = DevDdrBuf::new(
        operator_ddr_offset + elf_data.rw_segment_offset,
        elf_data.rw_segment_size,
        buf_id,
    );
    let rw_aligned = align_up(elf_data.rw_segment_size, 4096);
    let rw_buf = DevDdrBuf::new(
        operator_ddr_offset + elf_data.rw_segment_offset + rw_aligned,
        elf_data.rw_segment_size,
        buf_id,
    );

    // Create gotPltOffsets vector
    let got_plt_offsets = bldr.create_vector::<u32>(&elf_data.update_got_offsets);

    // Create exeName string
    let exe_name = bldr.create_string(&operator_name);

    // Create ExecInfo
    let exec_info = ExecInfo::create(
        &mut bldr,
        &ExecInfoArgs {
            exeName: Some(exe_name),
            exeType: AicExecutableType::SHARED_LIB,
            isCommonLib: false,
            gotPltOffsets: Some(got_plt_offsets),
            roSegmentBuf: Some(&ro_buf),
            rwSegmentBuf: Some(&rw_buf),
            goldenRWsegmentBuf: Some(&golden_buf),
        },
    );

    let exec_info_list = bldr.create_vector(&[exec_info]);

    // Empty, but we must pass an empty field to AicInit::create
    let init_semaphores = bldr.create_vector::<AicInitSemaphore>(&[]);

    // Create profilingRespInfo
    let profiling_resp_info = AicResponseInfo::new(128, 0);

    // Create profilingConfig
    let profiling_config = ProfilingConfiguration::create(
        &mut bldr,
        &ProfilingConfigurationArgs {
            profilingMetricBitMap: 0,
            profilingRespInfo: Some(&profiling_resp_info),
        },
    );

    // Create AicInit
    let aic_init = AicInit::create(
        &mut bldr,
        &AicInitArgs {
            execInfoList: Some(exec_info_list),
            initSemaphores: Some(init_semaphores),
            profilingConfig: Some(profiling_config),
        },
    );

    // Create bufferDescriptors
    let mut buf_descs = Vec::new();
    let mut args = Vec::new();

    let mut i = 0;
    for buf in entries {
        let eaddr = buf.addr;
        let esize = buf.size;
        buf_descs.push(AicBufferDescriptor::new(
            AicMemType::DDR,
            AicBufferResidency::BUF_RESIDENT_KEEP,
            AicBufferMappingRule::BUF_MAPPING_AUTO,
            eaddr,
            0,
            esize as u64,
            buf_id,
        ));
        args.push(AicParameter::new(
            AicParameterType::BUFFER_DESCRIPTOR,
            i as u32,
        ));
        i += 1;
    }
    let buffer_descriptors = bldr.create_vector(&buf_descs);
    let arg_list = bldr.create_vector(&args);

    // Create AicCmdDispatch
    let dispatch_cmd = AicCmdDispatch::create(
        &mut bldr,
        &AicCmdDispatchArgs {
            funcOffset: elf_data.subgraph_start_addr,
            exeIndex: 0,
            threadBitmap: thread_bitmap,
            numCores: num_cores,
            argList: Some(arg_list),
        },
    );

    // Wrap dispatch_cmd in a union
    let dispatch_union = flatbuffers_rust::WIPOffset::new(dispatch_cmd.value());

    // Empty values that must be passed to AicCommand::create
    let wait_sem_list = bldr.create_vector::<AicSemaphoreWait>(&[]);
    let signal_sem_list = bldr.create_vector::<AicSemaphoreSignal>(&[]);

    // Create AicCommand
    let command = AicCommand::create(
        &mut bldr,
        &AicCommandArgs {
            commandID: cmd_set_idx as u32 + 1,
            cmd_type: AicCmdUnion::AicCmdDispatch,
            cmd: Some(dispatch_union),
            cmdRespInfo: Some(&profiling_resp_info),
            waitSemList: Some(wait_sem_list),
            signalSemList: Some(signal_sem_list),
        },
    );

    let commands = bldr.create_vector(&[command]);

    // Create empty semaphores
    let empty_wait_sem_list = bldr.create_vector::<AicSemaphoreWait>(&[]);
    let empty_signal_sem_list = bldr.create_vector::<AicSemaphoreSignal>(&[]);

    let semaphores = AicCommandSetSemaphores::create(
        &mut bldr,
        &AicCommandSetSemaphoresArgs {
            entryWaitSemList: Some(empty_wait_sem_list),
            entrySignalSemList: Some(empty_signal_sem_list),
            exitWaitSemList: Some(empty_wait_sem_list),
            exitSignalSemList: Some(empty_signal_sem_list),
        },
    );

    let cmdset_args = AicCommandSetArgs {
        commandSetID: cmd_set_idx as u32 + 1,
        init: Some(aic_init),
        localStorage: None,
        bufferDescriptors: Some(buffer_descriptors),
        commands: Some(commands),
        semaphoares: Some(semaphores),
        ..Default::default()
    };
    create_aic_command_set(&mut bldr, &mut request.body, &cmdset_args);

    request.header.length = (request.body.len() + size_of::<AicCommandSetRequestHeader>()) as u32;
    request
}
