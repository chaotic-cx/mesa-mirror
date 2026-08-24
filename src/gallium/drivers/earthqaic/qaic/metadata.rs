/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use crate::qaic::qaic_rs_bindings::{qaic_sem, NNC_MAX_STRING_LENGTH};

#[derive(Debug, PartialEq)]
pub enum AicMdDoorbellOpSize {
    AicMdDoorbellOpSize8,
    AicMdDoorbellOpSize16,
    AicMdDoorbellOpSize32,
}

#[derive(Debug)]
#[repr(C)]
pub struct DoorbellOps {
    pub data: u32,
    pub mc_id: u32,
    pub offset: u64,
    pub size: AicMdDoorbellOpSize,
}

#[derive(Clone, Debug)]
#[allow(unused)]
pub enum AicMdSemaphoreSync {
    AicMdSemaphoreSyncPost = 0,
    AicMdSemaphoreSyncPre = 1,
}

#[derive(Clone, Debug)]
#[allow(unused)]
pub enum AicMdSemaphoreOpcode {
    AicMdSemaphoreCmdNOP = 0,
    AicMdSemaphoreCmdINIT = 1,
    AicMdSemaphoreCmdINC = 2,
    AicMdSemaphoreCmdDEC = 3,
    AicMdSemaphoreCmdWAITEQ = 4,
    AicMdSemaphoreCmdWAITGE = 5,
    AicMdSemaphoreCmdGT0 = 6,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct SemaphoreOps {
    pub in_sync_fences: u32,
    pub out_sync_fences: u32,
    pub pre_or_post: AicMdSemaphoreSync,
    pub sem_num: u32,
    pub sem_op: AicMdSemaphoreOpcode,
    pub sem_value: u32,
}

#[derive(Debug)]
#[allow(unused)]
pub enum AicMdDmaEntryAddrSpace {
    AicMdDmaAddrSpaceMC,
    AicMdDmaAddrSpaceDDR,
    AicMdDmaAddrSpaceDDRDynamicShared,
}

#[derive(Debug)]
#[repr(C)]
pub struct DmaRequest {
    pub dev_addr_space: AicMdDmaEntryAddrSpace,
    pub num: u32,
    pub dev_addr_offset: u64,
    pub host_addr_offset: u64,
    pub size: u64,
    pub in_out: u32,
    pub mc_id: u32,
    pub port_id: u32,
    pub semaphore_count: u32,
    pub db_ops: DoorbellOps,
    pub semaphore_ops: [SemaphoreOps; 4],
}

#[derive(Debug)]
#[repr(C)]
pub struct CompactMetadata {
    pub network_name: [u8; NNC_MAX_STRING_LENGTH as usize],
    pub nsp_count: u32,
    pub semaphore_count: u32,
    pub l2tcm_size: u32,
    pub vtcm_size: u32,
    pub dma_req_count: u32,
    pub static_constants_exist: bool,
    pub dma_requests: Vec<DmaRequest>,
}

pub fn read_md_struct_bin(path: &str) -> Result<CompactMetadata, std::io::Error> {
    let md_buf: Vec<u8> = std::fs::read(path)?; //.expect(format!("Unable to read md_struct.bin from {}", path));
    let md_consts = &md_buf[0..88 /* size of CompactMetadata w/o vector of DMA requests */];
    let mut md: CompactMetadata = unsafe { std::ptr::read(md_consts.as_ptr() as *const _) };
    md.dma_requests = Vec::new();
    let md_dma = &md_buf[88..];
    // Remaining bytes should contain only DmaRequests
    if md_dma.len() % size_of::<DmaRequest>() != 0 {
        eqaic_eprintln!("Misaligned metadata!")
    } else {
        let mut remaining_bytes = md_dma.len();
        let mut iter = 0;
        while remaining_bytes > 0 {
            let base = iter * size_of::<DmaRequest>();
            let md_dma_slice = &md_dma[base..base + size_of::<DmaRequest>()];
            let dma_req = unsafe { std::ptr::read(md_dma_slice.as_ptr() as *const _) };
            md.dma_requests.push(dma_req);
            remaining_bytes -= size_of::<DmaRequest>();
            iter += 1;
        }
        assert!(md.dma_requests.len() * size_of::<DmaRequest>() == md_buf.len() - 88);
    }
    Ok(md)
}

#[allow(unused)]
pub fn create_qaic_sem(sem: &SemaphoreOps) -> qaic_sem {
    debug_assert!(
        sem.sem_value <= u16::MAX as u32,
        "sem_value {} overflows u16",
        sem.sem_value
    );
    debug_assert!(
        sem.sem_num <= u8::MAX as u32,
        "sem_num {} overflows u8",
        sem.sem_num
    );
    qaic_sem {
        val: sem.sem_value as u16,
        index: sem.sem_num as u8,
        presync: sem.pre_or_post.clone() as u8,
        cmd: sem.sem_op.clone() as u8,
        flags: 0,
        pad: 0,
    }
}
