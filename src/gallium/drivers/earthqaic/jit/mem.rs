/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::fmt;

use crate::jit::flatbuf::*;
use crate::qaic::network::Network;

#[derive(Debug)]
pub enum MemError {
    InitializationError,
    AllocationError,
}

impl fmt::Display for MemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemError::InitializationError => write!(f, "Could not initialize device memory."),
            MemError::AllocationError => write!(f, "Could not allocate device memory from pool."),
        }
    }
}

impl std::error::Error for MemError {}

pub fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

#[derive(Copy, Clone, Default)]
pub struct MemRegion {
    pub base: u64,
    pub size: u64,
    pub cur: u64,
    pub buf_id: u32,
}

impl MemRegion {
    pub fn init(fd: i32, network: &Network) -> Result<Self, MemError> {
        let request = encode_ddr_alloc_request(0, 0);
        let response = send_cmdset(fd, network, request, CmdSetType::DdrAlloc, 0)
            .expect("mem::init failed to send command set");
        let tlvs = response.tlvs.unwrap();
        let status_code = tlvs[0].header.status;
        if status_code != 0 {
            return Err(MemError::InitializationError);
        }

        let region = MemRegion {
            base: tlvs[0].chunks[0].phys_addr,
            size: tlvs[0].chunks[0].size,
            cur: tlvs[0].chunks[0].phys_addr,
            buf_id: tlvs[0].chunks[0].buf_id,
        };
        eqaic_debugln!(
            "DDR pool: base=0x{:x} size=0x{:x} ({} MB) buf_id={}",
            region.base,
            region.size,
            region.size / (1024 * 1024),
            region.buf_id,
        );
        Ok(region)
    }

    pub fn alloc(&mut self, alloc_size: usize, alignment: usize) -> Result<u64, MemError> {
        let cur_ptr = (self.cur + alignment as u64 - 1) & !(alignment as u64 - 1);
        if cur_ptr + alloc_size as u64 > self.base + self.size {
            let used = self.cur - self.base;
            eqaic_eprintln!(
                "DDR alloc OVERFLOW: requested {} bytes align={} \
                 used={}/{}",
                alloc_size,
                alignment,
                used,
                self.size,
            );
            return Err(MemError::AllocationError);
        }
        let region = cur_ptr;
        self.cur = cur_ptr + alloc_size as u64;

        let used = self.cur - self.base;
        eqaic_debugln!(
            "DDR alloc: addr=0x{:x} size={} used={}/{} ({:.1}%)",
            region,
            alloc_size,
            used,
            self.size,
            used as f64 / self.size as f64 * 100.0
        );

        Ok(region)
    }

    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.base + self.size
    }

    pub fn free_to(&mut self, addr: u64) {
        assert!(
            addr >= self.base,
            "free_to 0x{:x} is below pool base 0x{:x}",
            addr,
            self.base
        );
        assert!(
            addr <= self.cur,
            "free_to 0x{:x} is above cursor 0x{:x}",
            addr,
            self.cur
        );
        let reclaimed = self.cur - addr;
        eqaic_debugln!(
            "DDR free_to: 0x{:x} (reclaim {} bytes, used now {}/{})",
            addr,
            reclaimed,
            addr - self.base,
            self.size,
        );
        self.cur = addr;
    }
}
