/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use crate::earthqaic_compute::EarthqaicComputeShader;
use crate::earthqaic_device::EarthqaicResource;
use crate::earthqaic_ml::EarthqaicMLSubgraph;
use crate::jit;
use crate::jit::error::JitError;
use crate::jit::flatbuf::*;
use crate::jit::mem::MemRegion;
use crate::jit::operators::*;
use crate::qaic;
use crate::qaic::data::DataDirection::*;
use crate::qaic::network::Network;
use crate::qaic::qaic_rs_bindings::*;
use crate::qaic::semaphores::*;

pub use crate::jit::mem::align_up;
pub use crate::jit::operators::ElfData;

#[derive(Copy, Clone, Default)]
pub struct JitDevice {
    ncmds: u64,
    mem: MemRegion,
    network: *const Network,
    fd: i32,
}

// TODO: rename this to JitBuffer
#[derive(Clone, Copy, Debug, Default)]
pub struct JitBuffer {
    pub addr: u64,
    pub size: usize,
}

/*
 * Attach a slice to handle, partially execute it, and wait for completion.
 * Shared by read_buffer/read_resource/write_buffer/write_resource, which
 * differ only in dir and which JitError variant to wrap failures in.
 */
fn transfer_bo(
    fd: i32,
    handle: u32,
    dbc_id: u32,
    addr: u64,
    size: u64,
    dir: qaic::data::DataDirection,
    err: fn(String) -> JitError,
) -> Result<(), JitError> {
    let mut slice = qaic_attach_slice_entry {
        size,
        sem0: empty_sem!(),
        sem1: empty_sem!(),
        sem2: empty_sem!(),
        sem3: empty_sem!(),
        dev_addr: addr,
        db_addr: 0,
        db_data: 0,
        db_len: 0,
        offset: 0,
    };

    let ret = qaic::data::attach_slice_bo(fd, 1, dbc_id, handle, dir, size, &mut slice);
    if ret != 0 {
        return Err(err(format!("Failed to slice BO: {}", ret)));
    }

    let mut exec_entry = qaic_partial_execute_entry {
        handle,
        dir: dir as u32,
        resize: size,
    };
    let ret = qaic::data::partial_execute_bo(fd, 1, handle, dbc_id, &mut exec_entry);
    if ret != 0 {
        return Err(err(format!("Failed to execute BO: {}", ret)));
    }

    let ret = qaic::data::wait_bo(fd, handle, 10000, dbc_id);
    if ret != 0 {
        return Err(err(format!("Failed to wait on BO: {}", ret)));
    }

    Ok(())
}

impl JitDevice {
    pub fn init(fd: i32, network: &Network) -> Result<JitDevice, JitError> {
        let jit = JitDevice {
            ncmds: 1,
            network: network,
            fd: fd,
            mem: MemRegion::init(fd, network)?,
        };
        Ok(jit)
    }

    // Returns an initialized memory region for the device
    pub fn alloc(&mut self, size: usize, alignment: usize) -> Result<u64, JitError> {
        Ok(self.mem.alloc(size, alignment)?)
    }

    pub fn alloc_cursor(&self) -> u64 {
        self.mem.cur
    }

    pub fn free_to(&mut self, addr: u64) {
        self.mem.free_to(addr);
    }

    fn check_addr(&self, addr: u64, label: &str) {
        if !self.mem.contains(addr) {
            eqaic_eprintln!(
                "BAD DDR addr in {}: 0x{:x} is outside pool \
                 [0x{:x}, 0x{:x})",
                label,
                addr,
                self.mem.base,
                self.mem.base + self.mem.size,
            );
        }
    }

    fn compile_operator(
        &mut self,
        operator_buf: &mut Vec<u8>,
        entry_symbol: &str,
    ) -> Result<(Vec<u8>, ElfData), JitError> {
        let qsymtab = read_qshim_symbols();
        let mut updated_operator_buf = operator_buf.clone();
        let elf_data = patch_qshim_symbols(
            operator_buf,
            &mut updated_operator_buf,
            qsymtab,
            entry_symbol,
        )?;

        if operator_buf.len() != updated_operator_buf.len() {
            return Err(JitError::MiscError(
                "Operator compilation failed.".to_string(),
            ));
        }

        Ok((updated_operator_buf, elf_data))
    }

    pub fn compile_subgraph(
        &mut self,
        subgraph: &EarthqaicMLSubgraph,
    ) -> Result<(Vec<u8>, ElfData, String), JitError> {
        let (mut operator_buf, exe_name) = jit::compiler::compile_subgraph(&subgraph)?;
        let (operator_buf, elf_data) = self.compile_operator(&mut operator_buf, "subgraph")?;
        Ok((operator_buf, elf_data, exe_name))
    }

    pub fn compile_nir(
        &mut self,
        shader: &EarthqaicComputeShader,
    ) -> Result<(Vec<u8>, ElfData, String), JitError> {
        let (mut operator_buf, exe_name) = jit::compiler::compile_nir(shader)?;
        if operator_buf.len() == 0 {
            return Ok((operator_buf, ElfData::default(), exe_name));
        }
        let (operator_buf, elf_data) = self.compile_operator(&mut operator_buf, "harness")?;
        Ok((operator_buf, elf_data, exe_name))
    }

    /*
     * read_buffer - create and execute BO for reading into a Vec<u8>
     * This read will create and mmap the necessary BO and read the result into a buffer.
     * This function should be called when the device read does not correspond to an
     * underlying pipe_resource and new BOs are required.
     */
    pub fn read_buffer(&mut self, addr: u64, size: u64) -> Result<Vec<u8>, JitError> {
        self.check_addr(addr, "read_buffer");
        let fd = self.fd;
        let network = self.network;
        let dbc_id = unsafe { (*network).dbc_id };
        let read_output_bo = qaic::data::MappedBo::new(fd, size)
            .ok_or_else(|| JitError::ReadError("create_and_mmap_bo failed.".to_string()))?;

        transfer_bo(
            fd,
            read_output_bo.handle,
            dbc_id,
            addr,
            size,
            DmaFromDevice,
            JitError::ReadError,
        )?;

        let output: Vec<u8> = unsafe {
            let ptr_cast = read_output_bo.ptr as *const u8;
            let slice = std::slice::from_raw_parts(ptr_cast, size as usize);
            slice.to_vec()
        };

        Ok(output)
    }

    /*
     * read_resource - execute exiting BO for reading.
     * This function assumes that BOs are already created and potentially mmaped.
     * This read should be called when reading into a pipe_resource.
     */
    pub fn read_resource(
        &mut self,
        addr: u64,
        resource: *mut EarthqaicResource,
        size: u64,
    ) -> Result<(), JitError> {
        let fd = self.fd;
        let network = self.network;
        let dbc_id = unsafe { (*network).dbc_id };
        let handle = unsafe { (*resource).handle };

        transfer_bo(
            fd,
            handle,
            dbc_id,
            addr,
            size,
            DmaFromDevice,
            JitError::ReadError,
        )
    }

    /*
     * write_buffer - create and execute BO for writing from a Vec<u8>
     * This write will create and mmap the necessary BO and write the buffered data to the device.
     * This function should be called when the buffered data does not correspond to an
     * underlying pipe_resource and new BOs are required.
     */
    // TODO: Take Buffer (soon to be JitBuffer) instead of data/size
    pub fn write_buffer(&mut self, addr: u64, data: &Vec<u8>, size: u64) -> Result<(), JitError> {
        self.check_addr(addr, "write_buffer");
        let fd = self.fd;
        let network = self.network;
        let dbc_id = unsafe { (*network).dbc_id };
        let load_input_bo = qaic::data::MappedBo::new(fd, size)
            .ok_or_else(|| JitError::WriteError("create_and_mmap_bo failed!".to_string()))?;

        // Copy input bytes (little endian) into BO
        unsafe {
            let bo_ptr_cast = load_input_bo.ptr as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), bo_ptr_cast, size as usize);
        }

        transfer_bo(
            fd,
            load_input_bo.handle,
            dbc_id,
            addr,
            size,
            DmaToDevice,
            JitError::WriteError,
        )
    }

    /*
     * write_resource - execute exiting BO for writing.
     * This function assumes that BOs are already created and potentially mmaped.
     * This write should be called when writing from an existing pipe_resource.
     */
    pub fn write_resource(
        &mut self,
        addr: u64,
        resource: *mut EarthqaicResource,
        size: u64,
    ) -> Result<(), JitError> {
        self.check_addr(addr, "write_resource");
        let fd = self.fd;
        let network = self.network;
        let dbc_id = unsafe { (*network).dbc_id };
        let handle = unsafe { (*resource).handle };
        eqaic_debugln!(
            "write_resource addr=0x{:x} size={} handle={} dbc={}",
            addr,
            size,
            handle,
            dbc_id
        );

        qaic::data::detach_slice_bo(fd, handle);

        transfer_bo(
            fd,
            handle,
            dbc_id,
            addr,
            size,
            DmaToDevice,
            JitError::WriteError,
        )
    }

    pub fn execute_operator(
        &mut self,
        operator_addr: u64,
        entries: &Vec<JitBuffer>,
        elf_data: &ElfData,
        operator_name: String,
        thread_bitmap: u32,
        num_cores: u32,
    ) -> Result<AicCommandSetResponseHeader, JitError> {
        self.check_addr(operator_addr, "execute_operator");
        let cmdset_idx = self.ncmds;
        let request = encode_execute_request(
            self.mem.buf_id,
            entries,
            elf_data,
            operator_addr - self.mem.base,
            operator_name,
            cmdset_idx,
            thread_bitmap,
            num_cores,
        );

        let netref = unsafe { &(*(self).network) };
        let response = send_cmdset(
            self.fd,
            netref,
            request,
            CmdSetType::OperatorExecute,
            cmdset_idx,
        )
        .ok_or_else(|| JitError::ExecutionError("send_cmdset failed".to_string()))?;

        // match the sem value to avoid overflows
        self.ncmds = (self.ncmds + 1) % JIT_CMD_SET_MAX_OUTPUT_SEM_VAL;
        if response.header.status != 0 {
            return Err(JitError::ExecutionError(format!(
                "Operator execution error: {}",
                String::from_utf8(response.header.err_msg.to_vec()).unwrap()
            )));
        }
        Ok(response.header)
    }
}
