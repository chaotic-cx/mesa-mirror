/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::collections::HashMap;

use elf::abi::PT_LOAD;
use elf::endian::LittleEndian;
use elf::parse::*;
use elf::relocation::Rela;
use elf::segment::SegmentTable;
use elf::ElfBytes;

use crate::jit::error::JitError;
use crate::jit::*;

const QSHIM_SYMBOLS: [&str; 12] = [
    "qshimGetRandomDataWait",
    "qshimGetRandomDataSubmit",
    "qshimGetBaseVtcmAddr",
    "qshimMmap",
    "qshimProfilingEntryEnd",
    "qshim2DUdmaSubmit",
    "qshimLinearUdmaSubmit",
    "qshimProfilingEntryStart",
    "qshimUDmaWait",
    "qshimSyncThreads",
    "qshimUnmap",
    "qshimLog",
];

/*
 * These functions will be mapped at these offsets in the NSP's
 * address space by the QShim. They must patched into each operator's
 * .got.plt section by the driver.
 */
pub fn read_qshim_symbols() -> HashMap<String, u64> {
    let map: HashMap<String, u64> = HashMap::from([
        ("qshimGetRandomDataWait".to_string(), 33581608u64),
        ("qshimGetRandomDataSubmit".to_string(), 33581400u64),
        ("qshimGetBaseVtcmAddr".to_string(), 33581776u64),
        ("qshimMmap".to_string(), 33581788u64),
        ("qshimProfilingEntryEnd".to_string(), 33574120u64),
        ("qshim2DUdmaSubmit".to_string(), 33580444u64),
        ("qshimLinearUdmaSubmit".to_string(), 33580040u64),
        ("qshimProfilingEntryStart".to_string(), 33573992u64),
        ("qshimUDmaWait".to_string(), 33581096u64),
        ("qshimSyncThreads".to_string(), 33579792u64),
        ("qshimUnmap".to_string(), 33582612u64),
        ("qshimLog".to_string(), 33555744u64),
    ]);
    map
}

#[derive(Default)]
pub struct ElfData {
    pub update_got_offsets: Vec<u32>,
    pub subgraph_start_addr: u64,
    pub rw_segment_offset: u64,
    pub rw_segment_size: u64,
}

/*
 * r_offsets and st_values are virtual addresses in shared object files:
 * https://docs.oracle.com/cd/E19683-01/816-1386/chapter6-54839/index.html
 * https://docs.oracle.com/cd/E19683-01/816-1386/6m7qcoblj/index.html#chapter6-35166
 * They must be translated to file offsets before they can be used to index
 * into the raw ELF bytes in patch_qshim_symbols.
 */
fn vaddr_to_file_offset(
    program_headers: &SegmentTable<LittleEndian>,
    vaddr: u64,
) -> Result<usize, JitError> {
    /*
     * Earthqaic operators contain multiple PT_LOAD segments.
     * Determine which one contains the virtual address, and resolve it.
     */
    for phdr in program_headers.iter() {
        if phdr.p_type == PT_LOAD && vaddr >= phdr.p_vaddr && vaddr < phdr.p_vaddr + phdr.p_filesz {
            return Ok((phdr.p_offset + (vaddr - phdr.p_vaddr)) as usize);
        }
    }
    Err(JitError::CompileError(format!(
        "virtual address {:#x} not contained in any PT_LOAD segment",
        vaddr
    )))
}

// TODO: can we do this using only one buffer?
pub fn patch_qshim_symbols(
    buf1: &mut Vec<u8>,
    buf2: &mut Vec<u8>,
    qsymtab: HashMap<String, u64>,
    entry_sym_name: &str,
) -> Result<ElfData, JitError> {
    let mut update_got_offsets: Vec<u32> = Vec::new();

    let elf = ElfBytes::<LittleEndian>::minimal_parse(&buf1)
        .map_err(|e| JitError::CompileError(format!("failed to parse operator ELF: {}", e)))?;
    let common = elf
        .find_common_data()
        .map_err(|e| JitError::CompileError(format!("failed to read ELF common data: {}", e)))?;
    let dynsyms = common
        .dynsyms
        .ok_or_else(|| JitError::CompileError("operator ELF has no .dynsym".to_string()))?;
    let strtab = common
        .dynsyms_strs
        .ok_or_else(|| JitError::CompileError("operator ELF has no .dynstr".to_string()))?;
    let program_headers = elf
        .segments()
        .ok_or_else(|| JitError::CompileError("operator ELF has no program headers".to_string()))?;

    // Update symbols from the .rela.plt
    let rela_plt_shdr = elf
        .section_header_by_name(".rela.plt")
        .map_err(|e| JitError::CompileError(format!("failed to look up .rela.plt: {}", e)))?
        .ok_or_else(|| JitError::CompileError(".rela.plt section missing".to_string()))?;
    let (rela_plt, _) = elf
        .section_data(&rela_plt_shdr)
        .map_err(|e| JitError::CompileError(format!("failed to read .rela.plt: {}", e)))?;
    let entsize = rela_plt_shdr.sh_entsize as usize;
    let count = rela_plt.len() / entsize;
    for i in 0..count {
        let mut offset = i * entsize;
        let rela = Rela::parse_at(
            LittleEndian,
            elf::file::Class::ELF32,
            &mut offset,
            &rela_plt,
        )
        .map_err(|e| JitError::CompileError(format!("failed to parse .rela.plt entry: {}", e)))?;
        let symbol = dynsyms
            .get(rela.r_sym as usize)
            .map_err(|e| JitError::CompileError(format!("failed to look up dynsym: {}", e)))?;
        let name = strtab
            .get(symbol.st_name as usize)
            .map_err(|e| JitError::CompileError(format!("failed to look up symbol name: {}", e)))?;
        let value = symbol.st_value;
        if value != 0 {
            // r_offset is a virtual address; translate to a file offset before indexing buf2.
            let got_plt_offset = vaddr_to_file_offset(&program_headers, rela.r_offset)?;
            // Hexagon is a 32-bit target; symbol/GOT addresses always fit in u32.
            let value_bytes: [u8; 4] = (value as u32).to_le_bytes();
            buf2[got_plt_offset..got_plt_offset + 4].copy_from_slice(&value_bytes);
            update_got_offsets.push(got_plt_offset as u32);
            eqaic_debugln!("UpdateGotOffsets: symbolName {} with value {}", name, value);
        } else if QSHIM_SYMBOLS.contains(&name) {
            // r_offset is a virtual address; translate to a file offset before indexing buf2.
            let got_plt_offset = vaddr_to_file_offset(&program_headers, rela.r_offset)?;
            // Hexagon is a 32-bit target; symbol/GOT addresses always fit in u32.
            let qshim_addr_bytes: [u8; 4] = (qsymtab[name] as u32).to_le_bytes();
            buf2[got_plt_offset..got_plt_offset + 4].copy_from_slice(&qshim_addr_bytes);
            eqaic_debugln!(
                "UpdateGotOffsets: symbolName {} with value {}",
                name,
                qsymtab[name] as u32
            );
        } else {
            eqaic_eprintln!("UpdateGotOffsets: {} in .rela.plt cannot be resolved", name);
            return Err(JitError::CompileError(format!(
                "could not resolve {}",
                name
            )));
        }
    }

    // Update symbols from the .rela.dyn
    if let Ok(Some(rela_dyn_shdr)) = elf.section_header_by_name(".rela.dyn") {
        let (rela_dyn, _) = elf
            .section_data(&rela_dyn_shdr)
            .map_err(|e| JitError::CompileError(format!("failed to read .rela.dyn: {}", e)))?;
        let entsize = rela_dyn_shdr.sh_entsize as usize;
        let count = rela_dyn.len() / entsize;
        for i in 0..count {
            let mut offset = i * entsize;
            let rela = Rela::parse_at(
                LittleEndian,
                elf::file::Class::ELF32,
                &mut offset,
                &rela_dyn,
            )
            .map_err(|e| {
                JitError::CompileError(format!("failed to parse .rela.dyn entry: {}", e))
            })?;
            let symbol = dynsyms
                .get(rela.r_sym as usize)
                .map_err(|e| JitError::CompileError(format!("failed to look up dynsym: {}", e)))?;
            let name = strtab.get(symbol.st_name as usize).map_err(|e| {
                JitError::CompileError(format!("failed to look up symbol name: {}", e))
            })?;
            let value = symbol.st_value;
            // __cxa_finalize_stub is already accounted for from .rela.plt
            if value != 0 && name != "__cxa_finalize_stub" {
                // r_offset is a virtual address; translate to a file offset before indexing buf2.
                let dyn_offset = vaddr_to_file_offset(&program_headers, rela.r_offset)?;
                // Hexagon is a 32-bit target; symbol/GOT addresses always fit in u32.
                let value_bytes: [u8; 4] = (value as u32).to_le_bytes();
                buf2[dyn_offset..dyn_offset + 4].copy_from_slice(&value_bytes);
                update_got_offsets.push(dyn_offset as u32);
                eqaic_debugln!("UpdateGotOffsets: symbolName {} with value {}", name, value);
            } else if value == 0 {
                eqaic_eprintln!("UpdateGotOffsets: {} in .rela.dyn cannot be resolved", name);
                return Err(JitError::CompileError(format!(
                    "could not resolve {}",
                    name
                )));
            }
        }
    }

    let mut rw_start: Option<u64> = None;
    let mut rw_end: u64 = 0;
    for phdr in program_headers.iter() {
        // type == PT_LOAD && flags == RW
        if phdr.p_type == PT_LOAD && phdr.p_flags == (0x4 | 0x2) {
            let start = phdr.p_offset;
            let end = phdr.p_offset + phdr.p_filesz;
            rw_start = Some(rw_start.map_or(start, |s| s.min(start)));
            rw_end = rw_end.max(end);
        }
    }
    let rw_segment_offset = rw_start.unwrap_or(4096);
    let rw_segment_size = rw_start.map_or(0, |start| rw_end - start);

    // Determine address of subgraph symbol
    let hash_table = common
        .sysv_hash
        .ok_or_else(|| JitError::CompileError("operator ELF has no SysV hash table".to_string()))?;
    let (_sym_idx, sym) = hash_table
        .find(entry_sym_name.as_bytes(), &dynsyms, &strtab)
        .map_err(|e| JitError::CompileError(format!("failed to look up entry symbol: {}", e)))?
        .ok_or_else(|| {
            JitError::CompileError(format!(
                "entry symbol '{}' not found in operator ELF",
                entry_sym_name
            ))
        })?;
    // Qshim expects the subgraph's start address to be represented as a file offset.
    // Since st_values are virtual addresses in shared object files, convert it to an offset.
    let subgraph_start_addr = vaddr_to_file_offset(&program_headers, sym.st_value)? as u64;

    Ok(ElfData {
        update_got_offsets: update_got_offsets,
        subgraph_start_addr: subgraph_start_addr,
        rw_segment_offset: rw_segment_offset,
        rw_segment_size: rw_segment_size,
    })
}

// This function is for debugging purposes to ensure the symbols get patched properly
#[allow(unused)]
fn compare_patched_symbols(buf1: &Vec<u8>, buf2: &Vec<u8>) {
    let bufs = vec![buf1, buf2];
    let mut first = true;
    for buf in bufs {
        if first {
            eqaic_debugln!("Original .got.plt:");
            first = false;
        } else {
            eqaic_debugln!("Patched .got.plt:");
        }
        let elf = ElfBytes::<LittleEndian>::minimal_parse(&buf).unwrap();
        let got_plt_shdr = elf.section_header_by_name(".got.plt").unwrap().unwrap();
        let (got_plt, _) = elf.section_data(&got_plt_shdr).unwrap();
        let count = got_plt.len();
        let mut line = String::new();
        for i in 0..count {
            if i % 4 == 0 && i > 0 {
                line.push(' ');
            }
            if i % 16 == 0 && i > 0 {
                eqaic_debugln!("{}", line);
                line.clear();
            }
            line += &format!("{:02x}", got_plt[i]);
        }
        eqaic_debugln!("{}", line);
    }
}
