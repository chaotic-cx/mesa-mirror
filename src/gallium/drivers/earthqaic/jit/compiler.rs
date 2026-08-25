/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::ffi::CString;
use std::fs;
use std::os::unix::io::FromRawFd;
use std::process::Command;
use std::ptr;

extern crate llvm_sys;
use llvm_sys::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};
use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use llvm_sys::target_machine::LLVMCodeGenFileType;
use llvm_sys::target_machine::*;

use crate::driver_dir;
use crate::earthqaic_compute::EarthqaicComputeShader;
use crate::earthqaic_ml::EarthqaicMLSubgraph;
use crate::jit::error::JitError;
use crate::jit::nir::*;
use crate::jit::subgraph_llvm::*;

pub fn link_to_so(objs: &[&str], shared: &str) -> bool {
    let mut cmd = Command::new("clang");

    cmd.arg("--target=hexagon-unknown-elf")
        .arg("-fuse-ld=lld")
        .arg("-Wl,-z,separate-loadable-segments")
        .arg("-Wl,--verbose")
        .arg("-ffreestanding")
        .arg("-fno-builtin")
        .arg("-nostdlib")
        .arg("-std=c++17")
        .arg("-mv68")
        .arg("-mhvx=v68")
        .arg("-mhvx-ieee-fp")
        .arg("-fPIC")
        .arg("-shared")
        .arg("-o")
        .arg(shared);

    // Add all object files
    cmd.args(objs);

    let output = cmd.output().expect("Failed to run linker");
    if !output.status.success() {
        eqaic_eprintln!(
            "Failed to link {}: {}",
            shared,
            String::from_utf8_lossy(&output.stderr)
        );
        return false;
    } else {
        eqaic_debugln!(
            "{} link output:\n{}",
            shared,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    true
}

unsafe fn build_llvm_for_hexagon(module: LLVMModuleRef, obj_file: &str) {
    LLVMInitializeHexagonTarget();
    LLVMInitializeHexagonTargetInfo();
    LLVMInitializeHexagonTargetMC();
    LLVMInitializeHexagonAsmPrinter();

    let mut error = std::ptr::null_mut();

    let triple = CString::new("hexagon").unwrap();
    LLVMSetTarget(module, triple.as_ptr());

    // Create target machine
    let mut target = ptr::null_mut();
    if LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, &mut error) != 0 {
        panic!("Failed to get hexagon target");
    }

    let isa = CString::new("hexagonv68").unwrap();
    let features = CString::new("+hvxv68,+hvx-ieee-fp").unwrap();
    let target_machine = LLVMCreateTargetMachine(
        target,
        triple.as_ptr(),
        isa.as_ptr(),
        features.as_ptr(),
        LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
        LLVMRelocMode::LLVMRelocPIC,
        LLVMCodeModel::LLVMCodeModelDefault,
    );

    // Emit assembly
    let output_file = CString::new(obj_file).unwrap();
    if LLVMTargetMachineEmitToFile(
        target_machine,
        module,
        output_file.as_ptr() as *mut _,
        LLVMCodeGenFileType::LLVMObjectFile,
        &mut error,
    ) != 0
    {
        panic!("Failed to emit earthqaic.o");
    }

    LLVMDisposeTargetMachine(target_machine);
}

/*
 * Create a uniquely-named temporary file with the given suffix (e.g.
 * ".o") in the system temp dir, returning both the open file (kept
 * alive to hold the fd reservation until the caller is done with the
 * path) and the path itself.
 */
fn make_temp_file(suffix: &str) -> Result<(fs::File, String), JitError> {
    let template = format!(
        "{}/earthqaicXXXXXX{}",
        std::env::temp_dir().display(),
        suffix
    );
    let mut template_bytes = CString::new(template).unwrap().into_bytes_with_nul();

    let fd = unsafe {
        libc::mkstemps(
            template_bytes.as_mut_ptr() as *mut libc::c_char,
            suffix.len() as libc::c_int,
        )
    };
    if fd < 0 {
        return Err(JitError::CompileError(
            "Failed to create temporary file.".to_string(),
        ));
    }

    template_bytes.pop(); // drop the trailing NUL
    let path = String::from_utf8(template_bytes).unwrap();
    let file = unsafe { fs::File::from_raw_fd(fd) };
    Ok((file, path))
}

/*
 * Verify an already-lowered LLVM module, emit it to a hexagon object
 * file, link it (with any extra_objs) into a shared object, and read
 * the resulting bytes back. Shared by compile_subgraph/compile_nir,
 * which differ only in how they produce `module`/`context` and in
 * extra_objs (compile_subgraph links against operators.o; compile_nir
 * doesn't).
 *
 * Also returns the linked file's basename, which callers use to tag
 * the operator when executing it on device -- it's already unique
 * (mkstemps guarantees that) and short enough to fit the qshim's
 * fixed-size exeName buffer, unlike the full temp path.
 */
unsafe fn build_and_link(
    context: LLVMContextRef,
    module: LLVMModuleRef,
    extra_objs: &[String],
) -> Result<(Vec<u8>, String), JitError> {
    let verify = LLVMVerifyModule(
        module,
        LLVMVerifierFailureAction::LLVMReturnStatusAction,
        std::ptr::null_mut(),
    );
    if verify != 0 {
        LLVMDisposeModule(module);
        LLVMContextDispose(context);
        return Err(JitError::CompileError(
            "LLVM verification failed.".to_string(),
        ));
    }

    let (_o_fd_guard, o_file) = make_temp_file(".o")?;
    build_llvm_for_hexagon(module, &o_file);

    LLVMDisposeModule(module);
    LLVMContextDispose(context);

    let (_so_fd_guard, so_file) = make_temp_file(".so")?;
    let exe_name = std::path::Path::new(&so_file)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let mut objs: Vec<&str> = vec![&o_file];
    objs.extend(extra_objs.iter().map(|s| s.as_str()));
    let linked = link_to_so(&objs, &so_file);

    match fs::remove_file(&o_file) {
        Ok(_) => {}
        Err(e) => eqaic_eprintln!("Error removing file: {}", e),
    }

    if !linked {
        match fs::remove_file(&so_file) {
            Ok(_) => {}
            Err(e) => eqaic_eprintln!("Error removing file: {}", e),
        }
        return Err(JitError::CompileError(format!(
            "Failed to link {}",
            so_file
        )));
    }

    let so_bytes = std::fs::read(&so_file).unwrap();
    match fs::remove_file(&so_file) {
        Ok(_) => {}
        Err(e) => eqaic_eprintln!("Error removing file: {}", e),
    }

    Ok((so_bytes, exe_name))
}

pub fn compile_subgraph(subgraph: &EarthqaicMLSubgraph) -> Result<(Vec<u8>, String), JitError> {
    let tfl_ops_file = format!("{}/tensor_ops.o", driver_dir());
    let hexagon_lib_file = format!("{}/libhexagon.a", driver_dir());
    unsafe {
        let context = LLVMContextCreate();
        let module = lower_subgraph(context, subgraph);
        build_and_link(context, module, &[tfl_ops_file, hexagon_lib_file])
    }
}

pub fn compile_nir(shader: &EarthqaicComputeShader) -> Result<(Vec<u8>, String), JitError> {
    let nir = unsafe { (*shader).base.ir.nir };
    unsafe {
        let context = LLVMContextCreate();
        let module = lower_nir(context, nir);
        build_and_link(context, module, &[])
    }
}
