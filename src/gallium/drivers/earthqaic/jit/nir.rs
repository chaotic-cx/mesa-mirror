/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::collections::HashMap;
use std::ffi::{CStr, CString};

use earthqaic_rs_bindings::*;

use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::LLVMIntPredicate;

struct BuildContext {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
    kernel: LLVMValueRef,
    kernel_param_types: Vec<LLVMTypeRef>,
    llvm_var_map: HashMap<u32, LLVMValueRef>,
    llvm_block_map: HashMap<u32, LLVMBasicBlockRef>,
}

// LLVM's C API requires a non-null, nul-terminated name for most builder
// calls, even when the name is purely diagnostic (dumps/-print-after-all).
// This wraps the CString::new(...).unwrap().as_ptr() dance used at nearly
// every call site.
fn cstr(name: &str) -> CString {
    CString::new(name).unwrap()
}

unsafe fn add_constant_load(inst: *mut nir_load_const_instr, build_context: &mut BuildContext) {
    let context = build_context.context;
    let llvm_var_map = &mut build_context.llvm_var_map;

    let i32_ty = LLVMInt32TypeInContext(context);

    let dest = (*inst).def;
    let value_ptr = (*inst).value.as_ptr() as *const nir_const_value;

    let num_components = (*inst).def.num_components;
    let bit_size = (*inst).def.bit_size;

    // TODO: determine the union type for this properly!
    // Now, we are assuming there is a single i32 component
    assert!(num_components == 1);
    assert!(bit_size == 32);
    let value = (*value_ptr).i32_;

    // Casting i32 to u64, maybe set sign extend param?
    let var = LLVMConstInt(i32_ty, value as u64, 0);
    llvm_var_map.insert(dest.index, var);
}

unsafe fn print_intrinsic_instr(inst: *mut nir_intrinsic_instr) {
    let dest = (*inst).def;
    let src_ptr = (*inst).src.as_ptr() as *const nir_src;

    // Like ALU instructions, intrinsics can be used to index nir_intrinsics_infos array
    let op_info = nir_intrinsic_infos[(*inst).intrinsic as usize];
    let op_name = std::ffi::CStr::from_ptr(op_info.name).to_str().unwrap();
    let src_count = op_info.num_srcs;

    // Maybe impl display for nir_alu_instr?
    let mut inst_str: String = format!("INTRINSIC {} = {:?}(", dest.index, op_name);
    for i in 0..src_count {
        inst_str += {
            let ssa = (*(src_ptr.wrapping_add(i as usize))).ssa;
            let src = (*ssa).index;
            &src.to_string()
        };
        if i < src_count - 1 {
            inst_str += ", ";
        }
    }
    inst_str += ")";
    eqaic_debugln!("{}", inst_str);
}

unsafe fn add_intrinsic_instr(
    inst: *mut nir_intrinsic_instr,
    build_context: &mut BuildContext,
    cur_param: &mut u32,
) {
    let module = build_context.module;
    let context = build_context.context;
    let builder = build_context.builder;
    let llvm_var_map = &mut build_context.llvm_var_map;
    let kernel = build_context.kernel;

    let i32_ty = LLVMInt32TypeInContext(context);
    let float_ty = LLVMFloatTypeInContext(context);

    let dest = (*inst).def;
    let src_ptr = (*inst).src.as_ptr() as *const nir_src;

    // Since we are passing get_global_id as the final kernel parameter:
    //
    // load_workgroup_size is going to load const 0 into dest
    // load_workgroup_id is going to load const 0 into dest
    // load_loacl_invocation_id is going to load the final kernel parameter into dest
    //
    // i = get_global_id(0) expands to:
    //   1 = load_workgroup_size()
    //   2 = load_workgroup_id()
    //   3 = load_local_invocation_id()
    //   4 = imul(2, 1)
    //   5 = add(3, 4) <-- 5 represents i
    //
    // Thus, the value of 3 must come from the kernel parameters, implying 4 must be 0.
    match (*inst).intrinsic {
        nir_intrinsic_op_nir_intrinsic_load_kernel_input => {
            // TODO: actually load kernel params properly with load_const
            // intrinsic instead of using cur_param
            let input_param = LLVMGetParam(kernel, *cur_param);
            llvm_var_map.insert(dest.index, input_param);
            *cur_param += 1;
        }
        nir_intrinsic_op_nir_intrinsic_load_workgroup_size => {
            let var = LLVMConstInt(i32_ty, 0, 0);
            llvm_var_map.insert(dest.index, var);
        }
        nir_intrinsic_op_nir_intrinsic_load_workgroup_id => {
            let var = LLVMConstInt(i32_ty, 0, 0);
            llvm_var_map.insert(dest.index, var);
        }
        nir_intrinsic_op_nir_intrinsic_load_base_workgroup_id => {
            let var = LLVMConstInt(i32_ty, 0, 0);
            llvm_var_map.insert(dest.index, var);
        }
        nir_intrinsic_op_nir_intrinsic_load_local_invocation_id => {
            let num_params = LLVMCountParams(kernel);
            let input_param = LLVMGetParam(kernel, num_params - 1);
            llvm_var_map.insert(dest.index, input_param);
        }
        nir_intrinsic_op_nir_intrinsic_load_global_invocation_index => {
            let num_params = LLVMCountParams(kernel);
            let input_param = LLVMGetParam(kernel, num_params - 1);
            llvm_var_map.insert(dest.index, input_param);
        }
        // Currently assuming all global values are floats!!
        // TODO: maybe the dest object has some type information?
        nir_intrinsic_op_nir_intrinsic_load_global => {
            let src_ssa_index = (*(*src_ptr).ssa).index;
            let src_ref = llvm_var_map.get(&src_ssa_index).unwrap();
            let ptr_name = cstr("fptr");
            let float_ptr_ty = LLVMPointerType(float_ty, 0);
            let ptr = LLVMBuildIntToPtr(builder, *src_ref, float_ptr_ty, ptr_name.as_ptr());
            let val_name = cstr("fval");
            let val = LLVMBuildLoad2(builder, float_ty, ptr, val_name.as_ptr());
            llvm_var_map.insert(dest.index, val);
        }
        nir_intrinsic_op_nir_intrinsic_store_global => {
            // Value to store
            let src1_ssa_index = (*(*src_ptr.wrapping_add(0)).ssa).index;
            let src1_ref = llvm_var_map.get(&src1_ssa_index).unwrap();
            // Location to store value
            let src2_ssa_index = (*(*src_ptr.wrapping_add(1)).ssa).index;
            let src2_ref = llvm_var_map.get(&src2_ssa_index).unwrap();

            let ptr_name = cstr("fptr");
            let float_ptr_ty = LLVMPointerType(float_ty, 0);
            let ptr = LLVMBuildIntToPtr(builder, *src2_ref, float_ptr_ty, ptr_name.as_ptr());
            LLVMBuildStore(builder, *src1_ref, ptr);
        }
        _ => {
            eqaic_eprintln!("NIR intrinsic {} not supported", (*inst).intrinsic);
            print_intrinsic_instr(inst);

            // Store the destination as a poisoned value in the mapping
            // This ensures that looking up the destination in subsequent
            // instructions doesnt fault.
            let poison = LLVMGetPoison(i32_ty);
            llvm_var_map.insert(dest.index, poison);

            // Purposely create an invalid instruction to force the verifier
            // to fail.
            let fty = LLVMFunctionType(
                llvm_sys::core::LLVMVoidTypeInContext(context),
                std::ptr::null_mut(),
                0,
                0,
            );
            let f = LLVMAddFunction(module, b"always_bad\0".as_ptr() as *const i8, fty);
            LLVMAppendBasicBlockInContext(context, f, b"broken\0".as_ptr() as *const i8);
        }
    }
}

unsafe fn get_alu_src_ssa_index(src_ptr: *const nir_alu_src, idx: usize) -> u32 {
    let nir_src = (*(src_ptr.wrapping_add(idx))).src;
    let ssa = nir_src.ssa;
    (*ssa).index
}

unsafe fn print_alu_instr(inst: *mut nir_alu_instr) {
    let dest = (*inst).def;
    let src_ptr = (*inst).src.as_ptr() as *const nir_alu_src;

    // ALU instruction opcodes can be used to index nir_op_infos array
    let op_info = nir_op_infos[(*inst).op as usize];
    let op_name = std::ffi::CStr::from_ptr(op_info.name).to_str().unwrap();
    let src_count = op_info.num_inputs;

    // Maybe impl display for nir_alu_instr?
    let mut inst_str: String = format!("ALU {} = {:?}(", dest.index, op_name);
    for i in 0..src_count {
        inst_str += &get_alu_src_ssa_index(src_ptr, i as usize).to_string();
        if i < src_count - 1 {
            inst_str += ", ";
        }
    }
    inst_str += ")";
    eqaic_debugln!("{}", inst_str);
}

unsafe fn add_alu_instr(inst: *mut nir_alu_instr, build_context: &mut BuildContext) {
    let module = build_context.module;
    let context = build_context.context;
    let builder = build_context.builder;
    let llvm_var_map = &mut build_context.llvm_var_map;

    let dest = (*inst).def;
    let src_ptr = (*inst).src.as_ptr() as *const nir_alu_src;

    let i32_ty = LLVMInt32TypeInContext(context);

    // TODO: check src_count?
    match (*inst).op {
        nir_op_nir_op_imul => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src2_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 1))
                .unwrap();

            let name = cstr("imul");
            let product = LLVMBuildMul(builder, *src1_ref, *src2_ref, name.as_ptr());

            llvm_var_map.insert(dest.index, product);
        }
        nir_op_nir_op_iadd => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src2_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 1))
                .unwrap();

            let name = cstr("iadd");
            let sum = LLVMBuildAdd(builder, *src1_ref, *src2_ref, name.as_ptr());

            llvm_var_map.insert(dest.index, sum);
        }
        nir_op_nir_op_ult => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src2_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 1))
                .unwrap();

            let name = cstr("cmp");
            let cmp = LLVMBuildICmp(
                builder,
                LLVMIntPredicate::LLVMIntULT,
                *src1_ref,
                *src2_ref,
                name.as_ptr(),
            );

            llvm_var_map.insert(dest.index, cmp);
        }
        nir_op_nir_op_i2i64 => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let i64_ty = LLVMInt64TypeInContext(context);

            let name = cstr("i2i64");
            let v64 = LLVMBuildZExt(builder, *src1_ref, i64_ty, name.as_ptr());

            llvm_var_map.insert(dest.index, v64);
        }
        nir_op_nir_op_ishl => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src2_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 1))
                .unwrap();

            // TODO: hardcoding
            let i64_ty = LLVMInt64TypeInContext(context);
            let name = cstr("ext");
            let src2_64 = LLVMBuildZExt(builder, *src2_ref, i64_ty, name.as_ptr());

            let name = cstr("ishl");
            let shl = LLVMBuildShl(builder, *src1_ref, src2_64, name.as_ptr());

            llvm_var_map.insert(dest.index, shl);
        }
        nir_op_nir_op_fmul => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src2_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 1))
                .unwrap();

            let name = cstr("fmul");
            let product = LLVMBuildFMul(builder, *src1_ref, *src2_ref, name.as_ptr());

            llvm_var_map.insert(dest.index, product);
        }
        nir_op_nir_op_fadd => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src2_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 1))
                .unwrap();
            let name = cstr("fadd");
            let sum = LLVMBuildFAdd(builder, *src1_ref, *src2_ref, name.as_ptr());
            llvm_var_map.insert(dest.index, sum);
        }
        nir_op_nir_op_fsub => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src2_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 1))
                .unwrap();
            let name = cstr("fsub");
            let diff = LLVMBuildFSub(builder, *src1_ref, *src2_ref, name.as_ptr());
            llvm_var_map.insert(dest.index, diff);
        }
        nir_op_nir_op_fdiv => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src2_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 1))
                .unwrap();
            let name = cstr("fdiv");
            let quot = LLVMBuildFDiv(builder, *src1_ref, *src2_ref, name.as_ptr());
            llvm_var_map.insert(dest.index, quot);
        }
        nir_op_nir_op_fneg => {
            let src1_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let name = cstr("fneg");
            let neg = LLVMBuildFNeg(builder, *src1_ref, name.as_ptr());
            llvm_var_map.insert(dest.index, neg);
        }
        // unsigned integer cast to i32 (well u32, but keeping the semantics)
        // so works both ways; u64 -> i32 or u8 -> i32
        nir_op_nir_op_u2u32 => {
            let src_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let src_ty = LLVMTypeOf(*src_ref);
            let src_width = LLVMGetIntTypeWidth(src_ty);
            // if greater; truncate
            let result = if src_width > 32 {
                let name = cstr("u2u32");
                LLVMBuildTrunc(builder, *src_ref, i32_ty, name.as_ptr())
            // if smaller; zero extend
            } else if src_width < 32 {
                let name = cstr("u2u32");
                LLVMBuildZExt(builder, *src_ref, i32_ty, name.as_ptr())
            } else {
                *src_ref
            };
            llvm_var_map.insert(dest.index, result);
        }
        // unsigned integer cast to u64; u32 -> u64 or u8 -> u64
        nir_op_nir_op_u2u64 => {
            let src_ref = llvm_var_map
                .get(&get_alu_src_ssa_index(src_ptr, 0))
                .unwrap();
            let i64_ty = LLVMInt64TypeInContext(context);
            let src_ty = LLVMTypeOf(*src_ref);
            let src_width = LLVMGetIntTypeWidth(src_ty);
            // if greater; truncate
            let result = if src_width > 64 {
                let name = cstr("u2u64");
                LLVMBuildTrunc(builder, *src_ref, i64_ty, name.as_ptr())
            // if smaller; zero extend
            } else if src_width < 64 {
                let name = cstr("u2u64");
                LLVMBuildZExt(builder, *src_ref, i64_ty, name.as_ptr())
            } else {
                *src_ref
            };
            llvm_var_map.insert(dest.index, result);
        }
        _ => {
            eqaic_eprintln!("NIR op {} not supported.", (*inst).op);
            print_alu_instr(inst);

            // Store the destination as a poisoned value in the mapping
            // This ensures that looking up the destination in subsequent
            // instructions doesnt fault.
            let poison = LLVMGetPoison(i32_ty);
            llvm_var_map.insert(dest.index, poison);

            // Purposely create an invalid instruction to force the verifier
            // to fail.
            let fty = LLVMFunctionType(
                llvm_sys::core::LLVMVoidTypeInContext(context),
                std::ptr::null_mut(),
                0,
                0,
            );
            let f = LLVMAddFunction(module, b"always_bad\0".as_ptr() as *const i8, fty);
            LLVMAppendBasicBlockInContext(context, f, b"broken\0".as_ptr() as *const i8);
        }
    }
}

unsafe fn add_block(block: *mut nir_block, build_context: &mut BuildContext) {
    let builder = build_context.builder;

    let llvm_block = {
        let llvm_block_map = &mut build_context.llvm_block_map;
        llvm_block_map.get(&(*block).index).unwrap().clone()
    };
    LLVMPositionBuilderAtEnd(builder, llvm_block);

    let mut cur_param = 0;
    let mut inst_node = (*block).instr_list.head_sentinel.next;
    while !(*inst_node).next.is_null() {
        let inst = inst_node as *mut nir_instr;

        match (*(inst as *mut nir_instr)).type_ {
            nir_instr_type_nir_instr_type_alu => {
                let typed_instr = inst as *mut nir_instr as *mut nir_alu_instr;
                add_alu_instr(typed_instr, build_context);
            }
            nir_instr_type_nir_instr_type_deref => {
                let typed_instr = inst as *mut nir_instr as *mut nir_deref_instr;
                eqaic_debugln!("DEREF: {:?}", *typed_instr);
            }
            nir_instr_type_nir_instr_type_call => {
                let typed_instr = inst as *mut nir_instr as *mut nir_call_instr;
                eqaic_debugln!("CALL: {:?}", *typed_instr);
            }
            nir_instr_type_nir_instr_type_tex => {
                let typed_instr = inst as *mut nir_instr as *mut nir_tex_instr;
                eqaic_debugln!("TEX: {:?}", *typed_instr);
            }
            nir_instr_type_nir_instr_type_intrinsic => {
                let typed_instr = inst as *mut nir_instr as *mut nir_intrinsic_instr;
                add_intrinsic_instr(typed_instr, build_context, &mut cur_param);
            }
            nir_instr_type_nir_instr_type_load_const => {
                let typed_instr = inst as *mut nir_instr as *mut nir_load_const_instr;
                add_constant_load(typed_instr, build_context);
            }
            nir_instr_type_nir_instr_type_jump => {
                let typed_instr = inst as *mut nir_instr as *mut nir_jump_instr;
                eqaic_debugln!("JUMP: {:?}", *typed_instr);
            }
            nir_instr_type_nir_instr_type_undef => {
                let typed_instr = inst as *mut nir_instr as *mut nir_undef_instr;
                eqaic_debugln!("UNDEF: {:?}", *typed_instr);
            }
            nir_instr_type_nir_instr_type_phi => {
                let typed_instr = inst as *mut nir_instr as *mut nir_phi_instr;
                eqaic_debugln!("PHI: {:?}", *typed_instr);
            }
            _ => eqaic_eprintln!(
                "Unsupported nir_instr type: {}",
                (*(inst as *mut nir_instr)).type_
            ),
        }
        inst_node = (*inst_node).next;
    }
}

unsafe fn add_if(if_node: *mut nir_if, build_context: &mut BuildContext) {
    // Find condition
    let src: nir_src = (*if_node).condition;
    let src_ssa_index = (*src.ssa).index;
    let src_ref = {
        let llvm_var_map = &mut build_context.llvm_var_map;
        llvm_var_map.get(&src_ssa_index).unwrap().clone()
    };

    // Currently assuming the immediate then/else nodes are nir_blocks,
    // but what if they are other if/loop nodes?
    let mut then_node = (*if_node).then_list.head_sentinel.next;
    let then_cf_node = then_node as *mut nir_cf_node;
    assert!((*then_cf_node).type_ == nir_cf_node_type_nir_cf_node_block);
    let then_bb = {
        let block_id = (*(then_cf_node as *mut nir_block)).index;
        let llvm_block_map = &mut build_context.llvm_block_map;
        llvm_block_map.get(&block_id).unwrap().clone()
    };

    let mut else_node = (*if_node).else_list.head_sentinel.next;
    let else_cf_node = else_node as *mut nir_cf_node;
    assert!((*else_cf_node).type_ == nir_cf_node_type_nir_cf_node_block);
    let else_bb = {
        let block_id = (*(else_cf_node as *mut nir_block)).index;
        let llvm_block_map = &mut build_context.llvm_block_map;
        llvm_block_map.get(&block_id).unwrap().clone()
    };

    LLVMBuildCondBr(build_context.builder, src_ref, then_bb, else_bb);

    while !(*then_node).next.is_null() {
        add_cf_node_to_module(then_node as *mut nir_cf_node, build_context);
        then_node = (*then_node).next;
    }
    while !(*else_node).next.is_null() {
        add_cf_node_to_module(else_node as *mut nir_cf_node, build_context);
        else_node = (*else_node).next;
    }
}

unsafe fn add_cf_node_to_module(node: *mut nir_cf_node, build_context: &mut BuildContext) {
    match (*node).type_ {
        nir_cf_node_type_nir_cf_node_block => {
            add_block(node as *mut nir_block, build_context);
        }
        nir_cf_node_type_nir_cf_node_if => {
            add_if(node as *mut nir_if, build_context);
        }
        nir_cf_node_type_nir_cf_node_loop => {}
        nir_cf_node_type_nir_cf_node_function => {}
        _ => {
            eqaic_eprintln!("Unrecognized nir cf node type.");
        }
    }
}

unsafe fn build_kernel(nir: *mut nir_shader, build_context: &mut BuildContext) {
    let context = build_context.context;
    let module = build_context.module;
    let builder = build_context.builder;

    let void_ty = LLVMVoidTypeInContext(context);
    let i8_ty = LLVMInt8TypeInContext(context);
    let i16_ty = LLVMInt16TypeInContext(context);
    let i32_ty = LLVMInt32TypeInContext(context);
    let u64_ty = LLVMInt64TypeInContext(context);

    // Bagin parsing compute kernel
    let func_impl = nir_shader_get_entrypoint(nir);

    // Blocks must be indexed before traversing the CFG
    nir_index_ssa_defs(func_impl);
    nir_index_blocks(func_impl);

    // Get kernel input parameters
    let mut kernel_param_tys = Vec::new();

    // Should check for mode, but all params are marked as uniform and not inputs
    let _input_mode: nir_variable_mode = nir_variable_mode_nir_var_shader_in;
    let mut var_node = (*nir).variables.head_sentinel.next;
    while !(*var_node).next.is_null() {
        let var = var_node as *mut nir_variable;
        let type_: glsl_base_type = (*((*var).type_)).base_type();
        match type_ {
            glsl_base_type_GLSL_TYPE_UINT8 => {
                kernel_param_tys.push(i8_ty);
            }
            glsl_base_type_GLSL_TYPE_UINT16 => {
                kernel_param_tys.push(i16_ty);
            }
            glsl_base_type_GLSL_TYPE_UINT => {
                kernel_param_tys.push(i32_ty);
            }
            glsl_base_type_GLSL_TYPE_UINT64 => {
                kernel_param_tys.push(u64_ty);
            }
            _ => {
                eqaic_eprintln!("Unimplemented glsl type: {}", type_);
            }
        }
        var_node = (*var_node).next;
    }
    // Manually add additional field to track work item id
    kernel_param_tys.push(i32_ty);

    // TODO: currently assume kernel output is void (for now)
    let kernel_fn_ty = LLVMFunctionType(
        void_ty,
        kernel_param_tys.as_mut_ptr(),
        kernel_param_tys.len() as u32,
        0,
    );
    let kernel_name = cstr("kernel");
    let kernel = LLVMAddFunction(module, kernel_name.as_ptr(), kernel_fn_ty);
    build_context.kernel = kernel;
    build_context.kernel_param_types = kernel_param_tys;

    let mut block = (*func_impl).body.head_sentinel.next as *mut nir_block;
    while !block.is_null() {
        let block_id = (*block).index;
        let block_name = cstr(&format!("bb.{}", block_id));
        let llvm_block = LLVMAppendBasicBlockInContext(context, kernel, block_name.as_ptr());
        build_context.llvm_block_map.insert(block_id, llvm_block);
        block = nir_block_cf_tree_next(block);
    }

    // Based off of nir_foreach_block macro
    let mut block_cf = (*func_impl).body.head_sentinel.next;
    while !(*block_cf).next.is_null() {
        let cf_node = block_cf as *mut nir_cf_node;
        add_cf_node_to_module(cf_node, build_context);
        block_cf = (*block_cf).next;
    }

    let mut block = (*func_impl).body.head_sentinel.next as *mut nir_block;
    while !block.is_null() {
        let llvm_block = build_context
            .llvm_block_map
            .get(&(*block).index)
            .unwrap()
            .clone();
        let terminator = LLVMGetBasicBlockTerminator(llvm_block);
        // Block is not properly terminated, so add it
        if terminator.is_null() {
            let successor1: *mut nir_block = (*block).successors[0];
            let successor2: *mut nir_block = (*block).successors[1];
            // Final block, add return
            if successor1.is_null() && successor2.is_null() {
                LLVMPositionBuilderAtEnd(builder, llvm_block);
                LLVMBuildRetVoid(builder);
            }
            // Block has successor, go to it immediately
            else if successor2.is_null() {
                let successor1_id = (*successor1).index;
                // For some reason, there is an inaccessible BB at the end of the CFG
                // If there is one successor with no entry in the index, we can assume
                // we are currently at the final block.
                // Otherwise, if the index entry exists, add it as expected.
                if let Some(successor_bb) = build_context.llvm_block_map.get(&successor1_id) {
                    LLVMPositionBuilderAtEnd(builder, llvm_block);
                    LLVMBuildBr(builder, successor_bb.clone());
                } else {
                    LLVMPositionBuilderAtEnd(builder, llvm_block);
                    LLVMBuildRetVoid(builder);
                }
            }
            // Block has successors, but a terminator was not properly inserted
            else {
                eqaic_eprintln!("Unreachable!");
            }
        }
        block = nir_block_cf_tree_next(block);
    }
}

// Generate the harness function
// The JIT interface expects its entry point to have the following type:
//      (AicJitEntryPointConfig *, AicJitPointerArray *) -> uint32_t
// Before invoking the compute kernel, the harness must interpret each buffer in
// the pointer array as inputs to the compute kernel.
// In this example, the first two buffers represent float arrays, the third buffer
// holds the count constant, and the last buffer holds the work index.
unsafe fn build_harness(build_context: &mut BuildContext) {
    let context = build_context.context;
    let module = build_context.module;
    let builder = build_context.builder;
    let kernel = build_context.kernel;

    let i8_ty = LLVMInt8TypeInContext(context);
    let i16_ty = LLVMInt16TypeInContext(context);
    let i32_ty = LLVMInt32TypeInContext(context);
    let ptr_ty = LLVMPointerType(i8_ty, 0);
    let u64_ty = LLVMInt64TypeInContext(context);

    // struct AicJitPointerArray { i32, [0 x ptr] }
    let arr0_ty = LLVMArrayType2(ptr_ty, 0);

    let struct_name = cstr("struct.AicJitPointerArray");
    let struct_ty = LLVMStructCreateNamed(context, struct_name.as_ptr());

    let mut struct_fields = [i32_ty, arr0_ty];
    LLVMStructSetBody(struct_ty, struct_fields.as_mut_ptr(), 2, 0);

    let mut harness_param_tys = [ptr_ty, ptr_ty];
    let harness_fn_ty = LLVMFunctionType(i32_ty, harness_param_tys.as_mut_ptr(), 2, 0);

    let harness_name = cstr("harness");
    let harness = LLVMAddFunction(module, harness_name.as_ptr(), harness_fn_ty);

    // Harness only needs one basic block to interpret pointers and call the kernel.
    let h_entry = LLVMAppendBasicBlockInContext(context, harness, cstr("entry").as_ptr());

    LLVMPositionBuilderAtEnd(builder, h_entry);

    let entry_cfg = LLVMGetParam(harness, 0);
    let ptr_array = LLVMGetParam(harness, 1);

    let entry_cfg_addr = LLVMBuildAlloca(builder, ptr_ty, cstr("entryConfig.addr").as_ptr());
    let ptr_array_addr = LLVMBuildAlloca(builder, ptr_ty, cstr("pointerArray.addr").as_ptr());

    LLVMBuildStore(builder, entry_cfg, entry_cfg_addr);
    LLVMBuildStore(builder, ptr_array, ptr_array_addr);

    // helper for computing pointerArray->pointers[n]
    let zero = LLVMConstInt(i32_ty, 0, 0);
    let get_ptr = |idx: i32| -> LLVMValueRef {
        let base = LLVMBuildLoad2(builder, ptr_ty, ptr_array_addr, cstr("").as_ptr());

        let gep1 = LLVMBuildInBoundsGEP2(
            builder,
            struct_ty,
            base,
            [zero, LLVMConstInt(i32_ty, 1, 0)].as_ptr() as *mut _,
            2,
            cstr("").as_ptr(),
        );

        let gep2 = LLVMBuildInBoundsGEP2(
            builder,
            arr0_ty,
            gep1,
            [zero, LLVMConstInt(i32_ty, idx as u64, 0)].as_ptr() as *mut _,
            2,
            cstr("").as_ptr(),
        );

        LLVMBuildLoad2(builder, ptr_ty, gep2, cstr("").as_ptr())
    };

    // From here, generate casts of AicJitPointer's to desired type for each kernel parameter.

    // TODO: figure out how to use LLVMGetParamTypes for this
    let kernel_types = &build_context.kernel_param_types;

    // Determine the number of total JIT buffers sent to the device.
    // Count every global buffer referenced, plus the constant buffer,
    // plus the workgroup information buffer.
    //
    // Cannot rely purely on the ordering of input parameters: the
    // kernel may never use the constant buffer, for example.
    //
    // TODO: find an easier way to determine the total number of jit buffers
    // to be sent to the device.
    let mut num_jit_buffers = 0;
    for param_type in kernel_types {
        // Assuming u64s are pointers to global memory, i.e., memory
        // buffers backed by a jit buffer
        if *param_type == u64_ty {
            num_jit_buffers += 1;
        }
    }
    // In addition to global buffers, include constant and workgroup buffers.
    num_jit_buffers += 2;

    // Vector to hold the LLVMValueRef for each parameter to pass
    let mut call_args = Vec::new();

    // After the global buffers are passed, the constant buffer is passed.
    // Kernel parameters are contained in this constant buffer.
    //
    // Theoretically, all kernel parameter values should be passed in the constant
    // buffer, including pointers to global buffers. However, since the JitDevice
    // retains physical address handles for memory objects that are remapped by the
    // qshim at runtime, pointer values in the constant buffer point to (invalid)
    // physical addresses.
    //
    // Instead, we must iterate past pointer values store in the constant buffer.
    // To do so, we assume that any u64 kernel parameter type is a pointer to global
    // memory.
    //
    // For every kernel parameter, we maintain an offset into the constant buffer
    // based on all of the types we have seen up to this point. This allows us to
    // account for types of varying size, e.g., u64 being 8B while u32 is only 4.
    let mut cb_offset = 0;

    // All data buffers are passed sequentially first, followed by the constant
    // buffer, and finally by the workgroup ID.
    let cb_ptr = get_ptr(num_jit_buffers - 2);

    // Keep track of the current data buffer
    let mut data_ptr_idx = 0;

    // Dont automatically parse the final value.
    // That is reserved for work group information and will be added at the end.
    for param_type in &kernel_types[..kernel_types.len() - 1] {
        // Assuming u64 types are pointers.
        // Since pointers cannot be stored properly in the constant buffer,
        // we must take the address of the JIT pointer, i.e.,
        //      &(pointerArray->pointers[param_idx];
        // without dereferencing.
        //
        // TODO: if it becomes possible to insert VAs into the constant buffer
        // during set_global_bindings, this logic can be dropped, and u64 values
        // can be handled normally by indexing straight into the constant buffer
        // like the other types beflow. This is definitely the proper method!!
        if *param_type == u64_ty {
            let data_ptr = get_ptr(data_ptr_idx);
            let val = LLVMBuildPtrToInt(builder, data_ptr, u64_ty, cstr("").as_ptr());
            call_args.push(val);
            cb_offset += 8;
            data_ptr_idx += 1;
        }
        // Non pointer type, must retrieve value from the constant buffer.
        // The constant buffer offset is treated as a byte array and indexed with
        // a byte offset. Therefore, we must first perform a GEP to get an i8 pointer,
        // then cast this pointer to a pointer of the desired type before we can
        // finally dereference and load the value
        else if *param_type == i32_ty || *param_type == i8_ty || *param_type == i16_ty {
            let cb_gep = LLVMBuildInBoundsGEP2(
                builder,
                i8_ty,
                cb_ptr,
                [LLVMConstInt(i8_ty, cb_offset, 0)].as_ptr() as *mut _,
                1,
                cstr("").as_ptr(),
            );
            let ptr_cast_type = LLVMPointerType(*param_type, 0);
            let casted_ptr = LLVMBuildBitCast(builder, cb_gep, ptr_cast_type, cstr("").as_ptr());
            let val = LLVMBuildLoad2(builder, *param_type, casted_ptr, cstr("").as_ptr());
            call_args.push(val);

            cb_offset += (LLVMGetIntTypeWidth(*param_type) / 8) as u64;
        }
        // TODO: handle other types
        else {
            eqaic_eprintln!("Unsupported kernel parameter!");
        }
    }

    // Calculate and append the final global pointer
    // global_id = wg_id * (num_cores * threads_per_nsp) + core_id * threads_per_nsp + thread_id
    // so each thread within each NSP processes a distinct element per execute_operator call.
    let ptr = get_ptr(num_jit_buffers - 1);
    let wg_id = LLVMBuildLoad2(builder, i32_ty, ptr, cstr("wg_id").as_ptr());

    // Load core_id from entryConfig at byte offset 0
    // From AicJitEntryPointConfig:
    // (core_id=0, num_cores=4, thread_id=8, num_treads=12)
    let core_id_gep = LLVMBuildInBoundsGEP2(
        builder,
        i8_ty,
        entry_cfg,
        [LLVMConstInt(i32_ty, 0, 0)].as_ptr() as *mut _,
        1,
        cstr("core_id_ptr").as_ptr(),
    );
    let core_id_ptr = LLVMBuildBitCast(
        builder,
        core_id_gep,
        LLVMPointerType(i32_ty, 0),
        cstr("core_id_casted").as_ptr(),
    );
    let core_id = LLVMBuildLoad2(builder, i32_ty, core_id_ptr, cstr("core_id").as_ptr());

    // Load num_cores from entryConfig at byte offset 4
    let num_cores_gep = LLVMBuildInBoundsGEP2(
        builder,
        i8_ty,
        entry_cfg,
        [LLVMConstInt(i32_ty, 4, 0)].as_ptr() as *mut _,
        1,
        cstr("num_cores_ptr").as_ptr(),
    );
    let num_cores_ptr = LLVMBuildBitCast(
        builder,
        num_cores_gep,
        LLVMPointerType(i32_ty, 0),
        cstr("num_cores_casted").as_ptr(),
    );
    let num_cores = LLVMBuildLoad2(builder, i32_ty, num_cores_ptr, cstr("num_cores").as_ptr());

    // Load thread_id from entryConfig at byte offset 8
    let thread_id_gep = LLVMBuildInBoundsGEP2(
        builder,
        i8_ty,
        entry_cfg,
        [LLVMConstInt(i32_ty, 8, 0)].as_ptr() as *mut _,
        1,
        cstr("thread_id_ptr").as_ptr(),
    );
    let thread_id_ptr = LLVMBuildBitCast(
        builder,
        thread_id_gep,
        LLVMPointerType(i32_ty, 0),
        cstr("thread_id_casted").as_ptr(),
    );
    let thread_id = LLVMBuildLoad2(builder, i32_ty, thread_id_ptr, cstr("thread_id").as_ptr());

    // Load threads_per_nsp from entryConfig at byte offset 12
    let threads_per_nsp_gep = LLVMBuildInBoundsGEP2(
        builder,
        i8_ty,
        entry_cfg,
        [LLVMConstInt(i32_ty, 12, 0)].as_ptr() as *mut _,
        1,
        cstr("threads_per_nsp_gep").as_ptr(),
    );
    let threads_per_nsp_ptr = LLVMBuildBitCast(
        builder,
        threads_per_nsp_gep,
        LLVMPointerType(i32_ty, 0),
        cstr("threads_per_nsp_ptr").as_ptr(),
    );
    let threads_per_nsp = LLVMBuildLoad2(
        builder,
        i32_ty,
        threads_per_nsp_ptr,
        cstr("threads_per_nsp").as_ptr(),
    );

    // TODO: Explore if there is a better way to get all these in single batch..

    // we should be able to get the global id:
    // global_id = wg_id * total_per_dispatch + thread_offset
    // where thread_offset = core_id * threads_per_nsp + thread_id
    // where total_per_dispatch = num_cores * threads_per_nsp

    // total_per_dispatch = num_cores * threads_per_nsp
    let total_per_dispatch = LLVMBuildMul(
        builder,
        num_cores,
        threads_per_nsp,
        cstr("total_per_dispatch").as_ptr(),
    );

    // wg_base = wg_id * total_per_dispatch
    let wg_base = LLVMBuildMul(builder, wg_id, total_per_dispatch, cstr("wg_base").as_ptr());

    // nsp_offset = core_id * threads_per_nsp
    let nsp_offset = LLVMBuildMul(builder, core_id, threads_per_nsp, cstr("nsp_off").as_ptr());

    // thread_offset = core_id * threads_per_nsp + thread_id
    // thread_offset = nsp_offset + thread_id
    let thread_offset = LLVMBuildAdd(builder, nsp_offset, thread_id, cstr("thread_off").as_ptr());

    // global_id = wg_id * total_per_dispatch + thread_offset
    // global_id = wg_base + thread_offset
    let global_id = LLVMBuildAdd(builder, wg_base, thread_offset, cstr("global_id").as_ptr());

    // push global_id
    call_args.push(global_id);
    assert!(call_args.len() == kernel_types.len());

    let kernel_fn_ty = LLVMGlobalGetValueType(kernel);
    LLVMBuildCall2(
        builder,
        kernel_fn_ty,
        kernel,
        call_args.as_ptr() as *mut _,
        call_args.len() as u32,
        cstr("").as_ptr(),
    );

    LLVMBuildRet(builder, LLVMConstInt(i32_ty, 0, 0));
}

pub unsafe fn lower_nir(context: LLVMContextRef, nir: *mut nir_shader) -> LLVMModuleRef {
    // nir_print_shader_stdout(nir);

    let entry_name_ptr = (*(*nir_shader_get_entrypoint(nir)).function).name;
    let module_name = if entry_name_ptr.is_null() {
        cstr("opencl_kernel")
    } else {
        CString::new(CStr::from_ptr(entry_name_ptr).to_bytes()).unwrap()
    };
    let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);
    let builder = LLVMCreateBuilderInContext(context);
    let mut build_context = BuildContext {
        module: module,
        context: context,
        builder: builder,
        kernel: std::ptr::null_mut(),
        kernel_param_types: Vec::new(),
        llvm_var_map: HashMap::new(),
        llvm_block_map: HashMap::new(),
    };

    build_kernel(nir, &mut build_context);
    build_harness(&mut build_context);
    LLVMDisposeBuilder(builder);

    return module;
}
