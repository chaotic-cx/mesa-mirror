/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::ffi::CString;

use crate::earthqaic_ml::EarthqaicMLSubgraph;
use earthqaic_rs_bindings::*;

use llvm_sys::core::*;
use llvm_sys::prelude::*;

pub unsafe fn lower_subgraph(
    context: LLVMContextRef,
    subgraph: &EarthqaicMLSubgraph,
) -> LLVMModuleRef {
    let module_name = CString::new("teflon_subgraph").unwrap();
    let module = LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context);
    let builder = LLVMCreateBuilderInContext(context);

    let i32_ty = LLVMInt32TypeInContext(context);
    let i8_ty = LLVMInt8TypeInContext(context);
    let f64_ty = LLVMDoubleTypeInContext(context);
    let ptr_ty = LLVMPointerType(i8_ty, 0);
    let void_ty = LLVMVoidTypeInContext(context);

    let i32_ptr = LLVMPointerType(i32_ty, 0);
    let f64_ptr = LLVMPointerType(f64_ty, 0);
    let i8_ptr = LLVMPointerType(i8_ty, 0);
    let i8_ptr_ptr = LLVMPointerType(i8_ptr, 0);

    // struct AicJitPointerArray { i32, [0 x ptr] }
    let arr0_ty = LLVMArrayType2(ptr_ty, 0);

    let struct_name = CString::new("struct.AicJitPointerArray").unwrap();
    let struct_ty = LLVMStructCreateNamed(context, struct_name.as_ptr());

    let mut struct_fields = [i32_ty, arr0_ty];
    LLVMStructSetBody(struct_ty, struct_fields.as_mut_ptr(), 2, 0);

    // Define struct tensor_list and struct tensor
    let tensor_list_name = CString::new("struct.tensor_list").unwrap();
    let tensor_list_ty = LLVMStructCreateNamed(context, tensor_list_name.as_ptr());
    let tensor_list_ptr_ty = LLVMPointerType(tensor_list_ty, 0);

    let mut tensor_list_fields = [
        i32_ptr,    // sizes
        i32_ptr,    // widths
        i32_ptr,    // heights
        i32_ptr,    // depths
        i32_ptr,    // zero_points
        f64_ptr,    // scales
        i8_ptr_ptr, // data
    ];
    LLVMStructSetBody(
        tensor_list_ty,
        tensor_list_fields.as_mut_ptr(),
        tensor_list_fields.len() as u32,
        0, // packed = false
    );

    let tensor_name = CString::new("struct.tensor").unwrap();
    let tensor_ty = LLVMStructCreateNamed(context, tensor_name.as_ptr());

    let mut tensor_fields = [
        i32_ty, // size (uint32_t)
        i32_ty, // width
        i32_ty, // height
        i32_ty, // depth
        i32_ty, // zero_point (int32_t)
        f64_ty, // scale
        i8_ptr, // data (uint8_t*)
    ];
    LLVMStructSetBody(
        tensor_ty,
        tensor_fields.as_mut_ptr(),
        tensor_fields.len() as u32,
        0, // packed = false
    );

    let mut subgraph_param_tys = [ptr_ty, ptr_ty];
    let subgraph_fn_ty = LLVMFunctionType(i32_ty, subgraph_param_tys.as_mut_ptr(), 2, 0);

    let subgraph_name = CString::new("subgraph").unwrap();
    let subgraph_fn = LLVMAddFunction(module, subgraph_name.as_ptr(), subgraph_fn_ty);

    // Harness only needs one basic block to interpret pointers and call the kernel.
    let subgraph_entry = LLVMAppendBasicBlockInContext(
        context,
        subgraph_fn,
        CString::new("entry").unwrap().as_ptr(),
    );

    LLVMPositionBuilderAtEnd(builder, subgraph_entry);

    let entry_cfg = LLVMGetParam(subgraph_fn, 0);
    let ptr_array = LLVMGetParam(subgraph_fn, 1);

    let entry_cfg_addr = LLVMBuildAlloca(
        builder,
        ptr_ty,
        CString::new("entryConfig.addr").unwrap().as_ptr(),
    );
    let ptr_array_addr = LLVMBuildAlloca(
        builder,
        ptr_ty,
        CString::new("pointerArray.addr").unwrap().as_ptr(),
    );

    LLVMBuildStore(builder, entry_cfg, entry_cfg_addr);
    LLVMBuildStore(builder, ptr_array, ptr_array_addr);

    // Call to init_tensor_list
    let mut init_param_tys = [tensor_list_ptr_ty, i32_ty, i32_ty, ptr_ty];
    let init_fn_ty = LLVMFunctionType(void_ty, init_param_tys.as_mut_ptr(), 4, 0);
    let init_fn_name = CString::new("init_tensor_list").unwrap();
    let init_fn = LLVMAddFunction(module, init_fn_name.as_ptr(), init_fn_ty);

    let weights_name = CString::new("weights").unwrap();
    let weights_ptr = LLVMBuildAlloca(builder, tensor_list_ty, weights_name.as_ptr());
    let biases_name = CString::new("biases").unwrap();
    let biases_ptr = LLVMBuildAlloca(builder, tensor_list_ty, biases_name.as_ptr());

    // Subgraph has weights and biases
    if (*subgraph).weights.num > 0 {
        // call init_tensor_list for weights:
        //  - struct tensor_list *list = &weights
        //  - int start = 0
        //  - int num = (*subgraph).weights.num
        //  - AicJitPointerArray *pointerArray = pointerArray
        let mut init_args_weight = [
            weights_ptr,
            LLVMConstInt(i32_ty, 0, 0),
            LLVMConstInt(i32_ty, (*subgraph).weights.num as u64, 0),
            ptr_array,
        ];
        let call_name = CString::new("").unwrap();
        LLVMBuildCall2(
            builder,
            init_fn_ty,
            init_fn,
            init_args_weight.as_mut_ptr(),
            init_args_weight.len() as u32,
            call_name.as_ptr(),
        );

        // call init_tensor_list for biases:
        //  - struct tensor_list *list = &biases
        //  - int start = (*subgraph).weights.num + 6
        //  - int num = (*subgraph).biases.num
        //  - AicJitPointerArray *pointerArray = pointerArray
        let mut init_args_bias = [
            biases_ptr,
            LLVMConstInt(i32_ty, (*subgraph).weights.num as u64 + 6, 0),
            LLVMConstInt(i32_ty, (*subgraph).biases.num as u64, 0),
            ptr_array,
        ];
        let call_name = CString::new("").unwrap();
        LLVMBuildCall2(
            builder,
            init_fn_ty,
            init_fn,
            init_args_bias.as_mut_ptr(),
            init_args_bias.len() as u32,
            call_name.as_ptr(),
        );
    }

    let io_name = CString::new("io").unwrap();
    let io_ptr = LLVMBuildAlloca(builder, tensor_list_ty, io_name.as_ptr());

    let io_start = if (*subgraph).weights.num == 0 {
        0
    } else {
        (*subgraph).weights.num as u64 + 6 + (*subgraph).biases.num as u64 + 6
    };

    // call init_tensor_list for io:
    //  - struct tensor_list *list = &io
    //  - int start = io_start
    //  - int num = (*subgraph).io.num
    //  - AicJitPointerArray *pointerArray = pointerArray
    let mut init_args_io = [
        io_ptr,
        LLVMConstInt(i32_ty, io_start, 0),
        LLVMConstInt(i32_ty, (*subgraph).io_tensors.num as u64, 0),
        ptr_array,
    ];
    let call_name = CString::new("").unwrap();
    LLVMBuildCall2(
        builder,
        init_fn_ty,
        init_fn,
        init_args_io.as_mut_ptr(),
        init_args_io.len() as u32,
        call_name.as_ptr(),
    );

    // Call to tensor_add
    let mut add_param_tys = [tensor_list_ptr_ty, i32_ty, i32_ty, i32_ty];
    let add_fn_ty = LLVMFunctionType(void_ty, add_param_tys.as_mut_ptr(), 4, 0);
    let add_fn_name = CString::new("tensor_add").unwrap();
    let add_fn = LLVMAddFunction(module, add_fn_name.as_ptr(), add_fn_ty);

    // Call to tensor_conv
    let mut conv_param_tys = [
        tensor_list_ptr_ty,
        i32_ty,
        i32_ty,
        tensor_list_ptr_ty,
        i32_ty,
        tensor_list_ptr_ty,
        i32_ty,
    ];
    let conv_fn_ty = LLVMFunctionType(void_ty, conv_param_tys.as_mut_ptr(), 7, 0);
    let conv_fn_name = CString::new("tensor_conv").unwrap();
    let conv_fn = LLVMAddFunction(module, conv_fn_name.as_ptr(), conv_fn_ty);

    // Now, insert calls to each subgraph operation
    for operation in (*subgraph).operations.iter() {
        if operation.type_ == pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_ADD {
            let input1 = (*subgraph)
                .io_tensors
                .tensors
                .get(&operation.inputs[0])
                .unwrap()
                .dev_id;
            let input2 = (*subgraph)
                .io_tensors
                .tensors
                .get(&operation.inputs[1])
                .unwrap()
                .dev_id;
            let output = (*subgraph)
                .io_tensors
                .tensors
                .get(&operation.outputs[0])
                .unwrap()
                .dev_id;

            let mut add_args = [
                io_ptr,
                LLVMConstInt(i32_ty, input1 as u64, 0),
                LLVMConstInt(i32_ty, input2 as u64, 0),
                LLVMConstInt(i32_ty, output as u64, 0),
            ];
            let call_name = CString::new("").unwrap();
            LLVMBuildCall2(
                builder,
                add_fn_ty,
                add_fn,
                add_args.as_mut_ptr(),
                add_args.len() as u32,
                call_name.as_ptr(),
            );
        } else if operation.type_ == pipe_ml_operation_type_PIPE_ML_OPERATION_TYPE_CONVOLUTION {
            let input = (*subgraph)
                .io_tensors
                .tensors
                .get(&operation.inputs[0])
                .unwrap()
                .dev_id;
            let output = (*subgraph)
                .io_tensors
                .tensors
                .get(&operation.outputs[0])
                .unwrap()
                .dev_id;
            let weight = (*subgraph)
                .weights
                .tensors
                .get(&operation.weight)
                .unwrap()
                .dev_id;
            let bias = (*subgraph)
                .biases
                .tensors
                .get(&operation.bias)
                .unwrap()
                .dev_id;

            let mut conv_args = [
                io_ptr,
                LLVMConstInt(i32_ty, input as u64, 0),
                LLVMConstInt(i32_ty, output as u64, 0),
                weights_ptr,
                LLVMConstInt(i32_ty, weight as u64, 0),
                biases_ptr,
                LLVMConstInt(i32_ty, bias as u64, 0),
            ];
            let call_name = CString::new("").unwrap();
            LLVMBuildCall2(
                builder,
                conv_fn_ty,
                conv_fn,
                conv_args.as_mut_ptr(),
                conv_args.len() as u32,
                call_name.as_ptr(),
            );
        } else {
            eqaic_eprintln!("generate_subgraph: Unsupported operation!");
        }
    }

    LLVMBuildRet(builder, LLVMConstInt(i32_ty, 0, 0));

    module
}
