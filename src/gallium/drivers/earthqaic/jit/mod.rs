/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

extern crate elf;
extern crate jit_generated;

mod compiler;
pub mod device;
mod error;
mod flatbuf;
mod mem;
mod nir;
mod operators;
mod subgraph_llvm;
