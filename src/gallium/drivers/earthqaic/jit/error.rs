/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use std::fmt;

use crate::jit::mem::MemError;

#[derive(Debug)]
pub enum JitError {
    MemoryError(String),
    ExecutionError(String),
    ReadError(String),
    WriteError(String),
    MiscError(String),
    CompileError(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::MemoryError(msg) => write!(f, "Memory allocation failed: {}", msg),
            JitError::ExecutionError(msg) => write!(f, "Operator execution failed: {}", msg),
            JitError::ReadError(msg) => write!(f, "Device read failed: {}", msg),
            JitError::WriteError(msg) => write!(f, "Device write failed: {}", msg),
            JitError::MiscError(msg) => write!(f, "JIT error: {}", msg),
            JitError::CompileError(msg) => write!(f, "Operator compilation failed: {}", msg),
        }
    }
}

impl From<MemError> for JitError {
    fn from(err: MemError) -> JitError {
        match err {
            MemError::InitializationError => JitError::MemoryError(format!("{}", err)),
            MemError::AllocationError => JitError::MemoryError(format!("{}", err)),
        }
    }
}

impl std::error::Error for JitError {}
