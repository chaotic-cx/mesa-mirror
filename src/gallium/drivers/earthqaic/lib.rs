/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#![allow(non_upper_case_globals)]
extern crate earthqaic_rs_bindings;
extern crate libc;

// Tags every error/failure message with the driver name so it's easy to
// grep/filter this driver's output from Mesa's shared stderr.
macro_rules! eqaic_eprintln {
    ($($arg:tt)*) => {
        eprintln!("[earthqaic] {}", format!($($arg)*))
    };
}

// Checked once and cached: set EARTHQAIC_DEBUG to enable eqaic_debugln! output.
fn debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("EARTHQAIC_DEBUG").is_ok())
}

// Like eqaic_eprintln!, but for verbose tracing that should stay silent
// unless EARTHQAIC_DEBUG is set.
macro_rules! eqaic_debugln {
    ($($arg:tt)*) => {
        if crate::debug_enabled() {
            println!("[earthqaic] {}", format!($($arg)*))
        }
    };
}

pub use earthqaic_driver_dirs::{DRIVER_BUILD_DIR, DRIVER_INSTALL_DIR};

// Allows runtime artifacts (qshim binaries, operators.o) to be found
// regardless of how the driver is being run:
//   1. EARTHQAIC_PATH, if set, overrides everything below with a direct
//      path to the artifacts.
//   2. The build dir, if it still exists on disk -- this is the normal
//      build-then-run-in-place dev loop, and takes priority over an
//      installed copy so a fresh rebuild is never shadowed by stale
//      artifacts left over from an earlier `ninja install`.
//   3. The installed dir, for when the build tree is gone (cleaned up,
//      or the artifacts were packaged/shipped to a machine that never
//      had it).
pub fn driver_dir() -> String {
    if let Ok(dir) = std::env::var("EARTHQAIC_PATH") {
        return dir;
    }
    if std::path::Path::new(DRIVER_BUILD_DIR).exists() {
        return DRIVER_BUILD_DIR.to_string();
    }
    DRIVER_INSTALL_DIR.to_string()
}

pub fn qshim_path(file: &str) -> String {
    format!("{}/qshim/{}", driver_dir(), file)
}

pub mod earthqaic_device;

mod earthqaic_compute;
mod earthqaic_ml;
mod jit;
mod qaic;
