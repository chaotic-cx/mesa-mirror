// Copyright © 2026 Igalia, S.L.
// SPDX-License-Identifier: MIT

//! Bindgen output for mesa_util, plus any impls on the generated types that
//! the orphan rule requires to live in the same crate (inherent methods and
//! foreign traits like Display).

mod util_bindings;

pub use util_bindings::*;

use std::ffi::CStr;
use std::fmt;

impl pipe_format {
    /// Returns the short name for the format (e.g. "r8g8b8a8_unorm").
    pub fn short_name(&self) -> &'static str {
        // SAFETY: The format names are static (UTF-8) C strings in the generated u_format_table.c.
        unsafe {
            CStr::from_ptr(util_format_short_name(*self))
                .to_str()
                .unwrap()
        }
    }
}

impl TryFrom<u32> for pipe_format {
    type Error = &'static str;

    fn try_from(f: u32) -> Result<pipe_format, Self::Error> {
        if f < PIPE_FORMAT_COUNT {
            // SAFETY: pipe_format is repr(u32), variants are contiguous, and
            // PIPE_FORMAT_COUNT is the number of variants.
            Ok(unsafe { std::mem::transmute(f) })
        } else {
            Err("pipe_format >= PIPE_FORMAT_COUNT")
        }
    }
}

impl fmt::Display for pipe_format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short_name())
    }
}
