// Copyright © 2026 Igalia, S.L.
// SPDX-License-Identifier: MIT

use mesa_util_bindings::*;
pub use mesa_util_bindings::{
    pipe_format, util_format_colorspace, util_format_description, util_format_layout,
};

/// Helper trait for safe access to mesa's C enum pipe_format and associated
/// utility functions.
///
/// This is located outside of the bindings crate so that it doesn't have to be
/// in a meson structured_sources() that breaks compiler error reporting.
pub trait PipeFormat {
    fn description(&self) -> &'static util_format_description;
    fn layout(&self) -> util_format_layout;
    fn has_alpha(&self) -> bool;
    fn has_depth(&self) -> bool;
    fn has_stencil(&self) -> bool;
    fn is_depth_or_stencil(&self) -> bool;
    fn is_depth_and_stencil(&self) -> bool;
    fn is_float(&self) -> bool;
    fn is_unorm(&self) -> bool;
    fn is_snorm(&self) -> bool;
    fn is_snorm8(&self) -> bool;
    fn is_scaled(&self) -> bool;
    fn is_integer(&self) -> bool;
    fn is_uint(&self) -> bool;
    fn is_sint(&self) -> bool;
    fn is_srgb(&self) -> bool;

    /// Returns the size in bytes of a block (not a pixel).
    fn blocksize(&self) -> u32;

    /// Returns the size in bits of a block (not a pixel).
    fn blocksizebits(&self) -> u32;

    /// Returns the number of blocks to cover @width pixels.
    fn nblocksx(&self, width: u32) -> u32;

    /// Returns the number of blocks to cover @height pixels.
    fn nblocksy(&self, height: u32) -> u32;

    /// Returns the equivalent unorm format for snorm formats, otherwise the original
    /// format.
    fn snorm_to_unorm(&self) -> Self;

    /// Returns the equivalent linear format for sRGB formats, otherwise the original
    /// format.
    fn to_linear(&self) -> Self;

    /// Returns the equivalent sRGB format for sRGB-compatible formats, otherwise
    /// PIPE_FORMAT_NONE.
    fn to_srgb(&self) -> Self;

    /// Returns true if self can be correctly copied to dst as a memcpy.
    fn compatible(&self, dst: pipe_format) -> bool;
}

impl PipeFormat for pipe_format {
    fn description(&self) -> &'static util_format_description {
        unsafe { &*util_format_description(*self) }
    }

    fn layout(&self) -> util_format_layout {
        self.description().layout
    }

    fn has_alpha(&self) -> bool {
        unsafe { util_format_has_alpha(*self) }
    }

    fn has_depth(&self) -> bool {
        unsafe { util_format_has_depth(self.description()) }
    }

    fn has_stencil(&self) -> bool {
        unsafe { util_format_has_stencil(self.description()) }
    }

    fn is_depth_or_stencil(&self) -> bool {
        self.has_depth() || self.has_stencil()
    }

    fn is_depth_and_stencil(&self) -> bool {
        self.has_depth() && self.has_stencil()
    }

    fn is_float(&self) -> bool {
        unsafe { util_format_is_float(*self) }
    }

    fn is_unorm(&self) -> bool {
        unsafe { util_format_is_unorm(*self) }
    }

    fn is_snorm(&self) -> bool {
        unsafe { util_format_is_snorm(*self) }
    }

    fn is_snorm8(&self) -> bool {
        unsafe { util_format_is_snorm8(*self) }
    }

    fn is_scaled(&self) -> bool {
        unsafe { util_format_is_scaled(*self) }
    }

    fn is_integer(&self) -> bool {
        unsafe { util_format_is_pure_integer(*self) }
    }

    fn is_uint(&self) -> bool {
        unsafe { util_format_is_pure_uint(*self) }
    }

    fn is_sint(&self) -> bool {
        unsafe { util_format_is_pure_sint(*self) }
    }

    fn is_srgb(&self) -> bool {
        unsafe { util_format_is_srgb(*self) }
    }

    fn blocksize(&self) -> u32 {
        unsafe { util_format_get_blocksize(*self) }
    }

    fn blocksizebits(&self) -> u32 {
        unsafe { util_format_get_blocksizebits(*self) }
    }

    fn nblocksx(&self, width: u32) -> u32 {
        unsafe { util_format_get_nblocksx(*self, width) }
    }

    fn nblocksy(&self, height: u32) -> u32 {
        unsafe { util_format_get_nblocksy(*self, height) }
    }

    fn snorm_to_unorm(&self) -> pipe_format {
        unsafe { util_format_snorm_to_unorm(*self) }
    }

    fn to_linear(&self) -> pipe_format {
        unsafe { util_format_linear(*self) }
    }

    fn to_srgb(&self) -> pipe_format {
        unsafe { util_format_srgb(*self) }
    }

    fn compatible(&self, dst: pipe_format) -> bool {
        unsafe { util_is_format_compatible(self.description(), dst.description()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipe_format::*;

    #[test]
    fn is_srgb() {
        assert!(PIPE_FORMAT_R8G8B8A8_SRGB.is_srgb());
        assert!(!PIPE_FORMAT_R8G8B8A8_UNORM.is_srgb());
    }
}
