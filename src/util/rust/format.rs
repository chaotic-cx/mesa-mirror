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

    #[test]
    fn srgb_roundtrip() {
        for i in 0..PIPE_FORMAT_COUNT {
            let format = pipe_format::try_from(i).unwrap();

            if format.is_srgb() {
                let linear = format.to_linear();
                assert!(format != linear);
                assert_eq!(format, linear.to_srgb());
            } else {
                assert_eq!(format.to_linear(), format);
            }
        }
    }

    #[test]
    fn formats_compatible() {
        for i in 0..PIPE_FORMAT_COUNT {
            let src = pipe_format::try_from(i).unwrap();
            for j in 0..PIPE_FORMAT_COUNT {
                let dst = pipe_format::try_from(j).unwrap();

                let expected = match (src, dst) {
                    (PIPE_FORMAT_A8R8G8B8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM) => Some(false),
                    (PIPE_FORMAT_R8G8B8X8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM) => Some(false),
                    (PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8X8_SRGB) => Some(false),

                    (PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8X8_UNORM) => Some(true),
                    (PIPE_FORMAT_L8_UNORM, PIPE_FORMAT_A8_UNORM) => Some(false),
                    (PIPE_FORMAT_L8_UNORM, PIPE_FORMAT_R8_UNORM) => Some(true),
                    (PIPE_FORMAT_I8_UNORM, PIPE_FORMAT_R8_UNORM) => Some(true),
                    (PIPE_FORMAT_I8_UNORM, PIPE_FORMAT_A8_UNORM) => Some(true),
                    (PIPE_FORMAT_I8_UNORM, PIPE_FORMAT_L8_UNORM) => Some(true),
                    (PIPE_FORMAT_I16_UNORM, PIPE_FORMAT_L16_UNORM) => Some(true),
                    (PIPE_FORMAT_Z24_UNORM_S8_UINT, PIPE_FORMAT_Z24X8_UNORM) => Some(true),
                    (PIPE_FORMAT_S8_UINT_Z24_UNORM, PIPE_FORMAT_X8Z24_UNORM) => Some(true),
                    (PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_R8G8B8X8_SRGB) => Some(true),
                    (PIPE_FORMAT_R10G10B10A2_USCALED, PIPE_FORMAT_R10G10B10X2_USCALED) => {
                        Some(true)
                    }
                    (PIPE_FORMAT_R10SG10SB10SA2U_NORM, PIPE_FORMAT_R10G10B10X2_SNORM) => Some(true),
                    _ => None,
                };

                let compatible = src.compatible(dst);

                // note: run test with --no-capture to see this output.
                if compatible {
                    println!("{} -> {}", src, dst);
                }

                if let Some(expected) = expected {
                    assert_eq!(compatible, expected, "{} -> {}", src, dst);
                }
            }
        }
    }
}
