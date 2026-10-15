// Copyright © 2024 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use mesa_util::format::pipe_format::*;
use mesa_util::format::util_format_layout::*;
use mesa_util::format::*;
use nil_rs_bindings::*;
use nvidia_headers::classes::{cla297, clb097, clb197};

use crate::extent::{units, Extent4D};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    p_format: pipe_format,
}

impl TryFrom<pipe_format> for Format {
    type Error = &'static str;

    fn try_from(p_format: pipe_format) -> Result<Self, Self::Error> {
        unsafe {
            if nil_format_table[p_format as usize].support() == 0 {
                Err("Unsupported pipe_format")
            } else {
                Ok(Format { p_format })
            }
        }
    }
}

impl From<Format> for pipe_format {
    fn from(format: Format) -> pipe_format {
        format.p_format
    }
}

impl Format {
    pub(crate) fn el_size_B(&self) -> u32 {
        let bits = self.p_format.blocksizebits();
        debug_assert!(bits % 8 == 0);
        bits / 8
    }

    pub(crate) fn el_extent_sa(&self) -> Extent4D<units::Samples> {
        let desc = self.p_format.description();
        Extent4D::new(desc.block.width, desc.block.height, desc.block.depth, 1)
    }

    pub(crate) fn info(&self) -> &nil_format_info {
        unsafe { &nil_format_table[self.p_format as usize] }
    }

    pub(crate) fn is_integer(&self) -> bool {
        self.p_format.is_integer()
    }

    pub(crate) fn is_depth_or_stencil(&self) -> bool {
        self.p_format.is_depth_or_stencil()
    }

    pub(crate) fn is_srgb(&self) -> bool {
        self.p_format.is_srgb()
    }

    pub fn supports_texturing(&self, dev: &nv_device_info) -> bool {
        if self.info().support() & NIL_FORMAT_SUPPORTS_TEXTURE_BIT == 0 {
            return false;
        }

        if self.p_format.layout() == UTIL_FORMAT_LAYOUT_ETC
            || self.p_format.layout() == UTIL_FORMAT_LAYOUT_ASTC
        {
            return dev.type_ == NV_DEVICE_TYPE_SOC
                && dev.cls_eng3d >= cla297::KEPLER_C;
        }

        true
    }

    pub fn supports_filtering(&self, dev: &nv_device_info) -> bool {
        self.supports_texturing(dev) && !self.is_integer()
    }

    pub fn supports_buffer(&self, dev: &nv_device_info) -> bool {
        if dev.cls_eng3d < clb097::MAXWELL_A && self.is_srgb() {
            return false;
        }
        self.info().support() & NIL_FORMAT_SUPPORTS_BUFFER_BIT != 0
    }

    pub fn supports_storage(&self, dev: &nv_device_info) -> bool {
        if dev.cls_eng3d >= clb097::MAXWELL_A {
            self.info().support() & NIL_FORMAT_SUPPORTS_STORAGE_BIT != 0
        } else {
            self.supports_kepler_storage()
        }
    }

    pub fn supports_color_targets(&self, _dev: &nv_device_info) -> bool {
        self.info().support() & NIL_FORMAT_SUPPORTS_RENDER_BIT != 0
    }

    pub fn supports_blending(&self, _dev: &nv_device_info) -> bool {
        self.info().support() & NIL_FORMAT_SUPPORTS_ALPHA_BLEND_BIT != 0
    }

    pub fn supports_depth_stencil(&self, dev: &nv_device_info) -> bool {
        if self.p_format == PIPE_FORMAT_S8_UINT {
            return dev.cls_eng3d >= clb197::MAXWELL_B;
        }
        self.info().support() & NIL_FORMAT_SUPPORTS_DEPTH_STENCIL_BIT != 0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_supports_texturing(
    dev: &nv_device_info,
    p_format: pipe_format,
) -> bool {
    Format::try_from(p_format).is_ok_and(|f| f.supports_texturing(dev))
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_supports_filtering(
    dev: &nv_device_info,
    p_format: pipe_format,
) -> bool {
    Format::try_from(p_format).is_ok_and(|f| f.supports_filtering(dev))
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_supports_buffer(
    dev: &nv_device_info,
    p_format: pipe_format,
) -> bool {
    Format::try_from(p_format).is_ok_and(|f| f.supports_buffer(dev))
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_supports_storage(
    dev: &nv_device_info,
    p_format: pipe_format,
) -> bool {
    Format::try_from(p_format).is_ok_and(|f| f.supports_storage(dev))
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_supports_color_targets(
    dev: &nv_device_info,
    p_format: pipe_format,
) -> bool {
    Format::try_from(p_format).is_ok_and(|f| f.supports_color_targets(dev))
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_supports_blending(
    dev: &nv_device_info,
    p_format: pipe_format,
) -> bool {
    Format::try_from(p_format).is_ok_and(|f| f.supports_blending(dev))
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_supports_depth_stencil(
    dev: &nv_device_info,
    p_format: pipe_format,
) -> bool {
    Format::try_from(p_format).is_ok_and(|f| f.supports_depth_stencil(dev))
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_to_color_target(p_format: pipe_format) -> u32 {
    Format::try_from(p_format).unwrap().info().czt()
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format_to_depth_stencil(p_format: pipe_format) -> u32 {
    Format::try_from(p_format).unwrap().info().czt()
}

#[unsafe(no_mangle)]
pub extern "C" fn nil_format(p_format: pipe_format) -> Format {
    p_format.try_into().unwrap()
}
