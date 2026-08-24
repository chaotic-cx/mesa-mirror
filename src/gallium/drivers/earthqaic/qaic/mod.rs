/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

pub extern crate qaic_rs_bindings;

#[allow(unused_imports, unused_macros)]
pub mod semaphores {
    macro_rules! define_sem {
        ($v:expr, $i:expr, $p:expr, $c:expr, $f:expr) => {
            qaic_rs_bindings::qaic_sem {
                val: $v,
                index: $i,
                presync: $p,
                cmd: $c,
                flags: $f,
                pad: 0,
            }
        };
    }
    pub(crate) use define_sem;

    macro_rules! empty_sem {
        () => {
            define_sem!(0, 0, 0, 0, 0)
        };
    }
    pub(crate) use empty_sem;

    macro_rules! default_pre_sem {
        ($v:expr, $i:expr) => {
            define_sem!($v, $i, 1, qaic_rs_bindings::QAIC_SEM_WAIT_EQUAL as u8, 0)
        };
    }
    pub(crate) use default_pre_sem;

    macro_rules! default_post_sem {
        ($v:expr, $i:expr) => {
            define_sem!($v, $i, 0, qaic_rs_bindings::QAIC_SEM_INIT as u8, 0)
        };
    }
    pub(crate) use default_post_sem;
}

pub mod control;
pub mod data;
pub mod metadata;
pub mod network;
