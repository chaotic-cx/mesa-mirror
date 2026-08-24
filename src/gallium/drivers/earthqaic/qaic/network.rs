/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

use crate::qaic;
use crate::qaic::metadata::*;
use crate::qaic::qaic_rs_bindings::*;

#[derive(Debug)]
pub struct Network {
    pub constants_id: u32,
    pub elf_id: u32,
    pub activation_id: u32,
    pub dbc_id: u32,
    pub static_ddr_base: u64,
    pub dynamic_ddr_base: u64,
    pub mcid_base: u64,
    pub eventfd: i32,
    pub metadata: CompactMetadata,
}

pub fn load_and_activate_network(fd: i32, load_consts: bool) -> Result<Network, &'static str> {
    let path = crate::qshim_path("md_struct.bin");
    let metadata = match qaic::metadata::read_md_struct_bin(&path) {
        Ok(md) => md,
        Err(_e) => {
            eqaic_eprintln!("Unable to read {}", path);
            return Err("Could not read md_struct.bin");
        }
    };

    let mut network = Network {
        metadata: metadata,
        constants_id: 0,
        elf_id: 0,
        activation_id: 0,
        dbc_id: 0,
        static_ddr_base: 0,
        dynamic_ddr_base: 0,
        mcid_base: 0,
        eventfd: -1,
    };

    let mut load_consts_resp = nnc_cmd_load_constants_response_t {
        ..Default::default()
    };

    if load_consts {
        let ret = qaic::control::load_constants_ex(
            fd,
            &crate::qshim_path("constantsdesc.bin"),
            None,
            &mut network.constants_id,
            &mut load_consts_resp,
        );
        if ret != 0 {
            return Err("Could not load constants!");
        }

        let ret = qaic::control::load_constants_payload(
            fd,
            &crate::qshim_path("constants.bin"),
            network.constants_id,
            &mut load_consts_resp,
        );
        if ret != 0 {
            return Err("Could not load payload!");
        }
    }

    let mut load_elf_resp = nnc_cmd_load_elf_response_t {
        ..Default::default()
    };
    let ret = qaic::control::load_elf(
        fd,
        &crate::qshim_path("network.elf"),
        None,
        &mut network.elf_id,
        &mut load_elf_resp,
    );
    if ret != 0 {
        return Err("Could not load elf!");
    }

    let mut activate_resp = nnc_cmd_activate_response_t {
        ..Default::default()
    };
    let ret = qaic::control::activate_default(
        fd,
        network.elf_id,
        network.constants_id,
        &mut network.activation_id,
        &mut network.dbc_id,
        &mut network.static_ddr_base,
        &mut network.dynamic_ddr_base,
        &mut network.mcid_base,
        &mut network.eventfd,
        &mut activate_resp,
    );
    if ret != 0 {
        return Err("Could not activate elf!");
    }
    Ok(network)
}

pub fn deactivate_and_unload_network(fd: i32, network: &Network, unload_consts: bool) {
    let mut deactivate_resp = nnc_cmd_deactivate_response_t {
        ..Default::default()
    };
    let ret = qaic::control::deactivate(
        fd,
        (*network).activation_id,
        (*network).dbc_id,
        1,
        &mut deactivate_resp,
    );
    // The VC's eventfd is owned by this process regardless of whether
    // deactivation succeeded, so it must be closed either way.
    if (*network).eventfd >= 0 {
        unsafe {
            libc::close((*network).eventfd);
        }
    }
    if ret != 0 {
        eqaic_eprintln!("Unable to deactivate elf! Err: {}", ret);
        return;
    }

    let mut unload_elf_resp = nnc_cmd_unload_elf_response_t {
        ..Default::default()
    };
    let ret = qaic::control::unload_elf(fd, (*network).elf_id, &mut unload_elf_resp);
    if ret != 0 {
        eqaic_eprintln!("Unable to unload elf! Err: {}", ret);
        return;
    }

    if unload_consts {
        let mut unload_consts_resp = nnc_cmd_unload_constants_response_t {
            ..Default::default()
        };
        let ret =
            qaic::control::unload_constants(fd, (*network).constants_id, &mut unload_consts_resp);
        if ret != 0 {
            eqaic_eprintln!("Unable to unload constants! Err: {}", ret);
            return;
        }
    }
}
