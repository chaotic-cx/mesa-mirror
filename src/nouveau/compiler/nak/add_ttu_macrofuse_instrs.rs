// Copyright © 2026 Valve Corporation.
// SPDX-License-Identifier: MIT

use crate::ir::*;

const TTU_MACROFUSE_IMM_LIMIT: usize = 0x1f;

impl Shader<'_> {
    pub fn add_ttu_macrofuse_instrs(&mut self) {
        for f in &mut self.functions {
            for b in &mut f.blocks {
                let mut ttu_current_idx_opt = None;

                let mut instrs = Vec::new();
                for instr in b.instrs.drain(..) {
                    let needs_macro_fuse = matches!(
                        instr.op,
                        Op::TtuGo(_) | Op::TtuSt(_) | Op::TtuLd(_)
                    );

                    if let Some(ttu_current_idx) = ttu_current_idx_opt {
                        let mut imm = instrs.len() - ttu_current_idx;

                        if needs_macro_fuse {
                            imm += 1;
                        }

                        // Ensure that we have not reached the limit of TTUMACROFUSE immediate
                        if !needs_macro_fuse || imm == TTU_MACROFUSE_IMM_LIMIT {
                            debug_assert!(imm <= TTU_MACROFUSE_IMM_LIMIT);
                            instrs.insert(
                                ttu_current_idx,
                                Instr::new(OpTtuMacroFuse { imm: imm as u8 }),
                            );
                            ttu_current_idx_opt = None;
                        }
                    } else if needs_macro_fuse {
                        ttu_current_idx_opt = Some(instrs.len());
                    }

                    instrs.push(instr);
                }

                if let Some(ttu_current_idx) = ttu_current_idx_opt.take() {
                    let imm = instrs.len() - ttu_current_idx;
                    debug_assert!(imm > 0 && imm <= TTU_MACROFUSE_IMM_LIMIT);
                    instrs.insert(
                        ttu_current_idx,
                        Instr::new(OpTtuMacroFuse { imm: imm as u8 }),
                    );
                }

                b.instrs = instrs;
            }
        }
    }
}
