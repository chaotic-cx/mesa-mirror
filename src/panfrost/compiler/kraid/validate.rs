// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::debug::*;
use crate::ir::*;
use rustc_hash::FxHashSet;

fn validate_instr(
    model: &dyn Model,
    shader_info: &ShaderInfo,
    instr: &Instr,
    ssa_vals: &mut FxHashSet<SSAValue>,
) {
    for (src, src_type) in instr.srcs_types() {
        if let SrcRef::SSA(vec) = &src.src_ref {
            for ssa in vec {
                assert!(ssa_vals.contains(ssa), "Undefined SSA value: {ssa}");
            }
        }

        assert_ne!(src.swizzle, Swizzle::ZERO);
        if src.swizzle.is_word_swizzle() {
            assert_eq!(src_type.bits(), 64);
        } else if src.swizzle.is_byte_swizzle() {
            assert!(src.src_ref.bytes_read() <= 4);
        }

        if shader_info.src_swizzles_legal {
            assert!(
                model.op_src_supports_swizzle(&instr.op, src, src.swizzle),
                "{instr} uses an unsupported swizzles for {src:?}"
            );
        }

        let src_bytes = src_type.total_bytes();
        if src_type == DataType::SR || src_bytes > 8 {
            assert!(src.swizzle == Swizzle::NONE);
        } else {
            if src_bytes == 1 {
                assert!(
                    src.swizzle.replicates_byte(),
                    "{instr} uses invalid swizzle for {src:?}",
                );
            } else if src_bytes == 2 {
                assert!(
                    src.swizzle.replicates_half(),
                    "{instr} uses invalid swizzle for {src:?}",
                );
            }

            let src_ref_byte_mask = match &src.src_ref {
                SrcRef::Zero => 0xff,
                SrcRef::Imm32(_) => 0xf,
                SrcRef::Imm64(_) => 0xff,
                SrcRef::FAU(fau) => {
                    if fau.load64 {
                        0xff
                    } else {
                        0xf
                    }
                }
                SrcRef::SSA(vec) => u8::MAX >> (8 - vec.bytes()),
                SrcRef::Reg(reg) => match reg.range {
                    RegRange::Byte0 => 0b0001,
                    RegRange::Byte1 => 0b0010,
                    RegRange::Byte2 => 0b0100,
                    RegRange::Byte3 => 0b1000,
                    RegRange::Half0 => 0b0011,
                    RegRange::Half1 => 0b1100,
                    RegRange::Regs(n) => u8::MAX >> (8 - (n * 4)),
                },
                SrcRef::Mem(mem) => u8::MAX >> (8 - mem.range),
            };
            let swizzle_byte_mask = src.swizzle.bytes_read(src_bytes);
            assert!(swizzle_byte_mask & !src_ref_byte_mask == 0);
        }
    }

    for (dst, dst_type) in instr.dsts_types() {
        match &dst.dst_ref {
            DstRef::None => continue,
            DstRef::SSA(vec) => {
                for ssa in vec {
                    ssa_vals.insert(*ssa);
                }
            }
            DstRef::Reg(_) => (),
            DstRef::Mem(_) => (),
        }

        if dst.lanes.is_f16_narrow() {
            assert_eq!(dst_type, DataType::F32);
            assert_eq!(dst.dst_ref.bytes_written(), 2);
            continue;
        }

        if dst_type == DataType::SR {
            let dst_bytes = dst.dst_ref.bytes_written();
            assert_eq!(dst_bytes, dst.lanes.bytes(dst_bytes));
            continue;
        }

        let dst_type_bits = dst_type.bits();
        let dst_type_comps = dst_type.comps();
        if dst_type_bits >= 32 {
            assert_eq!(dst.lanes, DstLanes::All);
            let nregs = (dst_type_bits * dst_type_comps).div_ceil(32);
            assert_eq!(nregs * 4, dst.dst_ref.bytes_written());
        } else {
            let dst_type_bytes =
                (dst_type_bits * dst_type_comps).next_power_of_two() / 8;
            let lane_bytes = dst.lanes.bytes(dst_type_bytes);
            assert!(dst_type_bytes <= lane_bytes * 8);
            assert_eq!(lane_bytes, dst.dst_ref.bytes_written());
        }
    }
}

impl Function {
    pub fn validate(&self, model: &dyn Model, shader_info: &ShaderInfo) {
        if !cfg!(debug_assertions) && !DEBUG.contains(DebugFlags::VALIDATE) {
            return;
        }

        let mut blocks: FxHashSet<Label> = Default::default();
        for bb in &self.blocks {
            blocks.insert(bb.label);
        }

        let mut ssa_vals: FxHashSet<SSAValue> = Default::default();
        for (bi, block) in self.blocks.iter().enumerate() {
            let mut is_prelude = true;
            for instr in block.instrs.iter() {
                if matches!(&instr.op, Op::RegIn(_)) {
                    assert!(bi == 0, "OpRegIn is not in the start block");
                }

                if BasicBlock::is_prelude_instr(instr) {
                    assert!(
                        is_prelude,
                        "Misplaced prelude instr in block {}: {}",
                        block.label, instr,
                    );
                } else {
                    is_prelude = false;
                }

                validate_instr(model, shader_info, instr, &mut ssa_vals);
            }

            let mut is_postlude = true;
            for instr in block.instrs.iter().rev() {
                if BasicBlock::is_postlude_instr(instr) {
                    assert!(
                        is_postlude,
                        "Misplaced postlude instr in block {}: {}",
                        block.label, instr,
                    );
                } else {
                    is_postlude = false;
                }
            }
        }
    }
}

impl Shader<'_> {
    pub fn validate(&self) {
        let mut entrypoint_count = 0;
        for f in self.functions.iter() {
            f.validate(self.model, &self.info);
            if f.is_entrypoint {
                entrypoint_count += 1;
            }
        }

        assert!(
            entrypoint_count == 1,
            "We must have exactly one entrypoint per shader"
        );
    }
}
