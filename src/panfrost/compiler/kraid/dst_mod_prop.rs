// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::ops::*;
use crate::ssa_value::AllocSSA;
use mesa_util::bitview::BitViewable;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Copy, Default, PartialEq)]
enum Uses<T: Clone + Copy + PartialEq> {
    #[default]
    None,
    One(T),
    Unknown,
}

impl<T: Clone + Copy + PartialEq> Uses<T> {
    fn add_use(&mut self, u: Uses<T>) {
        match u {
            Uses::None => panic!("None is only an initializer"),
            Uses::One(uv) => match *self {
                Uses::None => *self = u,
                Uses::One(sv) => {
                    if uv != sv {
                        *self = Uses::Unknown;
                    }
                }
                Uses::Unknown => (),
            },
            Uses::Unknown => *self = Uses::Unknown,
        }
    }
}

#[derive(Clone, Copy, Default)]
enum ClampUses {
    #[default]
    None,
    F16(FClamp),
    F32(FClamp),
    Unknown,
}

impl ClampUses {
    fn add_use(&mut self, use_clamp: ClampUses) {
        match (*self, use_clamp) {
            (_, ClampUses::None) => (),
            (ClampUses::None, _) => *self = use_clamp,
            (ClampUses::F16(s), ClampUses::F16(u)) => {
                *self = ClampUses::F16(s.max(u))
            }
            (ClampUses::F32(s), ClampUses::F32(u)) => {
                *self = ClampUses::F32(s.max(u))
            }
            _ => *self = ClampUses::Unknown,
        }
    }

    fn get_clamp(&self, dst_type: DataType) -> Option<FClamp> {
        match (self, dst_type) {
            (ClampUses::None, _) => Some(FClamp::None),
            (ClampUses::F16(clamp), DataType::F16 | DataType::V2F16) => {
                Some(*clamp)
            }
            (ClampUses::F32(clamp), DataType::F32) => Some(*clamp),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
enum SwizzleUses {
    #[default]
    None,
    Replicate,
    // Only matches replication for .m1
    SignExtend,
    Unknown,
}

impl SwizzleUses {
    fn from_swizzle(swizzle: Swizzle) -> Self {
        const SH0: Swizzle = Swizzle::widen_s16(0);

        match swizzle {
            Swizzle::H00 => SwizzleUses::Replicate,
            SH0 => SwizzleUses::SignExtend,
            _ => SwizzleUses::Unknown,
        }
    }

    fn add_use(&mut self, other: Self) {
        *self = (*self).max(other);
    }
}

#[derive(Default)]
struct UseMod {
    clamp: ClampUses,
    to_f16: Uses<FRound>,
    swizzle: SwizzleUses,
    bool_cvt: Uses<(CmpResultType, u8)>,
}

impl UseMod {
    const INVALID: UseMod = UseMod {
        clamp: ClampUses::Unknown,
        to_f16: Uses::Unknown,
        swizzle: SwizzleUses::Unknown,
        bool_cvt: Uses::Unknown,
    };

    fn add_use(&mut self, use_mod: UseMod) {
        self.clamp.add_use(use_mod.clamp);
        self.to_f16.add_use(use_mod.to_f16);
        self.swizzle.add_use(use_mod.swizzle);
        self.bool_cvt.add_use(use_mod.bool_cvt);
    }
}

#[derive(Clone, Copy, Default)]
enum SrcType {
    #[default]
    None,
    F16,
    F32,
}

impl SrcType {
    fn from_src(src: &Src, src_type: DataType) -> SrcType {
        match src_type {
            DataType::F16 | DataType::V2F16 => {
                debug_assert!(matches!(
                    src.swizzle,
                    Swizzle::H00 | Swizzle::H01 | Swizzle::H10 | Swizzle::H11
                ));
                SrcType::F16
            }
            DataType::F32 => {
                if src.swizzle.is_f16_widen() {
                    SrcType::F16
                } else if src.swizzle.is_none() {
                    SrcType::F32
                } else {
                    SrcType::None
                }
            }
            _ => SrcType::None,
        }
    }
}

struct DstModProp<'a, A> {
    ssa_alloc: &'a mut A,
    use_mods: FxHashMap<SSAValue, UseMod>,
    def_clamp: FxHashMap<SSAValue, ClampUses>,
    def_f16: FxHashMap<SSAValue, SSAValue>,
    def_repl: FxHashMap<SSAValue, SSAValue>,
    def_bool: FxHashSet<SSAValue>,
}

impl<A: AllocSSA> DstModProp<'_, A> {
    fn new<'a>(ssa_alloc: &'a mut A) -> DstModProp<'a, A> {
        DstModProp {
            ssa_alloc,
            use_mods: Default::default(),
            def_clamp: Default::default(),
            def_f16: Default::default(),
            def_repl: Default::default(),
            def_bool: Default::default(),
        }
    }

    fn add_use(&mut self, ssa: SSAValue, use_mod: UseMod) {
        self.use_mods.entry(ssa).or_default().add_use(use_mod);
    }

    fn set_def_clamp(&mut self, ssa: SSAValue, clamp: ClampUses) {
        let old = self.def_clamp.insert(ssa, clamp);
        debug_assert!(old.is_none());
    }

    fn set_def_f16(&mut self, ssa: SSAValue, f16: SSAValue) {
        let old = self.def_f16.insert(ssa, f16);
        debug_assert!(old.is_none());
    }

    fn set_def_repl(&mut self, ssa: SSAValue, repl: SSAValue) {
        let old = self.def_repl.insert(ssa, repl);
        debug_assert!(old.is_none());
    }

    fn set_def_bool(&mut self, ssa: SSAValue) {
        let is_new = self.def_bool.insert(ssa);
        debug_assert!(is_new);
    }

    fn add_fadd_uses(&mut self, op: &OpFAdd) -> bool {
        let src = if op.srcs[0].is_fneg_zero(op.dst_type) {
            &op.srcs[1]
        } else if op.srcs[1].is_fneg_zero(op.dst_type) {
            &op.srcs[0]
        } else {
            return false;
        };

        let SrcRef::SSA(vec) = &src.src_ref else {
            return false;
        };
        debug_assert!(vec.comps() == 1);
        let ssa = vec[0];

        if !src.src_mod.is_none() {
            return false;
        }

        let use_mod = match SrcType::from_src(src, op.dst_type) {
            SrcType::None => UseMod::INVALID,
            SrcType::F16 => UseMod {
                clamp: ClampUses::F16(op.clamp),
                ..UseMod::INVALID
            },
            SrcType::F32 => UseMod {
                clamp: ClampUses::F32(op.clamp),
                to_f16: if op.dst.lanes.is_f16_narrow() {
                    Uses::One(op.round)
                } else {
                    Uses::Unknown
                },
                ..UseMod::INVALID
            },
        };
        self.add_use(ssa, use_mod);
        true
    }

    // TODO: we can also fold SWZ when there are other, non-OpSwz uses, but
    //       it requires checking and rewriting sources from every Op.  For more
    //       details and a WIP impl check the MR
    fn add_swz_uses(&mut self, op: &OpSwz) -> bool {
        if !(op.src_type.total_bits() == 32 && op.src.src_mod.is_none()) {
            return false;
        }
        let SrcRef::SSA(vec) = &op.src.src_ref else {
            return false;
        };
        debug_assert_eq!(vec.len(), 1);
        let ssa = vec[0];

        self.add_use(
            ssa,
            UseMod {
                swizzle: SwizzleUses::from_swizzle(op.src.swizzle),
                ..UseMod::INVALID
            },
        );
        true
    }

    fn add_mux_uses(&mut self, op: &OpMux) -> bool {
        // TODO: In theory if src1 is zero we can invert cmp_op, doesn't seem to
        // actually ever happen in shader-db though (also, a bit more complex
        // with floats), nir probably canonicalizes it.
        if op.mux_op != MuxOp::IntZero
            || !op.sel.src_mod.is_none()
            || !op.src0.is_zero()
        {
            return false;
        }

        let bits = op.dst_type.bits();
        debug_assert!(bits <= 32);

        let val = &op.src1;
        let replicates = match bits {
            8 => val.replicates_byte(),
            16 => val.replicates_half(),
            32 => true,
            _ => unreachable!(),
        };
        let res_type = val
            .resolve_imm(op.dst_type)
            .filter(|_| replicates)
            .and_then(|imm32| {
                CmpResultType::from_true_bits(
                    imm32.get_bit_range_u64(0..bits.into()) as u32,
                    bits,
                )
            });
        let Some(res_type) = res_type else {
            return false;
        };

        let SrcRef::SSA(sel_ssa) = &op.sel.src_ref else {
            return false;
        };
        let sel_ssa = sel_ssa[0];

        self.add_use(
            sel_ssa,
            UseMod {
                bool_cvt: Uses::One((res_type, bits)),
                ..UseMod::INVALID
            },
        );
        true
    }

    fn add_instr_uses(&mut self, instr: &Instr) {
        let used = match &instr.op {
            Op::FAdd(op) => self.add_fadd_uses(op.as_ref()),
            Op::Swz(op) => self.add_swz_uses(op.as_ref()),
            Op::Mux(op) => self.add_mux_uses(op.as_ref()),
            _ => false,
        };

        if !used {
            for ssa in instr.iter_ssa_uses() {
                self.add_use(*ssa, UseMod::INVALID);
            }
        }
    }

    fn try_remove_clamp(
        &self,
        clamp: &mut FClamp,
        src_ssa: SSAValue,
        src_type: DataType,
    ) -> bool {
        if clamp.is_none() {
            return false;
        }

        let def_clamp = match self.def_clamp.get(&src_ssa) {
            None => return false,
            Some(ClampUses::F16(def_clamp)) => {
                debug_assert!(matches!(
                    src_type,
                    DataType::F16 | DataType::V2F16
                ));
                *def_clamp
            }
            Some(ClampUses::F32(def_clamp)) => {
                debug_assert!(src_type == DataType::F32);
                *def_clamp
            }
            Some(_) => unreachable!(),
        };

        if def_clamp == def_clamp.min(*clamp) {
            *clamp = FClamp::None;
            true
        } else {
            false
        }
    }

    fn try_fold_instr_src(&mut self, instr: &mut Instr) -> bool {
        // This must match the same cases as add_instr_uses()
        match &mut instr.op {
            Op::FAdd(op) => {
                let src = if op.srcs[0].is_fneg_zero(op.dst_type) {
                    &mut op.srcs[1]
                } else if op.srcs[1].is_fneg_zero(op.dst_type) {
                    &mut op.srcs[0]
                } else {
                    return false;
                };

                let SrcRef::SSA(vec) = &src.src_ref else {
                    return false;
                };
                debug_assert!(vec.comps() == 1);
                let ssa = vec[0];

                let clamp_changed =
                    self.try_remove_clamp(&mut op.clamp, ssa, op.dst_type);

                if let Some(ssa_f16) = self.def_f16.get(&ssa) {
                    debug_assert!(op.dst_type == DataType::F32);
                    op.dst.lanes = match op.dst.lanes {
                        DstLanes::AnyHF => DstLanes::AnyH,
                        DstLanes::HF0 => DstLanes::H0,
                        DstLanes::HF1 => DstLanes::H1,
                        _ => panic!("Must be a f16 narrow"),
                    };
                    op.dst_type = DataType::V2F16;
                    src.src_ref = (*ssa_f16).into();
                    src.swizzle = Swizzle::H00;
                    true
                } else {
                    clamp_changed
                }
            }
            Op::Swz(op) => {
                let SrcRef::SSA(vec) = &op.src.src_ref else {
                    return false;
                };
                if vec.comps() != 1 {
                    return false;
                }
                let ssa = vec[0];

                let Some(repl) = self.def_repl.get(&ssa) else {
                    return false;
                };
                debug_assert_eq!(repl.bits(), op.src_type.total_bits());

                instr.op = OpCopy {
                    dst: op.dst.clone(),
                    dst_type: DataType::i(repl.bits()),
                    src: (*repl).into(),
                }
                .into();

                true
            }
            Op::Mux(op) => {
                let SrcRef::SSA(vec) = &op.sel.src_ref else {
                    return false;
                };
                if !self.def_bool.contains(&vec[0]) {
                    return false;
                };

                let bytes = op.dst.lanes.bytes(op.dst_type.total_bytes());
                instr.op = Op::from(OpSwz {
                    dst: op.dst.clone(),
                    src_type: DataType::i(bytes * 8),
                    src: op.sel.clone(),
                });

                true
            }
            _ => false,
        }
    }

    fn fold_clamp(
        &mut self,
        dst: &Dst,
        mut dst_type: DataType,
        clamp: &mut FClamp,
    ) -> bool {
        let DstRef::SSA(vec) = &dst.dst_ref else {
            return false;
        };
        debug_assert!(vec.comps() == 1);
        let ssa = vec[0];

        if dst_type == DataType::F32 && dst.lanes.is_f16_narrow() {
            dst_type = DataType::F16;
        }

        let mut changed = false;
        if let Some(use_mod) = self.use_mods.get(&ssa) {
            if let Some(use_clamp) = use_mod.clamp.get_clamp(dst_type) {
                let new_clamp = clamp.min(use_clamp);
                changed = new_clamp != *clamp;
                *clamp = new_clamp;
            }
        }

        match dst_type {
            DataType::F16 | DataType::V2F16 => {
                self.set_def_clamp(ssa, ClampUses::F16(*clamp));
            }
            DataType::F32 => {
                self.set_def_clamp(ssa, ClampUses::F32(*clamp));
            }
            _ => panic!("Invlaid dst_type"),
        }

        changed
    }

    fn fold_to_f32(
        &mut self,
        dst: &mut Dst,
        dst_type: DataType,
        round: FRound,
    ) -> bool {
        if dst_type != DataType::F32 {
            return false;
        }

        let DstRef::SSA(vec) = &mut dst.dst_ref else {
            return false;
        };
        debug_assert!(vec.comps() == 1);
        let ssa = vec[0];

        let Some(use_mod) = self.use_mods.get(&ssa) else {
            return false;
        };

        if use_mod.to_f16 != Uses::One(round) {
            return false;
        }

        debug_assert!(ssa.bits() == 32);
        let ssa_f16 = self.ssa_alloc.alloc_ssa(16);
        self.set_def_f16(ssa, ssa_f16);
        dst.dst_ref = ssa_f16.into();
        dst.lanes = DstLanes::AnyHF;
        true
    }

    fn fold_cmp(&mut self, op: &mut OpCmp) -> bool {
        let DstRef::SSA(vec) = &op.dst.dst_ref else {
            return false;
        };

        debug_assert!(vec.comps() == 1);
        let ssa = vec[0];

        let Some(use_mod) = self.use_mods.get(&ssa) else {
            return false;
        };

        if use_mod.swizzle != SwizzleUses::Unknown {
            // 16-bit CMPs replicate their sources
            if op.accum_op != CmpAccumOp::None
                || op.src_type.bits() != 16
                || ssa.bits() != 16
                || !op.srcs.iter().all(Src::replicates_half)
            {
                return false;
            }
            match use_mod.swizzle {
                SwizzleUses::Replicate => (),
                // If at least one of our users needs sign-extension, we can
                // only fuse that for .m1
                SwizzleUses::SignExtend if op.res_type == CmpResultType::M1 => {
                }
                _ => return false,
            }

            let ssa_repl = self.ssa_alloc.alloc_ssa(32);
            self.set_def_repl(ssa, ssa_repl);
            op.dst = ssa_repl.into();

            true
        } else if let Uses::One((res_new, bits)) = use_mod.bool_cvt {
            if op.src_type.bits() != bits {
                return false;
            }

            op.res_type = res_new;
            self.set_def_bool(ssa);

            true
        } else {
            false
        }
    }

    fn try_fold_instr(&mut self, instr: &mut Instr) -> bool {
        let src_folded = self.try_fold_instr_src(instr);
        let dst_folded = match &mut instr.op {
            Op::FAdd(op) => {
                self.fold_clamp(&op.dst, op.dst_type, &mut op.clamp)
                    | self.fold_to_f32(&mut op.dst, op.dst_type, op.round)
            }
            Op::Fma(op) => {
                self.fold_clamp(&op.dst, op.dst_type, &mut op.clamp)
                    | self.fold_to_f32(&mut op.dst, op.dst_type, op.round)
            }
            Op::FMax(op) => {
                self.fold_clamp(&op.dst, op.dst_type, &mut op.clamp)
            }
            Op::FMin(op) => {
                self.fold_clamp(&op.dst, op.dst_type, &mut op.clamp)
            }
            Op::Cmp(op) => self.fold_cmp(op),
            _ => false,
        };
        src_folded | dst_folded
    }
}

impl Function {
    pub fn opt_dst_mod_prop(&mut self) -> bool {
        let mut mod_prop = DstModProp::new(&mut self.ssa_alloc);

        for block in self.blocks.iter() {
            for instr in block.instrs.iter() {
                mod_prop.add_instr_uses(instr);
            }
        }

        let mut progress = false;
        for block in self.blocks.iter_mut() {
            for instr in block.instrs.iter_mut() {
                progress |= mod_prop.try_fold_instr(instr);
            }
        }
        progress
    }
}

impl Shader<'_> {
    pub fn opt_dst_mod_prop(&mut self) -> bool {
        let mut progress = false;
        for f in self.functions.iter_mut() {
            progress |= f.opt_dst_mod_prop();
        }
        progress
    }
}
