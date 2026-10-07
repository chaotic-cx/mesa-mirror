use crate::ir::*;
use crate::ops::*;
use crate::ssa_value::SSAValueIndexedVec;

/// Checks if all sources for the Opcode support swizzles and modifiers
fn op_supports_srcs(model: &dyn Model, op: &Op) -> bool {
    op.srcs().iter().all(|src| {
        model.op_src_supports_swizzle(op, src, src.swizzle)
            && model.op_src_supports_mod(op, src, src.src_mod)
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct InstrIdx {
    instr_idx: usize,
    block_idx: usize,
}

// Default is OOB (for safety)
impl Default for InstrIdx {
    fn default() -> Self {
        Self {
            instr_idx: usize::MAX,
            block_idx: usize::MAX,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct SSAEntry {
    use_count: u32,
    def_site: InstrIdx,
}

struct Fuse {
    ssa_map: SSAValueIndexedVec<SSAEntry>,
}

impl Fuse {
    pub fn new(ssa_count: u32) -> Self {
        Self {
            ssa_map: SSAValueIndexedVec::with_count(ssa_count),
        }
    }

    pub fn prepass_visit(&mut self, i: &Instr, idx: InstrIdx) {
        for ssa in i.iter_ssa_uses() {
            self.ssa_map[ssa].use_count += 1;
        }
        for ssa in i.iter_ssa_defs() {
            debug_assert_eq!(self.ssa_map[ssa].def_site, Default::default());
            self.ssa_map[ssa].def_site = idx;
        }
    }

    /// Get the res_type of the compare instruction
    fn cmp_res_type(
        &self,
        f: &Function,
        ssa: SSAValue,
    ) -> Option<(CmpResultType, u8)> {
        let entry = &self.ssa_map[ssa];
        let def = entry.def_site;
        let i = &f.blocks[def.block_idx].instrs[def.instr_idx];

        if let Op::Cmp(op) = &i.op {
            Some((op.res_type, op.src_type.bits()))
        } else {
            None
        }
    }

    /// Returns a FCmp/ICmp entry only when its result is used once
    fn get_foldable_cmp<'a>(
        &self,
        f: &'a Function,
        ssa: SSAValue,
    ) -> Option<&'a OpCmp> {
        let entry = &self.ssa_map[ssa];

        if entry.use_count != 1 {
            return None;
        }

        let def = entry.def_site;
        let i = &f.blocks[def.block_idx].instrs[def.instr_idx];

        let Op::Cmp(cmp) = &i.op else {
            return None;
        };

        debug_assert!(cmp.res_type != CmpResultType::C);
        debug_assert!(cmp.src_type.bits() <= 32);

        Some(cmp.as_ref())
    }

    // y.h0 = FCMP.v2f16.lt.m1 x 0
    // DISCARD.f32.ne y.h0 0
    //      =>
    // DISCARD.f32.lt x.h0 0
    fn fuse_discard(&self, f: &Function, op: &OpDiscard) -> Option<Op> {
        let cond_src = if op.srcs[0].is_zero() {
            &op.srcs[1]
        } else if op.srcs[1].is_zero() {
            &op.srcs[0]
        } else {
            return None;
        };
        if op.cmp_op != CmpOp::Ne || !cond_src.src_mod.is_none() {
            return None;
        }

        let SrcRef::SSA(cond_vec) = &cond_src.src_ref else {
            return None;
        };
        debug_assert_eq!(cond_vec.len(), 1);
        let cond_ssa = cond_vec[0];

        let cmp = self.get_foldable_cmp(f, cond_ssa)?;
        if !cmp.src_type.is_float_type() || cmp.accum_op != CmpAccumOp::None {
            return None;
        }

        let srcs = if cond_ssa.bits() == 16 {
            if !op.srcs[0].swizzle.is_f16_widen() || cmp.src_type.bits() != 16 {
                return None;
            }

            cmp.srcs.clone().map(|x| x.swizzle(op.srcs[0].swizzle))
        } else {
            if !op.srcs[0].swizzle.is_none() || cmp.src_type.bits() != 32 {
                return None;
            }
            cmp.srcs.clone()
        };

        Some(Op::from(OpDiscard {
            cmp_op: cmp.cmp_op,
            srcs,
        }))
    }

    // c = FCMP.f32.lt a, b
    // d = MUX.i32.int_zero y, x, c
    //      =>
    // d = CSEL.f32.lt a, b, x, y
    // This actually upsets the spiller, presumably because we increase the max
    // register pressure.  It is still a (small) win with small shaders though
    fn fuse_mux_csel(&self, f: &Function, mux: &OpMux) -> Option<Op> {
        if mux.mux_op != MuxOp::IntZero || !mux.sel.src_mod.is_none() {
            return None;
        }
        debug_assert!(mux.dst_type.total_bits() <= 32);

        let SrcRef::SSA(sel_ssa) = &mux.sel.src_ref else {
            return None;
        };
        let sel_ssa = sel_ssa[0];
        let cmp = self.get_foldable_cmp(f, sel_ssa)?;
        if cmp.accum_op != CmpAccumOp::None
            || cmp.src_type.bits() != mux.dst_type.bits()
        {
            return None;
        }

        let cmp_srcs = cmp.srcs.clone().map(|x| x.swizzle(mux.sel.swizzle));
        let op = OpCSel {
            dst: mux.dst.clone(),
            cmp_type: cmp.src_type,
            cmp_op: cmp.cmp_op,
            cmp_srcs,
            sel_srcs: [mux.src1.clone(), mux.src0.clone()],
        };

        if !op.is_valid_variant(op.cmp_type) {
            return None;
        }
        Some(op.into())
    }

    // c = FCMP.f32.lt a, b
    // f = FCMP.f32.lt d, e
    // g = OR c, d
    //      =>
    // c = FCMP.f32.lt a, b
    // g = FCMP_OR.f32.lt d, e, c
    fn fuse_shift_lop_accum(
        &self,
        f: &Function,
        lop: &OpShiftLop,
    ) -> Option<Op> {
        if lop.not_result
            || lop.shift_op != ShiftOp::None
            || lop.dst_type.total_bits() > 32
        {
            return None;
        }

        let accum_op = match lop.logic_op {
            LogicOp::And => CmpAccumOp::And,
            LogicOp::Or => CmpAccumOp::Or,
            _ => return None,
        };

        let src0_ssa = lop.src0.as_ssa()?[0];
        let src2_ssa = lop.src2.as_ssa()?[0];

        let no_accum = |x: &&OpCmp| x.accum_op == CmpAccumOp::None;
        let (cmp_entry, accum_src) = if let Some(cmp) =
            self.get_foldable_cmp(f, src0_ssa).filter(no_accum)
        {
            (cmp, src2_ssa)
        } else if let Some(cmp) =
            self.get_foldable_cmp(f, src2_ssa).filter(no_accum)
        {
            (cmp, src0_ssa)
        } else {
            return None;
        };

        // Check if the other source is a cmp-result with the same type
        if (cmp_entry.res_type, cmp_entry.src_type.bits())
            != self.cmp_res_type(f, accum_src)?
        {
            return None;
        }

        Some(Op::from(OpCmp {
            dst: lop.dst.clone(),
            src_type: cmp_entry.src_type,
            res_type: cmp_entry.res_type,
            cmp_op: cmp_entry.cmp_op,
            srcs: cmp_entry.srcs.clone(),
            accum: accum_src.into(),
            accum_op,
        }))
    }

    // n = ICMP.ne c 0
    // BRANCH.not n
    //      =>
    // BRANCH.not c
    // Also flips the polarity:
    // n = ICMP.eq c 0
    // BRANCH.not n
    //      =>
    // BRANCH c
    // TODO: Needs support for swizzles as combine ops
    fn fuse_branch(&self, f: &Function, branch: &OpBranch) -> Option<Op> {
        if branch.combine_op != BranchCombineOp::None
            || !branch.cond.src_mod.is_none()
        {
            return None;
        }

        let SrcRef::SSA(cond_vec) = &branch.cond.src_ref else {
            return None;
        };
        debug_assert_eq!(cond_vec.len(), 1);
        let cond_ssa = cond_vec[0];

        let cmp = self.get_foldable_cmp(f, cond_ssa)?;
        if cmp.src_type.is_float_type() || cmp.accum_op != CmpAccumOp::None {
            return None;
        }

        if cmp.cmp_op != CmpOp::Eq && cmp.cmp_op != CmpOp::Ne {
            return None;
        }

        let c = if cmp.srcs[0].is_zero() {
            &cmp.srcs[1]
        } else if cmp.srcs[1].is_zero() {
            &cmp.srcs[0]
        } else {
            return None;
        };

        if cond_ssa.bits() != 32 || cmp.src_type.bits() != 32 {
            return None;
        }

        Some(Op::from(OpBranch {
            not: branch.not ^ (cmp.cmp_op == CmpOp::Eq),
            cond: c.clone(),
            combine_op: branch.combine_op,
            label: branch.label,
        }))
    }

    pub fn visit(&self, f: &Function, i: &Instr) -> Option<Op> {
        match &i.op {
            Op::Discard(op) => self.fuse_discard(f, op),
            Op::Mux(op) => self.fuse_mux_csel(f, op),
            Op::ShiftLop(op) => self.fuse_shift_lop_accum(f, op),
            Op::Branch(op) => self.fuse_branch(f, op),
            _ => None,
        }
    }
}

/// Fuses instruction sequences into equivalent but shorter ones
///
/// Many instructions can do an "implicit" AND X, OR X. For example:
/// FCMP.ge.or A B ACCUM.  This pass fuses (OR (FCMP A B) ACCUM) into
/// (FCMP_OR A B ACCUM).
///
/// TODO: Unit tests
impl Function {
    pub fn opt_fuse(&mut self, model: &dyn Model) -> bool {
        let mut ctx = Fuse::new(self.ssa_alloc.count());

        // Pass 1: gather SSA use_count and def_site
        for (block_idx, block) in self.blocks.iter().enumerate() {
            for (instr_idx, instr) in block.instrs.iter().enumerate() {
                let idx = InstrIdx {
                    instr_idx,
                    block_idx,
                };
                ctx.prepass_visit(instr, idx);
            }
        }

        // Pass 2: actual fusions
        let mut progress = false;
        for block_idx in 0..self.blocks.len() {
            for instr_idx in 0..self.blocks[block_idx].instrs.len() {
                let i = &self.blocks[block_idx].instrs[instr_idx];
                let repl =
                    ctx.visit(self, i).filter(|op| op_supports_srcs(model, op));
                let Some(repl_op) = repl else {
                    continue;
                };
                progress = true;
                // No need to update the prepass map
                self.blocks[block_idx].instrs[instr_idx].op = repl_op;
            }
        }
        progress
    }
}

impl Shader<'_> {
    pub fn opt_fuse(&mut self) -> bool {
        let mut progress = false;

        for f in self.functions.iter_mut() {
            progress |= f.opt_fuse(self.model);
        }
        progress
    }
}
