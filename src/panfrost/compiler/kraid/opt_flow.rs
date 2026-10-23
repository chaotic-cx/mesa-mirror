// Copyright © 2026 Arm Ltd.
// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use compiler::dataflow::{BackwardDataflow, ForwardDataflow};

use crate::flow::FlowWaitBit;
use crate::ir::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Unknown,
    Signaled,
    Waited,
}

impl SlotState {
    pub fn then(&mut self, other: Self) {
        *self = match (*self, other) {
            (a, SlotState::Unknown) => a,
            (_, b) => b,
        };
    }

    pub fn join(&mut self, other: Self) {
        *self = match (*self, other) {
            (SlotState::Unknown, a) | (a, SlotState::Unknown) => a,
            (SlotState::Waited, SlotState::Waited) => SlotState::Waited,
            _ => SlotState::Signaled,
        };
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct WaitSlots {
    resource: SlotState,
    zs: SlotState,
    barrier: SlotState,
}

impl WaitSlots {
    pub fn unknown() -> Self {
        Self {
            barrier: SlotState::Unknown,
            resource: SlotState::Unknown,
            zs: SlotState::Unknown,
        }
    }

    pub fn signaled() -> Self {
        Self {
            barrier: SlotState::Signaled,
            resource: SlotState::Signaled,
            zs: SlotState::Signaled,
        }
    }

    pub fn issue(&mut self, instr: &Instr) {
        if matches!(&instr.op, Op::Barrier(_)) {
            self.barrier = SlotState::Signaled;
        }
    }

    pub fn wait_bit(&mut self, bit: FlowWaitBit) {
        match bit {
            FlowWaitBit::Resource => {
                self.resource = SlotState::Waited;
            }
            FlowWaitBit::ZS => {
                self.zs = SlotState::Waited;
            }
            FlowWaitBit::Barrier => {
                self.barrier = SlotState::Waited;
                // resource is used to wait on read-only resources
                // barrier is used to wait on both read-only and read-write resources
                // so it effectively waits for both
                self.resource = SlotState::Waited;
                // barrier also waits for dependency slot 6
                self.zs = SlotState::Waited;
            }
            _ => {}
        }
    }

    pub fn wait(&mut self, flow: FlowCtrl) {
        for bit in
            [FlowWaitBit::Resource, FlowWaitBit::ZS, FlowWaitBit::Barrier]
        {
            if flow.get_wait_bit(bit) {
                self.wait_bit(bit);
            }
        }
    }

    pub fn is_waited(&self, bit: FlowWaitBit) -> bool {
        let slot = match bit {
            FlowWaitBit::Resource => self.resource,
            FlowWaitBit::ZS => self.zs,
            FlowWaitBit::Barrier => self.barrier,
            _ => return false,
        };
        slot == SlotState::Waited
    }

    pub fn then(&mut self, other: Self) {
        self.resource.then(other.resource);
        self.zs.then(other.zs);
        self.barrier.then(other.barrier);
    }

    pub fn join(&mut self, other: Self) {
        self.resource.join(other.resource);
        self.zs.join(other.zs);
        self.barrier.join(other.barrier);
    }
}

fn dce_flow(f: &mut Function, shader_info: &ShaderInfo) {
    let cfg = &f.blocks;
    // First pass: gather waits per-block
    let block_usage: Vec<_> = cfg
        .iter()
        .map(|b| {
            b.instrs.iter().fold(WaitSlots::unknown(), |mut s, instr| {
                s.issue(instr);
                s.wait(instr.flow);
                s
            })
        })
        .collect();

    // Second pass: forward dataflow to get the initial register state for each
    //              block
    let mut reg_in = vec![WaitSlots::unknown(); cfg.len()];
    let mut reg_out = vec![WaitSlots::unknown(); cfg.len()];
    reg_in[0] = WaitSlots::signaled();

    if shader_info.stage == ShaderStage::Blend {
        // blend shaders come from a BLEND call, those require a zs + barrier
        // wait, so we can clear those slots
        reg_in[0].wait_bit(FlowWaitBit::Barrier);
    }
    ForwardDataflow {
        cfg,
        block_in: &mut reg_in[..],
        block_out: &mut reg_out[..],
        transfer: |block_idx, _block, live_out, live_in| {
            let old = *live_out;
            let mut state = *live_in;
            state.then(block_usage[block_idx]);
            *live_out = state;

            *live_out != old
        },
        join: |live_in, pred_live_out| live_in.join(*pred_live_out),
    }
    .solve();

    // Third pass: Remove useless waits
    for (bi, block) in f.blocks.iter_mut().enumerate() {
        let mut state = reg_in[bi];
        for instr in &mut block.instrs {
            state.issue(instr);
            for bit in
                [FlowWaitBit::ZS, FlowWaitBit::Resource, FlowWaitBit::Barrier]
            {
                if state.is_waited(bit) {
                    instr.flow.take_wait_bit(bit);
                }
            }
            state.wait(instr.flow);
        }
    }
}

fn merge_waits(f: &mut Function) {
    for block in &mut f.blocks {
        for i in (1..block.instrs.len()).rev() {
            if !matches!(block.instrs[i].op, Op::Nop(_)) {
                continue;
            }
            let nop_flow = block.instrs[i].flow;
            if nop_flow.get_end_shader()
                || nop_flow.get_discard()
                || nop_flow.get_reconverge()
                || nop_flow == FlowCtrl::NONE
            {
                continue;
            }

            let prev_flow = &mut block.instrs[i - 1].flow;

            // In theory we can promote a .wait_resource into a .barrier but that
            // is not useful in practice (as it also waits for r/w resources)
            if nop_flow.get_wait_bit(FlowWaitBit::Resource)
                && prev_flow.wait != 0
            {
                continue;
            }

            prev_flow.wait |= nop_flow.wait;
            block.instrs[i].flow.wait = 0;
        }

        block.instrs.retain(|i| {
            i.flow != FlowCtrl::NONE || !matches!(i.op, Op::Nop(_))
        });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockHelpersUsage {
    /// Block does not use helper threads at all
    None,
    /// Block uses helper threads and the last instruction using them is at
    /// index .0
    LastUseAt(usize),
    /// Block needs helpers as they'll be used by a successor
    Successor,
}

impl BlockHelpersUsage {
    fn merge(&mut self, successor: Self) {
        match successor {
            BlockHelpersUsage::None => (),
            BlockHelpersUsage::Successor | BlockHelpersUsage::LastUseAt(_) => {
                *self = BlockHelpersUsage::Successor;
            }
        }
    }
}

// TODO: we probably want to insert NOP.discard before messaging instructions,
// so that helper threads don't send useless messages.  This needs serious
// consideration as to where, doing it before LD_VAR will regress every shader
// that doesn't need helpers at all.
fn insert_shader_discards(f: &mut Function) {
    // First pass: gather last instr using helper lanes
    let mut last_in = vec![BlockHelpersUsage::None; f.blocks.len()];
    let mut last_out = Vec::with_capacity(f.blocks.len());

    for block in f.blocks.iter() {
        let mut helper_usage = BlockHelpersUsage::None;
        for (ip, instr) in block.instrs.iter().enumerate().rev() {
            if instr.needs_discarded_threads() {
                helper_usage = BlockHelpersUsage::LastUseAt(ip);
                break;
            }
        }
        last_out.push(helper_usage);
    }
    debug_assert_eq!(last_in.len(), last_out.len());

    // Propagate helpers usage backwards (we don't really need the second
    // vector but let's keep it to make BackwardDataflow happy)
    BackwardDataflow {
        cfg: &f.blocks,
        block_in: &mut last_in,
        block_out: &mut last_out,
        transfer: |_, _, block_in, block_out| {
            let changed = *block_in != *block_out;
            *block_in = *block_out;
            changed
        },
        join: |block_out, succ_block_in| {
            block_out.merge(*succ_block_in);
        },
    }
    .solve();

    // Walk on all instructions, keeping track of when a discard needs to be
    // placed, and place it on the first free instruction found.
    for block_idx in 0..f.blocks.len() {
        let block_usage = last_out[block_idx];
        let (start_discard, start_ip) = match block_usage {
            BlockHelpersUsage::None => {
                // Shaders that start with no helpers needed get a free starting
                // discard as a treat
                let start_unused = block_idx == 0;
                // If any of the incoming edges kill discards, kill them now
                let killed_edge = f
                    .blocks
                    .pred_indices(block_idx)
                    .iter()
                    .any(|x| last_out[*x] == BlockHelpersUsage::Successor);

                (start_unused || killed_edge, 0)
            }
            BlockHelpersUsage::LastUseAt(ip) => (true, ip),
            BlockHelpersUsage::Successor => {
                // If a successor needs helpers, we can't discard them in this
                // block.
                continue;
            }
        };

        let mut pending_discard = start_discard;
        let block = &mut f.blocks[block_idx];
        // Iterate from the last instruction needing helpers (skip start_ip)
        for instr in block.instrs.iter_mut().skip(start_ip) {
            // .discard should not be present before this pass
            debug_assert!(!instr.flow.get_discard());

            // If this instruction can create new unneeded helpers, push a
            // pending discard.
            pending_discard |=
                matches!(instr.op, Op::Discard(_) | Op::ATest(_));
            if !pending_discard {
                continue;
            }

            // If a pending discard reaches the end of the shader we can just
            // drop it, the shader is going to end soon regardless
            let clears_discard = instr.flow.get_end_shader()
                || matches!(instr.op, Op::BlendCall(_));
            if clears_discard {
                pending_discard = false;
                continue;
            }

            // We can only insert .discard in the current instruction if the
            // wait slot is free, so if it has no .wait* nor a .reconverge
            if instr.flow.wait == 0 && !instr.flow.get_reconverge() {
                instr.flow.set_discard();
                pending_discard = false;
            }
        }

        // We finished the block with a pending discard, let's blame the next
        // block.  This doesn't work if the next block is behind us, but that
        // is fine as that's only for loops and helpers can't be killed in loops
        if pending_discard {
            last_out[block_idx] = BlockHelpersUsage::Successor;
        }
    }
}

fn has_nop_end(block: &BasicBlock) -> bool {
    block
        .instrs
        .last()
        .is_some_and(|i| matches!(&i.op, Op::Nop(_)) && i.flow.get_end_shader())
}

/// This pass will remove NOP.end instructions if the .end modifier can
/// be encoded on the instruction before. Barriers at the end of shaders
/// are also safe to remove.
fn merge_end(f: &mut Function) {
    if !f.is_entrypoint {
        if cfg!(debug_assertions) {
            for block in &f.blocks {
                debug_assert!(!has_nop_end(block));
            }
        }
        return;
    }

    for block in &mut f.blocks {
        if !has_nop_end(block) || block.instrs.len() == 0 {
            continue;
        }

        let nop_end = block.instrs.pop().unwrap();

        while block
            .instrs
            .last()
            .is_some_and(|i| matches!(i.op, Op::Barrier(_)))
        {
            block.instrs.pop();
        }

        if let Some(last) = block.instrs.last_mut() {
            let slot = last.flow.get_msg_slot_idx();
            last.flow = FlowCtrl::NONE;
            if let Some(slot) = slot {
                last.flow.set_msg_slot_idx(slot);
            }
            last.flow.set_end_shader();
        } else {
            block.instrs.push(nop_end);
        }
    }
}

impl Function {
    /// This pass optimizes and merges flow wait bits.
    ///
    /// 1. Performs data-flow analysis to remove waits on already waited slots
    /// 2. Merges NOP waits into previous instructions where possible
    /// 3. Merges NOP.end with the previous instructions
    /// 4. Insert .discard where helper threads aren't needed
    pub fn opt_flow(&mut self, shader_info: &ShaderInfo) {
        dce_flow(self, shader_info);
        merge_waits(self);
        merge_end(self);
        if shader_info.stage == ShaderStage::Fragment {
            insert_shader_discards(self);
        }
    }
}

impl Shader<'_> {
    pub fn opt_flow(&mut self) {
        for f in self.functions.iter_mut() {
            f.opt_flow(&self.info);
        }
    }
}
