// Copyright © 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use std::cmp::Reverse;
use std::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64};

use crate::ir::*;
use compiler::smallvec::SmallVec;
use kraid_bindings::*;
use rustc_hash::FxHashMap;

const DEBUG_PRINT: bool = false;

#[derive(Default, Debug, Clone, Copy)]
pub struct Val {
    pub val: u64,
    pub byte_size: u8,
    same_slot: bool,
    same_page: bool,
}

impl Val {
    fn new(byte_size: u8, val: u64) -> Val {
        assert!([1, 2, 4, 8].contains(&byte_size));
        Val {
            val,
            byte_size,
            ..Default::default()
        }
    }
}

impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val && self.byte_size == other.byte_size
    }
}

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.byte_size == 0 {
            write!(f, "__")
        } else {
            let digits = self.byte_size as usize * 2;
            write!(f, "{:0digits$x}", self.val)
        }
    }
}

#[derive(Default, Debug, Clone)]
struct Slot {
    entries: [Val; 8],
}

impl Slot {
    fn try_insert(&mut self, val: Val) -> Result<(), ()> {
        if val.byte_size == 0 {
            return Ok(());
        }
        if (self.bytes_used() + val.byte_size) > 8 {
            return Err(());
        }

        for idx in 0..self.entries.len() {
            if val.byte_size > self.entries[idx].byte_size {
                let move_range = idx..(self.entries.len() - 1);
                self.entries.copy_within(move_range, idx + 1);
                self.entries[idx] = val;
                return Ok(());
            }
        }

        Err(())
    }

    fn merge(&mut self, other: Slot) -> Result<(), ()> {
        for value in other.entries {
            self.try_insert(value)?;
        }
        Ok(())
    }

    /// If other has any kind of constraint on it, take all that have a constraint.
    /// Otherwise, take only other.
    fn split_by(&mut self, other: &Val) -> Option<Slot> {
        let Some(ret_const) = self
            .entries
            .iter()
            .find(|e| *e == other)
            .map(|e| e.same_page || e.same_slot)
        else {
            return None;
        };

        let (taken, remainder): (Slot, Slot) =
            self.entries.into_iter().partition(|e| {
                e == other || (ret_const && (e.same_slot || e.same_page))
            });

        *self = remainder;
        debug_assert!(taken.contains(other));
        Some(taken)
    }

    fn position(&self, val: &Val) -> Option<usize> {
        let mut offset = 0;
        for entry in &self.entries {
            if entry == val {
                return Some(offset);
            }
            offset += entry.byte_size as usize;
        }
        None
    }

    fn contains(&self, val: &Val) -> bool {
        self.entries.iter().any(|v| v == val)
    }

    fn find_mut(&mut self, val: &Val) -> Option<&mut Val> {
        self.entries.iter_mut().find(|entry| *entry == val)
    }

    fn bytes_used(&self) -> u8 {
        self.entries.iter().map(|e| e.byte_size).sum()
    }

    fn val(&self) -> u64 {
        let mut acc = 0;
        let mut offset = 0;
        for entry in self.entries {
            if entry.byte_size == 0 {
                break;
            }
            acc |= entry.val << offset;
            offset += entry.byte_size * 8;
        }
        acc
    }
}

impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for val in &self.entries {
            if val.byte_size == 0 {
                break;
            }
            write!(f, "{val} ")?;
        }
        Ok(())
    }
}

impl std::iter::Extend<Val> for Slot {
    fn extend<T: IntoIterator<Item = Val>>(&mut self, iter: T) {
        for elem in iter {
            self.try_insert(elem).unwrap();
        }
    }
}

struct Page {
    slots: Vec<Slot>,
}

impl Page {
    fn new(nslots: usize) -> Page {
        Page {
            slots: vec![Slot::default(); nslots],
        }
    }

    pub fn insert(&mut self, v: Val) -> Result<(), ()> {
        for slot in self.slots.iter_mut() {
            if slot.try_insert(v).is_ok() {
                return Ok(());
            }
        }

        Err(())
    }

    fn find_mut(&mut self, val: &Val) -> Option<&mut Val> {
        for slot in &mut self.slots {
            let v = slot.find_mut(val);
            if v.is_some() {
                return v;
            }
        }
        None
    }

    fn contains(&self, val: &Val) -> bool {
        self.slots.iter().any(|slot| slot.contains(val))
    }
}

pub struct FauAlloc {
    pages: Vec<Page>,
}

impl FauAlloc {
    pub fn new(nbytes: usize, page_size: usize, offset: usize) -> FauAlloc {
        assert!(nbytes % 8 == 0);
        assert!(page_size % 8 == 0);
        assert!(offset % 8 == 0);

        let slots_first_page = (page_size - offset) / 8;
        let slots_per_page = page_size / 8;
        let mut total_slots = nbytes / 8;
        let mut pages: Vec<_> = Default::default();

        while total_slots > 0 {
            let n = if pages.is_empty() {
                slots_first_page
            } else {
                slots_per_page
            }
            .min(total_slots);
            debug_assert!(n < 128);
            debug_assert!(n > 0);
            pages.push(Page::new(n));
            total_slots -= n;
        }

        FauAlloc { pages }
    }

    pub fn insert(&mut self, v: Val) -> bool {
        for page in self.pages.iter_mut() {
            if page.insert(v).is_ok() {
                return true;
            }
        }
        false
    }

    pub fn insert_same_page(&mut self, mut v: Val, other: &Val) -> bool {
        if DEBUG_PRINT {
            eprintln!("insert_same_page: {v} as {other}");
        }

        let Some(page) =
            self.pages.iter_mut().find(|page| page.contains(other))
        else {
            return false;
        };

        v.same_page = true;
        if page.insert(v).is_err() {
            return false;
        }
        page.find_mut(other).unwrap().same_page = true;

        true
    }

    pub fn insert_same_slot(&mut self, mut v: Val, other: &Val) -> bool {
        if v.byte_size + other.byte_size > 8 {
            return false;
        }
        if DEBUG_PRINT {
            eprintln!("insert_same_slot: {v} as {other}");
        }

        v.same_slot = true;

        let Some((page_idx, slot_idx)) = self.position(other) else {
            return false;
        };

        let slot = &mut self.pages[page_idx].slots[slot_idx];
        if slot.try_insert(v).is_ok() {
            slot.find_mut(other).unwrap().same_slot = true;
            return true;
        }

        let save = slot.clone();
        let mut evicted = slot.split_by(other).unwrap();
        if evicted.try_insert(v).is_err() {
            *slot = save;
            return false;
        }

        evicted.find_mut(other).unwrap().same_slot = true;

        let needs_same_page = evicted.entries.iter().any(|e| e.same_page);

        for (try_page_idx, page) in self.pages.iter_mut().enumerate() {
            if needs_same_page && try_page_idx != page_idx {
                continue;
            }
            for slot in &mut page.slots {
                if slot.bytes_used() + evicted.bytes_used() <= 8 {
                    slot.merge(evicted).unwrap();
                    return true;
                }
            }
        }

        // Didn't manage to insert elsewhere, put values back where they were before.
        self.pages[page_idx].slots[slot_idx] = save;

        false
    }

    fn slots(&self) -> impl Iterator<Item = &Slot> {
        self.pages.iter().flat_map(|p| p.slots.iter())
    }

    fn position(&self, val: &Val) -> Option<(usize, usize)> {
        for (page_idx, page) in self.pages.iter().enumerate() {
            for (slot_idx, slot) in page.slots.iter().enumerate() {
                if slot.contains(val) {
                    return Some((page_idx, slot_idx));
                }
            }
        }
        None
    }

    fn last_non_empty_idx(&self) -> Option<usize> {
        self.slots()
            .enumerate()
            .filter(|(_, s)| s.bytes_used() > 0)
            .map(|(idx, _)| idx)
            .max()
    }

    /// Iterate over double words as they would appear in the FAU
    pub fn iter_entries(&self) -> impl Iterator<Item = u64> + use<'_> {
        let end = self.last_non_empty_idx().map(|i| i + 1).unwrap_or(0);
        self.slots().take(end).map(|s| s.val())
    }

    pub fn byte_position(&self, val: &Val) -> Option<usize> {
        for (slot_idx, slot) in self.slots().enumerate() {
            if let Some(val_idx) = slot.position(val) {
                return Some(slot_idx * 8 + val_idx);
            }
        }
        None
    }

    pub fn contains(&self, val: &Val) -> bool {
        self.position(val).is_some()
    }
}

impl std::fmt::Display for FauAlloc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (page_idx, page) in self.pages.iter().enumerate() {
            writeln!(f, "--- Page {page_idx}:")?;
            for slot in &page.slots {
                write!(f, "{slot}\n")?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ConstEntry {
    I8(NonZeroU8),
    I16(NonZeroU16),
    I32(NonZeroU32),
    I64(NonZeroU64),
}

fn bitmask32_from_byte(bytes_read: u8) -> u32 {
    let mut acc = 0;
    for idx in 0..8 {
        if (1 << idx) & bytes_read > 0 {
            acc |= 0xff << (idx * 8);
        }
    }
    acc
}

impl ConstEntry {
    fn from_src_promotable(
        model: &dyn Model,
        op: &Op,
        src: &Src,
    ) -> Option<Self> {
        match &src.src_ref {
            SrcRef::Imm32(v) => {
                // Already encodable inline, so it needs no FAU slot
                if model.op_src_supports_imm32(op, src, v.get()) {
                    None
                } else {
                    let t = op.src_type(src);
                    let read_mask = src.swizzle.bytes_read(t.total_bytes());
                    let v = v.get() & bitmask32_from_byte(read_mask);
                    match u8::BITS - read_mask.leading_zeros() {
                        1 => {
                            Some(ConstEntry::I8((v as u8).try_into().unwrap()))
                        }
                        2 => Some(ConstEntry::I16(
                            (v as u16).try_into().unwrap(),
                        )),
                        3 | 4 => Some(ConstEntry::I32(v.try_into().unwrap())),
                        _ => unreachable!("must be Imm64"),
                    }
                }
            }
            SrcRef::Imm64(v) => Some(ConstEntry::I64(*v)),
            _ => None,
        }
    }

    fn bytes(self) -> u8 {
        match self {
            ConstEntry::I8(_) => 1,
            ConstEntry::I16(_) => 2,
            ConstEntry::I32(_) => 4,
            ConstEntry::I64(_) => 8,
        }
    }
}

struct ConstInstance<'a> {
    instr: &'a Instr,
    src_idx: usize,
}

#[derive(Default)]
struct ConstTable<'a>(FxHashMap<ConstEntry, SmallVec<ConstInstance<'a>>>);

impl<'a> ConstTable<'a> {
    fn add_op_src(&mut self, model: &dyn Model, instr: &'a Instr, src: &Src) {
        let op = &instr.op;
        let Some(key) = ConstEntry::from_src_promotable(model, op, src) else {
            return;
        };

        let src_idx = instr.src_idx(src);
        let inst = ConstInstance { instr, src_idx };
        self.0.entry(key).or_default().push(inst);
    }

    fn into_sorted(self) -> Vec<(ConstEntry, SmallVec<ConstInstance<'a>>)> {
        let mut weighted: Vec<_> = self.0.into_iter().collect();
        // Break ties using the entry itself
        weighted.sort_by_key(|(e, v)| Reverse((v.len(), *e)));
        weighted
    }
}

fn fau_available(fau: &pan_fau_layout) -> u32 {
    unsafe { pan_fau_available(fau) as u32 }
}

fn fau_emit_const(fau: &mut pan_fau_layout, imm: u32) -> u16 {
    unsafe { pan_fau_emit_const(fau, imm) }.try_into().unwrap()
}

impl From<ConstEntry> for Val {
    fn from(value: ConstEntry) -> Self {
        match value {
            ConstEntry::I8(imm8) => Val::new(1, imm8.get().into()),
            ConstEntry::I16(imm16) => Val::new(2, imm16.get().into()),
            ConstEntry::I32(imm32) => Val::new(4, imm32.get().into()),
            ConstEntry::I64(imm64) => Val::new(8, imm64.get()),
        }
    }
}

impl From<Val> for ConstEntry {
    fn from(value: Val) -> Self {
        match value.byte_size {
            1 => ConstEntry::I8(
                NonZeroU8::new(value.val.try_into().unwrap()).unwrap(),
            ),
            2 => ConstEntry::I16(
                NonZeroU16::new(value.val.try_into().unwrap()).unwrap(),
            ),
            4 => ConstEntry::I32(
                NonZeroU32::new(value.val.try_into().unwrap()).unwrap(),
            ),
            8 => ConstEntry::I64(NonZeroU64::new(value.val).unwrap()),
            _ => panic!("unexpected size"),
        }
    }
}

fn promote_source(
    instr: &Instr,
    src_idx: usize,
    alloc: &mut FauAlloc,
    model: &dyn Model,
) {
    let Some(entry) = ConstEntry::from_src_promotable(
        model,
        &instr.op,
        &instr.srcs()[src_idx],
    ) else {
        return;
    };

    if alloc.contains(&entry.into()) {
        return;
    }

    // Find the first already promoted source.
    let mut insert_with_entry = None;
    for other_idx in 0..instr.srcs().len() {
        if other_idx == src_idx {
            continue;
        }

        let Some(other) = ConstEntry::from_src_promotable(
            model,
            &instr.op,
            &instr.srcs()[other_idx],
        ) else {
            continue;
        };

        if alloc.contains(&other.into()) {
            insert_with_entry = Some(other);
            break;
        }
    }

    if let Some(other) = insert_with_entry {
        // This is not precise but good enough.
        if model.op_is_message(&instr.op) || !model.fau().single_fau_ram_index {
            alloc.insert_same_page(entry.into(), &other.into());
        } else {
            alloc.insert_same_slot(entry.into(), &other.into());
        }
    } else {
        alloc.insert(entry.into());
    }
}

fn promote_consts(s: &mut Shader, fau: &mut pan_fau_layout) {
    let mut const_table = ConstTable::default();
    for block in s.blocks.iter() {
        for instr in block.instrs.iter() {
            for src in instr.srcs() {
                const_table.add_op_src(s.model, instr, src);
            }
        }
    }

    // Align to a full 64-bit slot.
    // TODO: Find a nice way to fill this with a 32bit val if possible.
    if fau.count % 2 != 0 && fau_available(fau) > 0 {
        fau_emit_const(fau, 0xdeadbeef);
    }

    let fau_page_size = s.model.fau().user_page_words() * 4;
    let fau_offset = (fau.count as usize * 4) % fau_page_size;
    let mut alloc = FauAlloc::new(
        (fau_available(fau) * 4).try_into().unwrap(),
        fau_page_size,
        fau_offset,
    );

    for (_, instances) in const_table.into_sorted() {
        for ConstInstance { instr, src_idx } in instances {
            promote_source(instr, src_idx, &mut alloc, s.model);
        }
    }

    let first_emitted_idx = fau.count as u16;
    for entry in alloc.iter_entries() {
        debug_assert!(fau_available(fau) > 0);

        let lo = entry as u32;
        let hi = (entry >> 32) as u32;

        fau_emit_const(fau, lo);

        debug_assert!(!(hi > 0) || fau_available(fau) > 0);
        if fau_available(fau) > 0 {
            fau_emit_const(fau, hi);
        }
    }

    for block in s.blocks.iter_mut() {
        for instr in block.instrs.iter_mut() {
            for src_idx in 0..instr.srcs().len() {
                let src = &instr.srcs()[src_idx];

                let Some(entry) =
                    ConstEntry::from_src_promotable(s.model, &instr.op, src)
                else {
                    continue;
                };
                let Some(byte_idx) = alloc.byte_position(&entry.into()) else {
                    continue;
                };

                let align = entry.bytes();
                let offset_in_word = (byte_idx as u8) % 4;
                let extract_swizzle = match (align, offset_in_word) {
                    (1, offset) => Swizzle::replicate_byte(offset),
                    (2, offset) if offset % 2 == 0 => {
                        Swizzle::replicate_half(offset / 2)
                    }
                    (4, offset) if offset % 4 == 0 => Swizzle::NONE,
                    (8, 0) => Swizzle::NONE,
                    _ => panic!(
                        "invalid (align={}, offset={})",
                        align, offset_in_word
                    ),
                };

                let fau_word_idx = first_emitted_idx + (byte_idx / 4) as u16;
                debug_assert!(align < 8 || fau_word_idx % 2 == 0);

                let fau_ref = match entry {
                    ConstEntry::I8(_)
                    | ConstEntry::I16(_)
                    | ConstEntry::I32(_) => FAURef::user_i32(fau_word_idx),
                    ConstEntry::I64(_) => FAURef::user_i64(fau_word_idx),
                };

                instr.srcs_mut()[src_idx].swizzle =
                    extract_swizzle.swizzle(src.swizzle).unwrap();
                instr.srcs_mut()[src_idx].src_ref = SrcRef::FAU(fau_ref);
            }
        }
    }
}

impl Shader<'_> {
    pub fn opt_promote_consts(&mut self, fau: &mut pan_fau_layout) {
        promote_consts(self, fau);
    }
}
