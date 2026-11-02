// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use crate::ir::{DataType, SmallConstant};
use compiler::enum_as_u8::*;

#[derive(Debug)]
pub enum EncodeError {
    Int(std::num::TryFromIntError),
    Str(&'static str),
}

impl From<std::convert::Infallible> for EncodeError {
    fn from(_err: std::convert::Infallible) -> EncodeError {
        panic!("Invallable can't happen");
    }
}

impl From<std::num::TryFromIntError> for EncodeError {
    fn from(err: std::num::TryFromIntError) -> EncodeError {
        EncodeError::Int(err)
    }
}

impl From<&'static str> for EncodeError {
    fn from(err: &'static str) -> EncodeError {
        EncodeError::Str(err)
    }
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::Str(s) => write!(f, "{}", s),
            EncodeError::Int(e) => write!(f, "{}", e),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum InvalidInstrError {
    Any,
    Encoding(EncodeError),
}

impl From<std::convert::Infallible> for InvalidInstrError {
    fn from(_err: std::convert::Infallible) -> InvalidInstrError {
        panic!("Infallible can't happen");
    }
}

impl From<EncodeError> for InvalidInstrError {
    fn from(err: EncodeError) -> InvalidInstrError {
        InvalidInstrError::Encoding(err)
    }
}

impl std::fmt::Display for InvalidInstrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvalidInstrError::Any => write!(f, "Invalid instruction"),
            InvalidInstrError::Encoding(e) => {
                write!(f, "Invalid instruction ({e})")
            }
        }
    }
}

struct ArchSet {
    bits: u32,
}

impl ArchSet {
    pub fn contains(&self, arch: u8) -> bool {
        assert!(arch < 32);
        ((self.bits >> arch) & 1) != 0
    }
}

pub trait Encode {
    type Encoded;

    fn encode(self) -> Self::Encoded;
}

pub trait TryEncode {
    type Encoded;
    type Error;

    fn try_encode(self, arch: u8) -> Result<Self::Encoded, Self::Error>;
}

impl<T: Encode> TryEncode for T {
    type Encoded = <T as Encode>::Encoded;
    type Error = std::convert::Infallible;

    fn try_encode(self, _arch: u8) -> Result<Self::Encoded, Self::Error> {
        Ok(self.encode())
    }
}

pub trait TryDecode<T>: Sized {
    type Error;

    fn try_decode(value: T, arch: u8) -> Result<Self, Self::Error>;
}

pub trait SmallConstantTable: TryDecode<u8> {
    const TABLE_LEN: u8;
    fn name(&self) -> &'static str;
    fn bit_pattern(&self) -> u32;

    fn collect(arch: u8) -> Vec<SmallConstant> {
        let mut vec = Vec::new();
        for idx in 0..Self::TABLE_LEN {
            if let Ok(sc) = Self::try_decode(idx, arch) {
                vec.push(SmallConstant {
                    idx,
                    imm32: sc.bit_pattern(),
                    name: sc.name(),
                });
            }
        }
        vec
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExecUnit {
    Cvt,
    Fma,
    Msg,
    Sfu,
}

#[allow(dead_code)]
pub trait FauSpecialPageResolver {
    type P0: TryDecode<u8> + std::fmt::Display;
    type P1: TryDecode<u8> + std::fmt::Display;
    type P3: TryDecode<u8> + std::fmt::Display;

    fn get_name(page: u8, idx: u8, arch: u8) -> Result<String, EncodeError>
    where
        EncodeError: From<<Self::P0 as TryDecode<u8>>::Error>,
        EncodeError: From<<Self::P1 as TryDecode<u8>>::Error>,
        EncodeError: From<<Self::P3 as TryDecode<u8>>::Error>,
    {
        match page {
            0 => Self::P0::try_decode(idx, arch)
                .map(|x| format!("{}", x))
                .map_err(|e| e.into()),
            1 => Self::P1::try_decode(idx, arch)
                .map(|x| format!("{}", x))
                .map_err(|e| e.into()),
            3 => Self::P3::try_decode(idx, arch)
                .map(|x| format!("{}", x))
                .map_err(|e| e.into()),
            _ => Err("Invalid fau_page_index".into()),
        }
    }
}

pub struct InstructionSrcInfo<S: EnumAsU8> {
    pub allowed_swizzles: U8EnumSet<S, 2>,
    pub is_src64: bool,
    pub has_abs: bool,
    pub has_neg: bool,
    pub has_not: bool,
    // If it's a staging-register that reads
    // register_format/vecsize
    #[allow(dead_code)]
    pub has_vecsize: bool,
}

impl<S: EnumAsU8> InstructionSrcInfo<S> {
    #[allow(dead_code)]
    pub fn exists(&self) -> bool {
        !self.allowed_swizzles.is_empty()
    }
}

pub struct InstructionDstInfo<L: EnumAsU8> {
    pub is_sr: bool,
    pub allowed_lanes: U8EnumSet<L, 1>,
    // If it's a staging-register that reads
    // register_format/vecsize
    pub has_vecsize: bool,
}

pub struct InstructionInfo<S: EnumAsU8 + 'static, L: EnumAsU8 + 'static> {
    pub exec_unit: ExecUnit,
    pub exec_time: u8,
    pub is_message: bool,
    pub srcs: &'static [InstructionSrcInfo<S>],
    pub sr_src: Option<InstructionSrcInfo<S>>,
    pub dst: Option<InstructionDstInfo<L>>,
}

pub trait Instruction<S: EnumAsU8 + 'static, L: EnumAsU8 + 'static> {
    type Variant;

    fn get_info_for_variant(
        variant: Self::Variant,
        arch: u8,
    ) -> Option<&'static InstructionInfo<S, L>>;

    fn get_info(
        variant: impl TryInto<Self::Variant>,
        arch: u8,
    ) -> Option<&'static InstructionInfo<S, L>> {
        Self::get_info_for_variant(variant.try_into().ok()?, arch)
    }

    fn is_supported(variant: impl TryInto<Self::Variant>, arch: u8) -> bool {
        Self::get_info(variant, arch).is_some()
    }
}

#[derive(Clone, Copy)]
pub struct EncodedSrc<S: Copy> {
    pub encoded: u8,
    pub swizzle: S,
    pub abs: bool,
    pub neg: bool,
    pub not: bool,
}

#[derive(Clone, Copy)]
pub struct EncodedDst<L: Copy> {
    pub reg: u8,
    pub lanes: L,
}

#[derive(Clone, Copy)]
pub struct SrRead {
    pub index: u8,
    #[allow(dead_code)]
    pub count: u8,
    pub data_type: DataType,
}

#[derive(Clone, Copy)]
pub struct SrReadSwizzle<S: Copy> {
    pub index: u8,
    #[allow(dead_code)]
    pub count: u8,
    pub swizzle: S,
}

#[derive(Clone, Copy)]
pub struct SrWrite {
    pub index: u8,
    #[allow(dead_code)]
    pub count: u8,
    pub data_type: DataType,
}

#[derive(Clone, Copy)]
pub struct SrWriteLanes<L: Copy> {
    pub index: u8,
    #[allow(dead_code)]
    pub count: u8,
    pub lanes: L,
}

pub mod v9 {
    use kraid_proc_macros::*;
    gen_isa_encode!(9..=14);

    #[cfg(kraid_disasm)]
    pub mod decode {
        use super::*;
        use crate::isa::*;
        use rustc_hash::FxHashMap;

        enum DecodeStage {
            Inner {
                mask: u64,
                children: &'static [(u64, usize)],
            },
            // (mask, value, instr index)
            Leaf(&'static [(u64, u64, usize)]),
        }

        fn try_decode_table(
            instr: u64,
            arch: u8,
            table: &[DecodeStage],
        ) -> Result<usize, InvalidInstrError> {
            let mut stage_idx = 0;
            'stages: loop {
                match table[stage_idx] {
                    DecodeStage::Inner { mask, children } => {
                        let masked = instr & mask;
                        for (val, next) in children {
                            if masked == *val {
                                assert!(stage_idx < *next);
                                stage_idx = *next;
                                continue 'stages;
                            }
                        }
                        // Failed to decode
                        return Err(InvalidInstrError::Any);
                    }
                    DecodeStage::Leaf(options) => {
                        for (mask, value, instr_idx) in options {
                            let masked = instr & mask;
                            if masked == *value
                                && try_instr(*instr_idx, instr, arch).is_ok()
                            {
                                return Ok(*instr_idx);
                            }
                        }
                        // Failed to decode
                        return Err(InvalidInstrError::Any);
                    }
                }
            }
        }

        enum SourceEncodingX<const IS64: bool> {
            Register {
                idx: u8,
                last: bool,
                zext: bool,
            },
            Fau {
                idx: u8,
                page: u8,
                fau32: bool,
                zext: bool,
            },
            SmallConst(super::SmallConstantT),
            FauSpec {
                word_select: bool,
                name: String,
                zext: bool,
            },
        }

        impl<const IS64: bool> SourceEncodingX<IS64> {
            fn try_decode(
                v: u8,
                arch: u8,
                fau_page_index: u8,
                fau32: bool,
            ) -> std::result::Result<SourceEncodingX<IS64>, EncodeError>
            {
                if arch > 14 {
                    return Err("Only supports up to v14 atm".into());
                }
                /* V14 onwards, bit 0 sets the .zext flag in 64 bit sources */
                let bit0_is_zext = IS64 && arch >= 14;
                let bit0 = (v & 1) != 0;
                let zext = bit0_is_zext && bit0;
                let mode = (v >> 6) & 0b11;
                let mode2 = (v >> 5) & 0b1;
                match (mode, mode2) {
                    (0b00, _) | (0b01, _) => Ok(SourceEncodingX::Register {
                        idx: if bit0_is_zext { v & 0x3e } else { v & 0x3f },
                        last: mode != 0,
                        zext,
                    }),
                    (0b10, _) => Ok(SourceEncodingX::Fau {
                        idx: if bit0_is_zext { v & 0x3e } else { v & 0x3f },
                        page: fau_page_index,
                        fau32,
                        zext,
                    }),
                    (0b11, 0b0) => {
                        let as_enum =
                            SmallConstantT::try_decode((v & 0x1f) as u8, arch)?;
                        Ok(SourceEncodingX::SmallConst(as_enum))
                    }
                    (0b11, 0b1) => {
                        let idx32 = (v & 0x1f) >> 1;
                        let name = FauSpecialIndexPage::get_name(
                            fau_page_index,
                            idx32,
                            arch,
                        )?;
                        Ok(SourceEncodingX::FauSpec {
                            word_select: !IS64 && bit0,
                            name,
                            zext,
                        })
                    }
                    _ => Err("Invalid SourceEncoding".into()),
                }
            }
        }

        impl<const IS64: bool> std::fmt::Display for SourceEncodingX<IS64> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    SourceEncodingX::Register { idx, last, zext } => {
                        let tag = if *last { "^" } else { "" };
                        let zext_tag = if *zext { ".zext" } else { "" };
                        if IS64 {
                            write!(
                                f,
                                "[r{}{}:r{}{}]{}",
                                idx,
                                tag,
                                idx + 1,
                                tag,
                                zext_tag
                            )
                        } else {
                            write!(f, "r{}{}", idx, tag)
                        }
                    }
                    SourceEncodingX::SmallConst(value) => {
                        write!(f, "{}", value)
                    }
                    SourceEncodingX::Fau {
                        idx,
                        page,
                        fau32,
                        zext,
                    } => {
                        let zext_tag = if *zext { ".zext" } else { "" };
                        if *fau32 {
                            write!(f, "u{}{}", 64 * page + idx, zext_tag)
                        } else {
                            let hl = if IS64 {
                                ""
                            } else if (idx & 0b1) == 1 {
                                ".w1"
                            } else {
                                ".w0"
                            };
                            let idx32 = idx >> 1;
                            write!(
                                f,
                                "u{}{}{}",
                                32 * page + idx32,
                                hl,
                                zext_tag
                            )
                        }
                    }
                    SourceEncodingX::FauSpec {
                        name,
                        word_select,
                        zext,
                        ..
                    } => {
                        let zext_tag = if *zext { ".zext" } else { "" };
                        let hl = if IS64 {
                            ""
                        } else if *word_select {
                            ".w1"
                        } else {
                            ".w0"
                        };
                        write!(f, "{}{}{}", &name, hl, zext_tag)
                    }
                }
            }
        }

        type SourceEncoding = SourceEncodingX<false>;
        type SourceEncoding64 = SourceEncodingX<true>;

        type DecodeResult<T> = Result<Option<Box<T>>, InvalidInstrError>;
        type DecodeFn = fn(
            u32,
            &FxHashMap<&'static str, u32>,
            u8,
            &PrintCtx,
        ) -> DecodeResult<dyn std::fmt::Display>;

        enum LoadFn {
            Physical(fn(u64) -> u32),
            Virtual(
                fn(
                    &FxHashMap<&'static str, u32>,
                    u8,
                ) -> Result<u32, EncodeError>,
            ),
        }

        struct Field {
            name: &'static str,
            is_mod: bool,
            load: LoadFn,
            decode: DecodeFn,
        }

        pub struct Instr {
            name: &'static str,
            variant: Option<&'static str>,
            fields: &'static [u32],
        }

        impl Instr {
            /// Iterate over fields in the order they appear in the asm syntax.
            fn iter_fields(&self) -> impl Iterator<Item = &Field> {
                self.fields.iter().map(|idx| &FIELDS[*idx as usize])
            }
        }

        fn try_instr(
            instr_idx: usize,
            val: u64,
            arch: u8,
        ) -> Result<(), InvalidInstrError> {
            print_instr_int(
                &DISASM_TABLE[instr_idx],
                val,
                arch,
                &mut std::io::sink(),
                &Default::default(),
            )
            .map_err(|e| e.into_invalid_instr())
        }

        pub fn print_instr(
            instr: &Instr,
            val: u64,
            arch: u8,
            f: &mut dyn std::io::Write,
            ctx: &PrintCtx,
        ) -> std::io::Result<()> {
            let res = print_instr_int(instr, val, arch, f, ctx);
            match res {
                Err(PrintError::InvalidInstr(e)) => write!(f, "{}", e),
                Err(PrintError::Fmt(_)) => write!(f, "<format error>"),
                Err(PrintError::Io(e)) => Err(e),
                Ok(()) => Ok(()),
            }
        }

        fn print_instr_int(
            instr: &Instr,
            val: u64,
            arch: u8,
            f: &mut dyn std::io::Write,
            ctx: &PrintCtx,
        ) -> Result<(), PrintError> {
            let mut values: FxHashMap<&'static str, u32> = Default::default();

            for field in instr.iter_fields() {
                let LoadFn::Physical(load) = &field.load else {
                    continue;
                };
                values.insert(field.name, load(val));
            }

            for field in instr.iter_fields() {
                let LoadFn::Virtual(load) = &field.load else {
                    continue;
                };
                values.insert(field.name, load(&values, arch)?);
            }

            write!(f, "{}", instr.name)?;
            if let Some(variant) = instr.variant {
                write!(f, ".{}", variant)?;
            }

            let mut had_non_mod = false;

            for field in instr.iter_fields() {
                if let Some(val) = (field.decode)(
                    *values.get(field.name).unwrap(),
                    &values,
                    arch,
                    ctx,
                )? {
                    if field.is_mod {
                        write!(f, ".")?;
                    } else if had_non_mod {
                        write!(f, ", ")?;
                    } else {
                        write!(f, " ")?;
                    }
                    write!(f, "{val}")?;

                    had_non_mod |= !field.is_mod;
                }
            }

            Ok(())
        }

        gen_isa_decode!(9..=14);
    }
}
