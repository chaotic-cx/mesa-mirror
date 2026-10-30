// Copyright (c) 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

use crate::ident;
use crate::isa::*;

fn bitmask(n: u8) -> u64 {
    (1 << n) - 1
}

#[derive(Clone, Copy)]
struct DecoderVal<'a> {
    fixed_mask: u64,
    fixed_value: u64,
    instr: &'a Instr,
}

impl DecoderVal<'_> {
    fn new(instr: &Instr) -> DecoderVal<'_> {
        let (fixed_mask, fixed_value) = instr
            .fields
            .iter()
            .filter_map(|f| match f {
                InstrField::Physical(PhysicalField {
                    exact: Some(v),
                    bits,
                    ..
                }) => Some((
                    bitmask(bits.end - bits.start) << bits.start,
                    v << bits.start,
                )),
                _ => None,
            })
            .fold((0, 0), |(amask, aval), (mask, value)| {
                (amask | mask, aval | value)
            });

        DecoderVal {
            instr,
            fixed_mask,
            fixed_value,
        }
    }

    fn get_specificity(&self) -> u32 {
        self.fixed_mask.count_ones()
    }
}

impl Ord for DecoderVal<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_specificity()
            .cmp(&other.get_specificity())
            .reverse()
            .then(self.instr.name.as_str().cmp(other.instr.name.as_str()))
    }
}

impl PartialOrd for DecoderVal<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DecoderVal<'_> {
    fn eq(&self, other: &Self) -> bool {
        self <= other && self >= other
    }
}

impl Eq for DecoderVal<'_> {}

enum DecoderNode<'a> {
    Leaf {
        options: Vec<DecoderVal<'a>>,
    },
    Internal {
        mask: u64,
        children: HashMap<u64, Box<DecoderNode<'a>>>,
    },
}

impl DecoderNode<'_> {
    fn build<'a>(values: Vec<&'a Instr>) -> DecoderNode<'a> {
        return DecoderNode::build_internal(
            values.into_iter().map(DecoderVal::new).collect(),
            0,
        );
    }

    fn build_internal<'a>(
        values: Vec<DecoderVal<'a>>,
        already_used_bits: u64,
    ) -> DecoderNode<'a> {
        let mut biggest_mask = !(0 as u64);
        for v in &values {
            biggest_mask &= v.fixed_mask;
        }
        biggest_mask &= !already_used_bits;

        let mut buckets: HashMap<u64, Vec<DecoderVal>> = Default::default();
        for v in values {
            let key = v.fixed_value & biggest_mask;
            buckets.entry(key).or_insert(vec![]).push(v);
        }

        if buckets.len() == 1 {
            let mut all_values: Vec<DecoderVal> =
                buckets.into_values().next().unwrap();
            all_values.sort();

            return DecoderNode::Leaf {
                options: all_values,
            };
        }

        let mut children: HashMap<u64, Box<DecoderNode>> = Default::default();
        for (key, sub_values) in buckets {
            let node = Box::new(DecoderNode::build_internal(
                sub_values,
                already_used_bits | biggest_mask,
            ));
            children.insert(key, node);
        }

        DecoderNode::Internal {
            mask: biggest_mask,
            children: children,
        }
    }

    fn to_tokens<F, G>(&self, elem_cb: &F, check_cb: &G) -> TokenStream
    where
        F: Fn(&Instr) -> TokenStream,
        G: Fn(&Instr) -> TokenStream,
    {
        match self {
            DecoderNode::Leaf { options } => {
                let cases: TokenStream = options
                    .into_iter()
                    .map(|option| {
                        let fixed_mask = option.fixed_mask;
                        let expected_masked =
                            option.fixed_value & option.fixed_mask;
                        let check_ts = check_cb(option.instr);
                        let val_ts = elem_cb(option.instr);
                        quote! {
                            if (input_value & #fixed_mask) == #expected_masked {
                                if (#check_ts).is_ok() {
                                    return Ok(#val_ts);
                                }
                            }
                        }
                    })
                    .collect();

                quote! {
                    {
                        let mut result = Err(InvalidInstrError::Any);
                        #cases
                        return result;
                    }
                }
            }
            DecoderNode::Internal { mask, children } => {
                // For better readability, sort the keys for the output.
                let mut sorted: Vec<_> = children.keys().collect();
                sorted.sort();

                let cases: TokenStream = sorted
                    .into_iter()
                    .map(|key| {
                        let child_impl = children
                            .get(key)
                            .expect("Key must exist")
                            .to_tokens(elem_cb, check_cb);
                        quote! {
                            #key => #child_impl,
                        }
                    })
                    .collect();

                quote! {
                    match input_value & #mask {
                        #cases
                        _ => Err(InvalidInstrError::Any)
                    }
                }
            }
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
struct LoadField {
    name: String,
    load_fn: LoadFn,
    decode_fn: DecodeFn,
    is_mod: bool,
}

impl LoadField {
    fn new(name: &str, instr: &Instr) -> LoadField {
        let field = instr.get_named_field(name).unwrap();
        LoadField {
            name: String::from(name),
            load_fn: field.into(),
            // Default to not printing. Decoding still checks field restrictions.
            decode_fn: DecodeFn::new(field, None),
            is_mod: false,
        }
    }

    fn set_decode(&mut self, print_as: PrintAs, instr: &Instr) {
        let field = instr.get_named_field(&self.name).unwrap();
        self.decode_fn = DecodeFn::new(field, Some(&print_as));
    }

    fn table_entry(&self, ts: &mut TokenStream) {
        let name = &self.name;
        let is_mod = self.is_mod;
        let load_fn = self.load_fn.ref_ts();
        let decode_fn = self.decode_fn.ref_ts();
        ts.extend(quote! {
            Field {
                name: #name,
                is_mod: #is_mod,
                load: #load_fn,
                decode: #decode_fn,
            },
        });
    }
}

struct PrintAsStaging {
    count_id: Ident,
    offset: u8,
}

impl PrintAsStaging {
    fn new(input: bool, instr: &Instr) -> PrintAsStaging {
        let count_field = instr
            .get_sr_count_field(input)
            .expect("Missing sr count field");

        let count_id = count_field
            .ident()
            .cloned()
            .expect("Missing ident on sr count field");

        // TODO: Resolve fields with "exact" values to their constant value.
        // Make PrintAsStaging an enum and have a "Const" variant.

        // sr_count_t starts from 0. sr_write_count_t and ls_multi_sr_count start
        // from 1, meaning u8 == 0, means 1 reg is written.
        let offset: u8 = match count_field.field_type() {
            Some(FieldType::Enum(e))
                if e.name == "sr_write_count_t"
                    || e.name == "ls_multi_sr_count_m" =>
            {
                1
            }
            _ => 0,
        };

        PrintAsStaging { count_id, offset }
    }
}

pub enum ImmType {
    Uint,
    Int,
    Float,
    V2F16,
    V2I16,
    V4I8,
}

impl From<instr::ImmType> for ImmType {
    fn from(t: instr::ImmType) -> ImmType {
        match t {
            instr::ImmType::Uint => ImmType::Uint,
            instr::ImmType::Int => ImmType::Int,
            instr::ImmType::Float => ImmType::Float,
            instr::ImmType::V2F16 => ImmType::V2F16,
            instr::ImmType::V2I16 => ImmType::V2I16,
            instr::ImmType::V4I8 => ImmType::V4I8,
        }
    }
}

enum PrintAs {
    Id,
    Imm(ImmType),
    Src(bool),
    Dst(bool),
    Staging(PrintAsStaging),
    PcRelOffset,
}

impl PrintAs {
    fn additional_input_names(&self) -> Vec<String> {
        match self {
            PrintAs::PcRelOffset
            | PrintAs::Imm(_)
            | PrintAs::Dst(_)
            | PrintAs::Id => vec![],
            PrintAs::Src(_) => vec!["fau_page_index".into()],
            PrintAs::Staging(s) => {
                vec![s.count_id.to_string()]
            }
        }
    }

    fn to_tokens(&self, ident: &Ident, ts: &mut TokenStream) {
        match &self {
            PrintAs::Id => ts.extend(quote! {#ident}),
            PrintAs::Dst(is64) => {
                ts.extend(quote! {
                    PrintDst { reg: (#ident as u8), is64: #is64 }
                });
            }
            PrintAs::Imm(t) => {
                ts.extend(match t {
                    ImmType::V4I8 | ImmType::V2F16 | ImmType::V2I16 | // TODO Treat these correctly
                    ImmType::Float | ImmType::Uint | ImmType::Int => {
                        quote! { PrintImm{ val: #ident } }
                    }
                });
            }
            PrintAs::Src(is64) => {
                let n = if *is64 {
                    "SourceEncoding64"
                } else {
                    "SourceEncoding"
                };
                let cls_id = Ident::new(n, Span::call_site());
                ts.extend(quote! { #cls_id::try_decode(
                #ident as u8, arch, (fau_page_index as u8), ctx.fau32)? });
            }
            PrintAs::Staging(s) => {
                let count_id = &s.count_id;
                let offset = s.offset;
                ts.extend(quote! {
                    {
                        assert!(#ident <= u8::MAX as u32);
                        assert!(#count_id <= u8::MAX as u32);
                        PrintStaging {
                            regs: (#ident as u8)..(
                                      (#ident as u8) + (#count_id as u8) + #offset)
                        }
                    }
                });
            }
            PrintAs::PcRelOffset => {
                ts.extend(quote! {
                     PrintPcRelOffset{ val: #ident, pc_base: ctx.pc_base }
                });
            }
        };
    }

    fn declare_types(ts: &mut TokenStream) {
        ts.extend(quote! {
            struct PrintDst {
                reg: u8,
                is64: bool,
            }
            impl std::fmt::Display for PrintDst {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    if self.is64 {
                        write!(f, "[r{}:r{}]", self.reg, self.reg + 1)
                    } else {
                        write!(f, "r{}", self.reg)
                    }
               }
            }

            struct PrintImm<T> {
                val: T,
            }

            impl std::fmt::Display for PrintImm<i32>
            {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    if self.val < 0 {
                        write!(f, "#-{:#x}", -self.val)
                    } else {
                        write!(f, "#{:#x}", self.val)
                    }
                }
            }

            impl std::fmt::Display for PrintImm<u32>
            {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "#{:#x}", self.val)
                }
            }

            struct PrintStaging {
                regs: core::ops::Range<u8>,
            }

            impl std::fmt::Display for PrintStaging {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "@r{}", self.regs.start)?;
                    for r in ((self.regs.start + 1)..(self.regs.end)) {
                        write!(f, ":r{}", r)?;
                    }
                    Ok(())
                }
            }

            struct PrintPcRelOffset<T> {
                val: T,
                pc_base: u64,
            }

            impl std::fmt::Display for PrintPcRelOffset<i32> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "0x{:x}", self.pc_base.wrapping_add((self.val + 8) as u64))
               }
            }

            impl std::fmt::Display for PrintPcRelOffset<u32> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "0x{:x}", self.pc_base + self.val as u64 + 8)
               }
            }
        });
    }
}

struct InstrFields<'a> {
    instr: &'a Instr,
    loads: BTreeMap<String, (usize, LoadField)>,
}

impl<'a> InstrFields<'a> {
    pub fn new(instr: &Instr) -> InstrFields<'_> {
        InstrFields {
            instr,
            loads: Default::default(),
        }
    }

    pub fn fields(&self) -> impl Iterator<Item = &LoadField> {
        let mut sorted: Vec<_> = self.loads.values().collect();
        sorted.sort_by_key(|(idx, _)| *idx);
        sorted.into_iter().map(|(_, l)| l)
    }

    pub fn load(&mut self, name: &str) -> &mut LoadField {
        let field = self.instr.get_named_field(name).unwrap();
        let next_idx = self.loads.len();

        let (_, ret) = match field {
            InstrField::Physical(f) => {
                self.loads.entry(f.name.clone()).or_insert((
                    next_idx,
                    LoadField::new(f.name.as_str(), self.instr),
                ))
            }
            InstrField::Virtual(f) => {
                self.loads.entry(f.name.clone()).or_insert((
                    next_idx,
                    LoadField::new(f.name.as_str(), self.instr),
                ));

                for used_field in &f.expr.fields() {
                    self.load(&used_field.name);
                }

                self.loads.get_mut(&f.name).unwrap()
            }
            InstrField::Reserved(_) => unreachable!("Can't load reserved"),
        };

        ret
    }
}

struct InstrPrint<'a> {
    instr: &'a Instr,
    fields: InstrFields<'a>,
}

impl InstrPrint<'_> {
    pub fn new(instr: &Instr) -> InstrPrint<'_> {
        let mut fields = InstrFields::new(instr);

        // Collect all values needed for printing. This includes values that are
        // not directly printed, but used to compute printed virtual fields.
        for elem in &instr.syntax.elements {
            match elem {
                SyntaxElement::Name(n) => {
                    for mod_ in &n.mods {
                        let ld = fields.load(&mod_.name);
                        ld.set_decode(PrintAs::Id, instr);
                        ld.is_mod = true;
                    }
                }
                SyntaxElement::Staging(s) => {
                    let field = instr.get_sr_index_field(s.is_input).unwrap();
                    let print_as = PrintAsStaging::new(s.is_input, instr);
                    fields.load(&format!("{}", print_as.count_id));
                    let ld = fields.load(&field.name().unwrap());
                    ld.set_decode(PrintAs::Staging(print_as), instr);
                }
                SyntaxElement::Imm(s) => {
                    let ld = fields.load(&s.name);
                    ld.set_decode(PrintAs::Imm(s.type_.into()), instr);
                }
                SyntaxElement::Dst(d) => {
                    let fname = "dest_register_index";
                    let is64 = matches!(
                        instr.variant.as_deref(),
                        Some("i64") | Some("u64") | Some("s64")
                    );
                    let ld = fields.load(fname);
                    ld.set_decode(PrintAs::Dst(is64), instr);

                    for mod_ in &d.mods {
                        let ld = fields.load(&mod_.name);
                        ld.set_decode(PrintAs::Id, instr);
                        ld.is_mod = true;
                    }
                }
                SyntaxElement::PcRelOffset(o) => {
                    let ld = fields.load(&o.name);
                    ld.set_decode(PrintAs::PcRelOffset, instr);
                }
                SyntaxElement::Src(s) => {
                    let field = instr.get_named_field(&s.name).unwrap();
                    let is64 =
                        matches!(field.field_type(), Some(FieldType::Source64));
                    let ld = fields.load(&s.name);
                    ld.set_decode(PrintAs::Src(is64), instr);
                    fields.load("fau_page_index");

                    for mod_ in &s.mods {
                        let ld = fields.load(&mod_.name);
                        ld.set_decode(PrintAs::Id, instr);
                        ld.is_mod = true;
                    }
                }
            }
        }

        InstrPrint { instr, fields }
    }

    pub fn declare_types(ts: &mut TokenStream) {
        ts.extend(quote! {
            struct NullWriter;

            impl std::io::Write for NullWriter {
                fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                    Ok(buf.len())
                }
                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }

            #[derive(Default)]
            pub struct PrintCtx {
                pub fau32: bool,
                pub pc_base: u64,
            }

            enum PrintError {
                Fmt(std::fmt::Error),
                Io(std::io::Error),
                InvalidInstr(InvalidInstrError),
            }

            impl PrintError {
                fn into_invalid_instr(self) -> InvalidInstrError {
                    match self {
                        PrintError::InvalidInstr(e) => e,
                        _ => panic!("into_invalid_instr called on Fmt or Io"),
                    }
                }
            }

            impl From<std::convert::Infallible> for PrintError {
                fn from(_err: std::convert::Infallible) -> PrintError {
                    panic!("Infallible can't happen");
                }
            }

            impl From<std::fmt::Error> for PrintError {
                fn from(err: std::fmt::Error) -> PrintError {
                    PrintError::Fmt(err)
                }
            }

            impl From<std::io::Error> for PrintError {
                fn from(err: std::io::Error) -> PrintError {
                    PrintError::Io(err)
                }
            }

            impl From<EncodeError> for PrintError {
                fn from(err: EncodeError) -> PrintError {
                    PrintError::InvalidInstr(err.into())
                }
            }

            impl From<InvalidInstrError> for PrintError {
                fn from(err: InvalidInstrError) -> PrintError {
                    PrintError::InvalidInstr(err)
                }
            }
        });
    }

    fn table_entry(
        &self,
        fields: &mut BTreeMap<LoadField, usize>,
    ) -> TokenStream {
        let name = &self.instr.name;
        let variant = match &self.instr.variant {
            Some(v) => quote! {Some(#v)},
            None => quote! {None},
        };

        let mut fields_ts: TokenStream = Default::default();
        for load in self.fields.fields() {
            let next_idx = fields.len();
            let field_idx =
                *fields.entry(load.clone()).or_insert(next_idx) as u32;
            fields_ts.extend(quote! {#field_idx,});
        }

        quote! {
            Instr {
                name: #name,
                variant: #variant,
                fields: &[#fields_ts],
            }
        }
    }
}

#[derive(Clone)]
struct CmpTokenStream(TokenStream);

impl From<TokenStream> for CmpTokenStream {
    fn from(val: TokenStream) -> CmpTokenStream {
        CmpTokenStream(val)
    }
}

impl ToTokens for CmpTokenStream {
    fn to_tokens(&self, ts: &mut TokenStream) {
        self.0.to_tokens(ts);
    }
}

impl Ord for CmpTokenStream {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        quote! {#self}.to_string().cmp(&quote! {#other}.to_string())
    }
}

impl PartialOrd for CmpTokenStream {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CmpTokenStream {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for CmpTokenStream {}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum LoadFn {
    Physical {
        start: u8,
        end: u8,
        mod_: FieldMod,
    },
    Virtual {
        inputs: Vec<String>,
        expr: CmpTokenStream,
    },
}

impl LoadFn {
    fn ident(&self) -> Ident {
        match self {
            LoadFn::Physical { start, end, mod_ } => {
                let ident_mod = match mod_ {
                    FieldMod::Shr(n) => format!("_sh{}", n),
                    FieldMod::Minus(n) => format!("_mi{}", n),
                    FieldMod::Align(n) => format!("_al{}", n),
                    _ => "".into(),
                };
                Ident::new(
                    &format!("load_{}_{}{}", start, end, ident_mod),
                    Span::call_site(),
                )
            }
            LoadFn::Virtual { expr, .. } => {
                let mut s = std::hash::DefaultHasher::new();
                expr.0.to_string().hash(&mut s);
                let hash = s.finish();
                Ident::new(&format!("expr_{:x}", hash), Span::call_site())
            }
        }
    }

    fn ref_ts(&self) -> TokenStream {
        let id = self.ident();
        match self {
            LoadFn::Physical { .. } => {
                quote! {LoadFn::Physical(#id)}
            }
            LoadFn::Virtual { .. } => {
                quote! {LoadFn::Virtual(#id)}
            }
        }
    }
}

impl From<&InstrField> for LoadFn {
    fn from(value: &InstrField) -> Self {
        match value {
            InstrField::Physical(f) => LoadFn::Physical {
                start: f.bits.start,
                end: f.bits.end,
                mod_: f.mod_,
            },
            InstrField::Virtual(f) => {
                let mut inputs: Vec<String> = Default::default();
                for field in f.expr.fields() {
                    if inputs.contains(&field.name) {
                        continue;
                    }
                    inputs.push(field.name.clone());
                }
                LoadFn::Virtual {
                    inputs,
                    expr: f.expr.to_token_stream().into(),
                }
            }
            _ => unreachable!("Unexpected InstrField variant"),
        }
    }
}

impl ToTokens for LoadFn {
    fn to_tokens(&self, ts: &mut TokenStream) {
        match self {
            LoadFn::Physical { start, end, mod_ } => {
                let mask = bitmask(end - start);
                let mut load_ts = quote! {((v >> #start) & #mask)};

                load_ts = match mod_ {
                    FieldMod::Shr(n) => quote! {(#load_ts << #n)},
                    FieldMod::Minus(n) => quote! {(#load_ts + #n)},
                    FieldMod::Align(_) | FieldMod::None => load_ts,
                };

                // All values we unpack will fit into u32.
                assert!((end - start) + u8::from(mod_.extra_bits()) <= 32);

                let ident = self.ident();
                ts.extend(quote! {
                    fn #ident (v: u64) -> u32 {
                        #load_ts as u32
                    }
                });
            }
            LoadFn::Virtual { expr, inputs } => {
                let ident = self.ident();
                let mut load_ts = quote! {};
                for name in inputs {
                    let name_id = instr_field_ident(name);
                    load_ts.extend(quote! {let #name_id = values[#name];});
                }

                ts.extend(quote! {
                    fn #ident (values: &FxHashMap<&'static str, u32>, arch: u8)
                        -> Result<u32, EncodeError> {
                            #load_ts
                            Ok((#expr) as u32)
                    }
                });
            }
        }
    }
}

#[derive(PartialEq, Eq, Hash, Ord, PartialOrd, Clone)]
struct DecodeFn {
    ident: Ident,
    body: String,
}

impl DecodeFn {
    const DEFAULT_LOAD_TYPE: FieldType = FieldType::Uint(32);
    fn new(field: &InstrField, print_as: Option<&PrintAs>) -> Self {
        let field_type = field.field_type().unwrap_or(&Self::DEFAULT_LOAD_TYPE);

        let mut ts: TokenStream = Default::default();

        if let Some(print_as) = print_as {
            for name in print_as.additional_input_names() {
                let name_id = instr_field_ident(&name);
                ts.extend(quote! {let #name_id = values[#name];});
            }
        }

        let ident_in = Ident::new("val", Span::call_site());

        // Cast to the actual type of the field.
        let as_type = match field_type {
            FieldType::Enum(e) => {
                let ei = &e.ident;
                // All enums implement TryDecode<u8>
                quote! { #ei::try_decode((#ident_in) as u8, arch)? }
            }
            FieldType::Int(n) | FieldType::PcRelOffsetSigned(n) => {
                // n == "physical size" + mod extra bits
                // The mod has been reversed already, thus the "encoded" size
                // here is exactly n.
                quote! {sign_ext(#ident_in, #n)}
            }
            FieldType::Source
            | FieldType::Source64
            | FieldType::Uint(_)
            | FieldType::PcRelOffsetUnsigned(_) => quote! {#ident_in},
        };
        ts.extend(quote! {let #ident_in = #as_type; });

        // Check if in restricted set.
        match (field_type, field.restrict()) {
            (FieldType::Enum(e), Some(restrict)) => {
                let ei = &e.ident;
                let mut allowed: TokenStream = Default::default();

                for allowed_value in &restrict.values {
                    let ev = &allowed_value.value_ident;
                    allowed.extend(quote! {#ei::#ev,});
                }

                ts.extend(quote! {if ! [#allowed].contains(&#ident_in) {
                    return Err(InvalidInstrError::Any.into());
                }});
            }
            _ => (),
        };

        let maybe_none = match field_type {
            FieldType::Enum(enum_) => enum_.has_none,
            _ => false,
        };

        if maybe_none {
            ts.extend(quote! {if #ident_in.is_none() { return Ok(None); }});
        }

        if let Some(print_as) = print_as {
            let mut as_print_type_ts: TokenStream = Default::default();
            print_as.to_tokens(&ident_in, &mut as_print_type_ts);

            ts.extend(quote! {Ok(Some(Box::new(#as_print_type_ts)))});
        } else {
            ts.extend(quote! {Ok(None)});
        }

        let body = ts.to_string();

        let mut s = std::hash::DefaultHasher::new();
        body.hash(&mut s);
        let hash = s.finish();
        let ident = Ident::new(&format!("dec_{:x}", hash), Span::call_site());

        DecodeFn { ident, body }
    }

    fn ref_ts(&self) -> TokenStream {
        let id = &self.ident;
        quote! {#id}
    }
}

impl ToTokens for DecodeFn {
    fn to_tokens(&self, ts: &mut TokenStream) {
        let body: TokenStream = self.body.parse().unwrap();
        let ident = &self.ident;
        let ret_type = quote! {DecodeResult<dyn std::fmt::Display>};
        ts.extend(quote! {
            fn #ident (val: u32, values: &FxHashMap<&'static str, u32>,
                arch: u8, ctx: &PrintCtx) -> #ret_type {
                #body
            }
        });
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct InstrKey {
    name: String,
    variant: Option<String>,
    arch_from: u8,
    arch_to: u8,
}

impl From<&Instr> for InstrKey {
    fn from(value: &Instr) -> Self {
        InstrKey {
            name: value.name.clone(),
            variant: value.variant.clone(),
            arch_from: value.arch.start,
            arch_to: value.arch.end,
        }
    }
}

fn gen_print(
    instrs: &[Instr],
    instr_to_idx: &mut BTreeMap<InstrKey, usize>,
) -> TokenStream {
    let mut ts: TokenStream = Default::default();

    InstrPrint::declare_types(&mut ts);
    PrintAs::declare_types(&mut ts);

    let mut fields = Default::default();
    let mut instr_table_entries: TokenStream = Default::default();

    for (table_idx, instr) in instrs.iter().enumerate() {
        let instr_print = InstrPrint::new(instr);

        let entry = instr_print.table_entry(&mut fields);
        instr_table_entries.extend(quote! {#entry,});

        instr_to_idx.insert(instr.into(), table_idx);
    }

    let mut load_functions: BTreeSet<LoadFn> = Default::default();
    let mut decode_functions: BTreeSet<DecodeFn> = Default::default();

    let mut field_table_entries: TokenStream = Default::default();
    let mut fields_sorted: Vec<_> = fields.iter().collect();
    fields_sorted.sort_by_key(|(_, idx)| **idx);
    for (field, _) in fields_sorted {
        field.table_entry(&mut field_table_entries);

        load_functions.insert(field.load_fn.clone());
        decode_functions.insert(field.decode_fn.clone());
    }

    for f in load_functions {
        f.to_tokens(&mut ts);
    }

    for f in decode_functions {
        f.to_tokens(&mut ts);
    }

    let n_fields = fields.len();
    ts.extend(quote! {
        pub static FIELDS: [Field; #n_fields] = [#field_table_entries];
    });

    let n_instrs = instrs.len();
    ts.extend(quote! {
        pub static DISASM_TABLE: [Instr; #n_instrs] = [#instr_table_entries];
    });

    ts
}

struct SimpleEnum {
    name: String,
    values: Vec<String>,
}

impl SimpleEnum {
    pub fn new(name: &str, values: Vec<String>) -> SimpleEnum {
        SimpleEnum {
            name: String::from(name),
            values: values,
        }
    }

    pub fn ident(&self) -> Ident {
        Ident::new(&self.name, Span::call_site())
    }
}

impl ToTokens for SimpleEnum {
    fn to_tokens(&self, ts: &mut TokenStream) {
        let name_id = ident!("{}", &self.name);

        let mut idents: Vec<Ident> =
            self.values.iter().map(|n| ident!("{n}")).collect();
        idents.sort();

        let display_cases: Vec<TokenStream> = idents
            .iter()
            .zip(&self.values)
            .map(|(i, s)| quote! {#name_id::#i => write!(f, #s)})
            .collect();

        ts.extend(quote! {
            pub enum #name_id {
                #(#idents),*
            }

            impl std::fmt::Display for #name_id {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        #(#display_cases),*
                    }
                }
            }
        })
    }
}

fn gen_decode(
    isa: &ISA,
    instr_to_idx: &BTreeMap<InstrKey, usize>,
) -> TokenStream {
    let mut decode_per_arch: BTreeMap<_, Box<DecoderNode>> = Default::default();

    for target_arch in isa.arch.clone() {
        let instrs: Vec<&Instr> = isa
            .instrs
            .iter()
            .filter(|i| i.arch.contains(&target_arch))
            .collect();

        let decoder = DecoderNode::build(instrs);
        decode_per_arch.insert(target_arch, Box::new(decoder));
    }

    let make_ret_val = |instr: &Instr| {
        let idx = instr_to_idx.get(&instr.into()).unwrap();
        quote! { #idx }
    };

    let make_check = |instr: &Instr| {
        let idx = instr_to_idx.get(&instr.into()).unwrap();
        quote! {try_instr(#idx, input_value, arch)}
    };

    let decoder_cases_ts: TokenStream = decode_per_arch
        .iter()
        .map(|(arch, decoder)| {
            let per_arch = decoder.to_tokens(&make_ret_val, &make_check);
            quote! {
                #arch => #per_arch,
            }
        })
        .collect();

    quote! {
        pub fn try_decode(
            input_value: u64, arch: u8
        ) -> Result<usize, InvalidInstrError> {
            match arch {
                #decoder_cases_ts
                _ => Err(InvalidInstrError::Any),
            }
        }
    }
}

pub fn gen_decoder(
    xml_file: &str,
    arch: std::ops::Range<u8>,
) -> Result<TokenStream> {
    let isa = ISA::from_xml_file(std::fs::File::open(xml_file)?, arch)?;

    let instr_names: HashSet<String> =
        isa.instrs.iter().map(|i| i.name.clone()).collect();
    let instr_variants: HashSet<String> = isa
        .instrs
        .iter()
        .filter_map(|i| i.variant.clone())
        .collect();

    let instr_name_enum =
        SimpleEnum::new("Mnemonic", instr_names.into_iter().collect());
    let instr_var_enum =
        SimpleEnum::new("Variant", instr_variants.into_iter().collect());

    let mut instr_to_idx = Default::default();
    let printer_ts = gen_print(&isa.instrs, &mut instr_to_idx);

    let decode_ts = gen_decode(&isa, &instr_to_idx);

    Ok(quote! {
        fn sign_ext(v: u32, enc_width: u8) -> i32 {
            let r = 32 - enc_width;
            ((v << r) as i32) >> r
        }

        #instr_name_enum

        #instr_var_enum

        #decode_ts

        #printer_ts
    })
}
