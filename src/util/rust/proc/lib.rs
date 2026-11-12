// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

extern crate proc_macro;
extern crate proc_macro2;
#[macro_use]
extern crate quote;
extern crate syn;

use proc_macro::TokenStream;

mod enum_as_u8;

#[proc_macro_derive(EnumAsU8)]
pub fn derive_enum_as_u8(input: TokenStream) -> TokenStream {
    enum_as_u8::derive_enum_as_u8(input)
}
