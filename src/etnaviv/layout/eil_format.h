/*
 * Copyright © 2026 Daniel Lang
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include "common/etna_core_info.h"
#include "util/format/u_formats.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ENUM_PACKED eil_emulation {
   EIL_EMU_NONE,
   EIL_EMU_DEPTH32F,
   EIL_EMU_128BIT,
};

static_assert(sizeof(enum eil_emulation) == sizeof(uint8_t), "enum eil_emulation must be packed");

struct eil_format {
   enum pipe_format emulated; /* enum pipe_format the HW really sees */
   enum eil_emulation emulation;
   uint8_t vtx; /* FE_DATA_TYPE_* */
   enum etna_feature vtx_gate;
   uint8_t tex; /* TEXTURE_FORMAT_* */
   enum etna_feature tex_gate;
   uint8_t pe; /* PE_FORMAT_* */
   enum etna_feature pe_gate;

   bool has_vtx    : 1;
   bool has_tex    : 1;
   bool has_pe     : 1;
   bool vtx_gated  : 1;
   bool tex_ext    : 1; /* TEXTURE_FORMAT_EXT_* space */
   bool tex_astc   : 1; /* TEXTURE_FORMAT_ASTC_* space */
   bool tex_gated  : 1;
   bool pe_rb_swap : 1;
   bool pe_gated   : 1;
};

extern const struct eil_format eil_format_table[PIPE_FORMAT_COUNT];

static inline const struct eil_format *
eil_get_format_entry(enum pipe_format fmt)
{
   assert(fmt >= 0 && fmt < PIPE_FORMAT_COUNT);
   return &eil_format_table[fmt];
}

static inline bool
eil_format_is_emulated_z32f(enum pipe_format fmt)
{
   const struct eil_format *f = eil_get_format_entry(fmt);
   return f->emulation == EIL_EMU_DEPTH32F;
}

static inline bool
eil_format_is_emulated_128bit(enum pipe_format fmt)
{
   const struct eil_format *f = eil_get_format_entry(fmt);
   return f->emulation == EIL_EMU_128BIT;
}

/*
 * If a format can't be handled by hardware directly,
 * the pipe_format used in emulation is returned.
 * Otherwise fmt is returned unmodified.
 */
static inline enum pipe_format
eil_format_emulated(enum pipe_format fmt)
{
   const struct eil_format *f = eil_get_format_entry(fmt);
   return f->emulation != EIL_EMU_NONE ? f->emulated : fmt;
}

static inline bool
eil_gate_ok(const struct etna_core_info *info, bool gated, enum etna_feature gate)
{
   return !gated || etna_core_has_feature(info, gate);
}

static inline bool
eil_format_supports_sampling(const struct etna_core_info *info, enum pipe_format fmt)
{
   const struct eil_format *f = eil_get_format_entry(fmt);
   return f->has_tex && eil_gate_ok(info, f->tex_gated, f->tex_gate);
}

static inline bool
eil_format_supports_vertex_fetch(const struct etna_core_info *info, enum pipe_format fmt)
{
   const struct eil_format *f = eil_get_format_entry(fmt);
   return f->has_vtx && eil_gate_ok(info, f->vtx_gated, f->vtx_gate);
}

static inline bool
eil_format_supports_rendering(const struct etna_core_info *info, enum pipe_format fmt)
{
   const struct eil_format *f = eil_get_format_entry(fmt);
   return f->has_pe && eil_gate_ok(info, f->pe_gated, f->pe_gate);
}

#ifdef __cplusplus
}
#endif
