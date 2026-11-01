/*
 * Copyright © 2026 Rusi Genev
 * SPDX-License-Identifier: MIT
 */

#include "v3d_compiler.h"
#include "compiler/nir/nir_builder.h"

/** @file v3d_nir_lower_txf_buffer_rgb32.c
 *
 * The TMU has no 96-bit texel format, so RGB32 buffer textures are bound as
 * a 32-bit single-channel view of the same memory with three times as many
 * texels.  This pass lowers a txf on such a sampler to three scalar txfs at
 * 3 * i + {0, 1, 2} and builds the vec4 result, with alpha set to one.
 *
 * Every fetch reads channel 0 of its own result, so the view swizzle does
 * not matter as long as it leaves the data in X, which the R32 format
 * swizzle does.
 */

static bool
lower_txf_buffer_rgb32_filter(const nir_instr *instr, const void *data)
{
        if (instr->type != nir_instr_type_tex)
                return false;

        const nir_tex_instr *tex = nir_instr_as_tex(instr);
        const uint32_t *sampler_mask = data;

        if (tex->op != nir_texop_txf ||
            tex->sampler_dim != GLSL_SAMPLER_DIM_BUF)
                return false;

        assert(tex->texture_index < 32);
        return *sampler_mask & (1u << tex->texture_index);
}

static nir_def *
lower_txf_buffer_rgb32_instr(nir_builder *b, nir_instr *in_instr, void *data)
{
        nir_tex_instr *instr = nir_instr_as_tex(in_instr);
        b->cursor = nir_before_instr(&instr->instr);

        int coord_idx = nir_tex_instr_src_index(instr, nir_tex_src_coord);
        assert(coord_idx >= 0);

        nir_def *base = nir_imul_imm(b, instr->src[coord_idx].src.ssa, 3);

        nir_def *comps[4];
        for (unsigned i = 0; i < 3; i++) {
                /* Build the coordinate before the fetch that uses it. */
                nir_def *coord = nir_iadd_imm(b, base, i);

                nir_tex_instr *fetch =
                        nir_instr_as_tex(nir_instr_clone(b->shader,
                                                         &instr->instr));
                nir_builder_instr_insert(b, &fetch->instr);
                nir_src_rewrite(&fetch->src[coord_idx].src, coord);

                comps[i] = nir_channel(b, &fetch->def, 0);
        }

        bool is_float =
                nir_alu_type_get_base_type(instr->dest_type) == nir_type_float;
        comps[3] = is_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1);

        return nir_trim_vector(b, nir_vec(b, comps, 4),
                               instr->def.num_components);
}

bool
v3d_nir_lower_txf_buffer_rgb32(nir_shader *s, uint32_t sampler_mask)
{
        if (!sampler_mask)
                return false;

        return nir_shader_lower_instructions(s,
                                             lower_txf_buffer_rgb32_filter,
                                             lower_txf_buffer_rgb32_instr,
                                             &sampler_mask);
}
