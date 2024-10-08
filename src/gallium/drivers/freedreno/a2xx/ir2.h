/*
 * Copyright © 2018 Jonathan Marek <jonathan@marek.ca>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Jonathan Marek <jonathan@marek.ca>
 */

#ifndef IR2_H_
#define IR2_H_

#include "compiler/nir/nir.h"
#include "pipe/p_context.h"

struct ir2_fetch_info {
   /* dword offset of the fetch instruction */
   uint16_t offset;
   union {
      /* swizzle to merge with tgsi swizzle */
      struct {
         uint16_t dst_swiz;
      } vtx;
      /* sampler id to patch const_idx */
      struct {
         uint16_t samp_id;
         uint8_t src_swiz;
      } tex;
   };
};

struct ir2_shader_info {
   /* compiler shader */
   uint32_t *dwords;

   /* size of the compiled shader in dwords */
   uint16_t sizedwords;

   /* highest GPR # used by shader */
   int8_t max_reg;

   /* offset in dwords of first MEMORY export CF (for a20x hw binning) */
   int16_t mem_export_ptr;

   /* fetch instruction info for patching */
   uint16_t num_fetch_instrs;
   struct ir2_fetch_info fetch_info[64];
};

struct ir2_frag_linkage {
   unsigned inputs_count;
   struct {
      uint8_t slot;
      uint8_t ncomp;
   } inputs[16];

   /* driver_location of fragcoord.zw, -1 if not used */
   int fragcoord;
};

struct ir2_shader_variant {
   struct ir2_shader_info info;
   struct ir2_frag_linkage f;
};

struct fd2_shader_stateobj;
struct tgsi_token;

void ir2_compile(struct fd2_shader_stateobj *so, unsigned variant,
                 struct fd2_shader_stateobj *fp);

struct nir_shader *ir2_tgsi_to_nir(const struct tgsi_token *tokens,
                                   struct pipe_screen *screen);

const nir_shader_compiler_options *ir2_get_compiler_options(void);

int ir2_optimize_nir(nir_shader *s, bool lower);

#endif /* IR2_H_ */
