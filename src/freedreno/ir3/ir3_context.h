/*
 * Copyright © 2015-2018 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#ifndef IR3_CONTEXT_H_
#define IR3_CONTEXT_H_

#include "ir3.h"
#include "ir3_compiler.h"
#include "ir3_nir.h"

/* for conditionally setting boolean flag(s): */
#define COND(bool, val) ((bool) ? (val) : 0)

#define DBG(fmt, ...)                                                          \
   do {                                                                        \
      mesa_logd("%s:%d: " fmt, __func__, __LINE__, ##__VA_ARGS__);             \
   } while (0)

/**
 * The context for compilation of a single shader.
 */
struct ir3_context {
   struct ir3_compiler *compiler;
   const struct ir3_context_funcs *funcs;

   struct nir_shader *s;

   struct nir_instr *cur_instr; /* current instruction, just for debug */

   struct ir3 *ir;
   struct ir3_shader_variant *so;

   /* Tables of scalar inputs/outputs.  Because of the way varying packing
    * works, we could have inputs w/ fractional location, which is a bit
    * awkward to deal with unless we keep track of the split scalar in/
    * out components.
    *
    * These *only* have inputs/outputs that are touched by load_*input and
    * store_output.
    */
   unsigned ninputs, noutputs;
   struct ir3_instruction **inputs;
   struct ir3_instruction **outputs;

   struct ir3_block *block;    /* the current block */
   struct ir3_builder build;
   struct ir3_block *in_block; /* block created for shader inputs */

   nir_function_impl *impl;

   /* For fragment shaders, varyings are not actual shader inputs,
    * instead the hw passes a ij coord which is used with
    * bary.f.
    *
    * But NIR doesn't know that, it still declares varyings as
    * inputs.  So we do all the input tracking normally and fix
    * things up after compile_instructions()
    */
   struct ir3_instruction *ij[IJ_COUNT];

   /* for fragment shaders, for gl_FrontFacing and gl_FragCoord: */
   struct ir3_instruction *frag_face, *frag_coord;

   /* For vertex shaders, keep track of the system values sources */
   struct ir3_instruction *vertex_id, *basevertex, *instance_id, *base_instance,
      *draw_id, *view_index, *is_indexed_draw;

   /* For fragment shaders: */
   struct ir3_instruction *samp_id, *samp_mask_in;

   /* For geometry shaders: */
   struct ir3_instruction *primitive_id;
   struct ir3_instruction *gs_header;

   /* For tessellation shaders: */
   struct ir3_instruction *tcs_header;
   struct ir3_instruction *tess_coord;
   struct ir3_instruction *rel_patch_id;

   /* Compute shader inputs: */
   struct ir3_instruction *local_invocation_id, *work_group_id;

   struct ir3_instruction *frag_shading_rate;

   /* mapping from nir_register to defining instruction: */
   struct hash_table *def_ht;

   unsigned num_arrays;

   unsigned loop_depth;

   /* a common pattern for indirect addressing is to request the
    * same address register multiple times.  To avoid generating
    * duplicate instruction sequences (which our backend does not
    * try to clean up, since that should be done as the NIR stage)
    * we cache the address value generated for a given src value:
    *
    * Note that we have to cache these per alignment, since same
    * src used for an array of vec1 cannot be also used for an
    * array of vec4.
    */
   struct hash_table *addr0_ht[4];

   struct hash_table *sel_cond_conversions;
   struct hash_table *predicate_conversions;

   /* last dst array, for indirect we need to insert a var-store.
    */
   struct ir3_instruction **last_dst;
   unsigned last_dst_n;

   /* maps nir_block to ir3_block, mostly for the purposes of
    * figuring out the blocks successors
    */
   struct hash_table *block_ht;

   /* maps nir_block at the top of a loop to ir3_block collecting continue
    * edges.
    */
   struct hash_table *continue_block_ht;

   /* on a4xx, bitmask of samplers which need astc+srgb workaround: */
   unsigned astc_srgb;

   /* on a4xx, per-sampler per-component swizzles, for tg4: */
   uint16_t sampler_swizzles[16];

   unsigned samples; /* bitmask of x,y sample shifts */

   unsigned max_texture_index;

   unsigned prefetch_limit;

   bool has_relative_load_const_ir3;

   /* set if we encounter something we can't handle yet, so we
    * can bail cleanly and fallback to TGSI compiler f/e
    */
   bool error;
};

struct ir3_context_funcs {
   void (*emit_intrinsic_load_ssbo)(struct ir3_context *ctx,
                                    nir_intrinsic_instr *intr,
                                    struct ir3_instruction **dst);
   void (*emit_intrinsic_load_uav)(struct ir3_context *ctx,
                                   nir_intrinsic_instr *intr,
                                   struct ir3_instruction **dst);
   void (*emit_intrinsic_store_ssbo)(struct ir3_context *ctx,
                                     nir_intrinsic_instr *intr);
   struct ir3_instruction *(*emit_intrinsic_atomic_ssbo)(
      struct ir3_context *ctx, nir_intrinsic_instr *intr);
   void (*emit_intrinsic_load_image)(struct ir3_context *ctx,
                                     nir_intrinsic_instr *intr,
                                     struct ir3_instruction **dst);
   void (*emit_intrinsic_store_image)(struct ir3_context *ctx,
                                      nir_intrinsic_instr *intr);
   struct ir3_instruction *(*emit_intrinsic_atomic_image)(
      struct ir3_context *ctx, nir_intrinsic_instr *intr);
   void (*emit_intrinsic_image_size)(struct ir3_context *ctx,
                                     nir_intrinsic_instr *intr,
                                     struct ir3_instruction **dst);
   void (*emit_intrinsic_load_global_ir3)(struct ir3_context *ctx,
                                          nir_intrinsic_instr *intr,
                                          struct ir3_instruction **dst);
   void (*emit_intrinsic_store_global_ir3)(struct ir3_context *ctx,
                                           nir_intrinsic_instr *intr);
   struct ir3_instruction *(*emit_intrinsic_atomic_global)(
      struct ir3_context *ctx, nir_intrinsic_instr *intr);
};

extern const struct ir3_context_funcs ir3_a4xx_funcs;
extern const struct ir3_context_funcs ir3_a6xx_funcs;

struct ir3_context *ir3_context_init(struct ir3_compiler *compiler,
                                     struct ir3_shader *shader,
                                     struct ir3_shader_variant *so);
void ir3_context_free(struct ir3_context *ctx);

static inline void
ir3_context_set_block(struct ir3_context *ctx, struct ir3_block *block)
{
   ctx->block = block;
   ctx->build = ir3_builder_at(ir3_before_terminator(block));
}

struct ir3_instruction **ir3_get_dst_ssa(struct ir3_context *ctx,
                                         nir_def *dst, unsigned n);
struct ir3_instruction **ir3_get_def(struct ir3_context *ctx, nir_def *def,
                                     unsigned n);
struct ir3_instruction *const *ir3_get_src_maybe_shared(struct ir3_context *ctx,
                                                        nir_src *src);
struct ir3_instruction *const *ir3_get_src_shared(struct ir3_context *ctx,
                                                  nir_src *src, bool shared);

static inline struct ir3_instruction *const *
ir3_get_src(struct ir3_context *ctx, nir_src *src)
{
   return ir3_get_src_shared(ctx, src, false);
}

void ir3_put_def(struct ir3_context *ctx, nir_def *def);
void ir3_handle_bindless_cat6(struct ir3_instruction *instr, nir_src rsrc);
void ir3_handle_nonuniform(struct ir3_instruction *instr,
                           nir_intrinsic_instr *intrin);
void emit_intrinsic_image_size_tex(struct ir3_context *ctx,
                                   nir_intrinsic_instr *intr,
                                   struct ir3_instruction **dst);

NORETURN void ir3_context_error(struct ir3_context *ctx, const char *format,
                                ...);

#define compile_assert(ctx, cond)                                              \
   do {                                                                        \
      if (!(cond))                                                             \
         ir3_context_error((ctx), "failed assert: " #cond "\n");               \
   } while (0)

struct ir3_instruction *ir3_get_addr0(struct ir3_context *ctx,
                                      struct ir3_instruction *src, int align);
struct ir3_instruction *ir3_get_predicate(struct ir3_context *ctx,
                                          struct ir3_instruction *src);

void ir3_declare_array(struct ir3_context *ctx, nir_intrinsic_instr *decl);
struct ir3_array *ir3_get_array(struct ir3_context *ctx, nir_def *reg);
struct ir3_instruction *ir3_create_array_load(struct ir3_context *ctx,
                                              struct ir3_array *arr, int n,
                                              struct ir3_instruction *address);
void ir3_create_array_store(struct ir3_context *ctx, struct ir3_array *arr,
                            int n, struct ir3_instruction *src,
                            struct ir3_instruction *address);
void ir3_lower_imm_offset(struct ir3_context *ctx, nir_intrinsic_instr *intr,
                          nir_src *offset_src, unsigned imm_offset_bits,
                          struct ir3_instruction **offset,
                          unsigned *imm_offset);

static inline type_t
utype_for_size(unsigned bit_size)
{
   switch (bit_size) {
   case 32:
      return TYPE_U32;
   case 16:
      return TYPE_U16;
   case 8:
      return TYPE_U8;
   default:
      UNREACHABLE("bad bitsize");
      return ~0;
   }
}

static inline type_t
utype_src(nir_src src)
{
   return utype_for_size(nir_src_bit_size(src));
}

static inline type_t
utype_def(nir_def *def)
{
   return utype_for_size(def->bit_size);
}

/**
 * Convert nir bitsize to ir3 bitsize, handling the special case of 1b bools
 * which can be 16b or 32b depending on gen.
 */
static inline unsigned
ir3_bitsize(struct ir3_context *ctx, unsigned nir_bitsize)
{
   if (nir_bitsize == 1)
      return type_size(ctx->compiler->bool_type);
   return nir_bitsize;
}

#endif /* IR3_CONTEXT_H_ */
