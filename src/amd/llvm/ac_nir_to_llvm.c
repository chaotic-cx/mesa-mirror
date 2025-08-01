/*
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * SPDX-License-Identifier: MIT
 */

#include "ac_nir_to_llvm.h"
#include "ac_gpu_info.h"
#include "ac_binary.h"
#include "ac_llvm_build.h"
#include "ac_llvm_util.h"
#include "ac_shader_abi.h"
#include "ac_shader_util.h"
#include "ac_nir.h"
#include "nir/nir.h"
#include "nir/nir_deref.h"
#include "sid.h"
#include "util/bitscan.h"
#include "util/u_math.h"
#include <llvm/Config/llvm-config.h>

struct ac_nir_context {
   struct ac_llvm_context ac;
   struct ac_shader_abi *abi;
   const struct ac_shader_args *args;

   gl_shader_stage stage;
   shader_info *info;

   LLVMValueRef *ssa_defs;

   struct ac_llvm_pointer scratch;
   struct ac_llvm_pointer constant_data;
   struct ac_llvm_pointer lds;

   struct hash_table *defs;
   struct hash_table *phis;
   struct hash_table *verified_interp;

   LLVMValueRef main_function;
   LLVMBasicBlockRef continue_block;
   LLVMBasicBlockRef break_block;
};

static LLVMTypeRef get_def_type(struct ac_nir_context *ctx, const nir_def *def)
{
   LLVMTypeRef type = LLVMIntTypeInContext(ctx->ac.context, def->bit_size);
   if (def->num_components > 1) {
      type = LLVMVectorType(type, def->num_components);
   }
   return type;
}

static LLVMValueRef get_src(struct ac_nir_context *nir, nir_src src)
{
   return nir->ssa_defs[src.ssa->index];
}

static LLVMValueRef get_shared_mem_ptr(struct ac_nir_context *ctx, nir_src src, unsigned c_off)
{
   if (!ctx->lds.value) {
      unsigned lds_size = ctx->ac.gfx_level >= GFX7 ? 65536 : 32768;
      LLVMTypeRef type = LLVMArrayType(ctx->ac.i32, lds_size / 4);
      ctx->lds = (struct ac_llvm_pointer) {
         .value = LLVMBuildIntToPtr(ctx->ac.builder, ctx->ac.i32_0,
                     LLVMPointerTypeInContext(ctx->ac.context, AC_ADDR_SPACE_LDS), "lds"),
         .pointee_type = type,
      };
   }

   LLVMValueRef ptr = get_src(ctx, src);
   ptr = LLVMBuildAdd(ctx->ac.builder, ptr, LLVMConstInt(ctx->ac.i32, c_off, 0), "");
   /* LDS is used here as a i8 pointer. */
   return LLVMBuildGEP2(ctx->ac.builder, ctx->ac.i8, ctx->lds.value, &ptr, 1, "");
}

static LLVMBasicBlockRef get_block(struct ac_nir_context *nir, const struct nir_block *b)
{
   struct hash_entry *entry = _mesa_hash_table_search(nir->defs, b);
   return (LLVMBasicBlockRef)entry->data;
}

static LLVMValueRef get_alu_src(struct ac_nir_context *ctx, nir_alu_src src,
                                unsigned num_components)
{
   LLVMValueRef value = get_src(ctx, src.src);
   bool need_swizzle = false;

   assert(value);
   unsigned src_components = ac_get_llvm_num_components(value);
   for (unsigned i = 0; i < num_components; ++i) {
      assert(src.swizzle[i] < src_components);
      if (src.swizzle[i] != i)
         need_swizzle = true;
   }

   if (need_swizzle || num_components != src_components) {
      LLVMValueRef masks[] = {LLVMConstInt(ctx->ac.i32, src.swizzle[0], false),
                              LLVMConstInt(ctx->ac.i32, src.swizzle[1], false),
                              LLVMConstInt(ctx->ac.i32, src.swizzle[2], false),
                              LLVMConstInt(ctx->ac.i32, src.swizzle[3], false)};

      if (src_components > 1 && num_components == 1) {
         value = LLVMBuildExtractElement(ctx->ac.builder, value, masks[0], "");
      } else if (src_components == 1 && num_components > 1) {
         LLVMValueRef values[] = {value, value, value, value};
         value = ac_build_gather_values(&ctx->ac, values, num_components);
      } else {
         LLVMValueRef swizzle = LLVMConstVector(masks, num_components);
         value = LLVMBuildShuffleVector(ctx->ac.builder, value, value, swizzle, "");
      }
   }
   return value;
}

static LLVMValueRef emit_int_cmp(struct ac_llvm_context *ctx, LLVMIntPredicate pred,
                                 LLVMValueRef src0, LLVMValueRef src1)
{
   src0 = ac_to_integer(ctx, src0);
   src1 = ac_to_integer(ctx, src1);
   return LLVMBuildICmp(ctx->builder, pred, src0, src1, "");
}

static LLVMValueRef emit_float_cmp(struct ac_llvm_context *ctx, LLVMRealPredicate pred,
                                   LLVMValueRef src0, LLVMValueRef src1)
{
   src0 = ac_to_float(ctx, src0);
   src1 = ac_to_float(ctx, src1);
   return LLVMBuildFCmp(ctx->builder, pred, src0, src1, "");
}

static LLVMValueRef emit_intrin_1f_param(struct ac_llvm_context *ctx, const char *intrin,
                                         LLVMTypeRef result_type, LLVMValueRef src0)
{
   char name[64], type[64];
   LLVMValueRef params[] = {
      ac_to_float(ctx, src0),
   };

   ac_build_type_name_for_intr(LLVMTypeOf(params[0]), type, sizeof(type));
   ASSERTED const int length = snprintf(name, sizeof(name), "%s.%s", intrin, type);
   assert(length < sizeof(name));
   return ac_build_intrinsic(ctx, name, result_type, params, 1, 0);
}

static LLVMValueRef emit_intrin_1f_param_scalar(struct ac_llvm_context *ctx, const char *intrin,
                                                LLVMTypeRef result_type, LLVMValueRef src0)
{
   if (LLVMGetTypeKind(result_type) != LLVMVectorTypeKind)
      return emit_intrin_1f_param(ctx, intrin, result_type, src0);

   LLVMTypeRef elem_type = LLVMGetElementType(result_type);
   LLVMValueRef ret = LLVMGetUndef(result_type);

   /* Scalarize the intrinsic, because vectors are not supported. */
   for (unsigned i = 0; i < LLVMGetVectorSize(result_type); i++) {
      char name[64], type[64];
      LLVMValueRef params[] = {
         ac_to_float(ctx, ac_llvm_extract_elem(ctx, src0, i)),
      };

      ac_build_type_name_for_intr(LLVMTypeOf(params[0]), type, sizeof(type));
      ASSERTED const int length = snprintf(name, sizeof(name), "%s.%s", intrin, type);
      assert(length < sizeof(name));
      ret = LLVMBuildInsertElement(
         ctx->builder, ret,
         ac_build_intrinsic(ctx, name, elem_type, params, 1, 0),
         LLVMConstInt(ctx->i32, i, 0), "");
   }
   return ret;
}

static LLVMValueRef emit_intrin_2f_param(struct ac_llvm_context *ctx, const char *intrin,
                                         LLVMTypeRef result_type, LLVMValueRef src0,
                                         LLVMValueRef src1)
{
   char name[64], type[64];
   LLVMValueRef params[] = {
      ac_to_float(ctx, src0),
      ac_to_float(ctx, src1),
   };

   ac_build_type_name_for_intr(LLVMTypeOf(params[0]), type, sizeof(type));
   ASSERTED const int length = snprintf(name, sizeof(name), "%s.%s", intrin, type);
   assert(length < sizeof(name));
   return ac_build_intrinsic(ctx, name, result_type, params, 2, 0);
}

static LLVMValueRef emit_intrin_3f_param(struct ac_llvm_context *ctx, const char *intrin,
                                         LLVMTypeRef result_type, LLVMValueRef src0,
                                         LLVMValueRef src1, LLVMValueRef src2)
{
   char name[64], type[64];
   LLVMValueRef params[] = {
      ac_to_float(ctx, src0),
      ac_to_float(ctx, src1),
      ac_to_float(ctx, src2),
   };

   ac_build_type_name_for_intr(LLVMTypeOf(params[0]), type, sizeof(type));
   ASSERTED const int length = snprintf(name, sizeof(name), "%s.%s", intrin, type);
   assert(length < sizeof(name));
   return ac_build_intrinsic(ctx, name, result_type, params, 3, 0);
}

static LLVMValueRef emit_bcsel(struct ac_llvm_context *ctx, LLVMValueRef src0, LLVMValueRef src1,
                               LLVMValueRef src2)
{
   LLVMTypeRef src1_type = LLVMTypeOf(src1);
   LLVMTypeRef src2_type = LLVMTypeOf(src2);

   if (LLVMGetTypeKind(src1_type) == LLVMPointerTypeKind &&
       LLVMGetTypeKind(src2_type) != LLVMPointerTypeKind) {
      src2 = LLVMBuildIntToPtr(ctx->builder, src2, src1_type, "");
   } else if (LLVMGetTypeKind(src2_type) == LLVMPointerTypeKind &&
              LLVMGetTypeKind(src1_type) != LLVMPointerTypeKind) {
      src1 = LLVMBuildIntToPtr(ctx->builder, src1, src2_type, "");
   }

   return LLVMBuildSelect(ctx->builder, src0, ac_to_integer_or_pointer(ctx, src1),
                          ac_to_integer_or_pointer(ctx, src2), "");
}

static LLVMValueRef emit_iabs(struct ac_llvm_context *ctx, LLVMValueRef src0)
{
   return ac_build_imax(ctx, src0, LLVMBuildNeg(ctx->builder, src0, ""));
}

static LLVMValueRef emit_uint_carry(struct ac_llvm_context *ctx, const char *intrin,
                                    LLVMValueRef src0, LLVMValueRef src1)
{
   LLVMTypeRef ret_type;
   LLVMTypeRef types[] = {ctx->i32, ctx->i1};
   LLVMValueRef res;
   LLVMValueRef params[] = {src0, src1};
   ret_type = LLVMStructTypeInContext(ctx->context, types, 2, false);

   res = ac_build_intrinsic(ctx, intrin, ret_type, params, 2, 0);

   res = LLVMBuildExtractValue(ctx->builder, res, 1, "");
   res = LLVMBuildZExt(ctx->builder, res, ctx->i32, "");
   return res;
}

static LLVMValueRef emit_b2f(struct ac_llvm_context *ctx, LLVMValueRef src0, unsigned bitsize)
{
   assert(ac_get_elem_bits(ctx, LLVMTypeOf(src0)) == 1);

   switch (bitsize) {
   case 16:
      if (LLVMGetTypeKind(LLVMTypeOf(src0)) == LLVMVectorTypeKind) {
         assert(LLVMGetVectorSize(LLVMTypeOf(src0)) == 2);
         LLVMValueRef f[] = {
            LLVMBuildSelect(ctx->builder, ac_llvm_extract_elem(ctx, src0, 0),
                            ctx->f16_1, ctx->f16_0, ""),
            LLVMBuildSelect(ctx->builder, ac_llvm_extract_elem(ctx, src0, 1),
                            ctx->f16_1, ctx->f16_0, ""),
         };
         return ac_build_gather_values(ctx, f, 2);
      }
      return LLVMBuildSelect(ctx->builder, src0, ctx->f16_1, ctx->f16_0, "");
   case 32:
      return LLVMBuildSelect(ctx->builder, src0, ctx->f32_1, ctx->f32_0, "");
   case 64:
      return LLVMBuildSelect(ctx->builder, src0, ctx->f64_1, ctx->f64_0, "");
   default:
      UNREACHABLE("Unsupported bit size.");
   }
}

static LLVMValueRef emit_b2i(struct ac_llvm_context *ctx, LLVMValueRef src0, unsigned bitsize)
{
   switch (bitsize) {
   case 8:
      return LLVMBuildSelect(ctx->builder, src0, ctx->i8_1, ctx->i8_0, "");
   case 16:
      if (LLVMGetTypeKind(LLVMTypeOf(src0)) == LLVMVectorTypeKind) {
         assert(LLVMGetVectorSize(LLVMTypeOf(src0)) == 2);
         LLVMValueRef i[] = {
            LLVMBuildSelect(ctx->builder, ac_llvm_extract_elem(ctx, src0, 0),
                            ctx->i16_1, ctx->i16_0, ""),
            LLVMBuildSelect(ctx->builder, ac_llvm_extract_elem(ctx, src0, 1),
                            ctx->i16_1, ctx->i16_0, ""),
         };
         return ac_build_gather_values(ctx, i, 2);
      }
      return LLVMBuildSelect(ctx->builder, src0, ctx->i16_1, ctx->i16_0, "");
   case 32:
      return LLVMBuildSelect(ctx->builder, src0, ctx->i32_1, ctx->i32_0, "");
   case 64:
      return LLVMBuildSelect(ctx->builder, src0, ctx->i64_1, ctx->i64_0, "");
   default:
      UNREACHABLE("Unsupported bit size.");
   }
}

static LLVMValueRef emit_i2b(struct ac_llvm_context *ctx, LLVMValueRef src0)
{
   LLVMValueRef zero = LLVMConstNull(LLVMTypeOf(src0));
   return LLVMBuildICmp(ctx->builder, LLVMIntNE, src0, zero, "");
}

static LLVMValueRef emit_f2f16(struct ac_llvm_context *ctx, LLVMValueRef src0)
{
   LLVMValueRef result;
   LLVMValueRef cond = NULL;

   src0 = ac_to_float(ctx, src0);
   result = LLVMBuildFPTrunc(ctx->builder, src0, ctx->f16, "");

   if (ctx->gfx_level >= GFX8) {
      LLVMValueRef args[2];
      /* Check if the result is a denormal - and flush to 0 if so. */
      args[0] = result;
      args[1] = LLVMConstInt(ctx->i32, N_SUBNORMAL | P_SUBNORMAL, false);
      cond =
         ac_build_intrinsic(ctx, "llvm.amdgcn.class.f16", ctx->i1, args, 2, 0);
   }

   /* need to convert back up to f32 */
   result = LLVMBuildFPExt(ctx->builder, result, ctx->f32, "");

   if (ctx->gfx_level >= GFX8)
      result = LLVMBuildSelect(ctx->builder, cond, ctx->f32_0, result, "");
   else {
      /* for GFX6-GFX7 */
      /* 0x38800000 is smallest half float value (2^-14) in 32-bit float,
       * so compare the result and flush to 0 if it's smaller.
       */
      LLVMValueRef temp, cond2;
      temp = emit_intrin_1f_param(ctx, "llvm.fabs", ctx->f32, result);
      cond = LLVMBuildFCmp(
         ctx->builder, LLVMRealOGT,
         LLVMBuildBitCast(ctx->builder, LLVMConstInt(ctx->i32, 0x38800000, false), ctx->f32, ""),
         temp, "");
      cond2 = LLVMBuildFCmp(ctx->builder, LLVMRealONE, temp, ctx->f32_0, "");
      cond = LLVMBuildAnd(ctx->builder, cond, cond2, "");
      result = LLVMBuildSelect(ctx->builder, cond, ctx->f32_0, result, "");
   }
   return result;
}

static LLVMValueRef emit_umul_high(struct ac_llvm_context *ctx, LLVMValueRef src0,
                                   LLVMValueRef src1)
{
   LLVMValueRef dst64, result;

/* 64-bit multiplication by a constant is broken in old LLVM. Fixed in LLVM 19.1 and LLVM 20. */
#if LLVM_VERSION_MAJOR < 19 || (LLVM_VERSION_MAJOR == 19 && LLVM_VERSION_MINOR == 0)
   if (LLVMIsConstant(src0))
      ac_build_optimization_barrier(ctx, &src1, false);
   else
      ac_build_optimization_barrier(ctx, &src0, false);
#endif

   src0 = LLVMBuildZExt(ctx->builder, src0, ctx->i64, "");
   src1 = LLVMBuildZExt(ctx->builder, src1, ctx->i64, "");

   dst64 = LLVMBuildMul(ctx->builder, src0, src1, "");
   dst64 = LLVMBuildLShr(ctx->builder, dst64, LLVMConstInt(ctx->i64, 32, false), "");
   result = LLVMBuildTrunc(ctx->builder, dst64, ctx->i32, "");
   return result;
}

static LLVMValueRef emit_imul_high(struct ac_llvm_context *ctx, LLVMValueRef src0,
                                   LLVMValueRef src1)
{
   LLVMValueRef dst64, result;
   src0 = LLVMBuildSExt(ctx->builder, src0, ctx->i64, "");
   src1 = LLVMBuildSExt(ctx->builder, src1, ctx->i64, "");

   dst64 = LLVMBuildMul(ctx->builder, src0, src1, "");
   dst64 = LLVMBuildAShr(ctx->builder, dst64, LLVMConstInt(ctx->i64, 32, false), "");
   result = LLVMBuildTrunc(ctx->builder, dst64, ctx->i32, "");
   return result;
}

static LLVMValueRef emit_bfm(struct ac_llvm_context *ctx, LLVMValueRef bits, LLVMValueRef offset)
{
   /* mask = ((1 << bits) - 1) << offset */
   return LLVMBuildShl(
      ctx->builder,
      LLVMBuildSub(ctx->builder, LLVMBuildShl(ctx->builder, ctx->i32_1, bits, ""), ctx->i32_1, ""),
      offset, "");
}

static LLVMValueRef emit_bitfield_select(struct ac_llvm_context *ctx, LLVMValueRef mask,
                                         LLVMValueRef insert, LLVMValueRef base)
{
   /* Calculate:
    *   (mask & insert) | (~mask & base) = base ^ (mask & (insert ^ base))
    * Use the right-hand side, which the LLVM backend can convert to V_BFI.
    */
   return LLVMBuildXor(
      ctx->builder, base,
      LLVMBuildAnd(ctx->builder, mask, LLVMBuildXor(ctx->builder, insert, base, ""), ""), "");
}

static LLVMValueRef emit_pack_2x16(struct ac_llvm_context *ctx, LLVMValueRef src0,
                                   LLVMValueRef (*pack)(struct ac_llvm_context *ctx,
                                                        LLVMValueRef args[2]))
{
   LLVMValueRef comp[2];

   src0 = ac_to_float(ctx, src0);
   comp[0] = LLVMBuildExtractElement(ctx->builder, src0, ctx->i32_0, "");
   comp[1] = LLVMBuildExtractElement(ctx->builder, src0, ctx->i32_1, "");

   return LLVMBuildBitCast(ctx->builder, pack(ctx, comp), ctx->i32, "");
}

static LLVMValueRef emit_unpack_half_2x16(struct ac_llvm_context *ctx, LLVMValueRef src0)
{
   LLVMValueRef const16 = LLVMConstInt(ctx->i32, 16, false);
   LLVMValueRef temps[2], val;
   int i;

   for (i = 0; i < 2; i++) {
      val = i == 1 ? LLVMBuildLShr(ctx->builder, src0, const16, "") : src0;
      val = LLVMBuildTrunc(ctx->builder, val, ctx->i16, "");
      val = LLVMBuildBitCast(ctx->builder, val, ctx->f16, "");
      temps[i] = LLVMBuildFPExt(ctx->builder, val, ctx->f32, "");
   }
   return ac_build_gather_values(ctx, temps, 2);
}

static LLVMValueRef emit_ddxy(struct ac_nir_context *ctx, nir_intrinsic_op op, LLVMValueRef src0)
{
   unsigned mask;
   int idx;
   LLVMValueRef result;

   if (op == nir_intrinsic_ddx_fine)
      mask = AC_TID_MASK_LEFT;
   else if (op == nir_intrinsic_ddy_fine)
      mask = AC_TID_MASK_TOP;
   else
      mask = AC_TID_MASK_TOP_LEFT;

   /* for DDX we want to next X pixel, DDY next Y pixel. */
   if (op == nir_intrinsic_ddx_fine || op == nir_intrinsic_ddx_coarse || op == nir_intrinsic_ddx)
      idx = 1;
   else
      idx = 2;

   result = ac_build_ddxy(&ctx->ac, mask, idx, src0);
   return result;
}

struct waterfall_context {
   LLVMBasicBlockRef phi_bb[2];
   bool use_waterfall;
};

/* To deal with divergent descriptors we can create a loop that handles all
 * lanes with the same descriptor on a given iteration (henceforth a
 * waterfall loop).
 *
 * These helper create the begin and end of the loop leaving the caller
 * to implement the body.
 *
 * params:
 *  - ctx is the usual nir context
 *  - wctx is a temporary struct containing some loop info. Can be left uninitialized.
 *  - value is the possibly divergent value for which we built the loop
 *  - divergent is whether value is actually divergent. If false we just pass
 *     things through.
 */
static LLVMValueRef enter_waterfall(struct ac_nir_context *ctx, struct waterfall_context *wctx,
                                    LLVMValueRef value, bool divergent)
{
   /* If the app claims the value is divergent but it is constant we can
    * end up with a dynamic index of NULL. */
   if (!value)
      divergent = false;

   wctx->use_waterfall = divergent;
   if (!divergent)
      return value;

   ac_build_bgnloop(&ctx->ac, 6000);

   LLVMValueRef active = ctx->ac.i1true;
   LLVMValueRef scalar_value[NIR_MAX_VEC_COMPONENTS];

   for (unsigned i = 0; i < ac_get_llvm_num_components(value); i++) {
      LLVMValueRef comp = ac_llvm_extract_elem(&ctx->ac, value, i);
      scalar_value[i] = ac_build_readlane(&ctx->ac, comp, NULL);
      active = LLVMBuildAnd(ctx->ac.builder, active,
                            LLVMBuildICmp(ctx->ac.builder, LLVMIntEQ, comp, scalar_value[i], ""), "");
   }

   wctx->phi_bb[0] = LLVMGetInsertBlock(ctx->ac.builder);
   ac_build_ifcc(&ctx->ac, active, 6001);

   return ac_build_gather_values(&ctx->ac, scalar_value, ac_get_llvm_num_components(value));
}

static LLVMValueRef exit_waterfall(struct ac_nir_context *ctx, struct waterfall_context *wctx,
                                   LLVMValueRef value)
{
   LLVMValueRef ret = NULL;
   LLVMValueRef phi_src[2];
   LLVMValueRef cc_phi_src[2] = {
      ctx->ac.i32_0,
      LLVMConstInt(ctx->ac.i32, 0xffffffff, false),
   };

   if (!wctx->use_waterfall)
      return value;

   wctx->phi_bb[1] = LLVMGetInsertBlock(ctx->ac.builder);

   ac_build_endif(&ctx->ac, 6001);

   if (value) {
      phi_src[0] = LLVMGetUndef(LLVMTypeOf(value));
      phi_src[1] = value;

      ret = ac_build_phi(&ctx->ac, LLVMTypeOf(value), 2, phi_src, wctx->phi_bb);
   }

   /*
    * By using the optimization barrier on the exit decision, we decouple
    * the operations from the break, and hence avoid LLVM hoisting the
    * opteration into the break block.
    */
   LLVMValueRef cc = ac_build_phi(&ctx->ac, ctx->ac.i32, 2, cc_phi_src, wctx->phi_bb);
   ac_build_optimization_barrier(&ctx->ac, &cc, false);

   LLVMValueRef active =
      LLVMBuildICmp(ctx->ac.builder, LLVMIntNE, cc, ctx->ac.i32_0, "uniform_active2");
   ac_build_ifcc(&ctx->ac, active, 6002);
   ac_build_break(&ctx->ac);
   ac_build_endif(&ctx->ac, 6002);

   ac_build_endloop(&ctx->ac, 6000);
   return ret;
}

static LLVMValueRef
ac_build_const_int_vec(struct ac_llvm_context *ctx, LLVMTypeRef type, long long val, bool sign_extend)
{
   unsigned num_components = LLVMGetTypeKind(type) == LLVMVectorTypeKind ? LLVMGetVectorSize(type) : 1;

   if (num_components == 1)
      return LLVMConstInt(type, val, sign_extend);

   assert(num_components == 2);
   assert(ac_get_elem_bits(ctx, type) == 16);

   LLVMTypeRef elem_type = LLVMGetElementType(type);

   LLVMValueRef elems[2];
   for (unsigned i = 0; i < 2; ++i)
      elems[i] = LLVMConstInt(elem_type, val, sign_extend);

   return LLVMConstVector(elems, 2);
}

static bool visit_alu(struct ac_nir_context *ctx, const nir_alu_instr *instr)
{
   LLVMValueRef src[16], result = NULL;
   unsigned num_components = instr->def.num_components;
   LLVMTypeRef def_type = get_def_type(ctx, &instr->def);

   assert(nir_op_infos[instr->op].num_inputs <= ARRAY_SIZE(src));
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++)
      src[i] = get_alu_src(ctx, instr->src[i], nir_ssa_alu_instr_src_components(instr, i));

   switch (instr->op) {
   case nir_op_mov:
      result = src[0];
      break;
   case nir_op_fneg:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      result = LLVMBuildFNeg(ctx->ac.builder, src[0], "");
      break;
   case nir_op_inot:
      result = LLVMBuildNot(ctx->ac.builder, src[0], "");
      break;
   case nir_op_iadd:
      if (instr->no_unsigned_wrap)
         result = LLVMBuildNUWAdd(ctx->ac.builder, src[0], src[1], "");
      else if (instr->no_signed_wrap)
         result = LLVMBuildNSWAdd(ctx->ac.builder, src[0], src[1], "");
      else
         result = LLVMBuildAdd(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_uadd_sat:
   case nir_op_iadd_sat: {
      char name[64], type[64];
      ac_build_type_name_for_intr(def_type, type, sizeof(type));
      snprintf(name, sizeof(name), "llvm.%cadd.sat.%s",
               instr->op == nir_op_uadd_sat ? 'u' : 's', type);
      result = ac_build_intrinsic(&ctx->ac, name, def_type, src, 2, 0);
      break;
   }
   case nir_op_usub_sat:
   case nir_op_isub_sat: {
      char name[64], type[64];
      ac_build_type_name_for_intr(def_type, type, sizeof(type));
      snprintf(name, sizeof(name), "llvm.%csub.sat.%s",
               instr->op == nir_op_usub_sat ? 'u' : 's', type);
      result = ac_build_intrinsic(&ctx->ac, name, def_type, src, 2, 0);
      break;
   }
   case nir_op_fadd:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      src[1] = ac_to_float(&ctx->ac, src[1]);
      result = LLVMBuildFAdd(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_fsub:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      src[1] = ac_to_float(&ctx->ac, src[1]);
      result = LLVMBuildFSub(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_isub:
      if (instr->no_unsigned_wrap)
         result = LLVMBuildNUWSub(ctx->ac.builder, src[0], src[1], "");
      else if (instr->no_signed_wrap)
         result = LLVMBuildNSWSub(ctx->ac.builder, src[0], src[1], "");
      else
         result = LLVMBuildSub(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_imul24_relaxed:
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.mul.i24.i32", ctx->ac.i32, src, 2, 0);
      break;
   case nir_op_umul24_relaxed:
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.mul.u24.i32", ctx->ac.i32, src, 2, 0);
      break;
   case nir_op_imul:
      if (instr->no_unsigned_wrap)
         result = LLVMBuildNUWMul(ctx->ac.builder, src[0], src[1], "");
      else if (instr->no_signed_wrap)
         result = LLVMBuildNSWMul(ctx->ac.builder, src[0], src[1], "");
      else
         result = LLVMBuildMul(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_fmul:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      src[1] = ac_to_float(&ctx->ac, src[1]);
      result = LLVMBuildFMul(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_fmulz:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      src[1] = ac_to_float(&ctx->ac, src[1]);
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.fmul.legacy", ctx->ac.f32,
                                  src, 2, 0);
      break;
   case nir_op_frcp:
      result = emit_intrin_1f_param_scalar(&ctx->ac, "llvm.amdgcn.rcp",
                                           ac_to_float_type(&ctx->ac, def_type), src[0]);
      if (ctx->abi->clamp_div_by_zero)
         result = ac_build_fmin(&ctx->ac, result,
                                LLVMConstReal(ac_to_float_type(&ctx->ac, def_type), FLT_MAX));
      break;
   case nir_op_iand:
      result = LLVMBuildAnd(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_ior:
      result = LLVMBuildOr(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_ixor:
      result = LLVMBuildXor(ctx->ac.builder, src[0], src[1], "");
      break;
   case nir_op_ishl:
   case nir_op_ishr:
   case nir_op_ushr: {
      if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[1])) <
          ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])))
         src[1] = LLVMBuildZExt(ctx->ac.builder, src[1], LLVMTypeOf(src[0]), "");
      else if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[1])) >
               ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])))
         src[1] = LLVMBuildTrunc(ctx->ac.builder, src[1], LLVMTypeOf(src[0]), "");
      LLVMTypeRef type = LLVMTypeOf(src[1]);
      src[1] = LLVMBuildAnd(ctx->ac.builder, src[1],
                            ac_build_const_int_vec(&ctx->ac, type, ac_get_elem_bits(&ctx->ac, type) - 1, false), "");
      switch (instr->op) {
      case nir_op_ishl:
         result = LLVMBuildShl(ctx->ac.builder, src[0], src[1], "");
         break;
      case nir_op_ishr:
         result = LLVMBuildAShr(ctx->ac.builder, src[0], src[1], "");
         break;
      case nir_op_ushr:
         result = LLVMBuildLShr(ctx->ac.builder, src[0], src[1], "");
         break;
      default:
         break;
      }
      break;
   }
   case nir_op_ilt:
      result = emit_int_cmp(&ctx->ac, LLVMIntSLT, src[0], src[1]);
      break;
   case nir_op_ine:
      result = emit_int_cmp(&ctx->ac, LLVMIntNE, src[0], src[1]);
      break;
   case nir_op_ieq:
      result = emit_int_cmp(&ctx->ac, LLVMIntEQ, src[0], src[1]);
      break;
   case nir_op_ige:
      result = emit_int_cmp(&ctx->ac, LLVMIntSGE, src[0], src[1]);
      break;
   case nir_op_ult:
      result = emit_int_cmp(&ctx->ac, LLVMIntULT, src[0], src[1]);
      break;
   case nir_op_uge:
      result = emit_int_cmp(&ctx->ac, LLVMIntUGE, src[0], src[1]);
      break;
   case nir_op_feq:
      result = emit_float_cmp(&ctx->ac, LLVMRealOEQ, src[0], src[1]);
      break;
   case nir_op_fneu:
      result = emit_float_cmp(&ctx->ac, LLVMRealUNE, src[0], src[1]);
      break;
   case nir_op_fequ:
      result = emit_float_cmp(&ctx->ac, LLVMRealUEQ, src[0], src[1]);
      break;
   case nir_op_fneo:
      result = emit_float_cmp(&ctx->ac, LLVMRealONE, src[0], src[1]);
      break;
   case nir_op_flt:
      result = emit_float_cmp(&ctx->ac, LLVMRealOLT, src[0], src[1]);
      break;
   case nir_op_fge:
      result = emit_float_cmp(&ctx->ac, LLVMRealOGE, src[0], src[1]);
      break;
   case nir_op_fltu:
      result = emit_float_cmp(&ctx->ac, LLVMRealULT, src[0], src[1]);
      break;
   case nir_op_fgeu:
      result = emit_float_cmp(&ctx->ac, LLVMRealUGE, src[0], src[1]);
      break;
   case nir_op_funord:
      result = emit_float_cmp(&ctx->ac, LLVMRealUNO, src[0], src[1]);
      break;
   case nir_op_ford:
      result = emit_float_cmp(&ctx->ac, LLVMRealORD, src[0], src[1]);
      break;
   case nir_op_fabs:
      result =
         emit_intrin_1f_param(&ctx->ac, "llvm.fabs", ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_fsat:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      result = ac_build_fsat(&ctx->ac, src[0],
                             ac_to_float_type(&ctx->ac, def_type));
      break;
   case nir_op_iabs:
      result = emit_iabs(&ctx->ac, src[0]);
      break;
   case nir_op_imax:
      result = ac_build_imax(&ctx->ac, src[0], src[1]);
      break;
   case nir_op_imin:
      result = ac_build_imin(&ctx->ac, src[0], src[1]);
      break;
   case nir_op_umax:
      result = ac_build_umax(&ctx->ac, src[0], src[1]);
      break;
   case nir_op_umin:
      result = ac_build_umin(&ctx->ac, src[0], src[1]);
      break;
   case nir_op_isign:
      result = ac_build_isign(&ctx->ac, src[0]);
      break;
   case nir_op_fsign:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      result = ac_build_fsign(&ctx->ac, src[0]);
      break;
   case nir_op_ffloor:
      result =
         emit_intrin_1f_param(&ctx->ac, "llvm.floor", ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_ftrunc:
      result =
         emit_intrin_1f_param(&ctx->ac, "llvm.trunc", ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_fceil:
      result =
         emit_intrin_1f_param(&ctx->ac, "llvm.ceil", ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_fround_even:
      result =
         emit_intrin_1f_param(&ctx->ac, "llvm.rint", ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_ffract:
      result = emit_intrin_1f_param_scalar(&ctx->ac, "llvm.amdgcn.fract",
                                           ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_fsin_amd:
   case nir_op_fcos_amd:
      /* before GFX9, v_sin_f32 and v_cos_f32 had a valid input domain of [-256, +256] */
      if (ctx->ac.gfx_level < GFX9)
         src[0] = emit_intrin_1f_param_scalar(&ctx->ac, "llvm.amdgcn.fract",
                                              ac_to_float_type(&ctx->ac, def_type), src[0]);
      result =
         emit_intrin_1f_param(&ctx->ac, instr->op == nir_op_fsin_amd ? "llvm.amdgcn.sin" : "llvm.amdgcn.cos",
                              ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_fsqrt:
      result =
         emit_intrin_1f_param(&ctx->ac, "llvm.sqrt", ac_to_float_type(&ctx->ac, def_type), src[0]);
      LLVMSetMetadata(result, ctx->ac.fpmath_md_kind, ctx->ac.three_md);
      break;
   case nir_op_fexp2:
      result =
         emit_intrin_1f_param(&ctx->ac, "llvm.exp2", ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_flog2:
      result =
         emit_intrin_1f_param(&ctx->ac, "llvm.log2", ac_to_float_type(&ctx->ac, def_type), src[0]);
      break;
   case nir_op_frsq:
      result = emit_intrin_1f_param_scalar(&ctx->ac, "llvm.amdgcn.rsq",
                                           ac_to_float_type(&ctx->ac, def_type), src[0]);
      if (ctx->abi->clamp_div_by_zero)
         result = ac_build_fmin(&ctx->ac, result,
                                LLVMConstReal(ac_to_float_type(&ctx->ac, def_type), FLT_MAX));
      break;
   case nir_op_frexp_exp:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      result = ac_build_frexp_exp(&ctx->ac, src[0], ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])));
      if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])) == 16)
         result = LLVMBuildSExt(ctx->ac.builder, result, ctx->ac.i32, "");
      break;
   case nir_op_frexp_sig:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      result = ac_build_frexp_mant(&ctx->ac, src[0], instr->def.bit_size);
      break;
   case nir_op_fmax:
      result = emit_intrin_2f_param(&ctx->ac, "llvm.maxnum", ac_to_float_type(&ctx->ac, def_type),
                                    src[0], src[1]);
      if (ctx->ac.gfx_level < GFX9 && instr->def.bit_size == 32) {
         /* Only pre-GFX9 chips do not flush denorms. */
         result = ac_build_canonicalize(&ctx->ac, result, instr->def.bit_size);
      }
      break;
   case nir_op_fmin:
      result = emit_intrin_2f_param(&ctx->ac, "llvm.minnum", ac_to_float_type(&ctx->ac, def_type),
                                    src[0], src[1]);
      if (ctx->ac.gfx_level < GFX9 && instr->def.bit_size == 32) {
         /* Only pre-GFX9 chips do not flush denorms. */
         result = ac_build_canonicalize(&ctx->ac, result, instr->def.bit_size);
      }
      break;
   case nir_op_ffma:
      /* FMA is slow on gfx6-8, so it shouldn't be used. */
      assert(instr->def.bit_size != 32 || ctx->ac.gfx_level >= GFX9);
      result = emit_intrin_3f_param(&ctx->ac, "llvm.fma", ac_to_float_type(&ctx->ac, def_type),
                                    src[0], src[1], src[2]);
      break;
   case nir_op_ffmaz:
      assert(ctx->ac.gfx_level >= GFX10_3);
      src[0] = ac_to_float(&ctx->ac, src[0]);
      src[1] = ac_to_float(&ctx->ac, src[1]);
      src[2] = ac_to_float(&ctx->ac, src[2]);
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.fma.legacy", ctx->ac.f32,
                                  src, 3, 0);
      break;
   case nir_op_ldexp:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      if (ac_get_elem_bits(&ctx->ac, def_type) == 32)
         result = ac_build_intrinsic(&ctx->ac, "llvm.ldexp.f32.i32", ctx->ac.f32, src, 2, 0);
      else if (ac_get_elem_bits(&ctx->ac, def_type) == 16)
         result = ac_build_intrinsic(&ctx->ac, "llvm.ldexp.f16.i32", ctx->ac.f16, src, 2, 0);
      else
         result = ac_build_intrinsic(&ctx->ac, "llvm.ldexp.f64.i32", ctx->ac.f64, src, 2, 0);
      break;
   case nir_op_bfm:
      result = emit_bfm(&ctx->ac, src[0], src[1]);
      break;
   case nir_op_bitfield_select:
      result = emit_bitfield_select(&ctx->ac, src[0], src[1], src[2]);
      break;
   case nir_op_ubfe:
      result = ac_build_bfe(&ctx->ac, src[0], src[1], src[2], false);
      break;
   case nir_op_ibfe:
      result = ac_build_bfe(&ctx->ac, src[0], src[1], src[2], true);
      break;
   case nir_op_bitfield_reverse:
      result = ac_build_bitfield_reverse(&ctx->ac, src[0]);
      break;
   case nir_op_bit_count:
      result = ac_build_bit_count(&ctx->ac, src[0]);
      break;
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec5:
   case nir_op_vec8:
   case nir_op_vec16:
      for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++)
         src[i] = ac_to_integer(&ctx->ac, src[i]);
      result = ac_build_gather_values(&ctx->ac, src, num_components);
      break;
   case nir_op_f2i8:
   case nir_op_f2i16:
   case nir_op_f2i32:
   case nir_op_f2i64:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      result = LLVMBuildFPToSI(ctx->ac.builder, src[0], def_type, "");
      break;
   case nir_op_f2u8:
   case nir_op_f2u16:
   case nir_op_f2u32:
   case nir_op_f2u64:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      result = LLVMBuildFPToUI(ctx->ac.builder, src[0], def_type, "");
      break;
   case nir_op_i2f16:
   case nir_op_i2f32:
   case nir_op_i2f64:
      result = LLVMBuildSIToFP(ctx->ac.builder, src[0], ac_to_float_type(&ctx->ac, def_type), "");
      break;
   case nir_op_u2f16:
   case nir_op_u2f32:
   case nir_op_u2f64:
      result = LLVMBuildUIToFP(ctx->ac.builder, src[0], ac_to_float_type(&ctx->ac, def_type), "");
      break;
   case nir_op_f2f16_rtz: {
      src[0] = ac_to_float(&ctx->ac, src[0]);

      if (LLVMTypeOf(src[0]) == ctx->ac.f64)
         src[0] = LLVMBuildFPTrunc(ctx->ac.builder, src[0], ctx->ac.f32, "");

      /* Fast path conversion. This only works if NIR is vectorized
       * to vec2 16.
       */
      if (LLVMTypeOf(src[0]) == ctx->ac.v2f32) {
         LLVMValueRef args[] = {
            ac_llvm_extract_elem(&ctx->ac, src[0], 0),
            ac_llvm_extract_elem(&ctx->ac, src[0], 1),
         };
         result = ac_build_cvt_pkrtz_f16(&ctx->ac, args);
         break;
      }

      assert(ac_get_llvm_num_components(src[0]) == 1);
      LLVMValueRef param[2] = {src[0], LLVMGetUndef(ctx->ac.f32)};
      result = ac_build_cvt_pkrtz_f16(&ctx->ac, param);
      result = LLVMBuildExtractElement(ctx->ac.builder, result, ctx->ac.i32_0, "");
      break;
   }
   case nir_op_f2f16:
   case nir_op_f2f16_rtne:
   case nir_op_f2f32:
   case nir_op_f2f64:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])) < ac_get_elem_bits(&ctx->ac, def_type))
         result = LLVMBuildFPExt(ctx->ac.builder, src[0], ac_to_float_type(&ctx->ac, def_type), "");
      else
         result =
            LLVMBuildFPTrunc(ctx->ac.builder, src[0], ac_to_float_type(&ctx->ac, def_type), "");
      break;
   case nir_op_u2u8:
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_u2u64:
      if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])) < ac_get_elem_bits(&ctx->ac, def_type))
         result = LLVMBuildZExt(ctx->ac.builder, src[0], def_type, "");
      else
         result = LLVMBuildTrunc(ctx->ac.builder, src[0], def_type, "");
      break;
   case nir_op_i2i8:
   case nir_op_i2i16:
   case nir_op_i2i32:
   case nir_op_i2i64:
      if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src[0])) < ac_get_elem_bits(&ctx->ac, def_type))
         result = LLVMBuildSExt(ctx->ac.builder, src[0], def_type, "");
      else
         result = LLVMBuildTrunc(ctx->ac.builder, src[0], def_type, "");
      break;
   case nir_op_bcsel:
      result = emit_bcsel(&ctx->ac, src[0], src[1], src[2]);
      break;
   case nir_op_find_lsb:
      result = ac_find_lsb(&ctx->ac, ctx->ac.i32, src[0]);
      break;
   case nir_op_ufind_msb:
      result = ac_build_umsb(&ctx->ac, src[0], ctx->ac.i32, false);
      break;
   case nir_op_ifind_msb:
      result = ac_build_imsb(&ctx->ac, src[0], ctx->ac.i32);
      break;
   case nir_op_ufind_msb_rev:
      result = ac_build_umsb(&ctx->ac, src[0], ctx->ac.i32, true);
      break;
   case nir_op_ifind_msb_rev:
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.sffbh.i32", ctx->ac.i32, &src[0], 1,
                                  0);
      break;
   case nir_op_uclz: {
      LLVMValueRef params[2] = {
         src[0],
         ctx->ac.i1false,
      };
      result = ac_build_intrinsic(&ctx->ac, "llvm.ctlz.i32", ctx->ac.i32, params, 2, 0);
      break;
   }
   case nir_op_uadd_carry:
      result = emit_uint_carry(&ctx->ac, "llvm.uadd.with.overflow.i32", src[0], src[1]);
      break;
   case nir_op_usub_borrow:
      result = emit_uint_carry(&ctx->ac, "llvm.usub.with.overflow.i32", src[0], src[1]);
      break;
   case nir_op_b2f16:
   case nir_op_b2f32:
   case nir_op_b2f64:
      result = emit_b2f(&ctx->ac, src[0], instr->def.bit_size);
      break;
   case nir_op_b2i8:
   case nir_op_b2i16:
   case nir_op_b2i32:
   case nir_op_b2i64:
      result = emit_b2i(&ctx->ac, src[0], instr->def.bit_size);
      break;
   case nir_op_b2b1: /* after loads */
      result = emit_i2b(&ctx->ac, src[0]);
      break;
   case nir_op_b2b16: /* before stores */
      result = LLVMBuildZExt(ctx->ac.builder, src[0], ctx->ac.i16, "");
      break;
   case nir_op_b2b32: /* before stores */
      result = LLVMBuildZExt(ctx->ac.builder, src[0], ctx->ac.i32, "");
      break;
   case nir_op_fquantize2f16:
      result = emit_f2f16(&ctx->ac, src[0]);
      break;
   case nir_op_umul_high:
      result = emit_umul_high(&ctx->ac, src[0], src[1]);
      break;
   case nir_op_imul_high:
      result = emit_imul_high(&ctx->ac, src[0], src[1]);
      break;
   case nir_op_pack_half_2x16_rtz_split:
   case nir_op_pack_half_2x16_split:
      src[0] = ac_to_float(&ctx->ac, src[0]);
      src[1] = ac_to_float(&ctx->ac, src[1]);
      result = LLVMBuildBitCast(ctx->ac.builder,
                                ac_build_cvt_pkrtz_f16(&ctx->ac, src),
                                ctx->ac.i32, "");
      break;
   case nir_op_pack_snorm_2x16:
   case nir_op_pack_unorm_2x16: {
      unsigned bit_size = instr->src[0].src.ssa->bit_size;
      /* Only support 16 and 32bit. */
      assert(bit_size == 16 || bit_size == 32);

      LLVMValueRef data = src[0];
      /* Work around for pre-GFX9 GPU which don't have fp16 pknorm instruction. */
      if (bit_size == 16 && ctx->ac.gfx_level < GFX9) {
         data = LLVMBuildFPExt(ctx->ac.builder, data, ctx->ac.v2f32, "");
         bit_size = 32;
      }

      LLVMValueRef (*pack)(struct ac_llvm_context *ctx, LLVMValueRef args[2]);
      if (bit_size == 32) {
         pack = instr->op == nir_op_pack_snorm_2x16 ?
            ac_build_cvt_pknorm_i16 : ac_build_cvt_pknorm_u16;
      } else {
         pack = instr->op == nir_op_pack_snorm_2x16 ?
            ac_build_cvt_pknorm_i16_f16 : ac_build_cvt_pknorm_u16_f16;
      }
      result = emit_pack_2x16(&ctx->ac, data, pack);
      break;
   }
   case nir_op_pack_uint_2x16: {
      LLVMValueRef comp[2];

      comp[0] = LLVMBuildExtractElement(ctx->ac.builder, src[0], ctx->ac.i32_0, "");
      comp[1] = LLVMBuildExtractElement(ctx->ac.builder, src[0], ctx->ac.i32_1, "");

      result = ac_build_cvt_pk_u16(&ctx->ac, comp, 16, false);
      break;
   }
   case nir_op_pack_sint_2x16: {
      LLVMValueRef comp[2];

      comp[0] = LLVMBuildExtractElement(ctx->ac.builder, src[0], ctx->ac.i32_0, "");
      comp[1] = LLVMBuildExtractElement(ctx->ac.builder, src[0], ctx->ac.i32_1, "");

      result = ac_build_cvt_pk_i16(&ctx->ac, comp, 16, false);
      break;
   }
   case nir_op_unpack_half_2x16_split_x: {
      assert(ac_get_llvm_num_components(src[0]) == 1);
      LLVMValueRef tmp = emit_unpack_half_2x16(&ctx->ac, src[0]);
      result = LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->ac.i32_0, "");
      break;
   }
   case nir_op_unpack_half_2x16_split_y: {
      assert(ac_get_llvm_num_components(src[0]) == 1);
      LLVMValueRef tmp = emit_unpack_half_2x16(&ctx->ac, src[0]);
      result = LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->ac.i32_1, "");
      break;
   }
   case nir_op_unpack_64_4x16: {
      result = LLVMBuildBitCast(ctx->ac.builder, src[0], ctx->ac.v4i16, "");
      break;
   }

   case nir_op_unpack_64_2x32: {
      result = LLVMBuildBitCast(ctx->ac.builder, src[0],
            ctx->ac.v2i32, "");
      break;
   }
   case nir_op_unpack_64_2x32_split_x: {
      assert(ac_get_llvm_num_components(src[0]) == 1);
      LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, src[0], ctx->ac.v2i32, "");
      result = LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->ac.i32_0, "");
      break;
   }
   case nir_op_unpack_64_2x32_split_y: {
      assert(ac_get_llvm_num_components(src[0]) == 1);
      LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, src[0], ctx->ac.v2i32, "");
      result = LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->ac.i32_1, "");
      break;
   }

   case nir_op_pack_64_2x32_split: {
      LLVMValueRef tmp = ac_build_gather_values(&ctx->ac, src, 2);
      result = LLVMBuildBitCast(ctx->ac.builder, tmp, ctx->ac.i64, "");
      break;
   }

   case nir_op_pack_32_4x8: {
      result = LLVMBuildBitCast(ctx->ac.builder, src[0],
            ctx->ac.i32, "");
      break;
   }
   case nir_op_pack_32_2x16_split: {
      LLVMValueRef tmp = ac_build_gather_values(&ctx->ac, src, 2);
      result = LLVMBuildBitCast(ctx->ac.builder, tmp, ctx->ac.i32, "");
      break;
   }

   case nir_op_unpack_32_4x8:
      result = LLVMBuildBitCast(ctx->ac.builder, src[0], ctx->ac.v4i8, "");
      break;
   case nir_op_unpack_32_2x16: {
      result = LLVMBuildBitCast(ctx->ac.builder, src[0],
            ctx->ac.v2i16, "");
      break;
   }
   case nir_op_unpack_32_2x16_split_x: {
      LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, src[0], ctx->ac.v2i16, "");
      result = LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->ac.i32_0, "");
      break;
   }
   case nir_op_unpack_32_2x16_split_y: {
      LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, src[0], ctx->ac.v2i16, "");
      result = LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->ac.i32_1, "");
      break;
   }

   case nir_op_cube_amd: {
      src[0] = ac_to_float(&ctx->ac, src[0]);
      LLVMValueRef results[4];
      LLVMValueRef in[3];
      for (unsigned chan = 0; chan < 3; chan++)
         in[chan] = ac_llvm_extract_elem(&ctx->ac, src[0], chan);
      results[0] = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.cubetc", ctx->ac.f32, in, 3, 0);
      results[1] = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.cubesc", ctx->ac.f32, in, 3, 0);
      results[2] = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.cubema", ctx->ac.f32, in, 3, 0);
      results[3] = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.cubeid", ctx->ac.f32, in, 3, 0);
      result = ac_build_gather_values(&ctx->ac, results, 4);
      break;
   }

   case nir_op_extract_u8:
   case nir_op_extract_i8:
   case nir_op_extract_u16:
   case nir_op_extract_i16: {
      bool is_signed = instr->op == nir_op_extract_i16 || instr->op == nir_op_extract_i8;
      unsigned size = instr->op == nir_op_extract_u8 || instr->op == nir_op_extract_i8 ? 8 : 16;
      LLVMValueRef offset = LLVMBuildMul(ctx->ac.builder, src[1], ac_const_uint_vec(&ctx->ac, LLVMTypeOf(src[1]), size), "");
      result = LLVMBuildLShr(ctx->ac.builder, src[0], offset, "");

      LLVMTypeRef small_type = LLVMIntTypeInContext(ctx->ac.context, size);
      if (instr->def.num_components > 1)
         small_type = LLVMVectorType(small_type, instr->def.num_components);

      result = LLVMBuildTrunc(ctx->ac.builder, result, small_type, "");
      if (is_signed)
         result = LLVMBuildSExt(ctx->ac.builder, result, LLVMTypeOf(src[0]), "");
      else
         result = LLVMBuildZExt(ctx->ac.builder, result, LLVMTypeOf(src[0]), "");
      break;
   }

   case nir_op_insert_u8:
   case nir_op_insert_u16: {
      unsigned size = instr->op == nir_op_insert_u8 ? 8 : 16;
      LLVMValueRef offset = LLVMConstInt(LLVMTypeOf(src[0]), nir_src_as_uint(instr->src[1].src) * size, false);
      LLVMValueRef mask = LLVMConstInt(LLVMTypeOf(src[0]), BITFIELD_MASK(size), false);
      result = LLVMBuildShl(ctx->ac.builder, LLVMBuildAnd(ctx->ac.builder, src[0], mask, ""), offset, "");
      break;
   }

   case nir_op_sdot_4x8_iadd:
   case nir_op_sdot_4x8_iadd_sat: {
      if (ctx->ac.gfx_level >= GFX11) {
         result = ac_build_sudot_4x8(&ctx->ac, src[0], src[1], src[2],
                                     instr->op == nir_op_sdot_4x8_iadd_sat, 0x3);
      } else {
         const char *name = "llvm.amdgcn.sdot4";
         src[3] = LLVMConstInt(ctx->ac.i1, instr->op == nir_op_sdot_4x8_iadd_sat, false);
         result = ac_build_intrinsic(&ctx->ac, name, def_type, src, 4, 0);
      }
      break;
   }
   case nir_op_sudot_4x8_iadd:
   case nir_op_sudot_4x8_iadd_sat: {
      result = ac_build_sudot_4x8(&ctx->ac, src[0], src[1], src[2],
                                  instr->op == nir_op_sudot_4x8_iadd_sat, 0x1);
      break;
   }
   case nir_op_udot_4x8_uadd:
   case nir_op_udot_4x8_uadd_sat: {
      const char *name = "llvm.amdgcn.udot4";
      src[3] = LLVMConstInt(ctx->ac.i1, instr->op == nir_op_udot_4x8_uadd_sat, false);
      result = ac_build_intrinsic(&ctx->ac, name, def_type, src, 4, 0);
      break;
   }

   case nir_op_sdot_2x16_iadd:
   case nir_op_udot_2x16_uadd:
   case nir_op_sdot_2x16_iadd_sat:
   case nir_op_udot_2x16_uadd_sat: {
      const char *name = instr->op == nir_op_sdot_2x16_iadd ||
                         instr->op == nir_op_sdot_2x16_iadd_sat
                         ? "llvm.amdgcn.sdot2" : "llvm.amdgcn.udot2";
      src[0] = LLVMBuildBitCast(ctx->ac.builder, src[0], ctx->ac.v2i16, "");
      src[1] = LLVMBuildBitCast(ctx->ac.builder, src[1], ctx->ac.v2i16, "");
      src[3] = LLVMConstInt(ctx->ac.i1, instr->op == nir_op_sdot_2x16_iadd_sat ||
                                        instr->op == nir_op_udot_2x16_uadd_sat, false);
      result = ac_build_intrinsic(&ctx->ac, name, def_type, src, 4, 0);
      break;
   }

   case nir_op_bfdot2_bfadd: {
      const char *name = "llvm.amdgcn.fdot2.bf16.bf16";
      LLVMTypeRef vec2_type = ctx->ac.v2bf16;
      LLVMTypeRef scalar_type = ctx->ac.bf16;
#if LLVM_VERSION_MAJOR < 19 || (LLVM_VERSION_MAJOR == 19 && LLVM_VERSION_MINOR == 0)
      /* Before LLVM 19.1, bf16 fdot used integer operands. */
      vec2_type = ctx->ac.v2i16;
      scalar_type = ctx->ac.i16;
#endif
      src[0] = LLVMBuildBitCast(ctx->ac.builder, src[0], vec2_type, "");
      src[1] = LLVMBuildBitCast(ctx->ac.builder, src[1], vec2_type, "");
      src[2] = LLVMBuildBitCast(ctx->ac.builder, src[2], scalar_type, "");
      result = ac_build_intrinsic(&ctx->ac, name, scalar_type, src, 3, 0);
      break;
   }

   case nir_op_msad_4x8:
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.msad.u8", ctx->ac.i32,
                                  (LLVMValueRef[]){src[1], src[0], src[2]}, 3, 0);
      break;

   case nir_op_mqsad_4x8:
      src[1] = LLVMBuildBitCast(ctx->ac.builder, src[1], ctx->ac.i64, "");
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.mqsad.u32.u8", ctx->ac.v4i32,
                                  (LLVMValueRef[]){src[1], src[0], src[2]}, 3, 0);
      break;

   case nir_op_shfr:
      result = ac_build_intrinsic(&ctx->ac, "llvm.fshr.i32", ctx->ac.i32,
                                  (LLVMValueRef[]){src[0], src[1], src[2]}, 3, 0);
      break;

   case nir_op_alignbyte_amd:
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.alignbyte", ctx->ac.i32,
                                  (LLVMValueRef[]){src[0], src[1], src[2]}, 3, 0);
      break;

   default:
      fprintf(stderr, "Unknown NIR alu instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      return false;
   }

   if (result) {
      LLVMTypeKind type_kind = LLVMGetTypeKind(LLVMTypeOf(result));
      bool is_float = type_kind == LLVMHalfTypeKind || type_kind == LLVMFloatTypeKind || type_kind == LLVMDoubleTypeKind;
      if (ctx->ac.float_mode == AC_FLOAT_MODE_DENORM_FLUSH_TO_ZERO && is_float)
         result = ac_build_canonicalize(&ctx->ac, result, instr->def.bit_size);

      result = ac_to_integer_or_pointer(&ctx->ac, result);
      ctx->ssa_defs[instr->def.index] = result;
   }
   return true;
}

static void visit_load_const(struct ac_nir_context *ctx, const nir_load_const_instr *instr)
{
   assert(instr->def.num_components == 1);

   ctx->ssa_defs[instr->def.index] =
      LLVMConstInt(LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size),
                   nir_const_value_as_uint(instr->value[0], instr->def.bit_size), false);
}

/* Gather4 should follow the same rules as bilinear filtering, but the hardware
 * incorrectly forces nearest filtering if the texture format is integer.
 * The only effect it has on Gather4, which always returns 4 texels for
 * bilinear filtering, is that the final coordinates are off by 0.5 of
 * the texel size.
 *
 * The workaround is to subtract 0.5 from the unnormalized coordinates,
 * or (0.5 / size) from the normalized coordinates.
 *
 * However, cube textures with 8_8_8_8 data formats require a different
 * workaround of overriding the num format to USCALED/SSCALED. This would lose
 * precision in 32-bit data formats, so it needs to be applied dynamically at
 * runtime. In this case, return an i1 value that indicates whether the
 * descriptor was overridden (and hence a fixup of the sampler result is needed).
 */
static LLVMValueRef lower_gather4_integer(struct ac_llvm_context *ctx, struct ac_image_args *args,
                                          const nir_tex_instr *instr)
{
   nir_alu_type stype = nir_alu_type_get_base_type(instr->dest_type);
   LLVMValueRef wa_8888 = NULL;
   LLVMValueRef half_texel[2];
   LLVMValueRef result;

   assert(stype == nir_type_int || stype == nir_type_uint);

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
      LLVMValueRef formats;
      LLVMValueRef data_format;
      LLVMValueRef wa_formats;

      formats = LLVMBuildExtractElement(ctx->builder, args->resource, ctx->i32_1, "");

      data_format = LLVMBuildLShr(ctx->builder, formats, LLVMConstInt(ctx->i32, 20, false), "");
      data_format =
         LLVMBuildAnd(ctx->builder, data_format, LLVMConstInt(ctx->i32, (1u << 6) - 1, false), "");
      wa_8888 = LLVMBuildICmp(ctx->builder, LLVMIntEQ, data_format,
                              LLVMConstInt(ctx->i32, V_008F14_IMG_DATA_FORMAT_8_8_8_8, false), "");

      uint32_t wa_num_format = stype == nir_type_uint
                                  ? S_008F14_NUM_FORMAT(V_008F14_IMG_NUM_FORMAT_USCALED)
                                  : S_008F14_NUM_FORMAT(V_008F14_IMG_NUM_FORMAT_SSCALED);
      wa_formats = LLVMBuildAnd(ctx->builder, formats,
                                LLVMConstInt(ctx->i32, C_008F14_NUM_FORMAT, false), "");
      wa_formats =
         LLVMBuildOr(ctx->builder, wa_formats, LLVMConstInt(ctx->i32, wa_num_format, false), "");

      formats = LLVMBuildSelect(ctx->builder, wa_8888, wa_formats, formats, "");
      args->resource =
         LLVMBuildInsertElement(ctx->builder, args->resource, formats, ctx->i32_1, "");
   }

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_RECT) {
      assert(!wa_8888);
      half_texel[0] = half_texel[1] = LLVMConstReal(ctx->f32, -0.5);
   } else {
      struct ac_image_args resinfo = {0};
      LLVMBasicBlockRef bbs[2];

      LLVMValueRef unnorm = NULL;
      LLVMValueRef default_offset = ctx->f32_0;
      if (instr->sampler_dim == GLSL_SAMPLER_DIM_2D && !instr->is_array) {
         /* In vulkan, whether the sampler uses unnormalized
          * coordinates or not is a dynamic property of the
          * sampler. Hence, to figure out whether or not we
          * need to divide by the texture size, we need to test
          * the sampler at runtime. This tests the bit set by
          * radv_init_sampler().
          */
         LLVMValueRef sampler0 =
            LLVMBuildExtractElement(ctx->builder, args->sampler, ctx->i32_0, "");
         sampler0 = LLVMBuildLShr(ctx->builder, sampler0, LLVMConstInt(ctx->i32, 15, false), "");
         sampler0 = LLVMBuildAnd(ctx->builder, sampler0, ctx->i32_1, "");
         unnorm = LLVMBuildICmp(ctx->builder, LLVMIntEQ, sampler0, ctx->i32_1, "");
         default_offset = LLVMConstReal(ctx->f32, -0.5);
      }

      bbs[0] = LLVMGetInsertBlock(ctx->builder);
      if (wa_8888 || unnorm) {
         assert(!(wa_8888 && unnorm));
         LLVMValueRef not_needed = wa_8888 ? wa_8888 : unnorm;
         /* Skip the texture size query entirely if we don't need it. */
         ac_build_ifcc(ctx, LLVMBuildNot(ctx->builder, not_needed, ""), 2000);
         bbs[1] = LLVMGetInsertBlock(ctx->builder);
      }

      /* Query the texture size. */
      resinfo.dim = ac_get_sampler_dim(ctx->gfx_level, instr->sampler_dim, instr->is_array);
      resinfo.opcode = ac_image_get_resinfo;
      resinfo.dmask = 0xf;
      resinfo.lod = ctx->i32_0;
      resinfo.resource = args->resource;
      resinfo.attributes = AC_ATTR_INVARIANT_LOAD;
      LLVMValueRef size = ac_build_image_opcode(ctx, &resinfo);

      /* Compute -0.5 / size. */
      for (unsigned c = 0; c < 2; c++) {
         half_texel[c] =
            LLVMBuildExtractElement(ctx->builder, size, LLVMConstInt(ctx->i32, c, 0), "");
         half_texel[c] = LLVMBuildUIToFP(ctx->builder, half_texel[c], ctx->f32, "");
         half_texel[c] = ac_build_fdiv(ctx, ctx->f32_1, half_texel[c]);
         half_texel[c] =
            LLVMBuildFMul(ctx->builder, half_texel[c], LLVMConstReal(ctx->f32, -0.5), "");
      }

      if (wa_8888 || unnorm) {
         ac_build_endif(ctx, 2000);

         for (unsigned c = 0; c < 2; c++) {
            LLVMValueRef values[2] = {default_offset, half_texel[c]};
            half_texel[c] = ac_build_phi(ctx, ctx->f32, 2, values, bbs);
         }
      }
   }

   for (unsigned c = 0; c < 2; c++) {
      LLVMValueRef tmp;
      tmp = LLVMBuildBitCast(ctx->builder, args->coords[c], ctx->f32, "");
      args->coords[c] = LLVMBuildFAdd(ctx->builder, tmp, half_texel[c], "");
   }

   args->attributes = AC_ATTR_INVARIANT_LOAD;
   result = ac_build_image_opcode(ctx, args);

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
      LLVMValueRef tmp, tmp2;

      /* if the cube workaround is in place, f2i the result. */
      for (unsigned c = 0; c < 4; c++) {
         tmp = LLVMBuildExtractElement(ctx->builder, result, LLVMConstInt(ctx->i32, c, false), "");
         if (stype == nir_type_uint)
            tmp2 = LLVMBuildFPToUI(ctx->builder, tmp, ctx->i32, "");
         else
            tmp2 = LLVMBuildFPToSI(ctx->builder, tmp, ctx->i32, "");
         tmp = LLVMBuildBitCast(ctx->builder, tmp, ctx->i32, "");
         tmp2 = LLVMBuildBitCast(ctx->builder, tmp2, ctx->i32, "");
         tmp = LLVMBuildSelect(ctx->builder, wa_8888, tmp2, tmp, "");
         tmp = LLVMBuildBitCast(ctx->builder, tmp, ctx->f32, "");
         result =
            LLVMBuildInsertElement(ctx->builder, result, tmp, LLVMConstInt(ctx->i32, c, false), "");
      }
   }
   return result;
}

static LLVMValueRef build_tex_intrinsic(struct ac_nir_context *ctx, const nir_tex_instr *instr,
                                        struct ac_image_args *args)
{
   assert((!args->tfe || !args->d16) && "unsupported");

   if (instr->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
      unsigned mask = nir_def_components_read(&instr->def);

      /* Buffers don't support A16. */
      if (args->a16)
         args->coords[0] = LLVMBuildZExt(ctx->ac.builder, args->coords[0], ctx->ac.i32, "");

      return ac_build_buffer_load_format(&ctx->ac, args->resource, args->coords[0], ctx->ac.i32_0,
                                         util_last_bit(mask), 0, true,
                                         instr->def.bit_size == 16,
                                         args->tfe);
   }

   args->opcode = ac_image_sample;

   switch (instr->op) {
   case nir_texop_txf:
   case nir_texop_txf_ms:
      args->opcode = args->level_zero || instr->sampler_dim == GLSL_SAMPLER_DIM_MS
                        ? ac_image_load
                        : ac_image_load_mip;
      args->level_zero = false;
      break;
   case nir_texop_txs:
   case nir_texop_query_levels:
   case nir_texop_texture_samples:
      assert(!"should have been lowered");
      break;
   case nir_texop_tex:
      if (ctx->stage != MESA_SHADER_FRAGMENT &&
          (!gl_shader_stage_is_compute(ctx->stage) ||
           ctx->info->derivative_group == DERIVATIVE_GROUP_NONE)) {
         assert(!args->lod);
         args->level_zero = true;
      }
      break;
   case nir_texop_tg4:
      args->opcode = ac_image_gather4;
      if (!args->lod && !instr->is_gather_implicit_lod)
         args->level_zero = true;
      /* GFX11 supports implicit LOD, but the extension is unsupported. */
      assert(args->level_zero || ctx->ac.gfx_level < GFX11);
      break;
   case nir_texop_lod:
      args->opcode = ac_image_get_lod;
      break;
   case nir_texop_fragment_fetch_amd:
   case nir_texop_fragment_mask_fetch_amd:
      args->opcode = ac_image_load;
      args->level_zero = false;
      break;
   default:
      break;
   }

   /* MI200 doesn't have image_sample_lz, but image_sample behaves like lz. */
   if (!ctx->ac.info->has_3d_cube_border_color_mipmap)
      args->level_zero = false;

   if (instr->op == nir_texop_tg4 && ctx->ac.gfx_level <= GFX8 &&
       (instr->dest_type & (nir_type_int | nir_type_uint))) {
      return lower_gather4_integer(&ctx->ac, args, instr);
   }

   args->attributes = AC_ATTR_INVARIANT_LOAD;
   bool cs_derivs =
      gl_shader_stage_is_compute(ctx->stage) && ctx->info->derivative_group != DERIVATIVE_GROUP_NONE;
   if (ctx->stage == MESA_SHADER_FRAGMENT || cs_derivs) {
      /* Prevent texture instructions with implicit derivatives from being
       * sinked into branches. */
      switch (instr->op) {
      case nir_texop_tex:
      case nir_texop_txb:
      case nir_texop_lod:
         args->attributes |= AC_ATTR_CONVERGENT;
         break;
      default:
         break;
      }
   }

   return ac_build_image_opcode(&ctx->ac, args);
}

static LLVMValueRef extract_vector_range(struct ac_llvm_context *ctx, LLVMValueRef src,
                                         unsigned start, unsigned count)
{
   LLVMValueRef mask[] = {ctx->i32_0, ctx->i32_1, LLVMConstInt(ctx->i32, 2, false),
                          LLVMConstInt(ctx->i32, 3, false)};

   unsigned src_elements = ac_get_llvm_num_components(src);

   if (count == src_elements) {
      assert(start == 0);
      return src;
   } else if (count == 1) {
      assert(start < src_elements);
      return LLVMBuildExtractElement(ctx->builder, src, mask[start], "");
   } else {
      assert(start + count <= src_elements);
      assert(count <= 4);
      LLVMValueRef swizzle = LLVMConstVector(&mask[start], count);
      return LLVMBuildShuffleVector(ctx->builder, src, src, swizzle, "");
   }
}

static LLVMValueRef enter_waterfall_ssbo(struct ac_nir_context *ctx, struct waterfall_context *wctx,
                                         const nir_intrinsic_instr *instr, nir_src src)
{
   return enter_waterfall(ctx, wctx, get_src(ctx, src),
                          nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM);
}

static void visit_store_ssbo(struct ac_nir_context *ctx, nir_intrinsic_instr *instr)
{
   LLVMValueRef src_data = get_src(ctx, instr->src[0]);
   int elem_size_bytes = ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src_data)) / 8;
   unsigned writemask = nir_intrinsic_write_mask(instr);
   enum gl_access_qualifier access = ac_nir_get_mem_access_flags(instr);

   struct waterfall_context wctx;
   LLVMValueRef rsrc_base = enter_waterfall_ssbo(ctx, &wctx, instr, instr->src[1]);

   LLVMValueRef rsrc = ctx->abi->load_ssbo ?
      ctx->abi->load_ssbo(ctx->abi, rsrc_base, true, false) : rsrc_base;

   LLVMValueRef base_data = src_data;
   base_data = ac_trim_vector(&ctx->ac, base_data, instr->num_components);
   LLVMValueRef base_offset = get_src(ctx, instr->src[2]);

   while (writemask) {
      int start, count;
      LLVMValueRef data, offset;
      LLVMTypeRef data_type;

      u_bit_scan_consecutive_range(&writemask, &start, &count);

      if (count == 3 && elem_size_bytes != 4) {
         writemask |= 1 << (start + 2);
         count = 2;
      }
      int num_bytes = count * elem_size_bytes; /* count in bytes */

      /* we can only store 4 DWords at the same time.
       * can only happen for 64 Bit vectors. */
      if (num_bytes > 16) {
         writemask |= ((1u << (count - 2)) - 1u) << (start + 2);
         count = 2;
         num_bytes = 16;
      }

      /* check alignment of 8/16 Bit stores */
      uint32_t align_mul = nir_intrinsic_align_mul(instr);
      uint32_t align_offset = nir_intrinsic_align_offset(instr) + start * elem_size_bytes;
      uint32_t align = nir_combined_align(align_mul, align_offset & (align_mul - 1));
      if (align < MIN2(num_bytes, 4) || (ctx->ac.gfx_level == GFX6 && elem_size_bytes < 4)) {
         writemask |= BITFIELD_RANGE(start + 1, count - 1);
         count = 1;
         num_bytes = elem_size_bytes;
      }

      /* Due to alignment issues, split stores of 8-bit/16-bit
       * vectors.
       */
      if (ctx->ac.gfx_level == GFX6 && count > 1 && elem_size_bytes < 4) {
         writemask |= ((1u << (count - 1)) - 1u) << (start + 1);
         count = 1;
         num_bytes = elem_size_bytes;
      }

      data = extract_vector_range(&ctx->ac, base_data, start, count);

      offset = LLVMBuildAdd(ctx->ac.builder, base_offset,
                            LLVMConstInt(ctx->ac.i32, start * elem_size_bytes, false), "");

      if (num_bytes == 1) {
         ac_build_buffer_store_byte(&ctx->ac, rsrc, data, offset, ctx->ac.i32_0, access);
      } else if (num_bytes == 2) {
         ac_build_buffer_store_short(&ctx->ac, rsrc, data, offset, ctx->ac.i32_0, access);
      } else {
         switch (num_bytes) {
         case 16: /* v4f32 */
            data_type = ctx->ac.v4f32;
            break;
         case 12: /* v3f32 */
            data_type = ctx->ac.v3f32;
            break;
         case 8: /* v2f32 */
            data_type = ctx->ac.v2f32;
            break;
         case 4: /* f32 */
            data_type = ctx->ac.f32;
            break;
         default:
            UNREACHABLE("Malformed vector store.");
         }
         data = LLVMBuildBitCast(ctx->ac.builder, data, data_type, "");

         ac_build_buffer_store_dword(&ctx->ac, rsrc, data, NULL, offset,
                                     ctx->ac.i32_0, access);
      }
   }

   exit_waterfall(ctx, &wctx, NULL);
}

static LLVMValueRef emit_ssbo_comp_swap_64(struct ac_nir_context *ctx, LLVMValueRef descriptor,
                                           LLVMValueRef offset, LLVMValueRef compare,
                                           LLVMValueRef exchange, bool image)
{
   LLVMBasicBlockRef start_block = NULL, then_block = NULL;
   if (ctx->abi->robust_buffer_access || image) {
      LLVMValueRef size = ac_llvm_extract_elem(&ctx->ac, descriptor, 2);

      LLVMValueRef cond = LLVMBuildICmp(ctx->ac.builder, LLVMIntULT, offset, size, "");
      start_block = LLVMGetInsertBlock(ctx->ac.builder);

      ac_build_ifcc(&ctx->ac, cond, -1);

      then_block = LLVMGetInsertBlock(ctx->ac.builder);
   }

   if (image)
      offset = LLVMBuildMul(ctx->ac.builder, offset, LLVMConstInt(ctx->ac.i32, 8, false), "");

   LLVMValueRef ptr_parts[2] = {
      ac_llvm_extract_elem(&ctx->ac, descriptor, 0),
      LLVMBuildAnd(ctx->ac.builder, ac_llvm_extract_elem(&ctx->ac, descriptor, 1),
                   LLVMConstInt(ctx->ac.i32, 65535, 0), "")};

   ptr_parts[1] = LLVMBuildTrunc(ctx->ac.builder, ptr_parts[1], ctx->ac.i16, "");
   ptr_parts[1] = LLVMBuildSExt(ctx->ac.builder, ptr_parts[1], ctx->ac.i32, "");

   offset = LLVMBuildZExt(ctx->ac.builder, offset, ctx->ac.i64, "");

   LLVMValueRef ptr = ac_build_gather_values(&ctx->ac, ptr_parts, 2);
   ptr = LLVMBuildBitCast(ctx->ac.builder, ptr, ctx->ac.i64, "");
   ptr = LLVMBuildAdd(ctx->ac.builder, ptr, offset, "");
   ptr = LLVMBuildIntToPtr(ctx->ac.builder, ptr,
                           LLVMPointerTypeInContext(ctx->ac.context, AC_ADDR_SPACE_GLOBAL), "");

   LLVMValueRef result =
      ac_build_atomic_cmp_xchg(&ctx->ac, ptr, compare, exchange, "singlethread-one-as");
   result = LLVMBuildExtractValue(ctx->ac.builder, result, 0, "");

   if (ctx->abi->robust_buffer_access || image) {
      ac_build_endif(&ctx->ac, -1);

      LLVMBasicBlockRef incoming_blocks[2] = {
         start_block,
         then_block,
      };

      LLVMValueRef incoming_values[2] = {
         ctx->ac.i64_0,
         result,
      };
      LLVMValueRef ret = LLVMBuildPhi(ctx->ac.builder, ctx->ac.i64, "");
      LLVMAddIncoming(ret, incoming_values, incoming_blocks, 2);
      return ret;
   } else {
      return result;
   }
}

static const char *
translate_atomic_op_str(nir_atomic_op op)
{
   switch (op) {
   case nir_atomic_op_iadd:     return "add";
   case nir_atomic_op_imin:     return "smin";
   case nir_atomic_op_umin:     return "umin";
   case nir_atomic_op_imax:     return "smax";
   case nir_atomic_op_umax:     return "umax";
   case nir_atomic_op_iand:     return "and";
   case nir_atomic_op_ior:      return "or";
   case nir_atomic_op_ixor:     return "xor";
   case nir_atomic_op_fadd:     return "fadd";
   case nir_atomic_op_fmin:     return "fmin";
   case nir_atomic_op_fmax:     return "fmax";
   case nir_atomic_op_xchg:     return "swap";
   case nir_atomic_op_cmpxchg:  return "cmpswap";
   case nir_atomic_op_inc_wrap: return "inc";
   case nir_atomic_op_dec_wrap: return "dec";
   case nir_atomic_op_ordered_add_gfx12_amd: return "ordered.add";
   default: abort();
   }
}

static LLVMAtomicRMWBinOp
translate_atomic_op(nir_atomic_op op)
{
   switch (op) {
   case nir_atomic_op_iadd: return LLVMAtomicRMWBinOpAdd;
   case nir_atomic_op_xchg: return LLVMAtomicRMWBinOpXchg;
   case nir_atomic_op_iand: return LLVMAtomicRMWBinOpAnd;
   case nir_atomic_op_ior:  return LLVMAtomicRMWBinOpOr;
   case nir_atomic_op_ixor: return LLVMAtomicRMWBinOpXor;
   case nir_atomic_op_umin: return LLVMAtomicRMWBinOpUMin;
   case nir_atomic_op_umax: return LLVMAtomicRMWBinOpUMax;
   case nir_atomic_op_imin: return LLVMAtomicRMWBinOpMin;
   case nir_atomic_op_imax: return LLVMAtomicRMWBinOpMax;
   case nir_atomic_op_fadd: return LLVMAtomicRMWBinOpFAdd;
   default: UNREACHABLE("Unexpected atomic");
   }
}

static LLVMValueRef visit_atomic_ssbo(struct ac_nir_context *ctx, nir_intrinsic_instr *instr)
{
   nir_atomic_op nir_op = nir_intrinsic_atomic_op(instr);
   const char *op = translate_atomic_op_str(nir_op);
   bool is_float = nir_atomic_op_type(nir_op) == nir_type_float;

   LLVMTypeRef return_type = LLVMTypeOf(get_src(ctx, instr->src[2]));
   char name[64], type[8];
   LLVMValueRef params[6], descriptor;
   LLVMValueRef result;
   int arg_count = 0;

   struct waterfall_context wctx;
   LLVMValueRef rsrc_base = enter_waterfall_ssbo(ctx, &wctx, instr, instr->src[0]);

   descriptor = ctx->abi->load_ssbo ?
      ctx->abi->load_ssbo(ctx->abi, rsrc_base, true, false) : rsrc_base;

   if (instr->intrinsic == nir_intrinsic_ssbo_atomic_swap && return_type == ctx->ac.i64) {
      result = emit_ssbo_comp_swap_64(ctx, descriptor, get_src(ctx, instr->src[1]),
                                      get_src(ctx, instr->src[2]), get_src(ctx, instr->src[3]), false);
   } else {
      LLVMValueRef data = ac_llvm_extract_elem(&ctx->ac, get_src(ctx, instr->src[2]), 0);

      if (instr->intrinsic == nir_intrinsic_ssbo_atomic_swap) {
         params[arg_count++] = ac_llvm_extract_elem(&ctx->ac, get_src(ctx, instr->src[3]), 0);
      }
      if (is_float) {
         data = ac_to_float(&ctx->ac, data);
         return_type = LLVMTypeOf(data);
      }

      unsigned cache_flags =
         ac_get_hw_cache_flags(ctx->ac.gfx_level,
			       ac_nir_get_mem_access_flags(instr) | ACCESS_TYPE_ATOMIC).value;

      params[arg_count++] = data;
      params[arg_count++] = descriptor;
      params[arg_count++] = get_src(ctx, instr->src[1]); /* voffset */
      params[arg_count++] = ctx->ac.i32_0;               /* soffset */
      params[arg_count++] = LLVMConstInt(ctx->ac.i32, cache_flags, 0);

      ac_build_type_name_for_intr(return_type, type, sizeof(type));
      snprintf(name, sizeof(name), "llvm.amdgcn.raw.buffer.atomic.%s.%s", op, type);

      result = ac_build_intrinsic(&ctx->ac, name, return_type, params, arg_count, 0);

      if (is_float) {
         result = ac_to_integer(&ctx->ac, result);
      }
   }

   return exit_waterfall(ctx, &wctx, result);
}

static LLVMValueRef visit_load_buffer(struct ac_nir_context *ctx, nir_intrinsic_instr *instr)
{
   struct waterfall_context wctx;
   LLVMValueRef rsrc_base = enter_waterfall_ssbo(ctx, &wctx, instr, instr->src[0]);

   int elem_size_bytes = instr->def.bit_size / 8;
   int num_components = instr->num_components;
   enum gl_access_qualifier access = ac_nir_get_mem_access_flags(instr);

   LLVMValueRef offset = get_src(ctx, instr->src[1]);
   LLVMValueRef rsrc = ctx->abi->load_ssbo ?
      ctx->abi->load_ssbo(ctx->abi, rsrc_base, false, false) : rsrc_base;

   LLVMTypeRef def_type = get_def_type(ctx, &instr->def);
   LLVMTypeRef def_elem_type = num_components > 1 ? LLVMGetElementType(def_type) : def_type;

   LLVMValueRef results[4];
   for (int i = 0; i < num_components;) {
      int num_elems = num_components - i;
      /* Multi-component subdword loads are lowered by ac_nir_lower_subdword_loads. */
      assert(elem_size_bytes >= 4 || num_elems == 1);

      if (num_elems * elem_size_bytes > 16)
         num_elems = 16 / elem_size_bytes;
      int load_bytes = num_elems * elem_size_bytes;

      LLVMValueRef immoffset = LLVMConstInt(ctx->ac.i32, i * elem_size_bytes, false);
      LLVMValueRef voffset = LLVMBuildAdd(ctx->ac.builder, offset, immoffset, "");

      LLVMValueRef ret;

      if (load_bytes == 1) {
         ret = ac_build_buffer_load_byte(&ctx->ac, rsrc, voffset, ctx->ac.i32_0,
                                          access);
      } else if (load_bytes == 2) {
         ret = ac_build_buffer_load_short(&ctx->ac, rsrc, voffset, ctx->ac.i32_0,
                                           access);
      } else {
         assert(elem_size_bytes >= 4);
         int num_channels = load_bytes / 4;
         bool can_speculate = access & ACCESS_CAN_REORDER;

         ret = ac_build_buffer_load(&ctx->ac, rsrc, num_channels, NULL, voffset, ctx->ac.i32_0,
                                    ctx->ac.f32, access, can_speculate, false);
      }

      LLVMTypeRef ret_type = LLVMVectorType(def_elem_type, num_elems);
      ret = LLVMBuildBitCast(ctx->ac.builder, ret, ret_type, "");

      for (unsigned j = 0; j < num_elems; j++) {
         results[i + j] =
            LLVMBuildExtractElement(ctx->ac.builder, ret, LLVMConstInt(ctx->ac.i32, j, false), "");
      }
      i += num_elems;
   }

   LLVMValueRef ret = ac_build_gather_values(&ctx->ac, results, num_components);
   return exit_waterfall(ctx, &wctx, ret);
}

static LLVMValueRef enter_waterfall_ubo(struct ac_nir_context *ctx, struct waterfall_context *wctx,
                                        const nir_intrinsic_instr *instr)
{
   return enter_waterfall(ctx, wctx, get_src(ctx, instr->src[0]),
                          nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM);
}

static LLVMValueRef get_global_address(struct ac_nir_context *ctx,
                                       nir_intrinsic_instr *instr)
{
   bool is_store = instr->intrinsic == nir_intrinsic_store_global_amd;
   LLVMValueRef addr = get_src(ctx, instr->src[is_store ? 1 : 0]);

   LLVMTypeRef ptr_type = LLVMPointerTypeInContext(ctx->ac.context, AC_ADDR_SPACE_GLOBAL);

   uint32_t base = nir_intrinsic_base(instr);
   unsigned num_src = nir_intrinsic_infos[instr->intrinsic].num_srcs;
   LLVMValueRef offset = get_src(ctx, instr->src[num_src - 1]);
   offset = LLVMBuildAdd(ctx->ac.builder, offset, LLVMConstInt(ctx->ac.i32, base, false), "");

   addr = LLVMBuildIntToPtr(ctx->ac.builder, addr, ptr_type, "");
   return LLVMBuildGEP2(ctx->ac.builder, ctx->ac.i8, addr, &offset, 1, "");
}

static LLVMValueRef get_memory_addr(struct ac_nir_context *ctx, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global_amd:
   case nir_intrinsic_store_global_amd:
      return get_global_address(ctx, intr);
   case nir_intrinsic_load_shared:
   case nir_intrinsic_store_shared: {
      unsigned num_src = nir_intrinsic_infos[intr->intrinsic].num_srcs;
      return get_shared_mem_ptr(ctx, intr->src[num_src - 1], nir_intrinsic_base(intr));
   }
   default:
      UNREACHABLE("unexpected store intrinsic");
   }
}

static void set_mem_op_alignment(LLVMValueRef instr, nir_intrinsic_instr *intr, nir_def *type)
{
   unsigned align = nir_intrinsic_align(intr);
   bool has_dest = nir_intrinsic_infos[intr->intrinsic].has_dest;
   nir_def *val = has_dest ? &intr->def : intr->src[0].ssa;
   unsigned pot_size = (val->bit_size / 8) * (1 << util_logbase2(val->num_components));
   /* The alignment must be at most the stored size. */
   LLVMSetAlignment(instr, MIN2(align, pot_size));
}

static void set_coherent_volatile(LLVMValueRef instr, nir_intrinsic_instr *intr)
{
   if ((intr->intrinsic == nir_intrinsic_load_global ||
        intr->intrinsic == nir_intrinsic_store_global) &&
       nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE))
      LLVMSetOrdering(instr, LLVMAtomicOrderingMonotonic);
}

static void visit_store(struct ac_nir_context *ctx, nir_intrinsic_instr *intr)
{
   LLVMBuilderRef builder = ctx->ac.builder;
   LLVMValueRef data = get_src(ctx, intr->src[0]);
   LLVMValueRef ptr = get_memory_addr(ctx, intr);
   unsigned writemask = nir_intrinsic_write_mask(intr);
   /* nir_opt_shrink_stores should split stores with holes. */
   assert(writemask == BITFIELD_MASK(intr->src[0].ssa->num_components));

   LLVMValueRef store = LLVMBuildStore(builder, data, ptr);
   set_mem_op_alignment(store, intr, intr->src[0].ssa);
   set_coherent_volatile(store, intr);
}

static LLVMValueRef visit_load(struct ac_nir_context *ctx, nir_intrinsic_instr *intr)
{
   LLVMTypeRef result_type = get_def_type(ctx, &intr->def);
   LLVMValueRef ptr = get_memory_addr(ctx, intr);

   LLVMValueRef value = LLVMBuildLoad2(ctx->ac.builder, result_type, ptr, "");
   set_mem_op_alignment(value, intr, &intr->def);
   set_coherent_volatile(value, intr);
   return value;
}

static LLVMValueRef visit_global_atomic(struct ac_nir_context *ctx,
					nir_intrinsic_instr *instr)
{
   LLVMValueRef data = get_src(ctx, instr->src[1]);
   LLVMAtomicRMWBinOp op;
   LLVMValueRef result;

   /* use "singlethread" sync scope to implement relaxed ordering */
   const char *sync_scope = "singlethread-one-as";

   nir_atomic_op nir_op = nir_intrinsic_atomic_op(instr);
   bool is_float = nir_atomic_op_type(nir_op) == nir_type_float;

   LLVMTypeRef data_type = LLVMTypeOf(data);

   assert(instr->src[1].ssa->num_components == 1);
   if (is_float) {
      switch (instr->src[1].ssa->bit_size) {
      case 32:
         data_type = ctx->ac.f32;
         break;
      case 64:
         data_type = ctx->ac.f64;
         break;
      default:
         UNREACHABLE("Unsupported float bit size");
      }

      data = LLVMBuildBitCast(ctx->ac.builder, data, data_type, "");
   }

   LLVMValueRef addr = get_global_address(ctx, instr);

   if (instr->intrinsic == nir_intrinsic_global_atomic_swap_amd) {
      LLVMValueRef data1 = get_src(ctx, instr->src[2]);
      result = ac_build_atomic_cmp_xchg(&ctx->ac, addr, data, data1, sync_scope);
      result = LLVMBuildExtractValue(ctx->ac.builder, result, 0, "");
   } else if (nir_op == nir_atomic_op_ordered_add_gfx12_amd) {
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.global.atomic.ordered.add.b64", ctx->ac.i64,
                                  (LLVMValueRef[]){addr, data}, 2, 0);
   } else if (is_float) {
      const char *op = translate_atomic_op_str(nir_op);
      char name[64], type[8];
      LLVMValueRef params[2];
      int arg_count = 0;

      params[arg_count++] = addr;
      params[arg_count++] = data;

      ac_build_type_name_for_intr(data_type, type, sizeof(type));
      snprintf(name, sizeof(name), "llvm.amdgcn.global.atomic.%s.%s.p1.%s", op, type, type);

      result = ac_build_intrinsic(&ctx->ac, name, data_type, params, arg_count, 0);
   } else {
      op = translate_atomic_op(nir_op);
      result = ac_build_atomic_rmw(&ctx->ac, op, addr, ac_to_integer(&ctx->ac, data), sync_scope);
   }

   result = ac_to_integer(&ctx->ac, result);

   return result;
}

static LLVMValueRef visit_load_ubo_buffer(struct ac_nir_context *ctx, nir_intrinsic_instr *instr)
{
   struct waterfall_context wctx;
   LLVMValueRef rsrc_base = enter_waterfall_ubo(ctx, &wctx, instr);

   LLVMValueRef ret;
   LLVMValueRef rsrc = rsrc_base;
   LLVMValueRef offset = get_src(ctx, instr->src[1]);
   int num_components = instr->num_components;

   assert(instr->def.bit_size >= 32 && instr->def.bit_size % 32 == 0);

   if (ctx->abi->load_ubo)
      rsrc = ctx->abi->load_ubo(ctx->abi, rsrc);

   /* Convert to a 32-bit load. */
   if (instr->def.bit_size == 64)
      num_components *= 2;

   ret = ac_build_buffer_load(&ctx->ac, rsrc, num_components, NULL, offset, NULL,
                              ctx->ac.f32, 0, true, true);
   ret = LLVMBuildBitCast(ctx->ac.builder, ret, get_def_type(ctx, &instr->def), "");

   return exit_waterfall(ctx, &wctx, ret);
}

static void visit_store_output(struct ac_nir_context *ctx, nir_intrinsic_instr *instr)
{
   unsigned base = nir_intrinsic_base(instr);
   unsigned writemask = nir_intrinsic_write_mask(instr);
   unsigned component = nir_intrinsic_component(instr);
   LLVMValueRef src = ac_to_float(&ctx->ac, get_src(ctx, instr->src[0]));
   ASSERTED unsigned bit_size = ac_get_elem_bits(&ctx->ac, LLVMTypeOf(src));
   ASSERTED nir_src offset = *nir_get_io_offset_src(instr);

   /* Non-monolithic PS and also LS before TCS in radeonsi use this to forward outputs to
    * registers.
    */
   assert(bit_size == 16 || bit_size == 32);
   /* No indirect indexing is allowed here. */
   assert(nir_src_is_const(offset) && nir_src_as_uint(offset) == 0);

   writemask <<= component;

   for (unsigned chan = 0; chan < 8; chan++) {
      if (!(writemask & (1 << chan)))
         continue;

      LLVMValueRef value = ac_llvm_extract_elem(&ctx->ac, src, chan - component);
      LLVMTypeRef val_type = LLVMTypeOf(value);
      assert(val_type == ctx->ac.f32 || val_type == ctx->ac.f16);
      LLVMTypeRef output_type = ctx->stage == MESA_SHADER_FRAGMENT ? val_type : ctx->ac.f32;
      LLVMValueRef *output_addr = &ctx->abi->outputs[base * 4 + chan];

      /* Allocate the output variable on demand. */
      if (!*output_addr) {
         *output_addr = ac_build_alloca_undef(&ctx->ac, output_type, "");
         ctx->abi->is_16bit[base * 4 + chan] = output_type == ctx->ac.f16;
      }

      if (val_type == ctx->ac.f16 && output_type == ctx->ac.f32) {
         LLVMValueRef output, index;

         /* Insert the 16-bit value into the low or high bits of the 32-bit output
          * using read-modify-write.
          */
         index = LLVMConstInt(ctx->ac.i32, nir_intrinsic_io_semantics(instr).high_16bits, 0);

         output = LLVMBuildLoad2(ctx->ac.builder, ctx->ac.v2f16, *output_addr, "");
         output = LLVMBuildInsertElement(ctx->ac.builder, output, value, index, "");
         value = LLVMBuildBitCast(ctx->ac.builder, output, ctx->ac.f32, "");
      }
      LLVMBuildStore(ctx->ac.builder, value, *output_addr);
   }
}

static int image_type_to_components_count(enum glsl_sampler_dim dim, bool array)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_BUF:
      return 1;
   case GLSL_SAMPLER_DIM_1D:
      return array ? 2 : 1;
   case GLSL_SAMPLER_DIM_2D:
      return array ? 3 : 2;
   case GLSL_SAMPLER_DIM_MS:
      return array ? 4 : 3;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      return 3;
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_SUBPASS:
      return 2;
   case GLSL_SAMPLER_DIM_SUBPASS_MS:
      return 3;
   default:
      break;
   }
   return 0;
}

static void get_image_coords(struct ac_nir_context *ctx, const nir_intrinsic_instr *instr,
                             LLVMValueRef dynamic_desc_index, struct ac_image_args *args,
                             enum glsl_sampler_dim dim, bool is_array)
{
   LLVMValueRef src0 = get_src(ctx, instr->src[1]);
   int count;
   ASSERTED bool add_frag_pos =
      (dim == GLSL_SAMPLER_DIM_SUBPASS || dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
   bool is_ms = (dim == GLSL_SAMPLER_DIM_MS || dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
   bool gfx9_1d = ctx->ac.gfx_level == GFX9 && dim == GLSL_SAMPLER_DIM_1D;
   assert(!add_frag_pos && "Input attachments should be lowered by this point.");
   count = image_type_to_components_count(dim, is_array);

   if (count == 1 && !gfx9_1d) {
      if (instr->src[1].ssa->num_components)
         args->coords[0] = ac_llvm_extract_elem(&ctx->ac, src0, 0);
      else
         args->coords[0] = src0;
   } else {
      int chan;
      if (is_ms)
         count--;
      for (chan = 0; chan < count; ++chan) {
         args->coords[chan] = ac_llvm_extract_elem(&ctx->ac, src0, chan);
      }

      if (gfx9_1d) {
         if (is_array)
            args->coords[2] = args->coords[1];
         args->coords[1] = LLVMConstInt(LLVMTypeOf(args->coords[0]), 0, 0);
         count++;
      }
      if (ctx->ac.gfx_level == GFX9 && dim == GLSL_SAMPLER_DIM_2D && !is_array) {
         /* The hw can't bind a slice of a 3D image as a 2D
          * image, because it ignores BASE_ARRAY if the target
          * is 3D. The workaround is to read BASE_ARRAY and set
          * it as the 3rd address operand for all 2D images.
          */
         LLVMValueRef first_layer, const5, mask;

         const5 = LLVMConstInt(ctx->ac.i32, 5, 0);
         mask = LLVMConstInt(ctx->ac.i32, S_008F24_BASE_ARRAY(~0), 0);
         first_layer = LLVMBuildExtractElement(ctx->ac.builder, args->resource, const5, "");
         first_layer = LLVMBuildAnd(ctx->ac.builder, first_layer, mask, "");

         if (instr->intrinsic == nir_intrinsic_bindless_image_load ||
             instr->intrinsic == nir_intrinsic_bindless_image_sparse_load ||
             instr->intrinsic == nir_intrinsic_bindless_image_store) {
            int lod_index = instr->intrinsic == nir_intrinsic_bindless_image_store ? 4 : 3;
            bool has_lod = !nir_src_is_const(instr->src[lod_index]) ||
                           nir_src_as_uint(instr->src[lod_index]) != 0;
            if (has_lod) {
               /* If there's a lod parameter it matter if the image is 3d or 2d because
                * the hw reads either the fourth or third component as lod. So detect
                * 3d images and place the lod at the third component otherwise.
                */
               LLVMValueRef const3, const28, const4, rword3, type3d, type, is_3d, lod;
               const3 = LLVMConstInt(ctx->ac.i32, 3, 0);
               const28 = LLVMConstInt(ctx->ac.i32, 28, 0);
               const4 = LLVMConstInt(ctx->ac.i32, 4, 0);
               type3d = LLVMConstInt(ctx->ac.i32, V_008F1C_SQ_RSRC_IMG_3D, 0);
               rword3 = LLVMBuildExtractElement(ctx->ac.builder, args->resource, const3, "");
               type = ac_build_bfe(&ctx->ac, rword3, const28, const4, false);
               is_3d = emit_int_cmp(&ctx->ac, LLVMIntEQ, type, type3d);
               lod = get_src(ctx, instr->src[lod_index]);
               first_layer = emit_bcsel(&ctx->ac, is_3d, first_layer, lod);
            }
         }

         args->coords[count] = LLVMBuildTrunc(ctx->ac.builder, first_layer,
                                              LLVMTypeOf(args->coords[0]), "");
         count++;
      }

      if (is_ms) {
         /* sample index */
         args->coords[count] = ac_llvm_extract_elem(&ctx->ac, get_src(ctx, instr->src[2]), 0);
         count++;
      }
   }
}

static LLVMValueRef enter_waterfall_image(struct ac_nir_context *ctx,
                                          struct waterfall_context *wctx,
                                          const nir_intrinsic_instr *instr)
{
   /* src0 is desc when uniform, desc index when non uniform */
   LLVMValueRef value = get_src(ctx, instr->src[0]);

   return enter_waterfall(ctx, wctx, value, nir_intrinsic_access(instr) & ACCESS_NON_UNIFORM);
}

static LLVMValueRef visit_image_load(struct ac_nir_context *ctx, const nir_intrinsic_instr *instr)
{
   LLVMValueRef res;

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(instr);
   enum gl_access_qualifier access = nir_intrinsic_access(instr);
   bool is_array = nir_intrinsic_image_array(instr);

   struct waterfall_context wctx;
   LLVMValueRef dynamic_index = enter_waterfall_image(ctx, &wctx, instr);

   struct ac_image_args args = {0};

   args.access = ac_nir_get_mem_access_flags(instr);
   args.tfe = instr->intrinsic == nir_intrinsic_bindless_image_sparse_load;

   if (dim == GLSL_SAMPLER_DIM_BUF) {
      unsigned num_channels = util_last_bit(nir_def_components_read(&instr->def));
      if (instr->def.bit_size == 64)
         num_channels = num_channels < 4 ? 2 : 4;
      LLVMValueRef rsrc, vindex;

      rsrc = ctx->abi->load_sampler_desc(ctx->abi, dynamic_index, AC_DESC_BUFFER);
      vindex =
         LLVMBuildExtractElement(ctx->ac.builder, get_src(ctx, instr->src[1]), ctx->ac.i32_0, "");

      bool can_speculate = access & ACCESS_CAN_REORDER;
      res = ac_build_buffer_load_format(&ctx->ac, rsrc, vindex, ctx->ac.i32_0, num_channels,
                                        args.access, can_speculate,
                                        instr->def.bit_size == 16,
                                        args.tfe);
      res = ac_build_expand(&ctx->ac, res, num_channels, args.tfe ? 5 : 4);

      res = ac_trim_vector(&ctx->ac, res, instr->def.num_components);
      res = ac_to_integer(&ctx->ac, res);
   } else if (instr->intrinsic == nir_intrinsic_bindless_image_fragment_mask_load_amd) {
      assert(ctx->ac.gfx_level < GFX11);

      args.opcode = ac_image_load;
      args.resource = ctx->abi->load_sampler_desc(ctx->abi, dynamic_index, AC_DESC_FMASK);
      get_image_coords(ctx, instr, dynamic_index, &args, GLSL_SAMPLER_DIM_2D, is_array);
      args.dmask = 0x1;
      args.dim = is_array ? ac_image_2darray : ac_image_2d;
      args.attributes = AC_ATTR_INVARIANT_LOAD;
      args.a16 = ac_get_elem_bits(&ctx->ac, LLVMTypeOf(args.coords[0])) == 16;

      res = ac_build_image_opcode(&ctx->ac, &args);
   } else {
      bool level_zero = nir_src_is_const(instr->src[3]) && nir_src_as_uint(instr->src[3]) == 0;

      args.opcode = level_zero ? ac_image_load : ac_image_load_mip;
      args.resource = ctx->abi->load_sampler_desc(ctx->abi, dynamic_index, AC_DESC_IMAGE);
      get_image_coords(ctx, instr, dynamic_index, &args, dim, is_array);
      args.dim = ac_get_image_dim(ctx->ac.gfx_level, dim, is_array);
      if (!level_zero)
         args.lod = get_src(ctx, instr->src[3]);
      /* TODO: Fix in LLVM. LLVM doesn't reduce DMASK for D16 if optimization barriers are
       * present and even if the vector is trimmed before the optimization barriers.
       */
      args.dmask = BITFIELD_MASK(instr->def.num_components);
      args.attributes = access & ACCESS_CAN_REORDER ? AC_ATTR_INVARIANT_LOAD : 0;
      args.d16 = instr->def.bit_size == 16;
      args.a16 = ac_get_elem_bits(&ctx->ac, LLVMTypeOf(args.coords[0])) == 16;

      res = ac_build_image_opcode(&ctx->ac, &args);
   }

   if (instr->def.bit_size == 64) {
      LLVMValueRef code = NULL;
      if (args.tfe) {
         code = ac_llvm_extract_elem(&ctx->ac, res, 4);
         res = ac_trim_vector(&ctx->ac, res, 4);
      }

      res = LLVMBuildBitCast(ctx->ac.builder, res, LLVMVectorType(ctx->ac.i64, 2), "");
      LLVMValueRef x = LLVMBuildExtractElement(ctx->ac.builder, res, ctx->ac.i32_0, "");
      LLVMValueRef w = LLVMBuildExtractElement(ctx->ac.builder, res, ctx->ac.i32_1, "");

      if (code)
         code = LLVMBuildZExt(ctx->ac.builder, code, ctx->ac.i64, "");
      LLVMValueRef values[5] = {x, ctx->ac.i64_0, ctx->ac.i64_0, w, code};
      res = ac_build_gather_values(&ctx->ac, values, 4 + args.tfe);
   }

   if (instr->def.num_components < 4)
      res = ac_trim_vector(&ctx->ac, res, instr->def.num_components);

   return exit_waterfall(ctx, &wctx, res);
}

static void visit_image_store(struct ac_nir_context *ctx, const nir_intrinsic_instr *instr)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(instr);
   bool is_array = nir_intrinsic_image_array(instr);

   struct waterfall_context wctx;
   LLVMValueRef dynamic_index = enter_waterfall_image(ctx, &wctx, instr);

   struct ac_image_args args = {0};
   args.access = ac_nir_get_mem_access_flags(instr);

   LLVMValueRef src = get_src(ctx, instr->src[3]);
   if (instr->src[3].ssa->bit_size == 64) {
      /* only R64_UINT and R64_SINT supported */
      src = ac_llvm_extract_elem(&ctx->ac, src, 0);
      src = LLVMBuildBitCast(ctx->ac.builder, src, ctx->ac.v2f32, "");
   } else {
      src = ac_to_float(&ctx->ac, src);
   }

   if (dim == GLSL_SAMPLER_DIM_BUF) {
      LLVMValueRef rsrc = ctx->abi->load_sampler_desc(ctx->abi, dynamic_index, AC_DESC_BUFFER);
      unsigned src_channels = ac_get_llvm_num_components(src);
      LLVMValueRef vindex;

      if (src_channels == 3)
         src = ac_build_expand_to_vec4(&ctx->ac, src, 3);

      vindex =
         LLVMBuildExtractElement(ctx->ac.builder, get_src(ctx, instr->src[1]), ctx->ac.i32_0, "");

      ac_build_buffer_store_format(&ctx->ac, rsrc, src, vindex, ctx->ac.i32_0, args.access);
   } else {
      bool level_zero = nir_src_is_const(instr->src[4]) && nir_src_as_uint(instr->src[4]) == 0;

      args.opcode = level_zero ? ac_image_store : ac_image_store_mip;
      args.data[0] = src;
      args.resource = ctx->abi->load_sampler_desc(ctx->abi, dynamic_index, AC_DESC_IMAGE);
      get_image_coords(ctx, instr, dynamic_index, &args, dim, is_array);
      args.dim = ac_get_image_dim(ctx->ac.gfx_level, dim, is_array);
      if (!level_zero)
         args.lod = get_src(ctx, instr->src[4]);
      args.dmask = 15;
      args.d16 = ac_get_elem_bits(&ctx->ac, LLVMTypeOf(args.data[0])) == 16;
      args.a16 = ac_get_elem_bits(&ctx->ac, LLVMTypeOf(args.coords[0])) == 16;

      ac_build_image_opcode(&ctx->ac, &args);
   }

   exit_waterfall(ctx, &wctx, NULL);
}

static LLVMValueRef visit_image_atomic(struct ac_nir_context *ctx, const nir_intrinsic_instr *instr)
{
   LLVMValueRef params[7];
   int param_count = 0;

   nir_atomic_op op = nir_intrinsic_atomic_op(instr);
   bool cmpswap = op == nir_atomic_op_cmpxchg;
   const char *atomic_name = translate_atomic_op_str(op);
   char intrinsic_name[64];
   enum ac_atomic_op atomic_subop;
   ASSERTED int length;

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(instr);
   bool is_array = nir_intrinsic_image_array(instr);

   struct waterfall_context wctx;
   LLVMValueRef dynamic_index = enter_waterfall_image(ctx, &wctx, instr);

   switch (op) {
   case nir_atomic_op_iadd:
      atomic_subop = ac_atomic_add;
      break;
   case nir_atomic_op_imin:
      atomic_subop = ac_atomic_smin;
      break;
   case nir_atomic_op_umin:
      atomic_subop = ac_atomic_umin;
      break;
   case nir_atomic_op_imax:
      atomic_subop = ac_atomic_smax;
      break;
   case nir_atomic_op_umax:
      atomic_subop = ac_atomic_umax;
      break;
   case nir_atomic_op_iand:
      atomic_subop = ac_atomic_and;
      break;
   case nir_atomic_op_ior:
      atomic_subop = ac_atomic_or;
      break;
   case nir_atomic_op_ixor:
      atomic_subop = ac_atomic_xor;
      break;
   case nir_atomic_op_xchg:
      atomic_subop = ac_atomic_swap;
      break;
   case nir_atomic_op_cmpxchg:
      atomic_subop = 0; /* not used */
      break;
   case nir_atomic_op_inc_wrap:
      atomic_subop = ac_atomic_inc_wrap;
      break;
   case nir_atomic_op_dec_wrap:
      atomic_subop = ac_atomic_dec_wrap;
      break;
   case nir_atomic_op_fadd:
      atomic_subop = ac_atomic_fmin; /* Non-buffer fadd atomics are not supported. */
      break;
   case nir_atomic_op_fmin:
      atomic_subop = ac_atomic_fmin;
      break;
   case nir_atomic_op_fmax:
      atomic_subop = ac_atomic_fmax;
      break;
   default:
      abort();
   }

   if (cmpswap)
      params[param_count++] = get_src(ctx, instr->src[4]);
   params[param_count++] = get_src(ctx, instr->src[3]);

   if (atomic_subop == ac_atomic_fmin || atomic_subop == ac_atomic_fmax)
      params[0] = ac_to_float(&ctx->ac, params[0]);

   LLVMValueRef result;
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      params[param_count++] = ctx->abi->load_sampler_desc(ctx->abi, dynamic_index, AC_DESC_BUFFER);
      params[param_count++] = LLVMBuildExtractElement(ctx->ac.builder, get_src(ctx, instr->src[1]),
                                                      ctx->ac.i32_0, ""); /* vindex */
      params[param_count++] = ctx->ac.i32_0;                              /* voffset */
      if (cmpswap && instr->def.bit_size == 64) {
         result = emit_ssbo_comp_swap_64(ctx, params[2], params[3], params[1], params[0], true);
      } else {
         LLVMTypeRef data_type = LLVMTypeOf(params[0]);
         char type[8];
         unsigned cache_flags =
            ac_get_hw_cache_flags(ctx->ac.gfx_level,
				  ac_nir_get_mem_access_flags(instr) | ACCESS_TYPE_ATOMIC).value;

         params[param_count++] = ctx->ac.i32_0; /* soffset */
         params[param_count++] = LLVMConstInt(ctx->ac.i32, cache_flags, 0);

         ac_build_type_name_for_intr(data_type, type, sizeof(type));
         length = snprintf(intrinsic_name, sizeof(intrinsic_name),
                           "llvm.amdgcn.struct.buffer.atomic.%s.%s",
                           atomic_name, type);

         assert(length < sizeof(intrinsic_name));
         result = ac_build_intrinsic(&ctx->ac, intrinsic_name, LLVMTypeOf(params[0]), params, param_count, 0);
      }
   } else {
      struct ac_image_args args = {0};
      args.opcode = cmpswap ? ac_image_atomic_cmpswap : ac_image_atomic;
      args.atomic = atomic_subop;
      args.data[0] = params[0];
      if (cmpswap)
         args.data[1] = params[1];
      args.resource = ctx->abi->load_sampler_desc(ctx->abi, dynamic_index, AC_DESC_IMAGE);
      get_image_coords(ctx, instr, dynamic_index, &args, dim, is_array);
      args.dim = ac_get_image_dim(ctx->ac.gfx_level, dim, is_array);
      args.a16 = ac_get_elem_bits(&ctx->ac, LLVMTypeOf(args.coords[0])) == 16;
      args.access = ac_nir_get_mem_access_flags(instr);

      result = ac_build_image_opcode(&ctx->ac, &args);
   }

   return exit_waterfall(ctx, &wctx, result);
}

static void emit_discard(struct ac_nir_context *ctx, const nir_intrinsic_instr *instr)
{
   LLVMValueRef cond;

   if (instr->intrinsic == nir_intrinsic_terminate_if) {
      cond = LLVMBuildNot(ctx->ac.builder, get_src(ctx, instr->src[0]), "");
   } else {
      assert(instr->intrinsic == nir_intrinsic_terminate);
      cond = ctx->ac.i1false;
   }

   ac_build_kill_if_false(&ctx->ac, cond);
}

static void emit_demote(struct ac_nir_context *ctx, const nir_intrinsic_instr *instr)
{
   LLVMValueRef cond;

   if (instr->intrinsic == nir_intrinsic_demote_if) {
      cond = LLVMBuildNot(ctx->ac.builder, get_src(ctx, instr->src[0]), "");
   } else {
      assert(instr->intrinsic == nir_intrinsic_demote);
      cond = ctx->ac.i1false;
   }

   /* This demotes the pixel if the condition is false. */
   ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.wqm.demote", ctx->ac.voidt, &cond, 1, 0);
}

static LLVMValueRef visit_first_invocation(struct ac_nir_context *ctx)
{
   LLVMValueRef active_set = ac_build_ballot(&ctx->ac, ctx->ac.i32_1);
   const char *intr = ctx->ac.wave_size == 32 ? "llvm.cttz.i32" : "llvm.cttz.i64";

   /* The second argument is whether cttz(0) should be defined, but we do not care. */
   LLVMValueRef args[] = {active_set, ctx->ac.i1false};
   LLVMValueRef result = ac_build_intrinsic(&ctx->ac, intr, ctx->ac.iN_wavemask, args, 2, 0);

   return LLVMBuildTrunc(ctx->ac.builder, result, ctx->ac.i32, "");
}

static LLVMValueRef visit_load_shared2_amd(struct ac_nir_context *ctx,
                                           const nir_intrinsic_instr *instr)
{
   LLVMTypeRef pointee_type = LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size);
   LLVMValueRef ptr = get_shared_mem_ptr(ctx, instr->src[0], 0);

   LLVMValueRef values[2];
   uint8_t offsets[] = {nir_intrinsic_offset0(instr), nir_intrinsic_offset1(instr)};
   unsigned stride = nir_intrinsic_st64(instr) ? 64 : 1;
   for (unsigned i = 0; i < 2; i++) {
      LLVMValueRef index = LLVMConstInt(ctx->ac.i32, offsets[i] * stride, 0);
      LLVMValueRef derived_ptr = LLVMBuildGEP2(ctx->ac.builder, pointee_type, ptr, &index, 1, "");
      values[i] = LLVMBuildLoad2(ctx->ac.builder, pointee_type, derived_ptr, "");
   }

   LLVMValueRef ret = ac_build_gather_values(&ctx->ac, values, 2);
   return LLVMBuildBitCast(ctx->ac.builder, ret, get_def_type(ctx, &instr->def), "");
}

static void visit_store_shared2_amd(struct ac_nir_context *ctx, const nir_intrinsic_instr *instr)
{
   LLVMTypeRef pointee_type = LLVMIntTypeInContext(ctx->ac.context, instr->src[0].ssa->bit_size);
   LLVMValueRef ptr = get_shared_mem_ptr(ctx, instr->src[1], 0);
   LLVMValueRef src = get_src(ctx, instr->src[0]);

   uint8_t offsets[] = {nir_intrinsic_offset0(instr), nir_intrinsic_offset1(instr)};
   unsigned stride = nir_intrinsic_st64(instr) ? 64 : 1;
   for (unsigned i = 0; i < 2; i++) {
      LLVMValueRef index = LLVMConstInt(ctx->ac.i32, offsets[i] * stride, 0);
      LLVMValueRef derived_ptr = LLVMBuildGEP2(ctx->ac.builder, pointee_type, ptr, &index, 1, "");
      LLVMBuildStore(ctx->ac.builder, ac_llvm_extract_elem(&ctx->ac, src, i), derived_ptr);
   }
}

static LLVMValueRef visit_var_atomic(struct ac_nir_context *ctx, const nir_intrinsic_instr *instr,
                                     LLVMValueRef ptr, int src_idx)
{
   LLVMValueRef result;
   LLVMValueRef src = get_src(ctx, instr->src[src_idx]);
   nir_atomic_op nir_op = nir_intrinsic_atomic_op(instr);

   const char *sync_scope = "workgroup-one-as";

   if (nir_op == nir_atomic_op_cmpxchg) {
      LLVMValueRef src1 = get_src(ctx, instr->src[src_idx + 1]);
      result = ac_build_atomic_cmp_xchg(&ctx->ac, ptr, src, src1, sync_scope);
      result = LLVMBuildExtractValue(ctx->ac.builder, result, 0, "");
   } else if (nir_op == nir_atomic_op_fmin || nir_op == nir_atomic_op_fmax) {
      const char *op = translate_atomic_op_str(nir_op);
      char name[64], type[8];
      LLVMValueRef params[5];
      LLVMTypeRef src_type;
      int arg_count = 0;

      src = ac_to_float(&ctx->ac, src);
      src_type = LLVMTypeOf(src);

      params[arg_count++] = ptr;
      params[arg_count++] = src;
      params[arg_count++] = ctx->ac.i32_0;
      params[arg_count++] = ctx->ac.i32_0;
      params[arg_count++] = ctx->ac.i1false;

      ac_build_type_name_for_intr(src_type, type, sizeof(type));
      snprintf(name, sizeof(name), "llvm.amdgcn.ds.%s.%s", op, type);

      result = ac_build_intrinsic(&ctx->ac, name, src_type, params, arg_count, 0);
      result = ac_to_integer(&ctx->ac, result);
   } else {
      LLVMAtomicRMWBinOp op = translate_atomic_op(nir_op);
      LLVMValueRef val;

      if (nir_op == nir_atomic_op_fadd) {
         val = ac_to_float(&ctx->ac, src);
      } else {
         val = ac_to_integer(&ctx->ac, src);
      }

      result = ac_build_atomic_rmw(&ctx->ac, op, ptr, val, sync_scope);

      if (nir_op == nir_atomic_op_fadd) {
         result = ac_to_integer(&ctx->ac, result);
      }
   }

   return result;
}

static LLVMValueRef load_interpolated_input(struct ac_nir_context *ctx, LLVMValueRef interp_param,
                                            unsigned index, unsigned comp_start,
                                            unsigned num_components, unsigned bitsize,
                                            bool high_16bits)
{
   LLVMValueRef attr_number = LLVMConstInt(ctx->ac.i32, index, false);
   LLVMValueRef interp_param_f;

   interp_param_f = LLVMBuildBitCast(ctx->ac.builder, interp_param, ctx->ac.v2f32, "");
   LLVMValueRef i = LLVMBuildExtractElement(ctx->ac.builder, interp_param_f, ctx->ac.i32_0, "");
   LLVMValueRef j = LLVMBuildExtractElement(ctx->ac.builder, interp_param_f, ctx->ac.i32_1, "");

   /* Workaround for issue 2647: kill threads with infinite interpolation coeffs */
   if (ctx->verified_interp && !_mesa_hash_table_search(ctx->verified_interp, interp_param)) {
      LLVMValueRef cond = ac_build_is_inf_or_nan(&ctx->ac, i);
      ac_build_kill_if_false(&ctx->ac, LLVMBuildNot(ctx->ac.builder, cond, ""));
      _mesa_hash_table_insert(ctx->verified_interp, interp_param, interp_param);
   }

   LLVMValueRef values[4];
   assert(bitsize == 16 || bitsize == 32);
   for (unsigned comp = 0; comp < num_components; comp++) {
      LLVMValueRef llvm_chan = LLVMConstInt(ctx->ac.i32, comp_start + comp, false);
      if (bitsize == 16) {
         values[comp] = ac_build_fs_interp_f16(&ctx->ac, llvm_chan, attr_number,
                                               ac_get_arg(&ctx->ac, ctx->args->prim_mask), i, j,
                                               high_16bits);
      } else {
         values[comp] = ac_build_fs_interp(&ctx->ac, llvm_chan, attr_number,
                                           ac_get_arg(&ctx->ac, ctx->args->prim_mask), i, j);
      }
   }

   return ac_to_integer(&ctx->ac, ac_build_gather_values(&ctx->ac, values, num_components));
}

static LLVMValueRef visit_load_input(struct ac_nir_context *ctx, nir_intrinsic_instr *instr)
{
   LLVMValueRef values[8];
   LLVMTypeRef dest_type = get_def_type(ctx, &instr->def);
   unsigned base = nir_intrinsic_base(instr);
   unsigned component = nir_intrinsic_component(instr);
   unsigned count = instr->def.num_components;
   nir_src offset = *nir_get_io_offset_src(instr);

   assert(instr->def.bit_size == 16 || instr->def.bit_size == 32);
   /* No indirect indexing allowed. */
   assert(nir_src_is_const(offset) && nir_src_as_uint(offset) == 0);

   /* This is used to load TCS inputs from VGPRs in radeonsi. */
   if (ctx->stage == MESA_SHADER_TESS_CTRL) {
      LLVMTypeRef component_type = LLVMGetTypeKind(dest_type) == LLVMVectorTypeKind ?
                                      LLVMGetElementType(dest_type) : dest_type;

      LLVMValueRef result = ctx->abi->load_tess_varyings(ctx->abi, component_type,
                                                         base, component, count);
      if (instr->def.bit_size == 16) {
         result = ac_to_integer(&ctx->ac, result);
         result = LLVMBuildTrunc(ctx->ac.builder, result, dest_type, "");
      }
      return LLVMBuildBitCast(ctx->ac.builder, result, dest_type, "");
   }

   assert(ctx->stage == MESA_SHADER_FRAGMENT);
   unsigned vertex_id = 0; /* P0 */

   if (instr->intrinsic == nir_intrinsic_load_input_vertex)
      vertex_id = nir_src_as_uint(instr->src[0]);

   LLVMValueRef attr_number = LLVMConstInt(ctx->ac.i32, base, false);

   for (unsigned chan = 0; chan < count; chan++) {
      LLVMValueRef llvm_chan = LLVMConstInt(ctx->ac.i32, (component + chan) % 4, false);
      values[chan] = ac_build_fs_interp_mov(&ctx->ac, vertex_id, llvm_chan, attr_number,
                                            ac_get_arg(&ctx->ac, ctx->args->prim_mask));
      values[chan] = LLVMBuildBitCast(ctx->ac.builder, values[chan], ctx->ac.i32, "");
      if (instr->def.bit_size == 16 &&
          nir_intrinsic_io_semantics(instr).high_16bits)
         values[chan] = LLVMBuildLShr(ctx->ac.builder, values[chan], LLVMConstInt(ctx->ac.i32, 16, 0), "");
      values[chan] =
         LLVMBuildTruncOrBitCast(ctx->ac.builder, values[chan],
                                 instr->def.bit_size == 16 ? ctx->ac.i16 : ctx->ac.i32, "");
   }

   LLVMValueRef result = ac_build_gather_values(&ctx->ac, values, count);
   return LLVMBuildBitCast(ctx->ac.builder, result, dest_type, "");
}

static bool visit_intrinsic(struct ac_nir_context *ctx, nir_intrinsic_instr *instr)
{
   LLVMValueRef result = NULL;

   switch (instr->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
      result = emit_ddxy(ctx, instr->intrinsic, get_src(ctx, instr->src[0]));
      result = ac_to_integer(&ctx->ac, result);
      break;
   case nir_intrinsic_ballot:
   case nir_intrinsic_ballot_relaxed:
      result = ac_build_ballot(&ctx->ac, get_src(ctx, instr->src[0]));
      if (instr->def.bit_size > ctx->ac.wave_size) {
         LLVMTypeRef dest_type = LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size);
         result = LLVMBuildZExt(ctx->ac.builder, result, dest_type, "");
      }
      break;
   case nir_intrinsic_inverse_ballot: {
      LLVMValueRef src = get_src(ctx, instr->src[0]);
      if (instr->src[0].ssa->bit_size > ctx->ac.wave_size) {
         LLVMTypeRef src_type = LLVMIntTypeInContext(ctx->ac.context, ctx->ac.wave_size);
         src = LLVMBuildTrunc(ctx->ac.builder, src, src_type, "");
      }
      const char *intr_name = ctx->ac.wave_size == 64 ? "llvm.amdgcn.inverse.ballot.i64" : "llvm.amdgcn.inverse.ballot.i32";
      result = ac_build_intrinsic(&ctx->ac, intr_name, ctx->ac.i1, &src, 1, 0);
      break;
   }
   case nir_intrinsic_read_invocation:
      result =
         ac_build_readlane(&ctx->ac, get_src(ctx, instr->src[0]), get_src(ctx, instr->src[1]));
      break;
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_as_uniform:
      result = ac_build_readlane(&ctx->ac, get_src(ctx, instr->src[0]), NULL);
      break;
   case nir_intrinsic_load_workgroup_id: {
      LLVMValueRef values[3] = {ctx->ac.i32_0, ctx->ac.i32_0, ctx->ac.i32_0};

      for (int i = 0; i < 3; i++) {
         if (ctx->args->workgroup_ids[i].used) {
            if (ctx->ac.gfx_level >= GFX12) {
               char intr_name[256];
               snprintf(intr_name, sizeof(intr_name), "llvm.amdgcn.workgroup.id.%c", "xyz"[i]);
               values[i] = ac_build_intrinsic(&ctx->ac, intr_name, ctx->ac.i32, NULL, 0, 0);
            } else {
               values[i] = ac_get_arg(&ctx->ac, ctx->args->workgroup_ids[i]);
            }
         }
      }
      result = ac_build_gather_values(&ctx->ac, values, 3);
      break;
   }
   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_is_helper_invocation:
      result = ac_build_load_helper_invocation(&ctx->ac);
      break;
   case nir_intrinsic_load_num_workgroups:
      if (ctx->abi->load_grid_size_from_user_sgpr) {
         result = ac_get_arg(&ctx->ac, ctx->args->num_work_groups);
      } else {
         result = ac_build_load_invariant(&ctx->ac,
            ac_get_ptr_arg(&ctx->ac, ctx->args, ctx->args->num_work_groups), ctx->ac.i32_0);
      }
      break;
   case nir_intrinsic_load_subgroup_id:
      assert(gl_shader_stage_is_compute(ctx->stage) && ctx->ac.gfx_level >= GFX12);
      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.wave.id", ctx->ac.i32, NULL, 0, 0);
      break;
   case nir_intrinsic_first_invocation:
      result = visit_first_invocation(ctx);
      break;
   case nir_intrinsic_store_ssbo:
      visit_store_ssbo(ctx, instr);
      break;
   case nir_intrinsic_load_ssbo:
      result = visit_load_buffer(ctx, instr);
      break;
   case nir_intrinsic_load_global_amd:
   case nir_intrinsic_load_shared:
      result = visit_load(ctx, instr);
      break;
   case nir_intrinsic_store_global_amd:
   case nir_intrinsic_store_shared:
      visit_store(ctx, instr);
      break;
   case nir_intrinsic_global_atomic_amd:
   case nir_intrinsic_global_atomic_swap_amd:
      result = visit_global_atomic(ctx, instr);
      break;
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      result = visit_atomic_ssbo(ctx, instr);
      break;
   case nir_intrinsic_load_ubo:
      result = visit_load_ubo_buffer(ctx, instr);
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_primitive_input:
   case nir_intrinsic_load_input_vertex:
   case nir_intrinsic_load_per_vertex_input:
      result = visit_load_input(ctx, instr);
      break;
   case nir_intrinsic_store_output:
      visit_store_output(ctx, instr);
      break;
   case nir_intrinsic_load_shared2_amd:
      result = visit_load_shared2_amd(ctx, instr);
      break;
   case nir_intrinsic_store_shared2_amd:
      visit_store_shared2_amd(ctx, instr);
      break;
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_fragment_mask_load_amd:
      result = visit_image_load(ctx, instr);
      break;
   case nir_intrinsic_bindless_image_store:
      visit_image_store(ctx, instr);
      break;
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
      result = visit_image_atomic(ctx, instr);
      break;
   case nir_intrinsic_shader_clock:
      result = ac_build_shader_clock(&ctx->ac, nir_intrinsic_memory_scope(instr));
      break;
   case nir_intrinsic_terminate:
   case nir_intrinsic_terminate_if:
      emit_discard(ctx, instr);
      break;
   case nir_intrinsic_demote:
   case nir_intrinsic_demote_if:
      emit_demote(ctx, instr);
      break;
   case nir_intrinsic_barrier: {
      assert(!(nir_intrinsic_memory_semantics(instr) &
               (NIR_MEMORY_MAKE_AVAILABLE | NIR_MEMORY_MAKE_VISIBLE)));

      nir_variable_mode modes = nir_intrinsic_memory_modes(instr);

      unsigned wait_flags = 0;
      if (modes & (nir_var_mem_global | nir_var_mem_ssbo | nir_var_image))
         wait_flags |= AC_WAIT_LOAD | AC_WAIT_STORE;
      if (modes & nir_var_mem_shared)
         wait_flags |= AC_WAIT_DS;

      if (wait_flags)
         ac_build_waitcnt(&ctx->ac, wait_flags);

      if (nir_intrinsic_execution_scope(instr) == SCOPE_WORKGROUP)
         ac_build_s_barrier(&ctx->ac, ctx->stage);
      break;
   }
   case nir_intrinsic_optimization_barrier_vgpr_amd:
      result = get_src(ctx, instr->src[0]);
      ac_build_optimization_barrier(&ctx->ac, &result, false);
      break;
   case nir_intrinsic_optimization_barrier_sgpr_amd:
      result = get_src(ctx, instr->src[0]);
      ac_build_optimization_barrier(&ctx->ac, &result, true);
      break;
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap: {
      LLVMValueRef ptr = get_shared_mem_ptr(ctx, instr->src[0], nir_intrinsic_base(instr));
      result = visit_var_atomic(ctx, instr, ptr, 1);
      break;
   }
   case nir_intrinsic_load_interpolated_input: {
      /* We assume any indirect loads have been lowered away */
      ASSERTED nir_const_value *offset = nir_src_as_const_value(instr->src[1]);
      assert(offset);
      assert(offset[0].i32 == 0);

      LLVMValueRef interp_param = get_src(ctx, instr->src[0]);
      unsigned index = nir_intrinsic_base(instr);
      unsigned component = nir_intrinsic_component(instr);
      result = load_interpolated_input(ctx, interp_param, index, component,
                                       instr->def.num_components, instr->def.bit_size,
                                       nir_intrinsic_io_semantics(instr).high_16bits);
      break;
   }
   case nir_intrinsic_sendmsg_amd: {
      unsigned imm = nir_intrinsic_base(instr);
      LLVMValueRef m0_content = get_src(ctx, instr->src[0]);
      ac_build_sendmsg(&ctx->ac, imm, m0_content);
      break;
   }
   case nir_intrinsic_vote_all: {
      result = ac_build_vote_all(&ctx->ac, get_src(ctx, instr->src[0]));
      if (ctx->info->stage == MESA_SHADER_FRAGMENT)
         result = ac_build_wqm(&ctx->ac, result);
      break;
   }
   case nir_intrinsic_vote_any: {
      result = ac_build_vote_any(&ctx->ac, get_src(ctx, instr->src[0]));
      if (ctx->info->stage == MESA_SHADER_FRAGMENT)
         result = ac_build_wqm(&ctx->ac, result);
      break;
   }
   case nir_intrinsic_quad_vote_any: {
      result = ac_build_wqm_vote(&ctx->ac, get_src(ctx, instr->src[0]));
      break;
   }
   case nir_intrinsic_quad_vote_all: {
      LLVMValueRef src = LLVMBuildNot(ctx->ac.builder, get_src(ctx, instr->src[0]), "");
      result = LLVMBuildNot(ctx->ac.builder, ac_build_wqm_vote(&ctx->ac, src), "");
      break;
   }
   case nir_intrinsic_shuffle:
      if (ctx->ac.gfx_level == GFX8 || ctx->ac.gfx_level == GFX9 || ctx->ac.gfx_level == GFX12 ||
          (ctx->ac.gfx_level >= GFX10 && ctx->ac.wave_size == 32)) {
         result =
            ac_build_shuffle(&ctx->ac, get_src(ctx, instr->src[0]), get_src(ctx, instr->src[1]));
      } else {
         LLVMValueRef src = get_src(ctx, instr->src[0]);
         LLVMValueRef index = get_src(ctx, instr->src[1]);
         LLVMValueRef active = ac_build_ballot(&ctx->ac, ctx->ac.i32_1);
         LLVMValueRef undef = LLVMGetUndef(LLVMTypeOf(src));

         LLVMBasicBlockRef preheader = LLVMGetInsertBlock(ctx->ac.builder);
         ac_build_bgnloop(&ctx->ac, 7000);
         LLVMValueRef active_phi = ac_build_phi(&ctx->ac, LLVMTypeOf(active), 1, &active, &preheader);
         LLVMValueRef result_phi = ac_build_phi(&ctx->ac, LLVMTypeOf(src), 1, &undef, &preheader);

         LLVMValueRef args[] = {active_phi, ctx->ac.i1true};
         LLVMValueRef first = ac_build_intrinsic(&ctx->ac, "llvm.cttz.i64", ctx->ac.i64, args, 2, 0);
         first = LLVMBuildTrunc(ctx->ac.builder, first, ctx->ac.i32, "");

         LLVMValueRef scalar_index = ac_build_readlane_no_opt_barrier(&ctx->ac, index, first);
         LLVMValueRef scalar_result = ac_build_readlane_no_opt_barrier(&ctx->ac, src, scalar_index);
         LLVMValueRef equal = LLVMBuildICmp(ctx->ac.builder, LLVMIntEQ, index, scalar_index, "");

         result = LLVMBuildSelect(ctx->ac.builder, equal, scalar_result, result_phi, "");
         LLVMValueRef remove = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.ballot.i64", ctx->ac.i64, &equal, 1, 0);
         active = LLVMBuildXor(ctx->ac.builder, active_phi, remove, "");

         ac_build_ifcc(&ctx->ac, LLVMBuildICmp(ctx->ac.builder, LLVMIntEQ, active, ctx->ac.i64_0, ""), 7001);
         ac_build_break(&ctx->ac);
         ac_build_endif(&ctx->ac, 7001);

         LLVMBasicBlockRef cont = LLVMGetInsertBlock(ctx->ac.builder);
         LLVMAddIncoming(active_phi, &active, &cont, 1);
         LLVMAddIncoming(result_phi, &result, &cont, 1);
         ac_build_endloop(&ctx->ac, 7000);
      }
      break;
   case nir_intrinsic_reduce:
      result = ac_build_reduce(&ctx->ac, get_src(ctx, instr->src[0]), instr->const_index[0],
                               instr->const_index[1]);
      result = ac_to_integer(&ctx->ac, result);
      break;
   case nir_intrinsic_inclusive_scan:
      result =
         ac_build_inclusive_scan(&ctx->ac, get_src(ctx, instr->src[0]), instr->const_index[0]);
      result = ac_to_integer(&ctx->ac, result);
      break;
   case nir_intrinsic_exclusive_scan:
      result =
         ac_build_exclusive_scan(&ctx->ac, get_src(ctx, instr->src[0]), instr->const_index[0]);
      result = ac_to_integer(&ctx->ac, result);
      break;
   case nir_intrinsic_quad_broadcast: {
      unsigned lane = nir_src_as_uint(instr->src[1]);
      result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), lane, lane, lane, lane, false);
      if (ctx->info->stage == MESA_SHADER_FRAGMENT)
         result = ac_build_wqm(&ctx->ac, result);
      break;
   }
   case nir_intrinsic_quad_swap_horizontal:
      result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), 1, 0, 3, 2, false);
      if (ctx->info->stage == MESA_SHADER_FRAGMENT)
         result = ac_build_wqm(&ctx->ac, result);
      break;
   case nir_intrinsic_quad_swap_vertical:
      result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), 2, 3, 0, 1, false);
      if (ctx->info->stage == MESA_SHADER_FRAGMENT)
         result = ac_build_wqm(&ctx->ac, result);
      break;
   case nir_intrinsic_quad_swap_diagonal:
      result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), 3, 2, 1, 0, false);
      if (ctx->info->stage == MESA_SHADER_FRAGMENT)
         result = ac_build_wqm(&ctx->ac, result);
      break;
   case nir_intrinsic_quad_swizzle_amd: {
      uint32_t mask = nir_intrinsic_swizzle_mask(instr);
      result = ac_build_quad_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), mask & 0x3,
                                     (mask >> 2) & 0x3, (mask >> 4) & 0x3, (mask >> 6) & 0x3, false);
      if (ctx->info->stage == MESA_SHADER_FRAGMENT)
         result = ac_build_wqm(&ctx->ac, result);
      break;
   }
   case nir_intrinsic_masked_swizzle_amd: {
      uint32_t mask = nir_intrinsic_swizzle_mask(instr);
      result = ac_build_ds_swizzle(&ctx->ac, get_src(ctx, instr->src[0]), mask);
      break;
   }
   case nir_intrinsic_write_invocation_amd:
      result = ac_build_writelane(&ctx->ac, get_src(ctx, instr->src[0]),
                                  get_src(ctx, instr->src[1]), get_src(ctx, instr->src[2]));
      break;
   case nir_intrinsic_mbcnt_amd:
      result = ac_build_mbcnt_add(&ctx->ac, get_src(ctx, instr->src[0]), get_src(ctx, instr->src[1]));
      break;
   case nir_intrinsic_load_scratch: {
      LLVMValueRef offset = get_src(ctx, instr->src[0]);
      LLVMValueRef ptr = ac_build_gep0(&ctx->ac, ctx->scratch, offset);
      LLVMTypeRef comp_type = LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size);
      LLVMTypeRef vec_type = instr->def.num_components == 1
                                ? comp_type
                                : LLVMVectorType(comp_type, instr->def.num_components);
      result = LLVMBuildLoad2(ctx->ac.builder, vec_type, ptr, "");
      break;
   }
   case nir_intrinsic_store_scratch: {
      LLVMValueRef offset = get_src(ctx, instr->src[1]);
      LLVMValueRef ptr = ac_build_gep0(&ctx->ac, ctx->scratch, offset);
      LLVMTypeRef comp_type = LLVMIntTypeInContext(ctx->ac.context, instr->src[0].ssa->bit_size);
      LLVMValueRef src = get_src(ctx, instr->src[0]);
      unsigned wrmask = nir_intrinsic_write_mask(instr);
      while (wrmask) {
         int start, count;
         u_bit_scan_consecutive_range(&wrmask, &start, &count);

         LLVMValueRef offset = LLVMConstInt(ctx->ac.i32, start, false);
         LLVMValueRef offset_ptr = LLVMBuildGEP2(ctx->ac.builder, comp_type, ptr, &offset, 1, "");
         LLVMValueRef offset_src = ac_extract_components(&ctx->ac, src, start, count);
         LLVMBuildStore(ctx->ac.builder, offset_src, offset_ptr);
      }
      break;
   }
   case nir_intrinsic_load_constant: {
      unsigned base = nir_intrinsic_base(instr);
      unsigned range = nir_intrinsic_range(instr);

      LLVMValueRef offset = get_src(ctx, instr->src[0]);
      offset = LLVMBuildAdd(ctx->ac.builder, offset, LLVMConstInt(ctx->ac.i32, base, false), "");

      /* Clamp the offset to avoid out-of-bound access because global
       * instructions can't handle them.
       */
      LLVMValueRef size = LLVMConstInt(ctx->ac.i32, base + range, false);
      LLVMValueRef cond = LLVMBuildICmp(ctx->ac.builder, LLVMIntULT, offset, size, "");
      offset = LLVMBuildSelect(ctx->ac.builder, cond, offset, size, "");

      LLVMValueRef ptr = ac_build_gep0(&ctx->ac, ctx->constant_data, offset);

      /* TODO: LLVM doesn't sign-extend the result of s_getpc_b64 correctly, causing hangs.
       * Do it manually here.
       */
      if (ctx->ac.gfx_level == GFX12) {
         LLVMValueRef addr = LLVMBuildPtrToInt(ctx->ac.builder, ptr, ctx->ac.i64, "");
         addr = LLVMBuildOr(ctx->ac.builder, addr,
                            LLVMConstInt(ctx->ac.i64, 0xffff000000000000ull, 0), "");
         ptr = LLVMBuildIntToPtr(ctx->ac.builder, addr, LLVMTypeOf(ptr), "");
      }

      LLVMTypeRef comp_type = LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size);
      LLVMTypeRef vec_type = instr->def.num_components == 1
                                ? comp_type
                                : LLVMVectorType(comp_type, instr->def.num_components);
      result = LLVMBuildLoad2(ctx->ac.builder, vec_type, ptr, "");
      break;
   }
   case nir_intrinsic_set_vertex_and_primitive_count:
      /* Currently ignored. */
      break;
   case nir_intrinsic_load_typed_buffer_amd:
   case nir_intrinsic_load_buffer_amd:
   case nir_intrinsic_store_buffer_amd: {
      unsigned src_base = instr->intrinsic == nir_intrinsic_store_buffer_amd ? 1 : 0;
      bool idxen = !nir_src_is_const(instr->src[src_base + 3]) ||
                   nir_src_as_uint(instr->src[src_base + 3]);

      LLVMValueRef store_data = get_src(ctx, instr->src[0]);
      LLVMValueRef descriptor = get_src(ctx, instr->src[src_base + 0]);
      LLVMValueRef addr_voffset = get_src(ctx, instr->src[src_base + 1]);
      LLVMValueRef addr_soffset = get_src(ctx, instr->src[src_base + 2]);
      LLVMValueRef vidx = idxen ? get_src(ctx, instr->src[src_base + 3]) : NULL;
      unsigned num_components = instr->def.num_components;
      unsigned const_offset = nir_intrinsic_base(instr);
      bool reorder = nir_intrinsic_can_reorder(instr);
      enum gl_access_qualifier access = ac_nir_get_mem_access_flags(instr);
      bool uses_format = access & ACCESS_USES_FORMAT_AMD;

      LLVMValueRef voffset = LLVMBuildAdd(ctx->ac.builder, addr_voffset,
                                          LLVMConstInt(ctx->ac.i32, const_offset, 0), "");

      if (instr->intrinsic == nir_intrinsic_load_buffer_amd && uses_format) {
         assert(instr->def.bit_size == 16 || instr->def.bit_size == 32);
         result = ac_build_buffer_load_format(&ctx->ac, descriptor, vidx, voffset, num_components,
                                              access, reorder,
                                              instr->def.bit_size == 16, false);
         result = ac_to_integer(&ctx->ac, result);
      } else if (instr->intrinsic == nir_intrinsic_store_buffer_amd && uses_format) {
         assert(instr->src[0].ssa->bit_size == 16 || instr->src[0].ssa->bit_size == 32);
         ac_build_buffer_store_format(&ctx->ac, descriptor, store_data, vidx, voffset, access);
      } else if (instr->intrinsic == nir_intrinsic_load_buffer_amd ||
                 instr->intrinsic == nir_intrinsic_load_typed_buffer_amd) {
         /* LLVM is unable to select instructions for larger than 32-bit channel types.
          * Workaround by using i32 and casting to the correct type later.
          */
         const unsigned fetch_num_components =
            num_components * MAX2(32, instr->def.bit_size) / 32;

         LLVMTypeRef channel_type =
            LLVMIntTypeInContext(ctx->ac.context, MIN2(32, instr->def.bit_size));

         if (instr->intrinsic == nir_intrinsic_load_buffer_amd) {
            result = ac_build_buffer_load(&ctx->ac, descriptor, fetch_num_components, vidx, voffset,
                                          addr_soffset, channel_type, access, reorder, false);
         } else {
            const unsigned align_offset = nir_intrinsic_align_offset(instr);
            const unsigned align_mul = nir_intrinsic_align_mul(instr);
            const enum pipe_format format = nir_intrinsic_format(instr);

            result =
               ac_build_safe_tbuffer_load(&ctx->ac, descriptor, vidx, addr_voffset, addr_soffset,
                                          format, MIN2(32, instr->def.bit_size), const_offset, align_offset,
                                          align_mul, fetch_num_components, access, reorder);
         }

         /* Trim to needed vector components. */
         result = ac_trim_vector(&ctx->ac, result, fetch_num_components);

         /* Cast to larger than 32-bit sized components if needed. */
         if (instr->def.bit_size > 32) {
            LLVMTypeRef cast_channel_type =
               LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size);
            LLVMTypeRef cast_type =
               num_components == 1 ? cast_channel_type :
               LLVMVectorType(cast_channel_type, num_components);
            result = LLVMBuildBitCast(ctx->ac.builder, result, cast_type, "");
         }

         /* Cast the result to an integer (or vector of integers). */
         result = ac_to_integer(&ctx->ac, result);
      } else {
         unsigned writemask = nir_intrinsic_write_mask(instr);
         while (writemask) {
            int start, count;
            u_bit_scan_consecutive_range(&writemask, &start, &count);

            LLVMValueRef voffset = LLVMBuildAdd(
               ctx->ac.builder, addr_voffset,
               LLVMConstInt(ctx->ac.i32, const_offset + start * 4, 0), "");

            LLVMValueRef data = extract_vector_range(&ctx->ac, store_data, start, count);
            ac_build_buffer_store_dword(&ctx->ac, descriptor, data, vidx, voffset, addr_soffset,
                                        access);
         }
      }
      break;
   }
   case nir_intrinsic_is_subgroup_invocation_lt_amd: {
      unsigned offset = nir_intrinsic_base(instr);
      LLVMValueRef count = get_src(ctx, instr->src[0]);
      if (offset)
         count = LLVMBuildLShr(ctx->ac.builder, count, LLVMConstInt(ctx->ac.i32, offset, 0), "");
      count = LLVMBuildAnd(ctx->ac.builder, count, LLVMConstInt(ctx->ac.i32, 0xff, 0), "");
      result = LLVMBuildICmp(ctx->ac.builder, LLVMIntULT, ac_get_thread_id(&ctx->ac), count, "");
      break;
   }
   case nir_intrinsic_gds_atomic_add_amd: {
      LLVMValueRef store_val = get_src(ctx, instr->src[0]);
      LLVMValueRef addr = get_src(ctx, instr->src[1]);
      LLVMTypeRef gds_ptr_type = LLVMPointerTypeInContext(ctx->ac.context, AC_ADDR_SPACE_GDS);
      LLVMValueRef gds_base = LLVMBuildIntToPtr(ctx->ac.builder, addr, gds_ptr_type, "");
      ac_build_atomic_rmw(&ctx->ac, LLVMAtomicRMWBinOpAdd, gds_base, store_val, "workgroup-one-as");
      break;
   }
   case nir_intrinsic_elect:
      result = LLVMBuildICmp(ctx->ac.builder, LLVMIntEQ, visit_first_invocation(ctx),
                             ac_get_thread_id(&ctx->ac), "");
      break;
   case nir_intrinsic_lane_permute_16_amd: {
      const char *intr_name = LLVM_VERSION_MAJOR >= 19 ? "llvm.amdgcn.permlane16.i32" : "llvm.amdgcn.permlane16";
      result = ac_build_intrinsic(&ctx->ac, intr_name, ctx->ac.i32,
                                  (LLVMValueRef[]){get_src(ctx, instr->src[0]),
                                                   get_src(ctx, instr->src[0]),
                                                   get_src(ctx, instr->src[1]),
                                                   get_src(ctx, instr->src[2]),
                                                   ctx->ac.i1false,
                                                   ctx->ac.i1false}, 6, 0);
      break;
   }
   case nir_intrinsic_load_scalar_arg_amd:
   case nir_intrinsic_load_vector_arg_amd: {
      assert(nir_intrinsic_base(instr) < AC_MAX_ARGS);
      struct ac_arg arg;
      arg.arg_index = nir_intrinsic_base(instr);
      arg.used = true;
      result = ac_to_integer(&ctx->ac, ac_get_arg(&ctx->ac, arg));
      if (ac_get_elem_bits(&ctx->ac, LLVMTypeOf(result)) != 32)
         result = LLVMBuildBitCast(ctx->ac.builder, result, get_def_type(ctx, &instr->def), "");
      break;
   }
   case nir_intrinsic_load_smem_amd: {
      LLVMValueRef base = get_src(ctx, instr->src[0]);
      LLVMValueRef offset = get_src(ctx, instr->src[1]);

      bool is_addr_32bit = nir_src_bit_size(instr->src[0]) == 32;
      int addr_space = is_addr_32bit ? AC_ADDR_SPACE_CONST_32BIT : AC_ADDR_SPACE_CONST;

      LLVMTypeRef result_type = get_def_type(ctx, &instr->def);
      LLVMValueRef addr = LLVMBuildIntToPtr(ctx->ac.builder, base,
                                            LLVMPointerTypeInContext(ctx->ac.context, addr_space), "");
      /* see ac_build_load_custom() for 32bit/64bit addr GEP difference */
      addr = is_addr_32bit ?
         LLVMBuildInBoundsGEP2(ctx->ac.builder, ctx->ac.i8, addr, &offset, 1, "") :
         LLVMBuildGEP2(ctx->ac.builder, ctx->ac.i8, addr, &offset, 1, "");

      LLVMSetMetadata(addr, ctx->ac.uniform_md_kind, ctx->ac.empty_md);
      result = LLVMBuildLoad2(ctx->ac.builder, result_type, addr, "");
      LLVMSetMetadata(result, ctx->ac.invariant_load_md_kind, ctx->ac.empty_md);
      break;
   }
   case nir_intrinsic_ordered_xfb_counter_add_gfx11_amd: {
      /* Gfx11 GDS instructions only operate on the first active lane. All other lanes are
       * ignored. So are their EXEC bits. This uses the mutex feature of ds_ordered_count
       * to emulate a multi-dword atomic.
       *
       * This is the expected code:
       *    ds_ordered_count release=0 done=0   // lock mutex
       *    ds_add_gs_reg_rtn GDS_STRMOUT_DWORDS_WRITTEN_0
       *    ds_add_gs_reg_rtn GDS_STRMOUT_DWORDS_WRITTEN_1
       *    ds_add_gs_reg_rtn GDS_STRMOUT_DWORDS_WRITTEN_2
       *    ds_add_gs_reg_rtn GDS_STRMOUT_DWORDS_WRITTEN_3
       *    ds_ordered_count release=1 done=1   // unlock mutex
       *
       * GDS_STRMOUT_DWORDS_WRITTEN_n are just general-purpose global registers. We use them
       * because MCBP (mid-command-buffer preemption) saves and restores them, and it doesn't
       * save and restore GDS memory.
       */
      LLVMValueRef args[8] = {
         LLVMBuildIntToPtr(ctx->ac.builder, get_src(ctx, instr->src[0]),
                           LLVMPointerTypeInContext(ctx->ac.context, AC_ADDR_SPACE_GDS), ""),
         ctx->ac.i32_0,                             /* value to add */
         ctx->ac.i32_0,                             /* ordering */
         ctx->ac.i32_0,                             /* scope */
         ctx->ac.i1false,                           /* isVolatile */
         LLVMConstInt(ctx->ac.i32, 1 << 24, false), /* OA index, bits 24+: lane count */
         ctx->ac.i1false,                           /* wave release */
         ctx->ac.i1false,                           /* wave done */
      };

      /* Set release=0 to start a GDS mutex. Set done=0 because it's not the last one. */
      ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.ds.ordered.add", ctx->ac.i32,
                         args, ARRAY_SIZE(args), 0);
      ac_build_waitcnt(&ctx->ac, AC_WAIT_DS);

      LLVMValueRef global_count[4];
      LLVMValueRef count_vec = get_src(ctx, instr->src[1]);
      unsigned write_mask = nir_intrinsic_write_mask(instr);
      for (unsigned i = 0; i < instr->num_components; i++) {
         LLVMValueRef value =
            LLVMBuildExtractElement(ctx->ac.builder, count_vec,
                                    LLVMConstInt(ctx->ac.i32, i, false), "");
         if (write_mask & (1 << i)) {
            /* The offset is a relative offset from GDS_STRMOUT_DWORDS_WRITTEN_0. */
            global_count[i] =
               ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.ds.add.gs.reg.rtn.i32", ctx->ac.i32,
                                  (LLVMValueRef[]){value, LLVMConstInt(ctx->ac.i32, i * 4, 0)},
                                  2, 0);
         } else {
            global_count[i] = LLVMGetUndef(ctx->ac.i32);
         }
      }

      ac_build_waitcnt(&ctx->ac, AC_WAIT_DS);

      /* Set release=1 to end a GDS mutex. Set done=1 because it's the last one. */
      args[6] = args[7] = ctx->ac.i1true;
      ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.ds.ordered.add", ctx->ac.i32,
                         args, ARRAY_SIZE(args), 0);
      result = ac_build_gather_values(&ctx->ac, global_count, instr->num_components);
      break;
   }
   case nir_intrinsic_xfb_counter_sub_gfx11_amd: {
      /* must be called in a single lane of a workgroup. */
      LLVMValueRef sub_vec = get_src(ctx, instr->src[0]);
      unsigned write_mask = nir_intrinsic_write_mask(instr);

      for (unsigned i = 0; i < instr->num_components; i++) {
         if (write_mask & (1 << i)) {
            LLVMValueRef value =
               LLVMBuildExtractElement(ctx->ac.builder, sub_vec,
                                       LLVMConstInt(ctx->ac.i32, i, false), "");
            /* The offset is a relative offset from GDS_STRMOUT_DWORDS_WRITTEN_0. */
            ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.ds.sub.gs.reg.rtn.i32", ctx->ac.i32,
                               (LLVMValueRef[]){value, LLVMConstInt(ctx->ac.i32, i * 4, 0)},
                               2, 0);
         }
      }
      break;
   }
   case nir_intrinsic_ordered_add_loop_gfx12_amd: {
      const unsigned num_atomics = 6;
      char code[2048];
      char *ptr = code;

      /* Assembly outputs:
       *    i32 VGPR $0 = previous value in memory
       *
       * Assembly inputs:
       *    EXEC = one lane per counter (use nir_push_if, streamout should always enable 4 lanes)
       *    i64 SGPR $1 = atomic base address
       *    i32 VGPR $2 = 32-bit VGPR voffset (streamout should set local_invocation_index * 8)
       *    i32 SGPR $3 = orderedID
       *    i64 VGPR $4 = 64-bit VGPR atomic src (streamout should set {orderedID, numDwords})
       */

      /* Issue (num_atomics - 1) atomics to initialize the results.
       * There are no s_sleeps here because the atomics must be pipelined.
       */
      for (int i = 0; i < num_atomics - 1; i++) {
         /* global_atomic_ordered_add_b64 dst, offset, data, address */
         ptr += sprintf(ptr,
                        "global_atomic_ordered_add_b64 v[%u:%u], $2, $4, $1 th:TH_ATOMIC_RETURN\n"
                        "s_nop 15\n"
                        "s_nop 7\n",
                        3 + i * 2,
                        3 + i * 2 + 1);
      }

      /* This is an infinite while loop with breaks. The loop body executes "num_atomics"
       * atomics and the same number of conditional breaks.
       *
       * It's pipelined such that we only wait for the oldest atomic, so there is always
       * "num_atomics" atomics in flight while the shader is waiting.
       */
      unsigned inst_block_size = 3 + 1 + 3; /* size of the next sprintf in dwords */

      for (unsigned i = 0; i < num_atomics; i++) {
         unsigned issue_index = (num_atomics - 1 + i) % num_atomics;
         unsigned read_index = i;

         ptr += sprintf(ptr,
                        /* Issue (or repeat) the attempt. */
                        "global_atomic_ordered_add_b64 v[%u:%u], $2, $4, $1 th:TH_ATOMIC_RETURN\n"
                        "s_wait_loadcnt 0x%x\n"
                        /* if (result[check_index].ordered_id == ordered_id) {
                         *    return_value = result[check_index].value;
                         *    break;
                         * }
                         */
                        "v_cmp_eq_u32 %s, $3, v%u\n"
                        "v_mov_b32 $0, v%u\n"
                        "s_cbranch_vccnz 0x%x\n",
                        3 + issue_index * 2,
                        3 + issue_index * 2 + 1,
                        num_atomics - 1, /* wait count */
                        ctx->ac.wave_size == 32 ? "vcc_lo" : "vcc",
                        3 + read_index * 2, /* v_cmp_eq: src1 */
                        3 + read_index * 2 + 1, /* output */
                        inst_block_size * (num_atomics - i - 1) + 1); /* forward s_cbranch as loop break */
      }

      /* Jump to the beginning of the loop. */
      ptr += sprintf(ptr,
                     "s_branch 0x%x\n"
                     "s_wait_alu 0xfffe\n"
                     "s_wait_loadcnt 0x0\n",
                     (inst_block_size * -(int)num_atomics - 1) & 0xffff);

      LLVMTypeRef param_types[] = {ctx->ac.i64, ctx->ac.i32, ctx->ac.i32, ctx->ac.i64};
      LLVMTypeRef calltype = LLVMFunctionType(ctx->ac.i32, param_types, 4, false);

      /* =v means a VGPR output, =& means the dst register must be different from src registers,
       * s means an SGPR input, v means a VGPR input, ~{reg} means that the register is clobbered
       *
       * We need to list the registers manually because the clobber constraint doesn't prevent
       * input and output registers from being assigned the same registers as the ones that are
       * clobbered.
       *
       * Since registers in the clobber constraints are ignored by LLVM during computation of
       * register usage, we have to set the input register to the highest used register because
       * that one is included in the register usage computation.
       */
      char constraint[128];
      snprintf(constraint, sizeof(constraint), "=&{v0},{s[8:9]},{v%u},{s12},{v[1:2]},~{%s},~{v[3:%u]}",
               3 + num_atomics * 2,
               ctx->ac.wave_size == 32 ? "vcc_lo" : "vcc",
               3 + num_atomics * 2 - 1);

      LLVMValueRef inlineasm = LLVMConstInlineAsm(calltype, code, constraint, true, false);

      LLVMValueRef args[] = {
         get_src(ctx, instr->src[0]),
         get_src(ctx, instr->src[1]),
         get_src(ctx, instr->src[2]),
         get_src(ctx, instr->src[3]),
      };
      result = LLVMBuildCall2(ctx->ac.builder, calltype, inlineasm, args, 4, "");

      assert(ptr < code + sizeof(code));
      break;
   }
   case nir_intrinsic_export_amd: {
      unsigned flags = nir_intrinsic_flags(instr);
      unsigned target = nir_intrinsic_base(instr);
      unsigned write_mask = nir_intrinsic_write_mask(instr);

      struct ac_export_args args = {
         .target = target,
         .enabled_channels = write_mask,
         .compr = flags & AC_EXP_FLAG_COMPRESSED,
         .done = flags & AC_EXP_FLAG_DONE,
         .valid_mask = flags & AC_EXP_FLAG_VALID_MASK,
      };

      LLVMValueRef value = get_src(ctx, instr->src[0]);
      int num_components = ac_get_llvm_num_components(value);
      for (int i = 0; i < num_components; i++)
         args.out[i] = ac_llvm_extract_elem(&ctx->ac, value, i);

      ac_build_export(&ctx->ac, &args);
      break;
   }
   case nir_intrinsic_bvh64_intersect_ray_amd: {
      LLVMValueRef desc = get_src(ctx, instr->src[0]);
      LLVMValueRef node_id =
         LLVMBuildBitCast(ctx->ac.builder, get_src(ctx, instr->src[1]), ctx->ac.i64, "");
      LLVMValueRef t_max =
         LLVMBuildBitCast(ctx->ac.builder, get_src(ctx, instr->src[2]), ctx->ac.f32, "");
      LLVMValueRef origin =
         LLVMBuildBitCast(ctx->ac.builder, get_src(ctx, instr->src[3]), ctx->ac.v3f32, "");
      LLVMValueRef dir =
         LLVMBuildBitCast(ctx->ac.builder, get_src(ctx, instr->src[4]), ctx->ac.v3f32, "");
      LLVMValueRef inv_dir =
         LLVMBuildBitCast(ctx->ac.builder, get_src(ctx, instr->src[5]), ctx->ac.v3f32, "");

      LLVMValueRef args[6] = {
         node_id, t_max, origin, dir, inv_dir, desc,
      };

      result = ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.image.bvh.intersect.ray.i64.v3f32",
                                  ctx->ac.v4i32, args, ARRAY_SIZE(args), 0);
      break;
   }
   case nir_intrinsic_sleep_amd: {
      LLVMValueRef arg = LLVMConstInt(ctx->ac.i32, nir_intrinsic_base(instr), 0);
      ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.s.sleep", ctx->ac.voidt, &arg, 1, 0);
      break;
   }
   case nir_intrinsic_nop_amd: {
      LLVMValueRef arg = LLVMConstInt(ctx->ac.i16, nir_intrinsic_base(instr), 0);
      ac_build_intrinsic(&ctx->ac, "llvm.amdgcn.s.nop", ctx->ac.voidt, &arg, 1, 0);
      break;
   }
   default:
      fprintf(stderr, "Unknown intrinsic: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      return false;
   }
   if (result) {
      ctx->ssa_defs[instr->def.index] = result;
   }
   return true;
}

/* Disable anisotropic filtering if BASE_LEVEL == LAST_LEVEL.
 *
 * GFX6-GFX7:
 *   If BASE_LEVEL == LAST_LEVEL, the shader must disable anisotropic
 *   filtering manually. The driver sets img7 to a mask clearing
 *   MAX_ANISO_RATIO if BASE_LEVEL == LAST_LEVEL. The shader must do:
 *     s_and_b32 samp0, samp0, img7
 *
 * GFX8:
 *   The ANISO_OVERRIDE sampler field enables this fix in TA.
 */
static LLVMValueRef sici_fix_sampler_aniso(struct ac_nir_context *ctx, LLVMValueRef res,
                                           LLVMValueRef samp)
{
   LLVMBuilderRef builder = ctx->ac.builder;
   LLVMValueRef img7, samp0;

   if (ctx->ac.gfx_level >= GFX8)
      return samp;

   img7 = LLVMBuildExtractElement(builder, res, LLVMConstInt(ctx->ac.i32, 7, 0), "");
   samp0 = LLVMBuildExtractElement(builder, samp, ctx->ac.i32_0, "");
   samp0 = LLVMBuildAnd(builder, samp0, img7, "");
   return LLVMBuildInsertElement(builder, samp, samp0, ctx->ac.i32_0, "");
}

static void tex_fetch_ptrs(struct ac_nir_context *ctx, nir_tex_instr *instr,
                           struct waterfall_context *wctx, LLVMValueRef *res_ptr,
                           LLVMValueRef *samp_ptr)
{
   LLVMValueRef texture_dynamic_handle = NULL;
   LLVMValueRef sampler_dynamic_handle = NULL;
   int plane = -1;

   *res_ptr = NULL;
   *samp_ptr = NULL;
   for (unsigned i = 0; i < instr->num_srcs; i++) {
      switch (instr->src[i].src_type) {
      case nir_tex_src_texture_handle:
      case nir_tex_src_sampler_handle: {
         LLVMValueRef val = get_src(ctx, instr->src[i].src);
         if (LLVMGetTypeKind(LLVMTypeOf(val)) == LLVMVectorTypeKind) {
            if (instr->src[i].src_type == nir_tex_src_texture_handle)
               *res_ptr = val;
            else
               *samp_ptr = val;
         } else {
            if (instr->src[i].src_type == nir_tex_src_texture_handle)
               texture_dynamic_handle = val;
            else
               sampler_dynamic_handle = val;
         }
         break;
      }
      case nir_tex_src_plane:
         plane = nir_src_as_int(instr->src[i].src);
         break;
      default:
         break;
      }
   }

   enum ac_descriptor_type main_descriptor =
      instr->sampler_dim == GLSL_SAMPLER_DIM_BUF ? AC_DESC_BUFFER : AC_DESC_IMAGE;

   if (plane >= 0) {
      assert(instr->op != nir_texop_txf_ms);
      assert(instr->sampler_dim != GLSL_SAMPLER_DIM_BUF);

      main_descriptor = AC_DESC_PLANE_0 + plane;
   }

   if (instr->op == nir_texop_fragment_mask_fetch_amd) {
      /* The fragment mask is fetched from the compressed
       * multisampled surface.
       */
      assert(ctx->ac.gfx_level < GFX11);
      main_descriptor = AC_DESC_FMASK;
   }

   /* descriptor handles given through nir_tex_src_{texture,sampler}_handle */
   if (instr->texture_non_uniform)
      texture_dynamic_handle = enter_waterfall(ctx, &wctx[0], texture_dynamic_handle, true);

   if (instr->sampler_non_uniform)
      sampler_dynamic_handle = enter_waterfall(ctx, &wctx[1], sampler_dynamic_handle, true);

   if (texture_dynamic_handle)
      *res_ptr = ctx->abi->load_sampler_desc(ctx->abi, texture_dynamic_handle, main_descriptor);

   if (sampler_dynamic_handle) {
      *samp_ptr = ctx->abi->load_sampler_desc(ctx->abi, sampler_dynamic_handle, AC_DESC_SAMPLER);

      if (ctx->abi->disable_aniso_single_level && instr->sampler_dim < GLSL_SAMPLER_DIM_RECT)
         *samp_ptr = sici_fix_sampler_aniso(ctx, *res_ptr, *samp_ptr);
   }
}

static void visit_tex(struct ac_nir_context *ctx, nir_tex_instr *instr)
{
   LLVMValueRef result = NULL;
   struct ac_image_args args = {0};
   LLVMValueRef sample_index = NULL;
   LLVMValueRef ddx = NULL, ddy = NULL;
   struct waterfall_context wctx[2] = {{{0}}};

   tex_fetch_ptrs(ctx, instr, wctx, &args.resource, &args.sampler);

   for (unsigned i = 0; i < instr->num_srcs; i++) {
      switch (instr->src[i].src_type) {
      case nir_tex_src_coord: {
         LLVMValueRef coord = get_src(ctx, instr->src[i].src);
         args.a16 = instr->src[i].src.ssa->bit_size == 16;
         for (unsigned chan = 0; chan < instr->coord_components; ++chan)
            args.coords[chan] = ac_llvm_extract_elem(&ctx->ac, coord, chan);
         break;
      }
      case nir_tex_src_projector:
         break;
      case nir_tex_src_comparator:
         if (instr->is_shadow) {
            args.compare = get_src(ctx, instr->src[i].src);
            args.compare = ac_to_float(&ctx->ac, args.compare);
            assert(instr->src[i].src.ssa->bit_size == 32);
         }
         break;
      case nir_tex_src_offset:
         args.offset = get_src(ctx, instr->src[i].src);
         /* We pack it with bit shifts, so we need it to be 32-bit. */
         assert(ac_get_elem_bits(&ctx->ac, LLVMTypeOf(args.offset)) == 32);
         break;
      case nir_tex_src_bias:
         args.bias = get_src(ctx, instr->src[i].src);
         break;
      case nir_tex_src_lod:
         if (nir_src_is_const(instr->src[i].src) && nir_src_as_uint(instr->src[i].src) == 0)
            args.level_zero = true;
         else
            args.lod = get_src(ctx, instr->src[i].src);
         break;
      case nir_tex_src_ms_index:
         sample_index = get_src(ctx, instr->src[i].src);
         break;
      case nir_tex_src_ddx:
         ddx = get_src(ctx, instr->src[i].src);
         args.g16 = instr->src[i].src.ssa->bit_size == 16;
         break;
      case nir_tex_src_ddy:
         ddy = get_src(ctx, instr->src[i].src);
         assert(LLVMTypeOf(ddy) == LLVMTypeOf(ddx));
         break;
      case nir_tex_src_min_lod:
         args.min_lod = get_src(ctx, instr->src[i].src);
         break;
      case nir_tex_src_texture_offset:
      case nir_tex_src_sampler_offset:
      case nir_tex_src_plane:
      default:
         break;
      }
   }

   if (args.offset) {
      /* offset for txf has been lowered in nir. */
      assert(instr->op != nir_texop_txf);

      LLVMValueRef offset[3], pack;
      for (unsigned chan = 0; chan < 3; ++chan)
         offset[chan] = ctx->ac.i32_0;

      unsigned num_components = ac_get_llvm_num_components(args.offset);
      for (unsigned chan = 0; chan < num_components; chan++) {
         offset[chan] = ac_llvm_extract_elem(&ctx->ac, args.offset, chan);
         offset[chan] =
            LLVMBuildAnd(ctx->ac.builder, offset[chan], LLVMConstInt(ctx->ac.i32, 0x3f, false), "");
         if (chan)
            offset[chan] = LLVMBuildShl(ctx->ac.builder, offset[chan],
                                        LLVMConstInt(ctx->ac.i32, chan * 8, false), "");
      }
      pack = LLVMBuildOr(ctx->ac.builder, offset[0], offset[1], "");
      pack = LLVMBuildOr(ctx->ac.builder, pack, offset[2], "");
      args.offset = pack;
   }

   /* Section 8.23.1 (Depth Texture Comparison Mode) of the
    * OpenGL 4.5 spec says:
    *
    *    "If the texture’s internal format indicates a fixed-point
    *     depth texture, then D_t and D_ref are clamped to the
    *     range [0, 1]; otherwise no clamping is performed."
    *
    * TC-compatible HTILE promotes Z16 and Z24 to Z32_FLOAT,
    * so the depth comparison value isn't clamped for Z16 and
    * Z24 anymore. Do it manually here for GFX8-9; GFX10 has
    * an explicitly clamped 32-bit float format.
    */
   if (args.compare && ctx->ac.gfx_level >= GFX8 && ctx->ac.gfx_level <= GFX9 &&
       ctx->abi->clamp_shadow_reference) {
      LLVMValueRef upgraded, clamped;

      upgraded = LLVMBuildExtractElement(ctx->ac.builder, args.sampler,
                                         LLVMConstInt(ctx->ac.i32, 3, false), "");
      upgraded = LLVMBuildLShr(ctx->ac.builder, upgraded, LLVMConstInt(ctx->ac.i32, 29, false), "");
      upgraded = LLVMBuildTrunc(ctx->ac.builder, upgraded, ctx->ac.i1, "");
      clamped = ac_build_clamp(&ctx->ac, args.compare);
      args.compare = LLVMBuildSelect(ctx->ac.builder, upgraded, clamped, args.compare, "");
   }

   /* pack derivatives */
   if (ddx || ddy) {
      int num_deriv_channels;
      switch (instr->sampler_dim) {
      case GLSL_SAMPLER_DIM_3D:
         num_deriv_channels = 3;
         break;
      case GLSL_SAMPLER_DIM_2D:
      case GLSL_SAMPLER_DIM_CUBE:
      default:
         num_deriv_channels = 2;
         break;
      case GLSL_SAMPLER_DIM_1D:
         num_deriv_channels = 1;
         break;
      }

      for (unsigned i = 0; i < num_deriv_channels; i++) {
         args.derivs[i] = ac_to_float(&ctx->ac, ac_llvm_extract_elem(&ctx->ac, ddx, i));
         args.derivs[num_deriv_channels + i] =
            ac_to_float(&ctx->ac, ac_llvm_extract_elem(&ctx->ac, ddy, i));
      }
   }

   /* Pack sample index */
   if (sample_index && (instr->op == nir_texop_txf_ms || instr->op == nir_texop_fragment_fetch_amd))
      args.coords[instr->coord_components] = sample_index;

   bool is_new_style_shadow = instr->is_shadow && instr->is_new_style_shadow &&
                              instr->op != nir_texop_lod && instr->op != nir_texop_tg4;
   unsigned num_components = util_last_bit(nir_def_components_read(&instr->def));

   /* DMASK was repurposed for GATHER4. 4 components are always
    * returned and DMASK works like a swizzle - it selects
    * the component to fetch. The only valid DMASK values are
    * 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
    * (red,red,red,red) etc.) The ISA document doesn't mention
    * this.
    */
   if (instr->op == nir_texop_tg4) {
      if (instr->is_shadow)
         args.dmask = 1;
      else
         args.dmask = 1 << instr->component;
   } else if (is_new_style_shadow || instr->op == nir_texop_fragment_mask_fetch_amd) {
      args.dmask = 1;
   } else {
      args.dmask = BITFIELD_MASK(num_components);
   }

   if (instr->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
      args.dim = ac_get_sampler_dim(ctx->ac.gfx_level, instr->sampler_dim, instr->is_array);
      args.unorm = instr->sampler_dim == GLSL_SAMPLER_DIM_RECT;
   }

   /* Adjust the number of coordinates because we only need (x,y) for 2D
    * multisampled images and (x,y,layer) for 2D multisampled layered
    * images or for multisampled input attachments.
    */
   if (instr->op == nir_texop_fragment_mask_fetch_amd) {
      if (args.dim == ac_image_2dmsaa) {
         args.dim = ac_image_2d;
      } else {
         assert(args.dim == ac_image_2darraymsaa);
         args.dim = ac_image_2darray;
      }
   }

   /* Set TRUNC_COORD=0 for textureGather(). */
   if (instr->op == nir_texop_tg4 && !ctx->ac.info->conformant_trunc_coord) {
      LLVMValueRef dword0 = LLVMBuildExtractElement(ctx->ac.builder, args.sampler, ctx->ac.i32_0, "");
      dword0 = LLVMBuildAnd(ctx->ac.builder, dword0, LLVMConstInt(ctx->ac.i32, C_008F30_TRUNC_COORD, 0), "");
      args.sampler = LLVMBuildInsertElement(ctx->ac.builder, args.sampler, dword0, ctx->ac.i32_0, "");
   }

   args.d16 = instr->def.bit_size == 16;
   args.tfe = instr->is_sparse;

   result = build_tex_intrinsic(ctx, instr, &args);

   if (instr->op == nir_texop_fragment_mask_fetch_amd) {
      /* Use 0x76543210 if the image doesn't have FMASK. */
      LLVMValueRef tmp = LLVMBuildBitCast(ctx->ac.builder, args.resource, ctx->ac.v8i32, "");
      tmp = LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->ac.i32_1, "");
      tmp = LLVMBuildICmp(ctx->ac.builder, LLVMIntNE, tmp, ctx->ac.i32_0, "");
      result = LLVMBuildSelect(ctx->ac.builder, tmp,
                               LLVMBuildExtractElement(ctx->ac.builder, result, ctx->ac.i32_0, ""),
                               LLVMConstInt(ctx->ac.i32, 0x76543210, false), "");
   } else {
      /* ac_build_image_opcode always returns 4 components plus 1 optional value
       * containing residency information. Adjust result to match the output expected by
       * the NIR instruction.
       */
      LLVMValueRef code = NULL;
      if (instr->is_sparse)
         code = ac_llvm_extract_elem(&ctx->ac, result, 4);

      result = ac_trim_vector(&ctx->ac, result, num_components - (instr->is_sparse ? 1 : 0));

      if (code)
         result = ac_build_concat(&ctx->ac, result, code);
   }

   if (result) {
      result = ac_to_integer(&ctx->ac, result);

      for (int i = ARRAY_SIZE(wctx); --i >= 0;) {
         result = exit_waterfall(ctx, wctx + i, result);
      }

      ctx->ssa_defs[instr->def.index] = result;
   }
}

static void visit_phi(struct ac_nir_context *ctx, nir_phi_instr *instr)
{
   LLVMTypeRef type = get_def_type(ctx, &instr->def);
   LLVMValueRef result = LLVMBuildPhi(ctx->ac.builder, type, "");

   ctx->ssa_defs[instr->def.index] = result;
   _mesa_hash_table_insert(ctx->phis, instr, result);
}

static void visit_post_phi(struct ac_nir_context *ctx, nir_phi_instr *instr, LLVMValueRef llvm_phi)
{
   nir_foreach_phi_src (src, instr) {
      LLVMBasicBlockRef block = get_block(ctx, src->pred);
      LLVMValueRef llvm_src = get_src(ctx, src->src);

      LLVMAddIncoming(llvm_phi, &llvm_src, &block, 1);
   }
}

static void phi_post_pass(struct ac_nir_context *ctx)
{
   hash_table_foreach(ctx->phis, entry)
   {
      visit_post_phi(ctx, (nir_phi_instr *)entry->key, (LLVMValueRef)entry->data);
   }
}

static void visit_ssa_undef(struct ac_nir_context *ctx, const nir_undef_instr *instr)
{
   unsigned num_components = instr->def.num_components;
   LLVMTypeRef type = LLVMIntTypeInContext(ctx->ac.context, instr->def.bit_size);

   LLVMValueRef undef;

   if (num_components == 1)
      undef = LLVMGetUndef(type);
   else {
      undef = LLVMGetUndef(LLVMVectorType(type, num_components));
   }
   ctx->ssa_defs[instr->def.index] = undef;
}

static bool visit_jump(struct ac_llvm_context *ctx, const nir_jump_instr *instr)
{
   switch (instr->type) {
   case nir_jump_break:
      ac_build_break(ctx);
      break;
   case nir_jump_continue:
      ac_build_continue(ctx);
      break;
   default:
      fprintf(stderr, "Unknown NIR jump instr: ");
      nir_print_instr(&instr->instr, stderr);
      fprintf(stderr, "\n");
      return false;
   }
   return true;
}

static bool visit_cf_list(struct ac_nir_context *ctx, struct exec_list *list);

static bool visit_block(struct ac_nir_context *ctx, nir_block *block)
{
   LLVMBasicBlockRef blockref = LLVMGetInsertBlock(ctx->ac.builder);
   LLVMValueRef first = LLVMGetFirstInstruction(blockref);
   if (first) {
      /* ac_branch_exited() might have already inserted non-phis */
      LLVMPositionBuilderBefore(ctx->ac.builder, LLVMGetFirstInstruction(blockref));
   }

   nir_foreach_phi(phi, block) {
      visit_phi(ctx, phi);
   }

   LLVMPositionBuilderAtEnd(ctx->ac.builder, blockref);

   nir_foreach_instr (instr, block) {
      switch (instr->type) {
      case nir_instr_type_alu:
         if (!visit_alu(ctx, nir_instr_as_alu(instr)))
            return false;
         break;
      case nir_instr_type_load_const:
         visit_load_const(ctx, nir_instr_as_load_const(instr));
         break;
      case nir_instr_type_intrinsic:
         if (!visit_intrinsic(ctx, nir_instr_as_intrinsic(instr)))
            return false;
         break;
      case nir_instr_type_tex:
         visit_tex(ctx, nir_instr_as_tex(instr));
         break;
      case nir_instr_type_phi:
         break;
      case nir_instr_type_undef:
         visit_ssa_undef(ctx, nir_instr_as_undef(instr));
         break;
      case nir_instr_type_jump:
         if (!visit_jump(&ctx->ac, nir_instr_as_jump(instr)))
            return false;
         break;
      case nir_instr_type_deref:
         assert (!nir_deref_mode_is_one_of(nir_instr_as_deref(instr),
                                           nir_var_mem_shared | nir_var_mem_global));
         break;
      default:
         fprintf(stderr, "Unknown NIR instr type: ");
         nir_print_instr(instr, stderr);
         fprintf(stderr, "\n");
         return false;
      }
   }

   _mesa_hash_table_insert(ctx->defs, block, LLVMGetInsertBlock(ctx->ac.builder));

   return true;
}

static bool visit_if(struct ac_nir_context *ctx, nir_if *if_stmt)
{
   LLVMValueRef value = get_src(ctx, if_stmt->condition);

   nir_block *then_block = (nir_block *)exec_list_get_head(&if_stmt->then_list);

   ac_build_ifcc(&ctx->ac, value, then_block->index);

   if (!visit_cf_list(ctx, &if_stmt->then_list))
      return false;

   if (!exec_list_is_empty(&if_stmt->else_list)) {
      nir_block *else_block = (nir_block *)exec_list_get_head(&if_stmt->else_list);

      ac_build_else(&ctx->ac, else_block->index);
      if (!visit_cf_list(ctx, &if_stmt->else_list))
         return false;
   }

   ac_build_endif(&ctx->ac, then_block->index);
   return true;
}

static bool visit_loop(struct ac_nir_context *ctx, nir_loop *loop)
{
   assert(!nir_loop_has_continue_construct(loop));
   nir_block *first_loop_block = (nir_block *)exec_list_get_head(&loop->body);

   ac_build_bgnloop(&ctx->ac, first_loop_block->index);

   if (!visit_cf_list(ctx, &loop->body))
      return false;

   ac_build_endloop(&ctx->ac, first_loop_block->index);
   return true;
}

static bool visit_cf_list(struct ac_nir_context *ctx, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list)
   {
      switch (node->type) {
      case nir_cf_node_block:
         if (!visit_block(ctx, nir_cf_node_as_block(node)))
            return false;
         break;

      case nir_cf_node_if:
         if (!visit_if(ctx, nir_cf_node_as_if(node)))
            return false;
         break;

      case nir_cf_node_loop:
         if (!visit_loop(ctx, nir_cf_node_as_loop(node)))
            return false;
         break;

      default:
         return false;
      }
   }
   return true;
}

static void setup_scratch(struct ac_nir_context *ctx, struct nir_shader *shader)
{
   if (shader->scratch_size == 0)
      return;

   LLVMTypeRef type = LLVMArrayType(ctx->ac.i8, shader->scratch_size);
   ctx->scratch = (struct ac_llvm_pointer) {
      .value = ac_build_alloca_undef(&ctx->ac, type, "scratch"),
      .pointee_type = type
   };
}

static void setup_constant_data(struct ac_nir_context *ctx, struct nir_shader *shader)
{
   if (!shader->constant_data)
      return;

   LLVMValueRef data = LLVMConstStringInContext(ctx->ac.context, shader->constant_data,
                                                shader->constant_data_size, true);
   LLVMTypeRef type = LLVMArrayType(ctx->ac.i8, shader->constant_data_size);
   LLVMValueRef global =
      LLVMAddGlobalInAddressSpace(ctx->ac.module, type, "const_data", AC_ADDR_SPACE_CONST);

   LLVMSetInitializer(global, data);
   LLVMSetGlobalConstant(global, true);
   LLVMSetVisibility(global, LLVMHiddenVisibility);
   ctx->constant_data = (struct ac_llvm_pointer) {
      .value = global,
      .pointee_type = type
   };
}

static void setup_gds(struct ac_nir_context *ctx, nir_function_impl *impl)
{
   bool has_gds_atomic = false;

   if (ctx->ac.gfx_level >= GFX10 &&
       (ctx->stage == MESA_SHADER_VERTEX ||
        ctx->stage == MESA_SHADER_TESS_EVAL ||
        ctx->stage == MESA_SHADER_GEOMETRY)) {

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            has_gds_atomic |= intrin->intrinsic == nir_intrinsic_gds_atomic_add_amd;
         }
      }
   }

   unsigned gds_size = has_gds_atomic ? 0x100 : 0;

   if (gds_size)
      ac_llvm_add_target_dep_function_attr(ctx->main_function, "amdgpu-gds-size", gds_size);
}

bool ac_nir_translate(struct ac_llvm_context *ac, struct ac_shader_abi *abi,
                      const struct ac_shader_args *args, struct nir_shader *nir)
{
   struct ac_nir_context ctx = {0};
   struct nir_function *func;
   bool ret;

   ctx.ac = *ac;
   ctx.abi = abi;
   ctx.args = args;

   ctx.stage = nir->info.stage;
   ctx.info = &nir->info;

   ctx.main_function = LLVMGetBasicBlockParent(LLVMGetInsertBlock(ctx.ac.builder));

   ctx.defs = _mesa_hash_table_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);
   ctx.phis = _mesa_hash_table_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);

   if (ctx.abi->kill_ps_if_inf_interp)
      ctx.verified_interp =
         _mesa_hash_table_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);

   func = (struct nir_function *)exec_list_get_head(&nir->functions);

   nir_index_ssa_defs(func->impl);
   ctx.ssa_defs = calloc(func->impl->ssa_alloc, sizeof(LLVMValueRef));

   setup_scratch(&ctx, nir);
   setup_constant_data(&ctx, nir);
   setup_gds(&ctx, func->impl);

   if ((ret = visit_cf_list(&ctx, &func->impl->body)))
      phi_post_pass(&ctx);

   free(ctx.ssa_defs);
   ralloc_free(ctx.defs);
   ralloc_free(ctx.phis);
   if (ctx.abi->kill_ps_if_inf_interp)
      ralloc_free(ctx.verified_interp);

   return ret;
}
