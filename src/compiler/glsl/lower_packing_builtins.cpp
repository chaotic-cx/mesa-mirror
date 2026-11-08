/*
 * Copyright © 2012 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ir.h"
#include "ir_builder.h"
#include "ir_optimization.h"
#include "ir_rvalue_visitor.h"

enum lower_packing_builtins_op {
   LOWER_PACK_UNPACK_NONE               = 0x0000,
   LOWER_PACK_HALF_2x16                 = 0x0001,
   LOWER_UNPACK_HALF_2x16               = 0x0002,
};

namespace {

using namespace ir_builder;

/**
 * A visitor that lowers built-in half float pack/unpack expressions
 */
class lower_packing_builtins_visitor : public ir_rvalue_visitor {
public:
   explicit lower_packing_builtins_visitor()
      : progress(false)
   {
      factory.instructions = &factory_instructions;
   }

   virtual ~lower_packing_builtins_visitor()
   {
      assert(factory_instructions.is_empty());
   }

   lower_packing_builtins_visitor(const lower_packing_builtins_visitor &) = delete;
   lower_packing_builtins_visitor & operator=(const lower_packing_builtins_visitor &) = delete;

   bool get_progress() { return progress; }

   void handle_rvalue(ir_rvalue **rvalue)
   {
      if (!*rvalue)
	 return;

      ir_expression *expr = (*rvalue)->as_expression();
      if (!expr)
	 return;

      enum lower_packing_builtins_op lowering_op =
         choose_lowering_op(expr->operation);

      if (lowering_op == LOWER_PACK_UNPACK_NONE)
         return;

      setup_factory(expr->node_linalloc);

      ir_rvalue *op0 = expr->operands[0];

      switch (lowering_op) {
      case LOWER_PACK_HALF_2x16:
         *rvalue = lower_pack_half_2x16(op0);
         break;
      case LOWER_UNPACK_HALF_2x16:
         *rvalue = lower_unpack_half_2x16(op0);
         break;
      case LOWER_PACK_UNPACK_NONE:
         assert(!"not reached");
         break;
      }

      teardown_factory();
      progress = true;
   }

private:
   bool progress;
   ir_factory factory;
   ir_exec_list factory_instructions;

   /**
    * Determine the needed lowering operation by filtering \a expr_op
    * through \ref op_mask.
    */
   enum lower_packing_builtins_op
   choose_lowering_op(ir_expression_operation expr_op)
   {
      switch (expr_op) {
      case ir_unop_pack_half_2x16: return LOWER_PACK_HALF_2x16;
      case ir_unop_unpack_half_2x16: return LOWER_UNPACK_HALF_2x16;
      default: return LOWER_PACK_UNPACK_NONE;
      }
   }

   void
   setup_factory(linear_ctx *linalloc)
   {
      assert(factory.linalloc == NULL);
      assert(factory.instructions->is_empty());

      factory.linalloc = linalloc;
   }

   void
   teardown_factory()
   {
      base_ir->insert_before(factory.instructions);
      assert(factory.instructions->is_empty());
      factory.linalloc = NULL;
   }

   template <typename T>
   ir_constant*
   constant(T x)
   {
      return factory.constant(x);
   }

   /**
    * \brief Unpack a uint32 into two uint16's.
    *
    * Interpret the given uint32 as a uint16 pair where the uint32's least
    * significant bits specify the pair's first element. Return the uint16
    * pair as a uvec2.
    */
   ir_rvalue*
   unpack_uint_to_uvec2(ir_rvalue *uint_rval)
   {
      assert(uint_rval->type == &glsl_type_builtin_uint);

      /* uint u = UINT_RVAL; */
      ir_variable *u = factory.make_temp(&glsl_type_builtin_uint,
                                          "tmp_unpack_uint_to_uvec2_u");
      factory.emit(assign(u, uint_rval));

      /* uvec2 u2; */
      ir_variable *u2 = factory.make_temp(&glsl_type_builtin_uvec2,
                                           "tmp_unpack_uint_to_uvec2_u2");

      /* u2.x = u & 0xffffu; */
      factory.emit(assign(u2, bit_and(u, constant(0xffffu)), WRITEMASK_X));

      /* u2.y = u >> 16u; */
      factory.emit(assign(u2, rshift(u, constant(16u)), WRITEMASK_Y));

      return deref(u2).val;
   }

   /**
    * \brief Lower the component-wise calculation of packHalf2x16.
    *
    * \param f_rval is one component of packHafl2x16's input
    * \param e_rval is the unshifted exponent bits of f_rval
    * \param m_rval is the unshifted mantissa bits of f_rval
    *
    * \return a uint rvalue that encodes a float16 in its lower 16 bits
    */
   ir_rvalue*
   pack_half_1x16_nosign(ir_rvalue *f_rval,
                         ir_rvalue *e_rval,
                         ir_rvalue *m_rval)
   {
      assert(e_rval->type == &glsl_type_builtin_uint);
      assert(m_rval->type == &glsl_type_builtin_uint);

      /* uint u16; */
      ir_variable *u16 = factory.make_temp(&glsl_type_builtin_uint,
                                           "tmp_pack_half_1x16_u16");

      /* float f = FLOAT_RVAL; */
      ir_variable *f = factory.make_temp(&glsl_type_builtin_float,
                                          "tmp_pack_half_1x16_f");
      factory.emit(assign(f, f_rval));

      /* uint e = E_RVAL; */
      ir_variable *e = factory.make_temp(&glsl_type_builtin_uint,
                                          "tmp_pack_half_1x16_e");
      factory.emit(assign(e, e_rval));

      /* uint m = M_RVAL; */
      ir_variable *m = factory.make_temp(&glsl_type_builtin_uint,
                                          "tmp_pack_half_1x16_m");
      factory.emit(assign(m, m_rval));

      /* Preliminaries
       * -------------
       *
       * For a float16, the bit layout is:
       *
       *   sign:     15
       *   exponent: 10:14
       *   mantissa: 0:9
       *
       * Let f16 be a float16 value. The sign, exponent, and mantissa
       * determine its value thus:
       *
       *   if e16 = 0 and m16 = 0, then zero:       (-1)^s16 * 0                               (1)
       *   if e16 = 0 and m16!= 0, then subnormal:  (-1)^s16 * 2^(e16 - 14) * (m16 / 2^10)     (2)
       *   if 0 < e16 < 31, then normal:            (-1)^s16 * 2^(e16 - 15) * (1 + m16 / 2^10) (3)
       *   if e16 = 31 and m16 = 0, then infinite:  (-1)^s16 * inf                             (4)
       *   if e16 = 31 and m16 != 0, then           NaN                                        (5)
       *
       * where 0 <= m16 < 2^10.
       *
       * For a float32, the bit layout is:
       *
       *   sign:     31
       *   exponent: 23:30
       *   mantissa: 0:22
       *
       * Let f32 be a float32 value. The sign, exponent, and mantissa
       * determine its value thus:
       *
       *   if e32 = 0 and m32 = 0, then zero:        (-1)^s * 0                                (10)
       *   if e32 = 0 and m32 != 0, then subnormal:  (-1)^s * 2^(e32 - 126) * (m32 / 2^23)     (11)
       *   if 0 < e32 < 255, then normal:            (-1)^s * 2^(e32 - 127) * (1 + m32 / 2^23) (12)
       *   if e32 = 255 and m32 = 0, then infinite:  (-1)^s * inf                              (13)
       *   if e32 = 255 and m32 != 0, then           NaN                                       (14)
       *
       * where 0 <= m32 < 2^23.
       *
       * The minimum and maximum normal float16 values are
       *
       *   min_norm16 = 2^(1 - 15) * (1 + 0 / 2^10) = 2^(-14)   (20)
       *   max_norm16 = 2^(30 - 15) * (1 + 1023 / 2^10)         (21)
       *
       * The step at max_norm16 is
       *
       *   max_step16 = 2^5                                     (22)
       *
       * Observe that the float16 boundary values in equations 20-21 lie in the
       * range of normal float32 values.
       *
       *
       * Rounding Behavior
       * -----------------
       * Not all float32 values can be exactly represented as a float16. We
       * round all such intermediate float32 values to the nearest float16; if
       * the float32 is exactly between to float16 values, we round to the one
       * with an even mantissa. This rounding behavior has several benefits:
       *
       *   - It has no sign bias.
       *
       *   - It reproduces the behavior of real hardware: opcode F32TO16 in Intel's
       *     GPU ISA.
       *
       *   - By reproducing the behavior of the GPU (at least on Intel hardware),
       *     compile-time evaluation of constant packHalf2x16 GLSL expressions will
       *     result in the same value as if the expression were executed on the
       *     GPU.
       *
       * Calculation
       * -----------
       * Our task is to compute s16, e16, m16 given f32.  Since this function
       * ignores the sign bit, assume that s32 = s16 = 0.  There are several
       * cases consider.
       */

      factory.emit(

         /* Case 1) f32 is NaN
          *
          *   The resultant f16 will also be NaN.
          */

         /* if (e32 == 255 && m32 != 0) { */
         if_tree(logic_and(equal(e, constant(0xffu << 23u)),
                           logic_not(equal(m, constant(0u)))),

            assign(u16, constant(0x7fffu)),

         /* Case 2) f32 lies in the range [0, min_norm16).
          *
          *   The resultant float16 will be either zero, subnormal, or normal.
          *
          *   Solving
          *
          *     f32 = min_norm16       (30)
          *
          *   gives
          *
          *     e32 = 113 and m32 = 0  (31)
          *
          *   Therefore this case occurs if and only if
          *
          *     e32 < 113              (32)
          */

         /* } else if (e32 < 113) { */
         if_tree(less(e, constant(113u << 23u)),

            /* u16 = uint(round_to_even(abs(f32) * float(1u << 24u))); */
            assign(u16, f2u(round_even(mul(expr(ir_unop_abs, f),
                                           constant((float) (1 << 24)))))),

         /* Case 3) f32 lies in the range
          *         [min_norm16, max_norm16 + max_step16).
          *
          *   The resultant float16 will be either normal or infinite.
          *
          *   Solving
          *
          *     f32 = max_norm16 + max_step16           (40)
          *         = 2^15 * (1 + 1023 / 2^10) + 2^5    (41)
          *         = 2^16                              (42)
          *   gives
          *
          *     e32 = 143 and m32 = 0                   (43)
          *
          *   We already solved the boundary condition f32 = min_norm16 above
          *   in equation 31. Therefore this case occurs if and only if
          *
          *     113 <= e32 and e32 < 143
          */

         /* } else if (e32 < 143) { */
         if_tree(less(e, constant(143u << 23u)),

            /* The addition below handles the case where the mantissa rounds
             * up to 1024 and bumps the exponent.
             *
             * u16 = ((e - (112u << 23u)) >> 13u)
             *     + round_to_even((float(m) / (1u << 13u));
             */
            assign(u16, add(rshift(sub(e, constant(112u << 23u)),
                                   constant(13u)),
                            f2u(round_even(
                                  div(u2f(m), constant((float) (1 << 13))))))),

         /* Case 4) f32 lies in the range [max_norm16 + max_step16, inf].
          *
          *   The resultant float16 will be infinite.
          *
          *   The cases above caught all float32 values in the range
          *   [0, max_norm16 + max_step16), so this is the fall-through case.
          */

         /* } else { */

            assign(u16, constant(31u << 10u))))));

         /* } */

       return deref(u16).val;
   }

   /**
    * \brief Lower a packHalf2x16 expression.
    *
    * \param vec2_rval is packHalf2x16's input
    * \return packHalf2x16's output as a uint rvalue
    */
   ir_rvalue*
   lower_pack_half_2x16(ir_rvalue *vec2_rval)
   {
      /* From page 89 (95 of pdf) of the GLSL ES 3.00 spec:
       *
       *    highp uint packHalf2x16 (mediump vec2 v)
       *    ----------------------------------------
       *    Returns an unsigned integer obtained by converting the components of
       *    a two-component floating-point vector to the 16-bit floating-point
       *    representation found in the OpenGL ES Specification, and then packing
       *    these two 16-bit integers into a 32-bit unsigned integer.
       *
       *    The first vector component specifies the 16 least- significant bits
       *    of the result; the second component specifies the 16 most-significant
       *    bits.
       */

      assert(vec2_rval->type == &glsl_type_builtin_vec2);

      /* vec2 f = VEC2_RVAL; */
      ir_variable *f = factory.make_temp(&glsl_type_builtin_vec2,
                                         "tmp_pack_half_2x16_f");
      factory.emit(assign(f, vec2_rval));

      /* uvec2 f32 = bitcast_f2u(f); */
      ir_variable *f32 = factory.make_temp(&glsl_type_builtin_uvec2,
                                            "tmp_pack_half_2x16_f32");
      factory.emit(assign(f32, expr(ir_unop_bitcast_f2u, f)));

      /* uvec2 f16; */
      ir_variable *f16 = factory.make_temp(&glsl_type_builtin_uvec2,
                                        "tmp_pack_half_2x16_f16");

      /* Get f32's unshifted exponent bits.
       *
       *   uvec2 e = f32 & 0x7f800000u;
       */
      ir_variable *e = factory.make_temp(&glsl_type_builtin_uvec2,
                                          "tmp_pack_half_2x16_e");
      factory.emit(assign(e, bit_and(f32, constant(0x7f800000u))));

      /* Get f32's unshifted mantissa bits.
       *
       *   uvec2 m = f32 & 0x007fffffu;
       */
      ir_variable *m = factory.make_temp(&glsl_type_builtin_uvec2,
                                          "tmp_pack_half_2x16_m");
      factory.emit(assign(m, bit_and(f32, constant(0x007fffffu))));

      /* Set f16's exponent and mantissa bits.
       *
       *   f16.x = pack_half_1x16_nosign(e.x, m.x);
       *   f16.y = pack_half_1y16_nosign(e.y, m.y);
       */
      factory.emit(assign(f16, pack_half_1x16_nosign(swizzle_x(f),
                                                     swizzle_x(e),
                                                     swizzle_x(m)),
                           WRITEMASK_X));
      factory.emit(assign(f16, pack_half_1x16_nosign(swizzle_y(f),
                                                     swizzle_y(e),
                                                     swizzle_y(m)),
                           WRITEMASK_Y));

      /* Set f16's sign bits.
       *
       *   f16 |= (f32 & (1u << 31u) >> 16u;
       */
      factory.emit(
         assign(f16, bit_or(f16,
                            rshift(bit_and(f32, constant(1u << 31u)),
                                   constant(16u)))));


      /* return (f16.y << 16u) | f16.x; */
      ir_rvalue *result = bit_or(lshift(swizzle_y(f16),
                                        constant(16u)),
                                 swizzle_x(f16));

      assert(result->type == &glsl_type_builtin_uint);
      return result;
   }

   /**
    * \brief Lower the component-wise calculation of unpackHalf2x16.
    *
    * Given a uint that encodes a float16 in its lower 16 bits, this function
    * returns a uint that encodes a float32 with the same value. The sign bit
    * of the float16 is ignored.
    *
    * \param e_rval is the unshifted exponent bits of a float16
    * \param m_rval is the unshifted mantissa bits of a float16
    * \param a uint rvalue that encodes a float32
    */
   ir_rvalue*
   unpack_half_1x16_nosign(ir_rvalue *e_rval, ir_rvalue *m_rval)
   {
      assert(e_rval->type == &glsl_type_builtin_uint);
      assert(m_rval->type == &glsl_type_builtin_uint);

      /* uint u32; */
      ir_variable *u32 = factory.make_temp(&glsl_type_builtin_uint,
                                           "tmp_unpack_half_1x16_u32");

      /* uint e = E_RVAL; */
      ir_variable *e = factory.make_temp(&glsl_type_builtin_uint,
                                          "tmp_unpack_half_1x16_e");
      factory.emit(assign(e, e_rval));

      /* uint m = M_RVAL; */
      ir_variable *m = factory.make_temp(&glsl_type_builtin_uint,
                                          "tmp_unpack_half_1x16_m");
      factory.emit(assign(m, m_rval));

      /* Preliminaries
       * -------------
       *
       * For a float16, the bit layout is:
       *
       *   sign:     15
       *   exponent: 10:14
       *   mantissa: 0:9
       *
       * Let f16 be a float16 value. The sign, exponent, and mantissa
       * determine its value thus:
       *
       *   if e16 = 0 and m16 = 0, then zero:       (-1)^s16 * 0                               (1)
       *   if e16 = 0 and m16!= 0, then subnormal:  (-1)^s16 * 2^(e16 - 14) * (m16 / 2^10)     (2)
       *   if 0 < e16 < 31, then normal:            (-1)^s16 * 2^(e16 - 15) * (1 + m16 / 2^10) (3)
       *   if e16 = 31 and m16 = 0, then infinite:  (-1)^s16 * inf                             (4)
       *   if e16 = 31 and m16 != 0, then           NaN                                        (5)
       *
       * where 0 <= m16 < 2^10.
       *
       * For a float32, the bit layout is:
       *
       *   sign: 31
       *   exponent: 23:30
       *   mantissa: 0:22
       *
       * Let f32 be a float32 value. The sign, exponent, and mantissa
       * determine its value thus:
       *
       *   if e32 = 0 and m32 = 0, then zero:        (-1)^s * 0                                (10)
       *   if e32 = 0 and m32 != 0, then subnormal:  (-1)^s * 2^(e32 - 126) * (m32 / 2^23)     (11)
       *   if 0 < e32 < 255, then normal:            (-1)^s * 2^(e32 - 127) * (1 + m32 / 2^23) (12)
       *   if e32 = 255 and m32 = 0, then infinite:  (-1)^s * inf                              (13)
       *   if e32 = 255 and m32 != 0, then           NaN                                       (14)
       *
       * where 0 <= m32 < 2^23.
       *
       * Calculation
       * -----------
       * Our task is to compute s32, e32, m32 given f16.  Since this function
       * ignores the sign bit, assume that s32 = s16 = 0.  There are several
       * cases consider.
       */

      factory.emit(

         /* Case 1) f16 is zero or subnormal.
          *
          *   The simplest method of calcuating f32 in this case is
          *
          *     f32 = f16                       (20)
          *         = 2^(-14) * (m16 / 2^10)    (21)
          *         = m16 / 2^(-24)             (22)
          */

         /* if (e16 == 0) { */
         if_tree(equal(e, constant(0u)),

            /* u32 = bitcast_f2u(float(m) / float(1 << 24)); */
            assign(u32, expr(ir_unop_bitcast_f2u,
                                div(u2f(m), constant((float)(1 << 24))))),

         /* Case 2) f16 is normal.
          *
          *   The equation
          *
          *     f32 = f16                              (30)
          *     2^(e32 - 127) * (1 + m32 / 2^23) =     (31)
          *       2^(e16 - 15) * (1 + m16 / 2^10)
          *
          *   can be decomposed into two
          *
          *     2^(e32 - 127) = 2^(e16 - 15)           (32)
          *     1 + m32 / 2^23 = 1 + m16 / 2^10        (33)
          *
          *   which solve to
          *
          *     e32 = e16 + 112                        (34)
          *     m32 = m16 * 2^13                       (35)
          */

         /* } else if (e16 < 31)) { */
         if_tree(less(e, constant(31u << 10u)),

              /* u32 = ((e + (112 << 10)) | m) << 13;
               */
              assign(u32, lshift(bit_or(add(e, constant(112u << 10u)), m),
                                 constant(13u))),


         /* Case 3) f16 is infinite. */
         if_tree(equal(m, constant(0u)),

                 assign(u32, constant(255u << 23u)),

         /* Case 4) f16 is NaN. */
         /* } else { */

            assign(u32, constant(0x7fffffffu))))));

         /* } */

      return deref(u32).val;
   }

   /**
    * \brief Lower an unpackHalf2x16 expression.
    *
    * \param uint_rval is unpackHalf2x16's input
    * \return unpackHalf2x16's output as a vec2 rvalue
    */
   ir_rvalue*
   lower_unpack_half_2x16(ir_rvalue *uint_rval)
   {
      /* From page 89 (95 of pdf) of the GLSL ES 3.00 spec:
       *
       *    mediump vec2 unpackHalf2x16 (highp uint v)
       *    ------------------------------------------
       *    Returns a two-component floating-point vector with components
       *    obtained by unpacking a 32-bit unsigned integer into a pair of 16-bit
       *    values, interpreting those values as 16-bit floating-point numbers
       *    according to the OpenGL ES Specification, and converting them to
       *    32-bit floating-point values.
       *
       *    The first component of the vector is obtained from the
       *    16 least-significant bits of v; the second component is obtained
       *    from the 16 most-significant bits of v.
       */
      assert(uint_rval->type == &glsl_type_builtin_uint);

      /* uint u = RVALUE;
       * uvec2 f16 = uvec2(u.x & 0xffff, u.y >> 16);
       */
      ir_variable *f16 = factory.make_temp(&glsl_type_builtin_uvec2,
                                            "tmp_unpack_half_2x16_f16");
      factory.emit(assign(f16, unpack_uint_to_uvec2(uint_rval)));

      /* uvec2 f32; */
      ir_variable *f32 = factory.make_temp(&glsl_type_builtin_uvec2,
                                            "tmp_unpack_half_2x16_f32");

      /* Get f16's unshifted exponent bits.
       *
       *    uvec2 e = f16 & 0x7c00u;
       */
      ir_variable *e = factory.make_temp(&glsl_type_builtin_uvec2,
                                          "tmp_unpack_half_2x16_e");
      factory.emit(assign(e, bit_and(f16, constant(0x7c00u))));

      /* Get f16's unshifted mantissa bits.
       *
       *    uvec2 m = f16 & 0x03ffu;
       */
      ir_variable *m = factory.make_temp(&glsl_type_builtin_uvec2,
                                          "tmp_unpack_half_2x16_m");
      factory.emit(assign(m, bit_and(f16, constant(0x03ffu))));

      /* Set f32's exponent and mantissa bits.
       *
       *   f32.x = unpack_half_1x16_nosign(e.x, m.x);
       *   f32.y = unpack_half_1x16_nosign(e.y, m.y);
       */
      factory.emit(assign(f32, unpack_half_1x16_nosign(swizzle_x(e),
                                                       swizzle_x(m)),
                           WRITEMASK_X));
      factory.emit(assign(f32, unpack_half_1x16_nosign(swizzle_y(e),
                                                       swizzle_y(m)),
                           WRITEMASK_Y));

      /* Set f32's sign bit.
       *
       *    f32 |= (f16 & 0x8000u) << 16u;
       */
      factory.emit(assign(f32, bit_or(f32,
                                       lshift(bit_and(f16,
                                                      constant(0x8000u)),
                                              constant(16u)))));

      /* return bitcast_u2f(f32); */
      ir_rvalue *result = expr(ir_unop_bitcast_u2f, f32);
      assert(result->type == &glsl_type_builtin_vec2);
      return result;
   }
};

} // namespace anonymous

/**
 * \brief Lower the builtin packing functions.
 */
bool
lower_packing_builtins(ir_exec_list *instructions,
                       bool has_half_float_packing)
{
   if (has_half_float_packing)
      return false;

   lower_packing_builtins_visitor v;
   visit_list_elements(&v, instructions, true);
   return v.get_progress();
}
