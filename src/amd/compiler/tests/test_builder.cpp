/*
 * Copyright © 2020 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */
#include "helpers.h"

using namespace aco;

BEGIN_TEST(builder.v_mul_imm)
   for (unsigned i = GFX7; i <= GFX10; i++) {
      //>> v1: %a:v[0], v1: %b:v[1], s1: %c:s[0], s1: %d:s[1] = p_startpgm
      if (!setup_cs("v1 v1 s1 s1", (amd_gfx_level)i))
         continue;

      /* simple optimizations */

      //! p_unit_test 0, 0
      writeout(0, bld.v_mul_imm(bld.def(v1), inputs[0], 0));

      //! p_unit_test 1, %a
      writeout(1, bld.v_mul_imm(bld.def(v1), inputs[0], 1));

      //! v1: %res2 = v_lshlrev_b32 2, %a
      //! p_unit_test 2, %res2
      writeout(2, bld.v_mul_imm(bld.def(v1), inputs[0], 4));

      //! v1: %res3 = v_lshlrev_b32 31, %a
      //! p_unit_test 3, %res3
      writeout(3, bld.v_mul_imm(bld.def(v1), inputs[0], 2147483648u));

      /* single lshl+add/sub */

      //~gfx[78]! v1: %res4_tmp = v_lshlrev_b32 3, %a
      //~gfx[78]! v1: %res4,  s2: %_ = v_add_co_u32 %res4_tmp, %a
      //~gfx(9|10)! v1: %res4 = v_lshl_add_u32 %a, 3, %a
      //! p_unit_test 4, %res4
      writeout(4, bld.v_mul_imm(bld.def(v1), inputs[0], 9));

      //~gfx[78]! v1: %res5_tmp = v_lshlrev_b32 3, %a
      //~gfx[78]! v1: %res5,  s2: %_ = v_sub_co_u32 %res5_tmp, %a
      //~gfx(9|10)! v1: %res5 = p_v_mad_u32_u32_u16 %a, 7, 0
      //! p_unit_test 5, %res5
      writeout(5, bld.v_mul_imm(bld.def(v1), inputs[0], 7));

      /* lshl+add optimization with literal */

      //~gfx7! v1: %res6_tmp0 = v_lshlrev_b32 2, %a
      //~gfx7! v1: %res6_tmp1 = v_lshlrev_b32 6, %a
      //~gfx7! v1: %res6,  s2: %_ = v_add_co_u32 %res6_tmp1, %res6_tmp0
      //~gfx8! v1: %res6_tmp = p_parallelcopy 0x44
      //~gfx8! v1: %res6_tmp2 = v_mul_lo_u16 %a, %res6_tmp dst_sel:ubyte3 src0_sel:ubyte3 src1_sel:ubyte0
      //~gfx8! v1: %res6 = v_mad_u32_u24 %a, %res6_tmp, %res6_tmp2
      //~gfx9! s1: %res6_tmp = p_parallelcopy 0x44
      //~gfx9! v1: %res6 = p_v_mad_u32_u32_u16 %a, %res6_tmp, 0
      //~gfx10! v1: %res6 = p_v_mad_u32_u32_u16 %a, 0x44, 0
      //! p_unit_test 6, %res6
      writeout(6, bld.v_mul_imm(bld.def(v1), inputs[0], 4 | 64));

      //~gfx[79]! s1: %res7_tmp = p_parallelcopy 0x144
      //~gfx7! v1: %res7 = v_mul_lo_u32 %res7_tmp, %a
      //~gfx8! v1: %res7_tmp = p_parallelcopy 0x144
      //~gfx8! v1: %res7_tmp2 = v_mul_lo_u16 %a, %res7_tmp dst_sel:ubyte3 src0_sel:ubyte3 src1_sel:ubyte0
      //~gfx8! v1: %res7 = v_mad_u32_u24 %a, %res7_tmp, %res7_tmp2
      //~gfx9! v1: %res7 = p_v_mad_u32_u32_u16 %a, %res7_tmp, 0
      //~gfx10! v1: %res7 = p_v_mad_u32_u32_u16 %a, 0x144, 0
      //! p_unit_test 7, %res7
      writeout(7, bld.v_mul_imm(bld.def(v1), inputs[0], 4 | 64 | 256));

      //~gfx[79]! s1: %res8_tmp = p_parallelcopy 0x944
      //~gfx7! v1: %res8 = v_mul_lo_u32 %res8_tmp, %a
      //~gfx8! v1: %res8_tmp = p_parallelcopy 0x944
      //~gfx8! v1: %res8_tmp2 = v_mul_lo_u16 %a, %res8_tmp dst_sel:ubyte3 src0_sel:ubyte3 src1_sel:ubyte0
      //~gfx8! v1: %res8 = v_mad_u32_u24 %a, %res8_tmp, %res8_tmp2
      //~gfx9! v1: %res8 = p_v_mad_u32_u32_u16 %a, %res8_tmp, 0
      //~gfx10! v1: %res8 = p_v_mad_u32_u32_u16 %a, 0x944, 0
      //! p_unit_test 8, %res8
      writeout(8, bld.v_mul_imm(bld.def(v1), inputs[0], 4 | 64 | 256 | 2048));

      /* lshl+add optimization with inline constant */

      //~gfx7! v1: %res9_tmp0 = v_lshlrev_b32 1, %a
      //~gfx7! v1: %res9_tmp1 = v_lshlrev_b32 2, %a
      //~gfx7! v1: %res9,  s2: %_ = v_add_co_u32 %res9_tmp1, %res9_tmp0
      //~gfx8! v1: %res9_tmp = p_parallelcopy 6
      //~gfx8! v1: %res9_tmp2 = v_mul_lo_u16 %a, %res9_tmp dst_sel:ubyte3 src0_sel:ubyte3 src1_sel:ubyte0
      //~gfx8! v1: %res9 = v_mad_u32_u24 %a, 6, %res9_tmp2
      //~gfx(9|10)! v1: %res9 = p_v_mad_u32_u32_u16 %a, 6, 0
      //! p_unit_test 9, %res9
      writeout(9, bld.v_mul_imm(bld.def(v1), inputs[0], 2 | 4));

      //~gfx7! v1: %res10 = v_mul_lo_u32 14, %a
      //~gfx8! v1: %res10_tmp = p_parallelcopy 14
      //~gfx8! v1: %res10_tmp2 = v_mul_lo_u16 %a, %res10_tmp dst_sel:ubyte3 src0_sel:ubyte3 src1_sel:ubyte0
      //~gfx8! v1: %res10 = v_mad_u32_u24 %a, 14, %res10_tmp2
      //~gfx(9|10)! v1: %res10 = p_v_mad_u32_u32_u16 %a, 14, 0
      //! p_unit_test 10, %res10
      writeout(10, bld.v_mul_imm(bld.def(v1), inputs[0], 2 | 4 | 8));

      //~gfx7! v1: %res11 = v_mul_lo_u32 30, %a
      //~gfx8! v1: %res11_tmp = p_parallelcopy 30
      //~gfx8! v1: %res11_tmp2 = v_mul_lo_u16 %a, %res11_tmp dst_sel:ubyte3 src0_sel:ubyte3 src1_sel:ubyte0
      //~gfx8! v1: %res11 = v_mad_u32_u24 %a, 30, %res11_tmp2
      //~gfx(9|10)! v1: %res11 = p_v_mad_u32_u32_u16 %a, 30, 0
      //! p_unit_test 11, %res11
      writeout(11, bld.v_mul_imm(bld.def(v1), inputs[0], 2 | 4 | 8 | 16));

      finish_opt_test();
   }
END_TEST
