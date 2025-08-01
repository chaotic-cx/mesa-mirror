/*
 * Copyright © 2008 Keith Packard
 * Copyright © 2014 Intel Corporation
 *
 * Permission to use, copy, modify, distribute, and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies and that both that copyright
 * notice and this permission notice appear in supporting documentation, and
 * that the name of the copyright holders not be used in advertising or
 * publicity pertaining to distribution of the software without specific,
 * written prior permission.  The copyright holders make no representations
 * about the suitability of this software for any purpose.  It is provided "as
 * is" without express or implied warranty.
 *
 * THE COPYRIGHT HOLDERS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
 * DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
 * TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brw_disasm.h"
#include "brw_disasm_info.h"
#include "brw_eu_defines.h"
#include "brw_eu.h"
#include "brw_eu_inst.h"
#include "brw_isa_info.h"
#include "brw_reg.h"
#include "util/half_float.h"

bool
brw_has_jip(const struct intel_device_info *devinfo, enum opcode opcode)
{
   return opcode == BRW_OPCODE_IF ||
          opcode == BRW_OPCODE_ELSE ||
          opcode == BRW_OPCODE_ENDIF ||
          opcode == BRW_OPCODE_WHILE ||
          opcode == BRW_OPCODE_BREAK ||
          opcode == BRW_OPCODE_CONTINUE ||
          opcode == BRW_OPCODE_HALT ||
          opcode == BRW_OPCODE_GOTO ||
          opcode == BRW_OPCODE_JOIN;
}

bool
brw_has_uip(const struct intel_device_info *devinfo, enum opcode opcode)
{
   return opcode == BRW_OPCODE_IF ||
          opcode == BRW_OPCODE_ELSE ||
          opcode == BRW_OPCODE_BREAK ||
          opcode == BRW_OPCODE_CONTINUE ||
          opcode == BRW_OPCODE_HALT ||
          opcode == BRW_OPCODE_GOTO;
}

bool
brw_has_branch_ctrl(const struct intel_device_info *devinfo, enum opcode opcode)
{
   switch (opcode) {
   case BRW_OPCODE_IF:
   case BRW_OPCODE_ELSE:
   case BRW_OPCODE_GOTO:
   case BRW_OPCODE_BREAK:
   case BRW_OPCODE_CALL:
   case BRW_OPCODE_CALLA:
   case BRW_OPCODE_CONTINUE:
   case BRW_OPCODE_ENDIF:
   case BRW_OPCODE_HALT:
   case BRW_OPCODE_JMPI:
   case BRW_OPCODE_RET:
   case BRW_OPCODE_WHILE:
   case BRW_OPCODE_BRC:
   case BRW_OPCODE_BRD:
      /* TODO: "join" should also be here if added */
      return true;
   default:
      return false;
   }
}

static bool
is_logic_instruction(unsigned opcode)
{
   return opcode == BRW_OPCODE_AND ||
          opcode == BRW_OPCODE_NOT ||
          opcode == BRW_OPCODE_OR ||
          opcode == BRW_OPCODE_XOR;
}

static bool
is_send(unsigned opcode)
{
   return opcode == BRW_OPCODE_SEND ||
          opcode == BRW_OPCODE_SENDC ||
          opcode == BRW_OPCODE_SENDS ||
          opcode == BRW_OPCODE_SENDSC;
}

static bool
is_split_send(UNUSED const struct intel_device_info *devinfo, unsigned opcode)
{
   if (devinfo->ver >= 12)
      return is_send(opcode);
   else
      return opcode == BRW_OPCODE_SENDS ||
             opcode == BRW_OPCODE_SENDSC;
}

static bool
is_send_gather(const struct brw_isa_info *isa,
               const struct brw_eu_inst *inst)
{
   return isa->devinfo->ver >= 30 &&
          is_split_send(isa->devinfo, brw_eu_inst_opcode(isa, inst)) &&
          brw_eu_inst_send_src0_reg_file(isa->devinfo, inst) == ARF;
}

const char *const conditional_modifier[16] = {
   [BRW_CONDITIONAL_NONE] = "",
   [BRW_CONDITIONAL_Z]    = ".z",
   [BRW_CONDITIONAL_NZ]   = ".nz",
   [BRW_CONDITIONAL_G]    = ".g",
   [BRW_CONDITIONAL_GE]   = ".ge",
   [BRW_CONDITIONAL_L]    = ".l",
   [BRW_CONDITIONAL_LE]   = ".le",
   [BRW_CONDITIONAL_R]    = ".r",
   [BRW_CONDITIONAL_O]    = ".o",
   [BRW_CONDITIONAL_U]    = ".u",
};

static const char *const m_negate[2] = {
   [0] = "",
   [1] = "-",
};

static const char *const _abs[2] = {
   [0] = "",
   [1] = "(abs)",
};

static const char *const m_bitnot[2] = { "", "~" };

static const char *const vert_stride[16] = {
   [0] = "0",
   [1] = "1",
   [2] = "2",
   [3] = "4",
   [4] = "8",
   [5] = "16",
   [6] = "32",
   [15] = "VxH",
};

static const char *const width[8] = {
   [0] = "1",
   [1] = "2",
   [2] = "4",
   [3] = "8",
   [4] = "16",
};

static const char *const horiz_stride[4] = {
   [0] = "0",
   [1] = "1",
   [2] = "2",
   [3] = "4"
};

static const char *const chan_sel[4] = {
   [0] = "x",
   [1] = "y",
   [2] = "z",
   [3] = "w",
};

static const char *const debug_ctrl[2] = {
   [0] = "",
   [1] = ".breakpoint"
};

static const char *const saturate[2] = {
   [0] = "",
   [1] = ".sat"
};

static const char *const cmpt_ctrl[2] = {
   [0] = "",
   [1] = "compacted"
};

static const char *const accwr[2] = {
   [0] = "",
   [1] = "AccWrEnable"
};

static const char *const branch_ctrl[2] = {
   [0] = "",
   [1] = "BranchCtrl"
};

static const char *const wectrl[2] = {
   [0] = "",
   [1] = "WE_all"
};

static const char *const exec_size[8] = {
   [0] = "1",
   [1] = "2",
   [2] = "4",
   [3] = "8",
   [4] = "16",
   [5] = "32"
};

static const char *const pred_inv[2] = {
   [0] = "+",
   [1] = "-"
};

const char *const pred_ctrl_align16[16] = {
   [1] = "",
   [2] = ".x",
   [3] = ".y",
   [4] = ".z",
   [5] = ".w",
   [6] = ".any4h",
   [7] = ".all4h",
};

static const char *const pred_ctrl_align1[16] = {
   [BRW_PREDICATE_NORMAL]        = "",
   [BRW_PREDICATE_ALIGN1_ANYV]   = ".anyv",
   [BRW_PREDICATE_ALIGN1_ALLV]   = ".allv",
   [BRW_PREDICATE_ALIGN1_ANY2H]  = ".any2h",
   [BRW_PREDICATE_ALIGN1_ALL2H]  = ".all2h",
   [BRW_PREDICATE_ALIGN1_ANY4H]  = ".any4h",
   [BRW_PREDICATE_ALIGN1_ALL4H]  = ".all4h",
   [BRW_PREDICATE_ALIGN1_ANY8H]  = ".any8h",
   [BRW_PREDICATE_ALIGN1_ALL8H]  = ".all8h",
   [BRW_PREDICATE_ALIGN1_ANY16H] = ".any16h",
   [BRW_PREDICATE_ALIGN1_ALL16H] = ".all16h",
   [BRW_PREDICATE_ALIGN1_ANY32H] = ".any32h",
   [BRW_PREDICATE_ALIGN1_ALL32H] = ".all32h",
};

static const char *const xe2_pred_ctrl[4] = {
   [BRW_PREDICATE_NORMAL]        = "",
   [XE2_PREDICATE_ANY]           = ".any",
   [XE2_PREDICATE_ALL]           = ".all",
};

static const char *const thread_ctrl[4] = {
   [BRW_THREAD_NORMAL] = "",
   [BRW_THREAD_ATOMIC] = "atomic",
   [BRW_THREAD_SWITCH] = "switch",
};

static const char *const dep_ctrl[4] = {
   [0] = "",
   [1] = "NoDDClr",
   [2] = "NoDDChk",
   [3] = "NoDDClr,NoDDChk",
};

static const char *const access_mode[2] = {
   [0] = "align1",
   [1] = "align16",
};

static const char *const reg_file[4] = {
   [ARF]       = "A",
   [FIXED_GRF] = "g",
   [IMM]       = "imm",
};

static const char *const writemask[16] = {
   [0x0] = ".",
   [0x1] = ".x",
   [0x2] = ".y",
   [0x3] = ".xy",
   [0x4] = ".z",
   [0x5] = ".xz",
   [0x6] = ".yz",
   [0x7] = ".xyz",
   [0x8] = ".w",
   [0x9] = ".xw",
   [0xa] = ".yw",
   [0xb] = ".xyw",
   [0xc] = ".zw",
   [0xd] = ".xzw",
   [0xe] = ".yzw",
   [0xf] = "",
};

static const char *const end_of_thread[2] = {
   [0] = "",
   [1] = "EOT"
};

static const char *const brw_sfid[16] = {
   [BRW_SFID_NULL]                  = "null",
   [BRW_SFID_SAMPLER]               = "sampler",
   [BRW_SFID_MESSAGE_GATEWAY]       = "gateway",
   [BRW_SFID_HDC2]                  = "hdc2",
   [BRW_SFID_RENDER_CACHE]          = "render",
   [BRW_SFID_URB]                   = "urb",
   [BRW_SFID_THREAD_SPAWNER]        = "ts/btd",
   [BRW_SFID_RAY_TRACE_ACCELERATOR] = "rt accel",
   [BRW_SFID_HDC_READ_ONLY]         = "hdc:ro",
   [BRW_SFID_HDC0]                  = "hdc0",
   [BRW_SFID_PIXEL_INTERPOLATOR]    = "pi",
   [BRW_SFID_HDC1]                  = "hdc1",
   [BRW_SFID_SLM]                   = "slm",
   [BRW_SFID_TGM]                   = "tgm",
   [BRW_SFID_UGM]                   = "ugm",
};

static const char *const gfx7_gateway_subfuncid[8] = {
   [BRW_MESSAGE_GATEWAY_SFID_OPEN_GATEWAY] = "open",
   [BRW_MESSAGE_GATEWAY_SFID_CLOSE_GATEWAY] = "close",
   [BRW_MESSAGE_GATEWAY_SFID_FORWARD_MSG] = "forward msg",
   [BRW_MESSAGE_GATEWAY_SFID_GET_TIMESTAMP] = "get timestamp",
   [BRW_MESSAGE_GATEWAY_SFID_BARRIER_MSG] = "barrier msg",
   [BRW_MESSAGE_GATEWAY_SFID_UPDATE_GATEWAY_STATE] = "update state",
   [BRW_MESSAGE_GATEWAY_SFID_MMIO_READ_WRITE] = "mmio read/write",
};

static const char *const dp_rc_msg_type_gfx9[16] = {
   [GFX9_DATAPORT_RC_RENDER_TARGET_WRITE] = "RT write",
   [GFX9_DATAPORT_RC_RENDER_TARGET_READ] = "RT read"
};

static const char *const *
dp_rc_msg_type(const struct intel_device_info *devinfo)
{
   return dp_rc_msg_type_gfx9;
}

static const char *const m_rt_write_subtype[] = {
   [0b000] = "SIMD16",
   [0b001] = "SIMD16/RepData",
   [0b010] = "SIMD8/DualSrcLow",
   [0b011] = "SIMD8/DualSrcHigh",
   [0b100] = "SIMD8",
   [0b101] = "SIMD8/ImageWrite",   /* Gfx6+ */
   [0b111] = "SIMD16/RepData-111", /* no idea how this is different than 1 */
};

static const char *const m_rt_write_subtype_xe2[] = {
   [0b000] = "SIMD16",
   [0b001] = "SIMD32",
   [0b010] = "SIMD16/DualSrc",
   [0b011] = "invalid",
   [0b100] = "invalid",
   [0b101] = "invalid",
   [0b111] = "invalid",
};

static const char *const dp_dc0_msg_type_gfx7[16] = {
   [GFX7_DATAPORT_DC_OWORD_BLOCK_READ] = "DC OWORD block read",
   [GFX7_DATAPORT_DC_UNALIGNED_OWORD_BLOCK_READ] =
      "DC unaligned OWORD block read",
   [GFX7_DATAPORT_DC_OWORD_DUAL_BLOCK_READ] = "DC OWORD dual block read",
   [GFX7_DATAPORT_DC_DWORD_SCATTERED_READ] = "DC DWORD scattered read",
   [GFX7_DATAPORT_DC_BYTE_SCATTERED_READ] = "DC byte scattered read",
   [GFX7_DATAPORT_DC_UNTYPED_SURFACE_READ] = "DC untyped surface read",
   [GFX7_DATAPORT_DC_UNTYPED_ATOMIC_OP] = "DC untyped atomic",
   [GFX7_DATAPORT_DC_MEMORY_FENCE] = "DC mfence",
   [GFX7_DATAPORT_DC_OWORD_BLOCK_WRITE] = "DC OWORD block write",
   [GFX7_DATAPORT_DC_OWORD_DUAL_BLOCK_WRITE] = "DC OWORD dual block write",
   [GFX7_DATAPORT_DC_DWORD_SCATTERED_WRITE] = "DC DWORD scatterd write",
   [GFX7_DATAPORT_DC_BYTE_SCATTERED_WRITE] = "DC byte scattered write",
   [GFX7_DATAPORT_DC_UNTYPED_SURFACE_WRITE] = "DC untyped surface write",
};

static const char *const dp_oword_block_rw[8] = {
      [BRW_DATAPORT_OWORD_BLOCK_1_OWORDLOW]  = "1-low",
      [BRW_DATAPORT_OWORD_BLOCK_1_OWORDHIGH] = "1-high",
      [BRW_DATAPORT_OWORD_BLOCK_2_OWORDS]    = "2",
      [BRW_DATAPORT_OWORD_BLOCK_4_OWORDS]    = "4",
      [BRW_DATAPORT_OWORD_BLOCK_8_OWORDS]    = "8",
};

static const char *const dp_dc1_msg_type_hsw[32] = {
   [HSW_DATAPORT_DC_PORT1_UNTYPED_SURFACE_READ] = "untyped surface read",
   [HSW_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_OP] = "DC untyped atomic op",
   [HSW_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_OP_SIMD4X2] =
      "DC untyped 4x2 atomic op",
   [HSW_DATAPORT_DC_PORT1_MEDIA_BLOCK_READ] = "DC media block read",
   [HSW_DATAPORT_DC_PORT1_TYPED_SURFACE_READ] = "DC typed surface read",
   [HSW_DATAPORT_DC_PORT1_TYPED_ATOMIC_OP] = "DC typed atomic",
   [HSW_DATAPORT_DC_PORT1_TYPED_ATOMIC_OP_SIMD4X2] = "DC typed 4x2 atomic op",
   [HSW_DATAPORT_DC_PORT1_UNTYPED_SURFACE_WRITE] = "DC untyped surface write",
   [HSW_DATAPORT_DC_PORT1_MEDIA_BLOCK_WRITE] = "DC media block write",
   [HSW_DATAPORT_DC_PORT1_ATOMIC_COUNTER_OP] = "DC atomic counter op",
   [HSW_DATAPORT_DC_PORT1_ATOMIC_COUNTER_OP_SIMD4X2] =
      "DC 4x2 atomic counter op",
   [HSW_DATAPORT_DC_PORT1_TYPED_SURFACE_WRITE] = "DC typed surface write",
   [GFX9_DATAPORT_DC_PORT1_A64_SCATTERED_READ] = "DC A64 scattered read",
   [GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_SURFACE_READ] = "DC A64 untyped surface read",
   [GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_OP] = "DC A64 untyped atomic op",
   [GFX9_DATAPORT_DC_PORT1_A64_OWORD_BLOCK_READ] = "DC A64 oword block read",
   [GFX9_DATAPORT_DC_PORT1_A64_OWORD_BLOCK_WRITE] = "DC A64 oword block write",
   [GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_SURFACE_WRITE] = "DC A64 untyped surface write",
   [GFX8_DATAPORT_DC_PORT1_A64_SCATTERED_WRITE] = "DC A64 scattered write",
   [GFX9_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_FLOAT_OP] =
      "DC untyped atomic float op",
   [GFX9_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_FLOAT_OP] =
      "DC A64 untyped atomic float op",
   [GFX12_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_HALF_INT_OP] =
      "DC A64 untyped atomic half-integer op",
   [GFX12_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_HALF_FLOAT_OP] =
      "DC A64 untyped atomic half-float op",
};

static const char *const aop[16] = {
   [BRW_AOP_AND]    = "and",
   [BRW_AOP_OR]     = "or",
   [BRW_AOP_XOR]    = "xor",
   [BRW_AOP_MOV]    = "mov",
   [BRW_AOP_INC]    = "inc",
   [BRW_AOP_DEC]    = "dec",
   [BRW_AOP_ADD]    = "add",
   [BRW_AOP_SUB]    = "sub",
   [BRW_AOP_REVSUB] = "revsub",
   [BRW_AOP_IMAX]   = "imax",
   [BRW_AOP_IMIN]   = "imin",
   [BRW_AOP_UMAX]   = "umax",
   [BRW_AOP_UMIN]   = "umin",
   [BRW_AOP_CMPWR]  = "cmpwr",
   [BRW_AOP_PREDEC] = "predec",
};

static const char *const aop_float[5] = {
   [BRW_AOP_FMAX]   = "fmax",
   [BRW_AOP_FMIN]   = "fmin",
   [BRW_AOP_FCMPWR] = "fcmpwr",
   [BRW_AOP_FADD]   = "fadd",
};

static const char * const pixel_interpolator_msg_types[4] = {
    [GFX7_PIXEL_INTERPOLATOR_LOC_SHARED_OFFSET] = "per_message_offset",
    [GFX7_PIXEL_INTERPOLATOR_LOC_SAMPLE] = "sample_position",
    [GFX7_PIXEL_INTERPOLATOR_LOC_CENTROID] = "centroid",
    [GFX7_PIXEL_INTERPOLATOR_LOC_PER_SLOT_OFFSET] = "per_slot_offset",
};

static const char *const math_function[16] = {
   [BRW_MATH_FUNCTION_INV]    = "inv",
   [BRW_MATH_FUNCTION_LOG]    = "log",
   [BRW_MATH_FUNCTION_EXP]    = "exp",
   [BRW_MATH_FUNCTION_SQRT]   = "sqrt",
   [BRW_MATH_FUNCTION_RSQ]    = "rsq",
   [BRW_MATH_FUNCTION_SIN]    = "sin",
   [BRW_MATH_FUNCTION_COS]    = "cos",
   [BRW_MATH_FUNCTION_FDIV]   = "fdiv",
   [BRW_MATH_FUNCTION_POW]    = "pow",
   [BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER] = "intdivmod",
   [BRW_MATH_FUNCTION_INT_DIV_QUOTIENT]  = "intdiv",
   [BRW_MATH_FUNCTION_INT_DIV_REMAINDER] = "intmod",
   [GFX8_MATH_FUNCTION_INVM]  = "invm",
   [GFX8_MATH_FUNCTION_RSQRTM] = "rsqrtm",
};

static const char *const sync_function[16] = {
   [TGL_SYNC_NOP] = "nop",
   [TGL_SYNC_ALLRD] = "allrd",
   [TGL_SYNC_ALLWR] = "allwr",
   [TGL_SYNC_FENCE] = "fence",
   [TGL_SYNC_BAR] = "bar",
   [TGL_SYNC_HOST] = "host",
};

static const char *const gfx7_urb_opcode[] = {
   [GFX7_URB_OPCODE_ATOMIC_MOV] = "atomic mov",  /* Gfx7+ */
   [GFX7_URB_OPCODE_ATOMIC_INC] = "atomic inc",  /* Gfx7+ */
   [GFX8_URB_OPCODE_ATOMIC_ADD] = "atomic add",  /* Gfx8+ */
   [GFX8_URB_OPCODE_SIMD8_WRITE] = "SIMD8 write", /* Gfx8+ */
   [GFX8_URB_OPCODE_SIMD8_READ] = "SIMD8 read",  /* Gfx8+ */
   [GFX125_URB_OPCODE_FENCE] = "fence",  /* Gfx12.5+ */
   /* [10-15] - reserved */
};

static const char *const urb_swizzle[4] = {
   [BRW_URB_SWIZZLE_NONE]       = "",
   [BRW_URB_SWIZZLE_INTERLEAVE] = "interleave",
   [BRW_URB_SWIZZLE_TRANSPOSE]  = "transpose",
};

static const char *const gfx5_sampler_msg_type[] = {
   [GFX5_SAMPLER_MESSAGE_SAMPLE]              = "sample",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_BIAS]         = "sample_b",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_LOD]          = "sample_l",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_COMPARE]      = "sample_c",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_DERIVS]       = "sample_d",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_BIAS_COMPARE] = "sample_b_c",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_LOD_COMPARE]  = "sample_l_c",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_LD]           = "ld",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_GATHER4]      = "gather4",
   [GFX5_SAMPLER_MESSAGE_LOD]                 = "lod",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_RESINFO]      = "resinfo",
   [GFX6_SAMPLER_MESSAGE_SAMPLE_SAMPLEINFO]   = "sampleinfo",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_GATHER4_C]    = "gather4_c",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_GATHER4_PO]   = "gather4_po",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_GATHER4_PO_C] = "gather4_po_c",
   [HSW_SAMPLER_MESSAGE_SAMPLE_DERIV_COMPARE] = "sample_d_c",
   [GFX9_SAMPLER_MESSAGE_SAMPLE_LZ]           = "sample_lz",
   [GFX9_SAMPLER_MESSAGE_SAMPLE_C_LZ]         = "sample_c_lz",
   [GFX9_SAMPLER_MESSAGE_SAMPLE_LD_LZ]        = "ld_lz",
   [GFX9_SAMPLER_MESSAGE_SAMPLE_LD2DMS_W]     = "ld2dms_w",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_LD_MCS]       = "ld_mcs",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_LD2DMS]       = "ld2dms",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_LD2DSS]       = "ld2dss",
};

static const char *const xe2_sampler_msg_type[] = {
   [GFX5_SAMPLER_MESSAGE_SAMPLE]              = "sample",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_BIAS]         = "sample_b",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_LOD]          = "sample_l",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_COMPARE]      = "sample_c",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_DERIVS]       = "sample_d",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_BIAS_COMPARE] = "sample_b_c",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_LOD_COMPARE]  = "sample_l_c",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_LD]           = "ld",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_GATHER4]      = "gather4",
   [GFX5_SAMPLER_MESSAGE_LOD]                 = "lod",
   [GFX5_SAMPLER_MESSAGE_SAMPLE_RESINFO]      = "resinfo",
   [GFX6_SAMPLER_MESSAGE_SAMPLE_SAMPLEINFO]   = "sampleinfo",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_GATHER4_C]    = "gather4_c",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_GATHER4_PO]   = "gather4_po",
   [XE2_SAMPLER_MESSAGE_SAMPLE_MLOD]          = "sample_mlod",
   [XE2_SAMPLER_MESSAGE_SAMPLE_COMPARE_MLOD]  = "sample_c_mlod",
   [XE2_SAMPLER_MESSAGE_SAMPLE_GATHER4_I]    = "gather4_i",
   [XE2_SAMPLER_MESSAGE_SAMPLE_GATHER4_L]    = "gather4_l",
   [XE2_SAMPLER_MESSAGE_SAMPLE_GATHER4_B]    = "gather4_b",
   [XE2_SAMPLER_MESSAGE_SAMPLE_GATHER4_I_C]  = "gather4_i_c",
   [XE2_SAMPLER_MESSAGE_SAMPLE_GATHER4_L_C]  = "gather4_l_c",
   [XE2_SAMPLER_MESSAGE_SAMPLE_GATHER4_PO_L] = "gather4_po_l",
   [XE2_SAMPLER_MESSAGE_SAMPLE_GATHER4_PO_L_C] = "gather4_po_l_c",
   [XE2_SAMPLER_MESSAGE_SAMPLE_GATHER4_PO_B]  = "gather4_po_b",
   [HSW_SAMPLER_MESSAGE_SAMPLE_DERIV_COMPARE] = "sample_d_c",
   [GFX9_SAMPLER_MESSAGE_SAMPLE_LZ]           = "sample_lz",
   [GFX9_SAMPLER_MESSAGE_SAMPLE_C_LZ]         = "sample_c_lz",
   [GFX9_SAMPLER_MESSAGE_SAMPLE_LD_LZ]        = "ld_lz",
   [GFX9_SAMPLER_MESSAGE_SAMPLE_LD2DMS_W]     = "ld2dms_w",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_LD_MCS]       = "ld_mcs",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_LD2DMS]       = "ld2dms",
   [GFX7_SAMPLER_MESSAGE_SAMPLE_LD2DSS]       = "ld2dss",
};

static const char *const gfx5_sampler_simd_mode[7] = {
   [BRW_SAMPLER_SIMD_MODE_SIMD4X2]   = "SIMD4x2",
   [BRW_SAMPLER_SIMD_MODE_SIMD8]     = "SIMD8",
   [BRW_SAMPLER_SIMD_MODE_SIMD16]    = "SIMD16",
   [BRW_SAMPLER_SIMD_MODE_SIMD32_64] = "SIMD32/64",
   [GFX10_SAMPLER_SIMD_MODE_SIMD8H]  = "SIMD8H",
   [GFX10_SAMPLER_SIMD_MODE_SIMD16H] = "SIMD16H",
};

static const char *const xe2_sampler_simd_mode[7] = {
   [XE2_SAMPLER_SIMD_MODE_SIMD16]  = "SIMD16",
   [XE2_SAMPLER_SIMD_MODE_SIMD32]  = "SIMD32",
   [XE2_SAMPLER_SIMD_MODE_SIMD16H] = "SIMD16H",
   [XE2_SAMPLER_SIMD_MODE_SIMD32H] = "SIMD32H",
};

static const char *const lsc_operation[] = {
   [LSC_OP_LOAD]            = "load",
   [LSC_OP_LOAD_CMASK]      = "load_cmask",
   [LSC_OP_STORE]           = "store",
   [LSC_OP_STORE_CMASK]     = "store_cmask",
   [LSC_OP_FENCE]           = "fence",
   [LSC_OP_ATOMIC_INC]      = "atomic_inc",
   [LSC_OP_ATOMIC_DEC]      = "atomic_dec",
   [LSC_OP_ATOMIC_LOAD]     = "atomic_load",
   [LSC_OP_ATOMIC_STORE]    = "atomic_store",
   [LSC_OP_ATOMIC_ADD]      = "atomic_add",
   [LSC_OP_ATOMIC_SUB]      = "atomic_sub",
   [LSC_OP_ATOMIC_MIN]      = "atomic_min",
   [LSC_OP_ATOMIC_MAX]      = "atomic_max",
   [LSC_OP_ATOMIC_UMIN]     = "atomic_umin",
   [LSC_OP_ATOMIC_UMAX]     = "atomic_umax",
   [LSC_OP_ATOMIC_CMPXCHG]  = "atomic_cmpxchg",
   [LSC_OP_ATOMIC_FADD]     = "atomic_fadd",
   [LSC_OP_ATOMIC_FSUB]     = "atomic_fsub",
   [LSC_OP_ATOMIC_FMIN]     = "atomic_fmin",
   [LSC_OP_ATOMIC_FMAX]     = "atomic_fmax",
   [LSC_OP_ATOMIC_FCMPXCHG] = "atomic_fcmpxchg",
   [LSC_OP_ATOMIC_AND]      = "atomic_and",
   [LSC_OP_ATOMIC_OR]       = "atomic_or",
   [LSC_OP_ATOMIC_XOR]      = "atomic_xor",
   [LSC_OP_LOAD_CMASK_MSRT] = "load_cmask_msrt",
   [LSC_OP_STORE_CMASK_MSRT] = "store_cmask_msrt",
};

const char *
brw_lsc_op_to_string(unsigned op)
{
   assert(op < ARRAY_SIZE(lsc_operation));
   return lsc_operation[op];
}

static const char *const lsc_addr_surface_type[] = {
   [LSC_ADDR_SURFTYPE_FLAT] = "flat",
   [LSC_ADDR_SURFTYPE_BSS]  = "bss",
   [LSC_ADDR_SURFTYPE_SS]   = "ss",
   [LSC_ADDR_SURFTYPE_BTI]  = "bti",
};

const char *
brw_lsc_addr_surftype_to_string(unsigned t)
{
   assert(t < ARRAY_SIZE(lsc_addr_surface_type));
   return lsc_addr_surface_type[t];
}

static const char* const lsc_fence_scope[] = {
   [LSC_FENCE_THREADGROUP]     = "threadgroup",
   [LSC_FENCE_LOCAL]           = "local",
   [LSC_FENCE_TILE]            = "tile",
   [LSC_FENCE_GPU]             = "gpu",
   [LSC_FENCE_ALL_GPU]         = "all_gpu",
   [LSC_FENCE_SYSTEM_RELEASE]  = "system_release",
   [LSC_FENCE_SYSTEM_ACQUIRE]  = "system_acquire",
};

static const char* const lsc_flush_type[] = {
   [LSC_FLUSH_TYPE_NONE]       = "none",
   [LSC_FLUSH_TYPE_EVICT]      = "evict",
   [LSC_FLUSH_TYPE_INVALIDATE] = "invalidate",
   [LSC_FLUSH_TYPE_DISCARD]    = "discard",
   [LSC_FLUSH_TYPE_CLEAN]      = "clean",
   [LSC_FLUSH_TYPE_L3ONLY]     = "l3only",
   [LSC_FLUSH_TYPE_NONE_6]     = "none_6",
};

static const char* const lsc_addr_size[] = {
   [LSC_ADDR_SIZE_A16] = "a16",
   [LSC_ADDR_SIZE_A32] = "a32",
   [LSC_ADDR_SIZE_A64] = "a64",
};

static const char* const lsc_backup_fence_routing[] = {
   [LSC_NORMAL_ROUTING]  = "normal_routing",
   [LSC_ROUTE_TO_LSC]    = "route_to_lsc",
};

static const char* const lsc_data_size[] = {
   [LSC_DATA_SIZE_D8]      = "d8",
   [LSC_DATA_SIZE_D16]     = "d16",
   [LSC_DATA_SIZE_D32]     = "d32",
   [LSC_DATA_SIZE_D64]     = "d64",
   [LSC_DATA_SIZE_D8U32]   = "d8u32",
   [LSC_DATA_SIZE_D16U32]  = "d16u32",
   [LSC_DATA_SIZE_D16BF32] = "d16bf32",
};

const char *
brw_lsc_data_size_to_string(unsigned s)
{
   assert(s < ARRAY_SIZE(lsc_data_size));
   return lsc_data_size[s];
}

static const char* const lsc_vect_size_str[] = {
   [LSC_VECT_SIZE_V1] = "V1",
   [LSC_VECT_SIZE_V2] = "V2",
   [LSC_VECT_SIZE_V3] = "V3",
   [LSC_VECT_SIZE_V4] = "V4",
   [LSC_VECT_SIZE_V8] = "V8",
   [LSC_VECT_SIZE_V16] = "V16",
   [LSC_VECT_SIZE_V32] = "V32",
   [LSC_VECT_SIZE_V64] = "V64",
};

static const char* const lsc_cmask_str[] = {
   [LSC_CMASK_X]      = "x",
   [LSC_CMASK_Y]      = "y",
   [LSC_CMASK_XY]     = "xy",
   [LSC_CMASK_Z]      = "z",
   [LSC_CMASK_XZ]     = "xz",
   [LSC_CMASK_YZ]     = "yz",
   [LSC_CMASK_XYZ]    = "xyz",
   [LSC_CMASK_W]      = "w",
   [LSC_CMASK_XW]     = "xw",
   [LSC_CMASK_YW]     = "yw",
   [LSC_CMASK_XYW]    = "xyw",
   [LSC_CMASK_ZW]     = "zw",
   [LSC_CMASK_XZW]    = "xzw",
   [LSC_CMASK_YZW]    = "yzw",
   [LSC_CMASK_XYZW]   = "xyzw",
};

static const char* const lsc_cache_load[] = {
   [LSC_CACHE_LOAD_L1STATE_L3MOCS]   = "L1STATE_L3MOCS",
   [LSC_CACHE_LOAD_L1UC_L3UC]        = "L1UC_L3UC",
   [LSC_CACHE_LOAD_L1UC_L3C]         = "L1UC_L3C",
   [LSC_CACHE_LOAD_L1C_L3UC]         = "L1C_L3UC",
   [LSC_CACHE_LOAD_L1C_L3C]          = "L1C_L3C",
   [LSC_CACHE_LOAD_L1S_L3UC]         = "L1S_L3UC",
   [LSC_CACHE_LOAD_L1S_L3C]          = "L1S_L3C",
   [LSC_CACHE_LOAD_L1IAR_L3C]        = "L1IAR_L3C",
};

static const char* const lsc_cache_store[] = {
   [LSC_CACHE_STORE_L1STATE_L3MOCS]  = "L1STATE_L3MOCS",
   [LSC_CACHE_STORE_L1UC_L3UC]       = "L1UC_L3UC",
   [LSC_CACHE_STORE_L1UC_L3WB]       = "L1UC_L3WB",
   [LSC_CACHE_STORE_L1WT_L3UC]       = "L1WT_L3UC",
   [LSC_CACHE_STORE_L1WT_L3WB]       = "L1WT_L3WB",
   [LSC_CACHE_STORE_L1S_L3UC]        = "L1S_L3UC",
   [LSC_CACHE_STORE_L1S_L3WB]        = "L1S_L3WB",
   [LSC_CACHE_STORE_L1WB_L3WB]       = "L1WB_L3WB",
};

static const char* const xe2_lsc_cache_load[] = {
   [XE2_LSC_CACHE_LOAD_L1STATE_L3MOCS]   = "L1STATE_L3MOCS",
   [XE2_LSC_CACHE_LOAD_L1UC_L3UC]        = "L1UC_L3UC",
   [XE2_LSC_CACHE_LOAD_L1UC_L3C]         = "L1UC_L3C",
   [XE2_LSC_CACHE_LOAD_L1UC_L3CC]        = "L1UC_L3CC",
   [XE2_LSC_CACHE_LOAD_L1C_L3UC]         = "L1C_L3UC",
   [XE2_LSC_CACHE_LOAD_L1C_L3C]          = "L1C_L3C",
   [XE2_LSC_CACHE_LOAD_L1C_L3CC]         = "L1C_L3CC",
   [XE2_LSC_CACHE_LOAD_L1S_L3UC]         = "L1S_L3UC",
   [XE2_LSC_CACHE_LOAD_L1S_L3C]          = "L1S_L3C",
   [XE2_LSC_CACHE_LOAD_L1IAR_L3IAR]      = "L1IAR_L3IAR",
};

static const char* const xe2_lsc_cache_store[] = {
   [XE2_LSC_CACHE_STORE_L1STATE_L3MOCS]  = "L1STATE_L3MOCS",
   [XE2_LSC_CACHE_STORE_L1UC_L3UC]       = "L1UC_L3UC",
   [XE2_LSC_CACHE_STORE_L1UC_L3WB]       = "L1UC_L3WB",
   [XE2_LSC_CACHE_STORE_L1WT_L3UC]       = "L1WT_L3UC",
   [XE2_LSC_CACHE_STORE_L1WT_L3WB]       = "L1WT_L3WB",
   [XE2_LSC_CACHE_STORE_L1S_L3UC]        = "L1S_L3UC",
   [XE2_LSC_CACHE_STORE_L1S_L3WB]        = "L1S_L3WB",
   [XE2_LSC_CACHE_STORE_L1WB_L3WB]       = "L1WB_L3WB",
};

static const char* const dpas_systolic_depth[4] = {
   [0] = "16",
   [1] = "2",
   [2] = "4",
   [3] = "8"
};

static int column;

static int
string(FILE *file, const char *string)
{
   fputs(string, file);
   column += strlen(string);
   return 0;
}

static int
format(FILE *f, const char *format, ...) PRINTFLIKE(2, 3);

static int
format(FILE *f, const char *format, ...)
{
   char buf[1024];
   va_list args;
   va_start(args, format);

   vsnprintf(buf, sizeof(buf) - 1, format, args);
   va_end(args);
   string(f, buf);
   return 0;
}

static int
newline(FILE *f)
{
   putc('\n', f);
   column = 0;
   return 0;
}

static int
pad(FILE *f, int c)
{
   do
      string(f, " ");
   while (column < c);
   return 0;
}

static int
control(FILE *file, const char *name, const char *const ctrl[],
        unsigned id, int *space)
{
   if (!ctrl[id]) {
      fprintf(file, "*** invalid %s value %d ", name, id);
      return 1;
   }
   if (ctrl[id][0]) {
      if (space && *space)
         string(file, " ");
      string(file, ctrl[id]);
      if (space)
         *space = 1;
   }
   return 0;
}

static int
print_opcode(FILE *file, const struct brw_isa_info *isa,
             enum opcode id)
{
   const struct opcode_desc *desc = brw_opcode_desc(isa, id);
   if (!desc) {
      format(file, "*** invalid opcode value %d ", id);
      return 1;
   }
   string(file, desc->name);
   return 0;
}

static int
reg(FILE *file, unsigned _reg_file, unsigned _reg_nr)
{
   int err = 0;

   if (_reg_file == ARF) {
      switch (_reg_nr & 0xf0) {
      case BRW_ARF_NULL:
         string(file, "null");
         break;
      case BRW_ARF_ADDRESS:
         format(file, "a%d", _reg_nr & 0x0f);
         break;
      case BRW_ARF_ACCUMULATOR:
         format(file, "acc%d", _reg_nr & 0x0f);
         break;
      case BRW_ARF_FLAG:
         format(file, "f%d", _reg_nr & 0x0f);
         break;
      case BRW_ARF_MASK:
         format(file, "mask%d", _reg_nr & 0x0f);
         break;
      case BRW_ARF_STATE:
         format(file, "sr%d", _reg_nr & 0x0f);
         break;
      case BRW_ARF_SCALAR:
         format(file, "s%d", _reg_nr & 0x0f);
         break;
      case BRW_ARF_CONTROL:
         format(file, "cr%d", _reg_nr & 0x0f);
         break;
      case BRW_ARF_NOTIFICATION_COUNT:
         format(file, "n%d", _reg_nr & 0x0f);
         break;
      case BRW_ARF_IP:
         string(file, "ip");
         return -1;
         break;
      case BRW_ARF_TDR:
         format(file, "tdr0");
         return -1;
      case BRW_ARF_TIMESTAMP:
         format(file, "tm%d", _reg_nr & 0x0f);
         break;
      default:
         format(file, "ARF%d", _reg_nr);
         break;
      }
   } else {
      err |= control(file, "src reg file", reg_file, _reg_file, NULL);
      format(file, "%d", _reg_nr);
   }
   return err;
}

static int
dest(FILE *file, const struct brw_isa_info *isa, const brw_eu_inst *inst)
{
   const struct intel_device_info *devinfo = isa->devinfo;
   enum brw_reg_type type = brw_eu_inst_dst_type(devinfo, inst);
   unsigned elem_size = brw_type_size_bytes(type);
   int err = 0;

   if (is_split_send(devinfo, brw_eu_inst_opcode(isa, inst))) {
      /* These are fixed for split sends */
      type = BRW_TYPE_UD;
      elem_size = 4;
      if (devinfo->ver >= 12) {
         err |= reg(file, brw_eu_inst_send_dst_reg_file(devinfo, inst),
                    brw_eu_inst_dst_da_reg_nr(devinfo, inst));
         string(file, brw_reg_type_to_letters(type));
      } else if (brw_eu_inst_dst_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         err |= reg(file, brw_eu_inst_send_dst_reg_file(devinfo, inst),
                    brw_eu_inst_dst_da_reg_nr(devinfo, inst));
         unsigned subreg_nr = brw_eu_inst_dst_da16_subreg_nr(devinfo, inst);
         if (subreg_nr)
            format(file, ".%u", subreg_nr);
         string(file, brw_reg_type_to_letters(type));
      } else {
         string(file, "g[a0");
         if (brw_eu_inst_dst_ia_subreg_nr(devinfo, inst))
            format(file, ".%"PRIu64, brw_eu_inst_dst_ia_subreg_nr(devinfo, inst) /
                   elem_size);
         if (brw_eu_inst_send_dst_ia16_addr_imm(devinfo, inst))
            format(file, " %d", brw_eu_inst_send_dst_ia16_addr_imm(devinfo, inst));
         string(file, "]<");
         string(file, brw_reg_type_to_letters(type));
      }
   } else if (brw_eu_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
      if (brw_eu_inst_dst_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         err |= reg(file, brw_eu_inst_dst_reg_file(devinfo, inst),
                    brw_eu_inst_dst_da_reg_nr(devinfo, inst));
         if (err == -1)
            return 0;
         if (brw_eu_inst_dst_da1_subreg_nr(devinfo, inst))
            format(file, ".%"PRIu64, brw_eu_inst_dst_da1_subreg_nr(devinfo, inst) /
                   elem_size);
         string(file, "<");
         err |= control(file, "horiz stride", horiz_stride,
                        brw_eu_inst_dst_hstride(devinfo, inst), NULL);
         string(file, ">");
         string(file, brw_reg_type_to_letters(type));
      } else {
         string(file, "g[a0");
         if (brw_eu_inst_dst_ia_subreg_nr(devinfo, inst))
            format(file, ".%"PRIu64, brw_eu_inst_dst_ia_subreg_nr(devinfo, inst) /
                   elem_size);
         if (brw_eu_inst_dst_ia1_addr_imm(devinfo, inst))
            format(file, " %d", brw_eu_inst_dst_ia1_addr_imm(devinfo, inst));
         string(file, "]<");
         err |= control(file, "horiz stride", horiz_stride,
                        brw_eu_inst_dst_hstride(devinfo, inst), NULL);
         string(file, ">");
         string(file, brw_reg_type_to_letters(type));
      }
   } else {
      if (brw_eu_inst_dst_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         err |= reg(file, brw_eu_inst_dst_reg_file(devinfo, inst),
                    brw_eu_inst_dst_da_reg_nr(devinfo, inst));
         if (err == -1)
            return 0;
         if (brw_eu_inst_dst_da16_subreg_nr(devinfo, inst))
            format(file, ".%u", 16 / elem_size);
         string(file, "<1>");
         err |= control(file, "writemask", writemask,
                        brw_eu_inst_da16_writemask(devinfo, inst), NULL);
         string(file, brw_reg_type_to_letters(type));
      } else {
         err = 1;
         string(file, "Indirect align16 address mode not supported");
      }
   }

   return 0;
}

static enum brw_horizontal_stride
hstride_from_align1_3src_dst_hstride(enum brw_align1_3src_dst_horizontal_stride hstride)
{
   switch (hstride) {
   case BRW_ALIGN1_3SRC_DST_HORIZONTAL_STRIDE_1: return BRW_HORIZONTAL_STRIDE_1;
   case BRW_ALIGN1_3SRC_DST_HORIZONTAL_STRIDE_2: return BRW_HORIZONTAL_STRIDE_2;
   default:
      UNREACHABLE("not reached");
   }
}

static int
dest_3src(FILE *file, const struct intel_device_info *devinfo,
          const brw_eu_inst *inst)
{
   bool is_align1 = brw_eu_inst_3src_access_mode(devinfo, inst) == BRW_ALIGN_1;
   int err = 0;
   uint32_t reg_file;
   unsigned subreg_nr;
   enum brw_reg_type type;

   if (devinfo->ver < 10 && is_align1)
      return 0;

   if (devinfo->ver >= 12 || is_align1)
      reg_file = brw_eu_inst_3src_a1_dst_reg_file(devinfo, inst);
   else
      reg_file = FIXED_GRF;

   err |= reg(file, reg_file, brw_eu_inst_3src_dst_reg_nr(devinfo, inst));
   if (err == -1)
      return 0;

   if (is_align1) {
      type = brw_eu_inst_3src_a1_dst_type(devinfo, inst);
      subreg_nr = brw_eu_inst_3src_a1_dst_subreg_nr(devinfo, inst);
   } else {
      type = brw_eu_inst_3src_a16_dst_type(devinfo, inst);
      subreg_nr = brw_eu_inst_3src_a16_dst_subreg_nr(devinfo, inst);
   }
   subreg_nr /= brw_type_size_bytes(type);

   if (subreg_nr)
      format(file, ".%u", subreg_nr);
   string(file, "<");
   unsigned _horiz_stride = devinfo->ver == 9 ? BRW_HORIZONTAL_STRIDE_1 :
      hstride_from_align1_3src_dst_hstride(brw_eu_inst_3src_a1_dst_hstride(devinfo, inst));
   err |= control(file, "horiz_stride", horiz_stride, _horiz_stride, NULL);
   string(file, ">");

   if (!is_align1) {
      err |= control(file, "writemask", writemask,
                     brw_eu_inst_3src_a16_dst_writemask(devinfo, inst), NULL);
   }
   string(file, brw_reg_type_to_letters(type));

   return 0;
}

static int
dest_dpas_3src(FILE *file, const struct intel_device_info *devinfo,
               const brw_eu_inst *inst)
{
   uint32_t reg_file = brw_eu_inst_dpas_3src_dst_reg_file(devinfo, inst);

   if (reg(file, reg_file, brw_eu_inst_dpas_3src_dst_reg_nr(devinfo, inst)) == -1)
      return 0;

   enum brw_reg_type type = brw_eu_inst_dpas_3src_dst_type(devinfo, inst);
   unsigned subreg_nr = brw_eu_inst_dpas_3src_dst_subreg_nr(devinfo, inst);

   if (subreg_nr)
      format(file, ".%u", subreg_nr);
   string(file, "<1>");

   string(file, brw_reg_type_to_letters(type));

   return 0;
}

static int
src_align1_region(FILE *file,
                  unsigned _vert_stride, unsigned _width,
                  unsigned _horiz_stride)
{
   int err = 0;
   string(file, "<");
   err |= control(file, "vert stride", vert_stride, _vert_stride, NULL);
   string(file, ",");
   err |= control(file, "width", width, _width, NULL);
   string(file, ",");
   err |= control(file, "horiz_stride", horiz_stride, _horiz_stride, NULL);
   string(file, ">");
   return err;
}

static int
src_da1(FILE *file,
        const struct intel_device_info *devinfo,
        unsigned opcode,
        enum brw_reg_type type, unsigned _reg_file,
        unsigned _vert_stride, unsigned _width, unsigned _horiz_stride,
        unsigned reg_num, unsigned sub_reg_num, unsigned __abs,
        unsigned _negate)
{
   int err = 0;

   if (is_logic_instruction(opcode))
      err |= control(file, "bitnot", m_bitnot, _negate, NULL);
   else
      err |= control(file, "negate", m_negate, _negate, NULL);

   err |= control(file, "abs", _abs, __abs, NULL);

   err |= reg(file, _reg_file, reg_num);
   if (err == -1)
      return 0;
   if (sub_reg_num) {
      unsigned elem_size = brw_type_size_bytes(type);
      format(file, ".%d", sub_reg_num / elem_size);   /* use formal style like spec */
   }
   src_align1_region(file, _vert_stride, _width, _horiz_stride);
   string(file, brw_reg_type_to_letters(type));
   return err;
}

static int
src_ia1(FILE *file,
        const struct intel_device_info *devinfo,
        unsigned opcode,
        enum brw_reg_type type,
        int _addr_imm,
        unsigned _addr_subreg_nr,
        unsigned _negate,
        unsigned __abs,
        unsigned _horiz_stride, unsigned _width, unsigned _vert_stride)
{
   int err = 0;

   if (is_logic_instruction(opcode))
      err |= control(file, "bitnot", m_bitnot, _negate, NULL);
   else
      err |= control(file, "negate", m_negate, _negate, NULL);

   err |= control(file, "abs", _abs, __abs, NULL);

   string(file, "g[a0");
   if (_addr_subreg_nr)
      format(file, ".%d", _addr_subreg_nr);
   if (_addr_imm)
      format(file, " %d", _addr_imm);
   string(file, "]");
   src_align1_region(file, _vert_stride, _width, _horiz_stride);
   string(file, brw_reg_type_to_letters(type));
   return err;
}

static int
src_swizzle(FILE *file, unsigned swiz)
{
   unsigned x = BRW_GET_SWZ(swiz, BRW_CHANNEL_X);
   unsigned y = BRW_GET_SWZ(swiz, BRW_CHANNEL_Y);
   unsigned z = BRW_GET_SWZ(swiz, BRW_CHANNEL_Z);
   unsigned w = BRW_GET_SWZ(swiz, BRW_CHANNEL_W);
   int err = 0;

   if (x == y && x == z && x == w) {
      string(file, ".");
      err |= control(file, "channel select", chan_sel, x, NULL);
   } else if (swiz != BRW_SWIZZLE_XYZW) {
      string(file, ".");
      err |= control(file, "channel select", chan_sel, x, NULL);
      err |= control(file, "channel select", chan_sel, y, NULL);
      err |= control(file, "channel select", chan_sel, z, NULL);
      err |= control(file, "channel select", chan_sel, w, NULL);
   }
   return err;
}

static int
src_da16(FILE *file,
         const struct intel_device_info *devinfo,
         unsigned opcode,
         enum brw_reg_type type,
         unsigned _reg_file,
         unsigned _vert_stride,
         unsigned _reg_nr,
         unsigned _subreg_nr,
         unsigned __abs,
         unsigned _negate,
         unsigned swz_x, unsigned swz_y, unsigned swz_z, unsigned swz_w)
{
   int err = 0;

   if (is_logic_instruction(opcode))
      err |= control(file, "bitnot", m_bitnot, _negate, NULL);
   else
      err |= control(file, "negate", m_negate, _negate, NULL);

   err |= control(file, "abs", _abs, __abs, NULL);

   err |= reg(file, _reg_file, _reg_nr);
   if (err == -1)
      return 0;
   if (_subreg_nr) {
      unsigned elem_size = brw_type_size_bytes(type);

      /* bit4 for subreg number byte addressing. Make this same meaning as
         in da1 case, so output looks consistent. */
      format(file, ".%d", 16 / elem_size);
   }
   string(file, "<");
   err |= control(file, "vert stride", vert_stride, _vert_stride, NULL);
   string(file, ">");
   err |= src_swizzle(file, BRW_SWIZZLE4(swz_x, swz_y, swz_z, swz_w));
   string(file, brw_reg_type_to_letters(type));
   return err;
}

static enum brw_vertical_stride
vstride_from_align1_3src_vstride(const struct intel_device_info *devinfo,
                                 enum brw_align1_3src_vertical_stride vstride)
{
   switch (vstride) {
   case BRW_ALIGN1_3SRC_VERTICAL_STRIDE_0: return BRW_VERTICAL_STRIDE_0;
   case BRW_ALIGN1_3SRC_VERTICAL_STRIDE_2:
      if (devinfo->ver >= 12)
         return BRW_VERTICAL_STRIDE_1;
      else
         return BRW_VERTICAL_STRIDE_2;
   case BRW_ALIGN1_3SRC_VERTICAL_STRIDE_4: return BRW_VERTICAL_STRIDE_4;
   case BRW_ALIGN1_3SRC_VERTICAL_STRIDE_8: return BRW_VERTICAL_STRIDE_8;
   default:
      UNREACHABLE("not reached");
   }
}

static enum brw_horizontal_stride
hstride_from_align1_3src_hstride(enum brw_align1_3src_src_horizontal_stride hstride)
{
   switch (hstride) {
   case BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_0: return BRW_HORIZONTAL_STRIDE_0;
   case BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_1: return BRW_HORIZONTAL_STRIDE_1;
   case BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_2: return BRW_HORIZONTAL_STRIDE_2;
   case BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_4: return BRW_HORIZONTAL_STRIDE_4;
   default:
      UNREACHABLE("not reached");
   }
}

static enum brw_vertical_stride
vstride_from_align1_3src_hstride(enum brw_align1_3src_src_horizontal_stride hstride)
{
   switch (hstride) {
   case BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_0: return BRW_VERTICAL_STRIDE_0;
   case BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_1: return BRW_VERTICAL_STRIDE_1;
   case BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_2: return BRW_VERTICAL_STRIDE_2;
   case BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_4: return BRW_VERTICAL_STRIDE_4;
   default:
      UNREACHABLE("not reached");
   }
}

/* From "GFX10 Regioning Rules for Align1 Ternary Operations" in the
 * "Register Region Restrictions" documentation
 */
static enum brw_width
implied_width(enum brw_vertical_stride _vert_stride,
              enum brw_horizontal_stride _horiz_stride)
{
   /* "1. Width is 1 when Vertical and Horizontal Strides are both zero." */
   if (_vert_stride == BRW_VERTICAL_STRIDE_0 &&
       _horiz_stride == BRW_HORIZONTAL_STRIDE_0) {
      return BRW_WIDTH_1;

   /* "2. Width is equal to vertical stride when Horizontal Stride is zero." */
   } else if (_horiz_stride == BRW_HORIZONTAL_STRIDE_0) {
      switch (_vert_stride) {
      case BRW_VERTICAL_STRIDE_1: return BRW_WIDTH_1;
      case BRW_VERTICAL_STRIDE_2: return BRW_WIDTH_2;
      case BRW_VERTICAL_STRIDE_4: return BRW_WIDTH_4;
      case BRW_VERTICAL_STRIDE_8: return BRW_WIDTH_8;
      case BRW_VERTICAL_STRIDE_0:
      default:
         UNREACHABLE("not reached");
      }

   } else {
      /* FINISHME: Implement these: */

      /* "3. Width is equal to Vertical Stride/Horizontal Stride when both
       *     Strides are non-zero.
       *
       *  4. Vertical Stride must not be zero if Horizontal Stride is non-zero.
       *     This implies Vertical Stride is always greater than Horizontal
       *     Stride."
       *
       * Given these statements and the knowledge that the stride and width
       * values are encoded in logarithmic form, we can perform the division
       * by just subtracting.
       */
      return _vert_stride - _horiz_stride;
   }
}

static int
src0_3src(FILE *file, const struct intel_device_info *devinfo,
          const brw_eu_inst *inst)
{
   int err = 0;
   unsigned reg_nr, subreg_nr;
   enum brw_reg_file _file;
   enum brw_reg_type type;
   enum brw_vertical_stride _vert_stride;
   enum brw_width _width;
   enum brw_horizontal_stride _horiz_stride;
   bool is_scalar_region;
   bool is_align1 = brw_eu_inst_3src_access_mode(devinfo, inst) == BRW_ALIGN_1;

   if (devinfo->ver < 10 && is_align1)
      return 0;

   if (is_align1) {
      _file = brw_eu_inst_3src_a1_src0_reg_file(devinfo, inst);
      if (_file == IMM) {
         uint16_t imm_val = brw_eu_inst_3src_a1_src0_imm(devinfo, inst);
         enum brw_reg_type type = brw_eu_inst_3src_a1_src0_type(devinfo, inst);

         if (type == BRW_TYPE_W) {
            format(file, "%dW", imm_val);
         } else if (type == BRW_TYPE_UW) {
            format(file, "0x%04xUW", imm_val);
         } else if (type == BRW_TYPE_HF) {
            format(file, "0x%04xHF", imm_val);
         }
         return 0;
      }

      reg_nr = brw_eu_inst_3src_src0_reg_nr(devinfo, inst);
      subreg_nr = brw_eu_inst_3src_a1_src0_subreg_nr(devinfo, inst);
      type = brw_eu_inst_3src_a1_src0_type(devinfo, inst);
      _vert_stride = vstride_from_align1_3src_vstride(
         devinfo, brw_eu_inst_3src_a1_src0_vstride(devinfo, inst));
      _horiz_stride = hstride_from_align1_3src_hstride(
                         brw_eu_inst_3src_a1_src0_hstride(devinfo, inst));
      _width = implied_width(_vert_stride, _horiz_stride);
   } else {
      _file = FIXED_GRF;
      reg_nr = brw_eu_inst_3src_src0_reg_nr(devinfo, inst);
      subreg_nr = brw_eu_inst_3src_a16_src0_subreg_nr(devinfo, inst);
      type = brw_eu_inst_3src_a16_src_type(devinfo, inst);

      if (brw_eu_inst_3src_a16_src0_rep_ctrl(devinfo, inst)) {
         _vert_stride = BRW_VERTICAL_STRIDE_0;
         _width = BRW_WIDTH_1;
         _horiz_stride = BRW_HORIZONTAL_STRIDE_0;
      } else {
         _vert_stride = BRW_VERTICAL_STRIDE_4;
         _width = BRW_WIDTH_4;
         _horiz_stride = BRW_HORIZONTAL_STRIDE_1;
      }
   }
   is_scalar_region = _vert_stride == BRW_VERTICAL_STRIDE_0 &&
                      _width == BRW_WIDTH_1 &&
                      _horiz_stride == BRW_HORIZONTAL_STRIDE_0;

   subreg_nr /= brw_type_size_bytes(type);

   err |= control(file, "negate", m_negate,
                  brw_eu_inst_3src_src0_negate(devinfo, inst), NULL);
   err |= control(file, "abs", _abs, brw_eu_inst_3src_src0_abs(devinfo, inst), NULL);

   err |= reg(file, _file, reg_nr);
   if (err == -1)
      return 0;
   if (subreg_nr || is_scalar_region)
      format(file, ".%d", subreg_nr);
   src_align1_region(file, _vert_stride, _width, _horiz_stride);
   if (!is_scalar_region && !is_align1)
      err |= src_swizzle(file, brw_eu_inst_3src_a16_src0_swizzle(devinfo, inst));
   string(file, brw_reg_type_to_letters(type));
   return err;
}

static int
src1_3src(FILE *file, const struct intel_device_info *devinfo,
          const brw_eu_inst *inst)
{
   int err = 0;
   unsigned reg_nr, subreg_nr;
   enum brw_reg_file _file;
   enum brw_reg_type type;
   enum brw_vertical_stride _vert_stride;
   enum brw_width _width;
   enum brw_horizontal_stride _horiz_stride;
   bool is_scalar_region;
   bool is_align1 = brw_eu_inst_3src_access_mode(devinfo, inst) == BRW_ALIGN_1;

   if (devinfo->ver < 10 && is_align1)
      return 0;

   if (is_align1) {
      _file = brw_eu_inst_3src_a1_src1_reg_file(devinfo, inst);
      reg_nr = brw_eu_inst_3src_src1_reg_nr(devinfo, inst);
      subreg_nr = brw_eu_inst_3src_a1_src1_subreg_nr(devinfo, inst);
      type = brw_eu_inst_3src_a1_src1_type(devinfo, inst);

      _vert_stride = vstride_from_align1_3src_vstride(
         devinfo, brw_eu_inst_3src_a1_src1_vstride(devinfo, inst));
      _horiz_stride = hstride_from_align1_3src_hstride(
                         brw_eu_inst_3src_a1_src1_hstride(devinfo, inst));
      _width = implied_width(_vert_stride, _horiz_stride);
   } else {
      _file = FIXED_GRF;
      reg_nr = brw_eu_inst_3src_src1_reg_nr(devinfo, inst);
      subreg_nr = brw_eu_inst_3src_a16_src1_subreg_nr(devinfo, inst);
      type = brw_eu_inst_3src_a16_src_type(devinfo, inst);

      if (brw_eu_inst_3src_a16_src1_rep_ctrl(devinfo, inst)) {
         _vert_stride = BRW_VERTICAL_STRIDE_0;
         _width = BRW_WIDTH_1;
         _horiz_stride = BRW_HORIZONTAL_STRIDE_0;
      } else {
         _vert_stride = BRW_VERTICAL_STRIDE_4;
         _width = BRW_WIDTH_4;
         _horiz_stride = BRW_HORIZONTAL_STRIDE_1;
      }
   }
   is_scalar_region = _vert_stride == BRW_VERTICAL_STRIDE_0 &&
                      _width == BRW_WIDTH_1 &&
                      _horiz_stride == BRW_HORIZONTAL_STRIDE_0;

   subreg_nr /= brw_type_size_bytes(type);

   err |= control(file, "negate", m_negate,
                  brw_eu_inst_3src_src1_negate(devinfo, inst), NULL);
   err |= control(file, "abs", _abs, brw_eu_inst_3src_src1_abs(devinfo, inst), NULL);

   err |= reg(file, _file, reg_nr);
   if (err == -1)
      return 0;
   if (subreg_nr || is_scalar_region)
      format(file, ".%d", subreg_nr);
   src_align1_region(file, _vert_stride, _width, _horiz_stride);
   if (!is_scalar_region && !is_align1)
      err |= src_swizzle(file, brw_eu_inst_3src_a16_src1_swizzle(devinfo, inst));
   string(file, brw_reg_type_to_letters(type));
   return err;
}

static int
src2_3src(FILE *file, const struct intel_device_info *devinfo,
          const brw_eu_inst *inst)
{
   int err = 0;
   unsigned reg_nr, subreg_nr;
   enum brw_reg_file _file;
   enum brw_reg_type type;
   enum brw_vertical_stride _vert_stride;
   enum brw_width _width;
   enum brw_horizontal_stride _horiz_stride;
   bool is_scalar_region;
   bool is_align1 = brw_eu_inst_3src_access_mode(devinfo, inst) == BRW_ALIGN_1;

   if (devinfo->ver < 10 && is_align1)
      return 0;

   if (is_align1) {
      _file = brw_eu_inst_3src_a1_src2_reg_file(devinfo, inst);
      if (_file == IMM) {
         uint16_t imm_val = brw_eu_inst_3src_a1_src2_imm(devinfo, inst);
         enum brw_reg_type type = brw_eu_inst_3src_a1_src2_type(devinfo, inst);

         if (type == BRW_TYPE_W) {
            format(file, "%dW", imm_val);
         } else if (type == BRW_TYPE_UW) {
            format(file, "0x%04xUW", imm_val);
         } else if (type == BRW_TYPE_HF) {
            format(file, "0x%04xHF", imm_val);
         }
         return 0;
      }

      reg_nr = brw_eu_inst_3src_src2_reg_nr(devinfo, inst);
      subreg_nr = brw_eu_inst_3src_a1_src2_subreg_nr(devinfo, inst);
      type = brw_eu_inst_3src_a1_src2_type(devinfo, inst);
      /* FINISHME: No vertical stride on src2. Is using the hstride in place
       *           correct? Doesn't seem like it, since there's hstride=1 but
       *           no vstride=1.
       */
      _vert_stride = vstride_from_align1_3src_hstride(
                        brw_eu_inst_3src_a1_src2_hstride(devinfo, inst));
      _horiz_stride = hstride_from_align1_3src_hstride(
                         brw_eu_inst_3src_a1_src2_hstride(devinfo, inst));
      _width = implied_width(_vert_stride, _horiz_stride);
   } else {
      _file = FIXED_GRF;
      reg_nr = brw_eu_inst_3src_src2_reg_nr(devinfo, inst);
      subreg_nr = brw_eu_inst_3src_a16_src2_subreg_nr(devinfo, inst);
      type = brw_eu_inst_3src_a16_src_type(devinfo, inst);

      if (brw_eu_inst_3src_a16_src2_rep_ctrl(devinfo, inst)) {
         _vert_stride = BRW_VERTICAL_STRIDE_0;
         _width = BRW_WIDTH_1;
         _horiz_stride = BRW_HORIZONTAL_STRIDE_0;
      } else {
         _vert_stride = BRW_VERTICAL_STRIDE_4;
         _width = BRW_WIDTH_4;
         _horiz_stride = BRW_HORIZONTAL_STRIDE_1;
      }
   }
   is_scalar_region = _vert_stride == BRW_VERTICAL_STRIDE_0 &&
                      _width == BRW_WIDTH_1 &&
                      _horiz_stride == BRW_HORIZONTAL_STRIDE_0;

   subreg_nr /= brw_type_size_bytes(type);

   err |= control(file, "negate", m_negate,
                  brw_eu_inst_3src_src2_negate(devinfo, inst), NULL);
   err |= control(file, "abs", _abs, brw_eu_inst_3src_src2_abs(devinfo, inst), NULL);

   err |= reg(file, _file, reg_nr);
   if (err == -1)
      return 0;
   if (subreg_nr || is_scalar_region)
      format(file, ".%d", subreg_nr);
   src_align1_region(file, _vert_stride, _width, _horiz_stride);
   if (!is_scalar_region && !is_align1)
      err |= src_swizzle(file, brw_eu_inst_3src_a16_src2_swizzle(devinfo, inst));
   string(file, brw_reg_type_to_letters(type));
   return err;
}

static int
src0_dpas_3src(FILE *file, const struct intel_device_info *devinfo,
               const brw_eu_inst *inst)
{
   uint32_t reg_file = brw_eu_inst_dpas_3src_src0_reg_file(devinfo, inst);

   if (reg(file, reg_file, brw_eu_inst_dpas_3src_src0_reg_nr(devinfo, inst)) == -1)
      return 0;

   unsigned subreg_nr = brw_eu_inst_dpas_3src_src0_subreg_nr(devinfo, inst);
   enum brw_reg_type type = brw_eu_inst_dpas_3src_src0_type(devinfo, inst);

   if (subreg_nr)
      format(file, ".%d", subreg_nr);
   src_align1_region(file,
                     BRW_VERTICAL_STRIDE_1,
                     BRW_WIDTH_1,
                     BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_0);

   string(file, brw_reg_type_to_letters(type));

   return 0;
}

static int
src1_dpas_3src(FILE *file, const struct intel_device_info *devinfo,
               const brw_eu_inst *inst)
{
   uint32_t reg_file = brw_eu_inst_dpas_3src_src1_reg_file(devinfo, inst);

   if (reg(file, reg_file, brw_eu_inst_dpas_3src_src1_reg_nr(devinfo, inst)) == -1)
      return 0;

   unsigned subreg_nr = brw_eu_inst_dpas_3src_src1_subreg_nr(devinfo, inst);
   enum brw_reg_type type = brw_eu_inst_dpas_3src_src1_type(devinfo, inst);

   if (subreg_nr)
      format(file, ".%d", subreg_nr);
   src_align1_region(file,
                     BRW_VERTICAL_STRIDE_1,
                     BRW_WIDTH_1,
                     BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_0);

   string(file, brw_reg_type_to_letters(type));

   return 0;
}

static int
src2_dpas_3src(FILE *file, const struct intel_device_info *devinfo,
               const brw_eu_inst *inst)
{
   uint32_t reg_file = brw_eu_inst_dpas_3src_src2_reg_file(devinfo, inst);

   if (reg(file, reg_file, brw_eu_inst_dpas_3src_src2_reg_nr(devinfo, inst)) == -1)
      return 0;

   unsigned subreg_nr = brw_eu_inst_dpas_3src_src2_subreg_nr(devinfo, inst);
   enum brw_reg_type type = brw_eu_inst_dpas_3src_src2_type(devinfo, inst);

   if (subreg_nr)
      format(file, ".%d", subreg_nr);
   src_align1_region(file,
                     BRW_VERTICAL_STRIDE_1,
                     BRW_WIDTH_1,
                     BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_0);

   string(file, brw_reg_type_to_letters(type));

   return 0;
}

static int
imm(FILE *file, const struct brw_isa_info *isa, enum brw_reg_type type,
    const brw_eu_inst *inst)
{
   const struct intel_device_info *devinfo = isa->devinfo;

   switch (type) {
   case BRW_TYPE_UQ:
      format(file, "0x%016"PRIx64"UQ", brw_eu_inst_imm_uq(devinfo, inst));
      break;
   case BRW_TYPE_Q:
      format(file, "0x%016"PRIx64"Q", brw_eu_inst_imm_uq(devinfo, inst));
      break;
   case BRW_TYPE_UD:
      format(file, "0x%08xUD", brw_eu_inst_imm_ud(devinfo, inst));
      break;
   case BRW_TYPE_D:
      format(file, "%dD", brw_eu_inst_imm_d(devinfo, inst));
      break;
   case BRW_TYPE_UW:
      format(file, "0x%04xUW", (uint16_t) brw_eu_inst_imm_ud(devinfo, inst));
      break;
   case BRW_TYPE_W:
      format(file, "%dW", (int16_t) brw_eu_inst_imm_d(devinfo, inst));
      break;
   case BRW_TYPE_UV:
      format(file, "0x%08xUV", brw_eu_inst_imm_ud(devinfo, inst));
      break;
   case BRW_TYPE_VF:
      format(file, "0x%"PRIx64"VF", brw_eu_inst_bits(inst, 127, 96));
      pad(file, 48);
      format(file, "/* [%-gF, %-gF, %-gF, %-gF]VF */",
             brw_vf_to_float(brw_eu_inst_imm_ud(devinfo, inst)),
             brw_vf_to_float(brw_eu_inst_imm_ud(devinfo, inst) >> 8),
             brw_vf_to_float(brw_eu_inst_imm_ud(devinfo, inst) >> 16),
             brw_vf_to_float(brw_eu_inst_imm_ud(devinfo, inst) >> 24));
      break;
   case BRW_TYPE_V:
      format(file, "0x%08xV", brw_eu_inst_imm_ud(devinfo, inst));
      break;
   case BRW_TYPE_F:
      /* The DIM instruction's src0 uses an F type but contains a
       * 64-bit immediate
       */
      format(file, "0x%"PRIx64"F", brw_eu_inst_bits(inst, 127, 96));
      pad(file, 48);
      format(file, " /* %-gF */", brw_eu_inst_imm_f(devinfo, inst));
      break;
   case BRW_TYPE_DF:
      format(file, "0x%016"PRIx64"DF", brw_eu_inst_imm_uq(devinfo, inst));
      pad(file, 48);
      format(file, "/* %-gDF */", brw_eu_inst_imm_df(devinfo, inst));
      break;
   case BRW_TYPE_HF:
      format(file, "0x%04xHF",
             (uint16_t) brw_eu_inst_imm_ud(devinfo, inst));
      pad(file, 48);
      format(file, "/* %-gHF */",
             _mesa_half_to_float((uint16_t) brw_eu_inst_imm_ud(devinfo, inst)));
      break;
   case BRW_TYPE_UB:
   case BRW_TYPE_B:
   default:
      format(file, "*** invalid immediate type %d ", type);
   }
   return 0;
}

static int
src_sends_da(FILE *file,
             const struct intel_device_info *devinfo,
             enum brw_reg_type type,
             enum brw_reg_file _reg_file,
             unsigned _reg_nr,
             unsigned _reg_subnr)
{
   int err = 0;

   err |= reg(file, _reg_file, _reg_nr);
   if (err == -1)
      return 0;
   if (_reg_subnr)
      format(file, ".1");
   string(file, brw_reg_type_to_letters(type));

   return err;
}

static int
src_sends_ia(FILE *file,
             const struct intel_device_info *devinfo,
             enum brw_reg_type type,
             int _addr_imm,
             unsigned _addr_subreg_nr)
{
   string(file, "g[a0");
   if (_addr_subreg_nr)
      format(file, ".1");
   if (_addr_imm)
      format(file, " %d", _addr_imm);
   string(file, "]");
   string(file, brw_reg_type_to_letters(type));

   return 0;
}

static int
src_send_desc_ia(FILE *file,
                 const struct intel_device_info *devinfo,
                 unsigned _addr_subreg_nr)
{
   string(file, "a0");
   if (_addr_subreg_nr)
      format(file, ".%d", _addr_subreg_nr);
   format(file, "<0>UD");

   return 0;
}

static int
src0(FILE *file, const struct brw_isa_info *isa, const brw_eu_inst *inst)
{
   const struct intel_device_info *devinfo = isa->devinfo;

   if (is_split_send(devinfo, brw_eu_inst_opcode(isa, inst))) {
      if (devinfo->ver >= 30 &&
         brw_eu_inst_send_src0_reg_file(devinfo, inst) == ARF) {
         format(file, "r[");
         reg(file, ARF, brw_eu_inst_src0_da_reg_nr(devinfo, inst));
         format(file, ".%u]", (unsigned)brw_eu_inst_send_src0_subreg_nr(devinfo, inst) * 2);
         return 0;
      } else if (devinfo->ver >= 12) {
         return src_sends_da(file,
                             devinfo,
                             BRW_TYPE_UD,
                             brw_eu_inst_send_src0_reg_file(devinfo, inst),
                             brw_eu_inst_src0_da_reg_nr(devinfo, inst),
                             0);
      } else if (brw_eu_inst_send_src0_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return src_sends_da(file,
                             devinfo,
                             BRW_TYPE_UD,
                             FIXED_GRF,
                             brw_eu_inst_src0_da_reg_nr(devinfo, inst),
                             brw_eu_inst_src0_da16_subreg_nr(devinfo, inst));
      } else {
         return src_sends_ia(file,
                             devinfo,
                             BRW_TYPE_UD,
                             brw_eu_inst_send_src0_ia16_addr_imm(devinfo, inst),
                             brw_eu_inst_src0_ia_subreg_nr(devinfo, inst));
      }
   } else if (brw_eu_inst_src0_reg_file(devinfo, inst) == IMM) {
      return imm(file, isa, brw_eu_inst_src0_type(devinfo, inst), inst);
   } else if (brw_eu_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
      if (brw_eu_inst_src0_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return src_da1(file,
                        devinfo,
                        brw_eu_inst_opcode(isa, inst),
                        brw_eu_inst_src0_type(devinfo, inst),
                        brw_eu_inst_src0_reg_file(devinfo, inst),
                        brw_eu_inst_src0_vstride(devinfo, inst),
                        brw_eu_inst_src0_width(devinfo, inst),
                        brw_eu_inst_src0_hstride(devinfo, inst),
                        brw_eu_inst_src0_da_reg_nr(devinfo, inst),
                        brw_eu_inst_src0_da1_subreg_nr(devinfo, inst),
                        brw_eu_inst_src0_abs(devinfo, inst),
                        brw_eu_inst_src0_negate(devinfo, inst));
      } else {
         return src_ia1(file,
                        devinfo,
                        brw_eu_inst_opcode(isa, inst),
                        brw_eu_inst_src0_type(devinfo, inst),
                        brw_eu_inst_src0_ia1_addr_imm(devinfo, inst),
                        brw_eu_inst_src0_ia_subreg_nr(devinfo, inst),
                        brw_eu_inst_src0_negate(devinfo, inst),
                        brw_eu_inst_src0_abs(devinfo, inst),
                        brw_eu_inst_src0_hstride(devinfo, inst),
                        brw_eu_inst_src0_width(devinfo, inst),
                        brw_eu_inst_src0_vstride(devinfo, inst));
      }
   } else {
      if (brw_eu_inst_src0_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return src_da16(file,
                         devinfo,
                         brw_eu_inst_opcode(isa, inst),
                         brw_eu_inst_src0_type(devinfo, inst),
                         brw_eu_inst_src0_reg_file(devinfo, inst),
                         brw_eu_inst_src0_vstride(devinfo, inst),
                         brw_eu_inst_src0_da_reg_nr(devinfo, inst),
                         brw_eu_inst_src0_da16_subreg_nr(devinfo, inst),
                         brw_eu_inst_src0_abs(devinfo, inst),
                         brw_eu_inst_src0_negate(devinfo, inst),
                         brw_eu_inst_src0_da16_swiz_x(devinfo, inst),
                         brw_eu_inst_src0_da16_swiz_y(devinfo, inst),
                         brw_eu_inst_src0_da16_swiz_z(devinfo, inst),
                         brw_eu_inst_src0_da16_swiz_w(devinfo, inst));
      } else {
         string(file, "Indirect align16 address mode not supported");
         return 1;
      }
   }
}

static int
src1(FILE *file, const struct brw_isa_info *isa, const brw_eu_inst *inst)
{
   const struct intel_device_info *devinfo = isa->devinfo;

   if (is_split_send(devinfo, brw_eu_inst_opcode(isa, inst))) {
      return src_sends_da(file,
                          devinfo,
                          BRW_TYPE_UD,
                          brw_eu_inst_send_src1_reg_file(devinfo, inst),
                          brw_eu_inst_send_src1_reg_nr(devinfo, inst),
                          0 /* subreg_nr */);
   } else if (brw_eu_inst_src1_reg_file(devinfo, inst) == IMM) {
      return imm(file, isa, brw_eu_inst_src1_type(devinfo, inst), inst);
   } else if (brw_eu_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
      if (brw_eu_inst_src1_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return src_da1(file,
                        devinfo,
                        brw_eu_inst_opcode(isa, inst),
                        brw_eu_inst_src1_type(devinfo, inst),
                        brw_eu_inst_src1_reg_file(devinfo, inst),
                        brw_eu_inst_src1_vstride(devinfo, inst),
                        brw_eu_inst_src1_width(devinfo, inst),
                        brw_eu_inst_src1_hstride(devinfo, inst),
                        brw_eu_inst_src1_da_reg_nr(devinfo, inst),
                        brw_eu_inst_src1_da1_subreg_nr(devinfo, inst),
                        brw_eu_inst_src1_abs(devinfo, inst),
                        brw_eu_inst_src1_negate(devinfo, inst));
      } else {
         return src_ia1(file,
                        devinfo,
                        brw_eu_inst_opcode(isa, inst),
                        brw_eu_inst_src1_type(devinfo, inst),
                        brw_eu_inst_src1_ia1_addr_imm(devinfo, inst),
                        brw_eu_inst_src1_ia_subreg_nr(devinfo, inst),
                        brw_eu_inst_src1_negate(devinfo, inst),
                        brw_eu_inst_src1_abs(devinfo, inst),
                        brw_eu_inst_src1_hstride(devinfo, inst),
                        brw_eu_inst_src1_width(devinfo, inst),
                        brw_eu_inst_src1_vstride(devinfo, inst));
      }
   } else {
      if (brw_eu_inst_src1_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return src_da16(file,
                         devinfo,
                         brw_eu_inst_opcode(isa, inst),
                         brw_eu_inst_src1_type(devinfo, inst),
                         brw_eu_inst_src1_reg_file(devinfo, inst),
                         brw_eu_inst_src1_vstride(devinfo, inst),
                         brw_eu_inst_src1_da_reg_nr(devinfo, inst),
                         brw_eu_inst_src1_da16_subreg_nr(devinfo, inst),
                         brw_eu_inst_src1_abs(devinfo, inst),
                         brw_eu_inst_src1_negate(devinfo, inst),
                         brw_eu_inst_src1_da16_swiz_x(devinfo, inst),
                         brw_eu_inst_src1_da16_swiz_y(devinfo, inst),
                         brw_eu_inst_src1_da16_swiz_z(devinfo, inst),
                         brw_eu_inst_src1_da16_swiz_w(devinfo, inst));
      } else {
         string(file, "Indirect align16 address mode not supported");
         return 1;
      }
   }
}

static int
qtr_ctrl(FILE *file, const struct intel_device_info *devinfo,
         const brw_eu_inst *inst)
{
   int qtr_ctl = brw_eu_inst_qtr_control(devinfo, inst);
   int exec_size = 1 << brw_eu_inst_exec_size(devinfo, inst);
   const unsigned nib_ctl = devinfo->ver >= 20 ? 0 :
                            brw_eu_inst_nib_control(devinfo, inst);

   if (exec_size < 8 || nib_ctl) {
      format(file, " %dN", qtr_ctl * 2 + nib_ctl + 1);
   } else if (exec_size == 8) {
      switch (qtr_ctl) {
      case 0:
         string(file, " 1Q");
         break;
      case 1:
         string(file, " 2Q");
         break;
      case 2:
         string(file, " 3Q");
         break;
      case 3:
         string(file, " 4Q");
         break;
      }
   } else if (exec_size == 16) {
      if (qtr_ctl < 2)
         string(file, " 1H");
      else
         string(file, " 2H");
   }
   return 0;
}

static bool
inst_has_type(const struct brw_isa_info *isa,
              const brw_eu_inst *inst,
              enum brw_reg_type type)
{
   const struct intel_device_info *devinfo = isa->devinfo;
   const unsigned num_sources = brw_num_sources_from_inst(isa, inst);

   if (brw_eu_inst_dst_type(devinfo, inst) == type)
      return true;

   if (num_sources >= 3) {
      if (brw_eu_inst_3src_access_mode(devinfo, inst) == BRW_ALIGN_1)
         return brw_eu_inst_3src_a1_src0_type(devinfo, inst) == type ||
                brw_eu_inst_3src_a1_src1_type(devinfo, inst) == type ||
                brw_eu_inst_3src_a1_src2_type(devinfo, inst) == type;
      else
         return brw_eu_inst_3src_a16_src_type(devinfo, inst) == type;
   } else if (num_sources == 2) {
      return brw_eu_inst_src0_type(devinfo, inst) == type ||
             brw_eu_inst_src1_type(devinfo, inst) == type;
   } else {
      return brw_eu_inst_src0_type(devinfo, inst) == type;
   }
}

static int
swsb(FILE *file, const struct brw_isa_info *isa, const brw_eu_inst *inst)
{
   const struct intel_device_info *devinfo = isa->devinfo;
   const enum opcode opcode = brw_eu_inst_opcode(isa, inst);
   const uint32_t x = brw_eu_inst_swsb(devinfo, inst);
   const bool is_unordered =
      opcode == BRW_OPCODE_SEND || opcode == BRW_OPCODE_SENDC ||
      opcode == BRW_OPCODE_MATH || opcode == BRW_OPCODE_DPAS ||
      (devinfo->has_64bit_float_via_math_pipe &&
       inst_has_type(isa, inst, BRW_TYPE_DF));
   const struct tgl_swsb swsb = tgl_swsb_decode(devinfo, is_unordered, x, opcode);
   if (swsb.regdist)
      format(file, " %s@%d",
             (swsb.pipe == TGL_PIPE_FLOAT ? "F" :
              swsb.pipe == TGL_PIPE_INT ? "I" :
              swsb.pipe == TGL_PIPE_LONG ? "L" :
              swsb.pipe == TGL_PIPE_ALL ? "A"  :
              swsb.pipe == TGL_PIPE_MATH ? "M" :
              swsb.pipe == TGL_PIPE_SCALAR ? "S" : "" ),
             swsb.regdist);
   if (swsb.mode)
      format(file, " $%d%s", swsb.sbid,
             (swsb.mode & TGL_SBID_SET ? "" :
              swsb.mode & TGL_SBID_DST ? ".dst" : ".src"));
   return 0;
}

#if MESA_DEBUG
static __attribute__((__unused__)) int
brw_disassemble_imm(const struct brw_isa_info *isa,
                    uint32_t dw3, uint32_t dw2, uint32_t dw1, uint32_t dw0)
{
   brw_eu_inst inst;
   inst.data[0] = (((uint64_t) dw1) << 32) | ((uint64_t) dw0);
   inst.data[1] = (((uint64_t) dw3) << 32) | ((uint64_t) dw2);
   return brw_disassemble_inst(stderr, isa, &inst, false, 0, NULL);
}
#endif

static void
write_label(FILE *file, const struct intel_device_info *devinfo,
            const struct brw_label *root_label,
            int offset, int jump)
{
   if (root_label != NULL) {
      int to_bytes_scale = sizeof(brw_eu_inst) / brw_jump_scale(devinfo);
      const struct brw_label *label =
         brw_find_label(root_label, offset + jump * to_bytes_scale);
      if (label != NULL) {
         format(file, " LABEL%d", label->number);
      }
   }
}

static void
lsc_disassemble_ex_desc(const struct intel_device_info *devinfo,
                        uint32_t imm_desc,
                        uint32_t imm_ex_desc,
                        FILE *file)
{
   const unsigned addr_type = lsc_msg_desc_addr_type(devinfo, imm_desc);
   switch (addr_type) {
   case LSC_ADDR_SURFTYPE_FLAT:
      format(file, " base_offset %u ",
             lsc_flat_ex_desc_base_offset(devinfo, imm_ex_desc));
      break;
   case LSC_ADDR_SURFTYPE_BSS:
   case LSC_ADDR_SURFTYPE_SS:
      format(file, " surface_state_index %u ",
             lsc_bss_ex_desc_index(devinfo, imm_ex_desc));
      break;
   case LSC_ADDR_SURFTYPE_BTI:
      format(file, " BTI %u ",
             lsc_bti_ex_desc_index(devinfo, imm_ex_desc));
      format(file, " base_offset %u ",
             lsc_bti_ex_desc_base_offset(devinfo, imm_ex_desc));
      break;
   default:
      format(file, "unsupported address surface type %d", addr_type);
      break;
   }
}

static inline bool
brw_sfid_is_lsc(unsigned sfid)
{
   switch (sfid) {
   case BRW_SFID_UGM:
   case BRW_SFID_SLM:
   case BRW_SFID_TGM:
      return true;
   default:
      break;
   }

   return false;
}

int
brw_disassemble_inst(FILE *file, const struct brw_isa_info *isa,
                     const brw_eu_inst *inst, bool is_compacted,
                     int offset, const struct brw_label *root_label)
{
   const struct intel_device_info *devinfo = isa->devinfo;

   int err = 0;
   int space = 0;

   const enum opcode opcode = brw_eu_inst_opcode(isa, inst);
   const struct opcode_desc *desc = brw_opcode_desc(isa, opcode);

   if (brw_eu_inst_pred_control(devinfo, inst)) {
      string(file, "(");
      err |= control(file, "predicate inverse", pred_inv,
                     brw_eu_inst_pred_inv(devinfo, inst), NULL);
      format(file, "f%"PRIu64".%"PRIu64,
             brw_eu_inst_flag_reg_nr(devinfo, inst),
             brw_eu_inst_flag_subreg_nr(devinfo, inst));
      if (devinfo->ver >= 20) {
         err |= control(file, "predicate control", xe2_pred_ctrl,
                        brw_eu_inst_pred_control(devinfo, inst), NULL);
      } else if (brw_eu_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
         err |= control(file, "predicate control align1", pred_ctrl_align1,
                        brw_eu_inst_pred_control(devinfo, inst), NULL);
      } else {
         err |= control(file, "predicate control align16", pred_ctrl_align16,
                        brw_eu_inst_pred_control(devinfo, inst), NULL);
      }
      string(file, ") ");
   }

   err |= print_opcode(file, isa, opcode);

   if (!is_send(opcode))
      err |= control(file, "saturate", saturate, brw_eu_inst_saturate(devinfo, inst),
                     NULL);

   err |= control(file, "debug control", debug_ctrl,
                  brw_eu_inst_debug_control(devinfo, inst), NULL);

   if (opcode == BRW_OPCODE_MATH) {
      string(file, " ");
      err |= control(file, "function", math_function,
                     brw_eu_inst_math_function(devinfo, inst), NULL);

   } else if (opcode == BRW_OPCODE_SYNC) {
      string(file, " ");
      err |= control(file, "function", sync_function,
                     brw_eu_inst_cond_modifier(devinfo, inst), NULL);

   } else if (opcode == BRW_OPCODE_DPAS) {
      string(file, ".");

      err |= control(file, "systolic depth", dpas_systolic_depth,
                     brw_eu_inst_dpas_3src_sdepth(devinfo, inst), NULL);

      const unsigned rcount = brw_eu_inst_dpas_3src_rcount(devinfo, inst) + 1;

      format(file, "x%d", rcount);
   } else if (!is_send(opcode) &&
              (devinfo->ver < 12 ||
               brw_eu_inst_src0_reg_file(devinfo, inst) != IMM ||
               brw_type_size_bytes(brw_eu_inst_src0_type(devinfo, inst)) < 8)) {
      err |= control(file, "conditional modifier", conditional_modifier,
                     brw_eu_inst_cond_modifier(devinfo, inst), NULL);

      /* If we're using the conditional modifier, print which flags reg is
       * used for it.  Note that on gfx6+, the embedded-condition SEL and
       * control flow doesn't update flags.
       */
      if (brw_eu_inst_cond_modifier(devinfo, inst) &&
          (opcode != BRW_OPCODE_SEL &&
           opcode != BRW_OPCODE_CSEL &&
           opcode != BRW_OPCODE_IF &&
           opcode != BRW_OPCODE_WHILE)) {
         format(file, ".f%"PRIu64".%"PRIu64,
                brw_eu_inst_flag_reg_nr(devinfo, inst),
                brw_eu_inst_flag_subreg_nr(devinfo, inst));
      }
   }

   if (opcode != BRW_OPCODE_NOP) {
      string(file, "(");
      err |= control(file, "execution size", exec_size,
                     brw_eu_inst_exec_size(devinfo, inst), NULL);
      string(file, ")");
   }

   if (brw_has_uip(devinfo, opcode)) {
      /* Instructions that have UIP also have JIP. */
      pad(file, 16);
      string(file, "JIP: ");
      write_label(file, devinfo, root_label, offset, brw_eu_inst_jip(devinfo, inst));

      pad(file, 38);
      string(file, "UIP: ");
      write_label(file, devinfo, root_label, offset, brw_eu_inst_uip(devinfo, inst));
   } else if (brw_has_jip(devinfo, opcode)) {
      int jip = brw_eu_inst_jip(devinfo, inst);

      pad(file, 16);
      string(file, "JIP: ");
      write_label(file, devinfo, root_label, offset, jip);
   } else if (opcode == BRW_OPCODE_JMPI) {
      pad(file, 16);
      err |= src1(file, isa, inst);
   } else if (opcode == BRW_OPCODE_DPAS) {
      pad(file, 16);
      err |= dest_dpas_3src(file, devinfo, inst);

      pad(file, 32);
      err |= src0_dpas_3src(file, devinfo, inst);

      pad(file, 48);
      err |= src1_dpas_3src(file, devinfo, inst);

      pad(file, 64);
      err |= src2_dpas_3src(file, devinfo, inst);

   } else if (desc && desc->nsrc == 3) {
      pad(file, 16);
      err |= dest_3src(file, devinfo, inst);

      pad(file, 32);
      err |= src0_3src(file, devinfo, inst);

      pad(file, 48);
      err |= src1_3src(file, devinfo, inst);

      pad(file, 64);
      err |= src2_3src(file, devinfo, inst);
   } else if (desc) {
      if (desc->ndst > 0) {
         pad(file, 16);
         err |= dest(file, isa, inst);
      }

      if (desc->nsrc > 0) {
         pad(file, 32);
         err |= src0(file, isa, inst);
      }

      if (desc->nsrc > 1 && !is_send_gather(isa, inst)) {
         pad(file, 48);
         err |= src1(file, isa, inst);
      }
   }

   if (is_send(opcode)) {
      enum brw_sfid sfid = brw_eu_inst_sfid(devinfo, inst);

      bool has_imm_desc = false, has_imm_ex_desc = false;
      uint32_t imm_desc = 0, imm_ex_desc = 0;
      if (is_split_send(devinfo, opcode)) {
         pad(file, 64);
         if (brw_eu_inst_send_sel_reg32_desc(devinfo, inst)) {
            /* show the indirect descriptor source */
            err |= src_send_desc_ia(file, devinfo, 0);
         } else {
            has_imm_desc = true;
            imm_desc = brw_eu_inst_send_desc(devinfo, inst);
            fprintf(file, "0x%08"PRIx32, imm_desc);
         }

         pad(file, 80);
         if (brw_eu_inst_send_sel_reg32_ex_desc(devinfo, inst)) {
            /* show the indirect descriptor source */
            err |= src_send_desc_ia(file, devinfo,
                                    brw_eu_inst_send_ex_desc_ia_subreg_nr(devinfo, inst));
         } else {
            has_imm_ex_desc = true;
            imm_ex_desc = brw_eu_inst_sends_ex_desc(devinfo, inst,
                                                    is_send_gather(isa, inst));
            fprintf(file, "0x%08"PRIx32, imm_ex_desc);
         }
      } else {
         if (brw_eu_inst_src1_reg_file(devinfo, inst) != IMM) {
            /* show the indirect descriptor source */
            pad(file, 48);
            err |= src1(file, isa, inst);
            pad(file, 64);
         } else {
            has_imm_desc = true;
            imm_desc = brw_eu_inst_send_desc(devinfo, inst);
            pad(file, 48);
         }

         /* Print message descriptor as immediate source */
         fprintf(file, "0x%08"PRIx64, inst->data[1] >> 32);
      }

      newline(file);
      pad(file, 16);
      space = 0;

      err |= control(file, "SFID", brw_sfid, sfid, &space);
      string(file, " MsgDesc:");

      if (!has_imm_desc) {
         format(file, " indirect");
      } else {
         bool unsupported = false;
         switch (sfid) {
         case BRW_SFID_SAMPLER:
            if (devinfo->ver >= 20) {
               err |= control(file, "sampler message", xe2_sampler_msg_type,
                              brw_sampler_desc_msg_type(devinfo, imm_desc),
                              &space);
               err |= control(file, "sampler simd mode", xe2_sampler_simd_mode,
                              brw_sampler_desc_simd_mode(devinfo, imm_desc),
                              &space);
               if (brw_sampler_desc_return_format(devinfo, imm_desc)) {
                  string(file, " HP");
               }
               format(file, " Surface = %u Sampler = %u",
                      brw_sampler_desc_binding_table_index(devinfo, imm_desc),
                      brw_sampler_desc_sampler(devinfo, imm_desc));
            } else {
               err |= control(file, "sampler message", gfx5_sampler_msg_type,
                              brw_sampler_desc_msg_type(devinfo, imm_desc),
                              &space);
               err |= control(file, "sampler simd mode",
                              devinfo->ver >= 20 ? xe2_sampler_simd_mode :
                                                   gfx5_sampler_simd_mode,
                              brw_sampler_desc_simd_mode(devinfo, imm_desc),
                              &space);
               if (brw_sampler_desc_return_format(devinfo, imm_desc)) {
                  string(file, " HP");
               }
               format(file, " Surface = %u Sampler = %u",
                      brw_sampler_desc_binding_table_index(devinfo, imm_desc),
                      brw_sampler_desc_sampler(devinfo, imm_desc));
            }
            break;
         case BRW_SFID_HDC2:
         case BRW_SFID_HDC_READ_ONLY:
            format(file, " (bti %u, msg_ctrl %u, msg_type %u)",
                   brw_dp_desc_binding_table_index(devinfo, imm_desc),
                   brw_dp_desc_msg_control(devinfo, imm_desc),
                   brw_dp_desc_msg_type(devinfo, imm_desc));
            break;

         case BRW_SFID_RENDER_CACHE: {
            unsigned msg_type = brw_fb_desc_msg_type(devinfo, imm_desc);

            err |= control(file, "DP rc message type",
                           dp_rc_msg_type(devinfo), msg_type, &space);

            bool is_rt_write = msg_type ==
               GFX6_DATAPORT_WRITE_MESSAGE_RENDER_TARGET_WRITE;

            if (is_rt_write) {
               err |= control(file, "RT message type",
                              devinfo->ver >= 20 ? m_rt_write_subtype_xe2 : m_rt_write_subtype,
                              brw_eu_inst_rt_message_type(devinfo, inst), &space);
               if (brw_eu_inst_rt_slot_group(devinfo, inst))
                  string(file, " Hi");
               if (brw_fb_write_desc_last_render_target(devinfo, imm_desc))
                  string(file, " LastRT");
               if (devinfo->ver >= 10 &&
                   brw_fb_write_desc_coarse_write(devinfo, imm_desc))
                  string(file, " CoarseWrite");
            } else {
               format(file, " MsgCtrl = 0x%u",
                      brw_fb_desc_msg_control(devinfo, imm_desc));
            }

            format(file, " Surface = %u",
                   brw_fb_desc_binding_table_index(devinfo, imm_desc));
            break;
         }

         case BRW_SFID_URB: {
            if (devinfo->ver >= 20) {
               format(file, " (");
               const enum lsc_opcode op = lsc_msg_desc_opcode(devinfo, imm_desc);
               err |= control(file, "operation", lsc_operation,
                              op, &space);
               format(file, ",");
               err |= control(file, "addr_size", lsc_addr_size,
                              lsc_msg_desc_addr_size(devinfo, imm_desc),
                              &space);

               format(file, ",");
               err |= control(file, "data_size", lsc_data_size,
                              lsc_msg_desc_data_size(devinfo, imm_desc),
                              &space);
               format(file, ",");
               if (lsc_opcode_has_cmask(op)) {
                  err |= control(file, "component_mask",
                                 lsc_cmask_str,
                                 lsc_msg_desc_cmask(devinfo, imm_desc),
                                 &space);
               } else {
                  err |= control(file, "vector_size",
                                 lsc_vect_size_str,
                                 lsc_msg_desc_vect_size(devinfo, imm_desc),
                                 &space);
                  if (lsc_msg_desc_transpose(devinfo, imm_desc))
                     format(file, ", transpose");
               }
               switch(op) {
               case LSC_OP_LOAD_CMASK:
               case LSC_OP_LOAD:
               case LSC_OP_LOAD_CMASK_MSRT:
                  format(file, ",");
                  err |= control(file, "cache_load",
                                 devinfo->ver >= 20 ?
                                 xe2_lsc_cache_load :
                                 lsc_cache_load,
                                 lsc_msg_desc_cache_ctrl(devinfo, imm_desc),
                                 &space);
                  break;
               default:
                  format(file, ",");
                  err |= control(file, "cache_store",
                                 devinfo->ver >= 20 ?
                                 xe2_lsc_cache_store :
                                 lsc_cache_store,
                                 lsc_msg_desc_cache_ctrl(devinfo, imm_desc),
                                 &space);
                  break;
               }

               format(file, " dst_len = %u,",
                      brw_message_desc_rlen(devinfo, imm_desc) / reg_unit(devinfo));
               format(file, " src0_len = %u,",
                      brw_message_desc_mlen(devinfo, imm_desc) / reg_unit(devinfo));
               if (!is_send_gather(isa, inst))
                  format(file, " src1_len = %d",
                         brw_message_ex_desc_ex_mlen(devinfo, imm_ex_desc) / reg_unit(devinfo));
               err |= control(file, "address_type", lsc_addr_surface_type,
                              lsc_msg_desc_addr_type(devinfo, imm_desc), &space);
               format(file, " )");
            } else {
               unsigned urb_opcode = brw_eu_inst_urb_opcode(devinfo, inst);

               format(file, " offset %"PRIu64, brw_eu_inst_urb_global_offset(devinfo, inst));

               space = 1;

               err |= control(file, "urb opcode",
                              gfx7_urb_opcode, urb_opcode, &space);

               if (brw_eu_inst_urb_per_slot_offset(devinfo, inst)) {
                  string(file, " per-slot");
               }

               if (urb_opcode == GFX8_URB_OPCODE_SIMD8_WRITE ||
                   urb_opcode == GFX8_URB_OPCODE_SIMD8_READ) {
                  if (brw_eu_inst_urb_channel_mask_present(devinfo, inst))
                     string(file, " masked");
               } else if (urb_opcode != GFX125_URB_OPCODE_FENCE) {
                  err |= control(file, "urb swizzle", urb_swizzle,
                                 brw_eu_inst_urb_swizzle_control(devinfo, inst),
                                 &space);
               }
            }
            break;
         }
         case BRW_SFID_THREAD_SPAWNER:
            break;

         case BRW_SFID_MESSAGE_GATEWAY:
            format(file, " (%s)",
                   gfx7_gateway_subfuncid[brw_eu_inst_gateway_subfuncid(devinfo, inst)]);
            break;

         case BRW_SFID_SLM:
         case BRW_SFID_TGM:
         case BRW_SFID_UGM: {
            assert(devinfo->has_lsc);
            format(file, " (");
            const enum lsc_opcode op = lsc_msg_desc_opcode(devinfo, imm_desc);
            err |= control(file, "operation", lsc_operation,
                           op, &space);
            format(file, ",");
            err |= control(file, "addr_size", lsc_addr_size,
                           lsc_msg_desc_addr_size(devinfo, imm_desc),
                           &space);

            if (op == LSC_OP_FENCE) {
               format(file, ",");
               err |= control(file, "scope", lsc_fence_scope,
                              lsc_fence_msg_desc_scope(devinfo, imm_desc),
                              &space);
               format(file, ",");
               err |= control(file, "flush_type", lsc_flush_type,
                              lsc_fence_msg_desc_flush_type(devinfo, imm_desc),
                              &space);
               format(file, ",");
               err |= control(file, "backup_mode_fence_routing",
                              lsc_backup_fence_routing,
                              lsc_fence_msg_desc_backup_routing(devinfo, imm_desc),
                              &space);
            } else {
               format(file, ",");
               err |= control(file, "data_size", lsc_data_size,
                              lsc_msg_desc_data_size(devinfo, imm_desc),
                              &space);
               format(file, ",");
               if (lsc_opcode_has_cmask(op)) {
                  err |= control(file, "component_mask",
                                 lsc_cmask_str,
                                 lsc_msg_desc_cmask(devinfo, imm_desc),
                                 &space);
               } else {
                  err |= control(file, "vector_size",
                                 lsc_vect_size_str,
                                 lsc_msg_desc_vect_size(devinfo, imm_desc),
                                 &space);
                  if (lsc_msg_desc_transpose(devinfo, imm_desc))
                     format(file, ", transpose");
               }
               switch(op) {
               case LSC_OP_LOAD_CMASK:
               case LSC_OP_LOAD:
                  format(file, ",");
                  err |= control(file, "cache_load",
                                 devinfo->ver >= 20 ?
                                 xe2_lsc_cache_load :
                                 lsc_cache_load,
                                 lsc_msg_desc_cache_ctrl(devinfo, imm_desc),
                                 &space);
                  break;
               default:
                  format(file, ",");
                  err |= control(file, "cache_store",
                                 devinfo->ver >= 20 ?
                                 xe2_lsc_cache_store :
                                 lsc_cache_store,
                                 lsc_msg_desc_cache_ctrl(devinfo, imm_desc),
                                 &space);
                  break;
               }
            }
            format(file, " dst_len = %u,",
                   brw_message_desc_rlen(devinfo, imm_desc) / reg_unit(devinfo));
            format(file, " src0_len = %u,",
                   brw_message_desc_mlen(devinfo, imm_desc) / reg_unit(devinfo));

            if (!brw_eu_inst_send_sel_reg32_ex_desc(devinfo, inst) &&
                !is_send_gather(isa, inst))
               format(file, " src1_len = %d",
                      brw_message_ex_desc_ex_mlen(devinfo, imm_ex_desc) / reg_unit(devinfo));

            err |= control(file, "address_type", lsc_addr_surface_type,
                           lsc_msg_desc_addr_type(devinfo, imm_desc), &space);
            format(file, " )");
            break;
         }

         case BRW_SFID_HDC0:
            format(file, " (");
            space = 0;

            err |= control(file, "DP DC0 message type",
                           dp_dc0_msg_type_gfx7,
                           brw_dp_desc_msg_type(devinfo, imm_desc), &space);

            format(file, ", bti %u, ",
                   brw_dp_desc_binding_table_index(devinfo, imm_desc));

            switch (brw_eu_inst_dp_msg_type(devinfo, inst)) {
            case GFX7_DATAPORT_DC_UNTYPED_ATOMIC_OP:
               control(file, "atomic op", aop,
                       brw_dp_desc_msg_control(devinfo, imm_desc) & 0xf,
                       &space);
               break;
            case GFX7_DATAPORT_DC_OWORD_BLOCK_READ:
            case GFX7_DATAPORT_DC_OWORD_BLOCK_WRITE: {
               unsigned msg_ctrl = brw_dp_desc_msg_control(devinfo, imm_desc);
               assert(dp_oword_block_rw[msg_ctrl & 7]);
               format(file, "owords = %s, aligned = %d",
                     dp_oword_block_rw[msg_ctrl & 7], (msg_ctrl >> 3) & 3);
               break;
            }
            default:
               format(file, "%u",
                      brw_dp_desc_msg_control(devinfo, imm_desc));
            }
            format(file, ")");
            break;

         case BRW_SFID_HDC1: {
            format(file, " (");
            space = 0;

            unsigned msg_ctrl = brw_dp_desc_msg_control(devinfo, imm_desc);

            err |= control(file, "DP DC1 message type",
                           dp_dc1_msg_type_hsw,
                           brw_dp_desc_msg_type(devinfo, imm_desc), &space);

            format(file, ", Surface = %u, ",
                   brw_dp_desc_binding_table_index(devinfo, imm_desc));

            switch (brw_eu_inst_dp_msg_type(devinfo, inst)) {
            case HSW_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_OP:
            case HSW_DATAPORT_DC_PORT1_TYPED_ATOMIC_OP:
            case HSW_DATAPORT_DC_PORT1_ATOMIC_COUNTER_OP:
               format(file, "SIMD%d,", (msg_ctrl & (1 << 4)) ? 8 : 16);
               FALLTHROUGH;
            case HSW_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_OP_SIMD4X2:
            case HSW_DATAPORT_DC_PORT1_TYPED_ATOMIC_OP_SIMD4X2:
            case HSW_DATAPORT_DC_PORT1_ATOMIC_COUNTER_OP_SIMD4X2:
            case GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_OP:
            case GFX12_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_HALF_INT_OP:
               control(file, "atomic op", aop, msg_ctrl & 0xf, &space);
               break;
            case HSW_DATAPORT_DC_PORT1_UNTYPED_SURFACE_READ:
            case HSW_DATAPORT_DC_PORT1_UNTYPED_SURFACE_WRITE:
            case HSW_DATAPORT_DC_PORT1_TYPED_SURFACE_READ:
            case HSW_DATAPORT_DC_PORT1_TYPED_SURFACE_WRITE:
            case GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_SURFACE_WRITE:
            case GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_SURFACE_READ: {
               static const char *simd_modes[] = { "4x2", "16", "8" };
               format(file, "SIMD%s, Mask = 0x%x",
                      simd_modes[msg_ctrl >> 4], msg_ctrl & 0xf);
               break;
            }
            case GFX9_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_FLOAT_OP:
            case GFX9_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_FLOAT_OP:
            case GFX12_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_HALF_FLOAT_OP:
               format(file, "SIMD%d,", (msg_ctrl & (1 << 4)) ? 8 : 16);
               control(file, "atomic float op", aop_float, msg_ctrl & 0xf,
                       &space);
               break;
            case GFX9_DATAPORT_DC_PORT1_A64_OWORD_BLOCK_WRITE:
            case GFX9_DATAPORT_DC_PORT1_A64_OWORD_BLOCK_READ:
               assert(dp_oword_block_rw[msg_ctrl & 7]);
               format(file, "owords = %s, aligned = %d",
                     dp_oword_block_rw[msg_ctrl & 7], (msg_ctrl >> 3) & 3);
               break;
            default:
               format(file, "0x%x", msg_ctrl);
            }
            format(file, ")");
            break;
         }

         case BRW_SFID_PIXEL_INTERPOLATOR:
            format(file, " (%s, %s, 0x%02"PRIx64")",
                   brw_eu_inst_pi_nopersp(devinfo, inst) ? "linear" : "persp",
                   pixel_interpolator_msg_types[brw_eu_inst_pi_message_type(devinfo, inst)],
                   brw_eu_inst_pi_message_data(devinfo, inst));
            break;

         case BRW_SFID_RAY_TRACE_ACCELERATOR:
            if (devinfo->has_ray_tracing) {
               format(file, " SIMD%d,",
                      brw_rt_trace_ray_desc_exec_size(devinfo, imm_desc));
            } else {
               unsupported = true;
            }
            break;

         default:
            unsupported = true;
            break;
         }

         if (unsupported)
            format(file, "unsupported shared function ID %d", sfid);

         if (space)
            string(file, " ");
      }
      if (devinfo->verx10 >= 125 &&
          brw_eu_inst_send_sel_reg32_ex_desc(devinfo, inst) &&
          brw_eu_inst_send_ex_bso(devinfo, inst)) {
         format(file, " src1_len = %u",
                (unsigned) brw_eu_inst_send_src1_len(devinfo, inst));

         format(file, " ex_bso");
      }
      if (brw_sfid_is_lsc(sfid) ||
          (sfid == BRW_SFID_URB && devinfo->ver >= 20)) {
            lsc_disassemble_ex_desc(devinfo, imm_desc, imm_ex_desc, file);
      } else {
         if (has_imm_desc)
            format(file, " mlen %u", brw_message_desc_mlen(devinfo, imm_desc) / reg_unit(devinfo));
         if (has_imm_ex_desc) {
            format(file, " ex_mlen %u",
                   brw_message_ex_desc_ex_mlen(devinfo, imm_ex_desc) / reg_unit(devinfo));
         }
         if (has_imm_desc)
            format(file, " rlen %u", brw_message_desc_rlen(devinfo, imm_desc) / reg_unit(devinfo));
      }
   }
   pad(file, 64);
   if (opcode != BRW_OPCODE_NOP) {
      string(file, "{");
      space = 1;
      err |= control(file, "access mode", access_mode,
                     brw_eu_inst_access_mode(devinfo, inst), &space);
      err |= control(file, "write enable control", wectrl,
                     brw_eu_inst_mask_control(devinfo, inst), &space);

      if (devinfo->ver < 12) {
         err |= control(file, "dependency control", dep_ctrl,
                        ((brw_eu_inst_no_dd_check(devinfo, inst) << 1) |
                         brw_eu_inst_no_dd_clear(devinfo, inst)), &space);
      }

      err |= qtr_ctrl(file, devinfo, inst);

      if (devinfo->ver >= 12)
         err |= swsb(file, isa, inst);

      err |= control(file, "compaction", cmpt_ctrl, is_compacted, &space);
      err |= control(file, "thread control", thread_ctrl,
                     (devinfo->ver >= 12 ? brw_eu_inst_atomic_control(devinfo, inst) :
                                           brw_eu_inst_thread_control(devinfo, inst)),
                     &space);
      if (brw_has_branch_ctrl(devinfo, opcode)) {
         err |= control(file, "branch ctrl", branch_ctrl,
                        brw_eu_inst_branch_control(devinfo, inst), &space);
      } else if (devinfo->ver < 20) {
         err |= control(file, "acc write control", accwr,
                        brw_eu_inst_acc_wr_control(devinfo, inst), &space);
      }
      if (is_send(opcode))
         err |= control(file, "end of thread", end_of_thread,
                        brw_eu_inst_eot(devinfo, inst), &space);
      if (space)
         string(file, " ");
      string(file, "}");
   }
   string(file, ";");
   newline(file);
   return err;
}

int
brw_disassemble_find_end(const struct brw_isa_info *isa,
                         const void *assembly, int start)
{
   const struct intel_device_info *devinfo = isa->devinfo;
   int offset = start;

   /* This loop exits when send-with-EOT or when opcode is 0 */
   while (true) {
      const brw_eu_inst *insn = assembly + offset;

      if (brw_eu_inst_cmpt_control(devinfo, insn)) {
         offset += 8;
      } else {
         offset += 16;
      }

      /* Simplistic, but efficient way to terminate disasm */
      uint32_t opcode = brw_eu_inst_opcode(isa, insn);
      if (opcode == 0 || (is_send(opcode) && brw_eu_inst_eot(devinfo, insn))) {
         break;
      }
   }

   return offset;
}

void
brw_disassemble_with_errors(const struct brw_isa_info *isa,
                            const void *assembly, int start,
                            int64_t *lineno_offset, FILE *out)
{
   int end = brw_disassemble_find_end(isa, assembly, start);

   /* Make a dummy disasm structure that brw_validate_instructions
    * can work from.
    */
   struct disasm_info *disasm_info = disasm_initialize(isa, NULL);
   disasm_new_inst_group(disasm_info, start);
   disasm_new_inst_group(disasm_info, end);

   brw_validate_instructions(isa, assembly, start, end, disasm_info);

   void *mem_ctx = ralloc_context(NULL);
   const struct brw_label *root_label =
      brw_label_assembly(isa, assembly, start, end, mem_ctx);

   brw_foreach_list_typed(struct inst_group, group, link,
                      &disasm_info->group_list) {
      struct brw_exec_node *next_node = brw_exec_node_get_next(&group->link);
      if (brw_exec_node_is_tail_sentinel(next_node))
         break;

      struct inst_group *next =
         brw_exec_node_data(struct inst_group, next_node, link);

      int start_offset = group->offset;
      int end_offset = next->offset;

      brw_disassemble(isa, assembly, start_offset, end_offset,
                      root_label, lineno_offset, out);

      if (group->error) {
         fputs(group->error, out);
      }
   }

   ralloc_free(mem_ctx);
   ralloc_free(disasm_info);
}

void
brw_disassemble_with_lineno(const struct brw_isa_info *isa, uint32_t stage,
                            int dispatch_width, uint32_t src_hash,
                            const void *assembly, int start,
                            int64_t lineno_offset, FILE *out)
{
   fprintf(out, "\nDumping shader asm for %s", _mesa_shader_stage_to_abbrev(stage));
   if (dispatch_width > 0)
      fprintf(out, " SIMD%i", dispatch_width);
   fprintf(out, " (src_hash 0x%x):\n\n", src_hash);
   brw_disassemble_with_errors(isa, assembly, start, &lineno_offset, out);
}
