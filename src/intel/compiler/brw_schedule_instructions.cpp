/*
 * Copyright © 2010 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Eric Anholt <eric@anholt.net>
 *
 */

#include "brw_eu.h"
#include "brw_shader.h"
#include "brw_analysis.h"
#include "brw_cfg.h"
#include <new>

/** @file
 *
 * List scheduling of FS instructions.
 *
 * The basic model of the list scheduler is to take a basic block,
 * compute a DAG of the dependencies (RAW ordering with latency, WAW
 * ordering with latency, WAR ordering), and make a list of the DAG heads.
 * Heuristically pick a DAG head, then put all the children that are
 * now DAG heads into the list of things to schedule.
 *
 * The heuristic is the important part.  We're trying to be cheap,
 * since actually computing the optimal scheduling is NP complete.
 * What we do is track a "current clock".  When we schedule a node, we
 * update the earliest-unblocked clock time of its children, and
 * increment the clock.  Then, when trying to schedule, we just pick
 * the earliest-unblocked instruction to schedule.
 *
 * Note that often there will be many things which could execute
 * immediately, and there are a range of heuristic options to choose
 * from in picking among those.
 */

static bool debug = false;

struct schedule_node_child;

class schedule_node : public brw_exec_node
{
public:
   void set_latency(const struct brw_isa_info *isa);

   brw_inst *inst;
   schedule_node_child *children;
   int children_count;
   int children_cap;
   int initial_parent_count;
   int initial_unblocked_time;
   int latency;

   /**
    * This is the sum of the instruction's latency plus the maximum delay of
    * its children, or just the issue_time if it's a leaf node.
    */
   int delay;

   /**
    * Preferred exit node among the (direct or indirect) successors of this
    * node.  Among the scheduler nodes blocked by this node, this will be the
    * one that may cause earliest program termination, or NULL if none of the
    * successors is an exit node.
    */
   schedule_node *exit;

   /**
    * How many cycles this instruction takes to issue.
    *
    * Instructions in gen hardware are handled one simd4 vector at a time,
    * with 1 cycle per vector dispatched.  Thus SIMD8 pixel shaders take 2
    * cycles to dispatch and SIMD16 (compressed) instructions take 4.
    */
   int issue_time;

   /**
    * Whether the instruction reads any part of the address register (to speed
    * up instruction checks).
    */
   schedule_node **address_read;
   int address_read_count;
   int address_read_cap;

   /* Temporary data used during the scheduling process. */
   struct {
      int parent_count;
      int unblocked_time;

      /**
       * Which iteration of pushing groups of children onto the candidates list
       * this node was a part of.
       */
      unsigned cand_generation;
   } tmp;
};

struct schedule_node_child {
   schedule_node *n;
   int effective_latency;
};

static inline void
reset_node_tmp(schedule_node *n)
{
   n->tmp.parent_count = n->initial_parent_count;
   n->tmp.unblocked_time = n->initial_unblocked_time;
   n->tmp.cand_generation = 0;
}

/**
 * Lower bound of the scheduling time after which one of the instructions
 * blocked by this node may lead to program termination.
 *
 * exit_unblocked_time() determines a strict partial ordering relation '«' on
 * the set of scheduler nodes as follows:
 *
 *   n « m <-> exit_unblocked_time(n) < exit_unblocked_time(m)
 *
 * which can be used to heuristically order nodes according to how early they
 * can unblock an exit node and lead to program termination.
 */
static inline int
exit_tmp_unblocked_time(const schedule_node *n)
{
   return n->exit ? n->exit->tmp.unblocked_time : INT_MAX;
}

static inline int
exit_initial_unblocked_time(const schedule_node *n)
{
   return n->exit ? n->exit->initial_unblocked_time : INT_MAX;
}

void
schedule_node::set_latency(const struct brw_isa_info *isa)
{
   switch (inst->opcode) {
   case BRW_OPCODE_MAD:
      /* 2 cycles
       *  (since the last two src operands are in different register banks):
       * mad(8) g4<1>F g2.2<4,4,1>F.x  g2<4,4,1>F.x g3.1<4,4,1>F.x { align16 WE_normal 1Q };
       *
       * 3 cycles on IVB, 4 on HSW
       *  (since the last two src operands are in the same register bank):
       * mad(8) g4<1>F g2.2<4,4,1>F.x  g2<4,4,1>F.x g2.1<4,4,1>F.x { align16 WE_normal 1Q };
       *
       * 18 cycles on IVB, 16 on HSW
       *  (since the last two src operands are in different register banks):
       * mad(8) g4<1>F g2.2<4,4,1>F.x  g2<4,4,1>F.x g3.1<4,4,1>F.x { align16 WE_normal 1Q };
       * mov(8) null   g4<4,5,1>F                     { align16 WE_normal 1Q };
       *
       * 20 cycles on IVB, 18 on HSW
       *  (since the last two src operands are in the same register bank):
       * mad(8) g4<1>F g2.2<4,4,1>F.x  g2<4,4,1>F.x g2.1<4,4,1>F.x { align16 WE_normal 1Q };
       * mov(8) null   g4<4,4,1>F                     { align16 WE_normal 1Q };
       */

      /* Our register allocator doesn't know about register banks, so use the
       * higher latency.
       */
      latency = 18;
      break;

   case BRW_OPCODE_LRP:
      /* 2 cycles
       *  (since the last two src operands are in different register banks):
       * lrp(8) g4<1>F g2.2<4,4,1>F.x  g2<4,4,1>F.x g3.1<4,4,1>F.x { align16 WE_normal 1Q };
       *
       * 3 cycles on IVB, 4 on HSW
       *  (since the last two src operands are in the same register bank):
       * lrp(8) g4<1>F g2.2<4,4,1>F.x  g2<4,4,1>F.x g2.1<4,4,1>F.x { align16 WE_normal 1Q };
       *
       * 16 cycles on IVB, 14 on HSW
       *  (since the last two src operands are in different register banks):
       * lrp(8) g4<1>F g2.2<4,4,1>F.x  g2<4,4,1>F.x g3.1<4,4,1>F.x { align16 WE_normal 1Q };
       * mov(8) null   g4<4,4,1>F                     { align16 WE_normal 1Q };
       *
       * 16 cycles
       *  (since the last two src operands are in the same register bank):
       * lrp(8) g4<1>F g2.2<4,4,1>F.x  g2<4,4,1>F.x g2.1<4,4,1>F.x { align16 WE_normal 1Q };
       * mov(8) null   g4<4,4,1>F                     { align16 WE_normal 1Q };
       */

      /* Our register allocator doesn't know about register banks, so use the
       * higher latency.
       */
      latency = 14;
      break;

   case SHADER_OPCODE_RCP:
   case SHADER_OPCODE_RSQ:
   case SHADER_OPCODE_SQRT:
   case SHADER_OPCODE_LOG2:
   case SHADER_OPCODE_EXP2:
   case SHADER_OPCODE_SIN:
   case SHADER_OPCODE_COS:
      /* 2 cycles:
       * math inv(8) g4<1>F g2<0,1,0>F      null       { align1 WE_normal 1Q };
       *
       * 18 cycles:
       * math inv(8) g4<1>F g2<0,1,0>F      null       { align1 WE_normal 1Q };
       * mov(8)      null   g4<8,8,1>F                 { align1 WE_normal 1Q };
       *
       * Same for exp2, log2, rsq, sqrt, sin, cos.
       */
      latency = 16;
      break;

   case SHADER_OPCODE_POW:
      /* 2 cycles:
       * math pow(8) g4<1>F g2<0,1,0>F   g2.1<0,1,0>F  { align1 WE_normal 1Q };
       *
       * 26 cycles:
       * math pow(8) g4<1>F g2<0,1,0>F   g2.1<0,1,0>F  { align1 WE_normal 1Q };
       * mov(8)      null   g4<8,8,1>F                 { align1 WE_normal 1Q };
       */
      latency = 24;
      break;

   case FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD:
      /* testing using varying-index pull constants:
       *
       * 16 cycles:
       * mov(8)  g4<1>D  g2.1<0,1,0>F                  { align1 WE_normal 1Q };
       * send(8) g4<1>F  g4<8,8,1>D
       *   data (9, 2, 3) mlen 1 rlen 1                { align1 WE_normal 1Q };
       *
       * ~480 cycles:
       * mov(8)  g4<1>D  g2.1<0,1,0>F                  { align1 WE_normal 1Q };
       * send(8) g4<1>F  g4<8,8,1>D
       *   data (9, 2, 3) mlen 1 rlen 1                { align1 WE_normal 1Q };
       * mov(8)  null    g4<8,8,1>F                    { align1 WE_normal 1Q };
       *
       * ~620 cycles:
       * mov(8)  g4<1>D  g2.1<0,1,0>F                  { align1 WE_normal 1Q };
       * send(8) g4<1>F  g4<8,8,1>D
       *   data (9, 2, 3) mlen 1 rlen 1                { align1 WE_normal 1Q };
       * mov(8)  null    g4<8,8,1>F                    { align1 WE_normal 1Q };
       * send(8) g4<1>F  g4<8,8,1>D
       *   data (9, 2, 3) mlen 1 rlen 1                { align1 WE_normal 1Q };
       * mov(8)  null    g4<8,8,1>F                    { align1 WE_normal 1Q };
       *
       * So, if it's cache-hot, it's about 140.  If it's cache cold, it's
       * about 460.  We expect to mostly be cache hot, so pick something more
       * in that direction.
       */
      latency = 200;
      break;

   case SHADER_OPCODE_SEND:
      switch (inst->sfid) {
      case BRW_SFID_SAMPLER: {
         unsigned msg_type = (inst->desc >> 12) & 0x1f;
         switch (msg_type) {
         case GFX5_SAMPLER_MESSAGE_SAMPLE_RESINFO:
         case GFX6_SAMPLER_MESSAGE_SAMPLE_SAMPLEINFO:
            /* Testing textureSize(sampler2D, 0), one load was 420 +/- 41
             * cycles (n=15):
             * mov(8)   g114<1>UD  0D                  { align1 WE_normal 1Q };
             * send(8)  g6<1>UW    g114<8,8,1>F
             *   sampler (10, 0, 10, 1) mlen 1 rlen 4  { align1 WE_normal 1Q };
             * mov(16)  g6<1>F     g6<8,8,1>D                { align1 WE_normal 1Q };
             *
             *
             * Two loads was 535 +/- 30 cycles (n=19):
             * mov(16)   g114<1>UD  0D                 { align1 WE_normal 1H };
             * send(16)  g6<1>UW    g114<8,8,1>F
             *   sampler (10, 0, 10, 2) mlen 2 rlen 8  { align1 WE_normal 1H };
             * mov(16)   g114<1>UD  0D                 { align1 WE_normal 1H };
             * mov(16)   g6<1>F     g6<8,8,1>D         { align1 WE_normal 1H };
             * send(16)  g8<1>UW    g114<8,8,1>F
             *   sampler (10, 0, 10, 2) mlen 2 rlen 8  { align1 WE_normal 1H };
             * mov(16)   g8<1>F     g8<8,8,1>D         { align1 WE_normal 1H };
             * add(16)   g6<1>F     g6<8,8,1>F   g8<8,8,1>F  { align1 WE_normal 1H };
             *
             * Since the only caches that should matter are just the
             * instruction/state cache containing the surface state,
             * assume that we always have hot caches.
             */
            latency = 100;
            break;

         default:
            /* 18 cycles:
             * mov(8)  g115<1>F   0F                  { align1 WE_normal 1Q };
             * mov(8)  g114<1>F   0F                  { align1 WE_normal 1Q };
             * send(8) g4<1>UW    g114<8,8,1>F
             *   sampler (10, 0, 0, 1) mlen 2 rlen 4  { align1 WE_normal 1Q };
             *
             * 697 +/-49 cycles (min 610, n=26):
             * mov(8)  g115<1>F   0F                  { align1 WE_normal 1Q };
             * mov(8)  g114<1>F   0F                  { align1 WE_normal 1Q };
             * send(8) g4<1>UW    g114<8,8,1>F
             *   sampler (10, 0, 0, 1) mlen 2 rlen 4  { align1 WE_normal 1Q };
             * mov(8)  null       g4<8,8,1>F          { align1 WE_normal 1Q };
             *
             * So the latency on our first texture load of the batchbuffer
             * takes ~700 cycles, since the caches are cold at that point.
             *
             * 840 +/- 92 cycles (min 720, n=25):
             * mov(8)  g115<1>F   0F                  { align1 WE_normal 1Q };
             * mov(8)  g114<1>F   0F                  { align1 WE_normal 1Q };
             * send(8) g4<1>UW    g114<8,8,1>F
             *   sampler (10, 0, 0, 1) mlen 2 rlen 4  { align1 WE_normal 1Q };
             * mov(8)  null       g4<8,8,1>F          { align1 WE_normal 1Q };
             * send(8) g4<1>UW    g114<8,8,1>F
             *   sampler (10, 0, 0, 1) mlen 2 rlen 4  { align1 WE_normal 1Q };
             * mov(8)  null       g4<8,8,1>F          { align1 WE_normal 1Q };
             *
             * On the second load, it takes just an extra ~140 cycles, and
             * after accounting for the 14 cycles of the MOV's latency, that
             * makes ~130.
             *
             * 683 +/- 49 cycles (min = 602, n=47):
             * mov(8)  g115<1>F   0F                  { align1 WE_normal 1Q };
             * mov(8)  g114<1>F   0F                  { align1 WE_normal 1Q };
             * send(8) g4<1>UW    g114<8,8,1>F
             *   sampler (10, 0, 0, 1) mlen 2 rlen 4  { align1 WE_normal 1Q };
             * send(8) g50<1>UW   g114<8,8,1>F
             *   sampler (10, 0, 0, 1) mlen 2 rlen 4  { align1 WE_normal 1Q };
             * mov(8)  null       g4<8,8,1>F          { align1 WE_normal 1Q };
             *
             * The unit appears to be pipelined, since this matches up with
             * the cache-cold case, despite there being two loads here.  If
             * you replace the g4 in the MOV to null with g50, it's still
             * 693 +/- 52 (n=39).
             *
             * So, take some number between the cache-hot 140 cycles and the
             * cache-cold 700 cycles.  No particular tuning was done on this.
             *
             * I haven't done significant testing of the non-TEX opcodes.
             * TXL at least looked about the same as TEX.
             */
            latency = 200;
            break;
         }
         break;
      }

      case BRW_SFID_HDC_READ_ONLY:
         /* See FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD */
         latency = 200;
         break;

      case BRW_SFID_RENDER_CACHE:
         switch (brw_fb_desc_msg_type(isa->devinfo, inst->desc)) {
         case GFX7_DATAPORT_RC_TYPED_SURFACE_WRITE:
         case GFX7_DATAPORT_RC_TYPED_SURFACE_READ:
            /* See also SHADER_OPCODE_TYPED_SURFACE_READ */
            latency = 600;
            break;

         case GFX7_DATAPORT_RC_TYPED_ATOMIC_OP:
         case GFX7_DATAPORT_RC_MEMORY_FENCE:
            /* See also SHADER_OPCODE_TYPED_ATOMIC */
            latency = 14000;
            break;

         case GFX6_DATAPORT_WRITE_MESSAGE_RENDER_TARGET_WRITE:
            /* completely fabricated number */
            latency = 600;
            break;

         default:
            UNREACHABLE("Unknown render cache message");
         }
         break;

      case BRW_SFID_HDC0:
         switch ((inst->desc >> 14) & 0x1f) {
         case BRW_DATAPORT_READ_MESSAGE_OWORD_BLOCK_READ:
         case GFX7_DATAPORT_DC_UNALIGNED_OWORD_BLOCK_READ:
         case GFX6_DATAPORT_WRITE_MESSAGE_OWORD_BLOCK_WRITE:
            /* We have no data for this but assume it's a little faster than
             * untyped surface read/write.
             */
            latency = 200;
            break;

         case GFX7_DATAPORT_DC_DWORD_SCATTERED_READ:
         case GFX6_DATAPORT_WRITE_MESSAGE_DWORD_SCATTERED_WRITE:
         case HSW_DATAPORT_DC_PORT0_BYTE_SCATTERED_READ:
         case HSW_DATAPORT_DC_PORT0_BYTE_SCATTERED_WRITE:
            /* We have no data for this but assume it's roughly the same as
             * untyped surface read/write.
             */
            latency = 300;
            break;

         case GFX7_DATAPORT_DC_UNTYPED_SURFACE_READ:
         case GFX7_DATAPORT_DC_UNTYPED_SURFACE_WRITE:
            /* Test code:
             *   mov(8)    g112<1>UD       0x00000000UD       { align1 WE_all 1Q };
             *   mov(1)    g112.7<1>UD     g1.7<0,1,0>UD      { align1 WE_all };
             *   mov(8)    g113<1>UD       0x00000000UD       { align1 WE_normal 1Q };
             *   send(8)   g4<1>UD         g112<8,8,1>UD
             *             data (38, 6, 5) mlen 2 rlen 1      { align1 WE_normal 1Q };
             *   .
             *   . [repeats 8 times]
             *   .
             *   mov(8)    g112<1>UD       0x00000000UD       { align1 WE_all 1Q };
             *   mov(1)    g112.7<1>UD     g1.7<0,1,0>UD      { align1 WE_all };
             *   mov(8)    g113<1>UD       0x00000000UD       { align1 WE_normal 1Q };
             *   send(8)   g4<1>UD         g112<8,8,1>UD
             *             data (38, 6, 5) mlen 2 rlen 1      { align1 WE_normal 1Q };
             *
             * Running it 100 times as fragment shader on a 128x128 quad
             * gives an average latency of 583 cycles per surface read,
             * standard deviation 0.9%.
             */
            latency = 600;
            break;

         case GFX7_DATAPORT_DC_UNTYPED_ATOMIC_OP:
            /* Test code:
             *   mov(8)    g112<1>ud       0x00000000ud       { align1 WE_all 1Q };
             *   mov(1)    g112.7<1>ud     g1.7<0,1,0>ud      { align1 WE_all };
             *   mov(8)    g113<1>ud       0x00000000ud       { align1 WE_normal 1Q };
             *   send(8)   g4<1>ud         g112<8,8,1>ud
             *             data (38, 5, 6) mlen 2 rlen 1      { align1 WE_normal 1Q };
             *
             * Running it 100 times as fragment shader on a 128x128 quad
             * gives an average latency of 13867 cycles per atomic op,
             * standard deviation 3%.  Note that this is a rather
             * pessimistic estimate, the actual latency in cases with few
             * collisions between threads and favorable pipelining has been
             * seen to be reduced by a factor of 100.
             */
            latency = 14000;
            break;

         case GFX7_DATAPORT_DC_MEMORY_FENCE:
            latency = 14000;
            break;

         default:
            UNREACHABLE("Unknown data cache message");
         }
         break;

      case BRW_SFID_HDC1:
         switch (brw_dp_desc_msg_type(isa->devinfo, inst->desc)) {
         case HSW_DATAPORT_DC_PORT1_UNTYPED_SURFACE_READ:
         case HSW_DATAPORT_DC_PORT1_UNTYPED_SURFACE_WRITE:
         case HSW_DATAPORT_DC_PORT1_TYPED_SURFACE_READ:
         case HSW_DATAPORT_DC_PORT1_TYPED_SURFACE_WRITE:
         case GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_SURFACE_WRITE:
         case GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_SURFACE_READ:
         case GFX8_DATAPORT_DC_PORT1_A64_SCATTERED_WRITE:
         case GFX9_DATAPORT_DC_PORT1_A64_SCATTERED_READ:
         case GFX9_DATAPORT_DC_PORT1_A64_OWORD_BLOCK_READ:
         case GFX9_DATAPORT_DC_PORT1_A64_OWORD_BLOCK_WRITE:
            /* See also GFX7_DATAPORT_DC_UNTYPED_SURFACE_READ */
            latency = 300;
            break;

         case HSW_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_OP:
         case HSW_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_OP_SIMD4X2:
         case HSW_DATAPORT_DC_PORT1_TYPED_ATOMIC_OP_SIMD4X2:
         case HSW_DATAPORT_DC_PORT1_TYPED_ATOMIC_OP:
         case GFX9_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_FLOAT_OP:
         case GFX8_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_OP:
         case GFX9_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_FLOAT_OP:
         case GFX12_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_HALF_INT_OP:
         case GFX12_DATAPORT_DC_PORT1_A64_UNTYPED_ATOMIC_HALF_FLOAT_OP:
            /* See also GFX7_DATAPORT_DC_UNTYPED_ATOMIC_OP */
            latency = 14000;
            break;

         default:
            UNREACHABLE("Unknown data cache message");
         }
         break;

      case BRW_SFID_PIXEL_INTERPOLATOR:
         latency = 50; /* TODO */
         break;

      case BRW_SFID_UGM:
      case BRW_SFID_TGM:
      case BRW_SFID_SLM:
         switch (lsc_msg_desc_opcode(isa->devinfo, inst->desc)) {
         case LSC_OP_LOAD:
         case LSC_OP_STORE:
         case LSC_OP_LOAD_CMASK:
         case LSC_OP_STORE_CMASK:
         case LSC_OP_LOAD_CMASK_MSRT:
         case LSC_OP_STORE_CMASK_MSRT:
            latency = 300;
            break;
         case LSC_OP_FENCE:
         case LSC_OP_ATOMIC_INC:
         case LSC_OP_ATOMIC_DEC:
         case LSC_OP_ATOMIC_LOAD:
         case LSC_OP_ATOMIC_STORE:
         case LSC_OP_ATOMIC_ADD:
         case LSC_OP_ATOMIC_SUB:
         case LSC_OP_ATOMIC_MIN:
         case LSC_OP_ATOMIC_MAX:
         case LSC_OP_ATOMIC_UMIN:
         case LSC_OP_ATOMIC_UMAX:
         case LSC_OP_ATOMIC_CMPXCHG:
         case LSC_OP_ATOMIC_FADD:
         case LSC_OP_ATOMIC_FSUB:
         case LSC_OP_ATOMIC_FMIN:
         case LSC_OP_ATOMIC_FMAX:
         case LSC_OP_ATOMIC_FCMPXCHG:
         case LSC_OP_ATOMIC_AND:
         case LSC_OP_ATOMIC_OR:
         case LSC_OP_ATOMIC_XOR:
            latency = 1400;
            break;
         default:
            UNREACHABLE("unsupported new data port message instruction");
         }
         break;

      case BRW_SFID_MESSAGE_GATEWAY:
      case BRW_SFID_BINDLESS_THREAD_DISPATCH: /* or THREAD_SPAWNER */
      case BRW_SFID_RAY_TRACE_ACCELERATOR:
         /* TODO.
          *
          * We'll assume for the moment that this is pretty quick as it
          * doesn't actually return any data.
          */
         latency = 200;
         break;

      case BRW_SFID_URB:
         latency = 200;
         break;

      default:
         UNREACHABLE("Unknown SFID");
      }
      break;

   case BRW_OPCODE_DPAS:
      switch (inst->rcount) {
      case 1:
         latency = 21;
         break;
      case 2:
         latency = 22;
         break;
      case 8:
      default:
         latency = 32;
         break;
      }
      break;

   default:
      /* 2 cycles:
       * mul(8) g4<1>F g2<0,1,0>F      0.5F            { align1 WE_normal 1Q };
       *
       * 16 cycles:
       * mul(8) g4<1>F g2<0,1,0>F      0.5F            { align1 WE_normal 1Q };
       * mov(8) null   g4<8,8,1>F                      { align1 WE_normal 1Q };
       */
      latency = 14;
      break;
   }
}

class brw_instruction_scheduler {
public:
   brw_instruction_scheduler(void *mem_ctx, const brw_shader *s, int grf_count, int hw_reg_count,
                         int block_count, bool post_reg_alloc);

   void add_barrier_deps(schedule_node *n);
   void add_cross_lane_deps(schedule_node *n);
   void add_dep(schedule_node *before, schedule_node *after, int latency);
   void add_dep(schedule_node *before, schedule_node *after);
   void add_address_dep(schedule_node *before, schedule_node *after);

   void set_current_block(bblock_t *block, const brw_ip_ranges &ips);
   void compute_delays();
   void compute_exits();

   void schedule(schedule_node *chosen);
   void update_children(schedule_node *chosen);

   void calculate_deps();
   bool is_compressed(const brw_inst *inst);
   bool register_needs_barrier(const brw_reg &reg);
   bool address_register_interfere(const schedule_node *n);
   schedule_node *choose_instruction_to_schedule();
   int calculate_issue_time(const brw_inst *inst);

   void count_reads_remaining(const brw_inst *inst);
   void setup_liveness(cfg_t *cfg);
   void update_register_pressure(const brw_inst *inst);
   int get_register_pressure_benefit(const brw_inst *inst);
   void clear_last_grf_write();

   void schedule_instructions();
   void run(brw_instruction_scheduler_mode mode);

   int grf_index(const brw_reg &reg);

   void *mem_ctx;
   linear_ctx *lin_ctx;

   schedule_node *nodes;
   int nodes_len;

   /* Current block being processed. */
   struct {
      bblock_t *block;

      /* Range of nodes in the block.  End will point to first node
       * address after the block, i.e. the range is [start, end).
       */
      schedule_node *start;
      schedule_node *end;
      int len;

      int scheduled;

      unsigned cand_generation;
      int time;
      brw_exec_list available;

      /* Currently used address register */
      uint32_t address_register[16];
   } current;

   bool post_reg_alloc;
   int grf_count;
   unsigned max_vgrf_size;
   const brw_shader *s;

   /**
    * Last instruction to have written the grf (or a channel in the grf, for the
    * scalar backend)
    */
   schedule_node **last_grf_write;

   unsigned hw_reg_count;
   int reg_pressure;
   brw_instruction_scheduler_mode mode;

   /*
    * The register pressure at the beginning of each basic block.
    */

   int *reg_pressure_in;

   /*
    * The virtual GRF's whose range overlaps the beginning of each basic block.
    */

   BITSET_WORD **livein;

   /*
    * The virtual GRF's whose range overlaps the end of each basic block.
    */

   BITSET_WORD **liveout;

   /*
    * The hardware GRF's whose range overlaps the end of each basic block.
    */

   BITSET_WORD **hw_liveout;

   /*
    * Whether we've scheduled a write for this virtual GRF yet.
    */

   bool *written;

   /*
    * How many reads we haven't scheduled for this virtual GRF yet.
    */

   int *reads_remaining;

   /*
    * How many reads we haven't scheduled for this hardware GRF yet.
    */

   int *hw_reads_remaining;
};

brw_instruction_scheduler::brw_instruction_scheduler(void *mem_ctx, const brw_shader *s,
                                             int grf_count, int hw_reg_count,
                                             int block_count, bool post_reg_alloc)
   : s(s)
{
   this->mem_ctx = mem_ctx;
   this->lin_ctx = linear_context(this->mem_ctx);
   this->grf_count = grf_count;
   this->post_reg_alloc = post_reg_alloc;

   this->nodes_len = s->cfg->total_instructions;
   this->nodes = linear_zalloc_array(lin_ctx, schedule_node, this->nodes_len);

   const struct brw_isa_info *isa = &s->compiler->isa;

   schedule_node *n = nodes;
   foreach_block_and_inst(block, brw_inst, inst, s->cfg) {
      n->inst = inst;

      if (!post_reg_alloc)
         n->latency = 1;
      else
         n->set_latency(isa);

      n++;
   }
   assert(n == nodes + nodes_len);

   current.block = NULL;
   current.start = NULL;
   current.end = NULL;
   current.len = 0;
   current.time = 0;
   current.cand_generation = 0;
   current.available.make_empty();

   this->hw_reg_count = hw_reg_count;
   this->mode = BRW_SCHEDULE_NONE;
   this->reg_pressure = 0;

   if (!post_reg_alloc) {
      this->reg_pressure_in = linear_zalloc_array(lin_ctx, int, block_count);

      this->livein = linear_alloc_array(lin_ctx, BITSET_WORD *, block_count);
      for (int i = 0; i < block_count; i++)
         this->livein[i] = linear_zalloc_array(lin_ctx, BITSET_WORD,
                                         BITSET_WORDS(grf_count));

      this->liveout = linear_alloc_array(lin_ctx, BITSET_WORD *, block_count);
      for (int i = 0; i < block_count; i++)
         this->liveout[i] = linear_zalloc_array(lin_ctx, BITSET_WORD,
                                          BITSET_WORDS(grf_count));

      this->hw_liveout = linear_alloc_array(lin_ctx, BITSET_WORD *, block_count);
      for (int i = 0; i < block_count; i++)
         this->hw_liveout[i] = linear_zalloc_array(lin_ctx, BITSET_WORD,
                                             BITSET_WORDS(hw_reg_count));

      setup_liveness(s->cfg);

      this->written = linear_alloc_array(lin_ctx, bool, grf_count);

      this->reads_remaining = linear_alloc_array(lin_ctx, int, grf_count);

      this->hw_reads_remaining = linear_alloc_array(lin_ctx, int, hw_reg_count);
   } else {
      this->reg_pressure_in = NULL;
      this->livein = NULL;
      this->liveout = NULL;
      this->hw_liveout = NULL;
      this->written = NULL;
      this->reads_remaining = NULL;
      this->hw_reads_remaining = NULL;
      this->max_vgrf_size = MAX_VGRF_SIZE(s->devinfo);
   }

   this->last_grf_write = linear_zalloc_array(lin_ctx, schedule_node *, grf_count * this->max_vgrf_size);

   const brw_ip_ranges &ips = s->ip_ranges_analysis.require();

   foreach_block(block, s->cfg) {
      set_current_block(block, ips);

      for (schedule_node *n = current.start; n < current.end; n++)
         n->issue_time = calculate_issue_time(n->inst);

      calculate_deps();
      compute_delays();
      compute_exits();
   }
}

static bool
is_src_duplicate(const brw_inst *inst, int src)
{
   for (int i = 0; i < src; i++)
     if (inst->src[i].equals(inst->src[src]))
       return true;

  return false;
}

void
brw_instruction_scheduler::count_reads_remaining(const brw_inst *inst)
{
   assert(reads_remaining);

   for (int i = 0; i < inst->sources; i++) {
      if (is_src_duplicate(inst, i))
         continue;

      if (inst->src[i].file == VGRF) {
         reads_remaining[inst->src[i].nr]++;
      } else if (inst->src[i].file == FIXED_GRF) {
         if (inst->src[i].nr >= hw_reg_count)
            continue;

         for (unsigned j = 0; j < regs_read(s->devinfo, inst, i); j++)
            hw_reads_remaining[inst->src[i].nr + j]++;
      }
   }
}

void
brw_instruction_scheduler::setup_liveness(cfg_t *cfg)
{
   const brw_live_variables &live = s->live_analysis.require();
   const brw_ip_ranges &ips = s->ip_ranges_analysis.require();

   /* First, compute liveness on a per-GRF level using the in/out sets from
    * liveness calculation.
    */
   for (int block = 0; block < cfg->num_blocks; block++) {
      for (int i = 0; i < live.num_vars; i++) {
         if (BITSET_TEST(live.block_data[block].livein, i)) {
            int vgrf = live.vgrf_from_var[i];
            if (!BITSET_TEST(livein[block], vgrf)) {
               reg_pressure_in[block] += s->alloc.sizes[vgrf];
               BITSET_SET(livein[block], vgrf);
            }
         }

         if (BITSET_TEST(live.block_data[block].liveout, i))
            BITSET_SET(liveout[block], live.vgrf_from_var[i]);
      }
   }

   /* Now, extend the live in/live out sets for when a range crosses a block
    * boundary, which matches what our register allocator/interference code
    * does to account for force_writemask_all and incompatible exec_mask's.
    */
   for (int block = 0; block < cfg->num_blocks - 1; block++) {
      for (int i = 0; i < grf_count; i++) {
         const int block_end = ips.range(cfg->blocks[block]).last();
         const brw_range vgrf_range = live.vgrf_range[i];

         if (vgrf_range.contains(block_end) &&
             vgrf_range.contains(block_end + 1)) {
            if (!BITSET_TEST(livein[block + 1], i)) {
                reg_pressure_in[block + 1] += s->alloc.sizes[i];
                BITSET_SET(livein[block + 1], i);
            }

            BITSET_SET(liveout[block], i);
         }
      }
   }

   int *payload_last_use_ip = ralloc_array(NULL, int, hw_reg_count);
   s->calculate_payload_ranges(true, hw_reg_count, payload_last_use_ip);

   for (unsigned i = 0; i < hw_reg_count; i++) {
      if (payload_last_use_ip[i] == -1)
         continue;

      for (int block = 0; block < cfg->num_blocks; block++) {
         brw_range range = ips.range(cfg->blocks[block]);

         if (range.start <= payload_last_use_ip[i])
            reg_pressure_in[block]++;

         if (range.last() <= payload_last_use_ip[i])
            BITSET_SET(hw_liveout[block], i);
      }
   }

   this->max_vgrf_size = live.max_vgrf_size;

   ralloc_free(payload_last_use_ip);
}

void
brw_instruction_scheduler::update_register_pressure(const brw_inst *inst)
{
   assert(reads_remaining);

   if (inst->dst.file == VGRF) {
      written[inst->dst.nr] = true;
   }

   for (int i = 0; i < inst->sources; i++) {
      if (is_src_duplicate(inst, i))
          continue;

      if (inst->src[i].file == VGRF) {
         reads_remaining[inst->src[i].nr]--;
      } else if (inst->src[i].file == FIXED_GRF &&
                 inst->src[i].nr < hw_reg_count) {
         for (unsigned off = 0; off < regs_read(s->devinfo, inst, i); off++)
            hw_reads_remaining[inst->src[i].nr + off]--;
      }
   }
}

int
brw_instruction_scheduler::get_register_pressure_benefit(const brw_inst *inst)
{
   int benefit = 0;
   const int block_idx = current.block->num;

   if (inst->dst.file == VGRF) {
      if (!BITSET_TEST(livein[block_idx], inst->dst.nr) &&
          !written[inst->dst.nr])
         benefit -= s->alloc.sizes[inst->dst.nr];
   }

   for (int i = 0; i < inst->sources; i++) {
      if (is_src_duplicate(inst, i))
         continue;

      if (inst->src[i].file == VGRF &&
          !BITSET_TEST(liveout[block_idx], inst->src[i].nr) &&
          reads_remaining[inst->src[i].nr] == 1)
         benefit += s->alloc.sizes[inst->src[i].nr];

      if (inst->src[i].file == FIXED_GRF &&
          inst->src[i].nr < hw_reg_count) {
         for (unsigned off = 0; off < regs_read(s->devinfo, inst, i); off++) {
            int reg = inst->src[i].nr + off;
            if (!BITSET_TEST(hw_liveout[block_idx], reg) &&
                hw_reads_remaining[reg] == 1) {
               benefit++;
            }
         }
      }
   }

   return benefit;
}

void
brw_instruction_scheduler::set_current_block(bblock_t *block, const brw_ip_ranges &ips)
{
   current.block = block;
   current.start = nodes + ips.range(block).start;
   current.len = block->num_instructions;
   current.end = current.start + current.len;
   current.time = 0;
   current.scheduled = 0;
   current.cand_generation = 1;
}

/** Computation of the delay member of each node. */
void
brw_instruction_scheduler::compute_delays()
{
   for (schedule_node *n = current.end - 1; n >= current.start; n--) {
      if (!n->children_count) {
         n->delay = n->issue_time;
      } else {
         for (int i = 0; i < n->children_count; i++) {
            if (n->children[i].n->delay == 0) {
               /* This is a special case for address register, where a child
                * could be a prior instruction.
                *
                * This ensures that a address register write instruction will
                * always unblock the reader of the address register. Otherwise
                * we could end up with scheduling deadlocks.
                */
               assert(n->children[i].n->inst->dst.is_address());
               n->delay = MAX2(n->delay, 1);
            } else {
               n->delay = MAX2(n->delay, n->latency + n->children[i].n->delay);
            }
         }
      }
   }
}

void
brw_instruction_scheduler::compute_exits()
{
   /* Calculate a lower bound of the scheduling time of each node in the
    * graph.  This is analogous to the node's critical path but calculated
    * from the top instead of from the bottom of the block.
    */
   for (schedule_node *n = current.start; n < current.end; n++) {
      for (int i = 0; i < n->children_count; i++) {
         schedule_node_child *child = &n->children[i];
         child->n->initial_unblocked_time =
            MAX2(child->n->initial_unblocked_time,
                 n->initial_unblocked_time + n->issue_time + child->effective_latency);
      }
   }

   /* Calculate the exit of each node by induction based on the exit nodes of
    * its children.  The preferred exit of a node is the one among the exit
    * nodes of its children which can be unblocked first according to the
    * optimistic unblocked time estimate calculated above.
    */
   for (schedule_node *n = current.end - 1; n >= current.start; n--) {
      n->exit = (n->inst->opcode == BRW_OPCODE_HALT ? n : NULL);

      for (int i = 0; i < n->children_count; i++) {
         if (exit_initial_unblocked_time(n->children[i].n) < exit_initial_unblocked_time(n))
            n->exit = n->children[i].n->exit;
      }
   }
}

/**
 * Add a dependency between two instruction nodes.
 *
 * The @after node will be scheduled after @before.  We will try to
 * schedule it @latency cycles after @before, but no guarantees there.
 */
void
brw_instruction_scheduler::add_dep(schedule_node *before, schedule_node *after,
                               int latency)
{
   if (!before || !after)
      return;

   assert(before != after);

   for (int i = 0; i < before->children_count; i++) {
      schedule_node_child *child = &before->children[i];
      if (child->n == after) {
         child->effective_latency = MAX2(child->effective_latency, latency);
         return;
      }
   }

   if (before->children_cap <= before->children_count) {
      if (before->children_cap < 16)
         before->children_cap = 16;
      else
         before->children_cap *= 2;

      before->children = reralloc(mem_ctx, before->children,
                                  schedule_node_child,
                                  before->children_cap);
   }

   schedule_node_child *child = &before->children[before->children_count];
   child->n = after;
   child->effective_latency = latency;
   before->children_count++;
   after->initial_parent_count++;

   /* Propagate the dependency to the address register instructions. */
   for (int i = 0; i < after->address_read_count; i++)
      add_dep(before, after->address_read[i]);
}

void
brw_instruction_scheduler::add_dep(schedule_node *before, schedule_node *after)
{
   if (!before)
      return;

   add_dep(before, after, before->latency);
}

void
brw_instruction_scheduler::add_address_dep(schedule_node *before, schedule_node *after)
{
   assert(before && after);

   add_dep(before, after, before->latency);

   if (after->address_read_cap <= after->address_read_count) {
      after->address_read_cap = MAX2(2 * after->address_read_cap, 1);

      after->address_read = reralloc(mem_ctx, after->address_read,
                                     schedule_node *,
                                     after->address_read_cap);
   }

   after->address_read[after->address_read_count++] = before;
}

static bool
is_scheduling_barrier(const brw_inst *inst)
{
   return inst->opcode == SHADER_OPCODE_HALT_TARGET ||
          (inst->is_control_flow() && inst->opcode != BRW_OPCODE_HALT) ||
          inst->has_side_effects();
}

static bool
has_cross_lane_access(const brw_inst *inst)
{
   /* FINISHME:
    *
    * This function is likely incomplete in terms of identify cross lane
    * accesses.
    */
   if (inst->opcode == SHADER_OPCODE_BROADCAST ||
       inst->opcode == SHADER_OPCODE_CLUSTER_BROADCAST ||
       inst->opcode == SHADER_OPCODE_SHUFFLE ||
       inst->opcode == FS_OPCODE_LOAD_LIVE_CHANNELS ||
       inst->opcode == SHADER_OPCODE_LOAD_LIVE_CHANNELS ||
       inst->opcode == SHADER_OPCODE_FIND_LAST_LIVE_CHANNEL ||
       inst->opcode == SHADER_OPCODE_FIND_LIVE_CHANNEL)
      return true;

   for (unsigned s = 0; s < inst->sources; s++) {
      if (inst->src[s].file == VGRF) {
         if (inst->src[s].stride == 0)
            return true;
      }
   }

   return false;
}

/**
 * Some register access need dependencies on other instructions.
 */
bool
brw_instruction_scheduler::register_needs_barrier(const brw_reg &reg)
{
   if (reg.file != ARF || reg.is_null())
      return false;

   /* If you look at SR register layout, there is nothing in there that
    * depends on other instructions. This is just fixed dispatch information.
    *
    * ATSM PRMs, Volume 9: Render Engine, State Register Fields :
    *    sr0.0:
    *      - 0:2   TID
    *      - 4:13  Slice, DSS, Subslice, EU IDs
    *      - 20:22 Priority
    *      - 23:23 Priority class
    *      - 24:27 FFID
    *    sr0.1:
    *      - 0:5   IEEE Exception
    *      - 21:31 FFTID
    *    sr0.2:
    *      - 0:31  Dispatch Mask
    *    sr0.3:
    *      - 0:31  Vector Mask
    */
   if (reg.nr == BRW_ARF_STATE)
      return false;

   return true;
}

/**
 * Sometimes we really want this node to execute after everything that
 * was before it and before everything that followed it.  This adds
 * the deps to do so.
 */
void
brw_instruction_scheduler::add_barrier_deps(schedule_node *n)
{
   for (schedule_node *prev = n - 1; prev >= current.start; prev--) {
      add_dep(prev, n, 0);
      if (is_scheduling_barrier(prev->inst))
         break;
   }

   for (schedule_node *next = n + 1; next < current.end; next++) {
      add_dep(n, next, 0);
      if (is_scheduling_barrier(next->inst))
         break;
   }
}

/**
 * Because some instructions like HALT can disable lanes, scheduling prior to
 * a cross lane access should not be allowed, otherwise we could end up with
 * later instructions accessing uninitialized data.
 */
void
brw_instruction_scheduler::add_cross_lane_deps(schedule_node *n)
{
   for (schedule_node *prev = n - 1; prev >= current.start; prev--) {
      if (has_cross_lane_access((brw_inst*)prev->inst))
         add_dep(prev, n, 0);
   }
}

/* instruction scheduling needs to be aware of when an MRF write
 * actually writes 2 MRFs.
 */
bool
brw_instruction_scheduler::is_compressed(const brw_inst *inst)
{
   return inst->exec_size == 16;
}

/* Clears last_grf_write to be ready to start calculating deps for a block
 * again.
 *
 * Since pre-ra grf_count scales with instructions, and instructions scale with
 * BBs, we don't want to memset all of last_grf_write per block or you'll end up
 * O(n^2) with number of blocks.  For shaders using softfp64, we get a *lot* of
 * blocks.
 *
 * We don't bother being careful for post-ra, since then grf_count doesn't scale
 * with instructions.
 */
void
brw_instruction_scheduler::clear_last_grf_write()
{
   if (!post_reg_alloc) {
      for (schedule_node *n = current.start; n < current.end; n++) {
         brw_inst *inst = n->inst;

         if (inst->dst.file == VGRF) {
            /* Don't bother being careful with regs_written(), quicker to just clear 2 cachelines. */
            memset(&last_grf_write[inst->dst.nr * max_vgrf_size], 0,
                   sizeof(*last_grf_write) * max_vgrf_size);
         }
      }
   } else {
      memset(last_grf_write, 0,
             sizeof(*last_grf_write) * grf_count * max_vgrf_size);
   }
}

int
brw_instruction_scheduler::grf_index(const brw_reg &reg)
{
   if (post_reg_alloc)
      return reg.nr;
   return reg.nr * max_vgrf_size + reg.offset / REG_SIZE;
}

void
brw_instruction_scheduler::calculate_deps()
{
   /* Pre-register-allocation, this tracks the last write per VGRF offset.
    * After register allocation, reg_offsets are gone and we track individual
    * GRF registers.
    */
   schedule_node *last_conditional_mod[16] = {};
   schedule_node *last_accumulator_write = NULL;
   /* Fixed HW registers are assumed to be separate from the virtual
    * GRFs, so they can be tracked separately.  We don't really write
    * to fixed GRFs much, so don't bother tracking them on a more
    * granular level.
    */
   schedule_node *last_fixed_grf_write = NULL;
   schedule_node *last_address_write[16] = {};

   /* top-to-bottom dependencies: RAW and WAW. */

   if (!post_reg_alloc) {
      /* Address registers have virtual identifier, allowing us to identify
       * what instructions needs the values written to the register. The
       * address register is written/read in pairs of instructions (enforced
       * by the brw_validate.cpp).
       *
       * To allow scheduling of SEND messages, out of order, without the
       * address register tracking generating serialized dependency between
       * all the messages, we first track all the dependencies of the address
       * register. Those dependencies are added to the instructions consuming
       * the address register value. Then when doing the normal dependency
       * tracking, any node adding a dependency to an instruction consuming
       * the address register is also added as dependency to the instruction
       * writing the value to the address register.
       *
       * This scheme allows the scheduling done by
       * choose_instruction_to_schedule() to ensure that once an instruction
       * writing the address register is scheduled, we can always schedule all
       * instructions making use of the address register value. Otherwise we
       * could run into scheduling deadlocks.
       *
       * Here is a deadlock example :
       *
       *    mov    a0, 0x42
       *    send grf1, ..., a0
       *    mov    a0, 0x43
       *    send grf2, grf1, a0
       *
       * Let say choose_instruction_to_schedule() chooses the second mov
       * instruction first (mov a0, 0x43). Then it cannot schedule the second
       * send instruction because the first send instruction populating grf1
       * and has not been scheduled and we cannot schedule the first mov
       * either because the address register is already in use for another
       * message.
       *
       * In post-register-allocation mode, this scheme cannot work as all GRFs
       * can get reused and we have to serializae all address register usages
       * (like the accumulator, flag, etc...).
       */
      for (schedule_node *n = current.start; n < current.end; n++) {
         brw_inst *inst = n->inst;

         /* Pre pass going over instruction using the register flag as a
          * source.
          */
         for (int i = 0; i < inst->sources; i++) {
            if (!inst->src[i].is_address())
               continue;

            for (unsigned byte = 0; byte < inst->size_read(s->devinfo, i); byte += 2) {
               assert(inst->src[i].address_slot(byte) < ARRAY_SIZE(last_address_write));
               schedule_node *write_addr_node =
                  last_address_write[inst->src[i].address_slot(byte)];
               assert(write_addr_node->inst->dst.nr == inst->src[i].nr);
               add_address_dep(write_addr_node, n);
            }
         }

         if (inst->dst.is_address()) {
            for (unsigned byte = 0; byte < inst->size_written; byte += 2) {
               last_address_write[inst->dst.address_slot(byte)] = n;
            }
         }
      }
   }

   for (schedule_node *n = current.start; n < current.end; n++) {
      brw_inst *inst = n->inst;

      if (is_scheduling_barrier(inst))
         add_barrier_deps(n);

      if (inst->opcode == BRW_OPCODE_HALT ||
          inst->opcode == SHADER_OPCODE_HALT_TARGET)
          add_cross_lane_deps(n);

      /* read-after-write deps. */
      for (int i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == VGRF) {
            for (unsigned r = 0; r < regs_read(s->devinfo, inst, i); r++)
               add_dep(last_grf_write[grf_index(inst->src[i]) + r], n);
         } else if (inst->src[i].file == FIXED_GRF) {
            if (post_reg_alloc) {
               for (unsigned r = 0; r < regs_read(s->devinfo, inst, i); r++)
                  add_dep(last_grf_write[inst->src[i].nr + r], n);
            } else {
               add_dep(last_fixed_grf_write, n);
            }
         } else if (inst->src[i].is_accumulator()) {
            add_dep(last_accumulator_write, n);
         } else if (inst->src[i].is_address()) {
            if (post_reg_alloc) {
               for (unsigned byte = 0; byte < inst->size_read(s->devinfo, i); byte += 2)
                  add_dep(last_address_write[inst->src[i].address_slot(byte)], n);
            }
         } else if (register_needs_barrier(inst->src[i])) {
            add_barrier_deps(n);
         }
      }

      if (const unsigned mask = inst->flags_read(s->devinfo)) {
         assert(mask < (1 << ARRAY_SIZE(last_conditional_mod)));

         for (unsigned i = 0; i < ARRAY_SIZE(last_conditional_mod); i++) {
            if (mask & (1 << i))
               add_dep(last_conditional_mod[i], n);
         }
      }

      if (inst->reads_accumulator_implicitly()) {
         add_dep(last_accumulator_write, n);
      }

      /* write-after-write deps. */
      if (inst->dst.file == VGRF) {
         int grf_idx = grf_index(inst->dst);
         for (unsigned r = 0; r < regs_written(inst); r++) {
            add_dep(last_grf_write[grf_idx + r], n);
            last_grf_write[grf_idx + r] = n;
         }
      } else if (inst->dst.file == FIXED_GRF) {
         if (post_reg_alloc) {
            for (unsigned r = 0; r < regs_written(inst); r++) {
               add_dep(last_grf_write[inst->dst.nr + r], n);
               last_grf_write[inst->dst.nr + r] = n;
            }
         } else {
            add_dep(last_fixed_grf_write, n);
            last_fixed_grf_write = n;
         }
      } else if (inst->dst.is_accumulator()) {
         add_dep(last_accumulator_write, n);
         last_accumulator_write = n;
      } else if (inst->dst.is_address()) {
         if (post_reg_alloc) {
            for (unsigned byte = 0; byte < inst->size_written; byte += 2) {
               add_dep(last_address_write[inst->dst.address_slot(byte)], n);
               last_address_write[inst->dst.address_slot(byte)] = n;
            }
         }
      } else if (register_needs_barrier(inst->dst)) {
         add_barrier_deps(n);
      }

      if (const unsigned mask = inst->flags_written(s->devinfo)) {
         assert(mask < (1 << ARRAY_SIZE(last_conditional_mod)));

         for (unsigned i = 0; i < ARRAY_SIZE(last_conditional_mod); i++) {
            if (mask & (1 << i)) {
               add_dep(last_conditional_mod[i], n, 0);
               last_conditional_mod[i] = n;
            }
         }
      }

      if (inst->writes_accumulator_implicitly(s->devinfo) &&
          !inst->dst.is_accumulator()) {
         add_dep(last_accumulator_write, n);
         last_accumulator_write = n;
      }

      if (post_reg_alloc && inst->uses_address_register_implicitly()) {
         for (unsigned i = 0; i < ARRAY_SIZE(last_address_write); i++) {
            add_dep(last_address_write[i], n);
            last_address_write[i] = n;
         }
      }
   }

   clear_last_grf_write();

   /* bottom-to-top dependencies: WAR */
   memset(last_conditional_mod, 0, sizeof(last_conditional_mod));
   last_accumulator_write = NULL;
   last_fixed_grf_write = NULL;
   memset(last_address_write, 0, sizeof(last_address_write));

   for (schedule_node *n = current.end - 1; n >= current.start; n--) {
      brw_inst *inst = n->inst;

      /* write-after-read deps. */
      for (int i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == VGRF) {
            for (unsigned r = 0; r < regs_read(s->devinfo, inst, i); r++)
               add_dep(n, last_grf_write[grf_index(inst->src[i]) + r], 0);
         } else if (inst->src[i].file == FIXED_GRF) {
            if (post_reg_alloc) {
               for (unsigned r = 0; r < regs_read(s->devinfo, inst, i); r++)
                  add_dep(n, last_grf_write[inst->src[i].nr + r], 0);
            } else {
               add_dep(n, last_fixed_grf_write, 0);
            }
         } else if (inst->src[i].is_accumulator()) {
            add_dep(n, last_accumulator_write, 0);
         } else if (inst->src[i].is_address()) {
            if (post_reg_alloc) {
               for (unsigned byte = 0; byte < inst->size_read(s->devinfo, i); byte += 2) {
                  add_dep(n, last_address_write[inst->src[i].address_slot(byte)], 0);
               }
            }
         } else if (register_needs_barrier(inst->src[i])) {
            add_barrier_deps(n);
         }
      }

      if (const unsigned mask = inst->flags_read(s->devinfo)) {
         assert(mask < (1 << ARRAY_SIZE(last_conditional_mod)));

         for (unsigned i = 0; i < ARRAY_SIZE(last_conditional_mod); i++) {
            if (mask & (1 << i))
               add_dep(n, last_conditional_mod[i]);
         }
      }

      if (inst->reads_accumulator_implicitly()) {
         add_dep(n, last_accumulator_write);
      }

      if (post_reg_alloc && inst->uses_address_register_implicitly()) {
         for (unsigned i = 0; i < ARRAY_SIZE(last_address_write); i++)
            last_address_write[i] = n;
      }

      /* Update the things this instruction wrote, so earlier reads
       * can mark this as WAR dependency.
       */
      if (inst->dst.file == VGRF) {
         for (unsigned r = 0; r < regs_written(inst); r++)
            last_grf_write[grf_index(inst->dst) + r] = n;
      } else if (inst->dst.file == FIXED_GRF) {
         if (post_reg_alloc) {
            for (unsigned r = 0; r < regs_written(inst); r++)
               last_grf_write[inst->dst.nr + r] = n;
         } else {
            last_fixed_grf_write = n;
         }
      } else if (inst->dst.is_accumulator()) {
         last_accumulator_write = n;
      } else if (inst->dst.is_address()) {
         if (post_reg_alloc) {
            for (unsigned byte = 0; byte < inst->size_written; byte += 2)
               last_address_write[inst->dst.address_slot(byte)] = n;
         }
      } else if (register_needs_barrier(inst->dst)) {
         add_barrier_deps(n);
      }

      if (const unsigned mask = inst->flags_written(s->devinfo)) {
         assert(mask < (1 << ARRAY_SIZE(last_conditional_mod)));

         for (unsigned i = 0; i < ARRAY_SIZE(last_conditional_mod); i++) {
            if (mask & (1 << i))
               last_conditional_mod[i] = n;
         }
      }

      if (inst->writes_accumulator_implicitly(s->devinfo)) {
         last_accumulator_write = n;
      }
   }

   clear_last_grf_write();
}

bool
brw_instruction_scheduler::address_register_interfere(const schedule_node *n)
{
   if (n->inst->uses_address_register_implicitly()) {
      for (unsigned i = 0; i < ARRAY_SIZE(current.address_register); i++)
         if (current.address_register[i] != 0)
            return true;
      return false;
   }

   if (n->inst->dst.is_address()) {
      for (unsigned byte = 0; byte < n->inst->size_written; byte += 2) {
         if (current.address_register[n->inst->dst.address_slot(byte)] != 0 &&
             current.address_register[n->inst->dst.address_slot(byte)] != n->inst->dst.nr)
            return true;
      }
   }

   if (n->address_read_count > 0) {
      for (unsigned i = 0; i < n->inst->sources; i++) {
         if (!n->inst->src[i].is_address())
            continue;
         for (unsigned byte = 0; byte < n->inst->size_read(s->devinfo, i); byte += 2) {
            if (current.address_register[n->inst->src[i].address_slot(byte)] !=
                n->inst->src[i].nr)
               return true;
         }
      }
   }

   return false;
}

schedule_node *
brw_instruction_scheduler::choose_instruction_to_schedule()
{
   schedule_node *chosen = NULL;

   if (mode == BRW_SCHEDULE_PRE || mode == BRW_SCHEDULE_POST) {
      int chosen_time = 0;

      /* Of the instructions ready to execute or the closest to being ready,
       * choose the one most likely to unblock an early program exit, or
       * otherwise the oldest one.
       */
      brw_foreach_in_list(schedule_node, n, &current.available) {
         if (!post_reg_alloc && address_register_interfere(n))
            continue;

         if (!chosen ||
             exit_tmp_unblocked_time(n) < exit_tmp_unblocked_time(chosen) ||
             (exit_tmp_unblocked_time(n) == exit_tmp_unblocked_time(chosen) &&
              n->tmp.unblocked_time < chosen_time)) {
            chosen = n;
            chosen_time = n->tmp.unblocked_time;
         }
      }
   } else {
      int chosen_register_pressure_benefit = 0;

      /* Before register allocation, we don't care about the latencies of
       * instructions.  All we care about is reducing live intervals of
       * variables so that we can avoid register spilling, or get SIMD16
       * shaders which naturally do a better job of hiding instruction
       * latency.
       */
      brw_foreach_in_list(schedule_node, n, &current.available) {
         if (!post_reg_alloc && address_register_interfere(n))
            continue;

         if (!chosen) {
            chosen = n;
            chosen_register_pressure_benefit =
                  get_register_pressure_benefit(chosen->inst);
            continue;
         }

         /* Most important: If we can definitely reduce register pressure, do
          * so immediately.
          */
         int register_pressure_benefit = get_register_pressure_benefit(n->inst);

         if (register_pressure_benefit > 0 &&
             register_pressure_benefit > chosen_register_pressure_benefit) {
            chosen = n;
            chosen_register_pressure_benefit = register_pressure_benefit;
            continue;
         } else if (chosen_register_pressure_benefit > 0 &&
                    (register_pressure_benefit <
                     chosen_register_pressure_benefit)) {
            continue;
         }

         if (mode == BRW_SCHEDULE_PRE_LIFO) {
            /* Prefer instructions that recently became available for
             * scheduling.  These are the things that are most likely to
             * (eventually) make a variable dead and reduce register pressure.
             * Typical register pressure estimates don't work for us because
             * most of our pressure comes from texturing, where no single
             * instruction to schedule will make a vec4 value dead.
             */
            if (n->tmp.cand_generation > chosen->tmp.cand_generation) {
               chosen = n;
               chosen_register_pressure_benefit = register_pressure_benefit;
               continue;
            } else if (n->tmp.cand_generation < chosen->tmp.cand_generation) {
               continue;
            }
         }

         /* For instructions pushed on the cands list at the same time, prefer
          * the one with the highest delay to the end of the program.  This is
          * most likely to have its values able to be consumed first (such as
          * for a large tree of lowered ubo loads, which appear reversed in
          * the instruction stream with respect to when they can be consumed).
          */
         if (n->delay > chosen->delay) {
            chosen = n;
            chosen_register_pressure_benefit = register_pressure_benefit;
            continue;
         } else if (n->delay < chosen->delay) {
            continue;
         }

         /* Prefer the node most likely to unblock an early program exit.
          */
         if (exit_tmp_unblocked_time(n) < exit_tmp_unblocked_time(chosen)) {
            chosen = n;
            chosen_register_pressure_benefit = register_pressure_benefit;
            continue;
         } else if (exit_tmp_unblocked_time(n) > exit_tmp_unblocked_time(chosen)) {
            continue;
         }

         /* If all other metrics are equal, we prefer the first instruction in
          * the list (program execution).
          */
      }
   }

   return chosen;
}

int
brw_instruction_scheduler::calculate_issue_time(const brw_inst *inst)
{
   const struct brw_isa_info *isa = &s->compiler->isa;
   const unsigned overhead = s->grf_used && has_bank_conflict(isa, inst) ?
      DIV_ROUND_UP(inst->dst.component_size(inst->exec_size), REG_SIZE) : 0;
   if (is_compressed(inst))
      return 4 + overhead;
   else
      return 2 + overhead;
}

void
brw_instruction_scheduler::schedule(schedule_node *chosen)
{
   assert(current.scheduled < current.len);
   current.scheduled++;

   assert(chosen);
   chosen->remove();
   current.block->instructions.push_tail(chosen->inst);

   /* If we expected a delay for scheduling, then bump the clock to reflect
    * that.  In reality, the hardware will switch to another hyperthread
    * and may not return to dispatching our thread for a while even after
    * we're unblocked.  After this, we have the time when the chosen
    * instruction will start executing.
    */
   current.time = MAX2(current.time, chosen->tmp.unblocked_time);

   /* Update the clock for how soon an instruction could start after the
    * chosen one.
    */
   current.time += chosen->issue_time;

   if (debug) {
      fprintf(stderr, "clock %4d, scheduled: ", current.time);
      brw_print_instruction(*s, chosen->inst);
   }
}

void
brw_instruction_scheduler::update_children(schedule_node *chosen)
{
   if (chosen->address_read_count > 0) {
      for (unsigned i = 0; i < chosen->inst->sources; i++) {
         if (!chosen->inst->src[i].is_address())
            continue;
         for (unsigned byte = 0; byte < chosen->inst->size_read(s->devinfo, i); byte += 2) {
            assert(chosen->inst->src[i].address_slot(byte) <
                   ARRAY_SIZE(current.address_register));
            current.address_register[chosen->inst->src[i].address_slot(byte)] = 0;
         }
      }
   }

   if (chosen->inst->dst.is_address()) {
      for (unsigned byte = 0; byte < chosen->inst->size_written; byte += 2) {
         assert(chosen->inst->dst.address_slot(byte) <
                ARRAY_SIZE(current.address_register));
         current.address_register[
            chosen->inst->dst.address_slot(byte)] = chosen->inst->dst.nr;
      }
   } else if (chosen->inst->uses_address_register_implicitly()) {
      memset(current.address_register, 0, sizeof(current.address_register));
   }

   /* Now that we've scheduled a new instruction, some of its
    * children can be promoted to the list of instructions ready to
    * be scheduled.  Update the children's unblocked time for this
    * DAG edge as we do so.
    */
   for (int i = chosen->children_count - 1; i >= 0; i--) {
      schedule_node_child *child = &chosen->children[i];

      child->n->tmp.unblocked_time = MAX2(child->n->tmp.unblocked_time,
                                          current.time + child->effective_latency);

      if (debug) {
         fprintf(stderr, "\tchild %d, %d parents: ", i, child->n->tmp.parent_count);
         brw_print_instruction(*s, child->n->inst);
      }

      child->n->tmp.cand_generation = current.cand_generation;
      child->n->tmp.parent_count--;
      if (child->n->tmp.parent_count == 0) {
         if (debug) {
            fprintf(stderr, "\t\tnow available\n");
         }
         current.available.push_head(child->n);
      }
   }
   current.cand_generation++;
}

void
brw_instruction_scheduler::schedule_instructions()
{
   if (!post_reg_alloc)
      reg_pressure = reg_pressure_in[current.block->num];

   assert(current.available.is_empty());
   for (schedule_node *n = current.start; n < current.end; n++) {
      reset_node_tmp(n);

      /* Add DAG heads to the list of available instructions. */
      if (n->tmp.parent_count == 0)
         current.available.push_tail(n);
   }

   current.block->instructions.make_empty();

   memset(current.address_register, 0, sizeof(current.address_register));

   while (!current.available.is_empty()) {
      schedule_node *chosen = choose_instruction_to_schedule();
      schedule(chosen);

      if (!post_reg_alloc) {
         reg_pressure -= get_register_pressure_benefit(chosen->inst);
         update_register_pressure(chosen->inst);
         if (debug)
            fprintf(stderr, "(register pressure %d)\n", reg_pressure);
      }

      update_children(chosen);
   }
}

void
brw_instruction_scheduler::run(brw_instruction_scheduler_mode mode)
{
   this->mode = mode;

   if (debug && !post_reg_alloc) {
      fprintf(stderr, "\nInstructions before scheduling (reg_alloc %d)\n",
              post_reg_alloc);
         brw_print_instructions(*s);
   }

   if (!post_reg_alloc) {
      memset(reads_remaining, 0, grf_count * sizeof(*reads_remaining));
      memset(hw_reads_remaining, 0, hw_reg_count * sizeof(*hw_reads_remaining));
      memset(written, 0, grf_count * sizeof(*written));
   }

   const brw_ip_ranges &ips = s->ip_ranges_analysis.require();

   foreach_block(block, s->cfg) {
      set_current_block(block, ips);

      if (!post_reg_alloc) {
         for (schedule_node *n = current.start; n < current.end; n++)
            count_reads_remaining(n->inst);
      }

      schedule_instructions();
   }

   if (debug && !post_reg_alloc) {
      fprintf(stderr, "\nInstructions after scheduling (reg_alloc %d)\n",
              post_reg_alloc);
      brw_print_instructions(*s);
   }
}

brw_instruction_scheduler *
brw_prepare_scheduler(brw_shader &s, void *mem_ctx)
{
   const int grf_count = s.alloc.count;

   brw_instruction_scheduler *empty = rzalloc(mem_ctx, brw_instruction_scheduler);
   return new (empty) brw_instruction_scheduler(mem_ctx, &s, grf_count, s.first_non_payload_grf,
                                                s.cfg->num_blocks, /* post_reg_alloc */ false);
}

void
brw_schedule_instructions_pre_ra(brw_shader &s, brw_instruction_scheduler *sched,
                                 brw_instruction_scheduler_mode mode)
{
   if (mode == BRW_SCHEDULE_NONE)
      return;

   sched->run(mode);

   s.invalidate_analysis(BRW_DEPENDENCY_INSTRUCTIONS);
}

void
brw_schedule_instructions_post_ra(brw_shader &s)
{
   const bool post_reg_alloc = true;
   const int grf_count = reg_unit(s.devinfo) * s.grf_used;

   void *mem_ctx = ralloc_context(NULL);

   brw_instruction_scheduler sched(mem_ctx, &s, grf_count, s.first_non_payload_grf,
                                   s.cfg->num_blocks, post_reg_alloc);
   sched.run(BRW_SCHEDULE_POST);

   ralloc_free(mem_ctx);

   s.invalidate_analysis(BRW_DEPENDENCY_INSTRUCTIONS);
}
