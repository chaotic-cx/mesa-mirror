/*
 * Copyright 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "brw_shader.h"
#include "brw_builder.h"

#include <vector>

/**
 * \file
 *
 * Attempt to eliminate spurious fills and spills.
 *
 * NOTE: This pass is run after register allocation but before
 * brw_lower_vgrfs_to_fixed_grfs.
 */

static void
validate_spill_fill_counts(brw_shader &s)
{
#ifndef NDEBUG
   unsigned spill_count = 0;
   unsigned fill_count = 0;

   foreach_block_and_inst (block, brw_inst, inst, s.cfg) {
      switch (inst->opcode) {
      case SHADER_OPCODE_LSC_FILL:
         fill_count++;
         break;
      case SHADER_OPCODE_LSC_SPILL:
         spill_count++;
         break;
      default:
         break;
      }
   }

   /* It would be great if these could be checked for equality, but
    * nir_intrinsic_load_scratch_intel and nir_intrinsic_store_scratch_intel
    * translate into generic SENDs that count as fills and spills.
    */
   assert(spill_count <= s.shader_stats.spill_count);
   assert(fill_count <= s.shader_stats.fill_count);
#endif
}

static bool
scratch_intersects(const intel_device_info *devinfo,
                   const brw_scratch_inst *a, const brw_scratch_inst *b)
{
   const auto a_first = a->logical_offset;
   const auto a_last = (a->opcode == SHADER_OPCODE_LSC_SPILL ?
                        a->size_read(devinfo, SPILL_SRC_PAYLOAD2) :
                        a->size_written) + a_first - 1;
   const auto b_first = b->logical_offset;
   const auto b_last = (b->opcode == SHADER_OPCODE_LSC_SPILL ?
                        b->size_read(devinfo, SPILL_SRC_PAYLOAD2) :
                        b->size_written) + b_first - 1;

   return a_last >= b_first && b_last >= a_first;
}

static bool
scratch_superset(const intel_device_info *devinfo,
                   const brw_scratch_inst *super, const brw_scratch_inst *sub)
{
   const auto a_first = super->logical_offset;
   const auto a_last = (super->opcode == SHADER_OPCODE_LSC_SPILL ?
                        super->size_read(devinfo, SPILL_SRC_PAYLOAD2) :
                        super->size_written) + a_first - 1;
   const auto b_first = sub->logical_offset;
   const auto b_last = (sub->opcode == SHADER_OPCODE_LSC_SPILL ?
                        sub->size_read(devinfo, SPILL_SRC_PAYLOAD2) :
                        sub->size_written) + b_first - 1;

   return a_first <= b_first && a_last >= b_last;
}

static brw_inst *
generate_mov(brw_shader &s, brw_inst *inst, const brw_reg &src)
{
   const unsigned reg_count = DIV_ROUND_UP(inst->size_written, REG_SIZE);
   const unsigned max_reg_count = 2 * reg_unit(s.devinfo);
   brw_reg dst = brw_lower_vgrf_to_fixed_grf(s.devinfo, inst, inst->dst);

   assert(src.file != VGRF);

   /* This can occur for fills in wider SIMD modes. In SIMD32 on Xe2, a fill
    * to r16 followed by a fill to r17 from the same location can't be
    * trivially replaced. The resulting `mov(32) r17, r16` would have the same
    * problems of memcpy with overlapping ranges.
    */
   if (regions_overlap(dst, inst->size_written,
                       src, inst->size_written)) {
      brw_builder bld = brw_builder(inst);
      const unsigned exec_size = 8 * reg_unit(s.devinfo);
      const bool reverse = inst->dst.nr > src.nr;
      unsigned size_written = 0;

      brw_reg dst = inst->dst;
      brw_reg tmp = src;

      unsigned i = 0;
      while (size_written < inst->size_written) {
         brw_inst *mov = bld.group(exec_size, i++).MOV(dst, tmp);

         dst = offset(dst, mov->size_written / REG_SIZE);
         tmp = offset(tmp, mov->size_written / REG_SIZE);
         size_written += mov->size_written;

         if (reverse)
            bld = bld.before(mov);
      }

      assert(size_written == inst->size_written);
      return brw_transform_inst(s, inst, BRW_OPCODE_NOP);
   }

   /* If the resulting MOV would try to write more than 2 registers, emit
    * multiple MOVs instead.
    */
   if (reg_count > max_reg_count) {
      brw_builder bld = brw_builder(inst);
      const unsigned exec_size = MIN2(inst->exec_size, 16 * reg_unit(s.devinfo));
      unsigned size_written = 0;

      brw_reg dst = inst->dst;
      brw_reg tmp = src;

      unsigned i = 0;
      while (size_written < inst->size_written) {
         brw_inst *mov = bld.group(exec_size, i++).MOV(dst, tmp);

         dst = offset(dst, mov->size_written / REG_SIZE);
         tmp = offset(tmp, mov->size_written / REG_SIZE);
         size_written += mov->size_written;
      }

      assert(size_written == inst->size_written);
      return brw_transform_inst(s, inst, BRW_OPCODE_NOP);
   }

   inst = brw_transform_inst(s, inst, BRW_OPCODE_MOV);
   inst->src[0] = src;

   return inst;
}

static bool
propagate_fill_to_fill(brw_shader &s, brw_inst *inst, brw_inst **tmp)
{
   const intel_device_info *devinfo = s.devinfo;
   unsigned num_tmp = 0;
   bool dst_was_invalidated = false;

   brw_reg inst_dst = brw_lower_vgrf_to_fixed_grf(devinfo, inst, inst->dst);

   foreach_inst_in_block_starting_from(brw_inst, scan_inst, inst) {
      /* Instruction is a fill from the same location as the previous fill. */
      brw_reg scan_dst = brw_lower_vgrf_to_fixed_grf(devinfo, scan_inst,
                                                     scan_inst->dst);

      if (scan_inst->opcode == SHADER_OPCODE_LSC_FILL &&
          scan_inst->force_writemask_all == inst->force_writemask_all &&
          scan_inst->as_scratch()->logical_offset == inst->as_scratch()->logical_offset &&
          scan_inst->size_written == inst->size_written &&
          scan_inst->group == inst->group &&
          scan_inst->as_scratch()->use_transpose == inst->as_scratch()->use_transpose) {
         tmp[num_tmp++] = scan_inst;

         /* Write to the register being filled invalidates the value. */
         if (!scan_dst.equals(inst_dst) &&
             regions_overlap(scan_dst, scan_inst->size_written,
                             inst_dst, inst->size_written)) {
            dst_was_invalidated = true;
            break;
         }
      } else {
         /* A spill to the same location invalidates the value. */
         if (scan_inst->opcode == SHADER_OPCODE_LSC_SPILL &&
             scratch_intersects(devinfo, inst->as_scratch(),
                                scan_inst->as_scratch())) {
            break;
         }

         /* Write to the register being filled invalidates the value. */
         if (regions_overlap(scan_dst, scan_inst->size_written,
                             inst_dst, inst->size_written)) {
            dst_was_invalidated = true;
            break;
         }
      }
   }

   if (num_tmp == 0)
      return false;

   s.shader_stats.fill_count -= num_tmp;

   /* Process the marked copies in reverse order. This ensures that a sequence
    * like
    *
    *    fill    r30, XYZ
    *    ...
    *    fill    r31, XYZ
    *    ...
    *    fill    r33, XYZ
    *
    * will become
    *
    *    fill    r30, XYZ
    *    ...
    *    mov     r31, r30
    *    ...
    *    mov     r33, r30
    *
    * Otherwise the last MOV would be from r31.
    */
   while (num_tmp-- > 0) {
      brw_inst *scan_inst = tmp[num_tmp];
      brw_reg scan_dst = brw_lower_vgrf_to_fixed_grf(devinfo, scan_inst,
                                                     scan_inst->dst);

      /* Only attempt to copy-from-a-copy if the original fill destination was
       * actually invalidated by some other write.
       */
      if (dst_was_invalidated) {
         /* Using &tmp[num_tmp] seems sketchy, but it is safe. The tmp array
          * has enough space for every instruction in the block. The worst
          * case scenario is that every instruction before (but not including)
          * scan_inst is in tmp and every instruction after (but not
          * including) scan_inst will be added.
          */
         propagate_fill_to_fill(s, scan_inst, &tmp[num_tmp]);
      }

      if (scan_dst.equals(inst_dst)) {
         brw_transform_inst(s, scan_inst, BRW_OPCODE_NOP);
      } else {
         generate_mov(s, scan_inst, inst_dst);
      }
   }

   return true;
}

bool
brw_opt_fill_and_spill(brw_shader &s)
{
   assert(s.info.grf_used > 0);

   const intel_device_info *devinfo = s.devinfo;
   bool progress = false;

   std::vector<brw_inst *> tracked_spills, tracked_fills;

   brw_inst **tmp = NULL;
   unsigned tmp_size = 0;

   validate_spill_fill_counts(s);

   foreach_block(block, s.cfg) {
      bool block_progress = false;

      if (tmp_size < block->num_instructions) {
         tmp_size = block->num_instructions;
         delete[] tmp;
         tmp = new brw_inst *[tmp_size];
      }

      foreach_inst_in_block(brw_inst, inst, block) {
         if (inst->opcode != SHADER_OPCODE_LSC_SPILL)
            continue;

         tracked_spills.push_back(inst);

         const brw_reg spilled =
            brw_lower_vgrf_to_fixed_grf(devinfo, inst,
                                        inst->src[SPILL_SRC_PAYLOAD2]);

         /* Check for a fill from the same location while the register being
          * spilled still contains the data. In this case, replace the fill
          * with a simple move.
          */
         foreach_inst_in_block_starting_from(brw_inst, scan_inst, inst) {
            /* Write to the register being spilled invalidates the value. */
            const brw_reg scan_dst =
               brw_lower_vgrf_to_fixed_grf(devinfo, scan_inst, scan_inst->dst);

            if (regions_overlap(scan_dst, scan_inst->size_written,
                                spilled,
                                inst->size_read(devinfo, SPILL_SRC_PAYLOAD2))) {
               break;
            }

            /* Spill to the same location invalidates the value. */
            if (scan_inst->opcode == SHADER_OPCODE_LSC_SPILL &&
                scratch_intersects(devinfo, scan_inst->as_scratch(),
                                   inst->as_scratch())) {
               break;
            }

            /* Instruction is a fill from the same location as the spill. */
            if (scan_inst->opcode == SHADER_OPCODE_LSC_FILL &&
                scan_inst->force_writemask_all == inst->force_writemask_all &&
                scan_inst->as_scratch()->logical_offset == inst->as_scratch()->logical_offset) {
               /* This limitation is necessary because (currently) a spill may
                * be split into multiple writes while the correspoing fill is
                * implemented as a single transpose read. When this occurs,
                * this optimization pass would have to be smarter than it
                * currently is.
                *
                * FINISHME: This would not be an issue if the splitting
                * occured during spill lowering.
                */
               if (scan_inst->size_written != inst->size_read(devinfo, SPILL_SRC_PAYLOAD2))
                  continue;

               propagate_fill_to_fill(s, scan_inst, tmp);

               if (scan_inst->dst.equals(inst->src[SPILL_SRC_PAYLOAD2])) {
                  scan_inst = brw_transform_inst(s, scan_inst, BRW_OPCODE_NOP);
               } else {
                  scan_inst = generate_mov(s, scan_inst, spilled);
               }

               assert(scan_inst->opcode != SHADER_OPCODE_LSC_FILL);

               s.shader_stats.fill_count--;
               block_progress = true;
            }
         }

         /* Scan again. This time check whether there is a spill to the same
          * location without an intervening fill from that location. In this
          * case, the first spill is "killed" and can be removed.
          */
         foreach_inst_in_block_starting_from(brw_inst, scan_inst, inst) {
            if (scan_inst->opcode == SHADER_OPCODE_LSC_FILL &&
                scratch_intersects(devinfo, inst->as_scratch(),
                                   scan_inst->as_scratch())) {
               break;
            }

            if (scan_inst->opcode == SHADER_OPCODE_LSC_SPILL &&
                scratch_superset(devinfo, scan_inst->as_scratch(),
                                 inst->as_scratch())) {
               inst = brw_transform_inst(s, inst, BRW_OPCODE_NOP);
               s.shader_stats.spill_count--;
               block_progress = true;
               break;
            }
         }
      }

      /* Optimize multiple fills from the same offset in a single block. */
      foreach_inst_in_block(brw_inst, inst, block) {
         if (inst->opcode != SHADER_OPCODE_LSC_FILL)
            continue;

         tracked_fills.push_back(inst);

         if (propagate_fill_to_fill(s, inst, tmp))
            block_progress = true;
      }

      if (block_progress) {
         foreach_inst_in_block_safe(brw_inst, inst, block) {
            if (inst->opcode == BRW_OPCODE_NOP)
               inst->remove();
         }

         progress = true;
      }
   }

   delete[] tmp;
   tmp = NULL;

   /* Remove any left-over spill that has no fills.  This can
    * happen when RA decides to spill a value but the value remains
    * live for all its fills.
    *
    * Use the logical scratch offset here instead of the physical offset: the
    * physical storage may be reused by non-interfering spilled values, so a
    * remaining fill from the same physical byte range does not necessarily
    * read this spill's value.
    */
   {
      /* Spills and fills operate on register granularity. */
      const unsigned scratch_regs =
         DIV_ROUND_UP(s.last_logical_scratch, REG_SIZE);
      BITSET_WORD *covered =
         rzalloc_array(NULL, BITSET_WORD, BITSET_WORDS(scratch_regs));

      for (const brw_inst *inst : tracked_fills) {
         /* Some may have already been transformed. */
         if (inst->opcode != SHADER_OPCODE_LSC_FILL)
            continue;
         const brw_scratch_inst *fill = inst->as_scratch();
         assert(fill->logical_offset % REG_SIZE == 0 &&
                fill->size_written % REG_SIZE == 0);
         const unsigned first = fill->logical_offset / REG_SIZE;
         const unsigned end = first + fill->size_written / REG_SIZE;
         assert(end <= scratch_regs);
         BITSET_SET_RANGE(covered, first, end - 1);
      }

      for (brw_inst *inst : tracked_spills) {
         /* Some may have already been transformed. */
         if (inst->opcode != SHADER_OPCODE_LSC_SPILL)
            continue;
         const brw_scratch_inst *spill = inst->as_scratch();
         const unsigned size = spill->size_read(devinfo, SPILL_SRC_PAYLOAD2);
         assert(spill->logical_offset % REG_SIZE == 0 &&
                size % REG_SIZE == 0);
         const unsigned first = spill->logical_offset / REG_SIZE;
         const unsigned end = first + size / REG_SIZE;
         assert(end <= scratch_regs);
         if (!BITSET_TEST_RANGE(covered, first, end - 1)) {
            inst->remove();
            s.shader_stats.spill_count--;
            progress = true;
         }
      }

      ralloc_free(covered);
   }

   if (progress)
      s.invalidate_analysis(BRW_DEPENDENCY_INSTRUCTIONS |
                            BRW_DEPENDENCY_VARIABLES);

   validate_spill_fill_counts(s);
   return progress;
}
