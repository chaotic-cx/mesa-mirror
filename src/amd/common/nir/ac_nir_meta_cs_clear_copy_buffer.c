/*
 * Copyright 2024 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "ac_descriptors.h"
#include "ac_nir_meta.h"
#include "ac_nir_helpers.h"
#include "nir_builder.h"
#include "util/helpers.h"

static void
store_buffer(nir_builder *b, const ac_cs_clear_copy_buffer_key *const key,
             nir_def *store_val, nir_def *buf, nir_def *offset,
             const enum gl_access_qualifier access)
{
   if (key->addr_user_data) {
      nir_store_global_amd(b, store_val, buf, offset,
                           .access = access);
      return;
   }

   nir_store_ssbo(b, store_val, buf, offset,
                  .access = access);
}

static nir_def *
load_buffer(nir_builder *b, const ac_cs_clear_copy_buffer_key *const key,
            const unsigned num_components, const unsigned bit_size, nir_def *buf,
            nir_def *offset, const enum gl_access_qualifier access, const unsigned align_mul,
            const unsigned align_offset)
{
   if (key->addr_user_data) {
      return nir_load_global_amd(b, num_components, bit_size, buf, offset,
                                 .access = access);
   }

   return nir_load_ssbo(b, num_components, bit_size, buf, offset,
                        .access = access,
                        .align_mul = align_mul,
                        .align_offset = align_offset);
}

/* Same as load_buffer with special handling for sparse buffers. Normally, sparse buffer
 * loads return 0 for all components if a sparse load starts on a non-resident page, crosses
 * the page boundary, and ends on a resident page. For copy_buffer, we want it to return 0 only
 * for the portion of the load that's non-resident, and load values for the portion that's
 * resident. The workaround is to scalarize such loads and disallow vectorization.
 */
static nir_def *
load_buffer_sparse(nir_builder *b, const ac_cs_clear_copy_buffer_key *const key,
                   const unsigned num_components, const unsigned bit_size, nir_def *buf,
                   nir_def *offset, const enum gl_access_qualifier access, const unsigned align_mul,
                   const unsigned align_offset, const bool sparse)
{
   if (sparse && num_components > 1) {
      nir_def *vec[NIR_MAX_VEC_COMPONENTS];

      /* Split the vector load into scalar loads. */
      for (unsigned i = 0; i < num_components; i++) {
         const unsigned elem_offset = i * bit_size / 8;
         const unsigned split_align_offset = (align_offset + elem_offset) % align_mul;
         const enum gl_access_qualifier split_access = access | ACCESS_KEEP_SCALAR;
         nir_def *const split_offset = nir_iadd_imm(b, offset, elem_offset);

         vec[i] = load_buffer(b, key, 1, bit_size, buf, split_offset, split_access, align_mul, split_align_offset);
      }

      return nir_vec(b, vec, num_components);
   }

   return load_buffer(b, key, num_components, bit_size, buf, offset, access, align_mul, align_offset);
}

static nir_def *
load_buffer_sparse_split(nir_builder *b, const ac_cs_clear_copy_buffer_key *const key,
                         const unsigned num_vec_components, const unsigned num_removed_components,
                         const unsigned num_load_components, const unsigned bit_size, nir_def *buf,
                         nir_def *offset, const enum gl_access_qualifier access, const unsigned align_mul,
                         const unsigned align_offset, const bool sparse)
{
   nir_def *values[16] = {0};
   unsigned num_values = 0;

   assert(nir_num_components_valid(num_vec_components));
   assert(num_removed_components < num_vec_components);
   assert(num_load_components <= num_vec_components);

   const unsigned num_pad_components =
      (num_removed_components + num_load_components) > num_vec_components ? 0 :
      (num_vec_components - num_removed_components - num_load_components);
   assert(num_pad_components < num_vec_components);

   /* Fill the "removed" components at the beginning with undef. */
   if (num_removed_components) {
      for (unsigned i = 0; i < num_removed_components; ++i)
         values[num_values++] = nir_undef(b, 1, bit_size);
   }

   /* Split the load into multiple instructions because NIR doesn't support arbitrary vector sizes. */
   for (unsigned remaining = num_load_components; remaining; ) {
      const unsigned comps = MIN2(16, util_is_power_of_two_nonzero(remaining) ? remaining : (util_next_power_of_two(remaining) / 2));
      const unsigned byte_offset = (num_load_components - remaining) * bit_size / 8;
      assert(nir_num_components_valid(comps));
      remaining -= comps;
      values[num_values++] = load_buffer_sparse(b, key, comps, bit_size, buf,
                                                nir_iadd_imm(b, offset, byte_offset),
                                                ACCESS_RESTRICT, 4, (align_offset + byte_offset) % 4,
                                                key->src_scalarize_for_sparse);
   }

   /* Fill the "padding" components at the end with undef. */
   if (num_pad_components) {
      for (unsigned i = 0; i < num_pad_components; ++i)
         values[num_values++] = nir_undef(b, 1, bit_size);
   }

   assert(num_values <= ARRAY_SIZE(values));

   return nir_extract_bits(b, values, num_values, 0, num_vec_components, bit_size);
}

static nir_def *
load_buffer_addr_from_user_data(nir_builder *b,
                nir_def *user_data, const unsigned user_data_index)
{
   nir_def *addr_lo = nir_channel(b, user_data, user_data_index);
   nir_def *addr_hi = nir_channel(b, user_data, user_data_index + 1);
   return nir_pack_64_2x32_split(b, addr_lo, addr_hi);
}

/* Create a compute shader implementing clear_buffer or copy_buffer. */
nir_shader *
ac_create_clear_copy_buffer_cs(const ac_cs_clear_copy_buffer_options *const options,
                               const ac_cs_clear_copy_buffer_key *const key)
{
   if (options->print_key) {
      fprintf(stderr, "Internal shader: dma\n");
      fprintf(stderr, "   key.is_clear = %u\n", key->is_clear);
      fprintf(stderr, "   key.addr_user_data = %u\n", key->addr_user_data);
      fprintf(stderr, "   key.dwords_per_thread = %u\n", key->dwords_per_thread);
      fprintf(stderr, "   key.clear_value_size_is_12 = %u\n", key->clear_value_size_is_12);
      fprintf(stderr, "   key.clear_value_size_is_4 = %u\n", key->clear_value_size_is_4);
      fprintf(stderr, "   key.src_scalarize_for_sparse = %u\n", key->src_scalarize_for_sparse);
      fprintf(stderr, "   key.src_align_offset = %u\n", key->src_align_offset);
      fprintf(stderr, "   key.dst_align_offset = %u\n", key->dst_align_offset);
      fprintf(stderr, "   key.dst_last_thread_bytes = %u\n", key->dst_last_thread_bytes);
      fprintf(stderr, "   key.dst_single_thread_unaligned = %u\n", key->dst_single_thread_unaligned);
      fprintf(stderr, "\n");
   }

   assert(key->dwords_per_thread && key->dwords_per_thread <= 4);

   nir_builder b =
      nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE,
         options->nir_options,
         "%s%s%s_buffer_cs_dw%u_srcao_%u_dstao_%u_dstltb_%u_dststua_%u_hst_%u",
         key->is_clear ? "clear" : "copy",
         key->clear_value_size_is_12 ? "12" : key->clear_value_size_is_4 ? "4" : "",
         key->src_scalarize_for_sparse ? "_sparse" : "",
         key->dwords_per_thread,
         key->src_align_offset,
         key->dst_align_offset,
         key->dst_last_thread_bytes,
         key->dst_single_thread_unaligned,
         key->has_start_thread);

   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   if (!key->addr_user_data)
      b.shader->info.num_ssbos = key->is_clear ? 1 : 2;
   b.shader->info.cs.user_data_components_amd = 0;

   unsigned src_va_user_data_index = b.shader->info.cs.user_data_components_amd;
   if (key->addr_user_data && !key->is_clear)
      b.shader->info.cs.user_data_components_amd += 2;

   unsigned dst_va_user_data_index = b.shader->info.cs.user_data_components_amd;
   if (key->addr_user_data)
      b.shader->info.cs.user_data_components_amd += 2;

   unsigned clear_value_user_data_index = b.shader->info.cs.user_data_components_amd;
   if (key->is_clear) {
      b.shader->info.cs.user_data_components_amd +=
         key->clear_value_size_is_12 ? 3 :
         key->clear_value_size_is_4 ? 1 :
         key->dwords_per_thread;
   }

   /* Add the last thread ID value. */
   unsigned last_thread_user_data_index = b.shader->info.cs.user_data_components_amd;
   if (key->dst_last_thread_bytes)
      b.shader->info.cs.user_data_components_amd++;

   unsigned start_thread_user_data_index = b.shader->info.cs.user_data_components_amd;
   if (key->has_start_thread)
      b.shader->info.cs.user_data_components_amd++;

   nir_def *thread_id = ac_get_global_ids(&b, 1, 32);
   nir_def *user_data = nir_load_user_data_amd(&b);

   /* If the clear/copy area is unaligned, we launched extra threads at the beginning to make it
    * aligned. Skip those threads here.
    */
   nir_if *if_positive = NULL;
   if (key->has_start_thread) {
      nir_def *start_thread =
         nir_channel(&b, user_data, start_thread_user_data_index);
      thread_id = nir_isub(&b, thread_id, start_thread);
      if_positive = nir_push_if(&b, nir_ige_imm(&b, thread_id, 0));
      if_positive->control = nir_selection_control_divergent_always_taken;
   }

   /* Convert the global thread ID into bytes. */
   nir_def *offset = nir_imul_imm(&b, thread_id, 4 * key->dwords_per_thread);
   nir_def *value;

   if (key->is_clear) {
      const unsigned userdata_dwords = key->clear_value_size_is_4 ? 1 : key->dwords_per_thread;
      value = nir_extract_bits(&b, &user_data, 1, clear_value_user_data_index * 32, userdata_dwords, 32);

      /* We store 4 dwords per thread, but the clear value has 3 dwords. Swizzle it to 4 dwords.
       * Storing 4 dwords per thread is faster even when the ALU cost is worse.
       */
      if (key->clear_value_size_is_12 && key->dwords_per_thread == 4) {
         nir_def *dw_offset = nir_imul_imm(&b, thread_id, key->dwords_per_thread);
         nir_def *vec[3];

         /* Swizzle a 3-component clear value to get a 4-component clear value. Example:
          * 0 1 2 3 | 4 5 6 7 | 8 9 10 11  // dw_offset
          *              |
          *              V
          * 0 1 2 0 | 1 2 0 1 | 2 0 1 2    // clear value component indices
          */
         for (unsigned i = 0; i < 3; i++) {
            vec[i] = nir_vector_extract(&b, value,
                                        nir_umod_imm(&b, nir_iadd_imm(&b, dw_offset, i), 3));
         }
         value = nir_vec4(&b, vec[0], vec[1], vec[2], vec[0]);
      } else if (key->clear_value_size_is_4 && key->dwords_per_thread > 1) {
         nir_def *arr[] = {value, value, value, value};
         value = nir_vec(&b, arr, key->dwords_per_thread);
      }
   } else {
      /* The hw doesn't support unaligned 32-bit loads, and only supports single-component
       * unaligned 1-byte and 2-byte loads. Luckily, we don't have to use single-component loads
       * because ac_nir_lower_subdword_load converts 1-byte and 2-byte vector loads with unaligned
       * offsets into aligned 32-bit loads by loading an extra dword and then bit-shifting all bits
       * to get the expected result. We only have to set bit_size to 8 or 16 and align_offset to
       * 1..3 to indicate that this is an unaligned load. align_offset is the amount of
       * unalignment.
       *
       * Since the buffer binding offsets are rounded down to the clear/copy size of the thread
       * (i.e. dst_align_offset is subtracted from dst_offset, and src_align_offset is subtracted
       * from src_offset), the stores expect the loaded value to be byte-shifted accordingly.
       * realign_offset is the amount of byte-shifting we have to do.
       */
      assert(util_is_power_of_two_nonzero(key->dwords_per_thread));
      int realign_offset = key->src_align_offset - key->dst_align_offset;
      unsigned alignment = (unsigned)realign_offset % 4 == 0 ? 4 :
                           (unsigned)realign_offset % 2 == 0 ? 2 : 1;
      unsigned bit_size = alignment * 8;
      unsigned num_comps = key->dwords_per_thread * 4 / alignment;
      nir_if *if_first_thread = NULL;
      nir_def *value0 = NULL;
      nir_def *src_buf;

      if (key->addr_user_data)
         src_buf = load_buffer_addr_from_user_data(&b, user_data, src_va_user_data_index);
      else
         src_buf = nir_imm_int(&b, 0);

      if (realign_offset < 0) {
         /* if src_align_offset is less than dst_align_offset, realign_offset is
          * negative, which causes the first thread to use a negative buffer offset, which goes
          * entirely out of bounds because the offset is treated as unsigned. Instead of that,
          * the first thread should load from offset 0 by not loading the bytes before
          * the beginning of the buffer.
          */
         if_first_thread = nir_push_if(&b, nir_ieq_imm(&b, thread_id, 0));
         {
            unsigned num_removed_comps = -realign_offset / alignment;
            unsigned num_inbounds_comps = num_comps - num_removed_comps;

            /* Only 8 and 16 component vectors are valid after 5 in NIR. */
            while (!nir_num_components_valid(num_inbounds_comps))
               num_inbounds_comps = util_next_power_of_two(num_inbounds_comps);

            value0 = load_buffer_sparse_split(&b, key, num_comps, num_removed_comps, num_inbounds_comps,
                                              bit_size, src_buf, offset, ACCESS_RESTRICT, 4, 0,
                                              key->src_scalarize_for_sparse);
         }
         nir_push_else(&b, if_first_thread);
      }

      value = load_buffer_sparse(&b, key, num_comps, bit_size, src_buf, nir_iadd_imm(&b, offset, realign_offset),
                                 ACCESS_RESTRICT, 4, (unsigned)realign_offset % 4, key->src_scalarize_for_sparse);

      if (if_first_thread) {
         nir_pop_if(&b, if_first_thread);
         value = nir_if_phi(&b, value0, value);
      }

      /* Bitcast the vector to 32 bits. */
      if (value->bit_size != 32)
         value = nir_extract_bits(&b, &value, 1, 0, key->dwords_per_thread, 32);
   }

   nir_def *dst_buf;

   if (key->addr_user_data)
      dst_buf = load_buffer_addr_from_user_data(&b, user_data, dst_va_user_data_index);
   else
      dst_buf = nir_imm_int(&b, !key->is_clear);

   nir_if *if_first_thread = NULL, *if_last_thread = NULL;

   if (!key->dst_single_thread_unaligned) {
      /* dst_align_offset means how many bytes the first thread should skip because the offset of
       * the buffer binding is rounded down to the clear/copy size of thread, causing the bytes
       * before dst_align_offset to be writable. Above we used realign_offset to byte-shift
       * the value to compensate for the rounded-down offset, so that all stores are dword stores
       * regardless of the offset/size alignment except that the first thread shouldn't store
       * the first dst_align_offset bytes, and the last thread should only store the first
       * dst_last_thread_bytes. In both cases, there is a dword that must be only partially
       * written by splitting it into 8-bit and 16-bit stores.
       */
      if (key->dst_align_offset) {
         if_first_thread = nir_push_if(&b, nir_ieq_imm(&b, thread_id, 0));
         {
            unsigned local_offset = key->dst_align_offset;
            nir_def *first_dword = nir_channel(&b, value, local_offset / 4);

            if (local_offset % 2 == 1) {
               store_buffer(&b, key, nir_channel(&b, nir_unpack_32_4x8(&b, first_dword), local_offset % 4),
                            dst_buf, nir_iadd_imm_nuw(&b, offset, local_offset),
                            ACCESS_RESTRICT);
               local_offset++;
            }

            if (local_offset % 4 == 2) {
               store_buffer(&b, key, nir_unpack_32_2x16_split_y(&b, first_dword), dst_buf,
                            nir_iadd_imm_nuw(&b, offset, local_offset),
                            ACCESS_RESTRICT);
               local_offset += 2;
            }

            assert(local_offset % 4 == 0);
            unsigned num_dw_remaining = key->dwords_per_thread - local_offset / 4;

            if (num_dw_remaining) {
               nir_def *dwords =
                  nir_channels(&b, value, BITFIELD_RANGE(local_offset / 4, num_dw_remaining));

               store_buffer(&b, key, dwords, dst_buf, nir_iadd_imm_nuw(&b, offset, local_offset),
                            ACCESS_RESTRICT);
            }
         }
         nir_push_else(&b, if_first_thread);
      }

      if (key->dst_last_thread_bytes) {
         nir_def *last_thread_id =
            nir_channel(&b, user_data, last_thread_user_data_index);

         if_last_thread = nir_push_if(&b, nir_ieq(&b, thread_id, last_thread_id));
         {
            unsigned num_dwords = key->dst_last_thread_bytes / 4;
            bool write_short = (key->dst_last_thread_bytes - num_dwords * 4) / 2;
            bool write_byte = key->dst_last_thread_bytes % 2;
            nir_def *last_dword = nir_channel(&b, value, num_dwords);

            if (num_dwords) {
               nir_def *dwords = nir_channels(&b, value, nir_component_mask(num_dwords));
               store_buffer(&b, key, dwords, dst_buf, offset, ACCESS_RESTRICT);
            }

            if (write_short) {
               store_buffer(&b, key, nir_u2u16(&b, last_dword), dst_buf,
                            nir_iadd_imm_nuw(&b, offset, num_dwords * 4),
                            ACCESS_RESTRICT);
            }

            if (write_byte) {
               store_buffer(&b, key, nir_channel(&b, nir_unpack_32_4x8(&b, last_dword), write_short * 2),
                            dst_buf, nir_iadd_imm_nuw(&b, offset, num_dwords * 4 + write_short * 2),
                            ACCESS_RESTRICT);
            }
         }
         nir_push_else(&b, if_last_thread);
      }

      store_buffer(&b, key, value, dst_buf, offset, ACCESS_RESTRICT);

      if (if_last_thread)
         nir_pop_if(&b, if_last_thread);
      if (if_first_thread)
         nir_pop_if(&b, if_first_thread);
   } else {
      /* This shader only executes a single thread (tiny copy or clear) and it's unaligned at both
       * the beginning and the end. Walk the individual dwords/words/bytes that should be written
       * to split the store accordingly.
       */
      for (unsigned local_offset = key->dst_align_offset;
           local_offset < key->dst_last_thread_bytes;) {
         unsigned remaining = key->dst_last_thread_bytes - local_offset;
         nir_def *src_dword = nir_channel(&b, value, local_offset / 4);

         if (local_offset % 2 == 1 || remaining == 1) {
            /* 1-byte store. */
            nir_def *src_dword4x8 = nir_unpack_32_4x8(&b, src_dword);
            store_buffer(&b, key, nir_channel(&b, src_dword4x8, local_offset % 4), dst_buf,
                         nir_iadd_imm_nuw(&b, offset, local_offset), ACCESS_RESTRICT);
            local_offset++;
         } else if (local_offset % 4 == 2 || remaining == 2 || remaining == 3) {
            /* 2-byte store. */
            nir_def *src_dword2x16 = nir_unpack_32_2x16(&b, src_dword);
            store_buffer(&b, key, nir_channel(&b, src_dword2x16, (local_offset / 2) % 2), dst_buf,
                         nir_iadd_imm_nuw(&b, offset, local_offset), ACCESS_RESTRICT);
            local_offset += 2;
         } else {
            /* 1-N dwords. */
            unsigned dw_size = remaining / 4;
            assert(dw_size);
            assert(local_offset % 4 == 0);

            store_buffer(&b, key, nir_channels(&b, value, BITFIELD_RANGE(local_offset / 4, dw_size)),
                         dst_buf, nir_iadd_imm_nuw(&b, offset, local_offset),
                         ACCESS_RESTRICT);
            local_offset += dw_size * 4;
         }
      }
   }

   if (key->has_start_thread)
      nir_pop_if(&b, if_positive);

   return b.shader;
}

static bool
ac_clear_copy_can_use_cp_dma(const ac_cs_clear_copy_buffer_options *const options,
                             const ac_cs_clear_copy_buffer_info *const info)
{
   bool can_use_cp_dma = options->info->has_cp_dma;

   /* CP DMA doesn't support sparse on GFX6-9, so we must use compute for that. */
   if (!options->info->cp_dma_supports_sparse)
      can_use_cp_dma &= !info->src_is_sparse && !info->dst_is_sparse;

   /* CP DMA only supports dword-aligned clears and small clear values. */
   if (info->clear_value_size)
      can_use_cp_dma &= info->clear_value_size <= 4 && info->dst_offset % 4 == 0 && info->size % 4 == 0;

   /* CP DMA doesn't support the render condition (conditional rendering in Vulkan) */
   if (info->render_condition_enabled)
      can_use_cp_dma = false;

   return can_use_cp_dma;
}

static bool
ac_clear_copy_should_use_compute(const ac_cs_clear_copy_buffer_options *options,
                                 const ac_cs_clear_copy_buffer_info *info)
{
   const bool is_copy = info->clear_value_size == 0;
   const bool can_use_cp_dma = ac_clear_copy_can_use_cp_dma(options, info);

   if (!can_use_cp_dma)
      return true;

   if (!options->fail_if_slow)
      return true;

   /* CP DMA doesn't execute asynchronously on compute queues, meaning that every CP DMA packet
    * implicitly waits for CP DMA to finish, which slows down command buffer execution.
    */
   if (options->is_compute_queue)
      return true;

   switch (options->info->gfx_level) {
   /* GFX6-8: CP DMA clears are so slow that we risk getting a GPU timeout.
    * CP DMA copies are also slow but less.
    */
   case GFX6:
      /* Optimal for Tahiti. */
      if (is_copy) {
         if (!info->dst_is_vram || !info->src_is_vram ||
               info->size <= (info->dst_offset % 4 ||
                              (info->dst_offset == 4 && info->src_offset % 4) ? 32 * 1024 : 16 * 1024))
            return false;
      } else {
         if (info->dst_is_vram && info->size <= 1024)
            return false;
      }
      break;

   case GFX7:
      /* Optimal for Hawaii. */
      if (is_copy && info->dst_is_vram && info->src_is_vram && info->size <= 512)
         return false;
      break;

   case GFX8:
      /* Optimal for Tonga. */
      break;

   case GFX9:
      /* Optimal for Vega10. */
      if (is_copy) {
         if (info->src_is_vram) {
            if (info->dst_is_vram) {
               if (info->size < 4096)
                  return false;
            } else {
               if (info->size < (info->dst_offset % 64 ? 8192 : 2048))
                  return false;
            }
         } else {
            /* GTT->VRAM and GTT->GTT. */
            return false;
         }
      } else {
         if (!info->dst_is_vram && (info->size < 2048 || info->size >= 8 << 20 /* 8 MB */))
            return false;
      }
      break;

   case GFX10:
   case GFX10_3:
      /* Optimal for Navi21, Navi10. */
      break;

   case GFX11:
   default:
      /* Optimal for Navi31. */
      if (is_copy && info->size < 1024 && info->dst_offset % 256 && info->dst_is_vram && info->src_is_vram)
         return false;
      break;

   case GFX12:
      /* Optimal for Navi 48. */
      break;
   }

   return true;
}

static unsigned
ac_clear_copy_calc_dwords_per_thread(const ac_cs_clear_copy_buffer_options *options,
                                     const ac_cs_clear_copy_buffer_info *info,
                                     const int clear_value_size)
{
   const bool is_copy = clear_value_size == 0;

   /* Determine optimal dwords_per_thread for performance.
    * This is a good initial value to start with.
    */
   unsigned dwords_per_thread = info->size <= 64 * 1024 ? 2 : 4;

   /* Clearing 4 dwords per thread with a 3-dword clear value is faster with big sizes. */
   if (!is_copy && clear_value_size == 12)
      dwords_per_thread = info->size <= 4096 ? 3 : 4;

   switch (options->info->gfx_level) {
   case GFX6:
      /* Optimal for Tahiti. */
      if (is_copy) {
         if (info->dst_is_vram && info->src_is_vram)
            dwords_per_thread = 2;
      } else {
         if (info->dst_is_vram && clear_value_size != 12)
            dwords_per_thread = info->size <= 128 * 1024 || info->size >= 4 << 20 /* 4MB */ ? 2 : 4;

         if (clear_value_size == 12)
            dwords_per_thread = info->size <= (info->dst_is_vram ? 256 : 128) * 1024 ? 3 : 4;
      }
      break;

   case GFX7:
      /* Optimal for Hawaii. */
      if (is_copy) {
         if (info->dst_is_vram && info->src_is_vram && info->dst_offset % 4 == 0 &&
               info->size >= 8 << 20 /* 8MB */)
            dwords_per_thread = 2;
      } else {
         if (info->dst_is_vram && clear_value_size != 12)
            dwords_per_thread = info->size <= 32 * 1024 ? 2 : 4;

         if (clear_value_size == 12)
            dwords_per_thread = info->size <= 256 * 1024 ? 3 : 4;
      }
      break;

   case GFX8:
      /* Optimal for Tonga. */
      if (is_copy) {
         dwords_per_thread = 2;
      } else {
         if (clear_value_size == 12 && info->size < (2 << 20) /* 2MB */)
            dwords_per_thread = 3;
      }
      break;

   case GFX9:
      /* Optimal for Vega10. */
      if (is_copy && info->src_is_vram && info->dst_is_vram && info->size >= 8 << 20 /* 8 MB */)
         dwords_per_thread = 2;

      if (!info->dst_is_vram)
         dwords_per_thread = 2;
      break;

   case GFX10:
   case GFX10_3:
   case GFX11:
      /* Optimal for Navi31, Navi21, Navi10. */
      break;

   case GFX11_5:
   case GFX11_7:
      /* Optimal for Strix Halo. */
      dwords_per_thread = info->size <= 2 * 1024 ? 2 : 4;
      break;

   default:
   case GFX12:
      /* Optimal for Navi48. */
      if (!is_copy && clear_value_size == 12 && info->size <= 512 * 1024)
         dwords_per_thread = 3;
      break;
   }

   /* dwords_per_thread must be at least the size of the clear value. */
   if (!is_copy)
      dwords_per_thread = MAX2(dwords_per_thread, clear_value_size / 4);

   /* If dst is sparse, stores mustn't straddle a page boundary, which means the store size must
    * be 2^n. It can only be a non-power-of-two and 3 with GL buffer clears because VK doesn't
    * have 12-byte clear values.
    */
   if (info->dst_is_sparse && dwords_per_thread == 3)
      dwords_per_thread = 4;

   /* Override dwords per thread before validating the value. */
   if (info->dwords_per_thread)
      dwords_per_thread = info->dwords_per_thread;

   /* Validate dwords_per_thread. */
   if (info->dst_is_sparse && !util_is_power_of_two_nonzero(dwords_per_thread)) {
      fprintf(stderr, "ac_nir_meta_cs_clear_copy_buffer: dwords_per_thread must be a power of two "
                      "for a sparse destination\n");
      exit(1);
   }

   if (dwords_per_thread > 4) {
      fprintf(stderr, "ac_nir_meta_cs_clear_copy_buffer: dwords_per_thread must be <= 4\n");
      exit(1);
   }

   if (clear_value_size > dwords_per_thread * 4) {
      fprintf(stderr, "ac_nir_meta_cs_clear_copy_buffer: clear_value_size must be <= "
                      "dwords_per_thread\n");
      exit(1);
   }

   if (clear_value_size == 12 && info->dst_offset % 4) {
      fprintf(stderr, "ac_nir_meta_cs_clear_copy_buffer: if clear_value_size == 12, dst_offset "
                      "must be aligned to 4\n");
      exit(1);
   }

   return dwords_per_thread;
}

static unsigned
ac_prepare_clear_value_user_data(const int clear_value_size,
                                 const unsigned dwords_per_thread,
                                 const uint64_t dst_align_offset,
                                 const uint32_t *const clear_value,
                                 uint32_t *user_data_clear_value)
{
   if (clear_value_size == 4) {
      user_data_clear_value[0] = *clear_value;
      return 1;
   }

   assert(clear_value_size >= 4 && clear_value_size <= 16 &&
          (clear_value_size == 12 || util_is_power_of_two_or_zero(clear_value_size)));

   /* Since the clear value may start on an unaligned offset and we just pass user SGPRs
    * to dword stores as-is, we need to byte-shift the clear value to that offset and
    * replicate it because 1 invocation stores up to 4 dwords from user SGPRs regardless of
    * the clear value size.
    */
   const unsigned num_clear_user_data_terms = clear_value_size == 12 ? 3 : dwords_per_thread;
   const unsigned clear_user_data_size = num_clear_user_data_terms * 4;

   memcpy((uint8_t *)user_data_clear_value,
            (uint8_t*)clear_value + clear_value_size - dst_align_offset % clear_value_size,
            dst_align_offset % clear_value_size);
   unsigned offset = dst_align_offset % clear_value_size;

   while (offset + clear_value_size <= clear_user_data_size) {
      memcpy((uint8_t*)user_data_clear_value + offset, clear_value, clear_value_size);
      offset += clear_value_size;
   }

   if (offset < clear_user_data_size)
      memcpy((uint8_t*)user_data_clear_value + offset, clear_value, clear_user_data_size - offset);

   return num_clear_user_data_terms;
}

/* This returns false if CP DMA should be used. */
bool
ac_prepare_cs_clear_copy_buffer(const ac_cs_clear_copy_buffer_options *options,
                                const ac_cs_clear_copy_buffer_info *info,
                                ac_cs_clear_copy_buffer_dispatch *out)
{
   bool is_copy = info->clear_value_size == 0;

   memset(out, 0, sizeof(*out));
   out->cpdma_supported = ac_clear_copy_can_use_cp_dma(options, info);

   /* Expand 1-byte and 2-byte clear values to a dword. */
   int clear_value_size = info->clear_value_size;
   const uint32_t *clear_value = info->clear_value;
   uint32_t tmp_clear_value;

   if (!is_copy) {
      if (util_lower_clearsize_to_dword(clear_value, &clear_value_size, &tmp_clear_value))
         clear_value = &tmp_clear_value;

      assert(clear_value_size % 4 == 0);
   }

   if (!ac_clear_copy_should_use_compute(options, info))
      return false;

   const unsigned dwords_per_thread = ac_clear_copy_calc_dwords_per_thread(options, info, clear_value_size);
   if (!dwords_per_thread)
      return false;

   uint64_t dst_align_offset = info->dst_offset % (dwords_per_thread * 4);
   uint64_t dst_offset_bound = info->dst_offset - dst_align_offset;
   uint64_t src_align_offset = is_copy ? info->src_offset % 4 : 0;
   unsigned num_user_data_terms = 0;
   unsigned addr_user_data_offset = 0;

   if (options->addr_user_data)
      num_user_data_terms += is_copy ? 4 : 2;

   /* Set the clear value in user data SGPRs. */
   if (!is_copy) {
      num_user_data_terms +=
         ac_prepare_clear_value_user_data(clear_value_size, dwords_per_thread, dst_align_offset,
                                          clear_value, &out->user_data[num_user_data_terms]);
   }

   out->shader_key.key = 0;

   out->shader_key.addr_user_data = options->addr_user_data;
   out->shader_key.is_clear = !is_copy;
   assert(dwords_per_thread && dwords_per_thread <= 4);
   out->shader_key.dwords_per_thread = dwords_per_thread;
   out->shader_key.clear_value_size_is_12 = !is_copy && clear_value_size == 12;
   out->shader_key.clear_value_size_is_4 = !is_copy && clear_value_size == 4;
   /* If the src load size is aligned to 2^n and the src load address in every invocation is aligned
    * to the load size, loads are guaranteed to never be partially non-resident, so we don't have to
    * scalarize them. Every sparse buffer is aligned to a page, so we don't need to check whether the
    * buffer base address is aligned.
    */
   out->shader_key.src_scalarize_for_sparse =
      is_copy && info->src_is_sparse &&
      /* Each invocation must increment the offset by 2^n. */
      (!util_is_power_of_two_nonzero(dwords_per_thread) ||
       /* The buffer range bounds must be divisible by the copy size per invocation. */
       ((info->src_offset - dst_align_offset) % (dwords_per_thread * 4) != 0 ||
        (info->src_offset + info->size) % (dwords_per_thread * 4) != 0));
   out->shader_key.src_align_offset = src_align_offset;
   out->shader_key.dst_align_offset = dst_align_offset;

   if ((dst_align_offset + info->size) % (dwords_per_thread * 4))
      out->shader_key.dst_last_thread_bytes = (dst_align_offset + info->size) % (dwords_per_thread * 4);

   unsigned num_threads = DIV_ROUND_UP(dst_align_offset + info->size, dwords_per_thread * 4);
   out->shader_key.dst_single_thread_unaligned = num_threads == 1 && dst_align_offset &&
                                                 out->shader_key.dst_last_thread_bytes;

   /* start_thread offsets threads to make sure all non-zero waves start clearing/copying from
    * the beginning a 256B block and clear/copy whole 256B blocks. Clearing/copying a 256B block
    * partially for each wave is inefficient, which happens when dst_offset isn't aligned to 256.
    * Clearing/copying whole 256B blocks per wave isn't possible if dwords_per_thread isn't 2^n.
    */
   unsigned start_thread =
      dst_offset_bound % AMD_MEMCHANNEL_INTERLEAVE_BYTES &&
      util_is_power_of_two_nonzero(dwords_per_thread) ?
            DIV_ROUND_UP(AMD_MEMCHANNEL_INTERLEAVE_BYTES -
                         dst_offset_bound % AMD_MEMCHANNEL_INTERLEAVE_BYTES,
                         dwords_per_thread * 4) : 0;
   out->shader_key.has_start_thread = start_thread != 0;

   /* Set the value of the last thread ID, so that the shader knows which thread is the last one. */
   if (out->shader_key.dst_last_thread_bytes)
      out->user_data[num_user_data_terms++] = num_threads - 1;
   if (out->shader_key.has_start_thread)
      out->user_data[num_user_data_terms++] = start_thread;

   /* We need to bind whole dwords because of how we compute voffset. The bytes that shouldn't
    * be written are not written by the shader.
    */
   const uint64_t dst_buf_offset = dst_offset_bound;
   const uint64_t dst_buf_size = align(dst_align_offset + info->size, 4);

   if (!options->addr_user_data) {
      out->ssbo[is_copy].offset = dst_buf_offset;
      out->ssbo[is_copy].size = dst_buf_size;
   } else {
      out->user_data[addr_user_data_offset + 0 + !!is_copy * 2] = dst_buf_offset;
      out->user_data[addr_user_data_offset + 1 + !!is_copy * 2] = dst_buf_offset >> 32ull;
   }

   if (is_copy) {
      /* Since unaligned copies use 32-bit loads, any dword that's partially covered by the copy
       * range must be fully covered, so that the 32-bit loads succeed.
       */
      const uint64_t src_buf_offset = info->src_offset - src_align_offset;
      const uint64_t src_buf_size = align(src_align_offset + info->size, 4);
      assert(src_buf_offset % 4 == 0 && src_buf_size % 4 == 0);

      if (!options->addr_user_data) {
         out->ssbo[0].offset = src_buf_offset;
         out->ssbo[0].size = src_buf_size;
      } else {
         out->user_data[addr_user_data_offset + 0] = src_buf_offset;
         out->user_data[addr_user_data_offset + 1] = src_buf_offset >> 32ull;
      }
   }

   out->num_user_data = num_user_data_terms;
   out->workgroup_size = 64;
   out->num_threads = start_thread + num_threads;
   if (!options->addr_user_data)
      out->num_ssbos = is_copy ? 2 : 1;

   /* Determine optimal COMPUTE_DISPATCH_INTERLEAVE.INTERLEAVE/INTERLEAVE_1D.
    * Verified on Navi48.
    */
   if (is_copy)
      out->dispatch_interleave = info->size <= 2048 ? 64 : 256;
   else
      out->dispatch_interleave = info->size <= 256 * 1024 ? 64 : 256;

   return true;
}
