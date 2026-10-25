/* Copyright 2024 Advanced Micro Devices, Inc.
 * Copyright 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

/* This is a universal image clear/copy/blit/resolve compute shader generator optimized for AMD GPUs.
 * ac_prepare_compute_blit generates the shader key, computes dispatch parameters, and encodes user
 * data SGPRs. The caller should then call ac_create_blit_cs to generate the NIR shader from the key.
 *
 * This shader and dispatch logic performs significantly better across many bpp and tiling mode
 * combinations than trivial image copy shaders that execute only 1 image load and store per
 * invocation.
 *
 * The shader is based on the experimental observations that:
 * 1. Loading and storing a small number of bytes per invocation is slow.
 * 2. When a wave (more specifically a VMEM clause of image stores in a wave) only partially covers
 *    a microtile or partially covers a 256B block (memory channel interleave), performance suffers.
 *    It has also been observed that image stores with DCC decrease performance even further
 *    on older generations if a wave doesn't cover a whole 256B block.
 *
 * To eliminate those limitations:
 * - All loads and stores must be in a VMEM clause.
 * - The workgroup size is always 64 running as wave64.
 * - Waves usually store one or multiple 256B blocks. Individual lanes can store more than 1 pixel
 *   in certain configurations.
 * - Each 256B block intersecting the destination area is usually owned by only 1 wave. (there are
 *   a few rare exceptions to this where that's not possible for some tiling modes, and the closest
 *   larger or smaller power-of-two tile size is used instead)
 * - If the destination area covers some 256B blocks only partially, the intent is for each wave
 *   to own whole 256B blocks regardless, even though the invocations outside the destination area
 *   then do nothing, therefore, the top-left coordinates of the dispatch are shifted to align with
 *   the 256B block's top-left corner. That greatly benefits overall throughput since the 256B
 *   blocks that don't intersect the unaligned boundary of the destination area stay on the fast
 *   path throughout the memory system.
 * - The dispatch logic first selects the number of bytes each lane should store ("bytes_per_lane").
 *   If that's more than 1 pixel per lane, the dimensions of the tiny tile ("lane_tile") are
 *   determined, which can be 2x1, 1x2, or even 1x2x2, which is the footprint of one lane.
 *   The next level of pixel grouping is "tile_size", which is one 256B block or the closest block
 *   size that exists, and the block can be further reshaped for linear<->tiled copies. If one
 *   block doesn't cover enough invocations to fill wave64, the wave is filled with multiple
 *   spatially-adjacent 256B blocks. Such waves are then dispatched.
 * - To reduce complexity, all invocations always store their whole lane tile or do nothing.
 *   If the destination area is unaligned such that an invocation would have to only partially
 *   store its lane tile, the compute dispatch is split into multiple dispatches: First,
 *   the greatest aligned area is carved out from the destination area and processed by a compute
 *   dispatch using the ideal lane tile size, and the leftover regions are processed by separate
 *   dispatches using less optimal lane tile sizes. If the destination area is so small such that
 *   executing multiple dispatches would be limited by CP overhead instead of the shader, a less
 *   optimal lane tile size is selected for the whole destination area and only 1 compute dispatch
 *   is issued.
 * - Tiled->linear copies don't use contiguous 256B tiles of the destination, but use linear-ish
 *   tiles that are a compromise between the source and destination tiling, e.g. (64B, 4) and
 *   (64B, 2, 2).
 *
 * MSAA copies copy FMASK together with color samples and they copy only the color samples
 * referenced by FMASK.
 *
 * This also supports 96-bit (RGB32) formats as follows:
 * - Only 96-bit -> 96-bit clears and copies are supported.
 * - Only 2D non-layer non-mip, only linear layouts, max 64K x 64K.
 * - Images are viewed as dword arrays, they take the 32bpp path through the dispatch logic and
 *   shader generator, and are accessed using global dword load/store instructions with manual
 *   address calculations using a row stride.
 * - If adjacent dwords are loaded and stored by the same invocation, nir_opt_load_store_vectorize
 *   is expected to vectorize the instructions.
 * - The maximum supported addressable size in dword units is currently (256K - 1) x 64K dwords =
 *   almost 64 GB, but only 192K x 64K dwords = 48 GB is needed for 64K x 64K images.
 * - The shader clears or copies 4, 8, or 16 bytes per invocation, but not 12 bytes per invocation
 *   since that would use the memory system inefficiently.
 * - If clearing, each invocation shuffles dwords of the clear value to get the corresponding
 *   4-byte, 8-byte, or 16-byte clear value that the invocation needs.
 */

#include "ac_nir_meta.h"
#include "ac_nir_helpers.h"
#include "ac_surface.h"
#include "nir_format_convert.h"
#include "compiler/aco_interface.h"
#include "util/format_srgb.h"
#include "util/u_pack_color.h"

static nir_def *
convert_linear_to_srgb(nir_builder *b, nir_def *input)
{
   /* There are small precision differences compared to CB, so the gfx blit will return slightly
    * different results.
    */
   for (unsigned i = 0; i < MIN2(3, input->num_components); i++) {
      input = nir_vector_insert_imm(b, input,
                                    nir_format_linear_to_srgb(b, nir_channel(b, input, i)), i);
   }

   return input;
}

static nir_def *
apply_blit_output_modifiers(nir_builder *b, nir_def *color,
                            const ac_cs_blit_key *key)
{
   unsigned bit_size = color->bit_size;
   nir_def *zero = nir_imm_intN_t(b, 0, bit_size);

   if (key->sint_to_uint)
      color = nir_imax(b, color, zero);

   if (key->uint_to_sint) {
      color = nir_umin(b, color,
                       nir_imm_intN_t(b, bit_size == 16 ? INT16_MAX : INT32_MAX,
                                      bit_size));
   }

   if (key->dst_is_srgb)
      color = convert_linear_to_srgb(b, color);

   nir_def *one = key->use_integer_one ? nir_imm_intN_t(b, 1, bit_size) :
                                             nir_imm_floatN_t(b, 1, bit_size);

   if (key->is_clear) {
      if (key->last_dst_channel < 3)
         color = nir_trim_vector(b, color, key->last_dst_channel + 1);
   } else {
      assert(key->last_src_channel <= key->last_dst_channel);
      assert(color->num_components == key->last_src_channel + 1);

      /* Set channels not present in src to 0 or 1. */
      if (key->last_src_channel < key->last_dst_channel) {
         color = nir_pad_vector(b, color, key->last_dst_channel + 1);

         for (unsigned chan = key->last_src_channel + 1; chan <= key->last_dst_channel; chan++)
            color = nir_vector_insert_imm(b, color, chan == 3 ? one : zero, chan);
      }

      /* Discard channels not present in dst. The hardware fills unstored channels with 0. */
      if (key->last_dst_channel < key->last_src_channel)
         color = nir_trim_vector(b, color, key->last_dst_channel + 1);

      /* Precision issue workaround for piglit/texsubimage array pbo.
       * Determined by trial and error.
       */
      if (key->dst_is_rgb5)
         color = nir_fadd(b, color, nir_imm_floatN_t(b, -1.0/4096, bit_size));
   }

   /* Discard channels not present in dst. The hardware fills unstored channels with 0. */
   if (key->last_dst_channel < 3)
      color = nir_trim_vector(b, color, key->last_dst_channel + 1);

   return color;
}

static nir_def *
get_global_image2d_addr(nir_builder *b, const ac_cs_blit_key *key, nir_def *global_addr,
                        nir_def *coord2d, nir_def *dword_stride)
{
   if (key->addr_math_64bit) {
      coord2d = nir_u2u64(b, coord2d);
      dword_stride = nir_u2u64(b, dword_stride);
   }

   nir_def *offset =
      nir_iadd(b, nir_imul(b, nir_channel(b, coord2d, 1), dword_stride), nir_channel(b, coord2d, 0));
   offset = nir_ishl_imm(b, offset, 2);

   if (!key->addr_math_64bit)
      offset = nir_u2u64(b, offset);

   return nir_iadd(b, global_addr, offset);
}

static unsigned
get_num_user_data_terms(const ac_cs_blit_key *key)
{
   unsigned num;

   if (key->format_is_96bit) {
      num = 3;

      if (!key->is_clear)
         num += 2;
      else
         num += 3;
   } else {
      num = 2;

      if (key->dst_has_z || key->src_has_z)
         num++;

      if (!key->is_clear) {
         num++;
      } else {
         if (key->d16)
            num += DIV_ROUND_UP(key->last_dst_channel + 1, 2);
         else
            num += key->last_dst_channel + 1;
      }
   }

   return num;
}

static nir_def *
tex(nir_builder *b, const ac_cs_blit_key *key, nir_deref_instr *deref, nir_texop op,
    nir_def *coord, unsigned num_coords)
{
   bool fmask = op == nir_texop_fragment_mask_fetch_amd || op == nir_texop_samples_identical;

   nir_def *result =
      nir_build_tex(b, op,
                    .coord = nir_trim_vector(b, coord, num_coords - key->src_is_msaa),
                    .ms_index = key->src_is_msaa && !fmask ?
                                   nir_channel(b, coord, num_coords - 1) : NULL,
                    .texture_deref = deref,
                    .dim = deref->type->sampler_dimensionality,
                    .is_array = deref->type->sampler_array);
   return fmask ? result : nir_trim_vector(b, result, key->last_src_channel + 1);
}

static nir_def *
load_src(nir_builder *b, const ac_cs_blit_key *key, nir_deref_instr *deref, nir_texop op,
         nir_def *coord, unsigned num_coords)
{
   if (key->src_is_sampler)
      return tex(b, key, deref, op, coord, num_coords);

   if (op == nir_texop_fragment_mask_fetch_amd) {
      return nir_image_deref_fragment_mask_load_amd(b, &deref->def, coord,
                                                    .image_dim = GLSL_SAMPLER_DIM_2D,
                                                    .image_array = deref->type->sampler_array);
   }

   unsigned bit_size = glsl_base_type_get_bit_size(deref->type->sampled_type);
   assert(op != nir_texop_samples_identical);

   return nir_image_deref_load(b, key->last_src_channel + 1, bit_size,
                               &deref->def, coord,
                               nir_channel(b, coord, num_coords - 1),
                               nir_imm_intN_t(b, 0, coord->bit_size),
                               .image_dim = deref->type->sampler_dimensionality,
                               .image_array = deref->type->sampler_array,
                               .dest_type = nir_type_uint | bit_size,
                               .access = op == nir_texop_fragment_fetch_amd ?
                                            ACCESS_FMASK_LOWERED_AMD : 0);
}

static void
print_shader_key_field(FILE *f, const char *name, unsigned value)
{
   /* When logging, we want all fields on the same line. */
   if (f == stderr)
      fprintf(f, "   %s = %u\n", name, value);
   else
      fprintf(f, "%s = %u, ", name, value);
}

DEBUG_GET_ONCE_OPTION(log_keys, "AMD_CS_BLIT_LOG_KEYS_FILE", false)

/* The compute blit shader.
 *
 * Implementation details:
 * - Out-of-bounds dst coordinates are not clamped at all. The hw drops
 *   out-of-bounds stores for us.
 * - Out-of-bounds src coordinates are clamped by emulating CLAMP_TO_EDGE using
 *   the image_size NIR intrinsic.
 * - X/Y flipping just does this in the shader: -threadIDs - 1, assuming the starting coordinates
 *   are 1 pixel after the bottom-right corner, e.g. x + width, matching the gallium behavior.
 * - This list doesn't do it justice.
 */
nir_shader *
ac_create_blit_cs(const ac_cs_blit_options *options, const ac_cs_blit_key *key)
{
   if (options->print_key || debug_get_option_log_keys()) {
      static simple_mtx_t mtx = SIMPLE_MTX_INITIALIZER;
      FILE *f = stderr;

      if (debug_get_option_log_keys()) {
         /* The mutex prevents races during logging when shader variants are compiled in parallel. */
         simple_mtx_lock(&mtx);
         FILE *log = fopen(debug_get_option_log_keys(), "a");

         if (log) {
            f = log;
         } else {
            simple_mtx_unlock(&mtx);
            fprintf(stderr, "Can't open %s\n", debug_get_option_log_keys());
         }
      }

      if (f == stderr)
         fprintf(f, "Internal shader: compute_blit\n");

#define PRINT(field) print_shader_key_field(f, "key." #field, key->field)
      PRINT(use_aco);
      PRINT(wg_dim);
      PRINT(has_start_xyz);
      PRINT(format_is_96bit);
      PRINT(addr_math_64bit);
      PRINT(log_lane_width);
      PRINT(log_lane_height);
      PRINT(log_lane_depth);
      PRINT(is_clear);
      PRINT(src_is_sampler);
      PRINT(src_is_1d);
      PRINT(dst_is_1d);
      PRINT(src_is_msaa);
      PRINT(dst_is_msaa);
      PRINT(src_has_z);
      PRINT(dst_has_z);
      PRINT(src_has_non_identity_fmask);
      PRINT(dst_store_fmask);
      PRINT(copy_physical_samples);
      PRINT(dst_is_rgb5);
      PRINT(a16);
      PRINT(d16);
      PRINT(log_samples);
      PRINT(sample0_only);
      PRINT(x_clamp_to_edge);
      PRINT(y_clamp_to_edge);
      PRINT(flip_x);
      PRINT(flip_y);
      PRINT(sint_to_uint);
      PRINT(uint_to_sint);
      PRINT(dst_is_srgb);
      PRINT(use_integer_one);
      PRINT(last_src_channel);
      PRINT(last_dst_channel);
#undef PRINT

      fprintf(f, "\n");

      if (f != stderr) {
         fclose(f);
         simple_mtx_unlock(&mtx);
      }
   }

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options->nir_options,
                                                  "blit_non_scaled_cs");
   blake3_hasher blake3;
   _mesa_blake3_init(&blake3);
   _mesa_blake3_update(&blake3, b.shader->info.name, strlen(b.shader->info.name));
   _mesa_blake3_update(&blake3, key, sizeof(*key));
   _mesa_blake3_final(&blake3, b.shader->info.source_blake3);

   b.shader->info.use_aco_amd = options->use_aco ||
                                (key->use_aco && aco_is_gpu_supported(options->info));

   assert(!key->is_clear || (!key->src_is_sampler && !key->src_is_1d && !key->src_is_msaa &&
                             !key->src_has_z && !key->src_has_non_identity_fmask));
   const bool has_src_sampler = key->src_is_sampler;
   const bool has_src_image = !key->is_clear && !key->src_is_sampler;
   const bool has_dst_fmask = key->dst_store_fmask;

   /* The workgroup size varies depending on the tiling layout and blit dimensions. */
   b.shader->info.workgroup_size_variable = true;
   b.shader->info.cs.num_invocations_per_variable_workgroup = 64;
   b.shader->info.cs.user_data_components_amd = get_num_user_data_terms(key);

   nir_variable *img_src = NULL, *img_dst = NULL, *fmask_dst = NULL;
   nir_deref_instr *img_src_deref = NULL, *img_dst_deref = NULL, *fmask_dst_deref = NULL;
   nir_def *img_src_global_addr = NULL, *img_dst_global_addr = NULL;
   nir_def *zero = nir_imm_int(&b, 0);

   /* Declare resource variables. */
   if (key->format_is_96bit) {
      if (has_src_image) {
         nir_def *src_buf = nir_vulkan_resource_index(&b, 3, 32, zero,
                                                      .binding = 0,
                                                      .desc_type = nir_descriptor_type_storage_buffer);
         img_src_global_addr = nir_load_ssbo_address(&b, 1, 64, src_buf, zero);
      }

      nir_def *dst_buf = nir_vulkan_resource_index(&b, 3, 32, zero,
                                                   .binding = has_src_image,
                                                   .desc_type = nir_descriptor_type_storage_buffer);
      img_dst_global_addr = nir_load_ssbo_address(&b, 1, 64, dst_buf, zero);
   } else {
      /* These info fields are only used by radeonsi. */
      b.shader->info.num_images = has_src_image + 1 + has_dst_fmask;

      if (has_src_sampler)
         BITSET_SET(b.shader->info.textures_used, 0);

      if (has_src_image && key->src_is_msaa)
         BITSET_SET(b.shader->info.msaa_images, 0);

      if (key->dst_is_msaa)
         BITSET_SET(b.shader->info.msaa_images, has_src_image);

      const enum glsl_base_type img_data_type = key->d16 ? GLSL_TYPE_UINT16 : GLSL_TYPE_UINT;

      if (has_src_sampler) {
         const struct glsl_type *src_sampler_type =
            glsl_sampler_type(key->src_is_1d ? GLSL_SAMPLER_DIM_1D :
                              key->src_is_msaa ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
                              false, key->src_has_z, img_data_type);

         img_src = nir_variable_create(b.shader, nir_var_uniform, src_sampler_type, "src_sampler");
      }

      if (has_src_image) {
         const struct glsl_type *src_image_type =
            glsl_image_type(key->src_is_1d ? GLSL_SAMPLER_DIM_1D :
                            key->src_is_msaa ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
                            key->src_has_z, img_data_type);

         img_src = nir_variable_create(b.shader, nir_var_image, src_image_type, "src_image");
      }

      const struct glsl_type *dst_image_type =
         glsl_image_type(key->dst_is_1d ? GLSL_SAMPLER_DIM_1D :
                         key->dst_is_msaa ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
                         key->dst_has_z, img_data_type);

      img_dst = nir_variable_create(b.shader, nir_var_image, dst_image_type, "dst_image");
      /* NOTE: This is only correct for RADV where sampler and image bindings are in the same
       * descriptor list. radeonsi puts sampler bindings in a separate descriptor list, so this should
       * be 0 for has_src_sampler && !has_src_image when radeonsi starts using the sampler src.
       */
      img_dst->data.binding = has_src_sampler || has_src_image;

      if (has_dst_fmask) {
         const struct glsl_type *dst_fmask_type =
            glsl_image_type(GLSL_SAMPLER_DIM_2D, key->dst_has_z, GLSL_TYPE_UINT);

         fmask_dst = nir_variable_create(b.shader, nir_var_image, dst_fmask_type, "fmask_image");
         fmask_dst->data.binding = img_dst->data.binding + 1;
      }

      img_src_deref = img_src ? nir_build_deref_var(&b, img_src) : NULL;
      img_dst_deref = nir_build_deref_var(&b, img_dst);
      fmask_dst_deref = fmask_dst ? nir_build_deref_var(&b, fmask_dst) : NULL;
   }

   unsigned lane_width = 1 << key->log_lane_width;
   unsigned lane_height = 1 << key->log_lane_height;
   unsigned lane_depth = 1 << key->log_lane_depth;
   unsigned lane_size = lane_width * lane_height * lane_depth;
   assert(lane_size <= SI_MAX_COMPUTE_BLIT_LANE_SIZE);

   nir_def *zero_coord = nir_imm_intN_t(&b, 0, key->a16 ? 16 : 32);
   const unsigned coord_bit_size = key->a16 ? 16 : 32;

   /* Instructions. */
   /* Extract user data. */
   nir_def *user_data = nir_load_user_data_amd(&b);
   nir_def *start_x = NULL, *start_y = NULL, *start_z = NULL, *clear_value = NULL;
   nir_def *log_block_x = NULL, *log_block_y = NULL, *log_block_z = NULL;
   nir_def *dst_x = NULL, *dst_y = NULL, *dst_z = NULL;
   nir_def *src_x = NULL, *src_y = NULL, *src_z = NULL;
   nir_def *dst_dword_stride = NULL, *src_dword_stride = NULL;
   unsigned num = 0;

   if (key->format_is_96bit) {
      dst_dword_stride = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 0, 18);
      start_x = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 18, 6);
      start_y = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 24, 6);
      num++;
      dst_x = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 0, 18);
      log_block_x = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 18, 3);
      log_block_y = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 21, 3);
      num++;
      src_y = nir_u2u32(&b, nir_unpack_32_2x16_split_x(&b, nir_channel(&b, user_data, num)));
      dst_y = nir_u2u32(&b, nir_unpack_32_2x16_split_y(&b, nir_channel(&b, user_data, num)));
      num++;

      log_block_z = zero;
      start_z = zero;
      dst_z = zero;
      src_z = zero;

      if (!key->is_clear) {
         src_x = nir_channel(&b, user_data, num);
         num++;
         src_dword_stride = nir_channel(&b, user_data, num);
         num++;
      } else {
         clear_value = nir_channels(&b, user_data, BITFIELD_RANGE(num, 3));
         num += 3;
      }
   } else {
      start_x = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 0, 6);
      start_y = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 6, 6);
      start_z = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 12, 6);
      log_block_x = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 18, 3);
      log_block_y = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 21, 3);
      log_block_z = nir_ubfe_imm(&b, nir_channel(&b, user_data, num), 24, 3);
      num++;
      dst_x = nir_u2uN(&b, nir_unpack_32_2x16_split_x(&b, nir_channel(&b, user_data, num)),
                       coord_bit_size);
      dst_y = nir_u2uN(&b, nir_unpack_32_2x16_split_y(&b, nir_channel(&b, user_data, num)),
                       coord_bit_size);
      num++;

      if (key->dst_has_z || key->src_has_z) {
         dst_z = nir_u2uN(&b, nir_unpack_32_2x16_split_x(&b, nir_channel(&b, user_data, num)),
                          coord_bit_size);
         src_z = nir_u2uN(&b, nir_unpack_32_2x16_split_y(&b, nir_channel(&b, user_data, num)),
                          coord_bit_size);
         num++;
      } else {
         dst_z = zero_coord;
         src_z = zero_coord;
      }

      if (!key->is_clear) {
         src_x = nir_u2uN(&b, nir_unpack_32_2x16_split_x(&b, nir_channel(&b, user_data, num)),
                          coord_bit_size);
         src_y = nir_u2uN(&b, nir_unpack_32_2x16_split_y(&b, nir_channel(&b, user_data, num)),
                          coord_bit_size);
         num++;
      } else {
         if (key->d16) {
            unsigned num_clear_dwords = DIV_ROUND_UP(key->last_dst_channel + 1, 2);
            clear_value = nir_channels(&b, user_data, BITFIELD_RANGE(num, num_clear_dwords));

            if (num_clear_dwords == 2)
               clear_value = nir_unpack_64_4x16(&b, nir_pack_64_2x32(&b, clear_value));
            else
               clear_value = nir_unpack_32_2x16(&b, clear_value);

            num += num_clear_dwords;
         } else {
            clear_value = nir_channels(&b, user_data, BITFIELD_RANGE(num, key->last_dst_channel + 1));
            num += key->last_dst_channel + 1;
         }
      }
   }

   assert(num == b.shader->info.cs.user_data_components_amd);

   /* Let's work with 0-based src and dst coordinates (thread IDs) first. */
   nir_def *log_workgroup_size = nir_vec3(&b, log_block_x, log_block_y, log_block_z);
   nir_def *dst_xyz = nir_iadd_nuw(&b, nir_ishl(&b, nir_load_workgroup_id(&b), log_workgroup_size),
                                   nir_load_local_invocation_id(&b));
   dst_xyz = nir_u2uN(&b, nir_trim_vector(&b, dst_xyz, key->wg_dim), coord_bit_size);
   dst_xyz = nir_pad_vector_imm_int(&b, dst_xyz, 0, 3);

   /* If the blit area is unaligned, we launched extra threads to make it aligned.
    * Skip those threads here.
    */
   nir_if *if_positive = NULL;
   if (key->has_start_xyz) {
      nir_def *start_xyz = nir_u2uN(&b, nir_vec3(&b, start_x, start_y, start_z), coord_bit_size);
      start_xyz = nir_trim_vector(&b, start_xyz, 3);

      dst_xyz = nir_isub(&b, dst_xyz, start_xyz);
      nir_def *is_positive_xyz = nir_ige_imm(&b, dst_xyz, 0);
      nir_def *is_positive = nir_iand(&b, nir_channel(&b, is_positive_xyz, 0),
                                      nir_iand(&b, nir_channel(&b, is_positive_xyz, 1),
                                               nir_channel(&b, is_positive_xyz, 2)));
      if_positive = nir_push_if(&b, is_positive);
   }

   dst_xyz = nir_imul(&b, dst_xyz, nir_imm_ivec3_intN(&b, lane_width, lane_height, lane_depth,
                                                      coord_bit_size));
   nir_def *src_xyz = dst_xyz;

   /* Flip src coordinates. */
   for (unsigned i = 0; i < 2; i++) {
      if (i ? key->flip_y : key->flip_x) {
         /* A normal blit loads from (box.x + tid.x) where tid.x = 0..(width - 1).
          *
          * A flipped blit sets box.x = width, so we should make tid.x negative to load from
          * (width - 1)..0.
          *
          * Therefore do: x = -x - 1, which becomes (width - 1) to 0 after we add box.x = width.
          */
         nir_def *comp = nir_channel(&b, src_xyz, i);
         comp = nir_iadd_imm(&b, nir_ineg(&b, comp), -(int)(i ? lane_height : lane_width));
         src_xyz = nir_vector_insert_imm(&b, src_xyz, comp, i);
      }
   }

   /* Add box.xyz. */
   nir_def *base_coord_dst =
      nir_pad_vector(&b, nir_iadd(&b, nir_vec3(&b, dst_x, dst_y, dst_z), dst_xyz), 4);
   nir_def *base_coord_src = NULL;

   if (!key->is_clear) {
      base_coord_src =
         nir_pad_vector(&b, nir_iadd(&b, nir_vec3(&b, src_x, src_y, src_z), src_xyz), 4);
   }

/* Iterate over all pixels in the lane. num_samples is the only input.
 * (sample, x, y, z) are generated coordinates, while "i" is the coordinates converted to
 * an absolute index.
 */
#define foreach_pixel_in_lane(num_samples, sample, x, y, z, i) \
   for (unsigned z = 0; z < lane_depth; z++) \
      for (unsigned y = 0; y < lane_height; y++) \
         for (unsigned x = 0; x < lane_width; x++) \
            for (unsigned i = ((z * lane_height + y) * lane_width + x) * (num_samples), sample = 0; \
                 sample < (num_samples); sample++, i++) \

   /* Swizzle coordinates for 1D_ARRAY. */
   static const unsigned swizzle_xz[] = {0, 2, 0, 0};

   /* Execute image loads and stores. */
   unsigned num_src_coords = (key->src_is_1d ? 1 : 2) + key->src_has_z + key->src_is_msaa;
   unsigned num_dst_coords = (key->dst_is_1d ? 1 : 2) + key->dst_has_z + key->dst_is_msaa;
   unsigned num_samples = 1 << key->log_samples;
   unsigned src_samples = key->src_is_msaa && !key->sample0_only &&
                          !key->is_clear ? num_samples : 1;
   unsigned dst_samples = key->dst_is_msaa && !(key->is_clear && key->sample0_only) ? num_samples : 1;
   nir_def *color[SI_MAX_COMPUTE_BLIT_LANE_SIZE * SI_MAX_COMPUTE_BLIT_SAMPLES] = {0};
   nir_def *coord_dst[SI_MAX_COMPUTE_BLIT_LANE_SIZE * SI_MAX_COMPUTE_BLIT_SAMPLES] = {0};
   nir_def *global_addr_dst[SI_MAX_COMPUTE_BLIT_LANE_SIZE] = {0};
   nir_def *src_resinfo = NULL;
   nir_def *fmask[SI_MAX_COMPUTE_BLIT_LANE_SIZE];
   nir_def *max_sample_in_fmask = NULL;

   if (key->is_clear) {
      if (!key->format_is_96bit) {
         color[0] = clear_value;

         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            color[i] = color[0];
         }
      }
   } else {
      nir_def *coord_src[SI_MAX_COMPUTE_BLIT_LANE_SIZE * SI_MAX_COMPUTE_BLIT_SAMPLES] = {0};
      nir_def *global_addr_src[SI_MAX_COMPUTE_BLIT_LANE_SIZE] = {0};

      /* Initialize src coordinates, one vector per pixel. */
      foreach_pixel_in_lane(src_samples, sample, x, y, z, i) {
         unsigned tmp_x = x;
         unsigned tmp_y = y;

         /* Change the order from 0..N to N..0 for flipped blits. */
         if (key->flip_x)
            tmp_x = lane_width - 1 - x;
         if (key->flip_y)
            tmp_y = lane_height - 1 - y;

         coord_src[i] = nir_iadd(&b, base_coord_src,
                                     nir_imm_ivec4_intN(&b, tmp_x, tmp_y, z, 0, coord_bit_size));
         if (key->src_is_1d)
            coord_src[i] = nir_swizzle(&b, coord_src[i], swizzle_xz, 4);
         if (key->src_is_msaa) {
            coord_src[i] = nir_vector_insert_imm(&b, coord_src[i],
                                                 nir_imm_intN_t(&b, sample, coord_bit_size),
                                                 num_src_coords - 1);
         }

         /* Clamp to edge for src, only X and Y because Z can't be out of bounds. */
         for (unsigned chan = 0; chan < 2; chan++) {
            if (chan ? key->y_clamp_to_edge : key->x_clamp_to_edge) {
               assert(!key->src_is_1d || chan == 0);

               if (!src_resinfo) {
                  /* Always use the 32-bit return type because the image dimensions can be
                   * > INT16_MAX even if the blit box fits within sint16. The txs builder
                   * always sets the type to nir_type_int32.
                   */
                  if (has_src_sampler) {
                     src_resinfo = nir_txs(&b, .texture_deref = img_src_deref,
                                           .dim = img_src->type->sampler_dimensionality,
                                           .is_array = img_src->type->sampler_array);
                  } else {
                     src_resinfo = nir_image_deref_size(&b, 4, 32, &img_src_deref->def,
                                                        zero_coord);
                  }

                  if (coord_bit_size == 16) {
                     src_resinfo = nir_umin_imm(&b, src_resinfo, INT16_MAX);
                     src_resinfo = nir_i2i16(&b, src_resinfo);
                  }
               }

               nir_def *tmp = nir_channel(&b, coord_src[i], chan);
               tmp = nir_imax_imm(&b, tmp, 0);
               tmp = nir_imin(&b, tmp, nir_iadd_imm(&b, nir_channel(&b, src_resinfo, chan), -1));
               coord_src[i] = nir_vector_insert_imm(&b, coord_src[i], tmp, chan);
            }
         }

         /* Compute the src offset for global-addressed images. */
         if (key->format_is_96bit) {
            assert(src_samples == 1);

            global_addr_src[i] = get_global_image2d_addr(&b, key, img_src_global_addr, coord_src[i],
                                                         src_dword_stride);
         }
      }

      /* We don't want the computation of src coordinates to be interleaved with loads. */
      if (lane_size > 1 || src_samples > 1) {
         ac_optimization_barrier_vgpr_array(options->info, &b, coord_src,
                                            lane_size * src_samples, num_src_coords);

         if (key->format_is_96bit)
            ac_optimization_barrier_vgpr_array(options->info, &b, global_addr_src, lane_size, 1);
      }

      /* Use "samples_identical" for MSAA resolving if it's supported. */
      const bool is_resolve = src_samples > 1 && dst_samples == 1;
      const bool uses_samples_identical = key->src_has_non_identity_fmask && is_resolve;
      nir_def *samples_identical = NULL, *sample0[SI_MAX_COMPUTE_BLIT_LANE_SIZE] = {0};
      nir_if *if_identical = NULL;

      if (uses_samples_identical) {
         samples_identical = nir_imm_true(&b);

         /* If we are resolving multiple pixels per lane, AND all results of "samples_identical". */
         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            nir_def *iden;

            if (has_src_sampler) {
               iden = tex(&b, key, img_src_deref, nir_texop_samples_identical,
                          coord_src[i * src_samples], num_src_coords);
            } else {
               iden = nir_image_deref_samples_identical(&b, 1, &img_src_deref->def,
                                                        coord_src[i * src_samples],
                                                        .image_dim = GLSL_SAMPLER_DIM_MS,
                                                        .image_array = img_src->type->sampler_array);
            }
            samples_identical = nir_iand(&b, samples_identical, iden);
         }

         /* If all samples are identical, load only sample 0. */
         if_identical = nir_push_if(&b, samples_identical);
         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            sample0[i] = load_src(&b, key, img_src_deref, nir_texop_fragment_fetch_amd,
                                  coord_src[i * src_samples], num_src_coords);
         }
         nir_push_else(&b, if_identical);
      }

      /* Load FMASK if we are copying physical samples to know how many samples we need to copy. */
      if (key->copy_physical_samples) {
         assert(!key->is_clear);

         if (key->src_has_non_identity_fmask) {
            foreach_pixel_in_lane(1, sample, x, y, z, i) {
               fmask[i] = load_src(&b, key, img_src_deref, nir_texop_fragment_mask_fetch_amd,
                                   coord_src[i * src_samples], num_src_coords);
            }
         } else {
            nir_def *identity = nir_imm_int(&b, 0x76543210);

            foreach_pixel_in_lane(1, sample, x, y, z, i) {
               fmask[i] = identity;
            }
         }

         /* Compute the maximum sample index in FMASK for all pixels in the lane group. */
         max_sample_in_fmask = zero;

         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            for (unsigned s = 0; s < src_samples; s++) {
               max_sample_in_fmask = nir_umax(&b, max_sample_in_fmask,
                                              nir_ubfe_imm(&b, fmask[i], s * 4, 4));
            }
         }
      }

      /* Load src pixels, one per sample. */
      if (key->format_is_96bit) {
         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            color[i] = nir_load_global(&b, 1, 32, global_addr_src[i],
                                       .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE,
                                       .align_mul = 4);
         }
      } else if (key->copy_physical_samples) {
         /* Only the samples referenced by FMASK are loaded. */
         for (unsigned sample = 0; sample < src_samples; sample++) {
            /* The first sample is loaded unconditionally. Others are loaded conditionally.
             * If multiple pixels are loaded per lane, the whole lane group is loaded.
             */
            if (sample)
               nir_push_if(&b, nir_uge_imm(&b, max_sample_in_fmask, sample));

            foreach_pixel_in_lane(1, _, x, y, z, i) {
               unsigned index = i * src_samples + sample;

               color[index] = load_src(&b, key, img_src_deref, nir_texop_fragment_fetch_amd,
                                       coord_src[index], num_src_coords);
            }
         }

         /* Close the ifs and insert phis. */
         for (int sample = src_samples - 1; sample >= 1; sample--) {
            nir_pop_if(&b, NULL);

            /* We have to insert phis for all samples >= the current sample because all
             * those loads are in nested conditional blocks.
             */
            for (unsigned phi_sample = sample; phi_sample < src_samples; phi_sample++) {
               foreach_pixel_in_lane(1, _, x, y, z, i) {
                  nir_def **value = &color[i * src_samples + phi_sample];

                  *value = nir_if_phi(&b, *value, nir_undef(&b, (*value)->num_components,
                                                            (*value)->bit_size));
               }
            }
         }
      } else {
         foreach_pixel_in_lane(src_samples, sample, x, y, z, i) {
            color[i] = load_src(&b, key, img_src_deref,
                                key->src_is_msaa ?
                                   (key->src_has_non_identity_fmask ? nir_texop_txf_ms :
                                                                      nir_texop_fragment_fetch_amd) :
                                   nir_texop_txf, coord_src[i], num_src_coords);
         }
      }

      /* Resolve MSAA if necessary. */
      if (is_resolve) {
         /* We don't want the averaging of samples to be interleaved with image loads. */
         ac_optimization_barrier_vgpr_array(options->info, &b, color, lane_size * src_samples,
                                            key->last_src_channel + 1);

         /* This reduces the "color" array from "src_samples * lane_size" elements to only
          * "lane_size" elements.
          */
         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            color[i] = ac_average_samples(&b, &color[i * src_samples], src_samples);
         }
         src_samples = 1;
      }

      if (uses_samples_identical) {
         nir_pop_if(&b, if_identical);
         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            color[i] = nir_if_phi(&b, sample0[i], color[i]);
         }
      }
   }

   /* We need to load the descriptor here, otherwise the load would be after optimization
    * barriers waiting for image loads, i.e. after s_waitcnt vmcnt(0).
    */
   nir_def *img_dst_desc = NULL, *fmask_dst_desc = NULL;

   if (!key->format_is_96bit) {
      img_dst_desc =
         nir_image_deref_descriptor_amd(&b, 8, 32, &img_dst_deref->def,
                                        .image_dim = img_dst->type->sampler_dimensionality,
                                        .image_array = img_dst->type->sampler_array);

      if (key->dst_store_fmask) {
         fmask_dst_desc =
            nir_image_deref_descriptor_amd(&b, 8, 32, &fmask_dst_deref->def,
                                           .image_dim = fmask_dst->type->sampler_dimensionality,
                                           .image_array = fmask_dst->type->sampler_array);
      }

      if (lane_size > 1 && !b.shader->info.use_aco_amd) {
         img_dst_desc = nir_optimization_barrier_sgpr_amd(&b, 32, img_dst_desc);
         if (fmask_dst_desc)
            fmask_dst_desc = nir_optimization_barrier_sgpr_amd(&b, 32, fmask_dst_desc);
      }

      /* Apply the blit output modifiers, once per sample.  */
      foreach_pixel_in_lane(src_samples, sample, x, y, z, i) {
         color[i] = apply_blit_output_modifiers(&b, color[i], key);
      }
   }

   /* Initialize dst coordinates, one vector per pixel. */
   foreach_pixel_in_lane(dst_samples, sample, x, y, z, i) {
      coord_dst[i] = nir_iadd(&b, base_coord_dst,
                              nir_imm_ivec4_intN(&b, x, y, z, 0, coord_bit_size));
      if (key->dst_is_1d)
         coord_dst[i] = nir_swizzle(&b, coord_dst[i], swizzle_xz, 4);
      if (key->dst_is_msaa) {
         coord_dst[i] = nir_vector_insert_imm(&b, coord_dst[i],
                                              nir_imm_intN_t(&b, sample, coord_bit_size),
                                              num_dst_coords - 1);
      }

      if (key->format_is_96bit) {
         global_addr_dst[i] = get_global_image2d_addr(&b, key, img_dst_global_addr, coord_dst[i],
                                                      dst_dword_stride);
      }
   }

   /* Prepare the clear value for 96-bit formats. */
   if (key->is_clear && key->format_is_96bit) {
      foreach_pixel_in_lane(1, sample, x, y, z, i) {
         /* Select the clear value dword that belongs to this dword address. */
         nir_def *coord_x = nir_channel(&b, coord_dst[i], 0);
         color[i] = nir_vector_extract(&b, clear_value, nir_umod_imm(&b, coord_x, 3));
      }
   }

   /* We don't want the computation of dst coordinates to be interleaved with stores. */
   if (lane_size > 1 || dst_samples > 1) {
      ac_optimization_barrier_vgpr_array(options->info, &b, coord_dst, lane_size * dst_samples,
                                         num_dst_coords);

      if (key->format_is_96bit)
         ac_optimization_barrier_vgpr_array(options->info, &b, global_addr_dst, lane_size, 1);
   }

   /* We don't want the application of blit output modifiers to be interleaved with stores. */
   if (!key->is_clear && (lane_size > 1 || MIN2(src_samples, dst_samples) > 1)) {
      ac_optimization_barrier_vgpr_array(options->info, &b, color, lane_size * src_samples,
                                         key->last_dst_channel + 1);
   }

   /* Store the pixels, one per sample. */
   if (!key->is_clear && key->copy_physical_samples) {
      assert(!key->format_is_96bit);
      assert(src_samples > 1);
      assert(src_samples == dst_samples);

      /* Only the samples referenced by FMASK are stored. */
      for (unsigned sample = 0; sample < dst_samples; sample++) {
         /* The first sample is stored unconditionally. Others are stored conditionally.
          * If multiple pixels are stored per lane, the whole lane group is stored.
          */
         if (sample)
            nir_push_if(&b, nir_uge_imm(&b, max_sample_in_fmask, sample));

         foreach_pixel_in_lane(1, _, x, y, z, i) {
            unsigned index = i * dst_samples + sample;

            nir_bindless_image_store(&b, img_dst_desc, coord_dst[index],
                                     nir_channel(&b, coord_dst[index], num_dst_coords - 1),
                                     color[index], zero_coord,
                                     .image_dim = img_dst->type->sampler_dimensionality,
                                     .image_array = img_dst->type->sampler_array);
         }
      }

      /* Close the ifs and insert phis. */
      for (int sample = dst_samples - 1; sample >= 1; sample--)
         nir_pop_if(&b, NULL);
   } else {
      foreach_pixel_in_lane(dst_samples, sample, x, y, z, i) {
         if (key->format_is_96bit) {
            nir_store_global(&b, color[i], global_addr_dst[i],
                             .access = ACCESS_CAN_REORDER | ACCESS_NON_READABLE,
                             .align_mul = 4);
         } else {
            nir_bindless_image_store(&b, img_dst_desc, coord_dst[i],
                                     nir_channel(&b, coord_dst[i], num_dst_coords - 1),
                                     src_samples > 1 ? color[i] : color[i / dst_samples], zero_coord,
                                     .image_dim = img_dst->type->sampler_dimensionality,
                                     .image_array = img_dst->type->sampler_array);
         }
      }
   }

   /* Store FMASK pixels. */
   if (key->dst_store_fmask) {
      assert(!key->format_is_96bit);

      if (key->is_clear) {
         assert(key->sample0_only);

         /* Clear FMASK to 0. */
         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            nir_bindless_image_store(&b, fmask_dst_desc, coord_dst[i], zero_coord, zero, zero_coord,
                                     .image_dim = fmask_dst->type->sampler_dimensionality,
                                     .image_array = fmask_dst->type->sampler_array);
         }
      } else {
         /* Store FMASK that was loaded from src or identity. */
         foreach_pixel_in_lane(1, sample, x, y, z, i) {
            nir_def *value;

            /* Pack the sample indices in FMASK to get the physical encoding. */
            if (key->log_samples == 3) {
               value = fmask[i];
            } else if (key->log_samples == 2) {
               /* Pack from 4-bit indices to 2-bit indices. */
               value = nir_ior(&b, nir_ior(&b, nir_ubfe_imm(&b, fmask[i], 0, 2),
                                           nir_ishl_imm(&b, nir_ubfe_imm(&b, fmask[i], 4, 2), 2)),
                               nir_ior(&b, nir_ishl_imm(&b, nir_ubfe_imm(&b, fmask[i], 8, 2), 4),
                                       nir_ishl_imm(&b, nir_ubfe_imm(&b, fmask[i], 12, 2), 6)));
            } else if (key->log_samples == 1) {
               /* Pack from 4-bit indices to 1-bit indices. */
               value = nir_ior(&b, nir_ubfe_imm(&b, fmask[i], 0, 1),
                               nir_ishl_imm(&b, nir_ubfe_imm(&b, fmask[i], 4, 1), 1));
            } else {
               UNREACHABLE("invalid log_samples");
            }

            nir_bindless_image_store(&b, fmask_dst_desc, coord_dst[i * dst_samples],
                                     zero_coord, value, zero_coord,
                                     .image_dim = fmask_dst->type->sampler_dimensionality,
                                     .image_array = fmask_dst->type->sampler_array);
         }
      }
   }

   if (key->has_start_xyz)
      nir_pop_if(&b, if_positive);

   return b.shader;
}

static unsigned
set_work_size(ac_cs_blit_dispatch *dispatch,
              unsigned block_x, unsigned block_y, unsigned block_z,
              unsigned num_invoc_x, unsigned num_invoc_y, unsigned num_invoc_z)
{
   const unsigned block[3] = {block_x, block_y, block_z};
   const unsigned num_invoc[3] = {num_invoc_x, num_invoc_y, num_invoc_z};

   for (int i = 0; i < 3; ++i) {
      dispatch->wg_size[i] = block[i];
      dispatch->last_wg_size[i] = num_invoc[i] % dispatch->wg_size[i];
      dispatch->num_workgroups[i] = DIV_ROUND_UP(num_invoc[i], dispatch->wg_size[i]);
      dispatch->num_invocations[i] = num_invoc[i];
   }

   return num_invoc_z > 1 ? 3 : (num_invoc_y > 1 ? 2 : 1);
}

static bool
should_blit_clamp_to_edge(const ac_cs_blit_description *blit, unsigned coord_mask)
{
   return util_is_box_out_of_bounds(&blit->src.box, coord_mask, blit->src.width0,
                                    blit->src.height0, 1, blit->src.level);
}

/* Return a power-of-two alignment of a number. */
static unsigned
compute_alignment(unsigned x)
{
   return x ? BITFIELD_BIT(ffs(x) - 1) : BITFIELD_BIT(31);
}

/* Set the blit info, but change the dst box and trim the src box according to the new dst box. */
static void
set_trimmed_blit(const ac_cs_blit_description *old, const struct pipe_box *box,
                 bool is_clear, ac_cs_blit_description *out)
{
   assert(old->dst.box.x <= box->x);
   assert(old->dst.box.y <= box->y);
   assert(old->dst.box.z <= box->z);
   assert(box->x + box->width <= old->dst.box.x + old->dst.box.width);
   assert(box->y + box->height <= old->dst.box.y + old->dst.box.height);
   assert(box->z + box->depth <= old->dst.box.z + old->dst.box.depth);
   /* No scaling. */
   assert(is_clear || old->dst.box.width == abs(old->src.box.width));
   assert(is_clear || old->dst.box.height == abs(old->src.box.height));
   assert(is_clear || old->dst.box.depth == abs(old->src.box.depth));

   *out = *old;
   out->dst.box = *box;

   if (!is_clear) {
      if (out->src.box.width > 0) {
         out->src.box.x += box->x - old->dst.box.x;
         out->src.box.width = box->width;
      } else {
         out->src.box.x -= box->x - old->dst.box.x;
         out->src.box.width = -box->width;
      }

      if (out->src.box.height > 0) {
         out->src.box.y += box->y - old->dst.box.y;
         out->src.box.height = box->height;
      } else {
         out->src.box.y -= box->y - old->dst.box.y;
         out->src.box.height = -box->height;
      }

      out->src.box.z += box->z - old->dst.box.z;
      out->src.box.depth = box->depth;
   }
}

typedef struct {
   bool is_clear;
   bool dst_is_stencil;
   bool src_is_stencil;
   ac_block_extent dst_block256B_extent;
   ac_block_extent src_block256B_extent;
   unsigned dst_tiling_dim;
   unsigned src_tiling_dim;
   unsigned dst_samples;
   unsigned src_samples;
   bool is_resolve;
   bool is_upsampling;
   bool sample0_only;
   unsigned max_dst_chan_size;
   unsigned max_src_chan_size;
   unsigned dst_bpe;
} ac_cs_blit_derived_info;

static void
get_derived_info(const ac_cs_blit_description *blit, ac_cs_blit_derived_info *out)
{
   out->is_clear = !blit->src.surf;
   out->dst_is_stencil = util_format_is_stencil_only(blit->dst.format);
   out->src_is_stencil = util_format_is_stencil_only(blit->src.format);
   out->dst_block256B_extent =
      out->dst_is_stencil ? blit->dst.surf->stencil_block256B_extent : blit->dst.surf->block256B_extent;
   out->src_block256B_extent =
      out->is_clear ? (ac_block_extent){}
                    : out->src_is_stencil ? blit->src.surf->stencil_block256B_extent
                                          : blit->src.surf->block256B_extent;
   out->dst_tiling_dim = blit->dst.surf->is_linear ? 1 : out->dst_block256B_extent.depth == 1 ? 2 : 3;
   out->src_tiling_dim =
      out->is_clear ? 0 : blit->src.surf->is_linear ? 1 : out->src_block256B_extent.depth == 1 ? 2 : 3;
   out->dst_samples = MAX2(1, blit->dst.num_samples);
   out->src_samples = out->is_clear ? 1 : MAX2(1, blit->src.num_samples);
   out->is_resolve = !out->is_clear && out->dst_samples == 1 && out->src_samples >= 2 &&
                     !util_format_is_pure_integer(blit->dst.format);
   out->is_upsampling = !out->is_clear && out->src_samples == 1 && out->dst_samples >= 2;
   out->sample0_only = (out->is_clear && out->dst_samples >= 2 && blit->sample0_only) ||
                       (!out->is_clear && out->src_samples >= 2 && out->dst_samples == 1 &&
                        (blit->sample0_only || util_format_is_pure_integer(blit->dst.format)));
   /* Get the channel sizes. */
   out->max_dst_chan_size = util_format_get_max_channel_size(blit->dst.format);
   out->max_src_chan_size = out->is_clear ? 0 : util_format_get_max_channel_size(blit->src.format);
   out->dst_bpe = blit->format_is_96bit ? 4 : out->dst_is_stencil ? 1 : blit->dst.surf->bpe;
}

static bool
has_errors(const ac_cs_blit_description *blit)
{
   /* Reject blits with invalid parameters. */
   if (blit->dst.box.x < 0 || blit->dst.box.y < 0 || blit->dst.box.z < 0 ||
       blit->dst.box.width < 0 || blit->dst.box.height < 0 || blit->dst.box.depth < 0 ||
       blit->src.box.depth < 0) {
      assert(!"invalid box parameters"); /* this is reachable and prevents hangs */
      return true;
   }

   return false;
}

static bool
boxes_are_supported(const ac_cs_blit_description *blit)
{
   const bool is_clear = !blit->src.surf;

   /* Negative src is possible with piglit/copyteximage-clipping, but it's very rare. */
   if (blit->src.box.x < 0 || blit->src.box.y < 0 || blit->src.box.z < 0)
      return false;

   if (!is_clear &&
       /* Scaling is not implemented by the compute shader. */
       (blit->dst.box.width != abs(blit->src.box.width) ||
        blit->dst.box.height != abs(blit->src.box.height) ||
        blit->dst.box.depth != abs(blit->src.box.depth)))
      return false;

   return true;
}

bool
ac_should_use_compute_blit(const ac_cs_blit_options *options, const ac_cs_blit_description *blit)
{
   const struct radeon_info *info = options->info;
   ac_cs_blit_derived_info d;
   get_derived_info(blit, &d);

   /* We don't want the errors to cause the driver to use the gfx path, so return success, so that
    * the errors are contained. Then ac_prepare_compute_blit will skip everything.
    */
   if (has_errors(blit))
      return true;

   if (!boxes_are_supported(blit))
      return false;

   if (blit->dst.format == PIPE_FORMAT_A8R8_UNORM || /* This format fails AMD_TEST=imagecopy. */
       /* GFX8 doesn't seem to support R4G4B4A4_UINT and R5G5B5A1_UINT. VK doesn't have these formats. */
       (info->gfx_level <= GFX8 && d.max_dst_chan_size <= 6 &&
        util_format_is_pure_integer(blit->dst.format)) ||
       d.dst_samples > SI_MAX_COMPUTE_BLIT_SAMPLES ||
       (!d.is_clear && d.src_samples > SI_MAX_COMPUTE_BLIT_SAMPLES))
      return false;

   /* Return a failure if a compute blit is slower than a gfx blit. */
   if (blit->fail_if_slow) {
      /* TODO: This block is no longer up to date since the last rewrite. */
      if (d.is_clear) {
         /* Verified on: Tahiti, Hawaii, Tonga, Vega10, Navi10, Navi21, Navi31 */
         if (d.dst_tiling_dim == 3) {
            if (info->gfx_level == GFX6 && d.dst_bpe == 8)
               return false;
         } else if (d.dst_tiling_dim == 2) {
            /* Each condition inside the parentheses enables the compute clear for that case. */
            if (!(info->gfx_level == GFX6 && d.dst_bpe <= 4 && d.dst_samples == 1) &&
                !(info->gfx_level == GFX7 && d.dst_bpe == 1 && d.dst_samples == 1) &&
                !(info->gfx_level == GFX12 &&
                  (d.dst_bpe <= 2 ||
                   (d.dst_bpe == 4 && d.dst_samples == 8) ||
                   (d.dst_bpe == 16 && d.dst_samples <= 2))))
               return false;
         }
      } else {
         /* TODO: Optionally add an MSAA upsampling path for OpenGL. */
         if (d.is_upsampling)
            return false;

         switch (info->gfx_level) {
         case GFX6:
         case GFX7:
         case GFX8:
         case GFX9:
         case GFX10:
         case GFX10_3:
            /* Verified on: Tahiti, Hawaii, Tonga, Vega10, Navi10, Navi21 */
            if (d.is_resolve) {
               if (!(info->gfx_level == GFX7 && d.dst_bpe == 16))
                  return false;
            } else if (!blit->copy_physical_samples) {
               assert(d.dst_samples == d.src_samples || d.sample0_only);

               if (d.dst_tiling_dim == 2) {
                  if (d.dst_samples == 1) {
                     if (d.dst_bpe <= 8 &&
                         !(info->gfx_level <= GFX7 && d.dst_bpe == 1) &&
                         !(info->gfx_level == GFX6 && d.dst_bpe == 2 && d.src_tiling_dim == 1) &&
                         !(info->gfx_level == GFX7 && d.dst_bpe >= 2 && d.src_tiling_dim == 1) &&
                         !((info->gfx_level == GFX8 || info->gfx_level == GFX9) && d.dst_bpe >= 2 &&
                           d.src_tiling_dim == 1) &&
                         !(info->gfx_level == GFX10 && d.dst_bpe <= 2 && d.src_tiling_dim == 1) &&
                         !(info->gfx_level == GFX10_3 && d.dst_bpe == 8 && d.src_tiling_dim == 1))
                        return false;

                     if (info->gfx_level == GFX6 && d.dst_bpe == 16 && d.src_tiling_dim == 1 &&
                         blit->dst.dim != 3)
                        return false;

                     if (d.dst_bpe == 16 && d.src_tiling_dim >= 2 &&
                         /* Only GFX6 selects 2D tiling for 128bpp 3D textures. */
                         !(info->gfx_level == GFX6 && blit->dst.dim == 3) &&
                         info->gfx_level != GFX7)
                        return false;
                  } else {
                     /* MSAA copies - tested only without FMASK on Navi21. */
                     if (d.dst_bpe >= 4)
                        return false;
                  }
               }
            }
            break;

         case GFX11:
         case GFX11_5:
         case GFX11_7:
            /* Verified on Navi31. */
            if (d.is_resolve) {
               if (!((d.dst_bpe <= 2 && d.src_samples == 2) ||
                     (d.dst_bpe == 2 && d.src_samples == 4) ||
                     (d.dst_bpe == 16 && d.src_samples == 4)))
                  return false;
            } else if (!blit->copy_physical_samples) {
               assert(d.dst_samples == d.src_samples || d.sample0_only);

               if (d.dst_tiling_dim == 2) {
                  if (d.dst_bpe == 2 && d.src_tiling_dim == 1 && d.dst_samples == 1)
                     return false;

                  if (d.dst_bpe >= 4 && d.dst_samples == 1 && d.src_tiling_dim >= 2)
                     return false;

                  if (d.dst_bpe == 16 && d.dst_samples == 8)
                     return false;
               }
            }
            break;

         default:
         case GFX12:
            /* Verified on Navi48. */
            break;
         }
      }
   }

   return true;
}

typedef struct {
   unsigned x, y, z;
} uvec3;

/* This function uses the blit description to generate the shader key, prepare user SGPR constants,
 * and determine the parameters for up to 7 compute dispatches.
 *
 * The driver should use the shader key to create the shader, set the SGPR constants, and launch
 * compute dispatches.
 */
bool
ac_prepare_compute_blit(const ac_cs_blit_options *options,
                        const ac_cs_blit_description *blit,
                        ac_cs_blit_dispatches *out)
{
   const struct radeon_info *info = options->info;
   ac_cs_blit_derived_info d;
   get_derived_info(blit, &d);

   if (!options->is_nested)
      memset(out, 0, sizeof(*out));

   if (has_errors(blit))
      return true;

   /* Skip zero-area blits. */
   if (!blit->dst.box.width || !blit->dst.box.height || !blit->dst.box.depth ||
       (!d.is_clear && (!blit->src.box.width || !blit->src.box.height || !blit->src.box.depth)))
      return true;

   assert(boxes_are_supported(blit));

   /* Determine bytes per lane, which is the main tunable. */
   /* TODO: retune everything with RADV, especially MSAA copies on GFX10.3 */
   unsigned bytes_per_lane = d.dst_bpe;

   if (info->gfx_level >= GFX11) {
      if (d.is_resolve) {
         static const unsigned resolve_bpl[][3] = {
            /* This performs best maybe due to interleaved samples, not sure. */
            /* 2, 4, 8 samples */
            {4, 2, 1}, /* 1 bpe */
            {8, 4, 2}, /* 2 bpe */
            {8, 8, 4}, /* 4 bpe */
            {8, 8, 8}, /* 8 bpe */
            {16, 16, 16}, /* 16 bpe */
         };

         bytes_per_lane = resolve_bpl[util_logbase2(d.dst_bpe)][util_logbase2(d.src_samples) - 1];

         if (info->gfx_level >= GFX12 && d.dst_bpe == 8 && d.src_samples == 2)
            bytes_per_lane = 16;
      } else {
         static const unsigned bpl[][4] = {
            /* TODO: maybe rgba32f-8s would be faster if it copied fewer samples per lane */
            /* 1, 2, 4, 8 samples */
            {8, 8, 8, 8}, /* 1 bpe */
            {16, 16, 16, 16}, /* 2 bpe */
            {16, 32, 16, 32}, /* 4 bpe */
            {16, 32, 32, 64}, /* 8 bpe */
            {16, 32, 64, 128}, /* 16 bpe */
         };

         bytes_per_lane = bpl[util_logbase2(d.dst_bpe)][util_logbase2(d.dst_samples)];

         if (info->gfx_level >= GFX12 && d.dst_bpe == 4 && d.src_samples == 2)
            bytes_per_lane = 16;
      }
   } else if (info->gfx_level >= GFX10) {
      bytes_per_lane = MAX2(bytes_per_lane, 4);

      /* Special cases that were found to perform better than the default. */
      if (d.is_resolve && d.dst_bpe == 1 && d.src_samples >= 4)
         bytes_per_lane = 2;

      bytes_per_lane = MAX2(bytes_per_lane, d.dst_bpe * d.dst_samples);
   } else {
      /* GFX6-9. */
      bytes_per_lane = MAX3(bytes_per_lane, 8, d.dst_bpe * d.dst_samples);

      if (d.is_clear && d.dst_bpe >= 4 && d.dst_tiling_dim == 3)
         bytes_per_lane = 16;

      if (info->gfx_level <= GFX7 && d.is_resolve) {
         static const unsigned resolve_bpl[][3] = {
            /* 2, 4, 8 samples */
            {4, 4, 2}, /* 1 bpe */
            {8, 8, 4}, /* 2 bpe */
            {8, 8, 8}, /* 4 bpe */
            {8, 8, 8}, /* 8 bpe */
            {16, 16, 16}, /* 16 bpe */
         };

         bytes_per_lane = resolve_bpl[util_logbase2(d.dst_bpe)][util_logbase2(d.src_samples) - 1];
      }
   }

   /* Use global_store_dwordx4. */
   if (blit->format_is_96bit)
      bytes_per_lane = 16;

   uvec3 tile_size = {
      d.dst_block256B_extent.width,
      d.dst_block256B_extent.height,
      d.dst_block256B_extent.depth,
   };

   /* Tuning for tiled<->linear copies. */
   /* TODO: This is only flyby tuning. More careful tuning is needed if this is important. */
   if (d.dst_tiling_dim == 1 || d.src_tiling_dim == 1) {
      /* 2D->linear. */
      if (d.dst_tiling_dim == 1 && d.src_tiling_dim == 2) {
         /* Make the tile (N/4, 4) instead of (N, 1). */
         if (info->gfx_level >= GFX9 || d.dst_bpe <= 1) {
            assert(tile_size.x >= 4);
            tile_size.x /= 4;
            tile_size.y *= 4;
         } else if (d.dst_bpe == 2) {
            assert(tile_size.x >= 2);
            tile_size.x /= 2;
            tile_size.y *= 2;
            bytes_per_lane = 4;
         }

         if (info->gfx_level <= GFX7 && d.dst_bpe == 8) {
            assert(tile_size.x >= 2);
            tile_size.x /= 2;
            tile_size.y *= 2;
         }
      }

      /* 3D->linear. */
      if (d.dst_tiling_dim == 1 && d.src_tiling_dim == 3) {
         if (info->gfx_level >= GFX9 || d.dst_bpe == 1) {
            /* Make the tile (N/4, 2, 2) instead of (N, 1, 1). */
            assert(tile_size.x >= 4);
            tile_size.x /= 4;
            tile_size.y *= 2;
            tile_size.z *= 2;
            bytes_per_lane = MAX2(bytes_per_lane, 8);
         }

         if (info->gfx_level <= GFX7 && d.dst_bpe == 2) {
            assert(tile_size.x >= 2);
            tile_size.x /= 2;
            tile_size.y *= 2;
         }
      }

      /* Linear->2D. */
      if (info->gfx_level <= GFX7 && d.dst_bpe >= 2 && d.dst_tiling_dim == 2 && d.src_tiling_dim == 1) {
         assert(tile_size.y >= 2);
         tile_size.y /= 2;
         tile_size.x *= 2;
      }

      /* Linear->3D. */
      if (d.dst_tiling_dim == 3 && d.src_tiling_dim == 1) {
         if (info->gfx_level >= GFX11) {
            assert(tile_size.y >= 2 && tile_size.z >= 2);
            tile_size.y /= 2;
            tile_size.z /= 2;
            tile_size.x *= 4;
         }

         if (info->gfx_level == GFX9) {
            assert(tile_size.z >= 4);
            tile_size.z /= 4;
            tile_size.x *= 4;
         }

         if (info->gfx_level <= GFX8) {
            unsigned mul = MIN2(tile_size.z, 4);
            tile_size.z /= mul;
            tile_size.x *= mul;
         }
      }
   }

   const unsigned dst_element_size = d.dst_bpe * d.dst_samples;
   assert(bytes_per_lane >= dst_element_size);

   /* Determine the tile shape of each lane. */
   unsigned num_pixels = DIV_ROUND_UP(bytes_per_lane, dst_element_size);
   uvec3 lane_tile = (uvec3){1, 1, 1};
   uvec3 remaining = tile_size;

   while (num_pixels > 1) {
      /* Try to keep the lane tile equal in all dimensions, favoring Z, then Y, then X, but stay
       * within the tile size.
       */
      if (remaining.z > 1) {
         lane_tile.z *= 2;
         remaining.z /= 2;
         num_pixels /= 2;
      }

      if (num_pixels > 1 && remaining.y > 1) {
         lane_tile.y *= 2;
         remaining.y /= 2;
         num_pixels /= 2;
      }

      if (num_pixels > 1 && remaining.x > 1) {
         lane_tile.x *= 2;
         remaining.x /= 2;
         num_pixels /= 2;
      }
   }

   assert(num_pixels == 1);
   assert(lane_tile.x <= tile_size.x);
   assert(lane_tile.y <= tile_size.y);
   assert(lane_tile.z <= tile_size.z);

   assert(tile_size.x % lane_tile.x == 0);
   assert(tile_size.y % lane_tile.y == 0);
   assert(tile_size.z % lane_tile.z == 0);

   /* Check that the lane tile fits into the shader key. */
   static const ac_cs_blit_key max_lane_shape = {
      .log_lane_width = ~0,
      .log_lane_height = ~0,
      .log_lane_depth = ~0,
   };
   assert(util_logbase2(lane_tile.x) <= max_lane_shape.log_lane_width);
   assert(util_logbase2(lane_tile.y) <= max_lane_shape.log_lane_height);
   assert(util_logbase2(lane_tile.z) <= max_lane_shape.log_lane_depth);

   /* If the shader blits a block of pixels per lane, it must have the dst box aligned to that
    * block because it can't blit a subset of pixels per lane.
    *
    * If the blit dst box is not aligned to the lane tile, split it into multiple blits by cutting
    * off the unaligned sides of the box and blitting the middle that's aligned to the lane tile,
    * then blit the unaligned sides separately. This splits the blit into up to 7 blits for 3D,
    * and 5 blits for 2D.
    */
   if (blit->dst.box.x % lane_tile.x ||
       blit->dst.box.y % lane_tile.y ||
       blit->dst.box.z % lane_tile.z ||
       blit->dst.box.width % lane_tile.x ||
       blit->dst.box.height % lane_tile.y ||
       blit->dst.box.depth % lane_tile.z) {
      struct pipe_box middle;

      /* Cut off unaligned regions on the sides of the box. */
      middle.x = ALIGN_POT(blit->dst.box.x, lane_tile.x);
      middle.y = ALIGN_POT(blit->dst.box.y, lane_tile.y);
      middle.z = ALIGN_POT(blit->dst.box.z, lane_tile.z);

      middle.width = blit->dst.box.width - (middle.x - blit->dst.box.x);
      if (middle.width > 0)
         middle.width -= middle.width % lane_tile.x;
      middle.height = blit->dst.box.height - (middle.y - blit->dst.box.y);
      if (middle.height > 0)
         middle.height -= middle.height % lane_tile.y;
      middle.depth = blit->dst.box.depth - (middle.z - blit->dst.box.z);
      if (middle.depth > 0)
         middle.depth -= middle.depth % lane_tile.z;

      /* Only a few cases are regressed by this. The vast majority benefits a lot.
       * This was fine-tuned for Navi31, and might be suboptimal on different generations.
       */
      /* TODO: Retune this with RADV to improve perf of partial/unaligned ops. */
      bool slow = (d.dst_tiling_dim == 1 && d.src_tiling_dim == 1 &&
                   blit->dst.box.depth > 1) ||
                  (d.dst_tiling_dim == 3 &&
                   ((d.dst_bpe == 8 && d.is_clear) ||
                    (d.dst_bpe == 4 && d.src_tiling_dim == 1))) ||
                  (d.dst_tiling_dim != 3 &&
                   ((d.dst_bpe == 4 && d.dst_tiling_dim == 1 && d.src_tiling_dim == 1) ||
                    (d.dst_bpe == 8 && !d.is_clear && (d.dst_tiling_dim == 1) != (d.src_tiling_dim == 1)) ||
                    (d.is_resolve && d.dst_bpe == 4 && d.src_samples == 4) ||
                    (d.is_resolve && d.dst_bpe == 8 && d.src_samples == 2)));

      /* Only use this if the middle blit is large enough. */
      if (!slow && middle.width > 0 && middle.height > 0 && middle.depth > 0 &&
          (uint64_t)middle.width * middle.height * middle.depth * d.dst_bpe * d.dst_samples >
          128 * 1024) {
         /* Compute the size of unaligned regions on all sides of the box. */
         struct pipe_box top, left, right, bottom, front, back;

         assert(!options->is_nested);

         top = blit->dst.box;
         top.height = middle.y - top.y;

         bottom = blit->dst.box;
         bottom.y = middle.y + middle.height;
         bottom.height = blit->dst.box.height - top.height - middle.height;

         left = blit->dst.box;
         left.y = middle.y;
         left.height = middle.height;
         left.width = middle.x - left.x;

         right = blit->dst.box;
         right.y = middle.y;
         right.height = middle.height;
         right.x = middle.x + middle.width;
         right.width = blit->dst.box.width - left.width - middle.width;

         front = blit->dst.box;
         front.x = middle.x;
         front.y = middle.y;
         front.width = middle.width;
         front.height = middle.height;
         front.depth = middle.z - front.z;

         back = blit->dst.box;
         back.x = middle.x;
         back.y = middle.y;
         back.width = middle.width;
         back.height = middle.height;
         back.z = middle.z + middle.depth;
         back.depth = blit->dst.box.depth - front.depth - middle.depth;

         struct pipe_box boxes[] = {middle, top, bottom, left, right, front, back};

         /* Verify that the boxes don't intersect. */
         for (unsigned i = 0; i < ARRAY_SIZE(boxes); i++) {
            for (unsigned j = i + 1; j < ARRAY_SIZE(boxes); j++) {
               if (boxes[i].width > 0 && boxes[i].height > 0 && boxes[i].depth > 0 &&
                   boxes[j].width > 0 && boxes[j].height > 0 && boxes[j].depth > 0) {
                  if (u_box_test_intersection_3d(&boxes[i], &boxes[j])) {
                     printf("\b   (%u, %u, %u) -> (%u, %u, %u) | (%u, %u, %u) -> (%u, %u, %u)\n",
                            boxes[i].x, boxes[i].y, boxes[i].z,
                            boxes[i].x + boxes[i].width - 1,
                            boxes[i].y + boxes[i].height - 1,
                            boxes[i].z + boxes[i].depth - 1,
                            boxes[j].x, boxes[j].y, boxes[j].z,
                            boxes[j].x + boxes[j].width,
                            boxes[j].y + boxes[j].height,
                            boxes[j].z + boxes[j].depth);
                     assert(0);
                  }
               }
            }
         }

         ac_cs_blit_options nested_options = *options;
         nested_options.is_nested = true;

         for (unsigned i = 0; i < ARRAY_SIZE(boxes); i++) {
            if (boxes[i].width > 0 && boxes[i].height > 0 && boxes[i].depth > 0) {
               ac_cs_blit_description new_blit;
               ASSERTED bool ok;

               set_trimmed_blit(blit, &boxes[i], d.is_clear, &new_blit);
               ok = ac_prepare_compute_blit(&nested_options, &new_blit, out);
               assert(ok);
            }
         }
         return true;
      }
   }

   unsigned width = blit->dst.box.width;
   unsigned height = blit->dst.box.height;
   unsigned depth = blit->dst.box.depth;

   /* If the box can't be split, at least reduce the lane tile to the alignment of the box. */
   lane_tile.x = MIN3(lane_tile.x, compute_alignment(blit->dst.box.x), compute_alignment(width));
   lane_tile.y = MIN3(lane_tile.y, compute_alignment(blit->dst.box.y), compute_alignment(height));
   lane_tile.z = MIN3(lane_tile.z, compute_alignment(blit->dst.box.z), compute_alignment(depth));

   /* If the copy extents are below the tile size, reduce the tile size to the next power of two of
    * the copy extents.
    */
   if (width < tile_size.x)
      tile_size.x = util_next_power_of_two(width);
   if (height < tile_size.y)
      tile_size.y = util_next_power_of_two(height);
   if (depth < tile_size.z)
      tile_size.z = util_next_power_of_two(depth);

   unsigned start_x, start_y, start_z;

   /* If the blit destination area is unaligned, launch extra threads before 0,0,0 to make it
    * aligned. This makes sure that a wave doesn't straddle a DCC block boundary or a cache line
    * unnecessarily, so that each cache line is only stored by exactly 1 CU. The shader will skip
    * the extra threads. This makes unaligned compute blits faster.
    */
   start_x = blit->dst.box.x % tile_size.x;
   start_y = blit->dst.box.y % tile_size.y;
   start_z = blit->dst.box.z % tile_size.z;
   width += start_x;
   height += start_y;
   depth += start_z;

   /* Divide by the dispatch parameters by the lane tile. */
   assert(start_x % lane_tile.x == 0);
   assert(start_y % lane_tile.y == 0);
   assert(start_z % lane_tile.z == 0);
   assert(width % lane_tile.x == 0);
   assert(height % lane_tile.y == 0);
   assert(depth % lane_tile.z == 0);

   start_x /= lane_tile.x;
   start_y /= lane_tile.y;
   start_z /= lane_tile.z;
   width /= lane_tile.x;
   height /= lane_tile.y;
   depth /= lane_tile.z;

   /* Determine the workgroup dimensions for 64 invocations. */
   uvec3 wave_dim = {
      tile_size.x / lane_tile.x,
      tile_size.y / lane_tile.y,
      tile_size.z / lane_tile.z,
   };

   assert(util_is_power_of_two_nonzero(wave_dim.x));
   assert(util_is_power_of_two_nonzero(wave_dim.y));
   assert(util_is_power_of_two_nonzero(wave_dim.z));

   unsigned num_waves = wave_dim.x * wave_dim.y * wave_dim.z / 64;

   if (num_waves > 1) {
      /* If the workgroup size is more than 64 invocations, reduce it. */
      while (num_waves > 1) {
         if (wave_dim.z > 1) {
            wave_dim.z /= 2;
            num_waves /= 2;
         }

         if (wave_dim.y > 1 && num_waves > 1) {
            wave_dim.y /= 2;
            num_waves /= 2;
         }

         if (wave_dim.x > 1 && num_waves > 1) {
            wave_dim.x /= 2;
            num_waves /= 2;
         }
      }
   } else if (!num_waves) {
      /* If the workgroup size is less than 64 invocations, make it larger. */
      unsigned num_blocks_per_wave = 64 / (wave_dim.x * wave_dim.y * wave_dim.z);

      /* Try to enlarge it in the dimensions where we have enough pixels to process to minimize
       * no-op invocations.
       */
      int mul_threshold = 4;

      while (num_blocks_per_wave > 1) {
         const unsigned old_num_blocks = num_blocks_per_wave;

         /* The order of these three conditionals affects the tile shape of the wave, which
          * hugely impacts performance. Ideally the tile shape would converge to a square.
          */
         if (depth > wave_dim.z * mul_threshold && !blit->format_is_96bit) {
            wave_dim.z *= 2;
            num_blocks_per_wave /= 2;
         }

         if (height > wave_dim.y * mul_threshold && num_blocks_per_wave > 1) {
            wave_dim.y *= 2;
            num_blocks_per_wave /= 2;
         }

         if (width > wave_dim.x * mul_threshold && num_blocks_per_wave > 1) {
            wave_dim.x *= 2;
            num_blocks_per_wave /= 2;
         }

         if (old_num_blocks == num_blocks_per_wave) {
            /* No progress. Reduce the threshold. */
            mul_threshold /= 2;
         }
      }
   }

   assert(wave_dim.x * wave_dim.y * wave_dim.z == 64);

   /* Discard empty workgroups due to the start_x/y/z shift. */
   width -= start_x - start_x % wave_dim.x;
   height -= start_y - start_y % wave_dim.y;
   depth -= start_z - start_z % wave_dim.z;

   start_x %= wave_dim.x;
   start_y %= wave_dim.y;
   start_z %= wave_dim.z;

   /* Set the shader key and the dispatch info. */
   unsigned log_block_x = util_logbase2(wave_dim.x);
   unsigned log_block_y = util_logbase2(wave_dim.y);
   unsigned log_block_z = util_logbase2(wave_dim.z);

   unsigned index = out->num_dispatches++;
   assert(index < ARRAY_SIZE(out->dispatches));
   ac_cs_blit_dispatch *dispatch = &out->dispatches[index];
   unsigned wg_dim = set_work_size(dispatch, wave_dim.x, wave_dim.y, wave_dim.z, width, height, depth);

   const unsigned dst_dword_stride =
      ac_surface_get_plane_stride(options->info->gfx_level, blit->dst.surf, 0, 0) / 4;
   const unsigned src_dword_stride =
      d.is_clear ? 0 : ac_surface_get_plane_stride(options->info->gfx_level, blit->src.surf, 0, 0) / 4;

   /* Get the shader key. */
   ac_cs_blit_key key;
   key.key = 0;

   /* Only ACO can form VMEM clauses for image stores, which is a requirement for performance. */
   key.use_aco = true;
   key.is_clear = d.is_clear;
   key.wg_dim = wg_dim;
   key.has_start_xyz = start_x || start_y || start_z;
   key.format_is_96bit = blit->format_is_96bit;
   key.addr_math_64bit = key.format_is_96bit &&
                         ((uint64_t)blit->dst.box.y + blit->dst.box.height) *
                         dst_dword_stride * 4 > UINT32_MAX;
   key.log_lane_width = util_logbase2(lane_tile.x);
   key.log_lane_height = util_logbase2(lane_tile.y);
   key.log_lane_depth = util_logbase2(lane_tile.z);
   key.dst_is_1d = blit->dst.dim == 1;
   key.dst_is_msaa = d.dst_samples > 1;
   key.dst_has_z = blit->dst.dim == 3 || blit->dst.is_array;
   key.dst_store_fmask = blit->dst_store_fmask;
   key.copy_physical_samples = blit->copy_physical_samples;
   key.dst_is_rgb5 = d.max_dst_chan_size <= 6 &&
                     !util_format_is_pure_integer(blit->dst.format);
   key.last_dst_channel = key.format_is_96bit ? 0 : util_format_get_last_component(blit->dst.format);

   bool has_d16 = info->gfx_level >= GFX8 && !key.dst_is_rgb5;

   if (d.is_clear) {
      assert(d.dst_samples <= 8);
      key.log_samples = d.sample0_only ? 0 : util_logbase2(d.dst_samples);
      key.a16 = !key.format_is_96bit && info->gfx_level >= GFX9 && util_is_box_sint16(&blit->dst.box);
      key.d16 = has_d16 &&
                d.max_dst_chan_size <= (util_format_is_float(blit->dst.format) ||
                                      util_format_is_pure_integer(blit->dst.format) ? 16 : 11);
      key.sample0_only = d.sample0_only;
   } else {
      /* src.box.height is clamped to 0 because it's negative and src.box.y contains the upper
       * bound when the Y axis is flipped.
       */
      key.addr_math_64bit |= key.format_is_96bit &&
                             ((uint64_t)blit->src.box.y + MAX2(blit->src.box.height, 0)) *
                             src_dword_stride * 4 > UINT32_MAX;
      key.src_is_sampler = blit->src_is_sampler;
      key.src_is_1d = blit->src.dim == 1;
      key.src_is_msaa = d.src_samples > 1;
      key.src_has_z = blit->src.dim == 3 || blit->src.is_array;
      key.src_has_non_identity_fmask = d.src_samples > 1 && blit->src_has_non_identity_fmask;
      /* Resolving integer formats only copies sample 0. log_samples is then unused. */
      key.sample0_only = d.sample0_only;
      unsigned num_samples = MAX2(d.src_samples, d.dst_samples);
      assert(num_samples <= 8);
      key.log_samples = d.sample0_only ? 0 : util_logbase2(num_samples);
      key.x_clamp_to_edge = should_blit_clamp_to_edge(blit, BITFIELD_BIT(0));
      key.y_clamp_to_edge = should_blit_clamp_to_edge(blit, BITFIELD_BIT(1));
      key.flip_x = blit->src.box.width < 0;
      key.flip_y = blit->src.box.height < 0;
      key.sint_to_uint = util_format_is_pure_sint(blit->src.format) &&
                         util_format_is_pure_uint(blit->dst.format);
      key.uint_to_sint = util_format_is_pure_uint(blit->src.format) &&
                         util_format_is_pure_sint(blit->dst.format);
      key.dst_is_srgb = util_format_is_srgb(blit->dst.format);
      key.last_src_channel = key.format_is_96bit ? 0 :
                                 MIN2(util_format_get_last_component(blit->src.format),
                                      key.last_dst_channel);
      key.use_integer_one = util_format_is_pure_integer(blit->dst.format) &&
                            key.last_src_channel < key.last_dst_channel &&
                            key.last_dst_channel == 3;
      key.a16 = !key.format_is_96bit && info->gfx_level >= GFX9 &&
                util_is_box_sint16(&blit->dst.box) && util_is_box_sint16(&blit->src.box);
      key.d16 = has_d16 &&
                /* Blitting FP16 using D16 has precision issues. Resolving has precision
                 * issues all the way down to R11G11B10_FLOAT. */
                MIN2(d.max_dst_chan_size, d.max_src_chan_size) <=
                (util_format_is_pure_integer(blit->dst.format) ?
                    (key.sint_to_uint || key.uint_to_sint ? 10 : 16) :
                    (d.is_resolve ? 10 : 11));
   }

   dispatch->shader_key = key;

   assert(key.format_is_96bit ? !(blit->src.box.x & ~0x3ffff) : util_is_uint16(blit->src.box.x));
   assert(util_is_uint16(blit->src.box.y));
   assert(util_is_uint16(blit->src.box.z));
   assert(key.format_is_96bit ? !(blit->dst.box.x & ~0x3ffff) : util_is_uint16(blit->dst.box.x));
   assert(util_is_uint16(blit->dst.box.y));
   assert(util_is_uint16(blit->dst.box.z));
   assert(start_x <= 63 && start_y <= 63 && start_z <= 63);
   assert(log_block_x <= 7 && log_block_y <= 7 && log_block_z <= 7);
   assert(key.dst_has_z || !blit->dst.box.z);
   assert(key.src_has_z || !blit->src.box.z);

   /* Set user data SGPR values. */
   unsigned num = 0;

   if (blit->format_is_96bit) {
      assert(!key.a16);
      assert(!key.d16);
      assert(!key.x_clamp_to_edge);
      assert(!key.y_clamp_to_edge);
      assert(!key.log_samples);
      assert(!(dst_dword_stride & ~0x3ffff));
      assert(!(src_dword_stride & ~0x3ffff));
      assert(start_z == 0);
      assert(log_block_z == 0);
      assert(blit->dst.box.z == 0);
      assert(blit->src.box.z == 0);

      dispatch->user_data[num++] = (dst_dword_stride & 0x3ffff) |
                                   ((start_x & 0x3f) << 18) | ((start_y & 0x3f) << 24);
      dispatch->user_data[num++] = (blit->dst.box.x & 0x3ffff) |
                                   ((log_block_x & 0x7) << 18) | ((log_block_y & 0x7) << 21);
      dispatch->user_data[num++] = (blit->src.box.y & 0xffff) | (((uint32_t)blit->dst.box.y & 0xffff) << 16);

      if (!d.is_clear) {
         dispatch->user_data[num++] = blit->src.box.x;
         dispatch->user_data[num++] = src_dword_stride;
      } else {
         memcpy(&dispatch->user_data[num], &blit->clear_color, 3 * sizeof(uint32_t));
         num += 3;
      }
   } else {
      dispatch->user_data[num++] = (start_x & 0x3f) | ((start_y & 0x3f) << 6) | ((start_z & 0x3f) << 12) |
                                   ((log_block_x & 0x7) << 18) | ((log_block_y & 0x7) << 21) |
                                   ((log_block_z & 0x7) << 24);
      dispatch->user_data[num++] = (blit->dst.box.x & 0xffff) | ((uint32_t)(blit->dst.box.y & 0xffff) << 16);

      if (key.dst_has_z || key.src_has_z)
         dispatch->user_data[num++] = (blit->dst.box.z & 0xffff) | ((uint32_t)(blit->src.box.z & 0xffff) << 16);

      if (!d.is_clear) {
         dispatch->user_data[num++] = (blit->src.box.x & 0xffff) | ((uint32_t)(blit->src.box.y & 0xffff) << 16);
      } else {
         union pipe_color_union final_value;
         memcpy(&final_value, &blit->clear_color, sizeof(final_value));

         /* Do the conversion to sRGB here instead of the shader. */
         if (util_format_is_srgb(blit->dst.format)) {
            for (int i = 0; i < 3; i++)
               final_value.f[i] = util_format_linear_to_srgb_float(final_value.f[i]);
         }

         if (key.d16) {
            enum pipe_format data_format;

            if (util_format_is_pure_uint(blit->dst.format))
               data_format = PIPE_FORMAT_R16G16B16A16_UINT;
            else if (util_format_is_pure_sint(blit->dst.format))
               data_format = PIPE_FORMAT_R16G16B16A16_SINT;
            else
               data_format = PIPE_FORMAT_R16G16B16A16_FLOAT;

            util_pack_color_union(data_format, (union util_color *)&dispatch->user_data[num],
                                  &final_value);
            num += DIV_ROUND_UP(key.last_dst_channel + 1, 2);
         } else {
            memcpy(&dispatch->user_data[num], &final_value, sizeof(final_value));
            num += key.last_dst_channel + 1;
         }
      }
   }

   assert(get_num_user_data_terms(&dispatch->shader_key) == num);
   dispatch->num_user_data_terms = num;
   return true;
}
