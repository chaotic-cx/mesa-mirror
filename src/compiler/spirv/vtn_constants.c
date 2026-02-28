/*
 * Copyright © 2015 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "vtn_private.h"

#include "nir/nir_constant_expressions.h"

#include "util/bfloat.h"
#include "util/float8.h"

nir_constant *
vtn_null_constant(struct vtn_builder *b, struct vtn_type *type)
{
   nir_constant *c = rzalloc(b, nir_constant);

   switch (type->base_type) {
   case vtn_base_type_scalar:
   case vtn_base_type_vector:
   case vtn_base_type_cooperative_matrix:
      c->is_null_constant = true;
      /* Nothing to do here.  It's already initialized to zero */
      break;

   case vtn_base_type_pointer: {
      enum vtn_variable_mode mode = vtn_storage_class_to_mode(
         b, type->storage_class, type->pointed, NULL);
      nir_address_format addr_format = vtn_mode_to_address_format(b, mode);

      const nir_const_value *null_value = nir_address_format_null_value(addr_format);
      memcpy(c->values, null_value,
             sizeof(nir_const_value) * nir_address_format_num_components(addr_format));
      break;
   }

   case vtn_base_type_void:
   case vtn_base_type_image:
   case vtn_base_type_sampler:
   case vtn_base_type_sampled_image:
   case vtn_base_type_function:
   case vtn_base_type_event:
      /* For those we have to return something but it doesn't matter what. */
      break;

   case vtn_base_type_matrix:
   case vtn_base_type_array:
      vtn_assert(type->length > 0);
      c->is_null_constant = true;
      c->num_elements = type->length;
      c->elements = ralloc_array(b, nir_constant *, c->num_elements);

      c->elements[0] = vtn_null_constant(b, type->array_element);
      for (unsigned i = 1; i < c->num_elements; i++)
         c->elements[i] = c->elements[0];
      break;

   case vtn_base_type_struct:
      c->is_null_constant = true;
      c->num_elements = type->length;
      if (c->num_elements) {
         c->elements = ralloc_array(b, nir_constant *, c->num_elements);
         for (unsigned i = 0; i < c->num_elements; i++)
            c->elements[i] = vtn_null_constant(b, type->members[i]);
      }
      break;

   default:
      vtn_fail("Invalid type for null constant");
   }

   return c;
}

static void
spec_constant_decoration_cb(struct vtn_builder *b, UNUSED struct vtn_value *val,
                            ASSERTED int member,
                            const struct vtn_decoration *dec, void *data)
{
   vtn_assert(member == -1);
   if (dec->decoration != SpvDecorationSpecId)
      return;

   if (!b->specialization)
      return;

   for (unsigned i = 0; i < b->specialization->num_entries; i++) {
      const struct nir_spirv_specialization_entry *entry =
         &b->specialization->entries[i];

      if (entry->id == dec->operands[0] && entry->size > 0) {
         memcpy(data, entry->data, entry->size);
         return;
      }
   }
}

struct spec_constant_data_cb_data {
   nir_constant **elems;
   unsigned num_elems;
   unsigned elem_byte_size;
};

static void
spec_constant_data_decoration_cb(struct vtn_builder *b, UNUSED struct vtn_value *val,
                                 ASSERTED int member,
                                 const struct vtn_decoration *dec, void *data)
{
   vtn_assert(member == -1);
   if (dec->decoration != SpvDecorationSpecId)
      return;

   if (!b->specialization)
      return;

   const struct spec_constant_data_cb_data *ctx = data;

   for (unsigned i = 0; i < b->specialization->num_entries; i++) {
      const struct nir_spirv_specialization_entry *entry =
         &b->specialization->entries[i];

      if (entry->id != dec->operands[0])
         continue;

      if (entry->size == 0)
         continue;

      for (unsigned e = 0; e < ctx->num_elems; e++) {
         unsigned offset = e * ctx->elem_byte_size;
         assert(offset + ctx->elem_byte_size <= entry->size);
         memcpy(&ctx->elems[e]->values[0], entry->data + offset, ctx->elem_byte_size);
      }
      return;
   }
}

static void
handle_workgroup_size_decoration_cb(struct vtn_builder *b,
                                    struct vtn_value *val,
                                    ASSERTED int member,
                                    const struct vtn_decoration *dec,
                                    UNUSED void *data)
{
   vtn_assert(member == -1);
   if (dec->decoration != SpvDecorationBuiltIn ||
       dec->operands[0] != SpvBuiltInWorkgroupSize)
      return;

   vtn_assert(val->type->type == glsl_vector_type(GLSL_TYPE_UINT, 3));
   b->workgroup_size_builtin = val;
}

void
vtn_handle_constants(struct vtn_builder *b, SpvOp opcode,
                     const uint32_t *w, unsigned count)
{
   struct vtn_value *val = vtn_push_value(b, w[2], vtn_value_type_constant);
   val->constant = rzalloc(b, nir_constant);
   switch (opcode) {
   case SpvOpConstantTrue:
   case SpvOpConstantFalse:
   case SpvOpSpecConstantTrue:
   case SpvOpSpecConstantFalse: {
      vtn_fail_if(val->type->type != glsl_bool_type(),
                  "Result type of %s must be OpTypeBool",
                  spirv_op_to_string(opcode));

      bool bval = (opcode == SpvOpConstantTrue ||
                   opcode == SpvOpSpecConstantTrue);

      nir_const_value u32val = nir_const_value_for_uint(bval, 32);

      if (opcode == SpvOpSpecConstantTrue ||
          opcode == SpvOpSpecConstantFalse)
         vtn_foreach_decoration(b, val, spec_constant_decoration_cb, &u32val);

      val->constant->values[0].b = u32val.u32 != 0;
      break;
   }

   case SpvOpConstant:
   case SpvOpSpecConstant: {
      vtn_fail_if(val->type->base_type != vtn_base_type_scalar,
                  "Result type of %s must be a scalar",
                  spirv_op_to_string(opcode));
      int bit_size = glsl_get_bit_size(val->type->type);
      switch (bit_size) {
      case 64:
         val->constant->values[0].u64 = vtn_u64_literal(&w[3]);
         break;
      case 32:
         val->constant->values[0].u32 = w[3];
         break;
      case 16:
         val->constant->values[0].u16 = w[3];
         break;
      case 8:
         val->constant->values[0].u8 = w[3];
         break;
      default:
         vtn_fail("Unsupported SpvOpConstant bit size: %u", bit_size);
      }

      if (opcode == SpvOpSpecConstant)
         vtn_foreach_decoration(b, val, spec_constant_decoration_cb,
                                &val->constant->values[0]);
      break;
   }

   case SpvOpSpecConstantComposite:
   case SpvOpConstantComposite:
   case SpvOpConstantCompositeReplicateEXT:
   case SpvOpSpecConstantCompositeReplicateEXT: {
      const unsigned elem_count =
         val->type->base_type == vtn_base_type_cooperative_matrix ?
         1 : val->type->length;

      nir_constant **elems = ralloc_array(b, nir_constant *, elem_count);
      if (opcode == SpvOpConstantCompositeReplicateEXT ||
          opcode == SpvOpSpecConstantCompositeReplicateEXT) {
         struct vtn_value *elem_val = vtn_untyped_value(b, w[3]);

         if (elem_val->value_type == vtn_value_type_constant) {
            elems[0] = elem_val->constant;
            val->is_undef_constant = false;
         } else {
            vtn_fail_if(elem_val->value_type != vtn_value_type_undef,
                        "only constants or undefs allowed for %s",
                        spirv_op_to_string(opcode));
            /* to make it easier, just insert a NULL constant for now */
            elems[0] = vtn_null_constant(b, elem_val->type);
            val->is_undef_constant = true;
         }

         for (unsigned i = 1; i < elem_count; i++)
            elems[i] = elems[0];
      } else {
         vtn_fail_if(elem_count != count - 3,
                     "%s has %u constituents, expected %u",
                     spirv_op_to_string(opcode), count - 3, elem_count);

         val->is_undef_constant = true;
         for (unsigned i = 0; i < elem_count; i++) {
            struct vtn_value *elem_val = vtn_untyped_value(b, w[i + 3]);

            if (elem_val->value_type == vtn_value_type_constant) {
               elems[i] = elem_val->constant;
               val->is_undef_constant = val->is_undef_constant &&
                                        elem_val->is_undef_constant;
            } else {
               vtn_fail_if(elem_val->value_type != vtn_value_type_undef,
                           "only constants or undefs allowed for %s",
                           spirv_op_to_string(opcode));
               /* to make it easier, just insert a NULL constant for now */
               elems[i] = vtn_null_constant(b, elem_val->type);
            }
         }
      }

      switch (val->type->base_type) {
      case vtn_base_type_vector: {
         assert(glsl_type_is_vector(val->type->type));
         for (unsigned i = 0; i < elem_count; i++)
            val->constant->values[i] = elems[i]->values[0];
         break;
      }

      case vtn_base_type_matrix:
      case vtn_base_type_struct:
      case vtn_base_type_array:
         ralloc_steal(val->constant, elems);
         val->constant->num_elements = elem_count;
         val->constant->elements = elems;
         break;

      case vtn_base_type_cooperative_matrix:
         val->constant->values[0] = elems[0]->values[0];
         break;

      default:
         vtn_fail("Result type of %s must be a composite type",
                  spirv_op_to_string(opcode));
      }
      break;
   }

   case SpvOpSpecConstantOp: {
      nir_const_value u32op = nir_const_value_for_uint(w[3], 32);
      vtn_foreach_decoration(b, val, spec_constant_decoration_cb, &u32op);
      SpvOp opcode = u32op.u32;
      switch (opcode) {
      case SpvOpBitcast: {
         struct vtn_value *src = &b->values[w[4]];

         vtn_assert(src->value_type == vtn_value_type_constant ||
                    src->value_type == vtn_value_type_undef);

         unsigned src_len = glsl_get_vector_elements(src->type->type);
         unsigned dst_len = glsl_get_vector_elements(val->type->type);

         unsigned src_bit_size = glsl_get_bit_size(src->type->type);
         unsigned dst_bit_size = glsl_get_bit_size(val->type->type);

         vtn_assert(src_len * src_bit_size == dst_len * dst_bit_size);

         /* This will end up being zero */
         if (src->value_type == vtn_value_type_undef)
            break;

         if (src_bit_size == dst_bit_size) {
            /* This is just a copy */
            for (unsigned i = 0; i < src_len; i++)
               val->constant->values[i] = src->constant->values[i];
         } else {
            /* You can't non-trivially bitcast booleans */
            vtn_assert(src_bit_size >= 8 && dst_bit_size >= 8);
            const unsigned src_byte_size = src_bit_size / 8;
            const unsigned dst_byte_size = dst_bit_size / 8;

            vtn_assert(src_len <= NIR_MAX_VEC_COMPONENTS &&
                       dst_len <= NIR_MAX_VEC_COMPONENTS);

            uint8_t bits[NIR_MAX_VEC_COMPONENTS * sizeof(nir_const_value)];

            for (unsigned i = 0; i < src_len; i++) {
               uint64_t v = nir_const_value_as_int(src->constant->values[i],
                                                   src_bit_size);
               memcpy(bits + i * src_byte_size, &v, src_byte_size);
            }

            for (unsigned i = 0; i < dst_len; i++) {
               uint64_t v = 0;
               memcpy(&v, bits + i * dst_byte_size, dst_byte_size);
               val->constant->values[i] =
                  nir_const_value_for_uint(v, dst_bit_size);
            }
         }
         break;
      }
      case SpvOpVectorShuffle: {
         struct vtn_value *v0 = &b->values[w[4]];
         struct vtn_value *v1 = &b->values[w[5]];

         vtn_assert(v0->value_type == vtn_value_type_constant ||
                    v0->value_type == vtn_value_type_undef);
         vtn_assert(v1->value_type == vtn_value_type_constant ||
                    v1->value_type == vtn_value_type_undef);

         unsigned len0 = glsl_get_vector_elements(v0->type->type);
         unsigned len1 = glsl_get_vector_elements(v1->type->type);

         vtn_assert(len0 + len1 < 16);

         unsigned bit_size = glsl_get_bit_size(val->type->type);
         unsigned bit_size0 = glsl_get_bit_size(v0->type->type);
         unsigned bit_size1 = glsl_get_bit_size(v1->type->type);

         vtn_assert(bit_size == bit_size0 && bit_size == bit_size1);
         (void)bit_size0; (void)bit_size1;

         nir_const_value undef =
            nir_const_value_for_raw_uint(0xdeadbeefdeadbeef, bit_size);
         nir_const_value combined[NIR_MAX_VEC_COMPONENTS * 2];

         if (v0->value_type == vtn_value_type_constant) {
            for (unsigned i = 0; i < len0; i++)
               combined[i] = v0->constant->values[i];
         }
         if (v1->value_type == vtn_value_type_constant) {
            for (unsigned i = 0; i < len1; i++)
               combined[len0 + i] = v1->constant->values[i];
         }

         for (unsigned i = 0, j = 0; i < count - 6; i++, j++) {
            uint32_t comp = w[i + 6];
            if (comp == (uint32_t)-1) {
               /* If component is not used, set the value to a known constant
                * to detect if it is wrongly used.
                */
               val->constant->values[j] = undef;
            } else {
               vtn_fail_if(comp >= len0 + len1,
                           "All Component literals must either be FFFFFFFF "
                           "or in [0, N - 1] (inclusive).");
               val->constant->values[j] = combined[comp];
            }
         }
         break;
      }

      case SpvOpCompositeExtract:
      case SpvOpCompositeInsert: {
         struct vtn_value *comp;
         unsigned deref_start;
         struct nir_constant **c;
         if (opcode == SpvOpCompositeExtract) {
            comp = vtn_value(b, w[4], vtn_value_type_constant);
            deref_start = 5;
            c = &comp->constant;
         } else {
            comp = vtn_value(b, w[5], vtn_value_type_constant);
            deref_start = 6;
            val->constant = nir_constant_clone(comp->constant, b->shader);
            c = &val->constant;
         }

         int elem = -1;
         const struct vtn_type *type = comp->type;
         for (unsigned i = deref_start; i < count; i++) {
            if (type->base_type == vtn_base_type_cooperative_matrix) {
               /* Cooperative matrices are always scalar constants.  We don't
                * care about the index w[i] because it's always replicated.
                */
               type = type->component_type;
            } else {
               vtn_fail_if(w[i] >= type->length,
                           "%uth index of %s is %u but the type has only "
                           "%u elements", i - deref_start,
                           spirv_op_to_string(opcode), w[i], type->length);

               switch (type->base_type) {
               case vtn_base_type_vector:
                  elem = w[i];
                  type = type->array_element;
                  break;

               case vtn_base_type_matrix:
               case vtn_base_type_array:
                  c = &(*c)->elements[w[i]];
                  type = type->array_element;
                  break;

               case vtn_base_type_struct:
                  c = &(*c)->elements[w[i]];
                  type = type->members[w[i]];
                  break;

               default:
                  vtn_fail("%s must only index into composite types",
                           spirv_op_to_string(opcode));
               }
            }
         }

         if (opcode == SpvOpCompositeExtract) {
            if (elem == -1) {
               val->constant = *c;
            } else {
               unsigned num_components = type->length;
               for (unsigned i = 0; i < num_components; i++)
                  val->constant->values[i] = (*c)->values[elem + i];
            }
         } else {
            struct vtn_value *insert =
               vtn_value(b, w[4], vtn_value_type_constant);
            vtn_assert(insert->type == type);
            if (elem == -1) {
               *c = insert->constant;
            } else {
               unsigned num_components = type->length;
               for (unsigned i = 0; i < num_components; i++)
                  (*c)->values[elem + i] = insert->constant->values[i];
            }
         }
         break;
      }

      case SpvOpSelect: {
         struct vtn_value *cond = vtn_value(b, w[4], vtn_value_type_constant);
         vtn_fail_if(!glsl_type_is_boolean(cond->type->type),
                     "Condition of OpSelect must be a Boolean");

         if (glsl_type_is_scalar(cond->type->type)) {
            const uint32_t obj = cond->constant->values[0].b ? w[5] : w[6];
            val->constant = vtn_value(b, obj, vtn_value_type_constant)->constant;
         } else {
            vtn_fail_if(glsl_get_vector_elements(cond->type->type) !=
                        glsl_get_vector_elements(val->type->type),
                        "Vector Condition of OpSelect must have the same "
                        "number of components as the Result Type");

            nir_constant *c1 = vtn_value(b, w[5], vtn_value_type_constant)->constant;
            nir_constant *c2 = vtn_value(b, w[6], vtn_value_type_constant)->constant;

            unsigned num_components = glsl_get_vector_elements(val->type->type);
            for (unsigned i = 0; i < num_components; i++) {
               val->constant->values[i] = cond->constant->values[i].b ?
                  c1->values[i] : c2->values[i];
            }
         }
         break;
      }

      default: {
         const glsl_type *dst_type = val->type->type;

         const bool saturate = vtn_has_decoration(b, val, SpvDecorationSaturatedToLargestFloat8NormalConversionEXT);
         unsigned num_components = glsl_get_vector_elements(val->type->type);

         vtn_assert(count <= 7);

         const unsigned src_count = count - 4;
         struct vtn_value *src_val[3] = {0};
         const glsl_type *src_type[3] = {0};

         for (unsigned i = 0; i < src_count; i++) {
            src_val[i] = vtn_value(b, w[4 + i], vtn_value_type_constant);
            src_type[i] = src_val[i]->type->type;
         }

         const unsigned dst_bit_size =
            glsl_type_is_nonnative_float(dst_type) ? 32 : glsl_get_bit_size(dst_type);

         nir_op op;
         bool swap;

         switch (opcode) {
         case SpvOpConvertFToU:
         case SpvOpConvertFToS:
         case SpvOpConvertSToF:
         case SpvOpConvertUToF:
         case SpvOpSConvert:
         case SpvOpFConvert:
         case SpvOpUConvert: {
            /* We have a different source type in a conversion. */
            const unsigned conv_src_bit_size =
               glsl_type_is_nonnative_float(src_type[0]) ? 32 : glsl_get_bit_size(src_type[0]);

            nir_alu_type src_alu_type = vtn_convert_op_src_type(opcode) | conv_src_bit_size;
            nir_alu_type dst_alu_type = vtn_convert_op_dst_type(opcode) | dst_bit_size;
            op = nir_type_conversion_op(src_alu_type, dst_alu_type, nir_rounding_mode_undef);
            swap = false;
            break;
         }

         default: {
            unsigned extra_fp_math_ctrl;
            op = vtn_nir_alu_op_for_spirv_opcode(b, opcode, &swap, &extra_fp_math_ctrl);

            /* No SPIR-V opcodes handled through this path should set fast math.
             * Since it is ignored, assert on it.
             */
            assert(!extra_fp_math_ctrl);
            break;
         }
         };

         unsigned resolved_bit_size = dst_bit_size;

         nir_const_value src[3][NIR_MAX_VEC_COMPONENTS];

         for (unsigned i = 0; i < src_count; i++) {
            /* If this is an unsized source, pull the bit size from the
             * source; otherwise, we'll use the bit size from the destination.
             */
            if (!nir_alu_type_get_type_size(nir_op_infos[op].input_types[i])) {
               resolved_bit_size = glsl_type_is_nonnative_float(src_type[i]) ?
                  32 : glsl_get_bit_size(src_type[i]);
            }

            unsigned src_comps = nir_op_infos[op].input_sizes[i] ?
                                 nir_op_infos[op].input_sizes[i] :
                                 num_components;

            unsigned j = swap ? 1 - i : i;
            for (unsigned c = 0; c < src_comps; c++) {
               src[j][c] = src_val[i]->constant->values[c];
               if (glsl_type_is_bfloat_16(src_type[i]))
                  src[j][c].f32 = _mesa_bfloat16_bits_to_float(src[j][c].u16);
               else if (glsl_type_is_e4m3fn(src_type[i]))
                  src[j][c].f32 = _mesa_e4m3fn_to_float(src[j][c].u8);
               else if (glsl_type_is_e5m2(src_type[i]))
                  src[j][c].f32 = _mesa_e5m2_to_float(src[j][c].u8);
            }

            /* Fix up source to respect NIR expected sizes. */
            switch (op) {
            case nir_op_ishl:
            case nir_op_ishr:
            case nir_op_ushr: {
               /* Shift amount in NIR ops must be 32-bit. */
               vtn_assert(!swap);
               const unsigned shift_idx = 1;
               const unsigned shift_bit_size = glsl_get_bit_size(src_type[i]);
               if (i != shift_idx || shift_bit_size == 32)
                  break;
               for (unsigned c = 0; c < src_comps; c++) {
                  nir_const_value *shift = &src[shift_idx][c];
                  *shift = nir_const_value_for_uint(
                        nir_const_value_as_uint(*shift, shift_bit_size), 32);
               }
               break;
            }
            default:
               break;
            }
         }

         nir_const_value *srcs[3] = {
            src[0], src[1], src[2],
         };
         nir_eval_const_opcode(op, val->constant->values, NULL,
                               num_components, resolved_bit_size, srcs,
                               b->shader->info.float_controls_execution_mode);

         for (int i = 0; i < num_components; i++) {
            uint16_t conv;
            if (glsl_type_is_bfloat_16(dst_type)) {
               conv = _mesa_float_to_bfloat16_bits_rte(val->constant->values[i].f32);
            } else if (glsl_type_is_e4m3fn(dst_type)) {
               if (saturate)
                  conv = _mesa_float_to_e4m3fn_sat(val->constant->values[i].f32);
               else
                  conv = _mesa_float_to_e4m3fn(val->constant->values[i].f32);
            } else if (glsl_type_is_e5m2(dst_type)) {
               if (saturate)
                  conv = _mesa_float_to_e5m2_sat(val->constant->values[i].f32);
               else
                  conv = _mesa_float_to_e5m2(val->constant->values[i].f32);
            } else {
               continue;
            }

            val->constant->values[i] = nir_const_value_for_raw_uint(conv, glsl_get_bit_size(dst_type));
         }

         break;
      } /* default */
      }
      break;
   }

   case SpvOpConstantNull:
      val->constant = vtn_null_constant(b, val->type);
      val->is_null_constant = true;
      break;

   case SpvOpConstantSizeOfEXT: {
      vtn_fail_if(val->type->type != glsl_uint_type() &&
                  val->type->type != glsl_int_type() &&
                  val->type->type != glsl_uint64_t_type() &&
                  val->type->type != glsl_int64_t_type(),
                  "Result Type must be a 32-bit or 64-bit integer type scalar.");

      struct vtn_type *type = vtn_get_type(b, w[3]);
      switch (type->base_type) {
      case vtn_base_type_image:
         val->constant->values[0].u32 =
            align(b->options->image_descriptor_size,
                  b->options->image_descriptor_alignment);
         break;
      case vtn_base_type_sampler:
         val->constant->values[0].u32 =
            align(b->options->sampler_descriptor_size,
                  b->options->sampler_descriptor_alignment);
         break;
      case vtn_base_type_accel_struct:
      case vtn_base_type_buffer:
         val->constant->values[0].u32 =
            align(b->options->buffer_descriptor_size,
                  b->options->buffer_descriptor_alignment);
         break;
      default:
         vtn_fail("Type must be an OpTypeBufferKHR, OpTypeImage, "
                  "OpTypeAccelerationStructureKHR, OpTypeTensorARM, or "
                  "OpTypeSampler instruction.");
      }
      break;
   }

   case SpvOpConstantDataKHR:
   case SpvOpSpecConstantDataKHR: {
      struct vtn_type *elem_type = val->type->array_element;

      vtn_fail_if(val->type->base_type != vtn_base_type_array,
                  "Result type must be an array");
      vtn_fail_if(!glsl_type_is_integer(elem_type->type),
                  "Element type must be a scalar integer");

      unsigned num_elems = val->type->length;
      unsigned bit_size  = glsl_get_bit_size(elem_type->type);
      unsigned byte_size = bit_size / 8;

      const uint8_t *data = (const uint8_t *)&w[3];

      nir_constant **elems = ralloc_array(b, nir_constant *, num_elems);
      for (unsigned i = 0; i < num_elems; i++) {
         nir_constant *c = rzalloc(b, nir_constant);
         memcpy(&c->values[0], data + i * byte_size, byte_size);
         elems[i] = c;
      }

      if (opcode == SpvOpSpecConstantDataKHR) {
         struct spec_constant_data_cb_data ctx = { elems, num_elems, byte_size };
         vtn_foreach_decoration(b, val, spec_constant_data_decoration_cb, &ctx);
      }

      ralloc_steal(val->constant, elems);
      val->constant->num_elements = num_elems;
      val->constant->elements = elems;
      val->is_undef_constant = false;
      break;
   }

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }

   /* Now that we have the value, update the workgroup size if needed */
   if (mesa_shader_stage_uses_workgroup(b->entry_point_stage))
      vtn_foreach_decoration(b, val, handle_workgroup_size_decoration_cb,
                             NULL);
}
