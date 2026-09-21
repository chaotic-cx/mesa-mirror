/*
 * Copyright © 2025 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "helpers.h"

class Workarounds : public spirv_test {};

TEST_F(Workarounds, force_ssbo_non_uniform)
{
   /*
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpSource GLSL 460
               OpSourceExtension "GL_EXT_buffer_reference"
               OpName %main "main"
               OpName %data "data"
               OpMemberName %data 0 "v"
               OpName %ssbo "ssbo"
               OpDecorate %_runtimearr_float ArrayStride 4
               OpDecorate %data BufferBlock
               OpMemberDecorate %data 0 Offset 0
               OpDecorate %ssbo Binding 0
               OpDecorate %ssbo DescriptorSet 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_runtimearr_float = OpTypeRuntimeArray %float
       %data = OpTypeStruct %_runtimearr_float
%_ptr_Uniform_data = OpTypePointer Uniform %data
       %ssbo = OpVariable %_ptr_Uniform_data Uniform
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
       %main = OpFunction %void None %3
          %5 = OpLabel
         %15 = OpAccessChain %_ptr_Uniform_float %ssbo %int_0 %int_0
               OpStore %15 %float_0
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010000, 0x0008000b, 0x00000010, 0x00000000, 0x00020011,
      0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
      0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005,
      0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
      0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001cc,
      0x00070004, 0x455f4c47, 0x625f5458, 0x65666675, 0x65725f72, 0x65726566,
      0x0065636e, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00040005,
      0x00000008, 0x61746164, 0x00000000, 0x00040006, 0x00000008, 0x00000000,
      0x00000076, 0x00040005, 0x0000000a, 0x6f627373, 0x00000000, 0x00040047,
      0x00000007, 0x00000006, 0x00000004, 0x00030047, 0x00000008, 0x00000003,
      0x00050048, 0x00000008, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
      0x0000000a, 0x00000021, 0x00000000, 0x00040047, 0x0000000a, 0x00000022,
      0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
      0x00030016, 0x00000006, 0x00000020, 0x0003001d, 0x00000007, 0x00000006,
      0x0003001e, 0x00000008, 0x00000007, 0x00040020, 0x00000009, 0x00000002,
      0x00000008, 0x0004003b, 0x00000009, 0x0000000a, 0x00000002, 0x00040015,
      0x0000000b, 0x00000020, 0x00000001, 0x0004002b, 0x0000000b, 0x0000000c,
      0x00000000, 0x0004002b, 0x00000006, 0x0000000d, 0x00000000, 0x00040020,
      0x0000000e, 0x00000002, 0x00000006, 0x00050036, 0x00000002, 0x00000004,
      0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00060041, 0x0000000e,
      0x0000000f, 0x0000000a, 0x0000000c, 0x0000000c, 0x0003003e, 0x0000000f,
      0x0000000d, 0x000100fd, 0x00010038,
   };

   spirv_options.workarounds.force_ssbo_non_uniform = true;

   get_nir(sizeof(words) / sizeof(words[0]), words, MESA_SHADER_COMPUTE);

   nir_intrinsic_instr *intrinsic = find_intrinsic(nir_intrinsic_store_deref, 0);
   ASSERT_NE(intrinsic, nullptr);

   ASSERT_EQ(nir_intrinsic_access(intrinsic), ACCESS_NON_UNIFORM);
}

TEST_F(Workarounds, force_tex_non_uniform)
{
   /*
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %color %coord
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 460
               OpName %main "main"
               OpName %color "color"
               OpName %samplerColor "samplerColor"
               OpName %coord "coord"
               OpDecorate %color Location 0
               OpDecorate %samplerColor Binding 0
               OpDecorate %samplerColor DescriptorSet 0
               OpDecorate %coord Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %color = OpVariable %_ptr_Output_v4float Output
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
%samplerColor = OpVariable %_ptr_UniformConstant_11 UniformConstant
    %v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
      %coord = OpVariable %_ptr_Input_v2float Input
       %main = OpFunction %void None %3
          %5 = OpLabel
         %14 = OpLoad %11 %samplerColor
         %18 = OpLoad %v2float %coord
         %19 = OpImageSampleImplicitLod %v4float %14 %18
               OpStore %color %19
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010000, 0x0008000b, 0x00000014, 0x00000000, 0x00020011,
      0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
      0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000004,
      0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00000011, 0x00030010,
      0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001cc, 0x00040005,
      0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f6c6f63,
      0x00000072, 0x00060005, 0x0000000d, 0x706d6173, 0x4372656c, 0x726f6c6f,
      0x00000000, 0x00040005, 0x00000011, 0x726f6f63, 0x00000064, 0x00040047,
      0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000d, 0x00000021,
      0x00000000, 0x00040047, 0x0000000d, 0x00000022, 0x00000000, 0x00040047,
      0x00000011, 0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
      0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
      0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003,
      0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00090019,
      0x0000000a, 0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000,
      0x00000001, 0x00000000, 0x0003001b, 0x0000000b, 0x0000000a, 0x00040020,
      0x0000000c, 0x00000000, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d,
      0x00000000, 0x00040017, 0x0000000f, 0x00000006, 0x00000002, 0x00040020,
      0x00000010, 0x00000001, 0x0000000f, 0x0004003b, 0x00000010, 0x00000011,
      0x00000001, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
      0x000200f8, 0x00000005, 0x0004003d, 0x0000000b, 0x0000000e, 0x0000000d,
      0x0004003d, 0x0000000f, 0x00000012, 0x00000011, 0x00050057, 0x00000007,
      0x00000013, 0x0000000e, 0x00000012, 0x0003003e, 0x00000009, 0x00000013,
      0x000100fd, 0x00010038,
   };

   spirv_options.workarounds.force_tex_non_uniform = true;

   get_nir(sizeof(words) / sizeof(words[0]), words, MESA_SHADER_FRAGMENT);

   nir_tex_instr *tex_instr = find_tex_instr(nir_texop_tex, 0);
   ASSERT_NE(tex_instr, nullptr);

   ASSERT_TRUE(tex_instr->texture_non_uniform);
}

TEST_F(Workarounds, lower_terminate_to_discard)
{
   /*
               OpCapability Shader
               OpExtension "SPV_KHR_terminate_invocation"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %coord %color
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 460
               OpSourceExtension "GL_EXT_terminate_invocation"
               OpName %main "main"
               OpName %coord "coord"
               OpName %color "color"
               OpDecorate %coord Location 0
               OpDecorate %color Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
      %coord = OpVariable %_ptr_Input_v2float Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
    %float_0 = OpConstant %float 0
       %bool = OpTypeBool
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %color = OpVariable %_ptr_Output_v4float Output
         %24 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
       %main = OpFunction %void None %3
          %5 = OpLabel
         %13 = OpAccessChain %_ptr_Input_float %coord %uint_0
         %14 = OpLoad %float %13
         %17 = OpFOrdEqual %bool %14 %float_0
               OpSelectionMerge %19 None
               OpBranchConditional %17 %18 %19
         %18 = OpLabel
               OpTerminateInvocation
         %19 = OpLabel
               OpStore %color %24
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010000, 0x0008000b, 0x00000019, 0x00000000, 0x00020011,
      0x00000001, 0x0009000a, 0x5f565053, 0x5f52484b, 0x6d726574, 0x74616e69,
      0x6e695f65, 0x61636f76, 0x6e6f6974, 0x00000000, 0x0006000b, 0x00000001,
      0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000,
      0x00000001, 0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000,
      0x00000009, 0x00000017, 0x00030010, 0x00000004, 0x00000007, 0x00030003,
      0x00000002, 0x000001cc, 0x00080004, 0x455f4c47, 0x745f5458, 0x696d7265,
      0x6574616e, 0x766e695f, 0x7461636f, 0x006e6f69, 0x00040005, 0x00000004,
      0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x726f6f63, 0x00000064,
      0x00040005, 0x00000017, 0x6f6c6f63, 0x00000072, 0x00040047, 0x00000009,
      0x0000001e, 0x00000000, 0x00040047, 0x00000017, 0x0000001e, 0x00000000,
      0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
      0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000002,
      0x00040020, 0x00000008, 0x00000001, 0x00000007, 0x0004003b, 0x00000008,
      0x00000009, 0x00000001, 0x00040015, 0x0000000a, 0x00000020, 0x00000000,
      0x0004002b, 0x0000000a, 0x0000000b, 0x00000000, 0x00040020, 0x0000000c,
      0x00000001, 0x00000006, 0x0004002b, 0x00000006, 0x0000000f, 0x00000000,
      0x00020014, 0x00000010, 0x00040017, 0x00000015, 0x00000006, 0x00000004,
      0x00040020, 0x00000016, 0x00000003, 0x00000015, 0x0004003b, 0x00000016,
      0x00000017, 0x00000003, 0x0007002c, 0x00000015, 0x00000018, 0x0000000f,
      0x0000000f, 0x0000000f, 0x0000000f, 0x00050036, 0x00000002, 0x00000004,
      0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x0000000c,
      0x0000000d, 0x00000009, 0x0000000b, 0x0004003d, 0x00000006, 0x0000000e,
      0x0000000d, 0x000500b4, 0x00000010, 0x00000011, 0x0000000e, 0x0000000f,
      0x000300f7, 0x00000013, 0x00000000, 0x000400fa, 0x00000011, 0x00000012,
      0x00000013, 0x000200f8, 0x00000012, 0x00011140, 0x000200f8, 0x00000013,
      0x0003003e, 0x00000017, 0x00000018, 0x000100fd, 0x00010038,
   };

   spirv_options.workarounds.lower_terminate_to_discard = true;

   /* Emitting demote instead of terminate is needed to verify the SPIR-V
    * workaround, otherwise terminate is always used.
    */
   nir_options.discard_is_demote = true;

   get_nir(sizeof(words) / sizeof(words[0]), words, MESA_SHADER_FRAGMENT);

   nir_intrinsic_instr *intrinsic = find_intrinsic(nir_intrinsic_demote, 0);
   ASSERT_NE(intrinsic, nullptr);
}

/* Test fixup_dxc_image_derefs(). */
TEST_F(Workarounds, image_function_vars_simple)
{
   /*
               OpCapability Shader
               OpCapability ImageBuffer
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main" %2
               OpExecutionMode %1 LocalSize 8 8 1
               OpDecorate %2 DescriptorSet 0
               OpDecorate %2 Binding 0
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
          %7 = OpTypeImage %uint Buffer 2 0 0 2 R32ui
%_ptr_UniformConstant_7 = OpTypePointer UniformConstant %7
       %void = OpTypeVoid
         %10 = OpTypeFunction %void
%_ptr_Function_7 = OpTypePointer Function %7
%_ptr_Image_uint = OpTypePointer Image %uint
          %2 = OpVariable %_ptr_UniformConstant_7 UniformConstant
          %1 = OpFunction %void None %10
         %13 = OpLabel
         %14 = OpVariable %_ptr_Function_7 Function
         %15 = OpLoad %7 %2
               OpStore %14 %15
         %16 = OpImageTexelPointer %_ptr_Image_uint %14 %uint_2 %uint_0
         %17 = OpAtomicIAdd %uint %16 %uint_1 %uint_0 %uint_1
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010600, 0x00070000, 0x00000012, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x0000002f, 0x0003000e, 0x00000000, 0x00000001,
      0x0006000f, 0x00000005, 0x00000001, 0x6e69616d, 0x00000000, 0x00000002,
      0x00060010, 0x00000001, 0x00000011, 0x00000008, 0x00000008, 0x00000001,
      0x00040047, 0x00000002, 0x00000022, 0x00000000, 0x00040047, 0x00000002,
      0x00000021, 0x00000000, 0x00040015, 0x00000003, 0x00000020, 0x00000000,
      0x0004002b, 0x00000003, 0x00000004, 0x00000000, 0x0004002b, 0x00000003,
      0x00000005, 0x00000001, 0x0004002b, 0x00000003, 0x00000006, 0x00000002,
      0x00090019, 0x00000007, 0x00000003, 0x00000005, 0x00000002, 0x00000000,
      0x00000000, 0x00000002, 0x00000021, 0x00040020, 0x00000008, 0x00000000,
      0x00000007, 0x00020013, 0x00000009, 0x00030021, 0x0000000a, 0x00000009,
      0x00040020, 0x0000000b, 0x00000007, 0x00000007, 0x00040020, 0x0000000c,
      0x0000000b, 0x00000003, 0x0004003b, 0x00000008, 0x00000002, 0x00000000,
      0x00050036, 0x00000009, 0x00000001, 0x00000000, 0x0000000a, 0x000200f8,
      0x0000000d, 0x0004003b, 0x0000000b, 0x0000000e, 0x00000007, 0x0004003d,
      0x00000007, 0x0000000f, 0x00000002, 0x0003003e, 0x0000000e, 0x0000000f,
      0x0006003c, 0x0000000c, 0x00000010, 0x0000000e, 0x00000006, 0x00000004,
      0x000700ea, 0x00000003, 0x00000011, 0x00000010, 0x00000005, 0x00000004,
      0x00000005, 0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words, MESA_SHADER_COMPUTE);
   nir_validate_shader(shader, "after get_nir");

   nir_intrinsic_instr *atomic = find_intrinsic(nir_intrinsic_image_deref_atomic);
   ASSERT_NE(atomic, nullptr);
   EXPECT_EQ(nir_intrinsic_image_dim(atomic), GLSL_SAMPLER_DIM_BUF);
   EXPECT_FALSE(nir_intrinsic_image_array(atomic));
   EXPECT_EQ(nir_intrinsic_format(atomic), PIPE_FORMAT_R32_UINT);

   ASSERT_NE(nir_src_as_deref(atomic->src[0]), nullptr);
   EXPECT_EQ(nir_src_as_deref(atomic->src[0])->modes, nir_var_image);

   for (unsigned i = 1; i < 4; i++)
      EXPECT_TRUE(nir_def_is_undef(nir_scalar_resolved(atomic->src[1].ssa, i).def));
}

TEST_F(Workarounds, image_function_vars_array)
{
   /*
               OpCapability Shader
               OpCapability ImageBuffer
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main" %2
               OpExecutionMode %1 LocalSize 8 8 1
               OpDecorate %2 DescriptorSet 0
               OpDecorate %2 Binding 0
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
          %7 = OpTypeImage %uint Buffer 2 0 0 2 R32ui
%_runtimearr_7 = OpTypeRuntimeArray %7
%_ptr_UniformConstant__runtimearr_7 = OpTypePointer UniformConstant %_runtimearr_7
%_ptr_Function_7 = OpTypePointer Function %7
       %void = OpTypeVoid
         %12 = OpTypeFunction %void
%_ptr_Function_7_0 = OpTypePointer Function %7
%_ptr_Image_uint = OpTypePointer Image %uint
          %2 = OpVariable %_ptr_UniformConstant__runtimearr_7 UniformConstant
          %1 = OpFunction %void None %12
         %15 = OpLabel
         %16 = OpVariable %_ptr_Function_7_0 Function
         %17 = OpAccessChain %_ptr_Function_7 %2 %uint_2
         %18 = OpLoad %7 %17
               OpStore %16 %18
         %19 = OpImageTexelPointer %_ptr_Image_uint %16 %uint_2 %uint_0
         %20 = OpAtomicIAdd %uint %19 %uint_1 %uint_0 %uint_1
               OpReturn
               OpFunctionEnd
   */
   static const uint32_t words[] = {
      0x07230203, 0x00010600, 0x00070000, 0x00000015, 0x00000000, 0x00020011,
      0x00000001, 0x00020011, 0x0000002f, 0x0003000e, 0x00000000, 0x00000001,
      0x0006000f, 0x00000005, 0x00000001, 0x6e69616d, 0x00000000, 0x00000002,
      0x00060010, 0x00000001, 0x00000011, 0x00000008, 0x00000008, 0x00000001,
      0x00040047, 0x00000002, 0x00000022, 0x00000000, 0x00040047, 0x00000002,
      0x00000021, 0x00000000, 0x00040015, 0x00000003, 0x00000020, 0x00000000,
      0x0004002b, 0x00000003, 0x00000004, 0x00000000, 0x0004002b, 0x00000003,
      0x00000005, 0x00000001, 0x0004002b, 0x00000003, 0x00000006, 0x00000002,
      0x00090019, 0x00000007, 0x00000003, 0x00000005, 0x00000002, 0x00000000,
      0x00000000, 0x00000002, 0x00000021, 0x0003001d, 0x00000008, 0x00000007,
      0x00040020, 0x00000009, 0x00000000, 0x00000008, 0x00040020, 0x0000000a,
      0x00000007, 0x00000007, 0x00020013, 0x0000000b, 0x00030021, 0x0000000c,
      0x0000000b, 0x00040020, 0x0000000d, 0x00000007, 0x00000007, 0x00040020,
      0x0000000e, 0x0000000b, 0x00000003, 0x0004003b, 0x00000009, 0x00000002,
      0x00000000, 0x00050036, 0x0000000b, 0x00000001, 0x00000000, 0x0000000c,
      0x000200f8, 0x0000000f, 0x0004003b, 0x0000000d, 0x00000010, 0x00000007,
      0x00050041, 0x0000000a, 0x00000011, 0x00000002, 0x00000006, 0x0004003d,
      0x00000007, 0x00000012, 0x00000011, 0x0003003e, 0x00000010, 0x00000012,
      0x0006003c, 0x0000000e, 0x00000013, 0x00000010, 0x00000006, 0x00000004,
      0x000700ea, 0x00000003, 0x00000014, 0x00000013, 0x00000005, 0x00000004,
      0x00000005, 0x000100fd, 0x00010038,
   };

   get_nir(sizeof(words) / sizeof(words[0]), words, MESA_SHADER_COMPUTE);
   nir_validate_shader(shader, "after get_nir");
   nir_opt_cse(shader);

   nir_intrinsic_instr *store = find_intrinsic(nir_intrinsic_store_deref);
   nir_intrinsic_instr *load = find_intrinsic(nir_intrinsic_load_deref);

   nir_deref_instr *tmp_deref = nir_src_as_deref(store->src[0]);
   EXPECT_EQ(store->src[0].ssa, load->src[0].ssa);
   EXPECT_EQ(tmp_deref->modes, nir_var_function_temp);
   EXPECT_EQ(tmp_deref->type, &glsl_type_builtin_uint);

   ASSERT_TRUE(nir_src_is_const(store->src[1]));
   EXPECT_EQ(nir_src_as_uint(store->src[1]), 2);

   nir_intrinsic_instr *atomic = find_intrinsic(nir_intrinsic_image_deref_atomic);
   ASSERT_NE(atomic, nullptr);
   EXPECT_EQ(nir_intrinsic_image_dim(atomic), GLSL_SAMPLER_DIM_BUF);
   EXPECT_FALSE(nir_intrinsic_image_array(atomic));
   EXPECT_EQ(nir_intrinsic_format(atomic), PIPE_FORMAT_R32_UINT);

   nir_deref_instr *deref = nir_src_as_deref(atomic->src[0]);
   ASSERT_NE(deref, nullptr);
   EXPECT_EQ(deref->modes, nir_var_image);
   ASSERT_EQ(deref->deref_type, nir_deref_type_array);
   EXPECT_EQ(deref->arr.index.ssa, &load->def);

   for (unsigned i = 1; i < 4; i++)
      EXPECT_TRUE(nir_def_is_undef(nir_scalar_resolved(atomic->src[1].ssa, i).def));
}
