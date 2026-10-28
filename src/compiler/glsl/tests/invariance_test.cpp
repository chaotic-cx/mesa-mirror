/* SPDX-License-Identifier: MIT */

#include <gtest/gtest.h>

#include "ir.h"
#include "builtin_functions.h"
#include "glsl_parser_extras.h"
#include "main/mtypes.h"
#include "nir.h"
#include "pipe/p_screen.h"
#include "standalone_scaffolding.h"

class glsl_invariance : public ::testing::Test {
protected:
   void SetUp() override
   {
      glsl_type_singleton_init_or_ref();
      _mesa_glsl_builtin_functions_init_or_ref();
      initialize_context_to_defaults(&ctx, API_OPENGL_COMPAT);
      ctx.Version = 40;
      ctx.Const.GLSLVersion = 400;
      static const nir_shader_compiler_options options = {};
      ctx.screen->nir_options[MESA_SHADER_VERTEX] = &options;
      program = standalone_create_shader_program();
   }

   void TearDown() override
   {
      standalone_destroy_shader_program(program);
      free(ctx.screen);
      _mesa_glsl_builtin_functions_decref();
      glsl_type_singleton_decref();
   }

   void check_shader(const char *source, bool exact)
   {
      gl_shader *shader = standalone_add_shader_source(&ctx, program,
                                                      GL_VERTEX_SHADER, source);
      _mesa_glsl_compile_shader(&ctx, shader, NULL, false, false, true);
      ASSERT_EQ(shader->CompileStatus, COMPILE_SUCCESS) << shader->InfoLog;
      ASSERT_NE(shader->nir, nullptr);

      unsigned instructions = 0;
      nir_foreach_function_impl(impl, shader->nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op != nir_op_fmul && alu->op != nir_op_fadd &&
                   alu->op != nir_op_ffma)
                  continue;
               EXPECT_EQ(exact, nir_alu_instr_is_exact(alu))
                  << nir_op_infos[alu->op].name;
               instructions++;
            }
         }
      }
      ASSERT_GT(instructions, 0u);
   }

   gl_context ctx;
   gl_shader_program *program;
};

TEST_F(glsl_invariance, invariant_matrix_product)
{
   check_shader(R"(#version 120
      uniform mat4 a, b;
      attribute vec4 position;
      invariant gl_Position;
      void main() {
         gl_Position = (a * b) * position;
      }
   )", true);
}

TEST_F(glsl_invariance, precise_matrix_product)
{
   check_shader(R"(#version 400
      uniform mat4 a, b;
      in vec4 position;
      void main() {
         precise vec4 value = (a * b) * position;
         gl_Position = value;
      }
   )", true);
}

TEST_F(glsl_invariance, non_invariant_matrix_product)
{
   check_shader(R"(#version 120
      uniform mat4 a, b;
      attribute vec4 position;
      void main() {
         gl_Position = (a * b) * position;
      }
   )", false);
}

TEST_F(glsl_invariance, precise_double_dot)
{
   check_shader(R"(#version 400
      uniform dvec4 a, b;
      void main() {
         precise double value = dot(a, b);
         gl_Position = vec4(value);
      }
   )", true);
}

TEST_F(glsl_invariance, non_precise_double_dot)
{
   check_shader(R"(#version 400
      uniform dvec4 a, b;
      void main() {
         gl_Position = vec4(dot(a, b));
      }
   )", false);
}
