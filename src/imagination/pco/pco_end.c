/*
 * Copyright © 2024 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pco_end.c
 *
 * \brief PCO shader ending pass.
 */

#include "pco.h"
#include "pco_builder.h"
#include "pco_internal.h"
#include "util/macros.h"

#include <stdbool.h>

/**
 * \brief For vertex shaders, inserts a uvsw.endtask instruction at the end of
 *        the shader, or if possible, rewrites the last instruction into a
 *        uvsw.write.emit.endtask.
 *
 * \param[in,out] func PCO function.
 * \param[in,out] _last_instr Pointer to the last instruction in the shader.
 * \return True if the pass made progress.
 */
static
bool
pco_insert_uvsw_endtask(pco_func *func, pco_instr **_last_instr)
{
   assert(func->parent_shader->stage == MESA_SHADER_VERTEX);

   pco_instr *last_instr = *_last_instr;

   pco_cursor cursor =
      last_instr ? pco_cursor_after_instr(last_instr) :
                   pco_cursor_after_block(pco_func_last_block(func));

   pco_builder b = pco_builder_create(func, cursor);

   if (last_instr && last_instr->op == PCO_OP_UVSW_WRITE &&
       pco_instr_has_default_exec(last_instr) &&
       pco_instr_get_rpt(last_instr) == 1) {
      last_instr =
         pco_uvsw_write_emit_endtask(&b,
                                     last_instr->src[0],
                                     last_instr->src[1]);

      pco_instr_delete(*_last_instr);
      *_last_instr = last_instr;

      return true;
   }

   *_last_instr = pco_uvsw_emit_endtask(&b);
   return true;
}

/**
 * \brief Inserts a nop.end instruction at the end of the shader,
 *        or if possible, sets the end op mod on the last instruction.
 *
 * \param[in,out] func PCO function.
 * \param[in,out] _last_instr Pointer to the last instruction in the shader.
 * \return True if the pass made progress.
 */
static
bool
pco_insert_end(pco_func *func, pco_instr **_last_instr)
{
   pco_instr *last_instr = *_last_instr;

   pco_cursor cursor =
      last_instr ? pco_cursor_after_instr(last_instr) :
                   pco_cursor_after_block(pco_func_last_block(func));

   pco_builder b = pco_builder_create(func, cursor);

   if (last_instr && pco_instr_has_end(last_instr)) {
      pco_instr_set_end(last_instr, true);
      return true;
   }

   *_last_instr = pco_nop(&b, .end = true);

   return true;
}

/**
 * \brief Processes end of shader instruction(s).
 *
 * \param[in,out] shader PCO shader.
 * \return True if the pass made progress.
 */
bool pco_end(pco_shader *shader)
{
   bool progress = false;

   pco_foreach_func_in_shader(func, shader) {
      pco_block *last_block = pco_func_last_block(func);
      pco_instr *last_instr = pco_last_instr(last_block);

      if (shader->stage == MESA_SHADER_VERTEX &&
          func->type == PCO_FUNC_TYPE_ENTRYPOINT) {
         progress |= pco_insert_uvsw_endtask(func, &last_instr);
      }

      progress |= pco_insert_end(func, &last_instr);
   }

   return progress;
}
