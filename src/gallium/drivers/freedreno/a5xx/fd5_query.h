/*
 * Copyright © 2016 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#ifndef FD5_QUERY_H_
#define FD5_QUERY_H_

#include "pipe/p_context.h"

#include "freedreno_batch.h"

void fd5_query_context_init(struct pipe_context *pctx);

void fd5_start_prim_counters(struct fd_batch *batch,
                             struct fd_ringbuffer *ring);
void fd5_stop_prim_counters(struct fd_batch *batch, struct fd_ringbuffer *ring);

#endif /* FD5_QUERY_H_ */
