/*
 * Copyright © 2016 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#ifndef FREEDRENO_BATCH_CACHE_H_
#define FREEDRENO_BATCH_CACHE_H_

#include "pipe/p_state.h"

#include "freedreno_util.h"

BEGINC;

struct fd_resource;
struct fd_batch;
struct fd_context;
struct fd_screen;

struct hash_table;

struct fd_batch_cache {
   struct hash_table *ht;
   seqno_t cnt;

   /* set of active batches.. there is an upper limit on the number of
    * in-flight batches, for two reasons:
    * 1) to avoid big spikes in number of batches in edge cases, such as
    *    game startup (ie, lots of texture uploads, but no usages yet of
    *    the textures), etc.
    * 2) so we can use a simple bitmask in fd_resource to track which
    *    batches have reference to the resource
    */
   struct fd_batch *batches[32];
   uint32_t batch_mask;
};

/* note: if batches get unref'd in the body of the loop, they are removed
 * from the various masks.. but since we copy the mask at the beginning of
 * the loop into _m, we need the &= at the end of the loop to make sure
 * we don't have stale bits in _m
 */
#define foreach_batch(batch, cache, mask)                                      \
   for (uint32_t _m = (mask);                                                  \
        _m && ((batch) = (cache)->batches[u_bit_scan(&_m)]); _m &= (mask))

void fd_bc_init(struct fd_batch_cache *cache);
void fd_bc_fini(struct fd_batch_cache *cache);

struct fd_batch *fd_bc_last_batch(struct fd_context *ctx) assert_dt;
void fd_bc_add_flush_deps(struct fd_context *ctx, struct fd_batch *last_batch) assert_dt;
void fd_bc_flush_writer(struct fd_context *ctx, struct fd_resource *rsc) assert_dt;
void fd_bc_flush_readers(struct fd_context *ctx, struct fd_resource *rsc) assert_dt;
void fd_bc_dump(struct fd_context *ctx, const char *fmt, ...)
   _util_printf_format(2, 3);

void fd_bc_invalidate_batch(struct fd_batch *batch, bool destroy);
void fd_bc_invalidate_resource(struct fd_resource *rsc, bool destroy);
struct fd_batch *fd_bc_alloc_batch(struct fd_context *ctx,
                                   bool nondraw) assert_dt;

struct fd_batch *
fd_batch_from_fb(struct fd_context *ctx,
                 const struct pipe_framebuffer_state *pfb) assert_dt;

ENDC;

#endif /* FREEDRENO_BATCH_CACHE_H_ */
