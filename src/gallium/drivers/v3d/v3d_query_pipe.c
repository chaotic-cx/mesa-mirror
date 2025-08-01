/*
 * Copyright © 2014 Broadcom
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
 */

/**
 * Gallium query object support.
 *
 * The HW has native support for occlusion queries, with the query result
 * being loaded and stored by the TLB unit. From a SW perspective, we have to
 * be careful to make sure that the jobs that need to be tracking queries are
 * bracketed by the start and end of counting, even across FBO transitions.
 *
 * For the transform feedback PRIMITIVES_GENERATED/WRITTEN queries, we have to
 * do the calculations in software at draw time.
 */

#include "v3d_query.h"

struct v3d_query_pipe
{
        struct v3d_query base;

        enum pipe_query_type type;
        struct v3d_bo *bo;

        uint32_t start, end;
        uint32_t result;

        /* these fields are used for timestamp queries */
        uint64_t time_result;
        uint32_t sync[2];
};

static void
v3d_destroy_query_pipe(struct v3d_context *v3d, struct v3d_query *query)
{
        struct v3d_query_pipe *pquery = (struct v3d_query_pipe *)query;

        if (pquery->sync[0])
               drmSyncobjDestroy(v3d->fd, pquery->sync[0]);
        if (pquery->sync[1])
               drmSyncobjDestroy(v3d->fd, pquery->sync[1]);
        v3d_bo_unreference(&pquery->bo);
        free(pquery);
}

static bool
v3d_begin_query_pipe(struct v3d_context *v3d, struct v3d_query *query)
{
        struct v3d_query_pipe *pquery = (struct v3d_query_pipe *)query;

        switch (pquery->type) {
        case PIPE_QUERY_PRIMITIVES_GENERATED:
                /* If we are using PRIMITIVE_COUNTS_FEEDBACK to retrieve
                 * primitive counts from the GPU (which we need when a GS
                 * is present), then we need to update our counters now
                 * to discard any primitives generated before this.
                 */
                if (v3d->prog.gs)
                        v3d_update_primitive_counters(v3d);
                pquery->start = v3d->prims_generated;
                v3d->n_primitives_generated_queries_in_flight++;
                break;
        case PIPE_QUERY_PRIMITIVES_EMITTED:
                /* If we are inside transform feedback we need to update the
                 * primitive counts to skip primitives recorded before this.
                 */
                if (v3d->streamout.num_targets > 0)
                        v3d_update_primitive_counters(v3d);
                pquery->start = v3d->tf_prims_generated;
                break;
        case PIPE_QUERY_OCCLUSION_COUNTER:
        case PIPE_QUERY_OCCLUSION_PREDICATE:
        case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
                v3d_bo_unreference(&pquery->bo);
                pquery->bo = v3d_bo_alloc(v3d->screen, 4096, "query");
                uint32_t *map = v3d_bo_map(pquery->bo);
                *map = 0;

                v3d->current_oq = pquery->bo;
                v3d->dirty |= V3D_DIRTY_OQ;
                break;
        case PIPE_QUERY_TIME_ELAPSED:
                /* GL_TIME_ELAPSED​: Records the time that it takes for the GPU
                 * to execute all of the scoped commands.
                 *
                 * The timer starts when all commands before the scope have
                 * completed, and the timer ends when the last scoped command
                 * has completed.
                 */
                assert(pquery->bo);

                /* flush any pending jobs */
                v3d_flush(&v3d->base);

                /* submit time elapsed query to cpu queue */
                v3d_submit_timestamp_query(&v3d->base, pquery->bo,
                                           pquery->sync[0], 0);
                break;
        case PIPE_QUERY_TIMESTAMP_DISJOINT:
                break;
        default:
                UNREACHABLE("unsupported query type");
        }

        return true;
}

static bool
v3d_end_query_pipe(struct v3d_context *v3d, struct v3d_query *query)
{
        struct v3d_query_pipe *pquery = (struct v3d_query_pipe *)query;

        switch (pquery->type) {
        case PIPE_QUERY_PRIMITIVES_GENERATED:
                /* If we are using PRIMITIVE_COUNTS_FEEDBACK to retrieve
                 * primitive counts from the GPU (which we need when a GS
                 * is present), then we need to update our counters now.
                 */
                if (v3d->prog.gs)
                        v3d_update_primitive_counters(v3d);
                pquery->end = v3d->prims_generated;
                v3d->n_primitives_generated_queries_in_flight--;
                break;
        case PIPE_QUERY_PRIMITIVES_EMITTED:
                /* If transform feedback has ended, then we have already
                 * updated the primitive counts at glEndTransformFeedback()
                 * time. Otherwise, we have to do it now.
                 */
                if (v3d->streamout.num_targets > 0)
                        v3d_update_primitive_counters(v3d);
                pquery->end = v3d->tf_prims_generated;
                break;
        case PIPE_QUERY_OCCLUSION_COUNTER:
        case PIPE_QUERY_OCCLUSION_PREDICATE:
        case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
                v3d->current_oq = NULL;
                v3d->dirty |= V3D_DIRTY_OQ;
                break;
        case PIPE_QUERY_TIMESTAMP:
        case PIPE_QUERY_TIME_ELAPSED:
                /* Mesa only calls EndQuery and not BeginQuery for regular
                 * timestamp queries
                 *
                 * This will store into the query object the time when the GPU
                 * will have completed all previously issued commands.
                 */
                assert(pquery->bo);

                /* flush any pending jobs */
                v3d_flush(&v3d->base);

                /* submit time elapsed query to cpu queue */
                uint32_t offset = pquery->type == PIPE_QUERY_TIME_ELAPSED ?
                        sizeof(uint64_t) : 0;
                uint32_t sync = pquery->type == PIPE_QUERY_TIMESTAMP ? 0 : 1;
                v3d_submit_timestamp_query(&v3d->base, pquery->bo,
                                           pquery->sync[sync], offset);
                break;
        case PIPE_QUERY_TIMESTAMP_DISJOINT:
                break;
        default:
                UNREACHABLE("unsupported query type");
        }

        return true;
}

static bool
v3d_get_query_result_pipe(struct v3d_context *v3d, struct v3d_query *query,
                          bool wait, union pipe_query_result *vresult)
{
        struct v3d_query_pipe *pquery = (struct v3d_query_pipe *)query;

        if (pquery->bo) {
                /* For timestamp & time elapsed queries we already flush
                 * relevant jobs before submitting the query */
                if (pquery->type != PIPE_QUERY_TIMESTAMP &&
                    pquery->type != PIPE_QUERY_TIME_ELAPSED) {
                        v3d_flush_jobs_using_bo(v3d, pquery->bo);
                }

                if (wait) {
                        if (!v3d_bo_wait(pquery->bo, ~0ull, "query"))
                                return false;

                        assert((pquery->type != PIPE_QUERY_TIMESTAMP &&
                               pquery->type != PIPE_QUERY_TIME_ELAPSED) ||
                               drmSyncobjWait(v3d->fd, &pquery->sync[0], 1, 0,
                                              0, NULL) != -ETIME);

                        assert(pquery->type != PIPE_QUERY_TIME_ELAPSED ||
                                drmSyncobjWait(v3d->fd, &pquery->sync[1], 1,
                                               0, 0, NULL) != -ETIME);
                } else {
                        if (!v3d_bo_wait(pquery->bo, 0, "query"))
                                return false;
                }

                if (pquery->type == PIPE_QUERY_TIMESTAMP) {
                        uint64_t *map = v3d_bo_map(pquery->bo);
                        pquery->time_result = *map;
                } else if (pquery->type == PIPE_QUERY_TIME_ELAPSED) {
                        uint64_t *map = v3d_bo_map(pquery->bo);
                        pquery->time_result = map[1] - map[0];
                } else {
                        /* XXX: Sum up per-core values. */
                        uint32_t *map = v3d_bo_map(pquery->bo);
                        pquery->result = *map;

                        /* FIXME: we should move creation and destruction of
                         * the BO for all queries to query create/destruction,
                         * like we do with timestamps */
                        v3d_bo_unreference(&pquery->bo);
                }
        }

        switch (pquery->type) {
        case PIPE_QUERY_OCCLUSION_COUNTER:
                vresult->u64 = pquery->result;
                break;
        case PIPE_QUERY_OCCLUSION_PREDICATE:
        case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
                vresult->b = pquery->result != 0;
                break;
        case PIPE_QUERY_PRIMITIVES_GENERATED:
        case PIPE_QUERY_PRIMITIVES_EMITTED:
                vresult->u64 = pquery->end - pquery->start;
                break;
        case PIPE_QUERY_TIMESTAMP:
        case PIPE_QUERY_TIME_ELAPSED:
                vresult->u64 = pquery->time_result;
                break;
        case PIPE_QUERY_TIMESTAMP_DISJOINT:
                /* os_time_get_nano returns time in nanoseconds */
                vresult->timestamp_disjoint.frequency = UINT64_C(1000000000);
                vresult->timestamp_disjoint.disjoint = false;
           break;
        default:
                UNREACHABLE("unsupported query type");
        }

        return true;
}

static const struct v3d_query_funcs pipe_query_funcs = {
        .destroy_query = v3d_destroy_query_pipe,
        .begin_query = v3d_begin_query_pipe,
        .end_query = v3d_end_query_pipe,
        .get_query_result = v3d_get_query_result_pipe,
};

struct pipe_query *
v3d_create_query_pipe(struct v3d_context *v3d, unsigned query_type, unsigned index)
{
        if (query_type >= PIPE_QUERY_DRIVER_SPECIFIC)
                return NULL;

        struct v3d_query_pipe *pquery = calloc(1, sizeof(*pquery));
        struct v3d_query *query = &pquery->base;

        pquery->type = query_type;
        query->funcs = &pipe_query_funcs;

        /* FIXME: we should probably allocate BOs for occlusion queries here
         * as well
         */
        switch (pquery->type) {
        case PIPE_QUERY_TIMESTAMP:
        case PIPE_QUERY_TIME_ELAPSED:
                pquery->bo = v3d_bo_alloc(v3d->screen, 4096, "query");
                uint32_t *map = v3d_bo_map(pquery->bo);
                *map = 0;

                drmSyncobjCreate(v3d->fd, 0, &pquery->sync[0]);
                if (pquery->type == PIPE_QUERY_TIME_ELAPSED)
                        drmSyncobjCreate(v3d->fd, 0, &pquery->sync[1]);
                break;
        default:
                break;
        }

        /* Note that struct pipe_query isn't actually defined anywhere. */
        return (struct pipe_query *)query;
}
