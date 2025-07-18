/*
 * Copyright (c) 2008-2024 Broadcom. All Rights Reserved.
 * The term “Broadcom” refers to Broadcom Inc.
 * and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef SVGA_SAMPLER_VIEW_H
#define SVGA_SAMPLER_VIEW_H


#include "util/compiler.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include "svga_screen_cache.h"

struct pipe_context;
struct pipe_screen;
struct svga_context;
struct svga_pipe_sampler_view;
struct svga_winsys_surface;
struct svga_surface;
enum SVGA3dSurfaceFormat;


/**
 * A sampler's view into a texture
 *
 * We currently cache one sampler view on
 * the texture and in there by holding a reference
 * from the texture to the sampler view.
 *
 * Because of this we can not hold a refernce to the
 * texture from the sampler view. So the user
 * of the sampler views must make sure that the
 * texture has a reference take for as long as
 * the sampler view is refrenced.
 *
 * Just unreferencing the sampler_view before the
 * texture is enough.
 */
struct svga_sampler_view
{
   struct pipe_reference reference;

   struct pipe_resource *texture;

   int min_lod;
   int max_lod;

   unsigned age;

   struct svga_host_surface_cache_key key;
   struct svga_winsys_surface *handle;
};



extern struct svga_sampler_view *
svga_get_tex_sampler_view(struct pipe_context *pipe,
                          struct pipe_resource *pt,
                          unsigned min_lod, unsigned max_lod);

void
svga_validate_sampler_view(struct svga_context *svga, struct svga_sampler_view *v);

void
svga_destroy_sampler_view_priv(struct svga_sampler_view *v);

void
svga_debug_describe_sampler_view(char *buf, const struct svga_sampler_view *sv);

static inline void
svga_sampler_view_reference(struct svga_sampler_view **ptr, struct svga_sampler_view *v)
{
   struct svga_sampler_view *old = *ptr;

   if (pipe_reference_described(&(*ptr)->reference, &v->reference,
                                (debug_reference_descriptor)svga_debug_describe_sampler_view))
      svga_destroy_sampler_view_priv(old);
   *ptr = v;
}

bool
svga_check_sampler_view_resource_collision(const struct svga_context *svga,
                                           const struct svga_winsys_surface *res,
                                           enum pipe_shader_type shader);

bool
svga_check_sampler_framebuffer_resource_collision(struct svga_context *svga,
                                                  enum pipe_shader_type shader);

enum pipe_error
svga_validate_pipe_sampler_view(struct svga_context *svga,
                                struct svga_pipe_sampler_view *sv);

#endif
