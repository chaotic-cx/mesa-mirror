/**************************************************************************
 *
 * Copyright 2009 Younes Manton.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef PIPE_VIDEO_CONTEXT_H
#define PIPE_VIDEO_CONTEXT_H

#include "pipe/p_video_state.h"

#ifdef __cplusplus
extern "C" {
#endif

struct pipe_screen;
struct pipe_surface;
struct pipe_macroblock;
struct pipe_picture_desc;
struct pipe_fence_handle;

/**
 * Gallium video codec for a specific format/profile
 */
struct pipe_video_codec
{
   struct pipe_context *context;

   enum pipe_video_profile profile;
   unsigned level;
   enum pipe_video_entrypoint entrypoint;
   enum pipe_video_chroma_format chroma_format;
   unsigned width;
   unsigned height;
   unsigned max_references;
   bool expect_chunked_decode;
   struct pipe_enc_two_pass_encoder_config two_pass;

   /**
    * destroy this video decoder
    */
   void (*destroy)(struct pipe_video_codec *codec);

   /**
    * start decoding of a new frame
    */
   void (*begin_frame)(struct pipe_video_codec *codec,
                       struct pipe_video_buffer *target,
                       struct pipe_picture_desc *picture);

   /**
    * decode a macroblock
    */
   void (*decode_macroblock)(struct pipe_video_codec *codec,
                             struct pipe_video_buffer *target,
                             struct pipe_picture_desc *picture,
                             const struct pipe_macroblock *macroblocks,
                             unsigned num_macroblocks);

   /**
    * decode a bitstream
    */
   void (*decode_bitstream)(struct pipe_video_codec *codec,
                            struct pipe_video_buffer *target,
                            struct pipe_picture_desc *picture,
                            unsigned num_buffers,
                            const void * const *buffers,
                            const unsigned *sizes);

   /**
    * encode to a bitstream
    */
   void (*encode_bitstream)(struct pipe_video_codec *codec,
                            struct pipe_video_buffer *source,
                            struct pipe_resource *destination,
                            void **feedback);

   /**
    * encode an entire frame texture to a bitstream, but get notified asynchronously
    * in slice_fences[] as the slices are ready (can be of out order if multi engine encoder)
    * for frontend consumption before the full frame is finished.
    *
    * The different slices are written in each slice_destinations[] buffer
    *
    * num_slice_objects indicates the number of elements in the input
    * array slice_destinations and indicates the number of outputs expected
    * in slice_fences
    *
    * The frame NALs are attached to the first slice buffer
    * Any packed slice header (e.g SVC NAL prefix) is attached to each slice buffer
    *
    * get_feedback information/stats is still only available after full frame
    * completion is signaled (e.g pipe_picture_desc::fence)
    *
    *  Driver reports support for this function used with different codecs/profiles
    *  in PIPE_VIDEO_CAP_ENC_SLICED_NOTIFICATIONS, frontend must check before using it.
    */
   void (*encode_bitstream_sliced)(struct pipe_video_codec *codec,
                                   struct pipe_video_buffer *source,
                                   unsigned num_slice_objects,
                                   struct pipe_resource **slice_destinations,
                                   struct pipe_fence_handle **slice_fences,
                                   void **feedback);


   /**
    * Once encode_bitstream_sliced::slice_fences[slice_idx] is signaled, use this function
    * to retrieve the slice size and offset for readback from encode_bitstream_sliced::slice_destinations[slice_idx]
    * As the slice may include other packed headers, a list of codec_unit_location_t elements is returned
    */
   void (*get_slice_bitstream_data)(struct pipe_video_codec *codec,
                                    void *feedback, /* corresponding to the encode_bitstream_sliced frame call */
                                    unsigned slice_idx, /* [0..max_slices_expected] */
                                    struct codec_unit_location_t *codec_unit_metadata,
                                    unsigned *codec_unit_metadata_count);

   /**
    * Perform post-process effect
    */
   int (*process_frame)(struct pipe_video_codec *codec,
                         struct pipe_video_buffer *source,
                         const struct pipe_vpp_desc *process_properties);

   /**
    * end decoding of the current frame
    * returns 0 on success
    */
   int (*end_frame)(struct pipe_video_codec *codec,
                    struct pipe_video_buffer *target,
                    struct pipe_picture_desc *picture);

   /**
    * flush any outstanding command buffers to the hardware
    * should be called before a video_buffer is acessed by the gallium frontend again
    */
   void (*flush)(struct pipe_video_codec *codec);

   /**
    * get encoder feedback
    */
   void (*get_feedback)(struct pipe_video_codec *codec,
                        void *feedback,
                        unsigned *size,
                        struct pipe_enc_feedback_metadata* metadata /* opt NULL */);

   /**
    * Wait for fence.
    *
    * Can be used to query the status of the previous job denoted by
    * 'fence' given 'timeout'.
    *
    * A pointer to a fence pointer can be passed to the codecs before the
    * end_frame vfunc and the codec should then be responsible for allocating a
    * fence on command stream submission.
    */
   int (*fence_wait)(struct pipe_video_codec *codec,
                     struct pipe_fence_handle *fence,
                     uint64_t timeout);

   /**
    * Destroy fence.
    */
   void (*destroy_fence)(struct pipe_video_codec *codec,
                         struct pipe_fence_handle *fence);

   /**
    * Gets the bitstream headers for a given pipe_picture_desc
    * of an encode operation
    *
    * User passes a buffer and its allocated size and
    * driver writes the bitstream headers in the buffer,
    * updating the size parameter as well.
    *
    * Returns 0 on success or an errno error code otherwise.
    * such as ENOMEM if the buffer passed was not big enough
    */
   int (*get_encode_headers)(struct pipe_video_codec *codec,
                              struct pipe_picture_desc *picture,
                              void* bitstream_buf,
                              unsigned *size);

   /**
    * Creates a DPB buffer used for a single reconstructed picture.
    */
   struct pipe_video_buffer *(*create_dpb_buffer)(struct pipe_video_codec *codec,
                                                  struct pipe_picture_desc *picture,
                                                  const struct pipe_video_buffer *templat);
};

/**
 * output for decoding / input for displaying
 */
struct pipe_video_buffer
{
   struct pipe_context *context;

   enum pipe_format buffer_format;
   unsigned width;
   unsigned height;
   bool interlaced;
   unsigned bind;
   unsigned flags;
   bool contiguous_planes;

   /**
    * destroy this video buffer
    */
   void (*destroy)(struct pipe_video_buffer *buffer);

   /**
    * get an individual resource for each plane,
    * only returns existing resources by reference
    */
   void (*get_resources)(struct pipe_video_buffer *buffer, struct pipe_resource **resources);

   /**
    * get an individual sampler view for each plane
    */
   struct pipe_sampler_view **(*get_sampler_view_planes)(struct pipe_video_buffer *buffer);

   /**
    * get an individual sampler view for each component
    */
   struct pipe_sampler_view **(*get_sampler_view_components)(struct pipe_video_buffer *buffer);

   /**
    * get an individual surfaces for each plane
    */
   struct pipe_surface *(*get_surfaces)(struct pipe_video_buffer *buffer);

   /*
    * auxiliary associated data
    */
   void *associated_data;

   /*
    * codec where the associated data came from
    */
   struct pipe_video_codec *codec;

   /*
    * destroy the associated data
    */
   void (*destroy_associated_data)(void *associated_data);

   /*
    * encoded frame statistics for this particular picture
    */
   void *statistics_data;
};

#ifdef __cplusplus
}
#endif

#endif /* PIPE_VIDEO_CONTEXT_H */
