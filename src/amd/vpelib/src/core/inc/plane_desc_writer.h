/* Copyright 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors: AMD
 *
 */

#pragma once

#include "vpe_types.h"
#ifdef __cplusplus
extern "C" {
#endif
struct plane_desc_header {
    int32_t nps0;
    int32_t npd0;
    int32_t nps1;
    int32_t npd1;
    int32_t subop;
};

struct plane_desc_src {
    uint8_t                      tmz;
    enum vpe_swizzle_mode_values swizzle;
    enum vpe_rotation_angle      rotation;
    uint32_t                     base_addr_lo;
    uint32_t                     base_addr_hi;
    uint16_t                     pitch;
    uint16_t                     viewport_x;
    uint16_t                     viewport_y;
    uint16_t                     viewport_w;
    uint16_t                     viewport_h;
    uint8_t                      elem_size;
};

struct plane_desc_dst {
    uint8_t                      tmz;
    enum vpe_swizzle_mode_values swizzle;
    enum vpe_mirror              mirror;
    uint32_t                     base_addr_lo;
    uint32_t                     base_addr_hi;
    uint16_t                     pitch;
    uint16_t                     viewport_x;
    uint16_t                     viewport_y;
    uint16_t                     viewport_w;
    uint16_t                     viewport_h;
    uint8_t                      elem_size;
};


struct plane_desc_writer {
    struct vpe_buf *buf; /**< store the current buf pointer */

    /* store the base addr of the currnet config
     * i.e. config header
     * it is always constructed in emb_buf
     */
    uint64_t        base_gpu_va;
    uint64_t        base_cpu_va;

    int32_t         num_src;
    int32_t         num_dst;
    enum vpe_status status;

    void (*init)(
        struct plane_desc_writer *writer, struct vpe_buf *buf, struct plane_desc_header *header);
    void (*add_source)(
        struct plane_desc_writer *writer, struct plane_desc_src *source, bool is_plane0);
    void (*add_destination)(
        struct plane_desc_writer *writer, struct plane_desc_dst *destination, bool is_plane0);
    void (*add_meta)(struct plane_desc_writer *writer, struct plane_desc_src *src);
};

#ifdef __cplusplus
}
#endif
