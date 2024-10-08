/*
 * Copyright © 2012 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 */

#ifndef REDUMP_H_
#define REDUMP_H_

#include <unistd.h>

enum rd_sect_type {
   RD_NONE,
   RD_TEST,           /* ascii text */
   RD_CMD,            /* ascii text */
   RD_GPUADDR,        /* u32 gpuaddr, u32 size */
   RD_CONTEXT,        /* raw dump */
   RD_CMDSTREAM,      /* raw dump */
   RD_CMDSTREAM_ADDR, /* gpu addr of cmdstream */
   RD_PARAM,          /* u32 param_type, u32 param_val, u32 bitlen */
   RD_FLUSH,          /* empty, clear previous params */
   RD_PROGRAM,        /* shader program, raw dump */
   RD_VERT_SHADER,
   RD_FRAG_SHADER,
   RD_BUFFER_CONTENTS,
   RD_GPU_ID,
   RD_CHIP_ID,
   RD_SHADER_LOG_BUFFER, /* Specifies buffer which has logs from shaders */
   RD_CP_LOG_BUFFER, /* Specifies buffer which has logs from CP */
   RD_WRBUFFER,     /* Specifies buffer which has data that needs to be written out to a file */
};

/* RD_PARAM types: */
enum rd_param_type {
   RD_PARAM_SURFACE_WIDTH,
   RD_PARAM_SURFACE_HEIGHT,
   RD_PARAM_SURFACE_PITCH,
   RD_PARAM_COLOR,
   RD_PARAM_BLIT_X,
   RD_PARAM_BLIT_Y,
   RD_PARAM_BLIT_WIDTH,
   RD_PARAM_BLIT_HEIGHT,
   RD_PARAM_BLIT_X2, /* BLIT_X + BLIT_WIDTH */
   RD_PARAM_BLIT_Y2, /* BLIT_Y + BLIT_WIDTH */
};

static inline void
rd_write_section(int fd, enum rd_sect_type type, const void *buf, int sz)
{
   write(fd, &type, 4);
   write(fd, &sz, 4);
   write(fd, buf, sz);
}

#endif /* REDUMP_H_ */
