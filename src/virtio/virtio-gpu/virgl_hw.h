/*
 * Copyright 2014, 2015 Red Hat.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef VIRGL_HW_H
#define VIRGL_HW_H

#include <stdint.h>

struct virgl_box {
	uint32_t x, y, z;
	uint32_t w, h, d;
};

/* formats known by the HW device - based on gallium subset */
enum virgl_formats {
   VIRGL_FORMAT_NONE                    = 0,
   VIRGL_FORMAT_B8G8R8A8_UNORM          = 1,
   VIRGL_FORMAT_B8G8R8X8_UNORM          = 2,
   VIRGL_FORMAT_A8R8G8B8_UNORM          = 3,
   VIRGL_FORMAT_X8R8G8B8_UNORM          = 4,
   VIRGL_FORMAT_B5G5R5A1_UNORM          = 5,
   VIRGL_FORMAT_B4G4R4A4_UNORM          = 6,
   VIRGL_FORMAT_B5G6R5_UNORM            = 7,
   VIRGL_FORMAT_R10G10B10A2_UNORM       = 8,
   VIRGL_FORMAT_L8_UNORM                = 9,    /**< ubyte luminance */
   VIRGL_FORMAT_A8_UNORM                = 10,   /**< ubyte alpha */
   VIRGL_FORMAT_I8_UNORM                = 11,
   VIRGL_FORMAT_L8A8_UNORM              = 12,   /**< ubyte alpha, luminance */
   VIRGL_FORMAT_L16_UNORM               = 13,   /**< ushort luminance */
   VIRGL_FORMAT_UYVY                    = 14,
   VIRGL_FORMAT_YUYV                    = 15,
   VIRGL_FORMAT_Z16_UNORM               = 16,
   VIRGL_FORMAT_Z32_UNORM               = 17,
   VIRGL_FORMAT_Z32_FLOAT               = 18,
   VIRGL_FORMAT_Z24_UNORM_S8_UINT       = 19,
   VIRGL_FORMAT_S8_UINT_Z24_UNORM       = 20,
   VIRGL_FORMAT_Z24X8_UNORM             = 21,
   VIRGL_FORMAT_X8Z24_UNORM             = 22,
   VIRGL_FORMAT_S8_UINT                 = 23,   /**< ubyte stencil */
   VIRGL_FORMAT_R64_FLOAT               = 24,
   VIRGL_FORMAT_R64G64_FLOAT            = 25,
   VIRGL_FORMAT_R64G64B64_FLOAT         = 26,
   VIRGL_FORMAT_R64G64B64A64_FLOAT      = 27,
   VIRGL_FORMAT_R32_FLOAT               = 28,
   VIRGL_FORMAT_R32G32_FLOAT            = 29,
   VIRGL_FORMAT_R32G32B32_FLOAT         = 30,
   VIRGL_FORMAT_R32G32B32A32_FLOAT      = 31,

   VIRGL_FORMAT_R32_UNORM               = 32,
   VIRGL_FORMAT_R32G32_UNORM            = 33,
   VIRGL_FORMAT_R32G32B32_UNORM         = 34,
   VIRGL_FORMAT_R32G32B32A32_UNORM      = 35,
   VIRGL_FORMAT_R32_USCALED             = 36,
   VIRGL_FORMAT_R32G32_USCALED          = 37,
   VIRGL_FORMAT_R32G32B32_USCALED       = 38,
   VIRGL_FORMAT_R32G32B32A32_USCALED    = 39,
   VIRGL_FORMAT_R32_SNORM               = 40,
   VIRGL_FORMAT_R32G32_SNORM            = 41,
   VIRGL_FORMAT_R32G32B32_SNORM         = 42,
   VIRGL_FORMAT_R32G32B32A32_SNORM      = 43,
   VIRGL_FORMAT_R32_SSCALED             = 44,
   VIRGL_FORMAT_R32G32_SSCALED          = 45,
   VIRGL_FORMAT_R32G32B32_SSCALED       = 46,
   VIRGL_FORMAT_R32G32B32A32_SSCALED    = 47,

   VIRGL_FORMAT_R16_UNORM               = 48,
   VIRGL_FORMAT_R16G16_UNORM            = 49,
   VIRGL_FORMAT_R16G16B16_UNORM         = 50,
   VIRGL_FORMAT_R16G16B16A16_UNORM      = 51,

   VIRGL_FORMAT_R16_USCALED             = 52,
   VIRGL_FORMAT_R16G16_USCALED          = 53,
   VIRGL_FORMAT_R16G16B16_USCALED       = 54,
   VIRGL_FORMAT_R16G16B16A16_USCALED    = 55,

   VIRGL_FORMAT_R16_SNORM               = 56,
   VIRGL_FORMAT_R16G16_SNORM            = 57,
   VIRGL_FORMAT_R16G16B16_SNORM         = 58,
   VIRGL_FORMAT_R16G16B16A16_SNORM      = 59,

   VIRGL_FORMAT_R16_SSCALED             = 60,
   VIRGL_FORMAT_R16G16_SSCALED          = 61,
   VIRGL_FORMAT_R16G16B16_SSCALED       = 62,
   VIRGL_FORMAT_R16G16B16A16_SSCALED    = 63,

   VIRGL_FORMAT_R8_UNORM                = 64,
   VIRGL_FORMAT_R8G8_UNORM              = 65,
   VIRGL_FORMAT_R8G8B8_UNORM            = 66,
   VIRGL_FORMAT_R8G8B8A8_UNORM          = 67,
   VIRGL_FORMAT_X8B8G8R8_UNORM          = 68,

   VIRGL_FORMAT_R8_USCALED              = 69,
   VIRGL_FORMAT_R8G8_USCALED            = 70,
   VIRGL_FORMAT_R8G8B8_USCALED          = 71,
   VIRGL_FORMAT_R8G8B8A8_USCALED        = 72,

   VIRGL_FORMAT_R8_SNORM                = 74,
   VIRGL_FORMAT_R8G8_SNORM              = 75,
   VIRGL_FORMAT_R8G8B8_SNORM            = 76,
   VIRGL_FORMAT_R8G8B8A8_SNORM          = 77,

   VIRGL_FORMAT_R8_SSCALED              = 82,
   VIRGL_FORMAT_R8G8_SSCALED            = 83,
   VIRGL_FORMAT_R8G8B8_SSCALED          = 84,
   VIRGL_FORMAT_R8G8B8A8_SSCALED        = 85,

   VIRGL_FORMAT_R32_FIXED               = 87,
   VIRGL_FORMAT_R32G32_FIXED            = 88,
   VIRGL_FORMAT_R32G32B32_FIXED         = 89,
   VIRGL_FORMAT_R32G32B32A32_FIXED      = 90,

   VIRGL_FORMAT_R16_FLOAT               = 91,
   VIRGL_FORMAT_R16G16_FLOAT            = 92,
   VIRGL_FORMAT_R16G16B16_FLOAT         = 93,
   VIRGL_FORMAT_R16G16B16A16_FLOAT      = 94,

   VIRGL_FORMAT_L8_SRGB                 = 95,
   VIRGL_FORMAT_L8A8_SRGB               = 96,
   VIRGL_FORMAT_R8G8B8_SRGB             = 97,
   VIRGL_FORMAT_A8B8G8R8_SRGB           = 98,
   VIRGL_FORMAT_X8B8G8R8_SRGB           = 99,
   VIRGL_FORMAT_B8G8R8A8_SRGB           = 100,
   VIRGL_FORMAT_B8G8R8X8_SRGB           = 101,
   VIRGL_FORMAT_A8R8G8B8_SRGB           = 102,
   VIRGL_FORMAT_X8R8G8B8_SRGB           = 103,
   VIRGL_FORMAT_R8G8B8A8_SRGB           = 104,

   /* compressed formats */
   VIRGL_FORMAT_DXT1_RGB                = 105,
   VIRGL_FORMAT_DXT1_RGBA               = 106,
   VIRGL_FORMAT_DXT3_RGBA               = 107,
   VIRGL_FORMAT_DXT5_RGBA               = 108,

   /* sRGB, compressed */
   VIRGL_FORMAT_DXT1_SRGB               = 109,
   VIRGL_FORMAT_DXT1_SRGBA              = 110,
   VIRGL_FORMAT_DXT3_SRGBA              = 111,
   VIRGL_FORMAT_DXT5_SRGBA              = 112,

   /* rgtc compressed */
   VIRGL_FORMAT_RGTC1_UNORM             = 113,
   VIRGL_FORMAT_RGTC1_SNORM             = 114,
   VIRGL_FORMAT_RGTC2_UNORM             = 115,
   VIRGL_FORMAT_RGTC2_SNORM             = 116,

   VIRGL_FORMAT_R8G8_B8G8_UNORM         = 117,
   VIRGL_FORMAT_G8R8_G8B8_UNORM         = 118,

   VIRGL_FORMAT_R8SG8SB8UX8U_NORM       = 119,
   VIRGL_FORMAT_R5SG5SB6U_NORM          = 120,

   VIRGL_FORMAT_A8B8G8R8_UNORM          = 121,
   VIRGL_FORMAT_B5G5R5X1_UNORM          = 122,
   VIRGL_FORMAT_R10G10B10A2_USCALED     = 123,
   VIRGL_FORMAT_R11G11B10_FLOAT         = 124,
   VIRGL_FORMAT_R9G9B9E5_FLOAT          = 125,
   VIRGL_FORMAT_Z32_FLOAT_S8X24_UINT    = 126,
   VIRGL_FORMAT_R1_UNORM                = 127,
   VIRGL_FORMAT_R10G10B10X2_USCALED     = 128,
   VIRGL_FORMAT_R10G10B10X2_SNORM       = 129,

   VIRGL_FORMAT_L4A4_UNORM              = 130,
   VIRGL_FORMAT_B10G10R10A2_UNORM       = 131,
   VIRGL_FORMAT_R10SG10SB10SA2U_NORM    = 132,
   VIRGL_FORMAT_R8G8Bx_SNORM            = 133,
   VIRGL_FORMAT_R8G8B8X8_UNORM          = 134,
   VIRGL_FORMAT_B4G4R4X4_UNORM          = 135,
   VIRGL_FORMAT_X24S8_UINT              = 136,
   VIRGL_FORMAT_S8X24_UINT              = 137,
   VIRGL_FORMAT_X32_S8X24_UINT          = 138,
   VIRGL_FORMAT_B2G3R3_UNORM            = 139,

   VIRGL_FORMAT_L16A16_UNORM            = 140,
   VIRGL_FORMAT_A16_UNORM               = 141,
   VIRGL_FORMAT_I16_UNORM               = 142,

   VIRGL_FORMAT_LATC1_UNORM             = 143,
   VIRGL_FORMAT_LATC1_SNORM             = 144,
   VIRGL_FORMAT_LATC2_UNORM             = 145,
   VIRGL_FORMAT_LATC2_SNORM             = 146,

   VIRGL_FORMAT_A8_SNORM                = 147,
   VIRGL_FORMAT_L8_SNORM                = 148,
   VIRGL_FORMAT_L8A8_SNORM              = 149,
   VIRGL_FORMAT_I8_SNORM                = 150,
   VIRGL_FORMAT_A16_SNORM               = 151,
   VIRGL_FORMAT_L16_SNORM               = 152,
   VIRGL_FORMAT_L16A16_SNORM            = 153,
   VIRGL_FORMAT_I16_SNORM               = 154,

   VIRGL_FORMAT_A16_FLOAT               = 155,
   VIRGL_FORMAT_L16_FLOAT               = 156,
   VIRGL_FORMAT_L16A16_FLOAT            = 157,
   VIRGL_FORMAT_I16_FLOAT               = 158,
   VIRGL_FORMAT_A32_FLOAT               = 159,
   VIRGL_FORMAT_L32_FLOAT               = 160,
   VIRGL_FORMAT_L32A32_FLOAT            = 161,
   VIRGL_FORMAT_I32_FLOAT               = 162,

   VIRGL_FORMAT_YV12                    = 163,
   VIRGL_FORMAT_YV16                    = 164,
   VIRGL_FORMAT_IYUV                    = 165,  /**< aka I420 */
   VIRGL_FORMAT_NV12                    = 166,
   VIRGL_FORMAT_NV21                    = 167,

   VIRGL_FORMAT_A4R4_UNORM              = 168,
   VIRGL_FORMAT_R4A4_UNORM              = 169,
   VIRGL_FORMAT_R8A8_UNORM              = 170,
   VIRGL_FORMAT_A8R8_UNORM              = 171,

   VIRGL_FORMAT_R10G10B10A2_SSCALED     = 172,
   VIRGL_FORMAT_R10G10B10A2_SNORM       = 173,
   VIRGL_FORMAT_B10G10R10A2_USCALED     = 174,
   VIRGL_FORMAT_B10G10R10A2_SSCALED     = 175,
   VIRGL_FORMAT_B10G10R10A2_SNORM       = 176,

   VIRGL_FORMAT_R8_UINT                 = 177,
   VIRGL_FORMAT_R8G8_UINT               = 178,
   VIRGL_FORMAT_R8G8B8_UINT             = 179,
   VIRGL_FORMAT_R8G8B8A8_UINT           = 180,

   VIRGL_FORMAT_R8_SINT                 = 181,
   VIRGL_FORMAT_R8G8_SINT               = 182,
   VIRGL_FORMAT_R8G8B8_SINT             = 183,
   VIRGL_FORMAT_R8G8B8A8_SINT           = 184,

   VIRGL_FORMAT_R16_UINT                = 185,
   VIRGL_FORMAT_R16G16_UINT             = 186,
   VIRGL_FORMAT_R16G16B16_UINT          = 187,
   VIRGL_FORMAT_R16G16B16A16_UINT       = 188,

   VIRGL_FORMAT_R16_SINT                = 189,
   VIRGL_FORMAT_R16G16_SINT             = 190,
   VIRGL_FORMAT_R16G16B16_SINT          = 191,
   VIRGL_FORMAT_R16G16B16A16_SINT       = 192,
   VIRGL_FORMAT_R32_UINT                = 193,
   VIRGL_FORMAT_R32G32_UINT             = 194,
   VIRGL_FORMAT_R32G32B32_UINT          = 195,
   VIRGL_FORMAT_R32G32B32A32_UINT       = 196,

   VIRGL_FORMAT_R32_SINT                = 197,
   VIRGL_FORMAT_R32G32_SINT             = 198,
   VIRGL_FORMAT_R32G32B32_SINT          = 199,
   VIRGL_FORMAT_R32G32B32A32_SINT       = 200,

   VIRGL_FORMAT_A8_UINT                 = 201,
   VIRGL_FORMAT_I8_UINT                 = 202,
   VIRGL_FORMAT_L8_UINT                 = 203,
   VIRGL_FORMAT_L8A8_UINT               = 204,

   VIRGL_FORMAT_A8_SINT                 = 205,
   VIRGL_FORMAT_I8_SINT                 = 206,
   VIRGL_FORMAT_L8_SINT                 = 207,
   VIRGL_FORMAT_L8A8_SINT               = 208,

   VIRGL_FORMAT_A16_UINT                = 209,
   VIRGL_FORMAT_I16_UINT                = 210,
   VIRGL_FORMAT_L16_UINT                = 211,
   VIRGL_FORMAT_L16A16_UINT             = 212,

   VIRGL_FORMAT_A16_SINT                = 213,
   VIRGL_FORMAT_I16_SINT                = 214,
   VIRGL_FORMAT_L16_SINT                = 215,
   VIRGL_FORMAT_L16A16_SINT             = 216,

   VIRGL_FORMAT_A32_UINT                = 217,
   VIRGL_FORMAT_I32_UINT                = 218,
   VIRGL_FORMAT_L32_UINT                = 219,
   VIRGL_FORMAT_L32A32_UINT             = 220,

   VIRGL_FORMAT_A32_SINT                = 221,
   VIRGL_FORMAT_I32_SINT                = 222,
   VIRGL_FORMAT_L32_SINT                = 223,
   VIRGL_FORMAT_L32A32_SINT             = 224,

   VIRGL_FORMAT_B10G10R10A2_UINT        = 225,
   VIRGL_FORMAT_ETC1_RGB8               = 226,
   VIRGL_FORMAT_R8G8_R8B8_UNORM         = 227,
   VIRGL_FORMAT_G8R8_B8R8_UNORM         = 228,
   VIRGL_FORMAT_R8G8B8X8_SNORM          = 229,

   VIRGL_FORMAT_R8G8B8X8_SRGB           = 230,

   VIRGL_FORMAT_R8G8B8X8_UINT           = 231,
   VIRGL_FORMAT_R8G8B8X8_SINT           = 232,
   VIRGL_FORMAT_B10G10R10X2_UNORM       = 233,
   VIRGL_FORMAT_R16G16B16X16_UNORM      = 234,
   VIRGL_FORMAT_R16G16B16X16_SNORM      = 235,
   VIRGL_FORMAT_R16G16B16X16_FLOAT      = 236,
   VIRGL_FORMAT_R16G16B16X16_UINT       = 237,
   VIRGL_FORMAT_R16G16B16X16_SINT       = 238,
   VIRGL_FORMAT_R32G32B32X32_FLOAT      = 239,
   VIRGL_FORMAT_R32G32B32X32_UINT       = 240,
   VIRGL_FORMAT_R32G32B32X32_SINT       = 241,
   VIRGL_FORMAT_R8A8_SNORM              = 242,
   VIRGL_FORMAT_R16A16_UNORM            = 243,
   VIRGL_FORMAT_R16A16_SNORM            = 244,
   VIRGL_FORMAT_R16A16_FLOAT            = 245,
   VIRGL_FORMAT_R32A32_FLOAT            = 246,
   VIRGL_FORMAT_R8A8_UINT               = 247,
   VIRGL_FORMAT_R8A8_SINT               = 248,
   VIRGL_FORMAT_R16A16_UINT             = 249,
   VIRGL_FORMAT_R16A16_SINT             = 250,
   VIRGL_FORMAT_R32A32_UINT             = 251,
   VIRGL_FORMAT_R32A32_SINT             = 252,

   VIRGL_FORMAT_R10G10B10A2_UINT        = 253,
   VIRGL_FORMAT_B5G6R5_SRGB             = 254,

   VIRGL_FORMAT_BPTC_RGBA_UNORM         = 255,
   VIRGL_FORMAT_BPTC_SRGBA              = 256,
   VIRGL_FORMAT_BPTC_RGB_FLOAT          = 257,
   VIRGL_FORMAT_BPTC_RGB_UFLOAT         = 258,

   /*VIRGL_FORMAT_A8L8_UNORM              = 259, Removed and no user */
   /*VIRGL_FORMAT_A8L8_SNORM              = 260, Removed and no user */
   /*VIRGL_FORMAT_A8L8_SRGB               = 261, Removed and no user */
   /*VIRGL_FORMAT_A16L16_UNORM            = 262, Removed and no user */

   VIRGL_FORMAT_G8R8_UNORM              = 263,
   VIRGL_FORMAT_G8R8_SNORM              = 264,
   VIRGL_FORMAT_G16R16_UNORM            = 265,
   VIRGL_FORMAT_G16R16_SNORM            = 266,
   VIRGL_FORMAT_A8B8G8R8_SNORM          = 267,

   VIRGL_FORMAT_X8B8G8R8_SNORM          = 268,

   /* etc2 compressed */
   VIRGL_FORMAT_ETC2_RGB8               = 269,
   VIRGL_FORMAT_ETC2_SRGB8              = 270,
   VIRGL_FORMAT_ETC2_RGB8A1             = 271,
   VIRGL_FORMAT_ETC2_SRGB8A1            = 272,
   VIRGL_FORMAT_ETC2_RGBA8              = 273,
   VIRGL_FORMAT_ETC2_SRGBA8             = 274,
   VIRGL_FORMAT_ETC2_R11_UNORM          = 275,
   VIRGL_FORMAT_ETC2_R11_SNORM          = 276,
   VIRGL_FORMAT_ETC2_RG11_UNORM         = 277,
   VIRGL_FORMAT_ETC2_RG11_SNORM         = 278,

    /* astc compressed */
   VIRGL_FORMAT_ASTC_4x4                = 279,
   VIRGL_FORMAT_ASTC_5x4                = 280,
   VIRGL_FORMAT_ASTC_5x5                = 281,
   VIRGL_FORMAT_ASTC_6x5                = 282,
   VIRGL_FORMAT_ASTC_6x6                = 283,
   VIRGL_FORMAT_ASTC_8x5                = 284,
   VIRGL_FORMAT_ASTC_8x6                = 285,
   VIRGL_FORMAT_ASTC_8x8                = 286,
   VIRGL_FORMAT_ASTC_10x5               = 287,
   VIRGL_FORMAT_ASTC_10x6               = 288,
   VIRGL_FORMAT_ASTC_10x8               = 289,
   VIRGL_FORMAT_ASTC_10x10              = 290,
   VIRGL_FORMAT_ASTC_12x10              = 291,
   VIRGL_FORMAT_ASTC_12x12              = 292,
   VIRGL_FORMAT_ASTC_4x4_SRGB           = 293,
   VIRGL_FORMAT_ASTC_5x4_SRGB           = 294,
   VIRGL_FORMAT_ASTC_5x5_SRGB           = 295,
   VIRGL_FORMAT_ASTC_6x5_SRGB           = 296,
   VIRGL_FORMAT_ASTC_6x6_SRGB           = 297,
   VIRGL_FORMAT_ASTC_8x5_SRGB           = 298,
   VIRGL_FORMAT_ASTC_8x6_SRGB           = 299,
   VIRGL_FORMAT_ASTC_8x8_SRGB           = 300,
   VIRGL_FORMAT_ASTC_10x5_SRGB          = 301,
   VIRGL_FORMAT_ASTC_10x6_SRGB          = 302,
   VIRGL_FORMAT_ASTC_10x8_SRGB          = 303,
   VIRGL_FORMAT_ASTC_10x10_SRGB         = 304,
   VIRGL_FORMAT_ASTC_12x10_SRGB         = 305,
   VIRGL_FORMAT_ASTC_12x12_SRGB         = 306,

   VIRGL_FORMAT_R10G10B10X2_UNORM       = 308,
   VIRGL_FORMAT_A4B4G4R4_UNORM          = 311,

   VIRGL_FORMAT_R8_SRGB                 = 312,
   VIRGL_FORMAT_R8G8_SRGB               = 313,

   VIRGL_FORMAT_P010                    = 314,
   VIRGL_FORMAT_P012                    = 315,
   VIRGL_FORMAT_P016                    = 316,

   VIRGL_FORMAT_B8G8R8_UNORM            = 317,
   VIRGL_FORMAT_R3G3B2_UNORM            = 318,
   VIRGL_FORMAT_R4G4B4A4_UNORM          = 319,
   VIRGL_FORMAT_R5G5B5A1_UNORM          = 320,
   VIRGL_FORMAT_R5G6B5_UNORM            = 321,

   VIRGL_FORMAT_Y8_400_UNORM            = 322,
   VIRGL_FORMAT_Y8_U8_V8_444_UNORM      = 323,
   VIRGL_FORMAT_Y8_U8_V8_422_UNORM      = 324,
   VIRGL_FORMAT_NV16                    = 325, /* aka Y8_U8V8_422_UNORM */
   VIRGL_FORMAT_Y8_UNORM                = 326,
   VIRGL_FORMAT_YVYU                    = 327,
   VIRGL_FORMAT_Z16_UNORM_S8_UINT       = 328,
   VIRGL_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8 = 329,
   VIRGL_FORMAT_A1B5G5R5_UINT           = 330,
   VIRGL_FORMAT_A1B5G5R5_UNORM          = 331,
   VIRGL_FORMAT_A1R5G5B5_UINT           = 332,
   VIRGL_FORMAT_A1R5G5B5_UNORM          = 333,
   VIRGL_FORMAT_A2B10G10R10_UINT        = 334,
   VIRGL_FORMAT_A2B10G10R10_UNORM       = 335,
   VIRGL_FORMAT_A2R10G10B10_UINT        = 336,
   VIRGL_FORMAT_A2R10G10B10_UNORM       = 337,
   VIRGL_FORMAT_A4B4G4R4_UINT           = 338,
   VIRGL_FORMAT_A4R4G4B4_UINT           = 339,
   VIRGL_FORMAT_A4R4G4B4_UNORM          = 340,
   VIRGL_FORMAT_A8B8G8R8_SINT           = 341,
   VIRGL_FORMAT_A8B8G8R8_SSCALED        = 342,
   VIRGL_FORMAT_A8B8G8R8_UINT           = 343,
   VIRGL_FORMAT_A8B8G8R8_USCALED        = 344,
   VIRGL_FORMAT_A8R8G8B8_SINT           = 345,
   VIRGL_FORMAT_A8R8G8B8_SNORM          = 346,
   VIRGL_FORMAT_A8R8G8B8_UINT           = 347,
   VIRGL_FORMAT_ASTC_3x3x3              = 348,
   VIRGL_FORMAT_ASTC_3x3x3_SRGB         = 349,
   VIRGL_FORMAT_ASTC_4x3x3              = 350,
   VIRGL_FORMAT_ASTC_4x3x3_SRGB         = 351,
   VIRGL_FORMAT_ASTC_4x4x3              = 352,
   VIRGL_FORMAT_ASTC_4x4x3_SRGB         = 353,
   VIRGL_FORMAT_ASTC_4x4x4              = 354,
   VIRGL_FORMAT_ASTC_4x4x4_SRGB         = 355,
   VIRGL_FORMAT_ASTC_5x4x4              = 356,
   VIRGL_FORMAT_ASTC_5x4x4_SRGB         = 357,
   VIRGL_FORMAT_ASTC_5x5x4              = 358,
   VIRGL_FORMAT_ASTC_5x5x4_SRGB         = 359,
   VIRGL_FORMAT_ASTC_5x5x5              = 360,
   VIRGL_FORMAT_ASTC_5x5x5_SRGB         = 361,
   VIRGL_FORMAT_ASTC_6x5x5              = 362,
   VIRGL_FORMAT_ASTC_6x5x5_SRGB         = 363,
   VIRGL_FORMAT_ASTC_6x6x5              = 364,
   VIRGL_FORMAT_ASTC_6x6x5_SRGB         = 365,
   VIRGL_FORMAT_ASTC_6x6x6              = 366,
   VIRGL_FORMAT_ASTC_6x6x6_SRGB         = 367,
   VIRGL_FORMAT_ATC_RGB                 = 368,
   VIRGL_FORMAT_ATC_RGBA_EXPLICIT       = 369,
   VIRGL_FORMAT_ATC_RGBA_INTERPOLATED   = 370,
   VIRGL_FORMAT_AYUV                    = 371,
   VIRGL_FORMAT_B10G10R10A2_SINT        = 372,
   VIRGL_FORMAT_B10G10R10X2_SINT        = 373,
   VIRGL_FORMAT_B10G10R10X2_SNORM       = 374,
   VIRGL_FORMAT_B2G3R3_UINT             = 375,
   VIRGL_FORMAT_B4G4R4A4_UINT           = 376,
   VIRGL_FORMAT_B5G5R5A1_UINT           = 377,
   VIRGL_FORMAT_B5G6R5_UINT             = 378,
   VIRGL_FORMAT_B8G8R8A8_SINT           = 379,
   VIRGL_FORMAT_B8G8R8A8_SNORM          = 380,
   VIRGL_FORMAT_B8G8R8A8_SSCALED        = 381,
   VIRGL_FORMAT_B8G8R8A8_UINT           = 382,
   VIRGL_FORMAT_B8G8R8A8_USCALED        = 383,
   VIRGL_FORMAT_B8G8_R8G8_UNORM         = 384,
   VIRGL_FORMAT_B8G8R8_SINT             = 385,
   VIRGL_FORMAT_B8G8R8_SNORM            = 386,
   VIRGL_FORMAT_B8G8R8_SRGB             = 387,
   VIRGL_FORMAT_B8G8R8_SSCALED          = 388,
   VIRGL_FORMAT_B8G8R8_UINT             = 389,
   VIRGL_FORMAT_B8G8R8_USCALED          = 390,
   VIRGL_FORMAT_B8G8R8X8_SINT           = 391,
   VIRGL_FORMAT_B8G8R8X8_SNORM          = 392,
   VIRGL_FORMAT_B8G8R8X8_UINT           = 393,
   VIRGL_FORMAT_B8R8_G8R8_UNORM         = 394,
   VIRGL_FORMAT_FXT1_RGB                = 395,
   VIRGL_FORMAT_FXT1_RGBA               = 396,
   VIRGL_FORMAT_G16R16_SINT             = 397,
   VIRGL_FORMAT_G8B8_G8R8_UNORM         = 398,
   VIRGL_FORMAT_G8_B8_R8_420_UNORM      = 399,
   VIRGL_FORMAT_G8_B8R8_420_UNORM       = 400,
   VIRGL_FORMAT_G8R8_SINT               = 401,
   VIRGL_FORMAT_P030                    = 402,
   VIRGL_FORMAT_R10G10B10A2_SINT        = 403,
   VIRGL_FORMAT_R10G10B10X2_SINT        = 404,
   VIRGL_FORMAT_R3G3B2_UINT             = 405,
   VIRGL_FORMAT_R4G4B4A4_UINT           = 406,
   VIRGL_FORMAT_R4G4B4X4_UNORM          = 407,
   VIRGL_FORMAT_R5G5B5A1_UINT           = 408,
   VIRGL_FORMAT_R5G5B5X1_UNORM          = 409,
   VIRGL_FORMAT_R5G6B5_SRGB             = 410,
   VIRGL_FORMAT_R5G6B5_UINT             = 411,
   VIRGL_FORMAT_R64G64B64A64_SINT       = 412,
   VIRGL_FORMAT_R64G64B64A64_UINT       = 413,
   VIRGL_FORMAT_R64G64B64_SINT          = 414,
   VIRGL_FORMAT_R64G64B64_UINT          = 415,
   VIRGL_FORMAT_R64G64_SINT             = 416,
   VIRGL_FORMAT_R64G64_UINT             = 417,
   VIRGL_FORMAT_R64_SINT                = 418,
   VIRGL_FORMAT_R64_UINT                = 419, /**< raw doubles (ARB_vertex_attrib_64bit) */
   VIRGL_FORMAT_R8_B8_G8_420_UNORM      = 420,
   VIRGL_FORMAT_R8_B8G8_420_UNORM       = 421,
   VIRGL_FORMAT_R8B8_R8G8_UNORM         = 422,
   VIRGL_FORMAT_R8_G8_B8_420_UNORM      = 423,
   VIRGL_FORMAT_R8_G8B8_420_UNORM       = 424,
   VIRGL_FORMAT_R8_G8_B8_UNORM          = 425,
   VIRGL_FORMAT_VYUY                    = 426,
   VIRGL_FORMAT_X1B5G5R5_UNORM          = 427,
   VIRGL_FORMAT_X1R5G5B5_UNORM          = 428,
   VIRGL_FORMAT_XYUV                    = 429,
   VIRGL_FORMAT_X8B8G8R8_SINT           = 430,
   VIRGL_FORMAT_X8R8G8B8_SINT           = 431,
   VIRGL_FORMAT_X8R8G8B8_SNORM          = 432,
   VIRGL_FORMAT_Y16_U16_V16_420_UNORM   = 433,
   VIRGL_FORMAT_Y16_U16_V16_422_UNORM   = 434,
   VIRGL_FORMAT_Y16_U16V16_422_UNORM    = 435,
   VIRGL_FORMAT_Y16_U16_V16_444_UNORM   = 436,
   VIRGL_FORMAT_Y210                    = 437,
   VIRGL_FORMAT_Y212                    = 438,
   VIRGL_FORMAT_Y216                    = 439,
   VIRGL_FORMAT_Y410                    = 440,
   VIRGL_FORMAT_Y412                    = 441,
   VIRGL_FORMAT_Y416                    = 442,
   VIRGL_FORMAT_NV15                    = 443,
   VIRGL_FORMAT_NV20                    = 444,
   VIRGL_FORMAT_Y8_U8_V8_440_UNORM      = 445,
   VIRGL_FORMAT_R10_G10B10_420_UNORM    = 446,
   VIRGL_FORMAT_R10_G10B10_422_UNORM    = 447,
   VIRGL_FORMAT_X6G10_X6B10X6R10_420_UNORM = 448,
   VIRGL_FORMAT_X4G12_X4B12X4R12_420_UNORM = 449,
   VIRGL_FORMAT_X6R10_UNORM             = 450,
   VIRGL_FORMAT_X6R10X6G10_UNORM        = 451,
   VIRGL_FORMAT_X4R12_UNORM             = 452,
   VIRGL_FORMAT_X4R12X4G12_UNORM        = 453,
   VIRGL_FORMAT_R8_G8B8_422_UNORM       = 454,
   VIRGL_FORMAT_R8_B8G8_422_UNORM       = 455,
   VIRGL_FORMAT_G8_B8R8_422_UNORM       = 456,
   VIRGL_FORMAT_ASTC_4x4_FLOAT          = 457,
   VIRGL_FORMAT_ASTC_5x4_FLOAT          = 458,
   VIRGL_FORMAT_ASTC_5x5_FLOAT          = 459,
   VIRGL_FORMAT_ASTC_6x5_FLOAT          = 460,
   VIRGL_FORMAT_ASTC_6x6_FLOAT          = 461,
   VIRGL_FORMAT_ASTC_8x5_FLOAT          = 462,
   VIRGL_FORMAT_ASTC_8x6_FLOAT          = 463,
   VIRGL_FORMAT_ASTC_8x8_FLOAT          = 464,
   VIRGL_FORMAT_ASTC_10x5_FLOAT         = 465,
   VIRGL_FORMAT_ASTC_10x6_FLOAT         = 466,
   VIRGL_FORMAT_ASTC_10x8_FLOAT         = 467,
   VIRGL_FORMAT_ASTC_10x10_FLOAT        = 468,
   VIRGL_FORMAT_ASTC_12x10_FLOAT        = 469,
   VIRGL_FORMAT_ASTC_12x12_FLOAT        = 470,
   VIRGL_FORMAT_Y8U8V8_420_UNORM_PACKED = 471,
   VIRGL_FORMAT_Y10U10V10_420_UNORM_PACKED = 472,
   VIRGL_FORMAT_R8G8B8_420_UNORM_PACKED = 473,
   VIRGL_FORMAT_R10G10B10_420_UNORM_PACKED = 474,
   VIRGL_FORMAT_Y10X6_U10X6_V10X6_420_UNORM = 475,
   VIRGL_FORMAT_Y10X6_U10X6_V10X6_422_UNORM = 476,
   VIRGL_FORMAT_Y10X6_U10X6_V10X6_444_UNORM = 477,
   VIRGL_FORMAT_Y12X4_U12X4_V12X4_420_UNORM = 478,
   VIRGL_FORMAT_Y12X4_U12X4_V12X4_422_UNORM = 479,
   VIRGL_FORMAT_Y12X4_U12X4_V12X4_444_UNORM = 480,

   VIRGL_FORMAT_MAX /* = PIPE_FORMAT_COUNT */,

   /* Below formats must not be used in the guest. */
   VIRGL_FORMAT_B8G8R8X8_UNORM_EMULATED,
   VIRGL_FORMAT_B8G8R8A8_UNORM_EMULATED,
   VIRGL_FORMAT_MAX_EXTENDED
};

/* These are used by the capability_bits field in virgl_caps_v2. */
#define VIRGL_CAP_NONE 0u
#define VIRGL_CAP_TGSI_INVARIANT       (1u << 0)
#define VIRGL_CAP_TEXTURE_VIEW         (1u << 1)
#define VIRGL_CAP_SET_MIN_SAMPLES      (1u << 2)
#define VIRGL_CAP_COPY_IMAGE           (1u << 3)
#define VIRGL_CAP_TGSI_PRECISE         (1u << 4)
#define VIRGL_CAP_TXQS                 (1u << 5)
#define VIRGL_CAP_MEMORY_BARRIER       (1u << 6)
#define VIRGL_CAP_COMPUTE_SHADER       (1u << 7)
#define VIRGL_CAP_FB_NO_ATTACH         (1u << 8)
#define VIRGL_CAP_ROBUST_BUFFER_ACCESS (1u << 9)
#define VIRGL_CAP_TGSI_FBFETCH         (1u << 10)
#define VIRGL_CAP_SHADER_CLOCK         (1u << 11)
#define VIRGL_CAP_TEXTURE_BARRIER      (1u << 12)
#define VIRGL_CAP_TGSI_COMPONENTS      (1u << 13)
#define VIRGL_CAP_GUEST_MAY_INIT_LOG   (1u << 14)
#define VIRGL_CAP_SRGB_WRITE_CONTROL   (1u << 15)
#define VIRGL_CAP_QBO                  (1u << 16)
#define VIRGL_CAP_TRANSFER             (1u << 17)
#define VIRGL_CAP_FBO_MIXED_COLOR_FORMATS  (1u << 18)
#define VIRGL_CAP_HOST_IS_GLES         (1u << 19)
#define VIRGL_CAP_BIND_COMMAND_ARGS    (1u << 20)
#define VIRGL_CAP_MULTI_DRAW_INDIRECT  (1u << 21)
#define VIRGL_CAP_INDIRECT_PARAMS      (1u << 22)
#define VIRGL_CAP_TRANSFORM_FEEDBACK3  (1u << 23)
#define VIRGL_CAP_3D_ASTC              (1u << 24)
#define VIRGL_CAP_INDIRECT_INPUT_ADDR  (1u << 25)
#define VIRGL_CAP_COPY_TRANSFER        (1u << 26)
#define VIRGL_CAP_CLIP_HALFZ           (1u << 27)
#define VIRGL_CAP_APP_TWEAK_SUPPORT    (1u << 28)
#define VIRGL_CAP_BGRA_SRGB_IS_EMULATED (1u << 29)
#define VIRGL_CAP_CLEAR_TEXTURE        (1u << 30)
#define VIRGL_CAP_ARB_BUFFER_STORAGE   (1u << 31)

// Legacy alias
#define VIRGL_CAP_FAKE_FP64            VIRGL_CAP_HOST_IS_GLES

/* These are used by the capability_bits_v2 field in virgl_caps_v2. */
#define VIRGL_CAP_V2_BLEND_EQUATION       (1u << 0)
#define VIRGL_CAP_V2_UNTYPED_RESOURCE     (1u << 1)
#define VIRGL_CAP_V2_VIDEO_MEMORY         (1u << 2)
#define VIRGL_CAP_V2_MEMINFO              (1u << 3)
#define VIRGL_CAP_V2_STRING_MARKER        (1u << 4)
#define VIRGL_CAP_V2_DIFFERENT_GPU        (1u << 5)
#define VIRGL_CAP_V2_IMPLICIT_MSAA        (1u << 6)
#define VIRGL_CAP_V2_COPY_TRANSFER_BOTH_DIRECTIONS (1u << 7)
#define VIRGL_CAP_V2_SCANOUT_USES_GBM     (1u << 8)
#define VIRGL_CAP_V2_SSO                  (1u << 9)
#define VIRGL_CAP_V2_TEXTURE_SHADOW_LOD   (1u << 10)
#define VIRGL_CAP_V2_VS_VERTEX_LAYER      (1u << 11)
#define VIRGL_CAP_V2_VS_VIEWPORT_INDEX    (1u << 12)
#define VIRGL_CAP_V2_PIPELINE_STATISTICS_QUERY (1u << 13)
#define VIRGL_CAP_V2_DRAW_PARAMETERS      (1u << 14)
#define VIRGL_CAP_V2_GROUP_VOTE           (1u << 15)
#define VIRGL_CAP_V2_MIRROR_CLAMP_TO_EDGE (1u << 16)
#define VIRGL_CAP_V2_MIRROR_CLAMP         (1u << 17)

/* virgl bind flags - these are compatible with mesa 10.5 gallium.
 * but are fixed, no other should be passed to virgl either.
 */
#define VIRGL_BIND_DEPTH_STENCIL (1u << 0)
#define VIRGL_BIND_RENDER_TARGET (1u << 1)
#define VIRGL_BIND_SAMPLER_VIEW  (1u << 3)
#define VIRGL_BIND_VERTEX_BUFFER (1u << 4)
#define VIRGL_BIND_INDEX_BUFFER  (1u << 5)
#define VIRGL_BIND_CONSTANT_BUFFER (1u << 6)
#define VIRGL_BIND_DISPLAY_TARGET (1u << 7)
#define VIRGL_BIND_COMMAND_ARGS  (1u << 8)
#define VIRGL_BIND_STREAM_OUTPUT (1u << 11)
#define VIRGL_BIND_SHADER_BUFFER (1u << 14)
#define VIRGL_BIND_QUERY_BUFFER  (1u << 15)
#define VIRGL_BIND_CURSOR        (1u << 16)
#define VIRGL_BIND_CUSTOM        (1u << 17)
#define VIRGL_BIND_SCANOUT       (1u << 18)
/* Used for buffers that are backed by guest storage and
 * are only read by the host.
 */
#define VIRGL_BIND_STAGING       (1u << 19)
#define VIRGL_BIND_SHARED        (1u << 20)

#define VIRGL_BIND_PREFER_EMULATED_BGRA  (1u << 21) /* non-functional */

#define VIRGL_BIND_LINEAR (1u << 22)

#define VIRGL_BIND_SHARED_SUBFLAGS (0xffu << 24)

#define VIRGL_BIND_MINIGBM_CAMERA_WRITE (1u << 24)
#define VIRGL_BIND_MINIGBM_CAMERA_READ (1u << 25)
#define VIRGL_BIND_MINIGBM_HW_VIDEO_DECODER (1u << 26)
#define VIRGL_BIND_MINIGBM_HW_VIDEO_ENCODER (1u << 27)
#define VIRGL_BIND_MINIGBM_SW_READ_OFTEN (1u << 28)
#define VIRGL_BIND_MINIGBM_SW_READ_RARELY (1u << 29)
#define VIRGL_BIND_MINIGBM_SW_WRITE_OFTEN (1u << 30)
#define VIRGL_BIND_MINIGBM_SW_WRITE_RARELY (1u << 31)
#define VIRGL_BIND_MINIGBM_PROTECTED (0xfu << 28) // Mutually exclusive with SW_ flags

struct virgl_caps_bool_set1 {
        unsigned indep_blend_enable:1;
        unsigned indep_blend_func:1;
        unsigned cube_map_array:1;
        unsigned shader_stencil_export:1;
        unsigned conditional_render:1;
        unsigned start_instance:1;
        unsigned primitive_restart:1;
        unsigned blend_eq_sep:1;
        unsigned instanceid:1;
        unsigned vertex_element_instance_divisor:1;
        unsigned seamless_cube_map:1;
        unsigned occlusion_query:1;
        unsigned timer_query:1;
        unsigned streamout_pause_resume:1;
        unsigned texture_multisample:1;
        unsigned fragment_coord_conventions:1;
        unsigned depth_clip_disable:1;
        unsigned seamless_cube_map_per_texture:1;
        unsigned ubo:1;
        unsigned color_clamping:1; /* not in GL 3.1 core profile */
        unsigned poly_stipple:1; /* not in GL 3.1 core profile */
        unsigned mirror_clamp:1;
        unsigned texture_query_lod:1;
        unsigned has_fp64:1;
        unsigned has_tessellation_shaders:1;
        unsigned has_indirect_draw:1;
        unsigned has_sample_shading:1;
        unsigned has_cull:1;
        unsigned conditional_render_inverted:1;
        unsigned derivative_control:1;
        unsigned polygon_offset_clamp:1;
        unsigned transform_feedback_overflow_query:1;
        /* DO NOT ADD ANYMORE MEMBERS - need to add another 32-bit to v2 caps */
};

/* endless expansion capabilites - current gallium has 252 formats */
struct virgl_supported_format_mask {
        uint32_t bitmask[16];
};
/* capabilities set 2 - version 1 - 32-bit and float values */
struct virgl_caps_v1 {
        uint32_t max_version;
        struct virgl_supported_format_mask sampler;
        struct virgl_supported_format_mask render;
        struct virgl_supported_format_mask depthstencil;
        struct virgl_supported_format_mask vertexbuffer;
        struct virgl_caps_bool_set1 bset;
        uint32_t glsl_level;
        uint32_t max_texture_array_layers;
        uint32_t max_streamout_buffers;
        uint32_t max_dual_source_render_targets;
        uint32_t max_render_targets;
        uint32_t max_samples;
        uint32_t prim_mask;
        uint32_t max_tbo_size;
        uint32_t max_uniform_blocks;
        uint32_t max_viewports;
        uint32_t max_texture_gather_components;
};

struct virgl_video_caps {
        uint32_t profile:8;
        uint32_t entrypoint:8;
        uint32_t max_level:8;
        uint32_t stacked_frames:8;

        uint32_t max_width:16;
        uint32_t max_height:16;

        uint32_t prefered_format:16;
        uint32_t max_macroblocks:16;

        uint32_t npot_texture:1;
        uint32_t supports_progressive:1;
        uint32_t supports_interlaced:1;
        uint32_t prefers_interlaced:1;
        uint32_t max_temporal_layers:8;
        uint32_t reserved:20;
};

/*
 * This struct should be growable when used in capset 2,
 * so we shouldn't have to add a v3 ever.
 */
struct virgl_caps_v2 {
        struct virgl_caps_v1 v1;
        float min_aliased_point_size;
        float max_aliased_point_size;
        float min_smooth_point_size;
        float max_smooth_point_size;
        float min_aliased_line_width;
        float max_aliased_line_width;
        float min_smooth_line_width;
        float max_smooth_line_width;
        float max_texture_lod_bias;
        uint32_t max_geom_output_vertices;
        uint32_t max_geom_total_output_components;
        uint32_t max_vertex_outputs;
        uint32_t max_vertex_attribs;
        uint32_t max_shader_patch_varyings;
        int32_t min_texel_offset;
        int32_t max_texel_offset;
        int32_t min_texture_gather_offset;
        int32_t max_texture_gather_offset;
        uint32_t texture_buffer_offset_alignment;
        uint32_t uniform_buffer_offset_alignment;
        uint32_t shader_buffer_offset_alignment;
        uint32_t capability_bits;
        uint32_t sample_locations[8];
        uint32_t max_vertex_attrib_stride;
        uint32_t max_shader_buffer_frag_compute;
        uint32_t max_shader_buffer_other_stages;
        uint32_t max_shader_image_frag_compute;
        uint32_t max_shader_image_other_stages;
        uint32_t max_image_samples;
        uint32_t max_compute_work_group_invocations;
        uint32_t max_compute_shared_memory_size;
        uint32_t max_compute_grid_size[3];
        uint32_t max_compute_block_size[3];
        uint32_t max_texture_2d_size;
        uint32_t max_texture_3d_size;
        uint32_t max_texture_cube_size;
        uint32_t max_combined_shader_buffers;
        uint32_t max_atomic_counters[6];
        uint32_t max_atomic_counter_buffers[6];
        uint32_t max_combined_atomic_counters;
        uint32_t max_combined_atomic_counter_buffers;
        uint32_t host_feature_check_version;
        struct virgl_supported_format_mask supported_readback_formats;
        struct virgl_supported_format_mask scanout;
        uint32_t capability_bits_v2;
        uint32_t max_video_memory;
        char renderer[64];
        float max_anisotropy;
        // NOTE: this informs guest-side pipe_shader_caps.max_texture_samplers,
        // **NOT** GL_MAX_TEXTURE_IMAGE_UNITS!!!
        // Guest-side driver has always used it as such.
        uint32_t max_texture_samplers;
        struct virgl_supported_format_mask supported_multisample_formats;
        uint32_t max_const_buffer_size[6]; // PIPE_SHADER_TYPES
        uint32_t num_video_caps;
        struct virgl_video_caps video_caps[32];
        uint32_t max_uniform_block_size;
        uint32_t max_tcs_outputs;
        uint32_t max_tes_outputs;
        uint32_t max_shader_storage_blocks[6]; // PIPE_SHADER_TYPES
};

union virgl_caps {
        uint32_t max_version;
        struct virgl_caps_v1 v1;
        struct virgl_caps_v2 v2;
};

enum virgl_errors {
        VIRGL_ERROR_NONE,
        VIRGL_ERROR_UNKNOWN,
        VIRGL_ERROR_UNKNOWN_RESOURCE_FORMAT,
};

enum virgl_ctx_errors {
        VIRGL_ERROR_CTX_NONE,
        VIRGL_ERROR_CTX_UNKNOWN,
        VIRGL_ERROR_CTX_ILLEGAL_SHADER,
        VIRGL_ERROR_CTX_ILLEGAL_HANDLE,
        VIRGL_ERROR_CTX_ILLEGAL_RESOURCE,
        VIRGL_ERROR_CTX_ILLEGAL_SURFACE,
        VIRGL_ERROR_CTX_ILLEGAL_VERTEX_FORMAT,
        VIRGL_ERROR_CTX_ILLEGAL_CMD_BUFFER,
        VIRGL_ERROR_CTX_GLES_HAVE_TES_BUT_MISS_TCS,
        VIRGL_ERROR_GL_ANY_SAMPLES_PASSED,
        VIRGL_ERROR_CTX_ILLEGAL_FORMAT,
        VIRGL_ERROR_CTX_ILLEGAL_SAMPLER_VIEW_TARGET,
        VIRGL_ERROR_CTX_TRANSFER_IOV_BOUNDS,
        VIRGL_ERROR_CTX_ILLEGAL_DUAL_SRC_BLEND,
        VIRGL_ERROR_CTX_UNSUPPORTED_FUNCTION,
        VIRGL_ERROR_CTX_ILLEGAL_PROGRAM_PIPELINE,
        VIRGL_ERROR_CTX_TOO_MANY_VERTEX_ATTRIBUTES,
        VIRGL_ERROR_CTX_UNSUPPORTED_TEX_WRAP,
        VIRGL_ERROR_CTX_CUBE_MAP_FACE_OUT_OF_RANGE,
        VIRGL_ERROR_CTX_BLIT_AREA_OUT_OF_RANGE,
        VIRGL_ERROR_CTX_SSBO_BINDING_RANGE,
        VIRGL_ERROR_CTX_RESOURCE_OUT_OF_RANGE,
};

enum virgl_statistics_query_index {
   VIRGL_STAT_QUERY_IA_VERTICES = 0,
   VIRGL_STAT_QUERY_IA_PRIMITIVES = 1,
   VIRGL_STAT_QUERY_VS_INVOCATIONS = 2,
   VIRGL_STAT_QUERY_GS_INVOCATIONS = 3,
   VIRGL_STAT_QUERY_GS_PRIMITIVES = 4,
   VIRGL_STAT_QUERY_C_INVOCATIONS = 5,
   VIRGL_STAT_QUERY_C_PRIMITIVES = 6,
   VIRGL_STAT_QUERY_PS_INVOCATIONS = 7,
   VIRGL_STAT_QUERY_HS_INVOCATIONS = 8,
   VIRGL_STAT_QUERY_DS_INVOCATIONS = 9,
   VIRGL_STAT_QUERY_CS_INVOCATIONS = 10,
};

/**
 * Flags for the driver about resource behaviour:
 */
#define VIRGL_RESOURCE_Y_0_TOP (1 << 0)
#define VIRGL_RESOURCE_FLAG_MAP_PERSISTENT (1 << 1)
#define VIRGL_RESOURCE_FLAG_MAP_COHERENT   (1 << 2)

#endif
