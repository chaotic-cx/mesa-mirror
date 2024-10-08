/*
 * Copyright © 2012-2018 Rob Clark <robdclark@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef __UTIL_H__
#define __UTIL_H__

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>

#include "util/u_math.h"

/* old-style program binary XOR'd ascii w/ 0xff */
#ifndef ASCII_XOR
#define ASCII_XOR 0
#endif

static inline const char *
tab(int lvl)
{
   const char *TAB = "\t\t\t\t\t\t\t\t\0";
   return &TAB[strlen(TAB) - lvl];
}

static inline void
dump_hex(const void *buf, int sz)
{
   uint8_t *ptr = (uint8_t *)buf;
   uint8_t *end = ptr + sz;
   int i = 0;

   while (ptr < end) {
      uint32_t d = 0;

      printf((i % 8) ? " " : "\t");

      d |= *(ptr++) << 0;
      d |= *(ptr++) << 8;
      d |= *(ptr++) << 16;
      d |= (uint32_t)*(ptr++) << 24;

      printf("%08x", d);

      if ((i % 8) == 7) {
         printf("\n");
      }

      i++;
   }

   if (i % 8) {
      printf("\n");
   }
}

static inline void
dump_float(const void *buf, int sz)
{
   uint8_t *ptr = (uint8_t *)buf;
   uint8_t *end = ptr + sz - 3;
   int i = 0;

   while (ptr < end) {
      uint32_t d = 0;

      printf((i % 8) ? " " : "\t");

      d |= *(ptr++) << 0;
      d |= *(ptr++) << 8;
      d |= *(ptr++) << 16;
      d |= (uint32_t)*(ptr++) << 24;

      printf("%8f", uif(d));

      if ((i % 8) == 7) {
         printf("\n");
      }

      i++;
   }

   if (i % 8) {
      printf("\n");
   }
}

#define is_ok_ascii(c) (isascii(c) && ((c == '\t') || !iscntrl(c)))

static inline void
clean_ascii(char *buf, int sz)
{
   uint8_t *ptr = (uint8_t *)buf;
   uint8_t *end = ptr + sz;
   while (ptr < end) {
      *(ptr++) ^= ASCII_XOR;
   }
}

static inline void
dump_ascii(const void *buf, int sz)
{
   uint8_t *ptr = (uint8_t *)buf;
   uint8_t *end = ptr + sz;
   printf("\t");
   while (ptr < end) {
      uint8_t c = *(ptr++) ^ ASCII_XOR;
      if (c == '\n') {
         printf("\n\t");
      } else if (c == '\0') {
         printf("\n\t-----------------------------------\n\t");
      } else if (is_ok_ascii(c)) {
         printf("%c", c);
      } else {
         printf("?");
      }
   }
   printf("\n");
}

static inline void
dump_hex_ascii(const void *buf, int sz, int level)
{
   uint8_t *ptr = (uint8_t *)buf;
   uint8_t *end = ptr + sz;
   uint8_t *ascii = ptr;
   int i = 0;

   printf("%s-----------------------------------------------\n", tab(level));
   printf("%s%d (0x%x) bytes\n", tab(level), sz, sz);

   while (ptr < end) {
      uint32_t d = 0;

      if (i % 4) {
         printf(" ");
      } else {
         printf("%s%06x: ", tab(level), (uint32_t)(ptr - (uint8_t *)buf));
      }

      d |= *(ptr++) << 0;
      d |= *(ptr++) << 8;
      d |= *(ptr++) << 16;
      d |= (uint32_t)*(ptr++) << 24;

      printf("%08x", d);

      if ((i % 4) == 3) {
         int j;
         printf("\t|");
         for (j = 0; j < 16; j++) {
            uint8_t c = *(ascii++);
            c ^= ASCII_XOR;
            printf("%c", (isascii(c) && !iscntrl(c)) ? c : '.');
         }
         printf("|\n");
      }

      i++;
   }

   if (i % 4) {
      for (int j = 4 - (i % 4); j > 0; j--) {
         printf("         ");
      }
      printf("\t|");
      while (ascii < end) {
         uint8_t c = *(ascii++);
         c ^= ASCII_XOR;
         printf("%c", (isascii(c) && !iscntrl(c)) ? c : '.');
      }
      printf("|\n");
   }
}

#endif /* __UTIL_H__ */
