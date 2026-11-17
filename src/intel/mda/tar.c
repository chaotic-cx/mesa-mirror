/*
 * Copyright 2023 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "tar.h"

#include <assert.h>
#include <limits.h>
#include <string.h>
#include <time.h>

/* A tar archive contain a sequence of files, each files is composed of a
 * sequence of fixed size records.  The first record of a file has header,
 * defined by the table below:
 *
 *     Field Name   Byte Offset     Length in Bytes Field Type
 *     name         0               100             NUL-terminated if NUL fits
 *     mode         100             8
 *     uid          108             8
 *     gid          116             8
 *     size         124             12
 *     mtime        136             12
 *     chksum       148             8
 *     typeflag     156             1               see below
 *     linkname     157             100             NUL-terminated if NUL fits
 *     magic        256             6               must be TMAGIC (NUL term.)
 *     version      263             2               must be TVERSION
 *     uname        265             32              NUL-terminated
 *     gname        297             32              NUL-terminated
 *     devmajor     329             8
 *     devminor     337             8
 *     prefix       345             155             NUL-terminated if NUL fits
 *
 * The subsequent records contain the file contents, with extra padding to
 * fill a full record.  After that the header for the next file starts.
 * There's no archive-wide index.  See the code below for how checksum is
 * calculated.
 *
 * Comprehensive references for the tar archive are available in
 * https://www.loc.gov/preservation/digital/formats/fdd/fdd000531.shtml
 *
 * Note: the tar_writer implementation uses only the features and fields
 * needed for storing debug files.  The archive_reader implementation covers
 * only what's provided by the writer.
 */

enum {
   HEADER_NAME_OFFSET = 0,
   HEADER_NAME_LENGTH = 100,

   HEADER_MODE_OFFSET = 100,
   HEADER_MODE_LENGTH = 8,

   HEADER_SIZE_OFFSET = 124,
   HEADER_SIZE_LENGTH = 12,

   HEADER_MTIME_OFFSET = 136,
   HEADER_MTIME_LENGTH = 12,

   HEADER_CHECKSUM_OFFSET = 148,
   HEADER_CHECKSUM_LENGTH = 8,

   HEADER_TYPEFLAG_OFFSET = 156,

   HEADER_MAGIC_OFFSET = 257,
   HEADER_MAGIC_LENGTH = 6,

   HEADER_VERSION_OFFSET = 263,
   HEADER_VERSION_LENGTH = 2,

   HEADER_PREFIX_OFFSET = 345,
   HEADER_PREFIX_LENGTH = 155,
};

static unsigned
archive_calculate_checksum(const char *header)
{
   /* Checksum of the header assumes the checksum field itself is
    * filled with ASCII spaces (value 32).
    */
   unsigned checksum = 0;
   for (unsigned i = 0; i < RECORD_SIZE; i++) {
      unsigned char c = i >= HEADER_CHECKSUM_OFFSET &&
                        i < HEADER_CHECKSUM_OFFSET + HEADER_CHECKSUM_LENGTH ?
                        ' ' : (unsigned char)header[i];
      checksum += c;
   }
   return checksum;
}

static unsigned
archive_parse_checksum(const char *header)
{
   const char *p = &header[HEADER_CHECKSUM_OFFSET];
   const char *end = p + HEADER_CHECKSUM_LENGTH;

   while (p < end && *p == ' ')
      p++;

   unsigned stored = 0;
   bool digits = false;
   while (p < end && *p >= '0' && *p <= '7') {
      stored = stored * 8 + (*p - '0');
      digits = true;
      p++;
   }

   if (!digits || (p < end && *p != '\0' && *p != ' '))
      return UINT_MAX;

   return stored;
}

static void
archive_update_size(char *header, unsigned size)
{
   snprintf(&header[HEADER_SIZE_OFFSET], HEADER_SIZE_LENGTH, "%011o", size);

   unsigned checksum = archive_calculate_checksum(header);
   snprintf(&header[HEADER_CHECKSUM_OFFSET], HEADER_CHECKSUM_LENGTH, "%07o", checksum);
}

static bool
archive_set_path(char *header, const char *prefix, const char *filename)
{
   char path[HEADER_PREFIX_LENGTH + 1 + HEADER_NAME_LENGTH + 1];
   int len;
   if (!prefix || !*prefix) {
      len = snprintf(path, sizeof(path), "%s", filename);
   } else {
      len = snprintf(path, sizeof(path), "%s/%s", prefix, filename);
   }
   if (len < 0 || (size_t)len >= sizeof(path))
      return false;

   if ((size_t)len <= HEADER_NAME_LENGTH) {
      memcpy(&header[HEADER_NAME_OFFSET], path, len);
      return true;
   }

   /* Split at the last '/' that fits in the USTAR prefix field. */
   size_t split = (size_t)len - 1;
   if (split > HEADER_PREFIX_LENGTH)
      split = HEADER_PREFIX_LENGTH;
   while (split > 0 && path[split] != '/')
      split--;
   if (split == 0)
      return false;

   size_t name_len = (size_t)len - split - 1;
   if (name_len == 0 || name_len > HEADER_NAME_LENGTH)
      return false;

   memcpy(&header[HEADER_PREFIX_OFFSET], path, split);
   memcpy(&header[HEADER_NAME_OFFSET], path + split + 1, name_len);
   return true;
}

static bool
archive_start_header(char *header, const char *prefix, const char *filename,
                     char type, time_t timestamp)
{
   if (!archive_set_path(header, prefix, filename))
      return false;
   header[HEADER_TYPEFLAG_OFFSET] = type;

   const char *filemode = "0644";
   strcpy(&header[HEADER_MODE_OFFSET], filemode);

   snprintf(&header[HEADER_MTIME_OFFSET], HEADER_MTIME_LENGTH, "%011lo", (unsigned long)timestamp);

   const char *ustar_magic = "ustar";
   strcpy(&header[HEADER_MAGIC_OFFSET], ustar_magic);

   const char *ustar_version = "00";
   strcpy(&header[HEADER_VERSION_OFFSET], ustar_version);
   return true;
}

static unsigned
archive_calculate_padding(unsigned size)
{
   const unsigned m = size % RECORD_SIZE;
   return m ? RECORD_SIZE - m : 0;
}

static const char archive_empty_records[RECORD_SIZE * 2] = {0};

static bool
archive_write_padding(FILE *archive, unsigned contents_size)
{
   const unsigned padding_size = archive_calculate_padding(contents_size);
   return padding_size == 0 ||
          fwrite(archive_empty_records, padding_size, 1, archive) == 1;
}

static bool
archive_prewrite_end_of_archive(FILE *archive)
{
   /* Two empty records mark the proper end of the file, so always
    * keep them but reposition the cursor so next write overwrites
    * them.
    */
   if (fwrite(&archive_empty_records, RECORD_SIZE * 2, 1, archive) != 1 ||
       fflush(archive) != 0)
      return false;
   return fseek(archive, -RECORD_SIZE * 2, SEEK_END) == 0;
}

static bool
archive_write_path(FILE *archive, const char *prefix, const char *filename,
                   time_t timestamp)
{
   const char *directory = prefix && *prefix ? prefix : "";
   const char *separator = *directory ? "/" : "";

   /* The record length includes its own decimal digits. */
   int size = 0;
   while (true) {
      int next = snprintf(NULL, 0, "%d path=%s%s%s\n", size,
                          directory, separator, filename);
      if (next < 0)
         return false;
      if (next == size)
         break;
      size = next;
   }

   char header[RECORD_SIZE] = {0};
   if (!archive_start_header(header, NULL, "PaxHeaders/mda", 'x', timestamp))
      return false;
   archive_update_size(header, size);
   if (fwrite(header, RECORD_SIZE, 1, archive) != 1)
      return false;
   if (fprintf(archive, "%d path=%s%s%s\n", size,
               directory, separator, filename) != size)
      return false;
   if (!archive_write_padding(archive, size))
      return false;
   return ferror(archive) == 0;
}

static bool
archive_file_from_bytes(FILE *archive, const char *prefix, const char *filename,
                        const char *contents, unsigned contents_size, time_t timestamp)
{
   char header[RECORD_SIZE] = {0};

   if (!archive_start_header(header, prefix, filename, '0', timestamp)) {
      if (!archive_write_path(archive, prefix, filename, timestamp))
         return false;
      archive_start_header(header, NULL, "file", '0', timestamp);
   }
   archive_update_size(header, contents_size);

   if (fwrite(header, RECORD_SIZE, 1, archive) != 1)
      return false;

   if (contents_size && fwrite(contents, contents_size, 1, archive) != 1)
      return false;

   if (!archive_write_padding(archive, contents_size))
      return false;
   if (!archive_prewrite_end_of_archive(archive))
      return false;

   return ferror(archive) == 0;
}

void
tar_writer_init(tar_writer *tw, FILE *f)
{
   memset(tw, 0, sizeof(*tw));
   tw->file = f;
   tw->header_pos = -1;
   tw->error = false;
   tw->prefix = NULL;
   tw->timestamp = time(NULL);
}

void
tar_writer_start_file(tar_writer *tw, const char *filename)
{
   if (tw->error)
      return;

   assert(tw->header_pos == -1);
   memset(tw->header, 0, RECORD_SIZE);

   if (!archive_start_header(tw->header, tw->prefix, filename, '0', tw->timestamp)) {
      if (!archive_write_path(tw->file, tw->prefix, filename, tw->timestamp))
         goto fail;
      archive_start_header(tw->header, NULL, "file", '0', tw->timestamp);
   }
   archive_update_size(tw->header, 0);

   tw->header_pos = ftell(tw->file);
   if (tw->header_pos == -1)
      goto fail;

   if (fwrite(tw->header, RECORD_SIZE, 1, tw->file) != 1)
      goto fail;

   if (fflush(tw->file) != 0)
      goto fail;

   return;

fail:
   tw->error = true;
}

void
tar_writer_finish_file(tar_writer *tw)
{
   tw->error |= ferror(tw->file) != 0;
   if (tw->error)
      return;

   assert(tw->header_pos >= 0);

   const long end_pos = ftell(tw->file);
   if (end_pos == -1)
      goto fail;

   assert((tw->header_pos + RECORD_SIZE) <= end_pos);
   const unsigned size = end_pos - tw->header_pos - RECORD_SIZE;

   if (!archive_write_padding(tw->file, size))
      goto fail;

   archive_update_size(tw->header, size);

   if (fseek(tw->file, tw->header_pos, SEEK_SET) != 0)
      goto fail;
   if (fwrite(tw->header, RECORD_SIZE, 1, tw->file) != 1)
      goto fail;
   if (fseek(tw->file, end_pos + archive_calculate_padding(size), SEEK_SET) != 0)
      goto fail;

   if (!archive_prewrite_end_of_archive(tw->file))
      goto fail;

   if (ferror(tw->file))
      goto fail;

   goto end;

fail:
   tw->error = true;

end:
   tw->header_pos = -1;
   memset(tw->header, 0, RECORD_SIZE);
}

void
tar_writer_file_from_bytes(tar_writer *tw, const char *filename,
                           const char *contents, unsigned contents_size)
{
   if (tw->error)
      return;

   assert(tw->header_pos == -1);
   if (!archive_file_from_bytes(tw->file, tw->prefix, filename, contents, contents_size, tw->timestamp))
      tw->error = true;
}

void
tar_reader_init_from_bytes(tar_reader *ar, const char *contents, unsigned contents_size)
{
   memset(ar, 0, sizeof(*ar));
   ar->contents = (slice){contents, contents_size};
}

static bool
archive_read_entry(tar_reader *ar, tar_reader_entry *entry, char *type)
{
   if (ar->error)
      return false;

   if (ar->pos >= ar->contents.len)
      return false;

   if (ar->pos + RECORD_SIZE > ar->contents.len) {
      ar->error = true;
      return false;
   }

   const char *header = &ar->contents.data[ar->pos];
   const char *name   = &header[HEADER_NAME_OFFSET];
   const char *prefix = &header[HEADER_PREFIX_OFFSET];

   if (memcmp(header, archive_empty_records, RECORD_SIZE) == 0)
      return false;
   if (archive_parse_checksum(header) != archive_calculate_checksum(header)) {
      ar->error = true;
      return false;
   }

   ar->pos += RECORD_SIZE;
   *type = header[HEADER_TYPEFLAG_OFFSET];

   unsigned size = 0;
   if (sscanf((const char *)&header[HEADER_SIZE_OFFSET], "%11o", &size) != 1) {
      ar->error = true;
      return false;
   }

   unsigned long mtime = 0;
   if (*type != 'x' &&
       sscanf((const char *)&header[HEADER_MTIME_OFFSET], "%11lo", &mtime) != 1) {
      ar->error = true;
      return false;
   }

   const size_t remaining = ar->contents.len - ar->pos;
   if (size > remaining) {
      ar->error = true;
      return false;
   }

   const unsigned padded_size = size + archive_calculate_padding(size);
   if (padded_size > remaining) {
      ar->error = true;
      return false;
   }

   slice contents = slice_substr(ar->contents, ar->pos, ar->pos + size);
   ar->pos += padded_size;

   entry->prefix   = (slice){prefix, strnlen(prefix, HEADER_PREFIX_LENGTH)};
   entry->name     = (slice){name, strnlen(name, HEADER_NAME_LENGTH)};
   entry->contents = contents;
   entry->mtime    = (time_t)mtime;

   return true;
}

static bool
archive_read_path(slice contents, slice *path)
{
   while (!slice_is_empty(contents)) {
      slice_cut_result cut = slice_cut(contents, ' ');
      if (!cut.found)
         return false;

      int length = 0;
      for (int i = 0; i < cut.before.len; i++) {
         int digit = cut.before.data[i] - '0';
         if (digit < 0 || digit > 9 || length > (INT_MAX - digit) / 10)
            return false;
         length = length * 10 + digit;
      }

      int offset = cut.before.len + 1;
      if (length <= offset || length > contents.len || contents.data[length - 1] != '\n')
         return false;

      cut = slice_cut(slice_substr(contents, offset, length - 1), '=');
      if (!cut.found)
         return false;
      if (slice_equal_cstr(cut.before, "path")) {
         if (slice_is_empty(cut.after))
            return false;
         *path = cut.after;
      }
      contents = slice_substr_from(contents, length);
   }
   return true;
}

bool
tar_reader_next(tar_reader *ar, tar_reader_entry *entry)
{
   slice path = {0};
   char type;
   while (archive_read_entry(ar, entry, &type)) {
      if (type == 'x') {
         if (!archive_read_path(entry->contents, &path)) {
            ar->error = true;
            return false;
         }
         continue;
      }
      if (path.data) {
         entry->prefix = (slice){path.data, 0};
         entry->name = path;
      }
      return true;
   }
   return false;
}
