/*
 * Copyright 2024 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "tar.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

class TarTest : public testing::Test {
  protected:
   TarTest() : f{std::tmpfile()} {};
   ~TarTest() { std::fclose(f); }

   FILE *f;
};


TEST_F(TarTest, StopAtUnfinishedFile)
{
   tar_writer tw;
   tar_writer_init(&tw, f);
   tar_writer_file_from_bytes(&tw, "first", "a", 1);
   ASSERT_FALSE(tw.error);

   tar_writer_start_file(&tw, "unfinished");
   ASSERT_FALSE(tw.error);
   char payload[RECORD_SIZE] = "unfinished contents";
   ASSERT_EQ(fwrite(payload, sizeof(payload), 1, f), 1u);
   fflush(f);

   ASSERT_EQ(fseek(f, 0, SEEK_END), 0);
   long size = ftell(f);
   ASSERT_GT(size, 0);
   std::vector<char> contents(size);
   rewind(f);
   ASSERT_EQ(fread(contents.data(), size, 1, f), 1u);

   tar_reader reader;
   tar_reader_init_from_bytes(&reader, contents.data(), size);
   tar_reader_entry entry;
   ASSERT_TRUE(tar_reader_next(&reader, &entry));
   EXPECT_TRUE(slice_equal_cstr(entry.name, "first"));
   EXPECT_TRUE(slice_equal_cstr(entry.contents, "a"));

   ASSERT_TRUE(tar_reader_next(&reader, &entry));
   EXPECT_TRUE(slice_equal_cstr(entry.name, "unfinished"));
   EXPECT_TRUE(slice_is_empty(entry.contents));

   EXPECT_FALSE(tar_reader_next(&reader, &entry));
   EXPECT_TRUE(reader.error);
}

TEST_F(TarTest, NonAsciiFilenameChecksum)
{
   tar_writer tw;
   tar_writer_init(&tw, f);
   tar_writer_file_from_bytes(&tw, "caf\xc3\xa9", "data", 4);
   ASSERT_FALSE(tw.error);

   ASSERT_EQ(fseek(f, 0, SEEK_END), 0);
   long size = ftell(f);
   ASSERT_GT(size, 0);
   std::vector<char> contents(size);
   rewind(f);
   ASSERT_EQ(fread(contents.data(), size, 1, f), 1u);

   tar_reader reader;
   tar_reader_init_from_bytes(&reader, contents.data(), size);
   tar_reader_entry entry;
   ASSERT_TRUE(tar_reader_next(&reader, &entry));
   EXPECT_TRUE(slice_equal_cstr(entry.name, "caf\xc3\xa9"));
   EXPECT_TRUE(slice_equal_cstr(entry.contents, "data"));
   EXPECT_FALSE(tar_reader_next(&reader, &entry));
   EXPECT_FALSE(reader.error);
}

TEST_F(TarTest, RoundtripSmallFile)
{
   const char *test = "TEST TEST TEST";

   {
      tar_writer tw;
      tar_writer_init(&tw, f);

      tar_writer_start_file(&tw, "test");

      fwrite(test, strlen(test), 1, f);

      tar_writer_finish_file(&tw);
   }

   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   ASSERT_TRUE(size > 0);
   ASSERT_TRUE(size % 512 == 0);
   std::vector<char> contents(size);

   fseek(f, 0, SEEK_SET);
   ASSERT_EQ(fread(contents.data(), size, 1, f), 1);

   {
      tar_reader ar;
      tar_reader_init_from_bytes(&ar, contents.data(), size);

      tar_reader_entry entry;

      bool first_read = tar_reader_next(&ar, &entry);
      ASSERT_TRUE(first_read);

      ASSERT_EQ(entry.name.len, 4);
      ASSERT_TRUE(memcmp(entry.name.data, "test", 4) == 0);

      ASSERT_EQ(entry.contents.len, strlen(test));
      ASSERT_TRUE(memcmp(entry.contents.data, test, entry.contents.len) == 0);

      bool second_read = tar_reader_next(&ar, &entry);
      ASSERT_FALSE(second_read);
   }
}

TEST_F(TarTest, RoundtripContentsWithRecordSize)
{
   uint8_t test[512];

   for (unsigned i = 0; i < sizeof(test); i++) {
      test[i] = 'A' + (i % 26);
   }

   {
      tar_writer tw;
      tar_writer_init(&tw, f);
      tar_writer_file_from_bytes(&tw, "test", (const char*)test, sizeof(test));
      ASSERT_FALSE(tw.error);
   }

   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   ASSERT_TRUE(size > 0);
   ASSERT_TRUE(size % 512 == 0);
   std::vector<char> contents(size);

   fseek(f, 0, SEEK_SET);
   ASSERT_EQ(fread(contents.data(), size, 1, f), 1);

   {
      tar_reader ar;
      tar_reader_init_from_bytes(&ar, contents.data(), size);
      ASSERT_FALSE(ar.error);

      tar_reader_entry entry;

      bool first_read = tar_reader_next(&ar, &entry);
      ASSERT_TRUE(first_read);

      ASSERT_EQ(entry.name.len, 4);
      ASSERT_TRUE(memcmp(entry.name.data, "test", 4) == 0);

      ASSERT_EQ(entry.contents.len, sizeof(test));
      ASSERT_TRUE(memcmp(entry.contents.data, test, sizeof(test)) == 0);

      bool second_read = tar_reader_next(&ar, &entry);
      ASSERT_FALSE(second_read);
   }
}

TEST_F(TarTest, TimestampRoundtrip)
{
   const char *test = "TEST TIMESTAMP";
   const time_t test_timestamp = 1234567890; // Known timestamp (February 13, 2009)

   {
      tar_writer tw;
      tar_writer_init(&tw, f);
      tw.timestamp = test_timestamp;

      tar_writer_file_from_bytes(&tw, "timestamp_test", test, strlen(test));
      ASSERT_FALSE(tw.error);
   }

   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   ASSERT_TRUE(size > 0);
   ASSERT_TRUE(size % 512 == 0);
   std::vector<char> contents(size);

   fseek(f, 0, SEEK_SET);
   ASSERT_EQ(fread(contents.data(), size, 1, f), 1);

   {
      tar_reader ar;
      tar_reader_init_from_bytes(&ar, contents.data(), size);
      ASSERT_FALSE(ar.error);

      tar_reader_entry entry;

      bool first_read = tar_reader_next(&ar, &entry);
      ASSERT_TRUE(first_read);

      ASSERT_EQ(entry.name.len, strlen("timestamp_test"));
      ASSERT_TRUE(memcmp(entry.name.data, "timestamp_test", strlen("timestamp_test")) == 0);

      ASSERT_EQ(entry.contents.len, strlen(test));
      ASSERT_TRUE(memcmp(entry.contents.data, test, strlen(test)) == 0);

      // Verify the timestamp matches
      ASSERT_EQ(entry.mtime, test_timestamp);

      bool second_read = tar_reader_next(&ar, &entry);
      ASSERT_FALSE(second_read);
   }
}

TEST_F(TarTest, RoundtripEmptyMiddleFile)
{
   const struct {
      const char *name;
      const char *contents;
   } files[] = {
      {"first", "a"},
      {"empty", ""},
      {"last", "b"},
   };

   for (bool streamed : {false, true}) {
      SCOPED_TRACE(streamed ? "streamed" : "bytes");
      FILE *next = std::tmpfile();
      ASSERT_NE(next, nullptr);
      std::fclose(f);
      f = next;

      tar_writer writer;
      tar_writer_init(&writer, f);
      for (const auto &file : files) {
         size_t size = strlen(file.contents);
         if (streamed) {
            tar_writer_start_file(&writer, file.name);
            ASSERT_FALSE(writer.error);
            ASSERT_EQ(fwrite(file.contents, 1, size, f), size);
            tar_writer_finish_file(&writer);
         } else {
            tar_writer_file_from_bytes(&writer, file.name, file.contents, size);
         }
         ASSERT_FALSE(writer.error);
      }

      ASSERT_EQ(fseek(f, 0, SEEK_END), 0);
      long size = ftell(f);
      ASSERT_EQ(size, 7 * RECORD_SIZE);
      std::vector<char> contents(size);
      rewind(f);
      ASSERT_EQ(fread(contents.data(), size, 1, f), 1u);

      tar_reader reader;
      tar_reader_init_from_bytes(&reader, contents.data(), size);
      tar_reader_entry entry;
      for (const auto &file : files) {
         ASSERT_TRUE(tar_reader_next(&reader, &entry));
         EXPECT_TRUE(slice_equal_cstr(entry.name, file.name));
         EXPECT_TRUE(slice_equal_cstr(entry.contents, file.contents));
      }
      EXPECT_FALSE(tar_reader_next(&reader, &entry));
      EXPECT_FALSE(reader.error);
   }
}
