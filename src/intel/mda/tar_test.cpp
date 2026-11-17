/*
 * Copyright 2024 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "tar.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

class TarTest : public testing::Test {
  protected:
   TarTest() : f{std::tmpfile()} {};
   ~TarTest() { std::fclose(f); }

   FILE *f;

   static void append_entry(std::vector<char> &archive, const std::string &prefix,
                            const std::string &name, const std::string &data,
                            char type = '0')
   {
      ASSERT_LE(prefix.size(), 155u);
      ASSERT_LE(name.size(), 100u);
      char header[512] = {};
      memcpy(header, name.data(), name.size());
      memcpy(header + 345, prefix.data(), prefix.size());
      memcpy(header + 100, "0000644", 7);
      snprintf(header + 124, 12, "%011o", unsigned(data.size()));
      snprintf(header + 136, 12, "%011o", 123u);
      header[156] = type;
      memcpy(header + 257, "ustar", 5);
      memcpy(header + 263, "00", 2);
      memset(header + 148, ' ', 8);
      unsigned checksum = 0;
      for (unsigned char c : header)
         checksum += c;
      snprintf(header + 148, 8, "%06o", checksum);
      header[155] = ' ';
      archive.insert(archive.end(), header, header + sizeof(header));
      archive.insert(archive.end(), data.begin(), data.end());
      archive.resize((archive.size() + 511) / 512 * 512);
   }

   static std::string path_record(const std::string &path)
   {
      const std::string body = " path=" + path + "\n";
      size_t length = body.size();
      while (true) {
         const std::string record = std::to_string(length) + body;
         if (record.size() == length)
            return record;
         length = record.size();
      }
   }
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

TEST_F(TarTest, PathRoundtrip)
{
   const struct {
      std::string prefix;
      std::string name;
      bool pax;
   } paths[] = {
      {std::string(150, 'p'), "NIR0/0", false},
      {std::string(151, 'p'), "NIR0/0", false},
      {std::string(155, 'p'), std::string(100, 'n'), false},
      {"capture.mda/CS", "NIR0/" + std::string(96, 'n'), false},
      {"", std::string(100, 'n'), false},
      {std::string(156, 'p'), "mesa.txt", true},
      {"", std::string(101, 'n'), true},
      {std::string(600, 'p'), std::string(700, 'n'), true},
      {"", std::string(989, 'n'), true},
      {"", std::string(990, 'n'), true},
      {"", std::string(502, 'n'), true},
   };

   for (const auto &[prefix, name, pax] : paths) {
      SCOPED_TRACE(testing::Message() << "prefix length=" << prefix.size()
                                      << " name length=" << name.size());
      for (bool streamed : {false, true}) {
         SCOPED_TRACE(streamed ? "streamed" : "bytes");
         FILE *next = std::tmpfile();
         ASSERT_NE(next, nullptr);
         std::fclose(f);
         f = next;

         tar_writer writer;
         tar_writer_init(&writer, f);
         writer.prefix = prefix.c_str();
         if (streamed) {
            tar_writer_start_file(&writer, name.c_str());
            ASSERT_FALSE(writer.error);
            ASSERT_EQ(fwrite("test", 4, 1, f), 1u);
            tar_writer_finish_file(&writer);
         } else {
            tar_writer_file_from_bytes(&writer, name.c_str(), "test", 4);
         }
         ASSERT_FALSE(writer.error);

         ASSERT_FALSE(fseek(f, 0, SEEK_END));
         long size = ftell(f);
         ASSERT_GT(size, 0);
         std::vector<char> contents(size);
         rewind(f);
         ASSERT_EQ(fread(contents.data(), size, 1, f), 1u);

         EXPECT_EQ(contents[156], pax ? 'x' : '0');
         if (!pax) {
            EXPECT_EQ(size, 2048);
         }
         tar_reader reader;
         tar_reader_init_from_bytes(&reader, contents.data(), size);
         tar_reader_entry entry;
         ASSERT_TRUE(tar_reader_next(&reader, &entry));
         ASSERT_FALSE(reader.error);
         if (pax) {
            EXPECT_EQ(entry.prefix.len, 0);
         }

         std::string path(entry.prefix.data, entry.prefix.len);
         if (!path.empty())
            path += '/';
         path.append(entry.name.data, entry.name.len);
         EXPECT_EQ(path, prefix.empty() ? name : prefix + "/" + name);
         EXPECT_TRUE(slice_equal_cstr(entry.contents, "test"));
      }
   }
}

TEST_F(TarTest, LegacyUSTAR)
{
   const std::string prefix(155, 'p');
   const std::string name(100, 'n');
   std::vector<char> archive;
   append_entry(archive, prefix, name, "test");
   archive.resize(archive.size() + 1024);

   tar_reader reader;
   tar_reader_init_from_bytes(&reader, archive.data(), archive.size());
   tar_reader_entry entry;
   ASSERT_TRUE(tar_reader_next(&reader, &entry));
   EXPECT_TRUE(slice_equal_cstr(entry.prefix, prefix.c_str()));
   EXPECT_TRUE(slice_equal_cstr(entry.name, name.c_str()));
   EXPECT_TRUE(slice_equal_cstr(entry.contents, "test"));
   EXPECT_FALSE(tar_reader_next(&reader, &entry));
   EXPECT_FALSE(reader.error);
}

TEST_F(TarTest, PathOverrideIsLocal)
{
   const std::string long_prefix(156, 'p');
   tar_writer writer;
   tar_writer_init(&writer, f);
   writer.prefix = long_prefix.c_str();
   tar_writer_file_from_bytes(&writer, "NIR0/0", "first", 5);
   writer.prefix = "mda/VS";
   tar_writer_file_from_bytes(&writer, "NIR0/1", "second", 6);
   ASSERT_FALSE(writer.error);

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
   EXPECT_EQ(entry.prefix.len, 0);
   EXPECT_EQ(std::string(entry.name.data, entry.name.len), long_prefix + "/NIR0/0");
   EXPECT_TRUE(slice_equal_cstr(entry.contents, "first"));

   ASSERT_TRUE(tar_reader_next(&reader, &entry));
   std::string second_path(entry.prefix.data, entry.prefix.len);
   if (!second_path.empty())
      second_path += '/';
   second_path.append(entry.name.data, entry.name.len);
   EXPECT_EQ(second_path, "mda/VS/NIR0/1");
   EXPECT_TRUE(slice_equal_cstr(entry.contents, "second"));
   EXPECT_FALSE(tar_reader_next(&reader, &entry));
   EXPECT_FALSE(reader.error);
}

TEST_F(TarTest, TrailingPathHeader)
{
   std::vector<char> archive;
   append_entry(archive, "mda/CS", "NIR0/0", "test");
   append_entry(archive, "", "PaxHeaders/mda", path_record("unfinished"), 'x');
   const size_t unfinished_size = archive.size();
   archive.resize(archive.size() + 1024);

   for (size_t size : {unfinished_size, archive.size()}) {
      tar_reader reader;
      tar_reader_init_from_bytes(&reader, archive.data(), size);
      tar_reader_entry entry;
      ASSERT_TRUE(tar_reader_next(&reader, &entry));
      EXPECT_FALSE(tar_reader_next(&reader, &entry));
      EXPECT_FALSE(reader.error);
   }
}

TEST_F(TarTest, PathCopiedAtStart)
{
   char prefix[] = "mda/CS";
   std::string name = "NIR0/" + std::string(101, 'n');
   const std::string path = std::string(prefix) + "/" + name;
   tar_writer writer;
   tar_writer_init(&writer, f);
   writer.prefix = prefix;
   tar_writer_start_file(&writer, name.c_str());
   ASSERT_FALSE(writer.error);
   prefix[0] = 'x';
   name[0] = 'y';
   ASSERT_EQ(fwrite("test", 4, 1, f), 1u);
   tar_writer_finish_file(&writer);
   ASSERT_FALSE(writer.error);
   long size = ftell(f) + 1024;
   std::vector<char> contents(size);
   rewind(f);
   ASSERT_EQ(fread(contents.data(), size, 1, f), 1u);

   tar_reader reader;
   tar_reader_init_from_bytes(&reader, contents.data(), size);
   tar_reader_entry entry;
   ASSERT_TRUE(tar_reader_next(&reader, &entry));
   EXPECT_TRUE(slice_equal_cstr(entry.name, path.c_str()));
   EXPECT_TRUE(slice_equal_cstr(entry.contents, "test"));
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

TEST_F(TarTest, FileFromBytesAfterFailedStream)
{
   FILE *file = fopen("/dev/null", "r");
   if (!file) {
      GTEST_SKIP() << "Requires /dev/null";
   }
   tar_writer writer;
   tar_writer_init(&writer, file);
   tar_writer_start_file(&writer, "first");
   EXPECT_TRUE(writer.error);
   tar_writer_finish_file(&writer);
   clearerr(file);
   tar_writer_file_from_bytes(&writer, "last", "b", 1);
   EXPECT_TRUE(writer.error);
   fclose(file);
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
