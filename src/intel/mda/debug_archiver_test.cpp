/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "debug_archiver.h"
#include "tar.h"
#include "util/os_file.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <dirent.h>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

class DebugArchiverTest : public testing::Test {
protected:
   using Archive = std::unique_ptr<debug_archiver, decltype(&debug_archiver_close)>;

   static void SetUpTestSuite()
   {
      if (directory.empty()) {
         char path[] = "/tmp/mda-test-XXXXXX";
         ASSERT_NE(mkdtemp(path), nullptr);
         directory = path;
         setenv("MDA_OUTPUT_DIR", directory.c_str(), 1);
         unsetenv("MDA_PREFIX");
         unsetenv("MDA_FILTER");
      } else {
         ASSERT_EQ(os_mkdir(directory.c_str(), 0755), 0);
      }
   }

   static void TearDownTestSuite()
   {
      rmdir(directory.c_str());
   }

   void TearDown() override
   {
      for (const auto &path : files())
         unlink(path.c_str());
   }

   Archive open(const char *name)
   {
      return Archive(debug_archiver_open(nullptr, name, "test"),
                     debug_archiver_close);
   }

   std::vector<std::string> files()
   {
      std::vector<std::string> paths;
      DIR *dir = opendir(directory.c_str());
      if (!dir)
         return paths;

      while (struct dirent *entry = readdir(dir)) {
         if (entry->d_name[0] != '.')
            paths.push_back(directory + "/" + entry->d_name);
      }
      closedir(dir);
      return paths;
   }

   void check_file(const std::string &path, const char *name, const char *expected,
                   const char *prefix = "capture.mda/CS")
   {
      size_t size;
      char *data = os_read_file(path.c_str(), &size);
      ASSERT_NE(data, nullptr);
      std::unique_ptr<char, decltype(&free)> contents(data, free);

      tar_reader reader;
      tar_reader_init_from_bytes(&reader, data, size);
      tar_reader_entry entry;
      while (tar_reader_next(&reader, &entry)) {
         std::string fullpath(entry.prefix.data, entry.prefix.len);
         if (!fullpath.empty())
            fullpath += '/';
         fullpath.append(entry.name.data, entry.name.len);
         if (fullpath == std::string(prefix) + "/" + name) {
            EXPECT_TRUE(slice_equal_cstr(entry.contents, expected));
            return;
         }
      }
      FAIL() << "Missing " << name << " in " << path;
   }

   static std::string directory;
};

std::string DebugArchiverTest::directory;

TEST_F(DebugArchiverTest, VisibleWhileOpen)
{
   auto archive = open("capture");
   ASSERT_NE(archive, nullptr);
   debug_archiver_set_prefix(archive.get(), "CS");
   debug_archiver_write_file(archive.get(), "NIR0/0", "first", 5);

   auto paths = files();
   ASSERT_EQ(paths.size(), 1u);
   EXPECT_EQ(paths[0].find(directory + "/capture.tmp-"), 0u);
   EXPECT_EQ(paths[0].substr(paths[0].size() - 8), ".mda.tar");
   check_file(paths[0], "NIR0/0", "first");

   archive.reset();
   paths = files();
   ASSERT_EQ(paths.size(), 1u);
   EXPECT_EQ(paths[0], directory + "/capture.mda.tar");
   check_file(paths[0], "NIR0/0", "first");
}

TEST_F(DebugArchiverTest, LongInternalName)
{
   for (unsigned prefix_length : {120, 149}) {
      const std::string name = std::string(prefix_length, 'p') + "." +
                               std::string(64, '0') + ".anv-generated-indirect-draws";
      const std::string root = name + ".mda";
      auto archive = open(name.c_str());
      ASSERT_NE(archive, nullptr);
      debug_archiver_set_prefix(archive.get(), "FS");
      debug_archiver_write_file(archive.get(), "NIR0/0", "first", 5);
      debug_archiver_set_prefix(archive.get(), "CS_main");
      const std::string version = "NIR0/" + std::string(100, 'v');
      debug_archiver_write_file(archive.get(), version.c_str(), "second", 6);

      const auto paths = files();
      ASSERT_EQ(paths.size(), 1u);
      check_file(paths[0], "mesa.txt", "Mesa test\n", root.c_str());
      check_file(paths[0], "NIR0/0", "first", (root + "/FS").c_str());
      check_file(paths[0], version.c_str(), "second", (root + "/CS_main").c_str());

      archive.reset();
      const std::string final_path = directory + "/" + name + ".mda.tar";
      ASSERT_EQ(files().size(), 1u);
      check_file(final_path, "mesa.txt", "Mesa test\n", root.c_str());
      check_file(final_path, "NIR0/0", "first", (root + "/FS").c_str());
      check_file(final_path, version.c_str(), "second", (root + "/CS_main").c_str());
      unlink(final_path.c_str());
   }
}

TEST_F(DebugArchiverTest, OverlappingWriters)
{
   auto first = open("capture");
   auto second = open("capture");
   ASSERT_NE(first, nullptr);
   ASSERT_NE(second, nullptr);
   ASSERT_EQ(files().size(), 2u);

   debug_archiver_set_prefix(first.get(), "CS");
   debug_archiver_set_prefix(second.get(), "CS");
   debug_archiver_write_file(first.get(), "NIR0/0", "first", 5);
   debug_archiver_write_file(second.get(), "NIR0/0", "second", 6);

   second.reset();
   const std::string final_path = directory + "/capture.mda.tar";
   ASSERT_EQ(files().size(), 2u);
   check_file(final_path, "NIR0/0", "second");

   debug_archiver_write_file(first.get(), "NIR0/1", "later", 5);
   check_file(final_path, "NIR0/0", "second");

   first.reset();
   ASSERT_EQ(files().size(), 1u);
   check_file(final_path, "NIR0/0", "first");
   check_file(final_path, "NIR0/1", "later");
}
