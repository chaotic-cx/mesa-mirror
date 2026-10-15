/*
 * Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 *
 * SPDX-License-Identifier: MIT
 *
 * GTest-based end-to-end test suite for the Torx ExecuTorch backend.
 *
 * This binary runs directly on the NPU board.  It discovers pre-compiled
 * test artifacts (.pte files, CPU references, metadata) generated at
 * build time by compile_torx_tests.py, runs NPU inference locally, and
 * compares the result against the CPU reference.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "test_torx_helpers.h"

/* ------------------------------------------------------------------ */
/* Tensor dump helpers                                                 */
/* ------------------------------------------------------------------ */

static void
dump_tensor_float(const std::vector<float> &data,
                  const std::filesystem::path &path)
{
   std::ofstream out(path, std::ios::binary);
   if (!out.is_open()) {
      fprintf(stderr, "Cannot dump to %s\n", path.c_str());
      return;
   }
   out.write(reinterpret_cast<const char *>(data.data()),
             (std::streamsize)(data.size() * sizeof(float)));
}

/* ------------------------------------------------------------------ */
/* Output comparison                                                   */
/* ------------------------------------------------------------------ */

/* 5 quantization steps should be enough for individual operations */
static constexpr int step_budget = 5;

/* For whole models, drift can accumulate and be amplified, so go with 8 for now */
static constexpr int model_step_budget = 8;

static void
compare_tensors(const std::vector<float> &actual,
                const std::vector<float> &expected,
                float scale, int budget,
                const std::string &test_name)
{
   ASSERT_EQ(actual.size(), expected.size())
      << test_name << ": output size mismatch";

   int over_budget = 0;
   long worst = 0;

   for (size_t i = 0; i < actual.size(); i++) {
      long qa = std::lround((double)actual[i] / scale);
      long qe = std::lround((double)expected[i] / scale);
      long dev = std::labs(qa - qe);
      if (dev > budget) {
         if (over_budget < 24) {
            ADD_FAILURE()
               << test_name << "[" << i << "]: "
               << "expected " << expected[i]
               << " got " << actual[i]
               << " (" << dev << " steps)";
         }
         over_budget++;
      }
      worst = std::max(worst, dev);
   }

   printf("%s: worst deviation %ld steps (budget %d steps)\n",
          test_name.c_str(), worst, budget);

   EXPECT_EQ(over_budget, 0)
      << test_name << ": " << over_budget << "/"
      << actual.size() << " elements over budget"
      << " (worst=" << worst << " steps"
      << ", budget=" << budget << " steps)";
}

/* ------------------------------------------------------------------ */
/* Test fixture — shared by per-op and whole-model tests               */
/* ------------------------------------------------------------------ */

class TorxTest : public testing::Test {
 public:
   std::string test_name_;   /* e.g. "mobilenet_v2/000" or "mobilenet_v2" */
   std::string device_id_;   /* e.g. "ethosu-65-256-98304" */
   int budget_;              /* comparison budget, in output-scale steps */

   TorxTest(const std::string &test_name, const std::string &device_id,
            int budget)
      : test_name_(test_name), device_id_(device_id), budget_(budget) {}

   void TestBody() override
   {
      auto work_dir = get_work_dir() / test_name_;
      auto pte_path = work_dir / (device_id_ + ".pte");
      auto meta_path = work_dir / (device_id_ + ".json");

      ASSERT_TRUE(std::filesystem::exists(pte_path))
         << "No .pte for device " << device_id_
         << " in " << work_dir.string();

      /* 1. Load build-time metadata */
      auto meta = load_meta(meta_path);
      ASSERT_TRUE(meta.ok)
         << "Failed to load metadata for " << test_name_
         << ": " << meta.error;

      /* 2. Run inference on the local NPU */
      std::vector<std::vector<float>> outputs;
      int rc = run_on_npu(work_dir, pte_path, device_id_, outputs);
      ASSERT_EQ(rc, 0)
         << "NPU inference failed for " << test_name_;

      /* 3. Compare each output against the quantised CPU reference. */
      for (size_t i = 0; ; i++) {
         auto ref_path = work_dir / ("output-" + std::to_string(i) + ".data");
         if (!std::filesystem::exists(ref_path))
            break;
         ASSERT_LT(i, meta.output_scales.size())
            << "Metadata has only " << meta.output_scales.size()
            << " output(s) but output-" << i << ".data exists"
            << " — rebuild with fixed run_on_cpu.py / compile_torx_tests.py";
         ASSERT_LT(i, outputs.size())
            << "The NPU returned only " << outputs.size()
            << " output(s) but output-" << i << ".data exists";
         const auto &npu_float = outputs[i];
         auto expected = load_tensor_float(ref_path);

          /* Dump for offline analysis */
          auto dump_name = test_name_ + "-output-" + std::to_string(i);
          std::replace(dump_name.begin(), dump_name.end(), '/', '.');
          dump_tensor_float(expected, dump_name + "-expected.bin");
          dump_tensor_float(npu_float,  dump_name + "-actual.bin");

         compare_tensors(npu_float, expected, meta.output_scales[i],
                         budget_, test_name_);
      }
   }
};

/* ------------------------------------------------------------------ */
/* Dynamic test registration                                           */
/* ------------------------------------------------------------------ */

static void
register_model_tests()
{
   std::string device_id = detect_device_id();
   if (device_id.empty()) {
      fprintf(stderr, "No NPU hardware detected, skipping all tests\n");
      return;
   }

   /* Per-op tests: <model>/<op>/<device>.pte */
   auto tests = get_model_files(device_id);
   for (const auto &test : tests) {
      /* "mobilenet_v2/000" → suite "mobilenet_v2", name "000" */
      auto slash = test.find('/');
      std::string suite = test.substr(0, slash);
      std::string name  = test.substr(slash + 1);

      testing::RegisterTest(
         suite.c_str(), name.c_str(),
         nullptr, nullptr,
         __FILE__, __LINE__,
         [test, device_id]() -> TorxTest * {
            return new TorxTest(test, device_id, step_budget);
         });
   }

   /* Whole-model tests: <model>/<device>.pte (no op subdirectory) */
   auto models = get_whole_model_files(device_id);
   for (const auto &model : models) {
      testing::RegisterTest(
         "model", model.c_str(),
         nullptr, nullptr,
         __FILE__, __LINE__,
         [model, device_id]() -> TorxTest * {
            return new TorxTest(model, device_id, model_step_budget);
         });
   }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int
main(int argc, char **argv)
{
   testing::InitGoogleTest(&argc, argv);

   /* Discover and register tests from pre-built artifacts */
   register_model_tests();

   return RUN_ALL_TESTS();
}
