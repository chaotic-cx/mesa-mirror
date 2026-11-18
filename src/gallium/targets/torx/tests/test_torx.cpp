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

static float
softmax_at(const std::vector<float> &logits, size_t index)
{
   double sum = 0.0;
   for (float logit : logits)
      sum += std::exp((double)logit - logits[index]);
   return (float)(1.0 / sum);
}

static void
compare_best_class(const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   size_t classes,
                   const std::string &test_name)
{
   ASSERT_EQ(actual.size(), expected.size())
      << test_name << ": output size mismatch";
   ASSERT_EQ(actual.size(), classes)
      << test_name << ": output is not one row of " << classes
      << " classes";

   size_t got = std::max_element(actual.begin(), actual.end()) -
                actual.begin();
   size_t want = std::max_element(expected.begin(), expected.end()) -
                 expected.begin();

   printf("%s: best class %zu at %.3f, reference %zu at %.3f\n",
          test_name.c_str(), got, softmax_at(actual, got), want,
          softmax_at(expected, want));

   EXPECT_EQ(got, want) << test_name << ": best class differs";
}

/* Defaults from YOLOX's demo (tools/demo.py) */
static constexpr float yolox_score_threshold = 0.3f;
static constexpr float yolox_nms_threshold = 0.3f;
static constexpr float detection_match_iou = 0.9f;

struct detection {
   float x1, y1, x2, y2;
   float score;
   size_t cls;
};

static float
detection_iou(const detection &a, const detection &b)
{
   float w = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
   float h = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
   float intersection = w * h;
   float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
   float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
   return intersection / (area_a + area_b - intersection);
}

/* Decode as postprocess() in YOLOX's yolox/utils/boxes.py does. */
static std::vector<detection>
decode_yolox(const std::vector<float> &output, size_t row_size)
{
   std::vector<detection> candidates;

   for (size_t row = 0; row < output.size(); row += row_size) {
      const float *p = &output[row];
      const float *best = std::max_element(p + 5, p + row_size);
      float score = p[4] * *best;
      if (score < yolox_score_threshold)
         continue;
      candidates.push_back({p[0] - p[2] / 2, p[1] - p[3] / 2,
                            p[0] + p[2] / 2, p[1] + p[3] / 2,
                            score, (size_t)(best - (p + 5))});
   }

   std::stable_sort(candidates.begin(), candidates.end(),
                    [](const detection &a, const detection &b) {
                       return a.score > b.score;
                    });

   std::vector<detection> kept;
   for (const auto &candidate : candidates) {
      bool suppressed = std::any_of(
         kept.begin(), kept.end(), [&](const detection &k) {
            return detection_iou(k, candidate) > yolox_nms_threshold;
         });
      if (!suppressed)
         kept.push_back(candidate);
   }

   return kept;
}

static void
compare_detections(const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   size_t row_size,
                   const std::string &test_name)
{
   ASSERT_EQ(actual.size(), expected.size())
      << test_name << ": output size mismatch";
   ASSERT_GT(row_size, 5u)
      << test_name << ": output rows hold no class scores";
   ASSERT_EQ(actual.size() % row_size, 0u)
      << test_name << ": output is not a whole number of rows";

   auto got = decode_yolox(actual, row_size);
   auto want = decode_yolox(expected, row_size);

   printf("%s: %zu detections, reference %zu\n",
          test_name.c_str(), got.size(), want.size());

   ASSERT_FALSE(want.empty())
      << test_name << ": the reference output decodes to no detection";

   EXPECT_EQ(got.size(), want.size())
      << test_name << ": number of detections differs";

   std::vector<bool> matched(got.size(), false);
   for (const auto &reference : want) {
      size_t best = got.size();
      float best_iou = 0.0f;
      for (size_t j = 0; j < got.size(); j++) {
         if (matched[j] || got[j].cls != reference.cls)
            continue;
         float iou = detection_iou(got[j], reference);
         if (iou > best_iou) {
            best = j;
            best_iou = iou;
         }
      }

      printf("%s: reference class %zu score %.3f box (%.1f, %.1f, %.1f, "
             "%.1f), ", test_name.c_str(), reference.cls, reference.score,
             reference.x1, reference.y1, reference.x2, reference.y2);
      if (best < got.size()) {
         printf("best match score %.3f box (%.1f, %.1f, %.1f, %.1f) "
                "IoU %.3f\n", got[best].score, got[best].x1, got[best].y1,
                got[best].x2, got[best].y2, best_iou);
      } else {
         printf("no detection of its class\n");
      }

      if (best < got.size() && best_iou >= detection_match_iou) {
         matched[best] = true;
      } else {
         ADD_FAILURE()
            << test_name << ": reference detection of class "
            << reference.cls << " has no match at IoU "
            << detection_match_iou << " (best " << best_iou << ")";
      }
   }
}

typedef void (*output_check)(const std::vector<float> &actual,
                             const std::vector<float> &expected,
                             size_t last_dim,
                             const std::string &test_name);

static output_check
convention_check(const std::string &convention)
{
   if (convention == "logits")
      return compare_best_class;
   if (convention == "yolox")
      return compare_detections;
   return nullptr;
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

      output_check check = convention_check(meta.output_convention);
      ASSERT_TRUE(check || meta.output_convention.empty())
         << test_name_ << ": no check for output convention '"
         << meta.output_convention << "'";

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

         if (check) {
            ASSERT_LT(i, meta.output_shapes.size())
               << "Metadata has no shape for output " << i;
            ASSERT_FALSE(meta.output_shapes[i].empty())
               << "Metadata has an empty shape for output " << i;
            check(npu_float, expected, meta.output_shapes[i].back(),
                  test_name_);
         } else {
            compare_tensors(npu_float, expected, meta.output_scales[i],
                            budget_, test_name_);
         }
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
