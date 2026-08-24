/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef AICMETADATA_COMPRESSL2TCMINITSTATE_H
#define AICMETADATA_COMPRESSL2TCMINITSTATE_H
#include <cstdint>
#include <vector>
namespace {
struct ZeroRegion {
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t size = 0;
};

struct NonZeroRegion {
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t size = 0;
};

std::vector<ZeroRegion> FindZeroRegions(const std::vector<uint8_t> &data) {
  constexpr uint32_t minimumZeroRegionSize = 128;
  std::vector<ZeroRegion> zeroRegions{};
  uint64_t zeroRegionStart = 0;
  uint64_t zeroRegionEnd = 0;
  if (data.empty()) {
    return zeroRegions;
  }
  bool inZeroRegion = (data[0] == 0);
  for (size_t i = 0; i < data.size(); i++) {
    if (data[i] == 0 && !inZeroRegion) {
      zeroRegionStart = i;
      inZeroRegion = true;
    } else if (data[i] != 0 && inZeroRegion) {
      zeroRegionEnd = i;
      inZeroRegion = false;
      if ((zeroRegionEnd - zeroRegionStart) > minimumZeroRegionSize) {
        zeroRegions.push_back(
            {zeroRegionStart, zeroRegionEnd, zeroRegionEnd - zeroRegionStart});
      }
    }
  }
  if (inZeroRegion) {
    zeroRegionEnd = data.size();
    if ((zeroRegionEnd - zeroRegionStart) > minimumZeroRegionSize) {
      zeroRegions.push_back(
          {zeroRegionStart, zeroRegionEnd, zeroRegionEnd - zeroRegionStart});
    }
  }
  return zeroRegions;
}

inline std::vector<NonZeroRegion>
FindNonZeroRegions(const std::vector<ZeroRegion> &zeroRegions, uint64_t size) {
  std::vector<NonZeroRegion> nonZeroRegions{};

  // Handle the case when there are no zero regions
  if (zeroRegions.empty()) {
    nonZeroRegions.push_back({0, size, size});
    return nonZeroRegions;
  }

  // Handle the case when the first non-zero region is at the beginning of the
  // data
  if (zeroRegions[0].start > 0) {
    nonZeroRegions.push_back({0, zeroRegions[0].start, zeroRegions[0].start});
  }

  // Iterate through the zero regions to find the non-zero regions between them
  for (size_t i = 1; i < zeroRegions.size(); ++i) {
    uint64_t nonZeroStart = zeroRegions[i - 1].end;
    uint64_t nonZeroEnd = zeroRegions[i].start;
    uint64_t nonZeroSize = nonZeroEnd - nonZeroStart;

    if (nonZeroSize > 0) {
      nonZeroRegions.push_back({nonZeroStart, nonZeroEnd, nonZeroSize});
    }
  }

  // Handle the case when the last non-zero region is at the end of the data
  if (zeroRegions.back().end < size) {
    uint64_t nonZeroStart = zeroRegions.back().end;
    uint64_t nonZeroEnd = size;
    uint64_t nonZeroSize = nonZeroEnd - nonZeroStart;

    if (nonZeroSize > 0) {
      nonZeroRegions.push_back({nonZeroStart, nonZeroEnd, nonZeroSize});
    }
  }

  return nonZeroRegions;
}
} // namespace
#endif // AICMETADATA_COMPRESSL2TCMINITSTATE_H
