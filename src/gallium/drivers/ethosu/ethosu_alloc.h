/*
 * Copyright 2026 Arm Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef ETHOSU_ALLOC_H
#define ETHOSU_ALLOC_H

struct ethosu_feature_map;
struct ethosu_subgraph;

unsigned
ethosu_allocate_feature_map(struct ethosu_subgraph *subgraph,
                            const unsigned *required_sizes,
                            struct ethosu_feature_map *fm);

#endif /* ETHOSU_ALLOC_H */
