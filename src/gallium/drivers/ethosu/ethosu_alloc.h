/*
 * Copyright 2026 Arm Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef ETHOSU_ALLOC_H
#define ETHOSU_ALLOC_H

struct ethosu_subgraph;

void
ethosu_allocate_tensors(struct ethosu_subgraph *subgraph,
                        const unsigned *required_sizes);

#endif /* ETHOSU_ALLOC_H */
