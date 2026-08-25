/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef STUBS_H
#define STUBS_H

/*
 * tensor_ops.cpp is compiled freestanding (-ffreestanding -nostdlib), so no
 * standard headers are available to declare these. The real symbols are
 * resolved from the Hexagon toolchain's libm at link time (see
 * jit/compiler.rs::link_to_so).
 */
extern "C" {
double fmin(double x, double y);
double fmax(double x, double y);
double frexp(double value, int32_t *exp);
double round(double x);
}

#endif // STUBS_H
