/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

/*
 * fmin/fmax/frexp/round, hand-implemented for the freestanding Hexagon build
 * (-ffreestanding -fno-builtin -nostdlib, see teflon_ops/tensor_ops.cpp and
 * jit/compiler.rs::link_to_so). No distro/upstream LLVM ships a Hexagon
 * libm, so these can't be pulled from a system library; the signatures
 * match the extern declarations in teflon_ops/stubs.h.
 */

#include <stdint.h>

double fmin(double x, double y)
{
    if (x != x)
        return y;
    if (y != y)
        return x;
    return x < y ? x : y;
}

double fmax(double x, double y)
{
    if (x != x)
        return y;
    if (y != y)
        return x;
    return x > y ? x : y;
}

double round(double x)
{
    if (x != x || x == 0.0)
        return x;

    double ax = x < 0.0 ? -x : x;

    /* Already integral at this magnitude; avoids overflowing the
     * int64_t truncation below. */
    if (ax >= 4503599627370496.0 /* 2^52 */)
        return x;

    double truncated = (double)(int64_t)ax;
    double frac = ax - truncated;
    double result = frac >= 0.5 ? truncated + 1.0 : truncated;

    return x < 0.0 ? -result : result;
}

double frexp(double value, int32_t *exp)
{
    union {
        double d;
        uint64_t u;
    } v;
    v.d = value;

    uint64_t sign = v.u & 0x8000000000000000ULL;
    int32_t biased_exp = (int32_t)((v.u >> 52) & 0x7FF);
    uint64_t mantissa = v.u & 0xFFFFFFFFFFFFFULL;

    /* Infinity/NaN and zero pass through unchanged. */
    if (biased_exp == 0x7FF || (biased_exp == 0 && mantissa == 0)) {
        *exp = 0;
        return value;
    }

    if (biased_exp == 0) {
        /* Subnormal: normalize the mantissa so the same bias-1022
         * placement below yields a result in [0.5, 1). */
        int32_t shift = 0;
        while (!(mantissa & 0x10000000000000ULL)) {
            mantissa <<= 1;
            shift++;
        }
        mantissa &= 0xFFFFFFFFFFFFFULL;
        biased_exp = 1 - shift;
    }

    *exp = biased_exp - 1022;

    union {
        double d;
        uint64_t u;
    } result;
    result.u = sign | ((uint64_t)1022 << 52) | mantissa;
    return result.d;
}
