// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
/** @file nnc_prng_inf.h
 *  @brief NNC PRNG Interface Header file
 */

#ifndef __NNC_PRNG_INF_H__
#define __NNC_PRNG_INF_H__

#include <stdint.h>

/**
  @ingroup nnc_get_randNum_fp
  Function to generate random numbers from the PRNG HW in AI100.
  The PRNG block satisfies following NIST standard for pseudo random number
  generation. Recommendation for Random Number Generation Using Deterministic
  Random Bit Generators NIST Special Publication 800-90A. Please refer to the
  Platform Spec document for more information on the properties of the random
  numbers generated.

  @param[in,out] Pointer to the va address
  @param[in] Number of values to be generated. Each value is of size 4 bytes

  @return
  0 on success
  -1 on failure

  @dependencies
  None.
*/

typedef int (*nnc_get_randNum_fp)(uint32_t *out_ptr, uint32_t num_elements);

#endif //__NNC_PRNG_INF_H__
