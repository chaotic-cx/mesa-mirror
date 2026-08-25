/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#include "jit_qshim_api.h"
#include "jit_dev_exe_function.h"
#include "stubs.h"

struct tensor_list {
    uint32_t *sizes;
    uint32_t *widths;
    uint32_t *heights;
    uint32_t *depths;
    int32_t *zero_points;
    double *scales;
    uint8_t **data;
};

struct tensor {
    uint32_t size;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    int32_t zero_point;
    double scale;
    uint8_t *data;
};

extern "C" void init_tensor_list(struct tensor_list *list, int start, int num, const AicJitPointerArray *pointerArray)
{
    if (start < 0)
        return;
    list->sizes = (uint32_t *)pointerArray->pointers[start];
    list->widths = (uint32_t *)pointerArray->pointers[start + 1];
    list->heights = (uint32_t *)pointerArray->pointers[start + 2];
    list->depths = (uint32_t *)pointerArray->pointers[start + 3];
    list->zero_points = (int32_t *)pointerArray->pointers[start + 4];
    list->scales = (double *)pointerArray->pointers[start + 5];

    list->data = (uint8_t **)&pointerArray->pointers[start + 6];
}

struct tensor get_tensor(struct tensor_list *list, int idx)
{
    struct tensor tensor;
    tensor.size = list->sizes[idx];
    tensor.width = list->widths[idx];
    tensor.height = list->heights[idx];
    tensor.depth = list->depths[idx];
    tensor.zero_point = list->zero_points[idx];
    tensor.scale = list->scales[idx];
    tensor.data = list->data[idx];
    return tensor;
}

static inline void quantize_multiplier(double double_multiplier, int32_t *quantized_multiplier, int32_t *shift)
{
    if (double_multiplier == 0.0) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }

    const double q = frexp(double_multiplier, shift);
    int64_t q_fixed = (int64_t)(round(q * (1LL << 31)));

    if (q_fixed == (1LL << 31)) {
        q_fixed /= 2;
        ++*shift;
    }

    if (*shift < -31) {
        *shift = 0;
        q_fixed = 0;
    }

    *quantized_multiplier = (int)(q_fixed);
}

static inline int32_t multiply_by_quantized(int32_t x, int32_t quantized_multiplier, int32_t shift)
{
    int64_t total_shift = 31 - shift;

    if (total_shift <= 0 || total_shift >= 63) {
        // Invalid shift, return safe fallback
        return 0;
    }

    int64_t round = (int64_t)(1) << (total_shift - 1);
    int64_t result = x * (int64_t)(quantized_multiplier) + round;
    result = result >> total_shift;
    return (int32_t)(result);
}

extern "C" void tensor_add(struct tensor_list *io_list, int input1_idx, int input2_idx, int output_idx)
{
    uint8_t *input1_tens = io_list->data[input1_idx];
    uint8_t *input2_tens = io_list->data[input2_idx];
    uint8_t *output_tens = io_list->data[output_idx];

    uint32_t input1_size = io_list->sizes[input1_idx];
    uint32_t input2_size = io_list->sizes[input2_idx];
    uint32_t output_size = io_list->sizes[output_idx];

    int32_t input1_zero_point = io_list->zero_points[input1_idx];
    int32_t input2_zero_point = io_list->zero_points[input2_idx];
    int32_t output_zero_point = io_list->zero_points[output_idx];

    double input1_scale = io_list->scales[input1_idx];
    double input2_scale = io_list->scales[input2_idx];
    double output_scale = io_list->scales[output_idx];

    int32_t width = input1_size;

    for (int32_t i = 0; i < width; i++) {
        const int32_t input1_val = input1_tens[i] + -input1_zero_point;
        const int32_t input2_val = input2_tens[i] + -input2_zero_point;
        const int32_t shifted_input1_val = input1_val * (1 << 20);
        const int32_t shifted_input2_val = input2_val * (1 << 20);
        const double twice_max_input_scale = 2 * fmax(input1_scale, input2_scale);
        const double real_input1_multiplier = input1_scale / twice_max_input_scale;
        const double real_input2_multiplier = input2_scale / twice_max_input_scale;
        const double real_output_multiplier = twice_max_input_scale / ((1 << 20) * output_scale);
        int32_t input1_multiplier = 0;
        int32_t input1_shift = 0;
        int32_t input2_multiplier = 0;
        int32_t input2_shift = 0;
        int32_t output_multiplier = 0;
        int32_t output_shift = 0;

        quantize_multiplier(real_input1_multiplier, &input1_multiplier, &input1_shift);
        quantize_multiplier(real_input2_multiplier, &input2_multiplier, &input2_shift);
        quantize_multiplier(real_output_multiplier, &output_multiplier, &output_shift);

        const int32_t scaled_input1_val = multiply_by_quantized(shifted_input1_val, input1_multiplier, input1_shift);
        const int32_t scaled_input2_val = multiply_by_quantized(shifted_input2_val, input2_multiplier, input2_shift);
        const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
        const int32_t raw_output = multiply_by_quantized(raw_sum, output_multiplier, output_shift) + output_zero_point;
        const int32_t clamped_output = raw_output < 0 ? 0 : (raw_output > UINT8_MAX ? UINT8_MAX : raw_output);

        output_tens[i] = (unsigned char)(clamped_output);
    }
}

static inline int32_t idx(int32_t x_size, int32_t y_size, int32_t z_size, int32_t c_size, int32_t x, int32_t y, int32_t z, int32_t c)
{
    return x * (z_size) + y * (x_size * z_size) + z + c * (x_size * y_size * z_size);
}

extern "C" void tensor_conv(struct tensor_list *io_list, int input_idx, int output_idx, struct tensor_list *weight_list,
                             int weight_idx, struct tensor_list *bias_list, int bias_idx)
{
    uint8_t *input_tens = io_list->data[input_idx];
    uint8_t *weight_tens = weight_list->data[weight_idx];
    int32_t *bias_tens = (int32_t *)bias_list->data[bias_idx];
    uint8_t *output_tens = io_list->data[output_idx];

    int input_width = io_list->widths[input_idx];
    int input_height = io_list->heights[input_idx];
    int input_depth = io_list->depths[input_idx];
    int input_offset = io_list->zero_points[input_idx];
    double input_scale = io_list->scales[input_idx];
    int filter_width = weight_list->widths[weight_idx];
    int filter_height = weight_list->heights[weight_idx];
    int filter_depth = weight_list->depths[weight_idx];
    int filter_offset = weight_list->zero_points[weight_idx];
    double filter_scale = weight_list->scales[weight_idx];
    int output_channels = io_list->depths[output_idx];
    int output_width = io_list->widths[output_idx];
    int output_height = io_list->heights[output_idx];
    int output_offset = io_list->zero_points[output_idx];
    double output_scale = io_list->scales[output_idx];

    int stride_x = 1; // Hardcoding for now!
    int stride_y = 1; // Hardcoding for now!
    int has_padding = 0; // Hardcoding for now!

    int pad_x = 0;
    int pad_y = 0;
    if (has_padding) {
        // Compute the required input size based on the output size (number of
        // strides we will take plus the filter to get the end), then remove the
        // input we have.  This is the total required padding.
        pad_x = ((output_width - 1) * stride_x + filter_width - input_width) / 2;
        pad_y = ((output_height - 1) * stride_y + filter_height - input_height) / 2;
        pad_x = pad_x > 0 ? pad_x : 0;
        pad_y = pad_y > 0 ? pad_y : 0;
    }

    int sum = 0;

    // for every output layer (channel)
    for (int output_chan = 0; output_chan < output_channels; output_chan++) {
        // for every row in the image
        for (int input_y = 0 - pad_y, output_y = 0; output_y < output_height; input_y += stride_y, output_y++) {
            // for every column per row (every input element)
            for (int input_x = 0 - pad_x, output_x = 0; output_x < output_width; input_x += stride_x, output_x++) {
                sum = 0;
                // for every input channel in filter (layer).  Assume equal w/ input
                for (int filter_z = 0; filter_z < filter_depth; filter_z++) {
                    // for every row in the filter
                    for (int filter_y = 0; filter_y < filter_height; filter_y++) {
                        // for every column per filter row (every filter element)
                        for (int filter_x = 0; filter_x < filter_width; filter_x++) {
                            // If we are outside the input due to padding, skip mult
                            if (input_x + filter_x < 0 ||
                                input_x + filter_x >= input_width ||
                                input_y + filter_y < 0 ||
                                input_y + filter_y >= input_height)
                                continue;

                            int index = idx(filter_width, filter_height, filter_depth, output_channels,
                                             filter_x, filter_y, filter_z, output_chan);
                            int32_t filter_val = weight_tens[index];
                            index = idx(input_width, input_height, input_depth, 1,
                                        input_x + filter_x, input_y + filter_y, filter_z, 0);
                            int32_t input_val = input_tens[index];
                            sum += (filter_val - filter_offset) * (input_val - input_offset);
                        }
                    }
                }

                sum += bias_tens[output_chan];
                double real_multiplier = input_scale * filter_scale / output_scale;
                int32_t mult = 0;
                int32_t shift = 0;
                quantize_multiplier(real_multiplier, &mult, &shift);
                sum = multiply_by_quantized(sum, mult, shift);
                sum += output_offset;
                sum = sum < 0 ? 0 : (sum > UINT8_MAX ? UINT8_MAX : sum);
                int index = idx(output_width, output_height, output_channels, 1, output_x, output_y, output_chan, 0);
                output_tens[index] = (uint8_t)(sum);
            }
        }
    }
}
