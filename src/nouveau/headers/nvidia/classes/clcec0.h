/*******************************************************************************
    Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef HWFE_BLACKWELL_COMPUTE_B_SW_H
#define HWFE_BLACKWELL_COMPUTE_B_SW_H

#include "nvtypes.h"

#define BLACKWELL_COMPUTE_B    0xCEC0

#define NVCEC0_SET_OBJECT                                                                                  0x0000
#define NVCEC0_SET_OBJECT_CLASS_ID                                                                           15:0
#define NVCEC0_SET_OBJECT_ENGINE_ID                                                                         20:16

#define NVCEC0_NO_OPERATION                                                                                0x0100
#define NVCEC0_NO_OPERATION_V                                                                                31:0

#define NVCEC0_SET_NOTIFY_A                                                                                0x0104
#define NVCEC0_SET_NOTIFY_A_ADDRESS_UPPER                                                                    24:0

#define NVCEC0_SET_NOTIFY_B                                                                                0x0108
#define NVCEC0_SET_NOTIFY_B_ADDRESS_LOWER                                                                    31:0

#define NVCEC0_NOTIFY                                                                                      0x010c
#define NVCEC0_NOTIFY_TYPE                                                                                   31:0
#define NVCEC0_NOTIFY_TYPE_WRITE_ONLY                                                                  0x00000000
#define NVCEC0_NOTIFY_TYPE_WRITE_THEN_AWAKEN                                                           0x00000001

#define NVCEC0_WAIT_FOR_IDLE                                                                               0x0110
#define NVCEC0_WAIT_FOR_IDLE_V                                                                               31:0

#define NVCEC0_LOAD_MME_INSTRUCTION_RAM_POINTER                                                            0x0114
#define NVCEC0_LOAD_MME_INSTRUCTION_RAM_POINTER_V                                                            31:0

#define NVCEC0_LOAD_MME_INSTRUCTION_RAM                                                                    0x0118
#define NVCEC0_LOAD_MME_INSTRUCTION_RAM_V                                                                    31:0

#define NVCEC0_LOAD_MME_START_ADDRESS_RAM_POINTER                                                          0x011c
#define NVCEC0_LOAD_MME_START_ADDRESS_RAM_POINTER_V                                                          31:0

#define NVCEC0_LOAD_MME_START_ADDRESS_RAM                                                                  0x0120
#define NVCEC0_LOAD_MME_START_ADDRESS_RAM_V                                                                  31:0

#define NVCEC0_SET_MME_SHADOW_RAM_CONTROL                                                                  0x0124
#define NVCEC0_SET_MME_SHADOW_RAM_CONTROL_MODE                                                                1:0
#define NVCEC0_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK                                            0x00000000
#define NVCEC0_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK_WITH_FILTER                                0x00000001
#define NVCEC0_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_PASSTHROUGH                                      0x00000002
#define NVCEC0_SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_REPLAY                                           0x00000003

#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_A                                                                  0x0130
#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_A_OFFSET_UPPER                                                        7:0

#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_B                                                                  0x0134
#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_B_OFFSET_LOWER                                                       31:0

#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_C                                                                  0x0138
#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_C_MODE                                                                2:0
#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_C_MODE_FALSE                                                   0x00000000
#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_C_MODE_TRUE                                                    0x00000001
#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_C_MODE_CONDITIONAL                                             0x00000002
#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                         0x00000003
#define NVCEC0_SET_GLOBAL_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                     0x00000004

#define NVCEC0_SEND_GO_IDLE                                                                                0x013c
#define NVCEC0_SEND_GO_IDLE_V                                                                                31:0

#define NVCEC0_PM_TRIGGER                                                                                  0x0140
#define NVCEC0_PM_TRIGGER_V                                                                                  31:0

#define NVCEC0_PM_TRIGGER_WFI                                                                              0x0144
#define NVCEC0_PM_TRIGGER_WFI_V                                                                              31:0

#define NVCEC0_FE_ATOMIC_SEQUENCE_BEGIN                                                                    0x0148
#define NVCEC0_FE_ATOMIC_SEQUENCE_BEGIN_V                                                                    31:0

#define NVCEC0_FE_ATOMIC_SEQUENCE_END                                                                      0x014c
#define NVCEC0_FE_ATOMIC_SEQUENCE_END_V                                                                      31:0

#define NVCEC0_SET_INSTRUMENTATION_METHOD_HEADER                                                           0x0150
#define NVCEC0_SET_INSTRUMENTATION_METHOD_HEADER_V                                                           31:0

#define NVCEC0_SET_INSTRUMENTATION_METHOD_DATA                                                             0x0154
#define NVCEC0_SET_INSTRUMENTATION_METHOD_DATA_V                                                             31:0

#define NVCEC0_SET_REPORT_SEMAPHORE_PAYLOAD_LOWER                                                          0x0158
#define NVCEC0_SET_REPORT_SEMAPHORE_PAYLOAD_LOWER_PAYLOAD_LOWER                                              31:0

#define NVCEC0_SET_REPORT_SEMAPHORE_PAYLOAD_UPPER                                                          0x015c
#define NVCEC0_SET_REPORT_SEMAPHORE_PAYLOAD_UPPER_PAYLOAD_UPPER                                              31:0

#define NVCEC0_SET_REPORT_SEMAPHORE_ADDRESS_LOWER                                                          0x0160
#define NVCEC0_SET_REPORT_SEMAPHORE_ADDRESS_LOWER_LOWER                                                      31:0

#define NVCEC0_SET_REPORT_SEMAPHORE_ADDRESS_UPPER                                                          0x0164
#define NVCEC0_SET_REPORT_SEMAPHORE_ADDRESS_UPPER_UPPER                                                      24:0

#define NVCEC0_REPORT_SEMAPHORE_EXECUTE                                                                    0x0168
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_OPERATION                                                             1:0
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_OPERATION_RELEASE                                              0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_OPERATION_ACQUIRE                                              0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_OPERATION_REPORT_ONLY                                          0x00000002
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_OPERATION_TRAP                                                 0x00000003
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_AWAKEN_ENABLE                                                         2:2
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_AWAKEN_ENABLE_FALSE                                            0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_AWAKEN_ENABLE_TRUE                                             0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_STRUCTURE_SIZE                                                        4:3
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS                            0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_STRUCTURE_SIZE_SEMAPHORE_ONE_WORD                              0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS                             0x00000002
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_FLUSH_DISABLE                                                         5:5
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_FLUSH_DISABLE_FALSE                                            0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_FLUSH_DISABLE_TRUE                                             0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_ENABLE                                                      6:6
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_ENABLE_FALSE                                         0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_ENABLE_TRUE                                          0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP                                                          9:7
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP_RED_ADD                                           0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP_RED_MIN                                           0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP_RED_MAX                                           0x00000002
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP_RED_INC                                           0x00000003
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP_RED_DEC                                           0x00000004
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP_RED_AND                                           0x00000005
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP_RED_OR                                            0x00000006
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_OP_RED_XOR                                           0x00000007
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_FORMAT                                                    11:10
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_FORMAT_UNSIGNED                                      0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_REDUCTION_FORMAT_SIGNED                                        0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_PAYLOAD_SIZE64                                                      12:12
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_PAYLOAD_SIZE64_FALSE                                           0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_PAYLOAD_SIZE64_TRUE                                            0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_TRAP_TYPE                                                           14:13
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_TRAP_TYPE_TRAP_NONE                                            0x00000000
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_TRAP_TYPE_TRAP_UNCONDITIONAL                                   0x00000001
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_TRAP_TYPE_TRAP_CONDITIONAL                                     0x00000002
#define NVCEC0_REPORT_SEMAPHORE_EXECUTE_TRAP_TYPE_TRAP_CONDITIONAL_EXT                                 0x00000003

#define NVCEC0_LINE_LENGTH_IN                                                                              0x0180
#define NVCEC0_LINE_LENGTH_IN_VALUE                                                                          31:0

#define NVCEC0_LINE_COUNT                                                                                  0x0184
#define NVCEC0_LINE_COUNT_VALUE                                                                              31:0

#define NVCEC0_OFFSET_OUT_UPPER                                                                            0x0188
#define NVCEC0_OFFSET_OUT_UPPER_VALUE                                                                        24:0

#define NVCEC0_OFFSET_OUT                                                                                  0x018c
#define NVCEC0_OFFSET_OUT_VALUE                                                                              31:0

#define NVCEC0_PITCH_OUT                                                                                   0x0190
#define NVCEC0_PITCH_OUT_VALUE                                                                               31:0

#define NVCEC0_LAUNCH_DMA                                                                                  0x01b0
#define NVCEC0_LAUNCH_DMA_DST_MEMORY_LAYOUT                                                                   0:0
#define NVCEC0_LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR                                                0x00000000
#define NVCEC0_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH                                                      0x00000001
#define NVCEC0_LAUNCH_DMA_COMPLETION_TYPE                                                                     5:4
#define NVCEC0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_DISABLE                                                0x00000000
#define NVCEC0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY                                                   0x00000001
#define NVCEC0_LAUNCH_DMA_COMPLETION_TYPE_RELEASE_SEMAPHORE                                            0x00000002
#define NVCEC0_LAUNCH_DMA_INTERRUPT_TYPE                                                                      9:8
#define NVCEC0_LAUNCH_DMA_INTERRUPT_TYPE_NONE                                                          0x00000000
#define NVCEC0_LAUNCH_DMA_INTERRUPT_TYPE_INTERRUPT                                                     0x00000001
#define NVCEC0_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE                                                             12:12
#define NVCEC0_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_FOUR_WORDS                                             0x00000000
#define NVCEC0_LAUNCH_DMA_SEMAPHORE_STRUCT_SIZE_ONE_WORD                                               0x00000001
#define NVCEC0_LAUNCH_DMA_REDUCTION_ENABLE                                                                    1:1
#define NVCEC0_LAUNCH_DMA_REDUCTION_ENABLE_FALSE                                                       0x00000000
#define NVCEC0_LAUNCH_DMA_REDUCTION_ENABLE_TRUE                                                        0x00000001
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP                                                                      15:13
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP_RED_ADD                                                         0x00000000
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP_RED_MIN                                                         0x00000001
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP_RED_MAX                                                         0x00000002
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP_RED_INC                                                         0x00000003
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP_RED_DEC                                                         0x00000004
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP_RED_AND                                                         0x00000005
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP_RED_OR                                                          0x00000006
#define NVCEC0_LAUNCH_DMA_REDUCTION_OP_RED_XOR                                                         0x00000007
#define NVCEC0_LAUNCH_DMA_REDUCTION_FORMAT                                                                    3:2
#define NVCEC0_LAUNCH_DMA_REDUCTION_FORMAT_UNSIGNED_32                                                 0x00000000
#define NVCEC0_LAUNCH_DMA_REDUCTION_FORMAT_SIGNED_32                                                   0x00000001
#define NVCEC0_LAUNCH_DMA_SYSMEMBAR_DISABLE                                                                   6:6
#define NVCEC0_LAUNCH_DMA_SYSMEMBAR_DISABLE_FALSE                                                      0x00000000
#define NVCEC0_LAUNCH_DMA_SYSMEMBAR_DISABLE_TRUE                                                       0x00000001

#define NVCEC0_LOAD_INLINE_DATA                                                                            0x01b4
#define NVCEC0_LOAD_INLINE_DATA_V                                                                            31:0

#define NVCEC0_SET_I2M_SEMAPHORE_A                                                                         0x01dc
#define NVCEC0_SET_I2M_SEMAPHORE_A_OFFSET_UPPER                                                              24:0

#define NVCEC0_SET_I2M_SEMAPHORE_B                                                                         0x01e0
#define NVCEC0_SET_I2M_SEMAPHORE_B_OFFSET_LOWER                                                              31:0

#define NVCEC0_SET_I2M_SEMAPHORE_C                                                                         0x01e4
#define NVCEC0_SET_I2M_SEMAPHORE_C_PAYLOAD                                                                   31:0

#define NVCEC0_SET_SM_SCG_CONTROL                                                                          0x01e8
#define NVCEC0_SET_SM_SCG_CONTROL_COMPUTE_IN_GRAPHICS                                                         0:0
#define NVCEC0_SET_SM_SCG_CONTROL_COMPUTE_IN_GRAPHICS_FALSE                                            0x00000000
#define NVCEC0_SET_SM_SCG_CONTROL_COMPUTE_IN_GRAPHICS_TRUE                                             0x00000001

#define NVCEC0_SET_MME_SWITCH_STATE                                                                        0x01ec
#define NVCEC0_SET_MME_SWITCH_STATE_VALID                                                                     0:0
#define NVCEC0_SET_MME_SWITCH_STATE_VALID_FALSE                                                        0x00000000
#define NVCEC0_SET_MME_SWITCH_STATE_VALID_TRUE                                                         0x00000001
#define NVCEC0_SET_MME_SWITCH_STATE_SAVE_MACRO                                                               11:4
#define NVCEC0_SET_MME_SWITCH_STATE_RESTORE_MACRO                                                           19:12

#define NVCEC0_SET_I2M_SPARE_NOOP00                                                                        0x01f0
#define NVCEC0_SET_I2M_SPARE_NOOP00_V                                                                        31:0

#define NVCEC0_SET_I2M_SPARE_NOOP01                                                                        0x01f4
#define NVCEC0_SET_I2M_SPARE_NOOP01_V                                                                        31:0

#define NVCEC0_SET_I2M_SPARE_NOOP02                                                                        0x01f8
#define NVCEC0_SET_I2M_SPARE_NOOP02_V                                                                        31:0

#define NVCEC0_SET_I2M_SPARE_NOOP03                                                                        0x01fc
#define NVCEC0_SET_I2M_SPARE_NOOP03_V                                                                        31:0

#define NVCEC0_SET_VALID_SPAN_OVERFLOW_AREA_A                                                              0x0200
#define NVCEC0_SET_VALID_SPAN_OVERFLOW_AREA_A_ADDRESS_UPPER                                                   7:0

#define NVCEC0_SET_VALID_SPAN_OVERFLOW_AREA_B                                                              0x0204
#define NVCEC0_SET_VALID_SPAN_OVERFLOW_AREA_B_ADDRESS_LOWER                                                  31:0

#define NVCEC0_SET_VALID_SPAN_OVERFLOW_AREA_C                                                              0x0208
#define NVCEC0_SET_VALID_SPAN_OVERFLOW_AREA_C_SIZE                                                           31:0

#define NVCEC0_UPDATE_ALL_COMPUTE_CONTROL                                                                  0x020c
#define NVCEC0_UPDATE_ALL_COMPUTE_CONTROL_VALID                                                               0:0
#define NVCEC0_UPDATE_ALL_COMPUTE_CONTROL_VALID_FALSE                                                  0x00000000
#define NVCEC0_UPDATE_ALL_COMPUTE_CONTROL_VALID_TRUE                                                   0x00000001
#define NVCEC0_UPDATE_ALL_COMPUTE_CONTROL_STICK_IN_COMPUTE_IN_GRAPHICS_MODE                                   4:4
#define NVCEC0_UPDATE_ALL_COMPUTE_CONTROL_STICK_IN_COMPUTE_IN_GRAPHICS_MODE_FALSE                      0x00000000
#define NVCEC0_UPDATE_ALL_COMPUTE_CONTROL_STICK_IN_COMPUTE_IN_GRAPHICS_MODE_TRUE                       0x00000001

#define NVCEC0_PERFMON_TRANSFER                                                                            0x0210
#define NVCEC0_PERFMON_TRANSFER_V                                                                            31:0

#define NVCEC0_INVALIDATE_SHADER_CACHES                                                                    0x021c
#define NVCEC0_INVALIDATE_SHADER_CACHES_INSTRUCTION                                                           0:0
#define NVCEC0_INVALIDATE_SHADER_CACHES_INSTRUCTION_FALSE                                              0x00000000
#define NVCEC0_INVALIDATE_SHADER_CACHES_INSTRUCTION_TRUE                                               0x00000001
#define NVCEC0_INVALIDATE_SHADER_CACHES_DATA                                                                  4:4
#define NVCEC0_INVALIDATE_SHADER_CACHES_DATA_FALSE                                                     0x00000000
#define NVCEC0_INVALIDATE_SHADER_CACHES_DATA_TRUE                                                      0x00000001
#define NVCEC0_INVALIDATE_SHADER_CACHES_CONSTANT                                                            12:12
#define NVCEC0_INVALIDATE_SHADER_CACHES_CONSTANT_FALSE                                                 0x00000000
#define NVCEC0_INVALIDATE_SHADER_CACHES_CONSTANT_TRUE                                                  0x00000001
#define NVCEC0_INVALIDATE_SHADER_CACHES_LOCKS                                                                 1:1
#define NVCEC0_INVALIDATE_SHADER_CACHES_LOCKS_FALSE                                                    0x00000000
#define NVCEC0_INVALIDATE_SHADER_CACHES_LOCKS_TRUE                                                     0x00000001
#define NVCEC0_INVALIDATE_SHADER_CACHES_FLUSH_DATA                                                            2:2
#define NVCEC0_INVALIDATE_SHADER_CACHES_FLUSH_DATA_FALSE                                               0x00000000
#define NVCEC0_INVALIDATE_SHADER_CACHES_FLUSH_DATA_TRUE                                                0x00000001

#define NVCEC0_SET_RESERVED_SW_METHOD00                                                                    0x0220
#define NVCEC0_SET_RESERVED_SW_METHOD00_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD01                                                                    0x0224
#define NVCEC0_SET_RESERVED_SW_METHOD01_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD02                                                                    0x0228
#define NVCEC0_SET_RESERVED_SW_METHOD02_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD03                                                                    0x022c
#define NVCEC0_SET_RESERVED_SW_METHOD03_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD04                                                                    0x0230
#define NVCEC0_SET_RESERVED_SW_METHOD04_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD05                                                                    0x0234
#define NVCEC0_SET_RESERVED_SW_METHOD05_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD06                                                                    0x0238
#define NVCEC0_SET_RESERVED_SW_METHOD06_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD07                                                                    0x023c
#define NVCEC0_SET_RESERVED_SW_METHOD07_V                                                                    31:0

#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI                                                      0x0244
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES                                                   0:0
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES_ALL                                        0x00000000
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_LINES_ONE                                        0x00000001
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI_TAG                                                    25:4

#define NVCEC0_SET_CWD_REF_COUNTER                                                                         0x0248
#define NVCEC0_SET_CWD_REF_COUNTER_SELECT                                                                     5:0
#define NVCEC0_SET_CWD_REF_COUNTER_VALUE                                                                     23:8

#define NVCEC0_SET_RESERVED_SW_METHOD08                                                                    0x024c
#define NVCEC0_SET_RESERVED_SW_METHOD08_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD09                                                                    0x0250
#define NVCEC0_SET_RESERVED_SW_METHOD09_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD10                                                                    0x0254
#define NVCEC0_SET_RESERVED_SW_METHOD10_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD11                                                                    0x0258
#define NVCEC0_SET_RESERVED_SW_METHOD11_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD12                                                                    0x025c
#define NVCEC0_SET_RESERVED_SW_METHOD12_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD13                                                                    0x0260
#define NVCEC0_SET_RESERVED_SW_METHOD13_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD14                                                                    0x0264
#define NVCEC0_SET_RESERVED_SW_METHOD14_V                                                                    31:0

#define NVCEC0_SET_RESERVED_SW_METHOD15                                                                    0x0268
#define NVCEC0_SET_RESERVED_SW_METHOD15_V                                                                    31:0

#define NVCEC0_SET_COMPUTE_CLASS_VERSION                                                                   0x0280
#define NVCEC0_SET_COMPUTE_CLASS_VERSION_CURRENT                                                             15:0
#define NVCEC0_SET_COMPUTE_CLASS_VERSION_OLDEST_SUPPORTED                                                   31:16

#define NVCEC0_CHECK_COMPUTE_CLASS_VERSION                                                                 0x0284
#define NVCEC0_CHECK_COMPUTE_CLASS_VERSION_CURRENT                                                           15:0
#define NVCEC0_CHECK_COMPUTE_CLASS_VERSION_OLDEST_SUPPORTED                                                 31:16

#define NVCEC0_SET_QMD_VERSION                                                                             0x0288
#define NVCEC0_SET_QMD_VERSION_CURRENT                                                                       15:0
#define NVCEC0_SET_QMD_VERSION_OLDEST_SUPPORTED                                                             31:16

#define NVCEC0_INVALIDATE_SKED_CACHES                                                                      0x0298
#define NVCEC0_INVALIDATE_SKED_CACHES_FORCE_IGNORE_VEID                                                       0:0
#define NVCEC0_INVALIDATE_SKED_CACHES_FORCE_IGNORE_VEID_FALSE                                          0x00000000
#define NVCEC0_INVALIDATE_SKED_CACHES_FORCE_IGNORE_VEID_TRUE                                           0x00000001

#define NVCEC0_SET_SHADER_SHARED_MEMORY_WINDOW_A                                                           0x02a0
#define NVCEC0_SET_SHADER_SHARED_MEMORY_WINDOW_A_BASE_ADDRESS_UPPER                                          16:0

#define NVCEC0_SET_SHADER_SHARED_MEMORY_WINDOW_B                                                           0x02a4
#define NVCEC0_SET_SHADER_SHARED_MEMORY_WINDOW_B_BASE_ADDRESS                                                31:0

#define NVCEC0_SET_CWD_SLOT_COUNT                                                                          0x02b0
#define NVCEC0_SET_CWD_SLOT_COUNT_V                                                                           7:0

#define NVCEC0_SEND_PCAS_A                                                                                 0x02b4
#define NVCEC0_SEND_PCAS_A_QMD_ADDRESS_SHIFTED8                                                              31:0

#define NVCEC0_SEND_PCAS_B                                                                                 0x02b8
#define NVCEC0_SEND_PCAS_B_FROM                                                                              23:0
#define NVCEC0_SEND_PCAS_B_DELTA                                                                            31:24

#define NVCEC0_SEND_SIGNALING_PCAS_B                                                                       0x02bc
#define NVCEC0_SEND_SIGNALING_PCAS_B_INVALIDATE                                                               0:0
#define NVCEC0_SEND_SIGNALING_PCAS_B_INVALIDATE_FALSE                                                  0x00000000
#define NVCEC0_SEND_SIGNALING_PCAS_B_INVALIDATE_TRUE                                                   0x00000001
#define NVCEC0_SEND_SIGNALING_PCAS_B_SCHEDULE                                                                 1:1
#define NVCEC0_SEND_SIGNALING_PCAS_B_SCHEDULE_FALSE                                                    0x00000000
#define NVCEC0_SEND_SIGNALING_PCAS_B_SCHEDULE_TRUE                                                     0x00000001

#define NVCEC0_SEND_SIGNALING_PCAS2_B                                                                      0x02c0
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION                                                             3:0
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_NOP                                                  0x00000000
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE                                           0x00000001
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_SCHEDULE                                             0x00000002
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE                             0x00000003
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_PUT                                        0x00000006
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_DECREMENT_DEPENDENCE                                 0x00000007
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH                                             0x00000008
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE                                    0x00000009
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_PREFETCH_COPY_SCHEDULE                    0x0000000A
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_PREFETCH_COPY_FORCE_REQUIRE_SCHEDULING             0x0000000B
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_DEPENDENCE                                 0x0000000C
#define NVCEC0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_CWD_REF_COUNTER                            0x0000000D
#define NVCEC0_SEND_SIGNALING_PCAS2_B_SELECT                                                                 13:8
#define NVCEC0_SEND_SIGNALING_PCAS2_B_OFFSET_MINUS_ONE                                                      23:14

#define NVCEC0_SET_SKED_CACHE_CONTROL                                                                      0x02cc
#define NVCEC0_SET_SKED_CACHE_CONTROL_IGNORE_VEID                                                             0:0
#define NVCEC0_SET_SKED_CACHE_CONTROL_IGNORE_VEID_FALSE                                                0x00000000
#define NVCEC0_SET_SKED_CACHE_CONTROL_IGNORE_VEID_TRUE                                                 0x00000001

#define NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A                                                     0x02e4
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A_SIZE_UPPER                                             7:0

#define NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B                                                     0x02e8
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B_SIZE_LOWER                                            31:0

#define NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C                                                     0x02ec
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C_MAX_SM_COUNT                                           8:0
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C_MAX_PLURAL_TPC_GPC_COUNT                             23:16
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C_MAX_SINGLETON_TPC_GPC_COUNT                          31:24

#define NVCEC0_SET_SPA_VERSION                                                                             0x0310
#define NVCEC0_SET_SPA_VERSION_MINOR                                                                          7:0
#define NVCEC0_SET_SPA_VERSION_MAJOR                                                                         15:8

#define NVCEC0_SET_INLINE_QMD_ADDRESS_A                                                                    0x0318
#define NVCEC0_SET_INLINE_QMD_ADDRESS_A_QMD_ADDRESS_SHIFTED8_UPPER                                            8:0
#define NVCEC0_SET_INLINE_QMD_ADDRESS_A_INLINE_SIZE                                                         30:30
#define NVCEC0_SET_INLINE_QMD_ADDRESS_A_INLINE_SIZE_INLINE_256                                         0x00000000
#define NVCEC0_SET_INLINE_QMD_ADDRESS_A_INLINE_SIZE_INLINE_384                                         0x00000001

#define NVCEC0_SET_INLINE_QMD_ADDRESS_B                                                                    0x031c
#define NVCEC0_SET_INLINE_QMD_ADDRESS_B_QMD_ADDRESS_SHIFTED8_LOWER                                           31:0

#define NVCEC0_LOAD_INLINE_QMD_DATA(i)                                                             (0x0320+(i)*4)
#define NVCEC0_LOAD_INLINE_QMD_DATA_V                                                                        31:0

#define NVCEC0_SET_FALCON00                                                                                0x0500
#define NVCEC0_SET_FALCON00_V                                                                                31:0

#define NVCEC0_SET_FALCON01                                                                                0x0504
#define NVCEC0_SET_FALCON01_V                                                                                31:0

#define NVCEC0_SET_FALCON02                                                                                0x0508
#define NVCEC0_SET_FALCON02_V                                                                                31:0

#define NVCEC0_SET_FALCON03                                                                                0x050c
#define NVCEC0_SET_FALCON03_V                                                                                31:0

#define NVCEC0_SET_FALCON04                                                                                0x0510
#define NVCEC0_SET_FALCON04_V                                                                                31:0

#define NVCEC0_SET_FALCON05                                                                                0x0514
#define NVCEC0_SET_FALCON05_V                                                                                31:0

#define NVCEC0_SET_FALCON06                                                                                0x0518
#define NVCEC0_SET_FALCON06_V                                                                                31:0

#define NVCEC0_SET_FALCON07                                                                                0x051c
#define NVCEC0_SET_FALCON07_V                                                                                31:0

#define NVCEC0_SET_FALCON08                                                                                0x0520
#define NVCEC0_SET_FALCON08_V                                                                                31:0

#define NVCEC0_SET_FALCON09                                                                                0x0524
#define NVCEC0_SET_FALCON09_V                                                                                31:0

#define NVCEC0_SET_FALCON10                                                                                0x0528
#define NVCEC0_SET_FALCON10_V                                                                                31:0

#define NVCEC0_SET_FALCON11                                                                                0x052c
#define NVCEC0_SET_FALCON11_V                                                                                31:0

#define NVCEC0_SET_FALCON12                                                                                0x0530
#define NVCEC0_SET_FALCON12_V                                                                                31:0

#define NVCEC0_SET_FALCON13                                                                                0x0534
#define NVCEC0_SET_FALCON13_V                                                                                31:0

#define NVCEC0_SET_FALCON14                                                                                0x0538
#define NVCEC0_SET_FALCON14_V                                                                                31:0

#define NVCEC0_SET_FALCON15                                                                                0x053c
#define NVCEC0_SET_FALCON15_V                                                                                31:0

#define NVCEC0_SET_MME_MEM_ADDRESS_A                                                                       0x0550
#define NVCEC0_SET_MME_MEM_ADDRESS_A_UPPER                                                                   24:0

#define NVCEC0_SET_MME_MEM_ADDRESS_B                                                                       0x0554
#define NVCEC0_SET_MME_MEM_ADDRESS_B_LOWER                                                                   31:0

#define NVCEC0_SET_MME_DATA_RAM_ADDRESS                                                                    0x0558
#define NVCEC0_SET_MME_DATA_RAM_ADDRESS_WORD                                                                 31:0

#define NVCEC0_MME_DMA_READ                                                                                0x055c
#define NVCEC0_MME_DMA_READ_LENGTH                                                                           31:0

#define NVCEC0_MME_DMA_READ_FIFOED                                                                         0x0560
#define NVCEC0_MME_DMA_READ_FIFOED_LENGTH                                                                    31:0

#define NVCEC0_MME_DMA_WRITE                                                                               0x0564
#define NVCEC0_MME_DMA_WRITE_LENGTH                                                                          31:0

#define NVCEC0_MME_DMA_REDUCTION                                                                           0x0568
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP                                                                 2:0
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP_RED_ADD                                                  0x00000000
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP_RED_MIN                                                  0x00000001
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP_RED_MAX                                                  0x00000002
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP_RED_INC                                                  0x00000003
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP_RED_DEC                                                  0x00000004
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP_RED_AND                                                  0x00000005
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP_RED_OR                                                   0x00000006
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_OP_RED_XOR                                                  0x00000007
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_FORMAT                                                             5:4
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_FORMAT_UNSIGNED                                             0x00000000
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_FORMAT_SIGNED                                               0x00000001
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_SIZE                                                               8:8
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_SIZE_FOUR_BYTES                                             0x00000000
#define NVCEC0_MME_DMA_REDUCTION_REDUCTION_SIZE_EIGHT_BYTES                                            0x00000001

#define NVCEC0_MME_DMA_SYSMEMBAR                                                                           0x056c
#define NVCEC0_MME_DMA_SYSMEMBAR_V                                                                            0:0

#define NVCEC0_MME_DMA_SYNC                                                                                0x0570
#define NVCEC0_MME_DMA_SYNC_VALUE                                                                            31:0

#define NVCEC0_EVENT_START_TRIGGER                                                                         0x074c
#define NVCEC0_EVENT_START_TRIGGER_V                                                                         31:0

#define NVCEC0_EVENT_STOP_TRIGGER                                                                          0x075c
#define NVCEC0_EVENT_STOP_TRIGGER_V                                                                          31:0

#define NVCEC0_SET_SHADER_LOCAL_MEMORY_A                                                                   0x0790
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_A_ADDRESS_UPPER                                                       24:0

#define NVCEC0_SET_SHADER_LOCAL_MEMORY_B                                                                   0x0794
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_B_ADDRESS_LOWER                                                       31:0

#define NVCEC0_SET_TEXTURE_HEADER_VERSION                                                                  0x07ac
#define NVCEC0_SET_TEXTURE_HEADER_VERSION_MAJOR                                                               7:0

#define NVCEC0_SET_SHADER_LOCAL_MEMORY_WINDOW_A                                                            0x07b0
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_WINDOW_A_BASE_ADDRESS_UPPER                                           16:0

#define NVCEC0_SET_SHADER_LOCAL_MEMORY_WINDOW_B                                                            0x07b4
#define NVCEC0_SET_SHADER_LOCAL_MEMORY_WINDOW_B_BASE_ADDRESS                                                 31:0

#define NVCEC0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A                                                   0x07b8
#define NVCEC0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A_LOWER                                               31:0

#define NVCEC0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B                                                   0x07bc
#define NVCEC0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B_UPPER                                               24:0

#define NVCEC0_THROTTLE_SM                                                                                 0x07fc
#define NVCEC0_THROTTLE_SM_MULTIPLY_ADD                                                                       0:0
#define NVCEC0_THROTTLE_SM_MULTIPLY_ADD_FALSE                                                          0x00000000
#define NVCEC0_THROTTLE_SM_MULTIPLY_ADD_TRUE                                                           0x00000001

#define NVCEC0_SET_SHADER_CACHE_CONTROL                                                                    0x0d94
#define NVCEC0_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE                                                0:0
#define NVCEC0_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE_FALSE                                   0x00000000
#define NVCEC0_SET_SHADER_CACHE_CONTROL_ICACHE_PREFETCH_ENABLE_TRUE                                    0x00000001

#define NVCEC0_SET_SCG_COMPUTE_SCHEDULING_PARAMETERS(i)                                            (0x0da0+(i)*4)
#define NVCEC0_SET_SCG_COMPUTE_SCHEDULING_PARAMETERS_V                                                       31:0

#define NVCEC0_SET_SM_TIMEOUT_INTERVAL                                                                     0x0de4
#define NVCEC0_SET_SM_TIMEOUT_INTERVAL_COUNTER_BIT                                                            5:0

#define NVCEC0_MME_DMA_WRITE_METHOD_BARRIER                                                                0x0dec
#define NVCEC0_MME_DMA_WRITE_METHOD_BARRIER_V                                                                 0:0

#define NVCEC0_SET_WFI_FILTER_STATE                                                                        0x0f10
#define NVCEC0_SET_WFI_FILTER_STATE_FILTER_HOST_WFI                                                           0:0
#define NVCEC0_SET_WFI_FILTER_STATE_FILTER_HOST_WFI_FALSE                                              0x00000000
#define NVCEC0_SET_WFI_FILTER_STATE_FILTER_HOST_WFI_TRUE                                               0x00000001

#define NVCEC0_EVENT_STOP_TRIGGER_WFI                                                                      0x0ffc
#define NVCEC0_EVENT_STOP_TRIGGER_WFI_V                                                                      31:0

#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI                                                        0x1288
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES                                                     0:0
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES_ALL                                          0x00000000
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI_LINES_ONE                                          0x00000001
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI_TAG                                                      25:4

#define NVCEC0_ACTIVATE_PERF_SETTINGS_FOR_COMPUTE_CONTEXT                                                  0x12a8
#define NVCEC0_ACTIVATE_PERF_SETTINGS_FOR_COMPUTE_CONTEXT_ALL                                                 0:0
#define NVCEC0_ACTIVATE_PERF_SETTINGS_FOR_COMPUTE_CONTEXT_ALL_FALSE                                    0x00000000
#define NVCEC0_ACTIVATE_PERF_SETTINGS_FOR_COMPUTE_CONTEXT_ALL_TRUE                                     0x00000001

#define NVCEC0_INVALIDATE_SAMPLER_CACHE                                                                    0x1330
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_LINES                                                                 0:0
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_LINES_ALL                                                      0x00000000
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_LINES_ONE                                                      0x00000001
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_TAG                                                                  25:4

#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE                                                             0x1334
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_LINES                                                          0:0
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_LINES_ALL                                               0x00000000
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_LINES_ONE                                               0x00000001
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_TAG                                                           25:4

#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE                                                               0x1338
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_LINES                                                            0:0
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_LINES_ALL                                                 0x00000000
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_LINES_ONE                                                 0x00000001
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_TAG                                                             25:4

#define NVCEC0_INVALIDATE_SAMPLER_CACHE_NO_WFI                                                             0x1424
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_NO_WFI_LINES                                                          0:0
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_NO_WFI_LINES_ALL                                               0x00000000
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_NO_WFI_LINES_ONE                                               0x00000001
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_NO_WFI_TAG                                                           25:4

#define NVCEC0_INVALIDATE_SAMPLER_CACHE_BY_TAG                                                             0x1458
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_BY_TAG_TAG                                                           31:0

#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_BY_TAG                                                      0x145c
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_BY_TAG_TAG                                                    31:0

#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_BY_TAG                                                        0x1460
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_BY_TAG_TAG                                                      31:0

#define NVCEC0_SET_MME_TASK_EVENT_STATE                                                                    0x1464
#define NVCEC0_SET_MME_TASK_EVENT_STATE_FLUSH_MACRO                                                           7:0
#define NVCEC0_SET_MME_TASK_EVENT_STATE_TSG_EVENT_ENABLE                                                      8:8
#define NVCEC0_SET_MME_TASK_EVENT_STATE_TSG_EVENT_ENABLE_FALSE                                         0x00000000
#define NVCEC0_SET_MME_TASK_EVENT_STATE_TSG_EVENT_ENABLE_TRUE                                          0x00000001
#define NVCEC0_SET_MME_TASK_EVENT_STATE_HOST_WFI_ENABLE                                                       9:9
#define NVCEC0_SET_MME_TASK_EVENT_STATE_HOST_WFI_ENABLE_FALSE                                          0x00000000
#define NVCEC0_SET_MME_TASK_EVENT_STATE_HOST_WFI_ENABLE_TRUE                                           0x00000001
#define NVCEC0_SET_MME_TASK_EVENT_STATE_FE_COMPUTE_METHOD_WFI_ENABLE                                        10:10
#define NVCEC0_SET_MME_TASK_EVENT_STATE_FE_COMPUTE_METHOD_WFI_ENABLE_FALSE                             0x00000000
#define NVCEC0_SET_MME_TASK_EVENT_STATE_FE_COMPUTE_METHOD_WFI_ENABLE_TRUE                              0x00000001
#define NVCEC0_SET_MME_TASK_EVENT_STATE_FE_SUBCH_SWITCH_WFI_ENABLE                                          11:11
#define NVCEC0_SET_MME_TASK_EVENT_STATE_FE_SUBCH_SWITCH_WFI_ENABLE_FALSE                               0x00000000
#define NVCEC0_SET_MME_TASK_EVENT_STATE_FE_SUBCH_SWITCH_WFI_ENABLE_TRUE                                0x00000001

#define NVCEC0_SET_SHADER_EXCEPTIONS                                                                       0x1528
#define NVCEC0_SET_SHADER_EXCEPTIONS_ENABLE                                                                   0:0
#define NVCEC0_SET_SHADER_EXCEPTIONS_ENABLE_FALSE                                                      0x00000000
#define NVCEC0_SET_SHADER_EXCEPTIONS_ENABLE_TRUE                                                       0x00000001

#define NVCEC0_SET_RENDER_ENABLE_A                                                                         0x1550
#define NVCEC0_SET_RENDER_ENABLE_A_OFFSET_UPPER                                                               7:0

#define NVCEC0_SET_RENDER_ENABLE_B                                                                         0x1554
#define NVCEC0_SET_RENDER_ENABLE_B_OFFSET_LOWER                                                              31:0

#define NVCEC0_SET_RENDER_ENABLE_C                                                                         0x1558
#define NVCEC0_SET_RENDER_ENABLE_C_MODE                                                                       2:0
#define NVCEC0_SET_RENDER_ENABLE_C_MODE_FALSE                                                          0x00000000
#define NVCEC0_SET_RENDER_ENABLE_C_MODE_TRUE                                                           0x00000001
#define NVCEC0_SET_RENDER_ENABLE_C_MODE_CONDITIONAL                                                    0x00000002
#define NVCEC0_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL                                                0x00000003
#define NVCEC0_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL                                            0x00000004

#define NVCEC0_SET_TEX_SAMPLER_POOL_A                                                                      0x155c
#define NVCEC0_SET_TEX_SAMPLER_POOL_A_OFFSET_UPPER                                                           24:0

#define NVCEC0_SET_TEX_SAMPLER_POOL_B                                                                      0x1560
#define NVCEC0_SET_TEX_SAMPLER_POOL_B_OFFSET_LOWER                                                           31:0

#define NVCEC0_SET_TEX_SAMPLER_POOL_C                                                                      0x1564
#define NVCEC0_SET_TEX_SAMPLER_POOL_C_MAXIMUM_INDEX                                                          31:0

#define NVCEC0_SET_TEX_HEADER_POOL_A                                                                       0x1574
#define NVCEC0_SET_TEX_HEADER_POOL_A_OFFSET_UPPER                                                            24:0

#define NVCEC0_SET_TEX_HEADER_POOL_B                                                                       0x1578
#define NVCEC0_SET_TEX_HEADER_POOL_B_OFFSET_LOWER                                                            31:0

#define NVCEC0_SET_TEX_HEADER_POOL_C                                                                       0x157c
#define NVCEC0_SET_TEX_HEADER_POOL_C_MAXIMUM_INDEX                                                           31:0

#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI                                                             0x1698
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION                                                    0:0
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_FALSE                                       0x00000000
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE                                        0x00000001
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA                                                    4:4
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_FALSE                                       0x00000000
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE                                        0x00000001
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT                                                     12:12
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_FALSE                                          0x00000000
#define NVCEC0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE                                           0x00000001

#define NVCEC0_INVALIDATE_SAMPLER_CACHE_BY_TAG_NO_WFI                                                      0x1930
#define NVCEC0_INVALIDATE_SAMPLER_CACHE_BY_TAG_NO_WFI_TAG                                                    31:0

#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_BY_TAG_NO_WFI                                               0x1934
#define NVCEC0_INVALIDATE_TEXTURE_HEADER_CACHE_BY_TAG_NO_WFI_TAG                                             31:0

#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_BY_TAG_NO_WFI                                                 0x1938
#define NVCEC0_INVALIDATE_TEXTURE_DATA_CACHE_BY_TAG_NO_WFI_TAG                                               31:0

#define NVCEC0_SET_RENDER_ENABLE_OVERRIDE                                                                  0x1944
#define NVCEC0_SET_RENDER_ENABLE_OVERRIDE_MODE                                                                1:0
#define NVCEC0_SET_RENDER_ENABLE_OVERRIDE_MODE_USE_RENDER_ENABLE                                       0x00000000
#define NVCEC0_SET_RENDER_ENABLE_OVERRIDE_MODE_ALWAYS_RENDER                                           0x00000001
#define NVCEC0_SET_RENDER_ENABLE_OVERRIDE_MODE_NEVER_RENDER                                            0x00000002

#define NVCEC0_PIPE_NOP                                                                                    0x1a2c
#define NVCEC0_PIPE_NOP_V                                                                                    31:0

#define NVCEC0_SET_SPARE00                                                                                 0x1a30
#define NVCEC0_SET_SPARE00_V                                                                                 31:0

#define NVCEC0_SET_SPARE01                                                                                 0x1a34
#define NVCEC0_SET_SPARE01_V                                                                                 31:0

#define NVCEC0_SET_SPARE02                                                                                 0x1a38
#define NVCEC0_SET_SPARE02_V                                                                                 31:0

#define NVCEC0_SET_SPARE03                                                                                 0x1a3c
#define NVCEC0_SET_SPARE03_V                                                                                 31:0

#define NVCEC0_SET_REPORT_SEMAPHORE_A                                                                      0x1b00
#define NVCEC0_SET_REPORT_SEMAPHORE_A_OFFSET_UPPER                                                           24:0

#define NVCEC0_SET_REPORT_SEMAPHORE_B                                                                      0x1b04
#define NVCEC0_SET_REPORT_SEMAPHORE_B_OFFSET_LOWER                                                           31:0

#define NVCEC0_SET_REPORT_SEMAPHORE_C                                                                      0x1b08
#define NVCEC0_SET_REPORT_SEMAPHORE_C_PAYLOAD                                                                31:0

#define NVCEC0_SET_REPORT_SEMAPHORE_D                                                                      0x1b0c
#define NVCEC0_SET_REPORT_SEMAPHORE_D_OPERATION                                                               1:0
#define NVCEC0_SET_REPORT_SEMAPHORE_D_OPERATION_RELEASE                                                0x00000000
#define NVCEC0_SET_REPORT_SEMAPHORE_D_OPERATION_TRAP                                                   0x00000003
#define NVCEC0_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE                                                         20:20
#define NVCEC0_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                              0x00000000
#define NVCEC0_SET_REPORT_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                               0x00000001
#define NVCEC0_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE                                                        28:28
#define NVCEC0_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_FOUR_WORDS                                        0x00000000
#define NVCEC0_SET_REPORT_SEMAPHORE_D_STRUCTURE_SIZE_ONE_WORD                                          0x00000001
#define NVCEC0_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE                                                           2:2
#define NVCEC0_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                              0x00000000
#define NVCEC0_SET_REPORT_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                               0x00000001
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE                                                        3:3
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE_FALSE                                           0x00000000
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_ENABLE_TRUE                                            0x00000001
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP                                                           11:9
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_ADD                                             0x00000000
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_MIN                                             0x00000001
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_MAX                                             0x00000002
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_INC                                             0x00000003
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_DEC                                             0x00000004
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_AND                                             0x00000005
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_OR                                              0x00000006
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_OP_RED_XOR                                             0x00000007
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT                                                      18:17
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT_UNSIGNED_32                                     0x00000000
#define NVCEC0_SET_REPORT_SEMAPHORE_D_REDUCTION_FORMAT_SIGNED_32                                       0x00000001
#define NVCEC0_SET_REPORT_SEMAPHORE_D_CONDITIONAL_TRAP                                                      19:19
#define NVCEC0_SET_REPORT_SEMAPHORE_D_CONDITIONAL_TRAP_FALSE                                           0x00000000
#define NVCEC0_SET_REPORT_SEMAPHORE_D_CONDITIONAL_TRAP_TRUE                                            0x00000001

#define NVCEC0_SET_TRAP_HANDLER_A                                                                          0x25f8
#define NVCEC0_SET_TRAP_HANDLER_A_ADDRESS_UPPER                                                              24:0

#define NVCEC0_SET_TRAP_HANDLER_B                                                                          0x25fc
#define NVCEC0_SET_TRAP_HANDLER_B_ADDRESS_LOWER                                                              31:0

#define NVCEC0_SET_BINDLESS_TEXTURE                                                                        0x2608
#define NVCEC0_SET_BINDLESS_TEXTURE_CONSTANT_BUFFER_SLOT_SELECT                                               2:0

#define NVCEC0_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE(i)                                    (0x32f4+(i)*4)
#define NVCEC0_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_V                                               31:0

#define NVCEC0_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_UPPER(i)                              (0x3314+(i)*4)
#define NVCEC0_SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE_UPPER_V                                         31:0

#define NVCEC0_ENABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER                                                  0x3334
#define NVCEC0_ENABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_V                                                   0:0

#define NVCEC0_DISABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER                                                 0x3338
#define NVCEC0_DISABLE_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_V                                                  0:0

#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_VALUE_UPPER(i)                                       (0x333c+(i)*4)
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_VALUE_UPPER_V                                                  31:0

#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_VALUE(i)                                             (0x335c+(i)*4)
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_VALUE_V                                                        31:0

#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_EVENT(i)                                             (0x337c+(i)*4)
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_EVENT_EVENT                                                     7:0

#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A(i)                                         (0x339c+(i)*4)
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT0                                                1:0
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT0                                           4:2
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT1                                                6:5
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT1                                           9:7
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT2                                              11:10
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT2                                         14:12
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT3                                              16:15
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT3                                         19:17
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT4                                              21:20
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT4                                         24:22
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_EVENT5                                              26:25
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_BIT_SELECT5                                         29:27
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A_SPARE                                               31:30

#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B(i)                                         (0x33bc+(i)*4)
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_EDGE                                                  0:0
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_MODE                                                  2:1
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_WINDOWED                                              3:3
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B_FUNC                                                 19:4

#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_TRAP_CONTROL                                                 0x33dc
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_TRAP_CONTROL_MASK                                               7:0

#define NVCEC0_START_SHADER_PERFORMANCE_COUNTER                                                            0x33e0
#define NVCEC0_START_SHADER_PERFORMANCE_COUNTER_COUNTER_MASK                                                  7:0

#define NVCEC0_STOP_SHADER_PERFORMANCE_COUNTER                                                             0x33e4
#define NVCEC0_STOP_SHADER_PERFORMANCE_COUNTER_COUNTER_MASK                                                   7:0

#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_SCTL_FILTER                                                  0x33e8
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_SCTL_FILTER_V                                                  31:0

#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CORE_MIO_FILTER                                              0x33ec
#define NVCEC0_SET_SHADER_PERFORMANCE_COUNTER_CORE_MIO_FILTER_V                                              31:0

#define NVCEC0_SET_MME_SHADOW_SCRATCH(i)                                                           (0x3400+(i)*4)
#define NVCEC0_SET_MME_SHADOW_SCRATCH_V                                                                      31:0

#define NVCEC0_CALL_MME_MACRO(j)                                                                   (0x3800+(j)*8)
#define NVCEC0_CALL_MME_MACRO_V                                                                              31:0

#define NVCEC0_CALL_MME_DATA(j)                                                                    (0x3804+(j)*8)
#define NVCEC0_CALL_MME_DATA_V                                                                               31:0

#endif /* HWFE_BLACKWELL_COMPUTE_B_SW_H */
