/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef NSP_QSHIM_JIT_INTERFACE_H
#define NSP_QSHIM_JIT_INTERFACE_H

#ifdef __cplusplus
#include <stdint.h>
#include <assert.h>
extern "C" {
#else
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#endif

#define MAX_SEM_PER_VC 4
#define MAX_NUM_VC_ALLOC_ITEMS 48
#define MAX_NUM_VC_QUEUE_ALLOC_ITEMS 48
#if defined(CHIPSET_AIC_100)
#define MAX_NUM_VC_IATU_ALLOC_ITEMS 48
#define MAX_NUM_VC_OATU_ALLOC_ITEMS 48
#endif


#define SHDDR_QCCL_INFO_BASE_ADDR_OFFSET 0xa00000
/*1MB space for QCCL resource info starting from SHDDR_QCCL_INFO_BASE_ADDR_OFFSET*/
#define QCCL_INFO_MAX_SIZE 0x100000

typedef struct jit_heap_ddr_alloc_alp {
  uint64_t addr_base;
  uint64_t addr_size;
  uint32_t buf_id;
  uint8_t padding[4];
} jit_heap_ddr_alloc_alp_t;

typedef struct jit_ddr_buf_alp_info {
  uint64_t offset;
  uint32_t size;
  uint32_t buf_id;
} jit_ddr_buf_alp_info_t;

typedef struct jit_semaphore_info {
  uint32_t sem_handle;
  uint32_t sem_id;
  uint64_t bar2_offset;
} jit_semaphore_info_t;

typedef struct jit_vc_info {
  uint32_t vc_handle;
  uint32_t num_semaphore;
  uint64_t tailptr_bar2_offset;
  jit_semaphore_info_t semaphore_info[MAX_SEM_PER_VC];
} jit_vc_info_t;

typedef struct jit_vc_queue_info {
  uint32_t vcq_handle;
  uint32_t queue_va;
  /*Indicates the index at which incoming DMA req element is copied*/
  uint32_t dma_req_index;
  /*Indicates VC queue capacity (number of DMA req elements)*/
  uint32_t queue_capacity;
  /*Offset in BAR4 for remote queue or L2TCM for local queue*/
  uint64_t offset;
} jit_vc_queue_info_t;

typedef struct jit_iatu_handle_info {
  uint32_t atu_handle;
} jit_iatu_handle_info_t;

typedef struct jit_oatu_handle_info {
  uint64_t aperture_base_addr;
  uint32_t atu_handle;
  uint32_t virt_addr;
  uint32_t aperture_mapsize;
  uint32_t is_enabled;
} jit_oatu_handle_info_t;

typedef struct jit_buffer_info {
  uint64_t start_addr;
  uint64_t offset;
  uint64_t size;
} jit_buffer_info_t;

typedef struct jit_vc_setup_info {
  uint64_t request_q_addr;
  uint64_t response_q_addr;
  uint32_t request_q_size;
  uint32_t response_q_size;
  bool is_local_q;
} jit_vc_setup_info_t;

typedef struct jit_available_resource_info {
  uint8_t max_vc_count;
  uint8_t num_available_vc;
  uint8_t num_available_iatu_count;
  uint8_t num_available_oatu_count;
  uint8_t max_iatu_count;
  uint8_t max_oatu_count;
  /*IATU and OATU indexes are internal to Qshim.Not populated by RI*/
  uint8_t iatu_available_index;
  uint8_t oatu_available_index;
  uint16_t mcid0_phy_id;
  /*Below fields BAR4 and APERTURE available info indicates physical start_addr,
   * offset from the start addr available resource starts and remaining
   * size */
  uint8_t padding[6];
  jit_buffer_info_t bar4_available_info;
  jit_buffer_info_t aperture_available;
  jit_buffer_info_t bar2_available_info;
} jit_available_resource_info_t;

typedef enum { JIT_ATU_TYPE_OATU, JIT_ATU_TYPE_IATU } JitAtuType_t;

typedef struct jit_program_atu {
  uint64_t base_addr;
  uint64_t target_addr;
  uint32_t atu_handle;
  uint32_t size;
  JitAtuType_t atu_type;
  void **va;
  bool enable;
  uint8_t padding[3];
} jit_program_atu_t;

typedef struct jit_init_qccl_infra {
  uint64_t iatu_resource_offset;
  uint64_t oatu_resource_offset;
  uint64_t available_resource_offset;
  uint32_t host_pid;
  uint8_t padding[4];
} jit_init_qccl_infra_t;

typedef struct jit_ddr_malloc_params {
  uint64_t size;
  uint32_t host_pid;
  uint32_t reserve_size;
  jit_heap_ddr_alloc_alp_t *reserve_alloc;
  jit_heap_ddr_alloc_alp_t *ddr_alloc_alps;
  uint8_t *num_alps;
  uint8_t padding[4];
}jit_ddr_malloc_params_t;

typedef enum {
  JIT_ECC_ERR_CNT_ECC_DISABLED = -2, // ECC is disabled
  JIT_ECC_ERR_CNT_FAILURE = -1,      // failed fetching error counts
  JIT_ECC_ERR_CNT_SUCCESS = 0
} JitEccErrCountRetType_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // QSM_NSP_JIT_INTERFACE_H
