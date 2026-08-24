// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "os.h"

#include "../AICDefsInternal.h"
#include "libdev_assert.h"
#include "os-inlines.h"

using namespace aic;

void os_udma_report_error(uint32_t dm0) {
  assert(isUserDMAError(dm0));
  DBG_PRINT_INFO("User DMA error!");

  static const char *err_msg[] = {
      /*0*/ " DM command error",
      /*1*/ " Descriptor is not properly aligned for its type",
      /*2*/ " Unrecognized descriptor type",
      /*3*/ " DMA does not support the address",
      /*4*/ " Unsupported bypass mode",
      /*5*/ " Unsupported compression mode format",
      /*6*/ " Descriptor region of interest error",
      /*7*/ " Bus error on descriptor read/write",
      /*8*/ " Bus error on L2 data read",
      /*9*/ " Bus error on L2 data write",
      /*10*/ " Bus error on compression metadata read"};

  // DM4/DM5 are syndrome registers used to capture DMA errors.
  unsigned int dm4, dm5;
  auto udmaReadFunc = libdev_getcontext()->udmaReadFuncPtr;
  udmaReadFunc(4, &dm4);
  udmaReadFunc(5, &dm5);

  // DM4[7:4] - thread ID
  int threadId = dm4 >> 4 & 0xf;
  // DM4[15:8] - syndrome code
  unsigned sc = dm4 >> 8 & 0xff;
  // DM5[31:0] - syndrome address
  uint8_t *saddr = (uint8_t *)dm5;

  DBG_PRINT_INFO(" thread id = %d", threadId);
  DBG_PRINT_INFO(" addr = %p", saddr);
  if (sc <= 10)
    DBG_PRINT_INFO("%s", err_msg[sc]);
  else
    DBG_PRINT_INFO(" Unknown error: syndrome code %d", sc);

  DBG_PRINT_INFO("DM0: %" PRIu32, dm0);

  const DMADescriptor *desc = (const DMADescriptor *)(dm0 & 0xfffffff0);
  debugPrintUdmaDesc(desc);

  assert(0 && "User DMA error!");
  ERR_FATAL(libdev_getcontext()->errFuncPtr, "Halting due to UDMA error", 0, 0,
            0);
}

extern "C" {

enum {
  GSM_SEMOP_NOP = 0,
  GSM_SEMOP_INIT = 1,
  GSM_SEMOP_INC = 2,
  GSM_SEMOP_DEC = 3,
  GSM_SEMOP_WAITEQ = 4,
  GSM_SEMOP_WAITGE = 5,
  GSM_SEMOP_P = 6,
};

static void gsm_semaphore_command(hostsem_t s, uint8_t op, uint8_t semNum,
                                  uint16_t semVal) {
  assert(s && "hostsem==nullptr");
  uint32_t *sa = (uint32_t *)s;
  uint32_t command = (semVal & 0b111111111111) | ((semNum & 0b11111) << 16) |
                     ((op & 0b111) << 24);
  hexagon_atomic_store_nolock4b(sa, command);
}

void os_hostsem_init(hostsem_t s, uint32_t semNum, uint32_t val) {
  assert(s && "hostsem==nullptr");
  gsm_semaphore_command(s, GSM_SEMOP_INIT, semNum, val);
}

void os_hostsem_inc(hostsem_t s, uint32_t semNum) {
  assert(s && "hostsem==nullptr");
  gsm_semaphore_command(s, GSM_SEMOP_INC, semNum, 0);
}

void os_hostsem_dec(hostsem_t s, uint32_t semNum) {
  assert(s && "hostsem==nullptr");
  gsm_semaphore_command(s, GSM_SEMOP_DEC, semNum, 0);
}

}; // extern "C"
