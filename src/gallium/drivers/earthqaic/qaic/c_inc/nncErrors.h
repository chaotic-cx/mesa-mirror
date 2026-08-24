/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef AICLINUXRUNTIME_NNCERRORS_H
#define AICLINUXRUNTIME_NNCERRORS_H

#include <limits.h>
#include <stdint.h>

#define UNKNOWN_NNC_ERROR_STRING "Unknown Error"

typedef enum nnc_status_code {
  NNC_STATUS_UNDEFINED = INT_MIN, //!< Uninitialized response code value.
  NNC_STATUS_UNKNOWN_ERROR = INT_MIN + 1, //!< Error of unknown type.
  NNC_STATUS_LAST = -101, //!< Current last valid error enum. If new gets added,
                         //!< use this number and move this to next available
  NNC_STATUS_DEVICE_BUSY = -101, //!< Device is busy
  NNC_STATUS_DECRYPTION_ERROR =
      -100, //!< Failed to perform decryption as part of Model IP
  NNC_STATUS_NW_THREAD_AC_CLN_ACK_FAILED_ERROR =
      -99, //!< Wait from User PD failed in Network thread activation
  NNC_STATUS_NW_THREAD_AC_MQ_SEND_FAILED_ERROR =
      -98, //!< Failed to send msg to PD in Network thread activation
  NNC_STATUS_DMA_CONFIG_ERROR = -97, //!<DMA config failed>
  NNC_STATUS_INVALID_REQUEST =
      -96, //!< Activate / deactivate request on already active / idle network
  NNC_STATUS_RESOURCE_ERROR =
      -95, //!< Allocated resources required for this request are exhausted.
  NNC_STATUS_PHFF_P2V_NSP_FAILED = -94,
  NNC_STATUS_GLIBC_GET_NODE_FAILED = -93,
  NNC_STATUS_PHFF_PD_MAP_FAILED = -92,
  NNC_STATUS_HOST_REFCNT_NOT_ZERO_ERROR =
      -91, //!< NN records host refcnt non-zero
  NNC_STATUS_NN_RECORD_NOT_FOUND_ERROR = -90, //!< NN Record not found with ID
  NNC_STATUS_PA_L2_INIT_ERROR = -89,          //!< L2Init mapping failed
  NNC_STATUS_PA_GSM_CONF_MMAP_ERROR = -88,    //!< GSM mapping failed
  NNC_STATUS_ERROR_NO_NW_TO_DEACTIVATE = -87, //!< Nw already deactivated
  NNC_STATUS_PD_MMC_MAP_ERROR = -86, //!< MMC mapping failed in deactivate
  NNC_STATUS_DE_CLN_ACK_FAILED_ERROR =
      -85, //!< Wait from User PD failed in Deactivate
  NNC_STATUS_DE_MQ_SEND_FAILED_ERROR =
      -84, //!< Failed to send msg to PD in Deactivate
  NNC_STATUS_M_WAIT_ON_S_SYNC_CMD_TIMEOUT_ERROR =
      -83, //!< Master NSP Timeout while waiting for CMD/State
  NNC_STATUS_M_WAIT_ON_S_SYNC_CMD_FAILED_ERROR =
      -82, //!< Master NSP Timeout while waiting for CMD/State
  NNC_STATUS_PA_MCAST_RANGE_CHANGE_ERROR =
      -81, //!< Failed to change mcast range
  NNC_STATUS_PA_SET_MCAST_ERROR =
      -80, //!< Failed to set Mcast range in Pre-activate
  NNC_STATUS_BW_MON_MMAP_ERROR = -79, //!< Failed to map bw monitor region
  NNC_STATUS_PA_MN_RSP_W_ERROR = -78,
  NNC_STATUS_NW_MAP_ERROR = -77,       //!< Failed to create nnc mapping
  NNC_STATUS_MMCADDR_MAP_ERROR = -76,  //!< Failed to create mmcaddr mapping
  NNC_STATUS_MQ_SEND_ERROR = -75,      //!< Failed to send MQ
  NNC_STATUS_PD_SIG_WAIT_ERROR = -74,  //!< Wait from PD failed in activate
  NNC_STATUS_VER_MISMATCH_ERROR = -73, //!< NNC exec context version mismatch
  NNC_STATUS_MMAP_ERROR = -72,         //!< Failed to map region
  NNC_STATUS_AC_SYNC_ERROR = -71,
  NNC_STATUS_MQ_BAR4_ERROR = -70, //!< BAR4 size is less than 2GB
  NNC_STATUS_MQ_NOT_ENABLED =
      -69, //!< Multicard solution Feature is not enabled
  NNC_STATUS_MQ_UNSUPPORTED =
      -68, //!< Multicard solution Feature is unsupported on this device
  NNC_STATUS_AC_RECORD_NOT_FOUND = -67, //!< ac_record is null
  NNC_STATUS_DPS_NOT_ENABLED = -66,     //!< DPS not enabled on device
  NNC_STATUS_NSP_MCID_COUNT_EXCEED_MAX_LIMIT =
      -65, //!< MCID count exceeds more than max supported limit
  NNC_STATUS_PROTO_NOT_SUPPORTED = -64, //!< Invalid protocol type
  NNC_STATUS_GET_ACT_RECORD_ERROR =
      -63, //!< Failed to get activation record from lut
  NNC_STATUS_RETRIEVAL_NSP_4M_METADATA_ERROR =
      -62,                                   //!< Parse NSP from metadata failed
  NNC_STATUS_RETRIEVAL_METADATA_ERROR = -61, //!< Failed to parse metadata
  NNC_SWITCH_CMD_VCID_INVALID = -60,         //!< Invalid vcid
  NNC_STATUS_SWITCH_AC_STATE_ERROR = -59,    //!< Switch cmd sanity check failed
  NNC_STATUS_SWITCH_CMD_SEQ_MISMATCH_ERROR =
      -58, //!< Switch cmd sequence failed
  NNC_STATUS_SWITCH_PARAM_ERROR =
      -57, //!< Switch cmd param error(Invalid number of switch cmd)
  NNC_STATUS_NUMNSP_RETRIEVAL_ERROR =
      -56, //!< Failed to retrieve NSP count from metadata
  NNC_STATUS_VCID_RETRIEVAL_ERROR =
      -55, //!< Failed to retrieve vcid from activation record
  NNC_STATUS_META_RETRIEVAL_ERROR = -54, //!< Failed to parse metadata
  NNC_STATUS_RETRIEVAL_PNSP_ID_ERROR =
      -53, //!< Failed to retrive physical nsp id from ac record
  NNC_STATUS_VC_SETUP_ERROR = -52, //!< Failed to setup dma bridge
  NNC_STATUS_GSM_CONF_ERROR = -51, //!< Failed to configure GSM
  NNC_STATUS_DYN_CONST_RESOURCE_NOT_FOUND =
      -50, //!< Failed to parse dynamic const from metadata
  NNC_STATUS_L2TCM_INIT_STATE_INVALID_ERROR =
      -49, //!< Failed to get L2TCM Meminfo data from metadata
  NNC_STATUS_RETRIEVAL_L2TCM_INIT_STATE_ERROR =
      -48, //!< Failed to get L2TCM Meminfo from metadata
  NNC_STATUS_MCID_CONFIG_ERROR = -47, //!< Failed to configure mcid
  NNC_STATUS_RETRIEVAL_PRO_IMG_ERROR =
      -46, //!< Failed to parse Program ELF info from metadata
  NNC_STATUS_LUT_UPDATE_FAILED_ERROR = -45, //!< Failed to update activation lut
  NNC_STATUS_XPU_ERROR = -44,               //!< XPU program error
  /* FuSa Auto specific errors */
  NNC_STATUS_SAFETY_SWIV_ELF_ERROR =
      -43, //!< Safety SWIV check error on network elf image verification
  NNC_STATUS_SAFETY_SWIV_CONST_ERROR =
      -42, //!< Safety SWIV check error on constant image verification
  NNC_STATUS_SAFETY_SWIV_METADATA_ERROR =
      -41, //!< Safety SWIV check error on metadata image verification
  NNC_STATUS_SAFETY_BCS_MHI_ERROR = -40, //!< Safety BCS check error on MHI path
  NNC_STATUS_SAFETY_BCS_GLINK_ERROR =
      -39, //!< Safety BCS check error on Glink path
  NNC_STATUS_SAFETY_BCS_METADATA_ERROR =
      -38, //!< Safety BCS check error on Metadata corruption
  NNC_STATUS_SAFETY_BCS_AC_RECORD_ERROR =
      -37, //!< Safety BCS check error on Activation record corruption
  NNC_STATUS_SS_UNDER_SSR_ERROR =
      -36, //!< Operation failed due to NSP under SSR
  NNC_STATUS_VC_CONFIGURE_ERROR = -35, //!< VC configuration/enable error
  NNC_STATUS_QUEUE_TOO_SMALL = -34,    //!< P2P VC queue size too small
  NNC_STATUS_MQ_PORT_ALLOC_ERROR =
      -33, //!< Port not allocated in MQ activation record
  NNC_STATUS_MQ_PORT_INFO_GENERATE_ERROR =
      -32, //!< Failed to generate port info for NSP error
  NNC_STATUS_MQ_P2P_DMA_REQ_GENERATE_ERROR =
      -31, //!< Failed to generate DMA req for P2P VC
  NNC_STATUS_MQ_NO_P2P_DMA_REQ_INFO_ERROR =
      -30, //!< No DMA request element from preactivate info
  NNC_STATUS_MQ_P2P_DMA_REQ_INFO_ERROR =
      -29, //!< Failed to DMA request element from preactivate info
  NNC_STATUS_MQ_TOPOLOGY_INFO_ERROR =
      -28, //!< Failed to parse topology table error
  NNC_STATUS_MQ_REMOTE_PORT_NOT_FOUND_ERROR =
      -27, //!< Failed to find port on remote device/function
  NNC_STATUS_MQ_ACTIVATION_INFO_PARSER_ERROR =
      -26, //!< Failed to parse or fetch activation info error
  NNC_STATUS_MQ_ACTIVATION_INFO_DEVID_ERROR =
      -25, //!< Failed to devid in activation info
  NNC_STATUS_MQ_PREACTIVATE_INFO_ERROR =
      -24, //!< Failed to parse preactivation info blob error
  NNC_STATUS_MQ_P2P_ERROR = -23,           //!< Failed to access P2P device
  NNC_STATUS_IATU_ALLOC_ERROR = -22,       //!< Failed to allocate IATU resource
  NNC_STATUS_OATU_ALLOC_ERROR = -21,       //!< Failed to allocate OATU resource
  NNC_STATUS_DATABASE_IDS_EXHAUSTED = -20, //!< Database ID's got exhausted
  NNC_STATUS_DATABASE_ID_NOT_FOUND = -19,  //!< Database ID not found in Queue
  NNC_STATUS_RESERVATION_IDS_EXHAUSTED =
      -18,                             //!< Reservation ID's got exhausted
  NNC_STATUS_CMD_LENGTH_ERROR = -17,   //!< Unexpected command length
  NNC_STATUS_NSP_RESPONSE_ERROR = -16, //!< NSP response timeout /error
  NNC_STATUS_SEM_ALLOC_ERROR =
      -15, //!< Failed to allocate GSM semaphore resource
  NNC_STATUS_VC_ALLOC_ERROR = -14,  //!< Failed to allocate virtual channel
  NNC_STATUS_NSP_ALLOC_ERROR = -13, //!< Failed to allocate NSP resources
  NNC_STATUS_MCID_ALLOC_ERROR =
      -12, //!< Failed to allocate multicast ID resources
  NNC_STATUS_RESOURCE_NOT_FOUND = -11, //!< Requested Resource Not found>
  NNC_STATUS_BUFFER_TOO_SMALL = -10,   //!< Buffer too small
  NNC_STATUS_OPERATION_NOT_PERMITTED =
      -9,                          //!< NNC permission denied for invalid handle
                                   //! or attempt to release a resource in use
  NNC_STATUS_NOT_INITIALIZED = -8, //!< NNC not yet initialized.
  NNC_STATUS_TIMED_OUT_WAITING =
      -7, //!< Timed out waiting for a state change to occur.
  NNC_STATUS_PARAMETER_ERROR = -6, //!< Error caused by an invalid parameter
                                   //! passed to an internal function.
  NNC_STATUS_METADATA_ERROR =
      -5, //!< Version mismatch / error processing the ELF metadata
  NNC_STATUS_MEM_ALLOC_ERROR = -4,   //!< Memory resource exhausted.
  NNC_STATUS_UNDEFINED_REQUEST = -3, //!< The request specified is not defined.
  NNC_STATUS_DMA_ERROR =
      -2,                 //!< DMA transfer failed.
  NNC_STATUS_ERROR = -1,  //!< General error occurred.
  NNC_STATUS_SUCCESS = 0, //!< The specified request was successful.
  NNC_STATUS_BEGIN =
      NNC_STATUS_SUCCESS, //!< Begining of NNC error enums. This marks start of
                          //!< enum values but this itself is an invalid value.
                          //!< If new gets added use this number and assign next
                          //!< available.
  NNC_STATUS_NONE = 1,
  NNC_STATUS_INVALID_MAX = INT_MAX,
} nnc_status_code_t;

static inline const char *nncErrorStr(int status) {
  const char *retVal;

  switch (status) {
  case NNC_STATUS_HOST_REFCNT_NOT_ZERO_ERROR:
    retVal = "NN records host refcnt non-zero";
    break;
  case NNC_STATUS_NN_RECORD_NOT_FOUND_ERROR:
    retVal = "NN Record not found with ID";
    break;
  case NNC_STATUS_PA_L2_INIT_ERROR:
    retVal = "L2Init mapping failed";
    break;
  case NNC_STATUS_PA_GSM_CONF_MMAP_ERROR:
    retVal = "GSM mapping failed";
    break;
  case NNC_STATUS_ERROR_NO_NW_TO_DEACTIVATE:
    retVal = "Nw already deactivated";
    break;
  case NNC_STATUS_PD_MMC_MAP_ERROR:
    retVal = "MMC mapping failed in deactivate";
    break;
  case NNC_STATUS_DE_CLN_ACK_FAILED_ERROR:
    retVal = "Wait from User PD failed in Deactivate";
    break;
  case NNC_STATUS_DE_MQ_SEND_FAILED_ERROR:
    retVal = "Failed to send msg to PD in Deactivate";
    break;
  case NNC_STATUS_M_WAIT_ON_S_SYNC_CMD_TIMEOUT_ERROR:
    retVal = "Master NSP Timeout while waiting for CMD/State";
    break;
  case NNC_STATUS_M_WAIT_ON_S_SYNC_CMD_FAILED_ERROR:
    retVal = "Master NSP Error while waiting for CMD/State";
    break;
  case NNC_STATUS_PA_MCAST_RANGE_CHANGE_ERROR:
    retVal = "Failed to change mcast range";
    break;
  case NNC_STATUS_PA_SET_MCAST_ERROR:
    retVal = "Failed to set Mcast range in Pre-activate";
    break;
  case NNC_STATUS_BW_MON_MMAP_ERROR:
    retVal = "Failed to map bw monitor region";
    break;
  case NNC_STATUS_PA_MN_RSP_W_ERROR:
    retVal = "Failed in Pre-activate RSP";
    break;
  case NNC_STATUS_NW_MAP_ERROR:
    retVal = "Failed to create nnc mapping";
    break;
  case NNC_STATUS_MMCADDR_MAP_ERROR:
    retVal = "Failed to create mmcaddr mapping";
    break;
  case NNC_STATUS_MQ_SEND_ERROR:
    retVal = "Failed to send MQ";
    break;
  case NNC_STATUS_PD_SIG_WAIT_ERROR:
    retVal = "Wait from PD failed in activate";
    break;
  case NNC_STATUS_VER_MISMATCH_ERROR:
    retVal = "NNC exec context version mismatch";
    break;
  case NNC_STATUS_MMAP_ERROR:
    retVal = "Failed to map region";
    break;
  case NNC_STATUS_AC_SYNC_ERROR:
    retVal = "Sync error";
    break;
  case NNC_STATUS_MQ_BAR4_ERROR:
    retVal = "BAR4 size is less than 2GB";
    break;
  case NNC_STATUS_MQ_NOT_ENABLED:
    retVal = "Multicard solution is not enabled";
    break;
  case NNC_STATUS_MQ_UNSUPPORTED:
    retVal = "Multicard solution is unsupported on this device";
    break;
  case NNC_STATUS_DPS_NOT_ENABLED:
    retVal = "DPS not enabled on device";
    break;
  case NNC_STATUS_XPU_ERROR:
    retVal = "XPU program error";
    break;
  case NNC_STATUS_NSP_MCID_COUNT_EXCEED_MAX_LIMIT:
    retVal = "MCID count exceeds more than max supported limit";
    break;
  case NNC_STATUS_PROTO_NOT_SUPPORTED:
    retVal = "Invalid protocol type";
    break;
  case NNC_STATUS_GET_ACT_RECORD_ERROR:
    retVal = "Failed to get activation record from lut";
    break;
  case NNC_STATUS_RETRIEVAL_NSP_4M_METADATA_ERROR:
    retVal = "Parse NSP from metadata failed";
    break;
  case NNC_STATUS_RETRIEVAL_METADATA_ERROR:
    retVal = "Failed to parse metadata";
    break;
  case NNC_SWITCH_CMD_VCID_INVALID:
    retVal = "Invalid vcid";
    break;
  case NNC_STATUS_SWITCH_AC_STATE_ERROR:
    retVal = "Switch cmd sanity check failed";
    break;
  case NNC_STATUS_SWITCH_CMD_SEQ_MISMATCH_ERROR:
    retVal = "Switch cmd sequence failed";
    break;
  case NNC_STATUS_SWITCH_PARAM_ERROR:
    retVal = "Switch cmd param error(Invalid number of switch cmd)";
    break;
  case NNC_STATUS_NUMNSP_RETRIEVAL_ERROR:
    retVal = "Failed to retrieve NSP count from metadata";
    break;
  case NNC_STATUS_VCID_RETRIEVAL_ERROR:
    retVal = "Failed to retrieve vcid from activation record";
    break;
  case NNC_STATUS_META_RETRIEVAL_ERROR:
    retVal = "Failed to parse metadata";
    break;
  case NNC_STATUS_RETRIEVAL_PNSP_ID_ERROR:
    retVal = "Failed to retrive physical nsp id from ac record";
    break;
  case NNC_STATUS_VC_SETUP_ERROR:
    retVal = "Failed to setup dma bridge";
    break;
  case NNC_STATUS_GSM_CONF_ERROR:
    retVal = "Failed to configure GSM";
    break;
  case NNC_STATUS_DYN_CONST_RESOURCE_NOT_FOUND:
    retVal = "Failed to parse dynamic const from metadata";
    break;
  case NNC_STATUS_L2TCM_INIT_STATE_INVALID_ERROR:
    retVal = "Failed to get L2TCM Meminfo data from metadata";
    break;
  case NNC_STATUS_RETRIEVAL_L2TCM_INIT_STATE_ERROR:
    retVal = "Failed to get L2TCM Meminfo from metadata";
    break;
  case NNC_STATUS_MCID_CONFIG_ERROR:
    retVal = "Failed to configure mcid";
    break;
  case NNC_STATUS_RETRIEVAL_PRO_IMG_ERROR:
    retVal = "Failed to parse Program ELF info from metadata";
    break;
  case NNC_STATUS_LUT_UPDATE_FAILED_ERROR:
    retVal = "Failed to update activation lut";
    break;
  case NNC_STATUS_SAFETY_SWIV_ELF_ERROR:
    retVal = "Safety SWIV check error on network elf image verification";
    break;
  case NNC_STATUS_SAFETY_SWIV_CONST_ERROR:
    retVal = "Safety SWIV check error on constant image verification";
    break;
  case NNC_STATUS_SAFETY_SWIV_METADATA_ERROR:
    retVal = "Safety SWIV check error on metadata image verification";
    break;
  case NNC_STATUS_SAFETY_BCS_MHI_ERROR:
    retVal = "Safety BCS check error on MHI path";
    break;
  case NNC_STATUS_SAFETY_BCS_GLINK_ERROR:
    retVal = "Safety BCS check error on Glink path";
    break;
  case NNC_STATUS_SAFETY_BCS_METADATA_ERROR:
    retVal = "Safety BCS check error on Metadata corruption";
    break;
  case NNC_STATUS_SAFETY_BCS_AC_RECORD_ERROR:
    retVal = "Safety BCS check error on Activation record corruption";
    break;
  case NNC_STATUS_SS_UNDER_SSR_ERROR:
    retVal = "Operation failed due to NSP under SSR";
    break;
  case NNC_STATUS_VC_CONFIGURE_ERROR:
    retVal = "VC configuration/enable error";
    break;
  case NNC_STATUS_QUEUE_TOO_SMALL:
    retVal = "P2P VC queue size too small";
    break;
  case NNC_STATUS_MQ_PORT_ALLOC_ERROR:
    retVal = "Port not allocated in MQ activation record";
    break;
  case NNC_STATUS_MQ_PORT_INFO_GENERATE_ERROR:
    retVal = "Failed to generate port info for NSP error";
    break;
  case NNC_STATUS_MQ_P2P_DMA_REQ_GENERATE_ERROR:
    retVal = "Failed to generate DMA req for P2P VC";
    break;
  case NNC_STATUS_MQ_NO_P2P_DMA_REQ_INFO_ERROR:
    retVal = "No DMA request element from preactivate info";
    break;
  case NNC_STATUS_MQ_P2P_DMA_REQ_INFO_ERROR:
    retVal = "Failed to DMA request element from preactivate info";
    break;
  case NNC_STATUS_MQ_TOPOLOGY_INFO_ERROR:
    retVal = "Failed to parse topology table error";
    break;
  case NNC_STATUS_MQ_REMOTE_PORT_NOT_FOUND_ERROR:
    retVal = "Failed to find port on remote device/function";
    break;
  case NNC_STATUS_MQ_ACTIVATION_INFO_PARSER_ERROR:
    retVal = "Failed to parse or fetch activation info error";
    break;
  case NNC_STATUS_MQ_ACTIVATION_INFO_DEVID_ERROR:
    retVal = "Failed to devid in activation info";
    break;
  case NNC_STATUS_MQ_PREACTIVATE_INFO_ERROR:
    retVal = "Failed to parse preactivation info blob error";
    break;
  case NNC_STATUS_MQ_P2P_ERROR:
    retVal = "Failed to access P2P device";
    break;
  case NNC_STATUS_IATU_ALLOC_ERROR:
    retVal = "Failed to allocate IATU resource";
    break;
  case NNC_STATUS_OATU_ALLOC_ERROR:
    retVal = "Failed to allocate OATU resource";
    break;
  case NNC_STATUS_DATABASE_IDS_EXHAUSTED:
    retVal = "Database ID's got exhausted";
    break;
  case NNC_STATUS_DATABASE_ID_NOT_FOUND:
    retVal = "Database ID not found in Queue";
    break;
  case NNC_STATUS_RESERVATION_IDS_EXHAUSTED:
    retVal = "Reservation ID's got exhausted";
    break;
  case NNC_STATUS_CMD_LENGTH_ERROR:
    retVal = "Unexpected command length";
    break;
  case NNC_STATUS_NSP_RESPONSE_ERROR:
    retVal = "NSP response timeout /error";
    break;
  case NNC_STATUS_SEM_ALLOC_ERROR:
    retVal = "Failed to allocate GSM semaphore resource";
    break;
  case NNC_STATUS_VC_ALLOC_ERROR:
    retVal = "Failed to allocate virtual channel";
    break;
  case NNC_STATUS_NSP_ALLOC_ERROR:
    retVal = "Failed to allocate NSP resources";
    break;
  case NNC_STATUS_MCID_ALLOC_ERROR:
    retVal = "Failed to allocate multicast ID resources";
    break;
  case NNC_STATUS_RESOURCE_NOT_FOUND:
    retVal = "Requested Resource Not found";
    break;
  case NNC_STATUS_BUFFER_TOO_SMALL:
    retVal = "Buffer too small";
    break;
  case NNC_STATUS_OPERATION_NOT_PERMITTED:
    retVal = "NNC permission denied for invalid handle or attempt to release a "
             "resource in use";
    break;
  case NNC_STATUS_NOT_INITIALIZED:
    retVal = "NNC not yet initialized";
    break;
  case NNC_STATUS_TIMED_OUT_WAITING:
    retVal = "Timed out waiting for a state change to occur";
    break;
  case NNC_STATUS_PARAMETER_ERROR:
    retVal =
        "Error caused by an invalid parameter passed to an internal function";
    break;
  case NNC_STATUS_METADATA_ERROR:
    retVal = "Version mismatch / error processing the ELF metadata";
    break;
  case NNC_STATUS_MEM_ALLOC_ERROR:
    retVal = "Memory resource exhausted";
    break;
  case NNC_STATUS_UNDEFINED_REQUEST:
    retVal = "The request specified is not defined";
    break;
  case NNC_STATUS_DMA_CONFIG_ERROR:
    retVal = "DMA configuration error";
    break;
  case NNC_STATUS_DMA_ERROR:
    retVal = "DMA transfer error";
    break;
  case NNC_STATUS_AC_RECORD_NOT_FOUND:
    retVal = "AC record not found in FW";
    break;
  case NNC_STATUS_DECRYPTION_ERROR:
    retVal = "Failed to perform Model IP decryption";
    break;
  case NNC_STATUS_DEVICE_BUSY:
    retVal = "Device is busy";
    break;
  case NNC_STATUS_ERROR:
    retVal = "General error occurred";
    break;
  case NNC_STATUS_SUCCESS:
    retVal = "Successful";
    break;
  default:
    retVal = UNKNOWN_NNC_ERROR_STRING;
    break;
  }
  return retVal;
}

#endif // AICLINUXRUNTIME_NNCERRORS_H
