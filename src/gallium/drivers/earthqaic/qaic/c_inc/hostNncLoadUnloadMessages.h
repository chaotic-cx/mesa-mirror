/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef AICLINUXRUNTIME_HOSTNNCLOADUNLOADMESSAGES_H
#define AICLINUXRUNTIME_HOSTNNCLOADUNLOADMESSAGES_H

#include "nncErrors.h"
#include <stdint.h>

#define NNC_MAX_STRING_LENGTH 64
#define FIXED_AES_NONCE_LENGTH 12      // 96 bits
#define ENCRYPTED_MEK_LENGTH (16 + 16) // 16 bytes for MEK + 16 bytes for MAC
#define P384_PUB_KEY_LENGTH 97

/*****************************************************************************
 *  All the NNC protocol structures should end on a 8byte boundary and all   *
 *  its structure members must be naturally aligned to itself. It is the     *
 *  responsibility of the maintainer/programmer to make sure that the above  *
 *  criteria are met while updating the NNC protocol structures.            *
 * ***************************************************************************/

/**
 * Requests the loading of the weights and biases that will be used later when
 * activating the network.
 */
typedef struct nnc_cmd_load_constants_request {
  uint32_t type;
  /** Host value used for debugging and referenced by other modules running in
   * QSM (e.g. DM). */
  uint32_t process_id;
  /** Tag is used to correlate to data specified dma_xfer transaction */
  uint32_t tag;
  uint32_t constants_crc;
  uint64_t data_length;
  /** Null terminated ASCII string for the name of the constants. Set the first
   * byte to NULL if not used.*/
  char constants_name[NNC_MAX_STRING_LENGTH];
} nnc_cmd_load_constants_request_t;

/**
 * The load constants request EX request is used to deliver the constants
 * descriptor which is derived from metadata and it describes the sizes of the
 * constants sections and related attributes. There are no companion DMA
 * transfer transactions that go with the EX request.
 */
typedef struct nnc_cmd_load_constants_ex_request {
  uint32_t type;
  /** Host value used for debugging and referenced by other modules running in
   * QSM (e.g. DM). */
  uint32_t process_id;

  /** Null terminated ASCII string for the name of the constants. Set the first
   * byte to NULL if not used.*/
  char constants_name[NNC_MAX_STRING_LENGTH];

  /* AICConstantDescriptor from metadata */
  uint32_t descriptor_len;
  uint32_t padding;
  uint8_t descriptor[];
} nnc_cmd_load_constants_ex_request_t;

/**
 * Load constants to the region set up by nnc_cmd_load_constants_ex_request
 */
typedef struct nnc_cmd_load_constants_payload_request {
  uint32_t type;
  /** Host value used for debugging and referenced by other modules running in
   * QSM (e.g. DM). */
  uint32_t process_id;
  /** Tag is used to correlate to data specified dma_xfer transaction */
  uint32_t tag;
  uint32_t constants_id;
  uint64_t data_length;
  uint64_t offset;
} nnc_cmd_load_constants_payload_request_t;

/**
 * Response to nnc_cmd_load_constants_request and
 * nnc_cmd_load_constants_payload_request.
 */
typedef struct nnc_cmd_load_constants_response {
  uint32_t type;
  /**
   * Unique ID of the successfully loaded constants. A value of zero is used if
   * there was an error. See status_code for information on the nature of the
   * error.
   */
  uint32_t constants_id;
  /** Status of the nnc_cmd_load_request. */
  nnc_status_code_t status_code;
  uint32_t padding;
} nnc_cmd_load_constants_response_t;

/**
 * Encrypted MEK is decrypted using the provided MEK nonce and MEK public key
 * Procedure:
 * - Shared Key is computed from device private key and provided MEK Public Key
 * - Shared Key is extracted with SHA256 which gives us the HKDF (Salt is empty
 * string)
 * - HKDF is expanded to give us the AES Key (Info is empty string)
 * - This derived AES Key and provided MEK Nonce is used decrypt the encrypted
 * MEK
 *
 * This MEK is stored in device to decrypt network.elf and constants.bin
 * Procedure:
 * - MEK (decrypted from above procedure) is used as the AES Key with segment
 * specific nonce
 * - encrypted network.elf with nw_elf_nonce
 * - encrypted static.constants.bin with static_constants_nonce
 * - encrypted dynamic.constants.bin with dynamic_constants_nonce
 *
 * LRT sends constants decryption request after loading encrypted constants.
 * Note: MEK cannot change until device is rebooted. Once MEK is set, FW
 * ignores MEK in any subsequent NNC command.
 */
typedef struct nnc_cmd_constants_decryption_request {
  uint32_t type;
  /* Populate this id from NNC response to constantsdesc.bin load request */
  uint32_t constants_id;
  uint8_t static_constants_nonce[FIXED_AES_NONCE_LENGTH];
  uint8_t dynamic_constants_nonce[FIXED_AES_NONCE_LENGTH];
  uint8_t encrypted_mek_nonce[FIXED_AES_NONCE_LENGTH];
  uint8_t encrypted_mek[ENCRYPTED_MEK_LENGTH];
  uint8_t encrypted_mek_pub[P384_PUB_KEY_LENGTH];
  uint8_t padding[3];
} nnc_cmd_constants_decryption_request_t;

typedef struct nnc_cmd_constants_decryption_response {
  uint32_t type;
  nnc_status_code_t status_code;
} nnc_cmd_constants_decryption_response_t;

/**
 * Requests a load of the network ELF file and metadata at the specified host
 * address.
 */
typedef struct nnc_cmd_load_elf_request {
  uint32_t type;
  /** Tag is used to associate the request with a DMA_XFER transation with the
   * same tag */
  uint32_t tag;
  /** ID of the process used by the host OS.*/
  uint32_t process_id;
  uint32_t metadata_crc;
  /** Null terminated ASCII string for the name of the network. Set the first
   * byte to NULL if not used.*/
  char network_name[NNC_MAX_STRING_LENGTH];
  /** Size of network elf */
  uint64_t data_length;
  /**
   * This CRC is the CRC-32 of the section metadata_fb in network.ef
   */
  uint32_t metadata_flatbuffer_crc;
  uint8_t padding[4];
} nnc_cmd_load_elf_request_t;

/**
 * Response returned to the host after an nnc_cmd_load_elf_request.
 */
typedef struct nnc_cmd_load_elf_response {
  uint32_t type;
  /**
   * Unique ID of the successfully loaded constants. A value of zero is used if
   * there was an error. See status_code for information on the nature of the
   * error.
   */
  uint32_t elf_id;
  /** Status of the nnc_cmd_load_request. */
  nnc_status_code_t status_code;
  uint32_t padding;
} nnc_cmd_load_elf_response_t;

/**
 * Decrypt network_elf.bin using nonce value passed in this request and
 * previously passed MEK passed in nnc_cmd_constants_decryption_request_t
 */
typedef struct nnc_cmd_load_elf_decryption_request {
  nnc_cmd_load_elf_request_t load_req;
  uint8_t nw_elf_nonce[FIXED_AES_NONCE_LENGTH];
  uint8_t encrypted_mek_nonce[FIXED_AES_NONCE_LENGTH];
  uint8_t encrypted_mek[ENCRYPTED_MEK_LENGTH];
  uint8_t encrypted_mek_pub[P384_PUB_KEY_LENGTH];
  uint8_t padding[7];
} nnc_cmd_load_elf_decryption_request_t;

typedef struct nnc_cmd_load_elf_decryption_response {
  nnc_cmd_load_elf_response_t load_resp;
} nnc_cmd_load_elf_decryption_response_t;

/**
 * Unload previously loaded network weights and biases.
 */
typedef struct nnc_cmd_unload_constants_request {
  uint32_t type;
  /** Unique ID of previously loaded network weights and biases. */
  uint32_t constants_id;
} nnc_cmd_unload_constants_request_t;

/**
 * Unload previously loaded network weights and biases.
 */
typedef struct nnc_cmd_unload_constants_response {
  uint32_t type;
  /** Status of the nnc_cmd_unload_constants_request. */
  nnc_status_code_t status_code;
} nnc_cmd_unload_constants_response_t;

/**
 * Requests the unload of a previously loaded network.
 */
typedef struct nnc_cmd_unload_elf_request {
  uint32_t type;
  /** Unique ID of the previously loaded elf that is to be unloaded. */
  uint32_t elf_id;
} nnc_cmd_unload_elf_request_t;

/**
 * Response returned to the host after an nnc_cmd_unload_elf_request.
 */
typedef struct nnc_cmd_unload_elf_response {
  uint32_t type;
  /** Status of the nnc_cmd_unload_elf_request. */
  nnc_status_code_t status_code;
} nnc_cmd_unload_elf_response_t;

#endif // AICLINUXRUNTIME_HOSTNNCLOADUNLOADMESSAGES_H
