/* This file is generated by venus-protocol.  See vn_protocol_driver.h. */

/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VN_PROTOCOL_DRIVER_QUERY_POOL_H
#define VN_PROTOCOL_DRIVER_QUERY_POOL_H

#include "vn_ring.h"
#include "vn_protocol_driver_structs.h"

/* struct VkQueryPoolCreateInfo chain */

static inline size_t
vn_sizeof_VkQueryPoolCreateInfo_pnext(const void *val)
{
    /* no known/supported struct */
    return vn_sizeof_simple_pointer(NULL);
}

static inline size_t
vn_sizeof_VkQueryPoolCreateInfo_self(const VkQueryPoolCreateInfo *val)
{
    size_t size = 0;
    /* skip val->{sType,pNext} */
    size += vn_sizeof_VkFlags(&val->flags);
    size += vn_sizeof_VkQueryType(&val->queryType);
    size += vn_sizeof_uint32_t(&val->queryCount);
    size += vn_sizeof_VkFlags(&val->pipelineStatistics);
    return size;
}

static inline size_t
vn_sizeof_VkQueryPoolCreateInfo(const VkQueryPoolCreateInfo *val)
{
    size_t size = 0;

    size += vn_sizeof_VkStructureType(&val->sType);
    size += vn_sizeof_VkQueryPoolCreateInfo_pnext(val->pNext);
    size += vn_sizeof_VkQueryPoolCreateInfo_self(val);

    return size;
}

static inline void
vn_encode_VkQueryPoolCreateInfo_pnext(struct vn_cs_encoder *enc, const void *val)
{
    /* no known/supported struct */
    vn_encode_simple_pointer(enc, NULL);
}

static inline void
vn_encode_VkQueryPoolCreateInfo_self(struct vn_cs_encoder *enc, const VkQueryPoolCreateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_encode_VkFlags(enc, &val->flags);
    vn_encode_VkQueryType(enc, &val->queryType);
    vn_encode_uint32_t(enc, &val->queryCount);
    vn_encode_VkFlags(enc, &val->pipelineStatistics);
}

static inline void
vn_encode_VkQueryPoolCreateInfo(struct vn_cs_encoder *enc, const VkQueryPoolCreateInfo *val)
{
    assert(val->sType == VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
    vn_encode_VkStructureType(enc, &(VkStructureType){ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO });
    vn_encode_VkQueryPoolCreateInfo_pnext(enc, val->pNext);
    vn_encode_VkQueryPoolCreateInfo_self(enc, val);
}

static inline size_t vn_sizeof_vkCreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkCreateQueryPool_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_simple_pointer(pCreateInfo);
    if (pCreateInfo)
        cmd_size += vn_sizeof_VkQueryPoolCreateInfo(pCreateInfo);
    cmd_size += vn_sizeof_simple_pointer(pAllocator);
    if (pAllocator)
        assert(false);
    cmd_size += vn_sizeof_simple_pointer(pQueryPool);
    if (pQueryPool)
        cmd_size += vn_sizeof_VkQueryPool(pQueryPool);

    return cmd_size;
}

static inline void vn_encode_vkCreateQueryPool(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkCreateQueryPool_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    if (vn_encode_simple_pointer(enc, pCreateInfo))
        vn_encode_VkQueryPoolCreateInfo(enc, pCreateInfo);
    if (vn_encode_simple_pointer(enc, pAllocator))
        assert(false);
    if (vn_encode_simple_pointer(enc, pQueryPool))
        vn_encode_VkQueryPool(enc, pQueryPool);
}

static inline size_t vn_sizeof_vkCreateQueryPool_reply(VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkCreateQueryPool_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    VkResult ret;
    cmd_size += vn_sizeof_VkResult(&ret);
    /* skip device */
    /* skip pCreateInfo */
    /* skip pAllocator */
    cmd_size += vn_sizeof_simple_pointer(pQueryPool);
    if (pQueryPool)
        cmd_size += vn_sizeof_VkQueryPool(pQueryPool);

    return cmd_size;
}

static inline VkResult vn_decode_vkCreateQueryPool_reply(struct vn_cs_decoder *dec, VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkCreateQueryPool_EXT);

    VkResult ret;
    vn_decode_VkResult(dec, &ret);
    /* skip device */
    /* skip pCreateInfo */
    /* skip pAllocator */
    if (vn_decode_simple_pointer(dec)) {
        vn_decode_VkQueryPool(dec, pQueryPool);
    } else {
        pQueryPool = NULL;
    }

    return ret;
}

static inline size_t vn_sizeof_vkDestroyQueryPool(VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkDestroyQueryPool_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_VkQueryPool(&queryPool);
    cmd_size += vn_sizeof_simple_pointer(pAllocator);
    if (pAllocator)
        assert(false);

    return cmd_size;
}

static inline void vn_encode_vkDestroyQueryPool(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkDestroyQueryPool_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    vn_encode_VkQueryPool(enc, &queryPool);
    if (vn_encode_simple_pointer(enc, pAllocator))
        assert(false);
}

static inline size_t vn_sizeof_vkDestroyQueryPool_reply(VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkDestroyQueryPool_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    /* skip device */
    /* skip queryPool */
    /* skip pAllocator */

    return cmd_size;
}

static inline void vn_decode_vkDestroyQueryPool_reply(struct vn_cs_decoder *dec, VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkDestroyQueryPool_EXT);

    /* skip device */
    /* skip queryPool */
    /* skip pAllocator */
}

static inline size_t vn_sizeof_vkGetQueryPoolResults(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkGetQueryPoolResults_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_VkQueryPool(&queryPool);
    cmd_size += vn_sizeof_uint32_t(&firstQuery);
    cmd_size += vn_sizeof_uint32_t(&queryCount);
    cmd_size += vn_sizeof_size_t(&dataSize);
    cmd_size += vn_sizeof_simple_pointer(pData); /* out */
    cmd_size += vn_sizeof_VkDeviceSize(&stride);
    cmd_size += vn_sizeof_VkFlags(&flags);

    return cmd_size;
}

static inline void vn_encode_vkGetQueryPoolResults(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkGetQueryPoolResults_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    vn_encode_VkQueryPool(enc, &queryPool);
    vn_encode_uint32_t(enc, &firstQuery);
    vn_encode_uint32_t(enc, &queryCount);
    vn_encode_size_t(enc, &dataSize);
    vn_encode_array_size(enc, pData ? dataSize : 0); /* out */
    vn_encode_VkDeviceSize(enc, &stride);
    vn_encode_VkFlags(enc, &flags);
}

static inline size_t vn_sizeof_vkGetQueryPoolResults_reply(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkGetQueryPoolResults_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    VkResult ret;
    cmd_size += vn_sizeof_VkResult(&ret);
    /* skip device */
    /* skip queryPool */
    /* skip firstQuery */
    /* skip queryCount */
    /* skip dataSize */
    if (pData) {
        cmd_size += vn_sizeof_array_size(dataSize);
        cmd_size += vn_sizeof_blob_array(pData, dataSize);
    } else {
        cmd_size += vn_sizeof_array_size(0);
    }
    /* skip stride */
    /* skip flags */

    return cmd_size;
}

static inline VkResult vn_decode_vkGetQueryPoolResults_reply(struct vn_cs_decoder *dec, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkGetQueryPoolResults_EXT);

    VkResult ret;
    vn_decode_VkResult(dec, &ret);
    /* skip device */
    /* skip queryPool */
    /* skip firstQuery */
    /* skip queryCount */
    /* skip dataSize */
    if (vn_peek_array_size(dec)) {
        const size_t array_size = vn_decode_array_size(dec, dataSize);
        vn_decode_blob_array(dec, pData, array_size);
    } else {
        vn_decode_array_size_unchecked(dec);
        pData = NULL;
    }
    /* skip stride */
    /* skip flags */

    return ret;
}

static inline size_t vn_sizeof_vkResetQueryPool(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkResetQueryPool_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_VkQueryPool(&queryPool);
    cmd_size += vn_sizeof_uint32_t(&firstQuery);
    cmd_size += vn_sizeof_uint32_t(&queryCount);

    return cmd_size;
}

static inline void vn_encode_vkResetQueryPool(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkResetQueryPool_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    vn_encode_VkQueryPool(enc, &queryPool);
    vn_encode_uint32_t(enc, &firstQuery);
    vn_encode_uint32_t(enc, &queryCount);
}

static inline size_t vn_sizeof_vkResetQueryPool_reply(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkResetQueryPool_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    /* skip device */
    /* skip queryPool */
    /* skip firstQuery */
    /* skip queryCount */

    return cmd_size;
}

static inline void vn_decode_vkResetQueryPool_reply(struct vn_cs_decoder *dec, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkResetQueryPool_EXT);

    /* skip device */
    /* skip queryPool */
    /* skip firstQuery */
    /* skip queryCount */
}

static inline void vn_submit_vkCreateQueryPool(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkCreateQueryPool(device, pCreateInfo, pAllocator, pQueryPool);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkCreateQueryPool_reply(device, pCreateInfo, pAllocator, pQueryPool) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkCreateQueryPool(enc, cmd_flags, device, pCreateInfo, pAllocator, pQueryPool);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline void vn_submit_vkDestroyQueryPool(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkDestroyQueryPool(device, queryPool, pAllocator);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkDestroyQueryPool_reply(device, queryPool, pAllocator) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkDestroyQueryPool(enc, cmd_flags, device, queryPool, pAllocator);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline void vn_submit_vkGetQueryPoolResults(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkGetQueryPoolResults(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkGetQueryPoolResults_reply(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkGetQueryPoolResults(enc, cmd_flags, device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline void vn_submit_vkResetQueryPool(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkResetQueryPool(device, queryPool, firstQuery, queryCount);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkResetQueryPool_reply(device, queryPool, firstQuery, queryCount) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkResetQueryPool(enc, cmd_flags, device, queryPool, firstQuery, queryCount);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline VkResult vn_call_vkCreateQueryPool(struct vn_ring *vn_ring, VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
    VN_TRACE_FUNC();

    struct vn_ring_submit_command submit;
    vn_submit_vkCreateQueryPool(vn_ring, VK_COMMAND_GENERATE_REPLY_BIT_EXT, device, pCreateInfo, pAllocator, pQueryPool, &submit);
    struct vn_cs_decoder *dec = vn_ring_get_command_reply(vn_ring, &submit);
    if (dec) {
        const VkResult ret = vn_decode_vkCreateQueryPool_reply(dec, device, pCreateInfo, pAllocator, pQueryPool);
        vn_ring_free_command_reply(vn_ring, &submit);
        return ret;
    } else {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}

static inline void vn_async_vkCreateQueryPool(struct vn_ring *vn_ring, VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkCreateQueryPool(vn_ring, 0, device, pCreateInfo, pAllocator, pQueryPool, &submit);
}

static inline void vn_async_vkDestroyQueryPool(struct vn_ring *vn_ring, VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkDestroyQueryPool(vn_ring, 0, device, queryPool, pAllocator, &submit);
}

static inline VkResult vn_call_vkGetQueryPoolResults(struct vn_ring *vn_ring, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
    VN_TRACE_FUNC();

    struct vn_ring_submit_command submit;
    vn_submit_vkGetQueryPoolResults(vn_ring, VK_COMMAND_GENERATE_REPLY_BIT_EXT, device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags, &submit);
    struct vn_cs_decoder *dec = vn_ring_get_command_reply(vn_ring, &submit);
    if (dec) {
        const VkResult ret = vn_decode_vkGetQueryPoolResults_reply(dec, device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
        vn_ring_free_command_reply(vn_ring, &submit);
        return ret;
    } else {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}

static inline void vn_async_vkGetQueryPoolResults(struct vn_ring *vn_ring, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkGetQueryPoolResults(vn_ring, 0, device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags, &submit);
}

static inline void vn_async_vkResetQueryPool(struct vn_ring *vn_ring, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkResetQueryPool(vn_ring, 0, device, queryPool, firstQuery, queryCount, &submit);
}

#endif /* VN_PROTOCOL_DRIVER_QUERY_POOL_H */
