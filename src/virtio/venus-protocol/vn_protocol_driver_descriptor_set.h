/* This file is generated by venus-protocol.  See vn_protocol_driver.h. */

/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VN_PROTOCOL_DRIVER_DESCRIPTOR_SET_H
#define VN_PROTOCOL_DRIVER_DESCRIPTOR_SET_H

#include "vn_ring.h"
#include "vn_protocol_driver_structs.h"

/*
 * These structs/unions/commands are not included
 *
 *   vkUpdateDescriptorSetWithTemplate
 */

/* struct VkDescriptorSetVariableDescriptorCountAllocateInfo chain */

static inline size_t
vn_sizeof_VkDescriptorSetVariableDescriptorCountAllocateInfo_pnext(const void *val)
{
    /* no known/supported struct */
    return vn_sizeof_simple_pointer(NULL);
}

static inline size_t
vn_sizeof_VkDescriptorSetVariableDescriptorCountAllocateInfo_self(const VkDescriptorSetVariableDescriptorCountAllocateInfo *val)
{
    size_t size = 0;
    /* skip val->{sType,pNext} */
    size += vn_sizeof_uint32_t(&val->descriptorSetCount);
    if (val->pDescriptorCounts) {
        size += vn_sizeof_array_size(val->descriptorSetCount);
        size += vn_sizeof_uint32_t_array(val->pDescriptorCounts, val->descriptorSetCount);
    } else {
        size += vn_sizeof_array_size(0);
    }
    return size;
}

static inline size_t
vn_sizeof_VkDescriptorSetVariableDescriptorCountAllocateInfo(const VkDescriptorSetVariableDescriptorCountAllocateInfo *val)
{
    size_t size = 0;

    size += vn_sizeof_VkStructureType(&val->sType);
    size += vn_sizeof_VkDescriptorSetVariableDescriptorCountAllocateInfo_pnext(val->pNext);
    size += vn_sizeof_VkDescriptorSetVariableDescriptorCountAllocateInfo_self(val);

    return size;
}

static inline void
vn_encode_VkDescriptorSetVariableDescriptorCountAllocateInfo_pnext(struct vn_cs_encoder *enc, const void *val)
{
    /* no known/supported struct */
    vn_encode_simple_pointer(enc, NULL);
}

static inline void
vn_encode_VkDescriptorSetVariableDescriptorCountAllocateInfo_self(struct vn_cs_encoder *enc, const VkDescriptorSetVariableDescriptorCountAllocateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_encode_uint32_t(enc, &val->descriptorSetCount);
    if (val->pDescriptorCounts) {
        vn_encode_array_size(enc, val->descriptorSetCount);
        vn_encode_uint32_t_array(enc, val->pDescriptorCounts, val->descriptorSetCount);
    } else {
        vn_encode_array_size(enc, 0);
    }
}

static inline void
vn_encode_VkDescriptorSetVariableDescriptorCountAllocateInfo(struct vn_cs_encoder *enc, const VkDescriptorSetVariableDescriptorCountAllocateInfo *val)
{
    assert(val->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);
    vn_encode_VkStructureType(enc, &(VkStructureType){ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO });
    vn_encode_VkDescriptorSetVariableDescriptorCountAllocateInfo_pnext(enc, val->pNext);
    vn_encode_VkDescriptorSetVariableDescriptorCountAllocateInfo_self(enc, val);
}

/* struct VkDescriptorSetAllocateInfo chain */

static inline size_t
vn_sizeof_VkDescriptorSetAllocateInfo_pnext(const void *val)
{
    const VkBaseInStructure *pnext = val;
    size_t size = 0;

    while (pnext) {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO:
            size += vn_sizeof_simple_pointer(pnext);
            size += vn_sizeof_VkStructureType(&pnext->sType);
            size += vn_sizeof_VkDescriptorSetAllocateInfo_pnext(pnext->pNext);
            size += vn_sizeof_VkDescriptorSetVariableDescriptorCountAllocateInfo_self((const VkDescriptorSetVariableDescriptorCountAllocateInfo *)pnext);
            return size;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    }

    return vn_sizeof_simple_pointer(NULL);
}

static inline size_t
vn_sizeof_VkDescriptorSetAllocateInfo_self(const VkDescriptorSetAllocateInfo *val)
{
    size_t size = 0;
    /* skip val->{sType,pNext} */
    size += vn_sizeof_VkDescriptorPool(&val->descriptorPool);
    size += vn_sizeof_uint32_t(&val->descriptorSetCount);
    if (val->pSetLayouts) {
        size += vn_sizeof_array_size(val->descriptorSetCount);
        for (uint32_t i = 0; i < val->descriptorSetCount; i++)
            size += vn_sizeof_VkDescriptorSetLayout(&val->pSetLayouts[i]);
    } else {
        size += vn_sizeof_array_size(0);
    }
    return size;
}

static inline size_t
vn_sizeof_VkDescriptorSetAllocateInfo(const VkDescriptorSetAllocateInfo *val)
{
    size_t size = 0;

    size += vn_sizeof_VkStructureType(&val->sType);
    size += vn_sizeof_VkDescriptorSetAllocateInfo_pnext(val->pNext);
    size += vn_sizeof_VkDescriptorSetAllocateInfo_self(val);

    return size;
}

static inline void
vn_encode_VkDescriptorSetAllocateInfo_pnext(struct vn_cs_encoder *enc, const void *val)
{
    const VkBaseInStructure *pnext = val;

    while (pnext) {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO:
            vn_encode_simple_pointer(enc, pnext);
            vn_encode_VkStructureType(enc, &pnext->sType);
            vn_encode_VkDescriptorSetAllocateInfo_pnext(enc, pnext->pNext);
            vn_encode_VkDescriptorSetVariableDescriptorCountAllocateInfo_self(enc, (const VkDescriptorSetVariableDescriptorCountAllocateInfo *)pnext);
            return;
        default:
            /* ignore unknown/unsupported struct */
            break;
        }
        pnext = pnext->pNext;
    }

    vn_encode_simple_pointer(enc, NULL);
}

static inline void
vn_encode_VkDescriptorSetAllocateInfo_self(struct vn_cs_encoder *enc, const VkDescriptorSetAllocateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_encode_VkDescriptorPool(enc, &val->descriptorPool);
    vn_encode_uint32_t(enc, &val->descriptorSetCount);
    if (val->pSetLayouts) {
        vn_encode_array_size(enc, val->descriptorSetCount);
        for (uint32_t i = 0; i < val->descriptorSetCount; i++)
            vn_encode_VkDescriptorSetLayout(enc, &val->pSetLayouts[i]);
    } else {
        vn_encode_array_size(enc, 0);
    }
}

static inline void
vn_encode_VkDescriptorSetAllocateInfo(struct vn_cs_encoder *enc, const VkDescriptorSetAllocateInfo *val)
{
    assert(val->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    vn_encode_VkStructureType(enc, &(VkStructureType){ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO });
    vn_encode_VkDescriptorSetAllocateInfo_pnext(enc, val->pNext);
    vn_encode_VkDescriptorSetAllocateInfo_self(enc, val);
}

/* struct VkCopyDescriptorSet chain */

static inline size_t
vn_sizeof_VkCopyDescriptorSet_pnext(const void *val)
{
    /* no known/supported struct */
    return vn_sizeof_simple_pointer(NULL);
}

static inline size_t
vn_sizeof_VkCopyDescriptorSet_self(const VkCopyDescriptorSet *val)
{
    size_t size = 0;
    /* skip val->{sType,pNext} */
    size += vn_sizeof_VkDescriptorSet(&val->srcSet);
    size += vn_sizeof_uint32_t(&val->srcBinding);
    size += vn_sizeof_uint32_t(&val->srcArrayElement);
    size += vn_sizeof_VkDescriptorSet(&val->dstSet);
    size += vn_sizeof_uint32_t(&val->dstBinding);
    size += vn_sizeof_uint32_t(&val->dstArrayElement);
    size += vn_sizeof_uint32_t(&val->descriptorCount);
    return size;
}

static inline size_t
vn_sizeof_VkCopyDescriptorSet(const VkCopyDescriptorSet *val)
{
    size_t size = 0;

    size += vn_sizeof_VkStructureType(&val->sType);
    size += vn_sizeof_VkCopyDescriptorSet_pnext(val->pNext);
    size += vn_sizeof_VkCopyDescriptorSet_self(val);

    return size;
}

static inline void
vn_encode_VkCopyDescriptorSet_pnext(struct vn_cs_encoder *enc, const void *val)
{
    /* no known/supported struct */
    vn_encode_simple_pointer(enc, NULL);
}

static inline void
vn_encode_VkCopyDescriptorSet_self(struct vn_cs_encoder *enc, const VkCopyDescriptorSet *val)
{
    /* skip val->{sType,pNext} */
    vn_encode_VkDescriptorSet(enc, &val->srcSet);
    vn_encode_uint32_t(enc, &val->srcBinding);
    vn_encode_uint32_t(enc, &val->srcArrayElement);
    vn_encode_VkDescriptorSet(enc, &val->dstSet);
    vn_encode_uint32_t(enc, &val->dstBinding);
    vn_encode_uint32_t(enc, &val->dstArrayElement);
    vn_encode_uint32_t(enc, &val->descriptorCount);
}

static inline void
vn_encode_VkCopyDescriptorSet(struct vn_cs_encoder *enc, const VkCopyDescriptorSet *val)
{
    assert(val->sType == VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET);
    vn_encode_VkStructureType(enc, &(VkStructureType){ VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET });
    vn_encode_VkCopyDescriptorSet_pnext(enc, val->pNext);
    vn_encode_VkCopyDescriptorSet_self(enc, val);
}

static inline size_t vn_sizeof_vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkAllocateDescriptorSets_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_simple_pointer(pAllocateInfo);
    if (pAllocateInfo)
        cmd_size += vn_sizeof_VkDescriptorSetAllocateInfo(pAllocateInfo);
    if (pDescriptorSets) {
        cmd_size += vn_sizeof_array_size((pAllocateInfo ? pAllocateInfo->descriptorSetCount : 0));
        for (uint32_t i = 0; i < (pAllocateInfo ? pAllocateInfo->descriptorSetCount : 0); i++)
            cmd_size += vn_sizeof_VkDescriptorSet(&pDescriptorSets[i]);
    } else {
        cmd_size += vn_sizeof_array_size(0);
    }

    return cmd_size;
}

static inline void vn_encode_vkAllocateDescriptorSets(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkAllocateDescriptorSets_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    if (vn_encode_simple_pointer(enc, pAllocateInfo))
        vn_encode_VkDescriptorSetAllocateInfo(enc, pAllocateInfo);
    if (pDescriptorSets) {
        vn_encode_array_size(enc, (pAllocateInfo ? pAllocateInfo->descriptorSetCount : 0));
        for (uint32_t i = 0; i < (pAllocateInfo ? pAllocateInfo->descriptorSetCount : 0); i++)
            vn_encode_VkDescriptorSet(enc, &pDescriptorSets[i]);
    } else {
        vn_encode_array_size(enc, 0);
    }
}

static inline size_t vn_sizeof_vkAllocateDescriptorSets_reply(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkAllocateDescriptorSets_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    VkResult ret;
    cmd_size += vn_sizeof_VkResult(&ret);
    /* skip device */
    /* skip pAllocateInfo */
    if (pDescriptorSets) {
        cmd_size += vn_sizeof_array_size((pAllocateInfo ? pAllocateInfo->descriptorSetCount : 0));
        for (uint32_t i = 0; i < (pAllocateInfo ? pAllocateInfo->descriptorSetCount : 0); i++)
            cmd_size += vn_sizeof_VkDescriptorSet(&pDescriptorSets[i]);
    } else {
        cmd_size += vn_sizeof_array_size(0);
    }

    return cmd_size;
}

static inline VkResult vn_decode_vkAllocateDescriptorSets_reply(struct vn_cs_decoder *dec, VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkAllocateDescriptorSets_EXT);

    VkResult ret;
    vn_decode_VkResult(dec, &ret);
    /* skip device */
    /* skip pAllocateInfo */
    if (vn_peek_array_size(dec)) {
        const uint32_t iter_count = vn_decode_array_size(dec, (pAllocateInfo ? pAllocateInfo->descriptorSetCount : 0));
        for (uint32_t i = 0; i < iter_count; i++)
            vn_decode_VkDescriptorSet(dec, &pDescriptorSets[i]);
    } else {
        vn_decode_array_size_unchecked(dec);
        pDescriptorSets = NULL;
    }

    return ret;
}

static inline size_t vn_sizeof_vkFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkFreeDescriptorSets_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_VkDescriptorPool(&descriptorPool);
    cmd_size += vn_sizeof_uint32_t(&descriptorSetCount);
    if (pDescriptorSets) {
        cmd_size += vn_sizeof_array_size(descriptorSetCount);
        for (uint32_t i = 0; i < descriptorSetCount; i++)
            cmd_size += vn_sizeof_VkDescriptorSet(&pDescriptorSets[i]);
    } else {
        cmd_size += vn_sizeof_array_size(0);
    }

    return cmd_size;
}

static inline void vn_encode_vkFreeDescriptorSets(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkFreeDescriptorSets_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    vn_encode_VkDescriptorPool(enc, &descriptorPool);
    vn_encode_uint32_t(enc, &descriptorSetCount);
    if (pDescriptorSets) {
        vn_encode_array_size(enc, descriptorSetCount);
        for (uint32_t i = 0; i < descriptorSetCount; i++)
            vn_encode_VkDescriptorSet(enc, &pDescriptorSets[i]);
    } else {
        vn_encode_array_size(enc, 0);
    }
}

static inline size_t vn_sizeof_vkFreeDescriptorSets_reply(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkFreeDescriptorSets_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    VkResult ret;
    cmd_size += vn_sizeof_VkResult(&ret);
    /* skip device */
    /* skip descriptorPool */
    /* skip descriptorSetCount */
    /* skip pDescriptorSets */

    return cmd_size;
}

static inline VkResult vn_decode_vkFreeDescriptorSets_reply(struct vn_cs_decoder *dec, VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkFreeDescriptorSets_EXT);

    VkResult ret;
    vn_decode_VkResult(dec, &ret);
    /* skip device */
    /* skip descriptorPool */
    /* skip descriptorSetCount */
    /* skip pDescriptorSets */

    return ret;
}

static inline size_t vn_sizeof_vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkUpdateDescriptorSets_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_uint32_t(&descriptorWriteCount);
    if (pDescriptorWrites) {
        cmd_size += vn_sizeof_array_size(descriptorWriteCount);
        for (uint32_t i = 0; i < descriptorWriteCount; i++)
            cmd_size += vn_sizeof_VkWriteDescriptorSet(&pDescriptorWrites[i]);
    } else {
        cmd_size += vn_sizeof_array_size(0);
    }
    cmd_size += vn_sizeof_uint32_t(&descriptorCopyCount);
    if (pDescriptorCopies) {
        cmd_size += vn_sizeof_array_size(descriptorCopyCount);
        for (uint32_t i = 0; i < descriptorCopyCount; i++)
            cmd_size += vn_sizeof_VkCopyDescriptorSet(&pDescriptorCopies[i]);
    } else {
        cmd_size += vn_sizeof_array_size(0);
    }

    return cmd_size;
}

static inline void vn_encode_vkUpdateDescriptorSets(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkUpdateDescriptorSets_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    vn_encode_uint32_t(enc, &descriptorWriteCount);
    if (pDescriptorWrites) {
        vn_encode_array_size(enc, descriptorWriteCount);
        for (uint32_t i = 0; i < descriptorWriteCount; i++)
            vn_encode_VkWriteDescriptorSet(enc, &pDescriptorWrites[i]);
    } else {
        vn_encode_array_size(enc, 0);
    }
    vn_encode_uint32_t(enc, &descriptorCopyCount);
    if (pDescriptorCopies) {
        vn_encode_array_size(enc, descriptorCopyCount);
        for (uint32_t i = 0; i < descriptorCopyCount; i++)
            vn_encode_VkCopyDescriptorSet(enc, &pDescriptorCopies[i]);
    } else {
        vn_encode_array_size(enc, 0);
    }
}

static inline size_t vn_sizeof_vkUpdateDescriptorSets_reply(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkUpdateDescriptorSets_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    /* skip device */
    /* skip descriptorWriteCount */
    /* skip pDescriptorWrites */
    /* skip descriptorCopyCount */
    /* skip pDescriptorCopies */

    return cmd_size;
}

static inline void vn_decode_vkUpdateDescriptorSets_reply(struct vn_cs_decoder *dec, VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkUpdateDescriptorSets_EXT);

    /* skip device */
    /* skip descriptorWriteCount */
    /* skip pDescriptorWrites */
    /* skip descriptorCopyCount */
    /* skip pDescriptorCopies */
}

static inline void vn_submit_vkAllocateDescriptorSets(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkAllocateDescriptorSets(device, pAllocateInfo, pDescriptorSets);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkAllocateDescriptorSets_reply(device, pAllocateInfo, pDescriptorSets) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkAllocateDescriptorSets(enc, cmd_flags, device, pAllocateInfo, pDescriptorSets);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline void vn_submit_vkFreeDescriptorSets(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkFreeDescriptorSets(device, descriptorPool, descriptorSetCount, pDescriptorSets);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkFreeDescriptorSets_reply(device, descriptorPool, descriptorSetCount, pDescriptorSets) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkFreeDescriptorSets(enc, cmd_flags, device, descriptorPool, descriptorSetCount, pDescriptorSets);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline void vn_submit_vkUpdateDescriptorSets(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkUpdateDescriptorSets(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkUpdateDescriptorSets_reply(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkUpdateDescriptorSets(enc, cmd_flags, device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline VkResult vn_call_vkAllocateDescriptorSets(struct vn_ring *vn_ring, VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    VN_TRACE_FUNC();

    struct vn_ring_submit_command submit;
    vn_submit_vkAllocateDescriptorSets(vn_ring, VK_COMMAND_GENERATE_REPLY_BIT_EXT, device, pAllocateInfo, pDescriptorSets, &submit);
    struct vn_cs_decoder *dec = vn_ring_get_command_reply(vn_ring, &submit);
    if (dec) {
        const VkResult ret = vn_decode_vkAllocateDescriptorSets_reply(dec, device, pAllocateInfo, pDescriptorSets);
        vn_ring_free_command_reply(vn_ring, &submit);
        return ret;
    } else {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}

static inline void vn_async_vkAllocateDescriptorSets(struct vn_ring *vn_ring, VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkAllocateDescriptorSets(vn_ring, 0, device, pAllocateInfo, pDescriptorSets, &submit);
}

static inline VkResult vn_call_vkFreeDescriptorSets(struct vn_ring *vn_ring, VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    VN_TRACE_FUNC();

    struct vn_ring_submit_command submit;
    vn_submit_vkFreeDescriptorSets(vn_ring, VK_COMMAND_GENERATE_REPLY_BIT_EXT, device, descriptorPool, descriptorSetCount, pDescriptorSets, &submit);
    struct vn_cs_decoder *dec = vn_ring_get_command_reply(vn_ring, &submit);
    if (dec) {
        const VkResult ret = vn_decode_vkFreeDescriptorSets_reply(dec, device, descriptorPool, descriptorSetCount, pDescriptorSets);
        vn_ring_free_command_reply(vn_ring, &submit);
        return ret;
    } else {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}

static inline void vn_async_vkFreeDescriptorSets(struct vn_ring *vn_ring, VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkFreeDescriptorSets(vn_ring, 0, device, descriptorPool, descriptorSetCount, pDescriptorSets, &submit);
}

static inline void vn_async_vkUpdateDescriptorSets(struct vn_ring *vn_ring, VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkUpdateDescriptorSets(vn_ring, 0, device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies, &submit);
}

#endif /* VN_PROTOCOL_DRIVER_DESCRIPTOR_SET_H */
