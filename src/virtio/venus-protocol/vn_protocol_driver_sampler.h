/* This file is generated by venus-protocol.  See vn_protocol_driver.h. */

/*
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef VN_PROTOCOL_DRIVER_SAMPLER_H
#define VN_PROTOCOL_DRIVER_SAMPLER_H

#include "vn_ring.h"
#include "vn_protocol_driver_structs.h"

/* struct VkSamplerReductionModeCreateInfo chain */

static inline size_t
vn_sizeof_VkSamplerReductionModeCreateInfo_pnext(const void *val)
{
    /* no known/supported struct */
    return vn_sizeof_simple_pointer(NULL);
}

static inline size_t
vn_sizeof_VkSamplerReductionModeCreateInfo_self(const VkSamplerReductionModeCreateInfo *val)
{
    size_t size = 0;
    /* skip val->{sType,pNext} */
    size += vn_sizeof_VkSamplerReductionMode(&val->reductionMode);
    return size;
}

static inline size_t
vn_sizeof_VkSamplerReductionModeCreateInfo(const VkSamplerReductionModeCreateInfo *val)
{
    size_t size = 0;

    size += vn_sizeof_VkStructureType(&val->sType);
    size += vn_sizeof_VkSamplerReductionModeCreateInfo_pnext(val->pNext);
    size += vn_sizeof_VkSamplerReductionModeCreateInfo_self(val);

    return size;
}

static inline void
vn_encode_VkSamplerReductionModeCreateInfo_pnext(struct vn_cs_encoder *enc, const void *val)
{
    /* no known/supported struct */
    vn_encode_simple_pointer(enc, NULL);
}

static inline void
vn_encode_VkSamplerReductionModeCreateInfo_self(struct vn_cs_encoder *enc, const VkSamplerReductionModeCreateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_encode_VkSamplerReductionMode(enc, &val->reductionMode);
}

static inline void
vn_encode_VkSamplerReductionModeCreateInfo(struct vn_cs_encoder *enc, const VkSamplerReductionModeCreateInfo *val)
{
    assert(val->sType == VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO);
    vn_encode_VkStructureType(enc, &(VkStructureType){ VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO });
    vn_encode_VkSamplerReductionModeCreateInfo_pnext(enc, val->pNext);
    vn_encode_VkSamplerReductionModeCreateInfo_self(enc, val);
}

/* struct VkSamplerCustomBorderColorCreateInfoEXT chain */

static inline size_t
vn_sizeof_VkSamplerCustomBorderColorCreateInfoEXT_pnext(const void *val)
{
    /* no known/supported struct */
    return vn_sizeof_simple_pointer(NULL);
}

static inline size_t
vn_sizeof_VkSamplerCustomBorderColorCreateInfoEXT_self(const VkSamplerCustomBorderColorCreateInfoEXT *val)
{
    size_t size = 0;
    /* skip val->{sType,pNext} */
    size += vn_sizeof_VkClearColorValue(&val->customBorderColor);
    size += vn_sizeof_VkFormat(&val->format);
    return size;
}

static inline size_t
vn_sizeof_VkSamplerCustomBorderColorCreateInfoEXT(const VkSamplerCustomBorderColorCreateInfoEXT *val)
{
    size_t size = 0;

    size += vn_sizeof_VkStructureType(&val->sType);
    size += vn_sizeof_VkSamplerCustomBorderColorCreateInfoEXT_pnext(val->pNext);
    size += vn_sizeof_VkSamplerCustomBorderColorCreateInfoEXT_self(val);

    return size;
}

static inline void
vn_encode_VkSamplerCustomBorderColorCreateInfoEXT_pnext(struct vn_cs_encoder *enc, const void *val)
{
    /* no known/supported struct */
    vn_encode_simple_pointer(enc, NULL);
}

static inline void
vn_encode_VkSamplerCustomBorderColorCreateInfoEXT_self(struct vn_cs_encoder *enc, const VkSamplerCustomBorderColorCreateInfoEXT *val)
{
    /* skip val->{sType,pNext} */
    vn_encode_VkClearColorValue(enc, &val->customBorderColor);
    vn_encode_VkFormat(enc, &val->format);
}

static inline void
vn_encode_VkSamplerCustomBorderColorCreateInfoEXT(struct vn_cs_encoder *enc, const VkSamplerCustomBorderColorCreateInfoEXT *val)
{
    assert(val->sType == VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT);
    vn_encode_VkStructureType(enc, &(VkStructureType){ VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT });
    vn_encode_VkSamplerCustomBorderColorCreateInfoEXT_pnext(enc, val->pNext);
    vn_encode_VkSamplerCustomBorderColorCreateInfoEXT_self(enc, val);
}

/* struct VkSamplerBorderColorComponentMappingCreateInfoEXT chain */

static inline size_t
vn_sizeof_VkSamplerBorderColorComponentMappingCreateInfoEXT_pnext(const void *val)
{
    /* no known/supported struct */
    return vn_sizeof_simple_pointer(NULL);
}

static inline size_t
vn_sizeof_VkSamplerBorderColorComponentMappingCreateInfoEXT_self(const VkSamplerBorderColorComponentMappingCreateInfoEXT *val)
{
    size_t size = 0;
    /* skip val->{sType,pNext} */
    size += vn_sizeof_VkComponentMapping(&val->components);
    size += vn_sizeof_VkBool32(&val->srgb);
    return size;
}

static inline size_t
vn_sizeof_VkSamplerBorderColorComponentMappingCreateInfoEXT(const VkSamplerBorderColorComponentMappingCreateInfoEXT *val)
{
    size_t size = 0;

    size += vn_sizeof_VkStructureType(&val->sType);
    size += vn_sizeof_VkSamplerBorderColorComponentMappingCreateInfoEXT_pnext(val->pNext);
    size += vn_sizeof_VkSamplerBorderColorComponentMappingCreateInfoEXT_self(val);

    return size;
}

static inline void
vn_encode_VkSamplerBorderColorComponentMappingCreateInfoEXT_pnext(struct vn_cs_encoder *enc, const void *val)
{
    /* no known/supported struct */
    vn_encode_simple_pointer(enc, NULL);
}

static inline void
vn_encode_VkSamplerBorderColorComponentMappingCreateInfoEXT_self(struct vn_cs_encoder *enc, const VkSamplerBorderColorComponentMappingCreateInfoEXT *val)
{
    /* skip val->{sType,pNext} */
    vn_encode_VkComponentMapping(enc, &val->components);
    vn_encode_VkBool32(enc, &val->srgb);
}

static inline void
vn_encode_VkSamplerBorderColorComponentMappingCreateInfoEXT(struct vn_cs_encoder *enc, const VkSamplerBorderColorComponentMappingCreateInfoEXT *val)
{
    assert(val->sType == VK_STRUCTURE_TYPE_SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT);
    vn_encode_VkStructureType(enc, &(VkStructureType){ VK_STRUCTURE_TYPE_SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT });
    vn_encode_VkSamplerBorderColorComponentMappingCreateInfoEXT_pnext(enc, val->pNext);
    vn_encode_VkSamplerBorderColorComponentMappingCreateInfoEXT_self(enc, val);
}

/* struct VkSamplerCreateInfo chain */

static inline size_t
vn_sizeof_VkSamplerCreateInfo_pnext(const void *val)
{
    const VkBaseInStructure *pnext = val;
    size_t size = 0;

    while (pnext) {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO:
            size += vn_sizeof_simple_pointer(pnext);
            size += vn_sizeof_VkStructureType(&pnext->sType);
            size += vn_sizeof_VkSamplerCreateInfo_pnext(pnext->pNext);
            size += vn_sizeof_VkSamplerYcbcrConversionInfo_self((const VkSamplerYcbcrConversionInfo *)pnext);
            return size;
        case VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO:
            size += vn_sizeof_simple_pointer(pnext);
            size += vn_sizeof_VkStructureType(&pnext->sType);
            size += vn_sizeof_VkSamplerCreateInfo_pnext(pnext->pNext);
            size += vn_sizeof_VkSamplerReductionModeCreateInfo_self((const VkSamplerReductionModeCreateInfo *)pnext);
            return size;
        case VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT:
            if (!vn_cs_renderer_protocol_has_extension(288 /* VK_EXT_custom_border_color */))
                break;
            size += vn_sizeof_simple_pointer(pnext);
            size += vn_sizeof_VkStructureType(&pnext->sType);
            size += vn_sizeof_VkSamplerCreateInfo_pnext(pnext->pNext);
            size += vn_sizeof_VkSamplerCustomBorderColorCreateInfoEXT_self((const VkSamplerCustomBorderColorCreateInfoEXT *)pnext);
            return size;
        case VK_STRUCTURE_TYPE_SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT:
            if (!vn_cs_renderer_protocol_has_extension(412 /* VK_EXT_border_color_swizzle */))
                break;
            size += vn_sizeof_simple_pointer(pnext);
            size += vn_sizeof_VkStructureType(&pnext->sType);
            size += vn_sizeof_VkSamplerCreateInfo_pnext(pnext->pNext);
            size += vn_sizeof_VkSamplerBorderColorComponentMappingCreateInfoEXT_self((const VkSamplerBorderColorComponentMappingCreateInfoEXT *)pnext);
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
vn_sizeof_VkSamplerCreateInfo_self(const VkSamplerCreateInfo *val)
{
    size_t size = 0;
    /* skip val->{sType,pNext} */
    size += vn_sizeof_VkFlags(&val->flags);
    size += vn_sizeof_VkFilter(&val->magFilter);
    size += vn_sizeof_VkFilter(&val->minFilter);
    size += vn_sizeof_VkSamplerMipmapMode(&val->mipmapMode);
    size += vn_sizeof_VkSamplerAddressMode(&val->addressModeU);
    size += vn_sizeof_VkSamplerAddressMode(&val->addressModeV);
    size += vn_sizeof_VkSamplerAddressMode(&val->addressModeW);
    size += vn_sizeof_float(&val->mipLodBias);
    size += vn_sizeof_VkBool32(&val->anisotropyEnable);
    size += vn_sizeof_float(&val->maxAnisotropy);
    size += vn_sizeof_VkBool32(&val->compareEnable);
    size += vn_sizeof_VkCompareOp(&val->compareOp);
    size += vn_sizeof_float(&val->minLod);
    size += vn_sizeof_float(&val->maxLod);
    size += vn_sizeof_VkBorderColor(&val->borderColor);
    size += vn_sizeof_VkBool32(&val->unnormalizedCoordinates);
    return size;
}

static inline size_t
vn_sizeof_VkSamplerCreateInfo(const VkSamplerCreateInfo *val)
{
    size_t size = 0;

    size += vn_sizeof_VkStructureType(&val->sType);
    size += vn_sizeof_VkSamplerCreateInfo_pnext(val->pNext);
    size += vn_sizeof_VkSamplerCreateInfo_self(val);

    return size;
}

static inline void
vn_encode_VkSamplerCreateInfo_pnext(struct vn_cs_encoder *enc, const void *val)
{
    const VkBaseInStructure *pnext = val;

    while (pnext) {
        switch ((int32_t)pnext->sType) {
        case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO:
            vn_encode_simple_pointer(enc, pnext);
            vn_encode_VkStructureType(enc, &pnext->sType);
            vn_encode_VkSamplerCreateInfo_pnext(enc, pnext->pNext);
            vn_encode_VkSamplerYcbcrConversionInfo_self(enc, (const VkSamplerYcbcrConversionInfo *)pnext);
            return;
        case VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO:
            vn_encode_simple_pointer(enc, pnext);
            vn_encode_VkStructureType(enc, &pnext->sType);
            vn_encode_VkSamplerCreateInfo_pnext(enc, pnext->pNext);
            vn_encode_VkSamplerReductionModeCreateInfo_self(enc, (const VkSamplerReductionModeCreateInfo *)pnext);
            return;
        case VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT:
            if (!vn_cs_renderer_protocol_has_extension(288 /* VK_EXT_custom_border_color */))
                break;
            vn_encode_simple_pointer(enc, pnext);
            vn_encode_VkStructureType(enc, &pnext->sType);
            vn_encode_VkSamplerCreateInfo_pnext(enc, pnext->pNext);
            vn_encode_VkSamplerCustomBorderColorCreateInfoEXT_self(enc, (const VkSamplerCustomBorderColorCreateInfoEXT *)pnext);
            return;
        case VK_STRUCTURE_TYPE_SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT:
            if (!vn_cs_renderer_protocol_has_extension(412 /* VK_EXT_border_color_swizzle */))
                break;
            vn_encode_simple_pointer(enc, pnext);
            vn_encode_VkStructureType(enc, &pnext->sType);
            vn_encode_VkSamplerCreateInfo_pnext(enc, pnext->pNext);
            vn_encode_VkSamplerBorderColorComponentMappingCreateInfoEXT_self(enc, (const VkSamplerBorderColorComponentMappingCreateInfoEXT *)pnext);
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
vn_encode_VkSamplerCreateInfo_self(struct vn_cs_encoder *enc, const VkSamplerCreateInfo *val)
{
    /* skip val->{sType,pNext} */
    vn_encode_VkFlags(enc, &val->flags);
    vn_encode_VkFilter(enc, &val->magFilter);
    vn_encode_VkFilter(enc, &val->minFilter);
    vn_encode_VkSamplerMipmapMode(enc, &val->mipmapMode);
    vn_encode_VkSamplerAddressMode(enc, &val->addressModeU);
    vn_encode_VkSamplerAddressMode(enc, &val->addressModeV);
    vn_encode_VkSamplerAddressMode(enc, &val->addressModeW);
    vn_encode_float(enc, &val->mipLodBias);
    vn_encode_VkBool32(enc, &val->anisotropyEnable);
    vn_encode_float(enc, &val->maxAnisotropy);
    vn_encode_VkBool32(enc, &val->compareEnable);
    vn_encode_VkCompareOp(enc, &val->compareOp);
    vn_encode_float(enc, &val->minLod);
    vn_encode_float(enc, &val->maxLod);
    vn_encode_VkBorderColor(enc, &val->borderColor);
    vn_encode_VkBool32(enc, &val->unnormalizedCoordinates);
}

static inline void
vn_encode_VkSamplerCreateInfo(struct vn_cs_encoder *enc, const VkSamplerCreateInfo *val)
{
    assert(val->sType == VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
    vn_encode_VkStructureType(enc, &(VkStructureType){ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO });
    vn_encode_VkSamplerCreateInfo_pnext(enc, val->pNext);
    vn_encode_VkSamplerCreateInfo_self(enc, val);
}

static inline size_t vn_sizeof_vkCreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkCreateSampler_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_simple_pointer(pCreateInfo);
    if (pCreateInfo)
        cmd_size += vn_sizeof_VkSamplerCreateInfo(pCreateInfo);
    cmd_size += vn_sizeof_simple_pointer(pAllocator);
    if (pAllocator)
        assert(false);
    cmd_size += vn_sizeof_simple_pointer(pSampler);
    if (pSampler)
        cmd_size += vn_sizeof_VkSampler(pSampler);

    return cmd_size;
}

static inline void vn_encode_vkCreateSampler(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkCreateSampler_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    if (vn_encode_simple_pointer(enc, pCreateInfo))
        vn_encode_VkSamplerCreateInfo(enc, pCreateInfo);
    if (vn_encode_simple_pointer(enc, pAllocator))
        assert(false);
    if (vn_encode_simple_pointer(enc, pSampler))
        vn_encode_VkSampler(enc, pSampler);
}

static inline size_t vn_sizeof_vkCreateSampler_reply(VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkCreateSampler_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    VkResult ret;
    cmd_size += vn_sizeof_VkResult(&ret);
    /* skip device */
    /* skip pCreateInfo */
    /* skip pAllocator */
    cmd_size += vn_sizeof_simple_pointer(pSampler);
    if (pSampler)
        cmd_size += vn_sizeof_VkSampler(pSampler);

    return cmd_size;
}

static inline VkResult vn_decode_vkCreateSampler_reply(struct vn_cs_decoder *dec, VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkCreateSampler_EXT);

    VkResult ret;
    vn_decode_VkResult(dec, &ret);
    /* skip device */
    /* skip pCreateInfo */
    /* skip pAllocator */
    if (vn_decode_simple_pointer(dec)) {
        vn_decode_VkSampler(dec, pSampler);
    } else {
        pSampler = NULL;
    }

    return ret;
}

static inline size_t vn_sizeof_vkDestroySampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkDestroySampler_EXT;
    const VkFlags cmd_flags = 0;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type) + vn_sizeof_VkFlags(&cmd_flags);

    cmd_size += vn_sizeof_VkDevice(&device);
    cmd_size += vn_sizeof_VkSampler(&sampler);
    cmd_size += vn_sizeof_simple_pointer(pAllocator);
    if (pAllocator)
        assert(false);

    return cmd_size;
}

static inline void vn_encode_vkDestroySampler(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkDestroySampler_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkDevice(enc, &device);
    vn_encode_VkSampler(enc, &sampler);
    if (vn_encode_simple_pointer(enc, pAllocator))
        assert(false);
}

static inline size_t vn_sizeof_vkDestroySampler_reply(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkDestroySampler_EXT;
    size_t cmd_size = vn_sizeof_VkCommandTypeEXT(&cmd_type);

    /* skip device */
    /* skip sampler */
    /* skip pAllocator */

    return cmd_size;
}

static inline void vn_decode_vkDestroySampler_reply(struct vn_cs_decoder *dec, VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator)
{
    VkCommandTypeEXT command_type;
    vn_decode_VkCommandTypeEXT(dec, &command_type);
    assert(command_type == VK_COMMAND_TYPE_vkDestroySampler_EXT);

    /* skip device */
    /* skip sampler */
    /* skip pAllocator */
}

static inline void vn_submit_vkCreateSampler(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkCreateSampler(device, pCreateInfo, pAllocator, pSampler);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkCreateSampler_reply(device, pCreateInfo, pAllocator, pSampler) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkCreateSampler(enc, cmd_flags, device, pCreateInfo, pAllocator, pSampler);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline void vn_submit_vkDestroySampler(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkDestroySampler(device, sampler, pAllocator);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkDestroySampler_reply(device, sampler, pAllocator) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkDestroySampler(enc, cmd_flags, device, sampler, pAllocator);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

static inline VkResult vn_call_vkCreateSampler(struct vn_ring *vn_ring, VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
    VN_TRACE_FUNC();

    struct vn_ring_submit_command submit;
    vn_submit_vkCreateSampler(vn_ring, VK_COMMAND_GENERATE_REPLY_BIT_EXT, device, pCreateInfo, pAllocator, pSampler, &submit);
    struct vn_cs_decoder *dec = vn_ring_get_command_reply(vn_ring, &submit);
    if (dec) {
        const VkResult ret = vn_decode_vkCreateSampler_reply(dec, device, pCreateInfo, pAllocator, pSampler);
        vn_ring_free_command_reply(vn_ring, &submit);
        return ret;
    } else {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}

static inline void vn_async_vkCreateSampler(struct vn_ring *vn_ring, VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkCreateSampler(vn_ring, 0, device, pCreateInfo, pAllocator, pSampler, &submit);
}

static inline void vn_async_vkDestroySampler(struct vn_ring *vn_ring, VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator)
{
    struct vn_ring_submit_command submit;
    vn_submit_vkDestroySampler(vn_ring, 0, device, sampler, pAllocator, &submit);
}

#endif /* VN_PROTOCOL_DRIVER_SAMPLER_H */
