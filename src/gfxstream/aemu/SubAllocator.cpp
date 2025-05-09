/*
 * Copyright 2019 Google
 * SPDX-License-Identifier: MIT
 */

#include "Stream.h"
#include "SubAllocator.h"
#include "address_space.h"
#include "util/log.h"

namespace gfxstream {
namespace aemu {

class SubAllocator::Impl {
  public:
   Impl(void* _buffer, uint64_t _totalSize, uint64_t _pageSize)
       : buffer(_buffer),
         totalSize(_totalSize),
         pageSize(_pageSize),
         startAddr((uintptr_t)buffer),
         endAddr(startAddr + totalSize) {
      address_space_allocator_init(&addr_alloc, totalSize, 32);
   }

   ~Impl() { address_space_allocator_destroy_nocleanup(&addr_alloc); }

   void clear() {
      address_space_allocator_destroy_nocleanup(&addr_alloc);
      address_space_allocator_init(&addr_alloc, totalSize, 32);
   }

   bool save(Stream* stream) {
      address_space_allocator_iter_func_t allocatorSaver =
          [](void* context, struct address_space_allocator* allocator) {
             Stream* stream = reinterpret_cast<Stream*>(context);
             stream->putBe32(allocator->size);
             stream->putBe32(allocator->capacity);
             stream->putBe64(allocator->total_bytes);
          };
      address_block_iter_func_t allocatorBlockSaver =
          [](void* context, struct address_block* block) {
             Stream* stream = reinterpret_cast<Stream*>(context);
             stream->putBe64(block->offset);
             stream->putBe64(block->size_available);
          };
      address_space_allocator_run(&addr_alloc, (void*)stream, allocatorSaver,
                                  allocatorBlockSaver);

      stream->putBe64(pageSize);
      stream->putBe64(totalSize);
      stream->putBe32(allocCount);

      return true;
   }

   bool load(Stream* stream) {
      clear();
      address_space_allocator_iter_func_t allocatorLoader =
          [](void* context, struct address_space_allocator* allocator) {
             Stream* stream = reinterpret_cast<Stream*>(context);
             allocator->size = stream->getBe32();
             allocator->capacity = stream->getBe32();
             allocator->total_bytes = stream->getBe64();
          };
      address_block_iter_func_t allocatorBlockLoader =
          [](void* context, struct address_block* block) {
             Stream* stream = reinterpret_cast<Stream*>(context);
             block->offset = stream->getBe64();
             block->size_available = stream->getBe64();
          };
      address_space_allocator_run(&addr_alloc, (void*)stream, allocatorLoader,
                                  allocatorBlockLoader);

      pageSize = stream->getBe64();
      totalSize = stream->getBe64();
      allocCount = stream->getBe32();

      return true;
   }

   bool postLoad(void* postLoadBuffer) {
      buffer = postLoadBuffer;
      startAddr = (uint64_t)(uintptr_t)postLoadBuffer;
      return true;
   }

   void rangeCheck(const char* task, void* ptr) {
      uint64_t addr = (uintptr_t)ptr;
      if (addr < startAddr || addr > endAddr) {
         mesa_loge(
            "FATAL in SubAllocator: Task:%s ptr '0x%llx' is out of range! "
            "Range:[0x%llx - 0x%llx]", task, addr, startAddr, endAddr);
      }
   }

   uint64_t getOffset(void* checkedPtr) {
      uint64_t addr = (uintptr_t)checkedPtr;
      return addr - startAddr;
   }

   bool free(void* ptr) {
      if (!ptr) return false;

      rangeCheck("free", ptr);
      if (EINVAL ==
          address_space_allocator_deallocate(&addr_alloc, getOffset(ptr))) {
         return false;
      }

      --allocCount;
      return true;
   }

   void freeAll() {
      address_space_allocator_reset(&addr_alloc);
      allocCount = 0;
   }

   void* alloc(size_t wantedSize) {
      if (wantedSize == 0) return nullptr;

      uint64_t wantedSize64 = (uint64_t)wantedSize;

      size_t toPageSize = pageSize * ((wantedSize + pageSize - 1) / pageSize);

      uint64_t offset =
          address_space_allocator_allocate(&addr_alloc, toPageSize);

      if (offset == ANDROID_EMU_ADDRESS_SPACE_BAD_OFFSET) {
         return nullptr;
      }

      ++allocCount;
      return (void*)(uintptr_t)(startAddr + offset);
   }

   bool empty() const { return allocCount == 0; }

   void* buffer;
   uint64_t totalSize;
   uint64_t pageSize;
   uint64_t startAddr;
   uint64_t endAddr;
   struct address_space_allocator addr_alloc;
   uint32_t allocCount = 0;
};

SubAllocator::SubAllocator(void* buffer, uint64_t totalSize, uint64_t pageSize)
    : mImpl(new SubAllocator::Impl(buffer, totalSize, pageSize)) {}

SubAllocator::~SubAllocator() { delete mImpl; }

// Snapshotting
bool SubAllocator::save(Stream* stream) { return mImpl->save(stream); }

bool SubAllocator::load(Stream* stream) { return mImpl->load(stream); }

bool SubAllocator::postLoad(void* postLoadBuffer) {
   return mImpl->postLoad(postLoadBuffer);
}

void* SubAllocator::alloc(size_t wantedSize) {
   return mImpl->alloc(wantedSize);
}

bool SubAllocator::free(void* ptr) { return mImpl->free(ptr); }

void SubAllocator::freeAll() { mImpl->freeAll(); }

uint64_t SubAllocator::getOffset(void* ptr) { return mImpl->getOffset(ptr); }

bool SubAllocator::empty() const { return mImpl->empty(); }

}  // namespace aemu
}  // namespace gfxstream
