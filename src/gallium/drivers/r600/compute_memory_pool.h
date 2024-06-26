/*
 * Authors:
 *      Adam Rak <adam.rak@streamnovation.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef COMPUTE_MEMORY_POOL
#define COMPUTE_MEMORY_POOL

#include <stdlib.h>

#define ITEM_MAPPED_FOR_READING (1<<0)
#define ITEM_MAPPED_FOR_WRITING (1<<1)
#define ITEM_FOR_PROMOTING      (1<<2)
#define ITEM_FOR_DEMOTING       (1<<3)

#define POOL_FRAGMENTED (1<<0)

struct compute_memory_pool;

struct compute_memory_item
{
	int64_t id;		/**< ID of the memory chunk */

	uint32_t status;	/**< Will track the status of the item */

	/** Start pointer in dwords relative in the pool bo. If an item
	 * is unallocated, then this value must be -1 to indicate this. */
	int64_t start_in_dw;
	int64_t size_in_dw;	/**< Size of the chunk in dwords */

	/** Intermediate buffer associated with an item. It is used mainly for mapping
	 * items against it. They are listed in the pool's unallocated list */
	struct r600_resource *real_buffer;

	struct compute_memory_pool* pool;

	struct list_head link;
};

struct compute_memory_pool
{
	int64_t next_id;	/**< For generating unique IDs for memory chunks */
	int64_t size_in_dw;	/**< Size of the pool in dwords */

	struct r600_resource *bo;	/**< The pool buffer object resource */
	struct r600_screen *screen;

	uint32_t *shadow;	/**< host copy of the pool, used for growing the pool */

	uint32_t status;	/**< Status of the pool */

	/** Allocated memory items in the pool, they must be ordered by "start_in_dw" */
	struct list_head *item_list;

	/** Unallocated memory items, this list contains all the items that aren't
	 * yet in the pool */
	struct list_head *unallocated_list;
};


static inline int is_item_in_pool(struct compute_memory_item *item)
{
	return item->start_in_dw != -1;
}

static inline int is_item_user_ptr(struct compute_memory_item *item)
{
	assert(item->real_buffer);
	return item->real_buffer->b.is_user_ptr;
}

struct compute_memory_pool* compute_memory_pool_new(struct r600_screen *rscreen);

void compute_memory_pool_delete(struct compute_memory_pool* pool);

int compute_memory_finalize_pending(struct compute_memory_pool* pool,
	struct pipe_context * pipe);

void compute_memory_demote_item(struct compute_memory_pool *pool,
	struct compute_memory_item *item, struct pipe_context *pipe);

void compute_memory_free(struct compute_memory_pool* pool, int64_t id);

struct compute_memory_item* compute_memory_alloc(struct compute_memory_pool* pool,
	int64_t size_in_dw);

#endif
