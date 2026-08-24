/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef QUEUE_H
#define QUEUE_H
/**
  @file queue.h
  @brief This file contains types and declarations associated with the Queue
  Services.

*/
/*===========================================================================
NOTE: The @brief description and any detailed descriptions above do not appear
      in the PDF.

      The Utility_Services_API_mainpage.dox file contains all file/group
      descriptions that are in the output PDF generated using Doxygen and
      Latex. To edit or update any of the file/group text in the PDF, edit
      the Utility_Services_API_mainpage.dox file or contact Tech Pubs.

      The above description for this file is part of the "utils_queue"
      group description in the Utility_Services_API_mainpage.dox file.
===========================================================================*/
/*===========================================================================

            Q U E U E    S E R V I C E S    H E A D E R    F I L E

DESCRIPTION
 This file contains types and declarations associated with the Queue
 Services.

===========================================================================*/

/*===========================================================================

                     INCLUDE FILES FOR MODULE

===========================================================================*/
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FEATURE_Q_NO_SELF_QPTR

// TODO: err services? #include "err.h"
#ifndef ERR_FATAL
/* TODO: get real error fatal */
#define ERR_FATAL(str, a, b, c)                                                \
  {                                                                            \
    while (1) {                                                                \
    }                                                                          \
  }
#endif

/* Common Helper Macros for building error messages */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define AT __FILENAME__ ":" TOSTRING(__LINE__)

/* ************************************************************************
 * QUEUE_VERIFY macros
 * *********************************************************************** */

/*
 * QUEUE_VERIFY( expr )
 *
 * If the logical expression is not true, raise an ERR_FATAL.
 *
 * This is the basic macro that the other QUEUE_VERIFY macros extend.
 * The verbosity of the macro may be changed by using one of the below FEATUREs
 */

#if defined(FEATURE_QUEUE_VERIFY_VERBOSE)
#define VERIFY_MESSAGE(x) "Assertion failure: ( " x " ) " AT
#elif defined(FEATURE_QUEUE_VERIFY_AT)
#define VERIFY_MESSAGE(x) AT
#else
#define VERIFY_MESSAGE(x) 0
#endif

#ifndef QUEUE_VERIFY
#define QUEUE_VERIFY(x)                                                        \
  do {                                                                         \
    if (0 == (x)) {                                                            \
      ERR_FATAL(VERIFY_MESSAGE(#x), 0, 0, 0);                                  \
    }                                                                          \
  } while (0)
#endif

/*
 * QUEUE_VERIFY_PTR( ptr )
 *
 * Verify the ptr is not NULL
 */
#define QUEUE_VERIFY_PTR(ptr) QUEUE_VERIFY(NULL != (void *)(ptr))

/*===========================================================================

                        DATA DECLARATIONS

===========================================================================*/
/** @addtogroup utils_queue
@{ */

/** @brief Used as a link field with variables, which allows the variables
  to be moved on and off the queues. This is the principal structure of Queue
  Utility Services. Note that Users must not directly access the fields of
  a link; only Queue Utility Services is to access the fields.

  This structure is also used in the definition of the queues themselves.
*/
typedef struct q_link_struct {
  struct q_link_struct *next_ptr;
  /**< Pointer to the next link in the list. If NULL, there
  is no next link. */
  struct q_link_struct *prev_ptr;
  /**< Pointer to the previous link in the list. If NULL, there
  is no previous link. */
} q_link_type;

/*These are HTORPCMETA comments, Do not delete this.*/
/*~ FIELD q_link_struct.next_ptr VOID */
/*~ FIELD q_link_struct.prev_ptr VOID */

/** Used instead of q_link_type when the queue items are linked in
  a single link list. This avoids the overhead of traversing
  the entire link list when queueing at the end of the link list. The
  q_link_struct structure is to be accessed only by Queue Utility Services.
*/
typedef q_link_type q_head_link_type;

/** @brief Used by Queue Utility Services to represent a queue. Note that
  users must not directly access the fields of a queue; only Queue Utility
  Services is to access the fields.
*/
typedef struct q_struct {
  q_head_link_type link; /**< Used for linking the items into a queue. */

  int cnt; /**< Keeps track of the number of items enqueued.
                This field is not necessary for normal
                operations and is used for debugging
                purposes. */

  void *mutex;

} q_type;

/** @brief Used for generic items, which must have q_link_type as the
  first element.

  This structure allows the linear search function to traverse the list
  without having any information about the elements.
*/
typedef struct {
  q_link_type link; /**< Used for linking the items into a queue. */
} q_generic_item_type;

/** Used by the searching functions to determine if an item is in
  the queue. It returns nonzero if the element is to be operated on;
  otherwise, it returns 0.

  For linear searching, the operation is to return a pointer to the
  item and terminate the search.

  For linear deleting, the operation is to remove the item from the
  queue and continue the traversal.
*/
typedef int (*q_compare_func_type)(void *item_ptr, void *compare_val);

/** Used by the linear delete function to perform an action on an item
  that is being deleted from the list AFTER the item is deleted. To perform
  an action BEFORE the item is deleted, users can call the action
  function directly in the compare function callback.
*/
typedef void (*q_action_func_type)(void *item_ptr, void *param);

/*===========================================================================

                             Macro Definitions

===========================================================================*/

/** @cond
 */
#define Q_XCEPT_Q_INIT(q_ptr)
#define Q_XCEPT_Q_LINK(q_ptr, link_ptr)
#define Q_XCEPT_Q_PUT(q_ptr, link_ptr)
#define Q_XCEPT_Q_GET(q_ptr)
#define Q_XCEPT_Q_LAST_GET(q_ptr)
#define Q_XCEPT_Q_CNT(q_ptr)
#define Q_XCEPT_Q_CHECK(q_ptr)
#define Q_XCEPT_Q_LAST_CHECK(q_ptr)
#define Q_XCEPT_Q_NEXT(q_ptr, link_ptr)
#define Q_XCEPT_Q_INSERT(q_ptr, q_insert_ptr, q_item_ptr)

#define Q_XCEPT_Q_DELETE(q_ptr, q_delete_ptr)
/** @endcond */

/**
  Determines if the item passed in is in a queue.

  @param[in] q_link_ptr Pointer to the item to be tested.

  @return
  TRUE -- Item is already in a queue. \n
  FALSE -- Item is not in a queue.

  @dependencies
  None.
*/
#define Q_ALREADY_QUEUED(q_link_ptr) ((q_link_ptr)->next_ptr != NULL)

/*===========================================================================

                            Function Declarations

===========================================================================*/
#ifdef __cplusplus
extern "C" {
#endif

/**
  Initializes a specified queue. This function must be called for each
  queue before using the queue with other Queue Utility Services.

  @param[in] q_ptr Pointer to the queue to initialize.

  @return
  A pointer to the initialized queue.

  @dependencies
  None.
*/
q_type *q_init(q_type *q_ptr);

/**
  Initializes a specified link. This function must be called for each
  link before using the link with other Queue Utility Services.

  @param[in] item_ptr Item to be initialized.
  @param[in] link_ptr Link to be initialized.

  @return
  A pointer to the initialized link.

  @dependencies
  None.
*/
q_link_type *q_link(void *item_ptr, q_link_type *link_ptr);

/**
  Enqueues an item to a specified queue using a specific link.

  @param[in] q_ptr    Pointer to the item to be enqueued.
  @param[in] link_ptr Pointer to the link where the item is to be placed.

  @return
  None.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init(). \n
  The specified link field of the item must have been initialized previously
  via a call to q_link().

  @sideeffects
  The specified item is placed at the tail of a specific queue.
*/
void q_put(q_type *q_ptr, q_link_type *link_ptr);

/**
  Enqueues all items from a specified (source) queue to a
  specific (target) queue.

  @param[in] target_q_ptr Pointer to the target queue.
  @param[in] source_q_ptr Pointer to the source queue.

  @return
  None.

  @dependencies
  The specified queues must have been initialized previously via a call
  to q_init().

  @sideeffects
  The items in the specified source queue are placed at the tail of a
  specific target queue.
*/
void q_m_put(q_type *target_q_ptr, q_type *source_q_ptr);

/**
  Removes an item from the head of a specified queue.

  @param[in] q_ptr Pointer to the queue.

  @return
  Returns a pointer to the dequeued item, or NULL if the specified queue
  is empty.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
void *q_get(q_type *q_ptr);

/**
  Removes an item from the tail of a specified queue.

  @param[in] q_ptr Pointer to the queue.

  @return
  Returns a pointer to the dequeued item, or NULL if the specified queue
  is empty.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
void *q_last_get(q_type *q_ptr);

/**
  Returns the number of items currently queued on a specified queue.

  @param[in] q_ptr Pointer to the queue.

  @return
  The number of items.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
int q_cnt(q_type *q_ptr);

/**
  Returns a pointer to the data block at the head of the queue. The data
  block is not removed from the queue.

  @param[in] q_ptr Pointer to the queue.

  @return
  Returns a pointer to the queue item, or NULL if the specified queue
  is empty.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
void *q_check(q_type *q_ptr);

/**
  Returns a pointer to the data block at the tail of the queue. The data
  block is not removed from the queue.

  @param[in] q_ptr Pointer to the queue.

  @return
  Returns a pointer to the queue item, or NULL if the specified queue
  is empty.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
void *q_last_check(q_type *q_ptr);

/**
  Returns a pointer to the next item on the queue.

  @param[in] q_ptr    Pointer to the queue.
  @param[in] link_ptr Pointer to the link on the queue.

  @return
  Returns a pointer to the next item on the queue, or NULL if the end of
  the queue is reached.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
void *q_next(q_type *q_ptr, q_link_type *link_ptr);

/**
  Returns a pointer to the previous item on the queue.

  @param[in] q_ptr    Pointer to the queue.
  @param[in] link_ptr Pointer to the link on the queue.

  @return
  Returns pointer to the previous item on the queue.
  If the head of the queue is reached, then NULL is returned.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
void *q_prev(q_type *q_ptr, q_link_type *q_link_ptr);

/**
  Inserts an item before a specified item on the queue.

  @param[in] q_ptr        Pointer to the queue.
  @param[in] q_insert_ptr Pointer to where the item is to be inserted.
  @param[in] q_item_ptr   Pointer to the item to insert.

  @return
  None.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
void q_insert(
#ifdef FEATURE_Q_NO_SELF_QPTR
    q_type *q_ptr,
#endif
    q_link_type *q_insert_ptr, q_link_type *q_item_ptr);

/**
  Deletes an item from a specified queue.

  @param[in] q_ptr        Pointer to the queue.
  @param[in] q_delete_ptr Pointer to the item to be deleted.

  @return
  None.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
void q_delete(q_type *q_ptr, q_link_type *q_delete_ptr);

/**
  Removes an item from a specified queue.

  @param[in] q_ptr        Pointer to the queue.
  @param[in] q_delete_ptr Pointer to the item to be removed.

  @return
  TRUE -- Item was found and removed from the queue. \n
  FALSE -- Item was not found in the queue.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().
*/
bool q_delete_ext(q_type *q_ptr, q_link_type *q_delete_ptr);

/**
  Given a comparison function, this function traverses the elements in
  a queue, calls the compare function, and returns a pointer to the
  current element being compared if the user-passed compare function
  returns nonzero.

  The user compare function is to return 0 if the current element
  is not pertinent to the compare function.

  @param[in] q_ptr        Pointer to the queue to be traversed.
  @param[in] compare_func Comparison function to be used.
  @param[in] compare_val  Pointer to the value against which to compare.

  @return
  A pointer to the element.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init(). \n
  The user's queue elements must have q_link_type as the first element
  of the queued structure.
*/
void *q_linear_search(q_type *q_ptr, q_compare_func_type compare_func,
                      void *compare_val);

/**
  Given a comparison function, this function traverses the elements in
  a queue, calls the compare function, and returns a pointer to the
  current element being compared if the user-passed compare function
  returns nonzero. In addition, the item is removed from the queue.

  The user compare function is to return 0 if the current element
  is not pertinent to the compare function.

  @param[in] q_ptr        Pointer to the queue to be traversed.
  @param[in] compare_func Comparison function to be used.
  @param[in] compare_val  Pointer to the value against which to compare.
  @param[in] action_func  Action function to call if the item is found;
                          optional.
  @param[in] param        Parameter to pass to the action function if the
                          item is found; not required if there is no action
                          function.

  @return
  Returns a pointer to the element, or NULL if the element was not found.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init().\n
  The user's queue elements must have q_link_type as the first element
  of the queued structure.
*/
void *q_linear_delete_new(q_type *q_ptr, q_compare_func_type compare_func,
                          void *compare_val, q_action_func_type action_func,
                          void *param);

/**
  Given a comparison function, this function traverses the elements in
  a queue, calls the compare function, and returns a pointer to the
  current element being compared if the user-passed compare function
  returns nonzero. In addition, the item will be removed from the queue.

  The user compare function is to return 0 if the current element
  is not pertinent to the compare function.

  @note1hang The user's compare function will be passed NULL for the compare
  value.

  @param[in] q_ptr        Pointer to the queue to be traversed.
  @param[in] compare_func Comparison function to be used.
  @param[in] action_func  Action function to call if the item is found;
                          optional.
  @param[in] param        Parameter to pass to the action function if the
                          item is found; not required if there is no action
                          function.

  @return
  Returns a pointer to the element, or NULL if the element was not found.

  @dependencies
  The specified queue must have been initialized previously via a call
  to q_init(). \n
  The user's queue elements must have q_link_type as the first element
  of the queued structure.
*/
void q_linear_delete(q_type *q_ptr, q_compare_func_type compare_func,
                     void *param, q_action_func_type action_func);

/**
  Destroys a specified queue. This function is to be called if the queue
  is no longer required.

  @param[in] q_ptr Pointer to the queue to be destroyed.

  @return
  None.

  @dependencies
  None.

  @sideeffects
  Elements in the queue will not be accessible. To avoid leaks, it is user's
  responsibility to deallocate the memory allocated for the queue and its
  elements.
*/
void q_destroy(q_type *q_ptr);

#ifdef __cplusplus
}
#endif

/** @} */ /* end_addtogroup utils_queue */
#endif    /* QUEUE_H */
