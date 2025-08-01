/*
 * Copyright © 2022 Google, Inc.
 * SPDX-License-Identifier: MIT
 */

#ifndef FREEDRENO_COMMON_H_
#define FREEDRENO_COMMON_H_

#include "util/u_atomic.h"
#include "util/macros.h"

#ifdef __cplusplus

#include <tuple>

#define __FD_GPU_GENS A6XX, A7XX
#define FD_GENX(FUNC_NAME)                                                   \
   template <chip... CHIPs> constexpr auto FUNC_NAME##instantiate()          \
   {                                                                         \
      return std::tuple_cat(std::make_tuple(FUNC_NAME<CHIPs>)...);           \
   }                                                                         \
   static constexpr auto FUNC_NAME##tmpl __attribute__((used)) =             \
      FUNC_NAME##instantiate<__FD_GPU_GENS>();

#define FD_CALLX(info, thing)                                                \
   ({                                                                        \
      decltype(&thing<A6XX>) genX_thing;                                     \
      switch (info->chip) {                                                  \
      case 6:                                                                \
         genX_thing = &thing<A6XX>;                                          \
         break;                                                              \
      case 7:                                                                \
         genX_thing = &thing<A7XX>;                                          \
         break;                                                              \
      default:                                                               \
         UNREACHABLE("Unknown hardware generation");                         \
      }                                                                      \
      genX_thing;                                                            \
   })


template<typename E>
struct BitmaskEnum {
   E value;

   using underlying = typename std::underlying_type_t<E>;

#define FOREACH_TYPE(M, ...) \
   M(E,          ##__VA_ARGS__) \
   M(bool,       ##__VA_ARGS__) \
   M(uint8_t,    ##__VA_ARGS__) \
   M(int8_t,     ##__VA_ARGS__) \
   M(uint16_t,   ##__VA_ARGS__) \
   M(int16_t,    ##__VA_ARGS__) \
   M(uint32_t,   ##__VA_ARGS__) \
   M(int32_t,    ##__VA_ARGS__)

#define CONSTRUCTOR(T) BitmaskEnum(T value) :  value(static_cast<E>(value)) {}
   FOREACH_TYPE(CONSTRUCTOR)
#undef CONSTRUCTOR

#define CAST(T) inline operator T() const { return static_cast<T>(value); }
   FOREACH_TYPE(CAST)
#undef CAST

#define BOP(T, OP)                          \
   inline E operator OP(T rhs) const {      \
      return static_cast<E> (               \
         static_cast<underlying>(value) OP  \
         static_cast<underlying>(rhs)       \
      );                                    \
   }
   FOREACH_TYPE(BOP, |)
   FOREACH_TYPE(BOP, &)
#undef BOP

#define BOP(OP)                                                    \
   inline BitmaskEnum<E> operator OP(BitmaskEnum<E> rhs) const {   \
      return static_cast<E> (                                      \
         static_cast<underlying>(value) OP                         \
         static_cast<underlying>(rhs.value)                        \
      );                                                           \
   }
   BOP(|)
   BOP(&)
#undef BOP

#if defined(__GNUC__) && !defined(__clang)
/*
 * Silence:
 *
 *   ../src/freedreno/common/freedreno_common.h: In instantiation of 'E& BitmaskEnum<E>::operator|=(BitmaskEnum<E>::underlying) [with E = fd_dirty_3d_state; BitmaskEnum<E>::underlying = unsigned int]':
 *   ../src/gallium/drivers/freedreno/freedreno_context.h:620:16:   required from here
 *   ../src/freedreno/common/freedreno_common.h:68:39: error: dereferencing type-punned pointer will break strict-aliasing rules [-Werror=strict-aliasing]
 *      68 |         reinterpret_cast<underlying&>(value) OP static_cast<underlying>(rhs) ); \
 *         |                                       ^~~~~
 *
 * I cannot reproduce on gcc 12.2.1 or with clang 14.0.5 so I'm going to assume
 * this is a bug with gcc 10.x
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#define UOP(T, OP)                          \
   inline E& operator OP(T rhs) {           \
      return reinterpret_cast<E&>(          \
        reinterpret_cast<underlying&>(value) OP static_cast<underlying>(rhs) ); \
   }
   UOP(underlying, |=)
   UOP(underlying, &=)
#undef UOP

#if defined(__GNUC__) && !defined(__clang) && (__GNUC__ < 7)
#pragma GCC diagnostic pop
#endif

   inline E operator ~() const {
      static_assert(sizeof(E) == sizeof(BitmaskEnum<E>));
      return static_cast<E> (
            ~static_cast<underlying>(value)
      );
   }
#undef FOREACH_TYPE
};
#define BITMASK_ENUM(E) BitmaskEnum<E>
#else
#define BITMASK_ENUM(E) enum E
#endif

#ifdef __cplusplus
#  define EXTERNC extern "C"
#  define BEGINC EXTERNC {
#  define ENDC }
#else
#  define EXTERNC
#  define BEGINC
#  define ENDC
#endif

/* for conditionally setting boolean flag(s): */
#define COND(bool, val) ((bool) ? (val) : 0)

#define BIT(bit) BITFIELD64_BIT(bit)

/**
 * Helper for allocating sequence #s where zero is a non-valid seqno
 */
typedef struct {
   uint32_t counter;
} seqno_t;

static inline uint32_t
seqno_next(seqno_t *seq)
{
   uint32_t n;
   do {
      n = p_atomic_inc_return(&seq->counter);
   } while (n == 0);
   return n;
}

static inline uint16_t
seqno_next_u16(seqno_t *seq)
{
   uint16_t n;
   do {
      n = p_atomic_inc_return(&seq->counter);
   } while (n == 0);
   return n;
}

#endif /* FREEDRENO_COMMON_H_ */
