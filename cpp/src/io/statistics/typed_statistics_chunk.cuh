/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file typed_statistics_chunk.cuh
 * @brief Templated wrapper to generalize statistics chunk reduction and aggregation
 * across different leaf column types
 */

#pragma once

#include "statistics.cuh"
#include "statistics_type_identification.cuh"
#include "temp_storage_wrapper.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <math_constants.h>

#include <thrust/extrema.h>

namespace cudf {
namespace io {

/**
 * @brief Class used to get reference to members of unions related to statistics calculations
 */
class union_member {
  template <typename U, typename V>
  using reference_type = std::conditional_t<std::is_const_v<U>, const V&, V&>;

 public:
  template <typename T, typename U>
  using type =
    std::conditional_t<std::is_same_v<std::remove_cv_t<T>, string_view>,
                       reference_type<U, string_stats>,
                       std::conditional_t<std::is_same_v<std::remove_cv_t<T>, byte_array_view>,
                                          reference_type<U, byte_array_stats>,
                                          reference_type<U, T>>>;

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_integral_v<T> and std::is_unsigned_v<T>, type<T, U>>
  get(U& val)
  {
    return val.u_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_integral_v<T> and std::is_signed_v<T>, type<T, U>> get(
    U& val)
  {
    return val.i_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_same_v<T, __int128_t>, type<T, U>> get(U& val)
  {
    return val.d128_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_floating_point_v<T>, type<T, U>> get(U& val)
  {
    return val.fp_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_same_v<T, string_view>, type<T, U>> get(U& val)
  {
    return val.str_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_same_v<T, byte_array_view>, type<T, U>> get(U& val)
  {
    return val.byte_val;
  }
};

/**
 * @brief Templated structure used for merging and gathering of statistics chunks
 *
 * This uses the reduce function to compute the minimum, maximum and aggregate
 * values simultaneously.
 *
 * @tparam T The input type associated with the chunk
 * @tparam is_aggregation_supported Set to true if input type is meant to be aggregated
 */
template <typename T, bool is_aggregation_supported>
struct typed_statistics_chunk {
};

template <typename T>
struct typed_statistics_chunk<T, true> {
  using E = typename detail::extrema_type<T>::type;
  using A = typename detail::aggregation_type<T>::type;

  uint32_t non_nulls{0};   //!< number of non-null values in chunk
  uint32_t null_count{0};  //!< number of null values in chunk

  E minimum_value;
  E maximum_value;
  A aggregate;

  uint8_t has_minmax{false};  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum{false};     //!< Nonzero if sum is valid

  __device__ typed_statistics_chunk()
    : minimum_value(detail::minimum_identity<E>()),
      maximum_value(detail::maximum_identity<E>()),
      aggregate(0)
  {
  }

  __device__ void reduce(const T& elem)
  {
    non_nulls++;
    minimum_value = thrust::min<E>(minimum_value, detail::extrema_type<T>::convert(elem));
    maximum_value = thrust::max<E>(maximum_value, detail::extrema_type<T>::convert(elem));
    aggregate += detail::aggregation_type<T>::convert(elem);
    has_minmax = true;
  }

  __device__ void reduce(const statistics_chunk& chunk)
  {
    if (chunk.has_minmax) {
      minimum_value = thrust::min<E>(minimum_value, union_member::get<E>(chunk.min_value));
      maximum_value = thrust::max<E>(maximum_value, union_member::get<E>(chunk.max_value));
    }
    if (chunk.has_sum) { aggregate += union_member::get<A>(chunk.sum); }
    non_nulls += chunk.non_nulls;
    null_count += chunk.null_count;
  }

  struct min_functor {
    [[nodiscard]] __device__ __forceinline__ E operator()(const E& a, const E& b) const
    {
      return thrust::min<E>(a, b);
    }
  };

  struct max_functor {
    [[nodiscard]] __device__ __forceinline__ E operator()(const E& a, const E& b) const
    {
      return thrust::min<E>(a, b);
    }
  };

  struct less_or_equal_functor {
    template <typename R>
    [[nodiscard]] __device__ __forceinline__ bool operator()(const R& lhs, const R& rhs) const
    {
      return lhs <= rhs;
    }
  };
};

template <typename T>
struct typed_statistics_chunk<T, false> {
  using E = typename detail::extrema_type<T>::type;

  uint32_t non_nulls{0};   //!< number of non-null values in chunk
  uint32_t null_count{0};  //!< number of null values in chunk

  E minimum_value;
  E maximum_value;

  uint8_t has_minmax{false};  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum{false};     //!< Nonzero if sum is valid

  __device__ typed_statistics_chunk()
    : minimum_value(detail::minimum_identity<E>()), maximum_value(detail::maximum_identity<E>())
  {
  }

  struct comparison_base {
    /**
     * @brief Comparing byte_array_views. Each byte in the array is compared.
     *
     * @param el0 byte_array_view to compare with el1.
     * @param el1 byte_array_view to compare with el0
     * @return 0  If they compare equal.
     *         <0 Either the value of the first byte of el0 that does not match is greater in el1 or
     * all compared bytes match but el0 is shorter. >0 Either the value of the first byte of el0
     * that does not match is lower in el1 or all compared bytes match but el0 is longer.
     */
    [[nodiscard]] __device__ inline int32_t compare(byte_array_view const& el0,
                                                    byte_array_view const& el1) const
    {
      auto const len0  = el0.size_bytes();
      auto const len1  = el1.size_bytes();
      auto const* ptr0 = el0.data();
      auto const* ptr1 = el1.data();
      if ((ptr0 == ptr1) && (len0 == len1)) { return 0; }
      // if el0 is max, it is greater than el1
      if (ptr0 == nullptr && len0 == std::numeric_limits<byte_array_view::size_type>::max()) {
        return 1;
      }
      // if el1 is max, it is greater than el0
      if (ptr1 == nullptr && len1 == std::numeric_limits<byte_array_view::size_type>::max()) {
        return -1;
      }
      std::size_t idx = 0;
      for (; (idx < len0) && (idx < len1); ++idx) {
        if (ptr0[idx] != ptr1[idx]) {
          return static_cast<int32_t>(ptr0[idx]) - static_cast<int32_t>(ptr1[idx]);
        }
      }
      // if the el1 ran out of data, it is less than el0
      if (idx < len0) return 1;
      // if el0 ran out of data first, el0 is less than el1
      if (idx < len1) return -1;
      return 0;
    }
  };

  struct min_functor : comparison_base {
    template <typename R, std::enable_if_t<!std::is_same_v<R, byte_array_view>>* = nullptr>
    [[nodiscard]] __device__ __forceinline__ R operator()(const R& a, const R& b) const
    {
      return thrust::min<R>(a, b);
    }

    template <typename R, std::enable_if_t<std::is_same_v<R, byte_array_view>>* = nullptr>
    [[nodiscard]] __device__ __forceinline__ R operator()(const R& a, const R& b) const
    {
      return compare(a, b) < 0 ? a : b;
    }
  };

  struct max_functor : comparison_base {
    template <typename R, std::enable_if_t<!std::is_same_v<R, byte_array_view>>* = nullptr>
    [[nodiscard]] __device__ __forceinline__ R operator()(const R& a, const R& b) const
    {
      return thrust::max<R>(a, b);
    }

    template <typename R, std::enable_if_t<std::is_same_v<R, byte_array_view>>* = nullptr>
    [[nodiscard]] __device__ __forceinline__ R operator()(const R& a, const R& b) const
    {
      return compare(a, b) < 0 ? b : a;
    }
  };

  struct less_or_equal_functor : comparison_base {
    template <typename R, std::enable_if_t<!std::is_same_v<R, byte_array_view>>* = nullptr>
    [[nodiscard]] __device__ __forceinline__ bool operator()(const E& lhs, const E& rhs) const
    {
      return lhs <= rhs;
    }

    template <typename R, std::enable_if_t<std::is_same_v<R, byte_array_view>>* = nullptr>
    [[nodiscard]] __device__ __forceinline__ bool operator()(const E& lhs, const E& rhs) const
    {
      return compare(lhs, rhs) <= 0;
    }
  };

  __device__ void reduce(const T& elem)
  {
    non_nulls++;
    minimum_value = min_functor()(minimum_value, detail::extrema_type<T>::convert(elem));
    maximum_value = max_functor()(maximum_value, detail::extrema_type<T>::convert(elem));
    has_minmax    = true;
  }

  __device__ void reduce(const statistics_chunk& chunk)
  {
    if (chunk.has_minmax) {
      minimum_value =
        min_functor().template operator()<E>(minimum_value, union_member::get<E>(chunk.min_value));
      maximum_value =
        max_functor().template operator()<E>(maximum_value, union_member::get<E>(chunk.max_value));
    }
    non_nulls += chunk.non_nulls;
    null_count += chunk.null_count;
  }
};

/**
 * @brief Function to reduce members of a typed_statistics_chunk across a thread block
 *
 * @tparam T Type associated with typed_statistics_chunk
 * @tparam block_size Dimension of the thread block
 * @param chunk The input typed_statistics_chunk
 * @param storage Temporary storage to be used by cub calls
 */
template <typename T, bool include_aggregate, int block_size>
__inline__ __device__ typed_statistics_chunk<T, include_aggregate> block_reduce(
  typed_statistics_chunk<T, include_aggregate>& chunk, detail::storage_wrapper<block_size>& storage)
{
  typed_statistics_chunk<T, include_aggregate> output_chunk = chunk;

  using E              = typename detail::extrema_type<T>::type;
  using extrema_reduce = cub::BlockReduce<E, block_size>;
  using count_reduce   = cub::BlockReduce<uint32_t, block_size>;
  output_chunk.minimum_value =
    extrema_reduce(storage.template get<E>())
      .Reduce(output_chunk.minimum_value,
              typed_statistics_chunk<T, include_aggregate>::min_functor());
  __syncthreads();
  output_chunk.maximum_value =
    extrema_reduce(storage.template get<E>())
      .Reduce(output_chunk.maximum_value,
              typed_statistics_chunk<T, include_aggregate>::max_functor());
  __syncthreads();
  output_chunk.non_nulls =
    count_reduce(storage.template get<uint32_t>()).Sum(output_chunk.non_nulls);
  __syncthreads();
  output_chunk.null_count =
    count_reduce(storage.template get<uint32_t>()).Sum(output_chunk.null_count);
  __syncthreads();
  output_chunk.has_minmax = __syncthreads_or(output_chunk.has_minmax);

  // FIXME : Is another syncthreads needed here?
  if constexpr (include_aggregate) {
    if (output_chunk.has_minmax) {
      using A                = typename detail::aggregation_type<T>::type;
      using aggregate_reduce = cub::BlockReduce<A, block_size>;
      output_chunk.aggregate =
        aggregate_reduce(storage.template get<A>()).Sum(output_chunk.aggregate);
    }
  }
  return output_chunk;
}

/**
 * @brief Function to convert typed_statistics_chunk into statistics_chunk
 *
 * @tparam T Type associated with typed_statistics_chunk
 * @param chunk The input typed_statistics_chunk
 */
template <typename T, bool include_aggregate>
__inline__ __device__ statistics_chunk
get_untyped_chunk(const typed_statistics_chunk<T, include_aggregate>& chunk)
{
  using E = typename detail::extrema_type<T>::type;
  statistics_chunk stat{};
  stat.non_nulls  = chunk.non_nulls;
  stat.null_count = chunk.null_count;
  stat.has_minmax = chunk.has_minmax;
  stat.has_sum    = [&]() {
    if (!chunk.has_minmax) return false;
    // invalidate the sum if overflow or underflow is possible
    if constexpr (std::is_floating_point_v<E> or std::is_integral_v<E>) {
      return std::numeric_limits<E>::max() / chunk.non_nulls >=
               static_cast<E>(chunk.maximum_value) and
             std::numeric_limits<E>::lowest() / chunk.non_nulls <=
               static_cast<E>(chunk.minimum_value);
    }
    return true;
  }();
  if (chunk.has_minmax) {
    if constexpr (std::is_floating_point_v<E>) {
      union_member::get<E>(stat.min_value) =
        (chunk.minimum_value != 0.0) ? chunk.minimum_value : CUDART_NEG_ZERO;
      union_member::get<E>(stat.max_value) =
        (chunk.maximum_value != 0.0) ? chunk.maximum_value : CUDART_ZERO;
    } else {
      union_member::get<E>(stat.min_value) = chunk.minimum_value;
      union_member::get<E>(stat.max_value) = chunk.maximum_value;
    }
    if constexpr (include_aggregate) {
      using A                        = typename detail::aggregation_type<T>::type;
      union_member::get<A>(stat.sum) = chunk.aggregate;
    }
  }
  return stat;
}

}  // namespace io
}  // namespace cudf
