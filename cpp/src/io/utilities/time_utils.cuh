/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

// #include "column_buffer.hpp"
// #include <rmm/exec_policy.hpp>

namespace cudf {
namespace io {

/**
 * @brief Lookup table to compute power of ten
 */
static const __device__ __constant__ int32_t powers_of_ten[10] = {
  1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

struct get_period {
  template <typename T>
  constexpr int32_t operator()()
  {
    if constexpr (is_chrono<T>()) { return T::period::den; }
    CUDF_FAIL("Invalid, non chrono type");
  }
};

/**
 * @brief Function that translates cuDF time unit to clock frequency
 */
constexpr int32_t to_clockrate(type_id timestamp_type_id)
{
  return timestamp_type_id == type_id::EMPTY
           ? 0
           : type_dispatcher(data_type{timestamp_type_id}, get_period{});
}

/*namespace detail{
void print_stuff(column_buffer_base<cudf::io::detail::inline_column_buffer>& buffer,
rmm::cuda_stream_view stream) { thrust::for_each(rmm::exec_policy(stream),
       thrust::make_counting_iterator<size_type>(0),
       thrust::make_counting_iterator<size_type>(buffer.size),
       [d = (size_type *)buffer.data()] __device__(size_type o) { printf("offset %d: %d\n", o,
d[o]); });
}
void print_stuff(column_buffer_base<cudf::io::detail::gather_column_buffer>& buffer,
rmm::cuda_stream_view stream) {

}
}*/

}  // namespace io
}  // namespace cudf
