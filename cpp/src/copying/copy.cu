/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include "cudf/lists/lists_column_view.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"

namespace cudf {
namespace detail {
namespace {

template <typename T, typename Enable = void>
struct copy_if_else_functor_impl {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type for copy_if_else.");
  }
};

template <typename T>
struct copy_if_else_functor_impl<T, std::enable_if_t<is_rep_layout_compatible<T>()>> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto real_lhs = [lhs]() {
      if constexpr(std::is_same<Left, column_view>::value) {
        return *column_device_view::create(lhs);
      } else {
        return lhs;
      }
    }();
    auto real_rhs = [rhs]() {
      if constexpr(std::is_same<Right, column_view>::value) {
        return *column_device_view::create(rhs);
      } else {
        return rhs;
      }
    }();
    if (left_nullable) {
      if (right_nullable) {
        auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(real_lhs);
        auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(real_rhs);
        return detail::copy_if_else(
          true, lhs_iter, lhs_iter + size, rhs_iter, filter, real_lhs.type(), stream, mr);
      }
      auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(real_lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(real_rhs);
      return detail::copy_if_else(
        true, lhs_iter, lhs_iter + size, rhs_iter, filter, real_lhs.type(), stream, mr);
    }
    if (right_nullable) {
      auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(real_lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(real_rhs);
      return detail::copy_if_else(
        true, lhs_iter, lhs_iter + size, rhs_iter, filter, real_lhs.type(), stream, mr);
    }
    auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(real_lhs);
    auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(real_rhs);
    return detail::copy_if_else(
      false, lhs_iter, lhs_iter + size, rhs_iter, filter, real_lhs.type(), stream, mr);
  }
};

/**
 * @brief Specialization of copy_if_else_functor for string_views.
 */
template <>
struct copy_if_else_functor_impl<string_view> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    using T = string_view;

    auto real_lhs = [lhs]() {
      if constexpr(std::is_same<Left, column_view>::value) {
        return *column_device_view::create(lhs);
      } else {
        return lhs;
      }
    }();

    auto real_rhs = [rhs]() {
      if constexpr(std::is_same<Right, column_view>::value) {
        return *column_device_view::create(rhs);
      } else {
        return rhs;
      }
    }();

    if (left_nullable) {
      if (right_nullable) {
        auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(real_lhs);
        auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(real_rhs);
        return strings::detail::copy_if_else(
          lhs_iter, lhs_iter + size, rhs_iter, filter, stream, mr);
      }
      auto lhs_iter = cudf::detail::make_pair_iterator<T, true>(real_lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(real_rhs);
      return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, stream, mr);
    }
    if (right_nullable) {
      auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(real_lhs);
      auto rhs_iter = cudf::detail::make_pair_iterator<T, true>(real_rhs);
      return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, stream, mr);
    }
    auto lhs_iter = cudf::detail::make_pair_iterator<T, false>(real_lhs);
    auto rhs_iter = cudf::detail::make_pair_iterator<T, false>(real_rhs);
    return strings::detail::copy_if_else(lhs_iter, lhs_iter + size, rhs_iter, filter, stream, mr);
  }
};

/**
 * @brief Specialization of copy_if_else_functor for list_views.
 */
template <>
struct copy_if_else_functor_impl<list_view> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    rmm::device_uvector<bool> mask(size, stream);
    auto counting = thrust::make_counting_iterator(0);
    thrust::transform(counting, counting + size, mask.begin(), filter);

    auto real_lhs = [lhs]() {
      if constexpr(std::is_same<Left, std::unique_ptr<column_view>>::value) {
        return lhs.release();
      } else {
        return reinterpret_cast<list_scalar>(lhs).view();
      }
    }();

    auto real_rhs = [rhs]() {
      if constexpr(std::is_same<Right, std::unique_ptr<column_view>>::value) {
        return rhs.release();
      } else {
        return reinterpret_cast<list_scalar>(rhs).view();
      }
    }();

    // gather lhs column with gather map
    auto lhs_gathered =
        detail::gather(
          table_view({real_lhs}), mask.begin(), mask.end(), cudf::out_of_bounds_policy::NULLIFY, stream, mr);

    // build gather map for rhs to compress entries
    auto rhs_gather_map =
      thrust::make_transform_iterator(mask.begin(), [] __device__(auto i) { return !i; });

    // gather rhs
    auto rhs_gathered = detail::gather(table_view({real_rhs}),
                                       rhs_gather_map,
                                       rhs_gather_map + size,
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       mr);

    // scatter rhs into lhs
    detail::boolean_mask_scatter(
      rhs_gathered->view(), lhs_gathered->view(), column_view{data_type{type_id::BOOL8}, size, mask.data()}, stream, mr);
  }
};

template <>
struct copy_if_else_functor_impl<struct_view> {
  template <typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("copy_if_else not supported for struct_view yet");
  }
};

/**
 * @brief Functor called by the `type_dispatcher` to invoke copy_if_else on combinations
 *        of column_view and scalar
 */
struct copy_if_else_functor {
  template <typename T, typename Left, typename Right, typename Filter>
  std::unique_ptr<column> operator()(Left const& lhs,
                                     Right const& rhs,
                                     size_type size,
                                     bool left_nullable,
                                     bool right_nullable,
                                     Filter filter,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    copy_if_else_functor_impl<T> copier{};
    return copier(lhs, rhs, size, left_nullable, right_nullable, filter, stream, mr);
  }
};

// wrap up boolean_mask into a filter lambda
template <typename Left, typename Right>
std::unique_ptr<column> copy_if_else(Left const& lhs,
                                     Right const& rhs,
                                     bool left_nullable,
                                     bool right_nullable,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lhs.type() == rhs.type(), "Both inputs must be of the same type");
  CUDF_EXPECTS(boolean_mask.type() == data_type(type_id::BOOL8),
               "Boolean mask column must be of type type_id::BOOL8");

  if (boolean_mask.is_empty()) { return cudf::make_empty_column(lhs.type()); }

  auto bool_mask_device_p             = column_device_view::create(boolean_mask);
  column_device_view bool_mask_device = *bool_mask_device_p;

  if (boolean_mask.has_nulls()) {
    auto filter = [bool_mask_device] __device__(cudf::size_type i) {
      return bool_mask_device.is_valid_nocheck(i) and bool_mask_device.element<bool>(i);
    };
    return cudf::type_dispatcher<dispatch_storage_type>(lhs.type(),
                                                        copy_if_else_functor{},
                                                        lhs,
                                                        rhs,
                                                        boolean_mask.size(),
                                                        left_nullable,
                                                        right_nullable,
                                                        filter,
                                                        stream,
                                                        mr);
  } else {
    auto filter = [bool_mask_device] __device__(cudf::size_type i) {
      return bool_mask_device.element<bool>(i);
    };
    return cudf::type_dispatcher<dispatch_storage_type>(lhs.type(),
                                                        copy_if_else_functor{},
                                                        lhs,
                                                        rhs,
                                                        boolean_mask.size(),
                                                        left_nullable,
                                                        right_nullable,
                                                        filter,
                                                        stream,
                                                        mr);
  }
}
/*
std::unique_ptr<column> copy_if_else(column_view const& lhs, column_view const& rhs, column_view const& boolean_mask,
rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  // gather lhs column with gather map
  auto lhs_gathered =
      detail::gather(
        table_view({lhs}), boolean_mask.begin<bool>(), boolean_mask.end<bool>(), cudf::out_of_bounds_policy::NULLIFY, stream, mr);

  // build gather map for rhs to compress entries
  auto rhs_gather_map =
    thrust::make_transform_iterator(boolean_mask.begin<bool>(), [] __device__(auto i) { return !i; });

  // gather rhs
  auto rhs_gathered = detail::gather(table_view({rhs}),
                                     rhs_gather_map,
                                     rhs_gather_map + rhs.size(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);

  // scatter rhs into lhs
  return detail::boolean_mask_scatter(
    rhs_gathered, lhs_gathered, boolean_mask, mr);
}
*/
};  // namespace

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == lhs.size(),
               "Boolean mask column must be the same size as lhs and rhs columns");
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Both columns must be of the size");

  return copy_if_else(lhs,
                    rhs,
                    lhs.has_nulls(),
                    rhs.has_nulls(),
                    boolean_mask,
                    stream,
                    mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == rhs.size(),
               "Boolean mask column must be the same size as rhs column");
              return copy_if_else(lhs,
                      rhs,
                      !lhs.is_valid(),
                      rhs.has_nulls(),
                      boolean_mask,
                      stream,
                      mr);
}

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.size() == lhs.size(),
               "Boolean mask column must be the same size as lhs column");

  return copy_if_else(lhs,
                      rhs,
                      lhs.has_nulls(),
                      !rhs.is_valid(),
                      boolean_mask,
                      stream,
                      mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  return copy_if_else(lhs, rhs, !lhs.is_valid(), !rhs.is_valid(), boolean_mask, stream, mr);
}

};  // namespace detail

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     column_view const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> copy_if_else(column_view const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> copy_if_else(scalar const& lhs,
                                     scalar const& rhs,
                                     column_view const& boolean_mask,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_if_else(lhs, rhs, boolean_mask, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
