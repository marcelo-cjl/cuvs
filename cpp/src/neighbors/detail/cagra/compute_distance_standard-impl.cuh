/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_distance_standard.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {
namespace {
template <typename DATA_T, typename DISTANCE_T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(DATA_T a, DATA_T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::L2Expanded, DISTANCE_T>
{
  DISTANCE_T diff = a - b;
  return diff * diff;
}

template <typename DATA_T, typename DISTANCE_T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(DATA_T a, DATA_T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::InnerProduct ||
                        Metric == cuvs::distance::DistanceType::CosineExpanded,
                      DISTANCE_T>
{
  return -static_cast<DISTANCE_T>(a) * static_cast<DISTANCE_T>(b);
}

template <typename DATA_T, typename DISTANCE_T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(DATA_T a, DATA_T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::BitwiseHamming &&
                        std::is_integral_v<DATA_T>,
                      DISTANCE_T>
{
  // mask the result of xor for the integer promotion
  const auto v = (a ^ b) & 0xffu;
  return __popc(v);
}

template <typename DATA_T>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto is_dp4a_data_type()
{
  return std::is_same_v<DATA_T, int8_t> || std::is_same_v<DATA_T, uint8_t>;
}

template <typename DATA_T>
RAFT_DEVICE_INLINE_FUNCTION auto pack_dp4a_byte(DATA_T v, uint32_t byte_idx) -> uint32_t
{
  return static_cast<uint32_t>(static_cast<uint8_t>(v)) << (8 * byte_idx);
}

template <typename DATA_T>
RAFT_DEVICE_INLINE_FUNCTION auto dp4a_dot(uint32_t a, uint32_t b) -> int32_t
{
  if constexpr (std::is_same_v<DATA_T, int8_t>) {
    return __dp4a(static_cast<int32_t>(a), static_cast<int32_t>(b), int32_t{0});
  } else {
    return static_cast<int32_t>(
      __dp4a(static_cast<uint32_t>(a), static_cast<uint32_t>(b), uint32_t{0}));
  }
}

RAFT_DEVICE_INLINE_FUNCTION auto uint4_word(const uint4& v, uint32_t word_idx) -> uint32_t
{
  switch (word_idx) {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
    default: return v.w;
  }
}
}  // namespace

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct standard_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  constexpr static inline bool kUseIntegralDistance =
    std::is_integral_v<DataT> &&
    (Metric == cuvs::distance::DistanceType::L2Expanded ||
     Metric == cuvs::distance::DistanceType::InnerProduct ||
     Metric == cuvs::distance::DistanceType::CosineExpanded);
  constexpr static inline bool kUsePackedDp4aQuery =
    std::is_same_v<DataT, int8_t> &&
    (Metric == cuvs::distance::DistanceType::L2Expanded ||
     Metric == cuvs::distance::DistanceType::InnerProduct ||
     Metric == cuvs::distance::DistanceType::CosineExpanded);
  constexpr static inline bool kSupportsRowwiseSq8 =
    std::is_same_v<DataT, int8_t> && Metric == cuvs::distance::DistanceType::L2Expanded;
  using QUERY_T = typename std::conditional_t<Metric == cuvs::distance::DistanceType::BitwiseHamming ||
                                                kUseIntegralDistance,
                                              DataT,
                                              float>;
  using base_type::args;
  using base_type::smem_ws_size_in_bytes;
  using typename base_type::args_t;
  using typename base_type::compute_distance_type;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::LOAD_T;
  using typename base_type::setup_workspace_type;
  constexpr static inline auto kMetric          = Metric;
  constexpr static inline auto kTeamSize        = TeamSize;
  constexpr static inline auto kDatasetBlockDim = DatasetBlockDim;

  RAFT_INLINE_FUNCTION constexpr static auto get_query_buf_len(uint32_t dim) -> uint32_t
  {
    return raft::round_up_safe<uint32_t>(dim, DatasetBlockDim);
  }

  RAFT_INLINE_FUNCTION constexpr static auto get_packed_query_offset_in_bytes(uint32_t dim)
    -> uint32_t
  {
    return get_query_buf_len(dim) * sizeof(QUERY_T);
  }

  RAFT_INLINE_FUNCTION constexpr static auto get_packed_query_len(uint32_t dim) -> uint32_t
  {
    return raft::div_rounding_up_safe<uint32_t>(get_query_buf_len(dim), 4);
  }

  RAFT_INLINE_FUNCTION constexpr static auto get_sq8_query_param_offset_in_bytes(uint32_t dim)
    -> uint32_t
  {
    auto query_bytes = get_query_buf_len(dim) * sizeof(QUERY_T);
    auto packed_query_bytes =
      kUsePackedDp4aQuery ? get_packed_query_len(dim) * sizeof(uint32_t) : 0;
    return query_bytes + packed_query_bytes;
  }

  RAFT_INLINE_FUNCTION constexpr static auto get_sq8_query_param_size_in_bytes() -> uint32_t
  {
    return kSupportsRowwiseSq8 ? 4 * sizeof(float) : 0;
  }

  static constexpr RAFT_INLINE_FUNCTION auto ptr(const args_t& args) noexcept
    -> const DATA_T* const&
  {
    return (const DATA_T* const&)(args.extra_ptr1);
  }
  static constexpr RAFT_INLINE_FUNCTION auto ptr(args_t& args) noexcept -> const DATA_T*&
  {
    return (const DATA_T*&)(args.extra_ptr1);
  }

  static constexpr RAFT_INLINE_FUNCTION auto dataset_norms_ptr(const args_t& args) noexcept
    -> const DISTANCE_T* const&
  {
    return (const DISTANCE_T* const&)(args.extra_ptr2);
  }
  static constexpr RAFT_INLINE_FUNCTION auto dataset_norms_ptr(args_t& args) noexcept
    -> const DISTANCE_T*&
  {
    return (const DISTANCE_T*&)(args.extra_ptr2);
  }

  static constexpr RAFT_INLINE_FUNCTION auto rowwise_sq8_dataset_params_ptr(
    const args_t& args) noexcept -> const float* const&
  {
    return (const float* const&)(args.extra_ptr2);
  }
  static constexpr RAFT_INLINE_FUNCTION auto rowwise_sq8_dataset_params_ptr(args_t& args) noexcept
    -> const float*&
  {
    return (const float*&)(args.extra_ptr2);
  }

  static constexpr RAFT_INLINE_FUNCTION auto rowwise_sq8_query_params_ptr(
    const args_t& args) noexcept -> const float* const&
  {
    return (const float* const&)(args.extra_ptr3);
  }
  static constexpr RAFT_INLINE_FUNCTION auto rowwise_sq8_query_params_ptr(args_t& args) noexcept
    -> const float*&
  {
    return (const float*&)(args.extra_ptr3);
  }

  static constexpr RAFT_INLINE_FUNCTION auto ld(const args_t& args) noexcept -> const uint32_t&
  {
    return args.extra_word1;
  }
  static constexpr RAFT_INLINE_FUNCTION auto ld(args_t& args) noexcept -> uint32_t&
  {
    return args.extra_word1;
  }

  _RAFT_HOST_DEVICE standard_dataset_descriptor_t(setup_workspace_type* setup_workspace_impl,
                                                  compute_distance_type* compute_distance_impl,
                                                  const DATA_T* ptr,
                                                  INDEX_T size,
                                                  uint32_t dim,
                                                  uint32_t ld,
                                                  const DISTANCE_T* dataset_norms = nullptr,
                                                  const float* rowwise_sq8_dataset_params = nullptr,
                                                  const float* rowwise_sq8_query_params = nullptr)
    : base_type(setup_workspace_impl,
                compute_distance_impl,
                size,
                dim,
                raft::Pow2<TeamSize>::Log2,
                get_smem_ws_size_in_bytes(dim))
  {
    standard_dataset_descriptor_t::ptr(args)               = ptr;
    standard_dataset_descriptor_t::ld(args)                = ld;
    if constexpr (kSupportsRowwiseSq8) {
      standard_dataset_descriptor_t::rowwise_sq8_dataset_params_ptr(args) =
        rowwise_sq8_dataset_params;
      this->extra_ptr3 = const_cast<float*>(rowwise_sq8_query_params);
    } else {
      standard_dataset_descriptor_t::dataset_norms_ptr(args) = dataset_norms;
    }
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(standard_dataset_descriptor_t) == alignof(base_type));
  }

	 private:
  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    auto query_bytes = get_query_buf_len(dim) * sizeof(QUERY_T);
    auto packed_query_bytes =
      kUsePackedDp4aQuery ? get_packed_query_len(dim) * sizeof(uint32_t) : 0;
    return sizeof(standard_dataset_descriptor_t) + query_bytes + packed_query_bytes +
           get_sq8_query_param_size_in_bytes();
  }
};

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto setup_workspace_standard(
  const DescriptorT* that,
  void* smem_ptr,
  const typename DescriptorT::DATA_T* queries_ptr,
  uint32_t query_id) -> const DescriptorT*
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using base_type                 = typename DescriptorT::base_type;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  using word_type                 = uint32_t;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  auto* r                         = reinterpret_cast<DescriptorT*>(smem_ptr);
  auto* buf                       = reinterpret_cast<QUERY_T*>(r + 1);
  if (r != that) {
    constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(word_type);
    using blob_type           = word_type[kCount];
    auto& src                 = reinterpret_cast<const blob_type&>(*that);
    auto& dst                 = reinterpret_cast<blob_type&>(*r);
    for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
      dst[i] = src[i];
    }
    const auto smem_ptr_offset =
      reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
    if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(word_type))) {
      r->args.smem_ws_ptr = uint32_t(__cvta_generic_to_shared(buf));
    }
    __syncthreads();
  }

  uint32_t dim        = r->args.dim;
  auto buf_len        = raft::round_up_safe<uint32_t>(dim, kDatasetBlockDim);
  constexpr auto vlen = device::get_vlen<LOAD_T, DATA_T>();
  queries_ptr += dim * query_id;
  if constexpr (DescriptorT::kSupportsRowwiseSq8) {
    const auto* query_params = reinterpret_cast<const float*>(r->extra_ptr3);
    query_params =
      query_params == nullptr ? nullptr : query_params + static_cast<size_t>(query_id) * 4;
    if (threadIdx.x == 0) { DescriptorT::rowwise_sq8_query_params_ptr(r->args) = query_params; }
    if (query_params != nullptr && threadIdx.x < 4) {
      auto* smem_query_params = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(buf) + DescriptorT::get_sq8_query_param_offset_in_bytes(dim));
      smem_query_params[threadIdx.x] = query_params[threadIdx.x];
    }
  }
  if constexpr (DescriptorT::kUsePackedDp4aQuery) {
    auto* packed_buf = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uint8_t*>(buf) + DescriptorT::get_packed_query_offset_in_bytes(dim));
    const auto packed_len = DescriptorT::get_packed_query_len(dim);
    for (unsigned p = threadIdx.x; p < packed_len; p += blockDim.x) {
      uint32_t q_pack = 0;
      if ((dim % 4 == 0) && (p * 4 + 3 < dim)) {
        device::ldg_cg(q_pack, reinterpret_cast<const uint32_t*>(queries_ptr) + p);
      } else {
#pragma unroll
        for (uint32_t b = 0; b < 4; b++) {
          const auto i = p * 4 + b;
          DATA_T q     = 0;
          if (i < dim) { q = queries_ptr[i]; }
          q_pack |= pack_dp4a_byte(q, b);
        }
      }
      packed_buf[p] = q_pack;
    }
  } else {
    for (unsigned i = threadIdx.x; i < buf_len; i += blockDim.x) {
      unsigned j = device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(i);
      if (i < dim) {
        if constexpr (std::is_same_v<QUERY_T, DATA_T>) {
          buf[j] = queries_ptr[i];
        } else {
          buf[j] = cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(queries_ptr[i]);
        }
      } else {
        buf[j] = 0;
      }
    }
  }

  return const_cast<const DescriptorT*>(r);
}

template <typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION auto compute_distance_standard_worker(
  const typename DescriptorT::DATA_T* __restrict__ dataset_ptr,
  uint32_t dim,
  uint32_t query_smem_ptr,
  bool rowwise_sq8 = false) -> typename DescriptorT::DISTANCE_T
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using DISTANCE_T                = typename DescriptorT::DISTANCE_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr bool kUseIntegralDistance =
    DescriptorT::kMetric == cuvs::distance::DistanceType::L2Expanded ||
    DescriptorT::kMetric == cuvs::distance::DistanceType::InnerProduct ||
    DescriptorT::kMetric == cuvs::distance::DistanceType::CosineExpanded;
  constexpr bool kUseIntegralAccum =
    std::is_integral_v<DATA_T> && std::is_same_v<QUERY_T, DATA_T> && kUseIntegralDistance;
  constexpr bool kUseDp4aAccum =
    kUseIntegralAccum && is_dp4a_data_type<DATA_T>() &&
    (DescriptorT::kMetric == cuvs::distance::DistanceType::L2Expanded ||
     DescriptorT::kMetric == cuvs::distance::DistanceType::InnerProduct ||
     DescriptorT::kMetric == cuvs::distance::DistanceType::CosineExpanded);
  constexpr auto vlen             = device::get_vlen<LOAD_T, DATA_T>();
  constexpr auto reg_nelem =
    raft::div_rounding_up_unsafe<uint32_t>(kDatasetBlockDim, kTeamSize * vlen);

  using ACCUM_T = std::conditional_t<kUseDp4aAccum, int32_t, DISTANCE_T>;
  ACCUM_T r     = 0;
  for (uint32_t elem_offset = (threadIdx.x % kTeamSize) * vlen; elem_offset < dim;
       elem_offset += kDatasetBlockDim) {
    using DATA_LOAD_T = std::conditional_t<DescriptorT::kUsePackedDp4aQuery, LOAD_T, DATA_T[vlen]>;
    DATA_LOAD_T data[reg_nelem];
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = e * (kTeamSize * vlen) + elem_offset;
      if (k >= dim) break;
      device::ldg_cg(reinterpret_cast<LOAD_T&>(data[e]),
                     reinterpret_cast<const LOAD_T*>(dataset_ptr + k));
    }
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = e * (kTeamSize * vlen) + elem_offset;
      if (k >= dim) break;
      if constexpr (kUseDp4aAccum) {
        if constexpr (DescriptorT::kUsePackedDp4aQuery) {
          static_assert(std::is_same_v<LOAD_T, device::LOAD_128BIT_T>);
          static_assert(vlen == 16);
          const uint32_t packed_query_smem_ptr =
            query_smem_ptr + DescriptorT::get_packed_query_offset_in_bytes(dim);
#pragma unroll
          for (uint32_t word_idx = 0; word_idx < vlen / 4; word_idx++) {
            uint32_t q_pack;
            device::lds(q_pack, packed_query_smem_ptr + sizeof(uint32_t) * ((k / 4) + word_idx));
            const uint32_t x_pack = uint4_word(reinterpret_cast<const uint4&>(data[e]), word_idx);

            const int32_t dot = dp4a_dot<DATA_T>(q_pack, x_pack);
            if (rowwise_sq8) {
              r += dot;
            } else if constexpr (DescriptorT::kMetric == cuvs::distance::DistanceType::L2Expanded) {
              r += dp4a_dot<DATA_T>(q_pack, q_pack) + dp4a_dot<DATA_T>(x_pack, x_pack) -
                   2 * dot;
            } else {
              r -= dot;
            }
          }
        } else {
#pragma unroll
          for (uint32_t v = 0; v < vlen; v += 4) {
            uint32_t q_pack = 0;
            uint32_t x_pack = 0;
#pragma unroll
            for (uint32_t b = 0; b < 4; b++) {
              QUERY_T q;
              device::lds(
                q,
                query_smem_ptr +
                  sizeof(QUERY_T) *
                    device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(k + v + b));
              q_pack |= pack_dp4a_byte(q, b);
              x_pack |= pack_dp4a_byte(data[e][v + b], b);
            }

            const int32_t dot = dp4a_dot<DATA_T>(q_pack, x_pack);
            if (rowwise_sq8) {
              r += dot;
            } else if constexpr (DescriptorT::kMetric == cuvs::distance::DistanceType::L2Expanded) {
              r += dp4a_dot<DATA_T>(q_pack, q_pack) + dp4a_dot<DATA_T>(x_pack, x_pack) -
                   2 * dot;
            } else {
              r -= dot;
            }
          }
        }
      } else {
#pragma unroll
        for (uint32_t v = 0; v < vlen; v++) {
          // Note this loop can go above the dataset_dim for padded arrays. This is not a problem
          // because:
          // - Above the last element (dataset_dim-1), the query array is filled with zeros.
          // - The data buffer has to be also padded with zeros.
          QUERY_T d;
          device::lds(
            d,
            query_smem_ptr +
              sizeof(QUERY_T) * device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(k + v));
          if constexpr (kUseIntegralAccum &&
                        DescriptorT::kMetric == cuvs::distance::DistanceType::L2Expanded) {
            int32_t diff = static_cast<int32_t>(d) - static_cast<int32_t>(data[e][v]);
            r += diff * diff;
          } else if constexpr (kUseIntegralAccum &&
                               (DescriptorT::kMetric ==
                                  cuvs::distance::DistanceType::InnerProduct ||
                                DescriptorT::kMetric ==
                                  cuvs::distance::DistanceType::CosineExpanded)) {
            r -= static_cast<int32_t>(d) * static_cast<int32_t>(data[e][v]);
          } else {
            r += dist_op<QUERY_T, DISTANCE_T, DescriptorT::kMetric>(
              d, cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(data[e][v]));
          }
        }
      }
    }
  }
  return static_cast<DISTANCE_T>(r);
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_standard(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  if constexpr (DescriptorT::kSupportsRowwiseSq8) {
    const auto* dataset_params = DescriptorT::rowwise_sq8_dataset_params_ptr(args);
    const auto* query_params   = DescriptorT::rowwise_sq8_query_params_ptr(args);
    if (dataset_params != nullptr && query_params != nullptr) {
      auto dot_part = compute_distance_standard_worker<DescriptorT>(
        DescriptorT::ptr(args) + (static_cast<std::uint64_t>(DescriptorT::ld(args)) * dataset_index),
        args.dim,
        args.smem_ws_ptr,
        true);

      const float* x = dataset_params + static_cast<std::uint64_t>(dataset_index) * 4;
      const uint32_t query_params_smem_ptr =
        args.smem_ws_ptr + DescriptorT::get_sq8_query_param_offset_in_bytes(args.dim);

      float scale_q;
      device::lds(scale_q, query_params_smem_ptr + sizeof(float));
      const float scale_x = x[1];

      const float scale_prod = scale_q * scale_x;
      float distance_part    = -2.0f * scale_prod * static_cast<float>(dot_part);
      if ((threadIdx.x & (DescriptorT::kTeamSize - 1)) == 0) {
        const float norm_x = x[0];
        const float min_x  = x[2];
        const float sumq_x = x[3];
        float norm_q;
        float min_q;
        float sumq_q;
        device::lds(norm_q, query_params_smem_ptr);
        device::lds(min_q, query_params_smem_ptr + 2 * sizeof(float));
        device::lds(sumq_q, query_params_smem_ptr + 3 * sizeof(float));
        const float uint8_bias =
          128.0f * (sumq_q + sumq_x) - 16384.0f * static_cast<float>(args.dim);
        const float dot_correction =
          scale_prod * uint8_bias + min_x * scale_q * sumq_q + min_q * scale_x * sumq_x +
          min_q * min_x * static_cast<float>(args.dim);
        distance_part += norm_q + norm_x - 2.0f * dot_correction;
      }
      return distance_part;
    }
  }

  auto distance = compute_distance_standard_worker<DescriptorT>(
    DescriptorT::ptr(args) + (static_cast<std::uint64_t>(DescriptorT::ld(args)) * dataset_index),
    args.dim,
    args.smem_ws_ptr);

  if constexpr (DescriptorT::kMetric == cuvs::distance::DistanceType::CosineExpanded) {
    const auto* dataset_norms = DescriptorT::dataset_norms_ptr(args);
    auto norm                 = dataset_norms[dataset_index];
    if (norm > 0) { distance = distance / norm; }
  }

  return distance;
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
RAFT_KERNEL __launch_bounds__(1, 1)
  standard_dataset_descriptor_init_kernel(dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
                                          const DataT* ptr,
                                          IndexT size,
                                          uint32_t dim,
                                          uint32_t ld,
                                          const DistanceT* dataset_norms = nullptr,
                                          const float* rowwise_sq8_dataset_params = nullptr,
                                          const float* rowwise_sq8_query_params = nullptr)
{
  using desc_type =
    standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  using base_type = typename desc_type::base_type;
  new (out) desc_type(reinterpret_cast<typename base_type::setup_workspace_type*>(
                        &setup_workspace_standard<desc_type>),
                      reinterpret_cast<typename base_type::compute_distance_type*>(
                        &compute_distance_standard<desc_type>),
                      ptr,
	                      size,
	                      dim,
	                      ld,
	                      dataset_norms,
	                      rowwise_sq8_dataset_params,
	                      rowwise_sq8_query_params);
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
dataset_descriptor_host<DataT, IndexT, DistanceT>
standard_descriptor_spec<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>::init_(
  const cagra::search_params& params,
  const DataT* ptr,
  IndexT size,
	  uint32_t dim,
	  uint32_t ld,
	  const DistanceT* dataset_norms,
	  const float* rowwise_sq8_dataset_params,
	  const float* rowwise_sq8_query_params)
{
  using desc_type =
    standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  using base_type = typename desc_type::base_type;

  RAFT_EXPECTS(Metric != cuvs::distance::DistanceType::CosineExpanded || dataset_norms != nullptr,
               "Dataset norms must be provided for CosineExpanded metric");

  desc_type dd_host{
    nullptr, nullptr, ptr, size, dim, ld, dataset_norms, rowwise_sq8_dataset_params, rowwise_sq8_query_params};
  return host_type{dd_host,
                   [=](dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_ptr,
                       rmm::cuda_stream_view stream) {
                     standard_dataset_descriptor_init_kernel<Metric,
                                                             TeamSize,
                                                             DatasetBlockDim,
                                                             DataT,
                                                             IndexT,
                                                             DistanceT>
	                       <<<1, 1, 0, stream>>>(dev_ptr,
	                                             ptr,
	                                             size,
	                                             dim,
	                                             ld,
	                                             dataset_norms,
	                                             rowwise_sq8_dataset_params,
	                                             rowwise_sq8_query_params);
                     RAFT_CUDA_TRY(cudaPeekAtLastError());
                   }};
}

}  // namespace cuvs::neighbors::cagra::detail
