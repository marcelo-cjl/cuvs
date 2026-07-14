/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

namespace rowwise_sq8 {

constexpr int kWarpSize            = 32;
constexpr int kWarpsPerBlock       = 8;
constexpr int kBlockSize           = kWarpSize * kWarpsPerBlock;
constexpr size_t kMaxSharedBytes   = 48 * 1024;

__forceinline__ __device__ float warp_min(float v)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fminf(v, __shfl_down_sync(0xffffffff, v, offset));
  }
  return __shfl_sync(0xffffffff, v, 0);
}

__forceinline__ __device__ float warp_max(float v)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  }
  return __shfl_sync(0xffffffff, v, 0);
}

__forceinline__ __device__ float warp_sum(float v)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return __shfl_sync(0xffffffff, v, 0);
}

__forceinline__ __device__ int warp_sum(int v)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return __shfl_sync(0xffffffff, v, 0);
}

__forceinline__ __device__ int quantize_value(float v, float row_min, float inv)
{
  int q = __float2int_rz((v - row_min) * inv + 0.5f);
  q     = q < 0 ? 0 : q;
  return q > 255 ? 255 : q;
}

template <int RowsPerBlock, bool CacheRows>
static __global__ __launch_bounds__(RowsPerBlock * kWarpSize, 2) void encode_warp_rows_kernel(
  const float* __restrict__ input,
  int8_t* __restrict__ codes,
  float* __restrict__ params,
  int64_t rows,
  int cols)
{
  static_assert(RowsPerBlock > 0 && RowsPerBlock <= kWarpsPerBlock);

  const int lane       = threadIdx.x & (kWarpSize - 1);
  const int warp       = threadIdx.x >> 5;
  const int64_t row    = static_cast<int64_t>(blockIdx.x) * RowsPerBlock + warp;
  if (warp >= RowsPerBlock || row >= rows) { return; }

  const float* row_input = input + row * cols;
  int8_t* row_codes      = codes + row * cols;
  extern __shared__ float row_cache[];
  float* cached_row = nullptr;
  if constexpr (CacheRows) {
    cached_row = row_cache + static_cast<size_t>(warp) * cols;
  }

  float local_min  = std::numeric_limits<float>::max();
  float local_max  = -std::numeric_limits<float>::max();
  float local_norm = 0.0f;

  for (int i = lane; i < cols; i += kWarpSize) {
    const float v = __ldg(row_input + i);
    if constexpr (CacheRows) {
      cached_row[i] = v;
    }
    local_min = fminf(local_min, v);
    local_max = fmaxf(local_max, v);
    local_norm = fmaf(v, v, local_norm);
  }

  const float row_min  = warp_min(local_min);
  const float row_max  = warp_max(local_max);
  const float row_norm = warp_sum(local_norm);
  const float scale   = row_max == row_min ? 0.0f : (row_max - row_min) / 255.0f;
  const float inv     = row_max == row_min ? 0.0f : 255.0f / (row_max - row_min);

  int local_sum = 0;
  for (int i = lane; i < cols; i += kWarpSize) {
    float v;
    if constexpr (CacheRows) {
      v = cached_row[i];
    } else {
      v = __ldg(row_input + i);
    }
    const int q   = quantize_value(v, row_min, inv);
    row_codes[i] = static_cast<int8_t>(q - 128);
    local_sum += q;
  }

  const int sum_q = warp_sum(local_sum);
  if (lane == 0) {
    params[row * 4]     = row_norm;
    params[row * 4 + 1] = scale;
    params[row * 4 + 2] = row_min;
    params[row * 4 + 3] = static_cast<float>(sum_q);
  }
}

static __global__ __launch_bounds__(kBlockSize, 2) void encode_block_row_kernel(
  const float* __restrict__ input,
  int8_t* __restrict__ codes,
  float* __restrict__ params,
  int64_t rows,
  int cols)
{
  const int64_t row = blockIdx.x;
  if (row >= rows) { return; }

  const float* row_input = input + row * cols;
  int8_t* row_codes      = codes + row * cols;
  extern __shared__ float smem[];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;

  float local_min  = std::numeric_limits<float>::max();
  float local_max  = -std::numeric_limits<float>::max();
  float local_norm = 0.0f;
  for (int i = threadIdx.x; i < cols; i += kBlockSize) {
    const float v = __ldg(row_input + i);
    local_min = fminf(local_min, v);
    local_max = fmaxf(local_max, v);
    local_norm = fmaf(v, v, local_norm);
  }

  local_min  = warp_min(local_min);
  local_max  = warp_max(local_max);
  local_norm = warp_sum(local_norm);
  if (lane == 0) {
    smem[warp * 3]     = local_min;
    smem[warp * 3 + 1] = local_max;
    smem[warp * 3 + 2] = local_norm;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float row_min  = smem[0];
    float row_max  = smem[1];
    float row_norm = smem[2];
#pragma unroll
    for (int i = 1; i < kWarpsPerBlock; ++i) {
      row_min = fminf(row_min, smem[i * 3]);
      row_max = fmaxf(row_max, smem[i * 3 + 1]);
      row_norm += smem[i * 3 + 2];
    }
    smem[0]         = row_min;
    smem[1]         = row_max;
    params[row * 4] = row_norm;
  }
  __syncthreads();

  const float row_min = smem[0];
  const float row_max = smem[1];
  const float scale   = row_max == row_min ? 0.0f : (row_max - row_min) / 255.0f;
  const float inv     = row_max == row_min ? 0.0f : 255.0f / (row_max - row_min);

  int local_sum = 0;
  for (int i = threadIdx.x; i < cols; i += kBlockSize) {
    const int q  = quantize_value(__ldg(row_input + i), row_min, inv);
    row_codes[i] = static_cast<int8_t>(q - 128);
    local_sum += q;
  }

  local_sum = warp_sum(local_sum);
  if (lane == 0) { smem[warp] = static_cast<float>(local_sum); }
  __syncthreads();

  if (threadIdx.x == 0) {
    float sum_q = 0.0f;
#pragma unroll
    for (int i = 0; i < kWarpsPerBlock; ++i) {
      sum_q += smem[i];
    }
    params[row * 4 + 1] = scale;
    params[row * 4 + 2] = row_min;
    params[row * 4 + 3] = sum_q;
  }
}

}  // namespace rowwise_sq8

inline void encode_rowwise_sq8(raft::resources const& res,
                               raft::device_matrix_view<const float, int64_t, raft::row_major> input,
                               raft::device_matrix_view<int8_t, int64_t, raft::row_major> codes,
                               raft::device_matrix_view<float, int64_t, raft::row_major> params)
{
  RAFT_EXPECTS(input.extent(0) == codes.extent(0), "SQ8 input/code row count mismatch");
  RAFT_EXPECTS(input.extent(1) == codes.extent(1), "SQ8 input/code dim mismatch");
  RAFT_EXPECTS(input.extent(0) == params.extent(0), "SQ8 params row count mismatch");
  RAFT_EXPECTS(params.extent(1) == 4, "SQ8 params must have four values per row");

  constexpr int block_size = 256;
  const auto rows          = input.extent(0);
  const auto cols          = input.extent(1);
  if (rows == 0 || cols == 0) { return; }
  RAFT_EXPECTS(cols <= std::numeric_limits<int>::max(), "SQ8 dim exceeds int32 range");

  const auto cols32 = static_cast<int>(cols);
  auto launch_warp_rows = [&](auto rows_per_block_tag) {
    constexpr int rows_per_block = decltype(rows_per_block_tag)::value;
    const auto blocks            = (rows + rows_per_block - 1) / rows_per_block;
    RAFT_EXPECTS(blocks <= std::numeric_limits<uint32_t>::max(), "SQ8 row count exceeds grid range");
    const size_t smem_size = static_cast<size_t>(rows_per_block) * cols32 * sizeof(float);
    rowwise_sq8::encode_warp_rows_kernel<rows_per_block, true>
      <<<static_cast<uint32_t>(blocks),
         rows_per_block * rowwise_sq8::kWarpSize,
         smem_size,
         raft::resource::get_cuda_stream(res)>>>(input.data_handle(),
                                                 codes.data_handle(),
                                                 params.data_handle(),
                                                 rows,
                                                 cols32);
  };

  if (static_cast<size_t>(cols32) * rowwise_sq8::kWarpsPerBlock * sizeof(float) <=
      rowwise_sq8::kMaxSharedBytes) {
    launch_warp_rows(std::integral_constant<int, 8>{});
  } else if (static_cast<size_t>(cols32) * 4 * sizeof(float) <= rowwise_sq8::kMaxSharedBytes) {
    launch_warp_rows(std::integral_constant<int, 4>{});
  } else if (static_cast<size_t>(cols32) * 2 * sizeof(float) <= rowwise_sq8::kMaxSharedBytes) {
    launch_warp_rows(std::integral_constant<int, 2>{});
  } else if (static_cast<size_t>(cols32) * sizeof(float) <= rowwise_sq8::kMaxSharedBytes) {
    launch_warp_rows(std::integral_constant<int, 1>{});
  } else {
    RAFT_EXPECTS(rows <= std::numeric_limits<uint32_t>::max(), "SQ8 row count exceeds grid range");
    const size_t smem_size = rowwise_sq8::kWarpsPerBlock * 3 * sizeof(float);
    rowwise_sq8::encode_block_row_kernel<<<static_cast<uint32_t>(rows),
                                            block_size,
                                            smem_size,
                                            raft::resource::get_cuda_stream(res)>>>(
      input.data_handle(), codes.data_handle(), params.data_handle(), rows, cols32);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::cagra::detail
