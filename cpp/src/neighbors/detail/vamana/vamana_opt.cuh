#pragma once

#include "priority_queue.cuh"
#include "vamana_structs.cuh"

#include <cub/cub.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/util/warp_primitives.cuh>

namespace cuvs::neighbors::vamana::detail {

template <typename IdxT>
__device__ bool opt_bloom_check_and_mark(uint32_t* bloom_filter, int n_bits, IdxT id)
{
  uint32_t u    = static_cast<uint32_t>(id);
  uint32_t bit1 = (u * 2654435761u) % n_bits;
  uint32_t bit2 = (u * 2246822519u) % n_bits;
  uint32_t w1   = bloom_filter[bit1 / 32];
  uint32_t w2   = bloom_filter[bit2 / 32];
  bool seen     = ((w1 >> (bit1 % 32)) & 1u) && ((w2 >> (bit2 % 32)) & 1u);
  if (!seen) {
    bloom_filter[bit1 / 32] |= 1u << (bit1 % 32);
    bloom_filter[bit2 / 32] |= 1u << (bit2 % 32);
  }
  return seen;
}

template <typename T, typename accT, typename IdxT>
__forceinline__ __device__ void opt_enqueue_all_neighbors(int num_neighbors,
                                                          Point<T, accT>* query_vec,
                                                          const T* vec_ptr,
                                                          int* neighbor_array,
                                                          PriorityQueue<IdxT, accT>& heap_queue,
                                                          int dim,
                                                          cuvs::distance::DistanceType metric,
                                                          uint32_t* bloom_filter,
                                                          int bloom_bits,
                                                          int n)
{
  __shared__ bool skip_neighbor;
  for (int i = 0; i < num_neighbors; i++) {
    if (threadIdx.x == 0) {
      IdxT id = static_cast<IdxT>(neighbor_array[i]);
      skip_neighbor =
        id == raft::upper_bound<IdxT>() || id >= static_cast<IdxT>(n) ||
        opt_bloom_check_and_mark<IdxT>(bloom_filter, bloom_bits, id);
    }
    __syncthreads();
    if (!skip_neighbor) {
      accT dist_out = dist<T, accT>(
        query_vec->coords, &vec_ptr[(size_t)(neighbor_array[i]) * (size_t)(dim)], dim, metric);
      __syncthreads();
      if (threadIdx.x == 0) { heap_queue.insert_back(dist_out, neighbor_array[i]); }
      __syncthreads();
    }
  }
}

template <typename T,
          typename accT,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
__global__ void GreedySearchKernelOpt(
  const IdxT* graph,
  int graph_stride,
  int graph_degree_cap,
  const int* degree_count,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  void* query_list_ptr,
  int num_queries,
  int medoid_id,
  int topk,
  cuvs::distance::DistanceType metric,
  int max_queue_size,
  Node<accT>* topk_pq_mem)
{
  constexpr int bloom_words = 1024;
  constexpr int bloom_bits  = bloom_words * 32;
  int n                    = dataset.extent(0);
  int dim                  = dataset.extent(1);

  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  static __shared__ int topk_q_size;
  static __shared__ int cand_q_size;
  static __shared__ accT cur_k_max;
  static __shared__ int k_max_idx;
  static __shared__ Point<T, accT> s_query;
  static __shared__ int num_neighbors;

  union ShmemLayout {
    T coords;
    int neighborhood_arr;
    DistPair<IdxT, accT> candidate_queue;
    uint32_t bloom;
  };

  int align_padding = (((dim - 1) / alignof(ShmemLayout)) + 1) * alignof(ShmemLayout) - dim;
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];

  size_t smem_offset = 0;
  T* s_coords        = reinterpret_cast<T*>(&smem[smem_offset]);
  smem_offset += (dim + align_padding) * sizeof(T);
  smem_offset = (smem_offset + alignof(int) - 1) & ~(alignof(int) - 1);
  int* neighbor_array = reinterpret_cast<int*>(&smem[smem_offset]);
  smem_offset += graph_degree_cap * sizeof(int);
  smem_offset = (smem_offset + alignof(DistPair<IdxT, accT>) - 1) &
                ~(alignof(DistPair<IdxT, accT>) - 1);
  DistPair<IdxT, accT>* candidate_queue_smem =
    reinterpret_cast<DistPair<IdxT, accT>*>(&smem[smem_offset]);
  smem_offset += max_queue_size * sizeof(DistPair<IdxT, accT>);
  smem_offset = (smem_offset + alignof(uint32_t) - 1) & ~(alignof(uint32_t) - 1);
  uint32_t* bloom_filter = reinterpret_cast<uint32_t*>(&smem[smem_offset]);

  s_query.coords = s_coords;
  s_query.Dim    = dim;

  Node<accT>* topk_pq = &topk_pq_mem[blockIdx.x * topk];
  PriorityQueue<IdxT, accT> heap_queue;

  if (threadIdx.x == 0) {
    heap_queue.initialize(candidate_queue_smem, max_queue_size, &cand_q_size);
  }

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    __syncthreads();
    query_list[i].reset();
    for (int w = threadIdx.x; w < bloom_words; w += blockDim.x) {
      bloom_filter[w] = 0;
    }
    update_shared_point<T, accT>(&s_query, &dataset(0, 0), query_list[i].queryId, dim);

    if (threadIdx.x == 0) {
      topk_q_size = 0;
      cand_q_size = 0;
      s_query.id  = query_list[i].queryId;
      cur_k_max   = 0;
      k_max_idx   = 0;
      heap_queue.reset();
    }
    __syncthreads();

    Point<T, accT>* query_vec = &s_query;
    const T* medoid           = &dataset((size_t)medoid_id, 0);
    accT medoid_dist          = dist<T, accT>(query_vec->coords, medoid, dim, metric);
    if (threadIdx.x == 0) {
      opt_bloom_check_and_mark<IdxT>(bloom_filter, bloom_bits, static_cast<IdxT>(medoid_id));
      heap_queue.insert_back(medoid_dist, medoid_id);
    }
    __syncthreads();

    while (cand_q_size != 0) {
      int cand_num;
      accT cur_distance;
      if (threadIdx.x == 0) {
        DistPair<IdxT, accT> test_cand_out = heap_queue.pop();
        cand_num                           = test_cand_out.idx;
        cur_distance                       = test_cand_out.dist;
        if (query_list[i].size < query_list[i].maxSize) {
          query_list[i].ids[query_list[i].size]   = cand_num;
          query_list[i].dists[query_list[i].size] = cur_distance;
          query_list[i].size++;
        }
      }
      __syncthreads();

      cand_num     = raft::shfl(cand_num, 0);
      cur_distance = raft::shfl(cur_distance, 0);

      bool done      = false;
      bool pass_flag = false;
      if (topk_q_size == topk) {
        if (threadIdx.x == 0) {
          if (cur_k_max <= cur_distance) { done = true; }
        }
        done = raft::shfl(done, 0);
        if (done) {
          if (query_list[i].size < topk) {
            pass_flag = true;
          } else if (query_list[i].size >= topk) {
            break;
          }
        }
      }

      Node<accT> new_cand;
      new_cand.distance = cur_distance;
      new_cand.nodeid   = cand_num;
      if (check_duplicate(topk_pq, topk_q_size, new_cand) == false) {
        if (!pass_flag) {
          parallel_pq_max_enqueue<accT>(
            topk_pq, &topk_q_size, topk, new_cand, &cur_k_max, &k_max_idx);
          __syncthreads();
        }
      } else {
        continue;
      }

      if (threadIdx.x == 0) {
        int actual_degree = degree_count ? degree_count[cand_num] : graph_degree_cap;
        num_neighbors    = min(actual_degree, graph_degree_cap);
      }
      __syncthreads();

      for (int j = threadIdx.x; j < num_neighbors; j += blockDim.x) {
        neighbor_array[j] = graph[(size_t)cand_num * graph_stride + j];
      }
      __syncthreads();

      opt_enqueue_all_neighbors<T, accT, IdxT>(num_neighbors,
                                               query_vec,
                                               &dataset(0, 0),
                                               neighbor_array,
                                               heap_queue,
                                               dim,
                                               metric,
                                               bloom_filter,
                                               bloom_bits,
                                               n);
      __syncthreads();
    }

    bool self_found = false;
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {
      if (query_list[i].ids[j] == query_vec->id) {
        query_list[i].dists[j] = raft::upper_bound<accT>();
        query_list[i].ids[j]   = raft::upper_bound<IdxT>();
        self_found             = true;
      }
    }
    for (int j = query_list[i].size + threadIdx.x; j < query_list[i].maxSize; j += blockDim.x) {
      query_list[i].ids[j]   = raft::upper_bound<IdxT>();
      query_list[i].dists[j] = raft::upper_bound<accT>();
    }
    __syncthreads();
    if (self_found) query_list[i].size--;
  }
}

template <typename accT, typename IdxT = uint32_t>
__global__ void write_graph_edges_kernel_opt(IdxT* graph,
                                             int graph_stride,
                                             void* query_list_ptr,
                                             int degree,
                                             int num_queries,
                                             int* degree_count)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);
  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    int query_id   = query_list[i].queryId;
    int write_size = min(query_list[i].size, degree);
    for (int j = threadIdx.x; j < write_size; j += blockDim.x) {
      graph[(size_t)query_id * graph_stride + j] = query_list[i].ids[j];
    }
    for (int j = write_size + threadIdx.x; j < degree; j += blockDim.x) {
      graph[(size_t)query_id * graph_stride + j] = raft::upper_bound<IdxT>();
    }
    if (threadIdx.x == 0) { degree_count[query_id] = write_size; }
  }
}

template <typename accT, typename IdxT = uint32_t>
__global__ void write_graph_edges_and_reverse_pairs_kernel_opt(IdxT* graph,
                                                              int graph_stride,
                                                              void* query_list_ptr,
                                                              int degree,
                                                              int num_queries,
                                                              int* degree_count,
                                                              IdxT* edge_dest,
                                                              IdxT* edge_src)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);
  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    int query_id     = query_list[i].queryId;
    int write_size   = min(query_list[i].size, degree);
    size_t pair_base = (size_t)i * degree;
    for (int j = threadIdx.x; j < degree; j += blockDim.x) {
      IdxT dest = raft::upper_bound<IdxT>();
      IdxT src  = raft::upper_bound<IdxT>();
      if (j < write_size) {
        dest                                = query_list[i].ids[j];
        graph[(size_t)query_id * graph_stride + j] = dest;
        if (dest != raft::upper_bound<IdxT>()) { src = static_cast<IdxT>(query_id); }
      } else {
        graph[(size_t)query_id * graph_stride + j] = raft::upper_bound<IdxT>();
      }
      edge_dest[pair_base + j] = dest;
      edge_src[pair_base + j]  = src;
    }
    if (threadIdx.x == 0) { degree_count[query_id] = write_size; }
  }
}

template <typename IdxT = uint32_t>
__global__ void append_reverse_edges_kernel_opt(IdxT* graph,
                                                int graph_stride,
                                                int* degree_count,
                                                const IdxT* src_sorted,
                                                const IdxT* run_offsets,
                                                const IdxT* run_values,
                                                const IdxT* run_lengths,
                                                const int* n_runs)
{
  int nr = *n_runs;
  __shared__ int s_cur_deg;
  __shared__ int s_dup;
  for (int bid = blockIdx.x; bid < nr; bid += gridDim.x) {
    IdxT dest = run_values[bid];
    if (dest == raft::upper_bound<IdxT>()) { continue; }
    IdxT seg_start = run_offsets[bid];
    IdxT seg_len   = run_lengths[bid];
    if (threadIdx.x == 0) { s_cur_deg = min(degree_count[dest], graph_stride); }
    __syncthreads();
    for (IdxT i = 0; i < seg_len; i++) {
      if (s_cur_deg >= graph_stride) { break; }
      IdxT src = src_sorted[seg_start + i];
      if (src == raft::upper_bound<IdxT>()) { continue; }
      if (threadIdx.x == 0) { s_dup = 0; }
      __syncthreads();
      int cur = s_cur_deg;
      for (int k = threadIdx.x; k < cur; k += blockDim.x) {
        if (graph[(size_t)dest * graph_stride + k] == src) { s_dup = 1; }
      }
      __syncthreads();
      if (s_dup) { continue; }
      if (threadIdx.x == 0) {
        graph[(size_t)dest * graph_stride + s_cur_deg] = src;
        s_cur_deg++;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) { degree_count[dest] = s_cur_deg; }
    __syncthreads();
  }
}

template <typename IdxT = uint32_t>
__global__ void find_overflow_runs_kernel_opt(const int* degree_count,
                                              const IdxT* run_values,
                                              const int* n_runs,
                                              int degree,
                                              IdxT* overflow_nodes,
                                              int* overflow_count)
{
  int nr = *n_runs;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nr; i += blockDim.x * gridDim.x) {
    IdxT node = run_values[i];
    if (node != raft::upper_bound<IdxT>() && degree_count[node] > degree) {
      int pos             = atomicAdd(overflow_count, 1);
      overflow_nodes[pos] = node;
    }
  }
}

template <typename T,
          typename accT,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
__global__ void init_overflow_extra_query_list_kernel_opt(
  QueryCandidates<IdxT, accT>* query_list,
  IdxT* candidate_ids,
  accT* candidate_dists,
  const IdxT* overflow_nodes,
  int overflow_start,
  int n_queries,
  const IdxT* graph,
  int graph_stride,
  int degree,
  int extra_cap,
  const int* degree_count,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  cuvs::distance::DistanceType metric)
{
  int n   = dataset.extent(0);
  int dim = dataset.extent(1);
  __shared__ accT s_dist;

  for (int i = blockIdx.x; i < n_queries; i += gridDim.x) {
    IdxT node = overflow_nodes[overflow_start + i];
    int extra_size =
      node < static_cast<IdxT>(n) ? degree_count[node] - degree : 0;
    if (extra_size < 0) { extra_size = 0; }
    if (extra_size > extra_cap) { extra_size = extra_cap; }

    if (threadIdx.x == 0) {
      query_list[i].maxSize = extra_cap;
      query_list[i].size    = extra_size;
      query_list[i].queryId = node;
      query_list[i].ids     = &candidate_ids[(size_t)i * extra_cap];
      query_list[i].dists   = &candidate_dists[(size_t)i * extra_cap];
    }
    __syncthreads();

    for (int j = 0; j < extra_cap; j++) {
      IdxT id   = raft::upper_bound<IdxT>();
      accT dist_out = raft::upper_bound<accT>();
      bool valid = j < extra_size;
      if (valid) {
        id = graph[(size_t)node * graph_stride + degree + j];
        valid = id != raft::upper_bound<IdxT>() && id < static_cast<IdxT>(n) && id != node;
      }
      if (valid) {
        accT d = dist<T, accT>(&dataset((size_t)node, 0), &dataset((size_t)id, 0), dim, metric);
        if (threadIdx.x == 0) { s_dist = d; }
        __syncthreads();
        dist_out = s_dist;
      }
      if (threadIdx.x == 0) {
        query_list[i].ids[j]   = id;
        query_list[i].dists[j] = dist_out;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      int out = 0;
      for (int j = 0; j < extra_cap; j++) {
        IdxT id = query_list[i].ids[j];
        if (id != raft::upper_bound<IdxT>()) {
          query_list[i].ids[out]   = id;
          query_list[i].dists[out] = query_list[i].dists[j];
          out++;
        }
      }
      for (int j = out; j < extra_cap; j++) {
        query_list[i].ids[j]   = raft::upper_bound<IdxT>();
        query_list[i].dists[j] = raft::upper_bound<accT>();
      }
      query_list[i].size = out;
      for (int a = 1; a < out; a++) {
        IdxT id = query_list[i].ids[a];
        accT d  = query_list[i].dists[a];
        int b   = a - 1;
        while (b >= 0 &&
               (query_list[i].dists[b] > d ||
                (query_list[i].dists[b] == d && query_list[i].ids[b] > id))) {
          query_list[i].ids[b + 1]   = query_list[i].ids[b];
          query_list[i].dists[b + 1] = query_list[i].dists[b];
          b--;
        }
        query_list[i].ids[b + 1]   = id;
        query_list[i].dists[b + 1] = d;
      }
      for (int a = 1; a < out; a++) {
        if (query_list[i].ids[a] == query_list[i].ids[a - 1]) {
          query_list[i].ids[a - 1] = raft::upper_bound<IdxT>();
          query_list[i].dists[a - 1] = raft::upper_bound<accT>();
        }
      }
      int compact = 0;
      for (int a = 0; a < out; a++) {
        if (query_list[i].ids[a] != raft::upper_bound<IdxT>()) {
          if (compact != a) {
            query_list[i].ids[compact]   = query_list[i].ids[a];
            query_list[i].dists[compact] = query_list[i].dists[a];
          }
          compact++;
        }
      }
      for (int a = compact; a < extra_cap; a++) {
        query_list[i].ids[a]   = raft::upper_bound<IdxT>();
        query_list[i].dists[a] = raft::upper_bound<accT>();
      }
      query_list[i].size = compact;
    }
    __syncthreads();
  }
}

}
