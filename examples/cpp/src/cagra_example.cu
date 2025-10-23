/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <unordered_set>

#include "common.cuh"

void cagra_build_search_simple(raft::device_resources const& dev_resources,
                               raft::device_matrix_view<const float, int64_t> train_set,
                               raft::device_matrix_view<const float, int64_t> query_set,
                               raft::device_matrix_view<const int64_t, int64_t> truth_set)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 10;
  int64_t n_queries = query_set.extent(0);

  // create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // use default index parameters
  cagra::index_params index_params;
  // Set distance metric to Cosine
  index_params.metric = cuvs::distance::DistanceType::CosineExpanded;

  std::cout << "Building CAGRA index (search graph) with Cosine distance" << std::endl;
  auto index = cagra::build(dev_resources, index_params, train_set);

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  // use default search parameters
  cagra::search_params search_params;
  // search K nearest neighbors
  cagra::search(dev_resources, search_params, index, query_set, neighbors.view(), distances.view());

  // The call to cagra::search is asynchronous. Before accessing the data, sync by calling
  raft::resource::sync_stream(dev_resources);

  // Copy results from device to host for recall calculation
  auto h_neighbors = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  auto h_truth_set = raft::make_host_matrix<int64_t, int64_t>(truth_set.extent(0), truth_set.extent(1));
  
  raft::copy(h_neighbors.data_handle(), neighbors.data_handle(), neighbors.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::copy(h_truth_set.data_handle(), truth_set.data_handle(), truth_set.size(), raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // Calculate recall
  float recall = 0.0;
  for (int64_t i = 0; i < n_queries; i++) {
    std::unordered_set<uint32_t> truth_set_i;
    std::unordered_set<uint32_t> result_set_i;
    
    // Fill truth set
    for (int64_t j = 0; j < topk; j++) {
      truth_set_i.insert(static_cast<uint32_t>(h_truth_set(i, j)));
    }
    
    // Fill result set
    for (int64_t j = 0; j < topk; j++) {
      result_set_i.insert(h_neighbors(i, j));
    }
    
    // Count common elements
    int common = 0;
    for (auto it = result_set_i.begin(); it != result_set_i.end(); ++it) {
      if (truth_set_i.find(*it) != truth_set_i.end()) {
        common++;
      }
    }
    recall += static_cast<float>(common) / static_cast<float>(topk);
  }
  recall /= n_queries;
  std::cout << "Recall: " << recall << std::endl;
}

template <typename T, typename idxT>
raft::device_matrix<T, idxT> read_fbin(raft::device_resources const& dev_resources,
                                       std::string fname)
{
  std::ifstream datafile(fname, std::ifstream::binary);
  uint32_t N;
  uint32_t dim;
  datafile.read((char*)&N, sizeof(uint32_t));
  datafile.read((char*)&dim, sizeof(uint32_t));
  std::cout << "Read in file - N:" << N << ", dim:" << dim << std::endl;

  std::vector<T> data;
  data.resize((size_t)N * (size_t)dim);
  datafile.read(reinterpret_cast<char*>(data.data()), (size_t)N * (size_t)dim * sizeof(T));
  datafile.close();

  auto dataset = raft::make_device_matrix<T, idxT>(dev_resources, N, dim);
  raft::copy(dataset.data_handle(),
             data.data(),
             data.size(),
             raft::resource::get_cuda_stream(dev_resources));

  return dataset;
}

int main(int argc, char** argv)
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  std::string train_file = "/home/ubuntu/data/cohere/cohere.fbin";
  std::string query_file = "/home/ubuntu/data/cohere/cohere_query.fbin";
  std::string truth_file = "/home/ubuntu/data/cohere/cohere_query_COSINE_0.01_100.truth";

  auto train_set = read_bin_dataset<float, int64_t>(dev_resources, train_file);
  auto query_set = read_bin_dataset<float, int64_t>(dev_resources, query_file);
  auto truth_set = read_bin_dataset<int64_t, int64_t>(dev_resources, truth_file);

  std::cout << "Train set: rows=" << train_set.extent(0) << ", cols=" << train_set.extent(1) << std::endl;
  std::cout << "Query set: rows=" << query_set.extent(0) << ", cols=" << query_set.extent(1) << std::endl;
  std::cout << "Truth set: rows=" << truth_set.extent(0) << ", cols=" << truth_set.extent(1) << std::endl;

  cagra_build_search_simple(dev_resources,
                            raft::make_const_mdspan(train_set.view()),
                            raft::make_const_mdspan(query_set.view()),
                            raft::make_const_mdspan(truth_set.view()));
}
