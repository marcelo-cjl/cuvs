/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include <cstdlib>
#include <chrono>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

#include <cuvs/neighbors/cagra.hpp>

#include <cuvs/neighbors/nn_descent.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"

template <typename T>
void cagra_build_and_write(raft::device_resources const& dev_resources,
                           raft::device_matrix_view<const T, int64_t> dataset,
                           std::string out_fname,
                           bool include_dataset,
                           int max_iterations)
{
  using namespace cuvs::neighbors;

  cagra::index_params index_params;
  cagra::graph_build_params::nn_descent_params nn_descent_params;
  nn_descent_params.max_iterations = max_iterations;
  index_params.graph_build_params = nn_descent_params;

  std::cout << "Building CAGRA index (search graph)" << std::endl;

  auto start = std::chrono::system_clock::now();
  auto index = cagra::build(dev_resources, index_params, dataset);
  auto end   = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  std::cout << "Time to build index: " << elapsed_seconds.count() << "s\n";

  // Output index to file
  std::cout << "Serializing CAGRA index to " << out_fname << std::endl;
  serialize(dev_resources, out_fname, index, include_dataset);
  std::cout << "Serialization complete!" << std::endl;
}

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  std::string data = argv[1];
  int max_iterations = std::stoi(argv[2]);
  std::string input_file = "/home/ubuntu/data/" + data + "/" + data + ".fbin";
  std::string out_fname = "/home/ubuntu/data/" + data + "_cagra_" + std::to_string(max_iterations) + ".fbin";

  std::cout << "Reading dataset " + data + " from: " << input_file << std::endl;
  std::cout << "Writing index to: " << out_fname << std::endl;

  auto dataset = read_bin_dataset<float, int64_t>(dev_resources, input_file, INT_MAX);

  cagra_build_and_write<float>(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               out_fname,
                               false,
                               max_iterations);
}
