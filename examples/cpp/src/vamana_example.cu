/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/vamana.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"

template <typename T>
void vamana_build_and_write(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<const T, int64_t> dataset,
                            std::string out_fname,
                            int degree,
                            int visited_size)
{
  using namespace cuvs::neighbors;

  vamana::index_params index_params;
  index_params.graph_degree = degree;
  index_params.visited_size = visited_size;

  auto start = std::chrono::system_clock::now();
  auto index = vamana::build(dev_resources, index_params, dataset);
  auto end   = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  std::cout << "Time to build index: " << elapsed_seconds.count() << "s\n";

  serialize(dev_resources, out_fname, index, false);
}

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used
  // within RAFT algorithms). In that case only the internal arrays would use
  // the pool, any other allocation uses the default RMM memory resource. Here
  // is how to change the workspace memory resource to a pool with 2 GiB upper
  // limit. raft::resource::set_workspace_to_pool_resource(dev_resources, 2 *
  // 1024 * 1024 * 1024ull);

  // if (argc != 8 && argc != 9) usage();

  std::string data = argv[1];
  std::string input_file = "/home/ubuntu/data/" + data + "/" + data + ".fbin";
  std::string out_fname = "/home/ubuntu/data/" + data + "_vamana.fbin";
  int degree                  = 64;
  int max_visited             = 128;

  std::cout << "Reading dataset " + data + " from: " << input_file << std::endl;
  std::cout << "Writing index to: " << out_fname << std::endl;

  auto dataset = read_bin_dataset<float, int64_t>(dev_resources, input_file, INT_MAX);

  vamana_build_and_write<float>(dev_resources,
                                raft::make_const_mdspan(dataset.view()),
                                out_fname,
                                degree,
                                max_visited);
}
