#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  ./cuvs_cagra_self_search.sh build  <train.fbin> <index.cuvsindex>
  ./cuvs_cagra_self_search.sh search <query.fbin> <index.cuvsindex>

Run from the cuVS repository root, or set:
  CUVS_ROOT=/path/to/cuvs

Optional:
  CUVS_BUILD_DIR=/path/to/cuvs-build
  CUVS_BENCH_BUILD_DIR=/path/to/benchmark-build
  CMAKE_CUDA_ARCHITECTURES=native
  BUILD_JOBS=$(nproc)
EOF
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

cmd="$1"
fbin_path="$2"
index_path="$3"

if [[ "${cmd}" != "build" && "${cmd}" != "search" ]]; then
  usage
  exit 2
fi
if [[ ! -f "${fbin_path}" ]]; then
  echo "fbin not found: ${fbin_path}" >&2
  exit 1
fi

cuvs_root="${CUVS_ROOT:-$(pwd)}"
cuvs_root="$(cd "${cuvs_root}" && pwd)"
if [[ ! -f "${cuvs_root}/cpp/CMakeLists.txt" ||
      ! -f "${cuvs_root}/cpp/include/cuvs/neighbors/cagra.hpp" ]]; then
  echo "not a cuVS repository root: ${cuvs_root}" >&2
  echo "run from cuVS root or set CUVS_ROOT=/path/to/cuvs" >&2
  exit 1
fi

generator="${CMAKE_GENERATOR:-Ninja}"
jobs="${BUILD_JOBS:-$(nproc)}"
cuda_arch="${CMAKE_CUDA_ARCHITECTURES:-native}"
cuvs_build_dir="${CUVS_BUILD_DIR:-${cuvs_root}/build/cuvs_cagra_self_search_cuvs}"
bench_build_dir="${CUVS_BENCH_BUILD_DIR:-${cuvs_root}/build/cuvs_cagra_self_search_bench}"
bench_src_dir="${bench_build_dir}/src"

mkdir -p "${bench_src_dir}"

cat > "${bench_src_dir}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(cuvs_cagra_self_search LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cuvs CONFIG REQUIRED)

add_executable(cuvs_cagra_self_search cuvs_cagra_self_search.cu)
target_link_libraries(cuvs_cagra_self_search PRIVATE cuvs::cuvs CUDA::cudart)
EOF

cat > "${bench_src_dir}/cuvs_cagra_self_search.cu" <<'EOF'
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

struct Fbin {
    uint32_t rows = 0;
    uint32_t dim = 0;
    std::vector<float> data;
};

Fbin read_fbin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    Fbin out;
    in.read(reinterpret_cast<char*>(&out.rows), sizeof(out.rows));
    in.read(reinterpret_cast<char*>(&out.dim), sizeof(out.dim));
    if (!in || out.rows == 0 || out.dim == 0) {
        throw std::runtime_error("bad fbin header " + path);
    }
    out.data.resize(static_cast<size_t>(out.rows) * out.dim);
    in.read(reinterpret_cast<char*>(out.data.data()),
            static_cast<std::streamsize>(out.data.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("truncated fbin " + path);
    }
    return out;
}

double wall_ms(auto&& fn) {
    auto start = std::chrono::steady_clock::now();
    fn();
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

double cuda_ms(cudaStream_t stream, auto&& fn) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    cuda_check(cudaEventRecord(start, stream), "cudaEventRecord start");
    fn();
    cuda_check(cudaGetLastError(), "search launch");
    cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return ms;
}

cuvs::neighbors::cagra::index_params make_index_params(uint32_t rows, uint32_t dim) {
    cuvs::neighbors::cagra::index_params params;
    params.metric = cuvs::distance::DistanceType::CosineExpanded;
    params.graph_degree = 64;
    params.intermediate_graph_degree = 128;
    params.attach_dataset_on_build = true;

    cuvs::neighbors::cagra::graph_build_params::ivf_pq_params graph_params(
            raft::make_extents<int64_t>(rows, dim), params.metric);
    graph_params.refinement_rate = 2.0f;
    params.graph_build_params = graph_params;
    return params;
}

cuvs::neighbors::cagra::search_params make_search_params() {
    cuvs::neighbors::cagra::search_params params;
    params.itopk_size = 32;
    params.search_width = 1;
    params.max_iterations = 0;
    params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    return params;
}

int run_build(const std::string& train_path, const std::string& index_path) {
    auto train = read_fbin(train_path);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    {
        raft::resources res;
        raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(stream));
        auto train_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
                train.data.data(), static_cast<int64_t>(train.rows), static_cast<int64_t>(train.dim));
        auto index_params = make_index_params(train.rows, train.dim);

        cuvs::neighbors::cagra::index<float, uint32_t> index(res, index_params.metric);
        const double build_ms = wall_ms([&] {
            index = cuvs::neighbors::cagra::build(res, index_params, train_view);
            cuda_check(cudaStreamSynchronize(stream), "sync build");
        });
        cuvs::neighbors::cagra::serialize(res, index_path, index, true);
        cuda_check(cudaStreamSynchronize(stream), "sync serialize");

        std::cout << "stage,train_file,index_file,rows,dim,metric,graph_degree,build_ms\n";
        std::cout << "build," << train_path << "," << index_path << "," << train.rows << ","
                  << train.dim << ",CosineExpanded,64," << build_ms << "\n";
        std::cout.flush();
    }

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}

int run_search(const std::string& query_path, const std::string& index_path) {
    auto query = read_fbin(query_path);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    {
        raft::resources res;
        raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(stream));

        cuvs::neighbors::cagra::index<float, uint32_t> index(
                res, cuvs::distance::DistanceType::CosineExpanded);
        const double load_index_ms = wall_ms([&] {
            cuvs::neighbors::cagra::deserialize(res, index_path, &index);
            cuda_check(cudaStreamSynchronize(stream), "sync deserialize");
        });

        if (index.dim() != query.dim) {
            throw std::runtime_error("query dim does not match index dim");
        }

        float* query_dev = nullptr;
        uint32_t* labels_dev = nullptr;
        float* distances_dev = nullptr;
        const size_t values = static_cast<size_t>(query.rows) * query.dim;
        cuda_check(cudaMalloc(&query_dev, values * sizeof(float)), "cudaMalloc query");
        cuda_check(cudaMalloc(&labels_dev, static_cast<size_t>(query.rows) * sizeof(uint32_t)),
                   "cudaMalloc labels");
        cuda_check(cudaMalloc(&distances_dev, static_cast<size_t>(query.rows) * sizeof(float)),
                   "cudaMalloc distances");
        cuda_check(cudaMemcpyAsync(query_dev, query.data.data(), values * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpy query");
        cuda_check(cudaStreamSynchronize(stream), "sync h2d");

        auto query_view = raft::make_device_matrix_view<const float, int64_t>(
                query_dev, static_cast<int64_t>(query.rows), static_cast<int64_t>(query.dim));
        auto labels_view = raft::make_device_matrix_view<uint32_t, int64_t>(
                labels_dev, static_cast<int64_t>(query.rows), int64_t{1});
        auto distances_view = raft::make_device_matrix_view<float, int64_t>(
                distances_dev, static_cast<int64_t>(query.rows), int64_t{1});
        auto search_params = make_search_params();

        const double search_ms = cuda_ms(stream, [&] {
            cuvs::neighbors::cagra::search(
                    res, search_params, index, query_view, labels_view, distances_view);
        });
        const double qps = static_cast<double>(query.rows) * 1000.0 / search_ms;

        double self_recall_at_1 = -1.0;
        if (query.rows == index.size()) {
            std::vector<uint32_t> labels_host(query.rows);
            cuda_check(cudaMemcpyAsync(labels_host.data(), labels_dev,
                                       labels_host.size() * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpy labels");
            cuda_check(cudaStreamSynchronize(stream), "sync labels");
            uint64_t hit = 0;
            for (uint32_t i = 0; i < query.rows; ++i) {
                if (labels_host[i] == i) {
                    ++hit;
                }
            }
            self_recall_at_1 = static_cast<double>(hit) / query.rows;
        }

        std::cout << "stage,query_file,index_file,rows,dim,metric,itopk,search_width,max_iterations,load_index_ms,search_ms,qps,self_recall_at_1\n";
        std::cout << "search," << query_path << "," << index_path << "," << query.rows << ","
                  << query.dim << ",CosineExpanded,32,1,0," << load_index_ms << "," << search_ms
                  << "," << qps << "," << self_recall_at_1 << "\n";
        std::cout.flush();

        cuda_check(cudaFree(distances_dev), "cudaFree distances");
        cuda_check(cudaFree(labels_dev), "cudaFree labels");
        cuda_check(cudaFree(query_dev), "cudaFree query");
    }

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage:\n  " << argv[0] << " build <train.fbin> <index.cuvsindex>\n"
                      << "  " << argv[0] << " search <query.fbin> <index.cuvsindex>\n";
            return 2;
        }
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        const std::string cmd = argv[1];
        if (cmd == "build") {
            return run_build(argv[2], argv[3]);
        }
        if (cmd == "search") {
            return run_search(argv[2], argv[3]);
        }
        std::cerr << "unknown command: " << cmd << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
EOF

if [[ ! -f "${cuvs_build_dir}/cuvs-config.cmake" ]]; then
  env -u CUVS_ROOT cmake -S "${cuvs_root}/cpp" \
    -B "${cuvs_build_dir}" \
    -G "${generator}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
    -DBUILD_TESTS=OFF \
    -DBUILD_C_LIBRARY=OFF \
    -DBUILD_CAGRA_HNSWLIB=OFF \
    -DBUILD_MG_ALGOS=OFF \
    -DCUVS_COMPILE_DYNAMIC_ONLY=ON
fi

cmake --build "${cuvs_build_dir}" --target cuvs --parallel "${jobs}"

prefix_paths=("${cuvs_build_dir}")
for p in "${cuvs_build_dir}/_deps/"*-build \
  "${cuvs_build_dir}/_deps/cccl-src/lib/cmake" \
  "${cuvs_build_dir}/_deps/cccl-src/thrust/cmake" \
  "${cuvs_build_dir}/_deps/cccl-src/cub/cmake" \
  "${cuvs_build_dir}/_deps/cccl-src/libcudacxx/cmake"; do
  [[ -e "${p}" ]] && prefix_paths+=("${p}")
done
prefix_path="$(IFS=';'; echo "${prefix_paths[*]}")"

env -u CUVS_ROOT cmake -S "${bench_src_dir}" \
  -B "${bench_build_dir}" \
  -G "${generator}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
  -DCMAKE_PREFIX_PATH="${prefix_path}"

cmake --build "${bench_build_dir}" --target cuvs_cagra_self_search --parallel "${jobs}"

export LD_LIBRARY_PATH="${cuvs_build_dir}:${cuvs_build_dir}/_deps/rmm-build:${cuvs_build_dir}/_deps/rapids_logger-build:${LD_LIBRARY_PATH:-}"

"${bench_build_dir}/cuvs_cagra_self_search" "${cmd}" "${fbin_path}" "${index_path}"
