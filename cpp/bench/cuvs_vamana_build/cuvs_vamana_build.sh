#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  ./cuvs_vamana_build.sh <train.fbin> <result.csv> [index-prefix|-]

Run from the cuVS repository root, or set:
  CUVS_ROOT=/path/to/cuvs

Optional:
  CUVS_BUILD_DIR=/path/to/cuvs-build
  CUVS_BENCH_BUILD_DIR=/path/to/benchmark-build
  CMAKE_CUDA_ARCHITECTURES=native
  BUILD_JOBS=$(nproc)
  METRIC=COSINE|L2
  GRAPH_DEGREE=64
  VISITED_SIZE=128
  VAMANA_ITERS=1.0
  ALPHA=1.2
  MAX_FRACTION=0.06
  BATCH_BASE=2
  QUEUE_SIZE=127
  REVERSE_BATCHSIZE=1000000
  USE_OPT=0
  INCLUDE_DATASET=0
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

fbin_path="$1"
result_path="$2"
index_prefix="${3:--}"

if [[ ! -f "${fbin_path}" ]]; then
  echo "fbin not found: ${fbin_path}" >&2
  exit 1
fi

cuvs_root="${CUVS_ROOT:-$(pwd)}"
cuvs_root="$(cd "${cuvs_root}" && pwd)"
if [[ ! -f "${cuvs_root}/cpp/CMakeLists.txt" ||
      ! -f "${cuvs_root}/cpp/include/cuvs/neighbors/vamana.hpp" ]]; then
  echo "not a cuVS repository root: ${cuvs_root}" >&2
  echo "run from cuVS root or set CUVS_ROOT=/path/to/cuvs" >&2
  exit 1
fi

generator="${CMAKE_GENERATOR:-Ninja}"
jobs="${BUILD_JOBS:-$(nproc)}"
cuda_arch="${CMAKE_CUDA_ARCHITECTURES:-native}"
cuvs_build_dir="${CUVS_BUILD_DIR:-${cuvs_root}/build/cuvs_vamana_build_cuvs}"
bench_build_dir="${CUVS_BENCH_BUILD_DIR:-${cuvs_root}/build/cuvs_vamana_build_bench}"
bench_src_dir="${bench_build_dir}/src"

mkdir -p "${bench_src_dir}" "$(dirname "${result_path}")"

cat > "${bench_src_dir}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(cuvs_vamana_build LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cuvs CONFIG REQUIRED)

add_executable(cuvs_vamana_build cuvs_vamana_build.cu)
target_link_libraries(cuvs_vamana_build PRIVATE cuvs::cuvs CUDA::cudart)
EOF

cat > "${bench_src_dir}/cuvs_vamana_build.cu" <<'EOF'
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/vamana.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/normalize.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
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
    cuda_check(cudaGetLastError(), "cuda launch");
    cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return ms;
}

std::string env_string(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? std::string(fallback) : std::string(value);
}

int env_int(const char* name, int fallback) {
    const auto value = env_string(name, "");
    return value.empty() ? fallback : std::stoi(value);
}

float env_float(const char* name, float fallback) {
    const auto value = env_string(name, "");
    return value.empty() ? fallback : std::stof(value);
}

bool env_bool(const char* name, bool fallback) {
    const auto value = env_string(name, "");
    if (value.empty()) {
        return fallback;
    }
    return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON";
}

std::string upper_ascii(std::string s) {
    for (auto& c : s) {
        if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 'a' + 'A');
        }
    }
    return s;
}

std::string metric_name_from_env() {
    auto metric = upper_ascii(env_string("METRIC", "COSINE"));
    if (metric == "L2" || metric == "L2EXPANDED") {
        return "L2";
    }
    if (metric == "COSINE" || metric == "COSINEEXPANDED") {
        return "COSINE";
    }
    throw std::runtime_error("unsupported METRIC=" + metric + ", expected L2 or COSINE");
}

void write_result(const std::string& result_path,
                  const std::string& line,
                  bool write_header) {
    std::ofstream out(result_path, std::ios::app);
    if (!out) {
        throw std::runtime_error("cannot open result file " + result_path);
    }
    if (write_header) {
        out << "stage,train_file,index_prefix,rows,dim,input_metric,build_metric,graph_degree,"
               "visited_size,vamana_iters,alpha,max_fraction,batch_base,queue_size,"
               "reverse_batchsize,use_opt,include_dataset,data_load_ms,h2d_ms,normalize_ms,build_ms,"
               "serialize_ms,total_ms,medoid\n";
    }
    out << line << "\n";
}

int run(const std::string& train_path, const std::string& result_path, const std::string& index_prefix) {
    Fbin train;
    const double data_load_ms = wall_ms([&] { train = read_fbin(train_path); });

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    raft::resources res;
    raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(stream));

    const size_t values = static_cast<size_t>(train.rows) * train.dim;
    float* train_dev = nullptr;
    cuda_check(cudaMalloc(&train_dev, values * sizeof(float)), "cudaMalloc train");

    const double h2d_ms = cuda_ms(stream, [&] {
        cuda_check(cudaMemcpyAsync(train_dev, train.data.data(), values * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpy train");
    });

    auto train_view = raft::make_device_matrix_view<float, int64_t>(
            train_dev, static_cast<int64_t>(train.rows), static_cast<int64_t>(train.dim));

    const auto input_metric = metric_name_from_env();
    double normalize_ms = 0.0;
    if (input_metric == "COSINE") {
        normalize_ms = cuda_ms(stream, [&] {
            raft::linalg::row_normalize<raft::linalg::L2Norm>(
                    res, raft::make_const_mdspan(train_view), train_view);
        });
    }

    cuvs::neighbors::vamana::index_params params;
    params.metric = cuvs::distance::DistanceType::L2Expanded;
    params.graph_degree = static_cast<uint32_t>(env_int("GRAPH_DEGREE", 64));
    params.visited_size = static_cast<uint32_t>(env_int("VISITED_SIZE", 128));
    params.vamana_iters = env_float("VAMANA_ITERS", 1.0f);
    params.alpha = env_float("ALPHA", 1.2f);
    params.max_fraction = env_float("MAX_FRACTION", 0.06f);
    params.batch_base = env_float("BATCH_BASE", 2.0f);
    params.queue_size = static_cast<uint32_t>(env_int("QUEUE_SIZE", 127));
    params.reverse_batchsize = static_cast<uint32_t>(env_int("REVERSE_BATCHSIZE", 1000000));
    params.use_opt = env_bool("USE_OPT", false);
    const bool include_dataset = env_bool("INCLUDE_DATASET", false);

    auto const_train_view = raft::make_device_matrix_view<const float, int64_t>(
            train_dev, static_cast<int64_t>(train.rows), static_cast<int64_t>(train.dim));
    cuvs::neighbors::vamana::index<float, uint32_t> index(
            res, cuvs::distance::DistanceType::L2Expanded);

    const double build_ms = wall_ms([&] {
        index = cuvs::neighbors::vamana::build(res, params, const_train_view);
        cuda_check(cudaStreamSynchronize(stream), "sync vamana build");
    });

    double serialize_ms = 0.0;
    if (index_prefix != "-") {
        serialize_ms = wall_ms([&] {
            cuvs::neighbors::vamana::serialize(res, index_prefix, index, include_dataset, false);
            cuda_check(cudaStreamSynchronize(stream), "sync vamana serialize");
        });
    }

    const double total_ms = data_load_ms + h2d_ms + normalize_ms + build_ms + serialize_ms;
    const bool write_header = std::ifstream(result_path).peek() == std::ifstream::traits_type::eof();

    std::string line =
            "vamana_build," + train_path + "," + index_prefix + "," +
            std::to_string(train.rows) + "," + std::to_string(train.dim) + "," +
            input_metric + ",L2Expanded," + std::to_string(params.graph_degree) + "," +
            std::to_string(params.visited_size) + "," + std::to_string(params.vamana_iters) + "," +
            std::to_string(params.alpha) + "," + std::to_string(params.max_fraction) + "," +
            std::to_string(params.batch_base) + "," + std::to_string(params.queue_size) + "," +
            std::to_string(params.reverse_batchsize) + "," + (params.use_opt ? "1" : "0") + "," +
            (include_dataset ? "1" : "0") + "," +
            std::to_string(data_load_ms) + "," + std::to_string(h2d_ms) + "," +
            std::to_string(normalize_ms) + "," + std::to_string(build_ms) + "," +
            std::to_string(serialize_ms) + "," + std::to_string(total_ms) + "," +
            std::to_string(index.medoid());

    write_result(result_path, line, write_header);
    std::cout << line << "\n";

    cuda_check(cudaFree(train_dev), "cudaFree train");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: " << argv[0] << " <train.fbin> <result.csv> <index-prefix-or->\n";
            return 2;
        }
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        return run(argv[1], argv[2], argv[3]);
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

cmake --build "${bench_build_dir}" --target cuvs_vamana_build --parallel "${jobs}"

export LD_LIBRARY_PATH="${cuvs_build_dir}:${cuvs_build_dir}/_deps/rmm-build:${cuvs_build_dir}/_deps/rapids_logger-build:${LD_LIBRARY_PATH:-}"

"${bench_build_dir}/cuvs_vamana_build" "${fbin_path}" "${result_path}" "${index_prefix}"
