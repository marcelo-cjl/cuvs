/*
 * Benchmark: CAGRA build performance — float vs int8
 * Dataset: Cohere 1M x 768
 */

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuda_fp16.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// ---------- helpers ----------

struct FbinHeader {
  int32_t nrows;
  int32_t dim;
};

std::vector<float> load_fbin(const std::string& path, int32_t& nrows, int32_t& dim)
{
  std::ifstream fin(path, std::ios::binary);
  if (!fin) { throw std::runtime_error("Cannot open " + path); }

  FbinHeader hdr;
  fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  nrows = hdr.nrows;
  dim   = hdr.dim;

  printf("  Loading %s: %d rows x %d dims\n", path.c_str(), nrows, dim);

  std::vector<float> data(static_cast<size_t>(nrows) * dim);
  fin.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
  return data;
}

__global__ void float_to_int8_kernel(const float* src, int8_t* dst, size_t n, float scale)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = src[idx] * scale;
    val       = fminf(fmaxf(val, -128.0f), 127.0f);
    dst[idx]  = static_cast<int8_t>(rintf(val));
  }
}

// ---------- build wrappers ----------

double build_cagra_float(raft::device_resources& res,
                         const cuvs::neighbors::cagra::index_params& params,
                         const float* d_data,
                         int64_t nrows,
                         int64_t dim)
{
  auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(d_data, nrows, dim);

  cudaDeviceSynchronize();

  auto t0    = std::chrono::high_resolution_clock::now();
  auto index = cuvs::neighbors::cagra::build(res, params, dataset_view);
  raft::resource::sync_stream(res);
  auto t1 = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  printf("  [float]  graph_degree=%lu  size=%u  build_time=%.3f s\n",
         (unsigned long)index.graph_degree(),
         index.size(),
         elapsed);
  return elapsed;
}

double build_cagra_int8(raft::device_resources& res,
                        const cuvs::neighbors::cagra::index_params& params,
                        const int8_t* d_data,
                        int64_t nrows,
                        int64_t dim)
{
  auto dataset_view = raft::make_device_matrix_view<const int8_t, int64_t>(d_data, nrows, dim);

  cudaDeviceSynchronize();

  auto t0    = std::chrono::high_resolution_clock::now();
  auto index = cuvs::neighbors::cagra::build(res, params, dataset_view);
  raft::resource::sync_stream(res);
  auto t1 = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  printf("  [int8]   graph_degree=%lu  size=%u  build_time=%.3f s\n",
         (unsigned long)index.graph_degree(),
         index.size(),
         elapsed);
  return elapsed;
}

// ---------- main ----------

int main(int argc, char** argv)
{
  std::string data_path = "/home/ubuntu/data/cohere/cohere.fbin";
  int graph_degree      = 64;
  int num_runs          = 1;

  if (argc > 1) data_path = argv[1];
  if (argc > 2) graph_degree = std::atoi(argv[2]);
  if (argc > 3) num_runs = std::atoi(argv[3]);

  printf("=== CAGRA Build Benchmark: float vs int8 ===\n");
  printf("Data: %s\n", data_path.c_str());
  printf("Graph degree: %d, Runs: %d\n\n", graph_degree, num_runs);

  // 1. Load data
  int32_t nrows, dim;
  auto host_data = load_fbin(data_path, nrows, dim);

  // 2. Compute scale: map [min, max] -> [-128, 127]
  float vmin = host_data[0], vmax = host_data[0];
  for (size_t i = 1; i < host_data.size(); ++i) {
    if (host_data[i] < vmin) vmin = host_data[i];
    if (host_data[i] > vmax) vmax = host_data[i];
  }
  float abs_max = std::max(std::abs(vmin), std::abs(vmax));
  float scale   = (abs_max > 0) ? 127.0f / abs_max : 1.0f;
  printf("  Data range: [%.4f, %.4f], scale=%.6f\n\n", vmin, vmax, scale);

  // 3. Copy float data to device
  printf("  Copying float data to GPU...\n");
  float* d_float = nullptr;
  size_t total   = static_cast<size_t>(nrows) * dim;
  size_t bytes_f = total * sizeof(float);
  size_t bytes_i = total * sizeof(int8_t);
  cudaMalloc(&d_float, bytes_f);
  cudaMemcpy(d_float, host_data.data(), bytes_f, cudaMemcpyHostToDevice);

  // 4. Convert to int8 on GPU
  printf("  Converting to int8 on GPU... (%.1f MB -> %.1f MB)\n", bytes_f / 1e6, bytes_i / 1e6);
  int8_t* d_int8 = nullptr;
  cudaMalloc(&d_int8, bytes_i);
  {
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    float_to_int8_kernel<<<blocks, threads>>>(d_float, d_int8, total, scale);
    cudaDeviceSynchronize();
  }

  printf("  GPU memory: float=%.1f MB, int8=%.1f MB\n\n", bytes_f / 1e6, bytes_i / 1e6);

  // Free host data to save memory
  host_data.clear();
  host_data.shrink_to_fit();

  // 5. Setup CAGRA params
  cuvs::neighbors::cagra::index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = graph_degree;
  params.intermediate_graph_degree = graph_degree * 2;
  params.attach_dataset_on_build   = false;

  raft::device_resources res;

  // 6. Benchmark
  for (int run = 0; run < num_runs; ++run) {
    printf("--- Run %d/%d ---\n", run + 1, num_runs);

    double t_float = build_cagra_float(res, params, d_float, nrows, dim);
    double t_int8  = build_cagra_int8(res, params, d_int8, nrows, dim);

    printf("  Speedup (float/int8): %.2fx\n\n", t_float / t_int8);
  }

  // Cleanup
  cudaFree(d_float);
  cudaFree(d_int8);

  printf("Done.\n");
  return 0;
}
