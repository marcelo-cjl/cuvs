/*
 * Benchmark: Vamana build performance — float vs int8
 * Dataset: Cohere 1M x 768
 */

#include <cuvs/neighbors/vamana.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <chrono>
#include <cmath>
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

// ---------- build wrappers ----------

double build_vamana_float(raft::device_resources& res,
                          const cuvs::neighbors::vamana::index_params& params,
                          const float* d_data,
                          int64_t nrows,
                          int64_t dim)
{
  auto dataset_view =
    raft::make_device_matrix_view<const float, int64_t>(d_data, nrows, dim);

  cudaDeviceSynchronize();

  auto t0    = std::chrono::high_resolution_clock::now();
  auto index = cuvs::neighbors::vamana::build(res, params, dataset_view);
  raft::resource::sync_stream(res);
  auto t1 = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  printf("  [float]  graph_degree=%u  size=%u  build_time=%.3f s\n",
         index.graph_degree(),
         index.size(),
         elapsed);
  return elapsed;
}

double build_vamana_int8(raft::device_resources& res,
                         const cuvs::neighbors::vamana::index_params& params,
                         const int8_t* d_data,
                         int64_t nrows,
                         int64_t dim)
{
  auto dataset_view =
    raft::make_device_matrix_view<const int8_t, int64_t>(d_data, nrows, dim);

  cudaDeviceSynchronize();

  auto t0    = std::chrono::high_resolution_clock::now();
  auto index = cuvs::neighbors::vamana::build(res, params, dataset_view);
  raft::resource::sync_stream(res);
  auto t1 = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  printf("  [int8]   graph_degree=%u  size=%u  build_time=%.3f s\n",
         index.graph_degree(),
         index.size(),
         elapsed);
  return elapsed;
}

// ---------- main ----------

int main(int argc, char** argv)
{
  std::string data_path = "/home/ubuntu/data/cohere/cohere.fbin";
  int graph_degree      = 32;
  int visited_size      = 0;  // 0 = auto (graph_degree * 2)
  int num_runs          = 1;

  if (argc > 1) data_path = argv[1];
  if (argc > 2) graph_degree = std::atoi(argv[2]);
  if (argc > 3) visited_size = std::atoi(argv[3]);
  if (argc > 4) num_runs = std::atoi(argv[4]);

  printf("=== Vamana Build Benchmark: float vs int8 ===\n");
  printf("Data: %s\n", data_path.c_str());
  printf("Graph degree: %d, Runs: %d\n\n", graph_degree, num_runs);

  // 1. Load float data
  int32_t nrows, dim;
  auto host_float = load_fbin(data_path, nrows, dim);
  size_t total    = static_cast<size_t>(nrows) * dim;

  // 2. Convert to int8 on host: symmetric quantization
  printf("  Computing quantization scale...\n");
  float abs_max = 0;
  for (size_t i = 0; i < total; ++i) {
    float v = std::abs(host_float[i]);
    if (v > abs_max) abs_max = v;
  }
  float scale = (abs_max > 0) ? 127.0f / abs_max : 1.0f;
  printf("  Data range abs_max=%.4f, scale=%.6f\n", abs_max, scale);

  printf("  Converting to int8 on host... (%.1f MB -> %.1f MB)\n",
         total * sizeof(float) / 1e6,
         total * sizeof(int8_t) / 1e6);

  std::vector<int8_t> host_int8(total);
  for (size_t i = 0; i < total; ++i) {
    float val    = host_float[i] * scale;
    val          = std::fmin(std::fmax(val, -128.0f), 127.0f);
    host_int8[i] = static_cast<int8_t>(std::round(val));
  }
  printf("\n");

  // 3. Copy float data to device
  printf("  Copying float data to GPU...\n");
  float* d_float = nullptr;
  size_t bytes_f = total * sizeof(float);
  cudaMalloc(&d_float, bytes_f);
  cudaMemcpy(d_float, host_float.data(), bytes_f, cudaMemcpyHostToDevice);

  // 4. Copy int8 data to device
  printf("  Copying int8 data to GPU...\n");
  int8_t* d_int8 = nullptr;
  size_t bytes_i = total * sizeof(int8_t);
  cudaMalloc(&d_int8, bytes_i);
  cudaMemcpy(d_int8, host_int8.data(), bytes_i, cudaMemcpyHostToDevice);

  printf("  GPU memory: float=%.1f MB, int8=%.1f MB\n\n", bytes_f / 1e6, bytes_i / 1e6);

  // Free host data
  host_float.clear();
  host_float.shrink_to_fit();
  host_int8.clear();
  host_int8.shrink_to_fit();

  // 5. Setup Vamana params
  cuvs::neighbors::vamana::index_params params;
  params.metric       = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree = graph_degree;
  params.visited_size = (visited_size > 0) ? visited_size : graph_degree * 2;

  printf("  Vamana params: graph_degree=%d, visited_size=%d, alpha=%.1f\n\n",
         params.graph_degree,
         params.visited_size,
         params.alpha);

  raft::device_resources res;

  // 6. Benchmark
  for (int run = 0; run < num_runs; ++run) {
    printf("--- Run %d/%d ---\n", run + 1, num_runs);

    double t_float = build_vamana_float(res, params, d_float, nrows, dim);
    double t_int8  = build_vamana_int8(res, params, d_int8, nrows, dim);

    printf("  Speedup (float/int8): %.2fx\n\n", t_float / t_int8);
  }

  cudaFree(d_float);
  cudaFree(d_int8);

  printf("Done.\n");
  return 0;
}
