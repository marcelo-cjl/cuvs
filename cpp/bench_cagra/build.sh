#!/bin/bash
set -e

CUVS_ROOT=/home/ubuntu/cuvs/cpp
BUILD_DIR=${CUVS_ROOT}/build

INCLUDES=(
  -I${CUVS_ROOT}/include
  -I${BUILD_DIR}/_deps/raft-src/cpp/include
  -I${BUILD_DIR}/_deps/raft-build/include
  -I${BUILD_DIR}/_deps/rapids_logger-src/include
  -I${BUILD_DIR}/_deps/rmm-src/cpp/include
  -I${BUILD_DIR}/_deps/rmm-build/include
  -I${BUILD_DIR}/_deps/cccl-src/lib/cmake/thrust/../../../thrust
  -I${BUILD_DIR}/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include
  -I${BUILD_DIR}/_deps/cccl-src/lib/cmake/cub/../../../cub
  -I${BUILD_DIR}/_deps/nvtx3-src/c/include
  -I${BUILD_DIR}/_deps/cuco-src/include
  -I${BUILD_DIR}/_deps/nvidiacutlass-src/include
  -I${BUILD_DIR}/_deps/nvidiacutlass-build/include
  -I${BUILD_DIR}/_deps/dlpack-src/include
)

DEFINES=(
  -DCCCL_DISABLE_PDL
  -DCUB_DISABLE_NAMESPACE_MAGIC
  -DCUB_IGNORE_NAMESPACE_MAGIC_ERROR
  -DCUTLASS_NAMESPACE=raft_cutlass
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
  -DRAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO
  -DRAFT_SYSTEM_LITTLE_ENDIAN=1
  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
  -DTHRUST_DISABLE_ABI_NAMESPACE
  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP
  -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR
  -DCUDA_API_PER_THREAD_DEFAULT_STREAM
)

echo "Compiling bench_cagra_build.cu ..."
/usr/local/cuda-12.6/bin/nvcc \
  -std=c++17 \
  -O3 \
  --generate-code=arch=compute_89,code=sm_89 \
  --expt-extended-lambda \
  --expt-relaxed-constexpr \
  -Xcompiler=-fopenmp \
  ${INCLUDES[@]} \
  ${DEFINES[@]} \
  -L${BUILD_DIR} \
  -L${BUILD_DIR}/_deps/rmm-build \
  -lcuvs -lrmm \
  -Xlinker=-rpath,${BUILD_DIR} \
  -Xlinker=-rpath,${BUILD_DIR}/_deps/rmm-build \
  bench_cagra_build.cu \
  -o bench_cagra_build

echo "Build done: ./bench_cagra_build"
echo ""
echo "Usage: ./bench_cagra_build [data_path] [graph_degree] [num_runs]"
echo "  Default: ./bench_cagra_build /home/ubuntu/data/cohere/cohere.fbin 64 1"
