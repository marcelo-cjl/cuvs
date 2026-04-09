# Brute Force 索引详细分析

## 一、简介

Brute Force (暴力搜索) 是最简单的向量搜索方法，通过计算查询向量与数据集中所有向量的距离来找到最近邻。

**支持的数据类型**: float, half (float16)

**支持的距离度量**: L2Expanded, L2SqrtExpanded, InnerProduct, CosineExpanded, 等

**特点**:
- 精确搜索，100% 召回率
- 无需构建索引结构
- 支持预计算 norms 优化 L2/Cosine 距离
- 支持 Tiling 策略处理大规模数据
- 支持 Bitset/Bitmap 过滤
- 支持稀疏向量变体 (CSR 格式)

---

## 二、Build 流程总结

```
build() 入口
    │
    ├── 1. 参数验证
    │
    ├── 2. 数据集处理
    │   ├── 如果是列主序: 转置为行主序
    │   └── 如果在 Host: 拷贝到 Device
    │
    ├── 3. 预计算范数 (L2/Cosine 度量)
    │   ├── L2Expanded: 平方范数
    │   └── CosineExpanded: L2 范数 (带 sqrt)
    │
    └── 4. 构建 index 对象返回
```

---

## 三、Build 详细步骤

### 3.1 数据集处理

**文件**: `brute_force.cu:26-161`

```cpp
// 情况 1: Device 数据，行主序 - 直接使用视图
index(res, dataset_view, metric, metric_arg);

// 情况 2: Device 数据，列主序 - 转置
if constexpr (!row_major) {
  auto dataset_t = raft::make_device_matrix<T, int64_t>(res, n_rows, n_cols);
  raft::linalg::transpose(res, dataset, dataset_t.view());
}

// 情况 3: Host 数据 - 拷贝到 Device
auto dataset_device = raft::make_device_matrix<T, int64_t>(res, n_rows, n_cols);
raft::copy(res, dataset_device.view(), dataset);
```

### 3.2 范数预计算

**文件**: `knn_brute_force.cuh:788-813`

**目的**: 避免搜索时重复计算，利用公式优化距离计算

```
L2 距离: ||a - b||² = ||a||² + ||b||² - 2·(a·b)
                      ↑预计算   ↑预计算   ↑GEMM
```

```cpp
// 对 L2 度量预计算平方范数
if (metric == L2Expanded || metric == L2SqrtExpanded) {
  raft::linalg::norm<L2Norm, ALONG_ROWS>(res, dataset_view, norms->view());
}

// 对 Cosine 度量预计算 L2 范数 (带 sqrt)
if (metric == CosineExpanded) {
  raft::linalg::norm<L2Norm, ALONG_ROWS>(res, dataset_view, norms->view(), sqrt_op{});
}
```
**内存**: Main Pool, `N × sizeof(DistT)`

---

## 四、Search 流程总结

```
search() 入口
    │
    ├── 1. 计算 tile 大小
    │   └── chooseTileSize()
    │
    ├── 2. 预计算查询范数 (如果需要)
    │
    ├── 3. 分块搜索循环
    │   └── for 行 tile (queries):
    │       └── for 列 tile (dataset):
    │           ├── pairwise_distance()
    │           ├── 距离后处理 (norm fusion)
    │           ├── 应用过滤器 (可选)
    │           └── select_k() (tile 内 top-k)
    │
    ├── 4. 跨 tile 合并 (merge_tile_results)
    │
    └── 5. 输出最终 top-k 结果
```

---

## 五、Search 详细步骤

### 5.1 Tile 大小选择算法

**文件**: `faiss_distance_utils.h:13-51`

```cpp
void chooseTileSize(numQueries, numCentroids, dim, elementSize,
                    tileRows, tileCols)
{
    // 1. 确定首选行数
    uint32_t preferredTileRows = 512;
    if (dim <= 32) preferredTileRows = 1024;  // 小维度可以处理更多行
    tileRows = min(preferredTileRows, numQueries);

    // 2. 确定列数 (数据集分块大小)
    if (tileRows * numCentroids * elementSize * 2 <= 512 MB) {
        tileCols = numCentroids;  // 数据集足够小，不分块
    } else {
        // 根据 GPU 显存确定目标内存使用
        size_t targetUsage;
        if (gpuMemory > 8GB)      targetUsage = 1 GB;
        else if (gpuMemory > 4GB) targetUsage = 768 MB;
        else                      targetUsage = 512 MB;

        tileCols = min(targetUsage / (2 * elementSize * tileRows), numCentroids);
    }

    // 3. 约束: tileCols >= k
    tileCols = max(tileCols, k);
}
```

**设计原理**: 双缓冲策略使用 2x 内存存储距离/索引临时缓冲

### 5.2 Tiled Brute Force 搜索

**文件**: `knn_brute_force.cuh:73-321`

```
Tiled 搜索策略:
┌─────────────────────────────────────────────────────────────────┐
│                    数据集 [N × dim]                              │
├─────────────────────────────────────────────────────────────────┤
│  tile_0  │  tile_1  │  tile_2  │    ...    │  tile_n            │
│[tile_cols]│[tile_cols]│[tile_cols]│          │                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  对每个 tile:                                                    │
│  1. 计算距离矩阵 [tile_rows × tile_cols]                         │
│  2. 应用范数后处理 (L2/Cosine)                                   │
│  3. 应用过滤器 (如果启用)                                        │
│  4. select_k() 选择 tile 内 top-k                               │
├─────────────────────────────────────────────────────────────────┤
│  跨 tile 合并:                                                   │
│  - 临时缓冲: [tile_rows × k × num_col_tiles]                    │
│  - 最终 select_k() 从所有 tile 结果中选 top-k                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 距离计算与后处理

**文件**: `knn_brute_force.cuh:181-238`

```cpp
// 核心距离计算
cuvs::distance::pairwise_distance(res, queries_tile, dataset_tile,
                                   distances_tile, metric);
```

**L2 距离融合范数**:
```cpp
// dist[i,j] = query_norms[i] + dataset_norms[j] - 2 * inner_product[i,j]
raft::linalg::map_offset(res, distances,
    [query_norms, dataset_norms, n_cols] __device__(auto idx, auto d) {
        auto row = idx / n_cols;
        auto col = idx % n_cols;
        return query_norms[row] + dataset_norms[col] + d;  // d = -2*inner_product
    }, distances);
```

**Cosine 距离归一化**:
```cpp
// dist[i,j] = 1.0 - inner_product[i,j] / (query_norm[i] * dataset_norm[j])
raft::linalg::map_offset(res, distances,
    [query_norms, dataset_norms, n_cols] __device__(auto idx, auto d) {
        return 1.0f - d / (query_norms[row] * dataset_norms[col]);
    }, distances);
```

### 5.4 Top-K 选择

**文件**: `knn_brute_force.cuh:265-275`

```cpp
// Tile 内 top-k
bool select_min = is_min_close(metric);  // L2 选最小，InnerProduct 选最大
cuvs::selection::select_k(res, distances_tile, indices_tile, top_k, select_min);

// 跨 tile 合并
if (num_col_tiles > 1) {
    // 从 [tile_rows × k × num_col_tiles] 中选最终 top-k
    cuvs::selection::select_k(res, temp_distances, temp_indices, k, select_min);
}
```

### 5.5 Filter 支持

**文件**: `knn_brute_force.cuh:245-261`

| 类型 | 维度 | 说明 |
|------|------|------|
| **Bitset** | `[N / 32]` | 所有查询共享同一过滤器 |
| **Bitmap** | `[n_queries × N / 32]` | 每个查询独立过滤器 |

```cpp
// 过滤器应用逻辑
if (filter_bits != nullptr) {
    auto item_idx = g_idx >> 5;           // 除以 32
    auto bit_idx = g_idx & 31;            // 模 32
    if ((filter_bits[item_idx] & (1u << bit_idx)) == 0) {
        distances_ptr[idx] = masked_distance;  // 设为无穷大/最小值
    }
}
```

### 5.6 高稀疏度过滤优化

**文件**: `knn_brute_force.cuh:581-737`

```cpp
// 决策逻辑
if (sparsity < 0.9) {
    // 低稀疏度: 使用标准 tiled brute force + 过滤
    tiled_brute_force_knn(..., filter_bits, ...);
} else {
    // 高稀疏度 (>90%): 转换为 CSR 稀疏矩阵处理
    convert_bitmap_to_csr(filter_bits, csr_matrix);
    sparse_knn_search(csr_matrix);
}
```

### 5.7 稀疏向量搜索

**文件**: `sparse_knn.cuh:117-434`

```
稀疏搜索算法:
┌─────────────────────────────────────────────────────────────────┐
│  1. Query 批处理 (默认 batch_size = 262144 元素)                 │
│     └── csr_batcher_t: 按行切分 CSR 矩阵                        │
│                                                                 │
│  2. Index 批处理 (同上)                                          │
│                                                                 │
│  3. 双重循环:                                                    │
│     for query_batch in query_batches:                           │
│       for index_batch in index_batches:                         │
│         ├── 切片 CSR 行                                         │
│         ├── pairwise_distance() (稀疏版本)                      │
│         └── select_k() (批次内 top-k)                           │
│                                                                 │
│  4. 批次合并: knn_merge_parts()                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.8 Fused L2 KNN 优化

**文件**: `fused_l2_knn.cuh`

**启用条件**:
- `k <= 64`
- `rowMajorQuery == rowMajorIndex == true`
- 度量为 L2Expanded, L2SqrtExpanded, L2Unexpanded, 或 L2SqrtUnexpanded

**优化原理**:
- 融合距离计算 + top-k 选择
- 直接计算距离到堆结构
- 避免生成完整距离矩阵

---

## 六、内存使用总结

### 符号说明

| 符号 | 含义 |
|------|------|
| `N` | 数据集向量数量 |
| `dim` | 向量维度 |
| `n_queries` | 查询向量数量 |
| `k` | 返回的近邻数量 |
| `tile_rows` | 行方向 tile 大小 (queries) |
| `tile_cols` | 列方向 tile 大小 (dataset) |
| `num_col_tiles` | 列方向 tile 数量，`ceil(N / tile_cols)` |
| `nnz` | 稀疏矩阵非零元素数量 |
| `batch_size` | 稀疏搜索批处理大小 |
| `sizeof(T)` | 数据类型大小 (float=4, half=2) |
| `sizeof(DistT)` | 距离类型大小 (通常为 float=4) |
| `sizeof(IdxT)` | 索引类型大小 (通常为 int64_t=8) |

### 6.1 索引自身内存

| 组件 | 计算公式 |
|------|----------|
| `dataset_` | `N × dim × sizeof(T)` |
| `norms_` (可选) | `N × sizeof(DistT)` |

```
索引内存 = N × dim × sizeof(T) + N × sizeof(DistT)  (有范数预计算)
索引内存 = N × dim × sizeof(T)                       (无范数预计算)
```

### 6.2 Build 过程峰值内存

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Main Pool** | dataset_ | `N × dim × sizeof(T)` |
| | norms_ (L2/Cosine) | `N × sizeof(DistT)` |
| **Workspace Pool** | 范数计算临时空间 | 由 raft::linalg::rowNorm 决定 |
| **Large Workspace** | (未使用) | 0 |

```
Build 峰值 = N × dim × sizeof(T) + N × sizeof(DistT)
```

### 6.3 Search 过程峰值内存

**稠密搜索:**

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Workspace Pool** | 距离矩阵 tile | `tile_rows × tile_cols × sizeof(DistT)` |
| | 查询范数 | `n_queries × sizeof(DistT)` |
| | 临时 top-k 缓冲 | `tile_rows × k × num_col_tiles × (sizeof(DistT) + sizeof(IdxT))` |
| | Filter 缓冲 (可选) | Bitmap: `n_queries × N / 8`, Bitset: `N / 8` |

```
Search 峰值 ≈ tile_rows × tile_cols × sizeof(DistT)
            + n_queries × sizeof(DistT)
            + tile_rows × k × num_col_tiles × (sizeof(DistT) + sizeof(IdxT))
```

**稀疏搜索:**

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Workspace Pool** | Query 批次 CSR | `(batch_rows + 1 + query_nnz × 2) × sizeof(value_t)` |
| | Index 批次 CSR | `(batch_rows + 1 + index_nnz × 2) × sizeof(value_t)` |
| | 距离 tile | `batch_query_rows × batch_index_rows × sizeof(value_t)` |
| | 合并缓冲 | `batch_rows × k × 3 × (sizeof(DistT) + sizeof(IdxT))` |

```
Sparse Search 峰值 ≈ batch_query_rows × batch_index_rows × sizeof(value_t)
                   + batch_rows × k × 3 × (sizeof(DistT) + sizeof(IdxT))
                   + CSR 缓冲
```

