# IVF-Flat 索引详细分析

## 一、简介

IVF-Flat (Inverted File with Flat vectors) 是基于倒排索引的近似最近邻搜索算法，通过 K-Means 聚类将数据划分到多个列表，搜索时只在部分列表中进行精确距离计算。

**支持的数据类型**: float, half (float16), int8_t, uint8_t

**支持的距离度量**: L2Expanded, InnerProduct, CosineExpanded (仅 float/half)

**特点**:
- 聚类中心 + 倒排列表结构
- 存储原始向量，精确距离计算
- 支持动态扩展 (extend)
- 支持 Interleaved 内存布局优化

---

## 二、Build 流程总结

```
build() 入口
    │
    ├── 1. 准备训练数据
    │   ├── 采样训练集 (如果数据集过大)
    │   └── 复制到 workspace/large_workspace
    │
    ├── 2. K-Means 聚类训练
    │   ├── balanced_hierarchical_kmeans (当 n_lists > threshold)
    │   │   ├── 先聚类成 sqrt(n_lists) 个中间簇
    │   │   └── 再细分每个中间簇
    │   └── 或 flat_kmeans (当 n_lists 较小)
    │
    ├── 3. 聚类分配
    │   └── 为每个向量分配最近的聚类中心
    │
    ├── 4. 构建倒排列表
    │   └── 将向量按聚类 ID 组织到 lists_[k]
    │
    └── 5. 返回 index 对象
```

---

## 三、Build 详细步骤

### 3.1 准备训练数据

**文件**: `ivf_flat_build.cuh:76-112`

```cpp
// 采样训练集大小
int64_t n_rows_train = std::min<int64_t>(n_rows, params.kmeans_n_iters * n_lists);

// 如果需要采样
if (n_rows_train < n_rows) {
    train_set = sample_dataset(dataset, n_rows_train);
} else {
    train_set = dataset;
}
```

### 3.2 K-Means 聚类训练

**文件**: `kmeans_balanced.cuh`

#### 3.2.1 层次聚类 (n_lists > threshold)

```
层次聚类流程:
┌─────────────────────────────────────────────────────────────────┐
│  n_mesoclusters = sqrt(n_lists)                                 │
│                                                                 │
│  Phase 1: 训练中间聚类                                           │
│    kmeans(trainset, n_mesoclusters) → mesocluster_centers       │
│                                                                 │
│  Phase 2: 分配到中间聚类                                         │
│    for each sample:                                             │
│      mesocluster_id = argmin(dist(sample, mesocluster_centers)) │
│                                                                 │
│  Phase 3: 每个中间聚类内部再聚类                                 │
│    for each mesocluster m:                                      │
│      n_sub = n_lists / n_mesoclusters                          │
│      kmeans(samples_in_m, n_sub) → sub_centers                 │
│                                                                 │
│  Phase 4: 合并所有子聚类中心                                     │
│    centers = concat(all sub_centers)                            │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 K-Means 迭代

**文件**: `kmeans.cuh:200-350`

```cpp
for (iter = 0; iter < max_iters; iter++) {
    // 1. 分配: 计算每个样本到所有中心的距离，找最近中心
    compute_distance(samples, centers, distances);
    argmin_along_rows(distances, labels);

    // 2. 更新: 计算新的聚类中心
    compute_new_centers(samples, labels, new_centers);

    // 3. 收敛检查
    if (centers_unchanged(centers, new_centers)) break;
    centers = new_centers;
}
```

### 3.3 聚类分配

**文件**: `ivf_flat_build.cuh:150-180`

```cpp
// 为数据集中每个向量分配聚类 ID
auto labels = raft::make_device_vector<uint32_t>(res, n_rows);

// 批量处理
for (batch_offset = 0; batch_offset < n_rows; batch_offset += batch_size) {
    pairwise_distance(batch, centers, distances);
    argmin(distances, labels_batch);
}
```

### 3.4 构建倒排列表

**文件**: `ivf_flat_build.cuh:200-280`

```cpp
// 1. 统计每个聚类的大小
auto list_sizes = count_per_cluster(labels, n_lists);

// 2. 分配倒排列表内存
for (k = 0; k < n_lists; k++) {
    lists_[k].data = allocate(list_sizes[k] * dim * sizeof(T));
    lists_[k].indices = allocate(list_sizes[k] * sizeof(IdxT));
}

// 3. 填充倒排列表 (使用 CUB::DevicePartition 按聚类 ID 重排数据)
partition_by_cluster(dataset, labels, lists_);
```

### 3.5 Interleaved 内存布局

**文件**: `ivf_list.hpp:150-220`

```
标准布局 vs Interleaved 布局:

标准布局 (row-major):
┌──────────────────────────────────┐
│ v0[d0] v0[d1] ... v0[dim-1]      │  vector 0
│ v1[d0] v1[d1] ... v1[dim-1]      │  vector 1
│ ...                              │
└──────────────────────────────────┘

Interleaved 布局 (32-vector groups):
┌──────────────────────────────────┐
│ v0[d0] v1[d0] ... v31[d0]        │  dim 0, vectors 0-31
│ v0[d1] v1[d1] ... v31[d1]        │  dim 1, vectors 0-31
│ ...                              │
│ v0[dim-1] ... v31[dim-1]         │  dim dim-1, vectors 0-31
├──────────────────────────────────┤
│ v32[d0] v33[d0] ... v63[d0]      │  next group
│ ...                              │
└──────────────────────────────────┘

优势: 一个 warp 的 32 个线程可以 coalesced 加载 32 个向量的同一组维度
```

---

## 四、Search 流程总结

```
search() 入口
    │
    ├── 1. 计算 batch size
    │   └── 根据 workspace 剩余空间计算 max_queries
    │
    ├── 2. 对每个 batch 执行 search_impl
    │   ├── 2.1 粗搜索: 找到最近的 n_probes 个 clusters
    │   │   ├── 计算 query norms
    │   │   ├── GEMM: queries × centers^T
    │   │   ├── L2/Cosine 后处理
    │   │   └── select_k: 选择 top-n_probes
    │   │
    │   ├── 2.2 细搜索: 在选中的 clusters 内搜索
    │   │   └── ivfflat_interleaved_scan: 计算距离 + local topk
    │   │
    │   └── 2.3 合并结果
    │       ├── select_k: 合并所有 probes 的结果
    │       └── postprocess_neighbors: 转换为全局 ID
    │
    └── 3. 输出最终 top-k 结果
```

---

## 五、Search 详细步骤

### 5.1 搜索参数与 Batch Size 计算

**文件**: `ivf_flat_search.cuh:341-352`

```cpp
void search_with_filtering(handle, params, index, queries, ...) {
    uint32_t n_probes = min(params.n_probes, index.n_lists());
    bool manage_local_topk = is_local_topk_feasible(k);  // k <= 1024

    // 估算每个 query 需要的 workspace 大小
    uint64_t ws_size_per_query =
        4 * (2 * n_probes + n_lists + dim + 1) +  // 基础缓冲
        (manage_local_topk
            ? ((sizeof(IdxT) + 4) * n_probes * k)      // local topk 模式
            : (4 * (max_samples + n_probes + 1)));    // 全量模式

    // 根据 workspace 剩余空间计算 batch size
    max_queries = available_workspace / ws_size_per_query;
}
```

### 5.2 粗搜索 - 选择探测聚类

**文件**: `ivf_flat_search.cuh:200-250`

```cpp
// 1. 复制查询到对齐内存
auto float_queries = raft::make_device_matrix<float>(res, batch_size, dim);

// 2. 计算查询范数 (L2/Cosine)
auto query_norms = raft::make_device_vector<float>(res, batch_size);
raft::linalg::rowNorm(query_norms.view(), float_queries.view());

// 3. GEMM 计算查询到所有中心的内积
// distances = -2 * queries @ centers^T (for L2)
raft::linalg::gemm(float_queries, centers_T, distances, -2.0f, 0.0f);

// 4. L2 后处理: dist = query_norm + center_norm + inner_product
raft::linalg::map(distances, [query_norms, center_norms](i, j, d) {
    return query_norms[i] + center_norms[j] + d;
});

// 5. 选择 top-n_probes 个聚类
select_k(distances, coarse_indices, coarse_distances, n_probes);
```

### 5.3 细搜索 - Interleaved Scan

**文件**: `ivf_flat_interleaved_scan.cuh:200-400`

```
Interleaved Scan Kernel 设计:
┌─────────────────────────────────────────────────────────────────┐
│  gridDim.x = n_queries × n_probes                               │
│  blockDim.x = 根据 dim 选择 (32, 64, 128, 256, 512)             │
│                                                                 │
│  Shared Memory 布局:                                            │
│  ├─ query_smem[dim]           查询向量缓存                      │
│  └─ topk buffer               当前 top-k                        │
│                                                                 │
│  算法:                                                          │
│  1. 加载查询向量到 shared memory                                 │
│  2. for each vector_group in cluster (32 vectors):             │
│       a. 从 interleaved 布局加载 32 个向量                       │
│       b. 计算 32 个距离 (warp-level reduce)                     │
│       c. 更新 top-k heap                                        │
│  3. 写入 top-k 到全局内存                                        │
└─────────────────────────────────────────────────────────────────┘
```

**Warp-level 距离计算**:
```cpp
// 每个 warp 计算 32 个向量的距离
for (int d = lane_id; d < dim; d += 32) {
    float q = query_smem[d];
    float v = list_data[vector_offset + d];  // Interleaved 加载
    sum += (q - v) * (q - v);  // L2
}
// Warp reduce
sum = warp_reduce_sum(sum);
```

### 5.4 结果合并与后处理

**文件**: `ivf_flat_search.cuh:300-350`

```cpp
// 1. 合并所有 probe 的 local topk 结果
// 输入: [n_queries, n_probes, k] 的局部结果
// 输出: [n_queries, k] 的全局结果
select_k(
    local_topk_distances,   // [n_queries, n_probes * k]
    local_topk_indices,
    final_distances,
    final_neighbors,
    k
);

// 2. 将局部索引转换为全局索引
postprocess_neighbors(
    final_neighbors,          // 输入: (cluster_id, local_offset)
    index.inds_ptrs(),        // 每个 cluster 的索引数组
    global_neighbors          // 输出: 全局向量 ID
);
```

---

## 六、内存使用总结

### 符号说明

| 符号 | 含义 |
|------|------|
| `N` | 数据集向量数量 |
| `dim` | 向量维度 |
| `dim_ext` | 扩展维度 (对齐后) |
| `n_lists` | 聚类数量 (倒排列表数) |
| `n_probes` | 搜索时探测的聚类数量 |
| `n_queries` | 查询向量数量 |
| `k` | 返回的近邻数量 |
| `n_train` | 训练集样本数 |
| `n_mesoclusters` | 层次聚类的中间聚类数，`sqrt(n_lists)` |
| `list_size[i]` | 第 i 个倒排列表的向量数 |
| `max_samples` | 探测聚类中的最大总样本数 |
| `sizeof(T)` | 数据类型大小 (float=4, half=2, int8=1, uint8=1) |
| `sizeof(IdxT)` | 索引类型大小 (int64_t=8, uint32_t=4) |

### 6.1 索引自身内存

| 组件 | 计算公式 |
|------|----------|
| `centers_` | `n_lists × dim × sizeof(float)` |
| `center_norms_` (L2 度量) | `n_lists × sizeof(float)` |
| `lists_[k].data` (所有列表) | `N × dim × sizeof(T)` |
| `lists_[k].indices` (所有列表) | `N × sizeof(IdxT)` |

```
索引内存 = n_lists × dim × sizeof(float)              (centers)
         + n_lists × sizeof(float)                    (center_norms, 如果 L2)
         + N × dim × sizeof(T)                        (所有向量数据)
         + N × sizeof(IdxT)                           (所有索引)
```

### 6.2 Build 过程峰值内存

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Main Pool** | centers_ | `n_lists × dim × sizeof(float)` |
| | center_norms_ | `n_lists × sizeof(float)` |
| | lists_[k].data | `N × dim × sizeof(T)` |
| | lists_[k].indices | `N × sizeof(IdxT)` |
| **Workspace / Large Workspace** | trainset | `n_train × dim × sizeof(T)` |
| | labels | `n_train × sizeof(uint32_t)` |
| | distances (K-Means) | `batch_size × n_lists × sizeof(float)` |
| | mesocluster_centers | `n_mesoclusters × dim × sizeof(float)` |
| **Host Memory** | labels_host | `n_train × sizeof(uint32_t)` |

```
Build 峰值 ≈ 索引自身内存
           + n_train × dim × sizeof(T)                     (trainset)
           + n_train × sizeof(uint32_t)                    (labels)
           + batch_size × n_lists × sizeof(float)          (distances)
           + n_mesoclusters × dim × sizeof(float)          (mesoclusters)
```

### 6.3 Search 过程峰值内存

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Workspace Pool** | float_queries | `batch_size × dim × sizeof(float)` |
| | query_norms | `batch_size × sizeof(float)` |
| | distances | `batch_size × n_lists × sizeof(float)` |
| | coarse_indices | `batch_size × n_probes × sizeof(uint32_t)` |
| | coarse_distances | `batch_size × n_probes × sizeof(float)` |
| | local_topk (如果 k<=1024) | `batch_size × n_probes × k × (sizeof(float) + sizeof(IdxT))` |
| **Shared Memory** | query_smem | `dim × sizeof(float)` |
| | topk buffer | `k × (sizeof(float) + sizeof(IdxT))` |

```
Search 峰值 ≈ batch_size × dim × sizeof(float)             (float_queries)
            + batch_size × n_lists × sizeof(float)          (distances)
            + batch_size × n_probes × sizeof(uint32_t)      (coarse_indices)
            + batch_size × n_probes × k × 12                (local_topk, 如果启用)
```

