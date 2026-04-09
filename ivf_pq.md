# IVF-PQ 索引详细分析

## 一、简介

IVF-PQ (Inverted File with Product Quantization) 是一种结合倒排索引和乘积量化的高效向量搜索算法。通过两级量化大幅压缩存储空间，同时使用查找表加速距离计算。

**支持的数据类型**: float, half (float16), int8_t, uint8_t

**支持的距离度量**: L2Expanded, InnerProduct, CosineExpanded (int8/uint8 暂不支持 Cosine)

**特点**:
- 两级量化: IVF 粗量化 + PQ 细量化
- 高压缩比: 每向量仅需 `pq_dim × pq_bits / 8` 字节
- LUT 加速: 搜索时只需 `pq_dim` 次查表累加
- 支持可选旋转矩阵提高量化精度
- 支持 PER_SUBSPACE 和 PER_CLUSTER 两种码本模式

**核心参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_lists` | 1024 | 聚类数量（第一级量化） |
| `pq_bits` | 8 | PQ 编码位宽 [4,5,6,7,8] |
| `pq_dim` | 自动 | PQ 子空间数量 |
| `codebook_kind` | PER_SUBSPACE | 码本生成方式 |
| `kmeans_n_iters` | 20 | K-Means 迭代次数 |

**PQ 量化原理**:
```
原始向量 y 的两级近似：
y ≈ Q₁(y) + Q₂(y - Q₁(y))
     ↓           ↓
  聚类中心    残差的PQ编码

向量分解: 将 dim 维向量分成 pq_dim 个子向量，每个子向量长度 pq_len = dim / pq_dim

原始向量: [y₁, y₂, ..., y_dim]
          ↓ 分割成 pq_dim 个子空间
子向量:   [u₁], [u₂], ..., [u_pq_dim]
          ↓ 每个子向量量化为 pq_bits 位码
PQ码:     [c₁], [c₂], ..., [c_pq_dim]   (总共 pq_dim * pq_bits 位)
```

---

## 二、Build 流程总结

```
build() 入口
    │
    ├── 1. 采样训练集
    │   └── sample_rows(): 从数据集中采样
    │
    ├── 2. 训练第一级量化器 (Hierarchical Balanced K-Means)
    │   ├── kmeans_balanced::fit() 训练聚类中心
    │   └── kmeans_balanced::predict() 预测训练集标签
    │
    ├── 3. 生成旋转矩阵
    │   ├── 如果 dim % pq_dim != 0 或 force_random_rotation:
    │   │   └── 生成随机正交矩阵 (QR分解)
    │   └── 否则: 单位矩阵
    │
    ├── 4. 训练 PQ 码本
    │   ├── PER_SUBSPACE 模式: 对每个子空间独立训练码本
    │   └── PER_CLUSTER 模式: 对每个聚类独立训练码本
    │
    └── 5. 添加数据 (if add_data_on_build)
        └── 调用 extend() 将所有向量编码并存入索引
```

---

## 三、Build 详细步骤

### 3.1 采样训练集

**文件**: `ivf_pq_build.cuh:1362-1394`

```cpp
// 计算采样比例
trainset_ratio = n_rows / (kmeans_trainset_fraction * n_rows);
n_rows_train = n_rows / trainset_ratio;

// 采样训练数据
sample_rows(res, dataset, trainset, trainset_ratio);
```

**内存**: Workspace Pool, `n_rows_train × dim × sizeof(float)`

### 3.2 训练第一级量化器

**文件**: `ivf_pq_build.cuh:1414-1415`

使用分层平衡 K-Means 算法训练聚类中心，与 IVF-Flat 相同。

```cpp
// 训练聚类中心
kmeans_balanced::fit(res, trainset, n_lists, cluster_centers, ...);

// 预测训练集标签
kmeans_balanced::predict(res, cluster_centers, trainset, labels, ...);
```

**内存**:
- `cluster_centers`: Workspace Pool, `n_lists × dim × sizeof(float)`
- `labels`: Large Workspace, `n_rows_train × sizeof(uint32_t)`

### 3.3 生成旋转矩阵

**文件**: `ivf_pq_build.cuh:1434`

旋转矩阵用于在量化前变换向量，可以提高 PQ 量化精度。

```cpp
// 决策逻辑
if (dim % pq_dim != 0 || force_random_rotation) {
    // 生成随机正交矩阵
    make_rotation_matrix(res, rot_dim, dim, rotation_matrix);
} else {
    // 使用单位矩阵 (不旋转)
    raft::linalg::eye(res, rotation_matrix);
}
```

**随机正交矩阵生成**:
1. 生成随机矩阵
2. QR 分解获得正交矩阵 Q
3. `rotation_matrix = Q`

**内存**: Main Pool (索引成员), `rot_dim × dim × sizeof(float)`

### 3.4 训练 PQ 码本

#### PER_SUBSPACE 模式

**文件**: `ivf_pq_build.cuh:316`

```
PER_SUBSPACE 训练流程:
┌─────────────────────────────────────────────────────────────────┐
│  for j in 0..pq_dim:  (对每个子空间)                             │
│    1. 旋转训练集                                                 │
│       rotated = rotation_matrix × trainset                      │
│                                                                 │
│    2. 计算旋转残差的第j个子空间                                  │
│       sub_trainset = rotated[:, j*pq_len:(j+1)*pq_len]          │
│                    - centers_rot[:, j*pq_len:(j+1)*pq_len]      │
│                                                                 │
│    3. K-Means 训练 pq_book_size 个码字                          │
│       pq_centers[j, :, :] = kmeans_fit(sub_trainset)            │
└─────────────────────────────────────────────────────────────────┘

码本维度: [pq_dim, pq_len, pq_book_size]
```

#### PER_CLUSTER 模式

**文件**: `ivf_pq_build.cuh:398`

```
PER_CLUSTER 训练流程:
┌─────────────────────────────────────────────────────────────────┐
│  for l in 0..n_lists:  (对每个聚类)                              │
│    1. 选择属于 cluster l 的训练样本                              │
│       cluster_samples = trainset[labels == l]                   │
│                                                                 │
│    2. 计算旋转残差 (相对于 cluster l 的中心)                    │
│       rotated = rotation_matrix × cluster_samples               │
│       residuals = rotated - centers_rot[l]                      │
│                                                                 │
│    3. 将残差展平为 [cluster_size * pq_dim, pq_len]              │
│                                                                 │
│    4. K-Means 训练码本                                          │
│       pq_centers[l, :, :] = kmeans_fit(flattened_residuals)     │
└─────────────────────────────────────────────────────────────────┘

码本维度: [n_lists, pq_len, pq_book_size]
```

**内存**:
- `pq_centers_tmp`: Workspace Pool, `pq_dim × pq_len × pq_book_size × sizeof(float)`
- `sub_trainset`: Workspace Pool, `pq_n_rows × pq_len × sizeof(float)` (循环内)

### 3.5 Extend - 数据编码

**文件**: `ivf_pq_build.cuh` (extend 函数)

```
extend() 流程:
┌─────────────────────────────────────────────────────────────────┐
│  输入: new_vectors [n_rows, dim], new_indices [n_rows]          │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: 分配标签                                                │
│    - 批量调用 kmeans_balanced::predict() 预测每个向量的 cluster │
│    - new_data_labels [n_rows]                                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: 统计并调整 list 容量                                    │
│    - histogram() 统计每个 cluster 新增向量数                    │
│    - resize_list() 为每个 list 分配足够容量                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: 编码并填充 (process_and_fill_codes)                    │
│    批量处理:                                                     │
│    1. 旋转向量: rotated = rotation_matrix × vectors             │
│    2. 计算残差: residual = rotated - centers_rot[label]         │
│    3. PQ 编码: 对每个子空间找最近的码字                         │
│    4. 写入 list: 交错格式存储                                   │
└─────────────────────────────────────────────────────────────────┘
```

**PQ 编码过程** (`ivf_pq_process_and_fill_codes.cuh:72-137`):

```cpp
// encode_vectors::operator()(i, j) - 编码第 i 个向量的第 j 个子空间
__device__ auto operator()(IdxT i, uint32_t j) -> uint8_t {
    // 选择码本分区
    uint32_t partition_ix = (codebook_kind == PER_CLUSTER) ? cluster_ix : j;

    // 遍历所有 pq_book_size 个码字，找最近的
    float min_dist = INF;
    uint8_t code = 0;
    for (uint32_t l = lane_id; l < pq_book_size; l += SubWarpSize) {
        float d = 0.0f;
        for (uint32_t k = 0; k < pq_len; k++) {
            auto t = in_vectors(i, j, k) - pq_centers(partition_ix, k, l);
            d += t * t;  // L2 距离
        }
        if (d < min_dist) { min_dist = d; code = l; }
    }
    // warp 内归约找最小
    ...
    return code;  // 返回最近码字的索引 (pq_bits 位)
}
```

**交错存储格式** (`ivf_pq.hpp:226-287`):
```cpp
// PQ 编码数据采用交错格式存储
// list_extents = [ceildiv(n_rows, 32), ceildiv(pq_dim, chunk), 32, 16]
//                 ↓                     ↓                      ↓   ↓
//               行分组               PQ维度分块            组大小 向量化长度
```

---

## 四、Search 流程总结

```
search() 入口
    │
    ├── 1. Outer Loop: 批处理 queries (max_bs_outer)
    │
    ├── 2. Coarse Search - 选择要探测的 clusters
    │   ├── 转换 queries 类型
    │   ├── GEMM 计算 query-center 距离
    │   └── select_k 选择 top-n_probes 个 cluster
    │
    ├── 3. 旋转 queries
    │   └── rot_queries = rotation_matrix × queries
    │
    ├── 4. Inner Loop: 批处理 queries (max_bs_inner)
    │
    ├── 5. Fine Search - LUT-based 距离计算
    │   ├── 预计算 base_diff (可选)
    │   ├── 构建查找表 (LUT)
    │   ├── 使用 LUT 计算每个样本的距离
    │   └── 局部 Top-K (可选)
    │
    └── 6. 合并结果
        ├── 如果使用 local_topk: 合并 n_probes 个 probe 的 top-k
        └── 否则: 对所有距离进行全局 select_k
```

---

## 五、Search 详细步骤

### 5.1 Coarse Search - 粗搜索

**文件**: `ivf_pq_search.cuh`

```cpp
// 1. 转换 queries 类型 → float/half/int8
auto gemm_queries = workspace.alloc<float>(max_bs_outer * dim_ext);

// 2. GEMM 计算 query-center 距离
// distances = queries × centers^T
raft::linalg::gemm(res, queries, centers.T(), coarse_distances);

// 3. select_k 选择 top-n_probes 个 cluster
cuvs::selection::select_k(res, coarse_distances, clusters_to_probe, n_probes);
```

**内存**:
- `gemm_queries`: Workspace Pool, `max_bs_outer × dim_ext × sizeof(T)`
- `clusters_to_probe`: Workspace Pool, `max_bs_outer × n_probes × sizeof(uint32_t)`

### 5.2 旋转 Queries

```cpp
// 旋转查询向量
// rot_queries = rotation_matrix × queries
raft::linalg::gemm(res, rotation_matrix, queries.T(), rot_queries.T());
```

**内存**: `rot_queries`: Workspace Pool, `max_bs_outer × rot_dim × sizeof(float)`

### 5.3 Fine Search - LUT 距离计算

**文件**: `ivf_pq_compute_similarity_impl.cuh:385-430`

```
compute_similarity_kernel:
┌─────────────────────────────────────────────────────────────────┐
│  每个 block 处理一个 (query, probe) 对                           │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: 预计算 base_diff (可选)                                 │
│    base_diff[i] = query[i] - cluster_center[i]                  │
│    存储在 shared memory                                         │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: 构建查找表 (LUT)                                        │
│    for each pq_subspace j:                                      │
│      for each code c in [0, pq_book_size):                      │
│        lut[j * pq_book_size + c] = 子空间距离                   │
│    LUT 大小: pq_dim * pq_book_size                              │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: 计算每个样本的距离                                      │
│    for each sample in cluster:                                  │
│      score = Σ lut[j * pq_book_size + pq_code[sample][j]]       │
│            j=0..pq_dim                                          │
│      (只需 pq_dim 次查表累加!)                                  │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: 局部 Top-K (可选)                                       │
│    使用 warp_sort 在 block 内选择 top-k                         │
└─────────────────────────────────────────────────────────────────┘
```

**LUT 构建代码**:
```cpp
// 构建 LUT: 预计算 query 与所有 PQ 码字的距离
for (uint32_t i = threadIdx.x; i < lut_size; i += blockDim.x) {
    uint32_t i_pq = i >> PqBits;       // 子空间索引
    uint32_t code = i & PqMask;        // 码字索引
    uint32_t j = i_pq * pq_len;        // 维度起始

    float score = 0.0;
    for (k = 0; k < pq_len; k++) {
        // L2 距离
        float diff = (query[j+k] - cluster_center[j+k]) - pq_center[code][k];
        score += diff * diff;
    }
    lut_scores[i] = score;  // LUT[子空间][码字] = 距离
}
```

**LUT 使用代码**:
```cpp
// 使用 LUT 计算距离: 只需查表累加
__device__ auto ivfpq_compute_score(...) {
    float score = 0;
    for (j = 0; j < pq_dim; j++) {
        uint8_t code = pq_dataset[sample][j];  // 读取 PQ 码
        score += lut_scores[j * pq_book_size + code];  // 查表
    }
    return score;
}
```

**LUT 优化原理**:
```
┌─────────────────────────────────────────────────────────────────┐
│  不使用 LUT: 每个样本需要 pq_dim × pq_len 次浮点运算             │
│  使用 LUT:   每个样本只需要 pq_dim 次查表 + 累加                 │
│  加速比:     pq_len 倍                                          │
│                                                                 │
│  例如: dim=128, pq_dim=32, pq_len=4                             │
│  - 不使用 LUT: 128 次浮点运算/样本                               │
│  - 使用 LUT: 32 次查表+累加/样本                                 │
│  - 加速: 4 倍                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 结果合并

```cpp
// 如果使用 local_topk
if (local_topk_enabled) {
    // 合并 n_probes 个 probe 的 top-k 结果
    merge_topk_results(res, local_topk_results, final_neighbors, final_distances);
} else {
    // 对所有距离进行全局 select_k
    cuvs::selection::select_k(res, all_distances, neighbors, distances, k);
}
```

---

## 六、内存使用总结

### 符号说明

| 符号 | 含义 |
|------|------|
| `N` 或 `n_rows` | 数据集向量数量 |
| `dim` | 原始向量维度 |
| `rot_dim` | 旋转后的维度 (通常 = dim) |
| `dim_ext` | 扩展维度 (对齐后) |
| `pq_dim` | PQ 子空间数量 |
| `pq_len` | 每个子空间的维度，`rot_dim / pq_dim` |
| `pq_bits` | 每个 PQ 码的位数 (通常 4 或 8) |
| `pq_book_size` | 每个子空间的码本大小，`2^pq_bits` |
| `n_lists` | 聚类数量 (倒排列表数) |
| `n_probes` | 搜索时探测的聚类数量 |
| `n_queries` | 查询向量数量 |
| `k` | 返回的近邻数量 |
| `n_rows_train` | 训练集样本数 |
| `max_bs_outer` | 外层批处理大小 |
| `max_bs_inner` | 内层批处理大小 |
| `gridDim.x` | CUDA grid 的 x 维度 |
| `sizeof(T)` | 输入数据类型大小 (float=4, half=2, int8=1, uint8=1) |
| `sizeof(LutT)` | LUT 数据类型大小 (half=2, float=4) |
| `sizeof(IdxT)` | 索引类型大小 (int64_t=8, uint32_t=4) |

### 6.1 索引自身内存

| 组件 | 计算公式 |
|------|----------|
| `centers_` | `n_lists × dim_ext × sizeof(float)` |
| `centers_rot_` | `n_lists × rot_dim × sizeof(float)` |
| `center_norms_` (L2 度量) | `n_lists × sizeof(float)` |
| `pq_centers_` (PER_SUBSPACE) | `pq_dim × pq_len × pq_book_size × sizeof(float)` |
| `pq_centers_` (PER_CLUSTER) | `n_lists × pq_len × pq_book_size × sizeof(float)` |
| `rotation_matrix_` | `rot_dim × dim × sizeof(float)` (如果启用旋转) |
| `lists_[].codes` | `Σ(list_size_k) × ceil(pq_dim × pq_bits / 8)` |
| `lists_[].indices` | `Σ(list_size_k) × sizeof(IdxT)` |

```
索引内存 ≈ n_lists × dim_ext × sizeof(float)              (centers)
         + n_lists × rot_dim × sizeof(float)              (centers_rot)
         + n_lists × sizeof(float)                        (center_norms, 如果 L2)
         + pq_dim × pq_len × pq_book_size × sizeof(float) (pq_centers, PER_SUBSPACE)
         + rot_dim × dim × sizeof(float)                  (rotation_matrix, 可选)
         + N × ceil(pq_dim × pq_bits / 8)                 (所有 PQ 码)
         + N × sizeof(IdxT)                               (所有索引)
```

### 6.2 Build 过程峰值内存

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Workspace Pool** | `trainset` | `n_rows_train × dim × sizeof(float)` |
| | `cluster_centers` | `n_lists × dim × sizeof(float)` |
| | `pq_centers_tmp` | `pq_dim × pq_len × pq_book_size × sizeof(float)` |
| | `sub_trainset` | `pq_n_rows × pq_len × sizeof(float)` (循环内) |
| **Large Workspace** | `labels` | `n_rows_train × sizeof(uint32_t)` |
| | 大训练集 (如果数据量大) | 支持 managed memory 扩展 |

```
Build 峰值 ≈ n_rows_train × dim × sizeof(float)            (trainset)
           + n_lists × dim × sizeof(float)                 (cluster_centers)
           + pq_dim × pq_len × pq_book_size × sizeof(float) (pq_centers_tmp)
           + n_rows_train × sizeof(uint32_t)               (labels)
```

**Extend 过程内存**:

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Workspace Pool** | `new_data_labels` | `n_rows × sizeof(uint32_t)` |
| | `orig_list_sizes` | `n_lists × sizeof(uint32_t)` |
| | 批处理缓冲 | `batch_size × (dim + rot_dim) × sizeof(float)` |

### 6.3 Search 过程峰值内存

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Workspace Pool** | `gemm_queries` | `max_bs_outer × dim_ext × sizeof(T)` |
| | `rot_queries` | `max_bs_outer × rot_dim × sizeof(float)` |
| | `clusters_to_probe` | `max_bs_outer × n_probes × sizeof(uint32_t)` |
| | `lut_scores` (global) | `gridDim.x × pq_dim × pq_book_size × sizeof(LutT)` |
| **Shared Memory** | LUT (per block) | `pq_dim × pq_book_size × sizeof(LutT)` |
| | `base_diff` | `dim × sizeof(float)` (或 `dim × sizeof(double)` for IP) |
| | topk buffer | 依赖 k 和 warp_sort 配置 |

```
Search 峰值 ≈ max_bs_outer × dim_ext × sizeof(T)           (gemm_queries)
            + max_bs_outer × rot_dim × sizeof(float)       (rot_queries)
            + max_bs_outer × n_probes × sizeof(uint32_t)   (clusters_to_probe)
            + gridDim.x × pq_dim × pq_book_size × sizeof(LutT) (lut_scores)
```

**Shared Memory 布局**:
```
┌─────────────────────────────────────────────────────────────────┐
│  Shared Memory (per block):                                     │
│  ├── LUT: pq_dim × pq_book_size × sizeof(LutT)                 │
│  │        例: 32 × 256 × 2 = 16 KB (pq_dim=32, pq_bits=8, half)│
│  ├── base_diff: dim × sizeof(float)                            │
│  │        例: 128 × 4 = 512 bytes                              │
│  └── topk buffer: 依赖配置                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 附录: IVF-PQ 与 IVF-Flat 对比

| 特性 | IVF-Flat | IVF-PQ |
|------|----------|--------|
| 存储内容 | 原始向量 | PQ 压缩码 |
| 每向量存储 | `dim × sizeof(T)` | `pq_dim × pq_bits / 8` bytes |
| 距离计算 | 精确计算 | 查表近似 |
| Build 额外步骤 | 无 | PQ 码本训练 |
| Search 复杂度 | O(dim) | O(pq_dim) 查表 |
| 精度 | 精确 | 近似 (取决于 pq_bits/pq_dim) |

**压缩比示例** (dim=128, float):
- IVF-Flat: 128 × 4 = 512 bytes/向量
- IVF-PQ (pq_dim=32, pq_bits=8): 32 × 1 = 32 bytes/向量
- 压缩比: 16x
