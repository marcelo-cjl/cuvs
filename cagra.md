# CAGRA 索引详细分析

## 一、简介

CAGRA (Cuda Accelerated Graph-based nearest neighbor search) 是 cuVS 中基于图的近似最近邻搜索算法。

**支持的数据类型**: float, half (float16), int8_t, uint8_t

**支持的距离度量**: L2Expanded, InnerProduct

**特点**:
- 基于 k-NN 图构建的搜索结构
- 支持 IVF-PQ 或 NN-Descent 方式构建初始图
- 图优化 (剪枝) 提升搜索效率
- 支持 VPQ 压缩减少内存占用
- 高吞吐量 GPU 优化搜索

---

## 二、Build 流程总结

```
build() 入口
    │
    ├── 1. 参数验证
    │   └── intermediate_graph_degree >= graph_degree
    │
    ├── 2. 构建 k-NN 图 (选择一种方式)
    │   ├── IVF-PQ 方式 (默认): 构建索引 → 批量搜索 → 精排
    │   ├── NN-Descent 方式: 迭代优化图连接
    │   └── 迭代搜索方式: 初始小图 → CAGRA search 扩展
    │
    ├── 3. 图优化 (optimize)
    │   ├── MST 优化 (可选，保证连通性)
    │   ├── 2-hop detour 剪枝
    │   └── 合并正向/反向边
    │
    ├── 4. 数据集处理
    │   ├── IF compression: VPQ 压缩
    │   ├── ELIF attach_dataset: 复制到 GPU
    │   └── ELSE: 创建空索引
    │
    └── 5. 返回 index 对象
```

---

## 三、Build 详细步骤

### 3.1 初始化阶段

**文件**: `cagra_build.cuh:720-728`

```cpp
// 分配中间 KNN 图 (Host)
std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph(
  raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), intermediate_degree));
```
**内存**: Host, `N × intermediate_degree × sizeof(IdxT)`

### 3.2 IVF-PQ 图构建

**文件**: `cagra_build.cuh:122-384`

#### 3.2.1 参数计算

```cpp
const auto top_k = node_degree + 1;                              // 精排后保留的邻居数
uint32_t gpu_top_k = node_degree * pq.refinement_rate;           // IVF-PQ 返回的候选数
uint32_t max_queries = pq.search_params.max_internal_batch_size; // 批次大小，默认 1024
```

#### 3.2.2 Workspace 规划

**文件**: `cagra_build.cuh:175-214`

```cpp
// 计算期望的 workspace 大小
auto desired_workspace_size =
  max_queries * (sizeof(DataT) * dim           // queries 批次
               + sizeof(float) * gpu_top_k     // distances
               + sizeof(int64_t) * gpu_top_k   // neighbors
               + sizeof(float) * top_k         // refined_distances
               + sizeof(int64_t) * top_k);     // refined_neighbors

// 根据可用内存调整批次大小或切换到 Large Workspace
if (free_space_ratio < kMinWorkspaceRatio) {
    if (adjusted_max_queries >= kMinLargeBatchSize) {
        max_queries = adjusted_max_queries;  // 减小批次大小
    } else {
        use_large_workspace = true;          // 切换到 Large Workspace
    }
}
```

#### 3.2.3 批处理循环

```
批处理循环:
┌─────────────────────────────────────────────────────────────────┐
│  for (batch in dataset):                                        │
│    │                                                            │
│    ├── ivf_pq::search(batch, gpu_top_k)                        │
│    │   └── 返回每个查询的 gpu_top_k 个候选                       │
│    │                                                            │
│    ├── cuvs::neighbors::refine()                               │
│    │   └── 从 gpu_top_k 精排选 top_k                           │
│    │                                                            │
│    └── write_to_graph(knn_graph, refined_neighbors, offset)    │
│        └── 排除自边，写入邻居                                   │
│                                                                 │
│  返回: knn_graph [N, intermediate_degree] (Host)                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 图优化 (Pruning)

**文件**: `graph_core.cuh:1174-1596`

#### 3.3.1 优化流程

```
optimize() 入口
    │
    ├── MST 优化 (如果 guarantee_connectivity)
    │   └── 迭代添加边保证连通性
    │
    ├── 2-hop Detour 剪枝
    │   ├── 分配 detour_count [N, knn_graph_degree]
    │   ├── kern_prune<<<>>> 计算每条边的 2-hop detour 数
    │   └── 选择 detour 最少的边
    │
    ├── 构建反向图
    │   └── kern_make_rev_graph<<<>>>
    │
    └── 合并图
        ├── 优先保留 MST 边
        └── 合并正向/反向边
```

#### 3.3.2 2-hop Detour 剪枝算法

```
2-hop Detour 剪枝原理:
┌─────────────────────────────────────────────────────────────────┐
│  对于边 A → B:                                                   │
│    检查是否存在 2-hop 路径 A → D → B (D 是 A 的其他邻居)          │
│    如果存在，则边 A → B 可能是冗余的                              │
│                                                                 │
│       A ────────► B                                             │
│        \         ▲                                              │
│         \       /                                               │
│          ► D ─►                                                 │
│                                                                 │
│  detour_count[A→B] = 存在多少个这样的 D                          │
│  保留 detour_count 最小的边 (最"直接"的连接)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、Search 流程总结

```
search() 入口
    │
    ├── 1. 参数调整
    │   ├── 选择算法 (SINGLE_CTA / MULTI_CTA)
    │   └── 计算 hash 表参数
    │
    ├── 2. 初始化 Search Plan
    │   ├── 分配 hashmap 缓冲
    │   ├── 计算 shared memory 大小
    │   └── 设置 dataset descriptor (延迟初始化)
    │
    ├── 3. 批量查询循环
    │   └── search_kernel<<<>>>
    │       ├── 初始化: 计算种子节点距离
    │       └── 迭代搜索: Top-K → 选父节点 → 计算邻居距离
    │
    └── 4. 距离后处理 (缩放)
```

---

## 五、Search 详细步骤

### 5.1 搜索参数

```cpp
struct search_params {
  size_t max_queries = 0;          // 批处理大小 (0=自动)
  size_t itopk_size = 64;          // 中间 top-K 候选数
  size_t max_iterations = 0;       // 最大迭代次数 (0=自动)
  search_algo algo = AUTO;         // SINGLE_CTA, MULTI_CTA, MULTI_KERNEL
  size_t team_size = 0;            // 每次距离计算的线程数
  size_t search_width = 1;         // 图遍历分支宽度
};
```

### 5.2 Hash 表参数计算

**文件**: `search_plan.cuh:246-370`

**SINGLE_CTA 模式**:
```cpp
// Small hash (Shared Memory): 周期性重置
max_visited_nodes = itopk_size + (search_width * graph_degree * 1);
small_hash_bitlen = 8;  // 初始 256 entries
while (max_visited_nodes > hashmap::get_size(small_hash_bitlen) * max_fill_rate) {
    small_hash_bitlen += 1;
}
// 约束: small_hash_bitlen <= 13 (8K entries)

// 如果超出限制，切换到全局 hash (Device Memory)
if (small_hash_bitlen > 13) {
    hash_bitlen = 11;  // 初始 2K entries
    // ... 继续扩展直到满足需求
}
```

**MULTI_CTA 模式**:
```cpp
// Small visited hash (per-CTA, Shared Memory)
max_visited_nodes = mc_itopk_size + (graph_degree * 2);  // mc_itopk_size = 32

// Traversed hash (全局，跨 CTA 共享，Device Memory)
max_traversed_nodes = mc_num_cta_per_query * max(mc_itopk_size, max_iterations);
```

### 5.3 Shared Memory 布局

**文件**: `search_single_cta.cuh:132-157`

```
Shared Memory 布局 (SINGLE_CTA):
┌─────────────────────────────────────────────────────────────────┐
│  dataset_desc workspace                                          │
│  ├─ [descriptor]                                                │
│  └─ [query buffer, 对齐到 DatasetBlockDim]                       │
├─────────────────────────────────────────────────────────────────┤
│  result_buffer                                                   │
│  ├─ result_indices_buffer[0..itopk_size-1]     // Top-K 索引    │
│  ├─ result_distances_buffer[0..itopk_size-1]   // Top-K 距离    │
│  └─ result_*[itopk_size..]                     // 邻居候选      │
├─────────────────────────────────────────────────────────────────┤
│  visited_hash_buffer[0..2^small_hash_bitlen-1]                   │
├─────────────────────────────────────────────────────────────────┤
│  parent_list_buffer[0..search_width-1]                           │
├─────────────────────────────────────────────────────────────────┤
│  topk_ws[0..2] + terminate_flag                                  │
├─────────────────────────────────────────────────────────────────┤
│  (可选) Radix sort 工作区 (~2KB, 当 candidates > 256)            │
└─────────────────────────────────────────────────────────────────┘

其中: result_buffer_size = itopk_size + search_width × graph_degree
```

### 5.4 搜索 Kernel 核心算法

**文件**: `search_single_cta_kernel-inl.cuh:552-962`

```
search_core() 算法:
┌─────────────────────────────────────────────────────────────────┐
│  1. 初始化                                                       │
│     ├─ 复制 dataset descriptor 到 shared memory                  │
│     ├─ 初始化 visited hashmap (全部 ~0)                          │
│     └─ 计算初始种子节点距离，填充 result buffer                   │
├─────────────────────────────────────────────────────────────────┤
│  2. 迭代搜索 (iter = 0 to max_iterations)                        │
│     │                                                            │
│     ├─ Top-K 选择                                                │
│     │   ├─ candidates ≤ 256: Bitonic sort (单 warp)             │
│     │   └─ candidates > 256: Radix sort                         │
│     │                                                            │
│     ├─ Hash 表重置 (如果 (iter+1) % reset_interval == 0)         │
│     │                                                            │
│     ├─ 选择父节点 (pickup_next_parents)                          │
│     │   └─ 从 top-k 中选 MSB=0 的未使用节点                      │
│     │                                                            │
│     ├─ 距离计算 (compute_distance_to_child_nodes)                │
│     │   └─ 计算所有父节点邻居的距离                              │
│     │                                                            │
│     └─ 终止检查                                                  │
│         └─ 无新父节点 && iter >= min_iterations → 退出           │
├─────────────────────────────────────────────────────────────────┤
│  3. 输出结果                                                     │
│     ├─ 写入 top-k 到输出数组                                     │
│     └─ 记录迭代次数                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Dataset Descriptor

**文件**: `compute_distance.hpp:73-219`

```cpp
struct dataset_descriptor_base_t {
  struct args_t {
    void* extra_ptr1;              // 数据集指针
    void* extra_ptr2;
    uint32_t smem_ws_ptr;          // Shared memory 工作区指针
    uint32_t dim;                  // 维度
    uint32_t extra_word1;          // stride 或 codebook offset
    uint32_t extra_word2;
  };

  args_t args;
  setup_workspace_type* setup_workspace_impl;    // 设备函数指针
  compute_distance_type* compute_distance_impl;  // 设备函数指针
  INDEX_T size;
};
```

**特点**:
- 延迟初始化: 设备函数指针在首次搜索时解析
- 线程安全的 lazy evaluation
- 缓存 descriptor 避免重复初始化

---

## 六、内存使用总结

### 符号说明

| 符号 | 含义 |
|------|------|
| `N` | 数据集向量数量 |
| `dim` | 向量维度 |
| `aligned_dim` | 对齐后的维度，`align(dim, 16)` |
| `graph_degree` | 最终图的每节点邻居数 |
| `intermediate_degree` | 中间图的每节点邻居数 |
| `n_queries` | 查询向量数量 |
| `k` | 返回的近邻数量 |
| `max_queries` | 每批查询数量 |
| `gpu_top_k` | IVF-PQ 搜索返回的候选数，`intermediate_degree × refinement_rate` |
| `itopk_size` | 中间 top-K 候选数 |
| `hash_bitlen` | 全局 hashmap 位宽 |
| `small_hash_bitlen` | shared memory hashmap 位宽 |
| `result_buffer_size` | 结果缓冲大小，`itopk_size + search_width × graph_degree` |
| `pq_dim` | PQ 子空间数 (VPQ 压缩) |
| `pq_bits` | PQ 码位宽 (VPQ 压缩) |
| `sizeof(T)` | 数据类型大小 (float=4, half=2, int8=1) |
| `sizeof(IdxT)` | 索引类型大小 (通常 uint32_t=4, int64_t=8) |

### 6.1 索引自身内存

| 组件 | 计算公式 |
|------|----------|
| `graph_` | `N × graph_degree × sizeof(IdxT)` |
| `dataset_` (无压缩) | `N × aligned_dim × sizeof(T)` |
| `dataset_` (VPQ 压缩) | `N × ceil(pq_dim × pq_bits / 8) + pq_dim × 2^pq_bits × sizeof(codebook_type)` |

```
无压缩索引 = N × graph_degree × sizeof(IdxT) + N × aligned_dim × sizeof(T)
VPQ压缩索引 = N × graph_degree × sizeof(IdxT) + N × ceil(pq_dim × pq_bits / 8) + codebook
```

### 6.2 Build 过程峰值内存

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Main Pool** | 最终图 | `N × graph_degree × sizeof(IdxT)` |
| | 数据集 (如果 attach) | `N × aligned_dim × sizeof(T)` |
| **Workspace / Large Workspace** | IVF-PQ 索引 | 参见 IVF-PQ 文档 |
| | 距离缓冲 | `max_queries × gpu_top_k × sizeof(float)` |
| | 邻居缓冲 | `max_queries × gpu_top_k × sizeof(int64_t)` |
| | 精排缓冲 | `max_queries × intermediate_degree × (sizeof(float) + sizeof(int64_t))` |
| **Large Workspace** (图优化) | detour_count | `N × intermediate_degree × sizeof(uint8_t)` |
| | rev_graph | `N × graph_degree × sizeof(IdxT)` |
| **Host Memory** | knn_graph | `N × intermediate_degree × sizeof(IdxT)` |
| | 批处理缓冲 | `max_queries × (gpu_top_k + dim + intermediate_degree) × ~12 bytes` |

```
Build GPU 峰值 ≈ 索引自身 + IVF-PQ 索引 + 批处理缓冲 + 图优化临时空间
Build Host 峰值 = N × intermediate_degree × sizeof(IdxT) + 批处理 Host 缓冲
```

### 6.3 Search 过程峰值内存

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Workspace Pool** | 全局 hashmap | `max_queries × 2^hash_bitlen × sizeof(IdxT)` (当 small_hash 不足时) |
| | 迭代计数 | `max_queries × sizeof(uint32_t)` |
| | 随机种子 | `max_queries × num_seeds × sizeof(IdxT)` (可选) |
| **Shared Memory** | Dataset 工作区 | `sizeof(descriptor) + aligned_dim × sizeof(query_type)` |
| | 结果缓冲 | `result_buffer_size × (sizeof(IdxT) + sizeof(float))` |
| | Small hashmap | `2^small_hash_bitlen × sizeof(IdxT)` |
| | 父节点队列 | `search_width × sizeof(IdxT)` |
| | Top-K 工作区 | `3 × sizeof(uint32_t)` |
| | Radix-sort (可选) | `~2 KB` (当 itopk_size > 256) |

```
Search 峰值 ≈ max_queries × 2^hash_bitlen × sizeof(IdxT)  (全局 hashmap，如果使用)
            + max_queries × sizeof(uint32_t)              (迭代计数)
            + n_blocks × shared_memory_per_block          (通常 4-8 KB/block)
```

