# cuVS GPU 索引内存使用总结

## 一、文档目的

本文档旨在系统性分析 cuVS 库中各 GPU 向量索引的内存使用情况，帮助用户：

1. **容量规划**: 根据数据规模预估所需 GPU 显存，选择合适的硬件配置
2. **索引选型**: 根据显存限制选择合适的索引类型和参数
3. **性能调优**: 理解内存分配模式，优化批处理大小和并发度
4. **问题诊断**: 当遇到 OOM (Out of Memory) 错误时，快速定位内存瓶颈

**分析范围**:
- 索引类型: Brute Force, IVF-Flat, IVF-PQ, CAGRA, Vamana
- 关注阶段: Build (索引构建) 和 Search (向量搜索)
- 内存类型: Main Pool, Workspace Pool, Large Workspace

---

## 二、内存类型说明

cuVS 使用 RMM (RAPIDS Memory Manager) 管理 GPU 内存，通过 `raft::resources` 提供不同的内存分配方式。

### 2.1 内存类型概览

| 内存类型 | 用途 | 上游资源 (upstream) | 创建时机 |
|----------|------|---------------------|----------|
| **Main Pool** | 用户数据 (Query、输出结果)、索引持久数据 | `cuda_memory_resource` | 首次 Search 时 |
| **Workspace Pool** | 算法内部临时计算缓冲 | `cuda_memory_resource` | `initialize_raft()` 时 |
| **Large Workspace** | Build 时的大块临时数据 | `cuda_memory_resource` | 首次 Build 时 (懒加载) |

### 2.2 内存池关系

**重要**: Workspace Pool 和 Main Pool 是**并行**的两个独立池，都直接从 `cuda_memory_resource` 分配，而不是嵌套关系。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GPU 显存                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        cuda_memory_resource (cudaMalloc)                    │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                  │
│              │                     │                     │                  │
│              ▼                     ▼                     ▼                  │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│   │   Main Pool      │  │  Workspace Pool  │  │  Large Workspace │          │
│   │                  │  │                  │  │                  │          │
│   │ • Query 数据     │  │ • 距离矩阵       │  │ • 训练数据集     │          │
│   │ • 输出 IDs       │  │ • top-k 缓冲     │  │ • 临时图/边列表  │          │
│   │ • 输出 Distances │  │ • 查询缓冲       │  │ • 标签数组       │          │
│   │ • 索引持久数据   │  │                  │  │                  │          │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 创建时序

```
时间线 (典型使用场景: 先 Build 后 Search)
──────────────────────────────────────────────────────────────────────────────►

1. Knowhere 启动
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  initialize_raft()                                                          │
│  • 创建 Workspace Pool                                                      │
│  • upstream = get_current_device_resource() → cuda_memory_resource          │
│  • 存储到 params_.workspace_mrs，等待后续使用                                 │
└─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
2. 首次 Build 调用
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  get_device_resources_without_mempool() - 不触发 Main Pool 创建             │
│  • 使用 Large Workspace (懒加载创建)                                         │
│  • upstream = get_current_device_resource() → cuda_memory_resource          │
└─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
3. 首次 Search 调用
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  get_device_resources() → resource_components 构造                          │
│  • 创建 Main Pool (pool_mr_)                                                │
│  • upstream = cuda_memory_resource                                          │
│  • set_current_device_resource(Main Pool)  ← 全局 RMM 资源变为 Main Pool    │
│  • 获取之前创建的 Workspace Pool                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 各内存类型说明

#### Main Pool

- 首次调用 `get_device_resources()` 时创建 (`device_resources_manager.hpp:164-187`)
- 创建后通过 `set_current_device_resource()` 设为全局 RMM 资源
- 后续所有 `make_device_matrix` 等分配都会使用此池
- 用途: Query 数据拷贝、输出 IDs/Distances、索引持久数据
- 配置: `init_mem_pool_size_mb` (初始大小), `max_mem_pool_size_mb` (最大大小)

#### Workspace Pool

- 在 `initialize_raft()` 时创建 (`raft_initialization.cc:72-73`)
- **创建时 Main Pool 尚不存在**，所以 upstream 是 `cuda_memory_resource`
- 专门用于算法内部的临时工作空间
- 外层包装 `limiting_resource_adaptor` 限制最大使用量
- 配置: `max_workspace_size_mb`
- 池大小: 初始 `min(1GB, limit/2)`，最大 `min(limit+0.5GB, limit×1.5)`

#### Large Workspace

- 首次 Build 时懒加载创建 (`device_memory_resource.hpp:83-100`)
- 只在 Build 过程中使用，而 Build 通常在 Search 之前执行
- 因此 upstream 为 `cuda_memory_resource` (此时 Main Pool 尚未创建)
- 用途: 训练数据集、标签数组、临时图结构等大块数据
- 无额外大小限制，直接使用 upstream 的容量

### 2.5 Knowhere 的调用路径

| 操作 | 调用的函数 | 是否触发 Main Pool 创建 | 内存分配来源 |
|------|-----------|------------------------|-------------|
| Build | `get_device_resources_without_mempool()` | 否 | `cuda_memory_resource` (如果 Main Pool 不存在) |
| Search | `get_device_resources()` | 是 (首次时) | Main Pool |
| Serialize | `get_device_resources_without_mempool()` | 否 | `cuda_memory_resource` (如果 Main Pool 不存在) |
| Deserialize | `get_device_resources_without_mempool()` | 否 | `cuda_memory_resource` (如果 Main Pool 不存在) |

**注意**: 如果 Search 已经执行过（Main Pool 已创建），后续的 Build/Serialize/Deserialize 中的 `make_device_matrix` 也会使用 Main Pool，因为 `get_current_device_resource()` 已经是 Main Pool 了。

---

## 三、索引内存使用汇总

### 3.1 符号说明

| 符号 | 含义 |
|------|------|
| `N` | 数据集向量数量 |
| `dim` | 向量维度 |
| `sizeof(T)` | 输入数据类型大小 (float=4, half=2, int8=1, uint8=1) |
| `sizeof(IdxT)` | 索引类型大小 (int64_t=8, uint32_t=4) |
| `sizeof(DistT)` | 距离类型大小 (通常 float=4) |
| `n_lists` | IVF 聚类数量 |
| `n_queries` | 查询向量数量 |
| `k` | 返回的近邻数量 |
| `graph_degree` | 图索引的出度 |
| `pq_dim` | PQ 子空间数量 |
| `pq_bits` | PQ 编码位宽 |
| `n_rows_train` | 训练样本数 (通常为 N 的采样子集) |
| `intermediate_graph_degree` | CAGRA 中间图度数 |
| `visited_size` | Vamana GreedySearch 最大访问节点数 |
| `max_batchsize` | Vamana 最大批次大小 |

### 3.2 索引支持特性

| 索引类型 | 支持的数据类型 | 支持的距离度量 | GPU Search |
|----------|---------------|---------------|------------|
| **Brute Force** | float, half | L2Expanded, L2SqrtExpanded, InnerProduct, CosineExpanded | ✓ |
| **IVF-Flat** | float, half, int8, uint8 | L2Expanded, InnerProduct, CosineExpanded* | ✓ |
| **IVF-PQ** | float, half, int8, uint8 | L2Expanded, InnerProduct, CosineExpanded* | ✓ |
| **CAGRA** | float, half, int8, uint8 | L2Expanded, InnerProduct | ✓ |
| **Vamana** | float, int8, uint8 | L2Expanded | ✗ |

> *注: int8/uint8 类型暂不支持 CosineExpanded

### 3.3 内存使用汇总

| 索引类型 | 索引数据 (Main Pool) | Build 临时内存 | Search 临时内存 |
|----------|---------------------|----------------|-----------------|
| **Brute Force** | dataset: `N × dim × sizeof(T)` <br> norms (L2/Cosine): `N × sizeof(DistT)` | - | **Workspace:** <br> 距离矩阵: `tile_rows × tile_cols × sizeof(DistT)` |
| **IVF-Flat** | lists (向量): `N × dim × sizeof(T)` <br> lists (索引): `N × sizeof(IdxT)` | **Large Workspace:** <br> trainset: `n_rows_train × dim × sizeof(float)` | **Workspace:** <br> 查询缓冲: `n_queries × dim × sizeof(float)` |
| **IVF-PQ** | lists (PQ码): `N × ceil(pq_dim × pq_bits / 8)` <br> lists (索引): `N × sizeof(IdxT)` | **Large Workspace:** <br> trainset: `n_rows_train × dim × sizeof(float)` | **Workspace:** <br> 查询缓冲: `n_queries × dim × sizeof(float)` |
| **CAGRA** | graph: `N × graph_degree × sizeof(IdxT)` <br> dataset (可选): `N × dim × sizeof(T)` | **Large Workspace:** <br> knn_graph: `N × intermediate_graph_degree × sizeof(IdxT)` | **Workspace:** <br> hashmap: `n_queries × hashmap_size × sizeof(IdxT)` |
| **Vamana** | graph: `N × graph_degree × sizeof(IdxT)` <br> dataset: `N × dim × sizeof(T)` | **Large Workspace:** <br> visited_ids: `max_batchsize × visited_size × sizeof(IdxT)` <br> visited_dists: `max_batchsize × visited_size × sizeof(float)` | 不支持 GPU Search |
