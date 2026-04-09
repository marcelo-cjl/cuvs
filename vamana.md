# Vamana 索引详细分析

## 一、简介

Vamana 是 DiskANN 算法的核心图构建方法，通过增量插入和 RobustPrune 剪枝创建高质量的近邻搜索图。

**支持的数据类型**: float, int8_t, uint8_t

**支持的距离度量**: L2Expanded

**特点**:
- 仅支持 build 和 serialize，**不提供 GPU search API**
- 生成的图与 DiskANN 开源库兼容
- 可与 CAGRA search 配合使用

---

## 二、Build 流程总结

```
build() 入口
    │
    ├── 1. 初始化
    │   ├── 分配 GPU 图内存 [N, graph_degree]
    │   ├── 分配批处理工作内存
    │   ├── 生成随机插入顺序
    │   └── 随机选择 medoid (搜索起始点)
    │
    ├── 2. 批量插入循环 (批次大小指数增长)
    │   ├── GreedySearch: 搜索候选邻居
    │   ├── RobustPrune: α 剪枝选择高质量邻居
    │   ├── WriteEdges: 写入正向边
    │   ├── CreateReverseEdges: 创建反向边
    │   ├── RobustPrune: 反向边剪枝
    │   └── WriteEdges: 写入反向边
    │
    ├── 3. (可选) 量化数据集
    │
    └── 4. 构建 index 对象返回
```

---

## 三、Build 详细步骤

### 3.1 初始化阶段

**文件**: `vamana_build.cuh:154-198`

```cpp
// 1. 初始化空图 - 所有边设为无效值
auto d_graph = raft::make_device_matrix<IdxT, int64_t>(res, N, graph_degree);
raft::linalg::map(res, d_graph.view(), raft::const_op<IdxT>{raft::upper_bound<IdxT>()});
```
**内存**: Main Pool, `N × graph_degree × sizeof(IdxT)`

```cpp
// 2. 分配批处理工作内存
int max_batchsize = (int)(max_fraction * N);  // 默认 0.06 * N
auto query_ids = raft::make_device_vector<IdxT>(res, max_batchsize);
```

```cpp
// 3. 分配 QueryCandidates 结构和访问记录
auto query_list_ptr = raft::make_device_mdarray<QueryCandidates<IdxT, accT>>(
    res, large_workspace_mr, raft::make_extents<int64_t>(max_batchsize + 1));
auto visited_ids = raft::make_device_mdarray<IdxT>(
    res, large_workspace_mr, raft::make_extents<int64_t>(max_batchsize, visited_size));
auto visited_dists = raft::make_device_mdarray<accT>(
    res, large_workspace_mr, raft::make_extents<int64_t>(max_batchsize, visited_size));
```
**内存**: Large Workspace, `max_batchsize × visited_size × 8` bytes

```cpp
// 4. Host 端生成随机插入顺序
std::vector<IdxT> insert_order(N);
create_insert_permutation(insert_order, N);  // Fisher-Yates shuffle
medoid_id = rand() % N;  // 随机选择 medoid
```

### 3.2 GreedySearch - 搜索候选邻居

**文件**: `greedy_search.cuh:99-293`

**目的**: 从 medoid 开始贪心搜索，找到与待插入向量最近的节点作为候选邻居。

```
GreedySearch 算法:
┌─────────────────────────────────────────────────────────────────┐
│  输入: query 向量, 当前图 graph, medoid_id                       │
│  输出: visited_ids[], visited_dists[] (访问过的所有节点)         │
├─────────────────────────────────────────────────────────────────┤
│  1. candidate_queue.push(medoid, dist(query, medoid))           │
│                                                                 │
│  2. while candidate_queue 非空:                                 │
│       a. node = candidate_queue.pop_min()  // 弹出最近的        │
│       b. if node 已在 visited_list: continue                    │
│       c. visited_list.add(node)                                 │
│       d. if dist(node) > topk_list.max_dist 且 topk 已满: break │
│       e. topk_list.add(node)                                    │
│       f. for neighbor in graph[node]:                           │
│            candidate_queue.push(neighbor, dist(query, neighbor))│
│                                                                 │
│  3. return visited_list (按距离排序后的候选列表)                 │
└─────────────────────────────────────────────────────────────────┘
```

**Kernel 实现要点** (`greedy_search.cuh:180-250`):
```cpp
// 每个 block 处理一个 query
for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    // 将 query 向量加载到 shared memory
    update_shared_point(&s_query, dataset, query_list[i].queryId, dim);
    // 初始化: 从 medoid 开始
    heap_queue.insert_back(dist(query, medoid), medoid_id);

    while (cand_q_size != 0) {
        cand = heap_queue.pop();
        if (query_list[i].check_visited(cand.id, cand.dist)) continue;
        if (topk_q_size == topk && cur_k_max <= cand.dist) break;
        parallel_pq_max_enqueue(topk_pq, ...);
        // 获取邻居并计算距离
        enqueue_all_neighbors(neighbor_array, heap_queue, ...);
    }
}
```

**Shared Memory**:
```cpp
search_smem_total_size =
    (dim + align_padding) * sizeof(T) +           // s_query 向量
    degree * sizeof(int) +                         // neighbor_array
    queue_size * sizeof(DistPair<IdxT, accT>);    // candidate_queue
// 典型值 (dim=128, degree=64): ~2.5 KB per block
```

### 3.3 RobustPrune - α 剪枝算法

**文件**: `robust_prune.cuh:68-240`

**目的**: 从候选列表中选择高质量邻居，避免选择相互靠近的冗余邻居。

```
RobustPrune 算法 (α-剪枝):
┌─────────────────────────────────────────────────────────────────┐
│  输入: query, candidates[] (来自 GreedySearch), 当前 graph 边    │
│  输出: 最多 degree 个剪枝后的邻居                                │
├─────────────────────────────────────────────────────────────────┤
│  1. merged_list = merge(graph[query], candidates) 按距离排序     │
│  2. occlusion_list[i] = 0 for all i                             │
│  3. accepted_count = 0                                          │
│                                                                 │
│  4. for α = 1.0; α <= alpha; α *= 1.2:  // 逐渐放宽条件         │
│       for each c in merged_list:                                │
│         if occlusion_list[c] > α: continue  // 已被遮挡         │
│         if occlusion_list[c] == -∞: continue  // 已被接受       │
│                                                                 │
│         // 接受 c 作为邻居                                       │
│         result[accepted_count++] = c                            │
│         occlusion_list[c] = -∞  // 标记已接受                   │
│                                                                 │
│         // 检查 c 是否遮挡其他候选                               │
│         for each c' in remaining candidates:                    │
│           if dist(c, c') * α < dist(query, c'):                 │
│             occlusion_list[c'] = max(occlusion_list[c'], α)     │
│                                                                 │
│         if accepted_count >= degree: break                      │
│                                                                 │
│  5. return result[0:degree]                                     │
└─────────────────────────────────────────────────────────────────┘
```

**遮挡原理图示**:
```
                query
                  │
           ┌─────┼─────┐
           │     │     │
           ▼     ▼     ▼
          c1    c2    c3    (候选邻居)
           │           │
           └─────┬─────┘
                 │
                c1 遮挡 c3?  若 dist(c1,c3) * α < dist(query,c3)
                             则 c3 被遮挡，不选择
```

**Shared Memory**:
```cpp
prune_smem_total_size =
    (degree + visited_size) * sizeof(float) +           // occlusion_list
    (degree + visited_size) * sizeof(DistPair<IdxT, accT>);  // merged_list
// 典型值 (degree=64, visited_size=128): ~3 KB per block
```

### 3.4 反向边处理

**文件**: `vamana_build.cuh:331-529`

**目的**: Vamana 图是有向图，当 A→B 边被插入后，需要考虑是否也应该有 B→A 边。

```
反向边处理流程:
┌─────────────────────────────────────────────────────────────────┐
│  1. 收集本批次所有正向边: (src, dest, dist)                      │
│  2. 按 dist 排序 (保证反向边列表按距离有序)                      │
│  3. 按 dest 排序，分组得到反向边列表                            │
│     例: A→B, C→B, D→B  =>  B 的反向边候选: [A, C, D]            │
│  4. 对每个有反向边的节点:                                        │
│     a. 构建 reverse_list (反向边候选)                           │
│     b. 调用 RobustPrune 剪枝                                    │
│     c. 写入图                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**动态内存分配** (每批次):
```cpp
// 计算本批次总边数
prefix_sums_sizes<<<1, 1>>>(query_list, step_size, &total_edges);

// 分配边列表存储
auto edge_dist_pair = raft::make_device_mdarray<DistPair<IdxT, accT>>(
    res, large_ws, raft::make_extents<int64_t>(total_edges));
auto edge_dest = raft::make_device_mdarray<IdxT>(res, large_ws, ...);
auto edge_src = raft::make_device_mdarray<IdxT>(res, large_ws, ...);
```
**内存**: Large Workspace, `total_edges × (sizeof(DistPair) + 2 × sizeof(IdxT))`

### 3.5 批次大小增长策略

```cpp
int step_size = 1;  // 初始批次大小

for (int start = 0; start < insert_iters * N; ) {
    // 处理当前批次...
    start += step_size;
    if (start >= N) {
        start = 0;
        insert_iters -= 1.0;
        step_size = max_batchsize;  // 后续迭代直接用最大批次
    }
    step_size *= batch_base;  // 指数增长 (默认 base=2)
    step_size = min(step_size, max_batchsize);
}
```

**增长序列示例** (N=1M, max_fraction=0.06, batch_base=2):
```
批次: 1 → 2 → 4 → 8 → ... → 32768 → 60000 → 60000 → ...
```

---

## 四、Search 流程总结

**cuVS Vamana 不提供 GPU Search API。**

---

## 五、Search 详细步骤

**cuVS Vamana 不支持 GPU Search。** 使用方式：

### 方式 1: 导出给 DiskANN CPU 搜索
```cpp
vamana::serialize(res, "index_prefix", index);
// 使用 DiskANN 开源库进行 CPU 搜索
```

### 方式 2: 使用 CAGRA search (推荐)
```cpp
// Vamana 和 CAGRA 共享相同的图结构
auto graph_view = vamana_index.graph();
cagra::index<T, IdxT> cagra_idx(res, metric, dataset_view, graph_view);
cagra::search(res, search_params, cagra_idx, queries, neighbors, distances);
```

---

## 六、内存使用总结

### 符号说明

| 符号 | 含义 |
|------|------|
| `N` | 数据集向量数量 |
| `dim` | 向量维度 |
| `aligned_dim` | 对齐后的维度，`align(dim, 16)` |
| `graph_degree` | 图的出度，对应参数 `R` |
| `visited_size` | GreedySearch 最大访问节点数，对应参数 `L` |
| `max_batchsize` | 最大批次大小，`max_fraction × N` |
| `reverse_batch` | 反向边处理批次大小 |
| `total_edges` | 每批次产生的边数，约 `batch_size × graph_degree` |
| `codes_rowlen` | PQ 量化后每向量字节数，`ceil(pq_dim × pq_bits / 8)` |
| `sizeof(T)` | 数据类型大小 (float=4, int8=1, uint8=1) |
| `sizeof(IdxT)` | 索引类型大小 (uint32=4, uint64=8) |
| `sizeof(QueryCandidates)` | 查询候选结构体大小 (32 bytes) |
| `sizeof(Node)` | 节点结构体大小 (8 bytes) |
| `sizeof(DistPair)` | 距离对结构体大小 (8 bytes) |

### 6.1 索引自身内存

| 组件 | 计算公式 |
|------|----------|
| `graph_` | `N × graph_degree × sizeof(IdxT)` |
| `dataset_` | `N × aligned_dim × sizeof(T)` |
| `quantized_dataset_` (可选) | `N × codes_rowlen` |

```
索引内存 (无量化) = N × graph_degree × sizeof(IdxT) + N × aligned_dim × sizeof(T)
索引内存 (有量化) = 上述 + N × codes_rowlen
```

### 6.2 Build 过程峰值内存

| 内存类型 | 分配项 | 计算公式 |
|----------|--------|----------|
| **Main Pool** | `d_graph` | `N × graph_degree × sizeof(IdxT)` |
| | `query_ids` | `max_batchsize × sizeof(IdxT)` |
| **Large Workspace** | `query_list_ptr` | `max_batchsize × sizeof(QueryCandidates)` |
| | `visited_ids` | `max_batchsize × visited_size × sizeof(IdxT)` |
| | `visited_dists` | `max_batchsize × visited_size × sizeof(float)` |
| | `topk_pq_mem` | `max_batchsize × visited_size × sizeof(Node)` |
| | `s_coords_mem` | `min(10000, max_batchsize) × aligned_dim × sizeof(T)` |
| | `edge_*` 数组 | `total_edges × (sizeof(DistPair) + 2 × sizeof(IdxT))` (每批次) |
| | `reverse_*` 数组 | `reverse_batch × visited_size × (sizeof(IdxT) + sizeof(float))` (每批次) |
| **Workspace Pool** | (未使用) | 0 |
| **Host Memory** | `insert_order` | `N × sizeof(IdxT)` |
| | `vamana_graph` | `N × graph_degree × sizeof(IdxT)` |

```
GPU 峰值 ≈ Main Pool 分配 + Large Workspace 固定分配 + Large Workspace 动态峰值
Host 峰值 = N × sizeof(IdxT) + N × graph_degree × sizeof(IdxT)
```

### 6.3 Search 过程峰值内存

**cuVS Vamana 不支持 GPU Search API。**

