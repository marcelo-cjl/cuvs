# cuvs_cagra_self_search.sh

Minimal CAGRA float cosine self-search benchmark for cuVS.

The script builds cuVS, generates a tiny benchmark target, then runs either index build or search.

## Usage

Run from the cuVS repository root:

```bash
cd /path/to/cuvs
chmod +x ./cpp/bench/cuvs_cagra_self_search/cuvs_cagra_self_search.sh
```

Build an index:

```bash
./cpp/bench/cuvs_cagra_self_search/cuvs_cagra_self_search.sh build \
  /path/to/train.fbin \
  /path/to/index.cuvsindex
```

Search an index:

```bash
./cpp/bench/cuvs_cagra_self_search/cuvs_cagra_self_search.sh search \
  /path/to/query.fbin \
  /path/to/index.cuvsindex
```

To run outside the cuVS root:

```bash
CUVS_ROOT=/path/to/cuvs /path/to/cuvs_cagra_self_search.sh search \
  /path/to/query.fbin \
  /path/to/index.cuvsindex
```

## Parameters

| Parameter | Value |
|---|---:|
| dtype | float |
| metric | CosineExpanded |
| graph_degree | 64 |
| intermediate_graph_degree | 128 |
| topk | 1 |
| itopk_size | 32 |
| search_width | 1 |
| max_iterations | 0 |
| search_algo | SINGLE_CTA |

The input `.fbin` vectors are passed to cuVS unchanged. The script does not
pre-normalize them; `CosineExpanded` provides the cosine-distance semantics
for both index construction and search.

## Output

`build` prints `build_ms`.

`search` prints:

| Field | Meaning |
|---|---|
| load_index_ms | index load time |
| search_ms | CUDA event search time |
| qps | `rows * 1000 / search_ms` |
| self_recall_at_1 | fraction of `label == row_id` when query is the training set |

## Observed result

| Machine | GPU | Driver | Dataset | Rows | Dim | search_ms | QPS | self_recall_at_1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| g6.4xlarge | NVIDIA L4 23GB | 560.35.05 | cohere self-query | 1,000,000 | 768 | 22,993.9 | 43,489.8 | 0.995099 |
