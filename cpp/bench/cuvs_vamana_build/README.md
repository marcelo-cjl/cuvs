# cuvs_vamana_build.sh

Build-only benchmark for cuVS GPU Vamana.

It directly calls `cuvs::neighbors::vamana::build`; no search is run.

## Usage

```bash
METRIC=COSINE \
CUVS_ROOT=/path/to/cuvs \
CUVS_BUILD_DIR=/path/to/cuvs-build \
./cuvs_vamana_build.sh /path/to/train.fbin /path/to/result.csv -
```

The third argument is an optional Vamana index file prefix. Use `-` to skip serialization.

## Defaults

| Parameter | Value |
|---|---:|
| metric | COSINE |
| graph_degree | 64 |
| visited_size | 128 |
| vamana_iters | 1.0 |
| alpha | 1.2 |
| max_fraction | 0.06 |
| batch_base | 2 |
| queue_size | 127 |
| reverse_batchsize | 1000000 |
| use_opt | 0 |
| include_dataset | 0 |

cuVS Vamana supports L2 build. For `METRIC=COSINE`, this benchmark first normalizes rows on GPU, then builds with `L2Expanded`.

## Observed Result

| Machine | GPU | Driver | Dataset | Rows | Dim | Metric | graph_degree | visited_size | vamana_iters | use_opt | data_load_ms | h2d_ms | normalize_ms | build_ms | total_ms |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g6 | NVIDIA L4 23GB | 560.35.05 | cohere | 1,000,000 | 768 | COSINE | 64 | 128 | 1.0 | 0 | 1725.934 | 232.788 | 26.118 | 120432.361 | 122417.202 |
| g6 | NVIDIA L4 23GB | 560.35.05 | cohere | 1,000,000 | 768 | COSINE | 64 | 128 | 1.0 | 1 | 1726.990 | 232.806 | 26.188 | 87133.738 | 89119.722 |
| g6 | NVIDIA L4 23GB | 560.35.05 | cohere | 1,000,000 | 768 | COSINE | 64 | 128 | 2.0 | 0 | 1724.672 | 232.783 | 26.254 | 296388.573 | 298372.281 |
| g6 | NVIDIA L4 23GB | 560.35.05 | cohere | 1,000,000 | 768 | COSINE | 64 | 128 | 2.0 | 1 | 1730.016 | 232.849 | 26.232 | 214848.952 | 216838.048 |
