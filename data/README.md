# Dataset inputs

Set `JMLR_DATA_DIR` to your local data directory. The default is this repository's `data/` directory.

The numbered experiments consume little-endian float32 matrices with no file header. The registry in `experiments/utils.py` specifies each filename prefix and dimension. For example, Cohere uses `cohere_train.f32` with 768 columns. The loader retains the sampling seeds and normalization rules used by the original experiments.

The four-dataset controls use 4,000 vectors each from Cohere, MiniLM, SIFT and GIST. Recovered MiniLM/GIST bundles contain the frozen `pruning`, `certificate` and index arrays. `recover_samples.py` retains the source URLs and sampling procedure. Use `JMLR_SAMPLE_DIR` to select their directory.

`data/manifest.json` records available input identities and sample sizes. Raw vectors and generated indices are covered by `.gitignore`.

## E8 inputs

The graph controller reads the following layout relative to `JMLR_DATA_DIR`.

```text
inputs-200k/
  D03/
    base.f32
    queries.f32
    truth32.u32
  D02/
    base.f32
    queries.f32
    truth32.u32
```

`D03` is the 768-dimensional Cohere collection. `D02` is the 512-dimensional Wolt collection. Each has 200,000 database vectors, 744 queries and 32 ground-truth IDs per query. Ground-truth IDs are zero-based uint32 values. The controller checks file sizes and CRC32 identities before preparing paired graphs.

## Compact frozen results

The NPZ files under `experiments/results/` hold compact diagnostic outputs. Full base matrices, query matrices, graph indices and per-comparison output arrays are stored through the local data and run directories.

