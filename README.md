# Low-Bit Decision Stability in Vector Search

Research materials for **When Does Low-Bit Quantization Preserve the Decisions of Vector Search?** by Wenxuan Xiao and Xu Cao.

[Read the paper on arXiv](https://arxiv.org/abs/2609.09854)

The experiments study ranking and pruning comparisons, shared-endpoint errors, replay of neighbour selection, held-out certificates, and paired graph searches.

Git preserves the original file bytes so the published SHA-256 identities remain valid across platforms.

## Contents

| Directory | Contents |
| --- | --- |
| `paper/` | Current LaTeX manuscript, appendices, bibliography, PNG and editable SVG figures |
| `paper/figures/sources/` | Figure-generation programs |
| `experiments/` | Numbered decision-level experiments and shared utilities |
| `experiments/results/` | Frozen numerical summaries and compact diagnostic arrays |
| `experiments/controls/` | Covariance, calibration, cross-code and graph-search controls |
| `experiments/controls/calibration/native/` | Frozen Rust graph-search harness and its TriviumDB dependency |
| `data/` | Input conventions and input identities |
| `docs/` | Experiment-to-result map and reproduction commands |
| `publication-manifest.json` | Exported-file hashes, original-file hashes and path adaptations |

## Environment

Use Python 3.11 or newer. The numerical dependencies are listed in `requirements.txt`.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python tools/verify_artifacts.py
```

The paper's last calibration analysis used NumPy 2.4.2 and SciPy 1.17.0. The graph harness uses Rust 1.98.1. Its five-round single-CPU timing protocol is in `experiments/controls/calibration/remote-plan.json`.

## Reproduction

The following deterministic checks use synthetic comparisons and write to a separate output directory.

```bash
JMLR_RESULTS_DIR=runs/semantics python experiments/run_all.py 24 25 26 27
```

On PowerShell, set the variable before invoking Python.

```powershell
$env:JMLR_RESULTS_DIR = 'runs/semantics'
python experiments/run_all.py 24 25 26 27
```

Configure dataset inputs using `JMLR_DATA_DIR`. See [the data guide](data/README.md) and [the reproduction guide](docs/reproduction.md) for input layouts, commands and experiment dependencies.

## Compile the paper

```bash
cd paper
latexmk -xelatex jmlr_main.tex
```

The checked-in bibliography output supports submission systems that use a pre-generated `.bbl`. The source enables the JMLR preprint option.

## Frozen graph dependency

The graph harness includes the source snapshot used for E8. Its Rust implementation files retain their original bytes. The export adjusts Cargo workspace membership, benchmark declarations and dependency paths for this standalone repository. File identities are recorded in `publication-manifest.json`.

The bundled TriviumDB dependency is distributed under Apache-2.0. Its license is in `experiments/controls/calibration/native/frozen-tdb/LICENSE`.

