# Reproduction guide

## Experiments and results

| Paper component | Program or directory | Frozen result |
| --- | --- | --- |
| Decision margins and flip predictions | `experiments/07_decision_stability.py`, `08_decision_stability_refinement.py` | `experiments/results/probe24_decision_stability.json`, `probe25_decision_stability_refinement.json` |
| Shared-endpoint covariance and rotation | `experiments/09_covariance_certificate.py`, `29_candidate_selection_covariance.py` | `probe26_covariance_certificate.json`, `probe46_candidate_selection_covariance.json` |
| Exact-run pruning replay | `experiments/10_pruning_replay.py` through `14_path_certificate.py` | `probe27_standard_vamana_pruning_replay.json` through `probe31_standard_vamana_path_certificate.json` |
| Gaussian assumptions and component variance | `experiments/15_p5_eligibility_audit.py`, `21_p5_component_variance_audit.py` | `probe32_standard_vamana_p5_eligibility_audit.json`, `probe38_standard_vamana_p5_component_variance_audit.json` |
| Held-out selective certificates | `experiments/16_p6_heldout_calibration.py`, `17_p6_strict_certificate.py`, `28_d_selective_ratio_certificates.py` | `probe33_standard_vamana_p6_heldout_calibration.json`, `probe34_standard_vamana_p6_strict_certificate.json`, `probe45_d_selective_ratio_certificates.json` |
| Counterexamples and semantic checks | `experiments/18_necessity_counterexamples.py`, `24_vamana_semantics_audit.py` through `27_c3_contamination_trace_audit.py` | `probe35_necessity_counterexamples.json`, `probe41_vamana_semantics_audit.json` through `probe44_c3_contamination_trace_semantics.json` |
| Calibration and observable rules | `experiments/controls/calibration/` | `calibration-cross.json`, `observable-certificates.json`, `diagnostics.json` |
| Six-code controls and strongest blockers | `experiments/controls/stability/` | `six-codes.json`, `blockers.json`, `local-distance-controls.json`, associated CSV files |
| E8 paired graphs and queries | `experiments/controls/calibration/run_graph.py` and `native/` | `experiments/controls/calibration/graph-analysis.json` |
| Wolt connectivity and interventions | `experiments/controls/calibration/wolt_connectivity/` | `root-cause.json`, `pruning-replay.json`, intervention summaries |

Bare `probe*.json` names in the table are under `experiments/results/`.

## Input and output configuration

`JMLR_DATA_DIR` selects the directory containing raw float32 vectors. `JMLR_SAMPLE_DIR` selects recovered MiniLM/GIST sample bundles. `JMLR_RESULTS_DIR` selects the output directory for numbered experiments.

The stability controls additionally accept `JMLR_OUTPUT_DIR`. Set it to a fresh directory for each independent run. Their `save_json` routine preserves existing result files.

`JMLR_CALIBRATION_DIR` selects the calibration controls' output directory and the stability audit's calibration-array input. `JMLR_GRAPH_DIR` selects the paired-graph outputs. Configure these when regenerating the calibration and graph-dependent diagnostics.

```powershell
$env:JMLR_DATA_DIR = 'D:/datasets/low-bit-decisions'
$env:JMLR_SAMPLE_DIR = 'D:/datasets/low-bit-decisions/samples'
$env:JMLR_RESULTS_DIR = 'runs/base'
python experiments/run_all.py 18 24 25 26 27

$env:JMLR_OUTPUT_DIR = 'runs/six-codes'
python experiments/controls/stability/six_quantizers_v2.py
python experiments/controls/stability/local_distance_controls.py
```

`six_quantizers_v2.py` generates comparison arrays consumed by `local_distance_controls.py` and `verify_local_controls.py`. `blocker_audit_v3.py` generates the strongest-blocker arrays. `calibration_audit.py` reads the calibration-cross arrays and the E8 trace arrays. The independent verifier reads the complete output set described by `protocol.json`.

The reference JSON and CSV files provide the published summaries. Running the array-level verifiers requires the corresponding generated NPZ files. The raw corpus vectors and full graph traces are supplied separately through the input paths.

## Paired graph searches

Use Linux with Rust 1.98.1, `taskset`, at least five available logical CPUs, and capacity for the two 200K datasets and their paired graphs. The controller retains its 56 GiB memory protection and its 20 GiB disk reserve. Its complete input identities and parameters are in `experiments/controls/calibration/remote-plan.json`.

Provide `data/inputs-200k/D03/` and `data/inputs-200k/D02/` as described in the data guide, then run

```bash
python experiments/controls/calibration/run_graph.py
python experiments/controls/calibration/graph_analysis.py --root runs/graph
```

The controller constructs eight graphs, retains both insertion orders and both coordinate systems, evaluates widths 64/128/256, and measures five rounds on one pinned query CPU. `JMLR_GRAPH_DIR` can select another graph output directory.

The Wolt replay programs consume the graph outputs, instrumented entry-prune logs and intervention records specified in their source. The frozen summaries retain the disconnected insertion order alongside the other observations.

## Figures

```bash
python paper/figures/sources/build_figures.py
python paper/figures/sources/build_rotation_figure.py
```

Both programs read the frozen result JSON files. Editable SVGs and the PNGs included by LaTeX are supplied in `paper/figures/`.

## File identities

```bash
python tools/verify_artifacts.py
```

This command checks the exported files against `publication-manifest.json`. Each entry also retains the original source hash and describes export adjustments. The manifest records path adaptations separately from the frozen numerical observations.

