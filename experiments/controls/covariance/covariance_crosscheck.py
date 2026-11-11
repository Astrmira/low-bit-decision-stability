"""交叉固定两套原有校准器和两套原有查询样本，分解表间差异。"""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import importlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
module = importlib.import_module("09_covariance_certificate")
grouping = importlib.import_module("29_candidate_selection_covariance")
module.ROOT = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data"))
grouping.ROOT = module.ROOT
existing = json.loads((OUT / "covariance-controls.json").read_text(encoding="utf-8"))
old = json.loads((ROOT / "results/probe46_candidate_selection_covariance.json").read_text(encoding="utf-8"))
prior = next(row for row in old["results"] if row["dataset"] == "cohere")["quantizers"]["2bit"]
models = {
    "ranking_margin": existing["calibration_models"]["margin_fit"],
    "global_affine": [prior["calibration"]["intercept"], prior["calibration"]["slope"]],
}
first = module.load_sampled_dataset("cohere")
first_queries = np.random.default_rng(module.SEED).choice(4000, 240, replace=False)[:80]
second, _ = grouping.load_values("cohere")
second_queries = np.random.default_rng(grouping.SEED + 768).choice(6000, 160, replace=False)
output, arrays = {}, {}
for name, values, queries in [("original_4k_fit", first, first_queries), ("original_6k", second, second_queries)]:
    records = module.add_quantized_scores(module.build_exact_records(values, queries), module.encode(values), 2)
    # 使用逐坐标双精度求和复核全局拟合实验的原精确分数定义。
    for record in records:
        query = record["query_index"]
        candidates = module.build_exact_records(values, np.array([query]))[0]["candidates"]
        record["exact_scores"] = np.sum(values[query] * values[candidates], axis=1, dtype=np.float64)
    exact_top, exact_comp, code_top, code_comp = module.flatten_pair_records(records)
    ids = np.repeat(np.arange(len(records)), 31)
    output[name] = {}
    for model_name, (bias, slope) in models.items():
        left = bias + slope * code_top - exact_top
        right = bias + slope * code_comp - exact_comp
        output[name][model_name] = grouping.pair_metrics(left, right, ids, fixed_anchor=True)
        arrays[f"{name}_{model_name}_left"] = left
        arrays[f"{name}_{model_name}_right"] = right
    arrays[f"{name}_query_ids"] = ids
a = output["original_4k_fit"]["ranking_margin"]["pooled_correlation"]
b = output["original_4k_fit"]["global_affine"]["pooled_correlation"]
c = output["original_6k"]["ranking_margin"]["pooled_correlation"]
d = output["original_6k"]["global_affine"]["pooled_correlation"]
output["decomposition"] = {
    "total_rho_difference": a - d,
    "calibration_first": {"calibration": a - b, "population": b - d},
    "population_first": {"population": a - c, "calibration": c - d},
    "original_6k_reproduction_gap": d - prior["regimes"]["top_anchor"]["pooled_correlation"],
}
assert abs(output["decomposition"]["original_6k_reproduction_gap"]) < 1e-5
np.savez_compressed(OUT / "covariance-cross-arrays.npz", **arrays)
(OUT / "covariance-crosscheck.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print("两套原协议的协方差差异已交叉复核。", output["decomposition"], flush=True)
