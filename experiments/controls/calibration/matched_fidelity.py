"""在同一批留出近邻比较中核对校准不变的距离秩相关与翻转率。"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

OUT = Path(os.environ.get("JMLR_CALIBRATION_DIR", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
source = importlib.import_module("09_covariance_certificate")
source.ROOT = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data"))
values = source.load_sampled_dataset("cohere")
queries = np.random.default_rng(source.SEED).choice(4000, 240, replace=False)
rows, arrays = [], {}
for code in ("2bit", "rotated_2bit"):
    transformed = source.randomized_orthogonal_transform(values) if code.startswith("rotated") else values
    encoded = source.encode(transformed)
    for population, ids in (("fit80", queries[:80]), ("heldout160", queries[80:])):
        records = source.add_quantized_scores(source.build_exact_records(values, ids), encoded, 2)
        a, b, c, d = source.flatten_pair_records(records)
        distance_correlation = float(spearmanr(np.r_[a, b], np.r_[c, d]).statistic)
        flip = float(np.mean((c - d <= 0) & (a - b > 0)))
        rows.append({"code": code, "population": population, "distance_spearman": distance_correlation, "flip_rate": flip, "queries": len(ids)})
        for name, value in (("exact_top", a), ("exact_competitor", b), ("raw_top", c), ("raw_competitor", d)):
            arrays[f"{code}_{population}_{name}"] = value
np.savez_compressed(OUT / "matched-fidelity-arrays.npz", **arrays)
(OUT / "matched-fidelity.json").write_text(json.dumps({"protocol": "所有统计在同一原Probe26数据、同一旋转、同一查询及候选组上计算", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(rows)
