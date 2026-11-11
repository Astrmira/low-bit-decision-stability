"""直接复算坐标QQ拟合，并穷举既有诊断门槛附近的预定组合。"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

OUT = Path(os.environ.get("JMLR_CALIBRATION_DIR", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]


def qq_statistics(values):
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    normal = norm.ppf((np.arange(1, len(values) + 1) - 0.375) / (len(values) + 0.25))
    normal -= normal.mean()
    ordered = np.sort(values.astype(np.float64), axis=0)
    ordered -= ordered.mean(axis=0)
    variance = np.sum(ordered**2, axis=0)
    valid = variance > 0
    r2 = (normal @ ordered[:, valid]) ** 2 / ((normal @ normal) * variance[valid])
    return {"n": len(values), "dim": values.shape[1], "valid_coordinates": int(valid.sum()), "qq_r2_mean": float(r2.mean()), "qq_r2_median": float(np.median(r2)), "norm_cv": float(norms.std(ddof=1) / norms.mean())}


def main():
    cohere_path = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data")) / "cohere_train.f32"
    cohere = np.memmap(cohere_path, dtype=np.float32, mode="r", shape=(1000000, 768))
    indices = np.random.default_rng(20260823).choice(len(cohere), 4000, replace=False)
    sources = {"Cohere": (np.asarray(cohere[indices]).copy(), "保留原始半径的归档向量")}
    for name in ("minilm", "gist960"):
        with np.load(Path(os.environ.get("JMLR_SAMPLE_DIR", ROOT.parent / "data/samples")) / f"{name}-samples.npz") as source:
            sources[name] = (source["pruning"].copy(), "归档的单位范数输入")
    graph_root = Path(os.environ.get("JMLR_GRAPH_DIR", OUT))
    wolt = np.fromfile(graph_root / "graph-data/wolt/20261110/original/base.f32", dtype=np.float32, count=4000 * 512).reshape(4000, 512)
    sources["Wolt"] = (wolt, "归档的单位范数输入")
    qq = [{"dataset": name, "input_scale": scale, **qq_statistics(values)} for name, (values, scale) in sources.items()]
    frozen_path = ROOT / "results/probe26_covariance_certificate.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    records = []
    for dataset in frozen["results"]:
        for code, result in dataset["quantizers"].items():
            evaluation = result["evaluation"]
            records.append({"dataset": dataset["dataset"], "code": code, "slope": result["calibration_slope"], "score_correlation": evaluation["score_margin_spearman"], "residual_margin_correlation": evaluation["residual_margin_spearman"], "left_tail_ratio": evaluation["tail"]["max_left_ratio_through_2_5"], "original_eligible": result["calibration_gate"]["eligible"]})
    variants = []
    for minimum_score, maximum_dependence, maximum_tail in itertools.product((0.05, 0.10, 0.15), (0.40, 0.50, 0.60), (1.5, 2.0, 3.0)):
        decisions = [r["slope"] > 0 and r["score_correlation"] >= minimum_score and abs(r["residual_margin_correlation"]) <= maximum_dependence and r["left_tail_ratio"] <= maximum_tail for r in records]
        variants.append({"minimum_score": minimum_score, "maximum_dependence": maximum_dependence, "maximum_tail": maximum_tail, "eligible": decisions})
    baseline = next(r for r in variants if r["minimum_score"] == 0.1 and r["maximum_dependence"] == 0.5 and r["maximum_tail"] == 2)
    assert baseline["eligible"] == [r["original_eligible"] for r in records]
    for index, record in enumerate(records):
        record["passes_of_27"] = sum(v["eligible"][index] for v in variants)
    output = {"qq_method": "每一坐标的排序样本对正态Blom分位数做含截距线性拟合，汇总R²；四库均4000向量", "qq": qq, "gate_source_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(), "gate_records": records, "gate_variants": variants}
    (OUT / "diagnostics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("QQ表", qq)
    print("门槛敏感性", [(r["dataset"], r["code"], r["passes_of_27"]) for r in records])


if __name__ == "__main__":
    main()
