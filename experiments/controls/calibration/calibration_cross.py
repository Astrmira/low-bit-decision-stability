"""冻结比较拟合与距离拟合系数，交叉复核旋转效应及残差单位。"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import importlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(os.environ.get("JMLR_CALIBRATION_DIR", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
original = importlib.import_module("09_covariance_certificate")
grouping = importlib.import_module("29_candidate_selection_covariance")
DATA = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data"))
original.ROOT = DATA
grouping.ROOT = DATA


def records_for(values, queries):
    records = original.build_exact_records(values, queries)
    for record in records:
        record["exact_scores"] = np.sum(
            values[record["candidates"]] * values[record["query_index"]],
            axis=1,
            dtype=np.float64,
        )
    return records


def main():
    first = original.load_sampled_dataset("cohere")
    query_ids = np.random.default_rng(original.SEED).choice(4000, 240, replace=False)
    second, _ = grouping.load_values("cohere")
    second_ids = np.random.default_rng(grouping.SEED + 768).choice(6000, 160, replace=False)
    populations = [
        ("4K-fit80", first, query_ids[:80]),
        ("4K-heldout160", first, query_ids[80:]),
        ("6K-160", second, second_ids),
    ]
    prior = json.loads((ROOT / "results/probe26_covariance_certificate.json").read_text(encoding="utf-8"))
    prior = next(row for row in prior["results"] if row["dataset"] == "cohere")["quantizers"]
    old_cross = json.loads((ROOT / "controls/covariance/covariance-crosscheck.json").read_text(encoding="utf-8"))
    old_controls = json.loads((ROOT / "controls/covariance/covariance-controls.json").read_text(encoding="utf-8"))
    global_prior = json.loads((ROOT / "results/probe46_candidate_selection_covariance.json").read_text(encoding="utf-8"))
    global_prior = next(row for row in global_prior["results"] if row["dataset"] == "cohere")["quantizers"]["2bit"]["calibration"]
    output, arrays = [], {}
    for code in ("2bit", "rotated_2bit"):
        transformed_fit = original.randomized_orthogonal_transform(second) if code.startswith("rotated") else second
        encoded_fit = grouping.encode(transformed_fit)
        distance_model = grouping.fit_affine(second, encoded_fit, 2, np.random.default_rng(20261110))
        if code == "2bit":
            distance_model = global_prior
        comparison_a = prior[code]["calibration_slope"]
        models = {"comparison": {"intercept": 0.0, "slope": comparison_a}, "distance": distance_model}
        for population, values, queries in populations:
            transformed = original.randomized_orthogonal_transform(values) if code.startswith("rotated") else values
            records = original.add_quantized_scores(records_for(values, queries), original.encode(transformed), 2)
            exact_left, exact_right, raw_left, raw_right = original.flatten_pair_records(records)
            ids = np.repeat(np.arange(len(queries)), original.LOCAL_K - 1)
            exact_margin = exact_left - exact_right
            raw_margin = raw_left - raw_right
            for calibration, model in models.items():
                a, b = model["slope"], model["intercept"]
                left = b + a * raw_left - exact_left
                right = b + a * raw_right - exact_right
                residual = left - right
                metrics = grouping.pair_metrics(left, right, ids, fixed_anchor=True)
                row = {
                    "code": code, "population": population, "calibration": calibration,
                    "a_raw_to_metric": a, "b_raw_to_metric": b,
                    "c_Q_raw": 1 / a, "c_Q_metric": 1.0,
                    "mu_R_metric": float(np.mean(residual)),
                    "mu_R_raw": float(np.mean(residual) / a),
                    "v_raw": float(np.std(residual, ddof=1) / a),
                    "v_over_c_Q": float(np.std(residual, ddof=1)),
                    "metric_variance": float(np.var(residual, ddof=1)),
                    "mean_distance_variance": float((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2),
                    "flip_rate": float(np.mean((raw_margin <= 0) & (exact_margin > 0))),
                    "correlation": metrics["pooled_correlation"],
                    "covariance": metrics["pooled_covariance"],
                    "cancellation_factor": metrics["independence_ratio"],
                    "query_count": len(queries), "comparison_count": len(residual),
                }
                assert abs(row["metric_variance"] - row["v_over_c_Q"] ** 2) < 1e-14
                if code == "2bit" and population == "4K-fit80":
                    expected = old_cross["original_4k_fit"]["ranking_margin" if calibration == "comparison" else "global_affine"]
                    assert abs(row["correlation"] - expected["pooled_correlation"]) < 1e-6
                output.append(row)
                key = f"{code}_{population}_{calibration}"
                arrays[key + "_left"] = left
                arrays[key + "_right"] = right
                arrays[key + "_exact_margin"] = exact_margin
                arrays[key + "_raw_margin"] = raw_margin
                print("已计算", code, population, calibration, "相关", round(row["correlation"], 4), "比较方差", row["metric_variance"], flush=True)
    np.savez_compressed(OUT / "calibration-cross-arrays.npz", **arrays)
    (OUT / "calibration-cross.json").write_text(json.dumps({
        "rotation": "两个人口均使用原Probe26同一个signed-DCT旋转，隔离抽样变化",
        "units": "a把原始整数分数映射到精确内积单位；原始c_Q=1/a；度量残差c_Q=1；v_raw/c_Q等于度量残差标准差",
        "rows": output,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
