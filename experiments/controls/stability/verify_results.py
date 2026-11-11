"""从保留数组独立复算统计并核对冻结输入，导出逐项对照。"""


import os
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


OUT = Path(os.environ.get("JMLR_OUTPUT_DIR", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_risks(exact, estimate):
    residual = estimate - exact
    margins = np.abs(exact)
    positive = margins[margins > 0]
    if not len(positive):
        return 1.0, 1.0
    grid = np.unique(np.quantile(positive, [.02, .04, .08, .12, .18, .25, .35, .5, .7]))
    two, one = [], []
    for threshold in grid:
        boundary = np.count_nonzero(margins <= threshold) / len(margins)
        two.append(min(1.0, boundary + np.count_nonzero(np.abs(residual) >= threshold) / len(margins)))
        one.append(min(1.0, boundary + np.count_nonzero(-np.sign(exact) * residual >= threshold) / len(margins)))
    return min(two), min(one)


def rho(left, right):
    return 0.0 if np.ptp(left) == 0 or np.ptp(right) == 0 else float(spearmanr(left, right).statistic)


protocol = load(Path(__file__).resolve().parent / "protocol.json")
for path, expected in protocol["source_sha256"].items():
    resolved = Path(path) if Path(path).is_absolute() else ROOT.parent / path
    assert digest(resolved) == expected, f"冻结来源发生变化：{resolved}"
calibration = load(OUT / "calibration-audit.json")["rows"]
with np.load(OUT / "calibration-audit-arrays.npz") as arrays:
    for row in calibration:
        key = f"{row['population']}_{row['code']}_{row['role']}_{row['calibration']}".replace("/", "-")
        exact, estimate = arrays[key + "_fit_exact"], arrays[key + "_fit_approx"]
        two, one = direct_risks(exact, estimate)
        assert abs(two - row["fit"]["two_sided"]) < 1e-12
        assert abs(one - row["fit"]["opposing"]) < 1e-12
        assert one <= two + 1e-12
        assert np.mean(arrays[key + "_eval_flip"]) == row["eval_flip"]

six = load(OUT / "six-codes.json")["results"]
six_csv, paired = [], []
metrics = ["dist_two", "dist_opposing", "comp_two", "comp_opposing", "dist_mnr", "distance_mse", "distance_risk"]
for result in six:
    dataset = result["dataset"]
    with np.load(OUT / f"six-codes-{dataset}-arrays.npz") as arrays:
        assert len(np.unique(arrays["split_order"])) == 4000
        assert len(np.unique(arrays["fit_targets"])) == 96
        assert len(np.unique(arrays["eval_targets"])) == 160
        model_fit_exact = arrays["distance_exact"]
        query_draws = np.random.default_rng(20261111 + 171).integers(0, 160, size=(500, 160))
        for role in ("ranking", "pruning"):
            counts = np.bincount(arrays[f"{role}_eval_query"], minlength=160)
            failures, flip_rates = [], []
            for record in result["records"]:
                code = record["code"]
                model = record["distance_model"]
                raw = arrays[f"{code}_distance_raw"]
                if model.get("positive_slope_admissible", True):
                    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(raw)), raw]), model_fit_exact, rcond=None)[0]
                    assert abs(intercept - model["intercept"]) < 1e-12 and abs(slope - model["slope"]) < 1e-12
                else:
                    assert np.ptp(raw) == 0 and model["slope"] == 0
                exact = arrays[f"{role}_fit_exact"]
                estimate = arrays[f"{code}_{role}_fit_approx"]
                two, one = direct_risks(exact, estimate)
                predictors = record["roles"][role]["predictors"]
                assert abs(two - predictors["dist_two"]) < 1e-12
                assert abs(one - predictors["dist_opposing"]) < 1e-12
                gain = float(estimate @ exact / (estimate @ estimate)) if np.any(estimate) else 0.0
                if gain > 0:
                    comp_two, comp_one = direct_risks(exact, gain * estimate)
                    assert abs(comp_two - predictors["comp_two"]) < 1e-12
                    assert abs(comp_one - predictors["comp_opposing"]) < 1e-12
                labels = arrays[f"{code}_{role}_eval_flip"]
                eval_estimate = arrays[f"{code}_{role}_eval_approx"]
                if role == "ranking":
                    independent_labels = (eval_estimate < 0) | ((eval_estimate == 0) & arrays[f"{role}_eval_tie_flip"])
                else:
                    independent_labels = (arrays[f"{role}_eval_exact"] >= 0) != (eval_estimate >= 0)
                assert np.array_equal(labels, independent_labels)
                assert np.mean(labels) == record["roles"][role]["flip_rate"]
                flip_rates.append(float(np.mean(labels)))
                failures.append(np.bincount(arrays[f"{role}_eval_query"], weights=labels, minlength=160))
            failures = np.asarray(failures)
            boot_rates = failures[:, query_draws].sum(axis=2) / counts[query_draws].sum(axis=1)
            boot_rhos = {}
            for metric in metrics:
                predictor = np.asarray([record["roles"][role]["predictors"][metric] for record in result["records"]])
                assert abs(rho(predictor, flip_rates) - result["statistics"][role][metric]["spearman"]) < 1e-12
                samples = np.asarray([rho(predictor, boot_rates[:, index]) for index in range(500)])
                boot_rhos[metric] = samples
                interval = np.quantile(samples, [.025, .975])
                assert np.allclose(interval, result["statistics"][role][metric]["conditional_query_bootstrap95"], atol=1e-12)
                six_csv.append({"dataset": dataset, "role": role, "predictor": metric,
                                "rho": rho(predictor, flip_rates), "ci_low": interval[0], "ci_high": interval[1],
                                "pair_agreement": result["statistics"][role][metric]["agreement"]})
            for first in ("dist_two", "dist_opposing", "comp_two", "comp_opposing"):
                for second in ("dist_mnr", "distance_mse", "distance_risk"):
                    interval = np.quantile(boot_rhos[first] - boot_rhos[second], [.025, .975])
                    paired.append({"dataset": dataset, "role": role, "first": first, "second": second,
                                   "rho_difference": result["statistics"][role][first]["spearman"] - result["statistics"][role][second]["spearman"],
                                   "conditional_query_bootstrap95": interval.tolist()})

blockers = load(OUT / "blockers.json")["results"]
blocker_csv = []
for result in blockers:
    for code, metrics_by_type in result["codes"].items():
        with np.load(OUT / f"blockers-{result['dataset']}-{code}-arrays.npz") as arrays:
            assert np.all(arrays["edge_strongest"] <= arrays["edge_all_sum"] + 1e-10)
            rows = {"edge": (arrays["edge_all_sum"], arrays["edge_strongest"], arrays["edge_failure"], metrics_by_type["edges"])}
            for length in (1, 2, 3, 4):
                rows[f"path{length}"] = (arrays[f"length{length}_all_sum"], arrays[f"length{length}_strongest"], arrays[f"length{length}_failure"], metrics_by_type["paths"][str(length)])
            for level, (plain, tight, failure, summary) in rows.items():
                assert np.all(tight <= plain + 1e-10)
                assert np.mean(tight >= 1) == summary["strongest_saturation"]
                assert np.mean(plain >= 1) == summary["all_sum_saturation"]
                assert np.mean(failure) == summary["failure_rate"]
                blocker_csv.append({"dataset": result["dataset"], "code": code, "level": level, **summary})

def write_csv(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


write_csv("six-codes-statistics.csv", six_csv)
write_csv("blocker-comparison.csv", blocker_csv)
write_csv("calibration-comparison.csv", [{
    **{key: row[key] for key in ("population", "code", "role", "calibration", "fit_count", "eval_count", "eval_flip")},
    **row["fit"],
} for row in calibration])
(OUT / "paired-query-contrasts.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
receipt = {"source_hashes_unchanged": True, "calibration_positions": len(calibration),
           "six_quantizer_configurations": sum(len(row["records"]) for row in six),
           "datasets": len(six), "blocker_configurations": sum(len(row["codes"]) for row in blockers),
           "path_positions": sum(len(code["paths"]) for row in blockers for code in row["codes"].values()),
           "independent_array_recalculation": True,
           "sha256": {path.name: digest(path) for path in OUT.glob("*.npz")}}
(OUT / "independent-verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print("独立复算完成：48个校准位置、24个量化器配置、6组阻挡者及24组路径位置均核对通过。")
