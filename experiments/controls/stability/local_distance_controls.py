"""在同一决策池的距离端点上补强保真度与MSE对照。"""


import os
import csv

from common import OUT, SEED, batched_score, correlation, load_frozen, load_json, module, save_json
import numpy as np


quantizers = module("20_cross_quantizer_validation")
primary = load_json(OUT / "six-codes.json")["results"]
results = []
csv_rows = []
for dataset in primary:
    name = dataset["dataset"]
    values = load_frozen(name)
    with np.load(OUT / f"six-codes-{name}-arrays.npz") as archive:
        order = archive["split_order"]
        train = values[order[:1200]]
        combined = np.concatenate([values[order[1200:2000]], values[order[2000:]]])
        unit = combined.astype(np.float64)
        unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
        pairs_by_role = {}
        for role in ("ranking", "pruning"):
            first = np.column_stack([archive[f"{role}_fit_left_a"], archive[f"{role}_fit_right_a"]])
            second = np.column_stack([archive[f"{role}_fit_left_b"], archive[f"{role}_fit_right_b"]])
            repeated = np.concatenate([first, second])
            unique = np.unique(repeated, axis=0)
            pairs_by_role[role] = (unique, repeated)
        raw_arrays = {}
        records = []
        for old in dataset["records"]:
            code = old["code"]
            score, _ = quantizers.build_quantizer(code, train, combined)
            left, right = archive["distance_left"][:64], archive["distance_right"][:64]
            assert np.allclose(-batched_score(score, left, right), archive[f"{code}_distance_raw"][:64], rtol=0, atol=1e-10)
            intercept, slope = old["distance_model"]["intercept"], old["distance_model"]["slope"]
            record = {"code": code, "roles": {}}
            for role, populations in pairs_by_role.items():
                record["roles"][role] = {}
                for population, pairs in zip(("unique", "endpoint_weighted"), populations):
                    exact = 1 - np.sum(unit[pairs[:, 0]] * unit[pairs[:, 1]], axis=1)
                    raw = -batched_score(score, pairs[:, 0], pairs[:, 1])
                    approximate = intercept + slope * raw
                    values_by_metric = {"distance_mse": float(np.mean((approximate - exact) ** 2)),
                                        "distance_risk": 1 - correlation(raw, exact), "pair_count": len(pairs)}
                    record["roles"][role][population] = values_by_metric
                    key = f"{code}_{role}_{population}"
                    raw_arrays[key + "_pairs"] = pairs
                    raw_arrays[key + "_exact"] = exact
                    raw_arrays[key + "_raw"] = raw
                    raw_arrays[key + "_approximate"] = approximate
            records.append(record)
            print(f"同池距离对照完成：{name}/{code}", flush=True)
        stats = {}
        for role in ("ranking", "pruning"):
            observed = [record["roles"][role]["flip_rate"] for record in dataset["records"]]
            counts = np.bincount(archive[f"{role}_eval_query"], minlength=160)
            failures = np.stack([np.bincount(archive[f"{role}_eval_query"], weights=archive[f"{record['code']}_{role}_eval_flip"], minlength=160) for record in records])
            draws = np.random.default_rng(SEED + 171).integers(0, 160, size=(500, 160))
            boot = failures[:, draws].sum(axis=2) / counts[draws].sum(axis=1)
            stats[role] = {}
            for population in ("unique", "endpoint_weighted"):
                stats[role][population] = {}
                for metric in ("distance_mse", "distance_risk"):
                    predictor = np.asarray([record["roles"][role][population][metric] for record in records])
                    estimates = [correlation(predictor, boot[:, index]) for index in range(500)]
                    interval = np.quantile(estimates, [.025, .975])
                    point = correlation(predictor, observed)
                    stats[role][population][metric] = {"spearman": point, "conditional_query_bootstrap95": interval.tolist()}
                    csv_rows.append({"dataset": name, "role": role, "population": population, "metric": metric,
                                     "rho": point, "ci_low": interval[0], "ci_high": interval[1]})
        np.savez_compressed(OUT / f"local-distance-{name}-arrays.npz", **raw_arrays)
        results.append({"dataset": name, "records": records, "statistics": stats})
save_json("local-distance-controls.json", {"results": results, "scope": "只用拟合决策的唯一距离端点对计算局部指标，频率加权版本同时保留；距离映射与原结果相同"})
with (OUT / "local-distance-controls.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
print("同候选池距离对照全部完成。", flush=True)
