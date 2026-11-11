"""按冻结拟合规则交叉报告单侧、双侧及MNR，保留全部人口。"""


import os
import sys
from pathlib import Path

from common import CACHE, OLD, OUT, ROOT, load_json, module, positive_comparison_gain, risk_statistics, save_json
import numpy as np


rows = []
arrays = {}


def record(population, code, role, calibration, fit_exact, fit_approx, eval_exact, eval_approx, eval_flip, metadata):
    stats = risk_statistics(fit_exact, fit_approx)
    residual = eval_approx - eval_exact
    opposing = -np.sign(eval_exact) * residual
    row = {
        "population": population, "code": code, "role": role, "calibration": calibration,
        "fit_count": len(fit_exact), "eval_count": len(eval_exact),
        "fit": stats, "eval_flip": float(np.mean(eval_flip)),
        "eval_bias": float(np.mean(residual)), "eval_scale": float(np.std(residual, ddof=1)),
        **metadata,
    }
    for name, errors in (("two_sided", np.abs(residual)), ("opposing", opposing)):
        tau = stats[f"tau_{name}"]
        row[f"eval_at_frozen_tau_{name}"] = 1.0 if tau is None else min(1.0, float(np.mean(np.abs(eval_exact) <= tau) + np.mean(errors >= tau)))
    rows.append(row)
    key = f"{population}_{code}_{role}_{calibration}".replace("/", "-")
    arrays[key + "_fit_exact"] = fit_exact
    arrays[key + "_fit_approx"] = fit_approx
    arrays[key + "_eval_exact"] = eval_exact
    arrays[key + "_eval_approx"] = eval_approx
    arrays[key + "_eval_flip"] = eval_flip


def audit_e2():
    with np.load(OLD / "calibration-cross-arrays.npz") as source:
        for code in ("2bit", "rotated_2bit"):
            for calibration in ("comparison", "distance"):
                fit_key = f"{code}_4K-fit80_{calibration}"
                fit_exact = source[fit_key + "_exact_margin"]
                fit_approx = fit_exact + source[fit_key + "_left"] - source[fit_key + "_right"]
                for population in ("4K-heldout160", "6K-160"):
                    key = f"{code}_{population}_{calibration}"
                    exact = source[key + "_exact_margin"]
                    approximate = exact + source[key + "_left"] - source[key + "_right"]
                    flip = (source[key + "_raw_margin"] <= 0) & (exact > 0)
                    record("E2-" + population, code, "ranking", calibration,
                           fit_exact, fit_approx, exact, approximate, flip,
                           {"fit_queries": 80, "eval_queries": 160, "calibration_identity": "原校准交叉的冻结系数"})
    print("E2两个人口的校准交叉已完成。", flush=True)


def audit_e1():
    base = module("16_p6_heldout_calibration")
    base.ROOT = CACHE
    splits, _, _ = base.load_splits("cohere")
    records = {
        "calibration": base.target_candidates(splits["calibration"], base.N_CALIBRATION_TARGETS, base.SEED + 101),
        "validation": base.target_candidates(splits["validation"], base.N_VALIDATION_TARGETS, base.SEED + 102),
    }
    prior = next(row for row in load_json(ROOT / "results/probe33_standard_vamana_p6_heldout_calibration.json")["results"] if row["dataset"] == "cohere")
    for code in ("2bit", "rotated_2bit"):
        encoded = {name: base.encode(base.rotate(values) if code.startswith("rotated") else values) for name, values in splits.items()}
        fit_targets, fit_pools = records["calibration"]
        model = base.fit_calibration(splits["calibration"], fit_targets, fit_pools, encoded["calibration"])
        assert abs(model["slope"] - prior["quantizers"][code]["distance_calibration"]["slope"]) < 1e-10
        decisions, signed = {}, {}
        for name in ("calibration", "validation"):
            targets, pools = records[name]
            decisions[name] = base.collect_decisions(splits[name], targets, pools, encoded[name], model)
            signed[name] = {
                "ranking": decisions[name]["ranking"]["margin"],
                "pruning": np.asarray([comparison[3] for target, pool in zip(targets, pools)
                                       for comparison in base.select_exact(splits[name], target, pool)[1]]),
            }
        for role in ("ranking", "pruning"):
            fit_exact = signed["calibration"][role]
            fit_approx = fit_exact + decisions["calibration"][role]["residual"]
            exact = signed["validation"][role]
            approximate = exact + decisions["validation"][role]["residual"]
            flips = decisions["validation"][role]["flip"]
            original = prior["quantizers"][code]["roles"][role]
            fitted = risk_statistics(fit_exact, fit_approx)
            assert abs(fitted["two_sided"] - original["risk_model"]["plugin_risk"]) < 1e-12
            assert abs(float(np.mean(flips)) - original["validation"]["flip_rate"]) < 1e-12
            gain = positive_comparison_gain(fit_exact, fit_approx)
            assert gain > 0
            for calibration, multiplier in (("distance", 1.0), ("comparison", gain)):
                record("E1-48-candidates", code, role, calibration,
                       fit_exact, multiplier * fit_approx, exact, multiplier * approximate, flips,
                       {"comparison_gain": multiplier, "distance_fit": model,
                        "calibration_identity": "原48目标近邻单距离仿射拟合，比较对照为该比较输出的正比例拟合"})
    print("E1原风险分数及召回翻转率已逐项重现，校准对照完成。", flush=True)


def audit_e8():
    sys.path.insert(0, str(ROOT / "controls/calibration"))
    graph_root = Path(os.environ.get("JMLR_GRAPH_DIR", OLD))
    graph = module("graph_analysis")
    prior = load_json(graph_root / "graph-analysis.json")["diagnostics"]
    for job in load_json(graph_root / "graph-jobs.json")["jobs"]:
        dataset, seed, axis, dimension = job["dataset"], job["seed"], job["axis"], job["dim"]
        folder = graph_root / "graph-data" / dataset / str(seed) / axis
        result = graph_root / "graph-results" / dataset / str(seed) / axis
        base = np.memmap(folder / "base.f32", dtype=np.float32, mode="r", shape=(200000, dimension))
        queries = np.fromfile(folder / "queries.f32", dtype=np.float32).reshape(744, dimension)
        traces = graph.read_trace(result / "exact_l0-ef128.trace.u32")
        exact_distances, full_scores, cheap_scores = [], [], []
        for qi, block in enumerate(traces):
            selected = np.argsort(block[:, 2], kind="stable")[:32]
            ids = block[selected, 0]
            values, query = base[ids].astype(np.float64), queries[qi].astype(np.float64)
            exact = 1 - values @ query / (np.linalg.norm(values, axis=1) * np.linalg.norm(query))
            order = np.lexsort((ids, exact))
            ids, exact = ids[order], exact[order]
            full, cheap = graph.raw_scores(queries[qi], np.asarray(base[ids]))
            raw_map = {int(row[0]): (int(row[1]), int(row[3])) for row in block}
            assert np.array_equal(full, [raw_map[int(node)][1] for node in ids])
            assert np.array_equal(cheap, [raw_map[int(node)][0] for node in ids])
            exact_distances.append(exact)
            full_scores.append(full)
            cheap_scores.append(cheap)
        exact_distances = np.asarray(exact_distances)
        exact = (exact_distances[:, 1:] - exact_distances[:, :1]).ravel()
        for scorer, scores in (("full_sm", full_scores), ("native_cheap", cheap_scores)):
            scores = np.asarray(scores, dtype=np.float64)
            raw_margin = (scores[:, 1:] - scores[:, :1]).ravel()
            fit_count = 128 * 31
            slope = positive_comparison_gain(exact[:fit_count], raw_margin[:fit_count])
            approximate = slope * raw_margin
            old = next(row for row in prior if row["dataset"] == dataset and row["seed"] == seed
                       and row["axis"] == axis and row["ef"] == 128 and row["scorer"] == scorer)
            old_summary = graph.model_summary(exact, raw_margin)
            assert abs(old_summary["risk_score"] - old["risk_score"]) < 1e-10
            assert abs(old_summary["heldout_flip_rate"] - old["heldout_flip_rate"]) < 1e-12
            design = np.column_stack([np.ones(128 * 32), scores[:128].ravel()])
            intercept, distance_slope = np.linalg.lstsq(design, exact_distances[:128].ravel(), rcond=None)[0]
            for calibration, calibrated in (("comparison", approximate), ("distance", distance_slope * raw_margin)):
                record(f"E8-{dataset}-{seed}", axis + "-" + scorer, "ranking", calibration,
                       exact[:fit_count], calibrated[:fit_count], exact[fit_count:], calibrated[fit_count:],
                       (raw_margin[fit_count:] <= 0) & (exact[fit_count:] > 0),
                       {"positive_slope": bool((slope if calibration == "comparison" else distance_slope) > 0),
                        "distance_slope": float(distance_slope), "comparison_slope": slope,
                        "distance_intercept": float(intercept), "calibration_identity": "同一FP32已访问候选池的128拟合查询"})
        del base, traces
        print(f"E8轨迹校准复算完成：{dataset}，{seed}，{axis}", flush=True)


audit_e2()
audit_e1()
audit_e8()
np.savez_compressed(OUT / "calibration-audit-arrays.npz", **arrays)
save_json("calibration-audit.json", {"rows": rows, "scope": "阈值与系数均在拟合数据确定，所有固定变体同时报告"})
print(f"校准核验完成，共{len(rows)}个固定比较位置。", flush=True)
