"""同库六量化器共享数据、目标及候选池的冻结预测比较。"""


import os
from common import OUT, ROOT, SEED, batched_score, correlation, load_frozen, module, positive_comparison_gain, risk_statistics, save_json, load_json
import numpy as np


ALPHA = 1.2
original = module("20_cross_quantizer_validation")
METRICS = ("dist_two", "dist_opposing", "comp_two", "comp_opposing", "dist_mnr", "distance_mse", "distance_risk")


def exact_workload(values, targets, offset):
    """精确近邻及标准剪枝先生成一次，所有量化器共享。"""
    unit = values.astype(np.float64)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    distances = 1 - unit[targets] @ unit.T
    result = {role: {key: [] for key in ("left_a", "right_a", "left_b", "right_b", "exact", "query", "tie_flip")} for role in ("ranking", "pruning")}
    pools = []
    for qi, target in enumerate(targets):
        distances[qi, target] = np.inf
        pool = np.lexsort((np.arange(len(values)), distances[qi]))[:48]
        pools.append(pool + offset)
        top = pool[0]
        for candidate in pool[1:]:
            row = result["ranking"]
            row["left_a"].append(target + offset)
            row["right_a"].append(candidate + offset)
            row["left_b"].append(target + offset)
            row["right_b"].append(top + offset)
            row["exact"].append(float(distances[qi, candidate] - distances[qi, top]))
            row["query"].append(qi)
            row["tie_flip"].append(candidate < top)
        selected = []
        for candidate in pool:
            if len(selected) == 24:
                break
            dominated = False
            for neighbor in selected:
                psi = float(distances[qi, candidate] - ALPHA * (1 - unit[candidate] @ unit[neighbor]))
                row = result["pruning"]
                row["left_a"].append(target + offset)
                row["right_a"].append(candidate + offset)
                row["left_b"].append(candidate + offset)
                row["right_b"].append(neighbor + offset)
                row["exact"].append(psi)
                row["query"].append(qi)
                row["tie_flip"].append(False)
                if psi >= 0:
                    dominated = True
                    break
            if not dominated:
                selected.append(candidate)
    for role in result:
        result[role] = {key: np.asarray(values, dtype=np.float64 if key == "exact" else bool if key == "tie_flip" else np.int64) for key, values in result[role].items()}
    return result, np.asarray(pools)


def role_approximate(records, score, intercept, slope, role):
    first = -batched_score(score, records["left_a"], records["right_a"])
    second = -batched_score(score, records["left_b"], records["right_b"])
    coefficient = 1.0 if role == "ranking" else ALPHA
    return (1 - coefficient) * intercept + slope * (first - coefficient * second)


def flip_labels(role, records, approximate):
    if role == "ranking":
        return (approximate < 0) | ((approximate == 0) & records["tie_flip"])
    return (records["exact"] >= 0) != (approximate >= 0)


def agreement(prediction, outcome):
    correct, ties, count = 0, 0, 0
    for left in range(len(outcome)):
        for right in range(left + 1, len(outcome)):
            actual = outcome[left] - outcome[right]
            if actual == 0:
                continue
            delta = prediction[left] - prediction[right]
            count += 1
            correct += delta * actual > 0
            ties += delta == 0
    return {"concordant": int(correct), "predicted_ties": int(ties), "untied_pairs": count,
            "agreement": float((correct + .5 * ties) / count) if count else 1.0}


def summarize(records, query_counts, query_failures):
    output = {}
    rng = np.random.default_rng(SEED + 171)
    bootstrap = rng.integers(0, 160, size=(500, 160))
    for role in ("ranking", "pruning"):
        rates = np.asarray([record["roles"][role]["flip_rate"] for record in records])
        failures = np.stack([record[role] for record in query_failures])
        totals = query_counts[role]
        boot_rates = failures[:, bootstrap].sum(axis=2) / totals[bootstrap].sum(axis=1)
        output[role] = {}
        for metric in METRICS:
            prediction = np.asarray([record["roles"][role]["predictors"][metric] for record in records])
            estimates = np.asarray([correlation(prediction, boot_rates[:, index]) for index in range(500)])
            output[role][metric] = {
                "spearman": correlation(prediction, rates),
                "conditional_query_bootstrap95": np.quantile(estimates, [.025, .975]).tolist(),
                **agreement(prediction, rates),
            }
    return output


results = []
for dataset in ("cohere", "minilm", "sift128", "gist960"):
    checkpoint = OUT / f"six-codes-{dataset}.json"
    if checkpoint.exists():
        previous = load_json(checkpoint)
        assert previous["dataset"] == dataset and len(previous["records"]) == 6
        assert (OUT / f"six-codes-{dataset}-arrays.npz").exists()
        results.append(previous)
        print(f"复用完整检查点：{dataset}", flush=True)
        continue
    values = load_frozen(dataset)
    dimension = values.shape[1]
    order = np.random.default_rng(SEED + dimension).permutation(4000)
    train = values[order[:1200]]
    calibration = values[order[1200:2000]]
    evaluation = values[order[2000:]]
    combined = np.concatenate([calibration, evaluation])
    fit_targets = np.random.default_rng(SEED + 31).choice(800, 96, replace=False)
    eval_targets = np.random.default_rng(SEED + 32).choice(2000, 160, replace=False)
    fit_records, fit_pools = exact_workload(calibration, fit_targets, 0)
    eval_records, eval_pools = exact_workload(evaluation, eval_targets, 800)
    rng = np.random.default_rng(SEED + 33)
    left = rng.integers(0, 800, 24000)
    right = (left + rng.integers(1, 800, 24000)) % 800
    unit = calibration.astype(np.float64)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    exact_pair = 1 - np.sum(unit[left] * unit[right], axis=1)
    evidence = {"split_order": order, "fit_targets": fit_targets, "eval_targets": eval_targets,
                "fit_pools": fit_pools, "eval_pools": eval_pools, "distance_left": left,
                "distance_right": right, "distance_exact": exact_pair}
    query_counts = {role: np.bincount(eval_records[role]["query"], minlength=160) for role in ("ranking", "pruning")}
    records, query_failures = [], []
    for code in original.QUANTIZERS:
        score, _ = original.build_quantizer(code, train, combined)
        raw_pair = -batched_score(score, left, right)
        intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(left)), raw_pair]), exact_pair, rcond=None)[0]
        admissible = bool(np.ptp(raw_pair) > 0 and slope > 0)
        if not admissible:
            slope = 0.0
            intercept = float(np.mean(exact_pair))
            print(f"保留无有效正斜率的常量距离对照：{dataset}/{code}", flush=True)
        mse = float(np.mean((intercept + slope * raw_pair - exact_pair) ** 2))
        fidelity = correlation(raw_pair, exact_pair)
        row = {"code": code, "distance_model": {"intercept": float(intercept), "slope": float(slope), "mse": mse, "spearman": fidelity, "positive_slope_admissible": admissible}, "roles": {}}
        failures = {}
        evidence[f"{code}_distance_raw"] = raw_pair
        for role in ("ranking", "pruning"):
            fit_approx = role_approximate(fit_records[role], score, intercept, slope, role)
            approximate = role_approximate(eval_records[role], score, intercept, slope, role)
            exact = fit_records[role]["exact"]
            dist = risk_statistics(exact, fit_approx)
            gain = positive_comparison_gain(exact, fit_approx)
            comp = risk_statistics(exact, gain * fit_approx) if gain > 0 else {"two_sided": 1.0, "opposing": 1.0}
            flip = flip_labels(role, eval_records[role], approximate)
            if gain > 0:
                assert np.array_equal(flip, flip_labels(role, eval_records[role], gain * approximate))
            for name in ("two_sided", "opposing"):
                tau = dist[f"tau_{name}"]
                if tau is not None:
                    opposed = -np.sign(exact) * (fit_approx - exact)
                    assert np.all(~flip_labels(role, fit_records[role], fit_approx) | (np.abs(exact) <= tau) | (opposed >= tau))
            predictors = {"dist_two": dist["two_sided"], "dist_opposing": dist["opposing"],
                          "comp_two": comp["two_sided"], "comp_opposing": comp["opposing"],
                          "dist_mnr": dist["minus_mnr20"], "distance_mse": mse, "distance_risk": 1 - fidelity}
            row["roles"][role] = {"predictors": predictors, "distance_fit": dist, "comparison_fit": comp,
                                  "comparison_gain": gain, "comparison_admissible": gain > 0,
                                  "flip_rate": float(np.mean(flip)), "comparison_count": len(flip)}
            failures[role] = np.bincount(eval_records[role]["query"], weights=flip, minlength=160)
            evidence[f"{code}_{role}_fit_approx"] = fit_approx
            evidence[f"{code}_{role}_eval_approx"] = approximate
            evidence[f"{code}_{role}_eval_flip"] = flip
        records.append(row)
        query_failures.append(failures)
        del score
        print(f"同库量化器比较完成：{dataset}/{code}，排序翻转{row['roles']['ranking']['flip_rate']:.4f}，剪枝翻转{row['roles']['pruning']['flip_rate']:.4f}", flush=True)
    for role in fit_records:
        for key, array in fit_records[role].items():
            evidence[f"{role}_fit_{key}"] = array
        for key, array in eval_records[role].items():
            evidence[f"{role}_eval_{key}"] = array
    stats = summarize(records, query_counts, query_failures)
    result = {"dataset": dataset, "dimension": dimension, "records": records, "statistics": stats}
    np.savez_compressed(OUT / f"six-codes-{dataset}-arrays.npz", **evidence)
    save_json(f"six-codes-{dataset}.json", result)
    results.append(result)
save_json("six-codes.json", {"results": results, "primary": "dist_two", "bootstrap": "500次共享评估query簇重采样，拟合统计冻结"})
print("同库六量化器比较全部完成。", flush=True)
