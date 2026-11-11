"""复核评审所指的校准、协方差、预测器与数学边界，保留原始结果。"""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CACHE = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data"))
SEED = 20261109


def save(name, value):
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(left, right):
    result = float(spearmanr(left, right).statistic)
    return result if np.isfinite(result) else None


def predictor_controls():
    """在现有校准结果上预定组件对照，按数据集整簇重采样。"""
    path = ROOT / "results/probe33_standard_vamana_p6_heldout_calibration.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for role in ["ranking", "pruning"]:
        records = []
        for dataset in old["results"]:
            for quantizer, result in dataset["quantizers"].items():
                model = result["roles"][role]["risk_model"]
                validation = result["roles"][role]["validation"]
                scale = model["residual_scale"]
                cutoff = model["retention_cutoffs"][3]
                records.append({
                    "dataset": dataset["dataset"],
                    "quantizer": quantizer,
                    "eligible": result["eligibility"]["eligible"],
                    "flip": validation["flip_rate"],
                    "full": validation["frozen_calibration_risk"],
                    "boundary_quantile": -cutoff * scale,
                    "residual_scale": scale,
                    "standardized_quantile": -cutoff,
                })
        # 原协议使用80%保留率，故第4个阈值恰为精确margin的20%分位点。
        metrics = ["full", "boundary_quantile", "residual_scale", "standardized_quantile"]
        names = list(dict.fromkeys(row["dataset"] for row in records))
        groups = [np.array([i for i, row in enumerate(records) if row["dataset"] == name])
                  for name in names]
        assert all(len(group) == 2 for group in groups)
        rng = np.random.default_rng(SEED)
        drawn = rng.integers(0, len(groups), (2000, len(groups)))
        indices = np.stack(groups)[drawn].reshape(2000, -1)
        truth = np.array([row["flip"] for row in records])
        ranked_truth = rankdata(truth[indices], axis=1)
        ranked_truth -= ranked_truth.mean(axis=1, keepdims=True)
        bootstrap = {}
        summaries = {}
        for metric in metrics:
            scores = np.array([row[metric] for row in records])
            ranked = rankdata(scores[indices], axis=1)
            ranked -= ranked.mean(axis=1, keepdims=True)
            denominator = np.sqrt((ranked**2).sum(axis=1) * (ranked_truth**2).sum(axis=1))
            assert np.all(denominator > 0)
            bootstrap[metric] = (ranked * ranked_truth).sum(axis=1) / denominator
            eligible = np.array([row["eligible"] for row in records])
            leave_out = {
                name: correlation(scores[[row["dataset"] != name for row in records]],
                                  truth[[row["dataset"] != name for row in records]])
                for name in names
            }
            summaries[metric] = {
                "spearman": correlation(scores, truth),
                "eligible_spearman": correlation(scores[eligible], truth[eligible]),
                "dataset_cluster_95": np.quantile(bootstrap[metric], [.025, .975]).tolist(),
                "leave_one_dataset_out": leave_out,
            }
        output[role] = {
            "records": records,
            "metrics": summaries,
            "full_minus_component_95": {
                metric: np.quantile(bootstrap["full"] - bootstrap[metric], [.025, .975]).tolist()
                for metric in metrics[1:]
            },
        }
    save("predictor-controls.json", output)
    return output


def covariance_controls():
    """在同一批查询上隔离拟合目标，并精确分解组内与组间矩。"""
    module = importlib.import_module("09_covariance_certificate")
    grouping = importlib.import_module("29_candidate_selection_covariance")
    module.ROOT = CACHE
    vectors = module.load_sampled_dataset("cohere")
    rng = np.random.default_rng(module.SEED)
    queries = rng.choice(len(vectors), 240, replace=False)
    encoded = module.encode(vectors)
    calibration = module.add_quantized_scores(
        module.build_exact_records(vectors, queries[:80]), encoded, 2
    )
    evaluation = module.add_quantized_scores(
        module.build_exact_records(vectors, queries[80:]), encoded, 2
    )
    original = module.fit_model(calibration)
    x = np.concatenate([row["quantized_scores"] for row in calibration])
    y = np.concatenate([row["exact_scores"] for row in calibration])
    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    models = {
        "margin_fit": (0.0, original["slope"]),
        "edge_affine_fit": (float(intercept), float(slope)),
    }
    output = {"calibration_models": models, "splits": {}}
    arrays = {}
    for split, records in [("fit", calibration), ("heldout", evaluation)]:
        exact_top, exact_comp, code_top, code_comp = module.flatten_pair_records(records)
        query_ids = np.repeat(np.arange(len(records)), 31)
        output["splits"][split] = {}
        for name, (bias, factor) in models.items():
            left = bias + factor * code_top - exact_top
            right = bias + factor * code_comp - exact_comp
            metrics = grouping.pair_metrics(left, right, query_ids, fixed_anchor=True)
            assert abs(metrics["exact_covariance_identity_gap"]) < 1e-12
            assert abs(metrics["exact_difference_identity_gap"]) < 1e-12
            metrics["between_covariance_share"] = (
                metrics["exact_weighted_between_covariance"] / metrics["pooled_covariance"]
            )
            output["splits"][split][name] = metrics
            arrays[f"{split}_{name}_left"] = left
            arrays[f"{split}_{name}_right"] = right
        arrays[f"{split}_query_ids"] = query_ids
    old = json.loads((ROOT / "results/probe26_covariance_certificate.json").read_text(encoding="utf-8"))
    prior = next(row for row in old["results"] if row["dataset"] == "cohere")["quantizers"]["2bit"]
    output["original_ratio"] = prior["residual_structure"]["independence_over_empirical_ratio"]
    current = output["splits"]["fit"]["margin_fit"]["independence_ratio"]
    output["original_reproduction_absolute_gap"] = abs(current - output["original_ratio"])
    np.savez_compressed(OUT / "covariance-arrays.npz", **arrays)
    save("covariance-controls.json", output)
    return output


def fit_pruning_models(module, vectors, targets, candidates, encoded):
    exact, raw = [], []
    for target, pool in zip(targets, candidates):
        left = np.full(len(pool), target, dtype=np.int64)
        exact.extend(1 - vectors[pool] @ vectors[target])
        raw.extend(module.bq2_distances(left, pool, encoded))
        rng = np.random.default_rng(module.SEED + int(target))
        first = rng.choice(pool, 256, replace=True)
        second = rng.choice(pool, 256, replace=True)
        first, second = first[first != second], second[first != second]
        exact.extend(1 - np.sum(vectors[first] * vectors[second], axis=1))
        raw.extend(module.bq2_distances(first, second, encoded))
    exact, raw = np.asarray(exact), np.asarray(raw)
    origin_slope = float(raw @ exact / (raw @ raw))
    bias, affine_slope = np.linalg.lstsq(np.column_stack([np.ones(len(raw)), raw]), exact, rcond=None)[0]
    assert origin_slope > 0 and affine_slope > 0
    return {
        "original_origin": (0.0, origin_slope),
        "affine_slope_only": (0.0, float(affine_slope)),
        "affine": (float(bias), float(affine_slope)),
    }, {"fit_exact": exact, "fit_raw": raw}


def pruning_controls():
    """冻结候选池、顺序与参数，比较去截距前后的局部动作和输出。"""
    module = importlib.import_module("10_pruning_replay")
    module.ROOT = CACHE
    vectors = module.load_sampled_dataset("cohere")
    rng = np.random.default_rng(module.SEED)
    targets = rng.choice(len(vectors), 240, replace=False)
    fit_candidates = module.candidate_lists(vectors, targets[:80])
    eval_candidates = module.candidate_lists(vectors, targets[80:])
    old = json.loads((ROOT / "results/probe27_standard_vamana_pruning_replay.json").read_text(encoding="utf-8"))
    prior = next(row for row in old["results"] if row["dataset"] == "cohere")["quantizers"]
    output = {}
    for quantizer in module.QUANTIZERS:
        print(f"正在验证剪枝校准对照：{quantizer}", flush=True)
        transformed = module.randomized_orthogonal_transform(vectors) if quantizer.startswith("rotated") else vectors
        encoded = module.encode_bq2(transformed)
        models, arrays = fit_pruning_models(module, vectors, targets[:80], fit_candidates, encoded)
        rows = {name: [] for name in models}
        exact_rows = []
        local_values = {name: [] for name in models}
        labels = {name: [] for name in models}
        for target, pool in zip(targets[80:], eval_candidates):
            nodes, exact = module.distance_cache(vectors, target, pool)
            _, raw = module.distance_cache(vectors, target, pool, encoded, 1.0)
            positions = {int(node): index for index, node in enumerate(nodes)}
            reference = module.run_select(target, pool, nodes, exact)
            for name, (bias, slope) in models.items():
                approximate = bias + slope * raw
                fixed = module.run_select(target, pool, nodes, approximate,
                                          ordered_override=reference["ordered"])
                free = module.run_select(target, pool, nodes, approximate)
                original_set = set(reference["final_selected"])
                fixed_set, free_set = set(fixed["final_selected"]), set(free["final_selected"])
                rows[name].append([
                    len(original_set & fixed_set) / len(original_set | fixed_set),
                    len(original_set & free_set) / len(original_set | free_set),
                ])
                labels[name].append(list(fixed["final_selected"]) + [-1] * (32 - len(fixed_set)))
                for decision in reference["comparisons"]:
                    a, b, c = positions[int(target)], positions[decision["candidate"]], positions[decision["selected"]]
                    psi = exact[a, b] - module.ALPHA * exact[b, c]
                    approx_psi = approximate[a, b] - module.ALPHA * approximate[b, c]
                    local_values[name].append([psi, approx_psi])
            exact_rows.append(list(reference["final_selected"]) + [-1] * (32 - len(original_set)))
        summary = {"models": models, "variants": {}}
        for name, pairs in local_values.items():
            pairs = np.array(pairs)
            psi, approx = pairs[:, 0], pairs[:, 1]
            residual = approx - psi
            false_keep = (psi >= 0) & (approx < 0)
            false_prune = (psi < 0) & (approx >= 0)
            stats = np.array(rows[name])
            summary["variants"][name] = {
                "comparisons": len(pairs),
                "false_keep_rate": float(false_keep.mean()),
                "false_prune_rate": float(false_prune.mean()),
                "flip_rate": float((false_keep | false_prune).mean()),
                "residual_mean": float(residual.mean()),
                "residual_sd": float(residual.std(ddof=1)),
                "bias_over_sd": float(residual.mean() / residual.std(ddof=1)),
                "fixed_order_jaccard": float(stats[:, 0].mean()),
                "free_order_jaccard": float(stats[:, 1].mean()),
            }
            arrays[f"{name}_local"] = pairs
            arrays[f"{name}_jaccards"] = stats
            arrays[f"{name}_fixed_labels"] = np.array(labels[name], dtype=np.int64)
        arrays["exact_labels"] = np.array(exact_rows, dtype=np.int64)
        arrays["targets"] = targets
        summary["raw_scale_invariance"] = bool(
            np.array_equal(arrays["original_origin_fixed_labels"], arrays["affine_slope_only_fixed_labels"])
        )
        original_stats = summary["variants"]["original_origin"]
        summary["reproduction_gaps"] = {
            "local_flip": original_stats["flip_rate"] - prior[quantizer]["evaluation"]["local_disagreement_rate"],
            "fixed_jaccard": original_stats["fixed_order_jaccard"] - prior[quantizer]["evaluation"]["fixed_order_replay"]["mean_final_jaccard"],
        }
        np.savez_compressed(OUT / f"pruning-{quantizer}-arrays.npz", **arrays)
        output[quantizer] = summary
    save("pruning-controls.json", output)
    return output


def mathematical_audits():
    proof_dir = OUT / "mathematical-audits"
    proof_dir.mkdir(exist_ok=True)
    results = {}
    for name in ["25_c1_calibration_probability_audit", "26_c2_stein_magnitude_audit", "27_c3_contamination_trace_audit"]:
        print(f"正在复核数学与语义边界：{name}", flush=True)
        module = importlib.import_module(name)
        module.RESULTS_DIR = proof_dir
        module.main()
        results[name] = "passed"
    return results


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    scripts = ["09_covariance_certificate.py", "10_pruning_replay.py", "16_p6_heldout_calibration.py",
               "25_c1_calibration_probability_audit.py", "26_c2_stein_magnitude_audit.py",
               "27_c3_contamination_trace_audit.py", "29_candidate_selection_covariance.py"]
    save("protocol.json", {
        "seed": SEED,
        "blas_threads": 1,
        "covariance": "原4K向量、80拟合/160评估查询、32候选，匹配样本比较margin拟合与edge仿射拟合",
        "pruning": "原4K向量、80拟合/160评估target、64候选，原比例/仿射斜率/完整仿射三组，共同alpha1.2与M32",
        "predictors": "既有24配置，20%精确margin分位点、残差尺度及二者标准化分位点；12数据集整簇2000次bootstrap及留一法",
        "scope": "新的向量机制对照使用本地恢复的Cohere-1M，其他库只复核既有结果",
        "source_sha256": {name: sha256(ROOT / name) for name in scripts},
        "audit_script_sha256": sha256(Path(__file__)),
    })
    results = {"mathematical": mathematical_audits(), "predictors": predictor_controls()}
    print("正在复核Cohere协方差归因", flush=True)
    results["covariance"] = covariance_controls()
    results["pruning"] = pruning_controls()
    results["elapsed_seconds"] = time.monotonic() - started
    save("verification.json", results)
    print("复核已完成，证据已单独保存。", flush=True)


if __name__ == "__main__":
    main()
