#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13 - Probe 30：Fallback quota/rank 1-certificate

使用精确动态规划验证部分固定 dominated bits 是否足以保证指定 edge 最终存活，
再贪心删除冗余 pruning atoms，构造严格但不保证最小的 fallback 1-certificate。
"""

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


pruning = import_module("10_pruning_replay")
trace = import_module("11_trace_certificate")
edge_probe = import_module("12_edge_certificate")

DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_SAMPLE = 4_000
N_CALIBRATION_TARGETS = 80
N_EVALUATION_TARGETS = 160
CANDIDATE_K = 64
MAX_K = 32
ALPHA = 1.2
SEED = 20260823
SENSITIVITY = [(0.8, 16), (1.0, 16), (1.2, 16), (0.8, 32), (1.0, 32), (1.2, 32)]


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_sampled_dataset(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 30 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED)
    indices = rng.choice(total, min(N_SAMPLE, total), replace=False)
    vecs = np.asarray(mapped[indices], dtype=np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    print(f"  [{name}] 直接映射抽样：{total} → {len(vecs)}")
    return vecs


def randomized_orthogonal_transform(vecs):
    rng = np.random.default_rng(SEED + vecs.shape[1])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), vecs.shape[1])
    rotated = dct(vecs * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
    return rotated


def certificate_guarantees_survival(candidate_count, target_rank, fixed_bits, max_k):
    states = {(0, False, 0)}
    for rank in range(candidate_count):
        next_states = set()
        choices = [fixed_bits[rank]] if rank in fixed_bits else [False, True]
        for selected_count, target_selected, rejected_before in states:
            if selected_count >= max_k:
                next_states.add((selected_count, target_selected, rejected_before))
                continue
            for dominated in choices:
                if dominated:
                    next_states.add(
                        (
                            selected_count,
                            target_selected,
                            rejected_before + int(rank < target_rank),
                        )
                    )
                else:
                    next_states.add(
                        (
                            selected_count + 1,
                            target_selected or rank == target_rank,
                            rejected_before,
                        )
                    )
        states = next_states
    for selected_count, target_selected, rejected_before in states:
        if target_selected:
            continue
        if selected_count >= max_k:
            return False
        quota = max_k - selected_count
        if rejected_before >= quota:
            return False
    return True


def shortest_prefix_quota_certificate(bits, target_rank, max_k):
    full = {index: bit for index, bit in enumerate(bits)}
    if not certificate_guarantees_survival(len(bits), target_rank, full, max_k):
        raise AssertionError("完整 exact atom assignment 未保证 exact edge 存活")
    left = 0
    right = len(bits)
    while left < right:
        middle = (left + right) // 2
        fixed = {index: bits[index] for index in range(middle)}
        if certificate_guarantees_survival(len(bits), target_rank, fixed, max_k):
            right = middle
        else:
            left = middle + 1
    return tuple(range(left))


def exact_parameter_sensitivity(vecs, targets, candidates_by_target):
    original_alpha = pruning.ALPHA
    original_max_k = pruning.MAX_K
    rows = []
    for alpha, max_k in SENSITIVITY:
        pruning.ALPHA = alpha
        pruning.MAX_K = max_k
        runs = [
            pruning.run_select(
                int(target),
                candidates,
                *pruning.distance_cache(vecs, int(target), candidates),
                saturate=True,
            )
            for target, candidates in zip(targets, candidates_by_target)
        ]
        diversity = float(np.mean([len(run["diversity_selected"]) for run in runs]))
        fallback = float(np.mean([len(run["fallback_added"]) for run in runs]))
        rows.append(
            {
                "alpha": alpha,
                "max_k": max_k,
                "mean_diversity_selected": diversity,
                "mean_fallback_added": fallback,
                "fallback_share_of_output": fallback / max_k,
            }
        )
    pruning.ALPHA = original_alpha
    pruning.MAX_K = original_max_k
    return rows


def evaluate_target(vecs, target, candidates, encoded, model):
    slope = model["distance_calibration"]["slope"]
    exact_nodes, exact_distances = pruning.distance_cache(vecs, target, candidates)
    _, approx_distances = pruning.distance_cache(
        vecs, target, candidates, encoded=encoded, slope=slope
    )
    exact_run = pruning.run_select(
        target, candidates, exact_nodes, exact_distances, saturate=True
    )
    approx_fixed = pruning.run_select(
        target,
        candidates,
        exact_nodes,
        approx_distances,
        ordered_override=exact_run["ordered"],
        saturate=True,
    )
    positions = {int(node): index for index, node in enumerate(exact_nodes)}
    event_rows = [
        edge_probe.candidate_event(
            exact_distances,
            approx_distances,
            positions,
            target,
            decision,
            model,
        )
        for decision in exact_run["decisions"]
    ]
    exact_bits = [bool(decision["dominated"]) for decision in exact_run["decisions"]]
    approx_bits = [bool(event["approx_dominated"]) for event in event_rows]
    fixed_final = set(approx_fixed["final_selected"])
    exact_fallback = set(exact_run["fallback_added"])
    rows = []
    for candidate in exact_run["fallback_added"]:
        target_rank = exact_run["ordered"].index(candidate)
        certificate = shortest_prefix_quota_certificate(exact_bits, target_rank, MAX_K)
        certificate_good = all(approx_bits[index] == exact_bits[index] for index in certificate)
        survives = candidate in fixed_final
        if certificate_good and not survives:
            raise AssertionError("fallback quota/rank 1-certificate 充分性被违反")
        risk_sum = float(sum(event_rows[index]["risk"] for index in certificate))
        rows.append(
            {
                "edge_type": "fallback",
                "exact_fallback": candidate in exact_fallback,
                "target_rank": target_rank,
                "certificate_event_count": len(certificate),
                "certificate_good": certificate_good,
                "fixed_order_survives": survives,
                "risk_sum": risk_sum,
                "risk_clipped": min(1.0, risk_sum),
                "tie_count": sum(event_rows[index]["tie"] for index in certificate),
            }
        )
    return rows


def summarize(rows):
    failure = np.array([not row["fixed_order_survives"] for row in rows], dtype=np.float64)
    risk = np.array([row["risk_sum"] for row in rows], dtype=np.float64)
    counts = np.array([row["certificate_event_count"] for row in rows], dtype=np.float64)
    return {
        "edge_count": len(rows),
        "mean_certificate_event_count": float(counts.mean()),
        "median_certificate_event_count": float(np.median(counts)),
        "p90_certificate_event_count": float(np.quantile(counts, 0.90)),
        "fixed_order_failure_rate": float(failure.mean()),
        "certificate_pass_rate": float(np.mean([row["certificate_good"] for row in rows])),
        "certificate_saturation_rate": float(np.mean([row["risk_clipped"] >= 1.0 for row in rows])),
        "mean_risk_sum": float(risk.mean()),
        "risk_failure_spearman": safe_spearman(risk, failure),
        "event_count_failure_spearman": safe_spearman(counts, failure),
        "tie_event_mean": float(np.mean([row["tie_count"] for row in rows])),
        "certificate_sufficiency_holds": bool(
            all(not row["certificate_good"] or row["fixed_order_survives"] for row in rows)
        ),
    }


def run_dataset(name):
    vecs = load_sampled_dataset(name)
    rotated = randomized_orthogonal_transform(vecs)
    encodings = {"2bit": pruning.encode_bq2(vecs), "rotated_2bit": pruning.encode_bq2(rotated)}
    rng = np.random.default_rng(SEED)
    targets = rng.choice(
        len(vecs), N_CALIBRATION_TARGETS + N_EVALUATION_TARGETS, replace=False
    )
    calibration_targets = targets[:N_CALIBRATION_TARGETS]
    evaluation_targets = targets[N_CALIBRATION_TARGETS:]
    calibration_candidates = pruning.candidate_lists(vecs, calibration_targets)
    evaluation_candidates = pruning.candidate_lists(vecs, evaluation_targets)
    result = {
        "dataset": name,
        "dimension": int(vecs.shape[1]),
        "parameter_sensitivity": exact_parameter_sensitivity(
            vecs, calibration_targets, calibration_candidates
        ),
        "quantizers": {},
    }
    for quantizer in QUANTIZERS:
        encoded = encodings[quantizer]
        model = trace.fit_event_models(
            vecs, calibration_targets, calibration_candidates, encoded
        )
        rows = [
            edge
            for target, candidates in zip(evaluation_targets, evaluation_candidates)
            for edge in evaluate_target(vecs, int(target), candidates, encoded, model)
        ]
        result["quantizers"][quantizer] = {
            "calibration": model,
            "evaluation": summarize(rows),
        }
    return result


def save_results(results):
    output = {
        "probe": 30,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "n_sample": N_SAMPLE,
        "n_calibration_targets": N_CALIBRATION_TARGETS,
        "n_evaluation_targets": N_EVALUATION_TARGETS,
        "candidate_k": CANDIDATE_K,
        "max_k": MAX_K,
        "alpha": ALPHA,
        "variant": "标准RobustPrune后执行确定性饱和填充；填充不是原始RobustPrune步骤",
        "results": results,
    }
    path = RESULTS_DIR / "probe30_standard_vamana_saturated_variant.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        default = next(
            row
            for row in result["parameter_sensitivity"]
            if row["alpha"] == ALPHA and row["max_k"] == MAX_K
        )
        print(
            f"  exact 默认参数：diversity={default['mean_diversity_selected']:.2f}, "
            f"fallback={default['mean_fallback_added']:.2f}"
        )
        for quantizer, metrics in result["quantizers"].items():
            evaluation = metrics["evaluation"]
            print(
                f"  {quantizer:<13} edges={evaluation['edge_count']}, "
                f"slice={evaluation['mean_certificate_event_count']:.1f}, "
                f"sat={evaluation['certificate_saturation_rate']:.3f}, "
                f"fail={evaluation['fixed_order_failure_rate']:.4f}, "
                f"rho={evaluation['risk_failure_spearman']:.3f}"
            )
    save_results(results)


if __name__ == "__main__":
    main()
