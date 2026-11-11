#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10 - Probe 27：Vamana α-pruning 固定轨迹与自由重放实验

检验 P3 structural margin、pruning residual covariance、局部决策翻转、
候选顺序分叉、fallback fill 与最终邻接集合差异。
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.special import erfc
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_SAMPLE = 4_000
N_CALIBRATION_TARGETS = 80
N_EVALUATION_TARGETS = 160
CANDIDATE_K = 64
MAX_K = 32
ALPHA = 1.2
SEED = 20260823
TAIL_THRESHOLDS = np.array([1.0, 1.5, 2.0, 2.5, 3.0])


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_sampled_dataset(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 27 只支持 f32 数据集，收到 {data_format}")
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


def encode_bq2(vecs):
    signs = vecs > 0
    strongs = np.abs(vecs) > np.abs(vecs).mean(axis=1, keepdims=True)
    return signs, strongs


def bq2_distances(left_indices, right_indices, encoded):
    signs, strongs = encoded
    left_indices = np.asarray(left_indices, dtype=np.int64)
    right_indices = np.asarray(right_indices, dtype=np.int64)
    agreements = signs[left_indices] == signs[right_indices]
    signed = 2 * agreements.astype(np.int16) - 1
    weights = (1 + strongs[left_indices].astype(np.int16)) * (
        1 + strongs[right_indices].astype(np.int16)
    )
    dot = (signed * weights).sum(axis=1, dtype=np.int64)
    return (4 * signs.shape[1] - dot).astype(np.float64)


def exact_distance(vecs, left, right):
    return float(1.0 - np.dot(vecs[left], vecs[right]))


def candidate_lists(vecs, target_indices):
    similarities = vecs[target_indices] @ vecs.T
    output = []
    for row, target in enumerate(target_indices):
        similarities[row, target] = -np.inf
        candidates = np.argpartition(-similarities[row], CANDIDATE_K)[:CANDIDATE_K]
        order = np.lexsort((candidates, -similarities[row, candidates]))
        output.append(candidates[order].astype(np.int64))
    return output


def distance_cache(vecs, target, candidates, encoded=None, slope=1.0):
    nodes = np.concatenate([[target], candidates])
    if encoded is None:
        matrix = 1.0 - vecs[nodes] @ vecs[nodes].T
        return nodes, np.maximum(matrix.astype(np.float64), 0.0)
    left = np.repeat(nodes, len(nodes))
    right = np.tile(nodes, len(nodes))
    raw = bq2_distances(left, right, encoded).reshape(len(nodes), len(nodes))
    return nodes, slope * raw


def pruning_psi(distance_to_target, distance_to_selected, alpha=None):
    """标准 RobustPrune 的 signed functional；非负表示候选被支配。"""
    alpha = ALPHA if alpha is None else alpha
    return distance_to_target - alpha * distance_to_selected


def run_select(
    target,
    candidates,
    nodes,
    distances,
    ordered_override=None,
    saturate=False,
):
    positions = {int(node): index for index, node in enumerate(nodes)}
    target_pos = positions[int(target)]
    if ordered_override is None:
        ordered = sorted(
            (int(candidate) for candidate in candidates if candidate != target),
            key=lambda candidate: (distances[target_pos, positions[candidate]], candidate),
        )
    else:
        ordered = [int(candidate) for candidate in ordered_override]
    selected = []
    rejected = []
    comparisons = []
    decisions = []
    stopped_at_cap = False
    for candidate_rank, candidate in enumerate(ordered):
        if len(selected) >= MAX_K:
            stopped_at_cap = True
            break
        candidate_pos = positions[candidate]
        distance_to_target = float(distances[target_pos, candidate_pos])
        dominated = False
        witness = None
        prefix = tuple(selected)
        for selected_rank, selected_node in enumerate(selected):
            distance_to_selected = float(
                distances[candidate_pos, positions[selected_node]]
            )
            psi = pruning_psi(distance_to_target, distance_to_selected)
            comparison = {
                "candidate": candidate,
                "selected": selected_node,
                "candidate_rank": candidate_rank,
                "selected_rank": selected_rank,
                "distance_to_target": distance_to_target,
                "distance_to_selected": distance_to_selected,
                "psi": psi,
                "dominated": bool(psi >= 0.0),
                "selected_prefix": prefix,
            }
            comparisons.append(comparison)
            if psi >= 0.0:
                dominated = True
                witness = selected_node
                break
        if dominated:
            rejected.append((candidate, witness))
        else:
            selected.append(candidate)
        decisions.append(
            {
                "candidate": candidate,
                "candidate_rank": candidate_rank,
                "selected_prefix": prefix,
                "dominated": dominated,
                "witness": witness,
            }
        )
    diversity_selected = tuple(selected)
    fallback_added = []
    if saturate and len(selected) < MAX_K:
        for candidate in ordered:
            if len(selected) >= MAX_K:
                break
            if candidate not in selected:
                selected.append(candidate)
                fallback_added.append(candidate)
    return {
        "ordered": tuple(ordered),
        "diversity_selected": diversity_selected,
        "final_selected": tuple(selected),
        "rejected": tuple(rejected),
        "fallback_added": tuple(fallback_added),
        "comparisons": comparisons,
        "decisions": decisions,
        "stopped_at_cap": stopped_at_cap,
    }


def fit_distance_calibration(vecs, target_indices, candidates_by_target, encoded):
    exact_values = []
    raw_values = []
    for target, candidates in zip(target_indices, candidates_by_target):
        left = np.full(len(candidates), target, dtype=np.int64)
        exact_values.extend(1.0 - vecs[candidates] @ vecs[target])
        raw_values.extend(bq2_distances(left, candidates, encoded))
        pair_count = min(256, len(candidates) * (len(candidates) - 1) // 2)
        rng = np.random.default_rng(SEED + int(target))
        first = rng.choice(candidates, pair_count, replace=True)
        second = rng.choice(candidates, pair_count, replace=True)
        mask = first != second
        first = first[mask]
        second = second[mask]
        exact_values.extend(1.0 - np.sum(vecs[first] * vecs[second], axis=1))
        raw_values.extend(bq2_distances(first, second, encoded))
    exact_values = np.asarray(exact_values, dtype=np.float64)
    raw_values = np.asarray(raw_values, dtype=np.float64)
    denominator = float(np.dot(raw_values, raw_values))
    slope = float(np.dot(raw_values, exact_values) / denominator) if denominator else 0.0
    residual = slope * raw_values - exact_values
    return {
        "slope": slope,
        "distance_count": int(len(exact_values)),
        "residual_mean": float(np.mean(residual)),
        "residual_scale": float(np.std(residual, ddof=1)),
        "distance_spearman": safe_spearman(raw_values, exact_values),
    }


def evaluate_target(vecs, target, candidates, encoded, slope):
    exact_nodes, exact_distances = distance_cache(vecs, target, candidates)
    approx_nodes, approx_distances = distance_cache(
        vecs, target, candidates, encoded=encoded, slope=slope
    )
    assert np.array_equal(exact_nodes, approx_nodes)
    exact_run = run_select(target, candidates, exact_nodes, exact_distances)
    approx_fixed_order_run = run_select(
        target,
        candidates,
        approx_nodes,
        approx_distances,
        ordered_override=exact_run["ordered"],
    )
    approx_run = run_select(target, candidates, approx_nodes, approx_distances)
    positions = {int(node): index for index, node in enumerate(exact_nodes)}

    local = []
    candidate_groups = {}
    for comparison in exact_run["comparisons"]:
        candidate = comparison["candidate"]
        selected = comparison["selected"]
        target_pos = positions[int(target)]
        candidate_pos = positions[candidate]
        selected_pos = positions[selected]
        exact_tc = float(exact_distances[target_pos, candidate_pos])
        exact_cs = float(exact_distances[candidate_pos, selected_pos])
        approx_tc = float(approx_distances[target_pos, candidate_pos])
        approx_cs = float(approx_distances[candidate_pos, selected_pos])
        xi_tc = approx_tc - exact_tc
        xi_cs = approx_cs - exact_cs
        psi = pruning_psi(exact_tc, exact_cs)
        approx_psi = pruning_psi(approx_tc, approx_cs)
        exact_dominated = psi >= 0.0
        approx_dominated = approx_psi >= 0.0
        row = {
            "psi": psi,
            "approx_psi": approx_psi,
            "z": approx_psi - psi,
            "xi_tc": xi_tc,
            "xi_cs": xi_cs,
            "exact_dominated": exact_dominated,
            "approx_dominated": approx_dominated,
            "false_keep": bool(exact_dominated and not approx_dominated),
            "false_prune": bool(not exact_dominated and approx_dominated),
            "exact_tie": bool(psi == 0.0),
        }
        local.append(row)
        group = candidate_groups.setdefault(
            candidate, {"exact": False, "approx": False, "comparisons": 0}
        )
        group["exact"] = group["exact"] or exact_dominated
        group["approx"] = group["approx"] or approx_dominated
        group["comparisons"] += 1

    candidate_mismatches = sum(
        group["exact"] != group["approx"] for group in candidate_groups.values()
    )
    exact_final = set(exact_run["final_selected"])
    fixed_order_final = set(approx_fixed_order_run["final_selected"])
    approx_final = set(approx_run["final_selected"])
    exact_diversity = set(exact_run["diversity_selected"])
    fixed_order_diversity = set(approx_fixed_order_run["diversity_selected"])
    approx_diversity = set(approx_run["diversity_selected"])
    return {
        "local": local,
        "candidate_count": len(candidate_groups),
        "candidate_mismatches": candidate_mismatches,
        "fixed_order_diversity_output_changed": exact_diversity != fixed_order_diversity,
        "fixed_order_final_output_changed": exact_final != fixed_order_final,
        "fixed_order_final_jaccard": len(exact_final & fixed_order_final)
        / max(len(exact_final | fixed_order_final), 1),
        "fixed_order_fallback_count": len(approx_fixed_order_run["fallback_added"]),
        "fixed_order_reintroduced_count": sum(
            candidate in approx_fixed_order_run["fallback_added"]
            for candidate, _ in approx_fixed_order_run["rejected"]
        ),
        "fixed_order_stopped_at_cap": approx_fixed_order_run["stopped_at_cap"],
        "order_changed": exact_run["ordered"] != approx_run["ordered"],
        "diversity_output_changed": exact_diversity != approx_diversity,
        "final_output_changed": exact_final != approx_final,
        "final_jaccard": len(exact_final & approx_final) / max(len(exact_final | approx_final), 1),
        "exact_fallback_count": len(exact_run["fallback_added"]),
        "approx_fallback_count": len(approx_run["fallback_added"]),
        "exact_reintroduced_count": sum(
            candidate in exact_run["fallback_added"] for candidate, _ in exact_run["rejected"]
        ),
        "approx_reintroduced_count": sum(
            candidate in approx_run["fallback_added"] for candidate, _ in approx_run["rejected"]
        ),
        "exact_stopped_at_cap": exact_run["stopped_at_cap"],
        "approx_stopped_at_cap": approx_run["stopped_at_cap"],
    }


def tail_metrics(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    scale = max(float(np.std(values, ddof=1)), 1e-12)
    standardized = (values - mean) / scale
    empirical = np.array(
        [(np.abs(standardized) >= threshold).mean() for threshold in TAIL_THRESHOLDS]
    )
    gaussian = erfc(TAIL_THRESHOLDS / np.sqrt(2.0))
    return {
        "mean": mean,
        "scale": scale,
        "kurtosis": float(np.mean(standardized**4)),
        "thresholds": TAIL_THRESHOLDS.tolist(),
        "empirical_two_sided_survival": empirical.tolist(),
        "gaussian_two_sided_survival": gaussian.tolist(),
        "empirical_gaussian_ratio": (empirical / np.maximum(gaussian, 1e-12)).tolist(),
    }


def summarize(target_results):
    local = [row for target in target_results for row in target["local"]]
    psi = np.array([row["psi"] for row in local], dtype=np.float64)
    z = np.array([row["z"] for row in local], dtype=np.float64)
    xi_tc = np.array([row["xi_tc"] for row in local], dtype=np.float64)
    xi_cs = np.array([row["xi_cs"] for row in local], dtype=np.float64)
    false_keep = np.array([row["false_keep"] for row in local], dtype=bool)
    false_prune = np.array([row["false_prune"] for row in local], dtype=bool)
    exact_dominated = np.array([row["exact_dominated"] for row in local], dtype=bool)
    approx_dominated = np.array([row["approx_dominated"] for row in local], dtype=bool)
    covariance = np.cov(xi_tc, xi_cs, ddof=1)
    empirical_z_variance = float(np.var(z, ddof=1))
    covariance_z_variance = float(
        ALPHA**2 * covariance[0, 0]
        + covariance[1, 1]
        - 2.0 * ALPHA * covariance[0, 1]
    )
    return {
        "comparison_count": len(local),
        "false_keep_rate": float(false_keep.mean()),
        "false_prune_rate": float(false_prune.mean()),
        "local_disagreement_rate": float((false_keep | false_prune).mean()),
        "exact_dominated_rate": float(exact_dominated.mean()),
        "approx_dominated_on_exact_trace_rate": float(approx_dominated.mean()),
        "exact_tie_rate": float(np.mean(psi == 0.0)),
        "structural_margin_abs_median": float(np.median(np.abs(psi))),
        "standardized_margin_flip_spearman": safe_spearman(
            np.abs(psi) / max(np.std(z, ddof=1), 1e-12),
            (false_keep | false_prune).astype(np.float64),
        ),
        "residual": {
            "xi_tc_variance": float(covariance[0, 0]),
            "xi_cs_variance": float(covariance[1, 1]),
            "cross_covariance": float(covariance[0, 1]),
            "correlation": float(
                covariance[0, 1]
                / max(np.sqrt(covariance[0, 0] * covariance[1, 1]), 1e-12)
            ),
            "empirical_z_variance": empirical_z_variance,
            "covariance_formula_variance": covariance_z_variance,
            "independence_variance": float(
                covariance[0, 0] + ALPHA**2 * covariance[1, 1]
            ),
            "tail": tail_metrics(z),
        },
        "candidate_level": {
            "candidate_count": int(sum(row["candidate_count"] for row in target_results)),
            "mismatch_count": int(sum(row["candidate_mismatches"] for row in target_results)),
            "mismatch_rate": float(
                sum(row["candidate_mismatches"] for row in target_results)
                / max(sum(row["candidate_count"] for row in target_results), 1)
            ),
        },
        "fixed_order_replay": {
            "target_count": len(target_results),
            "diversity_output_divergence_rate": float(
                np.mean(
                    [row["fixed_order_diversity_output_changed"] for row in target_results]
                )
            ),
            "final_output_divergence_rate": float(
                np.mean([row["fixed_order_final_output_changed"] for row in target_results])
            ),
            "mean_final_jaccard": float(
                np.mean([row["fixed_order_final_jaccard"] for row in target_results])
            ),
            "mean_fallback_count": float(
                np.mean([row["fixed_order_fallback_count"] for row in target_results])
            ),
            "reintroduced_count": int(
                sum(row["fixed_order_reintroduced_count"] for row in target_results)
            ),
            "degree_cap_rate": float(
                np.mean([row["fixed_order_stopped_at_cap"] for row in target_results])
            ),
        },
        "free_replay": {
            "target_count": len(target_results),
            "order_divergence_rate": float(np.mean([row["order_changed"] for row in target_results])),
            "diversity_output_divergence_rate": float(
                np.mean([row["diversity_output_changed"] for row in target_results])
            ),
            "final_output_divergence_rate": float(
                np.mean([row["final_output_changed"] for row in target_results])
            ),
            "mean_final_jaccard": float(np.mean([row["final_jaccard"] for row in target_results])),
            "mean_exact_fallback_count": float(
                np.mean([row["exact_fallback_count"] for row in target_results])
            ),
            "mean_approx_fallback_count": float(
                np.mean([row["approx_fallback_count"] for row in target_results])
            ),
            "exact_reintroduced_count": int(
                sum(row["exact_reintroduced_count"] for row in target_results)
            ),
            "approx_reintroduced_count": int(
                sum(row["approx_reintroduced_count"] for row in target_results)
            ),
            "exact_degree_cap_rate": float(
                np.mean([row["exact_stopped_at_cap"] for row in target_results])
            ),
            "approx_degree_cap_rate": float(
                np.mean([row["approx_stopped_at_cap"] for row in target_results])
            ),
        },
    }


def run_dataset(name):
    vecs = load_sampled_dataset(name)
    rotated = randomized_orthogonal_transform(vecs)
    encodings = {"2bit": encode_bq2(vecs), "rotated_2bit": encode_bq2(rotated)}
    rng = np.random.default_rng(SEED)
    targets = rng.choice(
        len(vecs), N_CALIBRATION_TARGETS + N_EVALUATION_TARGETS, replace=False
    )
    calibration_targets = targets[:N_CALIBRATION_TARGETS]
    evaluation_targets = targets[N_CALIBRATION_TARGETS:]
    calibration_candidates = candidate_lists(vecs, calibration_targets)
    evaluation_candidates = candidate_lists(vecs, evaluation_targets)
    output = {"dataset": name, "dimension": int(vecs.shape[1]), "quantizers": {}}
    for quantizer in QUANTIZERS:
        encoded = encodings[quantizer]
        calibration = fit_distance_calibration(
            vecs, calibration_targets, calibration_candidates, encoded
        )
        target_results = [
            evaluate_target(vecs, int(target), candidates, encoded, calibration["slope"])
            for target, candidates in zip(evaluation_targets, evaluation_candidates)
        ]
        output["quantizers"][quantizer] = {
            "distance_calibration": calibration,
            "evaluation": summarize(target_results),
        }
    return output


def save_results(results):
    output = {
        "probe": 27,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "n_sample": N_SAMPLE,
        "n_calibration_targets": N_CALIBRATION_TARGETS,
        "n_evaluation_targets": N_EVALUATION_TARGETS,
        "candidate_k": CANDIDATE_K,
        "max_k": MAX_K,
        "alpha": ALPHA,
        "saturate": False,
        "results": results,
    }
    path = RESULTS_DIR / "probe27_standard_vamana_pruning_replay.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            evaluation = metrics["evaluation"]
            fixed = evaluation["fixed_order_replay"]
            free = evaluation["free_replay"]
            print(
                f"  {quantizer:<13} local={evaluation['local_disagreement_rate']:.4f}, "
                f"FK={evaluation['false_keep_rate']:.4f}, "
                f"FP={evaluation['false_prune_rate']:.4f}, "
                f"candidate={evaluation['candidate_level']['mismatch_rate']:.4f}, "
                f"fixed={fixed['final_output_divergence_rate']:.3f}, "
                f"order={free['order_divergence_rate']:.3f}, "
                f"final={free['final_output_divergence_rate']:.3f}, "
                f"J={free['mean_final_jaccard']:.3f}"
            )
    save_results(results)


if __name__ == "__main__":
    main()
