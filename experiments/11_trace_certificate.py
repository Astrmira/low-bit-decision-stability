#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11 - Probe 28：P4 固定语义轨迹证书实验

记录 adjacent sorting、counterfactual exact-prefix pruning、first semantic divergence，
并区分 theorem-form plug-in bound 与 top-r heuristic predictor。
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.special import logsumexp
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT
from importlib import import_module


pruning = import_module("10_pruning_replay")

DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_SAMPLE = 4_000
N_CALIBRATION_TARGETS = 80
N_EVALUATION_TARGETS = 160
CANDIDATE_K = 64
MAX_K = 32
ALPHA = 1.2
SEED = 20260823
TOP_R = [1, 3, 5, 10]


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_sampled_dataset(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 28 只支持 f32 数据集，收到 {data_format}")
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


def fit_event_models(vecs, targets, candidates_by_target, encoded):
    calibration = pruning.fit_distance_calibration(
        vecs, targets, candidates_by_target, encoded
    )
    slope = calibration["slope"]
    sorting_residuals = []
    pruning_residuals = []
    for target, candidates in zip(targets, candidates_by_target):
        exact_nodes, exact_distances = pruning.distance_cache(vecs, int(target), candidates)
        _, approx_distances = pruning.distance_cache(
            vecs, int(target), candidates, encoded=encoded, slope=slope
        )
        exact_run = pruning.run_select(int(target), candidates, exact_nodes, exact_distances)
        positions = {int(node): index for index, node in enumerate(exact_nodes)}
        target_pos = positions[int(target)]
        ordered = exact_run["ordered"]
        for left, right in zip(ordered[:-1], ordered[1:]):
            exact_margin = (
                exact_distances[target_pos, positions[right]]
                - exact_distances[target_pos, positions[left]]
            )
            approx_margin = (
                approx_distances[target_pos, positions[right]]
                - approx_distances[target_pos, positions[left]]
            )
            sorting_residuals.append(float(approx_margin - exact_margin))
        for row in exact_run["comparisons"]:
            candidate = row["candidate"]
            selected = row["selected"]
            exact_psi = row["psi"]
            approx_psi = pruning.pruning_psi(
                approx_distances[target_pos, positions[candidate]],
                approx_distances[positions[candidate], positions[selected]],
                ALPHA,
            )
            pruning_residuals.append(float(approx_psi - exact_psi))
    return {
        "distance_calibration": calibration,
        "sorting_mean": float(np.mean(sorting_residuals)),
        "sorting_scale": max(float(np.std(sorting_residuals, ddof=1)), 1e-12),
        "pruning_mean": float(np.mean(pruning_residuals)),
        "pruning_scale": max(float(np.std(pruning_residuals, ddof=1)), 1e-12),
        "sorting_count": len(sorting_residuals),
        "pruning_count": len(pruning_residuals),
    }


def event_risk(margin, mean, scale, orientation):
    if margin <= 0:
        return 1.0
    threshold = margin + mean if orientation == "left" else margin - mean
    if threshold <= 0:
        return 1.0
    return float(np.exp(-0.5 * (threshold / scale) ** 2))


def trace_events(vecs, target, candidates, encoded, model):
    slope = model["distance_calibration"]["slope"]
    exact_nodes, exact_distances = pruning.distance_cache(vecs, target, candidates)
    _, approx_distances = pruning.distance_cache(
        vecs, target, candidates, encoded=encoded, slope=slope
    )
    exact_run = pruning.run_select(target, candidates, exact_nodes, exact_distances)
    approx_run = pruning.run_select(target, candidates, exact_nodes, approx_distances)
    positions = {int(node): index for index, node in enumerate(exact_nodes)}
    approx_order_positions = {
        candidate: index for index, candidate in enumerate(approx_run["ordered"])
    }
    target_pos = positions[target]
    events = []

    for index, (left, right) in enumerate(
        zip(exact_run["ordered"][:-1], exact_run["ordered"][1:])
    ):
        exact_margin = float(
            exact_distances[target_pos, positions[right]]
            - exact_distances[target_pos, positions[left]]
        )
        approx_margin = float(
            approx_distances[target_pos, positions[right]]
            - approx_distances[target_pos, positions[left]]
        )
        events.append(
            {
                "stage": "sorting",
                "index": index,
                "margin": exact_margin,
                "approx_margin": approx_margin,
                "error": bool(
                    approx_order_positions[right] < approx_order_positions[left]
                ),
                "tie": bool(exact_margin == 0.0),
                "risk": 1.0
                if exact_margin == 0.0
                else event_risk(
                    exact_margin,
                    model["sorting_mean"],
                    model["sorting_scale"],
                    "left",
                ),
            }
        )

    exact_decisions = {row["candidate"]: row for row in exact_run["decisions"]}
    for index, candidate in enumerate(
        candidate for candidate in exact_run["ordered"] if candidate in exact_decisions
    ):
        decision = exact_decisions[candidate]
        prefix = decision["selected_prefix"]
        local = []
        for selected in prefix:
            exact_psi = float(
                pruning.pruning_psi(
                    exact_distances[target_pos, positions[candidate]],
                    exact_distances[positions[candidate], positions[selected]],
                    ALPHA,
                )
            )
            approx_psi = float(
                pruning.pruning_psi(
                    approx_distances[target_pos, positions[candidate]],
                    approx_distances[positions[candidate], positions[selected]],
                    ALPHA,
                )
            )
            local.append(
                {
                    "selected": selected,
                    "psi": exact_psi,
                    "approx_psi": approx_psi,
                }
            )
        approximate_bits = []
        local_risks = []
        tie = False
        for row in local:
            approx_psi = row["approx_psi"]
            approximate_bits.append(approx_psi >= 0.0)
            tie = tie or row["psi"] == 0.0
            orientation = "left" if row["psi"] >= 0.0 else "right"
            local_risks.append(
                event_risk(
                    abs(float(row["psi"])),
                    model["pruning_mean"],
                    model["pruning_scale"],
                    orientation,
                )
            )
        approx_dominated = any(approximate_bits)
        exact_dominated = bool(decision["dominated"])
        if not local:
            risk = 0.0
        elif exact_dominated:
            witness_risks = [
                local_risks[position]
                for position, row in enumerate(local)
                if row["psi"] >= 0.0
            ]
            risk = min(witness_risks) if witness_risks else 1.0
        else:
            risk = min(1.0, float(sum(local_risks)))
        events.append(
            {
                "stage": "pruning",
                "index": index,
                "candidate": candidate,
                "prefix_length": len(prefix),
                "margin": min((abs(float(row["psi"])) for row in local), default=None),
                "error": bool(exact_dominated != approx_dominated),
                "tie": tie,
                "risk": risk,
            }
        )

    sorting_errors = [event for event in events if event["stage"] == "sorting" and event["error"]]
    pruning_errors = [event for event in events if event["stage"] == "pruning" and event["error"]]
    event_error = bool(sorting_errors or pruning_errors)
    order_divergence = exact_run["ordered"] != approx_run["ordered"]
    semantic_divergence = bool(
        exact_run["ordered"] != approx_run["ordered"]
        or exact_run["diversity_selected"] != approx_run["diversity_selected"]
        or exact_run["fallback_added"] != approx_run["fallback_added"]
        or exact_run["final_selected"] != approx_run["final_selected"]
        or exact_run["stopped_at_cap"] != approx_run["stopped_at_cap"]
    )
    output_divergence = set(exact_run["final_selected"]) != set(approx_run["final_selected"])
    if sorting_errors:
        first_type = "sorting"
        first_index = min(event["index"] for event in sorting_errors)
    elif pruning_errors:
        first_type = "pruning"
        first_index = min(event["index"] for event in pruning_errors)
    else:
        first_type = None
        first_index = None

    risks = np.array([event["risk"] for event in events], dtype=np.float64)
    strict_sum = float(np.sum(risks))
    sorted_risks = np.sort(risks)[::-1]
    heuristic = {
        f"top_{count}_logrisk": float(
            logsumexp(np.log(np.maximum(sorted_risks[:count], 1e-300)))
        )
        for count in TOP_R
    }
    heuristic["all_logrisk"] = float(
        logsumexp(np.log(np.maximum(sorted_risks, 1e-300)))
    )
    return {
        "events": events,
        "sorting_event_count": sum(event["stage"] == "sorting" for event in events),
        "pruning_event_count": sum(event["stage"] == "pruning" for event in events),
        "sorting_error_count": len(sorting_errors),
        "pruning_error_count": len(pruning_errors),
        "tie_event_count": sum(event["tie"] for event in events),
        "event_error": event_error,
        "order_divergence": order_divergence,
        "semantic_divergence": semantic_divergence,
        "output_divergence": output_divergence,
        "first_divergence_type": first_type,
        "first_divergence_index": first_index,
        "strict_form_sum": strict_sum,
        "strict_form_clipped": min(1.0, strict_sum),
        "heuristic": heuristic,
        "exact_fallback_count": len(exact_run["fallback_added"]),
        "approx_fallback_count": len(approx_run["fallback_added"]),
        "exact_degree_cap": exact_run["stopped_at_cap"],
        "approx_degree_cap": approx_run["stopped_at_cap"],
    }


def summarize(rows):
    failures = np.array([row["semantic_divergence"] for row in rows], dtype=np.float64)
    outputs = np.array([row["output_divergence"] for row in rows], dtype=np.float64)
    strict = np.array([row["strict_form_clipped"] for row in rows], dtype=np.float64)
    strict_sum = np.array([row["strict_form_sum"] for row in rows], dtype=np.float64)
    heuristic_keys = list(rows[0]["heuristic"])
    heuristic_correlations = {
        key: safe_spearman([row["heuristic"][key] for row in rows], failures)
        for key in heuristic_keys
    }
    first_types = {kind: sum(row["first_divergence_type"] == kind for row in rows) for kind in ["sorting", "pruning", None]}
    return {
        "target_count": len(rows),
        "semantic_divergence_rate": float(failures.mean()),
        "output_divergence_rate": float(outputs.mean()),
        "order_divergence_rate": float(np.mean([row["order_divergence"] for row in rows])),
        "sorting_event_mean": float(np.mean([row["sorting_event_count"] for row in rows])),
        "pruning_event_mean": float(np.mean([row["pruning_event_count"] for row in rows])),
        "sorting_error_mean": float(np.mean([row["sorting_error_count"] for row in rows])),
        "pruning_error_mean": float(np.mean([row["pruning_error_count"] for row in rows])),
        "tie_event_mean": float(np.mean([row["tie_event_count"] for row in rows])),
        "first_divergence_counts": {str(key): value for key, value in first_types.items()},
        "strict_form": {
            "mean_sum": float(strict_sum.mean()),
            "mean_clipped": float(strict.mean()),
            "saturation_rate": float(np.mean(strict >= 1.0 - 1e-12)),
            "covers_semantic_empirically": bool(strict.mean() + 1e-12 >= failures.mean()),
            "status": "使用样本 residual scale 的 theorem-form plug-in diagnostic，不是已证明的条件 MGF certificate",
        },
        "heuristic_correlations": heuristic_correlations,
        "best_heuristic": max(heuristic_correlations, key=heuristic_correlations.get),
        "best_heuristic_spearman": max(heuristic_correlations.values()),
        "event_identity_holds": bool(
            all(row["semantic_divergence"] == row["event_error"] for row in rows)
        ),
        "output_subset_holds": bool(
            all(not row["output_divergence"] or row["semantic_divergence"] for row in rows)
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
    result = {"dataset": name, "dimension": int(vecs.shape[1]), "quantizers": {}}
    for quantizer in QUANTIZERS:
        encoded = encodings[quantizer]
        model = fit_event_models(
            vecs, calibration_targets, calibration_candidates, encoded
        )
        rows = [
            trace_events(vecs, int(target), candidates, encoded, model)
            for target, candidates in zip(evaluation_targets, evaluation_candidates)
        ]
        result["quantizers"][quantizer] = {
            "calibration": model,
            "evaluation": summarize(rows),
        }
    return result


def save_results(results):
    output = {
        "probe": 28,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "n_sample": N_SAMPLE,
        "n_calibration_targets": N_CALIBRATION_TARGETS,
        "n_evaluation_targets": N_EVALUATION_TARGETS,
        "candidate_k": CANDIDATE_K,
        "max_k": MAX_K,
        "alpha": ALPHA,
        "results": results,
    }
    path = RESULTS_DIR / "probe28_standard_vamana_trace_certificate.json"
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
            print(
                f"  {quantizer:<13} semantic={evaluation['semantic_divergence_rate']:.3f}, "
                f"output={evaluation['output_divergence_rate']:.3f}, "
                f"sort_err={evaluation['sorting_error_mean']:.1f}, "
                f"prune_err={evaluation['pruning_error_mean']:.1f}, "
                f"sat={evaluation['strict_form']['saturation_rate']:.3f}, "
                f"best={evaluation['best_heuristic']}:{evaluation['best_heuristic_spearman']:.3f}"
            )
    save_results(results)


if __name__ == "__main__":
    main()
