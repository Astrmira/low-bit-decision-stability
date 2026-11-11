#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12 - Probe 29：P4-G dependency-sliced edge certificate

区分 order-conditional pruning certificate 与包含 full-order events 的完整 edge certificate，
并验证 diversity-selected / fallback-added edge 的 1-certificate 充分性。
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

DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_SAMPLE = 4_000
N_CALIBRATION_TARGETS = 80
N_EVALUATION_TARGETS = 160
CANDIDATE_K = 64
MAX_K = 32
ALPHA = 1.2
SEED = 20260823


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_sampled_dataset(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 29 只支持 f32 数据集，收到 {data_format}")
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


def candidate_event(exact_distances, approx_distances, positions, target, decision, model):
    candidate = decision["candidate"]
    prefix = decision["selected_prefix"]
    target_pos = positions[target]
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
        orientation = "left" if exact_psi >= 0 else "right"
        risk = trace.event_risk(
            abs(exact_psi),
            model["pruning_mean"],
            model["pruning_scale"],
            orientation,
        )
        local.append((exact_psi, approx_psi, risk))
    exact_dominated = bool(decision["dominated"])
    approx_dominated = any(approx_psi >= 0 for _, approx_psi, _ in local)
    if not local:
        risk = 0.0
    elif exact_dominated:
        risk = min(risk for psi, _, risk in local if psi >= 0)
    else:
        risk = min(1.0, sum(risk for _, _, risk in local))
    return {
        "candidate": candidate,
        "rank": decision["candidate_rank"],
        "exact_dominated": exact_dominated,
        "approx_dominated": approx_dominated,
        "good": exact_dominated == approx_dominated,
        "risk": float(risk),
        "tie": any(psi == 0.0 for psi, _, _ in local),
    }


def evaluate_target(vecs, target, candidates, encoded, model):
    slope = model["distance_calibration"]["slope"]
    exact_nodes, exact_distances = pruning.distance_cache(vecs, target, candidates)
    _, approx_distances = pruning.distance_cache(
        vecs, target, candidates, encoded=encoded, slope=slope
    )
    exact_run = pruning.run_select(target, candidates, exact_nodes, exact_distances)
    approx_fixed = pruning.run_select(
        target,
        candidates,
        exact_nodes,
        approx_distances,
        ordered_override=exact_run["ordered"],
    )
    approx_free = pruning.run_select(target, candidates, exact_nodes, approx_distances)
    positions = {int(node): index for index, node in enumerate(exact_nodes)}
    target_pos = positions[target]

    approx_order_positions = {
        candidate: index for index, candidate in enumerate(approx_free["ordered"])
    }
    sorting_events = []
    for left, right in zip(exact_run["ordered"][:-1], exact_run["ordered"][1:]):
        exact_margin = float(
            exact_distances[target_pos, positions[right]]
            - exact_distances[target_pos, positions[left]]
        )
        sorting_events.append(
            {
                "good": approx_order_positions[left] < approx_order_positions[right],
                "risk": 1.0
                if exact_margin == 0.0
                else trace.event_risk(
                    exact_margin,
                    model["sorting_mean"],
                    model["sorting_scale"],
                    "left",
                ),
                "tie": exact_margin == 0.0,
            }
        )

    pruning_events = [
        candidate_event(
            exact_distances,
            approx_distances,
            positions,
            target,
            decision,
            model,
        )
        for decision in exact_run["decisions"]
    ]
    event_by_candidate = {event["candidate"]: index for index, event in enumerate(pruning_events)}
    exact_diversity = set(exact_run["diversity_selected"])
    exact_fallback = set(exact_run["fallback_added"])
    fixed_final = set(approx_fixed["final_selected"])
    free_final = set(approx_free["final_selected"])
    sorting_good = all(event["good"] for event in sorting_events)
    sorting_risk_sum = sum(event["risk"] for event in sorting_events)
    edges = []
    for candidate in exact_run["final_selected"]:
        edge_type = "diversity" if candidate in exact_diversity else "fallback"
        if edge_type == "diversity":
            end = event_by_candidate[candidate]
            required = pruning_events[: end + 1]
        else:
            required = pruning_events
        pruning_good = all(event["good"] for event in required)
        order_conditional_survives = candidate in fixed_final
        free_survives = candidate in free_final
        if pruning_good and not order_conditional_survives:
            raise AssertionError("order-conditional 1-certificate 充分性被违反")
        if sorting_good and pruning_good and not free_survives:
            raise AssertionError("full edge 1-certificate 充分性被违反")
        pruning_risk_sum = float(sum(event["risk"] for event in required))
        edges.append(
            {
                "edge_type": edge_type,
                "pruning_event_count": len(required),
                "sorting_event_count": len(sorting_events),
                "order_conditional_good": pruning_good,
                "full_certificate_good": sorting_good and pruning_good,
                "fixed_order_survives": order_conditional_survives,
                "free_order_survives": free_survives,
                "pruning_risk_sum": pruning_risk_sum,
                "pruning_risk_clipped": min(1.0, pruning_risk_sum),
                "full_risk_sum": pruning_risk_sum + sorting_risk_sum,
                "full_risk_clipped": min(1.0, pruning_risk_sum + sorting_risk_sum),
                "tie_event_count": sum(event["tie"] for event in required)
                + sum(event["tie"] for event in sorting_events),
            }
        )
    return edges


def group_summary(rows):
    if not rows:
        return {
            "edge_count": 0,
            "status": "该标准非饱和运行不存在此类边",
            "certificate_sufficiency_holds": True,
        }
    fixed_failure = np.array(
        [not row["fixed_order_survives"] for row in rows], dtype=np.float64
    )
    free_failure = np.array(
        [not row["free_order_survives"] for row in rows], dtype=np.float64
    )
    pruning_risk = np.array([row["pruning_risk_sum"] for row in rows], dtype=np.float64)
    full_risk = np.array([row["full_risk_sum"] for row in rows], dtype=np.float64)
    return {
        "edge_count": len(rows),
        "mean_pruning_event_count": float(
            np.mean([row["pruning_event_count"] for row in rows])
        ),
        "mean_sorting_event_count": float(
            np.mean([row["sorting_event_count"] for row in rows])
        ),
        "fixed_order_failure_rate": float(fixed_failure.mean()),
        "free_order_failure_rate": float(free_failure.mean()),
        "order_conditional_certificate_pass_rate": float(
            np.mean([row["order_conditional_good"] for row in rows])
        ),
        "full_certificate_pass_rate": float(
            np.mean([row["full_certificate_good"] for row in rows])
        ),
        "pruning_certificate_saturation": float(
            np.mean([row["pruning_risk_clipped"] >= 1.0 for row in rows])
        ),
        "full_certificate_saturation": float(
            np.mean([row["full_risk_clipped"] >= 1.0 for row in rows])
        ),
        "pruning_risk_failure_spearman": safe_spearman(pruning_risk, fixed_failure),
        "full_risk_failure_spearman": safe_spearman(full_risk, free_failure),
        "mean_pruning_risk_sum": float(pruning_risk.mean()),
        "mean_full_risk_sum": float(full_risk.mean()),
        "tie_event_mean": float(np.mean([row["tie_event_count"] for row in rows])),
        "certificate_sufficiency_holds": bool(
            all(
                not row["order_conditional_good"] or row["fixed_order_survives"]
                for row in rows
            )
            and all(
                not row["full_certificate_good"] or row["free_order_survives"]
                for row in rows
            )
        ),
    }


def summarize(edges):
    output = {"all": group_summary(edges)}
    for edge_type in ["diversity", "fallback"]:
        rows = [row for row in edges if row["edge_type"] == edge_type]
        output[edge_type] = group_summary(rows)
    return output


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
        model = trace.fit_event_models(
            vecs, calibration_targets, calibration_candidates, encoded
        )
        edges = [
            edge
            for target, candidates in zip(evaluation_targets, evaluation_candidates)
            for edge in evaluate_target(vecs, int(target), candidates, encoded, model)
        ]
        result["quantizers"][quantizer] = {
            "calibration": model,
            "evaluation": summarize(edges),
        }
    return result


def save_results(results):
    output = {
        "probe": 29,
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
    path = RESULTS_DIR / "probe29_standard_vamana_edge_certificate.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            all_edges = metrics["evaluation"]["all"]
            diversity = metrics["evaluation"]["diversity"]
            fallback = metrics["evaluation"]["fallback"]
            print(
                f"  {quantizer:<13} fixed_fail={all_edges['fixed_order_failure_rate']:.3f}, "
                f"free_fail={all_edges['free_order_failure_rate']:.3f}, "
                f"slice={all_edges['mean_pruning_event_count']:.1f}, "
                f"prune_sat={all_edges['pruning_certificate_saturation']:.3f}, "
                f"div/fb={diversity['edge_count']}/{fallback['edge_count']}"
            )
    save_results(results)


if __name__ == "__main__":
    main()
