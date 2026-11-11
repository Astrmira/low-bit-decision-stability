#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""16 - Probe 33：P6 严格分割 held-out calibration。"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = [
    "cohere",
    "minilm",
    "bge_m3",
    "wolt_clip",
    "codesearch_jina",
    "landmark_dino",
    "glove100",
    "sift128",
    "gist960",
    "sphere",
    "random",
    "msmarco_cohere",
]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_PER_SPLIT = 2_000
N_CALIBRATION_TARGETS = 48
N_VALIDATION_TARGETS = 96
CANDIDATE_K = 48
MAX_K = 24
ALPHA = 1.2
STRESS_FRACTION = 0.2
SEED = 20260824
TAU_QUANTILES = np.array([0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70])
RETENTION_QUANTILES = np.array([0.20, 0.40, 0.60, 0.80, 1.00])


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_splits(name):
    prefix, dimension, kind, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 33 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    required = min(3 * N_PER_SPLIT, total)
    if required < 300:
        raise ValueError(f"数据集 {name} 的可用向量不足：{total}")
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED + dimension)
    indices = rng.choice(total, required, replace=False)
    values = np.asarray(mapped[indices], dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    size = required // 3
    return {
        "calibration": values[:size],
        "validation": values[size : 2 * size],
        "stress": values[2 * size : 3 * size],
    }, total, kind


def rotate(values):
    rng = np.random.default_rng(SEED + 17 * values.shape[1])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), values.shape[1])
    output = dct(values * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return output


def encode(values):
    signs = values > 0
    strongs = np.abs(values) > np.abs(values).mean(axis=1, keepdims=True)
    return signs, strongs


def raw_distances(left, right, encoded):
    signs, strongs = encoded
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    agreement = signs[left] == signs[right]
    signed = 2 * agreement.astype(np.int16) - 1
    weights = (1 + strongs[left].astype(np.int16)) * (
        1 + strongs[right].astype(np.int16)
    )
    score = (signed * weights).sum(axis=1, dtype=np.int64)
    return (4 * signs.shape[1] - score).astype(np.float64)


def target_candidates(values, count, seed):
    rng = np.random.default_rng(seed)
    targets = rng.choice(len(values), min(count, len(values)), replace=False)
    similarities = values[targets] @ values.T
    output = []
    for row, target in enumerate(targets):
        similarities[row, target] = -np.inf
        candidates = np.argpartition(-similarities[row], CANDIDATE_K)[:CANDIDATE_K]
        order = np.lexsort((candidates, -similarities[row, candidates]))
        output.append(candidates[order].astype(np.int64))
    return targets, output


def fit_calibration(values, targets, candidate_lists, encoded):
    exact = []
    raw = []
    for target, candidates in zip(targets, candidate_lists):
        left = np.full(len(candidates), target, dtype=np.int64)
        exact.extend(1.0 - values[candidates] @ values[target])
        raw.extend(raw_distances(left, candidates, encoded))
    exact = np.asarray(exact, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    design = np.column_stack([np.ones(len(raw)), raw])
    beta, _, _, _ = np.linalg.lstsq(design, exact, rcond=None)
    residual = design @ beta - exact
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "distance_mse": float(np.mean(residual**2)),
        "distance_spearman": safe_spearman(raw, exact),
    }


def calibrated_distances(left, right, encoded, model):
    return model["intercept"] + model["slope"] * raw_distances(left, right, encoded)


def select_exact(values, target, candidates):
    selected = []
    comparisons = []
    target_distances = 1.0 - values[candidates] @ values[target]
    order = np.lexsort((candidates, target_distances))
    ordered = candidates[order]
    for candidate in ordered:
        if len(selected) >= MAX_K:
            break
        distance_tc = float(1.0 - np.dot(values[target], values[candidate]))
        dominated = False
        for selected_node in selected:
            distance_cs = float(1.0 - np.dot(values[candidate], values[selected_node]))
            psi = distance_tc - ALPHA * distance_cs
            comparisons.append((target, candidate, selected_node, psi))
            if psi >= 0:
                dominated = True
                break
        if not dominated:
            selected.append(int(candidate))
    return ordered, comparisons


def collect_decisions(values, targets, candidate_lists, encoded, model):
    ranking_margin = []
    ranking_residual = []
    ranking_flip = []
    pruning_margin = []
    pruning_residual = []
    pruning_flip = []
    for target, candidates in zip(targets, candidate_lists):
        exact_distances = 1.0 - values[candidates] @ values[target]
        exact_order = np.lexsort((candidates, exact_distances))
        ordered = candidates[exact_order]
        top = int(ordered[0])
        competitors = ordered[1:]
        approx_top = calibrated_distances(
            np.full(len(competitors), target), np.full(len(competitors), top), encoded, model
        )
        approx_competitors = calibrated_distances(
            np.full(len(competitors), target), competitors, encoded, model
        )
        exact_top = float(1.0 - np.dot(values[target], values[top]))
        exact_competitors = 1.0 - values[competitors] @ values[target]
        exact_role = exact_competitors - exact_top
        approx_role = approx_competitors - approx_top
        ranking_margin.extend(exact_role)
        ranking_residual.extend(approx_role - exact_role)
        ranking_flip.extend(approx_role <= 0)
        _, comparisons = select_exact(values, target, candidates)
        if comparisons:
            nodes = np.asarray([(t, c, s) for t, c, s, _ in comparisons], dtype=np.int64)
            exact_psi = np.asarray([psi for _, _, _, psi in comparisons])
            approx_tc = calibrated_distances(nodes[:, 0], nodes[:, 1], encoded, model)
            approx_cs = calibrated_distances(nodes[:, 1], nodes[:, 2], encoded, model)
            approx_psi = approx_tc - ALPHA * approx_cs
            pruning_margin.extend(np.abs(exact_psi))
            pruning_residual.extend(approx_psi - exact_psi)
            pruning_flip.extend((exact_psi >= 0) != (approx_psi >= 0))
    return {
        "ranking": arrays(ranking_margin, ranking_residual, ranking_flip),
        "pruning": arrays(pruning_margin, pruning_residual, pruning_flip),
    }


def arrays(margin, residual, flip):
    return {
        "margin": np.asarray(margin, dtype=np.float64),
        "residual": np.asarray(residual, dtype=np.float64),
        "flip": np.asarray(flip, dtype=bool),
    }


def fit_risk_model(decisions):
    margin = decisions["margin"]
    residual = decisions["residual"]
    positive = margin[margin > 0]
    taus = np.unique(np.quantile(positive, TAU_QUANTILES))
    risks = np.asarray(
        [np.mean(margin <= tau) + np.mean(np.abs(residual) >= tau) for tau in taus]
    )
    best = int(np.argmin(risks))
    scale = max(float(np.std(residual, ddof=1)), 1e-12)
    standardized = margin / scale
    cutoffs = np.quantile(standardized, 1 - RETENTION_QUANTILES)
    return {
        "tau": float(taus[best]),
        "plugin_risk": float(min(1.0, risks[best])),
        "residual_scale": scale,
        "residual_bias": float(np.mean(residual)),
        "retention_cutoffs": cutoffs.tolist(),
        "calibration_count": int(len(margin)),
    }


def evaluate_risk(decisions, risk_model, stress_only=False):
    margin = decisions["margin"]
    residual = decisions["residual"]
    flip = decisions["flip"]
    if stress_only:
        cutoff = np.quantile(margin, STRESS_FRACTION)
        mask = margin <= cutoff
        margin = margin[mask]
        residual = residual[mask]
        flip = flip[mask]
    tau = risk_model["tau"]
    boundary = float(np.mean(margin <= tau))
    tail = float(np.mean(np.abs(residual) >= tau))
    standardized = margin / risk_model["residual_scale"]
    selective = []
    for retention, cutoff in zip(RETENTION_QUANTILES, risk_model["retention_cutoffs"]):
        retained = standardized >= cutoff
        selective.append(
            {
                "target_retention": float(retention),
                "actual_retention": float(np.mean(retained)),
                "flip_rate": float(np.mean(flip[retained])) if np.any(retained) else None,
            }
        )
    order = np.argsort(margin)
    bins = []
    for indices in np.array_split(order, 5):
        bins.append(
            {
                "mean_margin": float(np.mean(margin[indices])),
                "flip_rate": float(np.mean(flip[indices])),
                "count": int(len(indices)),
            }
        )
    return {
        "count": int(len(margin)),
        "flip_rate": float(np.mean(flip)),
        "heldout_boundary_mass": boundary,
        "heldout_residual_tail": tail,
        "heldout_plugin_risk": float(min(1.0, boundary + tail)),
        "frozen_calibration_risk": risk_model["plugin_risk"],
        "frozen_risk_covers_flip": bool(risk_model["plugin_risk"] + 1e-12 >= np.mean(flip)),
        "margin_bin_flip_spearman": safe_spearman(
            [row["mean_margin"] for row in bins], [row["flip_rate"] for row in bins]
        ),
        "margin_bins": bins,
        "selective_risk": selective,
    }


def run_quantizer(splits, quantizer, split_records):
    transformed = {
        name: rotate(values) if quantizer == "rotated_2bit" else values
        for name, values in splits.items()
    }
    encoded = {name: encode(values) for name, values in transformed.items()}
    calibration_targets, calibration_candidates = split_records["calibration"]
    model = fit_calibration(
        splits["calibration"], calibration_targets, calibration_candidates, encoded["calibration"]
    )
    decisions = {}
    for split in ("calibration", "validation", "stress"):
        targets, candidates = split_records[split]
        decisions[split] = collect_decisions(
            splits[split], targets, candidates, encoded[split], model
        )
    roles = {}
    for role in ("ranking", "pruning"):
        risk_model = fit_risk_model(decisions["calibration"][role])
        roles[role] = {
            "risk_model": risk_model,
            "validation": evaluate_risk(decisions["validation"][role], risk_model),
            "stress": evaluate_risk(decisions["stress"][role], risk_model, stress_only=True),
        }
    return {
        "distance_calibration": model,
        "eligibility": {
            "eligible": model["slope"] > 0,
            "reasons": [] if model["slope"] > 0 else ["校准斜率非正"],
        },
        "roles": roles,
    }


def run_dataset(name):
    splits, total, kind = load_splits(name)
    split_records = {
        "calibration": target_candidates(
            splits["calibration"], N_CALIBRATION_TARGETS, SEED + 101
        ),
        "validation": target_candidates(
            splits["validation"], N_VALIDATION_TARGETS, SEED + 102
        ),
        "stress": target_candidates(
            splits["stress"], N_VALIDATION_TARGETS, SEED + 103
        ),
    }
    return {
        "dataset": name,
        "kind": kind,
        "dimension": int(splits["calibration"].shape[1]),
        "source_total": int(total),
        "split_size": int(len(splits["calibration"])),
        "quantizers": {
            quantizer: run_quantizer(splits, quantizer, split_records)
            for quantizer in QUANTIZERS
        },
    }


def aggregate(results):
    output = {}
    for role in ("ranking", "pruning"):
        records = []
        for dataset in results:
            for quantizer, metrics in dataset["quantizers"].items():
                validation = metrics["roles"][role]["validation"]
                records.append(
                    {
                        "dataset": dataset["dataset"],
                        "quantizer": quantizer,
                        "flip_rate": validation["flip_rate"],
                        "predicted_risk": validation["frozen_calibration_risk"],
                        "distance_spearman": metrics["distance_calibration"]["distance_spearman"],
                        "distance_mse": metrics["distance_calibration"]["distance_mse"],
                        "eligible": metrics["eligibility"]["eligible"],
                        "covered": validation["frozen_risk_covers_flip"],
                    }
                )
        eligible_records = [row for row in records if row["eligible"]]
        output[role] = {
            "configuration_count": len(records),
            "eligible_configuration_count": len(eligible_records),
            "coverage_rate": float(np.mean([row["covered"] for row in records])),
            "risk_flip_spearman": safe_spearman(
                [row["predicted_risk"] for row in records], [row["flip_rate"] for row in records]
            ),
            "distance_spearman_flip_spearman": safe_spearman(
                [-row["distance_spearman"] for row in records], [row["flip_rate"] for row in records]
            ),
            "distance_mse_flip_spearman": safe_spearman(
                [row["distance_mse"] for row in records], [row["flip_rate"] for row in records]
            ),
            "eligible_risk_flip_spearman": safe_spearman(
                [row["predicted_risk"] for row in eligible_records],
                [row["flip_rate"] for row in eligible_records],
            ),
            "eligible_coverage_rate": float(
                np.mean([row["covered"] for row in eligible_records])
            ),
            "records": records,
        }
    return output


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            ranking = metrics["roles"]["ranking"]["validation"]
            pruning = metrics["roles"]["pruning"]["validation"]
            print(
                f"  {quantizer:<13} rank={ranking['flip_rate']:.4f}/"
                f"{ranking['frozen_calibration_risk']:.4f}, prune={pruning['flip_rate']:.4f}/"
                f"{pruning['frozen_calibration_risk']:.4f}"
            )
    output = {
        "probe": 33,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "n_per_split": N_PER_SPLIT,
        "n_calibration_targets": N_CALIBRATION_TARGETS,
        "n_validation_targets": N_VALIDATION_TARGETS,
        "candidate_k": CANDIDATE_K,
        "max_k": MAX_K,
        "alpha": ALPHA,
        "stress_fraction": STRESS_FRACTION,
        "protocol": "disjoint calibration, validation, and stress vector pools",
        "status": "empirical held-out diagnostic, not a finite-sample theorem certificate",
        "aggregate": aggregate(results),
        "results": results,
    }
    path = RESULTS_DIR / "probe33_standard_vamana_p6_heldout_calibration.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


if __name__ == "__main__":
    main()
