#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""19 - Probe 36：H1 joint、谱、边界、atom、LOO 与 shell 审计。"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, kstest, kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = [
    "cohere",
    "minilm",
    "bge_m3",
    "wolt_clip",
    "landmark_dino",
    "glove100",
    "sift128",
    "gist960",
]
N_SAMPLE = 4_000
PAIR_SAMPLE = 100_000
SUBSET_TRIALS = 24
K_VALUES = [2, 4, 8]
SMALL_BALL_QUANTILES = [0.01, 0.02, 0.05, 0.10]
ALPHA = 1.2
SEED = 20260826


def load_artifact(name):
    prefix, dimension, kind, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 36 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED + dimension)
    indices = rng.choice(total, min(N_SAMPLE, total), replace=False)
    values = np.asarray(mapped[indices], dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    normalized = values / np.maximum(norms[:, None], 1e-12)
    return normalized, norms, total, kind


def covariance_audit(values):
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance).astype(np.float64), 0)
    trace = float(eigenvalues.sum())
    operator = float(eigenvalues[-1])
    frobenius_sq = float(np.sum(eigenvalues**2))
    return {
        "effective_rank": trace / max(operator, 1e-300),
        "stable_rank": frobenius_sq / max(operator**2, 1e-300),
        "top_eigenvalue_share": operator / max(trace, 1e-300),
        "top10_share": float(eigenvalues[-10:].sum() / max(trace, 1e-300)),
        "operator_to_frobenius": operator / max(np.sqrt(frobenius_sq), 1e-300),
    }


def joint_gaussian_audit(values):
    rng = np.random.default_rng(SEED + values.shape[1] + 1)
    output = {}
    for k in K_VALUES:
        records = []
        for _ in range(SUBSET_TRIALS):
            coordinates = rng.choice(values.shape[1], k, replace=False)
            sample = values[:, coordinates].astype(np.float64)
            mean = sample.mean(axis=0)
            covariance = np.cov(sample, rowvar=False, ddof=1)
            precision = np.linalg.pinv(covariance, rcond=1e-10)
            centered = sample - mean
            radius = np.einsum("ni,ij,nj->n", centered, precision, centered)
            ks = kstest(radius, chi2(df=k).cdf)
            whitened = centered @ np.linalg.cholesky(precision + 1e-10 * np.eye(k)).T
            records.append(
                {
                    "radius_ks": float(ks.statistic),
                    "radius_pvalue": float(ks.pvalue),
                    "mean_abs_skew": float(np.mean(np.abs(skew(whitened, axis=0, bias=False)))),
                    "mean_abs_excess_kurtosis": float(
                        np.mean(np.abs(kurtosis(whitened, axis=0, fisher=True, bias=False)))
                    ),
                }
            )
        output[str(k)] = {
            "median_radius_ks": float(np.median([row["radius_ks"] for row in records])),
            "p90_radius_ks": float(np.quantile([row["radius_ks"] for row in records], 0.90)),
            "rejection_rate_0_01": float(
                np.mean([row["radius_pvalue"] < 0.01 for row in records])
            ),
            "median_abs_skew": float(np.median([row["mean_abs_skew"] for row in records])),
            "median_abs_excess_kurtosis": float(
                np.median([row["mean_abs_excess_kurtosis"] for row in records])
            ),
        }
    return output


def pair_indices(size, count, seed):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, size, count)
    right = rng.integers(0, size, count)
    same = left == right
    right[same] = (right[same] + 1) % size
    return left, right


def encode(values):
    threshold = np.abs(values).mean(axis=1, keepdims=True)
    signs = values > 0
    strongs = np.abs(values) > threshold
    return signs, strongs


def raw_score(left, right, encoded):
    signs, strongs = encoded
    agreement = signs[left] == signs[right]
    signed = 2 * agreement.astype(np.int16) - 1
    weights = (1 + strongs[left].astype(np.int16)) * (
        1 + strongs[right].astype(np.int16)
    )
    return (signed * weights).sum(axis=1, dtype=np.int64)


def boundary_atom_audit(values):
    rng = np.random.default_rng(SEED + 22)
    nodes = rng.integers(0, len(values), (PAIR_SAMPLE, 3))
    q, x, y = nodes.T
    same = (q == x) | (q == y) | (x == y)
    y[same] = (y[same] + 1) % len(values)
    score_qx = np.sum(values[q] * values[x], axis=1)
    score_qy = np.sum(values[q] * values[y], axis=1)
    score_xy = np.sum(values[x] * values[y], axis=1)
    ranking = np.abs(score_qx - score_qy)
    pruning = np.abs((1 - score_qx) - ALPHA * (1 - score_xy))

    def curve(margins):
        scale = max(float(np.std(margins, ddof=1)), 1e-12)
        return [
            {
                "tau_over_sd": ratio,
                "mass": float(np.mean(margins <= ratio * scale)),
            }
            for ratio in SMALL_BALL_QUANTILES
        ]

    left, right = pair_indices(len(values), PAIR_SAMPLE, SEED + 23)
    score = raw_score(left, right, encode(values))
    _, counts = np.unique(score, return_counts=True)
    return {
        "ranking_exact_tie_rate_1e_12": float(np.mean(ranking <= 1e-12)),
        "pruning_exact_tie_rate_1e_12": float(np.mean(pruning <= 1e-12)),
        "ranking_small_ball": curve(ranking),
        "pruning_small_ball": curve(pruning),
        "quantized_unique_scores": int(len(counts)),
        "quantized_max_atom_mass": float(counts.max() / len(score)),
        "quantized_pair_tie_probability": float(np.sum((counts / len(score)) ** 2)),
    }


def threshold_loo_audit(values):
    absolute = np.abs(values).astype(np.float64)
    total = absolute.sum(axis=1, keepdims=True)
    full = total / values.shape[1]
    full_bits = absolute > full
    rng = np.random.default_rng(SEED + values.shape[1] + 31)
    removed_coordinates = rng.choice(values.shape[1], min(32, values.shape[1]), replace=False)
    flip_fractions = []
    vector_flip_rates = []
    threshold_influences = []
    for coordinate in removed_coordinates:
        loo = (total - absolute[:, coordinate : coordinate + 1]) / (values.shape[1] - 1)
        loo_bits = absolute > loo
        loo_bits[:, coordinate] = full_bits[:, coordinate]
        flips = full_bits != loo_bits
        flip_fractions.append(float(np.mean(flips)))
        vector_flip_rates.append(float(np.mean(np.any(flips, axis=1))))
        threshold_influences.append(float(np.mean(np.abs(loo - full))))
    return {
        "removed_coordinate_trials": int(len(removed_coordinates)),
        "other_bit_flip_fraction": float(np.mean(flip_fractions)),
        "vector_any_other_flip_rate": float(np.mean(vector_flip_rates)),
        "mean_threshold_influence": float(np.mean(threshold_influences)),
        "dimension_scaled_other_flip_fraction": float(
            values.shape[1] * np.mean(flip_fractions)
        ),
        "status": "leave-one-coordinate-out perturbation evaluated on all other magnitude bits",
    }


def shell_audit(norms):
    fraction_unit = float(np.mean(np.abs(norms - 1) < 1e-5))
    identifiable = fraction_unit < 0.999
    output = {
        "identifiable": identifiable,
        "fraction_within_1e_5_of_one": fraction_unit,
    }
    if identifiable:
        center = float(norms.mean())
        relative = np.abs(norms / center - 1)
        output |= {
            "coefficient_of_variation": float(norms.std(ddof=1) / center),
            "relative_deviation_p95": float(np.quantile(relative, 0.95)),
            "relative_deviation_p99": float(np.quantile(relative, 0.99)),
        }
    else:
        output["reason"] = "artifact is already unit-normalized"
    return output


def run_dataset(name):
    values, norms, total, kind = load_artifact(name)
    return {
        "dataset": name,
        "kind": kind,
        "dimension": int(values.shape[1]),
        "source_total": int(total),
        "joint_gaussianity": joint_gaussian_audit(values),
        "covariance_spectrum": covariance_audit(values),
        "boundary_and_atoms": boundary_atom_audit(values),
        "shared_threshold_loo": threshold_loo_audit(values),
        "shell": shell_audit(norms),
    }


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        joint = result["joint_gaussianity"]["8"]
        atom = result["boundary_and_atoms"]["quantized_max_atom_mass"]
        loo = result["shared_threshold_loo"]["other_bit_flip_fraction"]
        print(
            f"  k8-KS={joint['median_radius_ks']:.4f}, atom={atom:.4f}, "
            f"loo-flip={loo:.6f}, shell={result['shell']['identifiable']}"
        )
    output = {
        "probe": 36,
        "seed": SEED,
        "n_sample": N_SAMPLE,
        "pair_sample": PAIR_SAMPLE,
        "subset_trials": SUBSET_TRIALS,
        "k_values": K_VALUES,
        "results": results,
        "status": "empirical eligibility and falsification audit, not a proof of H1",
    }
    path = RESULTS_DIR / "probe36_standard_vamana_h1_robustness_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
