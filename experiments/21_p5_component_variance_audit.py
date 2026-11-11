#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""21 - Probe 38：P5 component variance 与 replacement influence 审计。"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = ["cohere", "minilm", "bge_m3", "wolt_clip", "sift128", "gist960", "random"]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_FIT = 1_200
N_EVALUATION = 2_400
N_FIT_PAIRS = 24_000
N_ROLE_SAMPLES = 20_000
ALPHA = 1.2
SEED = 20260828


def load_split(name):
    prefix, dimension, kind, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 38 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    count = min(N_FIT + N_EVALUATION, total)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED + dimension)
    indices = rng.choice(total, count, replace=False)
    values = np.asarray(mapped[indices], dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    cut = min(N_FIT, count // 2)
    return values[:cut], values[cut:], kind, total


def rotate(values, signs):
    output = dct(values * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return output


def encode(values, population_threshold):
    return values > 0, np.abs(values) > population_threshold


def raw_distance(left, right, encoded):
    signs, strongs = encoded
    agreement = signs[left] == signs[right]
    signed = 2 * agreement.astype(np.int16) - 1
    weights = (1 + strongs[left].astype(np.int16)) * (1 + strongs[right].astype(np.int16))
    score = (signed * weights).sum(axis=1, dtype=np.int64)
    return (4 * signs.shape[1] - score).astype(np.float64)


def pairs(size, count, seed):
    rng = np.random.default_rng(seed)
    output = rng.integers(0, size, (count, 2))
    equal = output[:, 0] == output[:, 1]
    output[equal, 1] = (output[equal, 1] + 1) % size
    return output[:, 0], output[:, 1]


def fit_calibration(exact_values, transformed, seed):
    threshold = float(np.abs(transformed).mean(axis=1).mean())
    encoded = encode(transformed, threshold)
    left, right = pairs(len(exact_values), N_FIT_PAIRS, seed)
    exact = 1.0 - np.sum(exact_values[left] * exact_values[right], axis=1)
    raw = raw_distance(left, right, encoded)
    design = np.column_stack([np.ones(len(raw)), raw])
    beta, _, _, _ = np.linalg.lstsq(design, exact, rcond=None)
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "threshold": threshold}


def role_values(exact_values, transformed, calibration, nodes, role):
    encoded = encode(transformed, calibration["threshold"])
    q, x, y = nodes.T

    def edge(left, right):
        exact_similarity = np.sum(exact_values[left] * exact_values[right], axis=1)
        score_component = calibration["slope"] * raw_distance(left, right, encoded)
        residual = calibration["intercept"] + score_component - (1.0 - exact_similarity)
        return residual, score_component, exact_similarity

    qx = edge(q, x)
    qy = edge(q, y)
    xy = edge(x, y)
    if role == "ranking":
        coefficients = (-1.0, 1.0, 0.0)
    else:
        coefficients = (1.0, 0.0, -ALPHA)
    total = sum(weight * item[0] for weight, item in zip(coefficients, (qx, qy, xy)))
    bounded = sum(weight * item[1] for weight, item in zip(coefficients, (qx, qy, xy)))
    quadratic = sum(weight * item[2] for weight, item in zip(coefficients, (qx, qy, xy)))
    return total, bounded, quadratic


def replacement_audit(exact_values, transformed, calibration, nodes, role, seed):
    base, _, _ = role_values(exact_values, transformed, calibration, nodes, role)
    rng = np.random.default_rng(seed)
    squared = []
    absolute = []
    for column in range(3):
        replaced = nodes.copy()
        replaced[:, column] = rng.integers(0, len(exact_values), len(nodes))
        changed, _, _ = role_values(exact_values, transformed, calibration, replaced, role)
        difference = base - changed
        squared.append(float(np.mean(difference**2)))
        absolute.append(np.abs(difference))
    all_absolute = np.concatenate(absolute)
    return {
        "efron_stein_proxy": 0.5 * sum(squared),
        "node_mean_squared_differences": squared,
        "absolute_difference_p95": float(np.quantile(all_absolute, 0.95)),
        "absolute_difference_p99": float(np.quantile(all_absolute, 0.99)),
        "absolute_difference_max": float(np.max(all_absolute)),
    }


def tail_audit(centered, proxies, scale):
    sd = max(float(np.std(centered, ddof=1)), 1e-12)
    rows = []
    for multiple in [1.0, 1.5, 2.0, 2.5, 3.0]:
        threshold = multiple * sd
        empirical = float(np.mean(np.abs(centered) >= threshold))
        bounds = {}
        for name, variance in proxies.items():
            bound = min(1.0, float(2 * np.exp(-(threshold**2) / (2 * (variance + scale * threshold / 3)))))
            bounds[name] = bound
        rows.append({"sd_multiple": multiple, "empirical_tail": empirical, "candidate_bounds": bounds})
    return rows


def summarize_role(exact_values, transformed, calibration, nodes, role, seed, coarse_proxy):
    total, bounded, quadratic = role_values(exact_values, transformed, calibration, nodes, role)
    total_centered = total - total.mean()
    bounded_centered = bounded - bounded.mean()
    quadratic_centered = quadratic - quadratic.mean()
    total_variance = float(np.var(total_centered, ddof=1))
    bounded_variance = float(np.var(bounded_centered, ddof=1))
    quadratic_variance = float(np.var(quadratic_centered, ddof=1))
    cross_covariance = float(np.cov(bounded_centered, quadratic_centered, ddof=1)[0, 1])
    component_proxy = (np.sqrt(bounded_variance) + np.sqrt(quadratic_variance)) ** 2
    replacement = replacement_audit(exact_values, transformed, calibration, nodes, role, seed)
    candidate_proxies = {
        "component_cauchy": component_proxy,
        "efron_stein": replacement["efron_stein_proxy"],
        "coarse": coarse_proxy,
    }
    scale = replacement["absolute_difference_max"]
    return {
        "total_variance": total_variance,
        "bounded_component_variance": bounded_variance,
        "quadratic_component_variance": quadratic_variance,
        "component_cross_covariance": cross_covariance,
        "variance_identity_error": abs(
            total_variance - (bounded_variance + quadratic_variance + 2 * cross_covariance)
        ),
        "component_cauchy_proxy": component_proxy,
        "component_cauchy_looseness": component_proxy / max(total_variance, 1e-300),
        "coarse_proxy": coarse_proxy,
        "coarse_looseness": coarse_proxy / max(total_variance, 1e-300),
        "replacement": replacement | {
            "looseness": replacement["efron_stein_proxy"] / max(total_variance, 1e-300)
        },
        "tail_audit": tail_audit(total_centered, candidate_proxies, scale),
        "status": "component and replacement proxies are empirical diagnostics, not theorem certificates",
    }


def analyze(name):
    fit, evaluation, kind, source_total = load_split(name)
    dimension = evaluation.shape[1]
    rng = np.random.default_rng(SEED + dimension * 31)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), dimension)
    nodes = rng.integers(0, len(evaluation), (N_ROLE_SAMPLES, 3))
    collision = (nodes[:, 0] == nodes[:, 1]) | (nodes[:, 0] == nodes[:, 2]) | (nodes[:, 1] == nodes[:, 2])
    nodes[collision, 2] = (nodes[collision, 2] + 1) % len(evaluation)
    configurations = []
    for quantizer in QUANTIZERS:
        fit_transformed = fit if quantizer == "2bit" else rotate(fit, signs)
        transformed = evaluation if quantizer == "2bit" else rotate(evaluation, signs)
        calibration = fit_calibration(fit, fit_transformed, SEED + dimension)
        beta = calibration["slope"]
        covariance = np.cov(transformed, rowvar=False, ddof=1)
        sigma_frobenius_sq = float(np.sum(covariance**2))
        sigma_operator = float(np.linalg.eigvalsh(covariance)[-1])
        mean = transformed.mean(axis=0)
        mu_sigma_mu = float(mean @ covariance @ mean)

        def coarse(sum_a_sq, sum_d_sq, matrix_op, a1):
            vq = sigma_frobenius_sq * sum_a_sq + mu_sigma_mu * sum_d_sq
            bounded_range = 8 * dimension * beta * a1
            return 2 * vq + bounded_range**2 / 2

        ranking_coarse = coarse(2.0, 2.0, np.sqrt(2) / 2, 2.0)
        pruning_coarse = coarse(
            ALPHA**2 + 1,
            ALPHA**2 + (ALPHA - 1) ** 2 + 1,
            np.sqrt(ALPHA**2 + 1) / 2,
            ALPHA + 1,
        )
        configurations.append(
            {
                "quantizer": quantizer,
                "calibration": calibration,
                "covariance_operator": sigma_operator,
                "ranking": summarize_role(
                    evaluation, transformed, calibration, nodes, "ranking", SEED + 1, ranking_coarse
                ),
                "pruning": summarize_role(
                    evaluation, transformed, calibration, nodes, "pruning", SEED + 2, pruning_coarse
                ),
            }
        )
    return {
        "dataset": name,
        "kind": kind,
        "dimension": dimension,
        "source_total": source_total,
        "fit_size": len(fit),
        "evaluation_size": len(evaluation),
        "quantizers": configurations,
    }


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = analyze(name)
        results.append(result)
        for configuration in result["quantizers"]:
            print(
                f"  {configuration['quantizer']:<13} "
                f"rank={configuration['ranking']['component_cauchy_looseness']:.2f}×/"
                f"{configuration['ranking']['replacement']['looseness']:.2f}×, "
                f"prune={configuration['pruning']['component_cauchy_looseness']:.2f}×/"
                f"{configuration['pruning']['replacement']['looseness']:.2f}×"
            )
    output = {
        "probe": 38,
        "seed": SEED,
        "n_fit": N_FIT,
        "n_evaluation": N_EVALUATION,
        "n_role_samples": N_ROLE_SAMPLES,
        "results": results,
    }
    path = RESULTS_DIR / "probe38_standard_vamana_p5_component_variance_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
