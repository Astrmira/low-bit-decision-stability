#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""22 - Probe 39：P5 dimension、correlation 与 global contamination stress。"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, kstest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR


DIMENSIONS = [64, 128, 256, 512, 1024, 2048]
REGIMES = ["clean", "equicorrelated", "block_correlated", "global_shock"]
N_SAMPLE = 3_000
N_PAIRS = 20_000
N_ROLES = 20_000
K_VALUES = [2, 4, 8]
K_TRIALS = 12
ALPHA = 1.2
SEED = 20260829


def generate(dimension, regime, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(N_SAMPLE, dimension)).astype(np.float32)
    metadata = {}
    if regime == "equicorrelated":
        rho = dimension**-0.5
        common = rng.normal(size=(N_SAMPLE, 1)).astype(np.float32)
        values = np.sqrt(1 - rho) * values + np.sqrt(rho) * common
        metadata["rho"] = rho
    elif regime == "block_correlated":
        rho = 0.25
        block_size = min(32, dimension)
        for start in range(0, dimension, block_size):
            stop = min(start + block_size, dimension)
            common = rng.normal(size=(N_SAMPLE, 1)).astype(np.float32)
            values[:, start:stop] = (
                np.sqrt(1 - rho) * values[:, start:stop] + np.sqrt(rho) * common
            )
        metadata |= {"rho": rho, "block_size": block_size}
    elif regime == "global_shock":
        probability = dimension**-0.5
        amplitude = 2.5
        shock = rng.random(N_SAMPLE) < probability
        pattern = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), dimension)
        values[shock] += amplitude * pattern
        metadata |= {
            "shock_probability": probability,
            "shock_amplitude": amplitude,
            "realized_shock_fraction": float(np.mean(shock)),
        }
    norms = np.linalg.norm(values, axis=1)
    normalized = values / np.maximum(norms[:, None], 1e-12)
    return values, normalized, norms, metadata


def fixed_k_audit(values, seed):
    rng = np.random.default_rng(seed)
    output = {}
    for k in K_VALUES:
        statistics = []
        rejections = []
        for _ in range(K_TRIALS):
            coordinates = rng.choice(values.shape[1], k, replace=False)
            sample = values[:, coordinates].astype(np.float64)
            centered = sample - sample.mean(axis=0)
            covariance = np.cov(centered, rowvar=False, ddof=1)
            precision = np.linalg.pinv(covariance, rcond=1e-10)
            radius = np.einsum("ni,ij,nj->n", centered, precision, centered)
            test = kstest(radius, chi2(df=k).cdf)
            statistics.append(float(test.statistic))
            rejections.append(test.pvalue < 0.01)
        output[str(k)] = {
            "median_radius_ks": float(np.median(statistics)),
            "rejection_rate_0_01": float(np.mean(rejections)),
        }
    return output


def spectrum_audit(values, seed):
    centered = values.astype(np.float64) - values.mean(axis=0, dtype=np.float64)
    trace = float(np.sum(centered**2) / (len(values) - 1))
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=values.shape[1])
    vector /= np.linalg.norm(vector)
    for _ in range(30):
        projected = centered.T @ (centered @ vector) / (len(values) - 1)
        norm = np.linalg.norm(projected)
        if norm <= 1e-15:
            break
        vector = projected / norm
    operator = float(np.sum((centered @ vector) ** 2) / (len(values) - 1))
    return {
        "trace": trace,
        "operator_power_estimate": operator,
        "effective_rank_estimate": trace / max(operator, 1e-300),
        "top_eigenvalue_share_estimate": operator / max(trace, 1e-300),
    }


def encode(values, threshold):
    return values > 0, np.abs(values) > threshold


def raw_distance(left, right, encoded):
    signs, strongs = encoded
    agreement = signs[left] == signs[right]
    signed = 2 * agreement.astype(np.int16) - 1
    weights = (1 + strongs[left].astype(np.int16)) * (1 + strongs[right].astype(np.int16))
    score = (signed * weights).sum(axis=1, dtype=np.int64)
    return (4 * signs.shape[1] - score).astype(np.float64)


def random_pairs(size, count, seed):
    rng = np.random.default_rng(seed)
    output = rng.integers(0, size, (count, 2))
    same = output[:, 0] == output[:, 1]
    output[same, 1] = (output[same, 1] + 1) % size
    return output[:, 0], output[:, 1]


def calibrate(normalized, shared, seed):
    left, right = random_pairs(len(normalized), N_PAIRS, seed)
    exact = 1.0 - np.sum(normalized[left] * normalized[right], axis=1)
    raw = raw_distance(left, right, shared)
    design = np.column_stack([np.ones(len(raw)), raw])
    beta, _, _, _ = np.linalg.lstsq(design, exact, rcond=None)
    return float(beta[0]), float(beta[1])


def cumulants(values, maximum=10):
    centered = values.astype(np.float64) - np.mean(values)
    moments = {0: 1.0, 1: 0.0}
    for order in range(2, maximum + 1):
        moments[order] = float(np.mean(centered**order))
    output = {1: 0.0}
    for order in range(2, maximum + 1):
        correction = sum(
            math.comb(order - 1, index - 1) * output[index] * moments[order - index]
            for index in range(1, order)
        )
        output[order] = moments[order] - correction
    variance = output[2]
    required = []
    for order in range(3, maximum + 1):
        ratio = 2 * abs(output[order]) / max(math.factorial(order) * variance, 1e-300)
        required.append(ratio ** (1 / (order - 2)))
    return {
        "cumulants_2_to_10": {str(order): output[order] for order in range(2, maximum + 1)},
        "required_b_2_to_10": float(max(required)),
        "b_over_sd": float(max(required) / max(np.sqrt(variance), 1e-300)),
        "standardized_cumulants_3_to_10": {
            str(order): float(output[order] / max(variance ** (order / 2), 1e-300))
            for order in range(3, maximum + 1)
        },
    }


def role_audit(normalized, shared, oracle, calibration, seed):
    rng = np.random.default_rng(seed)
    nodes = rng.integers(0, len(normalized), (N_ROLES, 3))
    collision = (nodes[:, 0] == nodes[:, 1]) | (nodes[:, 0] == nodes[:, 2]) | (
        nodes[:, 1] == nodes[:, 2]
    )
    while np.any(collision):
        nodes[collision] = rng.integers(0, len(normalized), (int(np.sum(collision)), 3))
        collision = (nodes[:, 0] == nodes[:, 1]) | (nodes[:, 0] == nodes[:, 2]) | (
            nodes[:, 1] == nodes[:, 2]
        )
    q, x, y = nodes.T
    intercept, slope = calibration

    def residual(left, right, encoding):
        exact = 1.0 - np.sum(normalized[left] * normalized[right], axis=1)
        return intercept + slope * raw_distance(left, right, encoding) - exact

    shared_qx = residual(q, x, shared)
    shared_qy = residual(q, y, shared)
    shared_xy = residual(x, y, shared)
    oracle_qx = residual(q, x, oracle)
    oracle_qy = residual(q, y, oracle)
    oracle_xy = residual(x, y, oracle)
    output = {}
    for role, actual, core in [
        ("ranking", shared_qy - shared_qx, oracle_qy - oracle_qx),
        ("pruning", shared_qx - ALPHA * shared_xy, oracle_qx - ALPHA * oracle_xy),
    ]:
        remainder = actual - core
        output[role] = {
            "actual_variance": float(np.var(actual, ddof=1)),
            "oracle_variance": float(np.var(core, ddof=1)),
            "threshold_remainder_variance": float(np.var(remainder, ddof=1)),
            "threshold_to_actual_variance_ratio": float(
                np.var(remainder, ddof=1) / max(np.var(actual, ddof=1), 1e-300)
            ),
            "oracle_cumulants": cumulants(core),
            "actual_cumulants": cumulants(actual),
        }
    return output


def analyze(dimension, regime):
    raw, normalized, norms, metadata = generate(
        dimension, regime, SEED + 1009 * dimension + REGIMES.index(regime)
    )
    threshold_per_vector = np.abs(normalized).mean(axis=1, keepdims=True)
    population = float(threshold_per_vector.mean())
    shared = encode(normalized, threshold_per_vector)
    oracle = encode(normalized, population)
    calibration = calibrate(normalized, shared, SEED + dimension)
    role = role_audit(normalized, shared, oracle, calibration, SEED + dimension * 13)
    norm_relative = norms / np.mean(norms) - 1
    return {
        "dimension": dimension,
        "regime": regime,
        "metadata": metadata,
        "fixed_k_gaussianity": fixed_k_audit(raw, SEED + dimension * 17),
        "spectrum": spectrum_audit(raw, SEED + dimension * 19),
        "shell": {
            "coefficient_of_variation": float(np.std(norms, ddof=1) / np.mean(norms)),
            "relative_deviation_p99": float(np.quantile(np.abs(norm_relative), 0.99)),
        },
        "threshold": {
            "mean": population,
            "cv": float(np.std(threshold_per_vector, ddof=1) / population),
            "magnitude_flip_fraction": float(np.mean(shared[1] != oracle[1])),
        },
        "calibration": {"intercept": calibration[0], "slope": calibration[1]},
        "roles": role,
    }


def main():
    results = []
    for dimension in DIMENSIONS:
        print(f"\n{'=' * 72}\n维度：{dimension}\n{'=' * 72}")
        for regime in REGIMES:
            result = analyze(dimension, regime)
            results.append(result)
            rank = result["roles"]["ranking"]
            print(
                f"  {regime:<18} k8={result['fixed_k_gaussianity']['8']['median_radius_ks']:.3f}, "
                f"shell={result['shell']['coefficient_of_variation']:.3f}, "
                f"flip={result['threshold']['magnitude_flip_fraction']:.3f}, "
                f"b/sd={rank['oracle_cumulants']['b_over_sd']:.3f}"
            )
    output = {
        "probe": 39,
        "seed": SEED,
        "n_sample": N_SAMPLE,
        "n_roles": N_ROLES,
        "dimensions": DIMENSIONS,
        "regimes": REGIMES,
        "results": results,
        "status": "dimension and contamination falsification diagnostic, not a D3 proof",
    }
    path = RESULTS_DIR / "probe39_standard_vamana_p5_dimension_contamination_stress.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
