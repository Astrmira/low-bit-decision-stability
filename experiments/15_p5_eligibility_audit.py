#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""15 - Probe 32：P5 eligibility 与 remainder audit。"""

import json
import math
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.stats import kurtosis, skew, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


pruning = import_module("10_pruning_replay")

DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_SAMPLE = 4_000
N_CALIBRATION = 80
N_EVALUATION = 160
CANDIDATE_K = 32
PAIR_SAMPLES = 12_000
ALPHA = 1.2
SEED = 20260823
H_RATIOS = [0.005, 0.01, 0.02, 0.05, 0.10]
ETA_VALUES = [0.01, 0.05, 0.10]


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_artifact(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 32 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED)
    indices = rng.choice(total, min(N_SAMPLE, total), replace=False)
    vecs = np.asarray(mapped[indices], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    normalized = vecs / np.maximum(norms[:, None], 1e-12)
    return normalized, norms, total


def shell_audit(norms):
    identifiable = bool(np.mean(np.abs(norms - 1) < 1e-5) < 0.999)
    if not identifiable:
        return {
            "identifiable": False,
            "reason": "source artifacts are already unit-normalized; pre-normalization norms are unavailable",
        }
    center = float(np.mean(norms))
    relative = norms / center - 1
    return {
        "identifiable": True,
        "center": center,
        "std": float(np.std(norms, ddof=1)),
        "coefficient_of_variation": float(np.std(norms, ddof=1) / center),
        "relative_deviation_quantiles": [
            float(value) for value in np.quantile(np.abs(relative), [0.90, 0.95, 0.99])
        ],
        "eta_tail": [
            {"eta": eta, "empirical_tail": float(np.mean(np.abs(relative) > eta))}
            for eta in ETA_VALUES
        ],
        "status": "observable radial diagnostic, not a theorem-certified shell bound",
    }


def rotate(vecs):
    rng = np.random.default_rng(SEED + vecs.shape[1])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), vecs.shape[1])
    output = dct(vecs * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return output


def encode(vecs, threshold):
    signs = vecs > 0
    strongs = np.abs(vecs) > threshold
    return signs, strongs


def raw_distance(left, right, encoded):
    signs, strongs = encoded
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    agreements = signs[left] == signs[right]
    signed = 2 * agreements.astype(np.int16) - 1
    weights = (1 + strongs[left].astype(np.int16)) * (
        1 + strongs[right].astype(np.int16)
    )
    score = (signed * weights).sum(axis=1, dtype=np.int64)
    return (4 * signs.shape[1] - score).astype(np.float64)


def fit_affine(raw, exact):
    design = np.column_stack([np.ones(len(raw)), raw])
    beta, _, _, _ = np.linalg.lstsq(design, exact, rcond=None)
    fitted = design @ beta
    residual = fitted - exact
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "residual_mean": float(residual.mean()),
        "residual_scale": float(residual.std(ddof=1)),
        "spearman": safe_spearman(raw, exact),
    }


def calibrated(raw, calibration):
    return calibration["intercept"] + calibration["slope"] * raw


def pair_indices(count, size, seed):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, size, count)
    right = rng.integers(0, size, count)
    equal = left == right
    right[equal] = (right[equal] + 1) % size
    return left, right


def threshold_audit(vecs, shared, oracle):
    thresholds = np.abs(vecs).mean(axis=1)
    population = float(thresholds.mean())
    scale_tests = np.array([0.25, 0.5, 2.0, 4.0], dtype=np.float32)
    scaled = vecs[: len(scale_tests)] * scale_tests[:, None]
    scaled_shared = encode(scaled, np.abs(scaled).mean(axis=1, keepdims=True))
    scale_mismatch = int(
        np.count_nonzero(shared[0][: len(scale_tests)] != scaled_shared[0])
        + np.count_nonzero(shared[1][: len(scale_tests)] != scaled_shared[1])
    )
    flip_fraction = float(np.mean(shared[1] != oracle[1]))
    covariance = np.cov(vecs, rowvar=False, ddof=1)
    operator = float(np.linalg.eigvalsh(covariance)[-1])
    rows = []
    for ratio in H_RATIOS:
        h = ratio * population
        threshold_bad = np.abs(thresholds - population) > h
        boundary = np.abs(np.abs(vecs) - population) <= h
        flips = shared[1] != oracle[1]
        coverage = bool(np.all(flips <= (threshold_bad[:, None] | boundary)))
        empirical_tail = float(threshold_bad.mean())
        envelope = min(1.0, float(2 * np.exp(-vecs.shape[1] * h * h / (2 * operator))))
        coordinate_occupancy = boundary.mean(axis=0)
        rows.append(
            {
                "h_ratio": ratio,
                "h": h,
                "empirical_threshold_tail": empirical_tail,
                "plugin_gaussian_envelope": envelope,
                "tail_to_envelope_ratio": empirical_tail / max(envelope, 1e-300),
                "boundary_fraction": float(boundary.mean()),
                "boundary_coordinate_p90": float(np.quantile(coordinate_occupancy, 0.90)),
                "boundary_coordinate_p99": float(np.quantile(coordinate_occupancy, 0.99)),
                "boundary_coordinate_max": float(coordinate_occupancy.max()),
                "flip_inclusion_holds": coverage,
            }
        )
    return {
        "threshold_mean": population,
        "threshold_std": float(thresholds.std(ddof=1)),
        "threshold_cv": float(thresholds.std(ddof=1) / population),
        "threshold_quantiles": [float(x) for x in np.quantile(thresholds, [0.01, 0.5, 0.99])],
        "magnitude_flip_fraction": flip_fraction,
        "positive_scale_code_mismatch": scale_mismatch,
        "sample_covariance_operator_norm": operator,
        "h_audit": rows,
        "status": "normalized-artifact plug-in diagnostic",
    }


def residual_samples(vecs, shared, oracle, calibration):
    left, right = pair_indices(PAIR_SAMPLES, len(vecs), SEED + 320)
    exact = 1.0 - np.sum(vecs[left] * vecs[right], axis=1)
    shared_raw = raw_distance(left, right, shared)
    oracle_raw = raw_distance(left, right, oracle)
    actual = calibrated(shared_raw, calibration) - exact
    oracle_residual = calibrated(oracle_raw, calibration) - exact
    threshold = actual - oracle_residual
    return left, right, actual, oracle_residual, threshold


def role_samples(vecs, shared, oracle, calibration):
    rng = np.random.default_rng(SEED + 321)
    nodes = rng.integers(0, len(vecs), (PAIR_SAMPLES, 3))
    collision = (nodes[:, 0] == nodes[:, 1]) | (nodes[:, 1] == nodes[:, 2]) | (
        nodes[:, 0] == nodes[:, 2]
    )
    nodes[collision, 2] = (nodes[collision, 2] + 1) % len(vecs)
    q, x, y = nodes.T
    exact_qx = 1.0 - np.sum(vecs[q] * vecs[x], axis=1)
    exact_qy = 1.0 - np.sum(vecs[q] * vecs[y], axis=1)
    exact_xy = 1.0 - np.sum(vecs[x] * vecs[y], axis=1)

    def residual(left, right, exact, encoding):
        return calibrated(raw_distance(left, right, encoding), calibration) - exact

    actual_qx = residual(q, x, exact_qx, shared)
    actual_qy = residual(q, y, exact_qy, shared)
    actual_xy = residual(x, y, exact_xy, shared)
    oracle_qx = residual(q, x, exact_qx, oracle)
    oracle_qy = residual(q, y, exact_qy, oracle)
    oracle_xy = residual(x, y, exact_xy, oracle)
    return {
        "ranking_actual": actual_qy - actual_qx,
        "ranking_oracle": oracle_qy - oracle_qx,
        "ranking_threshold": (actual_qy - actual_qx) - (oracle_qy - oracle_qx),
        "pruning_actual": actual_qx - ALPHA * actual_xy,
        "pruning_oracle": oracle_qx - ALPHA * oracle_xy,
        "pruning_threshold": (actual_qx - ALPHA * actual_xy)
        - (oracle_qx - ALPHA * oracle_xy),
        "edge_first": oracle_qx,
        "edge_second_shared": oracle_qy,
        "edge_second_chain": oracle_xy,
    }


def cumulant_audit(values):
    centered = values - values.mean()
    variance = float(np.var(centered, ddof=0))
    if variance <= 1e-30:
        return {"variance": variance, "degenerate": True}
    moments = {order: float(np.mean(centered**order)) for order in range(2, 7)}
    cumulants = {
        2: moments[2],
        3: moments[3],
        4: moments[4] - 3 * moments[2] ** 2,
        5: moments[5] - 10 * moments[3] * moments[2],
        6: moments[6]
        - 15 * moments[4] * moments[2]
        - 10 * moments[3] ** 2
        + 30 * moments[2] ** 3,
    }
    required_b = []
    for order in range(3, 7):
        denominator = math.factorial(order) * variance / 2
        required_b.append((abs(cumulants[order]) / max(denominator, 1e-300)) ** (1 / (order - 2)))
    b = float(max(required_b))
    standardized = centered / np.sqrt(variance)
    lambdas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    empirical_log_mgf = []
    envelope = []
    compatible = True
    for value in lambdas:
        shifted = value * centered
        maximum = shifted.max()
        log_mgf = float(maximum + np.log(np.exp(shifted - maximum).mean()))
        bound = float(value * value * variance / (2 * max(1 - b * value, 1e-12)))
        empirical_log_mgf.append(log_mgf)
        envelope.append(bound)
        compatible = compatible and (b * value >= 1 or log_mgf <= bound + 1e-12)
    return {
        "variance": variance,
        "skewness": float(skew(standardized, bias=False)),
        "excess_kurtosis": float(kurtosis(standardized, fisher=True, bias=False)),
        "cumulants_2_to_6": {str(order): cumulants[order] for order in range(2, 7)},
        "empirical_required_b": b,
        "b_over_sd": b / np.sqrt(variance),
        "mgf_lambdas": lambdas.tolist(),
        "empirical_log_mgf": empirical_log_mgf,
        "candidate_envelope": envelope,
        "empirical_d3_compatible": bool(compatible),
        "status": "falsification diagnostic only",
    }


def role_summary(values, threshold_values, coarse_variance):
    variance = float(np.var(values, ddof=1))
    threshold_variance = float(np.var(threshold_values, ddof=1))
    return {
        "mean": float(np.mean(values)),
        "variance": variance,
        "threshold_remainder_mean": float(np.mean(threshold_values)),
        "threshold_remainder_variance": threshold_variance,
        "threshold_to_full_variance_ratio": threshold_variance / max(variance, 1e-300),
        "absolute_quantiles": [float(x) for x in np.quantile(np.abs(threshold_values), [0.90, 0.95, 0.99])],
        "coarse_variance_proxy": coarse_variance,
        "coarse_proxy_looseness": coarse_variance / max(variance, 1e-300),
        "d3": cumulant_audit(values),
    }


def analyze_quantizer(vecs, transformed, quantizer, shell):
    thresholds = np.abs(transformed).mean(axis=1, keepdims=True)
    population = float(thresholds.mean())
    shared = encode(transformed, thresholds)
    oracle = encode(transformed, population)
    calibration_left, calibration_right = pair_indices(PAIR_SAMPLES, len(vecs), SEED + 300)
    exact_calibration = 1.0 - np.sum(
        vecs[calibration_left] * vecs[calibration_right], axis=1
    )
    calibration = fit_affine(
        raw_distance(calibration_left, calibration_right, shared), exact_calibration
    )
    left, right, actual, oracle_residual, threshold_residual = residual_samples(
        vecs, shared, oracle, calibration
    )
    roles = role_samples(vecs, shared, oracle, calibration)
    mean = transformed.mean(axis=0)
    covariance = np.cov(transformed, rowvar=False, ddof=1)
    sigma_frobenius_sq = float(np.sum(covariance * covariance))
    sigma_operator = float(np.linalg.eigvalsh(covariance)[-1])
    mu_sigma_mu = float(mean @ covariance @ mean)
    beta = calibration["slope"]

    def coarse(sum_a_sq, sum_d_sq, matrix_op, a1):
        vq = sigma_frobenius_sq * sum_a_sq + mu_sigma_mu * sum_d_sq
        ls = 8 * vecs.shape[1] * beta * a1
        return 2 * vq + ls * ls / 2, 4 * matrix_op * sigma_operator

    ranking_coarse, ranking_b = coarse(2.0, 2.0, np.sqrt(2) / 2, 2.0)
    pruning_sum_d = ALPHA**2 + (ALPHA - 1) ** 2 + 1
    pruning_coarse, pruning_b = coarse(
        ALPHA**2 + 1, pruning_sum_d, np.sqrt(ALPHA**2 + 1) / 2, ALPHA + 1
    )
    edge_variance = float(np.var(roles["edge_first"], ddof=1))
    shared_covariance = float(
        np.cov(roles["edge_first"], roles["edge_second_shared"], ddof=1)[0, 1]
    )
    chain_covariance = float(
        np.cov(roles["edge_first"], roles["edge_second_chain"], ddof=1)[0, 1]
    )
    kappa1 = 0.5 * (shared_covariance + chain_covariance)
    kappa2 = edge_variance - 2 * kappa1
    predicted_ranking = 2 * kappa1 + 2 * kappa2
    predicted_pruning = (
        (2 * ALPHA**2 - 2 * ALPHA + 2) * kappa1
        + (ALPHA**2 + 1) * kappa2
    )
    threshold = threshold_audit(transformed, shared, oracle)
    eligible = (
        calibration["slope"] > 0
        and threshold["positive_scale_code_mismatch"] == 0
        and all(row["flip_inclusion_holds"] for row in threshold["h_audit"])
        and cumulant_audit(roles["ranking_oracle"])["empirical_d3_compatible"]
        and cumulant_audit(roles["pruning_oracle"])["empirical_d3_compatible"]
    )
    return {
        "quantizer": quantizer,
        "calibration": calibration,
        "threshold": threshold,
        "shell": shell,
        "single_edge": {
            "actual_variance": float(np.var(actual, ddof=1)),
            "oracle_variance": float(np.var(oracle_residual, ddof=1)),
            "threshold_remainder_variance": float(np.var(threshold_residual, ddof=1)),
            "threshold_to_actual_variance_ratio": float(
                np.var(threshold_residual, ddof=1) / max(np.var(actual, ddof=1), 1e-300)
            ),
        },
        "oracle_covariance": {
            "edge_variance": edge_variance,
            "shared_first_covariance": shared_covariance,
            "chain_covariance": chain_covariance,
            "kappa1_estimate": kappa1,
            "kappa2_estimate": kappa2,
            "ranking_predicted_variance": predicted_ranking,
            "ranking_empirical_variance": float(np.var(roles["ranking_oracle"], ddof=1)),
            "pruning_predicted_variance": predicted_pruning,
            "pruning_empirical_variance": float(np.var(roles["pruning_oracle"], ddof=1)),
        },
        "ranking": role_summary(
            roles["ranking_oracle"], roles["ranking_threshold"], ranking_coarse
        )
        | {"coarse_scale_proxy": ranking_b},
        "pruning": role_summary(
            roles["pruning_oracle"], roles["pruning_threshold"], pruning_coarse
        )
        | {"coarse_scale_proxy": pruning_b},
        "empirical_eligibility": {
            "positive_slope": calibration["slope"] > 0,
            "threshold_identity": threshold["positive_scale_code_mismatch"] == 0,
            "boundary_inclusion": all(
                row["flip_inclusion_holds"] for row in threshold["h_audit"]
            ),
            "ranking_d3_compatible": cumulant_audit(roles["ranking_oracle"])[
                "empirical_d3_compatible"
            ],
            "pruning_d3_compatible": cumulant_audit(roles["pruning_oracle"])[
                "empirical_d3_compatible"
            ],
            "shell_identifiable": shell["identifiable"],
            "overall": (
                "empirically-eligible"
                if eligible and shell["identifiable"]
                else "partial-eligible" if eligible else "abstain"
            ),
            "status": "empirical falsification diagnostic, not theorem-certified eligibility",
        },
    }


def run_dataset(name):
    vecs, original_norms, total = load_artifact(name)
    transformed = {"2bit": vecs, "rotated_2bit": rotate(vecs)}
    shell = shell_audit(original_norms)
    return {
        "dataset": name,
        "dimension": int(vecs.shape[1]),
        "source_total": int(total),
        "artifact_norm_audit": {
            "mean": float(original_norms.mean()),
            "std": float(original_norms.std(ddof=1)),
            "min": float(original_norms.min()),
            "max": float(original_norms.max()),
            "fraction_within_1e_5_of_one": float(np.mean(np.abs(original_norms - 1) < 1e-5)),
            "pre_normalization_shell_identifiable": shell["identifiable"],
        },
        "quantizers": {
            quantizer: analyze_quantizer(vecs, values, quantizer, shell)
            for quantizer, values in transformed.items()
        },
    }


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            print(
                f"  {quantizer:<13} flip={metrics['threshold']['magnitude_flip_fraction']:.4f}, "
                f"rank_loose={metrics['ranking']['coarse_proxy_looseness']:.1f}, "
                f"prune_loose={metrics['pruning']['coarse_proxy_looseness']:.1f}, "
                f"eligibility={metrics['empirical_eligibility']['overall']}"
            )
    output = {
        "probe": 32,
        "seed": SEED,
        "n_sample": N_SAMPLE,
        "pair_samples": PAIR_SAMPLES,
        "h_ratios": H_RATIOS,
        "eta_values": ETA_VALUES,
        "results": results,
    }
    path = RESULTS_DIR / "probe32_standard_vamana_p5_eligibility_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


if __name__ == "__main__":
    main()
