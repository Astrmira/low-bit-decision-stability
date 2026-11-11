#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""23 - Probe 40：P8-B conditional MGF 与 stable truncation 最终验证。"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = ["cohere", "minilm", "bge_m3", "sift128", "gist960", "random"]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_FIT = 900
N_REFERENCE = 1_200
N_CALIBRATION = 1_200
N_VALIDATION = 1_200
N_FIT_PAIRS = 20_000
N_GROUPS = 160
N_SECOND = 16
N_THIRD = 12
N_INTEGRATION_SECOND = 24
N_INTEGRATION_THIRD = 24
ALPHA = 1.2
LAMBDA_GRID = np.array([0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5])
TRUNCATION_QUANTILES = [0.95, 0.975, 0.99]
SEED = 20260830


def load_pools(name):
    prefix, dimension, kind, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 40 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    count = min(N_FIT + N_REFERENCE + N_CALIBRATION + N_VALIDATION, total)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED + dimension)
    indices = rng.choice(total, count, replace=False)
    values = np.asarray(mapped[indices], dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    cuts = np.cumsum([N_FIT, N_REFERENCE, N_CALIBRATION])
    fit, reference, calibration, validation = np.split(values, cuts)
    return fit, reference, calibration, validation, kind, total


def rotate(values, signs):
    output = dct(values * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return output


def encode(values, threshold):
    return np.where(values >= 0, 1, -1).astype(np.int8) * (
        1 + (np.abs(values) > threshold).astype(np.int8)
    )


def edge(left_exact, right_exact, left_code, right_code, beta):
    score = np.sum(left_code.astype(np.int16) * right_code.astype(np.int16), axis=1)
    raw = 4 * left_code.shape[1] - score
    exact = 1.0 - np.sum(left_exact * right_exact, axis=1)
    return beta[0] + beta[1] * raw - exact


def fit_calibration(exact, code, seed):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, len(exact), (N_FIT_PAIRS, 2))
    equal = pairs[:, 0] == pairs[:, 1]
    pairs[equal, 1] = (pairs[equal, 1] + 1) % len(exact)
    left, right = pairs.T
    score = np.sum(code[left].astype(np.int16) * code[right].astype(np.int16), axis=1)
    raw = 4 * exact.shape[1] - score
    target = 1.0 - np.sum(exact[left] * exact[right], axis=1)
    design = np.column_stack([np.ones(len(raw)), raw])
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return float(beta[0]), float(beta[1])


def functional(left_exact, middle_exact, right_exact, left_code, middle_code, right_code, beta, role):
    e12 = edge(left_exact, middle_exact, left_code, middle_code, beta)
    e13 = edge(left_exact, right_exact, left_code, right_code, beta)
    e23 = edge(middle_exact, right_exact, middle_code, right_code, beta)
    return e13 - e12 if role == "ranking" else e12 - ALPHA * e23


def clipped_functional(values, center, cutoff):
    if cutoff is None:
        return values
    return center + np.clip(values - center, -cutoff, cutoff)


def sample_doob(pool_exact, pool_code, reference_exact, reference_code, beta, role, seed, center=0.0, cutoff=None):
    rng = np.random.default_rng(seed)
    size = len(pool_exact)
    reference_size = len(reference_exact)
    first = rng.integers(0, size, N_GROUPS)
    second = rng.integers(0, size, (N_GROUPS, N_SECOND))
    third = rng.integers(0, size, (N_GROUPS, N_SECOND, N_THIRD))
    g1 = np.repeat(first, N_SECOND * N_THIRD)
    g2 = np.repeat(second[:, :, None], N_THIRD, axis=2).reshape(-1)
    g3 = third.reshape(-1)
    values = functional(
        pool_exact[g1], pool_exact[g2], pool_exact[g3],
        pool_code[g1], pool_code[g2], pool_code[g3], beta, role,
    )
    values = clipped_functional(values, center, cutoff)
    tensor = values.reshape(N_GROUPS, N_SECOND, N_THIRD)

    prefix_first = np.repeat(first, N_SECOND)
    prefix_second = second.reshape(-1)
    integration_third = rng.integers(
        0, reference_size, (len(prefix_first), N_INTEGRATION_THIRD)
    )
    m2_first = np.repeat(prefix_first, N_INTEGRATION_THIRD)
    m2_second = np.repeat(prefix_second, N_INTEGRATION_THIRD)
    m2_third = integration_third.reshape(-1)
    conditional_second = functional(
        pool_exact[m2_first], pool_exact[m2_second], reference_exact[m2_third],
        pool_code[m2_first], pool_code[m2_second], reference_code[m2_third], beta, role,
    )
    conditional_second = clipped_functional(
        conditional_second, center, cutoff
    ).reshape(N_GROUPS, N_SECOND, N_INTEGRATION_THIRD).mean(axis=2)

    integration_second = rng.integers(
        0, reference_size, (N_GROUPS, N_INTEGRATION_SECOND)
    )
    integration_third_first = rng.integers(
        0,
        reference_size,
        (N_GROUPS, N_INTEGRATION_SECOND, N_INTEGRATION_THIRD),
    )
    m1_first = np.repeat(first, N_INTEGRATION_SECOND * N_INTEGRATION_THIRD)
    m1_second = np.repeat(
        integration_second[:, :, None], N_INTEGRATION_THIRD, axis=2
    ).reshape(-1)
    m1_third = integration_third_first.reshape(-1)
    conditional_first = functional(
        pool_exact[m1_first], reference_exact[m1_second], reference_exact[m1_third],
        pool_code[m1_first], reference_code[m1_second], reference_code[m1_third], beta, role,
    )
    conditional_first = clipped_functional(
        conditional_first, center, cutoff
    ).reshape(N_GROUPS, N_INTEGRATION_SECOND, N_INTEGRATION_THIRD).mean(axis=(1, 2))

    d1 = conditional_first - conditional_first.mean()
    d2 = conditional_second - conditional_first[:, None]
    d3 = tensor - conditional_second[:, :, None]
    increments = [d1, d2, d3]
    return values, increments


def required_scale(values, variance):
    centered = values - np.mean(values)
    sd = max(float(np.std(centered, ddof=1)), 1e-12)
    standardized = centered / sd
    required = 0.0
    compatible = True
    rows = []
    for lam in LAMBDA_GRID:
        for sign in [-1.0, 1.0]:
            log_mgf = float(np.log(np.mean(np.exp(np.clip(sign * lam * standardized, -40, 40)))))
            denominator = max(lam * log_mgf, 1e-12)
            candidate = max(0.0, (2 * log_mgf - lam**2) / (2 * denominator))
            required = max(required, candidate)
            rows.append({"lambda": float(sign * lam), "log_mgf": log_mgf})
            compatible &= np.isfinite(log_mgf)
    return {
        "variance": variance,
        "sd": sd,
        "required_b_over_sd": float(required),
        "mgf_compatible": bool(compatible),
        "mgf_grid": rows,
    }


def conditional_scale(increment):
    pooled = required_scale(increment.reshape(-1), float(np.var(increment, ddof=1)))
    if increment.ndim == 1:
        conditional = np.array([pooled["required_b_over_sd"]])
    else:
        conditional = np.array(
            [
                required_scale(row, float(np.var(row, ddof=1)))["required_b_over_sd"]
                for row in increment.reshape(-1, increment.shape[-1])
            ]
        )
    pooled["conditional_required_b_over_sd_p95"] = float(np.quantile(conditional, 0.95))
    pooled["conditional_required_b_over_sd_max"] = float(np.max(conditional))
    pooled["conditional_groups"] = int(len(conditional))
    return pooled


def analyze_role(
    reference_exact,
    reference_code,
    calibration_exact,
    calibration_code,
    validation_exact,
    validation_code,
    beta,
    role,
    seed,
):
    calibration_values, calibration_increments = sample_doob(
        calibration_exact, calibration_code, reference_exact, reference_code, beta, role, seed
    )
    validation_values, validation_increments = sample_doob(
        validation_exact, validation_code, reference_exact, reference_code, beta, role, seed + 10_000
    )
    calibration_centered = calibration_values - calibration_values.mean()
    validation_centered = validation_values - validation_values.mean()
    variance = float(np.var(calibration_centered, ddof=1))
    validation_variance = float(np.var(validation_centered, ddof=1))
    increments = []
    b_star = 0.0
    variance_sum = 0.0
    for calibration_increment, validation_increment in zip(calibration_increments, validation_increments):
        increment_variance = float(np.var(calibration_increment, ddof=1))
        diagnostic = conditional_scale(calibration_increment)
        validation_diagnostic = conditional_scale(validation_increment)
        diagnostic["validation_required_b_over_sd"] = validation_diagnostic["required_b_over_sd"]
        diagnostic["scale_transfer_ratio"] = validation_diagnostic["required_b_over_sd"] / max(
            diagnostic["required_b_over_sd"], 1e-12
        )
        increments.append(diagnostic)
        variance_sum += increment_variance
        b_star = max(
            b_star,
            diagnostic["conditional_required_b_over_sd_p95"] * diagnostic["sd"],
        )
    sd = max(np.sqrt(variance), 1e-12)
    truncations = []
    calibration_abs = np.abs(calibration_centered)
    for quantile in TRUNCATION_QUANTILES:
        cutoff = float(np.quantile(calibration_abs, quantile))
        exceptional_calibration = float(np.mean(calibration_abs > cutoff))
        validation_about_calibration = validation_values - calibration_values.mean()
        exceptional_validation = float(np.mean(np.abs(validation_about_calibration) > cutoff))
        truncated_calibration_values, truncated_calibration_increments = sample_doob(
            calibration_exact,
            calibration_code,
            reference_exact,
            reference_code,
            beta,
            role,
            seed,
            float(calibration_values.mean()),
            cutoff,
        )
        truncated_validation_values, _ = sample_doob(
            validation_exact,
            validation_code,
            reference_exact,
            reference_code,
            beta,
            role,
            seed + 10_000,
            float(calibration_values.mean()),
            cutoff,
        )
        truncated_variance = float(np.var(truncated_calibration_values, ddof=1))
        truncated_variance_sum = sum(
            float(np.var(increment, ddof=1)) for increment in truncated_calibration_increments
        )
        truncated_b_star = max(
            conditional_scale(increment)["conditional_required_b_over_sd_p95"]
            * max(float(np.std(increment, ddof=1)), 1e-12)
            for increment in truncated_calibration_increments
        )
        mean_shift = abs(
            float(calibration_values.mean() - truncated_calibration_values.mean())
        )
        validation_mean_shift = abs(
            float(validation_values.mean() - truncated_validation_values.mean())
        )
        rows = []
        for multiple in [2.0, 2.5, 3.0, 3.5, 4.0]:
            threshold = multiple * sd
            effective_threshold = max(threshold - mean_shift, 0.0)
            bound = min(
                1.0,
                float(
                    2
                    * np.exp(
                        -(effective_threshold**2)
                        / (
                            2
                            * (
                                max(truncated_variance_sum, 1e-12)
                                + max(truncated_b_star, 1e-12) * effective_threshold
                            )
                        )
                    )
                    + exceptional_calibration
                ),
            )
            empirical = float(np.mean(np.abs(validation_centered) >= threshold))
            rows.append({"sd_multiple": multiple, "bound": bound, "validation_tail": empirical})
        truncations.append(
            {
                "quantile": quantile,
                "cutoff": cutoff,
                "calibration_exceptional_mass": exceptional_calibration,
                "validation_exceptional_mass": exceptional_validation,
                "truncated_variance": truncated_variance,
                "truncated_doob_variance_sum": truncated_variance_sum,
                "truncated_b_star_over_sd": truncated_b_star / max(np.sqrt(truncated_variance), 1e-12),
                "calibration_mean_shift": mean_shift,
                "validation_mean_shift": validation_mean_shift,
                "tail_rows": rows,
            }
        )
    nonvacuous = any(row["bound"] < 1 for item in truncations for row in item["tail_rows"])
    covered = all(
        row["validation_tail"] <= row["bound"] + 1e-12
        for item in truncations
        for row in item["tail_rows"]
    )
    return {
        "calibration_variance": variance,
        "validation_variance": validation_variance,
        "variance_transfer_ratio": validation_variance / max(variance, 1e-12),
        "doob_variance_sum": variance_sum,
        "doob_variance_ratio": variance_sum / max(variance, 1e-12),
        "b_star_over_sd": b_star / sd,
        "increments": increments,
        "stable_truncation": truncations,
        "nonvacuous": nonvacuous,
        "heldout_covered": covered,
        "status": "empirical conditional-MGF falsification diagnostic, not a proof of Assumption CM",
    }


def analyze(name):
    fit, reference, calibration, validation, kind, source_total = load_pools(name)
    dimension = fit.shape[1]
    rng = np.random.default_rng(SEED + dimension * 17)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), dimension)
    results = []
    for quantizer in QUANTIZERS:
        transformed = []
        for values in [fit, reference, calibration, validation]:
            transformed.append(values if quantizer == "2bit" else rotate(values, signs))
        threshold = float(np.mean(np.abs(transformed[0])))
        codes = [encode(values, threshold) for values in transformed]
        beta = fit_calibration(fit, codes[0], SEED + dimension)
        roles = {}
        for offset, role in enumerate(["ranking", "pruning"]):
            roles[role] = analyze_role(
                reference,
                codes[1],
                calibration,
                codes[2],
                validation,
                codes[3],
                beta,
                role,
                SEED + dimension * 31 + offset,
            )
        results.append(
            {
                "quantizer": quantizer,
                "threshold": threshold,
                "calibration": {"intercept": beta[0], "slope": beta[1]},
                "roles": roles,
            }
        )
    return {
        "dataset": name,
        "kind": kind,
        "dimension": dimension,
        "source_total": source_total,
        "configurations": results,
    }


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = analyze(name)
        results.append(result)
        for configuration in result["configurations"]:
            rank = configuration["roles"]["ranking"]
            prune = configuration["roles"]["pruning"]
            print(
                f"  {configuration['quantizer']:<13} "
                f"rank b/sd={rank['b_star_over_sd']:.3f}, "
                f"prune b/sd={prune['b_star_over_sd']:.3f}, "
                f"nonvacuous={rank['nonvacuous']}/{prune['nonvacuous']}"
            )
    output = {
        "probe": 40,
        "seed": SEED,
        "protocol": {
            "independent_pools": ["fit", "reference", "calibration", "validation"],
            "groups": N_GROUPS,
            "second_nodes_per_first": N_SECOND,
            "third_nodes_per_prefix": N_THIRD,
            "integration_second_nodes": N_INTEGRATION_SECOND,
            "integration_third_nodes": N_INTEGRATION_THIRD,
            "lambda_grid": LAMBDA_GRID.tolist(),
            "truncation_quantiles": TRUNCATION_QUANTILES,
        },
        "results": results,
    }
    path = RESULTS_DIR / "probe40_standard_vamana_p8b_conditional_mgf.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
