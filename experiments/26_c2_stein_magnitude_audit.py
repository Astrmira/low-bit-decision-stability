#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26 - C2 Stein transfer与magnitude-bit命题审计

比较精确Gaussian下的Stein恒等式与近似Gaussian混合分布，
并区分受限等权正例、随机共享阈值和固定聚合scorer反例。
"""

import json

import numpy as np

from utils import RESULTS_DIR


SEED = 20260908
N = 300_000
D = 8


def correlation(left, right):
    return float(np.corrcoef(left, right)[0, 1])


def stein_prediction(values, weights):
    mu = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=0)
    sigma = np.sqrt(np.diag(covariance))
    density = np.exp(-0.5 * (mu / sigma) ** 2) / np.sqrt(2 * np.pi)
    coefficients = 2 * density / sigma
    return float(coefficients @ covariance @ weights)


def stein_observed(values, weights):
    sign_sum = np.sign(values).sum(axis=1)
    target = values @ weights
    return float(np.cov(sign_sum, target, ddof=0)[0, 1])


def stein_audit(rng):
    base = rng.normal(size=(N, D))
    transform = np.eye(D)
    transform[1:, :-1] += 0.18 * np.eye(D - 1)
    gaussian = base @ transform.T + np.linspace(-0.2, 0.2, D)
    weights = np.linspace(0.3, 1.2, D)

    mixture = gaussian.copy()
    contaminated = rng.random(N) < 0.01
    mixture[contaminated, 0] += 7.0
    mixture[contaminated, 1] -= 5.0

    output = {}
    for name, values in (("exact_gaussian", gaussian), ("near_gaussian_mixture", mixture)):
        observed = stein_observed(values, weights)
        predicted = stein_prediction(values, weights)
        output[name] = {
            "observed_covariance": observed,
            "gaussian_identity_prediction": predicted,
            "absolute_error": abs(observed - predicted),
            "relative_error": abs(observed - predicted) / max(abs(observed), 1e-12),
        }
    output["contamination_probability"] = 0.01
    output["statement"] = "精确Gaussian样本支持恒等式；小混合污染下使用均值协方差代入Gaussian公式不再是精确等式。"
    return output


def encode_scores(values, threshold):
    signs = np.where(values >= 0, 1.0, -1.0)
    magnitude = (np.abs(values) > threshold).astype(float)
    return signs.sum(axis=1), (signs * (1.0 + magnitude)).sum(axis=1)


def magnitude_case(rng, sigma, weights, threshold_mode):
    values = rng.normal(size=(N, len(sigma))) * sigma
    if threshold_mode == "population":
        threshold = np.full((N, 1), np.sqrt(2 / np.pi) * np.mean(sigma))
    else:
        threshold = np.mean(np.abs(values), axis=1, keepdims=True)
    one_bit, two_bit = encode_scores(values, threshold)
    target = values @ weights
    return {
        "one_bit_correlation": correlation(one_bit, target),
        "two_bit_correlation": correlation(two_bit, target),
        "gain": correlation(two_bit, target) - correlation(one_bit, target),
    }


def magnitude_audit(rng):
    isotropic_sigma = np.ones(D)
    equal_weights = np.ones(D)
    nuisance_sigma = np.array([1.0, 1.0, 1.0, 1.0, 8.0, 8.0, 8.0, 8.0])
    target_weights = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    restricted_population = magnitude_case(
        rng, isotropic_sigma, equal_weights, "population"
    )
    restricted_shared = magnitude_case(rng, isotropic_sigma, equal_weights, "shared")
    fixed_scorer_counterexample = magnitude_case(
        rng, nuisance_sigma, target_weights, "population"
    )

    heterogeneity_grid = []
    for scale in [1.0, 1.5, 2.0, 4.0, 8.0]:
        sigma = np.array([1.0] * 4 + [scale] * 4)
        heterogeneity_grid.append(
            {
                "nuisance_scale": scale,
                **magnitude_case(rng, sigma, target_weights, "population"),
            }
        )

    return {
        "restricted_equal_weight_population_threshold": restricted_population,
        "restricted_equal_weight_shared_threshold": restricted_shared,
        "fixed_scorer_counterexample": fixed_scorer_counterexample,
        "counterexample_sigma": nuisance_sigma.tolist(),
        "counterexample_target_weights": target_weights.tolist(),
        "nuisance_scale_grid": heterogeneity_grid,
        "statement": "更多magnitude信息不降低最优读出能力，但当前固定等权scorer可因放大目标无关高方差坐标而变差。",
    }


def main():
    rng = np.random.default_rng(SEED)
    output = {
        "probe": 43,
        "seed": SEED,
        "n": N,
        "dimension": D,
        "stein": stein_audit(rng),
        "magnitude": magnitude_audit(rng),
    }
    assert output["stein"]["exact_gaussian"]["relative_error"] < 0.01
    assert output["stein"]["near_gaussian_mixture"]["relative_error"] > 0.01
    assert output["magnitude"]["restricted_equal_weight_population_threshold"]["gain"] > 0
    assert output["magnitude"]["fixed_scorer_counterexample"]["gain"] < 0
    path = RESULTS_DIR / "probe43_c2_stein_magnitude_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Stein相对误差：Gaussian=",
        f"{output['stein']['exact_gaussian']['relative_error']:.4f}",
        "mixture=",
        f"{output['stein']['near_gaussian_mixture']['relative_error']:.4f}",
    )
    print(
        "Magnitude增益：受限正例=",
        f"{output['magnitude']['restricted_equal_weight_population_threshold']['gain']:.4f}",
        "固定scorer反例=",
        f"{output['magnitude']['fixed_scorer_counterexample']['gain']:.4f}",
    )
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
