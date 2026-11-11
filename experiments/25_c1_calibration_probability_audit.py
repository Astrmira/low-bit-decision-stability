#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25 - C1 校准、概率空间与尾部代理审计

用可复现合成实验区分过原点/仿射校准、固定数据上的确定性决策、
经验协方差与 MGF 代理，以及 no-atom 与方向控制。
"""

import json

import numpy as np

from utils import RESULTS_DIR


SEED = 20260907
N = 200_000
ALPHA = 1.2


def origin_fit(exact, approximate):
    slope = float(np.dot(exact, approximate) / np.dot(exact, exact))
    residual = approximate - slope * exact
    return slope, residual


def affine_fit(exact, approximate):
    design = np.column_stack([np.ones(len(exact)), exact])
    intercept, slope = np.linalg.lstsq(design, approximate, rcond=None)[0]
    residual = approximate - intercept - slope * exact
    return float(intercept), float(slope), residual


def calibration_audit(rng):
    exact = rng.normal(1.5, 0.8, N)
    approximate = 0.35 + 0.72 * exact + rng.normal(0, 0.18, N)
    origin_slope, origin_residual = origin_fit(exact, approximate)
    intercept, affine_slope, affine_residual = affine_fit(exact, approximate)

    edge_bias = 0.35
    ranking_intercept = edge_bias - edge_bias
    pruning_intercept = edge_bias - ALPHA * edge_bias
    return {
        "origin_fit": {
            "slope": origin_slope,
            "residual_mean": float(origin_residual.mean()),
            "exact_residual_cross_moment": float(np.mean(exact * origin_residual)),
            "exact_residual_covariance": float(np.cov(exact, origin_residual, ddof=0)[0, 1]),
        },
        "affine_fit": {
            "intercept": intercept,
            "slope": affine_slope,
            "residual_mean": float(affine_residual.mean()),
            "exact_residual_covariance": float(np.cov(exact, affine_residual, ddof=0)[0, 1]),
        },
        "shared_edge_intercept": {
            "edge_intercept": edge_bias,
            "ranking_difference_intercept": ranking_intercept,
            "standard_pruning_intercept": pruning_intercept,
        },
    }


def probability_space_audit():
    exact = np.array([0.12, -0.08, 0.03])
    deterministic_approximate = np.array([0.10, -0.02, -0.01])
    fixed_flip = (exact >= 0) != (deterministic_approximate >= 0)
    return {
        "fixed_exact": exact.tolist(),
        "fixed_deterministic_approximate": deterministic_approximate.tolist(),
        "fixed_flip": fixed_flip.tolist(),
        "conditional_probabilities_given_data": fixed_flip.astype(float).tolist(),
        "statement": "固定数据与确定性量化器后，决策概率只能是0或1；跨样本频率属于数据分布层。",
    }


def covariance_mgf_audit(rng):
    gaussian = rng.normal(0, 1, N)
    rare_probability = 0.01
    amplitude = 1.0 / np.sqrt(rare_probability)
    rare = np.zeros(N)
    active = rng.random(N) < rare_probability
    rare[active] = rng.choice([-amplitude, amplitude], active.sum())
    lambdas = [0.25, 0.5, 0.75]

    def summary(values):
        return {
            "mean": float(values.mean()),
            "variance": float(values.var()),
            "tail_at_3": float(np.mean(np.abs(values) >= 3.0)),
            "log_mgf": {
                str(lam): float(np.log(np.mean(np.exp(np.clip(lam * values, -40, 40)))))
                for lam in lambdas
            },
        }

    return {
        "gaussian": summary(gaussian),
        "rare_same_variance": summary(rare),
        "rare_probability": rare_probability,
        "rare_amplitude": amplitude,
        "statement": "近似相同二阶矩不决定尾部或MGF，因此经验covariance不能自动充当sub-Gaussian proxy。",
    }


def no_atom_audit(rng):
    centered = rng.normal(0, 1e-6, N)
    shifted = rng.normal(-2e-6, 1e-6, N)
    return {
        "centered_continuous": {
            "exact_zero_mass": float(np.mean(centered == 0)),
            "positive_probability": float(np.mean(centered > 0)),
        },
        "shifted_continuous": {
            "exact_zero_mass": float(np.mean(shifted == 0)),
            "positive_probability": float(np.mean(shifted > 0)),
        },
        "statement": "无原子只排除精确近似tie，不控制错误方向概率。",
    }


def main():
    rng = np.random.default_rng(SEED)
    output = {
        "probe": 42,
        "seed": SEED,
        "n": N,
        "calibration": calibration_audit(rng),
        "probability_space": probability_space_audit(),
        "covariance_vs_mgf": covariance_mgf_audit(rng),
        "no_atom": no_atom_audit(rng),
    }
    assert abs(output["calibration"]["origin_fit"]["exact_residual_cross_moment"]) < 1e-10
    assert abs(output["calibration"]["origin_fit"]["residual_mean"]) > 0.05
    assert abs(output["calibration"]["affine_fit"]["residual_mean"]) < 1e-10
    assert output["covariance_vs_mgf"]["rare_same_variance"]["tail_at_3"] > 0.005
    path = RESULTS_DIR / "probe42_c1_calibration_probability_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"过原点残差均值：{output['calibration']['origin_fit']['residual_mean']:.4f}")
    print(f"仿射残差均值：{output['calibration']['affine_fit']['residual_mean']:.3e}")
    print(f"连续对称变量正半轴概率：{output['no_atom']['centered_continuous']['positive_probability']:.3f}")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
