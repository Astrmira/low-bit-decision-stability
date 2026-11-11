#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""18 - Probe 35：决策稳定性的必要性反例。"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR


SEED = 20260826
N = 500_000
OUTLIER_RATE = 0.10
OUTLIER_SCALE = 0.35


def safe_spearman(left, right):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def margin_zero_counterexample():
    epsilon = np.logspace(-12, -2, 11)
    exact = np.zeros_like(epsilon)
    positive = epsilon
    negative = -epsilon
    return {
        "epsilon": epsilon.tolist(),
        "exact_margin": exact.tolist(),
        "positive_decision": (positive > 0).tolist(),
        "negative_decision": (negative > 0).tolist(),
        "arbitrarily_small_opposite_outcomes": bool(
            np.all(positive > 0) and np.all(negative < 0)
        ),
        "statement": "At zero exact margin, residuals of arbitrarily small norm can select opposite decisions.",
    }


def calibrated_score(exact, raw):
    design = np.column_stack([np.ones(len(raw)), raw])
    beta, _, _, _ = np.linalg.lstsq(design, exact, rcond=None)
    return design @ beta, beta


def summarize(exact, approximate, beta):
    residual = approximate - exact
    signs = np.sign(exact)
    flip = signs * approximate <= 0
    local = np.abs(exact) <= np.quantile(np.abs(exact), 0.20)
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "spearman": safe_spearman(exact, approximate),
        "mse": float(np.mean(residual**2)),
        "residual_variance": float(np.var(residual)),
        "residual_abs_p99": float(np.quantile(np.abs(residual), 0.99)),
        "residual_abs_p999": float(np.quantile(np.abs(residual), 0.999)),
        "flip_rate": float(np.mean(flip)),
        "hard_margin_flip_rate": float(np.mean(flip[local])),
    }


def matched_global_different_tail():
    rng = np.random.default_rng(SEED)
    exact = rng.normal(0, 1, N)
    boundary = np.abs(exact) <= np.quantile(np.abs(exact), OUTLIER_RATE)
    coupled_raw = exact.copy()
    coupled_raw[boundary] = -np.sign(exact[boundary]) * OUTLIER_SCALE
    target_spearman = safe_spearman(exact, coupled_raw)
    gaussian = rng.normal(0, 1, N)
    lower, upper = 0.0, 2.0
    for _ in range(36):
        middle = 0.5 * (lower + upper)
        correlation = safe_spearman(exact, exact + middle * gaussian)
        if correlation > target_spearman:
            lower = middle
        else:
            upper = middle
    light_raw = exact + 0.5 * (lower + upper) * gaussian
    light, beta_light = calibrated_score(exact, light_raw)
    heavy, beta_heavy = calibrated_score(exact, coupled_raw)
    light_summary = summarize(exact, light, beta_light)
    heavy_summary = summarize(exact, heavy, beta_heavy)
    return {
        "light_tail": light_summary,
        "boundary_coupled_tail": heavy_summary,
        "spearman_gap": abs(light_summary["spearman"] - heavy_summary["spearman"]),
        "mse_ratio": heavy_summary["mse"] / light_summary["mse"],
        "flip_ratio": heavy_summary["flip_rate"] / max(light_summary["flip_rate"], 1e-12),
        "hard_flip_ratio": heavy_summary["hard_margin_flip_rate"]
        / max(light_summary["hard_margin_flip_rate"], 1e-12),
        "boundary_fraction": OUTLIER_RATE,
        "boundary_flip_level": OUTLIER_SCALE,
        "statement": "Similar global rank fidelity does not identify residual tail shape or decision-flip risk.",
    }


def main():
    zero = margin_zero_counterexample()
    tail = matched_global_different_tail()
    output = {
        "probe": 35,
        "seed": SEED,
        "n": N,
        "margin_zero": zero,
        "matched_global_different_tail": tail,
    }
    path = RESULTS_DIR / "probe35_necessity_counterexamples.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Margin-zero 反例：", zero["arbitrarily_small_opposite_outcomes"])
    print(
        "异尾反例：spearman gap=",
        f"{tail['spearman_gap']:.4f}",
        "flip ratio=",
        f"{tail['flip_ratio']:.2f}",
    )
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
