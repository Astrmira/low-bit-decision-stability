#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27 - C3 污染量级与trace事件语义审计

使用精确有限维公式核对稀有污染构造的概率、薄壳、方差和四阶矩量级，
并用确定性布尔例验证保守flip只给inclusion、witness冗余和饱和变体边界。
"""

import json
from math import lgamma

import numpy as np

from utils import RESULTS_DIR


DIMENSIONS = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)


def gaussian_norm_mean(dimension):
    return np.sqrt(2.0) * np.exp(
        lgamma((dimension + 1.0) / 2.0) - lgamma(dimension / 2.0)
    )


def log_slope(dimensions, values):
    selected = dimensions >= 128
    return float(np.polyfit(np.log(dimensions[selected]), np.log(values[selected]), 1)[0])


def contamination_audit():
    rows = []
    for dimension in DIMENSIONS:
        epsilon = dimension ** -4
        amplitude = dimension ** 1.5
        gaussian_radius = gaussian_norm_mean(dimension)
        contaminated_radius = amplitude * np.sqrt(dimension)
        mean_radius = (1 - epsilon) * gaussian_radius + epsilon * contaminated_radius
        second_radius_moment = (1 - epsilon) * dimension + epsilon * amplitude**2 * dimension
        relative_shell_mse = (
            second_radius_moment - 2 * mean_radius**2 + mean_radius**2
        ) / mean_radius**2
        coordinate_variance = (1 - epsilon) + epsilon * amplitude**2
        ranking_variance = 2 * dimension * coordinate_variance**2 / mean_radius**4
        two_contaminated_event = epsilon**2 * (1 - epsilon)
        rademacher_fourth = 3 * dimension**2 - 2 * dimension
        fourth_moment_contribution = (
            two_contaminated_event
            * amplitude**8
            * rademacher_fourth
            / mean_radius**8
        )
        rows.append(
            {
                "dimension": int(dimension),
                "single_contamination_probability": epsilon,
                "two_contaminated_one_clean_probability": two_contaminated_event,
                "relative_shell_mse": relative_shell_mse,
                "relative_shell_l2_norm": np.sqrt(relative_shell_mse),
                "coordinate_variance_excess": coordinate_variance - 1.0,
                "oracle_bilinear_ranking_variance": ranking_variance,
                "conditional_rademacher_fourth_moment": rademacher_fourth,
                "unconditional_fourth_moment_contribution": fourth_moment_contribution,
            }
        )

    def series(key):
        return np.asarray([row[key] for row in rows], dtype=float)

    return {
        "rows": rows,
        "log_log_slopes_over_d_ge_128": {
            "single_contamination_probability": log_slope(
                DIMENSIONS, series("single_contamination_probability")
            ),
            "two_contaminated_one_clean_probability": log_slope(
                DIMENSIONS, series("two_contaminated_one_clean_probability")
            ),
            "relative_shell_mse": log_slope(DIMENSIONS, series("relative_shell_mse")),
            "relative_shell_l2_norm": log_slope(
                DIMENSIONS, series("relative_shell_l2_norm")
            ),
            "coordinate_variance_excess": log_slope(
                DIMENSIONS, series("coordinate_variance_excess")
            ),
            "oracle_bilinear_ranking_variance": log_slope(
                DIMENSIONS, series("oracle_bilinear_ranking_variance")
            ),
            "unconditional_fourth_moment_contribution": log_slope(
                DIMENSIONS, series("unconditional_fourth_moment_contribution")
            ),
        },
        "statement": "双节点污染事件为D^-8量级；其归一化bilinear四阶矩贡献为D^2量级。relative shell MSE为D^-1，L2范数误差为D^-1/2。",
    }


def action(value, tie_is_dominated=True):
    return value >= 0 if tie_is_dominated else value > 0


def trace_semantics_audit():
    exact_value = 1.0
    approximate_tie = 0.0
    conservative_flip = exact_value * approximate_tie <= 0 and exact_value != 0
    exact_action = action(exact_value, tie_is_dominated=True)
    approximate_action = action(approximate_tie, tie_is_dominated=True)

    exact_witness_bits = [True, True]
    approximate_witness_bits = [False, True]
    witness_disagreements = [
        exact != approximate
        for exact, approximate in zip(exact_witness_bits, approximate_witness_bits)
    ]
    exact_candidate_action = any(exact_witness_bits)
    approximate_candidate_action = any(approximate_witness_bits)

    diversity_selected = [1]
    ordered = [1, 2, 3]
    unsaturated_output = list(diversity_selected)
    saturated_output = diversity_selected + [
        candidate for candidate in ordered if candidate not in diversity_selected
    ][:2]

    return {
        "conservative_tie_case": {
            "exact_value": exact_value,
            "approximate_value": approximate_tie,
            "conservative_flip": conservative_flip,
            "exact_action": exact_action,
            "approximate_action": approximate_action,
            "action_disagreement": exact_action != approximate_action,
            "conclusion": "保守flip发生但tie-aware实际action一致，因此只能用事件包含，不能写事件相等。",
        },
        "witness_redundancy_case": {
            "exact_witness_bits": exact_witness_bits,
            "approximate_witness_bits": approximate_witness_bits,
            "witness_disagreements": witness_disagreements,
            "exact_candidate_or": exact_candidate_action,
            "approximate_candidate_or": approximate_candidate_action,
            "candidate_action_disagreement": exact_candidate_action != approximate_candidate_action,
            "conclusion": "一个witness bit翻转可被另一个witness补偿；逐witness一致是充分条件而非必要条件。",
        },
        "saturation_boundary_case": {
            "diversity_selected": diversity_selected,
            "standard_unsaturated_output": unsaturated_output,
            "explicit_saturated_output": saturated_output,
            "outputs_equal": unsaturated_output == saturated_output,
            "conclusion": "饱和填充是额外输出映射，不属于原始RobustPrune状态本身。",
        },
    }


def main():
    output = {
        "probe": 44,
        "contamination": contamination_audit(),
        "trace_semantics": trace_semantics_audit(),
    }
    slopes = output["contamination"]["log_log_slopes_over_d_ge_128"]
    assert abs(slopes["two_contaminated_one_clean_probability"] + 8) < 0.01
    assert abs(slopes["unconditional_fourth_moment_contribution"] - 2) < 0.05
    assert abs(slopes["relative_shell_mse"] + 1) < 0.05
    assert abs(slopes["relative_shell_l2_norm"] + 0.5) < 0.05
    tie_case = output["trace_semantics"]["conservative_tie_case"]
    assert tie_case["conservative_flip"] and not tie_case["action_disagreement"]
    redundancy = output["trace_semantics"]["witness_redundancy_case"]
    assert any(redundancy["witness_disagreements"])
    assert not redundancy["candidate_action_disagreement"]

    path = RESULTS_DIR / "probe44_c3_contamination_trace_semantics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("量级斜率：", {key: round(value, 3) for key, value in slopes.items()})
    print("保守事件不等于action disagreement：已验证")
    print("witness冗余：已验证")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
