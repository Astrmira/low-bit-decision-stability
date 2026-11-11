#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24 - RobustPrune 最小语义审计

验证标准 RobustPrune、历史参数化规则与可选饱和填充之间的差异。
该测试只使用确定性距离矩阵，不依赖数据集。
"""

import json
from pathlib import Path

import numpy as np

from utils import RESULTS_DIR


ALPHA = 1.2


def standard_dominated(distance_to_target, distance_to_selected, alpha):
    """标准 RobustPrune：alpha * d(s, c) <= d(t, c)。"""
    return distance_to_target - alpha * distance_to_selected >= 0.0


def historical_dominated(distance_to_target, distance_to_selected, alpha):
    """历史实验规则：d(s, c) < alpha * d(t, c)。"""
    return alpha * distance_to_target - distance_to_selected > 0.0


def select(distances, candidates, max_degree, alpha, rule, saturate):
    """按目标距离排序后执行逐候选多样性选择。"""
    ordered = sorted(candidates, key=lambda candidate: (distances[0, candidate], candidate))
    selected = []
    rejected = []
    decisions = []

    for candidate in ordered:
        if len(selected) >= max_degree:
            break
        witness = None
        comparisons = []
        for selected_node in selected:
            d_tc = float(distances[0, candidate])
            d_sc = float(distances[selected_node, candidate])
            dominated = bool(rule(d_tc, d_sc, alpha))
            comparisons.append(
                {
                    "selected": selected_node,
                    "distance_to_target": d_tc,
                    "distance_to_selected": d_sc,
                    "dominated": dominated,
                }
            )
            if dominated:
                witness = selected_node
                break
        is_dominated = witness is not None
        decisions.append(
            {
                "candidate": candidate,
                "dominated": is_dominated,
                "witness": witness,
                "comparisons": comparisons,
            }
        )
        if is_dominated:
            rejected.append(candidate)
        else:
            selected.append(candidate)

    diversity_selected = list(selected)
    fallback_added = []
    if saturate:
        for candidate in ordered:
            if len(selected) >= max_degree:
                break
            if candidate not in selected:
                selected.append(candidate)
                fallback_added.append(candidate)

    return {
        "ordered": ordered,
        "diversity_selected": diversity_selected,
        "rejected": rejected,
        "fallback_added": fallback_added,
        "final_selected": selected,
        "decisions": decisions,
    }


def main():
    direct_counterexample = {
        "distance_to_target": 1.0,
        "distance_to_selected": 1.0,
        "alpha": ALPHA,
        "standard_dominated": standard_dominated(1.0, 1.0, ALPHA),
        "historical_dominated": historical_dominated(1.0, 1.0, ALPHA),
    }
    assert direct_counterexample["standard_dominated"] is False
    assert direct_counterexample["historical_dominated"] is True

    alpha_grid = [0.8, 1.0, 1.2, 1.5]
    alpha_response = [
        {
            "alpha": alpha,
            "standard_dominated": standard_dominated(1.0, 0.8, alpha),
            "historical_dominated": historical_dominated(1.0, 0.8, alpha),
        }
        for alpha in alpha_grid
    ]
    standard_bits = [row["standard_dominated"] for row in alpha_response]
    historical_bits = [row["historical_dominated"] for row in alpha_response]
    assert standard_bits == [True, True, True, False]
    assert historical_bits == [False, True, True, True]

    distances = np.array(
        [
            [0.0, 1.0, 1.1, 1.2],
            [1.0, 0.0, 1.0, 1.5],
            [1.1, 1.0, 0.0, 0.7],
            [1.2, 1.5, 0.7, 0.0],
        ],
        dtype=np.float64,
    )
    candidates = [1, 2, 3]
    runs = {
        "standard_unsaturated": select(
            distances, candidates, 3, ALPHA, standard_dominated, False
        ),
        "standard_saturated": select(
            distances, candidates, 3, ALPHA, standard_dominated, True
        ),
        "historical_unsaturated": select(
            distances, candidates, 3, ALPHA, historical_dominated, False
        ),
        "historical_saturated": select(
            distances, candidates, 3, ALPHA, historical_dominated, True
        ),
    }

    assert runs["standard_unsaturated"]["final_selected"] == [1, 2]
    assert runs["standard_saturated"]["final_selected"] == [1, 2, 3]
    assert runs["historical_unsaturated"]["final_selected"] == [1, 3]
    assert runs["historical_saturated"]["final_selected"] == [1, 3, 2]

    output = {
        "probe": 41,
        "alpha": ALPHA,
        "direct_counterexample": direct_counterexample,
        "alpha_response": alpha_response,
        "runs": runs,
        "conclusions": {
            "rules_are_not_equivalent_at_equal_alpha": True,
            "alpha_response_is_opposite": True,
            "saturation_changes_output_semantics": True,
            "standard_prune_is_not_intrinsically_saturated": True,
        },
    }
    path = RESULTS_DIR / "probe41_vamana_semantics_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"语义审计通过，结果已保存：{path}")


if __name__ == "__main__":
    main()
