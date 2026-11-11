#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09 - Probe 26：协方差感知排序残差与证书覆盖实验

检验共享查询残差协方差、边际尾校准、P1/P2 形式界覆盖与弃权门禁。
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.special import erfc
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["1bit", "2bit", "rotated_1bit", "rotated_2bit"]
N_SAMPLE = 4_000
N_CALIBRATION_QUERIES = 80
N_EVALUATION_QUERIES = 160
LOCAL_K = 32
SEED = 20260823
TAIL_THRESHOLDS = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
BOUNDARY_QUANTILES = np.array([0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50])
TAIL_RATIO_LIMIT = 2.0


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def safe_variance(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.var(values, ddof=1)) if values.size > 1 else 0.0


def load_sampled_dataset(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 26 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED)
    indices = rng.choice(total, min(N_SAMPLE, total), replace=False)
    vecs = np.asarray(mapped[indices], dtype=np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    print(f"  [{name}] 直接映射抽样：{total} → {len(vecs)}")
    return vecs


def randomized_orthogonal_transform(vecs):
    rng = np.random.default_rng(SEED + vecs.shape[1])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), vecs.shape[1])
    rotated = dct(vecs * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
    return rotated


def encode(vecs):
    signs = vecs > 0
    strongs = np.abs(vecs) > np.abs(vecs).mean(axis=1, keepdims=True)
    return signs, strongs


def build_exact_records(vecs, query_indices):
    similarities = vecs[query_indices] @ vecs.T
    records = []
    for row_index, query_index in enumerate(query_indices):
        similarities[row_index, query_index] = -np.inf
        candidates = np.argpartition(-similarities[row_index], LOCAL_K)[:LOCAL_K]
        candidates = candidates[np.argsort(-similarities[row_index, candidates])]
        records.append(
            {
                "query_index": int(query_index),
                "candidates": candidates,
                "exact_scores": similarities[row_index, candidates].astype(np.float64),
            }
        )
    return records


def quantized_scores(query_index, candidates, signs, strongs, bitwidth):
    agreements = signs[candidates] == signs[query_index]
    signed = 2.0 * agreements.astype(np.float64) - 1.0
    if bitwidth == 1:
        return signed.sum(axis=1)
    weights = (1.0 + strongs[candidates]) * (1.0 + strongs[query_index])
    return (signed * weights).sum(axis=1)


def add_quantized_scores(records, encoded, bitwidth):
    signs, strongs = encoded
    output = []
    for row in records:
        scores = quantized_scores(
            row["query_index"], row["candidates"], signs, strongs, bitwidth
        )
        output.append(
            {
                "query_index": row["query_index"],
                "exact_scores": row["exact_scores"],
                "quantized_scores": scores,
            }
        )
    return output


def flatten_pair_records(records):
    exact_top = np.concatenate(
        [np.full(len(row["exact_scores"]) - 1, row["exact_scores"][0]) for row in records]
    )
    exact_competitor = np.concatenate([row["exact_scores"][1:] for row in records])
    score_top = np.concatenate(
        [
            np.full(len(row["quantized_scores"]) - 1, row["quantized_scores"][0])
            for row in records
        ]
    )
    score_competitor = np.concatenate([row["quantized_scores"][1:] for row in records])
    return exact_top, exact_competitor, score_top, score_competitor


def fit_model(records):
    exact_top, exact_competitor, score_top, score_competitor = flatten_pair_records(records)
    exact_margin = exact_top - exact_competitor
    score_margin = score_top - score_competitor
    denominator = float(np.dot(score_margin, score_margin))
    slope = float(np.dot(score_margin, exact_margin) / denominator) if denominator > 0 else 0.0
    top_residual = slope * score_top - exact_top
    competitor_residual = slope * score_competitor - exact_competitor
    margin_residual = top_residual - competitor_residual
    covariance = np.cov(top_residual, competitor_residual, ddof=1)
    covariance_variance = max(
        float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]), 0.0
    )
    independence_variance = max(float(covariance[0, 0] + covariance[1, 1]), 0.0)
    empirical_variance = safe_variance(margin_residual)
    return {
        "slope": slope,
        "exact_margin": exact_margin,
        "score_margin": score_margin,
        "top_residual": top_residual,
        "competitor_residual": competitor_residual,
        "margin_residual": margin_residual,
        "margin_residual_mean": float(np.mean(margin_residual)),
        "covariance": covariance,
        "covariance_variance": covariance_variance,
        "independence_variance": independence_variance,
        "empirical_variance": empirical_variance,
        "covariance_scale": max(np.sqrt(covariance_variance), 1e-12),
        "independence_scale": max(np.sqrt(independence_variance), 1e-12),
    }


def tail_metrics(residual, mean, scale):
    standardized = (residual - mean) / max(scale, 1e-12)
    empirical_two_sided = np.array(
        [(np.abs(standardized) >= threshold).mean() for threshold in TAIL_THRESHOLDS]
    )
    empirical_left = np.array(
        [(standardized <= -threshold).mean() for threshold in TAIL_THRESHOLDS]
    )
    gaussian_two_sided = erfc(TAIL_THRESHOLDS / np.sqrt(2.0))
    gaussian_one_sided = 0.5 * gaussian_two_sided
    two_sided_ratios = empirical_two_sided / np.maximum(gaussian_two_sided, 1e-12)
    left_ratios = empirical_left / np.maximum(gaussian_one_sided, 1e-12)
    return {
        "thresholds": TAIL_THRESHOLDS.tolist(),
        "empirical_two_sided_survival": empirical_two_sided.tolist(),
        "gaussian_two_sided_survival": gaussian_two_sided.tolist(),
        "two_sided_ratio": two_sided_ratios.tolist(),
        "empirical_left_survival": empirical_left.tolist(),
        "gaussian_one_sided_survival": gaussian_one_sided.tolist(),
        "left_ratio": left_ratios.tolist(),
        "max_two_sided_ratio_through_2_5": float(np.max(two_sided_ratios[:4])),
        "max_left_ratio_through_2_5": float(np.max(left_ratios[:4])),
        "standardized_kurtosis": float(np.mean(standardized**4)),
    }


def choose_boundary_threshold(model):
    margins = model["exact_margin"]
    positive = margins[margins > 0]
    taus = np.unique(np.quantile(positive, BOUNDARY_QUANTILES))
    mean = model["margin_residual_mean"]
    scale = model["covariance_scale"]
    risks = []
    for tau in taus:
        boundary = float(np.mean((margins > 0) & (margins <= tau)))
        threshold = tau + mean
        tail = 1.0 if threshold <= 0 else float(np.exp(-0.5 * (threshold / scale) ** 2))
        risks.append(min(1.0, boundary + tail))
    best = int(np.argmin(risks))
    return float(taus[best]), float(risks[best])


def evaluate(records, model):
    exact_top, exact_competitor, score_top, score_competitor = flatten_pair_records(records)
    exact_margin = exact_top - exact_competitor
    score_margin = score_top - score_competitor
    top_residual = model["slope"] * score_top - exact_top
    competitor_residual = model["slope"] * score_competitor - exact_competitor
    margin_residual = top_residual - competitor_residual
    predicted_margin = model["slope"] * score_margin
    flips = predicted_margin <= 0
    tau, calibration_bound = choose_boundary_threshold(model)
    evaluation_boundary = float(np.mean((exact_margin > 0) & (exact_margin <= tau)))
    threshold = tau + model["margin_residual_mean"]
    covariance_tail = (
        1.0
        if threshold <= 0
        else float(np.exp(-0.5 * (threshold / model["covariance_scale"]) ** 2))
    )
    independence_tail = (
        1.0
        if threshold <= 0
        else float(np.exp(-0.5 * (threshold / model["independence_scale"]) ** 2))
    )
    covariance_bound = min(1.0, evaluation_boundary + covariance_tail)
    independence_bound = min(1.0, evaluation_boundary + independence_tail)
    return {
        "exact_margin": exact_margin,
        "score_margin": score_margin,
        "margin_residual": margin_residual,
        "flips": flips,
        "decision_flip_rate": float(flips.mean()),
        "score_margin_spearman": safe_spearman(score_margin, exact_margin),
        "residual_margin_spearman": safe_spearman(margin_residual, exact_margin),
        "tail": tail_metrics(
            margin_residual,
            model["margin_residual_mean"],
            model["covariance_scale"],
        ),
        "boundary_bound": {
            "selected_tau": tau,
            "calibration_plugin_bound": calibration_bound,
            "evaluation_boundary_mass": evaluation_boundary,
            "covariance_plugin_tail": covariance_tail,
            "covariance_plugin_bound": covariance_bound,
            "independence_plugin_tail": independence_tail,
            "independence_plugin_bound": independence_bound,
            "covariance_covers_empirical": bool(covariance_bound + 1e-12 >= flips.mean()),
            "independence_covers_empirical": bool(independence_bound + 1e-12 >= flips.mean()),
            "status": "仅检验定理形式；样本方差不是已证明的 MGF proxy",
        },
    }


def eligibility(model, evaluated):
    reasons = []
    if model["slope"] <= 0:
        reasons.append("校准斜率非正")
    if evaluated["score_margin_spearman"] < 0.10:
        reasons.append("分数与精确 margin 的秩相关过低")
    if abs(evaluated["residual_margin_spearman"]) > 0.50:
        reasons.append("residual 与 margin 的相关过强")
    if evaluated["tail"]["max_left_ratio_through_2_5"] > TAIL_RATIO_LIMIT:
        reasons.append("左尾超过 Gaussian proxy 门限")
    return {"eligible": not reasons, "reasons": reasons}


def run_quantizer(exact_calibration, exact_evaluation, encoded, bitwidth):
    calibration = add_quantized_scores(exact_calibration, encoded, bitwidth)
    evaluation = add_quantized_scores(exact_evaluation, encoded, bitwidth)
    model = fit_model(calibration)
    evaluated = evaluate(evaluation, model)
    gate = eligibility(model, evaluated)
    covariance = model["covariance"]
    return {
        "calibration_slope": model["slope"],
        "effective_calibrated_scale": 1.0,
        "calibration_decision_count": int(len(model["exact_margin"])),
        "evaluation_decision_count": int(len(evaluated["exact_margin"])),
        "residual_structure": {
            "top_variance": float(covariance[0, 0]),
            "competitor_variance": float(covariance[1, 1]),
            "shared_query_cross_covariance": float(covariance[0, 1]),
            "residual_correlation": float(
                covariance[0, 1]
                / max(np.sqrt(covariance[0, 0] * covariance[1, 1]), 1e-12)
            ),
            "empirical_difference_variance": model["empirical_variance"],
            "covariance_aware_variance": model["covariance_variance"],
            "independence_variance": model["independence_variance"],
            "independence_over_empirical_ratio": float(
                model["independence_variance"] / max(model["empirical_variance"], 1e-12)
            ),
            "covariance_aware_over_empirical_ratio": float(
                model["covariance_variance"] / max(model["empirical_variance"], 1e-12)
            ),
            "margin_residual_mean": model["margin_residual_mean"],
        },
        "evaluation": {
            "decision_flip_rate": evaluated["decision_flip_rate"],
            "score_margin_spearman": evaluated["score_margin_spearman"],
            "residual_margin_spearman": evaluated["residual_margin_spearman"],
            "tail": evaluated["tail"],
            "boundary_bound": evaluated["boundary_bound"],
        },
        "calibration_gate": gate,
    }


def run_dataset(name):
    vecs = load_sampled_dataset(name)
    rotated = randomized_orthogonal_transform(vecs)
    rng = np.random.default_rng(SEED)
    query_indices = rng.choice(
        len(vecs), N_CALIBRATION_QUERIES + N_EVALUATION_QUERIES, replace=False
    )
    calibration_indices = query_indices[:N_CALIBRATION_QUERIES]
    evaluation_indices = query_indices[N_CALIBRATION_QUERIES:]
    exact_calibration = build_exact_records(vecs, calibration_indices)
    exact_evaluation = build_exact_records(vecs, evaluation_indices)
    encodings = {"original": encode(vecs), "rotated": encode(rotated)}
    result = {"dataset": name, "dimension": int(vecs.shape[1]), "quantizers": {}}
    for quantizer in QUANTIZERS:
        rotated_mode = quantizer.startswith("rotated")
        bitwidth = 1 if quantizer.endswith("1bit") else 2
        result["quantizers"][quantizer] = run_quantizer(
            exact_calibration,
            exact_evaluation,
            encodings["rotated" if rotated_mode else "original"],
            bitwidth,
        )
    return result


def save_results(results):
    output = {
        "probe": 26,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "n_sample": N_SAMPLE,
        "n_calibration_queries": N_CALIBRATION_QUERIES,
        "n_evaluation_queries": N_EVALUATION_QUERIES,
        "local_k": LOCAL_K,
        "tail_ratio_limit": TAIL_RATIO_LIMIT,
        "results": results,
    }
    path = RESULTS_DIR / "probe26_covariance_certificate.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            structure = metrics["residual_structure"]
            evaluation = metrics["evaluation"]
            gate = metrics["calibration_gate"]
            print(
                f"  {quantizer:<13} eligible={gate['eligible']}, "
                f"flip={evaluation['decision_flip_rate']:.4f}, "
                f"cov={structure['shared_query_cross_covariance']:.4g}, "
                f"ind/emp={structure['independence_over_empirical_ratio']:.3f}, "
                f"left-tail={evaluation['tail']['max_left_ratio_through_2_5']:.2f}x, "
                f"bound={evaluation['boundary_bound']['covariance_plugin_bound']:.3f}"
            )
            if gate["reasons"]:
                print(f"    弃权原因：{'；'.join(gate['reasons'])}")
    save_results(results)


if __name__ == "__main__":
    main()
