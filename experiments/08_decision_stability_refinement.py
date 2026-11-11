#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08 - 决策稳定性第二轮抽样实验

检验 residual tail、条件尺度、证书聚合方式与随机正交旋转。
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.special import erfc, logsumexp
from scipy.stats import kstest, norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["1bit", "2bit", "rotated_1bit", "rotated_2bit"]
N_SAMPLE = 4_000
N_CALIBRATION_QUERIES = 80
N_EVALUATION_QUERIES = 160
LOCAL_K = 32
CONDITIONAL_BINS = 6
SEED = 20260823


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_sampled_dataset(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"第二轮实验只支持 f32 数据集，收到 {data_format}")
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


def build_exact_candidates(vecs, query_indices):
    similarities = vecs[query_indices] @ vecs.T
    records = []
    for row_index, query_index in enumerate(query_indices):
        similarities[row_index, query_index] = -np.inf
        candidates = np.argpartition(-similarities[row_index], LOCAL_K)[:LOCAL_K]
        candidates = candidates[np.argsort(-similarities[row_index, candidates])]
        exact = similarities[row_index, candidates]
        records.append(
            {
                "query_index": int(query_index),
                "candidates": candidates,
                "exact_margins": exact[0] - exact[1:],
            }
        )
    return records


def add_quantized_margins(records, signs, strongs, bitwidth):
    output = []
    for row in records:
        query_index = row["query_index"]
        candidates = row["candidates"]
        agreements = signs[candidates] == signs[query_index]
        if bitwidth == 1:
            scores = (2.0 * agreements.astype(np.float64) - 1.0).sum(axis=1)
        else:
            weights = (1.0 + strongs[candidates]) * (1.0 + strongs[query_index])
            scores = ((2.0 * agreements.astype(np.float64) - 1.0) * weights).sum(axis=1)
        output.append(
            {
                "query_index": query_index,
                "exact_margins": row["exact_margins"],
                "score_margins": scores[0] - scores[1:],
            }
        )
    return output


def flatten(records, key):
    return np.concatenate([row[key] for row in records])


def fit_calibration(records):
    exact = flatten(records, "exact_margins")
    score = flatten(records, "score_margins")
    denominator = float(np.dot(score, score))
    slope = float(np.dot(score, exact) / denominator) if denominator > 0 else 0.0
    residual = slope * score - exact
    global_scale = max(float(np.std(residual, ddof=1)), 1e-12)
    edges = np.unique(np.quantile(exact, np.linspace(0.0, 1.0, CONDITIONAL_BINS + 1)))
    conditional_scales = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (exact >= left) & (exact <= right if right == edges[-1] else exact < right)
        local = residual[mask]
        scale = float(np.std(local, ddof=1)) if len(local) > 1 else global_scale
        conditional_scales.append(max(scale, global_scale * 0.05, 1e-12))
    return {
        "slope": slope,
        "global_scale": global_scale,
        "edges": edges,
        "conditional_scales": np.asarray(conditional_scales),
        "exact": exact,
        "residual": residual,
        "score": score,
    }


def conditional_scale(margins, model):
    indices = np.searchsorted(model["edges"][1:-1], margins, side="right")
    indices = np.clip(indices, 0, len(model["conditional_scales"]) - 1)
    return model["conditional_scales"][indices]


def residual_tail_metrics(model):
    standardized = (model["residual"] - model["residual"].mean()) / model["global_scale"]
    thresholds = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    empirical = np.array([(np.abs(standardized) >= value).mean() for value in thresholds])
    gaussian = erfc(thresholds / np.sqrt(2.0))
    ratios = empirical / np.maximum(gaussian, 1e-12)
    ks = kstest(standardized, norm.cdf)
    return {
        "standardized_mean": float(standardized.mean()),
        "standardized_kurtosis": float(np.mean(standardized**4)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "thresholds": thresholds.tolist(),
        "empirical_two_sided_survival": empirical.tolist(),
        "gaussian_two_sided_survival": gaussian.tolist(),
        "empirical_gaussian_ratio": ratios.tolist(),
    }


def query_risk_metrics(standardized, failure):
    terms = np.log(2.0) - 0.5 * standardized**2
    ordered = np.sort(terms)[::-1]
    values = {
        "min_margin": float(-np.min(standardized)),
        "mean_margin": float(-np.mean(standardized)),
    }
    for count in [1, 3, 5, 10, len(ordered)]:
        key = "all" if count == len(ordered) else str(count)
        values[f"top_{key}_logrisk"] = float(logsumexp(ordered[:count]))
    values["failure"] = int(failure)
    return values


def evaluate(records, model):
    decision_global = []
    decision_conditional = []
    flips = []
    queries_global = []
    queries_conditional = []
    for row in records:
        exact = row["exact_margins"]
        predicted = model["slope"] * row["score_margins"]
        row_flips = predicted <= 0
        global_margins = exact / model["global_scale"]
        conditional_margins = exact / conditional_scale(exact, model)
        failure = np.any(row_flips)
        queries_global.append(query_risk_metrics(global_margins, failure))
        queries_conditional.append(query_risk_metrics(conditional_margins, failure))
        decision_global.append(global_margins)
        decision_conditional.append(conditional_margins)
        flips.append(row_flips)
    return {
        "global_margins": np.concatenate(decision_global),
        "conditional_margins": np.concatenate(decision_conditional),
        "flips": np.concatenate(flips),
        "queries_global": queries_global,
        "queries_conditional": queries_conditional,
    }


def summarize_certificates(query_rows):
    failure = np.array([row["failure"] for row in query_rows], dtype=np.float64)
    keys = [key for key in query_rows[0] if key != "failure"]
    correlations = {
        key: safe_spearman([row[key] for row in query_rows], failure) for key in keys
    }
    best_key = max(correlations, key=correlations.get)
    return {
        "failure_rate": float(failure.mean()),
        "correlations": correlations,
        "best_metric": best_key,
        "best_spearman": correlations[best_key],
    }


def run_quantizer(exact_calibration, exact_evaluation, encoded, bitwidth):
    signs, strongs = encoded
    calibration = add_quantized_margins(exact_calibration, signs, strongs, bitwidth)
    evaluation = add_quantized_margins(exact_evaluation, signs, strongs, bitwidth)
    model = fit_calibration(calibration)
    evaluated = evaluate(evaluation, model)
    score_rho = safe_spearman(flatten(evaluation, "score_margins"), flatten(evaluation, "exact_margins"))
    residual_margin_rho = safe_spearman(model["residual"], model["exact"])
    eligible = bool(model["slope"] > 0 and score_rho >= 0.10 and abs(residual_margin_rho) <= 0.50)
    return {
        "calibration_slope": model["slope"],
        "global_residual_scale": model["global_scale"],
        "conditional_edges": model["edges"].tolist(),
        "conditional_scales": model["conditional_scales"].tolist(),
        "score_margin_spearman": score_rho,
        "residual_margin_spearman": residual_margin_rho,
        "calibration_eligible": eligible,
        "decision_flip_rate": float(evaluated["flips"].mean()),
        "global_margin_flip_spearman": safe_spearman(
            evaluated["global_margins"], evaluated["flips"].astype(np.float64)
        ),
        "conditional_margin_flip_spearman": safe_spearman(
            evaluated["conditional_margins"], evaluated["flips"].astype(np.float64)
        ),
        "tail": residual_tail_metrics(model),
        "global_certificate": summarize_certificates(evaluated["queries_global"]),
        "conditional_certificate": summarize_certificates(
            evaluated["queries_conditional"]
        ),
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
    exact_calibration = build_exact_candidates(vecs, calibration_indices)
    exact_evaluation = build_exact_candidates(vecs, evaluation_indices)
    encodings = {"original": encode(vecs), "rotated": encode(rotated)}
    output = {"dataset": name, "dimension": int(vecs.shape[1]), "quantizers": {}}
    for quantizer in QUANTIZERS:
        rotated_mode = quantizer.startswith("rotated")
        bitwidth = 1 if quantizer.endswith("1bit") else 2
        encoding = encodings["rotated" if rotated_mode else "original"]
        output["quantizers"][quantizer] = run_quantizer(
            exact_calibration, exact_evaluation, encoding, bitwidth
        )
    return output


def save_results(results):
    path = RESULTS_DIR / "probe25_decision_stability_refinement.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            tail_ratio = metrics["tail"]["empirical_gaussian_ratio"][-1]
            global_best = metrics["global_certificate"]
            conditional_best = metrics["conditional_certificate"]
            print(
                f"  {quantizer:<13} eligible={metrics['calibration_eligible']}, "
                f"flip={metrics['decision_flip_rate']:.4f}, "
                f"marginρ={metrics['score_margin_spearman']:.3f}, "
                f"tail@3σ={tail_ratio:.2f}x, "
                f"global={global_best['best_metric']}:{global_best['best_spearman']:.3f}, "
                f"conditional={conditional_best['best_metric']}:{conditional_best['best_spearman']:.3f}"
            )
    save_results(results)


if __name__ == "__main__":
    main()
