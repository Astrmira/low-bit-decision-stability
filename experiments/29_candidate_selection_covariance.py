#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""29 - Probe 46：iid、局部候选与 top-k 选择下的共享查询协方差。"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT

DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["1bit", "2bit", "rotated_2bit"]
N_SAMPLE = 6_000
N_FIT_PAIRS = 60_000
N_QUERIES = 160
PAIRS_PER_QUERY = 128
TOP_K = 32
LOCAL_POOL_K = 256
SEED = 20260908


def load_values(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 46 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED + dimension)
    indices = rng.choice(total, min(N_SAMPLE, total), replace=False)
    values = np.asarray(mapped[indices], dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values, int(total)


def rotate(values):
    rng = np.random.default_rng(SEED + 31 * values.shape[1])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), values.shape[1])
    output = dct(values * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return output


def encode(values):
    return values > 0, np.abs(values) > np.abs(values).mean(axis=1, keepdims=True)


def approximate_scores(left, right, encoded, bitwidth):
    signs, strongs = encoded
    agreement = signs[left] == signs[right]
    signed = 2 * agreement.astype(np.int16) - 1
    if bitwidth == 1:
        return signed.sum(axis=1, dtype=np.int64).astype(np.float64)
    weights = (1 + strongs[left].astype(np.int16)) * (1 + strongs[right].astype(np.int16))
    return (signed * weights).sum(axis=1, dtype=np.int64).astype(np.float64)


def fit_affine(values, encoded, bitwidth, rng):
    left = rng.integers(0, len(values), N_FIT_PAIRS)
    right = rng.integers(0, len(values), N_FIT_PAIRS)
    same = left == right
    right[same] = (right[same] + 1) % len(values)
    exact = np.sum(values[left] * values[right], axis=1, dtype=np.float64)
    approximate = approximate_scores(left, right, encoded, bitwidth)
    design = np.column_stack([np.ones(len(approximate)), approximate])
    intercept, slope = np.linalg.lstsq(design, exact, rcond=None)[0]
    return {"intercept": float(intercept), "slope": float(slope)}


def edge_residual(values, left, right, encoded, bitwidth, model):
    exact = np.sum(values[left] * values[right], axis=1, dtype=np.float64)
    approximate = model["intercept"] + model["slope"] * approximate_scores(
        left, right, encoded, bitwidth
    )
    return approximate - exact


def bootstrap_query_intervals(left_residual, right_residual, query_ids, seed, repeats=400):
    """按查询 block 重采样 pooled correlation 与 independence ratio。"""
    unique = np.unique(query_ids)
    if len(unique) < 2:
        return None
    grouped = [np.flatnonzero(query_ids == query_id) for query_id in unique]
    rng = np.random.default_rng(seed)
    correlations = []
    ratios = []
    for _ in range(repeats):
        sampled = rng.integers(0, len(grouped), len(grouped))
        indices = np.concatenate([grouped[index] for index in sampled])
        left = left_residual[indices]
        right = right_residual[indices]
        covariance = np.cov(left, right, ddof=1)
        correlation = float(
            covariance[0, 1]
            / max(np.sqrt(covariance[0, 0] * covariance[1, 1]), 1e-12)
        )
        independence = float(covariance[0, 0] + covariance[1, 1])
        difference = float(np.var(left - right, ddof=1))
        correlations.append(correlation)
        ratios.append(independence / max(difference, 1e-12))
    return {
        "repeats": repeats,
        "correlation_95_interval": np.quantile(correlations, [0.025, 0.975]).tolist(),
        "independence_ratio_95_interval": np.quantile(ratios, [0.025, 0.975]).tolist(),
    }


def pair_metrics(left_residual, right_residual, query_ids, bootstrap_seed=None, fixed_anchor=False):
    covariance = np.cov(left_residual, right_residual, ddof=1)
    difference = left_residual - right_residual
    pooled_covariance = float(covariance[0, 1])
    left_variance = float(covariance[0, 0])
    right_variance = float(covariance[1, 1])
    correlation = pooled_covariance / max(np.sqrt(left_variance * right_variance), 1e-12)
    independence_variance = left_variance + right_variance
    difference_variance = float(np.var(difference, ddof=1))

    unique = np.unique(query_ids)
    group_indices = [np.flatnonzero(query_ids == query_id) for query_id in unique]
    total_count = len(left_residual)
    within_cross_product = 0.0
    between_cross_product = 0.0
    within_left_cross_product = 0.0
    between_left_cross_product = 0.0
    within_right_cross_product = 0.0
    between_right_cross_product = 0.0
    grand_left = float(np.mean(left_residual))
    grand_right = float(np.mean(right_residual))
    within_covariances = []
    query_means = []
    for indices in group_indices:
        left = left_residual[indices]
        right = right_residual[indices]
        mean_left = float(np.mean(left))
        mean_right = float(np.mean(right))
        query_means.append((mean_left, mean_right))
        centered_left = left - mean_left
        centered_right = right - mean_right
        within_cross_product += float(np.dot(centered_left, centered_right))
        within_left_cross_product += float(np.dot(centered_left, centered_left))
        within_right_cross_product += float(np.dot(centered_right, centered_right))
        between_cross_product += len(indices) * (mean_left - grand_left) * (mean_right - grand_right)
        between_left_cross_product += len(indices) * (mean_left - grand_left) ** 2
        between_right_cross_product += len(indices) * (mean_right - grand_right) ** 2
        if len(indices) > 1:
            within_covariances.append(float(np.cov(left, right, ddof=1)[0, 1]))

    denominator = max(total_count - 1, 1)
    exact_within_covariance = within_cross_product / denominator
    exact_between_covariance = between_cross_product / denominator
    exact_within_left_variance = within_left_cross_product / denominator
    exact_between_left_variance = between_left_cross_product / denominator
    exact_within_right_variance = within_right_cross_product / denominator
    exact_between_right_variance = between_right_cross_product / denominator
    exact_difference_variance = (
        exact_within_left_variance + exact_between_left_variance
        + exact_within_right_variance + exact_between_right_variance
        - 2 * (exact_within_covariance + exact_between_covariance)
    )
    query_means = np.asarray(query_means, dtype=np.float64)
    query_covariances = np.asarray(within_covariances, dtype=np.float64)
    return {
        "pair_count": int(total_count),
        "query_count": int(len(unique)),
        "left_variance": left_variance,
        "right_variance": right_variance,
        "pooled_covariance": pooled_covariance,
        "pooled_correlation": float(correlation),
        "difference_variance": difference_variance,
        "independence_variance": independence_variance,
        "independence_ratio": float(independence_variance / max(difference_variance, 1e-12)),
        "exact_weighted_within_covariance": exact_within_covariance,
        "exact_weighted_between_covariance": exact_between_covariance,
        "exact_covariance_identity_gap": float(
            pooled_covariance - exact_within_covariance - exact_between_covariance
        ),
        "exact_difference_variance_from_components": exact_difference_variance,
        "exact_difference_identity_gap": float(difference_variance - exact_difference_variance),
        "within_query_correlation": None if fixed_anchor else float(
            exact_within_covariance / max(
                np.sqrt(exact_within_left_variance * exact_within_right_variance), 1e-12
            )
        ),
        "query_mean_covariance": exact_between_covariance,
        "query_mean_left_sd": float(np.std(query_means[:, 0], ddof=1)) if len(query_means) > 1 else 0.0,
        "query_mean_right_sd": float(np.std(query_means[:, 1], ddof=1)) if len(query_means) > 1 else 0.0,
        "query_within_covariance_quantiles": (
            np.quantile(query_covariances, [0.05, 0.5, 0.95]).tolist()
            if len(query_covariances) else None
        ),
        "query_block_bootstrap": (
            bootstrap_query_intervals(left_residual, right_residual, query_ids, bootstrap_seed)
            if bootstrap_seed is not None else None
        ),
    }


def sample_pairs(pool, count, rng):
    left = rng.choice(pool, count, replace=True)
    right = rng.choice(pool, count, replace=True)
    same = left == right
    while np.any(same):
        right[same] = rng.choice(pool, np.sum(same), replace=True)
        same = left == right
    return left.astype(np.int64), right.astype(np.int64)


def collect_regimes(values, query_indices, encoded, bitwidth, model, rng):
    similarities = values[query_indices] @ values.T
    records = {name: [[], [], []] for name in (
        "iid_with_replacement", "distinct_random_triples", "shared_query_random",
        "local_top256", "selected_top32", "top_anchor"
    )}
    all_indices = np.arange(len(values), dtype=np.int64)
    for row_index, query in enumerate(query_indices):
        similarities[row_index, query] = -np.inf
        order256 = np.argpartition(-similarities[row_index], LOCAL_POOL_K)[:LOCAL_POOL_K]
        order256 = order256[np.lexsort((order256, -similarities[row_index, order256]))]
        top32 = order256[:TOP_K]
        random_pool = all_indices[all_indices != query]

        regimes = {
            "shared_query_random": sample_pairs(random_pool, PAIRS_PER_QUERY, rng),
            "local_top256": sample_pairs(order256, PAIRS_PER_QUERY, rng),
            "selected_top32": sample_pairs(top32, PAIRS_PER_QUERY, rng),
        }
        anchor_left = np.full(TOP_K - 1, top32[0], dtype=np.int64)
        anchor_right = top32[1:].astype(np.int64)
        regimes["top_anchor"] = (anchor_left, anchor_right)

        for name, (left, right) in regimes.items():
            query_array = np.full(len(left), query, dtype=np.int64)
            left_residual = edge_residual(values, query_array, left, encoded, bitwidth, model)
            right_residual = edge_residual(values, query_array, right, encoded, bitwidth, model)
            records[name][0].append(left_residual)
            records[name][1].append(right_residual)
            records[name][2].append(np.full(len(left), row_index, dtype=np.int64))

    iid_count = len(query_indices) * PAIRS_PER_QUERY
    iid_query = rng.choice(all_indices, iid_count, replace=True)
    iid_left = rng.choice(all_indices, iid_count, replace=True)
    iid_right = rng.choice(all_indices, iid_count, replace=True)
    iid_left_residual = edge_residual(values, iid_query, iid_left, encoded, bitwidth, model)
    iid_right_residual = edge_residual(values, iid_query, iid_right, encoded, bitwidth, model)
    records["iid_with_replacement"] = [
        [iid_left_residual], [iid_right_residual], [np.arange(iid_count, dtype=np.int64)]
    ]

    distinct_query = iid_query.copy()
    distinct_left = iid_left.copy()
    distinct_right = iid_right.copy()
    for candidates in (distinct_left, distinct_right):
        same = candidates == distinct_query
        while np.any(same):
            candidates[same] = rng.choice(all_indices, np.sum(same), replace=True)
            same = candidates == distinct_query
    same = distinct_left == distinct_right
    while np.any(same):
        distinct_right[same] = rng.choice(all_indices, np.sum(same), replace=True)
        same = (distinct_left == distinct_right) | (distinct_right == distinct_query)
    left_residual = edge_residual(values, distinct_query, distinct_left, encoded, bitwidth, model)
    right_residual = edge_residual(values, distinct_query, distinct_right, encoded, bitwidth, model)
    records["distinct_random_triples"] = [
        [left_residual], [right_residual], [np.arange(iid_count, dtype=np.int64)]
    ]

    output = {}
    for name, (left_parts, right_parts, query_parts) in records.items():
        left = np.concatenate(left_parts)
        right = np.concatenate(right_parts)
        query_ids = np.concatenate(query_parts)
        metrics = pair_metrics(
            left,
            right,
            query_ids,
            bootstrap_seed=(SEED + sum(ord(character) for character in name))
            if name not in ("iid_with_replacement", "distinct_random_triples") else None,
            fixed_anchor=name == "top_anchor",
        )
        output[name] = metrics
    return output


def run_dataset(name):
    values, total = load_values(name)
    transformed = {"original": values, "rotated": rotate(values)}
    encodings = {key: encode(item) for key, item in transformed.items()}
    rng = np.random.default_rng(SEED + values.shape[1])
    query_indices = rng.choice(len(values), N_QUERIES, replace=False)
    quantizers = {}
    for quantizer in QUANTIZERS:
        mode = "rotated" if quantizer.startswith("rotated") else "original"
        bitwidth = 1 if quantizer.endswith("1bit") else 2
        model = fit_affine(values, encodings[mode], bitwidth, rng)
        quantizers[quantizer] = {
            "calibration": model,
            "regimes": collect_regimes(
                values, query_indices, encodings[mode], bitwidth, model, rng
            ),
        }
    return {
        "dataset": name,
        "dimension": int(values.shape[1]),
        "source_total": total,
        "quantizers": quantizers,
    }


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}", flush=True)
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            summary = metrics["regimes"]
            print(
                f"  {quantizer:<13} iid={summary['iid_with_replacement']['pooled_correlation']:.3f}, "
                f"distinct={summary['distinct_random_triples']['pooled_correlation']:.3f}, "
                f"random={summary['shared_query_random']['pooled_correlation']:.3f}, "
                f"local={summary['local_top256']['pooled_correlation']:.3f}, "
                f"top32={summary['selected_top32']['pooled_correlation']:.3f}, "
                f"anchor={summary['top_anchor']['pooled_correlation']:.3f}",
                flush=True,
            )
    output = {
        "probe": 46,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "n_sample": N_SAMPLE,
        "n_fit_pairs": N_FIT_PAIRS,
        "n_queries": N_QUERIES,
        "pairs_per_query": PAIRS_PER_QUERY,
        "top_k": TOP_K,
        "local_pool_k": LOCAL_POOL_K,
        "probability_scope": "empirical data distribution with a shared globally fitted edge scorer",
        "status": "mechanism comparison; covariance is empirical and is not an MGF certificate",
        "results": results,
    }
    path = RESULTS_DIR / "probe46_candidate_selection_covariance.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {path}", flush=True)


if __name__ == "__main__":
    main()
