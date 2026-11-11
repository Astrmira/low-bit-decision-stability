#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20 - Probe 37：跨量化器 decision-stability 接口验证。"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.fft import dct
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = ["cohere", "minilm", "sift128", "gist960"]
N_FIT = 1_200
N_EVALUATION = 2_000
N_FIT_PAIRS = 24_000
N_DECISIONS = 60_000
ALPHA = 1.2
SEED = 20260827
QUANTIZERS = ["sign", "shared_2bit", "rabitq", "bbq_like", "scalar_int4", "pq4x16"]


def safe_spearman(left, right):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_split(name):
    prefix, dimension, kind, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 37 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    count = min(N_FIT + N_EVALUATION, total)
    rng = np.random.default_rng(SEED + dimension)
    indices = rng.choice(total, count, replace=False)
    values = np.asarray(mapped[indices], dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values[: min(N_FIT, count // 2)], values[min(N_FIT, count // 2) :], kind, total


def rotate(values, signs):
    return dct(values * signs, type=2, norm="ortho", axis=1).astype(np.float32)


def fit_pq(values, subspaces, centroids, seed):
    blocks = np.array_split(np.arange(values.shape[1]), subspaces)
    codebooks = []
    for index, block in enumerate(blocks):
        rng = np.random.default_rng(seed + index)
        initial = values[rng.choice(len(values), centroids, replace=False)][:, block]
        codebook, _ = kmeans2(values[:, block], initial, iter=8, minit="matrix")
        codebooks.append(codebook.astype(np.float32))
    return blocks, codebooks


def encode_pq(values, blocks, codebooks):
    codes = np.empty((len(values), len(blocks)), dtype=np.int16)
    for index, (block, codebook) in enumerate(zip(blocks, codebooks)):
        distances = np.sum((values[:, None, block] - codebook[None, :, :]) ** 2, axis=2)
        codes[:, index] = np.argmin(distances, axis=1)
    return codes


def build_quantizer(name, fit, evaluation):
    dimension = fit.shape[1]
    if name == "sign":
        fit_codes = np.where(fit >= 0, 1.0, -1.0).astype(np.float32)
        codes = np.where(evaluation >= 0, 1.0, -1.0).astype(np.float32)
        return (
            lambda left, right: np.sum(codes[left] * codes[right], axis=1),
            lambda left, right: np.sum(fit_codes[left] * fit_codes[right], axis=1),
        )
    if name == "shared_2bit":
        def encode_shared(values):
            threshold = np.abs(values).mean(axis=1, keepdims=True)
            return np.where(values >= 0, 1.0, -1.0) * (
                1.0 + (np.abs(values) > threshold)
            )

        fit_codes = encode_shared(fit)
        codes = encode_shared(evaluation)
        return (
            lambda left, right: np.sum(codes[left] * codes[right], axis=1),
            lambda left, right: np.sum(fit_codes[left] * fit_codes[right], axis=1),
        )
    if name == "rabitq":
        rng = np.random.default_rng(SEED + dimension * 7)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), dimension)
        fit_transformed = rotate(fit, signs)
        transformed = rotate(evaluation, signs)
        fit_code = np.where(fit_transformed >= 0, 1.0, -1.0).astype(np.float32) / np.sqrt(
            dimension
        )
        code = np.where(transformed >= 0, 1.0, -1.0).astype(np.float32) / np.sqrt(dimension)
        fit_denominator = np.sum(fit_code * fit_transformed, axis=1)
        denominator = np.sum(code * transformed, axis=1)

        def score(left, right):
            numerator = np.sum(transformed[left] * code[right], axis=1)
            return numerator / np.maximum(denominator[right], 1e-8)

        def fit_score(left, right):
            numerator = np.sum(fit_transformed[left] * fit_code[right], axis=1)
            return numerator / np.maximum(fit_denominator[right], 1e-8)

        return score, fit_score
    if name == "bbq_like":
        centroid = fit.mean(axis=0)

        def encode_bbq(values):
            residual = values - centroid
            residual_norm = np.linalg.norm(residual, axis=1)
            stored = np.where(residual >= 0, 1.0, -1.0).astype(np.float32)
            stored *= (residual_norm / np.sqrt(dimension))[:, None]
            minimum = values.min(axis=1, keepdims=True)
            width = np.maximum(values.max(axis=1, keepdims=True) - minimum, 1e-8)
            query_code = np.rint((values - minimum) * 15.0 / width)
            query = minimum + query_code * width / 15.0
            return stored, query

        fit_stored, fit_query = encode_bbq(fit)
        stored, query = encode_bbq(evaluation)

        def score(left, right):
            return np.sum(query[left] * stored[right], axis=1) + query[left] @ centroid

        def fit_score(left, right):
            return np.sum(fit_query[left] * fit_stored[right], axis=1) + fit_query[left] @ centroid

        return score, fit_score
    if name == "scalar_int4":
        lower = np.quantile(fit, 0.005, axis=0)
        upper = np.quantile(fit, 0.995, axis=0)
        width = np.maximum(upper - lower, 1e-8)

        def reconstruct(values):
            code = np.rint(np.clip((values - lower) * 15.0 / width, 0, 15))
            output = lower + code * width / 15.0
            return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)

        fit_reconstructed = reconstruct(fit)
        reconstructed = reconstruct(evaluation)
        return (
            lambda left, right: np.sum(reconstructed[left] * reconstructed[right], axis=1),
            lambda left, right: np.sum(
                fit_reconstructed[left] * fit_reconstructed[right], axis=1
            ),
        )
    if name == "pq4x16":
        subspaces = min(16, dimension)
        blocks, codebooks = fit_pq(fit, subspaces, 16, SEED + dimension)
        fit_codes = encode_pq(fit, blocks, codebooks)
        codes = encode_pq(evaluation, blocks, codebooks)

        def score(left, right):
            distance = np.zeros(len(left), dtype=np.float64)
            for index, (block, codebook) in enumerate(zip(blocks, codebooks)):
                reconstructed = codebook[codes[right, index]]
                distance += np.sum((evaluation[left][:, block] - reconstructed) ** 2, axis=1)
            return -distance

        def fit_score(left, right):
            distance = np.zeros(len(left), dtype=np.float64)
            for index, (block, codebook) in enumerate(zip(blocks, codebooks)):
                reconstructed = codebook[fit_codes[right, index]]
                distance += np.sum((fit[left][:, block] - reconstructed) ** 2, axis=1)
            return -distance

        return score, fit_score
    raise ValueError(name)


def pair_indices(size, count, seed):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, size, (count, 2))
    same = pairs[:, 0] == pairs[:, 1]
    pairs[same, 1] = (pairs[same, 1] + 1) % size
    return pairs[:, 0], pairs[:, 1]


def fit_calibration(values, score, seed):
    left, right = pair_indices(len(values), N_FIT_PAIRS, seed)
    exact = np.sum(values[left] * values[right], axis=1)
    raw = score(left, right)
    design = np.column_stack([np.ones(len(raw)), raw])
    beta, _, _, _ = np.linalg.lstsq(design, exact, rcond=None)
    return float(beta[0]), float(beta[1]), safe_spearman(raw, exact)


def triples(size, count, seed):
    rng = np.random.default_rng(seed)
    nodes = rng.integers(0, size, (count, 3))
    invalid = (nodes[:, 0] == nodes[:, 1]) | (nodes[:, 0] == nodes[:, 2]) | (
        nodes[:, 1] == nodes[:, 2]
    )
    nodes[invalid, 2] = (nodes[invalid, 2] + 1) % size
    return nodes


def role_summary(exact, approximate):
    residual = approximate - exact
    margin = np.abs(exact)
    flip = (exact >= 0) != (approximate >= 0)
    hard = margin <= np.quantile(margin, 0.20)
    scale = max(float(np.std(residual, ddof=1)), 1e-12)
    standardized = margin / scale
    bins = np.quantile(standardized, np.linspace(0, 1, 6))
    bin_flip = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (standardized >= lower) & (standardized <= upper)
        bin_flip.append(float(np.mean(flip[selected])))
    tau_grid = np.unique(np.quantile(margin[margin > 0], [0.02, 0.05, 0.1, 0.2, 0.4]))
    p1_risks = [
        float(np.mean(margin <= tau) + np.mean(np.abs(residual) >= tau))
        for tau in tau_grid
    ]
    return {
        "count": int(len(exact)),
        "flip_rate": float(np.mean(flip)),
        "hard_20_flip_rate": float(np.mean(flip[hard])),
        "residual_sd": scale,
        "residual_abs_p99": float(np.quantile(np.abs(residual), 0.99)),
        "margin_to_flip_spearman": safe_spearman(standardized, flip.astype(float)),
        "standardized_margin_bin_flip": bin_flip,
        "minimum_empirical_p1_risk": min(p1_risks),
        "p1_empirical_coverage": bool(np.mean(flip) <= min(p1_risks) + 1e-12),
    }


def evaluate(values, score, calibration, seed):
    intercept, slope, distance_spearman = calibration

    def calibrated(left, right):
        return intercept + slope * score(left, right)

    nodes = triples(len(values), N_DECISIONS, seed)
    q, x, y = nodes.T
    exact_qx = np.sum(values[q] * values[x], axis=1)
    exact_qy = np.sum(values[q] * values[y], axis=1)
    approx_qx = calibrated(q, x)
    approx_qy = calibrated(q, y)
    ranking_exact = exact_qx - exact_qy
    ranking_approximate = approx_qx - approx_qy
    exact_tc = 1.0 - exact_qx
    exact_cs = 1.0 - np.sum(values[x] * values[y], axis=1)
    approx_tc = 1.0 - approx_qx
    approx_cs = 1.0 - calibrated(x, y)
    pruning_exact = exact_tc - ALPHA * exact_cs
    pruning_approximate = approx_tc - ALPHA * approx_cs
    return {
        "calibration": {
            "intercept": intercept,
            "slope": slope,
            "edge_spearman": distance_spearman,
            "eligible_positive_slope": slope > 0,
        },
        "ranking": role_summary(ranking_exact, ranking_approximate),
        "pruning": role_summary(pruning_exact, pruning_approximate),
    }


def main():
    results = []
    for dataset in DATASETS:
        fit, evaluation, kind, source_total = load_split(dataset)
        print(f"\n{'=' * 72}\n数据集：{dataset} ({evaluation.shape[1]}d)\n{'=' * 72}")
        records = []
        for index, quantizer in enumerate(QUANTIZERS):
            score, fit_score = build_quantizer(quantizer, fit, evaluation)
            calibration = fit_calibration(
                fit,
                fit_score,
                SEED + evaluation.shape[1] * 19 + index,
            )
            record = {"quantizer": quantizer} | evaluate(
                evaluation, score, calibration, SEED + index * 101 + evaluation.shape[1]
            )
            records.append(record)
            print(
                f"  {quantizer:<14} edgeρ={record['calibration']['edge_spearman']:.3f} "
                f"rank={record['ranking']['flip_rate']:.3f} "
                f"prune={record['pruning']['flip_rate']:.3f}"
            )
        results.append(
            {
                "dataset": dataset,
                "kind": kind,
                "dimension": int(evaluation.shape[1]),
                "source_total": int(source_total),
                "fit_size": int(len(fit)),
                "evaluation_size": int(len(evaluation)),
                "quantizers": records,
            }
        )
    output = {
        "probe": 37,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "results": results,
        "scope": {
            "rabitq": "ratio estimator with seeded orthogonal DCT transform",
            "bbq_like": "centroid-centered 1-bit stored and int4 query reference scorer, not Lucene clone",
            "pq4x16": "16-subspace 4-bit ADC reference estimator",
            "claim": "decision-interface validation, not production-system reproduction",
        },
    }
    path = RESULTS_DIR / "probe37_standard_vamana_cross_quantizer_validation.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
