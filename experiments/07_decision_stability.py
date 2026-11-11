#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07 - 量化决策稳定性首轮证伪实验

实验 A：标准化边界 Γ/v 与决策翻转率
实验 B：边界质量 small-ball 幂律拟合
实验 C：查询级固定决策集证书校准
实验 D：基于证书的量化器选择
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["1bit", "2bit"]
N_SAMPLE = 4_000
N_CALIBRATION_QUERIES = 80
N_EVALUATION_QUERIES = 160
LOCAL_K = 32
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
        raise ValueError(f"首轮实验只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED)
    indices = rng.choice(total, min(N_SAMPLE, total), replace=False)
    vecs = np.asarray(mapped[indices], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, 1e-12)
    print(f"  [{name}] 直接映射抽样：{total} → {len(vecs)}")
    return vecs


def encode(vecs):
    signs = (vecs > 0).astype(np.uint8)
    thresholds = np.abs(vecs).mean(axis=1, keepdims=True)
    strongs = (np.abs(vecs) > thresholds).astype(np.uint8)
    return signs, strongs


def local_scores(query_index, candidate_indices, signs, strongs, quantizer):
    sign_agree = signs[candidate_indices] == signs[query_index]
    if quantizer == "1bit":
        return (2.0 * sign_agree.astype(np.float64) - 1.0).sum(axis=1)
    weights = (1.0 + strongs[candidate_indices]) * (1.0 + strongs[query_index])
    return ((2.0 * sign_agree.astype(np.float64) - 1.0) * weights).sum(axis=1)


def collect_query_records(vecs, query_indices, signs, strongs, quantizer):
    similarities = vecs[query_indices] @ vecs.T
    rows = []
    for row_index, query_index in enumerate(query_indices):
        similarities[row_index, query_index] = -np.inf
        candidate_indices = np.argpartition(
            -similarities[row_index], LOCAL_K
        )[:LOCAL_K]
        order = np.argsort(-similarities[row_index, candidate_indices])
        candidate_indices = candidate_indices[order]
        exact = similarities[row_index, candidate_indices]
        scores = local_scores(
            query_index, candidate_indices, signs, strongs, quantizer
        )
        exact_margins = exact[0] - exact[1:]
        score_margins = scores[0] - scores[1:]
        rows.append(
            {
                "query_index": int(query_index),
                "exact_margins": exact_margins,
                "score_margins": score_margins,
                "top1_correct": bool(np.argmax(scores) == 0),
            }
        )
    return rows


def fit_calibration(records):
    exact = np.concatenate([row["exact_margins"] for row in records])
    score = np.concatenate([row["score_margins"] for row in records])
    denominator = float(np.dot(score, score))
    slope = float(np.dot(score, exact) / denominator) if denominator > 0 else 0.0
    residual = slope * score - exact
    scale = float(np.std(residual, ddof=1))
    scale = max(scale, 1e-12)
    return slope, scale, exact, residual


def fit_boundary_exponent(margins):
    positive = margins[margins > 0]
    quantiles = np.array([0.02, 0.04, 0.08, 0.12, 0.18, 0.25])
    taus = np.quantile(positive, quantiles)
    masses = np.array([(positive <= tau).mean() for tau in taus])
    valid = (taus > 0) & (masses > 0)
    design = np.column_stack([np.log(taus[valid]), np.ones(valid.sum())])
    beta, intercept = np.linalg.lstsq(
        design, np.log(masses[valid]), rcond=None
    )[0]
    prediction = design @ np.array([beta, intercept])
    target = np.log(masses[valid])
    total = np.sum((target - target.mean()) ** 2)
    r_squared = 1.0 - np.sum((target - prediction) ** 2) / max(total, 1e-12)
    return float(beta), float(r_squared), taus, masses


def evaluate_records(records, slope, scale):
    all_margins = []
    all_standardized = []
    all_flips = []
    query_results = []
    for row in records:
        exact = row["exact_margins"]
        predicted = slope * row["score_margins"]
        standardized = exact / scale
        flips = predicted <= 0
        log_terms = np.log(2.0) - 0.5 * standardized**2
        log_certificate = float(logsumexp(log_terms))
        certificate = float(min(1.0, np.exp(min(log_certificate, 0.0))))
        query_results.append(
            {
                "query_index": row["query_index"],
                "failure": int(np.any(flips)),
                "top1_correct": int(row["top1_correct"]),
                "certificate": certificate,
                "log_certificate": log_certificate,
                "min_standardized_margin": float(np.min(standardized)),
                "mean_standardized_margin": float(np.mean(standardized)),
            }
        )
        all_margins.append(exact)
        all_standardized.append(standardized)
        all_flips.append(flips)
    return (
        np.concatenate(all_margins),
        np.concatenate(all_standardized),
        np.concatenate(all_flips),
        query_results,
    )


def binned_flip_rates(standardized, flips):
    edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf])
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (standardized >= left) & (standardized < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "left": float(left),
                "right": None if np.isinf(right) else float(right),
                "count": int(mask.sum()),
                "flip_rate": float(flips[mask].mean()),
                "bound_at_left": float(min(1.0, 2.0 * np.exp(-0.5 * left**2))),
            }
        )
    return rows


def certificate_metrics(query_results):
    failure = np.array([row["failure"] for row in query_results], dtype=np.float64)
    risk = -np.array(
        [row["min_standardized_margin"] for row in query_results], dtype=np.float64
    )
    certificate = np.array(
        [row["certificate"] for row in query_results], dtype=np.float64
    )
    log_certificate = np.array(
        [row["log_certificate"] for row in query_results], dtype=np.float64
    )
    rho_risk = safe_spearman(risk, failure)
    rho_certificate = safe_spearman(certificate, failure)
    rho_log_certificate = safe_spearman(log_certificate, failure)
    brier = float(np.mean((certificate - failure) ** 2))
    return {
        "query_failure_rate": float(failure.mean()),
        "risk_failure_spearman": rho_risk,
        "certificate_failure_spearman": rho_certificate,
        "log_certificate_failure_spearman": rho_log_certificate,
        "certificate_brier": brier,
        "certificate_saturation": float(np.mean(certificate >= 1.0 - 1e-12)),
    }


def run_dataset(name):
    vecs = load_sampled_dataset(name)
    signs, strongs = encode(vecs)
    rng = np.random.default_rng(SEED)
    query_indices = rng.choice(
        len(vecs), N_CALIBRATION_QUERIES + N_EVALUATION_QUERIES, replace=False
    )
    calibration_indices = query_indices[:N_CALIBRATION_QUERIES]
    evaluation_indices = query_indices[N_CALIBRATION_QUERIES:]
    output = {"dataset": name, "dimension": int(vecs.shape[1]), "quantizers": {}}
    query_by_quantizer = {}
    for quantizer in QUANTIZERS:
        calibration_records = collect_query_records(
            vecs, calibration_indices, signs, strongs, quantizer
        )
        slope, scale, calibration_margins, calibration_residual = fit_calibration(
            calibration_records
        )
        evaluation_records = collect_query_records(
            vecs, evaluation_indices, signs, strongs, quantizer
        )
        margins, standardized, flips, query_results = evaluate_records(
            evaluation_records, slope, scale
        )
        beta, beta_r2, taus, masses = fit_boundary_exponent(margins)
        score_exact_rho = safe_spearman(
            np.concatenate([row["score_margins"] for row in evaluation_records]),
            margins,
        )
        residual_margin_rho = safe_spearman(calibration_residual, calibration_margins)
        standardized_flip_rho = safe_spearman(standardized, flips.astype(np.float64))
        metrics = certificate_metrics(query_results)
        output["quantizers"][quantizer] = {
            "calibration_slope": slope,
            "residual_scale": scale,
            "calibration_residual_kurtosis": float(
                np.mean(
                    ((calibration_residual - calibration_residual.mean()) / scale) ** 4
                )
            ),
            "calibration_residual_mean_over_scale": float(
                calibration_residual.mean() / scale
            ),
            "calibration_residual_margin_spearman": residual_margin_rho,
            "calibration_eligible": bool(
                slope > 0 and score_exact_rho >= 0.10 and abs(residual_margin_rho) <= 0.50
            ),
            "decision_count": int(len(margins)),
            "decision_flip_rate": float(flips.mean()),
            "score_margin_spearman": float(score_exact_rho),
            "standardized_margin_flip_spearman": standardized_flip_rho,
            "boundary_beta": beta,
            "boundary_r_squared": beta_r2,
            "boundary_taus": taus.tolist(),
            "boundary_masses": masses.tolist(),
            "binned_flip_rates": binned_flip_rates(standardized, flips),
            **metrics,
        }
        query_by_quantizer[quantizer] = query_results
    selected_failures = []
    abstained = 0
    for one_bit, two_bit in zip(
        query_by_quantizer["1bit"], query_by_quantizer["2bit"]
    ):
        eligible = [
            (name, row)
            for name, row in (("1bit", one_bit), ("2bit", two_bit))
            if output["quantizers"][name]["calibration_eligible"]
        ]
        if not eligible:
            abstained += 1
            continue
        _, selected = min(eligible, key=lambda item: item[1]["log_certificate"])
        selected_failures.append(selected["failure"])
    output["selector"] = {
        "certificate_selected_failure_rate": (
            float(np.mean(selected_failures)) if selected_failures else None
        ),
        "abstention_rate": float(abstained / len(query_by_quantizer["1bit"])),
        "always_1bit_failure_rate": output["quantizers"]["1bit"]["query_failure_rate"],
        "always_2bit_failure_rate": output["quantizers"]["2bit"]["query_failure_rate"],
    }
    return output


def save_outputs(results):
    json_path = RESULTS_DIR / "probe24_decision_stability.json"
    csv_path = RESULTS_DIR / "probe24_decision_stability.csv"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "dimension",
                "quantizer",
                "decision_flip_rate",
                "query_failure_rate",
                "score_margin_spearman",
                "boundary_beta",
                "boundary_r_squared",
                "residual_scale",
                "risk_failure_spearman",
                "certificate_failure_spearman",
                "certificate_saturation",
                        "calibration_eligible",
            ],
        )
        writer.writeheader()
        for result in results:
            for quantizer, metrics in result["quantizers"].items():
                writer.writerow(
                    {
                        "dataset": result["dataset"],
                        "dimension": result["dimension"],
                        "quantizer": quantizer,
                        **{key: metrics[key] for key in writer.fieldnames[3:]},
                    }
                )
    print(f"  已保存 {json_path}")
    print(f"  已保存 {csv_path}")


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            print(
                f"  {quantizer}: 翻转率={metrics['decision_flip_rate']:.4f}, "
                f"查询失败率={metrics['query_failure_rate']:.4f}, "
                f"β={metrics['boundary_beta']:.3f}, "
                f"marginρ={metrics['score_margin_spearman']:.3f}, "
                f"riskρ={metrics['risk_failure_spearman']:.3f}"
            )
        print(
            f"  证书选择失败率={result['selector']['certificate_selected_failure_rate']}, "
            f"弃权率={result['selector']['abstention_rate']:.4f}, "
            f"固定2bit={result['selector']['always_2bit_failure_rate']:.4f}"
        )
    save_outputs(results)


if __name__ == "__main__":
    main()
