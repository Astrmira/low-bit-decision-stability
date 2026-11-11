#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""17 - Probe 34：P6 严格 operational block certificate。"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


DATASETS = [
    "cohere",
    "minilm",
    "bge_m3",
    "wolt_clip",
    "codesearch_jina",
    "landmark_dino",
    "glove100",
    "sift128",
    "gist960",
    "sphere",
    "random",
    "msmarco_cohere",
]
QUANTIZERS = ["2bit", "rotated_2bit"]
POOL_SIZE = 2_000
N_BETA_BLOCKS = 128
N_RISK_BLOCKS = 512
N_VALIDATION_BLOCKS = 512
N_QUANTILE_BLOCKS = 256
N_STRESS_BLOCKS = 512
CANDIDATE_K = 32
MAX_K = 16
ALPHA = 1.2
DELTA_R = 0.05
STRESS_FRACTION = 0.20
TAU_RATIOS = np.array([0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
RETENTION_TARGETS = np.array([0.20, 0.40, 0.60, 0.80, 1.00])
SEED = 20260825


def load_pools(name):
    prefix, dimension, kind, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 34 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    required = min(5 * POOL_SIZE, total)
    if required < 500:
        raise ValueError(f"数据集 {name} 的可用向量不足：{total}")
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED + dimension)
    indices = rng.choice(total, required, replace=False)
    values = np.asarray(mapped[indices], dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    size = required // 5
    names = ["beta", "risk", "validation", "quantile", "stress"]
    pools = {key: values[i * size : (i + 1) * size] for i, key in enumerate(names)}
    return pools, total, kind


def rotate(values, signs):
    output = dct(values * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return output


def encode(values):
    signs = values > 0
    strongs = np.abs(values) > np.abs(values).mean(axis=1, keepdims=True)
    return signs, strongs


def raw_distances(left, right, encoded):
    signs, strongs = encoded
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    agreement = signs[left] == signs[right]
    signed = 2 * agreement.astype(np.int16) - 1
    weights = (1 + strongs[left].astype(np.int16)) * (
        1 + strongs[right].astype(np.int16)
    )
    score = (signed * weights).sum(axis=1, dtype=np.int64)
    return (4 * signs.shape[1] - score).astype(np.float64)


def sample_block(values, rng):
    indices = rng.choice(len(values), CANDIDATE_K + 1, replace=False)
    target = int(indices[0])
    candidates = indices[1:].astype(np.int64)
    similarities = values[candidates] @ values[target]
    order = np.lexsort((candidates, -similarities))
    return target, candidates[order]


def fit_affine(values, encoded, count, seed):
    rng = np.random.default_rng(seed)
    exact = []
    raw = []
    for _ in range(count):
        target, candidates = sample_block(values, rng)
        left = np.full(len(candidates), target, dtype=np.int64)
        exact.extend(1.0 - values[candidates] @ values[target])
        raw.extend(raw_distances(left, candidates, encoded))
    exact = np.asarray(exact, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    design = np.column_stack([np.ones(len(raw)), raw])
    beta, _, _, _ = np.linalg.lstsq(design, exact, rcond=None)
    residual = design @ beta - exact
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "distance_mse": float(np.mean(residual**2)),
        "distance_count": int(len(raw)),
    }


def calibrated(left, right, encoded, model):
    return model["intercept"] + model["slope"] * raw_distances(left, right, encoded)


def exact_pruning_comparisons(values, target, candidates):
    selected = []
    output = []
    for candidate in candidates:
        if len(selected) >= MAX_K:
            break
        exact_tc = float(1.0 - np.dot(values[target], values[candidate]))
        dominated = False
        for selected_node in selected:
            exact_cs = float(1.0 - np.dot(values[candidate], values[selected_node]))
            psi = exact_tc - ALPHA * exact_cs
            output.append((target, int(candidate), selected_node, psi))
            if psi >= 0:
                dominated = True
                break
        if not dominated:
            selected.append(int(candidate))
    return output


def collect_blocks(values, encoded, model, count, seed):
    rng = np.random.default_rng(seed)
    output = {"ranking": [], "pruning": []}
    for _ in range(count):
        target, candidates = sample_block(values, rng)
        top = int(candidates[0])
        competitors = candidates[1:]
        exact_top = float(1.0 - np.dot(values[target], values[top]))
        exact_competitors = 1.0 - values[competitors] @ values[target]
        approximate_top = calibrated(
            np.full(len(competitors), target), np.full(len(competitors), top), encoded, model
        )
        approximate_competitors = calibrated(
            np.full(len(competitors), target), competitors, encoded, model
        )
        exact_role = exact_competitors - exact_top
        approximate_role = approximate_competitors - approximate_top
        ranking_failure = (approximate_role < 0) | (
            (approximate_role == 0) & (competitors < top)
        )
        output["ranking"].append(
            {
                "margin": exact_role.astype(np.float64),
                "residual": (approximate_role - exact_role).astype(np.float64),
                "flip": ranking_failure,
            }
        )
        comparisons = exact_pruning_comparisons(values, target, candidates)
        if not comparisons:
            output["pruning"].append(empty_block())
            continue
        nodes = np.asarray([(t, c, s) for t, c, s, _ in comparisons], dtype=np.int64)
        exact_psi = np.asarray([psi for _, _, _, psi in comparisons], dtype=np.float64)
        approximate_tc = calibrated(nodes[:, 0], nodes[:, 1], encoded, model)
        approximate_cs = calibrated(nodes[:, 1], nodes[:, 2], encoded, model)
        approximate_psi = approximate_tc - ALPHA * approximate_cs
        output["pruning"].append(
            {
                "margin": np.abs(exact_psi),
                "residual": approximate_psi - exact_psi,
                "flip": (exact_psi >= 0) != (approximate_psi >= 0),
            }
        )
    return output


def empty_block():
    return {
        "margin": np.empty(0, dtype=np.float64),
        "residual": np.empty(0, dtype=np.float64),
        "flip": np.empty(0, dtype=bool),
    }


def nonempty(blocks):
    return [block for block in blocks if len(block["margin"]) > 0]


def flatten(blocks, key):
    blocks = nonempty(blocks)
    return np.concatenate([block[key] for block in blocks]) if blocks else np.empty(0)


def block_average(blocks, function):
    blocks = nonempty(blocks)
    return np.asarray([float(np.mean(function(block))) for block in blocks])


def fit_role(blocks):
    blocks = nonempty(blocks)
    residual = flatten(blocks, "residual")
    margin = flatten(blocks, "margin")
    scale = max(float(np.std(residual, ddof=1)), 1e-12)
    positive = margin[margin > 0]
    base_tau = max(float(np.median(positive)), 1e-12)
    taus = (base_tau * TAU_RATIOS).tolist()
    standardized = margin / scale
    cutoffs = np.quantile(standardized, 1 - RETENTION_TARGETS).tolist()
    return {
        "residual_scale": scale,
        "tau_grid": taus,
        "selector_cutoffs": cutoffs,
        "model_fit_block_count": len(blocks),
    }


def certify(blocks, role_model):
    blocks = nonempty(blocks)
    taus = role_model["tau_grid"]
    cutoffs = role_model["selector_cutoffs"]
    scale = role_model["residual_scale"]
    k = len(cutoffs)
    m = len(taus)
    penalty = float(np.sqrt(np.log(k * (m + 1) / DELTA_R) / (2 * len(blocks))))
    candidates = []
    for selector_index, cutoff in enumerate(cutoffs):
        coverage = block_average(
            blocks, lambda block: block["margin"] / scale >= cutoff
        ).mean()
        for tau_index, tau in enumerate(taus):
            union_mass = block_average(
                blocks,
                lambda block: (block["margin"] / scale >= cutoff)
                & (
                    (block["margin"] <= tau)
                    | (np.abs(block["residual"]) >= tau)
                ),
            ).mean()
            certificate = None
            if coverage > penalty:
                certificate = min(1.0, float((union_mass + penalty) / (coverage - penalty)))
            candidates.append(
                {
                    "selector_index": selector_index,
                    "tau_index": tau_index,
                    "target_retention": float(RETENTION_TARGETS[selector_index]),
                    "selector_cutoff": float(cutoff),
                    "tau": float(tau),
                    "calibration_coverage": float(coverage),
                    "calibration_union_mass": float(union_mass),
                    "certificate": certificate,
                }
            )
    feasible = [row for row in candidates if row["certificate"] is not None]
    selected = min(
        feasible,
        key=lambda row: (row["certificate"], -row["calibration_coverage"], row["tau"]),
    ) if feasible else None
    return {"penalty": penalty, "selected": selected, "candidate_count": len(candidates)}


def evaluate(blocks, role_model, certificate):
    blocks = nonempty(blocks)
    if certificate["selected"] is None:
        return {"abstained": True, "reason": "所有 selector 均未通过 denominator gate"}
    selected = certificate["selected"]
    cutoff = selected["selector_cutoff"]
    scale = role_model["residual_scale"]
    accept = lambda block: block["margin"] / scale >= cutoff
    coverage_numerator = block_average(blocks, accept)
    failure_numerator = block_average(
        blocks, lambda block: accept(block) & block["flip"]
    )
    coverage = float(coverage_numerator.mean())
    failure_mass = float(failure_numerator.mean())
    selective_risk = failure_mass / coverage if coverage > 0 else None
    direct_flip = float(block_average(blocks, lambda block: block["flip"]).mean())
    return {
        "abstained": False,
        "block_count": len(blocks),
        "block_uniform_flip_rate": direct_flip,
        "coverage": coverage,
        "failure_mass": failure_mass,
        "selective_risk": selective_risk,
        "certificate": selected["certificate"],
        "covered": bool(selective_risk is not None and selective_risk <= selected["certificate"] + 1e-12),
        "looseness": selected["certificate"] / max(selective_risk, 1e-12),
        "decision_weighted_flip_rate": float(np.mean(flatten(blocks, "flip"))),
    }


def frozen_stress_cutoff(quantile_blocks, scale):
    standardized = flatten(quantile_blocks, "margin") / scale
    return float(np.quantile(standardized, STRESS_FRACTION))


def evaluate_stress(blocks, scale, cutoff):
    blocks = nonempty(blocks)
    accept = lambda block: block["margin"] / scale <= cutoff
    coverage = float(block_average(blocks, accept).mean())
    failure_mass = float(
        block_average(blocks, lambda block: accept(block) & block["flip"]).mean()
    )
    return {
        "frozen_cutoff": cutoff,
        "coverage": coverage,
        "conditional_flip_rate": failure_mass / coverage if coverage > 0 else None,
        "block_count": len(blocks),
    }


def run_quantizer(pools, transformed, quantizer, dimension):
    encodings = {name: encode(values) for name, values in transformed.items()}
    model = fit_affine(
        pools["beta"], encodings["beta"], N_BETA_BLOCKS, SEED + dimension + 10
    )
    eligible = model["slope"] > 0
    if not eligible:
        return {
            "distance_calibration": model,
            "eligibility": {"eligible": False, "reasons": ["校准斜率非正"]},
            "roles": {},
        }
    beta_role_blocks = collect_blocks(
        pools["beta"],
        encodings["beta"],
        model,
        N_BETA_BLOCKS,
        SEED + dimension + 50,
    )
    counts = {
        "risk": N_RISK_BLOCKS,
        "validation": N_VALIDATION_BLOCKS,
        "quantile": N_QUANTILE_BLOCKS,
        "stress": N_STRESS_BLOCKS,
    }
    blocks = {
        split: collect_blocks(
            pools[split], encodings[split], model, count, SEED + dimension + 100 * i
        )
        for i, (split, count) in enumerate(counts.items(), start=1)
    }
    roles = {}
    for role in ("ranking", "pruning"):
        role_model = fit_role(beta_role_blocks[role])
        certificate = certify(blocks["risk"][role], role_model)
        cutoff = frozen_stress_cutoff(blocks["quantile"][role], role_model["residual_scale"])
        roles[role] = {
            "model": role_model,
            "certificate": certificate,
            "validation": evaluate(blocks["validation"][role], role_model, certificate),
            "stress": evaluate_stress(
                blocks["stress"][role], role_model["residual_scale"], cutoff
            ),
        }
    return {
        "distance_calibration": model,
        "eligibility": {"eligible": True, "reasons": []},
        "roles": roles,
    }


def run_dataset(name):
    pools, total, kind = load_pools(name)
    dimension = pools["beta"].shape[1]
    rng = np.random.default_rng(SEED + 31 * dimension)
    rotation_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), dimension)
    transformed_modes = {
        "2bit": pools,
        "rotated_2bit": {key: rotate(values, rotation_signs) for key, values in pools.items()},
    }
    return {
        "dataset": name,
        "kind": kind,
        "dimension": int(dimension),
        "source_total": int(total),
        "pool_size": int(len(pools["beta"])),
        "quantizers": {
            quantizer: run_quantizer(pools, transformed, quantizer, dimension)
            for quantizer, transformed in transformed_modes.items()
        },
    }


def aggregate(results):
    output = {}
    for role in ("ranking", "pruning"):
        records = []
        for dataset in results:
            for quantizer, metrics in dataset["quantizers"].items():
                if not metrics["eligibility"]["eligible"]:
                    continue
                validation = metrics["roles"][role]["validation"]
                records.append(
                    {
                        "dataset": dataset["dataset"],
                        "quantizer": quantizer,
                        "abstained": validation["abstained"],
                        "covered": validation.get("covered"),
                        "selective_risk": validation.get("selective_risk"),
                        "certificate": validation.get("certificate"),
                        "coverage": validation.get("coverage"),
                        "stress_risk": metrics["roles"][role]["stress"]["conditional_flip_rate"],
                    }
                )
        issued = [row for row in records if not row["abstained"]]
        output[role] = {
            "eligible_configuration_count": len(records),
            "issued_count": len(issued),
            "abstention_rate": 1 - len(issued) / max(len(records), 1),
            "issued_coverage_rate": float(np.mean([row["covered"] for row in issued])) if issued else None,
            "mean_selective_risk": float(np.mean([row["selective_risk"] for row in issued])) if issued else None,
            "mean_certificate": float(np.mean([row["certificate"] for row in issued])) if issued else None,
            "mean_retained_coverage": float(np.mean([row["coverage"] for row in issued])) if issued else None,
            "mean_stress_risk": float(np.mean([row["stress_risk"] for row in records])),
            "records": records,
        }
    return output


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            if not metrics["eligibility"]["eligible"]:
                print(f"  {quantizer:<13} abstain=校准斜率非正")
                continue
            ranking = metrics["roles"]["ranking"]["validation"]
            pruning = metrics["roles"]["pruning"]["validation"]
            print(
                f"  {quantizer:<13} rank={ranking.get('selective_risk')}/"
                f"{ranking.get('certificate')}, prune={pruning.get('selective_risk')}/"
                f"{pruning.get('certificate')}"
            )
    output = {
        "probe": 34,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "pool_size": POOL_SIZE,
        "block_counts": {
            "coefficient_fit": N_BETA_BLOCKS,
            "risk_calibration": N_RISK_BLOCKS,
            "validation": N_VALIDATION_BLOCKS,
            "stress_quantile": N_QUANTILE_BLOCKS,
            "stress_evaluation": N_STRESS_BLOCKS,
        },
        "candidate_k": CANDIDATE_K,
        "max_k": MAX_K,
        "alpha": ALPHA,
        "delta_r": DELTA_R,
        "tau_ratios": TAU_RATIOS.tolist(),
        "retention_targets": RETENTION_TARGETS.tolist(),
        "stress_fraction": STRESS_FRACTION,
        "protocol": "five disjoint empirical pools; independent-with-replacement query/call blocks within each pool",
        "estimand": "block-uniform selective risk; decision-weighted flip rate reported separately",
        "status": "operational fitted-rule finite-sample certificate for the empirical-pool block distribution",
        "aggregate": aggregate(results),
        "results": results,
    }
    path = RESULTS_DIR / "probe34_standard_vamana_p6_strict_certificate.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


if __name__ == "__main__":
    main()
