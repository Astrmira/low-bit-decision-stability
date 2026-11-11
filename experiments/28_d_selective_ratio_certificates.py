#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""28 - Probe 45：D1--D3 选择性 block-ratio 证书对照。"""

import importlib
import json
from pathlib import Path

import numpy as np

base = importlib.import_module("17_p6_strict_certificate")

DATASETS = base.DATASETS
QUANTIZERS = base.QUANTIZERS
SAMPLE_SIZES = [128, 256, 512, 1024]
MAX_RISK_BLOCKS = max(SAMPLE_SIZES)
N_VALIDATION_BLOCKS = 1024
DELTA = 0.05
SEED = 20260907
METHODS = ["double_hoeffding", "direct_union", "direct_failure", "empirical_bernstein_union"]


def block_triplets(blocks, role_model):
    """返回每个 selector/tau 对应的 C_i、U_i，以及每个 selector 的 F_i。"""
    cutoffs = role_model["selector_cutoffs"]
    taus = role_model["tau_grid"]
    scale = role_model["residual_scale"]
    count = len(blocks)
    coverage = np.zeros((count, len(cutoffs)), dtype=np.float64)
    failure = np.zeros_like(coverage)
    union = np.zeros((count, len(cutoffs), len(taus)), dtype=np.float64)
    invariant_violations = 0
    for block_index, block in enumerate(blocks):
        if len(block["margin"]) == 0:
            continue
        standardized = block["margin"] / scale
        for selector_index, cutoff in enumerate(cutoffs):
            accepted = standardized >= cutoff
            coverage[block_index, selector_index] = np.mean(accepted)
            failure[block_index, selector_index] = np.mean(accepted & block["flip"])
            for tau_index, tau in enumerate(taus):
                event = accepted & (
                    (block["margin"] <= tau)
                    | (np.abs(block["residual"]) >= tau)
                )
                union[block_index, selector_index, tau_index] = np.mean(event)
                if failure[block_index, selector_index] > union[block_index, selector_index, tau_index] + 1e-12:
                    invariant_violations += 1
    return coverage, union, failure, invariant_violations


def hoeffding_width(n, family_size, factor=1):
    return float(np.sqrt(np.log(factor * family_size / DELTA) / (2 * n)))


def empirical_bernstein_ratio(u, c, family_size):
    """反演有界 W_i(theta)=U_i-theta*C_i 的经验 Bernstein 上界。"""
    n = len(u)
    log_term = np.log(2 * family_size / DELTA)

    def upper_mean(theta):
        values = u - theta * c
        variance = float(np.var(values, ddof=1)) if n > 1 else 0.0
        radius = np.sqrt(2 * variance * log_term / n)
        radius += 7 * log_term / (3 * max(n - 1, 1))
        return float(np.mean(values) + radius)

    if upper_mean(1.0) > 0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if upper_mean(middle) <= 0:
            high = middle
        else:
            low = middle
    return high


def candidate_certificates(coverage, union, failure):
    n, selector_count = coverage.shape
    tau_count = union.shape[2]
    grid_size = selector_count * tau_count
    direct_width = hoeffding_width(n, grid_size)
    double_width = hoeffding_width(n, grid_size, factor=2)
    failure_width = hoeffding_width(n, selector_count)
    output = []
    for selector_index in range(selector_count):
        c = coverage[:, selector_index]
        c_mean = float(np.mean(c))
        f = failure[:, selector_index]
        direct_failure = min(1.0, float((np.mean(f) + failure_width) / c_mean)) if c_mean > 0 else None
        for tau_index in range(tau_count):
            u = union[:, selector_index, tau_index]
            u_mean = float(np.mean(u))
            double = None
            if c_mean > double_width:
                double = min(1.0, float((u_mean + double_width) / (c_mean - double_width)))
            direct = min(1.0, float((u_mean + direct_width) / c_mean)) if c_mean > 0 else None
            output.append({
                "selector_index": selector_index,
                "tau_index": tau_index,
                "coverage": c_mean,
                "union_mass": u_mean,
                "failure_mass": float(np.mean(f)),
                "double_hoeffding": double,
                "direct_union": direct,
                "direct_failure": direct_failure,
                "empirical_bernstein_union": empirical_bernstein_ratio(u, c, grid_size),
            })
    return output


def select_candidate(candidates, method):
    if method == "direct_failure":
        unique = {}
        for row in candidates:
            unique.setdefault(row["selector_index"], row)
        pool = list(unique.values())
    else:
        pool = candidates
    feasible = [row for row in pool if row[method] is not None]
    if not feasible:
        return None
    return min(feasible, key=lambda row: (row[method], -row["coverage"], row["tau_index"]))


def evaluate_selection(selected, method, coverage, union, failure, method_count=1):
    if selected is None:
        return {"abstained": True}
    selector_index = selected["selector_index"]
    tau_index = selected["tau_index"]
    c = coverage[:, selector_index]
    u = union[:, selector_index, tau_index]
    f = failure[:, selector_index]
    c_mean = float(np.mean(c))
    risk = float(np.mean(f) / c_mean) if c_mean > 0 else None
    union_ratio = float(np.mean(u) / c_mean) if c_mean > 0 else None
    width = hoeffding_width(len(c), method_count)
    validation_upper = min(1.0, float((np.mean(f) + width) / c_mean)) if c_mean > 0 else None
    return {
        "abstained": False,
        "selector_index": selector_index,
        "tau_index": tau_index,
        "certificate": selected[method],
        "validation_coverage": c_mean,
        "validation_risk": risk,
        "validation_union_ratio": union_ratio,
        "validation_direct_failure_upper": validation_upper,
        "observed_nonexceedance": bool(risk is not None and risk <= selected[method] + 1e-12),
    }


def fixed_after_selection(coverage, union, failure, total_n):
    """用前半预算选择策略，后半预算只认证冻结后的一个策略。"""
    split = total_n // 2
    selection_candidates = candidate_certificates(
        coverage[:split], union[:split], failure[:split]
    )
    plugin = min(
        selection_candidates,
        key=lambda row: (
            row["union_mass"] / max(row["coverage"], 1e-12),
            -row["coverage"],
        ),
    )
    selector_index = plugin["selector_index"]
    tau_index = plugin["tau_index"]
    c = coverage[split:total_n, selector_index]
    u = union[split:total_n, selector_index, tau_index]
    f = failure[split:total_n, selector_index]
    width = hoeffding_width(len(c), 1)
    c_mean = float(np.mean(c))
    direct_union = min(1.0, float((np.mean(u) + width) / c_mean)) if c_mean > 0 else None
    direct_failure = min(1.0, float((np.mean(f) + width) / c_mean)) if c_mean > 0 else None
    return {
        "selection_blocks": split,
        "certification_blocks": total_n - split,
        "selector_index": selector_index,
        "tau_index": tau_index,
        "certification_coverage": c_mean,
        "direct_union": direct_union,
        "direct_failure": direct_failure,
    }


def run_quantizer(pools, transformed, dimension):
    encodings = {name: base.encode(values) for name, values in transformed.items()}
    model = base.fit_affine(
        pools["beta"], encodings["beta"], base.N_BETA_BLOCKS, SEED + dimension + 10
    )
    beta_blocks = base.collect_blocks(
        pools["beta"], encodings["beta"], model, base.N_BETA_BLOCKS, SEED + dimension + 50
    )
    risk_blocks = base.collect_blocks(
        pools["risk"], encodings["risk"], model, MAX_RISK_BLOCKS, SEED + dimension + 100
    )
    validation_blocks = base.collect_blocks(
        pools["validation"], encodings["validation"], model, N_VALIDATION_BLOCKS, SEED + dimension + 200
    )
    roles = {}
    for role in ("ranking", "pruning"):
        role_model = base.fit_role(beta_blocks[role])
        risk_arrays = block_triplets(risk_blocks[role], role_model)
        validation_arrays = block_triplets(validation_blocks[role], role_model)
        coverage, union, failure, risk_violations = risk_arrays
        val_coverage, val_union, val_failure, val_violations = validation_arrays
        curves = []
        for n in SAMPLE_SIZES:
            candidates = candidate_certificates(coverage[:n], union[:n], failure[:n])
            selections = {
                method: evaluate_selection(
                    select_candidate(candidates, method),
                    method,
                    val_coverage,
                    val_union,
                    val_failure,
                    method_count=len(METHODS),
                )
                for method in METHODS
            }
            curves.append({
                "block_count": n,
                "simultaneous_grid_size": int(coverage.shape[1] * union.shape[2]),
                "selections": selections,
                "fixed_after_independent_selection": fixed_after_selection(
                    coverage, union, failure, n
                ),
            })
        roles[role] = {
            "risk_object": "E[F_i]/E[C_i]",
            "block_count": MAX_RISK_BLOCKS,
            "validation_block_count": N_VALIDATION_BLOCKS,
            "selector_count": int(coverage.shape[1]),
            "tau_count": int(union.shape[2]),
            "invariant_violations": int(risk_violations + val_violations),
            "curves": curves,
        }
    return {"distance_calibration": model, "roles": roles}


def run_dataset(name):
    pools, total, kind = base.load_pools(name)
    dimension = pools["beta"].shape[1]
    rng = np.random.default_rng(SEED + 31 * dimension)
    rotation_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), dimension)
    transformed_modes = {
        "2bit": pools,
        "rotated_2bit": {
            key: base.rotate(values, rotation_signs) for key, values in pools.items()
        },
    }
    return {
        "dataset": name,
        "kind": kind,
        "dimension": int(dimension),
        "source_total": int(total),
        "quantizers": {
            quantizer: run_quantizer(pools, transformed, dimension)
            for quantizer, transformed in transformed_modes.items()
        },
    }


def aggregate(results):
    output = {}
    for role in ("ranking", "pruning"):
        rows = []
        for dataset in results:
            for quantizer, metrics in dataset["quantizers"].items():
                for curve in metrics["roles"][role]["curves"]:
                    for method, selected in curve["selections"].items():
                        if selected["abstained"]:
                            continue
                        rows.append({
                            "dataset": dataset["dataset"],
                            "quantizer": quantizer,
                            "block_count": curve["block_count"],
                            "method": method,
                            **selected,
                        })
        summaries = []
        for n in SAMPLE_SIZES:
            for method in METHODS:
                subset = [row for row in rows if row["block_count"] == n and row["method"] == method]
                summaries.append({
                    "block_count": n,
                    "method": method,
                    "issued_count": len(subset),
                    "mean_certificate": float(np.mean([row["certificate"] for row in subset])) if subset else None,
                    "mean_validation_coverage": float(np.mean([row["validation_coverage"] for row in subset])) if subset else None,
                    "mean_validation_risk": float(np.mean([row["validation_risk"] for row in subset])) if subset else None,
                    "observed_nonexceedance_rate": float(np.mean([row["observed_nonexceedance"] for row in subset])) if subset else None,
                })
        output[role] = {"summaries": summaries, "records": rows}
    return output


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}", flush=True)
        result = run_dataset(name)
        results.append(result)
        for quantizer, metrics in result["quantizers"].items():
            rank = metrics["roles"]["ranking"]["curves"][-1]["selections"]
            prune = metrics["roles"]["pruning"]["curves"][-1]["selections"]
            print(
                f"  {quantizer:<13} "
                f"rank={rank['direct_union']['certificate']:.4f}/"
                f"{rank['direct_failure']['certificate']:.4f}, "
                f"prune={prune['direct_union']['certificate']:.4f}/"
                f"{prune['direct_failure']['certificate']:.4f}",
                flush=True,
            )
    output = {
        "probe": 45,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "sample_sizes": SAMPLE_SIZES,
        "validation_blocks": N_VALIDATION_BLOCKS,
        "delta": DELTA,
        "risk_object": "block-uniform selective risk E[F_i]/E[C_i]",
        "block_convention": "empty pruning block has C_i=U_i=F_i=0",
        "curve_protocol": "all sample sizes and methods are prespecified; each curve point has marginal confidence 1-delta and is not selected post hoc",
        "validation_status": "independent diagnostic with a simultaneous direct-failure upper bound over the 16 displayed method-by-sample-size policies",
        "aggregate": aggregate(results),
        "results": results,
    }
    path = base.RESULTS_DIR / "probe45_d_selective_ratio_certificates.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {path}", flush=True)


if __name__ == "__main__":
    main()
