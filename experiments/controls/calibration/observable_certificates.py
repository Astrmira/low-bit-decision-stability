"""独立拟合、认证和验证可观测量化margin规则，复用原有比较生成逻辑。"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import importlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(os.environ.get("JMLR_CALIBRATION_DIR", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
base = importlib.import_module("17_p6_strict_certificate")
covariance = importlib.import_module("09_covariance_certificate")
covariance.ROOT = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data"))
DELTA = 0.05
RETENTIONS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])


def blocks(values, encoded, model, count, seed):
    rng = np.random.default_rng(seed)
    result = {"ranking": [], "pruning": []}
    for _ in range(count):
        target, candidates = base.sample_block(values, rng)
        top, competitors = int(candidates[0]), candidates[1:]
        exact = (1 - values[competitors] @ values[target]) - (1 - np.dot(values[target], values[top]))
        quantized = base.calibrated(np.full(len(competitors), target), competitors, encoded, model) - base.calibrated(np.full(len(competitors), target), np.full(len(competitors), top), encoded, model)
        result["ranking"].append({"margin": exact.astype(np.float64), "residual": (quantized - exact).astype(np.float64), "flip": (quantized < 0) | ((quantized == 0) & (competitors < top)), "quantized_margin": np.abs(quantized)})
        comparisons = base.exact_pruning_comparisons(values, target, candidates)
        if comparisons:
            nodes = np.asarray([(t, c, s) for t, c, s, _ in comparisons], dtype=np.int64)
            exact = np.asarray([psi for _, _, _, psi in comparisons], dtype=np.float64)
            quantized = base.calibrated(nodes[:, 0], nodes[:, 1], encoded, model) - base.ALPHA * base.calibrated(nodes[:, 1], nodes[:, 2], encoded, model)
            result["pruning"].append({"margin": np.abs(exact), "residual": quantized - exact, "flip": (exact >= 0) != (quantized >= 0), "quantized_margin": np.abs(quantized)})
        else:
            result["pruning"].append({**base.empty_block(), "quantized_margin": np.empty(0)})
    return result


def block_arrays(data, cutoffs, taus, selector):
    coverage = np.zeros((len(data), len(cutoffs)))
    failure = np.zeros_like(coverage)
    union = np.zeros((len(data), len(cutoffs), len(taus)))
    for i, block in enumerate(data):
        if not len(block["margin"]):
            continue
        for j, cutoff in enumerate(cutoffs):
            accepted = block[selector] >= cutoff
            coverage[i, j] = np.mean(accepted)
            failure[i, j] = np.mean(accepted & block["flip"])
            for k, tau in enumerate(taus):
                event = (block["margin"] <= tau) | (np.abs(block["residual"]) >= tau)
                assert np.all(~block["flip"] | event)
                union[i, j, k] = np.mean(accepted & event)
    return coverage, failure, union


def report(fitting, certification, validation, selector):
    cutoffs = np.quantile(base.flatten(fitting, selector), 1 - RETENTIONS)
    tau_base = np.median(base.flatten(fitting, "margin"))
    taus = np.maximum(tau_base * base.TAU_RATIOS, 1e-12)
    c, f, u = block_arrays(certification, cutoffs, taus, selector)
    vc, vf, vu = block_arrays(validation, cutoffs, taus, selector)
    ch, fh, uh = c.mean(axis=0), f.mean(axis=0), u.mean(axis=0)
    e_direct = np.sqrt(np.log(len(cutoffs) / DELTA) / (2 * len(certification)))
    e_mechanism = np.sqrt(np.log(len(cutoffs) * len(taus) / DELTA) / (2 * len(certification)))
    candidates = []
    for j in range(len(cutoffs)):
        if ch[j] <= 0:
            continue
        candidates.append({"selector": j, "tau": None, "kind": "failure", "bound": min(1.0, float((fh[j] + e_direct) / ch[j]))})
        for k in range(len(taus)):
            candidates.append({"selector": j, "tau": k, "kind": "mechanism", "bound": min(1.0, float((uh[j, k] + e_mechanism) / ch[j]))})
    selected = {}
    for kind in ("failure", "mechanism"):
        best = min((r for r in candidates if r["kind"] == kind), key=lambda row: (row["bound"], -ch[row["selector"]]))
        j = best["selector"]
        selected[kind] = {**best, "cutoff": float(cutoffs[j]), "certification_acceptance": float(ch[j]), "validation_acceptance": float(vc[:, j].mean()), "validation_failure": float(vf[:, j].mean() / vc[:, j].mean()), "validation_below_certificate": bool(vf[:, j].mean() / vc[:, j].mean() <= best["bound"])}
    return {"cutoffs": cutoffs.tolist(), "tau_grid": taus.tolist(), "selected": selected}, {"C": c, "F": f, "U": u, "validation_C": vc, "validation_F": vf, "validation_U": vu}


def main():
    results = []
    for dataset in ("cohere", "minilm", "gist960"):
        if dataset == "cohere":
            values = covariance.load_sampled_dataset(dataset)
        else:
            with np.load(Path(os.environ.get("JMLR_SAMPLE_DIR", ROOT.parent / "data/samples")) / f"{dataset}-samples.npz") as source:
                values = source["pruning"].copy()
        for code in ("2bit", "rotated_2bit"):
            path = OUT / f"observable-{dataset}-{code}.json"
            if path.exists():
                results.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            transformed = covariance.randomized_orthogonal_transform(values) if code.startswith("rotated") else values
            encoded = base.encode(transformed)
            seed = 20261110 + values.shape[1]
            model = base.fit_affine(values, encoded, 128, seed)
            populations = {name: blocks(values, encoded, model, count, seed + offset) for name, count, offset in (("fit", 128, 100), ("cert", 512, 200), ("validation", 512, 300))}
            # 与原函数逐数组核对，新增字段仅记录可观测分数。
            check = base.collect_blocks(values, encoded, model, 8, seed + 100)
            for role in ("ranking", "pruning"):
                for expected, actual in zip(check[role], populations["fit"][role][:8]):
                    for key in ("margin", "residual", "flip"):
                        assert np.array_equal(expected[key], actual[key])
            result = {"dataset": dataset, "code": code, "model": model, "fit_blocks": 128, "certification_blocks": 512, "validation_blocks": 512, "roles": {}}
            arrays = {}
            for role in ("ranking", "pruning"):
                result["roles"][role] = {}
                for selector in ("margin", "quantized_margin"):
                    result["roles"][role][selector], evidence = report(populations["fit"][role], populations["cert"][role], populations["validation"][role], selector)
                    arrays.update({f"{role}_{selector}_{key}": value for key, value in evidence.items()})
                for split, data in populations.items():
                    selected = data[role]
                    arrays[f"{role}_{split}_offsets"] = np.cumsum([0] + [len(r["margin"]) for r in selected])
                    for key in ("margin", "residual", "flip", "quantized_margin"):
                        arrays[f"{role}_{split}_{key}"] = base.flatten(selected, key)
            np.savez_compressed(OUT / f"observable-{dataset}-{code}.npz", **arrays)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(result)
            print("可观测接受规则已完成", dataset, code, flush=True)
    (OUT / "observable-certificates.json").write_text(json.dumps({"sampling": "每个固定4K经验池内块间独立重抽，拟合、认证、验证使用独立种子", "delta_per_reported_certificate": DELTA, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
