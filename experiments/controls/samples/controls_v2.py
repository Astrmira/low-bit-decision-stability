"""补齐三库剪枝校准对照，并以独立数据池直接拆解风险组件。"""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "controls/covariance"))
from verify import fit_pruning_models

DATA = Path(os.environ.get("JMLR_SAMPLE_DIR", ROOT.parent / "data/samples"))
SEED = 20261109


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError(f"保留既有证据，禁止覆盖：{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_vectors(dataset, pool):
    if dataset == "cohere":
        path = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data")) / "cohere_train.f32"
        mapped = np.memmap(path, dtype="<f4", mode="r", shape=(1000000, 768))
        if pool == "pruning":
            indices = np.random.default_rng(20260823).choice(1000000, 4000, replace=False)
        else:
            indices = np.random.default_rng(20260824 + 768).choice(1000000, 6000, replace=False)[:2000]
        values = np.array(mapped[indices], dtype=np.float32)
    else:
        with np.load(DATA / f"{dataset}-samples.npz") as arrays:
            values = arrays[pool].copy()
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def pruning(dataset):
    module = importlib.import_module("10_pruning_replay")
    vectors = load_vectors(dataset, "pruning")
    targets = np.random.default_rng(module.SEED).choice(len(vectors), 240, replace=False)
    fit_candidates = module.candidate_lists(vectors, targets[:80])
    eval_candidates = module.candidate_lists(vectors, targets[80:])
    old = json.loads((ROOT / "results/probe27_standard_vamana_pruning_replay.json").read_text(encoding="utf-8"))
    prior = next(row for row in old["results"] if row["dataset"] == dataset)["quantizers"]
    output = {}
    for quantizer in module.QUANTIZERS:
        print(f"正在进行{dataset}剪枝校准对照 {quantizer}", flush=True)
        transformed = module.randomized_orthogonal_transform(vectors) if quantizer.startswith("rotated") else vectors
        encoded = module.encode_bq2(transformed)
        models, arrays = fit_pruning_models(module, vectors, targets[:80], fit_candidates, encoded)
        rows = {name: [] for name in models}
        local = {name: [] for name in models}
        labels = {name: [] for name in models}
        exact_rows = []
        for target, pool in zip(targets[80:], eval_candidates):
            nodes, exact = module.distance_cache(vectors, target, pool)
            _, raw = module.distance_cache(vectors, target, pool, encoded, 1.0)
            positions = {int(node): index for index, node in enumerate(nodes)}
            reference = module.run_select(target, pool, nodes, exact)
            original_set = set(reference["final_selected"])
            for name, (bias, slope) in models.items():
                approximate = bias + slope * raw
                fixed = module.run_select(target, pool, nodes, approximate, ordered_override=reference["ordered"])
                free = module.run_select(target, pool, nodes, approximate)
                fixed_set, free_set = set(fixed["final_selected"]), set(free["final_selected"])
                rows[name].append([len(original_set & fixed_set)/len(original_set | fixed_set), len(original_set & free_set)/len(original_set | free_set)])
                labels[name].append(list(fixed["final_selected"]) + [-1] * (32-len(fixed_set)))
                for decision in reference["comparisons"]:
                    a, b, c = positions[int(target)], positions[decision["candidate"]], positions[decision["selected"]]
                    local[name].append([exact[a,b]-module.ALPHA*exact[b,c], approximate[a,b]-module.ALPHA*approximate[b,c]])
            exact_rows.append(list(reference["final_selected"]) + [-1]*(32-len(original_set)))
        summary = {"models": models, "variants": {}}
        for name in models:
            pairs = np.array(local[name])
            psi, approximate = pairs[:,0], pairs[:,1]
            residual = approximate-psi
            false_keep = (psi >= 0) & (approximate < 0)
            false_prune = (psi < 0) & (approximate >= 0)
            summary["variants"][name] = {
                "comparisons": len(pairs), "false_keep_rate": float(false_keep.mean()),
                "false_prune_rate": float(false_prune.mean()), "flip_rate": float((false_keep | false_prune).mean()),
                "bias_over_sd": float(residual.mean()/residual.std(ddof=1)),
                "fixed_order_jaccard": float(np.array(rows[name])[:,0].mean()),
                "free_order_jaccard": float(np.array(rows[name])[:,1].mean()),
            }
            arrays[f"{name}_local"] = pairs
            arrays[f"{name}_jaccards"] = np.array(rows[name])
            arrays[f"{name}_fixed_labels"] = np.array(labels[name], dtype=np.int64)
        arrays["exact_labels"] = np.array(exact_rows, dtype=np.int64)
        arrays["targets"] = targets
        summary["scale_invariance"] = bool(np.array_equal(arrays["original_origin_fixed_labels"], arrays["affine_slope_only_fixed_labels"]))
        stats = summary["variants"]["original_origin"]
        summary["reproduction_gaps"] = {
            "local_flip": stats["flip_rate"]-prior[quantizer]["evaluation"]["local_disagreement_rate"],
            "fixed_jaccard": stats["fixed_order_jaccard"]-prior[quantizer]["evaluation"]["fixed_order_replay"]["mean_final_jaccard"],
            "slope": models["original_origin"][1]-prior[quantizer]["distance_calibration"]["slope"],
        }
        assert summary["scale_invariance"]
        assert abs(summary["reproduction_gaps"]["local_flip"]) < 1e-6, summary["reproduction_gaps"]
        assert abs(summary["reproduction_gaps"]["fixed_jaccard"]) < 1e-6, summary["reproduction_gaps"]
        np.savez_compressed(OUT / f"pruning-{dataset}-{quantizer}.npz", **arrays)
        output[quantizer] = summary
    save(f"pruning-{dataset}.json", output)
    return output


def replay_calibration(dataset):
    module = importlib.import_module("16_p6_heldout_calibration")
    values = load_vectors(dataset, "calibration")
    targets, pools = module.target_candidates(values, 48, module.SEED+101)
    old = json.loads((ROOT / "results/probe33_standard_vamana_p6_heldout_calibration.json").read_text(encoding="utf-8"))
    prior = next(row for row in old["results"] if row["dataset"] == dataset)["quantizers"]
    output = {}
    for quantizer in module.QUANTIZERS:
        transformed = module.rotate(values) if quantizer.startswith("rotated") else values
        encoded = module.encode(transformed)
        model = module.fit_calibration(values, targets, pools, encoded)
        decisions = module.collect_decisions(values, targets, pools, encoded, model)
        records, arrays = {}, {}
        for role, data in decisions.items():
            fitted = module.fit_risk_model(data)
            old_model = prior[quantizer]["roles"][role]["risk_model"]
            gaps = {key: fitted[key]-old_model[key] for key in ["tau", "plugin_risk", "residual_scale", "calibration_count"]}
            assert gaps["calibration_count"] == 0, (dataset, role, gaps)
            assert max(abs(gaps[key]) for key in ["tau", "plugin_risk", "residual_scale"]) < 1e-5, (dataset, role, gaps)
            tau = old_model["tau"]
            boundary = float(np.mean(data["margin"] <= tau))
            tail = float(np.mean(np.abs(data["residual"]) >= tau))
            assert abs(min(1,boundary+tail)-old_model["plugin_risk"]) < 1e-5
            records[role] = {"boundary": boundary, "tail": tail, "combined": min(1,boundary+tail), "tau": tau, "reproduction_gaps": gaps, "validation_flip": prior[quantizer]["roles"][role]["validation"]["flip_rate"]}
            for key, array in data.items():
                arrays[f"{role}_{key}"] = array
        np.savez_compressed(OUT/f"calibration-{dataset}-{quantizer}.npz", **arrays)
        output[quantizer] = {"model": model, "roles": records}
    save(f"calibration-{dataset}.json", output)
    return output


def correlation(left, right):
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def component_summary(records):
    datasets = list(dict.fromkeys(row["dataset"] for row in records))
    groups = np.array([[index for index,row in enumerate(records) if row["dataset"] == name] for name in datasets])
    drawn = np.random.default_rng(SEED).integers(0,len(groups),(2000,len(groups)))
    indices = groups[drawn].reshape(2000,-1)
    truth = np.array([row["flip"] for row in records])
    ranked_truth = rankdata(truth[indices], axis=1)
    ranked_truth -= ranked_truth.mean(axis=1, keepdims=True)
    scores, summaries = {}, {}
    for metric in ["boundary", "tail", "combined"]:
        values = np.array([row[metric] for row in records])
        ranks = rankdata(values[indices], axis=1)
        ranks -= ranks.mean(axis=1, keepdims=True)
        denominator = np.sqrt((ranks**2).sum(axis=1)*(ranked_truth**2).sum(axis=1))
        scores[metric] = np.divide((ranks*ranked_truth).sum(axis=1), denominator, out=np.full(len(indices),np.nan),where=denominator>0)
        eligible = np.array([row["eligible"] for row in records])
        summaries[metric] = {
            "spearman": correlation(values, truth),
            "eligible_spearman": correlation(values[eligible],truth[eligible]),
            "dataset_cluster_95": np.nanquantile(scores[metric],[.025,.975]).tolist(),
            "undefined_bootstrap_replicates": int(np.isnan(scores[metric]).sum()),
            "leave_one_dataset_out": {name: correlation(values[[row["dataset"]!=name for row in records]],truth[[row["dataset"]!=name for row in records]]) for name in datasets},
        }
    return {"records":records,"metrics":summaries,"combined_minus_component_95":{name:np.nanquantile(scores["combined"]-scores[name],[.025,.975]).tolist() for name in ["boundary","tail"]}}


def three_pool_reanalysis():
    """原首池冻结评分及tau，第二池估计组件，第三池评估预定难例。"""
    original = ROOT/"results/probe33_standard_vamana_p6_heldout_calibration.json"
    data = json.loads(original.read_text(encoding="utf-8"))
    output = {}
    for role in ["ranking","pruning"]:
        records = []
        for dataset in data["results"]:
            for quantizer, result in dataset["quantizers"].items():
                calibration = result["roles"][role]["validation"]
                test = result["roles"][role]["stress"]
                boundary,tail = calibration["heldout_boundary_mass"],calibration["heldout_residual_tail"]
                assert abs(min(1,boundary+tail)-calibration["heldout_plugin_risk"]) < 1e-12
                records.append({"dataset":dataset["dataset"],"quantizer":quantizer,"eligible":result["eligibility"]["eligible"],"boundary":boundary,"tail":tail,"combined":min(1,boundary+tail),"flip":test["flip_rate"],"tau":result["roles"][role]["risk_model"]["tau"]})
        output[role] = component_summary(records)
    output["protocol"] = {"source_sha256":sha256(original),"coefficient_and_tau_pool":"原calibration池2000向量，48个target", "component_estimation_pool":"原validation池2000向量，96个target", "evaluation_pool":"原stress池2000向量，96个target，预定最小20%精确margin", "parameter_selection":"沿用首池冻结评分及tau，不使用第三池结果选择预测器或阈值", "bootstrap_clusters":12,"bootstrap_replicates":2000,"seed":SEED}
    save("three-pool-components.json",output)
    print("24配置三池直接组件重分析完成",flush=True)


def main():
    parser = argparse.ArgumentParser(description="补充校准实验与风险组件重分析")
    parser.add_argument("task",choices=["freeze","three-pool","pruning","calibration"])
    parser.add_argument("--dataset",choices=["cohere","minilm","gist960"])
    args = parser.parse_args()
    if args.task == "freeze":
        save("protocol.json",{"seed":SEED,"datasets":["cohere","minilm","gist960"],"pruning":"原4000向量、80拟合/160评估target、64候选、alpha1.2、M32；比例/仿射斜率/完整仿射；原抽样索引","calibration_replay":"原第一池2000向量及48target，重现原tau与risk后拆分B/T","three_pool_reanalysis":"全部24配置，原首池冻结评分与tau，第二池仅估计B/T及其和，独立第三池预定最难20%margin的flip作为预测目标","statistics":"12数据集整簇2000次bootstrap，整簇留一及eligible子集；不按结果选择对照", "source_sha256":{name:sha256(ROOT/name) for name in ["10_pruning_replay.py","16_p6_heldout_calibration.py"]},"supplement_script_sha256":sha256(Path(__file__))})
    elif args.task == "three-pool":
        three_pool_reanalysis()
    else:
        if args.dataset is None:
            parser.error("向量实验必须指定数据集")
        {"pruning":pruning,"calibration":replay_calibration}[args.task](args.dataset)


if __name__ == "__main__":
    main()
