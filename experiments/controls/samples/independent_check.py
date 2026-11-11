"""独立整数评分、集合选择、组件频率与统计复算。"""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.fft import dct
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT))
from controls_v2 import load_vectors


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(matrix, labels, permutation):
    """直接按照标准剪枝定义生成输出，不调用实验的搜索实现。"""
    kept, trace = [], []
    for vertex in permutation:
        rejected = any(1.2*matrix[vertex, previous] <= matrix[0,vertex] for previous in kept)
        trace.append((vertex,tuple(kept),rejected))
        if not rejected:
            kept.append(vertex)
        if len(kept) == 32:
            break
    return [int(labels[index]) for index in kept],trace


def check_pruning(dataset):
    values = load_vectors(dataset,"pruning")
    dimension = values.shape[1]
    targets = np.random.default_rng(20260823).choice(4000,240,replace=False)[80:]
    similarities = values[targets] @ values.T
    controls = json.loads((OUT/f"pruning-{dataset}.json").read_text(encoding="utf-8"))
    checked = 0
    for quantizer, result in controls.items():
        transformed = values
        if quantizer == "rotated_2bit":
            signs = np.random.default_rng(20260823+dimension).choice(np.array([-1.,1.],dtype=np.float32),dimension)
            transformed = dct(values*signs,type=2,norm="ortho",axis=1).astype(np.float32)
            transformed /= np.maximum(np.linalg.norm(transformed,axis=1,keepdims=True),1e-12)
        code = np.where(transformed>0,1,-1).astype(np.int32)
        code *= 1+(np.abs(transformed)>np.abs(transformed).mean(axis=1,keepdims=True))
        with np.load(OUT/f"pruning-{dataset}-{quantizer}.npz") as arrays:
            for row,target in enumerate(targets):
                scores = similarities[row].copy()
                scores[target] = -np.inf
                pool = np.lexsort((np.arange(4000),-scores))[:64]
                labels = np.concatenate([[target],pool])
                exact = np.maximum((1-values[labels]@values[labels].T).astype(np.float64),0)
                raw = (4*dimension-code[labels]@code[labels].T).astype(np.float64)
                order = np.lexsort((pool,exact[0,1:]))+1
                expected,trace = select(exact,labels,order)
                assert np.array_equal(expected+[-1]*(32-len(expected)),arrays["exact_labels"][row])
                for name,(bias,slope) in result["models"].items():
                    approximate = bias+slope*raw
                    selected,_ = select(approximate,labels,order)
                    assert np.array_equal(selected+[-1]*(32-len(selected)),arrays[f"{name}_fixed_labels"][row])
                    free_order = np.lexsort((pool,approximate[0,1:]))+1
                    free,_ = select(approximate,labels,free_order)
                    expected_set,selected_set,free_set = set(expected),set(selected),set(free)
                    fixed_jaccard = len(expected_set & selected_set)/len(expected_set | selected_set)
                    free_jaccard = len(expected_set & free_set)/len(expected_set | free_set)
                    assert np.allclose([fixed_jaccard,free_jaccard],arrays[f"{name}_jaccards"][row],rtol=0,atol=1e-15)
                    disagreement = any(any(1.2*approximate[vertex,previous] <= approximate[0,vertex] for previous in prefix) != rejected for vertex,prefix,rejected in trace)
                    assert disagreement == (selected != expected)
                    checked += 1
    return checked


def check_calibration():
    checked = 0
    for dataset in ["cohere","minilm","gist960"]:
        result = json.loads((OUT/f"calibration-{dataset}.json").read_text(encoding="utf-8"))
        for quantizer,row in result.items():
            with np.load(OUT/f"calibration-{dataset}-{quantizer}.npz") as arrays:
                for role,record in row["roles"].items():
                    margins = arrays[f"{role}_margin"]
                    residuals = arrays[f"{role}_residual"]
                    boundary = np.count_nonzero(margins <= record["tau"])/len(margins)
                    tail = np.count_nonzero(np.abs(residuals) >= record["tau"])/len(margins)
                    assert boundary == record["boundary"] and tail == record["tail"]
                    assert min(1,boundary+tail) == record["combined"]
                    assert record["reproduction_gaps"]["calibration_count"] == 0
                    checked += 1
    return checked


def check_statistics():
    data = json.loads((OUT/"three-pool-components.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT/"results/probe33_standard_vamana_p6_heldout_calibration.json").read_text(encoding="utf-8"))
    identities = { (ds["dataset"],quantizer):result for ds in old["results"] for quantizer,result in ds["quantizers"].items()}
    checked = 0
    for role in ["ranking","pruning"]:
        records = data[role]["records"]
        for row in records:
            source = identities[(row["dataset"],row["quantizer"])]["roles"][role]
            assert row["boundary"] == source["validation"]["heldout_boundary_mass"]
            assert row["tail"] == source["validation"]["heldout_residual_tail"]
            assert row["flip"] == source["stress"]["flip_rate"]
        names = list(dict.fromkeys(row["dataset"] for row in records))
        rng = np.random.default_rng(20261109)
        bootstrap = {name:[] for name in ["boundary","tail","combined"]}
        for _ in range(2000):
            selected = [row for index in rng.integers(0,len(names),len(names)) for row in records if row["dataset"] == names[index]]
            for metric in bootstrap:
                bootstrap[metric].append(float(spearmanr([row[metric] for row in selected],[row["flip"] for row in selected]).statistic))
        for metric,samples in bootstrap.items():
            rho = float(spearmanr([row[metric] for row in records],[row["flip"] for row in records]).statistic)
            assert abs(rho-data[role]["metrics"][metric]["spearman"]) < 1e-14
            interval = np.nanquantile(samples,[.025,.975])
            assert np.allclose(interval,data[role]["metrics"][metric]["dataset_cluster_95"],rtol=0,atol=1e-14)
            checked += 1
        for metric in ["boundary","tail"]:
            interval = np.nanquantile(np.array(bootstrap["combined"])-bootstrap[metric],[.025,.975])
            assert np.allclose(interval,data[role]["combined_minus_component_95"][metric],rtol=0,atol=1e-14)
    return checked


def main():
    result = {"pruning":{},"calibration_components":check_calibration(),"statistical_controls":check_statistics()}
    for dataset in ["minilm","gist960"]:
        result["pruning"][dataset] = check_pruning(dataset)
        print(f"{dataset}独立整数评分与输出复核通过",flush=True)
    result["status"] = "passed"
    result["source_sha256"] = digest(Path(__file__))
    result["evidence_sha256"] = {path.name:digest(path) for pattern in ["*.npz","pruning-*.json","calibration-*.json","three-pool-components.json"] for path in OUT.glob(pattern)}
    final = OUT/"independent-verification.json"
    assert not final.exists()
    final.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("全部新增校准对照与重分析独立复核通过",flush=True)


if __name__ == "__main__":
    main()
