"""本轮固定协议、受限内存评分及数据身份工具。"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import hashlib
import importlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr


OUT = Path(os.environ.get("JMLR_OUTPUT_DIR", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper"
CACHE = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data"))
OLD = Path(os.environ.get("JMLR_CALIBRATION_DIR", ROOT / "controls/calibration"))
SAMPLES = Path(os.environ.get("JMLR_SAMPLE_DIR", CACHE / "samples"))
sys.path.insert(0, str(ROOT))
SEED = 20261111
TAU_QUANTILES = [.02, .04, .08, .12, .18, .25, .35, .50, .70]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(name, content):
    path = OUT / name
    if path.exists():
        raise FileExistsError(f"结果已存在，保留现场：{path}")
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def module(name):
    return importlib.import_module(name)


def normalized(values):
    values = np.asarray(values, dtype=np.float32).copy()
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def load_frozen(dataset):
    """复用原四千向量抽样，SIFT 按固定索引受限读取。"""
    path = OUT / f"sample-{dataset}.npz"
    if not path.exists():
        path = SAMPLES / f"sample-{dataset}.npz"
    if path.exists():
        with np.load(path) as archive:
            return archive["values"].copy()
    if dataset in ("minilm", "gist960"):
        source = SAMPLES / f"{dataset}-samples.npz"
        with np.load(source) as archive:
            values = archive["pruning"].copy()
            indices = archive["pruning_indices"].copy()
    elif dataset == "cohere":
        source = CACHE / "cohere_train.f32"
        total = source.stat().st_size // (4 * 768)
        indices = np.random.default_rng(20260823).choice(total, 4000, replace=False)
        mapped = np.memmap(source, dtype=np.float32, mode="r", shape=(total, 768))
        values = normalized(mapped[indices])
        del mapped
    elif dataset == "sift128":
        source = CACHE / "sift-128-euclidean.hdf5"
        with h5py.File(source, "r") as archive:
            train = archive["train"]
            indices = np.random.default_rng(SEED).choice(len(train), 4000, replace=False)
            order = np.argsort(indices)
            sorted_values = np.asarray(train[indices[order]], dtype=np.float32)
            values = np.empty_like(sorted_values)
            values[order] = sorted_values
            values = normalized(values)
    else:
        raise ValueError(f"协议未包含的数据集：{dataset}")
    assert values.shape[0] == 4000 and np.all(np.isfinite(values))
    np.savez_compressed(path, values=values, indices=indices)
    save_json(f"sample-{dataset}-identity.json", {
        "source": str(source), "sample_sha256": sha256(path),
        "shape": list(values.shape), "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "normalization": "旧样本保留归一化结果，新增原始读取采用float32逐向量L2归一化",
    })
    return values


def batched_score(score, left, right):
    output = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), 512):
        end = min(start + 512, len(left))
        output[start:end] = score(left[start:end], right[start:end])
    assert np.all(np.isfinite(output))
    return output


def risk_statistics(exact, approximate, quantiles=TAU_QUANTILES):
    """双侧与对抗单侧规则均仅使用拟合比较选择阈值。"""
    margin = np.abs(exact)
    residual = approximate - exact
    positive = margin[margin > 0]
    if not len(positive):
        return {"two_sided": 1.0, "opposing": 1.0, "minus_mnr20": 0.0,
                "scale": float(np.std(residual, ddof=1)), "bias": float(np.mean(residual)),
                "tau_two_sided": None, "tau_opposing": None}
    thresholds = np.unique(np.quantile(positive, quantiles))
    opposing = -np.sign(exact) * residual
    scores = {}
    for name, errors in (("two_sided", np.abs(residual)), ("opposing", opposing)):
        candidates = [min(1.0, float(np.mean(margin <= tau) + np.mean(errors >= tau))) for tau in thresholds]
        best = int(np.argmin(candidates))
        scores[name] = candidates[best]
        scores[f"tau_{name}"] = float(thresholds[best])
    scale = max(float(np.std(residual, ddof=1)), 1e-12)
    scores.update({"minus_mnr20": -float(np.quantile(margin, .2)) / scale,
                   "scale": scale, "bias": float(np.mean(residual))})
    for tau in thresholds:
        assert np.all((margin <= tau) | (opposing >= tau) | ((exact >= 0) == (approximate >= 0)))
    return scores


def positive_comparison_gain(exact, approximate):
    denominator = float(approximate @ approximate)
    gain = float(approximate @ exact / denominator) if denominator else 0.0
    return gain


def correlation(left, right):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(spearmanr(left, right).statistic)
