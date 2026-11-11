#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InfoNCE 分布探测实验 — 通用工具模块

提供数据集加载、L2 归一化、结果保存等共享功能。
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(os.environ.get("JMLR_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))  # 原始数据目录
HDF5_DIR = Path(__file__).resolve().parent / "dataset"   # VIBE HDF5 数据集目录
RESULTS_DIR = Path(os.environ.get("JMLR_RESULTS_DIR", Path(__file__).resolve().parent / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 数据集注册表：名称 → (前缀, 维度, 类型, 格式)
# 格式: "f32" = raw binary, "hdf5" = VIBE HDF5
DATASETS = {
    # --- 原有 f32 格式 ---
    "cohere":       ("cohere",              768,  "contrastive",  "f32"),
    "minilm":       ("minilm",              384,  "contrastive",  "f32"),
    "bge_m3":       ("bge_m3",              1024, "contrastive",  "f32"),
    "dbpedia1536":  ("dbpedia_openai",      1536, "contrastive",  "f32"),
    "dbpedia3072":  ("dbpedia_openai_3072", 3072, "contrastive",  "f32"),
    "wolt_clip":    ("wolt_clip",           512,  "multimodal",   "f32"),
    "gist960":      ("gist960",             960,  "euclidean",    "f32"),
    "random":       ("random",              768,  "random",       "f32"),
    "glove100":     ("glove100",            100,  "word2vec",     "f32"),
    "sift128":      ("sift128",             128,  "euclidean",    "f32"),
    "sphere":       ("sphere",              768,  "synthetic",    "f32"),
    # --- VIBE 数据集（已转换为 f32 格式，768 维同维对比集） ---
    "arxiv_nomic":      ("arxiv_nomic",       768, "contrastive",  "f32"),
    "ccnews_nomic":     ("ccnews_nomic",      768, "contrastive",  "f32"),
    "coco_nomic":       ("coco_nomic",        768, "multimodal",   "f32"),
    "codesearch_jina":  ("codesearch_jina",   768, "contrastive",  "f32"),
    "gooaq_roberta":    ("gooaq_roberta",     768, "contrastive",  "f32"),
    "landmark_nomic":   ("landmark_nomic",    768, "contrastive",  "f32"),
    "landmark_dino":    ("landmark_dino",     768, "ssl_vision",   "f32"),
    # --- MSMARCO（论文 scalability 实验数据集） ---
    "msmarco_cohere":   ("msmarco_cohere",   1024, "contrastive",  "f32"),
}

# Phase 1 核心对比集
PHASE1_DATASETS = ["cohere", "minilm", "gist960", "random"]

# Phase 5 同维对比集（全部 768 维）
PHASE5_768 = [
    "cohere", "arxiv_nomic", "ccnews_nomic", "coco_nomic",
    "codesearch_jina", "gooaq_roberta", "landmark_nomic",
    "landmark_dino", "random",
]


def load_dataset(name: str, n_sample: int = 50_000) -> np.ndarray:
    """加载数据集训练向量，采样 n_sample 条并 L2 归一化。
    
    自动识别 f32 binary 和 HDF5 两种格式。
    """
    if name not in DATASETS:
        raise ValueError(f"未知数据集: {name}，可选: {list(DATASETS.keys())}")

    prefix, dim, dtype, fmt = DATASETS[name]

    if fmt == "hdf5":
        # VIBE HDF5 格式
        if not HAS_H5PY:
            raise ImportError("需要 h5py 库来加载 HDF5 数据集: pip install h5py")
        path = HDF5_DIR / f"{prefix}.hdf5"
        if not path.exists():
            raise FileNotFoundError(
                f"HDF5 数据文件不存在: {path}\n"
                f"请从 https://huggingface.co/datasets/vector-index-bench/vibe 下载"
            )
        with h5py.File(str(path), 'r') as f:
            n_total = f['train'].shape[0]
            if n_sample < n_total:
                # 先读取连续块到内存（比随机索引快 100 倍以上）
                # 均匀间隔采样，避免偏差
                step = n_total // n_sample
                data = f['train'][::step][:n_sample]
                # 如果不够就补充读取
                if len(data) < n_sample:
                    data = f['train'][:n_sample]
                print(f"  [{name}] HDF5 加载完成: ({n_total}, {dim}) → 采样 {len(data)} 条")
            else:
                data = f['train'][:]
                print(f"  [{name}] HDF5 加载完成: {data.shape}")
    else:
        # 原有 f32 binary 格式
        path = ROOT / f"{prefix}_train.f32"
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}\n请按 docs/reproduction.md 配置数据目录与文件名")
        # 先映射文件并抽取指定行，避免小样本实验整库读入内存。
        if path.stat().st_size % (4 * dim):
            raise ValueError(f"数据文件长度与维度不匹配：{path}")
        data = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, dim)
        print(f"  [{name}] 加载完成: {data.shape}")
        if n_sample < len(data):
            np.random.seed(42)
            idx = np.random.choice(len(data), n_sample, replace=False)
            data = data[idx]
            print(f"  [{name}] 采样 {n_sample} 条")

    # L2 归一化
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / np.maximum(norms, 1e-12)

    return data


def save_result(name: str, probe_name: str, result: dict):
    """保存探针结果为 .npz 文件"""
    out_path = RESULTS_DIR / f"{probe_name}_{name}.npz"
    np.savez(str(out_path), **{k: np.array(v) for k, v in result.items()})
    print(f"  [保存] {out_path.name}")


def load_result(name: str, probe_name: str) -> dict:
    """加载探针结果"""
    path = RESULTS_DIR / f"{probe_name}_{name}.npz"
    data = np.load(str(path), allow_pickle=True)
    return dict(data)


# ═══════════════════════════════════════════════════════════════════
#  共享 BQ 计算函数（从 probe 8/9/11/13/14/15/18/23 中去重提取）
# ═══════════════════════════════════════════════════════════════════

def compute_sign_entropy(X: np.ndarray) -> float:
    """计算逐维度 sign entropy 的均值。

    sign entropy = 1.0 表示完美各向同性（每个维度 P(+) = 0.5）。
    InfoNCE 模型典型值在 [0.70, 0.85] 之间。
    """
    rate = np.mean(X > 0, axis=0)
    entropy = -(rate * np.log2(np.clip(rate, 1e-10, 1)) +
                (1 - rate) * np.log2(np.clip(1 - rate, 1e-10, 1)))
    return float(np.mean(entropy))


def bq2_encode(X: np.ndarray, threshold_mode: str = "median"):
    """BQ 2-bit 编码：sign bit + magnitude bit。

    Args:
        X: (N, D) L2 归一化向量
        threshold_mode: "median" 按每个维度的 median 做阈值（probe 8 原始方式），
                        "mean" 按每个向量的 mean(|x_j|) 做阈值（论文公式）

    Returns:
        signs: (N, D) uint8，sign bit
        mags:  (N, D) uint8，magnitude bit
    """
    signs = (X > 0).astype(np.uint8)
    abs_X = np.abs(X)
    if threshold_mode == "median":
        # 按维度的 median（早期 probe 使用）
        thresholds = np.median(abs_X, axis=0, keepdims=True)
    elif threshold_mode == "mean":
        # 按向量的 mean（论文公式 α_x = (1/D) Σ|x_j|）
        thresholds = abs_X.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"未知 threshold_mode: {threshold_mode}")
    mags = (abs_X > thresholds).astype(np.uint8)
    return signs, mags


def bq2_hamming_score(signs_a, mags_a, signs_b, mags_b):
    """BQ 2-bit 加权 Hamming 相似度。

    编码方式: sign=1 & mag=1 → +2, sign=1 & mag=0 → +1, sign=0 → -1
    等价于 (1 + strong_a) * (1 + strong_b) 在 sign 一致时加分。

    Args:
        signs_a, mags_a: 查询编码 (1, D) 或 (N, D)
        signs_b, mags_b: 数据库编码 (N, D)

    Returns:
        score: (N,) 每对的 BQ2 分数
    """
    s_agree = (signs_a == signs_b).astype(np.float32)
    w = (1 + mags_a.astype(np.float32)) * (1 + mags_b.astype(np.float32))
    score = np.sum(s_agree * w, axis=1)
    return score


def bq2_fidelity(X: np.ndarray, n_pairs: int = 200_000,
                 threshold_mode: str = "median") -> float:
    """BQ 2-bit 编码保真度 = Spearman ρ(cosine, BQ score)。

    衡量 BQ 编码保留成对排序信息的程度，对应论文中的 F。
    """
    from scipy.stats import spearmanr

    N, D = X.shape
    rng = np.random.RandomState(42)
    idx_a = rng.randint(0, N, n_pairs)
    idx_b = rng.randint(0, N, n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    # 精确 cosine
    cos_sim = np.sum(X[idx_a] * X[idx_b], axis=1)

    # BQ2 分数
    signs, mags = bq2_encode(X, threshold_mode)
    bq_score = bq2_hamming_score(signs[idx_a], mags[idx_a],
                                  signs[idx_b], mags[idx_b])

    rho, _ = spearmanr(cos_sim, bq_score)
    return float(rho)


def bq2_recall(X: np.ndarray, n_queries: int = 200, top_k: int = 10,
               threshold_mode: str = "median") -> float:
    """BQ 2-bit 模拟 Recall@K。

    对每个 query，比较精确 cosine top-K 和 BQ top-K 的重叠度。
    """
    N, D = X.shape
    rng = np.random.RandomState(42)
    q_idx = rng.choice(N, min(n_queries, N), replace=False)
    signs, mags = bq2_encode(X, threshold_mode)

    recalls = []
    for qi in q_idx:
        # 精确 cosine
        exact = X @ X[qi]
        exact[qi] = -2.0
        gt = set(np.argpartition(-exact, top_k)[:top_k])

        # BQ2 分数
        score = bq2_hamming_score(
            signs[qi:qi+1].repeat(N, axis=0), mags[qi:qi+1].repeat(N, axis=0),
            signs, mags
        )
        score[qi] = -1e9
        bq_top = set(np.argpartition(-score, top_k)[:top_k])
        recalls.append(len(gt & bq_top) / top_k)

    return float(np.mean(recalls))


def compute_snr(X: np.ndarray) -> np.ndarray:
    """从数据估计逐维度 SNR = μ_i / σ_i。"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return mu / np.maximum(sigma, 1e-10)


def random_orthogonal(D: int, seed: int = 0) -> np.ndarray:
    """生成 D×D 随机正交矩阵（Haar 均匀分布）。"""
    rng = np.random.RandomState(seed)
    H = rng.randn(D, D).astype(np.float32)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q
