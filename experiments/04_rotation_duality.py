#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - 旋转二象性 (Rotation Duality)

合并自: probe8_rotation.py, probe17_ivf_provides_g.py, probe21b_rotation_driver.py

对应论文:
  §6  Rotation Duality (QuIVer 坐标对齐 vs RaBitQ 各向同性)
  §8  Design Implications (Table 5: 旋转消融)
  §9  实验 (旋转效果驱动因子分析)

本文件包含:
  实验 A: 旋转消融 — sign entropy/BQ recall 变化 (原 probe 8)
  实验 B: RaBitQ 线性估计器 vs Hamming BQ — 完整 4 方法对比 (原 probe 17)
  实验 C: 旋转效果驱动因子 — sign_entropy vs CV(σ) (原 probe 21b)
"""

import sys
import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (load_dataset, compute_sign_entropy, bq2_recall,
                   random_orthogonal, RESULTS_DIR)


# ═══════════════════════════════════════════════════════════════════
#  共享编码函数
# ═══════════════════════════════════════════════════════════════════

def encode_2bit_mean(vecs):
    """2-bit 编码: sign + per-vector mean 阈值。"""
    signs = (vecs > 0).astype(np.int8)
    alphas = np.abs(vecs).mean(axis=1, keepdims=True)
    strongs = (np.abs(vecs) > alphas).astype(np.int8)
    return signs, strongs


def compute_f_batched(vecs, n_pairs=2_000_000):
    """大规模 pair 分批计算 F (Spearman ρ)。"""
    n = len(vecs)
    np.random.seed(42)
    idx_a = np.random.randint(0, n, n_pairs)
    idx_b = np.random.randint(0, n, n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    signs, strongs = encode_2bit_mean(vecs)

    BATCH = 500_000
    cos_all, bq_all = [], []
    for start in range(0, len(idx_a), BATCH):
        end = min(start + BATCH, len(idx_a))
        ia, ib = idx_a[start:end], idx_b[start:end]
        cos_sim = (vecs[ia] * vecs[ib]).sum(axis=1)
        cos_all.append(cos_sim)
        sm = (signs[ia] == signs[ib])
        w = (1 + strongs[ia]) * (1 + strongs[ib])
        bq_score = (sm * w).sum(axis=1).astype(np.float64)
        bq_all.append(bq_score)

    cos_all = np.concatenate(cos_all)
    bq_all = np.concatenate(bq_all)
    rho, _ = spearmanr(cos_all, bq_all)
    return rho


def bq_sign_encode(vecs):
    """Sign-only BQ 编码，返回 {-0.5, +0.5} 的 x_bar。"""
    return (vecs > 0).astype(np.float32) - 0.5


# ═══════════════════════════════════════════════════════════════════
#  实验 A: 旋转消融 (原 probe 8)
# ═══════════════════════════════════════════════════════════════════

def exp_a_rotation_ablation():
    """施加 Haar 随机旋转，测量 sign entropy / BQ recall 变化。"""
    print(f"\n{'=' * 60}")
    print("实验 A: 旋转消融 — sign entropy / BQ recall 变化")
    print(f"{'=' * 60}")

    datasets = ["cohere", "bge_m3", "minilm", "gist960", "random"]
    N_SAMPLE = 10_000
    n_rotations = 5

    print(f"  {'Dataset':<12} {'D':>5} {'ent_orig':>9} {'ent_rot':>9} {'Δent':>8} "
          f"{'rec_orig':>9} {'rec_rot':>9} {'Δrec':>8}")
    print(f"  {'─'*72}")

    for name in datasets:
        try:
            X = load_dataset(name, N_SAMPLE)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")
            continue

        D = X.shape[1]
        orig_ent = compute_sign_entropy(X)
        orig_rec = bq2_recall(X, n_queries=100, top_k=10)

        rot_ents, rot_recs = [], []
        for seed in range(n_rotations):
            Q = random_orthogonal(D, seed=seed + 100)
            X_rot = X @ Q.T
            norms = np.linalg.norm(X_rot, axis=1, keepdims=True)
            X_rot = X_rot / np.maximum(norms, 1e-12)
            rot_ents.append(compute_sign_entropy(X_rot))
            rot_recs.append(bq2_recall(X_rot, n_queries=100, top_k=10))

        d_ent = np.mean(rot_ents) - orig_ent
        d_rec = np.mean(rot_recs) - orig_rec
        print(f"  {name:<12} {D:>5} {orig_ent:>9.4f} {np.mean(rot_ents):>9.4f} {d_ent:>+8.4f} "
              f"{orig_rec:>9.4f} {np.mean(rot_recs):>9.4f} {d_rec:>+8.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 B: RaBitQ 线性估计器 vs Hamming BQ (原 probe 17)
# ═══════════════════════════════════════════════════════════════════

def _hamming_bq_scores(query, db_vecs):
    """Hamming BQ 得分 = sign 匹配数。"""
    return ((query > 0) == (db_vecs > 0)).sum(axis=1).astype(np.float32)


def _rabitq_precompute(db_vecs, centroid=None):
    """RaBitQ per-vector 线性校正因子预计算。"""
    N, D = db_vecs.shape
    if centroid is None:
        centroid = np.zeros(D, dtype=np.float32)

    residuals = db_vecs - centroid
    x_bar = bq_sign_encode(residuals)
    r_norm_sq = np.sum(residuals ** 2, axis=1)
    ip_r_xbar = np.sum(residuals * x_bar, axis=1)
    ip_r_xbar = np.where(np.abs(ip_r_xbar) < 1e-10, 1e-10, ip_r_xbar)
    ip_c_xbar = x_bar @ centroid
    ip_r_c = residuals @ centroid
    f_add = 1.0 - ip_r_c + r_norm_sq * ip_c_xbar / ip_r_xbar
    f_rescale = -r_norm_sq / ip_r_xbar
    return f_add, f_rescale, x_bar


def _rabitq_estimate(query, x_bar, f_add, f_rescale, centroid=None):
    """RaBitQ 距离估计 (IP metric → 相似度得分)。"""
    if centroid is None:
        centroid = np.zeros(query.shape[0], dtype=np.float32)
    q_r = query - centroid
    g_add = -np.dot(q_r, centroid)
    ip_qr_xbar = x_bar @ q_r
    k1xsumq = -0.5 * np.sum(q_r)
    return -(f_add + g_add + f_rescale * (ip_qr_xbar + k1xsumq))


def exp_b_rabitq_vs_hamming():
    """RaBitQ 线性估计器 vs Hamming BQ：4 方法对比。"""
    print(f"\n{'=' * 60}")
    print("实验 B: RaBitQ 线性估计器 vs Hamming BQ")
    print(f"{'=' * 60}")

    datasets = ["cohere", "random", "gist960", "arxiv_nomic"]
    N_SAMPLE = 10_000
    N_QUERIES = 300
    K = 10

    print(f"  {'Dataset':<16} {'Ham':>8} {'Ham+Rot':>8} {'RaB':>8} "
          f"{'RaB+Rot':>8} {'IVF+RaB':>8}")
    print(f"  {'─'*60}")

    for name in datasets:
        try:
            vecs = load_dataset(name, n_sample=N_SAMPLE)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")
            continue

        D = vecs.shape[1]
        N = len(vecs)
        np.random.seed(42)
        q_idx = np.random.choice(N, min(N_QUERIES, N), replace=False)

        # 旋转
        Q, _ = np.linalg.qr(np.random.randn(D, D).astype(np.float32))
        vecs_rot = vecs @ Q.T

        def _recall(db, score_fn):
            recs = []
            for qi in q_idx:
                exact = db @ db[qi]; exact[qi] = -999
                gt = set(np.argsort(exact)[-K:])
                s = score_fn(qi, db[qi]); s[qi] = -999
                pred = set(np.argsort(s)[-K:])
                recs.append(len(gt & pred) / K)
            return np.mean(recs)

        # A: Hamming 不旋转
        rec_A = _recall(vecs, lambda qi, q: _hamming_bq_scores(q, vecs))
        # A2: Hamming 旋转
        rec_A2 = _recall(vecs_rot, lambda qi, q: _hamming_bq_scores(q, vecs_rot))
        # B: RaBitQ 不旋转
        fa, fr, xb = _rabitq_precompute(vecs)
        rec_B = _recall(vecs, lambda qi, q: _rabitq_estimate(q, xb, fa, fr))
        # C: RaBitQ 旋转
        fa_r, fr_r, xb_r = _rabitq_precompute(vecs_rot)
        rec_C = _recall(vecs_rot, lambda qi, q: _rabitq_estimate(q, xb_r, fa_r, fr_r))

        print(f"  {name:<16} {rec_A:>8.4f} {rec_A2:>8.4f} {rec_B:>8.4f} {rec_C:>8.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 C: 旋转效果驱动因子 (原 probe 21b)
# ═══════════════════════════════════════════════════════════════════

def exp_c_rotation_drivers():
    """分析 sign_entropy vs CV(σ) 哪个更能预测旋转效果 ΔF。"""
    print(f"\n{'=' * 60}")
    print("实验 C: 旋转效果驱动因子 (sign_entropy vs CV(σ))")
    print(f"{'=' * 60}")

    datasets = ["cohere", "minilm", "arxiv_nomic", "codesearch_jina",
                "landmark_nomic", "landmark_dino", "coco_nomic",
                "dbpedia3072", "dbpedia1536", "gist960", "random"]
    N_SAMPLE = 20_000

    records = []
    print(f"  {'Dataset':<18} {'D':>5} {'CV(σ)':>7} {'H_sign':>7} "
          f"{'F_orig':>7} {'F_rot':>7} {'ΔF':>7}")
    print(f"  {'─'*64}")

    for ds_name in datasets:
        try:
            vecs = load_dataset(ds_name, n_sample=N_SAMPLE)
        except Exception as e:
            continue

        D = vecs.shape[1]
        sigma = vecs.std(axis=0)
        cv = np.std(sigma) / np.mean(sigma)
        h_sign = compute_sign_entropy(vecs)

        f_orig = compute_f_batched(vecs, n_pairs=2_000_000)

        # 旋转
        Q = random_orthogonal(D, seed=777)
        vecs_rot = vecs @ Q.T
        norms = np.linalg.norm(vecs_rot, axis=1, keepdims=True)
        vecs_rot = vecs_rot / np.maximum(norms, 1e-12)
        f_rot = compute_f_batched(vecs_rot, n_pairs=2_000_000)

        delta_f = f_rot - f_orig
        print(f"  {ds_name:<18} {D:>5} {cv:>7.4f} {h_sign:>7.4f} "
              f"{f_orig:>7.4f} {f_rot:>7.4f} {delta_f:>+7.4f}")
        records.append({"cv": cv, "h_sign": h_sign, "delta_f": delta_f})

    if len(records) >= 4:
        cvs = [r["cv"] for r in records]
        hs = [r["h_sign"] for r in records]
        dfs = [r["delta_f"] for r in records]
        rho_cv, p_cv = spearmanr(cvs, dfs)
        rho_h, p_h = spearmanr(hs, dfs)
        print(f"\n  ρ(CV(σ), ΔF) = {rho_cv:+.4f} (p={p_cv:.4f})")
        print(f"  ρ(H_sign, ΔF) = {rho_h:+.4f} (p={p_h:.4f})")
        winner = "sign_entropy" if abs(rho_h) > abs(rho_cv) else "CV(σ)"
        print(f"  → 更强驱动因子: {winner}")


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  04 - 旋转二象性 (Rotation Duality)")
    print("  合并自: probe 8, 17, 21b")
    print("=" * 70)

    exp_a_rotation_ablation()
    exp_b_rabitq_vs_hamming()
    exp_c_rotation_drivers()

    print(f"\n{'=' * 70}")
    print("全部实验完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
