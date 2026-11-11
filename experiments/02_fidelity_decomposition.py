#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 - F/G 保真度分解 (Fidelity Decomposition)

合并自: probe13_cluster_causal.py, probe14_fidelity_window.py, probe15_g_parameterization.py

对应论文:
  §5  Stein Fidelity → Recall = F × G 分解
  §9  实验 (Table 2: 同维跨模型对比)

本文件包含:
  实验 A: 簇结构因果验证 — 真实/iid/K-簇/shuffle 的 recall 对比 (原 probe 13)
  实验 B: 保真度窗口 — sign entropy vs BQ fidelity 的最优区间 (原 probe 14)
  实验 C: G 参数化 — 768 维同维跨模型的 F/G/Recall 分解 (原 probe 15)
"""

import sys
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (load_dataset, compute_sign_entropy, compute_snr,
                   bq2_encode, bq2_fidelity, bq2_recall,
                   RESULTS_DIR, PHASE5_768)


# ═══════════════════════════════════════════════════════════════════
#  共享工具
# ═══════════════════════════════════════════════════════════════════

def bq2_recall_local(X, n_queries=200, top_k=10):
    """BQ2 recall（本地快速版，用于合成数据）。"""
    N, D = X.shape
    rng = np.random.RandomState(42)
    q_idx = rng.choice(N, n_queries, replace=False)

    signs = (X > 0).astype(np.float32)
    abs_X = np.abs(X)
    thresholds = np.median(abs_X, axis=0, keepdims=True)
    mags = (abs_X > thresholds).astype(np.float32)

    recalls = []
    for qi in q_idx:
        exact = X @ X[qi]
        exact[qi] = -2.0
        gt = set(np.argpartition(-exact, top_k)[:top_k])

        s_agree = (signs * signs[qi:qi+1] + (1 - signs) * (1 - signs[qi:qi+1]))
        m_agree = (mags * mags[qi:qi+1] + (1 - mags) * (1 - mags[qi:qi+1]))
        score = np.sum(s_agree * (1.0 + 0.5 * m_agree) - (1 - s_agree), axis=1)
        score[qi] = -1e9
        bq_top = set(np.argpartition(-score, top_k)[:top_k])
        recalls.append(len(gt & bq_top) / top_k)
    return float(np.mean(recalls))


def bq2_fidelity_spearman(X, n_pairs=50000):
    """BQ2 编码保真度 = 成对 Spearman ρ（本地快速版）。"""
    N, D = X.shape
    rng = np.random.RandomState(42)
    idx_a = rng.choice(N, n_pairs, replace=True)
    idx_b = rng.choice(N, n_pairs, replace=True)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    exact_sim = np.sum(X[idx_a] * X[idx_b], axis=1)

    signs = (X > 0).astype(np.float32)
    abs_X = np.abs(X)
    thresholds = np.median(abs_X, axis=0, keepdims=True)
    mags = (abs_X > thresholds).astype(np.float32)

    s_a, s_b = signs[idx_a], signs[idx_b]
    m_a, m_b = mags[idx_a], mags[idx_b]
    s_agree = (s_a * s_b + (1 - s_a) * (1 - s_b))
    m_agree = (m_a * m_b + (1 - m_a) * (1 - m_b))
    bq_score = np.sum(s_agree * (1.0 + 0.5 * m_agree) - (1 - s_agree), axis=1)

    rho, _ = spearmanr(exact_sim, bq_score)
    return float(rho)


# ═══════════════════════════════════════════════════════════════════
#  实验 A: 簇结构因果验证 (原 probe 13)
# ═══════════════════════════════════════════════════════════════════

def _generate_iid(X_real, seed=42):
    """从 per-dim marginals 生成 iid 合成数据。"""
    N, D = X_real.shape
    mu = np.mean(X_real, axis=0)
    sigma = np.std(X_real, axis=0)
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, D).astype(np.float32) * sigma + mu
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, 1e-12)


def _generate_clustered(X_real, K, seed=42):
    """从 K-簇结构生成合成数据。"""
    N, D = X_real.shape
    rng = np.random.RandomState(seed)
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=seed,
                            batch_size=min(N, 5000), n_init=3)
    labels = kmeans.fit_predict(X_real)
    X_synth = np.zeros_like(X_real)
    for k in range(K):
        mask = labels == k
        n_k = np.sum(mask)
        if n_k < 2:
            X_synth[mask] = X_real[mask]
            continue
        mu_k = np.mean(X_real[mask], axis=0)
        sigma_k = np.std(X_real[mask], axis=0)
        Z_k = rng.randn(n_k, D).astype(np.float32) * sigma_k + mu_k
        norms = np.linalg.norm(Z_k, axis=1, keepdims=True)
        X_synth[mask] = Z_k / np.maximum(norms, 1e-12)
    return X_synth


def _shuffle_dims(X_real, seed=42):
    """对每个维度独立 shuffle，破坏联合结构。"""
    N, D = X_real.shape
    rng = np.random.RandomState(seed)
    X_shuf = X_real.copy()
    for d in range(D):
        rng.shuffle(X_shuf[:, d])
    norms = np.linalg.norm(X_shuf, axis=1, keepdims=True)
    return X_shuf / np.maximum(norms, 1e-12)


def exp_a_cluster_causal():
    """簇结构因果验证：真实/iid/K-簇/shuffle。"""
    print(f"\n{'=' * 60}")
    print("实验 A: 簇结构因果验证")
    print(f"{'=' * 60}")

    N_SAMPLE = 10_000
    X_real = load_dataset("cohere", N_SAMPLE)
    D = X_real.shape[1]

    rec_real = bq2_recall_local(X_real)
    rec_iid = bq2_recall_local(_generate_iid(X_real))
    rec_shuf = bq2_recall_local(_shuffle_dims(X_real))

    print(f"  真实 Cohere:   recall={rec_real:.4f}")
    print(f"  iid 合成:      recall={rec_iid:.4f}")
    print(f"  维度 shuffle:  recall={rec_shuf:.4f}")

    print(f"\n  K-簇合成扫描:")
    K_values = [1, 5, 10, 20, 50, 100, 200, 500]
    for K in K_values:
        rec = bq2_recall_local(_generate_clustered(X_real, K))
        print(f"    K={K:>4d}: recall={rec:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 B: 保真度窗口 (原 probe 14)
# ═══════════════════════════════════════════════════════════════════

def _generate_controlled_entropy(N, D, target_entropy, seed=42):
    """生成具有精确控制 sign entropy 的合成数据。"""
    from scipy.stats import norm as norm_dist

    rng = np.random.RandomState(seed)

    def compute_ent(snr_val):
        pos_rate = norm_dist.cdf(snr_val)
        ent = -(pos_rate * np.log2(np.clip(pos_rate, 1e-10, 1)) +
                (1 - pos_rate) * np.log2(np.clip(1 - pos_rate, 1e-10, 1)))
        return float(ent)

    # 二分法找 |SNR| 使得 entropy ≈ target
    lo, hi = 0.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if compute_ent(mid) > target_entropy:
            lo = mid
        else:
            hi = mid
    uniform_snr = (lo + hi) / 2

    signs = rng.choice([-1, 1], size=D).astype(np.float32)
    mu = np.full(D, uniform_snr, dtype=np.float32) * signs
    sigma = np.ones(D, dtype=np.float32)

    Z = rng.randn(N, D).astype(np.float32) * sigma + mu
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    X = Z / np.maximum(norms, 1e-12)
    return X, compute_ent(uniform_snr)


def exp_b_fidelity_window():
    """BQ 保真度 vs sign entropy 的最优窗口。"""
    print(f"\n{'=' * 60}")
    print("实验 B: 保真度窗口 (Fidelity vs Sign Entropy)")
    print(f"{'=' * 60}")

    D = 768
    N = 5000
    target_ents = [0.10, 0.30, 0.50, 0.60, 0.70, 0.75, 0.80,
                   0.85, 0.90, 0.95, 1.00]

    print(f"  {'target_ent':>10} {'actual_ent':>10} {'fidelity':>10}")
    print(f"  {'─' * 32}")

    best_ent, best_fid = 0, 0
    for te in target_ents:
        X, actual_ent = _generate_controlled_entropy(N, D, te)
        fid = bq2_fidelity_spearman(X, n_pairs=30000)
        print(f"  {te:>10.2f} {actual_ent:>10.4f} {fid:>10.4f}")
        if fid > best_fid:
            best_fid = fid
            best_ent = actual_ent

    print(f"\n  最优窗口: entropy ≈ {best_ent:.4f}, fidelity = {best_fid:.4f}")

    # 真实数据叠加验证
    print(f"\n  真实数据对照:")
    for name in ["cohere", "bge_m3", "minilm", "random"]:
        try:
            X = load_dataset(name, N)
            fid = bq2_fidelity_spearman(X, n_pairs=30000)
            ent = compute_sign_entropy(X)
            print(f"    {name:<12}: entropy={ent:.4f}, fidelity={fid:.4f}")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
#  实验 C: G 参数化 — 768 维同维跨模型 (原 probe 15)
# ═══════════════════════════════════════════════════════════════════

def _compute_g_proxies(X, n_queries=500, top_k=10):
    """计算 G-proxy 候选指标 (nn_cos, sim_gap, cluster_density)。"""
    N, D = X.shape
    rng = np.random.RandomState(42)
    q_idx = rng.choice(N, min(n_queries, N), replace=False)

    nn_cos_list, sim_gap_list = [], []
    for qi in q_idx:
        sims = X @ X[qi]
        sims[qi] = -2.0
        top100_idx = np.argpartition(-sims, 100)[:100]
        top100_sims = sims[top100_idx]
        topk_idx = top100_idx[np.argpartition(-top100_sims, top_k)[:top_k]]
        topk_sims = sims[topk_idx]
        nn_cos_list.append(float(np.mean(topk_sims)))
        sim_gap_list.append(float(np.mean(topk_sims) - np.mean(top100_sims)))

    # 簇密度
    km = MiniBatchKMeans(n_clusters=100, random_state=42, batch_size=2048, n_init=3)
    labels = km.fit_predict(X)
    intra_cos = []
    for c in range(100):
        members = X[labels == c]
        if len(members) > 1:
            center = members.mean(axis=0)
            center /= max(np.linalg.norm(center), 1e-10)
            intra_cos.append(float(np.mean(members @ center)))

    return {
        "nn_cos": float(np.mean(nn_cos_list)),
        "sim_gap": float(np.mean(sim_gap_list)),
        "cluster_density": float(np.mean(intra_cos)) if intra_cos else 0.0,
    }


def exp_c_g_parameterization():
    """768 维同维跨模型的 F/G/Recall 分解。"""
    print(f"\n{'=' * 60}")
    print("实验 C: G 参数化 — 768 维同维跨模型")
    print(f"{'=' * 60}")

    N_SAMPLE = 50_000
    results = []

    for name in PHASE5_768:
        try:
            X = load_dataset(name, N_SAMPLE)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")
            continue

        sign_ent = compute_sign_entropy(X)
        F = bq2_fidelity_spearman(X, n_pairs=50000)
        recall = bq2_recall_local(X, n_queries=500, top_k=10)
        G = recall / max(F, 1e-10)
        gp = _compute_g_proxies(X, n_queries=500, top_k=10)

        results.append({"name": name, "F": F, "recall": recall,
                        "G": G, "sign_ent": sign_ent, **gp})

    print(f"\n  {'Dataset':<20} {'F':>6} {'Recall':>7} {'G=R/F':>7} "
          f"{'nn_cos':>7} {'sim_gap':>8} {'cluster':>8}")
    print(f"  {'─'*72}")
    for r in results:
        print(f"  {r['name']:<20} {r['F']:>6.4f} {r['recall']:>7.4f} "
              f"{r['G']:>7.4f} {r['nn_cos']:>7.4f} "
              f"{r['sim_gap']:>8.5f} {r['cluster_density']:>8.5f}")

    # G-proxy 拟合
    if len(results) >= 4:
        G_arr = np.array([r["G"] for r in results])
        for pname in ["nn_cos", "sim_gap", "cluster_density"]:
            proxy = np.array([r[pname] for r in results])
            if np.std(proxy) > 1e-10:
                rho, p = spearmanr(proxy, G_arr)
                print(f"  ρ({pname}, G) = {rho:+.4f} (p={p:.4f})")


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  02 - F/G 保真度分解 (Fidelity Decomposition)")
    print("  合并自: probe 13, 14, 15")
    print("=" * 70)

    exp_a_cluster_causal()
    exp_b_fidelity_window()
    exp_c_g_parameterization()

    print(f"\n{'=' * 70}")
    print("全部实验完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
