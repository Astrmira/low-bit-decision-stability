#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 - Stein 保真度理论 (Stein Fidelity)

合并自: probe16_F_closure, probe16b-h (8个变体), probe18_magnitude_gain

对应论文:
  §4  Stein's Lemma → 保真度封闭公式
  §5  Magnitude bit 信息增益

本文件包含:
  实验 A: F 基础对角假设 (SNR → F) (原 probe 16)
  实验 B: F 按角度差分层 (原 probe 16b)
  实验 C: 近邻对差异结构 (sign flip 集中度) (原 probe 16c)
  实验 D: 完整协方差 vs 对角 — F gap 来源 (原 probe 16d) ★关键发现
  实验 E: 低秩近似 off-diagonal (原 probe 16e)
  实验 F: 穷举双变量回归 (原 probe 16f + 16f_deep)
  实验 G: 跨维度泛化验证 (原 probe 16g)
  实验 H: 高阶矩 / Copula (原 probe 16h)
  实验 I: Magnitude bit 信息增益 (原 probe 18)
"""

import sys
import numpy as np
from scipy.stats import spearmanr, norm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (load_dataset, compute_sign_entropy, compute_snr,
                   bq2_encode, bq2_fidelity, bq2_recall,
                   random_orthogonal, RESULTS_DIR, PHASE5_768)


# ═══════════════════════════════════════════════════════════════════
#  共享工具
# ═══════════════════════════════════════════════════════════════════

def generate_aniso_gaussian(N, D, mu, sigma, seed=42):
    """N(mu, diag(sigma²)) → L2 归一化。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D).astype(np.float32) * sigma[np.newaxis, :] + mu[np.newaxis, :]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def generate_full_cov_gaussian(N, Sigma, mu=None, seed=42):
    """N(mu, Σ) → L2 归一化。"""
    rng = np.random.RandomState(seed)
    D = Sigma.shape[0]
    if mu is None:
        mu = np.zeros(D, dtype=np.float64)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-8)
        L = np.linalg.cholesky(eigvecs @ np.diag(eigvals) @ eigvecs.T)
    X = (rng.randn(N, D) @ L.T + mu).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def bq2_fidelity_fast(X, n_pairs=200_000):
    """快速计算 BQ2 保真度 F (Spearman ρ)。"""
    N, D = X.shape
    rng = np.random.RandomState(42)
    ia = rng.randint(0, N, n_pairs)
    ib = rng.randint(0, N, n_pairs)
    m = ia != ib; ia, ib = ia[m], ib[m]
    cos_sim = np.sum(X[ia]*X[ib], axis=1)
    signs = (X > 0).astype(np.int8)
    abs_X = np.abs(X)
    alphas = abs_X.mean(axis=1, keepdims=True)
    strongs = (abs_X > alphas).astype(np.int8)
    sm = (signs[ia] == signs[ib])
    w = (1 + strongs[ia]) * (1 + strongs[ib])
    bq = (sm * w).sum(axis=1).astype(np.float64)
    rho, _ = spearmanr(cos_sim, bq)
    return float(rho)


def bq1_fidelity_fast(X, n_pairs=200_000):
    """快速计算 BQ1 保真度 F (1-bit sign only)。"""
    N, D = X.shape
    rng = np.random.RandomState(42)
    ia = rng.randint(0, N, n_pairs)
    ib = rng.randint(0, N, n_pairs)
    m = ia != ib; ia, ib = ia[m], ib[m]
    cos_sim = np.sum(X[ia]*X[ib], axis=1)
    signs = (X > 0).astype(np.int8)
    bq = (signs[ia] == signs[ib]).sum(axis=1).astype(np.float64)
    rho, _ = spearmanr(cos_sim, bq)
    return float(rho)


# ═══════════════════════════════════════════════════════════════════
#  实验 A: F 基础对角假设 (原 probe 16)
# ═══════════════════════════════════════════════════════════════════

def exp_a_diag_closure():
    """SNR → F: 对角假设下 F 的封闭近似。"""
    print(f"\n{'=' * 60}")
    print("实验 A: F 基础对角假设 (SNR → F)")
    print(f"{'=' * 60}")

    N = 20_000
    datasets = ["cohere", "bge_m3", "minilm", "gist960", "random"]

    for name in datasets:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        D = X.shape[1]
        F_real = bq2_fidelity_fast(X)
        # 对角合成
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        X_diag = generate_aniso_gaussian(N, D, mu, sigma)
        F_diag = bq2_fidelity_fast(X_diag)
        gap = F_real - F_diag
        print(f"  {name:<12}: F_real={F_real:.4f}, F_diag={F_diag:.4f}, gap={gap:+.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 B: F 按角度差分层 (原 probe 16b)
# ═══════════════════════════════════════════════════════════════════

def exp_b_per_gap():
    """按 cosine gap 分层的 BQ2 排序准确率。"""
    print(f"\n{'=' * 60}")
    print("实验 B: F 按角度差分层")
    print(f"{'=' * 60}")

    N = 20_000
    for name in ["cohere", "gist960"]:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        rng = np.random.RandomState(42)
        n_trip = 200_000
        ia = rng.randint(0, N, n_trip)
        ib = rng.randint(0, N, n_trip)
        ic = rng.randint(0, N, n_trip)
        m = (ia != ib) & (ib != ic) & (ia != ic)
        ia, ib, ic = ia[m], ib[m], ic[m]
        cos_ab = np.sum(X[ia]*X[ib], axis=1)
        cos_ac = np.sum(X[ia]*X[ic], axis=1)
        gap = np.abs(cos_ab - cos_ac)
        signs = (X > 0).astype(np.int8)
        abs_X = np.abs(X)
        alphas = abs_X.mean(axis=1, keepdims=True)
        strongs = (abs_X > alphas).astype(np.int8)
        def bq_s(a, b):
            sm = (signs[a] == signs[b])
            w = (1 + strongs[a]) * (1 + strongs[b])
            return (sm * w).sum(axis=1)
        bq_ab, bq_ac = bq_s(ia, ib), bq_s(ia, ic)
        correct = ((cos_ab > cos_ac) == (bq_ab > bq_ac))
        thresholds = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
        print(f"  {name}:")
        for t in thresholds:
            mask = gap > t
            if mask.sum() > 100:
                acc = correct[mask].mean()
                print(f"    gap>{t:.2f}: acc={acc:.4f} ({mask.sum()} triplets)")


# ═══════════════════════════════════════════════════════════════════
#  实验 C: 近邻对差异结构 (原 probe 16c)
# ═══════════════════════════════════════════════════════════════════

def exp_c_diff_structure():
    """近邻对的 sign flip 集中度分析。"""
    print(f"\n{'=' * 60}")
    print("实验 C: 近邻对差异结构")
    print(f"{'=' * 60}")

    N = 10_000
    for name in ["cohere", "gist960", "random"]:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        D = X.shape[1]
        rng = np.random.RandomState(42)
        n_q = 500
        q_idx = rng.choice(N, n_q, replace=False)
        gini_list = []
        for qi in q_idx:
            sims = X @ X[qi]; sims[qi] = -2.0
            nn = np.argpartition(-sims, 10)[:10]
            flips = (np.sign(X[qi]) != np.sign(X[nn])).mean(axis=0)
            sorted_f = np.sort(flips)
            n = len(sorted_f)
            g = 1 - 2*np.sum(np.cumsum(sorted_f))/(n*np.sum(sorted_f+1e-10)) + 1/n
            gini_list.append(g)
        print(f"  {name:<12}: flip Gini={np.mean(gini_list):.4f}±{np.std(gini_list):.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 D: 完整协方差 vs 对角 (原 probe 16d) ★关键发现
# ═══════════════════════════════════════════════════════════════════

def exp_d_full_cov():
    """F_diag vs F_full_cov vs F_real → off-diagonal 的贡献。"""
    print(f"\n{'=' * 60}")
    print("实验 D: 完整协方差 vs 对角 (关键发现)")
    print(f"{'=' * 60}")

    N = 20_000
    print(f"  {'Dataset':<12} {'F_diag':>7} {'F_cov':>7} {'F_real':>7} {'gap_cov':>8} {'gap_diag':>9}")

    for name in ["cohere", "bge_m3", "minilm", "gist960", "random"]:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        D = X.shape[1]
        mu_est = X.mean(axis=0).astype(np.float64)
        Sigma_est = np.cov(X.T).astype(np.float64)
        sigma_est = np.sqrt(np.diag(Sigma_est))

        F_real = bq2_fidelity_fast(X)
        X_diag = generate_aniso_gaussian(N, D, mu_est.astype(np.float32), sigma_est.astype(np.float32))
        F_diag = bq2_fidelity_fast(X_diag)
        X_cov = generate_full_cov_gaussian(N, Sigma_est, mu_est)
        F_cov = bq2_fidelity_fast(X_cov)

        print(f"  {name:<12} {F_diag:>7.4f} {F_cov:>7.4f} {F_real:>7.4f} "
              f"{F_real-F_cov:>+8.4f} {F_real-F_diag:>+9.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 E: 低秩近似 off-diagonal (原 probe 16e)
# ═══════════════════════════════════════════════════════════════════

def exp_e_lowrank():
    """低秩近似 off-diagonal 对 F 的解释力。"""
    print(f"\n{'=' * 60}")
    print("实验 E: 低秩近似 off-diagonal")
    print(f"{'=' * 60}")

    N = 20_000
    for name in ["cohere", "bge_m3"]:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        D = X.shape[1]
        Sigma = np.cov(X.T)
        diag_energy = np.sum(np.diag(Sigma)**2)
        off = Sigma.copy(); np.fill_diagonal(off, 0)
        full_off = np.sum(off**2)
        ranks = [1, 5, 10, 20, 50]
        U, S, Vt = np.linalg.svd(off)
        print(f"  {name}: off/diag ratio={np.sqrt(full_off/diag_energy):.4f}")
        for r in ranks:
            approx = (U[:,:r] * S[:r]) @ Vt[:r,:]
            residual = np.sum((off - approx)**2)
            explained = 1 - residual / full_off
            print(f"    rank-{r:>3}: explained={explained*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════
#  实验 F: 穷举双变量回归 (原 probe 16f + 16f_deep)
# ═══════════════════════════════════════════════════════════════════

def exp_f_combo_regression():
    """穷举统计量组合 → F 的最佳预测因子。"""
    print(f"\n{'=' * 60}")
    print("实验 F: 穷举双变量回归")
    print(f"{'=' * 60}")

    N = 20_000
    records = []
    for name in PHASE5_768 + ["bge_m3", "minilm", "gist960", "dbpedia1536", "dbpedia3072"]:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        D = X.shape[1]
        sigma = X.std(axis=0)
        Sigma = np.cov(X[:min(10000, N)].T)
        snr = compute_snr(X)
        sign_ent = compute_sign_entropy(X)
        F = bq2_fidelity_fast(X)
        diag_e = np.sum(np.diag(Sigma)**2)
        off_diag = np.sum(Sigma**2) - diag_e
        r_off = np.sqrt(max(off_diag, 0) / diag_e)
        cv = np.std(sigma) / np.mean(sigma)
        records.append({"name": name, "D": D, "F": F, "sign_ent": sign_ent,
                        "cv": cv, "log_r_off": np.log(max(r_off, 1e-10)),
                        "std_snr": np.std(np.abs(snr))})

    if len(records) < 4:
        print("  数据集不足，跳过"); return

    Fs = np.array([r["F"] for r in records])
    features = {k: np.array([r[k] for r in records])
                for k in ["sign_ent", "cv", "log_r_off", "std_snr"]}

    print(f"\n  单变量 Spearman:")
    for fname, vals in features.items():
        rho, p = spearmanr(vals, Fs)
        print(f"    ρ({fname}, F) = {rho:+.4f} (p={p:.4f})")

    # 双变量最佳
    keys = list(features.keys())
    best_r2, best_pair = 0, None
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            A = np.column_stack([features[keys[i]], features[keys[j]], np.ones(len(Fs))])
            beta, _, _, _ = np.linalg.lstsq(A, Fs, rcond=None)
            pred = A @ beta
            ss_res = np.sum((Fs - pred)**2)
            ss_tot = np.sum((Fs - Fs.mean())**2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            if r2 > best_r2:
                best_r2, best_pair = r2, (keys[i], keys[j])
    print(f"\n  最佳双变量: ({best_pair[0]}, {best_pair[1]}) → R²={best_r2:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 G: 跨维度泛化 (原 probe 16g)
# ═══════════════════════════════════════════════════════════════════

def exp_g_cross_dim():
    """跨维度泛化验证。"""
    print(f"\n{'=' * 60}")
    print("实验 G: 跨维度泛化验证")
    print(f"{'=' * 60}")

    N = 20_000
    cross_dim_ds = [("minilm", 384), ("bge_m3", 1024), ("gist960", 960),
                    ("cohere", 768), ("dbpedia1536", 1536)]
    for name, D_exp in cross_dim_ds:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        F = bq2_fidelity_fast(X)
        ent = compute_sign_entropy(X)
        print(f"  {name:<14} D={X.shape[1]:>5}: F={F:.4f}, entropy={ent:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 H: 高阶矩 / Copula (原 probe 16h)
# ═══════════════════════════════════════════════════════════════════

def exp_h_higher_moments():
    """高阶矩和 Gaussian Copula 对 F 的影响。"""
    print(f"\n{'=' * 60}")
    print("实验 H: 高阶矩 / Copula 影响")
    print(f"{'=' * 60}")

    N = 20_000
    for name in ["cohere", "bge_m3", "gist960"]:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        D = X.shape[1]
        F_real = bq2_fidelity_fast(X)
        # Copula: 匹配边际 + Gaussian 依赖
        Sigma = np.cov(X.T)
        X_full = generate_full_cov_gaussian(N, Sigma, X.mean(axis=0))
        F_copula = bq2_fidelity_fast(X_full)
        # 打乱边际（保留 marginals，破坏 joint）
        X_shuf = X.copy()
        rng = np.random.RandomState(42)
        for d in range(D):
            rng.shuffle(X_shuf[:, d])
        norms = np.linalg.norm(X_shuf, axis=1, keepdims=True)
        X_shuf = X_shuf / np.maximum(norms, 1e-12)
        F_shuf = bq2_fidelity_fast(X_shuf)
        print(f"  {name:<12}: F_real={F_real:.4f}, F_copula={F_copula:.4f}, "
              f"F_shuf={F_shuf:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 I: Magnitude bit 信息增益 + Per-vector 校正 (原 probe 18)
# ═══════════════════════════════════════════════════════════════════

def _xbar_1bit(vecs):
    """RaBitQ x_bar: sign(x) - 0.5 → {-0.5, +0.5}。"""
    return (vecs > 0).astype(np.float32) - 0.5

def _xbar_2bit(vecs):
    """2-bit x_bar: (1+mag) × (sign-0.5) → {-1, -0.5, +0.5, +1}。"""
    signs = (vecs > 0).astype(np.float32) - 0.5
    alpha = np.mean(np.abs(vecs), axis=1, keepdims=True)
    mags = (np.abs(vecs) > alpha).astype(np.float32)
    return (1 + mags) * signs

def _rabitq_factors(vecs, xb):
    """Per-vector RaBitQ 校正因子 (centroid=0)。"""
    x_norm_sq = np.sum(vecs ** 2, axis=1)
    ip_x_xbar = np.sum(vecs * xb, axis=1)
    ip_x_xbar = np.where(np.abs(ip_x_xbar) < 1e-10, 1e-10, ip_x_xbar)
    f_add = x_norm_sq
    f_rescale = -x_norm_sq / ip_x_xbar
    return f_add, f_rescale

def exp_i_magnitude_gain():
    """4 方法对比: 1-bit / 2-bit / 1-bit+校正 / 2-bit+校正 + 信息增益分解。"""
    print(f"\n{'=' * 60}")
    print("实验 I: Magnitude bit + Per-vector 校正 (4 方法 ABCD)")
    print(f"{'=' * 60}")

    N = 10_000
    N_PAIRS = 50_000
    datasets = ["cohere", "bge_m3", "minilm", "gist960",
                "random", "arxiv_nomic", "dbpedia3072"]

    for name in datasets:
        try:
            X = load_dataset(name, N)
        except Exception: continue
        D = X.shape[1]
        sigma = X.std(axis=0)
        cv = np.std(sigma) / np.mean(sigma)

        # 预编码
        signs_1 = (X > 0).astype(np.float32)
        signs_2 = (X > 0).astype(np.float32)
        alpha = np.mean(np.abs(X), axis=1, keepdims=True)
        mags_2 = (np.abs(X) > alpha).astype(np.float32)
        xb1 = _xbar_1bit(X)
        xb2 = _xbar_2bit(X)
        fa1, fr1 = _rabitq_factors(X, xb1)
        fa2, fr2 = _rabitq_factors(X, xb2)

        # 批量 F 计算
        np.random.seed(42)
        ia = np.random.randint(0, N, N_PAIRS)
        ib = np.random.randint(0, N, N_PAIRS)
        m = ia != ib; ia, ib = ia[m], ib[m]
        cos_sim = np.sum(X[ia]*X[ib], axis=1)

        # A: 1-bit Hamming
        sA = (signs_1[ia] == signs_1[ib]).sum(axis=1).astype(np.float32)
        F_A, _ = spearmanr(cos_sim, sA)

        # B: 2-bit 加权 Hamming
        sa = 2.0 * (signs_2[ia] == signs_2[ib]).astype(np.float32) - 1.0
        wt = (1 + mags_2[ia]) * (1 + mags_2[ib])
        sB = (sa * wt).sum(axis=1)
        F_B, _ = spearmanr(cos_sim, sB)

        # C: 1-bit + per-vector 校正
        ip_q_xb1 = np.sum(X[ia] * xb1[ib], axis=1)
        k1_sumq = -0.5 * np.sum(X[ia], axis=1)
        sC = -(fa1[ib] + fr1[ib] * (ip_q_xb1 + k1_sumq))
        F_C, _ = spearmanr(cos_sim, sC)

        # D: 2-bit + per-vector 校正
        ip_q_xb2 = np.sum(X[ia] * xb2[ib], axis=1)
        sD = -(fa2[ib] + fr2[ib] * ip_q_xb2)
        F_D, _ = spearmanr(cos_sim, sD)

        # 信息增益分解
        mag_gain = F_B - F_A
        corr_gain = F_C - F_A
        fusion_gain = F_D - F_A
        overlap = mag_gain + corr_gain - fusion_gain

        print(f"\n  {name} (D={D}, CV={cv:.4f}):")
        print(f"    F_A(1bit)={F_A:.4f}  F_B(2bit)={F_B:.4f}  "
              f"F_C(1b+corr)={F_C:.4f}  F_D(2b+corr)={F_D:.4f}")
        print(f"    mag_gain={mag_gain:+.4f}  corr_gain={corr_gain:+.4f}  "
              f"fusion={fusion_gain:+.4f}  overlap={overlap:+.4f}")


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  03 - Stein 保真度理论 (Stein Fidelity)")
    print("  合并自: probe 16 全系列 (9个) + probe 18")
    print("=" * 70)

    exp_a_diag_closure()
    exp_b_per_gap()
    exp_c_diff_structure()
    exp_d_full_cov()
    exp_e_lowrank()
    exp_f_combo_regression()
    exp_g_cross_dim()
    exp_h_higher_moments()
    exp_i_magnitude_gain()

    print(f"\n{'=' * 70}")
    print("全部实验完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
