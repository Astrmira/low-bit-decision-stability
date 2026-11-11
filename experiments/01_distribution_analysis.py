#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 - 分布特征分析 (Distribution Analysis)

合并自: probe1-7, probe9, probe10, probe12

对应论文:
  §3  H1 假设验证 (Gaussianity, SNR → sign entropy)
  附录 D (边际高斯性、坐标独立性、SNR 拟合)

本文件包含:
  实验 A: PCA 谱衰减 + 有效维度 (原 probe 1)
  实验 B: 成对角度分布 + Beta 拟合 (原 probe 2)
  实验 C: 坐标 sign vs Random HP overlap (原 probe 3)
  实验 D: vMF Mixture 簇结构 (原 probe 4)
  实验 E: FISTA 稀疏编码 + 局部流形 (原 probe 5)
  实验 F: NMF 有效维度 (原 probe 6)
  实验 G: DPP 角度体积 (原 probe 7)
  实验 H: BQ2 vs Random HP + 逐维度 entropy (原 probe 9)
  实验 I: 各向异性高斯模型验证 (原 probe 10)
  实验 J: 数学家前置包 — Gaussianity/Independence/SNR fit/Pre-norm (原 probe 12)
"""

import sys
import numpy as np
from scipy import stats
from scipy.stats import norm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (load_dataset, save_result, compute_sign_entropy, compute_snr,
                   bq2_recall, RESULTS_DIR, PHASE1_DATASETS)


# ═══════════════════════════════════════════════════════════════════
#  实验 A: PCA 谱衰减 (原 probe 1)
# ═══════════════════════════════════════════════════════════════════

def exp_a_pca(datasets=None, n_sample=50_000):
    """PCA 谱衰减 → 有效维度。"""
    if datasets is None:
        datasets = PHASE1_DATASETS
    print(f"\n{'=' * 60}")
    print("实验 A: PCA 谱衰减")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<14} {'D':>5} {'d_90':>6} {'d_90/D':>8} {'d_99':>6} {'d_99/D':>8} {'slope':>8}")

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})"); continue
        N, D = X.shape
        X_c = X - X.mean(axis=0)
        n_comp = min(D, 512, N)
        if D > 512:
            from sklearn.utils.extmath import randomized_svd
            _, S, _ = randomized_svd(X_c, n_components=n_comp, random_state=42)
        else:
            _, S, _ = np.linalg.svd(X_c, full_matrices=False)
            S = S[:n_comp]
        eig = (S ** 2) / N
        cumvar = np.cumsum(eig) / np.sum(eig)
        d_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        d_99 = int(np.searchsorted(cumvar, 0.99) + 1)
        n_fit = int(n_comp * 0.8)
        slope, _ = np.polyfit(np.log(np.arange(1, n_fit+1)), np.log(eig[:n_fit]+1e-30), 1)
        print(f"  {name:<14} {D:>5} {d_90:>6} {d_90/D:>8.4f} {d_99:>6} {d_99/D:>8.4f} {slope:>8.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 B: 成对角度分布 (原 probe 2)
# ═══════════════════════════════════════════════════════════════════

def exp_b_angles(datasets=None, n_sample=50_000):
    """成对角度分布 + Beta 拟合。"""
    if datasets is None:
        datasets = PHASE1_DATASETS
    print(f"\n{'=' * 60}")
    print("实验 B: 成对角度分布")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<12} {'D':>5} {'mean°':>7} {'std°':>6} {'Beta_a':>7} {'Beta_b':>7} {'KS':>7}")

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        np.random.seed(42)
        n_p = 100_000
        ia, ib = np.random.randint(0, len(X), n_p), np.random.randint(0, len(X), n_p)
        m = ia != ib; ia, ib = ia[m], ib[m]
        cos = np.clip(np.sum(X[ia]*X[ib], axis=1), -1, 1)
        th = np.arccos(cos)
        th_n = th / np.pi
        ba, bb, _, _ = stats.beta.fit(th_n, floc=0, fscale=1)
        ks, _ = stats.kstest(th_n, 'beta', (ba, bb))
        print(f"  {name:<12} {X.shape[1]:>5} {np.degrees(th.mean()):>7.1f} "
              f"{np.degrees(th.std()):>6.2f} {ba:>7.2f} {bb:>7.2f} {ks:>7.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 C: Coord sign vs Random HP (原 probe 3)
# ═══════════════════════════════════════════════════════════════════

def exp_c_sign_overlap(datasets=None, n_sample=50_000):
    """坐标 sign overlap vs Random Hyperplane vs GW 理论。"""
    if datasets is None:
        datasets = PHASE1_DATASETS
    print(f"\n{'=' * 60}")
    print("实验 C: Coord Sign vs Random HP")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<12} {'coord':>8} {'rh':>8} {'gw':>8} {'KL(c||r)':>10} {'entropy':>8}")

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        N, D = X.shape
        np.random.seed(42)
        n_p = 100_000
        ia, ib = np.random.randint(0, N, n_p), np.random.randint(0, N, n_p)
        m = ia != ib; ia, ib = ia[m], ib[m]
        xa, xb = X[ia], X[ib]
        cos = np.clip(np.sum(xa*xb, axis=1), -1, 1)
        th = np.arccos(cos)
        coord = np.mean(np.sign(xa) == np.sign(xb), axis=1).mean()
        R = np.random.randn(D, D).astype(np.float32)
        R = R / np.linalg.norm(R, axis=0, keepdims=True)
        rh = np.mean(np.sign(xa@R) == np.sign(xb@R), axis=1).mean()
        gw = (1.0 - th/np.pi).mean()
        ent = compute_sign_entropy(X)
        # KL
        p = np.clip(np.mean(np.sign(xa)==np.sign(xb), axis=1), 1e-10, 1-1e-10)
        q = np.clip(np.mean(np.sign(xa@R)==np.sign(xb@R), axis=1), 1e-10, 1-1e-10)
        kl = np.mean(p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q)))
        print(f"  {name:<12} {coord:>8.5f} {rh:>8.5f} {gw:>8.5f} {kl:>10.6f} {ent:>8.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 D: vMF Mixture (原 probe 4)
# ═══════════════════════════════════════════════════════════════════

def exp_d_vmf(datasets=None, n_sample=50_000):
    """vMF Mixture 簇结构分析 (K=64)。"""
    if datasets is None:
        datasets = PHASE1_DATASETS
    print(f"\n{'=' * 60}")
    print("实验 D: vMF Mixture 簇结构 (K=64)")
    print(f"{'=' * 60}")

    from sklearn.cluster import MiniBatchKMeans

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        K = 64
        km = MiniBatchKMeans(n_clusters=K, max_iter=100, batch_size=2048, random_state=42, n_init=3)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        cn = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(cn, 1e-12)
        # κ 估计
        kappas = []
        for c in range(K):
            cl = X[labels == c]
            if len(cl) >= 5:
                mv = np.mean(cl, axis=0)
                Rb = np.linalg.norm(mv)
                Rb = min(Rb, 1 - 1e-10)
                kappas.append(Rb * (X.shape[1] - Rb**2) / (1 - Rb**2))
        print(f"  {name:<14}: κ_mean={np.mean(kappas):.1f}, κ_median={np.median(kappas):.1f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 E: FISTA 稀疏编码 (原 probe 5) — 简化版
# ═══════════════════════════════════════════════════════════════════

def exp_e_fista(datasets=None, n_sample=10_000):
    """FISTA 稀疏编码 → 局部流形维度。"""
    if datasets is None:
        datasets = ["cohere", "minilm", "gist960", "random"]
    print(f"\n{'=' * 60}")
    print("实验 E: FISTA 局部流形 (λ=0.1)")
    print(f"{'=' * 60}")

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        N, D = X.shape
        np.random.seed(42)
        n_q, k_nn, lam = 100, 50, 0.1
        q_idx = np.random.choice(N, n_q, replace=False)
        sparsities = []
        for qi in q_idx:
            sims = X @ X[qi]; sims[qi] = -2.0
            top_k = np.argpartition(-sims, k_nn)[:k_nn]
            nb = X[top_k]
            DtD = nb @ nb.T; Dtq = nb @ X[qi]
            L = np.linalg.norm(DtD, ord=2)
            if L < 1e-10: continue
            step = 1.0/L; alpha = np.zeros(k_nn); t = 1.0
            for _ in range(200):
                beta = alpha + (t-1)/(t+2)*(alpha-alpha)
                grad = DtD@beta - Dtq
                alpha = np.sign(beta-step*grad)*np.maximum(np.abs(beta-step*grad)-step*lam, 0)
                t = (1+np.sqrt(1+4*t*t))/2
            sparsities.append(np.sum(np.abs(alpha) > 1e-6))
        print(f"  {name:<14}: sparsity={np.mean(sparsities):.1f}±{np.std(sparsities):.1f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 F: NMF 有效维度 (原 probe 6) — 简化版
# ═══════════════════════════════════════════════════════════════════

def exp_f_nmf(datasets=None, n_sample=10_000):
    """NMF 有效维度。"""
    if datasets is None:
        datasets = ["cohere", "minilm", "gist960", "random"]
    print(f"\n{'=' * 60}")
    print("实验 F: NMF 有效维度")
    print(f"{'=' * 60}")

    from sklearn.decomposition import MiniBatchNMF
    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        X_nn = np.clip((X + 1.0) / 2.0, 0, None)
        k_vals = [8, 32, 128, 256]
        k_vals = [k for k in k_vals if k < X.shape[1]]
        errs = []
        for k in k_vals:
            nmf = MiniBatchNMF(n_components=k, max_iter=200, random_state=42, batch_size=512)
            W = nmf.fit_transform(X_nn)
            err = np.linalg.norm(X_nn - W@nmf.components_) / np.linalg.norm(X_nn)
            errs.append(err)
        print(f"  {name:<14}: " + " | ".join(f"k={k}→{e:.3f}" for k, e in zip(k_vals, errs)))


# ═══════════════════════════════════════════════════════════════════
#  实验 G: DPP 角度体积 (原 probe 7) — 简化版
# ═══════════════════════════════════════════════════════════════════

def exp_g_dpp(datasets=None, n_sample=10_000):
    """DPP 角度体积 → 邻域多样性。"""
    if datasets is None:
        datasets = ["cohere", "minilm", "gist960", "random"]
    print(f"\n{'=' * 60}")
    print("实验 G: DPP 角度体积 (30-NN)")
    print(f"{'=' * 60}")

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        np.random.seed(42)
        n_q, k_nn = 200, 30
        q_idx = np.random.choice(len(X), n_q, replace=False)
        log_dets = []
        for qi in q_idx:
            sims = X @ X[qi]; sims[qi] = -2.0
            top_k = np.argpartition(-sims, k_nn)[:k_nn]
            K = X[top_k] @ X[top_k].T + np.eye(k_nn)*1e-6
            sign, ld = np.linalg.slogdet(K)
            if sign > 0: log_dets.append(ld)
        ld_mean = np.mean(log_dets) if log_dets else float('nan')
        print(f"  {name:<14}: log_det/k={ld_mean/k_nn:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 H: BQ2 vs Random HP + 逐维度 entropy (原 probe 9)
# ═══════════════════════════════════════════════════════════════════

def exp_h_minilm_analysis(datasets=None, n_sample=10_000):
    """BQ2 vs Random HP recall + 条件 sign entropy。"""
    if datasets is None:
        datasets = ["cohere", "bge_m3", "minilm", "gist960", "random"]
    print(f"\n{'=' * 60}")
    print("实验 H: BQ2 vs Random HP + 逐维度 Entropy")
    print(f"{'=' * 60}")

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        D = X.shape[1]
        bq_rec = bq2_recall(X, n_queries=200, top_k=10)
        # Random HP
        np.random.seed(777)
        R = np.random.randn(D, D).astype(np.float32)
        R = R / np.linalg.norm(R, axis=0, keepdims=True)
        hc = (X @ R > 0).astype(np.uint8)
        rng = np.random.RandomState(42)
        q_idx = rng.choice(len(X), 200, replace=False)
        rh_recs = []
        for qi in q_idx:
            exact = X @ X[qi]; exact[qi] = -2.0
            gt = set(np.argpartition(-exact, 10)[:10])
            agree = np.sum(hc == hc[qi:qi+1], axis=1).astype(np.float32); agree[qi] = -1
            pred = set(np.argpartition(-agree, 10)[:10])
            rh_recs.append(len(gt & pred) / 10)
        rh_rec = np.mean(rh_recs)
        ent = compute_sign_entropy(X)
        dim_ent = -(np.mean(X>0, axis=0)*np.log2(np.clip(np.mean(X>0, axis=0), 1e-10, 1)) +
                    (1-np.mean(X>0, axis=0))*np.log2(np.clip(1-np.mean(X>0, axis=0), 1e-10, 1)))
        low_frac = np.mean(dim_ent < 0.9)
        print(f"  {name:<12}: BQ2={bq_rec:.4f}, RH={rh_rec:.4f}, entropy={ent:.4f}, <0.9={low_frac*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════
#  实验 I: 各向异性高斯模型 (原 probe 10)
# ═══════════════════════════════════════════════════════════════════

def exp_i_gaussian_model(datasets=None, n_sample=50_000):
    """各向异性高斯模型验证: SNR → sign entropy 预测。"""
    if datasets is None:
        datasets = ["cohere", "bge_m3", "minilm", "gist960", "random"]
    print(f"\n{'=' * 60}")
    print("实验 I: 各向异性高斯模型 (SNR → sign entropy)")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<12} {'D':>5} {'R²':>8} {'MAE':>7} {'|SNR|_m':>8} {'>1.0':>7} {'ent_act':>8} {'ent_pred':>8}")

    def binary_entropy(p):
        p = np.clip(p, 1e-10, 1-1e-10)
        return -(p*np.log2(p) + (1-p)*np.log2(1-p))

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        D = X.shape[1]
        snr = compute_snr(X)
        pred_ent = binary_entropy(norm.cdf(snr))
        act_rate = np.mean(X > 0, axis=0)
        act_ent = binary_entropy(act_rate)
        ss_res = np.sum((act_ent - pred_ent)**2)
        ss_tot = np.sum((act_ent - np.mean(act_ent))**2)
        r2 = 1 - ss_res/max(ss_tot, 1e-10)
        mae = np.mean(np.abs(act_ent - pred_ent))
        snr_abs = np.abs(snr)
        print(f"  {name:<12} {D:>5} {r2:>8.4f} {mae:>7.4f} {snr_abs.mean():>8.4f} "
              f"{(snr_abs>1).mean()*100:>6.1f}% {np.mean(act_ent):>8.4f} {np.mean(pred_ent):>8.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 J: 数学家前置包 (原 probe 12)
# ═══════════════════════════════════════════════════════════════════

def exp_j_math_package(datasets=None, n_sample=50_000):
    """Gaussianity / Independence / SNR fit / Pre-norm。"""
    if datasets is None:
        datasets = ["cohere", "bge_m3", "minilm", "gist960", "random"]
    print(f"\n{'=' * 60}")
    print("实验 J: 数学家前置实验包")
    print(f"{'=' * 60}")

    for name in datasets:
        try:
            X = load_dataset(name, n_sample)
        except Exception: continue
        D = X.shape[1]

        # A: Gaussianity
        skews = np.array([stats.skew(X[:, i]) for i in range(D)])
        kurts = np.array([stats.kurtosis(X[:, i]) for i in range(D)])
        rng = np.random.RandomState(42)
        n_test = min(200, D)
        test_dims = rng.choice(D, n_test, replace=False)
        ad_pass = sum(1 for d in test_dims
                      if stats.anderson(rng.choice(X[:,d], 2000, replace=False), 'norm').statistic
                      < stats.anderson(rng.choice(X[:,d], 2000, replace=False), 'norm').critical_values[2])

        # B: Independence
        X_sub = X[:min(len(X), 10000)]
        X_c = X_sub - X_sub.mean(axis=0, keepdims=True)
        stds = np.std(X_sub, axis=0, keepdims=True)
        X_n = X_c / np.maximum(stds, 1e-10)
        corr = (X_n.T @ X_n) / len(X_sub)
        np.fill_diagonal(corr, 0)
        upper = corr[np.triu_indices(D, k=1)]
        mean_corr = np.mean(np.abs(upper))

        # C: SNR fit
        snr = np.abs(compute_snr(X))
        snr_pos = snr[snr > 0.001]
        best_name, best_ks = "none", 999
        for dist_name, dist_fn in [("exponential", stats.expon), ("gamma", stats.gamma),
                                    ("lognorm", stats.lognorm)]:
            try:
                params = dist_fn.fit(snr_pos, floc=0)
                ks, _ = stats.kstest(snr_pos, dist_name, params)
                if ks < best_ks: best_ks, best_name = ks, dist_name
            except Exception: pass

        # D: Anisotropy
        var_per_dim = np.var(X, axis=0)
        sorted_var = np.sort(var_per_dim)
        n = len(sorted_var)
        gini = 1 - 2*np.sum(np.cumsum(sorted_var))/(n*np.sum(sorted_var)) + 1/n

        print(f"\n  {name} (D={D}):")
        print(f"    Gaussianity: |skew|={np.mean(np.abs(skews)):.4f}, |kurt|={np.mean(np.abs(kurts)):.4f}, "
              f"AD pass={ad_pass}/{n_test}")
        print(f"    Independence: mean|corr|={mean_corr:.6f}")
        print(f"    SNR best fit: {best_name} (KS={best_ks:.4f})")
        print(f"    Anisotropy: Gini={gini:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  01 - 分布特征分析 (Distribution Analysis)")
    print("  合并自: probe 1-7, 9, 10, 12")
    print("=" * 70)

    exp_a_pca()
    exp_b_angles()
    exp_c_sign_overlap()
    exp_d_vmf()
    exp_e_fista()
    exp_f_nmf()
    exp_g_dpp()
    exp_h_minilm_analysis()
    exp_i_gaussian_model()
    exp_j_math_package()

    print(f"\n{'=' * 70}")
    print("全部实验完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
