#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 - Scaling Law 验证 + 盲测 + S0 边界 + 图导航 (Scaling & Validation)

合并自: probe11, probe19, probe19b, probe20, probe20b, probe21, probe22

对应论文:
  §7  Scaling Law (probe 11: SNR 泛函, probe 22: S0 边界)
  §9  实验 (probe 20/20b: 因果干预, probe 21: 盲测验证)
  附录 (probe 19/19b: 图导航单步, 已从论文删除但保留实验代码)

本文件包含:
  实验 A: SNR 分布 → Recall 泛函 (p-sweep, 等价性, 真实克隆, 幅度扫描)
  实验 B: 图导航 Local F vs Global F (1-bit vs 2-bit 单步精度)
  实验 C: 候选集召回率 (BQ 作为候选生成器)
  实验 D: 因果干预 (axis scaling → F 的因果关系)
  实验 E: Scaling Law 预测因子追踪 (log(r_off), std(SNR))
  实验 F: 盲测外推验证 (新数据集 F/Recall 预测)
  实验 G: S0 均值场闭包失效边界 (block-Σ 合成实验)
"""

import sys
import time
import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import sqrtm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (load_dataset, compute_sign_entropy, compute_snr,
                   bq2_encode, bq2_recall, bq2_fidelity,
                   RESULTS_DIR, PHASE5_768, DATASETS)


# ═══════════════════════════════════════════════════════════════════
#  共享工具（本文件内部复用）
# ═══════════════════════════════════════════════════════════════════

def generate_aniso_gaussian_sphere(N, D, mu, sigma, seed=42):
    """生成 N(mu, diag(sigma²)) 的球面投影数据。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D).astype(np.float32) * sigma[np.newaxis, :] + mu[np.newaxis, :]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def snr_to_mu_sigma(snr, base_sigma=1.0):
    """从 SNR 和基础 sigma 生成 mu, sigma。"""
    sigma = np.full_like(snr, base_sigma)
    mu = snr * sigma
    return mu, sigma


def make_snr_vector(D, p_high, snr_high=2.0, snr_low=0.1, seed=0):
    """构造混合 SNR 向量：p_high 比例的维度 SNR=snr_high，其余 SNR=snr_low。"""
    rng = np.random.RandomState(seed)
    n_high = int(D * p_high)
    snr = np.full(D, snr_low, dtype=np.float32)
    high_idx = rng.choice(D, n_high, replace=False)
    snr[high_idx] = snr_high
    return snr


def bq2_recall_fast(X, n_queries=200, top_k=10):
    """快速 BQ2 模拟 recall（本地实现，用于合成数据的快速测量）。"""
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
    return np.mean(recalls)


def encode_2bit_mean(vecs):
    """2-bit 编码，per-vector mean 阈值（匹配论文公式）。"""
    signs = (vecs > 0).astype(np.int8)
    abs_vals = np.abs(vecs)
    alphas = abs_vals.mean(axis=1, keepdims=True)
    strongs = (abs_vals > alphas).astype(np.int8)
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


# ═══════════════════════════════════════════════════════════════════
#  实验 A: SNR 分布 → Recall 泛函 (原 probe 11)
# ═══════════════════════════════════════════════════════════════════

def exp_a_snr_functional():
    """证明 BQ Recall 是 SNR 分布的泛函。"""
    print(f"\n{'=' * 60}")
    print("实验 A: SNR 分布 → Recall 泛函")
    print(f"{'=' * 60}")

    D = 768
    N = 10_000

    # A1: p-sweep
    print("\n  A1: 高 SNR 维度比例 p 扫描")
    p_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                0.40, 0.50, 0.60, 0.75, 1.00]
    for p in p_values:
        snr = make_snr_vector(D, p, snr_high=2.0, snr_low=0.1)
        mu, sigma = snr_to_mu_sigma(snr)
        X = generate_aniso_gaussian_sphere(N, D, mu, sigma)
        rec = bq2_recall_fast(X)
        ent = compute_sign_entropy(X)
        print(f"    p={p:.2f}: recall={rec:.4f}, sign_entropy={ent:.4f}")

    # A2: 等价性验证
    print("\n  A2: 相同 SNR 分布 → 相同 recall（等价性）")
    recalls = []
    for trial in range(5):
        snr = make_snr_vector(D, 0.25, 2.0, 0.1, seed=trial * 100)
        base_sigma = 0.5 + trial * 0.5
        mu, sigma = snr_to_mu_sigma(snr, base_sigma=base_sigma)
        X = generate_aniso_gaussian_sphere(N, D, mu, sigma, seed=trial)
        rec = bq2_recall_fast(X)
        recalls.append(rec)
        print(f"    trial={trial}, base_σ={base_sigma:.1f}: recall={rec:.4f}")
    print(f"    recall std={np.std(recalls):.4f} → {'✅ 等价性成立' if np.std(recalls) < 0.01 else '❌ 不成立'}")

    # A3: 真实 SNR 克隆
    print("\n  A3: 用真实 SNR 分布合成数据 → 验证 recall 匹配")
    for name in ["cohere", "bge_m3", "minilm", "random"]:
        try:
            X_real = load_dataset(name, N)
            snr_real = compute_snr(X_real)
            mu_s, sigma_s = snr_to_mu_sigma(snr_real, base_sigma=1.0)
            X_synth = generate_aniso_gaussian_sphere(N, X_real.shape[1], mu_s, sigma_s)
            rec_real = bq2_recall_fast(X_real)
            rec_synth = bq2_recall_fast(X_synth)
            print(f"    {name:<12}: real={rec_real:.4f}, synth={rec_synth:.4f}, Δ={rec_synth-rec_real:+.4f}")
        except Exception as e:
            print(f"    {name}: 跳过 ({e})")


# ═══════════════════════════════════════════════════════════════════
#  实验 B: Local F vs Global F (原 probe 19)
# ═══════════════════════════════════════════════════════════════════

def _score_1bit(q, db):
    """1-bit sign 匹配数。"""
    return ((q > 0) == (db > 0)).sum(axis=1).astype(np.float32)

def _score_2bit(q, db_vecs, db_signs, db_mags):
    """2-bit 带符号加权 Hamming。"""
    q_sign = (q > 0).astype(np.float32)
    alpha_q = np.mean(np.abs(q))
    q_mag = (np.abs(q) > alpha_q).astype(np.float32)
    sign_agree = 2.0 * (q_sign == db_signs).astype(np.float32) - 1.0
    weight = (1 + q_mag) * (1 + db_mags)
    return (sign_agree * weight).sum(axis=1)

def exp_b_local_f():
    """图导航单步：Local F vs Global F (1-bit vs 2-bit)。"""
    print(f"\n{'=' * 60}")
    print("实验 B: Local F vs Global F (图导航单步)")
    print(f"{'=' * 60}")

    datasets = ["cohere", "random", "minilm", "arxiv_nomic", "codesearch_jina"]
    N_SAMPLE = 10_000
    N_QUERIES = 500
    K_LOCAL = 32

    for name in datasets:
        try:
            vecs = load_dataset(name, n_sample=N_SAMPLE)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")
            continue

        N = len(vecs)
        db_signs = (vecs > 0).astype(np.float32)
        db_mags = (np.abs(vecs) > np.mean(np.abs(vecs), axis=1, keepdims=True)).astype(np.float32)

        q_indices = np.random.RandomState(42).choice(N, min(N_QUERIES, N), replace=False)
        local_1, local_2, acc_1, acc_2 = [], [], [], []

        for qi in q_indices:
            cos_all = vecs @ vecs[qi]
            cos_all[qi] = -999
            nn_idx = np.argsort(cos_all)[-K_LOCAL:][::-1]
            cos_nn = cos_all[nn_idx]

            s1 = _score_1bit(vecs[qi], vecs[nn_idx])
            s2 = _score_2bit(vecs[qi], vecs[nn_idx], db_signs[nn_idx], db_mags[nn_idx])

            if np.std(s1) > 0 and np.std(cos_nn) > 0:
                rho1, _ = spearmanr(cos_nn, s1)
                local_1.append(rho1)
            if np.std(s2) > 0 and np.std(cos_nn) > 0:
                rho2, _ = spearmanr(cos_nn, s2)
                local_2.append(rho2)

            acc_1.append(1.0 if nn_idx[np.argmax(s1)] == nn_idx[0] else 0.0)
            acc_2.append(1.0 if nn_idx[np.argmax(s2)] == nn_idx[0] else 0.0)

        F1_l = np.mean(local_1) if local_1 else float('nan')
        F2_l = np.mean(local_2) if local_2 else float('nan')
        print(f"  {name:<18}: F1_local={F1_l:.4f}, F2_local={F2_l:.4f}, "
              f"ΔF={F2_l-F1_l:+.4f}, acc1={np.mean(acc_1):.3f}, acc2={np.mean(acc_2):.3f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 C: 候选集召回率 (原 probe 19b)
# ═══════════════════════════════════════════════════════════════════

def exp_c_candidate_recall():
    """BQ 作为候选生成器的保真度。"""
    print(f"\n{'=' * 60}")
    print("实验 C: 候选集召回率 (Candidate Recall)")
    print(f"{'=' * 60}")

    datasets = ["cohere", "random", "minilm", "arxiv_nomic"]
    N_SAMPLE = 10_000
    N_QUERIES = 500
    K = 32
    C_VALUES = [1, 4, 8, 16]

    for name in datasets:
        try:
            vecs = load_dataset(name, n_sample=N_SAMPLE)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")
            continue

        N = len(vecs)
        q_indices = np.random.RandomState(42).choice(N, min(N_QUERIES, N), replace=False)
        recalls_2bit = {c: [] for c in C_VALUES}

        for qi in q_indices:
            cos_all = vecs @ vecs[qi]
            cos_all[qi] = -999
            nn_idx = np.argsort(cos_all)[-K:][::-1]

            s2 = _score_2bit(vecs[qi], vecs[nn_idx],
                             (vecs[nn_idx] > 0).astype(np.float32),
                             (np.abs(vecs[nn_idx]) > np.mean(np.abs(vecs[nn_idx]), axis=1, keepdims=True)).astype(np.float32))
            rank_2bit = np.argsort(s2)[::-1]

            for c in C_VALUES:
                true_topC = set(range(c))
                bq_topC = set(rank_2bit[:c].tolist())
                recalls_2bit[c].append(len(true_topC & bq_topC) / c)

        print(f"  {name:<18}: " + " | ".join(
            f"CR@{c}={np.mean(recalls_2bit[c]):.3f}" for c in C_VALUES))


# ═══════════════════════════════════════════════════════════════════
#  实验 D: 因果干预 (原 probe 20)
# ═══════════════════════════════════════════════════════════════════

def exp_d_intervention():
    """通过 axis scaling 干预验证 CV(σ) 与 F 的因果关系。"""
    print(f"\n{'=' * 60}")
    print("实验 D: 因果干预 (Axis Scaling → F)")
    print(f"{'=' * 60}")

    N_SAMPLE = 20_000
    datasets = ["cohere", "arxiv_nomic", "bge_m3"]

    for ds_name in datasets:
        try:
            vecs = load_dataset(ds_name, n_sample=N_SAMPLE)
        except Exception as e:
            print(f"  {ds_name}: 跳过 ({e})")
            continue

        D = vecs.shape[1]
        sigma = vecs.std(axis=0)

        # 基线
        f_base = compute_f_batched(vecs)
        cv_base = np.std(sigma) / np.mean(sigma)

        # Flatten: σ_i → mean(σ)
        scale = np.mean(sigma) / np.maximum(sigma, 1e-12)
        vecs_flat = vecs * scale[np.newaxis, :]
        norms = np.linalg.norm(vecs_flat, axis=1, keepdims=True)
        vecs_flat = vecs_flat / np.maximum(norms, 1e-12)
        f_flat = compute_f_batched(vecs_flat)

        # Amplify: σ_i → σ_i^1.5
        sigma_amp = sigma ** 1.5
        scale_amp = sigma_amp / np.maximum(sigma, 1e-12)
        vecs_amp = vecs * scale_amp[np.newaxis, :]
        norms = np.linalg.norm(vecs_amp, axis=1, keepdims=True)
        vecs_amp = vecs_amp / np.maximum(norms, 1e-12)
        f_amp = compute_f_batched(vecs_amp)

        print(f"  {ds_name:<18}: F_base={f_base:.4f} (CV={cv_base:.3f}) | "
              f"F_flat={f_flat:.4f} | F_amp={f_amp:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 E: Scaling Law 预测因子追踪 (原 probe 20b)
# ═══════════════════════════════════════════════════════════════════

def exp_e_predictor_tracking():
    """追踪 scaling law 预测因子在干预条件下的变化。"""
    print(f"\n{'=' * 60}")
    print("实验 E: Scaling Law 预测因子追踪")
    print(f"{'=' * 60}")

    N_SAMPLE = 20_000

    for ds_name in ["cohere", "arxiv_nomic"]:
        try:
            vecs = load_dataset(ds_name, n_sample=N_SAMPLE)
        except Exception as e:
            print(f"  {ds_name}: 跳过 ({e})")
            continue

        D = vecs.shape[1]
        sigma = vecs.std(axis=0)
        Sigma = np.cov(vecs.T)

        # 计算关键预测因子
        diag_energy = np.sum(sigma ** 2)
        off_diag_energy = np.sum(Sigma ** 2) - np.sum(np.diag(Sigma) ** 2)
        r_off = np.sqrt(max(off_diag_energy, 0)) / np.sqrt(diag_energy)
        snr = compute_snr(vecs)
        std_snr = np.std(np.abs(snr))
        f_val = compute_f_batched(vecs)

        print(f"  {ds_name:<18}: F={f_val:.4f} | log(r_off)={np.log(r_off):.4f} | "
              f"std(|SNR|)={std_snr:.4f} | CV(σ)={np.std(sigma)/np.mean(sigma):.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 F: 盲测外推验证 (原 probe 21)
# ═══════════════════════════════════════════════════════════════════

def exp_f_blind_validation():
    """在新数据集上验证 F/Recall 的 scaling law 预测。"""
    print(f"\n{'=' * 60}")
    print("实验 F: 盲测外推验证")
    print(f"{'=' * 60}")

    # 训练集（用于拟合 scaling law）和测试集
    test_datasets = ["coco_nomic", "landmark_dino", "codesearch_jina"]
    N_SAMPLE = 20_000

    for ds_name in test_datasets:
        try:
            vecs = load_dataset(ds_name, n_sample=N_SAMPLE)
        except Exception as e:
            print(f"  {ds_name}: 跳过 ({e})")
            continue

        D = vecs.shape[1]
        f_val = compute_f_batched(vecs)
        recall = bq2_recall(vecs, n_queries=500, top_k=10)
        sign_ent = compute_sign_entropy(vecs)

        print(f"  {ds_name:<18}: D={D}, F={f_val:.4f}, Recall@10={recall:.4f}, "
              f"sign_entropy={sign_ent:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  实验 G: S0 均值场闭包失效边界 (原 probe 22)
# ═══════════════════════════════════════════════════════════════════

def s0_diagnostic(sigma, Sigma):
    """S0 诊断量: ρ_rank(σ_i, avg|ρ_ij|)。"""
    D = len(sigma)
    d = np.sqrt(np.diag(Sigma))
    R = Sigma / np.outer(d + 1e-12, d + 1e-12)
    avg_abs_corr = (np.sum(np.abs(R), axis=1) - 1) / (D - 1)
    rho, _ = spearmanr(sigma, avg_abs_corr)
    return rho


def generate_block_sigma(D, base_sigma, block_size, block_strength, seed=42):
    """生成具有 SNR-对齐块结构的协方差矩阵。"""
    rng = np.random.RandomState(seed)
    order = np.argsort(-base_sigma)
    sorted_sigma = base_sigma[order]

    R = np.eye(D)
    bg_corr = rng.randn(D, D) * 0.05
    bg_corr = (bg_corr + bg_corr.T) / 2
    np.fill_diagonal(bg_corr, 0)
    R += bg_corr

    for i in range(block_size):
        for j in range(block_size):
            if i != j:
                R[i, j] = block_strength

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 0.01)
    R = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)

    Sigma = np.outer(sorted_sigma, sorted_sigma) * R
    return Sigma, sorted_sigma


def sample_from_sigma(Sigma, n_sample, seed=42):
    """从 N(0, Σ) 生成样本，归一化到单位球。"""
    rng = np.random.RandomState(seed)
    D = Sigma.shape[0]
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(Sigma_pd)

    Z = rng.randn(n_sample, D)
    g = Z @ L.T
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(norms, 1e-12)


def exp_g_s0_boundary():
    """S0 均值场闭包失效边界。"""
    print(f"\n{'=' * 60}")
    print("实验 G: S0 均值场闭包失效边界")
    print(f"{'=' * 60}")

    N_SAMPLE = 20_000
    vecs = load_dataset("cohere", n_sample=N_SAMPLE)
    D = vecs.shape[1]
    base_sigma = vecs.std(axis=0)

    f_baseline = compute_f_batched(vecs)
    Sigma_real = np.cov(vecs.T)
    rho_real = s0_diagnostic(base_sigma, Sigma_real)
    print(f"  Cohere 基线: F={f_baseline:.4f}, S0 诊断={rho_real:+.4f}")

    BLOCK_SIZES = [32, 64, 128]
    STRENGTHS = [0.0, 0.10, 0.30, 0.50, 0.70, 0.90]

    print(f"\n  {'block':>6} {'strength':>9} {'S0_diag':>9} {'F_actual':>9}")
    for bs in BLOCK_SIZES:
        for st in STRENGTHS:
            Sigma_syn, sigma_syn = generate_block_sigma(D, base_sigma, bs, st)
            rho_s0 = s0_diagnostic(sigma_syn, Sigma_syn)
            vecs_syn = sample_from_sigma(Sigma_syn, N_SAMPLE)
            f_actual = compute_f_batched(vecs_syn)
            print(f"  {bs:6d} {st:9.2f} {rho_s0:+9.4f} {f_actual:9.4f}")


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  05 - Scaling Law 验证 + 盲测 + S0 边界")
    print("  合并自: probe 11, 19, 19b, 20, 20b, 21, 22")
    print("=" * 70)

    exp_a_snr_functional()
    exp_b_local_f()
    exp_c_candidate_recall()
    exp_d_intervention()
    exp_e_predictor_tracking()
    exp_f_blind_validation()
    exp_g_s0_boundary()

    print(f"\n{'=' * 70}")
    print("全部实验完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
