#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 - 鲁棒性边界诊断 (Robustness Diagnostics)

合并自: probe23_mrl_threshold.py + MRL 方差断崖诊断

对应论文 §10 (Limitations):
  实验 A: MRL/Threshold 诊断 — mean vs median threshold 对 F 的影响
  实验 B: MRL 方差断崖检测 — 检查维度有序方差衰减模式

核心问题:
  1. MRL (Matryoshka) 训练是否导致 shared threshold α_x 被"绑架"？
  2. mean vs median threshold 在 Gaussian / non-Gaussian 数据上的表现差异
  3. OpenAI-3072 的 MRL 方差衰减特征（ρ(dim, σ) = -0.92）
"""

import sys
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_dataset, bq2_encode, RESULTS_DIR

# ═══════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════

N_SAMPLE = 20_000
N_PAIRS = 2_000_000
ALL_DATASETS = [
    "dbpedia3072", "dbpedia1536", "cohere", "minilm", "bge_m3",
    "arxiv_nomic", "gist960", "random",
]


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def encode_2bit_with_alpha(vecs, threshold_mode="mean"):
    """2-bit 编码，返回 signs, strongs, alphas（用于诊断）。

    与 utils.bq2_encode 类似，但额外返回每个向量的阈值 alpha。
    """
    signs = (vecs > 0).astype(np.int8)
    abs_vals = np.abs(vecs)
    if threshold_mode == "mean":
        alphas = abs_vals.mean(axis=1, keepdims=True)
    elif threshold_mode == "median":
        alphas = np.median(abs_vals, axis=1, keepdims=True)
    else:
        raise ValueError(f"未知 threshold mode: {threshold_mode}")
    strongs = (abs_vals > alphas).astype(np.int8)
    return signs, strongs, alphas.squeeze()


def compute_f_batched(vecs, threshold_mode="mean", n_pairs=N_PAIRS):
    """计算 F (Spearman ρ)，大规模 pair 分批计算避免内存爆炸。"""
    n = len(vecs)
    np.random.seed(42)
    idx_a = np.random.randint(0, n, n_pairs)
    idx_b = np.random.randint(0, n, n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    signs, strongs, _ = encode_2bit_with_alpha(vecs, threshold_mode)

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
#  实验 A: MRL/Threshold 诊断
# ═══════════════════════════════════════════════════════════════════

def exp_a_threshold_diagnostic(datasets=None):
    """对比 mean vs median threshold 的 F，诊断 threshold 绑架。"""
    if datasets is None:
        datasets = ALL_DATASETS

    print("=" * 90)
    print("实验 A: MRL/Threshold 诊断 — mean vs median threshold")
    print("=" * 90)

    results = []
    for ds_name in datasets:
        try:
            vecs = load_dataset(ds_name, n_sample=N_SAMPLE)
        except Exception as e:
            print(f"  {ds_name}: 加载失败 ({e})")
            continue

        D = vecs.shape[1]
        sigma = vecs.std(axis=0)
        cv = np.std(sigma) / np.mean(sigma)

        # --- Threshold 诊断 ---
        signs_m, strongs_m, alphas_mean = encode_2bit_with_alpha(vecs, "mean")
        signs_d, strongs_d, alphas_median = encode_2bit_with_alpha(vecs, "median")

        # 每个向量有多少维度的 magnitude bit = 1？
        mag_frac_mean = strongs_m.mean(axis=1).mean()
        mag_frac_median = strongs_d.mean(axis=1).mean()

        # 方差 top/bottom 10% 的 magnitude bit 命中率
        top_k = D // 10
        top_idx = np.argsort(-sigma)[:top_k]
        bot_idx = np.argsort(-sigma)[-top_k:]
        mag_hit_top = strongs_m[:, top_idx].mean()
        mag_hit_bot = strongs_m[:, bot_idx].mean()

        # 方差衰减比
        variance_ratio = sigma[top_idx].mean() / max(sigma[bot_idx].mean(), 1e-12)

        # --- F 对比 ---
        f_mean = compute_f_batched(vecs, "mean")
        f_median = compute_f_batched(vecs, "median")

        print(f"\n{'─' * 70}")
        print(f"数据集: {ds_name} (D={D})")
        print(f"  CV(σ) = {cv:.4f} | σ top/bot-10% 比 = {variance_ratio:.2f}x")
        print(f"  Mag=1 比例: mean={mag_frac_mean:.3f} | median={mag_frac_median:.3f}")
        print(f"  Top-10% mag=1 命中: {mag_hit_top:.3f} | Bot-10%: {mag_hit_bot:.3f}")
        print(f"  F(mean)={f_mean:.4f} | F(median)={f_median:.4f} | ΔF={f_median-f_mean:+.4f}")

        results.append({
            "dataset": ds_name, "D": D, "cv": cv,
            "variance_ratio": variance_ratio,
            "mag_frac_mean": mag_frac_mean,
            "mag_hit_top": mag_hit_top, "mag_hit_bot": mag_hit_bot,
            "f_mean": f_mean, "f_median": f_median,
            "delta_f": f_median - f_mean,
        })

    # 汇总表
    print(f"\n{'=' * 90}")
    print("汇总: Threshold 诊断")
    print(f"{'=' * 90}")
    print(f"{'Dataset':>18s} {'D':>5s} {'CV(σ)':>7s} {'σ ratio':>8s} "
          f"{'mag%':>6s} {'top hit':>8s} {'bot hit':>8s} "
          f"{'F_mean':>8s} {'F_median':>9s} {'ΔF':>7s}")
    for r in results:
        print(f"{r['dataset']:>18s} {r['D']:5d} {r['cv']:7.4f} {r['variance_ratio']:8.1f}x "
              f"{r['mag_frac_mean']:6.3f} {r['mag_hit_top']:8.3f} {r['mag_hit_bot']:8.3f} "
              f"{r['f_mean']:8.4f} {r['f_median']:9.4f} {r['delta_f']:+7.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  实验 B: MRL 方差断崖检测
# ═══════════════════════════════════════════════════════════════════

def exp_b_mrl_cliff(datasets=None):
    """检测各数据集是否有 MRL 特征的方差断崖（维度有序衰减）。"""
    if datasets is None:
        datasets = ALL_DATASETS

    print(f"\n{'=' * 70}")
    print("实验 B: MRL 方差断崖检测")
    print(f"{'=' * 70}")

    results = []
    for ds_name in datasets:
        try:
            vecs = load_dataset(ds_name, n_sample=10_000)
        except Exception as e:
            print(f"  {ds_name}: 加载失败 ({e})")
            continue

        D = vecs.shape[1]
        sigma = vecs.std(axis=0)

        # MRL 典型截断点的方差比
        cuts = [c for c in [64, 128, 256, 512] if c < D]
        for c in cuts:
            ratio = sigma[:c].mean() / max(sigma[c:].mean(), 1e-12)
            print(f"  {ds_name} dim[0:{c}]/[{c}:{D}] = {ratio:.2f}x")

        # 前 1/3 vs 后 1/3
        front = sigma[:D // 3].mean()
        back = sigma[2 * D // 3:].mean()
        mrl_ratio = front / max(back, 1e-12)

        # Spearman(维度序号, 平滑方差)
        window = max(D // 20, 5)
        smoothed = np.convolve(sigma, np.ones(window) / window, mode='valid')
        rho, _ = spearmanr(np.arange(len(smoothed)), smoothed)

        has_mrl = mrl_ratio > 1.5 and rho < -0.3
        tag = "🔥 MRL 模式" if has_mrl else "❌ 无 MRL"
        print(f"  → 前/后1/3比={mrl_ratio:.2f}x | ρ(dim,σ)={rho:+.4f} | {tag}\n")

        results.append({
            "dataset": ds_name, "D": D,
            "mrl_ratio": mrl_ratio, "rho_dim_sigma": rho,
            "has_mrl": has_mrl,
        })

    # 汇总
    print(f"{'Dataset':>18s} {'前/后1/3':>10s} {'ρ(dim,σ)':>10s} {'MRL?':>8s}")
    for r in results:
        tag = "🔥 YES" if r["has_mrl"] else "❌ NO"
        print(f"{r['dataset']:>18s} {r['mrl_ratio']:10.2f}x "
              f"{r['rho_dim_sigma']:+10.4f} {tag:>8s}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  06 - 鲁棒性边界诊断 (Robustness Diagnostics)")
    print("  合并自: probe23 + MRL 断崖诊断")
    print("=" * 70)

    results_a = exp_a_threshold_diagnostic()
    results_b = exp_b_mrl_cliff()

    print(f"\n{'=' * 70}")
    print("全部实验完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
