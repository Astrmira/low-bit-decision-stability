#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一实验 Runner — 运行全部或指定模块的实验

用法:
  python run_all.py                    # 运行全部
  python run_all.py 01 03              # 只运行 01 和 03
  python run_all.py 01:A 03:D          # 只运行 01 的实验 A 和 03 的实验 D
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = {
    "01": ("01_distribution_analysis", "分布特征分析"),
    "02": ("02_fidelity_decomposition", "F/G 保真度分解"),
    "03": ("03_stein_fidelity", "Stein 保真度理论"),
    "04": ("04_rotation_duality", "旋转二象性"),
    "05": ("05_scaling_validation", "Scaling Law 验证"),
    "06": ("06_robustness", "鲁棒性边界诊断"),
    "07": ("07_decision_stability", "量化决策稳定性"),
    "08": ("08_decision_stability_refinement", "决策稳定性细化"),
    "09": ("09_covariance_certificate", "协方差感知证书验证"),
    "10": ("10_pruning_replay", "Vamana 剪枝重放"),
    "11": ("11_trace_certificate", "固定语义轨迹证书"),
    "12": ("12_edge_certificate", "依赖切片边证书"),
    "13": ("13_fallback_certificate", "Fallback 配额证书"),
    "14": ("14_path_certificate", "固定路径组合证书"),
    "15": ("15_p5_eligibility_audit", "P5 适用域审计"),
    "16": ("16_p6_heldout_calibration", "P6 留出校准验证"),
    "17": ("17_p6_strict_certificate", "P6 严格证书验证"),
    "18": ("18_necessity_counterexamples", "必要性反例"),
    "19": ("19_h1_robustness_audit", "H1 稳健性审计"),
    "20": ("20_cross_quantizer_validation", "跨量化器接口验证"),
    "21": ("21_p5_component_variance_audit", "P5 组件方差审计"),
    "22": ("22_p5_dimension_contamination_stress", "P5 维度污染压力测试"),
    "23": ("23_p8b_conditional_mgf", "P8-B 条件 MGF 与稳定截断审计"),
    "24": ("24_vamana_semantics_audit", "Vamana 剪枝语义检查"),
    "25": ("25_c1_calibration_probability_audit", "比较拟合概率检查"),
    "26": ("26_c2_stein_magnitude_audit", "Stein 幅值位恒等式检查"),
    "27": ("27_c3_contamination_trace_audit", "污染尾界与轨迹检查"),
    "28": ("28_d_selective_ratio_certificates", "D1--D3 选择性比率证书对照"),
    "29": ("29_candidate_selection_covariance", "E1--E3 候选选择协方差机制"),
}


def run_module(mod_id, exp_filter=None):
    """运行指定模块（可选按实验字母过滤）。"""
    if mod_id not in MODULES:
        print(f"未知模块: {mod_id}，可选: {list(MODULES.keys())}")
        return

    mod_name, desc = MODULES[mod_id]
    print(f"\n{'═' * 70}")
    print(f"  运行模块 {mod_id}: {desc}")
    print(f"{'═' * 70}")

    mod = __import__(mod_name)
    t0 = time.time()

    if exp_filter:
        # 运行特定实验
        func_name = f"exp_{exp_filter.lower()}_"
        candidates = [f for f in dir(mod) if f.startswith(func_name) or f.startswith(f"exp_{exp_filter.lower()}")]
        if not candidates:
            # 尝试精确匹配
            for fname in dir(mod):
                if fname.startswith("exp_") and exp_filter.lower() in fname:
                    candidates.append(fname)
        for func_name in candidates:
            func = getattr(mod, func_name, None)
            if callable(func):
                func()
                break
        else:
            print(f"  未找到实验 {exp_filter}，可用: {[f for f in dir(mod) if f.startswith('exp_')]}")
    else:
        mod.main()

    elapsed = time.time() - t0
    print(f"\n  模块 {mod_id} 完成，耗时 {elapsed:.1f}s")


def main():
    args = sys.argv[1:]

    if not args:
        # 运行全部
        print("=" * 70)
        print("  运行全部实验模块")
        print("=" * 70)
        t_total = time.time()
        for mod_id in sorted(MODULES.keys()):
            run_module(mod_id)
        print(f"\n{'═' * 70}")
        print(f"  全部完成！总耗时 {time.time() - t_total:.1f}s")
        print(f"{'═' * 70}")
    else:
        for arg in args:
            if ":" in arg:
                mod_id, exp = arg.split(":", 1)
                run_module(mod_id, exp)
            else:
                run_module(arg)


if __name__ == "__main__":
    main()
