"""旋转悖论图：沿用四图样式，从冻结结果读取同协议的误差分解与三类决策翻转率。"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from build_figures import (
    GREY, INK, NAVY, ORANGE, RESULTS, SOURCE, TEAL, clean_axes, digest, export, panel_title,
)


def load_rotation_data():
    covariance_path = RESULTS / "probe26_covariance_certificate.json"
    block_path = RESULTS / "probe33_standard_vamana_p6_heldout_calibration.json"
    covariance = json.loads(covariance_path.read_text(encoding="utf-8"))
    blocks = json.loads(block_path.read_text(encoding="utf-8"))
    cohere = next(item for item in covariance["results"] if item["dataset"] == "cohere")["quantizers"]
    block = next(item for item in blocks["results"] if item["dataset"] == "cohere")["quantizers"]

    def structure(code):
        record = cohere[code]["residual_structure"]
        return {
            "distance_variance": (record["top_variance"] + record["competitor_variance"]) / 2,
            "comparison_variance": record["empirical_difference_variance"],
            "correlation": record["residual_correlation"],
            "flip_percent": 100 * cohere[code]["evaluation"]["decision_flip_rate"],
        }

    data = {
        "two_bit": structure("2bit"),
        "rotated_two_bit": structure("rotated_2bit"),
        "one_bit": structure("1bit"),
        "rotated_one_bit": structure("rotated_1bit"),
        "blocks": {
            code: {
                "distance_spearman": block[code]["distance_calibration"]["distance_spearman"],
                "ranking_flip_percent": 100 * block[code]["roles"]["ranking"]["validation"]["flip_rate"],
                "pruning_flip_percent": 100 * block[code]["roles"]["pruning"]["validation"]["flip_rate"],
            }
            for code in ["2bit", "rotated_2bit"]
        },
    }
    # 与正文数值逐项核对，防止读错字段。
    assert abs(data["two_bit"]["correlation"] - 0.8703) < 1e-4
    assert abs(data["rotated_two_bit"]["correlation"] - 0.4351) < 1e-4
    assert abs(data["two_bit"]["flip_percent"] - 5.8266) < 1e-3
    assert abs(data["rotated_two_bit"]["flip_percent"] - 8.8105) < 1e-3
    assert abs(data["blocks"]["2bit"]["distance_spearman"] - 0.5757) < 1e-4
    assert abs(data["blocks"]["rotated_2bit"]["distance_spearman"] - 0.9261) < 1e-4
    review = RESULTS.parent / "controls/calibration"
    matched_path = review / "matched-fidelity.json"
    cross_path = review / "calibration-cross.json"
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    crossed = json.loads(cross_path.read_text(encoding="utf-8"))
    data["matched_distance"] = {r["code"]: r["distance_spearman"] for r in matched["rows"] if r["population"] == "heldout160"}
    data["calibration_heldout"] = [r for r in crossed["rows"] if r["population"] == "4K-heldout160"]
    return data, [covariance_path, block_path, matched_path, cross_path]


def bar_pair(ax, values, title, ylabel, change_lines):
    colors = [NAVY, TEAL]
    positions = [0, 1]
    ax.bar(positions, values, width=0.56, color=colors, edgecolor="none")
    clean_axes(ax)
    ax.set_xticks(positions, ["Unrotated", "Rotated"])
    ax.set_xlim(-0.6, 1.6)
    top = max(values) * 1.55
    ax.set_ylim(0, top)
    # 整数刻度让纵轴标签宽度一致，避免挤出左边界。
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_ylabel(ylabel, labelpad=5)
    for x, value in zip(positions, values):
        ax.text(x, value + 0.025 * top, f"{value:.2f}", ha="center", va="bottom", fontsize=11.5)
    # 变化量放在柱顶上方的留白里，第二行给出解释该变化的辅助量。
    ax.text(0.5, 0.98, change_lines[0], transform=ax.transAxes, ha="center", va="top", color=ORANGE, fontsize=12)
    if len(change_lines) > 1:
        ax.text(0.5, 0.86, change_lines[1], transform=ax.transAxes, ha="center", va="top", color=GREY, fontsize=11.5)
    panel_title(ax, title)


def rotation_paradox(data):
    fig = plt.figure(figsize=(7.6, 3.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.45, 1.85], left=0.075, right=0.985, bottom=0.23, top=0.76, wspace=0.48)
    ax_distance = fig.add_subplot(grid[0, 0])
    ax_comparison = fig.add_subplot(grid[0, 1])
    ax_flips = fig.add_subplot(grid[0, 2])

    distance = [data["matched_distance"][code] for code in ("2bit", "rotated_2bit")]
    ax_distance.bar([0, 1], distance, width=0.56, color=[NAVY, TEAL], edgecolor="none")
    clean_axes(ax_distance)
    ax_distance.set_xticks([0, 1], ["Orig.", "Rot."])
    ax_distance.set_ylim(0, 1.13)
    ax_distance.set_yticks([0, 0.5, 1.0])
    ax_distance.set_ylabel(r"Distance Spearman $\rho$", labelpad=4)
    for x, value in enumerate(distance):
        ax_distance.text(x, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=11.5)
    panel_title(ax_distance, "(a) Distance fidelity")
    clean_axes(ax_comparison)
    all_variances = [r["metric_variance"] * 1e4 for r in data["calibration_heldout"]]
    for position, calibration in enumerate(("comparison", "distance")):
        before = next(r for r in data["calibration_heldout"] if r["code"] == "2bit" and r["calibration"] == calibration)["metric_variance"] * 1e4
        after = next(r for r in data["calibration_heldout"] if r["code"] == "rotated_2bit" and r["calibration"] == calibration)["metric_variance"] * 1e4
        xs = [position - 0.18, position + 0.18]
        ax_comparison.bar(xs, [before, after], width=0.31, color=[NAVY, TEAL], edgecolor="none")
        for x, value in zip(xs, [before, after]):
            ax_comparison.text(x, value + 0.10, f"{value:.2f}", ha="center", va="bottom", fontsize=11.5)
        delta = 100 * (after / before - 1)
        ax_comparison.text(position, max(before, after) + 0.73, f"{delta:+.0f}%", ha="center", va="bottom", fontsize=11.5, color=ORANGE)
    ax_comparison.set_ylim(0, max(all_variances) * 1.4)
    ax_comparison.set_xticks([0, 1], ["Comp.-fit", "Dist.-fit"])
    ax_comparison.set_ylabel(r"Error variance ($10^{-4}$)", labelpad=4)
    panel_title(ax_comparison, "(b) Comparison error")

    rows = [
        ("Nearest-vs-rest, 2-bit", data["two_bit"]["flip_percent"], data["rotated_two_bit"]["flip_percent"]),
        ("Nearest-vs-rest, 1-bit", data["one_bit"]["flip_percent"], data["rotated_one_bit"]["flip_percent"]),
        ("Ranking, blocks", data["blocks"]["2bit"]["ranking_flip_percent"], data["blocks"]["rotated_2bit"]["ranking_flip_percent"]),
        ("Pruning, blocks", data["blocks"]["2bit"]["pruning_flip_percent"], data["blocks"]["rotated_2bit"]["pruning_flip_percent"]),
    ]
    clean_axes(ax_flips)
    ax_flips.grid(axis="y", visible=False)
    ax_flips.grid(axis="x", color="#E5E9ED", linewidth=0.7)
    ys = np.arange(len(rows))[::-1]
    for y, (label, before, after) in zip(ys, rows):
        if abs(after - before) > 0.05 * before:
            color = ORANGE if after > before else TEAL
            ax_flips.annotate("", xy=(after, y), xytext=(before, y), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4, "shrinkA": 3.5, "shrinkB": 2.5, "mutation_scale": 9})
        else:
            # 相对变化不足5%时两点重叠，箭头无法辨认，改用文字标注。
            ax_flips.text(max(before, after) + 0.7, y, "unchanged", ha="left", va="center", fontsize=11.5, color=GREY)
        ax_flips.scatter([before], [y], s=30, color=NAVY, zorder=4)
        ax_flips.scatter([after], [y], s=30, color=TEAL, zorder=4)
        ax_flips.text(-0.6, y + 0.27, label, ha="left", va="bottom", fontsize=11.5, color=INK)
    ax_flips.set_yticks([])
    ax_flips.spines["left"].set_visible(False)
    ax_flips.set_xlim(-0.6, 17)
    ax_flips.set_ylim(-0.55, len(rows) - 0.05)
    ax_flips.set_xticks([0, 5, 10, 15])
    ax_flips.set_xlabel("Flip rate (%)", labelpad=5)
    panel_title(ax_flips, "(c) Held-out flips")
    ax_flips.scatter([], [], s=30, color=NAVY, label="Unrotated")
    ax_flips.scatter([], [], s=30, color=TEAL, label="Rotated")
    handles, labels = ax_flips.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, frameon=False, fontsize=11.5, handletextpad=0.3)
    return export(
        fig, "fig05-rotation-paradox",
        "Cohere。a、b及c前两行为原Probe26的同一批160条留出查询。b同时报告比较拟合与距离拟合。c后两行沿用原Probe33候选块协议。",
    )


def main():
    data, sources = load_rotation_data()
    exported = rotation_paradox(data)
    record = {
        "figure": exported,
        "plotted": data,
        "source_sha256": {path.name: digest(path) for path in sources},
        "builder": "build_rotation_figure.py",
    }
    (SOURCE / "rotation-figure.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("旋转悖论图及数据记录已生成", flush=True)


if __name__ == "__main__":
    main()
