"""按参考构图精绘四张原生SVG，并从SVG导出600 dpi PNG。"""

import csv
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
import resvg_py
from scipy.stats import spearmanr

SOURCE = Path(__file__).resolve().parent
FIGURES = SOURCE.parent
RESEARCH = FIGURES.parents[1] / "experiments"
RESULTS = RESEARCH / "results"
TEAL = "#147D92"
NAVY = "#253858"
ORANGE = "#D26943"
INK = "#25303E"
GREY = "#6A7582"
GRID = "#E5E9ED"
PALE_TEAL = "#E2F0F2"
PALE_ORANGE = "#FBECE5"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "axes.linewidth": 0.65,
    "svg.fonttype": "none",
    "svg.hashsalt": "jmlr-rewrite-four-figures-20261109",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel_title(ax, text):
    ax.set_title(text, loc="left", color=NAVY, fontfamily="DejaVu Sans", fontweight="bold", fontsize=11.5, pad=10)


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.65, pad=4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)


def export(fig, name, description):
    """先保存含可编辑文字的SVG，再由同一SVG渲染PNG。"""
    svg = FIGURES / f"{name}.svg"
    png = FIGURES / f"{name}.png"
    fig.savefig(svg, format="svg", metadata={"Date": None, "Description": description})
    root = ET.parse(svg).getroot()
    assert not root.findall(".//{http://www.w3.org/2000/svg}image"), "最终SVG须为原生矢量内容"
    rendered = resvg_py.svg_to_bytes(
        svg_path=str(svg), width=round(fig.get_figwidth() * 600), dpi=96.0,
        font_dirs=[str(FONT_DIR)], background="white",
        serif_family="STIXGeneral", text_rendering="geometric_precision",
    )
    with Image.open(io.BytesIO(rendered)) as raster:
        raster.convert("RGB").save(png, dpi=(600, 600), optimize=True)
    plt.close(fig)
    print(f"已导出 {name} 的SVG与600 dpi PNG", flush=True)
    return {"svg": svg.name, "png": png.name, "svg_sha256": digest(svg), "png_sha256": digest(png)}


def comparison_geometry():
    """固定相同误差与包络半径，仅改变精确裕度。"""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.05))
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.19, top=0.86, wspace=0.13)
    examples = [(0.6, "(a) Close call", "sign changes"), (2.6, "(b) Large margin", "sign preserved")]
    for ax, (margin, title, outcome) in zip(axes, examples):
        ax.set_xlim(-1.25, 4.0)
        ax.set_ylim(-0.85, 2.45)
        ax.axis("off")
        panel_title(ax, title)
        radius, error, baseline = 1.0, -0.9, 0.45
        lower, upper = margin - radius, margin + radius
        ax.add_patch(Rectangle((lower, baseline - 0.18), 2 * radius, 0.36, facecolor=PALE_TEAL, edgecolor="none"))
        if lower < 0:
            ax.add_patch(Rectangle((lower, baseline - 0.18), -lower, 0.36, facecolor=PALE_ORANGE, edgecolor="none"))
        ax.plot([0, 0], [-0.12, 2.1], color=GREY, lw=0.9, ls=(0, (3, 3)))
        ax.annotate("", xy=(3.9, baseline), xytext=(-1.17, baseline), arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8})
        ax.text(0, -0.1, "0", ha="center", va="top", fontsize=11.5)
        ax.plot([margin, margin], [baseline, 1.83], color=TEAL, lw=0.65, alpha=0.75)
        ax.annotate("", xy=(margin, 1.83), xytext=(0, 1.83), arrowprops={"arrowstyle": "<->", "color": TEAL, "lw": 1.05, "shrinkA": 0, "shrinkB": 0})
        ax.text(margin / 2, 2.03, r"calibrated margin $m$", ha="center", fontsize=11.5, color=TEAL)
        ax.annotate("", xy=(margin + error, 1.14), xytext=(margin, 1.14), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.4, "shrinkA": 0, "shrinkB": 0})
        ax.text(margin + error / 2, 1.30, r"$R$", color=ORANGE, ha="center", fontsize=12)
        ax.scatter([margin], [baseline], s=45, color=TEAL, edgecolor="white", linewidth=0.8, zorder=5)
        ax.scatter([margin + error], [baseline], s=45, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(margin, -0.01, r"$m$", ha="center", va="top", color=TEAL, fontsize=12)
        ax.text(margin + error - 0.08, -0.01, r"$m+R$", ha="right", va="top", color=ORANGE, fontsize=12)
        ax.text(1.35, -0.67, outcome, color=ORANGE if margin < radius else TEAL, ha="center", fontsize=11.5, fontstyle="italic")
    fig.text(0.5, 0.065, r"$m=c_Q\Gamma>0,\quad \widehat\Psi_Q=m+R$     ·     same error envelope $|R|\leq r$", ha="center", fontsize=11.5)
    return export(fig, "fig01-comparison-geometry", "示意图。两侧m分别为0.6与2.6，r=1且R=-0.9；包络只是图示范围。")


def state_label(ax, x, y, selected, color):
    if not selected:
        text = r"$\varnothing$"
    else:
        tokens = [rf"\mathbf{{c}}_{{{node}}}" if node == 3 else rf"c_{{{node}}}" for node in selected]
        text = r"$\{" + ",".join(tokens) + r"\}$"
    ax.text(x, y, text, ha="center", va="center", fontsize=11.5, color=color)


def first_divergence():
    """两条实际执行共享前三步前状态，首次差异后保留额外候选。"""
    fig = plt.figure(figsize=(7.6, 3.3))
    ax = fig.add_axes([0.015, 0.10, 0.97, 0.84])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = np.array([0.235, 0.397, 0.559, 0.721, 0.883])
    ax.add_patch(Rectangle((centers[2] - 0.079, 0.21), 0.158, 0.70, color=PALE_ORANGE, zorder=0))
    ax.text(centers[2], 0.97, r"first mismatch $j_\star$", color=ORANGE, ha="center", fontsize=11.5, fontweight="bold")
    ax.text(0.02, 0.86, "Candidate", color=GREY, fontsize=11.5)
    for step, x in enumerate(centers, 1):
        ax.text(x, 0.86, rf"$c_{{{step}}}$", ha="center", fontsize=13, color=INK)
    exact = [[1], [1], [1], [1, 4], [1, 4, 5]]
    quantized = [[1], [1], [1, 3], [1, 3, 4], [1, 3, 4, 5]]
    for name, y, color, states, actions in [
        ("Exact", 0.64, NAVY, exact, ["keep", "prune", "prune", "keep", "keep"]),
        ("Quantized", 0.32, TEAL, quantized, ["keep", "prune", "keep", "keep", "keep"]),
    ]:
        ax.text(0.02, y + 0.105, name, color=color, fontsize=11.5, fontfamily="DejaVu Sans", fontweight="bold")
        state_label(ax, 0.08, y, [], color)
        previous = 0.11
        for index, (x, state, action) in enumerate(zip(centers, states, actions)):
            ax.annotate("", xy=(x - 0.071, y), xytext=(previous, y), arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8, "shrinkA": 0, "shrinkB": 0}, zorder=1)
            edge = ORANGE if index == 2 else color
            ax.add_patch(FancyBboxPatch((x - 0.069, y - 0.055), 0.138, 0.11, boxstyle="round,pad=0.006,rounding_size=0.016", facecolor="white", edgecolor=edge, linewidth=1.15 if index == 2 else 0.8, zorder=2))
            state_label(ax, x, y, state, color)
            ax.text(x, y + 0.091, action, ha="center", color=ORANGE if index == 2 else color, fontsize=11.5, fontstyle="italic")
            previous = x + 0.076
    ax.plot([0.11, 0.11, 0.47, 0.47], [0.20, 0.17, 0.17, 0.20], color=GREY, lw=0.8)
    ax.text(0.29, 0.105, "shared prefix", ha="center", fontsize=11.5, color=GREY)
    ax.text(centers[2], 0.105, r"$S_{j_\star}=\{c_1\}$", ha="center", color=ORANGE, fontsize=11.5)
    fig.text(0.5, 0.045, r"$\widehat S_{\rm out}\ne S_{\rm out}\quad\Longleftrightarrow\quad \exists j:\ E_j$", ha="center", fontsize=12)
    return export(fig, "fig02-first-divergence", "确定性示例。固定c1至c5顺序、Delta=4、无回填；c3首次动作不同。后续状态均为各自实际执行。")


def load_margin_data():
    path = RESULTS / "probe24_decision_stability.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    groups = [[0], [1, 2, 3], [4, 5], [6], [7], [8]]
    output = []
    for dataset in source:
        bins = dataset["quantizers"]["2bit"]["binned_flip_rates"]
        rows = []
        for group in groups:
            count = sum(bins[index]["count"] for index in group)
            flips = sum(round(bins[index]["count"] * bins[index]["flip_rate"]) for index in group)
            assert all(abs(bins[index]["count"] * bins[index]["flip_rate"] - round(bins[index]["count"] * bins[index]["flip_rate"])) < 1e-8 for index in group)
            rows.append({"count": count, "flips": flips, "rate_percent": 100 * flips / count})
        assert sum(row["count"] for row in rows) == 4960
        output.append({"dataset": dataset["dataset"], "bins": rows})
    assert [row["count"] for row in output[0]["bins"]] == [70, 381, 1210, 1577, 1413, 309]
    return output, path


def margin_chart(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35), gridspec_kw={"width_ratios": [1.9, 1]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.235, wspace=0.31)
    styles = [("Cohere", TEAL, "o"), ("MiniLM", NAVY, "s"), ("GIST", ORANGE, "^")]
    labels = [r"$[0,.25)$", r"$[.25,1)$", r"$[1,2)$", r"$[2,3)$", r"$[3,5)$", r"$[5,\infty)$"]
    for index, (record, (name, color, marker)) in enumerate(zip(data, styles)):
        rates = [row["rate_percent"] for row in record["bins"]]
        axes[0].plot(range(6), rates, color=color, marker=marker, linewidth=1.5, markersize=4.4, markeredgewidth=0.7, label=name, clip_on=False)
        if index < 2:
            # 仅在小图分开零点的标记，横坐标仍位于对应分类区间。
            axes[1].plot(np.arange(3) + (-0.035 if index == 0 else 0.035), rates[3:], color=color, marker=marker, linewidth=1.5, markersize=4.5, markeredgewidth=0.7, clip_on=False)
    for ax in axes:
        clean_axes(ax)
        ax.set_xlabel("Margin-to-noise ratio (MNR)", labelpad=8)
        ax.set_ylabel("Flip rate (%)", labelpad=6)
    axes[0].set_ylim(0, 70)
    axes[0].set_xlim(-0.18, 5.18)
    axes[0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    axes[0].set_xticks(range(6), labels)
    axes[0].tick_params(axis="x", labelsize=11.5)
    axes[1].set_xlim(-0.18, 2.18)
    axes[1].set_ylim(0, 3)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_xticks(range(3), labels[3:])
    axes[1].set_xlabel("MNR", labelpad=8)
    panel_title(axes[0], "(a) All three datasets")
    panel_title(axes[1], "(b) Large margins")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.0, columnspacing=2.5, fontsize=11.5)
    return export(fig, "fig03-margin-flips", "原probe24结果。按比较计数合并区间，保留零翻转率；右图仅放大Cohere和MiniLM。")


def load_predictors():
    path = RESULTS / "probe33_standard_vamana_p6_heldout_calibration.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for role in ["ranking", "pruning"]:
        for dataset in source["results"]:
            for quantizer, record in dataset["quantizers"].items():
                validation = record["roles"][role]["validation"]
                rows.append({
                    "role": role, "dataset": dataset["dataset"], "quantizer": quantizer,
                    "bound": record["roles"][role]["risk_model"]["plugin_risk"],
                    "distance_risk": 1 - record["distance_calibration"]["distance_spearman"],
                    "flip_percent": 100 * validation["flip_rate"],
                    "eligible": record["eligibility"]["eligible"],
                })
        selected = [row for row in rows if row["role"] == role]
        assert len(selected) == 24
        for metric, key in [("bound", "risk_flip_spearman"), ("distance_risk", "distance_spearman_flip_spearman")]:
            rho = float(spearmanr([row[metric] for row in selected], [row["flip_percent"] for row in selected]).statistic)
            assert abs(rho - source["aggregate"][role][key]) < 1e-12
    return rows, path


def predictor_chart(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.55))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.29, top=0.88, wspace=0.25)
    predictors = [("bound", "Close-call risk score", TEAL), ("distance_risk", "Distance fidelity", ORANGE)]
    for index, (ax, role) in enumerate(zip(axes, ["ranking", "pruning"])):
        selected = [row for row in rows if row["role"] == role]
        maximum = max(row["flip_percent"] for row in selected)
        top = np.ceil(maximum / 5) * 5
        clean_axes(ax)
        for predictor_index, (metric, title, color) in enumerate(predictors):
            for record in selected:
                rotated = record["quantizer"] == "rotated_2bit"
                ax.scatter(
                    record[metric], record["flip_percent"],
                    marker="D" if rotated else "o", s=27 if rotated else 32,
                    facecolor=color if rotated else "white", edgecolor=color,
                    linewidth=0.95, alpha=0.90, zorder=3, clip_on=False,
                )
            rho = spearmanr([record[metric] for record in selected], [record["flip_percent"] for record in selected]).statistic
            ax.text(
                0.045, 0.94 - 0.15 * predictor_index,
                title + rf" $\rho={rho:.3f}$", transform=ax.transAxes,
                va="top", fontsize=11.5, color=color,
            )
        ax.set_ylim(-0.025 * top, 1.09 * top)
        ax.set_xlim(-0.025, 1.105)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".50", ".75", "1"])
        ax.set_yticks(np.arange(0, top + 0.1, 10 if top > 30 else 5))
        ax.set_xlabel("Calibration-only predictor", labelpad=5)
        ax.set_ylabel("Held-out flip rate (%)", labelpad=6)
        panel_title(ax, f"({chr(ord('a') + index)}) {role.capitalize()}")
    predictor_handles = [Line2D([], [], color=color, marker="o", ls="none", markersize=5, label=name) for _, name, color in predictors]
    code_handles = [Line2D([], [], color=GREY, marker="o", mfc="white", ls="none", markersize=5, label="2-bit"), Line2D([], [], color=GREY, marker="D", ls="none", markersize=4.5, label="Rotated 2-bit")]
    legend = fig.legend(handles=predictor_handles, loc="lower center", bbox_to_anchor=(0.52, 0.07), ncol=2, frameon=False, columnspacing=2.0, handletextpad=0.5, fontsize=11.5)
    fig.add_artist(legend)
    fig.legend(handles=code_handles, loc="lower center", bbox_to_anchor=(0.52, 0.015), ncol=2, frameon=False, columnspacing=2.0, handletextpad=0.5, fontsize=11.5)
    return export(fig, "fig04-predictor-scatter", "原probe33标准Vamana协议的48个角色配置，逐点绘制冻结校准值及留出翻转率；距离相关转为1-rho以统一风险方向。")


def main():
    FIGURES.mkdir(exist_ok=True)
    margin_rows, margin_source = load_margin_data()
    predictor_rows, predictor_source = load_predictors()
    plotted = {"margin_bins": margin_rows, "predictors": predictor_rows}
    (SOURCE / "plotted-data.json").write_text(json.dumps(plotted, ensure_ascii=False, indent=2), encoding="utf-8")
    with (SOURCE / "predictor-points.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictor_rows[0]))
        writer.writeheader()
        writer.writerows(predictor_rows)
    exports = [comparison_geometry(), first_divergence(), margin_chart(margin_rows), predictor_chart(predictor_rows)]
    manifest = {
        "figures": exports,
        "source_sha256": {path.name: digest(path) for path in [margin_source, predictor_source]},
        "builder_sha256": digest(Path(__file__)),
        "font": "STIXGeneral / STIX math；面板标题DejaVu Sans",
        "png_dpi": 600,
        "drawing": "参考生成图的布局，原生SVG重建；实验点由冻结JSON直接生成",
        "scope": "前两图为确定性示例，后两图为既有实验结果；图3按计数合并六档，图4每类决策含24配置及两种预测器",
        "references": [path.name for path in sorted((FIGURES / "references").glob("*.png"))],
    }
    (SOURCE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("四图全部生成，数据来源和逐点数值已保存", flush=True)


if __name__ == "__main__":
    main()
