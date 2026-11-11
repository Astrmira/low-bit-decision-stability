"""从真实生产图、查询访问轨迹和原生计时原始标签独立复算质量与工作量。"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
from scipy.stats import spearmanr


def read_trace(path):
    raw = np.fromfile(path, dtype="<u4")
    rows, position = [], 0
    while position < len(raw):
        query, count = map(int, raw[position:position + 2])
        position += 2
        block = raw[position:position + 4 * count].reshape(count, 4)
        position += 4 * count
        assert query == len(rows) and len(np.unique(block[:, 0])) == count
        rows.append(block)
    assert position == len(raw) and len(rows) == 744
    return rows


def read_graph(path):
    raw = np.fromfile(path, dtype="<u4")
    n, entry = map(int, raw[:2])
    if len(raw) == 2 + 65 * n and np.all(raw[2:].reshape(n, 65)[:, 0] == 64):
        neighbors = raw[2:].reshape(n, 65)[:, 1:].copy().ravel()
        offsets = np.arange(n + 1, dtype=np.int64) * 64
    else:
        neighbors = np.empty(len(raw) - n - 2, dtype=np.uint32)
        offsets = np.zeros(n + 1, dtype=np.int64)
        position = 2
        for node in range(n):
            degree = int(raw[position])
            neighbors[offsets[node]:offsets[node] + degree] = raw[position + 1:position + 1 + degree]
            offsets[node + 1] = offsets[node] + degree
            position += degree + 1
        assert position == len(raw)
    assert neighbors.max() < n
    graph = csr_matrix((np.ones(len(neighbors), dtype=np.uint8), neighbors, offsets), shape=(n, n))
    reachable = breadth_first_order(graph, entry, directed=True, return_predecessors=False)
    return offsets, neighbors, {"nodes": n, "entry": entry, "reachable_from_entry": len(reachable), "mean_degree": float(len(neighbors) / n)}


def raw_scores(query, vectors):
    # 与Rust逐项累加阈值的顺序一致，单独核对完整SM与便宜Hamming两条评分路径。
    qstrong = np.abs(query) > np.cumsum(np.abs(query), dtype=np.float32)[-1] / np.float32(len(query))
    strong = np.abs(vectors) > np.cumsum(np.abs(vectors), axis=1, dtype=np.float32)[:, -1, None] / np.float32(len(query))
    qsign, signs = query > 0, vectors > 0
    qcode = np.where(qsign, 1, -1) * (1 + qstrong.astype(np.int32))
    code = np.where(signs, 1, -1) * (1 + strong.astype(np.int32))
    full = 4 * len(query) - (code * qcode).sum(axis=1)
    cheap = (signs != qsign).sum(axis=1) + (strong != qstrong).sum(axis=1)
    return full, cheap


def model_summary(exact, raw):
    fit = 128 * 31
    x, y = raw[:fit], exact[:fit]
    denominator = float(x @ x)
    slope = float(x @ y / denominator) if denominator else 0.0
    residual = slope * raw - exact
    scale = max(float(np.std(residual[:fit], ddof=1)), 1e-12)
    positive = y[y > 0]
    taus = np.unique(np.quantile(positive, [0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7])) if len(positive) else np.array([1e-12])
    risks = [min(1.0, float(np.mean(y <= tau) + np.mean(np.abs(residual[:fit]) >= tau))) for tau in taus]
    selected = int(np.argmin(risks))
    return {"slope": slope, "positive_slope": slope > 0, "risk_score": risks[selected], "tau": float(taus[selected]), "calibration_scale": scale, "minus_mnr20": -float(np.quantile(y, 0.2)) / scale, "heldout_flip_rate": float(np.mean((raw[fit:] <= 0) & (exact[fit:] > 0))), "heldout_mean_error": float(np.mean(residual[fit:])), "heldout_error_scale": float(np.std(residual[fit:], ddof=1))}


def main(root):
    jobs = json.loads((root / "graph-jobs.json").read_text(encoding="utf-8"))["jobs"]
    rows, connectivity, diagnostics = [], [], []
    for job in jobs:
        dataset, seed, axis, dim = job["dataset"], job["seed"], job["axis"], job["dim"]
        data = root / "graph-data" / dataset / str(seed)
        result = root / "graph-results" / dataset / str(seed) / axis
        truth = np.load(data / "truth10.npy")
        base = np.memmap(data / axis / "base.f32", dtype=np.float32, mode="r", shape=(200000, dim))
        queries = np.fromfile(data / axis / "queries.f32", dtype=np.float32).reshape(744, dim)
        offsets, neighbors, graph = read_graph(result / "graph.u32")
        graph.update({"dataset": dataset, "seed": seed, "axis": axis})
        entry_vector = base[graph["entry"]]
        graph["entry_exact_duplicates"] = int(np.sum(np.all(base == entry_vector, axis=1)))
        connectivity.append(graph)
        timing = json.loads((result / "measurement.json").read_text()) if (result / "measurement.json").exists() else None
        for width in (64, 128, 256):
            for mode in ("native", "full_sm", "exact_l0", "cross"):
                records = json.loads((result / f"{mode}-ef{width}.json").read_text(encoding="utf-8"))["queries"]
                trace = read_trace(result / f"{mode}-ef{width}.trace.u32")
                recalls, visited_coverage, work = [], [], []
                for q in records[128:]:
                    qi = q["query"]
                    labels = {value[0] - 1 for value in q["labels"]}
                    assert len(labels) == len(q["labels"])
                    recalls.append(len(labels & set(truth[qi])) / 10)
                    visited_coverage.append(float(np.mean(np.isin(truth[qi], trace[qi][:, 0]))))
                    work.append(q["scored"])
                row = {"dataset": dataset, "seed": seed, "axis": axis, "ef": width, "mode": mode, "recall10": float(np.mean(recalls)), "mean_scored": float(np.mean(work)), "visited_gt_coverage": float(np.mean(visited_coverage)), "recall_by_query": recalls, "scored_by_query": work, "evaluation_queries": 616}
                if mode == "native" and timing:
                    timer = next(item for item in timing["measurements"] if item["ef"] == width)
                    for observation in timer["observations"]:
                        for query, measured in enumerate(observation["labels"], 128):
                            assert [value[0] for value in measured] == [value[0] for value in records[query]["labels"]], "原生计时标签与采集轨迹不一致"
                    latencies = np.asarray([r["latency_ns"] for r in timer["observations"]], dtype=np.float64)
                    row.update({"qps_rounds": [r["qps"] for r in timer["observations"]], "qps_mean": float(np.mean([r["qps"] for r in timer["observations"]])), "p95_ms": float(np.percentile(latencies, 95) / 1e6), "p99_ms": float(np.percentile(latencies, 99) / 1e6)})
                rows.append(row)
                if mode == "exact_l0":
                    exact_margins, full_margins, cheap_margins = [], [], []
                    for qi, block in enumerate(trace):
                        selected = np.argsort(block[:, 2], kind="stable")[:32]
                        ids = block[selected, 0]
                        assert len(ids) == 32
                        values = base[ids].astype(np.float64)
                        query = queries[qi].astype(np.float64)
                        exact = 1 - values @ query / (np.linalg.norm(values, axis=1) * np.linalg.norm(query))
                        order = np.lexsort((ids, exact))
                        exact, ids = exact[order], ids[order]
                        full, cheap = raw_scores(queries[qi], np.asarray(base[ids]))
                        raw_map = {int(r[0]): (int(r[1]), int(r[3])) for r in block}
                        assert np.array_equal(full, [raw_map[int(node)][1] for node in ids]), "独立SM评分与生产记录不一致"
                        assert np.array_equal(cheap, [raw_map[int(node)][0] for node in ids]), "独立Hamming评分与生产记录不一致"
                        exact_margins.extend(exact[1:] - exact[0])
                        full_margins.extend(full[1:] - full[0])
                        cheap_margins.extend(cheap[1:] - cheap[0])
                    exact = np.asarray(exact_margins)
                    for scorer, raw in (("full_sm", full_margins), ("native_cheap", cheap_margins)):
                        diagnostics.append({"dataset": dataset, "seed": seed, "axis": axis, "ef": width, "scorer": scorer, "population": "实际FP32-L0搜索访问集合中精确最近的32个已评分候选", **model_summary(exact, np.asarray(raw, dtype=np.float64))})
            print("已复算真实图位置", dataset, seed, axis, width, flush=True)
    overlaps = []
    for dataset in ("cohere", "wolt"):
        for seed in (20261110, 20261111):
            graphs = [read_graph(root / "graph-results" / dataset / str(seed) / axis / "graph.u32") for axis in ("original", "rotated")]
            nodes = np.random.default_rng(20261110).choice(200000, 2000, replace=False)
            jaccards = []
            for node in nodes:
                a = set(graphs[0][1][graphs[0][0][node]:graphs[0][0][node + 1]])
                b = set(graphs[1][1][graphs[1][0][node]:graphs[1][0][node + 1]])
                jaccards.append(len(a & b) / len(a | b))
            overlaps.append({"dataset": dataset, "seed": seed, "sampled_nodes": 2000, "mean_neighbor_jaccard": float(np.mean(jaccards))})
    output = {"rows": rows, "connectivity": connectivity, "diagnostics": diagnostics, "topology_overlap": overlaps, "selection": "全部冻结位置均报告，零召回位置保留", "trace_controls": "原生高层入口共享，L0导航与候选评分按对应控制替换，精排名额64"}
    (root / "graph-analysis-replayed.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("JMLR_GRAPH_DIR", Path(__file__).resolve().parents[3] / "runs/graph")))
    main(parser.parse_args().root)
