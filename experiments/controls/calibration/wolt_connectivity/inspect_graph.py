"""独立核对有向图闭合集合、外部入边、强连通分量和向量重复组成。"""


import os
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

SOURCE_DIR = Path(__file__).resolve().parent
GRAPH_DIR = Path(os.environ.get("JMLR_GRAPH_DIR", SOURCE_DIR.parents[3] / "runs/graph"))
OUT = Path(os.environ.get("JMLR_WOLT_DIR", GRAPH_DIR / "wolt_connectivity"))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = GRAPH_DIR
spec = importlib.util.spec_from_file_location("graph_reader", SOURCE_DIR.parent / "graph_analysis.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
results = []
for axis in ("original", "rotated"):
    offsets, neighbors, header = reader.read_graph(ROOT / "graph-results/wolt/20261110" / axis / "graph.u32")
    base = np.memmap(ROOT / "graph-data/wolt/20261110" / axis / "base.f32", dtype=np.float32, mode="r", shape=(200000, 512))
    graph = csr_matrix((np.ones(len(neighbors), dtype=np.uint8), neighbors, offsets), shape=(200000, 200000))
    reachable = breadth_first_order(graph, 0, directed=True, return_predecessors=False)
    mask = np.zeros(200000, dtype=bool)
    mask[reachable] = True
    sources = np.repeat(np.arange(200000), np.diff(offsets))
    outgoing = mask[sources] & ~mask[neighbors]
    incoming = ~mask[sources] & mask[neighbors]
    strong_count, strong_labels = connected_components(graph, directed=True, connection="strong")
    weak_count, weak_labels = connected_components(graph, directed=True, connection="weak")
    alternate = int(np.flatnonzero(~mask)[0])
    alternate_reach = breadth_first_order(graph, alternate, directed=True, return_predecessors=False)
    values = np.asarray(base[reachable])
    _, vector_counts = np.unique(values, axis=0, return_counts=True)
    threshold = np.cumsum(np.abs(values), axis=1, dtype=np.float32)[:, -1] / np.float32(512)
    codes = np.where(values > 0, 1, -1) * (1 + (np.abs(values) > threshold[:, None]))
    _, code_counts = np.unique(codes, axis=0, return_counts=True)
    exact = values.astype(np.float64) @ base[0].astype(np.float64)
    exact /= np.linalg.norm(values.astype(np.float64), axis=1) * np.linalg.norm(base[0].astype(np.float64))
    results.append({"axis": axis, **header, "reachable_ids": reachable.tolist(), "outgoing_from_reachable": int(outgoing.sum()), "incoming_from_outside": int(incoming.sum()), "strong_components": strong_count, "largest_strong_component": int(np.bincount(strong_labels).max()), "weak_components": weak_count, "largest_weak_component": int(np.bincount(weak_labels).max()), "alternate_entry": alternate, "alternate_reachable": len(alternate_reach), "entry_vector_total_copies_including_self": int(np.sum(np.all(base == base[0], axis=1))), "reachable_unique_vector_counts": sorted(vector_counts.tolist(), reverse=True), "reachable_unique_code_counts": sorted(code_counts.tolist(), reverse=True), "reachable_cosine_to_entry_min": float(exact.min()), "reachable_cosine_to_entry_max": float(exact.max())})
(OUT / "static-analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
for row in results:
    print("静态图核对完成", row["axis"], "强连通分量最大规模", row["largest_strong_component"], "闭合集合向量组", row["reachable_unique_vector_counts"], "最小余弦", row["reachable_cosine_to_entry_min"])
