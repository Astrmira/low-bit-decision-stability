"""从实际入口剪枝候选池独立重放，隔离度量、剪枝方向和零距离边界。"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"

import json
from pathlib import Path

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
GRAPH_DIR = Path(os.environ.get("JMLR_GRAPH_DIR", SOURCE_DIR.parents[3] / "runs/graph"))
OUT = Path(os.environ.get("JMLR_WOLT_DIR", GRAPH_DIR / "wolt_connectivity"))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = GRAPH_DIR
static = json.loads((OUT / "static-analysis.json").read_text(encoding="utf-8"))


def select(order, matrix, rule, alpha=np.float32(1.2)):
    selected = []
    for index in order:
        if len(selected) >= 64:
            break
        lhs = np.asarray(matrix[index, selected], dtype=np.float32)
        rhs = np.float32(matrix[index, -1])
        if rule == "production_strict":
            dominated = np.any(lhs < alpha * rhs)
        elif rule == "production_inclusive":
            dominated = np.any(lhs <= alpha * rhs)
        elif rule == "standard_strict":
            dominated = np.any(alpha * lhs < rhs)
        elif rule == "standard_inclusive":
            dominated = np.any(alpha * lhs <= rhs)
        else:
            raise ValueError("未知剪枝对照")
        if not dominated:
            selected.append(index)
    prefix = selected.copy()
    for index in order:
        if len(selected) >= 64:
            break
        if index not in selected:
            selected.append(index)
    return prefix, selected


cases = []
for folder in sorted((OUT / "diagnostic-evidence/instrumented-builds").glob("*")):
    axis = folder.name.rsplit("-", 1)[0]
    base = np.memmap(ROOT / "graph-data/wolt/20261110" / axis / "base.f32", dtype=np.float32, mode="r", shape=(200000, 512))
    cluster = set(next(x["reachable_ids"] for x in static if x["axis"] == axis))
    logs = [json.loads(line) for line in (folder / "root-prune.jsonl").read_text(encoding="utf-8").splitlines()]
    events, first_failure = [], None
    for ordinal, event in enumerate(logs):
        if event["kind"] != "prune":
            continue
        ids = np.asarray([row[1] for row in event["candidates"]], dtype=np.int64)
        assert len(ids) == len(set(ids)) and np.all(ids != 0)
        vectors = np.asarray(base[np.r_[ids, 0]])
        thresholds = np.cumsum(np.abs(vectors), axis=1, dtype=np.float32)[:, -1] / np.float32(512)
        codes = (np.where(vectors > 0, 1, -1) * (1 + (np.abs(vectors) > thresholds[:, None]))).astype(np.int32)
        dots = codes @ codes.T
        raw = 4 * 512 - dots
        assert np.array_equal(raw[:-1, -1], [row[0] for row in event["candidates"]]), "独立SM评分与实际构图候选不一致"
        order = np.arange(len(ids))
        prefix, selected = select(order, raw, "production_strict")
        assert ids[prefix].tolist() == event["prefix"] and ids[selected].tolist() == event["selected"], "实际剪枝/回填重放不一致"
        before = sum(int(x) not in cluster for x in ids)
        after = sum(int(x) not in cluster for x in ids[selected])
        row = {"ordinal": ordinal, "stage": event["stage"], "candidates": len(ids), "outside_candidates": before, "prefix_count": len(prefix), "backfill_count": len(selected) - len(prefix), "selected_outside": after}
        events.append(row)
        if first_failure is None and before > 0 and after == 0:
            first_failure = {"event": event, "ordinal": ordinal, "ids": ids, "vectors": vectors, "codes": codes, "raw": raw, "native_summary": row}
    assert first_failure is not None, "尚未捕获全部外部边被删除的实际决策"
    item = first_failure
    ids, vectors, raw = item["ids"], item["vectors"], item["raw"]
    norms = np.diag(item["codes"] @ item["codes"].T)
    squared_l2 = norms[:, None] + norms[None, :] - 2 * (item["codes"] @ item["codes"].T)
    normalized = vectors.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    cosine = np.maximum(0.0, 1 - normalized @ normalized.T)
    # 完全相同向量的精确距离按数学定义置零，消除浮点累加尾差。
    _, groups = np.unique(vectors, axis=0, return_inverse=True)
    cosine[groups[:, None] == groups[None, :]] = 0
    controls = []
    for metric, matrix in (("raw_sm", raw), ("code_squared_l2", squared_l2), ("fp64_cosine", cosine)):
        for ordering in ("recorded", "metric_sorted"):
            order = np.arange(len(ids)) if ordering == "recorded" else np.argsort(matrix[:-1, -1], kind="stable")
            for rule in ("production_strict", "production_inclusive", "standard_strict", "standard_inclusive"):
                prefix, selected = select(order, matrix, rule)
                controls.append({"metric": metric, "ordering": ordering, "rule": rule, "prefix_count": len(prefix), "prefix_outside": sum(int(x) not in cluster for x in ids[prefix]), "backfill_count": len(selected) - len(prefix), "selected_outside": sum(int(x) not in cluster for x in ids[selected]), "prefix": ids[prefix].tolist(), "selected": ids[selected].tolist()})
    first_id = item["event"]["prefix"][0]
    vector = base[first_id].astype(np.float64)
    reference = base[0].astype(np.float64)
    first_similarity = float(vector @ reference / (np.linalg.norm(vector) * np.linalg.norm(reference)))
    cases.append({"case": folder.name, "verified_native_prunes": len(events), "events": events, "witness": {"ordinal": item["ordinal"], "native_summary": item["native_summary"], "actual_candidates": item["event"]["candidates"], "first_selected": first_id, "first_selected_exact_copy": bool(np.array_equal(base[first_id], base[0])), "first_selected_cosine_to_target": first_similarity, "raw_self_distance": int(raw[-1, -1])}, "controls": controls})
(OUT / "pruning-replay.json").write_text(json.dumps({"cases": cases, "selection": "逐项重放全部实际入口剪枝，再检查第一次将候选池所有外部连边删除的事件", "rules": {"production_strict": "d(c,s)<alpha*d(c,t)", "production_inclusive": "d(c,s)<=alpha*d(c,t)", "standard_strict": "alpha*d(c,s)<d(c,t)", "standard_inclusive": "alpha*d(c,s)<=d(c,t)"}}, ensure_ascii=False, indent=2), encoding="utf-8")
for row in cases:
    print("实际入口剪枝重放通过", row["case"], row["verified_native_prunes"], "见证事件", row["witness"]["native_summary"])
    print("按各自度量排序的固定候选池", [(x["metric"], x["rule"], x["selected_outside"]) for x in row["controls"] if x["ordering"] == "metric_sorted"])
