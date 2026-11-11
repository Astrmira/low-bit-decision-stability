#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14 - Probe 31：Diversity-only fixed-path composition

在自洽 sampled exact graph 上预先冻结 diversity-only paths，组合 order-conditional
edge 1-certificates，并验证共享事件去重后的 fixed-path survival 充分性。
"""

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATASETS as DATASET_REGISTRY, RESULTS_DIR, ROOT


pruning = import_module("10_pruning_replay")
trace = import_module("11_trace_certificate")
edge_probe = import_module("12_edge_certificate")

DATASETS = ["cohere", "minilm", "gist960"]
QUANTIZERS = ["2bit", "rotated_2bit"]
N_SAMPLE = 4_000
N_CALIBRATION_TARGETS = 80
N_GRAPH_NODES = 320
CANDIDATE_K = 64
MAX_K = 32
ALPHA = 1.2
SEED = 20260823
PATH_LENGTHS = [1, 2, 3, 4]
MAX_PATHS_PER_LENGTH = 600


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    return float(np.nan_to_num(spearmanr(left, right).statistic))


def load_sampled_dataset(name):
    prefix, dimension, _, data_format = DATASET_REGISTRY[name]
    if data_format != "f32":
        raise ValueError(f"Probe 31 只支持 f32 数据集，收到 {data_format}")
    path = ROOT / f"{prefix}_train.f32"
    total = path.stat().st_size // (np.dtype(np.float32).itemsize * dimension)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dimension))
    rng = np.random.default_rng(SEED)
    indices = rng.choice(total, min(N_SAMPLE, total), replace=False)
    vecs = np.asarray(mapped[indices], dtype=np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    print(f"  [{name}] 直接映射抽样：{total} → {len(vecs)}")
    return vecs


def randomized_orthogonal_transform(vecs):
    rng = np.random.default_rng(SEED + vecs.shape[1])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), vecs.shape[1])
    rotated = dct(vecs * signs, type=2, norm="ortho", axis=1).astype(np.float32)
    rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
    return rotated


def restricted_candidate_lists(vecs, graph_nodes):
    similarities = vecs[graph_nodes] @ vecs[graph_nodes].T
    output = {}
    for row, target in enumerate(graph_nodes):
        similarities[row, row] = -np.inf
        local = np.argpartition(-similarities[row], CANDIDATE_K)[:CANDIDATE_K]
        labels = graph_nodes[local]
        order = np.lexsort((labels, -similarities[row, local]))
        output[int(target)] = labels[order].astype(np.int64)
    return output


def build_exact_graph(vecs, graph_nodes, candidates_by_target):
    calls = {}
    adjacency = {}
    for target in graph_nodes:
        target = int(target)
        candidates = candidates_by_target[target]
        nodes, distances = pruning.distance_cache(vecs, target, candidates)
        run = pruning.run_select(target, candidates, nodes, distances)
        calls[target] = {
            "candidates": candidates,
            "nodes": nodes,
            "distances": distances,
            "run": run,
        }
        adjacency[target] = tuple(int(node) for node in run["diversity_selected"])
    return calls, adjacency


def sample_paths(adjacency):
    rng = np.random.default_rng(SEED + 31)
    nodes = np.array(sorted(adjacency), dtype=np.int64)
    output = {}
    for length in PATH_LENGTHS:
        paths = set()
        attempts = 0
        while len(paths) < MAX_PATHS_PER_LENGTH and attempts < MAX_PATHS_PER_LENGTH * 200:
            attempts += 1
            current = int(rng.choice(nodes))
            path = [current]
            valid = True
            for _ in range(length):
                options = [node for node in adjacency[current] if node not in path]
                if not options:
                    valid = False
                    break
                current = int(rng.choice(options))
                path.append(current)
            if valid:
                paths.add(tuple(path))
        output[length] = tuple(sorted(paths))
    return output


def evaluate_calls(vecs, exact_calls, encoded, model):
    slope = model["distance_calibration"]["slope"]
    output = {}
    for target, exact in exact_calls.items():
        candidates = exact["candidates"]
        nodes = exact["nodes"]
        exact_distances = exact["distances"]
        _, approx_distances = pruning.distance_cache(
            vecs, target, candidates, encoded=encoded, slope=slope
        )
        exact_run = exact["run"]
        approx_run = pruning.run_select(
            target,
            candidates,
            nodes,
            approx_distances,
            ordered_override=exact_run["ordered"],
        )
        positions = {int(node): index for index, node in enumerate(nodes)}
        events = [
            edge_probe.candidate_event(
                exact_distances,
                approx_distances,
                positions,
                target,
                decision,
                model,
            )
            for decision in exact_run["decisions"]
        ]
        event_by_candidate = {
            event["candidate"]: index for index, event in enumerate(events)
        }
        edge_certificates = {}
        for candidate in exact_run["diversity_selected"]:
            end = event_by_candidate[candidate]
            event_ids = tuple((target, index) for index in range(end + 1))
            edge_certificates[int(candidate)] = event_ids
        output[target] = {
            "events": events,
            "edge_certificates": edge_certificates,
            "surviving": set(approx_run["final_selected"]),
        }
    return output


def evaluate_path(path, calls):
    edge_event_sets = []
    edge_survival = []
    for source, target in zip(path[:-1], path[1:]):
        edge_event_sets.append(set(calls[source]["edge_certificates"][target]))
        edge_survival.append(target in calls[source]["surviving"])
    unique_events = set().union(*edge_event_sets)
    naive_count = sum(len(events) for events in edge_event_sets)
    risks = [calls[source]["events"][index]["risk"] for source, index in unique_events]
    good = all(calls[source]["events"][index]["good"] for source, index in unique_events)
    survives = all(edge_survival)
    if good and not survives:
        raise AssertionError("fixed-path event-union certificate 充分性被违反")
    risk_sum = float(sum(risks))
    standardized = [
        np.sqrt(max(0.0, -2.0 * np.log(max(risk, 1e-300)))) for risk in risks
    ]
    minimum_standardized = min(standardized, default=np.inf)
    phase_gap = minimum_standardized**2 - 2.0 * np.log(max(len(unique_events), 1))
    return {
        "path_length": len(path) - 1,
        "unique_event_count": len(unique_events),
        "naive_event_count": naive_count,
        "deduplication_rate": 0.0
        if naive_count == 0
        else 1.0 - len(unique_events) / naive_count,
        "certificate_good": good,
        "all_edges_survive": survives,
        "risk_sum": risk_sum,
        "risk_clipped": min(1.0, risk_sum),
        "minimum_standardized_margin": float(minimum_standardized),
        "phase_gap": float(phase_gap),
        "phase_condition_holds": bool(phase_gap > 0.0),
    }


def summarize(rows):
    failures = np.array([not row["all_edges_survive"] for row in rows], dtype=np.float64)
    risks = np.array([row["risk_sum"] for row in rows], dtype=np.float64)
    counts = np.array([row["unique_event_count"] for row in rows], dtype=np.float64)
    naive = np.array([row["naive_event_count"] for row in rows], dtype=np.float64)
    return {
        "path_count": len(rows),
        "mean_unique_event_count": float(counts.mean()),
        "median_unique_event_count": float(np.median(counts)),
        "p90_unique_event_count": float(np.quantile(counts, 0.90)),
        "mean_naive_event_count": float(naive.mean()),
        "mean_deduplication_rate": float(
            np.mean([row["deduplication_rate"] for row in rows])
        ),
        "certificate_saturation_rate": float(
            np.mean([row["risk_clipped"] >= 1.0 for row in rows])
        ),
        "certificate_pass_rate": float(
            np.mean([row["certificate_good"] for row in rows])
        ),
        "path_failure_rate": float(failures.mean()),
        "risk_failure_spearman": safe_spearman(risks, failures),
        "event_count_failure_spearman": safe_spearman(counts, failures),
        "phase_condition_rate": float(
            np.mean([row["phase_condition_holds"] for row in rows])
        ),
        "mean_phase_gap": float(np.mean([row["phase_gap"] for row in rows])),
        "certificate_sufficiency_holds": bool(
            all(not row["certificate_good"] or row["all_edges_survive"] for row in rows)
        ),
    }


def run_dataset(name):
    vecs = load_sampled_dataset(name)
    rotated = randomized_orthogonal_transform(vecs)
    encodings = {
        "2bit": pruning.encode_bq2(vecs),
        "rotated_2bit": pruning.encode_bq2(rotated),
    }
    rng = np.random.default_rng(SEED)
    selected = rng.choice(
        len(vecs), N_CALIBRATION_TARGETS + N_GRAPH_NODES, replace=False
    )
    calibration_targets = selected[:N_CALIBRATION_TARGETS]
    graph_nodes = selected[N_CALIBRATION_TARGETS:]
    calibration_candidates = pruning.candidate_lists(vecs, calibration_targets)
    graph_candidates = restricted_candidate_lists(vecs, graph_nodes)
    exact_calls, adjacency = build_exact_graph(vecs, graph_nodes, graph_candidates)
    paths = sample_paths(adjacency)
    outdegrees = np.array([len(edges) for edges in adjacency.values()], dtype=np.float64)
    result = {
        "dataset": name,
        "dimension": int(vecs.shape[1]),
        "graph": {
            "node_count": len(graph_nodes),
            "mean_diversity_outdegree": float(outdegrees.mean()),
            "min_diversity_outdegree": int(outdegrees.min()),
            "max_diversity_outdegree": int(outdegrees.max()),
            "path_counts": {str(length): len(values) for length, values in paths.items()},
        },
        "quantizers": {},
    }
    for quantizer in QUANTIZERS:
        encoded = encodings[quantizer]
        model = trace.fit_event_models(
            vecs, calibration_targets, calibration_candidates, encoded
        )
        calls = evaluate_calls(vecs, exact_calls, encoded, model)
        by_length = {}
        stopped_after = None
        for length in PATH_LENGTHS:
            rows = [evaluate_path(path, calls) for path in paths[length]]
            summary = summarize(rows)
            by_length[str(length)] = summary
            if length == 2 and summary["certificate_saturation_rate"] >= 0.99:
                stopped_after = 2
                break
        result["quantizers"][quantizer] = {
            "calibration": model,
            "path_evaluation": by_length,
            "stopped_after_length": stopped_after,
        }
    return result


def save_results(results):
    output = {
        "probe": 31,
        "seed": SEED,
        "datasets": DATASETS,
        "quantizers": QUANTIZERS,
        "n_sample": N_SAMPLE,
        "n_calibration_targets": N_CALIBRATION_TARGETS,
        "n_graph_nodes": N_GRAPH_NODES,
        "candidate_k": CANDIDATE_K,
        "max_k": MAX_K,
        "alpha": ALPHA,
        "path_lengths": PATH_LENGTHS,
        "max_paths_per_length": MAX_PATHS_PER_LENGTH,
        "results": results,
    }
    path = RESULTS_DIR / "probe31_standard_vamana_path_certificate.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {path}")


def main():
    results = []
    for name in DATASETS:
        print(f"\n{'=' * 72}\n数据集：{name}\n{'=' * 72}")
        result = run_dataset(name)
        results.append(result)
        graph = result["graph"]
        print(
            f"  sampled graph：nodes={graph['node_count']}, "
            f"mean diversity outdegree={graph['mean_diversity_outdegree']:.2f}"
        )
        for quantizer, metrics in result["quantizers"].items():
            for length, summary in metrics["path_evaluation"].items():
                print(
                    f"  {quantizer:<13} L={length} paths={summary['path_count']}, "
                    f"events={summary['mean_unique_event_count']:.1f}, "
                    f"sat={summary['certificate_saturation_rate']:.3f}, "
                    f"fail={summary['path_failure_rate']:.3f}, "
                    f"rho={summary['risk_failure_spearman']:.3f}"
                )
    save_results(results)


if __name__ == "__main__":
    main()
