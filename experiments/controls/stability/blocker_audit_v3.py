"""独立重放原样本，核对最强阻挡者实现并比较较松的逐项和。"""


import os
from common import OUT, ROOT, load_frozen, load_json, module, save_json
import numpy as np


pruning = module("10_pruning_replay")
edge = module("12_edge_certificate")
paths_module = module("14_path_certificate")
original_cache = pruning.distance_cache
encoded_values = {}


def distance_cache(values, target, candidates, encoded=None, slope=1.0):
    if encoded is None:
        return original_cache(values, target, candidates)
    key = id(encoded)
    if key not in encoded_values:
        signs, strong = encoded
        encoded_values[key] = np.where(signs, 1, -1).astype(np.float32) * (1 + strong.astype(np.float32))
    nodes = np.concatenate([[target], candidates])
    codes = encoded_values[key][nodes]
    raw = (4 * codes.shape[1] - codes @ codes.T).astype(np.float64)
    return nodes, slope * raw


pruning.distance_cache = distance_cache


def events_for_call(exact, approximate, positions, target, run, model):
    rows = []
    for decision in run["decisions"]:
        candidate = decision["candidate"]
        selected = decision["selected_prefix"]
        cpos, tpos = positions[candidate], positions[target]
        spos = np.asarray([positions[node] for node in selected], dtype=np.int64)
        psi = exact[tpos, cpos] - 1.2 * exact[cpos, spos]
        quantized = approximate[tpos, cpos] - 1.2 * approximate[cpos, spos]
        blockers = psi >= 0
        assert bool(np.any(blockers)) == decision["dominated"]
        changed = bool(np.any(quantized >= 0)) != decision["dominated"]
        if decision["dominated"] and changed:
            assert np.all(quantized[blockers] < 0)
        if not decision["dominated"] and changed:
            assert np.any(quantized >= 0)
        threshold = np.where(psi >= 0, psi + model["pruning_mean"], -psi - model["pruning_mean"])
        risks = np.ones(len(psi), dtype=np.float64)
        valid = (psi != 0) & (threshold > 0)
        risks[valid] = np.exp(-.5 * (threshold[valid] / model["pruning_scale"]) ** 2)
        summed = float(np.sum(risks))
        tight = float(np.min(risks[blockers])) if np.any(blockers) else min(1.0, summed)
        legacy = edge.candidate_event(exact, approximate, positions, target, decision, model)
        assert abs(tight - legacy["risk"]) < 1e-12
        assert changed == (not legacy["good"])
        assert tight <= summed + 1e-12
        rows.append({"candidate": candidate, "risk": tight, "all_sum": summed, "good": not changed,
                     "exact_pruned": bool(decision["dominated"]), "blocker_count": int(np.sum(blockers))})
    return rows


def summarize_bound(all_sum, strongest, failure):
    all_sum, strongest, failure = map(np.asarray, (all_sum, strongest, failure))
    assert np.all(strongest <= all_sum + 1e-10)
    return {"count": len(all_sum), "failure_rate": float(np.mean(failure)),
            "all_sum_saturation": float(np.mean(all_sum >= 1)),
            "strongest_saturation": float(np.mean(strongest >= 1)),
            "all_sum_mean": float(np.mean(all_sum)), "strongest_mean": float(np.mean(strongest))}


def audit_edges(dataset, values, encoded, prior):
    rng = np.random.default_rng(20260823)
    targets = rng.choice(4000, 240, replace=False)
    pools = pruning.candidate_lists(values, targets[80:])
    model = prior["calibration"]
    slope = model["distance_calibration"]["slope"]
    all_risk, tight_risk, failures, order_changed, inversion, fixed_jaccard, free_jaccard = [], [], [], [], [], [], []
    call_id, step_sum, step_tight, step_exact_pruned, step_blockers = [], [], [], [], []
    for call, (target, pool) in enumerate(zip(targets[80:], pools)):
        nodes, exact = distance_cache(values, target, pool)
        _, approximate = distance_cache(values, target, pool, encoded, slope)
        run = pruning.run_select(target, pool, nodes, exact)
        fixed = pruning.run_select(target, pool, nodes, approximate, ordered_override=run["ordered"])
        free = pruning.run_select(target, pool, nodes, approximate)
        positions = {int(node): index for index, node in enumerate(nodes)}
        events = events_for_call(exact, approximate, positions, int(target), run, model)
        index_by_candidate = {event["candidate"]: index for index, event in enumerate(events)}
        for candidate in run["final_selected"]:
            required = events[:index_by_candidate[candidate] + 1]
            survives = candidate in fixed["final_selected"]
            assert not all(event["good"] for event in required) or survives
            all_risk.append(sum(event["all_sum"] for event in required))
            tight_risk.append(sum(event["risk"] for event in required))
            failures.append(not survives)
        for event in events:
            call_id.append(call)
            step_sum.append(event["all_sum"])
            step_tight.append(event["risk"])
            step_exact_pruned.append(event["exact_pruned"])
            step_blockers.append(event["blocker_count"])
        reference = set(run["final_selected"])
        fixed_set, free_set = set(fixed["final_selected"]), set(free["final_selected"])
        fixed_jaccard.append(len(reference & fixed_set) / len(reference | fixed_set))
        free_jaccard.append(len(reference & free_set) / len(reference | free_set))
        order_changed.append(tuple(run["ordered"]) != tuple(free["ordered"]))
        ranks = {node: index for index, node in enumerate(free["ordered"])}
        ordered_ranks = np.asarray([ranks[node] for node in run["ordered"]])
        inversion.append(float(np.sum(np.triu(ordered_ranks[:, None] > ordered_ranks[None, :], 1)) / (64 * 63 / 2)))
    result = summarize_bound(all_risk, tight_risk, failures)
    old = prior["evaluation"]["all"]
    assert result["count"] == old["edge_count"], f"旧边数量未复现：{dataset}"
    assert abs(result["strongest_saturation"] - old["pruning_certificate_saturation"]) < 1e-12
    assert abs(result["failure_rate"] - old["fixed_order_failure_rate"]) < 1e-12
    result["mean_risk_reproduction_gap"] = result["strongest_mean"] - old["mean_pruning_risk_sum"]
    result["mean_risk_relative_tolerance"] = float(np.finfo(np.float32).eps)
    assert np.isclose(result["strongest_mean"], old["mean_pruning_risk_sum"], rtol=np.finfo(np.float32).eps, atol=1e-10)
    result.update({"old_table_reproduced": True, "order_change_rate": float(np.mean(order_changed)),
                   "mean_order_inversion_fraction": float(np.mean(inversion)),
                   "fixed_order_jaccard": float(np.mean(fixed_jaccard)), "resorted_jaccard": float(np.mean(free_jaccard)),
                   "exact_pruned_steps": int(np.sum(step_exact_pruned)),
                   "mean_blockers_on_pruned_steps": float(np.mean(np.asarray(step_blockers)[step_exact_pruned]))})
    return result, {"all_sum": all_risk, "strongest": tight_risk, "failure": failures,
                    "call_id": call_id, "step_sum": step_sum, "step_strongest": step_tight,
                    "step_pruned": step_exact_pruned, "step_blockers": step_blockers}


def audit_paths(values, encoded, prior):
    selected = np.random.default_rng(20260823).choice(4000, 400, replace=False)
    graph_nodes = selected[80:]
    pools = paths_module.restricted_candidate_lists(values, graph_nodes)
    exact_calls, adjacency = paths_module.build_exact_graph(values, graph_nodes, pools)
    paths = paths_module.sample_paths(adjacency)
    model = prior["calibration"]
    calls = paths_module.evaluate_calls(values, exact_calls, encoded, model)
    all_step_risks = {}
    for target, info in exact_calls.items():
        nodes, exact = info["nodes"], info["distances"]
        _, approximate = distance_cache(values, target, info["candidates"], encoded, model["distance_calibration"]["slope"])
        positions = {int(node): index for index, node in enumerate(nodes)}
        independent = events_for_call(exact, approximate, positions, target, info["run"], model)
        all_step_risks[target] = independent
        assert len(independent) == len(calls[target]["events"])
    result, evidence = {}, {}
    for length in (1, 2, 3, 4):
        all_risks, tight_risks, failures = [], [], []
        for path in paths[length]:
            old_row = paths_module.evaluate_path(path, calls)
            events = set()
            for source, destination in zip(path[:-1], path[1:]):
                events.update(calls[source]["edge_certificates"][destination])
            all_risks.append(sum(all_step_risks[source][index]["all_sum"] for source, index in events))
            tight_risks.append(old_row["risk_sum"])
            failures.append(not old_row["all_edges_survive"])
        result[str(length)] = summarize_bound(all_risks, tight_risks, failures)
        old = prior["path_evaluation"].get(str(length))
        if old is not None:
            assert result[str(length)]["count"] == old["path_count"]
            assert abs(result[str(length)]["strongest_saturation"] - old["certificate_saturation_rate"]) < 1e-12
            assert abs(result[str(length)]["failure_rate"] - old["path_failure_rate"]) < 1e-12
        result[str(length)]["old_table_reproduced"] = old is not None
        evidence[f"length{length}_paths"] = np.asarray(paths[length], dtype=np.int64)
        evidence[f"length{length}_all_sum"] = all_risks
        evidence[f"length{length}_strongest"] = tight_risks
        evidence[f"length{length}_failure"] = failures
    return result, evidence


old_edges = {row["dataset"]: row for row in load_json(ROOT / "results/probe29_standard_vamana_edge_certificate.json")["results"]}
old_paths = {row["dataset"]: row for row in load_json(ROOT / "results/probe31_standard_vamana_path_certificate.json")["results"]}
results = []
for dataset in ("cohere", "minilm", "gist960"):
    checkpoint = OUT / f"blockers-{dataset}.json"
    if checkpoint.exists():
        previous = load_json(checkpoint)
        assert previous["dataset"] == dataset and len(previous["codes"]) == 2
        results.append(previous)
        print(f"复用完整阻挡者检查点：{dataset}", flush=True)
        continue
    values = load_frozen(dataset)
    result = {"dataset": dataset, "codes": {}}
    for code in ("2bit", "rotated_2bit"):
        transformed = pruning.randomized_orthogonal_transform(values) if code.startswith("rotated") else values
        encoded = pruning.encode_bq2(transformed)
        # 整数码内积与逐位参考式逐项相等，确认优化只改变复算开销。
        nodes = np.arange(9, dtype=np.int64)
        slow_nodes, slow = original_cache(values, 0, nodes, encoded, .001)
        fast_nodes, fast = distance_cache(values, 0, nodes, encoded, .001)
        assert np.array_equal(slow_nodes, fast_nodes) and np.array_equal(slow, fast)
        edge_result, edge_arrays = audit_edges(dataset, values, encoded, old_edges[dataset]["quantizers"][code])
        print(f"原边表离散结果已重现：{dataset}/{code}，逐项和饱和{edge_result['all_sum_saturation']:.4f}，最强阻挡者饱和{edge_result['strongest_saturation']:.4f}", flush=True)
        path_result, path_arrays = audit_paths(values, encoded, old_paths[dataset]["quantizers"][code])
        result["codes"][code] = {"edges": edge_result, "paths": path_result}
        np.savez_compressed(OUT / f"blockers-{dataset}-{code}-arrays.npz", **{f"edge_{key}": np.asarray(value) for key, value in edge_arrays.items()}, **path_arrays)
        encoded_values.clear()
        print(f"原路径表已复核：{dataset}/{code}，四个长度完整保留。", flush=True)
    save_json(f"blockers-{dataset}.json", result)
    results.append(result)
save_json("blockers.json", {"results": results, "identity": "原表已经使用最强阻挡者，新增逐项和仅作同现场对照"})
print("最强阻挡者核验全部完成。", flush=True)
