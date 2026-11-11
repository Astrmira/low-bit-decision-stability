"""从整数码与标准剪枝定义独立检查新对照，保存稿件修改前快照。"""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data")) / "cohere_train.f32"
ALPHA = 1.2


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def select(matrix, labels, permutation):
    """独立实现标准数学规则，矩阵0号行为target。"""
    kept = []
    trace = []
    for vertex in permutation:
        rejected = any(ALPHA * matrix[vertex, prior] <= matrix[0, vertex] for prior in kept)
        trace.append((vertex, tuple(kept), rejected))
        if not rejected:
            kept.append(vertex)
        if len(kept) == 32:
            break
    return [int(labels[index]) for index in kept], trace


def main():
    receipt = json.loads(DATA.with_name(DATA.name + ".receipt.json").read_text(encoding="utf-8"))
    actual_hash = digest(DATA)
    assert actual_hash == receipt["sha256"]
    source = np.memmap(DATA, dtype=np.float32, mode="r", shape=(1_000_000, 768))
    rng = np.random.default_rng(20260823)
    vectors = np.array(source[rng.choice(len(source), 4000, replace=False)], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    targets = np.random.default_rng(20260823).choice(4000, 240, replace=False)[80:]
    similarities = vectors[targets] @ vectors.T
    controls = json.loads((OUT / "pruning-controls.json").read_text(encoding="utf-8"))
    checked = 0
    coupling_checks = 0
    for quantizer, result in controls.items():
        if quantizer == "rotated_2bit":
            from scipy.fft import dct

            rng = np.random.default_rng(20260823 + 768)
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), 768)
            transformed = dct(vectors * signs, type=2, norm="ortho", axis=1).astype(np.float32)
            transformed /= np.maximum(np.linalg.norm(transformed, axis=1, keepdims=True), 1e-12)
        else:
            transformed = vectors
        # 直接构造{-2,-1,1,2}整数向量，与原位平面评分实现独立。
        code = np.where(transformed > 0, 1, -1).astype(np.int32)
        code *= 1 + (np.abs(transformed) > np.abs(transformed).mean(axis=1, keepdims=True))
        arrays = np.load(OUT / f"pruning-{quantizer}-arrays.npz")
        for row, target in enumerate(targets):
            scores = similarities[row].copy()
            scores[target] = -np.inf
            candidates = np.lexsort((np.arange(4000), -scores))[:64]
            labels = np.concatenate([[target], candidates])
            exact = np.maximum((1 - vectors[labels] @ vectors[labels].T).astype(np.float64), 0)
            raw = (4 * 768 - code[labels] @ code[labels].T).astype(np.float64)
            order = np.lexsort((candidates, exact[0, 1:])) + 1
            exact_output, exact_trace = select(exact, labels, order)
            assert np.array_equal(exact_output + [-1] * (32 - len(exact_output)), arrays["exact_labels"][row])
            for name, (bias, slope) in result["models"].items():
                approximate = bias + slope * raw
                selected, _ = select(approximate, labels, order)
                padded = selected + [-1] * (32 - len(selected))
                assert np.array_equal(padded, arrays[f"{name}_fixed_labels"][row])
                disagreement = any(
                    any(ALPHA * approximate[vertex, prior] <= approximate[0, vertex] for prior in prefix) != rejected
                    for vertex, prefix, rejected in exact_trace
                )
                assert disagreement == (selected != exact_output)
                checked += 1
                coupling_checks += 1
    original_files = [
        "probe26_covariance_certificate.json", "probe27_standard_vamana_pruning_replay.json",
        "probe33_standard_vamana_p6_heldout_calibration.json",
    ]
    manuscript = ROOT / "paper/jmlr_submission_20260908"
    snapshot = OUT / "manuscript-before"
    snapshot.mkdir(exist_ok=True)
    for name in ["jmlr_main.tex", "appendix_theory.tex", "appendix_experiments.tex"]:
        target = snapshot / name
        if not target.exists():
            shutil.copy2(manuscript / name, target)
    (OUT / "independent-verification.json").write_text(json.dumps({
        "status": "passed",
        "full_input_sha256": actual_hash,
        "independent_label_checks": checked,
        "counterfactual_coupling_checks": coupling_checks,
        "original_result_sha256": {name: digest(ROOT / "results" / name) for name in original_files},
        "source_sha256": digest(Path(__file__)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"独立复核通过，共核对{checked}组输出与{coupling_checks}组耦合事件。", flush=True)


if __name__ == "__main__":
    main()
