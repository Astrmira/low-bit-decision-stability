"""以受限内存串行完成冻结QuIVer图的旋转配对和真实查询轨迹。"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import psutil
from scipy.fft import dct

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data")) / "inputs-200k.zip"
EXE = Path(os.environ.get("JMLR_GRAPH_EXE", OUT / "target/release" / ("quiver-decision-trace.exe" if os.name == "nt" else "quiver-decision-trace")))
SEEDS = (20261110, 20261111)
DATASETS = (("cohere", "D03", 768), ("wolt", "D02", 512))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def run(command, config, destination):
    config_path = destination / (command + "-config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save(config_path, config)
    log = destination / (command + ".log")
    if shutil.disk_usage(OUT).free < 3 * 2**30:
        raise RuntimeError("实验剩余容量不足3GiB")
    with log.open("wb") as output:
        child = subprocess.Popen([str(EXE), command, str(config_path)], stdout=output, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        peak = 0
        while child.poll() is None:
            try:
                rss = process.memory_info().rss
                peak = max(peak, rss)
                if rss > 6 * 2**30 or psutil.virtual_memory().available < 4 * 2**30:
                    child.terminate()
                    child.wait()
                    raise RuntimeError("实验触发本地内存保护，现场已保留")
            except psutil.NoSuchProcess:
                pass
            time.sleep(1)
        if child.returncode != 0:
            raise RuntimeError(f"冻结路径执行失败，退出码{child.returncode}，日志{log}")
    save(destination / (command + "-receipt.json"), {"status": "completed", "exe_sha256": digest(EXE), "config_sha256": digest(config_path), "peak_rss": peak})


def prepare():
    if shutil.disk_usage(OUT).free < 10 * 2**30:
        raise RuntimeError("数据准备需要至少10GiB可用容量")
    jobs = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for dataset, prefix, dim in DATASETS:
            print("正在准备真实图输入", dataset, flush=True)
            values = np.frombuffer(archive.read(prefix + "/base.f32"), dtype="<f4").copy().reshape(-1, dim)
            queries = np.frombuffer(archive.read(prefix + "/queries.f32"), dtype="<f4").copy().reshape(-1, dim)
            truth = np.frombuffer(archive.read(prefix + "/truth32.u32"), dtype="<u4").copy().reshape(-1, 32)
            assert values.shape == (200000, dim) and len(queries) == len(truth) == 744
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
            signs = np.random.default_rng(20260823 + dim).choice(np.array([-1, 1], dtype=np.float32), dim)
            rotated = dct(values * signs, type=2, norm="ortho", axis=1, workers=2).astype(np.float32)
            rotated_q = dct(queries * signs, type=2, norm="ortho", axis=1, workers=2).astype(np.float32)
            rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
            rotated_q /= np.maximum(np.linalg.norm(rotated_q, axis=1, keepdims=True), 1e-12)
            gap = float(np.max(np.abs(values[:100] @ values[:100].T - rotated[:100] @ rotated[:100].T)))
            assert gap < 2e-6
            # 独立穷举抽查归档真值，允许同距离边界上的重复向量。
            query_ids = np.linspace(0, 743, 16, dtype=int)
            exact = queries[query_ids].astype(np.float64) @ values.astype(np.float64).T
            cutoff = np.partition(exact, -10, axis=1)[:, -10]
            supplied = np.take_along_axis(exact, truth[query_ids, :10], axis=1)
            assert np.all(supplied >= cutoff[:, None] - 2e-6), "归档真值不匹配当前200K向量"
            del exact
            for seed in SEEDS:
                permutation = np.random.default_rng(seed).permutation(len(values))
                inverse = np.empty(len(values), dtype=np.int64)
                inverse[permutation] = np.arange(len(values))
                directory = OUT / "graph-data" / dataset / str(seed)
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / "permutation.npy", permutation)
                np.save(directory / "truth10.npy", inverse[truth[:, :10]].astype(np.uint32))
                for axis, base, q in (("original", values, queries), ("rotated", rotated, rotated_q)):
                    (directory / axis).mkdir(exist_ok=True)
                    base[permutation].tofile(directory / axis / "base.f32")
                    q.tofile(directory / axis / "queries.f32")
                    jobs.append({"dataset": dataset, "seed": seed, "axis": axis, "dim": dim, "directory": str(directory / axis)})
            save(OUT / "graph-data" / dataset / "admission.json", {"n": 200000, "queries": 744, "dim": dim, "rotation_inner_product_max_gap": gap, "independent_truth_queries": 16, "source": str(ARCHIVE), "archive_member": prefix, "calibration_queries": 128, "validation_queries": 616})
            del values, queries, rotated, rotated_q
    save(OUT / "graph-jobs.json", {"jobs": jobs, "executable_sha256": digest(EXE)})


def campaign():
    jobs = json.loads((OUT / "graph-jobs.json").read_text(encoding="utf-8"))
    assert digest(EXE) == jobs["executable_sha256"]
    for job in jobs["jobs"]:
        source = Path(job["directory"])
        destination = OUT / "graph-results" / job["dataset"] / str(job["seed"]) / job["axis"]
        destination.mkdir(parents=True, exist_ok=True)
        config = {"out": str(destination), "base": str(source / "base.f32"), "queries": str(source / "queries.f32"), "dim": job["dim"]}
        save(OUT / "graph-state.json", {"status": "running", "job": job})
        if not (destination / "build-receipt.json").exists():
            print("开始实际QuIVer构图", job, flush=True)
            run("build", config, destination)
        other = source.parent / ("rotated" if job["axis"] == "original" else "original")
        config.update({"index": str(destination / "index.bin"), "modes": ["native", "full_sm", "exact_l0", "cross"], "widths": [64, 128, 256], "alternate_base": str(other / "base.f32"), "alternate_queries": str(other / "queries.f32")})
        if not (destination / "query-receipt.json").exists():
            print("开始实际QuIVer查询轨迹", job, flush=True)
            run("query", config, destination)
        print("配对位置完成", job, flush=True)
    save(OUT / "graph-state.json", {"status": "completed", "jobs": len(jobs["jobs"])})


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        campaign()
