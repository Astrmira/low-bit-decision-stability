"""在独立云端目录串行运行构图、轨迹和无采集器的五轮性能计量。"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import fcntl
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time
import zlib

import numpy as np
import psutil
from scipy.fft import dct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("JMLR_GRAPH_DIR", ROOT.parent / "runs/graph"))
OUT.mkdir(parents=True, exist_ok=True)
PLAN = json.loads((HERE / "remote-plan.json").read_text(encoding="utf-8"))
PLAN["source_root"] = str(Path(os.environ.get("JMLR_DATA_DIR", ROOT.parent / "data")) / "inputs-200k")
NATIVE = HERE / "native"
EXE = OUT / "target/release/quiver-decision-trace"
ALLOWED = sorted(os.sched_getaffinity(0))
COMPILE_CORES = ALLOWED[:4]
QUERY_CORE = ALLOWED[4]
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(peak_gib):
    free = shutil.disk_usage(OUT).free / 2**30
    if free < peak_gib + 20:
        raise RuntimeError(f"容量保护触发，剩余{free:.2f}GiB，下一阶段峰值{peak_gib}GiB")


def run(args, log, cores, env=None):
    preflight(2)
    with log.open("wb") as stream:
        child = subprocess.Popen(["taskset", "-c", ",".join(map(str, cores)), *map(str, args)], stdout=stream, stderr=subprocess.STDOUT, env=env)
        process = psutil.Process(child.pid)
        peak = 0
        while child.poll() is None:
            try:
                family = [process] + process.children(recursive=True)
            except psutil.NoSuchProcess:
                child.wait()
                break
            rss = 0
            for member in family:
                try:
                    rss += member.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            if rss > 56 * 2**30:
                for member in reversed(family):
                    try:
                        member.terminate()
                    except psutil.NoSuchProcess:
                        pass
                child.wait()
                raise RuntimeError("进程族RSS保护触发，已停止并保留现场")
            time.sleep(1)
        if child.returncode != 0:
            raise RuntimeError(f"子进程失败，退出码{child.returncode}，日志{log}")
    return peak


def verify_inputs():
    for dataset in PLAN["inputs"]:
        for name, expected in dataset["files"].items():
            path = Path(PLAN["source_root"]) / dataset["prefix"] / name
            assert path.stat().st_size == expected["bytes"]
            crc = 0
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    crc = zlib.crc32(chunk, crc)
            assert crc == expected["crc32"], f"既有输入与本地归档不同：{path}"
    save(OUT / "input-admission.json", {"status": "passed", "source": PLAN["source_root"], "checks": PLAN["inputs"]})


def prepare():
    preflight(8)
    jobs = []
    for dataset in PLAN["inputs"]:
        name, dim = dataset["dataset"], dataset["dim"]
        source = Path(PLAN["source_root"]) / dataset["prefix"]
        values = np.fromfile(source / "base.f32", dtype=np.float32).reshape(-1, dim)
        queries = np.fromfile(source / "queries.f32", dtype=np.float32).reshape(-1, dim)
        truth = np.fromfile(source / "truth32.u32", dtype=np.uint32).reshape(-1, 32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        signs = np.random.default_rng(20260823 + dim).choice(np.array([-1, 1], dtype=np.float32), dim)
        rotated = dct(values * signs, type=2, norm="ortho", axis=1, workers=2).astype(np.float32)
        rotated_q = dct(queries * signs, type=2, norm="ortho", axis=1, workers=2).astype(np.float32)
        rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
        rotated_q /= np.maximum(np.linalg.norm(rotated_q, axis=1, keepdims=True), 1e-12)
        assert np.max(np.abs(values[:100] @ values[:100].T - rotated[:100] @ rotated[:100].T)) < 2e-6
        for seed in PLAN["seeds"]:
            directory = OUT / "graph-data" / name / str(seed)
            directory.mkdir(parents=True, exist_ok=True)
            permutation = np.random.default_rng(seed).permutation(len(values))
            inverse = np.empty(len(values), dtype=np.int64)
            inverse[permutation] = np.arange(len(values))
            np.save(directory / "permutation.npy", permutation)
            np.save(directory / "truth10.npy", inverse[truth[:, :10]].astype(np.uint32))
            for axis, base, q in (("original", values, queries), ("rotated", rotated, rotated_q)):
                folder = directory / axis
                folder.mkdir(exist_ok=True)
                base[permutation].tofile(folder / "base.f32")
                q.tofile(folder / "queries.f32")
                jobs.append({"dataset": name, "seed": seed, "axis": axis, "dim": dim, "directory": str(folder)})
        del values, queries, rotated, rotated_q
    save(OUT / "graph-jobs.json", {"jobs": jobs, "executable_sha256": digest(EXE)})


def admission():
    folder = OUT / "admission"
    for axis in ("original", "rotated"):
        destination = folder / axis
        destination.mkdir(parents=True, exist_ok=True)
        for filename, rows in (("base.f32", 2000), ("queries.f32", 16)):
            with (OUT / "graph-data/cohere/20261110" / axis / filename).open("rb") as source:
                (destination / filename).write_bytes(source.read(rows * 768 * 4))
    result = folder / "result"
    result.mkdir(exist_ok=True)
    config = {"out": str(result), "dim": 768, "base": str(folder / "original/base.f32"), "queries": str(folder / "original/queries.f32"), "index": str(result / "index.bin"), "alternate_base": str(folder / "rotated/base.f32"), "alternate_queries": str(folder / "rotated/queries.f32"), "modes": ["native", "full_sm", "exact_l0", "cross"], "widths": [64, 2048]}
    save(folder / "config.json", config)
    run([EXE, "build", folder / "config.json"], result / "build.log", COMPILE_CORES)
    run([EXE, "query", folder / "config.json"], result / "query.log", [QUERY_CORE])
    base = np.fromfile(folder / "original/base.f32", dtype=np.float32).reshape(2000, 768)
    queries = np.fromfile(folder / "original/queries.f32", dtype=np.float32).reshape(16, 768)
    exact = queries.astype(np.float64) @ base.astype(np.float64).T
    threshold = np.partition(exact, -10, axis=1)[:, -10]
    measured = json.loads((result / "exact_l0-ef2048.json").read_text())
    for row in measured["queries"]:
        labels = [r[0] - 1 for r in row["labels"]]
        assert len(labels) == len(set(labels)) == 10
        assert min(exact[row["query"], labels]) >= threshold[row["query"]] - 2e-6
    save(folder / "admission.json", {"status": "passed", "public_entry_checks": 16, "exhaustive_exact_l0_queries": 16})


def main():
    with (OUT / ".benchmark.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory_limit = next(path for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")) if path.exists())
        save(OUT / "machine.json", {"lscpu": subprocess.check_output(["lscpu", "-J"], text=True), "allowed_cpus": ALLOWED, "query_cpu": QUERY_CORE, "cgroup_memory_limit": memory_limit.read_text().strip(), "numpy": np.__version__})
        save(OUT / "state.json", {"status": "preparing", "phase": "input-integrity"})
        verify_inputs()
        env = os.environ.copy()
        env.update(CARGO_TARGET_DIR=str(OUT / "target"), RUSTFLAGS="-C target-cpu=native")
        save(OUT / "state.json", {"status": "preparing", "phase": "compile"})
        run(["cargo", "+1.98.1", "build", "--locked", "--release", "--jobs", "4", "--manifest-path", NATIVE / "Cargo.toml"], OUT / "compile.log", COMPILE_CORES, env)
        save(OUT / "binary.json", {"sha256": digest(EXE), "compiler": subprocess.check_output(["rustc", "--version"], env=env, text=True), "source_sha256": digest(NATIVE / "main.rs")})
        save(OUT / "state.json", {"status": "preparing", "phase": "data"})
        prepare()
        admission()
        jobs = json.loads((OUT / "graph-jobs.json").read_text())["jobs"]
        for job in jobs:
            source = Path(job["directory"])
            destination = OUT / "graph-results" / job["dataset"] / str(job["seed"]) / job["axis"]
            destination.mkdir(parents=True, exist_ok=True)
            config = {"out": str(destination), "base": str(source / "base.f32"), "queries": str(source / "queries.f32"), "dim": job["dim"], "index": str(destination / "index.bin"), "modes": ["native", "full_sm", "exact_l0", "cross"], "widths": [64, 128, 256], "alternate_base": str(source.parent / ("rotated" if job["axis"] == "original" else "original") / "base.f32"), "alternate_queries": str(source.parent / ("rotated" if job["axis"] == "original" else "original") / "queries.f32")}
            config_path = destination / "config.json"
            save(config_path, config)
            for phase in ("build", "query", "measure"):
                save(OUT / "state.json", {"status": "running", "job": job, "phase": phase})
                peak = run([EXE, phase, config_path], destination / (phase + ".log"), COMPILE_CORES if phase == "build" else [QUERY_CORE])
                save(destination / (phase + "-receipt.json"), {"status": "completed", "binary_sha256": digest(EXE), "config_sha256": digest(config_path), "peak_rss": peak})
            print("远端正式位置完成", job, flush=True)
        save(OUT / "state.json", {"status": "completed", "jobs": len(jobs)})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save(OUT / "alert.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
