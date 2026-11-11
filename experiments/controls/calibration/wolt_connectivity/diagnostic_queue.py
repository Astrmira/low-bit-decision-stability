"""在独立源码副本中记录入口剪枝，不修改旧冻结源或原始结果。"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import psutil

SOURCE_DIR = Path(__file__).resolve().parent
PARENT = Path(os.environ.get("JMLR_GRAPH_DIR", SOURCE_DIR.parents[3] / "runs/graph"))
ROOT = Path(os.environ.get("JMLR_WOLT_DIR", PARENT / "wolt_connectivity"))
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = SOURCE_DIR.parent / "native/frozen-tdb"
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, log, env):
    assert shutil.disk_usage(ROOT).free > 22 * 2**30
    with log.open("wb") as stream:
        process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, env=env)
        parent = psutil.Process(process.pid)
        peak = 0
        while process.poll() is None:
            try:
                family = [parent] + parent.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            rss = 0
            for member in family:
                try:
                    rss += member.memory_info().rss
                    status = Path(f"/proc/{member.pid}/status").read_text()
                    rss += sum(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith(("HugetlbPages:",)))
                except (psutil.NoSuchProcess, FileNotFoundError):
                    pass
            peak = max(peak, rss)
            if rss > 56 * 2**30:
                for member in reversed(family):
                    try:
                        member.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.wait()
                raise RuntimeError("诊断进程族内存保护触发，现场保留")
            time.sleep(0.5)
        code = process.wait()
        assert code == 0, (args, code, str(log))
    return peak


def prepare():
    source = SOURCE
    clone = ROOT / "frozen-diagnostic"
    assert not clone.exists()
    shutil.copytree(source, clone)
    path = clone / "src/index/quiver.rs"
    text = path.read_text()
    helper = r'''
static DIAGNOSTIC_FINAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn diagnostic_emit(value: serde_json::Value) {
    use std::io::Write;
    static LOG: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> = std::sync::OnceLock::new();
    let Some(path) = std::env::var_os("TRIVIUM_DIAG_ROOT_LOG") else { return; };
    let file = LOG.get_or_init(|| std::sync::Mutex::new(std::fs::OpenOptions::new().create_new(true).write(true).open(path).expect("无法创建入口剪枝诊断")));
    let mut writer = file.lock().expect("入口剪枝诊断锁异常");
    serde_json::to_writer(&mut *writer, &value).expect("无法写入入口剪枝诊断");
    writeln!(&mut *writer).expect("无法写入入口剪枝换行");
}

'''
    marker = "struct ExperimentalBuildView<'a> {"
    assert text.count(marker) == 1
    text = text.replace(marker, helper + marker, 1)
    start = text.index("    fn vamana_select(")
    end = text.index("    /// 两阶段搜索", start)
    block = text[start:end]
    block = block.replace("        if selected.len() < max_k {", "        let diagnostic_prefix = (target == 0 && std::env::var_os(\"TRIVIUM_DIAG_ROOT_LOG\").is_some()).then(|| selected.clone());\n        if selected.len() < max_k {", 1)
    block = block.replace("        selected\n", '''        if let Some(prefix) = diagnostic_prefix {
            diagnostic_emit(serde_json::json!({"kind":"prune","stage":if DIAGNOSTIC_FINAL.load(std::sync::atomic::Ordering::Relaxed) {"final"} else {"reverse"},"target":target,"candidates":candidates,"prefix":prefix,"selected":selected}));
        }
        selected
''', 1)
    text = text[:start] + block + text[end:]
    start = text.index("    pub(crate) fn batch_build_from_store(")
    end = text.index("    /// 使用外部预构建签名的实验兼容入口", start)
    block = text[start:end]
    marker = "        // final-prune：把惰性追加导致 deg>m0 的节点并行收敛到 m0\n"
    assert block.count(marker) == 1
    block = block.replace(marker, '''        diagnostic_emit(serde_json::json!({"kind":"pre_final","neighbors":concurrent_adj.neighbors_locked(0, &locks)}));
        DIAGNOSTIC_FINAL.store(true, std::sync::atomic::Ordering::Relaxed);
''' + marker, 1)
    text = text[:start] + block + text[end:]
    path.write_text(text)
    changes = []
    for original in sorted(source.rglob("*")):
        if original.is_file():
            relative = original.relative_to(source)
            if digest(original) != digest(clone / relative):
                changes.append(str(relative))
    assert changes == ["src/index/quiver.rs"]
    save("diagnostic-source.json", {"source_quiver_sha256": digest(source / "src/index/quiver.rs"), "diagnostic_quiver_sha256": digest(path), "changed_files": changes, "scope": "仅增加入口剪枝和回填日志"})
    native = ROOT / "native"
    native.mkdir()
    runner = (SOURCE_DIR / "native-runner-source.rs").read_text()
    old = "num_threads(4)"
    assert runner.count(old) == 1
    runner = runner.replace(old, 'num_threads(config["build_threads"].as_u64().unwrap_or(4) as usize)')
    (native / "main.rs").write_text(runner)
    (native / "Cargo.toml").write_text('''[package]
name = "wolt-connectivity-trace"
version = "0.1.0"
edition = "2024"
[workspace]
[dependencies]
triviumdb = { path = "../frozen-diagnostic", features = ["ablation"] }
serde_json = "1.0"
rayon = "1.10"
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
[[bin]]
name = "wolt-connectivity-trace"
path = "main.rs"
''')
    shutil.copy2(SOURCE_DIR.parent / "native/Cargo.lock", native / "Cargo.lock")


try:
    with (PARENT.parent.parent / ".benchmark.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert shutil.disk_usage(ROOT).free > 30 * 2**30
        save("diagnostic-state.json", {"phase": "prepare"})
        prepare()
        env = os.environ.copy()
        env.update(CARGO_TARGET_DIR=str(PARENT / "target"), RUSTFLAGS="-C target-cpu=native")
        env["PATH"] = str(OLD / "toolchains/cargo/bin") + ":" + env["PATH"]
        original_binary = digest(PARENT / "target/release/quiver-decision-trace")
        save("diagnostic-state.json", {"phase": "compile"})
        run(["taskset", "-c", "0-3", "cargo", "build", "--offline", "--release", "--manifest-path", str(ROOT / "native/Cargo.toml"), "--bin", "wolt-connectivity-trace"], ROOT / "compile.log", env)
        assert digest(PARENT / "target/release/quiver-decision-trace") == original_binary
        for axis, threads in (("original", 4), ("original", 1), ("rotated", 1)):
            output = ROOT / "instrumented-builds" / f"{axis}-{threads}t"
            output.mkdir(parents=True)
            data = PARENT / "graph-data/wolt/20261110" / axis
            config = {"out": str(output), "dim": 512, "base": str(data / "base.f32"), "queries": str(data / "queries.f32"), "index": str(output / "index.bin"), "build_threads": threads}
            config_path = output / "config.json"
            config_path.write_text(json.dumps(config, indent=2))
            build_env = env.copy()
            build_env["TRIVIUM_DIAG_ROOT_LOG"] = str(output / "root-prune.jsonl")
            save("diagnostic-state.json", {"phase": "build", "axis": axis, "threads": threads})
            peak = run(["taskset", "-c", "0-3", str(PARENT / "target/release/wolt-connectivity-trace"), "build", str(config_path)], output / "build.log", build_env)
            (output / "receipt.json").write_text(json.dumps({"peak_rss_hugetlb_bytes": peak, "binary_sha256": digest(PARENT / "target/release/wolt-connectivity-trace")}, indent=2))
            print("入口剪枝诊断构图完成", axis, threads, flush=True)
        save("diagnostic-state.json", {"status": "completed", "builds": 3})
except Exception as error:
    save("diagnostic-alert.json", {"error": str(error)})
    raise
