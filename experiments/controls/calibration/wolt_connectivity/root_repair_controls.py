"""用固定实际候选池的对照选边替换入口一行，验证局部修复的端到端效果。"""


import os
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import shutil
import struct
import subprocess

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
PARENT = Path(os.environ.get("JMLR_GRAPH_DIR", SOURCE_DIR.parents[3] / "runs/graph"))
ROOT = Path(os.environ.get("JMLR_WOLT_DIR", PARENT / "wolt_connectivity"))
ROOT.mkdir(parents=True, exist_ok=True)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
plan = json.loads((SOURCE_DIR / "root-repair-plan.json").read_text())
results = []
with (PARENT.parent.parent / ".benchmark.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert shutil.disk_usage(ROOT).free > 22 * 2**30
    truth = np.load(PARENT / "graph-data/wolt/20261110/truth10.npy")[128:]
    for item in plan:
        axis = item["case"].rsplit("-", 1)[0]
        source = ROOT / "instrumented-builds" / item["case"] / "index.bin"
        original = source.read_bytes()
        dim, n = struct.unpack_from("<2I", original, 8)
        chunks, = struct.unpack_from("<I", original, 48)
        assert struct.unpack_from("<2I", original, 32) == (0, 0)
        offset = 52 + n * chunks * 16 + 4
        neighbors = item["neighbors"]
        assert len(neighbors) == len(set(neighbors)) == 64 and 0 not in neighbors
        modified = bytearray(original)
        struct.pack_into("<64I", modified, offset, *neighbors)
        assert modified[:offset] == original[:offset] and modified[offset + 256:] == original[offset + 256:]
        output = ROOT / "root-repair-results" / item["case"] / item["metric"]
        output.mkdir(parents=True)
        index = output / "index.bin"
        index.write_bytes(modified)
        evidence = {**item, "source_sha256": hashlib.sha256(original).hexdigest(), "modified_sha256": hashlib.sha256(modified).hexdigest(), "changed_offset": offset, "changed_bytes_range": 256, "all_other_bytes_identical": True, "entry": 0, "degree": 64}
        (output / "intervention.json").write_text(json.dumps(evidence, indent=2))
        data = PARENT / "graph-data/wolt/20261110" / axis
        config = {"out": str(output), "dim": dim, "base": str(data / "base.f32"), "queries": str(data / "queries.f32"), "index": str(index), "widths": [128]}
        (output / "config.json").write_text(json.dumps(config, indent=2))
        with (output / "run.log").open("wb") as log:
            subprocess.run(["taskset", "-c", "4", str(PARENT / "target/release/quiver-decision-trace"), "measure", str(output / "config.json")], stdout=log, stderr=subprocess.STDOUT, check=True)
        measurements = json.loads((output / "measurement.json").read_text())["measurements"][0]["observations"]
        labels = [[v[0] for v in q] for q in measurements[0]["labels"]]
        assert len(labels) == 616 and all([[v[0] for v in q] for q in r["labels"]] == labels for r in measurements)
        recall = float(np.mean([len({x - 1 for x in q} & set(t)) / 10 for q, t in zip(labels, truth)]))
        results.append({"case": item["case"], "metric": item["metric"], "recall10": recall, "degree": 64, "entry": 0, "root_neighbors_only": True})
        print("入口选边局部修复对照完成", item["case"], item["metric"], recall, flush=True)
    (ROOT / "root-repair-summary.json").write_text(json.dumps(results, indent=2))
