"""在原始索引副本中只改变入口或一条边，保持其余字节完全相同。"""


import os
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import time

SOURCE_DIR = Path(__file__).resolve().parent
PARENT = Path(os.environ.get("JMLR_GRAPH_DIR", SOURCE_DIR.parents[3] / "runs/graph"))
ROOT = Path(os.environ.get("JMLR_WOLT_DIR", PARENT / "wolt_connectivity"))
ROOT.mkdir(parents=True, exist_ok=True)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


with (PARENT.parent.parent / ".benchmark.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert shutil.disk_usage(ROOT).free > 23 * 2**30
    for axis in ("original", "rotated"):
        source = PARENT / "graph-results/wolt/20261110" / axis
        original = (source / "index.bin").read_bytes()
        assert original[:4] == b"QUIV"
        dim, n, m, m0, efc = struct.unpack_from("<5I", original, 8)
        entry, max_level = struct.unpack_from("<2I", original, 32)
        chunks, = struct.unpack_from("<I", original, 48)
        assert (dim, n, m0, entry, max_level, chunks) == (512, 200000, 64, 0, 0, 8)
        layer0 = 52 + n * chunks * 16
        assert struct.unpack_from("<I", original, layer0)[0] == 64
        for mode in ("entry1", "bridge1"):
            output = ROOT / "entry-results" / axis / mode
            output.mkdir(parents=True, exist_ok=True)
            modified = bytearray(original)
            offset = 32 if mode == "entry1" else layer0 + 4
            old, = struct.unpack_from("<I", modified, offset)
            struct.pack_into("<I", modified, offset, 1)
            assert modified[:offset] == original[:offset] and modified[offset + 4:] == original[offset + 4:]
            index = output / "index.bin"
            assert not index.exists()
            index.write_bytes(modified)
            config = {"out": str(output), "dim": dim, "base": str(PARENT / "graph-data/wolt/20261110" / axis / "base.f32"), "queries": str(PARENT / "graph-data/wolt/20261110" / axis / "queries.f32"), "index": str(index), "widths": [128]}
            (output / "config.json").write_text(json.dumps(config, indent=2))
            receipt = {"source_sha256": hashlib.sha256(original).hexdigest(), "modified_sha256": hashlib.sha256(modified).hexdigest(), "changed_u32_offset": offset, "old_value": old, "new_value": 1, "all_other_bytes_identical": True}
            (output / "intervention.json").write_text(json.dumps(receipt, indent=2))
            with (output / "run.log").open("wb") as log:
                subprocess.run(["taskset", "-c", "4", str(PARENT / "target/release/quiver-decision-trace"), "measure", str(output / "config.json")], stdout=log, stderr=subprocess.STDOUT, check=True)
            print("同图因果对照完成", axis, mode, flush=True)
    (ROOT / "entry-complete.json").write_text(json.dumps({"status": "completed", "positions": 4}))
