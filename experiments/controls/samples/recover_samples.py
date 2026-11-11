"""按原抽样索引恢复小样本，所有新产物保存在独立目录。"""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import h5py
import numpy as np
import pyarrow.parquet as pq
import requests

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT.parent
DATA = Path(os.environ.get("JMLR_SAMPLE_DIR", ROOT.parent / "data/samples"))
DATA.mkdir(exist_ok=True)
LOCAL = threading.local()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def save(name, value):
    path = DATA / name
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def byte_range(url, start, end):
    response = session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
    response.raise_for_status()
    expected = f"bytes {start}-{end}/"
    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
        raise RuntimeError(f"范围响应身份不符：{start}-{end}，{response.status_code}")
    if len(response.content) != end - start + 1:
        raise RuntimeError("范围响应长度不符")
    return response.content


class RemoteFile(io.RawIOBase):
    """只用于读取HDF5元数据，禁止下载完整向量库。"""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def tell(self):
        return self.position

    def readinto(self, buffer):
        data = byte_range(self.url, self.position, self.position + len(buffer) - 1)
        buffer[:] = data
        self.position += len(data)
        return len(data)

    def readable(self):
        return True

    def seekable(self):
        return True


def sample_indices(total, dimension):
    pruning = np.random.default_rng(20260823).choice(total, 4000, replace=False)
    calibration = np.random.default_rng(20260824 + dimension).choice(total, 6000, replace=False)[:2000]
    return pruning, calibration, np.union1d(pruning, calibration)


def recover_gist():
    final = DATA / "gist960-samples.npz"
    if final.exists():
        print("GIST恢复样本已存在，保留现有文件", flush=True)
        return
    url = "https://ann-benchmarks.com/gist-960-euclidean.hdf5"
    response = session().head(url, timeout=30)
    response.raise_for_status()
    size = int(response.headers["Content-Length"])
    with h5py.File(RemoteFile(url, size), "r") as handle:
        dataset = handle["train"]
        shape, dtype, offset, chunks, compression = dataset.shape, dataset.dtype, dataset.id.get_offset(), dataset.chunks, dataset.compression
    if shape != (1000000, 960) or dtype != np.dtype("<f4") or chunks is not None or compression is not None:
        raise RuntimeError("GIST远端布局与原始连续float32训练集不匹配")
    pruning, calibration, needed = sample_indices(1000000, 960)
    values = np.empty((len(needed), 960), dtype=np.float32)
    row_bytes = 960 * 4

    def fetch(index):
        position = offset + int(needed[index]) * row_bytes
        return index, byte_range(url, position, position + row_bytes - 1)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as executor:
        pending = [executor.submit(fetch, index) for index in range(len(needed))]
        for count, future in enumerate(as_completed(pending), 1):
            index, content = future.result()
            values[index] = np.frombuffer(content, dtype="<f4")
            if count % 500 == 0:
                print(f"GIST恢复进度 {count}/{len(needed)}，耗时{time.monotonic()-started:.1f}秒", flush=True)
    # 原prepare_all.py在落盘前归一化，原实验加载器还会再归一化一次。
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    np.savez_compressed(final, pruning=values[np.searchsorted(needed, pruning)], calibration=values[np.searchsorted(needed, calibration)], pruning_indices=pruning, calibration_indices=calibration)
    save("gist960-recovery.json", {"source": url, "source_bytes": size, "etag": response.headers.get("ETag"), "offset": offset, "shape": shape, "dtype": str(dtype), "sample_sha256": digest(final), "sampled_rows": len(needed), "transferred_vector_bytes": len(needed)*row_bytes, "elapsed_seconds": time.monotonic()-started})
    print("GIST原索引小样本恢复完成", flush=True)


def recover_minilm():
    final = DATA / "minilm-samples.npz"
    if final.exists():
        print("MiniLM恢复样本已存在，保留现有文件", flush=True)
        return
    repo = "maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2"
    revision = "ab22410c0589e39371431e3dd293e4f0fa0c4b26"
    files = session().get(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/data", timeout=30).json()[:5]
    cache = Path.home()/".cache/huggingface/hub/datasets--maloyan--wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2/blobs"
    permutation = np.random.RandomState(42).permutation(1001000)
    pruning, calibration, needed = sample_indices(1000000, 384)
    original_rows = permutation[needed]
    values = np.empty((len(needed), 384), dtype=np.float32)
    provenance = []
    start = 0
    for item in files:
        local = cache/item["lfs"]["oid"]
        if not local.exists():
            local = DATA / Path(item["path"]).name
            if not local.exists():
                url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{item['path']}"
                temporary = local.with_suffix(".part")
                print(f"正在补取MiniLM原始分片 {Path(item['path']).name}", flush=True)
                with session().get(url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as target:
                        for block in response.iter_content(8 << 20):
                            target.write(block)
                if temporary.stat().st_size != item["size"] or digest(temporary) != item["lfs"]["oid"]:
                    raise RuntimeError("MiniLM分片完整性核验失败")
                os.replace(temporary, local)
        if digest(local) != item["lfs"]["oid"]:
            raise RuntimeError("MiniLM分片SHA256与冻结版本不匹配")
        parquet = pq.ParquetFile(local)
        group_start = start
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            indices = np.flatnonzero((original_rows >= group_start) & (original_rows < group_start + count))
            if len(indices):
                column = parquet.read_row_group(group, columns=["emb"], use_threads=False).column(0).combine_chunks()
                block = column.values.to_numpy(zero_copy_only=False).reshape(count, 384)
                values[indices] = block[original_rows[indices] - group_start]
            group_start += count
        provenance.append({"path": item["path"], "sha256": item["lfs"]["oid"], "rows": parquet.metadata.num_rows})
        start += parquet.metadata.num_rows
        print(f"MiniLM已恢复原始前{start}行中的所需样本", flush=True)
    if start <= int(original_rows.max()) or not np.isfinite(values).all():
        raise RuntimeError("MiniLM样本恢复未覆盖全部原索引")
    np.savez_compressed(final, pruning=values[np.searchsorted(needed, pruning)], calibration=values[np.searchsorted(needed, calibration)], pruning_indices=pruning, calibration_indices=calibration)
    save("minilm-recovery.json", {"source": repo, "revision": revision, "preprocessing": "原始前1001000行，RandomState(42)排列，前1000000作为训练集；保留原float32幅值", "shards": provenance, "sample_sha256": digest(final), "sampled_rows": len(needed)})
    print("MiniLM原索引小样本恢复完成", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="恢复原实验的小样本")
    parser.add_argument("dataset", choices=["minilm", "gist960"])
    arguments = parser.parse_args()
    {"minilm": recover_minilm, "gist960": recover_gist}[arguments.dataset]()
