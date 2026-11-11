"""检查公开资料的内容哈希、Python 语法和清单完整性。"""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "publication-manifest.json").read_text(encoding="utf-8"))
for record in manifest["files"]:
    path = ROOT / record["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(f"公开文件身份不一致 {record['path']}")
    if path.suffix == ".py":
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"公开文件核验完成，共 {len(manifest['files'])} 项")

