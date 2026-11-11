"""独立从扁平决策记录重建认证统计，不调用产生结果的认证函数。"""


import os
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("JMLR_CALIBRATION_DIR", Path(__file__).resolve().parent))
summary = json.loads((ROOT / "observable-certificates.json").read_text(encoding="utf-8"))
checks = []
for record in summary["results"]:
    path = ROOT / f"observable-{record['dataset']}-{record['code']}.npz"
    raw = np.load(path)
    for role in ("ranking", "pruning"):
        for selector in ("margin", "quantized_margin"):
            report = record["roles"][role][selector]
            cutoffs = np.asarray(report["cutoffs"])
            taus = np.asarray(report["tau_grid"])
            rebuilt = {}
            for split in ("cert", "validation"):
                offsets = raw[f"{role}_{split}_offsets"]
                margins = raw[f"{role}_{split}_margin"]
                residuals = raw[f"{role}_{split}_residual"]
                flips = raw[f"{role}_{split}_flip"]
                selected_score = raw[f"{role}_{split}_{selector}"]
                coverage, failures, unions = [], [], []
                for first, last in zip(offsets[:-1], offsets[1:]):
                    accepted = selected_score[first:last, None] >= cutoffs
                    count = max(1, int(last - first))
                    event = (margins[first:last, None] <= taus) | (np.abs(residuals[first:last, None]) >= taus)
                    assert np.all(~flips[first:last, None] | event)
                    coverage.append(accepted.sum(axis=0) / count)
                    failures.append((accepted & flips[first:last, None]).sum(axis=0) / count)
                    unions.append((accepted[:, :, None] & event[:, None, :]).sum(axis=0) / count)
                rebuilt[split] = tuple(np.asarray(x) for x in (coverage, failures, unions))
                suffix = "" if split == "cert" else "validation_"
                for name, value in zip(("C", "F", "U"), rebuilt[split]):
                    assert np.allclose(value, raw[f"{role}_{selector}_{suffix}{name}"], rtol=0, atol=1e-15)
            means = [x.mean(axis=0) for x in rebuilt["cert"]]
            c, f, u = means
            vc, vf, _ = [x.mean(axis=0) for x in rebuilt["validation"]]
            for kind, chosen in report["selected"].items():
                size = len(cutoffs) if kind == "failure" else len(cutoffs) * len(taus)
                epsilon = np.sqrt(np.log(size / 0.05) / (2 * len(rebuilt["cert"][0])))
                upper = np.minimum(1, ((f if kind == "failure" else u) + epsilon) / (c if kind == "failure" else c[:, None]))
                assert abs(float(upper.min()) - chosen["bound"]) < 1e-14
                index = chosen["selector"]
                assert abs(float(vc[index]) - chosen["validation_acceptance"]) < 1e-14
                assert abs(float(vf[index] / vc[index]) - chosen["validation_failure"]) < 1e-14
                assert chosen["validation_failure"] <= chosen["bound"]
                checks.append({"dataset": record["dataset"], "code": record["code"], "role": role, "selector": selector, "certificate": kind, "passed": True})
result = {"status": "passed", "checks": checks, "source_sha256": hashlib.sha256((ROOT / "observable-certificates.json").read_bytes()).hexdigest()}
(ROOT / "certificate-independent-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("可观测证书的48组选择和全部原始块统计已独立复算")
