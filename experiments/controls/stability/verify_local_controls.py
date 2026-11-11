"""独立复算同池距离指标及其逐编码对方向。"""


import os
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


OUT = Path(os.environ.get("JMLR_OUTPUT_DIR", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
primary = json.loads((OUT / "six-codes.json").read_text(encoding="utf-8"))["results"]
local = json.loads((OUT / "local-distance-controls.json").read_text(encoding="utf-8"))["results"]
pair_rows = []
checked = 0
for old, dataset in zip(primary, local):
    assert old["dataset"] == dataset["dataset"]
    with np.load(OUT / f"local-distance-{dataset['dataset']}-arrays.npz") as arrays:
        for role in ("ranking", "pruning"):
            outcome = np.asarray([row["roles"][role]["flip_rate"] for row in old["records"]])
            for population in ("unique", "endpoint_weighted"):
                values = {"distance_mse": [], "distance_risk": []}
                for row in dataset["records"]:
                    key = f"{row['code']}_{role}_{population}"
                    exact, raw, approximate = arrays[key + "_exact"], arrays[key + "_raw"], arrays[key + "_approximate"]
                    mse = float(np.mean((exact - approximate) ** 2))
                    fidelity = 0.0 if np.ptp(raw) == 0 else float(spearmanr(exact, raw).statistic)
                    assert abs(mse - row["roles"][role][population]["distance_mse"]) < 1e-12
                    assert abs(1 - fidelity - row["roles"][role][population]["distance_risk"]) < 1e-12
                    if population == "unique":
                        assert len(np.unique(arrays[key + "_pairs"], axis=0)) == len(exact)
                    values["distance_mse"].append(mse)
                    values["distance_risk"].append(1 - fidelity)
                    checked += 1
                for metric, predictor in values.items():
                    point = float(spearmanr(predictor, outcome).statistic)
                    assert abs(point - dataset["statistics"][role][population][metric]["spearman"]) < 1e-12
                    for left in range(6):
                        for right in range(left + 1, 6):
                            delta = predictor[left] - predictor[right]
                            observed = outcome[left] - outcome[right]
                            pair_rows.append({"dataset": dataset["dataset"], "role": role, "population": population,
                                              "predictor": metric, "left": dataset["records"][left]["code"], "right": dataset["records"][right]["code"],
                                              "predicted_difference": delta, "flip_difference": observed,
                                              "concordant": delta * observed > 0, "predicted_tie": delta == 0})
with (OUT / "local-code-pair-comparisons.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
    writer.writeheader()
    writer.writerows(pair_rows)
receipt = {"endpoint_populations_verified": checked, "code_pair_rows": len(pair_rows),
           "input_sha256": hashlib.sha256((OUT / "local-distance-controls.json").read_bytes()).hexdigest(),
           "rule": "唯一端点对为局部对照主口径，重复端点按比较次数加权的口径同时保留"}
(OUT / "local-control-verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print("同池距离指标独立复算通过，唯一端点与频率加权两种口径均保留。")
