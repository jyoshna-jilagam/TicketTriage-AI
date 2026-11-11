"""Side by side comparison using AuditKIT's compare on the saved results is optional;
this prints the table from the saved metric files."""
import json, os
import pandas as pd

rows = []
for v in ("base", "raw", "curated"):
    p = f"reports/metrics_{v}.json"
    if os.path.exists(p):
        m = json.load(open(p))
        rows.append({k: m[k] for k in ("variant", "accuracy", "macro_f1", "weighted_f1",
                                       "invalid_output_rate", "latency_s_per_sample")})
print(pd.DataFrame(rows).round(4).to_string(index=False))
