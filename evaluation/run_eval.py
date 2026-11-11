"""Evaluate base / raw / curated on the same test set with AuditKIT plus sklearn.
Usage: python evaluation/run_eval.py base|raw|curated [max_samples]"""
import json, sys, time
import auditkit as ak
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from config import ADAPTERS
from training.inference import load_generator

variant = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
test = [json.loads(l) for l in open("data/processed/test.jsonl")][:limit]
labels = json.load(open("data/labels.json"))

generate = load_generator(None if variant == "base" else ADAPTERS[variant])
preds, t0 = [], time.time()


def model_fn(prompts):
    out = []
    for p in prompts:
        raw = generate(p).strip().split("\n")[0].strip()
        preds.append(raw)
        out.append(raw)
    return out


samples = [ak.Sample(input=r["text"], target=r["intent"]) for r in test]
result = ak.evaluate(samples, model=model_fn, scorers=["exact_match"])
seconds = time.time() - t0
print(result.summary())
open(f"reports/auditkit_{variant}.txt", "w").write(result.summary())

y = [r["intent"] for r in test]
p, r, f1, _ = precision_recall_fscore_support(y, preds, labels=labels, average="macro", zero_division=0)
wf1 = precision_recall_fscore_support(y, preds, labels=labels, average="weighted", zero_division=0)[2]
report = {
    "variant": variant, "n": len(y),
    "accuracy": accuracy_score(y, preds), "macro_precision": p, "macro_recall": r,
    "macro_f1": f1, "weighted_f1": wf1,
    "invalid_output_rate": sum(x not in labels for x in preds) / len(preds),
    "latency_s_per_sample": seconds / len(y), "throughput_per_s": len(y) / seconds,
    "confusion_matrix": confusion_matrix(y, preds, labels=labels).tolist(),
    "errors": [{"text": r["text"], "true": r["intent"], "pred": q}
               for r, q in zip(test, preds) if r["intent"] != q][:200],
}
json.dump(report, open(f"reports/metrics_{variant}.json", "w"))
print({k: v for k, v in report.items() if k not in ("confusion_matrix", "errors")})
