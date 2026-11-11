"""Curate the training split with CuratorKIT and build AlignTune-ready folders."""
import json, os, shutil
from collections import defaultdict
from curatorkit import Curator, CuratorConfig

load = lambda p: [json.loads(l) for l in open(p)]
def save(rows, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

to_alpaca = lambda r: {"instruction": r["text"], "input": "", "output": r["intent"]}
train, val = load("data/processed/train_raw.jsonl"), load("data/processed/validation.jsonl")

# label consistency check (custom, CuratorKIT does not know our labels):
# a message that appears with two different intents is ambiguous, so drop it
seen = defaultdict(set)
for r in train:
    seen[r["text"].strip().lower()].add(r["intent"])
conflicts = {t for t, s in seen.items() if len(s) > 1}
consistent = [r for r in train if r["text"].strip().lower() not in conflicts]
save([to_alpaca(r) for r in consistent], "data/curated/input_alpaca.jsonl")

result = Curator(CuratorConfig(
    dataset="data/curated/input_alpaca.jsonl",
    format="alpaca",
    dedup="minhash",
    clean=True,
    min_tokens=1,
    export_formats=["alpaca"],
    output_dir="data/curated/run",
)).run()
result.print_summary()

# AlignTune loads these folders with datasets.load_dataset(folder)
save([to_alpaca(r) for r in train], "data/sft_raw/train.jsonl")
save([to_alpaca(r) for r in val], "data/sft_raw/validation.jsonl")
shutil.copy("data/curated/run/sft_alpaca.jsonl", "data/sft_curated/train.jsonl")
save([to_alpaca(r) for r in val], "data/sft_curated/validation.jsonl")

stats = {"raw_train": len(train), "label_conflicts_removed": len(train) - len(consistent),
         "curated_train": sum(1 for _ in open("data/sft_curated/train.jsonl")),
         "curatorkit_rejected": len(result.rejected)}
json.dump(stats, open("reports/curation_stats.json", "w"), indent=1)
print(stats)
