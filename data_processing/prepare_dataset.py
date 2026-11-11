"""Download Banking77 and write raw, validation and test files."""
import json, os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from config import SEED, VAL_SIZE


def save(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


try:
    ds = load_dataset("PolyAI/banking77")
except Exception:
    ds = load_dataset("PolyAI/banking77", revision="refs/convert/parquet")

names = ds["train"].features["label"].names
assert len(names) == 77, "expected 77 intents"
conv = lambda split: [{"text": r["text"], "intent": names[r["label"]]} for r in ds[split]]
train_all, test = conv("train"), conv("test")

train, val = train_test_split(train_all, test_size=VAL_SIZE, random_state=SEED,
                              stratify=[r["intent"] for r in train_all])
assert not {r["text"] for r in test} & {r["text"] for r in train}, "train/test overlap"

save(train_all, "data/raw/banking77_train.jsonl")
save(test, "data/raw/banking77_test.jsonl")
save(train, "data/processed/train_raw.jsonl")
save(val, "data/processed/validation.jsonl")
save(test, "data/processed/test.jsonl")
json.dump(names, open("data/labels.json", "w"), indent=1)
print(len(train), len(val), len(test), "intents:", len(names))
