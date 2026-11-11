"""Single place for paths and settings."""
MODEL_NAME = "CohereLabs/tiny-aya-global"  # verify against the hackathon approved list
SEED = 42
VAL_SIZE = 0.1
MAX_NEW_TOKENS = 12

DATA = "data"
REPORTS = "reports"
ADAPTERS = {"raw": "artifacts/adapter_raw", "curated": "artifacts/adapter_curated"}
SFT_DIRS = {"raw": "data/sft_raw", "curated": "data/sft_curated"}


def build_prompt(text: str) -> str:
    """Same layout AlignTune uses for instruction data, so train and inference match."""
    return f"### Instruction:\n{text}\n\n### Response:\n"
