# TicketTriage AI

Fine-tuned customer support triage for banking messages. A small open-weight Cohere Labs model is fine-tuned to label each message with one of the 77 Banking77 intents, and a deterministic rule table routes the intent to a support department. Built for the Lexsi AI Safety and Alignment Hackathon, Track 1 (Fine-tuning and Data Curation), Problem 2 (Replacing a Closed-Model API).

## Pipeline

```
Banking77 -> CuratorKIT (clean, dedup) -> AlignTune (LoRA SFT) -> AuditKIT (evaluation) -> Streamlit app
```

| Stage | Tool | What it does here |
|---|---|---|
| Data curation | CuratorKIT | Text cleaning and MinHash near-duplicate removal on the training split |
| Fine-tuning | AlignTune (TRL backend) | LoRA supervised fine-tuning on raw and curated data |
| Evaluation | AuditKIT | Scores base, raw-tuned and curated-tuned models on the same test set |

A small custom step removes messages that appear with two different labels, since CuratorKIT has no knowledge of our label field. Precision, recall, macro F1 and the confusion matrix are computed with scikit-learn on the same predictions that AuditKIT scores.

## Experiments

| Experiment | Model | Training data |
|---|---|---|
| A | Base | none |
| B | Base + LoRA | raw training split |
| C | Base + LoRA | CuratorKIT curated split |

The original Banking77 test split is never used for training. Validation data is a stratified 10 percent of the original training split.

## Output format

```json
{
  "input_text": "Why was I charged a fee at the ATM?",
  "intent": "cash_withdrawal_charge",
  "department": "Cash and ATM Support",
  "requires_human_review": false,
  "validation_status": "valid"
}
```

If the model output is not one of the 77 labels, the ticket is returned with `validation_status: invalid_prediction` and `requires_human_review: true`. Departments come from `classifier/routing.py`, an ordered keyword table, so the same intent always maps to the same department.

## Setup

Requires Python 3.11 or 3.12. Training needs a GPU (Google Colab T4 is enough for a small model).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/Lexsi-Labs/aligntune.git && pip install -e aligntune
huggingface-cli login
```

## Run

```bash
python data_processing/prepare_dataset.py
python data_processing/curate_dataset.py
python training/train.py curated --smoke      # quick check
python training/train.py curated
python training/train.py raw
python evaluation/run_eval.py base
python evaluation/run_eval.py raw
python evaluation/run_eval.py curated
python evaluation/compare.py
streamlit run app.py
pytest tests
```

`training/colab_training.ipynb` runs the same steps on Colab.

## Results

Fill this table from `reports/metrics_*.json` after running evaluation. No numbers are published until the runs finish.

| Model | Accuracy | Macro F1 | Latency (s per message) |
|---|---|---|---|
| Base | | | |
| Fine-tuned on raw | | | |
| Fine-tuned on curated | | | |

## Project layout

```
config.py              paths, model name, prompt format
data_processing/       download, split, CuratorKIT curation
training/              AlignTune training, adapter inference
evaluation/            AuditKIT evaluation, comparison
classifier/            validation, routing, output schema
app.py                 Streamlit dashboard
tests/                 unit tests
```

## Limitations

- Banking77 is English only and has no loan or out-of-domain examples, so those are only caught when the model emits an invalid label.
- Human review is triggered by invalid output, not by a calibrated confidence score.
- The department table is rule based and reflects my own grouping of the 77 intents.

## Licenses

Dataset: PolyAI/banking77, CC-BY-4.0 (Casanueva et al., 2020). CuratorKIT and AuditKIT are source-available under Lexsi licenses, check their terms before commercial use. Cohere Labs model terms apply to the base model.
