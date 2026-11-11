# TicketTriage AI: Replacing a Closed-Model API for Banking Support Routing

## Problem

Banking support teams receive many messages about payments, cards, transfers, and account access. Sorting them manually is slow. Calling a closed-model API for every message can also add recurring costs and send customer text to an external provider.

TicketTriage AI explores whether an open-weight model can classify banking support messages locally and support consistent ticket routing.

## Approach

The project configuration selects **`CohereLabs/tiny-aya-global`** as its base model. The intended training approach is LoRA supervised fine-tuning to classify a customer message into one of the 77 Banking77 intent labels. A deterministic rule table then maps each predicted intent to a support department, and the application returns a structured JSON result.

The repository contains a pipeline for:

- **Data preparation:** download Banking77, preserve its original test split, and create a stratified validation split from the original training split.
- **Data curation:** use CuratorKIT for cleaning and MinHash near-duplicate removal. A separate label-consistency check is implemented in the project code.
- **Fine-tuning:** use AlignTune with a TRL backend and LoRA configuration, with separate raw-data and curated-data experiments.
- **Evaluation:** evaluate the base model and fine-tuned variants on the same held-out test set, with AuditKIT and supplementary classification metrics.
- **Application:** use Streamlit for ticket classification, deterministic department mapping, output validation, and human-review flags.

> **Reproducibility note:** The supplied project archive contains code and configuration, but does not contain completed training adapters, curation reports, or evaluation reports. The workflow and configuration below describe the project implementation; they should not be presented as completed experimental results until the corresponding scripts have successfully run.

## Dataset

- **Dataset:** [PolyAI/Banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **License:** CC-BY-4.0
- **Total examples:** 13,083
- **Original training split:** 10,003 examples
- **Original test split:** 3,080 examples
- **Validation split:** 1,001 examples (10% of the original training split, using stratification)
- **Training split after validation holdout:** 9,002 examples

The original test split is kept separate and is not used to train the model. The dataset attribution and license should be retained when redistributing dataset-derived artifacts.

## Curation

The project code applies a label-consistency check and then invokes CuratorKIT with text cleaning and MinHash deduplication. The exact number of examples retained after curation must be taken from the generated `reports/curation_stats.json` file after a successful run.

- Training examples before curation: **9,002**
- Training examples after curation: **[FILL FROM reports/curation_stats.json]**
- Curation report: **[LINK TO PUBLIC REPORT OR DATASET COLLECTION]**

Do not substitute an estimated count for the measured output.

## Fine-tuning

The configured base model is **`CohereLabs/tiny-aya-global`**. The training script is configured for LoRA supervised fine-tuning through AlignTune, with separate experiments for raw and curated training data.

- Base model: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)
- Fine-tuning toolkit: [AlignTune](https://github.com/Lexsi-Labs/aligntune)
- Curation toolkit: [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT)
- Evaluation toolkit: [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)

The actual adapter artifacts and training logs should be linked after successful training and upload.

## Results

**Evaluation results are pending.** The supplied archive does not include `metrics_base.json`, `metrics_raw.json`, or `metrics_curated.json`, so no measured scores or latency values can be reported yet.

| Model | Accuracy | Macro F1 | Seconds per message |
|---|---:|---:|---:|
| Base | Pending evaluation | Pending evaluation | Pending evaluation |
| Fine-tuned on raw data | Pending training/evaluation | Pending training/evaluation | Pending training/evaluation |
| Fine-tuned on curated data | Pending training/evaluation | Pending training/evaluation | Pending training/evaluation |

After running evaluation, replace these entries using the generated reports. Add a short interpretation that states whether curated-data fine-tuning improved, matched, or reduced performance compared with raw-data fine-tuning. Do not claim that curation helped unless the measured results support that conclusion.

## Safety and reliability

The classifier validates its generated intent against the 77 allowed Banking77 labels. An output outside that label set is marked as an invalid prediction and flagged for human review. The department recommendation is produced by a fixed mapping table in the application code; it is not a separately trained department classifier.

This validation prevents an unknown label from being silently treated as a valid route. It does not guarantee that a valid but incorrect intent prediction is safe, so human review remains important for operational use.

## Limitations

- The dataset is English-only and does not provide representative out-of-domain or loan-support examples.
- Human review is triggered for invalid labels; the current design does not use a calibrated confidence threshold.
- Department routing is rule-based and depends on the project's predefined intent-to-department mapping.
- Performance, latency, and the effect of curation cannot be assessed until the training and evaluation runs have completed.
- The selected base model's license, hackathon eligibility, and runtime compatibility should be checked before public or commercial deployment.

## Links

- **Code:** [ADD PUBLIC GITHUB REPOSITORY URL]
- **Base model:** https://huggingface.co/CohereLabs/tiny-aya-global
- **Model collection:** [ADD PUBLIC HUGGING FACE MODEL COLLECTION URL]
- **Dataset and reports collection:** [ADD PUBLIC HUGGING FACE DATASET COLLECTION URL]
- **Demo:** [ADD PUBLIC HUGGING FACE SPACE URL]

## Acknowledgements

This project uses the [PolyAI Banking77 dataset](https://huggingface.co/datasets/PolyAI/banking77), and the CuratorKIT, AlignTune, and AuditKIT toolkits from Lexsi Labs. Please retain the relevant dataset attribution and comply with the applicable licenses and terms for every model and toolkit used.
