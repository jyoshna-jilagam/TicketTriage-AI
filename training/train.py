"""Fine-tune with AlignTune SFT (LoRA). Usage: python training/train.py curated|raw [--smoke]"""
import json, sys, time, torch
from aligntune.core.backend_factory import create_sft_trainer
from config import MODEL_NAME, SEED, ADAPTERS, SFT_DIRS

which = sys.argv[1]
smoke = "--smoke" in sys.argv

trainer = create_sft_trainer(
    model_name=MODEL_NAME,
    dataset_name=SFT_DIRS[which],
    backend="trl",
    output_dir=ADAPTERS[which],
    task_type="instruction_following",
    epochs=1 if smoke else 2,
    max_steps=20 if smoke else -1,
    max_samples=200 if smoke else None,
    batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_seq_length=128,
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="all-linear",
    quantization={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                  "bnb_4bit_compute_dtype": "bfloat16"},
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    seed=SEED,
)
start = time.time()
trainer.train()
trainer.save_model(ADAPTERS[which])

info = {"variant": which, "model": MODEL_NAME, "seconds": round(time.time() - start),
        "smoke": smoke, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
json.dump(info, open(f"reports/train_{which}.json", "w"), indent=1)
print(info)
