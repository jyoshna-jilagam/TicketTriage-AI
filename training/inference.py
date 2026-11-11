"""Load the base model, optionally with a LoRA adapter, and return a generate function."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODEL_NAME, MAX_NEW_TOKENS, build_prompt


def load_generator(adapter_dir=None):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    def generate(text):
        inputs = tok(build_prompt(text), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate
