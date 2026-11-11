import json
cells = [
 "!git clone https://github.com/Lexsi-Labs/aligntune.git && pip install -e aligntune\n!pip install curatorkit 'auditkit[transformers]' peft bitsandbytes scikit-learn",
 "from huggingface_hub import login; login()  # paste your token when asked",
 "!git clone https://github.com/YOUR_USER/tickettriage-ai.git && %cd tickettriage-ai",
 "!python data_processing/prepare_dataset.py\n!python data_processing/curate_dataset.py",
 "!python training/train.py curated --smoke",
 "!python training/train.py curated\n!python training/train.py raw",
 "!python evaluation/run_eval.py base\n!python evaluation/run_eval.py raw\n!python evaluation/run_eval.py curated\n!python evaluation/compare.py",
]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU"},
      "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": c.splitlines(True)} for c in cells]}
json.dump(nb, open("training/colab_training.ipynb", "w"), indent=1)
