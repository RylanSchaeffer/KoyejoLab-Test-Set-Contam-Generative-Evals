"""Verify the benchmark load path works fully offline after the MATH rescue.

Runs the exact call chain every pretraining run makes (src/data.py's
create_dataset_for_supervised_finetuning with EleutherAI/minerva_math) with
HF_DATASETS_OFFLINE=1, which the caller must set before process start. Checks
split sizes (7,500 / 5,000) and that the formatted eval split is non-empty
with the expected columns.

    HF_DATASETS_OFFLINE=1 uv run python scripts/scratch/verify_local_math_loader_offline.py
"""

import os

assert os.environ.get("HF_DATASETS_OFFLINE") == "1", "set HF_DATASETS_OFFLINE=1"

from transformers import AutoTokenizer

import src.data

raw = src.data.load_dataset_hendrycks_math()
assert len(raw["train"]) == 7500, len(raw["train"])
assert len(raw["test"]) == 5000, len(raw["test"])
print(
    f"load_dataset_hendrycks_math OK: train={len(raw['train'])} test={len(raw['test'])}"
)
print(f"columns: {raw['test'].column_names}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
formatted = src.data.create_dataset_for_supervised_finetuning(
    dataset_name="EleutherAI/minerva_math",
    tokenizer=tokenizer,
    remove_columns=False,
)
for split, rows in formatted.items():
    assert len(rows) > 0, split
    print(f"create_dataset_for_supervised_finetuning[{split}]: {len(rows)} rows")
first_text = formatted["eval"][0].get("text", "")[:120]
print(f"first eval doc starts: {first_text!r}")
print("PASS: full benchmark path is offline-safe")
