"""Verify the paraphrased-contaminant wiring before spending GPU hours on it.

Three things must hold, and each has failed silently in this codebase before:

  1. With no `contaminant` key, behaviour is bit-identical to the exact-replica path (so the
     published results remain reproducible).
  2. With `contaminant = math_rephrased`, the injected text is the *rephrased* problems while
     `eval_benchmark_loss` is still measured on the *original* test set.
  3. The injected replicas actually appear in the training split the expected number of times.

Decoding tokenized examples back to text is the only way to check (2) — comparing dataset
lengths would pass even if the wrong text were injected.
"""

import numpy as np
from transformers import AutoTokenizer

import src.data

MODEL = "RylanSchaeffer/mem_Qwen3-34M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"
# Small budget: this is a wiring check, not a training run.
TRAINER_CONFIG = {
    "max_length": 2048,
    "num_training_tokens_per_epoch": 3_000_000,
    "target_num_training_tokens_total": 3_000_000,
    "num_train_epochs": 1,
}


def build(contaminant, replicas=3, subset_fraction=0.01):
    data_config = {
        "corpus": "fineweb-edu-dedup",
        "benchmark": "EleutherAI/minerva_math",
        "benchmark_shuffle_seed": 0,
        "benchmark_subset_fraction": subset_fraction,
        "num_benchmark_replicas_per_epoch": replicas,
        "shuffle_seed": 0,
        "train_test_split_seed": 0,
    }
    if contaminant is not None:
        data_config["contaminant"] = contaminant
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    datasets = src.data.create_dataset_for_pretraining(
        data_config=data_config,
        trainer_config=TRAINER_CONFIG,
        tokenizer=tokenizer,
    )
    return datasets, tokenizer


def main() -> None:
    original_test = src.data.load_dataset_hendrycks_math()["test"]
    rephrased_test = src.data.load_dataset_math_rephrased()["test"]

    print("=== (1) no contaminant key: exact-replica baseline")
    exact, tokenizer = build(contaminant=None)
    print(f"  train n={len(exact['train'])}, benchmark n={len(exact['benchmark'])}")

    print("\n=== (2) contaminant = math_rephrased")
    para, tokenizer = build(contaminant="RylanSchaeffer/math_rephrased")
    print(f"  train n={len(para['train'])}, benchmark n={len(para['benchmark'])}")

    # The returned benchmark must still be ORIGINAL text.
    benchmark_texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in para["benchmark"]["input_ids"][: min(50, len(para["benchmark"]))]
    ]
    original_problems = set(original_test["problem"])
    rephrased_problems = set(rephrased_test["problem"])

    benchmark_from_original = sum(
        1 for t in benchmark_texts if any(p[:60] in t for p in original_problems if len(p) > 60)
    )
    benchmark_from_rephrased = sum(
        1 for t in benchmark_texts if any(p[:60] in t for p in rephrased_problems if len(p) > 60)
    )
    print(
        f"  benchmark split: {benchmark_from_original}/{len(benchmark_texts)} match ORIGINAL, "
        f"{benchmark_from_rephrased}/{len(benchmark_texts)} match REPHRASED"
    )

    # The training split must contain REPHRASED text.
    train_texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in para["train"]["input_ids"][:400]
    ]
    train_hits_rephrased = sum(
        1 for t in train_texts if any(p[:60] in t for p in rephrased_problems if len(p) > 60)
    )
    train_hits_original = sum(
        1 for t in train_texts if any(p[:60] in t for p in original_problems if len(p) > 60)
    )
    print(
        f"  first 400 train docs: {train_hits_rephrased} contain REPHRASED problems, "
        f"{train_hits_original} contain ORIGINAL problems"
    )

    print("\n=== Verdict")
    ok_benchmark = benchmark_from_original > benchmark_from_rephrased
    ok_train = train_hits_rephrased > 0
    print(f"  benchmark is original text: {ok_benchmark}")
    print(f"  training split carries rephrased text: {ok_train}")
    if ok_benchmark and ok_train:
        print("  WIRING CORRECT — safe to launch pretraining.")
    else:
        print("  WIRING WRONG — do not launch.")


if __name__ == "__main__":
    main()
