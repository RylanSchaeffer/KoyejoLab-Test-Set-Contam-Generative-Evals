"""Adversarial verification of the math_perturbed contaminant injection.

Builds the pretraining dataset exactly as scripts/pretrain_language_model_v1.py does and counts,
by exact token-id match (not substring heuristics), how many training documents are perturbed
items and how many are original items; likewise for the benchmark split that eval_benchmark_loss
is computed on. The substring approach used by verify_paraphrased_contaminant.py cannot
distinguish "some perturbed items were injected" from "all of them were, and nothing else was".

Result (2026-07-30): train = 10,000 perturbed docs (2 x 5,000) and 0 original-only docs;
benchmark = 5,000 original docs and 0 perturbed-only docs, in the same order as the exact-replica
control. See reviews/2026_neurips/verification/PERTURBED_INJECTION_VERIFICATION.md.

Run (no GPU needed; the corpus load is the slow part, ~10 min):

    HF_HOME=/lfs/skampere1/0/shared_hf_cache PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= \
      ./mem_scoring_vs_sampling_env/bin/python scripts/scratch/verify_perturbed_contaminant.py \
      /path/to/scratch_dir

The scratch dir argument keeps save_to_disk artifacts out of the shared HF cache, whose
corpus_subset_tokenized directory the real pretraining runs use.
"""

import os
import sys

# datasets caches config.HF_DATASETS_CACHE at import time; override the env var afterwards so
# src.data writes its save_to_disk artifacts into scratch instead of clobbering the shared cache.
import datasets  # noqa: F401

SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/perturbed_verify"
os.makedirs(SCRATCH, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = SCRATCH

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import src.data  # noqa: E402

MODEL = "RylanSchaeffer/mem_Qwen3-34M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"
REPLICAS = 2
SUBSET_FRACTION = 1.0
TRAINER_CONFIG = {
    "max_length": 2048,
    "num_training_tokens_per_epoch": 4_000_000,
    "target_num_training_tokens_total": 4_000_000,
    "num_train_epochs": 1,
}


def build(contaminant):
    data_config = {
        "corpus": "fineweb-edu-dedup",
        "benchmark": "EleutherAI/minerva_math",
        "benchmark_shuffle_seed": 0,
        "benchmark_subset_fraction": SUBSET_FRACTION,
        "num_benchmark_replicas_per_epoch": REPLICAS,
        "shuffle_seed": 0,
        "train_test_split_seed": 0,
    }
    if contaminant is not None:
        data_config["contaminant"] = contaminant
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    ds = src.data.create_dataset_for_pretraining(
        data_config=data_config, trainer_config=TRAINER_CONFIG, tokenizer=tok
    )
    return ds, tok


def id_sets(tok):
    out = {}
    for name, dsname in [
        ("original", "EleutherAI/minerva_math"),
        ("rephrased", "RylanSchaeffer/math_rephrased"),
        ("perturbed", "RylanSchaeffer/math_perturbed"),
    ]:
        d = src.data.create_dataset_for_supervised_finetuning(
            dataset_name=dsname, tokenizer=tok, remove_columns=False
        )["eval"]
        out[name] = (set(tuple(x) for x in d["input_ids"]), d)
    return out


def count(split, sets, label):
    ids = [tuple(x) for x in split["input_ids"]]
    print(f"\n--- {label}: {len(ids)} documents")
    for name, (s, _) in sets.items():
        hits = sum(1 for t in ids if t in s)
        print(f"    exact token-id match to {name:10s}: {hits}")
    uniq = len(set(ids))
    print(f"    unique documents: {uniq}")
    return ids


def main():
    ds, tok = build("RylanSchaeffer/math_perturbed")
    sets = id_sets(tok)

    print("\n================ PERTURBED ARM ================")
    print(f"train n={len(ds['train'])}  benchmark n={len(ds['benchmark'])}  corpus-eval n={len(ds['eval'])}")

    train_ids = count(ds["train"], sets, "TRAIN split (what the model sees)")
    bench_ids = count(ds["benchmark"], sets, "BENCHMARK split (what eval_benchmark_loss uses)")

    # Perturbed-only / original-only, excluding documents that happen to be in both sets.
    p_only = sets["perturbed"][0] - sets["original"][0]
    o_only = sets["original"][0] - sets["perturbed"][0]
    both = sets["perturbed"][0] & sets["original"][0]
    print(f"\n  |perturbed-only|={len(p_only)}  |original-only|={len(o_only)}  |both|={len(both)}")
    print(f"  TRAIN: perturbed-only {sum(1 for t in train_ids if t in p_only)}, "
          f"original-only {sum(1 for t in train_ids if t in o_only)}, "
          f"ambiguous {sum(1 for t in train_ids if t in both)}")
    print(f"  BENCH: perturbed-only {sum(1 for t in bench_ids if t in p_only)}, "
          f"original-only {sum(1 for t in bench_ids if t in o_only)}, "
          f"ambiguous {sum(1 for t in bench_ids if t in both)}")

    # Token accounting.
    tl = np.asarray(ds["train"]["token_length"])
    print(f"\n  train tokens total: {tl.sum():,}")
    contam_tokens = sum(
        int(l) for t, l in zip(train_ids, ds["train"]["token_length"]) if t in sets["perturbed"][0]
    )
    print(f"  tokens in perturbed-matching train docs: {contam_tokens:,} "
          f"(expected {REPLICAS} x per-copy)")

    # Decoded spot check.
    print("\n--- decoded spot checks")
    pert_texts = sets["perturbed"][1]["text"]
    orig_texts = sets["original"][1]["text"]
    pert_lookup = {tuple(x): i for i, x in enumerate(sets["perturbed"][1]["input_ids"])}
    shown = 0
    for t in train_ids:
        if t in p_only and shown < 3:
            i = pert_lookup[t]
            print(f"\n  [train doc matches perturbed idx {i}]")
            print("  PERTURBED:", pert_texts[i][:300].replace("\n", " | "))
            print("  ORIGINAL :", orig_texts[i][:300].replace("\n", " | "))
            shown += 1
    b0 = tok.decode(ds["benchmark"]["input_ids"][0], skip_special_tokens=True)
    print("\n  BENCHMARK doc 0:", b0[:300].replace("\n", " | "))

    # Alignment: benchmark doc k and the k-th replica-ordered contaminant doc should correspond.
    orig_lookup = {tuple(x): i for i, x in enumerate(sets["original"][1]["input_ids"])}
    bench_idx = [orig_lookup.get(t, -1) for t in bench_ids]
    print(f"\n  benchmark split maps to original indices; first 10: {bench_idx[:10]}")
    print(f"  unmapped benchmark docs: {sum(1 for i in bench_idx if i < 0)}")

    # Exact-replica control: contaminant key absent must still inject originals.
    print("\n================ CONTROL: no contaminant key ================")
    ds2, tok2 = build(None)
    tr2 = count(ds2["train"], sets, "TRAIN split (exact-replica path)")
    bc2 = count(ds2["benchmark"], sets, "BENCHMARK split (exact-replica path)")
    ctrl_bench_idx = [orig_lookup.get(t, -1) for t in bc2]
    print(f"  control benchmark first 10 original indices: {ctrl_bench_idx[:10]}")
    print(f"  control benchmark == perturbed-arm benchmark order: {ctrl_bench_idx == bench_idx}")
    del tr2


if __name__ == "__main__":
    main()
