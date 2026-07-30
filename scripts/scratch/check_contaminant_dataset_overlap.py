"""Dataset-level checks for the perturbed-contaminant verification (no corpus, no GPU)."""

import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

import src.data

MODEL = "RylanSchaeffer/mem_Qwen3-34M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

orig_raw = src.data.load_dataset_hendrycks_math()["test"]
reph_raw = src.data.load_dataset_math_rephrased()["test"]
pert_raw = src.data.load_dataset_math_perturbed()["test"]

print("=== raw rows / columns")
for name, d in [("original", orig_raw), ("rephrased", reph_raw), ("perturbed", pert_raw)]:
    print(f"  {name:10s} n={len(d)} cols={d.column_names}")

n = min(len(orig_raw), len(reph_raw), len(pert_raw))
print("\n=== index-aligned identity vs original (n=%d)" % n)
for name, d in [("rephrased", reph_raw), ("perturbed", pert_raw)]:
    same_p = sum(orig_raw[i]["problem"] == d[i]["problem"] for i in range(n))
    same_s = sum(orig_raw[i]["solution"] == d[i]["solution"] for i in range(n))
    same_both = sum(
        orig_raw[i]["problem"] == d[i]["problem"] and orig_raw[i]["solution"] == d[i]["solution"]
        for i in range(n)
    )
    print(f"  {name:10s} problems identical {same_p}/{n}  solutions identical {same_s}/{n}  both {same_both}/{n}")

# set-level (not index-aligned) overlap: could a perturbed item match SOME original item?
orig_sol_set = set(orig_raw["solution"])
orig_prob_set = set(orig_raw["problem"])
print("\n=== set-level overlap with original (any index)")
for name, d in [("rephrased", reph_raw), ("perturbed", pert_raw)]:
    ps = sum(1 for p in d["problem"] if p in orig_prob_set)
    ss = sum(1 for s in d["solution"] if s in orig_sol_set)
    print(f"  {name:10s} problems in original set {ps}/{len(d)}  solutions in original set {ss}/{len(d)}")

print("\n=== tokens per full copy, formatted exactly as injected")
sft = {}
for name, dsname in [
    ("original", "EleutherAI/minerva_math"),
    ("rephrased", "RylanSchaeffer/math_rephrased"),
    ("perturbed", "RylanSchaeffer/math_perturbed"),
]:
    d = src.data.create_dataset_for_supervised_finetuning(
        dataset_name=dsname, tokenizer=tok, remove_columns=False
    )["eval"]
    sft[name] = d
    tl = np.asarray(d["token_length"])
    print(
        f"  {name:10s} n={len(d)} tokens={tl.sum():,} mean={tl.mean():.1f} "
        f"max={tl.max()} n_over_2048={(tl > 2048).sum()}"
    )
base = np.asarray(sft["original"]["token_length"]).sum()
for name in ["rephrased", "perturbed"]:
    t = np.asarray(sft[name]["token_length"]).sum()
    print(f"  {name} / original = {100.0 * t / base:.1f}%")

print("\n=== full formatted-document overlap (exact text as injected)")
orig_texts = set(sft["original"]["text"])
for name in ["rephrased", "perturbed"]:
    hits = sum(1 for t in sft[name]["text"] if t in orig_texts)
    print(f"  {name:10s} documents byte-identical to some original document: {hits}/{len(sft[name])}")

print("\n=== token-id-level document overlap (what the exact-match train check will use)")
orig_ids = set(tuple(x) for x in sft["original"]["input_ids"])
for name in ["rephrased", "perturbed"]:
    hits = sum(1 for x in sft[name]["input_ids"] if tuple(x) in orig_ids)
    print(f"  {name:10s} token-identical to some original document: {hits}/{len(sft[name])}")

print("\n=== shuffle(seed=0) permutation alignment across datasets")
perms = {}
for name in ["original", "rephrased", "perturbed"]:
    d = sft[name].add_column("__idx", list(range(len(sft[name]))))
    perms[name] = d.shuffle(seed=0)["__idx"]
same_op = perms["original"] == perms["perturbed"]
same_or = perms["original"] == perms["rephrased"]
print(f"  original vs perturbed permutation identical: {same_op}")
print(f"  original vs rephrased permutation identical: {same_or}")
print(f"  first 10 of original perm: {perms['original'][:10]}")
print(f"  first 10 of perturbed perm: {perms['perturbed'][:10]}")

# Alignment sanity: shuffled original[k] and shuffled perturbed[k] should be the same problem type.
if "type" in orig_raw.column_names and "type" in pert_raw.column_names:
    so = orig_raw.add_column("__idx", list(range(len(orig_raw)))).shuffle(seed=0)
    sp = pert_raw.add_column("__idx", list(range(len(pert_raw)))).shuffle(seed=0)
    m = sum(so[i]["type"] == sp[i]["type"] for i in range(min(len(so), len(sp))))
    print(f"  post-shuffle 'type' agreement original vs perturbed: {m}/{min(len(so), len(sp))}")
