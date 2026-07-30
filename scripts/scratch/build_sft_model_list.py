"""Emit the SFT checkpoint list for the 0-shot re-run, taken from the HF Hub.

An earlier version derived these names from W&B run configs. That failed with 404 on every
checkpoint: the SFT repos were transferred from `RylanSchaeffer/` to `jkazdan/` on the Hub after
those evaluations ran, so the configs record paths that no longer resolve. **The Hub is the
authority for what exists now; W&B records what existed then.**

Filters to `ot = 1` to match the 39 compute-optimal SFT checkpoints notebook 13 evaluates.
Parse `ot` as a float — the Hub carries both `ot_1_sft` and `ot_1.000_sft` spellings, and
string-matching silently drops half of them.
"""

import re

from huggingface_hub import HfApi

AUTHOR = "jkazdan"
OUT_PATH = "sweeps/eval_pt/math_overtrained/models_sft_rerun.txt"

NAME_RE = re.compile(
    r"^mem_Qwen3-(?P<size>[\d.]+M)_minerva_math_rep_(?P<rep>\d+)_sbst_"
    r"(?P<sbst>[\d.]+)_epch_(?P<epch>\d+)_ot_(?P<ot>[\d.]+)_sft$"
)


def main() -> None:
    api = HfApi()
    entries = []
    for model in api.list_models(author=AUTHOR, search="mem_Qwen3"):
        name = model.id.split("/")[-1]
        match = NAME_RE.match(name)
        if match is None:
            continue
        if abs(float(match.group("ot")) - 1.0) > 1e-9:
            continue  # SFT'd overtrained checkpoints also exist; not this experiment
        if abs(float(match.group("sbst")) - 1.0) > 1e-9:
            continue
        entries.append(
            (
                float(match.group("size").rstrip("M")),
                int(match.group("rep")),
                model.id,
            )
        )

    # Some configurations exist twice on the Hub under both `ot_1_sft` and `ot_1.000_sft`.
    # Prefer the undecorated spelling: that is the one notebook 13's 4-shot sweep evaluated, so
    # keeping it makes the 0-shot vs 4-shot comparison a comparison of protocols on identical
    # checkpoints rather than on two near-duplicate repos.
    best = {}
    for size, replicas, model_id in entries:
        key = (size, replicas)
        prefers = "_ot_1_sft" in model_id
        if key not in best or (prefers and "_ot_1_sft" not in best[key]):
            best[key] = model_id
    duplicates = len(entries) - len(best)
    if duplicates:
        print(f"Collapsed {duplicates} duplicate-spelling repo(s) to one per configuration.")

    entries = sorted((size, replicas, best[(size, replicas)]) for size, replicas in best)
    names = [model_id for _, _, model_id in entries]

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(names) + "\n")

    print(f"Wrote {len(names)} SFT checkpoints (ot=1) to {OUT_PATH}")
    by_size = {}
    for size, replicas, _ in entries:
        by_size.setdefault(size, []).append(replicas)
    for size in sorted(by_size):
        print(f"  {size:>6.0f}M: {sorted(by_size[size])}")


if __name__ == "__main__":
    main()
