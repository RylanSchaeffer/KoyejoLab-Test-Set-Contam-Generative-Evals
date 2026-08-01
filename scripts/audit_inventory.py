"""Audit what has actually been trained and evaluated.

Answers the question "does experiment X already exist?" without a cluster login, by querying
the HuggingFace Hub (which checkpoints exist) and Weights & Biases (which have been evaluated)
and diffing the two.

This exists because that question is otherwise answered by reading stale markdown. Regenerate
`docs/EXPERIMENT_INVENTORY.md` from this script's output rather than editing it by hand.

Usage:
    # Everything (slow: iterates all eval runs)
    python scripts/audit_inventory.py

    # Just the Hub side (fast, no W&B)
    python scripts/audit_inventory.py --skip-wandb

    # What could I evaluate right now that isn't evaluated?
    python scripts/audit_inventory.py --gaps-only

Note: on the local workstation `import wandb` needs PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python.
This script sets it automatically. See docs/INFRASTRUCTURE.md.
"""

import os

# The vendored protobuf in the workstation's Anaconda wandb build fails without this.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import ast
import collections
import re
from typing import Any, Dict, List, Optional


# Both namespaces, not just one. The pretrained checkpoints live under RylanSchaeffer
# and the SFT'd ones under jkazdan (a collaborator trained them), so auditing a single
# author silently reports "SFT'd: 0" -- which this script did until 2026-08-01.
HUB_AUTHORS = ("RylanSchaeffer", "jkazdan")
WANDB_ENTITY = "rylan"
EVAL_PROJECT = "memorization-scoring-vs-sampling-eval"

# Model IDs look like:
#   mem_Qwen3-344M_minerva_math_rep_100_sbst_1.0000_epch_1_ot_2
#   mem_Qwen3-344M_minerva_math_rep_100_sbst_1.0000_epch_1_ot_1_sft
# The optional _sft lives AFTER the ot field, so the suffix must be captured explicitly;
# a regex anchored on ot_([\d.]+)$ silently drops every SFT model.
MODEL_ID_PATTERN = re.compile(
    r"^mem_(?P<model>Qwen3-[\d\.]+[MB])_(?P<benchmark>.+?)"
    r"_rep_(?P<rep>[\d\.]+)_sbst_(?P<sbst>[\d\.]+)"
    r"_epch_(?P<epch>\d+)_ot_(?P<ot>[\d\.]+)(?P<suffix>.*)$"
)


def parse_model_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Parse a checkpoint name into its contamination parameters.

    Returns None for names that don't follow the convention (e.g. the `scale_mem_*` family).
    Numeric fields are floats: replica and subset-fraction values appear with inconsistent
    decimal formatting across sweep generations ("0.010" vs "0.0100"), so string comparison
    produces phantom distinct conditions.
    """
    match = MODEL_ID_PATTERN.match(model_id.split("/")[-1])
    if match is None:
        return None
    parsed = match.groupdict()
    return {
        "model_id": model_id.split("/")[-1],
        "model": parsed["model"],
        "benchmark": parsed["benchmark"],
        "replicas": float(parsed["rep"]),
        "subset_fraction": float(parsed["sbst"]),
        "epochs": int(parsed["epch"]),
        "overtrain": float(parsed["ot"]),
        "is_sft": parsed["suffix"] == "_sft",
    }


def fetch_hub_checkpoints() -> List[Dict[str, Any]]:
    """List every mem_Qwen3-* checkpoint on the Hub, parsed.

    Enumerates each namespace and filters by prefix rather than passing
    `search="mem_Qwen3"`. Hub full-text search is fuzzy: run Hub-wide it returns
    ~386 models, most of them unrelated ("meme" classifiers, Qwen3-TTS,
    memory-retrieval LoRAs), and it offers no guarantee of exhaustiveness for a
    prefix. That call is how docs/EXPERIMENT_INVENTORY.md came to claim 468 models
    and nine model sizes when the real figures are 266 and five.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    parsed = []
    for author in HUB_AUTHORS:
        for model in api.list_models(author=author, limit=None):
            if not model.id.split("/", 1)[1].startswith("mem_Qwen3"):
                continue
            entry = parse_model_id(model.id)
            if entry is not None:
                entry["author"] = author
                parsed.append(entry)
    return parsed


def fetch_evaluated_model_ids() -> collections.Counter:
    """Count finished generative eval runs per (model_id, dataset)."""
    import wandb

    api = wandb.Api(timeout=120)
    counts: collections.Counter = collections.Counter()
    for run in api.runs(
        f"{WANDB_ENTITY}/{EVAL_PROJECT}", filters={"state": "finished"}, per_page=200
    ):
        model_config = run.config.get("model_config", {})
        data_config = run.config.get("data_config", {})
        # W&B round-trips nested sweep config as a repr string.
        if isinstance(model_config, str):
            model_config = _literal_eval_or_empty(model_config)
        if isinstance(data_config, str):
            data_config = _literal_eval_or_empty(data_config)
        model_id = str(model_config.get("model", "?")).split("/")[-1]
        counts[(model_id, data_config.get("dataset", "?"))] += 1
    return counts


def _literal_eval_or_empty(value: str) -> Dict[str, Any]:
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def summarize_checkpoints(checkpoints: List[Dict[str, Any]]) -> None:
    full = [c for c in checkpoints if c["subset_fraction"] == 1.0]
    overtrained = [c for c in full if c["overtrain"] > 1.0 and not c["is_sft"]]
    sfted = [c for c in checkpoints if c["is_sft"]]

    print(f"Checkpoints on the Hub          : {len(checkpoints)}")
    print(f"  full test set (sbst = 1.0)    : {len(full)}")
    print(f"  overtrained (ot > 1)          : {len(overtrained)}")
    print(f"  SFT'd                         : {len(sfted)}")
    print(f"  subset sweeps (sbst < 1.0)    : {len(checkpoints) - len(full)}")

    print("\nOvertrained grid (model x overtrain -> replicas):")
    grid = collections.defaultdict(set)
    for entry in overtrained:
        grid[(entry["model"], entry["overtrain"])].add(entry["replicas"])
    for key in sorted(grid, key=lambda k: (k[0], k[1])):
        replicas = sorted(int(r) for r in grid[key])
        print(f"  {key[0]:<12s} ot={key[1]:<5g} {replicas}")


def report_gaps(
    checkpoints: List[Dict[str, Any]], evaluated: collections.Counter
) -> None:
    """Print checkpoints that exist on the Hub but have no finished eval run."""
    evaluated_ids = {model_id for model_id, _ in evaluated}
    missing = [c for c in checkpoints if c["model_id"] not in evaluated_ids]

    by_kind: collections.Counter = collections.Counter()
    for entry in missing:
        if entry["is_sft"]:
            kind = "sft"
        elif entry["overtrain"] > 1.0:
            kind = f"overtrained (ot={entry['overtrain']:g})"
        elif entry["subset_fraction"] < 1.0:
            kind = "subset sweep"
        else:
            kind = "compute-optimal"
        by_kind[kind] += 1

    print(f"\nUnevaluated checkpoints: {len(missing)} of {len(checkpoints)}")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind:<28s} {count}")

    print("\nUnevaluated overtrained checkpoints (the rebuttal-critical set):")
    for entry in sorted(
        (c for c in missing if c["overtrain"] > 1.0 and not c["is_sft"]),
        key=lambda c: (c["model"], c["overtrain"], c["replicas"]),
    ):
        print(f"  {entry['model_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-wandb", action="store_true", help="Hub inventory only; no W&B queries."
    )
    parser.add_argument(
        "--gaps-only",
        action="store_true",
        help="Only report checkpoints lacking a finished eval run.",
    )
    args = parser.parse_args()

    checkpoints = fetch_hub_checkpoints()
    if not args.gaps_only:
        summarize_checkpoints(checkpoints)

    if args.skip_wandb:
        return

    evaluated = fetch_evaluated_model_ids()
    if not args.gaps_only:
        print(f"\nFinished generative eval runs   : {sum(evaluated.values())}")
        by_dataset: collections.Counter = collections.Counter()
        for (_, dataset), count in evaluated.items():
            by_dataset[dataset] += count
        for dataset, count in by_dataset.most_common():
            print(f"  {dataset:<40s} {count}")

    report_gaps(checkpoints, evaluated)


if __name__ == "__main__":
    main()
