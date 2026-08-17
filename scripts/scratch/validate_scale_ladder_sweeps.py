"""Validate the v1-style contamination sweep configs before launching them.

Covers every v1-recipe pretraining sweep directory: the Qwen3 scale ladder
(`pt_v1_scale_ladder`), the Gemma 3 arm (`pt_gemma3`), and the GSM8K / MBPP
contamination sweeps (`pt_gsm8k`, `pt_mbpp`).

Checks the things that would otherwise fail hours into a run, or silently
produce runs that do not belong on the published ladder:

  1. The YAML parses.
  2. `program` is the v1 script (the v2 script KeyErrors on these files, and its
     optimizer differs on five axes -- see D4 in docs/EXPERIMENT_CHECKLIST.md).
  3. `train_test_split_seed` is present (src/data.py reads it unguarded; the
     published v1 YAMLs predate the key and die immediately without it).
  4. The v1 optimizer keys are present and the v2 ones are absent.
  5. The model size exists in its family's size table in src.models.
  6. The benchmark is one the injection path supports.
  7. Every dose fits the per-epoch token budget: create_dataset_for_pretraining
     raises ValueError when R x tokens/replica exceeds 20 x ot x N -- exactly
     how the published MATH 34M R>=1000 runs died.
  8. Reports predicted gradient-accumulation rounding, since v1 uses math.ceil:
     a value just above an integer overshoots the target batch by a full step.

Usage:
    uv run python scripts/scratch/validate_scale_ladder_sweeps.py
"""

import glob
import math
import sys

import yaml

import src.globals
from src.models import (
    gemma3_parameters_to_depths_widths_and_intermediates,
    qwen3_parameters_to_depths_and_widths,
)

SWEEP_GLOBS = [
    "sweeps/pt_v1_scale_ladder/*.yaml",
    "sweeps/pt_gemma3/*.yaml",
    "sweeps/pt_gsm8k/*.yaml",
    "sweeps/pt_mbpp/*.yaml",
]

V1_REQUIRED = ["warmup_steps", "weight_decay"]
V2_FORBIDDEN = ["adam_beta1", "adam_beta2", "warmup_ratio", "full_determinism"]

# Measured total parameter counts for the Gemma 3 dense sizes
# (scripts/scratch/smoke_test_gemma3_configs.py; src/models.py table comment).
GEMMA3_TOTAL_PARAMS = {
    "107M": 107_338_816,
    "163M": 163_064_000,
    "268M": 268_098_176,
    "497M": 497_378_176,
}

# Contaminant tokens per replica under the family tokenizer with the injection
# template. MATH/GSM8K/MBPP measured under the Qwen3 tokenizer 2026-08-17
# (scratch measurement recorded in sweeps/pt_gsm8k/README.md and
# sweeps/pt_mbpp/README.md); MATH matches the ~1.5e6 in checklist 3.2. Treat as
# estimates good to a few percent -- the hard authority is the ValueError in
# create_dataset_for_pretraining.
BENCHMARK_TOKENS_PER_REPLICA = {
    "EleutherAI/minerva_math": 1.5e6,
    "madrylab/gsm8k-platinum": 227_396,
    "google-research-datasets/mbpp": 47_343,
}


def predicted_grad_accum(
    num_parameters: float, world_size: int, batch: int, max_length: int, ot: float
) -> tuple[float, int, float]:
    """Mirror compute_derived_hyperparameters, with v1's math.ceil rounding."""
    target_tokens = int(20 * ot * num_parameters)
    tokens_per_opt_step = round(3.24e3 * target_tokens**0.264)
    tokens_per_fwd = world_size * batch * max_length
    unrounded = tokens_per_opt_step / tokens_per_fwd
    rounded = int(unrounded) if unrounded == int(unrounded) else int(unrounded) + 1
    realized = rounded * tokens_per_fwd
    overshoot = realized / tokens_per_opt_step - 1.0
    return unrounded, rounded, overshoot


def nominal_parameters(model_name: str, size: str) -> tuple[float | None, str]:
    """Return (nominal parameter count, family-table diagnostic) for a model."""
    if model_name.startswith("Qwen3/Qwen3-"):
        if size not in qwen3_parameters_to_depths_and_widths:
            return None, "size not in qwen3_parameters_to_depths_and_widths"
        depth, width = qwen3_parameters_to_depths_and_widths[size]
        nominal = src.globals.MODEL_NAMES_TO_PARAMETERS_DICT.get(size)
        return nominal, f"depth {depth}, width {width}"
    if model_name.startswith("Gemma3/Gemma3-"):
        if size not in gemma3_parameters_to_depths_widths_and_intermediates:
            return (
                None,
                "size not in gemma3_parameters_to_depths_widths_and_intermediates",
            )
        depth, width, inter = gemma3_parameters_to_depths_widths_and_intermediates[size]
        return (
            GEMMA3_TOTAL_PARAMS.get(size),
            f"depth {depth}, width {width}, MLP {inter}",
        )
    return None, f"unknown family for {model_name!r}"


def main() -> int:
    paths = sorted(p for pattern in SWEEP_GLOBS for p in glob.glob(pattern))
    if not paths:
        print(f"No configs matched {SWEEP_GLOBS}")
        return 1

    failures = 0
    for path in paths:
        print(f"=== {path} ===")
        try:
            config = yaml.safe_load(open(path))
        except yaml.YAMLError as exc:
            print(f"  FAIL: does not parse: {exc}")
            failures += 1
            continue

        program = config.get("program", "")
        if not program.endswith("pretrain_language_model_v1.py"):
            print(f"  FAIL: program is {program!r}, expected the v1 script")
            failures += 1
        else:
            print(f"  ok   program = {program}")

        if config.get("entity") != "rylan":
            print(f"  FAIL: entity is {config.get('entity')!r}, expected 'rylan'")
            failures += 1

        params = config["parameters"]
        data_params = params["data_config"]["parameters"]
        trainer_params = params["trainer_config"]["parameters"]

        if "train_test_split_seed" not in data_params:
            print("  FAIL: train_test_split_seed missing -> immediate KeyError")
            failures += 1
        else:
            print("  ok   train_test_split_seed present")

        missing = [k for k in V1_REQUIRED if k not in trainer_params]
        present_v2 = [k for k in V2_FORBIDDEN if k in trainer_params]
        if missing:
            print(f"  FAIL: v1 keys missing: {missing}")
            failures += 1
        if present_v2:
            print(f"  FAIL: v2 optimizer keys present: {present_v2}")
            failures += 1
        if not missing and not present_v2:
            print("  ok   v1 optimizer profile (warmup_steps/weight_decay, no v2 keys)")

        benchmark = data_params["benchmark"]["values"][0]
        if benchmark not in BENCHMARK_TOKENS_PER_REPLICA:
            print(
                f"  FAIL: benchmark {benchmark!r} not supported by the injection path"
            )
            failures += 1

        model_name = params["model_config"]["parameters"]["model_name"]["values"][0]
        size = model_name.split("-")[1]
        nominal, diag = nominal_parameters(model_name, size)
        if nominal is None:
            print(f"  FAIL: {model_name}: {diag}")
            failures += 1
            print()
            continue
        print(f"  ok   {model_name} -> {diag}")

        world = int(
            [c for c in config["command"] if str(c).startswith("--nproc_per_node")][
                0
            ].split("=")[1]
        )
        batch = trainer_params["per_device_train_batch_size"]["values"][0]
        max_len = trainer_params["max_length"]["values"][0]
        ot = trainer_params["overtrain_multiplier"]["values"][0]
        unrounded, rounded, overshoot = predicted_grad_accum(
            nominal, world, batch, max_len, ot
        )
        flag = "" if overshoot < 0.05 else "   <-- >5% overshoot, retune batch"
        print(
            f"  ..   world={world} batch={batch}: grad_accum {unrounded:.2f} "
            f"-> ceil {rounded}, effective batch {overshoot:+.1%} vs target{flag}"
        )

        doses = data_params["num_benchmark_replicas_per_epoch"]["values"]
        budget_per_epoch = 20 * ot * nominal
        tokens_per_replica = BENCHMARK_TOKENS_PER_REPLICA.get(benchmark, 0)
        infeasible = [r for r in doses if r * tokens_per_replica > budget_per_epoch]
        if infeasible:
            print(
                f"  FAIL: doses {infeasible} exceed the per-epoch budget "
                f"({budget_per_epoch:.3g} tokens at {tokens_per_replica:,.0f}/replica) "
                "-> ValueError in create_dataset_for_pretraining"
            )
            failures += 1
        else:
            print(f"  ..   {len(doses)} doses: {doses} (all fit the epoch budget)")
        print()

    print(f"{len(paths)} configs checked, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
