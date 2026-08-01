"""Validate the Phase 1 v1-style scale-ladder sweep configs before launching them.

Checks the things that would otherwise fail hours into a run, or silently produce
runs that do not belong on the published ladder:

  1. The YAML parses.
  2. `program` is the v1 script (the v2 script KeyErrors on these files, and its
     optimizer differs on five axes -- see D4 in docs/ICLR_2027_CHECKLIST.md).
  3. `train_test_split_seed` is present (src/data.py reads it unguarded; the
     published v1 YAMLs predate the key and die immediately without it).
  4. The v1 optimizer keys are present and the v2 ones are absent.
  5. The model size exists in src.models.qwen3_parameters_to_depths_and_widths.
  6. Reports predicted gradient-accumulation rounding, since v1 uses math.ceil:
     a value just above an integer overshoots the target batch by a full step.

Usage:
    uv run python scripts/scratch/validate_scale_ladder_sweeps.py
"""

import glob
import sys

import yaml

import src.globals
from src.models import qwen3_parameters_to_depths_and_widths

SWEEP_GLOB = "sweeps/pt_v1_scale_ladder/*.yaml"

V1_REQUIRED = ["warmup_steps", "weight_decay"]
V2_FORBIDDEN = ["adam_beta1", "adam_beta2", "warmup_ratio", "full_determinism"]


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


def main() -> int:
    paths = sorted(glob.glob(SWEEP_GLOB))
    if not paths:
        print(f"No configs matched {SWEEP_GLOB}")
        return 1

    failures = 0
    for path in paths:
        name = path.split("/")[-1]
        print(f"=== {name} ===")
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

        model_name = (
            data_params
            and params["model_config"]["parameters"]["model_name"]["values"][0]
        )
        size = model_name.split("-")[1]
        if size not in qwen3_parameters_to_depths_and_widths:
            print(f"  FAIL: size {size!r} not in qwen3_parameters_to_depths_and_widths")
            failures += 1
        else:
            depth, width = qwen3_parameters_to_depths_and_widths[size]
            print(f"  ok   {model_name} -> depth {depth}, width {width}")

        nominal = src.globals.MODEL_NAMES_TO_PARAMETERS_DICT.get(size)
        if nominal is None:
            print(
                f"  warn no nominal parameter count for {size!r}; skipping batch math"
            )
        else:
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
            print(f"  ..   {len(doses)} doses: {doses}")
        print()

    print(f"{len(paths)} configs checked, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
