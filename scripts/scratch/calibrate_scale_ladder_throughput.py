"""Checklist item 1.1: measure memory and throughput for the new Qwen3 scale-ladder sizes.

The Phase 1 batch sizes in sweeps/pt_v1_scale_ladder/ were chosen to fit 80 GB and to
make v1's math.ceil gradient-accumulation rounding overshoot by only 1-2%. Neither
property has been measured. This script measures both cheaply, before weeks of GPU
time are committed on the strength of arithmetic.

Synthetic data on purpose: the question is "does this batch fit, and how fast does a
step run", which random token IDs answer exactly. Building the real contaminated
corpus would cost far more and change nothing about the answer.

Reports, per size:
  - peak allocated / reserved GPU memory at the configured batch size
  - measured tokens/sec for one device
  - projected wall-clock for the full dose grid in that size's sweep config, at the
    14.3 tokens/parameter the legacy budget actually delivers

Runs on ONE GPU. Multi-GPU throughput is estimated by scaling, which is optimistic:
DDP gradient all-reduce is not free. Treat projections as lower bounds on time.

Usage:
    CUDA_VISIBLE_DEVICES=6 uv run python scripts/scratch/calibrate_scale_ladder_throughput.py
    CUDA_VISIBLE_DEVICES=6 uv run python scripts/scratch/calibrate_scale_ladder_throughput.py --sizes 499M
"""

import argparse
import glob
import os
import time

# Without this, CUDA_VISIBLE_DEVICES indexes GPUs in CUDA's own order, which need
# not match nvidia-smi's PCI order -- so "GPU 6" can silently land on a device that
# is already busy, and the resulting OOM looks like the batch size being too large.
# The eval scripts set this for the same reason.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import yaml

import src.models

# The published runs delivered 0.7136-0.7141 of the nominal 20 tok/param budget, and
# the new ladder reproduces that via PRETRAIN_LEGACY_TOKEN_BUDGET=1. Projecting with
# the nominal 20 would overstate the cost by 40%.
LEGACY_BUDGET_FRACTION = 0.7138
NOMINAL_TOKENS_PER_PARAM = 20


def load_sweep_configs(sizes_filter):
    configs = {}
    for path in sorted(glob.glob("sweeps/pt_v1_scale_ladder/*.yaml")):
        config = yaml.safe_load(open(path))
        params = config["parameters"]
        model_name = params["model_config"]["parameters"]["model_name"]["values"][0]
        size = model_name.split("-")[1]
        if sizes_filter and size not in sizes_filter:
            continue
        trainer = params["trainer_config"]["parameters"]
        world = int(
            [c for c in config["command"] if str(c).startswith("--nproc_per_node")][
                0
            ].split("=")[1]
        )
        configs[size] = {
            "model_name": model_name,
            "batch": trainer["per_device_train_batch_size"]["values"][0],
            "max_length": trainer["max_length"]["values"][0],
            "grad_ckpt": trainer["gradient_checkpointing"]["values"][0],
            "world_size": world,
            "doses": params["data_config"]["parameters"][
                "num_benchmark_replicas_per_epoch"
            ]["values"],
            "path": path.split("/")[-1],
        }
    return configs


def calibrate(size: str, spec: dict, num_steps: int, warmup_steps: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = src.models.create_causalm_for_pretraining(
        model_config_dict={
            "model_name": spec["model_name"],
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        }
    )
    model = model.cuda()
    if spec["grad_ckpt"]:
        model.gradient_checkpointing_enable()
    model.train()

    num_parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    batch, max_length = spec["batch"], spec["max_length"]
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (batch, max_length), device="cuda", dtype=torch.long
    )

    step_times = []
    for step in range(warmup_steps + num_steps):
        torch.cuda.synchronize()
        start = time.time()
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        if step >= warmup_steps:
            step_times.append(elapsed)

    mean_step = sum(step_times) / len(step_times)
    tokens_per_step = batch * max_length
    tokens_per_sec = tokens_per_step / mean_step

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    del model, optimizer, input_ids
    torch.cuda.empty_cache()

    return {
        "num_parameters": num_parameters,
        "mean_step_s": mean_step,
        "tokens_per_sec_1gpu": tokens_per_sec,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", default=None)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override per_device_train_batch_size from the config, for tuning.",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "No CUDA device visible."
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_mem, _ = torch.cuda.mem_get_info()
    print(
        f"Device: {device_name} ({total_mem:.0f} GB total, {free_mem / 1e9:.0f} GB free)"
    )
    print(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}"
    )
    if free_mem / total_mem < 0.9:
        print(
            "  WARNING: this device is already in use. Measurements will understate "
            "the batch size that fits on an idle GPU."
        )
    print()

    configs = load_sweep_configs(set(args.sizes) if args.sizes else None)
    if not configs:
        print("No matching sweep configs found.")
        return

    results = {}
    for size, spec in configs.items():
        if args.batch is not None:
            spec = {**spec, "batch": args.batch}
        print(f"=== {size} ({spec['path']}) ===")
        print(
            f"  batch={spec['batch']} max_length={spec['max_length']} "
            f"grad_ckpt={spec['grad_ckpt']} nproc={spec['world_size']}"
        )
        # The dominant allocation at this vocabulary size is the logits tensor, not
        # the model. 151,936 vocab x 2,048 positions x 4 bytes is ~1.2 GB per
        # sequence, and cross-entropy materialises a second copy. Batch size here is
        # bounded by vocabulary, not by parameter count -- which is why 1.44B at
        # batch 11 fits while 499M at batch 22 does not.
        logits_gb = spec["batch"] * spec["max_length"] * 151936 * 4 / 1e9
        print(
            f"  predicted logits: {logits_gb:.1f} GB (x2 for the cross-entropy copy "
            f"= {2 * logits_gb:.1f} GB)"
        )
        try:
            measured = calibrate(size, spec, args.num_steps, args.warmup_steps)
        except torch.cuda.OutOfMemoryError:
            print(
                f"  OOM at batch {spec['batch']} -- LOWER IT and re-check ceil rounding\n"
            )
            torch.cuda.empty_cache()
            continue
        results[size] = (spec, measured)

        headroom = total_mem - measured["peak_reserved_gb"]
        print(
            f"  parameters      : {measured['num_parameters'] / 1e6:.1f}M "
            f"(name says {size})"
        )
        print(
            f"  peak memory     : {measured['peak_alloc_gb']:.1f} GB allocated, "
            f"{measured['peak_reserved_gb']:.1f} GB reserved, "
            f"{headroom:.1f} GB headroom"
        )
        print(
            f"  step time       : {measured['mean_step_s'] * 1e3:.0f} ms  "
            f"({measured['tokens_per_sec_1gpu'] / 1e3:.1f}k tokens/s on 1 GPU)"
        )

        # Projection at the token budget the legacy flag actually delivers.
        tokens_per_run = (
            NOMINAL_TOKENS_PER_PARAM
            * measured["num_parameters"]
            * LEGACY_BUDGET_FRACTION
        )
        agg_tokens_per_sec = measured["tokens_per_sec_1gpu"] * spec["world_size"]
        hours_per_run = tokens_per_run / agg_tokens_per_sec / 3600
        n_doses = len(spec["doses"])
        print(
            f"  per run         : {tokens_per_run / 1e9:.1f}B tokens -> "
            f"{hours_per_run:.1f} h on {spec['world_size']} GPUs"
        )
        print(
            f"  full dose grid  : {n_doses} doses -> {hours_per_run * n_doses:.1f} GPU-group-hours "
            f"({hours_per_run * n_doses / 24:.1f} days sequential)"
        )
        print()

    if results:
        total_hours = 0.0
        print("=== Phase 1 total (sequential, one size at a time) ===")
        for size, (spec, measured) in results.items():
            tokens_per_run = (
                NOMINAL_TOKENS_PER_PARAM
                * measured["num_parameters"]
                * LEGACY_BUDGET_FRACTION
            )
            hours = (
                tokens_per_run
                / (measured["tokens_per_sec_1gpu"] * spec["world_size"])
                / 3600
                * len(spec["doses"])
            )
            total_hours += hours
            print(f"  {size:>6}: {hours:6.1f} h  ({len(spec['doses'])} doses)")
        print(f"  {'TOTAL':>6}: {total_hours:6.1f} h = {total_hours / 24:.1f} days")
        print(
            "\nNote: single-GPU throughput scaled by nproc, which ignores DDP all-reduce"
            "\noverhead and torch_compile speedup. Treat as an optimistic lower bound."
        )


if __name__ == "__main__":
    main()
