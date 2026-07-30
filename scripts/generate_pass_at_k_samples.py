"""Generate N samples per MATH problem for pass@k estimation using vLLM.

This script generates multiple stochastic completions for each problem in
the Hendrycks MATH test set, saving raw completions to disk in JSONL format.
It is designed to be resumable and interruptible: on startup it reads any
existing samples, counts how many each problem already has, and only generates
the remaining samples needed to reach --target_n per problem.

Generation and scoring are intentionally decoupled. This script handles only
GPU-intensive generation; scoring is done separately by score_pass_at_k.py.

Usage:
    python scripts/generate_pass_at_k_samples.py \
        --model_name "RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1" \
        --temperature 1.0 \
        --target_n 1000 \
        --max_tokens 2048 \
        --output_dir results/pass_at_k \
        --batch_n 50

Output:
    {output_dir}/{model_short_name}/temp={temperature}/samples.jsonl

    Each line is a JSON object:
    {"problem_idx": int, "sample_idx": int, "response_text": str,
     "level": str, "type": str, "problem": str}
"""

import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

# This is needed for deterministic to work.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 16.48 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large
# try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch

# Compiling seems to be causing problems down the line :/
torch.compiler.disable()
from vllm import LLM, SamplingParams

import src.data

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate N samples per MATH problem for pass@k estimation."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. RylanSchaeffer/mem_Qwen3-344M_...)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--target_n",
        type=int,
        required=True,
        help="Target number of samples per problem",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens per completion (default: 2048)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=4,
        choices=[0, 4],
        help=(
            "Prompt protocol. 4 (default) reproduces the original run. Use 0 to measure "
            "capability under the same protocol as Fig. 1 and the teacher-forced results; a "
            "4-shot pass@k cannot support a claim about 0-shot capability, because the prefix "
            "changes the conditioning context rather than merely demonstrating output format."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pass_at_k",
        help="Root output directory (default: results/pass_at_k)",
    )
    parser.add_argument(
        "--batch_n",
        type=int,
        default=50,
        help="Number of samples per vLLM call per problem (default: 50)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="First problem index to process (inclusive). Default: 0.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Last problem index to process (exclusive). Default: n_problems.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.device_count() > 0, "No CUDA devices available."
    print(f"CUDA VISIBLE DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # 1. Load MATH test set (same as eval_language_model.py).
    raw_datasets = src.data.load_dataset_hendrycks_math()
    test_dataset = raw_datasets["test"]
    doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT
    fewshot_prefix = src.data.build_fewshot_prefix() if args.num_fewshot else ""
    print(
        f"Prompt protocol: {args.num_fewshot}-shot "
        f"(prefix is {len(fewshot_prefix)} chars)"
    )
    formatted_problems = [
        fewshot_prefix + doc_to_text.format(problem=q, solution="").rstrip()
        for q in test_dataset["problem"]
    ]
    n_problems = len(formatted_problems)
    print(f"Loaded {n_problems} MATH test problems.")

    # 2. Determine problem index range.
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else n_problems
    start_idx = max(0, start_idx)
    end_idx = min(n_problems, end_idx)
    print(f"Processing problems [{start_idx}, {end_idx}) ({end_idx - start_idx} problems).")

    # 3. Determine output path.
    model_short_name = args.model_name.split("/")[-1]
    # Keep the 4-shot path exactly as it was so the existing run is not clobbered, and give
    # 0-shot its own directory. Protocol is part of the identity of a pass@k measurement.
    base_dir = Path(args.output_dir) / model_short_name / f"temp={args.temperature}"
    if args.num_fewshot != 4:
        base_dir = base_dir / f"{args.num_fewshot}shot"
    # Use a shard-specific filename when processing a subset.
    if args.start_idx is not None or args.end_idx is not None:
        filename = f"samples_shard_{start_idx}_{end_idx}.jsonl"
    else:
        filename = "samples.jsonl"
    output_path = base_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. Count existing samples per problem (for resumability).
    existing_counts = Counter()  # problem_idx -> count
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                existing_counts[record["problem_idx"]] += 1
        if existing_counts:
            print(
                f"Found existing samples. "
                f"Min per problem: {min(existing_counts.values())}, "
                f"Max: {max(existing_counts.values())}"
            )
        else:
            print("Found existing file but no valid samples.")

    # 5. Compute how many samples each problem still needs.
    remaining = {
        i: max(0, args.target_n - existing_counts.get(i, 0))
        for i in range(start_idx, end_idx)
    }
    total_remaining = sum(remaining.values())
    problems_needing_samples = sum(1 for v in remaining.values() if v > 0)

    if total_remaining == 0:
        print(
            f"All {n_problems} problems already have {args.target_n} samples. "
            f"Nothing to do."
        )
        return

    print(
        f"Need to generate {total_remaining} more samples "
        f"across {problems_needing_samples} problems."
    )

    # 5. Load vLLM model.
    model = LLM(
        model=args.model_name,
        dtype="bfloat16",
        enforce_eager=False,
    )

    # 7. Generate in batches, flush after each problem.
    n_in_range = end_idx - start_idx
    with open(output_path, "a") as f_out:
        for problem_idx in range(start_idx, end_idx):
            n_needed = remaining[problem_idx]
            if n_needed == 0:
                total_samples = existing_counts.get(problem_idx, 0)
                print(
                    f"Problem {problem_idx + 1 - start_idx}/{n_in_range} "
                    f"(idx={problem_idx}): "
                    f"{total_samples} total samples (generated 0 new)"
                )
                continue

            sample_idx_start = existing_counts.get(problem_idx, 0)
            n_generated_for_problem = 0

            # Generate in sub-batches of batch_n.
            while n_generated_for_problem < n_needed:
                batch_size = min(args.batch_n, n_needed - n_generated_for_problem)
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    n=batch_size,
                    # No seed -- we want diverse samples. Different runs
                    # naturally produce different samples.
                )
                outputs = model.generate(
                    prompts=[formatted_problems[problem_idx]],
                    sampling_params=sampling_params,
                )
                # outputs is a list of length 1 (one prompt);
                # outputs[0].outputs has batch_size completions.
                for j, completion in enumerate(outputs[0].outputs):
                    record = {
                        "problem_idx": problem_idx,
                        "sample_idx": sample_idx_start + n_generated_for_problem + j,
                        "response_text": completion.text,
                        "level": test_dataset["level"][problem_idx],
                        "type": test_dataset["type"][problem_idx],
                        "problem": test_dataset["problem"][problem_idx],
                    }
                    f_out.write(json.dumps(record) + "\n")

                n_generated_for_problem += batch_size

            f_out.flush()
            total_samples = sample_idx_start + n_needed
            print(
                f"Problem {problem_idx + 1 - start_idx}/{n_in_range} "
                f"(idx={problem_idx}): "
                f"{total_samples} total samples (generated {n_needed} new)"
            )

    print(f"Done. Samples saved to {output_path}")


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    main()
    logging.info("Finished generate_pass_at_k_samples.py!")
