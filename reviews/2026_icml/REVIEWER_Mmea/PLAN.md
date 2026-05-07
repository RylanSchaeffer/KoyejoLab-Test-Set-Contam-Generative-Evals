# Experiment Plan: pass@k on MATH with Uncontaminated Models

## ⚠️ STATUS UPDATE (2026-05-06)

**ICML 2026 was rejected; this work is being resubmitted to NeurIPS 2026.** This plan is still active for the NeurIPS submission. Current status:

| Component | Status | Reference |
|---|---|---|
| Boxed-only scoring integrated | DONE | commit `893f29d`, `src/scoring.py`, 94 tests in `tests/test_boxed_scoring.py` |
| 4-shot prompting integrated | DONE | commit `db75c5f`, `MINERVA_MATH_FEWSHOT_EXAMPLES` + `build_fewshot_prefix()` in `src/data.py` |
| Generation script | EXISTS | `scripts/generate_pass_at_k_samples.py` |
| Scoring script | EXISTS | `scripts/score_pass_at_k.py` |
| Launch wrappers | EXIST | `scripts/run_pass_at_k.sh`, `scripts/launch_pass_at_k_shards.sh`, `scripts/monitor_and_score_pass_at_k.sh` |
| **Phase 1 (N=1000 on uncontaminated 344M) end-to-end run** | **NOT YET RUN** | Single biggest remaining experimental item for the Mmea response |
| Phases 2–4 scaling | NOT YET RUN | Conditional on Phase 1 |

The plan below is preserved verbatim; "What to Run" and the phased compute schedule still apply, but should now be executed against the corrected scorer/prompt rather than the original ICML-era pipeline.

## Motivation

Reviewer Mmea's central objection is that 344M models "fundamentally lack the capacity to generalize or perform multi-step reasoning on competition-level mathematics," making contamination findings trivially about pure memorization. Running pass@k with large k can reveal whether uncontaminated models have any latent capability on MATH, even if pass@1 is ~0%.

- **If pass@k shows signal:** Directly refutes Mmea's premise. Even faint capability means the contamination findings are about amplifying vs. creating capability.
- **If pass@k is zero:** Supports the reframe that clean isolation of memorization is a feature of our design, not a flaw.

## What to Run

### Models

| Model | HuggingFace ID | Purpose |
|-------|---------------|---------|
| 344M (R=0) | `RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1` | Primary: largest uncontaminated model |
| 153M (R=0) | `RylanSchaeffer/mem_Qwen3-153M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1` | Optional: check if capability scales with size |
| 344M (R=100) | `RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_100_sbst_1.0000_epch_1_ot_1` | Optional: contaminated comparison |

### Parameters

- **Temperature:** 1.0 (maximizes diversity for pass@k)
- **max_tokens:** 2048 (existing default)
- **Benchmark:** Full MATH test set (5000 problems)

### Compute Scaling Strategy

We don't know what N (total samples per problem) is sufficient. Start small and scale up:

| Phase | N (samples/problem) | Total completions | Est. GPU-hours (A100) | When to run |
|-------|---------------------|-------------------|----------------------|-------------|
| 1 | 1,000 | 5M | ~35h | First |
| 2 | 10,000 | 50M | ~350h | If Phase 1 shows no signal |
| 3 | 50,000 | 250M | ~1750h | Only if Phase 2 is ambiguous |

pass@k is computed post-hoc from the accumulated samples, so all generation from prior phases is reused.

## Architecture: Two Decoupled Scripts

### Why two scripts instead of modifying `eval_language_model.py`

The existing eval pipeline (`eval_language_model.py`) is a single-pass workflow: load model → generate 1 sample per problem → score → log to W&B. It is not resumable, not interruptible, and couples generation to scoring. The pass@k workflow needs:

1. **Resumable generation** — generate N=1000, inspect results, then resume to N=10000 without re-generating the first 1000
2. **Decoupled scoring** — generation is the GPU bottleneck; scoring with math-verify is cheap and should be re-runnable independently

This is a fundamentally different workflow, so we create two new standalone scripts. No existing scripts are modified.

---

### Script 1: `scripts/generate_pass_at_k_samples.py`

**Purpose:** Generate N samples per problem using vLLM and save raw completions to disk. Resumable and interruptible.

**CLI interface:**
```bash
python scripts/generate_pass_at_k_samples.py \
    --model_name "RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1" \
    --temperature 1.0 \
    --target_n 1000 \
    --max_tokens 2048 \
    --output_dir results/pass_at_k \
    --batch_n 50
```

**Arguments:**

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--model_name` | str | required | HuggingFace model ID |
| `--temperature` | float | 1.0 | Sampling temperature |
| `--target_n` | int | required | Target number of samples per problem |
| `--max_tokens` | int | 2048 | Max tokens per completion |
| `--output_dir` | str | `results/pass_at_k` | Root output directory |
| `--batch_n` | int | 50 | Number of samples per vLLM call per problem (controls GPU memory) |

**Output file:**
```
{output_dir}/{model_short_name}/temp={temperature}/samples.jsonl
```

Where `model_short_name` is derived from the HF model ID by taking the part after the last `/` (e.g., `mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1`).

**JSONL format** — one line per sample:
```json
{"problem_idx": 0, "sample_idx": 0, "response_text": "We begin by...", "level": "Level 1", "type": "Algebra", "problem": "Find the value of..."}
```

**Implementation pseudocode:**

```python
import argparse
import json
import os
from collections import Counter
from pathlib import Path

# Same env var setup as eval_language_model.py (OMP_NUM_THREADS, CUDA, etc.)

from vllm import LLM, SamplingParams
import src.data


def main():
    args = parse_args()

    # 1. Load MATH test set (same as eval_language_model.py)
    raw_datasets = src.data.load_dataset_hendrycks_math()
    test_dataset = raw_datasets["test"]
    doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT
    formatted_problems = [
        doc_to_text.format(problem=q, solution="").rstrip()
        for q in test_dataset["problem"]
    ]
    n_problems = len(formatted_problems)

    # 2. Determine output path
    model_short_name = args.model_name.split("/")[-1]
    output_path = Path(args.output_dir) / model_short_name / f"temp={args.temperature}" / "samples.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Count existing samples per problem (for resumability)
    existing_counts = Counter()  # problem_idx -> count
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                record = json.loads(line)
                existing_counts[record["problem_idx"]] += 1
        print(f"Found existing samples. Min per problem: {min(existing_counts.values()) if existing_counts else 0}, "
              f"Max: {max(existing_counts.values()) if existing_counts else 0}")

    # 4. Compute how many samples each problem still needs
    remaining = {i: max(0, args.target_n - existing_counts.get(i, 0)) for i in range(n_problems)}
    total_remaining = sum(remaining.values())
    if total_remaining == 0:
        print(f"All {n_problems} problems already have {args.target_n} samples. Nothing to do.")
        return
    print(f"Need to generate {total_remaining} more samples across {sum(1 for v in remaining.values() if v > 0)} problems.")

    # 5. Load vLLM model
    model = LLM(
        model=args.model_name,
        dtype="bfloat16",
        enforce_eager=True,
    )

    # 6. Generate in batches, flush after each problem
    with open(output_path, "a") as f_out:  # append mode
        for problem_idx in range(n_problems):
            n_needed = remaining[problem_idx]
            if n_needed == 0:
                continue

            sample_idx_start = existing_counts.get(problem_idx, 0)
            n_generated_for_problem = 0

            # Generate in sub-batches of batch_n
            while n_generated_for_problem < n_needed:
                batch_size = min(args.batch_n, n_needed - n_generated_for_problem)
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    n=batch_size,
                    # No seed — we want diverse samples. Different runs
                    # naturally produce different samples.
                )
                outputs = model.generate(
                    prompts=[formatted_problems[problem_idx]],
                    sampling_params=sampling_params,
                )
                # outputs is a list of length 1 (one prompt); outputs[0].outputs has batch_size completions
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

            f_out.flush()  # flush after each problem for interruptibility
            print(f"Problem {problem_idx + 1}/{n_problems}: "
                  f"{sample_idx_start + n_needed} total samples (generated {n_needed} new)")

    print(f"Done. Samples saved to {output_path}")
```

**Key design decisions:**
- **One prompt at a time to vLLM** with `n=batch_n` — this generates `batch_n` diverse completions for the same prompt in one call, which is the most efficient way to use vLLM for pass@k (batching across prompts with n>1 could OOM)
- **Append mode** — never overwrites existing samples
- **Flush after each problem** — if the process is killed, all completed problems are safe on disk
- **No seed** — we want maximum diversity across samples. Different runs of the script naturally produce different samples.
- **No scoring** — keeps the script simple and GPU-focused

**Performance note:** Generating 50 samples per problem per vLLM call means vLLM processes them as a single batch, which is much more efficient than 50 separate calls. The `batch_n` parameter lets us tune GPU memory usage.

**Alternative: batch across problems for throughput.** The pseudocode above goes problem-by-problem for simplicity and resumability. A faster variant could submit all problems that need more samples at once with `n=batch_n`, process the batch, then repeat. This is more complex for resumability (need to track per-problem counts within a batch) but would better utilize vLLM's continuous batching. Consider implementing this optimization if Phase 1 is too slow.

---

### Script 2: `scripts/score_pass_at_k.py`

**Purpose:** Read saved samples from disk, score each with math-verify, compute pass@k, and output results stratified by difficulty level and subject.

**CLI interface:**
```bash
python scripts/score_pass_at_k.py \
    --samples_path results/pass_at_k/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1/temp=1.0/samples.jsonl \
    --k_values 1 10 100 1000 \
    --output_dir results/pass_at_k/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1/temp=1.0/
```

**Arguments:**

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--samples_path` | str | required | Path to samples.jsonl |
| `--k_values` | int[] | `[1, 10, 100]` | k values for pass@k |
| `--output_dir` | str | same dir as samples | Where to write results |

**Implementation pseudocode:**

```python
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from math_verify import parse, verify

import src.data


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.
    n = total samples for this problem
    c = number of correct samples
    k = k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    args = parse_args()

    # 1. Load MATH ground-truth solutions for scoring
    raw_datasets = src.data.load_dataset_hendrycks_math()
    test_dataset = raw_datasets["test"]
    ground_truth_solutions = test_dataset["solution"]

    # 2. Read all samples from JSONL
    samples_by_problem = defaultdict(list)  # problem_idx -> [response_text, ...]
    metadata_by_problem = {}  # problem_idx -> {"level": ..., "type": ...}

    with open(args.samples_path, "r") as f:
        for line in f:
            record = json.loads(line)
            pid = record["problem_idx"]
            samples_by_problem[pid].append(record["response_text"])
            if pid not in metadata_by_problem:
                metadata_by_problem[pid] = {
                    "level": record["level"],
                    "type": record["type"],
                }

    print(f"Loaded samples for {len(samples_by_problem)} problems.")

    # 3. Check for cached scores; score only unscored samples
    scores_path = Path(args.samples_path).with_name("scores.jsonl")
    cached_scores = {}  # (problem_idx, sample_idx) -> bool
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                record = json.loads(line)
                cached_scores[(record["problem_idx"], record["sample_idx"])] = record["correct"]
        print(f"Loaded {len(cached_scores)} cached scores.")

    # Score uncached samples and append to cache
    n_new_scores = 0
    scores_by_problem = defaultdict(list)  # problem_idx -> [bool, ...]

    with open(scores_path, "a") as f_scores:
        for pid in sorted(samples_by_problem.keys()):
            gold_parsed = parse(ground_truth_solutions[pid])
            for sample_idx, response_text in enumerate(samples_by_problem[pid]):
                cache_key = (pid, sample_idx)
                if cache_key in cached_scores:
                    correct = cached_scores[cache_key]
                else:
                    correct = bool(verify(gold=gold_parsed, target=parse(response_text)))
                    f_scores.write(json.dumps({
                        "problem_idx": pid,
                        "sample_idx": sample_idx,
                        "correct": correct,
                    }) + "\n")
                    n_new_scores += 1
                scores_by_problem[pid].append(correct)

    print(f"Scored {n_new_scores} new samples. Total scored: {n_new_scores + len(cached_scores)}")

    # 4. Compute pass@k per problem
    rows = []
    for pid in sorted(samples_by_problem.keys()):
        n_total = len(scores_by_problem[pid])
        n_correct = sum(scores_by_problem[pid])
        row = {
            "problem_idx": pid,
            "level": metadata_by_problem[pid]["level"],
            "type": metadata_by_problem[pid]["type"],
            "n_total": n_total,
            "n_correct": n_correct,
        }
        for k in args.k_values:
            if k <= n_total:
                row[f"pass@{k}"] = pass_at_k(n_total, n_correct, k)
            else:
                row[f"pass@{k}"] = None  # insufficient samples
        rows.append(row)

    df = pd.DataFrame(rows)

    # 5. Save per-problem results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results.csv", index=False)

    # 6. Print and save summary table stratified by level
    k_cols = [f"pass@{k}" for k in args.k_values if f"pass@{k}" in df.columns]
    summary = df.groupby("level").agg(
        n_problems=("problem_idx", "count"),
        n_total_samples=("n_total", "sum"),
        n_total_correct=("n_correct", "sum"),
        **{col: (col, "mean") for col in k_cols},
    ).reset_index()

    # Add overall row
    overall = pd.DataFrame([{
        "level": "Overall",
        "n_problems": len(df),
        "n_total_samples": df["n_total"].sum(),
        "n_total_correct": df["n_correct"].sum(),
        **{col: df[col].mean() for col in k_cols},
    }])
    summary = pd.concat([summary, overall], ignore_index=True)

    print("\n=== Results by Level ===")
    print(summary.to_markdown(index=False))

    # Also by type
    summary_type = df.groupby("type").agg(
        n_problems=("problem_idx", "count"),
        n_total_samples=("n_total", "sum"),
        n_total_correct=("n_correct", "sum"),
        **{col: (col, "mean") for col in k_cols},
    ).reset_index()

    print("\n=== Results by Type ===")
    print(summary_type.to_markdown(index=False))

    # Save markdown summary
    with open(output_dir / "summary.md", "w") as f:
        f.write("# pass@k Results\n\n")
        f.write(f"**Samples file:** `{args.samples_path}`\n\n")
        f.write("## By Difficulty Level\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## By Subject Type\n\n")
        f.write(summary_type.to_markdown(index=False))
        f.write("\n")

    print(f"\nResults saved to {output_dir}")
```

**Key design decisions:**
- **Score caching** via `scores.jsonl` — if you generate 1000 more samples and re-run scoring, only the new 1000 get scored. Previous scores are loaded from cache.
- **Uses the same math-verify scoring as `eval_language_model.py`** — `parse()` then `verify()` with gold and target
- **Stratified output** by both difficulty level and subject type
- **Markdown summary** for easy inclusion in the rebuttal

---

## Workflow

```
Phase 1:
  python scripts/generate_pass_at_k_samples.py \
      --model_name RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1 \
      --temperature 1.0 --target_n 1000 --batch_n 50
  # ~35 GPU-hours on A100. Can Ctrl+C and resume.

  python scripts/score_pass_at_k.py \
      --samples_path results/pass_at_k/mem_Qwen3-344M_.../temp=1.0/samples.jsonl \
      --k_values 1 10 100
  # Seconds on CPU. Inspect summary.md — any signal on Level 1-2?

Phase 2 (if needed):
  python scripts/generate_pass_at_k_samples.py \
      --model_name RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1 \
      --temperature 1.0 --target_n 10000 --batch_n 50
  # Resumes from 1000, generates 9000 more per problem.

  python scripts/score_pass_at_k.py \
      --samples_path results/pass_at_k/mem_Qwen3-344M_.../temp=1.0/samples.jsonl \
      --k_values 1 10 100 1000
  # Only scores the 9000 new samples (1000 cached). Inspect again.

Phase 3 (if needed):
  # Same pattern with --target_n 50000.
```

## Expected Output

**Per-problem CSV** (`results.csv`):

| problem_idx | level | type | n_total | n_correct | pass@1 | pass@10 | pass@100 |
|------------|-------|------|---------|-----------|--------|---------|----------|
| 0 | Level 1 | Algebra | 1000 | 3 | 0.003 | 0.030 | 0.260 |
| 1 | Level 3 | Geometry | 1000 | 0 | 0.000 | 0.000 | 0.000 |

**Summary table** (`summary.md`):

| Level   | n_problems | n_total_samples | n_total_correct | pass@1 | pass@10 | pass@100 |
|---------|-----------|----------------|-----------------|--------|---------|----------|
| Level 1 | 437       | 437000         | ...             | ...    | ...     | ...      |
| Level 2 | 894       | 894000         | ...             | ...    | ...     | ...      |
| ...     |           |                |                 |        |         |          |
| Overall | 5000      | 5000000        | ...             | ...    | ...     | ...      |

## What Success Looks Like

| Outcome | Interpretation | Rebuttal Impact |
|---------|---------------|-----------------|
| pass@100 > 0 on Level 1–2 | 344M has faint but real capability | Strong: refutes "fundamentally lacks capacity" |
| pass@100 = 0 everywhere | Pure memorization regime | Supports reframe: clean isolation is a feature |
| pass@100 > 0 only with contamination | Memorization unlocks otherwise-impossible outputs | Novel finding about contamination mechanism |

## Code Changes Summary

**Principle:** Modify existing scripts when backwards-compatible. Create new scripts when the workflow is fundamentally different.

### New scripts (workflow is fundamentally different from existing eval pipeline)

| File | Purpose | Reason for new script |
|------|---------|----------------------|
| `scripts/generate_pass_at_k_samples.py` | GPU generation (resumable, interruptible, JSONL) | Existing eval script is single-pass generate+score+log-to-W&B. Resumable accumulation with decoupled scoring is a different workflow. |
| `scripts/score_pass_at_k.py` | CPU scoring/analysis | Reads from disk, computes pass@k — no analog in existing scripts. |

### Backwards-compatible modifications to existing scripts (optional, independent of pass@k)

| File | Change | Backwards-compatible? |
|------|--------|----------------------|
| `scripts/eval_language_model.py` | Log `level` and `type` metadata per problem from the MATH dataset | Yes — additive W&B logging, no change to existing behavior |
