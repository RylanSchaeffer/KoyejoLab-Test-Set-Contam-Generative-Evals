# Experiment Inventory

**What exists, what has been evaluated, and — most usefully — what has not.**

Verified 2026-07-27 directly against the HuggingFace Hub API and the W&B API. Not derived from any
other document in this repo. Re-verify with `scripts/audit_inventory.py`.

---

## TL;DR

| | Count |
|---|---|
| Hub models matching `mem_Qwen3` | 468 |
| ...following the contamination naming convention | **377** |
| ...the other 91 | `scale_mem_*`, a separate project — ignore here |
| Convention-matching, overtrained (`ot > 1`) | **138** |
| Convention-matching, SFT'd (`_sft`) | **39** |
| Finished generative eval runs | **1,270** |
| ...against overtrained checkpoints | **0** |

The last row is the important one. Every finished generative evaluation is against an `ot=1`
(compute-optimal) or `ot=1_sft` checkpoint. **Finding #4 (overtraining dilutes contamination) has never
been measured in accuracy space** — only in cross-entropy. The checkpoints to fix that already exist, so
it is an inference-only job.

---

## Pretrained checkpoints (`benchmark_subset_fraction = 1.0`)

### Compute-optimal (`ot = 1`) — 115 checkpoints

| Model | Replicas available |
|---|---|
| Qwen3-34M | 0, 1, 3, 10, 18, 32, 56, 100, 178, 316 |
| Qwen3-48M | 0, 1, 3, 10, 18, 32, 56, 100, 178, 316, 562 |
| Qwen3-62M | 0, 1, 3, 10, 32, 100, 316 |
| Qwen3-63M | 0, 1, 3, 10, 18, 32, 56, 100, 178, 316, 562 |
| Qwen3-93M | 0, 1, 3, 10, 18, 32, 56, 100, 178, 316, 562, 1000 |
| Qwen3-153M | 0, 1, 3, 10, 32, 100, 316, 1000 |
| Qwen3-165M | 0, 1, 3, 10, 18, 32, 56, 316, 562, 1000, 1778 |
| Qwen3-344M | 0, 1, 3, 10, 32, 100, 316, 1000, 3162 |
| Qwen3-660M | 0, 1 |

Note the two near-duplicate size pairs (62M/63M, 153M/165M) from different sweep generations. The paper
uses 34M / 62M / 93M / 153M / 344M.

### Overtrained (`ot > 1`) — 138 checkpoints, **all unevaluated generatively**

| Model | ot=2 | ot=4 | ot=8 | ot=16 |
|---|---|---|---|---|
| Qwen3-34M | 0…316 | 0…1000 | 0…3162 | 0…3162 |
| Qwen3-62M | 0…1000 | 0…3162 | 0…3162 | 0…3162 |
| Qwen3-93M | 0…1000 | 0…3162 | 0…3162 | 0…3162 |
| Qwen3-153M | 0, 100 | 0, 100 | — | — |
| Qwen3-344M | 0…3162 | 0…3162 | 0…3162 | 0, 1, 3, 32 |

Replica ladder is 0, 1, 3, 10, 32, 100, 316, 1000, 3162 (truncated per cell as shown).

### SFT'd (`_sft`) — 39 checkpoints

All at `ot=1`, `sbst=1.0`, SFT on the MATH **train** split.

| Model | Replicas |
|---|---|
| Qwen3-34M | 0, 1, 3, 10, 32, 100, 316 |
| Qwen3-62M | 0, 1, 3, 10, 32, 100, 316 |
| Qwen3-93M | 0, 1, 3, 10, 32, 100, 316, 1000 |
| Qwen3-153M | 0, 1, 3, 10, 32, 100, 316, 1000 |
| Qwen3-344M | 0, 1, 3, 10, 32, 100, 316, 1000, 3162 |

### Subset-fraction sweeps (`sbst < 1.0`) — 85 checkpoints

Dose-response study varying what fraction of the test set is contaminated, at
`sbst ∈ {0.001, 0.01, 0.0316, 0.1, 0.3162, 0.5}` for 34M / 48M / 62M. Consumed by `notebooks/30_*`.
Beware: some names use 4-decimal formatting (`0.0100`) and others 3 (`0.010`) — **parse as float, never
string-match**.

---

## Evaluation coverage

Finished runs in `memorization-scoring-vs-sampling-eval`, grouped by checkpoint type and eval dataset:

| Checkpoint type | Dataset | Runs |
|---|---|---|
| `ot=1` | `EleutherAI/minerva_math` | 898 |
| `ot=1` | `RylanSchaeffer/math_perturbed` | 9 |
| `ot=1` | `RylanSchaeffer/math_rephrased` | 9 |
| `ot=1_sft` | `EleutherAI/minerva_math` | **351** |
| `ot ∈ {2,4,8,16}` | anything | **0** |

Perturbed/rephrased coverage is **344M only** (9 replica levels each), from sweeps `mprek7pj` (original),
`w8j3qnru` (perturbed), `25xeednq` (rephrased) — all read by `notebooks/15_*`.

> ⚠️ **Table 1 in `03_pretraining.tex` also reports 34M and 93M columns, which no sweep in the eval
> project produces.** Reviewer 8RFz's Q4 is literally "How are the values in Table 1 calculated?", so this
> must be resolved before answering: trace those numbers to a superseded sweep (possibly 0-shot, or the
> retired `stellaathena/*` datasets), re-run the two sizes against `RylanSchaeffer/math_{rephrased,perturbed}`
> (~16 runs, under two GPU-hours), or drop the columns.

### Evaluation protocol

Current results are **4-shot with a required `\boxed{}` answer**. Earlier 0-shot sweeps exist and are
commented out in the notebooks. Do not mix the two. Both `notebooks/11_*` (pretrained) and
`notebooks/13_*` (SFT) use the 4-shot protocol, so they are directly comparable.

Per-problem `math_verify_score` values are logged to W&B run history, so **bootstrap confidence intervals
over the 5,000 test problems require no new compute**.

Measured cost, over 401 sampled finished runs:

| Model | Median wallclock | p90 |
|---|---|---|
| 34M | 5.7 min | 14.8 min |
| 62M | 2.8 min | 17.5 min |
| 93M | 3.8 min | 19.0 min |
| 153M | 5.8 min | 15.9 min |
| 344M | 5.4 min | 20.5 min |

vLLM startup dominates. `eval_language_model.py` loads the model once per run and evaluates one
temperature; looping temperatures inside a single load would cut a 3-temperature sweep roughly 3×.

---

## Results that exist but are not in the manuscript

| Result | Where it lives | Status |
|---|---|---|
| Math Verify after SFT (flat ~1–2% across all contamination levels) | sweep `2zpwcnek`, `notebooks/13_*/results/` | Never folded in; manuscript uses only the loss version from `notebooks/12_*` |
| pass@k = 0.000% over 808,797 samples, uncontaminated 344M | `reviews/2026_icml/REBUTTALS.md:238` | Zero mentions of `pass@k` in `manuscript_neurips_2026/*.tex`. **Scored outputs not present on the local workstation** — locate on skampere1 |
| SFT lowers NLL on perturbed MATH at 14/17 conditions | sweep `onaspopu`, `notebooks/16_*` | Cited as one prose clause at `04_further_training.tex:64`; no figure |
| Rephrased/perturbed dataset QC + 4 spot-check batches | `reviews/2026_icml/REVIEWER_6RQA/` | Not in any appendix |

---

## Known gaps and traps

- **`memorization-scoring-vs-sampling-pt` does not exist on W&B** despite 16 references in the repo.
  Pretraining data is under `memorization-scoring-vs-sampling-pt-v2`. `notebooks/10_*` cannot refresh.
- **No intermediate checkpoints.** Sweeps use `save_strategy: no` with `hub_strategy: end`, so only final
  weights exist. Any training-dynamics analysis requires retraining.
- **Rephrased/perturbed data cannot currently be used as a pretraining contaminant.**
  `create_dataset_for_pretraining` routes through `create_dataset_for_supervised_finetuning`
  (`src/data.py:442`), which only branches on `EleutherAI/minerva_math` and `madrylab/gsm8k-platinum`.
  `load_dataset_math_rephrased()` exists but is never dispatched to. Wiring it up is ~1 hour of work and
  is the prerequisite for the paraphrased-contamination experiment reviewers asked for.
- **GSM8K is supported for SFT/eval but never exercised for contaminated pretraining.** Loaders and
  templates exist in `src/data.py`; budget debugging time.
- **`_sft` suffix parsing.** Model IDs put `_sft` after the `ot` field, so a naive
  `ot_([\d.]+)$` regex silently drops all SFT models. Capture the suffix.
