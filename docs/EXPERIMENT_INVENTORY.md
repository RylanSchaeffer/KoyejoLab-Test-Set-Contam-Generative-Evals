# Experiment Inventory

**What exists, what has been evaluated, and — most usefully — what has not.**

Verified 2026-07-27 directly against the HuggingFace Hub API and the W&B API. Not derived from any
other document in this repo. Re-verify with `scripts/audit_inventory.py`.

> **Updated 2026-07-30.** Several "not run" entries below have since been run. See
> [What has been run since](#what-has-been-run-since-2026-07-27) at the bottom before trusting any
> gap listed here.

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

The last row was the important one, and it has since been closed: **all 137 overtrained checkpoints
were evaluated in Math Verify space on 2026-07-29** (notebook 17). Finding #4 no longer rests on
cross-entropy alone. Accuracy tracks loss, and the finding sharpens to a threshold effect — at 93M
over ot 1×–16×, R=100 retains 0.0188 of its advantage while R=1000 retains 0.9966.

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

> ⚠️ **Corrected 2026-07-27. The previous version of this section was wrong in a way that
> invalidates cross-notebook comparisons.** It claimed notebooks 11 and 13 both use 4-shot and are
> directly comparable. They are not. See [`reviews/2026_neurips/PROTOCOL_CONFOUND.md`](../reviews/2026_neurips/PROTOCOL_CONFOUND.md).

Two protocols exist, and **the protocol changes the measured contamination effect by up to ~190x**:

| Protocol | Peak Math Verify over the whole pretrained grid (greedy) |
|---|---|
| 0-shot | **1.0000** |
| 4-shot | **0.0112** |

Under 0-shot the prompt reproduces the opening of the memorized training document and contaminated
models regurgitate the stored solution verbatim. The 4-shot prefix moves the prompt off that
memorized context and the same checkpoints fall to the uncontaminated floor. Prompts are ~687 tokens
at the median against a 2,048-token pretraining sequence length, so this is not context overflow, and
the 4-shot prompt is the standard well-formed Minerva format. It is a real effect, not a bug.

Which notebook reads which:

| Notebook | Declares | Actually reads | Protocol |
|---|---|---|---|
| `notebooks/11_*` (pretrained) | 4-shot sweep IDs | **0-shot cache** — `refresh=False` kept the stale file | **0-shot** |
| `notebooks/13_*` (SFT) | 4-shot sweep IDs | 4-shot | 4-shot |
| `notebooks/15_*` (rephrase/perturb) | 4-shot sweep IDs | 4-shot | 4-shot |

The notebook-11 cache is `678b1e19c88ea5fdaf60b14abccdb09e_*`, which is
`md5("sweeps=" + ",".join(<old 0-shot list>))`. Editing the sweep list without deleting the cache or
setting `refresh=True` silently keeps the old data — check the hash, not the source line.

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


---

## What has been run since 2026-07-27

| Study | Scale | Where | Outcome |
|---|---|---|---|
| Finding #4 in accuracy space | 137/137 overtrained checkpoints | `notebooks/17_*` | Accuracy tracks loss; dilution is threshold-dependent (93M: 0.0188 at R=100 vs 0.9966 at R=1000 over ot 1×–16×) |
| Table 1 at 0-shot | 39 + 39 checkpoints, 5 sizes | `notebooks/18_*` | Original 72.18% → Rephrased 2.78% → Perturbed 1.91%; uncontaminated floor exactly 0.00% |
| Finding #5 at 0-shot | 39/39 SFT checkpoints | `notebooks/19_*` | 70.89% → 3.00%, median retained 0.028 |
| Protocol rescore | all 76 protocol runs | `scripts/rescore_zeroshot_with_boxed_required.py` | Both protocols under one scorer; R=0 is exactly 0.0000 everywhere |
| Temperature rescore | 369/370 temperature runs | `scripts/rescore_temperature_response.py` | Advantage retained at τ=1.0 is 9.6%, not the 25% previously reported |
| **Contaminant ablation** | 34M × R ∈ {32,100,316} × 2 arms | sweeps `mxamktp0`, `vrxwx4dz` | Rephrased (solution-verbatim) transfer 0.979 / 0.902 / 0.784; perturbed arm running |
| 0-shot pass@k | uncontaminated 344M | `results/pass_at_k/.../0shot/` | 0 well-formed `\boxed{}` in >30,000 samples so far |

**Still not run**, and worth stating plainly:

- A second benchmark (GSM8K) contamination sweep. `notebooks/00_gsm8k_platinum/` is the starting point.
- A second model family (Llama/Gemma-style).
- Multiple seeds at any configuration. All error bars to date are test-set bootstrap, not seed variance.
- A contaminant arm using **disjoint** math problems, which is what would separate domain adaptation
  from item-level leakage in the ablation above.
