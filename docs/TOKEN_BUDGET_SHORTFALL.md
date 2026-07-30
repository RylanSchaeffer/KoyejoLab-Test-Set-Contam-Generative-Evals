# The published runs trained on 14.3 tokens/parameter, not 20

Discovered and verified 2026-07-30.

## How much does this matter? Very little. Read this before spending time on it.

**It is a one-line methods correction, not a problem with the results.** The paper says 20
tokens/parameter; the true figure is 14.3. The shortfall is uniform across every model size and
overtrain multiplier (0.7136–0.7141, spread ±0.0005), so it is a constant factor: every comparison
in the paper is unaffected, the overtrain multipliers stay exact in relative terms, and the
scaling-law fits never used the nominal budget in the first place. **No reported number changes.**

The one non-uniform consequence — total tokens rise 27% from R=0 to R=316 — is likewise inert: the
loss drop it would have to explain is 7.14 → 0.52, and the perturbed contaminant arm carries the
same token inflation with no dose-response in loss or accuracy.

So: fix the sentences, keep the assertion that now prevents a recurrence, and move on. The detailed
verification below exists because the finding was challenged twice during review and the evidence
had to be nailed down — **not** because the finding is consequential. Do not re-escalate it.

The rest of this document is the audit trail.

---

## What happened

`create_dataset_for_pretraining` implements the intended algorithm correctly: compute the target
token budget, compute the tokens consumed by `R` replicas of the benchmark, subtract, and fill the
remainder from the corpus. The subtraction (`src/data.py`, `corpus_tokens_needed_per_epoch`) is
right. **The fill is what failed.**

```python
avg_tokens_per_doc = 220e9 / 190168005            # 1157
estimated_docs_needed = int(1.05 * corpus_tokens_needed_per_epoch / avg_tokens_per_doc)
idx_to_keep = np.searchsorted(cumulative_lengths, corpus_tokens_needed_per_epoch)
corpus_train_dataset_subset = corpus_train_dataset_subset.select(range(idx_to_keep))
```

`1157` is fineweb-edu-dedup's **advertised** mean document length — 220B tokens over 190,168,005
rows, as measured by its authors, with their tokenizer, untruncated. Under **our** tokenizer,
truncated at `max_length = 2048`, the realised mean is **~786**. Measured directly on 4,000
documents: mean **766.5**, 95% CI [748.5, 784.5], median 582, 9.1% truncated at the cap. The
figure implied by the published run logs is 786.2.

So the estimate overshoots by ~47%, and the `1.05` headroom — comment: *"Round up a bit to ensure
we have more than we want"* — cannot absorb it. The sampled pool therefore holds only
`1.05 × 786 / 1157 ≈ 0.714` of the tokens requested.

Then `np.searchsorted(cumulative_lengths, target)` is asked for the cut point at a target that
**exceeds every cumulative sum**, so it returns `len(cumulative_lengths)`, and
`select(range(idx_to_keep))` keeps **every document**. The trim meant to hit the budget exactly
became a silent no-op — no exception, no warning, and the log line printed the tokens *requested*
rather than the tokens *delivered*, which is why this survived the whole project.

## The evidence

Four independent routes agree.

**1. The R=0 run alone.** With no contaminant at all, the 34M run logged **486,122,537** tokens
against a **681,237,120** target — 71.4%. Contamination cannot explain a shortfall in a run that
has none. (`train/num_input_tokens_seen` and `eval_after/num_input_tokens_seen` are identical, so
it is a terminal value; `train/epoch = 1.0` with `max_steps = -1` means one complete pass, so the
number *is* the dataset size.)

**2. The target recovered without assuming the 20N rule.**

| Quantity | Value |
|---|---|
| Sequences processed: `1636 steps × 9 grad-accum × 42 batch` | 618,408 |
| Documents sampled: `int(1.05 × 681,237,120 / 1156.87)` | 618,304 |
| Difference | 104 — under one batch (378) |
| Target recovered: `618,304 × 1156.87 / 1.05` | **681,236,623** vs `20N` = 681,237,120 |

Agreement to 497 tokens in 681 million. `world_size = 1` is forced by the step count, not assumed:
618,304 / 378 = 1635.7 → the 1,636 steps observed, where `world_size = 2` predicts 818.

**3. Step counts match a no-trim model exactly at every dose.** Predicting documents as
`R × 5000 + int(1.05 × (681,237,120 − R × 1,446,312) / 1156.87)` and dividing by 378:

| R | 0 | 1 | 3 | 10 | 32 | 100 | 316 |
|---|---|---|---|---|---|---|---|
| steps predicted | 1,636 | 1,646 | 1,665 | 1,734 | 1,948 | 2,612 | 4,719 |
| steps **observed** | 1,636 | 1,646 | 1,665 | 1,734 | 1,948 | 2,612 | 4,719 |
| tokens observed (M) | 486.1 | 486.6 | 487.4 | 490.4 | 499.3 | 527.4 | 617.2 |
| tokens/sequence | 786.2 | 782.3 | 774.4 | 748.5 | 678.2 | 534.4 | 346.0 |

Had the trim worked, fewer corpus documents would survive and every predicted step count would be
lower. Step count is logged independently of the token counter.

**4. Token counts fit the no-trim model to 0.024%.** With
`total(R) = f · 20N + (1 − f) · R · 1,446,312` and `f` fixed at its R=0 value, all seven doses
predict within 0.024%.

### Padding is not involved, and here is why that question comes up

HF's `num_input_tokens_seen` is `inputs[main_input_name].numel()`, which **does** include padding,
and the Trainer pads dynamically to the longest sequence in each batch — so this is a reasonable
thing to suspect. It does not apply here: `scripts/pretrain_language_model.py` uses
**`DataCollatorWithFlattening`**, which concatenates a mini-batch into one `[1, total_tokens]`
sequence and adds no padding. Verified empirically — collating inputs of length 10, 3 and 7 yields
shape `(1, 20)` and `numel() == 20`, where padding to the batch max would give 30. `group_by_length`
is `False` and is irrelevant under this collator.

Route 3 above is independent of this question anyway: tokens/sequence falling to **346** at R=316
is impossible under padding, since a padded batch maximum cannot fall below the longest document in
the batch (~1,500–2,048 with ~11% corpus documents present).

## What it does and does not affect

**Unaffected — the shortfall is uniform.** `delivered / (20 · m · N)` is **0.7136–0.7141** across
all five model sizes and all five overtrain multipliers, a spread of ±0.0005:

| | m=1 | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| 34M | 0.7136 | 0.7138 | 0.7138 | 0.7139 | 0.7138 |
| 62M | 0.7137 | 0.7138 | 0.7140 | 0.7138 | 0.7139 |
| 93M | 0.7138 | 0.7138 | 0.7138 | 0.7138 | 0.7139 |
| 153M | 0.7139 | — | — | — | — |
| 344M | 0.7139 | 0.7138 | 0.7140 | 0.7141 | 0.7139 |

Because it is a constant factor: every comparison across model size is untouched; the overtrain
multipliers `m` remain exact in *relative* terms; and the scaling-law fits are unaffected because
`src/analyze.py` uses *measured* `num_input_tokens_seen`, not the nominal budget.

**Affected — two claims in the paper.**

| Location | Claim | Reality |
|---|---|---|
| `02_methodology.tex:8` | "Each model was pretrained on 20 tokens-per-parameter" | **14.3** |
| `01_introduction.tex:28` (Fig. 1 caption) | "We pretrained compute-optimal language models" | 0.71× compute-optimal |
| `04_further_training.tex:12,17` | `D(m,N) ≐ m × 20 × N`; "m=1 compute-optimal training" | `m × 14.3 × N` |
| `99_appendix.tex:74` | "total training tokens remain constant across contamination levels, isolating the effect of contamination from the effect of additional data" | Not constant: **+27%** from R=0 to R=316 |

The appendix line is the one that matters most: unlike the others it is a claim about
**experimental validity** rather than a label, and it is false.

**The dose–compute confound has an empirical control we already ran.** Because the contaminant is
delivered in full while the corpus is short, total tokens rise 27% with dose. The perturbed
contaminant arm carries the same token inflation with dose and shows **no dose-response in either
loss or accuracy** (`reviews/2026_neurips/CONTAMINANT_ABLATION.md`), which bounds what the extra
27% can buy at close to nothing. The loss drop 7.14 → 0.52 is in any case far beyond what 27% more
tokens could produce.

## What changed in the code (2026-07-30)

- `CORPUS_MEAN_TOKENS_PER_DOC = 786.0` replaces the advertised 1157, and
  `CORPUS_SAMPLING_HEADROOM = 1.25` replaces 1.05.
- **A hard assertion** that the sampled pool covers the target. This is the important change: the
  failure mode was silence, and correctness no longer depends on the estimate being right.
- The log line now reports tokens **delivered** against tokens **requested**, plus how many sampled
  documents were kept — the original printed only the request.
- `PRETRAIN_LEGACY_TOKEN_BUDGET=1` restores the old constants and skips the assertion, so the
  published runs remain reproducible bit for bit. Do not use it for new experiments.
- The false docstring on `create_dataset_for_pretraining` now states what actually happened.

**New runs will not match published runs.** Anything trained after this change receives its full
nominal budget; every checkpoint on the Hub does not. Use the legacy flag to reproduce.

## Reproducing this analysis

Data: `notebooks/11_math_qwen3_pt_math_verify/data/c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv`
(the sole surviving copy of the pretraining sweeps — see `MISSING_PRETRAINING_DATA.md`). Columns
`train/num_input_tokens_seen`, `train/global_step`, `gradient_accumulation_steps`,
`per_device_train_batch_size`, `model/num_parameters`, and `trainer_config.overtrain_multiplier`.
The target is `20 × overtrain_multiplier × model/num_parameters`
(`scripts/pretrain_language_model.py`, `compute_derived_hyperparameters`).
