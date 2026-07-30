# The Missing Pretraining Data: All Established Facts

Investigated 2026-07-29. Everything below is measured, not inferred, unless labelled
**INFERENCE** or **UNVERIFIED**.

---

## What is missing

`notebooks/11_*` and `notebooks/20_*` both merge pretraining cross-entropy
(`eval_after/eval_benchmark_loss`) against Math Verify scores. This is the data behind:

- **Finding #3** — the irreducible-error result 8RFz called "striking and compelling"
- **Finding #1's** loss panel
- **Finding #4's** original cross-entropy version

They read it from **15 named sweep IDs** in project `memorization-scoring-vs-sampling-pt`:

```
rkx5xfde g31f7bsb ehxxzk5n              (34M)
u7dxxphm o6aoejzc 1nwyun1z xbiu535y     (62M)
ho49sshi x8gmmzlo u5xcf726              (93M)
sl086kx0                                (153M)
09c432gh gsx7gisg 6f9ah90l r9fixoce     (344M)
```

**All 15 are gone.** Searched every project of every entity the API key can see —
**325 projects across 9 entities** (`rylan`, `jkazdan`, `stellaathena`, plus teams
`harvardparkesateams`, `kreiman-sdm`, `fiete-lab`, `projectdeus`, `joshteam`, `brando-su`).
Zero of 15 found. `rylan/memorization-scoring-vs-sampling-pt` returns "Could not find project".

Reproduce: `python scripts/scratch/find_pretraining_sweeps.py`

## What survives, and where

The only copy is a local cache file:

```
notebooks/11_math_qwen3_pt_math_verify/data/c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv
```

626 KB. **Untracked by git.** Backed up 2026-07-28 to `~/irreplaceable_backups/` (AFS) and
`/dfs/scratch0/rschaef/irreplaceable_backups/`.

Contents: **228 rows, 177 distinct configurations, 181 rows carrying a benchmark loss.**
All `benchmark=EleutherAI/minerva_math`, all `benchmark_subset_fraction=1.0`. Spans:

| Field | Values present |
|---|---|
| Model size | 34M (40), 63M (42), 93M (43), 153M (8), 344M (52), unparseable (43) |
| Overtrain multiplier | 1 (55), 2 (41), 4 (41), 8 (42), 16 (49) |
| Replicas | 0, 1, 3, 10, 32, 100, 316, 1000, 3162 |

Note this cache covers the **overtrained** runs too, not only `ot=1` — so Finding #4's
cross-entropy version depends on it as well.

## How much is recoverable from W&B

Matching by *configuration* rather than sweep ID (a run for the same model/replicas/overtrain is
equally usable):

| | Count |
|---|---|
| Configurations in cache | **177** |
| Also live in W&B today | **22** — all in `rylan/memorization-scoring-vs-sampling-pt-v2` |
| **Cache-only (unrecoverable if the file is lost)** | **155** |

Cache-only spans the entire 63M ladder, the entire 153M ladder, and most of 344M.

Full per-configuration table: `reviews/2026_neurips/data/missing_pretraining_data_inventory.csv`
Reproduce: `python scripts/scratch/investigate_missing_pretraining_data.py`

## `jkazdan/memorization-scoring-vs-sampling-pt` is a different experiment, not a backup

It has 236 runs, 110 of which parse as pretraining configs, but **zero match any cached
configuration**. The reason is visible in its field distributions:

| Field | jkazdan | cache |
|---|---|---|
| `benchmark` | 148 `minerva_math`, 88 `fineweb-edu-dedup` | 228 `minerva_math` |
| `benchmark_subset_fraction` | mostly 0.1, 0.0001, 0.01, 0.001, 1e-05, 0.5 | **all 1.0** |
| `overtrain_multiplier` | 1 (210), 10 (16), 0.01 (10) | 1, 2, 4, 8, 16 |
| Model sizes | 34M, 93M, 48M | 34M, 63M, 93M, 153M, 344M |
| Runs with a benchmark loss | 39 of 236 | 181 of 228 |

It is a **subset-fraction dose-response sweep at small sizes**, with overtrain multipliers
(10, 0.01) that never appear in the main grid. 31 (size, replicas) pairs in the cache have no
counterpart there at all — including all of 63M and all of 153M.

Reproduce: `python scripts/scratch/compare_jkazdan_pretraining_configs.py`

## Timeline — the most informative fact

| Event | Timestamp |
|---|---|
| Cache file written | **2026-01-19 00:52** |
| `-pt-v2` earliest run | **2026-01-19 20:31** (≈20 h later) |
| `-pt-v2` latest run | 2026-01-22 14:49 |
| `jkazdan/...-pt` runs | 2025-09-06 → 2025-12-17 (older, unrelated) |

**INFERENCE:** the original project was lost or deleted around 2026-01-19, and `-pt-v2` was
started within a day as a replacement — but only 68 runs were ever redone, covering 22 of 177
configurations. A project *rename* would have carried all 177 across; it did not. The "-v2"
suffix is consistent with starting over.

Reproduce: `python scripts/scratch/check_ptv2_dates.py`

## The script question, resolved (supersedes two earlier wrong answers)

Investigated 2026-07-29 by ssh to skampere2 and by git archaeology. **skampere2 is reachable
with plain `ssh skampere2`** — an earlier session recorded it as unmounted and therefore
uncheckable, which was true of the filesystem but not of the host.

Two earlier claims were wrong and both are retracted:

1. *"This repo's `pretrain_language_model.py` diverged from what produced the published runs."*
   True in substance but stated as if unexplained.
2. *"This repo's pretraining path never produced the published runs; they came from
   `KoyejoLab-Pretraining-Variance` on skampere2."* **False.** That repo's pretraining script
   never builds a benchmark eval dataset at all — it passes a bare `eval_dataset` to the Trainer,
   so it cannot emit `eval_after/eval_benchmark_loss`. It is a different experiment.
   `docs/INFRASTRUCTURE.md` mapping *contamination* pretraining to that repo is an error.

**What actually happened**, from commit `934546a`, *"Add v2 pretraining configs with improved
optimizer settings"*:

| Event | Timestamp |
|---|---|
| notebook-11 cache written | 2026-01-19 **00:52** |
| commit `934546a` | 2026-01-19 **11:36** |
| first `-pt-v2` run | 2026-01-19 **20:31** |

`934546a` is the change. Its own message says it updated `TrainingArguments` to use `adam_beta1`,
`adam_beta2`, `warmup_ratio` and `full_determinism` (with `warmup_steps` commented out), reset
the optimizer defaults, renamed 62M→63M and 153M→165M, and **switched the W&B project to
`memorization-scoring-vs-sampling-pt-v2`**. All four "mystery" keys arrive in that one commit,
eleven hours after the cache was written and nine hours before the first `-pt-v2` run.

So: the published Fig. 3 runs are **v1 runs from this repo**, produced by the pre-`934546a`
script, and `-pt-v2` is a deliberate fresh start on the same day the config changed — which
independently confirms the "started over rather than renamed" inference above.

**Practical consequence: nothing needs to be guessed and nothing needs skampere2.**
`git show 934546a^:scripts/pretrain_language_model.py` and `git show 934546a^:src/globals.py`
recover the exact code and defaults behind the published runs. The full published
`trainer_config` is also recorded verbatim in the cache CSV (`warmup_steps: 250`,
`weight_decay: 0`, `logging_steps: 1`, `max_length: 2048`, betas left at HuggingFace defaults).

Two smaller corrections: it is **four** required keys, not five — this repo's script never
references `train_test_split_seed`. And the sweep YAML written for the paraphrased experiment
sets both `warmup_ratio: 0.0316` and `warmup_steps: 250`, but the current script reads
`warmup_ratio` and has `warmup_steps` commented out, so that run would silently not match the
published warmup schedule.

## Where the lost data could still be

skampere2 has been checked and is **eliminated**:

- `/lfs/skampere2/0/rschaef/wandb/` holds 6,444 run directories, but all are dated **2026-02
  through 2026-04** — after the 2026-01-19 loss — and **zero** contain `eval_benchmark_loss` in
  `wandb-summary.json`.
- No `trainer_state.json` exists anywhere under `/lfs/skampere2/0/rschaef`.
- `KoyejoLab-Pretraining-Variance` has an empty `wandb/` and only unrelated scaling-law caches.
- A second checkout of *this* repo exists there under a transposed name,
  `/lfs/skampere2/0/rschaef/KoyejoLab-Memorization-Scoring-vs-Sampling` (its script *does* log
  the benchmark loss), but it has **no `wandb/` directory and no notebook-11 cache**.

Remaining leads, ranked:

1. **W&B account-level trash** — deleted projects are sometimes restorable through support
   within a retention window. Not exposed through the public API. Worth an email.
2. **Other coauthors' accounts** — Stella Biderman ran the Table 1 experiments on TensorPool;
   Joshua Kazdan owns the SFT checkpoints. Neither account was visible beyond
   `jkazdan`/`stellaathena` (the latter shows 1 project, no matching runs).

The 626 KB CSV should be treated as the sole surviving copy.

## What is at risk if the cache is lost

- **Finding #3** cannot be refit — the irreducible-error result, which is the paper's strongest.
- **Finding #1's** loss panel and **Finding #4's** cross-entropy version lose their source.
- Only 22 of 177 configurations could be re-derived from W&B; the remaining 155 would require
  **retraining 155 models**.

The Math Verify (accuracy) side is unaffected — those runs live in
`memorization-scoring-vs-sampling-eval`, which is intact, plus everything generated this session.

## Immediate recommendation

Commit the cache file to git. It is 626 KB of CSV, so Git LFS is not involved and the exhausted
LFS budget is irrelevant. It is currently one `rm` away from making a headline finding
unreproducible.
