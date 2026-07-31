# Session handoff — 2026-07-30, for a fresh context

> ⛔ **HISTORICAL RECORD — do not quote numbers from this file.** It is a point-in-time session artifact. Several figures in it were later corrected. The current numbers live in `REBUTTAL_DRAFT.md` and `REBUTTAL_EVIDENCE.md`.

You are picking up the **NeurIPS 2026 rebuttal for submission 32216**. Discussion closes
**2026-08-03**. Scores: **8RFz 3** (Quality 2, Originality 2, conf 4) · **1wx9 4** · **aPBL 3**.
The AC named 8RFz's loss-vs-correctness objection the pivotal critique.

Everything is on branch `rebuttal/neurips-2026-protocol-and-evidence`, **pushed** (54 commits).
*[Update, later on 2026-07-30: that branch was fast-forward merged into `main` and deleted; all
work now lives on `main`.]*

---

## Your job, in order

Rylan's instruction, verbatim in intent:

1. **Review the rebuttals** — is the response actually good, and does it answer what was asked?
2. **Review the plan for responding**, including whether more experiments or analyses are needed.
3. **Critically review the implementations, methods, analyses, results, and how they are
   communicated.** Everything must be complete, correct and consistent across the paper.
4. **Confirm all documentation is updated.**

Treat (3) as the main event. The prior session found **five** errors in earlier work and **two**
in its own. That rate means more are likely. Be adversarial about your predecessor's work — it
was produced across a long overnight run and was motivated to find a good story.

---

## Read these first, in this order

| File | What it is |
|---|---|
| `REBUTTAL_DRAFT.md` | The deliverable. Paste-ready per-reviewer responses. No placeholders. |
| `VERIFICATION_HANDOFF.md` | **Ranked list of what is least verified and how to falsify it.** |
| `RETRIEVAL_KEY_RESULT.md` | The strongest new result. |
| `CONTAMINANT_ABLATION.md` | What the contaminant arms actually leak. Contains a dosing caveat. |
| `PROTOCOL_CONFOUND.md` | Read before quoting any Math Verify number. |
| `REBUTTAL_EVIDENCE.md` | Criticism → number map. |
| `HANDOFF.md` | State as of end of overnight run. |
| `MISSING_PRETRAINING_DATA.md`, `HF_TOKEN_INCIDENT.md` | Two incidents. |

Repo-level: `CLAUDE.md` (has two traps that already produced wrong numbers),
`docs/NOTEBOOK_DATA_SOURCES.md`, `docs/EXPERIMENT_INVENTORY.md`, `docs/INFRASTRUCTURE.md`.

---

## Three verification agents were launched 2026-07-30 ~03:45

They may still be running or may have finished. **Check for their reports before duplicating
work**, in `reviews/2026_neurips/verification/`:

| File | Verifying |
|---|---|
| `TEMPERATURE_VERIFICATION.md` | ✅ done — verdict **WRONG**. 9.6% retracted; τ=1.0 retention is **25%**. Corrections already propagated to `REBUTTAL_EVIDENCE.md`, `REBUTTAL_DRAFT.md`, `HANDOFF.md`. |
| `PERTURBED_INJECTION_VERIFICATION.md` | Whether `math_perturbed` is actually what gets injected |
| `NOTEBOOK_MERGE_VERIFICATION.md` | Notebooks 17/18/19 merge keys after a baseline-column swap |

Each was told to state CONFIRMED / WRONG / UNRESOLVED and to fix + commit if wrong. **If a report
says WRONG, propagating the correction through `reviews/2026_neurips/*.md` is your first task.**
If a report is missing, do that verification yourself — `VERIFICATION_HANDOFF.md` has the method.

---

## Environment — these will cost you an hour otherwise

```bash
cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
./mem_scoring_vs_sampling_env/bin/python          # ABSOLUTE path; it is a uv venv, not conda
export PYTHONPATH=$PWD
export HF_HOME=/lfs/skampere1/0/shared_hf_cache
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python     # required before `import wandb`
```

- **W&B auth** comes from `~/.netrc` (mode 600, yours). It is correct — entity `rylan`. The venv
  has nothing to do with it.
- **HF auth** comes from `$HF_HOME/token`, which is **mode 666 and owned by `brando9`, containing
  `ruili0`'s write-scoped token**. Any `push_to_hub()` lands in *their* namespace. Nothing of ours
  leaked (gated behind `PRETRAIN_SKIP_HUB_PUSH=1`). **Export a real `HF_TOKEN` before any upload.**
- Notebooks use relative paths: `cd` into the notebook directory before running.
- `math_verify.verify()` uses a **signal-based timeout**, so it raises outside the main thread.
  Any parallel scoring must use a **process** pool, and must not swallow exceptions — a bare
  `except: pass` silently zeroes every score. This already happened once.
- GPUs 2–5 belong to another user (`jchud`). 0, 1, 6, 7 were ours.

---

## The state of the science

### The protocol question is settled: 0-shot

The 4-shot switch was commit `db75c5f` (2026-03-29), self-initiated during the ICML rebuttal — no
reviewer asked. Its rationale was that uncontaminated models never see `\boxed{}` so 0-shot
conflates format with reasoning. That does not hold: 4-shot demonstrably teaches the format
(boxed rate 0 → 0.43–0.89) and buys **exactly zero** accuracy. R=0 is **0.0000 under both
protocols at all five sizes**.

Everything the rebuttal uses is 0-shot. Notebooks 13 and 15 are the 4-shot analyses, superseded by
19 and 18, and now carry docstrings saying so. No 4-shot figure is `\includegraphics`'d in the
paper (notebook 15's PDFs sit in `figures/` unused).

### Two traps that caused real errors — check for more instances

1. **Stale caches.** `src.analyze.download_wandb_project_runs_configs` hashes the sweep list into
   the cache filename; with `refresh=False` it never re-downloads. Notebook 11 declares 4-shot
   sweeps but serves a cache built from the commented-out 0-shot list. That is *why* Fig. 1 is
   0-shot. `docs/NOTEBOOK_DATA_SOURCES.md` audits every notebook for this.
2. **The scoring boundary.** `db75c5f` changed the prompt *and* the scorer (lenient
   `math_verify.parse()`, ~1.4% false positives → boxed-required). Four separate analyses compared
   across that boundary. Authoritative source is
   `notebooks/11_*/results/protocol_sensitivity_rescored.csv` (`strict_score`). **Never quote the
   0-shot column of `protocol_sensitivity.csv`.**

### Headline results

**Contaminant ablation** (Qwen3-34M, 1×OT; uncontaminated loss 7.1437, accuracy 0.00%):

| R | Loss exact | Loss reph | Loss pert | Acc exact | Acc reph | Acc pert |
|---|---|---|---|---|---|---|
| 32 | 2.5138 | 2.6125 | 3.0741 | 0.56% | 0.24% | 1.34% |
| 100 | 1.4526 | 2.0077 | 3.0113 | 1.70% | 1.58% | 1.16% |
| 316 | **0.5243** | 1.9573 | 3.3705 | **7.22%** | 1.52% | 1.60% |

Only exact-replica contamination produces a dose-response in accuracy (13×). Both arms whose
*problem text* differs plateau at ~1.5%.

**Retrieval-key result** — the strongest thing in the rebuttal. At R=316 the rephrased-contaminant
model has loss 78% of the way from clean to fully-contaminated, yet scores 1.52% (vs exact 7.22%)
and reproduces the gold solution verbatim **0/5000**. The positive control shows it memorizes
*exactly as strongly* as the exact arm on its own items (7.56% vs 7.22%, verbatim 5.34%). So:
**memorization is of the solution text; retrieval is keyed on the problem text.**

**Numbers corrected overnight** — *[Update, later on 2026-07-30: the posting checklist was
removed from `REBUTTAL_DRAFT.md` before posting; the criticism-to-number map is
`REBUTTAL_EVIDENCE.md`.]* Do not paste from older drafts or from `REBUTTAL_PLAN.md` /
`NEXT_STEPS.md`, which still contain pre-correction figures (flagged inline).

---

## Known-incomplete — candidates for your work

1. **The five new references are in `references_rylan.bib` but never `\citep`'d.** Uncited entries
   do not render, so the related-work fix (8RFz W3, a driver of Originality=2) is currently
   invisible. Keys: `palavalli2024taxonomy`, `mehrbakhsh2024confounders`, `dekoninck2024evading`,
   `dekoninck2024constat`, `godey2025gaperon`. Note 8RFz listed **Jiang et al. 2024 as uncited and
   it is not** — cited 3×, including an appendix paragraph.
2. **Manuscript `.tex` edits not started** — Rylan asked to hold. The 8 corrections are listed in
   `HANDOFF.md`. Confirm he still wants them held before starting.
3. **Fig. 1 still renders the leniently-scored PDF.** A strict-scored replacement exists
   (`notebooks/11_*/results/y=math_verify_strict_x=num_replicas_hue=params.pdf`) showing the true
   0.00% floor. Swapping the `\includegraphics` is a `.tex` edit — see (2).
4. **P3.2** SFT hyperparameters appendix (aPBL W4) and **P3.3** rephrase/perturbation validation
   appendix (aPBL Q2) — not written. Source material: `src/globals.py`, `sweeps/sft/`, and
   `reviews/2026_icml/REVIEWER_6RQA/`.
5. **OpenReview character limits unconfirmed** — the general response may need splitting.
6. **A disjoint-mathematics contaminant arm** would separate domain adaptation from item-level
   leakage in the ablation. Not run; stated as a limit.

---

## Judgement calls the prior session made — revisit if you disagree

- **Concede 8RFz's W1 by demonstrating it**, rather than defending Findings 4–5. The dissociation
  result makes the concession free, and Findings 4–5 still stand on accuracy.
- **Weaken Finding 4 rather than defend it** — dilution is threshold-dependent, and "overtraining
  dilutes contamination" is false in the heavy-leakage regime.
- **State outright that Table 1's printed values do not reproduce**, and replace them.
- **Report dose caveats that cut against us** — e.g. `math_perturbed` is 21.7% smaller per copy,
  so replicas are not an equal dose across arms.

---

## Two errors the prior session made in its own writing, as calibration

1. Claimed 4-shot "floors the uncontaminated baseline to zero" using numbers that confounded
   prompt format with scoring rule. The conclusion survived rescoring; the original evidence did not.
2. Predicted in writing that the perturbed arm's accuracy would sit at the 0.00% floor. It is
   1.34% at R=32 — *above* the exact arm — because the model picks up weak genuine competence from
   5,000 near-miss problems.

**The pattern in both: a mechanism that explained the data was extended past what it was tested
on.** Look for that same pattern elsewhere, including in the retrieval-key result.
