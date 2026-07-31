# Handoff — NeurIPS 2026 rebuttal, submission 32216

> ⛔ **HISTORICAL RECORD — do not quote numbers from this file.** It is a point-in-time session artifact. Several figures in it were later corrected. The current numbers live in `REBUTTAL_DRAFT.md` and `REBUTTAL_EVIDENCE.md`.

Updated **2026-07-30 03:30**, end of overnight session. **All experiments complete.** Scores: **8RFz 3** (Quality 2, Originality 2,
conf 4) · **1wx9 4** · **aPBL 3**. The AC named 8RFz's loss-vs-correctness objection the pivotal
critique. **Discussion closes 2026-08-03.**

## Read in this order

1. **[`REBUTTAL_DRAFT.md`](REBUTTAL_DRAFT.md)** — paste-ready per-reviewer responses. Start here.
2. **[`RETRIEVAL_KEY_RESULT.md`](RETRIEVAL_KEY_RESULT.md)** — the strongest new result.
3. **[`CONTAMINANT_ABLATION.md`](CONTAMINANT_ABLATION.md)** — what the contaminant arms actually leak.
4. **[`PROTOCOL_CONFOUND.md`](PROTOCOL_CONFOUND.md)** — read before quoting any Math Verify number.
5. **[`REBUTTAL_EVIDENCE.md`](REBUTTAL_EVIDENCE.md)** — criticism → number map.
6. **[`MISSING_PRETRAINING_DATA.md`](MISSING_PRETRAINING_DATA.md)** · **[`HF_TOKEN_INCIDENT.md`](HF_TOKEN_INCIDENT.md)**

---

## ⚠️ Three things needing you

1. **`git push` is blocked** by the sandbox permission classifier. **53 commits** sit on branch
   `rebuttal/neurips-2026-protocol-and-evidence`, unpushed. Run:
   `git push -u origin rebuttal/neurips-2026-protocol-and-evidence`
2. **The HF token on this node is `ruili0`'s**, not yours, world-readable, write-scoped. Nothing
   of ours leaked (gated behind `PRETRAIN_SKIP_HUB_PUSH=1`; `ruili0` owns 0 `mem_*` models,
   `RylanSchaeffer` owns 196). **Export your own `HF_TOKEN` before any upload**, and tell
   Rui/brando9 to rotate theirs.
3. **The paraphrased/perturbed checkpoints are local only** — `models/pt_language_model/*_cont_*`
   — because of (2). Push them to `RylanSchaeffer/` when convenient.

---

## The single most important new result

**Memorization without retrieval — loss and accuracy come apart.** Full detail in
[`RETRIEVAL_KEY_RESULT.md`](RETRIEVAL_KEY_RESULT.md).

Models pretrained on *rephrased problems with verbatim original solutions*, evaluated 0-shot on
the **original** problems (Qwen3-34M, uncontaminated = loss 7.1437 / accuracy 0.00%):

| R | Loss, exact | Loss, rephrased | Acc, exact | **Acc, rephrased** | Verbatim solution rate |
|---|---|---|---|---|---|
| 32 | 2.5138 | 2.6125 | 0.56% | 0.24% | 0.000 |
| 100 | 1.4526 | 2.0077 | 1.70% | 1.58% | 0.000 |
| 316 | 0.5243 | 1.9573 | 7.22% | **1.52%** | **0.000** |

At R=316 the rephrased model's loss is 78% of the way from clean to fully-contaminated — heavily
contaminated by any loss measure — yet it scores 1.52% and reproduces the gold solution **0 times
in 5,000**. It holds the answer and cannot retrieve it.

**Mechanism: memorization is of the solution text; retrieval is keyed on the problem text.**
Rephrasing at training time stores the solution without the key; rephrasing at evaluation time
(Table 1) withholds the key from a model that has one. This unifies Finding 2 with the ablation.

**Why it wins the rebuttal:** it demonstrates 8RFz's W1 — the AC's pivotal critique — on our own
data. Conceding by showing beats arguing. And it inverts their detection worry: perplexity would
flag these models loudly while benchmark scores are barely inflated, a false-positive mode for
loss-based detection and probably the commoner real-world case.

---

## What changed tonight

### Protocol: 0-shot, settled on the merits

Testing the 4-shot rationale required removing **a scoring confound I introduced while
investigating**: the 0-shot and 4-shot sweeps straddle `db75c5f`, which changed the *scorer*
(lenient → boxed-required) as well as the prompt. All 76 runs rescored from raw W&B responses,
one scorer, no GPU.

R=0 is **exactly 0.0000 under both protocols at all five sizes.** 4-shot demonstrably teaches the
format (boxed rate 0 → 0.43–0.89) and buys **zero** accuracy. At 0-shot the boxed rate rises with
dose — the contaminated model learns the format from the injected solutions.

**The same confound was in three more places**, all now fixed:
- Table 1 replacement (nb18) and SFT re-run (nb19) — lenient baseline vs strict treatments.
- **Finding #4 (nb17)** — the AC's pivotal claim. A lenient `ot=1` denominator inflated apparent
  dilution. Corrected it holds: 93M retains **0.0188** at R=100 vs **0.9966** at R=1000 (~53×).
- Temperature response — the matched difference does not fully cancel the artifact, but the
  residual is small. Retention at τ=1.0 is **25%** under strict scoring (0.2528) against 0.2495
  lenient. A brief claim of **9.6%** here is **RETRACTED**: the rescoring script had silently
  dropped 344M and switched estimators. See `verification/TEMPERATURE_VERIFICATION.md`.

### Corrected headline numbers

| Quantity | Old | **Corrected** |
|---|---|---|
| Table 1 (R≥100, n=14) | 70.19% → 2.74% / 1.89% | **72.18% → 2.78% / 1.91%** |
| Uncontaminated floor | ~1% | **exactly 0.00%** |
| SFT (14 conditions) | 72.31% → 3.00% | **72.95% → 2.80%**, median retained 0.022 |
| Finding #4 retention | 0.019 / 0.995 | **0.0188 / 0.9966** |
| Temperature at τ=1.0 | 25% | **25%** (unchanged — the 9.6% was retracted, see `verification/TEMPERATURE_VERIFICATION.md`) |
| Notebook 16 | 14/17, −4.72 nats | **17/17, −2.18 nats** |
| 0-shot pass@k | (didn't exist; 4-shot only) | **pass@25 = 0 on all 2,500 problems** |

"Rephrased/perturbed land at 2–3× the floor" is **dead** — the floor is exactly 0.00%, so quote
the residual in points (+2.78 pp, +1.91 pp).

### The contaminant ablation

`math_rephrased` rephrases the problem but keeps the solution (**99.8% byte-identical**), so that
arm is *solution-verbatim* leakage, not paraphrase. Reporting it as paraphrase transfer would have
been wrong and trivially checkable. `math_perturbed` differs on both sides (0.1% identical).

| Arm | Problem | Solution | Loss transfer (R=32/100/316) |
|---|---|---|---|
| Exact | same | same | 1.000 |
| Rephrased | differs | same | 0.979 / 0.902 / 0.784 |
| Perturbed | differs | differs | 0.879 / 0.726 / 0.570 |

**Read loss transfer with care**: both modified corpora are MATH-domain text, so much of the
perturbed arm's transfer is domain adaptation, not leakage. Accuracy is the honest metric — see
the complete table below. A fourth arm with **disjoint** math problems would separate these; not
run.

### The lost W&B data

Re-searched by **exact run ID** with a **validated matcher** (positive control 1 hit, fabricated
ID 0 hits). **0 of 218 run IDs across 305 projects in 7 entities.** Identity confirmed correct.
Absence is targeted: `-eval` (1,565), `-sft` (135), `-eval-teacher-forcing` (107) all resolve;
`-pt` alone does not. Rename/move ruled out; who removed it is not established.

More survives locally than documented: the cache exists in notebooks **10, 11 and 20**, and
`notebooks/04_*/data/43bce56c...csv` is the sole copy of a 41-configuration subset-fraction arm.
All committed. Still worth doing: W&B web UI deleted-projects view, and emailing W&B support.

---

## Nothing is running. All experiments finished 03:26.

### The contaminant ablation, complete

Qwen3-34M, 1×OT. Uncontaminated: loss **7.1437**, accuracy **0.00%**.

| R | Loss exact | Loss reph | Loss pert | Acc exact | Acc reph | Acc pert |
|---|---|---|---|---|---|---|
| 32 | 2.5138 | 2.6125 | 3.0741 | 0.56% | 0.24% | 1.34% |
| 100 | 1.4526 | 2.0077 | 3.0113 | 1.70% | 1.58% | 1.16% |
| 316 | **0.5243** | 1.9573 | 3.3705 | **7.22%** | 1.52% | 1.60% |

**Only exact-replica contamination produces a dose-response in accuracy** (13× climb). Both arms
whose problem text differs plateau at ~1.5%, inside the ±0.33 pp bootstrap half-width, with a
verbatim solution rate of 0.000 throughout. Loss meanwhile calls the perturbed arm 57–88% as
contaminated as the exact one — the two metrics disagree about the same models.

⚠️ **Dosing caveat**, caught by the notebook's own token-budget assertion: `math_perturbed` is
1,132,643 tokens per copy vs the original's 1,446,312 (**21.7% smaller**), so perturbed R=316 is
exact R≈247 in contaminated tokens. The bias runs *against* the conclusion, so it is conservative;
do not over-read the R=316 loss uptick. See `CONTAMINANT_ABLATION.md`.

---

## Manuscript corrections still needed

1. Label Fig. 1 **0-shot**.
2. Drop the "~60× SFT collapse" (artifact) → **72.95% → 2.80%**.
3. `04_further_training.tex:64` "14/17, −4.72 nats" → **17/17, −2.18 nats**.
4. Table 1's printed 0.00–0.04% do not reproduce → use the 0-shot re-run.
5. State the **11.64%** perturbed answer-overlap exclusion.
6. Drop "collapses to baseline" → residual in percentage points.
7. SFT format confound is scale-dependent → report `sft_score_given_boxed`.
8. Notebook 11's `Num. Tokens = 20 × Num. Parameters` omits the overtrain multiplier.

## What is left

- **Post the rebuttal.** Draft complete but for the two perturbed rows.
- **`\citep` the five new bib keys** — `palavalli2024taxonomy`, `mehrbakhsh2024confounders`,
  `dekoninck2024evading`, `dekoninck2024constat`, `godey2025gaperon`. In the `.bib` (verified
  against ACL Anthology and arXiv) but **not yet cited**, and uncited entries never render.
  8RFz wrongly listed **Jiang et al. 2024 as uncited** — it is cited three times; the draft
  corrects this politely.
- **Manuscript `.tex` edits** — deliberately not started, per your instruction to hold.
- **P3.2** SFT hyperparameters appendix; **P3.3** rephrase/perturbation validation appendix.
