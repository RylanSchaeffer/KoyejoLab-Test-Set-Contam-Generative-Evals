# Verification handoff — check the overnight work adversarially

Written 2026-07-30 by the session that produced the results, *for a fresh session that did not*.
The point is to find errors, not to confirm the work. Assume the author was tired and motivated to
find a good story.

Branch `rebuttal/neurips-2026-protocol-and-evidence`, 53 commits, pushed.

## How to approach this

The overnight session found **five** substantive errors in prior work and **two** in its own. That
rate suggests more remain. The highest-value thing you can do is re-derive a headline number by a
different route than the one used to produce it, rather than re-reading the code that produced it.

**Do not trust this document's own claims.** Every number below cites the artifact it came from;
go to the artifact.

---

## Tier 1 — claims the rebuttal leans on hardest. Check these first.

### 1.1 The temperature table came from parsing a log, not a CSV ⚠️ WEAKEST LINK

`scripts/rescore_temperature_response.py` uses a `ProcessPoolExecutor`; one worker hung inside
`math_verify`'s signal timeout and blocked `ex.map()` from returning, so the results CSV was never
written by the script. The published table was **reconstructed by regex-parsing the per-run lines
already printed to `logs/rescore_temp.log`** (369 of 370 runs; one condition missing).

This changed a quoted number: retention at τ=1.0 went from 25% to **9.6%**.

**Check:** re-run the script with a per-row exception cap so it completes, or independently
recompute the advantage table from `notebooks/11_*/results/temperature_response_rescored.csv`.
Confirm (a) the parsed CSV matches a clean run, (b) the missing condition cannot move the mean,
(c) the greedy-advantage denominator is computed per (model, R) and not pooled.

Claim to falsify: *fraction of greedy advantage retained at τ=1.0 is 9.6%.*

### 1.2 The `math_perturbed` injection was never verified at the token level

`scripts/scratch/verify_paraphrased_contaminant.py` (previous session) decoded tokenized examples
to confirm the **rephrased** contaminant path injects rephrased problems into training while
measuring loss on originals. **The perturbed arm inherited that assumption and was never checked
the same way.**

**Check:** run the equivalent verification for `contaminant=RylanSchaeffer/math_perturbed`. Decode
a sample of the training split and the benchmark split and confirm the training split carries
perturbed items and the benchmark split carries originals, with no crossover.

Claim to falsify: *the perturbed arm's `eval_benchmark_loss` is measured on original MATH while
perturbed items are what entered the corpus.*

If this is wrong, the entire perturbed arm (and the "only verbatim leakage has a dose-response"
conclusion) is invalid.

### 1.3 Notebooks 17, 18, 19 after the baseline column was swapped

All three had their baseline column changed from `math_verify_score` (lenient) to `strict_score`
in `protocol_sensitivity_rescored.csv`, then were re-run. Outputs looked sane, but the **merge
keys were not re-verified row-for-row**.

**Check:** for each notebook, confirm the baseline merge joins on the same (Parameters, Num.
Replicas) pairs it did before and produces the same row count; confirm no silent row loss from the
rename. Notebook 17 is the important one — its retained fraction is `score(ot=16)/score(ot=1)`, so
a misaligned denominator would be invisible and wrong.

Claims to falsify: *Table 1 is 72.18% → 2.78% / 1.91% (n=14)*; *SFT is 70.89% → 3.00%*;
*93M retains 0.0188 at R=100 and 0.9966 at R=1000.*

### 1.4 The retrieval-key result

`RETRIEVAL_KEY_RESULT.md`. Two things to check independently:

- **Is the eval genuinely 0-shot?** `scripts/eval_contaminant_checkpoints_zeroshot.py` builds
  prompts from `src.data.MINERVA_MATH_DOC_TO_TEXT` with no prefix. Confirm by printing a prompt.
- **Is `verbatim_solution_rate` measuring what it claims?** It is an exact substring test of the
  gold solution inside the response. Confirm the 5.34% on rephrased problems are genuine full
  copies, and spot-read a few of the 0.000% cases on original problems to confirm the model really
  is not producing the solution in near-verbatim form (a paraphrase would evade the substring test
  and would weaken the "cannot retrieve" claim).

Claim to falsify: *at R=316 the rephrased model reproduces the gold solution verbatim 5.34% of the
time on its own problems and 0.000% on the originals.*

---

## Tier 2 — supporting claims

### 2.1 The rescoring itself

`scripts/rescore_zeroshot_with_boxed_required.py`. It has a positive control (a known-good run
returns 1 hit), a negative control (fabricated id returns 0), an exception counter, and an
assertion that 0-shot runs with logged>0.5 still score high under strict rules (max deviation
0.0016). This is the best-defended piece of the night, but it underpins everything else.

**Check:** independently rescore 3–5 runs with your own code and compare.

Claim to falsify: *uncontaminated R=0 is exactly 0.0000 under both protocols at all five sizes.*

### 2.2 The contaminant ablation's comparability to Fig. 3

Two automated guards passed: `gradient_accumulation_steps == 9` (matching all 12 published 34M
ot=1 runs) and training tokens within 0.03–0.81% of published for the rephrased arm.

**But** the perturbed arm at R=316 is **−4.56%** on tokens, and `math_perturbed` is 21.8% smaller
per copy, so replicas are not an equal dose across arms.

**Check:** confirm the token-size measurement (1,127,643 vs 1,441,312 per copy) and decide whether
the cross-arm comparison should be re-expressed in contaminated tokens rather than replicas.

Claim to falsify: *the paraphrased runs differ from the published exact-replica runs in exactly one
variable.* (Strictly, they do not — dose in tokens differs too.)

### 2.3 The lost W&B runs

`MISSING_PRETRAINING_DATA.md`. Searched by exact run ID with a validated matcher: 0 of 218 across
305 projects in 7 entities.

**Check:** the matcher validation is in the doc; re-run `scripts/scratch/hunt_lost_pretraining_runs.py`.
Note it can only see entities the API key can reach — a team Rylan has left would be invisible.
The W&B *web UI* deleted-projects view has not been checked and the public API does not expose it.

### 2.4 Numbers not re-derived overnight

These were inherited from the previous session and spot-checked but not recomputed:
- Bootstrap CIs (median half-width 0.33 pp) — and note they are computed from **leniently
  scored** runs; flagged in `BOOTSTRAP_CIS.md`, argued not to matter for half-widths. Verify that
  argument.
- Irreducible error E(0)=3.5942 [3.5359, 3.6639], 33/35 — read from
  `IRREDUCIBLE_ERROR_ROBUSTNESS.md`, not refit.
- The 11.64% perturbed answer-overlap mask — recomputed (582/5000) but the mask file itself was
  inherited.

---

## Known-incomplete work (not errors — just not done)

1. **The five new references are in `references_rylan.bib` but never `\citep`'d.** An uncited
   entry does not appear in the bibliography, so as it stands the related-work fix is invisible.
2. **P3.2** SFT hyperparameters appendix — not written.
3. **P3.3** rephrase/perturbation validation appendix — not written.
4. **Manuscript `.tex` edits** — deliberately not started; Rylan asked to hold.
5. **OpenReview character limits** — not confirmed; the general response may need splitting.
6. A **disjoint-mathematics contaminant arm**, which is what would separate domain adaptation from
   item-level leakage in the ablation. Not run; the ablation states this as a limit.

---

## Things the author got wrong overnight, as calibration

Both are documented in place, but they show the failure modes to look for:

1. **Claimed 4-shot "floors the uncontaminated baseline to zero"** using numbers that confounded
   prompt format with scoring rule. The conclusion survived rescoring; the evidence originally
   offered for it did not.
2. **Predicted in writing that the perturbed arm's accuracy would sit at the 0.00% floor.** It is
   1.34% at R=32 — above the exact arm. The retrieval-key account explains unreachable *memorized*
   content, not the weak genuine competence a model picks up from 5,000 near-miss problems.

The pattern in both: a mechanism that explained the data was extended past what it was tested on.
Look for the same pattern elsewhere.
