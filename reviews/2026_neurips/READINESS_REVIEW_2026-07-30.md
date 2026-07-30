# Readiness review — NeurIPS 2026 rebuttal, submission 32216

Written 2026-07-30 by a fresh context that read the reviews first and then audited the evidence
adversarially. Discussion closes **2026-08-03**.

**Verdict: the evidence base is sound and the rebuttal is ready to write. It is not yet ready to
post.** One argument in the general response is broken in a way that hands the pivotal reviewer a
counter-argument, one table has two wrong cells, and one false claim in the manuscript is
reviewer-checkable and currently undisclosed. All three are writing fixes, not experiments.

---

## What I re-derived independently, and what reproduced

I recomputed these from raw artifacts rather than re-reading the code that produced them.
**Everything in this list reproduced exactly.**

| Claim | Method | Result |
|---|---|---|
| Retrieval-key result (all 11 contaminant/poscontrol runs) | Rescored all 55,000 saved generations with my own scorer + process pool | Every accuracy, boxed rate and verbatim rate matches to 4 dp |
| The bare `except Exception: pass` in the eval scorer | Counted gold-parse failures | **0 exceptions in 55,000 items** — inert here |
| Verbatim rate is not a whitespace artifact | Re-ran with whitespace-normalised matching | 5.34% → 5.36%; the 0.000% cases stay 0.000% |
| Eval is genuinely 0-shot | Printed the actual prompt | `'Problem:\n...\n\nSolution:'` — no prefix |
| The verbatim contrast is apples-to-apples | Compared the two datasets directly | Solutions **99.82%** byte-identical (4,991/5,000); problems **0%** identical |
| Irreducible error, 33/35 (94.3%) | Refiltered the raw loss cache, counted against 3.5359 | **35 contaminated runs, 33 below.** Exceptions are 34M at R=1 and R=3 |
| Finding #4 retention (AC's pivotal claim) | Read the per-condition grid | 93M ot 1→16: R=100 → 0.0188, R=1000 → 0.9966, ~53× |
| SFT collapse | Notebook 19 report | 72.95% → 2.80%, n=14, median retained 0.022 |
| Temperature at τ=1.0 | Notebook 11 rescored report | 0.2528, ratio of means, 13 conditions → 25% |
| Loss-recovery fractions (1wx9) | Recomputed from the ablation table | 97.9% / 90.2% / 78.4% ✓; "57–88% as contaminated" ✓ |
| Protocol contrast | Rescored CSV | 153M R=316: 0.9984 (0-shot) vs 0.0078 (4-shot) ✓ |
| pass@k capability floor | Summary + shard files | 5,000,000 samples, 0 correct, 0 well-formed `\boxed{}` ✓ |

The three overnight verification agents' conclusions also hold up where I spot-checked them.
The retrieval-key result — flagged in the handoff as the author's favourite and therefore least
well audited — **survives**. I could not break it.

---

## Problems found

### A. ⚠️ The "4-shot teaches the format" argument is not supported by the data. Fix before posting.

> **Status 2026-07-30: resolved, and superseded by something better.** Rather than patch the
> argument, 4-shot was dropped from the response entirely — see `REBUTTAL_DRAFT.md`'s checklist.
> The format concern is now carried by the lenient, format-free scorer, which has since been
> *validated* rather than assumed (229/229 recall on verbatim regurgitation; 100% on numeric
> answers across seven surface forms; superset of strict scoring; symbolic blind spot closed by a
> substring check at ≤0.78%). All 178 credited uncontaminated responses were inspected and are
> spurious. The 344M gap is filled from recovered sweeps. See `data/LENIENT_SCORER_AUDIT.md` and
> `scripts/audit_lenient_scorer.py`.

`REBUTTAL_DRAFT.md`, general response, item 2:

> The four-shot prefix **does** successfully teach the output format — the rate of well-formed
> `\boxed{}` responses rises from near zero to 0.43–0.89 — and uncontaminated accuracy nonetheless
> remains **exactly 0.0000** at every model size. **Removing the format barrier reveals no
> capability underneath it.**

Measured from `protocol_sensitivity_rescored.csv`, the 4-shot `\boxed{}` rate **at R = 0** is:

| Model | 34M | 62M | 93M | 153M | 344M |
|---|---|---|---|---|---|
| 4-shot boxed rate, R=0 | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

It is exactly zero at every size — identical to 0-shot. The 0.43–0.89 figures come from the
**R ≥ 100 contaminated** models, which learned the format from the injected solutions. The
format barrier was never removed for the uncontaminated models, so the syllogism does not close.

This is worse than a loose sentence. As written it invites 8RFz (confidence 4) to reply: *"your
uncontaminated models cannot emit `\boxed{}` at all, so your 0.00% floor is exactly the format
artifact the 4-shot switch was meant to fix"* — resurrecting the rationale the section is trying
to retire. The table's layout encourages the error: two `R = 0` columns sit beside an `R ≥ 100`
column in the same row.

**The correct argument is available and already evidenced.** Under *lenient*, format-free scoring
the uncontaminated models score **0.38–1.26%**, at or below the lenient scorer's own measured
**~1.38% false-positive rate** — established in `reviews/2026_icml/REVIEWER_Mmea/SCORING_INVESTIGATION.md`
over ~1,038,000 *scored* samples, of which ~14,300 were credited and **none contained a
`\boxed{}`**, with gold answers 0–5 accounting for 77.6%. (An earlier version of this line said
"manual inspection of ~1,038,000 samples" — wrong; that is the number scored, not inspected.)
**Closed 2026-07-30:** all 178 leniently-credited responses from the uncontaminated greedy runs
have now been inspected exhaustively — 0 contain a `\boxed{}`, 75.8% have single-digit gold
answers, and all 178 are false positives. See `data/LENIENT_SCORER_AUDIT.md`. That is a capability
bound that does not depend on the model emitting `\boxed{}` at all.

Recommended replacement claim: *4-shot fails to teach the format to uncontaminated models (boxed
rate stays exactly 0.0000), and under format-free lenient scoring their accuracy is
indistinguishable from the scorer's own noise floor. So the 4-shot rationale buys nothing:
it neither removes the format barrier for the baseline nor uncovers capability beneath it.*
The conclusion survives; only the evidence offered for it has to change.

### B. A silent data-pipeline bug: every model got 14.3 tokens/parameter, not 20, and the shortfall shrinks as contamination rises

> **Status 2026-07-30: closed, and downgraded to minor.** Code fixed (`src/data.py`: assertion,
> honest logging, `PRETRAIN_LEGACY_TOKEN_BUDGET=1`); write-up in `docs/TOKEN_BUDGET_SHORTFALL.md`.
>
> **This section is longer than the finding deserves.** In substance it is a one-line methods
> correction — 14.3 tokens/parameter rather than 20, uniformly, with no reported number affected.
> The length reflects two rounds of challenge to the evidence during review, not importance. It is
> **not** a rebuttal item and should not be raised unless a reviewer asks about compute matching.
> Kept only as the audit trail.

**This is a code bug, not an arithmetic slip and not a design decision.** The intended algorithm —
compute the target budget, compute the tokens needed for R replicas, subtract, take the difference
from the corpus — is exactly what `create_dataset_for_pretraining` is written to do. The
subtraction at `src/data.py:309` is correct. What fails is the step that fills the difference.

**The mechanism, in three lines.**

```python
avg_tokens_per_doc = 220e9 / 190168005            # data.py:333  -> 1157
estimated_docs_needed = int(1.05 * corpus_tokens_needed_per_epoch / avg_tokens_per_doc)   # :338
idx_to_keep = np.searchsorted(cumulative_lengths, corpus_tokens_needed_per_epoch)         # :362
```

`1157` is the corpus's advertised average tokens per document (220B tokens / 190,168,005 rows) —
measured by its publishers, with their tokenizer, untruncated. Under **our** tokenizer, truncated
at `max_length = 2048`, the realised mean is **786.3**. The estimate is therefore **47% too high**,
and the `1.05` safety factor — whose comment reads *"Round up a bit to ensure we have more than we
want"* — cannot cover a 47% shortfall.

So the sampled pool holds only `1.05 × 786.3 / 1157 = 0.714` of the tokens requested. Line 362 then
asks `searchsorted` for the cut point at the target, the target exceeds every cumulative sum, and
`searchsorted` returns `len(array)`. `select(range(idx_to_keep))` keeps **every** document. The trim
that was supposed to hit the budget exactly becomes a **silent no-op** — no exception, no warning,
and the printed log line reports the tokens *requested*, not the tokens *delivered*.

**The arithmetic is not in doubt.** Model the run as
`total(R) = f · 20N + (1 − f) · R · 1,446,312` with `f` fixed at the R=0 value:

| R | 0 | 1 | 3 | 10 | 32 | 100 | 316 |
|---|---|---|---|---|---|---|---|
| observed | 486,122,537 | 486,575,381 | 487,410,348 | 490,376,402 | 499,322,819 | 527,446,026 | 617,169,216 |
| predicted | — | 486,536,778 | 487,365,261 | 490,264,950 | 499,378,259 | 527,546,669 | 617,022,794 |
| error | — | −0.008% | −0.009% | −0.023% | +0.011% | +0.019% | −0.024% |

Seven points, every one inside 0.024%. The implied mean document length is **786.3** against the
hard-coded 1157 — derived from the run logs alone, and matching the figure obtained independently
by tokenizing.

**The single cleanest proof needs no model at all: at R = 0, with no contaminant whatsoever, the
34M run saw 486.1M tokens against a 681.2M target — 71.4%.** Contamination cannot explain a
shortfall in a run that has none.

#### Provenance of those two numbers, since both deserve scrutiny

**486,122,537 is logged, not derived.** `train/num_input_tokens_seen` and
`eval_after/num_input_tokens_seen` are *identical* for this run, so it is a terminal value rather
than a mid-training snapshot; `train/epoch = 1.0` with `max_steps = -1` means the run made exactly
one pass, so this number *is* the size of the training dataset.

**681,237,120 was originally a recomputation** (`20 × overtrain_multiplier × model/num_parameters`,
per `pretrain_language_model.py:320`) and is **not** logged in the config cache. It can, however, be
recovered without assuming the 20N rule at all:

| Route | Value |
|---|---|
| Sequences the trainer processed: `1636 steps × 9 grad-accum × 42 batch` | **618,408** |
| Documents the code sampled: `int(1.05 × 681,237,120 / 1156.87)` | **618,304** |
| Difference | **104** — smaller than one batch (378) |
| Target recovered backwards: `618,304 × 1156.87 / 1.05` | **681,236,623** vs 20N = 681,237,120 |

Agreement to **497 tokens in 681 million (7 × 10⁻⁵ %)**. `world_size = 1` is not assumed either —
it is forced by the step count: 618,304 docs / 378 per step = 1635.7 → the 1636 steps observed,
where `world_size = 2` would predict 818.

That the documents *kept* equal the documents *sampled* is the direct evidence the trim was a
no-op. Had it worked, fewer documents would have survived and the token count would have been
681,237,120 exactly.

**Padding is ruled out — but not for the reason first given here.** HF's `num_input_tokens_seen` is
`inputs[main_input_name].numel()`, which *does* include padding, and the Trainer pads dynamically to
the longest sequence in each batch (not to `max_length`). An earlier draft of this section compared
against a fixed 2048 and was wrong to. Two things settle it properly:

1. **The collator never pads.** `pretrain_language_model.py:247` uses `DataCollatorWithFlattening`,
   whose docstring reads: *"concatenates the entire mini batch into single long sequence of shape
   [1, total_tokens] ... no padding will be added"*. `numel()` is therefore the exact sum of real
   token lengths. `group_by_length` is `False`, and is irrelevant with this collator.
2. **The step counts prove it independently of the token counter.** Predicting documents as
   `R × 5000 + int(1.05 × (681,237,120 − R × 1,446,312) / 1156.87)` — i.e. assuming every sampled
   document is kept — and dividing by `9 × 42 = 378` per step reproduces the logged
   `train/global_step` **exactly at all seven doses**:

   | R | 0 | 1 | 3 | 10 | 32 | 100 | 316 |
   |---|---|---|---|---|---|---|---|
   | steps predicted | 1,636 | 1,646 | 1,665 | 1,734 | 1,948 | 2,612 | 4,719 |
   | steps observed | 1,636 | 1,646 | 1,665 | 1,734 | 1,948 | 2,612 | 4,719 |
   | tokens/sequence | 786.2 | 782.3 | 774.4 | 748.5 | 678.2 | 534.4 | 346.0 |

   Had the trim worked, fewer corpus documents would survive and every predicted step count would be
   lower. Separately, tokens/sequence falling to 346 is only possible under content counting: a
   padded batch maximum cannot fall below the longest document in the batch, which with ~11% corpus
   documents present would sit near 1,500–2,048.

**Document length, measured directly.** Tokenizing 4,000 fineweb-edu-dedup documents with the
Qwen3 tokenizer, truncated at 2,048: **mean 766.5**, 95% CI [748.5, 784.5], median 582, 9.1%
truncated — against the hard-coded **1156.9**. (This sample is the first 4,000 documents in stream
order rather than a uniform random draw, so it is not exactly the run's sampling distribution; the
value implied by the run logs is 786.2, 2.5% higher and just outside this CI. The discrepancy does
not matter: whether the true mean is 766 or 786, it is ~1.5× below the hard-coded 1157, and the
1.05 safety factor covers 5%.)

**Two separate consequences, and only one of them is the one I flagged before.**

1. **The budget is uniformly short.** Delivered / (20·m·N) is **0.7136–0.7141** across all five
   model sizes and all five overtrain multipliers — a spread of ±0.0005. Every model was trained
   on **14.27–14.28 tokens per parameter, not 20.**
2. **Dose is confounded with compute.** Because the corpus is short while the contaminant is
   delivered in full, total tokens *rise* with R: +26.96% from R=0 to R=316.

**What survives, and why.** Consequence (1) is a constant factor, so every comparison across model
size and across overtrain multiplier is untouched — the multipliers `m` remain exact in relative
terms because all of them are scaled by the same 0.714. The scaling-law fits are unaffected because
`src/analyze.py:601-607` uses *measured* `num_input_tokens_seen`, not the nominal budget. And
consequence (2) has a direct empirical control that we already ran: the perturbed ablation arm
carries the same token inflation with dose and shows **no dose-response in either loss or
accuracy**, which bounds what 27% more tokens can buy at close to nothing.

**What has to change in the manuscript — three claims, not one.**

| Location | Claim | Reality |
|---|---|---|
| `02_methodology.tex:8` | "Each model was pretrained on 20 tokens-per-parameter \citep{hoffman2022chinchilla}" | **14.3** |
| `01_introduction.tex:28` (Fig. 1 caption) | "We pretrained compute-optimal language models" | 0.71× compute-optimal |
| `04_further_training.tex:12,17` | `D(m,N) ≐ m × 20 × N`; "We term m=1 compute-optimal training" | `m × 14.3 × N`; m=1 is not compute-optimal |
| `99_appendix.tex:74` | "total training tokens remain constant across contamination levels, isolating the effect of contamination from the effect of additional data" | Not constant: +27% |

Nothing in the current rebuttal mentions any of this. All four are computable by a reviewer from
the logged `num_input_tokens_seen`, which is in every run's W&B history.

**Recommended handling — and a correction to my own first ranking of this.**

Fix the code: use the realised mean, or loop until the pool covers the target, and assert
`cumulative_lengths[-1] >= corpus_tokens_needed_per_epoch` so this can never again fail silently.

Fix the manuscript: replace "20 tokens-per-parameter" with the measured 14.3 and retire the
"compute-optimal" label, or keep `m` as an explicitly *relative* multiplier and define it that way.
The appendix sentence is the one that must go regardless, because unlike the others it is a claim
about **experimental validity** — "isolating the effect of contamination from the effect of
additional data" — that is false and checkable.

**But I initially over-ranked this as a rebuttal item, and it should sit below (A).** It changes no
reported number, answers no question any reviewer asked, and every scientific conclusion survives it
(uniform factor; scaling fits use measured tokens; the perturbed arm empirically bounds the
dose–compute confound). Disclosing it would be a *fourth* self-reported error in a rebuttal already
confessing three, and it invites fresh scrutiny of the overtraining framing in exchange for
answering nothing that was asked.

Recommendation: **fix it in the revision, do not foreground it in the rebuttal**, and hold a
prepared paragraph in case a reviewer raises compute-matching. That is a different call from the one
I made in the first draft of this document, and the difference is that a correction only earns
credibility when it answers something someone actually asked.

### C. The bootstrap CIs are computed on different scores than everything they are quoted beside

`BOOTSTRAP_CIS.md` computes intervals from the **leniently** scored logs; every point estimate in
the rebuttal is **strict**. Two consequences:

1. The CI table's own point estimates (R=0 reading 0.38–1.26%) **directly contradict** the
   rebuttal's central correction that the floor is *exactly 0.00%*. If that table enters the
   revision unchanged, the paper contradicts itself in two places.
2. Under strict scoring most near-zero cells become exactly 0, giving degenerate `[0, 0]`
   intervals, and the quoted **"median half-width 0.33 pp"** no longer describes the reported
   numbers.

The file argues half-widths are "insensitive to a shift of that size". That is not right in
direction — zero-count cells collapse to zero width — though it errs conservatively, so the
*conclusion* (effects dwarf sampling error) is safe or strengthened. The rebuttal nonetheless
promises intervals "for every Math Verify number", which is currently not what exists.

Fix: recompute from per-problem strict scores (needs a rescoring pass that emits per-problem
output), and use a rule-of-three upper bound (3/5001 ≈ 0.06%) for zero-count conditions rather
than printing `[0, 0]`.

### D. Two wrong cells in the general response's boxed-rate table

| Model | Draft says | Measured (R ≥ 100) |
|---|---|---|
| 34M | 0.33–0.65 | **0.40–0.52** |
| 62M | 0.65 | 0.65–0.66 ✓ |
| 93M | 0.43–0.70 | 0.43–0.70 ✓ |
| 153M | 0.60–0.89 | 0.60–0.89 ✓ |
| 344M | 0.57–0.66 | **0.59–0.66** |

The prose range "0.43–0.89" should be **0.40–0.89**. Small, but this table sits inside the
paragraph where the paper is confessing its own numerical errors; a wrong number there costs more
than it would anywhere else.

### E. The perturbed positive control was never run at R = 316

`RETRIEVAL_KEY_RESULT.md` says "we evaluated **each** contaminant arm on the very items it was
contaminated with". True for rephrased (R = 32/100/316); perturbed exists only at **R = 32 and
R = 100**. R = 316 is the one dose where the mechanism is visible. No quoted number is wrong — the
rebuttal only reports rephrased positive controls — but the claim of coverage overstates what was
run. Either run it (one checkpoint, one eval) or narrow the sentence.

### F. "5,000 test problems" is 5,001 rows everywhere

Every run has `n_problems = 5001` (4,996 distinct, 5 duplicated — a W&B pagination artifact,
identical across runs, so it cancels in every ratio). The rebuttal says 5,000 throughout, and
pass@k reports 5,000,000 = 5,000 × 1,000. Harmless, but it is the kind of thing a careful reviewer
finds. One footnote settles it.

### G. `RylanSchaeffer/math_rephrased` no longer resolves on the HF Hub

Loading falls back to a local cache with a "couldn't be found on the Hugging Face Hub" warning.
The contaminant datasets underpinning the single largest new experiment are therefore not
fetchable by a reviewer. Possibly downstream of the `HF_TOKEN` incident. Worth resolving before
any reproducibility claim is made.

---

## Completeness gaps still open (all previously known, all confirmed still open)

1. **The five new references are in the bib and `\citep`'d zero times.** Verified by grep:
   `palavalli2024taxonomy`, `mehrbakhsh2024confounders`, `dekoninck2024evading`,
   `dekoninck2024constat`, `godey2025gaperon` — all `cited_in_tex=0`. The related-work fix is a
   direct driver of 8RFz's Originality = 2 and is currently **invisible in the rendered paper**.
   (8RFz's claim that Jiang et al. 2024 is uncited is wrong — it is cited 3×, including an
   appendix paragraph. Confirmed.)
2. **SFT hyperparameters appendix (aPBL W4)** — not written. The appendix gives pretraining
   optimizer/batch/LR only.
3. **Rephrase/perturbation validation appendix (aPBL Q2)** — not written.
4. **Fig. 1 still `\includegraphics` the leniently-scored PDF**
   (`01_introduction.tex:24`). The strict replacement exists on disk, unwired.
5. **OpenReview character limits unconfirmed.** Measured section lengths: 8RFz **11,770**,
   1wx9 **9,431**, general **8,678**, aPBL **5,632**. NeurIPS official comments are commonly
   capped near 6,000 — three of four would need splitting. Confirm the limit before drafting the
   final text, since splitting changes how the argument is sequenced.
6. **Disjoint-mathematics contaminant arm** not run; correctly stated as a scope limit.

---

## Assessment against what the reviewers actually asked

| Reviewer | Ask | Status |
|---|---|---|
| 8RFz W1/Q1 (**pivotal**) | Do Findings 4–5 hold in Math Verify space? | **Answered with data**, 137 + 39 checkpoints, and the answer sharpened the finding. Strongest part of the response. |
| 8RFz W2/Q2 | Temperature: contamination-specific or general degradation? | **Answered** with the matched-τ control. Verified. |
| 8RFz W3/Q3 | Related work; conflict with prior rephrasing results | **Argued well** (capability-boundary reconciliation) but **not yet executed** — citations uncited. The reconciliation leans on the capability claim damaged by (A). |
| 8RFz Q4 | How is Table 1 computed? | **Answered**, including conceding it does not reproduce. Re-measured. |
| 1wx9 W1/Q1 | Paraphrased/partial/realistic leakage | **Ran it.** The three-arm ablation plus the retrieval-key result is the best new material in the rebuttal. |
| 1wx9 W2 | Table 1's direction is ambiguous | **Answered** cleanly. |
| aPBL W3 | Error bars | **Partially** — see (C); the intervals do not match the scoring of the numbers they accompany. |
| aPBL W4, Q2 | SFT hyperparameters; perturbation validation | **Promised, not written.** |
| aPBL Q3 | Is the irreducible-error claim an extrapolation artifact? | **Answered**, and I re-derived 33/35 independently. The logical reframing (measured vs extrapolated) is the right move. |
| All + AC | Scale / single benchmark | Conceded. Correct call. |

---

## What I would do, in order, with four days

1. **Fix (A).** ~30 minutes. Highest value in this list: it removes a counter-argument aimed at
   the pivotal reviewer.
2. **Fix (D)**, and re-check every remaining number in the draft against its artifact — (D)
   suggests the table was typed rather than generated.
3. **Handle (B) in the revision, not the rebuttal.** Fix the pipeline bug and its missing
   assertion; correct the four manuscript claims (14.3 tokens/parameter, not 20; retire
   "compute-optimal"; drop the appendix's "isolating..." sentence). Keep a prepared paragraph in
   case compute-matching is raised, but do not volunteer it — it answers nothing that was asked.
4. **`\citep` the five references.** Without this the single clearest fix to Originality = 2 does
   not exist in the PDF.
5. **Confirm OpenReview limits, then split.**
6. **Write the two appendices** (aPBL W4, Q2) — both are promised in the draft.
7. **Recompute the CIs on strict scores (C)**, or narrow the claim to what was actually computed.
8. Optional: perturbed positive control at R=316 (E); swap Fig. 1 to the strict PDF (4).

Items 1–5 are what stand between "ready to write" and "ready to post". Items 6–7 are promises made
in the draft that the revision has to honour.

---

## One structural observation

The failure mode the previous session named — *a mechanism extended past what it was tested on* —
recurred in (A), and it is the same shape both times: a measurement taken on **contaminated**
models (boxed rate 0.43–0.89) was used to support a claim about **uncontaminated** ones. That
this is now the third instance suggests the check worth institutionalising is narrow and
mechanical: for every claim of the form "X shows Y", confirm the rows that produced X are the rows
Y is about. Two of the three instances would have been caught by that one question.
