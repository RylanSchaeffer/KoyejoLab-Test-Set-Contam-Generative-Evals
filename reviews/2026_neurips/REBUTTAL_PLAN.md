# NeurIPS 2026 Rebuttal Plan — Submission 32216

Scores: **8RFz = 3** (conf 4, Quality 2, Originality 2) · **1wx9 = 4** (conf 4) · **aPBL = 3** (conf 3).
AC published a 5-bullet weakness checklist and explicitly named 8RFz's loss-vs-correctness objection
as *the* critique that questions whether the evidence supports the claims.

Constraint: pretraining new models is largely off the table (cluster contention). Evaluations are cheap.
This plan is built around that constraint.

---

## Bottom line up front

**The single most important finding of this audit: you already have the data for half of the AC's
pivotal critique, and it is not in the paper.**

- 351 finished Math Verify eval runs on SFT checkpoints exist (W&B sweep `2zpwcnek`), plotted at
  `notebooks/13_math_qwen3_sft_math_verify/results/y=math-verify_x=replicas_hue=compute_col=temp.png`.
  The NeurIPS manuscript cites **only** `figures/12_math_qwen3_sft_cross_entropy/` — the accuracy-space
  result was never folded in.
- The result is *stronger* than the loss story: post-SFT Math Verify is **flat at ~1–2% across every
  contamination level**, versus pretrained models reaching **~100% at R ≥ 316** (τ=0, same 4-shot
  boxed-required protocol, `notebooks/11_*`). SFT collapses the contamination advantage ~60× in accuracy.  <!-- ⚠️ SUPERSEDED 2026-07-30: the ~60× is an artifact of comparing 0-shot pretrained against 4-shot SFT. Matched at 0-shot it is 72.95% → 2.80%. See HANDOFF.md. -->
- This also **kills 8RFz's darker corollary** ("accuracy persists while loss rises → stealth contamination
  that evades perplexity detection"). It doesn't persist. That is very good news.

The matching overtraining result does **not** exist — and it is pure inference on checkpoints you already
have. That's the one experiment that must run.

**Realistic target.** All 5s is not a realistic outcome; 8RFz gave Quality 2 / Originality 2 with confidence 4,
and Originality complaints rarely move more than a point. The achievable and sufficient outcome is
**8RFz 3→4, aPBL 3→4, 1wx9 4→4/5**, plus an AC-facing response that maps 1:1 onto their five bullets.
P0+P1 below is what gets you there.

---

## Verified inventory (what exists, what doesn't)

| Asset | Status | Evidence |
|---|---|---|
| Overtrained checkpoints, ot ∈ {2,4,8,16} | **138 exist on HF Hub** | `RylanSchaeffer/mem_Qwen3-{34,62,93,344}M_..._ot_{2,4,8,16}`, R ∈ {0…3162}; 153M only R ∈ {0,100} |
| Math Verify on ot > 1 | **ZERO runs** | All 1,270 finished runs in `memorization-scoring-vs-sampling-eval` are `ot=1` or `ot=1_sft` |
| Math Verify on SFT checkpoints | **351 runs, DONE, not in paper** | sweep `2zpwcnek`, notebook 13 |
| pass@k = 0.000% / 808,797 samples (uncontam. 344M) | **DONE, not in paper** | `reviews/2026_icml/REBUTTALS.md:238`; zero hits for `pass@k` in `manuscript_neurips_2026/*.tex` |
| Perturbed-MATH teacher forcing on SFT ckpts (14/17) | **DONE, prose-only** | sweep `onaspopu`, notebook 16; cited at `04_further_training.tex:64`, no figure |
| Rephrase/perturbation QC + spot checks | **DONE, not in paper** | `reviews/2026_icml/REVIEWER_6RQA/{math_*_spot_check.md, spot_check_batch_{0..3}.md}` |
| Per-problem Math Verify scores | **Logged in W&B history** | notebook 15 groups by `run_id` on `math_verify_score` → bootstrap CIs are free |
| Table 1 "Original" column | **Absent** | `03_pretraining.tex:14–67` has only Reph./Pert. |
| Palavalli / Mehrbakhsh / Dekoninck / Godey citations | **Absent from .bib** | `grep -c` on `references_rylan.bib` = 0 for each |
| "first targeted examination" overclaim | **Present** | `06_discussion.tex:4` |
| Rephrased-as-contaminant pretraining | Needs ~1h code | `create_dataset_for_supervised_finetuning` (`src/data.py:442`) only branches on `minerva_math` / `gsm8k-platinum`; `load_dataset_math_rephrased()` exists but is unwired (and defined twice, `src/data.py:542` and `:556`) |

Eval cost calibration (sampled 401 finished runs): **median 3–6 min wallclock per run**, p90 ~20 min.

---

## P0 — Zero compute, already-collected data (do first, finishes in a day)

Highest value per hour in the entire plan. Nothing here needs a GPU.

**P0.1 — Fold the SFT Math Verify result into the paper.** Answers 8RFz Q1 (SFT half) and AC bullet 4.
New figure from notebook 13 next to Fig. 3, plus a paragraph rewriting Finding 5 in accuracy space.
*Required sanity check first (CPU, ~20 min):* pull the `response` column for the SFT eval runs and confirm
the ~1.5% floor is genuine failure, **not** a broken `\boxed{}` format rate after SFT. If format rate
dropped, the number is an artifact and the whole item changes character. Verify before writing a word of it.

**P0.2 — Put pass@k in the paper.** 0.000% across 808,797 samples on the uncontaminated 344M at τ=1.0.
This is a *zero-capability baseline*: every point of contaminated performance is unambiguously memorization.
It simultaneously (a) hardens Findings 1–2, (b) reframes aPBL's "small models" weakness as the design
feature that makes the causal claim clean, and (c) closes the R=0 corpus-provenance hole for free.
*Action:* locate the scored outputs on the cluster (`scripts/score_pass_at_k.py` writes `results.csv` /
`summary.md`; not present on this workstation) or re-derive from W&B. Do this early — it's the one P0 item
with a "where did the data go" risk.

**P0.3 — Bootstrap CIs on every Math Verify number.** Answers aPBL W3 and AC bullet 3. Per-problem scores
are already in W&B history; resample the 5,000 test problems. Pair with the argument that effect sizes
(1% → 100%, 10–100× loss changes) dwarf plausible seed variance. This is not the same as multiple seeds —
say so explicitly and commit to seeds for camera-ready rather than implying you've done them.

**P0.4 — Promote the notebook 16 result to a figure.** The 14/17 perturbed-MATH generalization finding is
currently a single clause at `04_further_training.tex:64`. It is the direct evidence that SFT induces real
transfer rather than pure forgetting, and it deserves a panel.

**P0.5 — Add the "Original" column to Table 1.** aPBL Q2 / 8RFz Q4. Makes the ~100% → ~0% collapse legible
in one table instead of forcing a cross-reference to Fig. 1.

---

## P1 — The one experiment that must run (inference only)

**P1.1 — Math Verify on all 138 overtrained checkpoints.**

This is the AC's pivotal critique. Finding 4 currently rests entirely on cross-entropy; 8RFz is correct
that loss on exact solution text ≠ correctness. The checkpoints exist; nothing needs retraining.

- Full grid, τ ∈ {0, 0.316, 1.0}: 414 runs ≈ **41 GPU-hours** (~10 h on 4 GPUs, ~5 h on 8).
- Minimum viable, τ = 0 only: 138 runs ≈ **14 GPU-hours** (~3.5 h on 4 GPUs).
- Optimization worth 30 min of coding: `eval_language_model.py` loads vLLM once per run. Loop the three
  temperatures inside one model load → roughly 3× cheaper, since startup dominates for these model sizes.
- New sweep YAMLs under `sweeps/eval_pt/math/` mirroring the `ot_1` files with the `ot_{2,4,8,16}` names.

Both outcomes are publishable, and say so in the response:
- Accuracy tracks loss → Finding 4 stands, objection dies.
- Accuracy decouples (stays high while loss rises) → **that is the stealth-contamination result 8RFz
  hypothesized**, it's more alarming than the current claim, and reframing Finding 4 around it reads as
  scientific maturity. Given P0.1 shows SFT collapses accuracy, decoupling is the less likely outcome — but
  prepare both paragraphs now so the result doesn't cost you a day.

Same format sanity check as P0.1 applies.

---

## P2 — Free re-analysis of existing generations (no GPU)

**P2.1 — Kill the temperature confound.** 8RFz W2/Q2. Two moves, both re-plots of data you have:
1. Normalize: plot `score(τ) / score(τ=0)` per contamination level. Generic incoherence degrades all
   populations similarly; your asymmetry (~2× at R ≤ 10 vs ~40× at R = 1000) is the answer.
2. Emphasize τ = 1.0 is *not* a hot setting — it is the model's own distribution — yet contaminated models
   fall to the uncontaminated floor while uncontaminated models barely move.
Concede that τ ∈ {1.29, 1.5} degrades everything and restrict the "truth serum" claim to τ ≤ 1.
*Optional coherence control:* the eval script already logs edit distance and logprobs; show τ ≤ 1
generations are comparably coherent across contamination levels.

**P2.2 — Harden the irreducible-error claim.** aPBL Q3. Refitting only. Bootstrap/profile-likelihood CIs on
E(0), refit under alternative functional forms, and make the logical structure explicit: **contaminated
losses are measured; only the uncontaminated asymptote is extrapolated**, so the claim needs only that
E(0)'s plausible lower bound exceed measured contaminated losses — not that the functional form be exact.

**P2.3 — Report the contaminated-token fraction per replica count.** aPBL Q1 ("contrived"). Pure arithmetic
from existing configs: R = 1 is a vanishing fraction of an 82–144GB corpus. Compare against published
real-world leakage estimates. Converts "contrived" into "we bracket the realistic range from below."

**P2.4 (upside item) — Turn temperature response into a detector.** Not requested by any reviewer, which is
exactly why it's worth doing: it adds a *new contribution* mid-rebuttal rather than only patching holes.
The τ-response differential separates contaminated from uncontaminated models without corpus access.
Report separability (ROC/AUC over your existing grid) from data already collected. This is the item most
likely to move someone from "solid but limited" to "I want this at the conference."

---

## P3 — Writing only, zero compute (do all of it regardless)

**P3.1 — Fix the originality framing.** This is what 8RFz's Originality = 2 is about, and it costs nothing.
- Delete "the first targeted examination of contamination in generative tasks" (`06_discussion.tex:4`).
  It is indefensible — you cite Kocyigit et al. 2025, which is controlled contamination of a generative task.
- Replace with a defensible claim: *the first controlled study spanning the full lifecycle — pretraining
  dose → overtraining → SFT → inference-time sampling — on a verifiable generative benchmark.*
- Add Palavalli 2024, Mehrbakhsh 2024, both Dekoninck 2024 papers, Godey 2025; engage Jiang 2024 beyond a
  bare citation.
- Add an explicit paragraph: Finding 1 **replicates** repeat-count dose effects (Jiang, Dekoninck,
  Magar & Schwartz, Bordt); Finding 2 **conflicts** with rephrasing-transfer results (Mehrbakhsh,
  Dekoninck, Yang 2023). Hypothesize why: your models are tiny and trained **from scratch**, with no general
  linguistic competence to bridge surface-form changes, whereas prior work injected contamination into
  already-capable pretrained models via finetuning or continued pretraining. **pass@k = 0 (P0.2) is direct
  evidence for this reconciliation** — your models provably have zero latent capability to bridge with.

  This converts your biggest framing liability into a scale/regime claim, and it is the single best shot at
  moving 8RFz's two low sub-scores.

**P3.2 — SFT hyperparameters appendix.** aPBL W4. LR, schedule, epochs, batch size, data formatting, seeds.
An afternoon, and it's a straight concession with no downside.

**P3.3 — Rephrase/perturbation validation appendix.** aPBL Q2. **Already written** — port the QC audits and
4 spot-check batches from `reviews/2026_icml/REVIEWER_6RQA/`. Include the token-length distribution
comparison and the answer-validation procedure, and explain the <0.1% verify rates so reviewers don't read
them as an artifact.

**P3.4 — Clarify what Table 1 actually tested.** 1wx9's W2 infers a symmetry you never tested: Table 1 is
*exact contamination in training, modified evaluation*, not *rephrased contamination in training*. State the
direction tested plainly, then commit to the symmetric direction (P4.1). Do not let this read as a dodge.

**P3.5 — Lifecycle summary figure.** One figure carrying pretraining dose → overtraining → SFT → inference
in Math Verify space. Makes the "full lifecycle" novelty claim from P3.1 visually self-evident, and it
becomes possible only once P1.1 lands.

**P3.6 — Concede scope, don't fight it.** All three reviewers raised external validity and the AC noted the
independence of that agreement. Restate the scale-for-control tradeoff, cite pass@k as what that control
buys you, and commit explicitly to ≥1B + a second family + a code benchmark for camera-ready. Arguing here
costs credibility you need elsewhere.

---

## P4 — Stretch, only if GPUs free up (decide end of day 2)

**P4.1 — Paraphrased contamination during pretraining.** The single most-requested experiment
(1wx9 Q1, aPBL Q1, AC bullet 1), and 1wx9 is your advocate at 4.

The arithmetic is more favorable than it feels: a 34M model at 1×OT is 20 × 34M = **680M tokens** —
roughly 3–8 GPU-hours per run at the sweep's 2-GPU config. A 3-point grid (34M × R ∈ {32, 100, 316}) is
**~10–25 GPU-hours total**, plus ~1 h of code (wire `load_dataset_math_rephrased()` into the
`create_dataset_for_supervised_finetuning` dispatch at `src/data.py:442` and add a preprocess fn) and the
corpus build. That is one quiet night on 2 GPUs, not a cluster campaign.

Evaluate on both the original test set and the paraphrase set. Expected outcome at this scale: little
transfer to the original set — which (a) empirically answers 1wx9's W2, (b) sharpens the memorization story,
(c) supports the P3.1 reconciliation with your own new data. Be ready for "so exact replicas don't tell us
about realistic leakage": the answer is that exact replicas **upper-bound** the effect, and the paper
characterizes that upper bound's lifecycle dynamics.

**Only promise this if it will land before discussion closes.** A missed promise is worse than a clean
"committed for camera-ready."

**P4.2 — GSM8K contamination mini-sweep at 34M/62M.** Defuses "MATH-specific" (aPBL W2, AC bullet 2).
GSM8K's test set is ~1.3k problems, a fraction of MATH's token footprint, and `src/data.py` already has
GSM8K-Platinum loaders and templates — but `create_dataset_for_pretraining` has never been run against it,
so budget real debugging time. Lower priority than P4.1: it answers a weaker objection at similar cost.

**P4.3 — Seeds at pivotal configs.** 2–3 seeds at 34M and 93M × R ∈ {0, 1, 10, 100}, concentrated on the
R ≈ 10–100 transition where variance actually matters. Nice-to-have; P0.3 covers the reviewer's stated
concern at zero cost.

---

## Explicitly NOT doing this cycle

State these as camera-ready commitments rather than attempting them:
- ≥1B parameter models, second architecture family (Llama/Gemma-style).
- Code benchmark (HumanEval/MBPP).
- Full leakage-mode grid (translation, partial/problem-only, embedded-in-discussion).
- Multi-seed across the full grid.
- Resolving the Huang et al. / Hayes et al. tension empirically.

---

## Sequencing

**Day 1**
- Launch P1.1 (overtraining Math Verify) — longest wall-clock, start it before anything else.
- Run the format sanity check, then P0.1 (SFT Math Verify figure + Finding 5 rewrite).
- Hunt down the pass@k artifacts (P0.2) — the only item with a data-loss risk.
- Decide P4.1 go/no-go based on observed cluster capacity.

**Day 2**
- P0.3–P0.5, P2.1–P2.3 (all CPU, parallelizable across co-authors).
- P3.1 related-work rewrite — the single highest-value writing task; give it to whoever knows the
  contamination literature best.
- P3.2, P3.3 (P3.3 is mostly porting existing text).

**Day 3**
- Fold P1.1 results into Finding 4 + the lifecycle figure (P3.5).
- P2.4 detector analysis if time remains.
- Assemble the response.

**Response structure.** Global response mirrors the AC's five weakness bullets **one-for-one** — they
published their checklist, so make verification effortless. Then per-reviewer:
- **8RFz** — the reviewer the AC amplified, and the one whose score most needs to move. All four questions
  are answerable with numbers. Lead with P0.1 + P1.1 (Q1), P2.1 (Q2), P3.1 (Q3), P0.5 (Q4).
- **aPBL** — cheapest reviewer to satisfy. Full compliance on P0.3 (seeds/CIs), P3.2 (SFT details),
  P3.3 (validation), P2.2 (scaling-law robustness), P0.2 (capability baseline) is a clean 3→4.
- **1wx9** — already your advocate. Lead with P3.4 (clarify the direction tested) and whatever P4.1 yields.

Concede gracefully wherever the reviewers are right: the metric gap is real, the framing did overclaim, and
the scope is limited. Fighting any of those three costs you the credibility you need for the claims worth
defending.
