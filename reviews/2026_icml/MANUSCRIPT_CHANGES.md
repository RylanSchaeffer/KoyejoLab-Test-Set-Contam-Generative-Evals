# Manuscript Changes Tracker (ICML 2026 review → NeurIPS 2026 resubmission)

**Paper**: "Quantifying the Effect of Test Set Contamination on Generative Evaluations"
**Submission ID (ICML 2026)**: 2433
**Status**: ICML 2026 was rejected; resubmitting to NeurIPS 2026. New manuscript directory: `manuscript_neurips_2026/` (full conversion documented in `manuscript_neurips_2026/CONVERSION_CHANGES.md`).

## ⚠️ STATUS UPDATE (2026-05-06)

This tracker now spans the ICML 2026 review and the NeurIPS 2026 resubmission. Checkbox legend in the body below: `[x]` done, `[~]` in progress (code/configs ready, end-to-end run or prose pending), `[ ]` not started.

**Headline progress since submission:**

- **Scoring fix shipped:** commit `893f29d`, `src/scoring.py` requires `\boxed{}`, 94 tests in `tests/test_boxed_scoring.py`. Lenient `parse()` false-positive fallback is no longer reachable in our pipeline.
- **4-shot prompting shipped:** commit `db75c5f`, `MINERVA_MATH_FEWSHOT_EXAMPLES` + `build_fewshot_prefix()` in `src/data.py`. Teacher-forcing intentionally stayed 0-shot.
- **Cleaned datasets:** `RylanSchaeffer/math_perturbed` (uploaded 2026-03-30) and `RylanSchaeffer/math_rephrased` (uploaded 2026-03-29) supersede the stellaathena/* versions. Teacher-forced eval support: commit `c95759c`.
- **Generative-eval sweep IDs configured** (runs not yet end-to-end): notebook 11 PT — `qx2c4702` (34M), `dkiui6we` (62M), `cx8y41bw` (93M), `4w5x8hez` (153M), `mprek7pj` (344M); notebook 13 SFT — `2zpwcnek`.
- **Reviewer 6RQA Weakness 1 (perturbed teacher-forcing):** EXECUTED end-to-end — sweep `onaspopu`, notebook 16, key finding "SFT reduces NLL on perturbed problems at all contamination levels (Δ up to −2.8 nats)".
- **Manuscript text fixes (language, Finding #8 expansion, regime cautions, generalizability):** DONE in `manuscript_neurips_2026/` per independent audit.
- **Still open:** P6 pass@k end-to-end run; figure regeneration end-to-end (sweep IDs configured); 4-shot + boxed-scoring methodology paragraph in `manuscript_neurips_2026/02_methodology.tex`; few-shot examples appendix entry in `manuscript_neurips_2026/99_appendix.tex`; experimental-details summary table in main body; checklist `\answerTODO{}` at `manuscript_neurips_2026/checklist.tex:22, 42, 102`; competing-interests disclosure (F1); workshop-note de-anon risk (F2).

---

## Changes Driven by Evaluation Fix (4-shot + boxed scoring)

### Figures to Regenerate
All figures using generative eval data will change because:
1. Prompts now include 4-shot prefix (uncontaminated models can now see `\boxed{}` format)
2. Scoring now requires `\boxed{}` in response (eliminates ~1.4% false positive rate)

- [~] **Figures 1, 2** (Math Verify heatmaps) — `notebooks/11_*` — sweep IDs `qx2c4702` (34M), `dkiui6we` (62M), `cx8y41bw` (93M), `4w5x8hez` (153M), `mprek7pj` (344M) configured in notebook 11; figures NOT yet regenerated end-to-end
- [ ] **Figure 5** (NLL curves, survival) — `notebooks/14_*` — **unchanged** (teacher-forcing stays 0-shot)
- [~] **Table 1** (rephrase/perturb) — `notebooks/15_*` — datasets ready (`RylanSchaeffer/math_perturbed`, `RylanSchaeffer/math_rephrased`); eval sweep configured but not yet run end-to-end
- [ ] **Scaling law figure** — `notebooks/20_*` — uses pretraining data only, **unchanged**
- [ ] **Figure 6** (phase diagram) — `notebooks/50_*` — theoretical, **unchanged** unless parameters shift
- [~] **SFT figures** — `notebooks/13_*` — sweep ID `2zpwcnek` configured in notebook 13; figures not yet regenerated end-to-end
- [ ] **Dose response curves** — `notebooks/30_*` — uses pretraining data only, **unchanged**

### Text Updates for Eval Methodology
- [ ] **Section 2 (Methodology)**: Document 4-shot prompting — "We use 4-shot in-context examples following the EleutherAI evaluation harness standard for minerva_math" — pending in `manuscript_neurips_2026/02_methodology.tex` (CODE DONE: commit `db75c5f`, `MINERVA_MATH_FEWSHOT_EXAMPLES` + `build_fewshot_prefix()` in `src/data.py`)
- [ ] **Section 2 (Methodology)**: Document boxed-required scoring — "Responses must contain `\boxed{}` to receive credit; we extract the boxed content via brace-depth matching before calling math-verify" — pending in `manuscript_neurips_2026/02_methodology.tex` (CODE DONE: commit `893f29d`, `src/scoring.py` + 94 tests in `tests/test_boxed_scoring.py`)
- [ ] **Appendix**: Add the 4 few-shot examples used (from `src/data.py:MINERVA_MATH_FEWSHOT_EXAMPLES`) — pending in `manuscript_neurips_2026/99_appendix.tex`

### Numerical Results to Update
- [ ] All Math Verify accuracy numbers in text (will change due to 4-shot + boxed scoring)
- [ ] Scaling law parameters (E, α) — may shift, especially E for uncontaminated models
- [ ] Phase diagram boundaries — may shift if scaling parameters change

---

## Changes Addressing Reviewer Concerns

### Reviewer 4xWn (Reject, score 2) — Soundness & Framing
- [x] Revise "more rigorous measurement" → "which enable the most direct causal measurement" — DONE per `manuscript_neurips_2026/01_introduction.tex`
- [x] Revise "mathematically discover" → "mathematically characterize" — DONE per `manuscript_neurips_2026/01_introduction.tex`
- [ ] Expand experimental details in main body (currently Appendix B) — add summary table
- [ ] Make scale limitation more prominent — acknowledge 344M is small, explain why it enables clean causal measurement
- [ ] Clarify that Kocyigit et al. 2025 IS cited (intro & appendix) — 4xWn's claim is factually incorrect
- [ ] Clarify MATH solution lengths are 15–1,949 tokens (chain-of-thought), not short final answers
- [ ] Address AC message: 4xWn's "multiple errors of fact" (e.g., 34M claim, Kocyigit citation)

### Reviewer Mmea (Weak Reject, score 3) — Model Capacity & SFT
- [ ] Add pass@k results for uncontaminated 344M (new experiment) — if pass@k > 0 on Level 1-2, refutes "fundamentally lacks capacity"
- [ ] Revise Finding #5 (SFT) language: emphasize asymmetry (improves at low contamination, degrades at high) as key finding, not mechanism
- [ ] If pass@k = 0: reframe as "clean-isolation" design choice, not limitation

### Reviewer 6RQA (Weak Accept, score 4) — Memorization vs. Generalization
- [ ] Expand Table 1 to show all model sizes (currently only 344M)
- [x] Add cross-entropy results from perturbed MATH to Table 1 (new experiment) — DONE: sweep `onaspopu` executed, analysis in `notebooks/16_sft_generalization_teacher_forcing_perturbed/`; manuscript integration into `manuscript_neurips_2026/04_further_training.tex` still pending
- [x] If SFT improves cross-entropy on perturbed MATH → direct evidence of generalization (not just forgetting) — CONFIRMED: SFT reduces NLL on perturbed problems at ALL contamination levels (Δ up to −2.8 nats); plots at `notebooks/16_sft_generalization_teacher_forcing_perturbed/results/y=delta_nll_perturbed_x=num_replicas_hue=model_size.{pdf,png}`

### Reviewer THKB (Weak Accept, score 4) — Finding #8 & Generalizability
- [x] Expand Finding #8 exposition — make survival process derivation more explicit (DONE in `manuscript_neurips_2026/05_generation.tex`, verified by independent audit):
  - **Decoherence (E > 0)**: Errors accumulate → memorization lost
  - **Lock-in (E ≈ 0, α > 1)**: Survival probability converges to positive constant
  - **Brittle (E ≈ 0, α ≤ 1)**: Stretched exponential decay
- [x] Add cautionary note: regimes are idealized asymptotic dynamics, should be validated before use as detection proxies — DONE per `manuscript_neurips_2026/06_discussion.tex` and `manuscript_neurips_2026/99_appendix.tex` Additional Limitations
- [x] Strengthen discussion of generalizability to larger models via scaling law bridge — DONE per `manuscript_neurips_2026/06_discussion.tex` and `manuscript_neurips_2026/99_appendix.tex` Additional Limitations
- [ ] Discuss realistic-setting differences: scale (344M vs billions), corpus composition, contamination mechanism

---

## General Presentation Improvements
- [ ] Strengthen transition paragraphs between lifecycle sections (pretraining → SFT → inference)
- [ ] Make limitations section more visually prominent (standalone heading?)
- [ ] Acknowledge single-benchmark limitation explicitly — MATH is tractable but needs validation on code, reasoning, etc.
- [ ] Emphasize generative (not discriminative) focus vs. Bordt et al.
- [ ] Highlight novel findings absent from Bordt et al.: temperature sensitivity, solution-length decay, survival process framework

---

## New Experiments (for rebuttal / camera-ready)

| Experiment | Status | Purpose | GPU-hours |
|-----------|--------|---------|-----------|
| P1: Generative eval (4-shot) | In progress (sweep IDs configured: `qx2c4702`/`dkiui6we`/`cx8y41bw`/`4w5x8hez`/`mprek7pj`; not yet end-to-end) | All Math Verify results | ~23 |
| P2: Teacher-forced eval | NOT re-running (stays 0-shot) | NLL curves unchanged | 0 |
| P3: Dose-response eval | Configs ready | Extended temperature sweep | ~65 |
| P4: SFT eval | In progress (sweep ID `2zpwcnek` configured in notebook 13; not yet end-to-end) | SFT figures | ~20 |
| P5: Rephrase/perturbed eval | Configs created (datasets `RylanSchaeffer/math_{perturbed,rephrased}` ready) | Table 1 (robustness) | ~3 |
| P6: Pass@k | Scripts + boxed scorer DONE (commit `893f29d`); sweep run still open ([ ]) | Reviewer Mmea rebuttal | ~35 |
| Perturbed MATH teacher-forcing | [x] DONE — sweep `onaspopu`, notebook 16, Δ up to −2.8 nats | Reviewer 6RQA (generalization) | ~7 |
