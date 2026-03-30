# Manuscript Changes Tracker (ICML 2026)

**Paper**: "Quantifying the Effect of Test Set Contamination on Generative Evaluations"
**Submission ID**: 2433
**Status**: ICML does not permit resubmission during review. Track changes here for camera-ready or resubmission.

---

## Changes Driven by Evaluation Fix (4-shot + boxed scoring)

### Figures to Regenerate
All figures using generative eval data will change because:
1. Prompts now include 4-shot prefix (uncontaminated models can now see `\boxed{}` format)
2. Scoring now requires `\boxed{}` in response (eliminates ~1.4% false positive rate)

- [ ] **Figures 1, 2** (Math Verify heatmaps) — `notebooks/11_*` — new eval sweep IDs needed
- [ ] **Figure 5** (NLL curves, survival) — `notebooks/14_*` — **unchanged** (teacher-forcing stays 0-shot)
- [ ] **Table 1** (rephrase/perturb) — `notebooks/15_*` — new generative eval on rephrased+perturbed datasets
- [ ] **Scaling law figure** — `notebooks/20_*` — uses pretraining data only, **unchanged**
- [ ] **Figure 6** (phase diagram) — `notebooks/50_*` — theoretical, **unchanged** unless parameters shift
- [ ] **SFT figures** — `notebooks/13_*` — new SFT eval sweep ID needed
- [ ] **Dose response curves** — `notebooks/30_*` — uses pretraining data only, **unchanged**

### Text Updates for Eval Methodology
- [ ] **Section 2 (Methodology)**: Document 4-shot prompting — "We use 4-shot in-context examples following the EleutherAI evaluation harness standard for minerva_math"
- [ ] **Section 2 (Methodology)**: Document boxed-required scoring — "Responses must contain `\boxed{}` to receive credit; we extract the boxed content via brace-depth matching before calling math-verify"
- [ ] **Appendix**: Add the 4 few-shot examples used (from `src/data.py:MINERVA_MATH_FEWSHOT_EXAMPLES`)

### Numerical Results to Update
- [ ] All Math Verify accuracy numbers in text (will change due to 4-shot + boxed scoring)
- [ ] Scaling law parameters (E, α) — may shift, especially E for uncontaminated models
- [ ] Phase diagram boundaries — may shift if scaling parameters change

---

## Changes Addressing Reviewer Concerns

### Reviewer 4xWn (Reject, score 2) — Soundness & Framing
- [ ] Revise "more rigorous measurement" → "which enable the most direct causal measurement"
- [ ] Revise "mathematically discover" → "mathematically characterize"
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
- [ ] Add cross-entropy results from perturbed MATH to Table 1 (new experiment)
- [ ] If SFT improves cross-entropy on perturbed MATH → direct evidence of generalization (not just forgetting)

### Reviewer THKB (Weak Accept, score 4) — Finding #8 & Generalizability
- [ ] Expand Finding #8 exposition — make survival process derivation more explicit:
  - **Decoherence (E > 0)**: Errors accumulate → memorization lost
  - **Lock-in (E ≈ 0, α > 1)**: Survival probability converges to positive constant
  - **Brittle (E ≈ 0, α ≤ 1)**: Stretched exponential decay
- [ ] Add cautionary note: regimes are idealized asymptotic dynamics, should be validated before use as detection proxies
- [ ] Strengthen discussion of generalizability to larger models via scaling law bridge
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
| P1: Generative eval (4-shot) | Running | All Math Verify results | ~23 |
| P2: Teacher-forced eval | NOT re-running (stays 0-shot) | NLL curves unchanged | 0 |
| P3: Dose-response eval | Configs ready | Extended temperature sweep | ~65 |
| P4: SFT eval | Configs ready | SFT figures | ~20 |
| P5: Rephrase/perturbed eval | Configs created | Table 1 (robustness) | ~3 |
| P6: Pass@k | Scripts ready | Reviewer Mmea rebuttal | ~35 |
| Perturbed MATH teacher-forcing | Config exists | Reviewer 6RQA (generalization) | ~7 |
