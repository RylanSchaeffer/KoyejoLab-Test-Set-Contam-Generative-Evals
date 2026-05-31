# Improvement Roadmap: Workshop → Top-Tier Workshop / Conference

Source: synthesized from the three ICML 2026 FoGen Workshop reviews + meta-review in
`reviews/2026_fdgm_icml_2026/REVIEWS.md` (Submission #9, accepted as Poster).

Prioritized by leverage. **Tier A changes the paper's class** (workshop → conference).
**Tier C alone gets to a top-tier *workshop* paper** quickly and cheaply.

All three reviewers + the AC converged on two blockers: **single benchmark** and **small scale**.

---

## Tier A — Conference blockers (change the paper's class)

- [ ] **A1. Break the single-benchmark ceiling — add ≥2 benchmarks spanning a *structure gradient*.**
  Raised by every reviewer + AC. Sharp version (N8Jt): MATH's clean boxed-answer format may be *why*
  the three-regime story is legible; test whether it holds on less-structured tasks.
  - GSM8K (infra exists in `src/data.py`) — similar structure, robustness check.
  - A code benchmark (HumanEval / MBPP) — different verification, non-MATH.
  - One open-ended / long-form task — real stress test for survival regimes.
  - Payoff is asymmetric: holds → bigger claim; breaks → characterizing *where/why* is itself a finding.

- [ ] **A2. Scale up to 1B+** (currently ≤344M, ~2 orders below frontier; stack supports up to 1.44B in `models.py`).
  Most damaging specific concern (N8Jt): overtraining **crossover point shifts with model size**, threatening
  extrapolation. Add 1B / 1.44B points to show whether crossover + irreducible-error-breach trends are scale-stable.

- [ ] **A3. Statistical rigor — multiple seeds + CIs** (LNEm 2.4; results appear single-run).
  Error bars on scaling fits, bootstrap CIs on irreducible-error estimate, seed variance on lifecycle crossovers.
  Cheap, near-mandatory for a conference, and de-risks A2 + A4.

- [ ] **A4. Validate the two "speculative/heuristic" mechanisms:**
  - [ ] **αeff(τ) ≈ α/τ** (LNEm 1.3): measure token-level logit gaps + per-token error rates directly; show they
    match the temperature-as-effective-exponent prediction. Converts heuristic → measured mechanism.
  - [ ] **SFT "generalize but forget"** (N8Jt, LNEm 2.5): measure forgetting vs generalization *separately*;
    ablate SFT on fresh math vs format-mismatched math. (In-progress notebook 16
    `sft_generalization_teacher_forcing_perturbed` is aimed at this — finish and fold in.)

---

## Tier B — High-value strengtheners (marginal accept → clear accept)

- [ ] **B1. Provenance check of base corpus (R=0 confound)** (LNEm 2.2): n-gram / substring search for MATH
  test+train items in FineWeb-Edu-Dedup. Cheap; closes a real hole in the baseline. Report explicitly.

- [ ] **B2. Ecological validity of contamination** (LNEm 1.4 / 2.1, pBwC):
  - [ ] Report the actual contaminated-token *fraction* per replica count; compare to real-world leakage rates.
  - [ ] Add ≥1 *realistic* contamination mode (partial-solution leakage or paraphrased contamination), not just
    verbatim full-test replicas.

- [ ] **B3. Turn "temperature as truth serum" into a detector** (N8Jt): plot contaminated vs uncontaminated across
  temperature; the *differential* is the usable contamination-audit signal.

- [ ] **B4. Resolve the Huang et al. 2024 / Hayes et al. 2025 discrepancy** (N8Jt) instead of deferring: test the
  distributional-distinctness hypothesis (quantify FineWeb↔MATH distance, or re-run with a closer corpus).

- [ ] **B5. Broaden inference beyond temperature** (LNEm 2.6): top-p, top-k, beam — show survival framework isn't a
  single-axis artifact.

---

## Tier C — Cheap polish (do all regardless; ≈ top-tier *workshop*)

- [ ] **C1. Temper the headline extrapolation claim** (LNEm 1.2): state "single replica beats irreducible error" as a
  *within-range empirical observation* with explicit extrapolation caveats, not an infinite-compute interpretation.
  (Likely already have language from `reviews/2026_icml/` tempering work.)

- [ ] **C2. Bugfix traceability** (LNEm 3.2): put precise task name, harness version, and commit ref in the main text
  (`minerva_math` pre-v3.0 → make it citable).

- [ ] **C3. In-text caveats for deferred derivations** (LNEm 3.1): one-line statements of where the core scaling
  equation's assumptions fail, instead of appendix-only.

- [ ] **C4. Document rephrased/perturbed QC** (LNEm 2.3): semantic-equivalence procedure, solver calibration, and an
  explanation for the <0.1% verify rates (reviewers read that number as a possible artifact).

---

## Recommended sequencing

- **Top-tier workshop camera-ready:** all of Tier C + A3 (seeds) + B1 (provenance). Low cost, removes cheap objections.
- **Conference submission:** Tier A is the price of entry. Sequence **A3 → A1 → A2 → A4** (seeds/provenance first so the
  bigger experiments are credible; A1/A2 generalize the central thesis).
- **Single highest-leverage item:** **A1** (multi-benchmark, structure gradient) — the only weakness shared by all four
  reviews; the difference between "interesting MATH case study" and "general theory of generative contamination."

**Framing note:** reviewers already agree the *thesis* is strong ("generative contamination is brittle verbatim
memorization, characterizable as a survival process"). No new idea needed — demonstrate that thesis *generalizes*
across task + scale and rests on *measured* mechanisms rather than fits. Tier A is exactly that demonstration.
