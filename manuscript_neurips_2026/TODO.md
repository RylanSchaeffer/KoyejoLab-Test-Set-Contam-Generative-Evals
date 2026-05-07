# NeurIPS 2026 Manuscript TODO

Source: promises made to reviewers in `reviews/2026_icml/REBUTTALS.md`.
Status verified by reading every `.tex` file in `manuscript_neurips_2026/`.

---

## Done

- **Expand Table 1 to all model sizes.** `03_pretraining.tex:10-71` (34M, 93M, 344M panels).
- **"most direct causal measurement" wording.** `01_introduction.tex:8`.
- **"mathematically characterize" wording** (replaces "derive three regimes"). `01_introduction.tex:60`.
- **Finding #8 expansion** (survival process, three regimes with E/α conditions). `05_generation.tex:39-78`.
- **Cautionary note that three regimes are idealized asymptotic dynamics.** `05_generation.tex:69`.
- **344M scale limitation more prominent.** `06_discussion.tex` (commit `172f3e1`).
- **MATH-only single-benchmark acknowledgment.** `06_discussion.tex` (commit `5c4f280`).

## Declined

- **Cross-entropy column in Table 1.**

## Partial — needs more

- **Experiment B (4-shot greedy on rephrased + perturbed MATH).** Table 1 has the columns (`03_pretraining.tex:10-71`) but methodology label "4-shot greedy" is missing from caption / table notes.
- **Single-replica distributional caveat.** `03_pretraining.tex:159-160` gestures at distributional difference between MATH and FineWeb-Edu-Dedup but does not frame it as a qualifier on the single-replica claim per rebuttal L133.
- **Finding #5 SFT asymmetry without mechanism claim.** Asymmetry present in `01_introduction.tex:58` and `04_further_training.tex:62-64`, but `04_further_training.tex` still includes "We conjecture that during SFT, contaminated models learn to generalize, but also forget..." — mechanism conjecture promised to be softened is still in.
- **Summary table of experimental details in main body.** Pointer in `02_methodology.tex:10` to App. B; App. B (`99_appendix.tex:30-86`) lists optimizer / batch / seq length in prose, but no consolidated summary table exists in main body or appendix. Doc count and explicit step count not given.
- **Realistic-setting differences (scale / corpus / mechanism).** Scale and corpus covered in `06_discussion.tex:23-25` and `99_appendix.tex:200-203`. **Contamination mechanism (exact replica vs near-duplicate) missing everywhere.**
- **Lifecycle transition paragraphs.** Brief one-sentence transitions exist (e.g., `05_generation.tex:15`, `04_further_training.tex:5`), but none are "strengthened" — single sentences only.
- **Limitations section visually prominent.** `06_discussion.tex:22` has `\paragraph{Limitations.}` heading; appendix uses `\paragraph{Additional Limitations}` at L200. Reviewer THKB asked for more prominent — open whether `\paragraph` is enough or whether `\section` is wanted.

## Not done — entirely missing

- **Experiment A (teacher-forced NLL on perturbed MATH, pre-SFT vs post-SFT).** Promised to reviewers 6RQA + Mmea. Not in any file.
- **Pass@k zero-capability baseline (1,000 samples/problem, T=1.0, 4-shot, ~0% across 808k samples).** Promised to reviewer Mmea. Not in any file.

## Notes

- Earlier `MANUSCRIPT_CHANGES.md` claims that the idealized-asymptotic caveat and scaling-law bridge were "done in `06_discussion.tex` + `99_appendix.tex`" are **incorrect** — neither is present. Treat MANUSCRIPT_CHANGES.md as untrustworthy until re-audited.
